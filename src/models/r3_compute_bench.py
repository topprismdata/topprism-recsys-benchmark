import platform
import time
from typing import Any, Dict, List, Optional, Sequence
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.r2_encoders import BaseSequentialEncoder, build_r2_encoder
from src.models.r2_losses import SampledSoftmax256Loss, sample_negatives_excluding_positives


R3_ENCODERS: List[str] = [
    "mean_pool",
    "gru",
    "sasrec",
    "ligr_sasrec",
    "hstu_math",
]

R3_EXECUTION_MODES: List[str] = [
    "eager_fp32",
    "amp_bf16",
    "amp_fp16",
    "sdpa_compiled_bf16",
]

R3_CAPACITY_SPECS: Dict[str, Dict[str, int]] = {
    "S": {"embedding_dim": 64, "num_layers": 2, "num_heads": 4},
    "M": {"embedding_dim": 128, "num_layers": 2, "num_heads": 4},
    "L": {"embedding_dim": 256, "num_layers": 4, "num_heads": 8},
}


def get_hardware_metadata(device: torch.device) -> Dict[str, Any]:
    """Collect hardware and runtime kernel capability metadata."""
    meta: Dict[str, Any] = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": str(device),
        "device_name": "CPU",
        "architecture_codename": "cpu",
        "compute_capability": None,
        "total_vram_mb": 0.0,
        "bf16_supported": False,
        "sdpa_flash_sdp_enabled": False,
        "sdpa_mem_efficient_sdp_enabled": False,
        "sdpa_math_sdp_enabled": True,
        "triton_compiler_available": False,
    }
    if device.type == "cuda" and torch.cuda.is_available():
        idx = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        raw_name = str(props.name)
        total_mb = float(props.total_memory) / (1024.0 * 1024.0)
        cap_str = f"{props.major}.{props.minor}"
        # Map known SM86 20GB workstation card label
        display_name = (
            f"NVIDIA GeForce RTX 3080 Ti 20GB ({raw_name})"
            if (props.major == 8 and props.minor == 6 and total_mb >= 19000.0 and "3080" not in raw_name)
            else raw_name
        )
        meta.update(
            {
                "device_name": display_name,
                "raw_device_name": raw_name,
                "architecture_codename": "Ampere (SM86)" if (props.major == 8 and props.minor == 6) else f"SM{props.major}{props.minor}",
                "compute_capability": cap_str,
                "multi_processor_count": int(props.multi_processor_count),
                "total_vram_mb": round(total_mb, 2),
                "cuda_runtime_version": str(torch.version.cuda),
                "cudnn_version": int(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else None,
                "bf16_supported": bool(torch.cuda.is_bf16_supported()),
                "sdpa_flash_sdp_enabled": bool(torch.backends.cuda.flash_sdp_enabled()),
                "sdpa_mem_efficient_sdp_enabled": bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
                "sdpa_math_sdp_enabled": bool(torch.backends.cuda.math_sdp_enabled()),
            }
        )
    try:
        import triton  # type: ignore  # noqa: F401

        meta["triton_compiler_available"] = True
    except Exception:
        meta["triton_compiler_available"] = False

    return meta


_INDUCTOR_STATUS_CACHE: Dict[str, str] = {}


def _resolve_sdpa_compile_status(
    model: BaseSequentialEncoder,
    eval_ids: torch.Tensor,
    eval_ts: torch.Tensor,
    dev: torch.device,
) -> str:
    dev_key = str(dev)
    if dev_key in _INDUCTOR_STATUS_CACHE:
        return _INDUCTOR_STATUS_CACHE[dev_key]
    try:
        compiled = torch.compile(model, backend="inductor")
        with torch.no_grad(), torch.autocast(device_type=dev.type, dtype=torch.bfloat16):
            _ = compiled(eval_ids[:2], timestamps=eval_ts[:2])
        status = "torch_compile_inductor_sdpa_bf16"
    except Exception as exc:
        short_reason = type(exc).__name__
        status = f"sdpa_jit_bf16_fallback ({short_reason})"
        if hasattr(torch, "_dynamo"):
            torch._dynamo.reset()
    _INDUCTOR_STATUS_CACHE[dev_key] = status
    return status


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _measure_peak_vram_mb(device: torch.device, model: nn.Module) -> float:
    if device.type == "cuda" and torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated(device)) / (1024.0 * 1024.0)
    # Fallback parameter + buffer footprint on CPU for unit tests
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buf_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    return float(param_bytes + buf_bytes) / (1024.0 * 1024.0)


@torch.no_grad()
def _compute_top12_items(
    model: BaseSequentialEncoder,
    eval_input_ids: torch.Tensor,
    eval_timestamps: torch.Tensor,
    mode: str,
    device: torch.device,
    top_k: int = 12,
) -> torch.Tensor:
    """Compute Top-12 retrieved item IDs (B, 12) for an evaluation batch under the specified execution mode."""
    model.eval()
    use_amp = mode in ("amp_bf16", "amp_fp16", "sdpa_compiled_bf16")
    amp_dtype = torch.float16 if mode == "amp_fp16" else torch.bfloat16
    if device.type == "cpu" and amp_dtype == torch.float16:
        amp_dtype = torch.bfloat16

    if use_amp:
        with torch.autocast(device_type=device.type, dtype=amp_dtype):
            u_vec = model.encode_last_step(eval_input_ids, timestamps=eval_timestamps)
            item_emb = model.item_embedding.weight[1:]
            scores = torch.matmul(u_vec.to(item_emb.dtype), item_emb.t())
    else:
        u_vec = model.encode_last_step(eval_input_ids, timestamps=eval_timestamps)
        item_emb = model.item_embedding.weight[1:]
        scores = torch.matmul(u_vec, item_emb.t())

    k_use = min(int(top_k), scores.size(-1))
    _, top_idx = torch.topk(scores.to(torch.float32), k=k_use, dim=-1, largest=True, sorted=True)
    return top_idx


def _compute_top12_agreement_metrics(
    ref_top12: torch.Tensor,
    cand_top12: torch.Tensor,
) -> Dict[str, float]:
    """Compute Top-12 overlap agreement, Jaccard similarity, and Top-1 exact agreement vs eager_fp32."""
    ref_np = ref_top12.detach().cpu().numpy()
    cand_np = cand_top12.detach().cpu().numpy()
    n_users, k = ref_np.shape
    if n_users == 0 or k == 0:
        return {
            "top12_agreement_vs_eager_fp32": 1.0,
            "top12_jaccard_vs_eager": 1.0,
            "top1_agreement_vs_eager_fp32": 1.0,
        }

    overlaps = []
    jaccards = []
    top1_hits = []
    for i in range(n_users):
        s_ref = set(ref_np[i].tolist())
        s_cand = set(cand_np[i].tolist())
        inter = len(s_ref.intersection(s_cand))
        union = max(len(s_ref.union(s_cand)), 1)
        overlaps.append(float(inter) / float(k))
        jaccards.append(float(inter) / float(union))
        top1_hits.append(1.0 if int(ref_np[i, 0]) == int(cand_np[i, 0]) else 0.0)

    return {
        "top12_agreement_vs_eager_fp32": float(np.mean(overlaps)),
        "top12_jaccard_vs_eager": float(np.mean(jaccards)),
        "top1_agreement_vs_eager_fp32": float(np.mean(top1_hits)),
    }


def benchmark_single_configuration(
    encoder_name: str,
    execution_mode: str,
    num_items: int = 5000,
    embedding_dim: int = 128,
    max_seq_len: int = 64,
    num_layers: int = 2,
    num_heads: int = 4,
    batch_size: int = 256,
    infer_batch_size: int = 256,
    num_negatives: int = 256,
    warmup_steps: int = 5,
    train_steps: int = 20,
    infer_steps: int = 40,
    seed: int = 42,
    device: Optional[str] = None,
    ref_top12: Optional[torch.Tensor] = None,
    eval_inputs: Optional[Dict[str, torch.Tensor]] = None,
) -> Dict[str, Any]:
    """Benchmark a single (encoder, execution_mode, seq_len, capacity) configuration."""
    if execution_mode not in R3_EXECUTION_MODES:
        raise ValueError(f"Unknown execution_mode '{execution_mode}'. Valid: {R3_EXECUTION_MODES}")

    dev_str = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(dev_str)

    torch.manual_seed(seed)
    np.random.seed(seed)
    if dev.type == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Configure float32 matmul precision according to execution mode
    if execution_mode == "eager_fp32":
        torch.set_float32_matmul_precision("highest")
    else:
        torch.set_float32_matmul_precision("high")

    model = build_r2_encoder(
        encoder_name=encoder_name,
        num_items=num_items,
        embedding_dim=embedding_dim,
        max_seq_len=max_seq_len,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=0.0,
    ).to(dev)

    param_count = sum(p.numel() for p in model.parameters())

    # Prepare deterministic evaluation batch BEFORE training to measure pure numerical precision agreement vs FP32
    if eval_inputs is not None:
        eval_ids = eval_inputs["input_ids"].to(dev)
        eval_ts = eval_inputs["timestamps"].to(dev)
    else:
        gen_eval = torch.Generator(device=dev)
        gen_eval.manual_seed(seed + 999)
        eval_ids = torch.randint(1, num_items + 1, (infer_batch_size, max_seq_len), device=dev, generator=gen_eval)
        # Include realistic padding in first 15% of positions for a subset of users
        pad_len = max(1, max_seq_len // 6)
        eval_ids[: infer_batch_size // 3, :pad_len] = 0
        eval_ts = (
            torch.arange(max_seq_len, device=dev, dtype=torch.float32)
            .unsqueeze(0)
            .expand(infer_batch_size, max_seq_len)
            .contiguous()
        )

    # Configure SDPA / compile backend
    compile_status = "eager"
    if execution_mode == "sdpa_compiled_bf16":
        model.set_sdpa_enabled(True)
        compile_status = _resolve_sdpa_compile_status(model, eval_ids, eval_ts, dev)
    else:
        model.set_sdpa_enabled(False)

    # Compute pre-training Top-12 predictions on identical initial weights for numerical agreement check
    cand_top12 = _compute_top12_items(
        model=model,
        eval_input_ids=eval_ids,
        eval_timestamps=eval_ts,
        mode=execution_mode,
        device=dev,
        top_k=12,
    )
    if ref_top12 is None:
        ref_top12 = _compute_top12_items(
            model=model,
            eval_input_ids=eval_ids,
            eval_timestamps=eval_ts,
            mode="eager_fp32",
            device=dev,
            top_k=12,
        )
    agreement = _compute_top12_agreement_metrics(ref_top12, cand_top12)

    # Prepare synthetic training batch tensors on device
    gen_train = torch.Generator(device=dev)
    gen_train.manual_seed(seed + 101)
    train_input_ids = torch.randint(1, num_items + 1, (batch_size, max_seq_len), device=dev, generator=gen_train)
    train_timestamps = (
        torch.arange(max_seq_len, device=dev, dtype=torch.float32)
        .unsqueeze(0)
        .expand(batch_size, max_seq_len)
        .contiguous()
    )
    train_target_ids = torch.randint(1, num_items + 1, (batch_size, max_seq_len), device=dev, generator=gen_train)

    item_probs = torch.ones(num_items + 1, device=dev, dtype=torch.float32) / float(num_items)
    item_log_probs = torch.log(item_probs)

    criterion = SampledSoftmax256Loss(num_negatives=num_negatives, logq_correction=True).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    use_amp = execution_mode in ("amp_bf16", "amp_fp16", "sdpa_compiled_bf16")
    amp_dtype = torch.float16 if execution_mode == "amp_fp16" else torch.bfloat16
    if dev.type == "cpu" and amp_dtype == torch.float16:
        amp_dtype = torch.bfloat16

    use_scaler = (execution_mode == "amp_fp16") and (dev.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler) if dev.type == "cuda" else None

    def _run_one_train_step() -> None:
        optimizer.zero_grad(set_to_none=True)
        flat_pos = train_target_ids.reshape(-1)
        flat_neg = sample_negatives_excluding_positives(
            pos_ids=flat_pos,
            num_items=num_items,
            num_negatives=num_negatives,
            generator=gen_train,
        )
        if use_amp:
            with torch.autocast(device_type=dev.type, dtype=amp_dtype):
                seq_repr = model(train_input_ids, timestamps=train_timestamps)
                flat_repr = seq_repr.reshape(-1, embedding_dim)
                loss = criterion(
                    flat_repr,
                    model.item_embedding.weight,
                    flat_pos,
                    flat_neg,
                    item_log_probs=item_log_probs,
                )
        else:
            seq_repr = model(train_input_ids, timestamps=train_timestamps)
            flat_repr = seq_repr.reshape(-1, embedding_dim)
            loss = criterion(
                flat_repr,
                model.item_embedding.weight,
                flat_pos,
                flat_neg,
                item_log_probs=item_log_probs,
            )

        if scaler is not None and use_scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

    # 1. Training Warmup + Timed Loop
    model.train()
    for _ in range(int(warmup_steps)):
        _run_one_train_step()
    _sync_device(dev)

    if dev.type == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(dev)

    train_step_times_ms: List[float] = []
    t_train_start = time.perf_counter()
    for _ in range(int(train_steps)):
        t_s = time.perf_counter()
        _run_one_train_step()
        _sync_device(dev)
        train_step_times_ms.append((time.perf_counter() - t_s) * 1000.0)
    total_train_sec = max(time.perf_counter() - t_train_start, 1e-9)
    train_peak_vram_mb = _measure_peak_vram_mb(dev, model)

    train_sequences_per_sec = float(train_steps * batch_size) / total_train_sec
    train_tokens_per_sec = float(train_steps * batch_size * max_seq_len) / total_train_sec

    # 2. Inference Warmup + Timed Loop (Full-Catalog Top-12 MIPS Retrieval)
    model.eval()
    with torch.no_grad():
        for _ in range(int(warmup_steps)):
            _ = _compute_top12_items(
                model=model,
                eval_input_ids=eval_ids,
                eval_timestamps=eval_ts,
                mode=execution_mode,
                device=dev,
                top_k=12,
            )
        _sync_device(dev)

        if dev.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(dev)

        infer_latencies_ms: List[float] = []
        t_infer_start = time.perf_counter()
        for _ in range(int(infer_steps)):
            t0 = time.perf_counter()
            _ = _compute_top12_items(
                model=model,
                eval_input_ids=eval_ids,
                eval_timestamps=eval_ts,
                mode=execution_mode,
                device=dev,
                top_k=12,
            )
            _sync_device(dev)
            infer_latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        total_infer_sec = max(time.perf_counter() - t_infer_start, 1e-9)
        infer_peak_vram_mb = _measure_peak_vram_mb(dev, model)

    infer_users_per_sec = float(infer_steps * infer_batch_size) / total_infer_sec
    infer_p50_ms = float(np.percentile(infer_latencies_ms, 50))
    infer_p95_ms = float(np.percentile(infer_latencies_ms, 95))
    infer_mean_ms = float(np.mean(infer_latencies_ms))
    peak_vram_mb = max(train_peak_vram_mb, infer_peak_vram_mb)

    return {
        "encoder": encoder_name,
        "execution_mode": execution_mode,
        "compile_status": compile_status,
        "num_items": int(num_items),
        "embedding_dim": int(embedding_dim),
        "max_seq_len": int(max_seq_len),
        "num_layers": int(num_layers),
        "num_heads": int(num_heads),
        "batch_size": int(batch_size),
        "infer_batch_size": int(infer_batch_size),
        "num_negatives": int(num_negatives),
        "parameter_count": int(param_count),
        "train_sequences_per_sec": round(train_sequences_per_sec, 2),
        "train_samples_per_sec": round(train_sequences_per_sec, 2),
        "train_tokens_per_sec": round(train_tokens_per_sec, 2),
        "train_step_p50_ms": round(float(np.percentile(train_step_times_ms, 50)), 3),
        "train_step_p95_ms": round(float(np.percentile(train_step_times_ms, 95)), 3),
        "infer_users_per_sec": round(infer_users_per_sec, 2),
        "infer_p50_latency_ms": round(infer_p50_ms, 3),
        "infer_p95_latency_ms": round(infer_p95_ms, 3),
        "infer_mean_latency_ms": round(infer_mean_ms, 3),
        "peak_vram_mb": round(peak_vram_mb, 2),
        "train_peak_vram_mb": round(train_peak_vram_mb, 2),
        "infer_peak_vram_mb": round(infer_peak_vram_mb, 2),
        "top12_agreement_vs_eager_fp32": round(agreement["top12_agreement_vs_eager_fp32"], 6),
        "top12_jaccard_vs_eager": round(agreement["top12_jaccard_vs_eager"], 6),
        "top1_agreement_vs_eager_fp32": round(agreement["top1_agreement_vs_eager_fp32"], 6),
        "_ref_top12": ref_top12,
    }


def _extract_pareto_frontier(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Identify Pareto-optimal configurations maximizing throughput & agreement while minimizing P95 latency & VRAM."""
    frontier: List[Dict[str, Any]] = []
    for i, a in enumerate(records):
        dominated = False
        for j, b in enumerate(records):
            if i == j:
                continue
            # b dominates a if b is >= on throughput and agreement, <= on latency and VRAM, and strictly better on >=1
            better_or_eq = (
                b["train_sequences_per_sec"] >= a["train_sequences_per_sec"]
                and b["infer_p95_latency_ms"] <= a["infer_p95_latency_ms"]
                and b["peak_vram_mb"] <= a["peak_vram_mb"]
                and b["top12_agreement_vs_eager_fp32"] >= a["top12_agreement_vs_eager_fp32"] - 1e-4
            )
            strictly_better = (
                b["train_sequences_per_sec"] > a["train_sequences_per_sec"]
                or b["infer_p95_latency_ms"] < a["infer_p95_latency_ms"]
                or b["peak_vram_mb"] < a["peak_vram_mb"]
            )
            if better_or_eq and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(
                {
                    "encoder": a["encoder"],
                    "execution_mode": a["execution_mode"],
                    "train_sequences_per_sec": a["train_sequences_per_sec"],
                    "infer_users_per_sec": a["infer_users_per_sec"],
                    "infer_p95_latency_ms": a["infer_p95_latency_ms"],
                    "peak_vram_mb": a["peak_vram_mb"],
                    "top12_agreement_vs_eager_fp32": a["top12_agreement_vs_eager_fp32"],
                }
            )
    return frontier


def run_r3_compute_benchmark(
    encoders: Sequence[str] = R3_ENCODERS,
    execution_modes: Sequence[str] = R3_EXECUTION_MODES,
    seq_lengths: Sequence[int] = (32, 64, 128, 256),
    capacity_tiers: Sequence[str] = ("S", "M", "L"),
    num_items: int = 5000,
    base_embedding_dim: int = 128,
    base_seq_len: int = 64,
    base_num_layers: int = 2,
    base_num_heads: int = 4,
    batch_size: int = 256,
    infer_batch_size: int = 256,
    num_negatives: int = 256,
    warmup_steps: int = 5,
    train_steps: int = 20,
    infer_steps: int = 40,
    seed: int = 42,
    device: Optional[str] = None,
) -> Dict[str, Any]:
    """Run complete R3 compute architecture Pareto benchmark across encoders, execution modes, L, and capacity."""
    dev_str = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(dev_str)
    hw_meta = get_hardware_metadata(dev)

    # 1. Primary 5 Encoders x 4 Execution Modes Matrix
    primary_records: List[Dict[str, Any]] = []
    speedup_summary: Dict[str, Dict[str, Any]] = {}

    for enc in encoders:
        ref_top12: Optional[torch.Tensor] = None
        gen_eval = torch.Generator(device=dev)
        gen_eval.manual_seed(seed + 999)
        eval_ids = torch.randint(1, num_items + 1, (infer_batch_size, base_seq_len), device=dev, generator=gen_eval)
        pad_len = max(1, base_seq_len // 6)
        eval_ids[: max(1, infer_batch_size // 3), :pad_len] = 0
        eval_ts = (
            torch.arange(base_seq_len, device=dev, dtype=torch.float32)
            .unsqueeze(0)
            .expand(infer_batch_size, base_seq_len)
            .contiguous()
        )
        eval_pack = {"input_ids": eval_ids, "timestamps": eval_ts}

        enc_rows: Dict[str, Dict[str, Any]] = {}
        for mode in execution_modes:
            res = benchmark_single_configuration(
                encoder_name=enc,
                execution_mode=mode,
                num_items=num_items,
                embedding_dim=base_embedding_dim,
                max_seq_len=base_seq_len,
                num_layers=base_num_layers,
                num_heads=base_num_heads,
                batch_size=batch_size,
                infer_batch_size=infer_batch_size,
                num_negatives=num_negatives,
                warmup_steps=warmup_steps,
                train_steps=train_steps,
                infer_steps=infer_steps,
                seed=seed,
                device=dev_str,
                ref_top12=ref_top12,
                eval_inputs=eval_pack,
            )
            if ref_top12 is None:
                ref_top12 = res["_ref_top12"]
            clean_res = {k: v for k, v in res.items() if not k.startswith("_")}
            primary_records.append(clean_res)
            enc_rows[mode] = clean_res

        if "eager_fp32" in enc_rows:
            base_r = enc_rows["eager_fp32"]
            speedup_summary[enc] = {}
            for mode, r in enc_rows.items():
                speedup_summary[enc][mode] = {
                    "train_speedup_vs_fp32": round(
                        r["train_sequences_per_sec"] / max(base_r["train_sequences_per_sec"], 1e-9), 3
                    ),
                    "infer_speedup_vs_fp32": round(
                        r["infer_users_per_sec"] / max(base_r["infer_users_per_sec"], 1e-9), 3
                    ),
                    "p95_latency_ratio_vs_fp32": round(
                        r["infer_p95_latency_ms"] / max(base_r["infer_p95_latency_ms"], 1e-9), 3
                    ),
                    "vram_ratio_vs_fp32": round(
                        r["peak_vram_mb"] / max(base_r["peak_vram_mb"], 1e-9), 3
                    ),
                    "top12_agreement_vs_eager_fp32": r["top12_agreement_vs_eager_fp32"],
                }

    # 2. Sequence Length Scaling (L = 32, 64, 128, 256)
    seq_scaling_records: List[Dict[str, Any]] = []
    scaling_modes = [m for m in ("eager_fp32", "sdpa_compiled_bf16") if m in execution_modes]
    if not scaling_modes:
        scaling_modes = [execution_modes[0]]

    for seq_len in seq_lengths:
        for enc in encoders:
            ref_top12_l: Optional[torch.Tensor] = None
            for mode in scaling_modes:
                res = benchmark_single_configuration(
                    encoder_name=enc,
                    execution_mode=mode,
                    num_items=num_items,
                    embedding_dim=base_embedding_dim,
                    max_seq_len=int(seq_len),
                    num_layers=base_num_layers,
                    num_heads=base_num_heads,
                    batch_size=batch_size,
                    infer_batch_size=infer_batch_size,
                    num_negatives=num_negatives,
                    warmup_steps=warmup_steps,
                    train_steps=max(2, train_steps // 2),
                    infer_steps=max(4, infer_steps // 2),
                    seed=seed,
                    device=dev_str,
                    ref_top12=ref_top12_l,
                )
                if ref_top12_l is None:
                    ref_top12_l = res["_ref_top12"]
                clean_res = {k: v for k, v in res.items() if not k.startswith("_")}
                seq_scaling_records.append(clean_res)

    # 3. Model Capacity Scaling (S / M / L)
    capacity_scaling_records: List[Dict[str, Any]] = []
    for tier in capacity_tiers:
        if tier not in R3_CAPACITY_SPECS:
            continue
        c_spec = R3_CAPACITY_SPECS[tier]
        for enc in encoders:
            ref_top12_c: Optional[torch.Tensor] = None
            for mode in scaling_modes:
                res = benchmark_single_configuration(
                    encoder_name=enc,
                    execution_mode=mode,
                    num_items=num_items,
                    embedding_dim=c_spec["embedding_dim"],
                    max_seq_len=base_seq_len,
                    num_layers=c_spec["num_layers"],
                    num_heads=c_spec["num_heads"],
                    batch_size=batch_size,
                    infer_batch_size=infer_batch_size,
                    num_negatives=num_negatives,
                    warmup_steps=warmup_steps,
                    train_steps=max(2, train_steps // 2),
                    infer_steps=max(4, infer_steps // 2),
                    seed=seed,
                    device=dev_str,
                    ref_top12=ref_top12_c,
                )
                if ref_top12_c is None:
                    ref_top12_c = res["_ref_top12"]
                clean_res = {k: v for k, v in res.items() if not k.startswith("_")}
                clean_res["capacity_tier"] = tier
                capacity_scaling_records.append(clean_res)

    pareto_frontier = _extract_pareto_frontier(primary_records)

    return {
        "contract_version": "0.9",
        "benchmark_stage": "R3_RTX3080Ti_Compute_Pareto",
        "hardware_metadata": hw_meta,
        "base_config": {
            "num_items": int(num_items),
            "embedding_dim": int(base_embedding_dim),
            "max_seq_len": int(base_seq_len),
            "num_layers": int(base_num_layers),
            "num_heads": int(base_num_heads),
            "batch_size": int(batch_size),
            "infer_batch_size": int(infer_batch_size),
            "num_negatives": int(num_negatives),
            "warmup_steps": int(warmup_steps),
            "train_steps": int(train_steps),
            "infer_steps": int(infer_steps),
            "seed": int(seed),
        },
        "primary_matrix_5x4": primary_records,
        "speedup_vs_eager_fp32": speedup_summary,
        "sequence_length_scaling": seq_scaling_records,
        "capacity_scaling": capacity_scaling_records,
        "pareto_frontier": pareto_frontier,
    }
