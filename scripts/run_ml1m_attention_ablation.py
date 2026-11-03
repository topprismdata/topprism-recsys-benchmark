"""ML-1M attention-operator causal ablation — PREREG-attention-ablation-v1.1 (GPT APPROVED 2026-10-31).

Arms (all trained from scratch, same data/optimizer/early-stop; paired seeds):
  A0 sasrec_full      — content-conditioned Q/K (full SASRec)
  A1 sasrec_uniform   — uniform 1/n_t causal-prefix weights (no adaptive weighting)
  A2 sasrec_pos_only  — Q/K from positional embedding only (capacity-matched main control)
  A3 mean_pool        — unordered pooling baseline (original definition)

Frozen protocol: 5-core, LOO full-catalog ranking (no seen-masking), SS256+logq,
d128/2L/4H/L200, 100 epochs cap, val-HR@10 early stop (patience 10 checks @5ep),
seeds 42/43/44, identical init + negative-sampling sequence within each seed.

Preregistered tests (only these):
  H1 primary  d1 = HR(A0) - HR(A2)  per-seed paired user-level bootstrap CI95
  H1 support  d2 = HR(A0) - HR(A1)  same
  H2          A1-A3, A2-A3 (Holm-corrected across the two)
Outputs: artifacts/eval/ml1m_ablation/{hits_<enc>_s<seed>.npz, results.json, verdicts.json}
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
SEEDS = [42, 43, 44, 45, 46, 47]  # v1.2 extension: 45-47 are the final preregistered addition

import torch

from run_ml1m_attention_v2 import load_ml1m, make_inputs
from src.models.r2_encoders import build_r2_encoder
from src.models.r2_losses import SampledSoftmax256Loss

L = 200
D = 128
LAYERS = 2
HEADS = 4
NEG = 256
BATCH = 256
EPOCHS = 100
LR = 2e-3
ENCODERS = ["sasrec_full", "sasrec_uniform", "sasrec_pos_only", "mean_pool"]
DEVICE = "cuda"
OUTDIR = REPO / "artifacts/eval/ml1m_ablation"
BOOT_N = 10000


def log(msg):
    print(msg, flush=True)


@torch.no_grad()
def eval_hr_peruser(model, item_w, inputs, targets):
    """Full-catalog ranking; returns per-user hit(10) / ndcg(10) vectors (test contract)."""
    model.eval()
    ids_all = inputs
    dev = next(model.parameters()).device
    hits, ndcgs = [], []
    for i in range(0, len(ids_all), 512):
        ids = ids_all[i:i + 512].to(dev)
        ts = torch.arange(L, dtype=torch.float32, device=dev).unsqueeze(0).expand(ids.size(0), -1)
        reps = model.encode_last_step(ids, timestamps=ts)
        scores = reps @ item_w.t()
        scores[:, 0] = -float("inf")
        top10 = torch.topk(scores, k=10, dim=1).indices.cpu().numpy()
        for j in range(len(top10)):
            tgt = targets[i + j]
            if tgt in top10[j]:
                hits.append(1)
                ndcgs.append(1.0 / np.log2(int(np.flatnonzero(top10[j] == tgt)[0]) + 2))
            else:
                hits.append(0)
                ndcgs.append(0.0)
    return np.array(hits, dtype=np.int8), np.array(ndcgs, dtype=np.float32)


def train_encoder(name, seqs, num_items, id_of, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    dev = torch.device(DEVICE)
    model = build_r2_encoder(name, num_items=num_items, embedding_dim=D, max_seq_len=L,
                             num_layers=LAYERS, num_heads=HEADS, dropout=0.1).to(dev)
    criterion = SampledSoftmax256Loss(num_negatives=NEG, logq_correction=True).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    users = sorted(seqs)
    item_pop = np.zeros(num_items + 1, dtype=np.float64)
    for s in seqs.values():
        for it in s:
            item_pop[id_of[it]] += 1
    logq = torch.tensor(np.log(item_pop / max(item_pop[1:].sum(), 1) + 1e-9), dtype=torch.float32, device=dev)

    train_users = [u for u in users if len(seqs[u]) >= 2]
    val_users = [u for u in users if len(seqs[u]) >= 3]
    val_rows, _vt = make_inputs([seqs[u][:-2] for u in val_users], id_of, L)
    val_targets = np.array([id_of[seqs[u][-2]] for u in val_users], dtype=np.int64)
    test_rows, _tt = make_inputs([seqs[u][:-1] for u in train_users], id_of, L)
    test_targets = np.array([id_of[seqs[u][-1]] for u in train_users], dtype=np.int64)

    rows, _tss = make_inputs([seqs[u][:-1] for u in train_users], id_of, L)
    n_rows = len(rows)
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)  # identical negative-sampling stream within a seed across arms
    item_w = model.item_embedding.weight

    def quick_val_hr():
        model.eval()
        with torch.no_grad():
            h = 0
            for i in range(0, len(val_rows), 512):
                ids = val_rows[i:i + 512].to(dev)
                ts = torch.arange(L, dtype=torch.float32, device=dev).unsqueeze(0).expand(ids.size(0), -1)
                reps = model.encode_last_step(ids, timestamps=ts)
                sc = reps @ item_w.t()
                sc[:, 0] = -float("inf")
                top = torch.topk(sc, k=10, dim=1).indices.cpu().numpy()
                h += sum(1 for j in range(len(top)) if val_targets[i + j] in top[j])
        return h / len(val_targets)

    best_val, best_state, bad = -1.0, None, 0
    t0 = time.time()
    epoch = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n_rows)
        losses = []
        for start in range(0, n_rows, BATCH):
            idx = perm[start:start + BATCH]
            b_inp = rows[idx].to(dev)
            pos_t = torch.arange(L, dtype=torch.float32, device=dev).unsqueeze(0).expand(b_inp.size(0), -1)
            seq_repr = model(b_inp, timestamps=pos_t)
            b_tgt = torch.zeros_like(b_inp)
            b_tgt[:, :-1] = b_inp[:, 1:]
            flat_mask = b_tgt > 0
            flat_repr = seq_repr[flat_mask]
            flat_pos = b_tgt[flat_mask]
            neg = torch.randint(1, num_items + 1, (flat_repr.size(0), NEG), device=dev, generator=gen)
            loss = criterion(flat_repr, item_w, flat_pos, neg, item_log_probs=logq)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss))
        if epoch % 5 == 0:
            v = quick_val_hr()
            if v > best_val:
                best_val, bad = v, 0
                best_state = {k: vv.detach().clone() for k, vv in model.state_dict().items()}
            else:
                bad += 1
                if bad >= 10:
                    log(f"  [{name}/s{seed}] early stop epoch {epoch}")
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    hits, ndcgs = eval_hr_peruser(model, model.item_embedding.weight, test_rows, test_targets)
    return {
        "hr10": float(hits.mean()),
        "ndcg10": float(ndcgs.mean()),
        "best_val_hr10": round(best_val, 4),
        "epochs_run": epoch,
        "seconds": round(time.time() - t0, 1),
    }, hits, ndcgs


def paired_bootstrap_ci(d, n=BOOT_N, seed=0):
    """User-level paired bootstrap of a mean difference; returns (mean, lo95, hi95, one_sided_p)."""
    rng = np.random.default_rng(seed)
    m = len(d)
    idx = rng.integers(0, m, size=(n, m))
    means = d[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    # one-sided p: fraction of bootstrap means <= 0 (for positive-effect hypothesis)
    p = float((means <= 0).mean())
    return float(d.mean()), float(lo), float(hi), p


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTDIR / "results.json"
    results = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else {}

    seqs, num_items = load_ml1m()
    id_of = {it: i + 1 for i, it in enumerate(sorted({i for s in seqs.values() for i in s}))}

    for seed in SEEDS:
        for enc in ENCODERS:
            key = f"{enc}_s{seed}"
            hits_path = OUTDIR / f"hits_{enc}_s{seed}.npz"
            if key in results and hits_path.exists():
                log(f"[cached] {key}: hr10={results[key]['hr10']:.4f}")
                continue
            log(f"[train] {key}")
            metrics, hits, ndcgs = train_encoder(enc, seqs, num_items, id_of, seed)
            np.savez_compressed(hits_path, hits=hits, ndcg=ndcgs)
            results[key] = metrics
            results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
            log(f"[{key}] TEST HR@10={metrics['hr10']:.4f} NDCG@10={metrics['ndcg10']:.4f} ({metrics['seconds']}s, {metrics['epochs_run']}ep)")

    # ---- preregistered tests ----
    verdicts = {"protocol": {"seeds": SEEDS, "boot_n": BOOT_N, "arms": ENCODERS,
                             "tests": ["H1_primary_d1=A0-A2", "H1_support_d2=A0-A1", "H2_a=A1-A3", "H2_b=A2-A3"]}}
    def hits_of(enc, seed):
        return np.load(OUTDIR / f"hits_{enc}_s{seed}.npz")["hits"].astype(np.float64)

    tests = {}
    for seed in SEEDS:
        d1 = hits_of("sasrec_full", seed) - hits_of("sasrec_pos_only", seed)
        d2 = hits_of("sasrec_full", seed) - hits_of("sasrec_uniform", seed)
        ha = hits_of("sasrec_uniform", seed) - hits_of("mean_pool", seed)
        hb = hits_of("sasrec_pos_only", seed) - hits_of("mean_pool", seed)
        tests[f"s{seed}"] = {
            "d1_A0_minus_A2": paired_bootstrap_ci(d1, seed=seed),
            "d2_A0_minus_A1": paired_bootstrap_ci(d2, seed=seed + 100),
            "h2a_A1_minus_A3": paired_bootstrap_ci(ha, seed=seed + 200),
            "h2b_A2_minus_A3": paired_bootstrap_ci(hb, seed=seed + 300),
        }

    def seed_mean(component, idx):
        vals = [tests[f"s{s}"][component][idx] for s in SEEDS]
        return round(float(np.mean(vals)), 5), round(float(min(vals)), 5), round(float(max(vals)), 5)

    summary = {}
    for comp in ("d1_A0_minus_A2", "d2_A0_minus_A1", "h2a_A1_minus_A3", "h2b_A2_minus_A3"):
        summary[comp] = {
            "per_seed": {f"s{s}": [round(x, 5) for x in tests[f"s{s}"][comp]] for s in SEEDS},
            "mean": seed_mean(comp, 0),
            "ci95_lo_mean": seed_mean(comp, 1),
            "ci95_hi_mean": seed_mean(comp, 2),
        }
    # H1 primary gate: d1 CI95 lower bound > 0 (per GPT: phase-2 entry condition)
    d1_lower_bounds = [tests[f"s{s}"]["d1_A0_minus_A2"][1] for s in SEEDS]
    summary["H1_primary_gate_d1_lower_all_positive"] = bool(all(b > 0 for b in d1_lower_bounds))
    # H2 Holm across the two secondary comparisons (bootstrap one-sided p values, per seed)
    for s in SEEDS:
        pa, pb = tests[f"s{s}"]["h2a_A1_minus_A3"][3], tests[f"s{s}"]["h2b_A2_minus_A3"][3]
        ordered = sorted([pa, pb])
        tests[f"s{s}"]["h2_holm"] = {
            "p_a_holm": round(min(ordered[0] * 2, 1.0), 5),
            "p_b_holm": round(ordered[1], 5),
            "both_significant_holm": bool(ordered[0] * 2 < 0.05 and ordered[1] < 0.05),
        }
    verdicts["tests"] = tests
    verdicts["summary"] = summary
    (OUTDIR / "verdicts.json").write_text(json.dumps(verdicts, indent=2), encoding="utf-8")
    log(json.dumps(summary, indent=2))
    log("DONE")


if __name__ == "__main__":
    main()
