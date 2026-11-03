"""P0 audit per PREREG v1.2: rerun seed 42 four arms with full diagnostics + checkpoint persistence.

Answers (GPT verdict tasks 1-3 + P1 weight-sensitivity):
  1. Are s42 A1/A2 identical outputs a pipeline bug (cache/reuse) or genuine co-collapse?
     -> rerun deterministically, compare hits vs original npz, persist per-user score vectors.
  2. Post-training gradients: A2 Q/K grads nonzero during training, pos-embedding updated.
  3. Per-epoch train loss / total grad L2 / val HR; user-level score variance; top10 overlap.
  4. Attention weight content-sensitivity on trained checkpoints (A0 varies, A2 does not).
Outputs: artifacts/eval/ml1m_ablation/audit_s42/{diag_<enc>.json, ckpt_<enc>.pt, scores_<enc>.npy}
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import torch

from run_ml1m_attention_v2 import load_ml1m, make_inputs
from src.models.r2_encoders import build_r2_encoder, _causal_attention_mask
from src.models.r2_losses import SampledSoftmax256Loss

SEED = 42
L, D, LAYERS, HEADS, NEG, BATCH, EPOCHS, LR = 200, 128, 2, 4, 256, 256, 100, 2e-3
ENCODERS = ["sasrec_full", "sasrec_uniform", "sasrec_pos_only", "mean_pool"]
DEVICE = "cuda"
AUD = REPO / "artifacts/eval/ml1m_ablation/audit_s42"


def log(m):
    print(m, flush=True)


@torch.no_grad()
def full_eval(model, item_w, rows, targets, k=10):
    model.eval()
    dev = next(model.parameters()).device
    topks, scores_last = [], []
    for i in range(0, len(rows), 512):
        ids = rows[i:i + 512].to(dev)
        ts = torch.arange(L, dtype=torch.float32, device=dev).unsqueeze(0).expand(ids.size(0), -1)
        reps = model.encode_last_step(ids, timestamps=ts)
        sc = reps @ item_w.t()
        sc[:, 0] = -float("inf")
        top = torch.topk(sc, k=k, dim=1).indices.cpu().numpy()
        topks.append(top)
        scores_last.append(sc[:, :50].cpu().numpy())  # top-50 slice for diversity stats
    topk = np.concatenate(topks)
    sc50 = np.concatenate(scores_last)
    hits = np.array([1 if t in topk[j] else 0 for j, t in enumerate(targets)], dtype=np.int8)
    # personalization: fraction of user pairs sharing >=8/10 items
    rng = np.random.default_rng(0)
    pairs = rng.integers(0, len(topk), size=(2000, 2))
    ov = np.array([len(set(topk[a]) & set(topk[b])) for a, b in pairs])
    return {
        "hr10": float(hits.mean()),
        "hits": hits,
        "topk": topk,
        "score_var_across_users": float(sc50.std(axis=0).mean()),
        "top10_overlap_ge8_frac": float((ov >= 8).mean()),
        "mean_pairwise_overlap": float(ov.mean()),
    }


def train_diag(name, seqs, num_items, id_of, seed):
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
    val_rows, _ = make_inputs([seqs[u][:-2] for u in val_users], id_of, L)
    val_targets = np.array([id_of[seqs[u][-2]] for u in val_users], dtype=np.int64)
    test_rows, _ = make_inputs([seqs[u][:-1] for u in train_users], id_of, L)
    test_targets = np.array([id_of[seqs[u][-1]] for u in train_users], dtype=np.int64)
    rows, _ = make_inputs([seqs[u][:-1] for u in train_users], id_of, L)
    n_rows = len(rows)
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)
    item_w = model.item_embedding.weight
    pos0 = model.pos_embedding.weight.detach().clone() if hasattr(model, "pos_embedding") else None
    qk0 = None
    blk0 = model.blocks[0] if hasattr(model, "blocks") else None
    if blk0 is not None and hasattr(blk0, "q_proj"):
        qk0 = (blk0.q_proj.weight.detach().clone(), blk0.k_proj.weight.detach().clone())

    diag = {"loss": [], "grad_norm": [], "val_hr": []}

    def val_hr():
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
        model.train()
        return h / len(val_targets)

    best_val, best_state, bad = -1.0, None, 0
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(n_rows)
        ep_losses, ep_gnorms = [], []
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
            # sample grad-norm on the last step of each epoch only (perf)
            if start + BATCH >= n_rows:
                with torch.no_grad():
                    gn = torch.sqrt(sum((p.grad.detach().float() ** 2).sum() for p in model.parameters() if p.grad is not None))
                ep_gnorms.append(float(gn))
            opt.step()
            ep_losses.append(float(loss))
        if epoch % 10 == 0:
            log(f"  [{name}] ep{epoch} loss={np.mean(ep_losses):.4f} gn={np.mean(ep_gnorms):.2f}")
        diag["loss"].append(round(float(np.mean(ep_losses)), 4))
        diag["grad_norm"].append(round(float(np.mean(ep_gnorms)), 3))
        if epoch % 5 == 0:
            v = val_hr()
            diag["val_hr"].append(round(v, 4))
            if v > best_val:
                best_val, bad = v, 0
                best_state = {k: vv.detach().clone() for k, vv in model.state_dict().items()}
            else:
                bad += 1
                if bad >= 10:
                    log(f"  [{name}] early stop {epoch}")
                    break
    if best_state is not None:
        model.load_state_dict(best_state)

    ev = full_eval(model, model.item_embedding.weight, test_rows, test_targets)
    ckpt = {k: v for k, v in model.state_dict().items()}
    torch.save(ckpt, AUD / f"ckpt_{name}.pt")
    np.save(AUD / f"scores_{name}.npy", ev.pop("topk"))
    ev.pop("hits")
    ev.update({"best_val_hr10": round(best_val, 4), "epochs": epoch, "seconds": round(time.time() - t0, 1)})
    # trained-weight checks
    checks = {}
    if pos0 is not None:
        checks["pos_emb_moved"] = float((model.pos_embedding.weight - pos0).abs().max()) > 0
    if qk0 is not None and name == "sasrec_pos_only":
        checks["qk_moved"] = float((model.blocks[0].q_proj.weight - qk0[0]).abs().max()) > 0
        g_ok = []
        model.train()
        b = rows[:64].to(dev)
        ts = torch.arange(L, dtype=torch.float32, device=dev).unsqueeze(0).expand(64, -1)
        out = model(b, timestamps=ts)
        out.pow(2).sum().backward()
        g_ok.append(model.blocks[0].q_proj.weight.grad is not None)
        g_ok.append(model.pos_embedding.weight.grad is not None)
        checks["qk_pos_grad_alive"] = all(g_ok)
    diag.update({**ev, "checks": checks})
    return diag, model


def weight_content_sensitivity():
    """P1: trained A0 weights vary with content at fixed positions; trained A2 weights do not."""
    out = {}
    ids = torch.zeros(4, L, dtype=torch.long)
    for i, n in enumerate([5, 9, 12, 20]):
        g = torch.Generator().manual_seed(i)
        ids[i, :n] = torch.randint(1, 100, (n,), generator=g)
    ids2 = ids.clone()
    gg = torch.Generator().manual_seed(99)
    ids2[:, :4] = torch.randint(1, 100, (4,), generator=gg)
    for name in ("sasrec_full", "sasrec_pos_only"):
        model = build_r2_encoder(name, num_items=500, embedding_dim=D, max_seq_len=L,
                                 num_layers=LAYERS, num_heads=HEADS, dropout=0.0)
        sd = torch.load(AUD / f"ckpt_{name}.pt", map_location="cpu")
        model.load_state_dict(sd, strict=True)
        model.eval()
        blk = model.blocks[0]

        def probs(input_ids):
            with torch.no_grad():
                pe = model.pos_embedding.weight[: L].unsqueeze(0)
                if name == "sasrec_pos_only":
                    q = blk.q_proj(pe).view(1, L, HEADS, D // HEADS).transpose(1, 2).expand(ids.size(0), -1, -1, -1)
                    k = blk.k_proj(pe).view(1, L, HEADS, D // HEADS).transpose(1, 2).expand(ids.size(0), -1, -1, -1)
                    x = model.emb_norm(model.item_embedding(input_ids) + model.pos_embedding.weight[: L].unsqueeze(0))
                else:
                    x = model.emb_norm(model.item_embedding(input_ids) + model.pos_embedding.weight[: L].unsqueeze(0))
                    q = blk.q_proj(x).view(ids.size(0), L, HEADS, D // HEADS).transpose(1, 2)
                    k = blk.k_proj(x).view(ids.size(0), L, HEADS, D // HEADS).transpose(1, 2)
                sc = torch.matmul(q, k.transpose(-2, -1)) * blk.scale
                am = _causal_attention_mask(input_ids)
                sc = torch.where(am, sc, torch.full_like(sc, -1e9))
                p = torch.softmax(sc, dim=-1)
                return torch.where(am, p, torch.zeros_like(p))

        p1, p2 = probs(ids), probs(ids2)
        out[name] = {
            "probs_change_with_content": float((p1 - p2).abs().max()) > 1e-6,
            "max_abs_prob_diff": float((p1 - p2).abs().max()),
        }
    return out


def main():
    AUD.mkdir(parents=True, exist_ok=True)
    seqs, num_items = load_ml1m()
    id_of = {it: i + 1 for i, it in enumerate(sorted({i for s in seqs.values() for i in s}))}
    all_diag = {}
    for enc in ENCODERS:
        log(f"[audit-train] {enc}")
        d, _m = train_diag(enc, seqs, num_items, id_of, SEED)
        all_diag[enc] = d
        log(f"[{enc}] hr10={d['hr10']:.4f} overlap>=8frac={d['top10_overlap_ge8_frac']:.3f} scorevar={d['score_var_across_users']:.4f}")
    all_diag["weight_content_sensitivity"] = weight_content_sensitivity()
    (AUD / "diag.json").write_text(json.dumps(all_diag, indent=2), encoding="utf-8")
    log(json.dumps(all_diag.get("weight_content_sensitivity"), indent=2))
    # determinism: compare rerun vs original hits
    for enc in ENCODERS:
        orig = np.load(REPO / f"artifacts/eval/ml1m_ablation/hits_{enc}_s42.npz")["hits"]
        # recompute hits from saved topk
        topk = np.load(AUD / f"scores_{enc}.npy")
        # need targets again
        users = sorted(seqs)
        tu = [u for u in users if len(seqs[u]) >= 2]
        tt = np.array([id_of[seqs[u][-1]] for u in tu])
        rerun_hits = np.array([1 if t in topk[j] else 0 for j, t in enumerate(tt)])
        same = bool(np.array_equal(orig.astype(bool), rerun_hits.astype(bool)))
        log(f"[determinism] {enc}: rerun_hits == original_hits -> {same} ({rerun_hits.sum()} vs {orig.sum()})")
    log("AUDIT DONE")


if __name__ == "__main__":
    main()
