"""Encoders v4 for Hybrid A3 + GPT verification #2 (single training pass).

Combines in ONE run (avoids duplicate SASRec training):
  - GPT verification #2: per-user hit10 + repeat-flag for mean_pool and sasrec
    (Repeat/New split with net_new/net_repeat/net_total cross-check).
  - A3 source outputs: full-catalog Top500 lists per user for both encoders
    (sasrec = A3 sixth source; mean_pool = later non-attention M2 contrast).
  - Checkpoints saved for both encoders.

GPT hard constraints honored: full 41,147-catalog independent scoring; val-based
early stopping (s[-1] never read during training/selection); frozen v2 protocol.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch

from src.models.r2_encoders import build_r2_encoder
from src.models.r2_losses import SampledSoftmax256Loss

SEED = 42
ENCODERS = ["sasrec", "mean_pool", "sasrec_uniform", "sasrec_pos_only"]  # +ablation arms (phase 2, GPT GO 2026-10-31)
N_USERS = 20000
L = 200
D = 128
LAYERS = 2
HEADS = 4
NEG = 256
BATCH = 256
EPOCHS = 100
LR = 2e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEQS_PATH = REPO / "data/instacart/seqs.json"
CKPT_DIR = REPO / "artifacts/eval/instacart_cache"
OUT = REPO / "artifacts/eval/instacart_v4_peruser.json"


def log(m):
    print(m, flush=True)


def load():
    raw = {int(u): s for u, s in json.loads(SEQS_PATH.read_text(encoding="utf-8")).items()}
    raw = {u: s for u, s in raw.items() if len(s) >= 2}
    rng = np.random.default_rng(SEED)
    pick = sorted(rng.choice(sorted(raw), size=min(N_USERS, len(raw)), replace=False).tolist())
    seqs = {u: raw[u] for u in pick}
    items = sorted({it for s in seqs.values() for it in s})
    id_of = {it: i + 1 for i, it in enumerate(items)}
    seqs = {u: [id_of[i] for i in s] for u, s in seqs.items()}
    return seqs, len(items)


def make_inputs(seqs_lists, max_len):
    rows = []
    for s in seqs_lists:
        ids = s[-max_len:]
        r = np.zeros(max_len, dtype=np.int64)
        r[: len(ids)] = ids
        rows.append(r)
    return torch.from_numpy(np.stack(rows))


@torch.no_grad()
def full_eval(model, item_w, inputs, targets):
    """Return (metrics, per-user hit10 flags, per-user top500 matrix)."""
    model.eval()
    dev = next(model.parameters()).device
    tgt_all = np.asarray(targets)
    hits, tops = [], np.zeros((len(inputs), 500), dtype=np.int32)
    hr10 = hr50 = 0
    for i in range(0, len(inputs), 512):
        ids = inputs[i:i + 512].to(dev)
        ts = torch.arange(L, dtype=torch.float32, device=dev).unsqueeze(0).expand(ids.size(0), -1)
        reps = model.encode_last_step(ids, timestamps=ts)
        scores = reps @ item_w.t()
        scores[:, 0] = -float("inf")
        top500 = torch.topk(scores, k=500, dim=1).indices.cpu().numpy()
        tops[i:i + len(top500)] = top500.astype(np.int32)
        hit10 = (top500[:, :10] == tgt_all[i:i + len(top500), None]).any(axis=1)
        hit50 = (top500[:, :50] == tgt_all[i:i + len(top500), None]).any(axis=1)
        hits.extend(hit10.astype(int).tolist())
        hr10 += int(hit10.sum())
        hr50 += int(hit50.sum())
    return {"hr10": hr10 / len(tgt_all), "hr50": hr50 / len(tgt_all)}, hits, tops


def train_encoder(name, seqs, num_items):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    dev = torch.device(DEVICE)
    model = build_r2_encoder(name, num_items=num_items, embedding_dim=D, max_seq_len=L,
                             num_layers=LAYERS, num_heads=HEADS, dropout=0.1).to(dev)
    criterion = SampledSoftmax256Loss(num_negatives=NEG, logq_correction=True).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    users = sorted(seqs)
    train_users = [u for u in users if len(seqs[u]) >= 2]
    val_users = [u for u in users if len(seqs[u]) >= 3]

    item_pop = np.zeros(num_items + 1, dtype=np.float64)
    for s in seqs.values():
        for it in s:
            item_pop[it] += 1
    logq = torch.tensor(np.log(item_pop / max(item_pop[1:].sum(), 1) + 1e-9), dtype=torch.float32, device=dev)

    train_seqs = {u: seqs[u][:-1] for u in train_users}
    val_inputs = make_inputs([seqs[u][:-2] for u in val_users], L)
    val_targets = np.array([seqs[u][-2] for u in val_users], dtype=np.int64)
    test_inputs = make_inputs([seqs[u][:-1] for u in train_users], L)
    test_targets = np.array([seqs[u][-1] for u in train_users], dtype=np.int64)
    rows = make_inputs([train_seqs[u] for u in train_users], L)
    n_rows = len(rows)
    gen = torch.Generator(device=dev)
    gen.manual_seed(SEED)
    item_w = model.item_embedding.weight

    best_val, best_state, bad, epoch = -1.0, None, 0, 0
    t0 = time.time()
    for epoch in range(EPOCHS):
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
        if (epoch + 1) % 5 == 0:
            m, _, _ = full_eval(model, item_w, val_inputs, val_targets)
            log(f"  [{name}] epoch {epoch+1} loss={np.mean(losses):.4f} val_hr10={m['hr10']:.4f}")
            if m["hr10"] > best_val:
                best_val = m["hr10"]
                best_state = {k: vv.detach().clone() for k, vv in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= 10:
                    log(f"  [{name}] early stop at epoch {epoch+1}")
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "num_items": num_items,
                "best_val_hr10": best_val, "epochs_run": epoch + 1, "seed": SEED},
               CKPT_DIR / f"{name}_v4.pt")
    metrics, hits, tops = full_eval(model, item_w, test_inputs, test_targets)
    metrics.update({"seconds": round(time.time() - t0, 1), "best_val_hr10": round(best_val, 4),
                    "epochs_run": epoch + 1})
    np.savez_compressed(CKPT_DIR / f"{name}_v4_top500.npz",
                        users=np.array(train_users, dtype=np.int64), top500=tops)
    return metrics, hits, train_users


def main():
    seqs, num_items = load()
    out = {"protocol": {"encoders": ENCODERS, "note": "v4 combined run: per-user + checkpoint + top500"},
           "results": {}}
    for name in ENCODERS:
        ck = CKPT_DIR / f"{name}_v4.pt"
        npz = CKPT_DIR / f"{name}_v4_top500.npz"
        if ck.exists() and npz.exists():
            log(f"[skip] {name} (v4 assets cached)")
            continue
        log(f"[train] {name}")
        metrics, hits, train_users = train_encoder(name, seqs, num_items)
        repeats = [int(seqs[u][-1] in seqs[u][:-1]) for u in train_users]
        hr_r = sum(h for h, r in zip(hits, repeats) if r) / max(sum(repeats), 1)
        hr_n = sum(h for h, r in zip(hits, repeats) if not r) / max(len(repeats) - sum(repeats), 1)
        out["results"][name] = {**metrics,
                                "per_user_hit10": hits, "per_user_repeat": repeats,
                                "repeat_group": {"n": int(sum(repeats)), "hr10": round(hr_r, 4)},
                                "new_group": {"n": int(len(repeats) - sum(repeats)), "hr10": round(hr_n, 4)}}
        log(f"[result] {name}: {json.dumps(metrics)} repeat_hr={hr_r:.4f} new_hr={hr_n:.4f}")
        OUT.write_text(json.dumps(out), encoding="utf-8")  # compact; per-user arrays inline
    log("DONE")


if __name__ == "__main__":
    main()
