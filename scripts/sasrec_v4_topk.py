"""SASRec v4 for Hybrid A3: frozen-protocol training + checkpoint + FULL-CATALOG Top500.

GPT hard constraints (2026-10-27 A3 pre-audit):
1. SASRec Top500 must be scored independently over the FULL 41,147-item catalog
   (no pre-filtering by A2 candidates).
2. Training / checkpoint selection / inference must not read any evaluated user's
   target s[-1] (val-based early stopping as in the frozen v2 protocol).
3. Sixth-source RRF interface identical to other sources (k=60, Top500, id tie-break)
   — enforced in the fusion script, not here.
4. A3 primary results report BOTH K=100 and K=500 — enforced in the fusion script.

Outputs:
  artifacts/eval/instacart_cache/sasrec_v4.pt          (checkpoint)
  artifacts/eval/instacart_cache/sasrec_v4_top500.npz  (uid -> top500 item ids, train_users order)
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
CKPT = REPO / "artifacts/eval/instacart_cache/sasrec_v4.pt"
TOPK_OUT = REPO / "artifacts/eval/instacart_cache/sasrec_v4_top500.npz"


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
def eval_hr(model, item_w, inputs, targets, ks=(10, 50)):
    model.eval()
    hr = {k: 0 for k in ks}
    n = 0
    dev = next(model.parameters()).device
    tgt_all = np.asarray(targets)
    for i in range(0, len(inputs), 512):
        ids = inputs[i:i + 512].to(dev)
        ts = torch.arange(L, dtype=torch.float32, device=dev).unsqueeze(0).expand(ids.size(0), -1)
        reps = model.encode_last_step(ids, timestamps=ts)
        scores = reps @ item_w.t()
        scores[:, 0] = -float("inf")
        top50 = torch.topk(scores, k=50, dim=1).indices.cpu().numpy()
        for k in ks:
            hit = (top50[:, :k] == tgt_all[i:i + len(top50), None]).any(axis=1)
            hr[k] += int(hit.sum())
        n += len(top50)
    return {f"hr{k}": hr[k] / n for k in ks}


def main():
    CKPT.parent.mkdir(parents=True, exist_ok=True)
    seqs, num_items = load()
    users = sorted(seqs)
    train_users = [u for u in users if len(seqs[u]) >= 2]
    val_users = [u for u in users if len(seqs[u]) >= 3]

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    dev = torch.device(DEVICE)
    model = build_r2_encoder("sasrec", num_items=num_items, embedding_dim=D, max_seq_len=L,
                             num_layers=LAYERS, num_heads=HEADS, dropout=0.1).to(dev)
    criterion = SampledSoftmax256Loss(num_negatives=NEG, logq_correction=True).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)

    item_pop = np.zeros(num_items + 1, dtype=np.float64)
    for s in seqs.values():
        for it in s:
            item_pop[it] += 1
    logq = torch.tensor(np.log(item_pop / max(item_pop[1:].sum(), 1) + 1e-9), dtype=torch.float32, device=dev)

    train_seqs = {u: seqs[u][:-1] for u in train_users}
    val_inputs = make_inputs([seqs[u][:-2] for u in val_users], L)   # never sees s[-1]
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
            v = eval_hr(model, item_w, val_inputs, val_targets, ks=(10,))
            log(f"  epoch {epoch+1} loss={np.mean(losses):.4f} val_hr10={v['hr10']:.4f}")
            if v["hr10"] > best_val:
                best_val = v["hr10"]
                best_state = {k: vv.detach().clone() for k, vv in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= 10:
                    log(f"  early stop at epoch {epoch+1}")
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save({"state_dict": model.state_dict(), "num_items": num_items,
                "best_val_hr10": best_val, "epochs_run": epoch + 1, "seed": SEED}, CKPT)
    test = eval_hr(model, item_w, test_inputs, test_targets)
    log(f"[result] test {json.dumps(test)} best_val={best_val:.4f}")

    # FULL-catalog Top500 for every train_user (constraint #1)
    model.eval()
    tops = np.zeros((len(train_users), 500), dtype=np.int32)
    with torch.no_grad():
        for i in range(0, len(test_inputs), 512):
            ids = test_inputs[i:i + 512].to(dev)
            ts = torch.arange(L, dtype=torch.float32, device=dev).unsqueeze(0).expand(ids.size(0), -1)
            reps = model.encode_last_step(ids, timestamps=ts)
            scores = reps @ item_w.t()
            scores[:, 0] = -float("inf")
            top500 = torch.topk(scores, k=500, dim=1).indices.cpu().numpy().astype(np.int32)
            tops[i:i + len(top500)] = top500
    np.savez_compressed(TOPK_OUT, users=np.array(train_users, dtype=np.int64), top500=tops)
    log(f"[done] checkpoint + top500 saved ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
