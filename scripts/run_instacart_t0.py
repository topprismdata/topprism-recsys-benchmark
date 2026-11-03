"""Experiment-2 t0-point assets: SASRec & MeanPool trained at the t0 cutoff.

GPT t0 time contract (frozen 2026-10-29):
- The reranker TRAIN task predicts s[-2] with visible history s[:-2].
- t0 models must NEVER see s[-2] or s[-1] of any user:
    * train next-item pairs live INSIDE s[:-3] (input s[:-3], targets shifted),
    * model selection (early stopping) uses s[:-4] -> s[-3] validation,
    * feature inference for the reranker uses the full visible history s[:-2].
- Outputs mirror v4: checkpoints + per-user FULL-catalog Top500 at t0
  (sasrec_t0 / meanpool_t0) for reranker-training features and t0 candidate pools.
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
ENCODERS = ["sasrec", "mean_pool", "sasrec_uniform", "sasrec_pos_only"]  # +ablation arms (phase 2, GPT GO 2026-10-31)
SEQS_PATH = REPO / "data/instacart/seqs.json"
CKPT_DIR = REPO / "artifacts/eval/instacart_cache"


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
def eval_hr10(model, item_w, inputs, targets):
    model.eval()
    dev = next(model.parameters()).device
    tgt_all = np.asarray(targets)
    hr10 = n = 0
    for i in range(0, len(inputs), 512):
        ids = inputs[i:i + 512].to(dev)
        ts = torch.arange(L, dtype=torch.float32, device=dev).unsqueeze(0).expand(ids.size(0), -1)
        reps = model.encode_last_step(ids, timestamps=ts)
        scores = reps @ item_w.t()
        scores[:, 0] = -float("inf")
        top10 = torch.topk(scores, k=10, dim=1).indices.cpu().numpy()
        hr10 += int((top10 == tgt_all[i:i + len(top10), None]).any(axis=1).sum())
        n += len(top10)
    return hr10 / max(n, 1)


@torch.no_grad()
def top500_all(model, item_w, inputs):
    model.eval()
    dev = next(model.parameters()).device
    tops = np.zeros((len(inputs), 500), dtype=np.int32)
    for i in range(0, len(inputs), 512):
        ids = inputs[i:i + 512].to(dev)
        ts = torch.arange(L, dtype=torch.float32, device=dev).unsqueeze(0).expand(ids.size(0), -1)
        reps = model.encode_last_step(ids, timestamps=ts)
        scores = reps @ item_w.t()
        scores[:, 0] = -float("inf")
        tops[i:i + 512] = torch.topk(scores, k=500, dim=1).indices.cpu().numpy().astype(np.int32)
    return tops


def train_t0(name, seqs, num_items):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    dev = torch.device(DEVICE)
    model = build_r2_encoder(name, num_items=num_items, embedding_dim=D, max_seq_len=L,
                             num_layers=LAYERS, num_heads=HEADS, dropout=0.1).to(dev)
    criterion = SampledSoftmax256Loss(num_negatives=NEG, logq_correction=True).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    users = sorted(seqs)
    # t0 population: users with enough history to train/validate at t0
    t0_users = [u for u in users if len(seqs[u]) >= 5]          # s[:-2] has >=3 for train pairs +1 val
    train_users = [u for u in t0_users if len(seqs[u]) >= 4]
    val_users = [u for u in t0_users if len(seqs[u]) >= 5]

    item_pop = np.zeros(num_items + 1, dtype=np.float64)
    for u in t0_users:
        for it in seqs[u][:-3]:                                  # train-internal stats only
            item_pop[it] += 1
    logq = torch.tensor(np.log(item_pop / max(item_pop[1:].sum(), 1) + 1e-9), dtype=torch.float32, device=dev)

    # TRAIN: next-item pairs inside s[:-3]; FINAL position of s[:-3] predicts nothing (no s[-2] leak)
    train_seqs = {u: seqs[u][:-3] for u in train_users}
    # VAL: input s[:-3] minus last -> predict s[-3]  (i.e. second-to-last of s[:-2])
    val_inputs = make_inputs([seqs[u][:-4] if len(seqs[u]) >= 5 else seqs[u][:-3] for u in val_users], L)
    val_targets = np.array([seqs[u][-4] if len(seqs[u]) >= 5 else seqs[u][-3] for u in val_users], dtype=np.int64)
    rows = make_inputs([train_seqs[u] for u in train_users], L)
    n_rows = len(rows)
    gen = torch.Generator(device=dev)
    gen.manual_seed(SEED)
    item_w = model.item_embedding.weight

    best_val, best_state, bad, epoch = -1.0, None, 0, 0
    t0c = time.time()
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
            v = eval_hr10(model, item_w, val_inputs, val_targets)
            log(f"  [{name}@t0] epoch {epoch+1} loss={np.mean(losses):.4f} val_hr10={v:.4f}")
            if v > best_val:
                best_val, bad = v, 0
                best_state = {k: vv.detach().clone() for k, vv in model.state_dict().items()}
            else:
                bad += 1
                if bad >= 10:
                    log(f"  [{name}@t0] early stop at epoch {epoch+1}")
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "num_items": num_items,
                "best_val_hr10": best_val, "epochs_run": epoch + 1, "seed": SEED},
               CKPT_DIR / f"{name}_t0.pt")
    # feature inference at t0: input = FULL visible history s[:-2]
    feat_users = [u for u in users if len(seqs[u]) >= 3]
    t0_inputs = make_inputs([seqs[u][:-2] for u in feat_users], L)
    tops = top500_all(model, item_w, t0_inputs)
    np.savez_compressed(CKPT_DIR / f"{name}_t0_top500.npz",
                        users=np.array(feat_users, dtype=np.int64), top500=tops)
    log(f"[{name}@t0] checkpoint + top500 saved ({time.time()-t0c:.0f}s), best_val={best_val:.4f}")


def main():
    seqs, num_items = load()
    log(f"[data] users={len(seqs)} items={num_items}")
    for name in ENCODERS:
        ck = CKPT_DIR / f"{name}_t0.pt"
        npz = CKPT_DIR / f"{name}_t0_top500.npz"
        if ck.exists() and npz.exists():
            log(f"[skip] {name} (t0 assets cached)")
            continue
        log(f"[train@t0] {name}")
        train_t0(name, seqs, num_items)
    log("DONE")


if __name__ == "__main__":
    main()
