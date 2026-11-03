"""ML-1M public-benchmark test of the attention hypothesis — v2 after implementation-check failure.

v1 postmortem (artifacts/eval/ml1m_run.log): SASRec HR@10=0.0386 vs published 0.2853.
Root causes identified: (a) 8-epoch frozen budget vs ~100 epochs in the literature;
(b) day-granularity sequences destroyed within-day order (ML-1M has second-level timestamps);
(c) no validation/early stopping. This v2 aligns the protocol to the public standard:
second-level ordering, 100 epochs, validation-HR@10 early stopping (patience 10 checks),
leave-one-out full ranking. Encoder budget stays d=128 / 2 layers / SS256 (internal
comparisons remain controlled; published numbers serve as reference band, not exact target).

Published reference band (HSTU paper README, full ranking, HR@10/NDCG@10):
SASRec .2853/.1603, GRU4Rec .2811/.1648, BERT4Rec .2843/.1537, HSTU .3097/.1720.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch
import torch.nn.functional as F

from src.models.r2_encoders import build_r2_encoder
from src.models.r2_losses import SampledSoftmax256Loss

SEED = 42
L = 200
D = 128
LAYERS = 2
HEADS = 4
NEG = 256
BATCH = 256
EPOCHS = 100
LR = 2e-3
ENCODERS = ["mean_pool", "gru", "sasrec", "hstu_aligned"]
DEVICE = "cuda"
OUT = REPO / "artifacts/eval/ml1m_attention.json"
CACHE = REPO / "artifacts/eval/ml1m_cache"


def log(msg: str) -> None:
    print(msg, flush=True)


def load_ml1m():
    df = pd.read_csv(REPO / "data/ml1m/ml-1m/ratings.dat", sep="::", engine="python",
                     names=["user_id", "item_id", "rating", "timestamp"])
    while True:
        keep_u = set(df["user_id"].value_counts()[lambda s: s >= 5].index)
        keep_i = set(df["item_id"].value_counts()[lambda s: s >= 5].index)
        f = df[df["user_id"].isin(keep_u) & df["item_id"].isin(keep_i)]
        if len(f) == len(df):
            break
        df = f
    df = df.sort_values(["user_id", "timestamp", "item_id"], kind="mergesort").reset_index(drop=True)
    seqs: Dict[int, list] = {}
    for u, g in df.groupby("user_id", sort=False):
        seqs[int(u)] = [int(x) for x in g["item_id"]]
    items = sorted(df["item_id"].unique())
    item_to_id = {it: i + 1 for i, it in enumerate(items)}  # 0 = pad
    seqs = {u: [item_to_id[i] for i in s] for u, s in seqs.items()}
    log(f"[data] users={len(seqs)} items={len(items)} events={sum(len(s) for s in seqs.values())}")
    return seqs, len(items)


def make_inputs(seqs_lists, item_to_row, max_len):
    rows, tss = [], []
    for s in seqs_lists:
        ids = [item_to_row[i] for i in s]
        ts = np.arange(len(ids), dtype=np.float32)  # position-based relative time (second-level ordering preserved)
        ids = ids[-max_len:]
        ts = ts[-max_len:]
        r = np.zeros(max_len, dtype=np.int64)
        t = np.zeros(max_len, dtype=np.float32)
        r[: len(ids)] = ids
        t[: len(ts)] = ts
        rows.append(r)
        tss.append(t)
    return torch.from_numpy(np.stack(rows)), torch.from_numpy(np.stack(tss))


def pad_at(ts_prefix, i):
    t = np.zeros(L, dtype=np.float32)
    t[: len(ts_prefix)] = ts_prefix
    return t


@torch.no_grad()
def eval_hr(model, item_w, inputs, targets, ks=(10, 50)):
    model.eval()
    ids_all, _ts_all = inputs
    hr = {k: 0 for k in ks}
    ndcg = {10: 0.0}
    n = 0
    dev = next(model.parameters()).device
    for i in range(0, len(ids_all), 512):
        ids = ids_all[i:i + 512].to(dev)
        ts = torch.arange(L, dtype=torch.float32, device=dev).unsqueeze(0).expand(ids.size(0), -1)
        reps = model.encode_last_step(ids, timestamps=ts)
        scores = reps @ item_w.t()
        scores[:, 0] = -float("inf")
        top50 = torch.topk(scores, k=50, dim=1).indices.cpu().numpy()
        for j in range(len(top50)):
            tgt = targets[i + j]
            n += 1
            if tgt in top50[j][:10]:
                hr[10] += 1
                ndcg[10] += 1.0 / np.log2(int(np.flatnonzero(top50[j][:10] == tgt)[0]) + 2)
            if tgt in top50[j]:
                hr[50] += 1
    return {**{f"hr{k}": hr[k] / n for k in ks}, "ndcg10": ndcg[10] / n, "n": n}


def train_encoder(name, seqs, num_items, id_of):
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
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

    train_seqs = {u: s[:-1] for u, s in seqs.items() if len(s) >= 2}
    val_inputs = make_inputs([s[:-2] for s in (seqs[u] for u in users) if len(s) >= 3],
                             id_of, L) if True else None
    # rebuild aligned with users list (some users may be filtered)
    train_users = [u for u in users if len(seqs[u]) >= 2]
    val_users = [u for u in users if len(seqs[u]) >= 3]
    val_inputs = make_inputs([seqs[u][:-2] for u in val_users], id_of, L)
    val_targets = np.array([id_of[seqs[u][-2]] for u in val_users], dtype=np.int64)
    test_inputs = make_inputs([seqs[u][:-1] for u in users if len(seqs[u]) >= 2], id_of, L)
    test_targets = np.array([id_of[seqs[u][-1]] for u in train_users], dtype=np.int64)

    rows, tss = make_inputs([train_seqs[u] for u in train_users], id_of, L)
    n_rows = len(rows)
    gen = torch.Generator(device=dev)
    gen.manual_seed(SEED)
    item_w = model.item_embedding.weight
    best_val, best_state, bad = -1.0, None, 0
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
            # next-item labels: position i predicts item at i+1; final position has no next item
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
            v = eval_hr(model, item_w, val_inputs, val_targets)
            log(f"  [{name}] epoch {epoch+1} loss={np.mean(losses):.4f} val_hr10={v['hr10']:.4f}")
            if v["hr10"] > best_val:
                best_val = v["hr10"]
                best_state = {k: vv.detach().clone() for k, vv in model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= 10:
                    log(f"  [{name}] early stop at epoch {epoch+1}")
                    break
    if best_state is not None:
        model.load_state_dict(best_state)
    test = eval_hr(model, model.item_embedding.weight, test_inputs, test_targets)
    test["seconds"] = round(time.time() - t0, 1)
    test["best_val_hr10"] = round(best_val, 4)
    test["epochs_run"] = epoch + 1
    return test


def main() -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    out = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    res_path = CACHE / "results_v2.json"
    results = json.loads(res_path.read_text(encoding="utf-8")) if res_path.exists() else {}
    seqs, num_items = load_ml1m()
    id_of = {it: i + 1 for i, it in enumerate(sorted({i for s in seqs.values() for i in s}))}
    out["protocol_v2"] = {"ordering": "second-level timestamp", "epochs": EPOCHS, "early_stopping": "val HR@10 patience 10 checks @5ep",
                          "budget": {"d": D, "layers": LAYERS, "heads": HEADS, "loss": "SS256+logq", "batch": BATCH, "lr": LR, "seed": SEED, "L": L},
                          "eval": "leave-one-out full ranking (no seen-masking), HR@10/50, NDCG@10"}
    for enc in ENCODERS:
        if enc in results:
            log(f"[cached] {enc}: {results[enc]}")
            continue
        log(f"[train] {enc}")
        results[enc] = train_encoder(enc, seqs, num_items, id_of)
        res_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        log(f"[{enc}] TEST HR@10={results[enc]['hr10']:.4f} NDCG@10={results[enc]['ndcg10']:.4f}")
    out["results_v2"] = results
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log("DONE")


if __name__ == "__main__":
    main()
