"""ML-1M LightGBM baseline (full ranking, same protocol as neural encoders).

Pre-registered feature set (statistics computed from train events only — each user's
last (test) and second-to-last (validation) events excluded):
  f1 item_pop       — global event count of the candidate item
  f2 user_active    — user's event count
  f3 cooc10         — co-occurrence sum over the user's last 10 items
  f4 cooc_decay     — position-decay weighted co-occurrence (exp(-dist/5))
  f5 repeat_count   — how many times the candidate appears in the user's history
Training: validation task (target = second-to-last event), 1 positive + 200 sampled
negatives per user, LightGBM classifier 250 trees seed 42 (frozen B1 budget).
Prediction: full ranking over all items for every user (6040 x items), HR@10/50, NDCG@10.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import lightgbm as lgb

SEED = 42
OUT = REPO / "artifacts/eval/ml1m_attention.json"
NEG = 200


def log(msg):
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
    seqs = {int(u): [int(x) for x in g["item_id"]] for u, g in df.groupby("user_id", sort=False)}
    items = sorted(df["item_id"].unique())
    id_of = {it: i + 1 for i, it in enumerate(items)}
    return seqs, len(items), id_of


def cooc_matrix(seqs, id_of, num_items, train_cut=True):
    """item-item co-occurrence (same-user, all pairs) from train portion of each sequence."""
    from scipy.sparse import csr_matrix
    rows, cols = [], []
    for s in seqs.values():
        ids = [id_of[i] for i in (s[:-2] if train_cut else s)]
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                rows.append(ids[a]); cols.append(ids[b])
                rows.append(ids[b]); cols.append(ids[a])
    data = np.ones(len(rows), dtype=np.float32)
    return csr_matrix((data, (rows, cols)), shape=(num_items + 1, num_items + 1))


def build_feature_block(user_ids, seqs, id_of, num_items, cooc, pop):
    """Return (N_users, num_items+1, 5) float32 feature tensor (item 0 masked later)."""
    U = len(user_ids)
    f = np.zeros((U, num_items + 1, 5), dtype=np.float32)
    f[:, :, 0] = pop[None, :]
    for ui, u in enumerate(user_ids):
        s = [id_of[i] for i in seqs[u]]
        f[ui, :, 1] = len(s)
        last10 = np.array(s[-10:], dtype=np.int64)
        c10 = np.asarray(cooc[last10].sum(axis=0)).ravel()
        f[ui, :, 2] = c10
        weights = np.exp(-np.arange(len(s), dtype=np.float32)[::-1] / 5.0)
        decay = np.zeros(num_items + 1, dtype=np.float32)
        for pos, it in enumerate(s):
            decay += weights[pos] * np.asarray(cooc[it].todense()).ravel()
        f[ui, :, 3] = decay
        cnt = np.zeros(num_items + 1, dtype=np.float32)
        for it in s:
            cnt[it] += 1
        f[ui, :, 4] = cnt
        if ui % 1000 == 0:
            log(f"  features {ui}/{U}")
    return f


def main():
    t0 = time.time()
    seqs, num_items, id_of = load_ml1m()
    users = sorted(seqs)
    pop = np.zeros(num_items + 1, dtype=np.float32)
    for s in seqs.values():
        for it in s[:-2]:
            pop[id_of[it]] += 1
    log("[build] co-occurrence matrix")
    cooc = cooc_matrix(seqs, id_of, num_items)
    cache_tr = REPO / "artifacts/eval/ml1m_cache/lgbm_F_train.npy"
    cache_te = REPO / "artifacts/eval/ml1m_cache/lgbm_F_test.npy"
    if cache_tr.exists() and cache_te.exists():
        F_train = np.load(cache_tr)
        F_test = np.load(cache_te)
        log("[build] loaded cached feature blocks")
    else:
        log("[build] feature blocks (train stats only)")
        F_train = build_feature_block(users, {u: seqs[u][:-2] for u in users}, id_of, num_items, cooc, pop)
        F_test = build_feature_block(users, {u: seqs[u][:-1] for u in users}, id_of, num_items, cooc, pop)
        np.save(cache_tr, F_train)
        np.save(cache_te, F_test)

    rng = np.random.default_rng(SEED)
    X_tr, y_tr = [], []
    for ui, u in enumerate(users):
        s = seqs[u]
        if len(s) < 3:
            continue
        pos = id_of[s[-2]]
        negs = rng.integers(1, num_items + 1, size=NEG)
        negs = negs[negs != pos]
        for it in [pos] + list(negs):
            X_tr.append(F_train[ui, it])
            y_tr.append(1 if it == pos else 0)
    X_tr = np.stack(X_tr)
    y_tr = np.array(y_tr)
    log(f"[train] LightGBM on {len(X_tr)} rows (pos={y_tr.sum()})")
    clf = lgb.LGBMClassifier(n_estimators=250, random_state=SEED, n_jobs=-1, verbose=-1)
    clf.fit(X_tr, y_tr)
    log(f"[train] done ({time.time()-t0:.0f}s); predicting full ranking")

    hr10 = hr50 = ndcg10 = 0
    n = 0
    B = 100
    for start in range(0, len(users), B):
        blk = F_test[start:start + B]
        U, I, K = blk.shape
        scores = clf.predict_proba(blk.reshape(U * I, K), num_threads=1)[:, 1].reshape(U, I).astype(np.float64)
        scores[:, 0] = -np.inf
        top50 = np.argsort(-scores, axis=1)[:, :50]
        for j, u in enumerate(users[start:start + B]):
            tgt = id_of[seqs[u][-1]]
            n += 1
            if tgt in top50[j][:10]:
                hr10 += 1
                ndcg10 += 1.0 / np.log2(int(np.flatnonzero(top50[j][:10] == tgt)[0]) + 2)
            if tgt in top50[j]:
                hr50 += 1
    res = {"n": n, "hr10": hr10 / n, "hr50": hr50 / n, "ndcg10": ndcg10 / n, "seconds": round(time.time() - t0, 1)}
    log(f"[lgbm] HR@10={res['hr10']:.4f} HR@50={res['hr50']:.4f} NDCG@10={res['ndcg10']:.4f}")
    out = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    out.setdefault("results_v2", {})["lgbm_full_rank"] = res
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log("DONE")


if __name__ == "__main__":
    main()
