"""Instacart LightGBM-5Count-Diagnostic (full ranking, same protocol as neural encoders).

Same 5 pre-registered count features as ML-1M (train statistics only), but computed
STREAMING in 100-user blocks — the (20000 x 13400 x 5) dense tensor would be ~5.4GB
and the co-occurrence pass is vectorized (H^T @ H per 1000-user chunk).
Named "-Diagnostic" per GPT ruling: feature-limited, NOT a strong tabular baseline.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import lightgbm as lgb

SEED = 42
N_USERS = 20000
NEG = 200
OUT = REPO / "artifacts/eval/instacart_attention.json"
SEQS_PATH = REPO / "data/instacart/seqs.json"
MODEL_CACHE = REPO / "artifacts/eval/instacart_cache/lgbm_model.joblib"
BLOCK = 25          # prediction block (peak-memory guard; crashed at 100 with exit 5)
PRED_THREADS = 2    # LightGBM predict threads (Windows large-block guard)


def log(msg):
    print(msg, flush=True)


def load_instacart():
    raw = {int(u): s for u, s in json.loads(SEQS_PATH.read_text(encoding="utf-8")).items()}
    raw = {u: s for u, s in raw.items() if len(s) >= 2}
    rng = np.random.default_rng(SEED)
    users_all = sorted(raw)
    pick = sorted(rng.choice(users_all, size=min(N_USERS, len(users_all)), replace=False).tolist())
    seqs = {u: raw[u] for u in pick}
    items = sorted({it for s in seqs.values() for it in s})
    id_of = {it: i + 1 for i, it in enumerate(items)}
    seqs = {u: [id_of[i] for i in s] for u, s in seqs.items()}
    log(f"[data] users={len(seqs)} items={len(items)}")
    return seqs, len(items)


def dense_cooc(seqs, num_items):
    """same-user item-item co-occurrence from train portion (s[:-2]); dense float32 (cached)."""
    cache = REPO / "artifacts/eval/instacart_cache/cooc_dense.npy"
    if cache.exists():
        C = np.load(cache)
        log(f"[cooc] loaded cached dense matrix {C.shape}")
        return C
    users = sorted(seqs)
    C = np.zeros((num_items + 1, num_items + 1), dtype=np.float32)
    for start in range(0, len(users), 1000):
        chunk = users[start:start + 1000]
        H = np.zeros((len(chunk), num_items + 1), dtype=np.float32)
        for r, u in enumerate(chunk):
            for it in seqs[u][:-2]:
                H[r, it] = 1.0
        C += H.T @ H
    np.fill_diagonal(C, 0.0)
    C[0, :] = C[:, 0] = 0.0
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, C)
    log(f"[cooc] dense matrix {C.shape} nnz>0={(C>0).sum()} (cached)")
    return C


def feature_rows_block(train_seqs_block, id_list, C, pop):
    """Return (len(block), len(id_list), 5) float32 features for candidate ids."""
    U = len(train_seqs_block)
    K = len(id_list)
    f = np.zeros((U, K, 5), dtype=np.float32)
    for ui, s in enumerate(train_seqs_block):
        last10 = np.array(s[-10:], dtype=np.int64) if s else np.zeros(0, dtype=np.int64)
        c10 = C[last10].sum(axis=0) if len(last10) else np.zeros(C.shape[0], dtype=np.float32)
        w = np.exp(-np.arange(len(s), dtype=np.float32)[::-1] / 5.0) if s else np.zeros(0, dtype=np.float32)
        W = np.zeros((1, C.shape[0]), dtype=np.float32)
        for pos, it in enumerate(s):
            W[0, it] += w[pos]
        decay = W @ C
        cnt = np.zeros(C.shape[0], dtype=np.float32)
        for it in s:
            cnt[it] += 1.0
        blk = f[ui]
        blk[:, 0] = pop[id_list]
        blk[:, 1] = len(s)
        blk[:, 2] = c10[id_list]
        blk[:, 3] = decay[0, id_list]
        blk[:, 4] = cnt[id_list]
    return f


def main():
    t0 = time.time()
    seqs, num_items = load_instacart()
    users = sorted(seqs)
    pop = np.zeros(num_items + 1, dtype=np.float32)
    for u in users:
        for it in seqs[u][:-2]:
            pop[it] += 1.0
    C = dense_cooc(seqs, num_items)

    all_ids = np.arange(1, num_items + 1)
    train_users = [u for u in users if len(seqs[u]) >= 3]
    MODEL_CACHE.parent.mkdir(parents=True, exist_ok=True)
    clf = None
    if MODEL_CACHE.exists():
        import joblib
        clf = joblib.load(MODEL_CACHE)
        log("[train] loaded cached LightGBM model; skipping train-row build")

    if clf is None:
        # ---- train rows: per user, 1 positive (2nd-to-last) + NEG sampled negatives
        rng = np.random.default_rng(SEED)
        train_users = [u for u in users if len(seqs[u]) >= 3]
        X_tr = np.zeros((len(train_users) * (NEG + 1), 5), dtype=np.float32)
        y_tr = np.zeros(len(train_users) * (NEG + 1), dtype=np.int32)
        r = 0
        for ui, u in enumerate(train_users):
            pos = seqs[u][-2]
            negs = rng.integers(1, num_items + 1, size=NEG)
            negs = negs[negs != pos][:NEG]
            cands = np.concatenate([[pos], negs])
            hist = seqs[u][:-2]
            f = feature_rows_block([hist], cands, C, pop)[0]
            X_tr[r:r + len(cands)] = f
            y_tr[r] = 1
            r += len(cands)
            if ui % 2000 == 0:
                log(f"  train rows {ui}/{len(train_users)}")
        X_tr = X_tr[:r]
        y_tr = y_tr[:r]
        log(f"[train] LightGBM on {len(X_tr)} rows (pos={int(y_tr.sum())})")
        clf = lgb.LGBMClassifier(n_estimators=250, random_state=SEED, n_jobs=-1, verbose=-1)
        clf.fit(X_tr, y_tr)
        import joblib
        joblib.dump(clf, MODEL_CACHE)
        log(f"[train] done ({time.time()-t0:.0f}s), model cached; full-ranking prediction (streaming)")

    hr10 = hr50 = ndcg10 = 0
    n = 0
    for start in range(0, len(train_users), BLOCK):
        ublk = train_users[start:start + BLOCK]
        hists = [seqs[u][:-1] for u in ublk]
        F = feature_rows_block(hists, all_ids, C, pop)
        U, I, K = F.shape
        try:
            scores = clf.predict_proba(F.reshape(U * I, K), num_threads=PRED_THREADS)[:, 1]
        except TypeError:
            scores = clf.predict_proba(F.reshape(U * I, K))[:, 1]
        scores = scores.reshape(U, I).astype(np.float64)
        scores[:, 0] = -np.inf
        top50 = np.argsort(-scores, axis=1)[:, :50]
        if start == 0:
            log(f"[diag] scores std={float(scores[:, 1:].std()):.4e} uniq={len(np.unique(scores))}")
            for j, u in enumerate(ublk[:3]):
                tgt = seqs[u][-1]
                rank = int((scores[j] >= scores[j][tgt]).sum())
                log(f"[diag] user {u}: target {tgt} rank {rank}/{I}")
        for j, u in enumerate(ublk):
            tgt = seqs[u][-1]
            n += 1
            if tgt in top50[j][:10]:
                hr10 += 1
                ndcg10 += 1.0 / np.log2(int(np.flatnonzero(top50[j][:10] == tgt)[0]) + 2)
            if tgt in top50[j]:
                hr50 += 1
        if (start // BLOCK) % 20 == 0:
            log(f"  score {start}/{len(train_users)} ({time.time()-t0:.0f}s)")
    res = {"n": n, "hr10": hr10 / n, "hr50": hr50 / n, "ndcg10": ndcg10 / n,
           "seconds": round(time.time() - t0, 1), "model_name": "LightGBM-5Count-Diagnostic"}
    log(f"[lgbm] HR@10={res['hr10']:.4f} HR@50={res['hr50']:.4f} NDCG@10={res['ndcg10']:.4f}")
    out = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    out.setdefault("results", {})["lgbm_5count_diagnostic"] = res
    OUT.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log("DONE")


if __name__ == "__main__":
    main()
