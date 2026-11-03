"""Hybrid v1.0 — Experiment 1 (recall audit, A0/A1/A2; GPT-approved 2026-10-27 design).

Pre-registered protocol (frozen before running):
- Sample: identical to §7 (5-core, 20,000 users seed 42, 41,147-item catalog).
- Statistics source for ALL retrieval arms: the prediction-input history s[:-1]
  (same information the neural encoders see for the test event); target = s[-1].
- Candidate construction (slot-filling, no fusion hyperparameters):
    A0: user-frequency ranking; slots left (history < K) filled by global popularity.
    A1: A0, then remaining slots filled by Item-CF scores
        (co-occurrence with the user's last 10 items, cooc counted over all users'
        s[:-1] histories, symmetric, diagonal zeroed).
    A2: A1, then remaining slots filled by SVD scores
        (TruncatedSVD d=64 on the 0/1 user-item matrix of s[:-1]; score = u_vec @ v_vec).
- Metrics: Candidate Recall@K for K in {100, 300, 500}, overall / Repeat-target /
  New-target groups; per-source attribution of hits (which arm's slot fill added them).
- A3 (SASRec retrieval) requires the v4 checkpoint run and is NOT in this script.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

REPO = Path(__file__).resolve().parents[1]
SEED = 42
N_USERS = 20000
KS = [100, 300, 500]
OUT = REPO / "artifacts/eval/hybrid_v1_recall_audit.json"
COOC_CACHE = REPO / "artifacts/eval/instacart_cache/cooc_s_minus1.npy"


def log(m):
    print(m, flush=True)


def load():
    raw = {int(u): s for u, s in json.loads((REPO / "data/instacart/seqs.json").read_text(encoding="utf-8")).items()}
    raw = {u: s for u, s in raw.items() if len(s) >= 2}
    rng = np.random.default_rng(SEED)
    pick = sorted(rng.choice(sorted(raw), size=min(N_USERS, len(raw)), replace=False).tolist())
    seqs = {u: raw[u] for u in pick}
    items = sorted({it for s in seqs.values() for it in s})
    id_of = {it: i + 1 for i, it in enumerate(items)}
    seqs = {u: [id_of[i] for i in s] for u, s in seqs.items()}
    return seqs, len(items)


def main():
    t0 = time.time()
    seqs, num_items = load()
    users = sorted(seqs)
    users = [u for u in users if len(seqs[u]) >= 3]
    log(f"[data] users={len(users)} items={num_items}")

    # ---- global popularity from s[:-1]
    pop = np.zeros(num_items + 1, dtype=np.float64)
    for u in users:
        for it in seqs[u][:-1]:
            pop[it] += 1
    pop[0] = -1
    pop_rank = np.argsort(-pop)  # item ids by popularity

    # ---- item-item co-occurrence from s[:-1] (cached)
    if COOC_CACHE.exists():
        C = np.load(COOC_CACHE)
        log("[cooc] loaded cache")
    else:
        C = np.zeros((num_items + 1, num_items + 1), dtype=np.float32)
        for start in range(0, len(users), 1000):
            chunk = users[start:start + 1000]
            H = np.zeros((len(chunk), num_items + 1), dtype=np.float32)
            for r, u in enumerate(chunk):
                for it in seqs[u][:-1]:
                    H[r, it] = 1.0
            C += H.T @ H
        np.fill_diagonal(C, 0.0)
        C[0, :] = C[:, 0] = 0.0
        COOC_CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.save(COOC_CACHE, C)
        log(f"[cooc] built ({time.time()-t0:.0f}s)")

    # ---- SVD on 0/1 user-item (s[:-1])
    rows, cols = [], []
    for u in users:
        for it in set(seqs[u][:-1]):
            rows.append(u)
            cols.append(it)
    M = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32),
                           (pd.factorize(pd.Series(rows))[0], cols)),
                          shape=(len(users), num_items + 1))
    M[:, 0] = 0
    log(f"[svd] fitting d=64 on {M.shape}")
    svd = TruncatedSVD(n_components=64, random_state=SEED)
    U = svd.fit_transform(M)
    V = svd.components_.T  # (items+1, 64)
    V[0] = -1e9
    log(f"[svd] done ({time.time()-t0:.0f}s), evr={svd.explained_variance_ratio_.sum():.3f}")

    # ---- per-user candidate construction (slot filling) + recall
    hits = {arm: {k: {"all": 0, "rep": 0, "new": 0} for k in KS} for arm in ["A0", "A1", "A2"]}
    src_attr = {k: {"A0": 0, "A1_fill": 0, "A2_fill": 0} for k in KS}
    n = {"all": 0, "rep": 0, "new": 0}
    uidx = {u: i for i, u in enumerate(users)}
    for u in users:
        s = seqs[u]
        hist, tgt = s[:-1], s[-1]
        rep = tgt in hist
        grp = "rep" if rep else "new"
        n["all"] += 1
        n[grp] += 1

        cnt = {}
        for it in hist:
            cnt[it] = cnt.get(it, 0) + 1
        freq_order = sorted(cnt.keys(), key=lambda k: (-cnt[k], k))

        last10 = np.array(hist[-10:], dtype=np.int64)
        cf_score = C[last10].sum(axis=0) if len(last10) else np.zeros(num_items + 1)
        cf_order = np.argsort(-cf_score)

        svd_score = U[uidx[u]] @ V.T
        svd_order = np.argsort(-svd_score)

        # Priority-chain slot filling (nested expansion, pop only tail-fills):
        #   A0: freq -> pop
        #   A1: freq -> Item-CF -> pop
        #   A2: freq -> Item-CF -> SVD -> pop
        def fill_chain(sources, budget):
            cand, seen = [], set()
            marks = []
            for si, order in enumerate(sources):
                marks.append(len(cand))
                for it in order:
                    if len(cand) >= budget:
                        break
                    if it != 0 and it not in seen:
                        cand.append(it)
                        seen.add(it)
            return cand, marks

        pop_src = pop_rank
        cf_src = [int(x) for x in cf_order]
        svd_src = [int(x) for x in svd_order]
        freq_src = [int(x) for x in freq_order]

        cand0, _ = fill_chain([freq_src, pop_src], KS[-1])
        cand1, _ = fill_chain([freq_src, cf_src, pop_src], KS[-1])
        cand2, m2 = fill_chain([freq_src, cf_src, svd_src, pop_src], KS[-1])
        # segment boundaries in cand2: [0,m2[0]) freq | [m2[0],m2[1]) CF | [m2[1],m2[2]) SVD | rest pop
        cf_end, svd_end = m2[1], m2[2]

        for arm, cand in [("A0", cand0), ("A1", cand1), ("A2", cand2)]:
            for k in KS:
                if tgt in cand[:k]:
                    hits[arm][k]["all"] += 1
                    hits[arm][k][grp] += 1
        # attribution within cand2 by segment: freq | CF-fill | SVD-fill | pop-tail
        for k in KS:
            if tgt in cand2[:k]:
                pos = cand2.index(tgt)
                if pos < cf_end:
                    src_attr[k]["A0"] += 1
                elif pos < svd_end:
                    src_attr[k]["A1_fill"] += 1
                else:
                    src_attr[k]["A2_fill"] += 1

    res = {
        "n": n,
        "recall": {arm: {str(k): {g: round(hits[arm][k][g] / n[g] if n[g] else 0, 4) for g in ["all", "rep", "new"]} for k in KS} for arm in ["A0", "A1", "A2"]},
        "attribution_final_pool": {str(k): src_attr[k] for k in KS},
        "protocol": {"stats_source": "s[:-1]", "target": "s[-1]", "slot_filling": "priority chains A0=freq>pop; A1=freq>ItemCF>pop; A2=freq>ItemCF>SVD64>pop", "seed": SEED},
        "seconds": round(time.time() - t0, 1),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    log(json.dumps(res["recall"], indent=2))
    log(json.dumps(res["attribution_final_pool"], indent=2))
    log("DONE")


if __name__ == "__main__":
    main()
