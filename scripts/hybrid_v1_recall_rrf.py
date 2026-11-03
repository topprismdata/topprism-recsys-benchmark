"""Hybrid v1.0 — Experiment 1 recall audit, RRF protocol (GPT-frozen 2026-10-27).

GPT code-level contract (approved FOURTH OPTION; all choices frozen):
- Sources (each native scoring, dedup, truncate Top500):
    FREQ     user purchase counts (desc)
    RECENCY  most-recent-purchase position (nearest first, full history list)
    POPULAR  global popularity from s[:-1]
    ITEM_CF  cooc score vs user's last 10 items (cooc from ALL users' s[:-1], diag 0)
    SVD      TruncatedSVD d=64 on 0/1 user-item matrix of s[:-1]; u @ v
    SASREC   reserved for A3 (v4 checkpoint; NOT in this run)
- Arms: A0 = FREQ+RECENCY+POPULAR; A1 = +ITEM_CF; A2 = +SVD.
- Fusion: RRF score(i) = sum over sources of 1/(60 + rank_s(i)); rank from 1;
  missing item contributes 0. Ties broken by product_id ascending. K=100/300/500
  are prefixes of one full fused ordering. k=60 frozen (Cormack SIGIR 2009).
- Legality: every source reads only s[:-1]; cooc/SVD statistics never see s[-1].
- Attribution layers: L1 source capability (+ unique targets), L2 fusion outcome
  (non-nested warning), L3 marginal gained/lost/net per K + unique opportunity.
- Selftests asserted at start (identical-source, empty-source, determinism,
  tie-break, unique-entry).
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
TOP = 500
RRF_K = 60
OUT = REPO / "artifacts/eval/hybrid_v1_recall_rrf.json"
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


def rrf_fuse(source_lists):
    score = {}
    for _name, lst in source_lists.items():
        for r, it in enumerate(lst, start=1):
            score[it] = score.get(it, 0.0) + 1.0 / (RRF_K + r)
    return sorted(score.keys(), key=lambda i: (-score[i], i))


def selftests():
    base = {"s1": [5, 3, 9, 1]}
    fused_a = rrf_fuse(base)
    assert rrf_fuse({"s1": [5, 3, 9, 1], "dup": [5, 3, 9, 1]})[0] == 5 and set(rrf_fuse({"s1": [5, 3, 9, 1], "dup": [5, 3, 9, 1]})) == set(fused_a)
    assert rrf_fuse({"s1": [5, 3, 9, 1], "empty": []}) == fused_a
    assert rrf_fuse(base) == rrf_fuse(base)
    assert rrf_fuse({"s1": [7], "s2": [4]})[0] == 4
    assert 42 in rrf_fuse({"s1": [5, 1], "s2": [5, 42]})
    log("[selftest] 5/5 PASS")


def main():
    t0 = time.time()
    selftests()
    seqs, num_items = load()
    users = sorted(u for u in seqs if len(seqs[u]) >= 3)
    log(f"[data] users={len(users)} items={num_items}")

    pop = np.zeros(num_items + 1, dtype=np.float64)
    for u in users:
        for it in seqs[u][:-1]:
            pop[it] += 1
    pop[0] = -1
    pop_order = [int(x) for x in np.argsort(-pop, kind="stable") if x != 0]

    if COOC_CACHE.exists():
        C = np.load(COOC_CACHE)
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

    rows, cols = [], []
    for u in users:
        for it in set(seqs[u][:-1]):
            rows.append(u)
            cols.append(it)
    M = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32),
                           (pd.factorize(pd.Series(rows))[0], cols)),
                          shape=(len(users), num_items + 1), copy=True)
    M[:, 0] = 0
    svd = TruncatedSVD(n_components=64, random_state=SEED)
    U = svd.fit_transform(M)
    V = svd.components_.T
    log(f"[svd] evr={svd.explained_variance_ratio_.sum():.3f} ({time.time()-t0:.0f}s)")

    ARMS = {
        "A0": ["FREQ", "RECENCY", "POPULAR"],
        "A1": ["FREQ", "RECENCY", "POPULAR", "ITEM_CF"],
        "A2": ["FREQ", "RECENCY", "POPULAR", "ITEM_CF", "SVD"],
    }
    SRC_NAMES = ["FREQ", "RECENCY", "POPULAR", "ITEM_CF", "SVD"]

    l1 = {s: {str(k): {"new": 0, "rep": 0} for k in KS} for s in SRC_NAMES}
    l1_unique = {s: 0 for s in SRC_NAMES}
    l2 = {a: {str(k): {"all": 0, "new": 0, "rep": 0} for k in KS} for a in ARMS}
    l3 = {
        "A1_adds_ITEM_CF": {"source_unique_opportunity": 0, **{str(k): {"gained": 0, "lost": 0} for k in KS}},
        "A2_adds_SVD": {"source_unique_opportunity": 0, **{str(k): {"gained": 0, "lost": 0} for k in KS}},
    }
    n = {"all": 0, "rep": 0, "new": 0}
    uidx = {u: i for i, u in enumerate(users)}

    for u in users:
        s = seqs[u]
        hist, tgt = s[:-1], s[-1]
        grp = "rep" if tgt in hist else "new"
        n["all"] += 1
        n[grp] += 1

        cnt, last_pos = {}, {}
        for pos, it in enumerate(hist):
            cnt[it] = cnt.get(it, 0) + 1
            last_pos[it] = pos
        freq_order = sorted(cnt, key=lambda k: (-cnt[k], k))
        rec_order = sorted(last_pos, key=lambda k: (-last_pos[k], k))
        last10 = np.array(hist[-10:], dtype=np.int64)
        cf_score = C[last10].sum(axis=0) if len(last10) else np.zeros(num_items + 1)
        cf_order = [int(x) for x in np.argsort(-cf_score, kind="stable") if x != 0]
        svd_vec = U[uidx[u]] @ V.T
        svd_order = [int(x) for x in np.argsort(-svd_vec, kind="stable") if x != 0]

        lists = {
            "FREQ": freq_order[:TOP],
            "RECENCY": rec_order[:TOP],
            "POPULAR": pop_order[:TOP],
            "ITEM_CF": cf_order[:TOP],
            "SVD": svd_order[:TOP],
        }

        # L1: per-source recall + unique targets (in this source's Top500, no other source's Top500)
        for src, lst in lists.items():
            for k in KS:
                if tgt in lst[:k]:
                    l1[src][str(k)][grp] += 1
            if tgt in lst and not any(tgt in lists[o] for o in lists if o != src):
                l1_unique[src] += 1

        # L2 + L3
        prev_hits = None
        for arm, srcs in ARMS.items():
            fused = rrf_fuse({s2: lists[s2] for s2 in srcs})
            hits = {}
            for k in KS:
                hit = tgt in fused[:k]
                hits[k] = hit
                if hit:
                    l2[arm][str(k)]["all"] += 1
                    l2[arm][str(k)][grp] += 1
            if prev_hits is not None:
                key = f"{arm}_adds_" + ("ITEM_CF" if arm == "A1" else "SVD")
                for k in KS:
                    kk = str(k)
                    if hits[k] and not prev_hits[k]:
                        l3[key][kk]["gained"] += 1
                    elif prev_hits[k] and not hits[k]:
                        l3[key][kk]["lost"] += 1
            prev_hits = hits

        # source unique opportunity for the two arm steps
        if tgt in lists["ITEM_CF"] and not any(tgt in lists[x] for x in ARMS["A0"]):
            l3["A1_adds_ITEM_CF"]["source_unique_opportunity"] += 1
        if tgt in lists["SVD"] and not any(tgt in lists[x] for x in ARMS["A1"]):
            l3["A2_adds_SVD"]["source_unique_opportunity"] += 1

    for key in l3:
        for k in KS:
            kk = str(k)
            l3[key][kk]["net"] = l3[key][kk]["gained"] - l3[key][kk]["lost"]

    res = {
        "protocol": {
            "fusion": f"RRF k={RRF_K} frozen (Cormack SIGIR 2009)", "top_per_source": TOP, "ks": KS,
            "tie_break": "product_id ascending", "stats_source": "s[:-1] (s[-1] never seen)",
            "arms": ARMS, "seed": SEED, "selftests": "5/5 PASS",
        },
        "n": n,
        "L1_source_capability": {
            s: {"recall": {k: {g: round(l1[s][k][g] / n[g], 4) for g in ["new", "rep"]} for k in l1[s]},
                "unique_targets_in_top500": l1_unique[s]} for s in SRC_NAMES
        },
        "L2_fusion_outcome": {
            a: {k: {g: round(l2[a][k][g] / n[g], 4) for g in ["all", "new", "rep"]} for k in l2[a]} for a in ARMS
        },
        "L3_marginal_contribution": l3,
        "seconds": round(time.time() - t0, 1),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    log(json.dumps(res["L2_fusion_outcome"], indent=2))
    log(json.dumps(res["L3_marginal_contribution"], indent=2))
    log("DONE")


if __name__ == "__main__":
    main()
