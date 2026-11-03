"""Hybrid v1.0 — Experiment 1 A3 stage: six-source RRF + negative control.

GPT-frozen contract (2026-10-27 A3 pre-audit):
- A3 arm   = FREQ + RECENCY + POPULAR + ITEM_CF + SVD + SASREC
- NEGCTRL  = same five statistical sources + SVD DUPLICATE as sixth source
  (checks whether any A3 gain is mere extra-vote mass rather than SASRec-specific info)
- SASREC Top500 comes from run_instacart_v4_combined.py (full-catalog, val-selected).
- Report per K in {100, 300, 500}, primary = K=100 (capacity risk) and K=500 (discovery):
    * L2 fusion: overall / New / Repeat Recall for A0/A1/A2/A3/NEGCTRL
    * L3 marginal A3-vs-A2 and NEGCTRL-vs-A2: net_new_hits, net_repeat_hits,
      net_total_hits (assert total == new + repeat)
    * SASRec source capability: unique New targets (in SASREC Top500, in NO five-source
      Top500) with dual denominators (all 11159; uncovered 6055-class residual),
      SASRec standalone New/Repeat Recall@K, overlap with each source
    * Six-source union New coverage (vs five-source 45.75%)
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
COOC_CACHE = REPO / "artifacts/eval/instacart_cache/cooc_s_minus1.npy"
SASREC_TOP = REPO / "artifacts/eval/instacart_cache/sasrec_v4_top500.npz"
OUT = REPO / "artifacts/eval/hybrid_v1_a3_rrf.json"


def log(m):
    print(m, flush=True)


def rrf_fuse(source_lists):
    score = {}
    for _name, lst in source_lists.items():
        for r, it in enumerate(lst, start=1):
            score[it] = score.get(it, 0.0) + 1.0 / (RRF_K + r)
    return sorted(score.keys(), key=lambda i: (-score[i], i))


def main():
    t0 = time.time()
    raw = {int(u): s for u, s in json.loads((REPO / "data/instacart/seqs.json").read_text(encoding="utf-8")).items()}
    raw = {u: s for u, s in raw.items() if len(s) >= 2}
    rng = np.random.default_rng(SEED)
    pick = sorted(rng.choice(sorted(raw), size=min(N_USERS, len(raw)), replace=False).tolist())
    seqs = {u: raw[u] for u in pick}
    items = sorted({it for s in seqs.values() for it in s})
    id_of = {it: i + 1 for i, it in enumerate(items)}
    seqs = {u: [id_of[i] for i in s] for u, s in seqs.items()}
    users = sorted(u for u in seqs if len(seqs[u]) >= 3)
    log(f"[data] users={len(users)} items={len(items)}")

    sas_npz = np.load(SASREC_TOP)
    sas_users = list(sas_npz["users"])
    sas_idx = {int(u): i for i, u in enumerate(sas_users)}
    assert set(users) <= set(sas_users), "v4 top500 user set mismatch"

    pop = np.zeros(len(items) + 1, dtype=np.float64)
    for u in users:
        for it in seqs[u][:-1]:
            pop[it] += 1
    pop[0] = -1
    pop_order = [int(x) for x in np.argsort(-pop, kind="stable") if x != 0][:TOP]
    C = np.load(COOC_CACHE)
    rows, cols = [], []
    for u in users:
        for it in set(seqs[u][:-1]):
            rows.append(u)
            cols.append(it)
    M = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32),
                           (pd.factorize(pd.Series(rows))[0], cols)),
                          shape=(len(users), len(items) + 1), copy=True)
    M[:, 0] = 0
    svd = TruncatedSVD(n_components=64, random_state=SEED)
    U = svd.fit_transform(M)
    V = svd.components_.T
    uidx = {u: i for i, u in enumerate(users)}

    ARMS = {
        "A2": ["FREQ", "RECENCY", "POPULAR", "ITEM_CF", "SVD"],
        "A3": ["FREQ", "RECENCY", "POPULAR", "ITEM_CF", "SVD", "SASREC"],
        "NEGCTRL": ["FREQ", "RECENCY", "POPULAR", "ITEM_CF", "SVD", "SVD_DUP"],
    }

    l2 = {a: {str(k): {"all": 0, "new": 0, "rep": 0} for k in KS} for a in ARMS}
    l3 = {step: {str(k): {"gained": {"new": 0, "rep": 0, "all": 0},
                          "lost": {"new": 0, "rep": 0, "all": 0}} for k in KS}
          for step in ["A3_vs_A2", "NEGCTRL_vs_A2"]}
    sas_cap = {"unique_new_targets": 0, "standalone": {str(k): {"new": 0, "rep": 0} for k in KS},
               "overlap_with": {s: 0 for s in ["FREQ", "RECENCY", "POPULAR", "ITEM_CF", "SVD"]}}
    six_union_new = 0
    n = {"all": 0, "rep": 0, "new": 0}
    per_user_sas_unique = []

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
        freq_o = sorted(cnt, key=lambda k: (-cnt[k], k))[:TOP]
        rec_o = sorted(last_pos, key=lambda k: (-last_pos[k], k))[:TOP]
        cf_sc = C[np.array(hist[-10:], dtype=np.int64)].sum(axis=0)
        cf_o = [int(x) for x in np.argsort(-cf_sc, kind="stable") if x != 0][:TOP]
        sv_o = [int(x) for x in np.argsort(-(U[uidx[u]] @ V.T), kind="stable") if x != 0][:TOP]
        sa_o = [int(x) for x in sas_npz["top500"][sas_idx[u]] if x != 0][:TOP]

        lists = {"FREQ": freq_o, "RECENCY": rec_o, "POPULAR": pop_order,
                 "ITEM_CF": cf_o, "SVD": sv_o, "SVD_DUP": sv_o, "SASREC": sa_o}
        five = ["FREQ", "RECENCY", "POPULAR", "ITEM_CF", "SVD"]

        # SASRec capability
        for k in KS:
            if tgt in sa_o[:k]:
                sas_cap["standalone"][str(k)][grp] += 1
        if tgt in sa_o:
            for src in five:
                if tgt in lists[src]:
                    sas_cap["overlap_with"][src] += 1
            if not any(tgt in lists[src] for src in five):
                sas_cap["unique_new_targets"] += 1 if grp == "new" else 0
                per_user_sas_unique.append(int(grp == "new"))
        # six-source union coverage
        six_union = set().union(*[set(lists[x]) for x in ARMS["A3"]])
        if grp == "new" and tgt in six_union:
            six_union_new += 1

        arm_hits = {}
        for arm, srcs in ARMS.items():
            fused = rrf_fuse({x: lists[x] for x in srcs})
            hits = {}
            for k in KS:
                hit = tgt in fused[:k]
                hits[k] = hit
                if hit:
                    l2[arm][str(k)]["all"] += 1
                    l2[arm][str(k)][grp] += 1
            arm_hits[arm] = hits
        for step, arm in [("A3_vs_A2", "A3"), ("NEGCTRL_vs_A2", "NEGCTRL")]:
            for k in KS:
                kk = str(k)
                a3h, a2h = arm_hits[arm][k], arm_hits["A2"][k]
                if a3h and not a2h:
                    l3[step][kk]["gained"][grp] += 1
                    l3[step][kk]["gained"]["all"] += 1
                elif a2h and not a3h:
                    l3[step][kk]["lost"][grp] += 1
                    l3[step][kk]["lost"]["all"] += 1

    # net three-caliber
    for step in l3:
        for k in KS:
            kk = str(k)
            g, l = l3[step][kk]["gained"], l3[step][kk]["lost"]
            l3[step][kk]["net"] = {"new": g["new"] - l["new"], "rep": g["rep"] - l["rep"],
                                   "total": g["all"] - l["all"]}
            assert l3[step][kk]["net"]["total"] == l3[step][kk]["net"]["new"] + l3[step][kk]["net"]["rep"]

    res = {
        "protocol": {"fusion": f"RRF k={RRF_K}", "top": TOP, "ks": KS, "arms": ARMS,
                     "sasrec_source": "v4 full-catalog val-selected", "seed": SEED},
        "n": n,
        "L2_fusion": {a: {k: {g: round(l2[a][k][g] / n[g], 4) for g in ["all", "new", "rep"]} for k in l2[a]} for a in ARMS},
        "L3_marginal": l3,
        "SASREC_capability": {
            "unique_new_targets": sas_cap["unique_new_targets"],
            "frac_of_all_new": round(sas_cap["unique_new_targets"] / n["new"], 4),
            "frac_of_five_source_uncovered_new": round(sas_cap["unique_new_targets"] / max(n["new"] - 5104, 1), 4),
            "standalone_recall": {k: {g: round(sas_cap["standalone"][k][g] / n[g], 4) for g in ["new", "rep"]} for k in sas_cap["standalone"]},
            "target_overlap_with_sources": sas_cap["overlap_with"],
        },
        "six_source_union_new_coverage": round(six_union_new / n["new"], 4),
        "five_source_union_new_coverage_reference": 0.4575,
        "seconds": round(time.time() - t0, 1),
    }
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    log(json.dumps(res["L2_fusion"], indent=2))
    log(json.dumps(res["L3_marginal"], indent=2))
    log(json.dumps(res["SASREC_capability"], indent=2))
    log("DONE")


if __name__ == "__main__":
    main()
