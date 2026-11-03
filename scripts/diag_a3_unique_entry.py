"""GPT-required zero-cost statistic: unique_new_entered_final_top500.

Of the 836 SASRec-unique New targets (in SASRec Top500, in NO five-source Top500),
how many actually enter the A3 fused RRF Top500? Connects SOURCE and FUSION layers.
Re-runs the frozen fusion identically (RRF k=60) and counts.
"""
from __future__ import annotations

import json
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
OUT = REPO / "artifacts/eval/hybrid_v1_a3_unique_entry.json"


def rrf_fuse(source_lists):
    score = {}
    for _n, lst in source_lists.items():
        for r, it in enumerate(lst, start=1):
            score[it] = score.get(it, 0.0) + 1.0 / (RRF_K + r)
    return sorted(score.keys(), key=lambda i: (-score[i], i))


def main():
    raw = {int(u): s for u, s in json.loads((REPO / "data/instacart/seqs.json").read_text(encoding="utf-8")).items()}
    raw = {u: s for u, s in raw.items() if len(s) >= 2}
    rng = np.random.default_rng(SEED)
    pick = sorted(rng.choice(sorted(raw), size=N_USERS, replace=False).tolist())
    seqs = {u: raw[u] for u in pick}
    items = sorted({it for s in seqs.values() for it in s})
    id_of = {it: i + 1 for i, it in enumerate(items)}
    seqs = {u: [id_of[i] for i in s] for u, s in seqs.items()}
    users = sorted(u for u in seqs if len(seqs[u]) >= 3)

    sas = np.load(SASREC_TOP)
    sas_idx = {int(u): i for i, u in enumerate(sas["users"])}

    pop = np.zeros(len(items) + 1)
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
                           (pd.factorize(pd.Series(rows))[0], cols)), shape=(len(users), len(items) + 1), copy=True)
    M[:, 0] = 0
    svd = TruncatedSVD(n_components=64, random_state=SEED)
    U = svd.fit_transform(M)
    V = svd.components_.T
    uidx = {u: i for i, u in enumerate(users)}

    five = ["FREQ", "RECENCY", "POPULAR", "ITEM_CF", "SVD"]
    stats = {"unique_new": 0, "entered_A3_top500": 0, "entered_A3_top100": 0,
             "rank_in_sasrec_of_entered": []}
    for u in users:
        s = seqs[u]
        hist, tgt = s[:-1], s[-1]
        if tgt in hist:
            continue
        cnt, last_pos = {}, {}
        for pos, it in enumerate(hist):
            cnt[it] = cnt.get(it, 0) + 1
            last_pos[it] = pos
        lists = {
            "FREQ": sorted(cnt, key=lambda k: (-cnt[k], k))[:TOP],
            "RECENCY": sorted(last_pos, key=lambda k: (-last_pos[k], k))[:TOP],
            "POPULAR": pop_order,
            "ITEM_CF": [int(x) for x in np.argsort(-C[np.array(hist[-10:], dtype=np.int64)].sum(axis=0), kind="stable") if x != 0][:TOP],
            "SVD": [int(x) for x in np.argsort(-(U[uidx[u]] @ V.T), kind="stable") if x != 0][:TOP],
        }
        sa_o = [int(x) for x in sas["top500"][sas_idx[u]] if x != 0][:TOP]
        if tgt in sa_o and not any(tgt in lists[x] for x in five):
            stats["unique_new"] += 1
            fused = rrf_fuse({**lists, "SASREC": sa_o})
            if tgt in fused[:TOP]:
                stats["entered_A3_top500"] += 1
                stats["rank_in_sasrec_of_entered"].append(sa_o.index(tgt) + 1)
            if tgt in fused[:100]:
                stats["entered_A3_top100"] += 1

    stats.pop("rank_in_sasrec_of_entered", None)
    stats["entry_rate_top500"] = round(stats["entered_A3_top500"] / max(stats["unique_new"], 1), 4)
    stats["entry_rate_top100"] = round(stats["entered_A3_top100"] / max(stats["unique_new"], 1), 4)
    OUT.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
