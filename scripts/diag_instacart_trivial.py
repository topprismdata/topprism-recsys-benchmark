"""Cheap diagnostic for the LGBM 0.0003 anomaly: trivial repeat baselines.

Same protocol data path as run_instacart_attention.py (seed-42 20k users, LOO full ranking,
no seen-masking). Baselines:
  A last_item  — rank by recency of last purchase (most recent first)
  B freq_item  — rank by user's historical frequency (ties: item id)
  C pop_global — rank by global popularity (train stats)
If these are also ~1e-4, the LGBM number is a task/catalog-difficulty reading, not a bug.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SEED = 42
N_USERS = 20000
SEQS_PATH = REPO / "data/instacart/seqs.json"


def main() -> None:
    raw = {int(u): s for u, s in json.loads(SEQS_PATH.read_text(encoding="utf-8")).items()}
    raw = {u: s for u, s in raw.items() if len(s) >= 2}
    rng = np.random.default_rng(SEED)
    pick = sorted(rng.choice(sorted(raw), size=min(N_USERS, len(raw)), replace=False).tolist())
    seqs = {u: raw[u] for u in pick}

    pop = {}
    for u in pick:
        for it in seqs[u][:-2]:
            pop[it] = pop.get(it, 0) + 1

    res = {}
    for name in ["last_item", "freq_item", "pop_global"]:
        hr10 = hr50 = ndcg = 0
        n = 0
        for u in pick:
            s = seqs[u]
            hist, tgt = s[:-1], s[-1]
            if name == "last_item":
                seen = {}
                for pos, it in enumerate(hist):
                    seen[it] = pos  # last position wins
                cand = sorted(seen.keys(), key=lambda k: (-seen[k], k))
            elif name == "freq_item":
                cnt = {}
                for it in hist:
                    cnt[it] = cnt.get(it, 0) + 1
                cand = sorted(cnt.keys(), key=lambda k: (-cnt[k], k))
            else:
                cand = [k for k, _ in sorted(pop.items(), key=lambda kv: -kv[1])]
            top10, top50 = cand[:10], cand[:50]
            n += 1
            if tgt in top10:
                hr10 += 1
                ndcg += 1.0 / np.log2(top10.index(tgt) + 2)
            if tgt in top50:
                hr50 += 1
        res[name] = {"hr10": round(hr10 / n, 5), "hr50": round(hr50 / n, 5),
                     "ndcg10": round(ndcg / n, 5), "n": n}
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
