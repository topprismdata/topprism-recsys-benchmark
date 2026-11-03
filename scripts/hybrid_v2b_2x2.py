"""Hybrid v2b: the GPT-required 2x2 system matrix + same-protocol freq pairing.

Cells (same test protocol: t1 assets, predict s[-1], unrecalled = zero):
  A2-M1, A2-M3  (t0 training rows rebuilt on the t0 A2 pool per GPT contract)
  A3-M1, A3-M3  (reference, re-run in the same deterministic pass)
  freq baseline per-user hits (freq top10) paired vs A3-M3.
Effects: candidate increment, fixed-pool feature increment, full-system gain,
interaction term, and A3-M3 vs freq.
"""
from __future__ import annotations

import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

import lightgbm as lgb

REPO = Path(__file__).resolve().parents[1]
SEED = 42
N_USERS = 20000
OUT = REPO / "artifacts/eval/hybrid_v2b_2x2.json"
CACHE = REPO / "artifacts/eval/instacart_cache"

sys.path.insert(0, str(REPO / "scripts"))
from hybrid_v2_m_matrix import (build_point_assets, user_lists, pool, cand_features,
                                cols_for, load_meta)


def log(m):
    print(m, flush=True)


def fit_eval(X_tr, y_tr, cols, users, test_pools, pool_arm, A1, aisle, dept,
             test_targets, user_groups):
    clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.08, num_leaves=63,
                             random_state=SEED, n_jobs=-1, verbose=-1)
    clf.fit(X_tr[:, cols], y_tr)
    hits = {"all": 0, "new": 0, "rep": 0}
    hit_flags = []
    for u in users:
        tgt = test_targets[u]
        grp = user_groups[u]
        p = test_pools[(u, pool_arm)]
        F = cand_features(u, p, A1, aisle, dept, None)
        scores = clf.predict_proba(F[:, cols])[:, 1]
        top10 = [p[int(i)] for i in np.argsort(-scores)[:10]]
        hit = tgt in top10
        hit_flags.append((u, int(hit), grp))
        hits["all"] += int(hit)
        hits[grp] += int(hit)
    n = {"all": len(users), "new": sum(1 for g in user_groups.values() if g == "new"),
         "rep": sum(1 for g in user_groups.values() if g == "rep")}
    return {g: round(hits[g] / n[g], 4) for g in hits}, hit_flags


def paired(a_flags, b_flags):
    ha = {u: h for u, h, _ in a_flags}
    hb = {u: h for u, h, _ in b_flags}
    us = sorted(ha)
    d = np.array([ha[u] - hb[u] for u in us])
    r = np.random.default_rng(SEED)
    boots = [r.choice(d, size=len(d), replace=True).mean() for _ in range(2000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"mean_diff": round(float(d.mean()), 5), "ci95": [round(lo, 5), round(hi, 5)]}


def main():
    t0 = time.time()
    seqs, num_items, aisle, dept = load_meta()
    users = sorted(u for u in seqs if len(seqs[u]) >= 5)
    log(f"[data] users={len(users)}")

    t1 = build_point_assets(seqs, users, -1, CACHE / "sasrec_v4_top500.npz", CACHE / "mean_pool_v4_top500.npz")
    t0a = build_point_assets(seqs, users, -2, CACHE / "sasrec_t0_top500.npz", CACHE / "mean_pool_t0_top500.npz")
    log(f"[assets] ready ({time.time()-t0:.0f}s)")

    PC = CACHE / "t1_pools.pkl"
    if PC.exists():
        with open(PC, "rb") as f:
            test_pools, test_targets, user_groups = pickle.load(f)
        log("[pools] loaded cached t1 pools")
    else:
        test_pools, test_targets, user_groups = {}, {}, {}
        for u in users:
            s = seqs[u]
            tgt = s[-1]
            user_groups[u] = "rep" if tgt in s[:-1] else "new"
            L1 = user_lists(u, t1)
            for arm in ["A2", "A3"]:
                test_pools[(u, arm)] = pool(L1, arm)
            test_targets[u] = tgt
        with open(PC, "wb") as f:
            pickle.dump((test_pools, test_targets, user_groups), f)
        log(f"[pools] t1 built + cached ({time.time()-t0:.0f}s)")

    TC = CACHE / "mtrain_rows_A2.npz"
    rng = np.random.default_rng(SEED)
    if TC.exists():
        z = np.load(TC)
        Xall2, y2 = z["X"], z["y"]
        log("[train] loaded cached A2 rows")
    else:
        Fb, ys = [], []
        for u in users:
            s = seqs[u]
            tgt = s[-2]
            p = pool(user_lists(u, t0a), "A2")
            if tgt not in p:
                continue
            pos_j = p.index(tgt)
            negs = rng.choice(len(p), size=60, replace=False)
            negs = [j for j in negs if j != pos_j][:50]
            Fb.append(cand_features(u, [p[j] for j in [pos_j] + negs], t0a, aisle, dept, None))
            ys.extend([1] + [0] * len(negs))
        Xall2 = np.concatenate(Fb, axis=0)
        y2 = np.array(ys, dtype=np.int32)
        np.savez_compressed(TC, X=Xall2, y=y2)
        log(f"[train] A2 rows={len(y2)} cached ({time.time()-t0:.0f}s)")

    z3 = np.load(CACHE / "mtrain_rows.npz")
    Xall3, y3 = z3["X"], z3["y"]

    res, flags = {}, {}
    for arm, m, Xa, ya in [("A2", "M1", Xall2, y2), ("A2", "M3", Xall2, y2),
                           ("A3", "M1", Xall3, y3), ("A3", "M3", Xall3, y3)]:
        key = f"{arm}-{m}"
        hr, fl = fit_eval(Xa, ya, cols_for(m), users, test_pools, arm, t1, aisle, dept,
                          test_targets, user_groups)
        res[key] = hr
        flags[key] = fl
        log(f"[{key}] {json.dumps(hr)}")

    freq_flags = []
    for u in users:
        s = seqs[u]
        hist, tgt = s[:-1], s[-1]
        cnt = {}
        for it in hist:
            cnt[it] = cnt.get(it, 0) + 1
        cand = sorted(cnt, key=lambda k: (-cnt[k], k))
        freq_flags.append((u, int(tgt in cand[:10]), user_groups[u]))
    res["freq_baseline"] = {g: round(sum(h for _, h, gg in freq_flags if gg == g) /
                                      max(sum(1 for _, _, gg in freq_flags if gg == g), 1), 4)
                            for g in ["all", "new", "rep"]}

    res["paired"] = {
        "cand_increment_A3M1_vs_A2M1": paired(flags["A3-M1"], flags["A2-M1"]),
        "feature_increment_A2M3_vs_A2M1": paired(flags["A2-M3"], flags["A2-M1"]),
        "full_system_A3M3_vs_A2M1": paired(flags["A3-M3"], flags["A2-M1"]),
        "A3M3_vs_freq": paired(flags["A3-M3"], freq_flags),
    }
    res["interaction"] = {
        "a3_feat_gain": round(res["A3-M3"]["all"] - res["A3-M1"]["all"], 4),
        "a2_feat_gain": round(res["A2-M3"]["all"] - res["A2-M1"]["all"], 4),
    }
    res["protocol"] = {"seed": SEED, "seconds": round(time.time() - t0, 1)}
    OUT.write_text(json.dumps(res, indent=2), encoding="utf-8")
    log(json.dumps(res["paired"], indent=2))
    log("DONE")


if __name__ == "__main__":
    main()
