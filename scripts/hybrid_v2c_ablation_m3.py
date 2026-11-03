"""Phase 2 (Instacart feature substitution) per GPT GO 2026-10-31 — PREREG v1.1/v1.2 boundary:
frozen A3@500 pool (A0-SASRec as 6th source), frozen LightGBM config, HR@10 + Repeat/New,
ITT (training failures of neural encoders count — collapsed top500 files are used as-is).

Arms (pool always the frozen A3@500 from ORIGINAL sasrec_v4 source):
  M1    statistics + SVD                      (cached hits_M1.npz, reference)
  M2    + MeanPool rank                       (cached hits_M2.npz, reference)
  M3A0  + SASRec rank/only/flag               (cached hits_M3.npz, reference = original M3)
  M3A1  + UniformAttn rank/only/flag          (new)
  M3A2  + PosOnlyAttn rank/only/flag          (new)

Preregistered paired comparisons: M3A0-M3A2 (content-conditioning value in rerank features),
M3A0-M3A1, M3A2-M2 (positional aggregation vs unordered pooling as features), M3A0-M1.
Train rows are REPLAYED with the same rng(SEED) stream => identical candidates/negatives as
the original mtrain_rows.npz; only neural feature columns 9/11/12 differ per arm.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from hybrid_v2_m_matrix import (CACHE, SEED, TOP, build_point_assets, cand_features, cols_for,
                                load_meta, pool, user_lists)

OUT = CACHE / "instacart_ablation_m3.json"
VARIANTS = {"M3A1": "sasrec_uniform", "M3A2": "sasrec_pos_only"}


def log(m):
    print(m, flush=True)


def replay_train_rows(seqs, users, t0_var, aisle, dept, tag):
    """Deterministic replay of the original mtrain construction with variant neural columns."""
    cache = CACHE / f"mtrain_rows_{tag}.npz"
    if cache.exists():
        z = np.load(cache)
        log(f"[train:{tag}] cached rows={len(z['y'])}")
        return z["X"], z["y"]
    rng = np.random.default_rng(SEED)
    Fblocks, ys = [], []
    for u in users:
        s = seqs[u]
        tgt = s[-2]
        L0 = user_lists(u, t0_var)
        p = pool(L0, "A3")
        if tgt not in p:
            continue
        pos_j = p.index(tgt)
        negs = rng.choice(len(p), size=60, replace=False)
        negs = [j for j in negs if j != pos_j][:50]
        idxs = [pos_j] + negs
        Fblocks.append(cand_features(u, [p[j] for j in idxs], t0_var, aisle, dept, None))
        ys.extend([1] + [0] * len(negs))
    X = np.concatenate(Fblocks, axis=0)
    y = np.array(ys, dtype=np.int32)
    np.savez_compressed(cache, X=X, y=y)
    log(f"[train:{tag}] rows={len(y)} pos={int(y.sum())} cached")
    return X, y


def main():
    t0 = time.time()
    seqs, num_items, aisle, dept = load_meta()
    users = sorted(u for u in seqs if len(seqs[u]) >= 5)
    log(f"[data] users={len(users)} items={num_items}")

    # frozen pools from the ORIGINAL A0 sasrec source (identical to hybrid_v2_m_matrix)
    t1_a0 = build_point_assets(seqs, users, -1, CACHE / "sasrec_v4_top500.npz", CACHE / "mean_pool_v4_top500.npz")
    test_pools, test_targets, user_groups = {}, {}, {}
    for u in users:
        s = seqs[u]
        tgt = s[-1]
        user_groups[u] = "rep" if tgt in s[:-1] else "new"
        L1 = user_lists(u, t1_a0)
        p = pool(L1, "A3")
        test_pools[u] = p
        test_targets[u] = (tgt, tgt in p)
    del t1_a0
    log(f"[pools] frozen A3@500 rebuilt ({time.time()-t0:.0f}s)")

    results = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}

    for arm, var in VARIANTS.items():
        hp = CACHE / f"hits_{arm}.npz"
        if arm in results and hp.exists():
            log(f"[skip] {arm}")
            continue
        t0_var = build_point_assets(seqs, users, -2, CACHE / f"{var}_t0_top500.npz", CACHE / "mean_pool_t0_top500.npz")
        X_tr, y_tr = replay_train_rows(seqs, users, t0_var, aisle, dept, arm)
        del t0_var
        t1_var = build_point_assets(seqs, users, -1, CACHE / f"{var}_v4_top500.npz", CACHE / "mean_pool_v4_top500.npz")
        cols = cols_for("M3")
        clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.08, num_leaves=63,
                                 random_state=SEED, n_jobs=-1, verbose=-1)
        clf.fit(X_tr[:, cols], y_tr)
        hits = {"all": 0, "new": 0, "rep": 0}
        hit_flags = []
        for u in users:
            tgt, tgt_in = test_targets[u]
            grp = user_groups[u]
            p = test_pools[u]
            F = cand_features(u, p, t1_var, aisle, dept, None)
            scores = clf.predict_proba(F[:, cols])[:, 1]
            top10_items = [p[int(i)] for i in np.argsort(-scores)[:10]]
            hit = bool(tgt_in and tgt in top10_items)
            hit_flags.append((u, int(hit), grp))
            hits[grp] += int(hit)
            hits["all"] += int(hit)
        n = {"all": len(users), "new": sum(1 for g in user_groups.values() if g == "new"),
             "rep": sum(1 for g in user_groups.values() if g == "rep")}
        results[arm] = {"hr10": {g: round(hits[g] / n[g], 4) for g in hits},
                        "encoder": var,
                        "val_note": "top500 assets used as-is (ITT: collapsed encoders counted)"}
        np.savez(hp, u=np.array([f[0] for f in hit_flags], dtype=np.int64),
                 h=np.array([f[1] for f in hit_flags], dtype=np.int8),
                 g=np.array([f[2] for f in hit_flags], dtype="U3"))
        ck = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
        ck[arm] = results[arm]
        OUT.write_text(json.dumps(ck, indent=2), encoding="utf-8")
        log(f"[{arm}] {json.dumps(results[arm]['hr10'])} ({time.time()-t0:.0f}s)")
        del t1_var

    # paired bootstrap vs cached reference arms
    def load_hits(name):
        z = np.load(CACHE / f"hits_{name}.npz")
        return {int(u): int(h) for u, h in zip(z["u"], z["h"])}

    paired = {}
    ref = {"M1": load_hits("M1"), "M2": load_hits("M2"), "M3A0": load_hits("M3")}
    new = {arm: load_hits(arm) for arm in VARIANTS}
    for a, b in [("M3A0", "M3A2"), ("M3A0", "M3A1"), ("M3A2", "M2"), ("M3A1", "M2"), ("M3A2", "M3A1"), ("M3A0", "M1")]:
        ha, hb = (ref | new)[a], (ref | new)[b]
        us = sorted(ha)
        d = np.array([ha[u] - hb[u] for u in us], dtype=np.float64)
        rng2 = np.random.default_rng(SEED)
        boots = [rng2.choice(d, size=len(d), replace=True).mean() for _ in range(2000)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        paired[f"{a}_vs_{b}"] = {"mean_diff": round(float(d.mean()), 5), "ci95": [round(lo, 5), round(hi, 5)]}
    ck = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    ck["paired_bootstrap"] = paired
    OUT.write_text(json.dumps(ck, indent=2), encoding="utf-8")
    log(json.dumps(paired, indent=2))
    log("PHASE2-M3 DONE")


if __name__ == "__main__":
    main()
