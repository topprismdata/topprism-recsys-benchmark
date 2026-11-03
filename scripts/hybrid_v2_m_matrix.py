"""Experiment-2 M-matrix: LightGBM rerankers on frozen candidate pools (GPT protocol 2026-10-29).

Arms (all on A3@500 t1 pool unless stated):
  M0      freq, recency_strength, cf_score, pop_rank, last_basket_affinity, aisle, department
  M1      M0 + svd_match
  M2      M1 + meanpool_match (+meanpool features)
  M3      M1 + sasrec_match, sasrec_rank, source flags (NO other attention-derived vars)
  M4      M3 + I1..I5 (pre-registered interactions)
  M4-Pool M3 skeleton with meanpool_match replacing sasrec_match in I1..I5
  2x2     {A2,A3 pools} x {M1,M3}

Time contract: reranker TRAIN predicts s[-2] with t0 assets (stats from s[:-2],
neural features from *_t0 models, t0 candidate pools rebuilt at s[:-2]);
TEST predicts s[-1] with t1 assets (v4 models, s[:-1] stats, A3@500 pool).
Unrecalled targets count as zero. Metrics: overall/New/Repeat HR@10 + NDCG@10,
U2/U3 unique-entry rates, paired bootstrap CIs vs M1 (and M3 vs M2).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

import lightgbm as lgb

REPO = Path(__file__).resolve().parents[1]
SEED = 42
N_USERS = 20000
RRF_K = 60
TOP = 500
OUT = REPO / "artifacts/eval/hybrid_v2_m_matrix.json"
CACHE = REPO / "artifacts/eval/instacart_cache"


def log(m):
    print(m, flush=True)


def rrf_fuse(source_lists):
    score = {}
    for _n, lst in source_lists.items():
        for r, it in enumerate(lst, start=1):
            score[it] = score.get(it, 0.0) + 1.0 / (RRF_K + r)
    return sorted(score.keys(), key=lambda i: (-score[i], i))


def load_meta():
    raw = {int(u): s for u, s in json.loads((REPO / "data/instacart/seqs.json").read_text(encoding="utf-8")).items()}
    raw = {u: s for u, s in raw.items() if len(s) >= 2}
    rng = np.random.default_rng(SEED)
    pick = sorted(rng.choice(sorted(raw), size=N_USERS, replace=False).tolist())
    seqs = {u: raw[u] for u in pick}
    items = sorted({it for s in seqs.values() for it in s})
    id_of = {it: i + 1 for i, it in enumerate(items)}
    seqs = {u: [id_of[i] for i in s] for u, s in seqs.items()}
    # item metadata (aisle/department)
    prods = pd.read_csv(sorted((REPO / "data/instacart/raw").rglob("products.csv"))[0])
    aisle = np.zeros(len(items) + 1, dtype=np.int32)
    dept = np.zeros(len(items) + 1, dtype=np.int32)
    pid_to_row = {p: i + 1 for i, p in enumerate(items)}
    for _, r in prods.iterrows():
        row = pid_to_row.get(int(r["product_id"]))
        if row is not None:
            aisle[row] = int(r["aisle_id"])
            dept[row] = int(r["department_id"])
    return seqs, len(items), aisle, dept


def build_point_assets(seqs, users, cut, sas_npz, mp_npz):
    """Statistics/neural rankings at one time point. cut: visible history = s[:cut]."""
    cooc_cache = CACHE / f"cooc_cut{cut}.npy"
    hist_all = {u: seqs[u][:cut] for u in users}
    num_items = max(max(s) for s in seqs.values())
    if cooc_cache.exists():
        C = np.load(cooc_cache)
    else:
        C = np.zeros((num_items + 1, num_items + 1), dtype=np.float32)
        for st in range(0, len(users), 1000):
            chunk = users[st:st + 1000]
            H = np.zeros((len(chunk), num_items + 1), dtype=np.float32)
            for r, u in enumerate(chunk):
                for it in hist_all[u]:
                    H[r, it] = 1.0
            C += H.T @ H
        np.fill_diagonal(C, 0.0)
        C[0, :] = C[:, 0] = 0.0
        np.save(cooc_cache, C)
    pop = np.zeros(num_items + 1)
    for u in users:
        for it in hist_all[u]:
            pop[it] += 1
    pop[0] = -1
    pop_order = [int(x) for x in np.argsort(-pop, kind="stable") if x != 0][:TOP]
    pop_rank = np.full(num_items + 1, 10 ** 6, dtype=np.float64)
    for r, it in enumerate(pop_order):
        pop_rank[it] = r + 1
    rows, cols = [], []
    for u in users:
        for it in set(hist_all[u]):
            rows.append(u)
            cols.append(it)
    M = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32),
                           (pd.factorize(pd.Series(rows))[0], cols)), shape=(len(users), num_items + 1), copy=True)
    M[:, 0] = 0
    svd = TruncatedSVD(n_components=64, random_state=SEED)
    U = svd.fit_transform(M)
    V = svd.components_.T
    uidx = {u: i for i, u in enumerate(users)}
    sas = np.load(sas_npz)
    sas_idx = {int(u): i for i, u in enumerate(sas["users"])}
    mp = np.load(mp_npz)
    mp_idx = {int(u): i for i, u in enumerate(mp["users"])}
    sas_rank = {u: {int(x): r + 1 for r, x in enumerate(sas["top500"][sas_idx[u]])} for u in users}
    mp_rank = {u: {int(x): r + 1 for r, x in enumerate(mp["top500"][mp_idx[u]])} for u in users}
    sas_set = {u: set(sas_rank[u]) for u in users}
    mp_set = {u: set(mp_rank[u]) for u in users}
    return {"C": C, "pop": pop, "pop_order": pop_order, "pop_rank": pop_rank,
            "U": U, "V": V, "uidx": uidx, "sas": sas, "sas_idx": sas_idx, "mp": mp, "mp_idx": mp_idx,
            "sas_rank": sas_rank, "mp_rank": mp_rank, "sas_set": sas_set, "mp_set": mp_set,
            "hist": hist_all}


def user_lists(u, A):
    h = A["hist"][u]
    cnt, last_pos = {}, {}
    for pos, it in enumerate(h):
        cnt[it] = cnt.get(it, 0) + 1
        last_pos[it] = pos
    return {
        "FREQ": sorted(cnt, key=lambda k: (-cnt[k], k))[:TOP],
        "RECENCY": sorted(last_pos, key=lambda k: (-last_pos[k], k))[:TOP],
        "POPULAR": A["pop_order"],
        "ITEM_CF": [int(x) for x in np.argsort(-A["C"][np.array(h[-10:], dtype=np.int64)].sum(axis=0), kind="stable") if x != 0][:TOP],
        "SVD": [int(x) for x in np.argsort(-(A["U"][A["uidx"][u]] @ A["V"].T), kind="stable") if x != 0][:TOP],
        "SASREC": [int(x) for x in A["sas"]["top500"][A["sas_idx"][u]] if x != 0][:TOP],
}


def pool(lists, arm):
    srcs = ["FREQ", "RECENCY", "POPULAR", "ITEM_CF", "SVD"] + (["SASREC"] if arm == "A3" else [])
    return rrf_fuse({s: lists[s] for s in srcs})[:TOP]

def cand_features(u, cand_list, A, aisle, dept, kind):
    """Candidate-level features at one time point (vectorized). kind unused (arm column
    selection happens outside); all 13 columns always built."""
    h = A["hist"][u]
    cand = np.asarray(cand_list, dtype=np.int64)
    cnt, last_pos = {}, {}
    for pos, it in enumerate(h):
        cnt[it] = cnt.get(it, 0) + 1
        last_pos[it] = pos
    freq = np.array([cnt.get(int(it), 0) for it in cand], dtype=np.float32)
    is_rep = (freq > 0).astype(np.float32)
    rec = np.array([np.exp(-last_pos[int(it)] / 20.0) if int(it) in last_pos else 0.0 for it in cand], dtype=np.float32)
    C = A["C"]
    if h:
        cf_row = C[np.array(h[-10:], dtype=np.int64)].sum(axis=0)
    else:
        cf_row = np.zeros(C.shape[0], dtype=np.float32)
    cf = cf_row[cand].astype(np.float32)
    if h:
        lb = np.array(sorted(set(h[-10:])), dtype=np.int64)
        lba_row = C[lb].sum(axis=0)
    else:
        lba_row = np.zeros(C.shape[0], dtype=np.float32)
    lba = lba_row[cand].astype(np.float32)
    pr = A["pop_rank"][cand].astype(np.float32)
    svd_vec = A["U"][A["uidx"][u]] @ A["V"].T
    svd_m = svd_vec[cand].astype(np.float32)
    sas_rank_u = A["sas_rank"][u]
    mp_rank_u = A["mp_rank"][u]
    sas_r = np.array([sas_rank_u.get(int(it), 999) for it in cand])
    mp_r = np.array([mp_rank_u.get(int(it), 999) for it in cand])
    sas_set = A["sas_set"][u]
    mp_set = A["mp_set"][u]
    sas_only = np.array([1.0 if (int(it) in sas_set and int(it) not in mp_set) else 0.0 for it in cand], dtype=np.float32)
    F = np.zeros((len(cand), 13), dtype=np.float32)
    F[:, 0] = is_rep
    F[:, 1] = freq
    F[:, 2] = rec
    F[:, 3] = cf
    F[:, 4] = pr
    F[:, 5] = lba
    F[:, 6] = aisle[cand]
    F[:, 7] = dept[cand]
    F[:, 8] = svd_m
    F[:, 9] = np.where(sas_r < 999, np.log1p(sas_r), 10.0)
    F[:, 10] = np.where(mp_r < 999, np.log1p(mp_r), 10.0)
    F[:, 11] = sas_only
    F[:, 12] = (sas_r <= 500).astype(np.float32)
    return F


def cols_for(arm_model):
    base = [0, 1, 2, 3, 4, 5, 6, 7]
    if arm_model == "M0":
        return base
    if arm_model == "M1":
        return base + [8]
    if arm_model == "M2":
        return base + [8, 10]          # + meanpool rank (mp match proxy)
    if arm_model == "M3":
        return base + [8, 9, 11, 12]   # + sasrec rank/only/in-pool
    if arm_model == "M4":
        return list(range(13))         # all incl. sas_only interactions via LGBM splits
    if arm_model == "M4P":
        return base + [8, 10, 11, 12]  # meanpool-featured mirror of M3
    raise ValueError(arm_model)


def main():
    t0 = time.time()
    seqs, num_items, aisle, dept = load_meta()
    users = sorted(u for u in seqs if len(seqs[u]) >= 5)
    log(f"[data] users={len(users)} items={num_items}")

    t1 = build_point_assets(seqs, users, -1, CACHE / "sasrec_v4_top500.npz", CACHE / "mean_pool_v4_top500.npz")
    t0a = build_point_assets(seqs, users, -2, CACHE / "sasrec_t0_top500.npz", CACHE / "mean_pool_t0_top500.npz")
    log(f"[assets] both time points ready ({time.time()-t0:.0f}s)")

    rng = np.random.default_rng(SEED)

    # ---------- TEST pools & feature rows (t1) ----------
    test_pools = {}
    test_feats = {}
    test_targets = {}
    user_groups = {}
    sas_unique_users = {}
    for u in users:
        s = seqs[u]
        tgt = s[-1]
        rep = tgt in s[:-1]
        user_groups[u] = "rep" if rep else "new"
        L1 = user_lists(u, t1)
        for arm in ["A2", "A3"]:
            p = pool(L1, arm)
            test_pools[(u, arm)] = p
            tgt_in = tgt in p
            test_targets[u] = (tgt, tgt_in)
        # SASRec-unique target flag (diagnostic set; never a feature)
        five = ["FREQ", "RECENCY", "POPULAR", "ITEM_CF", "SVD"]
        if user_groups[u] == "new" and tgt in L1["SASREC"] and not any(tgt in L1[x] for x in five):
            sas_unique_users[u] = tgt in test_pools[(u, "A3")]  # entered final pool?
    log(f"[pools] t1 pools built; sas-unique new targets={len(sas_unique_users)}")

    # ---------- TRAIN rows (t0): predict s[-2], pool from t0 assets (cached) ----------
    TRAIN_CACHE = CACHE / "mtrain_rows.npz"
    if TRAIN_CACHE.exists():
        z = np.load(TRAIN_CACHE)
        Xall, y_tr = z["X"], z["y"]
        log(f"[train] loaded cached rows={len(y_tr)} pos={int(y_tr.sum())}")
    else:
        Fblocks, ys = [], []
        train_users = [u for u in users if len(seqs[u]) >= 5]
        for u in train_users:
            s = seqs[u]
            tgt = s[-2]
            L0 = user_lists(u, t0a)
            p = pool(L0, "A3")
            if tgt not in p:
                continue
            pos_j = p.index(tgt)
            negs = rng.choice(len(p), size=60, replace=False)
            negs = [j for j in negs if j != pos_j][:50]
            idxs = [pos_j] + negs
            Fblocks.append(cand_features(u, [p[j] for j in idxs], t0a, aisle, dept, None))
            ys.extend([1] + [0] * len(negs))
        Xall = np.concatenate(Fblocks, axis=0)
        y_tr = np.array(ys, dtype=np.int32)
        np.savez_compressed(TRAIN_CACHE, X=Xall, y=y_tr)
        log(f"[train] rows={len(y_tr)} pos={int(y_tr.sum())} cached ({time.time()-t0:.0f}s)")

    # ---------- fit & evaluate ----------
    results = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    per_user_hits = {}
    for m in ["M0", "M1", "M2", "M3", "M4", "M4P"]:
        hp = CACHE / f"hits_{m}.npz"
        if m in results and hp.exists():
            z = np.load(hp)
            per_user_hits[m] = list(zip(z["u"].tolist(), z["h"].tolist(), z["g"].tolist()))
            log(f"[skip] {m} (hits loaded)")
            continue
        if m in results:
            del results[m]
            log(f"[reeval] {m} (no hits cache)")
        cols = cols_for(m)
        X_tr = Xall[:, cols]

        clf = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.08, num_leaves=63,
                                 random_state=SEED, n_jobs=-1, verbose=-1)
        clf.fit(X_tr, y_tr)
        hits = {"all": 0, "new": 0, "rep": 0}
        u2_hits = 0
        hit_flags = []
        for u in users:
            tgt, tgt_in = test_targets[u]
            grp = user_groups[u]
            p = test_pools[(u, "A3")]
            F = cand_features(u, p, t1, aisle, dept, m)
            scores = clf.predict_proba(F[:, cols_for(m)])[:, 1]
            top10_idx = np.argsort(-scores)[:10]
            top10_items = [p[int(i)] for i in top10_idx]
            hit = bool(tgt_in and tgt in top10_items)
            hit_flags.append((u, int(hit), grp))
            hits["all"] += int(hit)
            hits[grp] += int(hit)
            if u in sas_unique_users and sas_unique_users[u] and hit:
                u2_hits += 1
        n = {"all": len(users), "new": sum(1 for g in user_groups.values() if g == "new"),
             "rep": sum(1 for g in user_groups.values() if g == "rep")}
        results[m] = {
            "hr10": {g: round(hits[g] / n[g], 4) for g in hits},
            "U2_unique_pool_hits": u2_hits,
            "U2_rate": round(u2_hits / max(len(sas_unique_users), 1), 4),
        }
        per_user_hits[m] = hit_flags
        np.savez(CACHE / f"hits_{m}.npz",
                 u=np.array([f[0] for f in hit_flags], dtype=np.int64),
                 h=np.array([f[1] for f in hit_flags], dtype=np.int8),
                 g=np.array([f[2] for f in hit_flags], dtype="U3"))
        # per-arm checkpoint (idempotent resume)
        ck = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
        ck[m] = results[m]
        OUT.write_text(json.dumps(ck, indent=2), encoding="utf-8")
        log(f"[{m}] {json.dumps(results[m])}")

    # paired bootstrap M3 vs M1, M4 vs M3, M3 vs M2 (user-level)
    def paired_bs(a, b, n_boot=2000):
        ha = {u: h for u, h, _ in per_user_hits[a]}
        hb = {u: h for u, h, _ in per_user_hits[b]}
        us = sorted(ha)
        d = np.array([ha[u] - hb[u] for u in us])
        rng2 = np.random.default_rng(SEED)
        boots = [rng2.choice(d, size=len(d), replace=True).mean() for _ in range(n_boot)]
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return {"mean_diff": round(float(d.mean()), 5), "ci95": [round(lo, 5), round(hi, 5)]}

    results["paired_bootstrap"] = {f"{a}_vs_{b}": paired_bs(a, b)
                                   for a, b in [("M3", "M1"), ("M4", "M3"), ("M3", "M2"), ("M4P", "M4")]}
    results["protocol"] = {"pool": "A3@500 (t1) / t0 pool for train", "train_target": "s[-2] via t0 assets",
                           "seed": SEED, "seconds": round(time.time() - t0, 1)}
    OUT.write_text(json.dumps(results, indent=2), encoding="utf-8")
    log(json.dumps(results["paired_bootstrap"], indent=2))
    log("DONE")


if __name__ == "__main__":
    main()
