"""Confirmation experiments for GPT FINAL VERDICT (2026-10-26).

Experiment 2 — H&M enlarged frozen sample paired CI (artifacts/eval/hm_enlarged_paired_ci.json):
  40,000 customers stratified at cutoff 2020-09-08 (seed 42, pre-frozen), article universe =
  top 5,000 by pre-cutoff counts among sampled customers. Seven models (B1, B2, MeanPool_SS256,
  GRU_SS256, SAS_SS256, HSTU_SS256, Hybrid_B2_neural) evaluated on W4; customer-level paired
  bootstrap 95% CI (1000 resamples) for pre-registered contrasts, overall and per stratum.

Experiment 3 — order confirmation arms (artifacts/eval/order_confirmation_arms.json):
  GRU and SASRec (SampledSoftmax-256, frozen 8-epoch budget) on both domains:
    A  true-train / true-infer
    B  true-train / shuffled-infer (3 shuffle seeds)
    C  shuffled-train / shuffled-infer (matched, 3 seeds)
    D  reversed-train / reversed-infer (recency inversion control)

All runs respect sealed_cutoff and never touch post-cutoff labels.
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch

from src.data.adapters import load_hm_events, load_swire_events, stratified_customer_sample
from src.eval.metrics import _canonical_top_k, _label_sets, compute_user_ap12_components
from src.models.r1_baselines import R1LightGBMRanker, evaluate_r1_predictions
from src.models.r1_candidates import _prepare_history, compute_svd_embeddings, generate_r1_candidates
from src.models.r1_features import build_r1_feature_matrix
from src.models.r2_hybrid_runner import (
    HYBRID_NEURAL_FEATURE_COLS,
    R2HybridLightGBMRanker,
    compute_neural_oof_features,
    predict_track_a_candidates,
    train_r2_sequential_model,
)

HM_CUTOFFS = ["2020-08-18", "2020-08-25", "2020-09-01", "2020-09-08"]
HM_SEALED = "2020-09-15"
SEED = 42
DEVICE = "cuda"
N_SAMPLE = 40000
FALLBACK_N = 30000


def log(msg: str) -> None:
    print(msg, flush=True)


def build_windows(events: pd.DataFrame, cutoffs, sealed: str, k_pool: int = 100, tag: str = "hm"):
    safe = events.loc[
        events["event_date"].astype(str) <= str(sealed),
        ["customer_id", "article_id", "event_date", "value"],
    ].copy()
    for c in ("customer_id", "article_id", "event_date"):
        safe[c] = safe[c].astype(str)
    safe["value"] = safe["value"].astype(float)

    arts: Dict[str, Dict[str, Any]] = {}
    for idx, cutoff in enumerate(cutoffs, start=1):
        t0 = time.time()
        w = build_window(safe, cutoff_date=cutoff, horizon_days=7)
        strata_df = assign_customer_strata(safe, cutoff_date=cutoff)
        target_users = strata_df["customer_id"].astype(str).tolist()
        hist_prep = _prepare_history(safe, cutoff_date=cutoff)
        svd_pack = compute_svd_embeddings(hist_prep, embedding_dim=32, half_life_days=14.0, random_state=SEED)
        cands = generate_r1_candidates(
            safe, cutoff_date=cutoff, customer_ids=target_users, k_pool=k_pool,
            per_source_k=max(k_pool, 200), half_life_days=14.0, embedding_dim=32,
            random_state=SEED, svd_pack=svd_pack,
        )
        feat = build_r1_feature_matrix(
            safe, cands, cutoff_date=cutoff, half_life_days=14.0, embedding_dim=32,
            random_state=SEED, svd_pack=svd_pack,
        )
        arts[f"W{idx}"] = {
            "cutoff_date": cutoff, "window": w, "strata": strata_df,
            "target_users": target_users, "cands": cands, "feat": feat,
        }
        log(f"  [{tag}] window W{idx} cutoff={cutoff} cands={len(cands)} feat={len(feat)} ({time.time()-t0:.0f}s)")
    return safe, arts


from src.data.window_builder import assign_customer_strata, build_window  # noqa: E402


def train_seq(safe, cutoff, encoder, loss="sampled_softmax_256", epochs=8, sequence_order="chronological", shuffle_seed=SEED):
    return train_r2_sequential_model(
        events=safe, cutoff_date=cutoff, encoder_name=encoder, loss_name=loss,
        embedding_dim=128, max_seq_len=50, num_layers=2, num_heads=4, num_negatives=256,
        batch_size=256, epochs=epochs, learning_rate=2e-3, seed=SEED, device=DEVICE,
        sequence_order=sequence_order, shuffle_seed=shuffle_seed,
    )


def track_a_eval(trained, eval_art):
    preds = predict_track_a_candidates(trained, eval_art["cands"])
    ev = evaluate_r1_predictions(
        preds, candidates=eval_art["cands"], labels=eval_art["window"]["labels"],
        history=eval_art["window"]["history"], strata_df=eval_art["strata"], k=12,
    )
    return preds, ev


def per_customer_ap(preds, labels, history) -> Dict[str, float]:
    top12 = _canonical_top_k(preds, k=12)
    labs = _label_sets(labels)
    hist_df = history[["customer_id", "article_id"]].drop_duplicates()
    hist_df = hist_df.assign(customer_id=hist_df["customer_id"].astype(str), article_id=hist_df["article_id"].astype(str))
    hist_sets = hist_df.groupby("customer_id")["article_id"].apply(set).to_dict()
    out = {}
    for cust, true_items in labs.items():
        ap_total, _r, _e = compute_user_ap12_components(top12.get(cust, []), true_items, hist_sets.get(cust, set()))
        out[str(cust)] = float(ap_total)
    return out


def paired_ci(ap_a: Dict[str, float], ap_b: Dict[str, float], strata_df, n_boot=1000):
    rng = np.random.default_rng(SEED)
    strat = dict(zip(strata_df["customer_id"].astype(str), strata_df["stratum"].astype(str)))
    common = sorted(set(ap_a) & set(ap_b))
    diffs = {c: ap_a[c] - ap_b[c] for c in common}
    out: Dict[str, Any] = {}
    for scope, keys in [("ALL", common)] + [(s, [c for c in common if strat.get(c) == s]) for s in ("S0", "S1", "S2")]:
        if not keys:
            out[scope] = None
            continue
        vals = np.array([diffs[c] for c in keys], dtype=np.float64)
        idx = rng.integers(0, vals.size, size=(n_boot, vals.size))
        means = vals[idx].mean(axis=1)
        out[scope] = {
            "n_paired": int(vals.size),
            "mean_diff": float(vals.mean()),
            "ci_lower_95": float(np.quantile(means, 0.025)),
            "ci_upper_95": float(np.quantile(means, 0.975)),
            "p_value_two_sided": float(2 * min((means <= 0).mean(), (means >= 0).mean())),
        }
    return out


def shuffled_infer_pack(trained, seed):
    seq_pack = trained["seq_pack"]
    ids = seq_pack["infer_input_ids"].detach().cpu().numpy().copy()
    rng = np.random.default_rng(seed + 777)
    for r in range(ids.shape[0]):
        valid = np.flatnonzero(ids[r] > 0)
        if len(valid) >= 2:
            perm = rng.permutation(len(valid))
            ids[r, valid] = ids[r, valid[perm]]
    return {**trained, "seq_pack": {**seq_pack, "infer_input_ids": torch.from_numpy(ids)}}


def eval_summary(ev) -> Dict[str, float]:
    return {
        "map12": float(ev["map12"]),
        "s2_map12": float(ev["strata"]["S2"]["map12"]),
        "s1_map12": float(ev["strata"]["S1"]["map12"]),
        "s0_map12": float(ev["strata"]["S0"]["map12"]),
        "explore_ap12": float(ev["repeat_explore"]["map12_explore"]),
        "repeat_ap12": float(ev["repeat_explore"]["map12_repeat"]),
        "evaluated_customers": int(ev["evaluated_customers"]),
    }


# ---------------------------------------------------------------- H&M sample
def build_hm_enlarged_sample(events: pd.DataFrame, n_customers: int):
    ev6 = events.loc[
        (events["event_date"].astype(str) >= "2020-04-01") & (events["event_date"].astype(str) <= HM_SEALED)
    ].copy()
    sample = stratified_customer_sample(ev6, cutoff_date="2020-09-08", n_customers=n_customers, seed=SEED)
    custs = set(sample["customer_id"].astype(str))
    sub = ev6.loc[ev6["customer_id"].astype(str).isin(custs)]
    hist_part = sub.loc[sub["event_date"].astype(str) <= "2020-09-08"]
    top = hist_part["article_id"].value_counts().head(5000).index
    sample_events = sub.loc[sub["article_id"].isin(set(top))].copy()
    meta = {
        "sampled_customers": int(len(custs)),
        "active_catalog_articles": int(sample_events["article_id"].nunique()),
        "total_events": int(len(sample_events)),
        "date_range": [str(sample_events["event_date"].min()), str(sample_events["event_date"].max())],
        "sampling_cutoff": "2020-09-08",
        "article_universe_cutoff": "2020-09-08",
        "seed": SEED,
        "frozen": True,
    }
    return sample_events, meta


def main() -> None:
    t0 = time.time()

    # ============ Experiment 2: H&M enlarged paired CI ============
    log("[exp2] loading H&M events")
    hm_events = load_hm_events(REPO / "data/hm/transactions_train.parquet")
    try:
        hm_sample, hm_meta = build_hm_enlarged_sample(hm_events, N_SAMPLE)
    except MemoryError:
        log("[exp2] MemoryError at 40k, falling back to 30k")
        hm_sample, hm_meta = build_hm_enlarged_sample(hm_events, FALLBACK_N)
    log(f"[exp2] sample: {hm_meta}")

    log("[exp2] building windows")
    hm_safe, hm_arts = build_windows(hm_sample, HM_CUTOFFS, HM_SEALED, tag="hm40k")
    eval_art = hm_arts["W4"]
    train_ws = ["W1", "W2", "W3"]
    labels = eval_art["window"]["labels"]
    history = eval_art["window"]["history"]
    strata_df = eval_art["strata"]

    log("[exp2] LightGBM B1/B2")
    train_feat = pd.concat([hm_arts[tw]["feat"] for tw in train_ws], ignore_index=True)
    train_labels = pd.concat([hm_arts[tw]["window"]["labels"] for tw in train_ws], ignore_index=True)
    b1 = R1LightGBMRanker(use_embeddings=False, n_estimators=250, random_state=SEED).fit(train_feat, train_labels)
    preds_b1 = b1.predict(eval_art["feat"])
    b2 = R1LightGBMRanker(use_embeddings=True, n_estimators=250, random_state=SEED).fit(train_feat, train_labels)
    preds_b2 = b2.predict(eval_art["feat"])

    log("[exp2] neural models at W4 (chronological)")
    neural_trainers = {"M_POOL_SS256": "mean_pool", "M_GRU_SS256": "gru", "M_SAS_SS256": "sasrec", "M_HSTU_SS256": "hstu_math"}
    neural_packs, neural_preds = {}, {}
    for mid, enc in neural_trainers.items():
        pack = train_seq(hm_safe, eval_art["cutoff_date"], enc)
        neural_packs[mid] = pack
        preds, _ev = track_a_eval(pack, eval_art)
        neural_preds[mid] = preds
        log(f"[exp2] trained+predicted {mid}")

    log("[exp2] Hybrid_B2 (GRU OOF)")
    for tw in train_ws:
        tw_pack = train_seq(hm_safe, hm_arts[tw]["cutoff_date"], "gru")
        oof = compute_neural_oof_features(tw_pack, hm_arts[tw]["cands"])
        fh = hm_arts[tw]["feat"].copy()
        fh["sequence_logit"] = oof["sequence_logit"].to_numpy(dtype=np.float32)
        fh["embedding_similarity"] = oof["embedding_similarity"].to_numpy(dtype=np.float32)
        hm_arts[tw]["feat_hybrid"] = fh
    w4_oof = compute_neural_oof_features(neural_packs["M_GRU_SS256"], eval_art["cands"])
    val_fh = eval_art["feat"].copy()
    val_fh["sequence_logit"] = w4_oof["sequence_logit"].to_numpy(dtype=np.float32)
    val_fh["embedding_similarity"] = w4_oof["embedding_similarity"].to_numpy(dtype=np.float32)
    train_fh = pd.concat([hm_arts[tw]["feat_hybrid"] for tw in train_ws], ignore_index=True)
    hyb2 = R2HybridLightGBMRanker(extra_neural_cols=HYBRID_NEURAL_FEATURE_COLS, use_svd_embeddings=True,
                                  n_estimators=250, random_state=SEED).fit(train_fh, train_labels)
    preds_hyb2 = hyb2.predict(val_fh)

    log("[exp2] per-customer AP + paired CIs")
    ap_maps = {"B1_strong_lgbm": per_customer_ap(preds_b1, labels, history),
               "B2_lgbm_svd_emb": per_customer_ap(preds_b2, labels, history),
               "Hybrid_B2_neural": per_customer_ap(preds_hyb2, labels, history)}
    for mid in neural_preds:
        ap_maps[mid] = per_customer_ap(neural_preds[mid], labels, history)
    contrasts = [
        ("M_GRU_SS256", "B1_strong_lgbm"), ("M_GRU_SS256", "M_POOL_SS256"),
        ("M_SAS_SS256", "B1_strong_lgbm"), ("M_HSTU_SS256", "B1_strong_lgbm"),
        ("Hybrid_B2_neural", "B2_lgbm_svd_emb"), ("Hybrid_B2_neural", "B1_strong_lgbm"),
        ("M_GRU_SS256", "B2_lgbm_svd_emb"), ("B2_lgbm_svd_emb", "B1_strong_lgbm"),
    ]
    strat_map = dict(zip(strata_df["customer_id"].astype(str), strata_df["stratum"].astype(str)))
    strata_point = {}
    for m, ap in ap_maps.items():
        per: Dict[str, float] = {}
        for s in ("S0", "S1", "S2"):
            vals = [v for c, v in ap.items() if strat_map.get(c) == s]
            per[s] = float(np.mean(vals)) if vals else float("nan")
        strata_point[m] = per
    out2 = {
        "contract_version": "0.9", "experiment": "hm_enlarged_paired_ci",
        "sample_metadata": hm_meta,
        "point_map12": {m: float(np.mean(list(ap.values()))) for m, ap in ap_maps.items()},
        "strata_point_map12": strata_point,
        "paired_ci": {f"{a}_minus_{b}": paired_ci(ap_maps[a], ap_maps[b], strata_df) for a, b in contrasts},
    }
    (REPO / "artifacts/eval/hm_enlarged_paired_ci.json").write_text(json.dumps(out2, indent=2), encoding="utf-8")
    log("[exp2] saved artifacts/eval/hm_enlarged_paired_ci.json")

    # ============ Experiment 3: order confirmation arms ============
    log("[exp3] arms on H&M enlarged sample")
    out3: Dict[str, Any] = {"contract_version": "0.9", "experiment": "order_confirmation_arms",
                            "shuffle_seeds": [42, 43, 44], "domains": {}}
    for domain, safe, arts in (("hm_enlarged", hm_safe, hm_arts), ("swire_ro", None, None)):
        if domain == "swire_ro":
            log("[exp3] loading swire + windows")
            sw_events = load_swire_events()
            sw_art = json.loads((REPO / "artifacts/r2/swire_ro_r2_results.json").read_text(encoding="utf-8"))
            safe, arts = build_windows(sw_events, sw_art["validation_cutoffs"], sw_art["sealed_cutoff"], tag="swire")
        ev_art = arts["W4"]
        dom_out: Dict[str, Any] = {}
        for enc in ("gru", "sasrec"):
            log(f"[exp3] {domain}/{enc}: arm A true/true")
            pack_a = train_seq(safe, ev_art["cutoff_date"], enc)
            _p, ev_a = track_a_eval(pack_a, ev_art)
            arm_a = eval_summary(ev_a)
            arm_b = []
            for s in (42, 43, 44):
                _p, ev_b = track_a_eval(shuffled_infer_pack(pack_a, s), ev_art)
                arm_b.append({"seed": s, **eval_summary(ev_b)})
            arm_c = []
            for s in (42, 43, 44):
                pack_c = train_seq(safe, ev_art["cutoff_date"], enc, sequence_order="shuffled", shuffle_seed=s)
                _p, ev_c = track_a_eval(pack_c, ev_art)
                arm_c.append({"seed": s, **eval_summary(ev_c)})
            log(f"[exp3] {domain}/{enc}: arm D reversed/reversed")
            pack_d = train_seq(safe, ev_art["cutoff_date"], enc, sequence_order="reversed")
            _p, ev_d = track_a_eval(pack_d, ev_art)
            arm_d = eval_summary(ev_d)
            dom_out[enc] = {
                "A_true_true": arm_a,
                "B_true_train_shuffled_infer": arm_b,
                "C_shuffled_train_shuffled_infer": arm_c,
                "D_reversed_reversed": arm_d,
                "A_minus_B_mean_map12": float(arm_a["map12"] - np.mean([b["map12"] for b in arm_b])),
                "A_minus_C_mean_map12": float(arm_a["map12"] - np.mean([c["map12"] for c in arm_c])),
                "A_minus_D_map12": float(arm_a["map12"] - arm_d["map12"]),
                "A_minus_B_mean_s2": float(arm_a["s2_map12"] - np.mean([b["s2_map12"] for b in arm_b])),
                "A_minus_C_mean_s2": float(arm_a["s2_map12"] - np.mean([c["s2_map12"] for c in arm_c])),
                "A_minus_D_s2": float(arm_a["s2_map12"] - arm_d["s2_map12"]),
            }
            log(f"[exp3] {domain}/{enc} done: A={arm_a['map12']:.4f} "
                f"B_mean={np.mean([b['map12'] for b in arm_b]):.4f} "
                f"C_mean={np.mean([c['map12'] for c in arm_c]):.4f} D={arm_d['map12']:.4f}")
        out3["domains"][domain] = dom_out
        (REPO / "artifacts/eval/order_confirmation_arms.json").write_text(json.dumps(out3, indent=2), encoding="utf-8")

    log(f"DONE in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
