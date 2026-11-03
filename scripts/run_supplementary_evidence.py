"""Supplementary evidence runs requested by external scientific review (2026-10-26).

Blocks:
  A. H&M convergence probe: identical protocol and identical supplementary sample at
     epochs=8 vs epochs=24 (the epochs effect is identified within the probe; primary
     cross-domain comparisons remain the frozen epochs=8 artifacts).
  B. swire_ro sequence-order shuffle ablation for GRU/SASRec/HSTU-Math under
     SampledSoftmax-256 (merged into artifacts/r2/swire_ro_r2_results.json).
  C. Per-stratum (S0/S1/S2) customer-level bootstrap 95% CI of MAP@12 on W4 for
     B1_strong_lgbm / B2_lgbm_svd_emb / Hybrid_B2_neural / M_GRU_SS256 on both domains
     (artifacts/eval/w4_strata_bootstrap_ci.json).

All runs respect sealed_cutoff and never touch post-cutoff labels.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Sequence

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.data.adapters import load_hm_events, load_swire_events, stratified_customer_sample
from src.data.window_builder import assign_customer_strata, build_window
from src.eval.metrics import _canonical_top_k, _label_sets, compute_user_ap12_components
from src.models.r1_baselines import R1LightGBMRanker, evaluate_r1_predictions
from src.models.r1_candidates import _prepare_history, compute_svd_embeddings, generate_r1_candidates
from src.models.r1_features import build_r1_feature_matrix
from src.models.r2_hybrid_runner import (
    HYBRID_NEURAL_FEATURE_COLS,
    R2HybridLightGBMRanker,
    compute_neural_oof_features,
    evaluate_sequence_order_shuffle_ablation,
    evaluate_track_b_predictions,
    predict_track_a_candidates,
    predict_track_b_full_catalog,
    run_r2_orthogonal_benchmark,
    train_r2_sequential_model,
)

HM_CUTOFFS = ["2020-08-18", "2020-08-25", "2020-09-01", "2020-09-08"]
HM_SEALED = "2020-09-15"
SEED = 42
DEVICE = "cuda"


def build_windows(events: pd.DataFrame, cutoffs: Sequence[str], sealed: str, k_pool: int = 100):
    safe = events.loc[
        events["event_date"].astype(str) <= str(sealed),
        ["customer_id", "article_id", "event_date", "value"],
    ].copy()
    for c in ("customer_id", "article_id", "event_date"):
        safe[c] = safe[c].astype(str)
    safe["value"] = safe["value"].astype(float)

    arts: Dict[str, Dict[str, Any]] = {}
    for idx, cutoff in enumerate(cutoffs, start=1):
        w = build_window(safe, cutoff_date=cutoff, horizon_days=7)
        strata_df = assign_customer_strata(safe, cutoff_date=cutoff)
        target_users = strata_df["customer_id"].astype(str).tolist()
        hist_prep = _prepare_history(safe, cutoff_date=cutoff)
        svd_pack = compute_svd_embeddings(hist_prep, embedding_dim=32, half_life_days=14.0, random_state=SEED)
        cands = generate_r1_candidates(
            safe,
            cutoff_date=cutoff,
            customer_ids=target_users,
            k_pool=k_pool,
            per_source_k=max(k_pool, 200),
            half_life_days=14.0,
            embedding_dim=32,
            random_state=SEED,
            svd_pack=svd_pack,
        )
        feat = build_r1_feature_matrix(
            safe, cands, cutoff_date=cutoff, half_life_days=14.0, embedding_dim=32,
            random_state=SEED, svd_pack=svd_pack,
        )
        arts[f"W{idx}"] = {
            "cutoff_date": cutoff,
            "window": w,
            "strata": strata_df,
            "target_users": target_users,
            "cands": cands,
            "feat": feat,
        }
        print(f"  window W{idx} cutoff={cutoff} cands={len(cands)} feat={len(feat)}", flush=True)
    return safe, arts


def train_spec(safe: pd.DataFrame, cutoff: str, encoder: str, loss: str, epochs: int = 8):
    return train_r2_sequential_model(
        events=safe,
        cutoff_date=cutoff,
        encoder_name=encoder,
        loss_name=loss,
        embedding_dim=128,
        max_seq_len=50,
        num_layers=2,
        num_heads=4,
        num_negatives=256,
        batch_size=256,
        epochs=epochs,
        learning_rate=2e-3,
        seed=SEED,
        device=DEVICE,
    )


def per_customer_ap(preds: pd.DataFrame, labels: pd.DataFrame, history: pd.DataFrame) -> Dict[str, float]:
    top12 = _canonical_top_k(preds, k=12)
    labs = _label_sets(labels)
    hist_df = history[["customer_id", "article_id"]].drop_duplicates()
    hist_df = hist_df.assign(customer_id=hist_df["customer_id"].astype(str), article_id=hist_df["article_id"].astype(str))
    hist_sets = hist_df.groupby("customer_id")["article_id"].apply(set).to_dict()
    out: Dict[str, float] = {}
    for cust, true_items in labs.items():
        ranked = top12.get(cust, [])
        ap_total, _rep, _exp = compute_user_ap12_components(ranked, true_items, hist_sets.get(cust, set()))
        out[str(cust)] = float(ap_total)
    return out


def strata_bootstrap_ci(ap_map: Dict[str, float], strata_df: pd.DataFrame, n_bootstrap: int = 1000) -> Dict[str, Any]:
    rng = np.random.default_rng(SEED)
    strat = dict(zip(strata_df["customer_id"].astype(str), strata_df["stratum"].astype(str)))
    out: Dict[str, Any] = {}
    for s in ("S0", "S1", "S2"):
        vals = np.array([v for c, v in ap_map.items() if strat.get(c) == s], dtype=np.float64)
        if vals.size == 0:
            out[s] = None
            continue
        idx = rng.integers(0, vals.size, size=(n_bootstrap, vals.size))
        means = vals[idx].mean(axis=1)
        out[s] = {
            "n_scored": int(vals.size),
            "map12_point": float(vals.mean()),
            "ci_lower_95": float(np.quantile(means, 0.025)),
            "ci_upper_95": float(np.quantile(means, 0.975)),
        }
    overall = np.array(list(ap_map.values()), dtype=np.float64)
    idx = rng.integers(0, overall.size, size=(n_bootstrap, overall.size))
    means = overall[idx].mean(axis=1)
    out["ALL"] = {
        "n_scored": int(overall.size),
        "map12_point": float(overall.mean()),
        "ci_lower_95": float(np.quantile(means, 0.025)),
        "ci_upper_95": float(np.quantile(means, 0.975)),
    }
    return out


def domain_ci_block(safe: pd.DataFrame, arts: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    eval_art = arts["W4"]
    train_ws = ["W1", "W2", "W3"]
    labels = eval_art["window"]["labels"]
    history = eval_art["window"]["history"]

    train_feat = pd.concat([arts[tw]["feat"] for tw in train_ws], ignore_index=True)
    train_labels = pd.concat([arts[tw]["window"]["labels"] for tw in train_ws], ignore_index=True)

    b1 = R1LightGBMRanker(use_embeddings=False, n_estimators=250, random_state=SEED).fit(train_feat, train_labels)
    preds_b1 = b1.predict(eval_art["feat"])
    b2 = R1LightGBMRanker(use_embeddings=True, n_estimators=250, random_state=SEED).fit(train_feat, train_labels)
    preds_b2 = b2.predict(eval_art["feat"])

    # Neural OOF on train windows + W4 (GRU + SampledSoftmax-256, same protocol as primary artifacts)
    for tw in train_ws:
        tw_trained = train_spec(safe, arts[tw]["cutoff_date"], "gru", "sampled_softmax_256")
        oof = compute_neural_oof_features(tw_trained, arts[tw]["cands"])
        feat_h = arts[tw]["feat"].copy()
        feat_h["sequence_logit"] = oof["sequence_logit"].to_numpy(dtype=np.float32)
        feat_h["embedding_similarity"] = oof["embedding_similarity"].to_numpy(dtype=np.float32)
        arts[tw]["feat_hybrid"] = feat_h
    w4_trained = train_spec(safe, eval_art["cutoff_date"], "gru", "sampled_softmax_256")
    w4_oof = compute_neural_oof_features(w4_trained, eval_art["cands"])
    val_feat_h = eval_art["feat"].copy()
    val_feat_h["sequence_logit"] = w4_oof["sequence_logit"].to_numpy(dtype=np.float32)
    val_feat_h["embedding_similarity"] = w4_oof["embedding_similarity"].to_numpy(dtype=np.float32)

    train_feat_hyb = pd.concat([arts[tw]["feat_hybrid"] for tw in train_ws], ignore_index=True)
    hyb2 = R2HybridLightGBMRanker(
        extra_neural_cols=HYBRID_NEURAL_FEATURE_COLS, use_svd_embeddings=True,
        n_estimators=250, random_state=SEED,
    ).fit(train_feat_hyb, train_labels)
    preds_hyb2 = hyb2.predict(val_feat_h)

    preds_gru = predict_track_a_candidates(w4_trained, eval_art["cands"])

    strata_df = eval_art["strata"]
    block: Dict[str, Any] = {}
    for name, preds in (
        ("B1_strong_lgbm", preds_b1),
        ("B2_lgbm_svd_emb", preds_b2),
        ("Hybrid_B2_neural", preds_hyb2),
        ("M_GRU_SS256", preds_gru),
    ):
        ap = per_customer_ap(preds, labels, history)
        block[name] = strata_bootstrap_ci(ap, strata_df)
        print(f"  CI {name}: " + ", ".join(
            f"{s}={block[name][s]['map12_point']:.4f}[{block[name][s]['ci_lower_95']:.4f},{block[name][s]['ci_upper_95']:.4f}]"
            for s in ("S0", "S1", "S2") if block[name][s]
        ), flush=True)
    return block


def build_hm_supplementary_sample(events: pd.DataFrame):
    """Documented supplementary H&M sample (self-consistent; independent of the primary 10k artifact sample).

    Definition: events restricted to 2020-04-01..2020-09-15; 10,000 customers drawn by
    stratified_customer_sample at cutoff 2020-09-08 (seed 42); article universe = top 5,000
    articles by purchase count among sampled customers using only events <= 2020-09-08
    (no label-window information in article selection).
    """
    ev6 = events.loc[
        (events["event_date"].astype(str) >= "2020-04-01") & (events["event_date"].astype(str) <= HM_SEALED)
    ].copy()
    sample = stratified_customer_sample(ev6, cutoff_date="2020-09-08", n_customers=10000, seed=SEED)
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
    }
    print(f"  supplementary sample: {meta}", flush=True)
    return sample_events, meta


def main() -> None:
    t0 = time.time()
    out_ci: Dict[str, Any] = {
        "contract_version": "0.9",
        "purpose": "per-stratum customer-level bootstrap 95% CI of MAP@12 on W4",
        "domains": {},
    }

    # ---------------- H&M ----------------
    print("[H&M] loading events", flush=True)
    hm_events = load_hm_events(REPO / "data/hm/transactions_train.parquet")
    print("[H&M] building supplementary sample", flush=True)
    hm_sample, hm_meta = build_hm_supplementary_sample(hm_events)

    print("[H&M] A. convergence probe epochs=8 vs epochs=24 on identical sample", flush=True)
    probe: Dict[str, Any] = {
        "contract_version": "0.9",
        "supplementary": True,
        "supplementary_note": (
            "Convergence probe: identical protocol and identical sample, epochs=8 vs epochs=24. "
            "The epochs effect is identified within this probe; primary cross-domain comparisons "
            "remain the frozen epochs=8 artifacts (artifacts/r2/hm_r2_results.json)."
        ),
        "sample_metadata": hm_meta,
    }
    for ep in (8, 24):
        print(f"  probe epochs={ep}", flush=True)
        res = run_r2_orthogonal_benchmark(
            hm_sample,
            validation_cutoffs=HM_CUTOFFS,
            sealed_cutoff=HM_SEALED,
            epochs=ep,
            device=DEVICE,
        )
        probe[f"epochs_{ep}"] = res
    ta8 = probe["epochs_8"]["track_a_candidate_rerank"]
    ta24 = probe["epochs_24"]["track_a_candidate_rerank"]
    probe["epochs_effect_track_a_map12"] = {
        mid: float(ta24[mid]["map12"] - ta8[mid]["map12"])
        for mid in ta8
        if mid in ta24 and isinstance(ta8[mid], dict) and "map12" in ta8[mid]
    }
    (REPO / "artifacts/r2/hm_r2_convergence_probe_epochs8_vs_24.json").write_text(
        json.dumps(probe, indent=2), encoding="utf-8"
    )
    print("[H&M] probe saved", flush=True)

    print("[H&M] C. per-stratum bootstrap CI", flush=True)
    hm_safe, hm_arts = build_windows(hm_sample, HM_CUTOFFS, HM_SEALED)
    out_ci["domains"]["hm_10k_active"] = {
        "sample_metadata": hm_meta,
        "primary_eval_cutoff": HM_CUTOFFS[-1],
        "strata_ci": domain_ci_block(hm_safe, hm_arts),
    }

    # ---------------- swire_ro ----------------
    print("[swire] loading events", flush=True)
    sw_events = load_swire_events()
    sw_art_path = REPO / "artifacts/r2/swire_ro_r2_results.json"
    sw_art = json.loads(sw_art_path.read_text(encoding="utf-8"))
    sw_cutoffs = list(sw_art["validation_cutoffs"])
    sw_sealed = sw_art["sealed_cutoff"]

    print("[swire] building windows", flush=True)
    sw_safe, sw_arts = build_windows(sw_events, sw_cutoffs, sw_sealed)
    eval_art = sw_arts["W4"]

    print("[swire] B. shuffle ablation GRU/SASRec/HSTU SS256", flush=True)
    shuffle_out: Dict[str, Any] = dict(sw_art.get("sequence_order_shuffle_ablation", {}))
    for model_id, enc in (("M_GRU_SS256", "gru"), ("M_SAS_SS256", "sasrec"), ("M_HSTU_SS256", "hstu_math")):
        trained = train_spec(sw_safe, eval_art["cutoff_date"], enc, "sampled_softmax_256")
        preds_a = predict_track_a_candidates(trained, eval_art["cands"])
        eval_a = evaluate_r1_predictions(
            preds_a, candidates=eval_art["cands"], labels=eval_art["window"]["labels"],
            history=eval_art["window"]["history"], strata_df=eval_art["strata"], k=12,
        )
        preds_b = predict_track_b_full_catalog(trained, customer_ids=eval_art["target_users"], top_k=12)
        eval_b = evaluate_track_b_predictions(
            preds_b, labels=eval_art["window"]["labels"], history=eval_art["window"]["history"],
            strata_df=eval_art["strata"], k=12,
        )
        shuf = evaluate_sequence_order_shuffle_ablation(
            trained, eval_art, eval_a, eval_b, preds_a, encoder_name=enc,
            loss_name="sampled_softmax_256", seed=SEED,
        )
        shuffle_out[model_id] = shuf
        print(f"  shuffle {model_id}: dA={shuf['track_a_overall_delta_true_minus_shuffled']:.4f} "
              f"dA_S2={shuf['track_a_s2_delta_true_minus_shuffled']:.4f}", flush=True)
    sw_art["sequence_order_shuffle_ablation"] = shuffle_out
    sw_art["supplementary_shuffle_note"] = (
        "GRU/SASRec/HSTU-Math shuffle entries computed 2026-10-26 in scripts/run_supplementary_evidence.py "
        "with identical hyperparameters and seed as the primary run; MeanPool entry (order-invariant by "
        "construction) came from the primary run."
    )
    sw_art_path.write_text(json.dumps(sw_art, indent=2), encoding="utf-8")
    print("[swire] shuffle merged", flush=True)

    print("[swire] C. per-stratum bootstrap CI", flush=True)
    out_ci["domains"]["swire_ro"] = {
        "primary_eval_cutoff": sw_cutoffs[-1],
        "strata_ci": domain_ci_block(sw_safe, sw_arts),
    }

    (REPO / "artifacts/eval/w4_strata_bootstrap_ci.json").parent.mkdir(parents=True, exist_ok=True)
    (REPO / "artifacts/eval/w4_strata_bootstrap_ci.json").write_text(json.dumps(out_ci, indent=2), encoding="utf-8")
    print(f"DONE in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
