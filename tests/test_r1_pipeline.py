import numpy as np
import pandas as pd
import pytest

from src.data.window_builder import assign_customer_strata, build_window, compute_dataframe_hash
from src.eval.metrics import evaluate_candidate_oracle, evaluate_map12
from src.models.r1_candidates import (
    CANDIDATE_SOURCES,
    generate_candidates_by_source,
    generate_r1_candidates,
)
from src.models.r1_features import (
    B1_FEATURE_COLS,
    B2_FEATURE_COLS,
     SVD_FEATURE_COLS,
    build_r1_feature_matrix,
)
from src.models.r1_baselines import (
    R1LightGBMRanker,
    compute_paired_bootstrap_ci,
    evaluate_oracle_curve,
    evaluate_r1_predictions,
    predict_b0_rule,
    run_r1_benchmark,
)


def _synthetic_r1_events() -> pd.DataFrame:
    rows = [
        # u1: 3 historical baskets before 2020-08-18: 08-01 (idx 2), 08-10 (idx 1), 08-17 (idx 0)
        # item 001 bought on 08-01 (idx 2 -> 0.25) and 08-17 (idx 0 -> 1.0) => up_order_strike = 1.25
        {"customer_id": "u1", "article_id": "0000000001", "event_date": "2020-08-01", "value": 10.0},
        {"customer_id": "u1", "article_id": "0000000002", "event_date": "2020-08-01", "value": 12.0},
        {"customer_id": "u1", "article_id": "0000000002", "event_date": "2020-08-10", "value": 12.0},
        {"customer_id": "u1", "article_id": "0000000003", "event_date": "2020-08-10", "value": 15.0},
        {"customer_id": "u1", "article_id": "0000000001", "event_date": "2020-08-17", "value": 10.0},
        {"customer_id": "u1", "article_id": "0000000004", "event_date": "2020-08-17", "value": 20.0},
        # u1 post-cutoff label in (2020-08-18, 2020-08-25]
        {"customer_id": "u1", "article_id": "0000000001", "event_date": "2020-08-20", "value": 10.0},
        {"customer_id": "u1", "article_id": "0000000005", "event_date": "2020-08-21", "value": 18.0},
        # u2: co-buys 001, 004, 005, 006 before cutoff
        {"customer_id": "u2", "article_id": "0000000001", "event_date": "2020-08-03", "value": 10.0},
        {"customer_id": "u2", "article_id": "0000000005", "event_date": "2020-08-03", "value": 18.0},
        {"customer_id": "u2", "article_id": "0000000004", "event_date": "2020-08-12", "value": 20.0},
        {"customer_id": "u2", "article_id": "0000000005", "event_date": "2020-08-12", "value": 18.0},
        {"customer_id": "u2", "article_id": "0000000006", "event_date": "2020-08-16", "value": 22.0},
        {"customer_id": "u2", "article_id": "0000000005", "event_date": "2020-08-22", "value": 18.0},
        # u3: 5 baskets (S2 customer)
        {"customer_id": "u3", "article_id": "0000000002", "event_date": "2020-07-15", "value": 12.0},
        {"customer_id": "u3", "article_id": "0000000003", "event_date": "2020-07-22", "value": 15.0},
        {"customer_id": "u3", "article_id": "0000000006", "event_date": "2020-07-29", "value": 22.0},
        {"customer_id": "u3", "article_id": "0000000007", "event_date": "2020-08-05", "value": 9.0},
        {"customer_id": "u3", "article_id": "0000000008", "event_date": "2020-08-14", "value": 11.0},
        {"customer_id": "u3", "article_id": "0000000002", "event_date": "2020-08-19", "value": 12.0},
        # u4: 1 basket (S0 customer)
        {"customer_id": "u4", "article_id": "0000000007", "event_date": "2020-08-15", "value": 9.0},
        {"customer_id": "u4", "article_id": "0000000008", "event_date": "2020-08-15", "value": 11.0},
        {"customer_id": "u4", "article_id": "0000000007", "event_date": "2020-08-23", "value": 9.0},
    ]
    return pd.DataFrame(rows)


def test_r1_candidates_five_sources_dedup_and_provenance():
    events = _synthetic_r1_events()
    cutoff = "2020-08-18"
    target_users = ["u1", "u2", "u3", "u4", "u_cold"]

    raw_sources = generate_candidates_by_source(
        events,
        cutoff_date=cutoff,
        customer_ids=target_users,
        per_source_k=10,
    )
    assert set(CANDIDATE_SOURCES) == {"REPEAT", "TIME_DECAY", "POPULAR", "ITEM_COOC", "EMBEDDING"}
    assert set(raw_sources["source"].unique()) == set(CANDIDATE_SOURCES)

    cands = generate_r1_candidates(
        events,
        cutoff_date=cutoff,
        customer_ids=target_users,
        k_pool=20,
        per_source_k=10,
    )
    required_cols = {
        "customer_id",
        "cutoff_date",
        "article_id",
        "source",
        "source_score",
        "source_rank",
        "generator_version",
    }
    assert required_cols.issubset(set(cands.columns))
    assert (cands["cutoff_date"] == cutoff).all()
    assert set(cands["source"].unique()).issubset(set(CANDIDATE_SOURCES))

    # Strictly deduplicated on (customer_id, cutoff_date, article_id)
    dup_count = cands.duplicated(subset=["customer_id", "cutoff_date", "article_id"]).sum()
    assert dup_count == 0

    # Cold-start customer still receives candidates from POPULAR
    cold_cands = cands[cands["customer_id"] == "u_cold"]
    assert len(cold_cands) > 0
    assert (cold_cands["source"] == "POPULAR").all()


def test_r1_candidates_zero_post_cutoff_leakage():
    events = _synthetic_r1_events()
    cutoff = "2020-08-18"
    cands_clean = generate_r1_candidates(events, cutoff_date=cutoff, k_pool=15)

    future_noise = pd.DataFrame(
        [
            {"customer_id": "u1", "article_id": "9999999999", "event_date": "2020-08-19", "value": 500.0},
            {"customer_id": "u2", "article_id": "9999999999", "event_date": "2020-08-20", "value": 500.0},
            {"customer_id": "u1", "article_id": "0000000008", "event_date": "2020-08-25", "value": 100.0},
        ]
    )
    events_polluted = pd.concat([events, future_noise], ignore_index=True)
    cands_polluted = generate_r1_candidates(
        events_polluted,
        cutoff_date=cutoff,
        customer_ids=sorted(events["customer_id"].unique()),
        k_pool=15,
    )
    assert "9999999999" not in set(cands_polluted["article_id"])
    assert compute_dataframe_hash(cands_clean) == compute_dataframe_hash(cands_polluted)


def test_r1_features_up_order_strike_and_svd_zero_leakage():
    events = _synthetic_r1_events()
    cutoff = "2020-08-18"
    cands = generate_r1_candidates(events, cutoff_date=cutoff, k_pool=15)

    feat_clean = build_r1_feature_matrix(events, cands, cutoff_date=cutoff, embedding_dim=32)

    assert set(B1_FEATURE_COLS).issubset(set(feat_clean.columns))
    assert set(B2_FEATURE_COLS).issubset(set(feat_clean.columns))
    assert set(SVD_FEATURE_COLS).isdisjoint(set(B1_FEATURE_COLS))
    assert set(B2_FEATURE_COLS) == set(B1_FEATURE_COLS) | set(SVD_FEATURE_COLS)

    # Verify golden up_order_strike for (u1, 0000000001): bought in reverse basket 0 (2^0=1.0) and 2 (2^-2=0.25) -> 1.25
    row_u1_item1 = feat_clean[
        (feat_clean["customer_id"] == "u1") & (feat_clean["article_id"] == "0000000001")
    ].iloc[0]
    assert row_u1_item1["up_order_strike"] == pytest.approx(1.25)
    assert row_u1_item1["pair_freq"] == pytest.approx(2.0)

    # Verify zero post-cutoff leakage in features (including SVD and co-occurrence)
    future_noise = pd.DataFrame(
        [
            {"customer_id": "u1", "article_id": "0000000001", "event_date": "2020-08-19", "value": 999.0},
            {"customer_id": "u1", "article_id": "0000000008", "event_date": "2020-08-21", "value": 888.0},
            {"customer_id": "u3", "article_id": "0000000005", "event_date": "2020-08-22", "value": 777.0},
        ]
    )
    events_polluted = pd.concat([events, future_noise], ignore_index=True)
    feat_polluted = build_r1_feature_matrix(events_polluted, cands, cutoff_date=cutoff, embedding_dim=32)

    for col in B2_FEATURE_COLS:
        np.testing.assert_allclose(
            feat_clean[col].to_numpy(dtype=float),
            feat_polluted[col].to_numpy(dtype=float),
            rtol=1e-6,
            atol=1e-8,
            err_msg=f"Feature {col} leaked post-cutoff information",
        )


def test_r1_baselines_b0_b1_b2_and_oracle_upper_bound():
    events = _synthetic_r1_events()
    train_cutoff = "2020-08-12"
    val_cutoff = "2020-08-18"

    # Build train window
    w_train = build_window(events, cutoff_date=train_cutoff, horizon_days=6)
    cands_train = generate_r1_candidates(events, cutoff_date=train_cutoff, k_pool=15)
    feat_train = build_r1_feature_matrix(events, cands_train, cutoff_date=train_cutoff, embedding_dim=32)

    # Build validation window
    w_val = build_window(events, cutoff_date=val_cutoff, horizon_days=7)
    cands_val = generate_r1_candidates(events, cutoff_date=val_cutoff, k_pool=15)
    feat_val = build_r1_feature_matrix(events, cands_val, cutoff_date=val_cutoff, embedding_dim=32)
    strata_val = assign_customer_strata(events, cutoff_date=val_cutoff)

    # B0 Rule
    preds_b0 = predict_b0_rule(feat_val)
    # B1 Strong LightGBM (tabular + decay + cooc, no SVD)
    b1_model = R1LightGBMRanker(use_embeddings=False, n_estimators=25, random_state=42)
    b1_model.fit(feat_train, w_train["labels"])
    preds_b1 = b1_model.predict(feat_val)

    # B2 LightGBM + SVD Behavior Embeddings
    b2_model = R1LightGBMRanker(use_embeddings=True, n_estimators=25, random_state=42)
    b2_model.fit(feat_train, w_train["labels"])
    preds_b2 = b2_model.predict(feat_val)

    oracle = evaluate_candidate_oracle(cands_val, w_val["labels"], k=12)["oracle_map12"]

    for name, preds in [("B0", preds_b0), ("B1", preds_b1), ("B2", preds_b2)]:
        assert not preds["score"].isna().any(), f"{name} produced NaN scores"
        report = evaluate_r1_predictions(
            preds,
            candidates=cands_val,
            labels=w_val["labels"],
            history=w_val["history"],
            strata_df=strata_val,
            k=12,
        )
        assert report["map12"] <= oracle + 1e-9, f"{name} exceeded Candidate Oracle MAP@12"
        assert report["repeat_explore"]["map12_repeat"] + report["repeat_explore"]["map12_explore"] == pytest.approx(
            report["map12"]
        )
        assert set(report["strata"].keys()) == {"S0", "S1", "S2"}

    # Verify Oracle curve monotonicity across pool sizes
    curve = evaluate_oracle_curve(cands_val, w_val["labels"], pool_sizes=(2, 4, 8), k=12)
    assert curve["K=2"]["oracle_map12"] <= curve["K=4"]["oracle_map12"] + 1e-9
    assert curve["K=4"]["oracle_map12"] <= curve["K=8"]["oracle_map12"] + 1e-9

    # Verify paired bootstrap CI helper
    ci = compute_paired_bootstrap_ci(preds_b1, preds_b2, w_val["labels"], k=12, n_bootstrap=200, seed=42)
    assert ci["ci_lower"] <= ci["mean_diff"] <= ci["ci_upper"]


def test_r1_benchmark_runner_and_w5_seal_protection():
    events = _synthetic_r1_events()
    res_clean = run_r1_benchmark(
        events,
        validation_cutoffs=["2020-08-10", "2020-08-17"],
        sealed_cutoff="2020-08-25",
        horizon_days=7,
        k_pool=10,
        oracle_pool_sizes=(4, 8, 12),
        n_estimators=15,
        n_bootstrap=100,
        seed=42,
    )
    assert res_clean["sealed_w5_labels_touched"] is False
    assert "primary_w4_evaluation" in res_clean
    assert "walk_forward_w2_w4" in res_clean

    # Polluting with post-sealed_cutoff (W5) events must leave R1 results bit-for-bit identical
    w5_noise = pd.DataFrame(
        [
            {"customer_id": "u1", "article_id": "0000000099", "event_date": "2020-08-28", "value": 999.0},
            {"customer_id": "u2", "article_id": "0000000001", "event_date": "2020-08-29", "value": 999.0},
        ]
    )
    res_polluted = run_r1_benchmark(
        pd.concat([events, w5_noise], ignore_index=True),
        validation_cutoffs=["2020-08-10", "2020-08-17"],
        sealed_cutoff="2020-08-25",
        horizon_days=7,
        k_pool=10,
        oracle_pool_sizes=(4, 8, 12),
        n_estimators=15,
        n_bootstrap=100,
        seed=42,
    )
    assert res_clean == res_polluted
