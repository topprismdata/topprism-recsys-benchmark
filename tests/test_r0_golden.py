import hashlib
import json
from pathlib import Path
import pandas as pd
import pytest

from src.data.window_builder import (
    assign_customer_strata,
    build_daily_baskets,
    build_purchase_labels,
    build_window,
    compute_dataframe_hash,
)
from src.eval.metrics import (
    evaluate_candidate_oracle,
    evaluate_map12,
    evaluate_repeat_explore,
)
from src.governance.ir_bridge import build_recsys_experiment_ir, validate_ir_with_gates


def _sample_events() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"customer_id": "u1", "article_id": "0000000002", "event_date": "2020-08-01", "value": 10.0},
            {"customer_id": "u1", "article_id": "0000000001", "event_date": "2020-08-01", "value": 12.0},
            {"customer_id": "u1", "article_id": "0000000003", "event_date": "2020-08-10", "value": 15.0},
            {"customer_id": "u1", "article_id": "0000000001", "event_date": "2020-08-20", "value": 12.0},
            {"customer_id": "u1", "article_id": "0000000004", "event_date": "2020-08-21", "value": 20.0},
            {"customer_id": "u2", "article_id": "0000000005", "event_date": "2020-08-15", "value": 8.0},
            {"customer_id": "u3", "article_id": "0000000001", "event_date": "2020-07-01", "value": 5.0},
            {"customer_id": "u3", "article_id": "0000000002", "event_date": "2020-07-05", "value": 5.0},
            {"customer_id": "u3", "article_id": "0000000003", "event_date": "2020-07-10", "value": 5.0},
            {"customer_id": "u3", "article_id": "0000000004", "event_date": "2020-07-15", "value": 5.0},
            {"customer_id": "u3", "article_id": "0000000005", "event_date": "2020-07-20", "value": 5.0},
            {"customer_id": "u3", "article_id": "0000000006", "event_date": "2020-08-22", "value": 5.0},
        ]
    )


def test_g01_deterministic_window_hash():
    events = _sample_events()
    w_a = build_window(events, cutoff_date="2020-08-18", horizon_days=7)
    w_b = build_window(events, cutoff_date="2020-08-18", horizon_days=7)
    assert compute_dataframe_hash(w_a["history"]) == compute_dataframe_hash(w_b["history"])
    assert compute_dataframe_hash(w_a["labels"]) == compute_dataframe_hash(w_b["labels"])


def test_g02_no_post_cutoff_leakage_in_history():
    events = _sample_events()
    w = build_window(events, cutoff_date="2020-08-18", horizon_days=7)
    assert (w["history"]["event_date"] <= "2020-08-18").all()
    assert (w["labels"]["event_date"] > "2020-08-18").all()
    assert (w["labels"]["event_date"] <= "2020-08-25").all()


def test_g03_intraday_row_permutation_invariance():
    events = _sample_events()
    shuffled = events.iloc[::-1].reset_index(drop=True)
    b1 = build_daily_baskets(events, cutoff_date="2020-08-18")
    b2 = build_daily_baskets(shuffled, cutoff_date="2020-08-18")
    assert compute_dataframe_hash(b1) == compute_dataframe_hash(b2)


def test_g04_consistent_deduplication_between_labels_and_predictions():
    events = pd.DataFrame(
        [
            {"customer_id": "u1", "article_id": "0000000001", "event_date": "2020-08-20", "value": 10.0},
            {"customer_id": "u1", "article_id": "0000000001", "event_date": "2020-08-22", "value": 10.0},
        ]
    )
    labels = build_purchase_labels(events, cutoff_date="2020-08-18", horizon_days=7)
    assert len(labels) == 1
    preds = pd.DataFrame(
        [
            {"customer_id": "u1", "article_id": "0000000001", "score": 0.9},
            {"customer_id": "u1", "article_id": "0000000001", "score": 0.5},
        ]
    )
    res = evaluate_map12(preds, labels)
    assert res["map12"] == pytest.approx(1.0)


def test_g05_map12_exact_golden_calculation_and_tiebreak():
    labels = pd.DataFrame(
        [
            {"customer_id": "u1", "article_id": "0000000001"},
            {"customer_id": "u1", "article_id": "0000000003"},
        ]
    )
    # Tied score between 0000000002 and 0000000001 -> lexicographical tiebreak puts 0000000001 first (rank 1)
    # and 0000000003 at rank 3 -> P@1 = 1/1, P@3 = 2/3 -> AP = (1 + 2/3) / 2 = 5/6
    preds = pd.DataFrame(
        [
            {"customer_id": "u1", "article_id": "0000000002", "score": 1.0},
            {"customer_id": "u1", "article_id": "0000000001", "score": 1.0},
            {"customer_id": "u1", "article_id": "0000000003", "score": 0.5},
        ]
    )
    res = evaluate_map12(preds, labels)
    assert res["map12"] == pytest.approx(5.0 / 6.0)


def test_g06_non_purchasers_excluded_from_kaggle_map_denominator():
    labels = pd.DataFrame([{"customer_id": "u1", "article_id": "0000000001"}])
    preds = pd.DataFrame(
        [
            {"customer_id": "u1", "article_id": "0000000001", "score": 1.0},
            {"customer_id": "u2", "article_id": "0000000009", "score": 1.0},
        ]
    )
    res = evaluate_map12(preds, labels, all_customers=["u1", "u2"])
    assert res["map12"] == pytest.approx(1.0)
    assert res["evaluated_customers"] == 1
    assert res["all_customer_map12"] == pytest.approx(0.5)


def test_g07_zero_history_customer_still_receives_predictions():
    events = _sample_events()
    strata = assign_customer_strata(events, cutoff_date="2020-08-18", customer_universe=["u1", "u2", "u_cold"])
    cold_row = strata[strata["customer_id"] == "u_cold"].iloc[0]
    assert cold_row["historical_basket_count"] == 0
    assert cold_row["stratum"] == "S0"


def test_g08_strata_assigned_strictly_from_pre_cutoff_history():
    events = _sample_events()
    strata = assign_customer_strata(events, cutoff_date="2020-08-18", customer_universe=["u1", "u2", "u3"])
    s_map = dict(zip(strata["customer_id"], strata["stratum"]))
    # u2 has 1 basket before cutoff -> S0; u1 has 2 baskets (08-01, 08-10) -> S1; u3 has 5 baskets -> S2
    assert s_map["u2"] == "S0"
    assert s_map["u1"] == "S1"
    assert s_map["u3"] == "S2"


def test_g09_repeat_and_exploration_mutually_exclusive_and_exhaustive():
    events = _sample_events()
    w = build_window(events, cutoff_date="2020-08-18", horizon_days=7)
    preds = pd.DataFrame(
        [
            {"customer_id": "u1", "article_id": "0000000001", "score": 0.9},
            {"customer_id": "u1", "article_id": "0000000004", "score": 0.8},
            {"customer_id": "u3", "article_id": "0000000006", "score": 0.7},
        ]
    )
    rep_exp = evaluate_repeat_explore(preds, w["labels"], w["history"])
    assert rep_exp["repeat_label_count"] + rep_exp["explore_label_count"] == rep_exp["total_label_count"]
    assert rep_exp["repeat_label_count"] == 1  # u1 bought 0000000001 before and after
    assert rep_exp["explore_label_count"] == 2  # u1 bought 0000000004 (new), u3 bought 0000000006 (new)


def test_g10_additive_repeat_explore_ap_equals_total_ap():
    events = _sample_events()
    w = build_window(events, cutoff_date="2020-08-18", horizon_days=7)
    preds = pd.DataFrame(
        [
            {"customer_id": "u1", "article_id": "0000000002", "score": 0.95},
            {"customer_id": "u1", "article_id": "0000000001", "score": 0.90},
            {"customer_id": "u1", "article_id": "0000000004", "score": 0.80},
            {"customer_id": "u3", "article_id": "0000000006", "score": 0.70},
        ]
    )
    overall = evaluate_map12(preds, w["labels"])
    rep_exp = evaluate_repeat_explore(preds, w["labels"], w["history"])
    assert rep_exp["map12_repeat"] + rep_exp["map12_explore"] == pytest.approx(overall["map12"])


def test_g11_candidate_oracle_upper_bounds_actual_map():
    labels = pd.DataFrame(
        [
            {"customer_id": "u1", "article_id": "0000000001"},
            {"customer_id": "u1", "article_id": "0000000004"},
        ]
    )
    candidates = pd.DataFrame(
        [
            {"customer_id": "u1", "article_id": "0000000002", "score": 0.9},
            {"customer_id": "u1", "article_id": "0000000001", "score": 0.5},
        ]
    )
    actual = evaluate_map12(candidates, labels)["map12"]
    oracle = evaluate_candidate_oracle(candidates, labels)["oracle_map12"]
    assert oracle >= actual
    assert oracle == pytest.approx(0.5)  # 1 hit out of 2 true items placed at rank 1 -> AP = 1.0 / 2 = 0.5


def test_g12_experiment_ir_010_schema_and_hard_gates_pass(tmp_path: Path):
    contract_path = Path("C:/Users/ghb/ZCodeProject/recsys-benchmark/protocols/recsys-contract-v0.9.json")
    ir_doc = build_recsys_experiment_ir(contract_path=contract_path)
    ir_file = tmp_path / "exp_ir.json"
    ir_file.write_text(json.dumps(ir_doc, indent=2), encoding="utf-8")
    verdict = validate_ir_with_gates(ir_file)
    assert verdict["passed"] is True
    assert verdict["failed_gates"] == []


def test_r0_profiler_produces_five_mandatory_profiles():
    from src.data.profiler import compute_r0_five_profiles

    events = _sample_events()
    profile = compute_r0_five_profiles(
        events,
        validation_cutoffs=["2020-08-05", "2020-08-18"],
        sealed_cutoff="2020-08-25",
        horizon_days=7,
    )
    assert set(profile.keys()) >= {
        "p1_sequence_length",
        "p2_item_sparsity",
        "p3_repeat_vs_exploration",
        "p4_historical_oracle_coverage",
        "p5_window_drift",
        "sealed_w5_labels_touched",
    }
    assert profile["sealed_w5_labels_touched"] is False

    # Verify post-sealed_cutoff labels are strictly untouched
    events_with_w5_noise = pd.concat(
        [
            events,
            pd.DataFrame(
                [
                    {"customer_id": "u1", "article_id": "9999999999", "event_date": "2020-08-28", "value": 99.0},
                ]
            ),
        ],
        ignore_index=True,
    )
    profile_with_noise = compute_r0_five_profiles(
        events_with_w5_noise,
        validation_cutoffs=["2020-08-05", "2020-08-18"],
        sealed_cutoff="2020-08-25",
        horizon_days=7,
    )
    assert profile_with_noise == profile


def test_adapters_canonical_schema(tmp_path: Path):
    from src.data.adapters import (
        load_hm_events,
        load_swire_events,
        stratified_customer_sample,
    )

    raw_swire = pd.DataFrame(
        [
            {"OUTLET": 1001, "ARTICLE": 501, "ORDER_DATE": "2025-03-01 10:20:00", "SETTLE_CASE": 3.5},
            {"OUTLET": 1001, "ARTICLE": 502, "ORDER_DATE": "2025-03-01", "SETTLE_CASE": -1.0},
            {"OUTLET": 1002, "ARTICLE": 503, "ORDER_DATE": "2025-03-02", "SETTLE_CASE": 2.0},
        ]
    )
    csv_path = tmp_path / "swire_sample.csv"
    cache_path = tmp_path / "swire_events.parquet"
    raw_swire.to_csv(csv_path, index=False)

    swire_df = load_swire_events(csv_path, chunksize=2, cache_path=cache_path)
    assert list(swire_df.columns) == ["customer_id", "article_id", "event_date", "value"]
    assert len(swire_df) == 2
    assert swire_df["customer_id"].tolist() == ["1001", "1002"]
    assert swire_df["article_id"].tolist() == ["501", "503"]
    assert swire_df["event_date"].tolist() == ["2025-03-01", "2025-03-02"]
    assert cache_path.exists()

    raw_hm = pd.DataFrame(
        [
            {"customer_id": "abc123", "article_id": 108775015, "t_dat": "2020-09-01", "price": 0.033881},
        ]
    )
    hm_parquet = tmp_path / "hm_sample.parquet"
    raw_hm.to_parquet(hm_parquet, index=False)

    hm_df = load_hm_events(hm_parquet)
    assert list(hm_df.columns) == ["customer_id", "article_id", "event_date", "value"]
    assert hm_df.iloc[0]["article_id"] == "0108775015"
    assert hm_df.iloc[0]["event_date"] == "2020-09-01"
    assert hm_df.iloc[0]["value"] == pytest.approx(0.033881)

    events = _sample_events()
    s_a = stratified_customer_sample(events, cutoff_date="2020-08-18", n_customers=2, seed=42)
    s_b = stratified_customer_sample(events, cutoff_date="2020-08-18", n_customers=2, seed=42)
    assert compute_dataframe_hash(s_a) == compute_dataframe_hash(s_b)
    assert s_a["customer_id"].nunique() == 2
