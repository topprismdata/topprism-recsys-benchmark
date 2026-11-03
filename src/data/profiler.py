from datetime import datetime, timedelta
from typing import Any, Dict, List, Sequence, Set, Tuple
import numpy as np
import pandas as pd

from src.data.window_builder import assign_customer_strata, build_window
from src.eval.metrics import compute_user_ap12_components


def _quantiles_and_stats(series: pd.Series) -> Dict[str, float]:
    if series.empty:
        return {"p50": 0.0, "p90": 0.0, "p99": 0.0, "mean": 0.0, "max": 0.0}
    arr = series.to_numpy(dtype=float)
    q50, q90, q99 = np.percentile(arr, [50, 90, 99])
    return {
        "p50": float(q50),
        "p90": float(q90),
        "p99": float(q99),
        "mean": float(np.mean(arr)),
        "max": float(np.max(arr)),
    }


def _top_k_popular_articles(df: pd.DataFrame, k: int = 100) -> List[str]:
    if df.empty:
        return []
    counts = (
        df.groupby("article_id", as_index=False)
        .size()
        .rename(columns={"size": "cnt"})
        .sort_values(by=["cnt", "article_id"], ascending=[False, True], kind="mergesort")
    )
    return counts["article_id"].head(k).astype(str).tolist()


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return float(len(a & b) / len(union))


def derive_last_five_weekly_cutoffs(events: pd.DataFrame, horizon_days: int = 7) -> Dict[str, Any]:
    """Derive W1..W4 validation cutoffs and W5 sealed cutoff from the max event_date."""
    max_date_str = str(events["event_date"].max())
    min_date_str = str(events["event_date"].min())
    dt_max = datetime.strptime(max_date_str, "%Y-%m-%d")
    # W5 cutoff is T_max - horizon_days; W1..W4 step backward by horizon_days each
    cutoffs = [
        (dt_max - timedelta(days=horizon_days * k)).strftime("%Y-%m-%d")
        for k in (5, 4, 3, 2, 1)
    ]
    return {
        "min_event_date": min_date_str,
        "max_event_date": max_date_str,
        "validation_cutoffs": cutoffs[:4],
        "sealed_cutoff": cutoffs[4],
    }


def compute_r0_five_profiles(
    events: pd.DataFrame,
    validation_cutoffs: Sequence[str],
    sealed_cutoff: str,
    horizon_days: int = 7,
    k_top: int = 12,
    top_popular_k: int = 100,
) -> Dict[str, Any]:
    """Compute the 5 mandatory R0 profiles without touching post-sealed_cutoff labels."""
    val_cutoffs = list(validation_cutoffs)
    for c in val_cutoffs:
        dt_end = (datetime.strptime(c, "%Y-%m-%d") + timedelta(days=horizon_days)).strftime("%Y-%m-%d")
        if dt_end > sealed_cutoff:
            raise ValueError(
                f"Validation window for cutoff {c} + {horizon_days}d ({dt_end}) exceeds sealed_cutoff {sealed_cutoff}"
            )

    # Strictly truncate events at sealed_cutoff so W5 post-cutoff labels can never be touched.
    safe_events = events.loc[events["event_date"] <= sealed_cutoff, ["customer_id", "article_id", "event_date", "value"]].copy()
    safe_events["customer_id"] = safe_events["customer_id"].astype(str)
    safe_events["article_id"] = safe_events["article_id"].astype(str)
    safe_events["event_date"] = safe_events["event_date"].astype(str)
    safe_events["value"] = safe_events["value"].astype(float)

    # P1: Sequence Length & Strata Distribution (at sealed_cutoff pre-cutoff history)
    row_counts = safe_events.groupby("customer_id").size() if not safe_events.empty else pd.Series(dtype=float)
    basket_counts = (
        safe_events.groupby("customer_id")["event_date"].nunique()
        if not safe_events.empty
        else pd.Series(dtype=float)
    )
    strata_df = assign_customer_strata(safe_events, cutoff_date=sealed_cutoff)
    total_cust = len(strata_df)
    strata_vc = strata_df["stratum"].value_counts().to_dict() if total_cust > 0 else {}
    s0_cnt = int(strata_vc.get("S0", 0))
    s1_cnt = int(strata_vc.get("S1", 0))
    s2_cnt = int(strata_vc.get("S2", 0))

    p1_sequence_length = {
        "total_events": int(len(safe_events)),
        "total_customers": int(total_cust),
        "purchase_row_counts": _quantiles_and_stats(row_counts),
        "daily_basket_counts": _quantiles_and_stats(basket_counts),
        "strata_counts": {"S0": s0_cnt, "S1": s1_cnt, "S2": s2_cnt},
        "strata_shares": {
            "S0": float(s0_cnt / total_cust) if total_cust > 0 else 0.0,
            "S1": float(s1_cnt / total_cust) if total_cust > 0 else 0.0,
            "S2": float(s2_cnt / total_cust) if total_cust > 0 else 0.0,
        },
    }

    # P2: Item Sparsity
    item_counts = safe_events.groupby("article_id").size() if not safe_events.empty else pd.Series(dtype=float)
    total_articles = int(len(item_counts))
    single_purchase_items = int((item_counts == 1).sum()) if total_articles > 0 else 0
    p2_item_sparsity = {
        "total_distinct_articles": total_articles,
        "item_purchase_counts": _quantiles_and_stats(item_counts),
        "single_purchase_item_count": single_purchase_items,
        "single_purchase_item_share": float(single_purchase_items / total_articles) if total_articles > 0 else 0.0,
    }

    # P3: Repeat vs Exploration & P4: Historical / Popular Oracle Coverage across W1..W4
    p3_windows: Dict[str, Any] = {}
    p4_windows: Dict[str, Any] = {}

    total_rep_labels_all = 0
    total_exp_labels_all = 0
    rep_shares = []
    exp_shares = []

    hist_covs = []
    pop_covs = []
    comb_covs = []
    hist_oracles = []
    pop_oracles = []
    comb_oracles = []

    for idx, cutoff in enumerate(val_cutoffs, start=1):
        w_name = f"W{idx}"
        w = build_window(safe_events, cutoff_date=cutoff, horizon_days=horizon_days)
        hist_df = w["history"]
        label_df = w["labels"]

        hist_sets: Dict[str, Set[str]] = (
            hist_df.groupby("customer_id")["article_id"].apply(set).to_dict()
            if not hist_df.empty
            else {}
        )
        truth_sets: Dict[str, Set[str]] = (
            label_df.groupby("customer_id")["article_id"].apply(set).to_dict()
            if not label_df.empty
            else {}
        )
        eval_users = sorted(truth_sets.keys())
        top_pop_list = _top_k_popular_articles(hist_df, k=top_popular_k)
        top_pop_set = set(top_pop_list)

        rep_labels = 0
        exp_labels = 0
        pop_hit_labels = 0
        comb_hit_labels = 0

        cust_with_rep = 0
        cust_with_exp = 0
        cust_rep_only = 0
        cust_exp_only = 0
        cust_mixed = 0

        hist_oracle_aps = []
        pop_oracle_aps = []
        comb_oracle_aps = []
        hist_user_recalls = []
        pop_user_recalls = []
        comb_user_recalls = []

        # Also evaluate actual Popularity@12 baseline on this window
        pop_top12 = top_pop_list[:k_top]
        pop12_aps = []

        for u in eval_users:
            y_u = truth_sets[u]
            h_u = hist_sets.get(u, set())
            c_u = h_u | top_pop_set

            r_hits = len(y_u & h_u)
            e_hits = len(y_u - h_u)
            p_hits = len(y_u & top_pop_set)
            cb_hits = len(y_u & c_u)

            rep_labels += r_hits
            exp_labels += e_hits
            pop_hit_labels += p_hits
            comb_hit_labels += cb_hits

            if r_hits > 0:
                cust_with_rep += 1
            if e_hits > 0:
                cust_with_exp += 1
            if r_hits > 0 and e_hits == 0:
                cust_rep_only += 1
            elif e_hits > 0 and r_hits == 0:
                cust_exp_only += 1
            elif r_hits > 0 and e_hits > 0:
                cust_mixed += 1

            m = min(len(y_u), k_top)
            hist_oracle_aps.append(min(r_hits, k_top) / m)
            pop_oracle_aps.append(min(p_hits, k_top) / m)
            comb_oracle_aps.append(min(cb_hits, k_top) / m)

            hist_user_recalls.append(r_hits / len(y_u))
            pop_user_recalls.append(p_hits / len(y_u))
            comb_user_recalls.append(cb_hits / len(y_u))

            ap_pop12, _, _ = compute_user_ap12_components(pop_top12, y_u, h_u, k=k_top)
            pop12_aps.append(ap_pop12)

        tot_labels = rep_labels + exp_labels
        rep_share = float(rep_labels / tot_labels) if tot_labels > 0 else 0.0
        exp_share = float(exp_labels / tot_labels) if tot_labels > 0 else 0.0

        total_rep_labels_all += rep_labels
        total_exp_labels_all += exp_labels
        rep_shares.append(rep_share)
        exp_shares.append(exp_share)

        p3_windows[w_name] = {
            "cutoff_date": cutoff,
            "evaluated_customers": len(eval_users),
            "repeat_label_count": int(rep_labels),
            "explore_label_count": int(exp_labels),
            "total_label_count": int(tot_labels),
            "repeat_label_share": rep_share,
            "explore_label_share": exp_share,
            "customers_with_repeat": int(cust_with_rep),
            "customers_with_explore": int(cust_with_exp),
            "repeat_only_customers": int(cust_rep_only),
            "explore_only_customers": int(cust_exp_only),
            "mixed_customers": int(cust_mixed),
        }

        h_cov = float(rep_labels / tot_labels) if tot_labels > 0 else 0.0
        p_cov = float(pop_hit_labels / tot_labels) if tot_labels > 0 else 0.0
        c_cov = float(comb_hit_labels / tot_labels) if tot_labels > 0 else 0.0

        h_orc = float(np.mean(hist_oracle_aps)) if hist_oracle_aps else 0.0
        p_orc = float(np.mean(pop_oracle_aps)) if pop_oracle_aps else 0.0
        c_orc = float(np.mean(comb_oracle_aps)) if comb_oracle_aps else 0.0

        hist_covs.append(h_cov)
        pop_covs.append(p_cov)
        comb_covs.append(c_cov)
        hist_oracles.append(h_orc)
        pop_oracles.append(p_orc)
        comb_oracles.append(c_orc)

        p4_windows[w_name] = {
            "cutoff_date": cutoff,
            "evaluated_customers": len(eval_users),
            "historical_pair_coverage": h_cov,
            "historical_user_recall": float(np.mean(hist_user_recalls)) if hist_user_recalls else 0.0,
            "historical_oracle_map12": h_orc,
            "top100_popular_pair_coverage": p_cov,
            "top100_popular_user_recall": float(np.mean(pop_user_recalls)) if pop_user_recalls else 0.0,
            "top100_popular_oracle_map12": p_orc,
            "combined_pair_coverage": c_cov,
            "combined_user_recall": float(np.mean(comb_user_recalls)) if comb_user_recalls else 0.0,
            "combined_oracle_map12": c_orc,
            "popularity_top12_map12": float(np.mean(pop12_aps)) if pop12_aps else 0.0,
        }

    tot_all = total_rep_labels_all + total_exp_labels_all
    p3_repeat_vs_exploration = {
        "per_window": p3_windows,
        "total_repeat_labels": int(total_rep_labels_all),
        "total_explore_labels": int(total_exp_labels_all),
        "overall_repeat_label_share": float(total_rep_labels_all / tot_all) if tot_all > 0 else 0.0,
        "overall_explore_label_share": float(total_exp_labels_all / tot_all) if tot_all > 0 else 0.0,
        "mean_window_repeat_share": float(np.mean(rep_shares)) if rep_shares else 0.0,
        "mean_window_explore_share": float(np.mean(exp_shares)) if exp_shares else 0.0,
    }

    p4_historical_oracle_coverage = {
        "per_window": p4_windows,
        "mean_historical_pair_coverage": float(np.mean(hist_covs)) if hist_covs else 0.0,
        "mean_historical_oracle_map12": float(np.mean(hist_oracles)) if hist_oracles else 0.0,
        "mean_top100_popular_pair_coverage": float(np.mean(pop_covs)) if pop_covs else 0.0,
        "mean_top100_popular_oracle_map12": float(np.mean(pop_oracles)) if pop_oracles else 0.0,
        "mean_combined_pair_coverage": float(np.mean(comb_covs)) if comb_covs else 0.0,
        "mean_combined_oracle_map12": float(np.mean(comb_oracles)) if comb_oracles else 0.0,
    }

    # P5: Window Drift across validation_cutoffs (W1..W4) and pre-cutoff stats of sealed_cutoff (W5)
    all_cutoffs = val_cutoffs + [sealed_cutoff]
    p5_windows: Dict[str, Any] = {}
    recent_top100_by_window: List[Tuple[str, Set[str]]] = []
    cum_top100_by_window: List[Tuple[str, Set[str]]] = []

    for idx, cutoff in enumerate(all_cutoffs, start=1):
        w_name = f"W{idx}" if idx <= len(val_cutoffs) else f"W{idx}_sealed_precutoff"
        dt_cut = datetime.strptime(cutoff, "%Y-%m-%d")
        start_recent = (dt_cut - timedelta(days=horizon_days)).strftime("%Y-%m-%d")

        cum_df = safe_events.loc[safe_events["event_date"] <= cutoff]
        recent_df = safe_events.loc[
            (safe_events["event_date"] > start_recent) & (safe_events["event_date"] <= cutoff)
        ]

        rec_top100 = set(_top_k_popular_articles(recent_df, k=top_popular_k))
        cum_top100 = set(_top_k_popular_articles(cum_df, k=top_popular_k))
        recent_top100_by_window.append((w_name, rec_top100))
        cum_top100_by_window.append((w_name, cum_top100))

        p5_windows[w_name] = {
            "cutoff_date": cutoff,
            "recent_window_start_exclusive": start_recent,
            "recent_active_customers": int(recent_df["customer_id"].nunique()),
            "recent_active_articles": int(recent_df["article_id"].nunique()),
            "recent_event_count": int(len(recent_df)),
            "cumulative_active_customers": int(cum_df["customer_id"].nunique()),
            "cumulative_active_articles": int(cum_df["article_id"].nunique()),
            "cumulative_event_count": int(len(cum_df)),
        }

    consecutive_recent_top100_jaccard: Dict[str, float] = {}
    for i in range(len(recent_top100_by_window) - 1):
        w_a, set_a = recent_top100_by_window[i]
        w_b, set_b = recent_top100_by_window[i + 1]
        consecutive_recent_top100_jaccard[f"{w_a}_to_{w_b}"] = _jaccard(set_a, set_b)

    consecutive_cum_top100_jaccard: Dict[str, float] = {}
    for i in range(len(cum_top100_by_window) - 1):
        w_a, set_a = cum_top100_by_window[i]
        w_b, set_b = cum_top100_by_window[i + 1]
        consecutive_cum_top100_jaccard[f"{w_a}_to_{w_b}"] = _jaccard(set_a, set_b)

    p5_window_drift = {
        "per_window_precutoff": p5_windows,
        "consecutive_recent_top100_jaccard": consecutive_recent_top100_jaccard,
        "consecutive_cumulative_top100_jaccard": consecutive_cum_top100_jaccard,
        "mean_recent_top100_jaccard": (
            float(np.mean(list(consecutive_recent_top100_jaccard.values())))
            if consecutive_recent_top100_jaccard
            else 0.0
        ),
    }

    return {
        "validation_cutoffs": val_cutoffs,
        "sealed_cutoff": sealed_cutoff,
        "horizon_days": int(horizon_days),
        "p1_sequence_length": p1_sequence_length,
        "p2_item_sparsity": p2_item_sparsity,
        "p3_repeat_vs_exploration": p3_repeat_vs_exploration,
        "p4_historical_oracle_coverage": p4_historical_oracle_coverage,
        "p5_window_drift": p5_window_drift,
        "sealed_w5_labels_touched": False,
    }
