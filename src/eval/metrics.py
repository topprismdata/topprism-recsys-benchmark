from typing import Dict, Iterable, Optional, Set, Tuple
import numpy as np
import pandas as pd


def _canonical_top_k(predictions: pd.DataFrame, k: int = 12) -> Dict[str, list]:
    if predictions.empty:
        return {}
    df = predictions[["customer_id", "article_id", "score"]].copy()
    df["customer_id"] = df["customer_id"].astype(str)
    df["article_id"] = df["article_id"].astype(str)
    df["score"] = df["score"].astype(float)
    # Deduplicate same (customer_id, article_id) keeping max score
    df = df.groupby(["customer_id", "article_id"], as_index=False)["score"].max()
    # Deterministic tie-breaking: descending score, ascending article_id
    df = df.sort_values(
        by=["customer_id", "score", "article_id"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    df = df.groupby("customer_id", sort=False).head(k)
    return df.groupby("customer_id", sort=False)["article_id"].apply(list).to_dict()


def _label_sets(labels: pd.DataFrame) -> Dict[str, Set[str]]:
    if labels.empty:
        return {}
    df = labels[["customer_id", "article_id"]].drop_duplicates()
    df["customer_id"] = df["customer_id"].astype(str)
    df["article_id"] = df["article_id"].astype(str)
    return df.groupby("customer_id")["article_id"].apply(set).to_dict()


def compute_user_ap12_components(
    ranked_items: list,
    true_items: Set[str],
    history_items: Optional[Set[str]] = None,
    k: int = 12,
) -> Tuple[float, float, float]:
    if not true_items:
        return 0.0, 0.0, 0.0
    m = min(len(true_items), k)
    hits = 0.0
    ap_total = 0.0
    ap_repeat = 0.0
    ap_explore = 0.0
    hist = history_items or set()

    for idx, item in enumerate(ranked_items[:k], start=1):
        if item in true_items:
            hits += 1.0
            p_at_k = hits / idx
            contrib = p_at_k / m
            ap_total += contrib
            if item in hist:
                ap_repeat += contrib
            else:
                ap_explore += contrib
    return ap_total, ap_repeat, ap_explore


def evaluate_map12(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    all_customers: Optional[Iterable[str]] = None,
    k: int = 12,
) -> Dict[str, float]:
    truth = _label_sets(labels)
    preds = _canonical_top_k(predictions, k=k)
    eval_users = sorted(truth.keys())
    if not eval_users:
        return {"map12": 0.0, "evaluated_customers": 0, "all_customer_map12": 0.0}

    user_aps = [
        compute_user_ap12_components(preds.get(u, []), truth[u], k=k)[0]
        for u in eval_users
    ]
    map12 = float(np.mean(user_aps))
    if all_customers is not None:
        universe = sorted({str(u) for u in all_customers})
        denom = max(len(universe), 1)
        all_map12 = float(np.sum(user_aps) / denom)
    else:
        all_map12 = map12

    return {
        "map12": map12,
        "evaluated_customers": len(eval_users),
        "all_customer_map12": all_map12,
    }


def evaluate_repeat_explore(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    history: pd.DataFrame,
    k: int = 12,
) -> Dict[str, float]:
    truth = _label_sets(labels)
    hist_sets = _label_sets(history)
    preds = _canonical_top_k(predictions, k=k)
    eval_users = sorted(truth.keys())

    repeat_label_count = 0
    explore_label_count = 0
    totals, repeats, explores = [], [], []

    for u in eval_users:
        y_u = truth[u]
        h_u = hist_sets.get(u, set())
        repeat_label_count += len(y_u & h_u)
        explore_label_count += len(y_u - h_u)
        ap_t, ap_r, ap_e = compute_user_ap12_components(preds.get(u, []), y_u, h_u, k=k)
        totals.append(ap_t)
        repeats.append(ap_r)
        explores.append(ap_e)

    return {
        "map12_total": float(np.mean(totals)) if totals else 0.0,
        "map12_repeat": float(np.mean(repeats)) if repeats else 0.0,
        "map12_explore": float(np.mean(explores)) if explores else 0.0,
        "repeat_label_count": int(repeat_label_count),
        "explore_label_count": int(explore_label_count),
        "total_label_count": int(repeat_label_count + explore_label_count),
    }


def evaluate_candidate_oracle(
    candidates: pd.DataFrame,
    labels: pd.DataFrame,
    k: int = 12,
) -> Dict[str, float]:
    truth = _label_sets(labels)
    cand_sets = _label_sets(candidates)
    eval_users = sorted(truth.keys())
    if not eval_users:
        return {"oracle_map12": 0.0, "recall_at_pool": 0.0}

    oracle_aps = []
    recalls = []
    for u in eval_users:
        y_u = truth[u]
        c_u = cand_sets.get(u, set())
        hits = len(y_u & c_u)
        m = min(len(y_u), k)
        oracle_aps.append(min(hits, k) / m)
        recalls.append(hits / len(y_u))

    return {
        "oracle_map12": float(np.mean(oracle_aps)),
        "recall_at_pool": float(np.mean(recalls)),
    }
