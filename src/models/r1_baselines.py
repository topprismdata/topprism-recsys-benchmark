from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.data.window_builder import assign_customer_strata, build_window
from src.eval.metrics import (
    _canonical_top_k,
    _label_sets,
    compute_user_ap12_components,
    evaluate_candidate_oracle,
    evaluate_map12,
    evaluate_repeat_explore,
)
from src.models.r1_candidates import _prepare_history, compute_svd_embeddings, generate_r1_candidates
from src.models.r1_features import B1_FEATURE_COLS, B2_FEATURE_COLS, build_r1_feature_matrix

def _customer_minmax_norm(df: pd.DataFrame, col: str) -> np.ndarray:
    if col not in df.columns or df.empty:
        return np.zeros(len(df), dtype=float)
    vals = df[col].to_numpy(dtype=float)
    vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
    c_min = df.groupby("customer_id", sort=False)[col].transform("min").to_numpy(dtype=float)
    c_max = df.groupby("customer_id", sort=False)[col].transform("max").to_numpy(dtype=float)
    span = c_max - c_min
    return np.where(span > 1e-12, (vals - c_min) / np.maximum(span, 1e-12), np.where(c_max > 0, 1.0, 0.0))


def predict_b0_rule(feature_df: pd.DataFrame) -> pd.DataFrame:
    """Deterministic B0 Rule Ensemble combining normalized REPEAT/TIME_DECAY, ITEM_COOC, and POPULAR scores."""
    if feature_df.empty:
        return pd.DataFrame(columns=["customer_id", "article_id", "score"])

    norm_strike = _customer_minmax_norm(feature_df, "up_order_strike")
    norm_td = _customer_minmax_norm(feature_df, "time_decay_score")
    norm_rep = _customer_minmax_norm(feature_df, "score_repeat")
    norm_cooc = _customer_minmax_norm(feature_df, "cooccur_score")
    norm_pop = _customer_minmax_norm(feature_df, "score_popular")

    score = (
        0.30 * norm_strike
        + 0.25 * norm_td
        + 0.20 * norm_rep
        + 0.15 * norm_cooc
        + 0.10 * norm_pop
    )
    out = pd.DataFrame(
        {
            "customer_id": feature_df["customer_id"].astype(str),
            "article_id": feature_df["article_id"].astype(str),
            "score": score.astype(float),
        }
    )
    if "cutoff_date" in feature_df.columns:
        out["cutoff_date"] = feature_df["cutoff_date"].astype(str)
    return out


class R1LightGBMRanker:
    """Zero-leakage LightGBM baseline ranker for B1 (Tabular+Decay+Cooc) and B2 (B1 + 32d SVD Embeddings)."""

    def __init__(
        self,
        use_embeddings: bool = False,
        n_estimators: int = 250,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        self.use_embeddings = use_embeddings
        self.feature_cols: List[str] = list(B2_FEATURE_COLS if use_embeddings else B1_FEATURE_COLS)
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model_: Optional[lgb.LGBMClassifier] = None
        self._constant_prob: Optional[float] = None

    def _attach_labels(self, feature_df: pd.DataFrame, labels_df: pd.DataFrame) -> np.ndarray:
        if labels_df.empty:
            return np.zeros(len(feature_df), dtype=np.int32)

        join_cols = ["customer_id", "article_id"]
        if "cutoff_date" in feature_df.columns and "cutoff_date" in labels_df.columns:
            join_cols = ["customer_id", "cutoff_date", "article_id"]

        pos = labels_df[join_cols].drop_duplicates().copy()
        for c in join_cols:
            pos[c] = pos[c].astype(str)
        pos["_label"] = 1

        left = feature_df[join_cols].copy()
        for c in join_cols:
            left[c] = left[c].astype(str)

        merged = left.merge(pos, on=join_cols, how="left")
        return merged["_label"].fillna(0).to_numpy(dtype=np.int32)

    def fit(
        self,
        feature_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        active_purchasers_only: bool = True,
    ) -> "R1LightGBMRanker":
        train_df = feature_df
        if active_purchasers_only and not labels_df.empty:
            if "cutoff_date" in feature_df.columns and "cutoff_date" in labels_df.columns:
                active_keys = labels_df[["customer_id", "cutoff_date"]].drop_duplicates().copy()
                active_keys["customer_id"] = active_keys["customer_id"].astype(str)
                active_keys["cutoff_date"] = active_keys["cutoff_date"].astype(str)
                sub = feature_df.merge(active_keys, on=["customer_id", "cutoff_date"], how="inner")
            else:
                active_u = set(labels_df["customer_id"].astype(str).unique())
                sub = feature_df.loc[feature_df["customer_id"].astype(str).isin(active_u)]
            if len(sub) >= 10:
                train_df = sub

        y = self._attach_labels(train_df, labels_df)
        if len(np.unique(y)) < 2 and train_df is not feature_df:
            train_df = feature_df
            y = self._attach_labels(train_df, labels_df)

        x_df = train_df[self.feature_cols].astype(np.float32)

        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            self.model_ = None
            self._constant_prob = float(unique_classes[0]) if len(unique_classes) == 1 else 0.0
            return self

        self._constant_prob = None
        n_rows = len(x_df)
        min_child = min(20, max(1, n_rows // 15))
        num_leaves = min(self.num_leaves, max(2, n_rows // 2))

        self.model_ = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            num_leaves=num_leaves,
            min_child_samples=min_child,
            min_data_in_bin=1 if n_rows < 200 else 3,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=-1,
        )
        self.model_.fit(x_df, y)
        return self

    def predict(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        if feature_df.empty:
            return pd.DataFrame(columns=["customer_id", "article_id", "score"])

        x_df = feature_df[self.feature_cols].astype(np.float32)
        if self.model_ is not None:
            scores = self.model_.predict_proba(x_df)[:, 1].astype(float)
        else:
            fallback = predict_b0_rule(feature_df)["score"].to_numpy(dtype=float)
            scores = fallback
        out = pd.DataFrame(
            {
                "customer_id": feature_df["customer_id"].astype(str),
                "article_id": feature_df["article_id"].astype(str),
                "score": scores,
            }
        )
        if "cutoff_date" in feature_df.columns:
            out["cutoff_date"] = feature_df["cutoff_date"].astype(str)
        return out

    def get_top_features(self, top_n: int = 15) -> List[Dict[str, Any]]:
        if self.model_ is None:
            return []
        gains = self.model_.booster_.feature_importance(importance_type="gain")
        splits = self.model_.booster_.feature_importance(importance_type="split")
        pairs = [
            {
                "feature": col,
                "gain": float(g),
                "split": int(s),
            }
            for col, g, s in zip(self.feature_cols, gains, splits)
        ]
        pairs.sort(key=lambda x: (-x["gain"], x["feature"]))
        return pairs[:top_n]


def evaluate_oracle_curve(
    candidates: pd.DataFrame,
    labels: pd.DataFrame,
    pool_sizes: Sequence[int] = (50, 100, 200),
    k: int = 12,
) -> Dict[str, Dict[str, float]]:
    """Compute Candidate Oracle MAP@12 and Recall@Pool across nested candidate pool sizes K."""
    if candidates.empty:
        return {
            f"K={p}": {"oracle_map12": 0.0, "recall_at_pool": 0.0, "avg_pool_size": 0.0}
            for p in pool_sizes
        }

    ordered = candidates.copy()
    if "pool_rank" in ordered.columns:
        ordered = ordered.sort_values(["customer_id", "pool_rank", "article_id"], kind="mergesort")
    elif "pool_score" in ordered.columns:
        ordered = ordered.sort_values(
            ["customer_id", "pool_score", "article_id"],
            ascending=[True, False, True],
            kind="mergesort",
        )

    results: Dict[str, Dict[str, float]] = {}
    for p in pool_sizes:
        sub = ordered.groupby("customer_id", sort=False).head(int(p))
        ora = evaluate_candidate_oracle(sub, labels, k=k)
        avg_sz = float(sub.groupby("customer_id").size().mean()) if not sub.empty else 0.0
        results[f"K={p}"] = {
            "oracle_map12": float(ora["oracle_map12"]),
            "recall_at_pool": float(ora["recall_at_pool"]),
            "avg_pool_size": avg_sz,
        }
    return results


def compute_paired_bootstrap_ci(
    preds_a: pd.DataFrame,
    preds_b: pd.DataFrame,
    labels: pd.DataFrame,
    k: int = 12,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Customer-level paired bootstrap 95% confidence interval for MAP@12(B) - MAP@12(A)."""
    truth = _label_sets(labels)
    eval_users = sorted(truth.keys())
    if not eval_users:
        return {
            "mean_a": 0.0,
            "mean_b": 0.0,
            "mean_diff": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "p_value_two_sided": 1.0,
            "statistically_significant": False,
            "n_evaluated_customers": 0,
        }

    top_a = _canonical_top_k(preds_a, k=k)
    top_b = _canonical_top_k(preds_b, k=k)

    ap_a = np.array(
        [compute_user_ap12_components(top_a.get(u, []), truth[u], k=k)[0] for u in eval_users],
        dtype=np.float64,
    )
    ap_b = np.array(
        [compute_user_ap12_components(top_b.get(u, []), truth[u], k=k)[0] for u in eval_users],
        dtype=np.float64,
    )
    diffs = ap_b - ap_a
    mean_a = float(np.mean(ap_a))
    mean_b = float(np.mean(ap_b))
    mean_diff = float(np.mean(diffs))

    n_users = len(eval_users)
    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, n_users, size=(n_bootstrap, n_users))
    boot_means = diffs[boot_idx].mean(axis=1)

    ci_lower, ci_upper = np.percentile(boot_means, [2.5, 97.5])
    # Two-sided bootstrap p-value
    p_pos = float(np.mean(boot_means <= 0.0))
    p_neg = float(np.mean(boot_means >= 0.0))
    p_val = min(1.0, 2.0 * min(p_pos, p_neg))

    return {
        "mean_a": mean_a,
        "mean_b": mean_b,
        "mean_diff": mean_diff,
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "p_value_two_sided": float(p_val),
        "statistically_significant": bool(ci_lower > 0.0 or ci_upper < 0.0),
        "n_evaluated_customers": int(n_users),
    }


def evaluate_r1_predictions(
    predictions: pd.DataFrame,
    candidates: pd.DataFrame,
    labels: pd.DataFrame,
    history: pd.DataFrame,
    strata_df: pd.DataFrame,
    k: int = 12,
) -> Dict[str, Any]:
    """Comprehensive evaluation: MAP@12, Candidate Oracle bound, Repeat/Explore additive split, and S0/S1/S2 strata."""
    # Restrict predictions strictly to candidate pool pairs
    cand_pairs = candidates[["customer_id", "article_id"]].drop_duplicates().copy()
    cand_pairs["customer_id"] = cand_pairs["customer_id"].astype(str)
    cand_pairs["article_id"] = cand_pairs["article_id"].astype(str)

    preds_clean = predictions[["customer_id", "article_id", "score"]].copy()
    preds_clean["customer_id"] = preds_clean["customer_id"].astype(str)
    preds_clean["article_id"] = preds_clean["article_id"].astype(str)
    preds_in_pool = preds_clean.merge(cand_pairs, on=["customer_id", "article_id"], how="inner")

    all_custs = strata_df["customer_id"].astype(str).unique().tolist() if not strata_df.empty else None
    overall = evaluate_map12(preds_in_pool, labels, all_customers=all_custs, k=k)
    oracle = evaluate_candidate_oracle(cand_pairs, labels, k=k)
    rep_exp = evaluate_repeat_explore(preds_in_pool, labels, history, k=k)

    strata_report: Dict[str, Dict[str, Any]] = {}
    s_map = (
        dict(zip(strata_df["customer_id"].astype(str), strata_df["stratum"].astype(str)))
        if not strata_df.empty
        else {}
    )
    labels_with_s = labels.copy()
    labels_with_s["customer_id"] = labels_with_s["customer_id"].astype(str)
    labels_with_s["_stratum"] = labels_with_s["customer_id"].map(s_map).fillna("S0")

    for s_name in ("S0", "S1", "S2"):
        sub_labels = labels_with_s.loc[labels_with_s["_stratum"] == s_name, ["customer_id", "article_id"]]
        s_cust_all = [u for u, st in s_map.items() if st == s_name]
        if sub_labels.empty:
            strata_report[s_name] = {
                "map12": 0.0,
                "map12_repeat": 0.0,
                "map12_explore": 0.0,
                "evaluated_customers": 0,
                "total_stratum_customers": len(s_cust_all),
            }
        else:
            s_ov = evaluate_map12(preds_in_pool, sub_labels, all_customers=s_cust_all, k=k)
            s_re = evaluate_repeat_explore(preds_in_pool, sub_labels, history, k=k)
            strata_report[s_name] = {
                "map12": float(s_ov["map12"]),
                "all_customer_map12": float(s_ov["all_customer_map12"]),
                "map12_repeat": float(s_re["map12_repeat"]),
                "map12_explore": float(s_re["map12_explore"]),
                "evaluated_customers": int(s_ov["evaluated_customers"]),
                "total_stratum_customers": len(s_cust_all),
            }

    ora_val = float(oracle["oracle_map12"])
    actual_val = float(overall["map12"])
    return {
        "map12": actual_val,
        "evaluated_customers": int(overall["evaluated_customers"]),
        "all_customer_map12": float(overall["all_customer_map12"]),
        "oracle_map12": ora_val,
        "recall_at_pool": float(oracle["recall_at_pool"]),
        "oracle_utilization": float(actual_val / ora_val) if ora_val > 0 else 0.0,
        "oracle_upper_bound_satisfied": bool(actual_val <= ora_val + 1e-9),
        "repeat_explore": rep_exp,
        "strata": strata_report,
    }


def run_r1_benchmark(
    events: pd.DataFrame,
    validation_cutoffs: Sequence[str],
    sealed_cutoff: str,
    horizon_days: int = 7,
    k_pool: int = 100,
    oracle_pool_sizes: Sequence[int] = (50, 100, 200),
    half_life_days: float = 14.0,
    embedding_dim: int = 32,
    n_estimators: int = 250,
    n_bootstrap: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run end-to-end R1 evaluation across W1..W4 without touching W5 post-cutoff labels."""
    val_cutoffs = list(validation_cutoffs)
    for c in val_cutoffs:
        dt_end = (datetime.strptime(c, "%Y-%m-%d") + timedelta(days=horizon_days)).strftime("%Y-%m-%d")
        if dt_end > sealed_cutoff:
            raise ValueError(f"Validation window {c}..{dt_end} exceeds sealed_cutoff {sealed_cutoff}")

    # Strictly truncate events at sealed_cutoff so W5 post-cutoff labels are never touched
    safe_events = events.loc[
        events["event_date"].astype(str) <= str(sealed_cutoff),
        ["customer_id", "article_id", "event_date", "value"],
    ].copy()
    safe_events["customer_id"] = safe_events["customer_id"].astype(str)
    safe_events["article_id"] = safe_events["article_id"].astype(str)
    safe_events["event_date"] = safe_events["event_date"].astype(str)
    safe_events["value"] = safe_events["value"].astype(float)

    max_pool = max(int(k_pool), max(int(p) for p in oracle_pool_sizes))

    window_artifacts: Dict[str, Dict[str, Any]] = {}
    oracle_by_window: Dict[str, Any] = {}
    source_provenance_by_window: Dict[str, Any] = {}
    b0_by_window: Dict[str, Any] = {}

    for idx, cutoff in enumerate(val_cutoffs, start=1):
        w_name = f"W{idx}"
        w = build_window(safe_events, cutoff_date=cutoff, horizon_days=horizon_days)
        strata_df = assign_customer_strata(safe_events, cutoff_date=cutoff)
        target_users = strata_df["customer_id"].astype(str).tolist()

        hist_prep = _prepare_history(safe_events, cutoff_date=cutoff)
        svd_pack = compute_svd_embeddings(
            hist_prep,
            embedding_dim=embedding_dim,
            half_life_days=half_life_days,
            random_state=seed,
        )
        cands_max = generate_r1_candidates(
            safe_events,
            cutoff_date=cutoff,
            customer_ids=target_users,
            k_pool=max_pool,
            per_source_k=max_pool,
            half_life_days=half_life_days,
            embedding_dim=embedding_dim,
            random_state=seed,
            svd_pack=svd_pack,
        )
        oracle_by_window[w_name] = evaluate_oracle_curve(
            cands_max,
            w["labels"],
            pool_sizes=oracle_pool_sizes,
            k=12,
        )

        cands_k = cands_max.loc[cands_max["pool_rank"] <= k_pool].reset_index(drop=True)
        src_vc = cands_k["source"].value_counts().to_dict() if not cands_k.empty else {}
        source_provenance_by_window[w_name] = {
            "cutoff_date": cutoff,
            "k_pool": int(k_pool),
            "total_candidates": int(len(cands_k)),
            "avg_candidates_per_customer": float(len(cands_k) / max(len(target_users), 1)),
            "source_counts": {k_s: int(v_s) for k_s, v_s in src_vc.items()},
        }

        feat_k = build_r1_feature_matrix(
            safe_events,
            cands_k,
            cutoff_date=cutoff,
            half_life_days=half_life_days,
            embedding_dim=embedding_dim,
            random_state=seed,
            svd_pack=svd_pack,
        )
        preds_b0 = predict_b0_rule(feat_k)
        b0_eval = evaluate_r1_predictions(
            preds_b0,
            candidates=cands_k,
            labels=w["labels"],
            history=w["history"],
            strata_df=strata_df,
            k=12,
        )
        b0_by_window[w_name] = b0_eval

        window_artifacts[w_name] = {
            "cutoff_date": cutoff,
            "window": w,
            "strata": strata_df,
            "cands": cands_k,
            "feat": feat_k,
            "preds_b0": preds_b0,
        }

    # Rolling walk-forward evaluation across W2..W4 (train on W1..W_{m-1}, evaluate on W_m)
    walk_forward_windows: Dict[str, Any] = {}
    w4_models: Dict[str, R1LightGBMRanker] = {}

    w_names = [f"W{i}" for i in range(1, len(val_cutoffs) + 1)]
    for target_idx in range(1, len(w_names)):
        target_w = w_names[target_idx]
        train_ws = w_names[:target_idx]

        train_feat = pd.concat([window_artifacts[tw]["feat"] for tw in train_ws], ignore_index=True)
        train_labels = pd.concat([window_artifacts[tw]["window"]["labels"] for tw in train_ws], ignore_index=True)

        target_art = window_artifacts[target_w]
        val_feat = target_art["feat"]
        val_cands = target_art["cands"]
        val_labels = target_art["window"]["labels"]
        val_hist = target_art["window"]["history"]
        val_strata = target_art["strata"]
        preds_b0 = target_art["preds_b0"]

        b1_model = R1LightGBMRanker(
            use_embeddings=False,
            n_estimators=n_estimators,
            random_state=seed,
        ).fit(train_feat, train_labels)
        preds_b1 = b1_model.predict(val_feat)

        b2_model = R1LightGBMRanker(
            use_embeddings=True,
            n_estimators=n_estimators,
            random_state=seed,
        ).fit(train_feat, train_labels)
        preds_b2 = b2_model.predict(val_feat)

        eval_b0 = b0_by_window[target_w]
        eval_b1 = evaluate_r1_predictions(preds_b1, val_cands, val_labels, val_hist, val_strata, k=12)
        eval_b2 = evaluate_r1_predictions(preds_b2, val_cands, val_labels, val_hist, val_strata, k=12)

        ci_b1_vs_b0 = compute_paired_bootstrap_ci(preds_b0, preds_b1, val_labels, k=12, n_bootstrap=n_bootstrap, seed=seed)
        ci_b2_vs_b1 = compute_paired_bootstrap_ci(preds_b1, preds_b2, val_labels, k=12, n_bootstrap=n_bootstrap, seed=seed)
        ci_b2_vs_b0 = compute_paired_bootstrap_ci(preds_b0, preds_b2, val_labels, k=12, n_bootstrap=n_bootstrap, seed=seed)

        walk_forward_windows[target_w] = {
            "cutoff_date": target_art["cutoff_date"],
            "train_windows": train_ws,
            "oracle_curve": oracle_by_window[target_w],
            "models": {
                "B0_rule": eval_b0,
                "B1_strong_lgbm": eval_b1,
                "B2_lgbm_svd_emb": eval_b2,
            },
            "paired_bootstrap_ci": {
                "B1_vs_B0": ci_b1_vs_b0,
                "B2_vs_B1": ci_b2_vs_b1,
                "B2_vs_B0": ci_b2_vs_b0,
            },
        }
        if target_w == w_names[-1]:
            w4_models["B1"] = b1_model
            w4_models["B2"] = b2_model

    # Summarize walk-forward W2..W4 means
    wf_keys = list(walk_forward_windows.keys())
    wf_summary = {}
    for m_key in ("B0_rule", "B1_strong_lgbm", "B2_lgbm_svd_emb"):
        wf_summary[m_key] = {
            "mean_map12": float(np.mean([walk_forward_windows[w]["models"][m_key]["map12"] for w in wf_keys])),
            "mean_map12_repeat": float(
                np.mean([walk_forward_windows[w]["models"][m_key]["repeat_explore"]["map12_repeat"] for w in wf_keys])
            ),
            "mean_map12_explore": float(
                np.mean([walk_forward_windows[w]["models"][m_key]["repeat_explore"]["map12_explore"] for w in wf_keys])
            ),
            "mean_oracle_utilization": float(
                np.mean([walk_forward_windows[w]["models"][m_key]["oracle_utilization"] for w in wf_keys])
            ),
            "strata_mean_map12": {
                s_name: float(
                    np.mean([walk_forward_windows[w]["models"][m_key]["strata"][s_name]["map12"] for w in wf_keys])
                )
                for s_name in ("S0", "S1", "S2")
            },
        }

    # Average Oracle curve across W1..W4
    oracle_mean_w1_w4 = {
        f"K={p}": {
            "mean_oracle_map12": float(np.mean([oracle_by_window[w][f"K={p}"]["oracle_map12"] for w in w_names])),
            "mean_recall_at_pool": float(np.mean([oracle_by_window[w][f"K={p}"]["recall_at_pool"] for w in w_names])),
        }
        for p in oracle_pool_sizes
    }

    last_w = w_names[-1]
    return {
        "contract_version": "0.9",
        "generator_version": "r1_cand_v0.9",
        "validation_cutoffs": val_cutoffs,
        "sealed_cutoff": sealed_cutoff,
        "sealed_w5_labels_touched": False,
        "candidate_pool_size_k": int(k_pool),
        "oracle_curve_by_window": oracle_by_window,
        "oracle_curve_mean_w1_w4": oracle_mean_w1_w4,
        "source_provenance_by_window": source_provenance_by_window,
        "b0_all_validation_windows_w1_w4": {
            "per_window": b0_by_window,
            "mean_map12_w1_w4": float(np.mean([b0_by_window[w]["map12"] for w in w_names])),
        },
        "primary_w4_evaluation": walk_forward_windows[last_w],
        "walk_forward_w2_w4": {
            "per_window": walk_forward_windows,
            "summary_mean": wf_summary,
        },
        "feature_importance_w4": {
            "B1_top15": w4_models["B1"].get_top_features(15) if "B1" in w4_models else [],
            "B2_top15": w4_models["B2"].get_top_features(15) if "B2" in w4_models else [],
        },
    }
