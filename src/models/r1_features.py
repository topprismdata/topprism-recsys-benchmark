from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from scipy import sparse

from src.models.r1_candidates import _prepare_history, compute_svd_embeddings


B1_FEATURE_COLS: List[str] = [
    # Customer statistical features
    "cust_basket_count",
    "cust_total_events",
    "cust_distinct_items",
    "cust_avg_basket_size",
    "cust_days_since_last",
    "cust_days_tenure",
    "cust_value_sum",
    "cust_value_mean",
    "cust_stratum_code",
    # Item popularity & trend features
    "item_pop_all_count",
    "item_pop_all_customers",
    "item_pop_7d_count",
    "item_pop_28d_count",
    "item_pop_trend_7_28",
    "item_value_mean",
    "item_repurchase_rate",
    # Customer-Item interaction & temporal decay features
    "pair_freq",
    "pair_value_sum",
    "pair_value_mean",
    "pair_days_since_last",
    "pair_days_since_first",
    "up_order_strike",
    "time_decay_score",
    "reorder_ratio",
    "pair_last_basket_gap",
    "gap_mean",
    "gap_std",
    "cycle_due_ratio",
    "in_last_1_basket",
    "in_last_3_baskets",
    "cooccur_score",
    "cooccur_max_score",
    # Non-embedding recall provenance features
    "score_repeat",
    "rank_repeat",
    "score_time_decay",
    "rank_time_decay",
    "score_popular",
    "rank_popular",
    "score_item_cooc",
    "rank_item_cooc",
]

SVD_FEATURE_COLS: List[str] = [
    "svd_dot",
    "svd_cosine",
    "svd_recent_cosine",
    "svd_user_norm",
    "svd_item_norm",
    "score_embedding",
    "rank_embedding",
    "svd_prod_0",
    "svd_prod_1",
    "svd_prod_2",
    "svd_prod_3",
]

B2_FEATURE_COLS: List[str] = B1_FEATURE_COLS + SVD_FEATURE_COLS


def _compute_cooccurrence_features(
    hist: pd.DataFrame,
    cand_df: pd.DataFrame,
    cust_baskets: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    n_cands = len(cand_df)
    if hist.empty or n_cands == 0:
        return np.zeros(n_cands, dtype=np.float32), np.zeros(n_cands, dtype=np.float32)

    articles = sorted(hist["article_id"].unique().tolist())
    n_items = len(articles)
    if n_items <= 1:
        return np.zeros(n_cands, dtype=np.float32), np.zeros(n_cands, dtype=np.float32)

    art_to_idx = {a: idx for idx, a in enumerate(articles)}
    hist_idx = hist["article_id"].map(art_to_idx).to_numpy(dtype=np.int32)

    basket_codes, _ = pd.factorize(hist["customer_id"] + "|" + hist["event_date"], sort=False)
    n_baskets = int(basket_codes.max()) + 1 if len(basket_codes) > 0 else 0
    if n_baskets == 0:
        return np.zeros(n_cands, dtype=np.float32), np.zeros(n_cands, dtype=np.float32)

    basket_sizes = np.bincount(basket_codes, minlength=n_baskets).astype(np.float32)
    basket_weights = 1.0 / np.sqrt(np.maximum(basket_sizes[basket_codes], 1.0))

    mat_b = sparse.csr_matrix(
        (basket_weights, (basket_codes, hist_idx)),
        shape=(n_baskets, n_items),
        dtype=np.float32,
    )
    cooc = (mat_b.T @ mat_b).tocsr()
    cooc.setdiag(0.0)
    cooc.eliminate_zeros()

    item_df = np.bincount(hist_idx, minlength=n_items).astype(np.float32)
    idf = 1.0 / np.power(1.0 + np.log1p(item_df), 0.25)
    cooc = (cooc @ sparse.diags(idf, dtype=np.float32)).tocsr()

    recent_baskets = cust_baskets.loc[cust_baskets["rev_idx"] < 3]
    sub_recent = hist[["customer_id", "article_id", "days_ago"]].merge(
        recent_baskets, on=["customer_id", "days_ago"], how="inner"
    )
    if sub_recent.empty:
        return np.zeros(n_cands, dtype=np.float32), np.zeros(n_cands, dtype=np.float32)

    sub_recent["trig_w"] = np.power(2.0, -sub_recent["rev_idx"].to_numpy(dtype=np.float32))
    trig_agg = sub_recent.groupby(["customer_id", "article_id"], as_index=False)["trig_w"].sum()

    active_users = sorted(trig_agg["customer_id"].unique().tolist())
    u_to_idx = {u: idx for idx, u in enumerate(active_users)}
    u_indices = trig_agg["customer_id"].map(u_to_idx).to_numpy(dtype=np.int32)
    i_indices = trig_agg["article_id"].map(art_to_idx).to_numpy(dtype=np.int32)
    trig_vals = trig_agg["trig_w"].to_numpy(dtype=np.float32)

    user_trig = sparse.csr_matrix(
        (trig_vals, (u_indices, i_indices)),
        shape=(len(active_users), n_items),
        dtype=np.float32,
    )

    cand_u = cand_df["customer_id"].map(u_to_idx).fillna(-1).to_numpy(dtype=np.int32)
    cand_i = cand_df["article_id"].map(art_to_idx).fillna(-1).to_numpy(dtype=np.int32)
    valid_mask = (cand_u >= 0) & (cand_i >= 0)

    cooc_sum = np.zeros(n_cands, dtype=np.float32)
    cooc_max = np.zeros(n_cands, dtype=np.float32)
    if not valid_mask.any():
        return cooc_sum, cooc_max

    valid_idx = np.flatnonzero(valid_mask)
    cooc_t = cooc.T.tocsr()
    batch_size = 50000
    for start in range(0, len(valid_idx), batch_size):
        b_idx = valid_idx[start : start + batch_size]
        u_sub = user_trig[cand_u[b_idx]]
        w_sub = cooc_t[cand_i[b_idx]]
        prod = u_sub.multiply(w_sub)
        cooc_sum[b_idx] = np.asarray(prod.sum(axis=1)).reshape(-1).astype(np.float32)
        if prod.nnz > 0:
            cooc_max[b_idx] = np.asarray(prod.max(axis=1).todense()).reshape(-1).astype(np.float32)

    return cooc_sum, cooc_max


def build_r1_feature_matrix(
    events: pd.DataFrame,
    candidates: pd.DataFrame,
    cutoff_date: str,
    half_life_days: float = 14.0,
    embedding_dim: int = 32,
    random_state: int = 42,
    svd_pack: Optional[Dict[str, object]] = None,
) -> pd.DataFrame:
    """Build strictly zero-leakage R1 tabular + decay + co-occurrence + 32-dim SVD feature matrix."""
    hist = _prepare_history(events, cutoff_date=cutoff_date)
    df = candidates.copy().reset_index(drop=True)
    df["customer_id"] = df["customer_id"].astype(str)
    df["article_id"] = df["article_id"].astype(str)
    df["cutoff_date"] = str(cutoff_date)

    # Ensure provenance score/rank columns exist even if caller passed minimal candidate DataFrame
    for suffix, src_name in [
        ("repeat", "REPEAT"),
        ("time_decay", "TIME_DECAY"),
        ("popular", "POPULAR"),
        ("item_cooc", "ITEM_COOC"),
        ("embedding", "EMBEDDING"),
    ]:
        score_col = f"score_{suffix}"
        rank_col = f"rank_{suffix}"
        if score_col not in df.columns:
            if "source" in df.columns and "source_score" in df.columns:
                df[score_col] = np.where(df["source"] == src_name, df["source_score"].astype(float), 0.0)
            else:
                df[score_col] = 0.0
        if rank_col not in df.columns:
            if "source" in df.columns and "source_rank" in df.columns:
                df[rank_col] = np.where(df["source"] == src_name, df["source_rank"].astype(int), 999)
            else:
                df[rank_col] = 999

    if hist.empty or df.empty:
        for col in B2_FEATURE_COLS:
            if col not in df.columns:
                df[col] = 0.0
        return df

    # 1. Customer-level reverse basket indexing for up_order_strike
    cust_baskets = (
        hist[["customer_id", "days_ago"]]
        .drop_duplicates()
        .sort_values(["customer_id", "days_ago"], ascending=[True, True])
        .reset_index(drop=True)
    )
    cust_baskets["rev_idx"] = cust_baskets.groupby("customer_id", sort=False).cumcount()
    hist_with_rev = hist.merge(cust_baskets, on=["customer_id", "days_ago"], how="left")
    hist_with_rev["strike_w"] = np.power(
        2.0, -np.minimum(hist_with_rev["rev_idx"].to_numpy(dtype=np.float64), 50.0)
    )
    hist_with_rev["decay_w"] = np.power(
        2.0, -hist_with_rev["days_ago"].to_numpy(dtype=np.float64) / max(half_life_days, 1e-3)
    )

    # 2. Customer statistical features
    cust_agg = (
        hist.groupby("customer_id", as_index=False)
        .agg(
            cust_basket_count=("days_ago", "nunique"),
            cust_total_events=("article_id", "count"),
            cust_distinct_items=("article_id", "nunique"),
            cust_days_since_last=("days_ago", "min"),
            cust_days_tenure=("days_ago", "max"),
            cust_value_sum=("value", "sum"),
            cust_value_mean=("value", "mean"),
        )
    )
    cust_agg["cust_avg_basket_size"] = cust_agg["cust_total_events"] / np.maximum(
        cust_agg["cust_basket_count"], 1
    )
    b_cnt = cust_agg["cust_basket_count"].to_numpy(dtype=int)
    cust_agg["cust_stratum_code"] = np.where(b_cnt <= 1, 0, np.where(b_cnt <= 4, 1, 2))

    df = df.merge(cust_agg, on="customer_id", how="left")
    df["cust_basket_count"] = df["cust_basket_count"].fillna(0.0)
    df["cust_total_events"] = df["cust_total_events"].fillna(0.0)
    df["cust_distinct_items"] = df["cust_distinct_items"].fillna(0.0)
    df["cust_avg_basket_size"] = df["cust_avg_basket_size"].fillna(0.0)
    df["cust_days_since_last"] = df["cust_days_since_last"].fillna(999.0)
    df["cust_days_tenure"] = df["cust_days_tenure"].fillna(0.0)
    df["cust_value_sum"] = df["cust_value_sum"].fillna(0.0)
    df["cust_value_mean"] = df["cust_value_mean"].fillna(0.0)
    df["cust_stratum_code"] = df["cust_stratum_code"].fillna(0.0)

    # 3. Item popularity & trend features
    item_all = (
        hist.groupby("article_id", as_index=False)
        .agg(
            item_pop_all_count=("customer_id", "count"),
            item_pop_all_customers=("customer_id", "nunique"),
            item_value_mean=("value", "mean"),
        )
    )
    item_7d = (
        hist.loc[hist["days_ago"] <= 7]
        .groupby("article_id", as_index=False)
        .size()
        .rename(columns={"size": "item_pop_7d_count"})
    )
    item_28d = (
        hist.loc[hist["days_ago"] <= 28]
        .groupby("article_id", as_index=False)
        .size()
        .rename(columns={"size": "item_pop_28d_count"})
    )
    # Item repurchase rate across buyers
    ui_counts = hist.groupby(["article_id", "customer_id"], as_index=False).size()
    ui_counts["is_rep"] = (ui_counts["size"] >= 2).astype(float)
    item_rep = ui_counts.groupby("article_id", as_index=False)["is_rep"].mean().rename(
        columns={"is_rep": "item_repurchase_rate"}
    )

    item_feat = (
        item_all.merge(item_7d, on="article_id", how="left")
        .merge(item_28d, on="article_id", how="left")
        .merge(item_rep, on="article_id", how="left")
    )
    item_feat["item_pop_7d_count"] = item_feat["item_pop_7d_count"].fillna(0.0)
    item_feat["item_pop_28d_count"] = item_feat["item_pop_28d_count"].fillna(0.0)
    item_feat["item_repurchase_rate"] = item_feat["item_repurchase_rate"].fillna(0.0)
    item_feat["item_pop_trend_7_28"] = item_feat["item_pop_7d_count"] / np.maximum(
        item_feat["item_pop_28d_count"] / 4.0, 0.25
    )

    df = df.merge(item_feat, on="article_id", how="left")
    for col in [
        "item_pop_all_count",
        "item_pop_all_customers",
        "item_pop_7d_count",
        "item_pop_28d_count",
        "item_pop_trend_7_28",
        "item_value_mean",
        "item_repurchase_rate",
    ]:
        df[col] = df[col].fillna(0.0)

    # 4. Customer-Item interaction, up_order_strike, and inter-purchase gap features
    pair_agg = (
        hist_with_rev.groupby(["customer_id", "article_id"], as_index=False)
        .agg(
            pair_freq=("days_ago", "count"),
            pair_value_sum=("value", "sum"),
            pair_value_mean=("value", "mean"),
            pair_days_since_last=("days_ago", "min"),
            pair_days_since_first=("days_ago", "max"),
            up_order_strike=("strike_w", "sum"),
            time_decay_score=("decay_w", "sum"),
            pair_last_basket_gap=("rev_idx", "min"),
        )
    )

    # Vectorized inter-purchase gap_mean (telescopic exact) and gap_std
    pair_agg["gap_mean"] = np.where(
        pair_agg["pair_freq"] >= 2,
        (pair_agg["pair_days_since_first"] - pair_agg["pair_days_since_last"])
        / np.maximum(pair_agg["pair_freq"] - 1, 1),
        99.0,
    )

    sorted_hist = hist_with_rev[["customer_id", "article_id", "days_ago"]].sort_values(
        ["customer_id", "article_id", "days_ago"], ascending=[True, True, False]
    )
    same_pair = (
        (sorted_hist["customer_id"] == sorted_hist["customer_id"].shift(1))
        & (sorted_hist["article_id"] == sorted_hist["article_id"].shift(1))
    )
    gaps_df = sorted_hist.loc[same_pair, ["customer_id", "article_id"]].copy()
    gaps_df["gap"] = (
        sorted_hist["days_ago"].shift(1).loc[same_pair] - sorted_hist.loc[same_pair, "days_ago"]
    ).astype(float)
    if not gaps_df.empty:
        gap_std_df = (
            gaps_df.groupby(["customer_id", "article_id"], as_index=False)["gap"]
            .std(ddof=0)
            .rename(columns={"gap": "gap_std"})
        )
        pair_agg = pair_agg.merge(gap_std_df, on=["customer_id", "article_id"], how="left")
        pair_agg["gap_std"] = pair_agg["gap_std"].fillna(0.0)
    else:
        pair_agg["gap_std"] = 0.0

    df = df.merge(pair_agg, on=["customer_id", "article_id"], how="left")
    df["pair_freq"] = df["pair_freq"].fillna(0.0)
    df["pair_value_sum"] = df["pair_value_sum"].fillna(0.0)
    df["pair_value_mean"] = df["pair_value_mean"].fillna(0.0)
    df["pair_days_since_last"] = df["pair_days_since_last"].fillna(999.0)
    df["pair_days_since_first"] = df["pair_days_since_first"].fillna(999.0)
    df["up_order_strike"] = df["up_order_strike"].fillna(0.0)
    df["time_decay_score"] = df["time_decay_score"].fillna(0.0)
    df["pair_last_basket_gap"] = df["pair_last_basket_gap"].fillna(99.0)
    df["gap_mean"] = df["gap_mean"].fillna(99.0)
    df["gap_std"] = df["gap_std"].fillna(0.0)

    df["reorder_ratio"] = df["pair_freq"] / np.maximum(df["cust_basket_count"], 1.0)
    df["cycle_due_ratio"] = np.where(
        df["pair_freq"] >= 2,
        df["pair_days_since_last"] / np.maximum(df["gap_mean"], 1.0),
        0.0,
    )
    df["in_last_1_basket"] = (df["pair_last_basket_gap"] == 0).astype(float)
    df["in_last_3_baskets"] = (df["pair_last_basket_gap"] < 3).astype(float)

    # 5. Co-occurrence score features
    cooc_sum, cooc_max = _compute_cooccurrence_features(hist, df, cust_baskets)
    df["cooccur_score"] = cooc_sum.astype(float)
    df["cooccur_max_score"] = cooc_max.astype(float)

    # 6. 32-dim TruncatedSVD user-item behavior embedding features
    if svd_pack is None:
        svd_pack = compute_svd_embeddings(
            hist,
            embedding_dim=embedding_dim,
            half_life_days=half_life_days,
            random_state=random_state,
        )

    u_to_idx: Dict[str, int] = svd_pack["u_to_idx"]  # type: ignore[assignment]
    i_to_idx: Dict[str, int] = svd_pack["i_to_idx"]  # type: ignore[assignment]
    user_emb: np.ndarray = svd_pack["user_emb"]  # type: ignore[assignment]
    item_emb: np.ndarray = svd_pack["item_emb"]  # type: ignore[assignment]
    user_emb_norm: np.ndarray = svd_pack["user_emb_norm"]  # type: ignore[assignment]
    item_emb_norm: np.ndarray = svd_pack["item_emb_norm"]  # type: ignore[assignment]

    # Also compute recent-3-baskets user embedding for svd_recent_cosine
    recent_rows = hist_with_rev.loc[hist_with_rev["rev_idx"] < 3, ["customer_id", "article_id", "strike_w"]]
    if not recent_rows.empty and len(u_to_idx) > 0 and len(i_to_idx) > 0:
        r_agg = recent_rows.groupby(["customer_id", "article_id"], as_index=False)["strike_w"].sum()
        ru = r_agg["customer_id"].map(u_to_idx).to_numpy(dtype=np.int32)
        ri = r_agg["article_id"].map(i_to_idx).to_numpy(dtype=np.int32)
        rw = r_agg["strike_w"].to_numpy(dtype=np.float32)
        r_mat = sparse.csr_matrix((rw, (ru, ri)), shape=(len(u_to_idx), len(i_to_idx)), dtype=np.float32)
        r_emb = r_mat @ item_emb_norm
        r_norms = np.linalg.norm(r_emb, axis=1, keepdims=True)
        user_recent_norm = r_emb / np.maximum(r_norms, 1e-8)
    else:
        user_recent_norm = user_emb_norm

    n_rows = len(df)
    cand_u_idx = df["customer_id"].map(u_to_idx).fillna(-1).to_numpy(dtype=np.int32)
    cand_i_idx = df["article_id"].map(i_to_idx).fillna(-1).to_numpy(dtype=np.int32)

    valid_u = cand_u_idx >= 0
    valid_i = cand_i_idx >= 0
    valid_ui = valid_u & valid_i

    svd_dot = np.zeros(n_rows, dtype=np.float32)
    svd_cos = np.zeros(n_rows, dtype=np.float32)
    svd_rec_cos = np.zeros(n_rows, dtype=np.float32)
    svd_u_norm = np.zeros(n_rows, dtype=np.float32)
    svd_i_norm = np.zeros(n_rows, dtype=np.float32)
    svd_prod_8 = np.zeros((n_rows, 8), dtype=np.float32)
    svd_item_8 = np.zeros((n_rows, 8), dtype=np.float32)

    if valid_u.any():
        vu_idx = np.flatnonzero(valid_u)
        svd_u_norm[vu_idx] = np.linalg.norm(user_emb[cand_u_idx[vu_idx]], axis=1)

    if valid_i.any():
        vi_idx = np.flatnonzero(valid_i)
        svd_i_norm[vi_idx] = np.linalg.norm(item_emb[cand_i_idx[vi_idx]], axis=1)
        svd_item_8[vi_idx] = item_emb_norm[cand_i_idx[vi_idx], :8]

    if valid_ui.any():
        vui_idx = np.flatnonzero(valid_ui)
        u_sub = user_emb[cand_u_idx[vui_idx]]
        i_sub = item_emb[cand_i_idx[vui_idx]]
        un_sub = user_emb_norm[cand_u_idx[vui_idx]]
        in_sub = item_emb_norm[cand_i_idx[vui_idx]]
        urn_sub = user_recent_norm[cand_u_idx[vui_idx]]

        svd_dot[vui_idx] = np.sum(u_sub * i_sub, axis=1)
        svd_cos[vui_idx] = np.sum(un_sub * in_sub, axis=1)
        svd_rec_cos[vui_idx] = np.sum(urn_sub * in_sub, axis=1)
        svd_prod_8[vui_idx] = un_sub[:, :8] * in_sub[:, :8]

    df["svd_dot"] = svd_dot.astype(float)
    df["svd_cosine"] = svd_cos.astype(float)
    df["svd_recent_cosine"] = svd_rec_cos.astype(float)
    df["svd_user_norm"] = svd_u_norm.astype(float)
    df["svd_item_norm"] = svd_i_norm.astype(float)
    for d in range(8):
        df[f"svd_prod_{d}"] = svd_prod_8[:, d].astype(float)
        df[f"svd_item_{d}"] = svd_item_8[:, d].astype(float)
    for col in B2_FEATURE_COLS:
        df[col] = df[col].astype(float).fillna(0.0)

    return df
