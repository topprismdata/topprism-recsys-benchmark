from datetime import datetime
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD


CANDIDATE_SOURCES: Tuple[str, ...] = (
    "REPEAT",
    "TIME_DECAY",
    "POPULAR",
    "ITEM_COOC",
    "EMBEDDING",
)
GENERATOR_VERSION: str = "r1_cand_v0.9"


def _prepare_history(events: pd.DataFrame, cutoff_date: str) -> pd.DataFrame:
    """Strictly filter events to event_date <= cutoff_date and canonicalize daily (customer, article, date)."""
    mask = events["event_date"].astype(str) <= str(cutoff_date)
    sub = events.loc[mask, ["customer_id", "article_id", "event_date", "value"]].copy()
    if sub.empty:
        return pd.DataFrame(columns=["customer_id", "article_id", "event_date", "value", "days_ago"])

    sub["customer_id"] = sub["customer_id"].astype(str)
    sub["article_id"] = sub["article_id"].astype(str)
    sub["event_date"] = sub["event_date"].astype(str)
    sub["value"] = sub["value"].astype(float)

    dedup = (
        sub.groupby(["customer_id", "article_id", "event_date"], as_index=False)["value"]
        .sum()
        .sort_values(["customer_id", "event_date", "article_id"])
        .reset_index(drop=True)
    )
    dt_cut = datetime.strptime(cutoff_date, "%Y-%m-%d")
    unique_dates = dedup["event_date"].unique()
    date_to_days = {
        d: max(0, (dt_cut - datetime.strptime(str(d)[:10], "%Y-%m-%d")).days)
        for d in unique_dates
    }
    dedup["days_ago"] = dedup["event_date"].map(date_to_days).astype(int)
    return dedup


def _build_repeat_and_decay_sources(
    hist: pd.DataFrame,
    target_set: set,
    cutoff_date: str,
    per_source_k: int,
    half_life_days: float,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    empty_cols = [
        "customer_id",
        "cutoff_date",
        "article_id",
        "source",
        "source_score",
        "source_rank",
        "generator_version",
    ]
    if hist.empty:
        empty = pd.DataFrame(columns=empty_cols)
        return empty.copy(), empty.copy()

    sub = hist.loc[hist["customer_id"].isin(target_set)].copy()
    if sub.empty:
        empty = pd.DataFrame(columns=empty_cols)
        return empty.copy(), empty.copy()

    sub["decay_w"] = np.power(2.0, -sub["days_ago"].to_numpy(dtype=float) / max(half_life_days, 1e-3))

    agg = (
        sub.groupby(["customer_id", "article_id"], as_index=False)
        .agg(
            freq=("event_date", "count"),
            total_val=("value", "sum"),
            min_days=("days_ago", "min"),
            decay_score=("decay_w", "sum"),
        )
    )

    # REPEAT: ranked by frequency, then total value, then recency
    rep = agg.copy()
    rep["source_score"] = (
        rep["freq"].astype(float)
        + 1e-3 * np.log1p(np.maximum(rep["total_val"].to_numpy(dtype=float), 0.0))
        + 1e-4 / (1.0 + rep["min_days"].to_numpy(dtype=float))
    )
    rep = rep.sort_values(
        by=["customer_id", "source_score", "article_id"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    rep = rep.groupby("customer_id", sort=False).head(per_source_k).copy()
    rep["source_rank"] = rep.groupby("customer_id", sort=False).cumcount() + 1
    rep["cutoff_date"] = cutoff_date
    rep["source"] = "REPEAT"
    rep["generator_version"] = GENERATOR_VERSION

    # TIME_DECAY: ranked by exponential time decay score
    td = agg.copy()
    td["source_score"] = td["decay_score"].astype(float)
    td = td.sort_values(
        by=["customer_id", "source_score", "article_id"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    td = td.groupby("customer_id", sort=False).head(per_source_k).copy()
    td["source_rank"] = td.groupby("customer_id", sort=False).cumcount() + 1
    td["cutoff_date"] = cutoff_date
    td["source"] = "TIME_DECAY"
    td["generator_version"] = GENERATOR_VERSION

    return rep[empty_cols].reset_index(drop=True), td[empty_cols].reset_index(drop=True)


def _build_popular_source(
    hist: pd.DataFrame,
    target_users: List[str],
    cutoff_date: str,
    per_source_k: int,
) -> pd.DataFrame:
    empty_cols = [
        "customer_id",
        "cutoff_date",
        "article_id",
        "source",
        "source_score",
        "source_rank",
        "generator_version",
    ]
    if hist.empty or not target_users:
        return pd.DataFrame(columns=empty_cols)

    all_cnt = hist.groupby("article_id").size().rename("cnt_all")
    cnt_7d = hist.loc[hist["days_ago"] <= 7].groupby("article_id").size().rename("cnt_7d")
    cnt_28d = hist.loc[hist["days_ago"] <= 28].groupby("article_id").size().rename("cnt_28d")

    pop_df = pd.concat([all_cnt, cnt_7d, cnt_28d], axis=1).fillna(0.0).reset_index()
    max_all = max(float(pop_df["cnt_all"].max()), 1.0)
    max_7d = max(float(pop_df["cnt_7d"].max()), 1.0)
    max_28d = max(float(pop_df["cnt_28d"].max()), 1.0)

    pop_df["source_score"] = (
        0.55 * (pop_df["cnt_7d"] / max_7d)
        + 0.30 * (pop_df["cnt_28d"] / max_28d)
        + 0.15 * (pop_df["cnt_all"] / max_all)
    )
    pop_df = pop_df.sort_values(
        by=["source_score", "article_id"],
        ascending=[False, True],
        kind="mergesort",
    ).head(per_source_k)
    pop_df["source_rank"] = np.arange(1, len(pop_df) + 1, dtype=int)

    n_users = len(target_users)
    n_items = len(pop_df)
    if n_items == 0:
        return pd.DataFrame(columns=empty_cols)

    out = pd.DataFrame(
        {
            "customer_id": np.repeat(target_users, n_items),
            "cutoff_date": cutoff_date,
            "article_id": np.tile(pop_df["article_id"].to_numpy(), n_users),
            "source": "POPULAR",
            "source_score": np.tile(pop_df["source_score"].to_numpy(dtype=float), n_users),
            "source_rank": np.tile(pop_df["source_rank"].to_numpy(dtype=int), n_users),
            "generator_version": GENERATOR_VERSION,
        }
    )
    return out[empty_cols]


def _build_item_cooc_source(
    hist: pd.DataFrame,
    target_users: List[str],
    cutoff_date: str,
    per_source_k: int,
    max_cooc_neighbors: int = 64,
) -> pd.DataFrame:
    empty_cols = [
        "customer_id",
        "cutoff_date",
        "article_id",
        "source",
        "source_score",
        "source_rank",
        "generator_version",
    ]
    if hist.empty or not target_users:
        return pd.DataFrame(columns=empty_cols)

    articles = sorted(hist["article_id"].unique().tolist())
    n_items = len(articles)
    if n_items <= 1:
        return pd.DataFrame(columns=empty_cols)

    art_to_idx = {a: idx for idx, a in enumerate(articles)}
    hist_idx = hist["article_id"].map(art_to_idx).to_numpy(dtype=np.int32)

    # Assign basket IDs per (customer_id, event_date)
    basket_codes, _ = pd.factorize(
        hist["customer_id"] + "|" + hist["event_date"], sort=False
    )
    n_baskets = int(basket_codes.max()) + 1 if len(basket_codes) > 0 else 0
    if n_baskets == 0:
        return pd.DataFrame(columns=empty_cols)

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

    # Moderate IDF scaling so ultra-popular items do not dominate co-occurrence
    item_df = np.bincount(hist_idx, minlength=n_items).astype(np.float32)
    idf = 1.0 / np.power(1.0 + np.log1p(item_df), 0.25)
    cooc = cooc @ sparse.diags(idf, dtype=np.float32)
    cooc = cooc.tocsr()

    # Prune cooc rows to top max_cooc_neighbors if catalog is large
    if n_items > max_cooc_neighbors:
        indptr = [0]
        indices_list = []
        data_list = []
        for i in range(n_items):
            start, end = cooc.indptr[i], cooc.indptr[i + 1]
            row_idx = cooc.indices[start:end]
            row_val = cooc.data[start:end]
            if len(row_val) > max_cooc_neighbors:
                top_loc = np.argpartition(-row_val, max_cooc_neighbors)[:max_cooc_neighbors]
                row_idx = row_idx[top_loc]
                row_val = row_val[top_loc]
            indices_list.append(row_idx)
            data_list.append(row_val)
            indptr.append(indptr[-1] + len(row_idx))
        cooc = sparse.csr_matrix(
            (
                np.concatenate(data_list) if data_list else np.array([], dtype=np.float32),
                np.concatenate(indices_list) if indices_list else np.array([], dtype=np.int32),
                np.array(indptr, dtype=np.int32),
            ),
            shape=(n_items, n_items),
        )

    # Build recent-3-baskets trigger matrix for target customers present in history
    target_set = set(target_users)
    sub = hist.loc[hist["customer_id"].isin(target_set), ["customer_id", "article_id", "days_ago"]].copy()
    if sub.empty:
        return pd.DataFrame(columns=empty_cols)

    cust_baskets = (
        sub[["customer_id", "days_ago"]]
        .drop_duplicates()
        .sort_values(["customer_id", "days_ago"], ascending=[True, True])
    )
    cust_baskets["rev_idx"] = cust_baskets.groupby("customer_id", sort=False).cumcount()
    recent_baskets = cust_baskets.loc[cust_baskets["rev_idx"] < 3]

    sub_recent = sub.merge(recent_baskets, on=["customer_id", "days_ago"], how="inner")
    sub_recent["trig_w"] = np.power(2.0, -sub_recent["rev_idx"].to_numpy(dtype=np.float32))
    trig_agg = (
        sub_recent.groupby(["customer_id", "article_id"], as_index=False)["trig_w"].sum()
    )

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
    scores_mat = (user_trig @ cooc).tocsr()

    # Extract top per_source_k per active customer
    articles_arr = np.array(articles, dtype=object)
    out_u, out_a, out_s, out_r = [], [], [], []

    for u_idx, u_id in enumerate(active_users):
        start, end = scores_mat.indptr[u_idx], scores_mat.indptr[u_idx + 1]
        if start == end:
            continue
        c_idx = scores_mat.indices[start:end]
        c_val = scores_mat.data[start:end]
        pos_mask = c_val > 0
        if not pos_mask.any():
            continue
        c_idx = c_idx[pos_mask]
        c_val = c_val[pos_mask]
        c_art = articles_arr[c_idx]
        # Deterministic sort: (-score, article_id)
        order = np.lexsort((c_art, -c_val))[:per_source_k]
        k_take = len(order)
        out_u.extend([u_id] * k_take)
        out_a.extend(c_art[order].tolist())
        out_s.extend(c_val[order].astype(float).tolist())
        out_r.extend(range(1, k_take + 1))

    if not out_u:
        return pd.DataFrame(columns=empty_cols)

    return pd.DataFrame(
        {
            "customer_id": out_u,
            "cutoff_date": cutoff_date,
            "article_id": out_a,
            "source": "ITEM_COOC",
            "source_score": out_s,
            "source_rank": out_r,
            "generator_version": GENERATOR_VERSION,
        }
    )[empty_cols]


def compute_svd_embeddings(
    hist: pd.DataFrame,
    embedding_dim: int = 32,
    half_life_days: float = 14.0,
    random_state: int = 42,
) -> Dict[str, object]:
    """Fit zero-leakage TruncatedSVD item and user embeddings strictly on pre-cutoff history."""
    if hist.empty:
        return {
            "users": [],
            "articles": [],
            "u_to_idx": {},
            "i_to_idx": {},
            "user_emb": np.zeros((0, embedding_dim), dtype=np.float32),
            "item_emb": np.zeros((0, embedding_dim), dtype=np.float32),
            "user_emb_norm": np.zeros((0, embedding_dim), dtype=np.float32),
            "item_emb_norm": np.zeros((0, embedding_dim), dtype=np.float32),
        }

    users = sorted(hist["customer_id"].unique().tolist())
    articles = sorted(hist["article_id"].unique().tolist())
    u_to_idx = {u: idx for idx, u in enumerate(users)}
    i_to_idx = {a: idx for idx, a in enumerate(articles)}

    n_users = len(users)
    n_items = len(articles)

    hist_work = hist[["customer_id", "article_id", "days_ago"]].copy()
    hist_work["decay_w"] = np.power(2.0, -hist_work["days_ago"].to_numpy(dtype=np.float32) / max(half_life_days, 1e-3))
    agg = (
        hist_work.groupby(["customer_id", "article_id"], as_index=False)
        .agg(freq=("days_ago", "count"), decay_sum=("decay_w", "sum"))
    )
    u_idx = agg["customer_id"].map(u_to_idx).to_numpy(dtype=np.int32)
    i_idx = agg["article_id"].map(i_to_idx).to_numpy(dtype=np.int32)
    weights = np.log1p(agg["freq"].to_numpy(dtype=np.float32)) + agg["decay_sum"].to_numpy(dtype=np.float32)

    ui_mat = sparse.csr_matrix(
        (weights, (u_idx, i_idx)),
        shape=(n_users, n_items),
        dtype=np.float32,
    )

    max_rank = min(n_users, n_items) - 1
    n_comp = min(embedding_dim, max(1, max_rank))

    if max_rank >= 1 and n_items >= 2 and n_users >= 2:
        svd = TruncatedSVD(n_components=n_comp, random_state=random_state)
        raw_item_emb = svd.fit_transform(ui_mat.T).astype(np.float32)
        col_sums = np.sum(raw_item_emb, axis=0, keepdims=True)
        signs = np.where(col_sums >= 0.0, 1.0, -1.0).astype(np.float32)
        raw_item_emb = raw_item_emb * signs
    else:
        raw_item_emb = np.ones((n_items, 1), dtype=np.float32)
        n_comp = 1

    item_emb = np.zeros((n_items, embedding_dim), dtype=np.float32)
    item_emb[:, :n_comp] = raw_item_emb

    # User embedding is the interaction-weighted average of historical item embeddings
    row_sums = np.asarray(ui_mat.sum(axis=1)).reshape(-1, 1).astype(np.float32)
    user_emb = (ui_mat @ item_emb) / np.maximum(row_sums, 1e-8)

    item_norms = np.linalg.norm(item_emb, axis=1, keepdims=True)
    user_norms = np.linalg.norm(user_emb, axis=1, keepdims=True)
    item_emb_norm = item_emb / np.maximum(item_norms, 1e-8)
    user_emb_norm = user_emb / np.maximum(user_norms, 1e-8)

    return {
        "users": users,
        "articles": articles,
        "u_to_idx": u_to_idx,
        "i_to_idx": i_to_idx,
        "user_emb": user_emb,
        "item_emb": item_emb,
        "user_emb_norm": user_emb_norm,
        "item_emb_norm": item_emb_norm,
    }


def _build_embedding_source(
    hist: pd.DataFrame,
    target_users: List[str],
    cutoff_date: str,
    per_source_k: int,
    embedding_dim: int = 32,
    half_life_days: float = 14.0,
    random_state: int = 42,
    svd_pack: Optional[Dict[str, object]] = None,
) -> pd.DataFrame:
    empty_cols = [
        "customer_id",
        "cutoff_date",
        "article_id",
        "source",
        "source_score",
        "source_rank",
        "generator_version",
    ]
    if hist.empty or not target_users:
        return pd.DataFrame(columns=empty_cols)

    if svd_pack is None:
        svd_pack = compute_svd_embeddings(
            hist,
            embedding_dim=embedding_dim,
            half_life_days=half_life_days,
            random_state=random_state,
        )

    u_to_idx: Dict[str, int] = svd_pack["u_to_idx"]  # type: ignore[assignment]
    articles: List[str] = svd_pack["articles"]  # type: ignore[assignment]
    user_emb_norm: np.ndarray = svd_pack["user_emb_norm"]  # type: ignore[assignment]
    item_emb_norm: np.ndarray = svd_pack["item_emb_norm"]  # type: ignore[assignment]

    n_items = len(articles)
    if n_items == 0:
        return pd.DataFrame(columns=empty_cols)

    active_users = [u for u in target_users if u in u_to_idx]
    if not active_users:
        return pd.DataFrame(columns=empty_cols)

    k_take = min(per_source_k, n_items)
    articles_arr = np.array(articles, dtype=object)

    out_u, out_a, out_s, out_r = [], [], [], []
    batch_size = 1024
    for start in range(0, len(active_users), batch_size):
        batch_u = active_users[start : start + batch_size]
        batch_u_idx = [u_to_idx[u] for u in batch_u]
        sim_mat = user_emb_norm[batch_u_idx] @ item_emb_norm.T  # (B, I)

        for row_i, u_id in enumerate(batch_u):
            sims = sim_mat[row_i]
            if n_items > k_take * 2:
                top_loc = np.argpartition(-sims, k_take * 2)[: k_take * 2]
                sub_art = articles_arr[top_loc]
                sub_sim = np.round(sims[top_loc], 7)
                order = np.lexsort((sub_art, -sub_sim))[:k_take]
                chosen_art = sub_art[order]
                chosen_sim = sims[top_loc][order]
            else:
                sub_sim = np.round(sims, 7)
                order = np.lexsort((articles_arr, -sub_sim))[:k_take]
                chosen_art = articles_arr[order]
                chosen_sim = sims[order]

            out_u.extend([u_id] * len(chosen_art))
            out_a.extend(chosen_art.tolist())
            out_s.extend(chosen_sim.astype(float).tolist())
            out_r.extend(range(1, len(chosen_art) + 1))

    return pd.DataFrame(
        {
            "customer_id": out_u,
            "cutoff_date": cutoff_date,
            "article_id": out_a,
            "source": "EMBEDDING",
            "source_score": out_s,
            "source_rank": out_r,
            "generator_version": GENERATOR_VERSION,
        }
    )[empty_cols]


def generate_candidates_by_source(
    events: pd.DataFrame,
    cutoff_date: str,
    customer_ids: Optional[Iterable[str]] = None,
    per_source_k: int = 60,
    half_life_days: float = 14.0,
    embedding_dim: int = 32,
    random_state: int = 42,
    svd_pack: Optional[Dict[str, object]] = None,
) -> pd.DataFrame:
    """Generate raw multi-source candidate tables (REPEAT, TIME_DECAY, POPULAR, ITEM_COOC, EMBEDDING)."""
    hist = _prepare_history(events, cutoff_date=cutoff_date)
    if customer_ids is None:
        target_users = sorted(hist["customer_id"].unique().tolist())
    else:
        target_users = sorted({str(u) for u in customer_ids})

    target_set = set(target_users)
    rep_df, td_df = _build_repeat_and_decay_sources(
        hist,
        target_set=target_set,
        cutoff_date=cutoff_date,
        per_source_k=per_source_k,
        half_life_days=half_life_days,
    )
    pop_df = _build_popular_source(
        hist,
        target_users=target_users,
        cutoff_date=cutoff_date,
        per_source_k=per_source_k,
    )
    cooc_df = _build_item_cooc_source(
        hist,
        target_users=target_users,
        cutoff_date=cutoff_date,
        per_source_k=per_source_k,
    )
    emb_df = _build_embedding_source(
        hist,
        target_users=target_users,
        cutoff_date=cutoff_date,
        per_source_k=per_source_k,
        embedding_dim=embedding_dim,
        half_life_days=half_life_days,
        random_state=random_state,
        svd_pack=svd_pack,
    )

    combined = pd.concat([rep_df, td_df, pop_df, cooc_df, emb_df], ignore_index=True)
    return combined


def generate_r1_candidates(
    events: pd.DataFrame,
    cutoff_date: str,
    customer_ids: Optional[Iterable[str]] = None,
    k_pool: int = 100,
    per_source_k: Optional[int] = None,
    half_life_days: float = 14.0,
    embedding_dim: int = 32,
    random_state: int = 42,
    svd_pack: Optional[Dict[str, object]] = None,
) -> pd.DataFrame:
    """Generate deduplicated multi-source candidate pool with provenance and per-source rank/score features."""
    if per_source_k is None:
        per_source_k = max(k_pool, 60)

    raw = generate_candidates_by_source(
        events=events,
        cutoff_date=cutoff_date,
        customer_ids=customer_ids,
        per_source_k=per_source_k,
        half_life_days=half_life_days,
        embedding_dim=embedding_dim,
        random_state=random_state,
        svd_pack=svd_pack,
    )
    base_cols = [
        "customer_id",
        "cutoff_date",
        "article_id",
        "source",
        "source_score",
        "source_rank",
        "generator_version",
    ]
    if raw.empty:
        return pd.DataFrame(columns=base_cols)

    # Priority order when source_rank ties across sources
    source_priority = {
        "TIME_DECAY": 1,
        "REPEAT": 2,
        "ITEM_COOC": 3,
        "EMBEDDING": 4,
        "POPULAR": 5,
    }
    raw["_prio"] = raw["source"].map(source_priority).fillna(99).astype(int)

    # Primary provenance row is the source where (customer_id, article_id) achieved its best source_rank
    sorted_raw = raw.sort_values(
        by=["customer_id", "article_id", "source_rank", "_prio"],
        ascending=[True, True, True, True],
        kind="mergesort",
    )
    dedup = sorted_raw.drop_duplicates(
        subset=["customer_id", "cutoff_date", "article_id"], keep="first"
    )[base_cols].copy()

    # Attach per-source scores and ranks for B0 / B1 / B2 ranking
    source_key_map = {
        "REPEAT": "repeat",
        "TIME_DECAY": "time_decay",
        "POPULAR": "popular",
        "ITEM_COOC": "item_cooc",
        "EMBEDDING": "embedding",
    }
    default_rank = per_source_k + 50
    for src_name, suffix in source_key_map.items():
        sub_s = raw.loc[
            raw["source"] == src_name,
            ["customer_id", "article_id", "source_score", "source_rank"],
        ].rename(
            columns={
                "source_score": f"score_{suffix}",
                "source_rank": f"rank_{suffix}",
            }
        )
        dedup = dedup.merge(sub_s, on=["customer_id", "article_id"], how="left")
        dedup[f"score_{suffix}"] = dedup[f"score_{suffix}"].fillna(0.0).astype(float)
        dedup[f"rank_{suffix}"] = dedup[f"rank_{suffix}"].fillna(default_rank).astype(int)

    # Compute source_count and composite pool_score for deterministic top-K pool truncation
    dedup["source_count"] = (
        (dedup["rank_repeat"] < default_rank).astype(int)
        + (dedup["rank_time_decay"] < default_rank).astype(int)
        + (dedup["rank_popular"] < default_rank).astype(int)
        + (dedup["rank_item_cooc"] < default_rank).astype(int)
        + (dedup["rank_embedding"] < default_rank).astype(int)
    )

    def _rr(rank_series: pd.Series) -> np.ndarray:
        r = rank_series.to_numpy(dtype=float)
        return np.where(r < default_rank, 1.0 / (2.0 + r), 0.0)

    dedup["pool_score"] = (
        3.0 * _rr(dedup["rank_time_decay"])
        + 2.5 * _rr(dedup["rank_repeat"])
        + 1.5 * _rr(dedup["rank_item_cooc"])
        + 1.2 * _rr(dedup["rank_embedding"])
        + 1.0 * _rr(dedup["rank_popular"])
    )

    dedup = dedup.sort_values(
        by=["customer_id", "pool_score", "article_id"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    dedup = dedup.groupby("customer_id", sort=False).head(k_pool).reset_index(drop=True)
    dedup["pool_rank"] = dedup.groupby("customer_id", sort=False).cumcount() + 1
    return dedup
