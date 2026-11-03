from pathlib import Path
from typing import Optional, Union
import numpy as np
import pandas as pd

from src.data.window_builder import assign_customer_strata


DEFAULT_SWIRE_CSV = Path("C:/Users/ghb/ZCodeProject/swire_ro/26年订单.csv")
DEFAULT_SWIRE_CACHE = Path("C:/Users/ghb/ZCodeProject/recsys-benchmark/data/swire_events.parquet")


def load_swire_events(
    csv_path: Union[str, Path] = DEFAULT_SWIRE_CSV,
    chunksize: int = 500_000,
    cache_path: Optional[Union[str, Path]] = DEFAULT_SWIRE_CACHE,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Stream-read Swire orders CSV into canonical 4-column event table and cache to parquet."""
    csv_file = Path(csv_path)
    cache_file = Path(cache_path) if cache_path is not None else None

    if (
        use_cache
        and cache_file is not None
        and cache_file.exists()
        and csv_file.exists()
        and cache_file.stat().st_mtime >= csv_file.stat().st_mtime
    ):
        df = pd.read_parquet(cache_file)
        df["customer_id"] = df["customer_id"].astype(str)
        df["article_id"] = df["article_id"].astype(str)
        df["event_date"] = df["event_date"].astype(str)
        df["value"] = df["value"].astype(float)
        return df[["customer_id", "article_id", "event_date", "value"]]

    usecols = ["OUTLET", "ARTICLE", "ORDER_DATE", "SETTLE_CASE"]
    dtype_spec = {"OUTLET": str, "ARTICLE": str, "ORDER_DATE": str}
    chunks = []

    for chunk in pd.read_csv(
        csv_file,
        usecols=usecols,
        dtype=dtype_spec,
        chunksize=chunksize,
        low_memory=False,
    ):
        val = pd.to_numeric(chunk["SETTLE_CASE"], errors="coerce")
        valid_mask = (
            val.notna()
            & (val > 0)
            & chunk["OUTLET"].notna()
            & chunk["ARTICLE"].notna()
            & chunk["ORDER_DATE"].notna()
        )
        if not valid_mask.any():
            continue

        sub = pd.DataFrame(
            {
                "customer_id": chunk.loc[valid_mask, "OUTLET"].astype(str).str.strip(),
                "article_id": chunk.loc[valid_mask, "ARTICLE"].astype(str).str.strip(),
                "event_date": chunk.loc[valid_mask, "ORDER_DATE"].astype(str).str.strip().str.slice(0, 10),
                "value": val.loc[valid_mask].astype(float),
            }
        )
        sub_agg = sub.groupby(["customer_id", "article_id", "event_date"], as_index=False)["value"].sum()
        chunks.append(sub_agg)

    if not chunks:
        canonical = pd.DataFrame(columns=["customer_id", "article_id", "event_date", "value"])
    else:
        combined = pd.concat(chunks, ignore_index=True)
        canonical = (
            combined.groupby(["customer_id", "article_id", "event_date"], as_index=False)["value"]
            .sum()
            .sort_values(["customer_id", "event_date", "article_id"])
            .reset_index(drop=True)
        )
        canonical["customer_id"] = canonical["customer_id"].astype(str)
        canonical["article_id"] = canonical["article_id"].astype(str)
        canonical["event_date"] = canonical["event_date"].astype(str)
        canonical["value"] = canonical["value"].astype(float)

    if cache_file is not None:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        canonical.to_parquet(cache_file, index=False)

    return canonical[["customer_id", "article_id", "event_date", "value"]]


def load_hm_events(parquet_path: Union[str, Path]) -> pd.DataFrame:
    """Load H&M transactions_train.parquet into canonical 4-column event table."""
    pq_file = Path(parquet_path)
    df = pd.read_parquet(pq_file, columns=["customer_id", "article_id", "t_dat", "price"])
    canonical = pd.DataFrame(
        {
            "customer_id": df["customer_id"].astype(str).str.strip(),
            "article_id": df["article_id"].astype(str).str.strip().str.zfill(10),
            "event_date": df["t_dat"].astype(str).str.strip().str.slice(0, 10),
            "value": df["price"].astype(float),
        }
    )
    return canonical[["customer_id", "article_id", "event_date", "value"]]


def stratified_customer_sample(
    events: pd.DataFrame,
    cutoff_date: str,
    n_customers: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Deterministic stratified customer sampling across S0 / S1 / S2 strata at cutoff_date."""
    strata_df = assign_customer_strata(events, cutoff_date=cutoff_date)
    total_customers = len(strata_df)
    if n_customers <= 0 or total_customers == 0:
        return events.iloc[0:0].copy()
    if n_customers >= total_customers:
        return (
            events.copy()
            .sort_values(["customer_id", "event_date", "article_id"])
            .reset_index(drop=True)
        )

    strata_order = ["S0", "S1", "S2"]
    groups = {
        s: sorted(strata_df.loc[strata_df["stratum"] == s, "customer_id"].astype(str).tolist())
        for s in strata_order
    }
    counts = np.array([len(groups[s]) for s in strata_order], dtype=float)
    exact_quotas = counts / float(total_customers) * float(n_customers)
    alloc = np.floor(exact_quotas).astype(int)
    alloc = np.minimum(alloc, counts.astype(int))

    remainder = int(n_customers - alloc.sum())
    if remainder > 0:
        frac = exact_quotas - alloc
        # Sort by largest fractional remainder, tie-break by stratum index
        order = sorted(range(len(strata_order)), key=lambda i: (-frac[i], i))
        for idx in order:
            if remainder <= 0:
                break
            capacity = int(counts[idx]) - alloc[idx]
            if capacity > 0:
                take = min(remainder, capacity)
                alloc[idx] += take
                remainder -= take

    rng = np.random.default_rng(seed)
    sampled_customers = set()
    for idx, s in enumerate(strata_order):
        k_s = int(alloc[idx])
        if k_s > 0:
            chosen = rng.choice(groups[s], size=k_s, replace=False).tolist()
            sampled_customers.update(chosen)

    sub = events.loc[events["customer_id"].astype(str).isin(sampled_customers)].copy()
    return sub.sort_values(["customer_id", "event_date", "article_id"]).reset_index(drop=True)
