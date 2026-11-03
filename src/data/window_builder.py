from datetime import datetime, timedelta
import hashlib
from typing import Dict, Iterable, Optional
import numpy as np
import pandas as pd


def compute_dataframe_hash(df: pd.DataFrame) -> str:
    if df.empty:
        return hashlib.sha256(b"EMPTY").hexdigest()
    cols = sorted(df.columns.tolist())
    normalized = df[cols].copy()
    for c in cols:
        normalized[c] = normalized[c].astype(str)
    normalized = normalized.sort_values(by=cols).reset_index(drop=True)
    payload = normalized.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_purchase_labels(events: pd.DataFrame, cutoff_date: str, horizon_days: int = 7) -> pd.DataFrame:
    dt_cut = datetime.strptime(cutoff_date, "%Y-%m-%d")
    end_date = (dt_cut + timedelta(days=horizon_days)).strftime("%Y-%m-%d")
    mask = (events["event_date"] > cutoff_date) & (events["event_date"] <= end_date)
    sub = events.loc[mask, ["customer_id", "article_id", "event_date"]].copy()
    sub["customer_id"] = sub["customer_id"].astype(str)
    sub["article_id"] = sub["article_id"].astype(str)
    dedup = (
        sub.groupby(["customer_id", "article_id"], as_index=False)["event_date"]
        .min()
        .sort_values(["customer_id", "article_id"])
        .reset_index(drop=True)
    )
    dedup["cutoff_date"] = cutoff_date
    dedup["purchased_next_7d"] = 1
    return dedup


def build_window(events: pd.DataFrame, cutoff_date: str, horizon_days: int = 7) -> Dict[str, pd.DataFrame]:
    hist_mask = events["event_date"] <= cutoff_date
    history = events.loc[hist_mask, ["customer_id", "article_id", "event_date", "value"]].copy()
    history["customer_id"] = history["customer_id"].astype(str)
    history["article_id"] = history["article_id"].astype(str)
    history = history.sort_values(["customer_id", "event_date", "article_id"]).reset_index(drop=True)
    labels = build_purchase_labels(events, cutoff_date=cutoff_date, horizon_days=horizon_days)
    return {"history": history, "labels": labels}


def build_daily_baskets(events: pd.DataFrame, cutoff_date: str) -> pd.DataFrame:
    hist = events.loc[events["event_date"] <= cutoff_date, ["customer_id", "event_date", "article_id"]].copy()
    hist["customer_id"] = hist["customer_id"].astype(str)
    hist["article_id"] = hist["article_id"].astype(str)
    dedup = hist.drop_duplicates(subset=["customer_id", "event_date", "article_id"]).sort_values(
        ["customer_id", "event_date", "article_id"]
    )
    baskets = (
        dedup.groupby(["customer_id", "event_date"], as_index=False)["article_id"]
        .agg(lambda s: "|".join(s.tolist()))
        .rename(columns={"event_date": "basket_date", "article_id": "articles_sorted"})
    )
    baskets["cutoff_date"] = cutoff_date
    return baskets.sort_values(["customer_id", "basket_date"]).reset_index(drop=True)


def assign_customer_strata(
    events: pd.DataFrame,
    cutoff_date: str,
    customer_universe: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    hist = events.loc[events["event_date"] <= cutoff_date, ["customer_id", "event_date"]].copy()
    hist["customer_id"] = hist["customer_id"].astype(str)
    if customer_universe is None:
        universe = sorted(events["customer_id"].astype(str).unique().tolist())
    else:
        universe = sorted({str(u) for u in customer_universe})

    base = pd.DataFrame({"customer_id": universe})
    if hist.empty:
        base["cutoff_date"] = cutoff_date
        base["historical_basket_count"] = 0
        base["days_since_last_purchase"] = -1
        base["stratum"] = "S0"
        return base

    agg = (
        hist.groupby("customer_id", as_index=False)
        .agg(
            historical_basket_count=("event_date", "nunique"),
            last_date=("event_date", "max"),
        )
    )
    merged = base.merge(agg, on="customer_id", how="left")
    merged["historical_basket_count"] = merged["historical_basket_count"].fillna(0).astype(int)
    dt_cut = pd.to_datetime(cutoff_date)
    last_dt = pd.to_datetime(merged["last_date"])
    days = (dt_cut - last_dt).dt.days.fillna(-1).astype(int)
    merged["days_since_last_purchase"] = days
    merged["cutoff_date"] = cutoff_date

    b_cnt = merged["historical_basket_count"]
    merged["stratum"] = np.where(b_cnt <= 1, "S0", np.where(b_cnt <= 4, "S1", "S2"))
    return merged[["customer_id", "cutoff_date", "historical_basket_count", "days_since_last_purchase", "stratum"]]
