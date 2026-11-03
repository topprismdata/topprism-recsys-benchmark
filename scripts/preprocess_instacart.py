"""Instacart preprocessing + density audit (Stage 1 of the retail cross-check).

Input : data/instacart/raw/ (Kaggle mirror of the official competition files)
Output: data/instacart/seqs.json      {user_id: [product_id,...]} flattened SKU events in true order
        artifacts/eval/instacart_audit.json

Ordering policy (documented before running, single-variable discipline):
- Cross-order  : order_number ascending (true temporal order).
- Within-order : add_to_cart_order ascending (Instacart exposes REAL within-basket order,
                 unlike H&M/ML-1M where within-basket ties are arbitrary).
- 5-core iterative filter on users AND products (same as ML-1M v2 protocol).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data/instacart/raw"
OUT_SEQS = REPO / "data/instacart/seqs.json"
OUT_AUDIT = REPO / "artifacts/eval/instacart_audit.json"


def log(m: str) -> None:
    print(m, flush=True)


def find_file(name: str) -> Path:
    hits = sorted(RAW.rglob(name))
    if not hits:
        log(f"FATAL missing {name} under {RAW}")
        sys.exit(1)
    return hits[0]


def main() -> None:
    orders = pd.read_csv(find_file("orders.csv"))
    prior = pd.read_csv(find_file("order_products__prior.csv"))
    train_op = pd.read_csv(find_file("order_products__train.csv"))
    products = pd.read_csv(find_file("products.csv"))

    log(f"[raw] orders={len(orders)} prior_lines={len(prior)} train_lines={len(train_op)} products={len(products)}")

    op = pd.concat([prior, train_op], ignore_index=True)
    op = op.merge(orders[["order_id", "user_id", "order_number"]], on="order_id", how="left")

    n_users0 = op["user_id"].nunique()
    n_items0 = op["product_id"].nunique()
    n_events0 = len(op)

    # 5-core iterative filter (users and products)
    while True:
        ku = set(op["user_id"].value_counts()[lambda s: s >= 5].index)
        ki = set(op["product_id"].value_counts()[lambda s: s >= 5].index)
        f = op[op["user_id"].isin(ku) & op["product_id"].isin(ki)]
        if len(f) == len(op):
            break
        op = f

    n_users5 = op["user_id"].nunique()
    n_items5 = op["product_id"].nunique()
    n_events5 = len(op)

    # true order: order_number, then add_to_cart_order
    op = op.sort_values(["user_id", "order_number", "add_to_cart_order"], kind="mergesort")

    seqs: dict[int, list[int]] = {}
    for u, g in op.groupby("user_id", sort=False):
        seqs[int(u)] = [int(x) for x in g["product_id"]]

    ev = np.array([len(s) for s in seqs.values()])
    per_order = op.groupby(["user_id", "order_number"]).size().values
    audit = {
        "raw": {"users": int(n_users0), "products": int(n_items0), "events": int(n_events0)},
        "after_5core": {"users": int(n_users5), "products": int(n_items5), "events": int(n_events5)},
        "events_per_user": {
            "mean": round(float(ev.mean()), 1),
            "median": float(np.median(ev)),
            "p10": float(np.percentile(ev, 10)),
            "p90": float(np.percentile(ev, 90)),
            "min": int(ev.min()),
            "max": int(ev.max()),
        },
        "orders_per_user_median": float(np.median([len([1]) for s in [1]])) if False else None,
        "basket_sizes": {
            "mean": round(float(per_order.mean()), 2),
            "median": float(np.median(per_order)),
        },
        "users_len_ge2": int(sum(1 for v in ev if v >= 2)),
        "protocol": {
            "core": "5-core iterative (users+products)",
            "ordering": "order_number asc, add_to_cart_order asc (real within-basket order)",
            "dense_note": "flattened SKU events; comparable to ML-1M ratings events",
        },
    }

    # orders per user
    ord_per_user = op.groupby("user_id")["order_number"].nunique().values
    audit["orders_per_user"] = {
        "mean": round(float(ord_per_user.mean()), 1),
        "median": float(np.median(ord_per_user)),
    }

    OUT_SEQS.parent.mkdir(parents=True, exist_ok=True)
    OUT_SEQS.write_text(json.dumps(seqs), encoding="utf-8")
    OUT_AUDIT.parent.mkdir(parents=True, exist_ok=True)
    OUT_AUDIT.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    log(json.dumps(audit, indent=2))
    log("DONE")


if __name__ == "__main__":
    main()
