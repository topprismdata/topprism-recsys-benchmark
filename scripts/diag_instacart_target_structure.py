"""GPT-mandated verification #1 (no retraining): target-structure audit for §7.

Q1: does the LOO test target (and val target) belong to prior or train order?
Q2: target order_number / add_to_cart_order distribution; is the target in the SAME
    basket as the last input event (within-basket next-item) or a NEW basket?
Q3: exact repeat rate of test targets (target seen in input history?) — not the
    flattened-event average.
Q4: do val and test targets agree on these dimensions?
Plus: freq/last-item trivial baselines split by Repeat/New target group (GPT check #2,
trivial-baseline part; neural per-user records were not persisted — separate rerun).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data/instacart/raw"
SEED = 42
N_USERS = 20000


def find_file(name: str) -> Path:
    return sorted(RAW.rglob(name))[0]


def main() -> None:
    orders = pd.read_csv(find_file("orders.csv"))
    prior = pd.read_csv(find_file("order_products__prior.csv"))
    train_op = pd.read_csv(find_file("order_products__train.csv"))
    op = pd.concat([prior, train_op], ignore_index=True)
    op = op.merge(orders[["order_id", "user_id", "order_number", "eval_set"]], on="order_id", how="left")

    while True:
        ku = set(op["user_id"].value_counts()[lambda s: s >= 5].index)
        ki = set(op["product_id"].value_counts()[lambda s: s >= 5].index)
        f = op[op["user_id"].isin(ku) & op["product_id"].isin(ki)]
        if len(f) == len(op):
            break
        op = f
    op = op.sort_values(["user_id", "order_number", "add_to_cart_order"], kind="mergesort")

    rng = np.random.default_rng(SEED)
    pick = sorted(rng.choice(sorted(op["user_id"].unique()), size=N_USERS, replace=False).tolist())
    op = op[op["user_id"].isin(set(pick))]

    out = {}
    for tag, cut in [("val", -2), ("test", -1)]:
        rows = []
        for u, g in op.groupby("user_id", sort=False):
            if len(g) < 3:
                continue
            tgt = g.iloc[cut]
            hist = g.iloc[:cut]
            rows.append({
                "u": u,
                "eval_set": tgt["eval_set"],
                "order_number": int(tgt["order_number"]),
                "max_order": int(g["order_number"].max()),
                "atc": int(tgt["add_to_cart_order"]),
                "basket_size": int((g["order_number"] == tgt["order_number"]).sum()),
                "same_basket_as_prev": bool(tgt["order_number"] == hist.iloc[-1]["order_number"]),
                "repeat": bool((hist["product_id"] == tgt["product_id"]).any()),
            })
        df = pd.DataFrame(rows)
        out[tag] = {
            "n": len(df),
            "target_in_train_set_frac": round(float((df["eval_set"] == "train").mean()), 4),
            "target_order_is_last_order_frac": round(float((df["order_number"] == df["max_order"]).mean()), 4),
            "same_basket_as_prev_event_frac": round(float(df["same_basket_as_prev"].mean()), 4),
            "repeat_target_frac": round(float(df["repeat"].mean()), 4),
            "atc_pos_median": float(df["atc"].median()),
            "atc_pos_mean": round(float(df["atc"].mean()), 2),
            "atc_equals_1_frac": round(float((df["atc"] == 1).mean()), 4),
        }

    # freq/last baselines split by Repeat/New on TEST
    seqs = {int(u): [int(x) for x in g["product_id"]] for u, g in op.groupby("user_id", sort=False)}
    seqs = {u: s for u, s in seqs.items() if len(s) >= 3}
    res = {}
    for name in ["freq_item", "last_item"]:
        hit_r = hit_n = n_r = n_n = 0
        for u, s in seqs.items():
            hist, tgt = s[:-1], s[-1]
            if name == "freq_item":
                cnt = {}
                for it in hist:
                    cnt[it] = cnt.get(it, 0) + 1
                cand = sorted(cnt.keys(), key=lambda k: (-cnt[k], k))
            else:
                seen = {}
                for pos, it in enumerate(hist):
                    seen[it] = pos
                cand = sorted(seen.keys(), key=lambda k: (-seen[k], k))
            rep = tgt in hist
            if rep:
                n_r += 1
                if tgt in cand[:10]:
                    hit_r += 1
            else:
                n_n += 1
                if tgt in cand[:10]:
                    hit_n += 1
        res[name] = {"repeat_group": {"n": n_r, "hr10": round(hit_r / max(n_r, 1), 4)},
                     "new_group": {"n": n_n, "hr10": round(hit_n / max(n_n, 1), 4)}}
    out["trivial_by_target_group_test"] = res
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
