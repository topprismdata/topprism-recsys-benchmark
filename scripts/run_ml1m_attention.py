"""ML-1M public-benchmark test of the attention hypothesis (user directive 2026-10-26).

Purpose: test whether the H&M negative result was a dataset artifact. ML-1M has
~165 median events per user (vs 6 on H&M) and published full-ranking baselines
from the HSTU paper (verified numbers on meta-recsys/generative-recommenders README,
as of 2024-04-15): SASRec HR@10=.2853 NDCG@10=.1603, GRU4Rec .2811/.1648,
BERT4Rec .2843/.1537, HSTU .3097/.1720, HSTU-large .3294/.1858.

Protocol (pre-registered before any run):
  - ratings.dat, 5-core filter (iterative: items <5 ratings and users <5 ratings removed), time-sorted.
  - leave-one-out per user: last rating = test target, second-to-last timestamp = cutoff.
  - Training: sampled-softmax-256 (frozen R2 objective), d=128, 2 layers, 4 heads,
    L=200, batch 256, 8 epochs, AdamW lr 2e-3 wd 1e-4, seed 42 — same frozen budget as R2.
  - Evaluation: full ranking over ALL items (no seen-item masking), HR@10 / HR@50 / NDCG@10,
    unified last-step representation x item-embedding dot product for every encoder.
  - LightGBM baseline: full-ranking scoring with feature set {item popularity, user activity,
    co-occurrence(last 10 items), time-decayed co-occurrence(half-life 14d), user mean rating bucket},
    trained on the same leave-one-out task over all items.
  - Implementation check: our SASRec should land in a plausible band around the published .2853;
    if it does not, the pipeline is wrong and results are not interpretable.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch

from src.models.r2_hybrid_runner import train_r2_sequential_model

SEED = 42
L = 200
ENCODERS = ["mean_pool", "gru", "sasrec", "hstu_aligned"]
DEVICE = "cuda"
CACHE = REPO / "artifacts/eval/ml1m_cache"


def log(msg: str) -> None:
    print(msg, flush=True)


def load_ml1m() -> pd.DataFrame:
    df = pd.read_csv(
        REPO / "data/ml1m/ml-1m/ratings.dat",
        sep="::", engine="python",
        names=["user_id", "item_id", "rating", "timestamp"],
    )
    # 5-core iterative filter
    while True:
        ucnt = df["user_id"].value_counts()
        icnt = df["item_id"].value_counts()
        keep_u = set(ucnt[ucnt >= 5].index)
        keep_i = set(icnt[icnt >= 5].index)
        filtered = df[df["user_id"].isin(keep_u) & df["item_id"].isin(keep_i)]
        if len(filtered) == len(df):
            break
        df = filtered
    df = df.sort_values(["user_id", "timestamp", "item_id"], kind="mergesort").reset_index(drop=True)
    df["customer_id"] = df["user_id"].astype(str)
    df["article_id"] = df["item_id"].astype(str)
    # day-granularity dates (frozen R2 pipeline requirement); within-day order by item_id
    df["event_date"] = pd.to_datetime(df["timestamp"], unit="s").dt.strftime("%Y-%m-%d")
    df["value"] = 1.0
    log(f"[data] 5-core: users={df['customer_id'].nunique()} items={df['article_id'].nunique()} events={len(df)} dates={df['event_date'].min()}..{df['event_date'].max()}")
    return df[["customer_id", "article_id", "event_date", "value", "timestamp"]]


def build_loo(df: pd.DataFrame):
    """leave-one-out: per user, last event = test target; cutoff = timestamp of second-to-last."""
    events = df[["customer_id", "article_id", "event_date", "value"]].copy()
    test_rows = df.groupby("customer_id", sort=False).tail(1)
    cutoffs = df.drop_duplicates("customer_id", keep="last").set_index("customer_id")
    cutoffs = df.groupby("customer_id", sort=False).tail(2).groupby("customer_id", sort=False).tail(1).set_index("customer_id")
    # cutoff_date per user = event_date string of second-to-last event (inclusive => keeps seq up to it)
    cutoff_map = cutoffs["event_date"].to_dict()
    test = test_rows.set_index("customer_id")
    return events, cutoff_map, test["article_id"].to_dict()


def evaluate_full_ranking(pack, test_targets: Dict[str, str], device):
    """Rank all items per user by last-step rep . item_emb^T; HR@10/50, NDCG@10.

    Eval inputs exclude each user's held-out last event (LOO target); inputs are rebuilt
    here from the events table to guarantee no target leakage.
    """
    model = pack["model"]
    vocab = pack["vocab"]
    art_to_id = vocab["art_to_id"]
    num_items = len(vocab["articles"])
    model.eval()
    dev = next(model.parameters()).device
    user_ids = pack["eval_user_ids"]
    eval_ids = pack["eval_input_ids"].to(dev)
    eval_ts = pack["eval_input_ts"].to(dev)
    item_emb_t = model.item_embedding.weight.detach()
    hr10 = hr50 = ndcg10 = 0
    n = 0
    with torch.no_grad():
        for i in range(0, len(user_ids), 512):
            reps = model.encode_last_step(eval_ids[i:i + 512], timestamps=eval_ts[i:i + 512])
            scores = reps @ item_emb_t.t()
            scores[:, 0] = -float("inf")
            top50 = torch.topk(scores, k=50, dim=1).indices.cpu().numpy()
            for j, u in enumerate(user_ids[i:i + 512]):
                tgt = art_to_id.get(test_targets.get(str(u)))
                if tgt is None:
                    continue
                n += 1
                r50 = top50[j]
                if tgt in r50[:10]:
                    hr10 += 1
                    ndcg10 += 1.0 / np.log2(int(np.flatnonzero(r50[:10] == tgt)[0]) + 2)
                if tgt in r50:
                    hr50 += 1
    return {"n": n, "hr10": hr10 / n, "hr50": hr50 / n, "ndcg10": ndcg10 / n}


def build_eval_inputs(df: pd.DataFrame, art_to_id: Dict[str, int], max_seq_len: int):
    """Per-user input sequence = all events except the last (LOO target); ts = relative days to global max."""
    rows, tss, users = [], [], []
    cutoff_ts = df["timestamp"].max()
    for u, g in df.groupby("customer_id", sort=False):
        g = g.iloc[:-1]
        if g.empty:
            continue
        items = [art_to_id.get(a, 0) for a in g["article_id"]]
        ts = (cutoff_ts - g["timestamp"].to_numpy()) / 86400.0
        items = items[-max_seq_len:]
        ts = ts[-max_seq_len:]
        r = np.zeros(max_seq_len, dtype=np.int64)
        t = np.zeros(max_seq_len, dtype=np.float32)
        r[: len(items)] = items
        t[: len(items)] = ts
        rows.append(r)
        tss.append(t)
        users.append(u)
    return (
        torch.from_numpy(np.stack(rows)),
        torch.from_numpy(np.stack(tss)),
        users,
    )


def main() -> None:
    t0 = time.time()
    CACHE.mkdir(parents=True, exist_ok=True)
    out_path = REPO / "artifacts/eval/ml1m_attention.json"
    out: Dict[str, Any] = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {
        "experiment": "ml1m_attention_public_benchmark", "protocol_baselines_hstu_paper": {
            "SASRec": {"hr10": 0.2853, "ndcg10": 0.1603}, "GRU4Rec": {"hr10": 0.2811, "ndcg10": 0.1648},
            "BERT4Rec": {"hr10": 0.2843, "ndcg10": 0.1537}, "HSTU": {"hr10": 0.3097, "ndcg10": 0.1720}}}
    ap_path = CACHE / "results.json"
    results: Dict[str, Any] = json.loads(ap_path.read_text(encoding="utf-8")) if ap_path.exists() else {}

    df = load_ml1m()
    events, cutoff_map, test_targets = build_loo(df)

    # per-user single-cutoff training via the frozen R2 trainer:
    # events restricted per global cutoff is not possible (per-user cutoffs), so emulate:
    # train_r2_sequential_model uses cutoff_date on event_date strings; our event_date embeds the
    # within-user rank, so a global cutoff cannot express per-user LOO. Instead: drop each user's
    # last event from the training events (targets live outside), cutoff = max event_date (all rows).
    train_events = df.drop(df.groupby("customer_id", sort=False).tail(1).index)[
        ["customer_id", "article_id", "event_date", "value"]
    ].reset_index(drop=True)
    max_date = train_events["event_date"].max()

    out["data"] = {"users": int(df['customer_id'].nunique()), "items": int(df['article_id'].nunique()), "events": int(len(df))}
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    for enc in ENCODERS:
        if enc in results:
            log(f"[cached] {enc}: {results[enc]}")
            continue
        t1 = time.time()
        pack = train_r2_sequential_model(
            events=train_events, cutoff_date=max_date, encoder_name=enc, loss_name="sampled_softmax_256",
            embedding_dim=128, max_seq_len=L, num_layers=2, num_heads=4, num_negatives=256,
            batch_size=256, epochs=8, learning_rate=2e-3, seed=SEED, device=DEVICE,
            sequence_order="chronological", shuffle_seed=SEED,
        )
        eval_ids, eval_ts, eval_users = build_eval_inputs(df, pack["vocab"]["art_to_id"], L)
        pack["eval_user_ids"] = eval_users
        pack["eval_input_ids"] = eval_ids
        pack["eval_input_ts"] = eval_ts
        ev = evaluate_full_ranking(pack, test_targets, torch.device(DEVICE))
        results[enc] = ev
        log(f"[{enc}] HR@10={ev['hr10']:.4f} HR@50={ev['hr50']:.4f} NDCG@10={ev['ndcg10']:.4f} ({time.time()-t1:.0f}s)")
        ap_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    out["results"] = results
    out["total_seconds"] = round(time.time() - t0, 1)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log(f"DONE in {out['total_seconds']}s")


if __name__ == "__main__":
    main()
