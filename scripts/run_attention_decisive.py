"""Decisive attention experiment (user-approved 2026-10-26).

Pre-registered protocol (frozen before any run):
  Question: does attention provide gains under (a) an official-aligned HSTU implementation and
  (b) long sequences (L=200), the two loopholes of the R2 negative result?

  Arms:
    Batch A (L=50,  correction check): encoders {mean_pool, gru, sasrec, hstu_aligned} x SS256,
      8 epochs, seed 42 — H&M 40k frozen sample + swire_ro full. hstu_aligned vs the archived
      V1 HSTU numbers isolates the implementation-identity loophole.
    Batch B (L=200, long-sequence): same four encoders, only max_seq_len changes 50 -> 200.
  Everything else (candidate generation, windows, losses, optimizer, negatives, budget) is
  byte-identical to the R2/confirmation protocol. LightGBM B1/B2 are L-invariant and trained once.

  Output: artifacts/eval/attention_decisive.json (+ per-model AP caches under
  artifacts/eval/decisive_cache/ for resume).
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import torch

from src.data.adapters import load_hm_events, load_swire_events, stratified_customer_sample
from src.eval.metrics import _canonical_top_k, _label_sets, compute_user_ap12_components
from src.models.r1_baselines import R1LightGBMRanker, evaluate_r1_predictions
from src.models.r1_candidates import _prepare_history, compute_svd_embeddings, generate_r1_candidates
from src.models.r1_features import build_r1_feature_matrix
from src.models.r2_hybrid_runner import predict_track_a_candidates, train_r2_sequential_model
from src.data.window_builder import assign_customer_strata, build_window

HM_CUTOFFS = ["2020-08-18", "2020-08-25", "2020-09-01", "2020-09-08"]
HM_SEALED = "2020-09-15"
SEED = 42
DEVICE = "cuda"
N_SAMPLE = 40000
FALLBACK_N = 30000
ENCODERS = ["mean_pool", "gru", "sasrec", "hstu_aligned"]
CACHE = REPO / "artifacts/eval/decisive_cache"


def log(msg: str) -> None:
    print(msg, flush=True)


def build_windows(events, cutoffs, sealed, k_pool=100, tag="hm"):
    safe = events.loc[
        events["event_date"].astype(str) <= str(sealed),
        ["customer_id", "article_id", "event_date", "value"],
    ].copy()
    for c in ("customer_id", "article_id", "event_date"):
        safe[c] = safe[c].astype(str)
    safe["value"] = safe["value"].astype(float)
    arts: Dict[str, Dict[str, Any]] = {}
    for idx, cutoff in enumerate(cutoffs, start=1):
        t0 = time.time()
        w = build_window(safe, cutoff_date=cutoff, horizon_days=7)
        strata_df = assign_customer_strata(safe, cutoff_date=cutoff)
        target_users = strata_df["customer_id"].astype(str).tolist()
        hist_prep = _prepare_history(safe, cutoff_date=cutoff)
        svd_pack = compute_svd_embeddings(hist_prep, embedding_dim=32, half_life_days=14.0, random_state=SEED)
        cands = generate_r1_candidates(
            safe, cutoff_date=cutoff, customer_ids=target_users, k_pool=k_pool,
            per_source_k=max(k_pool, 200), half_life_days=14.0, embedding_dim=32,
            random_state=SEED, svd_pack=svd_pack,
        )
        feat = build_r1_feature_matrix(
            safe, cands, cutoff_date=cutoff, half_life_days=14.0, embedding_dim=32,
            random_state=SEED, svd_pack=svd_pack,
        )
        arts[f"W{idx}"] = {
            "cutoff_date": cutoff, "window": w, "strata": strata_df,
            "target_users": target_users, "cands": cands, "feat": feat,
        }
        log(f"  [{tag}] W{idx} cutoff={cutoff} cands={len(cands)} feat={len(feat)} ({time.time()-t0:.0f}s)")
    return safe, arts


def build_hm_enlarged_sample(events: pd.DataFrame, n_customers: int):
    ev6 = events.loc[
        (events["event_date"].astype(str) >= "2020-04-01") & (events["event_date"].astype(str) <= HM_SEALED)
    ].copy()
    sample = stratified_customer_sample(ev6, cutoff_date="2020-09-08", n_customers=n_customers, seed=SEED)
    custs = set(sample["customer_id"].astype(str))
    sub = ev6.loc[ev6["customer_id"].astype(str).isin(custs)]
    hist_part = sub.loc[sub["event_date"].astype(str) <= "2020-09-08"]
    top = hist_part["article_id"].value_counts().head(5000).index
    sample_events = sub.loc[sub["article_id"].isin(set(top))].copy()
    meta = {
        "sampled_customers": int(len(custs)),
        "active_catalog_articles": int(sample_events["article_id"].nunique()),
        "total_events": int(len(sample_events)),
        "sampling_cutoff": "2020-09-08",
        "article_universe_cutoff": "2020-09-08",
        "seed": SEED,
        "frozen": True,
    }
    return sample_events, meta


def train_seq(safe, cutoff, encoder, max_seq_len, loss="sampled_softmax_256", epochs=8):
    return train_r2_sequential_model(
        events=safe, cutoff_date=cutoff, encoder_name=encoder, loss_name=loss,
        embedding_dim=128, max_seq_len=max_seq_len, num_layers=2, num_heads=4, num_negatives=256,
        batch_size=256, epochs=epochs, learning_rate=2e-3, seed=SEED, device=DEVICE,
        sequence_order="chronological", shuffle_seed=SEED,
    )


def track_a_eval(trained, eval_art):
    preds = predict_track_a_candidates(trained, eval_art["cands"])
    ev = evaluate_r1_predictions(
        preds, candidates=eval_art["cands"], labels=eval_art["window"]["labels"],
        history=eval_art["window"]["history"], strata_df=eval_art["strata"], k=12,
    )
    return preds, ev


def per_customer_ap(preds, labels, history) -> Dict[str, float]:
    top12 = _canonical_top_k(preds, k=12)
    labs = _label_sets(labels)
    hist_df = history[["customer_id", "article_id"]].drop_duplicates()
    hist_df = hist_df.assign(customer_id=hist_df["customer_id"].astype(str), article_id=hist_df["article_id"].astype(str))
    hist_sets = hist_df.groupby("customer_id")["article_id"].apply(set).to_dict()
    out = {}
    for cust, true_items in labs.items():
        ap_total, _r, _e = compute_user_ap12_components(top12.get(cust, []), true_items, hist_sets.get(cust, set()))
        out[str(cust)] = float(ap_total)
    return out


def paired_ci(ap_a: Dict[str, float], ap_b: Dict[str, float], strata_df, n_boot=1000):
    rng = np.random.default_rng(SEED)
    strat = dict(zip(strata_df["customer_id"].astype(str), strata_df["stratum"].astype(str)))
    common = sorted(set(ap_a) & set(ap_b))
    diffs = {c: ap_a[c] - ap_b[c] for c in common}
    out: Dict[str, Any] = {}
    for scope, keys in [("ALL", common)] + [(s, [c for c in common if strat.get(c) == s]) for s in ("S0", "S1", "S2")]:
        if not keys:
            out[scope] = None
            continue
        vals = np.array([diffs[c] for c in keys], dtype=np.float64)
        idx = rng.integers(0, vals.size, size=(n_boot, vals.size))
        means = vals[idx].mean(axis=1)
        out[scope] = {
            "n_paired": int(vals.size),
            "mean_diff": float(vals.mean()),
            "ci_lower_95": float(np.quantile(means, 0.025)),
            "ci_upper_95": float(np.quantile(means, 0.975)),
            "p_value_two_sided": float(2 * min((means <= 0).mean(), (means >= 0).mean())),
        }
    return out


def strata_points(ap: Dict[str, float], strata_df) -> Dict[str, float]:
    strat = dict(zip(strata_df["customer_id"].astype(str), strata_df["stratum"].astype(str)))
    out = {}
    for s in ("S0", "S1", "S2"):
        vals = [v for c, v in ap.items() if strat.get(c) == s]
        out[s] = float(np.mean(vals)) if vals else float("nan")
    return out


def run_domain_batch(tag: str, safe, arts, max_seq_len: int, ap_store: Dict[str, Any], out: Dict[str, Any]) -> None:
    eval_art = arts["W4"]
    train_ws = ["W1", "W2", "W3"]
    strata_df = eval_art["strata"]

    # L-invariant LightGBM baselines (train once, cached in ap_store)
    if f"{tag}_B1" not in ap_store:
        log(f"[{tag}] LightGBM B1/B2 (L-invariant)")
        train_feat = pd.concat([arts[tw]["feat"] for tw in train_ws], ignore_index=True)
        train_labels = pd.concat([arts[tw]["window"]["labels"] for tw in train_ws], ignore_index=True)
        b1 = R1LightGBMRanker(use_embeddings=False, n_estimators=250, random_state=SEED).fit(train_feat, train_labels)
        preds_b1 = b1.predict(eval_art["feat"])
        b2 = R1LightGBMRanker(use_embeddings=True, n_estimators=250, random_state=SEED).fit(train_feat, train_labels)
        preds_b2 = b2.predict(eval_art["feat"])
        labels = eval_art["window"]["labels"]
        history = eval_art["window"]["history"]
        ap_store[f"{tag}_B1"] = per_customer_ap(preds_b1, labels, history)
        ap_store[f"{tag}_B2"] = per_customer_ap(preds_b2, labels, history)

    key = f"{tag}_L{max_seq_len}"
    out.setdefault("batches", {}).setdefault(key, {"max_seq_len": max_seq_len, "models": {}})
    batch = out["batches"][key]
    labels = eval_art["window"]["labels"]
    history = eval_art["window"]["history"]

    for enc in ENCODERS:
        ck = f"{key}_{enc}"
        if ck in ap_store:
            log(f"[{key}] cached: {enc}")
        else:
            t0 = time.time()
            pack = train_seq(safe, eval_art["cutoff_date"], enc, max_seq_len=max_seq_len)
            preds, ev = track_a_eval(pack, eval_art)
            ap_store[ck] = per_customer_ap(preds, labels, history)
            batch["models"][enc] = {"window_eval": ev["map12"], "seconds": round(time.time() - t0, 1)}
            log(f"[{key}] trained+evaluated {enc} map12={ev['map12']:.4f} ({time.time()-t0:.0f}s)")

    aps = {enc: ap_store[f"{key}_{enc}"] for enc in ENCODERS}
    aps["B1_strong_lgbm"] = ap_store[f"{tag}_B1"]
    batch["point_map12"] = {m: float(np.mean(list(ap.values()))) for m, ap in aps.items()}
    batch["strata_points"] = {m: strata_points(ap, strata_df) for m, ap in aps.items()}
    contrasts = [
        ("gru", "mean_pool"), ("sasrec", "mean_pool"), ("hstu_aligned", "mean_pool"),
        ("hstu_aligned", "gru"), ("hstu_aligned", "sasrec"),
        ("gru", "B1_strong_lgbm"), ("sasrec", "B1_strong_lgbm"), ("hstu_aligned", "B1_strong_lgbm"),
    ]
    batch["paired_ci"] = {f"{a}_minus_{b}": paired_ci(aps[a], aps[b], strata_df) for a, b in contrasts}


def main() -> None:
    t0 = time.time()
    CACHE.mkdir(parents=True, exist_ok=True)
    out_path = REPO / "artifacts/eval/attention_decisive.json"
    out: Dict[str, Any] = {
        "contract_version": "0.9",
        "experiment": "attention_decisive",
        "protocol": {"encoders": ENCODERS, "loss": "sampled_softmax_256", "epochs": 8, "seed": SEED,
                      "embedding_dim": 128, "num_layers": 2, "num_heads": 4, "num_negatives": 256,
                      "batch_size": 256, "lr": 2e-3, "optimizer": "adamw_wd1e-4",
                      "hstu_alignment": "meta-recsys/generative-recommenders STULayer semantics, parity max_abs_diff 1.1e-06 (scripts/hstu_alignment_check.py)"},
        "batches": {},
    }
    if out_path.exists():
        out_prev = json.loads(out_path.read_text(encoding="utf-8"))
        out["batches"] = out_prev.get("batches", {})

    ap_path = CACHE / "ap_store.json"
    ap_store: Dict[str, Any] = json.loads(ap_path.read_text(encoding="utf-8")) if ap_path.exists() else {}

    def flush() -> None:
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        ap_path.write_text(json.dumps(ap_store), encoding="utf-8")

    log("[data] loading H&M")
    hm_events = load_hm_events(REPO / "data/hm/transactions_train.parquet")
    try:
        hm_sample, hm_meta = build_hm_enlarged_sample(hm_events, N_SAMPLE)
    except MemoryError:
        hm_sample, hm_meta = build_hm_enlarged_sample(hm_events, FALLBACK_N)
    out["hm_meta"] = hm_meta
    log(f"[data] hm sample: {hm_meta}")

    log("[windows] H&M 40k")
    hm_safe, hm_arts = build_windows(hm_sample, HM_CUTOFFS, HM_SEALED, tag="hm40k")

    log("[batch A] H&M L=50 (alignment correction check)")
    run_domain_batch("hm", hm_safe, hm_arts, max_seq_len=50, ap_store=ap_store, out=out)
    flush()

    log("[batch B] H&M L=200 (long sequence)")
    run_domain_batch("hm", hm_safe, hm_arts, max_seq_len=200, ap_store=ap_store, out=out)
    flush()

    try:
        import os
        if os.environ.get("RUN_SWIRE", "0") != "1":
            log("[swire] skipped (set RUN_SWIRE=1 to include swire batches)")
            out["swire_status"] = "SKIPPED_BY_USER"
            flush()
        else:
            log("[data] loading swire")
            sw_events = load_swire_events()
            sw_art = json.loads((REPO / "artifacts/r2/swire_ro_r2_results.json").read_text(encoding="utf-8"))
            log("[windows] swire full")
            sw_safe, sw_arts = build_windows(sw_events, sw_art["validation_cutoffs"], sw_art["sealed_cutoff"], tag="swire")
            log("[batch A] swire L=50")
            run_domain_batch("swire", sw_safe, sw_arts, max_seq_len=50, ap_store=ap_store, out=out)
            flush()
            log("[batch B] swire L=200")
            run_domain_batch("swire", sw_safe, sw_arts, max_seq_len=200, ap_store=ap_store, out=out)
            flush()
    except MemoryError:
        log("[swire] MemoryError — swire batches skipped, H&M results preserved")
        out["swire_status"] = "SKIPPED_MEMORYERROR"
        flush()
    except Exception:
        log("[swire] FAILED:\n" + traceback.format_exc())
        out["swire_status"] = "FAILED"
        flush()

    out["total_seconds"] = round(time.time() - t0, 1)
    flush()
    log(f"DONE in {out['total_seconds']}s")


if __name__ == "__main__":
    main()
