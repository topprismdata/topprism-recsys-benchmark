from datetime import datetime, timedelta
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.window_builder import assign_customer_strata, build_window
from src.eval.metrics import (
    evaluate_candidate_oracle,
    evaluate_map12,
    evaluate_repeat_explore,
)
from src.models.r1_baselines import (
    R1LightGBMRanker,
    compute_paired_bootstrap_ci,
    evaluate_r1_predictions,
    predict_b0_rule,
)
from src.models.r1_candidates import (
    _prepare_history,
    compute_svd_embeddings,
    generate_r1_candidates,
)
from src.models.r1_features import B1_FEATURE_COLS, B2_FEATURE_COLS, build_r1_feature_matrix
from src.models.r2_encoders import BaseSequentialEncoder, build_r2_encoder
from src.models.r2_losses import (
    BCE1Loss,
    BCE256Loss,
    SampledSoftmax256Loss,
    sample_negatives_excluding_positives,
)


HYBRID_NEURAL_FEATURE_COLS: List[str] = [
    "sequence_logit",
    "embedding_similarity",
]

R2_MATRIX_SPECS: List[Dict[str, str]] = [
    # Historical anchor (1-neg BCE)
    {
        "model_id": "M_SAS_BCE1",
        "encoder": "sasrec",
        "loss": "bce_1",
        "role": "historical_anchor",
    },
    # Diagnostic (MeanPool without position or attention)
    {
        "model_id": "M_POOL_BCE256",
        "encoder": "mean_pool",
        "loss": "bce_256",
        "role": "diagnostic",
    },
    {
        "model_id": "M_POOL_SS256",
        "encoder": "mean_pool",
        "loss": "sampled_softmax_256",
        "role": "diagnostic",
    },
    # Core 2x4 Orthogonal Matrix: {BCE-256, SampledSoftmax-256} x {GRU, SASRec, LiGR-SASRec, HSTU-Math}
    {
        "model_id": "M_GRU_BCE256",
        "encoder": "gru",
        "loss": "bce_256",
        "role": "orthogonal_2x4",
    },
    {
        "model_id": "M_GRU_SS256",
        "encoder": "gru",
        "loss": "sampled_softmax_256",
        "role": "orthogonal_2x4",
    },
    {
        "model_id": "M_SAS_BCE256",
        "encoder": "sasrec",
        "loss": "bce_256",
        "role": "orthogonal_2x4",
    },
    {
        "model_id": "M_SAS_SS256",
        "encoder": "sasrec",
        "loss": "sampled_softmax_256",
        "role": "orthogonal_2x4",
    },
    {
        "model_id": "M_LIGR_BCE256",
        "encoder": "ligr_sasrec",
        "loss": "bce_256",
        "role": "orthogonal_2x4",
    },
    {
        "model_id": "M_LIGR_SS256",
        "encoder": "ligr_sasrec",
        "loss": "sampled_softmax_256",
        "role": "orthogonal_2x4",
    },
    {
        "model_id": "M_HSTU_BCE256",
        "encoder": "hstu_math",
        "loss": "bce_256",
        "role": "orthogonal_2x4",
    },
    {
        "model_id": "M_HSTU_SS256",
        "encoder": "hstu_math",
        "loss": "sampled_softmax_256",
        "role": "orthogonal_2x4",
    },
]


def _build_vocab_and_logq(hist: pd.DataFrame) -> Dict[str, Any]:
    """Build deterministic 1-indexed item vocabulary and LogQ empirical log-probabilities from history."""
    if hist.empty:
        return {
            "articles": [],
            "art_to_id": {},
            "id_to_art": {},
            "item_log_probs": torch.zeros(2, dtype=torch.float32),
            "popular_articles": [],
        }

    articles = sorted(hist["article_id"].astype(str).unique().tolist())
    art_to_id = {a: i + 1 for i, a in enumerate(articles)}
    id_to_art = {i + 1: a for i, a in enumerate(articles)}

    counts = hist["article_id"].astype(str).value_counts()
    probs = np.zeros(len(articles) + 1, dtype=np.float32)
    probs[0] = 1e-12
    for a, idx in art_to_id.items():
        probs[idx] = float(counts.get(a, 1))
    probs[1:] = probs[1:] / max(float(probs[1:].sum()), 1.0)
    log_probs = torch.from_numpy(np.log(np.maximum(probs, 1e-12)).astype(np.float32))

    # Popular fallback list ordered by (-count, article_id)
    pop_df = (
        hist.groupby("article_id", as_index=False)
        .size()
        .sort_values(["size", "article_id"], ascending=[False, True], kind="mergesort")
    )
    popular_articles = pop_df["article_id"].astype(str).tolist()
    return {
        "articles": articles,
        "art_to_id": art_to_id,
        "id_to_art": id_to_art,
        "item_log_probs": log_probs,
        "popular_articles": popular_articles,
    }


def _build_user_sequences(
    hist: pd.DataFrame,
    art_to_id: Dict[str, int],
    max_seq_len: int = 50,
    sequence_order: str = "chronological",
    shuffle_seed: int = 42,
) -> Dict[str, Any]:
    """Build right-padded causal item sequences and relative day timestamps strictly from pre-cutoff history.

    ``sequence_order`` controls the per-user tail permutation applied jointly to items and
    timestamps before train/infer splits:
      - ``chronological`` (default): oldest-first true purchase order.
      - ``shuffled``: deterministic per-user random permutation seeded by ``shuffle_seed``,
        destroying purchase order and recency-position correlation (train+infer matched).
      - ``reversed``: newest-first order (recency inversion control).
    """
    if sequence_order not in ("chronological", "shuffled", "reversed"):
        raise ValueError(f"Unknown sequence_order '{sequence_order}'")
    if hist.empty or not art_to_id:
        return {
            "users": [],
            "user_to_row": {},
            "input_ids": torch.zeros((0, max_seq_len), dtype=torch.long),
            "timestamps": torch.zeros((0, max_seq_len), dtype=torch.float32),
            "target_ids": torch.zeros((0, max_seq_len), dtype=torch.long),
        }

    df = hist[["customer_id", "article_id", "event_date", "days_ago"]].copy()
    df["item_id"] = df["article_id"].map(art_to_id).fillna(0).astype(np.int64)
    df = df.loc[df["item_id"] > 0]
    if df.empty:
        return {
            "users": [],
            "user_to_row": {},
            "input_ids": torch.zeros((0, max_seq_len), dtype=torch.long),
            "timestamps": torch.zeros((0, max_seq_len), dtype=torch.float32),
            "target_ids": torch.zeros((0, max_seq_len), dtype=torch.long),
        }

    # Chronological order: descending days_ago (oldest first), ascending article_id within same day
    df = df.sort_values(
        ["customer_id", "days_ago", "article_id"],
        ascending=[True, False, True],
        kind="mergesort",
    )
    max_days = float(df["days_ago"].max())
    df["rel_day"] = max_days - df["days_ago"].to_numpy(dtype=np.float32)

    # Keep last max_seq_len + 1 events per customer to form (input_ids, next-step / next-basket target_ids)
    tail_df = df.groupby("customer_id", sort=False).tail(max_seq_len + 1)

    users = sorted(tail_df["customer_id"].unique().tolist())
    user_to_row = {u: idx for idx, u in enumerate(users)}
    n_users = len(users)
    order_rng = np.random.default_rng(shuffle_seed)

    input_mat = np.zeros((n_users, max_seq_len), dtype=np.int64)
    ts_mat = np.zeros((n_users, max_seq_len), dtype=np.float32)
    target_mat = np.zeros((n_users, max_seq_len), dtype=np.int64)

    # Also build inference sequences (strictly the most recent max_seq_len items up to cutoff)
    infer_input_mat = np.zeros((n_users, max_seq_len), dtype=np.int64)
    infer_ts_mat = np.zeros((n_users, max_seq_len), dtype=np.float32)

    for u, grp in tail_df.groupby("customer_id", sort=False):
        r = user_to_row[u]
        items = grp["item_id"].to_numpy(dtype=np.int64)
        ts = grp["rel_day"].to_numpy(dtype=np.float32)
        if sequence_order == "shuffled" and len(items) >= 2:
            perm = order_rng.permutation(len(items))
            items = items[perm]
            ts = ts[perm]
        elif sequence_order == "reversed" and len(items) >= 2:
            items = items[::-1]
            ts = ts[::-1]

        # Inference sequence: last max_seq_len items
        inf_items = items[-max_seq_len:]
        inf_ts = ts[-max_seq_len:]
        L_inf = len(inf_items)
        infer_input_mat[r, :L_inf] = inf_items
        infer_ts_mat[r, :L_inf] = inf_ts
        if L_inf < max_seq_len and L_inf > 0:
            infer_ts_mat[r, L_inf:] = inf_ts[-1]

        # Cross-basket autoregressive training sequence: each step t predicts an item from the next basket date > ts[t]
        if len(items) >= 2:
            inp = items[:-1][-max_seq_len:]
            inp_ts = ts[:-1][-max_seq_len:]
            L_tr = len(inp)
            # Find next basket start index for each position; fall back to immediate next token if same day
            nxt = np.empty(L_tr, dtype=np.int64)
            full_tail_items = items[-(L_tr + 1) :]
            full_tail_ts = ts[-(L_tr + 1) :]
            for pos_i in range(L_tr):
                cur_t = full_tail_ts[pos_i]
                later_loc = np.flatnonzero(full_tail_ts[pos_i + 1 :] > cur_t)
                if len(later_loc) > 0:
                    first_later = pos_i + 1 + int(later_loc[0])
                    next_b_day = full_tail_ts[first_later]
                    b_items = full_tail_items[first_later:][full_tail_ts[first_later:] == next_b_day]
                    nxt[pos_i] = b_items[pos_i % len(b_items)]
                else:
                    nxt[pos_i] = full_tail_items[pos_i + 1]
            input_mat[r, :L_tr] = inp
            ts_mat[r, :L_tr] = inp_ts
            target_mat[r, :L_tr] = nxt
            if L_tr < max_seq_len and L_tr > 0:
                ts_mat[r, L_tr:] = inp_ts[-1]
        else:
            # Single-item sequence: self-prediction anchor so S0 single-item customers still have valid targets
            input_mat[r, 0] = items[0]
            ts_mat[r, :] = ts[0]
            target_mat[r, 0] = items[0]

    return {
        "users": users,
        "user_to_row": user_to_row,
        "input_ids": torch.from_numpy(input_mat),
        "timestamps": torch.from_numpy(ts_mat),
        "target_ids": torch.from_numpy(target_mat),
        "infer_input_ids": torch.from_numpy(infer_input_mat),
        "infer_timestamps": torch.from_numpy(infer_ts_mat),
    }


def _build_historical_horizon_samples(
    events: pd.DataFrame,
    cutoff_date: str,
    art_to_id: Dict[str, int],
    max_seq_len: int = 50,
    horizon_days: int = 7,
    num_subwindows: int = 3,
    max_pos_per_user_window: int = 16,
) -> Optional[Dict[str, Any]]:
    """Build grouped self-supervised next-7d basket training batches strictly inside event_date <= cutoff_date."""
    dt_cut = datetime.strptime(cutoff_date, "%Y-%m-%d")
    user_inputs: List[np.ndarray] = []
    user_ts: List[np.ndarray] = []
    pos_user_offsets: List[int] = []
    pos_item_ids: List[int] = []

    safe_hist = events.loc[events["event_date"].astype(str) <= str(cutoff_date)].copy()
    if safe_hist.empty or not art_to_id:
        return None

    for w_idx in range(1, num_subwindows + 1):
        sub_end = dt_cut - timedelta(days=(w_idx - 1) * horizon_days)
        sub_cut = sub_end - timedelta(days=horizon_days)
        sub_cut_str = sub_cut.strftime("%Y-%m-%d")
        sub_end_str = sub_end.strftime("%Y-%m-%d")

        hist_w = _prepare_history(safe_hist, cutoff_date=sub_cut_str)
        if hist_w.empty:
            continue
        seq_pack = _build_user_sequences(hist_w, art_to_id=art_to_id, max_seq_len=max_seq_len)
        if not seq_pack["users"]:
            continue

        lbl_mask = (safe_hist["event_date"].astype(str) > sub_cut_str) & (
            safe_hist["event_date"].astype(str) <= sub_end_str
        )
        lbl_df = safe_hist.loc[lbl_mask, ["customer_id", "article_id"]].drop_duplicates()
        if lbl_df.empty:
            continue

        u_to_row = seq_pack["user_to_row"]
        inf_ids_np = seq_pack["infer_input_ids"].numpy()
        inf_ts_np = seq_pack["infer_timestamps"].numpy()

        lbl_df["customer_id"] = lbl_df["customer_id"].astype(str)
        lbl_df["article_id"] = lbl_df["article_id"].astype(str)
        lbl_df = lbl_df.loc[lbl_df["customer_id"].isin(u_to_row) & lbl_df["article_id"].isin(art_to_id)]
        if lbl_df.empty:
            continue

        for u, grp in lbl_df.groupby("customer_id", sort=False):
            r = u_to_row[u]
            pos_arts = sorted(grp["article_id"].tolist())[:max_pos_per_user_window]
            if not pos_arts:
                continue
            u_slot = len(user_inputs)
            user_inputs.append(inf_ids_np[r])
            user_ts.append(inf_ts_np[r])
            for a in pos_arts:
                pos_user_offsets.append(u_slot)
                pos_item_ids.append(int(art_to_id[a]))

    if not pos_item_ids:
        return None

    return {
        "user_input_ids": torch.from_numpy(np.stack(user_inputs, axis=0)),
        "user_timestamps": torch.from_numpy(np.stack(user_ts, axis=0)),
        "pos_user_offsets": torch.tensor(pos_user_offsets, dtype=torch.long),
        "pos_item_ids": torch.tensor(pos_item_ids, dtype=torch.long),
    }


def prepare_r2_cutoff_dataset(
    events: pd.DataFrame,
    cutoff_date: str,
    max_seq_len: int = 50,
    sequence_order: str = "chronological",
    shuffle_seed: int = 42,
) -> Dict[str, Any]:
    """Precompute vocabulary, LogQ table, user sequence tensors, and historical 7d horizon batches once per cutoff."""
    hist = _prepare_history(events, cutoff_date=cutoff_date)
    vocab = _build_vocab_and_logq(hist)
    seq_pack = _build_user_sequences(
        hist, art_to_id=vocab["art_to_id"], max_seq_len=max_seq_len,
        sequence_order=sequence_order, shuffle_seed=shuffle_seed,
    )
    horizon_pack = _build_historical_horizon_samples(
        events=events,
        cutoff_date=cutoff_date,
        art_to_id=vocab["art_to_id"],
        max_seq_len=max_seq_len,
    )
    return {
        "hist": hist,
        "vocab": vocab,
        "seq_pack": seq_pack,
        "horizon_pack": horizon_pack,
    }

def _build_loss_module(loss_name: str, num_negatives: int = 256, logq_correction: bool = True) -> nn.Module:
    key = loss_name.strip().lower()
    if key == "bce_1":
        return BCE1Loss()
    if key == "bce_256":
        return BCE256Loss(num_negatives=num_negatives)
    if key in ("sampled_softmax_256", "ss_256"):
        return SampledSoftmax256Loss(num_negatives=num_negatives, logq_correction=logq_correction)
    raise ValueError(f"Unknown loss_name '{loss_name}'")


def train_r2_sequential_model(
    events: pd.DataFrame,
    cutoff_date: str,
    encoder_name: str,
    loss_name: str,
    embedding_dim: int = 128,
    max_seq_len: int = 50,
    num_layers: int = 2,
    num_heads: int = 4,
    num_negatives: int = 256,
    batch_size: int = 256,
    epochs: int = 8,
    learning_rate: float = 2e-3,
    dropout: float = 0.1,
    seed: int = 42,
    device: Optional[str] = None,
    precomputed_data: Optional[Dict[str, Any]] = None,
    sequence_order: str = "chronological",
    shuffle_seed: int = 42,
) -> Dict[str, Any]:
    """Train a single R2 sequential model strictly on events with event_date <= cutoff_date.

    ``sequence_order``/``shuffle_seed`` are only used when ``precomputed_data`` is not given;
    pass a precomputed pack built with the desired order otherwise.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dev_str = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dev = torch.device(dev_str)

    if precomputed_data is not None:
        vocab = precomputed_data["vocab"]
        seq_pack = precomputed_data["seq_pack"]
        horizon_pack = precomputed_data["horizon_pack"]
    else:
        hist = _prepare_history(events, cutoff_date=cutoff_date)
        vocab = _build_vocab_and_logq(hist)
        seq_pack = _build_user_sequences(
            hist, art_to_id=vocab["art_to_id"], max_seq_len=max_seq_len,
            sequence_order=sequence_order, shuffle_seed=shuffle_seed,
        )
        horizon_pack = _build_historical_horizon_samples(
            events=events,
            cutoff_date=cutoff_date,
            art_to_id=vocab["art_to_id"],
            max_seq_len=max_seq_len,
        )
    num_items = max(len(vocab["articles"]), 1)

    model = build_r2_encoder(
        encoder_name=encoder_name,
        num_items=num_items,
        embedding_dim=embedding_dim,
        max_seq_len=max_seq_len,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
    ).to(dev)

    if len(seq_pack["users"]) == 0 or len(vocab["articles"]) == 0:
        model.eval()
        return {
            "model": model,
            "vocab": vocab,
            "seq_pack": seq_pack,
            "train_loss_history": [],
            "train_wall_time_sec": 0.0,
            "device": str(dev),
        }

    criterion = _build_loss_module(loss_name, num_negatives=num_negatives, logq_correction=True).to(dev)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    input_ids_all = seq_pack["input_ids"].to(dev)
    timestamps_all = seq_pack["timestamps"].to(dev)
    target_ids_all = seq_pack["target_ids"].to(dev)
    item_log_probs = vocab["item_log_probs"].to(dev)

    n_seq = input_ids_all.size(0)
    k_neg = 1 if loss_name.strip().lower() == "bce_1" else int(num_negatives)
    gen = torch.Generator(device=dev)
    gen.manual_seed(seed)

    loss_history: List[float] = []
    t0 = time.perf_counter()
    model.train()

    for epoch in range(int(epochs)):
        perm = torch.randperm(n_seq, generator=gen, device=dev)
        epoch_losses: List[float] = []

        for start in range(0, n_seq, batch_size):
            idx = perm[start : start + batch_size]
            b_inp = input_ids_all[idx]
            b_ts = timestamps_all[idx]
            b_tgt = target_ids_all[idx]

            valid_mask = b_tgt > 0
            if not valid_mask.any():
                continue

            optimizer.zero_grad(set_to_none=True)
            seq_repr = model(b_inp, timestamps=b_ts)

            flat_repr = seq_repr[valid_mask]
            flat_pos = b_tgt[valid_mask]
            flat_neg = sample_negatives_excluding_positives(
                pos_ids=flat_pos,
                num_items=num_items,
                num_negatives=k_neg,
                generator=gen,
            )

            if isinstance(criterion, SampledSoftmax256Loss):
                loss = criterion(
                    flat_repr,
                    model.item_embedding.weight,
                    flat_pos,
                    flat_neg,
                    item_log_probs=item_log_probs,
                )
            else:
                loss = criterion(
                    flat_repr,
                    model.item_embedding.weight,
                    flat_pos,
                    flat_neg,
                )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().item()))

        # Self-supervised 7-day horizon step on encode_last_step (grouped by unique customer-window, 0 redundancy)
        if horizon_pack is not None and horizon_pack["pos_item_ids"].numel() > 0:
            h_u_inp_all = horizon_pack["user_input_ids"].to(dev)
            h_u_ts_all = horizon_pack["user_timestamps"].to(dev)
            h_pos_u_all = horizon_pack["pos_user_offsets"].to(dev)
            h_pos_i_all = horizon_pack["pos_item_ids"].to(dev)
            n_h_users = h_u_inp_all.size(0)

            for u_start in range(0, n_h_users, batch_size):
                u_end = min(u_start + batch_size, n_h_users)
                pos_mask = (h_pos_u_all >= u_start) & (h_pos_u_all < u_end)
                if not pos_mask.any():
                    continue
                b_u_inp = h_u_inp_all[u_start:u_end]
                b_u_ts = h_u_ts_all[u_start:u_end]
                local_u_idx = h_pos_u_all[pos_mask] - u_start
                b_pos_items = h_pos_i_all[pos_mask]

                optimizer.zero_grad(set_to_none=True)
                u_last_batch = model.encode_last_step(b_u_inp, timestamps=b_u_ts)
                u_last_expanded = u_last_batch[local_u_idx]
                h_neg = sample_negatives_excluding_positives(
                    pos_ids=b_pos_items,
                    num_items=num_items,
                    num_negatives=k_neg,
                    generator=gen,
                )
                if isinstance(criterion, SampledSoftmax256Loss):
                    h_loss = criterion(
                        u_last_expanded,
                        model.item_embedding.weight,
                        b_pos_items,
                        h_neg,
                        item_log_probs=item_log_probs,
                    )
                else:
                    h_loss = criterion(
                        u_last_expanded,
                        model.item_embedding.weight,
                        b_pos_items,
                        h_neg,
                    )
                h_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                epoch_losses.append(float(h_loss.detach().item()))

        loss_history.append(float(np.mean(epoch_losses)) if epoch_losses else 0.0)

    if dev.type == "cuda":
        torch.cuda.synchronize()
    wall_sec = time.perf_counter() - t0
    model.eval()
    return {
        "model": model,
        "vocab": vocab,
        "seq_pack": seq_pack,
        "train_loss_history": loss_history,
        "train_wall_time_sec": float(wall_sec),
        "device": str(dev),
    }


@torch.no_grad()
def _compute_all_user_representations(
    trained_pack: Dict[str, Any],
    batch_size: int = 512,
) -> Tuple[Dict[str, int], np.ndarray, np.ndarray]:
    """Return user_to_row, raw user vectors (U, d), and L2-normalized user vectors (U, d)."""
    model: BaseSequentialEncoder = trained_pack["model"]
    seq_pack = trained_pack["seq_pack"]
    users = seq_pack["users"]
    if not users:
        dim = model.embedding_dim
        return {}, np.zeros((0, dim), dtype=np.float32), np.zeros((0, dim), dtype=np.float32)

    dev = next(model.parameters()).device
    inf_ids = seq_pack["infer_input_ids"]
    inf_ts = seq_pack["infer_timestamps"]
    n_users = inf_ids.size(0)

    vecs = []
    for start in range(0, n_users, batch_size):
        b_ids = inf_ids[start : start + batch_size].to(dev)
        b_ts = inf_ts[start : start + batch_size].to(dev)
        u_vec = model.encode_last_step(b_ids, timestamps=b_ts)
        vecs.append(u_vec.detach().cpu().numpy().astype(np.float32))

    u_mat = np.concatenate(vecs, axis=0)
    u_norms = np.maximum(np.linalg.norm(u_mat, axis=1, keepdims=True), 1e-8)
    u_unit = u_mat / u_norms
    return seq_pack["user_to_row"], u_mat, u_unit


@torch.no_grad()
def compute_neural_oof_features(
    trained_pack: Dict[str, Any],
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    """Compute sequence_logit = <u_t, e_i> and embedding_similarity = cos(u_t, e_i) for candidate pairs."""
    out = candidates[["customer_id", "article_id"]].copy()
    out["customer_id"] = out["customer_id"].astype(str)
    out["article_id"] = out["article_id"].astype(str)
    n_rows = len(out)
    if n_rows == 0:
        out["sequence_logit"] = np.zeros(0, dtype=np.float32)
        out["embedding_similarity"] = np.zeros(0, dtype=np.float32)
        return out

    model: BaseSequentialEncoder = trained_pack["model"]
    vocab = trained_pack["vocab"]
    art_to_id = vocab["art_to_id"]
    u_to_row, u_mat, u_unit = _compute_all_user_representations(trained_pack)

    item_mat = model.item_embedding.weight.detach().cpu().numpy().astype(np.float32)
    item_norms = np.maximum(np.linalg.norm(item_mat, axis=1, keepdims=True), 1e-8)
    item_unit = item_mat / item_norms

    u_idx = out["customer_id"].map(u_to_row).fillna(-1).to_numpy(dtype=np.int64)
    i_idx = out["article_id"].map(art_to_id).fillna(0).to_numpy(dtype=np.int64)
    valid = (u_idx >= 0) & (i_idx > 0)

    logits = np.zeros(n_rows, dtype=np.float32)
    sims = np.zeros(n_rows, dtype=np.float32)
    if valid.any():
        v_loc = np.flatnonzero(valid)
        u_v = u_idx[v_loc]
        i_v = i_idx[v_loc]
        logits[v_loc] = np.sum(u_mat[u_v] * item_mat[i_v], axis=1)
        sims[v_loc] = np.sum(u_unit[u_v] * item_unit[i_v], axis=1)

    out["sequence_logit"] = logits
    out["embedding_similarity"] = sims
    return out


def predict_track_a_candidates(
    trained_pack: Dict[str, Any],
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    """Track A: Fixed Candidate Pool Reranking via sequence_logit = <u_t, e_i>."""
    scored = compute_neural_oof_features(trained_pack, candidates)
    out = pd.DataFrame(
        {
            "customer_id": scored["customer_id"],
            "article_id": scored["article_id"],
            "score": scored["sequence_logit"].astype(float),
        }
    )
    if "cutoff_date" in candidates.columns:
        out["cutoff_date"] = candidates["cutoff_date"].astype(str).to_numpy()
    return out


@torch.no_grad()
def predict_track_b_full_catalog(
    trained_pack: Dict[str, Any],
    customer_ids: Sequence[str],
    top_k: int = 12,
    user_batch_size: int = 512,
) -> pd.DataFrame:
    """Track B: Full-catalog inner-product MIPS retrieval Top-K with popular fallback for S0 zero-history users."""
    target_users = [str(u) for u in customer_ids]
    if not target_users:
        return pd.DataFrame(columns=["customer_id", "article_id", "score"])

    model: BaseSequentialEncoder = trained_pack["model"]
    vocab = trained_pack["vocab"]
    articles = vocab["articles"]
    popular_articles = vocab["popular_articles"][:top_k]
    if not articles:
        return pd.DataFrame(columns=["customer_id", "article_id", "score"])

    k_take = min(int(top_k), len(articles))
    u_to_row, u_mat, _ = _compute_all_user_representations(trained_pack, batch_size=user_batch_size)

    dev = next(model.parameters()).device
    item_emb = model.item_embedding.weight[1:].detach().to(dev)  # (V, d)
    articles_arr = np.array(articles, dtype=object)

    out_u: List[str] = []
    out_a: List[str] = []
    out_s: List[float] = []

    warm_users: List[str] = []
    warm_rows: List[int] = []
    cold_users: List[str] = []

    for u in target_users:
        if u in u_to_row:
            warm_users.append(u)
            warm_rows.append(u_to_row[u])
        else:
            cold_users.append(u)

    # Full-catalog matrix inner product on GPU/CPU in batches
    if warm_users:
        u_warm_tensor = torch.from_numpy(u_mat[warm_rows]).to(dev)
        for start in range(0, len(warm_users), user_batch_size):
            b_u = warm_users[start : start + user_batch_size]
            b_vec = u_warm_tensor[start : start + user_batch_size]
            scores_mat = torch.matmul(b_vec, item_emb.T)  # (B, V)
            top_vals, top_idx = torch.topk(scores_mat, k=k_take, dim=1, largest=True, sorted=True)
            top_vals_np = top_vals.cpu().numpy()
            top_idx_np = top_idx.cpu().numpy()
            for row_i, u_id in enumerate(b_u):
                out_u.extend([u_id] * k_take)
                out_a.extend(articles_arr[top_idx_np[row_i]].tolist())
                out_s.extend(top_vals_np[row_i].astype(float).tolist())

    # Cold-start S0 fallback to historical popular items
    if cold_users and popular_articles:
        k_pop = len(popular_articles)
        pop_scores = [float(k_pop - idx) for idx in range(k_pop)]
        for u_id in cold_users:
            out_u.extend([u_id] * k_pop)
            out_a.extend(popular_articles)
            out_s.extend(pop_scores)

    return pd.DataFrame({"customer_id": out_u, "article_id": out_a, "score": out_s})


def evaluate_track_b_predictions(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    history: pd.DataFrame,
    strata_df: pd.DataFrame,
    k: int = 12,
) -> Dict[str, Any]:
    """Evaluate Track B full-catalog predictions (unrestricted by candidate pool) with strata & Repeat/Explore."""
    all_custs = strata_df["customer_id"].astype(str).unique().tolist() if not strata_df.empty else None
    overall = evaluate_map12(predictions, labels, all_customers=all_custs, k=k)
    rep_exp = evaluate_repeat_explore(predictions, labels, history, k=k)

    s_map = (
        dict(zip(strata_df["customer_id"].astype(str), strata_df["stratum"].astype(str)))
        if not strata_df.empty
        else {}
    )
    labels_with_s = labels.copy()
    labels_with_s["customer_id"] = labels_with_s["customer_id"].astype(str)
    labels_with_s["_stratum"] = labels_with_s["customer_id"].map(s_map).fillna("S0")

    strata_report: Dict[str, Dict[str, Any]] = {}
    for s_name in ("S0", "S1", "S2"):
        sub_labels = labels_with_s.loc[labels_with_s["_stratum"] == s_name, ["customer_id", "article_id"]]
        s_cust_all = [u for u, st in s_map.items() if st == s_name]
        if sub_labels.empty:
            strata_report[s_name] = {
                "map12": 0.0,
                "all_customer_map12": 0.0,
                "map12_repeat": 0.0,
                "map12_explore": 0.0,
                "evaluated_customers": 0,
                "total_stratum_customers": len(s_cust_all),
            }
        else:
            s_ov = evaluate_map12(predictions, sub_labels, all_customers=s_cust_all, k=k)
            s_re = evaluate_repeat_explore(predictions, sub_labels, history, k=k)
            strata_report[s_name] = {
                "map12": float(s_ov["map12"]),
                "all_customer_map12": float(s_ov["all_customer_map12"]),
                "map12_repeat": float(s_re["map12_repeat"]),
                "map12_explore": float(s_re["map12_explore"]),
                "evaluated_customers": int(s_ov["evaluated_customers"]),
                "total_stratum_customers": len(s_cust_all),
            }

    return {
        "map12": float(overall["map12"]),
        "evaluated_customers": int(overall["evaluated_customers"]),
        "all_customer_map12": float(overall["all_customer_map12"]),
        "repeat_explore": rep_exp,
        "strata": strata_report,
    }


def evaluate_sequence_order_shuffle_ablation(
    trained_pack: Dict[str, Any],
    eval_art: Dict[str, Any],
    eval_a_true: Dict[str, Any],
    eval_b_true: Dict[str, Any],
    preds_a_true: pd.DataFrame,
    encoder_name: str,
    loss_name: str,
    seed: int = 42,
    n_bootstrap: int = 200,
) -> Dict[str, Any]:
    """Evaluate true chronological order vs shuffled sequence order on overall and S2 customers."""
    seq_pack = trained_pack["seq_pack"]
    inf_ids_np = seq_pack["infer_input_ids"].detach().cpu().numpy().copy()
    rng = np.random.default_rng(seed + 777)

    for r in range(inf_ids_np.shape[0]):
        valid_loc = np.flatnonzero(inf_ids_np[r] > 0)
        if len(valid_loc) >= 2:
            perm = rng.permutation(len(valid_loc))
            inf_ids_np[r, valid_loc] = inf_ids_np[r, valid_loc[perm]]

    shuffled_seq_pack = {
        **seq_pack,
        "infer_input_ids": torch.from_numpy(inf_ids_np),
    }
    shuffled_trained_pack = {
        **trained_pack,
        "seq_pack": shuffled_seq_pack,
    }

    preds_a_shuf = predict_track_a_candidates(shuffled_trained_pack, eval_art["cands"])
    eval_a_shuf = evaluate_r1_predictions(
        preds_a_shuf,
        candidates=eval_art["cands"],
        labels=eval_art["window"]["labels"],
        history=eval_art["window"]["history"],
        strata_df=eval_art["strata"],
        k=12,
    )

    preds_b_shuf = predict_track_b_full_catalog(
        shuffled_trained_pack,
        customer_ids=eval_art["target_users"],
        top_k=12,
    )
    eval_b_shuf = evaluate_track_b_predictions(
        preds_b_shuf,
        labels=eval_art["window"]["labels"],
        history=eval_art["window"]["history"],
        strata_df=eval_art["strata"],
        k=12,
    )

    strata_df = eval_art["strata"]
    s2_custs = set(
        strata_df.loc[strata_df["stratum"].astype(str) == "S2", "customer_id"].astype(str).tolist()
    )
    labels_df = eval_art["window"]["labels"]
    s2_labels = labels_df.loc[labels_df["customer_id"].astype(str).isin(s2_custs)]
    s2_ci = compute_paired_bootstrap_ci(
        preds_a_shuf,
        preds_a_true,
        s2_labels if not s2_labels.empty else labels_df,
        k=12,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    s2_a_true = float(eval_a_true["strata"]["S2"]["map12"])
    s2_a_shuf = float(eval_a_shuf["strata"]["S2"]["map12"])
    s2_b_true = float(eval_b_true["strata"]["S2"]["map12"])
    s2_b_shuf = float(eval_b_shuf["strata"]["S2"]["map12"])

    return {
        "encoder": encoder_name,
        "loss": loss_name,
        "track_a_overall_true_order_map12": float(eval_a_true["map12"]),
        "track_a_overall_shuffled_order_map12": float(eval_a_shuf["map12"]),
        "track_a_overall_delta_true_minus_shuffled": float(eval_a_true["map12"] - eval_a_shuf["map12"]),
        "track_a_s2_true_order_map12": s2_a_true,
        "track_a_s2_shuffled_order_map12": s2_a_shuf,
        "track_a_s2_delta_true_minus_shuffled": float(s2_a_true - s2_a_shuf),
        "track_a_explore_true_order_ap12": float(eval_a_true["repeat_explore"]["map12_explore"]),
        "track_a_explore_shuffled_order_ap12": float(eval_a_shuf["repeat_explore"]["map12_explore"]),
        "track_b_overall_true_order_map12": float(eval_b_true["map12"]),
        "track_b_overall_shuffled_order_map12": float(eval_b_shuf["map12"]),
        "track_b_overall_delta_true_minus_shuffled": float(eval_b_true["map12"] - eval_b_shuf["map12"]),
        "track_b_s2_true_order_map12": s2_b_true,
        "track_b_s2_shuffled_order_map12": s2_b_shuf,
        "track_b_s2_delta_true_minus_shuffled": float(s2_b_true - s2_b_shuf),
        "s2_paired_bootstrap_ci_true_vs_shuffled_track_a": s2_ci,
    }

class R2HybridLightGBMRanker(R1LightGBMRanker):
    """Track C Hybrid GBDT: B1 Tabular/Decay/Cooc features + Out-of-Time Neural sequence_logit & embedding_similarity."""

    DERIVED_NEURAL_COLS: List[str] = [
        "sequence_logit_cust_norm",
        "embedding_similarity_cust_norm",
        "sequence_logit_cust_z",
        "neural_sim_x_strike",
        "neural_sim_x_cooc",
    ]

    def __init__(
        self,
        extra_neural_cols: Optional[Sequence[str]] = None,
        use_svd_embeddings: bool = False,
        n_estimators: int = 250,
        learning_rate: float = 0.05,
        num_leaves: int = 31,
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        super().__init__(
            use_embeddings=use_svd_embeddings,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        base_cols = list(B2_FEATURE_COLS if use_svd_embeddings else B1_FEATURE_COLS)
        neural_cols = list(extra_neural_cols) if extra_neural_cols is not None else list(HYBRID_NEURAL_FEATURE_COLS)
        all_extra = neural_cols + [c for c in self.DERIVED_NEURAL_COLS if c not in neural_cols]
        self.feature_cols = base_cols + [c for c in all_extra if c not in base_cols]

    @staticmethod
    def _augment_hybrid_df(feature_df: pd.DataFrame) -> pd.DataFrame:
        df = feature_df.copy()
        if df.empty:
            for c in R2HybridLightGBMRanker.DERIVED_NEURAL_COLS:
                df[c] = 0.0
            return df

        grp_keys = ["customer_id", "cutoff_date"] if "cutoff_date" in df.columns else ["customer_id"]
        for raw_col, norm_col in [
            ("sequence_logit", "sequence_logit_cust_norm"),
            ("embedding_similarity", "embedding_similarity_cust_norm"),
        ]:
            vals = np.nan_to_num(df[raw_col].to_numpy(dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
            c_min = df.groupby(grp_keys, sort=False)[raw_col].transform("min").to_numpy(dtype=np.float32)
            c_max = df.groupby(grp_keys, sort=False)[raw_col].transform("max").to_numpy(dtype=np.float32)
            span = c_max - c_min
            df[norm_col] = np.where(span > 1e-8, (vals - c_min) / np.maximum(span, 1e-8), 0.0).astype(np.float32)

        logits = np.nan_to_num(df["sequence_logit"].to_numpy(dtype=np.float32), nan=0.0)
        c_mean = df.groupby(grp_keys, sort=False)["sequence_logit"].transform("mean").to_numpy(dtype=np.float32)
        c_std = df.groupby(grp_keys, sort=False)["sequence_logit"].transform("std").fillna(0.0).to_numpy(dtype=np.float32)
        df["sequence_logit_cust_z"] = np.where(c_std > 1e-6, (logits - c_mean) / np.maximum(c_std, 1e-6), 0.0).astype(
            np.float32
        )

        strike = np.log1p(np.maximum(df["up_order_strike"].to_numpy(dtype=np.float32), 0.0)) if "up_order_strike" in df.columns else 0.0
        cooc = np.log1p(np.maximum(df["cooccur_score"].to_numpy(dtype=np.float32), 0.0)) if "cooccur_score" in df.columns else 0.0
        sim_norm = df["embedding_similarity_cust_norm"].to_numpy(dtype=np.float32)
        df["neural_sim_x_strike"] = (sim_norm * strike).astype(np.float32)
        df["neural_sim_x_cooc"] = (sim_norm * cooc).astype(np.float32)
        return df

    def fit(
        self,
        feature_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        active_purchasers_only: bool = True,
    ) -> "R2HybridLightGBMRanker":
        aug_df = self._augment_hybrid_df(feature_df)
        super().fit(aug_df, labels_df, active_purchasers_only=active_purchasers_only)
        return self

    def predict(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        aug_df = self._augment_hybrid_df(feature_df)
        return super().predict(aug_df)
def run_r2_orthogonal_benchmark(
    events: pd.DataFrame,
    validation_cutoffs: Sequence[str],
    sealed_cutoff: str,
    horizon_days: int = 7,
    k_pool: int = 100,
    half_life_days: float = 14.0,
    svd_dim: int = 32,
    embedding_dim: int = 128,
    max_seq_len: int = 50,
    num_layers: int = 2,
    num_heads: int = 4,
    num_negatives: int = 256,
    batch_size: int = 256,
    epochs: int = 8,
    learning_rate: float = 2e-3,
    n_estimators: int = 250,
    n_bootstrap: int = 1000,
    seed: int = 42,
    device: Optional[str] = None,
    matrix_specs: Optional[Sequence[Dict[str, str]]] = None,
) -> Dict[str, Any]:
    """Run full R2 2x4+1 sequential orthogonal matrix + Track A/B + Track C Hybrid without touching W5."""
    val_cutoffs = list(validation_cutoffs)
    if len(val_cutoffs) < 2:
        raise ValueError("At least 2 validation cutoffs are required for Out-of-Time Track C Hybrid training")

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

    specs = list(matrix_specs) if matrix_specs is not None else list(R2_MATRIX_SPECS)

    # 1. Build R1 candidate pools and zero-leakage tabular features across W1..W4
    window_artifacts: Dict[str, Dict[str, Any]] = {}
    w_names = [f"W{i}" for i in range(1, len(val_cutoffs) + 1)]
    for idx, cutoff in enumerate(val_cutoffs, start=1):
        w_name = f"W{idx}"
        w = build_window(safe_events, cutoff_date=cutoff, horizon_days=horizon_days)
        strata_df = assign_customer_strata(safe_events, cutoff_date=cutoff)
        target_users = strata_df["customer_id"].astype(str).tolist()

        hist_prep = _prepare_history(safe_events, cutoff_date=cutoff)
        svd_pack = compute_svd_embeddings(
            hist_prep,
            embedding_dim=svd_dim,
            half_life_days=half_life_days,
            random_state=seed,
        )
        per_src_k = max(int(k_pool), 200) if int(k_pool) >= 100 else int(k_pool)
        cands_k = generate_r1_candidates(
            safe_events,
            cutoff_date=cutoff,
            customer_ids=target_users,
            k_pool=k_pool,
            per_source_k=per_src_k,
            half_life_days=half_life_days,
            embedding_dim=svd_dim,
            random_state=seed,
            svd_pack=svd_pack,
        )
        feat_k = build_r1_feature_matrix(
            safe_events,
            cands_k,
            cutoff_date=cutoff,
            half_life_days=half_life_days,
            embedding_dim=svd_dim,
            random_state=seed,
            svd_pack=svd_pack,
        )
        window_artifacts[w_name] = {
            "cutoff_date": cutoff,
            "window": w,
            "strata": strata_df,
            "target_users": target_users,
            "cands": cands_k,
            "feat": feat_k,
        }

    eval_w = w_names[-1]
    train_ws = w_names[:-1]
    eval_art = window_artifacts[eval_w]
    eval_cutoff = eval_art["cutoff_date"]

    # 2. Train all 2x4 + 1 (+ diagnostic) sequential models at W4 cutoff (strictly event_date <= eval_cutoff)
    track_a_results: Dict[str, Any] = {}
    track_b_results: Dict[str, Any] = {}
    model_diagnostics: Dict[str, Any] = {}
    shuffle_ablation_results: Dict[str, Any] = {}
    w4_neural_preds_track_a: Dict[str, pd.DataFrame] = {}
    w4_neural_oof_cols: Dict[str, pd.DataFrame] = {}
    eval_precomputed = prepare_r2_cutoff_dataset(
        events=safe_events,
        cutoff_date=eval_cutoff,
        max_seq_len=max_seq_len,
    )

    for spec in specs:
        m_id = spec["model_id"]
        enc_name = spec["encoder"]
        loss_name = spec["loss"]

        trained = train_r2_sequential_model(
            events=safe_events,
            cutoff_date=eval_cutoff,
            encoder_name=enc_name,
            loss_name=loss_name,
            embedding_dim=embedding_dim,
            max_seq_len=max_seq_len,
            num_layers=num_layers,
            num_heads=num_heads,
            num_negatives=num_negatives,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            seed=seed,
            device=device,
            precomputed_data=eval_precomputed,
        )
        oof_df = compute_neural_oof_features(trained, eval_art["cands"])
        w4_neural_oof_cols[m_id] = oof_df

        preds_a = pd.DataFrame(
            {
                "customer_id": oof_df["customer_id"],
                "article_id": oof_df["article_id"],
                "score": oof_df["sequence_logit"].astype(float),
            }
        )
        w4_neural_preds_track_a[m_id] = preds_a

        eval_a = evaluate_r1_predictions(
            preds_a,
            candidates=eval_art["cands"],
            labels=eval_art["window"]["labels"],
            history=eval_art["window"]["history"],
            strata_df=eval_art["strata"],
            k=12,
        )
        eval_a["encoder"] = enc_name
        eval_a["loss"] = loss_name
        eval_a["role"] = spec["role"]
        track_a_results[m_id] = eval_a

        preds_b = predict_track_b_full_catalog(
            trained,
            customer_ids=eval_art["target_users"],
            top_k=12,
        )
        eval_b = evaluate_track_b_predictions(
            preds_b,
            labels=eval_art["window"]["labels"],
            history=eval_art["window"]["history"],
            strata_df=eval_art["strata"],
            k=12,
        )
        eval_b["encoder"] = enc_name
        eval_b["loss"] = loss_name
        eval_b["role"] = spec["role"]
        track_b_results[m_id] = eval_b

        model_diagnostics[m_id] = {
            "encoder": enc_name,
            "loss": loss_name,
            "role": spec["role"],
            "device": trained["device"],
            "train_wall_time_sec": trained["train_wall_time_sec"],
            "final_train_loss": trained["train_loss_history"][-1] if trained["train_loss_history"] else 0.0,
            "train_loss_history": trained["train_loss_history"],
        }

        if loss_name == "sampled_softmax_256":
            shuffle_ablation_results[m_id] = evaluate_sequence_order_shuffle_ablation(
                trained_pack=trained,
                eval_art=eval_art,
                eval_a_true=eval_a,
                eval_b_true=eval_b,
                preds_a_true=preds_a,
                encoder_name=enc_name,
                loss_name=loss_name,
                seed=seed,
                n_bootstrap=min(int(n_bootstrap), 500),
            )

    # Select best core 2x4 sequential model on Track A (or M_HSTU_SS256 / M_LIGR_SS256)
    core_ids = [s["model_id"] for s in specs if s["role"] == "orthogonal_2x4"]
    best_neural_id = max(core_ids, key=lambda mid: (track_a_results[mid]["map12"], mid)) if core_ids else specs[0]["model_id"]
    best_spec = next(s for s in specs if s["model_id"] == best_neural_id)

    # 3. Track C Hybrid: Out-of-Time Neural features on W1..W3 injected into B1 LightGBM
    for tw in train_ws:
        tw_art = window_artifacts[tw]
        tw_trained = train_r2_sequential_model(
            events=safe_events,
            cutoff_date=tw_art["cutoff_date"],
            encoder_name=best_spec["encoder"],
            loss_name=best_spec["loss"],
            embedding_dim=embedding_dim,
            max_seq_len=max_seq_len,
            num_layers=num_layers,
            num_heads=num_heads,
            num_negatives=num_negatives,
            batch_size=batch_size,
            epochs=epochs,
            learning_rate=learning_rate,
            seed=seed,
            device=device,
        )
        tw_oof = compute_neural_oof_features(tw_trained, tw_art["cands"])
        tw_feat = tw_art["feat"].copy()
        tw_feat["sequence_logit"] = tw_oof["sequence_logit"].to_numpy(dtype=np.float32)
        tw_feat["embedding_similarity"] = tw_oof["embedding_similarity"].to_numpy(dtype=np.float32)
        tw_art["feat_hybrid"] = tw_feat

    val_feat_hybrid = eval_art["feat"].copy()
    best_w4_oof = w4_neural_oof_cols[best_neural_id]
    val_feat_hybrid["sequence_logit"] = best_w4_oof["sequence_logit"].to_numpy(dtype=np.float32)
    val_feat_hybrid["embedding_similarity"] = best_w4_oof["embedding_similarity"].to_numpy(dtype=np.float32)

    train_feat_b1 = pd.concat([window_artifacts[tw]["feat"] for tw in train_ws], ignore_index=True)
    train_feat_hyb = pd.concat([window_artifacts[tw]["feat_hybrid"] for tw in train_ws], ignore_index=True)
    train_labels = pd.concat([window_artifacts[tw]["window"]["labels"] for tw in train_ws], ignore_index=True)

    # Baseline B0, B1, B2 on W4
    preds_b0 = predict_b0_rule(eval_art["feat"])
    b1_model = R1LightGBMRanker(use_embeddings=False, n_estimators=n_estimators, random_state=seed).fit(
        train_feat_b1, train_labels
    )
    preds_b1 = b1_model.predict(eval_art["feat"])

    b2_model = R1LightGBMRanker(use_embeddings=True, n_estimators=n_estimators, random_state=seed).fit(
        train_feat_b1, train_labels
    )
    preds_b2 = b2_model.predict(eval_art["feat"])

    # Hybrid B1 + Neural OOF (sequence_logit + embedding_similarity)
    hybrid_b1_model = R2HybridLightGBMRanker(
        extra_neural_cols=HYBRID_NEURAL_FEATURE_COLS,
        use_svd_embeddings=False,
        n_estimators=n_estimators,
        random_state=seed,
    ).fit(train_feat_hyb, train_labels)
    preds_hybrid_b1 = hybrid_b1_model.predict(val_feat_hybrid)

    # Hybrid B2 + Neural OOF (B1 + SVD + Neural OOF)
    hybrid_b2_model = R2HybridLightGBMRanker(
        extra_neural_cols=HYBRID_NEURAL_FEATURE_COLS,
        use_svd_embeddings=True,
        n_estimators=n_estimators,
        random_state=seed,
    ).fit(train_feat_hyb, train_labels)
    preds_hybrid_b2 = hybrid_b2_model.predict(val_feat_hybrid)

    eval_b0 = evaluate_r1_predictions(
        preds_b0, eval_art["cands"], eval_art["window"]["labels"], eval_art["window"]["history"], eval_art["strata"]
    )
    eval_b1 = evaluate_r1_predictions(
        preds_b1, eval_art["cands"], eval_art["window"]["labels"], eval_art["window"]["history"], eval_art["strata"]
    )
    eval_b2 = evaluate_r1_predictions(
        preds_b2, eval_art["cands"], eval_art["window"]["labels"], eval_art["window"]["history"], eval_art["strata"]
    )
    eval_hybrid_b1 = evaluate_r1_predictions(
        preds_hybrid_b1,
        eval_art["cands"],
        eval_art["window"]["labels"],
        eval_art["window"]["history"],
        eval_art["strata"],
    )
    eval_hybrid_b2 = evaluate_r1_predictions(
        preds_hybrid_b2,
        eval_art["cands"],
        eval_art["window"]["labels"],
        eval_art["window"]["history"],
        eval_art["strata"],
    )

    val_labels = eval_art["window"]["labels"]
    preds_best_neural = w4_neural_preds_track_a[best_neural_id]

    ci_hybrid_b1_vs_b1 = compute_paired_bootstrap_ci(
        preds_b1, preds_hybrid_b1, val_labels, k=12, n_bootstrap=n_bootstrap, seed=seed
    )
    ci_hybrid_b1_vs_neural = compute_paired_bootstrap_ci(
        preds_best_neural, preds_hybrid_b1, val_labels, k=12, n_bootstrap=n_bootstrap, seed=seed
    )
    ci_hybrid_b2_vs_b2 = compute_paired_bootstrap_ci(
        preds_b2, preds_hybrid_b2, val_labels, k=12, n_bootstrap=n_bootstrap, seed=seed
    )
    ci_hybrid_b2_vs_b1 = compute_paired_bootstrap_ci(
        preds_b1, preds_hybrid_b2, val_labels, k=12, n_bootstrap=n_bootstrap, seed=seed
    )

    # Orthogonal attribution summaries (Loss effect vs Architecture effect on Track A and Track B)
    encoders_order = ["gru", "sasrec", "ligr_sasrec", "hstu_math"]
    bce256_ids = ["M_GRU_BCE256", "M_SAS_BCE256", "M_LIGR_BCE256", "M_HSTU_BCE256"]
    ss256_ids = ["M_GRU_SS256", "M_SAS_SS256", "M_LIGR_SS256", "M_HSTU_SS256"]

    orthogonal_attribution: Dict[str, Any] = {}
    if all(mid in track_a_results for mid in bce256_ids + ss256_ids + ["M_SAS_BCE1"]):
        orthogonal_attribution = {
            "negative_sampling_effect_sasrec_bce256_minus_bce1_track_a": float(
                track_a_results["M_SAS_BCE256"]["map12"] - track_a_results["M_SAS_BCE1"]["map12"]
            ),
            "negative_sampling_effect_sasrec_bce256_minus_bce1_track_b": float(
                track_b_results["M_SAS_BCE256"]["map12"] - track_b_results["M_SAS_BCE1"]["map12"]
            ),
            "loss_function_effect_ss256_minus_bce256_by_encoder_track_a": {
                enc: float(track_a_results[ss_id]["map12"] - track_a_results[bce_id]["map12"])
                for enc, bce_id, ss_id in zip(encoders_order, bce256_ids, ss256_ids)
            },
            "loss_function_effect_ss256_minus_bce256_mean_track_a": float(
                np.mean([track_a_results[ss_id]["map12"] - track_a_results[bce_id]["map12"] for bce_id, ss_id in zip(bce256_ids, ss256_ids)])
            ),
            "loss_function_effect_ss256_minus_bce256_mean_track_b": float(
                np.mean([track_b_results[ss_id]["map12"] - track_b_results[bce_id]["map12"] for bce_id, ss_id in zip(bce256_ids, ss256_ids)])
            ),
            "architecture_effect_under_ss256_track_a": {
                enc: float(track_a_results[ss_id]["map12"])
                for enc, ss_id in zip(encoders_order, ss256_ids)
            },
            "architecture_effect_under_ss256_track_b": {
                enc: float(track_b_results[ss_id]["map12"])
                for enc, ss_id in zip(encoders_order, ss256_ids)
            },
            "paired_bootstrap_ci_track_a": {
                "SAS_BCE256_vs_SAS_BCE1": compute_paired_bootstrap_ci(
                    w4_neural_preds_track_a["M_SAS_BCE1"],
                    w4_neural_preds_track_a["M_SAS_BCE256"],
                    val_labels,
                    k=12,
                    n_bootstrap=n_bootstrap,
                    seed=seed,
                ),
                "SAS_SS256_vs_SAS_BCE256": compute_paired_bootstrap_ci(
                    w4_neural_preds_track_a["M_SAS_BCE256"],
                    w4_neural_preds_track_a["M_SAS_SS256"],
                    val_labels,
                    k=12,
                    n_bootstrap=n_bootstrap,
                    seed=seed,
                ),
                "LIGR_SS256_vs_SAS_SS256": compute_paired_bootstrap_ci(
                    w4_neural_preds_track_a["M_SAS_SS256"],
                    w4_neural_preds_track_a["M_LIGR_SS256"],
                    val_labels,
                    k=12,
                    n_bootstrap=n_bootstrap,
                    seed=seed,
                ),
                "HSTU_SS256_vs_SAS_SS256": compute_paired_bootstrap_ci(
                    w4_neural_preds_track_a["M_SAS_SS256"],
                    w4_neural_preds_track_a["M_HSTU_SS256"],
                    val_labels,
                    k=12,
                    n_bootstrap=n_bootstrap,
                    seed=seed,
                ),
            },
        }

    return {
        "contract_version": "0.9",
        "generator_version": "r1_cand_v0.9",
        "validation_cutoffs": val_cutoffs,
        "primary_eval_window": eval_w,
        "primary_eval_cutoff": eval_cutoff,
        "sealed_cutoff": sealed_cutoff,
        "sealed_w5_labels_touched": False,
        "candidate_pool_size_k": int(k_pool),
        "hyperparameters": {
            "embedding_dim": int(embedding_dim),
            "max_seq_len": int(max_seq_len),
            "num_layers": int(num_layers),
            "num_heads": int(num_heads),
            "num_negatives": int(num_negatives),
            "batch_size": int(batch_size),
            "epochs": int(epochs),
            "learning_rate": float(learning_rate),
            "seed": int(seed),
        },
        "r1_reference_on_w4": {
            "B0_rule": eval_b0,
            "B1_strong_lgbm": eval_b1,
            "B2_lgbm_svd_emb": eval_b2,
        },
        "track_a_candidate_rerank": track_a_results,
        "track_b_full_catalog": track_b_results,
        "track_c_hybrid": {
            "selected_neural_model_id": best_neural_id,
            "injected_oof_features": list(HYBRID_NEURAL_FEATURE_COLS),
            "hybrid_b1_plus_neural": eval_hybrid_b1,
            "hybrid_b2_plus_neural": eval_hybrid_b2,
            "paired_bootstrap_ci": {
                "Hybrid_B1_vs_B1_Alone": ci_hybrid_b1_vs_b1,
                "Hybrid_B1_vs_Neural_Alone": ci_hybrid_b1_vs_neural,
                "Hybrid_B2_vs_B2_Alone": ci_hybrid_b2_vs_b2,
                "Hybrid_B2_vs_B1_Alone": ci_hybrid_b2_vs_b1,
            },
            "hybrid_b1_top15_features": hybrid_b1_model.get_top_features(15),
            "hybrid_b2_top15_features": hybrid_b2_model.get_top_features(15),
        },
        "orthogonal_attribution": orthogonal_attribution,
        "sequence_order_shuffle_ablation": shuffle_ablation_results,
        "model_training_diagnostics": model_diagnostics,
    }


# Alias for external benchmark scripts and CLI callers
run_r2_benchmark = run_r2_orthogonal_benchmark
