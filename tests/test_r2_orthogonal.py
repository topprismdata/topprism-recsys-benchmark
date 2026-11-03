import numpy as np
import pandas as pd
import pytest
import torch

from src.models.r2_losses import (
    BCE1Loss,
    BCE256Loss,
    SampledSoftmax256Loss,
    sample_negatives_excluding_positives,
)
from src.models.r2_encoders import (
    GRUEncoder,
    HSTUMathEncoder,
    LiGRSASRecEncoder,
    MeanPoolEncoder,
    SASRecEncoder,
    build_r2_encoder,
)
from src.models.r2_hybrid_runner import (
    R2_MATRIX_SPECS,
    compute_neural_oof_features,
    predict_track_a_candidates,
    predict_track_b_full_catalog,
    run_r2_orthogonal_benchmark,
    train_r2_sequential_model,
)


@pytest.mark.parametrize(
    "encoder_cls",
    [
        MeanPoolEncoder,
        GRUEncoder,
        SASRecEncoder,
        LiGRSASRecEncoder,
        HSTUMathEncoder,
    ],
)
def test_r2_causal_mask_zero_future_leakage_all_encoders(encoder_cls):
    torch.manual_seed(42)
    num_items = 60
    batch_size = 4
    seq_len = 12
    cutoff_pos = 6  # test positions 0..6 are invariant to changes at 7..11

    model = encoder_cls(
        num_items=num_items,
        embedding_dim=32,
        max_seq_len=seq_len,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
    ).eval()

    input_ids_a = torch.randint(1, num_items + 1, (batch_size, seq_len), dtype=torch.long)
    # Include a padded prefix in batch row 0 to verify padding + causal mask safety
    input_ids_a[0, :2] = 0
    timestamps_a = torch.cumsum(torch.randint(1, 5, (batch_size, seq_len)), dim=1).to(torch.float32)

    input_ids_b = input_ids_a.clone()
    timestamps_b = timestamps_a.clone()
    # Perturb future tokens and timestamps strictly after cutoff_pos (positions j > cutoff_pos)
    input_ids_b[:, cutoff_pos + 1 :] = torch.randint(
        1, num_items + 1, (batch_size, seq_len - cutoff_pos - 1), dtype=torch.long
    )
    timestamps_b[:, cutoff_pos + 1 :] = timestamps_a[:, cutoff_pos + 1 :] + 500.0

    out_a = model(input_ids_a, timestamps=timestamps_a)
    out_b = model(input_ids_b, timestamps=timestamps_b)

    assert out_a.shape == (batch_size, seq_len, 32)
    assert not torch.isnan(out_a).any(), f"{encoder_cls.__name__} produced NaNs"

    # Forward representations up to cutoff_pos must be identical
    torch.testing.assert_close(
        out_a[:, : cutoff_pos + 1, :],
        out_b[:, : cutoff_pos + 1, :],
        rtol=0.0,
        atol=1e-7,
    )

    # Backward gradient invariance: loss depending only on positions <= cutoff_pos
    model.zero_grad(set_to_none=True)
    loss_a = (out_a[:, : cutoff_pos + 1, :] ** 2).sum()
    loss_a.backward()
    grad_a = model.item_embedding.weight.grad.clone()

    model.zero_grad(set_to_none=True)
    loss_b = (out_b[:, : cutoff_pos + 1, :] ** 2).sum()
    loss_b.backward()
    grad_b = model.item_embedding.weight.grad.clone()

    torch.testing.assert_close(grad_a, grad_b, rtol=0.0, atol=1e-7)


def test_r2_losses_bce1_bce256_sampled_softmax256_and_logq_gradients():
    torch.manual_seed(123)
    num_items = 400
    batch_tokens = 32
    dim = 16
    num_neg = 256

    pos_ids = torch.randint(1, num_items + 1, (batch_tokens,), dtype=torch.long)
    neg_ids = sample_negatives_excluding_positives(
        pos_ids=pos_ids,
        num_items=num_items,
        num_negatives=num_neg,
    )
    assert neg_ids.shape == (batch_tokens, num_neg)
    assert int(neg_ids.min().item()) >= 1
    assert int(neg_ids.max().item()) <= num_items
    # Strictly zero collisions with positive item IDs
    assert int((neg_ids == pos_ids.unsqueeze(-1)).sum().item()) == 0

    item_probs = torch.arange(1, num_items + 2, dtype=torch.float32)
    item_probs[0] = 1e-8
    item_probs = item_probs / item_probs[1:].sum()
    item_log_probs = torch.log(torch.clamp(item_probs, min=1e-12))

    losses = [
        ("bce_1", BCE1Loss()),
        ("bce_256", BCE256Loss(num_negatives=num_neg)),
        ("ss_256_no_logq", SampledSoftmax256Loss(num_negatives=num_neg, logq_correction=False)),
        ("ss_256_logq", SampledSoftmax256Loss(num_negatives=num_neg, logq_correction=True)),
    ]

    val_map = {}
    for name, criterion in losses:
        user_repr = torch.randn(batch_tokens, dim, requires_grad=True)
        item_emb = torch.randn(num_items + 1, dim, requires_grad=True)
        if isinstance(criterion, SampledSoftmax256Loss):
            loss = criterion(user_repr, item_emb, pos_ids, neg_ids, item_log_probs=item_log_probs)
        else:
            loss = criterion(user_repr, item_emb, pos_ids, neg_ids)

        assert torch.isfinite(loss).item(), f"{name} produced non-finite loss"
        assert float(loss.item()) > 0.0
        loss.backward()
        assert user_repr.grad is not None and torch.isfinite(user_repr.grad).all()
        assert float(user_repr.grad.abs().sum().item()) > 0.0
        assert item_emb.grad is not None and torch.isfinite(item_emb.grad).all()
        assert float(item_emb.grad[1:].abs().sum().item()) > 0.0
        val_map[name] = float(loss.item())

    # LogQ correction must change loss when item distribution is non-uniform
    assert abs(val_map["ss_256_logq"] - val_map["ss_256_no_logq"]) > 1e-4


def _synthetic_r2_events() -> pd.DataFrame:
    rows = []
    # Create 6 customers across 8 items and 5 weekly windows:
    # W1=2020-08-04, W2=2020-08-11, W3=2020-08-18, W4=2020-08-25, W5(sealed)=2020-09-01
    base_patterns = {
        "u1": ["0000000001", "0000000002", "0000000003"],
        "u2": ["0000000002", "0000000004", "0000000005"],
        "u3": ["0000000001", "0000000003", "0000000006"],
        "u4": ["0000000004", "0000000007", "0000000008"],
        "u5": ["0000000005", "0000000006"],
    }
    dates = [
        "2020-07-10",
        "2020-07-18",
        "2020-07-25",
        "2020-08-01",
        "2020-08-03",
        "2020-08-07",
        "2020-08-10",
        "2020-08-14",
        "2020-08-17",
        "2020-08-21",
        "2020-08-24",
        "2020-08-28",
        "2020-08-30",
    ]
    for u, items in base_patterns.items():
        for d_idx, d in enumerate(dates):
            art = items[d_idx % len(items)]
            rows.append({"customer_id": u, "article_id": art, "event_date": d, "value": 10.0 + d_idx})
            next_art = items[(d_idx + 1) % len(items)]
            rows.append({"customer_id": u, "article_id": next_art, "event_date": d, "value": 12.0})

    # u6: S0 customer (only 1 historical basket before W4=2020-08-25, then buys in W4)
    rows.append({"customer_id": "u6", "article_id": "0000000001", "event_date": "2020-08-20", "value": 15.0})
    rows.append({"customer_id": "u6", "article_id": "0000000001", "event_date": "2020-08-27", "value": 15.0})
    return pd.DataFrame(rows)


def test_r2_track_a_track_b_and_track_c_hybrid_end_to_end():
    events = _synthetic_r2_events()
    val_cutoffs = ["2020-08-04", "2020-08-11", "2020-08-18", "2020-08-25"]
    sealed_cutoff = "2020-09-01"

    # Verify the 2x4 + 1 matrix spec list has all 9 mandatory models + MeanPool diagnostic
    mandatory_names = {s["model_id"] for s in R2_MATRIX_SPECS if s["role"] != "diagnostic"}
    assert mandatory_names == {
        "M_SAS_BCE1",
        "M_GRU_BCE256",
        "M_GRU_SS256",
        "M_SAS_BCE256",
        "M_SAS_SS256",
        "M_LIGR_BCE256",
        "M_LIGR_SS256",
        "M_HSTU_BCE256",
        "M_HSTU_SS256",
    }

    res_clean = run_r2_orthogonal_benchmark(
        events=events,
        validation_cutoffs=val_cutoffs,
        sealed_cutoff=sealed_cutoff,
        horizon_days=7,
        k_pool=6,
        embedding_dim=16,
        max_seq_len=12,
        num_layers=1,
        num_heads=2,
        num_negatives=4,
        batch_size=8,
        epochs=2,
        n_estimators=15,
        n_bootstrap=50,
        seed=42,
        device="cpu",
    )

    assert res_clean["sealed_w5_labels_touched"] is False
    assert set(mandatory_names).issubset(set(res_clean["track_a_candidate_rerank"].keys()))
    assert set(mandatory_names).issubset(set(res_clean["track_b_full_catalog"].keys()))
    assert "track_c_hybrid" in res_clean

    # Check Oracle upper bound and additive Repeat/Explore decomposition on Track A & Track C
    for m_id, m_rep in res_clean["track_a_candidate_rerank"].items():
        assert m_rep["map12"] <= m_rep["oracle_map12"] + 1e-9, f"{m_id} violated Oracle bound"
        assert (
            m_rep["repeat_explore"]["map12_repeat"] + m_rep["repeat_explore"]["map12_explore"]
            == pytest.approx(m_rep["map12"])
        )
        assert set(m_rep["strata"].keys()) == {"S0", "S1", "S2"}

    for m_id, m_rep in res_clean["track_b_full_catalog"].items():
        assert (
            m_rep["repeat_explore"]["map12_repeat"] + m_rep["repeat_explore"]["map12_explore"]
            == pytest.approx(m_rep["map12"])
        )
        assert set(m_rep["strata"].keys()) == {"S0", "S1", "S2"}

    hybrid_eval = res_clean["track_c_hybrid"]["hybrid_b1_plus_neural"]
    assert hybrid_eval["map12"] <= hybrid_eval["oracle_map12"] + 1e-9
    assert (
        hybrid_eval["repeat_explore"]["map12_repeat"] + hybrid_eval["repeat_explore"]["map12_explore"]
        == pytest.approx(hybrid_eval["map12"])
    )

    # Verify W5 post-sealed_cutoff noise leaves R2 results bit-for-bit identical
    w5_noise = pd.DataFrame(
        [
            {"customer_id": "u1", "article_id": "0000000099", "event_date": "2020-09-05", "value": 999.0},
            {"customer_id": "u2", "article_id": "0000000008", "event_date": "2020-09-06", "value": 888.0},
        ]
    )
    res_polluted = run_r2_orthogonal_benchmark(
        events=pd.concat([events, w5_noise], ignore_index=True),
        validation_cutoffs=val_cutoffs,
        sealed_cutoff=sealed_cutoff,
        horizon_days=7,
        k_pool=6,
        embedding_dim=16,
        max_seq_len=12,
        num_layers=1,
        num_heads=2,
        num_negatives=4,
        batch_size=8,
        epochs=2,
        n_estimators=15,
        n_bootstrap=50,
        seed=42,
        device="cpu",
    )
    assert res_clean["track_a_candidate_rerank"] == res_polluted["track_a_candidate_rerank"]
    assert res_clean["track_c_hybrid"]["hybrid_b1_plus_neural"] == res_polluted["track_c_hybrid"]["hybrid_b1_plus_neural"]


def test_r3_compute_benchmark_smoke():
    from src.models.r3_compute_bench import R3_ENCODERS, R3_EXECUTION_MODES, run_r3_compute_benchmark

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rep = run_r3_compute_benchmark(
        encoders=R3_ENCODERS,
        execution_modes=R3_EXECUTION_MODES,
        seq_lengths=(32, 64),
        capacity_tiers=("S", "M"),
        num_items=200,
        base_embedding_dim=32,
        base_seq_len=32,
        base_num_layers=1,
        base_num_heads=4,
        batch_size=16,
        infer_batch_size=16,
        num_negatives=16,
        warmup_steps=1,
        train_steps=2,
        infer_steps=4,
        seed=42,
        device=dev,
    )

    assert rep["contract_version"] == "0.9"
    assert "hardware_metadata" in rep
    assert len(rep["primary_matrix_5x4"]) == len(R3_ENCODERS) * len(R3_EXECUTION_MODES)
    assert len(rep["sequence_length_scaling"]) > 0
    assert len(rep["capacity_scaling"]) > 0
    assert len(rep["pareto_frontier"]) >= 1

    for row in rep["primary_matrix_5x4"]:
        assert row["train_sequences_per_sec"] > 0.0
        assert row["infer_users_per_sec"] > 0.0
        assert row["infer_p50_latency_ms"] > 0.0
        assert row["infer_p95_latency_ms"] >= row["infer_p50_latency_ms"]
        assert row["peak_vram_mb"] > 0.0
        assert 0.0 <= row["top12_agreement_vs_eager_fp32"] <= 1.0
        assert 0.0 <= row["top12_jaccard_vs_eager"] <= 1.0
        if row["execution_mode"] == "eager_fp32":
            assert row["top12_agreement_vs_eager_fp32"] == pytest.approx(1.0)
        else:
            # Mixed-precision / SDPA should maintain strong ranking agreement with FP32
            assert row["top12_agreement_vs_eager_fp32"] >= 0.70


def test_build_user_sequences_order_modes(tmp_path):
    """sequence_order: shuffled keeps item multiset but breaks order; reversed flips it; default unchanged."""
    from src.models.r2_hybrid_runner import _build_user_sequences

    hist = pd.DataFrame({
        "customer_id": ["u1"] * 5 + ["u2"] * 3,
        "article_id": [f"a{i}" for i in range(5)] + [f"b{i}" for i in range(3)],
        "event_date": ["2020-01-01", "2020-01-02", "2020-01-03", "2020-01-04", "2020-01-05",
                        "2020-02-01", "2020-02-02", "2020-02-03"],
        "days_ago": [4.0, 3.0, 2.0, 1.0, 0.0, 2.0, 1.0, 0.0],
    })
    art_to_id = {"a0": 1, "a1": 2, "a2": 3, "a3": 4, "a4": 5, "b0": 6, "b1": 7, "b2": 8}

    chrono = _build_user_sequences(hist, art_to_id, max_seq_len=4)
    assert chrono["input_ids"][0, 0].item() == 1  # oldest first (a0)
    assert chrono["input_ids"][0, 3].item() == 4  # train input excludes last event (a4 is target)

    rev = _build_user_sequences(hist, art_to_id, max_seq_len=4, sequence_order="reversed")
    r0 = rev["input_ids"][0][rev["input_ids"][0] > 0].tolist()
    # Full 5-event chronology has ids [1..5]; reversed (newest-first) train input window = [5,4,3,2]
    assert r0 == [5, 4, 3, 2]
    c0 = chrono["input_ids"][0][chrono["input_ids"][0] > 0].tolist()

    shuf = _build_user_sequences(hist, art_to_id, max_seq_len=4, sequence_order="shuffled", shuffle_seed=42)
    s0 = shuf["input_ids"][0][shuf["input_ids"][0] > 0].tolist()
    assert s0 != c0  # order broken
    assert set(s0) <= {1, 2, 3, 4, 5}  # same item universe
    shuf2 = _build_user_sequences(hist, art_to_id, max_seq_len=4, sequence_order="shuffled", shuffle_seed=42)
    s0b = shuf2["input_ids"][0][shuf2["input_ids"][0] > 0].tolist()
    assert s0b == s0  # deterministic given seed
    with pytest.raises(ValueError):
        _build_user_sequences(hist, art_to_id, max_seq_len=4, sequence_order="bogus")
