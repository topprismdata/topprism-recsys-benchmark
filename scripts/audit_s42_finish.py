"""Finish the s42 audit without retraining: load saved checkpoints, run (1) trained-weight
grad-alive checks for A2, (2) attention weight content-sensitivity for A0/A2 (GPT P1),
(3) per-arm personalization stats, (4) loss-curve extract for the collapse table."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.models.r2_encoders import build_r2_encoder, _causal_attention_mask

L, D, LAYERS, HEADS = 200, 128, 2, 4
N_ITEMS = 3416
AUD = REPO / "artifacts/eval/ml1m_ablation/audit_s42"


def load(name):
    m = build_r2_encoder(name, num_items=N_ITEMS, embedding_dim=D, max_seq_len=L,
                         num_layers=LAYERS, num_heads=HEADS, dropout=0.1)
    sd = torch.load(AUD / f"ckpt_{name}.pt", map_location="cpu", weights_only=True)
    m.load_state_dict(sd, strict=True)
    m.eval()
    return m


def main():
    out = {}
    # --- A2 grad-alive on trained checkpoint ---
    m2 = load("sasrec_pos_only")
    b = torch.zeros(8, L, dtype=torch.long)
    for i in range(8):
        b[i, : 5 + i] = torch.randint(1, N_ITEMS + 1, (5 + i,))
    ts = torch.arange(L, dtype=torch.float32).unsqueeze(0).expand(8, -1)
    m2.train()
    m2(b, timestamps=ts).pow(2).sum().backward()
    blk = m2.blocks[0]
    out["A2_trained"] = {
        "q_grad": float(blk.q_proj.weight.grad.abs().sum()) if blk.q_proj.weight.grad is not None else None,
        "k_grad": float(blk.k_proj.weight.grad.abs().sum()) if blk.k_proj.weight.grad is not None else None,
        "pos_emb_grad": float(m2.pos_embedding.weight.grad.abs().sum()) if m2.pos_embedding.weight.grad is not None else None,
    }
    out["A2_trained"]["qk_pos_grads_all_alive"] = all(v is not None and v > 0 for v in out["A2_trained"].values())

    # --- attention weight content-sensitivity on trained checkpoints ---
    sens = {}
    for name in ("sasrec_full", "sasrec_pos_only"):
        m = load(name)
        ids = torch.zeros(4, L, dtype=torch.long)
        for i, n in enumerate([5, 9, 12, 20]):
            g = torch.Generator().manual_seed(i)
            ids[i, :n] = torch.randint(1, N_ITEMS + 1, (n,), generator=g)
        ids2 = ids.clone()
        gg = torch.Generator().manual_seed(99)
        ids2[:, :4] = torch.randint(1, N_ITEMS + 1, (4,), generator=gg)
        blk = m.blocks[0]

        def probs(input_ids):
            with torch.no_grad():
                pe = m.pos_embedding.weight[: L].unsqueeze(0)
                x = m.emb_norm(m.item_embedding(input_ids) + pe)
                if name == "sasrec_pos_only":
                    q = blk.q_proj(pe).view(1, L, HEADS, D // HEADS).transpose(1, 2).expand(input_ids.size(0), -1, -1, -1)
                    k = blk.k_proj(pe).view(1, L, HEADS, D // HEADS).transpose(1, 2).expand(input_ids.size(0), -1, -1, -1)
                else:
                    q = blk.q_proj(x).view(input_ids.size(0), L, HEADS, D // HEADS).transpose(1, 2)
                    k = blk.k_proj(x).view(input_ids.size(0), L, HEADS, D // HEADS).transpose(1, 2)
                sc = torch.matmul(q, k.transpose(-2, -1)) * blk.scale
                am = _causal_attention_mask(input_ids)
                sc = torch.where(am, sc, torch.full_like(sc, -1e9))
                p = torch.softmax(sc, dim=-1)
                return torch.where(am, p, torch.zeros_like(p))

        p1, p2 = probs(ids), probs(ids2)
        sens[name] = {"max_abs_prob_diff": float((p1 - p2).abs().max()),
                      "weights_vary_with_content": bool(float((p1 - p2).abs().max()) > 1e-6)}
    out["weight_content_sensitivity"] = sens

    # --- per-arm personalization from saved topk ---
    per = {}
    for name in ("sasrec_full", "sasrec_uniform", "sasrec_pos_only", "mean_pool"):
        topk = np.load(AUD / f"scores_{name}.npy")
        rng = np.random.default_rng(0)
        pairs = rng.integers(0, len(topk), size=(2000, 2))
        ov = np.array([len(set(topk[a]) & set(topk[b])) for a, b in pairs])
        per[name] = {"mean_pairwise_top10_overlap": float(ov.mean()),
                     "frac_pairs_overlap_ge8": float((ov >= 8).mean())}
    out["personalization"] = per

    (AUD / "diag_final.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
