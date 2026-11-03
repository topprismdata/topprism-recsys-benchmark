"""Alignment check: HSTUMathAlignedEncoder vs a dense transcription of the official
meta-recsys/generative-recommenders STULayer math (ops/hstu_compute.py, ops/pytorch/
pt_hstu_attention.py, ops/pytorch/pt_hstu_linear.py, ops/pytorch/pt_position.py).

The reference below is written independently from the official source files so that a
pass means "our encoder implements the official math", not "our code matches itself".

Run:  python scripts/hstu_alignment_check.py
"""

import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from models.r2_encoders import HSTUMathAlignedEncoder  # noqa: E402


def reference_hstu_encoder(
    input_ids: torch.Tensor,
    timestamps: torch.Tensor,
    *,
    item_w: torch.Tensor,
    pos_w: torch.Tensor,
    ts_w: torch.Tensor,
    layers: list[dict],
    final_w: torch.Tensor,
    final_b: torch.Tensor,
    num_heads: int,
    num_time_buckets: int,
) -> torch.Tensor:
    """Dense, dropout-off transcription of the official reference path."""
    valid = input_ids > 0  # (B, N)
    vf = valid.float()
    bsz, seq_len = input_ids.shape

    # input embeddings: item + positional(distance-from-end) + time(log-minutes bucket)
    x = F.embedding(input_ids, item_w)
    lengths = valid.sum(dim=1)
    pos_idx = torch.clamp(
        lengths.unsqueeze(1) - torch.arange(seq_len, device=input_ids.device).unsqueeze(0),
        min=0,
        max=pos_w.shape[0] - 1,
    )
    safe_pos = torch.clamp(lengths - 1, min=0)
    query_time = timestamps.gather(1, safe_pos.unsqueeze(1))
    delta_min = (query_time - timestamps) * 1440.0
    delta_min = torch.where(valid, torch.clamp(delta_min, min=1.6667e-8), torch.zeros_like(delta_min))
    ts_idx = torch.clamp(torch.log(delta_min).nan_to_num(0.0), min=0).to(torch.long)
    ts_idx = torch.clamp(ts_idx, max=num_time_buckets)
    x = x + F.embedding(pos_idx, pos_w) * vf.unsqueeze(-1) + F.embedding(ts_idx, ts_w) * vf.unsqueeze(-1)
    x = x * vf.unsqueeze(-1)

    # official valid mask: eye | (row_dist > 0), restricted to seq_lengths
    ids = torch.arange(seq_len, device=input_ids.device).view(1, seq_len)
    max_ids = lengths.view(-1, 1, 1)
    row_ids = ids.view(-1, seq_len, 1).expand(-1, seq_len, seq_len)
    col_ids = ids.view(-1, 1, seq_len).expand(-1, seq_len, seq_len)
    dist = row_ids - col_ids
    mask = torch.eye(seq_len, dtype=torch.bool).unsqueeze(0) | (dist > 0)
    mask = mask & (col_ids < max_ids)  # keys within each row's length (padding_value=0 in jagged->padded)
    mask = mask.unsqueeze(1)  # (B, 1, N, N)

    for p in layers:
        h = F.layer_norm(x, p["ln1_w"].shape, p["ln1_w"], p["ln1_b"], eps=1e-6)
        uvqk = F.linear(h, p["uvqk_w"], p["uvqk_b"])
        ph = p["per_head"]
        sizes = [ph * num_heads, ph * num_heads, ph * num_heads, ph * num_heads]
        u, v, q, k = torch.split(uvqk, sizes, dim=-1)
        u = F.silu(u)
        q = q.view(bsz, seq_len, num_heads, ph).transpose(1, 2)
        k = k.view(bsz, seq_len, num_heads, ph).transpose(1, 2)
        v = v.view(bsz, seq_len, num_heads, ph).transpose(1, 2)
        alpha = 1.0 / math.sqrt(ph)
        qk = torch.einsum("bhxa,bhya->bhxy", q, k) * alpha
        qk = F.silu(qk) / float(seq_len)
        qk = qk * mask.to(qk.dtype)
        av = torch.einsum("bhxy,bhyv->bhxv", qk, v)
        av = av.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        y = F.layer_norm(av, p["ln2_w"].shape, p["ln2_w"], p["ln2_b"], eps=1e-6) * u
        y_cat = torch.cat([u, x, y], dim=-1).view(-1, 3 * x.shape[-1])
        x = (torch.addmm(x.view(-1, x.shape[-1]), y_cat, p["out_w"].t()) + p["out_b"]).view(bsz, seq_len, -1)
        x = x * vf.unsqueeze(-1)

    return F.layer_norm(x, final_w.shape, final_w, final_b, eps=1e-6) * vf.unsqueeze(-1)


def main() -> None:
    torch.manual_seed(7)
    bsz, seq_len, num_items, num_heads, num_time_buckets = 3, 50, 97, 4, 2048
    lengths = torch.tensor([7, 33, 50])
    ids = torch.zeros(bsz, seq_len, dtype=torch.long)
    ts = torch.zeros(bsz, seq_len)
    for b, ln in enumerate(lengths.tolist()):
        ids[b, :ln] = torch.randint(1, num_items, (ln,))
        ts[b, :ln] = torch.sort(torch.rand(ln) * 120.0).values  # days

    enc = HSTUMathAlignedEncoder(
        num_items=num_items,
        embedding_dim=128,
        max_seq_len=seq_len,
        num_layers=2,
        num_heads=num_heads,
        dropout=0.1,
        num_time_buckets=num_time_buckets,
    )
    enc.eval()

    layers = []
    for blk in enc.blocks:
        layers.append(
            {
                "per_head": blk.per_head,
                "ln1_w": blk.pre_norm.weight.detach().clone(),
                "ln1_b": blk.pre_norm.bias.detach().clone(),
                "uvqk_w": blk.uvqk.weight.detach().clone(),
                "uvqk_b": blk.uvqk.bias.detach().clone(),
                "ln2_w": blk.av_norm.weight.detach().clone(),
                "ln2_b": blk.av_norm.bias.detach().clone(),
                "out_w": blk.out_proj.weight.detach().clone(),
                "out_b": blk.out_proj.bias.detach().clone(),
            }
        )

    with torch.no_grad():
        ours = enc(ids, timestamps=ts)
        ref = reference_hstu_encoder(
            ids,
            ts,
            item_w=enc.item_embedding.weight.detach(),
            pos_w=enc.pos_embedding.weight.detach(),
            ts_w=enc.time_embedding.weight.detach(),
            layers=layers,
            final_w=enc.final_norm.weight.detach(),
            final_b=enc.final_norm.bias.detach(),
            num_heads=num_heads,
            num_time_buckets=num_time_buckets,
        )

    diff = (ours - ref).abs()
    denom = ref.abs().clamp(min=1e-6)
    print(f"max_abs_diff = {diff.max().item():.3e}")
    print(f"mean_abs_diff = {diff.mean().item():.3e}")
    print(f"max_rel_diff = {(diff / denom).max().item():.3e}")
    ok = diff.max().item() < 1e-4
    print("PARITY:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
