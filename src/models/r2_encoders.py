import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def _causal_attention_mask(input_ids: torch.Tensor) -> torch.Tensor:
    """Return boolean causal mask of shape (B, 1, N, N) where j <= i and (token_j != 0 or j == i)."""
    bsz, seq_len = input_ids.shape
    device = input_ids.device
    tril = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))
    valid_k = (input_ids > 0).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, N)
    diag = torch.eye(seq_len, dtype=torch.bool, device=device).unsqueeze(0).unsqueeze(0)
    return (tril.unsqueeze(0).unsqueeze(0) & valid_k) | diag


class BaseSequentialEncoder(nn.Module):
    """Unified base class for R2 sequential encoders sharing item embedding table and output contract."""

    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 128,
        max_seq_len: int = 50,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_items = int(num_items)
        self.embedding_dim = int(embedding_dim)
        self.max_seq_len = int(max_seq_len)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.dropout_rate = float(dropout)

        self.item_embedding = nn.Embedding(
            self.num_items + 1,
            self.embedding_dim,
            padding_idx=0,
        )
        nn.init.normal_(self.item_embedding.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.item_embedding.weight[0].zero_()

    def forward(
        self,
        input_ids: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        raise NotImplementedError

    def encode_last_step(
        self,
        input_ids: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Extract sequence representation (B, d) at the last non-padding position per user."""
        seq_out = self.forward(input_ids, timestamps=timestamps)
        valid_mask = input_ids > 0
        positions = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        last_idx = torch.where(valid_mask, positions, torch.full_like(positions, -1)).max(dim=1).values
        has_valid = last_idx >= 0
        safe_idx = torch.clamp(last_idx, min=0)
        batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)
        picked = seq_out[batch_indices, safe_idx]
        return picked * has_valid.unsqueeze(-1).to(picked.dtype)

    def set_sdpa_enabled(self, enabled: bool = True) -> None:
        """Enable or disable fused PyTorch SDPA kernel on supported attention blocks."""
        for module in self.modules():
            if hasattr(module, "use_sdpa"):
                setattr(module, "use_sdpa", bool(enabled))


class MeanPoolEncoder(BaseSequentialEncoder):
    """Diagnostic 0: Causal prefix mean-pooling encoder (no positional encoding, no attention)."""

    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 128,
        max_seq_len: int = 50,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            num_items=num_items,
            embedding_dim=embedding_dim,
            max_seq_len=max_seq_len,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(embedding_dim, embedding_dim)
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        valid = (input_ids > 0).to(self.item_embedding.weight.dtype).unsqueeze(-1)
        x = self.drop(self.item_embedding(input_ids)) * valid
        cum_sum = torch.cumsum(x, dim=1)
        cum_cnt = torch.clamp(torch.cumsum(valid, dim=1), min=1.0)
        pooled = cum_sum / cum_cnt
        out = self.norm(pooled + self.proj(pooled))
        return out * valid


class GRUEncoder(BaseSequentialEncoder):
    """Architecture 1: Unidirectional multi-layer GRU sequential encoder."""

    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 128,
        max_seq_len: int = 50,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            num_items=num_items,
            embedding_dim=embedding_dim,
            max_seq_len=max_seq_len,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.emb_drop = nn.Dropout(dropout)
        self.gru = nn.GRU(
            input_size=embedding_dim,
            hidden_size=embedding_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False,
        )
        self.norm = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        valid = (input_ids > 0).to(self.item_embedding.weight.dtype).unsqueeze(-1)
        x = self.emb_drop(self.item_embedding(input_ids)) * valid
        gru_out, _ = self.gru(x)
        out = self.norm(gru_out + x)
        return out * valid


class _SASRecBlock(nn.Module):
    """Standard Post-LN Causal Self-Attention + FFN block (SASRec)."""

    def __init__(self, embedding_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})")
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.use_sdpa = False
        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 4, embedding_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, dim = x.shape
        q = self.q_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_sdpa:
            drop_p = self.attn_drop.p if self.training else 0.0
            attn_out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=drop_p,
                is_causal=False,
            )
            context = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, dim)
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            neg_inf_val = -1e4 if scores.dtype == torch.float16 else -1e9
            neg_inf = torch.full_like(scores, neg_inf_val)
            scores = torch.where(attn_mask, scores, neg_inf)
            probs = F.softmax(scores, dim=-1)
            probs = torch.where(attn_mask, probs, torch.zeros_like(probs))
            probs = self.attn_drop(probs)
            context = torch.matmul(probs, v).transpose(1, 2).contiguous().view(bsz, seq_len, dim)
        x = self.norm1(x + self.resid_drop(self.out_proj(context)))
        x = self.norm2(x + self.ffn(x))
        return x


class SASRecEncoder(BaseSequentialEncoder):
    """Architecture 2: Standard Post-LN Softmax Self-Attention (SASRec)."""

    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 128,
        max_seq_len: int = 50,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            num_items=num_items,
            embedding_dim=embedding_dim,
            max_seq_len=max_seq_len,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.pos_embedding = nn.Embedding(max_seq_len, embedding_dim)
        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)
        self.emb_norm = nn.LayerNorm(embedding_dim)
        self.emb_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [_SASRecBlock(embedding_dim, num_heads, dropout) for _ in range(num_layers)]
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        pos_ids = torch.clamp(
            torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, seq_len),
            max=self.max_seq_len - 1,
        )
        valid = (input_ids > 0).to(self.item_embedding.weight.dtype).unsqueeze(-1)
        x = self.item_embedding(input_ids) + self.pos_embedding(pos_ids)
        x = self.emb_drop(self.emb_norm(x)) * valid

        attn_mask = _causal_attention_mask(input_ids)
        for blk in self.blocks:
            x = blk(x, attn_mask) * valid
        return x

class _SASRecAblationBlock(_SASRecBlock):
    """SASRec block with a switchable attention mode for operator causal ablation.

    Modes (PREREG-attention-ablation-v1.1, GPT-approved):
    - "full":      identical to _SASRecBlock (Q/K/V from content stream) — arm A0.
    - "uniform":   attention probs replaced by uniform 1/n_t over the valid causal
                   prefix (Q/K not used); V/out_proj/FFN/residual/LN unchanged — arm A1.
    - "pos_only":  Q/K projected from the raw positional embedding only, V from the
                   content stream; softmax, masks, and all other ops unchanged — arm A2.
    """

    VALID_MODES = ("full", "uniform", "pos_only")

    def __init__(self, embedding_dim: int, num_heads: int, dropout: float, attn_mode: str = "full") -> None:
        super().__init__(embedding_dim=embedding_dim, num_heads=num_heads, dropout=dropout)
        if attn_mode not in self.VALID_MODES:
            raise ValueError(f"attn_mode must be one of {self.VALID_MODES}, got '{attn_mode}'")
        self.attn_mode = attn_mode

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        pos_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.attn_mode == "full":
            return super().forward(x, attn_mask)
        bsz, seq_len, dim = x.shape
        v = self.v_proj(x).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        if self.attn_mode == "uniform":
            # A_{t,j} = 1/n_t over the valid causal prefix; rows with no valid history get all-zero weights.
            # attn_mask is already (b, 1, L, L); broadcast over heads directly.
            mask = attn_mask
            denom = mask.sum(dim=-1, keepdim=True).clamp(min=1.0)
            probs = torch.where(mask, mask.to(v.dtype) / denom, torch.zeros_like(mask, dtype=v.dtype))
        else:  # pos_only
            if pos_emb is None:
                raise ValueError("pos_only mode requires pos_emb (raw positional embedding)")
            q = self.q_proj(pos_emb).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            k = self.k_proj(pos_emb).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            neg_inf_val = -1e4 if scores.dtype == torch.float16 else -1e9
            neg_inf = torch.full_like(scores, neg_inf_val)
            scores = torch.where(attn_mask, scores, neg_inf)
            probs = F.softmax(scores, dim=-1)
            probs = torch.where(attn_mask, probs, torch.zeros_like(probs))
        probs = self.attn_drop(probs)
        context = torch.matmul(probs, v).transpose(1, 2).contiguous().view(bsz, seq_len, dim)
        x = self.norm1(x + self.resid_drop(self.out_proj(context)))
        x = self.norm2(x + self.ffn(x))
        return x


class SASRecAblationEncoder(SASRecEncoder):
    """SASRec encoder whose blocks run a fixed attention mode (A0 full / A1 uniform / A2 pos_only).

    Parameter layout, initialization order, and shapes are identical to SASRecEncoder so that
    same-seed construction yields bitwise-identical initial weights across modes (prereg contract).
    """

    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 128,
        max_seq_len: int = 50,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        attn_mode: str = "full",
    ) -> None:
        super().__init__(
            num_items=num_items,
            embedding_dim=embedding_dim,
            max_seq_len=max_seq_len,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.blocks = nn.ModuleList(
            [_SASRecAblationBlock(embedding_dim, num_heads, dropout, attn_mode=attn_mode) for _ in range(num_layers)]
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        pos_ids = torch.clamp(
            torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, seq_len),
            max=self.max_seq_len - 1,
        )
        valid = (input_ids > 0).to(self.item_embedding.weight.dtype).unsqueeze(-1)
        x = self.item_embedding(input_ids) + self.pos_embedding(pos_ids)
        x = self.emb_drop(self.emb_norm(x)) * valid

        attn_mask = _causal_attention_mask(input_ids)
        raw_pos = self.pos_embedding(pos_ids)  # A2 Q/K source: raw positional embedding
        for blk in self.blocks:
            x = blk(x, attn_mask, pos_emb=raw_pos) * valid
        return x


class _LiGRBlock(nn.Module):
    """Pre-LN + Gated Residual + SwiGLU FFN block (LiGR-SASRec / eSASRec backbone)."""

    def __init__(self, embedding_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})")
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.use_sdpa = False

        self.norm1 = nn.LayerNorm(embedding_dim)
        self.q_proj = nn.Linear(embedding_dim, embedding_dim)
        self.k_proj = nn.Linear(embedding_dim, embedding_dim)
        self.v_proj = nn.Linear(embedding_dim, embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        self.attn_gate = nn.Linear(embedding_dim * 2, embedding_dim)
        self.attn_drop = nn.Dropout(dropout)

        self.norm2 = nn.LayerNorm(embedding_dim)
        self.ffn_gate = nn.Linear(embedding_dim, embedding_dim * 2)
        self.ffn_val = nn.Linear(embedding_dim, embedding_dim * 2)
        self.ffn_out = nn.Linear(embedding_dim * 2, embedding_dim)
        self.ffn_res_gate = nn.Linear(embedding_dim * 2, embedding_dim)
        self.ffn_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, dim = x.shape
        h = self.norm1(x)
        q = self.q_proj(h).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h).view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        if self.use_sdpa:
            drop_p = self.attn_drop.p if self.training else 0.0
            sdpa_out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=drop_p,
                is_causal=False,
            )
            attn_out = self.out_proj(sdpa_out.transpose(1, 2).contiguous().view(bsz, seq_len, dim))
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            neg_inf_val = -1e4 if scores.dtype == torch.float16 else -1e9
            scores = torch.where(attn_mask, scores, torch.full_like(scores, neg_inf_val))
            probs = F.softmax(scores, dim=-1)
            probs = torch.where(attn_mask, probs, torch.zeros_like(probs))
            probs = self.attn_drop(probs)
            attn_out = self.out_proj(
                torch.matmul(probs, v).transpose(1, 2).contiguous().view(bsz, seq_len, dim)
            )
        gate_1 = torch.sigmoid(self.attn_gate(torch.cat([x, attn_out], dim=-1)))
        x = x + gate_1 * attn_out

        h2 = self.norm2(x)
        ffn_hidden = F.silu(self.ffn_gate(h2)) * self.ffn_val(h2)
        ffn_out = self.ffn_out(self.ffn_drop(ffn_hidden))
        gate_2 = torch.sigmoid(self.ffn_res_gate(torch.cat([x, ffn_out], dim=-1)))
        x = x + gate_2 * ffn_out
        return x


class LiGRSASRecEncoder(BaseSequentialEncoder):
    """Architecture 3: LiGR-SASRec (Pre-LN + Gated Residual + SwiGLU FFN, eSASRec backbone)."""

    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 128,
        max_seq_len: int = 50,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            num_items=num_items,
            embedding_dim=embedding_dim,
            max_seq_len=max_seq_len,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.pos_embedding = nn.Embedding(max_seq_len, embedding_dim)
        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)
        self.emb_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [_LiGRBlock(embedding_dim, num_heads, dropout) for _ in range(num_layers)]
        )
        self.final_norm = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        pos_ids = torch.clamp(
            torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, seq_len),
            max=self.max_seq_len - 1,
        )
        valid = (input_ids > 0).to(self.item_embedding.weight.dtype).unsqueeze(-1)
        x = self.emb_drop(self.item_embedding(input_ids) + self.pos_embedding(pos_ids)) * valid

        attn_mask = _causal_attention_mask(input_ids)
        for blk in self.blocks:
            x = blk(x, attn_mask) * valid
        return self.final_norm(x) * valid


@torch.jit.script
def _fused_hstu_pointwise_silu_attn(
    q_h: torch.Tensor,
    k_h: torch.Tensor,
    v_h: torch.Tensor,
    rel_bias: torch.Tensor,
    attn_mask: torch.Tensor,
    scale: float,
    inv_n: float,
) -> torch.Tensor:
    raw_scores = torch.matmul(q_h, k_h.transpose(-2, -1)) * scale + rel_bias
    masked_scores = torch.where(attn_mask, raw_scores, torch.zeros_like(raw_scores))
    attn_weights = (F.silu(masked_scores) * attn_mask.to(raw_scores.dtype)) * inv_n
    return torch.matmul(attn_weights, v_h)


class _HSTUMathBlock(nn.Module):
    """HSTU-Math block: Pointwise Aggregated SiLU Attention + relative time/pos bias + Gated Linear Unit.

    Mathematical definition (Spec §5.2):
        U, V, Q, K = Split(SiLU(Linear(LN(X))))
        A_{ij} = (1 / N) * (SiLU(Q_i K_j^T / sqrt(d_h) + b_rel(t_i - t_j, i - j)) * M_{ij}^{causal})
        Y = Linear(LayerNorm(A V) * U) + X
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        max_seq_len: int,
        num_time_buckets: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by num_heads ({num_heads})")
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.max_seq_len = max_seq_len
        self.num_time_buckets = num_time_buckets
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.use_sdpa = False

        self.pre_norm = nn.LayerNorm(embedding_dim)
        self.uvqk_proj = nn.Linear(embedding_dim, embedding_dim * 4)
        self.rel_pos_bias = nn.Embedding(max_seq_len, num_heads)
        self.rel_time_bias = nn.Embedding(num_time_buckets, num_heads)
        nn.init.zeros_(self.rel_pos_bias.weight)
        nn.init.zeros_(self.rel_time_bias.weight)

        self.av_norm = nn.LayerNorm(embedding_dim)
        self.out_proj = nn.Linear(embedding_dim, embedding_dim)
        self.drop = nn.Dropout(dropout)

    def _compute_relative_bias(
        self,
        seq_len: int,
        timestamps: Optional[torch.Tensor],
        causal_tril: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        # Relative position index (i - j) clamped to [0, max_seq_len - 1]
        idx = torch.arange(seq_len, device=device)
        rel_pos = torch.clamp(idx.unsqueeze(1) - idx.unsqueeze(0), min=0, max=self.max_seq_len - 1)
        pos_b = self.rel_pos_bias(rel_pos).permute(2, 0, 1).unsqueeze(0)  # (1, H, N, N)

        if timestamps is None:
            return pos_b

        # Compute causal time differences strictly on j <= i so future timestamps j > i are never read
        ts = timestamps.to(torch.float32)
        dt = ts.unsqueeze(2) - ts.unsqueeze(1)  # (B, N, N)
        dt_causal = torch.where(causal_tril.unsqueeze(0), torch.clamp(dt, min=0.0), torch.zeros_like(dt))
        # Log-bucketize day differences into [0, num_time_buckets - 1]
        time_buckets = torch.clamp(
            torch.floor(torch.log2(dt_causal + 1.0) * 4.0).to(torch.long),
            min=0,
            max=self.num_time_buckets - 1,
        )
        time_b = self.rel_time_bias(time_buckets).permute(0, 3, 1, 2)  # (B, H, N, N)
        return pos_b + time_b

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seq_len, dim = x.shape
        h = self.pre_norm(x)
        uvqk = F.silu(self.uvqk_proj(h))
        u, v, q, k = torch.chunk(uvqk, 4, dim=-1)

        q_h = q.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k_h = k.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v_h = v.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        causal_tril = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device))
        rel_bias = self._compute_relative_bias(seq_len, timestamps, causal_tril, x.device).to(q_h.dtype)
        inv_n = 1.0 / float(max(seq_len, 1))

        if self.use_sdpa:
            av_h = _fused_hstu_pointwise_silu_attn(
                q_h, k_h, v_h, rel_bias, attn_mask, self.scale, inv_n
            )
            av = av_h.transpose(1, 2).contiguous().view(bsz, seq_len, dim)
        else:
            raw_scores = torch.matmul(q_h, k_h.transpose(-2, -1)) * self.scale + rel_bias
            # Strictly zero out j > i BEFORE SiLU and multiply by causal mask (1/N scaling, no Softmax)
            masked_scores = torch.where(attn_mask, raw_scores, torch.zeros_like(raw_scores))
            attn_weights = (F.silu(masked_scores) * attn_mask.to(raw_scores.dtype)) * inv_n
            av = torch.matmul(attn_weights, v_h).transpose(1, 2).contiguous().view(bsz, seq_len, dim)
        gated = self.av_norm(av) * u
        return x + self.drop(self.out_proj(gated))


class HSTUMathEncoder(BaseSequentialEncoder):
    """Architecture 4: HSTU-Math (Pointwise Aggregated SiLU Attention + Relative Time Bias + GLU)."""

    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 128,
        max_seq_len: int = 50,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(
            num_items=num_items,
            embedding_dim=embedding_dim,
            max_seq_len=max_seq_len,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.pos_embedding = nn.Embedding(max_seq_len, embedding_dim)
        nn.init.normal_(self.pos_embedding.weight, mean=0.0, std=0.02)
        self.emb_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                _HSTUMathBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    max_seq_len=max_seq_len,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        input_ids: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        pos_ids = torch.clamp(
            torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, seq_len),
            max=self.max_seq_len - 1,
        )
        valid = (input_ids > 0).to(self.item_embedding.weight.dtype).unsqueeze(-1)
        x = self.emb_drop(self.item_embedding(input_ids) + self.pos_embedding(pos_ids)) * valid

        attn_mask = _causal_attention_mask(input_ids)
        for blk in self.blocks:
            x = blk(x, attn_mask, timestamps=timestamps) * valid
        return self.final_norm(x) * valid


class _HSTUAlignedBlock(nn.Module):
    """Official-semantics HSTU block (meta-recsys/generative-recommenders STULayer, dense transcription).

    Reference identity (ops/hstu_compute.py + ops/pytorch/pt_hstu_attention.py + pt_hstu_linear.py):
        h    = LayerNorm(x, eps=1e-6)
        uvqk = h @ W + b            split -> u, v, q, k  (per-head widths: hidden*2 + attn*2)
        u    = SiLU(u);  q, k, v raw (NO SiLU on q/k/v)
        A    = SiLU(q k^T * alpha) / N * mask        alpha = 1/sqrt(attn_dim); N = padded seq len
        av   = A @ v
        y    = Dropout( LayerNorm(av, eps=1e-6) * u )
        out  = x + cat([u, x, y], -1) @ W_out + b_out
    No relative bias inside attention (reference implements position/time via input embeddings).
    """

    def __init__(self, embedding_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.per_head = embedding_dim // num_heads  # official ratio: UV and QK per-head = d/H
        self.alpha = 1.0 / math.sqrt(self.per_head)

        self.pre_norm = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.uvqk = nn.Linear(embedding_dim, self.per_head * 4 * num_heads)
        nn.init.xavier_uniform_(self.uvqk.weight)
        nn.init.zeros_(self.uvqk.bias)
        self.av_norm = nn.LayerNorm(embedding_dim, eps=1e-6)
        self.out_proj = nn.Linear(embedding_dim * 3, embedding_dim)
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, dim = x.shape
        h = self.pre_norm(x)
        uvqk = self.uvqk(h)
        ph = self.per_head
        u, v, q, k = torch.split(
            uvqk,
            [ph * self.num_heads, ph * self.num_heads, ph * self.num_heads, ph * self.num_heads],
            dim=-1,
        )
        u = F.silu(u)
        q = q.view(bsz, seq_len, self.num_heads, ph).transpose(1, 2)
        k = k.view(bsz, seq_len, self.num_heads, ph).transpose(1, 2)
        v = v.view(bsz, seq_len, self.num_heads, ph).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.alpha
        attn_w = (F.silu(scores) / float(seq_len)) * attn_mask.to(scores.dtype)
        av = torch.matmul(attn_w, v).transpose(1, 2).contiguous().view(bsz, seq_len, dim)
        y = self.drop(self.av_norm(av) * u)
        return x + self.out_proj(torch.cat([u, x, y], dim=-1))


class HSTUMathAlignedEncoder(BaseSequentialEncoder):
    """HSTU with official reference semantics (input pos/time embeddings, silu-u-only, 3x output concat).

    Differences vs HSTUMathEncoder (V1), all traced to meta-recsys/generative-recommenders:
      D1  SiLU on u only (V1 applied SiLU to q/k/v as well).
      D2  Position (distance-from-sequence-END bucket) and time (log-minutes bucket) embeddings
          added to input tokens, d-dim each, uniform init sqrt(1/num_buckets) — replaces V1's
          absolute position embedding + per-head additive logit bias inside attention.
      D3  Output projection consumes cat([u, x, y]) (3x width); V1 projected y only.
      D4  LayerNorm eps 1e-6 (V1 default 1e-5); dropout on y before out_proj (V1: after).
    Shared output contract with all R2 encoders is unchanged.
    """

    def __init__(
        self,
        num_items: int,
        embedding_dim: int = 128,
        max_seq_len: int = 50,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        num_time_buckets: int = 2048,
    ) -> None:
        super().__init__(
            num_items=num_items,
            embedding_dim=embedding_dim,
            max_seq_len=max_seq_len,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.num_time_buckets = int(num_time_buckets)
        bound_pos = (1.0 / (max_seq_len + 1)) ** 0.5
        self.pos_embedding = nn.Embedding(max_seq_len + 1, embedding_dim)
        nn.init.uniform_(self.pos_embedding.weight, -bound_pos, bound_pos)
        bound_ts = (1.0 / num_time_buckets) ** 0.5
        self.time_embedding = nn.Embedding(num_time_buckets + 1, embedding_dim)
        nn.init.uniform_(self.time_embedding.weight, -bound_ts, bound_ts)
        self.emb_drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                _HSTUAlignedBlock(embedding_dim=embedding_dim, num_heads=num_heads, dropout=dropout)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(embedding_dim, eps=1e-6)

    def _input_embeddings(
        self,
        input_ids: torch.Tensor,
        timestamps: Optional[torch.Tensor],
    ) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        valid = input_ids > 0
        x = self.emb_drop(self.item_embedding(input_ids))
        lengths = valid.sum(dim=1)  # (B,)
        # Official _get_col_indices (no targets, no contextual): pos index = seq_len - i, clamp [?, max]
        pos_idx = torch.clamp(lengths.unsqueeze(1) - torch.arange(seq_len, device=input_ids.device).unsqueeze(0), min=0, max=self.max_seq_len)
        if timestamps is None:
            time_idx = torch.zeros_like(pos_idx)
        else:
            ts = timestamps.to(torch.float32)
            safe_pos = torch.clamp(lengths - 1, min=0)
            query_time = ts.gather(1, safe_pos.unsqueeze(1))  # last event's timestamp
            delta_minutes = (query_time - ts) * 1440.0  # days -> minutes (official: seconds/60)
            delta_minutes = torch.where(valid, torch.clamp(delta_minutes, min=1.6667e-8), torch.zeros_like(delta_minutes))
            time_idx = torch.clamp(torch.log(delta_minutes).nan_to_num(0.0), min=0).to(torch.long)
            time_idx = torch.clamp(time_idx, max=self.num_time_buckets)
        pos_emb = self.pos_embedding(pos_idx) * valid.unsqueeze(-1).to(x.dtype)
        time_emb = self.time_embedding(time_idx) * valid.unsqueeze(-1).to(x.dtype)
        return (x + pos_emb + time_emb) * valid.unsqueeze(-1).to(x.dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        valid_f = (input_ids > 0).to(self.item_embedding.weight.dtype).unsqueeze(-1)
        x = self._input_embeddings(input_ids, timestamps)
        attn_mask = _causal_attention_mask(input_ids)
        for blk in self.blocks:
            x = blk(x, attn_mask) * valid_f
        return self.final_norm(x) * valid_f


def build_r2_encoder(
    encoder_name: str,
    num_items: int,
    embedding_dim: int = 128,
    max_seq_len: int = 50,
    num_layers: int = 2,
    num_heads: int = 4,
    dropout: float = 0.1,
) -> BaseSequentialEncoder:
    key = encoder_name.strip().lower()
    mapping = {
        "mean_pool": MeanPoolEncoder,
        "pool": MeanPoolEncoder,
        "gru": GRUEncoder,
        "sasrec": SASRecEncoder,
        "ligr_sasrec": LiGRSASRecEncoder,
        "esasrec": LiGRSASRecEncoder,
        "hstu_math": HSTUMathEncoder,
        "hstu": HSTUMathEncoder,
        "hstu_aligned": HSTUMathAlignedEncoder,
        "sasrec_full": lambda **kw: SASRecAblationEncoder(**kw, attn_mode="full"),
        "sasrec_uniform": lambda **kw: SASRecAblationEncoder(**kw, attn_mode="uniform"),
        "sasrec_pos_only": lambda **kw: SASRecAblationEncoder(**kw, attn_mode="pos_only"),
    }
    if key not in mapping:
        raise ValueError(f"Unknown encoder_name '{encoder_name}'. Valid: {sorted(mapping.keys())}")
    return mapping[key](
        num_items=num_items,
        embedding_dim=embedding_dim,
        max_seq_len=max_seq_len,
        num_layers=num_layers,
        num_heads=num_heads,
        dropout=dropout,
    )
