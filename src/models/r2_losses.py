from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


def sample_negatives_excluding_positives(
    pos_ids: torch.Tensor,
    num_items: int,
    num_negatives: int = 256,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Vectorized collision-free negative sampling in [1, num_items] strictly excluding pos_ids."""
    out_shape = (*pos_ids.shape, int(num_negatives))
    if num_items <= 1:
        return torch.ones(out_shape, dtype=torch.long, device=pos_ids.device)

    # Draw uniform integers in [1, num_items - 1] and shift values >= pos_id by +1.
    # For any pos_id in [1, num_items], this maps bijectively onto {1..num_items} \ {pos_id}.
    raw = torch.randint(
        low=1,
        high=num_items,
        size=out_shape,
        dtype=torch.long,
        device=pos_ids.device,
        generator=generator,
    )
    pos_clamped = torch.clamp(pos_ids, min=1, max=num_items).unsqueeze(-1)
    neg_ids = raw + (raw >= pos_clamped).to(torch.long)
    return neg_ids


def _masked_mean(loss_per_pos: torch.Tensor, pos_ids: torch.Tensor, valid_mask: Optional[torch.Tensor]) -> torch.Tensor:
    mask = (pos_ids > 0) if valid_mask is None else valid_mask.to(torch.bool)
    mask_f = mask.to(loss_per_pos.dtype)
    denom = torch.clamp(mask_f.sum(), min=1.0)
    return (loss_per_pos * mask_f).sum() / denom


class BCE1Loss(nn.Module):
    """Historical anchor: Binary Cross-Entropy with 1 negative sample per positive step."""

    def __init__(self) -> None:
        super().__init__()

    def forward(
        self,
        user_repr: torch.Tensor,
        item_embeddings: torch.Tensor,
        pos_ids: torch.Tensor,
        neg_ids: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pos_emb = item_embeddings[pos_ids]
        neg_1 = neg_ids[..., 0]
        neg_emb = item_embeddings[neg_1]

        pos_logits = (user_repr * pos_emb).sum(dim=-1)
        neg_logits = (user_repr * neg_emb).sum(dim=-1)

        loss_per_pos = F.softplus(-pos_logits) + F.softplus(neg_logits)
        return _masked_mean(loss_per_pos, pos_ids, valid_mask)


class BCE256Loss(nn.Module):
    """Independent Sigmoid Binary Cross-Entropy over 1 positive and K=256 negative samples."""

    def __init__(self, num_negatives: int = 256) -> None:
        super().__init__()
        self.num_negatives = int(num_negatives)

    def forward(
        self,
        user_repr: torch.Tensor,
        item_embeddings: torch.Tensor,
        pos_ids: torch.Tensor,
        neg_ids: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        k_use = min(self.num_negatives, neg_ids.size(-1))
        neg_use = neg_ids[..., :k_use]

        if item_embeddings.size(0) <= 8192:
            u_2d = user_repr.reshape(-1, user_repr.size(-1))
            pos_1d = pos_ids.reshape(-1, 1)
            neg_2d = neg_use.reshape(-1, k_use)
            all_scores = torch.matmul(u_2d, item_embeddings.t())
            pos_logits = all_scores.gather(1, pos_1d).squeeze(1).reshape(pos_ids.shape)
            neg_logits = all_scores.gather(1, neg_2d).reshape(*pos_ids.shape, k_use)
        else:
            pos_emb = item_embeddings[pos_ids]
            neg_emb = item_embeddings[neg_use]
            pos_logits = (user_repr * pos_emb).sum(dim=-1)
            neg_logits = (user_repr.unsqueeze(-2) * neg_emb).sum(dim=-1)

        pos_loss = F.softplus(-pos_logits)
        neg_loss = F.softplus(neg_logits).mean(dim=-1)
        loss_per_pos = pos_loss + neg_loss
        return _masked_mean(loss_per_pos, pos_ids, valid_mask)


class SampledSoftmax256Loss(nn.Module):
    """Sampled Softmax Cross-Entropy over 1 positive + K=256 negatives with optional LogQ correction."""

    def __init__(
        self,
        num_negatives: int = 256,
        temperature: float = 1.0,
        logq_correction: bool = True,
    ) -> None:
        super().__init__()
        self.num_negatives = int(num_negatives)
        self.temperature = max(float(temperature), 1e-4)
        self.logq_correction = bool(logq_correction)

    def forward(
        self,
        user_repr: torch.Tensor,
        item_embeddings: torch.Tensor,
        pos_ids: torch.Tensor,
        neg_ids: torch.Tensor,
        item_log_probs: Optional[torch.Tensor] = None,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        k_use = min(self.num_negatives, neg_ids.size(-1))
        neg_use = neg_ids[..., :k_use]

        if item_embeddings.size(0) <= 8192:
            u_2d = user_repr.reshape(-1, user_repr.size(-1))
            pos_1d = pos_ids.reshape(-1, 1)
            neg_2d = neg_use.reshape(-1, k_use)
            all_scores = torch.matmul(u_2d, item_embeddings.t()) / self.temperature
            if self.logq_correction and item_log_probs is not None:
                logq_table = item_log_probs.to(device=user_repr.device, dtype=user_repr.dtype)
                all_scores = all_scores - logq_table.unsqueeze(0)
            pos_logits = all_scores.gather(1, pos_1d).squeeze(1).reshape(pos_ids.shape)
            neg_logits = all_scores.gather(1, neg_2d).reshape(*pos_ids.shape, k_use)
        else:
            pos_emb = item_embeddings[pos_ids]
            neg_emb = item_embeddings[neg_use]
            pos_logits = (user_repr * pos_emb).sum(dim=-1) / self.temperature
            neg_logits = (user_repr.unsqueeze(-2) * neg_emb).sum(dim=-1) / self.temperature
            if self.logq_correction and item_log_probs is not None:
                logq_table = item_log_probs.to(device=user_repr.device, dtype=user_repr.dtype)
                pos_logits = pos_logits - logq_table[pos_ids]
                neg_logits = neg_logits - logq_table[neg_use]

        all_logits = torch.cat([pos_logits.unsqueeze(-1), neg_logits], dim=-1)
        loss_per_pos = -F.log_softmax(all_logits, dim=-1)[..., 0]
        return _masked_mean(loss_per_pos, pos_ids, valid_mask)
