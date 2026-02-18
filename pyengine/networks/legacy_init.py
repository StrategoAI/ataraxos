from dataclasses import dataclass
import torch
import torch.nn as nn
import einops

from torch.nn.attention import sdpa_kernel, SDPBackend

from pyengine.utils import get_pystratego

pystratego = get_pystratego()


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        assert d_model % n_head == 0

        self.n_head = n_head
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, attn_mask=None, is_causal=False, kv_cache=None):
        """
        x: [batch, seq, d_model]
        attn_mask: [seq, seq]
        kv_cache: [2, batch, n_head, lookback_len, d_model]
        """
        qkv = self.qkv_proj(x)
        q, k, v = einops.rearrange(qkv, "b t (k h d) -> b k h t d", k=3, h=self.n_head).unbind(1)
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            attn_v = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, attn_mask=attn_mask, is_causal=is_causal
            )
        attn_v = einops.rearrange(attn_v, "b h t d -> b t (h d)")
        return self.out_proj(attn_v)


class TransformerLayer(nn.Module):
    def __init__(self, d_model, n_head, dropout, ff_factor=4):
        super().__init__()

        self.layer_norm1 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, n_head)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, ff_factor * d_model)
        self.linear2 = nn.Linear(ff_factor * d_model, d_model)

        self.dropout = dropout

    def forward(self, x, attn_mask=None, is_causal=False):
        x_ = self.mha(self.layer_norm1(x), attn_mask=attn_mask, is_causal=is_causal)
        x = x + nn.functional.dropout(x_, p=self.dropout)
        x = x + nn.functional.dropout(self._ff_block(self.layer_norm2(x)), p=self.dropout)
        return x

    def _ff_block(self, x):
        x = self.linear2(nn.functional.relu(self.linear1(x)))
        return x


@dataclass
class TransformerInitConfig:
    depth: int = 4
    embed_dim_per_head_over8: int = 8
    n_head: int = 8
    dropout: float = 0
    pos_emb_std: float = 0.1
    use_value_net: int = 1
    weight_counts: int = 1
    force_handedness: int = 1
    use_cat_vf: bool = True


class TransformerInitialization(nn.Module):
    def __init__(
        self,
        piece_counts,
        cfg: TransformerInitConfig,
        rank: int = 0,
    ) -> None:
        super().__init__()

        self.seq_len = 40
        self.n_class = pystratego.NUM_PIECE_TYPES
        assert piece_counts.size(0) == self.n_class

        self.piece_counts = piece_counts

        self.cfg = cfg
        # Ensure args satisfy:
        # 1. embed_dim is divisible by n_head
        # 2. embed_dim is divisible by 8
        self.cfg.embed_dim = 8 * cfg.embed_dim_per_head_over8 * cfg.n_head
        device = f"cuda:{rank}"

        assert piece_counts.sum() == self.seq_len

        self.embedder = torch.nn.Linear(self.n_class, cfg.embed_dim)

        # assume inputs are [batch, seq, embed_dim]
        self.positional_encoding = nn.Parameter(
            torch.empty(1, self.seq_len, cfg.embed_dim, device=device)
        )
        nn.init.trunc_normal_(self.positional_encoding, std=cfg.pos_emb_std)

        self.causal_mask = torch.tril(
            torch.ones(self.seq_len, self.seq_len, device=device), diagonal=0
        ).log()
        self.transformer = nn.Sequential(
            *[TransformerLayer(cfg.embed_dim, cfg.n_head, cfg.dropout) for _ in range(cfg.depth)]
        )
        self.norm_out = nn.LayerNorm(cfg.embed_dim)

        self.linear = nn.Linear(cfg.embed_dim, self.n_class)
        self.value_out = nn.Linear(cfg.embed_dim, 3)

        self.reg_out = nn.Linear(cfg.embed_dim, 1)

        self.to(device)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, seq: torch.Tensor) -> dict[str, torch.Tensor]:
        assert seq.shape[1] <= self.seq_len
        assert seq.shape[2] == self.n_class

        # Concatenate with dummy start
        seq = torch.cat(
            [torch.zeros(seq.shape[0], 1, self.n_class, device=self.device), seq], dim=1
        )
        # Remove final token if applicable
        seq = seq[:, : self.seq_len]

        seq_embedding = self.embedder(seq)
        x = seq_embedding + self.positional_encoding[:, : seq_embedding.shape[1]]

        for layer in self.transformer:
            x = layer(x, is_causal=True)
        x = self.norm_out(x)

        logits: torch.Tensor = self.linear(x)
        v: torch.Tensor = self.value_out(x)
        reg_pred = self.reg_out(x)

        # Mask predictions with invalid piece counts
        legal_action_mask = make_legal_action_mask(
            seq, self.piece_counts, self.cfg.force_handedness
        )
        assert legal_action_mask.any(dim=-1).all()
        logits.masked_fill_(~legal_action_mask, -1e10)

        if not self.cfg.use_value_net:
            v = torch.zeros_like(v, device=self.device)

        counts = remaining_counts(seq, self.piece_counts)
        if self.cfg.weight_counts:
            logits += counts.clamp(min=1).log()

        return {"logits": logits, "value": v, "ent_pred": reg_pred}


@torch.no_grad()
def remaining_counts(seq: torch.Tensor, piece_counts: torch.Tensor) -> torch.Tensor:
    # N.B. to use for masking, padding must have already been applied to seq
    cum_seq = seq.cumsum(dim=1)
    expanded_piece_counts = piece_counts.view(1, 1, -1).expand_as(cum_seq)
    return expanded_piece_counts - cum_seq


RIGHT_HAND = torch.tensor(
    4 * [False, False, False, False, False, True, True, True, True, True],
    dtype=torch.bool,
    device="cuda",
)


@torch.no_grad()
def make_legal_action_mask(
    seq: torch.Tensor, piece_counts: torch.Tensor, force_handedness: bool
) -> torch.Tensor:
    legal_mask = remaining_counts(seq, piece_counts) > 0
    if force_handedness:
        legal_mask[
            :, ~RIGHT_HAND[: legal_mask.size(1)].to(seq.device), 10
        ] = False  # 10 is the index of the flag
    assert legal_mask.any(dim=-1).all()
    return legal_mask
