
import math
import os
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN

try:
    from .choices import *
    from .configs import EConfig
    from .utils_c import *
except:
    from choices import *
    from configs import EConfig
    from utils import prepare_logits_processor
    from utils_c import *


# Copied from transformers.models.bart.modeling_bart._make_causal_mask
def _make_causal_mask(
    input_ids_shape: torch.Size,
    dtype: torch.dtype,
    device: torch.device,
    past_key_values_length: int = 0,
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.finfo(dtype).min, device=device)
    mask_cond = torch.arange(mask.size(-1), device=device)
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat(
            [
                torch.zeros(
                    tgt_len, past_key_values_length, dtype=dtype, device=device
                ),
                mask,
            ],
            dim=-1,
        )
    return mask[None, None, :, :].expand(
        bsz, 1, tgt_len, tgt_len + past_key_values_length
    )


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype, tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(
        inverted_mask.to(torch.bool), torch.finfo(dtype).min
    )


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False
        )

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


class LlamaLinearScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )
        t = t / self.scaling_factor

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False
        )


class LlamaDynamicNTKScalingRotaryEmbedding(LlamaRotaryEmbedding):
    """LlamaRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    def __init__(
        self,
        dim,
        max_position_embeddings=2048,
        base=10000,
        device=None,
        scaling_factor=1.0,
    ):
        self.scaling_factor = scaling_factor
        super().__init__(dim, max_position_embeddings, base, device)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings)
                - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (
                base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)

        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False
        )


class LlamaAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        if hasattr(config, "qkv_bias"):
            self.q_proj = nn.Linear(
                self.hidden_size, self.num_heads * self.head_dim, bias=config.qkv_bias
            )
            self.k_proj = nn.Linear(
                self.hidden_size,
                self.num_key_value_heads * self.head_dim,
                bias=config.qkv_bias,
            )
            self.v_proj = nn.Linear(
                self.hidden_size,
                self.num_key_value_heads * self.head_dim,
                bias=config.qkv_bias,
            )
        else:
            self.q_proj = nn.Linear(
                self.hidden_size, self.num_heads * self.head_dim, bias=False
            )
            self.k_proj = nn.Linear(
                self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
            )
            self.v_proj = nn.Linear(
                self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
            )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )
        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            if hasattr(self.config, "rope_theta"):
                self.rotary_emb = LlamaRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=self.config.rope_theta,
                )
            else:
                self.rotary_emb = LlamaRotaryEmbedding(
                    self.head_dim, max_position_embeddings=self.max_position_embeddings
                )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return (
            tensor.view(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )


    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        if self.config.pretraining_tp > 1:
            key_value_slicing = (
                self.num_key_value_heads * self.head_dim
            ) // self.config.pretraining_tp
            query_slices = self.q_proj.weight.split(
                (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
            )
            key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
            value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

            query_states = [
                F.linear(hidden_states, query_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            query_states = torch.cat(query_states, dim=-1)

            key_states = [
                F.linear(hidden_states, key_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            key_states = torch.cat(key_states, dim=-1)

            value_states = [
                F.linear(hidden_states, value_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            value_states = torch.cat(value_states, dim=-1)

        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, q_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        if position_ids is None:
            kv_seq_len = key_states.shape[-2]
            if past_key_value is not None:
                if len(past_key_value) == 2:
                    kv_seq_len += past_key_value[0].shape[-2]
                elif len(past_key_value) == 3:
                    kv_seq_len += past_key_value[2]
                else:
                    raise NotImplementedError
        else:
            kv_seq_len = position_ids.max() + 1
        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        query_states, key_states = apply_rotary_pos_emb(
            query_states, key_states, cos, sin, position_ids
        )

        if past_key_value is not None:
            # reuse k, v, self_attention
            key_states = torch.cat([past_key_value[0], key_states], dim=2)
            value_states = torch.cat([past_key_value[1], value_states], dim=2)

        if use_cache:
            min_thrh = -1e6
            compress_kv_cache = (
                past_key_value is None
                and bsz == 1
                and attention_mask is not None
                and (attention_mask[:, :, -1] <= min_thrh).any()
            )
            if compress_kv_cache:
                to_keep = (attention_mask[:, :, -1] > min_thrh).expand(
                    -1, self.num_key_value_heads, -1
                )
                cached_k = key_states[to_keep].reshape(
                    bsz, self.num_key_value_heads, -1, self.head_dim
                )
                cached_v = value_states[to_keep].reshape(
                    bsz, self.num_key_value_heads, -1, self.head_dim
                )
                past_key_value = (cached_k, cached_v, position_ids.max() + 1)
            else:
                past_key_value = (key_states, value_states, position_ids.max() + 1)
        else:
            past_key_value = None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if not output_attentions:
            attn_weights = None
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states.contiguous(),
                key_states.contiguous(),
                value_states.contiguous(),
                attn_mask=attention_mask.to(query_states.dtype),
            )
        else:
            attn_weights = (
                torch.matmul(query_states, key_states.transpose(-2, -1))
                * query_states.size(-1) ** -0.5
            )

            if attention_mask is not None:
                attn_weights = attn_weights + attention_mask.to(attn_weights.dtype)
            attn_weights = torch.softmax(attn_weights, dim=-1)
            attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        if self.config.pretraining_tp > 1:
            attn_output = attn_output.split(
                self.hidden_size // self.config.pretraining_tp, dim=2
            )
            o_proj_slices = self.o_proj.weight.split(
                self.hidden_size // self.config.pretraining_tp, dim=1
            )
            attn_output = sum(
                [
                    F.linear(attn_output[i], o_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ]
            )
        else:
            attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, past_key_value


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if self.config.pretraining_tp > 1:
            slice_ids = self.intermediate_size // self.config.pretraining_tp
            gate_proj_slices = self.gate_proj.weight.split(slice_ids, dim=0)
            up_proj_slices = self.up_proj.weight.split(slice_ids, dim=0)
            down_proj_slices = self.down_proj.weight.split(slice_ids, dim=1)

            gate_proj = torch.cat(
                [
                    F.linear(x, gate_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ],
                dim=-1,
            )
            up_proj = torch.cat(
                [
                    F.linear(x, up_proj_slices[i])
                    for i in range(self.config.pretraining_tp)
                ],
                dim=-1,
            )

            intermediate_states = (self.act_fn(gate_proj) * up_proj).split(
                slice_ids, dim=2
            )
            down_proj = [
                F.linear(intermediate_states[i], down_proj_slices[i])
                for i in range(self.config.pretraining_tp)
            ]
            down_proj = sum(down_proj)
        else:
            down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        return down_proj


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config, index):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaAttention(config=config)
        self.mlp = LlamaMLP(config)
        self.index = index
        if self.index != 0:
            self.input_layernorm = LlamaRMSNorm(
                config.hidden_size, eps=config.rms_norm_eps
            )
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """

        residual = hidden_states

        if self.index != 0:
            hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class ImgAdaptor(nn.Module):
    def __init__(self, config, num_q=2):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_q = num_q

        self.q = nn.Parameter(torch.empty(self.num_q, self.num_heads, self.head_dim))

        nn.init.normal_(self.q, mean=0, std=self.head_dim**-0.5)

        if hasattr(config, "qkv_bias"):
            bias = config.qkv_bias
        else:
            bias = False

        self.k_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=bias
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=bias
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

    def forward(self, hidden_states: torch.Tensor):
        bsz, seq_len, _ = hidden_states.size()

        query_states = self.q
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = (
            query_states.view(1, self.num_q, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .repeat_interleave(bsz, dim=0)
        )
        key_states = key_states.view(
            bsz, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query=query_states.contiguous(),
            key=key_states.contiguous(),
            value=value_states.contiguous(),
            is_causal=False,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, self.num_q, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output


class CrossAttentionPooling(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads

        if hasattr(config, "qkv_bias"):
            bias = config.qkv_bias
        else:
            bias = False

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=bias
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=bias
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

    def forward(self, visual_states: torch.Tensor, query_states: torch.Tensor):
        if visual_states.dim() != 3:
            raise ValueError("visual_states should have shape [batch, num_vis, hidden]")
        if query_states.dim() == 2:
            query_states = query_states.unsqueeze(1)
        elif query_states.dim() != 3:
            raise ValueError(
                "query_states should have shape [batch, hidden] or [batch, num_q, hidden]"
            )
        if visual_states.shape[0] != query_states.shape[0]:
            raise ValueError("visual_states and query_states batch size should match")
        if visual_states.shape[-1] != self.hidden_size:
            raise ValueError("visual_states hidden size should match model hidden size")
        if query_states.shape[-1] != self.hidden_size:
            raise ValueError("query_states hidden size should match model hidden size")

        bsz, num_vis, _ = visual_states.size()
        num_q = query_states.shape[1]

        query_states = self.q_proj(query_states)
        key_states = self.k_proj(visual_states)
        value_states = self.v_proj(visual_states)

        query_states = query_states.view(
            bsz, num_q, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, num_vis, self.num_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, num_vis, self.num_heads, self.head_dim
        ).transpose(1, 2)

        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query=query_states.contiguous(),
            key=key_states.contiguous(),
            value=value_states.contiguous(),
            is_causal=False,
        )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, num_q, self.hidden_size)
        return self.o_proj(attn_output)


class Model(nn.Module):

    def __init__(
        self,
        config,
        load_emb=False,
        path=None,
        bias=True,
        total_tokens=30,
        depth=3,
        top_k=8,
        threshold=1.0,
        num_q=2,
        vis_select_tokens=64,
        num_hidden_levels=3,
        min_vis_select_tokens=16,
        vis_entropy_alpha=1.2,
        vis_query_window=8,
        max_total_vis_select_tokens=None,
    ):
        super().__init__()

        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.num_hidden_levels = num_hidden_levels
        self.hidden_size = config.hidden_size

        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        if load_emb:
            import json

            from safetensors import safe_open
            from transformers import AutoModel, AutoModelForImageTextToText

            try:
                try:
                    try:
                        with open(
                            os.path.join(path, "model.safetensors.index.json"), "r"
                        ) as f:
                            index_json = json.loads(f.read())
                            emb_path = index_json["weight_map"][
                                "model.embed_tokens.weight"
                            ]
                        with safe_open(
                            os.path.join(path, emb_path), framework="pt", device="cpu"
                        ) as f:
                            tensor_slice = f.get_slice("model.embed_tokens.weight")
                            vocab_size, hidden_dim = tensor_slice.get_shape()
                            tensor = tensor_slice[:, :hidden_dim].float()
                    except:
                        with open(
                            os.path.join(path, "pytorch_model.bin.index.json"), "r"
                        ) as f:
                            index_json = json.loads(f.read())
                            emb_path = index_json["weight_map"][
                                "model.embed_tokens.weight"
                            ]
                        weights = torch.load(os.path.join(path, emb_path))
                        tensor = weights["model.embed_tokens.weight"].float()
                except:
                    m = AutoModelForImageTextToText.from_pretrained(
                        path, torch_dtype="auto"
                    )
                    try:
                        tensor = m.language_model.model.embed_tokens.weight.float()
                    except:
                        tensor = m.model.embed_tokens.weight.float()
                    del m
            except:
                tensor = torch.load(path)["embed_tokens.weight"].float()

            self.embed_tokens.weight.data = tensor

        self.top_k = top_k
        self.total_tokens = total_tokens - 1
        self.depth = depth
        self.threshold = math.log(threshold)
        self.vis_select_tokens = vis_select_tokens
        self.min_vis_select_tokens = min_vis_select_tokens
        self.vis_entropy_alpha = vis_entropy_alpha
        self.vis_query_window = vis_query_window
        self.max_total_vis_select_tokens = (
            max_total_vis_select_tokens
            if max_total_vis_select_tokens is not None and max_total_vis_select_tokens > 0
            else None
        )

        self.layers = nn.ModuleList(
            [
                LlamaDecoderLayer(config, index)
                for index in range(config.num_hidden_layers)
            ]
        )
        self.act = ACT2FN[config.hidden_act]
        self.logsoftmax = nn.LogSoftmax(dim=-1)

        # EAGLE-3 text fusion: [low, middle, high] -> g.
        text_mlp_input_dim = num_hidden_levels * config.hidden_size
        self.text_mlp = nn.Linear(text_mlp_input_dim, config.hidden_size, bias=bias)

        # Kept for checkpoint backward-compatibility with EAGLE/ViSpec checkpoints.
        self.fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=bias)
        # Global visual anchor fusion: [next_token_embedding, g/a, vis_anchor] -> draft hidden.
        self.anchor_fc = nn.Linear(
            3 * config.hidden_size, config.hidden_size, bias=bias
        )
        self._reset_anchor_fc_from_fc()

        self.vis_detail_pooler = CrossAttentionPooling(config)

        self.imadpt = ImgAdaptor(config, num_q)
        self.img_fc = nn.Linear(2 * config.hidden_size, config.hidden_size, bias=bias)

        nn.init.zeros_(self.img_fc.weight[:, config.hidden_size :])
        nn.init.eye_(self.img_fc.weight[:, : config.hidden_size])
        if self.img_fc.bias is not None:
            nn.init.zeros_(self.img_fc.bias)

        # Kept for checkpoint compatibility; direct visual selection does not use them.
        for param in self.imadpt.parameters():
            param.requires_grad = False
        for param in self.img_fc.parameters():
            param.requires_grad = False

        self.last_img_hidden = None
        self.last_vis_detail = None

        for param in self.embed_tokens.parameters():
            param.requires_grad = False

    def init_tree(self):
        self.register_buffer(
            "tree_mask_init",
            torch.eye(self.top_k, device=self.embed_tokens.weight.device)[None, None],
            persistent=False,
        )
        self.register_buffer(
            "position_ids",
            torch.zeros(
                self.top_k, device=self.embed_tokens.weight.device, dtype=torch.long
            ),
            persistent=False,
        )

    def reset(self):
        self.tree_mask = None

    def _reset_anchor_fc_from_fc(self):
        with torch.no_grad():
            self.anchor_fc.weight.zero_()
            self.anchor_fc.weight[:, : self.fc.in_features].copy_(self.fc.weight)
            if self.anchor_fc.bias is not None:
                if self.fc.bias is not None:
                    self.anchor_fc.bias.copy_(self.fc.bias)
                else:
                    self.anchor_fc.bias.zero_()

    def load_state_dict(self, state_dict, strict=True, *args, **kwargs):
        if not strict:
            model_state = self.state_dict()
            state_dict = {
                key: value
                for key, value in state_dict.items()
                if key in model_state and model_state[key].shape == value.shape
            }
        result = super().load_state_dict(
            state_dict, strict=strict, *args, **kwargs
        )
        if "anchor_fc.weight" not in state_dict:
            self._reset_anchor_fc_from_fc()
        return result

    def _prepare_decoder_attention_mask(
        self, attention_mask, input_shape, inputs_embeds, past_key_values_length
    ):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape,
                # inputs_embeds.dtype,
                torch.float32,  # [MODIFIED] force to cast to float32
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(
                attention_mask, torch.float32, tgt_len=input_shape[-1]
            ).to(inputs_embeds.device)
            combined_attention_mask = (
                expanded_attn_mask
                if combined_attention_mask is None
                else expanded_attn_mask + combined_attention_mask
            )

        # [MODIFIED] add tree mask
        if hasattr(self, "tree_mask") and self.tree_mask is not None:
            tree_mask = self.tree_mask
            _, _, tree_shape0, tree_shape1 = tree_mask.shape
            combined_attention_mask[:, :, -tree_shape0:, -tree_shape1:][
                tree_mask == 0
            ] = torch.finfo(torch.float32).min

        return combined_attention_mask


    def _align_image_mask(self, image_mask, seq_length):
        image_mask = image_mask[:, 1:]
        if image_mask.shape[1] < seq_length:
            pad = torch.zeros(
                image_mask.shape[0],
                seq_length - image_mask.shape[1],
                dtype=image_mask.dtype,
                device=image_mask.device,
            )
            image_mask = torch.cat((image_mask, pad), dim=1)
        elif image_mask.shape[1] > seq_length:
            image_mask = image_mask[:, :seq_length]
        return image_mask

    def _prepare_text_attn_vis(self, text_attn_vis, batch_size, seq_length, device):
        if text_attn_vis is None:
            return None

        if isinstance(text_attn_vis, (list, tuple)):
            text_attn_vis = text_attn_vis[-1]
        text_attn_vis = text_attn_vis.to(device)

        if text_attn_vis.dim() == 4:
            text_attn_vis = text_attn_vis.mean(dim=1)
        elif text_attn_vis.dim() == 3:
            if text_attn_vis.shape[0] not in (1, batch_size):
                text_attn_vis = text_attn_vis.mean(dim=0, keepdim=True)
        elif text_attn_vis.dim() == 2:
            text_attn_vis = text_attn_vis.unsqueeze(0)
        else:
            raise ValueError(
                "text_attn_vis should have shape [seq, seq], [batch, seq, seq], or [batch, heads, seq, seq]"
            )

        if text_attn_vis.shape[0] == 1 and batch_size > 1:
            text_attn_vis = text_attn_vis.expand(batch_size, -1, -1)

        q_len, k_len = text_attn_vis.shape[-2:]
        if q_len > seq_length:
            text_attn_vis = text_attn_vis[..., -seq_length:, :]
        if k_len > seq_length:
            text_attn_vis = text_attn_vis[..., -seq_length:]

        q_len, k_len = text_attn_vis.shape[-2:]
        if q_len < seq_length or k_len < seq_length:
            padded = text_attn_vis.new_zeros(batch_size, seq_length, seq_length)
            padded[:, :q_len, :k_len] = text_attn_vis
            text_attn_vis = padded

        return text_attn_vis

    def _prepare_vis_attn_scores(self, vis_attn_scores, batch_size, seq_length, device):
        if vis_attn_scores is None:
            return None

        if isinstance(vis_attn_scores, (list, tuple)):
            vis_attn_scores = vis_attn_scores[-1]
        vis_attn_scores = vis_attn_scores.to(device)

        if vis_attn_scores.dim() == 1:
            vis_attn_scores = vis_attn_scores.unsqueeze(0)
        elif vis_attn_scores.dim() == 3:
            if vis_attn_scores.shape[1] != 1:
                raise ValueError(
                    "vis_attn_scores should have shape [seq], [batch, seq], or [batch, 1, seq]"
                )
            vis_attn_scores = vis_attn_scores.squeeze(1)
        elif vis_attn_scores.dim() != 2:
            raise ValueError(
                "vis_attn_scores should have shape [seq], [batch, seq], or [batch, 1, seq]"
            )

        if vis_attn_scores.shape[0] == 1 and batch_size > 1:
            vis_attn_scores = vis_attn_scores.expand(batch_size, -1)
        if vis_attn_scores.shape[0] != batch_size:
            raise ValueError("vis_attn_scores batch size should match hidden_states")

        # Scores are stored in original base-model token coordinates. Align them
        # with next-token draft inputs exactly like image_mask.
        vis_attn_scores = vis_attn_scores[:, 1:]
        if vis_attn_scores.shape[1] < seq_length:
            pad = vis_attn_scores.new_zeros(
                batch_size, seq_length - vis_attn_scores.shape[1]
            )
            vis_attn_scores = torch.cat((vis_attn_scores, pad), dim=1)
        elif vis_attn_scores.shape[1] > seq_length:
            vis_attn_scores = vis_attn_scores[:, :seq_length]

        return vis_attn_scores.float()

    def _prepare_query_token_mask(self, query_token_mask, batch_size, seq_length, device):
        if query_token_mask is None:
            return None

        if isinstance(query_token_mask, (list, tuple)):
            query_token_mask = query_token_mask[-1]
        query_token_mask = query_token_mask.to(device=device, dtype=torch.bool)

        if query_token_mask.dim() == 1:
            query_token_mask = query_token_mask.unsqueeze(0)
        elif query_token_mask.dim() == 3:
            if query_token_mask.shape[1] != 1:
                raise ValueError(
                    "query_token_mask should have shape [seq], [batch, seq], or [batch, 1, seq]"
                )
            query_token_mask = query_token_mask.squeeze(1)
        elif query_token_mask.dim() != 2:
            raise ValueError(
                "query_token_mask should have shape [seq], [batch, seq], or [batch, 1, seq]"
            )

        if query_token_mask.shape[0] == 1 and batch_size > 1:
            query_token_mask = query_token_mask.expand(batch_size, -1)
        if query_token_mask.shape[0] != batch_size:
            raise ValueError("query_token_mask batch size should match hidden_states")

        # Stored in original base-model token coordinates; align with image_mask.
        query_token_mask = query_token_mask[:, 1:]
        if query_token_mask.shape[1] < seq_length:
            pad = torch.zeros(
                batch_size,
                seq_length - query_token_mask.shape[1],
                dtype=torch.bool,
                device=device,
            )
            query_token_mask = torch.cat((query_token_mask, pad), dim=1)
        elif query_token_mask.shape[1] > seq_length:
            query_token_mask = query_token_mask[:, :seq_length]

        return query_token_mask

    def _prepare_vis_anchor(self, vis_anchor, batch_size, reference):
        if vis_anchor is None:
            return None

        if isinstance(vis_anchor, (list, tuple)):
            vis_anchor = vis_anchor[-1]
        vis_anchor = vis_anchor.to(device=reference.device, dtype=reference.dtype)

        if vis_anchor.dim() == 1:
            vis_anchor = vis_anchor.unsqueeze(0)
        elif vis_anchor.dim() == 3:
            if vis_anchor.shape[1] != 1:
                raise ValueError(
                    "vis_anchor should be a single token with shape [batch, 1, hidden_size]"
                )
            vis_anchor = vis_anchor.squeeze(1)
        elif vis_anchor.dim() != 2:
            raise ValueError(
                "vis_anchor should have shape [hidden_size], [batch, hidden_size], or [batch, 1, hidden_size]"
            )

        if vis_anchor.shape[0] == 1 and batch_size > 1:
            vis_anchor = vis_anchor.expand(batch_size, -1)
        if vis_anchor.shape[0] != batch_size:
            raise ValueError(
                "vis_anchor batch size should match hidden_states batch size"
            )
        if vis_anchor.shape[-1] != self.hidden_size:
            raise ValueError(
                "vis_anchor hidden size should match model hidden size"
            )

        return vis_anchor

    def _expand_anchor_like(self, hidden_states, vis_anchor=None, batch_idx=None):
        if vis_anchor is None:
            return None

        if hidden_states.dim() == 2:
            anchor_idx = 0 if batch_idx is None else batch_idx
            return vis_anchor[anchor_idx].unsqueeze(0).expand(
                hidden_states.shape[0], -1
            )
        if hidden_states.dim() == 3:
            return vis_anchor[:, None, :].expand(
                -1, hidden_states.shape[1], -1
            )
        raise ValueError("hidden_states should be 2D or 3D when expanding vis_anchor")

    def _fuse_multilevel_text_tokens(self, multi_level_hidden):
        """
        EAGLE-3 text fusion: [low, middle, high] -> g.
        """
        if multi_level_hidden.numel() == 0:
            return multi_level_hidden

        expected_hidden_size = self.num_hidden_levels * self.hidden_size
        if multi_level_hidden.shape[-1] != expected_hidden_size:
            raise ValueError(
                f"multi_level_hidden last dimension should be {expected_hidden_size}, "
                f"got {multi_level_hidden.shape[-1]}"
            )

        return self.text_mlp(multi_level_hidden)

    def _build_text_context(self, hidden_states):
        hidden_dim = hidden_states.shape[-1]
        if hidden_dim == self.num_hidden_levels * self.hidden_size:
            return self._fuse_multilevel_text_tokens(hidden_states)
        if hidden_dim == self.hidden_size:
            return hidden_states
        raise ValueError(
            "hidden_states last dimension should be either "
            f"{self.num_hidden_levels * self.hidden_size} for multi-level target "
            f"states or {self.hidden_size} for draft autoregressive states, "
            f"got {hidden_dim}"
        )

    def _merge_token_and_hidden(
        self,
        inputs_embeds,
        hidden_context,
        vis_anchor=None,
        batch_idx=None,
    ):
        if hidden_context.numel() == 0:
            return hidden_context
        if hidden_context.shape[-1] != self.hidden_size:
            raise ValueError(
                f"hidden_context last dimension should be {self.hidden_size}, "
                f"got {hidden_context.shape[-1]}"
            )
        if inputs_embeds.shape[:-1] != hidden_context.shape[:-1]:
            raise ValueError(
                "inputs_embeds and hidden_context should have matching leading "
                f"dimensions, got {inputs_embeds.shape[:-1]} and "
                f"{hidden_context.shape[:-1]}"
            )
        if inputs_embeds.shape[-1] != self.hidden_size:
            raise ValueError(
                f"inputs_embeds last dimension should be {self.hidden_size}, "
                f"got {inputs_embeds.shape[-1]}"
            )

        inputs_embeds = inputs_embeds.to(hidden_context)
        if vis_anchor is None:
            anchor = torch.zeros_like(hidden_context)
        else:
            anchor = self._expand_anchor_like(
                hidden_context, vis_anchor=vis_anchor, batch_idx=batch_idx
            )

        return self.anchor_fc(
            torch.cat((inputs_embeds, hidden_context, anchor), dim=-1)
        )

    def _build_visual_pool_query(
        self, hidden_states, image_mask, query_token_mask, batch_idx
    ):
        seq_ids = torch.arange(image_mask.shape[1], device=image_mask.device)
        text_ids = None
        if query_token_mask is not None:
            text_ids = seq_ids[query_token_mask[batch_idx] & ~image_mask[batch_idx]]
        if text_ids is None or text_ids.numel() == 0:
            text_ids = seq_ids[~image_mask[batch_idx]]
        if text_ids.numel() == 0:
            return torch.zeros(
                1,
                self.hidden_size,
                dtype=hidden_states.dtype,
                device=hidden_states.device,
            )

        query_window = min(self.vis_query_window, text_ids.numel())
        text_ids = text_ids[-query_window:]
        text_context = self._build_text_context(hidden_states[batch_idx, text_ids])
        return text_context.mean(dim=0, keepdim=True)

    def _pool_selected_visual_tokens(self, visual_embeds, query_context):
        if visual_embeds.numel() == 0:
            return None
        if query_context is None or query_context.numel() == 0:
            query_context = torch.zeros(
                1,
                self.hidden_size,
                dtype=visual_embeds.dtype,
                device=visual_embeds.device,
            )
        if query_context.dim() == 1:
            query_context = query_context.unsqueeze(0)
        if query_context.dim() != 2:
            raise ValueError("query_context should have shape [hidden] or [1, hidden]")

        if visual_embeds.shape[-1] != self.hidden_size:
            raise ValueError(
                f"visual_embeds last dimension should be {self.hidden_size}, "
                f"got {visual_embeds.shape[-1]}"
            )

        visual_embeds = visual_embeds.unsqueeze(0)
        query_context = query_context.to(visual_embeds)
        pooled = self.vis_detail_pooler(visual_embeds, query_context)
        return pooled[:, 0]

    def _prepare_text_tokens(
        self,
        hidden_states,
        inputs_embeds,
        vis_anchor=None,
        batch_idx=None,
    ):
        hidden_context = self._build_text_context(hidden_states)
        return self._merge_token_and_hidden(
            inputs_embeds,
            hidden_context,
            vis_anchor=vis_anchor,
            batch_idx=batch_idx,
        )

    def _uniform_visual_selection(self, vis_local_ids, keep_num):
        return torch.linspace(
            0,
            vis_local_ids.numel() - 1,
            steps=keep_num,
            device=vis_local_ids.device,
        ).round().long()

    def _entropy_adaptive_budget(self, scores, num_vis_tokens, max_budget=None):
        if max_budget is None:
            max_budget = self.vis_select_tokens
        max_budget = min(max_budget, num_vis_tokens)
        min_budget = min(self.min_vis_select_tokens, max_budget)
        if max_budget <= min_budget:
            return max_budget

        scores = scores.float()
        scores = scores - scores.max()
        probs = torch.softmax(scores, dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-8))).sum()
        effective_tokens = torch.exp(entropy)
        keep_num = int(torch.ceil(self.vis_entropy_alpha * effective_tokens).item())
        keep_num = max(min_budget, keep_num)
        keep_num = min(max_budget, keep_num)
        return keep_num

    def _score_visual_ids(
        self,
        image_mask,
        text_attn_vis,
        vis_attn_scores,
        query_token_mask,
        batch_idx,
        vis_ids,
    ):
        if vis_ids.numel() == 0:
            return None
        if vis_attn_scores is not None:
            return vis_attn_scores[batch_idx, vis_ids].float()
        if text_attn_vis is None:
            return None

        seq_ids = torch.arange(image_mask.shape[1], device=image_mask.device)
        if query_token_mask is not None:
            text_ids = seq_ids[query_token_mask[batch_idx] & ~image_mask[batch_idx]]
        else:
            text_ids = seq_ids[~image_mask[batch_idx]]
        if text_ids.numel() == 0:
            return None
        query_window = min(self.vis_query_window, text_ids.numel())
        query_ids = text_ids[-query_window:]
        # image_mask is shifted to align with next-token embeddings, while
        # text_attn_vis indexes the original base-model sequence.
        attn_query_ids = (query_ids + 1).clamp(max=text_attn_vis.shape[-2] - 1)
        attn_vis_ids = (vis_ids + 1).clamp(max=text_attn_vis.shape[-1] - 1)
        scores = text_attn_vis[batch_idx][attn_query_ids][:, attn_vis_ids]
        return scores.mean(dim=0)

    def _select_global_visual_tokens(
        self, image_mask, text_attn_vis, vis_attn_scores, query_token_mask, batch_idx
    ):
        cur_img_msk = image_mask[batch_idx]
        vis_ids = torch.where(cur_img_msk)[0]
        num_vis_tokens = vis_ids.numel()
        if self.max_total_vis_select_tokens is not None:
            max_budget = self.max_total_vis_select_tokens
        else:
            max_budget = self.vis_select_tokens
        max_budget = min(max_budget, num_vis_tokens)
        if num_vis_tokens <= max_budget:
            return cur_img_msk

        keep_msk = torch.zeros_like(cur_img_msk)
        if max_budget <= 0:
            return keep_msk

        scores = self._score_visual_ids(
            image_mask, text_attn_vis, vis_attn_scores, query_token_mask, batch_idx, vis_ids
        )
        if scores is None:
            selected = self._uniform_visual_selection(vis_ids, max_budget)
        else:
            keep_num = self._entropy_adaptive_budget(
                scores, num_vis_tokens, max_budget=max_budget
            )
            selected = torch.topk(scores, keep_num, dim=-1).indices
            selected = torch.sort(selected).values

        keep_msk[vis_ids[selected]] = True
        return keep_msk

    def _select_visual_tokens(
        self,
        image_mask,
        text_attn_vis,
        vis_attn_scores,
        query_token_mask,
        batch_idx,
        img_id_start,
        img_id_end,
    ):
        """
        Select task-related vision tokens. When text-to-vision attention is
        available, use attention entropy to adapt the number of selected tokens;
        otherwise fall back to uniform sampling with the max budget.
        """
        cur_img_msk = image_mask[batch_idx, img_id_start:img_id_end]
        vis_local_ids = torch.where(cur_img_msk)[0]
        num_vis_tokens = vis_local_ids.numel()
        if num_vis_tokens <= self.vis_select_tokens:
            return cur_img_msk

        max_budget = min(self.vis_select_tokens, num_vis_tokens)
        keep_local = torch.zeros_like(cur_img_msk)

        vis_ids = img_id_start + vis_local_ids
        scores = self._score_visual_ids(
            image_mask, text_attn_vis, vis_attn_scores, query_token_mask, batch_idx, vis_ids
        )
        if scores is None:
            selected = self._uniform_visual_selection(vis_local_ids, max_budget)
        else:
            keep_num = self._entropy_adaptive_budget(scores, num_vis_tokens)
            selected = torch.topk(scores, keep_num, dim=-1).indices
            selected = torch.sort(selected).values

        keep_local[vis_local_ids[selected]] = True
        return keep_local

    def select_vis_tokens_by_text_attn(self, text_attn_vis, position_ids):
        # text_attn_vis: [bsz, num_heads, seq_len]
        # position_ids: [bsz, seq_len]
        bsz, num_heads, seq_len = text_attn_vis.shape
        topk = min(self.vis_select_tokens, seq_len)
        text_attn_vis = text_attn_vis.mean(dim=1)  # [bsz, seq_len]
        topk_values, topk_indices = torch.topk(text_attn_vis, k=topk, dim=-1)
        mask = torch.zeros_like(text_attn_vis).bool()
        mask.scatter_(dim=-1, index=topk_indices, value=True)
        selected_position_ids = []
        for b in range(bsz):
            selected_position_ids.append(position_ids[b][mask[b]])
        return selected_position_ids

    def forward(
        self,
        hidden_states,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        std=None,
        image_mask=None,
        text_attn_vis=None,
        vis_attn_scores=None,
        query_token_mask=None,
        vis_anchor=None,
    ):
        batch_size, seq_length, _ = hidden_states.shape
        model_device = self.embed_tokens.weight.device
        model_dtype = self.embed_tokens.weight.dtype
        hidden_states = hidden_states.to(device=model_device, dtype=model_dtype)
        if input_ids is not None:
            input_ids = input_ids.to(model_device)
        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds.to(device=model_device, dtype=model_dtype)
        if attention_mask is not None:
            attention_mask = attention_mask.to(model_device)
        if position_ids is not None:
            position_ids = position_ids.to(model_device)
        if image_mask is not None:
            image_mask = image_mask.to(device=model_device, dtype=torch.bool)
        if vis_attn_scores is not None:
            vis_attn_scores = vis_attn_scores.to(model_device)
        if query_token_mask is not None:
            query_token_mask = query_token_mask.to(model_device)

        seq_length_with_past = seq_length
        past_key_values_length = 0
        past_key_values_real_length = 0

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )
        if inputs_embeds is None:
            with torch.no_grad():
                inputs_embeds = self.embed_tokens(input_ids)

        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]
            if len(past_key_values[0]) == 2:
                past_key_values_real_length = past_key_values_length
            elif len(past_key_values[0]) == 3:
                past_key_values_real_length = past_key_values[0][2]
                # past_key_values_real_length = past_key_values_length  # TODO
            else:
                raise NotImplementedError
            seq_length_with_past += past_key_values_length

        if position_ids is None:
            device = (
                hidden_states.device
                if hidden_states is not None
                else inputs_embeds.device
            )
            position_ids = torch.arange(
                past_key_values_real_length,
                seq_length + past_key_values_real_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length_with_past),
                dtype=torch.bool,
                device=hidden_states.device,
            )
        base_attention_mask = attention_mask

        if image_mask is not None and past_key_values is None:
            image_mask = self._align_image_mask(image_mask, seq_length)
            text_attn_vis = self._prepare_text_attn_vis(
                text_attn_vis,
                batch_size,
                seq_length,
                hidden_states.device,
            )
            vis_attn_scores = self._prepare_vis_attn_scores(
                vis_attn_scores,
                batch_size,
                seq_length,
                hidden_states.device,
            )
            query_token_mask = self._prepare_query_token_mask(
                query_token_mask,
                batch_size,
                seq_length,
                hidden_states.device,
            )
            ends = torch.cat(
                [image_mask[:, :-1] & ~image_mask[:, 1:], image_mask[:, -1:]], dim=1
            )
            last_img_ids = [torch.where(ends[b])[0] for b in range(ends.shape[0])]

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask,
            (batch_size, seq_length),
            hidden_states,
            past_key_values_length,
        )

        inputs_embeds = inputs_embeds.to(hidden_states)
        vis_anchor = self._prepare_vis_anchor(vis_anchor, batch_size, hidden_states)

        trans_mat = None
        if image_mask is not None and past_key_values is None:
            new_hidden_states = []
            new_position_ids = []
            new_trans_mat = []
            new_attention_masks = []
            bsz = len(last_img_ids)
            if bsz != 1:
                raise NotImplementedError("Only support batch size 1")
            num_ids = len(last_img_ids[0])
            for b in range(bsz):
                img_id_start = 0
                h_s = []
                p_i = []
                a_m = []
                global_keep_img_msk = self._select_global_visual_tokens(
                    image_mask,
                    text_attn_vis,
                    vis_attn_scores,
                    query_token_mask,
                    b,
                )
                eye_m = torch.eye(
                    seq_length,
                    dtype=hidden_states.dtype,
                    device=hidden_states.device,
                )
                t_m = []
                self.last_img_hidden = torch.zeros_like(hidden_states[0, :1, ...])
                self.last_vis_detail = None
                pool_query = self._build_visual_pool_query(
                    hidden_states, image_mask, query_token_mask, b
                )
                pooled_detail = None
                detail_pos = None
                selected_vis_ids = torch.where(global_keep_img_msk)[0]
                if selected_vis_ids.numel() > 0:
                    pooled_detail = self._pool_selected_visual_tokens(
                        inputs_embeds[b, selected_vis_ids], pool_query
                    )
                    if pooled_detail is not None:
                        all_vis_ids = torch.where(image_mask[b])[0]
                        detail_pos = int(all_vis_ids[-1].item())
                        self.last_vis_detail = pooled_detail

                for idx in range(num_ids):
                    img_id_end = last_img_ids[b][idx] + 1
                    cur_img_msk = image_mask[b, img_id_start:img_id_end]

                    text_ids = torch.where(~cur_img_msk)[0] + img_id_start
                    if text_ids.numel() > 0:
                        h_s.append(
                            self._prepare_text_tokens(
                                hidden_states[b, text_ids],
                                inputs_embeds[b, text_ids],
                                vis_anchor=vis_anchor,
                                batch_idx=b,
                            )
                        )
                        p_i.append(position_ids[b, text_ids])
                        a_m.append(base_attention_mask[b, text_ids])
                        t_m.append(eye_m[text_ids, :])

                    if (
                        pooled_detail is not None
                        and detail_pos is not None
                        and img_id_start <= detail_pos < img_id_end
                    ):
                        h_s.append(pooled_detail.to(hidden_states))
                        p_i.append(position_ids[b, detail_pos : detail_pos + 1])
                        a_m.append(base_attention_mask[b, detail_pos : detail_pos + 1])
                        t_m.append(eye_m[detail_pos : detail_pos + 1, :])

                    img_id_start = img_id_end

                rst_ids = torch.arange(
                    img_id_start, seq_length, device=hidden_states.device
                )
                if rst_ids.numel() > 0:
                    h_s.append(
                        self._prepare_text_tokens(
                            hidden_states[b, rst_ids],
                            inputs_embeds[b, rst_ids],
                            vis_anchor=vis_anchor,
                            batch_idx=b,
                        )
                    )
                    p_i.append(position_ids[b, rst_ids])
                    a_m.append(base_attention_mask[b, rst_ids])
                    t_m.append(eye_m[rst_ids, :])

                if len(h_s) == 0:
                    fallback_id = seq_length - 1
                    fallback_detail = self.last_vis_detail
                    if fallback_detail is None:
                        fallback_detail = torch.zeros(
                            1,
                            self.hidden_size,
                            dtype=hidden_states.dtype,
                            device=hidden_states.device,
                        )
                    h_s.append(fallback_detail.to(hidden_states))
                    p_i.append(position_ids[b, fallback_id : fallback_id + 1])
                    a_m.append(base_attention_mask[b, fallback_id : fallback_id + 1])
                    t_m.append(eye_m[fallback_id : fallback_id + 1, :])

                h_s = torch.cat(h_s, dim=0).unsqueeze(0)
                p_i = torch.cat(p_i, dim=0).unsqueeze(0)
                a_m = torch.cat(a_m, dim=0).unsqueeze(0)
                t_m = torch.cat(t_m, dim=0).unsqueeze(0)
                new_hidden_states.append(h_s)
                new_position_ids.append(p_i)
                new_attention_masks.append(a_m)
                new_trans_mat.append(t_m)

            hidden_states = torch.cat(new_hidden_states, dim=0)
            position_ids = torch.cat(new_position_ids, dim=0)
            base_attention_mask = torch.cat(new_attention_masks, dim=0)
            trans_mat = torch.cat(new_trans_mat, dim=0)

            attention_mask = self._prepare_decoder_attention_mask(
                base_attention_mask,
                hidden_states.shape[:2],
                hidden_states,
                0,
            )
        else:
            if past_key_values is None:
                self.last_img_hidden = torch.zeros_like(hidden_states[0, :1, ...])
                self.last_vis_detail = None
            hidden_states = self._prepare_text_tokens(
                hidden_states,
                inputs_embeds,
                vis_anchor=vis_anchor,
            )

        all_hidden_states = () if output_hidden_states else None
        next_decoder_cache = () if use_cache else None

        for idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                attentions = layer_outputs[1]
            else:
                attentions = None

            if use_cache:
                next_decoder_cache += (layer_outputs[2 if output_attentions else 1],)

        if trans_mat is not None:
            hidden_states = torch.einsum(
                "bn...,bnm->bm...", hidden_states, trans_mat.to(hidden_states)
            )
            if attentions is not None:
                attentions = torch.einsum(
                    "bhn...,bnm->bhm...", attentions, trans_mat.to(attentions)
                )
                attentions = torch.einsum(
                    "bh...n,bnm->bh...m", attentions, trans_mat.to(attentions)
                )

        if use_cache:
            return hidden_states, next_decoder_cache

        if output_attentions:
            return hidden_states, attentions

        return hidden_states

    def reset_kv(self):
        self.stable_kv = None

    @torch.no_grad()
    def topK_genrate(
        self,
        hidden_states,
        input_ids,
        head,
        logits_processor,
        inputs_embeds=None,
        embed_weights=None,
        image_mask=None,
        text_attn_vis=None,
        vis_attn_scores=None,
        query_token_mask=None,
        vis_anchor=None,
    ):

        input_ids = input_ids.to(hidden_states.device)
        total_tokens = self.total_tokens
        depth = self.depth
        top_k = self.top_k

        sample_token = input_ids[:, -1]

        scores_list = []
        parents_list = []
        ss_token = []

        if inputs_embeds is not None:
            inputs_embeds = inputs_embeds.clone()
            if inputs_embeds.shape[-2] >= input_ids.shape[-1]:
                raise ValueError(
                    "inputs_embeds length must be less than input_ids length"
                )
            if embed_weights is not None:
                if embed_weights.dim() != 3 or embed_weights.shape[-1] != 1:
                    raise ValueError(
                        "embed_weights should be a 3D tensor with shape (vocab_size, hidden_size, 1)"
                    )
                inputs_embeds[
                    : embed_weights.shape[0], : embed_weights.shape[1]
                ] *= embed_weights
            inputs_embeds = inputs_embeds.to(input_ids.device)
            new_embeds = self.embed_tokens(input_ids[:, inputs_embeds.shape[-2] :])
            inputs_embeds = torch.cat((inputs_embeds[:, 1:, :], new_embeds), dim=-2)

        input_ids = input_ids[:, 1:]
        input_ids = input_ids.to(hidden_states.device)

        len_posi = input_ids.shape[1]
        self.reset()

        if hasattr(self, "stable_kv") and self.stable_kv is not None:
            out_hidden, past_key_values = self(
                hidden_states,
                input_ids=input_ids[:, -hidden_states.shape[1] :],
                past_key_values=self.stable_kv,
                use_cache=True,
                image_mask=image_mask,
                text_attn_vis=text_attn_vis,
                vis_attn_scores=vis_attn_scores,
                query_token_mask=query_token_mask,
                vis_anchor=vis_anchor,
            )
        else:
            if inputs_embeds is not None:
                input_ids = None
            out_hidden, past_key_values = self(
                hidden_states,
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                use_cache=True,
                image_mask=image_mask,
                text_attn_vis=text_attn_vis,
                vis_attn_scores=vis_attn_scores,
                query_token_mask=query_token_mask,
                vis_anchor=vis_anchor,
            )
        self.stable_kv = past_key_values
        last_hidden = out_hidden[:, -1]

        last_headout = head(last_hidden)

        last_p = self.logsoftmax(last_headout)
        top = torch.topk(last_p, top_k, dim=-1)
        topk_index, topk_p = top.indices, top.values
        scores = topk_p[0]
        scores_list.append(scores[None])
        parents_list.append(torch.zeros(1, dtype=torch.long, device=scores.device))
        ss_token.append(topk_index)
        input_ids = topk_index
        input_hidden = last_hidden[None].repeat(1, top_k, 1)
        tree_mask = self.tree_mask_init
        topk_cs_index = torch.arange(top_k, device=self.embed_tokens.weight.device)

        # 4
        for i in range(depth):
            self.tree_mask = tree_mask
            position_ids = len_posi + self.position_ids
            out_hidden, past_key_values = self(
                input_hidden,
                input_ids=input_ids,
                past_key_values=past_key_values,
                position_ids=position_ids,
                use_cache=True,
                image_mask=image_mask,
                text_attn_vis=text_attn_vis,
                vis_attn_scores=vis_attn_scores,
                query_token_mask=query_token_mask,
                vis_anchor=vis_anchor,
            )
            len_posi += 1

            bias1 = top_k if i > 0 else 0
            bias2 = max(0, i - 1)
            bias = 1 + top_k**2 * bias2 + bias1
            parents = topk_cs_index + bias
            parents_list.append(parents)

            last_headout = head(out_hidden[0])
            last_p = self.logsoftmax(last_headout)

            top = torch.topk(last_p, top_k, dim=-1)
            topk_index, topk_p = top.indices, top.values

            cu_scores = topk_p + scores[:, None]

            topk_cs = torch.topk(cu_scores.view(-1), top_k, dim=-1)
            topk_cs_index, topk_cs_p = topk_cs.indices, topk_cs.values
            scores = topk_cs_p

            out_ids = topk_cs_index // top_k
            input_hidden = out_hidden[:, out_ids]
            input_ids = topk_index.view(-1)[topk_cs_index][None]

            ss_token.append(topk_index)
            scores_list.append(cu_scores)
            tree_mask = torch.cat(
                (tree_mask[:, :, out_ids], self.tree_mask_init), dim=3
            )

        scores_list = torch.cat(scores_list, dim=0).view(-1)
        ss_token_list = torch.cat(ss_token, dim=0).view(-1)
        top_scores = torch.topk(scores_list, total_tokens, dim=-1)
        top_scores_index = top_scores.indices
        top_scores_index = torch.sort(top_scores_index).values

        draft_tokens = ss_token_list[top_scores_index]
        draft_tokens = torch.cat((sample_token, draft_tokens), dim=0)

        draft_parents = torch.cat(parents_list, dim=0)[top_scores_index // top_k].long()
        mask_index = torch.searchsorted(
            top_scores_index, draft_parents - 1, right=False
        )
        mask_index[draft_parents == 0] = -1
        mask_index = mask_index + 1
        mask_index_list = mask_index.tolist()
        tree_mask = torch.eye(total_tokens + 1).bool()
        tree_mask[:, 0] = True
        for i in range(total_tokens):
            tree_mask[i + 1].add_(tree_mask[mask_index_list[i]])

        tree_position_ids = torch.sum(tree_mask, dim=1) - 1

        tree_mask = tree_mask.float()[None, None]
        draft_tokens = draft_tokens[None]

        del parents_list, scores_list, ss_token, ss_token_list, draft_parents

        max_depth = torch.max(tree_position_ids) + 1
        noleaf_index = torch.unique(mask_index).tolist()
        noleaf_num = len(noleaf_index) - 1
        leaf_num = total_tokens - noleaf_num

        retrieve_indices = torch.zeros(leaf_num, max_depth.item(), dtype=torch.long) - 1
        retrieve_indices = retrieve_indices.tolist()

        rid = 0
        position_ids_list = tree_position_ids.tolist()

        for i in range(total_tokens + 1):
            if i not in noleaf_index:
                cid = i
                depth = position_ids_list[i]
                for j in reversed(range(depth + 1)):
                    retrieve_indices[rid][j] = cid
                    cid = mask_index_list[cid - 1]
                rid += 1

        if logits_processor is not None:
            maxitem = total_tokens + 5

            def custom_sort(lst):
                sort_keys = []
                for i in range(len(lst)):
                    sort_keys.append(lst[i] if lst[i] >= 0 else maxitem)
                return sort_keys

            retrieve_indices = sorted(retrieve_indices, key=custom_sort)

        retrieve_indices = torch.tensor(retrieve_indices, dtype=torch.long)
        del (
            mask_index,
            mask_index_list,
            noleaf_index,
            noleaf_num,
            leaf_num,
            max_depth,
            rid,
        )
        tree_position_ids = tree_position_ids.to(hidden_states.device)

        return draft_tokens, retrieve_indices, tree_mask, tree_position_ids


if __name__ == "__main__":
    config = EConfig.from_pretrained("config.json")
    model = Model(config, load_emb=False)
    print(model)
