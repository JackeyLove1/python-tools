'''
1, 旋转位置编码: RoPE (使用旋转矩阵实现的绝对位置编码，可以起到相对位置编码的效果)
2, 多头注意力: LlamaAttention (用于融合不同token之间的信息)
3, 前馈网络: LlamaMLP (用于逐位置将多头注意力融合后的信息进行高维映射变换)
4, 层归一化: LlamaRMSNorm (用于稳定输入，相当于保持每个词向量的方向不变，但对模长标准化。)
5, Llama解码层: LlamaDecoderLayer (同时具备信息融合，信息转换功能的基本结构单元)
6, Llama解码器: LlamaModel (多个解码层的堆叠)
7，Llama语言模型: LlamaForCausalLM (解码器加上语言模型head，可用于文本生成)
8，Llama分类模型: LlamaForSequenceClassification (解码器加上分类head，可用于文本分类)
'''

import math
from typing import List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast, \
    SequenceClassifierOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, \
    replace_return_docstrings

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LLAMA_INPUTS_DOCSTRING, LLAMA_START_DOCSTRING

from data import *

logger = logging.get_logger("llama")

config = LlamaConfig(
    vocab_size=len(vocab),
    hidden_size=512,
    intermediate_size=2752,
    num_hidden_layers=8,
    num_attention_heads=16,
    num_key_value_heads=4,
    rope_scaling = None,
    hidden_act='silu',
    max_position_embeddings=128,
    initializer_range=0.02,
    rms_norm_eps=1e-06,
    use_cache=True,
    pad_token_id=0,
    bos_token_id=1,
    eos_token_id=2,
    tie_word_embeddings=False,
    pretraining_tp = 1,
    max_new_tokens = 100
)

# RoPE
class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        # set sin and cos cache
        self._set_cos_sin_cache(self.max_position_embeddings, device=device, dtype=self.inv_freq.dtype)

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # torch.outer
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :].to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :].to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


def rotate_half(x):
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotray_pos_emb(q, k, cos, sin, position_ids):
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


# x = torch.randn(1, 8, 4, 2)
# rope = LlamaRotaryEmbedding(dim=8)
# cos, sin = rope.forward(x, seq_len=4)
# print(cos.shape)
# print(cos)

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads,n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

class LlamaAttention(nn.Module):
    def __init__(self, config : LlamaConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size # 512
        self.num_heads = config.num_attention_heads # 16
        self.head_dim = self.head_size // self.num_heads # 512 / 16 = 32
        self.num_key_value_heads = config.num_key_value_heads # 4
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads # 16 / 4 = 4
        self.max_position_embeddings = config.max_position_embeddings # 128

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        self._init_rope()

    def _init_rope(self):
        self.rotary_emb = LlamaRotaryEmbedding(self.head_dim, max_position_embeddings=self.max_position_embeddings)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        pass