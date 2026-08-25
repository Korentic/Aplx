"""
================================================================================
APLX_LLM  v3.1.0  —  Lightweight Decoder-Only Transformer (100M)
Approx. 100 Million Parameters | 2k Token Context Window
================================================================================

Optimized for CPU-efficient inference & training on local machines.
Perfect for conversation, reasoning, and code generation tasks.
Production-ready PyTorch implementation of a modern LLaMA-style transformer.

New in v3.1 (100M Optimized):
  - Scaled down from 1B to 100M parameters for CPU/mobile efficiency
  - Optimized architecture: 768 hidden dim, 20 layers, GQA with 4 KV heads
  - 2K token context window (vs 500K) for practical efficiency
  - Reduced sliding window (256 tokens) for faster inference
  - Better conversation + thinking performance with lower memory footprint
  - Gradient checkpointing enabled by default
  - INT8 quantization ready for even smaller models
  
Features:
  - ChatML chat template (system / user / assistant roles)
  - BPE tokenizer with byte-fallback encoding
  - Grouped Query Attention (GQA) for inference speedup
  - KV-Cache for efficient autoregressive generation
  - Sliding Window Attention (256 token window)
  - RMSNorm pre-normalization
  - Rotary Position Embeddings (RoPE)
  - SwiGLU feed-forward networks
  - Gradient Checkpointing (enabled by default)
  - Mixed-Precision (AMP) training support
  - Streaming text generation
  - Multi-turn conversation with history
  - Nucleus sampling + greedy decoding
"""

from __future__ import annotations

import sys
try:
    if callable(getattr(sys.stdout, 'reconfigure', None)):
        sys.stdout.reconfigure(encoding='utf-8')
    if callable(getattr(sys.stderr, 'reconfigure', None)):
        sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

import gc
import importlib
import importlib.util
import json
import math
import os
import re
import struct
import time
import hashlib
import random
import argparse
import copy
from abc import ABC, abstractmethod
from collections import OrderedDict, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from pathlib import Path
from typing import (
    Any, Callable, Dict, Generator, Iterator, List,
    Optional, Sequence, Set, Tuple, Union, cast
)

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint as gradient_checkpoint
from torch.utils.data import DataLoader, Dataset, DistributedSampler


# ==============================================================================
#  §0.  UTILITY — DEVICE SELECTION
# ==============================================================================

def get_default_device(prefer_gpu: bool = True) -> torch.device:
    """Select the best available device: CUDA > MPS > CPU."""
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model_config_for_device(
    device: torch.device,
    base_config: Optional["ModelConfig"] = None,
) -> "ModelConfig":
    """Reduce the model size automatically for small GPUs so training stays viable."""
    cfg = base_config or ModelConfig()
    if device.type != "cuda":
        return cfg

    try:
        vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    except Exception:
        return cfg

    if vram_gb <= 4.0:
        return ModelConfig(
            vocab_size=min(cfg.vocab_size, 2000),
            dim=min(cfg.dim, 128),
            n_layers=min(cfg.n_layers, 2),
            n_heads=min(cfg.n_heads, 4),
            n_kv_heads=min(cfg.n_kv_heads, 4),
            max_seq_len=min(cfg.max_seq_len, 1024),
            max_batch_size=min(cfg.max_batch_size, 2),
            training_seq_len=min(cfg.training_seq_len, 256),
            sliding_window_size=min(cfg.sliding_window_size, 256),
            inference_cache_len=min(cfg.inference_cache_len, 512),
            use_sliding_window=cfg.use_sliding_window,
            rope_theta=cfg.rope_theta,
            rope_scaling_factor=cfg.rope_scaling_factor,
            activation=cfg.activation,
            use_gated_ffn=cfg.use_gated_ffn,
            dropout=max(cfg.dropout, 0.1),
            attention_dropout=cfg.attention_dropout,
            embedding_dropout=cfg.embedding_dropout,
            use_gradient_checkpointing=True,
            use_flash_attention=False,
            tie_word_embeddings=cfg.tie_word_embeddings,
            model_name=cfg.model_name,
            model_version=cfg.model_version,
        )

    if vram_gb <= 8.0:
        return ModelConfig(
            vocab_size=min(cfg.vocab_size, 4000),
            dim=min(cfg.dim, 256),
            n_layers=min(cfg.n_layers, 4),
            n_heads=min(cfg.n_heads, 8),
            n_kv_heads=min(cfg.n_kv_heads, 8),
            max_seq_len=min(cfg.max_seq_len, 2048),
            max_batch_size=min(cfg.max_batch_size, 4),
            training_seq_len=min(cfg.training_seq_len, 512),
            sliding_window_size=min(cfg.sliding_window_size, 512),
            inference_cache_len=min(cfg.inference_cache_len, 1024),
            use_sliding_window=cfg.use_sliding_window,
            rope_theta=cfg.rope_theta,
            rope_scaling_factor=cfg.rope_scaling_factor,
            activation=cfg.activation,
            use_gated_ffn=cfg.use_gated_ffn,
            dropout=cfg.dropout,
            attention_dropout=cfg.attention_dropout,
            embedding_dropout=cfg.embedding_dropout,
            use_gradient_checkpointing=True,
            use_flash_attention=False,
            tie_word_embeddings=cfg.tie_word_embeddings,
            model_name=cfg.model_name,
            model_version=cfg.model_version,
        )

    return cfg



class NormType(Enum):
    RMSNORM   = auto()
    LAYERNORM = auto()


class PositionEmbeddingType(Enum):
    ROPE    = auto()
    ALIBI   = auto()
    LEARNED = auto()


class ActivationType(Enum):
    SILU = auto()
    GELU = auto()
    RELU = auto()
    SWISH = auto()


def build_model_config_for_device(
    device: torch.device,
    base_config: Optional["ModelConfig"] = None,
) -> "ModelConfig":
    """Reduce the model size automatically for small GPUs so training stays viable."""
    cfg = base_config or ModelConfig()
    if device.type != "cuda":
        return cfg

    try:
        vram_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
    except Exception:
        return cfg

    if vram_gb <= 4.0:
        return ModelConfig(
            vocab_size=min(cfg.vocab_size, 2000),
            dim=min(cfg.dim, 128),
            n_layers=min(cfg.n_layers, 2),
            n_heads=min(cfg.n_heads, 4),
            n_kv_heads=min(cfg.n_kv_heads, 4),
            max_seq_len=min(cfg.max_seq_len, 1024),
            max_batch_size=min(cfg.max_batch_size, 2),
            training_seq_len=min(cfg.training_seq_len, 256),
            sliding_window_size=min(cfg.sliding_window_size, 256),
            inference_cache_len=min(cfg.inference_cache_len, 512),
            use_sliding_window=cfg.use_sliding_window,
            rope_theta=cfg.rope_theta,
            rope_scaling_factor=cfg.rope_scaling_factor,
            activation=cfg.activation,
            use_gated_ffn=cfg.use_gated_ffn,
            dropout=max(cfg.dropout, 0.1),
            attention_dropout=cfg.attention_dropout,
            embedding_dropout=cfg.embedding_dropout,
            use_gradient_checkpointing=True,
            use_flash_attention=False,
            tie_word_embeddings=cfg.tie_word_embeddings,
            model_name=cfg.model_name,
            model_version=cfg.model_version,
        )

    if vram_gb <= 8.0:
        return ModelConfig(
            vocab_size=min(cfg.vocab_size, 4000),
            dim=min(cfg.dim, 256),
            n_layers=min(cfg.n_layers, 4),
            n_heads=min(cfg.n_heads, 8),
            n_kv_heads=min(cfg.n_kv_heads, 8),
            max_seq_len=min(cfg.max_seq_len, 2048),
            max_batch_size=min(cfg.max_batch_size, 4),
            training_seq_len=min(cfg.training_seq_len, 512),
            sliding_window_size=min(cfg.sliding_window_size, 512),
            inference_cache_len=min(cfg.inference_cache_len, 1024),
            use_sliding_window=cfg.use_sliding_window,
            rope_theta=cfg.rope_theta,
            rope_scaling_factor=cfg.rope_scaling_factor,
            activation=cfg.activation,
            use_gated_ffn=cfg.use_gated_ffn,
            dropout=cfg.dropout,
            attention_dropout=cfg.attention_dropout,
            embedding_dropout=cfg.embedding_dropout,
            use_gradient_checkpointing=True,
            use_flash_attention=False,
            tie_word_embeddings=cfg.tie_word_embeddings,
            model_name=cfg.model_name,
            model_version=cfg.model_version,
        )

    return cfg


@dataclass
class ModelConfig:
    """
    Complete configuration for the APLX_LLM model.

    Defaults produce a ~100M parameter model optimized for CPU inference.
    Suitable for conversation, reasoning, and code generation.
    """
    # --- Architecture (optimized for 100M params) ---
    vocab_size: int            = 32_000
    dim: int                   = 768         # 768 hidden dim for balance
    n_layers: int              = 20          # 20 layers for depth
    n_heads: int               = 12          # 768 / 12 = 64 head dim (efficient)
    n_kv_heads: int            = 4           # Grouped Query Attention (4:12 ratio)
    multiple_of: int           = 256         # SwiGLU hidden-dim alignment
    ffn_dim_multiplier: float  = 1.0
    max_seq_len: int           = 2048        # Reduced from 4096 for efficiency
    max_batch_size: int        = 4

    # --- Context & Sliding Window ---
    training_seq_len: int      = 256         # Reduced for faster training
    sliding_window_size: int   = 256         # 256-token window attention
    inference_cache_len: int   = 512
    use_sliding_window: bool   = True

    # --- Normalization ---
    norm_type: NormType             = NormType.RMSNORM
    norm_eps: float                 = 1e-6

    # --- Positional Encoding ---
    pos_emb_type: PositionEmbeddingType = PositionEmbeddingType.ROPE
    rope_theta: float               = 100_000.0  # Reduced for smaller context
    rope_scaling_factor: float      = 1.0

    # --- Activation ---
    activation: ActivationType      = ActivationType.SILU
    use_gated_ffn: bool             = True   # SwiGLU

    # --- Regularization ---
    dropout: float          = 0.1         # Light dropout for better generalization
    attention_dropout: float = 0.1
    embedding_dropout: float = 0.0

    # --- Initialization ---
    initializer_range: float = 0.02
    use_scaled_init: bool    = True

    # --- Training & Inference ---
    tie_word_embeddings: bool           = False
    use_gradient_checkpointing: bool    = True   # Always ON for CPU
    use_flash_attention: bool           = False  # Not needed on CPU

    # --- Metadata ---
    model_name: str    = "APLX_LLM_100M"
    model_version: str = "3.1.0"

    def __post_init__(self):
        assert self.dim % self.n_heads == 0, (
            f"dim ({self.dim}) must be divisible by n_heads ({self.n_heads})")
        assert self.n_heads % self.n_kv_heads == 0, (
            f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})")
        assert self.dim > 0 and self.n_layers > 0
        assert 0.0 <= self.dropout <= 1.0

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    @property
    def hidden_dim(self) -> int:
        hidden = int(2 * (4 * self.dim) / 3)
        hidden = int(hidden * self.ffn_dim_multiplier)
        hidden = self.multiple_of * ((hidden + self.multiple_of - 1) // self.multiple_of)
        return hidden

    def to_dict(self) -> dict:
        d = asdict(self)
        d["norm_type"]    = self.norm_type.name
        d["pos_emb_type"] = self.pos_emb_type.name
        d["activation"]   = self.activation.name
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        d = d.copy()
        if isinstance(d.get("norm_type"), str):
            d["norm_type"] = NormType[d["norm_type"]]
        if isinstance(d.get("pos_emb_type"), str):
            d["pos_emb_type"] = PositionEmbeddingType[d["pos_emb_type"]]
        if isinstance(d.get("activation"), str):
            d["activation"] = ActivationType[d["activation"]]
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def save(self, path: Union[str, Path]) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ModelConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def estimate_parameters(self) -> dict:
        embed            = self.vocab_size * self.dim
        attn_per_layer   = (
            self.dim * (self.n_heads * self.head_dim) +
            self.dim * (self.n_kv_heads * self.head_dim) +
            self.dim * (self.n_kv_heads * self.head_dim) +
            (self.n_heads * self.head_dim) * self.dim
        )
        ffn_per_layer = (
            self.dim * self.hidden_dim +
            self.hidden_dim * self.dim +
            self.dim * self.hidden_dim
        ) if self.use_gated_ffn else (
            self.dim * self.hidden_dim +
            self.hidden_dim * self.dim
        )
        norm_per_layer = 2 * self.dim
        layer_total    = attn_per_layer + ffn_per_layer + norm_per_layer
        all_layers     = self.n_layers * layer_total
        final_norm     = self.dim
        lm_head        = 0 if self.tie_word_embeddings else self.dim * self.vocab_size
        total          = embed + all_layers + final_norm + lm_head
        return {
            "embedding": embed, "attention_per_layer": attn_per_layer,
            "ffn_per_layer": ffn_per_layer, "norm_per_layer": norm_per_layer,
            "total_per_layer": layer_total, "all_layers": all_layers,
            "final_norm": final_norm, "lm_head": lm_head, "total": total,
        }


# ==============================================================================
#  §2.  NORMALIZATION LAYERS
# ==============================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (Zhang & Sennrich, 2019)."""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        return (self._norm(x.float()) * self.weight).type_as(x)

    def extra_repr(self) -> str:
        return f"dim={self.weight.shape[0]}, eps={self.eps}"


class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, bias: bool = False):
        super().__init__()
        self.eps    = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias   = nn.Parameter(torch.zeros(dim)) if bias else None

    def forward(self, x: Tensor) -> Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, self.eps)


def build_norm(config: ModelConfig) -> nn.Module:
    if config.norm_type == NormType.RMSNORM:
        return RMSNorm(config.dim, eps=config.norm_eps)
    return LayerNorm(config.dim, eps=config.norm_eps)


# ==============================================================================
#  §3.  ROTARY POSITION EMBEDDINGS (RoPE)
# ==============================================================================

class RotaryEmbedding(nn.Module):
    """
    RoPE with lazy cache expansion and optional NTK-aware scaling.

    The cache starts at min(max_seq_len, 8192) and grows on demand,
    so initialising with a 500k max_seq_len doesn't pre-allocate 500k
    cosine/sine rows upfront.
    """
    def __init__(
        self,
        dim: int,
        max_seq_len: int  = 1_000_000,
        theta: float      = 500_000.0,
        scaling_factor: float = 1.0,
    ):
        super().__init__()
        self.dim            = dim
        self.max_seq_len    = max_seq_len
        self.theta          = theta
        self.scaling_factor = scaling_factor

        effective_theta = theta
        if scaling_factor > 1.0:
            effective_theta = theta * (scaling_factor ** (dim / (dim - 2)))

        inv_freq = 1.0 / (
            effective_theta ** (torch.arange(0, dim, 2).float() / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.cos_cached: Optional[Tensor] = None
        self.sin_cached: Optional[Tensor] = None
        self._cached_len = 0
        self._build_cache(min(max_seq_len, 8192))

    def _build_cache(self, seq_len: int) -> None:
        inv_freq = self.inv_freq
        if inv_freq is None:
            return
        device = inv_freq.device
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        if self.scaling_factor > 1.0:
            t = t / self.scaling_factor
        freqs = torch.outer(t, inv_freq)
        emb   = torch.cat((freqs, freqs), dim=-1)
        cos_val = emb.cos()
        sin_val = emb.sin()
        
        # If buffers don't exist yet, register them; otherwise update directly
        if not hasattr(self, '_buffers') or 'cos_cached' not in self._buffers:
            self.register_buffer("cos_cached", cos_val, persistent=False)
            self.register_buffer("sin_cached", sin_val, persistent=False)
        else:
            # Update existing buffers in place
            self._buffers['cos_cached'] = cos_val
            self._buffers['sin_cached'] = sin_val
        
        self._cached_len = seq_len

    def forward(self, seq_len: int) -> Tuple[Tensor, Tensor]:
        if self.cos_cached is None or self.sin_cached is None:
            self._build_cache(min(self.max_seq_len, 8192))
        if self.cos_cached is None or self.sin_cached is None:
            raise RuntimeError("RoPE cache failed to initialize")
        if seq_len > self.cos_cached.shape[0]:
            new_len = min(max(seq_len, self.cos_cached.shape[0] * 2), self.max_seq_len)
            self._build_cache(new_len)
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def rotate_half(x: Tensor) -> Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: Tensor, k: Tensor, cos: Tensor, sin: Tensor,
    position_ids: Optional[Tensor] = None,
) -> Tuple[Tensor, Tensor]:
    if position_ids is not None:
        cos = cos[position_ids].unsqueeze(1)
        sin = sin[position_ids].unsqueeze(1)
    else:
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


# ==============================================================================
#  §4.  KV-CACHE
# ==============================================================================

class KVCache:
    """Per-layer Key-Value cache for efficient autoregressive generation."""
    def __init__(
        self,
        max_batch_size: int,
        max_seq_len: int,
        n_kv_heads: int,
        head_dim: int,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype   = torch.float32,
    ):
        self.max_batch_size = max_batch_size
        self.max_seq_len    = max_seq_len
        cache_shape = (max_batch_size, n_kv_heads, max_seq_len, head_dim)
        self.cache_k = torch.zeros(cache_shape, device=device, dtype=dtype)
        self.cache_v = torch.zeros(cache_shape, device=device, dtype=dtype)
        self.seq_len = 0

    def update(self, k: Tensor, v: Tensor, start_pos: int) -> Tuple[Tensor, Tensor]:
        bsz, _, new_seq_len, _ = k.shape
        end_pos = start_pos + new_seq_len
        self.cache_k[:bsz, :, start_pos:end_pos, :] = k
        self.cache_v[:bsz, :, start_pos:end_pos, :] = v
        self.seq_len = end_pos
        return self.cache_k[:bsz, :, :end_pos, :], self.cache_v[:bsz, :, :end_pos, :]

    def reset(self) -> None:
        self.cache_k.zero_()
        self.cache_v.zero_()
        self.seq_len = 0

    @property
    def memory_usage_mb(self) -> float:
        total = (self.cache_k.nelement() * self.cache_k.element_size() +
                 self.cache_v.nelement() * self.cache_v.element_size())
        return total / (1024 * 1024)


class KVCacheManager:
    """Manages per-layer KV caches for the entire model."""
    def __init__(
        self,
        config: ModelConfig,
        device: torch.device,
        dtype: torch.dtype,
        cache_len_override: Optional[int] = None,
    ):
        cache_len = cache_len_override or config.inference_cache_len
        self.cache_len = cache_len
        self.caches: List[KVCache] = [
            KVCache(
                max_batch_size=config.max_batch_size,
                max_seq_len=cache_len,
                n_kv_heads=config.n_kv_heads,
                head_dim=config.head_dim,
                device=device,
                dtype=dtype,
            )
            for _ in range(config.n_layers)
        ]

    def __getitem__(self, layer_idx: int) -> KVCache:
        return self.caches[layer_idx]

    def reset_all(self) -> None:
        for c in self.caches:
            c.reset()

    @property
    def total_memory_mb(self) -> float:
        return sum(c.memory_usage_mb for c in self.caches)


# ==============================================================================
#  §5.  ATTENTION MECHANISM  (FIXED: vectorised sliding window + mask skip)
# ==============================================================================

class GroupedQueryAttention(nn.Module):
    """
    Multi-Head / Grouped-Query Attention with Sliding Window support.

    Fixes vs v2.0:
      - Sliding-window mask is now built with vectorised torch ops (no Python loop)
      - Mask is NOT passed during single-token KV-cache decode steps (seqlen == 1)
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config     = config
        self.n_heads    = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim   = config.head_dim
        self.n_rep      = self.n_heads // self.n_kv_heads
        self.scale      = self.head_dim ** -0.5
        self.sliding_window = config.sliding_window_size if config.use_sliding_window else 0

        self.wq = nn.Linear(config.dim, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(config.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_heads * self.head_dim, config.dim, bias=False)

        self.attn_dropout  = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self._use_flash = config.use_flash_attention and self._check_flash_available()

    @staticmethod
    def _check_flash_available() -> bool:
        try:
            return importlib.util.find_spec("flash_attn") is not None
        except Exception:
            return False

    def _repeat_kv(self, x: Tensor) -> Tensor:
        if self.n_rep == 1:
            return x
        b, h, s, d = x.shape
        return (
            x[:, :, None, :, :]
             .expand(b, h, self.n_rep, s, d)
             .reshape(b, self.n_heads, s, d)
        )

    def _build_sliding_window_mask(self, seq_len: int, device: torch.device) -> Tensor:
        """
        Vectorised causal + sliding-window mask.
        O(n) memory, built without any Python-level loop.
        """
        # Causal: upper triangle = -inf
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)

        if self.sliding_window > 0 and seq_len > self.sliding_window:
            # Positions more than window_size steps in the past → -inf
            # row i can attend to [i - window + 1 .. i], mask the rest
            rows = torch.arange(seq_len, device=device).unsqueeze(1)  # (S, 1)
            cols = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, S)
            beyond_window = cols < (rows - self.sliding_window + 1)
            mask = mask.masked_fill(beyond_window, float("-inf"))

        return mask

    def _standard_attention(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor]
    ) -> Tensor:
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores + mask
        attn = F.softmax(scores, dim=-1, dtype=torch.float32).type_as(q)
        attn = self.attn_dropout(attn)
        return torch.matmul(attn, v)

    def _flash_attention(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        try:
            flash_attn = importlib.import_module("flash_attn")
            flash_attn_func = flash_attn.flash_attn_func
        except Exception:
            return self._standard_attention(q, k, v, None)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        out = flash_attn_func(
            q, k, v,
            dropout_p=self.config.attention_dropout if self.training else 0.0,
            causal=True,
            window_size=(self.sliding_window, 0) if self.sliding_window > 0 else (-1, -1),
        )
        return out.transpose(1, 2)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        start_pos: int = 0,
    ) -> Tensor:
        bsz, seqlen, _ = x.shape

        q = self.wq(x).view(bsz, seqlen, self.n_heads,    self.head_dim).transpose(1, 2)
        k = self.wk(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(bsz, seqlen, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        if kv_cache is not None:
            k, v = kv_cache.update(k, v, start_pos)

        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        # FIX: skip mask when decoding single token (it's always causal-safe)
        effective_mask = None if seqlen == 1 else mask

        if self._use_flash and effective_mask is None and kv_cache is None:
            output = self._flash_attention(q, k, v)
        else:
            output = self._standard_attention(q, k, v, effective_mask)

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.resid_dropout(self.wo(output))


# ==============================================================================
#  §6.  FEED-FORWARD NETWORKS
# ==============================================================================

class SwiGLUFeedForward(nn.Module):
    """SwiGLU FFN: output = W_down(SiLU(W_gate(x)) * W_up(x))"""
    def __init__(self, config: ModelConfig):
        super().__init__()
        h = config.hidden_dim
        self.w_gate    = nn.Linear(config.dim, h, bias=False)
        self.w_down    = nn.Linear(h, config.dim, bias=False)
        self.w_up      = nn.Linear(config.dim, h, bias=False)
        self.dropout   = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.w_down(F.silu(self.w_gate(x)) * self.w_up(x)))


class StandardFeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        h = config.hidden_dim
        self.w1      = nn.Linear(config.dim, h, bias=False)
        self.w2      = nn.Linear(h, config.dim, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.act     = {
            ActivationType.SILU: nn.SiLU(),
            ActivationType.SWISH: nn.SiLU(),  # Swish is SiLU
            ActivationType.GELU: nn.GELU(),
            ActivationType.RELU: nn.ReLU(),
        }.get(config.activation, nn.SiLU())

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.w2(self.act(self.w1(x))))


def build_ffn(config: ModelConfig) -> nn.Module:
    return SwiGLUFeedForward(config) if config.use_gated_ffn else StandardFeedForward(config)


# ==============================================================================
#  §7.  TRANSFORMER BLOCK
# ==============================================================================

class TransformerBlock(nn.Module):
    """
    Pre-norm decoder block:
        x → RMSNorm → Attention → residual
          → RMSNorm → FFN       → residual
    """
    def __init__(self, layer_id: int, config: ModelConfig):
        super().__init__()
        self.layer_id       = layer_id
        self.attention      = GroupedQueryAttention(config)
        self.feed_forward   = build_ffn(config)
        self.attention_norm = build_norm(config)
        self.ffn_norm       = build_norm(config)

    def forward(
        self,
        x: Tensor,
        cos: Tensor,
        sin: Tensor,
        mask: Optional[Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        start_pos: int = 0,
    ) -> Tensor:
        h   = x + self.attention(self.attention_norm(x), cos, sin, mask, kv_cache, start_pos)
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


# ==============================================================================
#  §8.  MAIN MODEL — AplexLLM
# ==============================================================================

class AplexLLM(nn.Module):
    """APLX_LLM: lightweight decoder-only transformer (~100M params, 2k context, CPU-optimized)."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        self.tok_embeddings    = nn.Embedding(config.vocab_size, config.dim)
        self.embedding_dropout = nn.Dropout(config.embedding_dropout)

        self.rotary_emb = RotaryEmbedding(
            dim=config.head_dim,
            max_seq_len=config.max_seq_len,
            theta=config.rope_theta,
            scaling_factor=config.rope_scaling_factor,
        )

        self.layers = nn.ModuleList([
            TransformerBlock(layer_id=i, config=config)
            for i in range(config.n_layers)
        ])

        self.norm    = build_norm(config)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

        if config.tie_word_embeddings:
            self.lm_head.weight = self.tok_embeddings.weight

        self.apply(self._init_weights)
        if config.use_scaled_init:
            self._apply_scaled_init()

        self._kv_cache_manager: Optional[KVCacheManager] = None

    # ---------- initialisation ----------

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)

    def _apply_scaled_init(self) -> None:
        scale = (2 * self.config.n_layers) ** -0.5
        for layer in self.layers:
            layer = cast(TransformerBlock, layer)
            nn.init.normal_(layer.attention.wo.weight, mean=0.0,
                            std=self.config.initializer_range * scale)
            ffn = layer.feed_forward
            tgt = ffn.w_down if hasattr(ffn, 'w_down') else getattr(ffn, 'w2', None)
            if tgt is not None:
                nn.init.normal_(tgt.weight, mean=0.0,
                                std=self.config.initializer_range * scale)

    # ---------- KV cache ----------

    def allocate_kv_cache(
        self,
        device: torch.device = torch.device("cpu"),
        dtype: torch.dtype   = torch.float32,
    ) -> None:
        self._kv_cache_manager = KVCacheManager(self.config, device, dtype)

    def reset_kv_cache(self) -> None:
        if self._kv_cache_manager:
            self._kv_cache_manager.reset_all()

    def free_kv_cache(self) -> None:
        self._kv_cache_manager = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---------- mask ----------

    def _build_causal_mask(self, seq_len: int, device: torch.device) -> Optional[Tensor]:
        if seq_len <= 1:
            return None
        # Use the first layer's sliding-window settings to build a single shared mask
        attn = cast(GroupedQueryAttention, self.layers[0].attention)
        if attn.sliding_window > 0:
            return attn._build_sliding_window_mask(seq_len, device)
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        return torch.triu(mask, diagonal=1)

    # ---------- forward ----------

    def forward(
        self,
        tokens: Tensor,
        start_pos: int = 0,
        targets: Optional[Tensor] = None,
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        bsz, seqlen = tokens.shape

        h = self.embedding_dropout(self.tok_embeddings(tokens))

        cos, sin = self.rotary_emb(start_pos + seqlen)
        cos = cos[start_pos : start_pos + seqlen]
        sin = sin[start_pos : start_pos + seqlen]

        mask = self._build_causal_mask(seqlen, tokens.device)

        for i, layer in enumerate(self.layers):
            kv_cache = self._kv_cache_manager[i] if self._kv_cache_manager else None
            if self.config.use_gradient_checkpointing and self.training:
                h = gradient_checkpoint(
                    layer, h, cos, sin, mask, kv_cache, start_pos,
                    use_reentrant=False,
                )
            else:
                h = layer(h, cos, sin, mask, kv_cache, start_pos)

        h      = self.norm(h)
        logits = self.lm_head(h)

        if targets is not None:
            # FIX: use ignore_index=-100 so padded positions don't affect loss
            shift_logits  = logits[..., :-1, :].contiguous()
            shift_targets = targets[..., 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_targets.view(-1),
                ignore_index=-100,
            )
            return loss, logits

        return logits

    # ---------- introspection ----------

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def num_parameters_non_embedding(self) -> int:
        ep = self.tok_embeddings.weight.numel()
        if not self.config.tie_word_embeddings:
            ep += self.lm_head.weight.numel()
        return self.num_parameters - ep

    def parameter_summary(self) -> str:
        lines = [f"\n{'='*60}", f"  {self.config.model_name} Parameter Summary", f"{'='*60}"]
        lines.append(f"  Token Embeddings:      {self.tok_embeddings.weight.numel():>15,}")
        for i, layer in enumerate(self.layers):
            lp = sum(p.numel() for p in layer.parameters())
            if i == 0 or i == len(self.layers) - 1:
                lines.append(f"  Layer {i:>2}:              {lp:>15,}")
            elif i == 1:
                lines.append(f"  ... (layers 1-{len(self.layers)-2} identical) ...")
        lines.append(f"  Final Norm:            {sum(p.numel() for p in self.norm.parameters()):>15,}")
        tied_str = " (tied)" if self.config.tie_word_embeddings else ""
        lines.append(f"  LM Head{tied_str}:          {self.lm_head.weight.numel():>15,}")
        lines.append(f"{'─'*60}")
        lines.append(f"  TOTAL:                 {self.num_parameters:>15,}")
        lines.append(f"  (~{self.num_parameters / 1e9:.3f}B parameters)")
        lines.append(f"{'='*60}\n")
        return "\n".join(lines)


# ==============================================================================
#  §9.  LoRA (LOW-RANK ADAPTATION)
# ==============================================================================

class LoRALinear(nn.Module):
    """
    LoRA wrapper for nn.Linear (Hu et al., 2021).
    W_new = W_frozen + (alpha/r) * B @ A
    """
    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int    = 16,
        alpha: float = 32.0,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.in_features  = original_linear.in_features
        self.out_features = original_linear.out_features
        self.rank    = rank
        self.alpha   = alpha
        self.scaling = alpha / rank

        self.weight = original_linear.weight
        self.weight.requires_grad_(False)
        self.bias = original_linear.bias
        if self.bias is not None:
            self.bias.requires_grad_(False)

        self.lora_A       = nn.Parameter(torch.zeros(rank, self.in_features))
        self.lora_B       = nn.Parameter(torch.zeros(self.out_features, rank))
        self.lora_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: Tensor) -> Tensor:
        return (
            F.linear(x, self.weight, self.bias) +
            (self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scaling
        )

    def merge_weights(self) -> nn.Linear:
        merged = nn.Linear(self.in_features, self.out_features, bias=self.bias is not None)
        merged.weight.data = self.weight.data + (self.lora_B @ self.lora_A) * self.scaling
        if self.bias is not None:
            merged.bias.data = self.bias.data
        return merged

    @property
    def lora_parameters(self) -> int:
        return self.lora_A.numel() + self.lora_B.numel()


def apply_lora(
    model: AplexLLM,
    rank: int                    = 16,
    alpha: float                 = 32.0,
    dropout: float               = 0.05,
    target_modules: Optional[List[str]] = None,
) -> AplexLLM:
    if target_modules is None:
        target_modules = ["wq", "wv"]
    for param in model.parameters():
        param.requires_grad_(False)
    count = total_lora = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            short = name.split(".")[-1]
            if any(t in short for t in target_modules):
                parts = name.rsplit(".", 1)
                parent = dict(model.named_modules())[parts[0]] if len(parts) == 2 else model
                attr   = parts[-1]
                lora   = LoRALinear(module, rank=rank, alpha=alpha, dropout=dropout)
                setattr(parent, attr, lora)
                count       += 1
                total_lora  += lora.lora_parameters
    print(f"LoRA applied: {count} layers, {total_lora:,} trainable params "
          f"({total_lora / model.num_parameters * 100:.2f}% of total)")
    return model


def merge_lora(model: AplexLLM) -> AplexLLM:
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            parts  = name.rsplit(".", 1)
            parent = dict(model.named_modules())[parts[0]] if len(parts) == 2 else model
            setattr(parent, parts[-1], module.merge_weights())
    return model


# ==============================================================================
#  §10. QUANTIZATION UTILITIES
# ==============================================================================

def quantize_dynamic_int8(model: nn.Module) -> nn.Module:
    quantized = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8)
    n = sum(1 for m in quantized.modules()
            if isinstance(m, torch.nn.quantized.dynamic.Linear))
    print(f"Dynamic INT8 quantization applied to {n} linear layers.")
    return quantized


def estimate_model_size(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    return {
        "parameters": total,
        "fp32_mb": total * 4 / (1024 ** 2),
        "fp16_mb": total * 2 / (1024 ** 2),
        "int8_mb": total * 1 / (1024 ** 2),
        "int4_mb": total * 0.5 / (1024 ** 2),
    }


# ==============================================================================
#  §11. BPE TOKENIZER  (UPGRADED: byte-fallback, no <unk> on rare chars)
# ==============================================================================

class BPETokenizer:
    """
    Byte-Pair Encoding tokenizer with byte-fallback.

    Every possible byte (0x00–0xFF) is in the base vocabulary, so any
    Unicode text can be encoded without emitting <unk>.  This mirrors how
    GPT-2 / tiktoken handle unknown characters.

    Special tokens:
        <pad>  (0)
        <unk>  (1)   — kept for compatibility; should never appear now
        <bos>  (2)
        <eos>  (3)
    """
    SPECIAL_TOKENS = {"<pad>": 0, "<unk>": 1, "<bos>": 2, "<eos>": 3}
    # Byte tokens are stored as "Ġ00".."Ġff" (hex), avoiding collisions
    _BYTE_PREFIX = "Ġ"

    def __init__(self, vocab_size: int = 32_000):
        self.target_vocab_size = vocab_size
        self.merges: List[Tuple[str, str]] = []
        self.vocab: Dict[str, int]         = {}
        self.inverse_vocab: Dict[int, str] = {}
        self._initialized = False
        self.cache: Dict[str, List[str]] = {}
        self._compiled_pattern = re.compile(
            r"'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w\d]+|\s+(?!\S)|\s+",
            re.UNICODE,
        )

    # ------ byte helpers ------

    @classmethod
    def _byte_token(cls, b: int) -> str:
        return f"{cls._BYTE_PREFIX}{b:02x}"

    def _text_to_byte_tokens(self, text: str) -> List[str]:
        """Encode text as a sequence of byte-level token strings."""
        return [self._byte_token(b) for b in text.encode("utf-8")]

    def _byte_tokens_to_text(self, tokens: List[str]) -> str:
        """Decode byte-level tokens back to a UTF-8 string."""
        byte_list = []
        for tok in tokens:
            if tok.startswith(self._BYTE_PREFIX) and len(tok) == 3:
                try:
                    byte_list.append(int(tok[1:], 16))
                except ValueError:
                    pass
            else:
                byte_list.extend(tok.encode("utf-8"))
        return bytes(byte_list).decode("utf-8", errors="replace")

    # ------ BPE core ------

    def _get_pair_counts(self, seqs: List[List[str]]) -> Dict[Tuple[str, str], int]:
        counts: Dict[Tuple[str, str], int] = defaultdict(int)
        for seq in seqs:
            for i in range(len(seq) - 1):
                counts[(seq[i], seq[i + 1])] += 1
        return counts

    def _merge_pair(self, seqs: List[List[str]], pair: Tuple[str, str]) -> List[List[str]]:
        merged = pair[0] + pair[1]
        out = []
        for seq in seqs:
            new_seq, i = [], 0
            while i < len(seq):
                if i < len(seq) - 1 and seq[i] == pair[0] and seq[i + 1] == pair[1]:
                    new_seq.append(merged)
                    i += 2
                else:
                    new_seq.append(seq[i])
                    i += 1
            out.append(new_seq)
        return out

    def train(self, texts: List[str], verbose: bool = False) -> None:
        """Train BPE on a list of texts.  Byte-fallback ensures no <unk>."""
        if verbose:
            print(f"Training BPE tokenizer (target vocab: {self.target_vocab_size})...")

        self.vocab = dict(self.SPECIAL_TOKENS)
        next_id = len(self.SPECIAL_TOKENS)

        # Step 1: add all 256 byte tokens to the base vocab
        for b in range(256):
            tok = self._byte_token(b)
            if tok not in self.vocab:
                self.vocab[tok] = next_id
                next_id += 1

        # Step 2: add printable ASCII tokens directly (convenience, faster encoding)
        printable = (
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789 \n\t.,!?;:\"'()[]{}/-_+=@#$%^&*<>|\\~`"
        )
        for ch in printable:
            if ch not in self.vocab:
                self.vocab[ch] = next_id
                next_id += 1

        # Step 3: Represent corpus as byte-level sequences
        token_sequences: List[List[str]] = []
        for text in texts:
            # Try to keep common chars as single tokens, fall back to bytes
            words = self._compiled_pattern.findall(text)
            for word in words:
                seq = []
                for ch in word:
                    if ch in self.vocab:
                        seq.append(ch)
                    else:
                        seq.extend(self._text_to_byte_tokens(ch))
                token_sequences.append(seq)

        # Step 4: BPE merge loop
        self.merges = []
        num_merges  = max(0, self.target_vocab_size - next_id)

        for step in range(num_merges):
            pair_counts = self._get_pair_counts(token_sequences)
            if not pair_counts:
                break
            best_pair = max(pair_counts.items(), key=lambda item: item[1])[0]
            if pair_counts[best_pair] < 2:
                break
            token_sequences = self._merge_pair(token_sequences, best_pair)
            merged = best_pair[0] + best_pair[1]
            if merged not in self.vocab:
                self.vocab[merged] = next_id
                next_id += 1
            self.merges.append(best_pair)
            if verbose and (step + 1) % 500 == 0:
                print(f"  Merge {step+1}/{num_merges}: "
                      f"'{best_pair[0]}' + '{best_pair[1]}' → '{merged}' "
                      f"(freq={pair_counts[best_pair]})")

        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        self.cache = {}
        self._initialized = True
        if verbose:
            print(f"Tokenizer trained: {len(self.vocab)} tokens.")

    def _tokenize_word(self, word: str) -> List[str]:
        """Encode a single word segment, applying learned merges."""
        if word in self.cache:
            return list(self.cache[word])

        # Start with char-or-byte sequence
        tokens = []
        for ch in word:
            if ch in self.vocab:
                tokens.append(ch)
            else:
                tokens.extend(self._text_to_byte_tokens(ch))

        # Apply merges in order
        for merge_pair in self.merges:
            i, new_tokens = 0, []
            while i < len(tokens):
                if (i < len(tokens) - 1
                        and tokens[i] == merge_pair[0]
                        and tokens[i + 1] == merge_pair[1]):
                    new_tokens.append(merge_pair[0] + merge_pair[1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        self.cache[word] = tokens
        return tokens

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = False,
    ) -> List[int]:
        tokens: List[int] = []
        if add_bos:
            tokens.append(self.SPECIAL_TOKENS["<bos>"])
        for word in self._compiled_pattern.findall(text):
            for tok in self._tokenize_word(word):
                tokens.append(self.vocab.get(tok, self.SPECIAL_TOKENS["<unk>"]))
        if add_eos:
            tokens.append(self.SPECIAL_TOKENS["<eos>"])
        return tokens

    def decode(self, token_ids: List[int], skip_special: bool = True) -> str:
        special_ids = set(self.SPECIAL_TOKENS.values()) if skip_special else set()
        raw_tokens  = []
        for tid in token_ids:
            if tid in special_ids:
                continue
            raw_tokens.append(self.inverse_vocab.get(tid, ""))

        # Fast path: check if all tokens are plain text (no byte tokens)
        if not any(t.startswith(self._BYTE_PREFIX) and len(t) == 3 for t in raw_tokens):
            return "".join(raw_tokens)

        # Slow path: byte-decode hybrid sequence
        result = []
        i = 0
        while i < len(raw_tokens):
            tok = raw_tokens[i]
            if tok.startswith(self._BYTE_PREFIX) and len(tok) == 3:
                # Collect consecutive byte tokens
                byte_run = []
                while (i < len(raw_tokens)
                       and raw_tokens[i].startswith(self._BYTE_PREFIX)
                       and len(raw_tokens[i]) == 3):
                    try:
                        byte_run.append(int(raw_tokens[i][1:], 16))
                    except ValueError:
                        pass
                    i += 1
                result.append(bytes(byte_run).decode("utf-8", errors="replace"))
            else:
                result.append(tok)
                i += 1
        return "".join(result)

    def batch_encode(
        self,
        texts: List[str],
        max_length: int = 2048,
        padding: bool   = True,
        add_bos: bool   = True,
        add_eos: bool   = False,
    ) -> Tuple[Tensor, Tensor]:
        encoded = [self.encode(t, add_bos=add_bos, add_eos=add_eos)[:max_length]
                   for t in texts]
        if padding:
            max_len = min(max(len(e) for e in encoded), max_length)
            pad_id  = self.SPECIAL_TOKENS["<pad>"]
            padded  = [e + [pad_id] * (max_len - len(e)) for e in encoded]
            masks   = [[1] * len(e) + [0] * (max_len - len(e)) for e in encoded]
            return torch.tensor(padded, dtype=torch.long), torch.tensor(masks, dtype=torch.long)
        return torch.tensor(encoded, dtype=torch.long), torch.ones(len(encoded), dtype=torch.long)

    def save(self, path: Union[str, Path]) -> None:
        Path(path).write_text(json.dumps({
            "vocab": self.vocab,
            "merges": self.merges,
            "target_vocab_size": self.target_vocab_size,
        }, ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BPETokenizer":
        data = json.loads(Path(path).read_text())
        tok  = cls(vocab_size=data["target_vocab_size"])
        tok.vocab         = data["vocab"]
        tok.merges        = [tuple(m) for m in data["merges"]]
        tok.inverse_vocab = {v: k for k, v in tok.vocab.items()}
        tok.cache         = {}
        tok._initialized  = True
        return tok

    def __len__(self) -> int:
        return len(self.vocab)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


# ==============================================================================
#  §12. CHAT TEMPLATE  (NEW)
# ==============================================================================

class ChatTemplate:
    """
    ChatML-style chat template for instruction-tuned models.

    Format:
        <|system|>
        {system_content}<|end|>
        <|user|>
        {user_content}<|end|>
        <|assistant|>
        {assistant_content}<|end|>

    Special tokens are added to the tokenizer vocabulary so they are always
    encoded as single tokens (not split by BPE).
    """
    ROLE_SYSTEM    = "<|system|>"
    ROLE_USER      = "<|user|>"
    ROLE_ASSISTANT = "<|assistant|>"
    TOKEN_END      = "<|end|>"

    CHAT_SPECIAL_TOKENS = [
        "<|system|>", "<|user|>", "<|assistant|>", "<|end|>",
        "<|im_start|>", "<|im_end|>",  # ShareGPT compat aliases
    ]

    DEFAULT_SYSTEM = (
        "You are Aplx AI, a helpful, honest, and friendly local assistant "
        "running entirely on your machine."
    )

    def __init__(self, tokenizer: Optional[BPETokenizer] = None):
        self.tokenizer = tokenizer

    @staticmethod
    def _fmt_message(role_token: str, content: str, add_end: bool = True) -> str:
        end = f"\n{ChatTemplate.TOKEN_END}\n" if add_end else "\n"
        return f"{role_token}\n{content.strip()}{end}"

    def apply(
        self,
        messages: List[Dict[str, str]],
        add_generation_prompt: bool = False,
        system: Optional[str]       = None,
    ) -> str:
        """
        Convert a list of {"role": ..., "content": ...} dicts into a prompt string.

        Args:
            messages: List of message dicts with "role" and "content" keys.
            add_generation_prompt: If True, append the assistant role token at
                                   the end (used during inference to prime generation).
            system: Override system prompt. Uses DEFAULT_SYSTEM if None and no
                    system message is present in `messages`.
        """
        parts   = []
        has_sys = any(m["role"] == "system" for m in messages)

        if not has_sys and system is not False:
            sys_text = system or self.DEFAULT_SYSTEM
            parts.append(self._fmt_message(self.ROLE_SYSTEM, sys_text))

        for msg in messages:
            role    = msg["role"].lower()
            content = msg.get("content", "")
            if role == "system":
                parts.append(self._fmt_message(self.ROLE_SYSTEM, content))
            elif role in ("user", "human"):
                parts.append(self._fmt_message(self.ROLE_USER, content))
            elif role in ("assistant", "gpt", "bot"):
                parts.append(self._fmt_message(self.ROLE_ASSISTANT, content))

        if add_generation_prompt:
            parts.append(f"{self.ROLE_ASSISTANT}\n")

        return "".join(parts)

    def encode_for_training(
        self,
        messages: List[Dict[str, str]],
        tokenizer: BPETokenizer,
        max_length: int = 2048,
        system: Optional[str] = None,
    ) -> Tuple[List[int], List[int]]:
        """
        Encode a conversation for training.

        Returns:
            (input_ids, labels) where labels for non-assistant tokens are -100
            (excluded from loss computation).
        """
        input_ids: List[int] = []
        labels:    List[int] = []

        has_sys = any(m["role"] == "system" for m in messages)
        all_messages = list(messages)
        if not has_sys:
            all_messages = [{"role": "system", "content": system or self.DEFAULT_SYSTEM}] + all_messages

        for msg in all_messages:
            role    = msg["role"].lower()
            content = msg.get("content", "")
            is_asst = role in ("assistant", "gpt", "bot")

            if role == "system":
                text = self._fmt_message(self.ROLE_SYSTEM, content)
            elif role in ("user", "human"):
                text = self._fmt_message(self.ROLE_USER, content)
            else:
                text = self._fmt_message(self.ROLE_ASSISTANT, content)

            ids = tokenizer.encode(text, add_bos=False, add_eos=False)

            if len(input_ids) == 0:
                # Prepend <bos>
                bos = [tokenizer.SPECIAL_TOKENS["<bos>"]]
                input_ids.extend(bos)
                labels.extend([-100] * len(bos))

            if is_asst:
                # Find where the actual response starts (after "<|assistant|>\n")
                header = tokenizer.encode(
                    f"{self.ROLE_ASSISTANT}\n", add_bos=False, add_eos=False
                )
                body   = ids[len(header):]
                input_ids.extend(ids)
                labels.extend([-100] * len(header) + body)
            else:
                input_ids.extend(ids)
                labels.extend([-100] * len(ids))

        # Truncate
        input_ids = input_ids[:max_length]
        labels    = labels[:max_length]

        return input_ids, labels

    def register_special_tokens(self, tokenizer: BPETokenizer) -> None:
        """Add chat special tokens to an existing BPE tokenizer vocabulary."""
        next_id = max(tokenizer.vocab.values()) + 1
        for tok in self.CHAT_SPECIAL_TOKENS:
            if tok not in tokenizer.vocab:
                tokenizer.vocab[tok] = next_id
                tokenizer.inverse_vocab[next_id] = tok
                next_id += 1


# ==============================================================================
#  §13. INSTRUCTION / CHAT DATASET  (NEW)
# ==============================================================================

class InstructionDataset(Dataset):
    """
    Instruction-tuning dataset that supports multiple JSONL formats:

      • Alpaca:    {"instruction": ..., "input": ..., "output": ...}
      • ChatML:    {"messages": [{"role": ..., "content": ...}]}
      • ShareGPT:  {"conversations": [{"from": "human"|"gpt", "value": ...}]}
      • Plain:     {"text": ...}  (treated as raw LM data)

    Only assistant tokens contribute to the loss; prompt tokens are masked (-100).
    """

    def __init__(
        self,
        data: Union[str, Path, List[dict]],
        tokenizer: BPETokenizer,
        chat_template: Optional[ChatTemplate] = None,
        max_length: int = 2048,
        system: Optional[str] = None,
    ):
        self.tokenizer      = tokenizer
        self.template       = chat_template or ChatTemplate(tokenizer)
        self.max_length     = max_length
        self.system         = system
        self.samples: List[Tuple[List[int], List[int]]] = []

        raw = self._load(data)
        for item in raw:
            try:
                ids, lbls = self._process(item)
                if len(ids) > 1:
                    self.samples.append((ids, lbls))
            except Exception:
                continue

    def _load(self, data) -> List[dict]:
        if isinstance(data, (str, Path)):
            path = Path(data)
            if not path.exists():
                return []
            with open(path, encoding="utf-8") as f:
                if path.suffix == ".json":
                    obj = json.load(f)
                    return obj if isinstance(obj, list) else [obj]
                else:  # .jsonl
                    return [json.loads(line) for line in f if line.strip()]
        return data  # already a list

    def _process(self, item: dict) -> Tuple[List[int], List[int]]:
        # Detect format
        if "messages" in item:
            messages = item["messages"]
        elif "conversations" in item:
            messages = [
                {"role": "user" if c["from"] == "human" else "assistant",
                 "content": c["value"]}
                for c in item["conversations"]
            ]
        elif "instruction" in item:
            user_text = item["instruction"]
            if item.get("input"):
                user_text += "\n\n" + item["input"]
            messages = [
                {"role": "user",      "content": user_text},
                {"role": "assistant", "content": item.get("output", "")},
            ]
        elif "text" in item:
            # Plain text — full sequence contributes to loss
            ids = self.tokenizer.encode(item["text"], add_bos=True, add_eos=True)
            ids = ids[:self.max_length]
            return ids, list(ids)
        else:
            raise ValueError(f"Unknown format: {list(item.keys())}")

        return self.template.encode_for_training(
            messages, self.tokenizer, self.max_length, self.system
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        ids, lbls = self.samples[idx]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(lbls, dtype=torch.long)

    @staticmethod
    def collate_fn(
        batch: List[Tuple[Tensor, Tensor]],
        pad_id: int = 0,
    ) -> Tuple[Tensor, Tensor]:
        """Pad a batch of variable-length sequences. Preserves loss mask (-100 in y)."""
        if not batch:
            return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)
        xs, ys   = zip(*batch)
        max_len  = max(x.size(0) for x in xs) if xs else 1
        padded_x = torch.full((len(xs), max_len), pad_id,  dtype=torch.long)
        padded_y = torch.full((len(ys), max_len), -100,    dtype=torch.long)
        for i, (x, y) in enumerate(zip(xs, ys)):
            x_len = min(x.size(0), max_len)
            y_len = min(y.size(0), max_len)
            padded_x[i, :x_len] = x[:x_len]
            padded_y[i, :y_len] = y[:y_len]
        return padded_x, padded_y


# ==============================================================================
#  §14. PLAIN TEXT DATASET (packed, zero-waste)
# ==============================================================================

class TextDataset(Dataset):
    """
    Language-model training dataset.

    Tokenises and concatenates all texts into a single token stream,
    then slices into fixed-length chunks.  No padding is needed
    (every sample is exactly seq_len tokens), so training is 100% efficient.
    """
    def __init__(
        self,
        texts: List[str],
        tokenizer: BPETokenizer,
        seq_len: int = 2048,
    ):
        self.seq_len = seq_len
        all_tokens: List[int] = []
        for text in texts:
            all_tokens.extend(tokenizer.encode(text, add_bos=True, add_eos=True))
        if not all_tokens:
            self.tokens = torch.empty(0, dtype=torch.long)
            self.n_samples = 0
            return

        self.tokens = torch.tensor(all_tokens, dtype=torch.long)
        self.n_samples = max(1, (len(self.tokens) - 1) // seq_len + 1)

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        s = idx * self.seq_len
        e = s + self.seq_len
        x = self.tokens[s:e]
        # Target y is x shifted by 1 token, clamped to token sequence bounds
        y = self.tokens[s + 1:min(e + 1, len(self.tokens))]
        # Ensure x and y have compatible lengths
        if len(x) > len(y):
            x = x[:len(y)]
        elif len(y) > len(x):
            y = y[:len(x)]
        # Fallback for empty slices
        if len(x) == 0:
            available = len(self.tokens)
            if available > 0:
                x = self.tokens[:min(self.seq_len, available)]
                y = self.tokens[1:min(self.seq_len + 1, available)]
            else:
                x = torch.tensor([], dtype=torch.long)
                y = torch.tensor([], dtype=torch.long)
        return x, y

    @staticmethod
    def collate_fn(
        batch: List[Tuple[Tensor, Tensor]],
        pad_id: int = 0,
    ) -> Tuple[Tensor, Tensor]:
        """Pad batch ensuring all sequences have same length with loss mask on padding."""
        if not batch:
            return torch.tensor([], dtype=torch.long), torch.tensor([], dtype=torch.long)
        xs, ys = zip(*batch)
        max_len = max(x.size(0) for x in xs) if xs else 1
        padded_x = torch.full((len(xs), max_len), pad_id, dtype=torch.long)
        padded_y = torch.full((len(ys), max_len), -100, dtype=torch.long)
        for i, (x, y) in enumerate(zip(xs, ys)):
            x_len = min(x.size(0), max_len)
            y_len = min(y.size(0), max_len)
            if x_len > 0:
                padded_x[i, :x_len] = x[:x_len]
            if y_len > 0:
                padded_y[i, :y_len] = y[:y_len]
        return padded_x, padded_y


# ==============================================================================
#  §15. TEXT DATA AUGMENTER  (NEW — fixes aplx_1.6.py import)
# ==============================================================================

class TextDataAugmenter:
    """
    Text-level data augmentation to expand a small training corpus.

    Techniques:
      - Random word deletion
      - Word order shuffling (within sentence boundaries)
      - Sentence order shuffling
      - Simple synonym-like substitution (word casing variants)

    Usage::
        augmenter = TextDataAugmenter(augment_factor=3)
        texts_out = augmenter.augment(texts_in)
    """

    def __init__(
        self,
        augment_factor: int  = 3,
        deletion_prob:  float = 0.05,
        swap_prob:      float = 0.05,
        shuffle_sents:  bool  = True,
        seed:           int   = 42,
    ):
        self.augment_factor = augment_factor
        self.deletion_prob  = deletion_prob
        self.swap_prob      = swap_prob
        self.shuffle_sents  = shuffle_sents
        self._rng           = random.Random(seed)

    # ---------- augmentation primitives ----------

    def _delete_words(self, text: str) -> str:
        words = text.split()
        kept  = [w for w in words if self._rng.random() > self.deletion_prob]
        return " ".join(kept) if kept else text

    def _swap_words(self, text: str) -> str:
        words = text.split()
        for i in range(len(words) - 1):
            if self._rng.random() < self.swap_prob:
                words[i], words[i + 1] = words[i + 1], words[i]
        return " ".join(words)

    def _shuffle_sentences(self, text: str) -> str:
        sents = re.split(r'(?<=[.!?])\s+', text)
        if len(sents) > 1:
            self._rng.shuffle(sents)
        return " ".join(sents)

    def _case_variant(self, text: str) -> str:
        """Randomly title-case or lower-case the text."""
        choice = self._rng.randint(0, 2)
        if choice == 0:
            return text.lower()
        elif choice == 1:
            return text  # unchanged
        return text  # leave as-is for safety

    # ---------- public API ----------

    def augment_one(self, text: str) -> str:
        """Apply a random combination of augmentations to a single text."""
        text = self._delete_words(text)
        text = self._swap_words(text)
        if self.shuffle_sents:
            text = self._shuffle_sentences(text)
        return text

    def augment(self, texts: List[str]) -> List[str]:
        """
        Return the original texts plus ``augment_factor - 1`` augmented copies.

        Total size = len(texts) * augment_factor.
        """
        result = list(texts)  # keep originals
        for _ in range(self.augment_factor - 1):
            for text in texts:
                result.append(self.augment_one(text))
        return result


# ==============================================================================
#  §16. DATA PREPROCESSING UTILITIES  (NEW)
# ==============================================================================

def preprocess_texts(
    texts: List[str],
    min_length:  int   = 50,
    max_length:  int   = 100_000,
    dedup:       bool  = True,
    quality_filter: bool = True,
) -> List[str]:
    """
    Clean and filter a list of raw training texts.

    Steps:
      1. Length filter (discard very short / very long documents)
      2. Deduplication by SHA-256 hash (exact dedup)
      3. Quality filter: discard texts with too many non-alpha characters
    """
    seen:   Set[str] = set()
    result: List[str] = []

    for text in texts:
        text = text.strip()

        # Length
        if len(text) < min_length or len(text) > max_length:
            continue

        # Dedup
        if dedup:
            h = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if h in seen:
                continue
            seen.add(h)

        # Quality: at least 50% alphabetic characters
        if quality_filter:
            alpha = sum(c.isalpha() for c in text)
            if alpha / max(len(text), 1) < 0.5:
                continue

        result.append(text)

    return result


# ==============================================================================
#  §17. GENERATION CONFIG & STRATEGIES
# ==============================================================================

@dataclass
class GenerationConfig:
    max_new_tokens:     int   = 256
    temperature:        float = 0.8
    top_k:              int   = 50
    top_p:              float = 0.9
    min_p:              float = 0.0    # Min-p sampling threshold
    repetition_penalty: float = 1.1
    do_sample:          bool  = True
    num_beams:          int   = 1
    length_penalty:     float = 1.0
    early_stopping:     bool  = True
    eos_token_id:       int   = 3
    pad_token_id:       int   = 0
    typical_p:          float = 1.0   # < 1.0 enables typical sampling


# ==============================================================================
#  §18. TEXT GENERATOR  (UPGRADED: streaming, chat session, min-p, typical)
# ==============================================================================

class TextGenerator:
    """
    Text generation engine with multiple decoding strategies.

    New in v3.0:
      - min-p sampling (Nguyen et al., 2023)
      - typical sampling (entropy-based)
      - stream_to_console(): live per-token output
      - chat(): full multi-turn conversation loop
    """

    def __init__(self, model: AplexLLM, tokenizer: Optional[BPETokenizer] = None):
        self.model     = model
        self.tokenizer = tokenizer
        self.device    = next(model.parameters()).device

    # ---------- sampling helpers ----------

    def _apply_repetition_penalty(
        self, logits: Tensor, generated: Tensor, penalty: float
    ) -> Tensor:
        if penalty == 1.0:
            return logits
        score = torch.gather(logits, 1, generated)
        score = torch.where(score < 0, score * penalty, score / penalty)
        logits.scatter_(1, generated, score)
        return logits

    def _sample_from_logits(self, logits: Tensor, cfg: GenerationConfig) -> Tensor:
        """Apply temperature, top-k, top-p, min-p, typical, then sample."""

        if cfg.temperature > 0:
            logits = logits / cfg.temperature

        # Top-K
        if cfg.top_k > 0:
            k       = min(cfg.top_k, logits.size(-1))
            thresh  = torch.topk(logits, k)[0][..., -1, None]
            logits  = logits.masked_fill(logits < thresh, float("-inf"))

        # Top-P (nucleus)
        if cfg.top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True)
            cumprobs  = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            remove    = (cumprobs - F.softmax(sorted_logits, dim=-1)) > cfg.top_p
            sorted_logits[remove] = float("-inf")
            logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

        # Min-P
        if cfg.min_p > 0.0:
            probs   = F.softmax(logits, dim=-1)
            max_p   = probs.max(dim=-1, keepdim=True).values
            thresh  = max_p * cfg.min_p
            logits  = logits.masked_fill(probs < thresh, float("-inf"))

        # Typical sampling
        if cfg.typical_p < 1.0:
            probs   = F.softmax(logits, dim=-1)
            # Entropy of the distribution
            entropy = -(probs * (probs + 1e-9).log()).sum(dim=-1, keepdim=True)
            # |H - log p(x)|
            neg_log = -(probs + 1e-9).log()
            shift   = (neg_log - entropy).abs()
            sorted_shift, sorted_idx = torch.sort(shift, dim=-1)
            sorted_probs = probs.gather(1, sorted_idx)
            cumprobs = sorted_probs.cumsum(dim=-1)
            remove   = cumprobs - sorted_probs > cfg.typical_p
            logits   = logits.masked_fill(
                remove.scatter(1, sorted_idx, remove), float("-inf")
            )

        if cfg.do_sample:
            probs = F.softmax(logits, dim=-1)
            return torch.multinomial(probs, num_samples=1)
        return torch.argmax(logits, dim=-1, keepdim=True)

    # ---------- generation loops ----------

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: Tensor,
        gen_config:    GenerationConfig = GenerationConfig(),
        stream:        bool             = False,
    ) -> Union[Tensor, Generator[Tensor, None, None]]:
        if gen_config.num_beams > 1:
            return self._beam_search(prompt_tokens, gen_config)
        if stream:
            return self._stream_generate(prompt_tokens, gen_config)
        return self._sample_generate(prompt_tokens, gen_config)

    def _sample_generate(
        self, prompt_tokens: Tensor, cfg: GenerationConfig
    ) -> Tensor:
        self.model.eval()
        prompt_tokens = prompt_tokens.to(self.device)
        bsz, prompt_len = prompt_tokens.shape

        self.model.allocate_kv_cache(
            device=self.device,
            dtype=next(self.model.parameters()).dtype,
        )

        logits    = self.model(prompt_tokens, start_pos=0)
        generated = prompt_tokens.clone()

        for i in range(cfg.max_new_tokens):
            next_logits = logits[:, -1, :]  # (bsz, vocab_size)
            next_logits = self._apply_repetition_penalty(next_logits, generated, cfg.repetition_penalty)
            next_token  = self._sample_from_logits(next_logits, cfg)
            generated   = torch.cat([generated, next_token], dim=1)

            if (next_token == cfg.eos_token_id).all():
                break

            logits = self.model(next_token, start_pos=prompt_len + i)

        self.model.free_kv_cache()
        return generated

    def _stream_generate(
        self, prompt_tokens: Tensor, cfg: GenerationConfig
    ) -> Generator[Tensor, None, None]:
        """Yield one token tensor at a time for streaming output."""
        self.model.eval()
        prompt_tokens = prompt_tokens.to(self.device)
        bsz, prompt_len = prompt_tokens.shape

        self.model.allocate_kv_cache(
            device=self.device,
            dtype=next(self.model.parameters()).dtype,
        )

        logits    = self.model(prompt_tokens, start_pos=0)
        generated = prompt_tokens.clone()

        for i in range(cfg.max_new_tokens):
            next_logits = logits[:, -1, :]  # (bsz, vocab_size)
            next_logits = self._apply_repetition_penalty(next_logits, generated, cfg.repetition_penalty)
            next_token  = self._sample_from_logits(next_logits, cfg)
            generated   = torch.cat([generated, next_token], dim=1)

            yield next_token

            if (next_token == cfg.eos_token_id).all():
                break

            logits = self.model(next_token, start_pos=prompt_len + i)

        self.model.free_kv_cache()

    @torch.inference_mode()
    def stream_to_console(
        self,
        prompt: str,
        gen_config: GenerationConfig = GenerationConfig(),
        end: str = "\n",
    ) -> str:
        """Stream generated text to stdout and return the full string."""
        assert self.tokenizer is not None, "Tokenizer required"
        tokens     = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        input_ids  = torch.tensor([tokens], dtype=torch.long)

        full_text = ""
        for token_tensor in self._stream_generate(input_ids, gen_config):
            token_text = self.tokenizer.decode(token_tensor[0].tolist(), skip_special=True)
            print(token_text, end="", flush=True)
            full_text += token_text

        print(end, end="", flush=True)
        return full_text

    def _beam_search(self, prompt_tokens: Tensor, cfg: GenerationConfig) -> Tensor:
        """Beam search decoding."""
        self.model.eval()
        prompt_tokens = prompt_tokens.to(self.device)
        num_beams     = cfg.num_beams

        beams: List[Tuple[float, List[int]]] = [(0.0, prompt_tokens[0].tolist())]
        completed: List[Tuple[float, List[int]]] = []

        for _ in range(cfg.max_new_tokens):
            all_candidates: List[Tuple[float, List[int]]] = []
            for score, seq in beams:
                input_ids = torch.tensor([seq], device=self.device)
                with torch.no_grad():
                    logits = self.model(input_ids)
                log_probs = F.log_softmax(logits[0, -1, :], dim=-1)
                topk_lp, topk_ids = torch.topk(log_probs, num_beams * 2)
                for j in range(num_beams * 2):
                    tid   = topk_ids[j].item()
                    tsc   = topk_lp[j].item()
                    new_s = (score * len(seq) + tsc) / (len(seq) + 1) ** cfg.length_penalty
                    if tid == cfg.eos_token_id:
                        completed.append((new_s, seq + [tid]))
                    else:
                        all_candidates.append((new_s, seq + [tid]))
            if not all_candidates:
                break
            all_candidates.sort(key=lambda x: x[0], reverse=True)
            beams = all_candidates[:num_beams]
            if cfg.early_stopping and len(completed) >= num_beams:
                break

        all_seqs = (completed + beams)
        all_seqs.sort(key=lambda x: x[0], reverse=True)
        return torch.tensor([all_seqs[0][1]], device=self.device)

    def generate_text(
        self,
        prompt: str,
        gen_config: GenerationConfig = GenerationConfig(),
        stream:     bool             = False,
    ) -> str:
        assert self.tokenizer is not None, "Tokenizer required"
        if stream:
            return self.stream_to_console(prompt, gen_config)
        tokens    = self.tokenizer.encode(prompt, add_bos=True, add_eos=False)
        input_ids = torch.tensor([tokens], dtype=torch.long)
        output    = self.generate(input_ids, gen_config)
        return self.tokenizer.decode(output[0].tolist(), skip_special=True)

    def chat(
        self,
        chat_template: Optional[ChatTemplate] = None,
        gen_config:    GenerationConfig       = GenerationConfig(),
        system:        Optional[str]          = None,
        stream:        bool                   = True,
    ) -> None:
        """
        Start an interactive multi-turn console chat session.

        Type 'quit', 'exit', or '/bye' to end the session.
        Type '/reset' to clear the conversation history.
        Type '/config' to show current generation settings.
        """
        assert self.tokenizer is not None, "Tokenizer required for chat"
        template = chat_template or ChatTemplate(self.tokenizer)
        history:  List[Dict[str, str]] = []

        print("\n" + "═" * 60)
        print("  Aplx AI — Chat Session")
        print("  Type 'exit' or '/bye' to quit, '/reset' to clear history")
        print("═" * 60 + "\n")

        while True:
            try:
                user_input = input("You: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n[Session ended]")
                break

            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "/bye"):
                print("[Session ended]")
                break
            if user_input.lower() == "/reset":
                history.clear()
                print("[History cleared]\n")
                continue
            if user_input.lower() == "/config":
                print(f"[GenerationConfig] {gen_config}\n")
                continue

            history.append({"role": "user", "content": user_input})
            prompt = template.apply(history, add_generation_prompt=True, system=system)

            print("Aplx: ", end="", flush=True)
            if stream:
                response = self.stream_to_console(prompt, gen_config, end="")
                print()
            else:
                response = self.generate_text(prompt, gen_config)
                # Strip the echoed prompt if present
                if response.startswith(prompt):
                    response = response[len(prompt):]
                print(response)

            # Keep only the assistant response (without template tokens)
            clean = response
            for tok in [ChatTemplate.ROLE_ASSISTANT, ChatTemplate.TOKEN_END, "\n"]:
                clean = clean.replace(tok, " ")
            clean = clean.strip()
            history.append({"role": "assistant", "content": clean})
            print()


# ==============================================================================
#  §19. TRAINING INFRASTRUCTURE
# ==============================================================================

@dataclass
class TrainingConfig:
    # Optimization
    learning_rate:     float = 3e-4
    min_learning_rate: float = 1e-5
    weight_decay:      float = 0.1
    beta1:             float = 0.9
    beta2:             float = 0.95
    eps:               float = 1e-8
    max_grad_norm:     float = 1.0

    # Schedule
    warmup_steps:   int = 2000
    total_steps:    int = 0
    lr_decay_style: str = "cosine"

    # Batching
    batch_size:                  int = 1
    gradient_accumulation_steps: int = 8

    # Mixed Precision
    use_amp:   bool = True
    amp_dtype: str  = "float16"

    # Logging
    log_interval:  int = 10
    eval_interval: int = 500
    save_interval: int = 1000

    # Checkpointing
    output_dir:       str          = "./checkpoints"
    save_total_limit: int          = 3
    resume_from:      Optional[str]= None

    # Distributed
    use_ddp: bool = False

    # WandB / TensorBoard
    use_wandb:       bool          = False
    wandb_project:   str           = "aplx_llm"
    wandb_run_name:  Optional[str] = None
    use_tensorboard: bool          = False
    tensorboard_dir: str           = "./runs"

    @property
    def effective_batch_size(self) -> int:
        return self.batch_size * self.gradient_accumulation_steps


class CosineWarmupScheduler:
    """Linear warmup → cosine decay LR schedule."""
    def __init__(
        self,
        optimizer:   torch.optim.Optimizer,
        warmup_steps: int,
        total_steps:  int,
        max_lr:       float,
        min_lr:       float,
    ):
        self.optimizer    = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps  = total_steps
        self.max_lr       = max_lr
        self.min_lr       = min_lr
        self.current_step = 0

    def get_lr(self) -> float:
        if self.total_steps == 0:
            if self.current_step < self.warmup_steps:
                return self.max_lr * self.current_step / max(1, self.warmup_steps)
            return self.max_lr
        if self.current_step < self.warmup_steps:
            return self.max_lr * self.current_step / max(1, self.warmup_steps)
        if self.current_step >= self.total_steps:
            return self.min_lr
        progress = (self.current_step - self.warmup_steps) / max(
            1, self.total_steps - self.warmup_steps)
        return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1.0 + math.cos(math.pi * progress))

    def step(self) -> float:
        lr = self.get_lr()
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        self.current_step += 1
        return lr


class MetricsTracker:
    def __init__(self, log_interval: int = 10):
        self.log_interval = log_interval
        self.history: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        self._running: Dict[str, float] = defaultdict(float)
        self._counts:  Dict[str, int]   = defaultdict(int)
        self._start    = time.time()

    def update(self, step: int, **metrics: float) -> None:
        for k, v in metrics.items():
            self._running[k] += v
            self._counts[k]  += 1

    def get_smoothed(self) -> Dict[str, float]:
        return {k: self._running[k] / self._counts[k]
                for k in self._running if self._counts[k] > 0}

    def reset_running(self) -> None:
        self._running.clear()
        self._counts.clear()

    def log(self, step: int, extra: Optional[Dict[str, Any]] = None) -> str:
        smoothed = self.get_smoothed()
        elapsed  = time.time() - self._start
        parts    = [f"step={step:>6d}", f"elapsed={elapsed:.1f}s"]
        for k, v in smoothed.items():
            if "loss" in k:
                parts.append(f"{k}={v:.4f}")
                parts.append(f"ppl={math.exp(min(v, 20)):.2f}")
            elif "lr" in k:
                parts.append(f"{k}={v:.2e}")
            else:
                parts.append(f"{k}={v:.4f}")
        if extra:
            for k, v in extra.items():
                parts.append(f"{k}={v}")
        for k, v in smoothed.items():
            self.history[k].append((step, v))
        self.reset_running()
        return " | ".join(parts)


class Trainer:
    """
    Full-featured training loop for AplexLLM.

    Supports:
      - Mixed precision (FP16 / BF16)
      - Gradient accumulation & clipping
      - Cosine warmup LR
      - Periodic evaluation & checkpointing
      - DDP
      - Optional WandB / TensorBoard logging
      - InstructionDataset (chat fine-tuning)
      - Plain TextDataset (pre-training)
    """

    def __init__(
        self,
        model:         AplexLLM,
        train_config:  TrainingConfig,
        train_dataset: Dataset,
        eval_dataset:  Optional[Dataset] = None,
        tokenizer:     Optional[BPETokenizer] = None,
    ):
        self.model        = model
        self.config       = train_config
        self.train_dataset= train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer    = tokenizer
        self.device       = next(model.parameters()).device

        self.optimizer = self._build_optimizer()
        self.scheduler = CosineWarmupScheduler(
            optimizer=self.optimizer,
            warmup_steps=train_config.warmup_steps,
            total_steps=train_config.total_steps,
            max_lr=train_config.learning_rate,
            min_lr=train_config.min_learning_rate,
        )
        self.scaler   = GradScaler(enabled=train_config.use_amp)
        self.amp_dtype = (
            torch.bfloat16 if train_config.amp_dtype == "bfloat16" else torch.float16
        )
        self.metrics  = MetricsTracker(log_interval=train_config.log_interval)

        # Dataloader
        collate = None
        if isinstance(train_dataset, InstructionDataset):
            pad_id  = tokenizer.SPECIAL_TOKENS["<pad>"] if tokenizer else 0
            collate = lambda b: InstructionDataset.collate_fn(b, pad_id=pad_id)
        elif isinstance(train_dataset, TextDataset):
            collate = lambda b: TextDataset.collate_fn(b, pad_id=0)

        sampler = DistributedSampler(train_dataset) if train_config.use_ddp else None
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=max(1, train_config.batch_size),
            shuffle=(sampler is None),
            sampler=sampler,
            num_workers=0,
            pin_memory=(self.device.type == "cuda"),
            drop_last=False,
            collate_fn=collate,
        )

        if eval_dataset is not None:
            eval_collate = collate if isinstance(eval_dataset, InstructionDataset) else None
            self.eval_loader = DataLoader(
                eval_dataset,
                batch_size=max(1, train_config.batch_size),
                shuffle=False,
                num_workers=0,
                pin_memory=(self.device.type == "cuda"),
                collate_fn=eval_collate,
            )
        else:
            self.eval_loader = None

        os.makedirs(train_config.output_dir, exist_ok=True)
        self.global_step    = 0
        self.best_eval_loss = float("inf")

        # Optional loggers
        self._wandb    = None
        self._tb_writer= None
        self._init_loggers()

    def _init_loggers(self) -> None:
        if self.config.use_wandb:
            try:
                import wandb
                self._wandb = wandb.init(
                    project=self.config.wandb_project,
                    name=self.config.wandb_run_name,
                    config=asdict(self.config),
                )
            except ImportError:
                print("[WARN] wandb not installed — skipping WandB logging")

        if self.config.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self._tb_writer = SummaryWriter(self.config.tensorboard_dir)
            except ImportError:
                print("[WARN] tensorboard not installed — skipping TensorBoard logging")

    def _log_external(self, step: int, metrics: Dict[str, float]) -> None:
        if self._wandb is not None:
            self._wandb.log(metrics, step=step)
        if self._tb_writer is not None:
            for k, v in metrics.items():
                self._tb_writer.add_scalar(k, v, step)

    def _build_optimizer(self) -> torch.optim.AdamW:
        decay, no_decay = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim == 1 or any(k in name for k in ("bias", "norm", "embedding")):
                no_decay.append(param)
            else:
                decay.append(param)
        return torch.optim.AdamW(
            [{"params": decay, "weight_decay": self.config.weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
            eps=self.config.eps,
        )

    def train(self) -> Dict[str, List]:
        self.model.train()
        print(f"\n{'='*60}")
        print(f"  Starting Training — {self.model.config.model_name}")
        print(f"  Parameters:          {self.model.num_parameters:,}")
        print(f"  Effective batch:     {self.config.effective_batch_size}")
        print(f"  Total steps:         {self.config.total_steps}")
        print(f"  AMP: {self.config.use_amp} ({self.config.amp_dtype})")
        print(f"{'='*60}\n")

        data_iter = iter(self.train_loader)

        step = 0
        while self.config.total_steps == 0 or step < self.config.total_steps:
            step += 1
            self.global_step = step
            total_loss = 0.0
            self.optimizer.zero_grad(set_to_none=True)

            for _ in range(self.config.gradient_accumulation_steps):
                try:
                    batch_x, batch_y = next(data_iter)
                except StopIteration:
                    data_iter = iter(self.train_loader)
                    try:
                        batch_x, batch_y = next(data_iter)
                    except StopIteration:
                        # Fallback for tiny datasets: use a single sample directly.
                        if len(self.train_dataset) > 0:
                            batch_x, batch_y = self.train_dataset[0]
                            batch_x = batch_x.unsqueeze(0) if batch_x.dim() == 1 else batch_x.unsqueeze(0) if batch_x.size(0) != len(batch_x) else batch_x
                            batch_y = batch_y.unsqueeze(0) if batch_y.dim() == 1 else batch_y.unsqueeze(0) if batch_y.size(0) != len(batch_y) else batch_y
                        else:
                            continue

                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                with autocast(enabled=self.config.use_amp, dtype=self.amp_dtype):
                    loss_out = self.model(batch_x, targets=batch_y)
                    if isinstance(loss_out, tuple):
                        loss = loss_out[0]
                    else:
                        loss = loss_out
                    # Normalize loss by gradient accumulation for stable training
                    loss = loss / self.config.gradient_accumulation_steps

                self.scaler.scale(loss).backward()
                total_loss += loss.item() * self.config.gradient_accumulation_steps  # Undo normalization for logging

            if self.config.max_grad_norm > 0:
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.grad is not None],
                    self.config.max_grad_norm
                )
            else:
                grad_norm = 0.0

            self.scaler.step(self.optimizer)
            self.scaler.update()
            current_lr = self.scheduler.step()

            self.metrics.update(step, loss=total_loss, lr=current_lr, grad_norm=float(grad_norm))

            if step % self.config.log_interval == 0:
                log_str = self.metrics.log(step)
                print(f"  [TRAIN] {log_str}")
                self._log_external(step, {"train/loss": total_loss, "train/lr": current_lr})

            if self.eval_loader and step % self.config.eval_interval == 0:
                eval_loss = self.evaluate()
                print(f"  [EVAL]  step={step:>6d} | eval_loss={eval_loss:.4f} | "
                      f"eval_ppl={math.exp(min(eval_loss, 20)):.2f}")
                self._log_external(step, {"eval/loss": eval_loss})
                if eval_loss < self.best_eval_loss:
                    self.best_eval_loss = eval_loss
                    self.save_checkpoint("best")
                self.model.train()

            if step % self.config.save_interval == 0:
                self.save_checkpoint(f"step_{step}")

        print(f"\n{'='*60}")
        print(f"  Training Complete! Best eval loss: {self.best_eval_loss:.4f}")
        print(f"{'='*60}\n")

        if self._tb_writer:
            self._tb_writer.close()

        return dict(self.metrics.history)

    @torch.no_grad()
    def evaluate(self) -> float:
        if self.eval_loader is None:
            return float("inf")
        self.model.eval()
        total, n = 0.0, 0
        for bx, by in self.eval_loader:
            bx = bx.to(self.device)
            by = by.to(self.device)
            with autocast(enabled=self.config.use_amp, dtype=self.amp_dtype):
                loss_out = self.model(bx, targets=by)
                if isinstance(loss_out, tuple):
                    loss = loss_out[0]
                else:
                    loss = loss_out
            total += loss.item()
            n     += 1
        return total / max(n, 1)

    def save_checkpoint(self, name: str) -> str:
        ckpt_dir = os.path.join(self.config.output_dir, name)
        os.makedirs(ckpt_dir, exist_ok=True)
        try:
            torch.save({
                "model_state_dict":     self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_step":       self.scheduler.current_step,
                "scaler_state_dict":    self.scaler.state_dict(),
                "global_step":          self.global_step,
                "best_eval_loss":       self.best_eval_loss,
                "model_config":         self.model.config.to_dict() if hasattr(self.model.config, 'to_dict') else None,
                "training_config":      asdict(self.config),
            }, os.path.join(ckpt_dir, "checkpoint.pt"))
        except Exception as e:
            print(f"  [WARN]  Failed to save checkpoint.pt: {e}")
        
        try:
            if hasattr(self.model.config, 'save'):
                self.model.config.save(os.path.join(ckpt_dir, "config.json"))
        except Exception as e:
            print(f"  [WARN]  Failed to save config.json: {e}")
        
        try:
            if self.tokenizer is not None and hasattr(self.tokenizer, 'save'):
                self.tokenizer.save(os.path.join(ckpt_dir, "tokenizer.json"))
        except Exception as e:
            print(f"  [WARN]  Failed to save tokenizer.json: {e}")
        
        print(f"  [CKPT]  Saved: {ckpt_dir}")
        self._cleanup_checkpoints()
        return ckpt_dir

    def load_checkpoint(self, path: str) -> None:
        try:
            ck = torch.load(path, map_location=self.device)
            if "model_state_dict" in ck:
                self.model.load_state_dict(ck["model_state_dict"], strict=False)
            if "optimizer_state_dict" in ck:
                try:
                    self.optimizer.load_state_dict(ck["optimizer_state_dict"])
                except Exception as e:
                    print(f"  [WARN]  Could not load optimizer state: {e}")
            if "scheduler_step" in ck:
                self.scheduler.current_step = ck["scheduler_step"]
            if "scaler_state_dict" in ck:
                try:
                    self.scaler.load_state_dict(ck["scaler_state_dict"])
                except Exception as e:
                    print(f"  [WARN]  Could not load scaler state: {e}")
            self.global_step    = ck.get("global_step", 0)
            self.best_eval_loss = ck.get("best_eval_loss", float("inf"))
            print(f"  [CKPT]  Resumed from step {self.global_step}")
        except Exception as e:
            print(f"  [ERROR] Failed to load checkpoint: {e}")

    def _cleanup_checkpoints(self) -> None:
        if self.config.save_total_limit <= 0:
            return
        import shutil
        checkpoints = sorted(
            [d for d in Path(self.config.output_dir).iterdir()
             if d.is_dir() and d.name.startswith("step_")],
            key=lambda d: int(d.name.split("_")[1]),
        )
        while len(checkpoints) > self.config.save_total_limit:
            shutil.rmtree(checkpoints.pop(0))


# ==============================================================================
#  §20. PERPLEXITY EVALUATOR  (NEW)
# ==============================================================================

@torch.no_grad()
def evaluate_perplexity(
    model:     AplexLLM,
    tokenizer: BPETokenizer,
    texts:     List[str],
    seq_len:   int = 512,
    device:    Optional[torch.device] = None,
) -> float:
    """
    Compute perplexity on a list of evaluation texts.

    Perplexity = exp(average negative log-likelihood per token).
    Lower is better.  GPT-2 scores ~29 on WikiText-103;
    a well-trained 1B model typically scores < 10.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    total_nll = 0.0
    total_tokens = 0

    dataset = TextDataset(texts, tokenizer, seq_len=seq_len)
    if len(dataset) == 0:
        return float("inf")

    loader = DataLoader(dataset, batch_size=4, shuffle=False)

    for bx, by in loader:
        bx = bx.to(device)
        by = by.to(device)
        loss_out = model(bx, targets=by)
        if isinstance(loss_out, tuple):
            loss = loss_out[0]
        else:
            loss = loss_out
        # loss is mean NLL; recover total NLL
        valid_tokens = (by != -100).sum().item()
        total_nll    += loss.item() * valid_tokens
        total_tokens += valid_tokens

    if total_tokens == 0:
        return float("inf")

    return math.exp(min(total_nll / total_tokens, 20))


# ==============================================================================
#  §21. DISTRIBUTED TRAINING UTILITIES
# ==============================================================================

def setup_distributed(rank: int, world_size: int, backend: str = "nccl") -> None:
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "12355")
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)


def cleanup_distributed() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def wrap_ddp(model: nn.Module, device_id: int) -> DDP:
    return DDP(model, device_ids=[device_id], output_device=device_id)


# ==============================================================================
#  §22. MODEL PERSISTENCE
# ==============================================================================

def save_model(
    model:     AplexLLM,
    path:      Union[str, Path],
    tokenizer: Optional[BPETokenizer] = None,
) -> None:
    """Save model weights + config + optional tokenizer to a directory."""
    save_dir = Path(path)
    save_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), save_dir / "model.pt")
    model.config.save(save_dir / "config.json")

    if tokenizer is not None:
        tokenizer.save(save_dir / "tokenizer.json")

    summary = {
        "model_name": model.config.model_name,
        "model_version": model.config.model_version,
        "total_parameters": model.num_parameters,
        "architecture": {
            "vocab_size":  model.config.vocab_size,
            "dim":         model.config.dim,
            "n_layers":    model.config.n_layers,
            "n_heads":     model.config.n_heads,
            "n_kv_heads":  model.config.n_kv_heads,
            "hidden_dim":  model.config.hidden_dim,
            "max_seq_len": model.config.max_seq_len,
        },
        "size_estimates": estimate_model_size(model),
    }
    (save_dir / "model_card.json").write_text(json.dumps(summary, indent=2))
    print(f"Model saved to {save_dir}")


def load_model(
    path:           Union[str, Path],
    device:         str  = "auto",
    load_tokenizer: bool = True,
) -> Tuple[AplexLLM, Optional[BPETokenizer]]:
    """Load a model package from disk."""
    load_dir = Path(path)
    resolved_device = device if device != "auto" else str(get_default_device())

    config = ModelConfig.load(load_dir / "config.json")
    model  = AplexLLM(config)
    model.load_state_dict(torch.load(load_dir / "model.pt", map_location=resolved_device,
                                     weights_only=False))
    model  = model.to(resolved_device)
    model.eval()

    tokenizer = None
    tok_path  = load_dir / "tokenizer.json"
    if load_tokenizer and tok_path.exists():
        tokenizer = BPETokenizer.load(tok_path)

    print(f"Model loaded from {load_dir} ({model.num_parameters:,} parameters)")
    return model, tokenizer


# ==============================================================================
#  §23. EXPORT UTILITIES  (NEW)
# ==============================================================================

def export_safetensors(
    model: AplexLLM,
    path:  Union[str, Path],
) -> None:
    """
    Export model weights in safetensors format (requires `pip install safetensors`).

    Safetensors is faster and safer than pickle-based .pt files and is the
    standard weight format for HuggingFace models.
    """
    try:
        from safetensors.torch import save_file
    except ImportError:
        raise ImportError(
            "safetensors not installed.  Run: pip install safetensors"
        )
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensors = {k: v.contiguous() for k, v in model.state_dict().items()}
    save_file(tensors, str(path))
    size_mb = path.stat().st_size / (1024 ** 2)
    print(f"Safetensors export: {path} ({size_mb:.1f} MB)")


def to_huggingface_config(config: ModelConfig) -> dict:
    """
    Generate a HuggingFace-compatible config.json.

    This allows loading the weights with `AutoModelForCausalLM.from_pretrained()`
    after renaming the parameter keys.
    """
    return {
        "architectures":            ["LlamaForCausalLM"],
        "model_type":               "llama",
        "bos_token_id":             2,
        "eos_token_id":             3,
        "pad_token_id":             0,
        "hidden_size":              config.dim,
        "intermediate_size":        config.hidden_dim,
        "max_position_embeddings":  config.max_seq_len,
        "num_attention_heads":      config.n_heads,
        "num_key_value_heads":      config.n_kv_heads,
        "num_hidden_layers":        config.n_layers,
        "rms_norm_eps":             config.norm_eps,
        "rope_theta":               config.rope_theta,
        "sliding_window":           config.sliding_window_size if config.use_sliding_window else None,
        "tie_word_embeddings":      config.tie_word_embeddings,
        "torch_dtype":              "float32",
        "transformers_version":     "4.40.0",
        "vocab_size":               config.vocab_size,
        "_aplx_model_version":      config.model_version,
    }


def export_huggingface_config(
    model: AplexLLM,
    path:  Union[str, Path],
    tokenizer: Optional[BPETokenizer] = None,
) -> None:
    """Write HuggingFace-compatible config.json and tokenizer_config.json."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    hf_cfg = to_huggingface_config(model.config)
    (path / "config.json").write_text(json.dumps(hf_cfg, indent=2))

    if tokenizer is not None:
        tok_cfg = {
            "bos_token": "<bos>",
            "eos_token": "<eos>",
            "unk_token": "<unk>",
            "pad_token": "<pad>",
            "model_max_length": model.config.max_seq_len,
            "tokenizer_class":  "PreTrainedTokenizerFast",
        }
        (path / "tokenizer_config.json").write_text(json.dumps(tok_cfg, indent=2))
        tokenizer.save(path / "aplx_tokenizer.json")

    print(f"HuggingFace config exported to {path}")


# ==============================================================================
#  §24. BENCHMARKING
# ==============================================================================

class ModelBenchmark:
    def __init__(self, model: AplexLLM, device: str = "cpu"):
        self.model  = model
        self.device = torch.device(device)
        self.model  = self.model.to(self.device)

    @torch.no_grad()
    def benchmark_throughput(
        self,
        batch_size: int = 1,
        seq_len:    int = 512,
        n_iter:     int = 10,
        warmup:     int = 3,
    ) -> Dict[str, float]:
        self.model.eval()
        dummy = torch.randint(0, self.model.config.vocab_size, (batch_size, seq_len), device=self.device)
        for _ in range(warmup):
            _ = self.model(dummy)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(n_iter):
            _ = self.model(dummy)
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        total   = batch_size * seq_len * n_iter
        return {
            "total_time_s":             elapsed,
            "avg_latency_ms":           elapsed / n_iter * 1000,
            "throughput_tokens_per_sec": total / elapsed,
        }

    def memory_profile(self) -> Dict[str, float]:
        size = estimate_model_size(self.model)
        r    = {"parameters": size["parameters"],
                "fp32_size_mb": size["fp32_mb"],
                "fp16_size_mb": size["fp16_mb"],
                "int8_size_mb": size["int8_mb"]}
        if self.device.type == "cuda":
            r["cuda_allocated_mb"] = torch.cuda.memory_allocated(self.device) / 1024**2
            r["cuda_reserved_mb"]  = torch.cuda.memory_reserved(self.device) / 1024**2
        return r


# ==============================================================================
#  §25. UTILITY FUNCTIONS
# ==============================================================================

def count_parameters(model: nn.Module, only_trainable: bool = True) -> int:
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


@contextmanager
def inference_mode():
    with torch.inference_mode():
        yield



# ==============================================================================
#  §26. QUICK DEMO
# ==============================================================================

def run_basic_training_demo():
    """
    End-to-end demo: create a small model, train it for 50 steps on 5
    paragraphs of text, then generate some output.  Runs on any CPU
    in about 30–60 seconds.
    """
    set_seed(42)

    print("\n" + "═" * 70)
    print("  APLX_LLM v3.0.0 — Training Demo (small model, CPU-friendly)")
    print("═" * 70)

    # ── Config ────────────────────────────────────────────────────────────────
    demo_config = ModelConfig(
        vocab_size=4096, dim=256, n_layers=4, n_heads=8, n_kv_heads=8,
        max_seq_len=500_000, training_seq_len=128, sliding_window_size=128,
        use_sliding_window=True, rope_theta=500_000.0,
        use_gradient_checkpointing=True, dropout=0.1, use_flash_attention=False,
        model_name="APLX_Demo",
    )
    est = demo_config.estimate_parameters()
    print(f"\n  Config: dim={demo_config.dim}, layers={demo_config.n_layers}, "
          f"heads={demo_config.n_heads}, ~{est['total']:,} params")

    device = get_default_device()
    print(f"  Device: {device}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = AplexLLM(demo_config).to(device)
    print(model.parameter_summary())

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    training_texts = [
        "The transformer architecture has revolutionised natural language processing. "
        "Self-attention mechanisms allow the model to weigh the importance of each word "
        "in the context of every other word, capturing long-range dependencies that "
        "recurrent networks struggled with.",

        "Machine learning models improve through experience. By exposing a neural network "
        "to large quantities of data and computing gradients of a loss function, the weights "
        "gradually shift to produce better predictions on unseen examples.",

        "Python is a versatile programming language with a clean syntax that makes it ideal "
        "for both beginners and experts. Libraries such as PyTorch, NumPy, and Pandas power "
        "the majority of modern machine learning research.",

        "The history of artificial intelligence spans decades, from symbolic rule-based systems "
        "to modern deep learning. The AlexNet paper in 2012 marked a turning point, showing "
        "that GPUs and large datasets could unlock unprecedented accuracy in image recognition.",

        "Language is the primary medium through which humans share knowledge across time and "
        "space. Training a model on text allows it to absorb this accumulated knowledge and "
        "apply it to new situations, acting as a conversational memory for humanity.",
    ]

    tokenizer = BPETokenizer(vocab_size=demo_config.vocab_size)
    tokenizer.train(training_texts, verbose=True)

    # Register chat special tokens
    template = ChatTemplate(tokenizer)
    template.register_special_tokens(tokenizer)

    # Quick tokenizer round-trip test
    test = "The transformer is amazing!"
    enc  = tokenizer.encode(test)
    dec  = tokenizer.decode(enc)
    print(f"\n  Tokenizer round-trip: '{test}' → {len(enc)} tokens → '{dec}'")

    # ── Dataset & Training ────────────────────────────────────────────────────
    split   = max(1, int(len(training_texts) * 0.8))
    tr_ds   = TextDataset(training_texts[:split], tokenizer, demo_config.training_seq_len)
    ev_ds   = TextDataset(training_texts[split:], tokenizer, demo_config.training_seq_len)
    print(f"  Training samples: {len(tr_ds)}  |  Eval samples: {len(ev_ds)}")

    tr_cfg  = TrainingConfig(
        learning_rate=3e-4, min_learning_rate=1e-5,
        warmup_steps=5, total_steps=50,
        batch_size=1, gradient_accumulation_steps=4,
        use_amp=(device.type == "cuda"),
        log_interval=10, eval_interval=25, save_interval=100,
        output_dir="./aplx_demo_checkpoints",
    )

    trainer = Trainer(model, tr_cfg, tr_ds, ev_ds, tokenizer)
    trainer.train()

    # ── Perplexity ────────────────────────────────────────────────────────────
    ppl = evaluate_perplexity(model, tokenizer, training_texts[:2], seq_len=128)
    print(f"\n  Post-training perplexity: {ppl:.2f}")

    # ── Generation ────────────────────────────────────────────────────────────
    model.eval()
    gen     = TextGenerator(model, tokenizer)
    gen_cfg = GenerationConfig(max_new_tokens=40, temperature=0.8, top_k=40, do_sample=True)

    print("\n  Text generation examples:")
    for prompt in ["The transformer", "Machine learning"]:
        out = gen.generate_text(prompt, gen_cfg)
        print(f"  Prompt: '{prompt}'\n  → '{out[:120]}'\n")

    # ── Chat template demo ────────────────────────────────────────────────────
    messages = [
        {"role": "user", "content": "What is a transformer?"}
    ]
    chat_prompt = template.apply(messages, add_generation_prompt=True)
    print(f"  ChatML prompt:\n{'─'*50}\n{chat_prompt}{'─'*50}\n")

    # ── RoPE extrapolation test ───────────────────────────────────────────────
    print("  RoPE extrapolation test:")
    for n in [256, 1024, 4096]:
        try:
            model.rotary_emb(n)
            print(f"    ✓ {n:>5} positions OK")
        except Exception as e:
            print(f"    ✗ {n:>5} positions FAILED: {e}")

    print(f"\n{'═'*70}")
    print("  ✓  Demo complete!")
    print(f"{'═'*70}\n")
    return model, tokenizer


# ==============================================================================
#  §27. CLI ENTRY POINT  (NEW)
# ==============================================================================

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="aplx_llm",
        description="APLX_LLM v3.0 — Production-Grade Local Language Model",
    )
    sub = p.add_subparsers(dest="mode", help="Operation mode")

    # ── demo ──────────────────────────────────────────────────────────────────
    sub.add_parser("demo", help="Run the quick training demo")

    # ── train ─────────────────────────────────────────────────────────────────
    tr = sub.add_parser("train", help="Pre-train on plain text files")
    tr.add_argument("--data",       default="/home/r3nz/Desktop/Aplx/Aplx", help="Path to a text file or directory of text files (defaults to the Desktop Aplx folder)")
    tr.add_argument("--output",     default="./aplx_checkpoints", help="Checkpoint output directory")
    tr.add_argument("--steps",      type=int, default=0, help="Total training steps (0 = run until manually stopped)")
    tr.add_argument("--lr",         type=float, default=2e-4)
    tr.add_argument("--batch",      type=int,   default=1)
    tr.add_argument("--seq-len",    type=int,   default=256)
    tr.add_argument("--vocab-size", type=int,   default=2000)
    tr.add_argument("--dim",        type=int,   default=128)
    tr.add_argument("--layers",     type=int,   default=2)
    tr.add_argument("--heads",      type=int,   default=4)
    tr.add_argument("--resume",     default=None, help="Resume from checkpoint dir")
    tr.add_argument("--wandb",      action="store_true")
    tr.add_argument("--bf16",       action="store_true", help="Use bfloat16 instead of float16")

    # ── finetune ──────────────────────────────────────────────────────────────
    ft = sub.add_parser("finetune", help="Instruction fine-tune from JSONL data")
    ft.add_argument("--model",      required=True, help="Path to pretrained checkpoint dir")
    ft.add_argument("--data",       required=True, help="Path to .jsonl instruction file")
    ft.add_argument("--output",     default="./aplx_ft_checkpoints")
    ft.add_argument("--steps",      type=int, default=2000)
    ft.add_argument("--lr",         type=float, default=1e-4)
    ft.add_argument("--lora",       action="store_true", help="Use LoRA for parameter-efficient fine-tuning")
    ft.add_argument("--lora-rank",  type=int,   default=16)

    # ── chat ──────────────────────────────────────────────────────────────────
    ch = sub.add_parser("chat", help="Interactive chat with a trained model")
    ch.add_argument("--model",       required=True, help="Path to checkpoint dir")
    ch.add_argument("--temperature", type=float, default=0.8)
    ch.add_argument("--top-k",       type=int,   default=50)
    ch.add_argument("--top-p",       type=float, default=0.9)
    ch.add_argument("--max-tokens",  type=int,   default=256)
    ch.add_argument("--no-stream",   action="store_true")
    ch.add_argument("--system",      default=None, help="Override system prompt")

    # ── eval ──────────────────────────────────────────────────────────────────
    ev = sub.add_parser("eval", help="Evaluate perplexity on text data")
    ev.add_argument("--model",   required=True)
    ev.add_argument("--data",    default="/home/r3nz/Desktop/Aplx/Aplx", help="Path to a text file or directory of text files")
    ev.add_argument("--seq-len", type=int, default=512)

    # ── export ────────────────────────────────────────────────────────────────
    ex = sub.add_parser("export", help="Export model weights")
    ex.add_argument("--model",  required=True, help="Source checkpoint dir")
    ex.add_argument("--output", required=True, help="Output path")
    ex.add_argument("--format", choices=["pt", "safetensors", "hf"], default="pt")

    return p


def _resolve_data_path(data_path: str) -> Path:
    p = Path(data_path).expanduser()
    if p.is_absolute():
        return p

    cwd_candidate = (Path.cwd() / p).resolve()
    script_candidate = (Path(__file__).resolve().parent / p).resolve()

    if cwd_candidate.exists():
        return cwd_candidate
    if script_candidate.exists():
        return script_candidate

    # Fall back to the script-relative location so the user gets a stable path
    # even when the current working directory differs from the script location.
    return script_candidate


def _load_texts_from_path(data_path: str) -> List[str]:
    p = _resolve_data_path(data_path)
    texts = []

    supported_exts = {".txt", ".md", ".markdown", ".text", ".rst"}

    def _collect_texts_from_dir(target_dir: Path) -> List[str]:
        collected = []
        candidates = []
        for ext in supported_exts:
            candidates.extend(sorted(target_dir.glob(f"**/*{ext}")))
        for f in sorted(set(candidates), key=lambda x: str(x)):
            collected.append(f.read_text(encoding="utf-8", errors="replace"))
        return collected

    if p.is_dir():
        texts = _collect_texts_from_dir(p)
    elif p.is_file() and p.suffix.lower() in supported_exts:
        texts.append(p.read_text(encoding="utf-8", errors="replace"))
    elif not p.exists() and Path(data_path).suffix == "":
        fallback_dir = Path.cwd()
        if fallback_dir != Path(__file__).resolve().parent:
            texts = _collect_texts_from_dir(fallback_dir)
        if not texts:
            raise ValueError(f"No supported text files found in {data_path}")
    else:
        raise FileNotFoundError(
            f"Data path not found: {data_path} (resolved to {p})"
        )

    if not texts:
        raise ValueError(f"No supported text files found in {data_path}")
    return texts


def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_parser()
    args   = parser.parse_args(argv)

    if args.mode is None or args.mode == "demo":
        run_basic_training_demo()
        return

    # ── train ─────────────────────────────────────────────────────────────────
    if args.mode == "train":
        data_path = args.data or "/home/r3nz/Desktop/Aplx/Aplx"
        print("[TRAIN] Loading data...")
        texts = _load_texts_from_path(data_path)
        texts = preprocess_texts(texts)
        print(f"[TRAIN] {len(texts)} documents loaded after preprocessing.")

        device = get_default_device()
        cfg = ModelConfig(
            vocab_size=args.vocab_size,
            dim=args.dim,
            n_layers=args.layers,
            n_heads=args.heads,
            n_kv_heads=args.heads,
            training_seq_len=args.seq_len,
            sliding_window_size=args.seq_len,
            use_sliding_window=True,
            model_name="APLX_Custom",
        )
        cfg = build_model_config_for_device(device, cfg)
        model  = AplexLLM(cfg).to(device)
        print(model.parameter_summary())

        tokenizer = BPETokenizer(vocab_size=args.vocab_size)
        tokenizer.train(texts)

        split  = max(1, int(len(texts) * 0.9))
        tr_ds  = TextDataset(texts[:split], tokenizer, args.seq_len)
        ev_ds  = TextDataset(texts[split:], tokenizer, args.seq_len) if texts[split:] else None

        tr_cfg = TrainingConfig(
            learning_rate=args.lr,
            total_steps=args.steps,
            batch_size=args.batch,
            output_dir=args.output,
            use_amp=(device.type == "cuda"),
            amp_dtype="bfloat16" if args.bf16 else "float16",
            use_wandb=args.wandb,
            resume_from=args.resume,
        )

        trainer = Trainer(model, tr_cfg, tr_ds, ev_ds, tokenizer)
        if args.resume:
            trainer.load_checkpoint(os.path.join(args.resume, "checkpoint.pt"))
        trainer.train()
        return

    # ── finetune ──────────────────────────────────────────────────────────────
    if args.mode == "finetune":
        model, tokenizer = load_model(args.model)
        device = next(model.parameters()).device

        if args.lora:
            model = apply_lora(model, rank=args.lora_rank)

        assert tokenizer is not None, "Tokenizer required for fine-tuning"
        template = ChatTemplate(tokenizer)
        tr_ds    = InstructionDataset(args.data, tokenizer, template)
        print(f"[FINETUNE] {len(tr_ds)} instruction samples loaded.")

        tr_cfg = TrainingConfig(
            learning_rate=args.lr,
            total_steps=args.steps,
            output_dir=args.output,
            use_amp=(device.type == "cuda"),
        )
        trainer = Trainer(model, tr_cfg, tr_ds, tokenizer=tokenizer)
        trainer.train()

        if args.lora:
            model = merge_lora(model)
        save_model(model, os.path.join(args.output, "merged"), tokenizer)
        return

    # ── chat ──────────────────────────────────────────────────────────────────
    if args.mode == "chat":
        model, tokenizer = load_model(args.model)
        template = ChatTemplate(tokenizer)
        gen_cfg  = GenerationConfig(
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            max_new_tokens=args.max_tokens,
        )
        gen = TextGenerator(model, tokenizer)
        gen.chat(
            chat_template=template,
            gen_config=gen_cfg,
            system=args.system,
            stream=not args.no_stream,
        )
        return

    # ── eval ──────────────────────────────────────────────────────────────────
    if args.mode == "eval":
        model, tokenizer = load_model(args.model)
        assert tokenizer is not None, "Tokenizer required for evaluation"
        data_path = args.data or "/home/r3nz/Desktop/Aplx/Aplx"
        texts = _load_texts_from_path(data_path)
        ppl   = evaluate_perplexity(model, tokenizer, texts, seq_len=args.seq_len)
        print(f"\n  Perplexity: {ppl:.4f}")
        return

    # ── export ────────────────────────────────────────────────────────────────
    if args.mode == "export":
        model, tokenizer = load_model(args.model)
        if args.format == "safetensors":
            export_safetensors(model, args.output)
        elif args.format == "hf":
            export_huggingface_config(model, args.output, tokenizer)
        else:
            save_model(model, args.output, tokenizer)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
