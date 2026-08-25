#!/usr/bin/env python3
"""
APLX AI 100M - Standalone Native LLM Engine
Self-contained decoder-only language model + byte tokenizer + trainer + native loader.

Architecture:
  vocab: 16,384
  hidden: 640
  layers: 16
  attention heads: 10
  KV heads: 5 (GQA)
  FFN: 1,920
  context: 2,048
  parameters: ~99.64M
"""

from __future__ import annotations

import dataclasses
import gc
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Generator

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Device Selection
# ---------------------------------------------------------------------------

def get_default_device(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class AplxConfig:
    vocab_size: int = 16384
    dim: int = 640
    n_layers: int = 16
    n_heads: int = 10
    n_kv_heads: int = 5
    ffn_dim: int = 1920
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    dropout: float = 0.0
    tie_embeddings: bool = False
    multiple_of: Optional[int] = None  # Added to prevent crashes from old configs
    model_name: str = "APLX_100M"
    model_version: str = "4.0.0"

    def __post_init__(self):
        if self.dim % self.n_heads != 0:
            raise ValueError("dim must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads != 0:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.ffn_dim <= 0:
            raise ValueError("ffn_dim must be positive")

    @property
    def head_dim(self) -> int:
        return self.dim // self.n_heads

    @property
    def estimated_parameters(self) -> int:
        embedding = self.vocab_size * self.dim
        lm_head = 0 if self.tie_embeddings else self.vocab_size * self.dim

        per_layer = (
            self.dim * self.dim
            + self.dim * (self.n_kv_heads * self.head_dim) * 2
            + self.dim * self.dim
            + 3 * self.dim * self.ffn_dim
            + 2 * self.dim
        )
        return embedding + lm_head + self.n_layers * per_layer + self.dim

    def save(self, path: Path):
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "AplxConfig":
        data = json.loads(path.read_text(encoding="utf-8"))
        # Filter out random junk keys so the dataclass doesn't throw a fit
        valid_keys = {f.name for f in dataclasses.fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data)


# ---------------------------------------------------------------------------
# Byte Tokenizer
# ---------------------------------------------------------------------------

class AplxTokenizer:
    """
    Deterministic UTF-8 byte tokenizer with special tokens.
    IDs 0..255 map to raw bytes.
    256 = PAD, 257 = BOS, 258 = EOS, 259 = UNK.
    """

    PAD = 256
    BOS = 257
    EOS = 258
    UNK = 259

    def __init__(self, vocab_size: int = 16384):
        if vocab_size < 260:
            raise ValueError("vocab_size must be >= 260")
        self.vocab_size = vocab_size

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        raw = text.encode("utf-8", errors="replace")
        ids = list(raw)
        if add_bos:
            ids.insert(0, self.BOS)
        if add_eos:
            ids.append(self.EOS)
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        data = bytearray()
        for idx in ids:
            idx = int(idx)
            if 0 <= idx <= 255:
                data.append(idx)
        return bytes(data).decode("utf-8", errors="replace")

    def save(self, path: Path):
        payload = {
            "type": "aplx_byte_tokenizer",
            "version": 1,
            "vocab_size": self.vocab_size,
            "special_tokens": {
                "pad": self.PAD,
                "bos": self.BOS,
                "eos": self.EOS,
                "unk": self.UNK,
            },
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "AplxTokenizer":
        if not path.exists():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(int(data.get("vocab_size", 16384)))


# ---------------------------------------------------------------------------
# Transformer Modules
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.float()
        y = y * torch.rsqrt(y.pow(2).mean(-1, keepdim=True) + self.eps)
        return y.to(dtype=x.dtype) * self.weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class RoPE(nn.Module):
    def __init__(self, dim: int, max_seq_len: int, theta: float):
        super().__init__()
        inv = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv, persistent=False)
        self.max_seq_len = max_seq_len
        self._cache_len = 0
        self.register_buffer("cos_cached", torch.empty(0), persistent=False)
        self.register_buffer("sin_cached", torch.empty(0), persistent=False)

    def _build(self, length: int, device: torch.device):
        t = torch.arange(length, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = emb.cos()
        self.sin_cached = emb.sin()
        self._cache_len = length

    def forward(self, length: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._cache_len < length or self.cos_cached.device != device:
            self._build(length, device)
        return self.cos_cached[:length], self.sin_cached[:length]


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: AplxConfig):
        super().__init__()
        self.cfg = cfg
        self.h = cfg.n_heads
        self.kv_h = cfg.n_kv_heads
        self.d = cfg.head_dim
        self.rep = self.h // self.kv_h

        self.q = nn.Linear(cfg.dim, cfg.dim, bias=False)
        self.k = nn.Linear(cfg.dim, self.kv_h * self.d, bias=False)
        self.v = nn.Linear(cfg.dim, self.kv_h * self.d, bias=False)
        self.o = nn.Linear(cfg.dim, cfg.dim, bias=False)

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        if self.rep == 1:
            return x
        b, h, s, d = x.shape
        return x[:, :, None, :, :].expand(b, h, self.rep, s, d).reshape(b, self.h, s, d)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_k: Optional[torch.Tensor] = None,
        past_v: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b, s, _ = x.shape

        q = self.q(x).view(b, s, self.h, self.d).transpose(1, 2)
        k = self.k(x).view(b, s, self.kv_h, self.d).transpose(1, 2)
        v = self.v(x).view(b, s, self.kv_h, self.d).transpose(1, 2)

        c = cos[:s].view(1, 1, s, self.d)
        si = sin[:s].view(1, 1, s, self.d)
        q = q * c + rotate_half(q) * si
        k = k * c + rotate_half(k) * si

        if past_k is not None:
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        full_k = self._repeat_kv(k)
        full_v = self._repeat_kv(v)

        scores = torch.matmul(q, full_k.transpose(-2, -1)) / math.sqrt(self.d)

        total = full_k.shape[-2]
        query_positions = torch.arange(total - s, total, device=x.device)
        key_positions = torch.arange(total, device=x.device)
        causal = key_positions.unsqueeze(0) <= query_positions.unsqueeze(1)
        scores = scores.masked_fill(~causal.view(1, 1, s, total), float("-inf"))

        weights = F.softmax(scores.float(), dim=-1).to(dtype=q.dtype)
        out = torch.matmul(weights, full_v)
        out = out.transpose(1, 2).contiguous().view(b, s, self.cfg.dim)
        return self.o(out), k, v


class SwiGLU(nn.Module):
    def __init__(self, cfg: AplxConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.dim, cfg.ffn_dim, bias=False)
        self.up = nn.Linear(cfg.dim, cfg.ffn_dim, bias=False)
        self.down = nn.Linear(cfg.ffn_dim, cfg.dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: AplxConfig):
        super().__init__()
        self.norm1 = RMSNorm(cfg.dim, cfg.norm_eps)
        self.attn = CausalSelfAttention(cfg)
        self.norm2 = RMSNorm(cfg.dim, cfg.norm_eps)
        self.ffn = SwiGLU(cfg)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_k: Optional[torch.Tensor] = None,
        past_v: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a, k, v = self.attn(self.norm1(x), cos, sin, past_k, past_v)
        x = x + a
        x = x + self.ffn(self.norm2(x))
        return x, k, v


class Aplx100M(nn.Module):
    def __init__(self, cfg: AplxConfig):
        super().__init__()
        self.config = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.dim)
        self.rope = RoPE(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm = RMSNorm(cfg.dim, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.dim, cfg.vocab_size, bias=False)

        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        past: Optional[List[Optional[Tuple[torch.Tensor, torch.Tensor]]]] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[Tuple[torch.Tensor, torch.Tensor]]]:
        b, s = input_ids.shape
        if s > self.config.max_seq_len:
            input_ids = input_ids[:, -self.config.max_seq_len:]
            s = input_ids.shape[1]
            past = None

        past = past or [None] * len(self.blocks)
        past_len = past[0][0].shape[2] if past and past[0] is not None else 0

        total_len = past_len + s
        if total_len > self.config.max_seq_len:
            trim = total_len - self.config.max_seq_len
            new_past = []
            for pair in past:
                if pair is None:
                    new_past.append(None)
                else:
                    pk, pv = pair
                    new_past.append((pk[:, :, trim:, :], pv[:, :, trim:, :]))
            past = new_past
            past_len -= trim
            total_len = past_len + s

        x = self.embed(input_ids)
        cos, sin = self.rope(total_len, x.device)
        new_past = []

        for i, block in enumerate(self.blocks):
            pair = past[i]
            pk, pv = pair if pair is not None else (None, None)
            x, k, v = block(
                x,
                cos[past_len:],
                sin[past_len:],
                pk,
                pv,
            )
            new_past.append((k, v))

        logits = self.lm_head(self.norm(x))
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return logits, loss, new_past


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

@dataclass
class GenerationConfig:
    max_new_tokens: int = 150
    temperature: float = 0.7
    top_k: int = 40
    top_p: float = 0.9
    repetition_penalty: float = 1.05
    do_sample: bool = True


def _sample(logits: torch.Tensor, cfg: GenerationConfig, generated_ids: Optional[List[int]] = None) -> torch.Tensor:
    logits = logits.float()

    if cfg.repetition_penalty != 1.0 and generated_ids:
        for token_id in set(generated_ids):
            if 0 <= token_id < logits.shape[-1]:
                if logits[0, token_id] < 0:
                    logits[0, token_id] *= cfg.repetition_penalty
                else:
                    logits[0, token_id] /= cfg.repetition_penalty

    if cfg.temperature <= 0 or not cfg.do_sample:
        return torch.argmax(logits, dim=-1)

    logits = logits / max(cfg.temperature, 1e-5)

    if cfg.top_k > 0:
        k = min(cfg.top_k, logits.shape[-1])
        values, _ = torch.topk(logits, k)
        cutoff = values[..., -1, None]
        logits = torch.where(logits < cutoff, torch.full_like(logits, -float("inf")), logits)

    if 0.0 < cfg.top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        remove = cumulative > cfg.top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, -float("inf"))
        logits = torch.full_like(logits, -float("inf"))
        logits.scatter_(dim=-1, index=sorted_indices, src=sorted_logits)

    return torch.multinomial(torch.softmax(logits, dim=-1), 1).squeeze(-1)


# ---------------------------------------------------------------------------
# Standalone Native Engine
# ---------------------------------------------------------------------------

class AplxNativeEngine:
    def __init__(
        self,
        path: str = "aplx_checkpoints/best",
        device: Optional[str] = None,
        dtype: Optional[str] = None,
    ):
        self.base_dir = Path(__file__).resolve().parent
        self.path = self._resolve_path(path)
        self.device = torch.device(device) if device else get_default_device()

        self.config = self._load_config()
        self.tokenizer = self._load_tokenizer()
        self.model = Aplx100M(self.config).to(self.device)

        self.loaded_checkpoint = False
        self.checkpoint_path = None
        self._load_weights()

        self.model.eval()
        if dtype:
            self.model = self.model.to(dtype=getattr(torch, dtype))

    def _resolve_path(self, path: str) -> Path:
        p = Path(path)
        if p.is_absolute():
            return p
        candidates = [
            p,
            self.base_dir / p,
            self.base_dir / "aplx_checkpoints" / "best",
        ]
        for c in candidates:
            if c.exists():
                return c
        return self.base_dir / p

    def _load_config(self) -> AplxConfig:
        cfg_path = self.path / "config.json"
        if cfg_path.exists():
            try:
                return AplxConfig.load(cfg_path)
            except Exception as e:
                print(f"[APLX] config load failed, using built-in config: {e}")
        return AplxConfig()

    def _load_tokenizer(self) -> AplxTokenizer:
        tok_path = self.path / "tokenizer.json"
        try:
            return AplxTokenizer.load(tok_path)
        except Exception:
            return AplxTokenizer(self.config.vocab_size)

    def _load_weights(self):
        candidates = [
            self.path / "model.pt",
            self.path / "model.pth",
            self.path / "checkpoint.pt",
            self.path / "checkpoint.pth",
        ]

        found = next((p for p in candidates if p.exists()), None)
        if found is None:
            return

        try:
            checkpoint = torch.load(found, map_location=self.device, weights_only=False)
            state = checkpoint

            if isinstance(checkpoint, dict):
                for key in ("model_state_dict", "state_dict", "model", "weights"):
                    if key in checkpoint and isinstance(checkpoint[key], dict):
                        state = checkpoint[key]
                        break

            cleaned = {}
            for key, value in state.items():
                if key.startswith("module."):
                    key = key[7:]
                if key.startswith("model."):
                    key = key[6:]
                cleaned[key] = value

            missing, unexpected = self.model.load_state_dict(cleaned, strict=False)
            if missing:
                print(f"[APLX] checkpoint loaded with {len(missing)} missing tensors")
            if unexpected:
                print(f"[APLX] checkpoint has {len(unexpected)} unused tensors")

            self.loaded_checkpoint = True
            self.checkpoint_path = found
        except Exception as e:
            print(f"[APLX] checkpoint load failed: {e}")

    @property
    def is_loaded(self) -> bool:
        return self.loaded_checkpoint

    @property
    def num_parameters(self) -> int:
        return self.model.num_parameters

    def save_package(self, path: Optional[str] = None):
        target = Path(path) if path else self.path
        target.mkdir(parents=True, exist_ok=True)
        self.config.save(target / "config.json")
        self.tokenizer.save(target / "tokenizer.json")
        torch.save(
            {
                "format": "aplx_native_100m_v1",
                "config": asdict(self.config),
                "model_state_dict": self.model.state_dict(),
            },
            target / "model.pt",
        )

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_tokens: int = 150,
        temperature: float = 0.7,
        top_k: int = 40,
        top_p: float = 0.9,
        repetition_penalty: float = 1.05,
        do_sample: bool = True,
        system: Optional[str] = None,
    ) -> str:
        pieces = []
        if system:
            pieces.append(f"System: {system}\n")
        pieces.append(f"User: {prompt}\nAssistant:")
        text = "".join(pieces)

        ids = self.tokenizer.encode(text, add_bos=True)
        ids = ids[-(self.config.max_seq_len - 1):]
        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)

        gen = GenerationConfig(
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
        )

        past = None
        generated = []

        for _ in range(max_tokens):
            logits, _, past = self.model(input_ids, past=past)
            next_id = _sample(logits[:, -1, :], gen, generated_ids=generated)
            token = int(next_id.item())

            if token == self.tokenizer.EOS:
                break

            generated.append(token)
            input_ids = next_id.view(1, 1)

        return self.tokenizer.decode(generated).strip()

    @torch.inference_mode()
    def stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        text = self.generate(prompt, **kwargs)
        for chunk in re.findall(r"\S+\s*", text):
            yield chunk

    def chat(self, prompt: str, **kwargs) -> str:
        return self.generate(prompt, **kwargs)


# ---------------------------------------------------------------------------
# Loader Contract
# ---------------------------------------------------------------------------

NATIVE_ENGINE_LOADER = True
IS_NATIVE_ENGINE = True
ENGINE_NAME = "APLX 100M Native Engine"
ENGINE_VERSION = "4.0.0"

_ENGINE: Optional[AplxNativeEngine] = None


def load_native_aplx_engine(path: str = "aplx_checkpoints/best", device: Optional[str] = None, **kwargs) -> Tuple[bool, Any]:
    global _ENGINE
    try:
        _ENGINE = AplxNativeEngine(path=path, device=device, **kwargs)
        return True, _ENGINE
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def load_engine(path: str = "aplx_checkpoints/best", device: Optional[str] = None, **kwargs):
    ok, engine = load_native_aplx_engine(path, device, **kwargs)
    if not ok:
        raise RuntimeError(engine)
    return engine


def get_engine(path: str = "aplx_checkpoints/best", device: Optional[str] = None, **kwargs):
    global _ENGINE
    if _ENGINE is None:
        return load_engine(path, device, **kwargs)
    return _ENGINE


def generate(prompt: str, **kwargs) -> str:
    return get_engine().generate(prompt, **kwargs)


def chat(prompt: str, **kwargs) -> str:
    return get_engine().chat(prompt, **kwargs)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

def train_from_text_file(
    text_file: str,
    output_dir: str = "aplx_checkpoints/best",
    steps: int = 1000,
    seq_len: int = 256,
    lr: float = 3e-4,
    grad_accum: int = 4,
    device: Optional[str] = None,
):
    """Train the standalone model on a plain text file."""
    engine = AplxNativeEngine(output_dir, device=device)
    text = Path(text_file).read_text(encoding="utf-8", errors="ignore")
    ids = engine.tokenizer.encode(text, add_bos=True, add_eos=True)
    if len(ids) < seq_len + 1:
        raise ValueError(f"Training text is too short: need at least {seq_len + 1} tokens")

    model = engine.model
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95), weight_decay=0.1)

    amp_enabled = engine.device.type == "cuda"
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0

        for _ in range(max(1, grad_accum)):
            start = torch.randint(0, len(ids) - seq_len, (1,)).item()
            chunk = torch.tensor(ids[start:start + seq_len], dtype=torch.long, device=engine.device)
            inp = chunk.unsqueeze(0)
            labels = chunk.unsqueeze(0)

            with torch.autocast(device_type=engine.device.type, dtype=torch.float16, enabled=amp_enabled):
                _, loss, _ = model(inp, labels=labels)
                loss = loss / max(1, grad_accum)

            scaler.scale(loss).backward()
            total_loss += float(loss.detach())

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if step == 1 or step % 10 == 0:
            print(f"[APLX train] step {step}/{steps} loss={total_loss:.4f}")

    model.eval()
    engine.loaded_checkpoint = True
    engine.save_package(output_dir)
    print(f"[APLX train] saved to {output_dir}")
    return engine


def print_model_info():
    cfg = AplxConfig()
    print("=" * 64)
    print(" APLX AI 100M - STANDALONE NATIVE ENGINE")
    print("=" * 64)
    print(f" Parameters: {cfg.estimated_parameters:,}")
    print(f" Layers:     {cfg.n_layers}")
    print(f" Hidden:     {cfg.dim}")
    print(f" Heads:      {cfg.n_heads} (KV: {cfg.n_kv_heads})")
    print(f" FFN:        {cfg.ffn_dim}")
    print(f" Context:    {cfg.max_seq_len}")
    print(f" Device:     {get_default_device()}")
    print("=" * 64)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="APLX 100M standalone native engine")
    parser.add_argument("--info", action="store_true")
    parser.add_argument("--path", default="aplx_checkpoints/best")
    parser.add_argument("--device", default=None)
    parser.add_argument("--query", default=None)
    parser.add_argument("--train-file", default=None)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--max-tokens", type=int, default=80)
    args = parser.parse_args()

    if args.info:
        print_model_info()

    if args.train_file:
        train_from_text_file(args.train_file, args.path, steps=args.steps, device=args.device)

    if args.query:
        engine = load_engine(args.path, args.device)
        print(engine.generate(args.query, max_tokens=args.max_tokens))