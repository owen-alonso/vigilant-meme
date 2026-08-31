"""Tiny Shakespeare download and char-level dataset."""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import torch
from torch.utils.data import Dataset

from mamba_lm.paths import anchor_to_repo
from mamba_lm.tokenizer import CharTokenizer


TINY_SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
    "tinyshakespeare/input.txt"
)
TINY_SHAKESPEARE_SHA256 = (
    "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"
)
TINY_SHAKESPEARE_MIN_BYTES = 1_000_000


def _validate_tiny_shakespeare(data: bytes) -> None:
    if len(data) < TINY_SHAKESPEARE_MIN_BYTES:
        raise ValueError(
            f"tiny shakespeare download too small ({len(data)} bytes); "
            "refusing to cache a truncated or error response"
        )
    digest = hashlib.sha256(data).hexdigest()
    if digest != TINY_SHAKESPEARE_SHA256:
        raise ValueError(
            "tiny shakespeare SHA-256 mismatch; "
            f"expected {TINY_SHAKESPEARE_SHA256}, got {digest}"
        )


def download_tiny_shakespeare(data_dir: str | Path) -> Path:
    data_dir = anchor_to_repo(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / "tinyshakespeare.txt"
    if path.exists() and path.stat().st_size >= TINY_SHAKESPEARE_MIN_BYTES:
        existing = path.read_bytes()
        try:
            _validate_tiny_shakespeare(existing)
            return path
        except ValueError:
            path.unlink(missing_ok=True)

    try:
        with urlopen(TINY_SHAKESPEARE_URL, timeout=60) as response:
            if getattr(response, "status", 200) != 200:
                raise ValueError(f"HTTP {response.status} fetching tiny shakespeare")
            text = response.read()
    except (HTTPError, URLError) as exc:
        raise RuntimeError(f"failed to download tiny shakespeare: {exc}") from exc

    _validate_tiny_shakespeare(text)
    with tempfile.NamedTemporaryFile(dir=data_dir, delete=False) as tmp:
        tmp.write(text)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)
    return path


def load_corpus(data_dir: str | Path) -> str:
    path = download_tiny_shakespeare(data_dir)
    return path.read_text(encoding="utf-8")


def train_val_split(text: str, val_fraction: float) -> tuple[str, str]:
    if not 0.0 < val_fraction < 1.0:
        raise ValueError("val_fraction must be in (0, 1)")
    split = int(len(text) * (1.0 - val_fraction))
    split = max(1, min(len(text) - 1, split))
    return text[:split], text[split:]


class CharLMDataset(Dataset):
    """Fixed non-overlapping chunks. ``__getitem__`` returns (x, y) int64 tokens."""

    def __init__(self, token_ids: list[int] | torch.Tensor, seq_len: int) -> None:
        if seq_len < 1:
            raise ValueError("seq_len must be positive")
        ids = torch.as_tensor(token_ids, dtype=torch.long)
        usable = ((len(ids) - 1) // seq_len) * seq_len
        if usable < seq_len:
            raise ValueError(
                f"corpus too short ({len(ids)} tokens) for seq_len={seq_len}"
            )
        self.seq_len = seq_len
        # +1 so each chunk can form next-token targets.
        self.ids = ids[: usable + 1]

    def __len__(self) -> int:
        return (len(self.ids) - 1) // self.seq_len

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = index * self.seq_len
        chunk = self.ids[start : start + self.seq_len + 1]
        return chunk[:-1], chunk[1:]


def build_datasets(
    data_dir: str | Path,
    seq_len: int,
    val_fraction: float = 0.1,
) -> tuple[CharTokenizer, CharLMDataset, CharLMDataset]:
    text = load_corpus(data_dir)
    train_text, val_text = train_val_split(text, val_fraction)
    tokenizer = CharTokenizer.from_text(train_text)
    train_ids = tokenizer.encode(train_text)
    val_ids = tokenizer.encode(val_text)
    return tokenizer, CharLMDataset(train_ids, seq_len), CharLMDataset(val_ids, seq_len)
