"""Character-level tokenizer. Vocab is derived from training text."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class CharTokenizer:
    """Maps characters <-> integer ids. Unknown chars raise KeyError on encode."""

    stoi: dict[str, int]
    itos: dict[int, str]

    @classmethod
    def from_text(cls, text: str) -> CharTokenizer:
        chars = sorted(set(text))
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for ch, i in stoi.items()}
        return cls(stoi=stoi, itos=itos)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.itos[int(i)] for i in ids)

    def to_dict(self) -> dict[str, dict]:
        return {"stoi": self.stoi, "itos": {str(k): v for k, v in self.itos.items()}}

    @classmethod
    def from_dict(cls, data: dict) -> CharTokenizer:
        stoi = {k: int(v) for k, v in data["stoi"].items()}
        itos = {int(k): v for k, v in data["itos"].items()}
        return cls(stoi=stoi, itos=itos)
