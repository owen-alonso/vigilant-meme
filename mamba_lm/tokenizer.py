"""Character-level tokenizer. Vocab is derived from training text."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

UNK_CHAR = "\ufffd"


@dataclass
class CharTokenizer:
    """Maps characters <-> integer ids.

    Characters missing from the training vocab map to ``unk_id`` rather than
    raising, so a val split that contains a rare glyph still encodes.
    """

    stoi: dict[str, int]
    itos: dict[int, str]
    unk_id: int = 0

    @classmethod
    def from_text(cls, text: str) -> CharTokenizer:
        chars = sorted(set(text))
        if UNK_CHAR not in chars:
            chars = [UNK_CHAR] + chars
        stoi = {ch: i for i, ch in enumerate(chars)}
        itos = {i: ch for ch, i in stoi.items()}
        return cls(stoi=stoi, itos=itos, unk_id=stoi[UNK_CHAR])

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> list[int]:
        unk = self.unk_id
        return [self.stoi.get(ch, unk) for ch in text]

    def decode(self, ids: Iterable[int]) -> str:
        return "".join(self.itos.get(int(i), UNK_CHAR) for i in ids)

    def to_dict(self) -> dict[str, dict | int]:
        return {
            "stoi": self.stoi,
            "itos": {str(k): v for k, v in self.itos.items()},
            "unk_id": self.unk_id,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CharTokenizer:
        stoi = {k: int(v) for k, v in data["stoi"].items()}
        itos = {int(k): v for k, v in data["itos"].items()}
        unk_id = int(data.get("unk_id", stoi.get(UNK_CHAR, 0)))
        return cls(stoi=stoi, itos=itos, unk_id=unk_id)
