"""Train-only vocab; unknown val characters map to UNK."""

from __future__ import annotations

from mamba_lm.tokenizer import UNK_CHAR, CharTokenizer


def test_from_text_includes_unk_and_encodes_unknown():
    tok = CharTokenizer.from_text("ab")
    assert UNK_CHAR in tok.stoi
    ids = tok.encode("abc")
    assert ids[0] == tok.stoi["a"]
    assert ids[1] == tok.stoi["b"]
    assert ids[2] == tok.unk_id
