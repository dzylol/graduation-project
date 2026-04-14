"""SMILES tokenizer for molecular property prediction."""

from __future__ import annotations

import functools
from typing import Dict, List, Optional, Tuple


smiles_token_tuple: tuple[str, ...] = (
    "(",
    ")",
    "[",
    "]",
    "=",
    "#",
    "%",
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "+",
    "-",
    "/",
    ".",
    ":",
    ";",
    "<",
    ">",
    "@",
    "B",
    "Br",
    "C",
    "Cl",
    "F",
    "H",
    "I",
    "N",
    "O",
    "P",
    "S",
    "Si",
    "Te",
    "Se",
    "At",
)

special_token_tuple: tuple[str, ...] = (
    "<pad>",
    "<unk>",
    "<bos>",
    "<eos>",
)


def build_default_vocab() -> Dict[str, int]:
    vocab: Dict[str, int] = {token: idx for idx, token in enumerate(special_token_tuple)}
    vocab.update(
        {char: idx + len(special_token_tuple) for idx, char in enumerate(smiles_token_tuple)}
    )
    return vocab


default_vocab: Dict[str, int] = build_default_vocab()
default_vocab_size: int = len(default_vocab)


class MoleculeTokenizer:
    """SMILES tokenizer with encode/decode methods."""

    def __init__(
        self,
        given_vocab_dict: Optional[Dict[str, int]] = None,
    ) -> None:
        if given_vocab_dict is None:
            self.vocab: Dict[str, int] = default_vocab
        else:
            self.vocab = given_vocab_dict
        self.inverse_vocab: Dict[int, str] = {idx: token for token, idx in self.vocab.items()}
        self.vocab_size: int = len(self.vocab)

    def encode(self, smiles: str, max_length: int = 512) -> Tuple[int, ...]:
        return tokenize_smiles_cached_internal(smiles, id(self.vocab), max_length)

    def decode(self, token_ids: List[int]) -> str:
        tokens: List[str] = []
        for token_id in token_ids:
            token: str = self.inverse_vocab.get(token_id, "")
            if token not in ["<pad>", "<unk>", "<bos>", "<eos>"]:
                tokens.append(token)
        return "".join(tokens)


@functools.lru_cache(maxsize=500000)
def tokenize_smiles_cached_internal(smiles: str, vocab_id: int, max_length: int) -> Tuple[int, ...]:
    """Tokenize SMILES string with caching.

    Note: vocab_id is passed to make cache key unique per vocab.
    Returns Tuple for hashability (required by lru_cache).
    """
    given_vocab_dict: Dict[str, int] = default_vocab if vocab_id == id(default_vocab) else {}
    tokens: List[int] = []
    i: int = 0
    while i < len(smiles):
        if i + 1 < len(smiles) and smiles[i : i + 2] in given_vocab_dict:
            tokens.append(given_vocab_dict[smiles[i : i + 2]])
            i += 2
        elif smiles[i] in given_vocab_dict:
            tokens.append(given_vocab_dict[smiles[i]])
            i += 1
        else:
            tokens.append(given_vocab_dict["<pad>"])
            i += 1
    pad_token_id: int = given_vocab_dict["<pad>"]
    if len(tokens) > max_length:
        return tuple(tokens[:max_length])
    return tuple(tokens + [pad_token_id] * (max_length - len(tokens)))
