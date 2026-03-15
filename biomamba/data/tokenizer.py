"""
SMILES Tokenizer for molecular property prediction.
Implements atom-level tokenization for SMILES strings.
"""

import re
from typing import List, Optional


class AtomTokenizer:
    """
    SMILES tokenizer that handles:
    - Single character atoms (C, O, N, etc.)
    - Double character elements (Cl, Br, etc.)
    - Ring closure digits (#, %)
    - Special characters ([, ], (, ), =, #, +, -, etc.)
    """

    def __init__(self, vocab: Optional[List[str]] = None):
        """
        Initialize tokenizer with optional vocabulary.

        Args:
            vocab: Pre-defined vocabulary list. If None, will build from SMILES.
        """
        if vocab is not None:
            self.vocab = vocab
        else:
            # Default vocabulary for SMILES
            self.vocab = self._build_default_vocab()

        self.token_to_id = {token: idx for idx, token in enumerate(self.vocab)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}
        self.pad_token = '[PAD]'
        self.unk_token = '[UNK]'
        self.cls_token = '[CLS]'
        self.sep_token = '[SEP]'
        self.mask_token = '[MASK]'

        # Special tokens
        self.pad_id = self.token_to_id.get(self.pad_token, 0)
        self.unk_id = self.token_to_id.get(self.unk_token, 1)
        self.cls_id = self.token_to_id.get(self.cls_token, 2)
        self.sep_id = self.token_to_id.get(self.sep_token, 3)
        self.mask_id = self.token_to_id.get(self.mask_token, 4)

    def _build_default_vocab(self) -> List[str]:
        """
        Build default vocabulary for SMILES tokenization.
        Includes common atoms, elements, and special tokens.
        """
        # Special tokens (must be first)
        special_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']

        # SMILES atoms (single letter)
        single_atoms = ['C', 'N', 'O', 'S', 'P', 'B', 'F', 'I', 'H']

        # SMILES atoms (double letter - transition metals and halogens)
        double_atoms = ['Cl', 'Br', 'Li', 'Na', 'K', 'Ca', 'Mg', 'Zn', 'Fe', 'Cu',
                       'Al', 'Si', 'Ge', 'Sn', 'Pb', 'Hg', 'Ag', 'Au', 'Pt', 'Ni']

        # Organic subset brackets
        bracket_atoms = ['[Cl]', '[Br]', '[C@@H]', '[C@H]', '[CH]', '[NH]',
                        '[O]', '[N]', '[S]', '[P]', '[B]', '[Si]', '[Se]']

        # Special characters
        special_chars = ['=', '#', '+', '-', '(', ')', '[', ']', '%',
                        '/', '\\', '@', '+', '-', ':', '.']

        # Ring digits (0-9)
        ring_digits = [str(i) for i in range(10)]

        # Combine all
        vocab = special_tokens + single_atoms + double_atoms + bracket_atoms + \
                special_chars + ring_digits

        return vocab

    def tokenize(self, smiles: str) -> List[str]:
        """
        Tokenize a SMILES string into tokens.

        Args:
            smiles: SMILES string to tokenize

        Returns:
            List of tokens
        """
        tokens = []
        i = 0
        smiles = smiles.strip()

        while i < len(smiles):
            # Check for two-character elements (Cl, Br, etc.)
            if i + 1 < len(smiles):
                two_char = smiles[i:i+2]
                if two_char in self.vocab and two_char not in ['[C', '[O', '[N', '[S', '[P', '[B', '[S']:
                    tokens.append(two_char)
                    i += 2
                    continue

            # Check for bracket atoms like [Cl], [Br], [N], etc.
            if smiles[i] == '[':
                # Find matching closing bracket
                j = i + 1
                while j < len(smiles) and smiles[j] != ']':
                    j += 1
                if j < len(smiles):
                    token = smiles[i:j+1]
                    tokens.append(token)
                    i = j + 1
                    continue

            # Check for % (ring closure with two digits)
            if smiles[i] == '%':
                if i + 2 < len(smiles):
                    tokens.append(smiles[i:i+3])
                    i += 3
                    continue

            # Single character
            token = smiles[i]
            if token in self.vocab or token.isalnum() or token in '()[]=#+-@%/\\.:':
                tokens.append(token)
            else:
                # Unknown character - skip or add as is
                tokens.append(token)
            i += 1

        return tokens

    def encode(self, smiles: str, max_length: int = 128,
               padding: bool = True, truncation: bool = True) -> List[int]:
        """
        Convert SMILES string to token IDs.

        Args:
            smiles: SMILES string
            max_length: Maximum sequence length
            padding: Whether to pad to max_length
            truncation: Whether to truncate if longer than max_length

        Returns:
            List of token IDs
        """
        tokens = self.tokenize(smiles)

        # Convert to IDs
        ids = []
        for token in tokens:
            if token in self.token_to_id:
                ids.append(self.token_to_id[token])
            else:
                # Try lowercase for atoms
                if token.lower() in self.token_to_id:
                    ids.append(self.token_to_id[token.lower()])
                else:
                    ids.append(self.unk_id)

        # Truncation
        if truncation and len(ids) > max_length - 2:  # Reserve for CLS and SEP
            ids = ids[:max_length - 2]

        # Add special tokens
        ids = [self.cls_id] + ids + [self.sep_id]

        # Padding
        if padding:
            if len(ids) < max_length:
                ids = ids + [self.pad_id] * (max_length - len(ids))

        return ids

    def decode(self, ids: List[int]) -> str:
        """
        Convert token IDs back to SMILES string.

        Args:
            ids: List of token IDs

        Returns:
            SMILES string
        """
        tokens = []
        for id_ in ids:
            if id_ == self.pad_id:
                continue
            if id_ in self.id_to_token:
                tokens.append(self.id_to_token[id_])
            else:
                tokens.append(self.unk_token)

        # Remove special tokens
        tokens = [t for t in tokens if t not in [self.cls_token, self.sep_token,
                                                   self.pad_token, self.mask_token]]

        return ''.join(tokens)

    def __len__(self) -> int:
        """Return vocabulary size."""
        return len(self.vocab)


def build_vocab_from_dataset(smiles_list: List[str]) -> List[str]:
    """
    Build vocabulary from a list of SMILES strings.

    Args:
        smiles_list: List of SMILES strings

    Returns:
        Vocabulary list
    """
    tokenizer = AtomTokenizer()
    vocab_set = set(tokenizer.vocab)

    for smiles in smiles_list:
        tokens = tokenizer.tokenize(smiles)
        vocab_set.update(tokens)

    # Sort vocabulary for consistency
    special_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
    vocab_list = special_tokens + sorted(list(vocab_set - set(special_tokens)))

    return vocab_list


if __name__ == "__main__":
    # Test tokenizer
    tokenizer = AtomTokenizer()

    test_smiles = [
        'CCO',           # Ethanol
        'c1ccccc1',      # Benzene
        'CC(=O)O',       # Acetic acid
        'CC(C)C',        # Isobutane
        '[Cl]CC(C)C',    # 2-chloro-2-methylpropane
    ]

    for smiles in test_smiles:
        tokens = tokenizer.tokenize(smiles)
        ids = tokenizer.encode(smiles)
        decoded = tokenizer.decode(ids)
        print(f"SMILES: {smiles}")
        print(f"Tokens: {tokens}")
        print(f"IDs: {ids}")
        print(f"Decoded: {decoded}")
        print("-" * 50)
