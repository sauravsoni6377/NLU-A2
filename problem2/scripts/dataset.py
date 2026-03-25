"""
Character-Level Name Dataset
===============================
Handles loading names, building character vocabulary, and preparing
training data for character-level sequence models.

Each name is treated as a sequence of characters. We add special tokens:
- SOS (Start of Sequence): signals the beginning of a name
- EOS (End of Sequence): signals the end of a name

For training, each name "Aarav" becomes:
  Input:  [SOS, A, a, r, a, v]
  Target: [A, a, r, a, v, EOS]
"""

import os
import torch
from torch.utils.data import Dataset

# Special tokens for sequence boundaries
SOS_TOKEN = '<SOS>'  # Start of sequence marker
EOS_TOKEN = '<EOS>'  # End of sequence marker
PAD_TOKEN = '<PAD>'  # Padding for batch alignment


class CharVocabulary:
    """
    Maps characters to integer indices and vice versa.

    Built from the training names, this vocabulary includes all unique
    characters plus special tokens (SOS, EOS, PAD). The vocabulary is
    essential for converting between string names and tensor representations.
    """

    def __init__(self):
        self.char2idx = {}
        self.idx2char = {}
        self.vocab_size = 0

    def build(self, names):
        """
        Builds the character vocabulary from a list of names.

        Adds special tokens first (PAD=0, SOS=1, EOS=2), then all unique
        characters found in the names, sorted alphabetically for consistency.

        Args:
            names: List of name strings
        """
        # Collect all unique characters across all names
        chars = set()
        for name in names:
            chars.update(name)

        # Add special tokens with fixed indices (PAD=0 for masking convenience)
        self.char2idx[PAD_TOKEN] = 0
        self.char2idx[SOS_TOKEN] = 1
        self.char2idx[EOS_TOKEN] = 2

        # Add all unique characters in sorted order for deterministic mapping
        for idx, char in enumerate(sorted(chars), start=3):
            self.char2idx[char] = idx

        # Build reverse mapping (index -> character)
        self.idx2char = {idx: char for char, idx in self.char2idx.items()}
        self.vocab_size = len(self.char2idx)

    @property
    def sos_idx(self):
        """Index of the Start-of-Sequence token."""
        return self.char2idx[SOS_TOKEN]

    @property
    def eos_idx(self):
        """Index of the End-of-Sequence token."""
        return self.char2idx[EOS_TOKEN]

    @property
    def pad_idx(self):
        """Index of the Padding token."""
        return self.char2idx[PAD_TOKEN]

    def encode(self, name):
        """
        Encodes a name string into a list of character indices.
        Does NOT add SOS/EOS tokens (caller decides).

        Args:
            name: String name to encode

        Returns:
            List of integer character indices
        """
        return [self.char2idx[c] for c in name]

    def decode(self, indices):
        """
        Decodes a list of character indices back into a string.
        Stops at the first EOS token if present.

        Args:
            indices: List or tensor of integer indices

        Returns:
            Decoded name string (without special tokens)
        """
        chars = []
        for idx in indices:
            idx = idx.item() if torch.is_tensor(idx) else idx
            if idx == self.eos_idx:
                break
            if idx == self.sos_idx or idx == self.pad_idx:
                continue
            chars.append(self.idx2char.get(idx, '?'))
        return ''.join(chars)


class NamesDataset(Dataset):
    """
    PyTorch Dataset for character-level name generation.

    Each sample consists of:
    - input_seq: [SOS, c1, c2, ..., cn] (teacher forcing input)
    - target_seq: [c1, c2, ..., cn, EOS] (expected output)

    The model learns to predict each next character given the previous ones,
    with SOS starting the generation and EOS terminating it.
    """

    def __init__(self, names, vocab):
        """
        Args:
            names: List of name strings
            vocab: CharVocabulary object with char<->index mappings
        """
        self.names = names
        self.vocab = vocab

        # Pre-encode all names for efficiency
        self.encoded = []
        for name in names:
            chars = vocab.encode(name)
            # Input: SOS + characters, Target: characters + EOS
            input_seq = [vocab.sos_idx] + chars
            target_seq = chars + [vocab.eos_idx]
            self.encoded.append((
                torch.tensor(input_seq, dtype=torch.long),
                torch.tensor(target_seq, dtype=torch.long)
            ))

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        return self.encoded[idx]


def collate_fn(batch):
    """
    Custom collation function for DataLoader that pads variable-length
    name sequences to the same length within a batch.

    Names have different lengths (e.g., "Om" vs "Balasubramanian"),
    so we pad shorter sequences to match the longest in the batch.

    Args:
        batch: List of (input_tensor, target_tensor) tuples

    Returns:
        Tuple of (padded_inputs, padded_targets, lengths) tensors
    """
    # Sort batch by sequence length (descending) for pack_padded_sequence
    batch.sort(key=lambda x: len(x[0]), reverse=True)

    inputs, targets = zip(*batch)
    lengths = [len(s) for s in inputs]

    # Pad sequences to max length in this batch (PAD_IDX = 0)
    max_len = max(lengths)
    padded_inputs = torch.zeros(len(batch), max_len, dtype=torch.long)
    padded_targets = torch.zeros(len(batch), max_len, dtype=torch.long)

    for i, (inp, tgt) in enumerate(zip(inputs, targets)):
        padded_inputs[i, :len(inp)] = inp
        padded_targets[i, :len(tgt)] = tgt

    return padded_inputs, padded_targets, torch.tensor(lengths, dtype=torch.long)


def load_names(filepath):
    """
    Loads names from a text file (one name per line).
    Strips whitespace and filters out empty lines.

    Args:
        filepath: Path to TrainingNames.txt

    Returns:
        List of name strings
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        names = [line.strip() for line in f if line.strip()]
    return names
