"""Tokenizer + fixed-canvas codec for the AAV VR-VIII diffusion model.

Pure Python / numpy (no torch) so it can be used by data prep, training, and
sampling alike. Turns a variant amino-acid sequence into a fixed-length integer
canvas and back.

Vocabulary (22 tokens):
    ids 0..19  -> the 20 canonical amino acids ("ACDEFGHIKLMNPQRSTVWY")
    id  20     -> [gap]   clean, semantic "empty slot / deletion"
    id  21     -> [mask]  the absorbing corruption state used by the forward
                          diffusion; it is NEVER a clean target, so the model
                          predicts over the 21 "clean" ids only.
Keeping [gap] and [mask] DISTINCT is a deliberate MDLM-style choice (a [gap] is
real data the model must learn to emit; a [mask] only appears while corrupting).
This differs from AAVDiff (Liu 2024), which merges deletion/padding into a single
[del] token in a 21-token vocab; the tradeoff is one extra token in exchange for
an unambiguous absorbing state. We do not assert either choice is universally
better here.

Canvas (default L=56):
    Interleaved as [anchor_0, ins_0, anchor_1, ins_1, ..., anchor_27, ins_27]
    where each WT position ("anchor") owns one substitution slot followed by
    `insertions_per_gap` insertion slot(s). Empty insertion slots hold [gap].
    Insertion slots sit AFTER their anchor, so internal and trailing insertions
    are representable but a leading insertion (before anchor 0) is not. This is
    the right trade for Bryant: trailing insertions are 45% of rows whereas
    leading ones are only 0.11%. encode() raises on the 0.11% leading-insertion
    rows; the data-prep step is expected to skip-and-log them.

Bryant input convention (verified against Data/raw/bryant/allseqs_20191230.csv):
    Each sequence has exactly `n_anchor` UPPERCASE residues (the WT-scaffold
    positions, whether conserved or substituted) and zero or more lowercase
    residues marking insertions placed inline after their preceding anchor. The
    processed *_seq.pt tensors were upper-cased, which DESTROYS the insertion
    markers; insertion-bearing sequences must therefore be encoded from the raw
    (case-preserving) data. Pure substitution 28-mers encode fine either way.
"""
import json
from pathlib import Path

import numpy as np

from config import TokenizerConfig


class AAVTokenizer:
    """Vocab + canvas encode/decode for AAV VR-VIII variants."""

    def __init__(self, config: TokenizerConfig = None):
        self.config = config or TokenizerConfig()
        amino_acids = list(self.config.amino_acids)
        # ids: 0..19 amino acids, 20 [gap], 21 [mask]
        self.id_to_token = amino_acids + [self.config.gap_token, self.config.mask_token]
        self.token_to_id = {token: i for i, token in enumerate(self.id_to_token)}
        self.gap_id = self.token_to_id[self.config.gap_token]
        self.mask_id = self.token_to_id[self.config.mask_token]
        self.vocab_size = len(self.id_to_token)            # 22
        self.clean_vocab_size = self.vocab_size - 1         # 21 (everything but [mask])

        self.n_anchor = self.config.n_anchor
        self.insertions_per_gap = self.config.insertions_per_gap
        self.canvas_len = self.config.canvas_len
        self._slots_per_anchor = 1 + self.insertions_per_gap  # one sub slot + its insertion slots

    # -- canvas geometry ---------------------------------------------------
    def anchor_slot(self, anchor_index: int) -> int:
        """Canvas index of the substitution slot for the given anchor."""
        return anchor_index * self._slots_per_anchor

    def insertion_slots(self, anchor_index: int) -> list:
        """Canvas indices of the insertion slot(s) following the given anchor."""
        base = anchor_index * self._slots_per_anchor
        return [base + 1 + k for k in range(self.insertions_per_gap)]

    @property
    def anchor_slots(self) -> list:
        return [self.anchor_slot(i) for i in range(self.n_anchor)]

    @property
    def insertion_slot_index(self) -> np.ndarray:
        """Boolean (canvas_len,) mask: True where a slot is an insertion slot."""
        is_insertion = np.zeros(self.canvas_len, dtype=bool)
        for anchor_index in range(self.n_anchor):
            for slot in self.insertion_slots(anchor_index):
                is_insertion[slot] = True
        return is_insertion

    def clean_logits_mask(self) -> np.ndarray:
        """Boolean (vocab_size,) that is False for [mask] so sampling never emits it."""
        keep = np.ones(self.vocab_size, dtype=bool)
        keep[self.mask_id] = False
        return keep

    # -- encode / decode ---------------------------------------------------
    def encode(self, seq: str) -> np.ndarray:
        """Encode one variant string into a (canvas_len,) int array of token ids.

        Uppercase chars fill anchor slots in order; lowercase chars are
        insertions placed in the slot(s) after the most recent anchor. Unused
        insertion slots are filled with [gap].

        Raises ValueError on inputs that the current canvas cannot represent
        (wrong anchor count, leading insertion, or a gap with more insertions
        than `insertions_per_gap` allows) with a message naming the fix.
        """
        ids = np.full(self.canvas_len, self.gap_id, dtype=np.int64)
        anchor_index = -1          # index of the most recent anchor placed
        insertions_in_gap = 0      # insertions already placed in the current gap
        for char in seq:
            if char.isupper():
                anchor_index += 1
                insertions_in_gap = 0
                if anchor_index >= self.n_anchor:
                    raise ValueError(
                        f"sequence has >{self.n_anchor} anchor (uppercase) residues; "
                        f"if this came from an upper-cased *_seq.pt tensor its insertion "
                        f"markers were lost -- encode from the raw case-preserving data."
                    )
                if char not in self.token_to_id:
                    raise ValueError(f"unknown amino acid {char!r} in {seq!r}")
                ids[self.anchor_slot(anchor_index)] = self.token_to_id[char]
            elif char.islower():
                amino_acid = char.upper()
                if amino_acid not in self.token_to_id:
                    raise ValueError(f"unknown amino acid {char!r} in {seq!r}")
                if anchor_index < 0:
                    raise ValueError(
                        f"leading insertion before the first anchor is not representable "
                        f"in the current canvas: {seq!r}"
                    )
                if insertions_in_gap >= self.insertions_per_gap:
                    raise ValueError(
                        f"more than insertions_per_gap={self.insertions_per_gap} insertions "
                        f"in one gap of {seq!r}; raise TokenizerConfig.insertions_per_gap to "
                        f"design multi-residue insertions."
                    )
                slot = self.insertion_slots(anchor_index)[insertions_in_gap]
                ids[slot] = self.token_to_id[amino_acid]
                insertions_in_gap += 1
            else:
                raise ValueError(f"unexpected character {char!r} in {seq!r}")

        if anchor_index + 1 != self.n_anchor:
            raise ValueError(
                f"expected {self.n_anchor} anchor residues, got {anchor_index + 1} in {seq!r}"
            )
        return ids

    def encode_batch(self, seqs) -> np.ndarray:
        """Encode a list of sequences into a (batch, canvas_len) int array."""
        return np.stack([self.encode(s) for s in seqs])

    def decode(self, ids) -> str:
        """Decode a (canvas_len,) id array back to the amino-acid string.

        Reads slots in canvas order, dropping [gap]; raises if [mask] is present
        (a clean sequence should never contain the absorbing state).
        """
        ids = np.asarray(ids).reshape(-1)
        chars = []
        for token_id in ids.tolist():
            if token_id == self.gap_id:
                continue
            if token_id == self.mask_id:
                raise ValueError("decode() got a [mask] token; sequence is not fully denoised")
            chars.append(self.id_to_token[token_id])
        return "".join(chars)

    # -- persistence -------------------------------------------------------
    def save(self, path) -> None:
        Path(path).write_text(json.dumps({
            "token_to_id": self.token_to_id,
            "gap_token": self.config.gap_token,
            "mask_token": self.config.mask_token,
            "n_anchor": self.n_anchor,
            "insertions_per_gap": self.insertions_per_gap,
            "canvas_len": self.canvas_len,
            "vocab_size": self.vocab_size,
        }, indent=2))

    @classmethod
    def load(cls, path) -> "AAVTokenizer":
        meta = json.loads(Path(path).read_text())
        config = TokenizerConfig(
            amino_acids="".join(t for t in meta["token_to_id"]
                                if t not in (meta["gap_token"], meta["mask_token"])),
            n_anchor=meta["n_anchor"],
            insertions_per_gap=meta["insertions_per_gap"],
            gap_token=meta["gap_token"],
            mask_token=meta["mask_token"],
        )
        return cls(config)


if __name__ == "__main__":
    # Smoke test on real Bryant-format examples (case preserved: lowercase = insertion).
    tokenizer = AAVTokenizer()
    assert tokenizer.vocab_size == 22 and tokenizer.canvas_len == 56
    assert tokenizer.gap_id == 20 and tokenizer.mask_id == 21

    examples = [
        "ADEEIRTTNPVATEQYGEVSTNLQRGNR",        # pure substitution 28-mer
        "ADEEIRTTNPVATEQYGSVSTNPvQRGNR",        # one insertion (lowercase v)
        "ADEEIRTTNPVATEQYGSVSTnNLQRnGNR",       # two insertions in different gaps
    ]
    for seq in examples:
        ids = tokenizer.encode(seq)
        assert ids.shape == (56,)
        assert ids.min() >= 0 and ids.max() < 22
        assert tokenizer.mask_id not in ids.tolist()
        assert tokenizer.decode(ids) == seq.upper()   # round-trips to the biological string
    # substitution-only variants leave every insertion slot as [gap]
    sub_ids = tokenizer.encode(examples[0])
    assert all(sub_ids[s] == tokenizer.gap_id
               for s in range(56) if tokenizer.insertion_slot_index[s])

    # batch + clean-logits mask
    assert tokenizer.encode_batch(examples).shape == (3, 56)
    assert tokenizer.clean_logits_mask().sum() == 21

    # informative errors
    for bad in ["ADEEIRTTNPVATEQYGSVSTNPVQRGNR",  # 29 anchors (insertion marker lost)
                "vADEEIRTTNPVATEQYGEVSTNLQRGNR"]:  # leading insertion
        try:
            tokenizer.encode(bad)
            raise AssertionError(f"expected ValueError for {bad!r}")
        except ValueError:
            pass

    print("tokenizer smoke test passed:",
          f"vocab={tokenizer.vocab_size}, L={tokenizer.canvas_len}, "
          f"examples round-tripped={len(examples)}")
