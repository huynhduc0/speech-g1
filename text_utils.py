"""
text_utils.py
─────────────
• Tokenizer  – character-level (Russian lower + space + blank)
• number_to_text()  – integer → Russian spoken form
• text_to_number()  – Russian spoken form → integer  (used at inference)
• ctc_greedy_decode()  – raw log-prob tensor → token id list
"""

from __future__ import annotations
import re
from typing import List

# ── Vocabulary ────────────────────────────────────────────────────────────────
BLANK_TOKEN = "<blank>"
SPACE_TOKEN = " "

# 33 Russian lowercase letters (а-я + ё inserted in canonical position)
_RUSSIAN = list("абвгдеёжзийклмнопрстуфхцчшщъыьэюя")
VOCAB = [BLANK_TOKEN] + [SPACE_TOKEN] + _RUSSIAN   # blank=0, space=1

CHAR2ID = {c: i for i, c in enumerate(VOCAB)}
ID2CHAR = {i: c for c, i in CHAR2ID.items()}
VOCAB_SIZE = len(VOCAB)   # 35
BLANK_ID = 0
SPACE_ID = 1


class Tokenizer:
    def encode(self, text: str) -> List[int]:
        """Russian lowercase text → token ids (unknown chars silently dropped)."""
        return [CHAR2ID[c] for c in text.lower() if c in CHAR2ID]

    def decode(self, ids: List[int]) -> str:
        return "".join(ID2CHAR.get(i, "") for i in ids)


# ── Russian number words ──────────────────────────────────────────────────────
_ONES_M = ["", "один", "два", "три", "четыре", "пять",
           "шесть", "семь", "восемь", "девять"]
_ONES_F = ["", "одна", "две", "три", "четыре", "пять",
           "шесть", "семь", "восемь", "девять"]
_TEENS  = ["десять", "одиннадцать", "двенадцать", "тринадцать",
           "четырнадцать", "пятнадцать", "шестнадцать", "семнадцать",
           "восемнадцать", "девятнадцать"]
_TENS   = ["", "", "двадцать", "тридцать", "сорок", "пятьдесят",
           "шестьдесят", "семьдесят", "восемьдесят", "девяносто"]
_HUNDS  = ["", "сто", "двести", "триста", "четыреста", "пятьсот",
           "шестьсот", "семьсот", "восемьсот", "девятьсот"]


def _chunk_to_text(n: int, feminine: bool = False) -> str:
    """Convert 1–999 → Russian word string."""
    assert 1 <= n <= 999
    parts: List[str] = []
    h = n // 100
    r = n % 100
    if h:
        parts.append(_HUNDS[h])
    if 10 <= r <= 19:
        parts.append(_TEENS[r - 10])
    else:
        t, u = r // 10, r % 10
        if t:
            parts.append(_TENS[t])
        if u:
            parts.append(_ONES_F[u] if feminine else _ONES_M[u])
    return " ".join(p for p in parts if p)


def _thou_suffix(n: int) -> str:
    """Return тысяча/тысячи/тысяч for the thousands count n."""
    last2, last1 = n % 100, n % 10
    if 11 <= last2 <= 19:
        return "тысяч"
    if last1 == 1:
        return "тысяча"
    if 2 <= last1 <= 4:
        return "тысячи"
    return "тысяч"


def number_to_text(n: int) -> str:
    """Convert integer 1 000–999 999 → Russian spoken form."""
    thou = n // 1000
    rem  = n % 1000
    parts: List[str] = []
    parts.append(_chunk_to_text(thou, feminine=True))
    parts.append(_thou_suffix(thou))
    if rem:
        parts.append(_chunk_to_text(rem, feminine=False))
    return " ".join(p for p in parts if p)


# ── Inference: Russian text → integer ────────────────────────────────────────
_WORD2NUM: dict[str, int] = {}
for _i, _w in enumerate(_ONES_M[1:], 1):
    _WORD2NUM[_w] = _i
for _i, _w in enumerate(_ONES_F[1:], 1):
    _WORD2NUM[_w] = _i
for _i, _w in enumerate(_TEENS, 10):
    _WORD2NUM[_w] = _i
for _i, _w in enumerate(_TENS[2:], 20):
    _WORD2NUM[_w] = _i * 10  # already correct: двадцать→20, тридцать→30 …
# fix: _TENS enumerates with step 10 but enumerate starts at 0; rebuild properly
_WORD2NUM.clear()
_TENS_VALS  = [0, 0, 20, 30, 40, 50, 60, 70, 80, 90]
_HUNDS_VALS = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900]
for _i in range(1, 10):
    _WORD2NUM[_ONES_M[_i]] = _i
    _WORD2NUM[_ONES_F[_i]] = _i
for _i, _w in enumerate(_TEENS, 10):
    _WORD2NUM[_w] = _i
for _i in range(2, 10):
    _WORD2NUM[_TENS[_i]] = _TENS_VALS[_i]
for _i in range(1, 10):
    _WORD2NUM[_HUNDS[_i]] = _HUNDS_VALS[_i]

_THOU_WORDS = {"тысяча", "тысячи", "тысяч"}


def _parse_chunk(tokens: List[str]) -> int:
    total = 0
    for t in tokens:
        total += _WORD2NUM.get(t, 0)
    return total


def text_to_number(text: str) -> int:
    """Convert Russian spoken number text → integer.

    Returns -1 if the result is clearly invalid (e.g. no thousands word found
    and also not parseable as sub-999).
    """
    tokens = re.sub(r"[^а-яё ]", "", text.lower()).split()

    thou_idx: int | None = None
    for i, t in enumerate(tokens):
        if t in _THOU_WORDS:
            thou_idx = i
            break

    if thou_idx is None:
        # Fallback: try to parse as-is (shouldn't normally happen)
        val = _parse_chunk(tokens)
        return val if val > 0 else -1

    before = tokens[:thou_idx]
    after  = tokens[thou_idx + 1:]
    thousands = _parse_chunk(before) if before else 1
    remainder = _parse_chunk(after)  if after  else 0
    return thousands * 1000 + remainder


# ── CTC greedy decoding ───────────────────────────────────────────────────────
def ctc_greedy_decode(log_probs, lengths=None) -> List[List[int]]:
    """Greedy CTC decode.

    Args:
        log_probs: (T, B, V) or (T, V) tensor / numpy array
        lengths:   (B,) tensor of valid frame lengths (optional)
    Returns:
        List of decoded id lists (one per batch element).
    """
    import torch
    squeeze = log_probs.dim() == 2
    if squeeze:
        log_probs = log_probs.unsqueeze(1)

    T, B, _ = log_probs.shape
    best = log_probs.argmax(-1)   # (T, B)
    results = []
    for b in range(B):
        T_b = int(lengths[b]) if lengths is not None else T
        seq = best[:T_b, b].tolist()
        decoded: List[int] = []
        prev = -1
        for tok in seq:
            if tok != prev:
                if tok != BLANK_ID:
                    decoded.append(tok)
                prev = tok
        results.append(decoded)
    return results
