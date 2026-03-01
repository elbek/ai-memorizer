"""Arabic text utilities for tashkeel stripping and normalization."""

import re
import unicodedata

# All Arabic diacritical marks (tashkeel) to strip
_TASHKEEL_PATTERN = re.compile(
    "["
    "\u0610-\u061A"   # Arabic sign ranges
    "\u064B-\u065F"   # Fathatan through Wavy Hamza Below
    "\u0670"          # Superscript Alef
    "\u06D6-\u06DC"   # Quranic annotation marks
    "\u06DF-\u06E4"   # More Quranic marks
    "\u06E7-\u06E8"   # Quranic marks
    "\u06EA-\u06ED"   # Quranic marks
    "\uFE70-\uFE74"   # Arabic presentation forms
    "\uFE76-\uFE7F"   # Arabic presentation forms
    "]+"
)

# Alef variants to normalize to plain alef
_ALEF_VARIANTS = {
    "\u0622": "\u0627",  # ALEF WITH MADDA ABOVE
    "\u0623": "\u0627",  # ALEF WITH HAMZA ABOVE
    "\u0625": "\u0627",  # ALEF WITH HAMZA BELOW
    "\u0671": "\u0627",  # ALEF WASLA
}
_ALEF_PATTERN = re.compile("[" + "".join(_ALEF_VARIANTS.keys()) + "]")

# Uthmani Quran script marks that have no pronunciation effect —
# these appear in Quran references but ASR models output standard Arabic.
_QURAN_MARKS = re.compile(
    "["
    "\u0640"    # TATWEEL (kashida)
    "\u06DC"    # SMALL HIGH SEEN
    "\u06DF"    # SMALL HIGH ROUNDED ZERO
    "\u06E2"    # SMALL HIGH MEEM ISOLATED FORM
    "\u06E5"    # SMALL WAW
    "\u06E6"    # SMALL YEH
    "\u06ED"    # SMALL LOW MEEM
    "]"
)

# Punctuation to remove
_PUNCTUATION_PATTERN = re.compile(r"[.،؟!:؛]")

# Whitespace collapse
_WHITESPACE_PATTERN = re.compile(r"\s+")


def strip_tashkeel(text: str) -> str:
    """Remove all Arabic diacritical marks (tashkeel) from text."""
    return _TASHKEEL_PATTERN.sub("", text)


def normalize_quran_text(text: str) -> str:
    """Normalize Uthmani Quran script to standard Arabic for fair comparison.

    Strips Quran-specific typographic marks, normalizes alef variants and
    superscript alef, but preserves tashkeel (diacritical marks).
    """
    text = _QURAN_MARKS.sub("", text)
    text = _ALEF_PATTERN.sub("\u0627", text)
    # Superscript alef (U+0670) → regular alef when standalone,
    # but it's a combining mark so just remove it (the base alef is already there)
    text = text.replace("\u0670", "")
    text = _PUNCTUATION_PATTERN.sub("", text)
    text = _WHITESPACE_PATTERN.sub(" ", text).strip()
    return text


def normalize_arabic(text: str) -> str:
    """Normalize Arabic text for de-diacritized comparison.

    Steps:
    1. Strip tashkeel (diacritical marks)
    2. Strip Quran-specific marks
    3. Normalize alef variants to plain alef
    4. Convert teh marbuta to heh
    5. Remove punctuation
    6. Collapse whitespace and strip
    """
    text = strip_tashkeel(text)
    text = _QURAN_MARKS.sub("", text)
    text = _ALEF_PATTERN.sub("\u0627", text)
    text = text.replace("\u0629", "\u0647")  # teh marbuta -> heh
    text = _PUNCTUATION_PATTERN.sub("", text)
    text = _WHITESPACE_PATTERN.sub(" ", text).strip()
    return text
