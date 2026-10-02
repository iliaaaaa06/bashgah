"""Lightweight Persian text normalization (no heavy NLP dependencies)."""
import re
import unicodedata

ZWNJ = "‌"

_CHAR_MAP = str.maketrans({
    "ي": "ی", "ى": "ی",  # Arabic yeh / alef maksura -> Persian yeh
    "ك": "ک", "ڪ": "ک",  # Arabic kaf -> Persian keheh
    "ۀ": "ه", "ة": "ه",  # heh with yeh / teh marbuta -> heh
    "ٱ": "ا",                      # alef wasla -> alef
    # Arabic-Indic digits -> Persian digits
    "٠": "۰", "١": "۱", "٢": "۲", "٣": "۳", "٤": "۴",
    "٥": "۵", "٦": "۶", "٧": "۷", "٨": "۸", "٩": "۹",
    # ZWJ / soft hyphen -> ZWNJ
    "‍": ZWNJ, "­": ZWNJ,
})

_PERSIAN_TO_LATIN_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹", "0123456789")

# Arabic diacritics (harakat), superscript alef, tatweel
_DIACRITICS = re.compile("[ً-ٰٟـ]")
# Bidi control characters and BOM
_BIDI = re.compile("[‎‏‪-‮⁦-⁩﻿]")
_MULTI_ZWNJ = re.compile(f"{ZWNJ}+")
# ZWNJ next to whitespace or line edges is meaningless
_ZWNJ_AROUND_SPACE = re.compile(rf"{ZWNJ}(?=\s|$)|(?:(?<=\s)|^){ZWNJ}", re.MULTILINE)
_SPACES = re.compile("[ \t  -​  　]+")
_MANY_NEWLINES = re.compile(r"\n{3,}")


def normalize(text: str, latin_digits: bool = False) -> str:
    """Normalize Persian text for consistent embedding & retrieval."""
    if not text:
        return ""
    # NFKC folds Arabic presentation forms (common in PDF extraction) into base letters
    text = unicodedata.normalize("NFKC", text)
    text = text.translate(_CHAR_MAP)
    text = _DIACRITICS.sub("", text)
    text = _BIDI.sub("", text)
    text = _MULTI_ZWNJ.sub(ZWNJ, text)
    text = _ZWNJ_AROUND_SPACE.sub("", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _SPACES.sub(" ", text)
    text = "\n".join(line.strip() for line in text.split("\n"))
    text = _MANY_NEWLINES.sub("\n\n", text)
    if latin_digits:
        text = text.translate(_PERSIAN_TO_LATIN_DIGITS)
    return text.strip()


# Separators ordered from strongest to weakest boundary, including Persian punctuation.
CHUNK_SEPARATORS = [
    "\n\n",
    "\n",
    ". ", "؟ ", "! ", "؛ ",   # . ؟ ! ؛
    ".", "؟", "!", "؛",
    "، ", ", ",                   # ، ,
    " ",
    ZWNJ,
    "",
]
