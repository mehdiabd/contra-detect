import re


def nli_text(text: str) -> str:
    """Whitespace-only cleanup. FarsTail baselines do not apply extra normalization to NLI input."""
    return " ".join(str(text or "").split())


class TextProcessor:
    """Persian text cleaning for TF-IDF pairing and lighter cleaning for ParsBERT."""

    url_re = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
    mention_re = re.compile(r"@\w+")
    email_re = re.compile(r"\S+@\S+")
    extra_space_re = re.compile(r"\s+")
    _trans = str.maketrans(
        {
            "ي": "ی",
            "ك": "ک",
            "ة": "ه",
            "ؤ": "و",
            "إ": "ا",
            "أ": "ا",
            "ٱ": "ا",
            "۰": "0",
            "۱": "1",
            "۲": "2",
            "۳": "3",
            "۴": "4",
            "۵": "5",
            "۶": "6",
            "۷": "7",
            "۸": "8",
            "۹": "9",
        }
    )

    def __init__(self):
        self.stopwords = set()

    def clean(self, text: str) -> str:
        text = "" if text is None else str(text)
        text = self.url_re.sub(" ", text)
        text = self.email_re.sub(" ", text)
        text = self.mention_re.sub(" ", text)
        text = text.replace("#", " ")
        text = self.extra_space_re.sub(" ", text)
        return text.strip()

    def normalize(self, text: str) -> str:
        text = self.clean(text).translate(self._trans)
        return self.extra_space_re.sub(" ", text).strip()

    def process(self, text: str) -> str:
        text = self.normalize(text)
        tokens = [t for t in text.split() if t not in self.stopwords]
        return " ".join(tokens)
