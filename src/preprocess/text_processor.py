import re

from hazm import Lemmatizer, Normalizer, word_tokenize

try:
    from hazm import stopwords_list
except ImportError:
    from hazm.utils import stopwords_list


class TextProcessor:
    """Persian text cleaning for TF-IDF pairing and lighter cleaning for ParsBERT."""

    url_re = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
    mention_re = re.compile(r"@\w+")
    email_re = re.compile(r"\S+@\S+")
    extra_space_re = re.compile(r"\s+")

    def __init__(self):
        self.normalizer = Normalizer()
        self.lemmatizer = Lemmatizer()
        self.stopwords = set(stopwords_list())

    def clean(self, text: str) -> str:
        text = "" if text is None else str(text)
        text = self.url_re.sub(" ", text)
        text = self.email_re.sub(" ", text)
        text = self.mention_re.sub(" ", text)
        text = text.replace("#", " ")
        text = self.extra_space_re.sub(" ", text)
        return text.strip()

    def normalize(self, text: str) -> str:
        return self.normalizer.normalize(self.clean(text))

    def process(self, text: str) -> str:
        """Full Hazm pipeline used for TF-IDF candidate selection."""
        text = self.normalize(text)
        tokens = word_tokenize(text)
        tokens = [t for t in tokens if t not in self.stopwords]
        tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        return " ".join(tokens)
