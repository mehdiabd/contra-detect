from collections import Counter, defaultdict
from itertools import combinations
from typing import Sequence

import numpy as np

from config import FEATURE_CONFIG
from src.data.schema import Pair, Post
from src.preprocess.text_processor import TextProcessor


def _ngrams(tokens: list[str], ngram_range: tuple[int, int]) -> list[str]:
    lo, hi = ngram_range
    out: list[str] = []
    for n in range(lo, hi + 1):
        if n == 1:
            out.extend(tokens)
            continue
        for i in range(len(tokens) - n + 1):
            out.append(" ".join(tokens[i : i + n]))
    return out


def _tfidf_cosine(docs: Sequence[str], max_features: int, ngram_range: tuple[int, int]) -> np.ndarray:
    tokenized = [_ngrams(doc.split(), ngram_range) for doc in docs]
    df: Counter[str] = Counter()
    for toks in tokenized:
        df.update(set(toks))
    vocab = [term for term, _ in df.most_common(max_features)]
    index = {term: i for i, term in enumerate(vocab)}
    if not index:
        return np.zeros((len(docs), len(docs)), dtype=np.float32)
    matrix = np.zeros((len(docs), len(index)), dtype=np.float32)
    n_docs = len(docs)
    for i, toks in enumerate(tokenized):
        counts = Counter(t for t in toks if t in index)
        length = max(len(toks), 1)
        for term, count in counts.items():
            tf = count / length
            idf = np.log((1 + n_docs) / (1 + df[term])) + 1.0
            matrix[i, index[term]] = tf * idf
        norm = float(np.linalg.norm(matrix[i]))
        if norm:
            matrix[i] /= norm
    return matrix @ matrix.T


class CandidateBuilder:
    """Build same-user post pairs that are topically related (TF-IDF cosine)."""

    def __init__(
        self,
        min_similarity: float = FEATURE_CONFIG["min_pair_similarity"],
        max_similarity: float = FEATURE_CONFIG["max_pair_similarity"],
        max_pairs_per_user: int = FEATURE_CONFIG["max_pairs_per_user"],
        min_text_length: int = FEATURE_CONFIG["min_text_length"],
    ):
        self.min_similarity = min_similarity
        self.max_similarity = max_similarity
        self.max_pairs_per_user = max_pairs_per_user
        self.min_text_length = min_text_length
        self.processor = TextProcessor()

    def build(self, posts: Sequence[Post]) -> list[Pair]:
        grouped: dict[str, list[Post]] = defaultdict(list)
        for post in posts:
            if len((post.text or "").strip()) >= self.min_text_length:
                grouped[post.user_id].append(post)

        pairs: list[Pair] = []
        for user_id, user_posts in grouped.items():
            pairs.extend(self._pairs_for_user(user_id, user_posts))
        return pairs

    def _pairs_for_user(self, user_id: str, posts: Sequence[Post]) -> list[Pair]:
        if len(posts) < 2:
            return []
        processed = [self.processor.process(p.text) for p in posts]
        valid_idx = [i for i, text in enumerate(processed) if text.strip()]
        if len(valid_idx) < 2:
            return []
        posts = [posts[i] for i in valid_idx]
        processed = [processed[i] for i in valid_idx]
        sims = _tfidf_cosine(
            processed,
            FEATURE_CONFIG["tfidf_max_features"],
            FEATURE_CONFIG["tfidf_ngram_range"],
        )
        scored: list[tuple[float, int, int]] = []
        for i, j in combinations(range(len(posts)), 2):
            sim = float(sims[i, j])
            if self.min_similarity <= sim <= self.max_similarity:
                scored.append((sim, i, j))
        if not scored:
            for i, j in combinations(range(len(posts)), 2):
                scored.append((float(sims[i, j]), i, j))
        scored.sort(key=lambda item: item[0], reverse=True)
        scored = scored[: self.max_pairs_per_user]
        return [
            Pair(
                text_a=posts[i].text,
                text_b=posts[j].text,
                user_id=user_id,
                post_id_a=posts[i].post_id,
                post_id_b=posts[j].post_id,
                platform_a=posts[i].platform,
                platform_b=posts[j].platform,
                timestamp_a=posts[i].timestamp,
                timestamp_b=posts[j].timestamp,
                similarity=round(sim, 4),
            )
            for sim, i, j in scored
        ]
