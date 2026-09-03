from collections import defaultdict
from itertools import combinations
from typing import Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from config import FEATURE_CONFIG
from src.data.schema import Pair, Post
from src.preprocess.text_processor import TextProcessor


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
        self.vectorizer = TfidfVectorizer(
            max_features=FEATURE_CONFIG["tfidf_max_features"],
            ngram_range=FEATURE_CONFIG["tfidf_ngram_range"],
        )

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
        matrix = self.vectorizer.fit_transform(processed)
        sims = cosine_similarity(matrix)
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
