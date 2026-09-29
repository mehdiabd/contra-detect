from collections import defaultdict
from typing import Iterable, Optional, Sequence

from config import CONTRADICTION_THRESHOLDS, SAMPLE_DATA_PATH
from src.data import elastic
from src.data.loader import filter_posts, load_posts_json
from src.data.schema import Pair, Post
from src.features.candidates import CandidateBuilder
from src.models.predictor import ContradictionPredictor


def _user_score(predictions: list[dict], pair_threshold: float) -> float:
    if not predictions:
        return 0.0
    above = [p["contradiction_score"] for p in predictions if p["contradiction_score"] >= pair_threshold]
    if not above:
        return round(max(p["contradiction_score"] for p in predictions) * 0.25, 4)
    return round(min(1.0, sum(above) / len(above)), 4)


def _serialize_hit(pred: dict) -> dict:
    pair: Pair = pred["pair"]
    return {
        "user_id": pair.user_id,
        "contradiction_score": round(pred["contradiction_score"], 4),
        "label": pred["label"],
        "similarity": pair.similarity,
        "post_a": {
            "post_id": pair.post_id_a,
            "platform": pair.platform_a,
            "timestamp": pair.timestamp_a,
            "text": pair.text_a,
        },
        "post_b": {
            "post_id": pair.post_id_b,
            "platform": pair.platform_b,
            "timestamp": pair.timestamp_b,
            "text": pair.text_b,
        },
    }


class ContradictionEngine:
    def __init__(self, predictor: Optional[ContradictionPredictor] = None):
        self.predictor = predictor
        self.candidates = CandidateBuilder()

    def resolve_posts(
        self,
        user_ids: Sequence[str],
        platforms: Optional[Iterable[str]] = None,
        posts: Optional[Sequence[Post]] = None,
    ) -> list[Post]:
        if posts:
            resolved = [p if isinstance(p, Post) else Post.from_dict(p) for p in posts]
            return filter_posts(resolved, user_ids=user_ids, platforms=platforms)
        sample = filter_posts(load_posts_json(SAMPLE_DATA_PATH), user_ids=user_ids, platforms=platforms)
        wanted = {str(u) for u in user_ids}
        if wanted and wanted.issubset({p.user_id for p in sample}):
            return sample
        if elastic.is_configured():
            return elastic.fetch_posts(user_ids, platforms=platforms)
        return sample

    def analyze_users(
        self,
        user_ids: Sequence[str],
        platforms: Optional[Iterable[str]] = None,
        posts: Optional[Sequence[Post]] = None,
        pair_threshold: float = CONTRADICTION_THRESHOLDS["pair"],
        user_threshold: float = CONTRADICTION_THRESHOLDS["user"],
    ) -> list[dict]:
        if self.predictor is None:
            self.predictor = ContradictionPredictor()
        posts = self.resolve_posts(user_ids, platforms=platforms, posts=posts)
        by_user: dict[str, list[Post]] = defaultdict(list)
        for post in posts:
            by_user[post.user_id].append(post)

        results = []
        for user_id in user_ids:
            user_posts = by_user.get(str(user_id), [])
            pairs = self.candidates.build(user_posts)
            predictions = self.predictor.predict_pairs(pairs)
            contradictory = [
                p
                for p in predictions
                if p["contradiction_score"] >= pair_threshold
                and str(p["label"]).lower() in {"contradiction", "contradicts", "contradict", "c"}
            ]
            score = _user_score(predictions, pair_threshold)
            results.append(
                {
                    "user_id": str(user_id),
                    "post_count": len(user_posts),
                    "pair_count": len(pairs),
                    "has_contradiction": bool(contradictory) and score >= user_threshold,
                    "score": score,
                    "contradictory_posts": [_serialize_hit(p) for p in contradictory],
                }
            )
        return results


def analyze_users(*args, **kwargs) -> list[dict]:
    return ContradictionEngine().analyze_users(*args, **kwargs)
