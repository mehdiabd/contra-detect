from datetime import date
from unittest import TestCase

from src.data.loader import filter_posts
from src.data.schema import Pair, Post
from src.detection.engine import ContradictionEngine


class FakeCandidates:
    def build(self, posts):
        if len(posts) < 2:
            return []
        return [
            Pair(
                text_a=posts[0].text,
                text_b=posts[1].text,
                user_id=posts[0].user_id,
                post_id_a=posts[0].post_id,
                post_id_b=posts[1].post_id,
                platform_a=posts[0].platform,
                platform_b=posts[1].platform,
                timestamp_a=posts[0].timestamp,
                timestamp_b=posts[1].timestamp,
            )
        ]


class FakePredictor:
    scores = {"low": 0.60, "high": 0.95, "none": 0.20}

    def predict_pairs(self, pairs):
        return [
            {
                "pair": pair,
                "contradiction_score": self.scores[pair.user_id],
                "label": "contradiction" if pair.user_id != "none" else "neutral",
            }
            for pair in pairs
        ]


class FilterAndTopResultTests(TestCase):
    def test_filter_posts_uses_inclusive_date_range(self):
        posts = [
            Post(user_id="u", text="a", timestamp="2026-01-01T09:00:00Z"),
            Post(user_id="u", text="b", timestamp="2026-01-31"),
            Post(user_id="u", text="c", timestamp="2026-02-01"),
            Post(user_id="u", text="d", timestamp=None),
        ]

        result = filter_posts(
            posts,
            start_date=date(2026, 1, 1),
            end_date=date(2026, 1, 31),
        )

        self.assertEqual([post.text for post in result], ["a", "b"])

    def test_empty_user_ids_return_ranked_contradictions(self):
        posts = []
        for user_id in ("low", "high", "none"):
            posts.extend(
                [
                    Post(user_id=user_id, text=f"{user_id}-a", platform="twitter"),
                    Post(user_id=user_id, text=f"{user_id}-b", platform="twitter"),
                ]
            )
        engine = ContradictionEngine(predictor=FakePredictor())
        engine.candidates = FakeCandidates()

        result = engine.analyze_users([], platforms=["twitter"], posts=posts)

        self.assertEqual([row["user_id"] for row in result], ["high", "low"])
        self.assertTrue(all(row["contradictory_posts"] for row in result))

