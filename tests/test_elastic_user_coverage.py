from unittest import TestCase
from unittest.mock import patch

from src.data import elastic
from src.data.errors import DataSourceError


def requested_users(value):
    users = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "terms" and isinstance(item, dict):
                for field, values in item.items():
                    if field != "type":
                        users.update(values)
            else:
                users.update(requested_users(item))
    elif isinstance(value, list):
        for item in value:
            users.update(requested_users(item))
    return users


class ElasticsearchUserCoverageTests(TestCase):
    def test_recent_prolific_user_cannot_hide_another_requested_user(self):
        def search(method, path, body):
            users = requested_users(body["query"])
            hits = []
            # All of prolific's posts are newer than quiet's posts.
            for user, count in (("prolific", 20), ("quiet", 3)):
                if user in users:
                    hits.extend({"_id": f"{user}-{i}", "_source": {
                        "user_name": user, "text": "متن معتبر برای بررسی پوشش کاربران", "type": "post",
                    }} for i in range(count))
            return {"hits": {"hits": hits[:body["size"]]}}

        with patch("src.data.elastic._request", side_effect=search):
            posts = elastic._fetch_from_index("twitter", "posts", ["prolific", "quiet"], 3, 100)
        self.assertEqual([p.user_id for p in posts], ["prolific"] * 3 + ["quiet"] * 3)

    def test_telegram_comment_identity_and_text_survive_source_projection(self):
        source = {"from_id": "comment-author", "comment": "متن یک دیدگاه واقعی برای تحلیل", "date": "2026-10-26"}

        def search(method, path, body):
            self.assertIn("from_id", str(body["query"]))
            projected = {k: v for k, v in source.items() if k in body["_source"]}
            return {"hits": {"hits": [{"_id": "comment-1", "_source": projected}]}}

        with patch("src.data.elastic._request", side_effect=search):
            posts = elastic._fetch_from_index("telegram", "comments", ["comment-author"], 15, 100)
        self.assertEqual(len(posts), 1)
        self.assertEqual(posts[0].user_id, "comment-author")
        self.assertEqual(posts[0].text, source["comment"])

    def test_total_limit_and_duplicate_user_ids_do_not_trigger_extra_reads(self):
        def search(method, path, body):
            user = next(iter(requested_users(body["query"])))
            return {"hits": {"hits": [{"_id": "p", "_source": {"user_name": user, "text": "نوشته معتبر"}}]}}

        with patch("src.data.elastic._request", side_effect=search) as request:
            posts = elastic._fetch_from_index("twitter", "posts", ["a", "a", "b"], 15, 1)
        self.assertEqual(len(posts), 1)
        self.assertEqual(request.call_count, 1)

    def test_second_user_source_failure_aborts_the_whole_read(self):
        with patch("src.data.elastic._request", side_effect=[{"hits": {"hits": []}}, DataSourceError("offline")]):
            with self.assertRaises(DataSourceError):
                elastic._fetch_from_index("twitter", "posts", ["a", "b"], 15, 100)

