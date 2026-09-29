"""Run contradiction detection on live X / Telegram / Instagram posts."""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import ES_CONFIG, PLATFORMS
from src.data import elastic
from src.data.schema import Post
from src.detection.engine import ContradictionEngine


USERS_PER_PLATFORM = 2
MIN_POSTS = 6
RECENT_SAMPLE = 120
REPORT_PATH = ROOT / "reports" / "live_three_platforms.json"


def _safe_search(index: str, body: dict) -> dict:
    try:
        return elastic._request("POST", f"/{index}/_search", body)
    except Exception as exc:
        return {"error": str(exc)}


def sample_recent(platform: str, index: str) -> list[Post]:
    if elastic._is_source_catalog(index):
        return []
    fields = elastic._fields_for(platform)
    filters = elastic._base_filters(platform)
    if platform == "telegram" and index.endswith("_temp_data"):
        filters.append({"term": {"language.label": "fa"}})
    query = {"bool": {"filter": filters}} if filters else {"match_all": {}}
    bodies = [
        {
            "size": RECENT_SAMPLE,
            "query": query,
            "sort": [{fields["timestamp"]: {"order": "desc"}}],
        },
        {"size": RECENT_SAMPLE, "query": query},
        {"size": RECENT_SAMPLE, "query": {"match_all": {}}},
    ]
    for body in bodies:
        resp = _safe_search(index, body)
        if resp.get("error"):
            continue
        posts: list[Post] = []
        for hit in ((resp.get("hits") or {}).get("hits") or []):
            post = elastic._source_to_post(str(hit.get("_id", "")), hit.get("_source") or {}, platform)
            if post:
                posts.append(post)
        if posts:
            return posts
    return []


def pick_users(posts: list[Post], already: set[str]) -> list[str]:
    counts: Counter[str] = Counter(p.user_id for p in posts)
    ranked = [u for u, n in counts.most_common() if u not in already and n >= 2]
    if not ranked:
        ranked = [u for u, _ in counts.most_common() if u not in already]
    return ranked[:USERS_PER_PLATFORM]


def fetch_user_posts(user_ids: list[str]) -> list[Post]:
    posts: list[Post] = []
    per_user = max(2, int(ES_CONFIG.get("max_posts_per_user") or 40))
    for platform, names in elastic.configured_indexes().items():
        for index in names:
            if elastic._is_source_catalog(index):
                continue
            try:
                posts.extend(elastic._fetch_from_index(platform, index, user_ids, per_user, 2000))
            except Exception as exc:
                print(f"  skip {platform}/{index}: {exc}")
    return posts


def summarize_posts(posts: list[Post]) -> dict:
    by_platform = Counter(p.platform for p in posts)
    by_user = Counter(p.user_id for p in posts)
    return {
        "post_count": len(posts),
        "platforms": dict(by_platform),
        "users": dict(by_user),
    }


def compact_result(row: dict) -> dict:
    hits = []
    for hit in row["contradictory_posts"][:5]:
        hits.append(
            {
                "contradiction_score": hit["contradiction_score"],
                "label": hit["label"],
                "similarity": hit["similarity"],
                "post_a": {
                    "platform": hit["post_a"]["platform"],
                    "timestamp": hit["post_a"]["timestamp"],
                    "text": hit["post_a"]["text"],
                },
                "post_b": {
                    "platform": hit["post_b"]["platform"],
                    "timestamp": hit["post_b"]["timestamp"],
                    "text": hit["post_b"]["text"],
                },
            }
        )
    return {
        "user_id": row["user_id"],
        "has_contradiction": row["has_contradiction"],
        "score": row["score"],
        "post_count": row["post_count"],
        "pair_count": row["pair_count"],
        "contradictory_posts": hits,
    }


def main() -> None:
    if not elastic.is_configured():
        raise SystemExit("Elasticsearch is not configured.")

    print("ES ping")
    ping = elastic.ping()
    print(json.dumps(ping, ensure_ascii=False, indent=2))

    users_by_platform: dict[str, list[str]] = {}
    seed_posts: list[Post] = []
    claimed: set[str] = set()

    for platform in PLATFORMS:
        platform_posts: list[Post] = []
        for index in elastic.configured_indexes().get(platform, []):
            print(f"sample {platform}/{index}")
            found = sample_recent(platform, index)
            print(f"  posts={len(found)}")
            platform_posts.extend(found)
        users = pick_users(platform_posts, claimed)
        claimed.update(users)
        users_by_platform[platform] = users
        seed_posts.extend(platform_posts)
        print(f"users {platform}: {users}")

    user_ids = [u for names in users_by_platform.values() for u in names]
    if not user_ids:
        raise SystemExit("No live users found in any platform index.")

    print(f"\nfetch posts for {user_ids}")
    posts = fetch_user_posts(user_ids)
    if not posts:
        # last resort: keep the sampled recent posts for the chosen users
        wanted = set(user_ids)
        posts = [p for p in seed_posts if p.user_id in wanted]
    print("loaded", summarize_posts(posts))

    engine = ContradictionEngine()
    results = engine.analyze_users(user_ids, posts=posts)
    compact = [compact_result(row) for row in results]

    payload = {
        "indexes": elastic.configured_indexes(),
        "elasticsearch": ping,
        "users_by_platform": users_by_platform,
        "posts": summarize_posts(posts),
        "results": compact,
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\nنتایج زنده هر سه پلتفرم")
    print("=" * 36)
    for row in compact:
        print(f"\nکاربر: {row['user_id']}")
        print(f"تناقض دارد؟ {'بله' if row['has_contradiction'] else 'خیر'}")
        print(f"امتیاز: {row['score']} | پست: {row['post_count']} | جفت: {row['pair_count']}")
        if not row["contradictory_posts"]:
            print("جفت متناقض: ندارد")
            continue
        for i, hit in enumerate(row["contradictory_posts"], 1):
            a, b = hit["post_a"], hit["post_b"]
            print(f"  {i}) [{a['platform']} {a['timestamp']}] {a['text'][:180]}")
            print(f"     [{b['platform']} {b['timestamp']}] {b['text'][:180]}")
            print(f"     score={hit['contradiction_score']}")
    print(f"\nsaved {REPORT_PATH}")


if __name__ == "__main__":
    main()
