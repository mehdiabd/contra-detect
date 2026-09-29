from typing import Any, Iterable, Optional

import requests

from config import ES_CONFIG
from src.data.schema import Post

REFERENCE_TYPES = tuple(ES_CONFIG.get("reference_types") or ("post", "tweet", "original"))
TEXT_FALLBACKS = ("normalized_text", "text", "content", "comment")


def is_configured() -> bool:
    has_auth = bool(ES_CONFIG.get("api_key") or (ES_CONFIG.get("user") and ES_CONFIG.get("password")))
    return bool(ES_CONFIG.get("enabled") and ES_CONFIG.get("index") and ES_CONFIG.get("hosts") and has_auth)


def _headers() -> dict[str, str]:
    headers = {"Accept": "application/json", "Content-Type": "application/json"}
    if ES_CONFIG.get("api_key"):
        headers["Authorization"] = f"ApiKey {ES_CONFIG['api_key']}"
    return headers


def _auth() -> Optional[tuple[str, str]]:
    if ES_CONFIG.get("user") and ES_CONFIG.get("password") and not ES_CONFIG.get("api_key"):
        return ES_CONFIG["user"], ES_CONFIG["password"]
    return None


def _request(method: str, path: str, body: Optional[dict] = None) -> dict[str, Any]:
    url = ES_CONFIG["hosts"][0].rstrip("/") + path
    response = requests.request(
        method,
        url,
        json=body,
        headers=_headers(),
        auth=_auth(),
        timeout=60,
        verify=bool(ES_CONFIG.get("verify_certs", True)),
    )
    response.raise_for_status()
    return response.json() if response.content else {}


def ping() -> dict:
    count = _request("GET", f"/{ES_CONFIG['index']}/_count")
    return {
        "ok": True,
        "index": ES_CONFIG["index"],
        "docs": count.get("count"),
    }


def _first_text(source: dict) -> str:
    fields = ES_CONFIG["fields"]
    ordered = [fields.get("text"), *TEXT_FALLBACKS]
    seen = set()
    for key in ordered:
        if not key or key in seen:
            continue
        seen.add(key)
        value = source.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _source_to_post(doc_id: str, source: dict) -> Optional[Post]:
    fields = ES_CONFIG["fields"]
    user_id = source.get(fields["user_id"]) or source.get("user_name") or source.get("user_id")
    text = _first_text(source)
    if not text or not user_id:
        return None
    tweet_type = str(source.get("type") or "post").strip().lower()
    if tweet_type in {"repost", "retweet", "re-tweet", "rt"}:
        return None
    if REFERENCE_TYPES and tweet_type and tweet_type not in REFERENCE_TYPES:
        return None
    timestamp = source.get(fields["timestamp"]) or source.get("date") or source.get("timestamp")
    post_id = str(source.get(fields["post_id"]) or source.get("post_id") or doc_id)
    return Post(
        user_id=str(user_id).lstrip("@"),
        text=text,
        platform="twitter",
        post_id=post_id,
        timestamp=str(timestamp) if timestamp is not None else None,
        metadata={
            "type": tweet_type,
            "user_title": source.get("user_title"),
        },
    )


def _date_filter() -> Optional[dict]:
    start = (ES_CONFIG.get("start_date") or "").strip()
    end = (ES_CONFIG.get("end_date") or "").strip()
    if not start and not end:
        return None
    rng: dict[str, str] = {"format": "yyyy-MM-dd"}
    if start:
        rng["gte"] = start
    if end:
        rng["lte"] = end
    return {"range": {ES_CONFIG["fields"]["timestamp"]: rng}}


def _base_filters(user_ids: Optional[Iterable[str]] = None) -> list[dict]:
    filters: list[dict] = []
    if REFERENCE_TYPES:
        filters.append({"terms": {"type": list(REFERENCE_TYPES)}})
    date_filter = _date_filter()
    if date_filter:
        filters.append(date_filter)
    if user_ids:
        handles = [str(u).lstrip("@") for u in user_ids if str(u).strip()]
        if handles:
            filters.append({"terms": {ES_CONFIG["fields"]["user_id"]: handles}})
    return filters


def _user_field_candidates() -> list[str]:
    field = ES_CONFIG["fields"]["user_id"]
    return [field, f"{field}.keyword"] if not field.endswith(".keyword") else [field]


def fetch_active_users(limit: int = 5, min_posts: int = 6) -> list[str]:
    last_error = None
    for field in _user_field_candidates():
        body = {
            "size": 0,
            "query": {"bool": {"filter": _base_filters()}},
            "aggs": {
                "users": {
                    "terms": {
                        "field": field,
                        "size": max(limit, 10),
                        "min_doc_count": min_posts,
                    }
                }
            },
        }
        try:
            resp = _request("POST", f"/{ES_CONFIG['index']}/_search", body)
        except requests.HTTPError as exc:
            last_error = exc
            continue
        buckets = (((resp.get("aggregations") or {}).get("users") or {}).get("buckets")) or []
        users = [str(b["key"]).lstrip("@") for b in buckets[:limit] if b.get("key")]
        if users:
            return users
    if last_error:
        raise last_error
    return []


def fetch_posts(
    user_ids: Iterable[str],
    platforms: Optional[Iterable[str]] = None,
    size: int = 5000,
) -> list[Post]:
    user_ids = [str(u).lstrip("@") for u in user_ids if str(u).strip()]
    if not user_ids:
        return []
    platforms = [p for p in (platforms or []) if p]
    if platforms and "twitter" not in platforms:
        return []

    per_user = max(2, int(ES_CONFIG.get("max_posts_per_user") or 40))
    fetch_size = min(size, max(len(user_ids) * per_user, 10))
    body = {
        "size": fetch_size,
        "query": {"bool": {"filter": _base_filters(user_ids)}},
        "_source": [
            "user_name",
            "user_title",
            "normalized_text",
            "text",
            "content",
            "date",
            "timestamp",
            "type",
            "entity.hashtag",
        ],
        "sort": [{ES_CONFIG["fields"]["timestamp"]: {"order": "desc"}}],
    }
    resp = _request("POST", f"/{ES_CONFIG['index']}/_search", body)
    posts: list[Post] = []
    counts: dict[str, int] = {u: 0 for u in user_ids}
    for hit in ((resp.get("hits") or {}).get("hits") or []):
        post = _source_to_post(str(hit.get("_id", "")), hit.get("_source") or {})
        if not post:
            continue
        if counts.get(post.user_id, 0) >= per_user:
            continue
        posts.append(post)
        counts[post.user_id] = counts.get(post.user_id, 0) + 1
        if len(posts) >= size:
            break
    return posts
