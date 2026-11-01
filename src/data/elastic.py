from datetime import date
from typing import Any, Iterable, Optional

import requests

from config import ES_CONFIG, PLATFORMS
from src.data.errors import DataSourceError
from src.data.schema import Post

REFERENCE_TYPES = tuple(ES_CONFIG.get("reference_types") or ("post", "tweet", "original"))
TEXT_FALLBACKS = ("normalized_text", "text", "content", "comment", "message", "caption")
USER_FALLBACKS = (
    "user_name",
    "channel_username",
    "sender",
    "username",
    "user_id",
    "from",
    "author",
    "from_id",
)
DISABLED_INDEX_VALUES = {"", "none", "off", "-", "false"}
SOURCE_INDEX_SUFFIXES = ("_source",)


def _has_auth() -> bool:
    return bool(ES_CONFIG.get("api_key") or (ES_CONFIG.get("user") and ES_CONFIG.get("password")))


def configured_indexes(platforms: Optional[Iterable[str]] = None) -> dict[str, list[str]]:
    wanted = [p for p in (platforms or PLATFORMS) if p in PLATFORMS]
    indexes = ES_CONFIG.get("indexes") or {}
    out: dict[str, list[str]] = {}
    for platform in wanted:
        raw = indexes.get(platform) or []
        names = [raw] if isinstance(raw, str) else list(raw)
        cleaned = [
            str(name).strip()
            for name in names
            if str(name).strip() and str(name).strip().lower() not in DISABLED_INDEX_VALUES
        ]
        if cleaned:
            out[platform] = cleaned
    if not out and not platforms:
        fallback = str(ES_CONFIG.get("index") or "").strip()
        if fallback.lower() not in DISABLED_INDEX_VALUES:
            out["twitter"] = [fallback]
    return out


def is_configured() -> bool:
    return bool(ES_CONFIG.get("enabled") and ES_CONFIG.get("hosts") and _has_auth() and configured_indexes())


def _headers() -> dict[str, str]:
    headers = {
        "Accept": "application/vnd.elasticsearch+json; compatible-with=8",
        "Content-Type": "application/vnd.elasticsearch+json; compatible-with=8",
    }
    if ES_CONFIG.get("api_key"):
        headers["Authorization"] = f"ApiKey {ES_CONFIG['api_key']}"
    return headers


def _auth() -> Optional[tuple[str, str]]:
    if ES_CONFIG.get("user") and ES_CONFIG.get("password") and not ES_CONFIG.get("api_key"):
        return ES_CONFIG["user"], ES_CONFIG["password"]
    return None


def _request(method: str, path: str, body: Optional[dict] = None) -> dict[str, Any]:
    url = ES_CONFIG["hosts"][0].rstrip("/") + path
    try:
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
        payload = response.json() if response.content else {}
    except (requests.RequestException, ValueError) as error:
        failed_response = getattr(error, "response", None)
        status_code = failed_response.status_code if failed_response is not None else None
        raise DataSourceError("Could not read the configured post source.", status_code) from error
    if not isinstance(payload, dict):
        raise DataSourceError("The post source returned an invalid response.")
    if payload.get("timed_out") or (payload.get("_shards") or {}).get("failed", 0):
        raise DataSourceError("The post source returned incomplete results.")
    return payload


def ping() -> dict:
    results: dict[str, list[dict]] = {}
    for platform, names in configured_indexes().items():
        items = []
        for index in names:
            try:
                count = _request("GET", f"/{index}/_count")
                items.append({"ok": True, "index": index, "docs": count.get("count")})
            except Exception as exc:
                items.append({"ok": False, "index": index, "error": str(exc)})
        results[platform] = items
    ok = any(item.get("ok") for items in results.values() for item in items) if results else False
    return {"ok": ok, "indexes": results}


def _fields_for(platform: str) -> dict[str, str]:
    per_platform = (ES_CONFIG.get("platform_fields") or {}).get(platform)
    return per_platform or ES_CONFIG["fields"]


def _nested_user(source: dict) -> str:
    user = source.get("user")
    if not isinstance(user, dict):
        return ""
    for key in ("username", "user_name", "name", "id", "first_name"):
        value = user.get(key)
        if value not in (None, ""):
            return str(value)
    return ""


def _first_text(source: dict, fields: dict[str, str]) -> str:
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


def _first_user(source: dict, fields: dict[str, str]) -> str:
    ordered = [fields.get("user_id"), *USER_FALLBACKS]
    seen = set()
    for key in ordered:
        if not key or key in seen:
            continue
        seen.add(key)
        value = source.get(key)
        if value not in (None, ""):
            return str(value)
    return _nested_user(source)


def _source_to_post(doc_id: str, source: dict, platform: str) -> Optional[Post]:
    fields = _fields_for(platform)
    user_id = _first_user(source, fields)
    text = _first_text(source, fields)
    if not text or not user_id:
        return None
    item_type = str(source.get("type") or "post").strip().lower()
    if platform == "twitter":
        if item_type in {"repost", "retweet", "re-tweet", "rt"}:
            return None
        if REFERENCE_TYPES and item_type and item_type not in REFERENCE_TYPES:
            return None
    resolved_platform = platform
    declared = source.get(fields.get("platform") or "") or source.get("platform")
    if declared:
        try:
            from src.data.schema import normalize_platform

            resolved_platform = normalize_platform(str(declared))
        except ValueError:
            resolved_platform = platform
    timestamp = source.get(fields["timestamp"]) or source.get("date") or source.get("timestamp")
    post_id = str(source.get(fields["post_id"]) or source.get("post_id") or doc_id)
    return Post(
        user_id=str(user_id).lstrip("@"),
        text=text,
        platform=resolved_platform,
        post_id=post_id,
        timestamp=str(timestamp) if timestamp is not None else None,
        metadata={
            "type": item_type,
            "user_title": source.get("user_title") or source.get("channel_title"),
            "index_platform": platform,
        },
    )


def _date_filter(
    platform: str,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> Optional[dict]:
    if start_date is None and end_date is None:
        start = (ES_CONFIG.get("start_date") or "").strip()
        end = (ES_CONFIG.get("end_date") or "").strip()
    else:
        start = start_date.isoformat() if start_date else ""
        end = end_date.isoformat() if end_date else ""
    if not start and not end:
        return None
    rng: dict[str, str] = {"format": "yyyy-MM-dd"}
    if start:
        rng["gte"] = start
    if end:
        rng["lte"] = end
    return {"range": {_fields_for(platform)["timestamp"]: rng}}


def _text_exists_filter() -> dict:
    return {
        "bool": {
            "should": [{"exists": {"field": name}} for name in TEXT_FALLBACKS],
            "minimum_should_match": 1,
        }
    }


def _user_terms_filter(platform: str, user_ids: Iterable[str]) -> Optional[dict]:
    handles = [str(u).lstrip("@") for u in user_ids if str(u).strip()]
    if not handles:
        return None
    fields = []
    for name in _user_field_candidates(platform):
        if name not in fields:
            fields.append(name)
    if not fields:
        return None
    if len(fields) == 1:
        return {"terms": {fields[0]: handles}}
    return {"bool": {"should": [{"terms": {field: handles}} for field in fields], "minimum_should_match": 1}}


def _is_source_catalog(index: str) -> bool:
    name = str(index).strip().lower()
    return any(name.endswith(suffix) for suffix in SOURCE_INDEX_SUFFIXES)


def _base_filters(
    platform: str,
    user_ids: Optional[Iterable[str]] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> list[dict]:
    filters: list[dict] = [_text_exists_filter()]
    if platform == "twitter" and REFERENCE_TYPES:
        filters.append({"terms": {"type": list(REFERENCE_TYPES)}})
    date_filter = _date_filter(platform, start_date=start_date, end_date=end_date)
    if date_filter:
        filters.append(date_filter)
    user_filter = _user_terms_filter(platform, user_ids or [])
    if user_filter:
        filters.append(user_filter)
    return filters


def _user_field_candidates(platform: str) -> list[str]:
    field = _fields_for(platform)["user_id"]
    names = [field, "user_name", "channel_username", "sender", "username"]
    if platform == "telegram":
        names.append("from_id")
    names.append("user.username")
    out: list[str] = []
    for name in names:
        if not name:
            continue
        if name not in out:
            out.append(name)
        keyword = f"{name}.keyword"
        if not name.endswith(".keyword") and keyword not in out:
            out.append(keyword)
    return out


def fetch_active_users_by_platform(
    per_platform: int = 1,
    min_posts: int = 6,
    platforms: Optional[Iterable[str]] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> dict[str, list[str]]:
    """Return active users grouped so every configured platform can appear in output."""
    indexes = configured_indexes(platforms)
    if not indexes:
        raise DataSourceError("No post source is configured for the requested platforms.")
    by_platform: dict[str, list[str]] = {}
    for platform, names in indexes.items():
        seen: list[str] = []
        for index in names:
            if _is_source_catalog(index) or len(seen) >= per_platform:
                continue
            index_queried = False
            mapping_error = None
            for field in _user_field_candidates(platform):
                body = {
                    "size": 0,
                    "query": {
                        "bool": {
                            "filter": _base_filters(
                                platform,
                                start_date=start_date,
                                end_date=end_date,
                            )
                        }
                    },
                    "aggs": {
                        "users": {
                            "terms": {
                                "field": field,
                                "size": max(per_platform, 10),
                                "min_doc_count": min_posts,
                            }
                        }
                    },
                }
                try:
                    resp = _request("POST", f"/{index}/_search", body)
                except DataSourceError as error:
                    if error.status_code != 400:
                        raise
                    mapping_error = error
                    continue
                index_queried = True
                buckets = ((resp.get("aggregations") or {}).get("users") or {}).get("buckets")
                if not isinstance(buckets, list):
                    raise DataSourceError("The post source returned an invalid user search response.")
                for bucket in buckets:
                    user = str(bucket.get("key") or "").lstrip("@")
                    if user and user not in seen:
                        seen.append(user)
                    if len(seen) >= per_platform:
                        break
                if len(seen) >= per_platform:
                    break
            if not index_queried and mapping_error is not None:
                raise mapping_error
        if seen:
            by_platform[platform] = seen[:per_platform]
    return by_platform


def fetch_active_users(
    limit: int = 5,
    min_posts: int = 6,
    platforms: Optional[Iterable[str]] = None,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> list[str]:
    indexes = configured_indexes(platforms)
    per_platform = max(1, (limit + max(len(indexes), 1) - 1) // max(len(indexes), 1))
    grouped = fetch_active_users_by_platform(
        per_platform=per_platform,
        min_posts=min_posts,
        platforms=platforms,
        start_date=start_date,
        end_date=end_date,
    )
    seen: list[str] = []
    for names in grouped.values():
        for user in names:
            if user not in seen:
                seen.append(user)
            if len(seen) >= limit:
                return seen
    return seen[:limit]


def _fetch_from_index(
    platform: str,
    index: str,
    user_ids: list[str],
    per_user: int,
    size: int,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> list[Post]:
    posts: list[Post] = []
    for user_id in dict.fromkeys(user_ids):
        remaining = size - len(posts)
        if remaining <= 0:
            break
        posts.extend(_fetch_user_from_index(
            platform, index, user_id, min(per_user, remaining),
            start_date=start_date, end_date=end_date,
        ))
    return posts


def _fetch_user_from_index(
    platform: str,
    index: str,
    user_id: str,
    limit: int,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> list[Post]:
    fields = _fields_for(platform)
    body = {
        "size": limit,
        "query": {
            "bool": {
                "filter": _base_filters(
                    platform,
                    [user_id],
                    start_date=start_date,
                    end_date=end_date,
                )
            }
        },
        "_source": list(dict.fromkeys([
            *fields.values(),
            *TEXT_FALLBACKS,
            *USER_FALLBACKS,
            "user",
            "user_title",
            "channel_title",
            "platform",
            "date",
            "timestamp",
            "type",
            "entity.hashtag",
        ])),
        "sort": [{fields["timestamp"]: {"order": "desc"}}],
    }
    resp = _request("POST", f"/{index}/_search", body)
    hits = (resp.get("hits") or {}).get("hits")
    if not isinstance(hits, list):
        raise DataSourceError("The post source returned an invalid post search response.")
    posts: list[Post] = []
    for hit in hits:
        post = _source_to_post(str(hit.get("_id", "")), hit.get("_source") or {}, platform)
        if not post or post.user_id != user_id:
            continue
        posts.append(post)
        if len(posts) >= limit:
            break
    return posts


def fetch_posts(
    user_ids: Iterable[str],
    platforms: Optional[Iterable[str]] = None,
    size: int = 5000,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
) -> list[Post]:
    user_ids = [str(u).lstrip("@") for u in user_ids if str(u).strip()]
    if not user_ids:
        return []
    indexes = configured_indexes(platforms)
    if not indexes:
        raise DataSourceError("No post source is configured for the requested platforms.")
    per_user = max(2, int(ES_CONFIG.get("max_posts_per_user") or 40))
    posts: list[Post] = []
    remaining = size
    for platform, names in indexes.items():
        for index in names:
            if remaining <= 0:
                break
            if _is_source_catalog(index):
                continue
            posts.extend(
                _fetch_from_index(
                    platform,
                    index,
                    user_ids,
                    per_user,
                    remaining,
                    start_date=start_date,
                    end_date=end_date,
                )
            )
            remaining = size - len(posts)
        if remaining <= 0:
            break
    return posts
