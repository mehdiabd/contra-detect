from typing import Iterable, Optional

from config import ES_CONFIG
from src.data.schema import Post


def is_configured() -> bool:
    return bool(ES_CONFIG.get("enabled") and ES_CONFIG.get("index") and ES_CONFIG.get("hosts"))


def _client():
    from elasticsearch import Elasticsearch

    kwargs = {"hosts": ES_CONFIG["hosts"]}
    if ES_CONFIG.get("api_key"):
        kwargs["api_key"] = ES_CONFIG["api_key"]
    elif ES_CONFIG.get("user") and ES_CONFIG.get("password"):
        kwargs["basic_auth"] = (ES_CONFIG["user"], ES_CONFIG["password"])
    return Elasticsearch(**kwargs)


def _source_to_post(doc_id: str, source: dict) -> Optional[Post]:
    fields = ES_CONFIG["fields"]
    text = source.get(fields["text"]) or source.get("text") or source.get("content")
    user_id = source.get(fields["user_id"]) or source.get("user_id")
    if not text or not user_id:
        return None
    platform = source.get(fields["platform"]) or source.get("platform") or "twitter"
    timestamp = source.get(fields["timestamp"]) or source.get("timestamp")
    post_id = str(source.get(fields["post_id"]) or source.get("post_id") or doc_id)
    return Post(
        user_id=str(user_id),
        text=str(text),
        platform=str(platform),
        post_id=post_id,
        timestamp=str(timestamp) if timestamp is not None else None,
        metadata={k: v for k, v in source.items() if k not in fields.values()},
    )


def fetch_posts(
    user_ids: Iterable[str],
    platforms: Optional[Iterable[str]] = None,
    size: int = 5000,
) -> list[Post]:
    from elasticsearch.helpers import scan

    fields = ES_CONFIG["fields"]
    user_ids = [str(u) for u in user_ids]
    if not user_ids:
        return []

    filters = [{"terms": {fields["user_id"]: user_ids}}]
    platforms = [p for p in (platforms or []) if p]
    if platforms:
        filters.append({"terms": {fields["platform"]: list(platforms)}})

    client = _client()
    posts: list[Post] = []
    for hit in scan(
        client,
        index=ES_CONFIG["index"],
        query={"query": {"bool": {"filter": filters}}},
        size=200,
    ):
        post = _source_to_post(str(hit.get("_id", "")), hit.get("_source") or {})
        if post:
            posts.append(post)
        if len(posts) >= size:
            break
    return posts
