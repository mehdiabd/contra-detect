from functools import lru_cache
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import API_CONFIG, ES_CONFIG, PLATFORMS
from src.data import elastic
from src.data.schema import Post
from src.detection.engine import ContradictionEngine
from src.models.predictor import resolve_model_source

app = FastAPI(
    title=API_CONFIG["title"],
    description=API_CONFIG["description"],
    version=API_CONFIG["version"],
)


class PostIn(BaseModel):
    user_id: str
    text: str
    platform: str = "twitter"
    post_id: str = ""
    timestamp: Optional[str] = None


class AnalyzeRequest(BaseModel):
    user_ids: list[str] = Field(default_factory=list)
    platforms: Optional[list[str]] = None
    posts: Optional[list[PostIn]] = None


@lru_cache(maxsize=1)
def get_engine() -> ContradictionEngine:
    return ContradictionEngine()


def _resolve_user_ids(req: AnalyzeRequest) -> list[str]:
    if req.user_ids:
        return [u.lstrip("@") for u in req.user_ids]
    if req.posts:
        return sorted({p.user_id.lstrip("@") for p in req.posts})
    if elastic.is_configured():
        users = elastic.fetch_active_users(limit=3, min_posts=6)
        if users:
            return users
    raise HTTPException(400, "user_ids is required when Elasticsearch has no active users.")


def _analyze(req: AnalyzeRequest) -> list[dict]:
    if req.platforms:
        unknown = [p for p in req.platforms if p not in PLATFORMS]
        if unknown:
            raise HTTPException(400, f"Unsupported platforms: {unknown}. Use {list(PLATFORMS)}")
    posts = [Post.from_dict(p.model_dump()) for p in req.posts] if req.posts else None
    try:
        return get_engine().analyze_users(_resolve_user_ids(req), platforms=req.platforms, posts=posts)
    except Exception as exc:
        raise HTTPException(503, f"Failed to read reference tweets from Elasticsearch: {exc}") from exc


@app.get("/health")
def health():
    payload = {
        "status": "ok",
        "model_source": resolve_model_source(),
        "es_enabled": elastic.is_configured(),
        "es_index": ES_CONFIG.get("index"),
    }
    if elastic.is_configured():
        try:
            payload["elasticsearch"] = elastic.ping()
        except Exception as exc:
            payload["status"] = "degraded"
            payload["elasticsearch_error"] = str(exc)
    return payload


@app.get("/api/v1/demo")
def demo():
    """Operational output: reference tweets from twitter_temp_data."""
    if not elastic.is_configured():
        raise HTTPException(503, "Elasticsearch is not configured.")
    try:
        elastic.ping()
        users = elastic.fetch_active_users(limit=3, min_posts=6)
    except Exception as exc:
        raise HTTPException(503, f"Elasticsearch is unreachable: {exc}") from exc
    if not users:
        raise HTTPException(404, "No users with enough reference tweets were found.")
    results = get_engine().analyze_users(users)
    return {
        "description": "خروجی عملیاتی از توییت‌های مرجع ایندکس twitter_temp_data",
        "index": ES_CONFIG.get("index"),
        "users": users,
        "has_contradiction": [
            {"user_id": row["user_id"], "has_contradiction": row["has_contradiction"], "score": row["score"]}
            for row in results
        ],
        "contradictory_posts": [
            {"user_id": row["user_id"], "contradictory_posts": row["contradictory_posts"]}
            for row in results
        ],
        "contradiction_score": [{"user_id": row["user_id"], "score": row["score"]} for row in results],
    }


@app.post("/api/v1/users/contradiction")
def detect_contradiction(req: AnalyzeRequest):
    results = _analyze(req)
    return {
        "results": [
            {
                "user_id": row["user_id"],
                "has_contradiction": row["has_contradiction"],
                "score": row["score"],
                "post_count": row["post_count"],
            }
            for row in results
        ]
    }


@app.post("/api/v1/users/contradictory-posts")
def extract_contradictory_posts(req: AnalyzeRequest):
    results = _analyze(req)
    return {
        "results": [
            {
                "user_id": row["user_id"],
                "contradictory_posts": row["contradictory_posts"],
            }
            for row in results
        ]
    }


@app.post("/api/v1/users/contradiction-score")
def contradiction_score(req: AnalyzeRequest):
    results = _analyze(req)
    return {
        "results": [
            {"user_id": row["user_id"], "score": row["score"]}
            for row in results
        ]
    }
