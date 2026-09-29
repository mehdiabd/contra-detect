from functools import lru_cache
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from config import API_CONFIG, PLATFORMS
from src.data import elastic
from src.data.schema import Post
from src.detection.engine import ContradictionEngine
from src.models.predictor import ContradictionPredictor, resolve_model_source

DEMO_USERS = ["tw_nima", "tg_maryam", "ig_kamran"]

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


@app.on_event("startup")
def warmup_model() -> None:
    engine = get_engine()
    if engine.predictor is None:
        engine.predictor = ContradictionPredictor()


def _resolve_user_ids(req: AnalyzeRequest) -> list[str]:
    if req.user_ids:
        return [u.lstrip("@") for u in req.user_ids]
    if req.posts:
        return sorted({p.user_id.lstrip("@") for p in req.posts})
    return list(DEMO_USERS)


def _analyze(req: AnalyzeRequest) -> list[dict]:
    if req.platforms:
        unknown = [p for p in req.platforms if p not in PLATFORMS]
        if unknown:
            raise HTTPException(400, f"Unsupported platforms: {unknown}. Use {list(PLATFORMS)}")
    posts = [Post.from_dict(p.model_dump()) for p in req.posts] if req.posts else None
    try:
        return get_engine().analyze_users(_resolve_user_ids(req), platforms=req.platforms, posts=posts)
    except Exception as error:
        raise HTTPException(503, f"Failed to analyze posts: {error}") from error


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    payload = {
        "status": "ok",
        "model_source": resolve_model_source(),
        "es_enabled": elastic.is_configured(),
        "es_indexes": elastic.configured_indexes(),
    }
    if elastic.is_configured():
        try:
            payload["elasticsearch"] = elastic.ping()
            if not payload["elasticsearch"].get("ok"):
                payload["status"] = "degraded"
        except Exception as error:
            payload["status"] = "degraded"
            payload["elasticsearch_error"] = str(error)
    return payload


@app.get("/api/v1/demo")
def demo():
    """Controlled sample: one user per platform (X, Telegram, Instagram)."""
    results = get_engine().analyze_users(DEMO_USERS)
    return {
        "description": "نمونه کنترل‌شده از هر سه بستر (ایکس، تلگرام، اینستاگرام)",
        "users": DEMO_USERS,
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
