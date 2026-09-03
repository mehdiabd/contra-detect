from functools import lru_cache
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from config import API_CONFIG, BEST_MODEL_DIR, PLATFORMS
from src.data.schema import Post
from src.detection.engine import ContradictionEngine

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
    user_ids: list[str] = Field(..., min_length=1)
    platforms: Optional[list[str]] = None
    posts: Optional[list[PostIn]] = None


def _model_is_ready() -> bool:
    return (Path(BEST_MODEL_DIR) / "config.json").exists()


@lru_cache(maxsize=1)
def get_engine() -> ContradictionEngine:
    if not _model_is_ready():
        raise HTTPException(
            status_code=503,
            detail="Trained model not found. Run `python scripts/train.py` first.",
        )
    return ContradictionEngine()


def _analyze(req: AnalyzeRequest) -> list[dict]:
    if req.platforms:
        unknown = [p for p in req.platforms if p not in PLATFORMS]
        if unknown:
            raise HTTPException(400, f"Unsupported platforms: {unknown}. Use {list(PLATFORMS)}")
    posts = [Post.from_dict(p.model_dump()) for p in req.posts] if req.posts else None
    return get_engine().analyze_users(req.user_ids, platforms=req.platforms, posts=posts)


@app.get("/health")
def health():
    return {"status": "ok", "model_ready": _model_is_ready()}


@app.post("/api/v1/users/contradiction")
def detect_contradiction(req: AnalyzeRequest):
    """Identify whether each input user has a functional/verbal contradiction."""
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
    """Extract contradictory post pairs for a list of users across platforms."""
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
    """Return a numeric contradiction indicator in [0, 1] for each user."""
    results = _analyze(req)
    return {
        "results": [
            {"user_id": row["user_id"], "score": row["score"]}
            for row in results
        ]
    }
