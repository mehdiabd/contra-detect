from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse

from config import API_CONFIG
from src.api.schemas import (
    AnalysisResponse,
    AnalyzeRequest,
    ContradictionResult,
    ContradictoryPostsResult,
    ErrorResponse,
    ScoreResult,
)
from src.data import elastic
from src.data.errors import DataSourceError
from src.data.schema import Post
from src.detection.engine import ContradictionEngine
from src.detection.schema import UserAnalysis
from src.models.predictor import ContradictionPredictor

DEMO_USERS = ["tw_nima", "tg_maryam", "ig_kamran"]

app = FastAPI(
    title=API_CONFIG["title"],
    description=API_CONFIG["description"],
    version=API_CONFIG["version"],
)


@lru_cache(maxsize=1)
def get_engine() -> ContradictionEngine:
    return ContradictionEngine()


@app.on_event("startup")
def warmup_model() -> None:
    engine = get_engine()
    if engine.predictor is None:
        engine.predictor = ContradictionPredictor()


def _analyze(req: AnalyzeRequest) -> list[UserAnalysis]:
    posts = None
    if req.posts is not None:
        posts = []
        for item in req.posts:
            raw = item.model_dump()
            raw["platform"] = raw.get("platform") or req.platform_name
            posts.append(Post.from_dict(raw))
    start_date = req.date_range.start_date if req.date_range else None
    end_date = req.date_range.end_date if req.date_range else None
    try:
        return get_engine().analyze_users(
            req.user_ids,
            platforms=[req.platform_name],
            posts=posts,
            start_date=start_date,
            end_date=end_date,
        )
    except DataSourceError as error:
        raise HTTPException(503, detail={
            "code": "data_source_unavailable",
            "message": "Post data could not be retrieved completely. Please retry.",
        }) from error
    except Exception as error:
        raise HTTPException(503, detail={
            "code": "analysis_failed",
            "message": "Post analysis could not be completed. Please retry.",
        }) from error


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    predictor = get_engine().predictor
    if predictor is None:
        return JSONResponse(status_code=503, content={"status": "not_ready", "reason": "Model is not loaded"})
    payload = {
        "status": "ok",
        "model_source": predictor.bundle_dir or predictor.model_dir,
        "model": {
            "sources": predictor.model_sources,
            "weights": predictor.member_weights,
            "max_length": predictor.max_length,
            "release": predictor.release_info,
        },
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


@app.post(
    "/api/v1/users/contradiction",
    response_model=AnalysisResponse[ContradictionResult],
    responses={503: {"model": ErrorResponse}},
)
def detect_contradiction(req: AnalyzeRequest):
    return {"results": _analyze(req)}


@app.post(
    "/api/v1/users/contradictory-posts",
    response_model=AnalysisResponse[ContradictoryPostsResult],
    responses={503: {"model": ErrorResponse}},
)
def extract_contradictory_posts(req: AnalyzeRequest):
    return {"results": _analyze(req)}


@app.post(
    "/api/v1/users/contradiction-score",
    response_model=AnalysisResponse[ScoreResult],
    responses={503: {"model": ErrorResponse}},
)
def contradiction_score(req: AnalyzeRequest):
    return {"results": _analyze(req)}
