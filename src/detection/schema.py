from enum import Enum
from typing import Optional, TypedDict


class AnalysisStatus(str, Enum):
    COMPLETED = "completed"
    INSUFFICIENT_DATA = "insufficient_data"


class UserAnalysis(TypedDict):
    user_id: str
    analysis_status: AnalysisStatus
    post_count: int
    pair_count: int
    has_contradiction: Optional[bool]
    score: Optional[float]
    contradictory_posts: list[dict]
