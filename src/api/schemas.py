from datetime import date
from typing import Generic, Literal, Optional, TypeVar

from pydantic import BaseModel, Field, field_validator, model_validator

from src.data.schema import normalize_platform
from src.detection.schema import AnalysisStatus


class PostIn(BaseModel):
    user_id: str
    text: str
    platform: Optional[str] = None
    post_id: str = ""
    timestamp: Optional[str] = None


class DateRange(BaseModel):
    start_date: Optional[date] = None
    end_date: Optional[date] = None

    @model_validator(mode="after")
    def validate_order(self):
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError("start_date must be before or equal to end_date")
        return self


class AnalyzeRequest(BaseModel):
    user_ids: list[str] = Field(default_factory=list)
    platform_name: str
    date_range: Optional[DateRange] = None
    posts: Optional[list[PostIn]] = None

    @field_validator("platform_name")
    @classmethod
    def validate_platform(cls, value: str) -> str:
        return normalize_platform(value)

    @field_validator("user_ids")
    @classmethod
    def normalize_user_ids(cls, values: list[str]) -> list[str]:
        return list(dict.fromkeys(str(value).strip().lstrip("@") for value in values if str(value).strip()))


class AnalysisSummary(BaseModel):
    user_id: str
    analysis_status: AnalysisStatus = Field(description="completed only when at least one post pair was analyzed")
    post_count: int = Field(ge=0)
    pair_count: int = Field(ge=0)


class ContradictionResult(AnalysisSummary):
    has_contradiction: Optional[bool] = Field(description="null when there is insufficient data")
    score: Optional[float] = Field(ge=0, le=1, description="null when there is insufficient data")


class ContradictoryPostsResult(AnalysisSummary):
    contradictory_posts: list[dict]


class ScoreResult(AnalysisSummary):
    score: Optional[float] = Field(ge=0, le=1, description="null when there is insufficient data")


ResultType = TypeVar("ResultType", bound=AnalysisSummary)


class AnalysisResponse(BaseModel, Generic[ResultType]):
    results: list[ResultType]


class AnalysisError(BaseModel):
    code: Literal["data_source_unavailable", "analysis_failed"]
    message: str


class ErrorResponse(BaseModel):
    detail: AnalysisError
