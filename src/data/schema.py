from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from config import PLATFORMS

PLATFORM_ALIASES = {
    "x": "twitter",
    "twitter": "twitter",
    "tweet": "twitter",
    "telegram": "telegram",
    "tg": "telegram",
    "instagram": "instagram",
    "ig": "instagram",
    "insta": "instagram",
}


def normalize_platform(value: Optional[str]) -> str:
    if not value:
        return "twitter"
    key = str(value).strip().lower()
    platform = PLATFORM_ALIASES.get(key, key)
    if platform not in PLATFORMS:
        raise ValueError(f"Unsupported platform '{value}'. Use one of: {PLATFORMS}")
    return platform


@dataclass
class Post:
    user_id: str
    text: str
    platform: str = "twitter"
    post_id: str = ""
    timestamp: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.user_id = str(self.user_id)
        self.platform = normalize_platform(self.platform)
        self.text = (self.text or "").strip()
        if not self.post_id:
            self.post_id = f"{self.user_id}:{self.platform}:{abs(hash(self.text))}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Post":
        meta = dict(raw.get("metadata") or {})
        known = {"user_id", "text", "platform", "post_id", "timestamp", "metadata"}
        for key, value in raw.items():
            if key not in known:
                meta.setdefault(key, value)
        return cls(
            user_id=raw.get("user_id") or raw.get("user") or "",
            text=raw.get("text") or raw.get("content") or "",
            platform=raw.get("platform") or "twitter",
            post_id=str(raw.get("post_id") or raw.get("id") or ""),
            timestamp=raw.get("timestamp") or raw.get("created_at"),
            metadata=meta,
        )


@dataclass
class Pair:
    text_a: str
    text_b: str
    label: Optional[int] = None
    user_id: Optional[str] = None
    post_id_a: Optional[str] = None
    post_id_b: Optional[str] = None
    platform_a: Optional[str] = None
    platform_b: Optional[str] = None
    timestamp_a: Optional[str] = None
    timestamp_b: Optional[str] = None
    similarity: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
