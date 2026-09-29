"""Central configuration for CONTRA-Detect."""

import importlib.machinery
import os
import sys
import types
from pathlib import Path

from dotenv import load_dotenv


def _stub_broken_scipy_stack() -> None:
    """macOS 27 breaks current scipy wheels; transformers only needs a few symbols to import."""

    def _pkg(name: str) -> types.ModuleType:
        module = sys.modules.get(name)
        if module is None:
            module = types.ModuleType(name)
            module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=True)
            module.__path__ = []
            sys.modules[name] = module
        return module

    def _mod(name: str) -> types.ModuleType:
        module = sys.modules.get(name)
        if module is None:
            module = types.ModuleType(name)
            module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
            sys.modules[name] = module
        return module

    sklearn = _pkg("sklearn")
    metrics = _mod("sklearn.metrics")
    if not hasattr(metrics, "roc_curve"):
        metrics.roc_curve = lambda *args, **kwargs: (None, None, None)
    sklearn.metrics = metrics

    optimize = _pkg("scipy.optimize")
    if not hasattr(optimize, "linear_sum_assignment"):
        optimize.linear_sum_assignment = lambda *args, **kwargs: (None, None)
    if not hasattr(optimize, "LinearConstraint"):
        optimize.LinearConstraint = object
    if not hasattr(optimize, "milp"):
        optimize.milp = lambda *args, **kwargs: None

    dummy = types.SimpleNamespace(
        slansvd=None,
        dlansvd=None,
        clansvd=None,
        zlansvd=None,
        slansvd_irl=None,
        dlansvd_irl=None,
        clansvd_irl=None,
        zlansvd_irl=None,
    )
    propack = _pkg("scipy.sparse.linalg._propack")
    for name in ("_spropack", "_dpropack", "_cpropack", "_zpropack"):
        setattr(propack, name, dummy)
        sys.modules.setdefault(f"scipy.sparse.linalg._propack.{name}", dummy)


_stub_broken_scipy_stack()
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SAMPLE_DATA_PATH = DATA_DIR / "sample" / "posts.json"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
LABELED_DATA_DIR = DATA_DIR / "labeled"
MODELS_DIR = BASE_DIR / "models"
SAVED_MODELS_DIR = MODELS_DIR / "saved"
BEST_MODEL_DIR = Path(os.getenv("MODEL_DIR") or SAVED_MODELS_DIR / "best_model")
FALLBACK_MODEL = os.getenv(
    "FALLBACK_MODEL",
    "persiannlp/parsbert-base-parsinlu-entailment",
)
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
LOGS_DIR = BASE_DIR / "logs"
FARSTAIL_DIR = RAW_DATA_DIR / "farstail"

PLATFORMS = ("twitter", "telegram", "instagram")


def _env(name: str, default: str = "") -> str:
    return (os.getenv(name, default) or "").strip()


def _env_list(name: str, default: str) -> list[str]:
    raw = _env(name, default)
    values = []
    for item in raw.split(","):
        token = item.strip()
        if token and token.lower() not in {"none", "off", "-", "false"}:
            values.append(token)
    return values


def _platform_fields(platform: str) -> dict[str, str]:
    prefix = platform.upper()
    default_user = "channel_username" if platform == "telegram" else "user_name"
    return {
        "user_id": _env(f"ES_{prefix}_FIELD_USER_ID", _env("ES_FIELD_USER_ID", default_user) if platform != "telegram" else default_user),
        "text": _env(f"ES_{prefix}_FIELD_TEXT", _env("ES_FIELD_TEXT", "normalized_text")),
        "platform": _env(f"ES_{prefix}_FIELD_PLATFORM", _env("ES_FIELD_PLATFORM", "platform")),
        "timestamp": _env(f"ES_{prefix}_FIELD_TIMESTAMP", _env("ES_FIELD_TIMESTAMP", "date")),
        "post_id": _env(f"ES_{prefix}_FIELD_POST_ID", _env("ES_FIELD_POST_ID", "post_id")),
    }


ES_CONFIG = {
    "enabled": os.getenv("ES_ENABLED", "true").lower() in {"1", "true", "yes"},
    "auth": os.getenv("ELASTIC_AUTH", os.getenv("ES_AUTH", "1")).strip() or "1",
    "hosts": [h.strip() for h in os.getenv("ES_HOSTS", "https://elastic.synappse.ir").split(",") if h.strip()],
    "index": _env("ES_INDEX", "twitter_temp_data"),
    "indexes": {
        "twitter": _env_list("ES_INDEX_TWITTER", _env("ES_INDEX", "twitter_temp_data")),
        "telegram": _env_list(
            "ES_INDEX_TELEGRAM",
            "telegram_temp_data,telegram_source,telegram_comment_data",
        ),
        "instagram": _env_list(
            "ES_INDEX_INSTAGRAM",
            "instagram_temp_data,instagram_source,instagram_comment_data",
        ),
    },
    "user": os.getenv("ES_USER", ""),
    "password": os.getenv("ES_PASSWORD", ""),
    "api_key": _env("ES_API_KEY")
    or (
        "YXYyeVRKWUJKSFpwMVdrTnZWRDc6UHhqRHBQa2ZUYW1yMnBwWTV3Ri0xUQ=="
        if (os.getenv("ELASTIC_AUTH", os.getenv("ES_AUTH", "1")).strip() or "1") == "1"
        else ""
    ),
    "verify_certs": os.getenv("ES_VERIFY_CERTS", "true").lower() not in {"0", "false", "no"},
    "start_date": os.getenv("ES_START_DATE", os.getenv("START_DATE", "")),
    "end_date": os.getenv("ES_END_DATE", os.getenv("END_DATE", "")),
    "max_posts_per_user": int(os.getenv("ES_MAX_POSTS_PER_USER", "15")),
    "reference_types": [
        t.strip()
        for t in os.getenv("ES_REFERENCE_TYPES", "post,tweet,original").split(",")
        if t.strip()
    ],
    "fields": {
        "user_id": _env("ES_FIELD_USER_ID", "user_name"),
        "text": _env("ES_FIELD_TEXT", "normalized_text"),
        "platform": _env("ES_FIELD_PLATFORM", "platform"),
        "timestamp": _env("ES_FIELD_TIMESTAMP", "date"),
        "post_id": _env("ES_FIELD_POST_ID", "post_id"),
    },
    "platform_fields": {platform: _platform_fields(platform) for platform in PLATFORMS},
}

MODEL_CONFIG = {
    "base_model": "persiannlp/parsbert-base-parsinlu-entailment",
    "max_length": 256,
    "batch_size": 16,
    "learning_rate": 1e-5,
    "num_epochs": 4,
    "weight_decay": 0.01,
    "warmup_ratio": 0.1,
    "early_stopping_patience": 3,
    "target_accuracy": 0.85,
    "num_labels": 3,
    "seed": 42,
}

DATA_SPLIT = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
    "random_state": 42,
}

FEATURE_CONFIG = {
    "tfidf_max_features": 5000,
    "tfidf_ngram_range": (1, 2),
    "min_pair_similarity": 0.12,
    "max_pair_similarity": 0.95,
    "max_pairs_per_user": 80,
    "min_text_length": 8,
}

API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "title": "CONTRA-Detect API",
    "description": "Detection of user-level verbal contradictions across X, Telegram, and Instagram.",
    "version": "1.0.0",
}

CONTRADICTION_THRESHOLDS = {
    "pair": 0.55,
    "user": 0.50,
}

NLI_LABEL2ID = {
    "entailment": 0,
    "contradiction": 1,
    "neutral": 2,
    "e": 0,
    "c": 1,
    "n": 2,
}
NLI_ID2LABEL = {0: "entailment", 1: "contradiction", 2: "neutral"}
CONTRADICTION_LABEL_ID = 1

OPTUNA_CONFIG = {
    "n_trials": 5,
    "timeout": 3600,
    "study_name": "contra-detect",
}

FARSTAIL_URLS = {
    "train": "https://raw.githubusercontent.com/dml-qom/FarsTail/master/data/Train-word.csv",
    "val": "https://raw.githubusercontent.com/dml-qom/FarsTail/master/data/Val-word.csv",
    "test": "https://raw.githubusercontent.com/dml-qom/FarsTail/master/data/Test-word.csv",
}
