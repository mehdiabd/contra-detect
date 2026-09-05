"""Central configuration for CONTRA-Detect."""

import os
from pathlib import Path

from dotenv import load_dotenv

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

ES_CONFIG = {
    "enabled": os.getenv("ES_ENABLED", "true").lower() in {"1", "true", "yes"},
    "auth": os.getenv("ELASTIC_AUTH", os.getenv("ES_AUTH", "1")).strip() or "1",
    "hosts": [h.strip() for h in os.getenv("ES_HOSTS", "https://elastic.synappse.ir").split(",") if h.strip()],
    "index": os.getenv("ES_INDEX", "twitter_temp_data"),
    "user": os.getenv("ES_USER", ""),
    "password": os.getenv("ES_PASSWORD", ""),
    "api_key": os.getenv("ES_API_KEY", ""),
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
        "user_id": os.getenv("ES_FIELD_USER_ID", "user_name"),
        "text": os.getenv("ES_FIELD_TEXT", "normalized_text"),
        "platform": os.getenv("ES_FIELD_PLATFORM", "platform"),
        "timestamp": os.getenv("ES_FIELD_TIMESTAMP", "date"),
        "post_id": os.getenv("ES_FIELD_POST_ID", "post_id"),
    },
}

MODEL_CONFIG = {
    "base_model": "HooshvareLab/bert-fa-base-uncased",
    "max_length": 256,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "num_epochs": 3,
    "weight_decay": 0.01,
    "early_stopping_patience": 2,
    "target_accuracy": 0.85,
    "num_labels": 3,
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
