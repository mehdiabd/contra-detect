import csv
import json
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import pandas as pd
from sklearn.model_selection import train_test_split

from config import DATA_SPLIT, FARSTAIL_DIR, FARSTAIL_URLS, NLI_ID2LABEL, NLI_LABEL2ID, SAMPLE_DATA_PATH
from src.data.schema import Pair, Post

PathLike = Union[str, Path]


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_posts_json(path: PathLike = SAMPLE_DATA_PATH) -> list[Post]:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    rows = payload["posts"] if isinstance(payload, dict) and "posts" in payload else payload
    return [Post.from_dict(row) for row in rows]


def load_posts_csv(path: PathLike) -> list[Post]:
    df = pd.read_csv(path)
    return [Post.from_dict(row) for row in df.to_dict(orient="records")]


def filter_posts(
    posts: Sequence[Post],
    user_ids: Optional[Iterable[str]] = None,
    platforms: Optional[Iterable[str]] = None,
) -> list[Post]:
    user_filter = {str(u) for u in user_ids} if user_ids else None
    platform_filter = set(platforms) if platforms else None
    out = []
    for post in posts:
        if user_filter is not None and post.user_id not in user_filter:
            continue
        if platform_filter is not None and post.platform not in platform_filter:
            continue
        if post.text:
            out.append(post)
    return out


def load_pairs_csv(path: PathLike) -> list[Pair]:
    df = pd.read_csv(path)
    pairs: list[Pair] = []
    for row in df.to_dict(orient="records"):
        label = row.get("label")
        if isinstance(label, str):
            label = NLI_LABEL2ID.get(label.strip().lower())
        elif pd.isna(label):
            label = None
        else:
            label = int(label)
        pairs.append(
            Pair(
                text_a=str(row.get("text_a") or row.get("premise") or ""),
                text_b=str(row.get("text_b") or row.get("hypothesis") or ""),
                label=label,
                user_id=row.get("user_id"),
                post_id_a=row.get("post_id_a"),
                post_id_b=row.get("post_id_b"),
                platform_a=row.get("platform_a"),
                platform_b=row.get("platform_b"),
                similarity=row.get("similarity"),
            )
        )
    return [p for p in pairs if p.text_a and p.text_b]


def save_pairs_csv(pairs: Sequence[Pair], path: PathLike) -> Path:
    path = Path(path)
    ensure_dir(path.parent)
    fieldnames = [
        "user_id",
        "text_a",
        "text_b",
        "label",
        "post_id_a",
        "post_id_b",
        "platform_a",
        "platform_b",
        "timestamp_a",
        "timestamp_b",
        "similarity",
    ]
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for pair in pairs:
            writer.writerow(pair.to_dict())
    return path


def split_pairs(pairs: Sequence[Pair]) -> tuple[list[Pair], list[Pair], list[Pair]]:
    labeled = [p for p in pairs if p.label is not None]
    if len(labeled) < 10:
        raise ValueError("Need at least 10 labeled pairs to split.")
    labels = [p.label for p in labeled]
    train, temp = train_test_split(
        labeled,
        test_size=1.0 - DATA_SPLIT["train"],
        random_state=DATA_SPLIT["random_state"],
        stratify=labels,
    )
    relative_test = DATA_SPLIT["test"] / (DATA_SPLIT["val"] + DATA_SPLIT["test"])
    val, test = train_test_split(
        temp,
        test_size=relative_test,
        random_state=DATA_SPLIT["random_state"],
        stratify=[p.label for p in temp],
    )
    return train, val, test


def _pick_column(columns: Sequence[str], candidates: Sequence[str]) -> str:
    lookup = {c.lower().strip(): c for c in columns}
    for name in candidates:
        if name.lower() in lookup:
            return lookup[name.lower()]
    raise ValueError(f"Could not find any of {candidates} in {list(columns)}")


def _download(url: str, dest: Path) -> Path:
    import requests

    ensure_dir(dest.parent)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    dest.write_bytes(response.content)
    return dest


def load_farstail_split(split: str, dest_dir: Path = FARSTAIL_DIR) -> list[Pair]:
    if split not in FARSTAIL_URLS:
        raise ValueError(f"Unknown FarsTail split '{split}'")
    ensure_dir(dest_dir)
    path = dest_dir / f"{split}.csv"
    if not path.exists():
        _download(FARSTAIL_URLS[split], path)
    df = pd.read_csv(path, sep="\t", encoding="utf-8")
    if len(df.columns) == 1:
        df = pd.read_csv(path, encoding="utf-8")
    premise_col = _pick_column(df.columns, ["premise", "text_a", "sentence1"])
    hypo_col = _pick_column(df.columns, ["hypothesis", "text_b", "sentence2"])
    label_col = _pick_column(df.columns, ["label", "gold_label"])
    pairs: list[Pair] = []
    for row in df.to_dict(orient="records"):
        raw_label = row.get(label_col)
        if isinstance(raw_label, (int, float)) and not pd.isna(raw_label):
            label = int(raw_label)
        else:
            label = NLI_LABEL2ID.get(str(raw_label).strip().lower())
        if label not in NLI_ID2LABEL:
            continue
        pairs.append(
            Pair(
                text_a=str(row.get(premise_col, "")),
                text_b=str(row.get(hypo_col, "")),
                label=label,
            )
        )
    return pairs


def load_farstail() -> tuple[list[Pair], list[Pair], list[Pair]]:
    return (
        load_farstail_split("train"),
        load_farstail_split("val"),
        load_farstail_split("test"),
    )
