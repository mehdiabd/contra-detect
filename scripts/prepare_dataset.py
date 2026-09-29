import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import FARSTAIL_DIR, FARSTAIL_URLS, LOGS_DIR, NLI_ID2LABEL
from src.data.loader import load_farstail, summarize_pairs


STATS_PATH = ROOT / "reports" / "farstail_dataset_stats.json"


def _print_split(name: str, stats: dict) -> None:
    labels = stats.get("labels") or {}
    print(f"{name:5}  n={stats['count']:<6}  " + "  ".join(f"{k}={labels.get(k, 0)}" for k in NLI_ID2LABEL.values()))


def main() -> None:
    FARSTAIL_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)

    print(f"downloading FarsTail into {FARSTAIL_DIR}")
    for split, url in FARSTAIL_URLS.items():
        print(f"  {split}: {url}")

    train, val, test = load_farstail()
    payload = {
        "source": "FarsTail",
        "citation": "https://github.com/dml-qom/FarsTail",
        "role": "official labeled training/evaluation set for CONTRA-Detect",
        "note": "Social-media posts from Elasticsearch are unlabeled inference data, not the gold test set.",
        "splits": {
            "train": summarize_pairs(train),
            "val": summarize_pairs(val),
            "test": summarize_pairs(test),
        },
        "total": summarize_pairs([*train, *val, *test]),
        "files": {split: str(FARSTAIL_DIR / f"{split}.csv") for split in FARSTAIL_URLS},
    }
    STATS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print()
    print("FarsTail splits (after dropping empty/duplicate pairs)")
    print("-" * 72)
    for name in ("train", "val", "test"):
        _print_split(name, payload["splits"][name])
    _print_split("all", payload["total"])
    print()
    print("class balance (share of each split):")
    for name in ("train", "val", "test"):
        balance = payload["splits"][name]["balance"]
        parts = "  ".join(f"{k}={v:.1%}" for k, v in balance.items())
        print(f"  {name:5}  {parts}")
    print(f"\nwrote {STATS_PATH}")


if __name__ == "__main__":
    main()
