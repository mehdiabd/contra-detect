import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import elastic
from src.detection.engine import ContradictionEngine


def _yes(flag: bool) -> str:
    return "بله" if flag else "خیر"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show contradiction output from org Elasticsearch tweets.")
    parser.add_argument("users", nargs="*", help="Twitter handles. If omitted, active users are taken from ES.")
    parser.add_argument("--limit", type=int, default=3)
    return parser.parse_args()


def resolve_users(args: argparse.Namespace) -> list[str]:
    if args.users:
        return [u.lstrip("@") for u in args.users]
    if not elastic.is_configured():
        raise SystemExit("Elasticsearch is not configured. Set ES_ENABLED=true and ES_API_KEY in .env")
    info = elastic.ping()
    print(f"اتصال ES برقرار شد. ایندکس={info['index']} تعداد سند={info['docs']}")
    users = elastic.fetch_active_users(limit=args.limit, min_posts=6)
    if not users:
        raise SystemExit("کاربری با توییت مرجع کافی در ایندکس پیدا نشد.")
    return users


def main() -> None:
    args = parse_args()
    users = resolve_users(args)
    print("کاربران:", ", ".join(users))
    engine = ContradictionEngine()
    results = engine.analyze_users(users)
    print("\nخروجی قابل نمایش پروژه")
    print("=" * 32)
    for row in results:
        print(f"\nکاربر: {row['user_id']}")
        print(f"تناقض دارد؟ {_yes(row['has_contradiction'])}")
        print(f"امتیاز تناقض: {row['score']}")
        print(f"تعداد توییت مرجع: {row['post_count']}")
        hits = row["contradictory_posts"]
        if not hits:
            print("پست متناقض: ندارد")
            continue
        print("پست‌های متناقض:")
        for i, hit in enumerate(hits, 1):
            a = hit["post_a"]
            b = hit["post_b"]
            print(f"  {i}) [{a['timestamp']}] {a['text']}")
            print(f"     [{b['timestamp']}] {b['text']}")
            print(f"     اطمینان: {hit['contradiction_score']}")
    print("\nJSON:")
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
