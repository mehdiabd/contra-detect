"""Verify the deployed API against frozen Elasticsearch samples; never write to ES."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api.schemas import (
    AnalysisResponse,
    ContradictionResult,
    ContradictoryPostsResult,
    ScoreResult,
)
from src.data.schema import Post
from src.features.candidates import CandidateBuilder

ENDPOINTS = {
    "contradiction": ContradictionResult,
    "contradictory-posts": ContradictoryPostsResult,
    "contradiction-score": ScoreResult,
}


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def check_rows(payload: dict, endpoint: str, cases: list[dict], frozen: bool) -> list[str]:
    """Validate schemas and independently derived counts, IDs, and returned evidence."""
    AnalysisResponse[ENDPOINTS[endpoint]].model_validate(payload)
    rows = payload["results"]
    errors = []
    if [r["user_id"] for r in rows] != [c["user_id"] for c in cases]:
        errors.append("requested_user_order_or_coverage")
    by_user = {r["user_id"]: r for r in rows}
    for case in cases:
        row = by_user.get(case["user_id"])
        if row is None:
            continue
        raw_posts = case["review_posts" if frozen else "source_posts"]
        posts = [Post.from_dict(p) for p in raw_posts]
        pairs = CandidateBuilder().build(posts)
        expected_status = "completed" if pairs else "insufficient_data"
        if (row["post_count"], row["pair_count"], row["analysis_status"]) != (
            len(posts), len(pairs), expected_status,
        ):
            errors.append(f'{case["case_id"]}:source_counts_or_status')
        if "score" in row:
            if not pairs and row["score"] is not None:
                errors.append(f'{case["case_id"]}:missing_data_score')
            if pairs and row["score"] is None:
                errors.append(f'{case["case_id"]}:completed_score_missing')
        if "has_contradiction" in row and not pairs and row["has_contradiction"] is not None:
            errors.append(f'{case["case_id"]}:missing_data_decision')
        allowed = {(str(p.post_id), p.text, p.platform, p.timestamp) for p in posts}
        for hit in row.get("contradictory_posts", []):
            if hit.get("user_id") != case["user_id"] or hit.get("label") != "contradiction":
                errors.append(f'{case["case_id"]}:hit_identity_or_label')
            if not 0.55 <= hit.get("contradiction_score", -1) <= 1:
                errors.append(f'{case["case_id"]}:hit_threshold')
            for side in ("post_a", "post_b"):
                p = hit[side]
                if (str(p["post_id"]), p["text"], p["platform"], p["timestamp"]) not in allowed:
                    errors.append(f'{case["case_id"]}:hit_not_in_source_snapshot')
    return sorted(set(errors))


def compare_endpoints(results: dict[str, dict]) -> list[str]:
    errors = []
    if set(results) != set(ENDPOINTS):
        return ["not_all_endpoints_available"]
    rows = {name: {r["user_id"]: r for r in payload["results"]} for name, payload in results.items()}
    users = set(rows["contradiction"])
    if any(set(items) != users for items in rows.values()):
        return ["endpoint_user_coverage_mismatch"]
    for user in users:
        a, b, c = (rows[name][user] for name in ENDPOINTS)
        for key in ("analysis_status", "post_count", "pair_count"):
            if len({r[key] for r in (a, b, c)}) != 1:
                errors.append("endpoint_summary_mismatch")
        if a["score"] != c["score"]:
            errors.append("endpoint_score_mismatch")
        if a["analysis_status"] == "completed" and a["has_contradiction"] != bool(b["contradictory_posts"]):
            errors.append("endpoint_decision_evidence_mismatch")
    return sorted(set(errors))


def verify_sources(cases: list[dict], discovery: dict) -> list[dict]:
    from src.data import elastic

    checks = []
    def signature(posts):
        return Counter((p["post_id"], p["text"], p["timestamp"]) for p in posts)
    for platform in ("twitter", "telegram", "instagram"):
        chosen = [c for c in cases if c["platform"] == platform][:2]
        dates = chosen[0]["date_range"]
        posts = elastic.fetch_posts(
            [c["user_id"] for c in chosen], platforms=[platform],
            start_date=date.fromisoformat(dates["start_date"]), end_date=date.fromisoformat(dates["end_date"]),
        )
        for case in chosen:
            found = [p.to_dict() for p in posts if p.user_id == case["user_id"]]
            checks.append({"name": f'{case["case_id"]}_fixed_live_source',
                           "expected_post_count": len(case["source_posts"]), "observed_post_count": len(found),
                           "passed": signature(case["source_posts"]) == signature(found)})
    samples = discovery.get("telegram_comment_data", {}).get("hits", [])
    users = {str(h.get("_source", {}).get("from_id") or "") for h in samples} - {""}
    users = sorted(users, key=lambda user: hashlib.sha256(user.encode()).hexdigest())[:3]
    posts = [p for user in users for p in elastic._fetch_from_index("telegram", "telegram_comment_data", [user], 3, 3)]
    checks.append({"name": "telegram_comment_from_id_fixed_live_source", "passed": bool(users) and bool(posts),
                   "observed_post_count": len(posts)})
    return checks


class AcceptanceRun:
    def __init__(self, base_url: str, output: Path, private: Path):
        self.base_url = base_url.rstrip("/")
        self.output, self.private = output, private
        self.session = requests.Session()
        self.checks = []
        self.receipts = []
        self.output.mkdir(parents=True, exist_ok=True)
        self.private.mkdir(parents=True, exist_ok=True)

    def request(self, name: str, endpoint: str, body: dict, cases: list[dict], frozen: bool = False,
                expected_status: int = 200) -> dict | None:
        started = time.monotonic()
        payload = None
        parsed_ok = False
        errors = []
        status = None
        try:
            response = self.session.post(f"{self.base_url}/api/v1/users/{endpoint}", json=body, timeout=180)
            status = response.status_code
            payload = response.json()
            if status != expected_status:
                errors.append(f"unexpected_http_{status}")
            elif status == 200:
                errors.extend(check_rows(payload, endpoint, cases, frozen))
                parsed_ok = True
        except (requests.RequestException, ValueError, KeyError, TypeError) as error:
            errors.append(type(error).__name__)
        elapsed = round(time.monotonic() - started, 3)
        response_rows = payload.get("results", []) if isinstance(payload, dict) else []
        response_rows = [r for r in response_rows if isinstance(r, dict)] if isinstance(response_rows, list) else []
        row = {
            "name": name, "endpoint": endpoint, "platform": body["platform_name"],
            "mode": "frozen_posts" if "posts" in body else "live_elasticsearch",
            "http_status": status, "seconds": elapsed, "passed": not errors, "errors": errors,
            "cases": [c["case_id"] for c in cases],
            "results": [{k: v for k, v in r.items() if k not in ("user_id", "contradictory_posts")}
                        | {"hit_count": len(r.get("contradictory_posts", []))}
                        for r in response_rows],
        }
        self.checks.append(row)
        self.receipts.append({"check": name, "request": body, "status": status, "response": payload})
        self.save()
        print(json.dumps({k: row[k] for k in ("name", "passed", "errors", "seconds")}), flush=True)
        return payload if parsed_ok else None

    def save(self):
        (self.output / "checks.json").write_text(json.dumps(self.checks, ensure_ascii=False, indent=2))
        (self.private / "api_receipts.json").write_text(json.dumps(self.receipts, ensure_ascii=False, indent=2))

    def run(self, cases: list[dict]):
        response = self.session.get(f"{self.base_url}/health", timeout=60)
        response.raise_for_status()
        health = response.json()
        (self.output / "health.json").write_text(json.dumps(health, ensure_ascii=False, indent=2))
        response = self.session.get(f"{self.base_url}/openapi.json", timeout=30)
        response.raise_for_status()
        spec = response.json()
        (self.output / "openapi.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2))
        release = health.get("model", {}).get("release", {})
        index_health = health.get("elasticsearch", {}).get("indexes", {})
        self.checks.append({
            "name": "all_configured_indexes_and_model_ready",
            "passed": bool(health.get("status") == "ok" and health.get("es_enabled") and release.get("verified")
                           and set(index_health) == {"twitter", "telegram", "instagram"}
                           and all(items and all(item.get("ok") for item in items) for items in index_health.values())),
        })
        for platform in ("twitter", "telegram", "instagram"):
            platform_cases = [c for c in cases if c["platform"] == platform]
            # The same fixed first two sampled users are checked individually and as a list.
            chosen = platform_cases[:2]
            for group_name, group in (("single", chosen[:1]), ("multi", chosen)):
                body = {"user_ids": [c["user_id"] for c in group], "platform_name": platform,
                        "date_range": group[0]["date_range"]}
                results = {}
                for endpoint in ENDPOINTS:
                    payload = self.request(f"{platform}_{group_name}_{endpoint}", endpoint, body, group)
                    if payload is not None:
                        results[endpoint] = payload
                errors = compare_endpoints(results)
                self.checks.append({"name": f"{platform}_{group_name}_cross_endpoint_consistency", "passed": not errors, "errors": errors})
            empty_case = {**chosen[0], "source_posts": [], "review_posts": []}
            body = {"user_ids": [empty_case["user_id"]], "platform_name": platform, "posts": []}
            for endpoint in ENDPOINTS:
                self.request(f"{platform}_explicit_empty_{endpoint}", endpoint, body, [empty_case], frozen=True)
            missing = {**empty_case, "user_id": "acceptance_missing_a892b6d170614cc09b9110bb"}
            self.request(f"{platform}_missing_source_user", "contradiction", {
                "user_ids": [missing["user_id"]], "platform_name": platform,
                "date_range": chosen[0]["date_range"],
            }, [missing])
            self.request(f"{platform}_invalid_date", "contradiction", {
                "platform_name": platform, "date_range": {"start_date": "2026-10-26", "end_date": "2026-10-01"},
            }, [], expected_status=422)

        predictions = []
        for case in cases:
            body = {"user_ids": [case["user_id"]], "platform_name": case["platform"], "posts": case["review_posts"]}
            results = {}
            for endpoint in ("contradiction", "contradictory-posts"):
                payload = self.request(f'{case["case_id"]}_frozen_{endpoint}', endpoint, body, [case], frozen=True)
                if payload is not None:
                    results[endpoint] = payload["results"][0]
            predictions.append({"case_id": case["case_id"], "results": results})
            (self.private / "predictions.json").write_text(json.dumps(predictions, ensure_ascii=False, indent=2))

        final_response = self.session.get(f"{self.base_url}/health", timeout=60)
        final_response.raise_for_status()
        final_health = final_response.json()
        self.checks.append({"name": "release_unchanged_during_run", "passed":
                            release.get("manifest_sha256") == final_health.get("model", {}).get("release", {}).get("manifest_sha256")})
        self.save()
        summary = {
            "completed_at_utc": datetime.now(timezone.utc).isoformat(), "api_url": self.base_url,
            "api_version": spec["info"]["version"], "release_id": release.get("release_id"),
            "manifest_sha256": release.get("manifest_sha256"), "cases_sha256": fingerprint(cases),
            "checks": len(self.checks), "passed": sum(c["passed"] for c in self.checks),
            "failed": sum(not c["passed"] for c in self.checks),
            "accuracy_status": "awaiting_independent_human_labels",
            "scope": "operational_checks_and_frozen_social_media_pilot_not_organizational_acceptance",
        }
        (self.output / "summary.json").write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="https://contra-detect-api.stinascloud.ir")
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--source-only", action="store_true", help="Verify the local reader against live Elasticsearch")
    parser.add_argument("--discovery", type=Path, help="Frozen discovery pool, required with --source-only")
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text())
    if args.source_only:
        if not args.discovery:
            parser.error("--source-only requires --discovery")
        checks = verify_sources(cases, json.loads(args.discovery.read_text()))
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / "fixed_source_checks.json").write_text(json.dumps(checks, indent=2))
        print(json.dumps(checks, indent=2))
        return int(any(not row["passed"] for row in checks))
    summary = AcceptanceRun(args.base_url, args.output, args.private).run(cases)
    return int(summary["failed"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
