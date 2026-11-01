"""Create a blind Persian review form and score independent binary human labels."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import sys
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.schema import Post
from src.features.candidates import CandidateBuilder

LABELS = {"contradiction", "not_contradiction", "uncertain"}


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def pair_key(a: dict, b: dict) -> frozenset:
    return frozenset((str(p["post_id"]), p["text"]) for p in (a, b))


def review_pairs(cases: list[dict]) -> list[dict]:
    rows = []
    for case in cases:
        for number, (a, b) in enumerate(combinations(range(len(case["review_posts"])), 2), 1):
            rows.append({"pair_id": f'{case["case_id"]}-p{number:02}', "case_id": case["case_id"],
                         "platform": case["platform"], "a": a, "b": b})
    return rows


def render_form(cases: list[dict], destination: Path):
    rows = review_pairs(cases)
    digits = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")
    counts = (f"این بسته شامل {len(cases)} حساب، {sum(len(c['review_posts']) for c in cases)} نوشته "
              f"و {len(rows)} جفت از دادهٔ واقعی است.").translate(digits)
    cards = []
    names = {"twitter": "ایکس", "telegram": "تلگرام", "instagram": "اینستاگرام"}
    for case in cases:
        posts = "".join(
            f'<article><h3>نوشتهٔ {i + 1}</h3><p class="date">تاریخ: <bdi dir="ltr">{html.escape(p["timestamp"] or "—")}</bdi></p>'
            f'<p class="text">{html.escape(p["text"])}</p></article>'
            for i, p in enumerate(case["review_posts"])
        )
        controls = "".join(
            f'<tr><td>نوشتهٔ {row["a"] + 1} و {row["b"] + 1}</td><td><select id="{row["pair_id"]}" aria-label="برچسب جفت">'
            '<option value="">انتخاب کنید</option><option value="contradiction">تناقض دارد</option>'
            '<option value="not_contradiction">تناقض ندارد</option><option value="uncertain">اطلاعات کافی نیست</option></select></td>'
            f'<td><input id="{row["pair_id"]}-reason" aria-label="دلیل برچسب" placeholder="دلیل کوتاه یا ابهام"></td></tr>'
            for row in rows if row["case_id"] == case["case_id"]
        )
        cards.append(f'<section><h2>نمونهٔ <bdi dir="ltr">{case["case_id"]}</bdi> — {names[case["platform"]]}</h2>'
                     f'<p>این چهار نوشته از یک حساب در یک بستر هستند. همهٔ شش جفت را بررسی کنید.</p><div class="posts">{posts}</div>'
                     f'<table><thead><tr><th>جفت نوشته‌ها</th><th>نظر شما</th><th>دلیل</th></tr></thead><tbody>{controls}</tbody></table></section>')
    document = '''<!doctype html><html lang="fa" dir="rtl"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>ارزیابی انسانی تشخیص تناقض</title>
<style>body{font-family:Tahoma,Arial,sans-serif;line-height:1.9;color:#172936;background:#f2f5f7;margin:0;direction:rtl;text-align:right}main{max-width:1050px;margin:auto;padding:24px}header,section{background:white;padding:24px;margin:16px 0;border-radius:12px;border:1px solid #dce5e8}h1{font-size:25px}h2{font-size:20px;color:#164f62}h3{font-size:16px;margin:0}.posts{display:grid;grid-template-columns:1fr 1fr;gap:12px}article{padding:14px;background:#f5f8fa;border-radius:8px}.text{white-space:pre-wrap;overflow-wrap:anywhere}.date{font-size:12px;color:#526575}table{width:100%;border-collapse:collapse;margin-top:16px}td,th{padding:10px;border-bottom:1px solid #dce5e8;text-align:right}input,select,button{font:inherit;padding:8px;max-width:100%;box-sizing:border-box}input,select{width:100%;direction:rtl}button{background:#176477;color:white;border:0;border-radius:6px;cursor:pointer}.toolbar{display:flex;gap:12px;align-items:center;flex-wrap:wrap}.toolbar input{width:auto}label{display:block}#status{font-weight:bold}@media(max-width:650px){.posts{grid-template-columns:1fr}main{padding:8px}section,header{padding:12px}td,th{padding:5px}}@media print{button,.toolbar{display:none}section{break-before:page}article{break-inside:avoid}body{background:white}}</style>
<main><header><h1>فرم ارزیابی انسانی تشخیص تناقض</h1>
<p>''' + counts + ''' پاسخ مدل در این فرم نمایش داده نمی‌شود. این نمونهٔ مقدماتی، جایگزین مجموعهٔ پذیرش مورد تأیید سازمان نیست.</p>
<p>تناقض یعنی دو ادعا دربارهٔ موضوع و شرایط یکسان نتوانند هم‌زمان درست باشند. تفاوت موضوع، قیمت دو کالای متفاوت، سؤال، نقل‌قول یا صرفاً تفاوت احساس، به‌تنهایی تناقض نیست. اگر زمان، گوینده یا زمینه روشن نیست، گزینهٔ «اطلاعات کافی نیست» را بزنید و دلیل بنویسید.</p>
<p>هر ارزیاب فرم را مستقل تکمیل کند. دو ارزیاب پاسخ‌های خود را جداگانه ذخیره کنند؛ اختلاف‌ها با نظر داور انسانی حل شوند و فایل نهاییِ مورد توافق برای محاسبهٔ نتیجه استفاده شود. شناسهٔ حساب‌ها نمایش داده نمی‌شود، اما متن‌ها ممکن است اطلاعات شخصی داشته باشند؛ فایل را فقط در اختیار ارزیابان پروژه بگذارید.</p>
<div class="toolbar"><label>نام یا شناسهٔ ارزیاب <input id="reviewer" autocomplete="off"></label><button id="save" type="button">ذخیرهٔ پاسخ‌ها</button><label>ادامه از فایل ذخیره‌شده <input id="load" type="file" accept=".json"></label></div>
<p id="status" role="status">پاسخ‌ها را پیش از بستن صفحه ذخیره کنید.</p></header>''' + "".join(cards) + '''</main>
<script>
const pairIds=PAIR_IDS;
const casesSha=CASES_SHA;
const status=document.getElementById('status');
document.getElementById('save').onclick=()=>{
 const reviewer=document.getElementById('reviewer').value.trim();
 if(!reviewer){status.textContent='نام یا شناسهٔ ارزیاب را وارد کنید.';return;}
 const labels=pairIds.map(pair_id=>({pair_id,label:document.getElementById(pair_id).value,reason:document.getElementById(pair_id+'-reason').value.trim()}));
 const payload={cases_sha256:casesSha,reviewer,labels};
 const blob=new Blob([JSON.stringify(payload,null,2)],{type:'application/json;charset=utf-8'});
 const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='human-labels.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);
 status.textContent='فایل پاسخ‌ها ذخیره شد. تعداد جفت‌های پاسخ‌داده‌شده: '+labels.filter(x=>x.label).length+' از '+pairIds.length;
};
document.getElementById('load').onchange=async(event)=>{
 try{
  const payload=JSON.parse(await event.target.files[0].text());
  if(payload.cases_sha256!==casesSha)throw Error();
  const allowed=new Set(['','contradiction','not_contradiction','uncertain']);
  if(!Array.isArray(payload.labels)||payload.labels.some(x=>!pairIds.includes(x.pair_id)||!allowed.has(x.label)))throw Error();
  document.getElementById('reviewer').value=payload.reviewer||'';
  for(const row of payload.labels){document.getElementById(row.pair_id).value=row.label;document.getElementById(row.pair_id+'-reason').value=row.reason||'';}
  status.textContent='پاسخ‌های ذخیره‌شده بارگذاری شدند.';
 }catch{status.textContent='فایل معتبر نیست یا به این بسته تعلق ندارد.';}
};
</script></html>'''
    document = document.replace("const pairIds=PAIR_IDS;", f'const pairIds={json.dumps([r["pair_id"] for r in rows])};')
    document = document.replace("const casesSha=CASES_SHA;", f'const casesSha={json.dumps(fingerprint(cases))};')
    destination.write_text(document)


def binary_metrics(truth: list[bool], predictions: list[bool]) -> dict:
    if len(truth) != len(predictions):
        raise ValueError("Prediction and label lengths differ.")
    tp = sum(t and p for t, p in zip(truth, predictions))
    tn = sum(not t and not p for t, p in zip(truth, predictions))
    fp = sum(not t and p for t, p in zip(truth, predictions))
    fn = sum(t and not p for t, p in zip(truth, predictions))
    n = len(truth)
    accuracy = (tp + tn) / n if n else None
    interval = None
    if n:
        z = 1.959963984540054
        center = (accuracy + z*z/(2*n)) / (1 + z*z/n)
        margin = z * math.sqrt(accuracy*(1-accuracy)/n + z*z/(4*n*n)) / (1 + z*z/n)
        interval = [center - margin, center + margin]
    return {"n": n, "tp": tp, "tn": tn, "fp": fp, "fn": fn, "accuracy": accuracy,
            "precision": tp/(tp+fp) if tp+fp else None,
            "recall": tp/(tp+fn) if tp+fn else None,
            "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,
            "wilson_95_percent": interval}


def score(cases: list[dict], predictions: list[dict], labels: dict) -> dict:
    if labels.get("cases_sha256") != fingerprint(cases):
        raise ValueError("Human labels belong to another case snapshot.")
    if not str(labels.get("reviewer") or "").strip():
        raise ValueError("An independent human reviewer must be identified.")
    rows = review_pairs(cases)
    expected = {r["pair_id"] for r in rows}
    human = {}
    for row in labels.get("labels", []):
        pair_id, label = row["pair_id"], row["label"]
        if pair_id not in expected or pair_id in human or label not in LABELS | {""}:
            raise ValueError("Unknown, duplicated, or invalid human label.")
        human[pair_id] = label
    predicted = {r["case_id"]: r["results"] for r in predictions}
    by_case = {c["case_id"]: c for c in cases}
    pair_results = []
    for row in rows:
        case = by_case[row["case_id"]]
        label = human.get(row["pair_id"])
        if label not in LABELS - {"uncertain"}:
            continue
        result = predicted[case["case_id"]]
        decision = result["contradiction"]
        evidence = result["contradictory-posts"]
        if any(r["analysis_status"] != "completed" or r["post_count"] != len(case["review_posts"])
               for r in (decision, evidence)):
            raise ValueError("A frozen case has incomplete model predictions.")
        hits = {pair_key(h["post_a"], h["post_b"]) for h in evidence["contradictory_posts"]}
        posts = case["review_posts"]
        key = pair_key(posts[row["a"]], posts[row["b"]])
        candidates = CandidateBuilder().build([Post.from_dict(p) for p in posts])
        selected = {frozenset(((str(p.post_id_a), p.text_a), (str(p.post_id_b), p.text_b))) for p in candidates}
        pair_results.append({**row, "truth": label == "contradiction", "prediction": key in hits,
                             "candidate_selected": key in selected})
    platform_metrics = {}
    for platform in ("twitter", "telegram", "instagram", "all"):
        selected = [r for r in pair_results if platform == "all" or r["platform"] == platform]
        platform_metrics[platform] = binary_metrics([r["truth"] for r in selected], [r["prediction"] for r in selected])
    user_results = []
    for case in cases:
        required = [r for r in rows if r["case_id"] == case["case_id"]]
        annotated = [r for r in pair_results if r["case_id"] == case["case_id"]]
        if len(annotated) == len(required):
            user_results.append({"platform": case["platform"], "truth": any(r["truth"] for r in annotated),
                                 "prediction": predicted[case["case_id"]]["contradiction"]["has_contradiction"]})
    user_metrics = {
        platform: binary_metrics([r["truth"] for r in user_results if platform == "all" or r["platform"] == platform],
                                 [r["prediction"] for r in user_results if platform == "all" or r["platform"] == platform])
        for platform in ("twitter", "telegram", "instagram", "all")
    }
    complete = len(pair_results) == len(rows)
    both_classes = all(m["tp"] + m["fn"] > 0 and m["tn"] + m["fp"] > 0
                       for metrics in (platform_metrics, user_metrics)
                       for platform, m in metrics.items() if platform != "all")
    targets = {scope: {platform: m["accuracy"] is not None and m["accuracy"] >= .85
                       for platform, m in metrics.items()}
               for scope, metrics in (("pairs", platform_metrics), ("users", user_metrics))}
    passed = complete and both_classes and all(value for scope in targets.values() for value in scope.values())
    return {"status": "pilot_target_met_pending_organizational_approval" if passed else
            "pilot_target_not_met" if complete and both_classes else "awaiting_complete_adjudicated_labels_or_class_coverage",
            "cases_sha256": fingerprint(cases), "reviewer": labels["reviewer"],
            "expected_pairs": len(rows), "scored_pairs": len(pair_results),
            "uncertain_pairs": sum(human.get(r["pair_id"]) == "uncertain" for r in rows),
            "pair_metrics": platform_metrics, "user_metrics": user_metrics,
            "proposed_accuracy_85_checks": targets,
            "gold_contradictions_not_selected": sum(r["truth"] and not r["candidate_selected"] for r in pair_results),
            "limitations": ["Small recent-user pilot; not a representative random population sample.",
                            "Four-post windows only; no claim about a person's entire history.",
                            "Wilson intervals assume independent observations; pairs share users.",
                            "An accuracy target alone does not establish adequate contradiction recall.",
                            "Organizational approval and operational checks are separate requirements."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "score"))
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--predictions", type=Path)
    args = parser.parse_args()
    cases = json.loads(args.cases.read_text())
    if args.command == "prepare":
        render_form(cases, args.output)
    else:
        if not args.labels or not args.predictions:
            parser.error("score requires --labels and --predictions")
        result = score(cases, json.loads(args.predictions.read_text()), json.loads(args.labels.read_text()))
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
        print(json.dumps({k: result[k] for k in ("status", "expected_pairs", "scored_pairs")}))


if __name__ == "__main__":
    main()
