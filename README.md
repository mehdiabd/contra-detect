# CONTRA-Detect

تشخیص تناقض گفتاری/عملکردی یک کاربر در پست‌های متنی ایکس، تلگرام و اینستاگرام. تناقض یعنی دو گفته از **یک کاربر** که از نظر معنایی با هم ناسازگارند. تصویر و ویدئو خارج از محدوده است.

## اجرا

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

دانلود و تأیید دیتاست برچسب‌دار [FarsTail](https://github.com/dml-qom/FarsTail) (آموزش و ادعای دقت فقط روی همین مجموعه است؛ پست‌های Elasticsearch لیبل طلایی ندارند):

```bash
python scripts/prepare_dataset.py
python scripts/train.py
python scripts/evaluate.py
```

آمار FarsTail بعد از پاک‌سازی جفت خالی/تکراری:

| split | n | entailment | contradiction | neutral |
|---|---:|---:|---:|---:|
| train | 7266 | 2429 | 2389 | 2448 |
| val | 1537 | 515 | 499 | 523 |
| test | 1564 | 519 | 510 | 535 |

آموزش کامل FarsTail روی CPU چند ساعت طول می‌کشد. برای اطمینان از درست بودن حلقه آموزش:

```bash
python scripts/train.py --max-samples 32 --epochs 1
```

سرویس:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

مستندات تعاملی: `http://localhost:8000/docs`

## استقرار با Docker

حداقل منابع پیشنهادی: ۲ هسته CPU، ۸ گیگابایت RAM، چند گیگابایت دیسک برای مدل ParsBERT.

```bash
cp .env.example .env
python scripts/train.py          # یک‌بار، تا models/saved/best_model ساخته شود
docker compose up --build -d
```

سلامت سرویس: `GET http://localhost:8000/health`  
مستندات: `http://localhost:8000/docs`

مدل از حجم `./models/saved` به کانتینر mount می‌شود. اگر مدل محلی نباشد، سرویس از `FALLBACK_MODEL` استفاده می‌کند.

## API

هر سه اندپوینت `platform_name`، لیست `user_ids` و بازهٔ اختیاری `date_range` می‌گیرند. لیست می‌تواند یک یا چند شناسه داشته باشد. اگر خالی باشد، کاربران فعال همان پلتفرم بررسی و نتایج بر اساس بیشترین امتیاز تناقض مرتب می‌شوند. اگر `posts` در بدنه باشد همان استفاده می‌شود؛ وگرنه در صورت `ES_ENABLED=true` از Elasticsearch و در غیر این صورت از `data/sample/posts.json` خوانده می‌شود.

| روش | مسیر | خروجی |
|---|---|---|
| POST | `/api/v1/users/contradiction` | وجود/عدم تناقض + امتیاز |
| POST | `/api/v1/users/contradictory-posts` | جفت پست‌های متناقض |
| POST | `/api/v1/users/contradiction-score` | نشان‌گر کمی در `[0, 1]` |

نمونه:

```bash
curl -s http://localhost:8000/api/v1/users/contradiction \
  -H 'Content-Type: application/json' \
  -d '{
    "user_ids": ["u_ali", "u_reza"],
    "platform_name": "twitter",
    "date_range": {
      "start_date": "2026-01-01",
      "end_date": "2026-01-31"
    }
  }'
```

برای دریافت برترین تناقض‌های یک پلتفرم، `user_ids` را خالی بفرستید:

```json
{
  "user_ids": [],
  "platform_name": "telegram"
}
```

## داده

- **آموزش و ارزیابی رسمی:** FarsTail (NLI فارسی: entailment / contradiction / neutral). آمار splitها بعد از `python scripts/prepare_dataset.py` در `reports/farstail_dataset_stats.json` ذخیره می‌شود.
- **سورس عملیاتی استنتاج:** ایندکس‌های Elasticsearch سازمان برای ایکس، تلگرام و اینستاگرام. نام ایندکس‌ها در `.env`:

```
ES_INDEX_TWITTER=twitter_temp_data
ES_INDEX_TELEGRAM=telegram_temp_data,telegram_source,telegram_comment_data
ES_INDEX_INSTAGRAM=instagram_temp_data,instagram_source,instagram_comment_data
```

آدرس و اعتبار همان کلاینت echo chamber است: `https://elastic.synappse.ir` با `ELASTIC_AUTH=1`. برای خاموش کردن یک بستر مقدار را `none` بگذارید. اگر فیلدهای تلگرام/اینستاگرام با توییتر فرق دارند، `ES_TELEGRAM_FIELD_USER_ID` و مشابه آن را در `.env` تنظیم کنید. این پست‌ها لیبل طلایی ندارند.
- جفت‌سازی پست‌های واقعی: `python scripts/build_pairs.py`
- برچسب ضعیف با مدل آموزش‌دیده: `python scripts/build_pairs.py --weak-label`
- داده سفارشی برچسب‌خورده: `python scripts/train.py --source csv --pairs data/labeled/pairs.csv`

مدل با `save_pretrained` ذخیره می‌شود (PyTorch / Hugging Face)، نه `.h5`.

## دقت ۸۵٪

هدف پروپوزال روی **تست FarsTail** اندازه‌گیری شد. نتیجه fine-tune ParsBERT:

| مجموعه | Accuracy | F1 | Precision | Recall |
|---|---:|---:|---:|---:|
| val | 0.8211 | 0.8185 | 0.8220 | 0.8196 |
| test | 0.8152 | 0.8127 | 0.8157 | 0.8141 |

هدف ۸۵٪ روی تست حاصل نشد (فاصله ۳٫۴۸ واحد). جزئیات در `reports/farstail_test_metrics.json`. روی پست‌های شبکه‌اجتماعی، بدون مجموعه طلایی سازمان، همان عدد قابل ادعا نیست.

گزارش پایانی فنی (بند ۶-۳ پروپوزال): [`reports/final-report.md`](reports/final-report.md)
