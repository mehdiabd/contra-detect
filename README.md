# CONTRA-Detect

تشخیص تناقض گفتاری/عملکردی یک کاربر در پست‌های متنی ایکس، تلگرام و اینستاگرام. تناقض یعنی دو گفته از **یک کاربر** که از نظر معنایی با هم ناسازگارند. تصویر و ویدئو خارج از محدوده است.

راهنمای پیگیری تحویل: [چک‌لیست](reports/delivery_checklist.md). محیط آزموده‌شده و دستورهای بازتولید: [راهنما](reports/reproducibility/README.md). آموزش عمومی در مسیر آزمایش جدا ذخیره می‌شود و مجاز به بازنویسی مدل عملیاتی نیست.

## اجرا

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
cp .env.example .env
```

دانلود و تأیید دیتاست برچسب‌دار [FarsTail](https://github.com/dml-qom/FarsTail) (آموزش و ادعای دقت فقط روی همین مجموعه است؛ پست‌های Elasticsearch لیبل طلایی ندارند):

```bash
python scripts/prepare_dataset.py
python scripts/train.py
python scripts/evaluate.py --model models/experiments/general/model --output models/experiments/general/evaluation.json
```

آمار FarsTail بعد از پاک‌سازی جفت خالی/تکراری:

| split | n | entailment | contradiction | neutral |
|---|---:|---:|---:|---:|
| train | 7266 | 2429 | 2389 | 2448 |
| val | 1537 | 515 | 499 | 523 |
| test | 1564 | 519 | 510 | 535 |

آموزش کامل FarsTail روی CPU چند ساعت طول می‌کشد. برای اطمینان از درست بودن حلقه آموزش:

```bash
python scripts/train.py --max-samples 8 --epochs 1 --batch-size 2 --cpu --no-final-train-val
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
# بستهٔ نهایی مدل‌ها را طبق راهنمای تحویل در models/saved قرار دهید
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

## هدف دقت ۸۵٪ و ارزیابی پیشین

هدف پروپوزال روی **تست FarsTail** اندازه‌گیری شد. نتیجه fine-tune ParsBERT:

| مجموعه | Accuracy | F1 | Precision | Recall |
|---|---:|---:|---:|---:|
| val | 0.8211 | 0.8185 | 0.8220 | 0.8196 |
| test | 0.8152 | 0.8127 | 0.8157 | 0.8141 |

هدف ۸۵٪ روی تست حاصل نشد (فاصله ۳٫۴۸ واحد). جزئیات در `reports/farstail_test_metrics.json`. روی پست‌های شبکه‌اجتماعی، بدون مجموعه طلایی سازمان، همان عدد قابل ادعا نیست.

گزارش پایانی فنی (بند ۶-۳ پروپوزال): [`reports/final-report.md`](reports/final-report.md)

## تحلیل خطا و آموزش هدفمند

گزارش مرحلهٔ اول در `reports/error_analysis/report.md` و خروجی آزمایش‌های بعدی در همان پوشه است. پیش‌بینی سه‌کلاسه و `contradiction_score` با ترتیب اصلی متن‌ها محاسبه می‌شوند؛ `bidirectional_contradiction_score` بیشترین امتیاز دو جهت را فقط برای بررسی نگه می‌دارد. آستانهٔ تصمیم API تغییر نکرده است.

```bash
HF_HUB_OFFLINE=1 .venv/bin/python scripts/analyze_errors.py --model models/saved/baseline_before_targeted_v1
.venv/bin/python scripts/build_targeted_data.py
HF_HUB_OFFLINE=1 .venv/bin/python scripts/train_targeted.py
```

آموزش هدفمند از مدل فعلی شروع می‌شود، فقط train و مثال‌های ساختگی مستقل را مصرف می‌کند و بهترین epoch را با validation انتخاب می‌کند. مدل نامزد در `models/saved/targeted_v1` ذخیره می‌شود. فایل‌های بازبینی validation برای آموزش مصرف نمی‌شوند. سابقهٔ کامل آموزش مدل اولیه معلوم نیست؛ این آزمایش به‌تنهایی اثبات ارزیابی کاملاً مستقل نیست.

فایل‌های تحلیل اولیه، رفتار قدیمی سرویس را ثبت کرده‌اند؛ ستون `service` در آن فایل‌ها یک snapshot پیش از اصلاح است. آزمایش اصلاح جهت در `direction_experiment.json` ثبت شده است. بازبینی ۱۰۰ نمونه توسط Codex انجام شده، نه داوری انسانی مستقل.

نتیجهٔ مرحلهٔ اول آموزش هدفمند ۲۰۲۶-۱۰-۰۴: مدل منتخب روی test به Accuracy برابر **۸۲٫۱۶٪** و F1 تناقض **۷۵٫۱۸٪** رسید و در آن مرحله در `models/saved/best_model` قرار گرفت؛ اکنون آن مسیر تنظیمات ترکیب سه‌مدلی دارد. مبنای بازتولیدشده قبل از این اجرا ۸۰٫۴۳٪ بود؛ این نتیجه تاریخی است؛ نتیجهٔ فعلی ۸۵٫۹۳٪ ترکیب سه‌مدلی در بخش بعد آمده است. مدل اولیه در `models/saved/baseline_before_targeted_v1` محفوظ است. گزارش کامل در [`reports/error_analysis/priority_followup.md`](reports/error_analysis/priority_followup.md) قرار دارد.

## نتیجهٔ فعلی: عبور از ۸۵٪

ترکیب منتخب سه‌مدلی روی test FarsTail به **Accuracy = ۸۵٫۹۳٪** و **F1 تناقض = ۸۰٫۲۱٪** رسید. نسبت به نسخهٔ تکی ۸۲٫۱۶٪، خطاها از ۲۷۹ به ۲۲۰ کاهش یافتند: ۱۲۳ اصلاح و ۶۴ خطای تازه، برابر **۵۹ اصلاح خالص**. مدل‌ها و وزن ترکیب فقط با validation انتخاب شدند؛ test در مراحل قبلی دیده شده، بنابراین تأیید مستقل نهایی به مجموعهٔ تازه نیاز دارد.

پیش‌بینی‌گر پیش‌فرض اکنون `models/saved/best_model/inference_config.json` را می‌خواند: ۲۵٪ targeted_v1، ۲۵٪ parsnlu_farstail و ۵۰٪ mdeberta_nli. ترتیب کلاس‌ها پیش از ترکیب هم‌ردیف می‌شود. هر سه پوشهٔ عضو را برای انتقال و Docker کنار best_model نگه دارید. سه مدل حافظه و زمان بیشتری از مدل تکی نیاز دارند. سرویس‌های در حال اجرا باید دوباره مدل را بارگیری کنند.

[گزارش نتایج و ریز اصلاح‌ها](reports/round2/report.md). تنظیمات نسخهٔ منتخب نیز در `models/saved/round2_ensemble` و مدل تکی قبلی در `models/saved/baseline_before_round2` محفوظ‌اند.
