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

## استقرار مستقل با Docker

ایمیج شامل کد، نمونه‌داده و بستهٔ inference نهایی است؛ انتقال `models/` جداگانه لازم نیست. حجم build context و ایمیج چند گیگابایت خواهد بود. برای build به دسترسی registry پکیج‌های Python نیاز دارید؛ برای اجرای inference، مدل‌ها داخل ایمیج‌اند و دانلود مدل از Hugging Face لازم نیست.

وزن‌ها و توکن‌سازهای نسخهٔ نهایی با **Git LFS در مخزن خصوصی GitLab** نگهداری می‌شوند. روی ماشین build ابتدا Git LFS را نصب و فایل‌ها را دریافت کنید؛ runner نیز باید فایل‌های LFS را دریافت کند و `GIT_LFS_SKIP_SMUDGE` فعال نباشد:

```bash
git lfs install
git lfs pull
python scripts/prepare_model_release.py
```

دستور آخر، هش هر ۱۵ فایل لازم را با `models/release.json` مقایسه می‌کند. فایل‌های pointer به‌جای وزن واقعی، فایل ناقص یا مدل متفاوت باعث شکست کنترل می‌شوند. گزارش همین انتشار از ارزیابی مجدد همهٔ ۱۵۶۴ جفت آزمون ساخته شده است: ۱۳۴۴ پاسخ درست و Accuracy برابر **۸۵٫۹۳٪**. بازسازی این گزارش، با دادهٔ FarsTail و محیط کامل ارزیابی، با `python scripts/prepare_model_release.py --evaluate` انجام می‌شود؛ این دستور وزن‌ها را تغییر نمی‌دهد.

```bash
docker compose build
docker compose up -d
```

پیش‌نیاز: Docker Engine و Docker Compose plugin. مقدار `ES_ENABLED` به‌صورت پیش‌فرض خاموش است، بنابراین API با `posts` ارسالی در درخواست کار می‌کند و به secret نیاز ندارد. برای اتصال به Elasticsearch، `.env` مقصد را جداگانه بسازید و تنظیمات ES را در آن وارد کنید؛ فایل `.env` را به Git اضافه نکنید. در صورت نیاز پورت میزبان را با `CONTRA_DETECT_PORT` تغییر دهید.

سلامت سرویس: `GET http://localhost:8000/health`  
مستندات: `http://localhost:8000/docs`

برای ساخت روی یک ماشین و اجرای آفلاین روی دیگری، build کنید و ایمیج را با `docker save`/`docker load` منتقل کنید یا به registry خصوصی push/pull کنید. هر دو روش خود مدل‌ها را داخل ایمیج نگه می‌دارند.

مدل ارزیابی‌شده داخل `/opt/contra-detect/models` قرار می‌گیرد. PVC قدیمی `/app/models/saved` صرفاً می‌تواند cache را نگه دارد و مدل داخل ایمیج را نمی‌پوشاند. در Docker و Kubernetes گزینهٔ `REQUIRE_MODEL_RELEASE=true` فعال است؛ نبود بسته یا اختلاف هش، شروع سرویس را متوقف می‌کند و مدل پایه جای آن اجرا نمی‌شود. منابع تنظیم‌شده برای سه مدل ۳ GiB حافظهٔ درخواستی و سقف ۸ GiB هستند؛ این تنظیم، ادعای اندازه‌گیری حداقل منابع نیست.

پس از مرج و استقرار، `GET /health` باید مسیر مدل در `/opt/contra-detect/models`، سه عضو با وزن‌های `0.25 / 0.25 / 0.50` و `model.release.verified=true` با شناسهٔ `farstail-ensemble-8593-v1` را گزارش کند. `model.release.evaluation` دقت و دامنهٔ ارزیابی همین فایل‌ها را نشان می‌دهد؛ عدد ۸۵٫۹۳٪ مربوط به FarsTail است، نه سنجش زندهٔ صحت تمام پست‌های شبکه‌های اجتماعی. نسخهٔ API این انتشار `1.1.1` است. مدل‌ها تا قبل از این کنترل بارگذاری نشوند، `/health` پاسخ ۵۰۳ می‌دهد.

منشأ و مجوز وزن‌های بیرونی مستقل از مجوز کد مخزن است: مدل پایهٔ [ParsNLI/ParsBERT](https://huggingface.co/persiannlp/parsbert-base-parsinlu-entailment) با برچسب `CC-BY-NC-SA-4.0` و مدل [mDeBERTa](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7) با برچسب MIT منتشر شده‌اند. شناسهٔ دقیق mDeBERTa در `scripts/download_nli_candidate.py` ثبت است؛ محدودیت سابقهٔ آموزش اعضای قدیمی و دیده‌شدن قبلی آزمون در گزارش انتشار محفوظ است.

## API

هر سه اندپوینت `platform_name`، لیست `user_ids` و بازهٔ اختیاری `date_range` می‌گیرند. لیست می‌تواند یک یا چند شناسه داشته باشد. اگر خالی باشد، کاربران فعال همان پلتفرم بررسی و نتایج بر اساس بیشترین امتیاز تناقض مرتب می‌شوند. اگر `posts` در بدنه باشد همان استفاده می‌شود؛ وگرنه در صورت `ES_ENABLED=true` از Elasticsearch و در غیر این صورت از `data/sample/posts.json` خوانده می‌شود.

| روش | مسیر | خروجی |
|---|---|---|
| POST | `/api/v1/users/contradiction` | وجود/عدم تناقض + امتیاز |
| POST | `/api/v1/users/contradictory-posts` | جفت پست‌های متناقض |
| POST | `/api/v1/users/contradiction-score` | نشان‌گر کمی در `[0, 1]` |

هر سه خروجی برای هر کاربر `analysis_status`، `post_count` و `pair_count` دارند:

- `completed`: حداقل یک جفت قابل بررسی تحلیل شده است. `has_contradiction=false` در این وضعیت یعنی در جفت‌های بررسی‌شده تناقضی مطابق آستانه پیدا نشده است.
- `insufficient_data`: پس از فیلترها و آماده‌سازی، جفت قابل بررسی وجود ندارد. در خروجی‌های مربوط، `has_contradiction` و `score` برابر `null` هستند؛ این وضعیت نتیجهٔ منفی محسوب نمی‌شود. در خروجی استخراج، فهرست پست‌ها خالی و وضعیت همچنان `insufficient_data` است.

نمونهٔ خروجی تشخیص برای کاربری که دادهٔ کافی ندارد:

```json
{
  "results": [{
    "user_id": "u_ali",
    "analysis_status": "insufficient_data",
    "post_count": 1,
    "pair_count": 0,
    "has_contradiction": null,
    "score": null
  }]
}
```

کلاینت باید مقادیر `null` را از تصمیم `false` و امتیاز صفر تفکیک کند. اگر یک منبع مورد نیاز خطا بدهد، timeout شود یا نتیجهٔ ناقص برگرداند، کل درخواست با HTTP `503` و `detail.code=data_source_unavailable` رد می‌شود؛ حتی اگر بخشی از داده از منابع دیگر دریافت شده باشد. پاسخ خطا نتیجهٔ تحلیل ندارد. دریافت موفق بدون پست، خطای منبع نیست و برای کاربر درخواستی `insufficient_data` برمی‌گرداند. در صورت ارسال `posts`، تحلیل به دریافت Elasticsearch وابسته نیست.

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
