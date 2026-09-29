# گزارش پایانی طرح پژوهشی CONTRA-Detect

تشخیص، تجزیه و تحلیل تناقضات عملکردی کاربران در شبکه اجتماعی ایکس، اینستاگرام و پیام‌رسان تلگرام

این سند خروجی مشترک بند ۶-۳ پروپوزال است: مستندات فنی، روش توسعه، ارزیابی مدل، و راهنمای راه‌اندازی با حداقل منابع سخت‌افزاری.

## ۱. مسئله و محدوده

تناقض گفتاری/عملکردی یعنی دو گفته از **یک کاربر** در زمان‌ها یا بسترهای مختلف که از نظر منطقی یا معنایی ناسازگارند. سامانه جفت‌پست همان کاربر را مقایسه می‌کند؛ طبقه‌بندی یک جملهٔ تکی انجام نمی‌شود.

- بسترها: ایکس (توییتر)، تلگرام، اینستاگرام
- ورودی: متن پست/پیام
- خارج از تعهد: تصویر و ویدئو

## ۲. معماری

```
Elasticsearch (twitter / telegram / instagram)
        │
        ▼
  schema واحد پست  →  پیش‌پردازش Hazm
        │
        ▼
  انتخاب جفت هم‌موضوع (TF-IDF)
        │
        ▼
  ParsBERT (NLI سه‌کلاسه)
        │
        ├── وجود/عدم تناقض
        ├── استخراج جفت پست متناقض
        └── نشان‌گر کمی در [0, 1]
```

کد اصلی:

| بخش | مسیر |
|---|---|
| پیش‌پردازش | `src/preprocess/text_processor.py` |
| داده و Elasticsearch | `src/data/` |
| آموزش و ارزیابی | `src/models/trainer.py`، `scripts/train.py`، `scripts/evaluate.py` |
| استخراج تناقض | `src/detection/engine.py` |
| API | `src/api/main.py` |

## ۳. داده

### ۳-۱. مجموعه برچسب‌دار آموزش و ارزیابی (FarsTail)

ادعای دقت فقط روی تست FarsTail اندازه‌گیری می‌شود. آمار پس از حذف جفت خالی و تکراری:

| split | تعداد | entailment | contradiction | neutral |
|---|---:|---:|---:|---:|
| train | 7266 | 2429 | 2389 | 2448 |
| val | 1537 | 515 | 499 | 523 |
| test | 1564 | 519 | 510 | 535 |

کلاس‌ها تقریباً متعادل‌اند (حدود ۳۳٪ هر برچسب). جزئیات در `reports/farstail_dataset_stats.json`.

آماده‌سازی:

```bash
python scripts/prepare_dataset.py
```

### ۳-۲. داده عملیاتی سازمان

پست‌های ایکس، تلگرام و اینستاگرام از Elasticsearch خوانده می‌شوند و **لیبل طلایی ندارند**. ایندکس پیش‌فرض:

- `ES_INDEX_TWITTER=twitter_temp_data`
- `ES_INDEX_TELEGRAM=telegram_temp_data,telegram_source,telegram_comment_data`
- `ES_INDEX_INSTAGRAM=instagram_temp_data,instagram_source,instagram_comment_data`

آدرس و اعتبار همان کلاینت پروژه echo chamber است (`https://elastic.synappse.ir`، `ELASTIC_AUTH=1`).

اگر نام ایندکس یا فیلدها در سازمان فرق دارد، در `.env` عوض شود. مقدار `none` یک بستر را خاموش می‌کند.

## ۴. پیش‌پردازش و آماده‌سازی

مطابق توصیهٔ همان پروپوزال برای فارسی، به‌جای NLTK/SpaCy از **Hazm** استفاده شده است:

- حذف لینک، ایمیل، منشن و نویز
- نرمال‌سازی و ریشه‌یابی
- حذف کلمات توقف
- استخراج ویژگی TF-IDF برای انتخاب جفت‌های هم‌موضوع همان کاربر

ذخیرهٔ میانی جفت‌ها با Pandas/CSV است (`scripts/build_pairs.py`). تعادل کلاس در FarsTail ذاتی است و نیاز به oversampling نداشت.

## ۵. آموزش مدل

- مدل پایه: ParsBERT (`HooshvareLab/bert-fa-base-uncased`) از Hugging Face
- ورودی جفت: `[CLS] text_a [SEP] text_b`
- برچسب‌ها: entailment / contradiction / neutral
- تقسیم FarsTail: splitهای رسمی train/val/test (معادل عملی ۷۰/۱۵/۱۵)
- بهینه‌سازی اختیاری هایپرپارامتر: `python scripts/train.py --tune` (Optuna سبک)
- ذخیره: `save_pretrained` در `models/saved/best_model` (PyTorch / Hugging Face، نه `.h5`)

```bash
python scripts/train.py
python scripts/evaluate.py
```

## ۶. ارزیابی

هدف پروپوزال: دقت حدود ۸۵٪ روی مجموعهٔ مورد استفاده برای ارزیابی (اینجا تست FarsTail).

متریک‌ها: Accuracy، Precision، Recall، F1 (macro) و F1 مخصوص کلاس contradiction.

نتایج آموزش کامل ParsBERT روی FarsTail (بهترین چک‌پوینت، `load_best_model_at_end`):

| معیار | اعتبارسنجی | تست | هدف |
|---|---:|---:|---:|
| Accuracy | ۰٫۸۲۱۱ | ۰٫۸۱۵۲ | ۰٫۸۵ |
| F1 (macro) | ۰٫۸۱۸۵ | ۰٫۸۱۲۷ | — |
| Precision (macro) | ۰٫۸۲۲۰ | ۰٫۸۱۵۷ | — |
| Recall (macro) | ۰٫۸۱۹۶ | ۰٫۸۱۴۱ | — |
| F1 کلاس contradiction | ۰٫۷۴۴۷ | ۰٫۷۳۱۴ | — |

فاصله تا هدف ۸۵٪ روی تست: **۳٫۴۸ واحد درصد**. ایپاک سوم نسبت به ایپاک دوم افت کرد؛ مدل ذخیره‌شده همان چک‌پوینت بهتر است. جزئیات در `reports/farstail_test_metrics.json`.

محدودیت صریح: این عدد روی پست‌های شبکه‌اجتماعی بدون مجموعه طلایی سازمان قابل ادعا نیست.

## ۷. سرویس و API

سه API الزامی پروپوزال روی لیست `user_ids` (با فیلتر اختیاری `platforms` و امکان ارسال `posts` در بدنه):

| روش | مسیر | خروجی |
|---|---|---|
| POST | `/api/v1/users/contradiction` | وجود/عدم تناقض + امتیاز |
| POST | `/api/v1/users/contradictory-posts` | جفت پست‌های متناقض |
| POST | `/api/v1/users/contradiction-score` | نشان‌گر کمی در `[0, 1]` |

سلامت: `GET /health`  
مستندات تعاملی: `/docs`

## ۸. راه‌اندازی و استقرار (حداقل منابع)

پیشنهاد حداقل: ۲ هسته CPU، ۸ گیگابایت RAM.

محلی:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

با Docker (پس از یک‌بار آموزش و وجود `models/saved/best_model`):

```bash
docker compose up --build -d
```

آموزش کامل FarsTail روی CPU چند ساعت و روی GPU/MPS معمولاً حدود یکی دو ساعت طول می‌کشد.

## ۹. آنچه عمداً پیاده نشد

این موارد در پروپوزال آمده‌اند ولی با داده و خروجی پذیرش طرح جور نیستند:

- اسکرپینگ وب و پایگاه SQL (سورس سازمان Elasticsearch است)
- TensorFlow و فایل `.h5`
- NLTK و SpaCy (Hazm جایگزین رسمی فارسی است)
- یادگیری تقویتی و مدل‌های ترکیبی چندمعماری

## ۱۰. انتقال به سازمان کاربر طرح

برای تحویل:

1. همین مخزن، فایل `.env.example`، و مدل ذخیره‌شده
2. اجرای `docker compose up` یا `uvicorn` روی سرور سازمان
3. تنظیم نام ایندکس تلگرام/اینستاگرام اگر با پیش‌فرض فرق دارد
4. مرور `/docs` و سه API روی چند `user_id` واقعی
5. این گزارش به‌همراه `reports/farstail_dataset_stats.json` و `reports/farstail_test_metrics.json`
