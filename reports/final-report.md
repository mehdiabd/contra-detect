# گزارش پایانی طرح پژوهشی CONTRA-Detect

به‌روزرسانی ۲۰۲۶-۱۰-۰۴: ترکیب منتخب سه‌مدلی به Accuracy برابر **۸۵٫۹۳٪** و F1 تناقض **۸۰٫۲۱٪** روی test FarsTail رسید. هدف ۸۵٪ در این ارزیابی حاصل شد. انتخاب اعضا و وزن‌ها با validation انجام شد؛ test قبلاً دیده شده است و تأیید مستقل نهایی به مجموعهٔ تازه نیاز دارد. جزئیات در [گزارش مرحلهٔ جدید](round2/report.md) است.

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
  schema واحد پست  →  پاک‌سازی و یکسان‌سازی محدود متن
        │
        ▼
  انتخاب جفت هم‌موضوع (TF-IDF)
        │
        ▼
  ترکیب سه مدل NLI با هم‌ردیف‌سازی کلاس‌ها
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

پیش‌پردازش واقعی دو مسیر دارد: برای انتخاب جفت، لینک، ایمیل و منشن حذف و برخی حروف و ارقام فارسی/عربی یکسان‌سازی می‌شوند؛ برای ورودی مدل NLI فقط فاصله‌های اضافی پاک می‌شوند. ریشه‌یابی، حذف کلمات توقف و فراخوانی Hazm در این مسیر اجرا نمی‌شوند. ویژگی TF-IDF برای انتخاب جفت‌های هم‌موضوع همان کاربر استفاده می‌شود. پذیرش این تفاوت روش با پروپوزال هنوز باید ثبت شود.

ذخیرهٔ میانی جفت‌ها با Pandas/CSV است (`scripts/build_pairs.py`). تعادل کلاس در FarsTail ذاتی است و نیاز به oversampling نداشت.

## ۵. آموزش مدل

- مدل پایهٔ مسیر عمومی: `persiannlp/parsbert-base-parsinlu-entailment`؛ نسخهٔ تحویل ترکیب سه عضو ذکرشده در بخش ارزیابی است
- ورودی جفت: `[CLS] text_a [SEP] text_b`
- برچسب‌ها: entailment / contradiction / neutral
- تقسیم FarsTail: splitهای رسمی train/val/test (معادل عملی ۷۰/۱۵/۱۵)
- مسیر اختیاری جست‌وجوی هایپرپارامتر با Optuna موجود است؛ مدرک اجرای آن برای نسخهٔ تحویل موجود نیست. جست‌وجوی وزن ترکیب اجرا شده و با این جست‌وجو متفاوت است.
- ذخیرهٔ آموزش عمومی: `save_pretrained` در مسیر آزمایش جدا (`models/experiments/general/model`). نسخهٔ عملیاتی در `models/saved/best_model` تنظیمات ترکیب دارد. قالب PyTorch / Hugging Face جایگزین قالب TensorFlow پیشنهادی شده؛ تأیید سازمان هنوز ثبت نشده است.

```bash
python scripts/train.py
python scripts/evaluate.py --model models/experiments/general/model --output models/experiments/general/evaluation.json
```

## ۶. ارزیابی

هدف پروپوزال: دقت حدود ۸۵٪ روی مجموعهٔ مورد استفاده برای ارزیابی (اینجا تست FarsTail).

متریک‌ها: Accuracy، Precision، Recall، F1 (macro) و F1 مخصوص کلاس contradiction.

نتیجهٔ فعلی ترکیب وزن‌دار targeted_v1 (۲۵٪)، parsnlu_farstail (۲۵٪) و mdeberta_nli (۵۰٪): Accuracy validation برابر ۸۶٫۰۸٪ و Accuracy test برابر ۸۵٫۹۳٪. F1 تناقض test برابر ۸۰٫۲۱٪ است. تنظیمات در `models/saved/best_model/inference_config.json` و خروجی ارزیابی در `reports/round2/evaluation.json` ثبت شده‌اند. پیش‌بینی‌گر این تنظیمات را به‌صورت پیش‌فرض می‌خواند.

جدول زیر نتیجهٔ تاریخی آموزش ParsBERT پیش از اصلاح و ترکیب مدل‌ها است:

| معیار | اعتبارسنجی | تست | هدف |
|---|---:|---:|---:|
| Accuracy | ۰٫۸۲۱۱ | ۰٫۸۱۵۲ | ۰٫۸۵ |
| F1 (macro) | ۰٫۸۱۸۵ | ۰٫۸۱۲۷ | — |
| Precision (macro) | ۰٫۸۲۲۰ | ۰٫۸۱۵۷ | — |
| Recall (macro) | ۰٫۸۱۹۶ | ۰٫۸۱۴۱ | — |
| F1 کلاس contradiction | ۰٫۷۴۴۷ | ۰٫۷۳۱۴ | — |

فاصلهٔ این اجرای تاریخی تا هدف ۸۵٪ روی تست: **۳٫۴۸ واحد درصد**؛ نتیجهٔ فعلی ترکیب مدل‌ها از هدف عبور کرده است. ایپاک سوم نسبت به ایپاک دوم افت کرد؛ مدل ذخیره‌شده همان چک‌پوینت بهتر است. جزئیات در `reports/farstail_test_metrics.json`.

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

با Docker (پس از انتقال بستهٔ نهایی شامل تنظیمات ترکیب و هر سه عضو در `models/saved`):

```bash
docker compose up --build -d
```

آموزش کامل FarsTail روی CPU چند ساعت و روی GPU/MPS معمولاً حدود یکی دو ساعت طول می‌کشد.

## ۹. آنچه عمداً پیاده نشد

وضعیت این موارد باید از تعهدهای پذیرش تفکیک شود؛ وجود این فهرست به معنی پذیرش سازمان نیست:

- اسکرپینگ وب و پایگاه SQL (سورس سازمان Elasticsearch است)
- TensorFlow و فایل `.h5`
- NLTK، SpaCy و Hazm در پیش‌پردازش فعلی فراخوانی نمی‌شوند؛ روش سادهٔ فعلی باید در جدول اختلاف‌ها تعیین تکلیف شود.
- یادگیری تقویتی

ترکیب وزن‌دار مدل‌های NLI با دو معماری BERT و DeBERTa در مرحلهٔ جدید پیاده و ارزیابی شده است.

## ۱۰. انتقال به سازمان کاربر طرح

برای تحویل:

1. همین مخزن، فایل `.env.example`، پوشهٔ best_model شامل inference_config و هر سه پوشهٔ عضو targeted_v1، parsnlu_farstail و mdeberta_nli
2. اجرای `docker compose up` یا `uvicorn` روی سرور سازمان
3. تنظیم نام ایندکس تلگرام/اینستاگرام اگر با پیش‌فرض فرق دارد
4. مرور `/docs` و سه API روی چند `user_id` واقعی
5. این گزارش به‌همراه آمار داده، `reports/round2/evaluation.json`، گزارش پذیرش نسخهٔ نهایی و مدارک تأیید سازمان
