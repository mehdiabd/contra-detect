# CONTRA-Detect

تشخیص تناقض گفتاری/عملکردی یک کاربر در پست‌های متنی ایکس، تلگرام و اینستاگرام. تناقض یعنی دو گفته از **یک کاربر** که از نظر معنایی با هم ناسازگارند. تصویر و ویدئو خارج از محدوده است.

## اجرا

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

آموزش مدل جفتی ParsBERT روی [FarsTail](https://github.com/dml-qom/FarsTail):

```bash
python scripts/train.py
python scripts/evaluate.py
```

آموزش کامل FarsTail روی CPU چند ساعت طول می‌کشد. برای اطمینان از درست بودن حلقه آموزش:

```bash
python scripts/train.py --max-samples 32 --epochs 1
```

سرویس:

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

مستندات تعاملی: `http://localhost:8000/docs`

## API

هر سه اندپوینت لیست `user_ids` می‌گیرند. اگر `posts` در بدنه باشد همان استفاده می‌شود؛ وگرنه در صورت `ES_ENABLED=true` از Elasticsearch و در غیر این صورت از `data/sample/posts.json` خوانده می‌شود.

| روش | مسیر | خروجی |
|---|---|---|
| POST | `/api/v1/users/contradiction` | وجود/عدم تناقض + امتیاز |
| POST | `/api/v1/users/contradictory-posts` | جفت پست‌های متناقض |
| POST | `/api/v1/users/contradiction-score` | نشان‌گر کمی در `[0, 1]` |

نمونه:

```bash
curl -s http://localhost:8000/api/v1/users/contradiction \
  -H 'Content-Type: application/json' \
  -d '{"user_ids":["u_ali","u_reza"]}'
```

## داده

- سورس عملیاتی: ایندکس Elasticsearch سازمان (mapping فیلدها در `.env`)
- آموزش اولیه: FarsTail (NLI فارسی: entailment / contradiction / neutral)
- جفت‌سازی پست‌های واقعی: `python scripts/build_pairs.py`
- برچسب ضعیف با مدل آموزش‌دیده: `python scripts/build_pairs.py --weak-label`
- داده سفارشی برچسب‌خورده: `python scripts/train.py --source csv --pairs data/labeled/pairs.csv`

مدل با `save_pretrained` ذخیره می‌شود (PyTorch / Hugging Face)، نه `.h5`.

## دقت ۸۵٪

روی **تست FarsTail** این هدف با fine-tune ParsBERT قابل پیگیری است. روی پست‌های شبکه‌اجتماعی، بدون مجموعه طلایی سازمان، همان عدد قابل ادعا نیست. `scripts/evaluate.py` برای ارزیابی نهایی روی داده مورد تأیید سازمان آماده است.
