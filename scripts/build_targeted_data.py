"""Author synthetic contrast sets independent of held-out texts; preserve the original train split."""
import hashlib
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.data.loader import load_farstail_split, save_pairs_csv
from src.data.schema import Pair


def build():
    rows=[]
    people=['آرمان','لیلا','نیما','مینا','سارا','کاوه','نادر','ترانه']
    for i,person in enumerate(people):
        other=people[(i+1)%len(people)]
        n=12+i; m=n+3
        # Each premise has one entailed, one contradicted and one unsupported hypothesis.
        # Exclusivity is explicit, avoiding closed-world assumptions about who can do what.
        scenarios=[
          ('entity_role',f'تنها نویسندهٔ گزارش آزمایش {person} است. {other} آن را ویرایش کرده است.',
           f'{person} گزارش آزمایش را نوشته است.',f'تنها نویسندهٔ گزارش آزمایش {other} است.',f'{person} در شیراز زندگی می‌کند.'),
          ('entity_role',f'در این مسابقه فقط {person} داور است و {other} بازیکن است.',
           f'{other} بازیکن این مسابقه است.',f'{other} داور این مسابقه است.',f'{person} پیش‌تر در مسابقهٔ دیگری داور بوده است.'),
          ('relation_direction',f'در جلسهٔ امروز {person} یک کتاب به {other} داد. {other} هیچ کتابی به {person} نداد.',
           f'{other} امروز از {person} کتاب گرفت.',f'{other} امروز به {person} کتاب داد.',f'کتاب جلد آبی دارد.'),
          ('relation_direction',f'در این نمودار، آب فقط از مخزن الف به مخزن ب جریان دارد و مسیر برگشت بسته است.',
           'آب از مخزن الف وارد مخزن ب می‌شود.','آب از مخزن ب وارد مخزن الف می‌شود.',f'گنجایش مخزن الف {n} لیتر است.'),
          ('number',f'در انبار {person} دقیقاً {n} جعبه وجود دارد.',
           f'تعداد جعبه‌های انبار {person} برابر {n} است.',f'در انبار {person} دقیقاً {m} جعبه وجود دارد.',f'جعبه‌های انبار {person} قرمز هستند.'),
          ('number',f'{person} در آزمون امروز دقیقاً {n} سؤال را درست پاسخ داد.',
           f'تعداد پاسخ‌های درست {person} در آزمون امروز {n} بود.',f'{person} در آزمون امروز {m} پاسخ درست داشت.',f'{person} برای آزمون امروز دو ساعت درس خواند.'),
          ('time',f'نشست امروز {person} فقط از ساعت {n} تا ساعت {n+1} برگزار شد.',
           f'نشست امروز {person} ساعت {n} آغاز شد.',f'نشست امروز {person} ساعت {n+2} آغاز شد.',f'{person} در نشست دربارهٔ آموزش صحبت کرد.'),
          ('time',f'فروشگاه {person} روز شنبه باز بود و روز یکشنبه تمام روز بسته بود.',
           f'فروشگاه {person} یکشنبه باز نبود.',f'فروشگاه {person} یکشنبه باز بود.',f'فروشگاه {person} دوشنبه باز بود.'),
          ('negation',f'{person} گفت من در سفر دیروز به رشت نرفتم، اما به قزوین رفتم.',
           f'{person} دیروز به قزوین سفر کرد.',f'{person} دیروز به رشت سفر کرد.',f'{person} دیروز با قطار سفر کرد.'),
          ('negation',f'در آزمایش امروز، {person} تأیید کرد که نمونه هیچ سربی ندارد.',
           'نمونهٔ آزمایش امروز فاقد سرب است.','نمونهٔ آزمایش امروز حاوی سرب است.','نمونهٔ آزمایش امروز حاوی مس است.'),
          ('quantifier',f'همهٔ اعضای گروه {person} در جلسهٔ امروز حاضر بودند.',
           f'هیچ عضو گروه {person} از جلسهٔ امروز غایب نبود.',f'بعضی از اعضای گروه {person} در جلسهٔ امروز غایب بودند.',f'همهٔ اعضای گروه {person} فردا در جلسه حاضر خواهند بود.'),
          ('quantifier',f'هیچ‌یک از کارکنان بخش {person} در آزمون امروز قبول نشدند.',
           f'همهٔ کارکنان بخش {person} در آزمون امروز مردود شدند.',f'حداقل یک کارمند بخش {person} در آزمون امروز قبول شد.',f'کارکنان بخش {person} در آزمون دیروز قبول شدند.'),
          ('condition',f'طبق مقررات شرکت {person}، ورود فقط با کارت مجاز است. {other} امروز کارت نداشت و اجازهٔ ورود نگرفت.',
           f'{other} امروز اجازهٔ ورود به شرکت {person} نداشت.',f'{other} امروز اجازهٔ ورود به شرکت {person} داشت.',f'{other} فردا کارت خواهد داشت.'),
          ('paraphrase_subset',f'{person} گزارش را نوشت، {other} آن را ترجمه کرد و نسخهٔ نهایی در کتابخانه قرار گرفت.',
           f'گزارش نوشتهٔ {person} در کتابخانه قرار گرفت.','نسخهٔ نهایی گزارش در کتابخانه قرار نگرفت.',f'گزارش نوشتهٔ {person} صد صفحه دارد.'),
          ('relation_mapping',f'در واژه‌نامهٔ {person}، آلفا به معنی روشن و بتا به معنی خاموش است و این دو معنی متفاوت‌اند.',
           'در این واژه‌نامه، معنی آلفا روشن است.','در این واژه‌نامه، معنی آلفا خاموش است.',f'{other} این واژه‌نامه را خریده است.'),
          ('unsupported_information',f'{person} امروز در شهر یزد یک سخنرانی ارائه کرد و هیچ سخنرانی دیگری نداشت.',
           f'{person} امروز در یزد سخنرانی کرد.',f'{person} امروز در شهر تبریز هم سخنرانی کرد.',f'موضوع سخنرانی امروز {person} هوش مصنوعی بود.'),
        ]
        for category,premise,entailed,contradicted,neutral in scenarios:
            for label,hypothesis in enumerate([entailed,contradicted,neutral]):
                rows.append((Pair(premise,hypothesis,label=label),category))
    return list({(p.text_a,p.text_b,p.label):(p,c) for p,c in rows}.values())


def main():
    out=ROOT/'data/labeled/targeted_v1';out.mkdir(parents=True,exist_ok=True)
    heldout=load_farstail_split('val')+load_farstail_split('test')
    heldout_texts={t for p in heldout for t in [p.text_a,p.text_b]}
    rows=build()
    assert all(p.text_a not in heldout_texts and p.text_b not in heldout_texts for p,c in rows)
    assert len({(p.text_a,p.text_b,p.label) for p,c in rows})==len(rows)
    save_pairs_csv([p for p,c in rows],out/'synthetic.csv')
    train=load_farstail_split('train')
    # Only exact held-out pairs are excluded; do not use held-out labels to select training examples.
    heldout_keys={(p.text_a,p.text_b) for p in heldout}
    clean_train=[p for p in train if (p.text_a,p.text_b) not in heldout_keys]
    save_pairs_csv(clean_train+[p for p,c in rows],out/'train_augmented.csv')
    manifest={'synthetic_n':len(rows),'original_train_n':len(train),'train_n':len(clean_train)+len(rows),
      'removed_exact_heldout_pairs':len(train)-len(clean_train),'per_class':{name:sum(p.label==i for p,c in rows) for i,name in enumerate(['entailment','contradiction','neutral'])},
      'categories':sorted({c for p,c in rows}),'source':'Independent fictional templates authored by Codex; not human-validated',
      'heldout_added_to_training':False,'synthetic_sha256':hashlib.sha256((out/'synthetic.csv').read_bytes()).hexdigest()}
    (ROOT/'reports/error_analysis/targeted_data_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print(manifest)

if __name__=='__main__':main()
