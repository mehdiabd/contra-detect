"""Export reproducible NLI error analysis without training or changing serving behavior."""
import argparse
import csv
import hashlib
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import config
import numpy as np
import torch
from src.data.loader import load_farstail_split
from src.models.predictor import ContradictionPredictor
from src.preprocess.text_processor import nli_text


def metrics(gold, pred):
    matrix = np.zeros((3, 3), dtype=int)
    np.add.at(matrix, (gold, pred), 1)
    classes = {}
    for i, name in config.NLI_ID2LABEL.items():
        tp = int(matrix[i, i])
        precision = tp / max(int(matrix[:, i].sum()), 1)
        recall = tp / max(int(matrix[i].sum()), 1)
        classes[name] = dict(precision=precision, recall=recall,
                            f1=2 * precision * recall / max(precision + recall, 1e-12),
                            support=int(matrix[i].sum()))
    return dict(accuracy=float(np.mean(gold == pred)), errors=int(np.sum(gold != pred)),
                confusion_matrix=matrix.tolist(), matrix_order=list(config.NLI_ID2LABEL.values()),
                per_class=classes)


def hints(a, b):
    # Surface cues only, not a diagnosis of the cause of an error.
    text = a + ' ' + b
    out = []
    if re.search(r'[0-9۰-۹٠-٩]', text): out.append('number')
    if re.search(r'\b(نیست|نبود|نه|هرگز|هیچ|نمی\S*|نخواهد|ندارد)\b', text): out.append('negation')
    if re.search(r'\b(همه|هر|بعضی|برخی|فقط|تنها|اگر|مگر)\b', text): out.append('quantifier_or_condition')
    return '|'.join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, default=config.BEST_MODEL_DIR)
    parser.add_argument('--splits', nargs='+', choices=['val', 'test'], default=['val', 'test'])
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/error_analysis')
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    torch.set_num_threads(4)
    args.output.mkdir(parents=True, exist_ok=True)
    predictor = ContradictionPredictor(args.model)
    if predictor.id2label != config.NLI_ID2LABEL:
        raise ValueError('Model label order differs from dataset; explicit remapping required.')
    summary = {'model': str(args.model.resolve()), 'max_length': config.MODEL_CONFIG['max_length'],
               'model_sha256': hashlib.sha256((args.model / 'model.safetensors').read_bytes()).hexdigest(),
               'torch_version': torch.__version__, 'device': str(predictor.device),
               'note': 'Diagnostic only. Tune on validation; never train on test errors. Cue tags are not causal labels.',
               'splits': {}}
    for split in args.splits:
        pairs = load_farstail_split(split)
        rows, forwards, services = [], [], []
        for start in range(0, len(pairs), args.batch_size):
            batch = pairs[start:start + args.batch_size]
            a = [nli_text(p.text_a) for p in batch]
            b = [nli_text(p.text_b) for p in batch]
            f = predictor._scores(a, b).numpy()
            r = predictor._scores(b, a).numpy()
            chosen = np.where((r[:, 1] >= f[:, 1])[:, None], r, f)
            lengths = predictor.tokenizer(a, b, truncation=False, padding=False)['input_ids']
            for offset, (p, pf, pr, ps, tokens) in enumerate(zip(batch, f, r, chosen, lengths)):
                fi, ri, si = int(pf.argmax()), int(pr.argmax()), int(ps.argmax())
                forwards.append(fi); services.append(si)
                row = dict(sample_id=hashlib.sha256((p.text_a + '\0' + p.text_b).encode()).hexdigest()[:16],
                           split=split, row_index=start + offset, text_a=p.text_a, text_b=p.text_b,
                           gold=config.NLI_ID2LABEL[p.label], forward=config.NLI_ID2LABEL[fi],
                           reverse=config.NLI_ID2LABEL[ri], service=config.NLI_ID2LABEL[si],
                           forward_error=fi != p.label, service_error=si != p.label,
                           direction_disagreement=fi != ri, service_changed=fi != si,
                           forward_confidence=float(pf.max()), service_confidence=float(ps.max()),
                           forward_margin=float(np.sort(pf)[-1] - np.sort(pf)[-2]),
                           token_length=len(tokens), truncated=len(tokens) > config.MODEL_CONFIG['max_length'],
                           cue_tags=hints(p.text_a, p.text_b), human_category='', human_notes='')
                for i, name in config.NLI_ID2LABEL.items():
                    row['forward_p_' + name] = float(pf[i])
                    row['reverse_p_' + name] = float(pr[i])
                rows.append(row)
            print(f'{split}: {min(start + args.batch_size, len(pairs))}/{len(pairs)}', flush=True)
        gold = np.array([p.label for p in pairs]); fids = np.array(forwards); sids = np.array(services)
        def export(name, selected):
            with (args.output / f'{split}_{name}.csv').open('w', encoding='utf-8-sig', newline='') as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader(); writer.writerows(selected)
        export('predictions', rows)
        errors = [row for row in rows if row['forward_error'] or row['service_error']]
        export('errors', sorted(errors, key=lambda x: x['forward_confidence'], reverse=True))
        # Include confident and uncertain errors, deterministically, without calling them manually reviewed.
        ordered = sorted(errors, key=lambda x: x['forward_confidence'])
        review = ordered[:50] + ordered[-50:]
        review = list({x['sample_id']: x for x in review}.values())
        export('review_100', review)
        summary['splits'][split] = dict(n=len(pairs), forward=metrics(gold, fids), service=metrics(gold, sids),
            dataset_sha256=hashlib.sha256(json.dumps([(p.text_a, p.text_b, p.label) for p in pairs], ensure_ascii=False).encode()).hexdigest(),
            service_fixed=int(np.sum((fids != gold) & (sids == gold))),
            service_broke=int(np.sum((fids == gold) & (sids != gold))),
            direction_disagreements=sum(x['direction_disagreement'] for x in rows),
            truncated=sum(x['truncated'] for x in rows),
            forward_confident_errors=sum(x['forward_error'] and x['forward_confidence'] >= .9 for x in rows),
            forward_error_cues={tag: sum(x['forward_error'] and tag in x['cue_tags'].split('|') for x in rows)
                                for tag in ['number', 'negation', 'quantifier_or_condition']},
            correct_needed_for_85=max(0, math.ceil(.85 * len(pairs)) - int(np.sum(fids == gold))))
        (args.output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
