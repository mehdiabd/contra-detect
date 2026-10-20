"""Compare old and corrected decisions using cached validation probabilities only."""
import csv,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import config
import numpy as np
from scripts.analyze_errors import metrics
rows=list(csv.DictReader((ROOT/'reports/error_analysis/val_predictions.csv').open(encoding='utf-8-sig')))
names=config.NLI_ID2LABEL;ids={v:k for k,v in names.items()}
gold=np.array([ids[r['gold']] for r in rows]);forward=np.array([ids[r['forward']] for r in rows]);old=np.array([ids[r['service']] for r in rows])
f=np.array([float(r['forward_p_contradiction']) for r in rows]);rev=np.array([float(r['reverse_p_contradiction']) for r in rows])
def binary(pred):
 true=gold==1;tp=int(np.sum(true&pred));fp=int(np.sum(~true&pred));fn=int(np.sum(true&~pred))
 return {'tp':tp,'fp':fp,'fn':fn,'precision':tp/max(tp+fp,1),'recall':tp/max(tp+fn,1),'f1':2*tp/max(2*tp+fp+fn,1)}
threshold=config.CONTRADICTION_THRESHOLDS['pair']
payload={'split':'validation only','threshold_unchanged':threshold,'n':len(rows),
 'legacy_three_class':metrics(gold,old),'corrected_three_class':metrics(gold,forward),
 'legacy_pair_decision':binary((old==1)&(np.maximum(f,rev)>=threshold)),
 'corrected_pair_decision':binary((forward==1)&(f>=threshold)),
 'scope':'Pair-level decisions on NLI; not end-to-end user API accuracy',
 'bidirectional_score':'Retained only as diagnostic; production score and label use forward probabilities'}
(ROOT/'reports/error_analysis/direction_experiment.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2))
print(json.dumps(payload,ensure_ascii=False,indent=2))
