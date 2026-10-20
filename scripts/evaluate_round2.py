"""Verify frozen bundle on validation, then score test once and count net corrections."""
import csv,hashlib,json,math,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import config,numpy as np,torch
from src.models.predictor import ContradictionPredictor
from src.data.loader import load_farstail_split
from scripts.analyze_errors import metrics


def main():
 torch.set_num_threads(4);out=ROOT/'reports/round2'
 selected=json.loads((out/'selection.json').read_text())
 if selected['status']!='frozen_before_test':raise RuntimeError('Selection was not frozen')
 model=ContradictionPredictor(selected['bundle_path'])
 val=load_farstail_split('val');expected=np.load(out/'selected_val.npz')['probs']
 actual=[]
 for start in range(0,len(val),16):
  batch=val[start:start+16]
  actual.append(model._scores([p.text_a for p in batch],[p.text_b for p in batch]).numpy())
 actual=np.concatenate(actual)
 if not np.allclose(actual,expected,atol=2e-6):raise RuntimeError('Bundle differs from cached validation scoring')
 if not np.array_equal(actual.argmax(1),expected.argmax(1)):raise RuntimeError('Validation labels changed')
 print('Full validation bundle parity passed',flush=True)
 pairs=load_farstail_split('test');gold=np.array([p.label for p in pairs]);chunks=[];started=time.monotonic()
 for start in range(0,len(pairs),16):
  batch=pairs[start:start+16]
  chunks.append(model._scores([p.text_a for p in batch],[p.text_b for p in batch]).numpy())
  if start%160==0:print(json.dumps({'split':'test','n':min(start+16,len(pairs)),'total':len(pairs),'elapsed_seconds':round(time.monotonic()-started)}),flush=True)
 probs=np.concatenate(chunks);pred=probs.argmax(1)
 baseline=np.load(ROOT/'reports/error_analysis/targeted_test_predictions.npz')
 if not np.array_equal(gold,baseline['gold']):raise RuntimeError('Test order changed')
 before=baseline['probs'].argmax(1);fixed=(before!=gold)&(pred==gold);broke=(before==gold)&(pred!=gold)
 np.savez(out/'selected_test.npz',probs=probs,gold=gold)
 with (out/'test_changes.csv').open('w',encoding='utf-8-sig',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=['row_index','sample_id','text_a','text_b','gold','before','after','change']);writer.writeheader()
  for i,p in enumerate(pairs):
   if before[i]==pred[i]:continue
   writer.writerow(dict(row_index=i,sample_id=hashlib.sha256((p.text_a+'\0'+p.text_b).encode()).hexdigest()[:16],text_a=p.text_a,text_b=p.text_b,gold=config.NLI_ID2LABEL[p.label],before=config.NLI_ID2LABEL[int(before[i])],after=config.NLI_ID2LABEL[int(pred[i])],change='fixed' if fixed[i] else ('broken' if broke[i] else 'still_wrong')))
 score=metrics(gold,pred)
 payload={'selection_split':'validation','test_used_for_selection':False,'manifest':selected['manifest'],
  'baseline_test':metrics(gold,before),'candidate_test':score,'fixed':int(fixed.sum()),'broken':int(broke.sum()),
  'net_corrections':int(fixed.sum()-broke.sum()),'meets_45_net':int(fixed.sum()-broke.sum())>=45,
  'meets_85_test':score['accuracy']>=.85,'correct_needed_for_85':max(0,math.ceil(.85*len(gold))-int(np.sum(pred==gold))),
  'full_validation_parity':True,'test_runtime_seconds':time.monotonic()-started,'default_adopted':False,
  'limitation':'This test has previously been inspected; a fresh independent set is required for unbiased external confirmation.'}
 (out/'evaluation.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2));print(json.dumps(payload),flush=True)
if __name__=='__main__':main()
