"""Cache forward-only validation probabilities for existing model candidates."""
import gc,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import config
import numpy as np
import torch
from src.data.loader import load_farstail_split
from src.models.predictor import ContradictionPredictor
from scripts.analyze_errors import metrics

sources=[('parsnlu','models/saved/parsnlu_farstail'),('fallback',config.FALLBACK_MODEL)]
def main():
 torch.set_num_threads(4);pairs=load_farstail_split('val');gold=np.array([p.label for p in pairs])
 out=ROOT/'reports/round2';out.mkdir(exist_ok=True)
 for name,source in sources:
  predictor=ContradictionPredictor(source);probs=[];start_time=time.monotonic()
  for start in range(0,len(pairs),16):
   chunk=pairs[start:start+16]
   probs.append(predictor._scores([p.text_a for p in chunk],[p.text_b for p in chunk]).numpy())
   if start%320==0:print(name,start,len(pairs),flush=True)
  probs=np.concatenate(probs);np.savez(out/f'{name}_val.npz',probs=probs,gold=gold)
  payload={'source':source,'split':'val','id2label':predictor.id2label,'metrics':metrics(gold,probs.argmax(1)),'runtime_seconds':time.monotonic()-start_time}
  (out/f'{name}_val.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2));print(json.dumps(payload),flush=True)
  del predictor;gc.collect()
 z=np.load(ROOT/'models/saved/targeted_v1/validation_predictions.npz');logits=z['logits'];p=np.exp(logits-logits.max(1,keepdims=True));p/=p.sum(1,keepdims=True)
 np.savez(out/'current_val.npz',probs=p,gold=z['gold'])
if __name__=='__main__':main()
