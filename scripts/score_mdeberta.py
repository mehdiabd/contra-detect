"""Score the independently pretrained multilingual architecture on validation only."""
import hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import config,numpy as np,torch
from src.models.predictor import ContradictionPredictor
from src.data.loader import load_farstail_split
from scripts.analyze_errors import metrics


def main():
 torch.set_num_threads(4)
 source=ROOT/'models/saved/mdeberta_nli';out=ROOT/'reports/round2'
 model=ContradictionPredictor(source);model.model.float();model.max_length=512
 pairs=load_farstail_split('val');gold=np.array([p.label for p in pairs]);probs=[];started=time.monotonic()
 lengths=model.tokenizer([p.text_a for p in pairs],[p.text_b for p in pairs],truncation=False)['input_ids']
 for start in range(0,len(pairs),16):
  batch=pairs[start:start+16]
  p=model._scores([p.text_a for p in batch],[p.text_b for p in batch]).numpy()
  if not np.all(np.isfinite(p)):raise RuntimeError('Non-finite model probabilities')
  probs.append(p)
  if start%160==0:print(json.dumps({'n':min(start+16,len(pairs)),'total':len(pairs),'elapsed_seconds':round(time.monotonic()-started)}),flush=True)
 p=np.concatenate(probs);np.savez(out/'mdeberta_val.npz',probs=p,gold=gold)
 h=hashlib.sha256()
 with (source/'model.safetensors').open('rb') as handle:
  for block in iter(lambda:handle.read(1024*1024),b''):h.update(block)
 payload={'source':'MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7','split':'val','max_length':512,
  'model_sha256':h.hexdigest(),'canonical_order':model.member_label_orders[0],
  'revision':(source/'.cache/huggingface/download/config.json.metadata').read_text().splitlines()[0],
  'dtype':str(next(model.model.parameters()).dtype),'metrics':metrics(gold,p.argmax(1)),
  'longer_than_256':sum(len(x)>256 for x in lengths),'truncated_at_512':sum(len(x)>512 for x in lengths),
  'runtime_seconds':time.monotonic()-started}
 (out/'mdeberta_val.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2));print(json.dumps(payload),flush=True)
if __name__=='__main__':main()
