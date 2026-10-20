"""Validation-only input normalization and relevant-sentence experiments."""
import re,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import config,numpy as np,torch
from src.models.predictor import ContradictionPredictor
from src.data.loader import load_farstail_split
from scripts.analyze_errors import metrics


def normalize(text):
 return ' '.join(text.translate(str.maketrans('يکك٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹','یکک01234567890123456789')).replace('\u200c',' ').split())


def relevant(premise,hypothesis):
 sentences=[s.strip() for s in re.split(r'(?<=[.!؟؛])\s+',premise) if s.strip()]
 if len(sentences)<3:return premise
 terms=set(normalize(hypothesis).split())
 scored=sorted(range(len(sentences)),key=lambda i:len(set(normalize(sentences[i]).split())&terms)/max(len(terms),1),reverse=True)
 return ' '.join(sentences[i] for i in sorted(scored[:2]))


def main():
 torch.set_num_threads(4);model=ContradictionPredictor(ROOT/'models/saved/targeted_v1');pairs=load_farstail_split('val');gold=np.array([p.label for p in pairs])
 out=ROOT/'reports/round2'
 for name,func in [('normalized',lambda a,b:(normalize(a),normalize(b))),('relevant',lambda a,b:(relevant(a,b),b))]:
  probs=[]
  for start in range(0,len(pairs),16):
   texts=[func(p.text_a,p.text_b) for p in pairs[start:start+16]]
   probs.append(model._scores([a for a,b in texts],[b for a,b in texts]).numpy())
   if start%320==0:print(name,start,len(pairs),flush=True)
  p=np.concatenate(probs);np.savez(out/f'{name}_val.npz',probs=p,gold=gold)
  score=metrics(gold,p.argmax(1));(out/f'{name}_val.json').write_text(json.dumps(score,indent=2));print(name,score['accuracy'],flush=True)
if __name__=='__main__':main()
