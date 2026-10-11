"""Search a fixed, coarse simplex of model weights using validation probabilities only."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.analyze_errors import metrics


def main():
 root=ROOT/'reports/round2';names=['current','parsnlu','fallback','mdeberta']
 data=[np.load(root/f'{n}_val.npz') for n in names];gold=data[0]['gold']
 if not all(np.array_equal(gold,d['gold']) for d in data):raise ValueError('Validation rows differ')
 ps=[d['probs'] for d in data];trials=[]
 for a in range(5):
  for b in range(5-a):
   for c in range(5-a-b):
    weights=np.array([a,b,c,4-a-b-c])/4
    if weights[-1]==0:continue
    p=sum(w*q for w,q in zip(weights,ps));score=metrics(gold,p.argmax(1))
    trials.append({'weights':weights.tolist(),'metrics':score})
 trials.sort(key=lambda x:x['metrics']['accuracy'],reverse=True)
 payload={'names':names,'selection_split':'val','weight_step':.25,'trials':trials}
 (root/'architecture_ensemble_trials.json').write_text(json.dumps(payload,indent=2))
 print('Validation configurations:',len(trials),'best accuracy:',trials[0]['metrics']['accuracy'])
if __name__=='__main__':main()
