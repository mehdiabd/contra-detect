"""Freeze a small validation-selected ensemble before evaluating test."""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scripts.analyze_errors import metrics


def main():
 root=ROOT/'reports/round2'
 trials=json.loads((root/'architecture_ensemble_trials.json').read_text())
 baseline=metrics(np.load(root/'current_val.npz')['gold'],np.load(root/'current_val.npz')['probs'].argmax(1))
 eligible=[x for x in trials['trials'] if x['metrics']['accuracy']>baseline['accuracy'] and x['metrics']['per_class']['contradiction']['f1']>=baseline['per_class']['contradiction']['f1']]
 if not eligible:raise RuntimeError('No validation improvement')
 selected=max(eligible,key=lambda x:(x['metrics']['accuracy'],x['metrics']['per_class']['contradiction']['f1']))
 names=trials['names'];paths={'current':'../targeted_v1','parsnlu':'../parsnlu_farstail','fallback':None,'mdeberta':'../mdeberta_nli'}
 assert selected['weights'][names.index('fallback')]==0, 'External fallback must be snapshotted before deployment'
 members=[{'path':paths[name],'weight':weight,'name':name} for name,weight in zip(names,selected['weights']) if weight>0]
 bundle=ROOT/'models/saved/round2_ensemble';bundle.mkdir(exist_ok=True)
 manifest={'version':1,'members':members,'max_length':512,'selection_split':'validation','canonical_labels':['entailment','contradiction','neutral']}
 (bundle/'inference_config.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
 z=np.load(root/'current_val.npz');p=sum(w*np.load(root/f'{n}_val.npz')['probs'] for n,w in zip(names,selected['weights']) if w>0)
 np.savez(root/'selected_val.npz',probs=p,gold=z['gold'])
 payload={'selection_split':'val','test_used_for_selection':False,'selected':selected,'baseline_validation':baseline,
          'bundle_path':str(bundle),'manifest':manifest,'target_test_accuracy':.85,'status':'frozen_before_test'}
 (root/'selection.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2))
 print(json.dumps(payload,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
