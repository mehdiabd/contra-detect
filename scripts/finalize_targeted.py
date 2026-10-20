"""Evaluate the validation-selected candidate once on test and write a decision report."""
import csv
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import config
import numpy as np
import torch
from src.data.loader import load_farstail_split
from src.models.predictor import ContradictionPredictor
from scripts.analyze_errors import metrics


def main():
    torch.set_num_threads(4)
    folder=ROOT/'reports/error_analysis'
    run=json.loads((folder/'targeted_training.json').read_text())
    if run['epochs_completed']!=run['epochs_requested']:raise RuntimeError('Training incomplete')
    baseline=run['baseline_validation'];candidate=run['candidate_validation']
    eligible=(candidate['accuracy']>baseline['accuracy'] and
              candidate['per_class']['contradiction']['f1']>=baseline['per_class']['contradiction']['f1'])
    decision={'selection_split':'validation','eligible_for_adoption':eligible,
      'reason':'Require improved validation accuracy and nondecreasing contradiction F1',
      'baseline_validation':baseline,'candidate_validation':candidate,'production_model_replaced':False}
    if eligible:
        predictor=ContradictionPredictor(run['model'])
        pairs=load_farstail_split('test');logits=[]
        for start in range(0,len(pairs),16):
            batch=pairs[start:start+16]
            logits.append(predictor._scores([p.text_a for p in batch],[p.text_b for p in batch]).numpy())
        probs=np.concatenate(logits);gold=np.array([p.label for p in pairs])
        decision['candidate_test']=metrics(gold,probs.argmax(axis=1))
        decision['meets_85_test']=decision['candidate_test']['accuracy']>=.85
        np.savez(folder/'targeted_test_predictions.npz',probs=probs,gold=gold)
        with (folder/'targeted_test_predictions.csv').open('w',encoding='utf-8-sig',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=['row_index','text_a','text_b','gold','prediction','p_entailment','p_contradiction','p_neutral'])
            writer.writeheader()
            for i,(pair,row) in enumerate(zip(pairs,probs)):
                writer.writerow(dict(row_index=i,text_a=pair.text_a,text_b=pair.text_b,gold=config.NLI_ID2LABEL[pair.label],prediction=config.NLI_ID2LABEL[int(row.argmax())],p_entailment=float(row[0]),p_contradiction=float(row[1]),p_neutral=float(row[2])))
    (folder/'adoption_decision.json').write_text(json.dumps(decision,ensure_ascii=False,indent=2))
    print(json.dumps(decision,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
