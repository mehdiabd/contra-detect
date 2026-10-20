"""Train an isolated candidate with native PyTorch, selecting epochs only on validation."""
import hashlib
import json
import random
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import config
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.data.loader import load_pairs_csv,load_farstail_split
from scripts.analyze_errors import metrics


def main():
    seed=42
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.set_num_threads(4)
    train_path=ROOT/'data/labeled/targeted_v1/train_augmented.csv'
    train=load_pairs_csv(train_path);val=load_farstail_split('val')
    output=ROOT/'models/saved/targeted_v1';output.mkdir(parents=True,exist_ok=True)
    archived=ROOT/'models/saved/baseline_before_targeted_v1'
    source=archived if (archived/'config.json').exists() else config.BEST_MODEL_DIR
    baseline_report=json.loads((ROOT/'reports/error_analysis/summary.json').read_text())
    hasher=hashlib.sha256()
    with (source/'model.safetensors').open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):hasher.update(block)
    if hasher.hexdigest()!=baseline_report['model_sha256']:
        raise ValueError('Baseline weights do not match error analysis; refresh the baseline analysis first.')
    tokenizer=AutoTokenizer.from_pretrained(str(source),local_files_only=True)
    model=AutoModelForSequenceClassification.from_pretrained(str(source),local_files_only=True)
    if {int(k):v for k,v in model.config.id2label.items()}!=config.NLI_ID2LABEL:raise ValueError('Label order mismatch')
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu');model.to(device)
    def encode(pairs):
        # Cache unpadded tokenization; dynamically pad each batch without dropping text beyond the existing limit.
        return tokenizer([p.text_a for p in pairs],[p.text_b for p in pairs],truncation=True,max_length=256,padding=False)
    train_enc=encode(train);val_enc=encode(val)
    def batch(encoded, pairs, indices):
        tensors=tokenizer.pad([{k:v[i] for k,v in encoded.items()} for i in indices],padding=True,return_tensors='pt')
        tensors={k:v.to(device) for k,v in tensors.items()}
        tensors['labels']=torch.tensor([pairs[i].label for i in indices],device=device)
        return tensors
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-6,weight_decay=.01)
    batch_size=16;epochs=2;steps_per_epoch=(len(train)+batch_size-1)//batch_size;total_steps=epochs*steps_per_epoch
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda step:min((step+1)/max(int(.1*total_steps),1),max(0,(total_steps-step)/max(int(.9*total_steps),1))))
    baseline=json.loads((ROOT/'reports/error_analysis/summary.json').read_text())['splits']['val']['forward']
    history=[];best=-1;global_step=0;started=time.monotonic()
    for epoch in range(epochs):
        model.train();order=np.random.default_rng(seed+epoch).permutation(len(train)).tolist();loss_sum=0
        for start in range(0,len(order),batch_size):
            indices=order[start:start+batch_size];optimizer.zero_grad(set_to_none=True)
            loss=model(**batch(train_enc,train,indices)).loss
            if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step();scheduler.step();loss_sum+=float(loss.detach());global_step+=1
            if global_step%10==0:
                print(json.dumps({'epoch':epoch+1,'step':global_step,'total_steps':total_steps,'loss':float(loss.detach()),'elapsed_seconds':round(time.monotonic()-started)}),flush=True)
        model.eval();logits=[]
        with torch.no_grad():
            for start in range(0,len(val),batch_size):
                inputs=batch(val_enc,val,list(range(start,min(start+batch_size,len(val)))));inputs.pop('labels')
                logits.append(model(**inputs).logits.cpu().numpy())
        logits=np.concatenate(logits);gold=np.array([p.label for p in val]);score=metrics(gold,logits.argmax(axis=1))
        history.append({'epoch':epoch+1,'mean_loss':loss_sum/steps_per_epoch,'validation':score})
        print(json.dumps(history[-1]),flush=True)
        if score['accuracy']>best:
            best=score['accuracy'];model.save_pretrained(output);tokenizer.save_pretrained(output)
            np.savez(output/'validation_predictions.npz',logits=logits,gold=gold)
        payload={'protocol':'warm-start current local model; train plus independent synthetic; validation selects epoch',
          'historical_baseline_training_provenance':'not fully established', 'train_n':len(train),
          'base_model':str(source),'base_model_sha256':hasher.hexdigest(),
          'train_sha256':hashlib.sha256(train_path.read_bytes()).hexdigest(),'epochs_requested':epochs,
          'epochs_completed':epoch+1,'learning_rate':3e-6,'seed':seed,'device':str(device),'model':str(output),
          'baseline_validation':baseline,'history':history,'candidate_validation':max(history,key=lambda x:x['validation']['accuracy'])['validation'],
          'production_model_replaced':False}
        (ROOT/'reports/error_analysis/targeted_training.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2))
    print('Training completed',flush=True)

if __name__=='__main__':main()
