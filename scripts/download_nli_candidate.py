"""Download a fixed, public multilingual NLI candidate into this workspace."""
from pathlib import Path
from huggingface_hub import snapshot_download

REPO='MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7'
REVISION='b5113eb38ab63efdd7f280f8c144ea8b13f978ce'
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
 path=snapshot_download(REPO,revision=REVISION,local_dir=ROOT/'models/saved/mdeberta_nli',
  allow_patterns=['config.json','model.safetensors','tokenizer.json','tokenizer_config.json','spm.model','special_tokens_map.json','README.md'],max_workers=3)
 print(path)
