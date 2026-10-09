"""Freeze real test prompts, official preprocessing and per-case lengths."""
import sys
from pathlib import Path
import json
import hashlib
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'engineering'))
from runtime import CODE, load_source, seed_everything
import numpy as np
import torch
from configs import get_config

def main():
    torch.set_num_threads(4)
    seed_everything(1234)
    module=load_source('probe_official_dataset',CODE/'datasets/interhuman.py')
    cfg=get_config(str(CODE/'configs/datasets_inter.yaml')).interhuman_test.clone()
    cfg.defrost();cfg.DATA_ROOT=str(ROOT/'data/InterHuman-engineering');cfg.CACHE=True;cfg.freeze()
    ds=module.InterHumanPipelineInferDataset(cfg)
    gt=OUT/'ground_truth';gt.mkdir(exist_ok=True)
    rows=[]
    for i in sorted(range(len(ds)),key=lambda i:int(ds.data_list[i]['name'])):
        name=ds.data_list[i]['name'];seed=1234+int(name)
        seed_everything(seed)
        name,text,text1,text2,m1,m2,length,_=ds[i]
        motion=np.stack([m1[:length],m2[:length]],axis=1).astype(np.float32)
        assert motion.shape==(length,2,262) and np.isfinite(motion).all()
        np.save(gt/(name+'.npy'),motion)
        rows.append({'id':name,'text':text,'text1':text1,'text2':text2,'length':int(length),
                     'preprocess_seed':seed,'ground_truth':str((gt/(name+'.npy')).relative_to(OUT))})
    original=(ROOT/'data/InterHuman/split/test.txt').read_text().splitlines()
    retained=(ROOT/'data/InterHuman-engineering/split/test.txt').read_text().splitlines()
    manifest={'seed':1234,'split':'test','original_count':len(original),'engineering_count':len(retained),
              'official_loader_count':len(ds),'excluded_from_original':sorted(set(original)-{r['id'] for r in rows}),
              'selection':'all official loader valid test cases, numeric ID order',
              'motion_length':'official per-case valid length <=300; identical frozen lengths for every mode',
              'normalization':'official global_mean.npy/global_std.npy; evaluator receives physical features',
              'data_preprocessing':'unmodified FreeMotion InterHumanPipelineInferDataset; per-case RNG seed freezes caption choice/crop/actor order',
              'rows':rows}
    (OUT/'test_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print('Frozen',len(rows),'test samples, exclusions',manifest['excluded_from_original'],flush=True)
if __name__=='__main__':main()
