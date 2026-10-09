"""Unmodified native InterGen generation with strict generator loading."""
import os
import sys
import argparse
from pathlib import Path
import json
import time
import random
import hashlib
from collections import defaultdict

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
CODE=ROOT/'code/InterGen-official'
sys.path.insert(0,str(CODE))
os.chdir(CODE)
import numpy as np
import torch
import clip
from configs import get_config
from models import InterGen
from utils.utils import MotionNormalizer

_load=clip.load
def local_clip(name,*args,**kwargs):
    kwargs.setdefault('download_root',str(ROOT/'assets/clip'))
    return _load(name,*args,**kwargs)
clip.load=local_clip

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)

def main():
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int,default=0);p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--mm-count',type=int,default=0);p.add_argument('--mm-repeats',type=int,default=30)
    args=p.parse_args()
    torch.set_num_threads(4);seed_all(1234)
    manifest=json.loads((OUT/'test_manifest.json').read_text())
    rows=manifest['rows'][:args.limit] if args.limit else manifest['rows']
    cfg=get_config(str(CODE/'configs/model.yaml'))
    net=InterGen(cfg)
    ckpt=torch.load(ROOT/'assets/intergen/intergen.ckpt',map_location='cpu')
    state={k[6:] if k.startswith('model.') else k:v for k,v in ckpt['state_dict'].items()}
    status=net.load_state_dict(state,strict=True)
    provenance={'generator_epoch':ckpt['epoch'],'generator_global_step':ckpt['global_step'],
                'keys':len(state),'missing_keys':status.missing_keys,'unexpected_keys':status.unexpected_keys,
                'strict_load':True,'single_path_present':hasattr(net,'forward_single') or hasattr(net,'single_forward'),
                'settings':{'seed':1234,'strategy':cfg.STRATEGY,'ddim_steps':50,'eta':0,'cfg_weight':cfg.CFG_WEIGHT,
                            'diffusion_steps':cfg.DIFFUSION_STEPS,'normalization':'native official','precision':'FP32',
                            'generation_batch_size':args.batch_size,'length_bucketing':'equal lengths only'},
                'source_hashes':{str(f.relative_to(CODE)):hashlib.sha256(f.read_bytes()).hexdigest() for f in CODE.rglob('*.py')},
                'requested_cases':len(rows),'mm_count':args.mm_count,'mm_repeats':args.mm_repeats}
    (OUT/'native_load.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('Native checkpoint strict load passed',flush=True)
    del ckpt,state
    net.eval().to('cuda:0')
    normalizer=MotionNormalizer()
    dest=OUT/'native_two_person';dest.mkdir(exist_ok=True)
    buckets=defaultdict(list)
    for row in rows:buckets[row['length']].append(row)
    start=time.monotonic();done=0
    with torch.no_grad(),(OUT/'generation_batches.jsonl').open('a',buffering=1) as log:
        for length,group in sorted(buckets.items()):
            for lo in range(0,len(group),args.batch_size):
                chunk=group[lo:lo+args.batch_size]
                if all((dest/(r['id']+'.npy')).is_file() for r in chunk):done+=len(chunk);continue
                batchseed=1234+int(chunk[0]['id']);seed_all(batchseed)
                before=time.monotonic()
                batch={'text':[r['text'] for r in chunk],'motion_lens':torch.tensor([length],device='cuda',dtype=torch.long)}
                result=net.forward_test(batch)['output'].reshape(len(chunk),length,2,262)
                physical=normalizer.backward(result.cpu().numpy()).astype(np.float32)
                assert np.isfinite(physical).all()
                for row,motion in zip(chunk,physical):np.save(dest/(row['id']+'.npy'),motion)
                done+=len(chunk)
                info={'ids':[r['id'] for r in chunk],'length':length,'seed':batchseed,
                      'seconds':time.monotonic()-before,'elapsed_seconds':time.monotonic()-start,'completed':done,'total':len(rows)}
                log.write(json.dumps(info)+'\n');print(json.dumps(info),flush=True)
                del result,physical
        if args.mm_count:
            mm=OUT/'native_two_person_mm';mm.mkdir(exist_ok=True)
            rng=random.Random(1234); chosen=rng.sample(rows,min(args.mm_count,len(rows)))
            (OUT/'mm_manifest.json').write_text(json.dumps({'ids':[r['id'] for r in chosen],'repeats':args.mm_repeats,'selection_seed':1234},indent=2)+'\n')
            for i,row in enumerate(chosen):
                path=mm/(row['id']+'.npy')
                if path.exists():continue
                seed=10001234+int(row['id']);seed_all(seed);parts=[];length=row['length']
                for lo in range(0,args.mm_repeats,args.batch_size):
                    b=min(args.batch_size,args.mm_repeats-lo)
                    batch={'text':[row['text']]*b,'motion_lens':torch.tensor([length],device='cuda',dtype=torch.long)}
                    result=net.forward_test(batch)['output'].reshape(b,length,2,262)
                    parts.append(normalizer.backward(result.cpu().numpy()).astype(np.float32))
                motions=np.concatenate(parts,axis=0);assert np.isfinite(motions).all();np.save(path,motions)
                print('MM',i+1,len(chosen),row['id'],'elapsed',time.monotonic()-start,flush=True)
    print('Generation finished elapsed',time.monotonic()-start,flush=True)

if __name__=='__main__':main()
