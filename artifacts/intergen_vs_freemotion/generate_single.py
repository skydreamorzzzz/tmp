"""Zero-B single generation, independent actors paired for Table2 evaluation."""
import argparse
import json
import time
from collections import defaultdict
import hashlib
from generate_native import ROOT,OUT,CODE,seed_all
import numpy as np
import torch
from configs import get_config
from models import InterGen
from utils.utils import MotionNormalizer
from single_path import install_single


def main():
    p=argparse.ArgumentParser();p.add_argument('--limit',type=int,default=0);p.add_argument('--batch-size',type=int,default=8)
    p.add_argument('--mm-count',type=int,default=0);p.add_argument('--mm-repeats',type=int,default=30);args=p.parse_args()
    torch.set_num_threads(4);seed_all(1234)
    manifest=json.loads((OUT/'test_manifest.json').read_text());rows=manifest['rows'][:args.limit] if args.limit else manifest['rows']
    cfg=get_config(str(CODE/'configs/model.yaml'));net=InterGen(cfg)
    c=torch.load(ROOT/'assets/intergen/intergen.ckpt',map_location='cpu')
    s={k[6:] if k.startswith('model.') else k:v for k,v in c['state_dict'].items()}
    loaded=net.load_state_dict(s,strict=True);install_single(net)
    meta={'strict_load':True,'keys':len(s),'missing_keys':loaded.missing_keys,'unexpected_keys':loaded.unexpected_keys,
          'checkpoint_modified':False,'added_parameter_count':0,'native_source_modified':False,
          'zero_B':'normalized zero input at every denoising step, B hidden stream updated through all8 blocks; return A only',
          'port_sha256':hashlib.sha256((OUT/'single_path.py').read_bytes()).hexdigest(),
          'settings':{'seed':1234,'strategy':cfg.STRATEGY,'ddim_steps':50,'eta':0,'cfg_weight':cfg.CFG_WEIGHT,
                      'diffusion_steps':cfg.DIFFUSION_STEPS,'precision':'FP32','generation_batch_size':args.batch_size,
                      'text':'separate test text1/text2, no global interactive text'},'cases':len(rows),
          'reference':'https://github.com/skydreamorzzzz/DPMotion/blob/main/models/nets.py',
          'reference_sampling_difference':'DPMotion current forward_single includes1000-step graph sampling; port uses native DDIM50 only as user instructed'}
    del c,s;net.eval().to('cuda:0')
    # Meaningful semantic check: port equals native two-person denoiser with
    # normalized zero B input, projected to actor A; all weights stay unchanged.
    with torch.no_grad():
        seed_all(1234);x=torch.randn(2,17,262,device='cuda');t=torch.tensor([400,800],device='cuda');cond=torch.randn(2,768,device='cuda')
        a=net.decoder.net.single_forward(x,t,cond=cond)
        b=net.decoder.net(torch.cat([x,torch.zeros_like(x)],dim=-1),t,cond=cond)[...,:262]
        err=float((a-b).abs().max())
        print('Zero-B native A equivalence max abs',err,flush=True)
        assert torch.allclose(a,b,atol=1e-5,rtol=1e-5)
        meta['zero_B_native_A_equivalence_max_abs']=err
        meta['zero_B_native_A_equivalence_tolerance']={'atol':1e-5,'rtol':1e-5}
    (OUT/'single_load.json').write_text(json.dumps(meta,indent=2)+'\n');print('Single strict load and zero-B numerical equivalence passed',flush=True)
    norm=MotionNormalizer();dest=OUT/'single';dest.mkdir(exist_ok=True);buckets=defaultdict(list)
    for row in rows:buckets[row['length']].append(row)
    start=time.monotonic();done=0
    with torch.no_grad(),(OUT/'single_generation_batches.jsonl').open('a',buffering=1) as log:
        for length,group in sorted(buckets.items()):
            for lo in range(0,len(group),args.batch_size):
                chunk=group[lo:lo+args.batch_size]
                if all((dest/(r['id']+'.npy')).exists() for r in chunk):done+=len(chunk);continue
                # Same case-batch seed as native double; separate calls give
                # independent noises for the two captions, without interactions.
                seed=1234+int(chunk[0]['id']);seed_all(seed);actors=[];before=time.monotonic()
                for actor in ['text1','text2']:
                    batch={'text':[r[actor] for r in chunk],'motion_lens':torch.tensor([length],device='cuda',dtype=torch.long)}
                    result=net.forward_test_single(batch)['output']
                    actors.append(norm.backward(result.cpu().numpy()).astype(np.float32))
                physical=np.stack(actors,axis=2);assert np.isfinite(physical).all()
                for row,m in zip(chunk,physical):np.save(dest/(row['id']+'.npy'),m)
                done+=len(chunk);info={'ids':[r['id'] for r in chunk],'length':length,'seed':seed,'seconds':time.monotonic()-before,
                                      'elapsed_seconds':time.monotonic()-start,'completed':done,'total':len(rows)}
                log.write(json.dumps(info)+'\n');print(json.dumps(info),flush=True)
        if args.mm_count:
            selection=json.loads((OUT/'mm_manifest.json').read_text());byid={r['id']:r for r in rows};chosen=[byid[i] for i in selection['ids']]
            mm=OUT/'single_mm';mm.mkdir(exist_ok=True)
            for i,row in enumerate(chosen):
                path=mm/(row['id']+'.npy')
                if path.exists():continue
                seed=10001234+int(row['id']);seed_all(seed);length=row['length'];parts=[]
                for lo in range(0,args.mm_repeats,args.batch_size):
                    count=min(args.batch_size,args.mm_repeats-lo);actors=[]
                    for actor in ['text1','text2']:
                        batch={'text':[row[actor]]*count,'motion_lens':torch.tensor([length],device='cuda',dtype=torch.long)}
                        result=net.forward_test_single(batch)['output'];actors.append(norm.backward(result.cpu().numpy()).astype(np.float32))
                    parts.append(np.stack(actors,axis=2))
                motions=np.concatenate(parts,axis=0);assert np.isfinite(motions).all();np.save(path,motions)
                print('MM',i+1,len(chosen),row['id'],'elapsed',time.monotonic()-start,flush=True)
    print('Single generation finished elapsed',time.monotonic()-start,flush=True)
if __name__=='__main__':main()
