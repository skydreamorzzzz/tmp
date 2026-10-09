"""Re-evaluate saved physical motions with the official FreeMotion evaluator.

Uses 96-candidate retrieval batches, official metric functions and saved features.
Single requires genuinely independently generated single motions paired as Table 2.
"""
import sys
import argparse
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'engineering'))
from runtime import CODE, seed_everything
import numpy as np
import torch
from configs import get_config
from datasets.evaluator import EvaluatorModelWrapper
from utils.metrics import (euclidean_distance_matrix,calculate_top_k,
    calculate_activation_statistics,calculate_frechet_distance,calculate_diversity,calculate_multimodality)

def batch_for(rows,folder):
    motions=[]
    for r in rows:
        m=np.load(folder/(r['id']+'.npy'))
        assert m.shape==(r['length'],2,262) and np.isfinite(m).all()
        motions.append(np.pad(m,((0,300-len(m)),(0,0),(0,0))))
    m=torch.from_numpy(np.stack(motions)).float()
    return ([r['id'] for r in rows],[r['text'] for r in rows],['placeholder']*len(rows),['placeholder']*len(rows),
            m[:,:,0,:],m[:,:,1,:],torch.tensor([r['length'] for r in rows]),torch.zeros(len(rows),0))

def encode_multimodality(wrapper,mode,allrows):
    selected=json.loads((OUT/'mm_manifest.json').read_text())
    byid={r['id']:r for r in allrows};mmfeatures=[];mmids=[]
    with torch.no_grad():
        for name in selected['ids']:
            m=np.load(OUT/(mode+'_mm')/(name+'.npy'));r=byid[name];b=len(m)
            assert m.shape==(selected['repeats'],r['length'],2,262)
            mt=torch.from_numpy(m).float()
            batch=([name]*b,[r['text']]*b,['placeholder']*b,['placeholder']*b,
                   mt[:,:,0,:],mt[:,:,1,:],torch.tensor([r['length']]*b),torch.zeros(b,0))
            emb=wrapper.get_motion_embeddings(batch).cpu().numpy()
            assert np.isfinite(emb).all();mmfeatures.append(emb);mmids.append(name)
    mmfeatures=np.stack(mmfeatures);seed_everything(1234)
    value=float(calculate_multimodality(mmfeatures,10))
    np.savez_compressed(OUT/(mode+'_mm_features.npz'),motion=mmfeatures,ids=np.array(mmids),
                        repeat_indices=np.argsort(np.ones(selected['repeats']))[::-1].copy())
    return value,mmids

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',default='native_two_person');p.add_argument('--sanity',action='store_true')
    p.add_argument('--skip-mm',action='store_true');p.add_argument('--mm-only',action='store_true');args=p.parse_args()
    torch.set_num_threads(4);seed_everything(1234)
    manifest=json.loads((OUT/'test_manifest.json').read_text());allrows=manifest['rows']
    folder=OUT/args.mode
    available=[r for r in allrows if (folder/(r['id']+'.npy')).exists()]
    if not args.sanity and len(available)!=len(allrows):raise RuntimeError('Formal evaluation requires all test cases generated')
    rows=available
    rng=np.random.RandomState(1234);order=rng.permutation(len(rows))
    rows=[rows[i] for i in order]
    if not args.sanity:rows=rows[:len(rows)//96*96]
    wrapper=EvaluatorModelWrapper(get_config(str(CODE/'configs/eval_model.yaml')),torch.device('cuda:0'))
    if args.mm_only:
        assert not args.sanity and not args.skip_mm
        result=json.loads((OUT/(args.mode+'_metrics.json')).read_text())
        value,mmids=encode_multimodality(wrapper,args.mode,allrows)
        result['metrics']['Multimodality']=value
        result['protocol'].update(mm_selected_cases=len(mmids),mm_repeats=30,mm_pairs=10)
        (OUT/(args.mode+'_metrics.json')).write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2),flush=True)
        return
    text_features=[];features=[];gtfeatures=[];ids=[];distance_sum=0.;topcount=np.zeros(3)
    gt_distance_sum=0.;gt_topcount=np.zeros(3)
    with torch.no_grad():
        for lo in range(0,len(rows),96):
            group=rows[lo:lo+96]
            batch=batch_for(group,folder)
            te,me=wrapper.get_co_embeddings(batch)
            gm=wrapper.get_motion_embeddings(batch_for(group,OUT/'ground_truth'))
            t,a,g=te.cpu().numpy(),me.cpu().numpy(),gm.cpu().numpy()
            assert np.isfinite(t).all() and np.isfinite(a).all() and np.isfinite(g).all()
            d=euclidean_distance_matrix(t,a);distance_sum+=np.trace(d)
            topcount+=calculate_top_k(np.argsort(d,axis=1),3).sum(axis=0)
            gd=euclidean_distance_matrix(t,g);gt_distance_sum+=np.trace(gd)
            gt_topcount+=calculate_top_k(np.argsort(gd,axis=1),3).sum(axis=0)
            text_features.append(t);features.append(a);gtfeatures.append(g)
            align=np.argsort([r['length'] for r in group])[::-1].copy();ids.extend([group[i]['id'] for i in align])
            print('Encoded',lo+len(group),'/',len(rows),flush=True)
        a=np.concatenate(features);g=np.concatenate(gtfeatures);t=np.concatenate(text_features)
        mu,cov=calculate_activation_statistics(a);gm,gcov=calculate_activation_statistics(g)
        fid=float(calculate_frechet_distance(gm,gcov,mu,cov))
        seed_everything(1234);diversity=float(calculate_diversity(a,300)) if len(a)>300 else None
        metrics={'FID':fid,'R_Precision':(topcount/len(a)).tolist(),'MM_Distance':float(distance_sum/len(a)),
                 'Diversity':diversity,'Multimodality':None}
        mmfolder=OUT/(args.mode+'_mm');mmids=[]
        if mmfolder.exists() and not args.skip_mm:
            metrics['Multimodality'],mmids=encode_multimodality(wrapper,args.mode,allrows)
    suffix='_sanity' if args.sanity else ''
    np.savez_compressed(OUT/(args.mode+suffix+'_features.npz'),motion=a,text=t,ground_truth=g,ids=np.array(ids))
    seed_everything(1234)
    gt_sanity={'MM_Distance':float(gt_distance_sum/len(g)),
               'R_Precision':(gt_topcount/len(g)).tolist(),
               'Diversity':float(calculate_diversity(g,300)) if len(g)>300 else None,
               'purpose':'Real motion calibration, not generated model metrics'}
    result={'label':'InterGen native two-person' if args.mode=='native_two_person' else 'InterGen single',
            'ground_truth_sanity':gt_sanity,
            'metrics':metrics,'protocol':{'sanity_only':args.sanity,'generated_cases':len(available),'evaluation_cases':len(a),
               'full_valid_test_cases':len(allrows),'retrieval_candidates':96 if not args.sanity else len(a),
               'drop_last':not args.sanity,'seed':1234,'generation_replications':1,'confidence_interval':None,
               'diversity_pairs':300 if diversity is not None else None,'mm_selected_cases':len(mmids),
               'mm_repeats':30 if mmids else None,'mm_pairs':10 if mmids else None,
               'fid_feature_scale':6,'diversity_feature_scale':6,'diversity_distance_divisor':2,
               'mm_distance_feature_scale':1,'multimodality_feature_scale':1,
               'evaluator':'unmodified FreeMotion EvaluatorModelWrapper and utils.metrics; interclip.ckpt strict load',
               'paper_comparison':'descriptive reference only; paper repeats metrics20x / MM5x; paper baseline InterGen* retrained'}}
    (OUT/(args.mode+suffix+'_metrics.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
