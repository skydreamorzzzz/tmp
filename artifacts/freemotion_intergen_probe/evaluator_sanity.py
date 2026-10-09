"""Official InterCLIP sanity on real test motions only; no generator involved."""
import sys
from pathlib import Path
import json
import hashlib
import random

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'engineering'))
from runtime import CODE, load_source, seed_everything
import numpy as np
import torch
from configs import get_config
from torch.utils.data._utils.collate import default_collate
from utils.metrics import (euclidean_distance_matrix, calculate_top_k,
                          calculate_activation_statistics, calculate_frechet_distance,
                          calculate_diversity, calculate_multimodality)
from datasets.evaluator import EvaluatorModelWrapper


def main():
    torch.set_num_threads(4)
    seed_everything(1234)
    checkpoint_path = CODE / 'eval_model/interclip.ckpt'
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state = checkpoint['state_dict']
    inventory = {k: {'shape': list(v.shape), 'numel': v.numel(), 'dtype': str(v.dtype)} for k,v in state.items()}
    (OUT / 'interclip_inventory.json').write_text(json.dumps(inventory, indent=2)+'\n')
    h = hashlib.sha256()
    with checkpoint_path.open('rb') as f:
        for chunk in iter(lambda: f.read(8*1024*1024), b''): h.update(chunk)
    meta = {'path': str(checkpoint_path.relative_to(ROOT)), 'resolved_path': str(checkpoint_path.resolve()),
            'sha256': h.hexdigest(), 'bytes': checkpoint_path.stat().st_size,
            'top_level': list(checkpoint), 'epoch': checkpoint.get('epoch'),
            'global_step': checkpoint.get('global_step'), 'keys': len(state),
            'purpose': 'InterCLIP real test motion evaluator sanity; these are not probe metrics'}
    del state, checkpoint
    wrapper = EvaluatorModelWrapper(get_config(str(CODE / 'configs/eval_model.yaml')), torch.device('cuda:0'))
    meta['strict_checkpoint_load'] = True
    print('Official evaluator strict load passed', flush=True)
    module = load_source('official_probe_dataset', CODE / 'datasets/interhuman.py')
    cfg = get_config(str(CODE / 'configs/datasets_inter.yaml')).interhuman_test.clone()
    cfg.defrost()
    cfg.DATA_ROOT = str(ROOT / 'data/InterHuman-engineering')
    cfg.CACHE = True
    cfg.freeze()
    ds = module.InterHumanPipelineInferDataset(cfg)
    indices = sorted(range(len(ds)), key=lambda i: int(ds.data_list[i]['name']))[:96]
    samples = []
    for i in indices:
        seed_everything(1234 + int(ds.data_list[i]['name']))
        samples.append(ds[i])
    batch = default_collate(samples)
    # Verify normalization round trip independently; evaluator expects physical
    # (denormalized) joint features, as in the official EvaluationDataset.
    from utils.utils import MotionNormalizer
    normalizer = MotionNormalizer()
    m = batch[4].numpy().astype(np.float32)
    rt = normalizer.backward(normalizer.forward(m))
    meta['normalization_roundtrip_max_abs'] = float(np.max(np.abs(rt-m)))
    with torch.no_grad():
        te, me = wrapper.get_co_embeddings(batch)
        direct = wrapper.get_motion_embeddings(batch)
        te_again, me_again = wrapper.get_co_embeddings(batch)
    t, a = te.cpu().numpy(), me.cpu().numpy()
    assert np.isfinite(t).all() and np.isfinite(a).all()
    assert torch.allclose(me, direct, atol=1e-6)
    assert torch.equal(me, me_again) and torch.equal(te, te_again)
    alignment = np.argsort(batch[6].tolist())[::-1].copy()
    aligned_names = [batch[0][i] for i in alignment]
    perm = list(batch)
    for j in [0,1,2,3]: perm[j] = batch[j][::-1]
    for j in [4,5,6,7]: perm[j] = batch[j].flip(0)
    with torch.no_grad(): pt, pm = wrapper.get_co_embeddings(perm)
    pa = np.argsort(perm[6].tolist())[::-1].copy()
    perm_names = [perm[0][i] for i in pa]
    idx = [perm_names.index(n) for n in aligned_names]
    motion_perm_error = float((me-pm[idx]).abs().max())
    text_perm_error = float((te-pt[idx]).abs().max())
    assert motion_perm_error < 1e-5 and text_perm_error < 1e-5
    np.savez_compressed(OUT / 'evaluator_sanity_embeddings.npz', text=t, motion=a,
                        names=np.array(aligned_names), lengths=batch[6].numpy()[alignment])
    dist = euclidean_distance_matrix(t,a)
    mu,cov = calculate_activation_statistics(a)
    self_fid = float(calculate_frechet_distance(mu,cov,mu,cov))
    assert abs(self_fid) < 1e-3
    seed_everything(1234)
    sanity = {'MM_Distance': float(np.trace(dist)/len(a)),
              'R_Precision': calculate_top_k(np.argsort(dist,axis=1),3).mean(axis=0).tolist(),
              'Diversity_sanity_30_pairs': float(calculate_diversity(a,30)),
              'FID_self_comparison': self_fid,
              'Multimodality_duplicate_embedding_sanity': float(calculate_multimodality(np.repeat(a[:,None,:],30,axis=1),10))}
    assert sanity['Multimodality_duplicate_embedding_sanity'] == 0
    meta.update(seed=1234, batch_size=96, selected_sample_count=len(samples),
                official_loader_filtered_test_count=len(ds), original_test_count=1177,
                engineered_test_count=1176, selected_ids=list(batch[0]),
                lengths=batch[6].tolist(), selection='first 96 numeric IDs; official stochastic preprocessing seeded per ID',
                generated_samples=0, embeddings_shape=list(a.shape), text_shape=list(t.shape),
                deterministic_repeat=True, motion_embedding_wrapper_agreement=True,
                permutation_motion_max_abs=motion_perm_error, permutation_text_max_abs=text_perm_error,
                ground_truth_sanity= sanity,
                comparison_to_paper_permitted=False,
                limitation='One batch of real test motions; no generator evaluation, no paper replication.')
    (OUT / 'evaluator_sanity.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta,indent=2),flush=True)


if __name__ == '__main__': main()
