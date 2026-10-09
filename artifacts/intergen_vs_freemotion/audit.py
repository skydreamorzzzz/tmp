"""Read-only checkpoint audit against unmodified official FreeMotion classes.

No generation or training. Only wrapper-prefix changes and the official
Stage1-to-Stage2 clone are applied. Cross-attention is never remapped to SA.
Run from project root: .venv/bin/python artifacts/intergen_vs_freemotion/audit.py
"""
import sys
from pathlib import Path
import hashlib
import json
import gc

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'engineering'))
from runtime import model, seed_everything, CODE
import torch


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def compare(net, source):
    target = net.state_dict()
    params = dict(net.named_parameters())
    common = sorted(set(target) & set(source))
    matched = [k for k in common if target[k].shape == source[k].shape]
    mismatched = [{'key': k, 'source': list(source[k].shape), 'target': list(target[k].shape)}
                  for k in common if target[k].shape != source[k].shape]
    def ratio(keys):
        total = sum(p.numel() for k, p in params.items() if k in keys)
        loaded = sum(params[k].numel() for k in matched if k in params and k in keys)
        return {'matched_parameters': loaded, 'target_parameters': total,
                'matched_parameter_ratio': loaded / total if total else None}
    groups = {
        'all': set(params),
        'single_backbone': {k for k in params if k.startswith('decoder.net.net.')},
        'interaction_control_branch': {k for k in params if k.startswith('decoder.net.control_branch.')},
        'zero_linear': {k for k in params if k.startswith('decoder.net.zero_linear.')},
        'clip_text_encoder': {k for k in params if k.startswith(('token_embedding.', 'clip_transformer.', 'ln_final.', 'positional_embedding'))},
        'text_adapter': {k for k in params if k.startswith(('clipTransEncoder.', 'clip_ln.'))},
        'trainable': {k for k, p in params.items() if p.requires_grad},
    }
    result = {'matched_keys': matched,
              'missing_keys': sorted(set(target) - set(source)),
              'unexpected_keys': sorted(set(source) - set(target)),
              'shape_mismatch': mismatched,
              'parameter_groups': {g: ratio(keys) for g, keys in groups.items()}}
    result.update({k+'_count': len(result[k]) for k in ['matched_keys', 'missing_keys', 'unexpected_keys', 'shape_mismatch']})
    result['matched_parameter_ratio'] = result['parameter_groups']['all']['matched_parameter_ratio']
    return result


def main():
    torch.set_num_threads(4)
    seed_everything(1234)
    p = ROOT / 'assets/intergen/intergen.ckpt'
    ckpt = torch.load(p, map_location='cpu')
    state = ckpt['state_dict']
    source = {k[6:] if k.startswith('model.') else k: v for k, v in state.items()}
    safe = {k.replace('decoder.net.', 'decoder.net.net.', 1) if k.startswith('decoder.net.') else k: v
            for k, v in source.items()}
    clone = dict(safe)
    for k, v in safe.items():
        if k.startswith('decoder.net.net.'):
            clone[k.replace('decoder.net.net.', 'decoder.net.control_branch.', 1)] = v
    result = {'generator': {'path': str(p.relative_to(ROOT)), 'sha256': sha256(p),
              'bytes': p.stat().st_size, 'top_level': list(ckpt), 'state_dict_keys': len(state),
              'epoch': ckpt.get('epoch'), 'global_step': ckpt.get('global_step')},
              'parameter_ratio_definition': 'matched named_parameters numel / all target named_parameters numel; includes frozen parameters, excludes buffers and duplicate aliases',
              'safe_mapping': ['strip leading model.', 'decoder.net.* -> decoder.net.net.* (wrapper only)',
                               'multi: clone backbone to control_branch exactly as official Stage1 initializer'],
              'semantic_remapping_applied': False}
    for stage, name in [(1, 'single'), (2, 'multi')]:
        net = model(stage)
        stages = {'raw': compare(net, state), 'lightning_prefix_only': compare(net, source),
                  'safe_wrapper_mapping': compare(net, safe)}
        if stage == 2:
            stages['safe_wrapper_and_official_stage2_clone'] = compare(net, clone)
        applied = safe if stage == 1 else clone
        filtered = {k: v for k, v in applied.items() if k in net.state_dict() and v.shape == net.state_dict()[k].shape}
        load = torch.nn.Module.load_state_dict(net, filtered, strict=False)
        stages['actual_filtered_load'] = {'missing_keys': load.missing_keys, 'unexpected_keys': load.unexpected_keys,
                                        'loaded_tensor_count': len(filtered), 'strict': False,
                                        'purpose': 'audit only; not used for generation'}
        if stage == 2:
            stages['zero_linear_max_abs_after_load'] = max(float(v.abs().max()) for k, v in net.state_dict().items() if 'zero_linear' in k)
        (OUT / (name + '_target_inventory.json')).write_text(json.dumps({k: {'shape': list(v.shape), 'numel': v.numel(),
                'is_named_parameter': k in dict(net.named_parameters())} for k, v in net.state_dict().items()}, indent=2)+'\n')
        result[name] = stages
        for label, row in stages.items():
            if isinstance(row, dict) and 'matched_keys_count' in row:
                print(name, label, {k: row[k] for k in ['matched_keys_count','missing_keys_count','unexpected_keys_count','shape_mismatch_count','matched_parameter_ratio']}, flush=True)
        del net
        gc.collect()
    result['ca_norm_vs_xf_norm'] = []
    for i in range(8):
        prefix = f'decoder.net.blocks.{i}.ca_block.'
        row = {'layer': i}
        for kind in ['weight', 'bias']:
            a = source[prefix+'norm.emb_layers.1.'+kind].float()
            b = source[prefix+'xf_norm.emb_layers.1.'+kind].float()
            row[kind] = {'equal': torch.equal(a, b), 'max_abs_difference': float((a-b).abs().max()),
                         'relative_l2_difference': float((a-b).norm()/a.norm())}
        result['ca_norm_vs_xf_norm'].append(row)
    result['decision'] = 'SEMANTIC_BLOCKER: 48 condition_sa tensors missing even in single. Mapping ca_block to condition_sa_block discards learned xf_norm and changes K/V input; user authorization required before generation.'
    result['official_source_hashes'] = {str(f.relative_to(ROOT)): sha256(f) for f in sorted(CODE.rglob('*.py')) if '__pycache__' not in str(f)}
    (OUT / 'weight_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(result['decision'], flush=True)


if __name__ == '__main__':
    main()
