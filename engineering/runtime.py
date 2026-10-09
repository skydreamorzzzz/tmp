"""Use upstream FreeMotion math/data classes without importing Lightning."""
import importlib.util
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / 'code/FreeMotion-Codes'
os.environ.setdefault('MPLBACKEND', 'Agg')
sys.dont_write_bytecode = True
sys.path.insert(0, str(CODE))
os.chdir(str(CODE))

import numpy as np
import torch
import clip
from configs import get_config
from models import InterGenSpatialControlNet

_clip_load = clip.load


def local_clip_load(name, *args, **kwargs):
    cache = ROOT / 'assets/clip'
    if not (cache / 'ViT-L-14-336px.pt').is_file():
        raise FileNotFoundError('Official CLIP asset must be downloaded first.')
    kwargs.setdefault('download_root', str(cache))
    return _clip_load(name, *args, **kwargs)


clip.load = local_clip_load


def load_source(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def dataset(stage, split='train', cache=True):
    module = load_source('official_interhuman', CODE / 'datasets/interhuman.py')
    cfg = get_config(str(CODE / ('configs/datasets_single.yaml' if stage == 1 else 'configs/datasets_inter.yaml'))).interhuman.clone()
    cfg.defrost()
    cfg.DATA_ROOT = str(ROOT / 'data/InterHuman-engineering')
    cfg.MODE = split
    cfg.CACHE = cache
    cfg.freeze()
    cls = module.SingleHumanDataset if stage == 1 else module.InterHumanDataset
    return cls(cfg)


def model(stage):
    path = CODE / ('configs/model_single.yaml' if stage == 1 else 'configs/model_inter.yaml')
    cfg = get_config(str(path))
    return InterGenSpatialControlNet(cfg)


def batch_to_model(data, device):
    name, text, text_multi_person, motion1, motion2, lengths, spatial = data
    first = motion1.to(device=device, dtype=torch.float32, non_blocking=True)
    motions = first if motion2.numel() == 0 else torch.cat([first, motion2.to(device=device, dtype=torch.float32, non_blocking=True)], dim=-1)
    return {
        'motions': motions, 'motion_lens': lengths.to(device=device, dtype=torch.long),
        'person_num': motions.shape[-1] // first.shape[-1],
        'text': None if isinstance(text, torch.Tensor) else text,
        'text_multi_person': None if isinstance(text_multi_person, torch.Tensor) else text_multi_person,
    }


def exact_load(net, checkpoint):
    # Stage-2's upstream load_state_dict also initializes a control branch. Bypass
    # that initializer when restoring an already-trained full checkpoint.
    state = checkpoint['model'] if 'model' in checkpoint else checkpoint['state_dict']
    state = {key.removeprefix('model.') if hasattr(key, 'removeprefix') else (key[6:] if key.startswith('model.') else key): value
             for key, value in state.items()}
    torch.nn.Module.load_state_dict(net, state, strict=True)
