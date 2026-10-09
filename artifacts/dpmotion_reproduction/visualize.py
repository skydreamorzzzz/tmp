#!/usr/bin/env python3
"""Render saved DPMotion 4-person motion arrays without changing their coordinates."""
import argparse
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--file-id', default='2636')
    p.add_argument('--conditions', nargs='+', default=['A_intergen_no_rag_no_guidance', 'D_rag_plus_guidance'])
    p.add_argument('--fps', type=int, default=30)
    p.add_argument('--stride', type=int, default=3)
    args = p.parse_args()
    import sys
    sys.path.insert(0, str(ROOT / 'code/DPMotion'))
    import utils.plot_script as plot_script
    from utils.paramUtil import kit_kinematic_chain
    # This environment has Pillow but no ffmpeg executable. Pass PillowWriter
    # explicitly; Matplotlib 3.2's automatic fallback uses a broken
    # `next(writers)` call when ffmpeg is unavailable.
    from matplotlib.animation import FuncAnimation as _FuncAnimation, PillowWriter

    class PillowFuncAnimation(_FuncAnimation):
        def save(self, filename, fps=None, **kwargs):
            writer = PillowWriter(fps=fps or 30)
            return super().save(filename, writer=writer, **kwargs)

    plot_script.FuncAnimation = PillowFuncAnimation
    visdir = OUT / 'visualizations'
    visdir.mkdir(parents=True, exist_ok=True)
    for condition in args.conditions:
        ablation_condition = {
            'A_intergen_no_rag_no_guidance': 'A_no_rag_no_guidance',
            'B_rag_no_guidance': 'B_rag_no_guidance',
            'C_no_rag_guidance': 'C_no_rag_guidance',
            'D_rag_plus_guidance': 'D_rag_guidance',
        }.get(condition, condition)
        src = OUT / 'ablations/full_2x2' / ablation_condition / 'joints' / ('scene_{}_4p.npy'.format(args.file_id))
        if not src.exists():
            src = OUT / 'ablations/four_way' / condition / ('scene_{}_4p.npy'.format(args.file_id))
        if not src.exists():
            src = OUT / 'samples/full' / ('scene_{}_4p.npy'.format(args.file_id))
        arr = np.load(src)
        if arr.ndim != 4 or arr.shape[0] != 4 or arr.shape[2:] != (22, 3):
            raise ValueError('{} has unexpected shape {}'.format(src, arr.shape))
        arr = arr[:, ::max(1, args.stride)].astype(np.float32, copy=True)
        dest = visdir / '{}_{}.gif'.format(args.file_id, condition)
        plot_script.plot_3d_motion_multi(str(dest), kit_kinematic_chain, arr,
                                         '{} | {}'.format(args.file_id, condition), fps=args.fps, radius=4)
        print('WROTE', dest.relative_to(ROOT), 'shape=', arr.shape, flush=True)


if __name__ == '__main__':
    main()
