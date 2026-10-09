"""Plain PyTorch harness. Upstream model, diffusion, losses and dataset reused."""
import argparse
import json
import os
import random
import time
from pathlib import Path

from runtime import ROOT, batch_to_model, dataset, exact_load, model, seed_everything
import numpy as np
import torch
from torch.utils.data import DataLoader
from models.utils import CosineWarmupScheduler
from models.losses import InterLoss


def worker_seed(worker_id):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def save_checkpoint(path, payload):
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', type=int, choices=[1, 2], required=True)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--physical-batch', type=int, help='Maximum samples per forward/backward; batch-size remains effective batch.')
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--pretrained', type=Path)
    p.add_argument('--resume', type=Path)
    p.add_argument('--max-steps', type=int, default=0, help='Only for harness verification; zero traverses full epochs.')
    p.add_argument('--log-every', type=int, default=25)
    args = p.parse_args()
    args.physical_batch = args.physical_batch or args.batch_size
    if not 1 <= args.physical_batch <= args.batch_size:
        p.error("physical batch must be between 1 and effective batch")
    run = args.run_dir.resolve()
    run.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('medium')
    device = torch.device('cuda:0')
    data = dataset(args.stage)
    generator = torch.Generator()
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=True, drop_last=False,
                        num_workers=args.workers, pin_memory=True, worker_init_fn=worker_seed,
                        generator=generator, persistent_workers=False)
    net = model(args.stage)
    if args.stage == 2 and not args.resume:
        if not args.pretrained:
            p.error('Stage 2 requires --pretrained with the trained Stage-1 checkpoint.')
        initial = torch.load(args.pretrained, map_location='cpu')
        assert initial['stage'] == 1
        result = net.load_state_dict(initial['model'], strict=False)
        if result.unexpected_keys or any('zero_linear' not in key for key in result.missing_keys):
            raise RuntimeError('Unexpected Stage1 -> Stage2 key mapping: ' + str(result))
        print('Stage1 -> Stage2 initialized:', result, flush=True)
    net.to(device)
    optimizer = torch.optim.AdamW([param for param in net.parameters() if param.requires_grad],
                                 lr=1e-4, weight_decay=2e-5)
    scheduler = CosineWarmupScheduler(optimizer, warmup=10, max_iters=2500 if args.stage == 1 else 1000)
    start_epoch = 0
    resume_batch = 0
    step = 0
    if args.resume:
        state = torch.load(args.resume, map_location='cpu')
        assert state['stage'] == args.stage
        exact_load(net, state)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        start_epoch, resume_batch, step = state['next_epoch'], state['next_batch'], state['step']
        random.setstate(state['rng']['python'])
        np.random.set_state(state['rng']['numpy'])
        torch.set_rng_state(state['rng']['torch'])
        torch.cuda.set_rng_state_all(state['rng']['cuda'])
        print('Resumed at', start_epoch, resume_batch, step, flush=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config.update(dataset_examples=len(data), device=torch.cuda.get_device_name(0),
                  model_parameters=sum(param.numel() for param in net.parameters()),
                  trainable_parameters=sum(param.numel() for param in net.parameters() if param.requires_grad),
                  method='engineering reconstruction', precision='FP32', upstream_model_modified=False)
    (run / 'run_config.json').write_text(json.dumps(config, indent=2) + '\n')
    print(json.dumps(config), flush=True)
    net.train()
    start = time.monotonic()
    window = []
    torch.cuda.reset_peak_memory_stats()
    with (run / 'metrics.jsonl').open('a', buffering=1) as metrics:
        for epoch in range(start_epoch, args.epochs):
            generator.manual_seed(args.seed + epoch)
            epoch_losses = []
            completed = True
            for batch_index, raw in enumerate(loader):
                if epoch == start_epoch and batch_index < resume_batch:
                    continue
                optimizer.zero_grad(set_to_none=True)
                count = len(raw[0])
                accumulated = {}
                # Preserve upstream loss reductions: simple is frame-weighted;
                # geometric/interaction terms average samples with t <= T_BAR.
                # Sampling all timesteps first lets microbatches share the exact
                # full-effective-batch denominators without retaining graphs.
                sampler = net.decoder.sampler
                original_sample = sampler.sample
                split = args.physical_batch < count
                if split:
                    times, weights = original_sample(count, device)
                    active = (times <= net.decoder.cfg.T_BAR).float()
                    frame_total = raw[5].sum().to(device)
                for offset in range(0, count, args.physical_batch):
                    end = min(offset + args.physical_batch, count)
                    piece = [value[offset:end] for value in raw]
                    batch = batch_to_model(piece, device)
                    if split:
                        sampler.sample = lambda n, dev, lo=offset, hi=end: (times[lo:hi], weights[lo:hi])
                    original_mix = InterLoss.mix_masked_mse
                    if split and args.stage == 2:
                        # Upstream RO reduces over the batch axis as well as time.
                        # Preserve that existing behavior, including zero-gate chunks.
                        def accumulated_mix(manager, prediction, target, mask, batch_mask, contact_mask=None, dm_mask=None):
                            if mask.ndim == 3:
                                batch_mask = torch.zeros_like(batch_mask)
                                batch_mask[0] = active.sum()
                            return original_mix(manager, prediction, target, mask, batch_mask, contact_mask, dm_mask)
                        InterLoss.mix_masked_mse = accumulated_mix
                    try:
                        loss, losses = net(batch)
                    finally:
                        sampler.sample = original_sample
                        InterLoss.mix_masked_mse = original_mix
                    if split:
                        frame_weight = (piece[5].sum().to(device) + 1.e-7 / batch['person_num']) / (frame_total + 1.e-7 / batch['person_num'])
                        gate_weight = (active[offset:end].sum() + 1.e-7) / (active.sum() + 1.e-7)
                        ro_weight = (piece[5].sum().to(device) + 1.e-7) / (frame_total + 1.e-7)
                        losses = {key: value * (frame_weight if key == 'simple' else ro_weight if key == 'RO' else gate_weight)
                                  for key, value in losses.items() if key != 'total'}
                        loss = losses['simple'] + losses['A']
                        if args.stage == 2:
                            loss = loss + sum(losses[key] for key in ('RO', 'JA', 'DM'))
                        losses['total'] = loss
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Non-finite loss at step %d' % step)
                    loss.backward()
                    for key, value in losses.items():
                        accumulated[key] = accumulated.get(key, 0.) + float(value.detach())
                    del loss, losses, batch
                grad_norm = torch.nn.utils.clip_grad_norm_(net.parameters(), 0.5, error_if_nonfinite=True)
                optimizer.step()
                step += 1
                scalar = accumulated['total']
                window.append(scalar)
                epoch_losses.append(scalar)
                row = {'step': step, 'epoch': epoch, 'batch': batch_index, 'loss': scalar,
                       'grad_norm': float(grad_norm), 'lr': optimizer.param_groups[0]['lr'],
                       'elapsed_seconds': time.monotonic() - start,
                       'components': accumulated, 'effective_batch': count, 'physical_batch': args.physical_batch}
                metrics.write(json.dumps(row) + '\n')
                if step % args.log_every == 0 or step == 1:
                    print('stage=%d epoch=%d step=%d loss=%.5f avg=%.5f lr=%.3g peak=%.2fGiB elapsed=%.1fs' %
                          (args.stage, epoch, step, scalar, np.mean(window), row['lr'],
                           torch.cuda.max_memory_allocated()/2**30, row['elapsed_seconds']), flush=True)
                    window.clear()
                if args.max_steps and step >= args.max_steps:
                    completed = batch_index == len(loader) - 1
                    break
            if completed:
                scheduler.step()
            checkpoint = {'model': net.state_dict(), 'optimizer': optimizer.state_dict(),
                          'scheduler': scheduler.state_dict(), 'stage': args.stage, 'step': step,
                          'next_epoch': epoch + 1 if completed else epoch,
                          'next_batch': 0 if completed else batch_index + 1,
                          'rng': {'python': random.getstate(), 'numpy': np.random.get_state(),
                                  'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()},
                          'config': config}
            save_checkpoint(run / 'last.pt', checkpoint)
            summary = {'epoch': epoch, 'completed': completed, 'steps': len(epoch_losses),
                       'mean_loss': float(np.mean(epoch_losses)), 'global_step': step}
            with (run / 'epochs.jsonl').open('a') as out:
                out.write(json.dumps(summary) + '\n')
            print('CHECKPOINT', str(run / 'last.pt'), summary, flush=True)
            resume_batch = 0
            if args.max_steps and step >= args.max_steps:
                break


if __name__ == '__main__':
    try:
        main()
    except torch.cuda.OutOfMemoryError:
        import traceback
        traceback.print_exc()
        raise SystemExit(86)
