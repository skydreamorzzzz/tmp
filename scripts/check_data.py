#!/usr/bin/env python3
"""Check received InterHuman files without changing the official splits."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

import numpy as np


def inspect_motion(path):
    try:
        with path.open('rb') as handle:
            version = np.lib.format.read_magic(handle)
            shape, fortran, dtype = np.lib.format._read_array_header(handle, version)
            expected_bytes = handle.tell() + int(np.prod(shape)) * dtype.itemsize
        if dtype.hasobject or len(shape) != 2 or shape[1] != 492:
            raise ValueError('expected numeric (frames, 492), got %s %s' % (shape, dtype))
        if path.stat().st_size != expected_bytes:
            raise ValueError('file length differs from NPY header')
        return {'id': path.stem, 'frames': shape[0], 'dtype': str(dtype)}
    except Exception as error:
        return {'id': path.stem, 'error': str(error), 'path': str(path)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    base = args.data_root.resolve()
    directories = {
        'annots': base / 'annots',
        'person1': base / 'motions_processed/person1',
        'person2': base / 'motions_processed/person2',
        'text1': base / 'separate_annots/text1',
        'text2': base / 'separate_annots/text2',
        'motions': base / 'motions',
    }
    suffixes = {name: '.npy' if name.startswith('person') else '.pkl' if name == 'motions' else '.txt'
                for name in directories}
    files = {name: sorted(folder.glob('*' + suffixes[name])) for name, folder in directories.items()}
    ids = {name: {path.stem for path in paths} for name, paths in files.items()}
    required = ('annots', 'person1', 'person2', 'text1', 'text2')
    complete = set.intersection(*(ids[name] for name in required))
    report = {'data_root': str(base), 'file_counts': {name: len(paths) for name, paths in files.items()},
              'logical_bytes': {name: sum(path.stat().st_size for path in paths) for name, paths in files.items()},
              'official_ignore_list_present': (base / 'split/ignore_list.txt').is_file(),
              'splits': {}, 'overlaps': {}, 'motion_header_errors': [], 'motion_pair_errors': [],
              'annotation_errors': [], 'sample_checks': []}
    split_ids = {}
    sort_ids = lambda values: sorted(values, key=lambda item: (not item.isdigit(), int(item) if item.isdigit() else item))
    for split in ('train', 'val', 'test'):
        entries = (base / 'split' / (split + '.txt')).read_text().splitlines()
        entries = [value.strip() for value in entries if value.strip()]
        split_ids[split] = set(entries)
        report['splits'][split] = {
            'rows': len(entries), 'unique_ids': len(set(entries)),
            'duplicate_ids': sort_ids(key for key, count in Counter(entries).items() if count > 1),
            'complete_ids': len(set(entries) & complete),
            'missing': {name: sort_ids(set(entries) - ids[name]) for name in required},
            'sample_ids': sort_ids(set(entries) & complete)[:3],
        }
    for left, right in (('train', 'val'), ('train', 'test'), ('val', 'test')):
        report['overlaps'][left + '/' + right] = sort_ids(split_ids[left] & split_ids[right])
    with ThreadPoolExecutor(max_workers=8) as pool:
        inspected = {person: list(pool.map(inspect_motion, files[person])) for person in ('person1', 'person2')}
    maps = {person: {entry['id']: entry for entry in entries} for person, entries in inspected.items()}
    report['motion_header_errors'] = [entry for entries in inspected.values() for entry in entries if 'error' in entry]
    for key in ids['person1'] & ids['person2']:
        first, second = maps['person1'][key], maps['person2'][key]
        if 'error' not in first and 'error' not in second and first['frames'] != second['frames']:
            report['motion_pair_errors'].append({'id': key, 'frames': [first['frames'], second['frames']]})

    def inspect_text(key):
        try:
            values = {name: (directories[name] / (key + '.txt')).read_text(encoding='utf-8-sig').splitlines()
                      for name in ('annots', 'text1', 'text2')}
            lengths = {name: len(lines) for name, lines in values.items()}
            if len(set(lengths.values())) != 1 or lengths['annots'] == 0 or any(not line.strip() for lines in values.values() for line in lines):
                return {'id': key, 'line_counts': lengths, 'error': 'empty or misaligned annotations'}
        except Exception as error:
            return {'id': key, 'error': str(error)}
        return None

    with ThreadPoolExecutor(max_workers=8) as pool:
        report['annotation_errors'] = [entry for entry in pool.map(inspect_text, sort_ids(complete)) if entry]
    for split, details in report['splits'].items():
        for key in details['sample_ids']:
            for person in ('person1', 'person2'):
                motion = np.load(str(directories[person] / (key + '.npy')), allow_pickle=False)
                report['sample_checks'].append({'split': split, 'id': key, 'person': person,
                                               'shape': list(motion.shape), 'finite': bool(np.isfinite(motion).all())})
    report['validation_scope'] = 'All NPY headers/file lengths and complete annotation triples; finite values sampled, no author checksums available.'
    report['ready_for_unmodified_official_loader'] = (
        not any(values for details in report['splits'].values() for values in details['missing'].values())
        and not report['motion_header_errors'] and not report['motion_pair_errors'] and not report['annotation_errors']
        and not any(report['overlaps'].values()) and all(entry['finite'] for entry in report['sample_checks']))
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps({key: report[key] for key in ('file_counts', 'splits', 'ready_for_unmodified_official_loader')}, indent=2))
    print('Header errors:', len(report['motion_header_errors']), 'Pair errors:', len(report['motion_pair_errors']),
          'Annotation errors:', len(report['annotation_errors']))
    return 0 if report['ready_for_unmodified_official_loader'] else 2


if __name__ == '__main__':
    sys.exit(main())
