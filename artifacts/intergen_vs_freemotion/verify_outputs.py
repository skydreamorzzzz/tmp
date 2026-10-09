"""Check all saved artifacts and the completed comparison, without generation."""
from pathlib import Path
import json
import hashlib
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]

def main():
    manifest=json.loads((OUT/'test_manifest.json').read_text())
    rows={r['id']:r for r in manifest['rows']}
    selected=json.loads((OUT/'mm_manifest.json').read_text())
    report={}
    for mode in ['native_two_person','single']:
        files=list((OUT/mode).glob('*.npy'))
        assert {f.stem for f in files}==set(rows)
        max_abs=0.;bytes_saved=0
        for f in files:
            a=np.load(f);assert a.shape==(rows[f.stem]['length'],2,262)
            assert a.dtype==np.float32 and np.isfinite(a).all()
            max_abs=max(max_abs,float(np.abs(a).max()));bytes_saved+=f.stat().st_size
        mmfiles=list((OUT/(mode+'_mm')).glob('*.npy'))
        assert {f.stem for f in mmfiles}==set(selected['ids'])
        minimum_variation=float('inf')
        for f in mmfiles:
            a=np.load(f);assert a.shape==(30,rows[f.stem]['length'],2,262)
            assert a.dtype==np.float32 and np.isfinite(a).all()
            minimum_variation=min(minimum_variation,float(a[:,:,:,:66].std(axis=0).mean()))
            bytes_saved+=f.stat().st_size
        assert minimum_variation>0
        features=np.load(OUT/(mode+'_features.npz'))
        for k in ['text','motion','ground_truth']:
            assert features[k].shape==(1056,512) and np.isfinite(features[k]).all()
        mmfeatures=np.load(OUT/(mode+'_mm_features.npz'))
        assert mmfeatures['motion'].shape==(100,30,512) and np.isfinite(mmfeatures['motion']).all()
        metrics=json.loads((OUT/(mode+'_metrics.json')).read_text())
        assert all(metrics['metrics'][k] is not None for k in ['FID','R_Precision','MM_Distance','Diversity','Multimodality'])
        report[mode]={'primary_files':len(files),'primary_motion_sequences':len(files)*2,
                      'mm_files':len(mmfiles),'mm_motion_sequences':len(mmfiles)*30*2,
                      'all_finite':True,'primary_max_abs_all_features':max_abs,
                      'minimum_mm_position_variation':minimum_variation,'motion_bytes':bytes_saved,
                      'evaluation_features':list(features['motion'].shape),'mm_features':list(mmfeatures['motion'].shape)}
    a=np.load(OUT/'native_two_person_features.npz');b=np.load(OUT/'single_features.npz')
    assert np.array_equal(a['ids'],b['ids'])
    assert np.array_equal(a['ground_truth'],b['ground_truth'])
    assert np.array_equal(a['text'],b['text'])
    report['comparison_alignment']={'identical_test_ids':True,'identical_ground_truth_features':True,
                                     'identical_text_features':True,'native_vs_single_motion_features_different':not np.array_equal(a['motion'],b['motion'])}
    assert report['comparison_alignment']['native_vs_single_motion_features_different']
    native=json.loads((OUT/'native_load.json').read_text())
    audit=json.loads((OUT/'weight_audit.json').read_text())
    single=json.loads((OUT/'single_load.json').read_text())
    for name,expected in native['source_hashes'].items():
        assert hashlib.sha256((ROOT/'code/InterGen-official'/name).read_bytes()).hexdigest()==expected
    for name,expected in audit['official_source_hashes'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected
    assert hashlib.sha256((OUT/'single_path.py').read_bytes()).hexdigest()==single['port_sha256']
    report['implementation_integrity']={'native_source_unchanged':True,'freemotion_source_unchanged':True,
                                      'single_port_matches_generation_hash':True}
    (OUT/'artifact_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
