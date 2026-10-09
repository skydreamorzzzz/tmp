# InterGen vs FreeMotion artifacts

No training. Native InterGen checkout: `code/InterGen-official`, commit `b1750df4a859443b8a69329f759ce3a7503ae179`.

- `test_manifest.json`: frozen original test prompts, per-person prompts, actor order, crop and valid lengths.
- `ground_truth/`: official preprocessed physical test motions.
- `native_two_person/`: native joint generation `[T,2,262]`.
- `single/`: two independent zero-B A-only single generations, paired as `[T,2,262]`; neither is a generated B stream.
- `*_mm/`: every one of30 sampled repeats, `[30,T,2,262]`, for100 fixed test cases.
- `single_features.npz`, `native_two_person_features.npz`: paired512-d text/motion features, GT features and IDs in evaluator length-sorted order.
- `*_mm_features.npz`: motion features `[100,30,512]`, case IDs and repeat ordering for Multimodality.
- `*_metrics.json`: official functions, protocol metadata; `*_sanity_*` files are small sanity runs, not formal metrics.
- `single_path.py`: minimal user-authorized DPMotion zero-B path, adds no parameters and preserves native checkpoint keys.
- `weight_audit.json`: historical FreeMotion mapping facts; no semantic remapping.
- `visualizations/`: fixed test1/4/6 GIFs and contact sheets in physical coordinates, no smoothing or actor shifting.

Run from `/home/featurize/work/FreeMotion-reproduction`:

```sh
.venv/bin/python artifacts/intergen_vs_freemotion/prepare_test.py
.venv/bin/python artifacts/intergen_vs_freemotion/generate_native.py --batch-size 8 --mm-count 100 --mm-repeats 30
.venv/bin/python artifacts/intergen_vs_freemotion/generate_single.py --batch-size 8 --mm-count 100 --mm-repeats 30
.venv/bin/python artifacts/intergen_vs_freemotion/evaluate_saved.py --mode native_two_person
.venv/bin/python artifacts/intergen_vs_freemotion/evaluate_saved.py --mode single
.venv/bin/python artifacts/intergen_vs_freemotion/visualize.py --mode native_two_person
.venv/bin/python artifacts/intergen_vs_freemotion/visualize.py --mode single
.venv/bin/python artifacts/intergen_vs_freemotion/verify_outputs.py
.venv/bin/python artifacts/intergen_vs_freemotion/build_report.py
```

Generation reuses complete files at batch boundaries. Use an empty output folder for clean regeneration; numerical reproducibility depends on the recorded hardware/software and generation batch size. Saved `.npy` files permit evaluation without generation.

The run evaluated single primary outputs with `--skip-mm` while repeat sampling continued, then used `--mm-only` to fill Multimodality without redoing primary features. Running the normal evaluation command above computes both from completed files. `artifact_validation.json` records output coverage and finite-value checks. `paper_protocol.json` records paper tables, confidence intervals and official metric definitions. The comparison is a reference comparison across recorded protocol differences, not a strict performance ranking.
