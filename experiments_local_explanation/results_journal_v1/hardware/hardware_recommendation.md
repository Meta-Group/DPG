# Journal Hardware Recommendation

- Logical CPUs available to Python: 36
- Load average: 18.21, 18.90, 18.74
- Available memory: 46.3 GiB
- Recommended DPG workers: 12
- Recommended baseline workers: 6
- Recommended `--rf_n_jobs`: 1
- Set native thread environment variables before launching multi-process runs:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
```

Use fewer workers for SHAP/LIME/Anchors than for DPG-only runs because those baselines have higher per-process memory and CPU variability.
