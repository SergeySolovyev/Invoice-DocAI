# Reproducibility Guide

## Environment

Primary requirements file:
- `invoice_docai/v2/requirements.txt`

Install:

```bash
pip install -r invoice_docai/v2/requirements.txt
```

## Data

Project uses SROIE 2019 dataset:
- Train: 626 receipts
- Validation/Test: 347 receipts

Expected paths:
- `invoice_docai/data/sroie/raw/SROIE2019/train/...`
- `invoice_docai/data/sroie/raw/SROIE2019/test/...`
- `invoice_docai/data/sroie/processed/manifest_train.csv`
- `invoice_docai/data/sroie/processed/manifest_val.csv`

## Execution options

### Option A — open saved executed artifacts

Use already generated files in:
- `invoice_docai/v2/outputs/`

### Option B — run notebooks

Run in order:
1. `01_prepare_sroie.ipynb`
2. `02_baseline_ocr.ipynb`
3. `03_donut_inference.ipynb`
4. `03b_donut_finetune.ipynb` (optional, GPU-intensive)
5. `04_robustness_eval.ipynb`
6. `05_summary.ipynb`
7. `06_deep_error_analysis.ipynb`
8. `07_literature_and_why.ipynb`

### Option C — single pipeline notebook

Run:
- `invoice_docai/v2/RUN_ALL_COLAB.ipynb`

## Important reproducibility scope

Published metrics in this repository correspond to **quick mode**:
- training subset: 240 docs
- validation subset: 80 docs

This is intentional and documented in article/report for constrained GPU runtime.
