# Project Structure Map

## Root-level

- `Invoice_DocAI_Preprint/paper.tex` — article LaTeX source
- `OCR-Based vs. End-to-End Transformer Pipelines.pdf` — article PDF
- `invoice_docai/` — project implementation and data artifacts

## Data

- `invoice_docai/data/sroie/raw/SROIE2019/` — raw dataset hierarchy
- `invoice_docai/data/sroie/processed/` — generated manifests (CSV/JSONL)

## Core code

- `invoice_docai/v2/src/docai_utils.py` — normalization, extraction, corruption, metrics

## Notebooks

- `invoice_docai/v2/notebooks/01_prepare_sroie.ipynb`
- `invoice_docai/v2/notebooks/02_baseline_ocr.ipynb`
- `invoice_docai/v2/notebooks/03_donut_inference.ipynb`
- `invoice_docai/v2/notebooks/03b_donut_finetune.ipynb`
- `invoice_docai/v2/notebooks/04_robustness_eval.ipynb`
- `invoice_docai/v2/notebooks/05_summary.ipynb`
- `invoice_docai/v2/notebooks/06_deep_error_analysis(.ipynb / _executed.ipynb)`
- `invoice_docai/v2/notebooks/07_literature_and_why(.ipynb / _executed.ipynb)`

## Outputs

- `invoice_docai/v2/outputs/results_summary_quick.csv`
- `invoice_docai/v2/outputs/robustness_results_quick.csv`
- multiple prediction CSV files and publication figures

## One-file full run

- `invoice_docai/v2/RUN_ALL_COLAB.ipynb`
