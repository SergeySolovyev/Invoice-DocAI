# Reported Results (Quick Mode)

Source: `invoice_docai/v2/outputs/results_summary_quick.csv`

## Main table

| Pipeline | Condition | Vendor F1 | Date F1 | Total F1 | Micro F1 |
|---|---:|---:|---:|---:|---:|
| OCR | clean | 0.4906 | 0.7846 | 0.6325 | 0.6459 |
| OCR | corrupted | 0.4000 | 0.5636 | 0.4466 | 0.4728 |
| Donut-PT | clean | 0.0000 | 0.0494 | 0.0000 | 0.0166 |
| Donut-FT | clean | 0.8235 | 0.6261 | 0.7786 | 0.7487 |
| Donut-FT | corrupted | 0.6885 | 0.5370 | 0.6441 | 0.6264 |

## Robustness degradation

Source: `invoice_docai/v2/outputs/robustness_results_quick.csv`

- OCR average F1 drop is larger than Donut-FT.
- Donut-FT remains stronger on Vendor and Total under corruption.
- OCR remains stronger on Date in clean setting.

## Interpretation

- **Best overall**: Donut-FT (highest micro F1 on clean).
- **Best Date (clean)**: OCR baseline due to regex-friendly field structure.
- **Best robustness**: Donut-FT, smaller degradation under messenger-grade corruption.

For deeper breakdown, see:
- `invoice_docai/v2/notebooks/06_deep_error_analysis_executed.ipynb`
- `invoice_docai/v2/notebooks/07_literature_and_why_executed.ipynb`
