# OCR-Based vs. End-to-End Transformer Pipelines — GitHub Materials

This folder is a **reader-friendly landing package** for the article:

**OCR-Based vs. End-to-End Transformer Pipelines for Receipt Information Extraction: A Comparative Study on SROIE 2019**

## What is inside

- `01_QUICK_START.md` — 5-minute start for readers
- `02_REPRODUCIBILITY.md` — reproducibility checklist and commands
- `03_RESULTS.md` — reported metrics and interpretation
- `04_PROJECT_STRUCTURE.md` — where each artifact lives in the repo
- `code/` — core source files mirrored for quick access
- `CITATION.cff` — citation metadata for GitHub
- `OCR-Based vs. End-to-End Transformer Pipelines.pdf` — article PDF (copied into this folder)

## Canonical code location

Core implementation is in:
- `invoice_docai/v2/src/docai_utils.py`
- `invoice_docai/v2/notebooks/`
- `invoice_docai/v2/outputs/`

Quick code mirror for readers is in:
- `github_article_materials/code/`

## Recommended reader flow

1. Read this file.
2. Open `01_QUICK_START.md`.
3. If you want to reproduce, follow `02_REPRODUCIBILITY.md`.
4. Cross-check metrics in `03_RESULTS.md`.

## Notes

- Current published numbers are **quick-mode** results (80 validation receipts).
- Some legacy top-level docs in repository may reference older archive states; this folder reflects the current `invoice_docai/v2` state.
