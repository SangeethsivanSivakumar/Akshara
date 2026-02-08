# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Sanskrit OCR Tool — extracts text from scanned Sanskrit/Devanagari book PDFs using Google Gemini vision models. Single-file Python application (`ocr_tool.py`).

## Setup & Running

```bash
# Environment (Python 3.9, venv)
source venv/bin/activate
pip install -r requirements.txt

# API key in .env (see .env.example)
# Requires poppler for pdf2image: brew install poppler

# Run OCR
python ocr_tool.py

# Translation-only mode (for already-OCR'd books)
python ocr_tool.py --translate
```

Interactive CLI — prompts for PDF selection, page range, and resume options. Uses `argparse` for CLI flags.

## Architecture

All logic is in `ocr_tool.py` (~1100 lines). The pipeline processes one page at a time:

**Pass 1 (OCR):** `gemini-2.5-flash` extracts text from a 400 DPI page image. Fast and cheap.

**Pass 2 (Verify):** `gemini-2.5-pro` with `ThinkingConfig(thinking_budget=2048)` compares the OCR text against the original image and corrects errors character-by-character. Uses page-type-specific prompts (TOC/verse/prose detected by `detect_page_type()`).

**Pass 3 (Recheck):** Auto-triggered when verify changes >20% of the OCR text (`compute_change_ratio()` using `difflib.SequenceMatcher`). Runs the verify prompt again on the already-verified text.

**Pass 4 (Translate):** `gemini-2.5-pro` with thinking translates Sanskrit to English. Uses a sliding window of the previous 3 pages as context for terminology continuity. Prefers already-translated English context, falls back to Sanskrit source.

**Pass 5 (Verify Translation):** `gemini-2.5-pro` compares the English translation against the Sanskrit source and corrects errors.

**Progress tracking** is file-based and crash-proof:
- `Output/{stem}_progress/page_NNNN_ocr.txt` → `_verified.txt` → `_rechecked.txt` → `_translated.txt` → `_translation_verified.txt`
- Stage is determined purely from which files exist (`get_page_stage()`)
- `progress.json` is a convenience summary; the source of truth is the per-page files
- On resume, the work list builder detects interrupted rechecks by comparing OCR vs verified text

**Output:** Combined `.txt` file + searchable PDF (scanned images with invisible text overlay via reportlab). Translation produces a separate `_translation.txt` and `_translation.pdf` (readable English text, US Letter, Helvetica 11pt).

## Key Technical Details

- `_call_gemini()` is the single API entry point — handles rate limits (`RateLimitError` → immediate stop and save), server errors (retry with backoff), safety filters, and token truncation
- Rate limit detection: check for "429" or "RESOURCE_EXHAUSTED" in exception string
- `normalize_whitespace()` collapses runs of 7+ spaces to 6 and 7+ dots to 4 — essential because Gemini OCR sometimes generates hundreds of spaces for columnar layouts
- `ThinkingConfig` uses `thinking_budget` param (not `thinking_level`) for Gemini 2.5 models
- Constants at top of file control all tunables: `DPI`, `MAX_OUTPUT_TOKENS`, model names, `RECHECK_THRESHOLD`, `THINKING_BUDGET_VERIFY`, `THINKING_BUDGET_TRANSLATE`, `CONTEXT_WINDOW_PAGES`, `INTER_CALL_DELAY`
- `_start_continuation_page()` is a shared helper used by `create_text_pdf()` for page overflow handling
- `select_pdf()` always returns a `Path` or exits — callers don't need to check for `None`

## Common Devanagari OCR Errors

The verify pass catches these recurring issues:
- Missing/wrong visargas (ः), anusvaras (ं), chandrabindus (ँ)
- Similar-looking conjunct characters (e.g., ग्र vs ग्न, ट vs ठ)
- Truncated lines at end of TOC pages
- Garbled compound words (samāsa) where characters merge
