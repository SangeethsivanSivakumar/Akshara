# Akshara (अक्षर)

**LLM-powered OCR for scanned Sanskrit and Devanagari books.**

Akshara uses Google Gemini's vision models to transcribe scanned book PDFs with a multi-pass pipeline that catches the subtle errors traditional OCR misses — wrong visargas, mangled conjuncts, truncated table-of-contents lines. It saves progress after every API call, so interrupted runs resume exactly where they left off.

## Features

- **Multi-pass accuracy pipeline** — Fast OCR with Gemini Flash, then character-by-character verification with Gemini Pro + extended thinking. A third pass auto-triggers when verification changes more than 20% of the text.
- **Page-type-aware prompts** — Detects tables of contents, verse pages, and prose commentary, then uses specialized verification instructions for each.
- **Crash-proof progress** — Every page result is saved to disk immediately. Resume any interrupted run without repeating API calls.
- **Dual output** — Produces both a plain text file and a searchable PDF (scanned images with invisible text overlay).
- **Rate limit handling** — Detects API rate limits, saves all progress, and exits cleanly so you can resume later.

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/SangeethsivanSivakumar/Akshara.git
cd Akshara
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Install poppler (required for PDF-to-image conversion)
brew install poppler        # macOS
# sudo apt install poppler-utils   # Ubuntu/Debian

# 3. Add your Gemini API key
cp .env.example .env
# Edit .env and add your key (free at https://aistudio.google.com/apikey)

# 4. Place PDFs in Source/ and run
mkdir -p Source
python ocr_tool.py
```

The interactive CLI will prompt you to select a PDF, choose a page range, and handle any existing progress.

## How It Works

```
Scanned PDF Page (400 DPI image)
        │
        ▼
┌─────────────────┐
│  Pass 1: OCR    │  gemini-2.5-flash — fast text extraction
│  (Flash)        │
└────────┬────────┘
         │ raw text
         ▼
┌─────────────────┐
│  Pass 2: Verify │  gemini-2.5-pro + thinking budget
│  (Pro)          │  compares text against image, fixes errors
└────────┬────────┘
         │ verified text
         ▼
    change > 20%?
     ╱          ╲
   yes           no
    │             │
    ▼             ▼
┌──────────┐   Done ✓
│ Pass 3:  │
│ Recheck  │   (same model, re-verifies high-change pages)
└────┬─────┘
     │
     ▼
   Done ✓
```

Each page's result is saved to `Output/{book}_progress/` as it completes:
- `page_0001_ocr.txt` → `page_0001_verified.txt` → `page_0001_rechecked.txt`

## Configuration

All tunables are constants at the top of `ocr_tool.py`:

| Constant | Default | Description |
|---|---|---|
| `DPI` | 400 | Resolution for PDF page rendering |
| `GEMINI_MODEL_OCR` | `gemini-2.5-flash` | Model for Pass 1 (OCR) |
| `GEMINI_MODEL_VERIFY` | `gemini-2.5-pro` | Model for Pass 2/3 (verification) |
| `THINKING_BUDGET_VERIFY` | 2048 | Thinking tokens for verification |
| `RECHECK_THRESHOLD` | 0.20 | Change ratio that triggers Pass 3 |
| `MAX_OUTPUT_TOKENS` | 16384 | Max tokens per API response |
| `INTER_CALL_DELAY` | 2.0 | Seconds between API calls |

## Output

After processing, you'll find in the `Output/` directory:

- **`{book}_ocr.txt`** — Combined text of all pages, with page markers
- **`{book}_ocr.pdf`** — Searchable PDF: original scanned images with an invisible text layer for copy/paste and search
- **`{book}_progress/`** — Per-page intermediate files (safe to delete after you're satisfied with the output)

## Requirements

- Python 3.9+
- [Poppler](https://poppler.freedesktop.org/) (`pdfinfo` and `pdftoppm` must be on PATH)
- A [Google Gemini API key](https://aistudio.google.com/apikey) (free tier works)
- A Unicode font with Devanagari support (for the searchable PDF text layer — Arial Unicode MS on macOS works automatically)

## Common OCR Challenges

The verification pass is specifically tuned to catch these recurring issues in Sanskrit texts:

- Missing or wrong **visargas** (ः), **anusvaras** (ं), **chandrabindus** (ँ)
- Confused conjunct characters (e.g., ग्र vs ग्न, ट vs ठ)
- Truncated lines at the end of table-of-contents pages
- Garbled compound words (samasa) where characters merge in the scan

## License

[MIT](LICENSE)
