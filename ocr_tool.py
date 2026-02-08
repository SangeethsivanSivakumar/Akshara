#!/usr/bin/env python3
"""
Sanskrit OCR Tool — LLM-powered OCR for scanned Sanskrit/Devanagari books.

Multi-pass pipeline for maximum accuracy:
  Pass 1 (OCR):     Extract text from the scanned page image (Flash).
  Pass 2 (Verify):  Compare OCR text against the image and correct errors (Pro + thinking).
  Pass 3 (Recheck): Auto-triggered when verify changes >20% of the OCR text.

Progress is saved after every step so interrupted runs resume exactly
where they left off — no wasted API calls.
"""

from __future__ import annotations

import argparse
import difflib
import json
import logging
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path
from subprocess import run as subprocess_run

from dotenv import load_dotenv
from google import genai
from google.genai import types
from pdf2image import convert_from_path
from PIL import Image
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SOURCE_DIR = Path(__file__).parent / "Source"
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "Output"
DPI = 400
MAX_RETRIES = 3
MAX_OUTPUT_TOKENS = 16384
INTER_CALL_DELAY = 2.0  # Seconds between API calls

# Models — use Flash for fast OCR, Pro for accurate verification
GEMINI_MODEL_OCR = "gemini-2.5-flash"
GEMINI_MODEL_VERIFY = "gemini-2.5-pro"

# Thinking budget (tokens) for the verify/recheck pass — lets the model
# reason about ambiguous characters before committing to a correction.
THINKING_BUDGET_VERIFY = 2048

# If the verify pass changes more than this fraction of the OCR text,
# automatically trigger a third "recheck" pass.
RECHECK_THRESHOLD = 0.20

# Quality gate: normal Sanskrit book pages produce 500–2000 chars.
# Anything above this after normalization is suspicious hallucination.
MAX_REASONABLE_CHARS = 8000

# Translation — Pro model with thinking for context-aware Sanskrit→English
GEMINI_MODEL_TRANSLATE = "gemini-2.5-pro"
THINKING_BUDGET_TRANSLATE = 2048
CONTEXT_WINDOW_PAGES = 3  # Include previous N pages as context

# Page processing stages
STAGE_PENDING = "pending"
STAGE_OCR = "ocr"
STAGE_VERIFIED = "verified"
STAGE_RECHECKED = "rechecked"
STAGE_TRANSLATED = "translated"
STAGE_TRANSLATION_VERIFIED = "translation_verified"

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

OCR_SYSTEM_INSTRUCTION = (
    "You are an expert OCR system specialising in Sanskrit (Devanagari script) "
    "and English text. You transcribe scanned book pages with perfect fidelity, "
    "preserving every character, punctuation mark, and the visual layout of the page."
)

OCR_PROMPT = """\
Transcribe ALL text visible on this scanned book page exactly as printed.

Rules:
1. Transcribe every character faithfully — do NOT translate, interpret, or correct.
2. Preserve the original script: Devanagari stays Devanagari, English stays English.
3. Reproduce the line-by-line layout of the page: keep the same line breaks the
   printed text has. Each printed line becomes one line in your output.
4. For columnar or tabular content (e.g. tables of contents), align columns using
   a small number of spaces (2-6). Do NOT pad with hundreds of spaces or dots.
5. Include page numbers, headers, footers, and marginal notes.
6. If a character is unclear, give your best reading — never omit text.
7. Output ONLY the transcribed text. No commentary, labels, or markdown formatting.
8. Preserve punctuation exactly: dandas |, double dandas ||, periods, commas, etc.
9. If the page is entirely blank, output exactly: [BLANK PAGE]"""

VERIFY_SYSTEM_INSTRUCTION = (
    "You are a meticulous proofreader specialising in Sanskrit (Devanagari) and "
    "English text from old printed books. You compare an OCR transcription against "
    "the original scanned page image and correct every error you find."
)

PAGE_TYPE_HINTS = {
    "toc": (
        "This page is a TABLE OF CONTENTS with columnar layout. Pay extra attention to:\n"
        "   - Column alignment: subject titles, verse/section numbers, page numbers\n"
        "   - Ensure ALL rows are present — OCR often truncates the last several lines\n"
        "   - Preserve ditto marks (,, or „) and dot leaders (....)\n"
    ),
    "verse": (
        "This page contains SANSKRIT VERSE with commentary. Pay extra attention to:\n"
        "   - Double dandas (॥) marking verse boundaries\n"
        "   - Verse numbering (e.g., ॥ ३ ॥)\n"
        "   - Distinction between mūla (root verse) and ṭīkā (commentary)\n"
    ),
    "prose": (
        "This page contains PROSE COMMENTARY. Pay extra attention to:\n"
        "   - Long compound words (samāsa) — ensure no characters are dropped\n"
        "   - Quotation markers and reference citations\n"
    ),
    "blank": "",
}

VERIFY_PROMPT_TEMPLATE = """\
Below is an OCR transcription of the scanned page shown in the image.
Compare the transcription against the original image line by line and fix
every error you find.

{page_type_hint}
=== OCR TRANSCRIPTION ===
{ocr_text}
=== END TRANSCRIPTION ===

Instructions:
1. Go through the page image line by line. For each line, compare the OCR text
   against what is actually printed. Fix any wrong, missing, or extra characters.
2. Pay special attention to:
   - Devanagari conjunct characters and similar-looking aksharas
   - Visargas (ः), anusvaras (ं), chandrabindus (ँ), and nuktas
   - Punctuation: dandas (।), double dandas (॥), periods, commas
   - Digits and page numbers
   - Words at the start and end of each line (OCR often truncates these)
3. Preserve the line-by-line layout exactly as it appears on the printed page.
4. For tables / columnar content, use 2-6 spaces between columns. No excessive
   whitespace padding.
5. Output ONLY the corrected transcription. No commentary or notes.
6. If the transcription is already correct, output it unchanged."""

# ---------------------------------------------------------------------------
# Translation prompts
# ---------------------------------------------------------------------------

TRANSLATE_SYSTEM_INSTRUCTION = (
    "You are an expert Sanskrit-to-English translator specialising in classical "
    "Sanskrit literature, Jyotish Shastra (Vedic astrology), and Indian philosophy. "
    "You produce scholarly, readable English translations that preserve the meaning, "
    "structure, and nuance of the original Sanskrit text."
)

TRANSLATE_PROMPT_TEMPLATE = """\
Translate the Sanskrit text below into clear, scholarly English.

{context_section}
=== CURRENT PAGE (Page {page_num}) ===
{sanskrit_text}
=== END CURRENT PAGE ===

Instructions:
1. Translate the CURRENT PAGE text into fluent, readable English.
2. Use the previous pages as context to maintain continuity of subjects,
   terminology, and narrative flow. Do NOT re-translate the context pages.
3. Preserve the structure: verse numbers, section headings, formatting.
4. For technical Sanskrit terms (philosophical, grammatical, astrological),
   provide the Sanskrit term in parentheses on first occurrence in each page.
5. For verses (śloka), maintain verse structure with line breaks.
6. For commentary (ṭīkā/bhāṣya), use clear paragraph breaks.
7. Translate quotations from other texts as quotes, citing the author if named.
8. Output ONLY the English translation. No commentary about the translation process.
9. If the page is marked [BLANK PAGE], output exactly: [BLANK PAGE]"""

CONTEXT_SECTION_TEMPLATE = """\
=== PREVIOUS PAGES (for context only — do NOT re-translate) ===
{context_text}
=== END PREVIOUS PAGES ===

"""

# ---------------------------------------------------------------------------
# Translation verification prompts
# ---------------------------------------------------------------------------

VERIFY_TRANSLATION_SYSTEM_INSTRUCTION = (
    "You are an expert Sanskrit-to-English translation reviewer specialising in "
    "classical Sanskrit literature, Jyotish Shastra (Vedic astrology), and Indian "
    "philosophy. You verify translations for accuracy, completeness, and scholarly quality."
)

VERIFY_TRANSLATION_PROMPT_TEMPLATE = """\
Below is a Sanskrit source text and its English translation. Compare the translation
against the original Sanskrit and correct any errors.

=== SANSKRIT SOURCE (Page {page_num}) ===
{sanskrit_text}
=== END SANSKRIT SOURCE ===

=== ENGLISH TRANSLATION ===
{english_text}
=== END ENGLISH TRANSLATION ===

Instructions:
1. Compare each sentence/verse of the English translation against the Sanskrit original.
2. Fix any errors in meaning, missing passages, or mistranslations.
3. Ensure technical Sanskrit terms are transliterated correctly and provided in
   parentheses where appropriate (e.g., navāṃśa, daśā, graha).
4. Verify verse numbers, section headings, and structural formatting are preserved.
5. Ensure consistency of terminology (same Sanskrit term → same English rendering).
6. Maintain the scholarly, readable tone of the original translation.
7. If the translation is already correct, output it unchanged.
8. Output ONLY the corrected translation. No commentary about the review process.
"""

console = Console()
log = logging.getLogger("akshara")


def setup_logging() -> Path:
    """Configure file logging. Returns the log file path."""
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = DEFAULT_OUTPUT_DIR / "akshara.log"

    # Avoid duplicate handlers if called more than once
    resolved = str(log_path.resolve())
    for h in log.handlers[:]:
        if isinstance(h, logging.FileHandler) and h.baseFilename == resolved:
            break
    else:
        handler = logging.FileHandler(log_path, encoding="utf-8")
        handler.setFormatter(logging.Formatter(
            "%(asctime)s  %(levelname)-5s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        log.addHandler(handler)

    log.setLevel(logging.DEBUG)

    log.info("=" * 60)
    log.info("Session started")
    return log_path


class RateLimitError(Exception):
    """Raised when API rate limit is hit so we can save progress and exit."""
    pass


# ---------------------------------------------------------------------------
# Font Setup
# ---------------------------------------------------------------------------

_registered_font_name = None


def register_devanagari_font() -> str:
    global _registered_font_name
    if _registered_font_name:
        return _registered_font_name

    for font_path in [
        "/Library/Fonts/Arial Unicode.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
    ]:
        if Path(font_path).exists():
            try:
                pdfmetrics.registerFont(TTFont("UnicodeFont", font_path))
                _registered_font_name = "UnicodeFont"
                return _registered_font_name
            except Exception:
                continue

    console.print(
        "  [yellow]Warning: No Devanagari TTF font found. "
        "PDF text layer may lack Devanagari characters.[/]"
    )
    _registered_font_name = "Helvetica"
    return _registered_font_name


# ---------------------------------------------------------------------------
# Progress tracking  (file-based: robust against crashes)
# ---------------------------------------------------------------------------


def get_progress_dir(pdf_path: Path) -> Path:
    return DEFAULT_OUTPUT_DIR / f"{pdf_path.stem}_progress"


def _page_file(pdf_path: Path, page_num: int, stage: str) -> Path:
    suffix_map = {
        STAGE_OCR: "_ocr",
        STAGE_VERIFIED: "_verified",
        STAGE_RECHECKED: "_rechecked",
        STAGE_TRANSLATED: "_translated",
        STAGE_TRANSLATION_VERIFIED: "_translation_verified",
    }
    suffix = suffix_map.get(stage, "_verified")
    return get_progress_dir(pdf_path) / f"page_{page_num:04d}{suffix}.txt"


def get_page_stage(pdf_path: Path, page_num: int) -> str:
    """Determine how far a page has been processed based on saved files."""
    if _page_file(pdf_path, page_num, STAGE_TRANSLATION_VERIFIED).exists():
        return STAGE_TRANSLATION_VERIFIED
    if _page_file(pdf_path, page_num, STAGE_TRANSLATED).exists():
        return STAGE_TRANSLATED
    if _page_file(pdf_path, page_num, STAGE_RECHECKED).exists():
        return STAGE_RECHECKED
    if _page_file(pdf_path, page_num, STAGE_VERIFIED).exists():
        return STAGE_VERIFIED
    if _page_file(pdf_path, page_num, STAGE_OCR).exists():
        return STAGE_OCR
    return STAGE_PENDING


def save_page_text(pdf_path: Path, page_num: int, text: str, stage: str):
    progress_dir = get_progress_dir(pdf_path)
    progress_dir.mkdir(parents=True, exist_ok=True)
    _page_file(pdf_path, page_num, stage).write_text(text, encoding="utf-8")


def load_page_text(pdf_path: Path, page_num: int, stage: str) -> str | None:
    f = _page_file(pdf_path, page_num, stage)
    if not f.exists():
        return None
    return f.read_text(encoding="utf-8")


def load_best_text(pdf_path: Path, page_num: int) -> str | None:
    """Load the best available text: rechecked > verified > OCR."""
    for s in (STAGE_RECHECKED, STAGE_VERIFIED, STAGE_OCR):
        t = load_page_text(pdf_path, page_num, s)
        if t is not None:
            return t
    return None


def is_page_error(text: str) -> bool:
    return text.startswith("[OCR ERROR") or text.startswith("[BLOCKED") or text.startswith("[GARBAGE")


def save_progress_json(pdf_path: Path, first_page: int, last_page: int):
    """Write a lightweight summary file (the real state lives in per-page files)."""
    progress_dir = get_progress_dir(pdf_path)
    progress_dir.mkdir(parents=True, exist_ok=True)

    all_pages = range(first_page, last_page + 1)
    rechecked = []
    verified = []
    ocr_only = []
    errors = []
    pending = []

    for p in all_pages:
        stage = get_page_stage(pdf_path, p)
        if stage == STAGE_RECHECKED:
            text = load_page_text(pdf_path, p, STAGE_RECHECKED)
            if text and is_page_error(text):
                errors.append(p)
            else:
                rechecked.append(p)
        elif stage == STAGE_VERIFIED:
            text = load_page_text(pdf_path, p, STAGE_VERIFIED)
            if text and is_page_error(text):
                errors.append(p)
            else:
                verified.append(p)
        elif stage == STAGE_OCR:
            text = load_page_text(pdf_path, p, STAGE_OCR)
            if text and is_page_error(text):
                errors.append(p)
            else:
                ocr_only.append(p)
        else:
            pending.append(p)

    data = {
        "pdf_name": pdf_path.name,
        "pdf_path": str(pdf_path),
        "first_page": first_page,
        "last_page": last_page,
        "rechecked_pages": rechecked,
        "verified_pages": verified,
        "ocr_only_pages": ocr_only,
        "error_pages": errors,
        "pending_pages": pending,
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (progress_dir / "progress.json").write_text(
        json.dumps(data, indent=2), encoding="utf-8"
    )


def load_progress_json(pdf_path: Path) -> dict | None:
    f = get_progress_dir(pdf_path) / "progress.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


# ---------------------------------------------------------------------------
# Text cleanup
# ---------------------------------------------------------------------------


def normalize_whitespace(text: str) -> str:
    """Collapse excessive repeated characters that OCR sometimes produces
    (e.g. thousands of dashes for a decorative rule), while preserving
    intentional layout (indentation, column gaps)."""
    lines = text.split("\n")
    cleaned = []
    prev_was_separator = False
    for line in lines:
        # Collapse runs of spaces longer than 6 → exactly 6
        line = re.sub(r" {7,}", "      ", line)
        # Collapse runs of dots (·….) longer than 6 → 4 dots
        line = re.sub(r"[·.…]{7,}", "....", line)
        # Collapse runs of dashes/hyphens longer than 6 → 6
        line = re.sub(r"[-–—]{7,}", "------", line)
        # Collapse runs of underscores / equals longer than 6 → 6
        line = re.sub(r"_{7,}", "______", line)
        line = re.sub(r"={7,}", "======", line)
        # Strip trailing whitespace per line
        line = line.rstrip()

        # Collapse consecutive "separator-only" lines (lines that are
        # nothing but dashes, dots, spaces, underscores, equals) into one.
        is_separator = bool(line) and not re.search(r"[^\s.\-–—_=·…]", line)
        if is_separator and prev_was_separator:
            continue
        prev_was_separator = is_separator

        cleaned.append(line)
    return "\n".join(cleaned)


def is_garbage_output(text: str, raw_len: int | None = None) -> tuple[bool, str]:
    """Detect hallucinated or garbage OCR output that would waste downstream API calls.

    Returns (is_garbage, reason). Checks raw size, normalized size,
    separator dominance, and character diversity.
    """
    # Check raw (pre-normalization) size — catches massive hallucinations
    if raw_len is not None and raw_len > MAX_REASONABLE_CHARS * 5:
        return True, f"raw output too large ({raw_len:,} chars, limit {MAX_REASONABLE_CHARS * 5:,})"

    # Check normalized size
    if len(text) > MAX_REASONABLE_CHARS:
        return True, f"normalized output too large ({len(text):,} chars, limit {MAX_REASONABLE_CHARS:,})"

    # Check separator dominance: if >80% of non-whitespace chars are separators,
    # the page is a decorative element, not text
    non_ws = re.sub(r"\s", "", text)
    if len(non_ws) > 50:
        separators = sum(1 for c in non_ws if c in ".-–—_=·…")
        if separators / len(non_ws) > 0.80:
            return True, f"separator dominance ({separators}/{len(non_ws)} = {separators/len(non_ws):.0%} separators)"

    # Check character diversity: if >100 non-ws chars but <1% unique, it's repetitive
    if len(non_ws) > 100:
        unique_ratio = len(set(non_ws)) / len(non_ws)
        if unique_ratio < 0.01:
            return True, f"low character diversity ({len(set(non_ws))} unique in {len(non_ws)} chars = {unique_ratio:.2%})"

    return False, ""


def detect_page_type(text: str) -> str:
    """Classify page content to select a specialised verify prompt."""
    lines = text.strip().split("\n")
    if not lines:
        return "blank"

    # Table of contents: many lines with "…." or "...." or „ and numbers
    dot_lines = sum(1 for l in lines if re.search(r"\.{3,}|…{2,}", l))
    if dot_lines >= 5 or (len(lines) > 3 and dot_lines / len(lines) > 0.4):
        return "toc"

    # Verse: heavy use of double dandas (॥)
    danda_count = text.count("॥")
    if danda_count >= 3:
        return "verse"

    return "prose"


def compute_change_ratio(text_a: str, text_b: str) -> float:
    """Return the fraction of text that changed between two versions (0–1).
    0 = identical, 1 = completely different."""
    if not text_a and not text_b:
        return 0.0
    matcher = difflib.SequenceMatcher(None, text_a, text_b)
    return 1.0 - matcher.ratio()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_api_key() -> str:
    load_dotenv(Path(__file__).parent / ".env")
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        console.print(
            "\n[bold red]Error:[/] No API key found.\n"
            "Set GEMINI_API_KEY in a .env file or as an environment variable.\n"
            "Get a free key at: https://aistudio.google.com/apikey\n"
        )
        sys.exit(1)
    return key


def get_pdf_page_count(pdf_path: Path) -> int | None:
    try:
        result = subprocess_run(
            ["pdfinfo", str(pdf_path)], capture_output=True, text=True
        )
        for line in result.stdout.splitlines():
            if line.startswith("Pages:"):
                return int(line.split(":")[1].strip())
    except Exception:
        pass
    return None


def format_file_size(size_bytes: int) -> str:
    if size_bytes >= 1_048_576:
        return f"{size_bytes / 1_048_576:.1f} MB"
    return f"{size_bytes / 1024:.1f} KB"


def get_pdf_info(pdf_path: Path) -> dict:
    return {
        "path": pdf_path,
        "name": pdf_path.name,
        "size": format_file_size(pdf_path.stat().st_size),
        "pages": get_pdf_page_count(pdf_path),
    }


def select_pdf() -> Path:
    console.print("\n[bold]Select a PDF file to process:[/]\n")

    pdf_files = (
        sorted(DEFAULT_SOURCE_DIR.glob("*.pdf"))
        if DEFAULT_SOURCE_DIR.exists()
        else []
    )

    if pdf_files:
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("#", style="dim", width=4)
        table.add_column("Filename")
        table.add_column("Size", justify="right")

        for i, f in enumerate(pdf_files, 1):
            table.add_row(str(i), f.name, format_file_size(f.stat().st_size))

        console.print(table)
        console.print(
            f"\n  [dim]Enter a number (1-{len(pdf_files)}) or type a full file path[/]"
        )
        choice = Prompt.ask("\n  Selection")

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(pdf_files):
                return pdf_files[idx]
            console.print("[red]Invalid selection.[/]")
            sys.exit(1)
        except ValueError:
            pass

        path = Path(choice).expanduser()
        if path.exists() and path.suffix.lower() == ".pdf":
            return path
        console.print(f"[red]File not found or not a PDF: {choice}[/]")
        sys.exit(1)

    console.print("  [dim]No PDFs found in Source/ directory.[/]")
    choice = Prompt.ask("  Enter full path to a PDF file")
    path = Path(choice).expanduser()
    if path.exists() and path.suffix.lower() == ".pdf":
        return path
    console.print(f"[red]File not found or not a PDF: {choice}[/]")
    sys.exit(1)


def select_page_range(total_pages: int) -> tuple:
    console.print(f"\n[bold]Page range[/] (total: {total_pages} pages)\n")
    console.print("  [1] Process all pages")
    console.print("  [2] Select a page range")

    choice = Prompt.ask("\n  Choice", choices=["1", "2"], default="1")

    if choice == "1":
        return 1, total_pages

    first = IntPrompt.ask("  Start page", default=1)
    last = IntPrompt.ask("  End page", default=total_pages)
    first = max(1, min(first, total_pages))
    last = max(first, min(last, total_pages))
    return first, last


def convert_single_page(pdf_path: Path, page_num: int) -> Image.Image:
    images = convert_from_path(
        str(pdf_path), dpi=DPI, first_page=page_num, last_page=page_num
    )
    return images[0]


# ---------------------------------------------------------------------------
# Gemini API calls
# ---------------------------------------------------------------------------


def _call_gemini(
    client: genai.Client,
    contents: list,
    system_instruction: str,
    page_num: int,
    label: str,
    model: str = GEMINI_MODEL_OCR,
    thinking_budget: int | None = None,
) -> str:
    """Shared API call logic with rate-limit detection and server-error retry."""
    for attempt in range(MAX_RETRIES):
        t0 = time.monotonic()
        try:
            config_kwargs = dict(
                system_instruction=system_instruction,
                temperature=0.0,
                max_output_tokens=MAX_OUTPUT_TOKENS,
            )
            if thinking_budget and thinking_budget > 0:
                config_kwargs["thinking_config"] = types.ThinkingConfig(
                    thinking_budget=thinking_budget
                )

            log.info(
                "API call start: page=%d, label=%s, model=%s, attempt=%d/%d",
                page_num, label, model, attempt + 1, MAX_RETRIES,
            )

            response = client.models.generate_content(
                model=model,
                contents=contents,
                config=types.GenerateContentConfig(**config_kwargs),
            )

            elapsed = time.monotonic() - t0

            if not response.candidates:
                log.warning(
                    "API call empty: page=%d, label=%s, duration=%.1fs — no candidates",
                    page_num, label, elapsed,
                )
                return f"[NO RESPONSE on page {page_num}]"

            candidate = response.candidates[0]
            finish = getattr(candidate, "finish_reason", None)

            if finish and str(finish).upper() == "SAFETY":
                log.warning(
                    "API call blocked: page=%d, label=%s, duration=%.1fs — safety filter",
                    page_num, label, elapsed,
                )
                console.print(
                    f"\n  [yellow]Page {page_num} ({label}): safety filter[/]"
                )
                return f"[BLOCKED BY SAFETY FILTERS on page {page_num}]"

            if finish and str(finish).upper() == "MAX_TOKENS":
                log.warning(
                    "API call truncated: page=%d, label=%s, duration=%.1fs — hit max tokens",
                    page_num, label, elapsed,
                )
                console.print(
                    f"\n  [yellow]Page {page_num} ({label}): truncated[/]"
                )

            result_text = response.text or ""
            log.info(
                "API call done: page=%d, label=%s, duration=%.1fs, chars=%d, finish=%s",
                page_num, label, elapsed, len(result_text), finish,
            )
            return result_text

        except Exception as e:
            elapsed = time.monotonic() - t0
            err = str(e)

            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                log.error(
                    "API rate limit: page=%d, label=%s, duration=%.1fs — %s",
                    page_num, label, elapsed, e,
                )
                raise RateLimitError(f"Rate limit on page {page_num} ({label}): {e}")

            if ("500" in err or "503" in err) and attempt < MAX_RETRIES - 1:
                wait = (2 ** attempt) * 2
                log.warning(
                    "API server error: page=%d, label=%s, duration=%.1fs, "
                    "retry in %ds (%d/%d) — %s",
                    page_num, label, elapsed, wait, attempt + 1, MAX_RETRIES, e,
                )
                console.print(
                    f"\n  [yellow]Page {page_num} ({label}): server error, "
                    f"retry in {wait}s ({attempt + 1}/{MAX_RETRIES})...[/]"
                )
                time.sleep(wait)
                continue

            log.error(
                "API call failed: page=%d, label=%s, duration=%.1fs — %s",
                page_num, label, elapsed, e,
            )
            console.print(f"\n  [red]Page {page_num} ({label}): {e}[/]")
            return f"[OCR ERROR on page {page_num}: {e}]"

    log.error("API max retries exhausted: page=%d, label=%s", page_num, label)
    return f"[OCR ERROR on page {page_num}: max retries]"


def ocr_page(client: genai.Client, image: Image.Image, page_num: int) -> str:
    """Pass 1 — extract text from the scanned page image (Flash, no thinking)."""
    return _call_gemini(
        client, [image, OCR_PROMPT], OCR_SYSTEM_INSTRUCTION, page_num, "OCR",
        model=GEMINI_MODEL_OCR,
    )


def verify_page(
    client: genai.Client, image: Image.Image, ocr_text: str, page_num: int,
    page_type: str = "prose",
) -> str:
    """Pass 2/3 — verify and correct text against the original image (Pro + thinking)."""
    hint = PAGE_TYPE_HINTS.get(page_type, "")
    prompt = VERIFY_PROMPT_TEMPLATE.format(ocr_text=ocr_text, page_type_hint=hint)
    return _call_gemini(
        client, [image, prompt], VERIFY_SYSTEM_INSTRUCTION, page_num, "verify",
        model=GEMINI_MODEL_VERIFY,
        thinking_budget=THINKING_BUDGET_VERIFY,
    )


# ---------------------------------------------------------------------------
# Searchable PDF creation
# ---------------------------------------------------------------------------


def create_searchable_pdf(
    output_path: Path,
    pdf_path: Path,
    page_nums: list[int],
    page_texts: list[str],
) -> int:
    if not page_nums:
        return 0

    font_name = register_devanagari_font()
    encoding_errors = 0

    first_img = convert_single_page(pdf_path, page_nums[0])
    iw, ih = first_img.size
    c = canvas.Canvas(
        str(output_path), pagesize=(iw * 72.0 / DPI, ih * 72.0 / DPI)
    )
    del first_img

    with tempfile.TemporaryDirectory() as tmp_dir:
        for page_num, text in zip(page_nums, page_texts):
            img = convert_single_page(pdf_path, page_num)
            iw, ih = img.size
            pw, ph = iw * 72.0 / DPI, ih * 72.0 / DPI
            c.setPageSize((pw, ph))

            tmp_img = Path(tmp_dir) / f"page_{page_num}.jpg"
            img.save(str(tmp_img), "JPEG", quality=85)
            c.drawImage(str(tmp_img), 0, 0, width=pw, height=ph)

            body = text.strip()
            if body and body != "[BLANK PAGE]":
                lines = body.split("\n")
                spacing = ph / (len(lines) + 1) if lines else ph
                for j, line in enumerate(lines):
                    s = line.strip()
                    if s:
                        y = ph - (j + 1) * spacing
                        try:
                            t = c.beginText(36, max(y, 10))
                            t.setTextRenderMode(3)
                            t.setFont(font_name, 1)
                            t.textLine(s)
                            c.drawText(t)
                        except Exception:
                            encoding_errors += 1

            c.showPage()
            del img

    c.save()
    return encoding_errors


# ---------------------------------------------------------------------------
# Translation (Pass 4: Sanskrit → English)
# ---------------------------------------------------------------------------


def translate_page(
    client: genai.Client,
    sanskrit_text: str,
    page_num: int,
    context_pages: list[tuple[int, str]] | None = None,
) -> str:
    """Pass 4 — translate Sanskrit to English with sliding window context."""
    context_section = ""
    if context_pages:
        parts = []
        for ctx_num, ctx_text in context_pages:
            parts.append(f"--- Page {ctx_num} ---\n{ctx_text}")
        context_section = CONTEXT_SECTION_TEMPLATE.format(
            context_text="\n\n".join(parts)
        )

    prompt = TRANSLATE_PROMPT_TEMPLATE.format(
        context_section=context_section,
        page_num=page_num,
        sanskrit_text=sanskrit_text,
    )

    return _call_gemini(
        client, [prompt], TRANSLATE_SYSTEM_INSTRUCTION, page_num, "translate",
        model=GEMINI_MODEL_TRANSLATE,
        thinking_budget=THINKING_BUDGET_TRANSLATE,
    )


def verify_translation(
    client: genai.Client, sanskrit_text: str, english_text: str, page_num: int,
) -> str:
    """Pass 5 — verify translation against Sanskrit source."""
    prompt = VERIFY_TRANSLATION_PROMPT_TEMPLATE.format(
        page_num=page_num,
        sanskrit_text=sanskrit_text,
        english_text=english_text,
    )
    return _call_gemini(
        client, [prompt], VERIFY_TRANSLATION_SYSTEM_INSTRUCTION,
        page_num, "verify_translation",
        model=GEMINI_MODEL_TRANSLATE,
        thinking_budget=THINKING_BUDGET_TRANSLATE,
    )


def _start_continuation_page(
    c: canvas.Canvas,
    page_w: float,
    page_h: float,
    margin: float,
    page_num: int,
    font_name: str,
    font_size: int,
) -> float:
    """Start a new PDF continuation page with a gray header."""
    c.showPage()
    c.setFont(font_name, 9)
    c.setFillColorRGB(0.4, 0.4, 0.4)
    c.drawRightString(page_w - margin, page_h - 50, f"Page {page_num} (cont.)")
    c.setFillColorRGB(0, 0, 0)
    c.setFont(font_name, font_size)
    return page_h - margin


def create_text_pdf(
    output_path: Path,
    page_nums: list[int],
    page_texts: list[str],
) -> None:
    """Create a text-only English PDF with visible text and page breaks."""
    if not page_nums:
        return

    page_w, page_h = 8.5 * 72, 11 * 72  # US Letter
    margin = 72  # 1 inch all around
    font_name = "Helvetica"
    font_size = 11
    leading = 14  # line spacing
    max_text_w = page_w - 2 * margin

    c = canvas.Canvas(str(output_path), pagesize=(page_w, page_h))

    for page_num, text in zip(page_nums, page_texts):
        body = text.strip()

        # Page header
        c.setFont(font_name, 9)
        c.setFillColorRGB(0.4, 0.4, 0.4)
        c.drawRightString(page_w - margin, page_h - 50, f"Page {page_num}")
        c.setFillColorRGB(0, 0, 0)

        if not body or body == "[BLANK PAGE]":
            c.setFont(font_name, 12)
            c.drawCentredString(page_w / 2, page_h / 2, "[BLANK PAGE]")
            c.showPage()
            continue

        # Render text with word wrapping
        y = page_h - margin
        c.setFont(font_name, font_size)

        for line in body.split("\n"):
            if not line.strip():
                # Blank line — add spacing
                y -= leading
                if y < margin:
                    y = _start_continuation_page(c, page_w, page_h, margin, page_num, font_name, font_size)
                continue

            # Word-wrap the line
            words = line.split()
            current = ""
            for word in words:
                test = f"{current} {word}".strip()
                if c.stringWidth(test, font_name, font_size) <= max_text_w:
                    current = test
                else:
                    if current:
                        c.drawString(margin, y, current)
                        y -= leading
                        if y < margin:
                            y = _start_continuation_page(c, page_w, page_h, margin, page_num, font_name, font_size)
                    current = word

            if current:
                c.drawString(margin, y, current)
                y -= leading
                if y < margin:
                    y = _start_continuation_page(c, page_w, page_h, margin, page_num, font_name, font_size)

        c.showPage()

    c.save()


def process_translation(
    client: genai.Client,
    pdf_path: Path,
    first_page: int,
    last_page: int,
) -> bool:
    """Translate OCR'd pages to English with sliding window context,
    then verify each translation against the Sanskrit source.

    Returns True if completed, False if rate-limited or cancelled.
    """
    all_page_nums = list(range(first_page, last_page + 1))
    num_pages = len(all_page_nums)

    # Build work list — determine what each page needs
    needs_translate = []   # needs translate + verify (2 API calls)
    needs_verify = []      # has _translated.txt but no _translation_verified.txt (1 call)
    already_done = 0
    no_source = []

    for p in all_page_nums:
        verified = load_page_text(pdf_path, p, STAGE_TRANSLATION_VERIFIED)
        if verified is not None:
            already_done += 1
            continue

        translated = load_page_text(pdf_path, p, STAGE_TRANSLATED)
        if translated is not None:
            if is_page_error(translated):
                already_done += 1  # error page, nothing more to do
            else:
                needs_verify.append(p)
            continue

        source = load_best_text(pdf_path, p)
        if source and not is_page_error(source):
            needs_translate.append(p)
        else:
            no_source.append(p)

    total_api_calls = len(needs_translate) * 2 + len(needs_verify)

    console.print(
        f"\n  [bold]Translation Status:[/]\n"
        f"  [bold]Pages to translate:[/]      {len(needs_translate)}"
    )
    if needs_verify:
        console.print(
            f"  [bold]Pages to verify only:[/]   {len(needs_verify)}"
        )
    console.print(
        f"  [bold]Already done:[/]            {already_done}\n"
        f"  [bold]No source text:[/]          {len(no_source)}\n"
        f"  [bold]API calls needed:[/]        {total_api_calls}"
    )

    if no_source:
        console.print(
            f"  [yellow]{len(no_source)} page(s) have no OCR text — "
            f"run OCR first.[/]"
        )

    if not needs_translate and not needs_verify:
        console.print("\n  [green]All pages already translated and verified![/]")
        return True

    if not Confirm.ask(
        f"\n  [bold]Start translation?[/] ({total_api_calls} API calls)",
        default=True,
    ):
        console.print("\n  [dim]Cancelled.[/]")
        return False

    # Build work items: (page_num, start_stage) sorted by page number
    work_items: list[tuple[int, str]] = []
    for p in needs_translate:
        work_items.append((p, "translate"))
    for p in needs_verify:
        work_items.append((p, "verify_only"))
    work_items.sort(key=lambda x: x[0])

    console.print()
    rate_limited = False
    api_calls_made = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
    ) as prog:
        task = prog.add_task("Translating...", total=total_api_calls)

        for page_num, start_stage in work_items:
            source_text = load_best_text(pdf_path, page_num)
            if not source_text or is_page_error(source_text):
                # Skip pages without usable source
                if start_stage == "translate":
                    prog.advance(task, 2)
                else:
                    prog.advance(task)
                continue

            # --- Translate step (if needed) ---
            if start_stage == "translate":
                if api_calls_made > 0:
                    time.sleep(INTER_CALL_DELAY)

                prog.update(task, description=f"Page {page_num}: translating...")

                # Blank pages — skip both translate and verify
                if source_text.strip() == "[BLANK PAGE]":
                    save_page_text(pdf_path, page_num, "[BLANK PAGE]", STAGE_TRANSLATED)
                    save_page_text(pdf_path, page_num, "[BLANK PAGE]", STAGE_TRANSLATION_VERIFIED)
                    save_progress_json(pdf_path, first_page, last_page)
                    prog.advance(task, 2)
                    continue

                # Build context window (previous N pages, oldest first)
                context_pages = []
                for offset in range(CONTEXT_WINDOW_PAGES, 0, -1):
                    ctx_num = page_num - offset
                    if ctx_num >= first_page:
                        # Prefer verified translation > raw translation > Sanskrit source
                        ctx_text = load_page_text(pdf_path, ctx_num, STAGE_TRANSLATION_VERIFIED)
                        if not ctx_text or is_page_error(ctx_text):
                            ctx_text = load_page_text(pdf_path, ctx_num, STAGE_TRANSLATED)
                        if not ctx_text or is_page_error(ctx_text):
                            ctx_text = load_best_text(pdf_path, ctx_num)
                        if ctx_text and not is_page_error(ctx_text) and ctx_text.strip() != "[BLANK PAGE]":
                            context_pages.append((ctx_num, ctx_text))

                log.info(
                    "Translation: page=%d, context=%s",
                    page_num, [p for p, _ in context_pages] or "none",
                )

                try:
                    translated = translate_page(
                        client, source_text, page_num, context_pages=context_pages
                    )
                except RateLimitError:
                    rate_limited = True
                    break
                api_calls_made += 1

                translated = normalize_whitespace(translated)

                # Quality gate
                garbage, reason = is_garbage_output(translated)
                if garbage:
                    log.warning(
                        "Page %d: garbage translation — %s", page_num, reason
                    )
                    console.print(
                        f"\n  [yellow]Page {page_num}: garbage translation "
                        f"({reason})[/]"
                    )
                    save_page_text(
                        pdf_path, page_num,
                        f"[GARBAGE OUTPUT on page {page_num}: translation {reason}]",
                        STAGE_TRANSLATED,
                    )
                    save_progress_json(pdf_path, first_page, last_page)
                    prog.advance(task, 2)  # skip both translate + verify progress
                    continue

                save_page_text(pdf_path, page_num, translated, STAGE_TRANSLATED)
                save_progress_json(pdf_path, first_page, last_page)
                prog.advance(task)
                log.info(
                    "Page %d: translation saved (%d chars)", page_num, len(translated)
                )

                # Fall through to verify step below
                time.sleep(INTER_CALL_DELAY)

            # --- Verify translation step ---
            english_text = load_page_text(pdf_path, page_num, STAGE_TRANSLATED)
            if not english_text or is_page_error(english_text):
                prog.advance(task)
                continue

            # Blank pages — just promote
            if english_text.strip() == "[BLANK PAGE]":
                save_page_text(pdf_path, page_num, "[BLANK PAGE]", STAGE_TRANSLATION_VERIFIED)
                save_progress_json(pdf_path, first_page, last_page)
                prog.advance(task)
                continue

            if start_stage == "verify_only" and api_calls_made > 0:
                time.sleep(INTER_CALL_DELAY)

            prog.update(task, description=f"Page {page_num}: verifying translation...")

            try:
                verified = verify_translation(
                    client, source_text, english_text, page_num
                )
            except RateLimitError:
                rate_limited = True
                break
            api_calls_made += 1

            verified = normalize_whitespace(verified)

            # Quality gate — if verify produced garbage, keep original translation
            garbage, reason = is_garbage_output(verified)
            if garbage:
                log.warning(
                    "Page %d: garbage translation verify — %s; keeping original",
                    page_num, reason,
                )
                console.print(
                    f"\n  [yellow]Page {page_num}: garbage verify output "
                    f"({reason}) — keeping original translation[/]"
                )
                save_page_text(
                    pdf_path, page_num, english_text, STAGE_TRANSLATION_VERIFIED
                )
            else:
                save_page_text(pdf_path, page_num, verified, STAGE_TRANSLATION_VERIFIED)
                log.info(
                    "Page %d: translation verified (%d chars)",
                    page_num, len(verified),
                )

            save_progress_json(pdf_path, first_page, last_page)
            prog.advance(task)

    if rate_limited:
        log.warning("Translation rate limited after %d calls", api_calls_made)
        save_progress_json(pdf_path, first_page, last_page)

        done = sum(
            1 for p in all_page_nums
            if load_page_text(pdf_path, p, STAGE_TRANSLATION_VERIFIED) is not None
        )
        translated_count = sum(
            1 for p in all_page_nums
            if load_page_text(pdf_path, p, STAGE_TRANSLATED) is not None
        )
        console.print(
            Panel(
                f"[bold red]Rate limit reached![/]\n\n"
                f"  [bold]Translated:[/]          {translated_count}/{num_pages} pages\n"
                f"  [bold]Verified:[/]            {done}/{num_pages} pages\n"
                f"  [bold]API calls made:[/]      {api_calls_made}\n\n"
                f"  Progress saved. Run again after the limit resets.",
                title="[bold red]Stopped — Rate Limit[/]",
                border_style="red",
                padding=(1, 2),
            )
        )
        return False

    return True


def assemble_translation_output(
    pdf_path: Path,
    first_page: int,
    last_page: int,
) -> None:
    """Generate final translation outputs: .txt and .pdf files."""
    all_page_nums = list(range(first_page, last_page + 1))
    num_pages = len(all_page_nums)

    page_texts = []
    pages_with_text = []
    missing = []

    for p in all_page_nums:
        # Prefer verified translation, fall back to raw translation
        text = load_page_text(pdf_path, p, STAGE_TRANSLATION_VERIFIED)
        if text is None:
            text = load_page_text(pdf_path, p, STAGE_TRANSLATED)
        if text is not None and not is_page_error(text):
            page_texts.append(text)
            pages_with_text.append(p)
        else:
            missing.append(p)

    if not pages_with_text:
        console.print("  [red]No translated pages yet.[/]")
        return

    if missing:
        console.print(
            f"  [yellow]{len(missing)} page(s) not yet translated.[/]"
        )

    stem = pdf_path.stem
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_txt = DEFAULT_OUTPUT_DIR / f"{stem}_translation.txt"
    output_pdf = DEFAULT_OUTPUT_DIR / f"{stem}_translation.pdf"

    # Text file
    with open(output_txt, "w", encoding="utf-8") as f:
        for p, text in zip(pages_with_text, page_texts):
            f.write(f"--- Page {p} ---\n")
            f.write(text.strip())
            f.write("\n\n")
    console.print(f"  [green]Translation text:[/]  {output_txt}")

    # Text PDF
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        transient=True,
    ) as prog:
        prog.add_task("Building translation PDF...", total=None)
        create_text_pdf(output_pdf, pages_with_text, page_texts)
    console.print(f"  [green]Translation PDF:[/]   {output_pdf}")

    # Summary
    n_translated = len(pages_with_text)
    all_done = n_translated == num_pages and not missing

    log.info(
        "Translation output: translated=%d/%d, missing=%d",
        n_translated, num_pages, len(missing),
    )

    summary = []
    if all_done:
        summary.append("[bold green]Translation Complete![/]\n")
    else:
        summary.append("[bold yellow]Translation Partially Complete[/]\n")

    summary.append(f"  [bold]PDF:[/]         {output_pdf}")
    summary.append(f"  [bold]Text file:[/]   {output_txt}")
    summary.append(f"  [bold]Translated:[/]  {n_translated}/{num_pages} pages")

    if missing:
        summary.append(f"  [bold]Missing:[/]     {len(missing)} (run translation again)")

    summary.append(
        f"  [bold]Size:[/]        {format_file_size(output_pdf.stat().st_size)}"
    )

    console.print()
    console.print(
        Panel(
            "\n".join(summary),
            title="[bold green]Translation[/]" if all_done else "[bold yellow]Translation[/]",
            border_style="green" if all_done else "yellow",
            padding=(1, 2),
        )
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    console.print(
        Panel(
            "[bold]Sanskrit OCR Tool[/]\n"
            "[dim]Multi-pass OCR + verification for maximum accuracy[/]\n"
            f"[dim]OCR: {GEMINI_MODEL_OCR}  |  Verify: {GEMINI_MODEL_VERIFY}[/]",
            border_style="cyan",
            padding=(1, 2),
        )
    )

    # --- Setup ---
    log_path = setup_logging()
    console.print(f"  [dim]Log: {log_path}[/]\n")

    api_key = load_api_key()
    client = genai.Client(api_key=api_key)

    pdf_path = select_pdf()
    info = get_pdf_info(pdf_path)

    console.print(f"\n  [bold]File:[/]  {info['name']}")
    console.print(f"  [bold]Size:[/]  {info['size']}")
    if info["pages"]:
        console.print(f"  [bold]Pages:[/] {info['pages']}")

    total_pages = info["pages"]
    if not total_pages:
        console.print("  [yellow]Could not determine page count.[/]")
        total_pages = IntPrompt.ask("  Total pages")

    # --- Check for existing progress ---
    progress = load_progress_json(pdf_path)
    resuming = False

    if progress:
        n_rechecked = len(progress.get("rechecked_pages", []))
        n_verified = len(progress.get("verified_pages", []))
        n_ocr = len(progress.get("ocr_only_pages", []))
        n_err = len(progress.get("error_pages", []))
        n_pending = len(progress.get("pending_pages", []))
        prev_first = progress["first_page"]
        prev_last = progress["last_page"]
        prev_total = prev_last - prev_first + 1

        console.print(
            Panel(
                f"[bold yellow]Previous progress found![/]\n\n"
                f"  [bold]Page range:[/]     {prev_first}-{prev_last} ({prev_total} pages)\n"
                f"  [bold]Rechecked:[/]      {n_rechecked}\n"
                f"  [bold]Verified:[/]       {n_verified}\n"
                f"  [bold]OCR only:[/]       {n_ocr} (need verification pass)\n"
                f"  [bold]Errors:[/]         {n_err}\n"
                f"  [bold]Pending:[/]        {n_pending}",
                title="[bold yellow]Resume Available[/]",
                border_style="yellow",
                padding=(1, 2),
            )
        )

        resume_choice = Prompt.ask(
            "\n  [bold]What would you like to do?[/]\n"
            "  [1] Resume (continue from where we left off)\n"
            "  [2] Start fresh (discard previous progress)\n"
            "  [3] New page range\n\n"
            "  Choice",
            choices=["1", "2", "3"],
            default="1",
        )

        if resume_choice == "1":
            resuming = True
            first_page = prev_first
            last_page = prev_last
        elif resume_choice == "2":
            d = get_progress_dir(pdf_path)
            if d.exists():
                shutil.rmtree(d)
            first_page, last_page = select_page_range(total_pages)
        else:
            first_page, last_page = select_page_range(total_pages)
    else:
        first_page, last_page = select_page_range(total_pages)

    num_pages = last_page - first_page + 1
    all_page_nums = list(range(first_page, last_page + 1))

    log.info("PDF: %s (%s, %d total pages)", pdf_path.name, info["size"], total_pages)
    log.info("Range: pages %d-%d (%d pages), resuming=%s", first_page, last_page, num_pages, resuming)

    # --- Build work list from file-based stage detection ---
    needs_ocr = []
    needs_verify = []
    needs_recheck = []

    for p in all_page_nums:
        stage = get_page_stage(pdf_path, p)
        if stage in (STAGE_TRANSLATED, STAGE_TRANSLATION_VERIFIED):
            # Already translated — OCR is fully done
            pass
        elif stage == STAGE_RECHECKED:
            # Fully done (3 passes completed)
            text = load_page_text(pdf_path, p, STAGE_RECHECKED)
            if text and is_page_error(text):
                needs_ocr.append(p)
        elif stage == STAGE_VERIFIED:
            text = load_page_text(pdf_path, p, STAGE_VERIFIED)
            if text and is_page_error(text):
                needs_ocr.append(p)
            else:
                # Check if verify changed a lot but recheck never ran
                # (can happen if interrupted between verify and recheck)
                ocr_text = load_page_text(pdf_path, p, STAGE_OCR)
                if ocr_text and text and compute_change_ratio(ocr_text, text) > RECHECK_THRESHOLD:
                    needs_recheck.append(p)
        elif stage == STAGE_OCR:
            text = load_page_text(pdf_path, p, STAGE_OCR)
            if text and is_page_error(text):
                needs_ocr.append(p)  # redo OCR
            else:
                needs_verify.append(p)  # just needs pass 2
        else:
            needs_ocr.append(p)

    # API calls: OCR pages need 2 (ocr + verify), verify-only need 1,
    # recheck-only need 1. New pages may also trigger rechecks dynamically.
    total_work = len(needs_ocr) * 2 + len(needs_verify) + len(needs_recheck)
    already_done = num_pages - len(needs_ocr) - len(needs_verify) - len(needs_recheck)

    log.info(
        "Work: ocr=%d, verify=%d, recheck=%d, done=%d, min_api_calls=%d",
        len(needs_ocr), len(needs_verify), len(needs_recheck), already_done, total_work,
    )

    console.print(
        f"\n  [bold]Page range:[/]   {first_page}-{last_page} ({num_pages} pages)"
    )
    console.print(f"  [bold]Complete:[/]     {already_done}")
    console.print(f"  [bold]Need OCR:[/]     {len(needs_ocr)} (2+ API calls each)")
    console.print(f"  [bold]Need verify:[/]  {len(needs_verify)} (1+ API call each)")
    if needs_recheck:
        console.print(f"  [bold]Need recheck:[/] {len(needs_recheck)} (1 API call each)")
    console.print(f"  [bold]Min API calls:[/] {total_work} (+ rechecks for high-change pages)")

    if total_work == 0:
        console.print("\n  [green]All pages fully verified![/]")
    else:
        if not Confirm.ask(
            f"\n  [bold]{'Resume' if resuming else 'Start'} processing?[/]",
            default=True,
        ):
            console.print("\n  [dim]Cancelled.[/]")
            return

        # --- Process pages ---
        work_items: list[tuple[int, str]] = []
        for p in needs_ocr:
            work_items.append((p, "ocr"))
        for p in needs_verify:
            work_items.append((p, "verify"))
        for p in needs_recheck:
            work_items.append((p, "recheck"))
        # Sort by page number for sequential processing
        work_items.sort(key=lambda x: x[0])

        console.print()
        rate_limited = False
        api_calls_made = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
        ) as prog:
            task = prog.add_task("Processing...", total=total_work)

            for page_num, start_stage in work_items:
                log.info("--- Page %d: starting stage=%s ---", page_num, start_stage)

                # --- Pass 1: OCR (if needed) ---
                if start_stage == "ocr":
                    if api_calls_made > 0:
                        time.sleep(INTER_CALL_DELAY)

                    prog.update(task, description=f"Page {page_num}: OCR...")

                    try:
                        img = convert_single_page(pdf_path, page_num)
                    except Exception as e:
                        log.error("Page %d: image extraction failed: %s", page_num, e)
                        console.print(
                            f"\n  [red]Page {page_num}: image extraction failed: {e}[/]"
                        )
                        save_page_text(
                            pdf_path, page_num,
                            f"[OCR ERROR on page {page_num}: image extraction failed]",
                            STAGE_OCR,
                        )
                        save_progress_json(pdf_path, first_page, last_page)
                        prog.advance(task, 2)  # skip both passes
                        continue

                    try:
                        raw_text = ocr_page(client, img, page_num)
                    except RateLimitError:
                        del img
                        rate_limited = True
                        break
                    api_calls_made += 1

                    raw_len = len(raw_text)
                    raw_text = normalize_whitespace(raw_text)

                    # Quality gate: detect garbage/hallucinated OCR output
                    garbage, reason = is_garbage_output(raw_text, raw_len=raw_len)
                    if garbage:
                        log.warning("Page %d: garbage OCR detected — %s", page_num, reason)
                        console.print(
                            f"\n  [yellow]Page {page_num}: garbage OCR detected ({reason}) — skipping verify[/]"
                        )
                        save_page_text(
                            pdf_path, page_num,
                            f"[GARBAGE OUTPUT on page {page_num}: {reason}]",
                            STAGE_OCR,
                        )
                        save_progress_json(pdf_path, first_page, last_page)
                        prog.advance(task, 2)  # skip OCR + verify progress steps
                        del img
                        continue

                    save_page_text(pdf_path, page_num, raw_text, STAGE_OCR)
                    save_progress_json(pdf_path, first_page, last_page)
                    prog.advance(task)
                    log.info("Page %d: OCR saved (%d chars)", page_num, len(raw_text))

                    if is_page_error(raw_text):
                        # Can't verify an error — skip pass 2
                        log.warning("Page %d: OCR returned error, skipping verify", page_num)
                        prog.advance(task)
                        del img
                        continue

                    # Small pause between the two calls for the same page
                    time.sleep(INTER_CALL_DELAY)
                elif start_stage == "recheck":
                    # Resume-only: page was verified but needs recheck
                    try:
                        img = convert_single_page(pdf_path, page_num)
                    except Exception as e:
                        console.print(
                            f"\n  [red]Page {page_num}: image extraction failed: {e}[/]"
                        )
                        prog.advance(task)
                        continue

                    verified_text = load_page_text(pdf_path, page_num, STAGE_VERIFIED)
                    if not verified_text or is_page_error(verified_text):
                        prog.advance(task)
                        del img
                        continue

                    if api_calls_made > 0:
                        time.sleep(INTER_CALL_DELAY)

                    page_type = detect_page_type(verified_text)
                    prog.update(task, description=f"Page {page_num}: recheck...")

                    try:
                        rechecked_text = verify_page(
                            client, img, verified_text, page_num,
                            page_type=page_type,
                        )
                    except RateLimitError:
                        del img
                        rate_limited = True
                        break
                    api_calls_made += 1

                    rechecked_text = normalize_whitespace(rechecked_text)

                    # Quality gate: if recheck produced garbage, keep verified text
                    garbage, reason = is_garbage_output(rechecked_text)
                    if garbage:
                        log.warning("Page %d: garbage recheck output (resumed) — %s; keeping verified text", page_num, reason)
                        console.print(
                            f"\n  [yellow]Page {page_num}: garbage recheck output ({reason}) — keeping verified text[/]"
                        )
                        save_page_text(
                            pdf_path, page_num, verified_text, STAGE_RECHECKED
                        )
                        save_progress_json(pdf_path, first_page, last_page)
                        prog.advance(task)
                        del img
                        continue

                    save_page_text(
                        pdf_path, page_num, rechecked_text, STAGE_RECHECKED
                    )
                    save_progress_json(pdf_path, first_page, last_page)
                    prog.advance(task)
                    log.info("Page %d: recheck saved (resumed, %d chars)", page_num, len(rechecked_text))
                    console.print(
                        f"\n  [dim]Page {page_num}: rechecked (resumed)[/]"
                    )
                    del img
                    continue

                else:
                    # Load image + existing OCR text for verify-only pages
                    try:
                        img = convert_single_page(pdf_path, page_num)
                    except Exception as e:
                        console.print(
                            f"\n  [red]Page {page_num}: image extraction failed: {e}[/]"
                        )
                        prog.advance(task)
                        continue

                    raw_text = load_page_text(pdf_path, page_num, STAGE_OCR)
                    if not raw_text or is_page_error(raw_text):
                        prog.advance(task)
                        del img
                        continue

                    if api_calls_made > 0:
                        time.sleep(INTER_CALL_DELAY)

                # --- Pass 2: Verify ---
                prog.update(task, description=f"Page {page_num}: verifying...")

                # Skip verification for blank pages
                if raw_text.strip() in ("[BLANK PAGE]", ""):
                    save_page_text(pdf_path, page_num, raw_text.strip() or "[BLANK PAGE]", STAGE_VERIFIED)
                    save_progress_json(pdf_path, first_page, last_page)
                    prog.advance(task)
                    del img
                    continue

                page_type = detect_page_type(raw_text)

                try:
                    verified_text = verify_page(
                        client, img, raw_text, page_num, page_type=page_type
                    )
                except RateLimitError:
                    del img
                    rate_limited = True
                    break
                api_calls_made += 1

                verified_text = normalize_whitespace(verified_text)

                # Quality gate: if verify produced garbage, keep OCR text instead
                garbage, reason = is_garbage_output(verified_text)
                if garbage:
                    log.warning("Page %d: garbage verify output — %s; keeping OCR text", page_num, reason)
                    console.print(
                        f"\n  [yellow]Page {page_num}: garbage verify output ({reason}) — keeping OCR text, skipping recheck[/]"
                    )
                    save_page_text(pdf_path, page_num, raw_text, STAGE_VERIFIED)
                    save_progress_json(pdf_path, first_page, last_page)
                    prog.advance(task)
                    del img
                    continue

                save_page_text(pdf_path, page_num, verified_text, STAGE_VERIFIED)
                save_progress_json(pdf_path, first_page, last_page)
                prog.advance(task)

                # --- Pass 3: Recheck (if verify changed a lot) ---
                change = compute_change_ratio(raw_text, verified_text)
                log.info(
                    "Page %d: verified (%d chars), change_ratio=%.2f (threshold=%.2f)",
                    page_num, len(verified_text), change, RECHECK_THRESHOLD,
                )
                if change > RECHECK_THRESHOLD:
                    prog.update(
                        task,
                        description=f"Page {page_num}: recheck ({change:.0%} changed)...",
                    )
                    time.sleep(INTER_CALL_DELAY)

                    try:
                        rechecked_text = verify_page(
                            client, img, verified_text, page_num,
                            page_type=page_type,
                        )
                    except RateLimitError:
                        del img
                        rate_limited = True
                        break
                    api_calls_made += 1

                    rechecked_text = normalize_whitespace(rechecked_text)

                    # Quality gate: if recheck produced garbage, keep verified text
                    garbage, reason = is_garbage_output(rechecked_text)
                    if garbage:
                        log.warning("Page %d: garbage recheck output — %s; keeping verified text", page_num, reason)
                        console.print(
                            f"\n  [yellow]Page {page_num}: garbage recheck output ({reason}) — keeping verified text[/]"
                        )
                        save_page_text(
                            pdf_path, page_num, verified_text, STAGE_RECHECKED
                        )
                        save_progress_json(pdf_path, first_page, last_page)
                        del img
                        continue

                    save_page_text(
                        pdf_path, page_num, rechecked_text, STAGE_RECHECKED
                    )
                    save_progress_json(pdf_path, first_page, last_page)
                    log.info(
                        "Page %d: rechecked (%d chars, triggered by %.0f%% change)",
                        page_num, len(rechecked_text), change * 100,
                    )
                    console.print(
                        f"\n  [dim]Page {page_num}: rechecked "
                        f"({change:.0%} change triggered 3rd pass)[/]"
                    )

                del img

            if rate_limited:
                log.warning("Rate limited after %d API calls", api_calls_made)
                save_progress_json(pdf_path, first_page, last_page)
                prog.stop()

                # Count current state (verified/rechecked/translated = done)
                done = sum(
                    1 for p in all_page_nums
                    if get_page_stage(pdf_path, p) in (STAGE_VERIFIED, STAGE_RECHECKED, STAGE_TRANSLATED, STAGE_TRANSLATION_VERIFIED)
                    and not is_page_error(load_best_text(pdf_path, p) or "")
                )

                console.print(
                    Panel(
                        f"[bold red]Rate limit reached![/]\n\n"
                        f"  [bold]Fully verified:[/] {done}/{num_pages} pages\n"
                        f"  [bold]API calls made this run:[/] {api_calls_made}\n\n"
                        f"  Progress saved. Run again after the limit resets.",
                        title="[bold red]Stopped — Rate Limit[/]",
                        border_style="red",
                        padding=(1, 2),
                    )
                )
                return

    # --- Assemble final outputs ---
    console.print()

    page_texts = []
    pages_with_text = []
    missing = []

    for p in all_page_nums:
        text = load_best_text(pdf_path, p)
        if text is not None and not is_page_error(text):
            page_texts.append(text)
            pages_with_text.append(p)
        else:
            missing.append(p)

    if not pages_with_text:
        console.print("  [red]No pages processed yet. Nothing to output.[/]")
        return

    if missing:
        console.print(
            f"  [yellow]{len(missing)} page(s) still incomplete "
            f"(run again to finish).[/]"
        )

    stem = pdf_path.stem
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_txt = DEFAULT_OUTPUT_DIR / f"{stem}_ocr.txt"
    output_pdf = DEFAULT_OUTPUT_DIR / f"{stem}_ocr.pdf"

    # Text file
    with open(output_txt, "w", encoding="utf-8") as f:
        for p, text in zip(pages_with_text, page_texts):
            f.write(f"--- Page {p} ---\n")
            f.write(text.strip())
            f.write("\n\n")
    console.print(f"  [green]Text file saved:[/] {output_txt}")

    # Searchable PDF
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]{task.description}"),
        transient=True,
    ) as prog:
        prog.add_task("Building searchable PDF...", total=None)
        enc_errors = create_searchable_pdf(
            output_pdf, pdf_path, pages_with_text, page_texts
        )

    if enc_errors > 0:
        console.print(
            f"  [yellow]{enc_errors} text lines could not be encoded in "
            f"the PDF text layer. The .txt file has the full text.[/]"
        )

    # --- Summary ---
    n_done = sum(
        1 for p in all_page_nums
        if get_page_stage(pdf_path, p) in (STAGE_VERIFIED, STAGE_RECHECKED, STAGE_TRANSLATED, STAGE_TRANSLATION_VERIFIED)
        and not is_page_error(load_best_text(pdf_path, p) or "")
    )
    n_rechecked = sum(
        1 for p in all_page_nums
        if get_page_stage(pdf_path, p) == STAGE_RECHECKED
    )
    all_done = n_done == num_pages and not missing

    log.info(
        "Session complete: verified=%d/%d, rechecked=%d, missing=%d, all_done=%s",
        n_done, num_pages, n_rechecked, len(missing), all_done,
    )

    lines = []
    if all_done:
        lines.append("[bold green]OCR Complete — All pages verified![/]\n")
    else:
        lines.append("[bold yellow]OCR Partially Complete[/]\n")

    lines.append(f"  [bold]Searchable PDF:[/]  {output_pdf}")
    lines.append(f"  [bold]Text file:[/]       {output_txt}")
    lines.append(f"  [bold]Verified pages:[/]  {n_done}/{num_pages}")
    if n_rechecked > 0:
        lines.append(f"  [bold]Rechecked:[/]       {n_rechecked} (3rd pass)")

    if missing:
        lines.append(f"  [bold]Incomplete:[/]      {len(missing)} (run again)")

    lines.append(
        f"  [bold]Output size:[/]     {format_file_size(output_pdf.stat().st_size)}"
    )

    if all_done:
        lines.append(f"\n  [dim]Progress dir: {get_progress_dir(pdf_path)}[/]")
        lines.append("  [dim]Safe to delete once you're happy with the output.[/]")

    console.print()
    console.print(
        Panel(
            "\n".join(lines),
            title="[bold green]Results[/]" if all_done else "[bold yellow]Results[/]",
            border_style="green" if all_done else "yellow",
            padding=(1, 2),
        )
    )

    # --- Translation prompt ---
    if all_done and Confirm.ask(
        "\n  [bold]Translate to English?[/]", default=False
    ):
        completed = process_translation(client, pdf_path, first_page, last_page)
        if completed:
            assemble_translation_output(pdf_path, first_page, last_page)


def translate_only() -> None:
    """Translation-only mode for already-OCR'd books."""
    setup_logging()

    console.print(
        Panel(
            "[bold]Sanskrit OCR Tool[/]\n"
            "[dim]Translation-only mode[/]",
            border_style="cyan",
            padding=(1, 2),
        )
    )

    pdf_path = select_pdf()  # exits on failure

    total_pages = get_pdf_page_count(pdf_path)
    console.print(f"\n  [bold]File:[/]  {pdf_path.name}")
    console.print(f"  [bold]Pages:[/] {total_pages}")

    # Detect page range from existing progress
    progress_dir = get_progress_dir(pdf_path)
    if not progress_dir.exists():
        console.print("\n  [red]No OCR progress found for this PDF. Run OCR first.[/]")
        return

    # Find page range from existing OCR files
    ocr_pages = []
    for f in sorted(progress_dir.glob("page_*_ocr.txt")):
        try:
            num = int(f.stem.split("_")[1])
            ocr_pages.append(num)
        except (IndexError, ValueError):
            continue

    if not ocr_pages:
        console.print("\n  [red]No OCR'd pages found. Run OCR first.[/]")
        return

    first_page = min(ocr_pages)
    last_page = max(ocr_pages)

    # Check how many pages are fully verified (have verified or rechecked text)
    verified_count = 0
    for p in range(first_page, last_page + 1):
        stage = get_page_stage(pdf_path, p)
        if stage in (STAGE_VERIFIED, STAGE_RECHECKED, STAGE_TRANSLATED, STAGE_TRANSLATION_VERIFIED):
            verified_count += 1

    console.print(f"  [bold]OCR range:[/] {first_page}-{last_page} ({last_page - first_page + 1} pages)")
    console.print(f"  [bold]Verified:[/]  {verified_count}")

    if not Confirm.ask("\n  [bold]Start translation?[/]", default=True):
        console.print("\n  [dim]Cancelled.[/]")
        return

    api_key = load_api_key()
    client = genai.Client(api_key=api_key)

    completed = process_translation(client, pdf_path, first_page, last_page)
    if completed:
        assemble_translation_output(pdf_path, first_page, last_page)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sanskrit OCR Tool — LLM-powered OCR for scanned Sanskrit/Devanagari books."
    )
    parser.add_argument(
        "--translate", action="store_true",
        help="Translation-only mode for already-OCR'd books",
    )
    args = parser.parse_args()

    if args.translate:
        translate_only()
    else:
        main()
