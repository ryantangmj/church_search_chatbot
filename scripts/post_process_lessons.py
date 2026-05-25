import re
import os
import sys
import argparse
import subprocess
import tempfile
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

try:
    from docx import Document
    from docx.oxml.ns import qn
except ImportError:
    print("❌  python-docx not installed. Run: pip install python-docx")
    sys.exit(1)


# ── Filename parsing ───────────────────────────────────────────────────────────
# Filenames follow: YYYYMMDD_<lessonNum>_<Title_Words>_P_Joseph.doc[x]
# e.g. 20200210_23-1_Trinity__P_Joseph.docx
FILENAME_RE = re.compile(
    r'^(\d{4})(\d{2})(\d{2})'          # date: YYYYMMDD
    r'_([0-9\-]+)'                       # lesson number (may include dash: 23-1)
    r'[\s_]+(.*?)'                       # space or underscore separator, then title
    r'(?:_+P[\s_]+\w+)?'               # optional _P Joseph or __P_Joseph suffix
    r'\.docx?$',                         # extension
    re.IGNORECASE
)

# ── Noise patterns ─────────────────────────────────────────────────────────────
 
# Page footer lines: "13 | 10 Conflict in Ignorance Joe's note 150718"
PAGE_FOOTER_RE = re.compile(r'^\d+\s*\|\s*\d+\s+\w', re.IGNORECASE)
 
# Attribution / compiler footer lines. Covers variations like:
#   "Compiled by Joseph Park"
#   "Compiled and translated by Joseph Park"
#   "Compilation and translation by Joseph Park"
#   "Based on Pastor Lee, Jihwa's lecture note"
#   "Based on Yun Kang's note"
#   "Review by Emily Wang"
#   "Reviewed by ..."
#   "Translated by ..."
# Requires "by" or "on" after the keyword — rules out body sentences like
# "Compilation of sins leads to..." which have no such connector.
# Also guarded by word count in is_noise_line().
REVIEWER_RE = re.compile(
    r'^(review(ed)?'
    r'|edit(ed)?'
    r'|compiled?(\s+and\s+\w+)?'
    r'|compilation(\s+and\s+\w+)?'
    r'|translat(ed|ion)?(\s+and\s+\w+)?'
    r'|based'
    r')\s+(by|on)\b',
    re.IGNORECASE
)
REVIEWER_MAX_WORDS = 10   # attribution lines are never long prose sentences
 
# Commentary instruction lines
COMMENTARY_INSTRUCTION_RE = re.compile(
    r'^show\s+commentar(y|ies)\s+here', re.IGNORECASE
)
 
# Commentary attributor lines — named theologian credit lines
# e.g. "Dr. Thomas L. Constable", "John Gill", "Matthew Henry"
COMMENTATOR_RE = re.compile(
    r'^(dr\.?\s+)?[A-Z][a-z]+(\s+[A-Z]\.?)?\s+[A-Z][a-z]+(,\s*.+)?$'
)
 
# Lesson reference footer: "23-1 Trinity by Joseph 200210"
LESSON_REF_FOOTER_RE = re.compile(
    r'^\d+[-\d]*\s+\w[\w\s]+\s+by\s+\w+\s+\d{6}$', re.IGNORECASE
)
 
# Attribution block trigger — any line starting with these words signals
# the start of the end-of-document attribution block to be dropped entirely.
# Matches all known variations:
#   "Compiled by Joseph Park"
#   "Compiled and translated by Joseph Park"
#   "Compilation and translation by Joseph Park"
#   "Based on Pastor Lee, Jihwa's lecture note"
#   "Based on Yun Kang's note"
ATTRIBUTION_TRIGGER_RE = re.compile(
    r'^(compiled|compilation|translated|translation|based\s+on)\b',
    re.IGNORECASE
)
 
# URL lines (hyperlinks left as plain text after conversion)
URL_RE = re.compile(r'https?://\S+')
 
# Scripture reference: bold line like "John 14:9" or "1 Kings 17:1-7"
# or inline like "(1King 16:30-33)"
SCRIPTURE_REF_RE = re.compile(
    r'^(\(?\d?\s?[A-Za-z]+\.?\s+\d+:\d+[\d\-,\s]*\)?)$'
)
 
# Full scripture block header: "Book Chapter:Verse" possibly with range
SCRIPTURE_HEADER_RE = re.compile(
    r'^(\d\s)?[A-Za-z]+[\w\s]*\d+:\d+[\d\-,\s]*$'
)
 
# Section header: numbered like "1.", "2.", "1)", "A.", "A)" — bold in doc
SECTION_HEADER_RE = re.compile(
    r'^(\d+[\.\-\d]*\.?|[A-Z]\.|[A-Z]\))\s+\S'
)
 
# Sub-section header: "1)", "2)" style
SUBSECTION_RE = re.compile(r'^\d+\)\s+\S')
 
# Italic/block quote style commentary attribution (hyperlinked names)
HYPERLINK_COMMENTATOR_RE = re.compile(r'^\[.+\]\(.+\)$')
 
 
# ── Paragraph helpers ──────────────────────────────────────────────────────────
 
def para_is_bold(para) -> bool:
    """Return True if ALL non-empty runs in the paragraph are bold."""
    runs = [r for r in para.runs if r.text.strip()]
    if not runs:
        return False
    return all(r.bold for r in runs)
 
 
def para_has_image(para) -> bool:
    """Return True if the paragraph contains an inline image (drawing/picture)."""
    xml = para._element.xml
    return '<w:drawing' in xml or '<w:pict' in xml
 
 
def para_is_hyperlink(para) -> bool:
    """Return True if the paragraph's primary content is a hyperlink."""
    for child in para._element:
        if child.tag == qn('w:hyperlink'):
            return True
    return False
 
 
def get_full_text(para) -> str:
    """
    Extract full paragraph text including text inside hyperlink elements,
    which python-docx's para.text misses by default.
    """
    parts = []
    for child in para._element.iter():
        if child.tag == qn('w:t'):
            parts.append(child.text or '')
    return ''.join(parts).strip()
 
 
# ── Filename parsing ───────────────────────────────────────────────────────────
 
@dataclass
class LessonMeta:
    date: str          # YYYY-MM-DD
    lesson_number: str # e.g. "17" or "23-1"
    title: str         # e.g. "Satan and Angels"
 
 
def parse_filename(filename: str) -> Optional[LessonMeta]:
    """Extract date, lesson number, and title from the filename."""
    m = FILENAME_RE.match(filename)
    if not m:
        return None
    year, month, day = m.group(1), m.group(2), m.group(3)
    lesson_num = m.group(4)
    raw_title = m.group(5)
    # Restore _s_ as apostrophe-s BEFORE splitting on underscores
    # e.g. "Raven_s_Food" → "Raven's Food"
    title = re.sub(r"_s_", "'s ", raw_title)
    # Convert remaining underscores to spaces, collapse multiple spaces
    title = re.sub(r'_+', ' ', title).strip()
    return LessonMeta(
        date=f"{year}-{month}-{day}",
        lesson_number=lesson_num,
        title=title if title else "Unknown Title",
    )
 
 
# ── .doc → .docx conversion ───────────────────────────────────────────────────
 
def convert_doc_to_docx(doc_path: Path, out_dir: Path) -> Path:
    """
    Convert a legacy .doc file to .docx using LibreOffice.
    Returns the path to the converted .docx file.
    """
    # Try the soffice.py wrapper first (sandbox-safe), fall back to bare soffice
    script_dir = Path(__file__).parent
    soffice_wrapper = script_dir / "scripts" / "office" / "soffice.py"
    soffice_env = os.environ.get("SOFFICE_PATH")
 
    if soffice_env:
        cmd = [sys.executable, soffice_env,
               "--convert-to", "docx", str(doc_path), "--outdir", str(out_dir)]
    elif soffice_wrapper.exists():
        cmd = [sys.executable, str(soffice_wrapper),
               "--convert-to", "docx", str(doc_path), "--outdir", str(out_dir)]
    else:
        cmd = ["soffice", "--headless", "--convert-to", "docx",
               str(doc_path), "--outdir", str(out_dir)]
 
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"LibreOffice conversion failed: {result.stderr}")
 
    out_path = out_dir / (doc_path.stem + ".docx")
    if not out_path.exists():
        raise FileNotFoundError(f"Converted file not found: {out_path}")
    return out_path
 
 
# ── Core document processor ───────────────────────────────────────────────────
 
def is_noise_line(text: str) -> bool:
    """Return True if a line should be dropped entirely."""
    if not text:
        return True
    if PAGE_FOOTER_RE.match(text):
        return True
    if REVIEWER_RE.match(text) and len(text.split()) <= REVIEWER_MAX_WORDS:
        return True
    if COMMENTARY_INSTRUCTION_RE.match(text):
        return True
    if LESSON_REF_FOOTER_RE.match(text):
        return True
    if URL_RE.fullmatch(text):
        return True
    if HYPERLINK_COMMENTATOR_RE.match(text):
        return True
    return False
 
 
def is_commentator_credit(text: str) -> bool:
    """
    Return True if a line looks like a standalone commentator attribution.
    e.g. "John Gill", "Matthew Henry", "Dr. Thomas L. Constable"
    These are kept out of the output since they break paragraph flow.
    """
    # Must be short (< 6 words) and match the name pattern
    words = text.split()
    if len(words) > 6:
        return False
    return bool(COMMENTATOR_RE.match(text))
 
 
def format_as_scripture_block(ref: str, verse_text: str) -> str:
    """
    Format a scripture reference + verse text into the chunk_sermons.py
    expected format: [Book Chapter:Verse] verse text
    """
    ref_clean = ref.strip().rstrip('.')
    verse_clean = verse_text.strip()
    if verse_clean:
        return f"[{ref_clean}] {verse_clean}"
    return f"[{ref_clean}]"
 
 
def find_attribution_cutoff(paragraphs) -> int:
    """
    Scan backwards from the end to find where the attribution block starts.
    Returns the index at which to stop processing (exclusive).
    The block is typically 1-3 lines at the very bottom, so we only
    look within the last 10 paragraphs.
    """
    total = len(paragraphs)
    cutoff = total
    window = min(10, total)
 
    for i in range(total - 1, total - window - 1, -1):
        text = get_full_text(paragraphs[i]).strip()
        if not text:
            continue
        if ATTRIBUTION_TRIGGER_RE.match(text):
            cutoff = i  # drop from this line onward
        # Stop once we hit real content that isn't noise
        elif not is_noise_line(text) and not LESSON_REF_FOOTER_RE.match(text):
            break
 
    return cutoff
 
 
def process_document(docx_path: Path, meta: LessonMeta) -> str:
    """
    Parse the docx and return cleaned text in the sermon-compatible format.
    """
    doc = Document(docx_path)
    # Truncate before the attribution block at the bottom
    all_paragraphs = doc.paragraphs
    cutoff = find_attribution_cutoff(all_paragraphs)
    paragraphs = all_paragraphs[:cutoff]
 
    output_blocks = []      # final output paragraphs/blocks
    i = 0
    skip_first_title = True  # first bold line = document title, skip it
 
    while i < len(paragraphs):
        para = paragraphs[i]
        text = get_full_text(para)
 
        # ── Skip image-only paragraphs ──────────────────────────────────────
        if para_has_image(para):
            i += 1
            continue
 
        # ── Skip empty paragraphs ───────────────────────────────────────────
        if not text:
            i += 1
            continue
 
        # ── Skip noise lines ────────────────────────────────────────────────
        if is_noise_line(text):
            i += 1
            continue
 
        # ── Skip commentator credit lines ───────────────────────────────────
        if is_commentator_credit(text):
            i += 1
            continue
 
        # ── Skip the document title (first bold non-empty line) ─────────────
        # Title comes from the filename — no need to include it in body
        if skip_first_title and para_is_bold(para):
            skip_first_title = False
            i += 1
            continue
 
        # ── Detect scripture header + following verse text ───────────────────
        # Pattern: bold paragraph matching "Book Chapter:Verse"
        # followed by one or more non-bold paragraphs containing the verse
        if para_is_bold(para) and SCRIPTURE_HEADER_RE.match(text):
            ref = text
            verse_parts = []
            j = i + 1
 
            # Collect verse text: non-bold paragraphs until next bold or end
            while j < len(paragraphs):
                next_para = paragraphs[j]
                next_text = get_full_text(next_para)
 
                if not next_text:
                    j += 1
                    continue
 
                # Stop if we hit another bold paragraph (next section/ref)
                if para_is_bold(next_para) and next_text:
                    break
 
                # Stop if this line looks like a new section header
                if SECTION_HEADER_RE.match(next_text):
                    break
 
                verse_parts.append(next_text)
                j += 1
 
            verse_text = " ".join(verse_parts)
            output_blocks.append(format_as_scripture_block(ref, verse_text))
            i = j
            continue
 
        # ── Section headers: prepend to next paragraph as context ───────────
        # e.g. "3. King Josiah's Accomplishments" → prepend to body text
        # This keeps chunking consistent (no hard section breaks) while
        # preserving section context inline for retrieval.
        if para_is_bold(para) and (
            SECTION_HEADER_RE.match(text) or
            text in ("Introduction", "Main Body", "Conclusion")
        ):
            section_label = text.strip().rstrip('.')
            # Look ahead for the first non-empty, non-bold body paragraph.
            # Cap at 3 paragraphs — if nothing follows immediately, emit label alone.
            # Without this cap, consecutive bold headers (e.g. items 39-44 in Cain doc)
            # cause an O(n²) walk to end of document for every header in the run.
            j = i + 1
            found_body = False
            while j < min(i + 4, len(paragraphs)):
                next_para = paragraphs[j]
                next_text = get_full_text(next_para)
                if not next_text or para_has_image(next_para):
                    j += 1
                    continue
                # If immediately followed by another bold line, just emit label alone
                if para_is_bold(next_para):
                    break
                # Prepend section label to the body paragraph
                merged = f"[{section_label}] {next_text}"
                output_blocks.append(merged)
                i = j + 1
                found_body = True
                break
 
            if not found_body:
                # No body paragraph follows; emit section label as standalone
                output_blocks.append(f"[{section_label}]")
                i = j
            continue
 
        # ── Sub-section headers (bold, short) ───────────────────────────────
        # e.g. "1) He completely destroyed all the idols."
        # Keep inline as a regular paragraph — they read naturally as prose.
        if para_is_bold(para) and SUBSECTION_RE.match(text):
            output_blocks.append(text)
            i += 1
            continue
 
        # ── Italic / block-quote style verse reference in parens ────────────
        # e.g. "(1King 16:30-33)" as a standalone line before verse text
        inline_ref_match = re.match(
            r'^\((\d?\s?[A-Za-z]+[\w\s]*\d+:\d+[\d\-,\s]*)\)\s*(.*)$', text
        )
        if inline_ref_match:
            ref = inline_ref_match.group(1).strip()
            rest = inline_ref_match.group(2).strip()
            # Collect following verse text if rest is empty
            if not rest:
                j = i + 1
                verse_parts = []
                while j < len(paragraphs):
                    next_para = paragraphs[j]
                    next_text = get_full_text(next_para)
                    if not next_text:
                        j += 1
                        continue
                    if para_is_bold(next_para):
                        break
                    verse_parts.append(next_text)
                    j += 1
                rest = " ".join(verse_parts)
                i = j   # consumed lookahead paragraphs
            else:
                i += 1  # ref+text were on one line, just advance past it
            output_blocks.append(format_as_scripture_block(ref, rest))
            continue
 
        # ── Regular body paragraph ───────────────────────────────────────────
        # Light cleanup: strip stray asterisks, collapse whitespace
        cleaned = re.sub(r'\*+', '', text)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
 
        if cleaned and len(cleaned.split()) >= 3:
            output_blocks.append(cleaned)
 
        i += 1
 
    # ── Assemble final output ─────────────────────────────────────────────────
    header = f"DATE: {meta.date}\nTITLE: {meta.title}\n"
    body = "\n\n".join(output_blocks)
    return header + "\n" + body
 
 
# ── Entry point ───────────────────────────────────────────────────────────────
 
def process_file(
    input_path: Path,
    output_dir: Optional[Path],
    dry_run: bool = False,
    verbose: bool = True,
) -> bool:
    """Process a single .doc or .docx file. Returns True on success."""
 
    meta = parse_filename(input_path.name)
    if not meta:
        print(f"  ⚠️  Skipping — filename doesn't match expected pattern: {input_path.name}")
        return False
 
    if verbose:
        print(f"\n📄  {input_path.name}")
        print(f"    Date: {meta.date}  |  Lesson: {meta.lesson_number}  |  Title: {meta.title}")
 
    tmp_dir = None
    try:
        # Convert .doc → .docx if needed
        if input_path.suffix.lower() == '.doc':
            tmp_dir = Path(tempfile.mkdtemp())
            docx_path = convert_doc_to_docx(input_path, tmp_dir)
        else:
            docx_path = input_path
 
        cleaned = process_document(docx_path, meta)
 
        if dry_run:
            print(f"\n{'─'*60}")
            print(cleaned[:2000])
            if len(cleaned) > 2000:
                print(f"... [{len(cleaned)-2000} more chars]")
            print(f"{'─'*60}")
        else:
            out_file = output_dir / (input_path.stem + "_pure.txt")
            out_file.write_text(cleaned, encoding="utf-8")
            block_count = cleaned.count("\n\n")
            if verbose:
                print(f"    ✅  Written → {out_file.name}  ({block_count} blocks)")
 
        return True
 
    except Exception as e:
        print(f"  ❌  Error processing {input_path.name}: {e}")
        return False
 
    finally:
        if tmp_dir:
            shutil.rmtree(tmp_dir, ignore_errors=True)
 
 
def main():
    parser = argparse.ArgumentParser(
        description="Clean bible lesson .doc/.docx files → sermon-compatible .txt"
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Input folder (or single .doc/.docx file)"
    )
    parser.add_argument(
        "--output", "-o", default="./cleaned_lessons",
        help="Output folder for cleaned .txt files (default: ./cleaned_lessons)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print output to console instead of writing files"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", default=True,
        help="Print progress (default: on)"
    )
    args = parser.parse_args()
 
    input_path = Path(args.input)
    output_dir = Path(args.output)
 
    # Single file mode
    if input_path.is_file():
        if not args.dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)
        process_file(input_path, output_dir, args.dry_run, args.verbose)
        return
 
    # Folder mode
    if not input_path.is_dir():
        print(f"❌  Input path not found: {input_path}")
        sys.exit(1)
 
    files = sorted(
        list(input_path.glob("*.docx")) + list(input_path.glob("*.doc"))
    )
    # Deduplicate: if both .doc and .docx exist for the same stem, prefer .docx
    seen_stems = set()
    deduped = []
    for f in sorted(files, key=lambda x: (x.stem, x.suffix)):
        if f.stem not in seen_stems:
            seen_stems.add(f.stem)
            deduped.append(f)
    files = sorted(deduped)
 
    if not files:
        print(f"No .doc/.docx files found in {input_path}")
        return
 
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
 
    ok, fail = 0, 0
    for f in files:
        success = process_file(f, output_dir, args.dry_run, args.verbose)
        if success:
            ok += 1
        else:
            fail += 1
 
    print(f"\n{'─'*60}")
    print(f"✅  {ok} processed  |  ❌  {fail} failed  |  📁  {output_dir}")
 
 
if __name__ == "__main__":
    main()