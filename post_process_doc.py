import os
import re
from docx import Document
from pathlib import Path
from dateutil import parser

# --- CONFIGURATION ---
SOURCE_DIR = "sermon_documents"
OUTPUT_DIR = "cleaned_sermons"

# Detection Constants
TITLE_BOUNDARY_REGEX = re.compile(r'본\s*문|SCRIPTURES|MAIN\s*TEXT|BIBLE', re.IGNORECASE)
FOOTER_STOP_TAGS = ["TR:", "TRX:", "ENPE:", "KOR:", "EDIT:", "TRANSLATED BY:"]
KOREAN_RE = re.compile(r'[\uac00-\ud7a3\u1100-\u11ff\u3130-\u318f]+')
DATE_EXTRACTOR = re.compile(
    r'(\d{1,2}[\.\/]\d{1,2}[\.\/]\d{4}|(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4})',
    re.IGNORECASE
)
SECTION_LABEL_RE = re.compile(r'^(scriptures|main\s*scriptures|hallelujah)[:\s!]*$', re.IGNORECASE)
SERVICE_HEADER_RE = re.compile(
    r'(sunday|monday|tuesday|wednesday|thursday|friday|saturday)'
    r'[\w\s]*(message|service|worship|prayer)',
    re.IGNORECASE
)
VERSION_TAG_RE = re.compile(r'^rev(ision)?\.?\s*[\d]*(?:\.[\d]+)*$', re.IGNORECASE)

if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)

def normalize_date(date_str):
    try:
        return parser.parse(date_str, fuzzy=True).strftime('%Y-%m-%d')
    except:
        return "Unknown Date"

def clean_noise(text):
    # Preserve scripture references BEFORE stripping brackets
    text = re.sub(r'<([A-Za-z0-9][^>]*\d+:\d+[^>]*)>', r'[\1]', text)
    text = KOREAN_RE.sub('', text)
    text = re.sub(r'\{.*?\}', '', text)
    # Remove <> and decorative chars but KEEP [] for preserved scripture tags
    text = re.sub(r'[<>"\'\u2018\u2019\(\)【】◎◯]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()

def is_substantive(text):
    """Returns True if the string contains actual English words."""
    return any(c.isalpha() for c in text)

def extract_title_from_blob(blob):
    """Extract title from a paragraph blob that may contain date, service header,
    version tag, and title all mixed together. Concatenates multi-point titles."""
    lines = [l.strip() for l in blob.splitlines() if l.strip()]
    title_parts = []
    for line in lines:
        if DATE_EXTRACTOR.search(line):
            continue
        if SERVICE_HEADER_RE.search(line):
            continue
        if VERSION_TAG_RE.match(line.strip()):
            continue
        english_phrases = re.findall(r"[A-Za-z][A-Za-z\s'\-\[\]]*[A-Za-z]", line)
        if english_phrases:
            candidate = max(english_phrases, key=len).strip()
            if is_substantive(candidate):
                title_parts.append(candidate)
    if title_parts:
        return " ".join(title_parts)
    return None

def process_document(file_path):
    print(f"Processing: {file_path.name}")
    try:
        doc = Document(file_path)
        raw_paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]

        extracted_date = "Unknown Date"
        boundary_idx = -1

        # --- STEP 1: Find Date and Scripture Boundary ---
        for i, line in enumerate(raw_paras[:20]):
            d_match = DATE_EXTRACTOR.search(line)
            if d_match and extracted_date == "Unknown Date":
                extracted_date = normalize_date(d_match.group(1))

            if TITLE_BOUNDARY_REGEX.search(line):
                boundary_idx = i
                break

        # --- STEP 2: Extract Title ---
        final_title = "Unknown Title"
        title_parts = []

        NUMBERED_TITLE_RE = re.compile(r'^\d+[\.\)]')  # matches "1." or "1)"

        for i, para in enumerate(raw_paras[:boundary_idx]):
            # Skip version tags
            if VERSION_TAG_RE.match(para.strip()):
                continue
            # Skip date/service header lines
            if DATE_EXTRACTOR.search(para) and SERVICE_HEADER_RE.search(clean_noise(para)):
                continue

            lines = [l.strip() for l in para.splitlines() if l.strip()]
            para_parts = []
            for line in lines:
                if DATE_EXTRACTOR.search(line):
                    continue
                if SERVICE_HEADER_RE.search(line):
                    continue
                if VERSION_TAG_RE.match(line.strip()):
                    continue
                english_phrases = re.findall(r"[A-Za-z][A-Za-z\s'\-\[\]]*[A-Za-z]", line)
                if english_phrases:
                    # Join all phrases to preserve full title including comma-separated clauses
                    candidate = ", ".join(p.strip() for p in english_phrases if is_substantive(p))
                    if candidate:
                        para_parts.append(candidate)

            if para_parts:
                title_parts.extend(para_parts)

        if title_parts:
            final_title = " ".join(title_parts)

        # Fallback: single-word title
        if final_title == "Unknown Title" and raw_paras:
            lines = [l.strip() for l in raw_paras[0].splitlines() if l.strip()]
            for line in lines:
                cleaned = clean_noise(line)
                if not is_substantive(cleaned):
                    continue
                if DATE_EXTRACTOR.search(line):
                    continue
                if SERVICE_HEADER_RE.search(line):
                    continue
                if VERSION_TAG_RE.match(line.strip()):
                    continue
                if re.search(r'[A-Za-z]', cleaned):
                    final_title = cleaned
                    break

        # --- STEP 3: Body Processing ---
        cleaned_body = []
        content_start = boundary_idx if boundary_idx != -1 else 0

        for para in raw_paras[content_start:]:
            if any(para.upper().startswith(tag) for tag in FOOTER_STOP_TAGS):
                break
            cleaned = clean_noise(para)
            if SECTION_LABEL_RE.match(cleaned):   
                continue
            if is_substantive(cleaned):
                cleaned_body.append(cleaned)

        # --- STEP 4: Output ---
        header = f"DATE: {extracted_date}\nTITLE: {final_title}\n\n"
        final_text = header + "\n\n".join(cleaned_body)

        (Path(OUTPUT_DIR) / (file_path.stem + "_pure.txt")).write_text(final_text, encoding='utf-8')
        print(f"   [OK] {extracted_date} | {final_title}")

    except Exception as e:
        print(f"   [ERR] {file_path.name}: {e}")

if __name__ == "__main__":
    files = list(Path(SOURCE_DIR).glob("*.docx"))
    for f in files:
        process_document(f)