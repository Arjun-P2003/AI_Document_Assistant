
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pymupdf


# =====================================================
# PDF CLEANING CONFIGURATION
# =====================================================

MIN_REPEATED_PAGES = 3
REPEATED_PAGE_RATIO = 0.25
MAX_HEADER_FOOTER_LENGTH = 160

# Inspect the first and last few extracted lines.
BOUNDARY_LINE_COUNT = 5


# =====================================================
# SPECIAL CHARACTER CONFIGURATION
# =====================================================

# Characters commonly produced when extracting
# text from academic PDF documents.

CHARACTER_REPLACEMENTS = {
    "\uf0b7": "•",     # Symbol-font bullet
    "\uf0a7": "•",     # Square bullet
    "\uf0d8": "•",     # Symbol-font arrow/bullet
    "\uf0a8": "•",     # Symbol-font bullet
    "\uf020": " ",     # Symbol-font space
    "\ufeff": "",      # Byte-order mark
    "\u200b": "",      # Zero-width space
    "\u200c": "",      # Zero-width non-joiner
    "\u200d": "",      # Zero-width joiner
    "\u2060": "",      # Word joiner
    "\u00ad": "",      # Soft hyphen
    "\u00a0": " ",     # Non-breaking space
    "\u202f": " ",     # Narrow no-break space
    "\u2007": " ",     # Figure space
    "\u2010": "-",    # Hyphen
    "\u2011": "-",    # Non-breaking hyphen
    "\u2212": "-",    # Mathematical minus
    "\ufb01": "fi",   # fi ligature
    "\ufb02": "fl",   # fl ligature
}

# Additional Unicode bullet symbols.
BULLET_CHARACTERS = {
    "•",
    "●",
    "▪",
    "▫",
    "◦",
    "‣",
    "⁃",
}


# =====================================================
# CLEAN SPECIAL CHARACTERS
# =====================================================

def clean_special_characters(text):
    """
    Clean unwanted Unicode and PDF symbol-font
    characters while preserving meaningful text.

    Examples:
        \\uf0b7 -> •
        \\uf020 -> space
        zero-width characters -> removed
    """

    if not text:
        return ""

    text = str(text)

    # Normalize compatible Unicode forms.
    text = unicodedata.normalize("NFKC", text)

    # Replace known PDF extraction artifacts.
    for old, new in CHARACTER_REPLACEMENTS.items():
        text = text.replace(old, new)

    # Normalize remaining common bullet symbols.
    for bullet in BULLET_CHARACTERS:
        if bullet != "•":
            text = text.replace(bullet, "•")

    # Remove private-use characters not explicitly
    # handled above. These often originate from
    # embedded PDF symbol fonts.
    text = "".join(
        character
        for character in text
        if not (
            unicodedata.category(character) == "Co"
        )
    )

    # Remove unwanted control characters while
    # preserving tabs and line breaks.
    text = "".join(
        character
        for character in text
        if (
            character in "\n\r\t"
            or unicodedata.category(character)
            not in {"Cc", "Cf", "Cs"}
        )
    )

    # Normalize line endings.
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Replace tabs with spaces.
    text = text.replace("\t", " ")

    # Normalize spaces around inline bullets.
    text = re.sub(
        r"\s*•\s*",
        " • ",
        text,
    )

    # Remove excessive horizontal whitespace.
    text = re.sub(
        r"[^\S\n]+",
        " ",
        text,
    )

    # Remove trailing whitespace.
    text = "\n".join(
        line.strip()
        for line in text.splitlines()
    )

    # Remove excessive empty lines.
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text,
    )

    return text.strip()


# =====================================================
# TEXT NORMALIZATION
# =====================================================

def normalize_line(line):
    """
    Normalize text for detecting repeated
    page headers and footers.
    """

    line = clean_special_characters(line)

    line = " ".join(line.split())

    # Replace numbers so changing page
    # numbers can match.
    line = re.sub(
        r"\b\d{1,4}\b",
        "#",
        line,
    )

    return line.lower().strip()


def is_page_number(line):
    """
    Check whether a line contains only
    a page number.
    """

    line = clean_special_characters(line)

    return bool(
        re.fullmatch(
            r"(?:page\s*)?\d{1,4}",
            line.strip(),
            flags=re.IGNORECASE,
        )
    )


def is_academic_year(line):
    """
    Detect standalone academic-year text,
    such as '2024 - 2025'.
    """

    line = clean_special_characters(line)

    return bool(
        re.fullmatch(
            r"(?:19|20)\d{2}\s*[-–]\s*(?:19|20)\d{2}",
            line.strip(),
        )
    )


# =====================================================
# IDENTIFY REPEATED HEADERS AND FOOTERS
# =====================================================

def find_repeated_boundary_lines(raw_pages):
    """
    Identify recurring lines near the
    beginning and end of PDF pages.
    """

    line_counts = Counter()

    for page in raw_pages:

        lines = [
            line.strip()
            for line in page["text"].splitlines()
            if line.strip()
        ]

        if not lines:
            continue

        boundary_lines = (
            lines[:BOUNDARY_LINE_COUNT]
            + lines[-BOUNDARY_LINE_COUNT:]
        )

        unique_lines = set()

        for line in boundary_lines:

            if len(line) > MAX_HEADER_FOOTER_LENGTH:
                continue

            normalized = normalize_line(line)

            if normalized:
                unique_lines.add(normalized)

        line_counts.update(unique_lines)

    total_pages = len(raw_pages)

    threshold = max(
        MIN_REPEATED_PAGES,
        int(total_pages * REPEATED_PAGE_RATIO),
    )

    return {
        line
        for line, count in line_counts.items()
        if count >= threshold
    }


# =====================================================
# CLEAN PAGE TEXT
# =====================================================

def clean_page_text(text, repeated_lines):
    """
    Remove repeated headers, footers,
    page numbers, and unwanted symbols
    while preserving document content.
    """

    text = clean_special_characters(text)

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    if not lines:
        return ""

    cleaned_lines = []
    total_lines = len(lines)

    for index, line in enumerate(lines):

        near_boundary = (
            index < BOUNDARY_LINE_COUNT
            or index >= total_lines - BOUNDARY_LINE_COUNT
        )

        if near_boundary:

            normalized = normalize_line(line)

            # Remove repeated header/footer patterns.
            if normalized in repeated_lines:
                continue

            # Remove standalone page numbers.
            if is_page_number(line):
                continue

            # Remove standalone academic-year headers.
            if is_academic_year(line):
                continue

        cleaned_lines.append(line)

    cleaned_text = "\n".join(
        cleaned_lines
    ).strip()

    return cleaned_text


# =====================================================
# EXTRACT PDF TEXT
# =====================================================

def extract_pdf_text(pdf_path):
    """
    Extract and clean text from a PDF.

    Preserves the existing output structure
    expected by the RAG pipeline:

    {
        "filename": str,
        "total_pages": int,
        "full_text": str,
        "pages": [
            {
                "page_number": int,
                "text": str
            }
        ]
    }
    """

    pdf_path = Path(pdf_path)

    # -------------------------------------------------
    # VALIDATE PDF FILE
    # -------------------------------------------------

    if not pdf_path.is_file():
        raise FileNotFoundError(
            f"PDF file not found: {pdf_path}"
        )

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(
            "Only PDF files are supported."
        )

    raw_pages = []

    # -------------------------------------------------
    # STEP 1: EXTRACT RAW PDF TEXT
    # -------------------------------------------------

    with pymupdf.open(pdf_path) as document:

        total_pages = len(document)

        for page_number, page in enumerate(
            document,
            start=1,
        ):

            extracted_text = page.get_text("text")

            # Clean PDF symbols immediately
            # after text extraction.
            cleaned_raw_text = (
                clean_special_characters(
                    extracted_text
                )
            )

            raw_pages.append({
                "page_number": page_number,
                "text": cleaned_raw_text,
            })

    # -------------------------------------------------
    # STEP 2: DETECT REPEATED PAGE CONTENT
    # -------------------------------------------------

    repeated_lines = (
        find_repeated_boundary_lines(
            raw_pages
        )
    )

    print(
        "Repeated header/footer patterns detected:",
        len(repeated_lines),
    )

    # -------------------------------------------------
    # STEP 3: CLEAN EXTRACTED PAGES
    # -------------------------------------------------

    pages = []

    for page in raw_pages:

        cleaned_text = clean_page_text(
            page["text"],
            repeated_lines,
        )

        pages.append({
            "page_number": page["page_number"],
            "text": cleaned_text,
        })

    # -------------------------------------------------
    # STEP 4: COMBINE DOCUMENT TEXT
    # -------------------------------------------------

    full_text = "\n\n".join(
        page["text"]
        for page in pages
        if page["text"]
    )

    # -------------------------------------------------
    # STEP 5: RETURN DOCUMENT INFORMATION
    # -------------------------------------------------

    return {
        "filename": pdf_path.name,
        "total_pages": total_pages,
        "full_text": full_text,
        "pages": pages,
    }


# =====================================================
# END OF PDF READER v2.0
# =====================================================
