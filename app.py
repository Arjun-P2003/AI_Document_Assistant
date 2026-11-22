
import hashlib
import html
import io
import json
import math
import re
import tempfile
from datetime import datetime
from pathlib import Path

import streamlit as st

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    HRFlowable,
    KeepTogether,
    ListFlowable,
    ListItem,
)

from modules.rag_pipeline import RAGPipeline


# =========================================================
# 1. APPLICATION CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Document AI | Intelligent PDF Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_VERSION = "3.6"
TOP_K = 5

SUGGESTED_QUESTIONS = [
    "Summarize the document.",
    "What are the main objectives?",
    "Explain the methodology step by step.",
    "What technologies are used?",
]


# =========================================================
# 2. SESSION STATE
# =========================================================

if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = RAGPipeline()

if "document_info" not in st.session_state:
    st.session_state.document_info = None

if "document_id" not in st.session_state:
    st.session_state.document_id = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "pending_question" not in st.session_state:
    st.session_state.pending_question = None


# =========================================================
# 3. PROFESSIONAL STREAMLIT UI
# =========================================================

st.markdown(
    """
    <style>
    .stApp {
        background: #F5F7FC;
        color: #1E293B;
        font-family: "Inter", "Segoe UI", sans-serif;
    }

    .main .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    section[data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 1px solid #E4EAF3;
    }

    h1, h2, h3, h4, h5, h6 {
        color: #14213D !important;
        font-weight: 750 !important;
    }

    .hero-section {
        background: linear-gradient(
            120deg,
            #142A59 0%,
            #1D4ED8 65%,
            #2563EB 100%
        );
        border-radius: 18px;
        padding: 34px 38px;
        margin-bottom: 25px;
        color: #FFFFFF;
        box-shadow: 0 12px 32px rgba(30,64,175,0.12);
    }

    .hero-eyebrow {
        color: #BFDBFE;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 2px;
        margin-bottom: 12px;
    }

    .hero-title {
        color: #FFFFFF !important;
        font-size: 34px;
        font-weight: 800;
        margin-bottom: 12px;
    }

    .hero-description {
        color: #DBEAFE;
        font-size: 15px;
        line-height: 1.8;
    }

    .section-eyebrow {
        color: #2563EB;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 1.7px;
        margin-bottom: 6px;
    }

    .section-title {
        font-size: 23px;
        font-weight: 750;
        color: #14213D;
        margin-bottom: 6px;
    }

    .section-description {
        color: #64748B;
        font-size: 14px;
        margin-bottom: 20px;
    }

    .info-card {
        background: #FFFFFF;
        border: 1px solid #E3EAF4;
        border-radius: 15px;
        padding: 22px 24px;
        min-height: 135px;
        box-shadow: 0 4px 15px rgba(15,23,42,0.025);
    }

    .info-card-label {
        color: #64748B;
        font-size: 12px;
        font-weight: 650;
        margin-bottom: 12px;
    }

    .info-card-value {
        color: #14213D;
        font-size: 25px;
        font-weight: 800;
        overflow-wrap: anywhere;
    }

    .info-card-description {
        color: #94A3B8;
        font-size: 12px;
        margin-top: 8px;
    }

    .status-ready {
        display: inline-block;
        background: #DCFCE7;
        color: #166534;
        padding: 7px 12px;
        border-radius: 8px;
        font-size: 12px;
        font-weight: 750;
    }

    .status-waiting {
        display: inline-block;
        background: #FEF3C7;
        color: #92400E;
        padding: 7px 12px;
        border-radius: 8px;
        font-size: 12px;
        font-weight: 750;
    }

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 10px;
        font-weight: 650;
        min-height: 43px;
    }

    .stButton > button[kind="primary"] {
        background: #2563EB;
        color: #FFFFFF;
        border: 1px solid #2563EB;
    }

    [data-testid="stFileUploader"] {
        background: #FFFFFF;
        border: 1px solid #E3EAF4;
        border-radius: 14px;
        padding: 12px;
    }

    [data-testid="stChatMessage"] {
        background: #FFFFFF;
        border: 1px solid #E3EAF4;
        border-radius: 14px;
        padding: 18px;
        margin-bottom: 12px;
    }

    [data-testid="stExpander"] {
        background: #FFFFFF;
        border: 1px solid #E3EAF4;
        border-radius: 10px;
    }

    /* Professional AI response formatting */

    [data-testid="stChatMessage"] h3 {
        font-size: 18px !important;
        font-weight: 750 !important;
        color: #1746B8 !important;
        margin-top: 22px !important;
        margin-bottom: 10px !important;
        padding-bottom: 7px;
        border-bottom: 1px solid #E5EAF4;
    }

    [data-testid="stChatMessage"] p {
        font-size: 14px;
        line-height: 1.85;
        color: #334155;
    }

    [data-testid="stChatMessage"] li {
        font-size: 14px;
        line-height: 1.8;
        margin-bottom: 6px;
    }

    [data-testid="stChatMessage"] strong {
        color: #142A59;
        font-weight: 750;
    }

    .custom-footer {
        text-align: center;
        color: #94A3B8;
        font-size: 12px;
        padding: 22px 0 10px;
    }

    @media (max-width: 768px) {
        .hero-section {
            padding: 25px 22px;
        }

        .hero-title {
            font-size: 27px;
        }

        .info-card-value {
            font-size: 21px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 4. HTML HELPERS
# =========================================================

def escape_html(value):
    return html.escape(str(value), quote=True)


def render_section_header(eyebrow, title, description=""):
    st.markdown(
        f"""
        <div class="section-eyebrow">
            {escape_html(eyebrow)}
        </div>
        <div class="section-title">
            {escape_html(title)}
        </div>
        <div class="section-description">
            {escape_html(description)}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_info_card(label, value, description=""):
    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-card-label">
                {escape_html(label)}
            </div>
            <div class="info-card-value">
                {escape_html(value)}
            </div>
            <div class="info-card-description">
                {escape_html(description)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# =========================================================
# 5. DOCUMENT MANAGEMENT
# =========================================================

def reset_document_state():
    st.session_state.document_info = None
    st.session_state.document_id = None
    st.session_state.chat_history = []
    st.session_state.pending_question = None

    pipeline = st.session_state.rag_pipeline
    pipeline.document_loaded = False
    pipeline.document_info = None
    pipeline.document_chunks = []


def process_uploaded_document(
    uploaded_file,
    file_bytes,
    document_id,
):
    temporary_path = None

    try:
        with st.spinner("Processing your PDF document..."):
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf",
            ) as temporary_file:
                temporary_file.write(file_bytes)
                temporary_path = Path(temporary_file.name)

            info = (
                st.session_state.rag_pipeline
                .process_document(temporary_path)
            )

            if not isinstance(info, dict):
                raise ValueError(
                    "Document processing returned invalid data."
                )

            if info.get("total_chunks", 0) <= 0:
                raise ValueError(
                    "No readable text was found in the PDF."
                )

            info["filename"] = uploaded_file.name

            st.session_state.document_info = info
            st.session_state.document_id = document_id
            st.session_state.chat_history = []
            st.session_state.pending_question = None

        st.success("Document processed successfully!")
        st.rerun()

    except Exception as error:
        reset_document_state()
        st.error(f"Document processing failed: {error}")

    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


# =========================================================
# 6. PROFESSIONAL CHAT ANSWER FORMATTING
# =========================================================

SUMMARY_HEADINGS = [
    "Project Overview",
    "Document Overview",
    "Problem Statement",
    "Main Objectives",
    "Objectives",
    "Methodology and Processing Workflow",
    "Methodology",
    "System Architecture",
    "Technologies Used",
    "Results or Expected Outcomes",
    "Expected Outcomes",
    "Results",
    "Conclusion",
    "Future Scope",
    "Limitations",
]


def format_ai_answer(answer):
    """
    Improve AI answer formatting without changing
    the original answer stored in chat history.

    Handles numbered bold headings such as:
        1. **Project Overview**
        2. **Problem Statement**

    Converts them into proper Markdown headings.
    """

    if not answer:
        return ""

    text = str(answer).strip()
    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    headings = {
        heading.casefold()
        for heading in SUMMARY_HEADINGS
    }

    formatted_lines = []
    inside_code_block = False

    for line in text.split("\n"):
        stripped = line.strip()

        # Preserve fenced code blocks.
        if stripped.startswith("```"):
            inside_code_block = not inside_code_block
            formatted_lines.append(line)
            continue

        if inside_code_block:
            formatted_lines.append(line)
            continue

        # Match numbered bold section headings:
        # 1. **Project Overview**
        # 2. **Problem Statement**
        numbered_bold = re.fullmatch(
            r"(\d{1,2})[.)]\s+\*\*(.+?)\*\*:?",
            stripped,
        )

        if numbered_bold:
            number = numbered_bold.group(1)
            heading = numbered_bold.group(2).strip()

            if heading.casefold() in headings:
                if (
                    formatted_lines
                    and formatted_lines[-1].strip()
                ):
                    formatted_lines.append("")

                formatted_lines.append(
                    f"### {number}. {heading}"
                )
                formatted_lines.append("")
                continue

        # Match unnumbered bold headings:
        # **Project Overview**
        bold_heading = re.fullmatch(
            r"\*\*(.+?)\*\*:?",
            stripped,
        )

        if bold_heading:
            heading = bold_heading.group(1).strip()

            if heading.casefold() in headings:
                if (
                    formatted_lines
                    and formatted_lines[-1].strip()
                ):
                    formatted_lines.append("")

                formatted_lines.append(
                    f"### {heading}"
                )
                formatted_lines.append("")
                continue

        # Match plain numbered section headings.
        plain_numbered = re.fullmatch(
            r"(\d{1,2})[.)]\s+(.+?):?",
            stripped,
        )

        if plain_numbered:
            number = plain_numbered.group(1)
            heading = plain_numbered.group(2).strip()

            if heading.casefold() in headings:
                if (
                    formatted_lines
                    and formatted_lines[-1].strip()
                ):
                    formatted_lines.append("")

                formatted_lines.append(
                    f"### {number}. {heading}"
                )
                formatted_lines.append("")
                continue

        # Match standalone plain headings.
        if stripped.rstrip(":").casefold() in headings:
            heading = stripped.rstrip(":")

            if (
                formatted_lines
                and formatted_lines[-1].strip()
            ):
                formatted_lines.append("")

            formatted_lines.append(
                f"### {heading}"
            )
            formatted_lines.append("")
            continue

        # Normalize indented bullets returned by the LLM.
        bullet_match = re.match(
            r"^\s*[-*+•]\s+(.+)$",
            line,
        )

        if bullet_match:
            formatted_lines.append(
                f"- {bullet_match.group(1)}"
            )
            continue

        # Remove unnecessary indentation from
        # paragraphs under numbered headings.
        if (
            line.startswith("   ")
            and stripped
            and not re.match(r"^\d+[.)]\s+", stripped)
        ):
            line = stripped

        formatted_lines.append(line)

    formatted_text = "\n".join(
        formatted_lines
    ).strip()

    # Remove excessive blank lines.
    formatted_text = re.sub(
        r"\n{3,}",
        "\n\n",
        formatted_text,
    )

    return formatted_text


def display_formatted_answer(answer):
    formatted = format_ai_answer(answer)
    st.markdown(formatted)


# =========================================================
# END OF PART 1
# =========================================================

# =========================================================
# 7. EXPORT DATA
# =========================================================

def clean_json_value(value):
    if value is None or isinstance(value, (str, int, bool)):
        return value

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    if isinstance(value, dict):
        return {
            str(key): clean_json_value(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [clean_json_value(item) for item in value]

    if hasattr(value, "item"):
        return clean_json_value(value.item())

    return str(value)


def build_export_data():
    return clean_json_value({
        "application": "AI Document Assistant",
        "version": APP_VERSION,
        "exported_at": datetime.now().astimezone().isoformat(),
        "document": st.session_state.document_info or {},
        "total_questions": len(st.session_state.chat_history),
        "conversation": st.session_state.chat_history,
    })


# =========================================================
# 8. JSON EXPORT
# =========================================================

def export_chat_json():
    return json.dumps(
        build_export_data(),
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


# =========================================================
# 9. TXT EXPORT
# =========================================================

def export_chat_txt():
    data = build_export_data()
    document = data["document"]

    lines = [
        "AI DOCUMENT ASSISTANT",
        "CHAT HISTORY REPORT",
        "=" * 65,
        "",
        f"Version: {APP_VERSION}",
        f"Exported: {data['exported_at']}",
        f"Document: {document.get('filename', 'Unknown')}",
        f"Questions: {data['total_questions']}",
        "",
        "=" * 65,
        "",
    ]

    for index, chat in enumerate(
        data["conversation"],
        start=1,
    ):
        lines.extend([
            f"QUESTION {index}",
            "-" * 65,
            str(chat.get("question", "")),
            "",
            "ANSWER",
            "-" * 65,
            str(chat.get("answer", "")),
            "",
            "DOCUMENT SOURCES",
            "-" * 65,
        ])

        sources = chat.get("sources", [])

        if not sources:
            lines.append("No sources available.")

        for source_index, source in enumerate(
            sources,
            start=1,
        ):
            page = source.get(
                "page_number",
                "Unknown",
            )

            chunk = source.get(
                "chunk_number",
                "Unknown",
            )

            lines.extend([
                (
                    f"Source {source_index}: "
                    f"Page {page}, Chunk {chunk}"
                ),
                str(source.get("text", "")),
                "",
            ])

        lines.extend([
            "",
            "=" * 65,
            "",
        ])

    return "\n".join(lines).encode("utf-8")


# =========================================================
# 10. PDF FONT AND TEXT HELPERS
# =========================================================

PDF_FONT = "Helvetica"
PDF_FONT_BOLD = "Helvetica-Bold"


def configure_pdf_fonts():
    global PDF_FONT, PDF_FONT_BOLD

    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont

    font_candidates = [
        (
            "DejaVuSans",
            "DejaVuSans-Bold",
            [
                Path("C:/Windows/Fonts/DejaVuSans.ttf"),
                Path(
                    "/usr/share/fonts/truetype/"
                    "dejavu/DejaVuSans.ttf"
                ),
            ],
            [
                Path(
                    "C:/Windows/Fonts/"
                    "DejaVuSans-Bold.ttf"
                ),
                Path(
                    "/usr/share/fonts/truetype/"
                    "dejavu/DejaVuSans-Bold.ttf"
                ),
            ],
        ),
        (
            "Arial",
            "Arial-Bold",
            [
                Path("C:/Windows/Fonts/arial.ttf"),
            ],
            [
                Path("C:/Windows/Fonts/arialbd.ttf"),
            ],
        ),
    ]

    for (
        regular_name,
        bold_name,
        regular_paths,
        bold_paths,
    ) in font_candidates:

        regular_path = next(
            (
                path
                for path in regular_paths
                if path.is_file()
            ),
            None,
        )

        bold_path = next(
            (
                path
                for path in bold_paths
                if path.is_file()
            ),
            None,
        )

        if not regular_path or not bold_path:
            continue

        try:
            registered = (
                pdfmetrics.getRegisteredFontNames()
            )

            if regular_name not in registered:
                pdfmetrics.registerFont(
                    TTFont(
                        regular_name,
                        str(regular_path),
                    )
                )

            if bold_name not in registered:
                pdfmetrics.registerFont(
                    TTFont(
                        bold_name,
                        str(bold_path),
                    )
                )

            pdfmetrics.registerFontFamily(
                regular_name,
                normal=regular_name,
                bold=bold_name,
                italic=regular_name,
                boldItalic=bold_name,
            )

            PDF_FONT = regular_name
            PDF_FONT_BOLD = bold_name
            return

        except Exception:
            continue


def pdf_safe_text(value):
    text = str(value)

    text = text.replace("\u00a0", " ")
    text = text.replace("\uf0b7", "•")
    text = text.replace("\uf0a7", "•")
    text = text.replace("\u2028", " ")
    text = text.replace("\u2029", "\n")

    if PDF_FONT == "Helvetica":
        replacements = {
            "•": "-",
            "–": "-",
            "—": "-",
            "‘": "'",
            "’": "'",
            "“": '"',
            "”": '"',
            "→": "->",
            "≥": ">=",
            "≤": "<=",
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        text = text.encode(
            "latin-1",
            errors="replace",
        ).decode("latin-1")

    return html.escape(
        text,
        quote=True,
    )


def markdown_to_pdf_markup(text):
    escaped = pdf_safe_text(text)

    escaped = re.sub(
        r"\*\*(.+?)\*\*",
        r"<b>\1</b>",
        escaped,
    )

    escaped = re.sub(
        r"__(.+?)__",
        r"<b>\1</b>",
        escaped,
    )

    escaped = re.sub(
        r"(?<!\*)\*([^*\n]+)\*(?!\*)",
        r"<i>\1</i>",
        escaped,
    )

    escaped = re.sub(
        r"`([^`\n]+)`",
        r'<font color="#1D4ED8">\1</font>',
        escaped,
    )

    return escaped


# =========================================================
# 11. PROFESSIONAL PDF STYLES
# =========================================================

def create_pdf_styles():
    styles = getSampleStyleSheet()

    navy = colors.HexColor("#142A59")
    blue = colors.HexColor("#1D4ED8")
    body_color = colors.HexColor("#263449")
    muted = colors.HexColor("#64748B")

    definitions = [
        ParagraphStyle(
            name="CustomTitle",
            fontName=PDF_FONT_BOLD,
            fontSize=21,
            leading=28,
            textColor=navy,
            alignment=TA_CENTER,
            spaceAfter=8,
        ),
        ParagraphStyle(
            name="CustomSubtitle",
            fontName=PDF_FONT,
            fontSize=10,
            leading=16,
            textColor=muted,
            alignment=TA_CENTER,
            spaceAfter=20,
        ),
        ParagraphStyle(
            name="CustomSection",
            fontName=PDF_FONT_BOLD,
            fontSize=13,
            leading=19,
            textColor=blue,
            spaceBefore=15,
            spaceAfter=9,
            keepWithNext=True,
        ),
        ParagraphStyle(
            name="CustomAnswerHeading",
            fontName=PDF_FONT_BOLD,
            fontSize=11,
            leading=16,
            textColor=navy,
            spaceBefore=13,
            spaceAfter=7,
            keepWithNext=True,
        ),
        ParagraphStyle(
            name="CustomSubheading",
            fontName=PDF_FONT_BOLD,
            fontSize=10,
            leading=15,
            textColor=navy,
            spaceBefore=9,
            spaceAfter=5,
            keepWithNext=True,
        ),
        ParagraphStyle(
            name="CustomBody",
            fontName=PDF_FONT,
            fontSize=9.5,
            leading=15.5,
            textColor=body_color,
            spaceAfter=7,
            alignment=TA_LEFT,
        ),
        ParagraphStyle(
            name="CustomBullet",
            fontName=PDF_FONT,
            fontSize=9.5,
            leading=15,
            textColor=body_color,
            leftIndent=12,
            spaceAfter=5,
        ),
        ParagraphStyle(
            name="CustomMetadata",
            fontName=PDF_FONT,
            fontSize=9,
            leading=15,
            textColor=body_color,
            spaceAfter=5,
        ),
        ParagraphStyle(
            name="CustomSourceHeading",
            fontName=PDF_FONT_BOLD,
            fontSize=9.5,
            leading=15,
            textColor=blue,
            spaceBefore=11,
            spaceAfter=5,
            keepWithNext=True,
        ),
        ParagraphStyle(
            name="CustomSourceBody",
            fontName=PDF_FONT,
            fontSize=8.5,
            leading=13.5,
            textColor=muted,
            leftIndent=10,
            rightIndent=6,
            spaceAfter=9,
        ),
    ]

    for style in definitions:
        styles.add(style)

    return styles


# =========================================================
# 12. MARKDOWN-AWARE PDF CONTENT
# =========================================================

def add_pdf_text(
    story,
    text,
    styles,
    source_mode=False,
):
    body_style = (
        styles["CustomSourceBody"]
        if source_mode
        else styles["CustomBody"]
    )

    lines = str(text).splitlines()

    paragraph_buffer = []
    list_buffer = []
    list_type = None

    def flush_paragraph():
        nonlocal paragraph_buffer

        if paragraph_buffer:
            content = " ".join(
                paragraph_buffer
            ).strip()

            if content:
                story.append(
                    Paragraph(
                        markdown_to_pdf_markup(
                            content
                        ),
                        body_style,
                    )
                )

            paragraph_buffer = []

    def flush_list():
        nonlocal list_buffer, list_type

        if not list_buffer:
            return

        items = [
            ListItem(
                Paragraph(
                    markdown_to_pdf_markup(item),
                    styles["CustomBullet"],
                ),
                leftIndent=4,
            )
            for item in list_buffer
        ]

        story.append(
            ListFlowable(
                items,
                bulletType=(
                    "1"
                    if list_type == "numbered"
                    else "bullet"
                ),
                start=(
                    "1"
                    if list_type == "numbered"
                    else None
                ),
                leftIndent=18,
                bulletFontName=PDF_FONT,
                bulletFontSize=8,
                spaceAfter=7,
            )
        )

        list_buffer = []
        list_type = None

    for raw_line in lines:
        line = raw_line.strip()

        if not line:
            flush_paragraph()
            flush_list()
            continue

        heading_match = re.match(
            r"^(#{1,6})\s+(.+)$",
            line,
        )

        numbered_heading = re.match(
            r"^(\d{1,2})\.\s+\*\*(.+?)\*\*\s*$",
            line,
        )

        plain_numbered_heading = re.match(
            r"^(\d{1,2})\.\s+"
            r"(Project Overview|Problem Statement|"
            r"Main Objectives|Methodology.*|"
            r"Technologies Used|Results.*|"
            r"Conclusion|Document Overview)\s*$",
            line,
            flags=re.IGNORECASE,
        )

        if heading_match:
            flush_paragraph()
            flush_list()

            level = len(
                heading_match.group(1)
            )

            heading_text = (
                heading_match.group(2)
            )

            style = (
                styles["CustomAnswerHeading"]
                if level <= 3
                else styles["CustomSubheading"]
            )

            story.append(
                Paragraph(
                    markdown_to_pdf_markup(
                        heading_text
                    ),
                    style,
                )
            )
            continue

        if (
            numbered_heading
            or plain_numbered_heading
        ):
            flush_paragraph()
            flush_list()

            match = (
                numbered_heading
                or plain_numbered_heading
            )

            heading_text = (
                f"{match.group(1)}. "
                f"{match.group(2)}"
            )

            story.append(
                Paragraph(
                    markdown_to_pdf_markup(
                        heading_text
                    ),
                    styles["CustomAnswerHeading"],
                )
            )
            continue

        bullet_match = re.match(
            r"^(?:[-*+•])\s+(.+)$",
            line,
        )

        numbered_match = re.match(
            r"^\d+[.)]\s+(.+)$",
            line,
        )

        if bullet_match or numbered_match:
            flush_paragraph()

            current_type = (
                "numbered"
                if numbered_match
                else "bullet"
            )

            if (
                list_buffer
                and list_type != current_type
            ):
                flush_list()

            list_type = current_type

            item_text = (
                numbered_match.group(1)
                if numbered_match
                else bullet_match.group(1)
            )

            list_buffer.append(item_text)
            continue

        flush_list()
        paragraph_buffer.append(line)

    flush_paragraph()
    flush_list()


# =========================================================
# 13. PDF HEADER AND FOOTER
# =========================================================

def add_page_number(canvas, document):
    canvas.saveState()

    width, height = document.pagesize

    canvas.setStrokeColor(
        colors.HexColor("#DCE5F2")
    )

    canvas.setLineWidth(0.6)

    canvas.line(
        48,
        height - 36,
        width - 48,
        height - 36,
    )

    canvas.setFont(PDF_FONT, 8)

    canvas.setFillColor(
        colors.HexColor("#64748B")
    )

    canvas.drawString(
        48,
        height - 27,
        "AI DOCUMENT ASSISTANT",
    )

    canvas.line(
        48,
        42,
        width - 48,
        42,
    )

    canvas.drawString(
        48,
        28,
        "AI-Powered Document Analysis",
    )

    canvas.drawRightString(
        width - 48,
        28,
        f"Page {document.page}",
    )

    canvas.restoreState()


# =========================================================
# 14. PDF REPORT GENERATION
# =========================================================

def export_chat_pdf():
    configure_pdf_fonts()

    data = build_export_data()
    document_info = data["document"]

    buffer = io.BytesIO()

    pdf = SimpleDocTemplate(
        buffer,
        pagesize=(595.28, 841.89),
        rightMargin=48,
        leftMargin=48,
        topMargin=55,
        bottomMargin=60,
        title=(
            "AI Document Assistant "
            "- Chat Report"
        ),
        author="AI Document Assistant",
    )

    styles = create_pdf_styles()
    story = []

    story.append(
        Paragraph(
            "AI DOCUMENT ASSISTANT",
            styles["CustomTitle"],
        )
    )

    story.append(
        Paragraph(
            "AI-Powered Document Analysis Report",
            styles["CustomSubtitle"],
        )
    )

    story.append(
        HRFlowable(
            width="100%",
            thickness=1,
            color=colors.HexColor(
                "#DCE5F2"
            ),
        )
    )

    story.append(Spacer(1, 16))

    metadata = [
        (
            "Document",
            document_info.get(
                "filename",
                "Unknown",
            ),
        ),
        (
            "Total Pages",
            document_info.get(
                "total_pages",
                "Unknown",
            ),
        ),
        (
            "Text Chunks",
            document_info.get(
                "total_chunks",
                "Unknown",
            ),
        ),
        (
            "Total Questions",
            data["total_questions"],
        ),
        (
            "Exported At",
            data["exported_at"],
        ),
    ]

    for label, value in metadata:
        story.append(
            Paragraph(
                (
                    f"<b>{pdf_safe_text(label)}:</b> "
                    f"{pdf_safe_text(value)}"
                ),
                styles["CustomMetadata"],
            )
        )

    story.append(Spacer(1, 15))

    story.append(
        HRFlowable(
            width="100%",
            thickness=1,
            color=colors.HexColor(
                "#DCE5F2"
            ),
        )
    )

    for index, chat in enumerate(
        data["conversation"],
        start=1,
    ):
        question = str(
            chat.get("question", "")
        )

        answer = str(
            chat.get("answer", "")
        )

        sources = chat.get(
            "sources",
            [],
        )

        story.append(
            KeepTogether([
                Paragraph(
                    f"Question {index}",
                    styles["CustomSection"],
                ),
                Paragraph(
                    pdf_safe_text(question),
                    styles["CustomBody"],
                ),
            ])
        )

        story.append(
            Paragraph(
                "AI Answer",
                styles["CustomSection"],
            )
        )

        # Apply professional formatting.
        add_pdf_text(
            story,
            format_ai_answer(answer),
            styles,
        )

        story.append(Spacer(1, 10))

        story.append(
            HRFlowable(
                width="100%",
                thickness=0.6,
                color=colors.HexColor(
                    "#E2E8F0"
                ),
            )
        )

        story.append(
            Paragraph(
                "Document References",
                styles["CustomSection"],
            )
        )

        if not sources:
            story.append(
                Paragraph(
                    "No document sources available.",
                    styles["CustomSourceBody"],
                )
            )

        for source_index, source in enumerate(
            sources,
            start=1,
        ):
            page = source.get(
                "page_number",
                "Unknown",
            )

            chunk = source.get(
                "chunk_number",
                "Unknown",
            )

            source_heading = (
                f"Source {source_index}"
                f" | Page {page}"
                f" | Chunk {chunk}"
            )

            story.append(
                Paragraph(
                    pdf_safe_text(
                        source_heading
                    ),
                    styles["CustomSourceHeading"],
                )
            )

            add_pdf_text(
                story,
                source.get("text", ""),
                styles,
                source_mode=True,
            )

            if source_index < len(sources):
                story.append(
                    HRFlowable(
                        width="100%",
                        thickness=0.4,
                        color=colors.HexColor(
                            "#E2E8F0"
                        ),
                        spaceBefore=4,
                        spaceAfter=6,
                    )
                )

        story.append(Spacer(1, 15))

        story.append(
            HRFlowable(
                width="100%",
                thickness=0.7,
                color=colors.HexColor(
                    "#DCE5F2"
                ),
            )
        )

    pdf.build(
        story,
        onFirstPage=add_page_number,
        onLaterPages=add_page_number,
    )

    buffer.seek(0)
    return buffer.getvalue()


# =========================================================
# END OF PART 2
# =========================================================

# =========================================================
# 15. DISPLAY DOCUMENT SOURCES
# =========================================================

def display_sources(sources):
    if not sources:
        st.caption("No document sources were returned.")
        return

    for index, source in enumerate(sources, start=1):
        page = source.get("page_number", "Unknown")
        chunk = source.get("chunk_number", "Unknown")
        similarity = source.get("similarity_score")
        source_text = source.get("text", "")

        st.markdown(
            f"**Source {index} · Page {page}**"
        )

        col1, col2 = st.columns(2)

        with col1:
            st.caption(f"Chunk {chunk}")

        with col2:
            if isinstance(similarity, (int, float)):
                st.caption(
                    f"Semantic similarity: {similarity:.2%}"
                )

        st.write(source_text)

        if index < len(sources):
            st.divider()


# =========================================================
# 16. CHAT HISTORY
# =========================================================

def display_chat_history():
    for chat in st.session_state.chat_history:
        with st.chat_message("user", avatar="👤"):
            st.write(chat["question"])

        with st.chat_message("assistant", avatar="🤖"):
            display_formatted_answer(chat["answer"])

            with st.expander("View Document Sources"):
                display_sources(
                    chat.get("sources", [])
                )


# =========================================================
# 17. GENERATE AI ANSWER
# =========================================================

def generate_answer(question):
    question = question.strip()

    if not question:
        return

    with st.chat_message("user", avatar="👤"):
        st.write(question)

    with st.chat_message("assistant", avatar="🤖"):
        try:
            with st.spinner("Searching your document..."):
                result = (
                    st.session_state.rag_pipeline
                    .ask_question(
                        question=question,
                        top_k=TOP_K,
                    )
                )

            answer = result.get("answer", "")
            sources = result.get("sources", [])

            if not isinstance(answer, str):
                answer = str(answer)

            if not answer.strip():
                answer = (
                    "The AI model did not return an answer. "
                    "Please try again."
                )

            display_formatted_answer(answer)

            with st.expander("View Document Sources"):
                display_sources(sources)

            st.session_state.chat_history.append({
                "question": question,
                "answer": answer,
                "sources": sources,
            })

            st.rerun()

        except Exception as error:
            st.error(
                f"Unable to generate answer: {error}"
            )


# =========================================================
# 18. SIDEBAR
# =========================================================

with st.sidebar:
    st.markdown("## 📄 Document AI")
    st.caption("INTELLIGENT DOCUMENT WORKSPACE")

    st.divider()
    st.markdown("### Workspace")

    if st.session_state.document_info:
        st.markdown(
            '<span class="status-ready">'
            '● Document Ready'
            '</span>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<span class="status-waiting">'
            '● Waiting for Document'
            '</span>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.markdown("### AI Engine")

    st.markdown("**Language Model**")
    st.caption("Qwen 2.5 · 3B Parameters")

    st.markdown("**Embedding Model**")
    st.caption("all-MiniLM-L6-v2")

    st.markdown("**Vector Database**")
    st.caption("FAISS · Hybrid Retrieval")

    st.markdown("**Answer Sources**")
    st.caption(
        f"Up to {TOP_K} chunks for regular questions; "
        "expanded coverage for summaries"
    )

    st.divider()
    st.markdown("### Document Status")

    if st.session_state.document_info:
        info = st.session_state.document_info

        st.caption(
            info.get("filename", "Unknown Document")
        )

        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Pages",
                info.get("total_pages", 0),
            )

        with col2:
            st.metric(
                "Chunks",
                info.get("total_chunks", 0),
            )
    else:
        st.caption("Upload a PDF to begin.")

    st.divider()
    st.markdown("### Conversation")

    st.caption(
        f"{len(st.session_state.chat_history)} "
        "questions asked"
    )

    if st.button(
        "Clear Conversation",
        use_container_width=True,
    ):
        st.session_state.chat_history = []
        st.session_state.pending_question = None
        st.rerun()

    # -----------------------------------------------------
    # EXPORT CONVERSATION
    # -----------------------------------------------------

    if st.session_state.chat_history:
        st.divider()
        st.markdown("### Export Conversation")

        st.caption(
            "Download your questions, answers, "
            "and document references."
        )

        document_name = (
            st.session_state.document_info.get(
                "filename",
                "document.pdf",
            )
            if st.session_state.document_info
            else "document.pdf"
        )

        export_name = Path(document_name).stem

        st.download_button(
            label="Download Chat History (TXT)",
            data=export_chat_txt(),
            file_name=f"{export_name}_chat_history.txt",
            mime="text/plain",
            use_container_width=True,
        )

        st.download_button(
            label="Download Chat History (JSON)",
            data=export_chat_json(),
            file_name=f"{export_name}_chat_history.json",
            mime="application/json",
            use_container_width=True,
        )

        try:
            pdf_bytes = export_chat_pdf()

            st.download_button(
                label="Download PDF Report",
                data=pdf_bytes,
                file_name=f"{export_name}_AI_report.pdf",
                mime="application/pdf",
                use_container_width=True,
                type="primary",
            )

        except Exception as error:
            st.error(
                f"PDF report generation failed: {error}"
            )

    st.divider()
    st.caption(f"Version {APP_VERSION}")
    st.caption("Private · Local AI Processing")


# =========================================================
# 19. HERO SECTION
# =========================================================

st.markdown(
    """
    <div class="hero-section">
        <div class="hero-eyebrow">
            AI-POWERED KNOWLEDGE ASSISTANT
        </div>
        <div class="hero-title">
            Chat with your documents.
        </div>
        <div class="hero-description">
            Transform your PDFs into searchable knowledge.
            Upload a document, ask questions in natural
            language, and receive AI-generated answers
            grounded in your document content.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 20. DOCUMENT UPLOAD
# =========================================================

render_section_header(
    "01 · DOCUMENT WORKSPACE",
    "Upload Your Document",
    "Select a PDF to begin intelligent document analysis.",
)

uploaded_file = st.file_uploader(
    "Choose a PDF document",
    type=["pdf"],
    help=(
        "Text-based PDFs are supported. "
        "Scanned PDFs may require OCR."
    ),
)

document_id = None

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()

    document_id = hashlib.sha256(
        file_bytes
    ).hexdigest()

    is_current_document = (
        document_id == st.session_state.document_id
        and st.session_state.document_info is not None
    )

    if is_current_document:
        st.success(
            "Document indexed and ready for questions."
        )

        with st.expander("Advanced Document Options"):
            st.caption(
                "Reprocessing rebuilds the FAISS index "
                "for this document."
            )

            if st.button(
                "Reprocess Document",
                use_container_width=True,
            ):
                process_uploaded_document(
                    uploaded_file,
                    file_bytes,
                    document_id,
                )
    else:
        st.info(
            "Your PDF is selected. Process it to "
            "enable AI-powered question answering."
        )

        if st.button(
            "Process Document",
            type="primary",
            use_container_width=True,
        ):
            process_uploaded_document(
                uploaded_file,
                file_bytes,
                document_id,
            )
else:
    st.caption(
        "Supported file type: PDF · "
        "Text extraction powered by PyMuPDF"
    )


# =========================================================
# 21. DOCUMENT OVERVIEW
# =========================================================

if st.session_state.document_info:
    info = st.session_state.document_info

    st.divider()

    render_section_header(
        "02 · DOCUMENT INTELLIGENCE",
        "Document Overview",
        "Your document has been indexed and is ready.",
    )

    col1, col2, col3 = st.columns([1.5, 1, 1])

    with col1:
        render_info_card(
            "ACTIVE DOCUMENT",
            info.get("filename", "Unknown"),
            "Current PDF knowledge source",
        )

    with col2:
        render_info_card(
            "TOTAL PAGES",
            info.get("total_pages", 0),
            "Pages extracted",
        )

    with col3:
        render_info_card(
            "TEXT CHUNKS",
            info.get("total_chunks", 0),
            "Searchable passages",
        )


# =========================================================
# 22. AI QUESTION ANSWERING
# =========================================================

st.divider()

render_section_header(
    "03 · AI CONVERSATION",
    "Ask Your Document",
    "Get answers supported by relevant PDF passages.",
)

document_ready = (
    st.session_state.document_info is not None
    and uploaded_file is not None
    and document_id == st.session_state.document_id
    and st.session_state.rag_pipeline.document_loaded
)

if not document_ready:
    st.info(
        "Upload and process a PDF document "
        "to activate the AI assistant."
    )
else:
    if not st.session_state.chat_history:
        st.markdown("#### Suggested Questions")

        st.caption(
            "Select a question or write your own below."
        )

        suggestion_cols = st.columns(2)

        for index, suggestion in enumerate(
            SUGGESTED_QUESTIONS
        ):
            with suggestion_cols[index % 2]:
                if st.button(
                    suggestion,
                    key=f"suggestion_{index}",
                    use_container_width=True,
                ):
                    st.session_state.pending_question = (
                        suggestion
                    )
                    st.rerun()

        st.write("")

    display_chat_history()

    question = st.chat_input(
        "Ask anything about your document..."
    )

    if st.session_state.pending_question:
        question = st.session_state.pending_question
        st.session_state.pending_question = None

    if question and question.strip():
        generate_answer(question)


# =========================================================
# 23. FOOTER
# =========================================================

st.divider()

st.markdown(
    f"""
    <div class="custom-footer">
        Document AI v{APP_VERSION}
        &nbsp;·&nbsp;
        Powered by Streamlit, FAISS,
        Sentence Transformers and Ollama
        <div>
            Intelligent Document Analysis
            &nbsp;·&nbsp;
            Local AI Processing
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# END OF PART 3
# COMPLETE APPLICATION — DOCUMENT AI v3.6
# =========================================================
