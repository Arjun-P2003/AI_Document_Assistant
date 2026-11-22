
import re


def normalize_text(text):
    """
    Clean PDF text while preserving paragraph boundaries.
    """

    if not isinstance(text, str):
        raise TypeError("Text must be a string.")

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Remove extra spaces and tabs.
    text = re.sub(r"[ \t]+", " ", text)

    # Join lines within paragraphs.
    paragraphs = re.split(r"\n\s*\n", text)

    cleaned_paragraphs = []

    for paragraph in paragraphs:
        cleaned = " ".join(paragraph.split())

        if cleaned:
            cleaned_paragraphs.append(cleaned)

    return "\n\n".join(cleaned_paragraphs)


def split_into_sentences(text):
    """
    Split text into sentences using punctuation.
    """

    if not text:
        return []

    sentences = re.split(
        r"(?<=[.!?])\s+",
        text
    )

    return [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]


def split_long_text(text, chunk_size):
    """
    Split long text at word boundaries when possible.
    """

    words = text.split()
    chunks = []
    current_words = []
    current_length = 0

    for word in words:
        extra_length = len(word) + (1 if current_words else 0)

        if current_words and current_length + extra_length > chunk_size:
            chunks.append(" ".join(current_words))
            current_words = []
            current_length = 0

        # Handle individual words longer than chunk_size.
        if len(word) > chunk_size:
            if current_words:
                chunks.append(" ".join(current_words))
                current_words = []
                current_length = 0

            for start in range(0, len(word), chunk_size):
                chunks.append(word[start:start + chunk_size])

            continue

        current_words.append(word)
        current_length += len(word) + (1 if current_length else 0)

    if current_words:
        chunks.append(" ".join(current_words))

    return chunks


def chunk_text(text, chunk_size=800, overlap=150):
    """
    Split text into sentence-aware overlapping chunks.

    Args:
        text: Input document text.
        chunk_size: Maximum characters per chunk.
        overlap: Approximate overlapping characters.

    Returns:
        List of text chunks.
    """

    if not isinstance(text, str):
        raise TypeError("Text must be a string.")

    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError(
            "chunk_size must be a positive integer."
        )

    if not isinstance(overlap, int) or overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size."
        )

    text = normalize_text(text)

    if not text:
        return []

    # Preserve paragraph boundaries during initial splitting.
    paragraphs = text.split("\n\n")
    units = []

    for paragraph in paragraphs:
        sentences = split_into_sentences(paragraph)

        for sentence in sentences:
            if len(sentence) <= chunk_size:
                units.append(sentence)
            else:
                units.extend(
                    split_long_text(sentence, chunk_size)
                )

    chunks = []
    current_units = []
    current_length = 0

    def save_current_chunk():
        if current_units:
            chunk = " ".join(current_units).strip()

            if chunk and (not chunks or chunk != chunks[-1]):
                chunks.append(chunk)

    for unit in units:
        additional_length = len(unit) + (
            1 if current_units else 0
        )

        if current_units and current_length + additional_length > chunk_size:
            save_current_chunk()

            # Keep complete trailing units as overlap.
            overlap_units = []
            overlap_length = 0

            for previous_unit in reversed(current_units):
                required = len(previous_unit) + (
                    1 if overlap_units else 0
                )

                if overlap_length + required > overlap:
                    break

                overlap_units.insert(0, previous_unit)
                overlap_length += required

            # Avoid overlap that prevents the next unit fitting.
            while overlap_units and (
                len(" ".join(overlap_units))
                + 1
                + len(unit)
                > chunk_size
            ):
                overlap_units.pop(0)

            current_units = overlap_units
            current_length = len(" ".join(current_units))

        if current_units:
            current_length += 1

        current_units.append(unit)
        current_length += len(unit)

    save_current_chunk()

    return chunks


def chunk_pdf_pages(pages, chunk_size=800, overlap=150):
    """
    Chunk PDF pages while preserving page numbers
    and chunk metadata.
    """

    if not isinstance(pages, list):
        raise TypeError("pages must be a list.")

    all_chunks = []

    for page in pages:
        page_number = page["page_number"]
        page_text = page.get("text", "")

        if not page_text or not page_text.strip():
            continue

        page_chunks = chunk_text(
            page_text,
            chunk_size=chunk_size,
            overlap=overlap
        )

        for index, chunk in enumerate(page_chunks, start=1):
            all_chunks.append({
                "page_number": page_number,
                "chunk_number": index,
                "text": chunk
            })

    return all_chunks
