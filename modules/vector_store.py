
import json
import re
from pathlib import Path

import faiss
import numpy as np


# =====================================================
# VECTOR DATABASE CONFIGURATION
# =====================================================

VECTOR_STORE_DIR = Path("vector_store")

INDEX_PATH = VECTOR_STORE_DIR / "faiss.index"
METADATA_PATH = VECTOR_STORE_DIR / "metadata.json"

DEFAULT_DIMENSION = 384

MIN_USEFUL_WORDS = 8
MIN_USEFUL_CHARACTERS = 60

CANDIDATE_MULTIPLIER = 10

# Hybrid ranking weights
SEMANTIC_WEIGHT = 0.45
KEYWORD_WEIGHT = 0.15
QUALITY_WEIGHT = 0.10
SECTION_WEIGHT = 0.30

FRONT_MATTER_PENALTY = 0.12
UNRELATED_SECTION_PENALTY = 0.10

# Minimum number of meaningful question terms
# required before keyword overlap receives full credit.
MIN_KEYWORD_TERMS = 3

# Methodology workflow scoring
WORKFLOW_WEIGHT = 0.22

# Conservative penalty for recognizable research tables
LITERATURE_TABLE_PENALTY = 0.12


# =====================================================
# SEARCH CONFIGURATION
# =====================================================

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at",
    "be", "by", "can", "do", "does", "for",
    "from", "how", "in", "is", "it", "its",
    "of", "on", "or", "the", "their",
    "this", "to", "was", "were", "what",
    "when", "where", "which", "who", "why",
    "with", "about", "main", "project",
    "document", "explain", "describe",
    "list", "give", "tell", "me"
}

SECTION_KEYWORDS = {
    "objective": {
        "objective", "objectives", "aim",
        "aims", "goal", "goals"
    },
    "methodology": {
        "methodology", "method", "methods",
        "approach", "implementation",
        "workflow", "pipeline", "process",
        "processing", "procedure"
    },
    "result": {
        "result", "results", "output",
        "outputs", "performance", "evaluation"
    },
    "conclusion": {
        "conclusion", "summary", "findings"
    },
    "architecture": {
        "architecture", "design", "structure",
        "components", "diagram"
    },
    "limitation": {
        "limitation", "limitations",
        "challenge", "challenges"
    },
    "future scope": {
        "future", "scope", "enhancement",
        "enhancements"
    },
    "technology": {
        "technology", "technologies",
        "framework", "frameworks",
        "software", "libraries", "tools",
        "requirements"
    },
    "literature": {
        "literature", "survey", "review",
        "related", "research"
    }
}

SECTION_HEADING_PATTERNS = {
    "objective": [
        r"objectives?",
        r"aims?",
        r"goals?"
    ],
    "methodology": [
        r"methodology",
        r"proposed methodology",
        r"proposed method",
        r"implementation",
        r"implementation methodology",
        r"proposed approach",
        r"working methodology",
        r"system workflow",
        r"processing pipeline"
    ],
    "result": [
        r"results?",
        r"results? and discussion",
        r"performance evaluation",
        r"evaluation"
    ],
    "conclusion": [
        r"conclusions?",
        r"conclusion and future scope",
        r"summary"
    ],
    "architecture": [
        r"(?:system )?architecture",
        r"system design",
        r"class diagram",
        r"activity diagram",
        r"sequence diagram"
    ],
    "limitation": [
        r"limitations?",
        r"challenges?"
    ],
    "future scope": [
        r"future scope",
        r"future work",
        r"future enhancements?"
    ],
    "technology": [
        r"software requirements?",
        r"technologies used",
        r"technology stack",
        r"tools and technologies",
        r"frameworks? used",
        r"libraries and tools"
    ],
    "literature": [
        r"literature survey",
        r"literature review",
        r"related work",
        r"research review"
    ]
}

FRONT_MATTER_TERMS = {
    "certificate",
    "acknowledgement",
    "acknowledgment",
    "declaration",
    "dissertation work submitted",
    "submitted by",
    "under the guidance"
}

FRONT_MATTER_REQUESTS = {
    "certificate",
    "cover page",
    "title page",
    "submitted by",
    "guide",
    "supervisor",
    "author",
    "authors",
    "student",
    "students",
    "acknowledgement",
    "declaration"
}

# These patterns describe common processing stages.
# They are not tied to a particular PDF or page number.
WORKFLOW_PATTERNS = {
    "input": [
        r"\bupload(?:ing|ed|s)?\b",
        r"\binput\b",
        r"\bsubmission\b",
        r"\bsubmitted\b"
    ],
    "validation": [
        r"\bvalidat(?:e|es|ed|ion)\b",
        r"\bfile format\b",
        r"\bfile size\b"
    ],
    "preprocessing": [
        r"\bpreprocess(?:ing|ed)?\b",
        r"\bextract(?:ion|ed|s)?\b",
        r"\bframes? extracted\b"
    ],
    "processing": [
        r"\bmodel\b",
        r"\binterpolat(?:ion|e|es|ed)\b",
        r"\binference\b",
        r"\bprocess(?:ing|ed|es)?\b"
    ],
    "postprocessing": [
        r"\bpost.process(?:ing|ed)?\b",
        r"\breconstruct(?:ion|ed|s)?\b",
        r"\bassembl(?:e|ed|y)\b"
    ],
    "output": [
        r"\boutput\b",
        r"\bdownload\b",
        r"\bresult\b",
        r"\bdeliver(?:y|ed)?\b"
    ]
}

# Signals frequently found in literature-survey tables.
# A single signal is not sufficient for a penalty.
LITERATURE_TABLE_PATTERNS = [
    r"\bresearchgate\b",
    r"\bieee\b",
    r"\b(?:springer|elsevier|arxiv|icasSP)\b",
    r"\b(?:2022|2023|2024|2025|2026)\b",
    r"\bproposed model\b",
    r"\bexisting solution\b",
    r"\bresearch gap\b",
    r"\bpaired training data\b",
    r"\bself.supervised learning\b"
]


# =====================================================
# TEXT PROCESSING HELPERS
# =====================================================

def tokenize_text(text):
    """
    Convert text into normalized searchable words.
    """

    if not isinstance(text, str):
        return set()

    words = re.findall(
        r"[a-z0-9]+",
        text.lower()
    )

    return {
        word
        for word in words
        if len(word) > 1
        and word not in STOP_WORDS
    }


def detect_requested_sections(query):
    """
    Identify the primary section intent of a question.

    Explicit section terms take priority.
    """

    if not isinstance(query, str):
        return set()

    query_lower = query.lower()

    query_words = set(
        re.findall(
            r"[a-z0-9]+",
            query_lower
        )
    )

    if "future scope" in query_lower:
        return {"future scope"}

    explicit_terms = {
        "methodology": {
            "methodology", "method", "methods",
            "approach", "workflow", "pipeline"
        },
        "objective": {
            "objective", "objectives",
            "aim", "aims", "goal", "goals"
        },
        "conclusion": {
            "conclusion", "conclusions"
        },
        "architecture": {
            "architecture", "design", "diagram"
        },
        "technology": {
            "technology", "technologies",
            "framework", "frameworks",
            "software", "libraries"
        },
        "result": {
            "result", "results", "evaluation"
        },
        "limitation": {
            "limitation", "limitations",
            "challenge", "challenges"
        },
        "future scope": {
            "enhancement", "enhancements"
        },
        "literature": {
            "literature", "survey"
        }
    }

    for section, terms in explicit_terms.items():

        if query_words.intersection(terms):
            return {section}

    return set()


def normalize_heading_text(text):
    """
    Remove common heading numbering.
    """

    if not isinstance(text, str):
        return ""

    text = " ".join(text.lower().split())

    text = re.sub(
        r"^chapter\s+\d+\s*[:.\-]?\s*",
        "",
        text
    )

    text = re.sub(
        r"^\d+(?:\.\d+)*\.?\s+",
        "",
        text
    )

    return text.strip(" :-.")


# =====================================================
# FAISS VECTOR STORE
# =====================================================

class FAISSVectorStore:
    """
    Store PDF embeddings and retrieve relevant chunks.

    Features:
    - FAISS cosine similarity
    - Keyword matching
    - Section-aware ranking
    - Workflow-aware methodology retrieval
    - Literature-table detection
    - Text quality scoring
    - Persistent vector storage
    """

    # =================================================
    # INITIALIZATION
    # =================================================

    def __init__(self, dimension=DEFAULT_DIMENSION):

        if (
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension <= 0
        ):
            raise ValueError(
                "Dimension must be a positive integer."
            )

        self.dimension = dimension

        self.index = faiss.IndexFlatIP(
            dimension
        )

        self.metadata = []

    # =================================================
    # TEXT QUALITY SCORING
    # =================================================

    @staticmethod
    def calculate_text_quality(text):
        """
        Estimate information quality of a chunk.
        """

        if not isinstance(text, str):
            return 0.0

        cleaned = " ".join(text.split())

        if not cleaned:
            return 0.0

        word_count = len(cleaned.split())
        character_count = len(cleaned)

        if (
            word_count < MIN_USEFUL_WORDS
            or character_count < MIN_USEFUL_CHARACTERS
        ):
            return 0.15

        if word_count < 15:
            return 0.45

        if word_count < 30:
            return 0.75

        return 1.0

    # =================================================
    # KEYWORD SCORING
    # =================================================

    @staticmethod
    def calculate_keyword_score(query, text):
        """
        Calculate keyword overlap without giving
        full credit for only one remaining query word.
        """

        query_words = tokenize_text(query)
        document_words = tokenize_text(text)

        if not query_words:
            return 0.0

        requested_sections = detect_requested_sections(
            query
        )

        if requested_sections:

            section_words = set()

            for section in requested_sections:
                section_words.update(
                    SECTION_KEYWORDS[section]
                )

            query_words -= section_words

            query_words -= {
                "real", "time", "frame",
                "interpolation"
            }

        if not query_words:
            return 0.0

        matched_words = query_words.intersection(
            document_words
        )

        # The minimum denominator prevents a
        # single keyword match from scoring 1.0.
        denominator = max(
            len(query_words),
            MIN_KEYWORD_TERMS
        )

        return len(matched_words) / denominator

    # =================================================
    # HEADING DETECTION
    # =================================================

    @staticmethod
    def _matches_section_heading(section, heading):
        """
        Check whether text matches a section heading.
        """

        heading = normalize_heading_text(
            heading
        )

        patterns = SECTION_HEADING_PATTERNS.get(
            section,
            []
        )

        return any(
            re.fullmatch(
                pattern,
                heading,
                flags=re.IGNORECASE
            )
            for pattern in patterns
        )

    @classmethod
    def calculate_section_score(cls, query, text):
        """
        Calculate section-heading relevance.

        1.00 = Explicit heading
        0.70 = Heading at beginning
        0.20 = Ordinary section-term mention
        0.00 = No matching section evidence
        """

        requested_sections = detect_requested_sections(
            query
        )

        if not requested_sections:
            return 0.0

        if not isinstance(text, str) or not text.strip():
            return 0.0

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        beginning = " ".join(
            text.split()
        )[:400]

        best_score = 0.0

        for section in requested_sections:

            patterns = SECTION_HEADING_PATTERNS.get(
                section,
                []
            )

            # Case 1: Standalone heading
            for line in lines[:5]:

                if cls._matches_section_heading(
                    section,
                    line
                ):
                    best_score = max(
                        best_score,
                        1.0
                    )

            # Case 2: Numbered heading
            for pattern in patterns:

                numbered_pattern = (
                    r"(?:^|\s)"
                    r"\d+(?:\.\d+)*\.?\s+"
                    r"(?:" + pattern + r")"
                    r"(?=\s|$|[.:])"
                )

                if re.search(
                    numbered_pattern,
                    beginning,
                    flags=re.IGNORECASE
                ):
                    best_score = max(
                        best_score,
                        1.0
                    )

            # Case 3: Heading at chunk start
            normalized = normalize_heading_text(
                beginning
            )

            for pattern in patterns:

                heading_at_start = (
                    r"^(?:" + pattern + r")"
                    r"(?=\s|$|[.:])"
                )

                if re.search(
                    heading_at_start,
                    normalized,
                    flags=re.IGNORECASE
                ):
                    best_score = max(
                        best_score,
                        0.70
                    )

            # Case 4: Ordinary body mention
            section_words = SECTION_KEYWORDS[section]

            text_words = set(
                re.findall(
                    r"[a-z0-9]+",
                    text.lower()
                )
            )

            if text_words.intersection(section_words):
                best_score = max(
                    best_score,
                    0.20
                )

        return best_score

    # =================================================
    # WORKFLOW SCORING
    # =================================================

    @staticmethod
    def calculate_workflow_score(query, text):
        """
        Reward procedural explanations for methodology
        questions.

        A chunk must describe multiple processing
        stages to receive a substantial bonus.
        """

        requested = detect_requested_sections(
            query
        )

        if "methodology" not in requested:
            return 0.0

        if not isinstance(text, str) or not text.strip():
            return 0.0

        text_lower = text.lower()

        matched_stages = set()

        for stage, patterns in WORKFLOW_PATTERNS.items():

            if any(
                re.search(
                    pattern,
                    text_lower,
                    flags=re.IGNORECASE
                )
                for pattern in patterns
            ):
                matched_stages.add(stage)

        stage_count = len(matched_stages)

        if stage_count < 2:
            return 0.0

        if stage_count == 2:
            return 0.25

        if stage_count == 3:
            return 0.50

        if stage_count == 4:
            return 0.75

        return 1.0

    # =================================================
    # LITERATURE-TABLE DETECTION
    # =================================================

    @staticmethod
    def calculate_literature_table_penalty(
        query,
        text
    ):
        """
        Penalize recognizable literature-survey
        table fragments for unrelated questions.

        This does not penalize explicit literature
        survey questions.
        """

        requested = detect_requested_sections(
            query
        )

        if not requested:
            return 0.0

        if "literature" in requested:
            return 0.0

        if not isinstance(text, str):
            return 0.0

        text_lower = text.lower()

        matches = sum(
            bool(
                re.search(
                    pattern,
                    text_lower,
                    flags=re.IGNORECASE
                )
            )
            for pattern in LITERATURE_TABLE_PATTERNS
        )

        if matches >= 3:
            return LITERATURE_TABLE_PENALTY

        if matches == 2:
            return LITERATURE_TABLE_PENALTY * 0.5

        return 0.0

    # =================================================
    # UNRELATED SECTION PENALTY
    # =================================================

    @staticmethod
    def calculate_unrelated_section_penalty(
        query,
        text
    ):
        """
        Penalize explicitly unrelated headings.
        """

        requested = detect_requested_sections(
            query
        )

        if not requested:
            return 0.0

        if not isinstance(text, str):
            return 0.0

        beginning = " ".join(
            text.lower().split()
        )[:250]

        if "methodology" in requested:

            if re.search(
                r"\bliterature\s+(survey|review)\b",
                beginning
            ):
                return UNRELATED_SECTION_PENALTY

        if "objective" in requested:

            if re.search(
                r"\b(?:conclusion|literature survey)\b",
                beginning
            ):
                return UNRELATED_SECTION_PENALTY

        if "conclusion" in requested:

            if re.search(
                r"\bliterature\s+(survey|review)\b",
                beginning
            ):
                return UNRELATED_SECTION_PENALTY

        return 0.0

    # =================================================
    # FRONT-MATTER PENALTY
    # =================================================

    @staticmethod
    def calculate_front_matter_penalty(query, chunk):
        """
        Reduce administrative-content relevance
        unless the question explicitly requests it.
        """

        text = chunk.get(
            "text",
            ""
        ).lower()

        query_lower = query.lower()

        if any(
            term in query_lower
            for term in FRONT_MATTER_REQUESTS
        ):
            return 0.0

        if any(
            term in text
            for term in FRONT_MATTER_TERMS
        ):
            return FRONT_MATTER_PENALTY

        return 0.0

    # =================================================
    # ADD DOCUMENTS
    # =================================================

    def add_documents(self, chunks, embeddings):
        """
        Add document chunks and embeddings to FAISS.
        """

        if not isinstance(chunks, list):
            raise TypeError(
                "Chunks must be a list."
            )

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32
        )

        if embeddings.ndim != 2:
            raise ValueError(
                "Embeddings must be a 2D array."
            )

        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                "Embedding dimension mismatch."
            )

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Number of chunks and embeddings must match."
            )

        if not np.isfinite(embeddings).all():
            raise ValueError(
                "Embeddings contain invalid values."
            )

        if not all(
            isinstance(chunk, dict)
            and isinstance(chunk.get("text"), str)
            and chunk["text"].strip()
            for chunk in chunks
        ):
            raise ValueError(
                "Every chunk must contain non-empty text."
            )

        if len(chunks) == 0:
            return

        normalized_embeddings = embeddings.copy()

        faiss.normalize_L2(
            normalized_embeddings
        )

        self.index.add(
            normalized_embeddings
        )

        self.metadata.extend(
            [chunk.copy() for chunk in chunks]
        )

        print(
            f"Added {len(chunks)} document chunks to FAISS."
        )

        print(
            "Total vectors:",
            self.index.ntotal
        )

    # =================================================
    # HYBRID SEARCH
    # =================================================

    def search(
        self,
        query_embedding,
        top_k=5,
        query_text=""
    ):
        """
        Search FAISS and rerank candidates.

        query_text activates hybrid ranking.
        """

        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or top_k <= 0
        ):
            raise ValueError(
                "top_k must be a positive integer."
            )

        if not isinstance(query_text, str):
            raise TypeError(
                "Query text must be a string."
            )

        if self.index.ntotal == 0:
            return []

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32
        )

        if query_embedding.size != self.dimension:
            raise ValueError(
                "Query embedding dimension mismatch."
            )

        query_embedding = query_embedding.reshape(
            1,
            self.dimension
        )

        if not np.isfinite(query_embedding).all():
            raise ValueError(
                "Query embedding contains invalid values."
            )

        normalized_query = query_embedding.copy()

        faiss.normalize_L2(
            normalized_query
        )

        candidate_count = min(
            max(
                top_k * CANDIDATE_MULTIPLIER,
                top_k
            ),
            self.index.ntotal
        )

        scores, indices = self.index.search(
            normalized_query,
            candidate_count
        )

        candidates = []

        for similarity, index in zip(
            scores[0],
            indices[0]
        ):

            if index < 0:
                continue

            chunk = self.metadata[index].copy()

            text = chunk.get(
                "text",
                ""
            )

            similarity_score = float(
                similarity
            )

            quality_score = self.calculate_text_quality(
                text
            )

            if query_text.strip():

                keyword_score = (
                    self.calculate_keyword_score(
                        query_text,
                        text
                    )
                )

                section_score = (
                    self.calculate_section_score(
                        query_text,
                        text
                    )
                )

                workflow_score = (
                    self.calculate_workflow_score(
                        query_text,
                        text
                    )
                )

                front_penalty = (
                    self.calculate_front_matter_penalty(
                        query_text,
                        chunk
                    )
                )

                unrelated_penalty = (
                    self.calculate_unrelated_section_penalty(
                        query_text,
                        text
                    )
                )

                literature_penalty = (
                    self.calculate_literature_table_penalty(
                        query_text,
                        text
                    )
                )

                ranking_score = (
                    SEMANTIC_WEIGHT * similarity_score
                    + KEYWORD_WEIGHT * keyword_score
                    + QUALITY_WEIGHT * quality_score
                    + SECTION_WEIGHT * section_score
                    + WORKFLOW_WEIGHT * workflow_score
                    - front_penalty
                    - unrelated_penalty
                    - literature_penalty
                )

            else:

                keyword_score = 0.0
                section_score = 0.0
                workflow_score = 0.0

                front_penalty = 0.0
                unrelated_penalty = 0.0
                literature_penalty = 0.0

                ranking_score = (
                    0.85 * similarity_score
                    + 0.15 * quality_score
                )

            chunk["similarity_score"] = (
                similarity_score
            )

            chunk["quality_score"] = (
                quality_score
            )

            chunk["keyword_score"] = (
                float(keyword_score)
            )

            chunk["section_score"] = (
                float(section_score)
            )

            chunk["workflow_score"] = (
                float(workflow_score)
            )

            chunk["ranking_score"] = (
                float(ranking_score)
            )

            candidates.append(
                chunk
            )

        candidates.sort(
            key=lambda item: item["ranking_score"],
            reverse=True
        )

        return candidates[:top_k]

    # =================================================
    # SAVE VECTOR DATABASE
    # =================================================

    def save(self):
        """
        Save FAISS index and document metadata.
        """

        VECTOR_STORE_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        faiss.write_index(
            self.index,
            str(INDEX_PATH)
        )

        with open(
            METADATA_PATH,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                self.metadata,
                file,
                indent=4,
                ensure_ascii=False
            )

        print(
            "FAISS vector store saved successfully!"
        )

    # =================================================
    # LOAD VECTOR DATABASE
    # =================================================

    def load(self):
        """
        Load FAISS index and metadata.

        Only load trusted FAISS index files.
        """

        if not INDEX_PATH.exists():
            raise FileNotFoundError(
                "FAISS index file not found."
            )

        if not METADATA_PATH.exists():
            raise FileNotFoundError(
                "Metadata file not found."
            )

        loaded_index = faiss.read_index(
            str(INDEX_PATH)
        )

        with open(
            METADATA_PATH,
            "r",
            encoding="utf-8"
        ) as file:

            loaded_metadata = json.load(
                file
            )

        if not isinstance(loaded_metadata, list):
            raise ValueError(
                "Invalid metadata format."
            )

        if loaded_index.d != self.dimension:
            raise ValueError(
                "Saved FAISS index dimension mismatch."
            )

        if loaded_index.ntotal != len(
            loaded_metadata
        ):
            raise ValueError(
                "FAISS index and metadata count mismatch."
            )

        self.index = loaded_index
        self.metadata = loaded_metadata

        print(
            "FAISS vector store loaded successfully!"
        )

        print(
            "Total loaded vectors:",
            self.index.ntotal
        )
