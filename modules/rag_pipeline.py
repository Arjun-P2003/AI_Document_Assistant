
from pathlib import Path
import math
import re

from modules.pdf_reader import extract_pdf_text
from modules.text_chunker import chunk_pdf_pages
from modules.embeddings import generate_embeddings
from modules.vector_store import (
    FAISSVectorStore,
    detect_requested_sections,
)
from modules.retriever import DocumentRetriever
from modules.llm_generator import LLMGenerator


# =====================================================
# CONFIGURATION
# =====================================================

EMBEDDING_DIMENSION = 384
CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

DEFAULT_TOP_K = 5
MIN_RETRIEVAL_CANDIDATES = 10
RETRIEVAL_MULTIPLIER = 2

MODEL_NAME = "qwen2.5:3b"

MIN_SEMANTIC_SIMILARITY = 0.25
MIN_STRONG_SECTION_SCORE = 0.70

MIN_METHODOLOGY_RELEVANCE = 0.30
MIN_METHODOLOGY_WORKFLOW = 0.50

MISSING_INFORMATION_MESSAGE = (
    "I could not find this information in the document."
)

SUMMARY_MAX_SOURCES = 10
SUMMARY_MAX_CONTEXT_CHARACTERS = 10000

# Each category should contribute useful information.
# These are generic document concepts, not fixed pages.

SUMMARY_CATEGORIES = {
    "overview": {
        "headings": [
            "abstract",
            "introduction",
            "project overview",
            "problem statement",
            "background",
        ],
        "terms": [
            "this project proposes",
            "the proposed system",
            "the main problem",
            "the aim of this project",
        ],
    },
    "objectives": {
        "headings": [
            "objectives",
            "objective",
            "project objectives",
        ],
        "terms": [
            "the main objectives",
            "the objectives are",
            "aims to achieve",
        ],
    },
    "methodology": {
        "headings": [
            "methodology",
            "proposed methodology",
            "working methodology",
            "system workflow",
            "proposed system",
        ],
        "terms": [
            "processing pipeline",
            "step-by-step",
            "processing stages",
            "frame extraction",
            "frame interpolation",
            "video reconstruction",
        ],
    },
    "implementation": {
        "headings": [
            "implementation",
            "system implementation",
            "working principle",
            "system architecture",
            "module description",
        ],
        "terms": [
            "the system processes",
            "the user uploads",
            "the application",
            "input video",
            "output video",
        ],
    },
    "technologies": {
        "headings": [
            "technologies used",
            "technology stack",
            "software requirements",
            "hardware requirements",
            "tools and technologies",
        ],
        "terms": [
            "python",
            "pytorch",
            "opencv",
            "streamlit",
            "flask",
            "gpu acceleration",
        ],
    },
    "results": {
        "headings": [
            "results and discussion",
            "experimental results",
            "performance evaluation",
            "expected outcomes",
            "expected outcome",
            "testing and results",
        ],
        "terms": [
            "frame rate improvement",
            "performance evaluation",
            "output quality",
            "processing time",
            "smooth video",
        ],
    },
    "conclusion": {
        "headings": [
            "conclusion",
            "conclusions",
            "future scope",
            "future enhancements",
        ],
        "terms": [
            "in conclusion",
            "the system demonstrates",
            "future improvements",
        ],
    },
}

WORKFLOW_TERMS = {
    "upload": [
        "video upload",
        "upload video",
        "uploaded video",
        "input video",
    ],
    "validation": [
        "validation",
        "validate",
        "file format",
        "video format",
    ],
    "extraction": [
        "frame extraction",
        "extract frames",
        "extracting frames",
    ],
    "interpolation": [
        "rife",
        "intermediate frame",
        "interpolated frames",
        "optical flow",
    ],
    "reconstruction": [
        "video reconstruction",
        "reconstruct",
        "video generation",
        "combine frames",
        "output video",
    ],
    "delivery": [
        "download",
        "output delivery",
        "processed video",
        "final video",
    ],
}


# =====================================================
# RAG PIPELINE
# =====================================================

class RAGPipeline:

    def __init__(self, model_name=MODEL_NAME):

        self.vector_store = FAISSVectorStore(
            dimension=EMBEDDING_DIMENSION
        )

        self.retriever = DocumentRetriever(
            vector_store=self.vector_store
        )

        self.llm = LLMGenerator(
            model_name=model_name
        )

        self.document_loaded = False
        self.document_info = None
        self.document_chunks = []

    # =================================================
    # DOCUMENT PROCESSING
    # =================================================

    def process_document(self, pdf_path):

        pdf_path = Path(pdf_path)

        if not pdf_path.is_file():
            raise FileNotFoundError(
                f"PDF not found: {pdf_path}"
            )

        if pdf_path.suffix.lower() != ".pdf":
            raise ValueError(
                "Only PDF documents are supported."
            )

        print("\n===== PROCESSING DOCUMENT =====")

        pdf_data = extract_pdf_text(pdf_path)

        total_pages = pdf_data.get("total_pages", 0)

        if total_pages <= 0:
            raise ValueError(
                "The PDF does not contain any pages."
            )

        chunks = chunk_pdf_pages(
            pdf_data["pages"],
            chunk_size=CHUNK_SIZE,
            overlap=CHUNK_OVERLAP,
        )

        if not chunks:
            raise ValueError(
                "No readable text found in the PDF."
            )

        texts = [
            chunk["text"]
            for chunk in chunks
        ]

        embeddings = generate_embeddings(texts)

        if embeddings.shape != (
            len(chunks),
            EMBEDDING_DIMENSION,
        ):
            raise ValueError(
                "Unexpected embedding shape."
            )

        new_vector_store = FAISSVectorStore(
            dimension=EMBEDDING_DIMENSION
        )

        new_vector_store.add_documents(
            chunks,
            embeddings,
        )

        if new_vector_store.index.ntotal != len(chunks):
            raise RuntimeError(
                "Not all chunks were indexed."
            )

        new_retriever = DocumentRetriever(
            vector_store=new_vector_store
        )

        new_vector_store.save()

        self.vector_store = new_vector_store
        self.retriever = new_retriever

        self.document_chunks = [
            chunk.copy()
            for chunk in chunks
        ]

        self.document_info = {
            "filename": pdf_data["filename"],
            "total_pages": total_pages,
            "total_chunks": len(chunks),
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "embedding_dimension": EMBEDDING_DIMENSION,
        }

        self.document_loaded = True

        print("Document:", pdf_data["filename"])
        print("Pages:", total_pages)
        print("Chunks:", len(chunks))
        print("SUCCESS: Document processed.")

        return self.document_info

    # =================================================
    # GENERAL HELPERS
    # =================================================

    @staticmethod
    def safe_score(value):

        try:
            score = float(value)
        except (TypeError, ValueError):
            return None

        if not math.isfinite(score):
            return None

        return score

    @staticmethod
    def normalize_text(text):

        if not isinstance(text, str):
            return ""

        return " ".join(text.lower().split())

    @staticmethod
    def renumber_sources(sources):

        results = []

        for index, source in enumerate(
            sources,
            start=1,
        ):
            item = source.copy()
            item["source_id"] = index
            results.append(item)

        return results

    # =================================================
    # FRONT MATTER
    # =================================================

    @staticmethod
    def is_front_matter(text):

        if not isinstance(text, str):
            return False

        beginning = " ".join(
            text.lower().split()
        )[:500]

        patterns = [
            r"\btable of contents\b",
            r"\blist of figures\b",
            r"\blist of tables\b",
            r"\backnowledg(?:e)?ments?\b",
            r"\bcertificate\b",
            r"\bit is certified that\b",
            r"\bsignature of the (?:guide|hod)\b",
            r"\bproject phase \d+ report has been approved\b",
        ]

        return any(
            re.search(pattern, beginning)
            for pattern in patterns
        )

    @staticmethod
    def requests_front_matter(question):

        if not isinstance(question, str):
            return False

        patterns = [
            r"\bcertificate\b",
            r"\bcertification\b",
            r"\backnowledg(?:e)?ments?\b",
            r"\btable of contents\b",
            r"\blist of figures\b",
            r"\blist of tables\b",
            r"\bapproval\b",
            r"\bapproved\b",
            r"\bsignature\b",
            r"\bguide\b",
            r"\bsupervisor\b",
            r"\bauthors?\b",
            r"\bstudents?\b",
            r"\bsubmitted by\b",
            r"\btitle page\b",
            r"\bcover page\b",
        ]

        return any(
            re.search(pattern, question.lower())
            for pattern in patterns
        )

    # =================================================
    # SUMMARY QUESTION DETECTION
    # =================================================

    @staticmethod
    def is_summary_question(question):

        if not isinstance(question, str):
            return False

        question = " ".join(
            question.lower().split()
        )

        patterns = [
            r"\bsummar(?:ize|ise)\s+(?:the|this|my)\s+(?:entire\s+|whole\s+|full\s+)?(?:document|pdf|report|file|project)\b",
            r"\bsummary\s+of\s+(?:the|this|my)\s+(?:entire\s+|whole\s+|full\s+)?(?:document|pdf|report|project)\b",
            r"\b(?:overall|complete|comprehensive)\s+summary\b",
            r"\b(?:document|project|report)\s+overview\b",
            r"\bwhat\s+is\s+(?:the|this)\s+(?:document|project|report)\s+(?:about|on)\b",
            r"\b(?:explain|describe)\s+(?:the|this)\s+(?:entire\s+|whole\s+)?(?:document|project|report)\b",
            r"\bgive\s+(?:me\s+)?(?:an?\s+)?(?:overview|summary)\s+of\s+(?:the|this|my)\s+(?:document|pdf|project|report)\b",
        ]

        return any(
            re.search(pattern, question)
            for pattern in patterns
        )

    # =================================================
    # SUMMARY NOISE FILTERING
    # =================================================

    @classmethod
    def is_summary_noise(cls, text):
        """
        Reject contents pages, bibliography passages,
        and literature-review entries when generating
        a general document summary.
        """

        normalized = cls.normalize_text(text)

        if len(normalized) < 70:
            return True

        beginning = normalized[:300]

        if cls.is_front_matter(text):
            return True

        excluded_headings = [
            "references",
            "bibliography",
            "literature survey",
            "literature review",
            "related works",
            "list of abbreviations",
        ]

        for heading in excluded_headings:

            if beginning.startswith(heading):
                return True

        # Bibliography entries commonly contain
        # publication metadata and external links.
        citation_markers = len(
            re.findall(r"\[\d+\]", normalized)
        )

        if (
            citation_markers >= 2
            and (
                "authors:" in normalized
                or "available:" in normalized
                or "doi.org" in normalized
                or "arxiv.org" in normalized
                or "this study" in normalized
            )
        ):
            return True

        if (
            normalized.startswith("[")
            and (
                "authors:" in normalized
                or "available:" in normalized
                or "summary: this study" in normalized
            )
        ):
            return True

        # Detect table-of-contents-like passages.
        toc_terms = [
            "class diagram",
            "activity diagram",
            "use case diagram",
            "appendices",
            "references",
        ]

        toc_count = sum(
            term in normalized
            for term in toc_terms
        )

        if (
            toc_count >= 4
            and len(normalized) < 700
        ):
            return True

        # Exclude reference-list style entries.
        if (
            normalized.count("http") >= 2
            and (
                "doi" in normalized
                or "arxiv" in normalized
            )
        ):
            return True

        return False

    # =================================================
    # SUMMARY SCORING
    # =================================================

    @classmethod
    def score_summary_category(
        cls,
        text,
        category,
    ):

        normalized = cls.normalize_text(text)

        if not normalized:
            return 0

        config = SUMMARY_CATEGORIES[category]

        beginning = normalized[:260]

        score = 0

        for heading in config["headings"]:

            if re.search(
                rf"(?<!\w){re.escape(heading)}(?!\w)",
                beginning,
            ):
                score += 12

            elif heading in normalized:
                score += 2

        for term in config["terms"]:

            if term in normalized:
                score += 3

        if category == "methodology":

            workflow_score = (
                cls.score_workflow_coverage(text)
            )

            score += workflow_score * 4

        if category == "implementation":

            workflow_score = (
                cls.score_workflow_coverage(text)
            )

            score += workflow_score * 2

        # Reward substantive paragraphs rather
        # than short headings.
        if len(normalized) >= 250:
            score += 2

        return score

    @classmethod
    def score_workflow_coverage(cls, text):

        normalized = cls.normalize_text(text)

        matched_stages = 0

        for terms in WORKFLOW_TERMS.values():

            if any(
                term in normalized
                for term in terms
            ):
                matched_stages += 1

        return matched_stages

    # =================================================
    # DOCUMENT CHUNK ACCESS
    # =================================================

    def get_document_chunks(self):

        if self.document_chunks:
            return self.document_chunks

        for attribute in (
            "documents",
            "chunks",
            "metadata",
            "metadata_list",
        ):

            value = getattr(
                self.vector_store,
                attribute,
                None,
            )

            if (
                isinstance(value, list)
                and value
                and all(
                    isinstance(item, dict)
                    and isinstance(item.get("text"), str)
                    for item in value
                )
            ):
                return value

        return []

    # =================================================
    # SELECT SUMMARY SOURCES
    # =================================================

    def select_summary_sources(self):
        """
        Select useful content across major sections.

        Unlike the previous version, this does not
        fill empty slots with arbitrary passages
        from the bibliography or literature survey.
        """

        chunks = self.get_document_chunks()

        candidates = []

        for index, chunk in enumerate(chunks):

            if not isinstance(chunk, dict):
                continue

            text = chunk.get("text", "")

            if not isinstance(text, str):
                continue

            if self.is_summary_noise(text):
                continue

            item = chunk.copy()
            item["_index"] = index

            candidates.append(item)

        if not candidates:
            return []

        selected = []
        used_indices = set()

        # One strong passage from each major section.
        for category in SUMMARY_CATEGORIES:

            best_candidate = None
            best_score = 0

            for candidate in candidates:

                if candidate["_index"] in used_indices:
                    continue

                score = self.score_summary_category(
                    candidate["text"],
                    category,
                )

                if score > best_score:

                    best_score = score
                    best_candidate = candidate

            # Require substantive category evidence.
            if best_candidate is None or best_score < 5:
                continue

            selected.append(best_candidate)

            used_indices.add(
                best_candidate["_index"]
            )

            print(
                "Summary category:",
                category,
                "| Page:",
                best_candidate.get("page_number"),
                "| Score:",
                best_score,
            )

        # Add extra workflow passages when needed.
        workflow_candidates = []

        for candidate in candidates:

            if candidate["_index"] in used_indices:
                continue

            workflow_score = (
                self.score_workflow_coverage(
                    candidate["text"]
                )
            )

            if workflow_score >= 2:

                workflow_candidates.append(
                    (
                        workflow_score,
                        candidate,
                    )
                )

        workflow_candidates.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        for _, candidate in workflow_candidates:

            if len(selected) >= SUMMARY_MAX_SOURCES:
                break

            selected.append(candidate)

            used_indices.add(
                candidate["_index"]
            )

        # Fill remaining slots with strongly
        # category-relevant passages only.
        if len(selected) < SUMMARY_MAX_SOURCES:

            remaining = []

            for candidate in candidates:

                if candidate["_index"] in used_indices:
                    continue

                score = max(
                    self.score_summary_category(
                        candidate["text"],
                        category,
                    )
                    for category in SUMMARY_CATEGORIES
                )

                if score >= 8:

                    remaining.append(
                        (
                            score,
                            candidate,
                        )
                    )

            remaining.sort(
                key=lambda item: item[0],
                reverse=True,
            )

            for _, candidate in remaining:

                if len(selected) >= SUMMARY_MAX_SOURCES:
                    break

                selected.append(candidate)

                used_indices.add(
                    candidate["_index"]
                )

        # Keep source references in document order.
        selected.sort(
            key=lambda item: item["_index"]
        )

        results = []

        for candidate in selected:

            item = candidate.copy()

            item.pop("_index", None)

            results.append(item)

        return self.renumber_sources(results)

    # =================================================
    # SUMMARY CONTEXT
    # =================================================

    @staticmethod
    def build_summary_context(sources):

        parts = []
        used_characters = 0

        for source in sources:

            text = source.get(
                "text",
                "",
            ).strip()

            if not text:
                continue

            page = source.get(
                "page_number",
                "Unknown",
            )

            chunk = source.get(
                "chunk_number",
                "Unknown",
            )

            header = (
                f"[Page {page}, Chunk {chunk}]\n"
            )

            remaining = (
                SUMMARY_MAX_CONTEXT_CHARACTERS
                - used_characters
            )

            if remaining <= len(header) + 100:
                break

            allowed = text[
                :remaining - len(header)
            ]

            passage = header + allowed

            parts.append(passage)

            used_characters += len(passage)

        return "\n\n".join(parts)

    # =================================================
    # GENERATE DOCUMENT SUMMARY
    # =================================================

    def summarize_document(self, question):

        print("\n===== DOCUMENT SUMMARY MODE =====")

        sources = self.select_summary_sources()

        if not sources:

            return {
                "question": question,
                "answer": MISSING_INFORMATION_MESSAGE,
                "sources": [],
            }

        context = self.build_summary_context(
            sources
        )

        if not context.strip():

            return {
                "question": question,
                "answer": MISSING_INFORMATION_MESSAGE,
                "sources": [],
            }

        summary_question = (
            "Summarize the supplied document passages "
            "accurately and clearly. Use only information "
            "supported by the passages.\n\n"

            "Organize the answer into these sections "
            "when information is available:\n"

            "1. Project Overview\n"
            "2. Problem Statement\n"
            "3. Main Objectives\n"
            "4. Methodology and Processing Workflow\n"
            "5. Technologies Used\n"
            "6. Results or Expected Outcomes\n"
            "7. Conclusion\n\n"

            "Important instructions:\n"

            "- Explain actual processing steps rather "
            "than merely listing diagram names.\n"

            "- Distinguish proposed features from "
            "features proven to be implemented.\n"

            "- Do not claim cloud deployment, testing, "
            "or performance achievements unless "
            "the supplied text confirms them.\n"

            "- Do not invent metrics or results.\n"

            "- Omit unavailable information instead "
            "of guessing.\n"

            "- Avoid irrelevant references, "
            "bibliography, and administrative details.\n"

            "- Write a readable summary with headings "
            "and concise explanations.\n\n"

            f"User request: {question}"
        )

        print(
            "Summary sources:",
            len(sources),
        )

        print(
            "Context length:",
            len(context),
        )

        answer = self.llm.generate_answer(
            question=summary_question,
            context=context,
        )

        if not isinstance(answer, str):
            raise RuntimeError(
                "The model returned an invalid answer."
            )

        answer = answer.strip()

        if not answer:
            raise RuntimeError(
                "The model returned an empty answer."
            )

        if self.is_missing_information_answer(answer):
            sources = []

        return {
            "question": question,
            "answer": answer,
            "sources": sources,
        }

    # =================================================
    # METHODOLOGY SCORING
    # =================================================

    @staticmethod
    def calculate_methodology_workflow(
        question,
        text,
    ):

        return (
            FAISSVectorStore.calculate_workflow_score(
                question,
                text,
            )
        )

    # =================================================
    # QUESTION-AWARE SOURCE FILTERING
    # =================================================

    @classmethod
    def filter_relevant_sources(
        cls,
        sources,
        question="",
    ):

        if not isinstance(sources, list):
            return []

        if not isinstance(question, str):
            question = ""

        requested_sections = detect_requested_sections(
            question
        )

        allow_front_matter = cls.requests_front_matter(
            question
        )

        candidates = []

        for source in sources:

            if not isinstance(source, dict):
                continue

            text = source.get("text", "")

            if not isinstance(text, str):
                continue

            if not text.strip():
                continue

            similarity = cls.safe_score(
                source.get("similarity_score")
            )

            section_score = cls.safe_score(
                source.get("section_score")
            )

            relevance_score = cls.safe_score(
                source.get("relevance_score")
            )

            if similarity is None:
                continue

            if section_score is None:
                section_score = 0.0

            if relevance_score is None:
                relevance_score = similarity

            if (
                not allow_front_matter
                and cls.is_front_matter(text)
            ):
                continue

            item = source.copy()

            item["section_score"] = section_score
            item["relevance_score"] = relevance_score

            if "methodology" in requested_sections:

                workflow_score = (
                    cls.calculate_methodology_workflow(
                        question,
                        text,
                    )
                )

                item["workflow_score"] = workflow_score

            candidates.append(item)

        if not candidates:
            return []

        # Methodology-specific filtering.
        if "methodology" in requested_sections:

            methodology_sources = []

            for source in candidates:

                workflow_score = source.get(
                    "workflow_score",
                    0.0,
                )

                relevance_score = source[
                    "relevance_score"
                ]

                section_score = source[
                    "section_score"
                ]

                strong_heading = (
                    section_score
                    >= MIN_STRONG_SECTION_SCORE
                )

                strong_workflow = (
                    workflow_score
                    >= MIN_METHODOLOGY_WORKFLOW
                    and relevance_score
                    >= MIN_METHODOLOGY_RELEVANCE
                )

                if strong_heading or strong_workflow:
                    methodology_sources.append(source)

            if methodology_sources:

                methodology_sources.sort(
                    key=lambda source: (
                        source.get(
                            "workflow_score",
                            0.0,
                        ),
                        source.get(
                            "relevance_score",
                            0.0,
                        ),
                    ),
                    reverse=True,
                )

                return methodology_sources

            return [
                source
                for source in candidates
                if source["similarity_score"]
                >= MIN_SEMANTIC_SIMILARITY
            ]

        # Section-specific filtering.
        if requested_sections:

            strong_matches = [
                source
                for source in candidates
                if source["section_score"]
                >= MIN_STRONG_SECTION_SCORE
            ]

            if strong_matches:
                return strong_matches

        # General semantic filtering.
        return [
            source
            for source in candidates
            if source["similarity_score"]
            >= MIN_SEMANTIC_SIMILARITY
        ]

    # =================================================
    # MISSING INFORMATION DETECTION
    # =================================================

    @staticmethod
    def is_missing_information_answer(answer):

        if not isinstance(answer, str):
            return False

        normalized = re.sub(
            r"\s+",
            " ",
            answer.strip().lower(),
        )

        normalized = normalized.strip(" .!?")

        known_responses = {
            "i could not find this information in the document",
            "i could not find relevant information in the document",
            "no relevant document information was found",
            "the document does not contain this information",
            "this information is not available in the document",
            "the requested information is not provided in the document",
            "the document does not provide this information",
        }

        return normalized in known_responses

    # =================================================
    # QUESTION ANSWERING
    # =================================================

    def ask_question(
        self,
        question,
        top_k=DEFAULT_TOP_K,
    ):

        if not self.document_loaded:
            raise RuntimeError(
                "Please process a PDF before "
                "asking questions."
            )

        if not isinstance(question, str):
            raise TypeError(
                "Question must be a string."
            )

        question = question.strip()

        if not question:
            raise ValueError(
                "Question cannot be empty."
            )

        if (
            isinstance(top_k, bool)
            or not isinstance(top_k, int)
            or top_k <= 0
        ):
            raise ValueError(
                "top_k must be a positive integer."
            )

        print("\n===== PROCESSING QUESTION =====")
        print("Question:", question)

        # Route whole-document summary requests.
        if self.is_summary_question(question):

            print("Mode: DOCUMENT SUMMARY")

            return self.summarize_document(
                question
            )

        # Preserve existing question-answering logic.
        candidate_count = max(
            MIN_RETRIEVAL_CANDIDATES,
            top_k * RETRIEVAL_MULTIPLIER,
        )

        retrieval_data = (
            self.retriever.retrieve_context(
                question,
                top_k=candidate_count,
            )
        )

        sources = retrieval_data.get(
            "sources",
            [],
        )

        relevant_sources = (
            self.filter_relevant_sources(
                sources,
                question=question,
            )
        )

        relevant_sources = relevant_sources[:top_k]

        if not relevant_sources:

            return {
                "question": question,
                "answer": MISSING_INFORMATION_MESSAGE,
                "sources": [],
            }

        relevant_sources = self.renumber_sources(
            relevant_sources
        )

        context = self.retriever.build_context(
            relevant_sources,
            question=question,
        )

        if not context.strip():

            return {
                "question": question,
                "answer": MISSING_INFORMATION_MESSAGE,
                "sources": [],
            }

        answer = self.llm.generate_answer(
            question=question,
            context=context,
        )

        if not isinstance(answer, str):
            raise RuntimeError(
                "The model returned an invalid answer."
            )

        answer = answer.strip()

        if not answer:
            raise RuntimeError(
                "The model returned an empty answer."
            )

        if self.is_missing_information_answer(answer):

            return {
                "question": question,
                "answer": answer,
                "sources": [],
            }

        return {
            "question": question,
            "answer": answer,
            "sources": relevant_sources,
        }
