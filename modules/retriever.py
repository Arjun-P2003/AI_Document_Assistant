
import re

from modules.embeddings import generate_query_embedding
from modules.vector_store import (
    FAISSVectorStore,
    detect_requested_sections
)


# =====================================================
# DOCUMENT RETRIEVER
# =====================================================

class DocumentRetriever:
    """
    Retrieve relevant PDF chunks using hybrid search.

    Features:
    - Query validation
    - Query embedding generation
    - Hybrid FAISS retrieval
    - Source metadata preparation
    - Question-aware context organization
    - Reusable context construction
    """

    # =================================================
    # INITIALIZATION
    # =================================================

    def __init__(self, vector_store=None):

        if vector_store is None:

            self.vector_store = FAISSVectorStore()
            self.vector_store.load()

        else:

            self.vector_store = vector_store

    # =================================================
    # RETRIEVE DOCUMENT CHUNKS
    # =================================================

    def retrieve(self, question, top_k=3):
        """
        Retrieve relevant document chunks
        using hybrid FAISS search.
        """

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

        query_embedding = generate_query_embedding(
            question
        )

        results = self.vector_store.search(
            query_embedding,
            top_k=top_k,
            query_text=question
        )

        processed_results = []

        for result in results:

            item = result.copy()

            similarity_score = float(
                item.get(
                    "similarity_score",
                    0.0
                )
            )

            ranking_score = float(
                item.get(
                    "ranking_score",
                    similarity_score
                )
            )

            item["relevance_score"] = ranking_score

            processed_results.append(item)

        return processed_results

    # =================================================
    # WORKFLOW CONTENT SCORING
    # =================================================

    @staticmethod
    def calculate_context_workflow_score(text):
        """
        Identify passages containing procedural steps.

        This is used only to organize the LLM context,
        not to modify FAISS retrieval rankings.
        """

        if not isinstance(text, str):
            return 0

        text_lower = text.lower()

        workflow_groups = [
            [
                r"\bupload\b",
                r"\buploaded\b",
                r"\binput\b"
            ],
            [
                r"\bvalidat",
                r"\bformat\b"
            ],
            [
                r"\bextract",
                r"\bpreprocess"
            ],
            [
                r"\binterpolat",
                r"\binference\b",
                r"\bprocessing\b"
            ],
            [
                r"\breconstruct",
                r"\brender",
                r"\bpost.process"
            ],
            [
                r"\bdownload\b",
                r"\boutput\b",
                r"\bdeliver"
            ]
        ]

        matched_groups = 0

        for patterns in workflow_groups:

            if any(
                re.search(pattern, text_lower)
                for pattern in patterns
            ):
                matched_groups += 1

        return matched_groups

    # =================================================
    # ORGANIZE CONTEXT SOURCES
    # =================================================

    @classmethod
    def organize_context_sources(
        cls,
        sources,
        question=""
    ):
        """
        Organize passages for the LLM.

        For methodology questions, prioritize
        passages containing multiple workflow stages.

        The original sources list is not modified.
        """

        organized = list(sources)

        if not question:
            return organized

        requested_sections = detect_requested_sections(
            question
        )

        if "methodology" not in requested_sections:
            return organized

        def sort_key(source):

            workflow_score = (
                cls.calculate_context_workflow_score(
                    source.get("text", "")
                )
            )

            relevance_score = float(
                source.get("relevance_score", 0.0)
            )

            return (
                workflow_score,
                relevance_score
            )

        return sorted(
            organized,
            key=sort_key,
            reverse=True
        )

    # =================================================
    # BUILD CONTEXT FROM SELECTED SOURCES
    # =================================================

    @classmethod
    def build_context(
        cls,
        sources,
        question=""
    ):
        """
        Construct LLM context from selected sources.

        Compatible with existing calls:
            DocumentRetriever.build_context(sources)

        Optional question-aware organization:
            DocumentRetriever.build_context(
                sources,
                question
            )
        """

        if not isinstance(sources, list):
            raise TypeError(
                "Sources must be a list."
            )

        organized_sources = (
            cls.organize_context_sources(
                sources,
                question
            )
        )

        context_parts = []

        for index, source in enumerate(
            organized_sources,
            start=1
        ):

            if not isinstance(source, dict):
                continue

            text = source.get(
                "text",
                ""
            )

            if not isinstance(text, str):
                continue

            text = text.strip()

            if not text:
                continue

            page_number = source.get(
                "page_number",
                "Unknown"
            )

            chunk_number = source.get(
                "chunk_number",
                "Unknown"
            )

            # Preserve the original source ID.
            source_id = source.get(
                "source_id",
                index
            )

            context_parts.append(
                f"[Source {source_id} | "
                f"Page {page_number} | "
                f"Chunk {chunk_number}]\n"
                f"{text}"
            )

        context = "\n\n".join(
            context_parts
        )

        if (
            question
            and "methodology" in detect_requested_sections(
                question
            )
            and context
        ):

            context = (
                "[Context organization note]\n"
                "The following passages describe "
                "different aspects of the document. "
                "When answering the methodology question, "
                "combine supported processing stages "
                "into their logical sequence. "
                "Do not invent missing steps or include "
                "unrelated objectives and future work.\n\n"
                + context
            )

        return context

    # =================================================
    # RETRIEVE CONTEXT FOR OLLAMA
    # =================================================

    def retrieve_context(self, question, top_k=3):
        """
        Retrieve document chunks and prepare
        context plus source metadata.
        """

        results = self.retrieve(
            question,
            top_k=top_k
        )

        if not results:

            return {
                "question": question,
                "context": "",
                "sources": []
            }

        sources = []

        for index, result in enumerate(
            results,
            start=1
        ):

            similarity_score = float(
                result.get(
                    "similarity_score",
                    0.0
                )
            )

            relevance_score = float(
                result.get(
                    "relevance_score",
                    similarity_score
                )
            )

            keyword_score = float(
                result.get(
                    "keyword_score",
                    0.0
                )
            )

            section_score = float(
                result.get(
                    "section_score",
                    0.0
                )
            )

            sources.append({

                "source_id": index,

                "page_number": result.get(
                    "page_number",
                    "Unknown"
                ),

                "chunk_number": result.get(
                    "chunk_number",
                    "Unknown"
                ),

                "similarity_score": similarity_score,

                "relevance_score": relevance_score,

                "keyword_score": keyword_score,

                "section_score": section_score,

                "text": result.get(
                    "text",
                    ""
                )

            })

        context = self.build_context(
            sources,
            question
        )

        return {
            "question": question,
            "context": context,
            "sources": sources
        }
