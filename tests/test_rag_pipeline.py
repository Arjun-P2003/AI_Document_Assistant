
from modules.rag_pipeline import RAGPipeline


def main():
    print("\n===== COMPLETE RAG PIPELINE TEST =====")

    try:
        # Step 1: Initialize the RAG pipeline
        rag = RAGPipeline(
            model_name="qwen2.5:3b"
        )

        print("RAG Pipeline initialized!")

        # Step 2: Process the sample PDF
        document_info = rag.process_document(
            "data/sample.pdf"
        )

        print("\n===== DOCUMENT INFORMATION =====")
        print("Filename:", document_info["filename"])
        print("Total Pages:", document_info["total_pages"])
        print("Total Chunks:", document_info["total_chunks"])

        # Step 3: Ask a question
        question = (
            "What does Retrieval Augmented Generation "
            "use to answer questions?"
        )

        print("\n===== USER QUESTION =====")
        print(question)

        # Step 4: Retrieve context and generate answer
        result = rag.ask_question(
            question,
            top_k=3
        )

        # Step 5: Display the answer
        print("\n===== AI GENERATED ANSWER =====")
        print(result["answer"])

        # Step 6: Display sources
        print("\n===== DOCUMENT SOURCES =====")

        for index, source in enumerate(
            result["sources"],
            start=1
        ):
            print(f"\nSource {index}")
            print("Page:", source["page_number"])
            print("Chunk:", source["chunk_number"])
            print(
                "Similarity:",
                round(source["similarity_score"], 4)
            )
            print("Text:", source["text"])

        # Step 7: Verify results
        assert document_info["total_chunks"] > 0
        assert result["answer"].strip()
        assert len(result["sources"]) > 0

        print(
            "\nSUCCESS: Complete RAG pipeline test completed!"
        )

    except Exception as error:
        print("\nERROR:", error)
        raise


if __name__ == "__main__":
    main()
