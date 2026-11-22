
from modules.retriever import DocumentRetriever


def main():
    print("\n===== DOCUMENT RETRIEVAL TEST =====")

    try:
        # Step 1: Load the saved FAISS database
        retriever = DocumentRetriever()

        print("Document retriever initialized!")

        # Step 2: Ask a question
        question = "What is Retrieval Augmented Generation?"

        print("\nUser Question:", question)

        # Step 3: Retrieve relevant document chunks
        results = retriever.retrieve(
            question,
            top_k=3
        )

        print("\n===== RETRIEVED DOCUMENT CHUNKS =====")

        for index, result in enumerate(results, start=1):
            print(f"\nResult {index}")

            print(
                "Page Number:",
                result["page_number"]
            )

            print(
                "Chunk Number:",
                result["chunk_number"]
            )

            print(
                "Similarity Score:",
                round(result["similarity_score"], 4)
            )

            print(
                "Text:",
                result["text"]
            )

        # Step 4: Generate context for the LLM
        context_data = retriever.retrieve_context(
            question,
            top_k=3
        )

        print("\n===== GENERATED LLM CONTEXT =====")
        print(context_data["context"])

        print("\n===== SOURCE INFORMATION =====")
        print(
            "Total Sources:",
            len(context_data["sources"])
        )

        # Step 5: Verify retrieval results
        assert len(results) > 0, (
            "No document chunks retrieved."
        )

        assert context_data["context"], (
            "Generated context is empty."
        )

        assert len(context_data["sources"]) == len(results)

        print(
            "\nSUCCESS: Document retrieval test completed!"
        )

    except Exception as error:
        print("\nERROR:", error)
        raise


if __name__ == "__main__":
    main()
