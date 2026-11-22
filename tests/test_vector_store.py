
from modules.pdf_reader import extract_pdf_text
from modules.text_chunker import chunk_pdf_pages
from modules.embeddings import (
    generate_embeddings,
    generate_query_embedding
)
from modules.vector_store import FAISSVectorStore


def main():
    print("\n===== FAISS VECTOR STORE TEST =====")

    try:
        # Step 1: Extract PDF text
        pdf_data = extract_pdf_text("data/sample.pdf")

        print("PDF Filename:", pdf_data["filename"])

        # Step 2: Split text into chunks
        chunks = chunk_pdf_pages(
            pdf_data["pages"],
            chunk_size=50,
            overlap=10
        )

        print("Total Chunks:", len(chunks))

        # Step 3: Generate embeddings
        texts = [chunk["text"] for chunk in chunks]

        embeddings = generate_embeddings(texts)

        print("Embedding Shape:", embeddings.shape)

        # Step 4: Create FAISS vector database
        vector_store = FAISSVectorStore(dimension=384)

        vector_store.add_documents(
            chunks,
            embeddings
        )

        print(
            "Total Vectors Stored:",
            vector_store.index.ntotal
        )

        # Step 5: Search using a question
        question = "What does Retrieval Augmented Generation use?"

        query_embedding = generate_query_embedding(
            question
        )

        results = vector_store.search(
            query_embedding,
            top_k=3
        )

        print("\n===== SEARCH RESULTS =====")

        for index, result in enumerate(results, start=1):
            print(f"\nResult {index}")
            print("Page Number:", result["page_number"])
            print("Chunk Number:", result["chunk_number"])
            print(
                "Similarity Score:",
                round(result["similarity_score"], 4)
            )
            print("Text:", result["text"])

        # Step 6: Save the vector database
        vector_store.save()

        # Step 7: Load and verify saved database
        loaded_store = FAISSVectorStore(dimension=384)
        loaded_store.load()

        assert loaded_store.index.ntotal == len(chunks)

        loaded_results = loaded_store.search(
            query_embedding,
            top_k=3
        )

        assert len(loaded_results) == len(results)

        print("\nSUCCESS: FAISS vector store test completed!")

    except Exception as error:
        print("\nERROR:", error)
        raise


if __name__ == "__main__":
    main()
