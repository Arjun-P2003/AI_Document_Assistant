
from modules.pdf_reader import extract_pdf_text
from modules.text_chunker import chunk_pdf_pages
from modules.embeddings import (
    generate_embeddings,
    generate_query_embedding
)


def main():
    print("\n===== EMBEDDING GENERATION TEST =====")

    try:
        # Step 1: Extract PDF text
        pdf_data = extract_pdf_text("data/sample.pdf")

        print("PDF:", pdf_data["filename"])

        # Step 2: Split PDF into chunks
        chunks = chunk_pdf_pages(
            pdf_data["pages"],
            chunk_size=50,
            overlap=10
        )

        texts = [chunk["text"] for chunk in chunks]

        print("Total Chunks:", len(texts))

        # Step 3: Generate document embeddings
        embeddings = generate_embeddings(texts)

        print("\n===== DOCUMENT EMBEDDINGS =====")
        print("Embedding Shape:", embeddings.shape)
        print("Embedding Data Type:", embeddings.dtype)

        if len(embeddings) > 0:
            print("First 10 Vector Values:")
            print(embeddings[0][:10])

        # Step 4: Generate a question embedding
        query = "What is Retrieval Augmented Generation?"

        query_embedding = generate_query_embedding(query)

        print("\n===== QUERY EMBEDDING =====")
        print("Question:", query)
        print("Query Embedding Shape:", query_embedding.shape)

        assert embeddings.shape == (len(texts), 384)
        assert query_embedding.shape == (384,)

        print("\nSUCCESS: Embedding generation completed!")

    except Exception as error:
        print("\nERROR:", error)
        raise


if __name__ == "__main__":
    main()
