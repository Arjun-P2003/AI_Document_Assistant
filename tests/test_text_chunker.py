
from modules.pdf_reader import extract_pdf_text
from modules.text_chunker import chunk_pdf_pages


def main():
    print("\n===== DOCUMENT CHUNKING TEST =====")

    pdf_path = "data/sample.pdf"

    try:
        # Extract text from the PDF
        result = extract_pdf_text(pdf_path)

        print("PDF Filename:", result["filename"])
        print("Total Pages:", result["total_pages"])

        # Split extracted pages into chunks
        chunks = chunk_pdf_pages(
            result["pages"],
            chunk_size=50,
            overlap=10
        )

        print("Total Chunks:", len(chunks))

        print("\n===== GENERATED CHUNKS =====")

        for index, chunk in enumerate(chunks, start=1):
            print(f"\nChunk {index}")
            print("Page Number:", chunk["page_number"])
            print("Chunk Number:", chunk["chunk_number"])
            print("Characters:", len(chunk["text"]))
            print("Text:", chunk["text"])

        if chunks:
            print("\nSUCCESS: Document chunking completed!")
        else:
            print("\nWARNING: No text chunks generated.")

    except Exception as error:
        print(f"\nERROR: {error}")


if __name__ == "__main__":
    main()
