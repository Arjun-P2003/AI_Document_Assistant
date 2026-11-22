
from pathlib import Path
from modules.pdf_reader import extract_pdf_text


def main():
    pdf_path = Path("data/sample.pdf")

    if not pdf_path.exists():
        print("ERROR: PDF file not found!")
        print("Please add a PDF named sample.pdf inside the data folder.")
        return

    try:
        result = extract_pdf_text(pdf_path)

        print("\n===== PDF EXTRACTION RESULTS =====")
        print("Filename:", result["filename"])
        print("Total Pages:", result["total_pages"])
        print("Extracted Characters:", len(result["full_text"]))

        print("\n===== EXTRACTED TEXT PREVIEW =====")
        print(result["full_text"][:1000])

        if not result["full_text"].strip():
            print("\nWARNING: No text was extracted.")
            print("The PDF may contain scanned images.")

        print("\n===== PAGE DETAILS =====")

        for page in result["pages"]:
            print(
                f"Page {page['page_number']}: "
                f"{len(page['text'])} characters"
            )

    except Exception as error:
        print(f"ERROR: {error}")


if __name__ == "__main__":
    main()
