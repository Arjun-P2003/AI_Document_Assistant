
# AI-Powered Document Assistant

An intelligent, locally powered document analysis application built using Python, Streamlit, FAISS, Sentence Transformers, and Ollama.

The application uses Retrieval-Augmented Generation (RAG) to understand uploaded PDF documents, answer questions based on their contents, generate structured summaries, and provide relevant source references.

## Features

- **PDF Document Upload:** Upload and process PDF documents.
- **Intelligent Text Extraction:** Extract and clean document text using PyMuPDF.
- **Semantic Text Chunking:** Divide documents into manageable text segments.
- **Vector Embeddings:** Convert document chunks into embeddings using Sentence Transformers.
- **FAISS Semantic Search:** Retrieve relevant document sections based on user questions.
- **AI Question Answering:** Generate context-aware answers using Ollama and Qwen2.5 3B.
- **Source References:** Display supporting document pages, chunks, and similarity scores.
- **Document Summarization:** Generate structured summaries of uploaded documents.
- **Suggested Questions:** Help users explore document contents.
- **Conversation History:** Maintain questions and answers during the application session.
- **Export Reports:** Download generated reports in PDF, TXT, and JSON formats.
- **Unsupported Question Handling:** Respond appropriately when requested information is unavailable in the document.
- **Local AI Processing:** Process documents and generate answers locally without relying on a paid cloud LLM API.

## Technology Stack

| Component | Technology |
|-----------|------------|
| Programming Language | Python 3.12 |
| Frontend | Streamlit |
| PDF Processing | PyMuPDF |
| Embedding Model | all-MiniLM-L6-v2 |
| Embedding Framework | Sentence Transformers |
| Vector Database | FAISS |
| Large Language Model | Qwen2.5 3B |
| Local Model Runtime | Ollama |
| PDF Report Generation | ReportLab |
| Data Processing | NumPy, Pandas |

## System Architecture

```text
                PDF Document
                     |
                     v
              Text Extraction
                  PyMuPDF
                     |
                     v
              Text Cleaning
                     |
                     v
               Text Chunking
                     |
                     v
            Sentence Transformers
               Text Embeddings
                     |
                     v
               FAISS Index
                     |
                     v
                User Query
                     |
                     v
              Query Embedding
                     |
                     v
             Semantic Retrieval
                     |
                     v
             Retrieved Context
                     |
                     v
             Ollama + Qwen2.5
                     |
                     v
              Generated Answer
                     |
                     v
           Streamlit User Interface
```

## Project Structure

```text
AI_Document_Assistant/
|
|-- app.py
|-- requirements.txt
|-- README.md
|-- .gitignore
|
|-- modules/
|   |-- embeddings.py
|   |-- llm_generator.py
|   |-- pdf_reader.py
|   |-- rag_pipeline.py
|   |-- retriever.py
|   |-- text_chunker.py
|   |-- vector_store.py
|
|-- tests/
|   |-- test_embeddings.py
|   |-- test_llm_generator.py
|   |-- test_pdf_reader.py
|   |-- test_rag_pipeline.py
|   |-- test_retriever.py
|   |-- test_text_chunker.py
|   |-- test_vector_store.py
|
|-- data/
|-- vector_store/
```

**Note:** The virtual environment, uploaded documents, and generated vector indexes are excluded from GitHub using `.gitignore`.

## Installation and Setup

### 1. Prerequisites

Install the following:

- Python 3.12
- Git
- Ollama

### 2. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/AI_Document_Assistant.git
cd AI_Document_Assistant
```

Replace `YOUR_USERNAME` with your GitHub username after creating the repository.

### 3. Create a Virtual Environment

```bash
python -m venv venv
```

Activate it on Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Install and Configure Ollama

Download Ollama from:

https://ollama.com/

Pull the Qwen2.5 3B model:

```bash
ollama pull qwen2.5:3b
```

Ensure the Ollama service is running before starting the application.

### 6. Run the Application

```bash
streamlit run app.py
```

Open the local address displayed in the terminal, typically:

```text
http://localhost:8501
```

## How It Works

1. Upload a PDF document through the Streamlit interface.
2. The application extracts and cleans the PDF text.
3. The extracted text is divided into smaller chunks.
4. Sentence Transformers generates embeddings for each chunk.
5. FAISS indexes the embeddings for semantic retrieval.
6. Enter a question about the uploaded document.
7. The system retrieves relevant document sections.
8. Ollama generates an answer using the retrieved context.
9. The application displays the answer and supporting source references.
10. Export the conversation or generated report when needed.

## Testing

The application has been manually tested for:

- PDF text extraction and cleaning
- Document chunking and embedding generation
- FAISS indexing and semantic retrieval
- Document-based question answering
- Structured document summarization
- Source reference display
- Handling questions unsupported by the document
- PDF, TXT, and JSON exports

## Current Limitations

- Supports PDF document analysis.
- Requires local Ollama installation and a downloaded language model.
- Performance depends on available CPU, RAM, and GPU resources.
- Scanned PDFs may require additional OCR processing.
- AI-generated answers may occasionally be incomplete or inaccurate.
- Source similarity scores indicate semantic relevance, not answer accuracy.

## Future Enhancements

- Support for DOCX and TXT document uploads.
- Multiple-document question answering.
- OCR support for scanned PDFs.
- Advanced document comparison.
- Improved citation and answer verification.
- Additional local language model options.
- Enhanced conversation management.

## Privacy

The application is designed to process documents locally using Ollama. Uploaded documents and generated vector indexes are excluded from the GitHub repository.

The initial download of model files and dependencies may require an internet connection.

## Author

**Arjun P**

GitHub: https://github.com/Arjun-P2003

## Project Version

Document AI v3.6
