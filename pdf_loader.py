import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


def load_pdf(file_path: str):
    """Load a PDF file and extract text."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"❌ PDF file not found: {file_path}")

    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    return documents


def chunk_documents(documents, chunk_size=512, chunk_overlap=50):
    """Split documents into chunks for vector databases."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunked_docs = text_splitter.split_documents(documents)
    return chunked_docs


def load_and_chunk_pdf(file_path: str, chunk_size=512, chunk_overlap=50):
    """Load a PDF and return chunked documents."""
    documents = load_pdf(file_path)
    return chunk_documents(documents, chunk_size, chunk_overlap)


# Example Usage
if __name__ == "__main__":
    pdf_path = "/Understanding_Climate_Change.pdf"  # Change to your file path
    try:
        chunked_docs = load_and_chunk_pdf(pdf_path)
        print(f"✅ Loaded and chunked {len(chunked_docs)} document sections.")
    except FileNotFoundError as e:
        print(e)
