import os
import faiss
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi


class VectorDB:
    def __init__(self, db_path="faiss_index", model_name="mxbai-embed-large:latest", use_ollama=True):
        """
        Initializes the vector database.

        :param db_path: Path for saving/loading FAISS index.
        :param model_name: Name of the embedding model.
        :param use_ollama: Whether to use Ollama embeddings.
        """
        self.db_path = db_path
        self.use_ollama = use_ollama
        self.embeddings = OllamaEmbeddings(model=model_name) if use_ollama else None
        self.vector_db = None
        self._load_or_create_db()

        # Retrieve initial documents for BM25 setup
        self.docs = self.similarity_search("dummy query", k=1000)

        # Ensure BM25 is only set up if there are documents
        if self.docs:
            self.tokenized_corpus = [doc.page_content.split() for doc in self.docs]
            self.bm25 = BM25Okapi(self.tokenized_corpus)
        else:
            print("No documents found in FAISS index, BM25 cannot be initialized.")
            pass


    def _load_or_create_db(self):
        """
        Load an existing FAISS index or create a new one.
        """
        if os.path.exists(self.db_path):
            print(f"🔄 Loading FAISS index from {self.db_path}")

            self.vector_db = FAISS.load_local(
                self.db_path,
                self.embeddings,
                allow_dangerous_deserialization=True  # Enabling dangerous deserialization
            )

        else:
            print(f"⚡ Creating new FAISS index")
            # FIX: Use len() to get embedding dimension
            embedding_dim = len(self.embeddings.embed_query("test"))
            index = faiss.IndexFlatL2(embedding_dim)  # Correct FAISS initialization

            self.vector_db = FAISS(
                index=index,
                embedding_function=self.embeddings,
                docstore=InMemoryDocstore({}),
                index_to_docstore_id={}
            )

    def add_documents(self, docs: list[Document], chunk_size=512, chunk_overlap=50):
        """
        Add documents to the vector database with chunking.

        :param docs: List of LangChain Document objects.
        :param chunk_size: Size of text chunks.
        :param chunk_overlap: Overlap between chunks.
        """
        if not docs:
            print("⚠️ No documents provided.")
            return

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # Extract text and split into chunks
        split_docs = []
        for doc in docs:
            chunks = text_splitter.split_text(doc.page_content)
            split_docs.extend([Document(page_content=chunk, metadata=doc.metadata) for chunk in chunks])

        if self.vector_db is None:
            embedding_dim = len(self.embeddings.embed_query("test"))
            index = faiss.IndexFlatL2(embedding_dim)  # Correct FAISS initialization
            self.vector_db = FAISS(
                index=index,
                embedding_function=self.embeddings,
                docstore=InMemoryDocstore({}),
                index_to_docstore_id={}
            )

        self.vector_db.add_documents(split_docs)
        self.save_index()

    def similarity_search(self, query: str, k=5):
        """Perform similarity search."""
        return self.vector_db.similarity_search(query, k=k)


    def bm25_search(self, query, k=5):
        """
        Performs BM25-based sparse retrieval.
        """
        tokenized_query = query.split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [self.docs[i] for i in top_indices]

    def max_marginal_relevance_search(self, query: str, k=5, lambda_mult=0.5):
        """Perform Max Marginal Relevance (MMR) search."""
        return self.vector_db.max_marginal_relevance_search(query, k=k, lambda_mult=lambda_mult)

    def hybrid_search(self, query: str, k=5, weight=0.5):
        """Perform hybrid search combining similarity & diversity."""
        sim_results = self.similarity_search(query, k)
        mmr_results = self.max_marginal_relevance_search(query, k)
        combined_results = sim_results[:int(k * weight)] + mmr_results[:int(k * (1 - weight))]
        return list(set(combined_results))  # Remove duplicates

    def clear_index(self):
        """Clear the FAISS index."""
        self.vector_db = None
        if os.path.exists(self.db_path):
            os.remove(self.db_path)
        print("🧹 FAISS index cleared.")

    def save_index(self):
        """Save the FAISS index to disk."""
        if self.vector_db:
            self.vector_db.save_local(self.db_path)
            print(f"💾 FAISS index saved to {self.db_path}.")

    def get_retriever(self):
        """Return a retriever instance."""
        return self.vector_db.as_retriever() if self.vector_db else None


# Import the PDF loader function
from pdf_loader import load_and_chunk_pdf


# Example Usage
if __name__ == "__main__":

    db = VectorDB(db_path="faiss_index", model_name="mxbai-embed-large:latest")

    pdf_path = "/Users/vanshksingh/PycharmProjects/RAG_Type/Understanding_Climate_Change.pdf"
    chunks = load_and_chunk_pdf(pdf_path, chunk_size=500, chunk_overlap=100)

    db.add_documents(chunks)

    query = "What is the test document about?"
    results = db.similarity_search(query)

    print("\n🔍 Search Results:")
    for doc in results:
        print(f"- {doc.page_content}")
