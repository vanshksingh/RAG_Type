import os
from vector_db import VectorDB
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from pdf_loader import load_and_chunk_pdf  # Ensure this utility is available

class HierarchicalRAG:
    def __init__(self, db_path="faiss_hierarchical_index", model_name="qwen2.5:0.5b-instruct", temperature=0, EMBED_model_name="mxbai-embed-large:latest"):
        """
        Initializes the Hierarchical Retrieval System with a vector DB and LLM.
        """
        self.db = VectorDB(db_path=db_path, model_name=EMBED_model_name)
        self.llm = ChatOllama(model=model_name, temperature=temperature)

    def add_pdf(self, pdf_path):
        """
        Loads and processes a PDF file into hierarchical chunks.
        """
        print(f"📄 Processing PDF: {pdf_path}")
        raw_chunks = load_and_chunk_pdf(pdf_path, chunk_size=1000, chunk_overlap=100)

        paragraph_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        section_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

        hierarchical_chunks = []
        for chunk in raw_chunks:
            sections = section_splitter.split_text(chunk.page_content)
            for section in sections:
                paragraphs = paragraph_splitter.split_text(section)
                hierarchical_chunks.extend([Document(page_content=para) for para in paragraphs])

        self.db.add_documents(hierarchical_chunks)
        print("✅ Hierarchical indexing completed.")

    def query(self, question, k=5):
        """
        Perform hierarchical retrieval and generate an LLM response.
        """
        retrieved_docs = self.db.similarity_search(question, k=k)
        context = "\n".join([doc.page_content for doc in retrieved_docs])

        prompt = f"""
        Answer in very short to the point manner
        Context:
        {context}

        Question: {question}

        Answer:
        """
        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, "content") else "No response generated.", context

# Example Usage
if __name__ == "__main__":
    rag = HierarchicalRAG()
    pdf_path = "/Users/vanshksingh/PycharmProjects/RAG_Type/Understanding_Climate_Change.pdf"  # Change this to your PDF file
    if os.path.exists(pdf_path):
        rag.add_pdf(pdf_path)

    query = "What does the document say about climate change?"
    response, context = rag.query(query)
    print("\n📄 Retrieved Context:\n", context)
    print("\n🤖 LLM Response:\n", response)
