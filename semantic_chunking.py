from vector_db import VectorDB
from langchain_community.chat_models import ChatOllama
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.documents import Document


def load_and_semantically_chunk(texts, model_name="mxbai-embed-large:latest"):
    """
    Loads documents and applies semantic chunking.

    :param texts: List of text documents.
    :param model_name: Embedding model name for chunking.
    :return: List of semantically chunked documents.
    """
    embedder = OllamaEmbeddings(model=model_name)
    chunker = SemanticChunker(embedding=embedder)

    documents = []
    for text in texts:
        chunks = chunker.split_text(text)
        documents.extend([Document(page_content=chunk) for chunk in chunks])

    return documents


class SemanticRAG:
    def __init__(self, db_path="faiss_index", model_name="qwen2.5:0.5b-instruct", temperature=0, EMBED_model_name="mxbai-embed-large:latest"):
        """
        Initializes the Semantic Chunking-based RAG system.

        :param db_path: Path to the FAISS index.
        :param model_name: Name of the LLM model to use.
        """
        self.db = VectorDB(db_path=db_path, model_name=EMBED_model_name)
        self.llm = ChatOllama(model=model_name, temperature=temperature)

    def query(self, question, k=5):
        """
        Perform a semantic chunking-based RAG pipeline by retrieving relevant chunks.

        :param question: The user query.
        :param k: Number of documents to retrieve.
        :return: Tuple containing LLM-generated response and retrieved context.
        """
        retrieved_docs = self.db.similarity_search(question, k=k)
        context = "\n".join([doc.page_content for doc in retrieved_docs])

        prompt = f"""
        Answer in very short to the point manner
        Context:
        {context}

        Question: {question}

        Answer: """

        response = self.llm.invoke(prompt)
        return response.content, context


if __name__ == "__main__":
    rag = SemanticRAG()
    question = "What insights does the document provide on climate change?"
    response, context = rag.query(question)
    print("\n📄 Retrieved Context:")
    print(context)
    print("\n🤖 LLM Response:")
    print(response)
