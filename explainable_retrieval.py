import logging
from langchain_ollama import ChatOllama
from vector_db import VectorDB

logging.basicConfig(level=logging.INFO)


class ExplainableRetrieval:
    def __init__(self, db_path="faiss_index", model_name="qwen2.5:0.5b-instruct", k=5, temperature=0, EMBED_model_name="mxbai-embed-large:latest"):
        """
        Initializes the explainable retrieval system with FAISS, BM25, and an LLM.
        """
        logging.info("🔄 Loading FAISS index and BM25 model...")
        self.db = VectorDB(db_path=db_path, model_name=EMBED_model_name)
        self.llm = ChatOllama(model=model_name, temperature=temperature)
        self.k = k

    def generate_explanations(self, query, documents):
        """
        Uses an LLM to explain why the retrieved documents are relevant.
        """
        prompt = f"Given the query: '{query}', explain why each of the following documents is relevant.\n\n"
        for i, doc in enumerate(documents):
            prompt += f"Document {i + 1}: {doc.page_content}\n"

        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)

    def query(self, question):
        logging.info("🔄 Retrieving documents...")
        faiss_results = self.db.similarity_search(question, self.k)
        bm25_results = self.db.bm25_search(question, self.k)
        logging.info(f"FAISS Results: {faiss_results}")
        logging.info(f"BM25 Results: {bm25_results}")

        combined_results = faiss_results + bm25_results
        explanations = self.generate_explanations(question, combined_results)
        logging.info(f"Explanations: {explanations}")

        context = "\n".join([doc.page_content for doc in combined_results])
        final_prompt = (
            f"Answer in very short to the point manner Based on the retrieved documents and their explanations, answer the following question: {question}\n\n"
            f"Context (retrieved documents):\n{context}\n\n"
            f"Explanations (relevance of the documents):\n{explanations}\n"
        )

        logging.info("🔄 Sending prompt to LLM...")
        final_response = self.llm.invoke(final_prompt)
        logging.info("🔄 Received response from LLM...")

        return final_response.content if hasattr(final_response, "content") else str(final_response), context


if __name__ == "__main__":
    explainable_rag = ExplainableRetrieval()
    question = "What is the document talking about?"
    response, context = explainable_rag.query(question)

    print("\n📄 Retrieved Context:\n", context)
    print("\n💡 LLM Response:\n", response)
