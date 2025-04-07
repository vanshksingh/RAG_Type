import logging
from langchain_ollama import ChatOllama
from vector_db import VectorDB  # FAISS + BM25 Hybrid Retrieval

logging.basicConfig(level=logging.INFO)

class SelfRAG:
    def __init__(self,db_path="faiss_index", model_name="qwen2.5:0.5b-instruct", k=5, max_iters=3, confidence_threshold=0.7 , temperature=0 , EMBED_model_name="mxbai-embed-large:latest") :
        """
        Initializes Self-RAG with an LLM, vector DB, and retrieval settings.
        """
        self.llm = ChatOllama(model=model_name, temperature=temperature)
        self.db = VectorDB(db_path=db_path,model_name=EMBED_model_name)  # FAISS + BM25 Hybrid Retrieval
        self.k = k
        self.max_iters = max_iters
        self.confidence_threshold = confidence_threshold

    def evaluate_confidence(self, context, query):
        """
        Uses the LLM to estimate confidence in generating an answer.
        """
        prompt = f"Based on the following retrieved context, rate your confidence (0 to 1) in answering the query.\n\nContext:\n{context}\n\nQuery: {query}\n\nConfidence Score:"
        response = self.llm.invoke(prompt)
        try:
            return float(response.content.strip())
        except ValueError:
            return 0.5  # Default confidence if parsing fails

    def query(self, query):
        """
        Dynamically decides whether to retrieve more documents or generate an answer.
        """
        retrieved_texts = []
        iteration = 0

        while iteration < self.max_iters:
            iteration += 1
            logging.info(f"🔄 Iteration {iteration}: Retrieving documents...")

            # Retrieve more documents
            new_docs = self.db.similarity_search(query, k=self.k)
            retrieved_texts.extend([doc.page_content for doc in new_docs])

            # Deduplicate retrieved results
            retrieved_texts = list(set(retrieved_texts))
            context = "\n\n".join(retrieved_texts)

            # Estimate confidence
            logging.info("🧠 Evaluating confidence level...")
            confidence = self.evaluate_confidence(context, query)
            logging.info(f"🔹 Confidence Score: {confidence:.2f}")

            # If confidence is high enough, generate answer
            if confidence >= self.confidence_threshold:
                break  # Stop retrieving and proceed to answer generation

        # Final LLM Response
        final_prompt = f"Answer in very short to the point manner Using the following retrieved documents, answer the question:\n\n{context}\n\nQuestion: {query}"
        logging.info("🧠 Generating final response from LLM...")
        final_response = self.llm.invoke(final_prompt)

        return final_response.content if hasattr(final_response, "content") else "No response generated.", context

# Example Usage
if __name__ == "__main__":
    self_rag = SelfRAG()
    question = "when was kyoto protocol adopted ?"

    response, context = self_rag.query(question)
    print("\n📄 Retrieved Context:\n", context)
    print("\n🔹 Final LLM Response:\n", response)
