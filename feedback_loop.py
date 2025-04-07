import logging
from langchain_ollama import ChatOllama
from vector_db import VectorDB

logging.basicConfig(level=logging.INFO)

class FeedbackLoopRetrieval:
    def __init__(self,db_path="faiss_index", model_name="qwen2.5:0.5b-instruct", max_iterations=3, confidence_threshold=0.9 , temperature=0, EMBED_model_name="mxbai-embed-large:latest"):
        """
        Initializes the retrieval system with FAISS, BM25, and LLM-based feedback loops.
        """
        logging.info("🔄 Loading FAISS index and BM25 model...")
        self.db = VectorDB(db_path=db_path,model_name=EMBED_model_name)
        self.llm = ChatOllama(model=model_name, temperature=temperature)
        self.max_iterations = max_iterations
        self.confidence_threshold = confidence_threshold

    def evaluate_relevance(self, query, documents):
        """
        Uses LLM to evaluate retrieved documents and determine relevance.
        """
        prompt = f"Given the query: '{query}', rate the relevance of these documents (0-1 scale):\n\n"
        for i, doc in enumerate(documents):
            prompt += f"{i + 1}. {doc.page_content}\n"

        prompt += "\nReturn a list of scores and list only (e.g., [0.8, 0.6, 0.9])."
        response = self.llm.invoke(prompt)

        # ✅ FIX: Extract content from AIMessage
        response_content = response.content if hasattr(response, "content") else str(response)

        try:
            relevance_scores = eval(response_content)  # Convert string to list of floats
            if isinstance(relevance_scores, list) and all(isinstance(score, float) for score in relevance_scores):
                return relevance_scores
            else:
                raise ValueError("Relevance scores are not in the expected format.")
        except Exception as e:
            logging.warning(f"LLM returned invalid feedback: {e}. Using default scores.")
            return [0.5] * len(documents)  # Default neutral scores if an error occurs

    def query(self, question, k=5):
        """
        Performs retrieval with iterative feedback loops.
        """
        iteration = 0
        confidence = 0
        best_docs = []

        while iteration < self.max_iterations and confidence < self.confidence_threshold:
            logging.info(f"🔄 Iteration {iteration + 1}: Retrieving documents...")

            faiss_results = self.db.similarity_search(question, k)
            bm25_results = self.db.bm25_search(question, k)

            combined_results = faiss_results + bm25_results
            relevance_scores = self.evaluate_relevance(question, combined_results)

            # Sort by relevance scores
            sorted_docs = sorted(zip(combined_results, relevance_scores), key=lambda x: x[1], reverse=True)
            best_docs = [doc[0].page_content for doc in sorted_docs[:3]]
            confidence = max(relevance_scores)

            logging.info(f"Confidence Score: {confidence:.2f}")
            if confidence >= self.confidence_threshold:
                break

            # ✅ FIX: Extract content from AIMessage
            refine_prompt = f"The current query is: '{question}'. Given the retrieved documents:\n" + "\n".join(
                best_docs) + "\n\nSuggest a refined query to improve retrieval."
            refined_query = self.llm.invoke(refine_prompt)
            question = refined_query.content if hasattr(refined_query, "content") else str(refined_query)

            iteration += 1

        # ✅ FIX: Extract content from AIMessage
        final_prompt = f"Answer in very short to the point manner Based on the retrieved documents, answer: {question}\n\nContext:\n" + "\n".join(best_docs)
        final_response = self.llm.invoke(final_prompt)

        # Return context and response
        return final_response.content if hasattr(final_response, "content") else str(final_response) , best_docs


# Run the retrieval system
if __name__ == "__main__":
    feedback_rag = FeedbackLoopRetrieval()
    question = "What is the document about?"
    response, context = feedback_rag.query(question)

    print("\n📄 Retrieved Context:\n", "\n".join(context))
    print("\n💡 LLM Response:\n", response)

