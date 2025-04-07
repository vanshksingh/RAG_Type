import logging
from langchain_community.chat_models import ChatOllama
from vector_db import VectorDB

logging.basicConfig(level=logging.INFO)

class FusionRetrieval:
    def __init__(self, db_path="faiss_index", model_name="qwen2.5:0.5b-instruct", temperature=0, EMBED_model_name="mxbai-embed-large:latest"):
        """
        Initializes the retrieval system with FAISS (dense) and BM25 (sparse) retrieval.
        """
        logging.info("🔄 Loading FAISS index and BM25 model...")
        self.db = VectorDB(db_path=db_path, model_name=EMBED_model_name)
        self.llm = ChatOllama(model=model_name, temperature=temperature)

    def reciprocal_rank_fusion(self, results1, results2, k=60):
        """
        Implements Reciprocal Rank Fusion (RRF) to combine FAISS & BM25 retrieval.
        """
        fused_scores = {}
        for rank, doc in enumerate(results1):
            fused_scores[doc.page_content] = fused_scores.get(doc.page_content, 0) + 1 / (rank + k)
        for rank, doc in enumerate(results2):
            fused_scores[doc.page_content] = fused_scores.get(doc.page_content, 0) + 1 / (rank + k)

        sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        return [doc[0] for doc in sorted_docs]

    def query(self, question, k=5):
        """
        Performs hybrid retrieval using FAISS and BM25, then fuses results.
        """
        logging.info("INFO: Retrieving using FAISS...")
        faiss_results = self.db.similarity_search(question, k)

        logging.info("INFO: Retrieving using BM25...")
        bm25_results = self.db.bm25_search(question, k)

        logging.info("INFO: Performing Fusion Retrieval...")
        fused_docs = self.reciprocal_rank_fusion(faiss_results, bm25_results)
        context = "\n".join(fused_docs[:3])

        logging.info("INFO: Generating response with LLM...")
        prompt = f"Answer in very short to the point manner Based on the retrieved documents, answer: {question}\n\nContext:\n" + context
        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, "content") else "No response generated.", context

# Run the retrieval system
if __name__ == "__main__":
    fusion_rag = FusionRetrieval()
    question = "What is the document about?"
    response, context = fusion_rag.query(question)

    print("\n📄 Retrieved Context:\n", context)
    print("\n💡 LLM Response:\n", response)
