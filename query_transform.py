import logging
from langchain_ollama import ChatOllama
from vector_db import VectorDB  # FAISS + BM25 Hybrid Retrieval

logging.basicConfig(level=logging.INFO)

class QueryTransformRetrieval:
    def __init__(self, db_path="faiss_index", model_name="qwen2.5:0.5b-instruct", k=5, temperature=0, EMBED_model_name="mxbai-embed-large:latest"):
        """
        Initializes the query transformation engine, vector DB, and LLM.
        """
        self.llm = ChatOllama(model=model_name, temperature=temperature)
        self.db = VectorDB(db_path=db_path, model_name=EMBED_model_name)  # FAISS + BM25 Hybrid Retrieval
        self.k = k

    def multi_query(self, query, num_queries=3):
        """
        Generates multiple reworded versions of the query.
        """
        prompt = f"Generate {num_queries} different ways to ask the following query:\n\n{query}"
        response = self.llm.invoke(prompt)
        return response.content.split("\n") if hasattr(response, "content") else [query]

    def query_decomposition(self, query):
        """
        Breaks a complex query into simpler sub-questions.
        """
        prompt = f"Break down the following query into simpler sub-questions:\n\n{query}"
        response = self.llm.invoke(prompt)
        return response.content.split("\n") if hasattr(response, "content") else [query]

    def query_expansion(self, query):
        """
        Expands the query with related keywords and synonyms.
        """
        prompt = f"Expand the following query by adding synonyms, related concepts, and variations:\n\n{query}"
        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, "content") else query

    def hyde(self, query):
        """
        Uses HyDE (Hypothetical Document Embedding) by generating a pseudo-relevant document.
        """
        prompt = f"Generate a hypothetical passage that could be the answer to:\n\n{query}"
        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, "content") else query

    def query(self, query):
        """
        Transforms the query, retrieves documents from FAISS, and generates a final LLM response.
        """
        logging.info("🔄 Generating Multi-Query Variations...")
        multi_queries = self.multi_query(query)

        logging.info("🔄 Performing Query Decomposition...")
        sub_queries = self.query_decomposition(query)

        logging.info("🔄 Expanding Query with Additional Context...")
        expanded_query = self.query_expansion(query)

        logging.info("🔄 Generating HyDE Hypothetical Document...")
        hyde_doc = self.hyde(query)

        # 🔄 Combine all query variations
        all_queries = list(set(multi_queries + sub_queries + [expanded_query, hyde_doc]))

        logging.info("🔎 Retrieving documents from FAISS...")
        retrieved_docs = []
        for q in all_queries:
            docs = self.db.similarity_search(q, k=self.k)
            retrieved_docs.extend(docs)

        # 🔹 Deduplicate results
        retrieved_texts = list(set([doc.page_content for doc in retrieved_docs]))

        # 🔹 Final LLM Response
        context = "\n\n".join(retrieved_texts)
        final_prompt = f"Answer in very short to the point manner Using the following retrieved documents, answer the question:\n\n{context}\n\nQuestion: {query}"

        logging.info("🧠 Generating final response from LLM...")
        final_response = self.llm.invoke(final_prompt)

        return final_response.content if hasattr(final_response, "content") else "No response generated.", context


# Example Usage
if __name__ == "__main__":
    query_transform_rag = QueryTransformRetrieval()
    question = "What does document talk about?"

    response, context = query_transform_rag.query(question)
    print("\n📄 Retrieved Context:\n", context)
    print("\n🔹 Final LLM Response:\n", response)
