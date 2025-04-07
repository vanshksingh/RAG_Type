import logging
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from vector_db import VectorDB

logging.basicConfig(level=logging.INFO)


class AdaptiveRetrieval:
    def __init__(self, db_path="faiss_index", model_name="qwen2.5:0.5b-instruct", temperature=0 , EMBED_model_name="mxbai-embed-large:latest"):
        """
        Initializes the adaptive retrieval system with an LLM and a vector database.
        """
        self.llm = ChatOllama(model=model_name , temperature=temperature)
        self.db = VectorDB(db_path=db_path,model_name=EMBED_model_name)

    def classify_query(self, query):
        """
        Uses an LLM to classify the type of query: factual, analytical, opinion-based, or contextual.
        """
        classification_prompt = f"""
        Classify the following query into one of the categories:
        - Factual
        - Analytical
        - Opinion-Based
        - Contextual

        Query: "{query}"
        Category:
        """

        messages = [HumanMessage(content=classification_prompt)]
        return self.llm.invoke(messages).content.strip()

    def enhance_factual_query(self, query):
        """
        Expands a factual query to improve retrieval accuracy.
        """
        expansion_prompt = f"""
        Improve the following factual query by making it more specific and contextually rich:

        Original Query: "{query}"
        Improved Query:
        """
        messages = [HumanMessage(content=expansion_prompt)]
        return self.llm.invoke(messages).content.strip()

    def generate_sub_queries(self, query):
        """
        Generates diverse sub-queries for analytical questions to improve retrieval coverage.
        """
        subquery_prompt = f"""
        Break down the following analytical query into multiple sub-queries to ensure a comprehensive response:

        Query: "{query}"
        Sub-Queries:
        """
        messages = [HumanMessage(content=subquery_prompt)]
        sub_queries = self.llm.invoke(messages).content.strip().split("\n")
        return [sq for sq in sub_queries if sq]

    def identify_opinion_viewpoints(self, query):
        """
        Identifies different viewpoints for opinion-based queries.
        """
        opinion_prompt = f"""
        Identify multiple perspectives on the following opinion-based query:

        Query: "{query}"
        Perspectives:
        """
        messages = [HumanMessage(content=opinion_prompt)]
        viewpoints = self.llm.invoke(messages).content.strip().split("\n")
        return [v for v in viewpoints if v]

    def query(self, question, k=5):
        """
        Processes the query adaptively, retrieves relevant documents, and generates a response.
        """
        query_type = self.classify_query(question)
        logging.info(f"Query Type Identified: {query_type}")

        # Initialize retrieved_docs to an empty list
        retrieved_docs = []

        if query_type == "Factual":
            question = self.enhance_factual_query(question)
            retrieved_docs = self.db.similarity_search(question, k=k)
        elif query_type == "Analytical":
            sub_queries = self.generate_sub_queries(question)
            for sub_query in sub_queries:
                retrieved_docs.extend(self.db.similarity_search(sub_query, k=k))
        elif query_type == "Opinion-Based":
            viewpoints = self.identify_opinion_viewpoints(question)
            for viewpoint in viewpoints:
                retrieved_docs.extend(self.db.similarity_search(viewpoint, k=k))
        else:  # Default retrieval for contextual queries
            retrieved_docs = self.db.similarity_search(question, k=k)

        # If no documents are retrieved, set a default context
        if not retrieved_docs:
            context = "No relevant documents found."
        else:
            context = "\n".join([doc.page_content for doc in retrieved_docs])

        # Create the prompt for the LLM
        prompt = f"""
        Answer in very short to the point manner
        Use the retrieved context to generate an accurate response.

        Context:
        {context}

        Question: {question}

        Answer:
        """
        messages = [HumanMessage(content=prompt)]
        answer = self.llm.invoke(messages).content.strip()

        # Return both the context and the answer
        return answer, context


if __name__ == "__main__":
    adaptive_rag = AdaptiveRetrieval()
    question = "What does document talk about?"
    response, context = adaptive_rag.query(question)

    print("\n📄 Retrieved Context:\n", context)
    print("\n💡 LLM Response:\n", response)
