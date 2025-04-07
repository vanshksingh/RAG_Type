from vector_db import VectorDB
from langchain_community.chat_models import ChatOllama


class SimpleRAG:
    def __init__(self, db_path="faiss_index", model_name="qwen2.5:0.5b-instruct", temperature=0, EMBED_model_name="mxbai-embed-large:latest"):
        """
        Initializes the Simple RAG system.

        :param db_path: Path to the FAISS index.
        :param model_name: Name of the LLM model to use.
        """
        self.db = VectorDB(db_path=db_path, model_name=EMBED_model_name)
        self.llm = ChatOllama(model=model_name, temperature=temperature)

    def query(self, question, k=5):
        """
        Perform a simple RAG pipeline by retrieving relevant documents and generating a response.

        :param question: The user query.
        :param k: Number of documents to retrieve.
        :return: Tuple containing LLM-generated response and retrieved context.
        """
        retrieved_docs = self.db.similarity_search(query=question, k=k)

        print(retrieved_docs)
        print("FAISS Index Size:", self.db.vector_db.index.ntotal)

        context = "\n".join([doc.page_content for doc in retrieved_docs])

        prompt = f"""Answer in very short to the point manner
        Context:
        {context}

        Question: {question}

        Answer: """

        response = self.llm.invoke(prompt)
        return response.content, context


if __name__ == "__main__":
    rag = SimpleRAG(temperature=0)
    question = "What does the document say about climate change?"
    response, context = rag.query(question)
    print("\n📄 Retrieved Context:")
    print(context)
    print("\n🤖 LLM Response:")
    print(response)
