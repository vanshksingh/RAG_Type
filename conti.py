from adaptive_retrieval import *
from Eval import *
from Excel_save import *
from explainable_retrieval import *
from feedback_loop import *
from fusion_retrieval import *
from hierarchical_indices import *
from pdf_loader import *
from query_transform import *
from self_rag import *
from semantic_chunking import *
from simple_rag import *
from vector_db import *
import json
import time
import os

# Define available embedding models and LLMs
embedding_models = ["mxbai-embed-large:latest"]
main_models = ["qwen2.5:0.5b-instruct" ]
temperatures = [0.0]  # Different temperature settings

# Load the PDF into FAISS for each embedding model
def FAISSdb(embedding_model):
    db = VectorDB(db_path=f"faiss_index_{embedding_model.replace(':', '_')}", model_name=embedding_model)
    pdf_path = "/Users/vanshksingh/PycharmProjects/RAG_Type/Understanding_Climate_Change.pdf"
    if os.path.exists(pdf_path):
        chunks = load_and_chunk_pdf(pdf_path, chunk_size=500, chunk_overlap=100)
        db.add_documents(chunks)
        rag = HierarchicalRAG()
        rag.add_pdf(pdf_path)

# Define test queries and reference answers
queries = {
    "What are the primary causes of climate change as identified in the document?":
        "The primary causes of climate change include the increase in greenhouse gases (such as CO₂, CH₄, and N₂O), burning fossil fuels (coal, oil, natural gas), deforestation, and agricultural activities (such as livestock emissions and fertilizer use).",
    "How does climate change affect global sea levels?":
        "Climate change contributes to rising sea levels through the melting of polar ice caps and glaciers, as well as the thermal expansion of seawater. The document notes that sea levels have risen by approximately 20 centimeters (8 inches) over the past century, posing threats to coastal communities and ecosystems.",
    "What are some of the mitigation strategies proposed to combat climate change?":
        "Mitigation strategies include transitioning to renewable energy sources (such as solar, wind, and hydroelectric power), improving energy efficiency, implementing carbon capture and storage (CCS) technologies, afforestation and reforestation, and promoting sustainable agricultural practices.",
    "How does climate change impact human health?":
        "Climate change affects human health through increased heat-related illnesses, the spread of vector-borne diseases (such as malaria and dengue), worsening air quality leading to respiratory and cardiovascular diseases, and food and water insecurity due to extreme weather events.",
    "What role do international agreements play in addressing climate change?":
        "International agreements, such as the Paris Agreement and the Kyoto Protocol, set global targets for reducing greenhouse gas emissions. They encourage countries to submit nationally determined contributions (NDCs) and promote cooperation on climate action, technology transfer, and financial support for developing nations.",
}

# Run RAG models with different configurations
def run_rag_models():
    results = []
    for embedding_model in embedding_models:
        # FAISSdb(embedding_model) # Load documents for each embedding model
        pass
        for main_model in main_models:
            for temperature in temperatures:
                rag_models = {
                    "SimpleRAG": SimpleRAG(db_path=f"faiss_index_{embedding_model.replace(':', '_')}", model_name=main_model, temperature=temperature, EMBED_model_name=embedding_model),
                    "AdaptiveRetrieval": AdaptiveRetrieval(db_path=f"faiss_index_{embedding_model.replace(':', '_')}", model_name=main_model, temperature=temperature, EMBED_model_name=embedding_model),
                    "SemanticRAG": SemanticRAG(db_path=f"faiss_index_{embedding_model.replace(':', '_')}", model_name=main_model, temperature=temperature, EMBED_model_name=embedding_model),
                    "SelfRAG": SelfRAG(db_path=f"faiss_index_{embedding_model.replace(':', '_')}", model_name=main_model, temperature=temperature, EMBED_model_name=embedding_model),
                    "QueryTransformRetrieval": QueryTransformRetrieval(db_path=f"faiss_index_{embedding_model.replace(':', '_')}", model_name=main_model, temperature=temperature, EMBED_model_name=embedding_model),
                    "HierarchicalRAG": HierarchicalRAG(db_path=f"faiss_index_{embedding_model.replace(':', '_')}", model_name=main_model, temperature=temperature, EMBED_model_name=embedding_model),
                    "FusionRetrieval": FusionRetrieval(db_path=f"faiss_index_{embedding_model.replace(':', '_')}", model_name=main_model, temperature=temperature, EMBED_model_name=embedding_model),
                    "FeedbackLoopRetrieval": FeedbackLoopRetrieval(db_path=f"faiss_index_{embedding_model.replace(':', '_')}", model_name=main_model, temperature=temperature, EMBED_model_name=embedding_model),
                    "ExplainableRetrieval": ExplainableRetrieval(db_path=f"faiss_index_{embedding_model.replace(':', '_')}", model_name=main_model, temperature=temperature, EMBED_model_name=embedding_model)
                }
                for rag_name, rag_instance in rag_models.items():
                    for question, reference_answer in queries.items():
                        response, context = rag_instance.query(question)
                        save_to_csv("model_params.csv", rag_name, embedding_model, main_model, temperature, question, reference_answer, response, context)
                        print(f"\n📄 {rag_name} - Retrieved Context:\n", context)
                        print(f"\n💡 {rag_name} - LLM Response:\n", response)
                        eval_results = evaluate_rag(query=question, retrieved_context=context, generated_answer=response, reference_answer=reference_answer, model="mistral:7b-instruct") #eval model mistral:7b-instruct
                        save_results("eval_params.csv", rag_name, embedding_model, main_model, temperature, eval_results["faithfulness"], eval_results["answer_relevancy"], eval_results["context_precision"], eval_results["context_recall"], eval_results["answer_correctness"], eval_results["answer_similarity"] )
                        time.sleep(1)
    return None

if __name__ == "__main__":
    results = run_rag_models()
    print("COMPLETED")
