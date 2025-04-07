import ollama
from Excel_save import *


def truncate_text(text, max_length):
    if isinstance(text, list):  # Convert list to string if needed
        text = " ".join(text)
    return text[:max_length] + ('...' if len(text) > max_length else '')


def ask_ollama(prompt, model='qwen2.5:0.5b-instruct'):
    """Helper function to query Ollama and return the response."""
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "user", "content": prompt + "\nOnly return a single number between 0 and 1. and nothing else no explanation needed"}],
        options={"temperature": 0}
    )
    return response.get("message", {}).get("content", "")


def evaluate_rag(query, retrieved_context, generated_answer, reference_answer=None, model='qwen2.5:0.5b-instruct'):
    """Evaluates RAG performance using an LLM from Ollama."""

    # Define max length for each input field
    MAX_QUERY_LENGTH = 200
    MAX_CONTEXT_LENGTH = 500
    MAX_ANSWER_LENGTH = 300
    MAX_REF_ANSWER_LENGTH = 300

    # Truncate inputs
    query = truncate_text(query, MAX_QUERY_LENGTH)
    retrieved_context = truncate_text(retrieved_context, MAX_CONTEXT_LENGTH)
    generated_answer = truncate_text(generated_answer, MAX_ANSWER_LENGTH)
    reference_answer = truncate_text(reference_answer, MAX_REF_ANSWER_LENGTH) if reference_answer else None

    # Define evaluation prompts
    prompts = {
        "faithfulness": f"""
        Given the retrieved context:
        "{retrieved_context}"

        And the generated answer:
        "{generated_answer}"

        Does the generated answer contain information that is incorrect or not supported by the context? 
        Respond with a score from 0 (completely hallucinated) to 1 (fully faithful).
        """,

        "answer_relevancy": f"""
        Given the question:
        "{query}"

        And the generated answer:
        "{generated_answer}"

        How relevant is the answer to the question? Score between 0 (completely irrelevant) to 1 (fully relevant).
        """,

        "context_precision": f"""
        Given the question:
        "{query}"

        And the retrieved context:
        "{retrieved_context}"

        How well does the retrieved context focus only on information relevant to the question? 
        Score between 0 (completely irrelevant) to 1 (perfectly precise).
        """,

        "context_recall": f"""
        Given the question:
        "{query}"

        And the retrieved context:
        "{retrieved_context}"

        Does the context contain all necessary information to answer the question? 
        Score between 0 (missing critical details) to 1 (fully complete).
        """,

        "answer_correctness": f"""
        Given the reference answer:
        "{reference_answer}"

        And the generated answer:
        "{generated_answer}"

        How correct is the generated answer compared to the reference answer? 
        Score between 0 (completely incorrect) to 1 (fully correct).
        """ if reference_answer else None,

        "answer_similarity": f"""
        Compare the generated answer:
        "{generated_answer}"

        With the reference answer:
        "{reference_answer}"

        Compute a semantic similarity score between 0 (completely different) to 1 (identical meaning).
        """ if reference_answer else None,
    }

    # Compute scores
    scores = {}
    for key, prompt in prompts.items():
        if prompt:  # Some metrics are skipped if reference answer is unavailable
            score_response = ask_ollama(prompt, model=model)
            try:
                scores[key] = float(score_response.strip())
            except ValueError:
                scores[key] = score_response  # Store raw response if conversion fails

    return scores


if __name__ == "__main__":
    # Example usage
    data = {
        "query": "What is the capital of France?",
        "retrieved_context": "Paris is the capital and most populous city of France.",
        "generated_answer": "The capital of France is in paris",
        "reference_answer": "Paris is the capital of France."
    }

    results = evaluate_rag(
        query=data["query"],
        retrieved_context=data["retrieved_context"],
        generated_answer=data["generated_answer"],
        reference_answer=data.get("reference_answer"),
        model='mistral:7b-instruct'  # Change model if needed
    )

    print(list(results.values()))
    print((results))

    #save_results(json.dumps(results),"MistralX")  # Convert dictionary to JSON string
