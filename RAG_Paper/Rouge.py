from evaluate import load


def calculate_rouge(reference, candidate):
    """
    Calculate the ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L) for a given candidate sentence
    against a reference sentence using Hugging Face evaluate.

    :param reference: List of words in the reference sentence (ground truth)
    :param candidate: List of words in the candidate sentence (generated text)
    :return: ROUGE scores (dict) containing ROUGE-1, ROUGE-2, and ROUGE-L
    """
    rouge = load("rouge")
    reference = " ".join(reference)  # Convert list of words to string
    candidate = " ".join(candidate)  # Convert list of words to string
    result = rouge.compute(predictions=[candidate], references=[reference])
    return {
        "ROUGE-1": result["rouge1"],
        "ROUGE-2": result["rouge2"],
        "ROUGE-L": result["rougeL"]
    }


# Example usage
if __name__ == "__main__":
    reference_sentence = "This is a test sentence for ROUGE score calculation".split()
    candidate_sentence = "This is a test for ROUGE calculation".split()

    rouge_scores = calculate_rouge(reference_sentence, candidate_sentence)
    print("ROUGE Scores:", rouge_scores)
