from evaluate import load


def calculate_bleu(reference, candidate):
    """
    Calculate the BLEU score for a given candidate sentence against a reference sentence using Hugging Face evaluate.

    :param reference: List of words in the reference sentence (ground truth)
    :param candidate: List of words in the candidate sentence (generated text)
    :return: BLEU score (float)
    """
    bleu = load("bleu")
    reference = [" ".join(reference)]  # Convert list of words to string
    candidate = [" ".join(candidate)]  # Convert list of words to string
    result = bleu.compute(predictions=candidate, references=reference, max_order=1)
    return result["bleu"]


# Example usage
if __name__ == "__main__":
    reference_sentence = "This is a test sentence for BLEU score calculation".split()
    candidate_sentence = "This is a test for BLEU calculation".split()

    bleu_score = calculate_bleu(reference_sentence, candidate_sentence)
    print(f"BLEU Score: {bleu_score:.4f}")
