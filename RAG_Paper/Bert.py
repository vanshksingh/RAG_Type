from evaluate import load
import torch


def calculate_bert_score(reference, candidate):
    """
    Fast computation of BERTScore using a smaller model and GPU if available.

    :param reference: List of words in the reference sentence (ground truth)
    :param candidate: List of words in the candidate sentence (generated text)
    :return: BERTScore (dict) containing precision, recall, and F1-score
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bertscore = load("bertscore")
    reference = " ".join(reference)  # Convert list of words to string
    candidate = " ".join(candidate)  # Convert list of words to string
    result = bertscore.compute(predictions=[candidate], references=[reference], model_type="roberta-base",
                               device=device)
    return {
        "Precision": round(result["precision"][0], 4),
        "Recall": round(result["recall"][0], 4),
        "F1": round(result["f1"][0], 4)
    }


# Example usage
if __name__ == "__main__":
    reference_sentence = "This is a test sentence for BERTScore calculation".split()
    candidate_sentence = "This is a test for BERTScore calculation".split()

    bert_scores = calculate_bert_score(reference_sentence, candidate_sentence)
    print("BERT Scores:", bert_scores)
