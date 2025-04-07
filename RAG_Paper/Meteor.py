import evaluate


def calculate_meteor(reference, candidate):
    """
    Compute METEOR score using the Hugging Face evaluate module.

    :param reference: Reference sentence (string)
    :param candidate: Candidate sentence (string)
    :return: METEOR score
    """
    meteor = evaluate.load("meteor")
    result = meteor.compute(predictions=[candidate], references=[[reference]])
    return {"METEOR Score": round(result["meteor"], 4)}


# Example usage
if __name__ == "__main__":
    reference_sentence = "This is a test sentence for METEOR calculation"
    candidate_sentence = "This is a test for METEOR calculation"

    meteor_score = calculate_meteor(reference_sentence, candidate_sentence)
    print("METEOR Score:", meteor_score)