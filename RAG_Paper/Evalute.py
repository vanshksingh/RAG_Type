import torch
from evaluate import load


def calculate_bert_score(reference, candidate):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bertscore = load("bertscore")
    reference = " ".join(reference.split())
    candidate = " ".join(candidate.split())
    result = bertscore.compute(predictions=[candidate], references=[reference], model_type="roberta-base",
                               device=device)
    return {
        "Precision": round(result["precision"][0], 4),
        "Recall": round(result["recall"][0], 4),
        "F1": round(result["f1"][0], 4)
    }


def calculate_bleu(reference, candidate):
    bleu = load("bleu")
    reference = [" ".join(reference.split())]
    candidate = [" ".join(candidate.split())]
    result = bleu.compute(predictions=candidate, references=reference, max_order=1)
    return {"BLEU Score": round(result["bleu"], 4)}


def calculate_meteor(reference, candidate):
    meteor = load("meteor")
    result = meteor.compute(predictions=[" ".join(candidate.split())], references=[[" ".join(reference.split())]])
    return {"METEOR Score": round(result["meteor"], 4)}


def calculate_rouge(reference, candidate):
    rouge = load("rouge")
    reference = " ".join(reference.split())
    candidate = " ".join(candidate.split())
    result = rouge.compute(predictions=[candidate], references=[reference])
    return {
        "ROUGE-1": round(result["rouge1"], 4),
        "ROUGE-2": round(result["rouge2"], 4),
        "ROUGE-L": round(result["rougeL"], 4)
    }


def evaluate_text(reference_sentence, candidate_sentence):
    bert_scores = calculate_bert_score(reference_sentence, candidate_sentence)
    bleu_score = calculate_bleu(reference_sentence, candidate_sentence)
    meteor_score = calculate_meteor(reference_sentence, candidate_sentence)
    rouge_scores = calculate_rouge(reference_sentence, candidate_sentence)

    return {**bert_scores, **bleu_score, **meteor_score, **rouge_scores}


if __name__ == "__main__":
    reference_sentence = "This is a test sentence for evaluation"
    candidate_sentence = "This is a test for evaluation"

    scores = evaluate_text(reference_sentence, candidate_sentence)
    #print("Evaluation Scores:", scores)
    #print(scores)
    print("BERT Precision:", scores["Precision"])
    print("BERT Recall:", scores["Recall"])
    print("BERT F1 Score:", scores["F1"])
    print("BLEU Score:", scores["BLEU Score"])
    print("METEOR Score:", scores["METEOR Score"])
    print("ROUGE-1:", scores["ROUGE-1"])
    print("ROUGE-2:", scores["ROUGE-2"])
    print("ROUGE-L:", scores["ROUGE-L"])
