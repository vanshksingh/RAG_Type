import json
import csv
import os


def save_results(file_path, param0, param1, param2, param3, param4 , param5 , param6 , param7 , param8 , param9 ):
    """Appends data to a CSV file without repeating the header."""
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write header only if the file is new/empty
        if not file_exists or os.stat(file_path).st_size == 0:
            writer.writerow(["RAG_Type", "Embedding_Type", "Model_Type", "Temp","faithfulness","answer_relevancy","context_precision","context_recall","answer_correctness","answer_similarity"])

        # Append the data
        writer.writerow([param0, param1, param2, param3, param4, param5 , param6 , param7 , param8 , param9 ])


def save_to_csv(file_path, param0, param1, param2, param3, param4 , param5 , param6 , param7):
    """Appends data to a CSV file without repeating the header."""
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write header only if the file is new/empty
        if not file_exists or os.stat(file_path).st_size == 0:
            writer.writerow(["RAG_Type", "Embedding_Type", "Model_Type", "Temp","Question","Reference_Answer","Response","Context"])

        # Append the data
        writer.writerow([param0, param1, param2, param3, param4, param5 , param6 , param7 ])



def save_standardised(file_path, param0, param1, param2, param3, param4 , param5 , param6 , param7 , param8 , param9 ):
    """Appends data to a CSV file without repeating the header."""
    file_exists = os.path.isfile(file_path)

    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)

        # Write header only if the file is new/empty
        if not file_exists or os.stat(file_path).st_size == 0:
            writer.writerow(["RAG_Type","Model_Type","Precision","Recall","F1","BLEU","METEOR","ROUGE-1","ROUGE-2","ROUGE-L"])

        # Append the data
        writer.writerow([param0, param1, param2, param3, param4, param5 , param6 , param7 , param8 , param9 ])



# Example usage
# json_dump = '{"faithfulness": 1.0, "answer_relevancy": 0.9, "context_precision": 0.85, "context_recall": 0.92, "answer_correctness": 0.88, "answer_similarity": 0.91}'
# save_results(json_dump, "Model_A")
