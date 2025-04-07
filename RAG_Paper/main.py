import csv
from Evalute import evaluate_text
from Excel_save import save_standardised


def read_csv_as_dict(file_path):
    with open(file_path, newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)  # Reads data as a list of dictionaries
        data = [row for row in reader]  # Convert reader object to a list
    return data

# Example usage
file_path = "model_params.csv"
csv_data = read_csv_as_dict(file_path)

# Print structured output
for row in csv_data:
    #print(row)  # Each row is a dictionary with column names as keys
    #print(row[" RAG_Type"] , row["Model_Type"] , row["Reference_Answer"] , row["Response"])  # Access a specific column value

    scores = evaluate_text(row["Reference_Answer"], row["Response"])
    save_standardised("standardised_results.csv", row[" RAG_Type"], row["Model_Type"], scores["Precision"], scores["Recall"], scores["F1"], scores["BLEU Score"], scores["METEOR Score"], scores["ROUGE-1"], scores["ROUGE-2"], scores["ROUGE-L"])




