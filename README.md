# RAG_Type

---

## **Project Overview**

**RAG_Type** is a Python-based implementation of Retrieval-Augmented Generation (RAG) techniques, designed to enhance information retrieval and generation processes. This project focuses on adaptive retrieval, semantic chunking, hierarchical indexing, explainability, and feedback loops to improve the performance of RAG models.

---

## **Key Features**

1. **Adaptive Retrieval**  
   Dynamically adjusts the retrieval process based on the query requirements.

2. **Explainable Retrieval**  
   Provides insights into why specific chunks or documents were retrieved.

3. **Semantic Chunking**  
   Breaks down large documents into semantically meaningful chunks for efficient processing.

4. **Hierarchical Indexing**  
   Implements hierarchical FAISS indices for scalable and efficient document retrieval.

5. **Feedback Loops**  
   Incorporates user feedback to refine retrieval and generation processes.

6. **Fusion Retrieval**  
   Combines multiple retrieval strategies for improved results.

---

## **Repository Structure**

| File/Folder                | Description                                                                 |
|----------------------------|-----------------------------------------------------------------------------|
| `adaptive_retrieval.py`    | Implements adaptive retrieval techniques.                                  |
| `explainable_retrieval.py` | Adds explainability to the retrieval process.                              |
| `semantic_chunking.py`     | Handles semantic chunking of documents.                                    |
| `hierarchical_indices.py`  | Builds hierarchical FAISS indices for document retrieval.                  |
| `fusion_retrieval.py`      | Implements fusion-based retrieval strategies.                              |
| `feedback_loop.py`         | Incorporates user feedback into the system.                                |
| `Main.py`                  | Main script to run the RAG pipeline.                                       |
| `Eval.py`                  | Evaluation script for measuring model performance.                         |
| `Excel_save.py`            | Saves evaluation results in Excel format.                                  |
| `pdf_loader.py`            | Loads PDF documents for processing.                                        |
| `Understanding_Climate_Change.pdf` | Example dataset used for testing purposes.                          |

---

## **Installation**

### Step 1: Clone the Repository
```bash
git clone https://github.com/vanshksingh/RAG_Type.git
cd RAG_Type
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Install FAISS (for efficient indexing)
```bash
pip install faiss-cpu
```

---

## **Usage Instructions**

1. Load your dataset using `pdf_loader.py`.
2. Configure retrieval settings in the respective scripts (`adaptive_retrieval.py`, etc.).
3. Run the main pipeline:
   ```bash
   python Main.py
   ```
4. Evaluate results:
   ```bash
   python Eval.py
   ```

---

## **Example Dataset**

The repository includes an example PDF file, *Understanding_Climate_Change.pdf*, to test and demonstrate the functionality of RAG_Type.

---

## **Contributing Guidelines**

We welcome contributions! Follow these steps to contribute:

1. Fork this repository.
2. Create a feature branch:
   ```bash
   git checkout -b feature-name
   ```
3. Commit your changes and push them to your forked repository.
4. Submit a pull request describing your changes.

---

## **License**

This project is licensed under the [MIT License](LICENSE).

---

## **Contact**

For issues or suggestions, feel free to open a GitHub issue in this repository!

