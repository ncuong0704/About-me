# Email/SMS Spam Classification with Vector Database (FAISS + Embedding)

This project demonstrates how to build a simple **vector database** for the problem of **spam/ham text classification** using **sentence embedding** and **k-NN retrieval**.

Instead of training a traditional classifier, the notebook does the following:

- Turns message text into vectors using the embedding model **`intfloat/multilingual-e5-base`** (Transformers).
- Stores those vectors in **FAISS** (a vector search index).
- When you input a new message: the text is encoded to a vector → searches for the top-\(k\) most similar vectors → uses **majority voting** among neighbors to predict `spam` or `ham`. It also shows the neighbor messages for explanation.

## Folder Contents

```text
nlp/email_spam_vector_database/
├─ main.ipynb              # Main notebook
└─ cls_spam_text.csv       # Dataset (Category, Message)
```

## Dependencies

The notebook installs these Python packages:

- `faiss-cpu`
- `transformers`
- `torch`
- `pandas`
- `numpy`
- `scikit-learn`
- `tqdm`

To install (recommended to use a virtual environment):

```bash
pip install faiss-cpu transformers torch pandas numpy scikit-learn tqdm
```

## How to Run

Open `main.ipynb` and run each cell in order using Jupyter Notebook or VSCode.

## Notebook Pipeline

1. **Read the data** from `cls_spam_text.csv` (columns: `Category`, `Message`)
2. **Load the embedding model**:
   - `MODEL_NAME = "intfloat/multilingual-e5-base"`
   - Picks `cuda` if available, otherwise `cpu`
3. **Create embeddings**:
   - Tokenize messages with `AutoTokenizer`
   - Take `last_hidden_state` and apply **average pooling** using the `attention_mask`
   - **L2-normalize** the embeddings (so inner product is nearly cosine similarity)
4. **Train/test split** (default is 90/10, stratified by label)
5. **Build the FAISS index**:
   - `faiss.IndexFlatIP(d)` where \(d\) is embedding dimension
   - `index.add(X_train_emb.astype("float32"))`
6. **Classify with k-NN**:
   - Encode new message → `index.search(query_vector, k)`
   - Get top neighbors + similarity scores
   - Use **majority vote** to predict the label

## Current “Vector DB” Approach

- **Index type**: `faiss.IndexFlatIP` (exact search, no compression/ANN).
- **Similarity metric**: inner product (IP). Since our embeddings are normalized, IP is nearly **cosine similarity**.
- **Interpretability**: shows top neighbors (label, score, content) so you can see why the prediction was made.

## Example Usage (in notebook)

The notebook comes with a list of `test_examples` and runs:

- `spam_classifier_pipeline(example, k=3)`

The function will display:

- Input message
- Predicted label (`HAM`/`SPAM`)
- Top-\(k\) nearest neighbors and their similarity scores
- Label distribution among the neighbors

## Notes

- **Speed**: Generating embeddings for the whole dataset is the slowest step. Using a GPU is much faster.
- **Windows + HuggingFace cache warning**: You might see a symlink warning when downloading models on Windows; it does not affect functionality (but may use more disk space).
- **Evaluation**: The notebook currently showcases retrieval and test examples, but does not include true performance metrics. For a proper evaluation, run predictions for the test set (retrieval from train set) and compute accuracy/F1.

## Possible Extensions

- Use **FAISS ANN** (like IVF or HNSW) for greater scalability.
- Save the index + metadata to disk so that you can reuse it next time without recomputing all embeddings.
- Add a **preprocessing** step (lowercase, normalize unicode, remove URLs/phone numbers) depending on your dataset.