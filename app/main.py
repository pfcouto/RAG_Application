import os
from flask import Flask, request, jsonify
import faiss
from embeddings import loadDataset, get_embeddings, get_relevant_snippets
from model_integration import generate_answer
from dotenv import load_dotenv
import traceback


app = Flask(__name__)
load_dotenv()

DATASET_PATH = "data/dataset.txt"
INDEX_PATH = "data/dataset.index"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GENERATION_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
API_KEY = os.environ.get("API_KEY")


def initialize_embeddings():
    """
    Load dataset and create/load FAISS index.
    """
    dataset = loadDataset(DATASET_PATH)
    if os.path.exists(INDEX_PATH):
        print("Loading FAISS index from disk...")
        index = faiss.read_index(INDEX_PATH)

        if index.ntotal != len(dataset):
            print("Index and dataset mismatch. Regenerating index...")
            _, index = get_embeddings(dataset, model_name=EMBEDDING_MODEL)
            faiss.write_index(index, INDEX_PATH)
    else:
        print("Generating embeddings and creating FAISS index...")
        _, index = get_embeddings(dataset, model_name=EMBEDDING_MODEL)
        faiss.write_index(index, INDEX_PATH)
        print(f"FAISS index saved to {INDEX_PATH}.")

    return dataset, index


print("Initializing embeddings and FAISS index...")
dataset, index = initialize_embeddings()
print("Initialization complete.")


@app.route("/ask", methods=["POST"])
def ask():
    """
    POST /ask endpoint to handle user questions.
    """
    try:
        data = request.get_json()
        if "question" not in data:
            return jsonify({"error": "Missing 'question' in request body."}), 400

        question = data["question"]

        snippets = get_relevant_snippets(
            query=question,
            index=index,
            dataset=dataset,
            model_name=EMBEDDING_MODEL,
            top_k=3,
        )

        answer = generate_answer(
            query=question,
            snippets=snippets,
            model_name=GENERATION_MODEL,
            api_key=API_KEY,
        )

        response = {
            "answer": answer,
            "relevant_snippets": snippets,
        }

        return jsonify(response)

    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()
        return jsonify({"error": "An internal error occurred."}), 500


# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
