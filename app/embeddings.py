from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from dotenv import load_dotenv
import os

load_dotenv()
DEVICE = os.getenv("DEVICE", "cpu")


def loadDataset(dataset_path: str) -> list:
    """
    Load the dataset from the given file path, splitting it into paragraphs.

    Args:
    - dataset_path (str): Path to the dataset file.

    Returns:
    - list: A list of paragraphs from the dataset.
    """

    with open(dataset_path, "r") as file:
        text = file.read()
        dataset = []
        for para in text.split("\n\n"):
            if para.strip():
                dataset.append(para.strip())
    # print(f"Dataset:\n {dataset}")
    return dataset


def get_embeddings(dataset: list, model_name: str = "all-MiniLM-L6-v2") -> tuple:
    """
    Generate embeddings for the dataset using SentenceTransformers and create a FAISS index.

    Args:
    - dataset (list): List of paragraphs.
    - model_name (str): The SentenceTransformer model to use for embeddings.

    Returns:
    - np.ndarray: The generated embeddings.
    - faiss.IndexFlatL2: A FAISS index containing the embeddings.
    """

    embedding_model = SentenceTransformer(model_name_or_path=model_name, device=DEVICE)
    embeddings = embedding_model.encode(dataset)

    embeddings = embeddings.astype(np.float32)

    index = faiss.IndexFlatL2(embedding_model.get_sentence_embedding_dimension())
    index.add(embeddings)

    return embeddings, index


def get_relevant_snippets(
    query: str,
    index: faiss.IndexFlatL2,
    dataset: list,
    model_name="all-MiniLM-L6-v2",
    top_k: int = 3,
) -> list:
    """
    Retrieve the top-k most relevant paragraphs for a given query.

    Args:
    - query (str): The user's question or query.
    - index (faiss.IndexFlatL2): The FAISS index for similarity search.
    - dataset (list): List of paragraphs from the dataset.
    - model_name (str): The SentenceTransformer model used for embeddings.
    - top_k (int): Number of paragraphs to retrieve.

    Returns:
    - list: A list of the most relevant paragraphs.
    """

    embedding_model = SentenceTransformer(model_name_or_path=model_name, device=DEVICE)

    query_embedding = embedding_model.encode([query]).astype(np.float32)

    distances, indices = index.search(query_embedding, top_k)
    relevant_snippets = []
    for i in indices[0]:
        if i < len(dataset):
            relevant_snippets.append(dataset[i])

    return relevant_snippets
