import requests


def generate_prompt(query: str, snippets: list[str]) -> str:
    """
    Create a structured prompt for the LLM using the query and relevant snippets.

    Args:
    - query (str): The user's question.
    - snippets (List[str]): The relevant paragraphs retrieved from the dataset.

    Returns:
    - str: A formatted prompt combining the context and the query.
    """

    context = "\n\n".join(snippets)

    # Some problems lead to having to create this prompt structure
    # 1. The model I am using would create follow-up questions by default from the information present in the context, I dont think there was an option in the API to disable this,
    # so to fix the problem I thought of having a "system" message to limit the model to only answer the question asked and not generate follow-up questions
    # 2. First this structure was a simple list with 2 dictionaries, but the model can only accept a string as input, so I had to format it as a string

    prompt = f"""You are an AI assistant. Use the provided context to answer the question.

Context:
{context}

Question: {query}

Instructions:
- Keep your response concise and relevant.
- ONLY use the information from the context to answer the question.
- If the answer cannot be found in the context, respond with: "Unable to find an answer in the provided context."

Answer:"""

    return prompt


def generate_answer(
    query: str,
    snippets: list[str],
    api_key: str,
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct",
) -> str:
    """
    Call a Hugging Face-hosted LLM to generate an answer based on the given prompt.

    Args:
    - query (str): The prompt containing context and the user's question.
    - snippets (List[str]): The relevant paragraphs retrieved from the dataset.
    - api_key (str): The Hugging Face API key for authentication.
    - model_name (str): The Hugging Face model to use.

    Returns:
    - str: The generated answer from the model.
    """

    API_URL = f"https://api-inference.huggingface.co/models/{model_name}"
    headers = {"Authorization": f"Bearer {api_key}"}

    prompt = generate_prompt(query, snippets)

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 150,
            "pad_token_id": 50256,
            "return_full_text": False,
        },
    }

    response = requests.post(API_URL, headers=headers, json=payload)

    if response.status_code == 200:
        return response.json()[0]["generated_text"].strip()
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return "An error occurred while generating the answer."


# if __name__ == "__main__":
#     query = "What is the capital of France?"
#     snippets = [
#         "Paris is the capital of France. It is known for its rich history and iconic landmarks.",
#         "The Eiffel Tower, built in 1889, is a global symbol of Paris and attracts millions of visitors every year.",
#     ]


#     answer = generate_answer(query, snippets)

#     print("\nFinal Answer:")
#     print(answer)
