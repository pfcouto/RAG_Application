# **RAG Application: Minimal Dependency Example**

This project demonstrates a lightweight implementation of a Retrieval-Augmented Generation (RAG) application. The goal is to achieve effective results with minimal dependencies and concise code.

### **Overview**

- **Embedding Model**: Uses SBERT (e.g., `all-MiniLM-L6-v2`) for generating vector representations of text.
- **Language Model**: Calls a Large Language Model (LLM) from Hugging Face's API (e.g., `Llama-3.2-1B-Instruct`) to generate answers.
- **Vector Indexing**: Leverages Meta's Faiss library for efficient vector similarity search.

The example dataset consists of 18 paragraphs. Based on the user's question, 1-3 relevant paragraphs are retrieved to provide context to the LLM for generating a response. You can modify the dataset and prompts to experiment with different embeddings and results.



## **Instructions**

Create `.env` and `.env.docker` files.

- Use `.env.example` and `.env.docker.example` as templates.
- Ensure you add your **Hugging Face API Key**.

### **Run Locally**

1. Install required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Run the main application:
   ```bash
   python app/main.py
   ```

### **Run using Docker**

1. Use the `start.sh` script to create and run the container in Docker:
   ```bash
   ./start.sh
   ```



## Testing the Application

Once the system is running (locally or in docker), you can test the solution by sending a POST request.

### Example Request

- Run the following command in a terminal:
  ```bash
  curl -X POST -H "Content-Type: application/json" -d '{"question": "What is the capital of France?"}' http://127.0.0.1:8000/ask
  ```

### Example Response

- The application will return a JSON response like this:

```json
{
  "answer": "The capital of France is Paris.",
  "relevant snippets": [
    "As the capital of France, Paris is the epicenter of French language and culture.",
    "Paris has been a cradle of artistic and intellectual movements, inspiring figures like Victor Hugo, Voltaire, and Edith Piaf.",
    "Paris is also known for its culinary delights, offering everything from croissants and baguettes to haute cuisine."
  ]
}
```



## **Deploying to AWS**

To deploy this container to AWS I would use Amazon Elastic Container Service (ECS). I would start by building and pushing the Docker image to Amazon Elastic Container Registry (ECR). Then, create an ECS cluster and define a task using the pushed image. After that, I would configure the necessary security groups, set environment variables (like the Hugging Face API key), and expose the service.
