# llm-rag-mlops-demo
A demo repository for RAG implementations in LLMs with MLOps automation scripts using open-source tools.


Explanation of Changes

Fixed Imports:

Added python-dotenv installation to resolve ModuleNotFoundError.
Ensured TextLoader and other imports from langchain_community are correctly referenced.


Environment Setup:

Added a check for the Hugging Face API token to prevent runtime errors.
Used os.getenv to safely load the token.


File Handling:

Added error handling for TextLoader to catch missing file issues.
You must create data/sample_doc.txt (e.g., in your project folder) with some text for testing.


LLM Upgrade:

Replaced gpt2 with google/flan-t5-base, which is more suitable for Q&A tasks and freely accessible via Hugging Face’s API. If you have access to meta-llama/Llama-2-7b-chat-hf, you can use it instead (update the repo_id).


MLflow Logging:

Moved the MLflow block after qa_chain creation to avoid referencing an undefined variable.
Ensured query is defined before use.


Error Prevention:

Added try-except blocks and validation to make the code robust.
Ensured all steps are executed in order (load, split, embed, query, log).



How to Run

Install Dependencies:

Run the pip command above in your Command Prompt or Jupyter cell:
bashpip install python-dotenv langchain langchain_community faiss-cpu huggingface_hub sentence-transformers mlflow



Set Up .env File:

Create a file named .env in your project directory with:
textHUGGINGFACEHUB_API_TOKEN=your_token_here

Replace your_token_here with your Hugging Face API token.


Create Sample Data:

Create a folder named data in your project directory.
Add a file sample_doc.txt with some text, e.g., "This document discusses artificial intelligence and its applications."


Run in Jupyter Notebook:

Copy the corrected code into a Jupyter cell.
Ensure you’re in the correct directory (use os.chdir("your_project_path") if needed).
Execute the cell. The notebook should load the document, create a vector store, run the query, log results to MLflow, and print the answer.



Notes

Performance: google/flan-t5-base is lightweight but may give basic answers. For better results, consider meta-llama/Llama-2-7b-chat-hf (requires Pro access) or a local model via LlamaCpp if you have a powerful machine.
MLflow: Ensure MLflow is set up (e.g., run mlflow ui in a terminal to view logs at http://localhost:5000).
FAISS: Uses faiss-cpu. For GPU support, install faiss-gpu instead.
Troubleshooting:

If you get HUGGINGFACEHUB_API_TOKEN errors, verify your .env file and token.
If sample_doc.txt is missing, create it or adjust the path.
For other errors, check package versions or share the error for further debugging.
