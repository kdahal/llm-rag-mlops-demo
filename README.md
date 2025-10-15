# LLM RAG MLOps Demo

[![Tests](https://github.com/kdahal/llm-rag-mlops-demo/actions/workflows/ci.yml/badge.svg)](https://github.com/kdahal/llm-rag-mlops-demo/actions/workflows/ci.yml) [![Python](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/release/python-313/)

A demo repository showcasing a Retrieval-Augmented Generation (RAG) pipeline for Large Language Models (LLMs) with MLOps integration. This project uses open-source tools like LangChain for orchestration, FAISS for vector search, Hugging Face for embeddings and local LLM inference, and MLflow for experiment tracking and model logging. It's designed for reproducibility, testing, and CI/CD automation.

## Features
- **Document Processing**: Load and split text documents using LangChain.
- **Embeddings & Vector Store**: Generate embeddings with Sentence Transformers and store/index with FAISS.
- **RAG Pipeline**: Retrieve relevant chunks and generate responses with a local Flan-T5 LLM.
- **MLOps Integration**: Log parameters, metrics, and models to MLflow for tracking and reproducibility.
- **Automated Testing**: Pytest suite covering loading, embeddings, querying, and MLflow.
- **CI/CD**: GitHub Actions for end-to-end testing and automation on pushes/PRs.
- **Local Inference**: Offline LLM execution (CPU/GPU support).

## Architecture Overview

The following Mermaid diagram illustrates the high-level RAG pipeline with MLOps flow:

```mermaid
graph TD
    A(Document Input<br/>(e.g., sample_doc.txt)) --> B(Load & Split<br/>(LangChain TextLoader + CharacterTextSplitter))
    B --> C(Generate Embeddings<br/>(HuggingFaceEmbeddings: all-MiniLM-L6-v2))
    C --> D(Vector Store Indexing<br/>(FAISS.from_documents))
    E(User Query) --> F(Retrieve Relevant Chunks<br/>(vectorstore.as_retriever))
    D --> F
    F --> G(Generate Response<br/>(RetrievalQA + HuggingFacePipeline: Flan-T5-base))
    G --> H[Output Response]
    G --> I[Log to MLflow<br/>(params, metrics, model w/ loader_fn)]
    I --> J[Artifacts: FAISS Index + Serialized Chain]
    style A fill:#e1f5fe
    style E fill:#f3e5f5
    style H fill:#e8f5e8
    style J fill:#fff3e0
```

## Prerequisites
- Python 3.13+ (recommended; tested with 3.13.3)
- Git (for cloning)
- Optional: NVIDIA GPU for faster inference (install `faiss-gpu` instead of `faiss-cpu`)

## Installation

1. **Clone the Repository**:
   ```
   git clone https://github.com/kdahal/llm-rag-mlops-demo.git
   cd llm-rag-mlops-demo
   ```

2. **Install Dependencies**:
   ```
   pip install -r requirements.txt
   ```
   - This includes LangChain, Hugging Face libs, FAISS, MLflow, Transformers, and pytest.
   - For GPU: Replace `faiss-cpu` with `faiss-gpu` in `requirements.txt` and reinstall.

3. **Prepare Sample Data** (optional for quick start):
   Create `data/sample_doc.txt`:
   ```
   This document discusses artificial intelligence and its applications in modern technology. AI has revolutionized fields like healthcare, finance, and entertainment.
   ```

## Usage

### Running the Demo Script
For a standalone run (no Jupyter needed):
```
python scripts/mlops_automation.py
```
- Builds the pipeline, runs a sample query ("What is discussed in the document?"), logs to MLflow.
- Expected output: "CI RAG Test Passed: Response: artificial intelligence..." + run details.

### Jupyter Notebook Demo
1. Launch Jupyter:
   ```
   jupyter notebook
   ```

2. Create a new notebook and run this cell (full pipeline + MLflow logging):

   ```python
   import os
   import platform
   import logging
   os.environ["TRANSFORMERS_VERBOSITY"] = "error"  # Suppress warnings
   logging.getLogger("mlflow").setLevel(logging.ERROR)
   from dotenv import load_dotenv
   from langchain_community.document_loaders import TextLoader
   from langchain.text_splitter import CharacterTextSplitter
   from langchain_huggingface import HuggingFaceEmbeddings
   from langchain_community.vectorstores import FAISS
   from langchain_huggingface import HuggingFacePipeline
   from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
   from langchain.chains import RetrievalQA
   import mlflow
   import mlflow.langchain

   load_dotenv()

   data_path = "data/sample_doc.txt"
   os.makedirs(os.path.dirname(data_path), exist_ok=True)
   with open(data_path, "w") as f:  # Create sample if missing
       f.write("This document discusses artificial intelligence and its applications in modern technology. AI has revolutionized fields like healthcare, finance, and entertainment.")

   # Load and split
   loader = TextLoader(data_path)
   documents = loader.load()
   text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
   texts = text_splitter.split_documents(documents)

   # Embeddings and vector store
   embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
   vectorstore = FAISS.from_documents(texts, embeddings)

   # Local LLM
   model_id = "google/flan-t5-base"
   tokenizer = AutoTokenizer.from_pretrained(model_id)
   model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
   pipe = pipeline(
       "text2text-generation",
       model=model,
       tokenizer=tokenizer,
       max_new_tokens=512,
       temperature=0.5,
       do_sample=True,
       device="cpu"  # "cuda" for GPU
   )
   llm = HuggingFacePipeline(pipeline=pipe)

   # RAG chain
   qa_chain = RetrievalQA.from_chain_type(
       llm=llm,
       chain_type="stuff",
       retriever=vectorstore.as_retriever()
   )

   # Query
   query = "What is discussed in the document?"
   result = qa_chain.invoke({"query": query})

   # Log to MLflow
   with mlflow.start_run():
       mlflow.log_param("query", query)
       mlflow.log_param("model", model_id)
       mlflow.log_metric("response_length", len(result["result"]))
       
       # Save FAISS
       artifacts_root = mlflow.active_run().info.artifact_uri.replace("file://", "")
       if platform.system() == "Windows":
           artifacts_root = artifacts_root.lstrip('/')
       faiss_path = os.path.join(artifacts_root, "faiss_index")
       os.makedirs(os.path.dirname(faiss_path), exist_ok=True)
       vectorstore.save_local(faiss_path)
       
       # Loader function for serialization
       def loader_fn(model_config):
           embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
           model_id = "google/flan-t5-base"
           tokenizer = AutoTokenizer.from_pretrained(model_id)
           model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
           pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512, temperature=0.5, do_sample=True, device="cpu")
           llm = HuggingFacePipeline(pipeline=pipe)
           artifact_uri = model_config["artifact_uri"].replace("file://", "")
           if platform.system() == "Windows":
               artifact_uri = artifact_uri.lstrip('/')
           faiss_path = os.path.join(artifact_uri, "faiss_index")
           vectorstore = FAISS.load_local(faiss_path, embeddings, allow_dangerous_deserialization=True)
           return RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())

       mlflow.langchain.log_model(qa_chain, name="rag_model", loader_fn=loader_fn)
       
       run_id = mlflow.active_run().info.run_id
       print(f"Response: {result['result']}")
       print(f"MLflow Run ID: {run_id} (view: http://localhost:5000/#/experiments/0/runs/{run_id})")
   ```

3. **View MLflow UI** (in new terminal):
   ```
   mlflow ui
   ```
   - Open [http://localhost:5000](http://localhost:5000) to inspect runs, params, metrics, and artifacts (e.g., `rag_model` with FAISS index).

### Model Alternatives
- **Default**: `google/flan-t5-base` (lightweight, ~250MB).
- **Advanced Local**: `google/flan-t5-large` (better quality, ~1GB).
- **GPU-Optimized**: Set `device="cuda"` in pipeline; use larger models like `meta-llama/Llama-2-7b-chat-hf` (requires HF token).

### Reloading Logged Model
Test reproducibility:
```python
import mlflow
loaded_model = mlflow.langchain.load_model("runs:/[your_run_id]/rag_model")  # Replace [your_run_id]
print(loaded_model.invoke({"query": "What fields does AI revolutionize?"}))
```
- Expected: "healthcare, finance, and entertainment."

## Testing
Automated tests ensure pipeline integrity:
```
pytest tests/ -v
```
- **Coverage**: Document loading/splitting, embeddings/FAISS, RAG querying, MLflow logging.
- All pass in ~23s on CPU; zero warnings.

## Troubleshooting
- **Model Download Slow**: First run caches (~250MB); use `--trusted-host huggingface.co` if network issues.
- **FAISS Path Errors (Windows)**: Handled in code; if persists, check MLflow tracking URI (`mlflow.set_tracking_uri(".")`).
- **Deprecations**: Suppressed in code; update LangChain for long-term.
- **GPU Issues**: Ensure CUDA 12+ and `torch` with CUDA support.
- **CI Failures**: Check Actions tab; common: Cache miss or dep conflicts (rerun).

## Contributing
1. Fork the repo.
2. Create a feature branch (`git checkout -b feature/your-feature`).
3. Commit changes (`git commit -m "Add your feature"`).
4. Push (`git push origin feature/your-feature`).
5. Open a Pull Request.

## License
MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments
- [LangChain](https://langchain.com/) for RAG orchestration.
- [FAISS](https://github.com/facebookresearch/faiss) for vector search.
- [Hugging Face Transformers](https://huggingface.co/) for models/embeddings.
- [MLflow](https://mlflow.org/) for MLOps.
- Inspired by RAG tutorials on Medium and official docs (as of October 2025).