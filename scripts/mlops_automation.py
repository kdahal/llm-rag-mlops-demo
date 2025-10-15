# scripts/mlops_automation.py
import os
import platform
import mlflow
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.chains import RetrievalQA

# Create sample data for CI
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)
sample_path = os.path.join(data_dir, "sample_doc.txt")
with open(sample_path, "w") as f:
    f.write("This document discusses artificial intelligence and its applications in modern technology. AI has revolutionized fields like healthcare, finance, and entertainment.")

# Load and process
loader = TextLoader(sample_path)
documents = loader.load()
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(texts, embeddings)

model_id = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
pipe = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=50,
    temperature=0.5,
    do_sample=True,
    device="cpu"
)
llm = HuggingFacePipeline(pipeline=pipe)

qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())

# Run query
query = "What is discussed in the document?"
result = qa_chain.invoke({"query": query})

# Log to MLflow (with loader_fn for RetrievalQA)
with mlflow.start_run():
    mlflow.log_param("query", query)
    mlflow.log_param("model", model_id)
    mlflow.log_metric("response_length", len(result["result"]))
    
    # Save FAISS to artifact root
    artifacts_root = mlflow.active_run().info.artifact_uri.replace("file://", "")
    if platform.system() == "Windows":
        artifacts_root = artifacts_root.lstrip('/')
    faiss_path = os.path.join(artifacts_root, "faiss_index")
    os.makedirs(os.path.dirname(faiss_path), exist_ok=True)
    vectorstore.save_local(faiss_path)
    
    # Define loader_fn to reconstruct the chain
    def loader_fn(model_config):
        # Recreate embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Recreate LLM pipeline
        model_id = "google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=50,
            temperature=0.5,
            do_sample=True,
            device="cpu"
        )
        llm = HuggingFacePipeline(pipeline=pipe)
        
        # Load vectorstore from artifacts
        artifact_uri = model_config["artifact_uri"].replace("file://", "")
        if platform.system() == "Windows":
            artifact_uri = artifact_uri.lstrip('/')
        faiss_path = os.path.join(artifact_uri, "faiss_index")
        vectorstore = FAISS.load_local(
            faiss_path, 
            embeddings, 
            allow_dangerous_deserialization=True
        )
        
        # Rebuild and return the chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )
        return qa_chain

    # Log the model with loader_fn
    mlflow.langchain.log_model(qa_chain, name="rag_model", loader_fn=loader_fn)
    
    print(f"CI RAG Test Passed: Response: {result['result']}")

print("MLOps Automation Complete!")