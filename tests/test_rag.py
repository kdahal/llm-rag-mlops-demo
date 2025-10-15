import pytest
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader  # Fixed import
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings  # Updated
from langchain_community.vectorstores import FAISS  # Fixed import
from langchain_huggingface import HuggingFacePipeline  # Fixed import
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain.chains import RetrievalQA
import mlflow

load_dotenv()

@pytest.fixture
def sample_data():
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    with open(os.path.join(data_dir, "sample_doc.txt"), "w") as f:
        f.write("This document discusses artificial intelligence and its applications in modern technology. AI has revolutionized fields like healthcare, finance, and entertainment.")
    return os.path.join(data_dir, "sample_doc.txt")

def test_document_loading_and_splitting(sample_data):
    loader = TextLoader(sample_data)
    documents = loader.load()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = splitter.split_documents(documents)
    assert len(texts) > 0
    assert "artificial intelligence" in texts[0].page_content

def test_embeddings_and_vectorstore(sample_data):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    loader = TextLoader(sample_data)
    documents = loader.load()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(texts, embeddings)
    assert len(vectorstore.index_to_docstore_id) > 0

def test_rag_query(sample_data):
    # Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    loader = TextLoader(sample_data)
    documents = loader.load()
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(texts, embeddings)

    # LLM (lightweight test; skip full pipeline if slow)
    model_id = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
    pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer, max_new_tokens=50, temperature=0.5, device="cpu")
    llm = HuggingFacePipeline(pipeline=pipe)

    qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=vectorstore.as_retriever())
    result = qa_chain.invoke({"query": "What is discussed in the document?"})
    assert "AI" in result["result"] or "artificial intelligence" in result["result"].lower()

def test_mlflow_logging(tmp_path):
    # Mock run; test logging doesn't crash
    with mlflow.start_run():
        mlflow.log_param("test_param", "value")
        assert mlflow.active_run() is not None