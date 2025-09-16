import mlflow
from rag_demo import qa_chain  # Import from your notebook logic

def evaluate_rag(queries):
    with mlflow.start_run():
        scores = []
        for q in queries:
            result = qa_chain({"query": q})
            score = len(result["result"])  # Placeholder; use real eval like ROUGE
            scores.append(score)
        mlflow.log_metric("avg_score", sum(scores)/len(scores))

if __name__ == "__main__":
    queries = ["What is RAG?", "Explain LLMs."]
    evaluate_rag(queries)