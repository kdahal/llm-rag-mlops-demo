if __name__ == "__main__":

import mlflow
loaded_model = mlflow.langchain.load_model("runs:/bde585204dc34280af8dc55ddd2dc74e/rag_model")  # Your run_id
print(loaded_model.invoke({"query": "What fields does AI revolutionize?"}))
