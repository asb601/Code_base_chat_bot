import os
from weaviate import Client
from weaviate.auth import AuthApiKey
from sentence_transformers import SentenceTransformer
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

class VectorDBManager:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(model_name)
        self.client = Client(
            url=os.getenv("WEAVIATE_URL"),
            auth_client_secret=AuthApiKey(os.getenv("WEAVIATE_API_KEY"))
        )
        self._init_schema()

    def _init_schema(self):
        schema = {
            "classes": [{
                "class": "CodeChunk",
                "properties": [
                    {"name": "content", "dataType": ["text"]},
                    {"name": "language", "dataType": ["text"]},
                    {"name": "file_path", "dataType": ["text"]},
                    {"name": "start_line", "dataType": ["int"]},
                    {"name": "end_line", "dataType": ["int"]},
                    {"name": "chunk_type", "dataType": ["text"]},
                    {"name": "timestamp", "dataType": ["date"]},
                ],
                "vectorizer": "none"
            }]
        }
        
        # Safely delete existing class
        if self.client.schema.get("CodeChunk"):
            self.client.schema.delete_class("CodeChunk")
        
        self.client.schema.create(schema)

    def store_chunks(self, chunks):
        """Process and store chunks with local embeddings"""
        texts = [chunk['content'] for chunk in chunks]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        with self.client.batch as batch:
            for chunk, vector in zip(chunks, embeddings):
                batch.add_data_object(
                    data_object={
                        "content": chunk['content'],
                        "language": chunk['language'],
                        "file_path": chunk['file'],
                        "start_line": chunk['start_line'],
                        "end_line": chunk['end_line'],
                        "chunk_type": chunk['type'],
                        "timestamp": datetime.now().isoformat()
                    },
                    class_name="CodeChunk",
                    vector=vector.tolist()
                )

if __name__ == "__main__":
    import json
    
    with open('code_metadata.json') as f:
        data = json.load(f)
    
    db = VectorDBManager()
    db.store_chunks(data['chunks'])
    
    # Verify insertion
    result = db.client.query.get(
        "CodeChunk",
        ["content", "file_path"]
    ).with_limit(2).do()
    print("First 2 entries:", result)