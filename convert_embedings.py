import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

class LocalEmbedder:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(model_name)
    
    def embed_chunks(self, chunks):
        texts = [chunk['content'] for chunk in chunks]
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        return embeddings
    
    def save_embeddings(self, embeddings, metadata, output_dir="output"):
        os.makedirs(output_dir, exist_ok=True)
        np.save(os.path.join(output_dir, "embeddings.npy"), embeddings)
        with open(os.path.join(output_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"✅ Saved embeddings to {output_dir}/embeddings.npy")
        print(f"✅ Saved metadata to {output_dir}/metadata.json")

    def run(self, input_json):
        with open(input_json) as f:
            data = json.load(f)
        chunks = data["chunks"]
        embeddings = self.embed_chunks(chunks)

        metadata = []
        for i, chunk in enumerate(chunks):
            metadata.append({
                "index": i,
                "file_path": chunk.get('file') or chunk.get('file_path') or "unknown",
                "language": chunk.get('language') or "unknown",
                "start_line": chunk.get('start_line') or 0,
                "end_line": chunk.get('end_line') or 0,
                "chunk_type": chunk.get('type') or chunk.get('chunk_type') or "code"
            })

        self.save_embeddings(embeddings, metadata)

if __name__ == "__main__":
    embedder = LocalEmbedder()
    embedder.run("data/code_metadata.json")
