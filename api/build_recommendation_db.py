"""
Build the local recommendation ChromaDB from the Kaggle CSV.
Same logic as the notebook — just runs locally.

Usage:
    python build_recommendation_db.py
"""
import os
import pandas as pd
import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer
import kagglehub

# 1. Load CSV
dataset_path = kagglehub.dataset_download('sumitm004/arxiv-scientific-research-papers-dataset')
csv_path = os.path.join(dataset_path, 'arXiv_scientific_dataset.csv')
print(f"Loading CSV from: {csv_path}")
df = pd.read_csv(csv_path)
print(f"Total papers: {len(df)}")

# 2. Prepare data (same as notebook)
df.drop(columns=['summary_word_count', 'published_date', 'updated_date', 'first_author'], inplace=True)
df["tags"] = (df["title"] + " " + df["summary"] + " " + df["authors"]).str.lower()

# Sample 100k papers
df_small = df.sample(n=100000, random_state=42).reset_index(drop=True)

# Create arxiv_id by stripping 'abs-' prefix
df_small["arxiv_id"] = df_small["id"].astype(str).str.replace(r"^abs-", "", regex=True)

print(f"Sampled {len(df_small)} papers. Encoding...")

# 3. Encode with SentenceTransformer
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
vectors = model.encode(df_small["tags"].values.tolist(), show_progress_bar=True)
print(f"Encoded {len(vectors)} vectors of dimension {vectors.shape[1]}")

# 4. Store in local ChromaDB
db_path = os.path.join(os.path.dirname(__file__), "recommendation_db")
os.makedirs(db_path, exist_ok=True)

client = chromadb.PersistentClient(path=db_path)

# Delete old collection if it exists
try:
    client.delete_collection("research_papers_v3")
except Exception:
    pass

collection = client.get_or_create_collection(name="research_papers_v3")

# 5. Add in batches (same as notebook)
batch_size = 5000
for start in range(0, len(df_small), batch_size):
    end = min(start + batch_size, len(df_small))
    collection.add(
        ids=[str(i) for i in range(start, end)],
        embeddings=vectors[start:end].tolist(),
        documents=df_small["tags"].iloc[start:end].tolist(),
        metadatas=[
            {
                "title": row["title"],
                "category_code": row["category_code"],
                "arvix_id": row["arxiv_id"]   # same typo as notebook
            }
            for _, row in df_small.iloc[start:end].iterrows()
        ]
    )
    print(f"Added {start} to {end}")

print(f"\nDone! ChromaDB saved to: {db_path}")
print(f"   Collection: research_papers_v3 ({collection.count()} documents)")

