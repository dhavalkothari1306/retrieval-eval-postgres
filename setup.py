"""
setup.py — run once to:
  1. Create the documents table with a tsvector column for keyword search
  2. Create the embeddings table with a pgvector column for semantic search
  3. Insert all 40 documents
  4. Generate embeddings using sentence-transformers (runs on CPU, no API key needed)
  5. Index both columns for fast retrieval
"""

import json
import psycopg2
from sentence_transformers import SentenceTransformer

DB = dict(host="localhost", port=5432, dbname="searchdb",
          user="postgres", password="postgres")

# Load documents
with open("documents.json") as f:
    documents = json.load(f)

conn = psycopg2.connect(**DB)
cur = conn.cursor()

# Enable pgvector extension
cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

# Drop and recreate tables cleanly
cur.execute("DROP TABLE IF EXISTS documents;")
cur.execute("""
    CREATE TABLE documents (
        id        INTEGER PRIMARY KEY,
        text      TEXT NOT NULL,
        tsv       TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
        embedding VECTOR(384)
    );
""")

# GIN index for keyword search
cur.execute("CREATE INDEX IF NOT EXISTS idx_tsv ON documents USING GIN(tsv);")

conn.commit()
print("Tables created.")

# Load embedding model (downloads ~90MB on first run)
print("Loading embedding model (may take a moment on first run)...")
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Insert documents and embeddings
print("Generating embeddings and inserting documents...")
texts = [doc["text"] for doc in documents]
embeddings = model.encode(texts, show_progress_bar=True)

for doc, embedding in zip(documents, embeddings):
    cur.execute(
        "INSERT INTO documents (id, text, embedding) VALUES (%s, %s, %s)",
        (doc["id"], doc["text"], embedding.tolist())
    )

conn.commit()

# HNSW index for fast vector search
cur.execute("""
    CREATE INDEX IF NOT EXISTS idx_embedding
    ON documents USING hnsw (embedding vector_cosine_ops);
""")
conn.commit()

cur.close()
conn.close()
print(f"Done. {len(documents)} documents inserted and indexed.")
