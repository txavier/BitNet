import json
import csv
import os
import sys
import argparse
import chromadb
from sentence_transformers import SentenceTransformer


def load_documents(data_path):
    """Load documents from a JSON, JSONL, CSV, or plain text file."""
    documents = []
    ext = os.path.splitext(data_path)[1].lower()

    if ext == ".jsonl":
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line.strip())
                text = row.get("text") or row.get("content") or str(row)
                documents.append(text)

    elif ext == ".json":
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for row in data:
                    if isinstance(row, str):
                        documents.append(row)
                    else:
                        text = row.get("text") or row.get("content") or str(row)
                        documents.append(text)

    elif ext == ".csv":
        with open(data_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row.get("text") or row.get("content") or " ".join(row.values())
                documents.append(text)

    elif ext == ".txt":
        with open(data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    documents.append(line)

    else:
        print(f"Unsupported file format: {ext}")
        sys.exit(1)

    return documents


def index_documents(data_path, collection_name="bitnet_rag", db_path="rag/chroma_db",
                    embedding_model="all-MiniLM-L6-v2", batch_size=100):
    """Load documents, compute embeddings, and store in ChromaDB."""
    print(f"Loading documents from {data_path}...")
    documents = load_documents(data_path)
    print(f"Loaded {len(documents)} documents.")

    if not documents:
        print("No documents found.")
        sys.exit(1)

    print(f"Loading embedding model: {embedding_model}...")
    model = SentenceTransformer(embedding_model)

    print("Initializing ChromaDB...")
    client = chromadb.PersistentClient(path=db_path)

    # Delete existing collection if it exists
    try:
        client.delete_collection(name=collection_name)
    except (ValueError, chromadb.errors.NotFoundError):
        pass

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    print(f"Indexing {len(documents)} documents in batches of {batch_size}...")
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        ids = [f"doc_{j}" for j in range(i, i + len(batch))]
        embeddings = model.encode(batch).tolist()
        collection.add(
            ids=ids,
            documents=batch,
            embeddings=embeddings,
        )
        print(f"  Indexed {min(i + batch_size, len(documents))}/{len(documents)}")

    print(f"Done. {len(documents)} documents indexed in {db_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index documents for RAG")
    parser.add_argument("data_path", type=str, help="Path to data file (JSON, JSONL, CSV, or TXT)")
    parser.add_argument("--collection", type=str, default="bitnet_rag", help="ChromaDB collection name")
    parser.add_argument("--db-path", type=str, default="rag/chroma_db", help="Path to store ChromaDB")
    parser.add_argument("--embedding-model", type=str, default="all-MiniLM-L6-v2", help="Sentence transformer model")
    parser.add_argument("--batch-size", type=int, default=100, help="Indexing batch size")
    args = parser.parse_args()

    index_documents(args.data_path, args.collection, args.db_path, args.embedding_model, args.batch_size)
