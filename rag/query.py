import argparse
import json
import sys
import requests
import chromadb
from sentence_transformers import SentenceTransformer

DEFAULT_CHROMA_SERVER = "http://localhost:8000"


def _connect_chroma(chroma_server=DEFAULT_CHROMA_SERVER, db_path="rag/chroma_db"):
    """Connect to ChromaDB server with automatic fallback to local storage."""
    if chroma_server:
        try:
            client = chromadb.HttpClient(host=chroma_server)
            client.heartbeat()
            print(f"Connected to ChromaDB server at {chroma_server}")
            return client
        except Exception:
            print(f"  ChromaDB server at {chroma_server} not reachable, falling back to local storage...")

    print(f"Using local ChromaDB at {db_path}")
    return chromadb.PersistentClient(path=db_path)


def retrieve(query, collection, model, top_k=3):
    """Retrieve the most relevant documents for a query."""
    query_embedding = model.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k,
    )
    return results["documents"][0] if results["documents"] else []


def generate(prompt, server_url="http://127.0.0.1:8080", max_tokens=512, temperature=0.3):
    """Send a prompt to the BitNet llama-server and return the response."""
    try:
        response = requests.post(
            f"{server_url}/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=120,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.ConnectionError:
        print(f"Error: Cannot connect to BitNet server at {server_url}")
        print("Start the server first:")
        print("  python run_inference_server.py -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf -n 512")
        sys.exit(1)
    except Exception as e:
        print(f"Error generating response: {e}")
        sys.exit(1)


def build_prompt(query, documents, system_prompt=None):
    """Build a RAG prompt with retrieved context."""
    if system_prompt is None:
        system_prompt = (
            "You are a helpful assistant. Answer the question based on the provided context. "
            "If the context doesn't contain enough information, say so. Do not make up facts."
        )

    context = "\n\n".join(f"[Document {i+1}]: {doc}" for i, doc in enumerate(documents))

    return f"""{system_prompt}

Context:
{context}

Question: {query}

Answer:"""


def interactive_mode(collection, model, server_url, top_k, max_tokens, temperature):
    """Run an interactive RAG Q&A loop."""
    print("BitNet RAG - Interactive Mode")
    print(f"Server: {server_url} | Top-K: {top_k} | Max tokens: {max_tokens}")
    print("Type 'quit' to exit.\n")

    while True:
        query = input("You: ").strip()
        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            break

        documents = retrieve(query, collection, model, top_k)

        if not documents:
            print("No relevant documents found.\n")
            continue

        print(f"  [Retrieved {len(documents)} documents]")

        prompt = build_prompt(query, documents)
        answer = generate(prompt, server_url, max_tokens, temperature)

        print(f"Assistant: {answer}\n")


def single_query(query, collection, model, server_url, top_k, max_tokens, temperature):
    """Answer a single question."""
    documents = retrieve(query, collection, model, top_k)

    if not documents:
        print("No relevant documents found.")
        return

    prompt = build_prompt(query, documents)
    answer = generate(prompt, server_url, max_tokens, temperature)

    print(f"\nRetrieved {len(documents)} documents:")
    for i, doc in enumerate(documents):
        preview = doc[:150] + "..." if len(doc) > 150 else doc
        print(f"  [{i+1}] {preview}")

    print(f"\nAnswer: {answer}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG query with BitNet")
    parser.add_argument("-q", "--query", type=str, help="Single question to ask (omit for interactive mode)")
    parser.add_argument("--server-url", type=str, default="http://127.0.0.1:8080", help="BitNet server URL")
    parser.add_argument("--collection", type=str, default="bitnet_rag", help="ChromaDB collection name")
    parser.add_argument("--db-path", type=str, default="rag/chroma_db", help="Local ChromaDB fallback path")
    parser.add_argument("--chroma-server", type=str, default=DEFAULT_CHROMA_SERVER,
                        help=f"ChromaDB server URL (default: {DEFAULT_CHROMA_SERVER}). "
                             "Falls back to local --db-path if unreachable.")
    parser.add_argument("--embedding-model", type=str, default="all-MiniLM-L6-v2", help="Sentence transformer model")
    parser.add_argument("--top-k", type=int, default=3, help="Number of documents to retrieve")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens for response")
    parser.add_argument("--temperature", type=float, default=0.3, help="Temperature for generation")
    args = parser.parse_args()

    print("Loading embedding model...")
    model = SentenceTransformer(args.embedding_model)

    print("Connecting to ChromaDB...")
    client = _connect_chroma(args.chroma_server, args.db_path)
    try:
        collection = client.get_collection(name=args.collection)
    except ValueError:
        print(f"Collection '{args.collection}' not found. Index your data first:")
        print(f"  python rag/index_data.py <your_data_file>")
        sys.exit(1)

    print(f"Collection '{args.collection}' loaded with {collection.count()} documents.\n")

    if args.query:
        single_query(args.query, collection, model, args.server_url, args.top_k, args.max_tokens, args.temperature)
    else:
        interactive_mode(collection, model, args.server_url, args.top_k, args.max_tokens, args.temperature)
