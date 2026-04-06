import json
import csv
import os
import sys
import shutil
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import chromadb
from sentence_transformers import SentenceTransformer

DEFAULT_CHROMA_SERVER = "http://localhost:8000"


def _connect_chroma(chroma_server=DEFAULT_CHROMA_SERVER, db_path="rag/chroma_db"):
    """Connect to ChromaDB server with automatic fallback to local storage.

    Tries the server first. If unreachable, falls back to PersistentClient.
    """
    if chroma_server:
        try:
            client = chromadb.HttpClient(host=chroma_server)
            client.heartbeat()  # verify connection
            print(f"Connected to ChromaDB server at {chroma_server}")
            return client, chroma_server
        except Exception:
            print(f"  ChromaDB server at {chroma_server} not reachable, falling back to local storage...")

    print(f"Using local ChromaDB at {db_path}")
    return chromadb.PersistentClient(path=db_path), db_path


def load_documents(data_path, workers=1):
    """Load documents from a file or directory of files."""
    if os.path.isdir(data_path):
        return load_directory(data_path, workers=workers)

    documents = []
    ext = os.path.splitext(data_path)[1].lower()

    if ext == ".pdf":
        documents.extend(load_pdf(data_path))

    elif ext == ".jsonl":
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

    elif ext in (".txt", ".md"):
        documents.extend(load_text_file(data_path))

    else:
        print(f"Unsupported file format: {ext}")
        sys.exit(1)

    return documents


def load_pdf(file_path):
    """Load a PDF file, extracting text with OCR fallback."""
    try:
        from rag.pdf_handler import pdf_to_chunks
    except ImportError:
        try:
            from pdf_handler import pdf_to_chunks
        except ImportError:
            print(f"  Error: pdf_handler not available. Install: pip install pymupdf pytesseract pdf2image")
            print(f"  Skipping {file_path}")
            return []
    return pdf_to_chunks(file_path)


def load_text_file(file_path):
    """Load a text or markdown file, splitting into sections."""
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()

    ext = os.path.splitext(file_path)[1].lower()
    filename = os.path.basename(file_path)

    if ext == ".md":
        # Split markdown by headings to create meaningful chunks
        sections = []
        current_section = []
        current_heading = filename

        for line in content.split("\n"):
            if line.startswith("#"):
                # Save previous section if it has content
                text = "\n".join(current_section).strip()
                if text:
                    sections.append(f"[{filename}] {current_heading}\n{text}")
                current_heading = line.lstrip("#").strip()
                current_section = []
            else:
                current_section.append(line)

        # Save last section
        text = "\n".join(current_section).strip()
        if text:
            sections.append(f"[{filename}] {current_heading}\n{text}")

        return [s for s in sections if len(s.strip()) > 20]
    else:
        return [line.strip() for line in content.split("\n") if line.strip()]


def _files_identical(path_a, path_b):
    """Check if two files have identical content using SHA-256."""
    import hashlib

    def _sha256(path):
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    return _sha256(path_a) == _sha256(path_b)


def load_directory(dir_path, workers=1):
    """Load all supported files from a directory, jsonify them, and move originals.

    Safe to re-run:
    - Files only in jsonified/original/ (no top-level copy): loaded from cached JSON.
    - Files at top level that match jsonified/original/ exactly: skipped (unchanged).
    - Files at top level that differ from jsonified/original/: re-processed (updated).
    - New files not in jsonified/original/: processed, jsonified, and original moved.

    Args:
        dir_path: Path to directory containing files to process.
        workers: Number of parallel workers for file loading/OCR (default: 1).
    """
    supported_extensions = {".json", ".jsonl", ".csv", ".txt", ".md", ".pdf"}
    documents = []

    # jsonified/ lives alongside the data dir, not inside it
    parent_dir = os.path.dirname(os.path.abspath(dir_path))
    jsonified_dir = os.path.join(parent_dir, "jsonified")
    original_dir = os.path.join(jsonified_dir, "original")
    os.makedirs(jsonified_dir, exist_ok=True)
    os.makedirs(original_dir, exist_ok=True)

    # Collect already-processed filenames
    already_processed = set(os.listdir(original_dir)) if os.path.isdir(original_dir) else set()

    # Build set of top-level supported files
    top_level_files = set()
    for filename in os.listdir(dir_path):
        if filename == "jsonified":
            continue
        ext = os.path.splitext(filename)[1].lower()
        if ext not in supported_extensions:
            continue
        file_path = os.path.join(dir_path, filename)
        if os.path.isfile(file_path):
            top_level_files.add(filename)

    all_filenames = sorted(top_level_files | already_processed)

    cached = 0
    unchanged = 0
    updated = 0
    new = 0

    # --- Phase 1: Handle cached & unchanged (fast, serial) ---
    files_to_process = []  # (filename, top_level_path, json_path, original_path, label)

    for filename in all_filenames:
        ext = os.path.splitext(filename)[1].lower()
        if ext not in supported_extensions:
            continue

        json_filename = os.path.splitext(filename)[0] + ".json"
        json_path = os.path.join(jsonified_dir, json_filename)
        original_path = os.path.join(original_dir, filename)
        top_level_path = os.path.join(dir_path, filename)

        # Case 1: Only in jsonified/original, no top-level copy — load from cache
        if filename in already_processed and filename not in top_level_files:
            if os.path.isfile(json_path):
                with open(json_path, "r", encoding="utf-8") as f:
                    json_data = json.load(f)
                documents.extend(item.get("text", "") for item in json_data)
                cached += 1
                print(f"  Cached: {filename}")
            continue

        # Case 2: Top-level file exists and was previously processed
        if filename in already_processed and filename in top_level_files:
            if _files_identical(top_level_path, original_path):
                # Unchanged — load from cache and remove the duplicate top-level file
                if os.path.isfile(json_path):
                    with open(json_path, "r", encoding="utf-8") as f:
                        json_data = json.load(f)
                    documents.extend(item.get("text", "") for item in json_data)
                os.remove(top_level_path)
                unchanged += 1
                print(f"  Unchanged: {filename} (removed duplicate)")
                continue
            else:
                files_to_process.append((filename, top_level_path, json_path, original_path, "updated"))
                continue

        # Case 3: New file, not previously processed
        files_to_process.append((filename, top_level_path, json_path, original_path, "new"))

    # --- Phase 2: Process new/updated files (parallel when workers > 1) ---
    def _process_one(item):
        """Load documents from a single file (thread-safe)."""
        fn, src_path, _, _, _ = item
        docs = load_documents(src_path)
        return item, docs

    effective_workers = min(workers, len(files_to_process)) if files_to_process else 1

    if effective_workers > 1:
        print(f"  Processing {len(files_to_process)} files with {effective_workers} workers...")
        with ThreadPoolExecutor(max_workers=effective_workers) as pool:
            futures = {pool.submit(_process_one, item): item for item in files_to_process}
            for future in as_completed(futures):
                item, docs = future.result()
                filename, top_level_path, json_path, original_path, label = item
                json_filename = os.path.splitext(filename)[0] + ".json"
                documents.extend(docs)
                json_data = [{"source": filename, "index": i, "text": doc} for i, doc in enumerate(docs)]
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(json_data, f, indent=2, ensure_ascii=False)
                shutil.move(top_level_path, original_path)
                if label == "updated":
                    updated += 1
                    print(f"  Updated: {filename} ({len(docs)} chunks)")
                else:
                    new += 1
                    print(f"  New: {filename} -> jsonified/{json_filename} ({len(docs)} chunks)")
    else:
        for item in files_to_process:
            filename, top_level_path, json_path, original_path, label = item
            json_filename = os.path.splitext(filename)[0] + ".json"
            if label == "updated":
                print(f"  Updated: {filename} (content changed, re-processing)")
            else:
                print(f"  New: {filename}")
            docs = load_documents(top_level_path)
            documents.extend(docs)
            json_data = [{"source": filename, "index": i, "text": doc} for i, doc in enumerate(docs)]
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            shutil.move(top_level_path, original_path)
            if label == "updated":
                updated += 1
                print(f"    -> Updated jsonified/{json_filename} ({len(docs)} chunks)")
            else:
                new += 1
                print(f"    -> Saved jsonified/{json_filename} ({len(docs)} chunks)")

    summary = []
    if new:
        summary.append(f"{new} new")
    if updated:
        summary.append(f"{updated} updated")
    if unchanged:
        summary.append(f"{unchanged} unchanged")
    if cached:
        summary.append(f"{cached} cached")
    if summary:
        print(f"  Summary: {', '.join(summary)}")

    if not documents:
        print(f"No supported files found in {dir_path}")
        sys.exit(1)

    return documents


def index_documents(data_path, collection_name="bitnet_rag", db_path="rag/chroma_db",
                    embedding_model="all-MiniLM-L6-v2", batch_size=100, workers=1,
                    chroma_server=DEFAULT_CHROMA_SERVER):
    """Load documents, compute embeddings, and store in ChromaDB.

    Args:
        data_path: Path to data file or directory.
        collection_name: ChromaDB collection name.
        db_path: Path for local ChromaDB storage (fallback when server is unreachable).
        embedding_model: Sentence transformer model name.
        batch_size: Number of documents per indexing batch.
        workers: Parallel workers for file loading/OCR.
        chroma_server: ChromaDB server URL (default: http://localhost:8000).
                       Falls back to local storage if unreachable.
    """
    print(f"Loading documents from {data_path}...")
    documents = load_documents(data_path, workers=workers)
    print(f"Loaded {len(documents)} documents.")

    if not documents:
        print("No documents found.")
        sys.exit(1)

    print(f"Loading embedding model: {embedding_model}...")
    model = SentenceTransformer(embedding_model)

    client, target = _connect_chroma(chroma_server, db_path)

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

    print(f"Done. {len(documents)} documents indexed in {target}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Index documents for RAG")
    parser.add_argument("data_path", type=str, help="Path to data file or directory")
    parser.add_argument("--collection", type=str, default="bitnet_rag", help="ChromaDB collection name")
    parser.add_argument("--db-path", type=str, default="rag/chroma_db", help="Local ChromaDB fallback path")
    parser.add_argument("--embedding-model", type=str, default="all-MiniLM-L6-v2", help="Sentence transformer model")
    parser.add_argument("--batch-size", type=int, default=100, help="Indexing batch size")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers for file loading/OCR")
    parser.add_argument("--chroma-server", type=str, default=DEFAULT_CHROMA_SERVER,
                        help=f"ChromaDB server URL (default: {DEFAULT_CHROMA_SERVER}). "
                             "Falls back to local --db-path if unreachable.")
    args = parser.parse_args()

    index_documents(args.data_path, args.collection, args.db_path, args.embedding_model,
                    args.batch_size, args.workers, args.chroma_server)
