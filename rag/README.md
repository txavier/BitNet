# BitNet RAG Pipeline

Retrieval-Augmented Generation using BitNet for CPU-friendly Q&A over your own data.

## How It Works

```mermaid
flowchart TD
    subgraph INDEX["Indexing (one-time)"]
        A[Your Data<br/>JSON / CSV / TXT / PDF / Web] -->|index_data.py| B[Load Documents]
        B --> C[Sentence Transformer<br/>all-MiniLM-L6-v2]
        C -->|Compute embeddings| D[(ChromaDB<br/>Vector Store)]
    end

    subgraph SCRAPE["Web Scraping (optional)"]
        S1[Website URL] -->|scrape_web.py| S2[trafilatura<br/>Extract text]
        S2 --> S3[MD files]
        S4[PDF links] -->|pdf_handler.py| S5[PyMuPDF + OCR]
        S5 --> S3
        S3 -->|Feed into| A
    end

    subgraph QUERY["Query (each question)"]
        E[User Question] --> F[Sentence Transformer<br/>Encode question]
        F -->|Similarity search| D
        D -->|Top-K documents| G[Build Prompt]
        G -->|"Context + Question"| H[BitNet Server<br/>llama-server :8080]
        H -->|Generated answer| I[Response to User]
    end

    style INDEX fill:#1a1a2e,stroke:#e94560,color:#eee
    style SCRAPE fill:#1a1a2e,stroke:#16c79a,color:#eee
    style QUERY fill:#1a1a2e,stroke:#0f3460,color:#eee
```

## Setup

```bash
pip install -r rag/requirements.txt
```

## Index Your Data

Supports JSON, JSONL, CSV, TXT, MD, and PDF files. Can also index a directory of files.

```bash
# Index a directory of files
python rag/index_data.py rag/sampling_data

# Or a single file
python rag/index_data.py rag/sampling_data.json

# Your own data
python rag/index_data.py /path/to/your/data.csv
```

Options:
```
python rag/index_data.py <data_path> [--collection NAME] [--db-path PATH] [--embedding-model MODEL] [--batch-size N] [--workers N] [--chroma-server URL]
```

| Option | Default | Description |
|---|---|---|
| `--collection` | `bitnet_rag` | ChromaDB collection name |
| `--db-path` | `rag/chroma_db` | Path to store local ChromaDB |
| `--embedding-model` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `--batch-size` | `100` | Documents per indexing batch |
| `--workers` | `1` | Parallel workers for file loading/OCR |
| `--chroma-server` | `http://localhost:8000` | ChromaDB server URL (auto-falls back to local `--db-path` if unreachable) |

**Multi-node indexing:** Run a [ChromaDB server](https://docs.trychroma.com/docs/run-chroma/chroma-server) on a shared host — all tools auto-connect to `localhost:8000` by default:
```bash
# Start ChromaDB server (once)
chroma run --host 0.0.0.0 --port 8000

# On each node — no extra flags needed
python rag/index_data.py /local/data --workers 4

# Or override for a remote server
python rag/index_data.py /local/data --chroma-server http://chroma-host:8000
```

If no server is running, both `index_data.py` and `query.py` automatically fall back to local storage at `--db-path`.

## Scrape a Website

Crawl a website and extract content as MD files, then index them.
PDF links found during crawling are automatically downloaded and OCR'd.

```bash
# Scrape a single page
python rag/scrape_web.py https://www.saabnet.com/tsn/faq/c900.html

# Crawl a section (follows links up to depth 2)
python rag/scrape_web.py https://www.saabnet.com/tsn/faq/ --crawl --max-pages 50

# Multiple sites in parallel (one thread per domain)
python rag/scrape_web.py https://www.saabnet.com/tsn/faq/ https://www.saabplanet.com/tech/ --crawl --workers 2

# Then index the scraped data
python rag/index_data.py rag/sampling_data
```

Options:
```
python rag/scrape_web.py <url> [url2 ...] [-o OUTPUT_DIR] [--crawl] [--max-depth N] [--max-pages N] [--delay SECS] [--workers N]
```

| Option | Default | Description |
|---|---|---|
| `-o` | `rag/sampling_data` | Output directory for scraped files |
| `--crawl` | off | Follow links from the start URL |
| `--max-depth` | `2` | Max link depth when crawling |
| `--max-pages` | `50` | Max pages to scrape per URL |
| `--delay` | `1.0` | Seconds between requests (be polite) |
| `--force` | off | Re-download files even if they already exist |
| `--workers` | `1` | Parallel workers for multi-URL scraping |

The scraper respects `robots.txt` and identifies itself as `BitNet-RAG-Scraper`.

## Process PDFs

Extract text from PDFs using PyMuPDF with automatic OCR fallback for scanned pages.

```bash
# Single PDF
python rag/pdf_handler.py manual.pdf -o rag/sampling_data/

# Directory of PDFs
python rag/pdf_handler.py /path/to/pdfs/ -o rag/sampling_data/

# PDFs in sampling_data are also auto-detected by index_data.py
python rag/index_data.py rag/sampling_data
```

PDFs are handled three ways:
1. **Standalone** — `pdf_handler.py` converts PDFs to MD files
2. **During indexing** — `index_data.py` auto-detects `.pdf` files in directories
3. **During scraping** — `scrape_web.py` downloads and processes PDF links found on pages

## Query

### Start the BitNet server (Terminal 1)
```bash
python run_inference_server.py -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf -n 512
```

### Interactive mode (Terminal 2)
```bash
python rag/query.py
```

### Single question
```bash
python rag/query.py -q "How much horsepower does a Saab 9000 have?"
```

### All query options
```
python rag/query.py [-q QUERY] [--server-url URL] [--collection NAME] [--db-path PATH] [--embedding-model MODEL] [--top-k N] [--max-tokens N] [--temperature T]
```

| Option | Default | Description |
|---|---|---|
| `-q` | *(interactive)* | Single question to ask |
| `--server-url` | `http://127.0.0.1:8080` | BitNet server URL |
| `--collection` | `bitnet_rag` | ChromaDB collection name |
| `--db-path` | `rag/chroma_db` | Local ChromaDB fallback path |
| `--chroma-server` | `http://localhost:8000` | ChromaDB server URL (auto-falls back to local) |
| `--top-k` | `3` | Number of documents to retrieve |
| `--max-tokens` | `512` | Max tokens for response |
| `--temperature` | `0.3` | Generation temperature |

## Data Formats

**JSON** — array of objects with `text` or `content` field:
```json
[{"text": "Your document here."}, {"text": "Another document."}]
```

**JSONL** — one JSON object per line:
```
{"text": "Your document here."}
{"text": "Another document."}
```

**CSV** — with `text` or `content` column:
```
text
"Your document here."
"Another document."
```

**TXT** — one document per line:
```
Your document here.
Another document.
```

**PDF** — text PDFs are extracted directly; scanned PDFs are OCR'd automatically.

System dependencies for PDF/OCR:
```bash
apt-get install tesseract-ocr poppler-utils
```


## Reference

Non-RAG flow

```mermaid
flowchart TD
    subgraph SETUP["setup_env.py (Build & Prepare)"]
        A[parse_args] --> B[main]
        B --> C[setup_gguf]
        C -->|pip install gguf-py| C1[Install 3rdparty/llama.cpp/gguf-py]
        B --> D[gen_code]
        D --> D1{Detect CPU arch}
        D1 -->|ARM| D2[codegen_tl1.py]
        D1 -->|x86_64| D3[codegen_tl2.py]
        D2 --> D4[Generate bitnet-lut-kernels.h]
        D3 --> D4
        D4 --> D5{Use pretuned?}
        D5 -->|Yes| D6[Copy preset_kernels/ headers]
        D5 -->|No| D7[Generate fresh kernels]
        B --> E[compile]
        E --> E1["cmake -B build (generate build files)"]
        E1 --> E2["cmake --build build (compile C++)"]
        E2 --> E3[Build binaries in build/bin/]
        B --> F[prepare_model]
        F --> F1{Source?}
        F1 -->|HF repo| F2[huggingface-cli download]
        F1 -->|Local dir| F3[Load from model_dir]
        F2 --> F4{GGUF exists?}
        F3 --> F4
        F4 -->|No| F5{Quant type?}
        F4 -->|Yes| F6[Skip conversion]
        F5 -->|tl1/tl2| F7[convert-hf-to-gguf-bitnet.py<br/>with --outtype tl]
        F5 -->|i2_s| F8[convert-hf-to-gguf-bitnet.py<br/>--outtype f32]
        F8 --> F9[llama-quantize → i2_s GGUF]
        F7 --> F10[ggml-model-*.gguf]
        F9 --> F10
    end

    subgraph INFER["run_inference.py (CLI Inference)"]
        G[parse_args] --> H[run_inference]
        H --> I["build/bin/llama-cli"]
        I --> I1["-m model.gguf<br/>-n tokens -t threads<br/>-p prompt -ngl 0 -b 1"]
        I1 --> I2{--conversation?}
        I2 -->|Yes| I3[Chat mode -cnv]
        I2 -->|No| I4[Text completion]
    end

    subgraph SERVER["run_inference_server.py (HTTP Server)"]
        J[parse_args] --> K[run_server]
        K --> L["build/bin/llama-server"]
        L --> L1["-m model.gguf<br/>--host --port<br/>-cb continuous batching"]
    end

    SETUP -->|"Produces GGUF model<br/>& compiled binaries"| INFER
    SETUP -->|"Produces GGUF model<br/>& compiled binaries"| SERVER
```