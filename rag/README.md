# BitNet RAG Pipeline

Retrieval-Augmented Generation using BitNet for CPU-friendly Q&A over your own data.

## How It Works

```mermaid
flowchart TD
    subgraph INDEX["Indexing (one-time)"]
        A[Your Data<br/>JSON / CSV / TXT] -->|index_data.py| B[Load Documents]
        B --> C[Sentence Transformer<br/>all-MiniLM-L6-v2]
        C -->|Compute embeddings| D[(ChromaDB<br/>Vector Store)]
    end

    subgraph QUERY["Query (each question)"]
        E[User Question] --> F[Sentence Transformer<br/>Encode question]
        F -->|Similarity search| D
        D -->|Top-K documents| G[Build Prompt]
        G -->|"Context + Question"| H[BitNet Server<br/>llama-server :8080]
        H -->|Generated answer| I[Response to User]
    end

    style INDEX fill:#1a1a2e,stroke:#e94560,color:#eee
    style QUERY fill:#1a1a2e,stroke:#0f3460,color:#eee
```

## Setup

```bash
pip install -r rag/requirements.txt
```

## Index Your Data

Supports JSON, JSONL, CSV, TXT, and MD files. Can also index a directory of files.

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
python rag/index_data.py <data_path> [--collection NAME] [--db-path PATH] [--embedding-model MODEL] [--batch-size N]
```

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
| `--db-path` | `rag/chroma_db` | ChromaDB storage path |
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