# RAG Advanced — Learning Repository

A hands-on learning repository for advanced Retrieval-Augmented Generation (RAG) techniques. Each module targets a specific concept, building toward a fully production-style RAG pipeline with hybrid retrieval, structured outputs, and agentic workflows.

---

## Topics Covered

| Module | Concepts |
|--------|----------|
| **Data Ingest & Parsing** | Multi-format document loaders, text splitting strategies, metadata preservation |
| **Vector Embeddings** | Embedding models (local & API), cosine similarity, semantic search |
| **Vector Stores & Databases** | ChromaDB, FAISS, persistence, similarity search |
| **Hybrid Search** | BM25 (sparse) + semantic (dense) retrieval, score fusion, weighted ranking |
| **Structured Output** | Pydantic models, JSON schema validation, LLM-enforced output formats |
| **Agents** | Tool use, agent loops, context management, human-in-the-loop approval |
| **Streaming** | Token-by-token response streaming in a live Gradio UI |
| **Local LLMs** | Ollama integration (gemma3:12b, qwen3-embedding:0.6b) |

---

## Project Structure

```
RAG_Advanced/
├── Agents/                          # LangChain agent patterns
│   ├── langchain_agent_basic.ipynb      # Tool binding, agent loops
│   ├── context_management_strategy.ipynb# Message history, token-based summarization
│   └── human_in_the_loop.ipynb          # Approval workflows
│
├── DataIngestParsing/               # Document loading & chunking
│   ├── dataingestion.ipynb              # PDF, Excel, CSV, JSON, DOCX, SQLite loaders
│   └── data/                            # Git-ignored — local data files
│       ├── csv_excel/                       # Excel & CSV tables (policies, users, cars)
│       ├── pdf/                             # PDF documents (terms, privacy policy, samples)
│       ├── text_files/                      # Plain text files
│       ├── json/                            # JSON data files
│       ├── database/                        # SQLite databases
│       └── docx/                            # Word documents
│
├── Pydantic/                        # Structured LLM outputs
│   ├── structured_output.ipynb          # Pydantic BaseModel + LLM schema enforcement
│   └── agent_structured_output.ipynb    # Agents returning typed objects
│
├── VectorEmbedding/                 # Embedding model exploration
│   └── embedding.ipynb                  # HuggingFace vs OpenAI embeddings, similarity metrics
│
└── VectorStoreAndVectorDatabase/    # Production RAG application
    ├── fnx.py                           # Core RAG engine (hybrid retrieval, streaming)
    ├── app.py                           # Gradio web UI (RTL / Hebrew support)
    ├── chromadb.ipynb                   # ChromaDB setup and queries
    ├── fnx.ipynb                        # Interactive development notebook
    └── terms_condition*.ipynb           # PDF-to-vector pipeline examples
```

---

## Key Techniques

### Hybrid Search (BM25 + Semantic)
`VectorStoreAndVectorDatabase/fnx.py` implements a custom hybrid retriever that combines:
- **BM25** (80% weight) — exact keyword matches, great for IDs, policy numbers, names
- **Chroma semantic search** (20% weight) — dense vector similarity for conceptual queries
- **Direct metadata lookup** — short-circuit for exact ID/number matches, bypassing the ranker entirely

### Document Loaders Covered
`TextLoader`, `PyPDFLoader`, `PyMuPDFLoader`, `Docx2txtLoader`, `CSVLoader`, `JSONLoader`, `SQLDatabaseLoader`

### Text Splitting Strategies
`CharacterTextSplitter`, `RecursiveCharacterTextSplitter`, `TokenTextSplitter` — each with chunk size and overlap tuning

### Embedding Models
| Model | Provider | Dimensions |
|-------|----------|-----------|
| `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace (local) | 384 |
| `text-embedding-3-small` | OpenAI API | 1536 |
| `qwen3-embedding:0.6b` | Ollama (local) | varies |

---

## Tech Stack

**LLM Providers**: OpenAI, Groq, Ollama (local)

**Vector Stores**: ChromaDB, FAISS

**Core Libraries**:
- `langchain`, `langchain-community`, `langchain-openai`, `langchain-ollama`, `langchain-chroma`
- `rank-bm25` — BM25 ranking
- `sentence-transformers` — local embeddings
- `pymupdf`, `python-docx`, `unstructured`, `openpyxl` — document parsing
- `gradio` — web UI
- `pydantic` — structured outputs
- `tiktoken` — token counting

---

## Getting Started

### Prerequisites
- Python 3.13+
- [uv](https://github.com/astral-sh/uv) (recommended) or pip
- [Ollama](https://ollama.com/) for local models

### Install

```bash
uv sync
# or
pip install -r requirements.txt
```

### Environment Variables

Create a `.env` file:

```env
OPENAI_API_KEY=your_key_here
HUGGINGFACEHUB_API_TOKEN=your_token_here
```

### Run the Demo App

```bash
python VectorStoreAndVectorDatabase/app.py
```

### Explore Notebooks

Open any `.ipynb` in Jupyter Lab or VS Code. Each notebook is self-contained and walks through a specific concept with runnable cells and inline explanations.

---

## Learning Path (Suggested Order)

1. `DataIngestParsing/dataingestion.ipynb` — load and chunk documents
2. `VectorEmbedding/embedding.ipynb` — understand embeddings and similarity
3. `VectorStoreAndVectorDatabase/chromadb.ipynb` — store and query vectors
4. `VectorStoreAndVectorDatabase/fnx.ipynb` — build the hybrid retrieval engine
5. `Pydantic/structured_output.ipynb` — enforce output schemas
6. `Agents/langchain_agent_basic.ipynb` — add tool use and reasoning loops
7. `Agents/context_management_strategy.ipynb` — manage long conversations
8. `Agents/human_in_the_loop.ipynb` — add human approval steps
