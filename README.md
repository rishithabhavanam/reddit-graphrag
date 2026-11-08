# Hybrid GraphRAG for Time-Series Reddit Intelligence

Python 3.10+ implementation using official Reddit API/PRAW, Neo4j, ChromaDB, Sentence Transformers, Ollama and Reciprocal Rank Fusion.

## Setup
```powershell
cd C:\projects\reddit-graphrag
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
Copy `.env.example` to `.env`. Set Neo4j credentials. Start Ollama and Neo4j:
```powershell
ollama pull llama3.2:3b
ollama run llama3.2:3b
```
Then:
```powershell
python ingest.py
python verify.py
python demo.py
```

## Assignment coverage
Official Reddit API, multiple time windows, temporal metadata, post/comment thread structure, LLM topic/entity/sentiment extraction, entity relationships, Neo4j graph retrieval/traversal, ChromaDB vector retrieval with metadata filters, vector-only/graph-only/RRF results, LLM query understanding and answer generation, and temporal comparisons.

## Reddit API
The default is `DATA_SOURCE=sample` so the project can be developed before API approval. After official Reddit API approval, set `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT`, and `DATA_SOURCE=reddit`. No unofficial scraping is used.

## GitHub
Commit source code, sample data, README, requirements and `.env.example`. Do not commit `.env`, `.venv`, `chroma_db`, credentials or generated caches.
