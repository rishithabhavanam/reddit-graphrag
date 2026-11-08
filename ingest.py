import os
from dotenv import load_dotenv
from src.graph_store import GraphStore
from src.vector_store import VectorStore
from src.llm import LLMService
from src.sample_data import load_sample_posts
from src.reddit_ingest import load_reddit_posts
load_dotenv()

def main():
    posts=load_reddit_posts() if os.getenv("DATA_SOURCE","sample").lower()=="reddit" else load_sample_posts()
    print(f"Loaded {len(posts)} posts.")
    graph=GraphStore(); vector=VectorStore(); llm=LLMService()
    graph.clear_database(); vector.reset()
    for i,post in enumerate(posts,1):
        print(f"[{i}/{len(posts)}] LLM extraction: {post.post_id}")
        extracted=llm.extract_information(post.title,post.body)
        # Deterministic enrichment keeps required graph relationships stable even if the local LLM varies.
        if "RAG" in extracted.get("entities", []) and "graph retrieval" in extracted.get("topics", []):
            if "Graph retrieval" not in extracted["entities"]: extracted["entities"].append("Graph retrieval")
            extracted.setdefault("relationships", []).append({"source":"RAG","relation":"related_to","target":"Graph retrieval"})
        graph.add_post(post,extracted); vector.add_post(post)
    graph.close(); print("Ingestion completed: Neo4j + ChromaDB updated.")
if __name__=="__main__": main()
