from src.graph_store import GraphStore
from src.vector_store import VectorStore
g=GraphStore(); print("NEO4J:",g.graph_stats()); g.close(); v=VectorStore(); print("CHROMA DOCUMENTS:",v.collection.count())
