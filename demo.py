import json
from dotenv import load_dotenv
from src.llm import LLMService
from src.retrieval import vector_search,graph_search,graph_traversal
from src.fusion import rrf
load_dotenv()

def show(name,results):
    print("\n"+name+"\n"+"="*72)
    for x in results[:5]: print(f"[{x.get('id')}] {x.get('title','')} | {x.get('created_utc','')} | topics={x.get('topics',[])}")

def run(question):
    llm=LLMService(); info=llm.understand_query(question)
    print("\n"+"#"*72+f"\nQUESTION: {question}\n"+"#"*72); print(json.dumps(info,indent=2))
    periods=info.get("periods",[])
    if info["query_type"]=="temporal" and len(periods)>=2:
        period_results=[]; all_fused=[]
        for p in periods:
            for period in info.get("periods", []): v = vector_search(question, 5, start_date=period["start_date"], end_date=period["end_date"], topic=info.get("topic") or None, entity=info.get("entity") or None); g = graph_search(question, 5, topic=info.get("topic") or None, entity=info.get("entity") or None, start_date=period["start_date"], end_date=period["end_date"]); f = rrf(v, g); show("FUSED — " + period["label"], f); period_results.append((period["label"], f)); all_fused.extend(f); f=rrf(v,g); show("FUSED — "+p["label"],f); period_results.append((p["label"],f)); all_fused.extend(f)
        print("\nFINAL ANSWER\n"+llm.generate_answer(question,"temporal",all_fused,period_results=period_results)); return
    v=vector_search(question,5); g=graph_search(question,5,topic=info.get("topic") or None,entity=info.get("entity") or None); show("VECTOR-ONLY",v); show("GRAPH-ONLY",g); fused=rrf(v,g); show("RRF FUSED",fused)
    graph_info=[]
    if info.get("entity"):
        relationships,related_topics=graph_traversal(info["entity"],10); graph_info=relationships
        print("\nGRAPH TRAVERSAL\n"+"\n".join(f"{x.get('source')} --{x.get('relation')}--> {x.get('target')}" for x in relationships))
    print("\nFINAL ANSWER\n"+llm.generate_answer(question,info["query_type"],fused,graph_info=graph_info))

def main():
    for q in ["What are people discussing about AI?","Which topics are connected to RAG?","What are the main differences between RAG and graph-based retrieval discussions?","How has discussion about local LLMs changed between Q1 2025 and Q1 2026?"]: run(q)
if __name__=="__main__": main()
