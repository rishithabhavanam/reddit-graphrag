def normalize(results):
    if isinstance(results, dict):
        ids=results.get("ids",[[]]); docs=results.get("documents",[[]]); metas=results.get("metadatas",[[]])
        ids=ids[0] if ids and isinstance(ids[0],list) else ids; docs=docs[0] if docs and isinstance(docs[0],list) else docs; metas=metas[0] if metas and isinstance(metas[0],list) else metas
        return [{"id":str(k),"text":docs[i] if i<len(docs) else "",**(metas[i] if i<len(metas) and metas[i] else {})} for i,k in enumerate(ids)]
    return [x for x in results if isinstance(x,dict)] if isinstance(results,list) else []

def reciprocal_rank_fusion(result_lists, k=60):
    scores, detail = {}, {}
    for results in result_lists:
        for rank, item in enumerate(normalize(results), 1):
            key=str(item.get("id") or item.get("post_id") or "")
            if key:
                scores[key]=scores.get(key,0)+1/(k+rank); detail.setdefault(key,item.copy())
    out=[]
    for key in sorted(scores,key=scores.get,reverse=True):
        x=detail[key]; x["rrf_score"]=scores[key]; out.append(x)
    return out

def rrf(vector_results, graph_results, k=60): return reciprocal_rank_fusion([vector_results,graph_results],k)
