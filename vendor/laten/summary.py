"""Original V6 648-row counts and BASE/CF pairing, unchanged function."""
from .prompts import vector_from_variant

def summarize(rows,variants):
    def aggregate(items):
        n=len(items);bits=sum(r['information_bit_total'] for r in items);correct=sum(r.get('information_bit_correct_count',0) for r in items)
        return {'sample_count':n,'success_count':sum(r['status']=='success' for r in items),'vector_exact':sum(bool(r.get('information_vector_exact')) for r in items),'bit_correct':correct,'bit_total':bits,'functional_action_exact':sum(bool(r.get('functional_action_exact')) for r in items),'model_action_exact':sum(bool(r.get('model_action_exact')) for r in items),'reasoning_cap_count':sum(bool(r.get('reasoning_cap_hit')) for r in items)}
    result={'completed':len(rows),'expected':len(variants),'metrics':aggregate(rows),'by':{}}
    for field in ('split','domain','graph_level','information_level','reasoning_level'):
        result['by'][field]={v:aggregate([r for r in rows if r[field]==v]) for v in sorted({r[field] for r in rows})}
    byid={r['variant_id']:r for r in rows};base={v.case_id:v for v in variants if v.target_fact_id is None};pairs=[]
    for v in variants:
        if v.target_fact_id is None:continue
        b=base[v.case_id]
        if b.variant_id not in byid or v.variant_id not in byid:continue
        rb,rc=byid[b.variant_id],byid[v.variant_id];gb,gc=vector_from_variant(b),vector_from_variant(v)
        k=next(i for i,(a,z) in enumerate(zip(gb,gc)) if a!=z);pb,pc=rb.get('predicted_information_vector') or '',rc.get('predicted_information_vector') or ''
        valid=len(pb)==len(gb) and len(pc)==len(gc) and rb['status']=='success' and rc['status']=='success'
        pairs.append({'case_id':v.case_id,'split':v.split,'graph_level':v.graph_level,'information_level':v.information_level,'target_pair_exact':valid and pb[k]==gb[k] and pc[k]==gc[k],'non_target_pair_exact':valid and all(pb[j]==gb[j] and pc[j]==gc[j] for j in range(len(gb)) if j!=k),'both_vectors_exact':valid and pb==gb and pc==gc})
    def ps(items):return {'pair_count':len(items),**{f:sum(bool(r[f]) for r in items) for f in ('target_pair_exact','non_target_pair_exact','both_vectors_exact')}}
    result['pairs']=ps(pairs);result['pairs_by']={f:{v:ps([r for r in pairs if r[f]==v]) for v in sorted({r[f] for r in pairs})} for f in ('split','graph_level','information_level')}
    return result
