"""Retain original question-only pools; re-retrieve sanitized queries from frozen DB."""
import json,sqlite3,re,collections,hashlib,time
from pathlib import Path
import numpy as np, scipy.sparse as sparse, torch, sentencepiece as spm
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[1]
OLD=ROOT/'experiments/stage4_retraining/actual_retrieval'
def toks(s): return [p[i:i+2] for p in re.findall(r'[\u4e00-\u9fffA-Za-z]+',s) for i in range(len(p)-1)]
def save(n,v):(BASE/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
tasks=[t for t in json.loads((BASE/'reviewed_tasks.json').read_text(encoding='utf-8')) if t['included']]
original=[json.loads(s) for s in (OLD/'actual_retrieval_candidates.jsonl').read_text(encoding='utf-8').splitlines()]
bytask=collections.defaultdict(list)
for row in original:bytask[row['task_id']].append(row)
sp=spm.SentencePieceProcessor(model_file='/home/<user>/models/mdeberta-v3-base/spm.model')
changed=[t for t in tasks if t['input_changed']];newrows={}
if changed:
    words=sorted({w for t in changed for w in toks(t['question'])});wi={w:i for i,w in enumerate(words)}
    db=sqlite3.connect(f'file:{OLD / "retrieval_corpus.sqlite"}?mode=ro',uri=True)
    rr=[];cc=[];vv=[];lengths=[];docids=[]
    for i,(docid,text) in enumerate(db.execute('SELECT id,text FROM documents ORDER BY id')):
        count=collections.Counter(toks(text));lengths.append(sum(count.values()));docids.append(docid)
        for w,n in count.items():
            if w in wi:rr.append(i);cc.append(wi[w]);vv.append(n)
        if (i+1)%200000==0: print('SCAN',i+1,flush=True)
    n=len(lengths);mat=sparse.csr_matrix((np.asarray(vv,dtype=np.float32),(rr,cc)),shape=(n,len(words)))
    df=np.asarray((mat>0).sum(0)).ravel();idf=np.log(1+(n-df+.5)/(df+.5));lens=np.asarray(lengths)
    lengthnorm=1.2*(.25+.75*lens/max(lens.mean(),1));coo=mat.tocoo();coo.data=coo.data*2.2/(coo.data+lengthnorm[coo.row])*idf[coo.col];bm=coo.tocsr()
    for task in changed:
        qwords=set(toks(task['question']));indices=[wi[w] for w in qwords if w in wi]
        qv=sparse.csr_matrix((np.ones(len(indices)),(indices,np.zeros(len(indices),dtype=int))),shape=(len(words),1))
        scores=(bm@qv).toarray().ravel();top=np.argpartition(scores,-20)[-20:];top=sorted(top,key=lambda i:(-scores[i],int(docids[i])))
        result=[]
        for rank,index in enumerate(top,1):
            source,a,b,text,h=db.execute('SELECT source,start_char,end_char,text,sha256 FROM documents WHERE id=?',(int(docids[index]),)).fetchone()
            if len(sp.encode(text))>320:
                low,high=0,len(text)
                while low+1<high:
                    mid=(low+high)//2
                    if len(sp.encode(text[:mid]))<=320:low=mid
                    else:high=mid
                text=text[:low]
            result.append({'task_id':task['task_id'],'candidate_id':f"{task['task_id']}__review_retrieved_{rank:02d}",'question':task['question'],'text':text,
                           'retrieval':{'source':source,'start_char':a,'end_char':a+len(text),'document_sha256':h,'rank':rank,'bm25_score':float(scores[index])}})
        newrows[task['task_id']]=result
    db.close()
pool=[];prepared=[];meta={}
for task in tasks:
    tid=task['task_id'];meta[tid]={'required':[p['id'] for p in task['necessary_points']],'source':task['source'],'group':tid,'split':task['split']}
    for original_row in newrows.get(tid,bytask[tid]):
        row={k:v for k,v in original_row.items() if k not in ['anchor_covers','anchor_match_support','annotation_status','support_extent']}
        row.update(question=task['question'],source=task['source'],group=tid,y_support=None,y_relevance=None,covers=[],constructed=False,evaluation_split=task['split'],annotation_status='AI_full_pool_review_pending')
        pool.append(row);ids=[1]+sp.encode(row['question'])+[2]+sp.encode(row['text'])+[2]
        assert len(ids)<=512,(tid,len(ids));prepared.append(row|{'ids':ids})
assert len(pool)==len(tasks)*20
with (BASE/'candidates_unlabeled.jsonl').open('w',encoding='utf-8') as f:
    for row in pool:f.write(json.dumps(row,ensure_ascii=False)+'\n')
torch.save({'test':prepared,'meta':meta},BASE/'prepared_unlabeled.pt')
save('POOL_MANIFEST.json',{'tasks':len(tasks),'pairs':len(pool),'new_question_queries':[t['task_id'] for t in changed],
                          'reference_insertion':False,'reference_used_for_retrieval':False,'model_scores_used_for_sampling_or_labels':False,
                          'same_train_only_corpus':True,'original_corpus_manifest':json.loads((OLD/'retrieval_manifest.json').read_text(encoding='utf-8')),
                          'candidate_sha256':hashlib.sha256((BASE/'candidates_unlabeled.jsonl').read_bytes()).hexdigest()})
print('POOL_READY',len(tasks),len(pool),flush=True)
