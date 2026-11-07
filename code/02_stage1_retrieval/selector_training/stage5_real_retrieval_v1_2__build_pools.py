"""Score-blind stratified sampling and nested BM25 pools. No inferred gold labels."""
import json,re,hashlib,collections,sqlite3,time,traceback
from pathlib import Path
import numpy as np
from scipy import sparse
BASE=Path(__file__).resolve().parent; ROOT=BASE.parents[1]
OLD=ROOT/'experiments/stage4_retraining'; CORPUS=OLD/'actual_retrieval'
def save(name,data):
    (BASE/name).write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def digest(s):return hashlib.sha256(s.encode()).hexdigest()
def toks(s):return [x[i:i+2] for x in re.findall(r'[\u4e00-\u9fffA-Za-z]+',s) for i in range(len(x)-1)]
def main():
    refs=load(CORPUS/'retrieval_verification_tasks.json')
    records=load(ROOT/'dataset/stage1_sections_v1_1/medical_records_stage1.json')
    splits=load(OLD/'split_manifest.json')['task_ids'];trainids=set(splits['train'])
    foldbyid={tid:fold for fold,ids in splits.items() for tid in ids}
    dataset=load(OLD/'dataset.json');trainrefs=[t for t in dataset if t['id'] in trainids]
    seen={t['id'] for t in refs}
    refs.extend(dict(t,evaluation_split=foldbyid[t['id']]+'_retrospective') for t in dataset if t['id'] not in seen and t['id'] not in trainids)
    prior={t['task_id'] for t in load(ROOT/'experiments/stage4_verification_v1/reviewed_tasks.json')}
    eligible=collections.defaultdict(list); excluded=[]
    for t,role in [(t,'evaluation') for t in refs if t['id'] not in trainids]+[(t,'train') for t in trainrefs]:
        tid=t['id'];source='hallu' if tid.startswith('hallu') else 'med'
        if tid in prior:continue
        if role=='train':fold='train'
        else:fold='dev' if t['evaluation_split'].startswith('dev') else 'test'
        q=t['question'].replace('\\n','\n')
        if source=='hallu' and (re.search('选药|选方|方剂|常用|灸|首选药物|谁|创立|年代|创始|移植|医学伦理',q) or not re.search('病机|证候|治法|治疗原则|辨证|证可见|治疗|原因',q)):
            excluded.append({'id':tid,'reason':'outside core scope'});continue
        reference=t.get('answer','');analysis=None;treatment=None
        if source=='med':
            rec=records[int(tid.rsplit('_',1)[1])];analysis=rec['病情病因诊断分析'];treatment=rec['治则及治疗方案']
            if not re.search('治疗原则|治则|治法|治疗方法|治疗初期应该|治疗应|治疗以',analysis+treatment):
                excluded.append({'id':tid,'reason':'no explicit source principle'});continue
        # Source points define eligibility strata, never retrieval terms.
        for typ in ['diagnosis','principle']:
            def principle(p):
                if source=='hallu':return bool(re.search('治疗|治法|治则',q))
                if p.get('type'):return bool(re.search('治疗|治则|治法',p['type']))
                field=p.get('source_field','');start=p.get('source_start_char',0)
                surrounding=(analysis or '')[max(0,start-18):start]
                return field=='reference_treatment' or bool(re.search('治疗|治则|治法',surrounding))
            points=[p for p in t.get('answer_points',[]) if principle(p)==(typ=='principle')]
            if not points:continue
            background=q.split('请解释')[0].strip() if source=='med' else q
            question=background+('\n请解释该病例的诊断、证候与病因病机。' if typ=='diagnosis' else '\n请说明适用于该病例的治疗原则，不要求方药剂量。') if source=='med' else q
            eligible[(fold,source,typ)].append({'task_id':tid,'source':source,'split':fold,'question_type':typ,'question':question,'original_question':q,
                'provisional_points':points,'reference_answer':reference,'reference_analysis':analysis,'reference_treatment':treatment,
                'point_review_status':'pending','original_fold':t.get('evaluation_split','train')})
    chosen=[];used=set();counts=[]
    for fold in ['test','dev','train']:
        for source in ['med','hallu']:
            for typ in ['diagnosis','principle']:
                target=20 if source=='med' else 10
                pool=sorted(eligible[(fold,source,typ)],key=lambda t:digest('20261012:'+fold+':'+typ+':'+t['task_id']))
                accepted=[]
                for t in pool:
                    if t['task_id'] in used:continue
                    accepted.append(t);used.add(t['task_id'])
                    if len(accepted)==target:break
                chosen.extend(accepted);counts.append({'split':fold,'source':source,'question_type':typ,'eligible':len(pool),'target':target,'actual':len(accepted)})
    assert all(t['task_id'] in trainids for t in chosen if t['split']=='train')
    assert not any(t['task_id'] in trainids for t in chosen if t['split']!='train')
    if (BASE/'SAMPLING_MANIFEST.json').exists():
        assert load(BASE/'review_tasks.json')==chosen,'Frozen selection changed'
    else:
        save('review_tasks.json',chosen)
        save('SAMPLING_MANIFEST.json',{'seed':20261012,'counts':counts,'exclusions':excluded,'scores_used':False,'sampling_sha256':digest(json.dumps(chosen,ensure_ascii=False,sort_keys=True)),
            'clinical_expert_gold':False,'retrospective_evaluation':True,'reference_used_for_retrieval':False})
    save('STATUS.json',{'status':'SAMPLING_FROZEN','tasks':len(chosen)})
    # Only questions enter query-term projection; answers remain in sealed review file.
    words=sorted({w for t in chosen for w in toks(t['question'])});wi={w:i for i,w in enumerate(words)}
    db=sqlite3.connect(f'file:{CORPUS/"retrieval_corpus.sqlite"}?mode=ro',uri=True)
    rr=[];cc=[];vv=[];lengths=[];docids=[]
    for i,(docid,text) in enumerate(db.execute('SELECT id,text FROM documents ORDER BY id')):
        count=collections.Counter(toks(text));lengths.append(sum(count.values()));docids.append(docid)
        for w,n in count.items():
            if w in wi:rr.append(i);cc.append(wi[w]);vv.append(n)
        if (i+1)%200000==0:print('SCAN',i+1,flush=True)
    mat=sparse.csr_matrix((np.asarray(vv,dtype=np.float32),(rr,cc)),shape=(len(lengths),len(words)))
    del rr,cc,vv
    n=len(lengths);df=np.asarray((mat>0).sum(0)).ravel();idf=np.log(1+(n-df+.5)/(df+.5));lens=np.asarray(lengths)
    norm=1.2*(.25+.75*lens/max(lens.mean(),1));coo=mat.tocoo();coo.data=coo.data*2.2/(coo.data+norm[coo.row])*idf[coo.col];bm=coo.tocsr();del coo,mat
    rows=[]
    for task in chosen:
        indices=[wi[w] for w in set(toks(task['question']))];qv=sparse.csr_matrix((np.ones(len(indices)),(indices,np.zeros(len(indices),dtype=int))),shape=(len(words),1))
        scores=(bm@qv).toarray().ravel();top=np.argpartition(scores,-100)[-100:];top=sorted(top,key=lambda i:(-scores[i],int(docids[i])))
        for rank,index in enumerate(top,1):
            source,a,b,text,h=db.execute('SELECT source,start_char,end_char,text,sha256 FROM documents WHERE id=?',(int(docids[index]),)).fetchone()
            rows.append({'task_id':task['task_id'],'candidate_id':f"{task['task_id']}__s5_{rank:03d}",'split':task['split'],'question_type':task['question_type'],'question':task['question'],
                'text':text,'title':None,'context_text':None,'document_id':int(docids[index]),'retrieval':{'source':source,'start_char':a,'end_char':b,'sha256':h,'rank':rank,'score':float(scores[index])},
                'y_support':None,'y_relevance':None,'covers':None,'partial_support':None,'condition_conflict':None,'annotation_status':'unreviewed'})
        print('QUERY',task['task_id'],flush=True)
    db.close()
    # Read source lines in original character coordinate system. Keep authentic short
    # headings only; an unknown title remains null. Context never comes from answers.
    bysource=collections.defaultdict(list)
    for row in rows:
        if not row['retrieval']['source'].startswith('medical_train'):bysource[row['retrieval']['source']].append(row)
        else:row['context_text']=row['text'];row['context_status']='original_medical_train_fragment'
    for source,group in bysource.items():
        group.sort(key=lambda r:r['retrieval']['start_char']);j=0;offset=0;heading=None
        with (ROOT/'dataset'/source).open(encoding='utf-8-sig') as f:
            for line in f:
                cleaned=line.strip()
                if 2<=len(cleaned)<=60 and re.match(r'^(第[一二三四五六七八九十百0-9]+[章节篇]|[一二三四五六七八九十]+[、．.]|【)',cleaned):heading=cleaned
                while j<len(group) and group[j]['retrieval']['start_char']<offset+len(line):
                    row=group[j];pos=row['retrieval']['start_char']-offset
                    if pos>=0:
                        # Preserve actual nearby context; never fabricate source titles.
                        context=line[max(0,pos-160):min(len(line),pos+len(row['text'])+160)].strip()
                        row['title']=heading;row['context_text']=((heading+'\n') if heading else '')+context
                        row['context_status']='source_line_context_title_present' if heading else 'source_line_context_title_unavailable'
                    else:row['context_text']=row['text'];row['context_status']='offset_unresolved_original_retained'
                    j+=1
                offset+=len(line)
                if j==len(group):break
        for row in group[j:]:row['context_text']=row['text'];row['context_status']='offset_unresolved_original_retained'
    with (BASE/'candidates_unlabeled.jsonl').open('w',encoding='utf-8') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    save('POOL_MANIFEST.json',{'tasks':len(chosen),'pairs':len(rows),'nested_k':[20,50,100],'reference_insertion':False,'reference_retrieval':False,'labels_complete':False,
        'contexts':dict(collections.Counter(r['context_status'] for r in rows)),'input_sha256':hashlib.sha256((BASE/'candidates_unlabeled.jsonl').read_bytes()).hexdigest()})
    save('STATUS.json',{'status':'RETRIEVAL_COMPLETE_REVIEW_PENDING','tasks':len(chosen),'pairs':len(rows),'training_started':False})
    print('RETRIEVAL_COMPLETE_REVIEW_PENDING',len(rows),flush=True)
if __name__=='__main__':
    try:main()
    except Exception:
        save('STATUS.json',{'status':'BUILD_FAILED','traceback':traceback.format_exc()});raise
