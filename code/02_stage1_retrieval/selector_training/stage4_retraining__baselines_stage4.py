"""Train-only lexical baselines, scored on the same fixed candidate pools."""
import json,collections,time
from pathlib import Path
import numpy as np,torch,sentencepiece as spm
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from policy_stage4 import export_run
from evaluate_v2 import evaluate,choose_threshold,write_json,per_question,summarize
BASE=Path('/home/<user>/UUM/EEE_research/experiments/stage4_retraining')
SP=spm.SentencePieceProcessor(model_file='/home/<user>/models/mdeberta-v3-base/spm.model')
def tokenize(s): return [str(i) for i in SP.encode(s) if i>3]
def features(rows):
 result=[]
 for p in rows:
  q=set(SP.encode(p['question']));e=set(SP.encode(p['text']));q-=set(range(4));e-=set(range(4));over=len(q&e)
  result.append([len(SP.encode(p['question'])),len(SP.encode(p['text'])),over,over/max(1,len(q)),over/max(1,len(e)),over/max(1,len(q|e))])
 return np.array(result)
def main():
 torch.set_num_threads(4);data=torch.load(BASE/'prepared.pt',map_location='cpu',weights_only=False);train=data['train'];out=BASE/'baselines';out.mkdir(exist_ok=True)
 allrows=train+data['dev']+data['test'];qtexts=[p['question'] for p in allrows];etexts=[p['text'] for p in allrows]
 n=len(train);nd=len(data['dev']);methods={}
 methods['random']=np.random.default_rng(42).random(len(allrows));methods['majority']=np.full(len(allrows),np.mean([p['y_support'] for p in data['train_full']]))
 corpus=list(dict.fromkeys(p['text'] for p in data['train_full']))
 tf=TfidfVectorizer(analyzer=tokenize,dtype=np.float32);tf.fit(corpus)
 qe=tf.transform(qtexts);ee=tf.transform(etexts);methods['tfidf']=np.asarray(qe.multiply(ee).sum(axis=1)).ravel()
 cv=CountVectorizer(analyzer=tokenize,vocabulary=tf.vocabulary_,dtype=np.float32);raw=cv.transform(corpus);df=np.asarray((raw>0).sum(axis=0)).ravel();idf=np.log(1+(len(corpus)-df+.5)/(df+.5));avgl=float(raw.sum(axis=1).mean())
 ep=cv.transform(etexts).tocoo();length=np.asarray(cv.transform(etexts).sum(axis=1)).ravel();ep.data=ep.data*2.2/(ep.data+1.2*(.25+.75*length[ep.row]/avgl))*idf[ep.col];ep=ep.tocsr();qp=cv.transform(qtexts);qp.data[:]=1
 bm=np.asarray(ep.multiply(qp).sum(axis=1)).ravel();scale=max(1.,float(np.quantile(bm[:n],.95)));methods['bm25']=1-np.exp(-bm/scale)
 xx=features(allrows);lr=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,random_state=42));lr.fit(xx[:n],[p['y_support'] for p in train]);methods['length_overlap']=lr.predict_proba(xx)[:,1]
 summary={}
 for name,scores in methods.items():
  dest=out/name;dest.mkdir(exist_ok=True);dev=scores[n:n+nd];test=scores[n+nd:];threshold=choose_threshold(data['dev'],dev)
  evaluate(data['dev'],dev,data['meta'],threshold,dest/'dev');summary[name]=evaluate(data['test'],test,data['meta'],threshold,dest/'test');np.save(dest/'test_scores.npy',test)
  np.save(dest/'dev_scores.npy',dev);export_run(data,dev,test,dest,'support')
  write_json(dest/'config.json',{'method':name,'threshold':threshold,'fit_split':'train_only','bm25_k1':1.2,'bm25_b':.75,'bm25_scale':scale,'seed':42})
  print(name,summary[name]['all'],flush=True)
 # Pool ceiling is necessary when provenance filtering removes a supporting reference.
 ceilings={}
 for split in ['dev','test']:
  ceilings[split]={}
  for subset in ['all','nonconstructed']:
   rr=[p for p in data[split] if subset=='all' or not p['constructed']]
   summary=summarize(per_question(rr,np.ones(len(rr)),data['meta'],0,m=100000,policy='threshold_only'))
   ceilings[split][subset]={k:summary[k] for k in ['questions','coverage','full_coverage','rankable_questions']}
 write_json(BASE/'candidate_pool_ceiling.json',ceilings);write_json(out/'DONE.json',{'methods':list(methods),'status':'COMPLETE'})
if __name__=='__main__': main()
