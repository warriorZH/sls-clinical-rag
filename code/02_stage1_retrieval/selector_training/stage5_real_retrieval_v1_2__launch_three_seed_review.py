import subprocess,sys,os,json,traceback
from pathlib import Path
B=Path(__file__).resolve().parent
def save(n,v):(B/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
 query=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
 assert all(int(line.strip())<6000 for line in query.splitlines()),'Insufficient available GPU capacity; not interrupting other jobs'
 procs=[];logs=[]
 for gpu,obj in enumerate(['support','relevance']):
  env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=str(gpu),PYTHONUTF8='1',PYTHONIOENCODING='utf-8',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='2')
  log=(B/f'direct_three_seed_{obj}.log').open('a',encoding='utf-8');logs.append(log)
  p=subprocess.Popen([sys.executable,'-u',str(B/'score_three_seed_review.py'),'--objective',obj],stdout=log,stderr=subprocess.STDOUT,env=env);procs.append(p)
 save('DIRECT_THREE_SEED_STATUS.json',{'status':'SCORING_RUNNING','pids':[p.pid for p in procs],'seeds':[42,2026,3407],'training_started':False,'labels_complete':False})
 codes=[p.wait() for p in procs]
 for log in logs:log.close()
 assert codes==[0,0],codes
 save('DIRECT_THREE_SEED_STATUS.json',{'status':'ALL_SEED_PREDICTIONS_EXPORTED_LABEL_REVIEW_PENDING','training_started':False,'labels_complete':False})
if __name__=='__main__':
 try:main()
 except Exception:save('DIRECT_THREE_SEED_FAILED.json',{'traceback':traceback.format_exc()});raise
