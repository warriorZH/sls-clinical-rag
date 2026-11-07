"""Finite six-run queue; never terminate external GPU jobs."""
import json,os,sys,time,subprocess,fcntl,traceback
from pathlib import Path
OUT=Path(__file__).resolve().parent;ROOT=OUT.parents[1]
def save(s):
 s['updated_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());p=OUT/'queue_status.tmp';p.write_text(json.dumps(s,indent=2));p.replace(OUT/'queue_status.json')
def run(script,args=(),gpu=None,name=None):
 env=os.environ.copy();env.update({'PYTHONPATH':str(ROOT),'OMP_NUM_THREADS':'4','CUBLAS_WORKSPACE_CONFIG':':4096:8','TOKENIZERS_PARALLELISM':'false'})
 if gpu is not None:env['CUDA_VISIBLE_DEVICES']=str(gpu)
 f=(OUT/((name or Path(script).stem)+'.log')).open('w')
 p=subprocess.Popen([sys.executable,'-u',str(OUT/script),*args],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT);f.close();return p
def idle(gpu):
 r=subprocess.run(['nvidia-smi','-i',str(gpu),'--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True)
 if r.returncode:return False
 mem,util=map(int,r.stdout.strip().split(','));return mem<1024 and util<10
def main():
 lock=(OUT/'queue.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not (OUT/'queue_status.json').exists()
 s={'status':'PREPARING','launcher_pid':os.getpid(),'completed':[],'failed':[],'running':[]};save(s)
 for script in ['clean_stage4.py','materialize_stage4.py','prepare_stage4.py']:
  p=run(script);code=p.wait()
  if code:s.update(status='PREPARATION_FAILED',failed=[{'script':script,'exit':code}]);save(s);return
 baseline=run('baselines_stage4.py');pending={0:[('support',seed) for seed in [42,2026,3407]],1:[('relevance',seed) for seed in [42,2026,3407]]};active={};smoke=set();last_report=0
 while any(pending.values()) or active:
  for gpu in [0,1]:
   if gpu not in active and pending[gpu] and not s['failed'] and idle(gpu):
    obj,seed=pending[gpu][0];issmoke=gpu not in smoke;name=('smoke_' if issmoke else '')+f'{obj}_{seed}';args=['--objective',obj,'--seed',str(seed)]+(['--smoke'] if issmoke else [])
    if not issmoke:pending[gpu].pop(0)
    p=run('train_stage4.py',args,gpu,name);active[gpu]=(p,name,issmoke);print('START',gpu,name,p.pid,flush=True)
  for gpu,(p,name,issmoke) in list(active.items()):
   code=p.poll()
   if code is not None:
    d=OUT/name/'DONE.json'
    if code or not d.exists():s['failed'].append({'run':name,'exit':code})
    elif issmoke:
     v=json.loads(d.read_text());assert v['finite_dev_predictions'] and v['backbone_max_delta']>0;smoke.add(gpu)
    else:s['completed'].append(name)
    del active[gpu]
  s['status']='FAILED_DRAINING' if s['failed'] else 'RUNNING' if active else 'WAITING_FOR_EXTERNAL_GPU'
  s['running']=[{'gpu':g,'run':n,'pid':p.pid} for g,(p,n,k) in active.items()];s['pending']={str(g):[f'{o}_{seed}' for o,seed in pp] for g,pp in pending.items()};save(s)
  if time.monotonic()-last_report>300:
   p=run('report_stage4.py');p.wait();last_report=time.monotonic()
  if s['failed'] and not active:break
  time.sleep(20)
 baseline.wait()
 if baseline.returncode:s['failed'].append({'run':'baselines','exit':baseline.returncode})
 p=run('report_stage4.py');p.wait()
 s['status']='FAILED' if s['failed'] or p.returncode else 'TRAINING_COMPLETE_RETRIEVAL_ADJUDICATION_PENDING';save(s)
if __name__=='__main__':
 try:main()
 except Exception:
  (OUT/'LAUNCH_FAILED.txt').write_text(traceback.format_exc());raise
