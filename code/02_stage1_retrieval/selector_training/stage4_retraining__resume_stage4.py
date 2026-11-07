"""Resume only missing work, preserving frozen data and original failure evidence."""
import os, sys, json, time, subprocess, hashlib, shutil, fcntl, traceback
from pathlib import Path
BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[1]
STATE = BASE/'resume_status.json'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(2**20), b''): h.update(chunk)
    return h.hexdigest()

def save(state):
    state['updated_server_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    temp = STATE.with_suffix('.tmp')
    temp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(STATE)

def idle(gpu):
    result = subprocess.run(['nvidia-smi', '-i', str(gpu), '--query-gpu=memory.used,utilization.gpu', '--format=csv,noheader,nounits'], capture_output=True, text=True)
    if result.returncode: return False
    memory, utilization = map(int, result.stdout.strip().split(','))
    return memory < 1024 and utilization < 10

def launch(obj, seed, gpu):
    env = os.environ.copy()
    env.update(PYTHONPATH=str(ROOT), CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4',
               CUBLAS_WORKSPACE_CONFIG=':4096:8', TOKENIZERS_PARALLELISM='false',
               PYTHONUTF8='1', PYTHONIOENCODING='utf-8', LANG='C.UTF-8')
    script = 'finish_stage4_42.py' if seed == 42 else 'train_stage4.py'
    name = f'{obj}_{seed}'
    with (BASE/f'resume_{name}.log').open('a', encoding='utf-8') as log:
        process = subprocess.Popen([sys.executable, '-u', str(BASE/script), '--objective', obj, '--seed', str(seed)], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    print('START', name, gpu, process.pid, flush=True)
    return process

def main():
    lock = (BASE/'queue.lock').open('a', encoding='utf-8')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    immutable = {name: sha(BASE/name) for name in ['dataset.json', 'prepared.pt', 'split_manifest.json']}
    assert immutable['dataset.json'] == 'e117c403e9274f5265d6d9b2d8ae9c1de046741d1603995f63457b7e163b1fc2'
    pending = {0: [('support', n) for n in [42,2026,3407] if not (BASE/f'support_{n}/DONE.json').exists()],
               1: [('relevance', n) for n in [42,2026,3407] if not (BASE/f'relevance_{n}/DONE.json').exists()]}
    state = {'status':'RESUMING', 'launcher_pid':os.getpid(), 'immutable_hashes':immutable,
             'original_queue_status_preserved':True, 'completed':[], 'failed':[], 'running':[],
             'clinical_expert_gold':False}
    active = {}
    save(state)
    while any(pending.values()) or active:
        for gpu in [0,1]:
            if gpu not in active and pending[gpu] and not state['failed'] and idle(gpu):
                obj, seed = pending[gpu].pop(0)
                active[gpu] = (launch(obj,seed,gpu),f'{obj}_{seed}')
        for gpu,(process,name) in list(active.items()):
            code = process.poll()
            if code is not None:
                if code or not (BASE/name/'DONE.json').exists(): state['failed'].append({'run':name,'exit':code})
                else: state['completed'].append(name)
                del active[gpu]
        state['running'] = [{'gpu':g,'run':n,'pid':p.pid} for g,(p,n) in active.items()]
        state['pending'] = {str(g):[f'{o}_{s}' for o,s in jobs] for g,jobs in pending.items()}
        state['status'] = 'FAILED_DRAINING' if state['failed'] else 'RUNNING' if active else 'WAITING_FOR_EXTERNAL_GPU'
        save(state)
        if state['failed'] and not active: break
        time.sleep(20)
    assert immutable == {name:sha(BASE/name) for name in immutable}, 'Frozen data changed'
    env = os.environ.copy(); env.update(PYTHONUTF8='1', PYTHONIOENCODING='utf-8', PYTHONPATH=str(ROOT), OMP_NUM_THREADS='4')
    with (BASE/'resume_report.log').open('w',encoding='utf-8') as log:
        code = subprocess.call([sys.executable,str(BASE/'report_stage4.py')],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    if code: state['failed'].append({'run':'report','exit':code})
    state['status'] = 'FAILED' if state['failed'] else 'SIX_RUNS_AND_ACTUAL_PREDICTIONS_COMPLETE'
    state['frozen_data_hashes_verified'] = True
    save(state)

if __name__ == '__main__':
    try: main()
    except Exception:
        (BASE/'RESUME_FAILED.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise
