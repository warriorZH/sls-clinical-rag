# -*- coding: utf-8 -*-
"""eval_v2 公共函数：路径、文件读写、哈希、数据加载、模型获取、环境信息。

服务器上的目录结构（与 eval_v1 并列）：
    expert_testset_v1/
        _sealed_负责人保存_勿发给标注员/   final_questions.json、sealed_pools.jsonl（eval_v1 时已在服务器上）
        eval_v1/                           calib/、scores/、models.json（eval_v1 的结果，只读）
        eval_v2/                           本目录：代码、config/、data/、runs/（全部输出）、results/
本目录下的程序不读取任何证据支持标签（phase2_gold_labels.jsonl 不在服务器上）。
"""
import datetime as dt
import hashlib
import json
import os
import platform
import re
import socket
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TESTSET = HERE.parent
EVAL_V1 = TESTSET / 'eval_v1'
RUNS = HERE / 'runs'
DONE_DIR = RUNS / 'done'
CONFIG_DIR = HERE / 'config'
RESULTS = HERE / 'results'
K = 20


# ----------------------------------------------------------------------------- 基础

def now():
    return dt.datetime.now().astimezone().isoformat(timespec='seconds')


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(str(path), 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_text(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


def read_json(path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding='utf-8'))


def _json_default(o):
    """把 numpy 的数值、数组等转成普通 Python 类型，避免写 JSON 时出错。"""
    for attr in ('item', 'tolist'):
        if hasattr(o, attr):
            try:
                return getattr(o, attr)()
            except Exception:  # noqa: BLE001
                pass
    if isinstance(o, (set, tuple)):
        return list(o)
    if isinstance(o, Path):
        return str(o)
    return str(o)


def dumps(obj, **kw):
    return json.dumps(obj, ensure_ascii=False, default=_json_default, **kw)


def write_json(path, obj):
    """先写临时文件再改名，避免中断时留下半个文件。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(dumps(obj, indent=2), encoding='utf-8')
    os.replace(str(tmp), str(path))


def read_jsonl(path):
    path = Path(path)
    out = []
    if not path.exists():
        return out
    with path.open(encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # 进程被中断时最后一行可能不完整，跳过
                continue
    return out


def append_jsonl(path, rec, fh=None):
    line = dumps(rec) + '\n'
    if fh is not None:
        fh.write(line)
        fh.flush()
        return
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(line)
        f.flush()


def write_jsonl(path, recs):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        for r in recs:
            f.write(dumps(r) + '\n')
    os.replace(str(tmp), str(path))


class Log:
    def __init__(self, path=None):
        self.path = Path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, *msg):
        line = '[{}] {}'.format(dt.datetime.now().strftime('%m-%d %H:%M:%S'), ' '.join(str(m) for m in msg))
        print(line, flush=True)
        if self.path:
            with self.path.open('a', encoding='utf-8') as f:
                f.write(line + '\n')


def mark_done(job_id, info=None):
    """作业成功结束时写完成标记；流水线据此判断作业已完成。"""
    write_json(DONE_DIR / '{}.json'.format(job_id), dict(info or {}, job=job_id, finished=now()))


def is_done(job_id):
    return (DONE_DIR / '{}.json'.format(job_id)).exists()


# ----------------------------------------------------------------------------- 配置

def load_config():
    """读取 config/pipeline.json；环境变量 EVAL_V2_CONFIG 可指定另一份（测试用）。"""
    p = Path(os.environ.get('EVAL_V2_CONFIG') or (CONFIG_DIR / 'pipeline.json'))
    cfg = json.loads(p.read_text(encoding='utf-8'))
    cfg['_path'] = str(p)
    return cfg


def resolve(p):
    """相对路径以 expert_testset_v1 为基准；~ 展开为家目录。"""
    p = Path(os.path.expanduser(str(p)))
    return p if p.is_absolute() else (TESTSET / p)


def read_secrets():
    """读取 secrets/judge_keys.env（KEY=VALUE），环境变量优先。不打印任何密钥。"""
    keys = {}
    p = HERE / 'secrets' / 'judge_keys.env'
    if p.exists():
        for line in p.read_text(encoding='utf-8-sig').splitlines():
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            k, v = line.split('=', 1)
            keys[k.strip()] = v.strip().strip('"').strip("'")
    for k in list(keys) + ['DEEPSEEK_API_KEY', 'ARK_API_KEY', 'QIANFAN_API_KEY']:
        if os.environ.get(k):
            keys[k] = os.environ[k]
    return {k: v for k, v in keys.items() if v and not v.startswith('<') and v.lower() not in ('none', 'null')}


# ----------------------------------------------------------------------------- 数据

def evidence_text(c):
    """与 eval_v1 打分时完全相同的证据文本：context_text（缺失时用 text）。"""
    return c.get('context_text') or c.get('text') or ''


def load_target(cfg, target, check_hash=True):
    """返回 (questions, pools)。questions 为纳入分析的题目列表（按 qid 排序）；
    pools[qid] 为按检索名次排好的 20 条候选。只读题目和候选池，不读任何证据标签。"""
    t = cfg['targets'][target]
    qp, pp = resolve(t['questions']), resolve(t['pools'])
    expected = dict(t.get('sha256') or {})
    if t.get('manifest'):                      # 在服务器上生成的题集（外部验证）：哈希记录在生成时写出的清单里
        mp = resolve(t['manifest'])
        if not mp.exists():
            raise FileNotFoundError('找不到 {}'.format(mp))
        outs = (read_json(mp) or {}).get('outputs', {})
        expected.setdefault('questions', outs.get(qp.name))
        expected.setdefault('pools', outs.get(pp.name))
    for kind, p in (('questions', qp), ('pools', pp)):
        if not p.exists():
            raise FileNotFoundError('找不到 {}'.format(p))
        if check_hash and expected.get(kind):
            h = sha256_file(p)
            if h != expected[kind]:
                raise RuntimeError('{} 的 SHA256 为 {}，与冻结值 {} 不符'.format(p.name, h, expected[kind]))
    questions = [q for q in json.loads(qp.read_text(encoding='utf-8')) if q.get('included', True)]
    questions.sort(key=lambda q: q['qid'])
    qids = {q['qid'] for q in questions}
    pools = {}
    for r in read_jsonl(pp):
        if r['qid'] in qids:
            r = dict(r, document_id=str(r['document_id']), rank=int(r['rank']))
            pools.setdefault(r['qid'], []).append(r)
    for qid in pools:
        pools[qid].sort(key=lambda c: c['rank'])
    n_expected = t.get('n_questions')
    if n_expected and len(questions) != n_expected:
        raise RuntimeError('{} 纳入题数 {}，应为 {}'.format(target, len(questions), n_expected))
    for q in questions:
        ranks = [c['rank'] for c in pools.get(q['qid'], [])]
        if ranks != list(range(1, K + 1)):
            raise RuntimeError('{} 的候选名次不是 1..{}'.format(q['qid'], K))
    return questions, pools


# ----------------------------------------------------------------------------- 环境

def environment():
    env = {'python': sys.version.split()[0], 'platform': platform.platform(), 'host': socket.gethostname(),
           'HF_ENDPOINT': os.environ.get('HF_ENDPOINT', '')}
    try:
        import torch
        env['torch'] = torch.__version__
        env['cuda_available'] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            env['visible_gpus'] = ['{} ({:.1f} GB, cc {}.{})'.format(
                torch.cuda.get_device_name(i), torch.cuda.get_device_properties(i).total_memory / 2 ** 30,
                *torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())]
    except Exception as e:  # noqa: BLE001
        env['torch'] = '未安装（{}）'.format(type(e).__name__)
    try:
        import transformers
        env['transformers'] = transformers.__version__
    except Exception as e:  # noqa: BLE001
        env['transformers'] = '未安装（{}）'.format(type(e).__name__)
    env['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES')
    return env


def pick_device():
    try:
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    except Exception:  # noqa: BLE001
        return 'cpu'


def set_determinism(seed=42):
    import random
    import numpy as np
    import torch
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:  # noqa: BLE001
        pass


def from_pretrained(cls, path, **kw):
    """transformers 5.x 用 dtype=，旧版用 torch_dtype=；P40 上一律显式 fp32。"""
    import torch
    kw.setdefault('dtype', torch.float32)
    try:
        return cls.from_pretrained(str(path), **kw)
    except TypeError:
        kw['torch_dtype'] = kw.pop('dtype')
        return cls.from_pretrained(str(path), **kw)


# ----------------------------------------------------------------------------- 模型

WEIGHT_SUFFIXES = ('.safetensors', '.bin', '.pt', '.pth')


def weights_fingerprint(d):
    d = Path(d)
    files = sorted(x for x in d.iterdir() if x.is_file() and x.suffix in WEIGHT_SUFFIXES
                   and not any(s in x.name for s in ('optimizer', 'scheduler', 'rng_state', 'training_args')))
    return {x.name: sha256_file(x) for x in files}


def model_complete(d):
    d = Path(d)
    if not (d / 'config.json').exists():
        return False
    idx = list(d.glob('*.safetensors.index.json'))
    if idx:
        try:
            files = set(json.loads(idx[0].read_text(encoding='utf-8'))['weight_map'].values())
        except Exception:  # noqa: BLE001
            return False
        return all((d / f).exists() for f in files)
    return any(d.glob('*.safetensors')) or (d / 'pytorch_model.bin').exists()


def find_local_model(repo_ids, search_dirs, log=print):
    """在常见目录中查找已下载的模型（按 config.json 的 _name_or_path 或目录名匹配）。"""
    names = {r.split('/')[-1].lower() for r in repo_ids}
    full = {r.lower() for r in repo_ids}
    for root in search_dirs:
        root = Path(os.path.expanduser(str(root)))
        if not root.exists():
            continue
        for cfgf in list(root.glob('*/config.json')) + list(root.glob('*/*/config.json')) + \
                list(root.glob('models--*/snapshots/*/config.json')):
            d = cfgf.parent
            cand = {d.name.lower()}
            m = re.match(r'models--([^/]+)--([^/]+)$', d.parent.parent.name)
            if m:
                cand.add('{}/{}'.format(m.group(1), m.group(2)).lower())
                cand.add(m.group(2).lower())
            try:
                nop = json.loads(cfgf.read_text(encoding='utf-8')).get('_name_or_path') or ''
                cand.add(str(nop).lower())
                cand.add(str(nop).split('/')[-1].lower())
            except Exception:  # noqa: BLE001
                pass
            if (cand & names or cand & full) and model_complete(d):
                return d
    return None


def fetch_model(repo_ids, log=print, allow_download=True, search_dirs=None):
    """依次尝试候选名：先找本地已有的完整副本，再经 HF_ENDPOINT（hf-mirror）下载。
    返回 (本地目录, 实际仓库名, 版本号)。"""
    if isinstance(repo_ids, str):
        repo_ids = [repo_ids]
    for r in repo_ids:
        p = Path(os.path.expanduser(r))
        if p.exists() and model_complete(p):
            return p, r, None
    search_dirs = search_dirs or ['~/.cache/huggingface/hub', '/home/<user>/models', '~/models', '~/.cache/modelscope/hub']
    local = find_local_model(repo_ids, search_dirs, log)
    if local is not None:
        log('  使用本地已有副本：{}'.format(local))
        return local, repo_ids[0], local.name if local.parent.name == 'snapshots' else None
    if not allow_download:
        raise FileNotFoundError('本地找不到 {}'.format(repo_ids))
    from huggingface_hub import snapshot_download
    last = None
    allow = ['*.json', '*.safetensors', '*.model', '*.txt', '*.tiktoken', '*.jinja', 'tokenizer*', 'vocab*', 'merges*',
             '*.py']
    for r in repo_ids:
        for attempt in range(1, 6):
            try:
                log('  下载 {}（第 {} 次，HF_ENDPOINT={}）'.format(r, attempt, os.environ.get('HF_ENDPOINT', '默认')))
                d = Path(snapshot_download(repo_id=r, allow_patterns=allow))
                if not model_complete(d):
                    d = Path(snapshot_download(repo_id=r, allow_patterns=allow + ['*.bin']))
                if model_complete(d):
                    return d, r, d.name
                raise RuntimeError('下载后权重文件不完整')
            except Exception as e:  # noqa: BLE001
                last = e
                log('  失败：{}: {}'.format(type(e).__name__, str(e)[:300]))
                time.sleep(min(60, 10 * attempt))
    raise RuntimeError('无法获取模型 {}：{}'.format(repo_ids, last))


def model_registry():
    return read_json(RUNS / 'models' / 'registry.json', {}) or {}


def register_model(key, info):
    """几个下载作业可能同时结束：用文件锁保证读改写不丢失别的作业刚登记的内容。"""
    import fcntl
    (RUNS / 'models').mkdir(parents=True, exist_ok=True)
    with open(str(RUNS / 'models' / 'registry.lock'), 'w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            reg = model_registry()
            reg[key] = info
            write_json(RUNS / 'models' / 'registry.json', reg)
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)


def registered_path(key):
    info = model_registry().get(key)
    if not info:
        raise RuntimeError('模型 {} 尚未下载（应由 fetch 作业完成）'.format(key))
    return Path(info['path'])


def registration():
    """本机启动器上传的 OSF 登记确认。"""
    return read_json(HERE / 'REGISTRATION_v2.json')
