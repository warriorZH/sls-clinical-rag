#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""补充分析的公用部分：路径、配置、读写、OpenAI 兼容客户端（带重试与审计记录）。"""
import hashlib
import json
import os
import random
import re
import threading
import time
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CFG = json.loads((ROOT / 'config' / 'sup.json').read_text(encoding='utf-8'))
PROMPTS = json.loads((ROOT / 'config' / 'prompts_sup.json').read_text(encoding='utf-8'))
DATA = ROOT / 'data'
OFFICIAL = ROOT / 'official'
WORK = Path(os.environ.get('SUP_WORK') or (ROOT / 'work')).resolve()
RESULTS = WORK / 'results'
LOGS = WORK / 'logs'
for _d in (RESULTS, LOGS):
    _d.mkdir(parents=True, exist_ok=True)

_lock = threading.Lock()


def now():
    return time.strftime('%Y-%m-%dT%H:%M:%S%z')


def read_json(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def read_jsonl(p):
    p = Path(p)
    if not p.exists():
        return []
    out = []
    with p.open(encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass          # 进程被中断时最后一行可能不完整，忽略
    return out


def append_jsonl(p, rec):
    line = json.dumps(rec, ensure_ascii=False) + '\n'
    with _lock:
        with open(p, 'a', encoding='utf-8') as f:
            f.write(line)


def write_json(p, obj):
    tmp = Path(str(p) + '.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding='utf-8')
    os.replace(tmp, p)


def sha256_text(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


def log(msg, name='pipeline'):
    line = '[{}] {}'.format(time.strftime('%m-%d %H:%M:%S'), msg)
    print(line, flush=True)
    with _lock:
        with open(LOGS / '{}.log'.format(name), 'a', encoding='utf-8') as f:
            f.write(line + '\n')


def base_url(role):
    return 'http://127.0.0.1:{}/v1'.format(CFG['models'][role]['port'])


def make_client(role, timeout=900):
    from openai import OpenAI
    return OpenAI(api_key='EMPTY', base_url=base_url(role), timeout=timeout, max_retries=0)


# ----------------------------------------------------------------------------- 服务中断与卡死的处理
EXIT_SERVER_DOWN = 3          # 约定的退出码：服务中断或卡死，run_all.sh 会重启服务后从断点继续（不计入失败次数）
ABORT = threading.Event()


class ServerDown(RuntimeError):
    pass


def server_ports(roles):
    """当前运行中的服务端口（读 state/servers.json；评审可能有两个副本）。"""
    p = WORK / 'state' / 'servers.json'
    inst = {}
    if p.exists():
        try:
            inst = json.loads(p.read_text(encoding='utf-8')).get('instances') or {}
        except Exception:  # noqa: BLE001
            inst = {}
    out = []
    for role in roles:
        ports = [v['port'] for v in inst.values() if v.get('role') == role]
        out += ports or [CFG['models'][role]['port']]
    return out


def judge_ports():
    return server_ports(['judge'])


def port_healthy(port, timeout=10):
    import urllib.request
    try:
        with urllib.request.urlopen('http://127.0.0.1:{}/health'.format(port), timeout=timeout) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def servers_ok(roles, recheck=3):
    """全部服务健康返回 True；不健康时隔 20 秒复查，连续 recheck 次都不健康才判定为中断。"""
    for i in range(recheck):
        if all(port_healthy(p) for p in server_ports(roles)):
            return True
        if i < recheck - 1:
            time.sleep(20)
    return False


def check_abort_on_failure(roles, name):
    """某条任务失败后调用：若服务已中断，设置 ABORT（不计这次失败）并返回 True。"""
    if ABORT.is_set():
        return True
    if not servers_ok(roles):
        if not ABORT.is_set():
            ABORT.set()
            log('检测到 vLLM 服务中断（{}），停止派发新任务；本次失败不计入重试次数，稍后重启服务后从断点继续'.format(
                ','.join(roles)), name)
        return True
    return False


class Watchdog:
    """长时间没有任何任务完成（服务卡死）时，记录日志并以退出码 3 结束进程。"""

    def __init__(self, name, minutes=None):
        self.name = name
        self.limit = 60 * float(minutes if minutes is not None else CFG.get('serve', {}).get('stall_minutes', 30))
        self.last = time.time()
        self.done = False
        threading.Thread(target=self._run, daemon=True).start()

    def tick(self):
        self.last = time.time()

    def stop(self):
        self.done = True

    def _run(self):
        while not self.done:
            time.sleep(30)
            if not self.done and time.time() - self.last > self.limit:
                log('已有 {:.0f} 分钟没有任何任务完成，判定服务卡死；退出并由 run_all.sh 重启服务后续跑'.format(
                    self.limit / 60), self.name)
                os._exit(EXIT_SERVER_DOWN)


def call_retry(fn, attempts=6, what='请求'):
    """网络错误、服务器 5xx、超时时重试（指数退避）。其他错误直接抛出。服务已判定中断时立即放弃。"""
    last = None
    for i in range(attempts):
        if ABORT.is_set():
            raise ServerDown('服务已中断')
        try:
            return fn()
        except Exception as e:  # noqa: BLE001
            last = e
            name = type(e).__name__
            msg = str(e)
            transient = any(k in name for k in ('Connection', 'Timeout', 'InternalServer', 'APIError', 'ServiceUnavailable',
                                                 'RateLimit')) or any(k in msg for k in ('502', '503', '504', 'Connection'))
            if not transient or i == attempts - 1:
                raise
            time.sleep(min(60.0, 2 ** i) + random.random())
    raise last


THINK_RE = re.compile(r'<think>.*?</think>', re.S)


def clean_think(text):
    """与官方 APIModelInterface._clean_think_tags 相同的规则。"""
    if '<think>' in text and '</think>' in text:
        return THINK_RE.sub('', text).strip()
    if '</think>' in text and '<think>' not in text:
        return text[text.find('</think>') + len('</think>'):].strip()
    return text.strip()


def usage_of(r):
    u = getattr(r, 'usage', None)
    if not u:
        return None
    return {'prompt': getattr(u, 'prompt_tokens', None), 'completion': getattr(u, 'completion_tokens', None)}


class _AuditedCompletions:
    """包住真实的 chat.completions：可附加 extra_body，逐条记下返回内容（不改变请求与结果）。"""

    def __init__(self, real, tag, audit, extra_body=None):
        self.real, self.tag, self.audit, self.extra_body = real, tag, audit, extra_body

    def create(self, **kw):
        if self.extra_body:
            kw['extra_body'] = dict(kw.get('extra_body') or {}, **self.extra_body)
        t0 = time.time()
        try:
            r = call_retry(lambda: self.real.create(**kw), what=self.tag)
        except Exception as e:  # noqa: BLE001  记下后照常抛出（官方代码可能自行捕获并继续，这里要能发现）
            self.audit.append({'tag': self.tag, 'error': repr(e)[:300]})
            raise
        if kw.get('stream'):
            self.audit.append({'tag': self.tag, 'stream': True})
            return r
        ch = r.choices[0] if r.choices else None
        content = (ch.message.content if ch and ch.message else '') or ''
        reasoning = getattr(ch.message, 'reasoning_content', None) if ch and ch.message else None
        prompt_text = ''.join(str(m.get('content', '')) for m in kw.get('messages', []))
        self.audit.append({'tag': self.tag, 'finish_reason': getattr(ch, 'finish_reason', None), 'content': content,
                           'reasoning_chars': len(reasoning or ''), 'prompt_sha': sha256_text(prompt_text)[:16],
                           'usage': usage_of(r), 'sec': round(time.time() - t0, 2),
                           'json_found': bool(re.search(r'\{[\s\S]*\}', content))})
        return r


class AuditedClient:
    """供官方评测器使用的客户端替身：只有 chat.completions.create。"""

    def __init__(self, real_client, tag, audit, extra_body=None):
        self.chat = types.SimpleNamespace(completions=_AuditedCompletions(real_client.chat.completions, tag, audit, extra_body))
