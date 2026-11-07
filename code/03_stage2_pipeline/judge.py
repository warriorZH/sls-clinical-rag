#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""答案评审（方案 4.1–4.2）。默认两位评审都是在服务器 GPU 上运行的开源模型（不需要任何 API 密钥）：

    评审 A（主评审）：GLM-4-9B-0414，评全部答案；主指标用它的标签
    评审 B（第二评审）：tencent/Hunyuan-7B-Instruct（关闭思考模式），只评按题号哈希固定抽取的约 25% 题目
                      （封存集 45 题），用于一致性与敏感性分析
两者都是 fp32、贪心解码；输出到第一个完整 json 对象即停止。也保留 OpenAI 兼容 API 的写法（backend=api）。

    python judge.py --smoke --job e0_judge                        E0：依次加载两位评审，在校准集冒烟答案上试评
    python judge.py --judge A --sets main --job judge_main_A        正式评审（只评已冻结的答案文件）

任务：
  kpr         要点召回：A 评全部答案，B 评子集
  scr_split   断言拆分：只由 A 拆分一次，两位评审共用同一组断言
  scr_verify  断言核对：A 全部核对，B 核对子集
评审看到的只有题目、答案、参考要点（或证据），看不到条件名称。每个请求的结果按内容哈希缓存，断点续跑时不会重做。
"""
import argparse
import concurrent.futures as cf
import json
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

from common import (CONFIG_DIR, RUNS, Log, evidence_text, load_config, load_target, mark_done, model_registry, now,
                    read_json, read_jsonl, read_secrets, sha256_file, sha256_text, write_json, write_jsonl)
import conditions as C

LABELS_KPR = ('2', '1', '0', 'X')
ORDER = {'X': 0, '0': 1, '1': 2, '2': 3}
SPLITTER = 'A'                     # 断言只由评审 A 拆分一次
TASKS = ('kpr', 'scr_split', 'scr_verify')
HTTP_MAX_TOKENS = {'kpr': 2048, 'scr_split': 3000, 'scr_verify': 3000}


# ----------------------------------------------------------------------------- HTTP（可选的 API 评审）

class ApiError(Exception):
    def __init__(self, msg, code=None, retry=False):
        super().__init__(msg)
        self.code = code
        self.retry = retry


class Judge:
    def __init__(self, name, jcfg, key, chosen=None):
        self.name = name
        self.cfg = jcfg
        self.key = key
        chosen = chosen or {}
        self.model = chosen.get('model') or jcfg['models'][0]
        self.json_mode = chosen.get('json_mode', jcfg.get('json_mode', True))
        self.extra = chosen.get('extra', jcfg.get('extra', {}))
        self.url = jcfg['base_url'].rstrip('/') + '/chat/completions'
        self.timeout = int(jcfg.get('timeout', 180))
        self.usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'calls': 0}
        self.lock = threading.Lock()

    def post(self, messages, max_tokens=2048):
        body = {'model': self.model, 'messages': messages, 'temperature': 0, 'max_tokens': max_tokens}
        body.update(self.extra)
        if self.json_mode:
            body['response_format'] = {'type': 'json_object'}
        data = json.dumps(body, ensure_ascii=False).encode('utf-8')
        req = urllib.request.Request(self.url, data=data, method='POST',
                                     headers={'Content-Type': 'application/json; charset=utf-8',
                                              'Authorization': 'Bearer ' + self.key})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                text = resp.read().decode('utf-8', errors='replace')
        except urllib.error.HTTPError as e:
            detail = ''
            try:
                detail = e.read().decode('utf-8', errors='replace')[:500]
            except Exception:  # noqa: BLE001
                pass
            raise ApiError('HTTP {} {}'.format(e.code, detail), e.code, retry=(e.code == 429 or e.code >= 500))
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
            raise ApiError('网络错误 {}: {}'.format(type(e).__name__, e), None, retry=True)
        try:
            r = json.loads(text)
            content = r['choices'][0]['message'].get('content') or ''
        except Exception:  # noqa: BLE001
            raise ApiError('无法解析的响应：{}'.format(text[:300]), None, retry=True)
        u = r.get('usage') or {}
        with self.lock:
            self.usage['prompt_tokens'] += int(u.get('prompt_tokens') or 0)
            self.usage['completion_tokens'] += int(u.get('completion_tokens') or 0)
            self.usage['calls'] += 1
        if not content.strip():
            raise ApiError('空响应', None, retry=True)   # DeepSeek JSON 模式偶发空内容
        return content, r.get('model')

    def call(self, messages, max_tokens=2048, attempts=8):
        last = None
        for i in range(attempts):
            try:
                return self.post(messages, max_tokens)
            except ApiError as e:
                last = e
                if not e.retry:
                    raise
                time.sleep(min(60.0, 2 ** i) + random.random())
        raise last


def parse_json(text):
    t = text.strip()
    t = re.sub(r'^```(?:json)?\s*', '', t)
    t = re.sub(r'\s*```$', '', t)
    i, j = t.find('{'), t.rfind('}')
    if i < 0 or j <= i:
        raise ValueError('输出中没有 json 对象')
    return json.loads(t[i:j + 1])


def norm(s):
    return re.sub(r'[\s　]+', '', str(s or ''))


def norm_loose(s):
    return re.sub(r'[\s　，。、；：！？,.;:!?“”"‘’\'（）()【】\[\]《》<>…—\-]+', '', str(s or ''))


def quote_in(quote, text):
    q = norm(quote)
    if not q:
        return False
    return q in norm(text) or (norm_loose(quote) and norm_loose(quote) in norm_loose(text))


# ----------------------------------------------------------------------------- 三类任务：消息

def kpr_messages(P, q, answer, points, seed):
    pts = list(points)
    random.Random(seed).shuffle(pts)    # 要点顺序按固定种子打乱，避免位置偏差
    lines = '\n'.join(P['kpr']['point_line'].format(pid=p['pid'], type=p['type'], text=p['text']) for p in pts)
    user = P['kpr']['user'].format(question=q['question_final'], points=lines, answer=answer)
    return [{'role': 'system', 'content': P['kpr']['system']}, {'role': 'user', 'content': user}]


def check_kpr(obj, points, answer):
    items = obj.get('items')
    if not isinstance(items, list):
        return None, '缺少 items 列表'
    want = {p['pid'] for p in points}
    labels, quotes, qok = {}, {}, {}
    for it in items:
        if not isinstance(it, dict):
            continue
        pid = str(it.get('pid', '')).strip().upper()
        lab = str(it.get('label', '')).strip().upper()
        if pid in want and lab in LABELS_KPR:
            labels[pid] = lab
            quotes[pid] = str(it.get('quote') or '')
            qok[pid] = quote_in(quotes[pid], answer) if lab in ('2', '1') else True
    missing = sorted(want - set(labels))
    if missing:
        return None, '缺少要点 {}'.format(','.join(missing))
    bad = [p for p in labels if not qok[p]]
    res = {'labels': labels, 'quotes': quotes, 'quote_ok': qok, 'abstain': bool(obj.get('abstain', False))}
    return res, ('quote 不是答案原文：{}'.format(','.join(bad)) if bad else None)


def split_messages(P, q, answer):
    user = P['scr_split']['user'].format(question=q['question_final'], answer=answer)
    return [{'role': 'system', 'content': P['scr_split']['system']}, {'role': 'user', 'content': user}]


def verify_messages(P, q, evid_texts, claims):
    ev = '\n\n'.join(P['scr_verify']['evidence_item'].format(i=i + 1, text=t) for i, t in enumerate(evid_texts))
    cl = '\n'.join(P['scr_verify']['claim_line'].format(cid=c['cid'], text=c['text']) for c in claims)
    user = P['scr_verify']['user'].format(question=q['question_final'], evidence=ev, claims=cl)
    return [{'role': 'system', 'content': P['scr_verify']['system']}, {'role': 'user', 'content': user}], ev


# ----------------------------------------------------------------------------- 三类任务：提示与检查

def parse_problem(e):
    return '无法解析为 json（{}）'.format(type(e).__name__)


def kpr_spec(P, q, rec, points):
    msgs = kpr_messages(P, q, rec['answer'], points, rec['aid'])

    def check(content):
        try:
            return check_kpr(parse_json(content), points, rec['answer'])
        except Exception as e:  # noqa: BLE001
            return None, parse_problem(e)
    return msgs, check, P['kpr']['retry']


def split_spec(P, q, rec):
    msgs = split_messages(P, q, rec['answer'])

    def check(content):
        try:
            obj = parse_json(content)
        except Exception as e:  # noqa: BLE001
            return None, parse_problem(e)
        if not isinstance(obj.get('claims'), list):
            return None, '缺少 claims 列表'
        claims = []
        for c in obj['claims']:
            if not isinstance(c, dict):
                continue
            text = str(c.get('text') or '').strip()
            kind = str(c.get('kind') or '结论').strip()
            if text:
                claims.append({'cid': 'c{}'.format(len(claims) + 1), 'text': text,
                               'kind': kind if kind in ('题干复述', '声明', '结论') else '结论'})
        return {'claims': claims}, None
    return msgs, check, P['scr_split']['retry']


def verify_spec(P, q, evid_texts, claims):
    """只核对“结论”类断言；没有需要核对的断言时返回 None（直接记为有效、空标签）。"""
    claims = [c for c in claims if c['kind'] == '结论']
    if not claims:
        return None
    msgs, ev = verify_messages(P, q, evid_texts, claims)
    want = {c['cid'] for c in claims}

    def check(content):
        try:
            obj = parse_json(content)
        except Exception as e:  # noqa: BLE001
            return None, parse_problem(e)
        labels, qok = {}, {}
        for it in obj.get('items') or []:
            if not isinstance(it, dict):
                continue
            cid = str(it.get('cid', '')).strip()
            lab = str(it.get('label', '')).strip().upper()
            if cid in want and lab in ('S', 'C', 'N'):
                labels[cid] = lab
                qok[cid] = quote_in(it.get('quote'), ev) if lab in ('S', 'C') else True
        missing = sorted(want - set(labels))
        if missing:
            return None, '缺少断言 {}'.format(','.join(missing))
        bad = [c for c in labels if not qok[c]]
        return {'labels': labels, 'quote_ok': qok}, ('quote 不是证据原文：{}'.format(','.join(bad)) if bad else None)
    return msgs, check, P['scr_verify']['retry']


def make_spec(P, job):
    task, jn, c, r, q, ev, key = job[:7]
    if task == 'kpr':
        return kpr_spec(P, q, r, list(q['points']))
    if task == 'scr_split':
        return split_spec(P, q, r)
    return verify_spec(P, q, ev, job[7])


def outcome(best, problem, attempts, resp_model):
    out = {'valid': best is not None, 'attempts': attempts, 'resp_model': resp_model, 'problem': problem}
    if best:
        out.update(best)
    return out


def run_http(J, spec, task):
    """API 评审：同一请求最多 3 次；输出不合规时把问题告诉评审，请它重写。"""
    if spec is None:
        return {'valid': True, 'labels': {}, 'quote_ok': {}, 'attempts': 0, 'resp_model': None}
    msgs, check, retry = spec
    best, problem, resp_model, attempts = None, None, None, 0
    for _ in range(3):
        attempts += 1
        content, resp_model = J.call(msgs, max_tokens=HTTP_MAX_TOKENS[task])
        res, problem = check(content)
        if res is not None:
            best = res
            if problem is None:
                break
        msgs = msgs[:2] + [{'role': 'assistant', 'content': content},
                           {'role': 'user', 'content': retry.format(problem=problem)}]
    return outcome(best, problem, attempts, resp_model)


# ----------------------------------------------------------------------------- 本地评审（服务器 GPU）

def local_model_id(model_key):
    info = model_registry().get(model_key) or {}
    fp = sha256_text(json.dumps(info.get('weights_sha256') or {}, sort_keys=True))[:12]
    return 'local:{}@{}'.format(info.get('repo') or model_key, fp)


class LocalJudge:
    """在服务器 GPU 上运行的开源评审模型：fp32、贪心解码、按长度排序后分批生成；
    每条输出写到第一个完整 json 对象即停止。"""
    backend = 'local'

    def __init__(self, name, jcfg, log, model_id=None):
        from generate import Generator
        self.name, self.cfg, self.log = name, jcfg, log
        self.max_new = {k: int(v) for k, v in (jcfg.get('max_new') or {}).items()}
        self.user_prefix = jcfg.get('user_prefix') or ''      # 例如 Hunyuan 的 /no_think（关闭思考模式）
        for t in TASKS:
            self.max_new.setdefault(t, 1024)
        gcfg = dict(jcfg)
        gcfg['max_new_tokens'] = max(self.max_new.values())
        self.model = local_model_id(jcfg['model_key'])
        if model_id and model_id != self.model:
            raise RuntimeError('评审 {} 的模型与 E0 选定的不一致：{} ≠ {}'.format(name, self.model, model_id))
        log('加载评审 {}：{}'.format(name, self.model))
        if jcfg.get('test_generator'):      # 只用于没有 GPU 的流程测试：用模拟的生成器代替真实模型
            import importlib.util
            spec = importlib.util.spec_from_file_location('test_generator', jcfg['test_generator'])
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self.G = mod.Generator(gcfg, log)
        else:
            self.G = Generator(gcfg, log)
        self.usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'calls': 0, 'batches': 0, 'seconds': 0.0}

    def close(self):
        self.G.close()


def model_msgs(LJ, msgs):
    """送入模型前的消息：只在第一条用户消息前加上模型专用的控制前缀（如 /no_think），评审规程文本不变。"""
    if not LJ.user_prefix:
        return msgs
    out, done = [], False
    for m in msgs:
        if m['role'] == 'user' and not done:
            m = dict(m, content='{}{}'.format(LJ.user_prefix, m['content']))
            done = True
        out.append(m)
    return out


def run_local(LJ, jobs, P, store, log):
    """本地评审：所有请求按任务分组、按提示长度降序分批；不合规的输出把问题告诉评审后进入下一轮重写，
    最多 3 轮。每批结束就写入结果文件，中断后可续跑。"""
    states = []
    for job in jobs:
        spec = make_spec(P, job)
        st = {'job': job, 'task': job[0], 'attempts': 0, 'best': None, 'problem': None, 'resp_model': LJ.model}
        if spec is None:
            put_result(store, job, LJ.model, {'valid': True, 'labels': {}, 'quote_ok': {}, 'attempts': 0,
                                              'resp_model': None})
            continue
        st['msgs'], st['check'], st['retry'] = spec
        states.append(st)
    total = len(states)
    done = {'n': 0, 'invalid': 0}
    t_start = time.time()
    pending = states
    for rnd in range(1, 4):
        if not pending:
            break
        nxt = []
        for task in TASKS:
            sts = [s for s in pending if s['task'] == task]
            if not sts:
                continue
            # 先只记长度（不保留上万条词元序列），排序分批后再逐批编码
            items = sorted(({'n': len(LJ.G.encode(model_msgs(LJ, s['msgs']))), 's': s} for s in sts),
                           key=lambda x: (-x['n'], x['s']['job'][6]))
            batches = LJ.G.plan_batches(items, max_new=LJ.max_new[task])
            log('评审 {}：{}，第 {} 轮，{} 个请求，{} 批'.format(LJ.name, task, rnd, len(items), len(batches)))
            for bi, b in enumerate(batches, 1):
                for it in b:
                    it['ids'] = LJ.G.encode(model_msgs(LJ, it['s']['msgs']))
                LJ.G.release()
                t0 = time.time()
                outs, _ = LJ.G.generate(b, max_new=LJ.max_new[task], stop_json=True)
                dt = time.time() - t0
                LJ.usage['batches'] += 1
                LJ.usage['seconds'] += dt
                for it, out in zip(b, outs):
                    s = it['s']
                    s['attempts'] += 1
                    LJ.usage['calls'] += 1
                    LJ.usage['prompt_tokens'] += len(it['ids'])
                    LJ.usage['completion_tokens'] += out['n_out']
                    res, problem = s['check'](out['answer'])
                    if res is not None:
                        s['best'] = res
                    s['problem'] = problem
                    if (res is not None and problem is None) or s['attempts'] >= 3:
                        put_result(store, s['job'], LJ.model, outcome(s['best'], problem, s['attempts'], LJ.model))
                        done['n'] += 1
                        done['invalid'] += 0 if s['best'] is not None else 1
                    else:
                        s['msgs'] = s['msgs'][:2] + [{'role': 'assistant', 'content': out['answer']},
                                                     {'role': 'user', 'content': s['retry'].format(problem=problem)}]
                        nxt.append(s)
                el = time.time() - t_start
                rate = done['n'] / el if el > 0 and done['n'] else None
                eta = ''
                if rate:
                    eta = '，预计还需 {:.1f} 小时'.format((total - done['n']) / rate / 3600)
                log('  评审 {} {} 第 {}/{} 批：{} 条，最长输入 {}，{:.0f} 秒；已完成 {}/{}（无效 {}）{}'.format(
                    LJ.name, task, bi, len(batches), len(b), max(len(x['ids']) for x in b), dt, done['n'], total,
                    done['invalid'], eta))
                for it in b:
                    it.pop('ids', None)
        pending = nxt
    return done


def put_result(store, job, model_id, res):
    task, jn, c, r, q, ev, key = job[:7]
    rec = {'key': key, 'task': task, 'judge': jn, 'model': model_id, 'aid': r['aid'], 'qid': r['qid'],
           'cond': c['id'], 'answer_sha': sha256_text(r['answer']), 'time': now()}
    rec.update(res)
    if task == 'scr_verify':
        rec['claims'] = job[7]
    store(task, jn, c['target']).put(rec)


# ----------------------------------------------------------------------------- 选择 API 模型（只用于 backend=api）

def probe_judge(name, jcfg, key, P, log):
    """依次尝试候选模型；参数不被接受时去掉 response_format 或 extra 再试。返回可用设置。"""
    msgs = [{'role': 'system', 'content': P['probe']['system']}, {'role': 'user', 'content': P['probe']['user']}]
    errors = []
    for model in jcfg['models']:
        variants = [(jcfg.get('json_mode', True), jcfg.get('extra', {})), (False, jcfg.get('extra', {})),
                    (jcfg.get('json_mode', True), {}), (False, {})]
        seen = set()
        for json_mode, extra in variants:
            sig = (json_mode, json.dumps(extra, sort_keys=True))
            if sig in seen:
                continue
            seen.add(sig)
            J = Judge(name, jcfg, key, {'model': model, 'json_mode': json_mode, 'extra': extra})
            try:
                content, resp_model = J.call(msgs, max_tokens=50, attempts=3)
                parse_json(content)
                log('  评审 {}：模型 {} 可用（json_mode={}，extra={}，响应模型 {}）'.format(
                    name, model, json_mode, json.dumps(extra, ensure_ascii=False), resp_model))
                return {'model': model, 'json_mode': json_mode, 'extra': extra, 'resp_model': resp_model}
            except Exception as e:  # noqa: BLE001
                errors.append('{} json={} extra={}: {}'.format(model, json_mode, bool(extra), str(e)[:200]))
                code = getattr(e, 'code', None)
                if code in (401, 403):
                    log('  评审 {}：密钥无效或无权限（HTTP {}）'.format(name, code))
                    return {'error': 'auth', 'detail': errors[-1]}
    log('  评审 {}：全部候选模型都不可用'.format(name))
    return {'error': 'unavailable', 'detail': errors[-6:]}


# ----------------------------------------------------------------------------- 结果文件

class Store:
    """追加写入的结果文件；按 key 去重，支持断点续跑。"""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.data = {r['key']: r for r in read_jsonl(self.path) if r.get('valid')}
        self.lock = threading.Lock()

    def has(self, key):
        return key in self.data

    def put(self, rec):
        with self.lock:
            with self.path.open('a', encoding='utf-8') as f:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
            if rec.get('valid'):
                self.data[rec['key']] = rec


def load_answers(cfg, sets):
    """只收集已冻结（有 .done.json）的答案文件。"""
    out = []
    allc = C.conditions(cfg)
    conds = [c for s in sets for c in allc[s]] if sets != ['smoke'] else [c for c in C.smoke_conditions() if c['id'] in ('S_G00', 'S_G01')]
    for c in conds:
        p = RUNS / 'gen' / c['target'] / c['gen'] / '{}.jsonl'.format(c['id'])
        if not p.with_suffix('.done.json').exists():
            continue
        for r in read_jsonl(p):
            out.append((c, r))
    return out


# ----------------------------------------------------------------------------- 评审一组答案

def subset_mod(cfg, jn):
    return int(cfg['judges'].get(jn, {}).get('subset_mod', 1))


def in_subset(cfg, jn, r):
    """按题号哈希固定抽取题目（评审 B：约 25% 的题目，封存集 45 题），评这些题在所有条件下的答案，
    这样同一道题在不同条件之间仍可配对比较。"""
    m = subset_mod(cfg, jn)
    return m <= 1 or int(sha256_text('{}:{}'.format(jn, r['qid'])), 16) % m == 0


def judge_set(cfg, engines, ids, P, P_sha, log, answers, target_cache, smoke=False):
    """engines：本作业要运行的评审（已加载）；ids：全部评审的模型标识（用于查缓存与合成标签）。"""
    stores = {}

    def store(task, jn, target):
        k = (task, jn, target)
        if k not in stores:
            stores[k] = Store(RUNS / 'judge' / target / '{}_{}.jsonl'.format(task, jn))
        return stores[k]

    def ctx(c, r):
        if c['target'] not in target_cache:
            target_cache[c['target']] = load_target(cfg, c['target'])
        questions, pools = target_cache[c['target']]
        qmap = {q['qid']: q for q in questions}
        q = qmap[r['qid']]
        bydoc = {x['document_id']: x for x in pools[r['qid']]}
        ev = [evidence_text(bydoc[d]) for d in r.get('docs') or []]
        return q, ev

    content_key = lambda *parts: sha256_text('|'.join(str(x) for x in parts))  # noqa: E731
    # 1. 要点召回 + 断言拆分
    jobs = []
    for c, r in answers:
        q, ev = ctx(c, r)
        for jn in engines:
            if smoke or in_subset(cfg, jn, r):
                key = content_key('kpr', ids[jn], P_sha, r['aid'], r['answer'])
                if not store('kpr', jn, c['target']).has(key):
                    jobs.append(('kpr', jn, c, r, q, ev, key))
        if SPLITTER in engines and c.get('scr') and r.get('docs'):
            key = content_key('scr_split', ids[SPLITTER], P_sha, r['aid'], r['answer'])
            if not store('scr_split', SPLITTER, c['target']).has(key):
                jobs.append(('scr_split', SPLITTER, c, r, q, ev, key))
    random.Random(20261026).shuffle(jobs)
    log('待评审请求 {} 个（已有结果的跳过）'.format(len(jobs)))
    run_jobs(jobs, engines, P, store, log)

    # 2. 断言核对（拆分完成之后；拆分来自评审 A）
    jobs = []
    split_index = {}
    for c, r in answers:
        if not (c.get('scr') and r.get('docs')):
            continue
        if c['target'] not in split_index:
            split_index[c['target']] = {(x['aid'], x.get('answer_sha')): x
                                        for x in store('scr_split', SPLITTER, c['target']).data.values()
                                        if x.get('model') == ids.get(SPLITTER)}
        rec = split_index[c['target']].get((r['aid'], sha256_text(r['answer'])))
        if not rec:
            continue
        q, ev = ctx(c, r)
        claims = rec['claims']
        for jn in engines:
            if smoke or in_subset(cfg, jn, r):
                key = content_key('scr_verify', ids[jn], P_sha, r['aid'], json.dumps(claims, ensure_ascii=False))
                if not store('scr_verify', jn, c['target']).has(key):
                    jobs.append(('scr_verify', jn, c, r, q, ev, key, claims))
    random.Random(20261027).shuffle(jobs)
    log('待核对断言请求 {} 个'.format(len(jobs)))
    run_jobs(jobs, engines, P, store, log)
    return store


def run_jobs(jobs, engines, P, store, log):
    if not jobs:
        return
    by_judge = {}
    for j in jobs:
        by_judge.setdefault(j[1], []).append(j)
    for jn, js in by_judge.items():
        J = engines[jn]
        if getattr(J, 'backend', 'api') == 'local':
            d = run_local(J, js, P, store, log)
            log('评审 {}：完成 {} 个请求（无效 {}）'.format(jn, d['n'], d['invalid']))
            continue
        done = {'n': 0, 'err': 0}
        lock = threading.Lock()
        total = len(js)
        t0 = time.time()

        def work(j, J=J):
            try:
                res = run_http(J, make_spec(P, j), j[0])
            except Exception as e:  # noqa: BLE001
                res = {'valid': False, 'error': '{}: {}'.format(type(e).__name__, str(e)[:300])}
            put_result(store, j, J.model, res)
            with lock:
                done['n'] += 1
                done['err'] += 0 if res.get('valid') else 1
                if done['n'] % 200 == 0 or done['n'] == total:
                    log('  评审 {} 进度 {}/{}，失败 {}，{:.0f} 秒'.format(jn, done['n'], total, done['err'],
                                                                   time.time() - t0))
        with cf.ThreadPoolExecutor(max_workers=int(J.cfg.get('workers', 4))) as ex:
            for f in cf.as_completed([ex.submit(work, j) for j in js]):
                f.result()


def kpr_get(store, jn, c, r, ids, P_sha):
    if jn not in ids:
        return None
    key = sha256_text('|'.join(str(x) for x in ('kpr', ids[jn], P_sha, r['aid'], r['answer'])))
    return store('kpr', jn, c['target']).data.get(key)


def effective(rec):
    """引用核对失败的“命中/部分”按“未提及”计（预先规定的保守规则）。"""
    out = {}
    for pid, lab in rec['labels'].items():
        if lab in ('2', '1') and not rec.get('quote_ok', {}).get(pid, True):
            lab = '0'
        out[pid] = lab
    return out


def finalize(cfg, answers, ids, P_sha, store, log):
    """最终标签 = 评审 A 的标签（引用核对失败的“命中/部分”按“未提及”计）。
    评审 B 只评固定子集，它的标签一并保存（labels_B），用于一致性与敏感性分析，不进入主指标。"""
    out = {}
    stats = {'n': 0, 'missing_A': 0, 'with_B': 0, 'subset_B': 0, 'missing_B': 0, 'points_B': 0, 'agree_points_B': 0}
    for c, r in answers:
        la = kpr_get(store, 'A', c, r, ids, P_sha)
        if 'B' in ids and in_subset(cfg, 'B', r):
            stats['subset_B'] += 1
        if not la:
            stats['missing_A'] += 1
            continue
        lb = kpr_get(store, 'B', c, r, ids, P_sha)
        ea = effective(la)
        eb = effective(lb) if lb else None
        if 'B' in ids and in_subset(cfg, 'B', r) and not lb:
            stats['missing_B'] += 1
        if eb:
            stats['with_B'] += 1
            for pid, lab in ea.items():
                if pid in eb:
                    stats['points_B'] += 1
                    stats['agree_points_B'] += int(eb[pid] == lab)
        stats['n'] += 1
        out.setdefault(c['target'], []).append({
            'aid': r['aid'], 'qid': r['qid'], 'cond': c['id'], 'group': c['group'], 'set': c['set'],
            'gen': c['gen'], 'labels': ea, 'labels_A': ea, 'labels_B': eb, 'abstain': bool(la.get('abstain')),
            'abstain_A': la.get('abstain'), 'abstain_B': lb.get('abstain') if lb else None,
            'judge_A': ids.get('A'), 'judge_B': ids.get('B') if lb else None})
    for target, recs in out.items():
        path = RUNS / 'judge' / target / 'kpr_final.jsonl'
        old = {x['aid']: x for x in read_jsonl(path)}
        for x in recs:
            old[x['aid']] = x
        write_jsonl(path, sorted(old.values(), key=lambda x: (x['cond'], x['qid'])))
    log('最终标签：{}'.format(json.dumps(stats, ensure_ascii=False)))
    return stats


# ----------------------------------------------------------------------------- 选择与加载评审

def load_engine(jn, jc, chosen, secrets, log):
    if jc.get('backend', 'api') == 'local':
        return LocalJudge(jn, jc, log, model_id=(chosen or {}).get('model'))
    key = secrets.get(jc['key_env'])
    if not key:
        raise RuntimeError('缺少评审 {} 的密钥 {}'.format(jn, jc['key_env']))
    return Judge(jn, jc, key, chosen)


def smoke_validity(jn, model_id):
    """冒烟答案上各任务的有效率（只统计当前模型的结果；同一请求以最后一次记录为准）。"""
    valid = {}
    for task in TASKS:
        rows = {}
        for x in read_jsonl(RUNS / 'judge' / 'calib' / '{}_{}.jsonl'.format(task, jn)):
            if x.get('model') == model_id:
                rows[x['key']] = x
        if rows:
            valid['{}_{}'.format(task, jn)] = {'n': len(rows), 'valid': sum(1 for x in rows.values() if x.get('valid'))}
    return valid


def run_smoke(a, cfg, P, P_sha, secrets, log):
    """E0：依次加载评审 A、B，在校准集冒烟答案上试评；记录选定的模型与设置。
    评审 A 必须可用且要点评审的有效率 ≥ 90%；评审 B 不可用时只记录原因，不影响主实验。"""
    answers = load_answers(cfg, ['smoke'])
    if not answers:
        log('没有找到 E0 的冒烟答案')
        sys.exit(1)
    chosen, usage, ids_all = {}, {}, {}
    for jn in ('A', 'B'):
        jc = cfg['judges'].get(jn)
        if not jc:
            continue
        t0 = time.time()
        if jc.get('backend', 'api') == 'local':
            # 每位本地评审在单独的进程里加载并试评：进程异常退出（例如 10 月 28 日加载评审 B 时的 SIGBUS）
            # 只算这一位评审不可用，不会连带已经通过的评审 A
            res = smoke_one_subprocess(a, jn, ids_all, log)
            if res.get('error'):
                chosen[jn] = {'error': res['error'], 'model_key': jc['model_key']}
                log('评审 {} 不可用：{}'.format(jn, res['error']))
                continue
            ch, usage[jn] = res['chosen'], res['usage']
            model_id = ch['model']
        else:
            try:
                key = secrets.get(jc['key_env'])
                if not key:
                    raise RuntimeError('没有密钥 {}'.format(jc['key_env']))
                ch = probe_judge(jn, jc, key, P, log)
                if ch.get('error'):
                    raise RuntimeError(json.dumps(ch, ensure_ascii=False)[:300])
                ch = dict(ch, backend='api')
                J = Judge(jn, jc, key, ch)
            except Exception as e:  # noqa: BLE001
                chosen[jn] = {'error': '{}: {}'.format(type(e).__name__, str(e)[:300])}
                log('评审 {} 不可用：{}'.format(jn, chosen[jn]['error']))
                continue
            ids = dict(ids_all, **{jn: J.model})          # 评审 B 核对断言时要用评审 A 的拆分结果
            judge_set(cfg, {jn: J}, ids, P, P_sha, log, answers, {}, smoke=True)
            usage[jn] = dict(J.usage, load_and_judge_seconds=round(time.time() - t0, 1))
            model_id = J.model
        v = smoke_validity(jn, model_id)
        kv = v.get('kpr_{}'.format(jn), {'n': 0, 'valid': 0})
        ch['smoke_validity'] = v
        if kv['n'] == 0 or kv['valid'] < 0.9 * kv['n']:
            ch = {'error': 'smoke_invalid', 'detail': v, 'model': ch.get('model')}
            log('评审 {} 的冒烟输出有效率不足 90%：{}'.format(jn, json.dumps(v, ensure_ascii=False)))
        else:
            ids_all[jn] = model_id
            log('评审 {} 可用：{}'.format(jn, json.dumps(v, ensure_ascii=False)))
        chosen[jn] = ch
    write_json(RUNS / 'e0' / 'judge_models.json', dict(chosen, chosen_at=now()))
    ids = dict(ids_all)
    stats = finalize(cfg, answers, ids, P_sha, judge_stores(), log) if 'A' in ids else None
    rep = {'passed': 'A' in ids, 'judges': chosen, 'usage': usage, 'kpr_stats': stats, 'time': now()}
    write_json(RUNS / 'e0' / 'judge_smoke.json', rep)
    log('E0 评审冒烟：{}'.format('通过' if rep['passed'] else '未通过'))
    if 'B' not in ids:
        log('提醒：评审 B 不可用，后续只用评审 A；一致性分析将缺失。')
    if not rep['passed']:
        sys.exit(1)
    mark_done(a.job, {'judges': {jn: ch.get('model') for jn, ch in chosen.items() if isinstance(ch, dict)}})


def smoke_one_subprocess(a, jn, ids_all, log):
    """在子进程里加载评审 jn 并试评冒烟答案。返回 {'chosen', 'usage'}，或 {'error'}。"""
    import signal
    import subprocess
    out = RUNS / 'e0' / 'smoke_{}.json'.format(jn)
    try:
        out.unlink()
    except FileNotFoundError:
        pass
    cmd = [sys.executable, str(Path(__file__).resolve()), '--smoke-one', jn, '--ids', json.dumps(ids_all),
           '--job', a.job]
    log('评审 {}：在单独的进程中加载并试评'.format(jn))
    rc = subprocess.run(cmd).returncode
    res = read_json(out) if out.exists() else None
    if rc == 0 and res and (res.get('chosen') or res.get('error')):
        return res
    if res and res.get('error'):
        return res
    if rc < 0:
        try:
            why = '信号 {}'.format(signal.Signals(-rc).name)
        except ValueError:
            why = '信号 {}'.format(-rc)
    else:
        why = '退出码 {}'.format(rc)
    return {'error': '加载或试评时进程异常退出（{}），调用栈见本日志'.format(why)}


def run_smoke_one(a, cfg, P, P_sha, log):
    """子进程：加载一位本地评审，在冒烟答案上试评，把结果写到 runs/e0/smoke_<评审>.json。"""
    jn = a.smoke_one
    out = RUNS / 'e0' / 'smoke_{}.json'.format(jn)
    t0 = time.time()
    try:
        jc = cfg['judges'][jn]
        answers = load_answers(cfg, ['smoke'])
        J = LocalJudge(jn, jc, log)
        ch = {'backend': 'local', 'model': J.model, 'model_key': jc['model_key'], 'settings': J.G.settings()}
        ids = dict(json.loads(a.ids or '{}'), **{jn: J.model})
        judge_set(cfg, {jn: J}, ids, P, P_sha, log, answers, {}, smoke=True)
        usage = dict(J.usage, load_and_judge_seconds=round(time.time() - t0, 1))
        if hasattr(J, 'close'):
            J.close()
        write_json(out, {'chosen': ch, 'usage': usage})
    except Exception as e:  # noqa: BLE001
        import traceback
        log(traceback.format_exc())
        write_json(out, {'error': '{}: {}'.format(type(e).__name__, str(e)[:300])})


def release_gpu():
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def judge_stores():
    stores = {}

    def store(task, jn, target):
        k = (task, jn, target)
        if k not in stores:
            stores[k] = Store(RUNS / 'judge' / target / '{}_{}.jsonl'.format(task, jn))
        return stores[k]
    return store


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sets', default='')
    ap.add_argument('--judge', default='A', help='本作业运行的评审：A 或 B')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--smoke-one', dest='smoke_one', default='', help='（E0 内部使用）在本进程中试评一位本地评审')
    ap.add_argument('--ids', default='', help='（E0 内部使用）已通过的评审模型')
    ap.add_argument('--job', required=True)
    a = ap.parse_args()
    cfg = load_config()
    log = Log(RUNS / 'logs' / 'judge_{}.log'.format(a.job))
    P_path = CONFIG_DIR / 'judge_prompts.json'
    P = json.loads(P_path.read_text(encoding='utf-8'))
    P_sha = sha256_file(P_path)
    secrets = read_secrets()
    log('=' * 60)
    log('评审作业 {}'.format(a.job))
    if a.smoke_one:
        run_smoke_one(a, cfg, P, P_sha, log)
        return
    if a.smoke:
        run_smoke(a, cfg, P, P_sha, secrets, log)
        return
    chosen = read_json(RUNS / 'e0' / 'judge_models.json')
    if not chosen:
        log('E0 尚未选定评审模型')
        sys.exit(3)
    ids = {jn: ch['model'] for jn, ch in chosen.items() if isinstance(ch, dict) and ch.get('model')
           and not ch.get('error')}
    jn = a.judge
    if jn not in ids:
        if jn != 'A':
            log('评审 {} 在 E0 中不可用（{}），本作业跳过。'.format(jn, (chosen.get(jn) or {}).get('error')))
            mark_done(a.job, {'skipped': 'judge {} unavailable in E0'.format(jn)})
            return
        log('评审 A 在 E0 中不可用')
        sys.exit(3)
    sets = [s for s in a.sets.split(',') if s]
    answers = load_answers(cfg, sets)
    log('评审 {}，集合 {}：答案 {} 条'.format(jn, ','.join(sets), len(answers)))
    try:
        engine = load_engine(jn, cfg['judges'][jn], chosen.get(jn), secrets, log)
    except RuntimeError as e:
        log(str(e))
        sys.exit(3)
    t0 = time.time()
    store = judge_set(cfg, {jn: engine}, ids, P, P_sha, log, answers, {})
    stats = finalize(cfg, answers, ids, P_sha, store, log)
    usage = dict(engine.usage, seconds_total=round(time.time() - t0, 1))
    log('用量：{}'.format(json.dumps(usage, ensure_ascii=False)))
    if hasattr(engine, 'close'):
        engine.close()
    if jn == 'A' and stats['missing_A'] > 0.02 * max(1, len(answers)):
        log('有 {} 个答案缺少评审 A 的有效标签（超过 2%），本作业记为失败，可重新启动续评。'.format(stats['missing_A']))
        sys.exit(1)
    if jn == 'B' and stats['missing_B'] > 0.02 * max(1, stats['subset_B']) + 1:
        log('评审 B 子集中有 {} 个答案缺少有效标签（超过 2%），本作业记为失败。'.format(stats['missing_B']))
        sys.exit(1)
    mark_done(a.job, {'judge': jn, 'answers': len(answers), 'kpr_stats': stats, 'usage': usage})


if __name__ == '__main__':
    main()
