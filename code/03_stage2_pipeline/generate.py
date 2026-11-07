#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""问答生成（E0 冒烟、E3、E4、E5）。

    python generate.py --conds G00,G01,... --job e3_gen_1        生成指定条件
    python generate.py --smoke --job e0_gen                       E0：校准集上跑两遍，检查可复现性、显存与速度

固定设置（见方案附录 A 与 2.3 节）：fp32；贪心解码（do_sample=False，temperature/top_p/top_k 置空）；
max_new_tokens 见 config；按提示长度排序后按固定的词元预算分批（与所在 GPU 无关，保证两张卡结果一致）。
每条答案写一行，断点续跑时跳过已完成的（条件, 题号）。
"""
import argparse
import copy
import gc as pygc
import json
import sys
import time

from common import (CONFIG_DIR, RUNS, Log, environment, evidence_text, from_pretrained, load_config, load_target,
                    mark_done, now, read_json, read_jsonl, registered_path, set_determinism, sha256_file,
                    sha256_text, write_json, write_jsonl)
import conditions as C
import p40
from select_evidence import ensure_selection


# ----------------------------------------------------------------------------- 提示

def load_prompts():
    p = CONFIG_DIR / 'prompts.json'
    return json.loads(p.read_text(encoding='utf-8')), sha256_file(p)


def build_messages(prompts, cond, question, evid_texts):
    if cond['k'] == 0 or not evid_texts:
        t = prompts['{}_closed_book'.format(cond['prompt'])]
        user = t['user'].format(question=question)
    else:
        t = prompts['{}_evidence'.format(cond['prompt'])]
        block = '\n\n'.join(t['evidence_item'].format(i=i + 1, text=x) for i, x in enumerate(evid_texts))
        user = t['user'].format(question=question, evidence=block)
    return [{'role': 'system', 'content': t['system']}, {'role': 'user', 'content': user}]


def answer_id(cond_id, qid, gen):
    return sha256_text('{}|{}|{}'.format(cond_id, qid, gen))[:16]


# ----------------------------------------------------------------------------- 质量标记

def repeated(text):
    """粗略的重复检测：长度 10/20/40 的片段出现 4 次及以上。"""
    for n in (10, 20, 40):
        if len(text) < n * 4:
            continue
        seen = {}
        for i in range(0, len(text) - n + 1):
            s = text[i:i + n]
            seen[s] = seen.get(s, 0) + 1
            if seen[s] >= 4 and s.strip():
                return True
    return False


# ----------------------------------------------------------------------------- 评审输出：json 完整即停

def json_stopper(tok, start, n):
    """评审只需要输出中的第一个完整 json 对象：每条序列在最外层 {…} 闭合后就停止生成。
    逐个新词元更新括号深度，忽略字符串内的括号；对生成内容没有任何改动，只是不再继续往后写。"""
    import torch
    from transformers import StoppingCriteria

    class JsonComplete(StoppingCriteria):
        def __init__(self):
            self.seen = start
            self.st = [{'depth': 0, 'started': False, 'in_str': False, 'esc': False, 'done': False}
                       for _ in range(n)]

        def __call__(self, input_ids, scores, **kwargs):
            new = input_ids[:, self.seen:].tolist()
            self.seen = input_ids.shape[1]
            for i, toks in enumerate(new):
                s = self.st[i]
                for t in toks:
                    if s['done']:
                        break
                    for ch in tok.decode([t], skip_special_tokens=False):
                        if not s['started']:
                            if ch == '{':
                                s['started'], s['depth'] = True, 1
                            continue
                        if s['in_str']:
                            if s['esc']:
                                s['esc'] = False
                            elif ch == '\\':
                                s['esc'] = True
                            elif ch == '"':
                                s['in_str'] = False
                            continue
                        if ch == '"':
                            s['in_str'] = True
                        elif ch == '{':
                            s['depth'] += 1
                        elif ch == '}':
                            s['depth'] -= 1
                            if s['depth'] == 0:
                                s['done'] = True
                                break
            return torch.tensor([s['done'] for s in self.st], dtype=torch.bool, device=input_ids.device)

    return JsonComplete()


# ----------------------------------------------------------------------------- 生成器（问答模型与本地评审共用）

class Generator:
    def __init__(self, gcfg, log):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.log = log
        reg = read_json(RUNS / 'models' / 'registry.json', {}).get(gcfg['model_key'], {})
        if reg.get('fallback') and gcfg.get('fallback_settings'):
            gcfg = dict(gcfg, **gcfg['fallback_settings'])  # 用了单卡备选模型：换成对应的批量设置
            log('  使用单卡备选模型 {}，批量设置改为 {}'.format(reg.get('repo'), gcfg['fallback_settings']))
        self.cfg = gcfg
        mode = gcfg.get('grouped_decode', 'auto')          # auto：只在 P40 一类没有 FlashAttention 的 GPU 上启用
        self.sdpa_patch = p40.sdpa_without_gqa(torch, log, force=(mode == 'on'))
        self.memory_cap_gib = p40.equalize_memory(torch, log)
        self.prefill_chunk = int(gcfg['prefill_chunk']) if gcfg.get('prefill_chunk') else None
        self.prefill_chunk_on_oom = int(gcfg.get('prefill_chunk_on_oom', 1024))
        self.chat_kwargs = dict(gcfg.get('chat_template_kwargs') or {})
        self.path = registered_path(gcfg['model_key'])
        self.tok = AutoTokenizer.from_pretrained(str(self.path))
        kw = {}
        dm = reg['device_map'] if 'device_map' in reg else gcfg.get('device_map')
        self.multi = dm == 'auto' and torch.cuda.device_count() > 1
        if self.multi:
            mem = gcfg.get('max_memory_gib', 20)
            mem = mem if isinstance(mem, list) else [mem] * torch.cuda.device_count()
            kw.update(device_map='auto', max_memory={i: '{}GiB'.format(mem[i]) for i in range(len(mem))})
        attn = gcfg.get('attn_implementation', 'sdpa')
        try:
            self.model = from_pretrained(AutoModelForCausalLM, self.path, attn_implementation=attn, **kw)
        except Exception as e:  # noqa: BLE001
            log('  attn_implementation={} 加载失败（{}），改用 eager'.format(attn, type(e).__name__))
            self.model = from_pretrained(AutoModelForCausalLM, self.path, attn_implementation='eager', **kw)
        if not self.multi:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model.to(self.device)
        else:
            placed = set(str(d) for d in getattr(self.model, 'hf_device_map', {}).values())
            off = sorted(d for d in placed if d in ('cpu', 'disk', 'meta'))
            if off:
                raise RuntimeError('模型无法完整放进两张 GPU（部分层被放到 {}）'.format(off))
            self.device = 'cuda:0'
        self.model.eval()
        if self.model.dtype != torch.float32:
            raise RuntimeError('模型精度为 {}，应为 float32'.format(self.model.dtype))
        gc = copy.deepcopy(self.model.generation_config)
        gc.do_sample = False
        gc.num_beams = 1
        for attr in ('temperature', 'top_p', 'top_k', 'min_p', 'typical_p', 'repetition_penalty'):
            if hasattr(gc, attr):
                setattr(gc, attr, None)
        gc.max_new_tokens = int(gcfg['max_new_tokens'])
        eos = gc.eos_token_id if gc.eos_token_id is not None else self.tok.eos_token_id
        self.eos = set(eos if isinstance(eos, (list, tuple)) else [eos])
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else gc.pad_token_id
        if pad is None:
            pad = sorted(self.eos)[0]
        self.pad = int(pad)
        gc.pad_token_id = self.pad
        self.gc = gc
        self.max_new = int(gcfg['max_new_tokens'])
        self.budget = int(gcfg.get('token_budget', 12000))
        self.max_batch = int(gcfg.get('max_batch', 8))
        self.revision = self.path.name
        self.repo = reg.get('repo', str(self.path))
        if mode == 'on' or (mode == 'auto' and p40.pascal_like(torch)):
            self.grouped_decode, self.grouped_check = p40.enable_grouped_decode(self.model, self.pad, self.device, log)
        else:
            p40.set_grouped_decode(False)
            self.grouped_decode, self.grouped_check = False, None
        log('  模型 {}，dtype={}，attn={}，多卡={}，max_new_tokens={}，词元预算={}，预填充分块={}'.format(
            self.repo, self.model.dtype, getattr(self.model.config, '_attn_implementation', '?'), self.multi,
            self.max_new, self.budget, self.prefill_chunk))

    def settings(self):
        return {'repo': self.repo, 'revision': self.revision, 'multi_gpu': self.multi, 'sdpa_patch': self.sdpa_patch,
                'grouped_decode': self.grouped_decode, 'grouped_check': self.grouped_check,
                'memory_cap_gib': self.memory_cap_gib, 'prefill_chunk': self.prefill_chunk,
                'token_budget': self.budget, 'max_batch': self.max_batch, 'chat_template_kwargs': self.chat_kwargs}

    def encode(self, messages):
        ids = self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, **self.chat_kwargs)
        if isinstance(ids, dict) or hasattr(ids, 'keys'):
            ids = ids['input_ids']
        return [int(x) for x in ids]

    def plan_batches(self, items, max_new=None):
        """items 已按长度降序排列；按 (批大小 × (最长提示 + max_new)) ≤ 预算 组批。"""
        mn = int(max_new or self.max_new)
        ln = lambda x: x['n'] if 'n' in x else len(x['ids'])  # noqa: E731
        batches, cur = [], []
        for it in items:
            longest = max([ln(x) for x in cur] + [ln(it)])
            if cur and (len(cur) + 1 > self.max_batch or (len(cur) + 1) * (longest + mn) > self.budget):
                batches.append(cur)
                cur = []
            cur.append(it)
        if cur:
            batches.append(cur)
        return batches

    def run_batch(self, batch, chunk=None, force_full=False, max_new=None, stop_json=False):
        torch = self.torch
        mn = int(max_new or self.max_new)
        chunk = chunk or self.prefill_chunk
        n = max(len(x['ids']) for x in batch)
        ids = torch.full((len(batch), n), self.pad, dtype=torch.long)
        mask = torch.zeros((len(batch), n), dtype=torch.long)
        for i, x in enumerate(batch):
            ids[i, n - len(x['ids']):] = torch.tensor(x['ids'])
            mask[i, n - len(x['ids']):] = 1
        gc = self.gc
        if chunk or force_full or mn != self.max_new:
            gc = copy.deepcopy(self.gc)
            gc.max_new_tokens = mn
            if chunk:
                gc.prefill_chunk_size = int(chunk)
            if force_full:
                gc.min_new_tokens = mn                   # 只用于 E0 显存压力测试
        kw = {}
        if stop_json:
            from transformers import StoppingCriteriaList
            kw['stopping_criteria'] = StoppingCriteriaList([json_stopper(self.tok, n, len(batch))])
        with torch.inference_mode():
            out = self.model.generate(input_ids=ids.to(self.device), attention_mask=mask.to(self.device),
                                      generation_config=gc, **kw)
        res = []
        for i in range(len(batch)):
            gen = out[i, n:].tolist()
            k = next((j for j, t in enumerate(gen) if t in self.eos or t == self.pad), None)
            n_out = (k + 1) if k is not None else len(gen)
            text = self.tok.decode(gen[:n_out], skip_special_tokens=True).strip()
            res.append({'answer': text, 'n_out': n_out, 'hit_max': k is None and len(gen) >= mn,
                        'prefill_chunk': int(chunk) if chunk else None})
        del out, ids, mask
        return res

    def release(self):
        pygc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def close(self):
        self.model = None
        self.release()

    def generate(self, batch, depth=0, force_full=False, max_new=None, stop_json=False):
        """显存不足时把这一批对半拆开重试；只剩一条仍不足时，改用（更小的）分块预填充再试一次。"""
        torch = self.torch
        try:
            return self.run_batch(batch, force_full=force_full, max_new=max_new, stop_json=stop_json), depth
        except torch.cuda.OutOfMemoryError:
            pass
        self.release()                      # 离开 except 块后，异常引用的显存才会真正释放
        if len(batch) > 1:
            self.log('  显存不足，批大小 {} 拆半重试'.format(len(batch)))
            h = len(batch) // 2
            a, d1 = self.generate(batch[:h], depth + 1, force_full, max_new, stop_json)
            b, d2 = self.generate(batch[h:], depth + 1, force_full, max_new, stop_json)
            return a + b, max(d1, d2)
        chunk = max(64, self.prefill_chunk // 4) if self.prefill_chunk else self.prefill_chunk_on_oom
        if not chunk:
            raise RuntimeError('显存不足：单条输入 {} 词元'.format(len(batch[0]['ids'])))
        self.log('  显存不足（单条，输入 {} 词元），改用分块预填充（每块 {} 词元）重试'.format(len(batch[0]['ids']), chunk))
        return self.run_batch(batch, chunk=chunk, force_full=force_full, max_new=max_new,
                              stop_json=stop_json), depth + 1


# ----------------------------------------------------------------------------- 主流程

def out_path(cond, gen, root=None):
    return (root or (RUNS / 'gen')) / cond['target'] / gen / '{}.jsonl'.format(cond['id'])


def run_condition(G, cond, cfg, prompts, prompts_sha, cache, log, root=None):
    target = cond['target']
    if target not in cache:
        cache[target] = load_target(cfg, target)
    questions, pools = cache[target]
    qmap = {q['qid']: q for q in questions}
    sel = ensure_selection(cond, cfg, cache)
    path = out_path(cond, cond['gen'], root)
    done = {r['qid'] for r in read_jsonl(path) if r.get('answer') is not None}
    items = []
    for qid in sorted(sel):
        if qid in done:
            continue
        docs = sel[qid]
        bydoc = {c['document_id']: c for c in pools[qid]}
        texts = [evidence_text(bydoc[d]) for d in docs]
        msgs = build_messages(prompts, cond, qmap[qid]['question_final'], texts)
        items.append({'qid': qid, 'docs': docs, 'ids': G.encode(msgs)})
    log('{}：共 {} 题，已完成 {}，待生成 {}'.format(cond['id'], len(sel), len(done), len(items)))
    if items:
        items.sort(key=lambda x: (-len(x['ids']), x['qid']))
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('a', encoding='utf-8') as fh:
            for b in G.plan_batches(items):
                G.release()                  # 每批从相同的显存状态开始，拆批行为不受之前批次影响
                t0 = time.time()
                res, depth = G.generate(b)
                dt = time.time() - t0
                tok_out = sum(r['n_out'] for r in res)
                for x, r in zip(b, res):
                    rec = {'aid': answer_id(cond['id'], x['qid'], cond['gen']), 'cond': cond['id'],
                           'group': cond['group'], 'set': cond['set'], 'qid': x['qid'], 'target': target,
                           'gen': cond['gen'], 'gen_repo': G.repo, 'gen_revision': G.revision,
                           'prompt': cond['prompt'], 'prompts_sha256': prompts_sha, 'k': cond['k'],
                           'docs': x['docs'], 'answer': r['answer'], 'n_in': len(x['ids']), 'n_out': r['n_out'],
                           'hit_max': r['hit_max'], 'has_think': '<think>' in r['answer'],
                           'repeat': repeated(r['answer']), 'batch_size': len(b), 'oom_split': depth,
                           'prefill_chunk': r.get('prefill_chunk'),
                           'seconds_per_item': round(dt / len(b), 2), 'time': now()}
                    fh.write(json.dumps(rec, ensure_ascii=False) + '\n')
                fh.flush()
                log('  批 {} 条，最长输入 {}，输出 {} 词元，{:.1f} 秒（{:.1f} 词元/秒）'.format(
                    len(b), max(len(x['ids']) for x in b), tok_out, dt, tok_out / max(dt, 1e-6)))
    recs = {}
    for r in read_jsonl(path):
        if r.get('answer') is not None and r['qid'] not in recs:
            recs[r['qid']] = r
    missing = sorted(set(sel) - set(recs))
    if missing:
        raise RuntimeError('{} 仍有 {} 题没有答案'.format(cond['id'], len(missing)))
    write_jsonl(path, [recs[q] for q in sorted(recs)])
    write_json(path.with_suffix('.done.json'), {'cond': cond['id'], 'n': len(recs), 'sha256': sha256_file(path),
                                                'finished': now()})
    return [recs[q] for q in sorted(recs)]


def stress_test(G, cfg, prompts, cache, log):
    """E0 显存压力测试：先量出所有题目“全部 20 条证据”提示的最长词元数（只分词，不生成），
    再用校准集题目和循环拼接的校准集证据造一条同样长的提示，强制生成满 max_new_tokens，
    记录显存峰值、是否用到分块预填充与速度。答案不保存，也不进入任何分析。"""
    torch = G.torch
    names = sorted(k[:-len('_evidence')] for k in prompts if k.endswith('_evidence'))
    longest, where = 0, None
    for target in ('calib', 'sealed'):
        if target not in cache:
            cache[target] = load_target(cfg, target)
        qs, pools = cache[target]
        for name in names:
            cond = {'k': 20, 'prompt': name}
            for q in qs:
                texts = [evidence_text(c) for c in pools[q['qid']]]
                n = len(G.encode(build_messages(prompts, cond, q['question_final'], texts)))
                if n > longest:
                    longest, where = n, {'target': target, 'prompt': name, 'qid': q['qid']}
    qs, pools = cache['calib']
    q = max(qs, key=lambda x: len(x['question_final']))
    cond = {'k': 20, 'prompt': where['prompt']}
    base = [evidence_text(c) for c in pools[q['qid']]]
    ev = list(base)
    ids = G.encode(build_messages(prompts, cond, q['question_final'], ev))
    i = 0
    while len(ids) < longest:
        ev.append(base[i % len(base)])
        i += 1
        ids = G.encode(build_messages(prompts, cond, q['question_final'], ev))
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    res, depth = G.generate([{'qid': 'stress', 'ids': ids}], force_full=True)
    dt = time.time() - t0
    rep = {'longest_real_prompt_tokens': longest, 'longest_at': where, 'stress_prompt_tokens': len(ids),
           'output_tokens': res[0]['n_out'], 'seconds': round(dt, 1),
           'tokens_per_second': round(res[0]['n_out'] / max(dt, 1e-6), 2),
           'prefill_chunk_needed': res[0].get('prefill_chunk') is not None, 'settings': G.settings()}
    if torch.cuda.is_available():
        rep['peak_allocated_gib'] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
        rep['peak_reserved_gib'] = round(torch.cuda.max_memory_reserved() / 2 ** 30, 2)
    log('E0 显存压力测试：' + json.dumps(rep, ensure_ascii=False))
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--conds', default='')
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--job', required=True)
    a = ap.parse_args()
    cfg = load_config()
    set_determinism(42)
    log = Log(RUNS / 'logs' / 'gen_{}.log'.format(a.job))
    prompts, prompts_sha = load_prompts()
    allc = {c['id']: c for c in C.all_conditions(cfg)}
    conds = C.smoke_conditions() if a.smoke else [allc[x] for x in a.conds.split(',') if x]
    gens = sorted({c['gen'] for c in conds})
    if len(gens) != 1:
        raise SystemExit('一个作业只能用一个问答模型')
    gname = gens[0]
    log('=' * 60)
    log('作业 {}：{} 个条件，问答模型 {}'.format(a.job, len(conds), gname))
    log('环境：' + json.dumps(environment(), ensure_ascii=False))
    G = Generator(cfg['generators'][gname], log)
    cache = {}
    t0 = time.time()
    results = {}
    for cond in conds:
        if out_path(cond, cond['gen']).with_suffix('.done.json').exists() and not a.smoke:
            log('{}：已完成，跳过'.format(cond['id']))
            continue
        try:
            results[cond['id']] = run_condition(G, cond, cfg, prompts, prompts_sha, cache, log)
        except FileNotFoundError as e:
            if not cond.get('optional'):
                raise
            log('[跳过可选条件] {}：{}'.format(cond['id'], e))
    if a.smoke:
        import torch
        rep = {'started': now(), 'env': environment(), 'conditions': {}}
        for cond in conds:
            again = run_condition(G, cond, cfg, prompts, prompts_sha, cache, log, root=RUNS / 'e0' / 'repeat')
            first = results.get(cond['id']) or read_jsonl(out_path(cond, cond['gen']))
            same = [x['answer'] == y['answer'] for x, y in zip(sorted(first, key=lambda r: r['qid']),
                                                             sorted(again, key=lambda r: r['qid']))]
            rep['conditions'][cond['id']] = {
                'n': len(first), 'identical_on_rerun': all(same), 'n_identical': sum(same),
                'has_think': sum(r['has_think'] for r in first), 'hit_max': sum(r['hit_max'] for r in first),
                'repeat': sum(r['repeat'] for r in first), 'max_input_tokens': max(r['n_in'] for r in first),
                'mean_output_tokens': round(sum(r['n_out'] for r in first) / max(1, len(first)), 1),
                'mean_seconds_per_item': round(sum(r['seconds_per_item'] for r in first) / max(1, len(first)), 2)}
        if torch.cuda.is_available():
            rep['peak_memory_gib'] = round(torch.cuda.max_memory_allocated() / 2 ** 30, 2)
        rep['stress'] = stress_test(G, cfg, prompts, cache, log)
        rep['passed'] = all(v['identical_on_rerun'] and v['has_think'] == 0 for v in rep['conditions'].values())
        rep['seconds'] = round(time.time() - t0, 1)
        write_json(RUNS / 'e0' / 'gen_smoke.json', rep)
        log('E0 冒烟：{}'.format(json.dumps(rep['conditions'], ensure_ascii=False)))
        if not rep['passed']:
            log('E0 冒烟未通过（两次运行结果不一致，或出现 <think>）')
            sys.exit(1)
    mark_done(a.job, {'conditions': [c['id'] for c in conds], 'seconds': round(time.time() - t0, 1)})
    log('作业 {} 完成，用时 {:.0f} 秒'.format(a.job, time.time() - t0))


if __name__ == '__main__':
    main()
