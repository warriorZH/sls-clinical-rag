#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""封存集模型打分（eval_v1）：在服务器上运行。

    python eval_v1/server_score.py discover                  查找检查点、公开模型和既往训练代码，
                                                             写出 discover_report.md 与 models.json（建议值，需人工确认）
    python eval_v1/server_score.py score --target calib      冒烟测试：5 道校准题（开发集，不属于封存集）
    python eval_v1/server_score.py score --target sealed     封存集：每个方法只运行一次

本脚本只读取问题文本和候选池，不读取任何金标文件。设置依据见 eval_v1/实验对比方案.md。
需要 Python 3.8+；打分需要 torch 与 transformers（Qwen3-Reranker 需要 transformers>=4.51）。
"""
import argparse
import datetime as dt
import hashlib
import json
import math
import os
import platform
import re
import shutil
import socket
import sys
import time
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
TESTSET = EVAL_DIR.parent
SEALED_DIR = TESTSET / '_sealed_负责人保存_勿发给标注员'
DEFAULT_ROOT = '/home/<user>/UUM/EEE_research'
K = 20

TARGETS = {
    'sealed': {
        'questions': SEALED_DIR / 'final_questions.json',
        'pools': SEALED_DIR / 'sealed_pools.jsonl',
        'sha256': {'questions': 'af0b2b4d34a84a4be57524c84788df938e295fd3e9afd263b25d7b7c70b5aeb3',
                   'pools': '6c228959456ad06765adc84f86811eb0abe410901769fe5013bd7d6559de4659'},
        'n_questions': 176,
    },
    'calib': {
        'questions': EVAL_DIR / 'calib' / 'calibration_questions.json',
        'pools': EVAL_DIR / 'calib' / 'calibration_pools.jsonl',
        'sha256': {'questions': '15ebdb0d02cc560e9d9b891169e73b5cf32935684e1cdd1d0c1049fb1192e71f',
                   'pools': 'd77436300a56dfb08e60a5087b7fe523427161e0b283408835c48c637d68acc3'},
        'n_questions': 5,
    },
}

# Qwen3-Reranker 官方提示模板（https://huggingface.co/Qwen/Qwen3-Reranker-0.6B）
QWEN3_PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the '
                'Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')
QWEN3_SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
QWEN3_INSTRUCTION = ('Given a Traditional Chinese Medicine clinical case and a question about it, '
                     'retrieve passages that help answer the question')

# 可选 LLM 逐点判断：只给问题和证据，不给金标要点
LLM_JUDGE_SYSTEM = '你是一名中医临床专家，负责判断候选证据能否支撑对病例问题的回答。'
LLM_JUDGE_PROMPT = ('【病例问题】\n{question}\n\n【候选证据】\n{evidence}\n\n'
                    '请判断：这段证据是否明确写出了回答本题所需的某个要点（中医病名、证候、病机或治疗原则之一），'
                    '并且其适用条件与本病例相符，足以作为该要点的完整支持？只回答“是”或“否”。')

# 进入设置指纹的字段：会影响分数的设置。batch_size、enabled、note 不影响分数，不计入。
SPEC_KEYS = ('name', 'method', 'seed', 'kind', 'path', 'max_length', 'dtype', 'positive_index',
             'pair_order', 'truncation', 'query_prefix', 'passage_prefix', 'instruction',
             'base_model', 'spm', 'query_max_tokens', 'micro')

HF_ALLOW = ['*.json', '*.safetensors', '*.model', '*.txt', '*.tiktoken', 'tokenizer*', 'vocab*', 'merges*']


# ----------------------------------------------------------------------------- 通用

def now():
    return dt.datetime.now().astimezone().isoformat(timespec='seconds')


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def sha256_text(s):
    return hashlib.sha256(s.encode('utf-8')).hexdigest()


def batches(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


class Log:
    def __init__(self, path=None):
        self.path = path

    def __call__(self, *msg):
        line = ' '.join(str(m) for m in msg)
        print(line, flush=True)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open('a', encoding='utf-8') as f:
                f.write(line + '\n')


def spec_hash(spec):
    core = {k: spec.get(k) for k in SPEC_KEYS}
    return sha256_text(json.dumps(core, ensure_ascii=False, sort_keys=True))


# ----------------------------------------------------------------------------- 输入

def load_inputs(target, questions_path=None, pools_path=None):
    t = TARGETS[target]
    qp = Path(questions_path) if questions_path else t['questions']
    pp = Path(pools_path) if pools_path else t['pools']
    hashes = {}
    for kind, p in (('questions', qp), ('pools', pp)):
        if not p.exists():
            sys.exit('找不到 {}。请确认文件位置（见 运行步骤.md）。'.format(p))
        h = sha256_file(p)
        if h != t['sha256'][kind]:
            sys.exit('{} 的 SHA256 为 {}，与冻结值 {} 不符，已停止。'.format(p.name, h, t['sha256'][kind]))
        hashes[kind] = h
    questions = [q for q in json.loads(qp.read_text(encoding='utf-8')) if q.get('included', True)]
    qmap = {q['qid']: q for q in questions}
    pairs = []
    with pp.open(encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r['qid'] not in qmap:
                continue
            evidence = r.get('context_text') or r.get('text') or ''
            pairs.append({'qid': r['qid'], 'document_id': str(r['document_id']), 'rank': int(r['rank']),
                          'query': qmap[r['qid']]['question_final'], 'evidence': evidence})
    pairs.sort(key=lambda x: (x['qid'], x['rank']))
    per_q = {}
    for p in pairs:
        per_q.setdefault(p['qid'], []).append(p)
    problems = []
    if len(qmap) != t['n_questions']:
        problems.append('纳入题数 {}，应为 {}'.format(len(qmap), t['n_questions']))
    for qid in qmap:
        ranks = sorted(x['rank'] for x in per_q.get(qid, []))
        if ranks != list(range(1, K + 1)):
            problems.append('{} 的候选名次不是 1..{}'.format(qid, K))
    empty = sum(1 for p in pairs if not p['evidence'].strip())
    if problems:
        sys.exit('输入检查未通过：' + '；'.join(problems[:10]))
    meta = {'questions_file': str(qp), 'pools_file': str(pp), 'questions_sha256': hashes['questions'],
            'pools_sha256': hashes['pools'], 'n_questions': len(qmap), 'n_pairs': len(pairs),
            'empty_evidence': empty}
    return qmap, pairs, meta


# ----------------------------------------------------------------------------- 模型

def torch_dtype(name, device='cuda'):
    """按设备取实际精度：CPU 上一律用 fp32；GPU 不支持 bf16 时改用 fp16。实际精度写入运行清单。"""
    import torch
    if not str(device).startswith('cuda'):
        name = 'fp32'
    elif name == 'bf16' and not torch.cuda.is_bf16_supported():
        name = 'fp16'
    return {'fp32': torch.float32, 'fp16': torch.float16, 'bf16': torch.bfloat16}[name], name


def resolve_model(path_or_id, log):
    """本地目录直接使用；否则视为 Hugging Face 模型名，下载或读取缓存。返回 (本地目录, 版本号或 None)。"""
    p = Path(os.path.expanduser(str(path_or_id)))
    if p.exists():
        return p, None
    if re.match(r'^[\w.\-]+/[\w.\-]+$', str(path_or_id)):
        from huggingface_hub import snapshot_download
        log('  从 Hugging Face 获取 {}（HF_ENDPOINT={}）'.format(path_or_id, os.environ.get('HF_ENDPOINT', '默认')))
        local = Path(snapshot_download(repo_id=str(path_or_id), allow_patterns=HF_ALLOW))
        if not any(local.glob('*.safetensors')):
            local = Path(snapshot_download(repo_id=str(path_or_id), allow_patterns=HF_ALLOW + ['*.bin']))
        return local, local.name
    raise FileNotFoundError('找不到模型 {}'.format(path_or_id))


def weights_fingerprint(d):
    files = sorted(x for x in Path(d).iterdir()
                   if x.is_file() and x.suffix in ('.safetensors', '.bin', '.pt', '.pth') and 'optimizer' not in x.name
                   and 'scheduler' not in x.name and 'rng_state' not in x.name and 'training_args' not in x.name)
    return {x.name: sha256_file(x) for x in files}


def score_bm25(spec, pairs, device, log):
    return [-float(p['rank']) for p in pairs], {'note': '分数为负的 BM25 名次'}


def score_seqcls(spec, pairs, device, log):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    path = spec['_local_path']
    tok = AutoTokenizer.from_pretrained(str(path))
    dtype, dtype_name = torch_dtype(spec.get('dtype', 'fp32'), device)
    model = AutoModelForSequenceClassification.from_pretrained(str(path), torch_dtype=dtype).to(device).eval()
    n_labels = int(model.config.num_labels)
    pos = int(spec.get('positive_index', 1))
    if n_labels > 1 and not 0 <= pos < n_labels:
        raise ValueError('positive_index={} 超出 num_labels={}'.format(pos, n_labels))
    max_len = int(spec.get('max_length', 512))
    order = spec.get('pair_order', 'query_first')
    if order not in ('query_first', 'passage_first'):
        raise ValueError('pair_order 只能是 query_first 或 passage_first')
    qpre, ppre = spec.get('query_prefix', '') or '', spec.get('passage_prefix', '') or ''
    trunc = spec.get('truncation') or ('only_second' if order == 'query_first' else 'only_first')
    if trunc not in ('only_first', 'only_second', 'longest_first'):
        raise ValueError('truncation 只能是 only_first、only_second 或 longest_first')

    def texts(batch):
        q = [qpre + x['query'] for x in batch]
        e = [ppre + x['evidence'] for x in batch]
        return (q, e) if order == 'query_first' else (e, q)

    n_trunc = 0
    for b in batches(pairs, 256):
        a, c = texts(b)
        enc = tok(a, c, truncation=False)
        n_trunc += sum(len(ids) > max_len for ids in enc['input_ids'])
    scores, fallback = [], 0
    bs = int(spec.get('batch_size', 32))
    with torch.inference_mode():
        for b in batches(pairs, bs):
            a, c = texts(b)
            try:
                enc = tok(a, c, truncation=trunc, max_length=max_len, padding=True, return_tensors='pt')
            except Exception:
                fallback += len(b)
                enc = tok(a, c, truncation='longest_first', max_length=max_len, padding=True, return_tensors='pt')
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits.float()
            s = logits[:, 0] if n_labels == 1 else torch.log_softmax(logits, dim=-1)[:, pos]
            scores.extend(s.cpu().tolist())
    info = {'num_labels': n_labels, 'id2label': {str(k): v for k, v in model.config.id2label.items()},
            'positive_index': pos if n_labels > 1 else None,
            'score': 'logit' if n_labels == 1 else 'log_softmax[positive_index]',
            'max_length': max_len, 'truncation': trunc, 'pair_order': order, 'n_truncated': n_trunc,
            'longest_first_fallback': fallback, 'tokenizer_class': type(tok).__name__, 'dtype_used': dtype_name}
    del model
    return scores, info


def score_qwen3(spec, pairs, device, log):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = spec['_local_path']
    tok = AutoTokenizer.from_pretrained(str(path), padding_side='left')
    dtype, dtype_name = torch_dtype(spec.get('dtype', 'fp16'), device)
    model = AutoModelForCausalLM.from_pretrained(str(path), torch_dtype=dtype).to(device).eval()
    yes_id, no_id = tok.convert_tokens_to_ids('yes'), tok.convert_tokens_to_ids('no')
    if yes_id is None or no_id is None or yes_id == tok.unk_token_id or no_id == tok.unk_token_id:
        raise ValueError('分词器中找不到 yes/no 词元')
    pre = tok.encode(QWEN3_PREFIX, add_special_tokens=False)
    suf = tok.encode(QWEN3_SUFFIX, add_special_tokens=False)
    inst = spec.get('instruction') or QWEN3_INSTRUCTION
    max_len = int(spec.get('max_length', 2048))
    room = max_len - len(pre) - len(suf)

    def fmt(x):
        return '<Instruct>: {}\n<Query>: {}\n<Document>: {}'.format(inst, x['query'], x['evidence'])

    n_trunc = sum(len(tok(fmt(x), add_special_tokens=False)['input_ids']) > room for x in pairs)
    scores = []
    bs = int(spec.get('batch_size', 8))
    with torch.inference_mode():
        for b in batches(pairs, bs):
            enc = tok([fmt(x) for x in b], padding=False, truncation='longest_first',
                      return_attention_mask=False, max_length=room)
            enc['input_ids'] = [pre + ids + suf for ids in enc['input_ids']]
            enc = tok.pad(enc, padding=True, return_tensors='pt', max_length=max_len)
            enc = {k: v.to(device) for k, v in enc.items()}
            last = model(**enc).logits[:, -1, :].float()
            s = torch.log_softmax(torch.stack([last[:, no_id], last[:, yes_id]], dim=1), dim=1)[:, 1]
            scores.extend(s.cpu().tolist())
    info = {'score': 'log P(yes)', 'instruction': inst, 'max_length': max_len, 'n_truncated': n_trunc,
            'yes_id': yes_id, 'no_id': no_id, 'dtype_used': dtype_name}
    del model
    return scores, info


def score_llm_judge(spec, pairs, device, log):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    path = spec['_local_path']
    tok = AutoTokenizer.from_pretrained(str(path), padding_side='left')
    tok.truncation_side = 'left'
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype, dtype_name = torch_dtype(spec.get('dtype', 'bf16'), device)
    model = AutoModelForCausalLM.from_pretrained(str(path), torch_dtype=dtype).to(device).eval()
    yes, no = tok.encode('是', add_special_tokens=False), tok.encode('否', add_special_tokens=False)
    if len(yes) != 1 or len(no) != 1:
        log('  注意：“是/否”不是单个词元（{} / {}），取首个词元。'.format(yes, no))
    yes_id, no_id = yes[0], no[0]
    max_len = int(spec.get('max_length', 2048))

    def build(x):
        msgs = [{'role': 'system', 'content': LLM_JUDGE_SYSTEM},
                {'role': 'user', 'content': LLM_JUDGE_PROMPT.format(question=x['query'], evidence=x['evidence'])}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    texts = [build(x) for x in pairs]
    n_trunc = sum(len(tok(t, add_special_tokens=False)['input_ids']) > max_len for t in texts)
    scores = []
    bs = int(spec.get('batch_size', 4))
    with torch.inference_mode():
        for b in batches(texts, bs):
            enc = tok(b, add_special_tokens=False, padding=True, truncation=True, max_length=max_len,
                      return_tensors='pt')
            enc = {k: v.to(device) for k, v in enc.items()}
            last = model(**enc).logits[:, -1, :].float()
            s = torch.log_softmax(torch.stack([last[:, no_id], last[:, yes_id]], dim=1), dim=1)[:, 1]
            scores.extend(s.cpu().tolist())
    info = {'score': 'log P(是)/(P(是)+P(否))', 'max_length': max_len, 'n_truncated': n_trunc,
            'prompt_sha256': sha256_text(LLM_JUDGE_SYSTEM + '\n' + LLM_JUDGE_PROMPT), 'yes_id': yes_id, 'no_id': no_id,
            'dtype_used': dtype_name}
    del model
    return scores, info


# ---- Support-CE / Relevance-CE：与 E3 完全相同的加载、编码与打分 ----
# 依据服务器上的 experiments/stage5_real_retrieval_v1_2/score_three_seed_review.py 与
# experiments/stage4_retraining/train_stage4.py 中的 predict()：
#   基座 AutoModelForSequenceClassification(num_labels=2, fp32) + load_state_dict(best_checkpoint.pt['model'])
#   输入 [1] + spm(问题)[:160] + [2] + spm(证据)[:509-len(问题)] + [2]，只给 input_ids 与 attention_mask
#   按长度排序、每批 4 条，分数为 softmax(logits)[:, 1]

def eee_load(spec, device):
    import torch
    from transformers import AutoModelForSequenceClassification
    ck = torch.load(str(spec['path']), map_location='cpu', weights_only=False)
    cfg = ck.get('config') or {}
    base = spec.get('base_model') or cfg.get('model')
    if not base:
        raise ValueError('检查点中没有 config.model，请在 models.json 中写 base_model')
    try:
        model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=2, local_files_only=True,
                                                                   dtype=torch.float32)
    except TypeError:
        model = AutoModelForSequenceClassification.from_pretrained(base, num_labels=2, local_files_only=True,
                                                                   torch_dtype=torch.float32)
    model.load_state_dict(ck['model'])
    meta = {'base_model': base, 'checkpoint_epoch': ck.get('epoch'), 'checkpoint_step': ck.get('step'),
            'train_config': {k: cfg.get(k) for k in ('objective', 'seed', 'max_length', 'precision', 'model', 'gpu')}}
    del ck
    model.to(device).eval()
    return model, meta


def eee_encode(sp, query, evidence, qmax=160, total=512):
    q = sp.encode(query)[:qmax]
    t = sp.encode(evidence)[:total - 3 - len(q)]
    return [1] + q + [2] + t + [2]


def eee_predict(model, rows, micro, device):
    import numpy as np
    import torch
    model.eval()
    result = np.empty(len(rows))
    order = sorted(range(len(rows)), key=lambda i: len(rows[i]))
    with torch.no_grad():
        for k in range(0, len(order), micro):
            ix = order[k:k + micro]
            n = max(len(rows[i]) for i in ix)
            ids = torch.zeros((len(ix), n), dtype=torch.long)
            mask = torch.zeros_like(ids)
            for j, i in enumerate(ix):
                ids[j, :len(rows[i])] = torch.tensor(rows[i])
                mask[j, :len(rows[i])] = 1
            v = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
            if not torch.isfinite(v).all():
                raise RuntimeError('Nonfinite inference logits')
            result[ix] = v.softmax(-1)[:, 1].cpu().numpy()
    return result


def eee_repro(model, sp, repro, spec, device, log):
    """用同一检查点重算 E3 已保存的分数（context 版本），确认编码与打分方式一致。"""
    import numpy as np
    rows = []
    with open(repro['candidates'], encoding='utf-8') as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                if r.get('split') != 'train':
                    rows.append(r)
    ref = np.load(repro['scores'])
    out = {'candidates': repro['candidates'], 'scores': repro['scores'], 'rows': len(rows), 'ref_len': int(ref.shape[0])}
    if len(rows) != ref.shape[0]:
        out.update(passed=False, error='候选行数与参考分数长度不一致')
        return out
    n = int(repro.get('n', 200))
    stride = max(1, len(rows) // n)
    idx = list(range(0, len(rows), stride))[:n]
    got = eee_predict(model, [eee_encode(sp, rows[i]['question'], rows[i]['context_text']) for i in idx],
                      int(spec.get('micro', 4)), device)
    exp = ref[idx]
    diff = np.abs(got - exp)
    out.update(n=len(idx), max_abs_diff=float(diff.max()), mean_abs_diff=float(diff.mean()),
               spearman=spearman([float(x) for x in got], [float(x) for x in exp]),
               tolerance=float(repro.get('tol', 1e-3)))
    done = Path(repro['scores']).parent / 'DIRECT_REVIEW_SCORES_DONE.json'
    if done.exists():
        try:
            out['e3_checkpoint_sha256'] = json.loads(done.read_text(encoding='utf-8')).get('checkpoint_sha256')
        except Exception:  # noqa: BLE001
            pass
    mine = list((spec.get('_weights_sha256') or {}).values())
    if out.get('e3_checkpoint_sha256') and mine:
        out['checkpoint_matches_e3'] = out['e3_checkpoint_sha256'] == mine[0]
    out['passed'] = out['max_abs_diff'] < out['tolerance'] and out.get('checkpoint_matches_e3', True)
    log('  复现核对：{} 条，最大差 {:.2e}，平均差 {:.2e}，Spearman {:.6f}，检查点与 E3 一致：{} → {}'.format(
        out['n'], out['max_abs_diff'], out['mean_abs_diff'], out['spearman'], out.get('checkpoint_matches_e3', '无记录'),
        '通过' if out['passed'] else '未通过'))
    return out


def score_eee_ce(spec, pairs, device, log):
    import sentencepiece as spm
    model, meta = eee_load(spec, device)
    sp = spm.SentencePieceProcessor(model_file=str(spec.get('spm') or Path(meta['base_model']) / 'spm.model'))
    qmax, total = int(spec.get('query_max_tokens', 160)), int(spec.get('max_length', 512))
    rows, n_trunc = [], 0
    for x in pairs:
        q_full, e_full = sp.encode(x['query']), sp.encode(x['evidence'])
        n_trunc += len(q_full) > qmax or len(e_full) > total - 3 - min(len(q_full), qmax)
        rows.append(eee_encode(sp, x['query'], x['evidence'], qmax, total))
    scores = eee_predict(model, rows, int(spec.get('micro', 4)), device)
    info = dict(meta, score='softmax(logits)[:,1]', encoding='[1]+spm(q)[:{}]+[2]+spm(e)[:{}-3-len(q)]+[2]'.format(qmax, total),
                n_truncated=n_trunc, micro=int(spec.get('micro', 4)), dtype_used='fp32')
    if spec.get('repro') and spec.get('_target') == 'calib':
        info['repro'] = eee_repro(model, sp, spec['repro'], spec, device, log)
        if not info['repro'].get('passed'):
            del model
            raise RuntimeError('复现核对未通过：{}'.format(json.dumps(info['repro'], ensure_ascii=False)))
    del model
    return [float(v) for v in scores], info


SCORERS = {'bm25_rank': score_bm25, 'hf_seqcls': score_seqcls, 'qwen3_reranker': score_qwen3, 'eee_ce_pt': score_eee_ce,
           'llm_judge': score_llm_judge}


# ----------------------------------------------------------------------------- 环境

def environment():
    env = {'python': sys.version.split()[0], 'platform': platform.platform(), 'host': socket.gethostname()}
    try:
        import torch
        env['torch'] = torch.__version__
        env['cuda_available'] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            env['gpus'] = ['{} ({:.0f} GB)'.format(torch.cuda.get_device_name(i),
                                                   torch.cuda.get_device_properties(i).total_memory / 2 ** 30)
                           for i in range(torch.cuda.device_count())]
    except Exception as e:  # noqa: BLE001
        env['torch'] = '未安装（{}）'.format(type(e).__name__)
    try:
        import transformers
        env['transformers'] = transformers.__version__
    except Exception as e:  # noqa: BLE001
        env['transformers'] = '未安装（{}）'.format(type(e).__name__)
    env['HF_ENDPOINT'] = os.environ.get('HF_ENDPOINT', '')
    return env


def pick_device(name):
    if name != 'auto':
        return name
    try:
        import torch
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    except Exception:  # noqa: BLE001
        return 'cpu'


# ----------------------------------------------------------------------------- score

def load_config(path, target='calib'):
    path = Path(path)
    if not path.exists():
        sys.exit('找不到配置文件 {}。请先运行 discover，并核对生成的 models.json。'.format(path))
    cfg = json.loads(path.read_text(encoding='utf-8'))
    names = [m['name'] for m in cfg['models']]
    if len(names) != len(set(names)):
        sys.exit('models.json 中的 name 有重复。')
    for m in cfg['models']:
        if m.get('enabled', True) and '<' in str(m.get('path', '')):
            if target == 'sealed':
                sys.exit('{} 的 path 仍是占位符，请在 models.json 中填写真实路径，或把 enabled 设为 false。'.format(m['name']))
            print('[跳过] {}：path 仍是占位符（{}）'.format(m['name'], m['path']))
            m['enabled'] = False
        if m['kind'] not in SCORERS:
            sys.exit('{} 的 kind={} 不受支持。'.format(m['name'], m['kind']))
    return cfg


def spearman(x, y):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2.0
            i = j + 1
        return r
    rx, ry = ranks(x), ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    return sxy / math.sqrt(sxx * syy) if sxx and syy else float('nan')


def read_scores(path):
    out = {}
    with path.open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[(r['qid'], r['document_id'])] = r['score']
    return out


def cmd_score(a):
    cfg = load_config(a.config, a.target)
    qmap, pairs, inputs = load_inputs(a.target, a.questions, a.pools)
    out_dir = EVAL_DIR / 'scores' / a.target
    out_dir.mkdir(parents=True, exist_ok=True)
    log = Log(out_dir / 'run_log.txt')
    mpath = out_dir / 'RUN_MANIFEST.json'
    manifest = json.loads(mpath.read_text(encoding='utf-8')) if mpath.exists() else {
        'target': a.target, 'created': now(), 'inputs': inputs, 'runs': {}, 'reruns': [], 'skipped': []}
    for key in ('questions_sha256', 'pools_sha256'):
        if manifest['inputs'][key] != inputs[key]:
            sys.exit('RUN_MANIFEST.json 记录的输入与当前输入不同，已停止。')
    calib_manifest = {}
    cpath = EVAL_DIR / 'scores' / 'calib' / 'RUN_MANIFEST.json'
    if a.target == 'sealed' and cpath.exists():
        calib_manifest = json.loads(cpath.read_text(encoding='utf-8')).get('runs', {})
    device = pick_device(a.device)
    env = environment()
    script_sha = sha256_file(Path(__file__))
    config_sha = sha256_file(Path(a.config))
    only = {x.strip() for x in a.only.split(',')} if a.only else None
    log('=' * 70)
    log('{}  target={}  device={}  题数={}  候选={}'.format(now(), a.target, device, inputs['n_questions'],
                                                       inputs['n_pairs']))
    log('环境：' + json.dumps(env, ensure_ascii=False))

    for spec in cfg['models']:
        name = spec['name']
        if only is not None and name not in only:
            continue
        if not spec.get('enabled', True):
            log('[跳过] {}：enabled=false'.format(name))
            continue
        out = out_dir / '{}.jsonl'.format(name)
        sh = spec_hash(spec)
        if out.exists() and a.target == 'sealed' and not a.rerun_reason:
            log('[跳过] {}：已有封存集结果。每个方法只运行一次；确需重跑请加 --rerun-reason "原因"。'.format(name))
            continue
        log('-' * 70)
        log('[{}] {}  kind={}  path={}'.format(name, spec.get('method', ''), spec['kind'], spec.get('path', '')))
        t0 = time.time()
        rec = {'name': name, 'method': spec.get('method'), 'seed': spec.get('seed'), 'kind': spec['kind'],
               'spec': {k: spec.get(k) for k in SPEC_KEYS}, 'spec_sha256': sh, 'started': now(),
               'device': device, 'env': env, 'script_sha256': script_sha, 'config_sha256': config_sha,
               'inputs': {'questions_sha256': inputs['questions_sha256'], 'pools_sha256': inputs['pools_sha256']}}
        try:
            if spec['kind'] == 'eee_ce_pt':
                ck = Path(spec['path'])
                if not ck.is_file():
                    raise FileNotFoundError('找不到检查点 {}'.format(ck))
                log('  计算检查点 SHA256 …')
                rec['resolved_path'] = str(ck)
                rec['weights_sha256'] = {ck.name: sha256_file(ck)}
                base = Path(spec.get('base_model') or '')
                rec['base_files_sha256'] = {x.name: sha256_file(x) for x in (base / 'config.json', base / 'spm.model')
                                            if x.is_file()}
                spec = dict(spec, _target=a.target, _weights_sha256=rec['weights_sha256'])
            elif spec['kind'] != 'bm25_rank':
                local, revision = resolve_model(spec['path'], log)
                spec = dict(spec, _local_path=local, _target=a.target)
                rec['resolved_path'] = str(local)
                rec['hf_revision'] = revision
                log('  计算权重文件 SHA256 …')
                rec['weights_sha256'] = weights_fingerprint(local)
                if not rec['weights_sha256']:
                    raise FileNotFoundError('{} 中没有权重文件'.format(local))
            else:
                rec['weights_sha256'] = {}
            if a.target == 'sealed' and not a.skip_calib_check:
                c = calib_manifest.get(name)
                if not c or c.get('status') != 'ok' or c.get('spec_sha256') != sh \
                        or c.get('weights_sha256') != rec['weights_sha256']:
                    log('[跳过] {}：没有找到相同设置、相同权重且通过的校准集运行。请先运行 --target calib。'.format(name))
                    manifest['skipped'].append({'name': name, 'time': now(), 'reason': 'calib check failed'})
                    continue
            if a.target == 'sealed' and a.skip_calib_check:
                rec['calib_check_skipped'] = True
            if out.exists() and a.target == 'sealed':
                sup = out_dir / '_superseded'
                sup.mkdir(exist_ok=True)
                stamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
                shutil.move(str(out), str(sup / '{}.{}.jsonl'.format(name, stamp)))
                manifest['reruns'].append({'name': name, 'time': now(), 'reason': a.rerun_reason,
                                           'previous': manifest['runs'].get(name),
                                           'archived_as': '_superseded/{}.{}.jsonl'.format(name, stamp)})
            scores, info = SCORERS[spec['kind']](spec, pairs, device, log)
            rec['info'] = info
            bad = [i for i, s in enumerate(scores) if not (isinstance(s, float) and math.isfinite(s))]
            if len(scores) != len(pairs) or bad:
                raise ValueError('分数数目 {}（应为 {}），非有限值 {} 个'.format(len(scores), len(pairs), len(bad)))
            tmp = out.with_suffix('.jsonl.tmp')
            with tmp.open('w', encoding='utf-8') as f:
                for p, s in zip(pairs, scores):
                    f.write(json.dumps({'qid': p['qid'], 'document_id': p['document_id'], 'rank': p['rank'],
                                        'score': s}, ensure_ascii=False) + '\n')
            tmp.replace(out)
            const_q = 0
            for qid in qmap:
                vals = {s for p, s in zip(pairs, scores) if p['qid'] == qid}
                const_q += len(vals) == 1
            rec.update({'status': 'ok', 'n_pairs': len(scores), 'score_min': min(scores), 'score_max': max(scores),
                        'score_mean': sum(scores) / len(scores), 'questions_all_tied': const_q,
                        'scores_sha256': sha256_file(out)})
            log('  完成：{} 条，分数范围 [{:.4f}, {:.4f}]，截断 {} 条，全部同分的题 {} 道'.format(
                len(scores), min(scores), max(scores), info.get('n_truncated', 0), const_q))
        except Exception as e:  # noqa: BLE001
            rec.update({'status': 'failed', 'error': '{}: {}'.format(type(e).__name__, e)})
            log('  失败：{}: {}'.format(type(e).__name__, e))
        finally:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
        rec['finished'] = now()
        rec['seconds'] = round(time.time() - t0, 1)
        if rec.get('status') == 'failed':
            manifest.setdefault('failures', []).append(rec)
        manifest['runs'][name] = rec
        mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')

    mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    log('=' * 70)
    log('汇总（{}）：'.format(a.target))
    for name, r in manifest['runs'].items():
        log('  {:<28} {:<7} {}'.format(name, r.get('status'), r.get('error', '') or
                                       '截断 {}，{} 秒'.format(r.get('info', {}).get('n_truncated', 0), r.get('seconds'))))
    groups = {}
    for name, r in manifest['runs'].items():
        if r.get('status') == 'ok' and r.get('seed') is not None:
            groups.setdefault(r.get('method'), []).append(name)
    for method, names in groups.items():
        if len(names) > 1:
            sc = {n: read_scores(out_dir / '{}.jsonl'.format(n)) for n in names}
            keys = sorted(sc[names[0]])
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    rho = spearman([sc[names[i]][k] for k in keys], [sc[names[j]][k] for k in keys])
                    log('  种子间 Spearman  {} vs {}：{:.3f}'.format(names[i], names[j], rho))
    log('运行清单：{}'.format(mpath))


# ----------------------------------------------------------------------------- discover

def walk_dirs(root, max_depth, skip_names=('.git', '__pycache__', 'node_modules', '.ipynb_checkpoints', 'wandb')):
    root = Path(root)
    if not root.exists():
        return
    base = len(root.parts)
    for d, dirs, files in os.walk(str(root), followlinks=False):
        depth = len(Path(d).parts) - base
        dirs[:] = [x for x in dirs if x not in skip_names and not (x.startswith('.') and x != '.cache')]
        if depth >= max_depth:
            dirs[:] = []
        yield Path(d), files


def find_model_dirs(roots, max_depth=9):
    found, seen = [], set()
    for root in roots:
        for d, files in walk_dirs(root, max_depth):
            if 'config.json' not in files:
                continue
            weights = [f for f in files if f.endswith('.safetensors') or f in ('pytorch_model.bin',)
                       or f.endswith('.index.json')]
            if not weights:
                continue
            key = str(d.resolve())
            if key in seen:
                continue
            seen.add(key)
            try:
                c = json.loads((d / 'config.json').read_text(encoding='utf-8'))
            except Exception:  # noqa: BLE001
                continue
            n_labels = c.get('num_labels') or (len(c['id2label']) if isinstance(c.get('id2label'), dict) else None)
            if n_labels is None and any('SequenceClassification' in x for x in (c.get('architectures') or [])):
                n_labels = '2（默认）'
            size = 0
            for f in files:
                if f.endswith(('.safetensors', '.bin')):
                    try:
                        size += (d / f).stat().st_size
                    except OSError:
                        pass
            mml = None
            if (d / 'tokenizer_config.json').exists():
                try:
                    mml = json.loads((d / 'tokenizer_config.json').read_text(encoding='utf-8')).get('model_max_length')
                except Exception:  # noqa: BLE001
                    pass
            extra = [f for f in files if f in ('training_args.bin', 'trainer_state.json', 'args.json',
                                               'train_args.json', 'README.md', 'eval_results.json')]
            found.append({'path': str(d), 'model_type': c.get('model_type'), 'architectures': c.get('architectures') or [],
                          'num_labels': n_labels, 'id2label': c.get('id2label'), 'name_or_path': c.get('_name_or_path'),
                          'size_gb': round(size / 2 ** 30, 2), 'mtime': (d / 'config.json').stat().st_mtime,
                          'tokenizer_model_max_length': mml, 'extra_files': extra})
    return found


CODE_PAT = re.compile(r'(SequenceClassification|CrossEncoder|from_pretrained\(|max_length\s*=|max_len\w*\s*=|'
                      r'truncation\s*=|tokenizer\(|tok\(|num_labels|problem_type|BCEWithLogits|CrossEntropy|'
                      r'context_text|title|seed)')


def scan_code(root, max_files=60, max_lines=60):
    hits = []
    for d, files in walk_dirs(root, 8):
        for f in files:
            if not f.endswith('.py'):
                continue
            p = d / f
            try:
                if p.stat().st_size > 2 * 2 ** 20:
                    continue
                text = p.read_text(encoding='utf-8', errors='ignore')
            except OSError:
                continue
            if not re.search(r'SequenceClassification|CrossEncoder|rerank|Trainer\(', text):
                continue
            lines = [(i + 1, ln.rstrip()) for i, ln in enumerate(text.splitlines()) if CODE_PAT.search(ln)]
            hits.append({'path': str(p), 'mtime': p.stat().st_mtime, 'lines': lines[:max_lines]})
    hits.sort(key=lambda h: (-('stage4' in h['path'] or 'stage5' in h['path']), -h['mtime']))
    return hits[:max_files]


def scan_score_files(root, limit=60):
    out = []
    for d, files in walk_dirs(root, 8):
        for f in files:
            if not re.search(r'(score|rerank|pred)', f, re.I) or not f.endswith(('.jsonl', '.json', '.csv')):
                continue
            p = d / f
            try:
                size = p.stat().st_size
                with p.open(encoding='utf-8', errors='ignore') as fh:
                    head = fh.read(4000)
            except OSError:
                continue
            first = head.splitlines()[0] if head else ''
            keys = None
            if f.endswith('.jsonl'):
                try:
                    obj = json.loads(first)
                    keys = {k: (list(v.keys())[:12] if isinstance(v, dict) else type(v).__name__) for k, v in obj.items()}
                except Exception:  # noqa: BLE001
                    keys = None
            out.append({'path': str(p), 'size_mb': round(size / 2 ** 20, 1),
                        'head': (json.dumps(keys, ensure_ascii=False) if keys else first[:300])})
    return out[:limit]


def propose_config(models, cuda=True):
    ce = [m for m in models if (m['model_type'] or '').startswith('deberta')
          and any('SequenceClassification' in x for x in m['architectures'])]
    best, alternatives = {}, {}
    for m in ce:
        p = m['path'].lower()
        if 'shuffl' in p:
            continue
        fam = 'support' if 'support' in p else ('relevance' if 'relev' in p else None)
        seeds = re.findall(r'(?<!\d)(42|2026|3407)(?!\d)', p)
        if not fam or not seeds:
            continue
        pref = (3 if 'stage4_retraining' in p else 0) + (0 if re.search(r'checkpoint-\d+', p) else 1) \
            + (1 if re.search(r'best|final', p) else 0)
        key = (fam, seeds[-1])
        alternatives.setdefault(key, []).append(m['path'])
        if key not in best or (pref, m['mtime']) > best[key][0]:
            best[key] = ((pref, m['mtime']), m)

    def local(name_parts, hf_id):
        for m in models:
            low = m['path'].lower().replace('_', '-')
            if all(x in low for x in name_parts):
                return m['path'], True
        return hf_id, False

    entries = []
    for fam, method in (('support', 'Support-CE'), ('relevance', 'Relevance-CE')):
        for seed in ('42', '2026', '3407'):
            m = best.get((fam, seed), (None, None))[1]
            entries.append({'name': '{}_ce_s{}'.format(fam, seed), 'method': method, 'seed': int(seed),
                            'kind': 'hf_seqcls', 'path': m['path'] if m else '<请填写 {} 种子 {} 的检查点目录>'.format(method, seed),
                            'max_length': 512, 'dtype': 'fp32', 'positive_index': 1, 'pair_order': 'query_first',
                            'truncation': 'only_second', 'query_prefix': '', 'passage_prefix': '', 'batch_size': 32,
                            'enabled': True,
                            'note': '最大长度、输入顺序须与训练时一致；num_labels={}'.format(m['num_labels'] if m else '?')})
    entries.append({'name': 'bm25', 'method': 'BM25', 'seed': None, 'kind': 'bm25_rank', 'enabled': True})
    bge, bge_local = local(['bge-reranker-v2-m3'], 'BAAI/bge-reranker-v2-m3')
    entries.append({'name': 'bge_reranker_v2_m3', 'method': 'bge-reranker-v2-m3', 'seed': None, 'kind': 'hf_seqcls',
                    'path': bge, 'max_length': 1024, 'dtype': 'fp16', 'positive_index': 0, 'pair_order': 'query_first',
                    'truncation': 'only_second', 'query_prefix': '', 'passage_prefix': '', 'batch_size': 32,
                    'enabled': True})
    qw, qw_local = local(['qwen3-reranker-0.6b'], 'Qwen/Qwen3-Reranker-0.6B')
    entries.append({'name': 'qwen3_reranker_0_6b', 'method': 'Qwen3-Reranker-0.6B', 'seed': None,
                    'kind': 'qwen3_reranker', 'path': qw, 'max_length': 2048, 'dtype': 'fp16',
                    'instruction': QWEN3_INSTRUCTION, 'batch_size': 8, 'enabled': True})
    llm, llm_local = local(['qwen2.5-7b-instruct'], 'Qwen/Qwen2.5-7B-Instruct')
    entries.append({'name': 'llm_judge_qwen2_5_7b', 'method': 'LLM逐点判断(Qwen2.5-7B-Instruct)', 'seed': None,
                    'kind': 'llm_judge', 'path': llm, 'max_length': 2048, 'dtype': 'bf16', 'batch_size': 4,
                    'enabled': bool(llm_local and cuda),
                    'note': '可选方法；本机未找到或没有 GPU 时默认不运行，下载约 15 GB 后可改为 true'})
    return {'device': 'auto', 'models': entries}, alternatives, {'bge': bge_local, 'qwen3': qw_local, 'llm': llm_local}


def cmd_discover(a):
    root = Path(a.root)
    log = Log()
    env = environment()
    log('环境：' + json.dumps(env, ensure_ascii=False))
    checks = []
    for t, spec in TARGETS.items():
        for kind in ('questions', 'pools'):
            p = spec[kind]
            ok = p.exists() and sha256_file(p) == spec['sha256'][kind]
            checks.append((t, kind, str(p), '存在且哈希一致' if ok else ('存在但哈希不符' if p.exists() else '缺失')))
    hf_roots = [os.environ.get('HF_HOME', ''), os.environ.get('HUGGINGFACE_HUB_CACHE', ''),
                os.environ.get('TRANSFORMERS_CACHE', ''), str(Path.home() / '.cache' / 'huggingface')]
    roots = [str(root.parent)] + [x for x in hf_roots if x] + (a.scan or [])
    log('扫描模型目录：' + '；'.join(roots))
    models = find_model_dirs([r for r in roots if Path(r).exists()])
    log('找到含权重的模型目录 {} 个'.format(len(models)))
    code_root = root / 'experiments' if (root / 'experiments').exists() else root
    code = scan_code(code_root)
    scores = scan_score_files(code_root)
    proposal, alternatives, public_local = propose_config(models, cuda=bool(env.get('cuda_available')))

    target_cfg = EVAL_DIR / 'models.json'
    cfg_path = target_cfg if not target_cfg.exists() else EVAL_DIR / 'models.proposed.json'
    cfg_path.write_text(json.dumps(proposal, ensure_ascii=False, indent=2), encoding='utf-8')

    L = ['# discover 报告', '', '生成时间：{}，主机：{}'.format(now(), env.get('host')), '',
         '## 环境', '', '```', json.dumps(env, ensure_ascii=False, indent=2), '```', '']
    tv = env.get('transformers', '')
    if re.match(r'^\d+\.\d+', tv) and tuple(int(x) for x in tv.split('.')[:2]) < (4, 51):
        L += ['**注意**：transformers {} 低于 4.51，Qwen3-Reranker 无法加载。'.format(tv), '']
    L += ['## 输入文件', '', '| 目标 | 文件 | 路径 | 状态 |', '|---|---|---|---|']
    L += ['| {} | {} | `{}` | {} |'.format(*c) for c in checks]
    L += ['', '## 建议的 models.json', '', '已写出：`{}`。**请逐项核对**，尤其是 Support-CE、Relevance-CE 的检查点、'
          '最大长度（max_length）和输入顺序，必须与训练时一致。'.format(cfg_path), '']
    for e in proposal['models']:
        L.append('- `{}`：{}'.format(e['name'], e.get('path', '（无需模型）')))
    L += ['', '公开模型在本机找到：bge={}，Qwen3={}，Qwen2.5-7B={}。未找到的会在运行时从 Hugging Face 下载，'
          '国内可先 `export HF_ENDPOINT=https://hf-mirror.com`。'.format(*[public_local[k] for k in ('bge', 'qwen3', 'llm')]), '']
    L += ['## CE 检查点的候选（每个方法、种子的全部候选目录）', '']
    for (fam, seed), paths in sorted(alternatives.items()):
        L.append('- {} 种子 {}：'.format(fam, seed))
        L += ['  - `{}`'.format(p) for p in paths]
    L += ['', '## 找到的全部模型目录', '', '| 路径 | model_type | 架构 | num_labels | id2label | 大小 GB | 修改时间 | max_length(分词器) | 其他文件 |',
          '|---|---|---|---|---|---|---|---|---|']
    for m in sorted(models, key=lambda x: x['path']):
        L.append('| `{}` | {} | {} | {} | {} | {} | {} | {} | {} |'.format(
            m['path'], m['model_type'], ','.join(m['architectures']), m['num_labels'],
            json.dumps(m['id2label'], ensure_ascii=False) if m['id2label'] else '', m['size_gb'],
            dt.datetime.fromtimestamp(m['mtime']).strftime('%Y-%m-%d %H:%M'), m['tokenizer_model_max_length'],
            ','.join(m['extra_files'])))
    L += ['', '## 训练与打分代码中的相关行（用于核对输入格式与最大长度）', '']
    for h in code:
        L += ['### `{}`'.format(h['path']), '', '```']
        L += ['{:>5}: {}'.format(n, s[:200]) for n, s in h['lines']]
        L += ['```', '']
    L += ['## 既往打分文件（可用于复现核对）', '']
    L += ['- `{}`（{} MB）：{}'.format(s['path'], s['size_mb'], s['head']) for s in scores]
    rpath = EVAL_DIR / 'discover_report.md'
    rpath.write_text('\n'.join(L) + '\n', encoding='utf-8')
    log('已写出：{}'.format(rpath))
    log('已写出：{}'.format(cfg_path))
    for e in proposal['models']:
        log('  {:<24} {}'.format(e['name'], e.get('path', '')))


def cmd_check_calib(a):
    """当前 models.json 中每个启用的方法，是否都已以相同设置通过校准集（权重在封存集运行时再次核对）。"""
    cfg = load_config(a.config, 'sealed')
    cpath = EVAL_DIR / 'scores' / 'calib' / 'RUN_MANIFEST.json'
    if not cpath.exists():
        print('CALIB_MISSING')
        sys.exit(1)
    runs = json.loads(cpath.read_text(encoding='utf-8')).get('runs', {})
    stale = [m['name'] for m in cfg['models'] if m.get('enabled', True) and not (
        runs.get(m['name']) and runs[m['name']].get('status') == 'ok' and runs[m['name']].get('spec_sha256') == spec_hash(m))]
    if stale:
        print('CALIB_STALE ' + ','.join(stale))
        sys.exit(1)
    print('CALIB_CURRENT')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd')
    c = sub.add_parser('check-calib')
    c.add_argument('--config', default=str(EVAL_DIR / 'models.json'))
    d = sub.add_parser('discover')
    d.add_argument('--root', default=DEFAULT_ROOT, help='EEE_research 目录（默认 {}）'.format(DEFAULT_ROOT))
    d.add_argument('--scan', nargs='*', help='另外要扫描的模型目录')
    s = sub.add_parser('score')
    s.add_argument('--target', choices=sorted(TARGETS), required=True)
    s.add_argument('--config', default=str(EVAL_DIR / 'models.json'))
    s.add_argument('--only', default='', help='只运行这些方法（逗号分隔的 name）')
    s.add_argument('--device', default='auto')
    s.add_argument('--questions', default=None, help='（一般不用）覆盖问题文件路径')
    s.add_argument('--pools', default=None, help='（一般不用）覆盖候选池路径')
    s.add_argument('--rerun-reason', default='', help='封存集重跑时必须写明原因；旧结果会归档并记录')
    s.add_argument('--skip-calib-check', action='store_true', help='（不建议）跳过“须先通过校准集”的检查，会记入清单')
    a = ap.parse_args()
    if a.cmd == 'discover':
        cmd_discover(a)
    elif a.cmd == 'check-calib':
        cmd_check_calib(a)
    elif a.cmd == 'score':
        cmd_score(a)
    else:
        ap.print_help()


if __name__ == '__main__':
    main()
