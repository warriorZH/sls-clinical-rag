#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""E1 / E2 重排打分（eval_v2）。

    python score_v2.py --stage e1_ce  --target calib|sealed --job <作业名>
    python score_v2.py --stage e1_pub --target calib|sealed --job <作业名>
    python score_v2.py --stage e2     --target calib|sealed --job <作业名>
    python score_v2.py --stage ext    --target external     --job <作业名>    外部验证集（方案第十六节）

打分函数直接复用 eval_v1/server_score.py（已在服务器上验证过的实现）：
  - CE（Support-CE / Relevance-CE）：B = 完整题干、512 窗口；C = 不截断（≤1024）。A 编码的分数沿用 eval_v1。
  - 公开模型的“文本对等”档：先按 CE 的 SentencePiece 规则截断题干与证据，再把截断后的原文交给 bge / Qwen3-Reranker。
  - 每个方法每个目标集只运行一次；封存集运行前要求同一设置在校准集上通过。
  - 外部验证集：CE 用 B 编码（设置与校准集上已通过的相同）；bge 与 Qwen3-Reranker-0.6B 用 eval_v1 的完整输入设置，
    先在校准集上重算并与 eval_v1 的校准集分数核对，通过后才给外部验证集打分。
只读题目与候选池，不读任何证据标签。
"""
import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

from common import (EVAL_V1, RUNS, Log, environment, load_config, load_target, mark_done, now, pick_device, read_json,
                    registered_path, resolve, sha256_file, sha256_text, write_json, evidence_text)
import conditions

SPEC_KEYS = ('name', 'kind', 'objective', 'seed', 'encoding', 'match', 'path', 'model_key', 'base_model', 'spm',
             'query_max_tokens', 'max_length', 'micro', 'dtype', 'instruction')


def load_v1():
    p = EVAL_V1 / 'server_score.py'
    spec = importlib.util.spec_from_file_location('server_score_v1', str(p))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def spec_hash(m):
    return sha256_text(json.dumps({k: m.get(k) for k in SPEC_KEYS}, ensure_ascii=False, sort_keys=True))


# ----------------------------------------------------------------------------- 截断与文本对等

def prefix_within(sp, text, limit):
    """返回 text 的最长前缀，使其 SentencePiece 词元数不超过 limit。优先用词元的字节偏移，失败时二分查找。"""
    if limit <= 0:
        return ''
    ids = sp.encode(text)
    if len(ids) <= limit:
        return text
    try:
        proto = sp.encode(text, out_type='immutable_proto')
        end = proto.pieces[limit - 1].end
        cut = text.encode('utf-8')[:end].decode('utf-8', errors='ignore')
        if len(sp.encode(cut)) <= limit:
            return cut
    except Exception:  # noqa: BLE001
        pass
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(sp.encode(text[:mid])) <= limit:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo]


def ce_rows(sp, pairs, qmax, total):
    """与 E3 / eval_v1 相同的编码规则：[1] + q[:qmax] + [2] + e[:total-3-len(q)] + [2]。同时记录截断情况，
    以及实际送入模型的题干、证据词元数（外部验证集有很长的病例，CE 留给证据的长度可能很少，供敏感性分析用）。"""
    rows, flags = [], []
    for x in pairs:
        q, e = sp.encode(x['query']), sp.encode(x['evidence'])
        qt = q[:qmax]
        et = e[:max(0, total - 3 - len(qt))]
        rows.append([1] + qt + [2] + et + [2])
        flags.append({'q_trunc': len(qt) < len(q), 'e_trunc': len(et) < len(e), 'q_tokens': len(qt),
                      'e_tokens': len(et)})
    return rows, flags


def matched_pairs(sp, pairs, enc):
    """把题干、证据截成 CE 在编码 enc 下实际看到的那一段原文。"""
    qmax = 160 if enc == 'A' else 100000
    total = 512
    out, flags = [], []
    for x in pairs:
        q_ids = sp.encode(x['query'])
        q_text = prefix_within(sp, x['query'], qmax) if len(q_ids) > qmax else x['query']
        q_len = min(len(q_ids), qmax)
        budget = total - 3 - q_len
        e_ids = sp.encode(x['evidence'])
        e_text = prefix_within(sp, x['evidence'], budget) if len(e_ids) > budget else x['evidence']
        out.append(dict(x, query=q_text, evidence=e_text))
        flags.append({'q_trunc': q_text != x['query'], 'e_trunc': e_text != x['evidence']})
    return out, flags


# ----------------------------------------------------------------------------- 打分

def read_scores(path):
    out = {}
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if line.strip():
            r = json.loads(line)
            out[(r['qid'], str(r['document_id']))] = float(r['score'])
    return out


def write_scores(path, pairs, scores, flags=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', encoding='utf-8') as f:
        for i, (p, s) in enumerate(zip(pairs, scores)):
            r = {'qid': p['qid'], 'document_id': p['document_id'], 'rank': p['rank'], 'score': float(s)}
            if flags:
                r.update(flags[i])
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    tmp.replace(path)


def max_diff_vs(path_ref, pairs, scores, mask):
    if not Path(path_ref).exists():
        return None
    ref = read_scores(path_ref)
    diffs = [abs(s - ref[(p['qid'], p['document_id'])]) for p, s, m in zip(pairs, scores, mask)
             if m and (p['qid'], p['document_id']) in ref]
    return {'n': len(diffs), 'max_abs_diff': max(diffs) if diffs else 0.0}


def run_ce(v1, methods, pairs, target, device, log, manifest, out_dir):
    """按检查点分组：每个检查点只加载一次，依次计算 B、C 两种编码。"""
    import sentencepiece as spm
    groups = {}
    for m in methods:
        groups.setdefault(m['path'], []).append(m)
    for path, ms in groups.items():
        todo = [m for m in ms if not done_ok(manifest, m)]
        if not todo:
            continue
        log('-' * 60)
        log('CE 检查点 {}  （{}）'.format(path, ', '.join(m['name'] for m in todo)))
        ck = Path(path)
        if not ck.is_file():
            raise FileNotFoundError('找不到检查点 {}'.format(ck))
        wsha = {ck.name: sha256_file(ck)}
        spec0 = dict(todo[0], _target=target, _weights_sha256=wsha)
        model, meta = v1.eee_load(spec0, device)
        sp = spm.SentencePieceProcessor(model_file=str(todo[0]['spm']))
        repro = None
        if target == 'calib' and todo[0].get('repro'):
            try:
                repro = v1.eee_repro(model, sp, todo[0]['repro'], spec0, device, log)
            except Exception as e:  # noqa: BLE001
                repro = {'passed': False, 'error': '{}: {}'.format(type(e).__name__, e)}
        for m in todo:
            t0 = time.time()
            rec = base_record(m, target, device)
            rec.update(weights_sha256=wsha, checkpoint_meta=meta)
            try:
                if repro is not None:
                    rec['repro_vs_e3'] = repro
                    if not repro.get('passed'):
                        raise RuntimeError('与 E3 分数的复现核对未通过')
                rows, flags = ce_rows(sp, pairs, int(m['query_max_tokens']), int(m['max_length']))
                scores = [float(x) for x in v1.eee_predict(model, rows, int(m.get('micro', 4)), device)]
                check_finite(scores, pairs)
                # 有效性核对：题干与证据在 A 编码下都没有截断的输入，分数应与 eval_v1（A 编码）相同
                rows_a, flags_a = ce_rows(sp, pairs, 160, 512)
                mask = [not (fa['q_trunc'] or fa['e_trunc']) and not (f['q_trunc'] or f['e_trunc'])
                        for fa, f in zip(flags_a, flags)]
                ref = EVAL_V1 / 'scores' / target / '{}_ce_s{}.jsonl'.format(m['objective'], m['seed'])
                chk = max_diff_vs(ref, pairs, scores, mask)
                rec['check_vs_v1_A'] = chk
                if chk is not None and chk['n'] and chk['max_abs_diff'] > 1e-3:
                    raise RuntimeError('未截断输入的分数与 eval_v1 不一致（最大差 {:.2e}）'.format(chk['max_abs_diff']))
                out = out_dir / '{}.jsonl'.format(m['name'])
                write_scores(out, pairs, scores, flags)
                finish_record(rec, scores, flags, out, t0)
                log('  {}：完成，题干截断 {} 条，证据截断 {} 条，与 eval_v1 未截断输入最大差 {}'.format(
                    m['name'], rec['n_q_trunc'], rec['n_e_trunc'],
                    '{:.2e}'.format(chk['max_abs_diff']) if chk else '无参照'))
            except Exception as e:  # noqa: BLE001
                rec.update(status='failed', error='{}: {}'.format(type(e).__name__, e), seconds=round(time.time() - t0, 1))
                log('  {}：失败 {}'.format(m['name'], rec['error']))
            manifest['runs'][m['name']] = rec
            save_manifest(out_dir, manifest, m['name'])
        del model
        free_cuda()


def run_other(v1, methods, pairs, target, device, log, manifest, out_dir, spm_path):
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=str(spm_path)) if spm_path else None
    for m in methods:
        if done_ok(manifest, m):
            continue
        log('-' * 60)
        log('{}  kind={}  match={}'.format(m['name'], m['kind'], m.get('match', '无')))
        t0 = time.time()
        rec = base_record(m, target, device)
        try:
            if m.get('model_key'):
                local = registered_path(m['model_key'])
            else:
                local = resolve(m['path'])
                if not local.exists():
                    local, _ = v1.resolve_model(m['path'], log)
            rec['resolved_path'] = str(local)
            log('  计算权重 SHA256 …')
            rec['weights_sha256'] = v1.weights_fingerprint(local)
            use_pairs, flags = pairs, None
            if m.get('match'):
                use_pairs, flags = matched_pairs(sp, pairs, m['match'])
            spec = dict(m, _local_path=local, _target=target)
            scorer = {'hf_seqcls': v1.score_seqcls, 'qwen3_reranker': v1.score_qwen3,
                      'llm_judge': v1.score_llm_judge}[m['kind']]
            # 显存不足时把批量减半重试（外部验证的病例比封存集长得多）。批量只影响浮点舍入，
            # eval_v1 已规定它不计入设置指纹；实际用的批量记入运行记录。
            bs = int(spec.get('batch_size') or 8)
            while True:
                oom = False
                try:
                    scores, info = scorer(dict(spec, batch_size=bs), use_pairs, device, log)
                except Exception as e:  # noqa: BLE001
                    if 'out of memory' not in str(e).lower() or bs <= 1:
                        raise
                    oom = True
                if not oom:
                    break
                free_cuda()                  # 离开 except 块之后，异常引用的模型和张量才能释放
                bs = max(1, bs // 2)
                log('  显存不足，批量减为 {} 后重试'.format(bs))
            rec['batch_size_used'] = bs
            scores = [float(s) for s in scores]
            check_finite(scores, pairs)
            rec['info'] = info
            ref_name = m.get('v1_ref')
            if m.get('match'):
                ref_name = 'bge_reranker_v2_m3' if m['kind'] == 'hf_seqcls' else 'qwen3_reranker_0_6b'
            if ref_name and target in ('calib', 'sealed'):
                mask = [not (f['q_trunc'] or f['e_trunc']) for f in flags] if flags else [True] * len(pairs)
                rec['check_vs_v1_N'] = max_diff_vs(EVAL_V1 / 'scores' / target / '{}.jsonl'.format(ref_name),
                                                   pairs, scores, mask)
                chk = rec['check_vs_v1_N']
                if chk is not None and chk['n'] and chk['max_abs_diff'] > 1e-3:
                    raise RuntimeError('未截断输入的分数与 eval_v1 不一致（最大差 {:.2e}）'.format(chk['max_abs_diff']))
            out = out_dir / '{}.jsonl'.format(m['name'])
            write_scores(out, pairs, scores, flags)
            finish_record(rec, scores, flags, out, t0)
            log('  完成：{} 条，用时 {} 秒'.format(len(scores), rec['seconds']))
        except Exception as e:  # noqa: BLE001
            rec.update(status='failed', error='{}: {}'.format(type(e).__name__, e), seconds=round(time.time() - t0, 1))
            log('  失败：{}'.format(rec['error']))
        manifest['runs'][m['name']] = rec
        save_manifest(out_dir, manifest, m['name'])
        free_cuda()


# ----------------------------------------------------------------------------- 记录

def base_record(m, target, device):
    return {'name': m['name'], 'kind': m['kind'], 'spec': {k: m.get(k) for k in SPEC_KEYS}, 'spec_sha256': spec_hash(m),
            'target': target, 'started': now(), 'device': device, 'env': environment(),
            'script_sha256': sha256_file(Path(__file__)), 'optional': bool(m.get('optional'))}


def finish_record(rec, scores, flags, out, t0):
    rec.update(status='ok', n_pairs=len(scores), score_min=min(scores), score_max=max(scores),
               scores_sha256=sha256_file(out), finished=now(), seconds=round(time.time() - t0, 1))
    if flags:
        rec['n_q_trunc'] = sum(f['q_trunc'] for f in flags)
        rec['n_e_trunc'] = sum(f['e_trunc'] for f in flags)
    else:
        rec['n_q_trunc'] = rec['n_e_trunc'] = None


def check_finite(scores, pairs):
    bad = [i for i, s in enumerate(scores) if not math.isfinite(s)]
    if len(scores) != len(pairs) or bad:
        raise ValueError('分数数目 {}（应为 {}），非有限值 {} 个'.format(len(scores), len(pairs), len(bad)))


def done_ok(manifest, m):
    r = manifest['runs'].get(m['name'])
    return bool(r and r.get('status') == 'ok' and r.get('spec_sha256') == spec_hash(m))


def save_manifest(out_dir, manifest, name):
    """同一目标集的几个打分作业可能同时运行（如校准集上的 CE、公开模型与外部验证的校准核对），
    它们共用一个 RUN_MANIFEST.json：加文件锁，重新读取后只写入本作业这一条记录，避免互相覆盖。"""
    import fcntl
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(str(out_dir / 'RUN_MANIFEST.lock'), 'w') as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        try:
            cur = read_json(out_dir / 'RUN_MANIFEST.json') or {k: v for k, v in manifest.items() if k != 'runs'}
            cur.setdefault('runs', {})[name] = manifest['runs'][name]
            write_json(out_dir / 'RUN_MANIFEST.json', cur)
        finally:
            fcntl.flock(lk, fcntl.LOCK_UN)


def free_cuda():
    try:
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


# ----------------------------------------------------------------------------- 主程序

def target_pairs(cfg, target):
    questions, pools = load_target(cfg, target)
    qmap = {q['qid']: q for q in questions}
    pairs = [{'qid': qid, 'document_id': c['document_id'], 'rank': c['rank'], 'query': qmap[qid]['question_final'],
              'evidence': evidence_text(c)} for qid in sorted(pools) for c in pools[qid]]
    return questions, pairs


def v1_spm():
    for m in conditions.v1_models().values():
        if m.get('spm'):
            return m['spm']
    return None


def score_target(v1, methods, cfg, target, device, log, gate_calib):
    """给一个目标集打分；gate_calib 为真时，只运行在校准集上以相同设置成功运行过的方法。返回该目标集的运行记录。"""
    out_dir = RUNS / 'scores' / target
    manifest = read_json(out_dir / 'RUN_MANIFEST.json') or {'target': target, 'created': now(), 'runs': {}}
    questions, pairs = target_pairs(cfg, target)
    log('=' * 60)
    log('打分 target={} device={} 题数={} 输入对={} 方法={}'.format(target, device, len(questions), len(pairs),
                                                           ','.join(m['name'] for m in methods)))
    if gate_calib:
        calib = (read_json(RUNS / 'scores' / 'calib' / 'RUN_MANIFEST.json') or {}).get('runs', {})
        keep = []
        for m in methods:
            c = calib.get(m['name'])
            if c and c.get('status') == 'ok' and c.get('spec_sha256') == spec_hash(m):
                keep.append(m)
            else:
                log('[跳过] {}：校准集上没有相同设置的成功运行'.format(m['name']))
                if m['name'] not in manifest['runs']:
                    manifest['runs'][m['name']] = {'status': 'skipped_no_calib', 'spec_sha256': spec_hash(m)}
                    save_manifest(out_dir, manifest, m['name'])
        methods = keep
    ce = [m for m in methods if m['kind'] == 'ce_v2']
    other = [m for m in methods if m['kind'] != 'ce_v2']
    if ce:
        run_ce(v1, ce, pairs, target, device, log, manifest, out_dir)
    if other:
        run_other(v1, other, pairs, target, device, log, manifest, out_dir, v1_spm())
    return manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=['e1_ce', 'e1_pub', 'e2', 'ext'])
    ap.add_argument('--target', required=True, choices=['calib', 'sealed', 'external'])
    ap.add_argument('--job', required=True)
    a = ap.parse_args()
    if (a.stage == 'ext') != (a.target == 'external'):
        raise SystemExit('外部验证集只能用 --stage ext 打分')
    cfg = load_config()
    log = Log(RUNS / 'logs' / 'score_{}_{}.log'.format(a.stage, a.target))
    methods = conditions.score_methods(cfg, a.stage)
    v1 = load_v1()
    device = pick_device()
    if a.stage == 'ext':
        # 公开模型的完整输入设置先在校准集上重算，并与 eval_v1 的校准集分数核对（CE 的 B 编码已由 e1_calib_ce 核对）
        score_target(v1, [m for m in methods if m['kind'] != 'ce_v2'], cfg, 'calib', device, log, gate_calib=False)
        manifest = score_target(v1, methods, cfg, 'external', device, log, gate_calib=True)
    else:
        manifest = score_target(v1, methods, cfg, a.target, device, log, gate_calib=(a.target == 'sealed'))
    names = [m['name'] for m in methods]
    optional = {m['name'] for m in methods if m.get('optional')}
    status = {n: (manifest['runs'].get(n) or {}).get('status') for n in names}
    ok = [n for n in names if status[n] == 'ok']
    failed_required = [n for n in names if status[n] != 'ok' and n not in optional]
    log('完成 {}/{}；必需方法失败：{}'.format(len(ok), len(names), ', '.join(failed_required) or '无'))
    if failed_required:
        sys.exit(1)
    mark_done(a.job, {'ok': ok, 'optional_failed': [n for n in names if n not in ok]})


if __name__ == '__main__':
    main()
