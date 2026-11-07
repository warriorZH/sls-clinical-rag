#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""外部验证集（TCM-BEST4SDT）在服务器上的准备（方案第十六节）。只读题目、语料和训练病例，不运行任何模型。

    python build_external.py --job ext_build

1. 排除近重复病例。与 stage4 训练集病例（CE 的训练数据）字符 2–3 元 TF-IDF 余弦 ≥ 0.85 的病例不纳入；
   集内近重复按编号保留先出现的一例。TF-IDF 与 scripts/build_sealed_testset.py 所用的 sklearn 设置等价
   （analyzer='char'、ngram_range=(2,3)、sublinear_tf、平滑 idf、l2 归一化），这里用 numpy/scipy 实现。
2. 建立候选池。与 scripts/server_build_pools.py 完全相同：同一语料、字符二元组 BM25（k1=1.2，b=0.75）、
   只用题干检索、取 20 条、标题加前后 160 字的上下文。
   封存集按记录编号排除病例自身的原始记录；外部病例在语料中没有编号，改按文本判断，下列任一条成立即排除该候选：
     a. 候选证据包含病例原文字符二元组的比例 ≥ 0.6；
     b. 候选证据含处方中 ≥ 60% 的药名（处方至少 4 味），且上述包含比例 ≥ 0.2；
     c. 候选证据逐字包含标准答案“病因”“病机”中一个 ≥ 10 字的分句，或两个以上 ≥ 6 字的分句。
   被排除的候选由后面的名次依次递补。全部检出候选的上述指标写入 pool_filter_log.jsonl，供敏感性分析。
3. 输出 runs/external/：external_questions.json、external_pools.jsonl、pool_filter_log.jsonl、EXT_MANIFEST.json。
   这些文件只生成一次；再次运行时核对 EXT_MANIFEST 中的哈希后直接结束。
"""
import argparse
import collections
import json
import re
import sqlite3
import sys
import time
from array import array
from pathlib import Path

from common import RUNS, Log, load_config, mark_done, now, read_json, read_jsonl, resolve, sha256_file, write_json

OUT = RUNS / 'external'
HEADING = re.compile(r'^(第[一二三四五六七八九十百0-9]+[章节篇]|[一二三四五六七八九十]+[、．.]|【)')
CLAUSE_SEP = re.compile(r'[，。；;、,：:！？!?\s]')
CHUNK = 200000


# ----------------------------------------------------------------------------- 与 server_build_pools.py 相同的分词

def toks(s):
    return [x[i:i + 2] for x in re.findall(r'[一-鿿A-Za-z]+', s) for i in range(len(x) - 1)]


# ----------------------------------------------------------------------------- TF-IDF（与 sklearn TfidfVectorizer 等价）

_WS2 = re.compile(r'\s\s+')


def char_ngrams(text, lo=2, hi=3):
    """sklearn 的 char 分析器：先转小写，再把连续 2 个以上的空白换成一个空格，然后取 lo..hi 元。"""
    t = _WS2.sub(' ', text.lower())
    n = len(t)
    return [t[i:i + k] for k in range(lo, min(hi + 1, n + 1)) for i in range(n - k + 1)]


def tfidf(texts):
    """返回 l2 归一化的 TF-IDF 稀疏矩阵（行 = 文本）。sublinear_tf：1 + ln(tf)；idf = ln((1+n)/(1+df)) + 1。"""
    import numpy as np
    from scipy import sparse
    vocab, rows, cols, vals = {}, [], [], []
    for i, t in enumerate(texts):
        cnt = collections.Counter(char_ngrams(t))
        for g, c in cnt.items():
            j = vocab.setdefault(g, len(vocab))
            rows.append(i)
            cols.append(j)
            vals.append(c)
    m = sparse.csr_matrix((np.asarray(vals, dtype=np.float64), (rows, cols)), shape=(len(texts), len(vocab)))
    m.sum_duplicates()
    df = np.bincount(m.indices, minlength=len(vocab))
    idf = np.log((1.0 + len(texts)) / (1.0 + df)) + 1.0
    m.data = (np.log(m.data) + 1.0) * idf[m.indices]
    norms = np.sqrt(np.asarray(m.multiply(m).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    return sparse.diags(1.0 / norms) @ m


def background_of(question):
    return question.split('\n')[0].strip()


def dedupe(questions, cfg_ext, log):
    """返回 {qid: (是否纳入, 原因, 与训练病例的最大相似度)}。"""
    import numpy as np
    ds = json.loads(resolve(cfg_ext['stage4_dataset']).read_text(encoding='utf-8'))
    splits = json.loads(resolve(cfg_ext['split_manifest']).read_text(encoding='utf-8'))['task_ids']
    fold = {tid: s for s, ids in splits.items() for tid in ids}
    train = [background_of(t['question']) for t in ds if fold.get(t['id']) == 'train']
    if not train:
        raise RuntimeError('stage4 数据中没有训练集病例')
    log('近重复检查：外部病例 {} 例，CE 训练病例 {} 例'.format(len(questions), len(train)))
    mat = tfidf([q['background'] for q in questions] + train)
    ext, tr = mat[:len(questions)], mat[len(questions):]
    sim_train = np.asarray((ext @ tr.T).max(axis=1).todense()).ravel()
    sim_ext = (ext @ ext.T).toarray()
    thr = float(cfg_ext.get('dup_threshold', 0.85))
    res, kept = {}, []
    for i, q in enumerate(questions):                    # 按编号顺序
        if sim_train[i] >= thr:
            res[q['qid']] = (False, 'near_duplicate_of_training_case', float(sim_train[i]))
            continue
        if kept and max(sim_ext[i, j] for j in kept) >= thr:
            res[q['qid']] = (False, 'near_duplicate_within_external_set', float(sim_train[i]))
            continue
        kept.append(i)
        res[q['qid']] = (True, None, float(sim_train[i]))
    return res, {'n_train_cases': len(train), 'threshold': thr,
                 'max_sim_to_training': {'max': float(sim_train.max()), 'median': float(np.median(sim_train))}}


# ----------------------------------------------------------------------------- BM25（与 server_build_pools.py 相同）

def bm25_matrix(db, words, log):
    import numpy as np
    from scipy import sparse
    wi = {w: i for i, w in enumerate(words)}
    parts, lengths, docids = [], [], []
    rr, cc, vv = array('i'), array('i'), array('f')
    base = 0

    def flush(n_rows):
        nonlocal rr, cc, vv
        m = sparse.csr_matrix((np.frombuffer(vv, dtype=np.float32).copy(),
                               (np.frombuffer(rr, dtype=np.int32).copy(), np.frombuffer(cc, dtype=np.int32).copy())),
                              shape=(n_rows, len(words)))
        parts.append(m)
        rr, cc, vv = array('i'), array('i'), array('f')

    for i, (docid, text) in enumerate(db.execute('SELECT id,text FROM documents ORDER BY id')):
        count = collections.Counter(toks(text or ''))
        lengths.append(sum(count.values()))
        docids.append(docid)
        for w, n in count.items():
            j = wi.get(w)
            if j is not None:
                rr.append(i - base)
                cc.append(j)
                vv.append(n)
        if (i + 1) % CHUNK == 0:
            flush(i + 1 - base)
            base = i + 1
            log('  已扫描语料 {} 条'.format(i + 1))
    flush(len(lengths) - base)
    mat = sparse.vstack(parts, format='csr')
    n = len(lengths)
    df = np.asarray((mat > 0).sum(0)).ravel()
    idf = np.log(1 + (n - df + .5) / (df + .5))
    lens = np.asarray(lengths)
    norm = 1.2 * (.25 + .75 * lens / max(lens.mean(), 1))
    coo = mat.tocoo()
    coo.data = coo.data * 2.2 / (coo.data + norm[coo.row]) * idf[coo.col]
    bm = coo.tocsr()
    return bm, wi, docids


def ranked(bm, wi, docids, query, m):
    """按 BM25 分数取前 m 名（分数相同按文档编号），与 server_build_pools.py 的排序规则相同；边界上同分的全部保留。"""
    import numpy as np
    from scipy import sparse
    idx = [wi[w] for w in set(toks(query))]
    qv = sparse.csr_matrix((np.ones(len(idx)), (idx, np.zeros(len(idx), dtype=int))), shape=(len(wi), 1))
    scores = (bm @ qv).toarray().ravel()
    m = min(m, len(scores))
    thr = np.partition(scores, -m)[-m]
    top = np.nonzero(scores >= thr)[0] if thr > 0 else np.nonzero(scores > 0)[0]
    top = sorted(top, key=lambda i: (-scores[i], int(docids[i])))
    return [(int(docids[i]), float(scores[i])) for i in top]


# ----------------------------------------------------------------------------- 上下文（与 server_build_pools.py 相同）

def attach_context(docs, dataset_root):
    """docs：{document_id: row}。medical_train 片段保留原文；其余从原始文件取标题和前后 160 字。"""
    bysource = collections.defaultdict(list)
    for row in docs.values():
        if row['source'].startswith('medical_train'):
            row['context_text'] = row['text']
            row['context_status'] = 'original_medical_train_fragment'
        else:
            bysource[row['source']].append(row)
    for source, group in bysource.items():
        group.sort(key=lambda r: r['start_char'])
        j, offset, heading = 0, 0, None
        path = Path(dataset_root) / source
        if not path.exists():
            for row in group:
                row['context_text'] = row['text']
                row['context_status'] = 'source_file_missing_original_retained'
            continue
        with path.open(encoding='utf-8-sig') as f:
            for line in f:
                cleaned = line.strip()
                if 2 <= len(cleaned) <= 60 and HEADING.match(cleaned):
                    heading = cleaned
                while j < len(group) and group[j]['start_char'] < offset + len(line):
                    row = group[j]
                    pos = row['start_char'] - offset
                    if pos >= 0:
                        context = line[max(0, pos - 160):min(len(line), pos + len(row['text']) + 160)].strip()
                        row['title'] = heading
                        row['context_text'] = ((heading + '\n') if heading else '') + context
                        row['context_status'] = ('source_line_context_title_present' if heading
                                                 else 'source_line_context_title_unavailable')
                    else:
                        row['context_text'] = row['text']
                        row['context_status'] = 'offset_unresolved_original_retained'
                    j += 1
                offset += len(line)
                if j == len(group):
                    break
        for row in group[j:]:
            row['context_text'] = row['text']
            row['context_status'] = 'offset_unresolved_original_retained'


# ----------------------------------------------------------------------------- 排除病例自身的原始记录

def leak_features(case_bigrams, herb_keys, clauses, evidence):
    eb = set(toks(evidence))
    cont = len(case_bigrams & eb) / len(case_bigrams) if case_bigrams else 0.0
    herb = None
    if herb_keys:
        herb = sum(1 for keys in herb_keys if any(k in evidence for k in keys)) / len(herb_keys)
    flat = re.sub(r'\s+', '', evidence)
    hits = [c for c in clauses if c in flat]
    return {'containment': round(cont, 4), 'herb_frac': None if herb is None else round(herb, 4),
            'n_herbs': len(herb_keys), 'clause_hits': hits}


def leak_reason(f, th):
    if f['containment'] >= th['containment']:
        return 'a_case_text'
    if (f['n_herbs'] >= th['herb_min'] and f['herb_frac'] is not None and f['herb_frac'] >= th['herb_frac']
            and f['containment'] >= th['herb_containment']):
        return 'b_prescription'
    hits = f['clause_hits']
    if any(len(c) >= th['clause_long_chars'] for c in hits) or len(hits) >= th['clause_min_hits']:
        return 'c_reference_clause'
    return None


def ref_clauses(ref, min_chars):
    out = []
    for field in ('cause', 'mechanism'):
        for c in CLAUSE_SEP.split(ref.get(field) or ''):
            c = c.strip()
            if len(c) >= min_chars and c not in out:
                out.append(c)
    return out


# ----------------------------------------------------------------------------- 主流程

def outputs_ok(log):
    man = read_json(OUT / 'EXT_MANIFEST.json')
    if not man:
        return None
    for name, h in man['outputs'].items():
        p = OUT / name
        if not p.exists() or sha256_file(p) != h:
            log('{} 与 EXT_MANIFEST.json 记录的哈希不符'.format(name))
            return False
    return man


def build(cfg, log):
    import numpy as np
    ce = cfg['external']
    th = dict({'containment': 0.6, 'herb_frac': 0.6, 'herb_min': 4, 'herb_containment': 0.2, 'clause_min_chars': 6,
               'clause_long_chars': 10, 'clause_min_hits': 2}, **(ce.get('leak') or {}))
    k = int(ce.get('k', 20))
    t0 = time.time()
    qa_path, ref_path = resolve(ce['questions_all']), resolve(ce['reference'])
    questions = sorted(json.loads(qa_path.read_text(encoding='utf-8')), key=lambda q: q['qid'])
    refs = {r['qid']: r for r in read_jsonl(ref_path)}
    # 1. 近重复
    dd, dd_info = dedupe(questions, ce, log)
    for q in questions:
        inc, why, sim = dd[q['qid']]
        q['included'] = inc
        q['exclusion'] = why
        q['max_sim_to_training'] = round(sim, 4)
    included = [q for q in questions if q['included']]
    log('纳入 {} 例，排除 {} 例：{}'.format(len(included), len(questions) - len(included),
                                      dict(collections.Counter(q['exclusion'] for q in questions if not q['included']))))
    # 2. BM25
    corpus = resolve(ce['corpus'])
    db = sqlite3.connect('file:{}?mode=ro'.format(corpus), uri=True)
    n_docs = db.execute('SELECT COUNT(*) FROM documents').fetchone()[0]
    words = sorted({w for q in included for w in toks(q['question_final'])})
    log('BM25：语料 {} 条，查询词 {} 个'.format(n_docs, len(words)))
    bm, wi, docids = bm25_matrix(db, words, log)
    dataset_root = resolve(ce['dataset_root'])
    info = {q['qid']: (set(toks(q['background'])), (refs.get(q['qid']) or {}).get('herb_keys') or [],
                       ref_clauses(refs.get(q['qid']) or {}, int(th['clause_min_chars']))) for q in included}
    qtext = {q['qid']: q['question_final'] for q in included}
    pending = {q['qid']: k + 30 for q in included}
    all_docs, results, rnd = {}, {}, 0
    while pending:
        rnd += 1
        cands = {}
        for i, (qid, m) in enumerate(sorted(pending.items()), 1):
            cands[qid] = (m, ranked(bm, wi, docids, qtext[qid], m))
            if i % 50 == 0:
                log('  检索 {}/{}（第 {} 轮）'.format(i, len(pending), rnd))
        new = {}
        for qid, (m, c) in cands.items():
            for d, _ in c:
                if d not in all_docs and d not in new:
                    src, s0, s1, text, h = db.execute(
                        'SELECT source,start_char,end_char,text,sha256 FROM documents WHERE id=?', (d,)).fetchone()
                    new[d] = {'document_id': d, 'source': src, 'start_char': s0, 'end_char': s1, 'sha256': h,
                              'text': text, 'title': None, 'context_text': None}
        attach_context(new, dataset_root)              # 每个原始文件只读一遍，与 server_build_pools.py 相同
        all_docs.update(new)
        nxt = {}
        for qid, (m, c) in cands.items():
            case_bg, hk, clauses = info[qid]
            kept, log_rows = [], []
            for pos, (d, score) in enumerate(c, 1):
                row = all_docs[d]
                f = leak_features(case_bg, hk, clauses, row['context_text'] or row['text'] or '')
                why = leak_reason(f, th)
                take = why is None and len(kept) < k
                log_rows.append(dict(f, qid=qid, bm25_rank=pos, document_id=d, excluded=why, kept=take))
                if take:
                    kept.append(dict(row, bm25_score=score))
            if len(kept) < k and m < len(docids):
                nxt[qid] = m * 4
            else:
                results[qid] = (kept, log_rows)
        pending = nxt
    rows, flog, reasons = [], [], collections.Counter()
    for qid in sorted(results):
        kept, log_rows = results[qid]
        if len(kept) < k:
            raise RuntimeError('{} 排除后不足 {} 条候选'.format(qid, k))
        for r, row in enumerate(kept, 1):
            rows.append({'qid': qid, 'rank': r, 'document_id': row['document_id'], 'source': row['source'],
                         'start_char': row['start_char'], 'end_char': row['end_char'], 'sha256': row['sha256'],
                         'bm25_score': row['bm25_score'], 'text': row['text'], 'title': row['title'],
                         'context_text': row['context_text'], 'context_status': row['context_status']})
        last_kept = max(x['bm25_rank'] for x in log_rows if x['kept'])
        for x in log_rows:
            if x['bm25_rank'] <= last_kept:
                flog.append(x)
                if x['excluded']:
                    reasons[x['excluded']] += 1
    db.close()
    # 3. 写出
    OUT.mkdir(parents=True, exist_ok=True)
    qp, pp, lp = OUT / 'external_questions.json', OUT / 'external_pools.jsonl', OUT / 'pool_filter_log.jsonl'
    qp.write_text(json.dumps(questions, ensure_ascii=False, indent=1), encoding='utf-8')
    with pp.open('w', encoding='utf-8') as f:
        for row in sorted(rows, key=lambda r: (r['qid'], r['rank'])):
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    with lp.open('w', encoding='utf-8') as f:
        for row in flog:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    excl_q = collections.Counter(x['qid'] for x in flog if x['excluded'])
    kept_cont = [x['containment'] for x in flog if x['kept']]
    man = {
        'created': now(),
        'questions_all': len(questions), 'questions_included': len(included),
        'excluded': {q['qid']: {'reason': q['exclusion'], 'max_sim_to_training': q['max_sim_to_training']}
                     for q in questions if not q['included']},
        'dedupe': dd_info,
        'by_stratum_and_type': dict(collections.Counter('{}/{}'.format(q['stratum'], q['question_type'])
                                                        for q in included)),
        'pairs': len(rows), 'k': k, 'corpus': str(corpus), 'corpus_docs': n_docs,
        'retrieval': 'BM25 char bigrams k1=1.2 b=0.75 over the full corpus; query = final question text only '
                     '(same as scripts/server_build_pools.py)',
        'reference_used_for_retrieval': False,
        'own_record_filter': {'thresholds': th, 'candidates_excluded': dict(reasons),
                              'questions_with_exclusions': len(excl_q),
                              'kept_containment': {'max': max(kept_cont) if kept_cont else None,
                                                   'median': float(np.median(kept_cont)) if kept_cont else None}},
        'contexts': dict(collections.Counter(r['context_status'] for r in rows)),
        'seconds': round(time.time() - t0, 1),
        'inputs': {'questions_all': sha256_file(qa_path), 'reference': sha256_file(ref_path),
                   'stage4_dataset': sha256_file(resolve(ce['stage4_dataset'])),
                   'split_manifest': sha256_file(resolve(ce['split_manifest']))},
        'script_sha256': sha256_file(Path(__file__)),
        'outputs': {qp.name: sha256_file(qp), pp.name: sha256_file(pp), lp.name: sha256_file(lp)},
    }
    write_json(OUT / 'EXT_MANIFEST.json', man)
    log('外部验证集：纳入 {} 题，候选 {} 对，排除自身记录候选 {}，用时 {:.0f} 秒'.format(
        len(included), len(rows), dict(reasons), time.time() - t0))
    return man


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--job', required=True)
    a = ap.parse_args()
    cfg = load_config()
    log = Log(RUNS / 'logs' / 'build_external.log')
    log('=' * 60)
    if not cfg.get('external'):
        log('配置中没有 external 一节')
        sys.exit(3)
    man = outputs_ok(log)
    if man is False:
        log('runs/external/ 中的文件与清单不符。为保证外部候选池只生成一次，本作业不会覆盖它们；请把本日志发给 Claude。')
        sys.exit(3)
    if man is None:
        man = build(cfg, log)
    else:
        log('外部验证集已生成（{}），核对哈希一致，直接结束'.format(man['created']))
    mark_done(a.job, {'questions_included': man['questions_included'], 'pairs': man['pairs'],
                      'own_record_excluded': man['own_record_filter']['candidates_excluded']})


if __name__ == '__main__':
    main()
