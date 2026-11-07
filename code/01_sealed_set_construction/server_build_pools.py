"""Run ON THE SERVER after Phase 1 is finalised: build the frozen BM25 candidate pool
for every included sealed question.

It reuses the retrieval settings of experiments/stage5_real_retrieval_v1_2/build_pools.py
(same corpus SQLite, character-bigram BM25 k1=1.2 b=0.75, question-only query,
heading + surrounding source text as context) so results stay comparable with E3.
No reference answer is used for retrieval and no model is run here.

Usage on the server:
    cd /home/<user>/UUM/EEE_research/expert_testset_v1
    python scripts/server_build_pools.py --check            # 1 minute: inputs and corpus schema
    python scripts/server_build_pools.py --k 20              # full run, writes sealed_pools.jsonl
"""
import argparse, collections, hashlib, json, re, sqlite3, sys, time
from pathlib import Path

import numpy as np
from scipy import sparse

SEALED = '_sealed_负责人保存_勿发给标注员'


def toks(s):
    return [x[i:i + 2] for x in re.findall(r'[一-鿿A-Za-z]+', s) for i in range(len(x) - 1)]


def own_record_index(source):
    m = re.match(r'medical_train:(\d+)', source or '')
    return int(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/home/<user>/UUM/EEE_research')
    ap.add_argument('--testset', default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument('--corpus', default=None, help='retrieval_corpus.sqlite (default: <root>/experiments/stage4_retraining/actual_retrieval/)')
    ap.add_argument('--k', type=int, default=20)
    ap.add_argument('--check', action='store_true')
    a = ap.parse_args()
    root, testset = Path(a.root), Path(a.testset)
    corpus = Path(a.corpus) if a.corpus else root / 'experiments/stage4_retraining/actual_retrieval/retrieval_corpus.sqlite'
    sealed = testset / SEALED
    final = [q for q in json.loads((sealed / 'final_questions.json').read_text(encoding='utf-8')) if q['included']]
    meta = {q['qid']: q for q in json.loads((sealed / 'sealed_questions_full.json').read_text(encoding='utf-8'))}
    out_path = sealed / 'sealed_pools.jsonl'
    if out_path.exists() and not a.check:
        sys.exit(f'{out_path} 已存在。封存候选池只能生成一次；如确需重建，请先改名备份旧文件并在 README 中记录原因。')

    db = sqlite3.connect(f'file:{corpus}?mode=ro', uri=True)
    n_docs = db.execute('SELECT COUNT(*) FROM documents').fetchone()[0]
    cols = [r[1] for r in db.execute('PRAGMA table_info(documents)')]
    need = {'id', 'source', 'start_char', 'end_char', 'text', 'sha256'}
    if not need <= set(cols):
        sys.exit(f'语料表字段不符，缺少 {need - set(cols)}')
    sources = [r[0] for r in db.execute('SELECT DISTINCT source FROM documents LIMIT 50')]
    print(json.dumps({'questions': len(final), 'corpus_docs': n_docs, 'columns': cols, 'example_sources': sources[:5]}, ensure_ascii=False))
    if a.check:
        return

    start = time.monotonic()
    words = sorted({w for q in final for w in toks(q['question_final'])})
    wi = {w: i for i, w in enumerate(words)}
    rr, cc, vv, lengths, docids = [], [], [], [], []
    for i, (docid, text) in enumerate(db.execute('SELECT id,text FROM documents ORDER BY id')):
        count = collections.Counter(toks(text))
        lengths.append(sum(count.values()))
        docids.append(docid)
        for w, n in count.items():
            if w in wi:
                rr.append(i); cc.append(wi[w]); vv.append(n)
        if (i + 1) % 200000 == 0:
            print('SCAN', i + 1, flush=True)
    mat = sparse.csr_matrix((np.asarray(vv, dtype=np.float32), (rr, cc)), shape=(len(lengths), len(words)))
    del rr, cc, vv
    n = len(lengths)
    df = np.asarray((mat > 0).sum(0)).ravel()
    idf = np.log(1 + (n - df + .5) / (df + .5))
    lens = np.asarray(lengths)
    norm = 1.2 * (.25 + .75 * lens / max(lens.mean(), 1))
    coo = mat.tocoo()
    coo.data = coo.data * 2.2 / (coo.data + norm[coo.row]) * idf[coo.col]
    bm = coo.tocsr()
    del coo, mat

    rows, skipped_self = [], 0
    for q in final:
        idx = [wi[w] for w in set(toks(q['question_final']))]
        qv = sparse.csr_matrix((np.ones(len(idx)), (idx, np.zeros(len(idx), dtype=int))), shape=(len(words), 1))
        scores = (bm @ qv).toarray().ravel()
        top = np.argpartition(scores, -(a.k + 10))[-(a.k + 10):]
        top = sorted(top, key=lambda i: (-scores[i], int(docids[i])))
        own = meta.get(q['qid'], {}).get('stage1_index')
        rank = 0
        for index in top:
            source, s0, s1, text, h = db.execute('SELECT source,start_char,end_char,text,sha256 FROM documents WHERE id=?',
                                                 (int(docids[index]),)).fetchone()
            if own is not None and own_record_index(source) == own:
                skipped_self += 1
                continue
            rank += 1
            rows.append({'qid': q['qid'], 'rank': rank, 'document_id': int(docids[index]), 'source': source,
                         'start_char': s0, 'end_char': s1, 'sha256': h, 'bm25_score': float(scores[index]),
                         'text': text, 'title': None, 'context_text': None})
            if rank == a.k:
                break
        print('QUERY', q['qid'], flush=True)
    db.close()

    # Context: authentic heading + up to 160 characters around the fragment, read from the
    # original source file (same rule as build_pools.py).  Medical-record fragments keep their text.
    bysource = collections.defaultdict(list)
    for row in rows:
        if row['source'].startswith('medical_train'):
            row['context_text'] = row['text']; row['context_status'] = 'original_medical_train_fragment'
        else:
            bysource[row['source']].append(row)
    for source, group in bysource.items():
        group.sort(key=lambda r: r['start_char'])
        j, offset, heading = 0, 0, None
        with (root / 'dataset' / source).open(encoding='utf-8-sig') as f:
            for line in f:
                cleaned = line.strip()
                if 2 <= len(cleaned) <= 60 and re.match(r'^(第[一二三四五六七八九十百0-9]+[章节篇]|[一二三四五六七八九十]+[、．.]|【)', cleaned):
                    heading = cleaned
                while j < len(group) and group[j]['start_char'] < offset + len(line):
                    row = group[j]
                    pos = row['start_char'] - offset
                    if pos >= 0:
                        context = line[max(0, pos - 160):min(len(line), pos + len(row['text']) + 160)].strip()
                        row['title'] = heading
                        row['context_text'] = ((heading + '\n') if heading else '') + context
                        row['context_status'] = 'source_line_context_title_present' if heading else 'source_line_context_title_unavailable'
                    else:
                        row['context_text'] = row['text']; row['context_status'] = 'offset_unresolved_original_retained'
                    j += 1
                offset += len(line)
                if j == len(group):
                    break
        for row in group[j:]:
            row['context_text'] = row['text']; row['context_status'] = 'offset_unresolved_original_retained'

    with out_path.open('w', encoding='utf-8') as f:
        for row in sorted(rows, key=lambda r: (r['qid'], r['rank'])):
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    manifest = {'questions': len(final), 'pairs': len(rows), 'k': a.k, 'corpus': str(corpus), 'corpus_docs': n_docs,
                'retrieval': 'BM25 char bigrams k1=1.2 b=0.75 over the full corpus; query = final question text only',
                'reference_used_for_retrieval': False, 'own_record_fragments_skipped': skipped_self,
                'contexts': dict(collections.Counter(r['context_status'] for r in rows)),
                'seconds': round(time.monotonic() - start, 1),
                'final_questions_sha256': hashlib.sha256((sealed / 'final_questions.json').read_bytes()).hexdigest(),
                'pools_sha256': hashlib.sha256(out_path.read_bytes()).hexdigest()}
    (sealed / 'POOL_MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
