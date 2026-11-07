#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""由金标导出 E4 的证据清单 O1–O4（只含候选编号，不含任何标签）。在本机或分析环境运行，不在服务器上运行。

    python build_oracle_lists.py <expert_testset_v1 目录> <输出目录>
规则（方案 3.5 节）：
  O1 支持证据：在至少部分支持某要点的候选中选 2 条，使完整支持覆盖的要点最多；其次分级覆盖最多；
     其次两条的检索名次之和最小。只有 1 条支持候选时，用检索名次最前的无关候选补齐。
  O2 相关但不支持：相关性为 1、对所有要点的支持度都为 0，取检索名次最前的 2 条（不足 2 条的题不进入）。
  O3 无关：相关性为 0、对所有要点的支持度都为 0，取检索名次最前的 2 条（不足 2 条的题不进入）。
  O4 同病异证或条件不符：该标记为 1 的候选，按检索名次最多取 2 条（探索性）。
"""
import hashlib
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path


def main(root, out_dir):
    root, out_dir = Path(root), Path(out_dir)
    sealed = root / '_sealed_负责人保存_勿发给标注员'
    gold_path = sealed / 'phase2_gold_labels.jsonl'
    qs = {q['qid']: q for q in json.loads((sealed / 'final_questions.json').read_text(encoding='utf-8'))
          if q.get('included')}
    gold = defaultdict(list)
    for line in gold_path.read_text(encoding='utf-8').splitlines():
        if line.strip():
            r = json.loads(line)
            if r['qid'] in qs:
                gold[r['qid']].append(r)
    out, counts = [], defaultdict(int)
    for qid in sorted(gold):
        rows = sorted(gold[qid], key=lambda r: r['retrieval_rank'])
        pids = [p['pid'] for p in qs[qid]['points']]
        sup = lambda r: max([r['support'].get(p, 0) or 0 for p in pids] or [0])  # noqa: E731
        supportive = [r for r in rows if sup(r) >= 1]
        relnon = [r for r in rows if r['relevance'] == 1 and sup(r) == 0]
        irr = [r for r in rows if r['relevance'] == 0 and sup(r) == 0]
        mism = [r for r in rows if r.get('mismatch') == 1]

        def cov(sel):
            strict = sum(1 for p in pids if max(r['support'].get(p, 0) or 0 for r in sel) == 2)
            graded = sum(max(r['support'].get(p, 0) or 0 for r in sel) / 2 for p in pids)
            return (strict, graded, -sum(r['retrieval_rank'] for r in sel))
        o1 = None
        if len(supportive) >= 2:
            o1 = list(max(itertools.combinations(supportive, 2), key=lambda pair: (cov(pair), [-r['retrieval_rank'] for r in pair])))
            o1.sort(key=lambda r: (-sup(r), r['retrieval_rank']))
        elif len(supportive) == 1 and irr:
            o1 = [supportive[0], irr[0]]
        lists = {'O1': o1, 'O2': relnon[:2] if len(relnon) >= 2 else None,
                 'O3': irr[:2] if len(irr) >= 2 else None, 'O4': mism[:2] if mism else None}
        for name, sel in lists.items():
            if sel:
                out.append({'list': name, 'qid': qid, 'docs': [str(r['document_id']) for r in sel]})
                counts[name] += 1
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / 'oracle_lists.jsonl').open('w', encoding='utf-8') as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    have = defaultdict(set)
    for r in out:
        have[r['list']].add(r['qid'])
    meta = {'rules': __doc__.strip().splitlines()[3:], 'counts': dict(counts),
            'subset_O1_O2_O3': len(have['O1'] & have['O2'] & have['O3']),
            'gold_sha256': hashlib.sha256(gold_path.read_bytes()).hexdigest(),
            'lists_sha256': hashlib.sha256((out_dir / 'oracle_lists.jsonl').read_bytes()).hexdigest(),
            'note': '只含候选编号，不含支持度、相关性等标签；服务器上只有在非 O 条件的答案冻结之后才会读取。'}
    (out_dir / 'oracle_lists_meta.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
