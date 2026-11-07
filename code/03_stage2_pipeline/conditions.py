# -*- coding: utf-8 -*-
"""全部打分方法（E1、E2）与生成条件（E3–E5）的唯一定义处。

对应《实验设计方案_v2》第三节。条件编号与方案一致：
    G00 闭卷  G01 BM25  G02 随机  G03 全池20  G03b 全池按Support-CE排序
    G04 Relevance-CE(B)  G05 Support-CE(B)  G06 Relevance-CE(A)  G07 Support-CE(A)
    G08 bge  G09 Qwen3-Reranker-0.6B  G10 BGE-S  G11 BGE-R  G12 LLM零样本判断  G13 Qwen3-Reranker-4B
    O1 支持证据  O2 相关但不支持  O3 无关  O4 条件不符（由金标导出的候选编号清单）
    外部验证（TCM-BEST4SDT，方案第十六节）：X00 闭卷  X01 BM25  X02 随机  X04 Relevance-CE(B)  X05 Support-CE(B)
    X08 bge（完整输入）  X09 Qwen3-Reranker-0.6B（完整输入）；与封存集同编号的条件设置完全相同
编码：A = 题干前160词元（E3原版，eval_v1已有分数）；B = 完整题干、512窗口；C = 不截断（≤1024）；N = 模型完整输入。
"""
from common import EVAL_V1, read_json

SEEDS = (42, 2026, 3407)
OBJECTIVES = ('support', 'relevance')
RANDOM_SEEDS = (101, 202, 303)
ENCODINGS = {'B': {'query_max_tokens': 100000, 'max_length': 512},
             'C': {'query_max_tokens': 100000, 'max_length': 1024}}


def v1_models():
    cfg = read_json(EVAL_V1 / 'models.json', {}) or {}
    return {m['name']: m for m in cfg.get('models', [])}


def score_methods(cfg, stage='e1'):
    """返回需要在 v2 中新打分的方法列表（eval_v1 已有的 A 编码与 N 输入分数直接复用）。"""
    v1 = v1_models()
    ms = []
    if stage == 'e1_ce':
        for obj in OBJECTIVES:
            for seed in SEEDS:
                src = v1.get('{}_ce_s{}'.format(obj, seed))
                if not src:
                    raise RuntimeError('eval_v1/models.json 中找不到 {}_ce_s{}'.format(obj, seed))
                for enc, e in ENCODINGS.items():
                    ms.append({'name': '{}_ce_{}_s{}'.format(obj, enc, seed), 'kind': 'ce_v2', 'objective': obj,
                               'seed': seed, 'encoding': enc, 'path': src['path'], 'base_model': src['base_model'],
                               'spm': src['spm'], 'micro': 4, 'repro': src.get('repro'), **e})
    elif stage == 'e1_pub':
        bge, qr = v1.get('bge_reranker_v2_m3'), v1.get('qwen3_reranker_0_6b')
        for enc in ('A', 'B'):
            ms.append({'name': 'bge_{}'.format(enc), 'kind': 'hf_seqcls', 'match': enc, 'path': bge['path'],
                       'max_length': bge.get('max_length', 1024), 'batch_size': 16, 'dtype': 'fp32'})
            ms.append({'name': 'qr06_{}'.format(enc), 'kind': 'qwen3_reranker', 'match': enc, 'path': qr['path'],
                       'max_length': qr.get('max_length', 2048), 'batch_size': 8, 'dtype': 'fp32',
                       'instruction': qr.get('instruction')})
        if cfg['optional'].get('qwen3_reranker_4b', True):
            ms.append({'name': 'qr4b_N', 'kind': 'qwen3_reranker', 'model_key': 'qr4b', 'max_length': 2048,
                       'batch_size': 4, 'dtype': 'fp32', 'instruction': qr.get('instruction'), 'optional': True})
        if cfg['optional'].get('llm_selector', True):
            ms.append({'name': 'llmsel_N', 'kind': 'llm_judge', 'model_key': 'g1', 'max_length': 2048,
                       'batch_size': 4, 'dtype': 'fp32', 'optional': True})
    elif stage == 'ext':
        # 外部验证集：CE 用 B 编码（与 G04、G05 相同的设置），公开模型用 eval_v1 的完整输入设置（与 G08、G09 相同）
        ms = [m for m in score_methods(cfg, 'e1_ce') if m['encoding'] == 'B']
        for name, src in (('bge_N', 'bge_reranker_v2_m3'), ('qr06_N', 'qwen3_reranker_0_6b')):
            if src not in v1:
                raise RuntimeError('eval_v1/models.json 中找不到 {}'.format(src))
            m = {k: v for k, v in v1[src].items() if k not in ('enabled', 'note')}
            ms.append(dict(m, name=name, v1_ref=src))
    elif stage == 'e2':
        for obj in OBJECTIVES:
            for seed in SEEDS:
                ms.append({'name': 'bge_{}_ft_s{}'.format(obj, seed), 'kind': 'hf_seqcls',
                           'path': 'eval_v2/runs/e2/{}_s{}/model'.format(obj, seed), 'max_length': 1024,
                           'batch_size': 16, 'dtype': 'fp32', 'objective': obj, 'seed': seed})
    return ms


def _score(name, source):
    return {'type': 'score', 'name': name, 'source': source}


def _base_selectors():
    """k=2 主条件中每个组的证据选择方式（不含 k、提示词、问答模型）。"""
    s = {'G00': [('G00', {'type': 'none'})],
         'G01': [('G01', {'type': 'bm25'})],
         'G02': [('G02_r{}'.format(i + 1), {'type': 'random', 'seed': r}) for i, r in enumerate(RANDOM_SEEDS)],
         'G03': [('G03', {'type': 'all'})],
         'G03b': [('G03b', {'type': 'score_mean', 'names': ['support_ce_B_s{}'.format(x) for x in SEEDS],
                            'source': 'v2'})],
         'G04': [('G04_s{}'.format(x), _score('relevance_ce_B_s{}'.format(x), 'v2')) for x in SEEDS],
         'G05': [('G05_s{}'.format(x), _score('support_ce_B_s{}'.format(x), 'v2')) for x in SEEDS],
         'G06': [('G06_s{}'.format(x), _score('relevance_ce_s{}'.format(x), 'v1')) for x in SEEDS],
         'G07': [('G07_s{}'.format(x), _score('support_ce_s{}'.format(x), 'v1')) for x in SEEDS],
         'G08': [('G08', _score('bge_reranker_v2_m3', 'v1'))],
         'G09': [('G09', _score('qwen3_reranker_0_6b', 'v1'))],
         'G10': [('G10_s{}'.format(x), _score('bge_support_ft_s{}'.format(x), 'v2')) for x in SEEDS],
         'G11': [('G11_s{}'.format(x), _score('bge_relevance_ft_s{}'.format(x), 'v2')) for x in SEEDS],
         'G12': [('G12', _score('llmsel_N', 'v2'))],
         'G13': [('G13', _score('qr4b_N', 'v2'))],
         'O1': [('O1', {'type': 'list', 'list': 'O1'})],
         'O2': [('O2', {'type': 'list', 'list': 'O2'})],
         'O3': [('O3', {'type': 'list', 'list': 'O3'})],
         'O4': [('O4', {'type': 'list', 'list': 'O4'})]}
    return s


def _cond(cid, group, sel, k, prompt, gen, cset, scr, target='sealed'):
    return {'id': cid, 'group': group, 'selector': sel, 'k': k, 'prompt': prompt, 'gen': gen, 'set': cset,
            'scr': scr, 'target': target, 'optional': group in ('G12', 'G13')}


def conditions(cfg):
    """展开全部生成条件。返回按集合分组的列表：main、ksens、palt、e2、oracle、g2、ext（外部验证）。"""
    b = _base_selectors()
    opt = cfg['optional']
    out = {'main': [], 'ksens': [], 'palt': [], 'e2': [], 'oracle': [], 'g2': [], 'ext': []}
    main_groups = ['G00', 'G01', 'G02', 'G03', 'G03b', 'G04', 'G05', 'G06', 'G07', 'G08', 'G09']
    if opt.get('llm_selector', True):
        main_groups.append('G12')
    if opt.get('qwen3_reranker_4b', True):
        main_groups.append('G13')
    for g in main_groups:
        k = 20 if g in ('G03', 'G03b') else (0 if g == 'G00' else 2)
        for cid, sel in b[g]:
            out['main'].append(_cond(cid, g, sel, k, 'main', 'G1', 'main', scr=(g != 'G00')))
    for k in (1, 3, 5):
        for g in ('G01', 'G04', 'G05', 'G08', 'G09'):
            for cid, sel in b[g]:
                out['ksens'].append(_cond('K{}_{}'.format(k, cid), g, sel, k, 'main', 'G1', 'ksens', scr=False))
    for g in ('G00', 'G01', 'G02', 'G04', 'G05', 'G08', 'G09'):
        items = b[g][:1] if g == 'G02' else b[g]
        for cid, sel in items:
            out['palt'].append(_cond('PA_{}'.format(cid), g, sel, 0 if g == 'G00' else 2, 'alt', 'G1', 'palt',
                                     scr=False))
    if opt.get('e2_finetune', True):
        for g in ('G10', 'G11'):
            for cid, sel in b[g]:
                out['e2'].append(_cond(cid, g, sel, 2, 'main', 'G1', 'e2', scr=True))
    for g in ('O1', 'O2', 'O3', 'O4'):
        for cid, sel in b[g]:
            out['oracle'].append(_cond(cid, g, sel, 2, 'main', 'G1', 'oracle', scr=True))
    if opt.get('g2_generator', True):
        for g in ('G00', 'G01', 'G04', 'G05', 'G08', 'G09', 'O1', 'O2', 'O3'):
            for cid, sel in b[g]:
                out['g2'].append(_cond('G2_{}'.format(cid), g, sel, 0 if g == 'G00' else 2, 'main', 'G2', 'g2',
                                       scr=False))
    if opt.get('external', True) and cfg.get('external'):
        ext = [('X00', 'X00', {'type': 'none'}, 0), ('X01', 'X01', {'type': 'bm25'}, 2)]
        ext += [('X02_r{}'.format(i + 1), 'X02', {'type': 'random', 'seed': r}, 2) for i, r in enumerate(RANDOM_SEEDS)]
        ext += [('X04_s{}'.format(x), 'X04', _score('relevance_ce_B_s{}'.format(x), 'v2'), 2) for x in SEEDS]
        ext += [('X05_s{}'.format(x), 'X05', _score('support_ce_B_s{}'.format(x), 'v2'), 2) for x in SEEDS]
        ext += [('X08', 'X08', _score('bge_N', 'v2'), 2), ('X09', 'X09', _score('qr06_N', 'v2'), 2)]
        for cid, g, sel, k in ext:
            out['ext'].append(_cond(cid, g, sel, k, 'main', 'G1', 'ext', scr=False, target='external'))
    return out


def smoke_conditions():
    """E0 技术冒烟：校准集 5 题，闭卷、BM25 Top-2、全池 20 条（最长输入）。"""
    return [_cond('S_G00', 'G00', {'type': 'none'}, 0, 'main', 'G1', 'smoke', scr=True, target='calib'),
            _cond('S_G01', 'G01', {'type': 'bm25'}, 2, 'main', 'G1', 'smoke', scr=True, target='calib'),
            _cond('S_G03', 'G03', {'type': 'all'}, 20, 'main', 'G1', 'smoke', scr=False, target='calib')]


def all_conditions(cfg):
    c = conditions(cfg)
    return [x for k in ('main', 'ksens', 'palt', 'e2', 'oracle', 'g2', 'ext') for x in c[k]] + smoke_conditions()


def cost(c):
    """生成成本的粗略相对值（用于把条件分配到几个生成作业）。"""
    k = c['k']
    return 1.0 if k <= 2 else (1.4 if k == 3 else (1.9 if k == 5 else 5.0))
