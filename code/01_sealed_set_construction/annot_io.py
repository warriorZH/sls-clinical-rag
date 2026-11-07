"""Parse filled annotation sheets (Phase 1 and Phase 2) into plain Python structures."""
import re
from pathlib import Path

from common import read_csv, find_col, norm_int

POINT_TYPES = ('疾病诊断', '证候', '病机', '治疗原则')
TYPE_ALIASES = {'诊断': '疾病诊断', '疾病': '疾病诊断', '病名': '疾病诊断', '证型': '证候', '证': '证候',
                '病机解释': '病机', '病因病机': '病机', '治则': '治疗原则', '治法': '治疗原则', '治则治法': '治疗原则'}


def annotator_files(folder, pattern):
    """Return {annotator_name: path} for files like '<pattern>_标注员A.csv'."""
    out = {}
    for p in sorted(Path(folder).glob(f'{pattern}_标注员*.csv')):
        out[p.stem.split('_标注员')[-1]] = p
    return out


def split_multi(text, seps=r'\|\||｜｜'):
    return [x.strip() for x in re.split(seps, text or '') if x.strip()]


def norm_type(t, default=''):
    t = (t or '').strip().replace('：', '').replace(':', '')
    t = TYPE_ALIASES.get(t, t)
    return t if t in POINT_TYPES else default


def parse_point_list(text, default_type=''):
    """'证候:肝郁脾虚；治疗原则:疏肝健脾' -> [(type, text), ...]"""
    items = []
    for chunk in re.split(r'[；;]\s*|\n', text or ''):
        chunk = chunk.strip()
        if not chunk:
            continue
        m = re.match(r'^\s*([^:：]{1,6})[:：]\s*(.+)$', chunk)
        if m and norm_type(m.group(1)):
            items.append((norm_type(m.group(1)), m.group(2).strip()))
        else:
            items.append((default_type, chunk))
    return items


def normalize_text(s):
    return re.sub(r'[\s，。、；;,.:：“”"\'（）()【】\[\]]', '', s or '')


CLAUSE_SEP = r'[，,。；;、：:\s|]+'


def clauses(fragments):
    """Split copied fragments into clauses: [(normalized key, original text)] in order, de-duplicated.
    Two annotators who copied the same text but cut it into different pieces get the same keys."""
    out, seen = [], set()
    for frag in fragments or []:
        for c in re.split(CLAUSE_SEP, frag):
            key = normalize_text(c)
            if key and key not in seen:
                seen.add(key)
                out.append((key, c.strip()))
    return out


def clause_set(fragments):
    return frozenset(k for k, _ in clauses(fragments))


def load_phase1(folder, prefix=''):
    """Returns {annotator: {qid: {...question fields..., 'points': {pid: {...}}}}}"""
    qfiles = annotator_files(folder, f'{prefix}phase1_题目核验')
    pfiles = annotator_files(folder, f'{prefix}phase1_要点核验')
    data = {}
    for who, qpath in qfiles.items():
        qrows = read_csv(qpath)
        if not qrows:
            continue
        h = qrows[0]
        c_inc, c_why = find_col(h, '纳入'), find_col(h, '排除原因')
        c_leak, c_frag = find_col(h, '问题含答案泄露'), find_col(h, '应删除的泄露片段')
        c_add, c_miss = find_col(h, '应补充的关键病情信息'), find_col(h, '遗漏的必要要点')
        c_note = find_col(h, '备注')
        qs = {}
        for r in qrows:
            qs[r['题目编号']] = {
                'question_type_cn': r.get('题型', ''), 'question_draft': r.get('检索用问题文本(草稿)', ''),
                'include': norm_int(r[c_inc], {0, 1}), 'exclude_reason': r[c_why],
                'leak': norm_int(r[c_leak], {0, 1}), 'leak_fragments': split_multi(r[c_frag]),
                'add_info': split_multi(r[c_add]),
                'missing_points': parse_point_list(r[c_miss]), 'note': r[c_note], 'points': {}}
        ppath = pfiles.get(who)
        if ppath:
            prow = read_csv(ppath)
            if prow:
                h = prow[0]
                c_dec, c_rev = find_col(h, '判定'), find_col(h, '修改后内容')
                c_typ, c_note = find_col(h, '修改后类型'), find_col(h, '备注')
                for r in prow:
                    q = qs.setdefault(r['题目编号'], {'points': {}})
                    q['points'][r['要点编号']] = {
                        'draft_type': r.get('草拟要点类型', ''), 'draft_text': r.get('草拟要点内容', ''),
                        'decision': norm_int(r[c_dec], {0, 1, 2}),
                        'revised': [x.strip() for x in re.split(r'[；;]', r[c_rev] or '') if x.strip()],
                        'revised_type': norm_type(r[c_typ]), 'note': r[c_note]}
        filled = any(q.get('include') is not None or q.get('leak') is not None or q.get('leak_fragments')
                     or q.get('missing_points') or any(pp['decision'] is not None for pp in q.get('points', {}).values())
                     for q in qs.values())
        if not filled:
            print(f'提示：标注员{who}的 Phase 1 文件尚未填写，已跳过。')
            continue
        data[who] = qs
    return data


def final_points_of(q):
    """The point list an annotator proposes for one question (after keep/modify/delete + additions)."""
    out = []
    for pid in sorted(q['points'], key=lambda x: int(re.sub(r'\D', '', x) or 0)):
        p = q['points'][pid]
        if p['decision'] == 0:
            continue
        if p['decision'] == 2 and p['revised']:
            typ = p['revised_type'] or norm_type(p['draft_type'], p['draft_type'])
            out.extend((typ, t) for t in p['revised'])
        else:  # 1, or blank treated as keep (reported as missing decision)
            out.append((norm_type(p['draft_type'], p['draft_type']), p['draft_text']))
    out.extend(q.get('missing_points', []))
    seen, uniq = set(), []
    for typ, text in out:
        key = normalize_text(text)
        if key and key not in seen:
            seen.add(key)
            uniq.append((typ, text))
    return uniq


def render_point_list(points):
    return '；'.join(f'{t}:{x}' if t else x for t, x in points)


def load_phase2(folder, prefix='phase2_证据标注'):
    """Returns {annotator: {(qid, cand): {'relevance':..,'support':{1:..},'mismatch':..,'uncertain':..,'note':..}}}"""
    files = annotator_files(folder, prefix)
    data = {}
    for who, path in files.items():
        rows = read_csv(path)
        if not rows:
            continue
        h = rows[0]
        c_rel = find_col(h, '相关性')
        c_mis = find_col(h, '同病异证或条件不符')
        c_unc = find_col(h, '无法判断')
        c_note = find_col(h, '备注')
        pcols = {}
        for k in h:
            m = re.match(r'^(?:【填】)?P(\d+)支持', k)
            if m:
                pcols[int(m.group(1))] = k
        ann = {}
        for r in rows:
            sup = {}
            for i, col in pcols.items():
                raw = (r[col] or '').strip().upper()
                if raw == 'NA':
                    continue
                sup[i] = norm_int(raw, {0, 1, 2})
            ann[(r['题目编号'], r['候选编号'])] = {
                'relevance': norm_int(r[c_rel], {0, 1}), 'support': sup,
                'mismatch': norm_int(r[c_mis], {0, 1}),
                'uncertain': 1 if norm_int(r[c_unc], {0, 1}) == 1 else 0,
                'note': r[c_note], 'row': r}
        filled = any(rec['relevance'] is not None or rec['mismatch'] is not None
                     or any(v is not None for v in rec['support'].values()) for rec in ann.values())
        if not filled:
            print(f'提示：标注员{who}的 Phase 2 文件尚未填写，已跳过。')
            continue
        data[who] = ann
    return data
