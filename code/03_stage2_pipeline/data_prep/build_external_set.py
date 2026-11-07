#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""把公开数据集 TCM-BEST4SDT 的辨证论治病例（TCM_SDT.json，300 例）转成 eval_v2 外部验证用的题目与参考要点。

    python build_external_set.py --src TCM_SDT.json --out ../server/data/external

只做确定性的规则转换，不调用任何模型，也不做人工或 AI 修改：
  - 每个病例出一道题。题型由病例编号的哈希固定分配（约各一半）：诊断题 = 病例原文 + DIAG_Q；治则题 = 病例原文 + PRIN_Q。
    两句题型指令与封存集（scripts/build_sealed_testset.py）逐字相同。病例原文只去掉首尾空白，不做其他改动。
  - 参考要点全部摘自数据集作者（中医专家）给出的标准答案字段，规则见 point_rules()：
      诊断题：疾病诊断（“中医疾病诊断”字段）+ 证候（“证型”字段）+ 病机（“病机”字段的前 2 个合格分句）
      治则题：治疗原则（“治则治法”字段）
  - 明显的录入错误只按 CORRECTIONS 表更正（目前 1 处），并写入清单。
  - 病例分层（预先规定）：编号 1–150 为名家医案改写与教材式病例，151–300 为现代临床病例。
输出：
  external_questions_all.json   题目与要点（服务器上再排除与 CE 训练病例近重复的题）
  external_reference.jsonl      标准答案原文与处方药名（只用于服务器上排除病例自身原始记录的过滤器与事后分析，不给任何模型看）
  BUILD_MANIFEST.json           来源文件哈希、规则、计数
"""
import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

DIAG_Q = '\n请解释该病例的诊断、证候与病因病机。'
PRIN_Q = '\n请说明适用于该病例的治疗原则，不要求方药剂量。'
QTYPE_SALT = 'EXT-QTYPE'
CLASSICAL_MAX_ID = 150
MAX_SYNDROME = 3
MAX_MECH = 2
CORRECTIONS = {'白庁': '白疕'}          # 病例 168：病史为银屑病（白疕），“庁”为录入错误
RULES_VERSION = 'ext-points-1'

# ---- 以下 TCM_ACTION、split_terms、leak_fragments 及相关正则逐字复制自 scripts/draft_points.py（封存集的要点草拟规则）
TCM_ACTION = '补益清化活疏健温滋养祛利散通止安理调宣降平软固收解凉燥润行开潜熄息镇和消逐泻泄攻敛摄升托透渗燥除'


def _clean(s):
    s = re.sub(r'\s+', '', s or '')
    return s.strip('，。；;：:、 ')


def split_terms(phrase, min_len=2, max_len=16, split_he=True):
    parts = re.split(r'[，、；;,]|以及|及', phrase)
    if not split_he:
        expanded = []
        for part in parts:
            m = re.fullmatch(r'(.{4,})和(.{4,})', _clean(part))
            if m and m.group(1)[0] in TCM_ACTION and m.group(2)[0] in TCM_ACTION:
                expanded.extend([m.group(1), m.group(2)])
            else:
                expanded.append(part)
        parts = expanded
    if split_he:
        expanded = []
        for part in parts:
            pieces = re.split(r'(?<=[^一-鿿不调])和(?=[^胃络中营解血卫])|(?<=[一-鿿]{2})和(?=[^胃络中营解血卫])', part)
            pieces = [x for x in pieces if x]
            if len(pieces) > 1 and all(len(_clean(x)) >= 2 for x in pieces):
                expanded.extend(pieces)
            else:
                expanded.append(part)
        parts = expanded
    out = []
    for p in parts:
        p = _clean(p)
        p = re.sub(r'^(?:了|为|是|属于?|证属|以|通过|采用|采取|运用|予以?|给予|主要是|应以)+', '', p)
        p = re.sub(r'(?:的情况|的证候|之证|等证|为主|的治疗方法|的治法|的治则|的方法来治疗|的方法|来治疗|治法|治则)$', '', p)
        p = _clean(p)
        if min_len <= len(p) <= max_len and p not in out:
            out.append(p)
    return out


STRONG_LEAK = re.compile(r'证属|证为|证候|辨证|诊断为|诊断是|病机|治则|治法|治疗原则|属于.{0,8}证|所致|乃.{0,6}所致|系.{0,6}所致')
WEAK_LEAK = re.compile(r'导致|引起|由于|因为|以致|致使|因此|所以|故|化热|化火|化燥|生痰|无力|内蕴|侵袭|郁结|不足|亏虚|失调|失养|阻滞|内阻|内停|上扰|上犯|下注|佐证|判断')
TCM_TERMS = re.compile(r'[虚实寒热湿痰瘀郁气血阴阳风火毒]')


def leak_fragments(background, points):
    clauses = [c for c in re.split(r'[；;。]', background) if c.strip()]
    flagged = []
    terms = [p['text'] for p in points if p['type'] != '疾病诊断' and 2 <= len(p['text']) <= 12]
    for c in clauses:
        hit_terms = [t for t in terms if t in c]
        weak = WEAK_LEAK.search(c) and len(TCM_TERMS.findall(c)) >= 2
        if hit_terms or STRONG_LEAK.search(c) or weak:
            flagged.append(c.strip())
    return flagged
# ---- 复制结束


PAREN = re.compile(r'[（(][^（）()]*[）)]')
SEP = re.compile(r'[，、；;,/]')
CLAUSE_SEP = re.compile(r'[，。；;、,：:！？!?]')
PATHO = set('虚实寒热湿痰瘀郁气血阴阳风火毒燥滞阻逆亢衰盛蕴结闭扰失伤损耗亏犯')
MECH_SKIP_START = re.compile(r'^(?:导致|因此|所以|从而|以致|致使|遂|乃|而|引起|出现|表现|症见|可见|加之|兼之|又|且|并|终致|进而|日久|久之|久则|发为|提示|属|病情|尚未|未|非)')
MECH_SKIP_ANY = re.compile(r'[则故]|导致|所致|引起|症状|表现为|大便|小便|吞咽|食欲|饮食|睡眠')
PRIN_LEAD = re.compile(r'^(?:佐以|兼以|辅以|并以|同时|再以|酌加|佐|兼|并)')


def fix(s):
    for a, b in CORRECTIONS.items():
        s = s.replace(a, b)
    return s


def no_paren(s):
    prev = None
    while prev != s:
        prev, s = s, PAREN.sub('', s)
    return s


def strip_suffix(t):
    """证候要点与封存集的写法一致：去掉末尾的“证”“型”（剩余至少 4 个字时；“阳明经证”“脾虚证”等短名保持原样）。"""
    if len(t) >= 5 and t[-1] in '证型':
        return t[:-1]
    return t


def disease_points(raw):
    s = no_paren(fix(raw))
    parts = [_clean(p) for p in SEP.split(s)]
    parts = [p for p in parts if p]
    if parts and all(p.endswith('证') and len(p) >= 5 and not p.endswith('诸证') for p in parts):
        return [], 'disease_field_is_syndrome'       # 疾病诊断字段写的是证型（如病例 64、91），不作为疾病要点
    out = []
    for p in parts:
        if 2 <= len(p) <= 12 and p not in out:
            out.append(p)
    return out, None


def syndrome_points(raw, exclude=()):
    s = no_paren(fix(raw))
    out = []
    for p in SEP.split(s):
        p = _clean(p)
        m = re.fullmatch(r'(.{2,}?)证?兼(.{2,})', p)          # “风寒表实证兼阳气郁闭证”拆成两个证候要素
        for q in ([m.group(1), m.group(2)] if m else [p]):
            q = strip_suffix(_clean(q))
            if 2 <= len(q) <= 12 and q not in out and q not in exclude:
                out.append(q)
    return out[:MAX_SYNDROME]


def mechanism_points(raw, exclude):
    s = no_paren(fix(raw))
    out = []
    for c in CLAUSE_SEP.split(s):
        c = _clean(c)
        if not (4 <= len(c) <= 12) or MECH_SKIP_START.search(c) or MECH_SKIP_ANY.search(c):
            continue
        if not any(ch in PATHO for ch in c) or c in exclude or c in out:
            continue
        out.append(c)
        if len(out) == MAX_MECH:
            break
    return out


def principle_points(raw):
    """“治则治法”字段本身就是专家写的治则短语，拆分后全部保留（封存集用治法用字过滤，是因为要点从病例分析的自由文本中抽取）。"""
    s = no_paren(fix(raw))
    out = []
    for p in split_terms(s, 2, 14, split_he=False):
        p = _clean(PRIN_LEAD.sub('', p))
        if 2 <= len(p) <= 14 and p not in out:
            out.append(p)
    return out


HERB_SPLIT = re.compile(r'[、，,；;\n]|(?<=[gG克])\s+(?=[\u4e00-\u9fff])')
HERB_LEAD = re.compile(r'^(?:第[一二三四五六七八九十]+(?:步)?方(?:剂)?|丸剂|汤剂|散剂|方药|处方)[：:]')
HERB_DOSE = re.compile(r'[\d０-９.．~～\-—]+\s*(?:g|G|克|ml|mL|毫升|枚|个|只|条|片|粒|丸|钱|两|分|斤|升|合|cm|厘米)?.*$')
HERB_CN_DOSE = re.compile(r'[一二三四五六七八九十百半两钱]+大?(?:枚|个|只|条|片|粒|钱|两|斤|撮|具|分|升|合)?(?:大者)?$')
HERB_PROC_LEAD = re.compile(r'^(?:去皮|去心|去核|捣碎|炒捣|轧细)+')
HERB_TAIL = re.compile(r'(?:各|适量|少许|若干|冲服|后下|先煎|包煎|烊化|另煎|研末|研细|为末|等量|等分|主药|捣细|轧细|捣碎)+$')
HERB_STOP = re.compile(r'制成|服用|煎服|煎汤|丸剂|散剂|汤剂|加减|剂量|调入|送服|一日|每次|间隔|小时|西药')
HERB_PREFIX = ('镜面', '蜜炙', '生', '炒', '炙', '制', '酒', '醋', '焦', '煅', '净', '鲜', '大', '野', '广', '川', '怀', '杭', '淡',
               '法', '清', '姜', '蜜', '盐', '熟', '粉', '旱', '全', '碎', '干', '辽', '乌', '高丽', '西', '白', '赤')
HERB_SUFFIX = ('细末', '细面', '头末', '末', '粉', '片', '仁', '绒')


def herb_keys(name):
    """药名及其去掉炮制、产地等前缀和“细末”等后缀后的核心名（至少 2 字），用于在候选证据中查找。"""
    keys, core = [name], name
    changed = True
    while changed:
        changed = False
        for suf in HERB_SUFFIX:
            if core.endswith(suf) and len(core) - len(suf) >= 2:
                core, changed = core[:-len(suf)], True
        for pre in HERB_PREFIX:
            if core.startswith(pre) and len(core) - len(pre) >= 2:
                core, changed = core[len(pre):], True
    if core not in keys:
        keys.append(core)
    return keys


def herbs_of(raw):
    """处方药名（去掉剂量与炮制说明）。只用于服务器上识别病例自身的原始记录（见 server/build_external.py）。"""
    out = []
    for p in HERB_SPLIT.split(no_paren(HERB_LEAD.sub('', (raw or '').strip()))):
        p = re.sub(r'\s+', '', p)
        p = HERB_LEAD.sub('', p)
        p = re.split(r'[：:]', p)[0]
        p = HERB_DOSE.sub('', p)
        p = HERB_CN_DOSE.sub('', p)
        p = HERB_TAIL.sub('', HERB_PROC_LEAD.sub('', p))
        if HERB_STOP.search(p):
            continue
        if 2 <= len(p) <= 8 and re.fullmatch(r'[\u4e00-\u9fff]+', p) and p not in out:
            out.append(p)
    return out


def point_rules():
    return {
        'version': RULES_VERSION,
        '题型': 'sha256("{}:"+病例编号) 为偶数 → 诊断题，否则治则题'.format(QTYPE_SALT),
        '疾病诊断': '“中医疾病诊断”字段；去掉括号内的说明（多为证型或西医病名）；按 ，、；/ 拆分；'
                '各部分都写成 4 字以上加“证”的（字段实为证型）不出疾病要点',
        '证候': '“证型”字段；去掉括号内容；按 ，、；/ 和“兼”拆成证候要素；去掉末尾的“证/型”（剩余不少于 4 字时）；'
              '与疾病要点相同的不重复；保留前 {} 个'.format(MAX_SYNDROME),
        '病机': '“病机”字段；去掉括号内容后按标点拆成分句；取前 {} 个合格分句：4–12 字、含病机用字（{}）、'
              '不以“导致、因此、日久、尚未……”等词开头、不含“则、故、导致、所致、引起、症状、表现为”及“大便、小便、吞咽、食欲、饮食、睡眠”等症状描述、不与证候要点重复'.format(
                  MAX_MECH, ''.join(sorted(PATHO))),
        '治疗原则': '“治则治法”字段；去掉括号内容；按 scripts/draft_points.py 的 split_terms 拆分；去掉开头的“佐以、兼……”；'
                '保留 2–14 字的短语',
        '更正': CORRECTIONS,
        '分层': '编号 1–{} 为名家医案改写与教材式病例（classical），其余为现代临床病例（modern）'.format(CLASSICAL_MAX_ID),
        '题型指令': {'diagnosis': DIAG_Q.strip(), 'principle': PRIN_Q.strip()},
    }


def parse_output(lines):
    o = {}
    for s in lines:
        k, _, v = s.partition('：')
        o[k.strip()] = v.strip()
    return o


def qtype_of(case_id):
    h = int(hashlib.sha256('{}:{}'.format(QTYPE_SALT, case_id).encode('utf-8')).hexdigest(), 16)
    return 'diagnosis' if h % 2 == 0 else 'principle'


def build(src):
    data = json.loads(Path(src).read_text(encoding='utf-8'))
    questions, refs, notes = [], [], Counter()
    for x in sorted(data, key=lambda r: int(r['id'])):
        cid = int(x['id'])
        o = parse_output(x['output'])
        background = (x['instruction'] or '').strip()
        qtype = qtype_of(cid)
        pts = []
        if qtype == 'diagnosis':
            dis, why = disease_points(x.get('中医疾病诊断') or '')
            if why:
                notes[why] += 1
            syn = syndrome_points(o.get('证型', ''), set(dis))
            mech = mechanism_points(o.get('病机', ''), set(syn) | set(dis))
            pts += [('疾病诊断', t, '中医疾病诊断') for t in dis]
            pts += [('证候', t, '证型') for t in syn]
            pts += [('病机', t, '病机') for t in mech]
        else:
            pts += [('治疗原则', t, '治则治法') for t in principle_points(o.get('治则治法', ''))]
        points = [{'pid': 'P{}'.format(i), 'type': ty, 'text': t, 'source_field': f}
                  for i, (ty, t, f) in enumerate(pts, 1)]
        if not points:
            raise RuntimeError('病例 {} 没有可用的参考要点'.format(cid))
        qid = 'EX{:03d}'.format(cid)
        questions.append({'qid': qid, 'source_id': cid, 'question_type': qtype,
                          'stratum': 'classical' if cid <= CLASSICAL_MAX_ID else 'modern',
                          'included': True, 'background': background,
                          'question_final': background + (DIAG_Q if qtype == 'diagnosis' else PRIN_Q),
                          'points': points, 'leak_flags': leak_fragments(background, points),
                          'rules_version': RULES_VERSION})
        corrected = [a for a in CORRECTIONS if a in json.dumps(x, ensure_ascii=False)]
        refs.append({'qid': qid, 'source_id': cid, 'disease': x.get('中医疾病诊断'), 'syndrome': o.get('证型'),
                     'cause': o.get('病因'), 'mechanism': o.get('病机'), 'nature': o.get('病性'),
                     'location': o.get('病位'), 'principle': o.get('治则治法'),
                     'prescription': o.get('药物组成及用量'), 'herbs': herbs_of(o.get('药物组成及用量')),
                     'herb_keys': [herb_keys(h) for h in herbs_of(o.get('药物组成及用量'))],
                     'corrections': corrected})
    return questions, refs, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='TCM-BEST4SDT 仓库中的 TCM_SDT.json')
    ap.add_argument('--out', required=True)
    ap.add_argument('--commit', default='', help='数据集仓库的 git 提交号（写入清单）')
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    questions, refs, notes = build(a.src)
    qp, rp = out / 'external_questions_all.json', out / 'external_reference.jsonl'
    qp.write_text(json.dumps(questions, ensure_ascii=False, indent=1), encoding='utf-8')
    with rp.open('w', encoding='utf-8') as f:
        for r in refs:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    cnt = Counter((q['stratum'], q['question_type']) for q in questions)
    ptypes = Counter(p['type'] for q in questions for p in q['points'])
    npts = Counter(len(q['points']) for q in questions)
    manifest = {
        'dataset': 'TCM-BEST4SDT（辨证论治任务 TCM_SDT.json）',
        'source_url': 'https://github.com/DYJG-research/TCM-BEST4SDT',
        'source_commit': a.commit or None,
        'source_sha256': sha(a.src),
        'citation': 'Li K, Guo J, Shang Z, et al. A benchmark dataset for evaluating Syndrome Differentiation and '
                    'Treatment in large language models. arXiv:2512.02816, 2025.',
        'archive': 'Figshare DOI 10.6084/m9.figshare.30615956',
        'license': '数据集以 CC BY 4.0 发布；代码仓库为 Apache-2.0。使用时须注明出处。',
        'script_sha256': sha(__file__),
        'rules': point_rules(),
        'n_cases': len(questions),
        'by_stratum_and_type': {'{}/{}'.format(*k): v for k, v in sorted(cnt.items())},
        'point_types': dict(ptypes),
        'points_per_question': {str(k): v for k, v in sorted(npts.items())},
        'notes': dict(notes),
        'questions_with_leak_flags': sum(1 for q in questions if q['leak_flags']),
        'outputs': {qp.name: sha(qp), rp.name: sha(rp)},
    }
    (out / 'BUILD_MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: manifest[k] for k in ('n_cases', 'by_stratum_and_type', 'point_types', 'points_per_question',
                                                'notes', 'questions_with_leak_flags')}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
