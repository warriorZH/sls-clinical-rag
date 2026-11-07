"""Draft necessary answer points for expert verification (NOT gold labels).

All drafts are rule-based extractions from the original physician's analysis
text (or the official answer for exam items).  Experts confirm, edit, split or
delete them in Phase 1.  Nothing here invents content that is absent from the
source text.
"""
import re

TCM_ACTION = '补益清化活疏健温滋养祛利散通止安理调宣降平软固收解凉燥润行开潜熄息镇和消逐泻泄攻敛摄升托透渗燥除'
PLACEHOLDER = re.compile(r'以上是|详细描述|治疗方案。?$|如上所述|医生的建议|希望.*能够|相信')

def _clean(s):
    s = re.sub(r'\s+', '', s or '')
    return s.strip('，。；;：:、 ')

def split_terms(phrase, min_len=2, max_len=16, split_he=True):
    """Split a phrase into terms on punctuation and on 及/以及; split on 和 only
    when it is clearly a conjunction (not part of 和胃, 和络, 营卫不和 ...)."""
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
            pieces = re.split(r'(?<=[^\u4e00-\u9fff不调])和(?=[^胃络中营解血卫])|(?<=[\u4e00-\u9fff]{2})和(?=[^胃络中营解血卫])', part)
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

def sentences(text):
    text = re.sub(r'\s+', '', text or '')
    return [s for s in re.split(r'(?<=[。；！？])', text) if s.strip('。；！？ ')]

DISEASE_PAT = [
    r'中医(?:的)?诊断(?:为|是|：|:|属于?)\s*[“"]?([^，。；;,\n”"]+)',
    r'中医(?:方面的)?诊断(?:为|是)\s*([^，。；;,\n]+)',
    r'诊断(?:为|是)中医(?:的)?([^，。；;,\n]+)',
]
SYNDROME_PAT = [
    r'中医(?:的)?证候(?:为|是|：|:|属于?)\s*([^。；;\n]+)',
    r'证候(?:为|是|属于?)\s*([^。；;\n]+)',
    r'证属(?:于)?\s*([^。；;\n]+)',
    r'辨证(?:为|属于?)\s*([^。；;\n]+)',
    r'(?:患者|病情|她|他)?(?:的病情)?属于([^。；;\n]{2,24}?)(?:的情况|的证候|之证|证)',
    r'属于([^。；;\n，]{2,14}?证)',
    r'(?:患者|病情|本病)(?:的病情)?属于([^。；;\n]{2,30})[。；]',
    r'的病(?:情)?(?:属于|为)([^。；;\n]{2,20})[。；]',
]
PRINCIPLE_PAT = [
    r'(?:治则|治法|治疗原则|治疗方法)(?:为|是|：|:|应以|宜|当)\s*([^。；\n]{2,40})',
    r'(?:采用|采取|运用|通过|予以?|给予|以)了?([^，。；\n]{2,30}?)(?:的)?(?:治则治法|治疗原则|治疗方法|治法|治则|为主|为治)',
    r'治宜([^。；，\n]{2,30})',
    r'法当([^。；，\n]{2,30})',
    r'治以([^。；，\n]{2,30})',
    r'治疗(?:上)?(?:应|应当|宜)(?:以)?([^。；，\n]{2,30})',
    r'(?:治疗方案|治疗方法|治疗)(?:主要)?(?:是|为)(?:以)?([^。；\n]{2,40}?)(?:为主|[。；]|，少佐)',
]
EXPLICIT_PRINCIPLE = re.compile(r'治疗原则|治则|治法|治疗方法|治疗应|治疗以|治宜|法当|治以')

def _first(patterns, text):
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return _clean(m.group(1)), m.group(0)
    return None, None

def _source_sentence(text, fragment):
    for s in sentences(text):
        if fragment and fragment[:6] in s:
            return s
    return fragment or ''

def draft_diagnosis_points(analysis):
    """Disease + syndrome + one pathogenesis statement from a case analysis."""
    analysis = analysis or ''
    pts = []
    disease, hit = _first(DISEASE_PAT, analysis)
    if disease:
        disease = re.split(r'西医|，|,', disease)[0]
        for d in split_terms(disease, 2, 12):
            pts.append({'type': '疾病诊断', 'text': d, 'source': _source_sentence(analysis, hit)})
    syndrome, hit = _first(SYNDROME_PAT, analysis)
    if syndrome:
        syndrome = re.split(r'[，,]?\s*(?:治则|治法|治疗原则|治疗方法|中医诊断|西医)', syndrome)[0]
        for s in split_terms(syndrome, 2, 14):
            if s not in [p['text'] for p in pts]:
                pts.append({'type': '证候', 'text': s, 'source': _source_sentence(analysis, hit)})
    used = {p['source'] for p in pts}
    mech = [s for s in sentences(analysis)
            if s not in used and not re.search(r'西医|中医(?:的)?诊断|证候(?:为|是)', s)
            and len(re.findall(r'[虚实寒热湿痰瘀郁气血阴阳风火毒]', s)) >= 2]
    if mech:
        text = _clean(mech[0])
        text = re.sub(r'^(?:根据|经过|通过)?辨证分析[，,：:]?', '', text)
        pts.append({'type': '病机', 'text': text, 'source': mech[0], 'needs_condensing': len(text) > 30})
    return pts

def draft_principle_points(analysis, treatment):
    """Treatment-principle atoms; only when the source states a principle explicitly."""
    pts = []
    for field in (analysis or '', treatment or ''):
        phrase, hit = _first(PRINCIPLE_PAT, field)
        if phrase:
            phrase = re.split(r'的治疗|来治疗|进行治疗|治疗', phrase)[0]
            for a in split_terms(phrase, 2, 14, split_he=False):
                if any(ch in a for ch in TCM_ACTION) and a not in [p['text'] for p in pts]:
                    pts.append({'type': '治疗原则', 'text': a, 'source': _source_sentence(field, hit)})
            if pts:
                break
    return pts

def atomize_principle_sentence(sentence):
    phrase, hit = _first(PRINCIPLE_PAT, sentence)
    if not phrase:
        return []
    phrase = re.split(r'的治疗|来治疗|进行治疗|治疗', phrase)[0]
    return [a for a in split_terms(phrase, 2, 14, split_he=False) if any(ch in a for ch in TCM_ACTION)]

STRONG_LEAK = re.compile(r'证属|证为|证候|辨证|诊断为|诊断是|病机|治则|治法|治疗原则|属于.{0,8}证|所致|乃.{0,6}所致|系.{0,6}所致')
WEAK_LEAK = re.compile(r'导致|引起|由于|因为|以致|致使|因此|所以|故|化热|化火|化燥|生痰|无力|内蕴|侵袭|郁结|不足|亏虚|失调|失养|阻滞|内阻|内停|上扰|上犯|下注|佐证|判断')
TCM_TERMS = re.compile(r'[虚实寒热湿痰瘀郁气血阴阳风火毒]')

def leak_fragments(background, points):
    """Clauses in the question background that may reveal an answer point.
    Disease names are not used as leak terms because many TCM disease names
    coincide with the chief complaint (e.g. 头痛, 发热)."""
    clauses = [c for c in re.split(r'[；;。]', background) if c.strip()]
    flagged = []
    terms = [p['text'] for p in points if p['type'] != '疾病诊断' and 2 <= len(p['text']) <= 12]
    for c in clauses:
        hit_terms = [t for t in terms if t in c]
        weak = WEAK_LEAK.search(c) and len(TCM_TERMS.findall(c)) >= 2
        if hit_terms or STRONG_LEAK.search(c) or weak:
            flagged.append(c.strip())
    return flagged


NAME_PREFIX = re.compile(r'^(?:根据(?:中医的)?辨证分析[，,：:]?|经过辨证分析[，,：:]?|辨证分析[：:，,]?)?'
                         r'(?:[\u4e00-\u9fff]{1,3}医生认为)?(?:[\u4e00-\u9fff]{1,2}某(?:女士|先生|患者|小朋友)?(?:的)?)?')


def polish_points(points):
    """Cosmetic clean-up of drafts after selection: drop analysis openers and patient
    names, and re-type statements that are really a disease diagnosis."""
    out, seen = [], set()
    for p in points:
        p = dict(p)
        m = re.search(r'中医(?:的)?诊断(?:是|为)([^，。；,;]{1,12})', p['text'])
        if p['type'] in ('病机', '疾病诊断') and m and len(p['text']) <= 20:
            p['type'], p['text'] = '疾病诊断', _clean(m.group(1))
        else:
            p['text'] = _clean(NAME_PREFIX.sub('', p['text'])) or p['text']
        key = (p['type'], p['text'])
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out
