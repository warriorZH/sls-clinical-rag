"""Prepare and independently validate the phase-1 adjudication decisions (CSV authored in JS)."""
import csv, json, re, sys, hashlib
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from annot_io import clauses, split_multi, parse_point_list, normalize_text, POINT_TYPES
from adjudicate import apply_edits, _multi

SRC = ROOT / '仲裁/phase1_仲裁表.csv'
OUT = ROOT / '仲裁/phase1_仲裁表_v2_已仲裁.csv'
WORK = ROOT / '仲裁/.phase1_v2_work'
if not OUT.exists() and (WORK/'validation.json').exists():
    saved=json.loads((WORK/'validation.json').read_text(encoding='utf-8'))
    backup=ROOT/'仲裁/phase1_仲裁表.bak.csv'
    if SRC.exists() and backup.exists() and hashlib.sha256(SRC.read_bytes()).hexdigest()==saved['新表sha256'] and hashlib.sha256(backup.read_bytes()).hexdigest()==saved['原表sha256']:
        OUT,SRC=SRC,backup
COLS = ['【仲裁】最终纳入(1/0)', '【仲裁】最终删除片段(||分隔)', '【仲裁】最终补充信息(||分隔)', '【仲裁】最终要点列表(类型:内容,用；分隔)', '【仲裁】备注']

# Each override was reviewed against the three expert lists and the original analysis.
DECISIONS = {
'ST008': ('疾病诊断:消渴；病机:饮食伤脾；病机:升降失调；病机:痰湿内蕴', 'A/C均提出升降失调，原文明确，属不同病理环节，予以保留。'),
'ST011': ('疾病诊断:盗汗；病机:余邪未清；证候:阴虚湿滞；病机:营血有热；病机:心阴受损', '采用B的核心内容；余邪未清描述余邪未尽的致病状态，按C改为病机；营血有热与心阴受损并非同义。'),
'ST023': ('治疗原则:温阳化气；治疗原则:利水；治疗原则:活血', '温阳化气整体保留；利水与活血为可独立核验的并列治法，按A拆分。'),
'ST026': ('疾病诊断:不孕；病机:阴血不足；病机:热伏冲任；证候:阴虚火旺', 'A/C的阴血不足为原文病因；不将阴虚内热与阴虚火旺重复计权。'),
'ST027': ('疾病诊断:咳嗽；病机:脾失健运；病机:湿浊内生；病机:湿郁化热；病机:肺失宣降', '以B为主，保留A/C有原文依据的脾运障碍；将酿成湿热规范为湿郁化热，保留病因到肺失宣降的不同环节。'),
'ST032': ('疾病诊断:痹证；病机:湿热蕴毒；病机:热伤血络；证候:湿热内蕴；证候:瘀血阻络', 'B复合病机按A/C拆为两条；证候按B/C分别核验，不将证候和致病关系相互替代。'),
'ST035': ('疾病诊断:胃痛；病机:肝气郁结；病机:郁热犯胃；病机:胃失通降', '综合B的肝郁化热和A/C的胃失通降；将化热及犯胃合为保留因果关系的郁热犯胃，避免同一热邪重复计权。'),
'ST037': ('疾病诊断:眩晕；证候:气阴两虚；病机:阳气偏亢于上；病机:肾气不足', '内容以B为主；阳气偏亢于上为病理动态，依A/C改为病机；肾气不足维持B的病机类型。'),
'ST044': ('疾病诊断:梅核气；病机:肺胃不降；病机:肝气郁滞；病机:痰气交阻；证候:脾肾阳虚', '采用A/C的类型：肺胃不降是升降失常病机，脾肾阳虚是完整证候；不因原文写在证候后即机械定类。'),
'ST047': ('疾病诊断:眩晕；疾病诊断:泄泻；证候:脾胃虚弱；病机:清浊不分；病机:痰浊上蒙清窍', '以B为主，将脾胃虚弱按A改为证候；清浊不分与痰浊蒙窍为原文两个独立机制。'),
'ST051': ('疾病诊断:胃痞；病机:肝气犯胃；病机:痰气积聚', '纳入从B；原病情否认畏寒，分析却以畏寒论脾阳虚，且淡绛少苔不能直接确证虚寒；依规范3.4.5剔除依赖冲突的脾胃虚寒，保留可分离、有原文明示的病名与病机。'),
'ST055': ('疾病诊断:胸痹；证候:气虚；证候:痰瘀互阻', '采用B的本虚标实内容，按C区分两个完整证候；气虚为本精简为气虚，不将本标关系另设要点。'),
'ST056': ('疾病诊断:瘾疹；疾病诊断:虚劳；证候:风热犯表；证候:气阴两虚；病机:虚热内生', '采用A/C共识；B新增内外合邪犯表与已保留的风热犯表及虚热内生重复，不另计权。'),
'ST060': ('疾病诊断:咳嗽；证候:风寒外束肺肾阴虚；病机:肺失宣降；病机:肾不纳气', '原文将风寒外束肺肾阴虚明确作为完整复合证候，按A整体保留；采用B的两条核心病机。'),
'ST064': ('疾病诊断:不寐；病机:肝气郁结；病机:肝郁乘脾；病机:肝阴耗伤', '以B为主；A/C均提出且原文明确的肝阴耗伤予以保留，肝郁乘脾已涵盖本题脾运失职的上下位说明。'),
'ST066': ('证候:肾虚血瘀；病机:肾虚腰府失养；病机:瘀血阻滞', '以B为主，肾虚外府失荣规范为肾虚腰府失养；子宫次全切后为手术状态，不另立中医病名。'),
'ST069': ('疾病诊断:不孕；病机:水湿内停化热；病机:湿热阻滞冲任；证候:脾肾气虚；证候:湿热内蕴', '采用B的病理演变，结合C为冲任阻滞补明原文湿热病因，避免与一般湿热内蕴同层重复。'),
'ST071': ('疾病诊断:白疕；病机:脾虚湿盛；病机:湿郁化热；病机:营血耗损', '采用A/C；原文为内外相搏耗损营血后产生风燥，B的风燥伤营倒置因果，不采用。'),
'ST079': ('疾病诊断:痹证；病机:肝肾不足；病机:瘀浊阻络；病机:筋骨失养', 'B的气血瘀浊阻络与A/C的瘀浊阻络指同一过程，按A/C精简，不增设气滞断言。'),
'ST082': ('疾病诊断:肺胀；病机:肺脾两虚；病机:痰热瘀互结；病机:肺失宣降', '以B为主；A/C均保留且原文明确的肺之宣降受阻为核心病机，予以保留。'),
'ST085': ('疾病诊断:腹胀；病机:脾失健运；病机:湿热内生；病机:湿阻气机', '以B为主；A/C均提出湿阻气机，原文明确，且区别于湿热内生，保留。'),
'ST086': ('疾病诊断:咳嗽；证候:肺阴亏虚；病机:虚热内灼；病机:肺失润降', '保留B的阴亏虚热核心，按A将肺阴亏虚列证候；阴虚内热与虚热内灼不重复，保留A/C的肺失润降。'),
'ST087': ('疾病诊断:血尿；证候:下焦湿热；病机:湿热伤络；病机:热迫血行', '以下焦湿热保留B证候；伤络与迫血妄行按A/C各立病机，不再重复一般湿热蕴郁下焦。'),
'ST089': ('疾病诊断:不孕；病机:冲任虚弱；证候:肝肾阴虚；证候:热毒蕴结；病机:瘀阻冲任', '以B为主；复合病机肝肾不足、冲任虚弱精简为冲任虚弱，前者与已保留肝肾阴虚高度重叠，避免重复计权。'),
'ST093': ('疾病诊断:湿疮；证候:湿热内壅；病机:复感风邪；病机:湿热蕴蒸肌肤', '以B为主；湿热内盛与湿热内壅合并；蕴蒸肌肤补明原文湿热主体，保留部位迁移机制。'),
'ST097': ('疾病诊断:脊痹；病机:肾气不足；病机:督脉空虚；病机:风寒入里化热；病机:痹阻脉络', '以B为主，将肾气不足与督脉空虚按A/C拆为两个独立断言；化热的病因关系保留。'),
'ST103': ('治疗原则:行气；治疗原则:活血化瘀；治疗原则:调经；治疗原则:祛斑', '采用A的并列治法拆分；活血化瘀为一个既定治法单位，不拆为活血与化瘀。'),
'ST106': ('疾病诊断:胸痹心痛；证候:肝肾阴虚；病机:阴虚脉道涩滞；病机:痰瘀阻滞胸中；证候:痰瘀互结', '以B为主；一般气血不畅以A更具体且原文明确的痰瘀阻滞胸中替代，保留不同病理环节。'),
'ST109': ('疾病诊断:眩晕；疾病诊断:水肿；证候:肾虚水泛；病机:浊气上犯清窍；证候:肾阳虚弱', '采用A/B，肾虚水泛和肾阳虚弱不机械合成为原文未直接命名的肾阳虚水泛。'),
'ST113': ('疾病诊断:血证；病机:脾胃气虚；病机:湿热内阻；病机:阴虚内热；病机:热迫血行', '以B为主，补回A/C均提出且原文明确的脾胃气虚；本题这些内容位于病机分析，沿用B的病机尺度。'),
'ST119': ('疾病诊断:胃脘痛；病机:湿热内生；病机:胃腑气机不通', '采用B的核心机制，复合气机阻滞、胃腑不通按C压缩为一条，不将同一局部阻滞重复计权。'),
'ST125': ('治疗原则:补气养血；治疗原则:止血；治疗原则:调经', '补气养血按既定治法保留，止血和调经可独立核验，按A拆分。'),
'ST126': ('治疗原则:健脾补肾；治疗原则:调和冲任；治疗原则:固涩止血', '以B的总结治则为主；A/C提出的固涩止血在原始分析中明确写为治疗要求，非据药物反推，保留。'),
'ST133': ('疾病诊断:肺风粉刺；病机:相火过旺；病机:冲任不调；证候:肝肾不足；证候:阴虚内热；病机:肺胃火热上蒸', '以B为主；复合病机中的肾阴不足与已保留肝肾不足及阴虚内热重叠，精简为相火过旺；肺胃火热上蒸为明确独立机制。'),
'ST152': ('疾病诊断:妇人腹痛；疾病诊断:不孕；证候:气滞血瘀；病机:胞宫胞脉受损；病机:胞脉瘀阻', '保留B/C明确诊断证候；A/C共同提出的手术后胞宫胞脉损伤为核心病因，保留；冲任血瘀及胞脉闭阻按A合为胞脉瘀阻，肝郁气滞不与气滞血瘀重复。'),
'ST153': ('疾病诊断:急劳；病机:化疗耗伤气阴；病机:阴血亏虚', '病名采用B/C的急劳，原文同时写虚劳，按上下位关系不双计；依A/C保留化疗耗伤气阴的明确病因，不能丢失化疗关系。'),
'ST163': ('治疗原则:养肝；治疗原则:益气血；治疗原则:宁心；治疗原则:疏风', '以B为主；A/C的疏风在原始分析的治疗重点中明确提出，非据方药推导，补回。'),
'ST165': ('疾病诊断:腰痛；病机:外伤致气血阻滞', 'B的腰部挫伤和气血阻滞按C合为保留外伤因果关系的单一病机。'),
'ST167': ('疾病诊断:崩漏；病机:湿热内蕴；证候:血热夹瘀；病机:热迫血行', '病名采用B及原文明示的崩漏，不将经间期出血直接等同崩漏；血热夹瘀按A/C改为证候；热迫血行为原文明确的出血机制。'),
'ST168': ('疾病诊断:水肿；病机:湿邪化热伤络动血；病机:脾肾阳虚；病机:外邪引动内饮', '采用B，湿邪化热、伤络动血压缩为保留致血尿关系的单一病机，不丢失伤络动血。'),
'ST172': ('治疗原则:补血养血；治疗原则:行气；治疗原则:安神', '补血养血按B/C保留原文表达；行气和安神为可独立核验的并列治法，按A拆分。'),
'ST175': ('疾病诊断:中风；证候:气虚痰热血瘀；病机:脾肾亏虚；病机:痰浊化热阻于脑脉；病机:气虚致瘀', '以B为主，完整证候不拆；末项采用A的气虚致瘀，保留原文气推无力而成瘀的关系，避免与证候仅字面重复。'),
'ST177': ('治疗原则:养血；治疗原则:温肾；治疗原则:通络止痛', '养血与温肾作用不同，可独立核验，按A拆分；通络止痛按既定治法单位保留。'),
'ST178': ('治疗原则:补脾益肾；治疗原则:益气养血；治疗原则:活血；治疗原则:泻浊', '补脾益肾和益气养血整体保留；活血与泻浊为独立并列治法，按A拆分。'),
}

SPECIAL_NOTES = {
'ST002':'纳入采用B。原始病情末尾截断且未记录具体舌脉，保留胃痛、反酸和胃镜结果作为可用信息；不得将分析中的舌红苔白脉数稍滑补入病情，金标仍限于原文答案，辨证适用性存在信息限制。',
'ST151':'纳入采用B，治则有原文明确表达。原病情未载绿色便、尿少、多汗，舌脉写无异常，与分析的舌红不一致；这些信息不补入，清热化湿的病例适用依据偏弱，保留该限制供后续使用。',
'ST036':'原分析有肾虚使肝火旺盛等不严谨因果；只保留原文明示诊断、证候和痰湿郁积化热，不扩展该因果。',
'ST081':'原文气血阳虚表述不够规范，沿用B的来源性断言，不另补气血两虚或阴阳两虚等新证名。',
'ST116':'采用B保留来源用字湿热淤滞；不擅自改为湿热郁滞，二者不据字近自动视为同义。',
'ST057':'原始病情字段混入辨证分析；大汗出、渴望饮冷水只出现于其中分析段，不能视为古籍核实后的实测症状。',
}

# Exact source substrings. Preserve observations when only an inference suffix leaks the answer.
DELETIONS = {
'ST021':['他住院接受中药汤剂的补肾扶正、活血化瘀治疗'],
'ST045':['受风寒之邪'],
'ST057':['患儿的肌表受到风热之邪的侵袭','导致出现恶寒、发热等症状','风热邪气上攻咽喉','舌脉的表现也符合风热外感的典型特征','表明体内有热邪郁结'],
'ST067':['等外感邪毒和内有湿热的征象'],
'ST075':['是由感染湿热疫毒侵蚀肝脏引起的','因此出现了'],
'ST080':['采用清热化痰；活血通络法治疗了一位中风患者','和口服化痰通络开窍中药'],
'ST131':['久而化热为痰','因此出现咳嗽和咳痰的症状','气虚无力行血','舌脉的情况也佐证了这一判断'],
'ST135':['内蕴湿热','现在受到外寒的袭击','寒气侵袭太阳经络','导致项痛、肩痛、胸痛彻背','导致便燥已持续3天未行'],
'ST158':['等症状都属于肝肾阴虚的表现','导致气血上行','从而引起头晕头胀的症状','夜间无法进入阴','因此导致夜间睡眠不安'],
'ST166':['根据症状分析','无法主持气机和纳气','因此出现','他的肺肾阴虚导致','肺气不足和血液循环受阻导致气虚血瘀','也都是气阴两虚的表现'],
'ST179':['综合脉证和四诊合参的结果','血液停留在脉外形成瘀血','下肢的脉络也会受到影响','舌质紫黯和脉象弦涩都是瘀血内停的征象'],
'ST180':['等症状与肾虚肝郁有关'],
}
DEL_NOTES = {
'ST045':'C提出的受风寒之邪为辨证性质的邪气判断，删除；饱食等客观诱因保留。',
'ST067':'只删结论后缀，保留脉弦滑。',
'ST075':'删病机及因果引导语，保留小便黄及舌脉客观描述。',
'ST135':'按A删除因果性分析整句；项痛、胸痛及便燥已在前文记录，避免残留导致字样。',
'ST158':'仅删诊断后缀及推断，保留头晕、视力模糊、腰酸、面部潮红等症状。',
'ST166':'删除病机前后缀并保留气喘声低、痰液黏稠及舌脉等客观表现。',
'ST180':'仅删肾虚肝郁判断后缀，保留舌尖红；草稿另有舌苔细数这一来源错误，予以标记，未据此新增要点。',
}

DROP_ADDS = {
'ST040': {'目前'}, 'ST041': {'此前'},
'ST095': {'出生于1960年10月23日','于2009年11月19日初诊妇科门诊'},
'ST121': {'一周前'},
}

REFS = {
'R1': ('205-金匮翼.txt',4994,5008,'痹的风寒湿及经络、筋脉阻滞边界'),
'R2': ('437-黄帝内经素问.txt',1404,1410,'痹的病因与病位；不据此新增病例湿邪'),
'R3': ('575-医宗金鉴.txt',19292,19303,'心脾及心神失养概念；不凭方剂反推治则'),
'R4': ('575-医宗金鉴.txt',27714,27719,'盗汗有心虚、心火伤阴等分支，不机械视为单一证候'),
'R5': ('575-医宗金鉴.txt',30302,30312,'崩漏的描述边界，不能证明所有经间期出血同义'),
'R6': ('207-医门法律.txt',154,159,'胸痹与通阳概念边界，不把症状直接等同治法'),
'R7': ('502-金匮玉函要略述义.txt',770,776,'开结豁痰、通阳的语义边界'),
}
ROW_REFS = {'ST001':['R1','R2'],'ST011':['R4'],'ST063':['R3'],'ST102':['R1','R2'],'ST167':['R5'],'ST073':['R6','R7'],'ST106':['R6'],'ST135':['R1']}

def addition_decision(row, deletions):
    per = {a: dict(clauses(split_multi(row[a+'_补充信息']))) for a in 'ABC'}
    keys = list(dict.fromkeys(k for a in 'ABC' for k in per[a]))
    raw = row['原始病情描述(参考)']
    edited, _ = apply_edits(row['检索用问题文本(草稿)'], deletions, [])
    kept, audit = [], []
    for key in keys:
        who = ''.join(a for a in 'ABC' if key in per[a])
        s = next(per[a][key] for a in 'ABC' if key in per[a])
        reason = '保留'
        if len(who) < 2: reason = '不足两人提出'
        elif s not in raw: reason = '非原始病情连续片段'
        elif s in DROP_ADDS.get(row['题目编号'], set()): reason = '非关键病情或孤立时间词'
        elif s in edited: reason = '问题中已具备，去重'
        if reason == '保留': kept.append((raw.index(s),raw.index(s)+len(s),s))
        audit.append({'分句':s,'提出者':who,'人数':len(who),'裁定':reason})
    # Rejoin adjacent selected clauses using the exact punctuation found in the source.
    groups = []
    for start,end,s in sorted(kept):
        if groups and start >= groups[-1][1] and re.fullmatch(r'[，,。；;、：:\s|]*', raw[groups[-1][1]:start]):
            groups[-1] = (groups[-1][0],end,raw[groups[-1][0]:end])
        else: groups.append((start,end,s))
    return [s for _,_,s in groups], audit

def prepare():
    rows = list(csv.DictReader(SRC.open(encoding='utf-8-sig',newline='')))
    original_hash = hashlib.sha256(SRC.read_bytes()).hexdigest()
    output, audit = [], []
    for row in rows:
        qid=row['题目编号']; out=row.copy(); notes=[]
        out[COLS[0]]=row['B_纳入']
        dels=DELETIONS.get(qid,_multi(row[COLS[1]]))
        if not row[COLS[1]].strip() and qid not in DELETIONS:
            dels=split_multi(row['B_删除片段'])
        out[COLS[1]]=' || '.join(dels) or '无'
        adds, add_audit=addition_decision(row,dels)
        out[COLS[2]]=' || '.join(adds) or '无'
        if qid in DECISIONS:
            points, why=DECISIONS[qid]; notes.append('要点：'+why)
        else:
            points=row['B_最终要点']
            if '要点' in row['需仲裁字段']:
                notes.append('要点：对照原始分析，采用B的内容及类型；同义措辞、额外症状解释不另增权重。')
        # Excluded questions retain expert-sourced candidate points, but never enter the gold set.
        if out[COLS[0]]=='0':
            notes.append('三人一致排除：'+(row['B_排除原因'] or row['A_排除原因'])+'；所列要点仅存档，不进入纳入题金标。')
        out[COLS[3]]=points or '无'
        if qid in SPECIAL_NOTES: notes.insert(0,SPECIAL_NOTES[qid])
        if qid=='ST051' and qid in DECISIONS: pass
        elif '纳入' in row['需仲裁字段'] and qid not in SPECIAL_NOTES: notes.append('纳入采用B。')
        if '删除片段' in row['需仲裁字段'] or qid in DEL_NOTES:
            notes.append('删除：'+DEL_NOTES.get(qid,'采用B的原文可匹配片段，删除答案、病机、治则及残余分析语。'))
        if row['A_补充信息'] or row['B_补充信息'] or row['C_补充信息']:
            votes=sum(x['裁定']=='保留' for x in add_audit)
            notes.append(f'补充：按分句至少2人提出并核对原始病情，保留{votes}个分句；已有信息去重，单人意见不因B优先而豁免。')
        if qid in ROW_REFS: notes.append('术语校验：'+ '、'.join(ROW_REFS[qid])+'（出处见仲裁说明；不作为本病例额外证据）。')
        out[COLS[4]]=' '.join(notes) or '三人一致，核对原始分析后保留。'
        output.append(out)
        audit.append({'题目编号':qid,'补充分句':add_audit})
    WORK.mkdir(exist_ok=True)
    payload={'headers':list(rows[0]),'rows':output,'source_sha256':original_hash,'addition_audit':audit}
    (WORK/'decisions.json').write_text(json.dumps(payload,ensure_ascii=False,indent=1),encoding='utf-8')
    print(json.dumps({'rows':len(output),'included':sum(r[COLS[0]]=='1' for r in output),'points':sum(len(parse_point_list(r[COLS[3]])) for r in output if r[COLS[0]]=='1'),'overrides':len(DECISIONS),'payload':str(WORK/'decisions.json')},ensure_ascii=False))

def validate():
    src=list(csv.DictReader(SRC.open(encoding='utf-8-sig',newline='')))
    dst=list(csv.DictReader(OUT.open(encoding='utf-8-sig',newline='')))
    data=json.loads((WORK/'decisions.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(SRC.read_bytes()).hexdigest()==data['source_sha256']
    assert len(src)==len(dst)==180
    assert list(src[0])==list(dst[0])
    additions=0
    for old,new,expected,detail in zip(src,dst,data['rows'],data['addition_audit']):
        qid=old['题目编号']
        assert new==expected, qid+' export mismatch'
        for col in old:
            if col not in COLS: assert old[col]==new[col],(qid,col)
        assert all(new[c].strip() for c in COLS),(qid,'blank decision')
        assert new[COLS[0]] in ('0','1')
        text,missing=apply_edits(old['检索用问题文本(草稿)'],_multi(new[COLS[1]]),_multi(new[COLS[2]]))
        assert not missing,(qid,missing)
        per={a:dict(clauses(split_multi(old[a+'_补充信息']))) for a in 'ABC'}
        for frag in _multi(new[COLS[2]]):
            assert frag in old['原始病情描述(参考)'],(qid,frag)
            for k,s in clauses([frag]):
                assert sum(k in per[a] for a in 'ABC')>=2,(qid,s,'under 2 votes')
                additions+=1
        if new[COLS[0]]=='1':
            pts=parse_point_list(new[COLS[3]])
            allowed={'治疗原则'} if old['题型']=='治疗原则' else set(POINT_TYPES)-{'治疗原则'}
            assert pts and all(t in allowed for t,s in pts),(qid,pts)
            assert len({normalize_text(s) for t,s in pts})==len(pts),(qid,'duplicate')
    counts=Counter()
    for r in dst:
        if r[COLS[0]]=='1': counts.update(t for t,s in parse_point_list(r[COLS[3]]))
    summary={'总题数':180,'原需仲裁':sum(r['是否一致']=='需仲裁' for r in src),'已填仲裁字段':180*5,'纳入':sum(r[COLS[0]]=='1' for r in dst),'排除':[r['题目编号'] for r in dst if r[COLS[0]]=='0'],'纳入题要点数':sum(counts.values()),'要点类型':dict(counts),'最终有补充信息题数':sum(r[COLS[2]]!='无' for r in dst),'保留补充分句数':additions,'有删除片段题数':sum(r[COLS[1]]!='无' for r in dst),'原表sha256':data['source_sha256'],'新表sha256':hashlib.sha256(OUT.read_bytes()).hexdigest(),'验证':'通过：行列/顺序/原始意见完整；仲裁列无空缺；补充至少两票、连续来自病情；删除可匹配；要点类型、题型和去重检查通过。'}
    (WORK/'validation.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))

def integration_and_report():
    import tempfile, shutil, contextlib, io
    from adjudicate import p1_finalize, SEALED
    data=json.loads((WORK/'decisions.json').read_text(encoding='utf-8'))
    stats=json.loads((WORK/'validation.json').read_text(encoding='utf-8'))
    # Use the unchanged production finalizer in an isolated root. Never replace frozen gold files.
    with tempfile.TemporaryDirectory(prefix='phase1_v2_check_') as directory:
        tmp=Path(directory); (tmp/SEALED).mkdir(); (tmp/'仲裁').mkdir()
        shutil.copyfile(ROOT/SEALED/'sealed_questions_full.json',tmp/SEALED/'sealed_questions_full.json')
        shutil.copyfile(OUT,tmp/'仲裁/phase1_仲裁表.csv')
        with contextlib.redirect_stdout(io.StringIO()): p1_finalize(tmp,tmp/'仲裁')
        final=json.loads((tmp/SEALED/'final_questions.json').read_text(encoding='utf-8'))
        assert len(final)==180
        assert sum(q['included'] for q in final)==176
        assert sum(len(q['points']) for q in final if q['included'])==547
    rejected=Counter(x['裁定'] for q in data['addition_audit'] for x in q['补充分句'])
    differences=Counter()
    rows=data['rows']
    old=list(csv.DictReader(SRC.open(encoding='utf-8-sig',newline='')))
    for before,after in zip(old,rows):
        for c in COLS:
            if before[c]!=after[c]: differences[c]+=1
    report=[
        '# Phase 1 仲裁说明 v2',
        '',
        f'日期：2026-10-25。正式使用文件：`{OUT.name}`（UTF-8 BOM）。原始 `{SRC.name}` 保留供追溯。仲裁结果先另存为v2文件，核验后当前工作目录已将其设为正式表；通过SHA-256确认内容完全一致。',
        '',
        '## 结果',
        '',
        '180题全部完成，其中原标记需仲裁167题。最终纳入176题、排除4题（ST016、ST100、ST107、ST170，三人一致排除）。纳入题共547条要点：疾病诊断108、证候93、病机181、治疗原则165。',
        '',
        '127题保留补充信息，共400个分句。26题有最终删除片段。“需仲裁字段”“是否一致”保留为原始分歧记录，不改写为处理状态；完成状态以已填满的5个【仲裁】字段为准。',
        '',
        '## 裁定规则',
        '',
        '1. 补充信息：沿用现有脚本的分句尺度，按中文/英文标点及空白拆分，每人对同一分句最多计一票；只保留至少2人提出、能在原始病情连续匹配的分句。B的单人补充同样不予保留。去除已有信息、出生/就诊日期和孤立时间词；原文相邻且都通过的分句用原标点合并，避免将FSH、LH或检查数值与单位割裂。分句门槛是必要条件，不是所有两票内容都必须增补。',
        '2. 纳入与要点：分歧较大以B为主，结合A/C意见逐题对照原始分析；对明显因果倒置、类型误置、重复计权或依赖冲突的内容修正，并在对应题备注中解释。没有新的临床症状或新证名来自古籍。',
        '3. 独立的并列治法按规范拆分；既定治法和完整复合证候整体保留。每题2至5条是建议值，不为满足数量强删独立必要内容。',
        '4. 删除片段必须原样匹配草稿；优先只删除诊断、治则及病机推断，保留可分离的客观症状、舌脉。删除与补充均使用原表的 || 格式。',
        '',
        '## 三道纳入争议题',
        '',
        '| 题号 | 结果 | 依据与限制 |',
        '|---|---|---|',
        '| ST002 | 纳入 | 从B；用原病情的胃痛、反酸、胃镜资料。病情末尾截断且没有实测舌脉，不把分析中的舌红苔白脉数稍滑倒填。保留信息不足限制。 |',
        '| ST051 | 纳入 | 从B保留可分离断言；删除依赖畏寒矛盾的脾胃虚寒，仅保留胃痞、肝气犯胃、痰气积聚。病情的否认畏寒不能被分析覆盖。 |',
        '| ST151 | 纳入 | 从B，保留明示治则清热化湿、调理脾胃；分析中的绿便、尿少、多汗未载于病情，舌红与无异常记载不一致，不补入，记录适用依据偏弱。 |',
        '',
        '这三题的保留体现本次用户指定的B优先裁定，不表示已消除来源记录中的缺失或矛盾。',
        '',
        '## 参考资料核验',
        '',
        '参考目录：`C:/Users/<user>/Desktop/UUM/tcmData/ref`。以下行号以GB18030解码原文件后按splitlines()从1计数。古籍只校验术语边界，不能证明现代病例的完整诊断或提供候选证据外的新症状；旧专家备注中的其他引用保留为原始意见，未统一视为已经复核。',
        '',
        '| 编号 | 文件 | 行号 | 本次用途 |',
        '|---|---|---|---|',
    ]
    for key,(file,lo,hi,purpose) in REFS.items():
        report.append(f'| {key} | {file} | {lo}—{hi} | {purpose} |')
    report += [
        '',
        '核验要点：《金匮翼》4996行及《素问》1404行论风寒湿合而为痹，5004—5005行区分脉、筋等病位；不能据此给病例额外加上湿邪。《医宗金鉴》归脾相关段落区分心脾受损及表现，盗汗段落指出心虚不固与心火伤阴等分支。崩漏总括区分淋沥不断与忽然大下，故ST167保留原文明示病名“崩漏”但记录与经间期出血的边界，未将二者普遍定义为同义。《医门法律》及《金匮玉函要略述义》用于核对通阳、开结、豁痰概念，不据方药反推本题治则。',
        '',
        '## 非直接照抄B的要点裁定',
        '',
        '其余要点分歧以B为主要裁定基础，具体最终文本和备注均见CSV。以下列出44题内容、类型、拆分或去重调整，以及维持A/B一致而拒绝C合并的边界裁定。',
        '',
        '| 题号 | 处理理由 |',
        '|---|---|',
    ]
    for qid,(_,note) in DECISIONS.items(): report.append(f'| {qid} | {note} |')
    report += [
        '',
        '## 完整性与兼容性校验',
        '',
        '- 180行、34列、题号及行列顺序保持；29个原始/参考字段逐单元格与原表一致。只填写或裁定5个【仲裁】字段。',
        '- 900个仲裁单元格均非空；“无”明确代表无删除/无补充，排除题的原有候选要点只作存档，不进入纳入题金标。',
        '- 每条补充均可在原始病情连续匹配，拆回分句后逐条验证至少2位专家提出；所有删除片段均能在草稿匹配。',
        '- 纳入题要点类型合法且符合诊断题/治则题边界，单题内容去重通过。',
        '- 在临时目录内运行未修改的scripts/adjudicate.py的p1_finalize，成功生成180题、纳入176题、纳入要点547条，未写入项目的封存金标。',
        '- CSV没有工作簿格式、公式或图形，按数据记录核验；经Artifact Tool导入、编辑并读取后输出CSV，另用Python独立回读核对。',
        '',
        '补充分句处理统计（统计范围为三人提出分句的并集）：'+ '；'.join(f'{key}：{value}' for key,value in rejected.items())+'。',
        '',
        ('后续定稿：现有finalize命令固定读取“仲裁/phase1_仲裁表.csv”，当前正式表已是本次核验通过的仲裁版本。' if OUT.name=='phase1_仲裁表.csv' else '后续定稿注意：现有finalize命令固定读取“仲裁/phase1_仲裁表.csv”，运行前需将新版本作为其输入。')+'本次没有重算专家原始一致性，仲裁后的结果不能当作独立标注一致性。',
        '',
        '原表SHA-256：`'+stats['原表sha256']+'`',
        '',
        '新表SHA-256：`'+stats['新表sha256']+'`',
    ]
    target=ROOT/'仲裁/phase1_仲裁说明_v2.md'
    target.write_text('\n'.join(report)+'\n',encoding='utf-8')
    print('官方定稿函数临时目录兼容性校验通过；已保存：'+str(target))

if __name__=='__main__':
    if '--report' in sys.argv: integration_and_report()
    elif '--validate' in sys.argv: validate()
    else: prepare()
