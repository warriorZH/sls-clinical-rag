"""Evaluate frozen three-seed Top-2 outputs after complete union annotation."""
import json,csv,collections,random,statistics,sys,hashlib
from pathlib import Path
B=Path(__file__).resolve().parent
sys.path.insert(0,str(B.parents[1]/'.local-tools/analysis'))
def load(n): return json.loads((B/n).read_text(encoding='utf-8'))
def mean(x): return statistics.mean(x)
fields=['coverage','full_coverage','support_precision','relevance_precision','partial_rate','conflict_rate']
tasks={t['task_id']:t for t in load('CODEX_ALL_TASK_STANDARDS.json')['tasks'] if t['eligible'] and t['split']=='test'}
groups=load('DIRECT_CASE_GROUPS.json'); assert len(tasks)==48
labels=[json.loads(s) for s in (B/'CODEX_ALL_SEED_OUTPUT_REVIEW_QUEUE.jsonl').read_text(encoding='utf-8').splitlines()]
lm={(r['candidate_id'],r['variant']):r for r in labels if r['split']=='test'}
assert all(r['y_support'] is not None and r['annotation_uncertain']==0 for r in lm.values())
results={}; summary=[]
for s in load('DIRECT_ALL_SEED_SELECTIONS.json'):
    if s['split']!='test': continue
    t=tasks[s['task_id']]; selected=[lm[c,s['variant']] for c in s['top2_ids']]; assert len(selected)==2
    required={p['id'] for p in t['necessary_points']}
    covered=set().union(*(set(r['covers']) for r in selected))
    assert covered<=required
    q={'task_id':t['task_id'],'group':groups[t['task_id']],'source':t['source'],'question_type':t['question_type'],
       'coverage':len(covered)/len(required),'full_coverage':int(required<=covered),
       'support_precision':mean([r['y_support'] for r in selected]),'relevance_precision':mean([r['y_relevance'] for r in selected]),
       'partial_rate':mean([r['partial_support'] for r in selected]),'conflict_rate':mean([r['condition_conflict'] for r in selected]),
       'selected_ids':s['top2_ids']}
    key=f"{s['objective']}/{s['seed']}/{s['variant']}/{s['k']}"
    results.setdefault(key,[]).append(q)
assert len(results)==36 and all(len(q)==48 for q in results.values())
# Validate against the existing seed42 report without rewriting it.
old=load('CODEX_FIXED_OUTPUT_MATRIX_RESULTS.json')
for key,qs in results.items():
    objective,seed,v,k=key.split('/')
    if seed!='42': continue
    oo={r['task_id']:r for r in old['question_results'][f'parent_{objective}/{v}/{k}/top_2']}
    for q in qs:
        assert q['selected_ids']==oo[q['task_id']]['selected_ids']
        assert all(abs(q[f]-oo[q['task_id']][f])<1e-12 for f in fields)
for key,qs in sorted(results.items()):
    o,s,v,k=key.split('/')
    for subset in ['all','med','hallu','diagnosis','principle']:
        sub=[q for q in qs if subset=='all' or q['source']==subset or q['question_type']==subset]
        summary.append({'objective':o,'seed':int(s),'variant':v,'k':int(k),'subset':subset,'questions':len(sub),
                        **{f:mean([q[f] for q in sub]) for f in fields}})
aggregate=[]; averaged={}
for o in ['support','relevance']:
 for v in ['fragment','context']:
  for k in [20,50,100]:
   byseed=[{r['task_id']:r for r in results[f'{o}/{s}/{v}/{k}']} for s in [42,2026,3407]]
   averaged[f'{o}/{v}/{k}']=[{**byseed[0][t],**{f:mean([r[t][f] for r in byseed]) for f in fields}} for t in sorted(tasks)]
   for subset in ['all','med','hallu','diagnosis','principle']:
    rows=[r for r in summary if r['objective']==o and r['variant']==v and r['k']==k and r['subset']==subset]
    assert len(rows)==3
    aggregate.append({'objective':o,'variant':v,'k':k,'subset':subset,'questions':rows[0]['questions'],
        **{f+'_mean':mean([r[f] for r in rows]) for f in fields},
        **{f+'_seed_sd':statistics.stdev([r[f] for r in rows]) for f in fields}})
def paired(a,b):
    aa={r['task_id']:r for r in averaged[a]}; bb={r['task_id']:r for r in averaged[b]}
    gs=collections.defaultdict(list)
    for t in sorted(tasks): gs[aa[t]['group']].append(aa[t]['coverage']-bb[t]['coverage'])
    values=list(gs.values()); rng=random.Random(20261012); boot=[]
    for _ in range(5000):
        draw=rng.choices(values,k=len(values)); boot.append(sum(map(sum,draw))/sum(map(len,draw)))
    boot.sort()
    return {'a':a,'b':b,'difference':mean([aa[t]['coverage']-bb[t]['coverage'] for t in tasks]),
        'ci95':[boot[124],boot[4874]],'groups':len(gs),'questions':48,'replicates':5000,
        'bootstrap_unit':'case_group','seed_handling':'average_per_question_before_group_resampling'}
comparisons=[]
for v in ['fragment','context']:
 for k in [20,50,100]: comparisons.append({'comparison':'support_minus_relevance',**paired(f'support/{v}/{k}',f'relevance/{v}/{k}')})
for o in ['support','relevance']:
 for k in [20,50,100]: comparisons.append({'comparison':'context_minus_fragment',**paired(f'{o}/context/{k}',f'{o}/fragment/{k}')})
 for v in ['fragment','context']:
  for k in [50,100]: comparisons.append({'comparison':'larger_k_minus_20',**paired(f'{o}/{v}/{k}',f'{o}/{v}/20')})
hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [B/'CODEX_ALL_TASK_STANDARDS.json',
        B/'DIRECT_ALL_SEED_SELECTIONS.json',B/'CODEX_ALL_SEED_OUTPUT_REVIEW_QUEUE.jsonl']}
data={'seeds':[42,2026,3407],'questions':48,'configurations':36,'unique_test_output_records':len(lm),
      'per_seed_summary':summary,'aggregate':aggregate,'paired_bootstrap':comparisons,'question_results':results,
      'seed42_original_metrics_unchanged':True,'full_pool_ceiling_available':False,'dev_policy_complete':False,
      'clinical_expert_gold':False,'artifact_sha256':hashes}
(B/'CODEX_THREE_SEED_OUTPUT_RESULTS.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
for name,rows in [('CODEX_THREE_SEED_OUTPUT_SUMMARY.csv',aggregate),('CODEX_THREE_SEED_PER_SEED.csv',summary)]:
 with (B/name).open('w',encoding='utf-8-sig',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
md=['# 三种子真实检索固定Top-2核验报告','',
    '本报告完整覆盖同48题、Support/Relevance、fragment/context、k20/50/100、种子42/2026/3407的36个配置。只使用冻结的既有模型预测；新补的选入材料标签完成后统一计算，没有重训或调test阈值。',
    '',f'测试输出并集共{len(lm)}条候选×输入版本记录，全部核验。seed42逐题选择和六项指标与先前报告逐项一致。',
    '', '集合覆盖为选入两段材料完整支持的必要回答点并集/该题必要点总数，再按题宏平均。部分支持单独记录，不计作完整点。正例意味着至少完整支持一个必要点，不等于完整答案充分。相关性P@2独立报告。',
    '', '## 覆盖率：三种子均值±样本标准差','', '|监督目标|输入|k20|k50|k100|','|---|---|---:|---:|---:|']
for o in ['support','relevance']:
 for v in ['fragment','context']:
  row=[next(r for r in aggregate if r['objective']==o and r['variant']==v and r['k']==k and r['subset']=='all') for k in [20,50,100]]
  md.append(f'|{o}|{v}|'+ '|'.join(f"{100*r['coverage_mean']:.2f}% ± {100*r['coverage_seed_sd']:.2f}" for r in row)+'|')
md+=['','BM25固定Top-2作为无监督参照：此前同48题两种输入的覆盖均为5.42%；它不涉及训练种子。','',
     '## 支持监督减相关监督：三种子题内平均、病例组配对bootstrap','',
     '|输入|候选数|覆盖差（百分点）|95%区间（百分点）|病例组数|','|---|---:|---:|---|---:|']
for c in comparisons:
 if c['comparison']=='support_minus_relevance':
  _,v,k=c['a'].split('/'); md.append(f"|{v}|{k}|{100*c['difference']:.2f}|[{100*c['ci95'][0]:.2f}, {100*c['ci95'][1]:.2f}]|{c['groups']}|")
md+=['','5000次按病例组有放回重采样，先对每题的三种子覆盖取平均；没有把种子×题当作144个独立病例。区间条件于这套直接AI标注，不包含标注错误。以下多配置比较为探索性，未作多重检验校正。',
     '', '## 来源和任务分层：k100','', '|监督|输入|分层|题数|覆盖均值|种子SD（百分点）|','|---|---|---|---:|---:|---:|']
for r in aggregate:
 if r['k']==100 and r['subset']!='all':md.append(f"|{r['objective']}|{r['variant']}|{r['subset']}|{r['questions']}|{r['coverage_mean']:.2%}|{100*r['coverage_seed_sd']:.2f}|")
md+=['','## 相关性目标自己的选入指标','', '|监督|输入|k20相关性P@2|k50相关性P@2|k100相关性P@2|','|---|---|---:|---:|---:|']
for o in ['support','relevance']:
 for v in ['fragment','context']:
  rr=[next(r for r in aggregate if r['objective']==o and r['variant']==v and r['k']==k and r['subset']=='all') for k in [20,50,100]]
  md.append(f'|{o}|{v}|'+ '|'.join(f"{r['relevance_precision_mean']:.2%}" for r in rr)+'|')
md+=['','## 解释与尚未完成项','',
     '三种子均值在六个检索设置中均为Support高于Relevance，覆盖差为8.52–14.65个百分点。fragment/k20的95%区间包含0，其余五个设置的区间下界大于0；这些属于探索性比较，不能忽略多配置选择及标注条件。',
     '', 'Support上下文相对片段的平均增益在k20/50/100分别为2.08、1.04、0.23个百分点。对应区间触及或跨过0，尚不能声称上下文稳定有效。Support从20增加到50/100候选的覆盖增益区间也均跨过0，不能把单个种子的上升写成稳定检索收益。k50/context当前均值最高只是描述，不据此在test选择最终配置。',
     '', '论文主线可以集中为“同一实际检索候选池、相同选择预算下，任务定向支持监督比主题相关监督更有助于覆盖必要回答点”。候选数和上下文作为有边界的敏感性实验；目前支持率仍有限，尤其治疗原则题，结论应限定于辅助证据选择而非完整问答或临床决策。',
     '',
     '本轮可以检验真实检索下固定Top-2选入材料的支持覆盖差，以及候选数量/上下文对最终选择的影响。不能据选中材料推断整个候选池覆盖上限；全池标注未完成，AP/MRR/nDCG也暂不报告。dev回答点已固定，但候选标签和基于集合覆盖/负例预算的阈值与Top-m选择尚待完成。',
     '', '当前是直接AI文本核验，并非中医临床专家金标。旧测试样例及先前总结果已有接触，属于既有测试集标注修订后的重评；不能称全新双盲外部临床测试。必要点标准、原输入、模型和评分版本均保留。论文可写真实检索固定预算下的探索性支持监督优势，不可写完整临床有效性或全面优越性。']
(B/'CODEX_THREE_SEED_OUTPUT_REPORT.md').write_text('\n'.join(md)+'\n',encoding='utf-8')
try:
 import matplotlib; matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(1,2,figsize=(10,4.4),sharey=True)
 for ax,v in zip(axes,['fragment','context']):
  for o,name in [('support','Support'),('relevance','Relevance')]:
   rr=[next(r for r in aggregate if r['objective']==o and r['variant']==v and r['k']==k and r['subset']=='all') for k in [20,50,100]]
   ax.errorbar([20,50,100],[100*r['coverage_mean'] for r in rr],yerr=[100*r['coverage_seed_sd'] for r in rr],marker='o',capsize=4,label=name)
  ax.axhline(100*next(r for r in old['summary'] if r['method']=='bm25' and r['variant']==v and r['k']==20 and r['subset']=='all')['coverage'],ls='--',color='gray',label='BM25')
  ax.set_title(v);ax.set_xticks([20,50,100]);ax.set_xlabel('Real retrieval candidates');ax.grid(alpha=.25);ax.legend()
 axes[0].set_ylabel('Required-point coverage (%)')
 fig.suptitle('48 questions | fixed Top-2 | 3 seeds mean +/- SD | AI text review')
 fig.tight_layout();fig.savefig(B/'CODEX_THREE_SEED_OUTPUT.png',dpi=180);plt.close(fig)
except ImportError as e: print('Plot unavailable:',e)
print('\n'.join(md[:19]))
