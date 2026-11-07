import json,csv,collections
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('/home/<user>/UUM/EEE_research');OUT=ROOT/'experiments/stage4_retraining'
def load(p):return json.loads(p.read_text(encoding='utf-8'))
def save(n,v):(OUT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def qs(p):return [json.loads(s) for s in p.read_text(encoding='utf-8').splitlines() if s.strip()]
def bootstrap(a,b,key):
 bm={q['task_id']:q for q in b};groups=collections.defaultdict(list)
 for q in a:
  if q['task_id'] in bm:groups[q['group']].append(q[key]-bm[q['task_id']][key])
 if not groups:return None
 arr=list(groups.values());tot=np.array([sum(x) for x in arr]);n=np.array([len(x) for x in arr]);rng=np.random.default_rng(20261010);sample=[]
 for _ in range(5000):
  ix=rng.integers(len(arr),size=len(arr));sample.append(tot[ix].sum()/n[ix].sum())
 return {'difference':float(tot.sum()/n.sum()),'ci95':np.quantile(sample,[.025,.975]).tolist(),'groups':len(arr),'questions':int(n.sum()),'replicates':5000,'seed':20261010}
def main():
 records=[];boot={}
 runs=[(obj,seed,OUT/f'{obj}_{seed}') for obj in ['support','relevance'] for seed in [42,2026,3407]]
 runs += [(name,42,OUT/'baselines'/name) for name in ['bm25','tfidf','random','majority','length_overlap']]
 for obj,seed,d in runs:
   if obj in ['support','relevance'] and not (d/'DONE.json').exists():continue
   if not (d/'set_metrics.json').exists():continue
   own=load(d/'own_target_metrics.json');metrics=load(d/'set_metrics.json')
   for policy,subs in metrics.items():
    for source,m in subs.items():records.append({'objective':obj,'seed':seed,'policy':policy,'source':source,**m,'own_target_ap':own['test']['auprc'],'own_target_f1':own['test']['f1']})
 for seed in [42,2026,3407]:
  for policy in ['budget_0.1','top_2']:
   aa=OUT/f'support_{seed}/{policy}_questions.jsonl';bb=OUT/f'relevance_{seed}/{policy}_questions.jsonl'
   if aa.exists() and bb.exists() and (OUT/f'support_{seed}/DONE.json').exists() and (OUT/f'relevance_{seed}/DONE.json').exists():
    a,b=qs(aa),qs(bb)
    for source in ['all','med','hallu']:
     boot[f'{seed}/{policy}/{source}']={k:bootstrap([q for q in a if source=='all' or q['source']==source],[q for q in b if source=='all' or q['source']==source],k) for k in ['coverage','full_coverage']}
 save('paired_group_bootstrap.json',boot)
 if records:
  with (OUT/'results.csv').open('w',newline='',encoding='utf-8') as f:w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
  if any(x['objective'] in ['support','relevance'] for x in records):
   fig,ax=plt.subplots(figsize=(8,4))
   for j,obj in enumerate(['support','relevance']):
    r=[x for x in records if x['objective']==obj and x['policy']=='top_2' and x['source']=='all']
    if r:ax.bar(j,np.mean([x['coverage'] for x in r]),yerr=np.std([x['coverage'] for x in r],ddof=1) if len(r)>1 else 0,label=obj)
   ax.set_xticks([0,1],['Support-CE','Relevance-CE']);ax.set_ylabel('Necessary point coverage');ax.set_ylim(0,1);fig.tight_layout();fig.savefig(OUT/'coverage_by_objective.png',dpi=180);plt.close(fig)
 fig,axes=plt.subplots(1,2,figsize=(10,4))
 for ax,obj in zip(axes,['support','relevance']):
  for seed in [42,2026,3407]:
   p=OUT/f'{obj}_{seed}/step_log.jsonl'
   if p.exists():
    rr=[json.loads(s) for s in p.read_text(encoding='utf-8').splitlines(keepends=True) if s.endswith('\n')];ax.plot([r['step'] for r in rr],[r['loss'] for r in rr],label=str(seed))
  ax.set_title(obj);ax.set_xlabel('Optimizer step');ax.set_ylabel('CE loss')
  if ax.lines:ax.legend()
 fig.tight_layout();fig.savefig(OUT/'training_loss.png',dpi=180);plt.close(fig)
 done=sum((OUT/f'{o}_{s}/DONE.json').exists() for o in ['support','relevance'] for s in [42,2026,3407])
 text=f'# Stage 4 实验状态\n\n正式训练完成 {done}/6。未完成运行不进入成绩表。\n\n'+('当前 results.csv 可包含已完成的传统基线；正式模型成绩只在 DONE 后加入。训练损失图随日志刷新。\n\n'+('已完成的成对模型见 paired_group_bootstrap.json；三种子标准差与问题抽样 CI 分别报告。\n\n' if boot else '当前尚无成对模型结果，bootstrap 与模型覆盖对比图仍待生成。\n\n') if records else '当前没有正式训练成绩。\n\n')+'数据为可追溯的机器辅助修订标签，尚非独立临床专家金标；Hallu 为回顾性评价。真实检索池待裁决标签不得作为确定负例计算医学准确率。\n'
 (OUT/'RESULTS.md').write_text(text,encoding='utf-8')
if __name__=='__main__':main()
