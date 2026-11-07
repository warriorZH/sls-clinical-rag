"""Create versioned runners with exact source substitutions and explicit checks."""
from pathlib import Path
ROOT=Path('/home/<user>/UUM/EEE_research');OUT=ROOT/'experiments/stage4_retraining'
def replace(s,a,b):
 assert a in s,a;s=s.replace(a,b);return s
def main():
 prep=(ROOT/'prepare_v2_experiment.py').read_text()
 prep=replace(prep,"OUT=BASE/'experiments/stage2_optimized_v2'","OUT=BASE/'experiments/stage4_retraining'")
 prep=replace(prep,"DATA=BASE/'dataset/stage2_optimized_v2/dataset.json'","DATA=OUT/'dataset.json'")
 prep=replace(prep,' groups=collections.defaultdict(list)'," id_index={t['id']:i for i,t in enumerate(tasks)}\n for a,b in json.loads((OUT/'confirmed_group_links.json').read_text()):\n  if a in id_index and b in id_index:union(id_index[a],id_index[b]);reasons['reviewed_same_case']+=1\n groups=collections.defaultdict(list)")
 prep=replace(prep,"   origin=e.get('origin_task_id')","   origins=set(re.findall(r'(?:med|hallu)_\\d{5}',str(e.get('source',''))))|{e.get('origin_task_id')}\n   if any(o in assignment and assignment[o]!=s for o in origins): removed[s]+=1;continue\n   origin=e.get('origin_task_id')")
 (OUT/'prepare_stage4.py').write_text(prep)
 train=(ROOT/'train_v2.py').read_text()
 train=replace(train,"BASE=Path('/home/<user>/UUM/EEE_research/experiments/stage2_optimized_v2')","BASE=Path('/home/<user>/UUM/EEE_research/experiments/stage4_retraining')\nfrom policy_stage4 import checkpoint_key,export_run")
 train=replace(train,"'checkpoint_selection':'dev_support_AUPRC'","'checkpoint_selection':'dev_common_set_budget_then_predeclared_top2_fallback'")
 train=replace(train,'step=0;best=-1;','step=0;best=(-1,);')
 train=replace(train,"  if metrics['auprc']>best:\n   best=metrics['auprc'];patience=0","  key,chosen,grid=checkpoint_key(data['dev'],scores,data['meta'])\n  own=classify([{**p,'y_support':p[label]} for p in data['dev']],scores,.5)\n  emit(out/'checkpoint_selection.jsonl',{'epoch':epoch+1,'key':key,'selected_policy':chosen,'own_target':own})\n  if key>best:\n   best=key;patience=0")
 train=replace(train," evaluate(data['dev'],dev_scores,data['meta'],threshold,out/'dev')"," evaluate(data['dev'],dev_scores,data['meta'],threshold,out/'dev')")
 train=replace(train," write_json(out/'DONE.json',{'best_epoch'"," export_run(data,dev_scores,scores,out,a.objective)\n write_json(out/'DONE.json',{'best_epoch'")
 train=replace(train,"'best_dev_support_auprc':best","'best_common_dev_key':best")
 train=replace(train," export_run(data,dev_scores,scores,out,a.objective)"," export_run(data,dev_scores,scores,out,a.objective)\n from actual_predictions_stage4 import export as export_actual\n export_actual(model,predict,a.micro,out)")
 (OUT/'train_stage4.py').write_text(train)
 baseline=(ROOT/'baselines_v2.py').read_text();baseline=replace(baseline,"experiments/stage2_optimized_v2","experiments/stage4_retraining")
 # Baselines get the same dev set-policy export as learned models.
 baseline=replace(baseline,'from evaluate_v2 import evaluate,','from policy_stage4 import export_run\nfrom evaluate_v2 import evaluate,')
 baseline=replace(baseline,"  write_json(dest/'config.json'","  np.save(dest/'dev_scores.npy',dev);export_run(data,dev,test,dest,'support')\n  write_json(dest/'config.json'")
 (OUT/'baselines_stage4.py').write_text(baseline)
 print('MATERIALIZED',flush=True)
if __name__=='__main__':main()
