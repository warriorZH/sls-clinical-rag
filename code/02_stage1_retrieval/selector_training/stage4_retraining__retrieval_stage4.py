"""Fresh query-only candidates; absent literal anchors remain UNRESOLVED."""
import json,re,hashlib,random,collections,sys
from pathlib import Path
ROOT=Path('/home/<user>/UUM/EEE_research');sys.path.insert(0,str(ROOT))
OUT=ROOT/'experiments/stage4_retraining/actual_retrieval';OUT.mkdir(exist_ok=True)
import retrieval_stage3 as old
from optimize_stage2_dataset import Builder
from build_verification_atomic_draft import PHRASES
def save(n,v):(OUT/n).write_text(json.dumps(v,ensure_ascii=False,indent=2))
def atomic(analysis,therapy):
 found=[]
 for word in sorted(PHRASES,key=len,reverse=True):
  m=re.search(re.escape(word),analysis)
  if m and not any(a<=m.start() and m.end()<=b for a,b,_ in found) and not re.search(r'不是|并非|无',analysis[max(0,m.start()-3):m.start()]):found.append((m.start(),m.end(),word))
 for m in re.finditer(r'中医诊断(?:为|是)([^，,。；;\n]{2,15})',analysis):
  if not re.search(r'患者|医生|经过|治疗|西医',m.group(1)):found.append((m.start(1),m.end(1),m.group(1)));break
 pts=[]
 for a,b,word in sorted(found):
  if word in [p['text'] for p in pts]:continue
  pts.append({'id':f'a{len(pts)+1}','text':word,'required':True,'source_field':'reference_analysis','source_start_char':a,'source_end_char':b,'source_assertion':analysis[max(0,analysis.rfind('。',0,a)+1):analysis.find('。',b)+1 if analysis.find('。',b)>=0 else len(analysis)]})
 return pts
def make_tasks():
 data=ROOT/'dataset';previous=json.loads((data/'stage2_optimized_v2/dataset.json').read_text());current=json.loads((ROOT/'experiments/stage4_retraining/dataset.json').read_text());manifest=json.loads((ROOT/'experiments/stage4_retraining/split_manifest.json').read_text());assignment={tid:s for s,ids in manifest['task_ids'].items() for tid in ids};records=json.loads((data/'stage1_sections_v1_1/medical_records_stage1.json').read_text());used={t['source']['index'] for t in previous if t['id'].startswith('med_')}
 for t in previous:
  for e in t['evidence']:
   used.update(int(x) for x in re.findall(r'med_(\d{5})',str(e.get('source',''))))
 prior=json.loads((ROOT/'experiments/stage3_validation/retrieval_verification_tasks.json').read_text());used.update(t['stage1_index'] for t in prior if 'stage1_index' in t)
 patients={old.patient(t.get('original_question',t['question'])) for t in previous};patients.discard(None);builder=Builder(ROOT);eligible=[];excluded=collections.Counter()
 for i,r in enumerate(records):
  if i in used:excluded['previous_case_or_material_or_pilot']+=1;continue
  raw=r['病情描述'];analysis=r['病情病因诊断分析'];therapy=r['治则及治疗方案'];q=builder.clinical_query(raw);pts=atomic(analysis,therapy)
  if old.patient(raw) in patients:excluded['same_identifiable_patient']+=1;continue
  if not q or len(pts)<2:excluded['no_query_or_less_than_two_source_points']+=1;continue
  eligible.append({'id':f'fresh_med_{i:05d}','source':'med','stage1_index':i,'question':q,'original_question':raw,'answer_points':pts,'reference_analysis':analysis,'reference_treatment':therapy,'previously_evaluated':False,'evaluation_split':'test_unseen_med','annotation_status':'source_literal_points_pending_conditional_semantic_review'})
 # Conservative pre-prediction screen: all >=.85 character neighbors are reserved out.
 from sklearn.feature_extraction.text import TfidfVectorizer
 from sklearn.neighbors import NearestNeighbors
 if eligible:
  oldq=[t['question'].split('\n')[0] for t in previous];newq=[t['question'].split('\n')[0] for t in eligible]
  tf=TfidfVectorizer(analyzer='char',ngram_range=(2,3),min_df=2);mat=tf.fit_transform(oldq+newq);nn=NearestNeighbors(n_neighbors=1,metric='cosine',algorithm='brute',n_jobs=4).fit(mat[:len(oldq)]);d,ix=nn.kneighbors(mat[len(oldq):]);audit=[];kept=[]
  for t,dist,ind in zip(eligible,d[:,0],ix[:,0]):
   sim=float(1-dist);audit.append({'task_id':t['id'],'nearest_old_id':previous[int(ind)]['id'],'cosine':sim,'excluded_conservatively':sim>=.85})
   if sim<.85:kept.append(t)
  eligible=kept;save('fresh_neighbor_screen.json',audit)
 rng=random.Random(20261010);rng.shuffle(eligible);fresh=eligible[:100];selected=[]
 for fold in ['train','dev','test']:
  for source in ['med','hallu']:
   ts=[t for t in current if assignment[t['id']]==fold and t['id'].startswith(source+'_')];rng.shuffle(ts)
   limit=100 if fold=='train' else 50
   if fold=='test' and source=='med':continue
   for t in ts[:limit]:selected.append({**t,'source':source,'previously_evaluated':True,'evaluation_split':fold+'_retrospective'})
 tasks=selected+fresh;save('retrieval_verification_tasks.json',tasks);save('fresh_sampling.json',{'eligible_after_all_filters':len(eligible),'fresh_selected':len(fresh),'previous_indices_or_material_sources_excluded':len(used),'exclusions':dict(excluded),'seed':20261010,'new_hallu_available':False,'cases_by_fold':dict(collections.Counter(t['evaluation_split'] for t in tasks)),'independent_clinical_gold':False})
 # Case bank strictly follows NEW split, replacing old stage3 training bank.
 for t in current:t['original_split']=assignment[t['id']]
 return tasks,records,current
def main():
 old.OUT=OUT;old.make_tasks=make_tasks
 source=Path(old.__file__).read_text();begin=source.index('def main():');end=source.index("if __name__=='__main__':")
 code=source[begin:end]
 code=code.replace("'y_support':int(bool(covers))","'y_support':None,'anchor_match_support':int(bool(covers))")
 code=code.replace("'covers':covers","'covers':[],'anchor_covers':covers")
 code=code.replace("sum(p['y_support'] for p in pool)","sum(p['anchor_match_support'] for p in pool)")
 # Preserve unresolved status, rather than turning non-matches into negative labels.
 code=code.replace("'annotation_status':'strict_reference_anchor_proxy_not_independent_clinical_gold'","'annotation_status':'UNRESOLVED_pending_semantic_and_condition_adjudication'")
 code=code.replace("'support_extent':'none_established' if not covers", "'support_extent':'unresolved' if not covers")
 namespace=dict(old.__dict__);exec(code,namespace);namespace['main']()
 save('ADJUDICATION_REQUIRED.json',{'status':'pending','rule':'anchor match is a review hint, not a semantic label; absence is not a negative','all_actual_support_and_relevance_labels_null':True,'train_pool_ready_for_training':False,'semantic_and_condition_review_required':True})
if __name__=='__main__':main()
