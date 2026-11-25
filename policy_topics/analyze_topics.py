from pathlib import Path
import argparse,json,hashlib,itertools
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import jensenshannon

def main():
 p=argparse.ArgumentParser();p.add_argument('--tokens',type=Path,required=True);p.add_argument('--metadata',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--engine',required=True,choices=['icu','jieba']);p.add_argument('--dictionary',type=Path,default=Path(__file__).with_name('theme_dictionary.tsv'));a=p.parse_args();o=a.output;o.mkdir(parents=True,exist_ok=True)
 m=pd.read_csv(a.metadata,sep='\t');names=m['文件名'].tolist();assert len(names)==len(set(names))==59
 ds=pd.read_csv(a.dictionary,sep='\t');lex={r['主题']:set(r['词语'].split()) for _,r in ds.iterrows()};assert sum(map(len,lex.values()))==len(set.union(*lex.values()))
 docs={f:[(a.tokens/f'tokens_{mode}'/f).read_text(encoding='utf-8').split() for mode in ['base','exploratory']] for f in names};assert all(all(z for z in d) for d in docs.values());assert all(set(x.name for x in (a.tokens/f'tokens_{mode}').glob('*.txt'))==set(names) for mode in ['base','exploratory'])
 def save(x,n):x.to_csv(o/n,index=False,encoding='utf-8-sig',float_format='%.8f')
 tokenhash=[];rows=[];hits=[];samples=[];audit=[];matched={}
 for modeid,mode in enumerate(['base','exploratory']):
  tot=Counter(w for f in names for w in docs[f][modeid]);matched[mode]=sum(tot[w]>0 for w in set.union(*lex.values()))
  for t,words in lex.items():
   for w in sorted(words):audit.append([mode,t,w,tot[w],sum(w in docs[f][modeid] for f in names)])
  for i,f in enumerate(names):
   z=docs[f][modeid];c=Counter(z);counts={t:sum(c[w] for w in words) for t,words in lex.items()};n=sum(counts.values());tokenhash.append([mode,f,hashlib.sha256((a.tokens/f'tokens_{mode}'/f).read_bytes()).hexdigest(),len(z)])
   for t,words in lex.items():
    rows.append([mode,*m.iloc[i].tolist(),t,counts[t],len(z),n,10000*counts[t]/len(z),counts[t]/n if n else np.nan]);hits.extend([mode,f,t,w,c[w]] for w in sorted(words) if c[w])
    if mode=='base':
     loc=[j for j,w in enumerate(z) if w in words]
     if loc:
      j=loc[len(loc)//2];samples.append([f,t,z[j],j,' '.join(z[max(0,j-12):j+13])])
 r=pd.DataFrame(rows,columns=['过滤方案',*m.columns,'主题','命中词次','保留词数','总主题命中','每万词命中','主题内占比']);save(r,'document_theme_scores.csv');save(pd.DataFrame(hits,columns=['过滤方案','文件名','主题','词语','次数']),'matched_terms.csv');save(pd.DataFrame(audit,columns=['过滤方案','主题','词语','词频','文档频数']),'dictionary_audit.csv');save(pd.DataFrame(samples,columns=['文件名','主题','命中词','词序号','分词上下文']),'review_samples.csv');save(pd.DataFrame(tokenhash,columns=['过滤方案','文件名','SHA256','词数']),'token_input_hashes.csv')
 base=r[r['过滤方案']=='base'];agg=base.groupby('主题',sort=False).agg(命中词次=('命中词次','sum'),平均每万词命中=('每万词命中','mean'),平均主题内占比=('主题内占比','mean'),含主题文档数=('命中词次',lambda x:(x>0).sum())).reset_index();save(agg,'theme_overview.csv')
 s=base[base['层级']=='中央'];central=s.groupby(['年份','主题']).agg(平均每万词命中=('每万词命中','mean'),政策数=('文件名','nunique')).reset_index();save(central,'central_year_pilot.csv')
 s=base[base['层级']=='省级'];province=s.groupby(['区域','省份/范围','主题']).agg(平均每万词命中=('每万词命中','mean'),政策数=('文件名','nunique')).reset_index();save(province,'province_pilot.csv');region=province.groupby(['区域','主题']).agg(平均每万词命中=('平均每万词命中','mean'),省份数=('省份/范围','nunique'),政策数=('政策数','sum')).reset_index();save(region,'region_pilot.csv')
 texts=[' '.join(docs[f][1]) for f in names];v=CountVectorizer(tokenizer=str.split,token_pattern=None,lowercase=False,min_df=3,max_df=.90);X=v.fit_transform(texts);features=v.get_feature_names_out();B=(X>0).astype('int32');co=(B.T@B).toarray();freq=np.asarray(B.sum(0)).ravel();N=len(names)
 def coherence(idx):
  vals=[]
  for i,j in itertools.combinations(idx,2):
   n=co[i,j];p=n/N;vals.append(-1.0 if n==0 else float(np.log(p/((freq[i]/N)*(freq[j]/N)))/-np.log(p)) if p<1 else 0.)
  return np.mean(vals)
 models={};metrics=[];topicrows=[];allweights=[]
 for k in range(3,9):
  comps=[];vals=[];perplex=[]
  for seed in [11,42,73]:
   model=LatentDirichletAllocation(n_components=k,learning_method='batch',max_iter=25,random_state=seed,n_jobs=1);W=model.fit_transform(X);P=model.components_/model.components_.sum(axis=1,keepdims=True);comps.append(P);tops=np.argsort(P,axis=1)[:,-10:][:,::-1];cv=float(np.mean([coherence(t) for t in tops]));vals.append(cv);perplex.append(model.perplexity(X));models[k,seed]=(P,W,tops)
   for ti,t in enumerate(tops):topicrows.append([k,seed,ti+1,' '.join(features[t]),coherence(t)])
   for i,f in enumerate(names):
    for j,w in enumerate(W[i]):allweights.append([k,seed,f,j+1,w])
  st=[]
  for A,Bb in itertools.combinations(comps,2):
   dist=np.array([[jensenshannon(x,y,base=2) for y in Bb] for x in A]);ii,jj=linear_sum_assignment(dist);st.append(1-dist[ii,jj].mean())
  metrics.append([k,np.mean(vals),np.std(vals),np.mean(st),np.mean(perplex)]);print('K',k,'NPMI',np.mean(vals),'stability',np.mean(st),flush=True)
 met=pd.DataFrame(metrics,columns=['K','平均NPMI','NPMI标准差','平均匹配稳定性','训练集困惑度']);save(met,'lda_model_comparison.csv');save(pd.DataFrame(topicrows,columns=['K','seed','主题编号','TOP10词语','主题NPMI']),'lda_topics_all.csv');save(pd.DataFrame(allweights,columns=['K','seed','文件名','主题编号','权重']),'lda_document_weights_all.csv')
 k=int(met.sort_values(['平均NPMI','平均匹配稳定性'],ascending=False).iloc[0]['K']);seed=42;P,W,tops=models[k,seed];save(pd.DataFrame([[j+1,' '.join(features[t]),coherence(t),W[:,j].mean()] for j,t in enumerate(tops)],columns=['主题编号','TOP10词语','主题NPMI','平均权重']),'lda_selected_topics.csv');assert np.allclose(W.sum(1),1);assert (r['总主题命中']<=r['保留词数']).all()
 summary={'engine':a.engine,'documents':len(names),'dictionary_terms':sum(map(len,lex.values())),'dictionary_matched_terms':matched,'base_tokens':sum(len(docs[f][0]) for f in names),'exploratory_tokens':sum(len(docs[f][1]) for f in names),'base_hits':int(base['命中词次'].sum()),'base_docs_with_hits':int(base.groupby('文件名')['总主题命中'].first().gt(0).sum()),'lda_shape':list(X.shape),'lda_models':len(models),'selected_k':k,'selected_seed':seed,'theme_overview':agg.to_dict('records'),'model_comparison':met.to_dict('records')};(o/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()
