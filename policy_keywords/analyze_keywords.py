"""Document-level TF-IDF, group comparisons, filtering sensitivity. No causal inference."""
import argparse, json, hashlib
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import sklearn

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--tokens',type=Path,required=True); ap.add_argument('--metadata',type=Path,required=True); ap.add_argument('--out',type=Path,required=True); a=ap.parse_args()
    out=a.out; out.mkdir(parents=True,exist_ok=True)
    meta=pd.read_csv(a.metadata,sep='\t' if a.metadata.suffix=='.tsv' else ',',encoding='utf-8-sig').sort_values('文件名').reset_index(drop=True)
    assert len(meta)==59 and meta['文件名'].is_unique
    info=json.loads((a.tokens/'summary.json').read_text(encoding='utf-8'))
    assert info['documents']==59 and info['source_hashes_verified']==59
    def save(df,name):df.to_csv(out/name,index=False,encoding='utf-8-sig',float_format='%.8f')
    groups=[('全样本','全部',meta.index.to_list(),False)]
    for l,g in meta.groupby('层级'):groups.append(('层级',l,g.index.to_list(),False))
    central=meta[meta['层级']=='中央'];prov=meta[meta['层级']=='省级']
    for y,g in central.groupby('年份'):groups.append(('中央年度',str(y),g.index.to_list(),False))
    for r,g in prov.groupby('区域'):groups.append(('区域省份等权',r,g.index.to_list(),True))
    for r,g in prov[prov['政策类型']!='数字产业与流通'].groupby('区域'):groups.append(('区域去除产业流通',r,g.index.to_list(),True))
    for r,g in prov.groupby('省份/范围'):groups.append(('省份',r,g.index.to_list(),False))
    for t,g in meta.groupby('政策类型'):groups.append(('政策类型',t,g.index.to_list(),False))
    groups += [('连续年度工作要点',str(meta.loc[i,'年份']),[i],False) for i in central.index if meta.loc[i,'文件名'].startswith(('C006_','C007_','C008_','C009_'))]
    summary={'engine':info['engine'],'engine_version':info['engine_version'],'sklearn':sklearn.__version__,'documents':len(meta),'min_df':2,'sublinear_tf':True,'smooth_idf':True,'norm':'l2','metadata_sha256':hashlib.sha256(a.metadata.read_bytes()).hexdigest(),'modes':{}}
    allrank={};token_stats=[];group_rows=[];doc_rows=[];freq_rows=[]
    for mode in ['base','exploratory']:
        folder=a.tokens/f'tokens_{mode}';assert set(p.name for p in folder.glob('*.txt'))==set(meta['文件名'])
        docs=[(folder/n).read_text(encoding='utf-8').split() for n in meta['文件名']];assert all(docs)
        assert sum(map(len,docs))==info[f'{mode}_tokens']
        for i,d in enumerate(docs):token_stats.append([mode,meta.loc[i,'文件名'],len(d),len(set(d)),hashlib.sha256((folder/meta.loc[i,'文件名']).read_bytes()).hexdigest()])
        c=Counter(w for d in docs for w in d);df=Counter(w for d in docs for w in set(d))
        freq_rows.extend([mode,w,n,df[w],n/sum(c.values())*10000] for w,n in sorted(c.items(),key=lambda x:(-x[1],x[0])))
        v=TfidfVectorizer(analyzer=lambda s:s, min_df=2, sublinear_tf=True,smooth_idf=True,norm='l2');X=v.fit_transform(docs).toarray();terms=v.get_feature_names_out();assert np.allclose(np.linalg.norm(X,axis=1),1)
        def top(scores,k):return [i for i in sorted(range(len(terms)),key=lambda i:(-scores[i],terms[i]))[:k] if scores[i]>0]
        for i in range(len(meta)):
            for rank,j in enumerate(top(X[i],20),1):doc_rows.append([mode,meta.loc[i,'文件名'],rank,terms[j],X[i,j]])
        allrank[mode]={}
        for typ,g,idx,equal_prov in groups:
            sub=meta.loc[idx];nprov=sub.loc[sub['层级']=='省级','省份/范围'].nunique()
            if equal_prov:
                scores=np.mean([X[ids].mean(axis=0) for ids in sub.groupby('省份/范围').groups.values()],axis=0)
            else:scores=X[idx].mean(axis=0)
            indices=top(scores,50);allrank[mode][typ,g]=[terms[j] for j in indices]
            for rank,j in enumerate(indices[:30],1):group_rows.append([mode,typ,g,len(idx),nprov,'省份等权' if equal_prov else '文档等权',rank,terms[j],scores[j],int((X[idx,j]>0).sum())])
        summary['modes'][mode]={'tokens':sum(c.values()),'vocab_all':len(c),'vocab_min_df2':len(terms),'shape':list(X.shape),'top10_mean_tfidf':[[str(terms[j]),float(X.mean(axis=0)[j])] for j in top(X.mean(axis=0),10)],'top10_frequency':c.most_common(10)}
    sens=[]
    for key,b in allrank['base'].items():
        e=allrank['exploratory'][key];s1=set(b[:20]);s2=set(e[:20]);s3=set(b[:50]);s4=set(e[:50]);sens.append([*key,len(s1&s2),len(s1&s2)/len(s1|s2) if s1|s2 else np.nan,len(s3&s4),len(s3&s4)/len(s3|s4) if s3|s4 else np.nan])
    save(pd.DataFrame(token_stats,columns=['过滤方案','文件名','词数','不同词数','分词文件SHA256']),'token_stats.csv');save(pd.DataFrame(freq_rows,columns=['过滤方案','词语','词频','文档频数','每万词频']),'frequency.csv');save(pd.DataFrame(doc_rows,columns=['过滤方案','文件名','排名','词语','TFIDF']),'document_keywords.csv');save(pd.DataFrame(group_rows,columns=['过滤方案','分析维度','分组','文档数','省份数','汇总口径','排名','词语','平均TFIDF','组内含词文档数']),'group_keywords.csv');save(pd.DataFrame(sens,columns=['分析维度','分组','TOP20重合数','TOP20_Jaccard','TOP50重合数','TOP50_Jaccard']),'filter_sensitivity.csv')
    save(meta.groupby(['年份','层级']).size().rename('政策数').reset_index(),'year_distribution.csv');save(prov.groupby(['区域','政策类型']).agg(政策数=('文件名','size'),省份数=('省份/范围','nunique')).reset_index(),'region_type_distribution.csv')
    summary['global_top20_overlap']=sens[0][2];summary['global_top50_overlap']=sens[0][4];(out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
