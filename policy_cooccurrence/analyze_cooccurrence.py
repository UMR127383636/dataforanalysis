"""Policy keyword co-occurrence. ICU pilot or explicit Jieba; no source modification."""
import argparse, csv, hashlib, importlib.util, itertools, json, re, unicodedata
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--preprocess',type=Path,required=True);p.add_argument('--qc',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--engine',choices=['icu','jieba'],default='jieba');p.add_argument('--previous-tokens',type=Path);p.add_argument('--font',type=Path);a=p.parse_args()
    out=a.output;out.mkdir(parents=True,exist_ok=True)
    spec=importlib.util.spec_from_file_location('preprocess',a.preprocess);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    cfg=a.preprocess.parent/'config';terms=sorted(set(m.lines(cfg/'domain_terms.txt')),key=lambda x:(-len(x),x));termset=set(terms);base=set(m.lines(cfg/'stopwords_base.txt'));extra=set(m.lines(cfg/'stopwords_policy_optional.txt'));syn=dict(x.split('\t') for x in m.lines(cfg/'synonyms.tsv'))
    if a.engine=='icu':cut,version=m.icu_splitter()
    else:
        import jieba
        j=jieba.Tokenizer();j.initialize()
        for t in terms:j.add_word(t,freq=1000000)
        cut=lambda s:j.lcut(s,HMM=False);version=jieba.__version__
    pat=re.compile('('+'|'.join(map(re.escape,terms))+')');lexical=re.compile(r'[\u3400-\u9fff]|[A-Za-z]');han=re.compile(r'[\u3400-\u9fff]')
    repair=[(re.compile(r'[ \t]*(?:\n[ \t]*)?'.join(map(re.escape,t))),t) for t in terms if len(t)>1]
    def normalize(s):
        s=unicodedata.normalize('NFKC',s).replace('\ufeff','').replace('\r\n','\n').replace('\r','\n');s=re.sub(r'https?://[^\s，。；、）)]+|www\.[^\s，。；、）)]+',' ',s)
        for rx,t in repair:s=rx.sub(t,s)
        return s
    def tokenize(s):
        raw=[]
        for part in pat.split(s):
            if part:raw.extend([part] if part in termset else cut(part))
        raw=[w.strip() for w in raw if w.strip() and lexical.search(w) and (han.search(w) or len(w.strip())>=2)]
        b=[syn.get(w,w) for w in raw if w not in base and (len(w)>1 or w in termset)]
        return b,[w for w in b if w not in extra]
    with a.qc.open(encoding='utf-8-sig') as f:expected={r['name']:r['new_sha256'] for r in csv.DictReader(f)}
    files=sorted(a.input.glob('*.txt'));assert len(files)==59 and set(expected)=={f.name for f in files}
    docs=[];sources=[];verified=0;tokens_verified=0
    for f in files:
        data=f.read_bytes();assert hashlib.sha256(data).hexdigest()==expected[f.name];verified+=1;s=normalize(data.decode('utf-8-sig'));sources.append(s);b,e=tokenize(s);docs.append(e)
        if a.previous_tokens:
            for mode,ts in [('base',b),('exploratory',e)]:assert (a.previous_tokens/f'tokens_{mode}'/f.name).read_text(encoding='utf-8').split()==ts,(f.name,mode)
            tokens_verified+=1
    v=TfidfVectorizer(analyzer=lambda x:x,min_df=2,sublinear_tf=True,smooth_idf=True,norm='l2');X=v.fit_transform(docs);words=v.get_feature_names_out();weights=np.asarray(X.mean(axis=0)).ravel();idx=sorted(range(len(words)),key=lambda i:(-weights[i],words[i]))[:60];selected=[str(words[i]) for i in idx];vocab=set(selected);rank={w:i+1 for i,w in enumerate(selected)};wt={str(words[i]):float(weights[i]) for i in idx}
    def save(name,head,rows):
        with (out/name).open('w',encoding='utf-8-sig',newline='') as f:w=csv.writer(f);w.writerow(head);w.writerows(rows)
    def calc(sep,mode):
        cnt=Counter();pairs=Counter();pairdocs=Counter();units=0;detail=[];samples=[]
        for f,s in zip(files,sources):
            dset=set();du=0;seen=0
            for part in re.split(sep,s):
                b,e=tokenize(part);ts=e if mode=='exploratory' else b
                if not ts:continue
                units+=1;du+=1;ws=sorted(set(ts)&vocab);cnt.update(ws)
                if len(ws)>=2:
                    seen+=1;pp=list(itertools.combinations(ws,2));pairs.update(pp);dset.update(pp)
                    if len(samples)<12 and not any(x[0]==f.name for x in samples):samples.append([f.name,part.strip().replace('\n',' ')[:300],' '.join(ws)])
            pairdocs.update(dset);detail.append([f.name,du,seen,len(dset)])
        G=nx.Graph();G.add_nodes_from(selected)
        for (x,y),n in pairs.items():
            cos=n/(cnt[x]*cnt[y])**.5
            if n>=5 and pairdocs[x,y]>=3 and cos>=.1:G.add_edge(x,y,weight=cos,count=n,docs=pairdocs[x,y])
        return G,cnt,pairs,pairdocs,units,detail,samples
    # sentence and clause use punctuation boundaries, not arbitrary PDF line breaks
    G,cnt,pairs,pairdocs,units,detail,samples=calc(r'[。！？!?；;]+|\n\s*\n','exploratory')
    G2,c2,p2,d2,u2,_,_=calc(r'[。！？!?]+|\n\s*\n','exploratory')
    G3,c3,p3,d3,u3,_,_=calc(r'[。！？!?；;]+|\n\s*\n','base')
    # Fixed exploratory vocabulary: sensitivity isolates window/filter effect on the same nodes.
    active=G.subgraph([n for n in G if G.degree(n)>0]).copy();groups=sorted(nx.community.louvain_communities(active,weight='weight',seed=42),key=lambda s:(-len(s),sorted(s)));labels={w:i+1 for i,g in enumerate(groups) for w in g};mod=nx.community.modularity(active,groups,weight='weight')
    strength=dict(G.degree(weight='weight'));degree=dict(G.degree());docfreq=Counter(w for ts in docs for w in set(ts));freq=Counter(w for ts in docs for w in ts)
    nodes=sorted(selected,key=lambda w:(-strength[w],-degree[w],w));edges=sorted(G.edges(data=True),key=lambda e:(-e[2]['weight'],-e[2]['count'],e[0],e[1]));edgehead=['词语1','词语2','共现句段数','共现政策数','余弦归一化权重']
    save('nodes.csv',['词语','平均TFIDF排名','平均TFIDF','词频','政策文档频数','含词句段数','度','加权度','群组'],[[w,rank[w],wt[w],freq[w],docfreq[w],cnt[w],degree[w],strength[w],labels.get(w,0)] for w in nodes])
    save('edges.csv',edgehead,[[x,y,d['count'],d['docs'],d['weight']] for x,y,d in edges]);save('all_pairs.csv',edgehead,[[x,y,pairs[x,y],pairdocs[x,y],pairs[x,y]/(cnt[x]*cnt[y])**.5 if cnt[x] and cnt[y] else 0] for x,y in itertools.combinations(sorted(selected),2)])
    save('cooccurrence_matrix.csv',['词语']+selected,[[w]+[cnt[w] if w==v else pairs[tuple(sorted((w,v)))] for v in selected] for w in selected]);save('document_unit_stats.csv',['文件名','有效句段数','含两个以上入选词的句段数','不同共现词对数'],detail);save('unit_samples.csv',['文件名','句段片段（最多300字符）','入选词'],samples)
    save('communities.csv',['群组','词语数','词语（按加权度排序）'],[[i+1,len(g),'、'.join(sorted(g,key=lambda w:(-strength[w],w)))] for i,g in enumerate(groups)])
    def eset(g):return {tuple(sorted(e)) for e in g.edges()}
    def compare(g,u,label):
        a,b=eset(G),eset(g);return [label,u,len(b),len(a&b),len(a&b)/len(a|b) if a|b else 1]
    sens=[compare(G,units,'句段_探索版'),compare(G2,u2,'较长句_探索版'),compare(G3,u3,'句段_基础版')];save('sensitivity.csv',['方案','有效单位数','边数','与主方案重合边','边集合Jaccard'],sens)
    # GraphML is importable into Gephi; Unicode labels preserved.
    for w in G:
        G.nodes[w].update(label=w,community=int(labels.get(w,0)),strength=float(strength[w]),frequency=int(freq[w]),tfidf_rank=int(rank[w]))
    nx.write_graphml(G,out/'network.graphml')
    # Visual only: strongest 90 edges plus maximum spanning forest; preserve connectivity and community assignments.
    H=nx.Graph();H.add_nodes_from(active);H.add_edges_from(edges[:90]);H.add_edges_from(nx.maximum_spanning_tree(active,weight="weight").edges(data=True));pos=nx.spring_layout(active,seed=42,weight="weight",k=.60,iterations=250)
    fp=FontProperties(fname=str(a.font)) if a.font else FontProperties(family='sans-serif')
    fig,ax=plt.subplots(figsize=(14,11));colors=plt.get_cmap('tab10');nx.draw_networkx_edges(H,pos,ax=ax,alpha=.24,width=[.6+3*d['weight'] for _,_,d in H.edges(data=True)],edge_color='#6b7280');nx.draw_networkx_nodes(H,pos,ax=ax,node_size=[120+480*strength[w]/max(strength.values()) for w in H],node_color=[colors((labels[w]-1)%10) for w in H],alpha=.9,linewidths=.7,edgecolors='white')
    for w,(x,y) in pos.items():ax.text(x,y+.028,w,fontproperties=fp,fontsize=9 if w not in nodes[:15] else 11,ha='center',va='bottom',color='#172030',bbox=dict(facecolor='white',alpha=.68,edgecolor='none',pad=.3))
    ax.set_title('数字乡村政策关键词共现网络（ICU试验）' if a.engine=='icu' else '数字乡村政策关键词共现网络（Jieba）',fontproperties=fp,fontsize=19,pad=18);ax.text(.5,-.045,'节点颜色：Louvain群组；节点大小：全网络加权度；绘制最强90条边及最大生成森林',transform=ax.transAxes,ha='center',fontproperties=fp,fontsize=11);ax.axis('off');fig.tight_layout();fig.savefig(out/'network.png',dpi=180,bbox_inches='tight');plt.close(fig)
    # compact heatmap of 20 most central terms, counts
    top=nodes[:20];M=np.array([[0 if x==y else pairs[tuple(sorted((x,y)))] for y in top] for x in top]);fig,ax=plt.subplots(figsize=(10,9));im=ax.imshow(M,cmap='Blues');ax.set_xticks(range(len(top)),top,rotation=60,ha='right',fontproperties=fp,fontsize=9);ax.set_yticks(range(len(top)),top,fontproperties=fp,fontsize=9);ax.set_title('核心词共现次数（对角线置零）',fontproperties=fp,fontsize=16);fig.colorbar(im,ax=ax,shrink=.8);fig.tight_layout();fig.savefig(out/'heatmap.png',dpi=180);plt.close(fig)
    summary=dict(engine=a.engine,engine_version=version,documents=len(files),source_hashes_verified=verified,previous_token_files_verified=2*tokens_verified,selected_keywords=len(selected),candidate_pairs=len(selected)*(len(selected)-1)//2,units=units,units_with_2_or_more_selected=sum(r[2] for r in detail),observed_pairs=len(pairs),edges=G.number_of_edges(),active_nodes=active.number_of_nodes(),isolated_nodes=sorted(set(G)-set(active)),communities=len(groups),modularity=mod,community_members=[sorted(g,key=lambda w:(-strength[w],w)) for g in groups],top_nodes=[[w,degree[w],round(strength[w],4)] for w in nodes[:10]],top_edges=[[x,y,d['count'],d['docs'],round(d['weight'],4)] for x,y,d in edges[:10]],sensitivity=sens,thresholds={'min_units':5,'min_documents':3,'min_cosine':.1},networkx_version=nx.__version__)
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
