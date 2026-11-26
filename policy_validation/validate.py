from pathlib import Path
import argparse, itertools, hashlib,json
import numpy as np,pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score,adjusted_rand_score
from sklearn.preprocessing import normalize
from scipy.cluster.hierarchy import linkage,dendrogram
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import font_manager
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,default=Path(__file__).parent);a=p.parse_args();O=a.out;O.mkdir(exist_ok=True,parents=True);F=O/'figures';F.mkdir(exist_ok=True)
fp=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
if fp.exists():
 font_manager.fontManager.addfont(str(fp)); font=font_manager.FontProperties(fname=str(fp)).get_name()
else:
 font=next((f for f in ['Microsoft YaHei','SimHei','Noto Sans CJK SC'] if any(x.name==f for x in font_manager.fontManager.ttflist)), 'sans-serif')
plt.rcParams.update({'font.family':font,'axes.unicode_minus':False,'font.size':10,'savefig.dpi':240,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'path'})
colors=['#7899B5','#D9A17F','#8FA58A','#BA858C','#A499BD','#9F9185','#B5A56F'];themes=['数字基础设施','智慧农业','数字产业与流通','数据平台与技术','数字公共服务','数字治理','生态绿色建设'];regions=['东部','中部','西部','东北'];v='每万词命中';typ='数字产业与流通';P=a.input/'pilot_icu'
x=pd.read_csv(P/'document_theme_scores.csv');b=x[x.过滤方案=='base'].copy();e=x[x.过滤方案=='exploratory'].copy();s=pd.read_csv(P/'strict_document_scores.csv');assert len(b)==len(e)==len(s)==59*7
assert np.allclose(b[v],10000*b['命中词次']/b['保留词数']);assert np.allclose(s['严格每万词命中'],10000*s['严格命中']/s['保留词数']);assert not b.duplicated(['文件名','主题']).any()
def save(d,n):d.to_csv(O/n,index=False,encoding='utf-8-sig',float_format='%.8f')
def figsave(fig,n):
 for ext in ['png','svg']:fig.savefig(F/f'{n}.{ext}',bbox_inches='tight',facecolor='white')
 plt.close(fig)
def profiles(d,col):return d[d.层级=='省级'].groupby(['区域','省份/范围','主题'])[col].mean().unstack()[themes].sort_index()
pr=profiles(b,v);ps=profiles(s,'严格每万词命中');pe=profiles(e,v);pn=profiles(b[b.政策类型!=typ],v);assert len(pr)==29 and len(pn)==22
# L2 normalization for directional comparison, not scaling each theme to variance=1
scenarios={'基础词典':pr,'严格词典':ps,'探索停用词':pe,'排除流通专项':pn,'固定22省排除前':pr.loc[pn.index]};rows=[];labs={};chosen={}
for name,d in scenarios.items():
 z=normalize(d.to_numpy());assert (np.linalg.norm(z,axis=1)>0).all()
 for k in range(2,7):
  labels=[];sil=[]
  for seed in [11,23,42,73,97]:
   y=KMeans(n_clusters=k,n_init=20,random_state=seed).fit_predict(z);labels.append(y);sil.append(silhouette_score(z,y,metric='euclidean'))
  stability=np.mean([adjusted_rand_score(a,b) for a,b in itertools.combinations(labels,2)]);rows.append([name,len(d),k,np.mean(sil),stability,min(np.bincount(labels[2]))]);labs[(name,k)]=pd.Series(labels[2],index=d.index)
r=pd.DataFrame(rows,columns=['方案','省份数','K','平均轮廓系数','随机种子ARI','最小群组规模']);save(r,'cluster_model_comparison.csv');k=int(r[r.方案=='基础词典'].sort_values('平均轮廓系数',ascending=False).iloc[0].K);base=labs[('基础词典',k)];print('chosen K',k);print(r.round(3).to_string(index=False))
# canonical label ordered by commerce share descending
norm=pr.div(pr.sum(axis=1),axis=0);order=norm.assign(cluster=base).groupby('cluster')[typ].mean().sort_values(ascending=False).index;mapping={c:i+1 for i,c in enumerate(order)};base=base.map(mapping);assignment=pr.reset_index();assignment['群组']=base.values;save(assignment,'province_clusters.csv');cent=norm.assign(群组=base).groupby('群组').mean();size=base.value_counts().sort_index();ct=cent.reset_index();ct['省份数']=ct.群组.map(size);save(ct,'cluster_profiles.csv');print('members',[(int(c),list(base[base==c].index.get_level_values(1))) for c in size.index]);print('profiles',ct.round(3).to_string(index=False))
ari=[]
for name,d in scenarios.items():
 y=labs[(name,k)];common=base.index.intersection(y.index);ari.append([name,len(common),k,adjusted_rand_score(base.loc[common],y.loc[common])]);
# isolate removal effect fixed identical sample fitted independently
ari.append(['固定22省排除前后',22,k,adjusted_rand_score(labs[('固定22省排除前',k)],labs[('排除流通专项',k)])]);ar=pd.DataFrame(ari,columns=['比较方案','共同省份数','K','ARI']);save(ar,'cluster_sensitivity.csv');print(ar.round(3).to_string(index=False))
# leave one province out regional ranges, descriptive not CI
lo=[]
for name,d in [('全样本',pr),('排除流通专项',pn)]:
 for reg in regions:
  q=d.xs(reg,level=0);lv=np.array([q.drop(i).mean().values for i in q.index]);
  for j,t in enumerate(themes):lo.append([name,reg,t,len(q),q[t].mean(),lv[:,j].min(),lv[:,j].max()])
ld=pd.DataFrame(lo,columns=['方案','区域','主题','省份数','均值','逐省剔除最小值','逐省剔除最大值']);save(ld,'leave_one_province_out.csv');print(ld[ld.主题==typ].round(1).to_string(index=False))
# central workpoints three scoring variants, same docs
ids=['C006_2022.txt','C007_2023.txt','C008_2024.txt','C009_2025.txt'];w=[]
for name,d,col in [('基础词典',b,v),('严格词典',s,'严格每万词命中'),('探索停用词',e,v)]:
 q=d[d.文件名.isin(ids)].pivot(index='年份',columns='主题',values=col)[themes];q['方案']=name;w.append(q.reset_index())
ww=pd.concat(w);save(ww,'workpoint_sensitivity.csv');print(ww[['年份','方案','智慧农业','数据平台与技术','数字公共服务']].round(1).to_string(index=False))
# plots
fig,ax=plt.subplots(figsize=(7.8,4.5));
for i,(name,d) in enumerate(r.groupby('方案',sort=False)):ax.plot(d.K,d.平均轮廓系数,'o-',color=colors[i],label=name,lw=1.8)
ax.set_xticks(range(2,7));ax.set_xlabel('聚类数量 K');ax.set_ylabel('平均轮廓系数');ax.grid(axis='y',alpha=.18);ax.legend(frameon=False,fontsize=9);ax.set_title('省际聚类数量比较（五个随机种子）',loc='left',fontweight='bold');figsave(fig,'10_cluster_model_comparison')
fig,ax=plt.subplots(figsize=(9,4.8));im=ax.imshow(cent.to_numpy()*100,cmap=LinearSegmentedColormap.from_list('soft',['#F5F8FA','#D5E3EA','#A2BFCF','#7197AD','#506F87']),aspect='auto',vmin=0,vmax=100);ax.set_xticks(range(7),themes,rotation=25,ha='right');ax.set_yticks(range(k),[f'群组 {c}（{size[c]}省）' for c in cent.index]);
for i,j in itertools.product(range(k),range(7)):ax.text(j,i,f'{cent.iloc[i,j]*100:.1f}',ha='center',va='center',color='#18314a' if cent.iloc[i,j]<.8 else 'white')
fig.colorbar(im,ax=ax,label='词典命中构成（%）');ax.set_title('省际政策群组主题构成（探索性）',loc='left',fontweight='bold');fig.text(.02,.01,'注：主题命中构成不是真实语义占比；聚类使用L2归一化的七维主题密度。',fontsize=9,color='#555');fig.subplots_adjust(bottom=.26);figsave(fig,'11_cluster_profiles')
fig,ax=plt.subplots(figsize=(10,4.8));z=linkage(normalize(pr.to_numpy()),method='ward');dendrogram(z,labels=list(pr.index.get_level_values(1)),leaf_rotation=90,leaf_font_size=9,link_color_func=lambda _:colors[0],ax=ax);ax.set_ylabel('Ward合并距离');ax.set_title('省际政策主题层次聚类（辅助展示）',loc='left',fontweight='bold');figsave(fig,'12_hierarchical_tree')
fig,ax=plt.subplots(figsize=(9,4.7));q=ld[ld.主题==typ];idx=np.arange(4)
for i,(name,g) in enumerate(q.groupby('方案',sort=False)):
 g=g.set_index('区域').loc[regions];y=g['均值'].to_numpy();ax.errorbar(idx+(-.1 if i==0 else .1),y,yerr=[y-g['逐省剔除最小值'],g['逐省剔除最大值']-y],fmt='o',capsize=5,color=colors[i],label=name,lw=1.8)
ax.set_xticks(idx,regions);ax.set_ylabel('每万词主题命中次数');ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.18);ax.legend(frameon=False);ax.set_title('产业流通主题的逐省剔除敏感性',loc='left',fontweight='bold');fig.text(.02,.01,'误差线表示逐一剔除省份后的最小—最大范围，并非置信区间。',fontsize=9,color='#555');fig.subplots_adjust(bottom=.18);figsave(fig,'13_leave_one_province_out')
summary={'selected_k':k,'cluster_sizes':{int(c):int(size[c]) for c in size.index},'silhouette':float(r[(r.方案=='基础词典')&(r.K==k)].iloc[0].平均轮廓系数),'seed_ari':float(r[(r.方案=='基础词典')&(r.K==k)].iloc[0].随机种子ARI),'input_sha256':{n:hashlib.sha256((P/n).read_bytes()).hexdigest() for n in ['document_theme_scores.csv','strict_document_scores.csv']}};(O/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
