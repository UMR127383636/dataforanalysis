from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.spatial.distance import pdist, squareform
p=argparse.ArgumentParser(); p.add_argument('--input',type=Path,default=Path(__file__).parent/'input/policy_topics');p.add_argument('--out',type=Path,default=Path(__file__).parent);a=p.parse_args()
D=a.input/'pilot_icu'; O=a.out; F=O/'figures';T=O/'tables';F.mkdir(parents=True,exist_ok=True);T.mkdir(parents=True,exist_ok=True)
font_path=Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')
if font_path.exists():
 font_manager.fontManager.addfont(str(font_path))
 font=font_manager.FontProperties(fname=str(font_path)).get_name()
else:
 font=next((x for x in ['Noto Sans CJK SC','Microsoft YaHei','SimHei','Arial Unicode MS'] if any(f.name==x for f in font_manager.fontManager.ttflist)),None)
if font:plt.rcParams['font.family']=font
plt.rcParams.update({'axes.unicode_minus':False,'font.size':11,'figure.dpi':150,'savefig.dpi':240,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'path'})
themes=['数字基础设施','智慧农业','数字产业与流通','数据平台与技术','数字公共服务','数字治理','生态绿色建设']; regions=['东部','中部','西部','东北']; v='每万词命中'; typ='数字产业与流通'
x=pd.read_csv(D/'document_theme_scores.csv');s=pd.read_csv(D/'strict_document_scores.csv');m=pd.read_csv(a.input/'metadata_snapshot.tsv',sep='\t');b=x[x['过滤方案']=='base'].copy();e=x[x['过滤方案']=='exploratory'].copy();assert len(b)==59*7 and len(e)==59*7 and len(s)==59*7 and not b.duplicated(['文件名','主题']).any();assert len(m)==59 and set(b.文件名)==set(m.文件名) and (b['保留词数']>0).all();assert np.allclose(b[v],10000*b['命中词次']/b['保留词数']);assert np.allclose(s['严格每万词命中'],10000*s['严格命中']/s['保留词数']);cols=['文件名','层级','省份/范围','年份','区域','政策类型'];assert b[cols].drop_duplicates().sort_values('文件名').reset_index(drop=True).equals(m[cols].sort_values('文件名').reset_index(drop=True));
def csv(d,n):d.to_csv(T/n,encoding='utf-8-sig',float_format='%.8f')
def save(fig,n):
 fig.savefig(F/(n+'.png'),bbox_inches='tight',facecolor='white');fig.savefig(F/(n+'.svg'),bbox_inches='tight',facecolor='white');plt.close(fig)
def note(fig,t):fig.text(.01,.01,t,ha='left',fontsize=9,color='#555555')
def plot_lines(d,counts,title,n):
 fig,ax=plt.subplots(figsize=(10,5.6));d=d[themes];
 for k in themes:ax.plot(d.index,d[k],marker='o',lw=1.9,label=k)
 ax.set_xticks(d.index, [f'{y}\n(n={counts.loc[y]})' for y in d.index]);ax.set_ylabel('每万词主题命中次数');ax.set_xlabel('发文年份（政策数）');ax.set_ylim(bottom=0);ax.grid(axis='y',alpha=.2);ax.legend(loc='upper left',bbox_to_anchor=(1.01,1),frameon=False);ax.set_title(title,loc='left',fontweight='bold',pad=15);fig.subplots_adjust(bottom=.20,right=.77);note(fig,'来源：V2政策语料，ICU探索性分词；基础词典，政策等权。指标为词典命中密度，非真实语义占比。');save(fig,n)
def mat(d,col=v):return d.groupby(['区域','省份/范围','主题'])[col].mean().groupby(['区域','主题']).mean().unstack().reindex(regions)[themes]
def nreg(d):return d.groupby('区域').agg(政策数=('文件名','nunique'),省份数=('省份/范围','nunique')).reindex(regions)
def heat(d,labels,title,n,vmax=None,annotation=None):
 fig,ax=plt.subplots(figsize=(8.4,5.5 if len(d)<10 else 12));im=ax.imshow(d.to_numpy(),cmap='Blues',vmin=0,vmax=vmax,aspect='auto');ax.set_xticks(range(len(d.columns)),d.columns,rotation=28,ha='right');ax.set_yticks(range(len(d.index)),labels);ax.set_title(title,loc='left',fontweight='bold',pad=14);mx=im.norm.vmax
 for i in range(d.shape[0]):
  for j in range(d.shape[1]):
   z=d.iloc[i,j];ax.text(j,i,f'{z:.1f}',ha='center',va='center',fontsize=9 if len(d)<10 else 7,color='white' if z>mx*.62 else '#18314a')
 fig.colorbar(im,ax=ax,shrink=.75,label='每万词主题命中次数');fig.subplots_adjust(bottom=.27 if len(d)<10 else .14);note(fig,annotation or '来源：V2政策语料，ICU探索性分词；基础词典。先按省份平均，再按区域等权汇总。');save(fig,n)
# 1. sampling
counts=m.groupby(['年份','层级']).size().unstack(fill_value=0).reindex(range(2019,2026));csv(counts,'year_counts.csv');fig,ax=plt.subplots(figsize=(8.5,4.5));ax.bar(counts.index,counts['中央'],label='中央');ax.bar(counts.index,counts['省级'],bottom=counts['中央'],label='省级');
for y,row in counts.iterrows():ax.text(y,row.sum()+.25,str(row.sum()),ha='center',fontsize=10)
ax.set_xticks(counts.index);ax.set_ylim(0,17);ax.set_ylabel('政策数量（份）');ax.set_title('图1  2019—2025年政策样本构成',loc='left',fontweight='bold');ax.legend(frameon=False);ax.grid(axis='y',alpha=.15);fig.subplots_adjust(bottom=.15);note(fig,'来源：V2政策清单；59份代表性样本，不是全国政策发布总量。');save(fig,'01_sample_distribution')
# 2,3 central
c=b[b.层级=='中央']; annual=c.pivot_table(index='年份',columns='主题',values=v).reindex(columns=themes);csv(annual,'central_year.csv');plot_lines(annual,c.groupby('年份').文件名.nunique(),'图2  中央政策主题密度年度变化（全部中央样本）','02_central_all_years');ids=['C006_2022.txt','C007_2023.txt','C008_2024.txt','C009_2025.txt'];w=c[c.文件名.isin(ids)]; assert w.文件名.nunique()==4; wy=w.pivot_table(index='年份',columns='主题',values=v).reindex(columns=themes);csv(wy,'central_annual_workpoints.csv');plot_lines(wy,w.groupby('年份').文件名.nunique(),'图3  2022—2025年中央数字乡村工作要点（同类型）','03_central_workpoints')
# 4 5 6 regions and controls
pr=b[b.层级=='省级'];non=pr[pr.政策类型!=typ];eligible=set(non['省份/范围']);fixed=pr[pr['省份/范围'].isin(eligible)];ag=pr[pr.政策类型=='数字农业与信息化'];comp=pr[pr.政策类型=='综合性数字乡村'];groups={'全样本':pr,'排除产业流通专项':non,'固定省份原样本':fixed,'仅数字农业与信息化':ag,'仅综合性数字乡村':comp};rs=[];ns=[]
for label,d in groups.items():
 h=mat(d);csv(h,'region_'+{'全样本':'all','排除产业流通专项':'no_commerce','固定省份原样本':'fixed_all','仅数字农业与信息化':'agriculture','仅综合性数字乡村':'comprehensive'}[label]+'.csv');nr=nreg(d);nr['方案']=label;ns.append(nr.reset_index());long=h.stack().rename('密度').reset_index();long['方案']=label;rs.append(long)
csv(pd.concat(rs).set_index(['方案','区域','主题']),'region_sensitivity.csv');csv(pd.concat(ns).set_index(['方案','区域']),'region_coverage.csv')
mx=max(mat(pr).max().max(),mat(non).max().max());heat(mat(pr),[f'{r}（{nreg(pr).loc[r,"省份数"]}省）' for r in regions],'图4  四大区域政策主题密度（全部省级样本）','04_region_all',mx);heat(mat(non),[f'{r}（{nreg(non).loc[r,"省份数"]}省）' for r in regions],'图5  排除产业与流通专项后的区域主题密度','05_region_no_commerce',mx, '来源：V2政策语料；ICU试验。排除10份专项政策后仅覆盖22省；与图4共用色标。')
# fixed province comparison (same 22, but changed documents)
fig,ax=plt.subplots(figsize=(9,5.1));idx=np.arange(4);width=.25;forplot=[('全部29省',mat(pr)),('固定22省·排除前',mat(fixed)),('固定22省·排除后',mat(non))]
for i,(label,d) in enumerate(forplot):
 bars=ax.bar(idx+(i-1)*width,d[typ],width,label=label);ax.bar_label(bars,fmt='%.1f',fontsize=8,padding=3)
ax.set_xticks(idx,regions);ax.set_ylabel('产业与流通主题每万词命中');ax.set_title('图6  区域差异对政策类型与省份覆盖的敏感性',loc='left',fontweight='bold');ax.legend(frameon=False,loc='upper right');ax.set_ylim(0,400);ax.grid(axis='y',alpha=.18);fig.subplots_adjust(bottom=.16);note(fig,'固定22省指具有至少一份非产业流通政策的省份；排除前后使用相同省份，政策数仍不同。');save(fig,'06_region_commerce_sensitivity')
# 7 strict dictionary and filter comparison
sens=[]
for lab,d,col in [('基础词典',b,v),('严格词典',s,'严格每万词命中'),('探索停用词',e,v)]:
 for group,sub in [('中央',d[d.层级=='中央']),('省级',d[d.层级=='省级'])]:
  agg=sub.groupby('主题')[col].mean() if group=='中央' else sub.groupby(['省份/范围','主题'])[col].mean().groupby('主题').mean()
  for k,z in agg.items():sens.append([lab,group,k,z])
csv(pd.DataFrame(sens,columns=['方案','层级','主题','密度']).set_index(['方案','层级','主题']),'dictionary_filter_sensitivity.csv')
fig,ax=plt.subplots(figsize=(9,5.2));ind=np.arange(7);a0=b.groupby('主题')[v].mean().reindex(themes);a1=s.groupby('主题')['严格每万词命中'].mean().reindex(themes);ax.barh(ind-.18,a0,.35,label='基础词典');ax.barh(ind+.18,a1,.35,label='严格词典');ax.set_yticks(ind,themes);ax.invert_yaxis();ax.set_xlabel('每万词主题命中次数');ax.set_title('图7  主题词典宽严设置的敏感性（59份政策）',loc='left',fontweight='bold');ax.legend(frameon=False);ax.grid(axis='x',alpha=.15);fig.subplots_adjust(bottom=.17,left=.22);note(fig,'严格词典删除41个语境歧义词；两方案分母均为基础过滤词数。严格方案不等于人工验证结果。');save(fig,'07_dictionary_sensitivity')
# 8 province profiles
prov=pr.groupby(['区域','省份/范围','主题'])[v].mean().unstack()[themes];prov=prov.reindex(pd.MultiIndex.from_tuples([(r,p) for r in regions for p in sorted(pr[pr.区域==r]['省份/范围'].unique())],names=['区域','省份/范围']));csv(prov,'province_profiles.csv');heat(prov,[f'{r} · {p}' for r,p in prov.index],'图8  29个省级行政区的政策主题特征','08_province_profiles',annotation='来源：V2政策语料；ICU试验。同一省份政策等权。样本类型不同，不能直接作治理绩效排名。')
# 9 cosine profiles (not cluster: exploratory)
assert (np.linalg.norm(prov.to_numpy(),axis=1)>0).all();sim=1-squareform(pdist(prov.to_numpy(),metric='cosine'));labels=[p for r,p in prov.index];csv(pd.DataFrame(sim,index=labels,columns=labels),'province_cosine_similarity.csv');fig,ax=plt.subplots(figsize=(10,9));im=ax.imshow(sim,cmap='Blues',vmin=0,vmax=1);ax.set_xticks(range(29),labels,rotation=90,fontsize=9);ax.set_yticks(range(29),labels,fontsize=9);ax.set_title('图9  省际主题特征余弦相似度（探索性）',loc='left',fontweight='bold');fig.colorbar(im,ax=ax,shrink=.8,label='余弦相似度');fig.subplots_adjust(bottom=.13);note(fig,'基于7维主题密度向量；相似度受政策类型及词典覆盖影响，不表示区域政策互相影响。');save(fig,'09_province_cosine_similarity')
print('documents',len(m),'eligible provinces',len(eligible),'central annual workpoints\n',wy.round(1));print('fixed all\n',mat(fixed).round(1));print('region sensitivity\n',pd.concat(ns).to_string());print('all and filtered commerce',mat(pr)[typ].round(1).to_dict(),mat(non)[typ].round(1).to_dict());print('strict workpoints\n',s[s.文件名.isin(ids)].pivot_table(index='年份',columns='主题',values='严格每万词命中').round(1));
