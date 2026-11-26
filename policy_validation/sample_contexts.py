from pathlib import Path
import argparse,pandas as pd
p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
d=pd.read_csv(a.input/'review_samples.csv');strict=set((a.input/'strict_excluded_terms.txt').read_text(encoding='utf-8-sig').split());d['歧义词']=d['命中词'].isin(strict);out=[]
for k,g in d.groupby('主题',sort=False):
 for amb in [False,True]:
  q=g[g['歧义词']==amb].drop_duplicates('文件名');out.append(q.sample(min(3,len(q)),random_state=42))
x=pd.concat(out).reset_index(drop=True);x.insert(0,'编号',range(1,len(x)+1));x.to_csv(a.output,index=False,encoding='utf-8-sig')
