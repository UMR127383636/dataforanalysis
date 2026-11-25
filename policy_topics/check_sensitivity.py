from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--qc',type=Path,required=True);p.add_argument('--stats',type=Path,required=True);a=p.parse_args();o=a.output
r=pd.read_csv(o/'document_theme_scores.csv');h=pd.read_csv(o/'matched_terms.csv');q=pd.read_csv(a.qc);s=pd.read_csv(a.stats);v=q.merge(s,left_on='name',right_on='文件名',validate='one_to_one');assert len(v)==59 and (v.new_sha256==v.源SHA256).all();
for mode,col in [('base','基础过滤词数'),('exploratory','探索过滤词数')]:
 x=r[r['过滤方案']==mode].groupby('文件名',as_index=False)['保留词数'].first().merge(s,on='文件名',validate='one_to_one');assert len(x)==59 and (x['保留词数']==x[col]).all()
# Conservative variant excludes context-ambiguous terms; this is a heuristic, not a validated classifier.
ambiguous='通信 电信 广电 广播电视 农机 无人机 遥感 设施农业 设施种植 智能控制 物流 快递 配送 直播 供应链 普惠金融 大数据 物联网 科技创新 技术创新 关键核心技术 信息系统 医疗 教育 养老 信息服务 公共服务 政务服务 数字鸿沟 农产品质量安全 食品安全 应急管理 党建 治安 农村集体资产 绿色发展 绿色农业 耕地保护 耕地质量 节水灌溉 绿色生产'.split()
b=r[r['过滤方案']=='base'].copy();hh=h[(h['过滤方案']=='base') & h['词语'].isin(ambiguous)].groupby(['文件名','主题'])['次数'].sum().rename('removed').reset_index();b=b.merge(hh,on=['文件名','主题'],how='left');b['严格命中']=b['命中词次']-b.removed.fillna(0);b['严格每万词命中']=10000*b['严格命中']/b['保留词数'];agg=b.groupby('主题').agg(原始均值=('每万词命中','mean'),严格均值=('严格每万词命中','mean'),原始词次=('命中词次','sum'),严格词次=('严格命中','sum')).reset_index();agg['原始排序']=agg.原始均值.rank(ascending=False,method='min').astype(int);agg['严格排序']=agg.严格均值.rank(ascending=False,method='min').astype(int);agg.to_csv(o/'dictionary_sensitivity.csv',index=False,encoding='utf-8-sig',float_format='%.8f');b.to_csv(o/'strict_document_scores.csv',index=False,encoding='utf-8-sig',float_format='%.8f');(o/'strict_excluded_terms.txt').write_text('\n'.join(ambiguous)+'\n',encoding='utf-8');print(agg.to_string(index=False));print('Verified V2 source hash recorded in prior preprocessing and all 118 token counts.')
