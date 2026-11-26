"""Windows-friendly validation of the frozen digital-village V2 corpus.
Run from repository root: python policy_final_validation/validation.py prepare
Annotation CSVs open in Excel. No model predictions in annotation files.
"""
from pathlib import Path
import argparse, csv, hashlib, importlib.util, json, random, re, subprocess, sys, unicodedata
from collections import Counter, defaultdict

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / 'policy_analysis_v2' / 'policies'
QC = ROOT / 'policy_analysis_v2' / 'qc_changes.csv'
META = ROOT / 'policy_topics' / 'metadata_snapshot.tsv'
PRE = ROOT / 'policy_preprocessing' / 'preprocess.py'
DICT = ROOT / 'policy_topics' / 'theme_dictionary.tsv'
DEFAULT_OUT = ROOT / 'validation_runs' / 'baseline'
THEMES = ['数字基础设施','智慧农业','数字产业与流通','数据平台与技术','数字公共服务','数字治理','生态绿色建设']
LABELS = ['T'+str(i) for i in range(1,8)]
NONE = '无命中'
SEED = 20261126

def read_csv(path, delimiter=','):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f, delimiter=delimiter))

def save_csv(path, fields, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); w.writeheader(); w.writerows(rows)

def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def jsonsave(path, obj): Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

def read_lex(path):
    rows=read_csv(path,'\t'); out={r['主题']:set(r['词语'].split()) for r in rows}
    if list(out)!=THEMES or len(rows)!=7 or sum(map(len,out.values()))!=len(set.union(*out.values())):
        raise ValueError('主题词典必须为七类，词项不可跨类重复，顺序与原词典一致。')
    return out

def make_tokenizer(pre=PRE, engine='jieba'):
    # Mirrors the existing preprocess.py and co-occurrence tokenizer.
    spec=importlib.util.spec_from_file_location('existing_preprocess',pre)
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    cfg=Path(pre).parent/'config'
    terms=sorted(set(mod.lines(cfg/'domain_terms.txt')), key=lambda x:(-len(x),x)); termset=set(terms)
    base=set(mod.lines(cfg/'stopwords_base.txt')); extra=set(mod.lines(cfg/'stopwords_policy_optional.txt'))
    syn=dict(x.split('\t') for x in mod.lines(cfg/'synonyms.tsv'))
    if engine=='jieba':
        import jieba
        j=jieba.Tokenizer(); j.initialize()
        for t in terms: j.add_word(t,freq=1000000)
        cut=lambda s:j.lcut(s,HMM=False)
    elif engine=='icu': cut,_=mod.icu_splitter()
    else: raise ValueError(engine)
    pat=re.compile('('+'|'.join(map(re.escape,terms))+')')
    repairs=[(re.compile(r'[ \t]*(?:\n[ \t]*)?'.join(map(re.escape,t))),t) for t in terms if len(t)>1]
    lexical=re.compile(r'[\u3400-\u9fff]|[A-Za-z]'); han=re.compile(r'[\u3400-\u9fff]')
    def tokenize(s):
        s=unicodedata.normalize('NFKC',s).replace('\ufeff','').replace('\r\n','\n').replace('\r','\n')
        s=re.sub(r'https?://[^\s，。；、）)]+|www\.[^\s，。；、）)]+',' ',s)
        for rx,t in repairs:s=rx.sub(t,s)
        raw=[]
        for p in pat.split(s):
            if p:raw.extend([p] if p in termset else cut(p))
        raw=[w.strip() for w in raw if w.strip() and lexical.search(w) and (han.search(w) or len(w.strip())>=2)]
        b=[syn.get(w,w) for w in raw if w not in base and (len(w)>1 or w in termset)]
        return b,[w for w in b if w not in extra]
    return tokenize

def verify_sources(src=SRC,qc=QC,meta=META):
    expected={r['name']:r['new_sha256'] for r in read_csv(qc)}
    metadata={r['文件名']:r for r in read_csv(meta,'\t')}
    files={p.name:p for p in Path(src).glob('*.txt')}
    if len(files)!=59 or set(files)!=set(expected) or set(files)!=set(metadata):
        raise ValueError('59份文件与V2清单未一一对应，请确认仓库下载完整。')
    bad=[n for n,p in files.items() if digest(p)!=expected[n]]
    if bad: raise ValueError('SHA256不一致：'+', '.join(bad)+'。请使用 git -c core.autocrlf=false clone 重新下载，勿直接修改正文。')
    return files,metadata

def doc_split(metadata):
    # Split before choosing sentences; all excerpts of a document stay on one side.
    groups=defaultdict(list)
    for n,m in sorted(metadata.items()):groups[(m['层级'],m['区域'])].append(n)
    rng=random.Random(SEED); result={}
    for key,names in sorted(groups.items()):
        rng.shuffle(names); k=len(names)//2
        for i,n in enumerate(names):result[n]='dev' if i<k else 'test'
    return result

def units(text):
    # Keep raw text, split on sentence punctuation / blank lines, not PDF single line wraps.
    for i,part in enumerate(re.split(r'(?<=[。！？!?])|\n[ \t]*\n',text)):
        part=part.strip()
        size=len(re.sub(r'\s','',part))
        if 30<=size<=500 and len(re.findall(r'[\u3400-\u9fff]',part))>=20:
            yield i+1,part

def predicted(tokens, lex):
    s=set(tokens)
    return [int(bool(s & lex[t])) for t in THEMES]

def sample_batch(candidates, stage):
    # Assign each candidate to one exclusive stratum (rarest hit theme first).
    freq=Counter(t for r in candidates for t,y in zip(THEMES,r['pred']) if y)
    pools=defaultdict(list)
    for r in candidates:
        hit=[t for t,y in zip(THEMES,r['pred']) if y]
        key=min(hit,key=lambda t:(freq[t],THEMES.index(t))) if hit else NONE
        pools[key].append(r)
    quotas={t:(8 if (i<3)==(stage=='dev') else 7) for i,t in enumerate(THEMES)}
    quotas[NONE]=70-sum(quotas.values())
    rng=random.Random(SEED+(1 if stage=='dev' else 2)); sample=[]; stats=[]
    for key,n in quotas.items():
        pool=pools[key]; N=len(pool)
        if N<n: raise ValueError(f'{stage} {key}候选仅{N}条，要求{n}条；请保留日志，勿手工补造样本。')
        for r in rng.sample(pool,n):sample.append(dict(r,stratum=key,weight=N/n))
        stats.append({'stage':stage,'stratum':key,'pool':N,'sample':n})
    rng.shuffle(sample)
    for i,r in enumerate(sample,1):r['ID']=('D' if stage=='dev' else 'V')+f'{i:03d}'
    return sample,stats

def run(*args):
    print('RUN',*map(str,args),flush=True);subprocess.run([sys.executable,*map(str,args)],cwd=ROOT,check=True)

def prepare(out):
    if (out/'sample_manifest.json').exists():raise FileExistsError('该目录已有冻结样本，请不要覆盖标注。使用新 --out 目录。')
    files,meta=verify_sources();out.mkdir(parents=True,exist_ok=True)
    tok=make_tokenizer();lex=read_lex(DICT)
    run(PRE,'--input',SRC,'--qc',QC,'--output',out/'jieba','--engine','jieba')
    split=doc_split(meta);candidates=defaultdict(list);seen=set()
    for n,p in sorted(files.items()):
        raw=p.read_text(encoding='utf-8-sig')
        b,e=tok(raw)
        for mode,ts in [('base',b),('exploratory',e)]:
            assert ts==(out/'jieba'/f'tokens_{mode}'/n).read_text(encoding='utf-8').split(), f'分词复现不一致 {n} {mode}'
        for unit,part in units(raw):
            canonical=re.sub(r'\s','',unicodedata.normalize('NFKC',part))
            if canonical in seen:continue
            seen.add(canonical)
            tokens,_=tok(part)
            if not tokens:continue
            candidates[split[n]].append(dict(filename=n,unit=unit,text=part,pred=predicted(tokens,lex),**{k:meta[n][k] for k in ['年份','区域','层级','政策类型']}))
    sample=[];stats=[]
    for stage in ['dev','test']:
        rows,st=sample_batch(candidates[stage],stage);stats+=st
        for r in rows:r['stage']=stage
        sample+=rows
        blind=[dict(ID=r['ID'],文件名=r['filename'],句段编号=r['unit'],原文=r['text'],**{k:'' for k in LABELS},备注='') for r in rows]
        for coder in ['A','B']:
            save_csv(out/'annotation'/f'{stage}_{coder}.csv',['ID','文件名','句段编号','原文',*LABELS,'备注'],blind)
    assert len(sample)==140 and len({r['ID'] for r in sample})==140
    assert not ({r['filename'] for r in sample if r['stage']=='dev'} & {r['filename'] for r in sample if r['stage']=='test'})
    jsonsave(out/'sample_manifest.json',{'seed':SEED,'engine':'jieba','dictionary_sha256':digest(DICT),'config_sha256':{p.name:digest(p) for p in sorted((PRE.parent/'config').glob('*.txt'))}|{'synonyms.tsv':digest(PRE.parent/'config/synonyms.tsv')},'document_split':split,'sample':sample,'strata':stats})
    save_csv(out/'sample_strata.csv',['stage','stratum','pool','sample'],stats)
    print('完成：59份Jieba分词，开发70条，验证70条，两批文档不重叠。打开 annotation 文件夹。')

def labels(path,ids):
    data=read_csv(path);d={r['ID']:r for r in data}
    if len(d)!=len(data) or set(d)!=set(ids):raise ValueError(f'{path} 的ID与冻结样本不一致。')
    for i,r in d.items():
        if any(r.get(k,'').strip() not in ['0','1'] for k in LABELS):raise ValueError(f'{path} {i} T1—T7 必须全部填写0或1（不确定先复核）。')
    return {i:[int(d[i][k]) for k in LABELS] for i in ids}

def divide(a,b):return a/b if b else None

def measures(y,p,w):
    tp=sum(wi for yi,pi,wi in zip(y,p,w) if yi and pi);fp=sum(wi for yi,pi,wi in zip(y,p,w) if not yi and pi);fn=sum(wi for yi,pi,wi in zip(y,p,w) if yi and not pi);tn=sum(wi for yi,pi,wi in zip(y,p,w) if not yi and not pi)
    return dict(TP=tp,FP=fp,FN=fn,TN=tn,precision=divide(tp,tp+fp),recall=divide(tp,tp+fn),F1=divide(2*tp,2*tp+fp+fn))

def kappa(a,b):
    n=len(a);po=sum(x==y for x,y in zip(a,b))/n;pa=sum(a)/n;pb=sum(b)/n;pe=pa*pb+(1-pa)*(1-pb)
    return divide(po-pe,1-pe)

def evaluate(out,stage,a_path,b_path,gold_path,dictionary,pre):
    manifest=json.loads((out/'sample_manifest.json').read_text(encoding='utf-8'));sample=[r for r in manifest['sample'] if r['stage']==stage];ids=[r['ID'] for r in sample]
    a=labels(a_path,ids);b=labels(b_path,ids) if b_path else None
    if b and not gold_path:
        # Do not silently use either annotator as the reference.
        rows=[]
        for r in sample:
            i=r['ID'];row={'ID':i,'文件名':r['filename'],'句段编号':r['unit'],'原文':r['text'],'备注':''}
            row.update({k:str(a[i][j]) if a[i][j]==b[i][j] else '' for j,k in enumerate(LABELS)})
            row.update({k+'_A':a[i][j] for j,k in enumerate(LABELS)});row.update({k+'_B':b[i][j] for j,k in enumerate(LABELS)});rows.append(row)
        target=out/'annotation'/f'{stage}_consensus.csv'
        if target.exists():raise FileExistsError(f'已有 {target}，请填写后使用 --gold，勿覆盖。')
        save_csv(target,['ID','文件名','句段编号','原文',*LABELS,*[k+'_A' for k in LABELS],*[k+'_B' for k in LABELS],'备注'],rows)
        print(f'已生成 {target}。两人讨论分歧并填完T1—T7，再用 --gold 运行。未计算模型准确率。');return
    if gold_path and not b:raise ValueError('使用 --gold 时同时提供 --coder-b。')
    gold=labels(gold_path,ids) if gold_path else a
    lex=read_lex(dictionary);tok=make_tokenizer(pre);pp={r['ID']:predicted(tok(r['text'])[0],lex) for r in sample}
    output=[]
    for mode in ['原始分层样本','按层抽样权重']:
        w=[1 if mode=='原始分层样本' else r['weight'] for r in sample]
        for j,t in enumerate(THEMES):
            z=measures([gold[i][j] for i in ids],[pp[i][j] for i in ids],w);output.append(dict(口径=mode,主题=t,**z))
        z=measures([v for i in ids for v in gold[i]],[v for i in ids for v in pp[i]],[wi for wi in w for _ in LABELS]);output.append(dict(口径=mode,主题='Micro',**z))
        fs=[x['F1'] for x in output if x['口径']==mode and x['主题'] in THEMES and x['F1'] is not None]
        output.append(dict(口径=mode,主题='Macro',F1=sum(fs)/len(fs) if fs else None))
    dest=out/'evaluation'/stage;dest.mkdir(parents=True,exist_ok=True)
    save_csv(dest/'metrics.csv',['口径','主题','TP','FP','FN','TN','precision','recall','F1'],output)
    detail=[dict(ID=i,**{k+'_人工':gold[i][j] for j,k in enumerate(LABELS)},**{k+'_程序':pp[i][j] for j,k in enumerate(LABELS)}) for i in ids]
    save_csv(dest/'comparison.csv',['ID',*[k+'_人工' for k in LABELS],*[k+'_程序' for k in LABELS]],detail)
    if b:save_csv(dest/'annotator_agreement.csv',['主题','Kappa','一致率'],[dict(主题=t,Kappa=kappa([a[i][j] for i in ids],[b[i][j] for i in ids]),一致率=sum(a[i][j]==b[i][j] for i in ids)/len(ids)) for j,t in enumerate(THEMES)])
    jsonsave(dest/'evaluation_info.json',dict(stage=stage,sample=len(ids),reference='two coders consensus' if b else 'one human coder',dictionary_sha256=digest(dictionary),config_hashes={p.name:digest(p) for p in sorted((pre.parent/'config').glob('*')) if p.is_file()},sample_manifest_sha256=digest(out/'sample_manifest.json'),annotator_a_sha256=digest(a_path),annotator_b_sha256=digest(b_path) if b_path else None,consensus_sha256=digest(gold_path) if gold_path else None,limitations='仅针对抽样框内30—500字符句段；分层富集，未加权指标不是总体估计；词典命中标签验证不等于分词准确率或词频权重的验证。'))
    print(f'结果已保存 {dest}，空值表示分母为零。')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','evaluate','keywords']);p.add_argument('--out',type=Path,default=DEFAULT_OUT);p.add_argument('--stage',choices=['dev','test']);p.add_argument('--coder-a',type=Path);p.add_argument('--coder-b',type=Path);p.add_argument('--gold',type=Path);p.add_argument('--dictionary',type=Path,default=DICT);p.add_argument('--preprocess',type=Path,default=PRE);a=p.parse_args();o=a.out.resolve()
    if a.action=='prepare':prepare(o)
    elif a.action=='keywords':run(ROOT/'policy_keywords'/'analyze_keywords.py','--tokens',o/'jieba','--metadata',META,'--out',o/'keywords')
    else:
        if not a.stage or not a.coder_a:p.error('evaluate 需要 --stage 和 --coder-a')
        evaluate(o,a.stage,a.coder_a,a.coder_b,a.gold,a.dictionary,a.preprocess)
if __name__=='__main__':main()
