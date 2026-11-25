"""Reproducible Chinese policy tokenization. Default Jieba; explicit ICU pilot only."""
import argparse,csv,ctypes as C,ctypes.util,hashlib,json,re,unicodedata
from pathlib import Path
from collections import Counter
ROOT=Path(__file__).resolve().parent

def lines(p):return [x.strip() for x in p.read_text(encoding='utf-8-sig').splitlines() if x.strip() and not x.startswith('#')]
def csvout(p,header,rows):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.writer(f);w.writerow(header);w.writerows(rows)

def icu_splitter():
    lib=C.CDLL(ctypes.util.find_library('icuuc')); v=re.search(r'\.so\.(\d+)',lib._name); suffix='_'+v.group(1) if v else ''
    def func(n,rest,args):
        f=getattr(lib,n+suffix);f.restype=rest;f.argtypes=args;return f
    ptr=C.POINTER(C.c_uint16)
    op=func('ubrk_open',C.c_void_p,[C.c_int,C.c_char_p,ptr,C.c_int32,C.POINTER(C.c_int32)])
    nx=func('ubrk_next',C.c_int32,[C.c_void_p]);close=func('ubrk_close',None,[C.c_void_p]);first=func('ubrk_first',C.c_int32,[C.c_void_p])
    def cut(s):
        if not s:return []
        b=s.encode('utf-16-le');a=(C.c_uint16*(len(b)//2)).from_buffer_copy(b);err=C.c_int32(0);it=op(1,b'zh_CN',a,len(a),C.byref(err))
        if err.value>0 or not it:raise RuntimeError(f'ICU error {err.value}')
        out=[]
        try:
            start=first(it)
            while (end:=nx(it))!=-1:
                out.append(b[start*2:end*2].decode('utf-16-le'));start=end
        finally:close(it)
        return out
    return cut,lib._name

def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--engine',choices=['jieba','icu'],default='jieba');p.add_argument('--qc',type=Path);args=p.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=True);(out/'tokens_base').mkdir(exist_ok=True);(out/'tokens_exploratory').mkdir(exist_ok=True)
    terms=lines(ROOT/'config/domain_terms.txt');terms=sorted(set(terms),key=lambda x:(-len(x),x));base=set(lines(ROOT/'config/stopwords_base.txt'));extra=set(lines(ROOT/'config/stopwords_policy_optional.txt'));syn=dict(x.split('\t') for x in lines(ROOT/'config/synonyms.tsv'))
    if args.engine=='jieba':
        import jieba
        j=jieba.Tokenizer();j.initialize()
        for t in terms:j.add_word(t,freq=1000000)
        cut=lambda s:j.lcut(s,HMM=False);version=jieba.__version__
    else:cut,version=icu_splitter()
    # The same longest-first domain matching protects named terms under either engine.
    pat=re.compile('('+'|'.join(map(re.escape,terms))+')');termset=set(terms)
    lexical=re.compile(r'[\u3400-\u9fff]|[A-Za-z]'); han=re.compile(r'[\u3400-\u9fff]')
    def normalize(s):
        s=unicodedata.normalize('NFKC',s).replace('\ufeff','').replace('\r\n','\n').replace('\r','\n');s=re.sub(r'https?://[^\s，。；、）)]+|www\.[^\s，。；、）)]+',' ',s)
        # Recover exact domain terms split by PDF whitespace without joining arbitrary lines.
        return s
    # Repair only whitespace internal to recognized terms, not general paragraph breaks.
    def repair(s):
        # For PDF single newlines, protect a term spanning a line only when no blank line intervenes.
        for t in terms:
            if len(t)>1:
                rx=re.compile(r'[ \t]*(?:\n[ \t]*)?'.join(map(re.escape,t)))
                s=rx.sub(t,s)
        return s
    def tokenize(s):
        raw=[];matches=Counter()
        for part in pat.split(s):
            if not part:continue
            if part in termset:raw.append(part);matches[part]+=1
            else:raw.extend(cut(part))
        # Retain Chinese words, ASCII technical terms (>=2 incl digit), no isolated punctuation or pure numbers.
        raw=[w.strip() for w in raw if w.strip() and lexical.search(w) and (han.search(w) or len(w.strip())>=2)]
        b=[syn.get(w,w) for w in raw if w not in base and (len(w)>1 or w in termset)]
        e=[w for w in b if w not in extra];return raw,b,e,matches
    expected={}
    if args.qc:
        with args.qc.open(encoding='utf-8-sig') as f:expected={r['name']:r['new_sha256'] for r in csv.DictReader(f)}
    files=sorted(args.input.glob('*.txt'));assert files and (not expected or set(expected)=={x.name for x in files})
    stats=[];counts=Counter();docfreq=Counter();domain=Counter();snips=[];rawtot=0;btot=0;etot=0
    for i,f in enumerate(files):
        data=f.read_bytes();h=hashlib.sha256(data).hexdigest();assert not expected or h==expected[f.name],f.name
        src=data.decode('utf-8-sig');s=repair(normalize(src));raw,b,e,mt=tokenize(s);assert b and e,f.name
        for mode,tok in [('base',b),('exploratory',e)]: (out/f'tokens_{mode}'/f.name).write_text(' '.join(tok)+'\n',encoding='utf-8')
        stats.append([f.name,len(re.sub(r'\s','',src)),len(raw),len(b),len(e),len(set(b)),sum(mt.values()),h]);rawtot+=len(raw);btot+=len(b);etot+=len(e);counts.update(e);docfreq.update(set(e));domain.update(mt)
        if i in [0,2,5,10,14,18,23,30,35,40,48,57]:
            # sample substantive paragraphs, not headings
            candidates=[z.strip() for z in re.split(r'(?<=[。！？])|\n\s*\n',s) if len(z.strip())>=65]
            if candidates:
                s0=candidates[0];rr,bb,ee,mm=tokenize(s0);snips.append([f.name,s0,' / '.join(rr),' / '.join(bb),' / '.join(ee)])
    # Synthetic protection tests: exact expected phrase survives even inside a sentence.
    checks=[]
    for t in terms:
        rr,bb,ee,mm=tokenize('推进：'+t+'。');checks.append([t,int(t in rr),domain[t],syn.get(t,t)])
    csvout(out/'document_stats.csv',['文件名','源非空白字符','原始词数','基础过滤词数','探索过滤词数','基础词汇量','领域词命中数','源SHA256'],stats)
    csvout(out/'domain_term_audit.csv',['术语','构造测试保留完整','语料匹配次数','标准词'],checks)
    csvout(out/'segmentation_samples.csv',['文件名','原文片段','原始切分','基础过滤','探索过滤'],snips)
    csvout(out/'pilot_frequency.csv',['词语','探索词频','文档频数'],[(w,c,docfreq[w]) for w,c in counts.most_common()])
    summary=dict(base_stopwords=len(base),optional_stopwords=len(extra),synonym_rules=len(syn),engine=args.engine,engine_version=version,documents=len(files),source_hashes_verified=len(files) if expected else 0,raw_tokens=rawtot,base_tokens=btot,exploratory_tokens=etot,exploratory_vocabulary=len(counts),domain_terms=len(terms),domain_terms_observed=sum(v>0 for v in domain.values()),domain_matches=sum(domain.values()),synthetic_tests_passed=sum(x[1] for x in checks),synthetic_tests=len(checks),sample_docs=len(snips),top20=counts.most_common(20))
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8');print(json.dumps(summary,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
