# 数字乡村政策共词分析（探索性）

本阶段基于 policy_analysis_v2 的59份政策和 policy_preprocessing 的同一套词典及停用词配置，重新分词并逐份核对源SHA256和118份既有两套分词结果。运行环境没有Jieba，本次实际结果为ICU；不与未来Jieba结果混用。

## 方法

1. 采用探索版全样本平均TF-IDF的前60个词，TF-IDF配置与上一阶段一致（min_df=2、sublinear_tf、smooth_idf、L2归一化）。
2. 句号、问叹号、分号、空行作为句段边界；避免把PDF机械换行当作段落。每个有效句段内每词、每词对至多计一次。统计含词句段数 n_i、共同出现句段数 c_ij、共同出现政策数；余弦归一化 c_ij/sqrt(n_i*n_j)。
3. 筛边规则：共现句段>=5、共现政策>=3、余弦>=0.10。阈值为探索性操作选择，非统计显著性判据。用全网络加权度（相连边权重之和）衡量网络连接性；Louvain（seed=42）探索群组，不等同于LDA主题。
4. 敏感性：固定主分析的60词，比较不以分号分割的较长句，以及基础停用词过滤方案。固定词表下后者的边一致是预期现象，不能据此声称全部停用词选择稳健。下一轮还需改变候选词选择和边阈值检验。
5. 网络图为可读性仅绘制最强90条边及最大生成森林；群组和加权度均按完整网络计算。GraphML可直接导入Gephi。矩阵对角线为含词句段数，热力图对角线置零方便展示。

## 结果与限制

59份政策、8654个有效句段、6388个含至少两个候选词的句段，1770个候选词对均至少共现一次，筛后357条边、60个节点，5个群组，模块度0.2731。较长句方案388条边，主方案357条全部保留，边Jaccard 0.9201。群组是词汇结构的初步证据，不能直接解释为政策关注度或因果关系。词对高关联有时来自复合词被拆分（基础/设施、经营/主体），不是两种独立政策议题。建议正式分词前复核基础设施、经营主体、实数融合、高质量等词语，保留重要领域词、适度剔除通用词，再整体重跑。原V2语料未修改。词频和共现计数仍受长文本影响；后续年度和区域主题比较使用文档/省份等权。

## 运行（在仓库根目录）

依赖 Python、jieba（正式版）、scikit-learn、numpy、networkx>=3.2、matplotlib。ICU试验需要系统libicuuc。中文图形可通过 --font 指定本机中文字体路径。

```
python policy_cooccurrence/analyze_cooccurrence.py --input policy_analysis_v2/policies --qc policy_analysis_v2/qc_changes.csv --preprocess policy_preprocessing/preprocess.py --output policy_cooccurrence/formal_jieba --engine jieba
```

在已有分词结果时可传入 --previous-tokens 核验两个方案的逐篇分词。完整CSV、GraphML及PNG附于本地ZIP；仓库保存程序、说明及核心统计表。
