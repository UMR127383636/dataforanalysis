# 政策主题识别与量化（ICU探索性分析）

数据：V2语料59份，沿用已核验的ICU分词试验包（两种停用词版本）。本次检查上轮分词统计记录的59个原文SHA256与V2清单一致，以及两套共118份分词文件词数一致；另记录本次分词文件的SHA256。不宣称重新从原始政策全文执行Jieba。完整政策为独立文档；主题词典为246个互斥词项、七类候选主题，人工根据前期文献和政策词典整理，尚未进行独立人工编码的准确率/召回率评估。生态绿色为候选维度。文档主题命中密度 = 主题词次数 / 保留总词数 × 10000；主题内占比 = 主题命中 / 全部词典命中。二者并非完整语义占比。基础版为主要词典分析口径；基础与探索分词均计算，严格词典另排除宽泛歧义词以观察敏感性。严格词典并不一定更准确；被排除词项在真实语境可能具有主题意义。完整结果包保留全部匹配词与抽查上下文。未匹配词不能当作政策未关注。词典命中仅覆盖基础词次约5.96%，不宜将跨主题词频排序直接作为政府政策偏好。

LDA使用探索版词频矩阵（min_df=3,max_df=0.90），而非TF-IDF；59份完整文档、2438词项。K=3至8，随机种子11、42、73，batch，max_iter=25。每次用完整文档二值词共现计算TOP10词平均NPMI（零共现=-1）；用Jensen-Shannon距离及匈牙利匹配计算不同随机种子主题分布一致性（1-平均匹配距离）。训练集困惑度仅供参考，非独立测试。按平均NPMI选择示范K=3，固定seed=42展示，非最优主题数证明。LDA与词典使用同一语料，属于方法对照而非独立外部验证；粗粒度潜在主题不等同于七维建设主题。没有用LDA权重替代主题词典指标。

年度中央样本按政策等权，省级按省份等权后汇总四大区域。缺失天津、海南；样本类型、年份与地区覆盖不均衡，零命中不能解释为零关注，不能进行因果推断或强显著性结论。此次汇总仅为后续分析输入。后续需在正式Jieba（或经验证的替代分词器）上重跑、修正术语、以原文段落做人工编码抽查，再检查同类型年度比较及排除电商样本、排除指南等敏感性。

运行（仓库根目录）：
```
python -m pip install numpy pandas scipy scikit-learn
python policy_topics/analyze_topics.py --tokens policy_preprocessing/jieba_results --metadata policy_topics/metadata_snapshot.tsv --output policy_topics/jieba_results --engine jieba
python policy_topics/check_sensitivity.py --output policy_topics/jieba_results --qc policy_analysis_v2/qc_changes.csv --stats policy_preprocessing/jieba_results/document_stats.csv
```
ICU试验复现时改为试验包中的 `pilot_icu` 分词目录和 `--engine icu`。正式运行时需更新词典并保持版本记录。GitHub保留代码、配置和核心结果；ZIP包括全部词级结果、模型比较、模型文档权重、抽查分词上下文、输入哈希与敏感性结果。
