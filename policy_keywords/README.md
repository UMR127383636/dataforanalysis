# 数字乡村政策关键词分析（ICU 试验版）

输入：V2 59份正式政策文本（中央17、省级42），原始语料不修改。分词输入为 `policy_preprocessing/pilot_icu` 的既有真实分词结果（ICU 76、领域词最长匹配），非正式 Jieba 结果。当前环境尝试安装 jieba 0.42.1 仍失败。`metadata_snapshot.tsv` 是从 GitHub V2政策清单（提交 d748b962d06f95079ce647e14e539e8d75fd2938）提取的六个分析字段；最终运行优先读取完整政策清单。

## 方法
- 两套停用词方案分别建模。基础版保留政策动作词，探索版剔除部分常见政策用语。两套分词分别153233及122698词。
- 每份政策为一个文档。全59篇共同拟合IDF（不是各组单独拟合）。min_df=2、sublinear_tf=True（1+ln(tf)）、smooth_idf=True（ln((1+N)/(1+df))+1）、文档L2归一化。基础版3132维，探索版3073维；低频专名等仅出现于1份政策的词不参与此次TF-IDF，但在词频表保留。
- 全样本、年度、层级、类型等按文档等权平均；四大区域先计算各省份的文档平均向量，再按省份等权平均。排除数字产业与流通政策作为额外敏感性场景，未覆盖省份不按零处理；中央2022—2025连续四年工作要点另行对照。
- 基础与探索两套全样本平均TF-IDF的TOP20重合7个，TOP50重合28个，提示过滤策略影响结果；不以某一套排名直接证明主题重要性。省域名称、申报类词语及普通政策套语仍可影响关键词，不能将排名当作最终政策主题。
- 样本数量、政策类型与年份不均衡；西部排除产业流通后17份/12省降至10份/6省，东北3份/3省降至2份/2省，敏感性结果同时受覆盖变化影响。年度结果尤其受文件类型与少数样本影响。
- 前一轮人工浏览片段发现部分复合词仍被切分，如实数融合、高质量、党的二十大。正式分词前需要补充词典并复核。没有人工金标准，不声称分词准确率。TF-IDF只是特征词筛查，下一步结合原文验证、主题词典及共词分析。

## 运行
在仓库根目录：
```
python -m pip install -r policy_preprocessing/requirements.txt
python -m pip install pandas numpy scikit-learn
python policy_preprocessing/preprocess.py --input policy_analysis_v2/policies --qc policy_analysis_v2/qc_changes.csv --output policy_preprocessing/jieba_results --engine jieba
python policy_keywords/analyze_keywords.py --tokens policy_preprocessing/jieba_results --metadata policy_analysis_v2/政策清单.csv --out policy_keywords/jieba_results
```
正式 Jieba 尚未执行，分析程序已在 ICU 输入上成功执行。若需要改词典，应保留旧配置，重新运行分词与本程序，并做人工抽查。GitHub中只保存精简结果和程序；完整试验结果包包含逐篇TOP20、分组TOP30、词频、分词文件哈希、敏感性对比等。
