# 数字乡村政策分词预处理（试验版）

输入为 GitHub `policy_analysis_v2/policies` 的59份正文，提交 d748b962d06f95079ce647e14e539e8d75fd2938。本次运行前逐份核对 V2 SHA256。原文不改动。

- `config/domain_terms.txt`：254个领域和政策常见复合词，基于研究方向和第一轮切分反馈手工整理；最长匹配保护复合词，内部空白/单换行修复限于已列术语。
- `config/stopwords_base.txt`：85个基本停用词（虚词、编号和发文套语），另剔除单字（专业词例外）及纯数字、标点和URL。
- `config/stopwords_policy_optional.txt`：61个可选政策通用词；基础版保留政策动作词，探索版额外过滤；不得用于政策工具分析而不加复核。
- `config/synonyms.tsv`：2组保守同义词归一规则（农村电子商务→农村电商；电商直播→直播电商）。不合并数字农业与智慧农业等不同概念。
- `pilot_icu/`：真实运行结果，ICU 76 中文分词 + 领域词最长匹配。当前环境未安装 Jieba，尝试安装失败。因此本目录结果仅用于预处理检查，不作为正式 Jieba 结果或研究结论。
- `domain_term_audit.csv`：构造句保护测试与语料最长匹配次数，测试不是人工分词准确率；嵌套术语不重复计数。`segmentation_samples.csv` 为12篇固定位置样本的切分片段（尚无人工金标准）。
- `tokens_base/`、`tokens_exploratory/` 分别为基础版和探索版分词；各59个文档。`pilot_frequency.csv` 仅探索版试验词频，未按文档长度或政策类型标准化。数字和年份在原文及元数据中保留，分词输出去除纯数字；后续如研究定量目标须另提取。

## 在 Windows / VSCode 重跑正式 Jieba 版本

在仓库根目录，安装依赖：

```powershell
python -m pip install -r policy_preprocessing/requirements.txt
python policy_preprocessing/preprocess.py --input policy_analysis_v2/policies --qc policy_analysis_v2/qc_changes.csv --output policy_preprocessing/jieba_results --engine jieba
```

本脚本默认 Jieba 0.42.1（HMM=False）并采用相同词典保护、归一和过滤策略；Jieba 路径已编写但当前未运行验证。先核对分词样本与专业术语，再使用正式输出计算 TF-IDF、共词和主题模型。ICU 路径仅限具有 libicuuc 的 Linux 环境。
