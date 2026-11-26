# 期末报告最终验证工具（Windows / VSCode）

## 1. 准备环境

建议使用新的仓库副本，避免旧目录中的改动或 Windows 换行符转换影响 SHA256。

```powershell
cd D:\
git -c core.autocrlf=false clone https://github.com/UMR127383636/dataforanalysis.git digital_village_validation
cd digital_village_validation
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r policy_final_validation/requirements.txt
.\.venv\Scripts\python.exe policy_final_validation/validation.py prepare
```

若已下载仓库，进入仓库根目录并先执行 `git pull`。如果出现 SHA256 不一致，建议重新 clone，而非手工改正文或跳过核验。若不使用虚拟环境，也可把 `.\.venv\Scripts\python.exe` 替换成 `python`。

`prepare` 固定原 V2 59份正文、核对 SHA256，以原预处理配置执行 Jieba 0.42.1，核对全篇分词与抽样用分词函数的一致性。随后将政策按层级及区域划分为开发与验证文档组，在各组中按主题命中情况分层随机抽样，共140个原文句段（开发70、验证70；命中105、未命中35）。每个句段只进入一个抽样层，按较少见的命中主题优先确定。两组政策文件互不重叠，完全相同的归一化句段仅保留一次。句段以句末标点或空行切分，长度限30—500非空白字符，至少20个汉字。样本为分层富集样本，不能直接用未加权结果估计完整语料表现。句段来源为V2清洗文本，保留原始句段文字。记录抽样框、层大小、权重和随机种子。

## 2. 打开标注文件

文件夹 `validation_runs/baseline/annotation/`：

- `dev_A.csv`：第一位编码者的70条开发样本。
- `dev_B.csv`：相同70条开发样本，供第二位编码者独立填写。
- `test_A.csv`、`test_B.csv`：最终验证样本，各70条。开发期间不打开、不修改词典以适配此批数据。

CSV 使用 UTF-8 BOM，可以直接用 Excel 打开；如果出现乱码，使用 Excel 数据 → 自文本/CSV → 文件原始格式 UTF-8。填写 T1—T7 列，每一列填写 0 或 1；同一行允许多个 1；全部 0 表示无明确建设主题。保留原文件名、ID和原文，不改表头，保存为 CSV UTF-8。备注列用于记下边界案例。两个编码者不能先看彼此的答案或程序预测。若只有一名编码者，填写 A 即可，并在论文中说明。

不要在标注前打开 `sample_manifest.json`（含程序预测）和 `sample_strata.csv`。保留全部输出文件及两份独立标注文件。主题定义见 `coding_guide.md`。

## 3. 先评价开发样本

一名编码者：

```powershell
.\.venv\Scripts\python.exe policy_final_validation/validation.py evaluate --stage dev --coder-a validation_runs/baseline/annotation/dev_A.csv
```

两名编码者：

```powershell
.\.venv\Scripts\python.exe policy_final_validation/validation.py evaluate --stage dev --coder-a validation_runs/baseline/annotation/dev_A.csv --coder-b validation_runs/baseline/annotation/dev_B.csv
```

这会生成 `dev_consensus.csv`，一致标签自动填入，分歧标签留空。两人对照原文讨论，补齐 T1—T7 后运行：

```powershell
.\.venv\Scripts\python.exe policy_final_validation/validation.py evaluate --stage dev --coder-a validation_runs/baseline/annotation/dev_A.csv --coder-b validation_runs/baseline/annotation/dev_B.csv --gold validation_runs/baseline/annotation/dev_consensus.csv
```

结果保存至 `validation_runs/baseline/evaluation/dev/`，含逐主题 Precision、Recall、F1，Micro、Macro F1，以及两位编码者的逐主题 Kappa 和一致率。分母为0时保留空值。指标分为未加权的抽样样本结果和按层逆抽样率加权的结果；后者仅针对各自文档组、去重后的合格句段抽样框，不能外推为所有政策文本的精确率或召回率。

请先把 `validation_runs/baseline` 整个目录压缩发回，并说明是否有两名编码者。我们将根据开发样本错误分析修订领域词典及主题词典，锁定版本后再独立标注/评价 test，不能依据 test 结果继续调词典并用同一 test 当独立验证。

## 4. 可选：计算原配置的正式 Jieba TF-IDF

```powershell
.\.venv\Scripts\python.exe policy_final_validation/validation.py keywords
```

输出到 `validation_runs/baseline/keywords`。这一步使用未修订配置，目的是与已有ICU试验进行分词器对照，并非最终修订后的正式研究结果。

## 5. 修订后验证（暂不执行）

新版主题词典可用 `evaluate --dictionary <新版.tsv>` 指定。若领域词典也修订，需复制 `preprocess.py` 和其同级 `config/` 到独立目录（保留旧版），再用 `--preprocess <新版/preprocess.py>` 指定。冻结样本不重抽。开发阶段修改后复评，最终确定词典及预处理配置后才对 `--stage test` 运行。两人编码同理生成 test_consensus.csv。测试结果仅报告一次，不用于继续调参。

现有后续分析程序仍在 `policy_keywords`、`policy_cooccurrence`、`policy_topics`、`policy_research_results` 和 `policy_validation`。待词典验证完成后统一重跑并对比原ICU结果；图表脚本原先部分写死ICU标签及路径，需要同步更新，不能直接把原图改称Jieba结果。

**验证范围**：此工具评价句段多标签主题命中，不直接验证完整政策语义权重，也不等于分词边界准确率。另需核对 `jieba/segmentation_samples.csv`、`jieba/domain_term_audit.csv`，必要时增加人工分词金标准；部分政策原件核验仍需补齐。原仓库数据与ICU探索性分析均不被覆盖。
