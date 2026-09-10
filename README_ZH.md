# B200 基线评测：续跑与统计 TXT 交接

本包比较 **LatentMAS、LatentMAS-H2O、LatentMAS-Hidden、LatCom、Interlat** 在 Qwen3-4B / Qwen3-8B 上的表现。七个通用任务固定抽样约 10%，**LATEN 自有 Benchmark 保留全量 648 条 / 486 对**。完整矩阵为 80 个评测单元、12,390 次任务推理；训练数据生成与训练另计。

本版修复 collect 过滤与独立分支调度，并将交付结果统一为 **统计 TXT 文件，每个最多 88,000 字节，严格低于 90 KB**。不导出问题、答案、代码、推理或 token IDs。已完成且身份一致的工作继续复用。

## 1. 接手现有服务器：直接继续未完成部分

先确认旧管线已退出，再在原 Git 仓库中更新。**保留原 LATEN_STORE 和配置**；不要删除模型、数据、运行目录或重建一份空 store。

GPU 3 继续原 4B 任务：

```bash
git pull --ff-only
GPU_ID=3 LATEN_STORE=/data/laten_baselines bash run_remaining.sh --family 4b
```

另一个终端中，GPU 5 执行 8B：

```bash
GPU_ID=5 LATEN_STORE=/data/laten_baselines_8b bash run_remaining.sh --family 8b
```

**目录名字带 `_8b` 不会选择 8B，必须显式写 `--family 8b`。** 该目录历史上重复产生的 4B 结果保留，但不算新的独立实验。两张卡必须使用不同 store；不要同时对同一个 store 启动两个管线。建议在 tmux 中运行。

如果原来使用了自定义配置，在命令末尾加 `--config configs/local.json`，使用原内容。只有一张卡时，不指定 family 会依次完成两个规模：

```bash
GPU_ID=0 LATEN_STORE=/data/laten_baselines bash run_remaining.sh
```

脚本依次做本地环境/资产校验、免训练基线补测、collect、四阶段训练、已训练方法评测和 TXT 汇总。某个方法失败会记录阻塞并继续独立分支；最后仍有失败时返回非零状态，不会把失败当成零分或宣称完成。

### 哪些会跳过，哪些需要继续

| 已有内容 | 本版行为 |
|---|---|
| 版本满足要求的 Python 环境 | 跳过 pip 安装 |
| 校验通过的模型、数据、源码、代码参考解测试 | 复用本地文件和测试记录；资产完整时不访问下载接口 |
| 已完整评分、配置与来源身份一致的评测 | 跳过模型加载与重复推理；已完成评测也可替代对应 smoke |
| 未完成的评测分片 | 仅继续剩余样本；已完成分片用于合并 |
| 新版 v2 collect / 训练已完成或有合法断点 | 验证后复用，或从断点继续 |
| 旧版失败的 training_cache | 原样保留；用修复后的过滤在 training_cache_v2 重新采集 |
| 未知代码或配置身份不匹配 | 明确报错、保留旧文件，不混合不同实验 |

本次只改结果持久化、汇总与续跑调度，保留 `471c487` 修复版的计算身份；三个免训练方法还兼容核验后的 `3272ff7` 身份。旧 collect 的错误门槛不能直接改成通过，必要的重新采集不属于重复已完成评测。

**磁盘：报告中的 90 GiB 剩余空间不足以安全训练 8B。** 续跑主机预检至少要求 150 GiB，训练阶段还会按实际模型/优化器大小检查更高峰值。两份 store 共用分区同时训练时，建议先留出 400–500 GiB；检查不会为另一进程预留空间，也不会自动删除用户文件。

## 2. 只导出统计，不启动实验

下面命令使用系统 Python，不加载模型、不占 GPU、不下载、不重新测试；可以在训练期间单独执行以刷新统计：

```bash
LATEN_STORE=/data/laten_baselines bash run.sh export-txt
LATEN_STORE=/data/laten_baselines_8b bash run.sh export-txt
```

`bash run.sh status` 和 `bash run.sh report` 也会刷新同一 TXT 目录。使用自定义配置时同样追加 `--config`。全部阶段完成或分支结束时，管线会自动刷新报告；长训练期间用上面的命令获取最新已写入的步数与损失。

**只需传回各 store 的 `STATS_TXT/` 中全部 TXT 文件。** 先等导出命令返回再复制，两个 store 分开放置，避免同名文件覆盖。

```text
/data/laten_baselines/STATS_TXT/
├── 00_README.txt                 # 第一个读：范围、口径与交接说明
├── 01_INDEX.txt                  # 第二个读：顺序、字节数、SHA256
├── 02_overview.p001.txt          # 完成数、模型版本、种子、配置上限
├── 03_general.p001.txt           # 七任务得分、微/宏平均、整路径 Token
├── 04_laten.p001.txt             # LATEN 总指标与效率
├── 05_laten_4b_*.p001.txt        # 各方法分层、CF、条件错误率
├── 05_laten_8b_*.p001.txt
├── 06_training.p001.txt          # 各阶段步数、数值损失记录
├── 07_collect.p001.txt           # 扫描量、保留量、各方法门槛
├── 08_failures.p001.txt          # 历史阻塞次数、错误类型（无原始异常文本）
└── 09_scheduler.p001.txt         # 已记录的各次调度历时
```

超长内容自动分为 `.p001.txt`、`.p002.txt` 等；**按 UTF-8 字节切分，不是按字符数**，中文也满足限制。索引列出实际生成的全部统计分片。`NA` 表示暂缺，不是 0 分；历史失败计数在成功续跑后仍保留，最终完成情况看 `02` / `03` / `04`。

交接时以 store1 的 4B、store2 的 8B 为主要来源；各目录也可能显示另一规模的历史数据或待运行项，不要把重复 4B 结果相加。

## 3. 本机续跑资产与交付结果的区别

`STATS_TXT/` 是唯一交付目录，只有 TXT。本机仍需保留模型、数据、训练张量/目标、权重、优化器断点、身份清单、最小统计续跑记录和必要运行状态，否则无法断点续跑。这些**不需要传出服务器**。

新评测不再保存推理文本、生成代码或 token IDs；本地语义评分记录保留配对统计所需的预测 bit 向量，TXT 只输出聚合指标。训练本身需要的教师目标留在训练资产中，不进入交付目录。旧版本已有原始文件不自动删除，也不复制到新报告。新通用样本若在评分落盘前被中断，该未完成样本可能重新生成；已经完成评分的样本不重做。

内部路径（用于本机维护，不作为交付清单）：

```text
LATEN_STORE/
  models/                         # 固定版本模型
  data/                           # 原始数据与固定抽样清单
  runs/<4b或8b>/training_cache_v2/  # 修复后采集资产
  runs/<4b或8b>/training_collect_v2/# 四阶段训练与断点
  runs/<4b或8b>/evaluation_sample10/             # 三个免训练方法
  runs/<4b或8b>/semantic_parallel/
  runs/<4b或8b>/evaluation_sample10_collect_v2/  # LatCom / Interlat
  runs/<4b或8b>/semantic_parallel_collect_v2/
  STATS_TXT/                      # 唯一对外交接目录
```

细节见 [TXT 交接策略](docs/TXT_HANDOFF.md)。旧文档提到的 JSON/CSV 汇总与原始输出交付要求，由本版 TXT 策略替代。

## 4. 测评范围与指标

| 数据集 | 源数据题数 | 每方法/模型实际题数 | 指标 | 输出 token 上限 |
|---|---:|---:|---|---:|
| GSM8K | 1,319 | 132 | 数值准确率 | 2,048 |
| ARC-E | 2,376 | 238 | 选择题准确率 | 2,048 |
| ARC-C | 1,172 | 117 | 选择题准确率 | 2,048 |
| MedQA | 300 | 30 | 选择题准确率 | 4,096 |
| MBPP+ | 378 | 38 | 扩展测试 pass@1 | 4,096 |
| HumanEval+ | 164 | 16 | 扩展测试 pass@1 | 4,096 |
| GPQA-Diamond | 198 | 20 | 选择题准确率 | 8,192 |
| LATEN | 648 | **648** | 事实向量、动作、CF 等 | 2,048 |

通用任务每组 591 题，种子 42，按复杂度代理的三等分分层抽取，包含较长/复杂任务；不是仅挑简单题。代理不是已验证的难度标签。MedQA 使用已有 300 题子集；代码任务使用固定 EvalPlus parquet 扩展测试。所有方法和模型共享固定 ID，不因答错、OOM 或解析失败换题。采样分数不能称为全量通用测试分数。

`03_general` 中逐任务准确率为正确数 / 已评分数；`micro_accuracy_completed` 按已评分题数加权，未完成时只是部分结果；`macro_accuracy_complete_only` 仅在七任务全部完成后计算等权平均。数值范围 0–1。代码测试基础设施错误不写成模型零分；报告中的全零不能单凭样本少归因。

LATEN 保持 162 案例、648 条件、486 BASE/CF 对，dev 108 / test 540，G4/G5/G6、I3/I6/I9、LOOKUP/RULE/DERIVE 均保留。报告向量精确率、逐 bit、确定性动作、模型动作诊断、目标/非目标 CF、整对向量与动作正确条件下的事实错误率；完整定义见 [LATEN 协议](docs/LATEN_BENCHMARK_PROTOCOL.md)。test 是已暴露的诊断划分。

效率覆盖**整条 Agent 任务路径**：生成文本 tokens、latent 位置、prompt tokens、模型 prefill、通信位置/字节分别统计，不能加在一起统称文本 token。并行耗时仅作争用条件下诊断；重叠任务累计秒数不等于阶段历时。`09_scheduler` 报告已结束调度尝试的记录，包含加载、评分和重试；缺失记录不补造。

## 5. collect 修复与训练方法

修复了 `Answer: Paris<|im_end|>` 因控制标记误判为错误的解析问题。只清理已知控制标记并匹配最终答案，不用推理中出现 gold 的子串放行。

LatCom 使用与阶段一一致的 `[single_gold_Z; q]` 完整轨迹可答性门槛，并排除仅问题可解样本；Interlat 独立要求带证据的文本计划正确回答，不被冻结 Receiver 的 raw latent 可读性统一阻塞。两者各至少 32 条合格主样本，64 条辅助数学/代码样本不能充数。仍不满足时报告阻塞并继续其他独立任务，不训练随机替代权重。见 [collect 修复说明](docs/COLLECT_REPAIR.md)。

训练来源为 HotpotQA / MuSiQue train 与辅助数学/代码 train，LATEN 不进入训练。默认扫描 1,024 主候选、最多保留 512 主样本，每模型四阶段各 300 optimizer steps：LatCom stage1 / stage2、Interlat receiver / compression。训练串行，每 25 步保存断点；不缩减原训练预算。

本包是统一协议迁移实现：**Interlat / LatCom 为 `paper_derived_port`，不是作者原始 checkpoint，也不是本工程 V6 权重。** 具体通信输入、目标函数及来源边界见 [方法协议](docs/PROTOCOL.md)。

## 6. 新服务器首次运行与并行

Linux，Python 3.10–3.12 + venv，Git，可用 Docker，B200 至少 170 GiB 显存、主机内存至少 256 GB；首次运行至少 500 GiB 空闲磁盘，建议 1 TB。固定 PyTorch 2.7.1 / CUDA 12.8，检查 sm_100 和 BF16；不改系统驱动、不使用 sudo。大文件默认放 `storage/`，可指定路径：

```bash
GPU_ID=0 LATEN_STORE=/data/laten_baselines bash run.sh
```

仅缺少资产时下载固定模型/数据、准备代码评分容器和参考解测试。需访问 Hugging Face、GitHub、PyTorch 和 Docker 源；凭据/代理由目标机配置。

每个评测阶段默认 2 个推理进程共享一张卡，各 45% PyTorch 分配器上限；模型加载/realignment 初始化串行。OOM 分片等待其他进程结束后以单进程 90% 上限继续，保留完成结果、不减题、不降 token 上限。训练与模型规模之间串行。分配器上限不覆盖所有 CUDA 开销，不能保证任意输入都不 OOM。

## 7. 常用命令与本地验证边界

```bash
# 单独继续收集（已有环境与数据）
GPU_ID=3 LATEN_STORE=/data/laten_baselines bash run.sh collect --family 4b
# 训练完成后只继续评测
GPU_ID=3 LATEN_STORE=/data/laten_baselines bash run.sh eval-all --family 4b
# 无 GPU 的静态/调度/续跑/TXT 检查
bash run.sh validate
# 已安装模型依赖时的小模型 CPU 检查
.venv/bin/python -m unittest discover -s tests -p 'test_tiny*.py' -v
```

本版 **43 项静态/调度测试 + 16 项小模型 CPU 测试通过**，包括旧身份复用、EOS 过滤修复、独立门槛、分片续跑、完整 648/486 指标一致性、禁止字段移除、中文 TXT 字节限制与索引。验证记录见 `evidence/validation.json`。这些证明本地工程检查通过；没有在准备服务器重跑完整模型/B200 实验，修复后的实际保留数、训练收敛及跑分仍由目标机续跑结果决定。
