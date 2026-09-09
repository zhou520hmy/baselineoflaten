# 单张 B200：五方法 × 两种模型规模 × 七个通用数据集及 LATEN Benchmark

这个目录可以整体复制到另一台服务器。默认模型为 **Qwen3-4B、Qwen3-8B**；单张 B200 串行运行，不使用当前工程的 V6 权重。正式矩阵为 **80 个单元，65,550 次任务推理**，另外包括训练数据生成、两种方法的训练和少量接口检查。

**重要定位：这是可运行的统一协议复现实验包，不是五种方法的作者原始 checkpoint 集合。** LatentMAS 系列依据固定版本公开实现与论文描述重建接口。Interlat 原项目的模型/任务设置与这里不同；LatCom 尚无核实到的公开作者代码和训练权重。因此，这两个方法包含本包的训练实现，结果始终标记 `paper_derived_port`。它们不能直接称为“完全复现作者结果”。详细公式、输入接口与差异见 [PROTOCOL.md](docs/PROTOCOL.md)。

## 一键启动

目标服务器需要 Linux、Python 3.10–3.12（含 venv）、Git、可供当前用户运行的 Docker、空闲 B200 和正常的 NVIDIA 驱动。代码任务必须使用 Docker 执行。准备至少 **256 GB 主机内存、500 GiB 空闲磁盘**，建议 1 TB 磁盘；这是工程容量门槛，尚未经过完整 8B 训练实测。环境安装固定使用 PyTorch 2.7.1 + CUDA 12.8，运行时检查 Blackwell `sm_100` 和 BF16 运算。脚本不使用 sudo，不修改系统驱动。

解压后进入本目录：

```bash
bash run.sh
```

这一条命令自动完成环境安装、固定版本下载、数据锁定、代码评分容器构建、全部 542 条代码参考解校验、接口检查、训练、评测和汇总。默认使用物理 GPU 0。需要自定大文件位置时：

```bash
GPU_ID=0 LATEN_STORE=/data/laten_baselines bash run.sh
```

`/data/laten_baselines` 是目标服务器的示例路径，须对当前用户可写。之后查询和续跑也使用同一个 `LATEN_STORE`。没有此环境变量时，全部大文件放在本目录 `storage/`。

建议在目标服务器的 tmux 会话中运行；需要后台运行并保存总日志时：

```bash
nohup bash run.sh > run.log 2>&1 &
```

网络必须能访问 Hugging Face、GitHub、PyTorch wheel 站和 Docker 镜像源。可按目标服务器现有配置设置 `HTTPS_PROXY`、`HF_TOKEN`；包中不包含任何凭据。

## 测试范围

| 数据集 | 每个方法/模型的题数 | 指标 | Receiver 输出上限 |
|---|---:|---|---:|
| GSM8K | 1,319 | 数值答案准确率 | 2,048 |
| ARC-E | 2,376 | 选择题准确率 | 2,048 |
| ARC-C | 1,172 | 选择题准确率 | 2,048 |
| MedQA | 300 | 选择题准确率 | 4,096 |
| MBPP+ | 378 | 扩展测试 pass@1 | 4,096 |
| HumanEval+ | 164 | 扩展测试 pass@1 | 4,096 |
| GPQA-Diamond | 198 | 选择题准确率 | 8,192 |

七个通用数据集每组共 5,907 题；新增 LATEN Benchmark 每组 648 条，总计每组 6,555 次任务。MedQA 使用 LatentMAS 已有的 300 题子集，**不是完整 MedQA 测试集**。代码任务固定使用 HF EvalPlus parquet 内的扩展测试，与现有项目协议一致，**不是重新生成的最新版 EvalPlus 测试**。所有版本、分母和原始文件哈希都会保留；完整源码地址/版本见 `evidence/hf_sources.json` 和 `evidence/medqa_provenance.json`。

同一题在各方法中使用相同公开问题、题目顺序与角色提示词，采样温度 0.6、top-p 0.95、关闭 top-k、种子 42 派生的逐题随机种子。每题生成一次回答。不同方法使用各自的通信接口，不能把“统一任务协议”理解为通信内容完全一样。

| 方法名/配置键 | 接口与运行步骤 |
|---|---|
| LatentMAS / `latentmas` | 三名 Sender 各 40 步 realigned latent，按官方层级实现连续复用 KV；Receiver 接收最终 KV，无新增训练。 |
| LatentMAS-H2O / `latentmas_h2o` | 同上，以实际 latent attention 对每层每个 KV head 的当前 prompt 位置排序；保留 64 个位置、历史和全部 latent，裁剪后保留绝对位置编码。64 是显式实验参数。 |
| LatentMAS-Hidden / `latentmas_hidden` | 保留三名 Sender 的 prompt input embeddings 和实际回灌的 aligned latent embeddings，在 Receiver 用户消息内部注入。 |
| LatCom / `latcom` | 冻结 Sender/Receiver，先训练同规模压缩器；40 步/源，经包含公开问题的压缩器生成全局 64 个槽位。 |
| Interlat / `interlat` | 先训练接收端 Adapter 与完整 Receiver，再训练自回归压缩 Sender；每个源 21 个 hidden states，三源共 63 个，外加通信边界向量。 |

Interlat 使用独立压缩 Sender；其余方法保留所依据公开层级实现中的 Sender 间 KV 历史。所有这些差异均在协议和原始输出中注明。

## 训练具体如何进行

每个模型规模单独构造训练缓存。主数据来自 **HotpotQA distractor train 与 MuSiQue-Ans train**，默认扫描 1,024 个候选、最多保留 512 个主样本；排除仅看问题已可回答的样本，以及完整 gold latent 路径仍无法回答的样本。少量辅助样本来自 GSM8K train 和与 MBPP+ 测试 task ID 隔离的 MBPP train，各 32 条。完整七项测试数据不参与优化。

利用训练样本的支持事实和 gold answer 生成 rationale+answer 目标，保存真实 full latent、完整文本计划 hidden states、独立错误消息候选来源等资产。生成的 rationale 没有额外人工/独立模型语义验证，相关标识写入缓存。

每个规模运行四个训练阶段，默认各 **300 个 optimizer steps**：

1. LatCom 阶段一：完整推理/答案 CE + 消息错配 margin + full-latent 分布 JS/均值锚定；global batch 64。
2. LatCom 阶段二：多源 gold/irrelevant/mixed 输入，CE + gold 源替换对比；global batch 64。
3. Interlat 接收训练：任务 CE、正确/错误消息分布分离、文本计划参照对齐，逐渐从 token 计划切换到 latent；global batch 16。
4. Interlat 压缩训练：固定已训练 Receiver，学习 21 步压缩 Sender，以任务 CE、信息增益加权分布蒸馏和表示一致性训练；global batch 4。

全部使用 microbatch 1 和梯度累积、梯度检查点、FP32 可训练参数/BF16 autocast、保存激活到 CPU 的选项；不是 LoRA 替代版。每 25 个 optimizer steps 保存模型、优化器和随机状态。完成后导出 BF16 推理权重，默认删除本阶段庞大的最终优化器断点以节省磁盘。

**300 步、候选数、辅助数据量、部分权重和调度规则是可修改的迁移默认值，不是论文公开的完整训练配方，也不保证收敛。** 详见 `configs/default.json`。首次配置应在运行前确定；修改配置后需使用新的 `LATEN_STORE`，不能混用已封存结果。训练缓存保留不足 32 个主样本会明确停止，不会拿随机压缩器继续产出“基线分数”。

单卡按照“4B 三个免训练基线 → 4B 数据生成/训练/评测 → 8B 同样流程”执行，所有重型阶段独立进程串行退出。两种规模共 2,400 个 optimizer steps，但每步计算成本不同；尚无 B200 实测速度，不能保证若干小时内完成整套任务。

## 新增：我们自己的 LATEN Benchmark

默认一键流程已经包含 **五种方法 × Qwen3-4B/8B × 648 条条件**，新增 10 个评测单元；不需要单独准备 Benchmark 数据。冻结数据直接放在 `vendor/laten/data/`，逐文件校验原始 SHA256，不在目标服务器重新生成题目。

- 162 个基础任务、648 个 BASE/CF 条件、486 个配对；dev 108 条、test 540 条，保留原划分。
- 原 G4/G5/G6 角色顺序、I3/I6/I9 信息量、LOOKUP/RULE/DERIVE 策略和角色私有事实边界均保留。
- Receiver 贪心思考上限 2048；正确闭合后，逐位受约束输出事实向量。原 prompt 中“尽量少于 384 tokens”的软要求不改动。
- 同时报告完整向量、逐事实、确定性动作、冻结模型 A–H 动作诊断、目标 CF、非目标 CF、整对向量、截断数，以及“动作正确时的事实错误率”。
- 使用原效率聚合器报告整条路径平均/p50/p95 秒数、总时间、任务吞吐率、正确任务吞吐率、推理/答案/强制格式 Token、latent 位置和逻辑通信字节；额外模型动作诊断独立计时。
- 本 Benchmark 不参与新增训练。test 是项目已使用过的诊断划分，不称为全新未见测试集。

方法保留各自接口：LatentMAS/H2O 使用连续 KV；Hidden 使用“全部历史原始 prompt embeddings + 当前 10 步 latent”；LatCom 在每个接收边用已训练压缩器把上一角色轨迹转为 64 个槽位；Interlat 使用 21 步已训练压缩 Sender 和接收 Adapter，最后使用训练后的 Receiver LM。后两者是明确标记的串行多跳迁移，不能称为作者官方多跳实现。为了延续本 Benchmark 的 10 步 Sender 条件，这里使用 10 步，七个通用数据集仍为 40 步；各方法的消息字节数不是等额预算。详见 [协议](docs/LATEN_BENCHMARK_PROTOCOL.md)。

已有上一版本结果与完整训练权重时，先等正在运行的旧任务结束，然后：

```bash
git pull --ff-only
bash run.sh semantic-all
```

继续设置原有 `LATEN_STORE`。该命令只补测 LATEN，不重新跑七个通用数据集、不训练新权重。仅接受与原版本 `1f98984` 配置吻合且权重哈希正确的完整导出；不在新代码下恢复旧优化器断点。若旧版未完成训练，先在旧版完成；要用新版重新运行整套实验则指定新的 `LATEN_STORE`，避免旧源码身份与新源码混用。

可单独测一组，或跑独立的 G4/G5/G6 接口 smoke：

```bash
bash run.sh semantic --family 4b --method latentmas_hidden
bash run.sh semantic-smoke --family 4b --method latcom
```

正式语义结果在 `storage/runs/<规模>/semantic/<方法>/results.jsonl`；smoke 存入 `semantic_smoke/`，不混入正式分母。通用任务大表仍为 `summary.csv`，语义完整大表为 `semantic_summary.csv`，原始分层指标在 `semantic_summary.json`，全套 80 个单元完成情况在 `suite_summary.json`。`SEMANTIC_COMPLETE.json` 表示仅新增 10 个单元完成，`COMPLETE.json` 表示默认全流程完成。

## 下载与结果位置

| 内容 | 默认路径 |
|---|---|
| Python 环境 | `.venv/` |
| Qwen3-4B / Qwen3-8B | `storage/models/4b/`、`storage/models/8b/` |
| HF 原始数据文件 | `storage/raw/` |
| 固定评测题目、训练候选与分母清单 | `storage/data/` |
| 固定 commit 的参考仓库 | `storage/upstream/LatentMAS/`、`storage/upstream/Interlat/` |
| HF / pip / torch / Triton 缓存 | `storage/cache/` |
| 训练缓存 | `storage/runs/<4b或8b>/training_cache/` |
| 四阶段训练日志/权重 | `storage/runs/<规模>/training/<阶段>/` |
| 原始回答与逐题评分 | `storage/runs/<规模>/evaluation/<方法>/` |
| 三个最终表格 | `storage/reports/summary.csv`、`summary.md`、`summary.json` |
| 全部完成标记 | `storage/COMPLETE.json` |
| 最近一次失败 | `storage/last_error.json` |

设置 `LATEN_STORE` 后替换上表中的 `storage/` 前缀。Docker 镜像由宿主 Docker daemon 管理，其文件位置遵循目标服务器已有 Docker 配置；保存使用的不可变 image ID，不自动修改 daemon 存储位置。

## 查询、续跑与分步执行

查看两种模型的大表、每个单元的已完成题数和训练阶段进度：

```bash
bash run.sh status
```

再次执行 `bash run.sh` 即可续跑：已评分题目跳过；已经生成但评分受阻的题目沿用原回答；训练从保存的模型、优化器、随机状态恢复。配置、数据或源码不一致会拒绝混入旧结果。Ctrl+C/SIGTERM 请求在当前题目或 optimizer step 完成后停下；不要反复强杀正在保存的进程。断电/强杀最多丢失尚未保存的训练步数，损坏的末尾 JSONL 会备份后修复。

下面这些命令用于排查或分段调度，**一次只运行一个 GPU 命令**：

```bash
bash run.sh validate                          # 无需 GPU/下载，16 项本地协议检查
bash run.sh bootstrap                         # 目标 B200 环境检查与安装
bash run.sh download                          # 固定版本模型、数据、源码与评分容器
bash run.sh smoke --family 4b                  # 三个免训练方法的真实模型接口检查
bash run.sh all --family 4b                    # 只运行 4B；下载阶段仍准备两种规模
bash run.sh collect --family 4b
bash run.sh train --family 4b --stage latcom_stage1
bash run.sh train --family 4b --stage latcom_stage2
bash run.sh evaluate --family 4b --method latcom
bash run.sh report
```

Interlat 的阶段键为 `interlat_receiver`、`interlat_compression`。可用 `--config configs/your_config.json` 指定复制后的配置文件。默认全矩阵配置下，只完成 4B 时不会生成完整 80 单元的完成标记。

数值/梯度检查需要已装 torch/transformers 的 Python，但不使用 GPU：

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_tiny*.py' -v
```

## 如何读分数、耗时和 Token

`summary.csv` 每行是模型 × 方法 × 数据集，包括正确数、完成数、固定分母、准确率、整条任务路径的平均/中位耗时、总推理秒数与文本 Token 数等。未完成的行明确标记 `incomplete`，只有七项全齐时才给对应方法的 macro accuracy。

- **整条路径耗时**包括三名 Sender、压缩/Adapter、最终 Receiver 的 prefill 和解码；CUDA 同步计时。模型加载、环境下载、训练、代码评分不混入推理耗时。
- **生成文本 Token**、**latent 位置数**、**prompt Token**、**模型 prefill 位置数**分别统计，不能直接相加称为文本 Token。
- **通信量**按每条逻辑 Agent 边累加，包含被转发的 KV 历史、连续消息和压缩器输出；保留位置数和实际 dtype 的字节数。它表示逻辑载荷，不是单机上的网络传输测量。
- 每条回答保留提示词、输出 token IDs、停止原因、每个角色耗时、消息/缓存位置数与评分状态，能够回查异常。
- OOM、下载错误、数据损坏、评分容器启动失败属于运行错误并停止当前阶段；不会冒充错误答案进入准确率。模型正常输出错误、格式解析失败和候选代码的内部执行超时计为失败。

## 当前验证边界

准备阶段通过 **16 项静态/数据/评分检查 + 12 项小型 Qwen3 CPU 数值/梯度检查**，覆盖五种通信路径、H2O 裁剪、对齐向量、冻结 Receiver 的梯度传播和训练/推理自回归一致性。未在准备服务器下载完整模型或启动 GPU 训练，**尚无正式 B200 跑分、端到端 8B 显存峰值或收敛结论**。验证记录见 `evidence/validation.json`。所有大模型检验将由目标服务器脚本实际执行，失败会保留具体记录。
