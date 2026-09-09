# 单张 B200：训练串行、推理并行的基线评测包

默认测试 Interlat、LatentMAS、LatentMAS-H2O、LatentMAS-Hidden、LatCom，模型为 Qwen3-4B 和 Qwen3-8B。七个通用数据集各缩减到约 10%，**我们自己的 LATEN Benchmark 全量保留**。合计 80 个评测单元、12,390 次任务推理（旧全量矩阵为 65,550 次）。训练数据生成和四阶段训练预算没有缩减。

本包是统一协议迁移实现。Interlat/LatCom 使用本包的论文设计迁移训练，标记 `paper_derived_port`，不是作者原始 checkpoint；也不使用我们工程的 V6 权重。方法、损失和来源差异见 [PROTOCOL.md](docs/PROTOCOL.md)。

## 一键启动

目标服务器需要 Linux、Python 3.10–3.12（含 venv）、Git、可运行的 Docker、空闲 B200（至少 170 GiB 显存）、至少 256 GB 主机内存及首次运行 500 GiB 空闲磁盘。建议 1 TB 磁盘。固定使用 PyTorch 2.7.1 + CUDA 12.8，检查 Blackwell sm_100 和 BF16；不使用 sudo 或修改系统驱动。

```bash
bash run.sh
```

指定物理 GPU 和所有大文件位置：

```bash
GPU_ID=0 LATEN_STORE=/data/laten_baselines bash run.sh
```

没有设置 `LATEN_STORE` 时使用本目录 `storage/`；之后续跑和查询必须用同一路径。建议用 tmux，或者：

```bash
nohup bash run.sh > run.log 2>&1 &
```

命令会安装环境、下载固定版本模型/数据、建立固定抽样清单、构建代码评分容器、校验选中的 54 条代码参考解、运行接口检查、训练、推理和汇总。完整测试源数据仍保留，用于校验和训练排重。网络需要访问 Hugging Face、GitHub、PyTorch wheel 站和 Docker 镜像源；按目标机需要设置代理和 HF_TOKEN，仓库不包含凭据。

## 评测数量和抽样

| 数据集 | 原始可用题数 | 当前每个方法/模型题数 | 指标 | Receiver 输出上限 |
|---|---:|---:|---|---:|
| GSM8K | 1,319 | 132 | 数值准确率 | 2,048 |
| ARC-E | 2,376 | 238 | 选择题准确率 | 2,048 |
| ARC-C | 1,172 | 117 | 选择题准确率 | 2,048 |
| MedQA | 300 | 30 | 选择题准确率 | 4,096 |
| MBPP+ | 378 | 38 | 扩展测试 pass@1 | 4,096 |
| HumanEval+ | 164 | 16 | 扩展测试 pass@1 | 4,096 |
| GPQA-Diamond | 198 | 20 | 选择题准确率 | 8,192 |
| LATEN | 648 | **648** | 向量/事实/动作/CF 指标 | 2,048 |

七项通用任务每组 591 题，加 LATEN 为 1,239 题，五方法 × 两模型共 12,390 次。MedQA 源数据是已有 300 题子集，不是整个官方测试集。代码任务使用固定 HF EvalPlus parquet 扩展测试，不重新生成最新版测试。

通用任务使用固定种子 42，按复杂度代理排序分为三等分，每层抽取近似相同题数，避免仅保留短题。非代码任务代理是公开题目字符长度；代码任务优先使用参考代码中的分支、循环、推导式等 AST 节点数，再按题长排序。**这些是分层代理，不是经过验证的难度等级**；ARC-C、GPQA-Diamond 仍按同一比例测试。选择不使用任何方法的正确率、生成长度或运行时间。

选中 ID、原划分/文件哈希、分层数量和样本原文写入 `storage/data/sampled10/`。两种模型、五种方法共享相同清单，续跑重新校验，不能因 OOM、解析失败或答错而换题。抽样结果不能称为完整通用测试集分数。

## 单卡并行如何工作

每次只运行一个方法/模型的评测阶段，阶段内默认两个独立进程共享同一张 B200，各自处理不重叠分片。进程之间不共享模型对象、KV 缓存或生成随机状态。通用题按固定次序分片，LATEN 按完整 BASE/CF 任务组分片，各 324 条、243 对。逐题种子不依赖 worker ID。

- 每个并行进程的 PyTorch 显存分配器上限为总显存 45%；模型加载和 realignment 初始化通过锁串行完成，推理并行。
- 显存不足的分片退出后，等待本阶段其他进程结束，再以单进程、90% 分配器上限继续。已生成或已评分结果不会重做。
- 该上限不覆盖 CUDA 上下文和所有外部库分配，不保证任何长输入都不 OOM。串行仍不足则明确停止，保留记录；不降低 token 上限或删除题目。
- 每个 worker 有独立原始输出、日志和状态；协调器原子合并结果。任务完整性和结果身份校验保留。
- 训练数据生成、LatCom/Interlat 训练和模型规模之间仍串行，避免与推理争用资源。

默认配置为 `inference_parallel.workers=2`。需要从一开始串行时可复制配置并设为 1，使用新的运行目录；中途直接改配置会触发身份校验。正常 OOM 自动降并发不需要改配置。不要另外手工启动多个 GPU 命令；由脚本内部调度并行。

## 训练与通信协议

主训练数据仍是 HotpotQA distractor train / MuSiQue-Ans train：扫描 1,024 个候选，最多保留 512 个主样本，辅助 GSM8K train、排除测试 task ID 的 MBPP train 各 32 条。过滤问题独立可解或完整 latent 路径仍不能回答的主样本，最少保留 32 条，否则明确停止。rationale 利用支持事实和 gold answer 构造，没有额外独立模型或人工验证。LATEN 不进入训练。

每种模型四阶段各 300 optimizer steps：LatCom 阶段一和二 global batch 64；Interlat 接收端 global batch 16；Interlat 压缩端 global batch 4。共 2,400 optimizer steps。保持原任务 CE、对比分离、教师分布和表示对齐等目标，具体公式见方法协议。microbatch 1、梯度累积/检查点、FP32 可训练参数和 BF16 autocast，默认 CPU 保存激活；每 25 步保存断点，完整导出 BF16 权重后默认删除最终优化器断点。不是 LoRA 替代实现，不保证默认预算收敛。

通用任务仍使用三名 Sender，LatentMAS/Hidden/H2O/LatCom 的源轨迹 40 步；Interlat 每源 21 步。LatCom 压缩 64 个槽位，H2O 每源每头保留 64 个 prompt 位置。角色 prompt、输出上限、温度 0.6 / top-p 0.95 / top-k 0、逐题随机种子及评分器均保留。

## LATEN Benchmark 保持全量

冻结数据原样放在 `vendor/laten/data/`，逐文件核对 SHA256。162 个任务、648 个条件、486 个 BASE/CF 配对；dev 108 条、test 540 条。G4/G5/G6、I3/I6/I9、LOOKUP/RULE/DERIVE 和各角色私有事实边界均保留。test 是已暴露的诊断划分。

Sender 轨迹深度仍为 10（Interlat 使用其训练后的 21 步接口），Receiver 贪心推理最多 2048 tokens，闭合后受约束逐位读出事实向量。继续报告完整向量、逐事实、确定性动作、模型 A–H 动作诊断、目标/非目标 CF、整对向量、截断和动作正确条件下的事实错误率。模型动作诊断另行计时。

LatentMAS/H2O 传递历史 KV；Hidden 传递所有历史原始 prompt embeddings + 当前角色 10 步 latent；LatCom 在各接收边压缩为 64 槽位；Interlat 使用已训练压缩 Sender、接收 Adapter 和最终 Receiver LM。后两者是明确标记的非官方多跳迁移。细节见 [LATEN 协议](docs/LATEN_BENCHMARK_PROTOCOL.md)。

## 更新与旧权重复用

旧任务仍在运行时先等它结束。已有完整训练权重时更新代码并只测评：

```bash
git pull --ff-only
LATEN_STORE=/data/laten_baselines bash run.sh eval-all
```

`eval-all` 不训练，使用已完成的导出；接受已知旧版 `1f98984`、`4af7b41` 的严格配置/身份匹配和权重 SHA 校验。缺少完整导出会报错。旧优化器断点不在新代码下恢复；要重新训练整套使用新的 `LATEN_STORE`。

只运行全量 LATEN：`bash run.sh semantic-all`。单个方法：

```bash
bash run.sh evaluate --family 4b --method latcom
bash run.sh semantic --family 4b --method latentmas_hidden
bash run.sh semantic-smoke --family 4b --method latcom
```

新通用结果在 `runs/<模型>/evaluation_sample10/<方法>/`，全量 LATEN 新并行结果在 `runs/<模型>/semantic_parallel/<方法>/`。旧 `evaluation/`、`semantic/` 原始结果不覆盖；`reports/` 更新为当前配置汇总。分片日志在各方法目录的 `logs/`，分片原始数据在 `shards/0/`、`shards/1/`，根目录 `results.jsonl` 是协调器合并后的当前结果。

## Token、时间和结果读取

`bash run.sh status` 输出 80 单元完成情况。通用表为 `reports/summary.csv`，语义表为 `reports/semantic_summary.csv`，总体为 `reports/suite_summary.json`。两个表均保留逐任务路径的 Token 及其他效率字段；LATEN 分层详情在 `semantic_summary.json`。

生成文本 Token、latent 位置、prompt Token、模型 prefill、逻辑通信位置/字节分别统计，覆盖整条 Agent 路径，不能混称为文本 Token。原始 token IDs、停止原因和通信记录保留。并行不会改变计数定义，但浮点运行环境不保证跨配置输出逐 bit 一致。

耗时只作有争用条件下的诊断，不据此宣称独占 GPU 速度优劣；多个任务耗时相加是重叠任务的累计秒数，不是整阶段历时。各阶段 `scheduler_wall.jsonl` 额外保存包含加载、评分和重试的协调器历时。每行 `execution` 标记并发上限、分片和重试模式。OOM/下载/容器基础设施错误保持未评分，正常错误答案、代码测试不通过、原生推理截断按既定规则计分。

重复 `bash run.sh` 续跑相同新版配置；已评分跳过、已生成沿用，训练从合法断点恢复。模型在 `storage/models/`，缓存 `storage/cache/`，数据 `storage/data/`，训练 `storage/runs/<模型>/training/`；环境 `.venv/`。Docker 镜像按宿主 daemon 的存储位置管理。

## 本地验证边界

```bash
bash run.sh validate
.venv/bin/python -m unittest discover -s tests -p 'test_tiny*.py' -v
```

本地验证覆盖分层抽样、完整 BASE/CF 分片、真实 CPU 子进程并发、OOM 调度重试夹具、合并/断点/身份校验、原评分协议，以及小型 Qwen3 CPU 数值和梯度。详细结果见 `evidence/validation.json`。**没有在准备服务器下载完整模型或启动本套 GPU 训练，尚无目标 B200 的并行显存峰值、加速倍数或正式跑分。**
