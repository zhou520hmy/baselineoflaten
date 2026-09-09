# LATEN Baselines — Single B200

单张 B200 上运行 **Interlat、LatentMAS、LatentMAS-H2O、LatentMAS-Hidden、LatCom**，模型为 **Qwen3-4B / Qwen3-8B**。

**当前默认：推理双进程并行，七个通用数据集分层抽样约 10%，我们自己的 LATEN Benchmark 保持全量。** 共 80 个评测单元、12,390 次任务。训练和训练数据生成保持串行，训练预算不变。

```bash
bash run.sh
```

自定义大文件位置：

```bash
GPU_ID=0 LATEN_STORE=/data/laten_baselines bash run.sh
```

已有旧版完整训练权重，等旧任务结束再更新，仅运行新版测评：

```bash
git pull --ff-only
LATEN_STORE=/data/laten_baselines bash run.sh eval-all
```

目标环境：空闲 B200（至少 170 GiB）、Python 3.10–3.12、Git、Docker、至少 256 GB 主机内存和首次运行 500 GiB 空闲磁盘。脚本固定下载版本和路径，支持断点恢复；双进程显存不足时保留结果，自动串行重试受影响分片。

| 数据集 | 每个模型/方法的评测量 |
|---|---:|
| GSM8K | 132 / 1,319 |
| ARC-E | 238 / 2,376 |
| ARC-C | 117 / 1,172 |
| MedQA | 30 / 300 |
| MBPP+ | 38 / 378 |
| HumanEval+ | 16 / 164 |
| GPQA-Diamond | 20 / 198 |
| LATEN Benchmark | **648 / 648，486 对 BASE/CF** |

所有方法和模型使用同一固定样本。通用数据按题长、代码结构复杂度代理分为三层等额抽取，保留长题和复杂代码；代理不等于经验证的难度标签。抽样不使用模型得分。整条任务路径的 Token、latent、prefill 和通信量继续报告；耗时标记为并行争用下的诊断值，不作为独占显卡速度对比。

- [完整中文运行说明](README_ZH.md)
- [抽样与并行协议](docs/SAMPLED_PARALLEL_PROTOCOL.md)
- [五方法协议和迁移差异](docs/PROTOCOL.md)
- [全量 LATEN 协议](docs/LATEN_BENCHMARK_PROTOCOL.md)
- [默认配置](configs/default.json)
- [本地验证记录](evidence/validation.json)

Interlat 和 LatCom 是按论文设计实现的非官方 Qwen3 训练迁移，不是作者原始 checkpoint 的完全复现。MedQA 的来源本身是 300 题子集；代码使用固定 HF EvalPlus 扩展测试。尚无本包在目标 B200 上的正式训练和跑分结果。模型、凭据、缓存与运行结果不纳入 Git。
