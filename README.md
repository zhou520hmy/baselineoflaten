# LATEN Baselines — Single B200

单张 B200 的可迁移基线训练与评测包：**Interlat、LatentMAS、LatentMAS-H2O、LatentMAS-Hidden、LatCom**；模型为 **Qwen3-4B / Qwen3-8B**。

覆盖 GSM8K、ARC-E、ARC-C、MedQA、MBPP+、HumanEval+、GPQA-Diamond，并纳入项目 **LATEN 语义 Benchmark（648 条、486 对 BASE/CF）**，共 **80 个评测单元**。

## 一键运行

将本仓库克隆到目标服务器，进入仓库目录后执行：

```bash
bash run.sh
```

自定义大文件下载位置：

```bash
GPU_ID=0 LATEN_STORE=/data/laten_baselines bash run.sh
```

目标环境：空闲 B200、Python 3.10–3.12、Git、Docker、至少 256 GB 主机内存和 500 GiB 空闲磁盘。下载、训练和评测串行运行，支持断点恢复。

- [中文完整说明](README_ZH.md)：运行、存储、续跑和结果读取。
- [协议与实现差异](docs/PROTOCOL.md)：来源、训练目标、指标与复现边界。
- [默认配置](configs/default.json)：模型版本、题数、步数和预算。
- [准备阶段验证记录](evidence/validation.json)：CPU 检查结果。

## 新增 LATEN Benchmark

默认 `bash run.sh` 会自动测试五种方法在两个模型规模上的 LATEN 表现。保留原数据、角色提示词、G4/G5/G6、I3/I6/I9 和 2048-token 贪心推理/逐位读出协议；输出事实向量、逐事实、动作、CF 配对及完整任务路径效率指标。结果见 `storage/reports/semantic_summary.csv` 和 `semantic_summary.md`，全套完成状态见 `suite_summary.json`。

已有旧版完整训练权重时，更新后仅补测：

```bash
bash run.sh semantic-all
```

此命令不重新训练，缺少完整权重时会明确报错。必须继续使用原来的 `LATEN_STORE`。旧版尚在运行时先等它完成，再更新代码。详细接口差异和指标说明见 [LATEN 协议](docs/LATEN_BENCHMARK_PROTOCOL.md)。

## 实验边界

Interlat 和 LatCom 是按论文设计实现的非官方 Qwen3 训练迁移，不是作者原始 checkpoint 的完全复现。MedQA 使用 300 题子集；MBPP+ 使用固定版本 378 题扩展测试。此包尚未完成正式 B200 全模型训练与跑分。

模型、缓存、日志、运行结果和本地凭据均不纳入 Git 版本控制。第三方源码/数据来源保留在协议和 evidence 中；vendored LatentMAS 源码附带原许可证。
