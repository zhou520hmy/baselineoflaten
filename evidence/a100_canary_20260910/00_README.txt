A100 本地工程测试（2026-09-10，北京时间）

读取顺序：本文件 -> 01_INDEX.txt -> real/00_README.txt -> real/01_INDEX.txt -> real/02_canary.p001.txt -> tiny/ 对应文件。
全部为统计 TXT，每个小于 90,000 字节，无推理文本、预测答案或代码输出。

结论：15 个检查通过；本轮未发现需要修改生产计算代码的运行错误。
真实模型：已有 Qwen3-4B，A100-PCIE-40GB，物理 GPU 2。
软件：PyTorch 2.5.1+cu121 / Transformers 4.57.6。
B200 正式环境仍固定 PyTorch 2.7.1+cu128；正式硬件门槛未放宽。

真实 4B 检查（11 项）：
1. 加载与 realignment：矩阵 2560x2560，数值有限。
2. collect：4 个人工构造候选完整扫描，保留 4 个资产。
   其中 2 条主候选：LatCom 合格 1 条、Interlat 合格 2 条；另 2 条为辅助候选。
   两种方法的正式 minimum_retained=32 保持不变，所以正式 method_ready=False 是预期结果。
   再次 collect 时没有模型加载，记录哈希未变。
3-7. 五方法通用通信接口：均完成真实 4B 前向生成，3 名 Sender。
   LatCom/Interlat 使用未训练模块且共享冻结预训练 backbone，只检验接口，不能作为基线效果或完整加载峰值。
   通用输出 cap=48；源 latent 深度与槽位数沿用默认配置。
8-10. 三个免训练方法：各完成一个相同 G4/I3 的 LATEN dev BASE 样例，实际 greedy cap=2048。
   execution_status=success 是接口状态，不宣称正确率，也不是完整 CF 配对测评。
11. 真实 ReceiverAdapter：一次 AdamW 更新；loss=4.466794，裁剪前 grad_norm=391.473206，
    output_scale 参数变化约 9.99e-5；冻结基座无参数梯度。
    这是 adapter-only 检查，不是正式全模型训练。

同架构随机小模型 GPU 检查（4 项）：
Qwen3 hidden_size=32，2 层，保留真实 tokenizer/vocab；2 个人工构造训练资产。
LatCom stage1/stage2、Interlat receiver/compression，各调用原 train_stage 完成 2 optimizer steps。
全参数梯度与原目标函数、AdamW、梯度检查点、CPU saved activations、权重导出和重新加载均实际执行。
第 1 阶段在第 1 步触发实际暂停，保存优化器/RNG；随后恢复至第 2 步。
四阶段完成后再次调用时，均不重新加载模型，导出哈希不变。
缩小模型的梯度、保存和恢复通过，不能证明 4B/8B 全参数训练显存峰值或最终收敛。

本次使用 GPU 2，未停止其他 GPU 上的任务；结束后 GPU 2 已释放。
未下载新模型，未改正式采样、loss、collect 门槛或 B200 硬件检查。
未验证：真实 HotpotQA/MuSiQue 的最终保留数、B200 双进程峰值、4B/8B 完整训练、代码 Docker 测试、全量跑分。
复现脚本：scripts/run_a100_canary.py。运行步骤见 README_ZH.md 的 A100 工程检查段落。
