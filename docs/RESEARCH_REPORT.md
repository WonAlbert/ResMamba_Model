# ResMamba 工程现状纪要

**日期：** 2026-08-31  
**范围：** `/root/autodl-tmp/ResMamba_Signal_Model` 当前可运行实现。  
**权威文档：** [`README.md`](../README.md)、[`AGENTS.md`](../AGENTS.md)。冲突时以可运行代码与门控指标为准。

> 2026-08-18 旧版研究报告中的 UTI v2 / `modulation`·`emitter` 双任务主路径叙述已过时；**持续学习（continual）能力保留**。设计论文外链见文末。

---

## 1. 当前管线（一句话）

联合能量 RevIN（`joint_energy`）→ task/family 硬路由 MoE Tokenizer/Encoder → Encoder 池化 **`z_enc`** → SharedDecoder（预训练 MAE）→ 下游六任务（`z_enc` → TaskAdapter → head）。**不**构建 UTI（`build_task_interface=false`）。

下游六任务：`ld_intrapulse` / `ld_model` / `tx_modulation` / `ld_clustering` / `tx_clustering` / `prediction`。

训练入口：`scripts/train.py`，阶段 `pretrain` → `stage2` → `stage3` → `joint` →（可选）`continual`。旧 `configs/downstream.yaml` 已移除（由 stage2/3/joint 取代）。

---

## 2. 已落地 vs 未宣称

| 项 | 状态 |
|---|---|
| 六任务数据池与 H5 协议（train/test/val） | 已实现；见 `configs/datasets.yaml`、[`docs/data_preprocess.md`](data_preprocess.md) |
| MAE + 物理/结构 + VICReg（`z_enc` / `h_enc`） | 已实现；见 `configs/pretrain.yaml` |
| 同质 H5 batch + sequence packing | 已实现（`HomogeneousTokenBudgetSampler`） |
| Stage2 LP / Stage3 Hybrid-LoRA+ / Joint SharedTaskAdapter | 已实现 |
| 持续学习：共享 LoRA、置信蒸馏、原型锚、未知吸收 | 已实现；`configs/continual.yaml` + `training/continual.py`；`--stage continual` |
| Tiny + synthetic 全流程冒烟 | 实现验收口径；非正式 acc 门控 |
| 正式宽度 72h / 多 seed / unified 分列表 | **未跑满**；见 [`docs/sota_gate.md`](sota_gate.md) |
| 同协议公开基线对照数字 | **无**；不得宣称 SOTA |

---

## 3. 架构要点（勿与旧 UTI 叙述混淆）

- **Encoder**：`M-M-M-M-M-T`；预训练 `encode_visible_only`；默认 **GatingPool** → `z_enc` / `h_enc`（分类身份；`z` / `z_general` 仅为别名）。
- **Decoder**：token 重建；`[DEC]` + AttnPool → `z_recon`（重建/物理 readout，不作分类身份）。
- **MoE**：三路专家硬路由（`ld_intrapulse` / `ld_model` / `tx_modulation`），非内容 gate。
- **归一化**：Loader `iq_normalize: none`；模型内 `joint_energy`；`amp_aux` 不进 `z_enc`。
- **持续学习**：冻结骨干；只训共享低秩 adapter / 原型 / 任务头；`continual_sessions` 分段；会话边界刷新 teacher；`absorb_unknown` 写入低计数原型槽。不建 per-task LoRA 池（`shared_lora: true`）。

细节与命令见 README；改模型时对照 AGENTS「架构要点」。

---

## 4. 验收口径

1. `pytest -q` 与 tiny synthetic 各 stage 冒烟（含可选 `continual`；见 README **Tiny 冒烟**）。
2. 正式 acc 门控阈值待预训练跑满后写入 `docs/sota_gate.md`；勿再引用旧 `modulation` / `emitter` 阈值。

---

## 5. 外链（设计依据，非本仓库实验结果）

- [ROSE, ICML 2025](https://proceedings.mlr.press/v267/wang25ci.html)
- [SEMPO, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/ecfb69ce6be017deb5a926c2718f6bc1-Paper-Conference.pdf)
- [MambaSSL, ICLR 2026 Poster](https://openreview.net/forum?id=YDl4vqQqGP)
- [TimePerceiver, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/c6c682ba9bd8839104f2a82901da4109-Abstract-Conference.html)
- [RoMAE, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/c2626ef6cdaaa6a18927832820079e1d-Abstract-Conference.html)
- [EMTC, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/39777)
- [SD-LoRA, ICLR 2025 Oral](https://openreview.net/forum?id=5U1rlpX68A)
- [CMD-PCL, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/39568)
