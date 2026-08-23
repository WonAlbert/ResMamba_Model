# ResMamba Signal Model

任务无关的射频 I/Q 基础模型：联合能量 RevIN → 内容路由 MoE Tokenizer/Encoder → `z_enc` → 预训练 Decoder（MAE）→ 下游六任务独立头（`z_enc` → TaskAdapter → head）。

主类：`SignalFoundationModel` / `SignalModelConfig`（`resmamba_signal_model`）。

下游任务：`ld_intrapulse` / `ld_model` / `tx_modulation` / `ld_clustering` / `tx_clustering` / `prediction`。

## 安装

```bash
cd /root/autodl-tmp/ResMamba_Signal_Model
source scripts/env.sh
pip install -r requirements.txt
```

`won` 环境需匹配 PyTorch / `mamba-ssm`。CI 与 tiny 配置允许 Mamba fallback。

```bash
export PYTHONPATH="$(pwd):$PYTHONPATH"
export RFDATA_ROOT="/root/autodl-tmp/ResMamba_Signal_Model/dataset"
```

## 数据准备

```bash
python scripts/prepare_datasets.py --output "${RFDATA_ROOT}" --datasets all
```

**划分协议**（H5 后缀；白名单见 `configs/datasets.yaml`）：

| 后缀 | 用途 |
|------|------|
| `*_train.h5` | 阶段一无标签 MAE 预训练 |
| `*_test.h5` | 阶段二 / 三有标签训练 |
| `*_val.h5` | 全阶段验证、早停与推理评估 |

缺 `*_test` 的数据集（如 `radar_mod15`）可从 `*_train` 按类切 20% 补 test，再刷新 pool：

```bash
python scripts/prepare_datasets.py --output "${RFDATA_ROOT}" \
  --ensure-missing-test-splits --rebuild-task-pools
```

仅重映射 pool、不重做 rebalance：

```bash
python scripts/prepare_datasets.py --output "${RFDATA_ROOT}" --rebuild-task-pools
```

Loader 保持 `iq_normalize: none`；幅度由模型内 **联合能量 RevIN** 处理（I/Q 共享 Winsorized RMS，非逐通道 z-score）。绝对功率 / RSSI 经 `amp_aux` 旁路进 Decoder FiLM 与物理读出，**不**写入 `z_enc` 或分类头。

## 快速验证

```bash
python scripts/smoke_forward.py
python scripts/count_params.py --model-config configs/model.yaml
pytest -q
```

## 架构要点

- **Encoder**：`M-M-M-M-M-T`。预训练 `encode_visible_only`：只对可见 token 编码并 **GatingPool**（默认；可切 `attn_pool`）→ **`z_enc` / `h_enc`**（`z_general` / `z` 别名；默认 L2 归一化）。下游无 mask 时对全有效 patch 池化。
- **Decoder**：1 层 `DecoderBlock`；token 通路重建；`[DEC]` + AttnPool → **`z_recon`**（重建 / 物理 readout，不作分类身份）。
- **下游**：冻结 `z_enc` → TaskAdapter → 六任务独立头；联合仅 `SharedTaskAdapter`。
- **MoE**：Tokenizer / Encoder / Decoder 共用三路专家（`ld_intrapulse` / `ld_model` / `tx_modulation`）；`phase_plugin: true`。
- **归一化**：`revin_scale_mode: joint_energy`（抗峰 Winsorize + 逐时刻 PAPR 钳制）；`amp_aux` 保留 RSSI 供 Decoder / 物理损失，不进分类表征。
- **物理约束**：patch 级相对特征 + `log_power_rel + log_scale` 还原绝对功率；软 SmoothL1 + 硬能量投影。分类在归一化空间，重建在 denorm 后。
- **变长**：`sequence_packing=true`；预训练 **`HomogeneousTokenBudgetSampler`**（每 batch 单一 H5）+ `combine_then_pack: false`；其它阶段用 `TokenBudgetSampler`；`L < 16` 报错；`L > 8192` 重叠切块；`p_trunc=0.3`。
- **多域**：Domain GRL 仅约束 UTI 不变视图（默认 `semantic`）；预训练 `domain: 0`。预训练 collate 防火墙：标签 / `dataset_id` / 采集元数据不进 forward。

预训练 MAE + MoE 负载均衡；`HomogeneousTokenBudgetSampler`（族配额 + sqrt(N)）+ `combine_then_pack: false`。

## 训练

唯一入口：`scripts/train.py`。阶段：`pretrain` → `stage2` → `stage3`（六任务之一）→ `joint`。

```bash
python scripts/train.py --stage pretrain --config configs/pretrain.yaml --profile tiny --synthetic
python scripts/train.py --stage stage2 --config configs/stage2.yaml --profile tiny --synthetic
python scripts/train.py --stage stage3 --task tx_modulation --profile tiny --synthetic
python scripts/train.py --stage joint --config configs/joint.yaml --profile tiny --synthetic
```

## 验收

**Tiny 全流程冒烟**（见 `AGENTS.md`）：`smoke_forward` + 各 stage `--profile tiny --synthetic` + `pytest -q`。正式 acc 门控待预训练跑满后更新。

## 配置

| 文件 | 作用 |
|------|------|
| `configs/model.yaml` | 正式宽度（`d_model=640`，`revin_scale_mode: joint_energy`，需 Mamba kernel） |
| `configs/model_tiny.yaml` | CI / CPU |
| `configs/pretrain.yaml` | 预训练；`homogeneous_batch` / MAE 策略 / SSL 损失权重 |
| `configs/stage2.yaml` | 冻结骨干 LP 探测 |
| `configs/stage3.yaml` | 单任务 Hybrid-LoRA+ |
| `configs/joint.yaml` | 联合 PEFT |
| `configs/downstream.yaml` | 旧下游入口 |
| `configs/datasets.yaml` | 数据池白名单（pretrain = 各阶段 train 并集） |
| `configs/val_subset.yaml` | 验证子集 |

更完整的 Agent 约定、代码树与门控细节见 [`AGENTS.md`](AGENTS.md)。



！！！！！！！！！！！
idea：
mamba本质为线性RNN
能否将decoder的某一层的中间状态反馈到encoder上
比如第n步的decoder中间状态反馈到第n+1步的encoder上
！！！！！！！！！！！
