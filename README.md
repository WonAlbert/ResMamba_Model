# ResMamba Signal Model

任务无关的射频 I/Q 基础模型：联合能量 RevIN → task/family 硬路由 MoE → `z_enc` → 预训练 Decoder（MAE）→ 下游六任务独立头（`z_enc` → TaskAdapter → head；无 UTI）。

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
- **MoE**：Tokenizer / Encoder / Decoder 共用三路专家（`ld_intrapulse` / `ld_model` / `tx_modulation`）；预训练按 H5 stem、下游按 task 硬路由；`phase_plugin: true`。
- **归一化**：`revin_scale_mode: joint_energy`（抗峰 Winsorize + 逐时刻 PAPR 钳制）；`amp_aux` 保留 RSSI 供 Decoder / 物理损失，不进分类表征。
- **物理约束**：patch 级相对特征 + `log_power_rel + log_scale` 还原绝对功率；软 SmoothL1 + 硬能量投影。分类在归一化空间，重建在 denorm 后。
- **变长**：`sequence_packing=true`；全阶段 **`HomogeneousTokenBudgetSampler`**（每 batch 单一 H5）+ `combine_then_pack: false`；`L < 16` 报错；`L > 8192` 重叠切块；`p_trunc=0.3`。
- **预训练**：`build_task_interface=false`（无 UTI）；`domain: 0`；collate 防火墙：标签 / `dataset_id` 不进 forward（`moe_route_stem` 仅用于 MoE 族路由）。

预训练 MAE + `z_enc`/token VICReg + 物理/结构损失；`HomogeneousTokenBudgetSampler`（族配额 + sqrt(N)）。

## 训练

唯一入口：`scripts/train.py`。正式宽度使用 `configs/model.yaml`（需 Mamba kernel）。数据根由 `RFDATA_ROOT` 或各 YAML 的 `rfdata_root` 指定（默认 `dataset/`）。全阶段 `homogeneous_batch: true`（`HomogeneousTokenBudgetSampler`，每 batch 单一 H5）；预训练另设 `combine_then_pack: false`。

阶段顺序：`pretrain` → `stage2` → `stage3`（六任务各训一次）→ `joint`。日志与 checkpoint：`runs/experiments/<run_name>/`。

| 阶段 | 配置 | 冻结策略 |
|------|------|----------|
| 一 预训练 | `configs/pretrain.yaml` | 全量骨干（无 UTI） |
| 二 LP 探测 | `configs/stage2.yaml` | 冻结骨干；训当前任务头（+ 可选 `z_enc` 探针）；截断反传 |
| 三 单任务适配 | `configs/stage3.yaml` `--task <name>` | Hybrid-LoRA+ + TaskAdapter + 该任务头 |
| 四 联合 | `configs/joint.yaml` | 各任务 LoRA/Adapter/头 + `SharedTaskAdapter` |

### 正式命令

```bash
# 阶段一：MAE 预训练
python scripts/train.py --stage pretrain --config configs/pretrain.yaml

# 阶段二：六任务 LP 探测（task_schedule 串行；ld_intrapulse 15 · ld_model 15 · tx_modulation 20 · ld_clustering 15 · tx_clustering 15 · prediction 5 epoch）
python scripts/train.py --stage stage2 --config configs/stage2.yaml \
  --init-from runs/experiments/<pretrain_run>/ckpts/best.ckpt

# 阶段三：单任务 Hybrid-LoRA+（每个任务独立 run；从 stage2 best 初始化）
python scripts/train.py --stage stage3 --config configs/stage3.yaml --task ld_intrapulse \
  --init-from runs/experiments/<stage2_run>/ckpts/best.ckpt
python scripts/train.py --stage stage3 --config configs/stage3.yaml --task ld_model \
  --init-from runs/experiments/<stage2_run>/ckpts/best.ckpt
python scripts/train.py --stage stage3 --config configs/stage3.yaml --task tx_modulation \
  --init-from runs/experiments/<stage2_run>/ckpts/best.ckpt
python scripts/train.py --stage stage3 --config configs/stage3.yaml --task ld_clustering \
  --init-from runs/experiments/<stage2_run>/ckpts/best.ckpt
python scripts/train.py --stage stage3 --config configs/stage3.yaml --task tx_clustering \
  --init-from runs/experiments/<stage2_run>/ckpts/best.ckpt
python scripts/train.py --stage stage3 --config configs/stage3.yaml --task prediction \
  --init-from runs/experiments/<stage2_run>/ckpts/best.ckpt

# 阶段四：联合 PEFT（stage2 骨干 + 各 stage3 specialist）
python scripts/train.py --stage joint --config configs/joint.yaml \
  --init-from runs/experiments/<stage2_run>/ckpts/best.ckpt \
  --adapter-dir ld_intrapulse=runs/experiments/stage3_ld_intrapulse_<ts> \
  --adapter-dir ld_model=runs/experiments/stage3_ld_model_<ts> \
  --adapter-dir tx_modulation=runs/experiments/stage3_tx_modulation_<ts> \
  --adapter-dir ld_clustering=runs/experiments/stage3_ld_clustering_<ts> \
  --adapter-dir tx_clustering=runs/experiments/stage3_tx_clustering_<ts> \
  --adapter-dir prediction=runs/experiments/stage3_prediction_<ts>
```

阶段二可用 `--start-task <name>` 从 `task_schedule` 中间任务续训。阶段三任务名：`ld_intrapulse` / `ld_model` / `tx_modulation` / `ld_clustering` / `tx_clustering` / `prediction`；各任务 checkpoint monitor 见 `configs/stage3.yaml` 的 `profiles`。

### Tiny 冒烟

实现验收（非正式 acc 门控）：各 stage 加 `--profile tiny --synthetic`。

```bash
python scripts/smoke_forward.py
python scripts/train.py --stage pretrain --config configs/pretrain.yaml --profile tiny --synthetic
python scripts/train.py --stage stage2 --config configs/stage2.yaml --profile tiny --synthetic
python scripts/train.py --stage stage3 --config configs/stage3.yaml --task ld_intrapulse --profile tiny --synthetic
python scripts/train.py --stage stage3 --config configs/stage3.yaml --task ld_model --profile tiny --synthetic
python scripts/train.py --stage stage3 --config configs/stage3.yaml --task tx_modulation --profile tiny --synthetic
python scripts/train.py --stage stage3 --config configs/stage3.yaml --task ld_clustering --profile tiny --synthetic
python scripts/train.py --stage stage3 --config configs/stage3.yaml --task tx_clustering --profile tiny --synthetic
python scripts/train.py --stage stage3 --config configs/stage3.yaml --task prediction --profile tiny --synthetic
python scripts/train.py --stage joint --config configs/joint.yaml --profile tiny --synthetic
pytest -q
```

## 验收

正式 acc 门控待预训练跑满后在 `docs/sota_gate.md` 更新。实现冒烟见上文 **Tiny 冒烟** 与 [`AGENTS.md`](AGENTS.md)。

`--profile tiny` 在各 stage YAML 的 `profiles.tiny` 中设 `num_workers: 0`（避免 CI/冒烟时 DataLoader 多进程 fork）；Lightning 可能在 sanity check 提示把 `num_workers` 提到约 CPU 核数，**可忽略**。正式训练勿加 `--profile tiny`；默认 `num_workers` 见各 stage YAML（pretrain/stage2 多为 `20`）。

## 配置

| 文件 | 作用 |
|------|------|
| `configs/model.yaml` | 正式宽度（`d_model=640`，`revin_scale_mode: joint_energy`，需 Mamba kernel） |
| `configs/model_tiny.yaml` | CI / CPU |
| `configs/pretrain.yaml` | 预训练；`homogeneous_batch` / MAE 策略 / SSL 损失权重 |
| `configs/stage2.yaml` | 冻结骨干 LP 探测 |
| `configs/stage3.yaml` | 单任务 Hybrid-LoRA+ |
| `configs/joint.yaml` | 联合 PEFT |
| `configs/datasets.yaml` | 数据池白名单（pretrain = 各阶段 train 并集） |
| `configs/val_subset.yaml` | 验证子集 |

更完整的 Agent 约定、代码树与门控细节见 [`AGENTS.md`](AGENTS.md)。


## 待实现的idea（不得擅自删除或修改）
！！！！！！！！！！！
idea：
mamba本质为线性RNN
能否将decoder的某一层的中间状态反馈到encoder上
比如第n步的decoder中间状态反馈到第n+1步的encoder上
！！！！！！！！！！！
