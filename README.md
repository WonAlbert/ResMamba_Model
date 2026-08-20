# ResMamba Signal Model

任务无关的射频 I/Q 基础模型：RevIN → 时频 Tokenizer → Encoder（5×BiMamba2 + 1×RoPE MemoryTransformer，预训练只看可见 token）→ SharedDecoder（1×Mamba-2 + skip / 物理 FiLM / 双通路）。

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

数据 I/O（H5、`packing.py`、`datasets.yaml`、标签工具）保持不变。预训练默认 `iq_normalize: none`，幅度对齐由模型内 RevIN 完成。

## 快速验证

```bash
python scripts/smoke_forward.py          # 128 与 4096 混 pack 前向
python scripts/count_params.py           # 默认 tiny
python scripts/count_params.py --model-config configs/model.yaml
pytest -q
```

## 架构要点

- **Encoder**：`M-M-M-M-M-T`。预训练 `encode_visible_only`：只打包未 mask 的 token。超长序列上 Transformer 走 chunk 均值记忆压缩 + RoPE，无 `max_tokens` 位置表。
- **Decoder**：恰好 1 层 `DecoderBlock`（BiMamba2 + SwiGLU）。mask 填补、tokenizer gated skip、物理 FiLM；token 通路重建，readout 通路 `[DEC]` + AttnPool 得到 `z`。
- **Tokenizer**：共享 stem + depthwise 多尺度时域 + 与时间对齐的频域分带；无 dataset/task token。
- **物理约束**：patch 级 log_power / PAPR / IQ 相关 / 方差比 / 谱质心；软约束 SmoothL1 + 硬约束能量投影。分类/聚类在归一化表征空间，重建在 RevIN denorm 之后计算。
- **变长协议**：`sequence_packing=true`；`TokenBudgetSampler`；`L < L_min=16` 报错；`L > chunk_len=8192` 重叠切块；训练截断增强 `p_trunc=0.3`。
- **多域**：`z` 上 Domain GRL；跨域 InfoNCE 仅在标签可对齐时（如下游调制）。

主类：`SignalFoundationModel` / `SignalModelConfig`（`resmamba_signal_model`）。

## 训练

唯一入口：`scripts/train.py`（PyTorch Lightning + CombinedLoader + tqdm + TensorBoard + `train.log`）。

```bash
# 阶段一：预训练（多数据源 token 份额混合 + combine_then_pack）
python scripts/train.py --stage pretrain --config configs/pretrain.yaml

# tiny / 无 H5 冒烟（1 个 epoch）
python scripts/train.py --stage pretrain --config configs/pretrain.yaml --profile tiny --synthetic

# 旧下游命令仍可用（不解冻截断；UTI + 五头，按 yaml 的 train_encoder/decoder/heads）
python scripts/train.py --stage downstream --config configs/downstream.yaml

# 阶段二：冻结骨干 + 截断反传，只训 UTI 与五个头（--init-from 只加载权重）
python scripts/train.py --stage stage2 --config configs/stage2.yaml \
  --init-from runs/experiments/<pretrain>/ckpts/best.ckpt

python scripts/train.py --stage stage2 --config configs/stage2.yaml --profile tiny --synthetic

# 阶段三 3a：单任务 Hybrid-LoRA+（--task 为目录中任意任务名）
python scripts/train.py --stage stage3 --task modulation --config configs/stage3.yaml \
  --init-from runs/experiments/<stage2>/ckpts/best.ckpt
python scripts/train.py --stage stage3 --task emitter --init-from runs/experiments/<stage2>/ckpts/best.ckpt
python scripts/train.py --stage stage3 --task clustering --init-from runs/experiments/<stage2>/ckpts/best.ckpt
python scripts/train.py --stage stage3 --task prediction --init-from runs/experiments/<stage2>/ckpts/best.ckpt
python scripts/train.py --stage stage3 --task imputation --init-from runs/experiments/<stage2>/ckpts/best.ckpt

python scripts/train.py --stage stage3 --task modulation --profile tiny --synthetic

# 阶段三 3b：联合训练并写出一份可推理 best.ckpt
python scripts/train.py --stage joint --config configs/joint.yaml \
  --init-from runs/experiments/<stage2>/ckpts/best.ckpt \
  --adapter-dir runs/experiments/stage3_modulation_<ts> \
  --adapter-dir runs/experiments/stage3_emitter_<ts> \
  --adapter-dir runs/experiments/stage3_clustering_<ts> \
  --adapter-dir runs/experiments/stage3_prediction_<ts> \
  --adapter-dir runs/experiments/stage3_imputation_<ts>

python scripts/train.py --stage joint --profile tiny --synthetic
```

阶段三任务数量不限，由 yaml 的 `tasks` / `task_pools` / `task_kinds` 决定。新任务示例：

```yaml
tasks: [modulation, emitter, sonar]
task_kinds: {sonar: classification}
task_pools:
  sonar: [downstream_sonar_train, downstream_sonar_val]
```

```bash
python scripts/train.py --stage stage3 --task sonar --task-kind classification
```

日志目录：`runs/experiments/<run>/`（`config.yaml`、`train.log`、`tb/`、`csv/`）。AutoDL 面板会 symlink 到 `/root/tf-logs/resmamba_current`。

## 72h / 6×5090 实验门控

**未跑满门控前不得宣称 SOTA。** 编排脚本只保证「可启动」，不保证「已跑完」。

```bash
python scripts/run_sota_gate.py --dry-run
python scripts/run_sota_gate.py --smoke
# 正式 6 卡需显式窗口，见 docs/sota_gate.md
python scripts/run_sota_gate.py --execute --windows A --gpus 0,1,2,3,4,5
```

本机无 mamba kernel、H5 未 stamp semantic 列、无 split manifest 时，正式 `model.yaml` 训练会被门控拦截。

推理（同一 `SignalFoundationModel.forward`；joint/best 会按 ckpt 注入 Hybrid-LoRA+ 再 `load_state_dict`）：

```bash
python scripts/infer.py --task modulation --checkpoint runs/experiments/<run>/ckpts/best.ckpt \
  --model-config configs/model.yaml --datasets rml2016_10a
```

冻结：阶段二骨干全冻；阶段三只训当前任务 LoRA / TaskAdapter / 头；联合阶段再解冻 tokenizer 尾部（`time_fuse` / `freq_proj` / `gate` / `physics_proj` / `norm`）。旧三项 `train_encoder` / `train_decoder` / `train_heads` 仍用于 `--stage downstream`。

## 配置

| 文件 | 作用 |
|------|------|
| `configs/model.yaml` | 正式宽度（`d_model=640`，要求 Mamba kernel） |
| `configs/model_tiny.yaml` | CI / CPU（`d_model=64`，允许 fallback） |
| `configs/pretrain.yaml` | 预训练损失与 token 混合 |
| `configs/downstream.yaml` | 分类 / 聚类 / 预测 / 插补（旧下游入口） |
| `configs/stage2.yaml` | 冻结骨干探测：UTI + 五头 + 截断反传 |
| `configs/stage3.yaml` | 单任务 Hybrid-LoRA+（`profiles` 覆盖 task / pool / monitor） |
| `configs/joint.yaml` | 联合 PEFT + tokenizer 尾部，写出一份 best.ckpt |
| `configs/datasets.yaml` | 数据池白名单 |
| `configs/val_subset.yaml` | 验证子集抽样比例 |

预训练损失：`L_mae + λ_phys L_phys + λ_impute L_span + λ_readout L_global_phys + λ_domain L_grl`。
