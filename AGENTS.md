# AGENTS.md — ResMamba Signal Model

给在本仓库工作的 Cursor Agent / 协作者的项目指南。实现与验收以本文件与 `README.md` 为准；冲突时以可运行代码与门控指标为准。

## 项目一句话

任务无关的射频 I/Q 基础模型：RevIN → 时频 Tokenizer → Encoder（5×BiMamba2 + 1×RoPE MemoryTransformer，预训练只看可见 token）→ SharedDecoder（1×Mamba-2 + skip / 物理 FiLM / 双通路）。主类：`SignalFoundationModel` / `SignalModelConfig`（包名 `resmamba_signal_model`）。

## Agent 工作约定

1. **每改必提**：每完成一个逻辑完整的变更（文档、配置、代码、测试），立刻 `git add` + `git commit`；不要攒一大包再提。
2. **分支**：功能/文档在独立分支上改；本仓库默认远程为 `ResMamba_Model`（`git@github.com:WonAlbert/ResMamba_Model.git`）。推送前确认分支名，勿直接推 `main` 除非用户明确要求。
3. **勿提交**：密钥与私钥（`.ssh` / `.ssh.pub` / `.env*`）、大数据（`dataset/**`、`*.h5`、`runs/**`、checkpoint、TensorBoard 事件）、缓存与 `__pycache__`。
4. **训练入口唯一**：`scripts/train.py`。不要恢复已删除的多入口脚本，除非用户明确要求。
5. **未跑满门控前不得宣称 SOTA**；编排脚本只保证可启动，不保证已跑完（见 `docs/sota_gate.md`）。
6. **改动范围**：只改任务相关文件；风格与现有模块一致；不主动写用户未要的 markdown。

## 环境与快速验证

```bash
cd /root/autodl-tmp/ResMamba_Signal_Model
source scripts/env.sh
pip install -r requirements.txt
export PYTHONPATH="$(pwd):$PYTHONPATH"
export RFDATA_ROOT="/root/autodl-tmp/ResMamba_Signal_Model/dataset"

python scripts/prepare_datasets.py --output "${RFDATA_ROOT}" --datasets all
python scripts/smoke_forward.py
python scripts/count_params.py --model-config configs/model.yaml
pytest -q
```

`won` 环境需匹配 PyTorch / `mamba-ssm`。CI 与 tiny 配置允许 Mamba fallback；正式 `configs/model.yaml`（`d_model=640`）要求真实 Mamba kernel。

## 训练阶段（必须按序）

| 阶段 | 命令要点 | 冻结策略 |
|------|----------|----------|
| 一 预训练 | `--stage pretrain --config configs/pretrain.yaml` | 全量训练骨干 |
| 二 LP 探测 | `--stage stage2 --config configs/stage2.yaml --init-from <pretrain>/ckpts/best.ckpt` | 冻结骨干；只训 UTI + 任务头；截断反传 |
| 三 单任务适配 | `--stage stage3 --task <name> --config configs/stage3.yaml --init-from <stage2>/best.ckpt` | Hybrid-LoRA+ / TaskAdapter / 当前任务头 |
| 三 联合 | `--stage joint --config configs/joint.yaml` + 多个 `--adapter-dir` | 联合 PEFT；可解冻 tokenizer 尾部 |

冒烟：任意阶段加 `--profile tiny --synthetic`。日志：`runs/experiments/<run>/`（`config.yaml`、`train.log`、`tb/`、`csv/`、`ckpts/`）。

旧入口 `--stage downstream` 仍可用，但不作为验收主路径。

### 常用命令摘要

```bash
python scripts/train.py --stage pretrain --config configs/pretrain.yaml
python scripts/train.py --stage stage2 --config configs/stage2.yaml \
  --init-from runs/experiments/<pretrain>/ckpts/best.ckpt
python scripts/train.py --stage stage3 --task modulation --config configs/stage3.yaml \
  --init-from runs/experiments/<stage2>/ckpts/best.ckpt
python scripts/train.py --stage stage3 --task emitter --init-from runs/experiments/<stage2>/ckpts/best.ckpt
python scripts/infer.py --task modulation --checkpoint runs/experiments/<run>/ckpts/best.ckpt \
  --model-config configs/model.yaml --datasets rml2016_10a
```

## 验收第一标准（硬门控）

在**完成阶段一预训练**之后，用正式配置（非 tiny / 非 synthetic）评估。未达标则阶段二/三工作视为未通过，不得宣称下游成功或 SOTA。

### 阶段二（仅训练任务头 / 冻结骨干）

条件：`--stage stage2`，骨干冻结，`train_heads: true`，截断反传；从预训练 `best.ckpt` 初始化。

| 任务 | 指标 | 阈值 |
|------|------|------|
| 调制分类（modulation） | val accuracy | **≥ 80%** |
| 个体分类（emitter） | val accuracy | **≥ 70%** |

### 阶段三（进一步微调适配）

条件：在阶段二 checkpoint 上做 `--stage stage3`（Hybrid-LoRA+ 等），再测同一验证协议。

| 任务 | 相对阶段二 | 要求 |
|------|------------|------|
| 调制分类 | 相对提升 | **≥ 10%**（`(acc₃ − acc₂) / acc₂ ≥ 0.10`；例：阶段二 80% → 阶段三 ≥ **88%**） |
| 个体分类 | 相对提升 | **≥ 10%**（同上；例：阶段二 70% → 阶段三 ≥ **77%**） |

记录方式：在对应 `runs/experiments/<run>/` 保留 `train.log` / 监控指标与 `ckpts/best.ckpt`；对比时写明 stage2 / stage3 的 run 名与准确率数值。提升按**相对比例**计算（相对阶段二 acc 提升不少于 10%），**不是**绝对加 10 个百分点。

## 架构要点（改模型时勿破坏）

- **Encoder**：`M-M-M-M-M-T`；预训练 `encode_visible_only`；超长序列 chunk 均值记忆 + RoPE。
- **Decoder**：恰好 1 层 `DecoderBlock`；token 通路重建，readout 通路 `[DEC]` + AttnPool → `z`。
- **Tokenizer**：共享 stem + 多尺度时域 + 复数双侧频谱分带（`fft` + `fftshift` 后再均分）；无 dataset/task token。
- **物理约束**：patch 级 log_power / PAPR / IQ 相关 / 方差比 / 谱质心（`fftfreq`）；软约束 SmoothL1 + 硬约束能量投影。分类在归一化表征空间，重建在 RevIN denorm 后。
- **变长**：`sequence_packing=true`；`TokenBudgetSampler`；`L < 16` 报错；`L > 8192` 重叠切块。
- **多域**：`z` 上 Domain GRL；跨域 InfoNCE 仅标签可对齐时启用。

预训练损失：`L_mae + λ_phys L_phys + λ_impute L_span + λ_readout L_global_phys + λ_domain L_grl`。默认 `iq_normalize: none`（幅度由 RevIN 处理）。

## 配置索引

| 文件 | 作用 |
|------|------|
| `configs/model.yaml` | 正式宽度（需 Mamba kernel） |
| `configs/model_tiny.yaml` | CI / CPU |
| `configs/pretrain.yaml` | 预训练 |
| `configs/stage2.yaml` | 冻结骨干 LP 探测 |
| `configs/stage3.yaml` | 单任务 Hybrid-LoRA+ |
| `configs/joint.yaml` | 联合 PEFT |
| `configs/downstream.yaml` | 旧下游入口 |
| `configs/datasets.yaml` | 数据池白名单 |
| `configs/val_subset.yaml` | 验证子集 |
| `configs/experiments/` | 消融 / 门控实验 overlay |

## 代码树（源码与配置；不含 dataset / runs）

```text
ResMamba_Signal_Model/
├── AGENTS.md
├── README.md
├── pyproject.toml
├── requirements.txt
├── environment.yml
├── configs/
│   ├── model.yaml
│   ├── model_tiny.yaml
│   ├── pretrain.yaml
│   ├── stage2.yaml
│   ├── stage3.yaml
│   ├── joint.yaml
│   ├── downstream.yaml
│   ├── continual.yaml
│   ├── datasets.yaml
│   ├── val_subset.yaml
│   └── experiments/          # 消融、validity、eval、overlays
├── docs/
│   ├── RESEARCH_REPORT.md
│   └── sota_gate.md
├── figures/                  # 波形示意等
├── resmamba_signal_model/
│   ├── __init__.py
│   ├── thread_env.py
│   ├── config/
│   │   └── yaml_config.py
│   ├── data/
│   │   ├── contracts.py
│   │   ├── rfdata.py
│   │   ├── packing.py
│   │   ├── sampling.py
│   │   ├── labels.py
│   │   ├── splits.py
│   │   ├── val_subset.py
│   │   ├── cjr_mix.py
│   │   └── wisig_manytx.py
│   ├── models/
│   │   ├── model.py          # SignalFoundationModel
│   │   ├── tokenizer.py
│   │   ├── backbone.py
│   │   ├── mamba_backbone.py
│   │   ├── transformer.py
│   │   ├── decoder.py
│   │   ├── revin.py
│   │   ├── physics.py
│   │   ├── domain.py
│   │   ├── heads.py
│   │   ├── peft.py           # Hybrid-LoRA+
│   │   ├── adapters.py
│   │   ├── signal_adapter.py
│   │   ├── task_interface.py
│   │   ├── prototypes.py
│   │   ├── varlen.py
│   │   ├── norms.py
│   │   ├── ema.py
│   │   └── mamba_kernel_check.py
│   └── training/
│       ├── lit_module.py
│       ├── data_module.py
│       ├── losses.py
│       ├── metrics.py
│       ├── freeze.py
│       ├── checkpointing.py
│       ├── task_catalog.py
│       ├── mix.py
│       ├── sota_gate.py
│       ├── continual.py
│       └── ...               # lr / early_stop / labels / logging
├── scripts/
│   ├── train.py              # 唯一训练入口
│   ├── infer.py
│   ├── prepare_datasets.py
│   ├── smoke_forward.py
│   ├── count_params.py
│   ├── run_sota_gate.py
│   ├── env.sh
│   └── ...                   # 数据下载/转换/作图
└── tests/                    # pytest：模型、数据、阶段冻结、门控等
```

数据根目录 `dataset/`（H5、外部源、split manifest）与 `runs/` 为运行产物，默认不入库；I/O 契约见 `resmamba_signal_model/data/` 与 `configs/datasets.yaml`。

## 改代码时的优先落点

| 意图 | 优先文件 |
|------|----------|
| 模型结构 | `models/model.py`, `backbone.py`, `decoder.py`, `tokenizer.py` |
| 损失 / 指标 | `training/losses.py`, `training/metrics.py` |
| 冻结 / 阶段行为 | `training/freeze.py`, `training/lit_module.py`, `scripts/train.py` |
| PEFT / 阶段三 | `models/peft.py`, `models/adapters.py`, `configs/stage3.yaml` |
| 数据池 / 标签 | `data/rfdata.py`, `data/labels.py`, `configs/datasets.yaml` |
| 门控编排 | `training/sota_gate.py`, `scripts/run_sota_gate.py`, `docs/sota_gate.md` |

## 完成定义（Definition of Done）

- [ ] 相关 `pytest` 通过；涉及正式宽度时注明是否依赖 Mamba kernel。
- [ ] 训练相关改动：tiny synthetic 冒烟可跑通对应 `--stage`。
- [ ] 若触及下游分类：对照上文**验收第一标准**记录 stage2 / stage3 的 modulation / emitter acc。
- [ ] 本次变更已单独 commit；需要远端时已推送到当前功能分支。
