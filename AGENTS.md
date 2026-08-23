# AGENTS.md — ResMamba Signal Model

给在本仓库工作的 Cursor Agent / 协作者的项目指南。实现与验收以本文件与 `README.md` 为准；冲突时以可运行代码与门控指标为准。

## 项目一句话

任务无关的射频 I/Q 基础模型：联合能量 RevIN（`joint_energy`）→ 内容路由 MoE Tokenizer/Encoder → Encoder（`z_enc`）→ SharedDecoder（预训练 MAE）→ 下游六任务独立头（`z_enc` → TaskAdapter → head）。主类：`SignalFoundationModel` / `SignalModelConfig`（包名 `resmamba_signal_model`）。

下游六任务：`ld_intrapulse` / `ld_model` / `tx_modulation` / `ld_clustering` / `tx_clustering` / `prediction`。

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
| 一 预训练 | `--stage pretrain --config configs/pretrain.yaml` | 全量训练骨干 + UTI（`view_div`） |
| 二 LP 探测 | `--stage stage2 --config configs/stage2.yaml --init-from <pretrain>/ckpts/best.ckpt` | 冻结骨干；只训**当前任务头**（+ 可选 `z_enc` 线性探针）；截断反传 |
| 三 单任务适配 | `--stage stage3 --task <name> --config configs/stage3.yaml --init-from <stage2>/best.ckpt` | 该任务 Hybrid-LoRA+ + TaskAdapter + 头 |
| 三 联合 | `--stage joint --config configs/joint.yaml` + `--adapter-dir` | 各任务 LoRA/Adapter/头 + **仅** `SharedTaskAdapter` |

冒烟：任意阶段加 `--profile tiny --synthetic`。日志：`runs/experiments/<run>/`。

### 常用命令摘要

```bash
python scripts/train.py --stage pretrain --config configs/pretrain.yaml
python scripts/train.py --stage stage2 --config configs/stage2.yaml \
  --init-from runs/experiments/<pretrain>/ckpts/best.ckpt
python scripts/train.py --stage stage3 --task tx_modulation --config configs/stage3.yaml \
  --init-from runs/experiments/<stage2>/ckpts/best.ckpt
python scripts/train.py --stage joint --config configs/joint.yaml \
  --init-from runs/experiments/<stage2>/ckpts/best.ckpt
```

## 验收（当前）

**Tiny 全流程冒烟**（实现验收，非正式 acc 门控）：

```bash
python scripts/smoke_forward.py
python scripts/train.py --stage pretrain --profile tiny --synthetic
python scripts/train.py --stage stage2 --profile tiny --synthetic
python scripts/train.py --stage stage3 --task ld_intrapulse --profile tiny --synthetic
python scripts/train.py --stage stage3 --task ld_model --profile tiny --synthetic
python scripts/train.py --stage stage3 --task tx_modulation --profile tiny --synthetic
python scripts/train.py --stage stage3 --task ld_clustering --profile tiny --synthetic
python scripts/train.py --stage stage3 --task tx_clustering --profile tiny --synthetic
python scripts/train.py --stage stage3 --task prediction --profile tiny --synthetic
python scripts/train.py --stage joint --profile tiny --synthetic
pytest -q
```

正式宽度 acc 门控待预训练跑满后在 `docs/sota_gate.md` 更新；勿再引用旧 modulation/emitter 阈值。

## 架构要点（改模型时勿破坏）

- **Encoder**：`M-M-M-M-M-T`；预训练 `encode_visible_only`（MAE）；可见 token **GatingPool**（默认；`encoder_pool_type: attn_pool` 可切回 AttnPool）→ **`z_enc` / `h_enc`**（分类身份；`z_general`/`z` 别名指向此处；默认 L2 归一化；复用 MAE encode，不二次全序列）；无 mask / 下游 encode 时对全有效 patch 池化；超长序列 chunk 均值记忆 + RoPE。
- **Decoder**：恰好 1 层 `DecoderBlock`；token 通路重建；`[DEC]` + AttnPool → **`z_recon`**（重建/物理 readout，不作分类身份）。
- **下游**：冻结 `z_enc` → 可选 TaskAdapter → 任务私有头；**不**经共享 UTI。联合阶段仅新增 `SharedTaskAdapter`（不解冻 tokenizer 尾部 / 无 `shared_lora`）。
- **UTI**：仅预训练 `view_div` 等；下游路径禁用。
- **Tokenizer**：共享 stem + 多尺度时域 + 复数双侧频谱分带（`fft` + `fftshift` 后再均分）；无 dataset/task token。
- **归一化**：模型内 `revin_scale_mode: joint_energy`（I/Q 共享 Winsorized RMS；`amp_aux` 旁路绝对功率进 Decoder FiLM / 物理读出，不进 `z_enc`）。Loader 保持 `iq_normalize: none`。
- **物理约束**：patch 级 log_power / PAPR / IQ 相关 / 方差比 / 谱质心（`fftfreq`）；绝对 `log_power` 由 `log_power_rel + log_scale` 还原。软约束 SmoothL1 + 硬约束能量投影。分类在归一化表征空间，重建在 RevIN denorm 后。
- **变长**：`sequence_packing=true`；预训练 `HomogeneousTokenBudgetSampler`（每 batch 单一 H5）+ `combine_then_pack: false`；`TokenBudgetSampler` 用于其它阶段；`L < 16` 报错；`L > 8192` 重叠切块。
- **多域**：Domain GRL **只**约束 UTI 声明不变的低秩视图（默认 `semantic`），梯度不进入 `z_enc`；预训练默认关闭 domain 损失。跨域对比学习仅在下游标签可对齐时启用（如调制 contrastive），预训练无 InfoNCE。

预训练损失（默认权重见 `configs/pretrain.yaml`）：`L_mae + λ_phys L_phys + λ_impute L_span + λ_struct L_structure + λ_phase L_structure_phase + λ_readout L_global_phys + λ_vicreg L_VICReg(z_enc) + λ_token L_VICReg(h_enc) + λ_view L_view_div`。MAE 每 batch 从 `{random, contiguous, mixed}` 抽样（`mae_mask_probs`）。`latent` / `domain` / `uti_*` 默认关闭。预训练 `use_labels=False` + collate 防火墙（无 `dataset_id`/标签/采集元数据进 forward）。

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

数据根目录 `dataset/`（H5、外部源、split manifest）与 `runs/` 为运行产物，默认不入库。协议：`*_train.h5` 无标签预训练；`*_test.h5` 阶段二/三有标签训练；`*_val.h5` 全阶段验证。缺 test 时 `prepare_datasets.py --rebuild-pools-only` 从 train 分层切 20%。评估默认 `*_val.h5`（`infer.py --split val`）。I/O 契约见 `resmamba_signal_model/data/` 与 `configs/datasets.yaml`。

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
