# ResMamba Signal Model

`ResMamba_Signal_Model` 是独立于 `signal_large_model` 的 RF 信号子项目。第一版聚焦工程骨架、tiny smoke forward 和预训练损失接口，便于后续扩展到完整大模型训练。

**约定**：除 RFData 数据集外，所有脚本与文档均在本目录内；不跨项目调用 `signal_large_model` / `rf_physmind` 的训练脚本。

## 安装

本项目**绑定 conda 环境 `won`**（见根目录 `environment.yml`、`.conda-env`）。推荐用统一入口脚本加载环境：

```bash
cd /root/autodl-tmp/ResMamba_Signal_Model
source scripts/env.sh
pip install -r requirements.txt   # 首次或依赖变更时
```

或在任意命令前加 `scripts/run_won.sh`（无需手动 activate）：

```bash
scripts/run_won.sh python scripts/train_pipeline.py --stage pretrain ...
```

`won` 环境需匹配 `requirements.txt` 中的 PyTorch / `mamba-ssm` 版本。若见 mamba fallback 警告，请安装与 CUDA 匹配的 `mamba-ssm` 与 `causal-conv1d`。

## 公共环境变量

所有训练命令均在本目录下执行；先加载 `won` 环境：

```bash
cd /root/autodl-tmp/ResMamba_Signal_Model
source scripts/env.sh
```

等效于手动设置：

```bash
export PYTHONPATH="$(pwd):$PYTHONPATH"
export RFDATA_ROOT="/root/autodl-tmp/ResMamba_Signal_Model/dataset"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-20}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-20}"
```

## 数据准备（首次或重建 H5）

原始数据需位于 workspace 下的 `辐射源（非个体）识别数据/` 与 `个体辐射源数据/`。生成 RFData H5 与 `label_maps.json`：

```bash
cd /root/autodl-tmp/ResMamba_Signal_Model
python scripts/prepare_datasets.py --output "${RFDATA_ROOT}" --datasets all
```

仅重建部分数据集或同步 task pool 映射：

```bash
# 单个数据集
python scripts/prepare_datasets.py --output "${RFDATA_ROOT}" --datasets adsb2 wifi150

# 仅刷新 label_maps.json（不重新清洗 H5）
python scripts/prepare_datasets.py --output "${RFDATA_ROOT}" --sync-label-maps
```

## 快速验证

```bash
source scripts/env.sh
python scripts/smoke_forward.py
python scripts/count_params.py
# 带 stage2 PEFT 配置统计可训练参数量（示例：调制 task_path）
python scripts/count_params.py \
  --model-config configs/model_resmamba_400m.yaml \
  --train-config configs/stage2_modulation.yaml \
  --task modulation
pytest -q tests/test_smoke_forward.py
```

查看 ~0.3B 目标配置参数量：

```bash
source scripts/env.sh
python - <<'PY'
import yaml
from pathlib import Path
from resmamba_signal_model import ResMambaSignalConfig, ResMambaSignalModel
raw = yaml.safe_load(Path("configs/model_resmamba_400m.yaml").read_text())
m = raw["model"]
cfg = ResMambaSignalConfig(**{k: m[k] for k in ResMambaSignalConfig.__dataclass_fields__ if k in m})
model = ResMambaSignalModel(cfg)
total = sum(p.numel() for p in model.parameters())
print(f"total_params={total:,}")
PY
```

## 设计要点

- RFData 默认不归一化、不裁剪，保留生成数据中的原始 I/Q 数值尺度。
- 保留 `complex_absmax` 作为消融选项。
- tokenizer 使用多尺度 ResNet 1D patch frontend，base stride 为 `patch_size=8`，并注入物理统计 token `[log_power, log_peak, papr, iq_corr, iq_var_ratio]`。
- task token 仅表示 `modulation=0` 和 `emitter=1`。prediction/clustering 不创建独立 task token。
- backbone 为 ResMamba block：局部 ResConv1D 分支 + 全局 Mamba2/BiMamba2 fallback 分支。
- routing 输出四个 embedding space：`mod_specific`、`emitter_specific`、`long_context_shared`、`cross_domain_shared`。
- 聚类表示使用 `long_context_shared + cross_domain_shared`，不依赖独立聚类 task token。
- Stage2 支持参数高效微调（PEFT）：`peft_mode` 可选 `head_only` / `task_path`（默认）/ `lora_task_path` / `full_backbone`；实现见 `configure_peft_stage2`（`resmamba_signal_model/training/stages.py`），LoRA 见 `resmamba_signal_model/models/lora.py`。推荐优先级：**task_path → lora_task_path → head_only → full_backbone**（详见下文 [参数高效微调（PEFT）](#参数高效微调peft)）。

## RFData 数据池

默认路径：项目内的 `dataset`（也可由 `${RFDATA_ROOT}` 指定）。数据由
`scripts/prepare_datasets.py` 统一生成，清洗详情见 `dataset/cleaning_report.json`。

### 划分规则（H5 后缀 → 用途）

本项目**不设独立测试/评测集**；开发过程中的验证、早停、选模与指标报告统一使用 `*_val.h5`。

| H5 后缀 | 用途 | 对应 task pool（示例） |
|---------|------|------------------------|
| `*_train.h5` | **仅 MAE 预训练** | `pretrain_train` |
| `*_val.h5` | **各阶段验证**（预训练 / stage2 / 微调；含原「测试阶段」） | `pretrain_val`、`downstream_*_val`、`clustering_val` |
| `*_test.h5` | **下游头训练与微调训练**（适配集，**不作评测**） | `downstream_*_train`、`clustering_train`、`downstream_prediction_train` |

说明：task pool 名称里的 `train`（如 `downstream_modulation_train`）表示「该阶段训练用 pool」，实际读取 `*_test.h5`，与预训练用的 `*_train.h5` 严格分离。`label_maps.json` 中**不再注册** `pretrain_test`、`downstream_*_test` 等评测 pool。完整映射见 `task_pools`。

| 阶段 | 训练 pool（读哪类 H5） | 验证 pool |
|------|------------------------|-----------|
| pretrain | `pretrain_train`（`*_train.h5`） | `pretrain_val`（`*_val.h5`） |
| stage2 / 微调 | `downstream_*_train` 或 `clustering_train`（`*_test.h5`） | 对应 `*_val`（`*_val.h5`） |

预训练 loss 权重见 `configs/pretrain.yaml`；stage2 冻结 backbone 与 PEFT 策略见 `configs/stage2_heads.yaml`（`peft_mode`、`lora_rank` / `lora_alpha` / `lora_dropout`）。

### Stage2 验证指标与选模

| 任务 | 验证指标 | 默认 `selection_metric`（`best.pt` / 早停） | 分数据集日志 |
|------|----------|---------------------------------------------|--------------|
| modulation | `acc`、`macro F1` | `f1` | TensorBoard：`val/acc_<dataset>`、`val/f1_<dataset>` |
| emitter | `acc`、`macro F1`、各子集 `acc/*` | `per_dataset_macro_acc` | 已按子数据集汇报 |
| prediction | 样本加权 `SSIM`（I/Q 重建） | `ssim` | `val/ssim_<dataset>` |
| clustering | `NMI`（`global_label_id` vs 原型） | `nmi` | `val/nmi_<dataset>` |

- `selection_metric: auto` 时按上表默认；可显式设为 `val_loss` / `f1` / `nmi` / `ssim` 等（见 `resmamba_signal_model/training/selection.py`）。
- **clustering** 依赖 H5 中的 `global_label_id`（`dataset_id * 100000 + 局部标签`）。更新数据后请执行：  
  `python scripts/prepare_datasets.py --sync-label-maps`（会重写 `global_label_id` 并刷新 `label_maps.json` 中的 `clustering_*` pool）。

---

## 训练命令

入口脚本：`scripts/train_pipeline.py`。支持多 epoch 训练、验证、早停、混合精度、梯度累计、**LR warmup + cosine 衰减**、checkpoint 与 TensorBoard。

**训练前请先执行上文「公共环境变量」中的 export。**

### 参数一览

| 参数 | 说明 | 默认 |
|------|------|------|
| `--stage` | `pretrain`（MAE 预训练）或 `stage2`（下游头） | `pretrain` |
| `--task` | stage2 任务：`modulation` / `emitter` / `prediction` / `clustering` | `modulation` |
| `--model-config` | 模型结构 YAML | `configs/model_tiny.yaml` |
| `--config` | 训练 YAML | pretrain → `configs/pretrain.yaml`；stage2 → `configs/stage2_heads.yaml` |
| `--rfdata-root` | RFData 根目录 | `dataset/` |
| `--pool` | 训练用 task pool | `pretrain_train` |
| `--val-pool` | 验证 task pool；省略时按 pool 自动推断为对应 `*_val` | 自动推断 |
| `--epochs` | 训练轮数 | 读 YAML；`pretrain.yaml` 为 `5` |
| `--batch-size` | micro-batch 大小 | 读 YAML；`pretrain.yaml` 为 `64`；`stage2_heads.yaml` 为 `32` |
| `--lr` | 基准学习率（micro-batch）；配合梯度累计缩放得到 `peak_lr` | pretrain `1e-4`；stage2 `3e-4` |
| `--num-workers` | DataLoader worker 数 | 读 YAML；`pretrain.yaml` 为 `20` |
| `--amp` / `--no-amp` | CUDA 混合精度 | 默认开启 |
| `--gradient-accumulation-steps` | 梯度累计步数；`1` 等价于关闭 | 读 YAML，默认 `4` |
| `--warmup-steps` | 线性 warmup 优化器步数 | 读 YAML；pretrain `1500`，stage2 `100`–`500` |
| `--lr-schedule` | 学习率调度：`warmup_cosine`（默认）或 `warmup_constant` | 读 YAML，默认 `warmup_cosine` |
| `--lr-min-ratio` | cosine 最低 lr = `peak_lr × ratio` | 读 YAML，默认 `0.1` |
| `--lr-scale-with-grad-accum` / `--no-lr-scale-with-grad-accum` | 按累计步数线性缩放 peak lr | 读 YAML，默认 `true` |
| `--early-stopping-patience` | 选模指标无改善容忍 epoch；`0` 禁用 | 读 YAML；pretrain `0`，未配置时 `10` |
| `--early-stopping-min-delta` | 视为改善的最小选模指标变化量 | 读 YAML，默认 `0` |
| `--dataset-balanced-sampling` / `--no-dataset-balanced-sampling` | 开启跨子 H5 平衡采样 | 读 YAML；`pretrain.yaml` / `stage2_heads.yaml` 均为 `true` |
| `--balanced-sampling-strategy` | `length_bucket`（默认）：同 I/Q 长度组 batch、桶内子 H5 等权；`dataset`：WeightedRandomSampler 等权 | 读 YAML，默认 `length_bucket` |
| `--pretrained-checkpoint` | stage2 加载预训练权重路径（仅 model，不含 optimizer） | — |
| `--resume` | 从训练 checkpoint 续训（`last.pt` / `epoch_XXX.pt`），恢复 optimizer/LR/步数 | — |
| `--output-dir` | checkpoint 目录 | `runs/checkpoints` |
| `--log-dir` | TensorBoard 目录 | `runs/tensorboard` |
| `--max-steps` | 调试：训练步数上限（设后等价 1 epoch 内截断，并禁用早停） | — |
| `--val-max-batches` | 调试：验证 batch 上限 | — |

**Pool 与 H5 后缀对应关系**（详见上文划分规则）：

| 阶段 | `--pool`（训练） | `--val-pool`（验证） |
|------|------------------|----------------------|
| 预训练 | `pretrain_train` → `*_train.h5` | `pretrain_val` → `*_val.h5` |
| 调制识别 | `downstream_modulation_train` → `*_test.h5` | `downstream_modulation_val` |
| 个体识别 | `downstream_emitter_train` → `*_test.h5` | `downstream_emitter_val` |
| 预测 | `downstream_prediction_train` → `*_test.h5` | `downstream_prediction_val` |
| 聚类 | `clustering_train` → `*_test.h5` | `clustering_val` |

省略 `--val-pool` 时自动映射：`pretrain_train`→`pretrain_val`，`downstream_modulation_train`→`downstream_modulation_val`，以此类推。

**数据集排除**：
- `configs/excluded_datasets.yaml` 中的子集（当前为 `communication_emitters`）会从**所有** task pool（含预训练、预测、聚类）中移除。
- `configs/downstream_excluded_datasets.yaml` 中的子集（当前为 `xidian14`）**仅**从 stage2 下游与微调 pool 移除，预训练仍保留。
- `configs/downstream_modulation_extra_datasets.yaml` 将 `open_real_data`、`electromagnetic_0926` 额外纳入 `downstream_modulation_*`（H5 内 `mod_label_id` 与 `source_label_id` 对齐）。
- `open_real_data` **不参与 MAE 预训练**；仅 `test/val`（8:2），不做类别均衡（见 `prepare_datasets.py` 中 `VAL_TEST_ONLY_DATASETS`）。
- 个体识别下游另见 `configs/emitter_downstream.yaml` 白名单。

**平衡采样策略**（`dataset_balanced_sampling: true` 时生效）：

| 策略 | YAML / CLI | 行为 |
|------|------------|------|
| `length_bucket` | `balanced_sampling_strategy: length_bucket` | 同 I/Q 长度组 batch；桶间等权、桶内子 H5 等权 |
| `length_bucket_class` | `balanced_sampling_strategy: length_bucket_class` | 同长度组 batch；桶间按类别数加权（个体识别推荐） |
| `dataset` | `balanced_sampling_strategy: dataset` | 全局子 H5 等权（调制/聚类推荐，batch 内长度可混合） |

**各任务默认采样**（`configs/stage2_heads.yaml` 的 `task_configs`，emitter 见 `stage2_emitter.yaml`）：

| 任务 | 策略 | 原因 |
|------|------|------|
| pretrain | `length_bucket` | 多长度混合 MAE，减少 padding |
| modulation | `dataset` | 避免 rml2018 等大集淹没小集 |
| emitter | `length_bucket_class` | 100/150/10 类规模差异大 |
| prediction | `length_bucket` | 多域重建，桶内等权保护小域 |
| clustering | `dataset` | 跨域对比，各子数据集等权 |

预训练 pool 典型长度桶：`128`（RML2016 等）、`256`、`1000`、`1024`、`1200`、`2048`（electromagnetic_0926）。

### 输出与监控

每个 run 在 `--output-dir/<run_name>/` 下生成：

- `best.pt` / `last.pt` / `epoch_XXX.pt`：模型、优化器、AMP scaler、LR 调度、早停、global_step
- `run_config.json`：本次超参与数据 pool 快照

**续训**：使用 `--resume` 指向 `last.pt` / `best.pt` / `epoch_XXX.pt`（与 `--pretrained-checkpoint` 互斥）。恢复 model、optimizer、scaler、lr_scheduler、global_step、best_val；从**已完成 epoch 的下一个**继续；checkpoint 写入原 run 目录。各任务完整续训命令见下文 **[续训命令（各任务）](#续训命令各任务)**。

TensorBoard（loss、lr、val 指标、数据 pool 文本）：

```bash
# 日志写在 runs/tensorboard/<run_name>/，不是 /root/tf-logs
tensorboard --logdir runs/tensorboard --port 6006 --bind_all
```

**AutoDL 内置面板**默认读 `/root/tf-logs`。训练脚本会自动创建 symlink：

`/root/tf-logs/resmamba_current` → 当前 run 的日志目录

若面板仍为空，请确认 Log directory 指向 `/root/tf-logs/resmamba_current`，或改用上面的 `--logdir runs/tensorboard`。

### 常见问题

| 现象 | 原因 | 处理 |
|------|------|------|
| loss 无明显下降趋势 | 预训练含 8 项 loss，raw loss 噪声大；LR warmup 后原先保持恒定 | 看 `train/loss_ema`；已启用 **warmup + cosine 衰减**（`lr_min_ratio=0.1`） |
| `train/lr` 不下降 | 旧版仅 linear warmup，结束后保持 `peak_lr` | 现已改为 warmup 后 cosine 衰减至 `peak_lr × 0.1` |
| loss 跳变 / NaN / OOM | 原始 I/Q 幅度差大；batch 过大；显存碎片 | 已默认 `joint_power`；模型约 **0.3B**、`batch_size=64`；`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| 进度条 `amp=on` | 混合精度训练已开启（旧版显示 `amp=1` 即 True） | 显存不足时加 `--no-amp` |
| TensorBoard 无曲线 | 监控目录错误（如 `/root/tf-logs` 为空） | 用 `runs/tensorboard` 或 `/root/tf-logs/resmamba_current` |
| GPU 占用率锯齿波动 | DataLoader 供数跟不上 GPU（H5 随机读 + 多 worker 争抢 CPU） | 已默认 `pin_memory`、`prefetch_factor=4`；25 vCPU 建议 `num_workers=12` 而非 20 |

---

### 0. Smoke 验证（1 step，无需完整 epoch）

**预训练：**

```bash
python scripts/train_pipeline.py \
  --stage pretrain \
  --model-config configs/model_tiny.yaml \
  --config configs/pretrain.yaml \
  --rfdata-root "${RFDATA_ROOT}" \
  --pool pretrain_train \
  --val-pool pretrain_val \
  --batch-size 2 \
  --max-steps 1 \
  --val-max-batches 5 \
  --num-workers 0 \
  --output-dir runs/checkpoints/smoke_pretrain \
  --log-dir runs/tensorboard
```

**Stage2（以 modulation 为例，其余任务见下文）：**

```bash
python scripts/train_pipeline.py \
  --stage stage2 \
  --task modulation \
  --model-config configs/model_tiny.yaml \
  --config configs/stage2_heads.yaml \
  --rfdata-root "${RFDATA_ROOT}" \
  --pool downstream_modulation_train \
  --val-pool downstream_modulation_val \
  --batch-size 2 \
  --max-steps 1 \
  --val-max-batches 5 \
  --num-workers 0 \
  --output-dir runs/checkpoints/smoke_stage2_mod \
  --log-dir runs/tensorboard
```

---

### 1. Pretrain 正式训练（~0.3B）

`configs/model_resmamba_400m.yaml` 为约 **0.3B** 骨干；`configs/pretrain.yaml` 默认：`batch_size=64`、`gradient_accumulation_steps=4`（有效 batch **256**）、`epochs=5`、`num_workers=12`、`learning_rate=1e-4`（缩放后 peak lr **4e-4**，warmup 后 **cosine 衰减**至 4e-5）、`warmup_steps=1000`、长度分桶等权采样、CUDA 混合精度；早停已禁用。

```bash
python scripts/train_pipeline.py \
  --stage pretrain \
  --model-config configs/model_resmamba_400m.yaml \
  --config configs/pretrain.yaml \
  --rfdata-root "${RFDATA_ROOT}" \
  --pool pretrain_train \
  --val-pool pretrain_val \
  --epochs 5 \
  --batch-size 64 \
  --lr 1e-4 \
  --num-workers 20 \
  --amp \
  --early-stopping-patience 0 \
  --early-stopping-min-delta 0.0 \
  --output-dir runs/checkpoints/pretrain_400m \
  --log-dir runs/tensorboard
```

最简启动（超参全部由 `configs/pretrain.yaml` 提供，仅指定模型与输出目录）：

```bash
python scripts/train_pipeline.py \
  --stage pretrain \
  --model-config configs/model_resmamba_400m.yaml \
  --config configs/pretrain.yaml \
  --rfdata-root "${RFDATA_ROOT}" \
  --output-dir runs/checkpoints/pretrain_400m \
  --log-dir runs/tensorboard
```

**Tiny 低显存调试：**

```bash
python scripts/train_pipeline.py \
  --stage pretrain \
  --model-config configs/model_tiny.yaml \
  --config configs/pretrain.yaml \
  --pool pretrain_train \
  --val-pool pretrain_val \
  --epochs 2 \
  --batch-size 4 \
  --lr 1e-4 \
  --max-steps 20 \
  --val-max-batches 10 \
  --num-workers 0 \
  --early-stopping-patience 0 \
  --output-dir runs/checkpoints/pretrain_tiny \
  --log-dir runs/tensorboard
```

将 `runs/checkpoints/pretrain_400m/<run_name>/best.pt` 作为 stage2 的 `--pretrained-checkpoint`（下文记为 `${PRETRAIN_CKPT}`）。

---

### 2. Stage2 — 调制识别（modulation）

训练读 `*_test.h5`，验证读 `*_val.h5`。**推荐** `configs/stage2_modulation.yaml`：仅解冻 `mod_fuse` + `mod_specific`（不解冻 emitter 路径）、`length_bucket` 训练采样、较低学习率 `5e-5`（有效 peak `2e-4`）。选模指标 `f1`。

若出现 `loss=nan` / `skipped non-finite`：从 `best.pt` 续训并确认使用本配置；已损坏的权重需从预训练 checkpoint 重开。

```bash
export PRETRAIN_CKPT="runs/checkpoints/pretrain_400m/pretrain_20260630_093705/best.pt"

python scripts/train_pipeline.py \
  --stage stage2 \
  --task modulation \
  --model-config configs/model_resmamba_400m.yaml \
  --config configs/stage2_modulation.yaml \
  --rfdata-root "${RFDATA_ROOT}" \
  --pool downstream_modulation_train \
  --val-pool downstream_modulation_val \
  --pretrained-checkpoint "${PRETRAIN_CKPT}" \
  --num-workers 10 \
  --amp \
  --output-dir runs/checkpoints/stage2_modulation \
  --log-dir runs/tensorboard
```

---

### 3. Stage2 — 个体识别（emitter）

**务必使用** `configs/stage2_emitter.yaml`（勿仅用 `stage2_heads.yaml`）：内含 `length_bucket_class` 采样、`unfreeze_modulation_backbone: false`（仅解冻 `emitter_fuse` + `emitter_specific`）、`selection_metric: per_dataset_macro_acc` 与 `early_stopping_patience: 8`。下游子集白名单见 `emitter_downstream_datasets`（`adsb2` / `wifi150` / `radar_emitters`）。

```bash
export PRETRAIN_CKPT="runs/checkpoints/pretrain_400m/pretrain_20260630_093705/best.pt"

python scripts/train_pipeline.py \
  --stage stage2 \
  --task emitter \
  --model-config configs/model_resmamba_400m.yaml \
  --config configs/stage2_emitter.yaml \
  --rfdata-root "${RFDATA_ROOT}" \
  --pool downstream_emitter_train \
  --val-pool downstream_emitter_val \
  --pretrained-checkpoint "${PRETRAIN_CKPT}" \
  --num-workers 10 \
  --amp \
  --output-dir runs/checkpoints/stage2_emitter \
  --log-dir runs/tensorboard
```

`epochs`（30）、`batch_size`、`learning_rate`、`early_stopping_patience` 等由 `stage2_emitter.yaml` 提供；需要时用 CLI 覆盖。

跨域仍偏低时，可再跑全 backbone 微调（`freeze_backbone: false`，即 `peft_mode=full_backbone`；详见 [参数高效微调（PEFT）](#参数高效微调peft)）：

```bash
python scripts/train_pipeline.py \
  --stage stage2 \
  --task emitter \
  --model-config configs/model_resmamba_400m.yaml \
  --config configs/stage2_emitter_finetune.yaml \
  --rfdata-root "${RFDATA_ROOT}" \
  --pool downstream_emitter_train \
  --val-pool downstream_emitter_val \
  --pretrained-checkpoint runs/checkpoints/stage2_emitter/<run_name>/best.pt \
  --num-workers 10 \
  --amp \
  --output-dir runs/checkpoints/stage2_emitter_finetune \
  --log-dir runs/tensorboard
```

---

### 4. Stage2 — 预测（prediction）

**务必使用** `configs/stage2_prediction.yaml`：含 `length_bucket` 训练采样、验证子集上限（`val_subset_fraction: 0.05` + 每子 H5 最多 800 条）、`val_length_bucket_batching`（验证按 I/Q 长度分桶组 batch）。前向仅单遍 encoder+decoder（不做无用 space 计算）。选模指标 `ssim`。

```bash
export PRETRAIN_CKPT="runs/checkpoints/pretrain_400m/pretrain_20260630_093705/best.pt"

python scripts/train_pipeline.py \
  --stage stage2 \
  --task prediction \
  --model-config configs/model_resmamba_400m.yaml \
  --config configs/stage2_prediction.yaml \
  --rfdata-root "${RFDATA_ROOT}" \
  --pool downstream_prediction_train \
  --val-pool downstream_prediction_val \
  --pretrained-checkpoint "${PRETRAIN_CKPT}" \
  --num-workers 4 \
  --amp \
  --output-dir runs/checkpoints/stage2_prediction \
  --log-dir runs/tensorboard
```

调试时可加 `--val-max-batches 50` 进一步缩短验证。`epochs`、`early_stopping_patience` 等由 YAML 提供。

---

### 5. Stage2 — 聚类（clustering）

使用 `configs/stage2_heads.yaml`（`task_configs.clustering` → `dataset` 等权采样）；训练 `clustering_head`，选模指标 `nmi`。需 H5 含 `global_label_id`（见上文 clustering 说明）。

```bash
export PRETRAIN_CKPT="runs/checkpoints/pretrain_400m/pretrain_20260630_093705/best.pt"

python scripts/train_pipeline.py \
  --stage stage2 \
  --task clustering \
  --model-config configs/model_resmamba_400m.yaml \
  --config configs/stage2_heads.yaml \
  --rfdata-root "${RFDATA_ROOT}" \
  --pool clustering_train \
  --val-pool clustering_val \
  --pretrained-checkpoint "${PRETRAIN_CKPT}" \
  --epochs 30 \
  --num-workers 10 \
  --amp \
  --early-stopping-patience 8 \
  --output-dir runs/checkpoints/stage2_clustering \
  --log-dir runs/tensorboard
```

---

### 参数高效微调（PEFT）

Stage2 下游训练通过 `peft_mode` 控制可训练参数范围；逻辑由 `configure_peft_stage2`（`resmamba_signal_model/training/stages.py`）统一应用，训练启动时会打印 `peft_mode` 与 `format_param_stats` 分模块统计。配置项定义在 `configs/stage2_heads.yaml`：

| 配置项 | 说明 | 默认 |
|--------|------|------|
| `peft_mode` | `head_only` / `task_path` / `lora_task_path` / `full_backbone` | `task_path` |
| `lora_rank` | LoRA 秩（仅 `lora_task_path`） | `8` |
| `lora_alpha` | LoRA 缩放系数 | `16.0` |
| `lora_dropout` | LoRA dropout | `0.05` |
| `freeze_backbone` | `false` 时等价 `peft_mode=full_backbone` | `true` |
| `unfreeze_modulation_backbone` | 调制任务是否解冻 `mod_specific` 路径 | 按任务默认 |
| `unfreeze_emitter_backbone` | 个体任务是否解冻 `emitter_specific` 路径 | 按任务默认 |

#### 四种 `peft_mode` 简述

| 模式 | 可训练范围 | 适用场景 |
|------|------------|----------|
| `task_path` | 任务头 + 对应 fuse / space_proj / **space_backbone**（全量解冻任务路径） | **默认首选**；与历史 stage2 行为一致 |
| `lora_task_path` | 任务头 + fuse / space_proj + 任务路径 backbone 上的 **LoRA 适配器**（基座权重冻结） | 参数量接近 `head_only` 但保留 backbone 适配能力 |
| `head_only` | 仅任务头 + fuse（backbone 与 space 全冻结） | 小样本、快速试探、极低显存 |
| `full_backbone` | tokenizer + encoder + 全部 space 路径 + 任务头 | **最后手段**；或由 `freeze_backbone: false` 触发 |

`stage2_modulation.yaml` / `stage2_emitter.yaml` 未显式写 `peft_mode` 时，默认 `task_path`（与 `freeze_backbone: true` + 各任务 `unfreeze_*_backbone` 开关配合）。

#### 参数量对比（~0.3B，`model_resmamba_400m.yaml`，调制任务）

| `peft_mode` | 可训练参数 | 占比 |
|-------------|-----------|------|
| `head_only` | **20.5M** | 6.5% |
| `lora_task_path` | **21.6M** | 6.9% |
| `task_path` | **45.5M** | 14.5% |
| `full_backbone` | **294M** | 93.7% |

聚类 / 个体 / 预测任务的可训练参数量因解冻路径不同而略有差异；可用下文对比脚本一键查看。

#### 选模指标提醒

PEFT 不改变验证指标与 `best.pt` 选模规则，各任务仍以：

| 任务 | 选模指标（`selection_metric`） |
|------|-------------------------------|
| modulation | `f1` |
| emitter | `per_dataset_macro_acc` |
| prediction | `ssim` |
| clustering | `nmi` |

`selection_metric: auto` 时自动映射为上表（见 `configs/stage2_heads.yaml` 与上文 [Stage2 验证指标与选模](#stage2-验证指标与选模)）。

#### 参数统计

```bash
cd /root/autodl-tmp/ResMamba_Signal_Model
source scripts/env.sh

# 裸模型（无 PEFT）
python scripts/count_params.py --model-config configs/model_resmamba_400m.yaml

# 按训练 YAML 应用 peft_mode / 解冻开关
python scripts/count_params.py \
  --model-config configs/model_resmamba_400m.yaml \
  --train-config configs/stage2_modulation.yaml \
  --task modulation

python scripts/count_params.py \
  --model-config configs/model_resmamba_400m.yaml \
  --train-config configs/stage2_emitter.yaml \
  --task emitter
```

训练启动时 `train_pipeline.py` 同样会输出 `total_params` / `trainable_params` 及 `format_param_stats` 分模块明细。

#### 四种策略对比

```bash
source scripts/env.sh
python scripts/compare_peft_strategies.py \
  --model-config configs/model_resmamba_400m.yaml \
  --task modulation

# 其他任务
python scripts/compare_peft_strategies.py --task emitter
python scripts/compare_peft_strategies.py --task prediction
python scripts/compare_peft_strategies.py --task clustering
```

#### Stage2 默认 `task_path`（modulation / emitter）

现有正式训练命令**无需修改**；`configs/stage2_modulation.yaml` 与 `configs/stage2_emitter.yaml` 在 `freeze_backbone: true` 下即 `task_path`（分别解冻 mod / emitter 路径）。示例见上文 §2、§3。

#### `lora_task_path` 示例

在专用 YAML 中设置 `peft_mode: lora_task_path` 及 LoRA 超参（可复制 `stage2_modulation.yaml` 并追加）：

```yaml
# configs/stage2_modulation_lora.yaml（示例片段）
peft_mode: lora_task_path
lora_rank: 8
lora_alpha: 16.0
lora_dropout: 0.05
freeze_backbone: true
unfreeze_modulation_backbone: true
unfreeze_emitter_backbone: false
# …其余字段同 stage2_modulation.yaml
```

```bash
source scripts/env.sh
export PRETRAIN_CKPT="runs/checkpoints/stage2_modulation/stage2_modulation_20260704_121846/best.pt"

python scripts/train_pipeline.py \
  --stage stage2 \
  --task modulation \
  --model-config configs/model_resmamba_400m.yaml \
  --config configs/stage2_modulation_lora_pro6000_r32.yaml \
  --rfdata-root "${RFDATA_ROOT}" \
  --pool downstream_modulation_train \
  --val-pool downstream_modulation_val \
  --pretrained-checkpoint "${PRETRAIN_CKPT}" \
  --num-workers 10 \
  --amp \
  --output-dir runs/checkpoints/stage2_modulation_lora_pro6000_r32 \
  --log-dir runs/tensorboard
```

亦可在 `configs/stage2_heads.yaml` 顶层或 `task_configs.<task>` 中覆盖 `peft_mode`（使用 `--config configs/stage2_heads.yaml` 时生效）。

#### `head_only` 小样本快速适配

仅训练识别头与 fuse，适合极少标注或快速验证管线。复制 `stage2_modulation.yaml` 为专用配置并设置：

```yaml
# configs/stage2_modulation_head_only.yaml（示例片段）
peft_mode: head_only
freeze_backbone: true
unfreeze_modulation_backbone: false
unfreeze_emitter_backbone: false
# …其余字段同 stage2_modulation.yaml
```

```bash
source scripts/env.sh
export PRETRAIN_CKPT="runs/checkpoints/pretrain_400m/pretrain_20260630_093705/best.pt"

python scripts/train_pipeline.py \
  --stage stage2 \
  --task modulation \
  --model-config configs/model_resmamba_400m.yaml \
  --config configs/stage2_modulation_head_only.yaml \
  --rfdata-root "${RFDATA_ROOT}" \
  --pool downstream_modulation_train \
  --val-pool downstream_modulation_val \
  --pretrained-checkpoint "${PRETRAIN_CKPT}" \
  --epochs 10 \
  --batch-size 64 \
  --lr 1e-4 \
  --num-workers 10 \
  --amp \
  --output-dir runs/checkpoints/stage2_mod_head_only \
  --log-dir runs/tensorboard
```

亦可在 `configs/stage2_heads.yaml` 的 `task_configs.modulation` 中覆盖 `peft_mode: head_only` 与 `unfreeze_modulation_backbone: false`（使用 `--config configs/stage2_heads.yaml` 时生效）。

#### `full_backbone` 全量微调（最后手段）

个体识别跨域仍偏低时，使用 `configs/stage2_emitter_finetune.yaml`（`freeze_backbone: false` → 自动 `full_backbone`）：

```bash
source scripts/env.sh

python scripts/train_pipeline.py \
  --stage stage2 \
  --task emitter \
  --model-config configs/model_resmamba_400m.yaml \
  --config configs/stage2_emitter_finetune.yaml \
  --rfdata-root "${RFDATA_ROOT}" \
  --pool downstream_emitter_train \
  --val-pool downstream_emitter_val \
  --pretrained-checkpoint runs/checkpoints/stage2_emitter/<run_name>/best.pt \
  --num-workers 10 \
  --amp \
  --output-dir runs/checkpoints/stage2_emitter_finetune \
  --log-dir runs/tensorboard
```

显存需求显著高于 `task_path`；仅在 `task_path` / `lora_task_path` 收益饱和后再尝试。续训见下文 [Stage2 — 个体全量微调（emitter finetune）](#stage2--个体全量微调emitter-finetune)。

---

### 续训命令（各任务）

将 `<run_name>` 换为实际目录名（如 `stage2_modulation_20260703_112411`）。续训时**不要**传 `--pretrained-checkpoint`；`--output-dir` / `--log-dir` 可省略（自动沿用 checkpoint 所在 run 目录及其中保存的 `log_dir`）。

| 场景 | 用法 |
|------|------|
| OOM / 手动中断 | `--resume .../last.pt` |
| `last.pt` 损坏但早期 `best.pt` 正常 | `--resume .../best.pt` |
| 原计划 epoch 已跑完，想加长训练 | 续训时加大 `--epochs`（须 **大于** checkpoint 内已完成 epoch） |
| 权重已 NaN（大量 `skipped non-finite`） | **勿续训**；用对应任务的「正式训练」命令从 `${PRETRAIN_CKPT}` 重开 |

#### Pretrain

```bash
python scripts/train_pipeline.py \
  --stage pretrain \
  --model-config configs/model_resmamba_400m.yaml \
  --config configs/pretrain.yaml \
  --rfdata-root "${RFDATA_ROOT}" \
  --pool pretrain_train \
  --val-pool pretrain_val \
  --resume runs/checkpoints/pretrain_400m/<run_name>/last.pt \
  --epochs 10 \
  --num-workers 20 \
  --amp
```

#### Stage2 — 调制（modulation）

```bash
python scripts/train_pipeline.py \
  --stage stage2 \
  --task modulation \
  --model-config configs/model_resmamba_400m.yaml \
  --config configs/stage2_modulation.yaml \
  --rfdata-root "${RFDATA_ROOT}" \
  --pool downstream_modulation_train \
  --val-pool downstream_modulation_val \
  --resume runs/checkpoints/stage2_modulation/<run_name>/last.pt \
  --num-workers 10 \
  --amp
```

延长训练示例（原 40 epoch 已完成，再训到 60）：

```bash
python scripts/train_pipeline.py \
  --stage stage2 --task modulation \
  --model-config configs/model_resmamba_400m.yaml \
  --config configs/stage2_modulation.yaml \
  --rfdata-root "${RFDATA_ROOT}" \
  --pool downstream_modulation_train \
  --val-pool downstream_modulation_val \
  --resume runs/checkpoints/stage2_modulation/stage2_modulation_20260703_112411/best.pt \
  --epochs 100 \
  --num-workers 10 --amp
```

#### Stage2 — 个体（emitter）

```bash
python scripts/train_pipeline.py \
  --stage stage2 \
  --task emitter \
  --model-config configs/model_resmamba_400m.yaml \
  --config configs/stage2_emitter.yaml \
  --rfdata-root "${RFDATA_ROOT}" \
  --pool downstream_emitter_train \
  --val-pool downstream_emitter_val \
  --resume runs/checkpoints/stage2_emitter/stage2_emitter_20260703_123213/best.pt \
  --epochs 100 \
  --num-workers 10 \
  --amp
```

#### Stage2 — 个体全量微调（emitter finetune）

从 stage2 emitter 的 `best.pt` **新开** finetune run 见上文 §3；若 finetune run 本身中断，则对其目录续训：

```bash
python scripts/train_pipeline.py \
  --stage stage2 \
  --task emitter \
  --model-config configs/model_resmamba_400m.yaml \
  --config configs/stage2_emitter_finetune.yaml \
  --rfdata-root "${RFDATA_ROOT}" \
  --pool downstream_emitter_train \
  --val-pool downstream_emitter_val \
  --resume runs/checkpoints/stage2_emitter_finetune/<run_name>/last.pt \
  --num-workers 10 \
  --amp
```

#### Stage2 — 预测（prediction）

```bash
python scripts/train_pipeline.py \
  --stage stage2 \
  --task prediction \
  --model-config configs/model_resmamba_400m.yaml \
  --config configs/stage2_prediction.yaml \
  --rfdata-root "${RFDATA_ROOT}" \
  --pool downstream_prediction_train \
  --val-pool downstream_prediction_val \
  --resume runs/checkpoints/stage2_prediction/<run_name>/last.pt \
  --num-workers 4 \
  --amp
```

#### Stage2 — 聚类（clustering）

```bash
python scripts/train_pipeline.py \
  --stage stage2 \
  --task clustering \
  --model-config configs/model_resmamba_400m.yaml \
  --config configs/stage2_heads.yaml \
  --rfdata-root "${RFDATA_ROOT}" \
  --pool clustering_train \
  --val-pool clustering_val \
  --resume runs/checkpoints/stage2_clustering/<run_name>/last.pt \
  --num-workers 10 \
  --amp
```

---

### 6. Stage2 tiny 短程验证（四任务）

无预训练权重时可省略 `--pretrained-checkpoint`（随机初始化 head + backbone，仅验证管线）。

```bash
# modulation
python scripts/train_pipeline.py \
  --stage stage2 --task modulation \
  --model-config configs/model_tiny.yaml \
  --config configs/stage2_heads.yaml \
  --rfdata-root "${RFDATA_ROOT}" \
  --pool downstream_modulation_train \
  --val-pool downstream_modulation_val \
  --epochs 1 --batch-size 4 --lr 3e-4 \
  --max-steps 5 --val-max-batches 5 --num-workers 0 \
  --early-stopping-patience 0 \
  --output-dir runs/checkpoints/stage2_mod_tiny \
  --log-dir runs/tensorboard

# emitter
python scripts/train_pipeline.py \
  --stage stage2 --task emitter \
  --model-config configs/model_tiny.yaml \
  --config configs/stage2_heads.yaml \
  --rfdata-root "${RFDATA_ROOT}" \
  --pool downstream_emitter_train \
  --val-pool downstream_emitter_val \
  --epochs 1 --batch-size 4 --lr 3e-4 \
  --max-steps 5 --val-max-batches 5 --num-workers 0 \
  --early-stopping-patience 0 \
  --output-dir runs/checkpoints/stage2_emit_tiny \
  --log-dir runs/tensorboard

# prediction
python scripts/train_pipeline.py \
  --stage stage2 --task prediction \
  --model-config configs/model_tiny.yaml \
  --config configs/stage2_heads.yaml \
  --rfdata-root "${RFDATA_ROOT}" \
  --pool downstream_prediction_train \
  --val-pool downstream_prediction_val \
  --epochs 1 --batch-size 4 --lr 3e-4 \
  --max-steps 5 --val-max-batches 5 --num-workers 0 \
  --early-stopping-patience 0 \
  --output-dir runs/checkpoints/stage2_pred_tiny \
  --log-dir runs/tensorboard

# clustering
python scripts/train_pipeline.py \
  --stage stage2 --task clustering \
  --model-config configs/model_tiny.yaml \
  --config configs/stage2_heads.yaml \
  --rfdata-root "${RFDATA_ROOT}" \
  --pool clustering_train \
  --val-pool clustering_val \
  --epochs 1 --batch-size 4 --lr 3e-4 \
  --max-steps 5 --val-max-batches 5 --num-workers 0 \
  --early-stopping-patience 0 \
  --output-dir runs/checkpoints/stage2_clu_tiny \
  --log-dir runs/tensorboard
```

---

### 推荐训练顺序

1. `prepare_datasets.py --datasets all` 生成 H5  
2. Smoke：`--max-steps 1` 确认数据与模型前向  
3. Pretrain ~0.3B → 取 `best.pt`  
4. 按任务依次 stage2（modulation / emitter / prediction / clustering）；默认 `peft_mode=task_path`，可选其他 PEFT 策略见 [参数高效微调（PEFT）](#参数高效微调peft)  
5. 全程用 `*_val` 做验证与选模；`*_test.h5` 仅通过 `downstream_*_train` / `clustering_train` 参与头训练或后续微调

---

## 配置文件说明

| 文件 | 用途 |
|------|------|
| `environment.yml` / `.conda-env` | 绑定 conda 环境 **`won`** |
| `scripts/env.sh` | `source` 后激活 `won` 并设置 `PYTHONPATH`、`RFDATA_ROOT`、线程数 |
| `scripts/run_won.sh` | 在 `won` 中执行任意命令（无需手动 activate） |
| `configs/model_tiny.yaml` | GPU/CPU smoke 与调试 |
| `configs/model_resmamba_400m.yaml` | 预训练骨干（约 **0.3B**；文件名保留以兼容 checkpoint 路径） |
| `configs/pretrain.yaml` | 预训练默认：`epochs=5`、`batch_size=64`（有效 batch 256）、`num_workers=20`、`gradient_accumulation_steps=4`、`learning_rate=1e-4`、`warmup_steps=1500`、`lr_schedule=warmup_cosine`、`lr_min_ratio=0.1`、loss 权重、长度分桶等权采样、早停关闭 |
| `configs/stage2_heads.yaml` | stage2 公共默认：`peft_mode=task_path`、`lora_*`、`batch_size=64`、`learning_rate=1e-4`、梯度累计 / warmup / cosine、`task_configs` 采样覆盖 |
| `scripts/count_params.py` | 模型参数量统计；`--train-config` 应用 stage2 PEFT / 解冻配置 |
| `scripts/compare_peft_strategies.py` | 对比四种 `peft_mode` 的可训练参数量 |
| `configs/stage2_modulation.yaml` | 调制专用：length_bucket、仅 mod 路径解冻、较低 lr |
| `configs/stage2_modulation_lora.yaml` | 调制 LoRA 任务路径：`peft_mode=lora_task_path` |
| `configs/stage2_modulation_head_only.yaml` | 调制 head-only 快速适配：`peft_mode=head_only` |
| `configs/stage2_emitter.yaml` | 个体识别专用：采样、解冻、选模、`val_subset` |
| `configs/stage2_prediction.yaml` | 预测专用：验证子集上限、长度分桶验证 batch、较快 SSIM 验证 |
| `configs/stage2_emitter_finetune.yaml` | 个体全 backbone 微调 |

## 当前限制

- 尚未支持多 GPU DDP；大规模训练请单卡运行或自行封装。
- ~0.3B + 有效 batch 256 建议 ≥48GB 显存；tiny 配置仅用于管线验证。
- `mamba-ssm` 未安装时会自动回退到 fallback block；生产训练请安装与 CUDA 匹配的 `mamba-ssm` 与 `causal-conv1d`。
- 不设独立 hold-out 测试集；模型选择与报告指标均以 `*_val.h5` 为准。
