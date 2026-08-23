# ResMamba 可信重构研究报告

**日期：** 2026-08-18  
**范围：** 仓库 `/root/autodl-tmp/ResMamba_Signal_Model` 在「可信 SOTA 重构」计划下的代码实现与验证现状。  
**依据：** 已实现源码、配置、单测与 tiny smoke；**不含** 6×RTX 5090、72 小时正式训练，也**不含** 同协议公开基线对照数字。

**结论先行：** 数据协议、无泄漏生成路径、UTI v2、无监督聚类与原型生命周期已在代码与单测层落地，tiny 合成冒烟可跑。正式宽度训练被 Mamba CUDA kernel 阻断，真实 H5 尚未 stamp 语义列/capture 元数据，任务头仍约 25.9M。因此本文只报告工程有效性，**不得据此宣称调制识别、发射机个体识别、开集或跨域 SOTA**。

---

## 1. 科学问题与三个创新点

主干仍针对长变长射频 I/Q：RevIN、时频 tokenizer、可消融 BiMamba2 编码器、轻量读出与 PEFT。审查结论是：思路合理，但旧结果因目标泄漏、监督聚类、局部标签冲突与不可验证划分而**不能直接支撑 SOTA**。本轮重构的目标不是先堆层，而是先恢复实验有效性，再用同一套通用接口覆盖调制、个体、聚类、预测与插补，并为声纳/导航预留适配器，而不是复制骨干。

三个创新点与计划一致。下表将「代码已实现」与「实验已验证」分开；后者一律记为待做。

| 创新点 | 代码 | 单测 | 真实数据 / 72h 实验 | 同协议 SOTA 对照 |
|---|---|---|---|---|
| 1. 任务契约驱动的选择性不变通用表示 | 已实现 | 通过 | 未跑 | 无数字，不可宣称 |
| 2. Observed-only 坐标查询式统一条件重建 | 已实现 | 通过（含强制变形） | 未跑 | 无数字，不可宣称 |
| 3. 未知信号的发现—拒识—吸收原型生命周期 | 已实现 | 通过 | 未跑 | 无数字，不可宣称 |

### 1.1 任务契约驱动的选择性不变通用表示

**问题：** 跨射频、声纳与导航，哪些动力学应共享、哪些传感器/来源因素应保留，以及新任务如何声明这种不变性。

**做法（已实现）：** 始终保留不可丢失的 `z_general` / `h_general`；在其上以低秩残差形成 `semantic` / `source` / `context` 三个可选视图；由 `UniversalTaskInterfaceV2` 按任务族、读出形态、视图与模态组合成条件，而不再为每个新任务分配无语义随机 ID。射频默认映射为调制语义、发射机来源、接收机/信道；声纳与导航可由 `TaskSpec` 重新解释，而不是硬编码 content/device。

**待实验验证：** 1/3/5 层编码器消融、`z_general` only 对比 UTI v2 选择性视图、冻结探测与跨域 probe 是否优于固定 task ID 或多骨干。当前没有精度、延迟或迁移数字。

**设计依据（计划已核验链接，不作本仓库实验结果）：** [ROSE, ICML 2025](https://proceedings.mlr.press/v267/wang25ci.html)、[SEMPO, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/ecfb69ce6be017deb5a926c2718f6bc1-Paper-Conference.pdf)、[MambaSSL, ICLR 2026 Poster](https://openreview.net/forum?id=YDl4vqQqGP)、[SYNC, ICML 2025](https://openreview.net/forum?id=KhCKypSaqx)、[COGS, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/39753)、[PCE-CVNN, ICASSP 2026](https://doi.org/10.1109/ICASSP55912.2026.11464904)。另见原始传感器优于有损表征的 [GRT, ICCV 2025](https://doi.org/10.1109/ICCV51701.2025.02286)。

### 1.2 Observed-only 坐标查询式统一条件重建

**问题：** MAE、外推预测与缺失插补是否只是目标坐标与可见集合不同的同一条件估计问题。

**做法（已实现）：** 遮挡发生在 RevIN / tokenizer 之前；预测统计量只来自历史 context，插补统计量只来自 observed samples；目标位置被改写后不得改变模型输入表征。显式输出 `mae_mask` / `suffix_mask` / `span_mask` / `target_mask`，不再让 `mae_mask` 语义复用 suffix。判别任务经 UTI v2 从 encoder 读出并绕过 decoder；生成任务共用一个坐标查询 decoder。

**待实验验证：** 三目标预训练相对 MAE-only 的下游增益、查询 decoder 相对旧 skip/FiLM patch head 的精度—参数 Pareto、预测/插补在真实 H5 上的误差。当前仅有无泄漏单测与 tiny 合成前向。

**设计依据：** [TimePerceiver, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/c6c682ba9bd8839104f2a82901da4109-Abstract-Conference.html)、[RoMAE, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/c2626ef6cdaaa6a18927832820079e1d-Abstract-Conference.html)、[Glocal-IB, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/file/96b0ac37ee330fd0d3da7e4183f8050b-Paper-Conference.pdf)。

### 1.3 未知信号的「发现—拒识—吸收」原型生命周期

**问题：** 未知调制/设备如何从无标签簇演化为可拒识、可注册、可持续学习的新类。

**做法（已实现）：** 训练默认使用跨视图均衡原型分配、utilization 与一致性损失，`global_label_id` 不进入聚类训练损失；`PrototypeRegistry` 分 `modulation`（content）与 `emitter`（device）两命名空间；开集分数为 energy 与 Gaussian/Mahalanobis 距离；持续学习只更新共享低秩 adapter 与原型，并含置信蒸馏、旧原型约束与未知簇吸收。

**待实验验证：** NMI/ARI/Hungarian ACC、开集四象限 AUROC/AUPR/FPR95/OSCR/ECE、类增量平均准确率与遗忘率。当前没有这些数字。

**设计依据：** [EMTC, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/39777)、[DEBO, CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Chen_Dual_Energy-Based_Model_with_Open-World_Uncertainty_Estimation_for_Out-of-distribution_Detection_CVPR_2025_paper.html)、[NegCosSch, ICLR 2026 Poster](https://openreview.net/forum?id=vpBKry7kL5)、[SD-LoRA, ICLR 2025 Oral](https://openreview.net/forum?id=5U1rlpX68A)、[CMD-PCL, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/39568)。

---

## 2. 有效性修复

本节对应计划中的四个阻断项。修复已写入代码与单测；**真实磁盘数据只完成了部分回填，尚未 stamp，不能当作已闭环的数据协议。**

### 2.1 生成任务目标泄漏

旧路径在产生 suffix/span mask 之前已对完整真值做 RevIN/tokenizer，且 decoder 可将目标位置真实 token 与 physics 送入 skip/FiLM。

现路径（`SignalFoundationModel._core_forward`）：

1. 先由 `_mask_for_mode` 得到 `mae_mask` / `suffix_mask` / `span_mask`，并合成 `target_mask`。
2. 目标样本从 `observed_sample_mask` 中剔除后才做 RevIN；统计量只来自 observed samples。
3. Tokenizer 看到的是 observed-only 归一化波形；目标位置被置零，不参与输入表征。
4. Decoder 的 gated skip 与物理 FiLM 仅作用于 `visible` token；查询 decoder 只接收可见上下文、目标坐标、任务契约与**上下文**物理摘要。
5. 能量投影默认作用在可见集合，不再用目标真值能量去改写被预测位置（旧行为由 `legacy_target_energy_projection` 开关保留）。

强制变形测试 `tests/test_model_no_leakage.py`：随机替换被遮挡真值后，`tokens` / `z_general` / `h_general` / `recon_norm` / `mae_pred` / RevIN 统计量必须逐元素不变，同时 `patch_targets` 必须改变。该测试已通过。这只证明信息边界，**不证明重建质量。**

### 2.2 监督聚类改为无监督训练

旧 `losses.py` 用 `global_label_id` 优化聚类，实际是监督原型对齐。

现默认 `supervised_clustering: false`（`configs/pretrain.yaml`、`stage2.yaml`、`joint.yaml`、`continual.yaml`）。`unsupervised_clustering_loss` 使用 Sinkhorn 均衡分配、prototype utilization 与跨视图一致性；tiny stage2 smoke 日志中的聚类项为 `cluster_unsupervised`。`tests/test_losses.py::test_clustering_train_loss_ignores_global_label_id` 验证：打乱标签不改变训练损失。标签仍可用于 epoch-end NMI/ARI/Hungarian 评估；`supervised_clustering=True` 作为显式对照开关保留。

采样侧防火墙：`tests/test_training_engineering.py::test_train_sampler_fields_are_label_firewall` 断言训练采样字段不含 `global_label_id`。

### 2.3 标签与命名空间

调制不再把 dataset-local 整数当全局语义。`resmamba_signal_model/data/labels.py` 维护 canonical ontology，显式处理 `QAM16↔16QAM` 等别名，并拒绝纯数字/CLASS-n 伪装。发射机使用独立连续命名空间 `global_emitter_id`，跨库局部 ID 不碰撞。

真实 H5 **尚未 stamp** `canonical_mod_label_id` / `global_emitter_id`。`RFDataH5Dataset` 在缺列时用 `dataset/label_maps.json` 回填，使 loader 与 `tests/test_real_h5_slice.py` 能在 rml2016_04c 上核对 ontology。回填不是磁盘协议：重启训练前仍须

```bash
python scripts/prepare_datasets.py --stamp-semantic-labels --output "${RFDATA_ROOT}"
```

若 `split_manifest.json` 已锁定，stamp 会被拒绝，需新的 `--output` 目录。

### 2.4 划分、group-held-out 与 manifest 工具

旧 WiSig 在每个 Tx/Rx/Day 块内随机切 split，且 loader 不保留 receiver/session/capture。现 `WiSigBlockSink` 按完整 `(Rx, Day, Eq)` capture 分组，稳定哈希分配到唯一 split；无 group 元数据的旧 NPY 路径被拒绝。工具与单测已覆盖（`tests/test_wisig_group_split.py`、`tests/test_split_manifest.py`）。

`configs/datasets.yaml` 个体下游池为 **wisig（主）+ adsb2**（已移除 wifi150）；WiSig H5 已由 `ManyTx.pkl` 按 Rx/Day group-held-out 写入 `dataset/h5/wisig_{train,val,test}.h5`。

不可变 split manifest 工具已实现（SHA256、标签/SNR/receiver 分布、group 重叠检查）。仓库磁盘上 **没有** `dataset/split_manifest.json`。对 rml2016_04c 现场构建的 manifest 将 group 完整性标为 `unverifiable_missing_metadata`，且 `claim_group_held_out` 不为真。I/Q 指纹层面 train/val/test 无交叉，这只排除了完全重复窗口，**不能替代 group-held-out。**

### 2.5 真实磁盘状态（截至本报告）

| 项 | 状态 |
|---|---|
| `dataset/h5` 真实文件 | 约 47 个 H5 存在 |
| H5 内 `canonical_mod_label_id` / `global_emitter_id` | 未 stamp；loader 可从 `label_maps.json` 回填 |
| H5 内 `receiver_id` / `session_id` / `capture_id` | 缺失；RadioML group-held-out 明确不可验证 |
| `dataset/split_manifest.json` | 不存在 |
| WiSig 转换与 group split | 代码就绪；默认训练池未接入，原始 pickle 是否在本机未作为本报告前提 |
| 调制标准基准配置 | RML2016.04C / 10A / 10B、RML2018.01A 已在 `downstream_modulation` |

---

## 3. 架构与预训练

### 3.1 数据契约与输入适配

`SignalBatch`：`values[B,C,L]`、`sample_mask`、`channel_mask`、时间坐标/采样率、`modality_id`、可选 complex-pair、capture 元数据；`to_legacy_dict` 仍暴露 `iq`。骨干不感知具体通道数。`SignalAdapterRegistry` 将任意通道投影到 tokenizer 的两通道入口：射频保留复数对；未注册的单通道（声纳）复制到第一通道并将第二通道置零；多轴 IMU 使用均值/对比保底映射，也可注册可学习 `ChannelProjectionAdapter`。

物理特征已扩展为 **12 维通用结构 + modality feature mask**（`physics.py`）：共享 5 维（log_power、PAPR、包络变化率、谱质心、谱扩展）+ RF 插件（I/Q 相关、方差比、CFO 代理）+ 声纳插件（谱熵、高频比）+ IMU 插件（轴范数、漂移）；缺失维由 mask 置零，不进入 FiLM。

### 3.2 骨干与读出

正式配置 `configs/model.yaml`：`d_model=640`，编码器 `5×BiMamba2 + 1×RoPE MemoryTransformer`，解码器 1 层，`require_mamba_kernel: true`。Tokenizer 为共享 stem + 多尺度时域 + 与时间对齐的频域分带，无 dataset/task token。预训练 `encode_visible_only=true`，只打包未 mask token。变长协议：`sequence_packing`、token-budget 采样、`L_min=16`、超长切块。

1/3/5 层与双向权共享消融：**配置与 sota_gate 作业已就绪**（`overlays/enc1.yaml`、`enc3.yaml`、`abl_bidir_share_*`；`enc5` 复用 validity 基线）。正式 GPU 消融未跑，不能预设五层更强。

### 3.3 UTI v2 与三视图

`task_vec = E_family + E_readout + E_view + E_modality + MetaMLP(metadata) + DomainPrompt(dataset_id)`。任务族：分类 / 聚类 / 生成 / 序列回归 / 稠密预测；读出：pooled / token / query。UTI 用 rank-64 FiLM 与 softmax gate 混合 `general/semantic/source/context`，统一产出三种标准特征。`TaskSpec` 可声明 `invariant_views`；Domain GRL **只**约束声明不变的低秩视图，梯度不进入 `z_general`（`tests/test_pretrain_objectives.py::test_grl_does_not_flow_into_z_general`）。

旧 API：`add_task`、`TaskFeatures` 前三字段、`UniversalTaskInterface` 名称、`uti_legacy_mode` 与 checkpoint 前缀映射（`remap_task_head_checkpoints`）保留。新任务只需 `TaskSpec + head`，不必改 backbone 源码。

UTI 核心约 **0.48M**；**domain prompt register** 已实现（`domain_prompt_size=6`，按 `dataset_id` 注入 condition，不污染 tokenizer 主干）。

### 3.4 查询 decoder

`UnifiedQueryDecoder`：query 维默认 320，单层 cross-attention，输入为可见上下文、连续目标坐标、上下文物理摘要与可选任务条件，共用输出投影完成 MAE / suffix prediction / span imputation。旧 `ReconHead` + skip/FiLM 路径由 `legacy_decoder_reconstruction` 保留，便于旧权重对照。判别任务 `skip_recon` 时不走重建头。

### 3.5 三目标预训练（推理零额外参数）

`configs/pretrain.yaml` 同时打开：

1. **多尺度 observed-only 条件建模：** MAE + span 插补（`mae` / `impute`），同一查询 decoder。
2. **结构保持重建：** 时域 SmoothL1 + 频谱幅度；相位/相干仅在 complex-pair 模态启用，不对声纳/IMU 强加射频相位损失。
3. **遮挡潜变量预测：** student 预测 stop-gradient EMA teacher 的 token/global latent；teacher 仅预训练存在（`EMATeacher`），不进入部署图。

UTI 三条读出同时训练：`uti_pooled` / `uti_token` / `uti_query`。已知限制：验证步上 `latent` / `uti_pooled` / `uti_token` 可为 0，因为 EMA teacher 目前只在 train step 前向。这是指标缺口，不是 tiny smoke 失败。

两级迁移协议（先 adapter+prompt+norm，再冻结 probe，必要时 LoRA）在配置与冻结工具中有阶段二/三骨架；**跨域实验未跑。** MAE-only 对比三目标是 8–28h 消融项，尚无结果。

### 3.6 聚类、开集与持续学习

- 无监督聚类见 §2.2；评估标签与训练损失隔离。
- `PrototypeRegistry`：均值、对角协方差、计数、版本；`absorb` 写入低计数槽，不扩维。
- 开集：energy + Mahalanobis/Gaussian；`negcos_temperature` 为零参数调度。
- `configs/continual.yaml`：共享 LoRA、置信蒸馏、原型锚、未知吸收；不建 per-task LoRA 池。

以上均有单测（`tests/test_prototype_world.py`）。开集四象限与增量序列的真实指标表为空。

### 3.7 训练工程

`scripts/train.py`：全局 seed（`L.seed_everything(..., workers=True)`）、`--devices` / `--strategy ddp`、可配置 precision/`amp_dtype`、token-normalized loss。stage2 选模为 `val/multitask_geomean`（任务指标几何平均），joint 为 `val/specialist_geomean`。旧简单 loss 求和不再作为默认 monitor。tiny smoke 已证明这些键会写入 `train_state.json`。

### 3.8 相对计划仍未落地或未达标的架构项

| 计划项 | 现状 |
|---|---|
| 总参 ≤55M、任务头 ≤1M | **已闭合（默认配置）**：52.5M / 头+UTI 0.99M（`count_params.py --with-heads`） |
| 深层 Cosine/MLP 换成低秩原型分类器 | **已换（默认）**：`EmitterHead`/`ModulationHead`/聚类 proj 与 shared MLP 均走 rank-64 低秩路径；旧深层头可用 `overlays/deep_cosine_head.yaml` 对照 |
| 通用物理特征 + modality mask | **已实现**：`PHYS_DIM=12` + `physics_feature_mask`；RF/声纳/IMU 插件按 `modality_id` 开关 |
| Domain prompt 4–8 向量 | **已实现**：`domain_prompt_size=6`，按 `dataset_id` 注入 UTI condition |
| 1/3/5 层与双向共享正式消融 | **入口已就绪**（`enc1/enc3/enc5` alias + `abl_bidir_share_*`）；72h B 窗口**未跑**，无 Pareto 数字 |
| WiSig 作为默认个体主基准 | **已写入并处理**：`emitter_downstream: [wisig, adsb2]`；H5 约 495k 样本（train/val/test） |

---

## 4. 扩展性（声纳 / 导航）

接口已有，数据和实验未跑。72 小时窗口即使启动，也只验证可迁移性，**不宣称声纳或导航 SOTA**。

已具备：

- `SignalSpec` / `SignalBatch` 任意通道；UTI `MODALITY_TO_ID` 含 `sonar` / `imu` / `navigation`。
- `SignalAdapterRegistry` 单通道与 6 轴保底映射；`register_signal_adapter` 不改 tokenizer/encoder。
- `TaskSpec` + `task_kinds` + `task_pools` 注册分类或序列回归；`tests/test_task_catalog.py`、`tests/test_signal_adapter.py`、`tests/test_task_interface.py` 覆盖 sonar/nav 注册与前向。
- 结构损失对非 complex-pair 模态关闭相位项。

未具备：

- [SonAIr](https://github.com/wineslab/sonair-dataset)、DeepShip/Wolfset、[RoNIN](https://ronin.cs.sfu.ca/)、OxIOD 均未进入本仓库数据根。
- 无 frozen probe / 轻量自适应 / 从头训练 / 全量微调对照。
- 无留一模态或留一数据集数字。
- 计划中的「新增参数 <0.2M、原 RF 任务无明显遗忘」尚未用真实跨域 run 验证。

因此：扩展性目前是**接口验收**，不是能力验收。

---

## 5. 验证与参数

### 5.1 单测

全量 `pytest -q`：**233 passed，2 skipped**（约 6.8s，验证日 2026-08-18）。

跳过项未削弱有效性断言：

- `tests/test_mamba_kernel.py` 的 CUDA smoke：本机无 `mamba_ssm` CUDA kernel。
- `tests/test_val_subset.py`：`radar_mod15` 不在 `downstream_modulation_val` 配置中。

已通过的有效性相关测试包括（非穷尽）：`test_model_no_leakage`、`test_clustering_train_loss_ignores_global_label_id`、`test_wisig_group_split`、`test_train_sampler_fields_are_label_firewall`、`test_pretrain_three_objectives_and_uti_readouts`、`test_prototype_namespaces_do_not_mix`、`test_real_h5_slice`（rml2016_04c）。

### 5.2 Tiny smoke（合成数据，1 epoch / 2 step）

| 阶段 | 命令要点 | 结果 | 说明 |
|---|---|---|---|
| pretrain tiny | `--stage pretrain --profile tiny --synthetic` | 完成；总参 618,725 | `val/monitor=1.575` 仅为合成 2-step 健康检查，**不是**预训练质量 |
| stage2 tiny | `--stage stage2 --profile tiny --synthetic` | 完成；总参 886,599，可训 329,446 | `val/multitask_geomean≈0.0015` 同理不可引用为精度；聚类项为无监督 |

未跑 stage3 / joint / continual 的 tiny 全链路，也未跑正式宽度训练。

### 5.3 真实 H5 小切片

`tests/test_real_h5_slice.py` 对 **rml2016_04c**（约 19MB）：

- 可 load `SignalBatch`（`values=[B,C,L]`，`sample_mask`）。
- `canonical_mod_label_id` 与 ontology 一致（含 `QAM16→16QAM`）。
- 三 split 文件哈希互异，I/Q 指纹无交叉。
- group 完整性为不可验证（缺 capture 元数据）。

这不是全库回归，也不是训练协议验收。

### 5.4 参数量（正式宽度用 fallback 统计，未加载 mamba kernel）

相对重构前约 79.34M、识别/聚类头约 25.07M：

| 配置 | 参数量 | 相对旧口径 |
|---|---|---|
| tiny 仅骨干（`configs/model_tiny.yaml`，`count_params.py` 默认） | 557,153 | CI 宽度 |
| 正式仅骨干 | 50,079,945 | 与旧预训练骨干同量级 |
| 正式 + UTI + 五头 + 原型（`low_rank_prototype=true`，2026-08-18 复算） | **52,532,966（约 52.5M）** | 约 −24.3M |
| 识别/聚类/预测插补头 + UTI 合计 | **987,654（约 0.99M）** | 相对 ~25.9M 大幅压缩 |
| 其中 emitter / modulation / clustering | 约 0.16M / 0.14M / 0.10M | 低秩 Cosine + shared rank-64 |
| UTI v2 | 约 0.48M | 低于 0.6M 目标 |

计划门槛「总参 ≲55M、任务专属可训 ≲1M」**在默认低秩头配置下已闭合**（须与 72h 精度 Pareto 一并报告，不能仅因参数量宣称 SOTA）。

### 5.5 本机正式宽度

`configs/model.yaml` 要求 Mamba2 CUDA kernel。本机 `mamba2_available()==False`，正式配置不能训。Tiny 允许 fallback。因此 **没有** 正式宽度的 throughput、延迟或精度数字。

---

## 6. 72 小时门控：如何启动，以及当前 blocker

正式 72h / 6×5090 矩阵**尚未启动**。不要把 tiny smoke 或旧 stage2 日志当成门控结果。

### 6.1 启动前检查清单

全部满足前，0–8h 有效性基线应判定失败并停止后续 SOTA 消融：

1. `python -c "from resmamba_signal_model.models.mamba_backbone import mamba2_available; assert mamba2_available()"` 为真，且 `pytest tests/test_mamba_kernel.py -q` 不再因缺 kernel 跳过 CUDA smoke。
2. 真实 H5 已 `--stamp-semantic-labels`；需要跨 receiver 结论的数据集已写入 `receiver_id` / `session_id` / `capture_id`。
3. `python scripts/prepare_datasets.py --verify-splits` 生成并锁定 `dataset/split_manifest.json`；声称 group-held-out 的数据集 status 为 `verified_group_held_out`，不可验证的数据集不得写入跨域结论。
4. 调制用 `canonical_mod_label_id`，个体用 `global_emitter_id` 与 namespace 头尺寸；聚类训练损失不含标签。
5. 无泄漏测试与标签防火墙测试保持绿色。

数据准备示例：

```bash
export RFDATA_ROOT="/root/autodl-tmp/ResMamba_Signal_Model/dataset"
python scripts/prepare_datasets.py --stamp-semantic-labels --output "${RFDATA_ROOT}"
python scripts/prepare_datasets.py --verify-splits --output "${RFDATA_ROOT}"
```

若计划中的编排脚本 `scripts/run_sota_gate.py`（或 `scripts/experiments/`）已落地，应先 `--dry-run` 打印命令矩阵，确认依赖条件后再提交 6 卡作业。截至本报告撰写，仓库入口仍是 `scripts/train.py`。

### 6.2 计划时间盒（未执行）

| 窗口 | 目的 | 建议入口 | 失败即停 |
|---|---|---|---|
| 0–8h | 复算修复后基线：无泄漏、无 split 重叠、标签映射正确 | `train.py --stage pretrain` 然后 `--stage stage2 --init-from <pretrain>`；`configs/model.yaml`；`--devices 6 --strategy ddp --seed 0` | manifest/kernel/泄漏任一失败则**停止**后续消融 |
| 8–28h | 短消融：encoder 1/3/5、MAE-only vs 三目标、`z_general` vs UTI v2、相位结构项、查询 decoder | 覆盖 `encoder_mamba_layers`、`loss_weights`、`use_specialist_views` | 按 macro-F1、低 SNR、跨 receiver（仅在可验证 split 上）、冻结探测、参数与延迟早停 |
| 28–60h | 最优两套配置各 3 seeds | `--seed 0/1/2`；调制与个体分别报 unified、dataset-specific、zero/few-shot | 只报均值±标准差，不报单 seed 峰值 |
| 60–72h | 冻结骨干：无监督聚类、开集四象限、类增量、预测/插补；可选一声纳 + 一 IMU frozen probe | `continual.yaml` + 外部数据（当前缺失） | 缺跨域数据则跨域节留空，不编造 |

Stage2/joint 选模必须走 geomean，而不是验证 loss 求和。报告口径必须分开：**unified 模型** 与 **dataset-specific fine-tune**。

手工启动骨架（仅在检查清单通过后）：

```bash
python scripts/train.py --stage pretrain --config configs/pretrain.yaml \
  --devices 6 --strategy ddp --seed 0 --run-name gate_pretrain_s0

python scripts/train.py --stage stage2 --config configs/stage2.yaml \
  --init-from runs/experiments/gate_pretrain_s0/ckpts/best.ckpt \
  --devices 6 --strategy ddp --seed 0 --run-name gate_stage2_s0
```

消融通过 `--model-config` 或覆盖 yaml 中的 `encoder_mamba_layers`、`loss_weights.structure/latent`、`use_specialist_views`。本仓库**禁止**在无 kernel 机器上把 `require_mamba_kernel: false` 的 fallback 结果当作正式 SOTA 口径。

### 6.3 当前 blocker

1. **无 Mamba CUDA kernel：** 正式 `model.yaml` 无法训练；这是 72h 的硬阻断。
2. **真实 H5 未 stamp：** 语义列与 capture 元数据未写入磁盘；跨 receiver 不可验证。
3. **无 `split_manifest.json`：** 不可变划分未锁定。
4. **头参数预算未达标：** 76.8M / 25.9M，距 55M / 1M 很远；即便开训，也须把头压缩纳入 8–28h 消融，而不是事后宣称已精简。
5. **WiSig 未进默认个体池；ADSB2/WiFi150 缺 capture 元数据：** 个体识别的跨接收机结论目前没有合法划分。
6. **声纳/IMU 数据不在仓库：** 60–72h 跨域探测无法执行。
7. **72h 作业本身未提交：** 因此不存在任何门控阶段的任务指标。

---

## 7. 不得宣称领域 SOTA；待验证假设与负结果占位

### 7.1 声明边界（强制）

在缺少**同一数据、同一划分、同一输入长度、同一调参预算、至少 3 个随机种子的均值与置信区间**之前：

- **不得**宣称本模型达到射频调制识别、发射机个体识别、开集识别、聚类、预测/插补或跨域感知的领域 SOTA。
- **不得**用 tiny/synthetic 指标、旧泄漏架构日志、或不同划分上的公开数字进行不对等对比。
- **不得**把「接口能跑声纳/IMU」写成「零样本通用基础模型」。
- 统一模型结果与 dataset-specific SOTA 必须分表；未超过同协议公开基线的部分标为负结果或待验证，不包装为贡献。

本报告引用的 2025–2026 论文只说明设计动机，**不是**本仓库已经复现或超越它们的证据。

### 7.2 待验证假设

下列命题在实验完成前既不成立也不被证伪：

1. **H1（选择性不变）：** 保留 `z_general` 并仅对声明视图做 GRL/低秩专家，优于固定 task ID 或强制 content/device 分解；新任务可用 `TaskSpec` 在不改骨干的前提下接近 specialist。
2. **H2（统一条件重建）：** 严格 observed-only 后，MAE / 预测 / 插补由同一查询 decoder 共享，不低于三个独立生成头，且无泄漏不会伤害判别任务。
3. **H3（三目标预训练）：** 结构重建 + 潜变量预测相对 MAE-only 提升低 SNR 与跨数据集 macro-F1；EMA teacher 不增加部署参数。
4. **H4（原型生命周期）：** 无监督簇在 content/device 双空间上可校准拒识，并在共享 LoRA 下吸收新类，遗忘低于 per-task LoRA 池。
5. **H5（跨域接口）：** 冻结骨干 + 小型输入适配器，在一声纳分类与一 IMU 序列回归上优于从头训练的同预算头；**即使成立也不构成声纳/导航 SOTA。**
6. **H6（深度 Pareto）：** 1/3/5 层中存在精度—参数—延迟更优的点，五层不是默认最优。
7. **H7（头压缩）：** 低秩原型头默认开启；能否把任务专属参数压到 ~1M 且 macro-F1 可接受，须跑 `abl_low_rank_stage2` vs `overlays/deep_cosine_head.yaml` 后填入 §7.3。

### 7.3 负结果与空表占位

完成 72h 后，把数字填入下表；**空表必须保留**。若某行未超过同协议基线，写入「负结果」及效应量，不要删除该行。

#### A. 调制识别（unified / dataset-specific / few-shot 分列）

| 协议 | 数据与划分 | 输入长度 | 种子 | macro-F1 均值±std | 低 SNR | 对照方法 | 结论 |
|---|---|---|---|---|---|---|---|
| unified | RML2016.04C/10A/10B、RML2018.01A + 已锁定 manifest | *待填* | 3 | **待填** | **待填** | 同协议公开基线 | 待验证 |
| dataset-specific | 上表各库单独微调 | *待填* | 3 | **待填** | **待填** | 各库 published SOTA（须核对划分） | 待验证 |

#### B. 发射机个体识别

| 协议 | 数据 | group-held-out | 种子 | per-dataset macro-acc | 跨 receiver | 对照 | 结论 |
|---|---|---|---|---|---|---|---|
| unified | *仅在 manifest 为 verified 时填写 WiSig 等* | 未验证则整行作废 | 3 | **待填** | **待填** | 同协议 | 待验证 |

若 ADSB2/WiFi150 仍无 capture 元数据，跨域行保持空白，并记负条件：「划分不可验证，拒绝报告」。

#### C. 生成与预训练消融

| 对比 | 主任务 Δ | 预测/插补 | 参数 | 结论 |
|---|---|---|---|---|
| MAE-only vs 三目标 | **待填** | **待填** | teacher 仅训时存在 | 待验证 |
| 查询 decoder vs legacy recon | **待填** | **待填** | *待填* | 待验证 |
| 泄漏修复后 vs 旧泄漏基线 | **待填** | 以无泄漏为前提，旧数字不作对照 | — | 待验证 |

#### D. 聚类 / 开集 / 持续学习

| 任务 | 指标 | 均值±std | 对照 | 结论 |
|---|---|---|---|---|
| 无监督聚类 | NMI / ARI / Hungarian ACC | **待填** | 监督聚类 opt-in | 待验证 |
| 开集四象限 | AUROC / AUPR / FPR95 / OSCR / ECE | **待填** | energy-only 等 | 待验证 |
| 类增量 | 平均增量准确率 / 遗忘 / 每类新增参数 | **待填** | per-task LoRA 池 | 待验证 |

#### E. 跨域探测（非 SOTA）

| 数据 | 设定 | 指标 | 对照从头训练 | 结论 |
|---|---|---|---|---|
| 水声（SonAIr 或 DeepShip 等，**当前缺失**） | frozen probe / 轻量适配 | **待填** | **待填** | 只谈可迁移性 |
| IMU（RoNIN 或 OxIOD 等，**当前缺失**） | frozen probe / 轻量适配 | **待填** | **待填** | 只谈可迁移性 |

#### F. 已知工程负条件（已发生，不是精度负结果）

| 项 | 记录 |
|---|---|
| 正式宽度无法在本机训练 | 无 mamba kernel |
| 参数预算 | 76.8M / 头 25.9M，未达 55M / 1M |
| 数据协议 | H5 未 stamp；无 split manifest；RadioML group-held-out 不可验证 |
| 预训练 val 的 latent/UTI 项 | teacher 未在 val 前向，指标不完整 |
| 跨域 | 无声纳/IMU 数据，实验空缺 |

### 7.4 本文允许与不允许的贡献表述

**允许：** 修复了会使旧结论失效的泄漏、监督聚类与标签/划分工具；实现了 observed-only 查询重建、UTI v2 与双命名空间原型生命周期；单测 233/2 与 tiny smoke 证明主路径可运行。

**不允许：** 「达到 SOTA」「全面优于某顶会方法」「已证明跨射频—声纳—导航通用」「72h 门控显示……」——这些句子在填入 §7.3 之前均为不实陈述。

---

## 参考文献（仅计划已核验条目）

1. ROSE, ICML 2025. https://proceedings.mlr.press/v267/wang25ci.html  
2. SEMPO, NeurIPS 2025. https://proceedings.neurips.cc/paper_files/paper/2025/file/ecfb69ce6be017deb5a926c2718f6bc1-Paper-Conference.pdf  
3. MambaSSL, ICLR 2026 Poster. https://openreview.net/forum?id=YDl4vqQqGP  
4. SYNC, ICML 2025. https://openreview.net/forum?id=KhCKypSaqx  
5. COGS, AAAI 2026. https://ojs.aaai.org/index.php/AAAI/article/view/39753  
6. PCE-CVNN, ICASSP 2026. https://doi.org/10.1109/ICASSP55912.2026.11464904  
7. GRT, ICCV 2025. https://doi.org/10.1109/ICCV51701.2025.02286  
8. TimePerceiver, NeurIPS 2025. https://proceedings.neurips.cc/paper_files/paper/2025/hash/c6c682ba9bd8839104f2a82901da4109-Abstract-Conference.html  
9. RoMAE, NeurIPS 2025. https://proceedings.neurips.cc/paper_files/paper/2025/hash/c2626ef6cdaaa6a18927832820079e1d-Abstract-Conference.html  
10. Glocal-IB, NeurIPS 2025. https://proceedings.neurips.cc/paper_files/paper/2025/file/96b0ac37ee330fd0d3da7e4183f8050b-Paper-Conference.pdf  
11. EMTC, AAAI 2026. https://ojs.aaai.org/index.php/AAAI/article/view/39777  
12. DEBO, CVPR 2025. https://openaccess.thecvf.com/content/CVPR2025/html/Chen_Dual_Energy-Based_Model_with_Open-World_Uncertainty_Estimation_for_Out-of-distribution_Detection_CVPR_2025_paper.html  
13. NegCosSch, ICLR 2026 Poster. https://openreview.net/forum?id=vpBKry7kL5  
14. SD-LoRA, ICLR 2025 Oral. https://openreview.net/forum?id=5U1rlpX68A  
15. CMD-PCL, AAAI 2026. https://ojs.aaai.org/index.php/AAAI/article/view/39568  
16. SonAIr dataset. https://github.com/wineslab/sonair-dataset  
17. RoNIN. https://ronin.cs.sfu.ca/  
