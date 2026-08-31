# 统一 IQ 数据预处理落地方案（ResMamba 预训练池）

> 适用范围：`configs/datasets.yaml` → `pretrain` 白名单 **7 库**（**不含** `xidian14`）。
> 实现入口：`scripts/prepare_datasets.py`；Loader 契约：`resmamba_signal_model/data/rfdata.py`；
> 训练归一化：模型内 `joint_energy` RevIN（`configs/pretrain.yaml` → `iq_normalize: none`）。

---

## 0. 与本项目对照：合理性审查

### 0.1 原文档中仍适用的原则

| 原则 | 说明 |
|------|------|
| 不统一采样率 | 各库保留真实 \(f_s\)；已知常量可写 H5 **文件级** attrs，不必强行 per-sample |
| 不做全局 z-score / 跨库幅度对齐 | 与 `iq_normalize: none` + RevIN 一致 |
| 主输入仅为 I/Q | Tokenizer 只吃 `[I, Q]`，不默认拼接 \(A,P,\phi,f_i\) |
| 派生物理量在线算 | log_power / PAPR / 谱质心等由 `physics.py`、RevIN `amp_aux` 在训练侧算 |
| 先划分再 crop | `group_id` / 原始 sample 级划分后再做训练时 random crop，防泄漏 |
| SNR 分层评估 | 对有 SNR 的库（RML / RadChar / Panoradio / CJR）可按 bucket 报 acc |

### 0.2 原文档与当前实现不一致处（已在本版修正）

| 原文档表述 | 项目现状 | 本版处理 |
|------------|----------|----------|
| H5 同时存 `iq_raw` + `iq_norm`，预处理阶段做 RMS 归一化 | H5 只存 **`iq`**；幅度语义在模型 RevIN（`joint_energy`） | 删除双通道 IQ；**禁止**在 prepare 阶段写 norm IQ |
| 「禁止 interpolation 统一 L」 | `canonical_iq.py` 按库定长档 128/512/1024（长则 FFT resample，短则 pad） | 写明定长策略与 `DEFAULT_DATASET_LENGTHS` 对齐 |
| 统一 `SNR ≥ 8 dB` | **已落地**：有限 snr 一律 `snr >= 8`；radar_mod15 名义 10 dB | `prepare_datasets.UNIFIED_MIN_SNR_DB`；可用 `--apply-min-snr` 就地过滤现库 |
| 大量 per-sample 物理量永久入库（rms/papr/dc/谱峰…） | Loader / Writer 未读这些列；RevIN 在线统计 | 移出**共有** schema，改在线计算 |
| `group_id` / `class_id` / `label_type` | 实际用 `mod_label_id` / `source_label_id` + `dataset_id` | 对齐 `Writer.FIELDS` |
| 「7 个数据集」含 xidian14 | 预训练池 7 库 = 下表，**排除 xidian14** | 全文移除 xidian14 |
| 80/20 train/val 或无 test | 项目协议需 train/test/val 三份 | 统一 **7:2:1**；用途见 §12 |

### 0.3 预训练 7 库一览（不含 xidian14）

| dataset_id | stem | 族 | 定长 L | SNR | 主标签字段 |
|------------|------|-----|--------|-----|------------|
| 2–4 | rml2016_04c / 10a / 10b | tx_comm | 128 | 有，**≥8 dB** | mod_label_id |
| 16 | radchar | ld_radar | 512 | 有，**≥8 dB** | mod_label_id, source_label_id |
| 10 | radar_mod15 | ld_radar | 1024 | **名义 10 dB**（全样本统一） | source_label_id, mod_label_id |
| 31 | cjr_mix | ld_radar | 1024 | 有，**≥8 dB** | mod_label_id, source_label_id |
| 12 | panoradio_hf | tx_comm | 1024 | 有，**≥8 dB** | mod_label_id |

预训练 collate 会剥离全部标签与 `dataset_id`（`pretrain_collate_firewall`）；标签字段仍写入 H5 供 stage2/3 复用。

---

## 1. 总体原则

7 库统一采用：

$$
\boxed{
\text{原始 IQ 保真}
\rightarrow
\text{格式统一 }[N,2,L]
\rightarrow
\text{质量检查}
\rightarrow
\text{可选 canonical\_iq（去 DC / 谱峰居中 / 定长档）}
\rightarrow
\text{按 sample 划分 train/test/val（7:2:1）}
\rightarrow
\text{训练时 RevIN + 动态 crop/Tokenizer}
}
$$

核心原则：

1. **不统一采样率**（per-sample 列非共有；常量写 attrs）。
2. **定长档** 128 / 512 / 1024（见 `canonical_iq.DEFAULT_DATASET_LENGTHS`），非「全库同一 L」。
3. **H5 v2 离线 joint_energy**；Loader `iq_normalize: none`；训练跳过在线 RevIN 统计（见 §17）。
4. **主输入只使用 I/Q**（归一化在模型前向完成）。
5. **最终入库字段 = 7 库共有 schema**；库特有雷达/干扰参数不进共有层。
6. **派生物理量在线计算**，不大规模预存。
7. **按原始 sample 划分 train/test/val = 7:2:1**，禁止 crop 泄漏；各 split 用途见 §12。

---

## 2. 统一数据格式

所有 H5 信号张量：

```text
iq: float32 [N, 2, L]
channel 0 = I
channel 1 = Q
```

* \(N\)：样本数
* \(L\)：该 H5 文件固定长度（库内统一，库间可 128 / 512 / 1024）

启用 `--canonical-iq` 时（推荐）：

```text
去 DC → 谱峰居中 → 定长（长则 FFT resample，短则居中零填充）
scale_policy = canonical_iq_no_amp_norm
```

**禁止**在 H5 写入 energy-normalized / z-score / max-IQ 波形副本。

---

## 3. Raw 层

原始数据完全保留，不修改：

```text
raw/
├── iq（或等价原始波形）
├── 原始 SNR / 标签 / 采样率（若源提供）
├── original_sample_id
└── 库特有参数（PRI、脉宽等，仅作转换参考）
```

Raw 层禁止：normalization、全局滤波、随机 crop、增广。

---

## 4. Canonical / H5 层 —— 7 库共有字段（最终保存）

以下 schema 为**当前 7 库任务相关**的共有字段（调制/型号识别；**不含辐射源个体**）；
实现上 `Writer` 可能仍预留 `emitter_id` 等列（恒填 -1），但本方案**不写入、不读取、不文档化**。

### 4.1 信号

```text
iq                    # float32 [N, 2, L] — 唯一波形数据集
```

### 4.2 共有 metadata（per-sample）

**两层含义**（勿混淆）：

1. **物理列**：`Writer` 对每个 H5 **一律创建**下列 dataset（见 §14.1）。
2. **语义值**：预训练 7 库中，许多列仅为 fillvalue / 占位；**真正有含义**的列见 §4.4、§14.2。

```text
# —— 7 库均有有效语义 ——
iq                    # float32 [N, 2, L]
length                # int32，= 该 H5 固定 L
dataset_id            # int32，库 ID（见 label_maps.json）
task_type_id          # int8，当前 7 库均为 0（调制/型号类）
snr                   # float32；radar_mod15 名义 10.0，其余来自源
mod_label_id          # int32，库内调制/型号类

# —— 部分库有语义 / 或 prepare 后 stamp ——
source_label_id       # int32；radchar / radar_mod15 / cjr_mix 有；RML / panoradio 恒 -1
canonical_mod_label_id# int32；prepare ``finalize`` 时 ``stamp_semantic_namespaces`` 写入；
                      # 未 stamp 时为 -1；Loader 也可按 label_maps 运行时回填

# —— 7 库仅为 schema 占位（恒 __missing__，无真实采集信息）——
receiver_id           # UTF-8；预训练 7 库均未提供
session_id
channel_id
capture_id
capture_date
```

> **不再作为共有字段**：`group_id`、`class_id`、`label_type`、`duration`、
> `rms`、`energy`、`norm_scale`、`papr_db`、`dc_i`、`dc_q`、
> `frequency_peak`、`bandwidth`、`iq_raw`、`iq_norm`、`quality_flag`（质量差样本直接丢弃或计数，不入库列）。

### 4.3 文件级 attrs（推荐，非 per-sample）

```text
source_path
scale_policy          # none 或 canonical_iq_no_amp_norm
quality_filter
channel_axis          # 1
signal_contract_version
missing_metadata_marker
canonical_iq          # JSON，启用定长预处理时
sampling_rate_hz      # 可选 attrs；radchar=3.2e6；其余见 §13.2
```

### 4.4 字段语义可用性（7 库，非“列是否存在”）

| 字段 | 04c/10a/10b | radchar | radar_mod15 | cjr_mix | panoradio_hf |
|------|:-----------:|:-------:|:-----------:|:-------:|:------------:|
| iq | ✓ | ✓ | ✓ | ✓ | ✓ |
| length / dataset_id / task_type_id | ✓ | ✓ | ✓ | ✓ | ✓ |
| snr | 源 SNR | 源 SNR | **10.0 名义** | 源 SNR | 源 SNR |
| mod_label_id | ✓ | ✓ | ✓ | ✓ | ✓ |
| source_label_id | **-1** | ✓ | ✓ | ✓ | **-1** |
| canonical_mod_label_id | finalize 后 stamp | 同左 | 同左 | 同左 | 同左 |
| receiver/session/channel/capture/capture_date | **占位** | **占位** | **占位** | **占位** | **占位** |

> **说明**：上表 ✓ = 写入时有真实语义；**-1** / **占位** = 列存在但无业务含义（`Writer` fillvalue / `__missing__`）。

---

## 5. SNR 筛选（统一 `SNR ≥ 8 dB`）

| 库 | 规则 | 备注 |
|----|------|------|
| rml2016_* | **snr ≥ 8 dB** | 源 metadata |
| panoradio_hf | **snr ≥ 8 dB** | CSV tags |
| radchar | **snr ≥ 8 dB** | 源 `signal_to_noise_ratio` |
| cjr_mix | **snr ≥ 8 dB** | parquet snr |
| radar_mod15 | 名义 SNR = **10 dB** | 恒 ≥8， naturally 通过 |
| rml2018_1a（若启用） | val/test **snr ≥ 8 dB** | 与统一门槛对齐 |

就地过滤现有 H5（无需从源重转）：

```bash
python scripts/prepare_datasets.py --output "${RFDATA_ROOT}" \
  --apply-min-snr --allow-locked-rebuild --min-snr-db 8
```

**评估**仍可按更高 bucket（10–12 / … dB）分层报告。

---

## 6. 数据质量检查

每个 sample 检查（`prepare_datasets.quality_mask`）：

```python
finite(IQ)
non_silent          # 非全零
robust_power        # 8×MAD 功率异常
PAPR <= 100
lag1_correlation    # 结构代理
label_valid         # 按类均衡过滤时
```

异常样本计数写入 prepare report；**不**单独存 `quality_flag` 列。

---

## 7. 归一化方案（训练侧，非 H5）

H5 **只存原始尺度 IQ**。训练时：

1. Loader：`iq_normalize: none`
2. 模型 RevIN：`revin_scale_mode: joint_energy`（I/Q 共享 Winsorized RMS）
3. `amp_aux` 旁路绝对功率 → Decoder FiLM / 物理读出，**不进** `z_enc`

禁止作为默认方案：

* 全库 global z-score
* 跨库 amplitude distribution 对齐
* Max-IQ normalization（可作 ablation，非主路径）
* H5 阶段写入 `iq_norm`

---

## 8. DC / 谱峰（canonical_iq，非 per-sample 字段）

启用 `--canonical-iq` 时在 **prepare 阶段**对 IQ 做：

* 逐通道去 DC
* 复谱主峰旋至零频

结果直接写进 `iq` 数据集；**不**另存 `dc_i` / `dc_q` / `frequency_peak`。

---

## 9. 在线计算的物理量（不永久保存）

以下在训练 / 损失中在线算（`physics.py`、`revin.py`）：

```text
|x|, |x|², arg(x), 瞬时频率, FFT/PSD, STFT, 自相关, 高阶累积量
log_power, PAPR, IQ 相关, 方差比, 谱质心
RevIN amp_aux: log_scale, log_peak, papr_preclip, scale_gap
```

---

## 10. Tokenizer 主输入

$$
\boxed{
X = \text{RevIN}(\text{IQ}) = [\tilde I, \tilde Q]
}
$$

不默认拼接 \([I,Q,A,P,\phi,f_i]\)（可由 IQ 确定性导出，冗余）。

---

## 11. 库特有字段（扩展层，非预训练共有）

以下 **不得** 写入 7 库共有 schema；若下游任务需要，可放独立 sidecar / 任务 H5：

### RadChar（脉内调制）

```text
# attrs：radchar_signal_types, sampling_rate_hz=3.2e6（§13.2）
```

### CJR_MIX

```text
# parquet：infer_class → mod_label_id / source_label_id；fs=20e6（§13.2，convert 待写入 H5）
```

### radar_mod15

```text
# CSV 幅度 → Hilbert 解析信号 → IQ
# snr=10.0 名义；逐样本 fs → dataset/radar_mod15_sample_rates.json（§13.2）
# attrs: snr_policy=nominal_10dB, snr_db_nominal=10.0
```

### RML2016.*

```text
mod_label_id + snr（pickle 源）；fs=1e6 整库常量（§13.2，H5 待写 attrs）
```

### Panoradio HF

```text
mod_label_id + snr（tags CSV）；fs=6e3 整库常量（§13.2，H5 待写 attrs）
```

---

## 12. Train / Test / Val 划分

### 12.1 比例与用途

全库统一按**原始 sample**（或 verified `group_id`）随机划分：

$$
\boxed{
\text{train : test : val} = 7 : 2 : 1
}
$$

| H5 文件 | 占比 | 训练阶段 | 说明 |
|---------|------|----------|------|
| `{stem}_train.h5` | **70%** | **阶段一 预训练**（MAE） | 无标签泄漏路径；仅 MAE / SSL |
| `{stem}_test.h5` | **20%** | **阶段二 / 三 下游训练与微调** | stage2 LP 探测、stage3 单任务适配的有标签 **训练集** |
| `{stem}_val.h5` | **10%** | **验证与测试** | 全阶段验证、早停、模型选择；`infer.py --split val` 默认评估 |

对应配置池（`configs/datasets.yaml`）：

```text
pretrain          → *_train.h5
downstream_*      → *_test.h5（训练）
各 task val pool  → *_val.h5（验证 / 测试，不作下游训练）
```

**禁止**用 `*_test.h5` 做最终评测或早停；**禁止**用 `*_val.h5` 做下游 finetune 训练。

### 12.2 划分顺序（防泄漏）

$$
\boxed{
\text{Raw sample}
\rightarrow
\text{Train / Test / Val（7:2:1）}
\rightarrow
\text{Crop}
\rightarrow
\text{Augmentation}
}
$$

* **7 库均需产出** `{stem}_train.h5`、`{stem}_test.h5`、`{stem}_val.h5` 三份（含 cjr_mix）。
* 在**同一原始 sample / group** 内切分；固定随机种子，建议**按类分层**（stratified）以保持类别比例。
* 禁止同一原始 sample 的 crop 跨 split：

```text
同一原始 sample
├── crop_1 → train   ✓
├── crop_2 → val     ✗
└── crop_3 → test    ✗
```

### 12.3 实现参考

`scripts/prepare_datasets.py` 当前部分库仍为 80/20（train/val）或仅 train；**应以本节约束为准**，逐步改为 7:2:1 并补全 `*_test.h5`。

---

## 13. 长度与采样率

### 13.1 定长档

**定长档**（`canonical_iq.DEFAULT_DATASET_LENGTHS`）：

```text
128  : rml2016_04c, rml2016_10a, rml2016_10b
512  : radchar
1024 : radar_mod15, cjr_mix, panoradio_hf
```

训练阶段在定长 IQ 上再 random crop → patch/tokenizer（`token_budget`、`l_min`）。

**原则**：不重采样统一 \(f_s\)；各库保留真实采样率语义（常量写 attrs，逐样本写 sidecar 或 H5 列）。

### 13.2 各库采样率 \(f_s\) 获取（预训练 7 库）

| 库 | \(f_s\) | 粒度 | 获取方式 | 项目落盘（当前） | 来源 / 依据 |
|----|---------|------|----------|------------------|-------------|
| rml2016_04c | **1 MHz** | 整库常量 | 公开文献 + 128 点 × 128 µs 反推 | H5 **未写**；Loader 默认 NaN | [O'Shea et al. 2016](https://arxiv.org/abs/1602.04105) GNU Radio 合成；[radioML/dataset#27](https://github.com/radioML/dataset/issues/27) |
| rml2016_10a | **1 MHz** | 整库常量 | 同 04c 系列 | 同左 | 同上 |
| rml2016_10b | **1 MHz** | 整库常量 | 同 04c 系列 | 同左 | 同上 |
| radchar | **3.2 MHz** | 整库常量 | 源 H5 / 论文 README | H5 **attrs** `sampling_rate_hz=3200000` | [RadChar GitHub](https://github.com/abcxyzi/RadChar)；`RADCHAR_SAMPLE_RATE_HZ` in `prepare_datasets.py` |
| radar_mod15 | **~200–300 MHz**（逐样本） | **per-sample** | 源 CSV `time(s)` → \(f_s=1/\mathrm{median}(\Delta t)\) | **`dataset/radar_mod15_sample_rates.json`** | `scripts/build_radar_mod15_sample_rates.py`；类级汇总见 `dataset/radar_mod15_samplingrate.txt` |
| cjr_mix | **20 MHz** | parquet 行级（实际恒 20M） | 源 parquet 列 **`fs`** | H5 **未写**（convert 未读 `fs`） | [HuggingFace CJR-mix](https://huggingface.co/datasets/LapplandSaluzzo/CJR-mix) schema |
| panoradio_hf | **6 kHz** | 整库常量 | 官方 readme | H5 **未写** | [dataset_panoradio_hf_readme.txt](https://www.panoradio-sdr.de/wp-content/uploads/dataset_panoradio_hf_readme.txt)；[Scholl 2019](https://arxiv.org/abs/1906.04459) |

> **时长校验**：\(T \approx L / f_s\)（native L：RML 128 → 128 µs；RadChar 512 → 160 µs；Panoradio 2048 → 341 ms；CJR ~1024 @ 20 MHz → ~51 µs）。

#### rml2016_04c / 10a / 10b（1 MHz，整库常量）

```text
源：pickle（RadioML 2016 系列），无显式 fs 字段
fs = 1_000_000 Hz
依据：论文 "roughly 1 MSamp/sec"，128 样本窗口对应 128 µs
建议：prepare 写 H5 attrs sampling_rate_hz=1e6（或 sidecar 常量 JSON）
```

#### radchar（3.2 MHz，整库常量）

```text
源：RadChar-Small.h5（external/radchar/）
fs = 3_200_000 Hz
实现：radchar() → writer.f.attrs["sampling_rate_hz"] = RADCHAR_SAMPLE_RATE_HZ
验证：tests/test_prepare_radchar.py
```

#### radar_mod15（逐样本，sidecar JSON）

源 CSV 在 `open_realData/outputv2/round7_dataset/<class_id>/*.csv`，含列 `time(s)` 与幅度。

**生成逐样本采样率索引**（H5 与源 CSV 按 IQ fingerprint 对齐）：

```bash
cd /root/autodl-tmp/ResMamba_Signal_Model
python scripts/build_radar_mod15_sample_rates.py \
  --output dataset/radar_mod15_sample_rates.json
```

输出 `dataset/radar_mod15_sample_rates.json` 结构：

```text
dataset: radar_mod15
inference_method: sample_rate_hz = 1 / median(diff(time(s)))
splits:
  train / val / test:
    [{ index, class_index, class_name, source_csv,
       signal_length, sample_rate_hz, duration_us, match_status }, ...]
```

类级 \(f_s\) 分布摘要（手工统计）：`dataset/radar_mod15_samplingrate.txt`（按 class × fs 聚合，非 per-sample）。

当前 **H5 无** `sample_rate_hz` 列；训练侧需 merge sidecar 或后续 prepare 写入。

#### cjr_mix（20 MHz，源 parquet 列 `fs`）

```text
源：dataset/CJR-mix/{train|val}/*.parquet
列：iq, infer_class, snr, fs, dataset_name
fs = 20_000_000 Hz（HF 数据集中各行均为 20M）
检查：python scripts/inspect_cjr_mix.py --split train --num-samples 5
```

当前 `cjr_mix()` 只读 `iq/infer_class/snr`，**未**把 `fs` 写入 H5；Loader 读 H5 时 `sample_rate_hz=NaN`。应扩展 convert 写 attrs 或 per-sample 列。

#### panoradio_hf（6 kHz，整库常量）

```text
源：external/panoradio_hf/dataset_panoradio_hf.npy + dataset_panoradio_hf_tags.csv
fs = 6_000 Hz（官方 readme：2048 复样本 × 6 kHz ≈ 341 ms）
建议：prepare 写 H5 attrs sampling_rate_hz=6000
```

### 13.3 采样率落盘建议（与 Loader 对齐）

`resmamba_signal_model/data/rfdata.py` 读取顺序：

1. H5 **per-sample** 数据集 `sample_rate_hz` / `sampling_rate`（若存在）
2. 否则 H5 **文件 attrs** `sample_rate_hz` / `sampling_rate`
3. 否则 `NaN`（坐标退化为 `sample_index`）

| 优先级 | 方式 | 适用库 |
|--------|------|--------|
| A | H5 attrs 整库常量 | rml2016_*、radchar、panoradio_hf |
| B | H5 per-sample 列 | cjr_mix（源已有 `fs`）；radar_mod15（若 merge JSON） |
| C | 外部 sidecar JSON | radar_mod15 → `dataset/radar_mod15_sample_rates.json` |
| D | 运行时常量表 | 兜底；不推荐长期依赖 |

**禁止**：为统一 batch 将各库重采样到同一 \(f_s\)；仅记录真实 \(f_s\) 供物理量 / 可选 metadata embedding 使用。


---

## 14. HDF5 结构：必要字段、类型与体积

> **体积事实**：metadata 合计通常 **≪ 1%**；`iq`（float32 `[N,2,L]`）占 **>99%**。
> 省空间优先：**去掉无用列**、**常量改 attrs**、`iq` 保持 **lzf** 压缩；**不要**贸然把 `iq` 改 float16（MAE/物理损失风险大）。

### 14.1 当前 `Writer` 物理列（legacy，非全必要）

`scripts/prepare_datasets.py` → `Writer` 现对每个 H5 **一律创建**：

```text
iq                        [N, 2, L]   float32   # 必要
length                    [N]         int32     # 可省（见 14.2）
dataset_id                [N]         int32     # 可省 → 文件 attrs
task_type_id              [N]         int8      # 可省 → 文件 attrs
snr                       [N]         float32   # 建议保留（可 float16）
mod_label_id              [N]         int32     # 下游必要（可 int16）
canonical_mod_label_id    [N]         int32     # 可选（可 int16 或 load 时算）
source_label_id           [N]         int32     # 部分库必要（可 int16）
receiver_id … capture_date [N] ×5     string    # 7 库可删（恒占位）
```

### 14.2 按阶段：哪些字段**必要**

| 字段 | 预训练 MAE | 下游 train（`*_test.h5`） | 验证/推理（`*_val.h5`） | 7 库预训练池建议 |
|------|:----------:|:-------------------------:|:---------------------:|:----------------:|
| `iq` | **必要** | **必要** | **必要** | **保留** float32 |
| `snr` | 不用（collate 剥离） | 可选（分层评估） | 可选 | **保留**；便于 SNR bucket 报告 |
| `mod_label_id` | 不用 | **必要** | **必要**（metrics） | **保留** |
| `source_label_id` | 不用 | radchar/mod15/cjr **必要**；RML/panoradio 无 | 同左 | **按库保留**；另两库可不建列 |
| `canonical_mod_label_id` | 不用 | tx_modulation 等跨库任务 **建议** | 同左 | finalize **stamp** 或 load 时由 `label_maps` 算 |
| `length` | 可推导 `iq.shape[-1]` | 同左 | 同左 | **删 per-sample 列**；attrs `signal_length=L` |
| `dataset_id` | 不用（`moe_route_stem` 来自文件名） | 聚类任务用 | 同左 | **删 per-sample 列**；attrs `dataset_id` + 文件名 stem |
| `task_type_id` | 不用 |  rarely | rarely | **删**；attrs 即可（当前恒 0） |
| `receiver_id` 等 5 字符串 | 不用 | 不用 | 不用 | **7 库删除**（无真实采集 ID） |

**最小共有 schema（推荐新 H5）**：

```text
iq                  [N, 2, L]   float32
snr                 [N]         float16   # 或 float32；dB 量级 float16 通常够用
mod_label_id        [N]         int16     # -1 表示缺失
source_label_id     [N]         int16     # 仅 radchar / radar_mod15 / cjr_mix 文件写入
canonical_mod_label_id [N]      int16     # 可选；也可仅 stamp 后写入

@attrs（替代 per-sample 冗余）:
  signal_length, dataset_id, task_type_id, sampling_rate_hz（若整库常量）
  source_path, scale_policy, channel_axis=1, signal_contract_version
```

`radar_mod15` 若需逐样本 \(f_s\)：另建 `sample_rate_hz [N] float32` 或 sidecar JSON（§13.2），**不要**强行与其他库合并为同一列类型。

### 14.3 推荐 dtype（在保证 Loader 安全前提下尽量小）

| 字段 | 现 Writer | 推荐 | 说明 |
|------|-----------|------|------|
| `iq` | float32 | **float32** | 主存储；已 `compression="lzf"`；float16 需专门验收 |
| `length` | int32 × N | **attrs `signal_length`** | `rfdata` 已在全文件等长时跳过读 `length` 列 |
| `dataset_id` | int32 × N | **uint8 attrs** | 当前 id ≤ 31；不必 per-sample |
| `task_type_id` | int8 × N | **uint8 attrs** | 7 库恒 0 |
| `snr` | float32 | **float16** 或 float32 | RML 等多为整 dB 步进；float16 省 50%；radar_mod15 名义 10.0 无影响 |
| `mod_label_id` | int32 | **int16** | 7 库类数 ≤ 18；Loader 读入转 int64，无问题 |
| `source_label_id` | int32 | **int16** | 同上 |
| `canonical_mod_label_id` | int32 | **int16** | 全局 ontology 规模有限；或省略列运行时映射 |
| `sample_rate_hz` | （多未写） | **float32 attrs** 或 float32×N | 常量库用 attrs；radar_mod15 逐样本用 float32 列 |
| 5× capture 字符串 | UTF-8 × N | **不存** | 7 库零信息；每 500k 行约省 **数十 MB** 且无 loss |

**Loader 兼容性**：`rfdata._read_int_column` 会把整型列 `astype(int64)`；H5 存 int8/int16 **安全**。
字符串列若不存在，`CAPTURE_METADATA_KEYS` 自动填 `__missing__`。

### 14.4 体积估算（metadata vs iq）

以 `N=500{,}000`, `L=1024` 为例：

```text
iq float32        ≈ 500k × 2 × 1024 × 4 B  ≈ 3.8 GiB（主导项）
legacy metadata   ≈ 500k × (4+4+1+4+4+4+4 + 5×~12) B  ≈ 15–60 MiB
推荐 metadata     ≈ 500k × (2+2+2+2) B               ≈ 4 MiB（仅 snr+3 标签 int16/f16）
```

**结论**：删占位字符串、常量改 attrs、标签改 int16，可省 **~10–50 MiB/大文件**；
相对 iq 占比小，但 schema 更干净。**不要**为省 metadata 牺牲 `iq` float32 精度。

### 14.5 预训练 7 库：哪些列**真有语义**（复查）

| 列 | 7 库是否均有语义 | 现状 |
|----|------------------|------|
| `iq` | ✓ | 唯一波形；必有 |
| `snr` | ✓ | 源 metadata 或 radar_mod15 名义 10.0 |
| `mod_label_id` | ✓ | 7 库均有库内类 ID |
| `source_label_id` | **否** | 仅 radchar、radar_mod15、cjr_mix |
| `canonical_mod_label_id` | **finalize 后** | stamp 或 load 时映射 |
| `length` / `dataset_id` / `task_type_id` | 可 attrs | per-sample 冗余 |
| `receiver_id` … `capture_date` | **否** | 恒占位 → **建议删除** |

### 14.6 文件级 attrs（因库而异）

| attr | 7 库是否都有 |
|------|-------------|
| `source_path`, `scale_policy`, `quality_filter`, `channel_axis`, `signal_contract_version`, `missing_metadata_marker` | ✓（Writer 默认写入） |
| `signal_length`, `dataset_id` | **建议新增**（替代 per-sample 列） |
| `canonical_iq` | 仅 `--canonical-iq` 启用时 |
| `sampling_rate_hz` | **radchar** attrs ✓；rml/panoradio/cjr **建议补 attrs**；radar_mod15 见 sidecar JSON |
| `snr_policy`, `snr_db_nominal` | 仅 **radar_mod15** |
| `radchar_signal_types` | 仅 **radchar** |
| `modulation_ontology_version` | `stamp_semantic_namespaces` 之后 |

---

## 15. 训练输入（与 configs/pretrain.yaml 一致）

模型 batch（经 collate firewall 后）：

```text
iq, sample_mask, length, sample_rate_hz（若 attrs/列有）, moe_route_stem
```

**不含**：mod_label_id、dataset_id、snr 等（预训练剥离，防泄漏）。

归一化与物理辅助量在 `SignalFoundationModel` 前向内完成。

---

## 17. 执行环境、purge 与 H5 v2 预计算（训练加速）

### 17.1 原则

1. **源数据只读**：`dataset/external/`、`NON_EMITTER/` 等原始 pickle/CSV/parquet **永不删除**。
2. **先 purge 旧 H5**：重建前删除不符合 schema 的 `dataset/h5/{stem}_*.h5`，释放磁盘。
3. **离线尽一切可能**：joint_energy 归一化、SNR、采样率、RevIN 统计（`norm_scale` / `amp_aux`）**写入 H5**；训练读入后**跳过在线 RevIN 统计**。
4. **环境**：数据处理在 conda **`won`** 中进行；CPU **最多 22 进程**并行（按数据集）；joint_energy 预计算可用 **GPU（`--device cuda`）**。

### 17.2 推荐命令（预训练 7 库）

```bash
cd /root/autodl-tmp/ResMamba_Signal_Model
source scripts/env.sh          # conda activate won + PYTHONPATH + RFDATA_ROOT

# 1) 删除旧 H5（仅 h5/，不碰源数据）
# 2) canonical_iq 定长档 + H5 v2 预计算 + 最多 22 路并行
python scripts/prepare_datasets.py \
  --output "${RFDATA_ROOT}" \
  --datasets rml2016_04c rml2016_10a rml2016_10b radchar radar_mod15 cjr_mix panoradio_hf \
  --pretrain-h5-v2 \
  --purge-h5 \
  --canonical-iq \
  --jobs 22 \
  --device cuda \
  --precompute-chunk 512

# radar_mod15 逐样本 fs 索引（可选，与 H5 内 sample_rate_hz 列互补）
python scripts/build_radar_mod15_sample_rates.py \
  --output dataset/radar_mod15_sample_rates.json
```

`--pretrain-h5-v2` 自动：`--canonical-iq`（若未指定）、`--purge-h5`、默认 `--jobs=min(22, CPU)`。

### 17.3 H5 v2 schema（`h5_schema_version=2`）

```text
iq                 [N,2,L]  float32   # 已 joint_energy 归一化（训练主输入）
revin_mean         [N,2]    float16   # RevIN 去均值（denorm / MAE 目标）
snr                [N]      float16
sample_rate_hz     [N]      float32
mod_label_id       [N]      int16
source_label_id    [N]      int16     # 部分库
canonical_mod_label_id [N]  int16     # finalize stamp
norm_scale         [N]      float16
log_scale/log_peak/papr_preclip/scale_gap  [N]  float16  # amp_aux

@attrs: h5_schema_version, iq_preprocessed=joint_energy, signal_length,
        dataset_id, task_type_id, sampling_rate_hz（整库常量时）
```

**不含**：capture 字符串占位列、per-sample `length`/`dataset_id` 冗余、emitter 个体列。

### 17.4 训练侧配合

* Loader：`iq_normalize: none`（`configs/pretrain.yaml` 已设）。
* H5 v2：`iq` 已是归一化波形；batch 带 `iq_preprocessed=True` + 预存 RevIN 统计时，模型 **跳过在线 `revin.normalize`**。
* 仍在线：physics / patch 级量（Tokenizer 内）、MAE mask、增广。

### 17.5 并行策略

| 层级 | 机制 | 默认 |
|------|------|------|
| 跨数据集 | `ProcessPoolExecutor`，`--jobs` | v2 时 `min(22, n_cpu)` |
| joint_energy 预计算 | PyTorch batch，`--device cuda` | chunk=256~512 |
| 源数据 | 只读，不删 | — |

### 17.6 工程摘要

$$
\boxed{
\text{iq（已 joint\_energy）}
+
\text{snr + sample\_rate\_hz}
+
\text{mod/source/canonical 标签（int16）}
+
\text{RevIN 统计 + amp\_aux（float16）}
}
$$

### 在线计算（训练时仍做）

$$
\boxed{
\text{physics / patch 物理}
+
\text{MAE mask / 增广}
+
A,\phi,f_i,\text{PSD},\ldots
}
$$

### 不做

$$
\boxed{
\text{删除源数据}
\quad
\text{跳过 purge 直接覆盖混 schema H5}
\quad
\text{global z-score}
\quad
\text{forced 统一 }f_s
\quad
\text{crop 先于 split}
}
$$

### 数据链路

```text
源数据（只读，不删）
      ↓
purge 旧 H5（--purge-h5）
      ↓
won 环境 + CPU 22 并行 / GPU 预计算
      ↓
canonical_iq 定长 + joint_energy 写入 iq
      ↓
snr / fs / RevIN 统计写入 H5 v2
      ↓
train/test/val 7:2:1
      ↓
训练：Loader 直读预计算字段，跳过在线 RevIN 统计
```

**本版为 ResMamba 预训练池可直接实施的标准；H5 v2 以 §17 为准。**

python scripts/train.py --stage stage2 --config configs/stage2.yaml \
  --init-from runs/experiments/pretrain_fast_fullval_20260829_215033/ckpts/best.ckpt
autodl-tmp/ResMamba_Signal_Model/runs/experiments/pretrain_fast_fullval_20260829_215033/ckpts/best.ckpt