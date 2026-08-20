# 72 小时 / 6×5090 实验门控

**未跑满本门控（窗口 A→D，含 3 seeds 与 unified / dataset-specific 分列表）前，不得宣称 SOTA。**  
本仓库现状只证明「脚本可启动」，不证明「已经达到 SOTA」。

编排入口：

```bash
python scripts/run_sota_gate.py --dry-run
python scripts/run_sota_gate.py --dry-run --windows A,B --gpus 0,1,2,3,4,5
python scripts/run_sota_gate.py --smoke          # tiny + synthetic，1 个 gate 调用 train.py
# 正式 GPU（需显式窗口；不要无参 --execute）
python scripts/run_sota_gate.py --execute --windows A --gpus 0,1,2,3,4,5
python scripts/run_sota_gate.py --execute --windows A,B,C,D --gpus 0,1,2,3,4,5 --fail-fast
python scripts/run_sota_gate.py --aggregate
```

无 `--windows` / `--jobs` 的 `--execute` 会被拒绝，避免误启 72h。

## 窗口

| 窗口 | 时间 | 内容 | 失败即停 |
| --- | --- | --- | --- |
| A_validity | 0–8h | `validity_pytest` → 当前架构 pretrain → stage2 | 是。不通过则停止后续消融 |
| B_ablation | 8–28h | encoder 1/3/5、MAE-only vs 三目标、z_general vs UTI v2、相位插件、低秩头、查询 decoder vs legacy | 依赖 A |
| C_confirm | 28–60h | `selected.json` 最优两套 × seeds 0/1/2 | 依赖 B |
| D_eval | 60–72h | 聚类 / 开集 / 增量 / 生成；跨域探测缺数据则跳过 | 依赖 C |

encoder=5、三目标、UTI v2、相位关闭、查询 decoder 与 A 基线相同，B 中以 **alias** 复用，不重跑。

## 依赖条件

每个 train job 的 `requires` 在启动前检查，缺一则 **blocked**（`optional` 的跨域探测改为 skip）：

| 条件 | 含义 | 本机常见状态 |
| --- | --- | --- |
| `mamba_kernel` | 官方 `mamba-ssm` CUDA kernel | **BLOCK**：无 kernel，正式 `configs/model.yaml` 不能训 |
| `split_manifest` | `dataset/split_manifest.json` 存在且哈希未变 | **BLOCK**：尚未写出 immutable manifest |
| `h5_stamp` | H5 含 `canonical_mod_label_id` / `global_emitter_id` | **BLOCK**：真实 H5 未 stamp；训练虽可用 `label_maps` 回填，正式门控仍要求 stamp |
| `cross_domain_data` | 水声/IMU 小基准 | 缺则跳过 `probe_cross_domain`，**禁止下载外部大数据** |

`--smoke` / `--skip-requires` 可绕过检查，仅用于 tiny 冒烟，不能当正式结果。

## 输出目录

Lightning run：

```
runs/experiments/sota_gate/<window>/<job_id>/seed<k>/
  config.yaml  train.log  train_state.json  ckpts/best.ckpt  csv/  tb/
```

门控索引：

```
runs/sota_gate/
  dry_run.txt       命令矩阵
  blockers.json     本机 blocker
  index.json        最近一次 execute 状态
  selected.json     B 之后的两套优胜配置（未完成时为占位）
  metrics.json      均值±std
  metrics.md        人读表
```

`--init-from` 指向依赖 job 的 `ckpts/best.ckpt`。依赖失败则后续 job 不再启动（`--fail-fast`，默认开）。

## 指标表格式

`metrics.md` / `metrics.json` 必须分列：

- **unified**：`val/f1_modulation`、`val/acc_emitter`、`val/multitask_geomean`（全数据混合）
- **dataset-specific**：`val/f1_modulation/<dataset>`、`val/acc_emitter/<dataset>`（同一 ckpt 按数据集拆开，不是 dataset-specific fine-tune 本身）

多 seed 写作 `mean ± std (n=k)`。`n=1` 时 std=0。`sota_claim_allowed` 在满 72h 完成前恒为 `false`。

dataset-specific **fine-tune** 与 **zero/few-shot**：C 窗口 3 seeds 的 stage2 即冻结探测（few-shot/linear probe 口径）；按数据集拆开的列是 eval 口径。若要真正按库微调，在 B 选模后对 `confirm_slot_*_stage2.yaml` 收窄 `task_pools`，仍须 3 seeds，并与 unified 分开写。

## YAML

薄 overlay 在 `configs/experiments/`，继承 `configs/pretrain.yaml` / `stage2.yaml` / `stage3.yaml` / `continual.yaml`。

| 文件 | 作用 |
| --- | --- |
| `validity_*.yaml` | A 基线 |
| `overlays/enc1.yaml` `enc3.yaml` | 深度消融 |
| `overlays/mae_only.yaml` | 只留 MAE |
| `overlays/z_general.yaml` | `use_specialist_views: false` |
| `overlays/phase_on.yaml` | tokenizer 相位插件 |
| `overlays/low_rank_head.yaml` | `low_rank_prototype: true`（默认仍旧头） |
| `overlays/legacy_decoder.yaml` | 对照查询 decoder |
| `confirm_slot_{a,b}_*.yaml` | C 占位；B 结束后 `selected.json` 会改写 `--config` |
| `eval_*.yaml` `probe_cross_domain.yaml` | D |

## 本机 blocker 列表（现状）

1. **无 mamba kernel**（`mamba-ssm` 不可用）→ 正式宽度 `d_model=640` 不能训；tiny/fallback 仅冒烟。
2. **无 `dataset/split_manifest.json`** → 不能证明 group-held-out / 文件哈希冻结。数据准备时用 `python scripts/prepare_datasets.py --verify-splits`（会哈希全部 H5）。
3. **真实 H5 未 stamp semantic 列** → `canonical_mod_label_id` / `global_emitter_id` 不在文件内。准备步骤：`python scripts/prepare_datasets.py --stamp-semantic-labels`（会改 H5，须在写 manifest 之前）。
4. **参数预算未达标** → 总参约 76.8M、识别头约 25.9M；`low_rank_prototype` 默认关闭以免破坏现有 ckpt。
5. **72h GPU 矩阵尚未执行** → 不得把 tiny smoke 或单次 stage2 写成 SOTA。
6. **无水声/IMU 小基准** → 跨域只做探测，且当前会 skip；72h 内不宣称声纳/导航 SOTA。

## 6×5090 建议

```bash
# A：单作业占满 6 卡 DDP
python scripts/run_sota_gate.py --execute --windows A --gpus 0,1,2,3,4,5

# B：一卡一消融（脚本按 job 轮转 CUDA_VISIBLE_DEVICES；当前编排是顺序 fail-fast，
#    若要并行请按 dry-run 矩阵手工拆到 6 个 tmux）
python scripts/run_sota_gate.py --dry-run --windows B --gpus 0,1,2,3,4,5
```

当前 `--execute` 为顺序调度 + 失败即停，避免 6 路 DDP 抢卡。dry-run 会打印 `CUDA_VISIBLE_DEVICES` 建议。并行启动时每个消融复制一条 `--jobs abl_enc1_pretrain` 到对应 GPU。

## 低秩头

`model.low_rank_prototype` 默认 `false`，Cosine/MLP 结构与旧 ckpt 一致。消融 `abl_low_rank_stage2` 才打开 `prototype_rank: 64`。不要把该开关改成默认以免测试和旧权重变红。
