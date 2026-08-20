from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "runs" / "sota_gate"
TRAIN_PY = ROOT / "scripts" / "train.py"
SEMANTIC_COLUMNS = ("canonical_mod_label_id", "global_emitter_id")
UNIFIED_METRIC_KEYS = (
    "monitor_value",
    "val/multitask_geomean",
    "val/f1_modulation",
    "val/acc_modulation",
    "val/macro_f1_modulation",
    "val/macro_acc_modulation",
    "val/f1_emitter",
    "val/acc_emitter",
    "val/macro_f1_emitter",
    "val/macro_acc_emitter",
    "val/nmi",
    "val/ari",
    "val/ssim",
    "val/ssim_prediction",
    "val/ssim_imputation",
    "val/osr_auroc",
    "val/osr_aupr",
    "val/osr_fpr95",
)
VALIDITY_PYTEST = (
    "tests/test_model_no_leakage.py",
    "tests/test_split_manifest.py",
    "tests/test_data_label_namespaces.py",
    "tests/test_clustering_labels.py",
    "tests/test_wisig_group_split.py",
    "tests/test_real_h5_slice.py",
)

WINDOWS = {
    "A": ("A_validity", "0-8h", "有效性基线：无泄漏/划分/标签，不通过则停止消融"),
    "B": ("B_ablation", "8-28h", "短消融：encoder 1/3/5、MAE-only vs 三目标、z_general vs UTI v2、相位插件、低秩头、查询 decoder"),
    "C": ("C_confirm", "28-60h", "最优两套 × 3 seeds；unified 与 dataset-specific 分列报告"),
    "D": ("D_eval", "60-72h", "聚类/开集/增量/生成 + 可选跨域探测"),
}


@dataclass(frozen=True)
class GateJob:
    id: str
    window: str
    kind: str
    description: str
    stage: str | None = None
    config: str | None = None
    task: str | None = None
    seeds: tuple[int, ...] = (0,)
    extra_args: tuple[str, ...] = ()
    depends_on: tuple[str, ...] = ()
    init_from_job: str | None = None
    requires: tuple[str, ...] = ()
    optional: bool = False
    gpu_mode: str = "one"
    alias_of: str | None = None
    metrics: tuple[str, ...] = UNIFIED_METRIC_KEYS
    argv: tuple[str, ...] = ()


@dataclass
class Blocker:
    name: str
    ok: bool
    detail: str


@dataclass
class PlannedCommand:
    job_id: str
    seed: int
    window: str
    kind: str
    run_name: str
    argv: list[str]
    env: dict[str, str] = field(default_factory=dict)
    depends_on: tuple[str, ...] = ()
    requires: tuple[str, ...] = ()
    optional: bool = False
    gpu_mode: str = "one"
    alias_of: str | None = None
    description: str = ""
    init_from: str | None = None


def _job(**kwargs: Any) -> GateJob:
    return GateJob(**kwargs)


def build_gate_jobs() -> list[GateJob]:
    """72h / 6×5090 门控矩阵。默认不执行；由 CLI 显式 --execute / --smoke 启动。"""
    full = ("mamba_kernel", "split_manifest", "h5_stamp")
    train_metrics = UNIFIED_METRIC_KEYS
    jobs: list[GateJob] = [
        _job(
            id="validity_pytest",
            window="A_validity",
            kind="pytest",
            description="泄漏/划分/标签防火墙单测；失败即停后续 SOTA 消融",
            argv=("python", "-m", "pytest", *VALIDITY_PYTEST, "-q"),
            requires=(),
        ),
        _job(
            id="validity_pretrain",
            window="A_validity",
            kind="train",
            stage="pretrain",
            config="configs/experiments/validity_pretrain.yaml",
            description="修复后当前架构预训练基线（encoder=5，三目标，UTI v2）",
            depends_on=("validity_pytest",),
            requires=full,
            gpu_mode="all",
            metrics=("monitor_value", "val/ssim", "val/impute_mse"),
        ),
        _job(
            id="validity_stage2",
            window="A_validity",
            kind="train",
            stage="stage2",
            config="configs/experiments/validity_stage2.yaml",
            description="冻结骨干探测：主任务 macro-F1 / 跨数据集口径",
            depends_on=("validity_pretrain",),
            init_from_job="validity_pretrain",
            requires=full,
            gpu_mode="all",
            metrics=train_metrics,
        ),
        _job(
            id="abl_enc5",
            window="B_ablation",
            kind="alias",
            alias_of="validity_stage2",
            description="encoder=5 与有效性基线相同，复用 validity_stage2，不重跑",
            depends_on=("validity_stage2",),
        ),
        _job(
            id="abl_triple",
            window="B_ablation",
            kind="alias",
            alias_of="validity_stage2",
            description="三目标预训练与基线相同，复用 validity_stage2",
            depends_on=("validity_stage2",),
        ),
        _job(
            id="abl_uti_v2",
            window="B_ablation",
            kind="alias",
            alias_of="validity_stage2",
            description="UTI v2 与基线相同，复用 validity_stage2",
            depends_on=("validity_stage2",),
        ),
        _job(
            id="abl_phase_off",
            window="B_ablation",
            kind="alias",
            alias_of="validity_stage2",
            description="关闭相位插件（默认），复用 validity_stage2",
            depends_on=("validity_stage2",),
        ),
        _job(
            id="abl_query_decoder",
            window="B_ablation",
            kind="alias",
            alias_of="validity_stage2",
            description="查询 decoder 为默认，复用 validity_stage2",
            depends_on=("validity_stage2",),
        ),
        _job(
            id="abl_enc1_pretrain",
            window="B_ablation",
            kind="train",
            stage="pretrain",
            config="configs/experiments/abl_enc1_pretrain.yaml",
            description="encoder 1 层预训练",
            depends_on=("validity_stage2",),
            requires=full,
            metrics=("monitor_value", "val/ssim"),
        ),
        _job(
            id="abl_enc1_stage2",
            window="B_ablation",
            kind="train",
            stage="stage2",
            config="configs/experiments/abl_enc1_stage2.yaml",
            description="encoder 1 层冻结探测",
            depends_on=("abl_enc1_pretrain",),
            init_from_job="abl_enc1_pretrain",
            requires=full,
            metrics=train_metrics,
        ),
        _job(
            id="abl_enc3_pretrain",
            window="B_ablation",
            kind="train",
            stage="pretrain",
            config="configs/experiments/abl_enc3_pretrain.yaml",
            description="encoder 3 层预训练",
            depends_on=("validity_stage2",),
            requires=full,
            metrics=("monitor_value", "val/ssim"),
        ),
        _job(
            id="abl_enc3_stage2",
            window="B_ablation",
            kind="train",
            stage="stage2",
            config="configs/experiments/abl_enc3_stage2.yaml",
            description="encoder 3 层冻结探测",
            depends_on=("abl_enc3_pretrain",),
            init_from_job="abl_enc3_pretrain",
            requires=full,
            metrics=train_metrics,
        ),
        _job(
            id="abl_mae_only_pretrain",
            window="B_ablation",
            kind="train",
            stage="pretrain",
            config="configs/experiments/abl_mae_only_pretrain.yaml",
            description="MAE-only 预训练（关闭结构/潜变量/UTI 读出损失）",
            depends_on=("validity_stage2",),
            requires=full,
            metrics=("monitor_value", "val/ssim"),
        ),
        _job(
            id="abl_mae_only_stage2",
            window="B_ablation",
            kind="train",
            stage="stage2",
            config="configs/experiments/abl_mae_only_stage2.yaml",
            description="MAE-only 冻结探测",
            depends_on=("abl_mae_only_pretrain",),
            init_from_job="abl_mae_only_pretrain",
            requires=full,
            metrics=train_metrics,
        ),
        _job(
            id="abl_z_general_pretrain",
            window="B_ablation",
            kind="train",
            stage="pretrain",
            config="configs/experiments/abl_z_general_pretrain.yaml",
            description="z_general only（关闭 UTI specialist views）",
            depends_on=("validity_stage2",),
            requires=full,
            metrics=("monitor_value", "val/ssim"),
        ),
        _job(
            id="abl_z_general_stage2",
            window="B_ablation",
            kind="train",
            stage="stage2",
            config="configs/experiments/abl_z_general_stage2.yaml",
            description="z_general only 冻结探测",
            depends_on=("abl_z_general_pretrain",),
            init_from_job="abl_z_general_pretrain",
            requires=full,
            metrics=train_metrics,
        ),
        _job(
            id="abl_phase_on_pretrain",
            window="B_ablation",
            kind="train",
            stage="pretrain",
            config="configs/experiments/abl_phase_on_pretrain.yaml",
            description="开启 tokenizer 相位/共轭插件",
            depends_on=("validity_stage2",),
            requires=full,
            metrics=("monitor_value", "val/ssim"),
        ),
        _job(
            id="abl_phase_on_stage2",
            window="B_ablation",
            kind="train",
            stage="stage2",
            config="configs/experiments/abl_phase_on_stage2.yaml",
            description="相位插件冻结探测",
            depends_on=("abl_phase_on_pretrain",),
            init_from_job="abl_phase_on_pretrain",
            requires=full,
            metrics=train_metrics,
        ),
        _job(
            id="abl_legacy_decoder_pretrain",
            window="B_ablation",
            kind="train",
            stage="pretrain",
            config="configs/experiments/abl_legacy_decoder_pretrain.yaml",
            description="旧 skip/FiLM 解码器对照（非查询 decoder）",
            depends_on=("validity_stage2",),
            requires=full,
            metrics=("monitor_value", "val/ssim"),
        ),
        _job(
            id="abl_legacy_decoder_stage2",
            window="B_ablation",
            kind="train",
            stage="stage2",
            config="configs/experiments/abl_legacy_decoder_stage2.yaml",
            description="旧解码器冻结探测",
            depends_on=("abl_legacy_decoder_pretrain",),
            init_from_job="abl_legacy_decoder_pretrain",
            requires=full,
            metrics=train_metrics,
        ),
        _job(
            id="abl_low_rank_stage2",
            window="B_ablation",
            kind="train",
            stage="stage2",
            config="configs/experiments/abl_low_rank_stage2.yaml",
            description="低秩原型头显式 rank=64（与默认配置对照，验证 H7）",
            depends_on=("validity_pretrain", "validity_stage2"),
            init_from_job="validity_pretrain",
            requires=full,
            metrics=train_metrics,
        ),
        _job(
            id="abl_bidir_share_pretrain",
            window="B_ablation",
            kind="train",
            stage="pretrain",
            config="configs/experiments/abl_bidir_share_pretrain.yaml",
            description="BiMamba2 双向权重共享预训练",
            depends_on=("validity_stage2",),
            requires=full,
            metrics=("monitor_value", "val/ssim"),
        ),
        _job(
            id="abl_bidir_share_stage2",
            window="B_ablation",
            kind="train",
            stage="stage2",
            config="configs/experiments/abl_bidir_share_stage2.yaml",
            description="双向权重共享冻结探测",
            depends_on=("abl_bidir_share_pretrain",),
            init_from_job="abl_bidir_share_pretrain",
            requires=full,
            metrics=train_metrics,
        ),
        _job(
            id="confirm_slot_a_pretrain",
            window="C_confirm",
            kind="train",
            stage="pretrain",
            config="configs/experiments/confirm_slot_a_pretrain.yaml",
            description="消融优胜配置 A：3 seeds 预训练（slot 在 B 结束后物化）",
            seeds=(0, 1, 2),
            depends_on=("abl_enc1_stage2", "abl_enc3_stage2", "abl_mae_only_stage2", "abl_z_general_stage2", "abl_phase_on_stage2", "abl_legacy_decoder_stage2", "abl_low_rank_stage2", "abl_bidir_share_stage2"),
            requires=full,
            gpu_mode="one",
            metrics=("monitor_value", "val/ssim"),
        ),
        _job(
            id="confirm_slot_a_stage2",
            window="C_confirm",
            kind="train",
            stage="stage2",
            config="configs/experiments/confirm_slot_a_stage2.yaml",
            description="优胜配置 A：3 seeds unified 冻结探测",
            seeds=(0, 1, 2),
            depends_on=("confirm_slot_a_pretrain",),
            init_from_job="confirm_slot_a_pretrain",
            requires=full,
            metrics=train_metrics,
        ),
        _job(
            id="confirm_slot_b_pretrain",
            window="C_confirm",
            kind="train",
            stage="pretrain",
            config="configs/experiments/confirm_slot_b_pretrain.yaml",
            description="消融次优配置 B：3 seeds 预训练",
            seeds=(0, 1, 2),
            depends_on=("abl_enc1_stage2", "abl_enc3_stage2", "abl_mae_only_stage2", "abl_z_general_stage2", "abl_phase_on_stage2", "abl_legacy_decoder_stage2", "abl_low_rank_stage2", "abl_bidir_share_stage2"),
            requires=full,
            metrics=("monitor_value", "val/ssim"),
        ),
        _job(
            id="confirm_slot_b_stage2",
            window="C_confirm",
            kind="train",
            stage="stage2",
            config="configs/experiments/confirm_slot_b_stage2.yaml",
            description="次优配置 B：3 seeds unified 冻结探测",
            seeds=(0, 1, 2),
            depends_on=("confirm_slot_b_pretrain",),
            init_from_job="confirm_slot_b_pretrain",
            requires=full,
            metrics=train_metrics,
        ),
        _job(
            id="eval_clustering",
            window="D_eval",
            kind="train",
            stage="stage3",
            task="clustering",
            config="configs/experiments/eval_clustering.yaml",
            description="冻结最终骨干后的无监督聚类（NMI/ARI，标签只进评估）",
            depends_on=("confirm_slot_a_stage2",),
            init_from_job="confirm_slot_a_stage2",
            requires=full,
            metrics=("val/nmi", "val/ari", "val/nmi_within_domain"),
        ),
        _job(
            id="eval_openset",
            window="D_eval",
            kind="train",
            stage="stage2",
            config="configs/experiments/eval_openset.yaml",
            description="开集四象限：harvest val/osr_*（AUROC/AUPR/FPR95）",
            depends_on=("confirm_slot_a_stage2",),
            init_from_job="confirm_slot_a_stage2",
            requires=full,
            metrics=("val/osr_auroc", "val/osr_aupr", "val/osr_fpr95", "val/f1_modulation"),
        ),
        _job(
            id="eval_continual",
            window="D_eval",
            kind="train",
            stage="continual",
            config="configs/experiments/eval_continual.yaml",
            description="类增量：共享低秩 adapter + 原型吸收",
            depends_on=("confirm_slot_a_stage2",),
            init_from_job="confirm_slot_a_stage2",
            requires=full,
            metrics=("val/multitask_geomean", "val/f1_modulation", "val/acc_emitter"),
        ),
        _job(
            id="eval_prediction",
            window="D_eval",
            kind="train",
            stage="stage3",
            task="prediction",
            config="configs/experiments/eval_generative.yaml",
            description="suffix 预测",
            depends_on=("confirm_slot_a_stage2",),
            init_from_job="confirm_slot_a_stage2",
            requires=full,
            metrics=("val/mse_prediction", "val/ssim_prediction"),
        ),
        _job(
            id="eval_imputation",
            window="D_eval",
            kind="train",
            stage="stage3",
            task="imputation",
            config="configs/experiments/eval_generative.yaml",
            description="span 插补",
            depends_on=("confirm_slot_a_stage2",),
            init_from_job="confirm_slot_a_stage2",
            requires=full,
            metrics=("val/mse_imputation",),
        ),
        _job(
            id="probe_cross_domain",
            window="D_eval",
            kind="train",
            stage="stage2",
            config="configs/experiments/probe_cross_domain.yaml",
            description="可选水声/IMU frozen probe；缺数据则跳过，不下载外部集",
            depends_on=("confirm_slot_a_stage2",),
            init_from_job="confirm_slot_a_stage2",
            requires=("cross_domain_data",),
            optional=True,
            metrics=("val/f1_modulation",),
        ),
    ]
    return jobs


def parse_window_token(token: str) -> str:
    raw = str(token).strip()
    key = raw.split("_", 1)[0].upper()
    if key in WINDOWS:
        return WINDOWS[key][0]
    for _letter, (name, _hours, _desc) in WINDOWS.items():
        if raw == name:
            return name
    raise ValueError(f"未知 window {token!r}，可选 A/B/C/D 或 {tuple(v[0] for v in WINDOWS.values())}")


def filter_jobs(
    jobs: Sequence[GateJob],
    *,
    windows: Sequence[str] | None = None,
    job_ids: Sequence[str] | None = None,
) -> list[GateJob]:
    selected = list(jobs)
    if windows:
        names = {parse_window_token(item) for item in windows}
        selected = [job for job in selected if job.window in names]
    if job_ids:
        keep = {str(item) for item in job_ids}
        selected = [job for job in selected if job.id in keep]
        aliases = {job.alias_of for job in selected if job.alias_of}
        extra = [job for job in jobs if job.id in aliases and job not in selected]
        selected = extra + selected
    return selected


def run_name_for(job: GateJob, seed: int, *, prefix: str = "sota_gate") -> str:
    return f"{prefix}/{job.window}/{job.id}/seed{seed}"


def experiment_dir(root: Path, job: GateJob, seed: int, *, prefix: str = "sota_gate") -> Path:
    return root / "runs" / "experiments" / run_name_for(job, seed, prefix=prefix)


def best_ckpt_for(root: Path, job_id: str, seed: int, jobs: Sequence[GateJob] | None = None) -> Path | None:
    catalog = {job.id: job for job in (jobs or build_gate_jobs())}
    job = catalog.get(job_id)
    if job is None:
        return None
    if job.alias_of:
        return best_ckpt_for(root, job.alias_of, seed, jobs=jobs)
    run_dir = experiment_dir(root, job, seed)
    for candidate in (run_dir / "ckpts" / "best.ckpt", run_dir / "best.ckpt", run_dir / "ckpts" / "last.ckpt"):
        if candidate.is_file():
            return candidate
    return None


def check_mamba_kernel() -> Blocker:
    try:
        from resmamba_signal_model.models.mamba_backbone import get_mamba_runtime_info, mamba2_available
    except Exception as exc:  # pragma: no cover
        return Blocker("mamba_kernel", False, f"无法导入 mamba 运行时: {exc}")
    info = get_mamba_runtime_info()
    ok = bool(mamba2_available()) and not bool(info.get("fallback"))
    detail = f"kernel={info.get('mamba_kernel')} fallback={info.get('fallback')}"
    if not ok:
        detail += "；正式 configs/model.yaml 不能训（require_mamba_kernel=true）"
    return Blocker("mamba_kernel", ok, detail)


def check_split_manifest(root: Path) -> Blocker:
    path = root / "dataset" / "split_manifest.json"
    h5_dir = root / "dataset" / "h5"
    if not path.is_file():
        return Blocker("split_manifest", False, f"缺少 {path}；请在数据准备步骤运行 python scripts/prepare_datasets.py --verify-splits（会哈希全部 H5，勿在门控进程内现算）")
    try:
        from resmamba_signal_model.data.splits import assert_manifest_files_unchanged, load_manifest

        payload = load_manifest(path)
        assert_manifest_files_unchanged(payload, h5_dir)
    except Exception as exc:
        return Blocker("split_manifest", False, f"{path} 校验失败: {exc}")
    return Blocker("split_manifest", True, str(path))


def check_h5_stamp(root: Path, *, sample_files: Sequence[str] | None = None) -> Blocker:
    h5_dir = root / "dataset" / "h5"
    names = list(sample_files or ("rml2016_04c_train.h5", "adsb2_train.h5", "rml2016_10a_train.h5"))
    missing_files = [name for name in names if not (h5_dir / name).is_file()]
    if missing_files:
        return Blocker("h5_stamp", False, f"缺少样例 H5: {missing_files}")
    try:
        import h5py
    except Exception as exc:  # pragma: no cover
        return Blocker("h5_stamp", False, f"无法导入 h5py: {exc}")
    missing_cols: list[str] = []
    for name in names:
        path = h5_dir / name
        with h5py.File(path, "r") as handle:
            absent = [col for col in SEMANTIC_COLUMNS if col not in handle]
            if absent:
                missing_cols.append(f"{name}:{','.join(absent)}")
    if missing_cols:
        return Blocker(
            "h5_stamp",
            False,
            "真实 H5 未写入 semantic 列（训练期虽可用 label_maps 回填，正式门控仍要求 stamp）: " + "; ".join(missing_cols),
        )
    return Blocker("h5_stamp", True, f"已检查 {len(names)} 个文件含 {SEMANTIC_COLUMNS}")


def check_cross_domain_data(root: Path) -> Blocker:
    h5_dir = root / "dataset" / "h5"
    markers = (
        "sonar",
        "sonair",
        "deepship",
        "wolfset",
        "ronin",
        "oxiod",
        "imu",
    )
    hits: list[str] = []
    if h5_dir.is_dir():
        for path in h5_dir.glob("*.h5"):
            lowered = path.name.lower()
            if any(token in lowered for token in markers):
                hits.append(path.name)
    extra = root / "dataset" / "external"
    if extra.is_dir():
        for child in extra.iterdir():
            lowered = child.name.lower()
            if any(token in lowered for token in markers):
                hits.append(f"external:{child.name}")
    if not hits:
        return Blocker("cross_domain_data", False, "未找到水声/IMU 基准；72h 内跳过跨域探测，禁止下载外部大数据")
    return Blocker("cross_domain_data", True, "found " + ", ".join(hits[:8]))


def collect_blockers(root: Path) -> list[Blocker]:
    return [
        check_mamba_kernel(),
        check_split_manifest(root),
        check_h5_stamp(root),
        check_cross_domain_data(root),
        Blocker(
            "sota_claim",
            False,
            "未跑满 72h 门控（A→D、3 seeds、unified 与 dataset-specific 表）前不得宣称 SOTA",
        ),
        Blocker(
            "param_budget",
            False,
            "正式宽度默认 low_rank_prototype=true；总参/头参须用 count_params.py --with-heads 复算后写入 blocker",
        ),
    ]


def blockers_by_name(root: Path) -> dict[str, Blocker]:
    return {item.name: item for item in collect_blockers(root)}


def missing_requirements(job: GateJob, status: dict[str, Blocker]) -> list[Blocker]:
    failed: list[Blocker] = []
    for name in job.requires:
        item = status.get(name)
        if item is None or not item.ok:
            failed.append(item or Blocker(name, False, "未检查"))
    return failed


def _python_exe() -> str:
    return sys.executable or "python"


def build_train_argv(
    job: GateJob,
    *,
    seed: int,
    root: Path,
    profile: str | None = None,
    synthetic: bool = False,
    devices: str | None = None,
    strategy: str | None = None,
    init_from: str | Path | None = None,
    extra: Sequence[str] = (),
    run_prefix: str = "sota_gate",
) -> list[str]:
    if job.config is None or job.stage is None:
        raise ValueError(f"{job.id} 不是 train job")
    argv = [
        _python_exe(),
        str((root / "scripts" / "train.py").resolve()),
        "--stage",
        job.stage,
        "--config",
        str((root / job.config).resolve()),
        "--run-name",
        run_name_for(job, seed, prefix=run_prefix),
        "--seed",
        str(int(seed)),
    ]
    if job.task:
        argv.extend(["--task", job.task])
    if profile:
        argv.extend(["--profile", profile])
    if synthetic:
        argv.append("--synthetic")
    if devices:
        argv.extend(["--devices", str(devices)])
    if strategy:
        argv.extend(["--strategy", strategy])
    if init_from:
        argv.extend(["--init-from", str(init_from)])
    argv.extend(job.extra_args)
    argv.extend(extra)
    return argv


def build_pytest_argv(job: GateJob, *, root: Path) -> list[str]:
    argv = list(job.argv)
    if argv and argv[0] == "python":
        argv[0] = _python_exe()
    return argv


def plan_commands(
    jobs: Sequence[GateJob],
    *,
    root: Path,
    profile: str | None = None,
    synthetic: bool = False,
    gpus: Sequence[int] | None = None,
    extra: Sequence[str] = (),
    run_prefix: str = "sota_gate",
) -> list[PlannedCommand]:
    catalog = {job.id: job for job in build_gate_jobs()}
    planned: list[PlannedCommand] = []
    gpu_list = list(gpus or [])
    gpu_cursor = 0
    all_devices = ",".join(str(item) for item in gpu_list) if gpu_list else None
    for job in jobs:
        for seed in job.seeds:
            env: dict[str, str] = {}
            devices = None
            strategy = None
            if job.kind == "train" and gpu_list:
                if job.gpu_mode == "all":
                    devices = all_devices
                    strategy = "ddp" if len(gpu_list) > 1 else None
                else:
                    devices = "1"
                    env["CUDA_VISIBLE_DEVICES"] = str(gpu_list[gpu_cursor % len(gpu_list)])
                    gpu_cursor += 1
            init_from = None
            if job.init_from_job:
                ckpt = best_ckpt_for(root, job.init_from_job, seed, jobs=list(catalog.values()))
                if ckpt is None and seed != 0:
                    ckpt = best_ckpt_for(root, job.init_from_job, 0, jobs=list(catalog.values()))
                init_from = str(ckpt) if ckpt is not None else f"{{ckpt:{job.init_from_job}:seed{seed}}}"
            if job.kind == "train":
                argv = build_train_argv(
                    job,
                    seed=seed,
                    root=root,
                    profile=profile,
                    synthetic=synthetic,
                    devices=devices,
                    strategy=strategy,
                    init_from=None if (init_from and init_from.startswith("{ckpt:")) else init_from,
                    extra=extra,
                    run_prefix=run_prefix,
                )
                if init_from and init_from.startswith("{ckpt:"):
                    argv.extend(["--init-from", init_from])
            elif job.kind == "pytest":
                argv = build_pytest_argv(job, root=root)
            elif job.kind == "alias":
                argv = ["# alias", job.id, "->", str(job.alias_of)]
            else:
                argv = list(job.argv)
            planned.append(
                PlannedCommand(
                    job_id=job.id,
                    seed=int(seed),
                    window=job.window,
                    kind=job.kind,
                    run_name=run_name_for(job, seed, prefix=run_prefix),
                    argv=argv,
                    env=env,
                    depends_on=job.depends_on,
                    requires=job.requires,
                    optional=job.optional,
                    gpu_mode=job.gpu_mode,
                    alias_of=job.alias_of,
                    description=job.description,
                    init_from=init_from,
                )
            )
    return planned


def format_command_matrix(planned: Sequence[PlannedCommand], *, blockers: Sequence[Blocker] | None = None) -> str:
    lines = [
        "# ResMamba 72h/6×5090 实验门控命令矩阵",
        "# 未跑满 A→D 且 3 seeds 复验前，不得宣称 SOTA。",
        f"# generated_at={datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}",
        "",
    ]
    if blockers:
        lines.append("## 本机 blocker")
        for item in blockers:
            mark = "PASS" if item.ok else "BLOCK"
            lines.append(f"- [{mark}] {item.name}: {item.detail}")
        lines.append("")
    current_window = None
    for item in planned:
        if item.window != current_window:
            current_window = item.window
            hours = next((row[1] for row in WINDOWS.values() if row[0] == item.window), "")
            lines.append(f"## {item.window} ({hours})")
        env = " ".join(f"{key}={value}" for key, value in item.env.items())
        prefix = f"{env} " if env else ""
        cmd = " ".join(item.argv)
        dep = f" depends_on={list(item.depends_on)}" if item.depends_on else ""
        req = f" requires={list(item.requires)}" if item.requires else ""
        opt = " optional" if item.optional else ""
        alias = f" alias_of={item.alias_of}" if item.alias_of else ""
        lines.append(f"- {item.job_id} seed={item.seed} [{item.kind}]{alias}{dep}{req}{opt}")
        lines.append(f"    {item.description}")
        lines.append(f"    run={item.run_name}")
        if item.init_from:
            lines.append(f"    init_from={item.init_from}")
        lines.append(f"    {prefix}{cmd}")
    lines.append("")
    lines.append("输出目录: runs/experiments/sota_gate/<window>/<job_id>/seed<k>/")
    lines.append("门控索引: runs/sota_gate/{index.json,metrics.json,metrics.md,blockers.json}")
    lines.append("指标表: 均值±std；unified 用 val/f1_modulation，dataset-specific 用 val/f1_modulation/<dataset>")
    return "\n".join(lines) + "\n"


def _is_number(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def load_run_metrics(run_dir: Path) -> dict[str, float]:
    metrics: dict[str, float] = {}
    state_path = run_dir / "train_state.json"
    if state_path.is_file():
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        monitor = payload.get("monitor")
        value = payload.get("monitor_value", payload.get("best_model_score"))
        if _is_number(value):
            metrics["monitor_value"] = float(value)
            if monitor:
                metrics[str(monitor)] = float(value)
    csv_candidates = list(run_dir.glob("csv/**/metrics.csv")) + list(run_dir.glob("**/metrics.csv"))
    if csv_candidates:
        path = csv_candidates[0]
        with path.open("r", encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in reversed(rows):
            for key, raw in row.items():
                if key in ("epoch", "step") or not _is_number(raw):
                    continue
                if key not in metrics:
                    metrics[key] = float(raw)
            if any(key.startswith("val/") for key in row):
                break
    return metrics


def split_unified_and_specific(metrics: dict[str, float]) -> tuple[dict[str, float], dict[str, float]]:
    unified: dict[str, float] = {}
    specific: dict[str, float] = {}
    for key, value in metrics.items():
        if key.startswith("val/") and key.count("/") >= 2:
            specific[key] = value
        else:
            unified[key] = value
    return unified, specific


def _mean_std(values: Sequence[float]) -> dict[str, float]:
    cleaned = [float(item) for item in values if math.isfinite(float(item))]
    if not cleaned:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    mean = float(statistics.fmean(cleaned))
    std = float(statistics.stdev(cleaned)) if len(cleaned) > 1 else 0.0
    return {"mean": mean, "std": std, "n": len(cleaned)}


def aggregate_job_metrics(root: Path, jobs: Sequence[GateJob]) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "sota_claim_allowed": False,
        "note": "未跑满 72h 门控前不得宣称 SOTA。unified 与 dataset-specific 必须分列。",
        "jobs": {},
    }
    for job in jobs:
        if job.kind == "alias":
            continue
        seed_metrics = []
        for seed in job.seeds:
            run_dir = experiment_dir(root, job, seed)
            if not run_dir.exists():
                continue
            seed_metrics.append(load_run_metrics(run_dir))
        if not seed_metrics:
            continue
        unified_series: dict[str, list[float]] = {}
        specific_series: dict[str, list[float]] = {}
        for metrics in seed_metrics:
            unified, specific = split_unified_and_specific(metrics)
            for key, value in unified.items():
                unified_series.setdefault(key, []).append(value)
            for key, value in specific.items():
                specific_series.setdefault(key, []).append(value)
        payload["jobs"][job.id] = {
            "window": job.window,
            "n_seeds_found": len(seed_metrics),
            "n_seeds_planned": len(job.seeds),
            "unified": {key: _mean_std(vals) for key, vals in unified_series.items()},
            "dataset_specific": {key: _mean_std(vals) for key, vals in specific_series.items()},
        }
    return payload


def format_mean_std(stats: dict[str, float]) -> str:
    if int(stats.get("n") or 0) <= 0 or not math.isfinite(stats.get("mean", float("nan"))):
        return "n/a"
    return f"{stats['mean']:.4f} ± {stats['std']:.4f} (n={int(stats['n'])})"


def render_metrics_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# SOTA 门控指标表",
        "",
        "**未跑满 72h（窗口 A→D）且 3 seeds 复验前，不得宣称 SOTA。**",
        "",
        "口径：`unified` 为全数据混合指标（如 `val/f1_modulation`）；`dataset-specific` 为 `val/<metric>/<dataset>`。",
        "",
        "| job | window | seeds | unified macro-F1 / geomean | dataset-specific |",
        "| --- | --- | --- | --- | --- |",
    ]
    for job_id, row in payload.get("jobs", {}).items():
        unified = row.get("unified") or {}
        f1 = unified.get("val/f1_modulation") or unified.get("val/macro_f1_modulation") or unified.get("val/multitask_geomean") or unified.get("monitor_value")
        specific = row.get("dataset_specific") or {}
        ds_parts = []
        for key in sorted(specific):
            if key.startswith("val/f1_modulation/") or key.startswith("val/acc_emitter/"):
                ds_parts.append(f"{key.split('/', 2)[-1]} {format_mean_std(specific[key])}")
        ds_text = "; ".join(ds_parts) if ds_parts else "—"
        lines.append(
            f"| {job_id} | {row.get('window')} | {row.get('n_seeds_found')}/{row.get('n_seeds_planned')} | {format_mean_std(f1 or {})} | {ds_text} |"
        )
    if not payload.get("jobs"):
        lines.append("| *(empty)* | — | 0/0 | n/a | 尚未执行任何正式 GPU 实验 |")
    lines.extend(["", f"sota_claim_allowed: **{bool(payload.get('sota_claim_allowed'))}**", ""])
    return "\n".join(lines)


ABLATION_STAGE2_JOBS = (
    "validity_stage2",
    "abl_enc1_stage2",
    "abl_enc3_stage2",
    "abl_mae_only_stage2",
    "abl_z_general_stage2",
    "abl_phase_on_stage2",
    "abl_legacy_decoder_stage2",
    "abl_low_rank_stage2",
    "abl_bidir_share_stage2",
)

SLOT_PRETRAIN = {
    "validity_stage2": "configs/experiments/validity_pretrain.yaml",
    "abl_enc1_stage2": "configs/experiments/abl_enc1_pretrain.yaml",
    "abl_enc3_stage2": "configs/experiments/abl_enc3_pretrain.yaml",
    "abl_mae_only_stage2": "configs/experiments/abl_mae_only_pretrain.yaml",
    "abl_z_general_stage2": "configs/experiments/abl_z_general_pretrain.yaml",
    "abl_phase_on_stage2": "configs/experiments/abl_phase_on_pretrain.yaml",
    "abl_legacy_decoder_stage2": "configs/experiments/abl_legacy_decoder_pretrain.yaml",
    "abl_low_rank_stage2": "configs/experiments/validity_pretrain.yaml",
    "abl_bidir_share_stage2": "configs/experiments/abl_bidir_share_pretrain.yaml",
}
SLOT_STAGE2 = {
    "validity_stage2": "configs/experiments/validity_stage2.yaml",
    "abl_enc1_stage2": "configs/experiments/abl_enc1_stage2.yaml",
    "abl_enc3_stage2": "configs/experiments/abl_enc3_stage2.yaml",
    "abl_mae_only_stage2": "configs/experiments/abl_mae_only_stage2.yaml",
    "abl_z_general_stage2": "configs/experiments/abl_z_general_stage2.yaml",
    "abl_phase_on_stage2": "configs/experiments/abl_phase_on_stage2.yaml",
    "abl_legacy_decoder_stage2": "configs/experiments/abl_legacy_decoder_stage2.yaml",
    "abl_low_rank_stage2": "configs/experiments/abl_low_rank_stage2.yaml",
    "abl_bidir_share_stage2": "configs/experiments/abl_bidir_share_stage2.yaml",
}


def select_confirm_slots(metrics_payload: dict[str, Any]) -> dict[str, Any]:
    scored: list[tuple[float, str]] = []
    jobs = metrics_payload.get("jobs") or {}
    for job_id in ABLATION_STAGE2_JOBS:
        row = jobs.get(job_id) or {}
        unified = row.get("unified") or {}
        stats = unified.get("val/multitask_geomean") or unified.get("val/f1_modulation") or unified.get("monitor_value")
        if not stats or int(stats.get("n") or 0) <= 0:
            continue
        scored.append((float(stats["mean"]), job_id))
    scored.sort(reverse=True)
    slot_a = scored[0][1] if scored else "validity_stage2"
    slot_b = scored[1][1] if len(scored) > 1 else ("abl_enc3_stage2" if slot_a != "abl_enc3_stage2" else "abl_mae_only_stage2")
    return {
        "slot_a": {
            "from_job": slot_a,
            "pretrain_config": SLOT_PRETRAIN.get(slot_a, "configs/experiments/validity_pretrain.yaml"),
            "stage2_config": SLOT_STAGE2.get(slot_a, "configs/experiments/validity_stage2.yaml"),
            "placeholder": not bool(scored),
        },
        "slot_b": {
            "from_job": slot_b,
            "pretrain_config": SLOT_PRETRAIN.get(slot_b, "configs/experiments/abl_enc3_pretrain.yaml"),
            "stage2_config": SLOT_STAGE2.get(slot_b, "configs/experiments/abl_enc3_stage2.yaml"),
            "placeholder": len(scored) < 2,
        },
        "note": "B 窗口未完成时 slot 仅为占位，不得把占位配置的结果写成 SOTA。",
    }


def apply_selected_slot_configs(
    planned: Sequence[PlannedCommand],
    selected: Mapping[str, Any] | None,
    *,
    root: Path,
) -> list[PlannedCommand]:
    """B 完成后把 C 窗口 --config 换成 selected.json 中的优胜 yaml。"""
    if not selected:
        return list(planned)
    mapping = {
        "confirm_slot_a_pretrain": ("slot_a", "pretrain_config"),
        "confirm_slot_a_stage2": ("slot_a", "stage2_config"),
        "confirm_slot_b_pretrain": ("slot_b", "pretrain_config"),
        "confirm_slot_b_stage2": ("slot_b", "stage2_config"),
    }
    rewritten: list[PlannedCommand] = []
    for item in planned:
        spec = mapping.get(item.job_id)
        if spec is None or item.kind != "train":
            rewritten.append(item)
            continue
        slot_key, field = spec
        slot = (selected.get(slot_key) or {}) if isinstance(selected, Mapping) else {}
        config = slot.get(field)
        if not config:
            rewritten.append(item)
            continue
        argv = list(item.argv)
        if "--config" in argv:
            idx = argv.index("--config")
            argv[idx + 1] = str((root / str(config)).resolve()) if not Path(str(config)).is_absolute() else str(config)
        rewritten.append(
            PlannedCommand(
                job_id=item.job_id,
                seed=item.seed,
                window=item.window,
                kind=item.kind,
                run_name=item.run_name,
                argv=argv,
                env=item.env,
                depends_on=item.depends_on,
                requires=item.requires,
                optional=item.optional,
                gpu_mode=item.gpu_mode,
                alias_of=item.alias_of,
                description=item.description + f" [slot {slot.get('from_job', '')}]",
                init_from=item.init_from,
            )
        )
    return rewritten


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _command_key(item: PlannedCommand) -> str:
    return f"{item.job_id}:seed{item.seed}"


def run_planned(
    planned: Sequence[PlannedCommand],
    *,
    root: Path,
    out_dir: Path,
    fail_fast: bool = True,
    skip_requires: bool = False,
    dry_run: bool = False,
    status: dict[str, Blocker] | None = None,
) -> dict[str, Any]:
    catalog = {job.id: job for job in build_gate_jobs()}
    blockers = status or blockers_by_name(root)
    results: dict[str, Any] = {"commands": [], "failed": [], "skipped": [], "completed": []}
    done: set[str] = set()
    failed_jobs: set[str] = set()
    out_dir.mkdir(parents=True, exist_ok=True)
    for item in planned:
        record = {
            "job_id": item.job_id,
            "seed": item.seed,
            "kind": item.kind,
            "run_name": item.run_name,
            "argv": item.argv,
        }
        job = catalog[item.job_id]
        if item.kind == "alias":
            record["status"] = "alias"
            record["alias_of"] = item.alias_of
            results["completed"].append(_command_key(item))
            done.add(_command_key(item))
            done.add(item.job_id)
            results["commands"].append(record)
            continue
        missing_deps = [dep for dep in item.depends_on if dep in failed_jobs]
        if missing_deps:
            record["status"] = "blocked"
            record["reason"] = f"依赖失败: {missing_deps}"
            results["skipped"].append(record)
            results["commands"].append(record)
            failed_jobs.add(item.job_id)
            if fail_fast:
                break
            continue
        missing = [] if skip_requires else missing_requirements(job, blockers)
        if missing:
            detail = "; ".join(f"{item.name}: {item.detail}" for item in missing)
            record["status"] = "skipped" if job.optional else "blocked"
            record["reason"] = detail
            results["skipped"].append(record)
            results["commands"].append(record)
            if not job.optional:
                failed_jobs.add(item.job_id)
                if fail_fast:
                    break
            continue
        if dry_run:
            record["status"] = "dry_run"
            results["commands"].append(record)
            done.add(_command_key(item))
            done.add(item.job_id)
            continue
        env = os.environ.copy()
        env.update(item.env)
        cwd = str(root)
        completed = subprocess.run(item.argv, cwd=cwd, env=env)
        record["returncode"] = int(completed.returncode)
        if completed.returncode != 0:
            record["status"] = "failed"
            results["failed"].append(record)
            results["commands"].append(record)
            failed_jobs.add(item.job_id)
            if fail_fast:
                break
            continue
        record["status"] = "ok"
        results["completed"].append(_command_key(item))
        results["commands"].append(record)
        done.add(_command_key(item))
        done.add(item.job_id)
    write_json(out_dir / "last_run.json", results)
    return results


def parse_gpus(value: str | None) -> list[int]:
    if not value:
        return []
    text = str(value).strip()
    if text.lower() in {"auto", "all"}:
        count = 0
        try:
            import torch

            count = int(torch.cuda.device_count())
        except Exception:
            count = 0
        return list(range(count))
    return [int(part.strip()) for part in text.split(",") if part.strip()]


def default_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="72h / 6×5090 SOTA 实验门控。默认只打印命令矩阵，不启动正式训练。",
    )
    parser.add_argument("--dry-run", action="store_true", help="打印将要执行的命令矩阵（默认行为）")
    parser.add_argument("--smoke", action="store_true", help="tiny/synthetic 跑 1 个 gate，证明可调用 train.py")
    parser.add_argument("--execute", action="store_true", help="真正执行选中的窗口；必须同时给 --windows 或 --jobs")
    parser.add_argument("--aggregate", action="store_true", help="只汇总已有 run 的指标表")
    parser.add_argument("--windows", default="", help="逗号分隔 A,B,C,D")
    parser.add_argument("--jobs", default="", help="逗号分隔 job id")
    parser.add_argument("--gpus", default="", help="如 0,1,2,3,4,5 或 auto")
    parser.add_argument("--profile", default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--skip-requires", action="store_true")
    parser.add_argument("--fail-fast", dest="fail_fast", action="store_true", default=True)
    parser.add_argument("--no-fail-fast", dest="fail_fast", action="store_false")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--root", default=str(ROOT))
    return parser


def _csv_list(raw: str) -> list[str]:
    return [part.strip() for part in str(raw or "").split(",") if part.strip()]


def main(argv: Sequence[str] | None = None) -> int:
    args = default_arg_parser().parse_args(list(argv) if argv is not None else None)
    root = Path(args.root).resolve()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    jobs = build_gate_jobs()
    windows = _csv_list(args.windows)
    job_ids = _csv_list(args.jobs)
    blockers = collect_blockers(root)
    write_json(out_dir / "blockers.json", [asdict(item) for item in blockers])
    status = {item.name: item for item in blockers}

    smoke = bool(args.smoke)
    execute = bool(args.execute)
    dry_run = bool(args.dry_run) or (not execute and not smoke and not args.aggregate)

    if execute and not smoke and not windows and not job_ids:
        print(
            "拒绝启动完整 72h：请显式指定 --windows A（或 B/C/D）或 --jobs <id>，或改用 --smoke / --dry-run。\n"
            "未跑满 72h 门控前不得宣称 SOTA。",
            file=sys.stderr,
        )
        return 2

    profile = args.profile
    synthetic = bool(args.synthetic)
    skip_requires = bool(args.skip_requires)
    extra: list[str] = []
    selected = jobs
    run_prefix = "sota_gate"
    if smoke:
        selected = filter_jobs(jobs, job_ids=["validity_pretrain"])
        profile = profile or "tiny"
        synthetic = True
        skip_requires = True
        extra = ["--limit-train-batches", "2", "--limit-val-batches", "1", "--max-epochs", "1"]
        execute = True
        dry_run = False
        run_prefix = "sota_gate/smoke"
    elif windows or job_ids:
        selected = filter_jobs(jobs, windows=windows or None, job_ids=job_ids or None)

    gpus = parse_gpus(args.gpus)
    planned = plan_commands(
        selected,
        root=root,
        profile=profile,
        synthetic=synthetic,
        gpus=gpus,
        extra=extra,
        run_prefix=run_prefix,
    )
    selected_path = out_dir / "selected.json"
    selected_slots = None
    if selected_path.is_file():
        try:
            selected_slots = json.loads(selected_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            selected_slots = None
    planned = apply_selected_slot_configs(planned, selected_slots, root=root)
    matrix = format_command_matrix(planned, blockers=blockers)
    (out_dir / "dry_run.txt").parent.mkdir(parents=True, exist_ok=True)
    (out_dir / "dry_run.txt").write_text(matrix, encoding="utf-8")
    print(matrix, end="", flush=True)

    if args.aggregate or execute:
        metrics = aggregate_job_metrics(root, selected if args.aggregate else jobs)
        selected_slots = select_confirm_slots(metrics)
        metrics["selected"] = selected_slots
        write_json(out_dir / "metrics.json", metrics)
        (out_dir / "metrics.md").write_text(render_metrics_markdown(metrics), encoding="utf-8")
        write_json(out_dir / "selected.json", selected_slots)

    if dry_run and not execute:
        print(f"已写入 {out_dir / 'dry_run.txt'} ；未启动训练。", flush=True)
        return 0

    if execute:
        result = run_planned(
            planned,
            root=root,
            out_dir=out_dir,
            fail_fast=bool(args.fail_fast),
            skip_requires=skip_requires,
            dry_run=False,
            status=status,
        )
        write_json(out_dir / "index.json", result)
        metrics = aggregate_job_metrics(root, jobs)
        write_json(out_dir / "metrics.json", metrics)
        (out_dir / "metrics.md").write_text(render_metrics_markdown(metrics), encoding="utf-8")
        failed = result.get("failed") or []
        blocked = [row for row in result.get("skipped") or [] if row.get("status") == "blocked"]
        if failed or blocked:
            print(json.dumps({"failed": failed, "blocked": blocked}, ensure_ascii=False, indent=2), flush=True)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
