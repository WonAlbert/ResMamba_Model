from __future__ import annotations

from pathlib import Path

from resmamba_signal_model.config.yaml_config import load_yaml_config
from resmamba_signal_model.training.sota_gate import (
    apply_selected_slot_configs,
    build_gate_jobs,
    collect_blockers,
    filter_jobs,
    format_command_matrix,
    format_mean_std,
    main,
    missing_requirements,
    plan_commands,
    render_metrics_markdown,
    run_planned,
    select_confirm_slots,
    split_unified_and_specific,
)

ROOT = Path(__file__).resolve().parents[1]


def test_gate_matrix_covers_plan_windows_and_train_py() -> None:
    jobs = build_gate_jobs()
    windows = {job.window for job in jobs}
    assert windows == {"A_validity", "B_ablation", "C_confirm", "D_eval"}
    ids = {job.id for job in jobs}
    assert "validity_pretrain" in ids
    assert "abl_enc1_pretrain" in ids and "abl_enc3_stage2" in ids
    assert "abl_mae_only_pretrain" in ids and "abl_z_general_stage2" in ids
    assert "abl_phase_on_pretrain" in ids and "abl_low_rank_stage2" in ids
    assert "abl_bidir_share_pretrain" in ids and "abl_bidir_share_stage2" in ids
    assert "confirm_slot_a_stage2" in ids and "eval_continual" in ids
    aliases = {job.id: job.alias_of for job in jobs if job.kind == "alias"}
    assert aliases["abl_enc5"] == "validity_stage2"
    assert aliases["abl_uti_v2"] == "validity_stage2"
    planned = plan_commands(jobs, root=ROOT, gpus=[0, 1, 2, 3, 4, 5])
    train_cmds = [item for item in planned if item.kind == "train"]
    assert train_cmds
    assert any("scripts/train.py" in part for part in train_cmds[0].argv)
    smoke = plan_commands(filter_jobs(jobs, job_ids=["validity_pretrain"]), root=ROOT, run_prefix="sota_gate/smoke")
    assert smoke[0].run_name.startswith("sota_gate/smoke/")
    confirm = [item for item in planned if item.job_id == "confirm_slot_a_pretrain"]
    assert {item.seed for item in confirm} == {0, 1, 2}
    matrix = format_command_matrix(planned)
    assert "不得宣称 SOTA" in matrix
    assert "validity_pretrain" in matrix


def test_execute_without_window_is_refused() -> None:
    code = main(["--execute"])
    assert code == 2


def test_dry_run_writes_matrix(tmp_path: Path) -> None:
    code = main(["--dry-run", "--windows", "A", "--out-dir", str(tmp_path), "--root", str(ROOT)])
    assert code == 0
    text = (tmp_path / "dry_run.txt").read_text(encoding="utf-8")
    assert "validity_pytest" in text
    assert "scripts/train.py" in text
    assert (tmp_path / "blockers.json").is_file()


def test_fail_fast_blocks_when_requires_missing(tmp_path: Path) -> None:
    from resmamba_signal_model.training.sota_gate import Blocker

    jobs = filter_jobs(build_gate_jobs(), job_ids=["validity_pretrain"])
    planned = plan_commands(jobs, root=ROOT)
    status = {
        "mamba_kernel": Blocker("mamba_kernel", False, "no kernel"),
        "split_manifest": Blocker("split_manifest", False, "no manifest"),
        "h5_stamp": Blocker("h5_stamp", False, "no stamp"),
    }
    result = run_planned(planned, root=ROOT, out_dir=tmp_path, fail_fast=True, skip_requires=False, status=status)
    assert result["failed"] == []
    assert result["skipped"]
    assert result["skipped"][0]["status"] == "blocked"


def test_metrics_table_mean_std_and_split_scopes() -> None:
    unified, specific = split_unified_and_specific(
        {
            "val/f1_modulation": 0.8,
            "val/f1_modulation/rml2016_10a": 0.9,
            "monitor_value": 0.7,
        }
    )
    assert "val/f1_modulation" in unified
    assert "val/f1_modulation/rml2016_10a" in specific
    payload = {
        "sota_claim_allowed": False,
        "jobs": {
            "confirm_slot_a_stage2": {
                "window": "C_confirm",
                "n_seeds_found": 3,
                "n_seeds_planned": 3,
                "unified": {"val/f1_modulation": {"mean": 0.81, "std": 0.02, "n": 3}},
                "dataset_specific": {
                    "val/f1_modulation/rml2016_10a": {"mean": 0.83, "std": 0.01, "n": 3},
                },
            }
        },
    }
    md = render_metrics_markdown(payload)
    assert "0.8100 ± 0.0200" in md
    assert "rml2016_10a" in md
    assert "sota_claim_allowed: **False**" in md
    assert "不得宣称 SOTA" in md
    assert "n/a" in format_mean_std({"mean": float("nan"), "std": float("nan"), "n": 0})


def test_confirm_slot_rewrites_config_from_selected_json() -> None:
    jobs = filter_jobs(build_gate_jobs(), job_ids=["confirm_slot_a_pretrain"])
    planned = plan_commands(jobs, root=ROOT)
    rewritten = apply_selected_slot_configs(
        planned,
        {
            "slot_a": {
                "from_job": "abl_enc3_stage2",
                "pretrain_config": "configs/experiments/abl_enc3_pretrain.yaml",
                "stage2_config": "configs/experiments/abl_enc3_stage2.yaml",
            }
        },
        root=ROOT,
    )
    idx = rewritten[0].argv.index("--config")
    assert rewritten[0].argv[idx + 1].endswith("abl_enc3_pretrain.yaml")


def test_local_blockers_are_reported() -> None:
    blockers = {item.name: item for item in collect_blockers(ROOT)}
    assert blockers["sota_claim"].ok is False
    assert "不得宣称 SOTA" in blockers["sota_claim"].detail
    jobs = filter_jobs(build_gate_jobs(), job_ids=["validity_pretrain"])
    missing = missing_requirements(jobs[0], blockers)
    names = {item.name for item in missing}
    assert names <= {"mamba_kernel", "split_manifest", "h5_stamp"}


def test_select_slots_placeholder_when_empty() -> None:
    slots = select_confirm_slots({"jobs": {}})
    assert slots["slot_a"]["placeholder"] is True
    assert "占位" in slots["note"]
