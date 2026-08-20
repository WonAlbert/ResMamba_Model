from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"
_META_KEYS = frozenset({"_base_", "profile", "profiles"})


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in _META_KEYS:
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _resolve_config_path(path: Path, base_ref: str) -> Path:
    candidate = (path.parent / base_ref).resolve()
    if candidate.is_file():
        return candidate
    return (CONFIG_ROOT / base_ref).resolve()


def load_yaml_config(
    path: str | Path,
    *,
    profile: str | None = None,
    _stack: set[Path] | None = None,
) -> dict[str, Any]:
    """加载 YAML 配置，支持 _base_ 继承与 profiles 预设。"""
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    visited = set(_stack or ())
    if config_path in visited:
        chain = " -> ".join(str(p) for p in (*visited, config_path))
        raise ValueError(f"配置文件继承出现循环: {chain}")
    visited.add(config_path)

    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"配置文件根节点必须是 dict: {config_path}")

    merged: dict[str, Any] = {}
    bases = raw.get("_base_")
    if bases is not None:
        base_refs = [bases] if isinstance(bases, str) else list(bases)
        for base_ref in base_refs:
            base_path = _resolve_config_path(config_path, str(base_ref))
            try:
                base_cfg = load_yaml_config(base_path, profile=profile, _stack=visited)
            except ValueError as exc:
                if profile is None or "未知 profile" not in str(exc):
                    raise
                base_cfg = load_yaml_config(base_path, profile=None, _stack=visited)
            merged = deep_merge(merged, base_cfg)

    profiles = raw.get("profiles") or {}
    direct = {k: v for k, v in raw.items() if k not in _META_KEYS}
    merged = deep_merge(merged, direct)

    profile_name = profile if profile is not None else raw.get("profile")
    if profile_name is None and profiles and "default" in profiles:
        profile_name = "default"
    if profile_name:
        if not isinstance(profiles, dict):
            raise ValueError(f"profiles 必须是 dict: {config_path}")
        if profile_name in profiles:
            profile_cfg = profiles[profile_name]
            if not isinstance(profile_cfg, dict):
                raise ValueError(f"profile {profile_name!r} 必须是 dict: {config_path}")
            merged = deep_merge(merged, profile_cfg)
        elif profiles:
            available = ", ".join(sorted(profiles)) or "(empty)"
            raise ValueError(f"未知 profile {profile_name!r}（{config_path}），可选: {available}")
        elif not bases:
            raise ValueError(f"未知 profile {profile_name!r}（{config_path}），当前文件没有 profiles")

    return merged


def load_yaml(path: str | Path, *, profile: str | None = None) -> dict[str, Any]:
    """向后兼容的 YAML 加载入口。"""
    return load_yaml_config(path, profile=profile)
