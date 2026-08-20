from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any, Iterable

from resmamba_signal_model.models.task_interface import DEFAULT_TASKS

KIND_ALIASES: dict[str, str] = {
    "classification": "classification",
    "modulation": "classification",
    "cls": "classification",
    "label": "classification",
    "emitter": "emitter",
    "sei": "emitter",
    "fingerprint": "emitter",
    "clustering": "clustering",
    "cluster": "clustering",
    "prediction": "prediction",
    "forecast": "prediction",
    "imputation": "imputation",
    "impute": "imputation",
}

KIND_MASK_MODE: dict[str, str] = {
    "classification": "none",
    "emitter": "none",
    "clustering": "none",
    "prediction": "suffix",
    "imputation": "span",
}

KIND_MONITOR: dict[str, str] = {
    "classification": "f1",
    "emitter": "per_dataset_macro_acc",
    "clustering": "nmi",
    "prediction": "ssim",
    "imputation": "ssim",
}

SOURCE_TO_BUILTIN_TASK: dict[str, str] = {
    "classification": "modulation",
    "emitter": "emitter",
    "clustering": "clustering",
    "prediction": "prediction",
    "imputation": "imputation",
    "modulation": "modulation",
}

BUILTIN_HEAD_ATTR: dict[str, str] = {
    "modulation": "modulation_head",
    "emitter": "emitter_head",
    "clustering": "clustering_head",
    "prediction": "prediction_head",
    "imputation": "imputation_head",
}


def normalize_kind(value: str | None, *, default: str = "classification") -> str:
    if value is None or not str(value).strip():
        return default
    key = str(value).strip().lower()
    if key not in KIND_ALIASES:
        raise ValueError(f"未知任务 kind {value!r}，可选: {sorted(set(KIND_ALIASES.values()))}")
    return KIND_ALIASES[key]


@dataclass
class TaskSpec:
    name: str
    kind: str
    source: str
    num_classes: int | None = None
    num_prototypes: int | None = None
    monitor: str | None = None
    label_field: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.name = str(self.name)
        self.kind = normalize_kind(self.kind, default="classification")
        self.source = str(self.source or self.name)
        if not self.monitor:
            self.monitor = KIND_MONITOR[self.kind]
        if not self.label_field:
            if self.kind == "emitter":
                self.label_field = "global_emitter_id"
            elif self.kind == "clustering":
                self.label_field = "global_label_id"
            elif self.kind == "classification":
                self.label_field = "canonical_mod_label_id"

    def mask_mode(self) -> str:
        return KIND_MASK_MODE[self.kind]

    def head_attr(self) -> str | None:
        return BUILTIN_HEAD_ATTR.get(self.name)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        extra = payload.pop("extra") or {}
        payload.update(extra)
        return {k: v for k, v in payload.items() if v is not None}


def builtin_spec(name: str) -> TaskSpec:
    presets: dict[str, TaskSpec] = {
        "modulation": TaskSpec("modulation", "classification", "classification", label_field="canonical_mod_label_id"),
        "emitter": TaskSpec("emitter", "emitter", "emitter", label_field="global_emitter_id"),
        "clustering": TaskSpec("clustering", "clustering", "clustering", label_field="global_label_id"),
        "prediction": TaskSpec("prediction", "prediction", "prediction"),
        "imputation": TaskSpec("imputation", "imputation", "imputation"),
    }
    if name not in presets:
        raise KeyError(f"不是内置任务: {name}")
    return replace(presets[name])


def _spec_from_mapping(entry: dict[str, Any], *, task_kinds: dict[str, str]) -> TaskSpec:
    name = str(entry.get("name") or entry.get("task") or "")
    if not name:
        raise ValueError(f"任务条目缺少 name: {entry}")
    if name in SOURCE_TO_BUILTIN_TASK.values() and "kind" not in entry and name not in task_kinds:
        spec = builtin_spec(name)
        if entry.get("source"):
            spec.source = str(entry["source"])
    else:
        kind = entry.get("kind") or entry.get("type") or task_kinds.get(name)
        if kind is None and name in SOURCE_TO_BUILTIN_TASK.values():
            spec = builtin_spec(name)
        else:
            spec = TaskSpec(
                name=name,
                kind=normalize_kind(kind, default="classification"),
                source=str(entry.get("source") or name),
            )
    if entry.get("source"):
        spec.source = str(entry["source"])
    if entry.get("num_classes") is not None:
        spec.num_classes = int(entry["num_classes"])
    if entry.get("num_prototypes") is not None:
        spec.num_prototypes = int(entry["num_prototypes"])
    if entry.get("monitor"):
        spec.monitor = str(entry["monitor"])
    if entry.get("label_field"):
        spec.label_field = str(entry["label_field"])
    reserved = {"name", "task", "kind", "type", "source", "num_classes", "num_prototypes", "monitor", "label_field"}
    spec.extra.update({k: v for k, v in entry.items() if k not in reserved})
    return spec


def _spec_from_name(name: str, *, task_kinds: dict[str, str], task_pools: dict[str, Any]) -> TaskSpec:
    if name in SOURCE_TO_BUILTIN_TASK.values() and name not in task_kinds:
        spec = builtin_spec(name)
        if spec.source not in task_pools and name in task_pools:
            spec.source = name
        return spec
    kind = task_kinds.get(name)
    if kind is None and name in SOURCE_TO_BUILTIN_TASK:
        return builtin_spec(SOURCE_TO_BUILTIN_TASK[name])
    source = name
    if name == "modulation" and "classification" in task_pools:
        source = "classification"
    elif name in task_pools:
        source = name
    return TaskSpec(name=name, kind=normalize_kind(kind, default="classification"), source=source)


class TaskCatalog:
    def __init__(self, specs: Iterable[TaskSpec]) -> None:
        seen: dict[str, TaskSpec] = {}
        for spec in specs:
            seen[spec.name] = spec
        if not seen:
            seen = {name: builtin_spec(name) for name in DEFAULT_TASKS}
        self.specs: list[TaskSpec] = list(seen.values())
        self.by_name: dict[str, TaskSpec] = {spec.name: spec for spec in self.specs}
        self.source_to_task: dict[str, str] = {spec.source: spec.name for spec in self.specs}
        for src, task in SOURCE_TO_BUILTIN_TASK.items():
            self.source_to_task.setdefault(src, task)
        self.task_to_source: dict[str, str] = {spec.name: spec.source for spec in self.specs}

    @property
    def names(self) -> list[str]:
        return [spec.name for spec in self.specs]

    def get(self, name: str) -> TaskSpec | None:
        return self.by_name.get(name)

    def require(self, name: str) -> TaskSpec:
        spec = self.get(name)
        if spec is None:
            raise KeyError(f"任务 {name!r} 不在目录中: {self.names}")
        return spec

    def kind(self, name: str) -> str:
        spec = self.get(name)
        if spec is not None:
            return spec.kind
        if name in SOURCE_TO_BUILTIN_TASK.values():
            return builtin_spec(name).kind
        return "classification"

    def mask_mode(self, name: str) -> str:
        spec = self.get(name)
        if spec is not None:
            return spec.mask_mode()
        return KIND_MASK_MODE.get(self.kind(name), "none")

    def with_task(self, spec: TaskSpec) -> "TaskCatalog":
        merged = {item.name: item for item in self.specs}
        merged[spec.name] = spec
        return TaskCatalog(merged.values())

    def to_dicts(self) -> list[dict[str, Any]]:
        return [spec.to_dict() for spec in self.specs]


def resolve_task_catalog(train_cfg: dict[str, Any] | None) -> TaskCatalog:
    """从训练配置解析任意数量下游任务。

    优先级：``resolved_tasks`` > ``tasks`` 列表 > ``task_pools`` 键 > 内置五任务。
    新任务可写 ``task_kinds: {sonar: classification}`` 与对应 ``task_pools``。
    """
    train_cfg = train_cfg or {}
    task_kinds = {str(k): str(v) for k, v in dict(train_cfg.get("task_kinds") or {}).items()}
    raw_pools = train_cfg.get("task_pools") or {}
    task_pools = raw_pools if isinstance(raw_pools, dict) else {}
    raw = train_cfg.get("resolved_tasks") or train_cfg.get("tasks")
    specs: list[TaskSpec] = []
    if isinstance(raw, dict):
        raw = list(raw.values()) if all(isinstance(v, dict) for v in raw.values()) else list(raw)
    if isinstance(raw, (list, tuple)) and raw:
        for entry in raw:
            if isinstance(entry, str):
                specs.append(_spec_from_name(entry, task_kinds=task_kinds, task_pools=task_pools))
            elif isinstance(entry, dict):
                specs.append(_spec_from_mapping(entry, task_kinds=task_kinds))
            else:
                raise TypeError(f"tasks 条目必须是字符串或映射，当前 {type(entry)}")
    elif task_pools:
        for source in task_pools:
            task = SOURCE_TO_BUILTIN_TASK.get(str(source), str(source))
            specs.append(_spec_from_name(task, task_kinds=task_kinds, task_pools=task_pools))
            specs[-1].source = str(source)
    else:
        specs = [builtin_spec(name) for name in DEFAULT_TASKS]
    extra_kinds = [name for name in task_kinds if name not in {spec.name for spec in specs}]
    for name in extra_kinds:
        specs.append(_spec_from_name(name, task_kinds=task_kinds, task_pools=task_pools))
    return TaskCatalog(specs)


def apply_catalog_to_train_cfg(train_cfg: dict[str, Any], catalog: TaskCatalog) -> dict[str, Any]:
    train_cfg = dict(train_cfg)
    train_cfg["resolved_tasks"] = catalog.to_dicts()
    train_cfg["tasks"] = catalog.names
    return train_cfg


def apply_catalog_to_model_cfg(model_cfg: Any, catalog: TaskCatalog) -> Any:
    names = tuple(catalog.names)
    model_cfg.task_names = names
    model_cfg.task_kinds = {spec.name: spec.kind for spec in catalog.specs}
    model_cfg.num_task_types = max(int(getattr(model_cfg, "num_task_types", 8) or 8), len(names) + 4)
    return model_cfg
