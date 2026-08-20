from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Mapping, Sequence

import torch


SIGNAL_CONTRACT_VERSION = 1
MISSING_METADATA = "__missing__"
CAPTURE_METADATA_KEYS = (
    "receiver_id",
    "session_id",
    "channel_id",
    "capture_id",
)


def _valid_sample_rate(value: float | int | None) -> bool:
    if value is None:
        return False
    try:
        return math.isfinite(float(value)) and float(value) > 0
    except (TypeError, ValueError):
        return False


def _normalize_complex_pairs(
    pairs: Sequence[Sequence[int]] | None,
) -> tuple[tuple[int, int], ...]:
    if pairs is None:
        return ()
    normalized: list[tuple[int, int]] = []
    for pair in pairs:
        if len(pair) != 2:
            raise ValueError(f"complex pair 必须含两个通道索引，实际为 {pair!r}")
        normalized.append((int(pair[0]), int(pair[1])))
    return tuple(normalized)


@dataclass(frozen=True)
class SignalSpec:
    """描述单个信号样本的通道、坐标和模态语义。

    ``num_channels`` 不再固定为 I/Q 两通道；旧 RF 调用可使用
    :meth:`SignalSpec.rf`，其默认行为仍是两个通道和一个复数通道对。
    """

    num_channels: int
    modality_id: str = "unknown"
    channel_names: tuple[str, ...] = ()
    sample_rate_hz: float | None = None
    coordinate_unit: str = "sample_index"
    complex_pairs: tuple[tuple[int, int], ...] = ()
    metadata_fields: tuple[str, ...] = CAPTURE_METADATA_KEYS
    schema_version: int = SIGNAL_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if int(self.num_channels) <= 0:
            raise ValueError(f"num_channels 必须为正数，实际为 {self.num_channels}")
        if self.channel_names and len(self.channel_names) != int(self.num_channels):
            raise ValueError(
                f"channel_names 数量 {len(self.channel_names)} 与 num_channels={self.num_channels} 不一致"
            )
        if len(set(self.channel_names)) != len(self.channel_names):
            raise ValueError("channel_names 不能重复")
        if self.sample_rate_hz is not None and not _valid_sample_rate(self.sample_rate_hz):
            raise ValueError(f"sample_rate_hz 必须为正有限值或 None，实际为 {self.sample_rate_hz}")
        for real_idx, imag_idx in self.complex_pairs:
            if real_idx == imag_idx:
                raise ValueError("complex pair 的实部和虚部通道不能相同")
            if min(real_idx, imag_idx) < 0 or max(real_idx, imag_idx) >= int(self.num_channels):
                raise ValueError(
                    f"complex pair {(real_idx, imag_idx)} 超出 {self.num_channels} 个通道的范围"
                )

    @classmethod
    def rf(
        cls,
        *,
        num_channels: int = 2,
        sample_rate_hz: float | None = None,
        channel_names: Sequence[str] | None = None,
    ) -> SignalSpec:
        names = tuple(channel_names or (("i", "q") if num_channels == 2 else ()))
        pairs = ((0, 1),) if num_channels == 2 else ()
        return cls(
            num_channels=int(num_channels),
            modality_id="rf",
            channel_names=names,
            sample_rate_hz=sample_rate_hz,
            coordinate_unit="seconds" if _valid_sample_rate(sample_rate_hz) else "sample_index",
            complex_pairs=pairs,
        )

    def with_sample_rate(self, sample_rate_hz: float | None) -> SignalSpec:
        return replace(
            self,
            sample_rate_hz=sample_rate_hz,
            coordinate_unit="seconds" if _valid_sample_rate(sample_rate_hz) else "sample_index",
        )

    def to_model_dict(self) -> dict[str, Any]:
        """仅使用模型侧最小 SignalSpec 可接受的键，避免 data/models 强耦合。"""
        return {
            "name": self.modality_id,
            "num_channels": int(self.num_channels),
            "modality": self.modality_id,
            "complex_pairs": self.complex_pairs,
            "adapter_key": None,
        }


@dataclass(frozen=True)
class SignalBatch:
    """任意通道、可变长度信号的批契约。

    ``values`` 为 ``[B,C,L]``，``sample_mask`` 与 ``channel_mask`` 分别描述
    有效时间位置和有效通道。``to_legacy_dict`` 同时暴露 ``iq``，让现有
    两通道训练代码可以渐进迁移。
    """

    values: torch.Tensor
    sample_mask: torch.Tensor
    channel_mask: torch.Tensor
    time_coordinates: torch.Tensor
    sample_rate_hz: torch.Tensor
    modality_id: tuple[str, ...]
    complex_pairs: tuple[tuple[tuple[int, int], ...], ...]
    coordinate_unit: tuple[str, ...]
    metadata: Mapping[str, tuple[Any, ...]]
    specs: tuple[SignalSpec, ...]
    schema_version: int = SIGNAL_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.values.ndim != 3:
            raise ValueError(f"values 必须为 [B,C,L]，实际为 {tuple(self.values.shape)}")
        batch, channels, length = self.values.shape
        if self.sample_mask.shape != (batch, length):
            raise ValueError(
                f"sample_mask 必须为 {(batch, length)}，实际为 {tuple(self.sample_mask.shape)}"
            )
        if self.channel_mask.shape != (batch, channels):
            raise ValueError(
                f"channel_mask 必须为 {(batch, channels)}，实际为 {tuple(self.channel_mask.shape)}"
            )
        if self.time_coordinates.shape != (batch, length):
            raise ValueError(
                f"time_coordinates 必须为 {(batch, length)}，实际为 {tuple(self.time_coordinates.shape)}"
            )
        if self.sample_rate_hz.shape != (batch,):
            raise ValueError(
                f"sample_rate_hz 必须为 {(batch,)}，实际为 {tuple(self.sample_rate_hz.shape)}"
            )
        if self.sample_mask.dtype != torch.bool or self.channel_mask.dtype != torch.bool:
            raise TypeError("sample_mask 和 channel_mask 必须为 bool")
        for field_name, values in (
            ("modality_id", self.modality_id),
            ("complex_pairs", self.complex_pairs),
            ("coordinate_unit", self.coordinate_unit),
            ("specs", self.specs),
        ):
            if len(values) != batch:
                raise ValueError(f"{field_name} 数量 {len(values)} 与 batch={batch} 不一致")
        for key, values in self.metadata.items():
            if len(values) != batch:
                raise ValueError(f"metadata[{key!r}] 数量 {len(values)} 与 batch={batch} 不一致")

    @property
    def iq(self) -> torch.Tensor:
        """旧代码兼容别名；不表示输入一定只有两个通道。"""
        return self.values

    @property
    def lengths(self) -> torch.Tensor:
        return self.sample_mask.sum(dim=-1).to(dtype=torch.long)

    @property
    def batch_size(self) -> int:
        return int(self.values.shape[0])

    def to(self, *args: Any, **kwargs: Any) -> SignalBatch:
        moved_values = self.values.to(*args, **kwargs)
        device = moved_values.device
        return replace(
            self,
            values=moved_values,
            sample_mask=self.sample_mask.to(device=device),
            channel_mask=self.channel_mask.to(device=device),
            time_coordinates=self.time_coordinates.to(device=device),
            sample_rate_hz=self.sample_rate_hz.to(device=device),
        )

    def to_legacy_dict(self) -> dict[str, Any]:
        per_sample_specs = [spec.to_model_dict() for spec in self.specs]
        uniform_spec = (
            all(spec == per_sample_specs[0] for spec in per_sample_specs)
            and self.specs[0].num_channels == self.values.shape[1]
        )
        if uniform_spec:
            model_spec = per_sample_specs[0]
        else:
            model_spec = {
                "name": "mixed",
                "num_channels": int(self.values.shape[1]),
                "modality": "mixed",
                "complex_pairs": (),
                "adapter_key": None,
            }
        out: dict[str, Any] = {
            "iq": self.values,
            "values": self.values,
            "sample_mask": self.sample_mask,
            "channel_mask": self.channel_mask,
            "time_coordinates": self.time_coordinates,
            "sample_rate_hz": self.sample_rate_hz,
            "length": self.lengths,
            "modality_id": list(self.modality_id),
            "complex_pairs": list(self.complex_pairs),
            "coordinate_unit": list(self.coordinate_unit),
            "signal_spec": model_spec,
            "signal_specs": per_sample_specs,
            "signal_contract_version": self.schema_version,
            "capture_metadata": dict(self.metadata),
        }
        out.update({key: list(values) for key, values in self.metadata.items()})
        return out

    @classmethod
    def from_legacy_dict(cls, batch: Mapping[str, Any]) -> SignalBatch:
        values = batch.get("values", batch.get("iq"))
        if not torch.is_tensor(values) or values.ndim != 3:
            raise ValueError("legacy batch 必须含 [B,C,L] 的 values 或 iq Tensor")
        batch_size, channels, length = values.shape
        sample_mask = batch.get("sample_mask")
        if sample_mask is None:
            sample_mask = torch.ones(batch_size, length, dtype=torch.bool, device=values.device)
        else:
            sample_mask = torch.as_tensor(sample_mask, dtype=torch.bool, device=values.device)
            if sample_mask.ndim == 1:
                sample_mask = sample_mask.unsqueeze(0).expand(batch_size, -1)
            elif sample_mask.shape[0] == 1 and batch_size > 1:
                sample_mask = sample_mask.expand(batch_size, -1)
        channel_mask = batch.get("channel_mask")
        if channel_mask is None:
            channel_mask = torch.ones(batch_size, channels, dtype=torch.bool, device=values.device)
        else:
            channel_mask = torch.as_tensor(channel_mask, dtype=torch.bool, device=values.device)
            if channel_mask.ndim == 1:
                channel_mask = channel_mask.unsqueeze(0).expand(batch_size, -1)
            elif channel_mask.shape[0] == 1 and batch_size > 1:
                channel_mask = channel_mask.expand(batch_size, -1)
        rates = batch.get("sample_rate_hz")
        if rates is None:
            rates = torch.full((batch_size,), float("nan"), dtype=torch.float32, device=values.device)
        else:
            rates = torch.as_tensor(rates, dtype=torch.float32, device=values.device)
            if rates.ndim == 0:
                rates = rates.repeat(batch_size)
            elif rates.numel() == 1 and batch_size > 1:
                rates = rates.reshape(1).repeat(batch_size)
        coordinates = batch.get("time_coordinates")
        if coordinates is None:
            coordinates = torch.arange(length, dtype=torch.float32, device=values.device).expand(
                batch_size, -1
            )
        else:
            coordinates = torch.as_tensor(coordinates, dtype=torch.float32, device=values.device)
            if coordinates.ndim == 1:
                coordinates = coordinates.unsqueeze(0).expand(batch_size, -1)
            elif coordinates.shape[0] == 1 and batch_size > 1:
                coordinates = coordinates.expand(batch_size, -1)
        raw_modalities = batch.get("modality_id", ["rf"] * batch_size)
        if isinstance(raw_modalities, str):
            raw_modalities = [raw_modalities] * batch_size
        modalities = tuple(str(v) for v in raw_modalities)
        raw_pairs = batch.get(
            "complex_pairs",
            [((0, 1),) if channels == 2 else () for _ in range(batch_size)],
        )
        if isinstance(raw_pairs, (list, tuple)) and len(raw_pairs) == 0:
            raw_pairs = [()] * batch_size
        elif (
            isinstance(raw_pairs, (list, tuple))
            and len(raw_pairs) > 0
            and isinstance(raw_pairs[0], (list, tuple))
            and len(raw_pairs[0]) == 2
            and all(isinstance(value, int) for value in raw_pairs[0])
        ):
            raw_pairs = [raw_pairs] * batch_size
        elif isinstance(raw_pairs, (list, tuple)) and len(raw_pairs) == 1 and batch_size > 1:
            raw_pairs = list(raw_pairs) * batch_size
        pairs = tuple(_normalize_complex_pairs(value) for value in raw_pairs)
        raw_units = batch.get("coordinate_unit", ["sample_index"] * batch_size)
        if isinstance(raw_units, str):
            raw_units = [raw_units] * batch_size
        units = tuple(str(v) for v in raw_units)
        specs_raw = batch.get("signal_specs", batch.get("signal_spec"))
        if specs_raw is None:
            specs = tuple(
                SignalSpec(
                    num_channels=int(channel_mask[i].sum().item()),
                    modality_id=modalities[i],
                    sample_rate_hz=float(rates[i]) if _valid_sample_rate(float(rates[i])) else None,
                    coordinate_unit=units[i],
                    complex_pairs=pairs[i],
                )
                for i in range(batch_size)
            )
        else:
            if isinstance(specs_raw, (SignalSpec, Mapping)):
                specs_raw = [specs_raw] * batch_size
            converted: list[SignalSpec] = []
            for index, raw_spec in enumerate(specs_raw):
                if isinstance(raw_spec, SignalSpec):
                    converted.append(raw_spec)
                    continue
                if not isinstance(raw_spec, Mapping):
                    raise TypeError("signal_spec/signal_specs 必须为 SignalSpec 或 mapping")
                converted.append(
                    SignalSpec(
                        num_channels=int(raw_spec.get("num_channels", int(channel_mask[index].sum()))),
                        modality_id=str(
                            raw_spec.get("modality_id", raw_spec.get("modality", modalities[index]))
                        ),
                        complex_pairs=_normalize_complex_pairs(raw_spec.get("complex_pairs")),
                        sample_rate_hz=(
                            float(rates[index])
                            if _valid_sample_rate(float(rates[index]))
                            else None
                        ),
                        coordinate_unit=units[index],
                    )
                )
            specs = tuple(converted)
        metadata_raw = batch.get("capture_metadata", {})
        metadata = {}
        for key in CAPTURE_METADATA_KEYS:
            raw_values = metadata_raw.get(key, batch.get(key, [MISSING_METADATA] * batch_size))
            if isinstance(raw_values, (str, bytes)) or not hasattr(raw_values, "__len__"):
                raw_values = [raw_values] * batch_size
            metadata[key] = tuple(raw_values)
        return cls(
            values=values,
            sample_mask=sample_mask,
            channel_mask=channel_mask,
            time_coordinates=coordinates,
            sample_rate_hz=rates.to(dtype=torch.float32),
            modality_id=modalities,
            complex_pairs=pairs,
            coordinate_unit=units,
            metadata=metadata,
            specs=specs,
        )


def _sample_values(sample: Mapping[str, Any]) -> torch.Tensor:
    values = sample.get("values", sample.get("iq"))
    if not torch.is_tensor(values):
        values = torch.as_tensor(values)
    if values.ndim == 1:
        values = values.unsqueeze(0)
    if values.ndim != 2:
        raise ValueError(f"单样本 values/iq 必须为 [C,L]，实际为 {tuple(values.shape)}")
    return values


def _sample_rate(sample: Mapping[str, Any]) -> float:
    for key in ("sample_rate_hz", "sampling_rate"):
        value = sample.get(key)
        if _valid_sample_rate(value):
            return float(value)
    return float("nan")


def collate_signal_batch(samples: Sequence[Mapping[str, Any]]) -> SignalBatch:
    if not samples:
        raise ValueError("不能 collate 空 batch")
    tensors = [_sample_values(sample) for sample in samples]
    first = tensors[0]
    if any(tensor.dtype != first.dtype for tensor in tensors):
        tensors = [tensor.to(dtype=torch.float32) for tensor in tensors]
        first = tensors[0]
    batch_size = len(tensors)
    max_channels = max(int(tensor.shape[0]) for tensor in tensors)
    max_length = max(int(tensor.shape[-1]) for tensor in tensors)
    values = torch.zeros(
        batch_size,
        max_channels,
        max_length,
        dtype=first.dtype,
        device=first.device,
    )
    sample_mask = torch.zeros(batch_size, max_length, dtype=torch.bool, device=first.device)
    channel_mask = torch.zeros(batch_size, max_channels, dtype=torch.bool, device=first.device)
    coordinates = torch.full(
        (batch_size, max_length),
        float("nan"),
        dtype=torch.float32,
        device=first.device,
    )
    rates = torch.full((batch_size,), float("nan"), dtype=torch.float32, device=first.device)
    modalities: list[str] = []
    all_pairs: list[tuple[tuple[int, int], ...]] = []
    coordinate_units: list[str] = []
    specs: list[SignalSpec] = []
    metadata = {key: [] for key in CAPTURE_METADATA_KEYS}

    for index, (sample, tensor) in enumerate(zip(samples, tensors, strict=True)):
        channels, tensor_length = (int(tensor.shape[0]), int(tensor.shape[-1]))
        declared_length = int(sample.get("length", tensor_length))
        valid_length = max(0, min(declared_length, tensor_length))
        values[index, :channels, :tensor_length] = tensor

        raw_sample_mask = sample.get("sample_mask")
        if raw_sample_mask is None:
            sample_mask[index, :valid_length] = True
        else:
            raw_mask = torch.as_tensor(raw_sample_mask, dtype=torch.bool, device=first.device).reshape(-1)
            mask_length = min(max_length, int(raw_mask.numel()), tensor_length, valid_length)
            sample_mask[index, :mask_length] = raw_mask[:mask_length]

        raw_channel_mask = sample.get("channel_mask")
        if raw_channel_mask is None:
            channel_mask[index, :channels] = True
        else:
            raw_mask = torch.as_tensor(raw_channel_mask, dtype=torch.bool, device=first.device).reshape(-1)
            mask_channels = min(channels, int(raw_mask.numel()))
            channel_mask[index, :mask_channels] = raw_mask[:mask_channels]

        rate = _sample_rate(sample)
        if _valid_sample_rate(rate):
            rates[index] = rate
        raw_coordinates = sample.get("time_coordinates", sample.get("coordinates"))
        if raw_coordinates is None:
            base = torch.arange(tensor_length, dtype=torch.float32, device=first.device)
            if _valid_sample_rate(rate):
                base = base / rate
                coordinate_unit = "seconds"
            else:
                coordinate_unit = "sample_index"
        else:
            base = torch.as_tensor(raw_coordinates, dtype=torch.float32, device=first.device).reshape(-1)
            if base.numel() < tensor_length:
                raise ValueError(
                    f"样本 {index} 的坐标长度 {base.numel()} 小于信号长度 {tensor_length}"
                )
            coordinate_unit = str(sample.get("coordinate_unit", "continuous"))
        coordinates[index, :tensor_length] = base[:tensor_length]

        modality = str(sample.get("modality_id", "rf" if "iq" in sample else "unknown"))
        raw_pairs = sample.get("complex_pairs")
        pairs = _normalize_complex_pairs(
            raw_pairs if raw_pairs is not None else (((0, 1),) if modality == "rf" and channels == 2 else ())
        )
        raw_spec = sample.get("signal_spec")
        if raw_spec is None:
            spec = SignalSpec(
                num_channels=channels,
                modality_id=modality,
                sample_rate_hz=rate if _valid_sample_rate(rate) else None,
                coordinate_unit=coordinate_unit,
                complex_pairs=pairs,
            )
        elif isinstance(raw_spec, SignalSpec):
            spec = raw_spec
            if spec.num_channels != channels:
                raise ValueError(
                    f"样本 {index} 的 SignalSpec 声明 {spec.num_channels} 通道，实际为 {channels}"
                )
        else:
            raise TypeError(f"样本 {index} 的 signal_spec 必须为 SignalSpec")
        modalities.append(modality)
        all_pairs.append(pairs)
        coordinate_units.append(coordinate_unit)
        specs.append(spec)
        for key in CAPTURE_METADATA_KEYS:
            value = sample.get(key, MISSING_METADATA)
            metadata[key].append(MISSING_METADATA if value is None or str(value) == "" else str(value))

    return SignalBatch(
        values=values,
        sample_mask=sample_mask,
        channel_mask=channel_mask,
        time_coordinates=coordinates,
        sample_rate_hz=rates,
        modality_id=tuple(modalities),
        complex_pairs=tuple(all_pairs),
        coordinate_unit=tuple(coordinate_units),
        metadata={key: tuple(values) for key, values in metadata.items()},
        specs=tuple(specs),
    )
