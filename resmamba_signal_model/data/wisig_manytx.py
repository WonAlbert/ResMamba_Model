"""WiSig ManyTx.pkl 流式解析：在有限内存下逐块读取并写出样本。"""
from __future__ import annotations

import pickle
import pickle as _pickle_mod
import hashlib
import inspect
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from resmamba_signal_model.data.contracts import MISSING_METADATA


WISIG_GROUP_SPLIT_STRATEGY = "receiver_day_group_held_out_v1"


@dataclass
class WiSigManyTxMeta:
    tx_list: list[str] = field(default_factory=list)
    rx_list: list[str] = field(default_factory=list)
    capture_date_list: list[str] = field(default_factory=list)
    equalized_list: list[int] = field(default_factory=list)
    max_sig: int | None = None
    split_strategy: str = WISIG_GROUP_SPLIT_STRATEGY
    group_assignments: dict[str, str] = field(default_factory=dict)
    receiver_ids: dict[str, str] = field(default_factory=dict)
    session_ids: dict[str, str] = field(default_factory=dict)


class _NestList(list):
    """WiSig data 四维嵌套列表；depth=3 时收到 ndarray 后立即交给 sink。"""

    __slots__ = ("depth", "parent", "index")

    def __init__(self, depth: int = 0, parent: _NestList | None = None) -> None:
        super().__init__()
        self.depth = depth
        self.parent = parent
        self.index = -1

    def append(self, item: object) -> None:
        if self.depth == 3 and isinstance(item, np.ndarray):
            tx_i, rx_i, day_i, eq_i = self._indices()
            _ACTIVE_SINK.on_block(item, tx_i, rx_i, day_i, eq_i)
            return
        if self.depth < 3 and isinstance(item, list) and not isinstance(item, _NestList):
            item = _NestList(self.depth + 1, parent=self)
        super().append(item)
        if isinstance(item, _NestList):
            item.index = len(self) - 1
            item.parent = self

    def extend(self, items) -> None:
        base = len(self)
        for offset, item in enumerate(items):
            if isinstance(item, _NestList):
                item.parent = self
                item.index = base + offset
            self.append(item)

    def _indices(self) -> tuple[int, int, int, int]:
        eq_list = self
        day_subtree = eq_list.parent
        rx_subtree = day_subtree.parent if day_subtree is not None else None
        tx_subtree = rx_subtree.parent if rx_subtree is not None else None
        if tx_subtree is None or rx_subtree is None or day_subtree is None:
            raise RuntimeError("WiSig data 嵌套结构不完整")
        return tx_subtree.index, rx_subtree.index, day_subtree.index, eq_list.index


class _TopDict(dict):
    def __setitem__(self, key: str, value: object) -> None:
        if key == "data":
            value = []
        super().__setitem__(key, value)


class _WiSigUnpickler(_pickle_mod._Unpickler):
    def __init__(self, file) -> None:
        super().__init__(file)
        self._await_data_root = False
        self._inside_data = False
        self._list_stack: list[_NestList] = []
        self.dispatch = self.dispatch.copy()
        self.dispatch[_pickle_mod.BINUNICODE[0]] = _WiSigUnpickler.load_binunicode
        self.dispatch[_pickle_mod.SHORT_BINUNICODE[0]] = _WiSigUnpickler.load_short_binunicode
        self.dispatch[_pickle_mod.EMPTY_LIST[0]] = _WiSigUnpickler.load_empty_list
        self.dispatch[_pickle_mod.APPEND[0]] = _WiSigUnpickler.load_append
        self.dispatch[_pickle_mod.APPENDS[0]] = _WiSigUnpickler.load_appends
        self.dispatch[_pickle_mod.BINPUT[0]] = _WiSigUnpickler.load_binput
        self.dispatch[_pickle_mod.LONG_BINPUT[0]] = _WiSigUnpickler.load_long_binput

    def load_binput(self) -> None:
        i = self.read(1)[0]
        if not (self._inside_data and isinstance(self.stack[-1], np.ndarray)):
            self.memo[i] = self.stack[-1]

    def load_long_binput(self) -> None:
        (i,) = struct.unpack("<I", self.read(4))
        if not (self._inside_data and isinstance(self.stack[-1], np.ndarray)):
            self.memo[i] = self.stack[-1]

    def load_binunicode(self) -> None:
        (n,) = struct.unpack("<I", self.read(4))
        data = self.read(n)
        text = str(data, "utf-8", "surrogatepass")
        if text == "data":
            self._await_data_root = True
        self.append(text)

    def load_short_binunicode(self) -> None:
        n = self.read(1)[0]
        data = self.read(n)
        text = str(data, "utf-8", "surrogatepass")
        if text == "data":
            self._await_data_root = True
        self.append(text)

    def load_empty_list(self) -> None:
        if self._await_data_root:
            self._await_data_root = False
            self._inside_data = True
            self._list_stack = []
            lst = _NestList(0)
            self._list_stack.append(lst)
            self.append(lst)
            return
        if self._inside_data:
            parent = self._nest_target()
            if parent is not None and parent.depth < 3:
                depth = parent.depth + 1
                self._list_stack = self._list_stack[:depth]
                lst = _NestList(depth, parent=parent)
                self._list_stack.append(lst)
                self.append(lst)
                return
        self.append([])

    def _nest_target(self) -> _NestList | None:
        for i in range(len(self.stack) - 1, -1, -1):
            if isinstance(self.stack[i], _NestList):
                return self.stack[i]
        return None

    def load_append(self) -> None:
        if self._inside_data and len(self.stack) >= 2 and isinstance(self.stack[-2], _NestList):
            list_obj = self.stack[-2]
            item = self.stack[-1]
            list_obj.append(item)
            if isinstance(item, _NestList) and self._list_stack and self._list_stack[-1] is item:
                self._list_stack.pop()
            del self.stack[-2:]
            return
        super().load_append()

    def load_appends(self) -> None:
        if not self._inside_data or not self.metastack:
            super().load_appends()
            return
        items = self.pop_mark()
        target = self.stack[-1] if self.stack else None
        if isinstance(target, _NestList):
            target.extend(items)
            for item in items:
                if isinstance(item, _NestList) and self._list_stack:
                    self._list_stack.pop()
            del items
            return
        if isinstance(target, list):
            kept: list[object] = []
            for item in items:
                if isinstance(item, np.ndarray) and self._list_stack:
                    leaf = self._list_stack[-1]
                    if isinstance(leaf, _NestList) and leaf.depth == 3:
                        tx_i, rx_i, day_i, eq_i = leaf._indices()
                        _ACTIVE_SINK.on_block(item, tx_i, rx_i, day_i, eq_i)
                        continue
                kept.append(item)
            if kept:
                target.extend(kept)
            del items
            return
        raise pickle.UnpicklingError("WiSig data APPENDS 目标无效")

    def find_class(self, module: str, name: str):
        if module == "builtins" and name == "dict":
            return _TopDict
        return super().find_class(module, name)


class WiSigBlockSink:
    """将完整 Rx/Day capture 分配给唯一 split 后交给回调。

    旧实现会在每个 ``(Tx,Rx,Day,Eq)`` 块内随机拆样本，导致同一 capture
    同时出现在训练和验证/测试中。这里用稳定哈希按 ``(Rx,Day,Eq)`` 分组，
    因而不同 Tx 在同一接收机和日期上的样本也不会跨 split。

    ``on_batch`` 继续兼容旧的三参数签名；四参数签名会额外收到 capture
    元数据字典。
    """

    def __init__(
        self,
        *,
        equalized: int = 0,
        train_frac: float = 0.6,
        val_frac: float = 0.2,
        seed: int = 20260629,
        on_batch: Callable[..., None],
        group_assignments: Mapping[str, str] | None = None,
    ) -> None:
        if not 0 <= train_frac <= 1 or not 0 <= val_frac <= 1 or train_frac + val_frac > 1:
            raise ValueError("WiSig split 比例必须满足 train>=0、val>=0 且 train+val<=1")
        self.equalized = equalized
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.seed = int(seed)
        self.on_batch = on_batch
        self.group_assignments: dict[str, str] = dict(group_assignments or {})
        invalid = set(self.group_assignments.values()) - {"train", "val", "test"}
        if invalid:
            raise ValueError(f"WiSig group_assignments 含非法 split: {sorted(invalid)}")
        self._callback_accepts_metadata = self._accepts_metadata(on_batch)
        self.raw = 0

    @staticmethod
    def _accepts_metadata(callback: Callable[..., None]) -> bool:
        try:
            parameters = tuple(inspect.signature(callback).parameters.values())
        except (TypeError, ValueError):
            return False
        return any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in parameters) or len(parameters) >= 4

    @staticmethod
    def group_id(rx_i: int, day_i: int, eq_i: int) -> str:
        return f"wisig:rx={int(rx_i)}:day={int(day_i)}:eq={int(eq_i)}"

    def split_for_group(self, group_id: str) -> str:
        existing = self.group_assignments.get(group_id)
        if existing is not None:
            return existing
        payload = f"{self.seed}:{group_id}".encode("utf-8")
        digest = hashlib.sha256(payload).digest()
        score = int.from_bytes(digest[:8], "big") / float(1 << 64)
        if score < self.train_frac:
            split = "train"
        elif score < self.train_frac + self.val_frac:
            split = "val"
        else:
            split = "test"
        self.group_assignments[group_id] = split
        return split

    @staticmethod
    def block_metadata(rx_i: int, day_i: int, eq_i: int) -> dict[str, str]:
        capture_id = WiSigBlockSink.group_id(rx_i, day_i, eq_i)
        return {
            "receiver_id": f"wisig:rx={int(rx_i)}",
            "session_id": f"wisig:day={int(day_i)}",
            "channel_id": MISSING_METADATA,
            "capture_id": capture_id,
            # 原始列表在 pickle 完成后才可用；索引是可验证且不虚构的稳定标识。
            "capture_date": f"wisig:day_index={int(day_i)}",
        }

    def on_block(self, arr: np.ndarray, tx_i: int, rx_i: int, day_i: int, eq_i: int) -> None:
        if eq_i != self.equalized:
            return
        arr = np.asarray(arr)
        if arr.size == 0 or arr.shape[0] == 0:
            return
        iq = np.transpose(arr, (0, 2, 1))
        n = len(iq)
        self.raw += n
        group_id = self.group_id(rx_i, day_i, eq_i)
        split = self.split_for_group(group_id)
        labels = np.full(n, np.int32(tx_i), dtype=np.int32)
        if self._callback_accepts_metadata:
            self.on_batch(split, iq, labels, self.block_metadata(rx_i, day_i, eq_i))
        else:
            self.on_batch(split, iq, labels)


_ACTIVE_SINK: WiSigBlockSink


def iterate_manytx_payload(payload: Mapping[str, Any], sink: WiSigBlockSink) -> None:
    """从已加载的 ManyTx dict 逐块写出 IQ，供标准 pickle 路径使用。"""
    signal = payload.get("data")
    if not isinstance(signal, list):
        return
    for tx_i, tx_blocks in enumerate(signal):
        if not isinstance(tx_blocks, list):
            continue
        for rx_i, rx_blocks in enumerate(tx_blocks):
            if not isinstance(rx_blocks, list):
                continue
            for day_i, day_blocks in enumerate(rx_blocks):
                if not isinstance(day_blocks, list):
                    continue
                for eq_i, arr in enumerate(day_blocks):
                    if isinstance(arr, np.ndarray):
                        sink.on_block(arr, tx_i, rx_i, day_i, eq_i)


def stream_manytx_blocks(
    pkl_path: Path,
    sink: WiSigBlockSink,
) -> WiSigManyTxMeta:
    global _ACTIVE_SINK
    _ACTIVE_SINK = sink
    with pkl_path.open("rb") as f:
        payload = _WiSigUnpickler(f).load()
    if sink.raw == 0 and isinstance(payload, Mapping):
        with pkl_path.open("rb") as f:
            payload = pickle.load(f)
        iterate_manytx_payload(payload, sink)
    meta = WiSigManyTxMeta(
        tx_list=list(payload.get("tx_list", [])),
        rx_list=list(payload.get("rx_list", [])),
        capture_date_list=list(payload.get("capture_date_list", [])),
        equalized_list=[int(v) for v in payload.get("equalized_list", [])],
        max_sig=payload.get("max_sig"),
        group_assignments=dict(sink.group_assignments),
    )
    meta.receiver_ids = {
        f"wisig:rx={idx}": str(name)
        for idx, name in enumerate(meta.rx_list)
    }
    meta.session_ids = {
        f"wisig:day={idx}": str(name)
        for idx, name in enumerate(meta.capture_date_list)
    }
    return meta


def emitter_labels(meta: WiSigManyTxMeta) -> dict[str, int]:
    return {str(name): idx for idx, name in enumerate(meta.tx_list)}
