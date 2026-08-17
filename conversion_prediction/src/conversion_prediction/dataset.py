"""
将预处理后的 session 列表转换成可供 PyTorch 训练的批次。
把预处理产出的「变长 list」对齐成 PyTorch 能训练的「定长 batch + mask」
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


SEQUENCE_COLUMNS = (
    "event_ids",
    "event_type_ids",
    "page_ids",
    "category_ids",
    "poi_ids",
    "stage_ids",
    "time_gap_ids",
)


class SessionDataset(Dataset):
    """按数据集切分读取 session，单个样本仍保留变长序列。"""

    def __init__(self, path: str | Path, split: str, sample_limit: int | None = None):
        # Parquet 过滤会只读取目标 split，避免先加载全量数据再切片。
        frame = pd.read_parquet(path, filters=[("split", "==", split)])
        if sample_limit:
            frame = frame.head(int(sample_limit))
        frame = frame.reset_index(drop=True)

        # 一次性把 list 列转成 numpy 数组并常驻内存，供 __getitem__ 直接按行索引。
        # 之前用 self.frame.iloc[index] 逐行取数走 pandas 慢路径，50 万行 × 每个epoch 都会成为训练的数据加载瓶颈，这里用底层数组切片把它消除
        self.sequences: Dict[str, np.ndarray] = {
            name: frame[name].to_numpy() for name in SEQUENCE_COLUMNS
        }
        self.session_ids: np.ndarray = frame["session_id"].astype(str).to_numpy()
        self.labels: np.ndarray = frame["label"].to_numpy(dtype=np.float32)
        self._length: int = len(self.labels)

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int) -> Dict[str, object]:
        # Parquet 中的 list 列在这里统一转换为 int64，供 Embedding 查表使用。
        sample: Dict[str, object] = {}
        for name in SEQUENCE_COLUMNS:
            sample[name] = np.asarray(self.sequences[name][index], dtype=np.int64)
        sample["session_id"] = str(self.session_ids[index])
        sample["label"] = float(self.labels[index])
        return sample


def make_collate_fn(max_seq_len: int):
    """ 变长序列 → 定长序列
    生成批处理函数，在每个 batch 内动态补零并截断过长序列，对齐序列长度。
    """

    def collate(samples: Sequence[Dict[str, np.ndarray]]) -> Dict[str, object]:
        batch_size = len(samples)
        lengths = [min(len(sample["event_ids"]), max_seq_len) for sample in samples] #每个样本的序列长度，对于超长序列截断到max_seq_len
        width = max(lengths) # 这批batch内样本的统一长度
        batch: Dict[str, object] = {}

        # 保留最近 max_seq_len 个行为，因为临近预测时点的行为通常最有价值。
        for name in SEQUENCE_COLUMNS:
            values = np.zeros((batch_size, width), dtype=np.int64) 
            for index, sample in enumerate(samples):
                sequence = sample[name][-max_seq_len:] 
                values[index, : len(sequence)] = sequence
            batch[name] = torch.from_numpy(values)

        # position 表示行为在截断后序列中的相对顺序；mask 区分真实行为和 padding。
        positions = np.arange(width, dtype=np.int64)[None, :].repeat(batch_size, axis=0)
        batch["positions"] = torch.from_numpy(positions)
        batch["mask"] = torch.arange(width)[None, :] < torch.tensor(lengths)[:, None]
        batch["lengths"] = torch.tensor(lengths, dtype=torch.long)
        batch["labels"] = torch.tensor([sample["label"] for sample in samples], dtype=torch.float32)
        batch["session_ids"] = [sample["session_id"] for sample in samples]
        return batch

    return collate
