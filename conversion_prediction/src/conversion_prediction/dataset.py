"""将预处理后的 session 列表转换成可供 PyTorch 训练的批次。"""

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
        self.frame = frame.reset_index(drop=True)
        self.labels = self.frame["label"].to_numpy(dtype=np.float32)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> Dict[str, object]:
        # Parquet 中的 list 列在这里统一转换为 int64，供 Embedding 查表使用。
        row = self.frame.iloc[index]
        sample = {name: np.asarray(row[name], dtype=np.int64) for name in SEQUENCE_COLUMNS}
        sample.update(
            session_id=str(row["session_id"]),
            label=float(row["label"]),
        )
        return sample


def make_collate_fn(max_seq_len: int):
    """生成批处理函数，在每个 batch 内动态补零并截断过长序列。"""

    def collate(samples: Sequence[Dict[str, object]]) -> Dict[str, object]:
        batch_size = len(samples)
        lengths = [min(len(sample["event_ids"]), max_seq_len) for sample in samples]
        width = max(lengths)
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
