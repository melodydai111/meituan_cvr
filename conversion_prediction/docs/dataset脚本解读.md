# Dataset 脚本（dataset.py）解读与发现

> 对象：`conversion_prediction/src/conversion_prediction/dataset.py`
> 生成日期：2026-08-16
> 配套阅读：`预处理脚本解读.md`（上游：变长 list 样本从哪来）、`../README.md`（实验设计）

---

## 一、一句话概述

`dataset.py` 把预处理产出的**变长 session 序列**（`session_samples.parquet` 里的 list 列）转成 PyTorch 能训练的**定长 batch + mask**。它填补的是整条链路的第 ④ 步缺口：「变长 → 定长」。

**上游**：`preprocessing.py` 产出 `session_samples.parquet`（一行一条 session，7 个 list 列 + label + split）
**下游**：`models.py` 的 `forward` 消费 batch（整数序列 + mask + positions + labels）

整个文件只有两个东西：一个 `SessionDataset` 类（按条取样本），一个 `make_collate_fn` 工厂函数（把一批样本对齐成张量）。

---

## 二、职责定位（承上启下）

```
preprocessing.py             dataset.py                 models.py
─────────────────           ─────────────              ─────────────
变长 list 样本   ──────►   定长 batch + mask   ──────►  Embedding + Transformer
(session_samples.parquet)   (tensor [B, width, 64])    (概率输出)
```

预处理只负责「行 → 序列 + 编码」，**不负责对齐**；对齐（截断、补零、mask）全部在 dataset.py 完成。

---

## 三、核心组件详解

### 3.1 `SEQUENCE_COLUMNS` 常量

```python
SEQUENCE_COLUMNS = ("event_ids", "event_type_ids", "page_ids",
                    "category_ids", "poi_ids", "stage_ids", "time_gap_ids")
```

就是预处理 `sessions` CTE 产出的 7 个 list 列。**注意：这里 7 列无条件全部加载**，不管当前实验用哪个 `feature_set`。用不用由 `models.py` 决定（`FEATURE_SETS` 控制 `use_context/use_stage/use_time`），dataset 层不挑。代价是内存略浪费，好处是 dataset 层不用感知 feature_set、逻辑简单。

### 3.2 `SessionDataset` —— 按条取样本

**`__init__`**：

```python
frame = pd.read_parquet(path, filters=[("split", "==", split)])
```

- 用 pyarrow 的 **filter 下推**，只读目标 split 的行（train/val/test 各加载各的），**不把 50 万行全载入再切片**。
- 可选 `sample_limit` 截取前 N 条（调试用）。
- 关键优化：把 7 个 list 列**一次性转成 numpy 数组常驻内存**。之前用 `self.frame.iloc[index]` 逐行取走 pandas 慢路径，50 万行 × 每 epoch 都取会成为数据加载瓶颈；改成底层数组切片后 `__getitem__` 直接按行索引。

**`__getitem__`**：返回**一条 session**（仍为变长 numpy 数组 + session_id + label）：

```python
sample[name] = np.asarray(self.sequences[name][index], dtype=np.int64)   # 该 session 的整数序列
sample["session_id"] = str(self.session_ids[index])                       # numpy.str_ → Python str
sample["label"]      = float(self.labels[index])                          # numpy.float32 → Python float
```

两处 `str()` / `float()` 的作用是**把 numpy 标量换成 Python 原生标量**：

- `session_id` 这个转换是**刚需**——它是纯标识符，后续要导出/序列化（如解释结果写 JSON）。`numpy.str_` 在 `json.dumps` 时会抛 `TypeError`，转成 Python `str` 才安全。
- `label` 的 `float()` 更多是防御性/图干净——`torch.tensor([np.float32(1.0)])` 其实也能工作，但返回规范 Python 标量让调用方不依赖"这是 numpy 还是 Python 标量"。

序列列不这么转，是因为它们要继续以 numpy 数组形态被 collate 切片、补零。

### 3.3 `make_collate_fn` —— 工厂函数 + collate

**为什么外面套一层？** 因为 `DataLoader` 的 `collate_fn` 必须是**固定签名**的函数——它只会收到一个参数（这批样本的 list）。你没法通过 DataLoader 给 collate 传 `max_seq_len`。所以用工厂函数把 `max_seq_len` 通过闭包"封进去"：

```python
# training.py
collate = make_collate_fn(config["data"]["max_seq_len"])   # 先把 50 封进闭包
DataLoader(..., collate_fn=collate)                         # 只传 collate 本身
```

**`collate` 内部逐行**：

```python
lengths = [min(len(sample["event_ids"]), max_seq_len) ...]  # 每条先按 50 封顶
width   = max(lengths)                                       # 本 batch 对齐宽度
```

关键机制一：**动态 padding**。`width = max(lengths)`，即「本 batch 最长的那个（≤50）」，不是固定 50。一个全短会话的 batch 就补到 7，而不是白白补到 50 浪费 90% 算力。Transformer 靠 mask 处理变长，不需要固定宽度。

```python
values = np.zeros((batch_size, width), dtype=np.int64)
sequence = sample[name][-max_seq_len:]      # 尾部截断
values[index, : len(sequence)] = sequence   # 左对齐填入，右侧补 0
```

关键机制二：**`max_seq_len` 与 `width` 分工不同**：

| 变量 | 形态 | 职责 | 用在 |
|---|---|---|---|
| `max_seq_len`(50) | 标量(常量) | **截断**单条样本(全局上限) | `[-max_seq_len:]` |
| `width` | 标量(每 batch 变) | **补零**对齐 batch | `np.zeros((batch, width))` |
| `lengths` | 列表(每样本一个) | 每个样本多长(画 mask) | `mask`、`batch["lengths"]` |

- 截断 `[-max_seq_len:]` 用 50：把单条超过 50 的砍成**最后 50 步**（临近下单的行为最值钱）。
- 补零用 `width`：决定这个 batch 的张量多宽。
- `lengths` 是「每个样本各自多长」的清单，`width = max(lengths)` 是从它派生的标量；两者职责不同，不能互相替代。

（`[-width:]` 与 `[-max_seq_len:]` 碰巧结果等价——`width ≤ 50`，且 `width < 50` 时所有样本都短于 50、无需截断——但语义上截断该用全局上限 `max_seq_len`。）

```python
positions = np.arange(width).repeat(batch_size)               # 相对位置 [0,1,2,...]
mask      = torch.arange(width)[None, :] < torch.tensor(lengths)[:, None]   # 真行为 True
```

- `positions`：截断后重新从 0 编号，喂给 position embedding。
- `mask`：拿 `[0..width-1]` 和**每个样本各自的 length** 比，标记哪些位置是真实行为、哪些是补的 0。这里必须用 `lengths`（列表），光有 `width` 分不出"样本 1 只有 3 长"。

---

## 四、完整代码（原样保留）

```python
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

    def collate(samples: Sequence[Dict[str, object]]) -> Dict[str, object]:
        batch_size = len(samples)
        lengths = [min(len(sample["event_ids"]), max_seq_len) for sample in samples] # pyright: ignore[reportArgumentType]
        width = max(lengths)
        batch: Dict[str, object] = {}

        # 保留最近 max_seq_len 个行为，因为临近预测时点的行为通常最有价值。
        for name in SEQUENCE_COLUMNS:
            values = np.zeros((batch_size, width), dtype=np.int64)
            for index, sample in enumerate(samples):
                sequence = sample[name][-max_seq_len:] # pyright: ignore[reportIndexIssue]
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
```

---

## 五、一个 batch 的完整走查（动态 padding）

假设 `max_seq_len = 50`，两个不同 batch。

**Batch 1：长度 120 / 30 / 45**（有一条超长）

```python
lengths = [min(120,50), min(30,50), min(45,50)] = [50, 30, 45]
width   = max(50, 30, 45) = 50
```

```
        ┌────────────────── 50 列 ──────────────────┐
A: [e71 … e120                                  ]    ← 120→截断到后 50,填满
B: [e1 … e30  0 0 0 … 0                        ]    ← 30 个 + 20 个 0
C: [e1 … e45  0 0 0 0 0                        ]    ← 45 个 + 5 个 0
```

**Batch 2：长度 7 / 3 / 5**（全短）

```python
lengths = [7, 3, 5]
width   = 7
```

```
        ┌── 7 列 ──┐
A: [e1 … e7      ]    ← 填满
B: [e1 e2 e3 0 0 0 0] ← 3 个 + 4 个 0
C: [e1 … e5 0 0]    ← 5 个 + 2 个 0
```

**结论**：Transformer 输入宽度 = `width`，**每个 batch 不同**（50 / 7），不是固定 50。`max_seq_len` 只在「截断单条」时起作用（Batch 1 的 A 从 120 砍到 50），不决定张量宽度。

最终 batch 的 key：

```python
batch = {
  "event_ids":   tensor([[...], [...]])  # 7 个 *_ids 列, shape [B, width]
  "positions":   tensor([[...]]),        # [0,1,...,width-1]
  "mask":        tensor([[...]]),        # bool, 真行为 True
  "lengths":     tensor([...]),          # 每样本实际长度
  "labels":      tensor([...]),          # float32
  "session_ids": ["...", "..."],         # 字符串列表,不进模型
}
```

这些 key 正好是 `models.py` 的 `forward` 要用到的输入。

---

## 六、需要注意的点 / 发现清单

1. **动态宽度，不是固定 50**：batch 宽 = 本 batch 最长（≤50）。同一模型在不同 batch 看到的序列长度不同，靠 mask 处理 padding。
2. **`max_seq_len=50` 是硬约束**：position embedding 表只有 50 行（`models.py` 的 `nn.Embedding(max_len, hidden)`），`positions` 取值 `0..width-1`，不截断会越界报错。
3. **50 会不会砍掉信息？会**：报告 5.12 显示 session 总长度「最少 2、中位数 10、**最多 159**」。负样本保留完整 session（最长 159），正样本首单前缀也可能超 50，都会被 `[-50:]` 截成尾部 50 步。50 是 `analysis.py` 用 p90/p95 分布定的"覆盖大多数、只砍长尾"的折中。
4. **尾部截断是设计选择**：超长序列丢**头部**（最早行为）、留**尾部**（最靠近下单），理由是临近预测时点的行为信号最强。若想验证头部行为是否有价值，这里可改。
5. **正样本只留首单前缀**（上游预处理决定）：多单 session 的首单之后（含第 2、3 次下单）全部丢弃，模型只学"首次下单"二分类。这是任务定义范围，非 bug。
6. **7 列无条件加载**：即使 GRU 基线只用 `event_ids`，page/poi 等 6 列也载进内存了。数据量大时是内存浪费，换取 dataset 层不感知 feature_set 的简洁。
7. **`str()`/`float()` 是 numpy→Python 标量转换**：`session_id` 的 `str()` 是刚需（否则 JSON 导出抛错），`label` 的 `float()` 是防御性。
8. **两处 `# pyright: ignore` 暴露类型标注太宽**：`collate` 参数是 `Sequence[Dict[str, object]]`，`sample[name]` 被当成 `object`，于是 `len(...)`、`[...]` 切片被类型检查器误报。根因是 `object` 丢掉了"值其实是 ndarray"的信息；更干净的修法是把类型收紧（如 `Dict[str, np.ndarray]` 或 TypedDict），而不是加 `ignore` 静音。
9. **工厂函数是必须的**：DataLoader 的 `collate_fn` 只收一个参数，`make_collate_fn` 用闭包把 `max_seq_len` 封进去，让它可配置。
10. **`sample_limit` 与 `row_limit` 不同**：前者是训练侧按"样本条数"截断（快跑训练循环），后者是预处理侧按"事件行数"截断（快跑预处理）。
