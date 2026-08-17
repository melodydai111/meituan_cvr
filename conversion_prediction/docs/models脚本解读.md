# Models 脚本（models.py）解读与发现

> 对象：`conversion_prediction/src/conversion_prediction/models.py`
> 生成日期：2026-08-17
> 配套阅读：`dataset脚本解读.md`（上游：batch 的 key 从哪来）、`预处理脚本解读.md`、`../configs/base.yaml`（`hidden_dim`、`num_heads` 等超参）、`../README.md`（实验设计）

---

## 一、一句话概述

`models.py` 定义了两套「行为序列 → 是否转化」的二分类模型——**GRU 基线**和 **Transformer 主模型**——以及它们**共用**的特征表示层（`BehaviorEmbedding`）和序列汇总层（`AttentionPooling`）。整个文件围绕两个设计目标组织：**模型对比公平**（两模型共用同一套 Embedding 和消融开关）和**序列 → 标量的降维**（Transformer 出序列、池化出单向量、分类头出概率）。

**上游**：`dataset.py` 产出 batch（`event_ids`、`positions`、`mask`、`lengths`、`labels` 等整数张量）
**下游**：`training.py` 用 `build_model` 建模型、跑 `forward` 拿 `logits`，`analysis.py` 用返回的 `pooling_weights` 做解释

---

## 二、职责定位（承上启下）

```
dataset.py                  models.py                        training.py / analysis.py
─────────────              ─────────────────────────        ─────────────────────
batch(整数ID+mask)  ────►   BehaviorEmbedding(查表相加)      logits(二分类损失)
                          → TransformerEncoder(序列编码)     pooling_weights(解释)
                          → AttentionPooling(序列→单向量)
                          → head(单向量→logit)
```

数据在模型内部的形状变化（以 `max_seq_len=50`、`hidden_dim=64` 为例）：

```
[B, 50] 整数ID  →  [B, 50, 64]  →  [B, 50, 64]  →  [B, 64]  →  [B]  logits
  Embedding 查表       Transformer 编码        AttentionPooling   head 输出
```

---

## 三、核心组件详解

### 3.1 `FEATURE_SETS` —— 消融实验开关

```python
FEATURE_SETS = {
    "event":              (False, False, False),
    "context":            (True,  False, False),
    "context_stage":      (True,  True,  False),
    "context_stage_time": (True,  True,  True),
}
```

三个布尔值依次表示是否使用：**上下文特征（page/category/poi）、消费阶段（stage）、时间间隔（time_gap）**。4 个 key 从少到多排列，用于**消融实验**——每关掉一个特征，就少建一组 Embedding 参数，从而对比每个特征对预测的贡献。

| 名字 | context | stage | time | 含义 |
|------|:--:|:--:|:--:|------|
| `event` | ✗ | ✗ | ✗ | 只用事件本身（基线） |
| `context` | ✓ | ✗ | ✗ | + 页面/类目/POI |
| `context_stage` | ✓ | ✓ | ✗ | + 消费阶段 |
| `context_stage_time` | ✓ | ✓ | ✓ | + 时间间隔（全量） |

**关键设计**：没启用的特征**不创建参数**（见 3.2 里的 `if self.use_context:`），保证「feature_set 不同 → 参数量不同 → 消融对比公平」，而不是建了参数却不用。

### 3.2 `BehaviorEmbedding` —— 特征表示层

把一个行为位置上的多个离散字段各自查 Embedding 表，然后**相加**到同一个隐空间。两个模型共用这一个类（对比公平的关键）。

#### (a) `hidden` 是什么

`hidden = config["model"]["hidden_dim"]` = **64**，即「隐空间维度」——每个行为被表示成的向量长度。所有 Embedding 表的输出维度都设成 64，**因为它们要相加，而向量相加的前提是维度相同**。64 是全局约定，后续 Transformer 的 `d_model` 也是 64，全链路维度一致。

#### (b) `padding_idx=0`

```python
self.event      = nn.Embedding(buckets["event"] + 1, hidden, padding_idx=0)  # 4097 行
self.event_type = nn.Embedding(4, hidden, padding_idx=0)                      # 4 行
```

- `padding_idx=0` 表示 ID=0 是 padding 占位，它的向量被**固定为 0 且不参与梯度**。这样 padding 位置即使参与相加，加进去的也是 0，不污染真实行为。
- `buckets["event"] + 1`：有效 ID 从 1 开始（0 留给 padding），所以表要 +1 行。

#### (c) `position` —— 位置编码（可学习的绝对位置 Embedding）

```python
self.position = nn.Embedding(max_len, hidden)   # nn.Embedding(50, 64)
...
output = self.event(...) + self.event_type(...) + self.position(batch["positions"])
```

**这是本代码的位置编码方案**，几个要点：

1. **是可学习的绝对位置 Embedding，不是原论文的正弦编码**。`positions` 是 `[0, 1, 2, ..., width-1]` 的序号（由 `dataset.py` 的 collate 生成，截断后重新从 0 编号），查表得到每位置的 64 维向量，加到 token 表示上。
2. **Transformer 本身对顺序不敏感**（自注意力是集合运算），所以必须显式注入位置信息，这就是它的作用。
3. **为什么选可学习 Embedding 而非正弦编码**：`max_seq_len=50` 固定且短，一张 50×64=3200 参数的表就够，可学习向量比固定正弦更灵活，能自己学出「哪个位置更重要」。
4. **`position` 表没有设 `padding_idx=0`**（对比 event/event_type 都有）：因为位置 0 是合法的「第一个行为」，不能被当 padding 屏蔽。后果是 padding 位置也会查到一个非零位置向量加进去——但无影响，因为下游 mask 会把 padding 彻底屏蔽。
5. **`max_len=50` 是硬约束**：位置序号必须 < 50，否则越界。这依赖 `dataset.py` 的截断 `[-max_seq_len:]` 保证 `width ≤ 50`。

**关于「同位置用同一个向量是否区分不了内容」**：位置 embedding 是**全局共享表**，同位置同向量是设计使然——它只负责「你是第几个行为」。内容的区分靠 `event` embedding，上下文的区分靠自注意力。三者相加 + 编码器处理后，不同序列里同一位置的行为最终表示会不同。

**位置编码谱系补充**（讨论延展）：绝对位置 Embedding 的短板是分不清「相对距离」、不能外推；工业界对 NLP 常上 RoPE/相对位置偏置/ALiBi。但**用户行为序列是特殊情况**——更看重「时间间隔」而非「先后顺序」，所以本代码额外加了 `time_gap` 特征（见下），比纠结位置编码更贴合转化预测。

#### (d) 相加而非拼接

每个字段的 embedding 直接求和（维度保持 64 不变），这是经典做法，参数量小。以全量 `context_stage_time` 为例，一个位置的 64 维向量 = `event + event_type + position + page + category + poi + stage + time_gap`，共 8 项相加。

#### (e) `self.norm = nn.LayerNorm(hidden)` 的作用

```python
return self.dropout(self.norm(output))
```

`LayerNorm` 对每个行为位置**独立地**在 64 维特征方向做归一化（均值 0、方差 1，再乘可学习 γ 加 β）。**为什么需要**：`output` 是 8 个各自独立训练、量纲可能差异很大的 Embedding 直接相加，幅度和内部分布不稳定；LayerNorm 把每个位置的向量重新拉到统一尺度，让训练更稳、收敛更快。顺序是**先 norm 再 dropout**（标准顺序）。

#### (f) 「先定义再调用」是 PyTorch 的必须范式

```python
self.event = nn.Embedding(...)      # ① __init__ 里定义（登记）
output = self.event(batch[...])     # ② forward 里调用
```

**触发登记的真正开关是 `self.` 前缀，不是「写在 __init__ 里」**。`nn.Module.__init__` 会拦截 `self.xxx = 子模块` 这种赋值，把子模块记进内部清单，从而带来三件事：

| 能力 | 谁在用 | 没登记会怎样 |
|------|--------|-------------|
| 梯度追踪 | 优化器 `model.parameters()` | 遍历不到，参数永不更新 |
| 导出/恢复权重 | `state_dict()` / `load_state_dict()` | 存下来是空 `OrderedDict()` |
| 搬设备 | `model.to(device)` | 无参可搬 |

若在 `forward` 里写 `fc = nn.Linear(...)`（无 `self.` 前缀），它是普通局部变量，`loss.backward()` **不会报错**但梯度无处可存——模型「静默失效」（看起来在训练，实际一个参数没更新）。

### 3.3 `AttentionPooling` —— 序列 → 单向量

```python
scores  = self.score(sequence).squeeze(-1)              # 每位置打 1 个分数 [B, S]
scores  = scores.masked_fill(~mask, -1e9)               # padding → 极小值
weights = torch.softmax(scores, dim=-1)                 # 分数 → 权重(和=1)
pooled  = torch.sum(sequence * weights.unsqueeze(-1), dim=1)  # 加权求和 [B, 64]
```

#### (a) 它做什么

把编码器输出的 `[B, S, 64]` 序列**加权求和成 1 个 `[B, 64]` 摘要向量**：`self.score`（`Linear(64→64)→Tanh→Linear(64→1)`）给每个位置打一个「重要性分数」，softmax 成权重，再对所有位置向量加权求和。`dim=1`（序列维）被 sum 掉。

#### (b) 它和 encoder 里的自注意力**不是一回事**

两者都叫「注意力」，但方向相反：

| | 自注意力（encoder 内部） | AttentionPooling |
|---|---|---|
| 注意力对象 | 位置 vs 位置（S×S 矩阵） | 位置 vs 全局（S 个标量） |
| 输出 | `[B, S, 64]`——**还是 S 个向量** | `[B, 64]`——**压成 1 个** |
| 目的 | 让位置互相交流、丰富每个位置 | 把位置汇总成 1 个摘要 |

#### (c) 为什么需要这步（标准 Transformer 里没有）

标准 Transformer 是 **seq2seq（编码器-解码器）**，输入输出都是序列，**从头到尾不需要把序列压成单向量**，所以没有 pooling。而本任务是**序列分类**——输出是「是否转化」这 1 个标量，编码器却输出 S 个向量，中间**必须**有一步把 S 个向量汇总成 1 个。**这步 pooling 不是 Transformer 的一部分，而是「给序列编码器接上分类任务」时额外加的一层适配器**。

所有「用序列编码器做分类」的模型都必须有汇总这一步，只是实现不同：

| 汇总方式 | 做法 | 谁在用 |
|---------|------|--------|
| [CLS] token | 序列开头插特殊 token，取它的最终向量 | BERT |
| 取最后位置 | 用序列末尾 token 的 hidden state | GPT、RNN |
| 平均池化 | 所有位置向量取平均 | 简单模型 |
| **注意力池化** | 学权重加权求和 | **本代码** |

#### (d) 注意力池化 vs BERT [CLS]

两者本质上是「同一个操作放在不同阶段」：

- **BERT [CLS]**：在序列开头插一个可学习 token，它**每一层都 attend 到所有位置**，逐层「吸收」全局信息，融合发生在**编码器内部**（渐进式）。最终 `[CLS]` 向量 = 最后一层里对它做的一次注意力加权求和 + FFN/残差。
- **注意力池化**：编码器结束后，用一个**独立训练的打分网络**对所有最终表示做一次性加权，融合发生在**编码器之后**（一次性）。

| 维度 | BERT [CLS] | 注意力池化 |
|------|-----------|-----------|
| 汇总方式 | 多插一个 token，取它的向量 | 学权重，加权求和 |
| 融合时机 | 编码器内部（每层吸收） | 编码器之后（一次性） |
| 额外参数 | 1 个 embedding 向量（~64） | 一个小 MLP（~4k） |
| 可解释性 | 弱（看不出每 token 贡献） | **强**（`pooling_weights` 直接给出） |
| 工程改动 | 需改预处理（插 token） | 纯加模块，不动数据 |

本代码选注意力池化，主要是**实现简单（不动数据）+ 自带可解释性**；代价是融合不如 [CLS] 层层累积，但短序列 + 业务特征已加足的场景下可忽略。

#### (e) 可解释性价值（`pooling_weights`）

`pooling_weights` 是 `[B, S]`，每条 session 里每个行为的**相对重要性**。把它和 `event_type / stage / category / poi / time_gap` 对齐，能回答「什么行为、什么阶段、什么类目在驱动转化」——这正是「业务 cross 分析」想要的、能反哺运营决策的东西，且**不用额外做 SHAP/LIME/attention 可视化**。

**诚实的边界**：(1) 注意力权重 ≠ 因果（"attention is not explanation"），适合探索规律、提假设，不适合当因果证据；(2) 权重是相对权重（每条 session 内和=1），表达「本 session 内哪个更重要」，不能跨 session 比绝对值。

### 3.4 `GRUClassifier` —— 基线模型

```python
self.embedding = BehaviorEmbedding(config, feature_set)   # 与 Transformer 共用同一套
self.encoder = nn.GRU(hidden, hidden, num_layers=layers, batch_first=True, ...)
self.head = nn.Sequential(LayerNorm(hidden), Dropout, Linear(hidden, 1))
```

- **共用同一套 `BehaviorEmbedding`**（注释明示「保证模型对比公平」）：GRU 和 Transformer 的唯一区别是序列编码器，特征表示层完全一致，这样对比才干净。
- `pack_padded_sequence(..., batch["lengths"].cpu(), enforce_sorted=False)`：pack 后 GRU 在 padding 位置**不继续更新隐藏状态**。
- 取最后一层隐藏状态 `hidden[-1]` → 单层 `Linear(hidden→1)` 出 logit（GRU 用「取最后一步」汇总，不需要 AttentionPooling）。

### 3.5 `TransformerClassifier` —— 主模型

```python
self.embedding = BehaviorEmbedding(config, feature_set)
encode_layer  = nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=256,
                                           dropout=0.10, activation="gelu",
                                           batch_first=True, norm_first=True)
self.encoder  = nn.TransformerEncoder(encode_layer, num_layers=2)
self.pooling  = AttentionPooling(hidden)
self.head     = nn.Sequential(LayerNorm(64), Linear(64→32), GELU, Dropout, Linear(32→1))
```

#### (a) 三个构造参数

| 参数 | 值 | 含义 |
|------|----|------|
| `dim_feedforward`(ff_dim) | 256 | FFN 中间扩张宽度（64→256→64），单位置非线性变换容量，常见 4×d_model |
| `batch_first` | True | 张量维度顺序约定：`[batch, seq, dim]` 而非默认 `[seq, batch, dim]`，省去手写 transpose |
| `norm_first` | True | **Pre-LN**：先归一化再算子层，训练更稳、不需要 warmup（对比默认 Post-LN） |

#### (b) QKV 在哪里（不在本文件里，在 PyTorch 内部）

`nn.TransformerEncoderLayer.__init__` 内部有一行 `self.self_attn = MultiheadAttention(d_model, nhead, ...)`，QKV 投影就藏在 `MultiheadAttention` 的 **`in_proj_weight`**（`[3·d_model, d_model]` = `[192, 64]`，一次投影出 Q/K/V）。**真正调用 QKV 的地方**在 `_sa_block`：

```python
def _sa_block(self, x, ...):
    x = self.self_attn(x, x, x, ...)   # ← 三个参数：query, key, value 都是同一个 x
    return self.dropout1(x)
```

**`self_attn(x, x, x)` 传三次同一个 x** 就是「自」注意力的含义：Q=K=V 都来自同一序列（自己 attend 自己）。

#### (c) 残差连接在哪里

在 `TransformerEncoderLayer.forward` 底部（`norm_first=True` 分支）：

```python
x = src
x = x + self._sa_block(self.norm1(x), ...)   # ① 残差：x + 注意力结果
x = x + self._ff_block(self.norm2(x))        # ② 残差：x + FFN 结果
```

**`x = x + ...` 里的 `x +` 就是残差连接**：把子层输入短路直通、加到输出上。每个 Transformer 层有**两个**残差（注意力一个、FFN 一个），残差路径直通是深层梯度能顺畅回传的原因。

#### (d) 多头注意力的实现

`d_model=64, nhead=4` → 每头 `head_dim = 64/4 = 16`。核心是把 64 维切成 4 份，各自独立做注意力：

```
Q/K/V [B, 4, S, 16]  (reshape 成 batch, head, seq, head_dim)
scores = Q·Kᵀ/√16 → [B, 4, S, S]   # 每个头一张 S×S 分数矩阵
weights = softmax(scores)
output  = weights·V → [B, 4, S, 16]
拼接 4 头 → [B, S, 64] → out_proj
```

多头让模型**并行捕捉多种依赖关系**（头1学事件-类目对应、头2学时间远近、头3学阶段转移…），类似卷积里多个卷积核。

#### (e) encoder 的 forward 具体过程

`nn.TransformerEncoder.forward` 核心是**循环过 2 个 layer**：

```python
for mod in self.layers:                 # 2 个 TransformerEncoderLayer
    output = mod(output, src_key_padding_mask=...)
```

每层内部（`norm_first=True`）：

```python
x = x + self._sa_block(self.norm1(x), ...)   # 先 norm → 多头自注意力 → 残差
x = x + self._ff_block(self.norm2(x))        # 先 norm → FFN → 残差
```

其中 `_ff_block` = `linear2(dropout(activation(linear1(x))))`（64→256→GELU→dropout→64）。**层与层之间形状不变**（都是 `[B, S, 64]`），所以可任意堆叠；第 1 层让相邻位置交流，第 2 层能看到更远的间接依赖。

#### (f) 为什么没有 decoder

**因为这是分类不是 seq2seq**。decoder 的作用是「自回归地逐 token 生成输出序列」（靠 masked attention + 交叉 attention），只有需要「生成一个序列」时才用得上。转化预测输出的是「是否转化」这 1 个标签，没有序列要生成，所以只留 encoder。

Transformer 家族三类结构：

| 类型 | 结构 | 典型任务 | 例子 |
|------|------|---------|------|
| Encoder-only | 编码器 + 汇总层 | 序列分类 | BERT、**本代码** |
| Decoder-only | 解码器 | 生成 | GPT |
| Encoder-Decoder | 两者 | seq2seq | 原始 Transformer、T5 |

本代码就是 **BERT 式的 encoder-only 分类架构**：编码器提取序列特征 → AttentionPooling 汇总 → 分类头输出标签，只是用注意力池化替代 BERT 的 [CLS]。

#### (g) `self.head` —— 分类头（不是 FFN）

```python
nn.Sequential(
    LayerNorm(64),          # 池化向量归一化
    Linear(64 → 32),        # 降维到一半
    GELU(),                 # 非线性
    Dropout(0.10),          # 正则化
    Linear(32 → 1),         # 输出 1 个 logit
)
```

- 中间「64→32→1」而非「64→1」一步到位：单层 Linear 只能线性变换，加 32 维隐层 + GELU 让分类头能学**非线性**决策边界。
- 输出**没有 sigmoid**：1 个 logit 交给 `BCEWithLogitsLoss`，它内部自己套 sigmoid，数值更稳。
- **别和 encoder 的 FFN 混淆**：encoder 的 FFN 是逐位置加工 token 表示，`self.head` 是把全局摘要映射成预测，是「分类头/输出头」。

#### (h) `forward` 里的 mask 取反

```python
encoded = self.encoder(embedded, src_key_padding_mask=~batch["mask"])
```

`batch["mask"]` 里 `True`=有效行为、`False`=padding；而 PyTorch 的 `src_key_padding_mask` 语义**相反**——`True`=要忽略。所以取反 `~batch["mask"]`，把「有效位」翻成「忽略位」。这个 mask 让 padding 位置在所有头里被屏蔽（既不被 attend 也 attend 不到别人）。

### 3.6 `build_model` —— 工厂函数

```python
model_class = {"gru": GRUClassifier, "transformer": TransformerClassifier}[model_name]
return model_class(config, feature_set)
```

按实验名创建 GRU 或 Transformer，训练侧用 `model_name` 切换基线/主模型。

---

## 四、完整代码（原样保留）

```python
"""行为特征表示、GRU 基线和 Transformer 转化预测模型。"""

from __future__ import annotations

from typing import Dict

import torch
from torch import nn


# 三个布尔值依次表示是否使用：上下文特征、消费阶段、时间间隔。
FEATURE_SETS = {
    "event": (False, False, False),
    "context": (True, False, False),
    "context_stage": (True, True, False),
    "context_stage_time": (True, True, True),
}


class BehaviorEmbedding(nn.Module):
    """把一个行为位置上的多个离散字段映射到同一隐空间后相加。"""

    def __init__(self, config: Dict, feature_set: str):
        super().__init__()
        self.use_context, self.use_stage, self.use_time = FEATURE_SETS[feature_set]
        hidden = config["model"]["hidden_dim"]
        max_len = config["data"]["max_seq_len"]
        buckets = config["data"]["hash_buckets"]

        # 基线特征，只包含event和event_type，以及transformer要求的位置编码position
        # 0 专门留给 padding；预处理生成的有效 ID 从 1 开始。
        self.event = nn.Embedding(buckets["event"] + 1, hidden, padding_idx=0)
        self.event_type = nn.Embedding(4, hidden, padding_idx=0)
        self.position = nn.Embedding(max_len, hidden)

        # 消融实验未使用的特征不创建参数，保证模型参数量与实验定义一致。
        if self.use_context:
            self.page = nn.Embedding(buckets["page"] + 1, hidden, padding_idx=0)
            self.category = nn.Embedding(buckets["category"] + 1, hidden, padding_idx=0)
            self.poi = nn.Embedding(buckets["poi"] + 1, hidden, padding_idx=0)
        if self.use_stage:
            # ORDER 已从输入删除，因此只有 padding、探索、发现和决策四种 ID。
            self.stage = nn.Embedding(4, hidden, padding_idx=0)
        if self.use_time:
            self.time_gap = nn.Embedding(7, hidden, padding_idx=0)

        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(config["model"]["dropout"])

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # 基础表示始终包含具体事件、事件大类和发生位置。
        output = self.event(batch["event_ids"]) + self.event_type(batch["event_type_ids"]) + self.position(batch["positions"])

        # 根据 feature_set 逐步加入业务上下文、消费阶段和决策节奏。
        if self.use_context:
            output = output + self.page(batch["page_ids"])
            output = output + self.category(batch["category_ids"])
            output = output + self.poi(batch["poi_ids"])
        if self.use_stage:
            output = output + self.stage(batch["stage_ids"])
        if self.use_time:
            output = output + self.time_gap(batch["time_gap_ids"])
        return self.dropout(self.norm(output))


class AttentionPooling(nn.Module):
    """学习每个行为对 session 表示的贡献，并得到加权序列向量。"""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False),
        )

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor):
        # padding 位置设为极小值，使 softmax 后的权重接近 0。
        scores = self.score(sequence).squeeze(-1).masked_fill(~mask, -1e9)
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.sum(sequence * weights.unsqueeze(-1), dim=1) # [B, 64]
        return pooled, weights


class GRUClassifier(nn.Module):
    """用 GRU 最后一层隐藏状态表示整个用户行为序列。"""

    def __init__(self, config: Dict, feature_set: str):
        super().__init__()
        hidden = config["model"]["hidden_dim"]
        layers = config["model"]["num_layers"]
        dropout = config["model"]["dropout"]

        # 与 Transformer 共用同一套行为 Embedding，保证模型对比公平。
        self.embedding = BehaviorEmbedding(config, feature_set)
        self.encoder = nn.GRU(
            hidden,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, batch: Dict[str, torch.Tensor]):
        embedded = self.embedding(batch)

        # pack 后 GRU 不会在 padding 位置继续更新隐藏状态。
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded,
            batch["lengths"].cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.encoder(packed)
        logits = self.head(hidden[-1]).squeeze(-1)
        return {"logits": logits}


class TransformerClassifier(nn.Module):
    """编码行为序列，并通过注意力池化预测 session 转化概率。"""

    def __init__(self, config: Dict, feature_set: str):
        super().__init__()
        model_cfg = config["model"]
        hidden = model_cfg["hidden_dim"]
        dropout = model_cfg["dropout"]

        self.embedding = BehaviorEmbedding(config, feature_set)
        encode_layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=model_cfg["num_heads"],
            dim_feedforward=model_cfg["ff_dim"],
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN
        )
        self.encoder = nn.TransformerEncoder(encode_layer, model_cfg["num_layers"])
        self.pooling = AttentionPooling(hidden)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, batch: Dict[str, torch.Tensor]):
        embedded = self.embedding(batch) # 维度[B, 50, 64]

        # PyTorch 的 key padding mask 中 True 表示忽略，因此对有效位 mask 取反。
        encoded = self.encoder(embedded, src_key_padding_mask=~batch["mask"]) # mask后的
        pooled, pooling_weights = self.pooling(encoded, batch["mask"]) # [B, 64]
        logits = self.head(pooled).squeeze(-1) # [B]
        return {"logits": logits, "pooling_weights": pooling_weights}


def build_model(config: Dict, model_name: str, feature_set: str) -> nn.Module:
    """根据实验名称创建 GRU 或 Transformer。"""
    model_class = {"gru": GRUClassifier, "transformer": TransformerClassifier}[model_name]
    return model_class(config, feature_set)
```

---

## 五、一次 forward 的完整走查（batch=2, seq=3）

用 `B=2, L=3, hidden=64, nhead=4(→head_dim=16), ff_dim=256, num_layers=2` 推演 Transformer 主模型的完整数据流。

### 5.1 输入 → Embedding

```
event_ids / event_type_ids / positions  各自 [2, 3]
  → 查表得 event [2,3,64]、event_type [2,3,64]、position [2,3,64]
  → 相加 → [2,3,64]
  → LayerNorm + Dropout → embedded [2,3,64]
```

### 5.2 Encoder（2 层，形状全程不变）

第 1 层内部（`norm_first=True`）：

**① 自注意力子块 `_sa_block`**：

```
norm1(x)                    [2,3,64]
self_attn(x, x, x):         Q=K=V=x
  in_proj_weight [192,64] → Q/K/V 各 [2,3,64]
  reshape 成 4 头           → Q/K/V 各 [2,4,3,16]
  scores = Q·Kᵀ/√16        → [2,4,3,3]   每头一张 3×3 分数矩阵
  softmax                   → 权重(每行和=1)
  weights·V                 → [2,4,3,16]
  拼接 4 头 + out_proj      → [2,3,64]
  dropout1
x = x + 注意力输出           → [2,3,64]   (残差)
```

**② FFN 子块 `_ff_block`**：

```
norm2(x)       [2,3,64]
linear1        [2,3,64] → [2,3,256]   (逐位置升维)
GELU + dropout [2,3,256]
linear2        [2,3,256] → [2,3,64]
x = x + FFN输出 → [2,3,64]            (残差)
```

第 2 层重复同样流程（**另一套独立参数**），输出 `encoded [2,3,64]`。

**关键观察**：encoder 全程形状不变（输入 `[2,3,64]`、输出 `[2,3,64]`），只负责让 3 个位置互相交流、丰富每个位置，不改变 token 数和维度。

### 5.3 AttentionPooling

```
sequence [2,3,64]
score(sequence)  → Linear(64→64)→Tanh→Linear(64→1) → scores [2,3]   每位置 1 个标量
masked_fill(~mask, -1e9)                                          padding→极小值
softmax(dim=-1)   → weights [2,3]                                  每条 session 内和=1
pooled = sum(sequence * weights, dim=1) → [2,64]                   3 个向量 → 1 个摘要
```

示意（session0 三位置权重 0.15/0.75/0.10）：

```
pooled[session0] = 0.15·v0 + 0.75·v1 + 0.10·v2    (v_i 是各位置 64 维向量)
```

### 5.4 head

```
pooled [2,64] → LayerNorm → Linear(64→32) → GELU → Dropout → Linear(32→1) → [2,1] → squeeze → logits [2]
```

最终返回 `{"logits": [2], "pooling_weights": [2,3]}`。

---

## 六、需要注意的点 / 发现清单

1. **两模型共用同一套 `BehaviorEmbedding`**：这是「模型对比公平」的核心设计。GRU 和 Transformer 的唯一区别是序列编码器，特征表示层完全一致，对比才干净。
2. **消融开关直接决定参数量**：未启用的特征不建参数（`if self.use_context:` 等），feature_set 不同 → 参数量不同 → 消融对比公平。
3. **`position` 用的是可学习绝对位置 Embedding，不是正弦编码**：因为 `max_seq_len=50` 固定且短，查表更灵活、成本可忽略。且 `position` 表**没有** `padding_idx=0`（位置 0 是合法位置）。
4. **`max_len=50` 是硬约束**：位置序号必须 < 50，依赖 `dataset.py` 的截断 `[-max_seq_len:]` 保证不越界。
5. **Transformer 出序列、池化出单向量**：`AttentionPooling` 是「序列分类」必需的汇总层，标准 seq2seq Transformer 里没有它。它等价于 BERT 的 [CLS]，但实现更简单 + 自带可解释性。
6. **自注意力和池化注意力方向相反**：前者「位置 vs 位置」产出 S 个向量，后者「位置 vs 全局」压成 1 个，别混。
7. **QKV、残差、层循环都不在本文件里**：它们封装在 PyTorch 的 `nn.TransformerEncoderLayer` / `nn.TransformerEncoder` 内部（`in_proj_weight`、`x = x + ...`、`for mod in self.layers`），本文件看不到，属正常。
8. **没有 decoder 是任务决定的**：分类任务只输出单个标签，不需要逐 token 生成序列，所以是 encoder-only 架构（BERT 式）。
9. **`src_key_padding_mask=~batch["mask"]` 的取反**：`batch["mask"]` 的 `True`=有效，PyTorch 的 key padding mask 语义相反（`True`=忽略），所以必须取反。
10. **`self.head` 输出无 sigmoid**：1 个 logit 交给 `BCEWithLogitsLoss`（内部自带 sigmoid），数值更稳。
11. **GRU 用「取最后一步」汇总，Transformer 用「注意力池化」汇总**：两者汇总方式不同，但都靠 mask 忽略 padding。
12. **`pooling_weights` 可解释但非因果**：能看出哪个行为最关键，适合探索规律、提假设，不适合当因果证据下结论。
13. **`time_gap` 补足了绝对位置表达不了的「节奏」**：行为序列里时间间隔往往比先后顺序更本质（同样「浏览→下单」，间隔 30 秒是冲动、5 天是犹豫），这是比 fancy 位置编码更贴合转化预测的特征设计。
