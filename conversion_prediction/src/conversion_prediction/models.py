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
        output = self.event(batch["event_ids"]) + self.event_type(batch["event_type_ids"])
        output = output + self.position(batch["positions"])

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
        pooled = torch.sum(sequence * weights.unsqueeze(-1), dim=1)
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
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=model_cfg["num_heads"],
            dim_feedforward=model_cfg["ff_dim"],
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, model_cfg["num_layers"])
        self.pooling = AttentionPooling(hidden)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, batch: Dict[str, torch.Tensor]):
        embedded = self.embedding(batch)

        # PyTorch 的 key padding mask 中 True 表示忽略，因此对有效位 mask 取反。
        encoded = self.encoder(embedded, src_key_padding_mask=~batch["mask"])
        pooled, pooling_weights = self.pooling(encoded, batch["mask"])
        logits = self.head(pooled).squeeze(-1)
        return {"logits": logits, "pooling_weights": pooling_weights}


def build_model(config: Dict, model_name: str, feature_set: str) -> nn.Module:
    """根据实验名称创建 GRU 或 Transformer。"""
    model_class = {"gru": GRUClassifier, "transformer": TransformerClassifier}[model_name]
    return model_class(config, feature_set)
