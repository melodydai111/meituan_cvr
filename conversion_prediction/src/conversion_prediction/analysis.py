"""原始数据 EDA、订单前序行为分析和 Transformer 行为权重解释。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import duckdb
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .dataset import SessionDataset, make_collate_fn
from .models import build_model
from .training import move_batch, resolve_device


def _literal(value: str | Path) -> str:
    """转义供 DuckDB SQL 使用的本地路径。"""
    return "'" + str(value).replace("'", "''") + "'"


def run_eda(config: Dict) -> Path:
    """生成数据概览、事件分布、阶段转化率和典型行为路径。"""
    raw_path = Path(config["paths"]["raw_data"])
    samples_path = Path(config["paths"]["processed_dir"]) / "session_samples.parquet"
    output_dir = Path(config["paths"]["artifacts_dir"]) / "eda"
    output_dir.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect()

    # 原始日志概览回答数据有多大、覆盖多久以及订单事件是否稀少。
    raw_row = connection.execute(
        f"""
        SELECT
          COUNT(*) AS events,
          COUNT(DISTINCT user_id) AS users,
          COUNT(DISTINCT session_id) AS sessions,
          SUM(CASE WHEN event_type = 'ORDER' THEN 1 ELSE 0 END) AS order_events,
          MIN(to_timestamp(event_timestamp / 1000.0)) AS start_time,
          MAX(to_timestamp(event_timestamp / 1000.0)) AS end_time
        FROM read_parquet({_literal(raw_path)})
        """
    ).fetchone()
    raw_summary = dict(
        zip(
            ["events", "users", "sessions", "order_events", "start_time", "end_time"],
            raw_row,
        )
    )
    raw_summary["start_time"] = str(raw_summary["start_time"])
    raw_summary["end_time"] = str(raw_summary["end_time"])

    # 事件分布用于观察浏览、点击和订单之间是否形成明显漏斗。
    event_distribution = connection.execute(
        f"""
        SELECT event_type, event_name, COUNT(*) AS events
        FROM read_parquet({_literal(raw_path)})
        GROUP BY event_type, event_name
        ORDER BY events DESC
        """
    ).df()
    event_distribution.to_csv(output_dir / "event_distribution.csv", index=False)

    # 品类分布保留出现次数最多的组合，避免长尾类别生成过大的分析文件。
    category_distribution = connection.execute(
        f"""
        SELECT first_cate_name, second_cate_name, COUNT(*) AS events
        FROM read_parquet({_literal(raw_path)})
        WHERE first_cate_name IS NOT NULL
        GROUP BY first_cate_name, second_cate_name
        ORDER BY events DESC
        LIMIT 200
        """
    ).df()
    category_distribution.to_csv(output_dir / "category_distribution.csv", index=False)

    # 用窗口函数一次性取得 ORDER 前一个行为，替代旧脚本逐 session 的 Python 循环。
    order_context = connection.execute(
        f"""
        WITH ordered AS (
          SELECT
            event_type,
            LAG(event_type) OVER behavior_window AS previous_event_type,
            LAG(event_name) OVER behavior_window AS previous_event_name,
            LAG(page_name) OVER behavior_window AS previous_page_name,
            LAG(classification_name) OVER behavior_window AS previous_classification,
            LAG(first_cate_name) OVER behavior_window AS previous_category
          FROM read_parquet({_literal(raw_path)})
          WINDOW behavior_window AS (
            PARTITION BY session_id
            ORDER BY event_timestamp, event_type, event_name
          )
        )
        SELECT
          previous_event_type,
          previous_event_name,
          previous_page_name,
          previous_classification,
          previous_category,
          COUNT(*) AS orders
        FROM ordered
        WHERE event_type = 'ORDER'
        GROUP BY ALL
        ORDER BY orders DESC
        LIMIT 100
        """
    ).df()
    order_context.to_csv(output_dir / "order_previous_behavior.csv", index=False)

    # 导出一个同时包含 PV、MC、ORDER 的中等长度 session，便于人工检查业务路径。
    session_example = connection.execute(
        f"""
        WITH candidate AS (
          SELECT session_id
          FROM read_parquet({_literal(raw_path)})
          GROUP BY session_id
          HAVING COUNT(*) BETWEEN 8 AND 15
             AND bool_or(event_type = 'PV')
             AND bool_or(event_type = 'MC')
             AND bool_or(event_type = 'ORDER')
          LIMIT 1
        )
        SELECT
          event_timestamp,
          event_type,
          event_name,
          page_name,
          classification_name,
          poi_name,
          first_cate_name,
          second_cate_name
        FROM read_parquet({_literal(raw_path)})
        WHERE session_id = (SELECT session_id FROM candidate)
        ORDER BY event_timestamp, event_type, event_name
        """
    ).df()
    session_example.to_csv(output_dir / "session_example.csv", index=False)

    # 预处理样本概览用于决定 max_seq_len，并核对最终正负样本比例。
    sample_row = connection.execute(
        f"""
        SELECT
          COUNT(*) AS sessions,
          COUNT(DISTINCT user_id) AS users,
          SUM(label) AS positive_sessions,
          AVG(label) AS positive_rate,
          AVG(sequence_length) AS mean_length,
          median(sequence_length) AS median_length,
          quantile_cont(sequence_length, 0.90) AS p90_length,
          quantile_cont(sequence_length, 0.95) AS p95_length
        FROM read_parquet({_literal(samples_path)})
        """
    ).fetchone()
    sample_keys = [
        "sessions",
        "users",
        "positive_sessions",
        "positive_rate",
        "mean_length",
        "median_length",
        "p90_length",
        "p95_length",
    ]
    sample_summary = dict(zip(sample_keys, sample_row))

    split_distribution = connection.execute(
        f"""
        SELECT split, COUNT(*) AS sessions, AVG(label) AS positive_rate
        FROM read_parquet({_literal(samples_path)})
        GROUP BY split
        ORDER BY split
        """
    ).df()
    split_distribution.to_csv(output_dir / "split_distribution.csv", index=False)

    # 一个 session 只要出现过某阶段就计数一次，用于比较阶段后的转化倾向。
    stage_conversion = connection.execute(
        f"""
        WITH expanded AS (
          SELECT session_id, label, unnest(list_distinct(stage_ids)) AS stage_id
          FROM read_parquet({_literal(samples_path)})
        )
        SELECT stage_id, COUNT(*) AS sessions, AVG(label) AS conversion_rate
        FROM expanded
        GROUP BY stage_id
        ORDER BY stage_id
        """
    ).df()
    stage_conversion.to_csv(output_dir / "stage_conversion.csv", index=False)

    # 图中截到 P95，避免极长 session 把主体分布压缩在左侧。
    length_distribution = connection.execute(
        f"""
        SELECT sequence_length, COUNT(*) AS sessions
        FROM read_parquet({_literal(samples_path)})
        GROUP BY sequence_length
        ORDER BY sequence_length
        """
    ).df()
    length_distribution.to_csv(
        output_dir / "sequence_length_distribution.csv", index=False
    )
    clipped = length_distribution[
        length_distribution["sequence_length"] <= sample_summary["p95_length"]
    ]
    plt.figure(figsize=(8, 4.5))
    plt.bar(clipped["sequence_length"], clipped["sessions"], width=1.0)
    plt.xlabel("Sequence length")
    plt.ylabel("Sessions")
    plt.title("Session sequence length distribution (<= P95)")
    plt.tight_layout()
    plt.savefig(output_dir / "sequence_length_distribution.png", dpi=160)
    plt.close()

    summary = {"raw_data": raw_summary, "session_samples": sample_summary}
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    connection.close()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return output_dir


@torch.no_grad()
def explain_checkpoint(checkpoint_path: str | Path, max_sessions: int = 200) -> Path:
    """导出测试样本的预测概率和各行为的池化权重。"""
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["model_name"] != "transformer":
        raise ValueError("行为权重分析只适用于 Transformer checkpoint")

    config: Dict = checkpoint["config"]
    device = resolve_device(config["training"]["device"])
    model = build_model(config, checkpoint["model_name"], checkpoint["feature_set"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # 解释阶段只抽取部分测试 session，无需遍历整个测试集。
    samples_path = Path(config["paths"]["processed_dir"]) / "session_samples.parquet"
    dataset = SessionDataset(samples_path, "test", max_sessions)
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=make_collate_fn(config["data"]["max_seq_len"]),
    )

    # event_lookup 将哈希 ID 还原为最常见的业务事件名称。
    lookup_path = Path(config["paths"]["processed_dir"]) / "event_lookup.csv"
    lookup = pd.read_csv(lookup_path).set_index("event_id")["event_label"].to_dict()

    rows = []
    for batch in loader:
        batch = move_batch(batch, device)
        output = model(batch)
        probabilities = torch.sigmoid(output["logits"]).cpu().numpy()
        weights = output["pooling_weights"].cpu().numpy()
        event_ids = batch["event_ids"].cpu().numpy()
        lengths = batch["lengths"].cpu().numpy()
        labels = batch["labels"].cpu().numpy()

        # 每个行为输出一行，后续可按事件、位置或正负样本聚合。
        for index, session_id in enumerate(batch["session_ids"]):
            for position in range(int(lengths[index])):
                event_id = int(event_ids[index, position])
                rows.append(
                    {
                        "session_id": session_id,
                        "label": int(labels[index]),
                        "probability": float(probabilities[index]),
                        "position": position,
                        "event_id": event_id,
                        "event_label": lookup.get(event_id, "<UNKNOWN>"),
                        "pooling_weight": float(weights[index, position]),
                    }
                )

    output_path = checkpoint_path.parent / "pooling_attention.csv"
    pd.DataFrame(rows).to_csv(output_path, index=False)
    print(f"已保存 {output_path}")
    return output_path
