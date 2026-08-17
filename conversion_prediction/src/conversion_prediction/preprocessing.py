"""
把原始行为日志加工成无标签泄漏的 session 序列样本。
输入是原始事件日志的 Parquet，输出是聚合好的 session 样本 + 元数据
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import duckdb


# 消费阶段和时间桶的名称会写入 metadata，便于 EDA 和结果解释。
STAGE_NAMES = {
    0: "padding",
    1: "需求探索",
    2: "商户发现",
    3: "消费决策",
}

TIME_GAP_NAMES = {
    0: "padding",
    1: "序列起点",
    2: "0-1分钟",
    3: "1-5分钟",
    4: "5-30分钟",
    5: "30分钟-2小时",
    6: "2小时以上",
}


def _literal(value: str | Path) -> str:
    """将文件路径转义为可安全嵌入 DuckDB SQL 的字符串。"""
    return "'" + str(value).replace("'", "''") + "'"


def _hash_id(expression: str, buckets: int, missing: str | None = None) -> str:
    """生成离散特征哈希表达式，0 留给缺失值和 padding。"""
    hashed = f"CAST(1 + hash({expression}) % {buckets} AS INTEGER)"
    return f"CASE WHEN {missing} THEN 0 ELSE {hashed} END" if missing else hashed


def build_session_samples(config: Dict[str, Any]) -> Path:
    """把事件日志聚合为 session 序列样本。

    正样本只保留首个 ORDER 之前的行为，防止模型直接看到答案；负样本保留
    完整 session。所有 session 按起始时间切分，避免随机切分造成未来信息泄漏。
    """
    # 所有路径和参数都来自同一个 YAML，避免脚本内部写死本地目录。
    raw_path = Path(config["paths"]["raw_data"])
    output_dir = Path(config["paths"]["processed_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "session_samples.parquet"
    lookup_path = output_dir / "event_lookup.csv"
    # 预处理结果可以重复生成，开始前移除上一版派生文件。
    for generated_file in (output_path, lookup_path):
        generated_file.unlink(missing_ok=True)

    data_cfg = config["data"]
    buckets = data_cfg["hash_buckets"]
    row_limit = data_cfg.get("row_limit")
    limit_sql = f"LIMIT {int(row_limit)}" if row_limit else ""

    # DuckDB 直接扫描 Parquet 并聚合 list 列，不需要把 680 万行全部载入 pandas。
    connection = duckdb.connect()
    connection.execute(f"SET temp_directory={_literal(output_dir / 'duckdb_tmp')}")

    # event 使用“事件大类:事件名称”形成 token；高基数特征通过哈希桶控制参数量。
    event_label = "concat(coalesce(event_type, '<NULL>'), ':', coalesce(event_name, '<NONE>'))"
    event_id = _hash_id(event_label, buckets["event"])
    page_id = _hash_id("page_name", buckets["page"], "page_name IS NULL")
    category_id = _hash_id(
        "first_cate_name", buckets["category"], "first_cate_name IS NULL"
    )
    poi_id = _hash_id("CAST(poi_id AS VARCHAR)", buckets["poi"], "poi_id IS NULL")

    # 用中英文业务关键词识别消费阶段；未命中时 PV 为探索、MC 为商户发现。
    text = "lower(concat_ws(' ', event_name, page_name, classification_name, first_cate_name))"
    decision_pattern = (
        "coupon|pay|checkout|submit|deal|package|review|address|business.hour|detail|"
        "优惠|券|支付|提交订单|套餐|评价|地址|营业|详情"
    )
    discovery_pattern = "poi|merchant|shop|store|list|card|商家|门店|列表|卡片"
    stage_id = f"""
        CASE
          WHEN regexp_matches({text}, '{decision_pattern}') THEN 3
          WHEN event_type = 'MC' THEN 2
          WHEN regexp_matches({text}, '{discovery_pattern}') THEN 2
          ELSE 1
        END
    """

    train_ratio = float(data_cfg["split"]["train"])
    val_ratio = float(data_cfg["split"]["validation"])

    # SQL 的数据流为：标记标签 → 截取订单前缀 → 计算时间间隔 → 编码 → 聚合 session。
    query = f"""
    WITH raw AS (
      SELECT
        CAST(session_id AS VARCHAR) AS session_id,
        CAST(user_id AS VARCHAR) AS user_id,
        CAST(event_timestamp AS BIGINT) AS event_timestamp,
        event_type, event_name, page_name, classification_name, first_cate_name, poi_id
      FROM read_parquet({_literal(raw_path)})
      WHERE session_id IS NOT NULL AND event_timestamp IS NOT NULL
      {limit_sql}
    ), marked AS (
      -- 在删除 ORDER 前先生成 session 标签和首个订单时刻。
      SELECT *,
        MAX(CASE WHEN event_type = 'ORDER' THEN 1 ELSE 0 END)
          OVER (PARTITION BY session_id) AS label,
        MIN(CASE WHEN event_type = 'ORDER' THEN event_timestamp END)
          OVER (PARTITION BY session_id) AS first_order_timestamp
      FROM raw
    ), prefixes AS (
      -- 正样本只保留首个订单前的行为，防止模型直接看到答案。
      SELECT * FROM marked
      WHERE event_type <> 'ORDER'
        AND (label = 0 OR event_timestamp < first_order_timestamp)
    ), with_gaps AS (
      -- 相邻行为时间差刻画用户决策节奏。
      SELECT *,
        event_timestamp - LAG(event_timestamp) OVER (
          PARTITION BY session_id ORDER BY event_timestamp, event_type, event_name
        ) AS gap_ms
      FROM prefixes
    ), tokens AS (
      -- 将类别字段、消费阶段和时间间隔转换为 Embedding ID。
      SELECT *,
        {event_id} AS event_id,
        CASE event_type WHEN 'PV' THEN 1 WHEN 'MC' THEN 2 ELSE 3 END AS event_type_id,
        {page_id} AS page_id,
        {category_id} AS category_id,
        {poi_id} AS poi_id_hash,
        {stage_id} AS stage_id,
        CASE
          WHEN gap_ms IS NULL THEN 1
          WHEN gap_ms <= 60000 THEN 2
          WHEN gap_ms <= 300000 THEN 3
          WHEN gap_ms <= 1800000 THEN 4
          WHEN gap_ms <= 7200000 THEN 5
          ELSE 6
        END AS time_gap_id
      FROM with_gaps
    ), sessions AS (
      -- 按时间顺序把逐行事件聚合为一个变长 session 样本。
      SELECT
        session_id,
        any_value(user_id) AS user_id,
        MIN(event_timestamp) AS session_start,
        MAX(label)::INTEGER AS label,
        COUNT(*)::INTEGER AS sequence_length,
        list(event_id ORDER BY event_timestamp, event_type, event_name) AS event_ids,
        list(event_type_id ORDER BY event_timestamp, event_type, event_name) AS event_type_ids,
        list(page_id ORDER BY event_timestamp, event_type, event_name) AS page_ids,
        list(category_id ORDER BY event_timestamp, event_type, event_name) AS category_ids,
        list(poi_id_hash ORDER BY event_timestamp, event_type, event_name) AS poi_ids,
        list(stage_id ORDER BY event_timestamp, event_type, event_name) AS stage_ids,
        list(time_gap_id ORDER BY event_timestamp, event_type, event_name) AS time_gap_ids
      FROM tokens
      GROUP BY session_id
      HAVING COUNT(*) >= {int(data_cfg['min_seq_len'])}
    ), ordered AS (
      -- 按 session 起始时间切分，模拟用过去数据预测未来数据。
      SELECT *,
        ROW_NUMBER() OVER (ORDER BY session_start, session_id) AS split_row,
        COUNT(*) OVER () AS total_sessions
      FROM sessions
    )
    SELECT * EXCLUDE (split_row, total_sessions),
      CASE
        WHEN split_row <= total_sessions * {train_ratio} THEN 'train'
        WHEN split_row <= total_sessions * {train_ratio + val_ratio} THEN 'validation'
        ELSE 'test'
      END AS split
    FROM ordered
    """
    connection.execute(
        f"COPY ({query}) TO {_literal(output_path)} (FORMAT PARQUET, COMPRESSION ZSTD)"
    )

    # 保存各 split 的样本规模和序列长度，训练前即可检查数据分布。
    stats_rows = connection.execute(
        f"""
        SELECT split, COUNT(*) AS sessions, SUM(label) AS positives,
               AVG(label) AS positive_rate, AVG(sequence_length) AS mean_length,
               quantile_cont(sequence_length, 0.95) AS p95_length
        FROM read_parquet({_literal(output_path)}) GROUP BY split ORDER BY split
        """
    ).fetchall()
    metadata = {
        "samples_path": str(output_path),
        "hash_buckets": buckets,
        "stage_names": STAGE_NAMES,
        "time_gap_names": TIME_GAP_NAMES,
        "splits": [
            dict(
                zip(
                    ["split", "sessions", "positives", "positive_rate", "mean_length", "p95_length"],
                    row,
                )
            )
            for row in stats_rows
        ],
    }
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    # 哈希可能发生碰撞，解释时为每个 event_id 保留出现频率最高的原始名称。
    lookup_query = f"""
    WITH counts AS (
      SELECT {event_id} AS event_id, {event_label} AS event_label, COUNT(*) AS frequency
      FROM (
        SELECT event_type, event_name
        FROM read_parquet({_literal(raw_path)})
        {limit_sql}
      ) AS lookup_source
      WHERE event_type <> 'ORDER'
      GROUP BY event_id, event_label
    ), ranked AS (
      SELECT *, ROW_NUMBER() OVER (PARTITION BY event_id ORDER BY frequency DESC) AS rank
      FROM counts
    ) SELECT event_id, event_label, frequency FROM ranked WHERE rank = 1 ORDER BY event_id
    """
    connection.execute(
        f"COPY ({lookup_query}) TO {_literal(lookup_path)} (HEADER, DELIMITER ',')"
    )
    connection.close()
    return output_path
