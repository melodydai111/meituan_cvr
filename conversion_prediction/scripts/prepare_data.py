"""将项目根目录中的 UTF-8 行为日志 CSV 分块转换为 Parquet。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import io
from typing import cast

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"


def configure_console() -> None:
    """保证转换进度和中文路径在 PowerShell 中正常显示。"""
    stdout = cast(io.TextIOWrapper, sys.stdout)
    stderr = cast(io.TextIOWrapper, sys.stderr)
    stdout.reconfigure(encoding="utf-8", errors="backslashreplace") 
    stderr.reconfigure(encoding="utf-8", errors="backslashreplace")


def convert(csv_path: Path, parquet_path: Path, chunk_size: int) -> None:
    """分块读取 CSV 并转换为 Parquet。"""
    reader = pd.read_csv(csv_path, encoding="utf-8", chunksize=chunk_size, low_memory=False)
    writer = None
    total_rows = 0
    for chunk in reader:
        table = pa.Table.from_pandas(chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(parquet_path, table.schema, compression="zstd")
        writer.write_table(table)
        total_rows += len(chunk)
    if writer is not None:
        writer.close()
    print(f"转换完成：{total_rows:,} 行")


def main() -> None:
    """解析输入、输出和分块大小参数。"""
    configure_console()
    parser = argparse.ArgumentParser(description="UTF-8 CSV 转 Parquet")
    parser.add_argument("--input", type=Path, default=RAW_DATA_DIR / "view_data.csv")
    parser.add_argument("--output", type=Path, default=RAW_DATA_DIR / "view_data.parquet")
    parser.add_argument("--chunk-size", type=int, default=500_000)
    args = parser.parse_args()
    convert(args.input, args.output, args.chunk_size)


if __name__ == "__main__":
    main()
