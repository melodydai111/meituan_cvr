"""将项目根目录中的 UTF-8 行为日志 CSV 分块转换为 Parquet。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_DIR / "data" / "raw"


def configure_console() -> None:
    """保证转换进度和中文路径在 PowerShell 中正常显示。"""
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace")


def convert(csv_path: Path, parquet_path: Path, chunk_size: int, overwrite: bool) -> None:
    """分块读取 CSV，并使用第一块数据确定整份 Parquet 的 schema。"""
    # 原始 Parquet 较大，必须显式指定 --overwrite 才允许替换。
    if parquet_path.exists() and not overwrite:
        raise FileExistsError(f"输出已存在：{parquet_path}；如需重建请添加 --overwrite")

    # 先写临时文件，全部成功后再替换目标，避免留下半份 Parquet。
    temporary_path = parquet_path.with_suffix(parquet_path.suffix + ".tmp")
    temporary_path.unlink(missing_ok=True)
    writer = None
    total_rows = 0
    try:
        # strict 会在遇到非法 UTF-8 时立即报错，不让替换字符进入数据。
        reader = pd.read_csv(
            csv_path,
            encoding="utf-8",
            encoding_errors="strict",
            chunksize=chunk_size,
            low_memory=False,
        )
        schema = None
        for chunk_index, chunk in enumerate(reader, start=1):
            # 第一块推断 schema，后续块全部按相同 schema 转换。
            if schema is None:
                table = pa.Table.from_pandas(chunk, preserve_index=False)
                schema = table.schema
                writer = pq.ParquetWriter(temporary_path, schema, compression="zstd")
            else:
                table = pa.Table.from_pandas(
                    chunk,
                    schema=schema,
                    preserve_index=False,
                    safe=False,
                )
            writer.write_table(table)
            total_rows += len(chunk)
            print(f"Chunk {chunk_index}: {len(chunk):,} 行；累计 {total_rows:,} 行")
    except Exception:
        # 转换失败时清理临时文件，现有正式 Parquet 不受影响。
        if writer is not None:
            writer.close()
        temporary_path.unlink(missing_ok=True)
        raise
    else:
        if writer is not None:
            writer.close()
        temporary_path.replace(parquet_path)

    size_mb = parquet_path.stat().st_size / 1024**2
    print(f"转换完成：{total_rows:,} 行，{size_mb:.1f} MB")
    print(f"输出路径：{parquet_path}")


def main() -> None:
    """解析输入、输出和分块大小参数。"""
    configure_console()
    parser = argparse.ArgumentParser(description="UTF-8 CSV 转 Parquet")
    parser.add_argument("--input", type=Path, default=RAW_DATA_DIR / "view_data.csv")
    parser.add_argument("--output", type=Path, default=RAW_DATA_DIR / "view_data.parquet")
    parser.add_argument("--chunk-size", type=int, default=500_000)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    convert(args.input, args.output, args.chunk_size, args.overwrite)


if __name__ == "__main__":
    main()
