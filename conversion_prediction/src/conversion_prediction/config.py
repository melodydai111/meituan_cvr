"""读取项目配置，并集中管理数据和实验产物路径。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path) -> Dict[str, Any]:
    """读取 YAML，并把数据与产物路径解析为绝对路径。"""
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # 配置文件位于 configs/，其上一级就是 conversion_prediction 项目目录。
    project_dir = config_path.parent.parent
    # 将相对路径转成绝对路径，checkpoint 在其他目录加载时仍能找到数据。
    for key in ("raw_data", "processed_dir", "artifacts_dir"):
        value = Path(config["paths"][key])
        if not value.is_absolute():
            value = project_dir / value
        config["paths"][key] = str(value.resolve())
    return config


def ensure_directories(config: Dict[str, Any]) -> None:
    """创建预处理数据目录和实验产物目录。"""
    Path(config["paths"]["processed_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["paths"]["artifacts_dir"]).mkdir(parents=True, exist_ok=True)
