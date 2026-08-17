"""项目命令行入口，把预处理、训练、评估和分析统一到一组命令中。"""

from __future__ import annotations

import argparse
import sys

from .analysis import explain_checkpoint, run_eda
from .config import ensure_directories, load_config
from .preprocessing import build_session_samples
from .training import evaluate_checkpoint, train_experiment


def configure_utf8_console() -> None:
    """统一 PowerShell 下的中文输出编码。"""
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace") # pyright: ignore[reportAttributeAccessIssue]
    sys.stderr.reconfigure(encoding="utf-8", errors="backslashreplace") # pyright: ignore[reportAttributeAccessIssue]


def make_parser() -> argparse.ArgumentParser:
    """定义命令及其参数，模型名和特征组合在入口处统一约束。"""
    parser = argparse.ArgumentParser(description="本地生活 session 转化预测")
    parser.add_argument("--config", default="configs/base.yaml", help="YAML 配置路径")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("preprocess", help="构造无标签泄漏的 session 序列")
    commands.add_parser("eda", help="生成序列、样本和消费阶段 EDA")

    train = commands.add_parser("train", help="训练一个实验")
    train.add_argument("--model", choices=["gru", "transformer"], required=True)
    train.add_argument(
        "--features",
        choices=["event", "context", "context_stage", "context_stage_time"],
        required=True,
    )
    evaluate = commands.add_parser("evaluate", help="评估 checkpoint")
    evaluate.add_argument("checkpoint")
    explain = commands.add_parser("explain", help="导出 Transformer pooling 权重")
    explain.add_argument("checkpoint")
    explain.add_argument("--max-sessions", type=int, default=200)
    return parser


def main() -> None:
    # 先调整控制台编码，保证后续日志中的中文正常显示。
    configure_utf8_console()
    args = make_parser().parse_args()
    config = load_config(args.config)
    ensure_directories(config)

    # 每个子命令只负责路由，具体逻辑放在独立模块中，便于测试和复用。
    if args.command == "preprocess":
        print(f"已生成 {build_session_samples(config)}")
    elif args.command == "eda":
        print(f"EDA 产物目录: {run_eda(config)}")
    elif args.command == "train":
        print(f"最佳 checkpoint: {train_experiment(config, args.model, args.features)}")
    elif args.command == "evaluate":
        evaluate_checkpoint(args.checkpoint)
    elif args.command == "explain":
        explain_checkpoint(args.checkpoint, args.max_sessions)


if __name__ == "__main__":
    main()
