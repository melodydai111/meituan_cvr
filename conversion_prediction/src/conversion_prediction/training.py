"""模型训练、离线指标、早停、checkpoint 保存和独立测试流程。"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from .dataset import SessionDataset, make_collate_fn
from .models import build_model


def set_seed(seed: int) -> None:
    """固定随机数生成器，使不同模型实验具有可比性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    """auto 模式优先使用 GPU，没有 GPU 时使用 CPU。"""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def move_batch(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """只移动张量，session_id 等解释性字段仍保留在 CPU。"""
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def make_loaders(config: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """为训练、验证和测试集建立使用同一补零规则的 DataLoader。"""
    data_path = Path(config["paths"]["processed_dir"]) / "session_samples.parquet"
    sample_limit = config["training"].get("sample_limit")
    datasets = [
        SessionDataset(data_path, split, sample_limit)
        for split in ("train", "validation", "test")
    ]
    collate = make_collate_fn(config["data"]["max_seq_len"])
    common = dict(
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        collate_fn=collate,
        pin_memory=torch.cuda.is_available(),
    )

    # 只有训练集打乱顺序；验证和测试保持确定顺序。
    return (
        DataLoader(datasets[0], shuffle=True, **common),
        DataLoader(datasets[1], shuffle=False, **common),
        DataLoader(datasets[2], shuffle=False, **common),
    )


def binary_metrics(labels, probabilities) -> Dict[str, float]:
    """计算 AUC、PR-AUC 和概率预测误差 Logloss。"""
    labels = np.asarray(labels, dtype=np.int64)

    # Logloss 不能接收严格的 0 或 1，只做数值稳定所需的极小裁剪。
    probabilities = np.clip(np.asarray(probabilities), 1e-7, 1 - 1e-7)
    return {
        "auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "logloss": float(log_loss(labels, probabilities, labels=[0, 1])),
    }


@torch.no_grad()
def evaluate_loader(model, loader, device: torch.device) -> Dict[str, float]:
    """收集整个数据集的预测概率，再统一计算指标。"""
    model.eval()
    labels, probabilities = [], []
    for batch in loader:
        batch = move_batch(batch, device)
        output = model(batch)
        probabilities.extend(torch.sigmoid(output["logits"]).cpu().tolist())
        labels.extend(batch["labels"].cpu().tolist())
    return binary_metrics(labels, probabilities)


def train_experiment(config: Dict, model_name: str, feature_set: str) -> Path:
    """训练一个模型×特征组合，并在验证集最优点评估测试集。"""
    set_seed(config["seed"])
    device = resolve_device(config["training"]["device"])
    train_loader, validation_loader, test_loader = make_loaders(config)
    model = build_model(config, model_name, feature_set).to(device)

    # 按训练集类别比例提高正样本损失权重，缓解订单样本稀少的问题。
    labels = train_loader.dataset.labels
    positives = float(labels.sum())
    negatives = float(len(labels) - labels.sum())
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(negatives / positives, device=device)
    ) # 带pos_weight的BCE损失
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )

    experiment = f"{model_name}__{feature_set}"
    artifact_dir = Path(config["paths"]["artifacts_dir"]) / experiment
    artifact_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = artifact_dir / "best.pt"
    history, best_auc, stale_epochs = [], -float("inf"), 0

    # 每轮训练后使用验证集 AUC 决定是否保存 checkpoint。
    for epoch in range(1, config["training"]["epochs"] + 1):
        model.train()
        losses = []
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True) # 梯度清0
            loss = criterion(model(batch)["logits"], batch["labels"])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0) #梯度裁剪
            optimizer.step()
            losses.append(loss.item())

        validation = evaluate_loader(model, validation_loader, device)
        epoch_result = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            **validation,
        }
        history.append(epoch_result)
        print(json.dumps(epoch_result, ensure_ascii=False))

        if validation["auc"] > best_auc:
            best_auc = validation["auc"]
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": config,
                    "model_name": model_name,
                    "feature_set": feature_set,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= config["training"]["early_stopping_patience"]:
                break

    # 测试集只在训练结束后使用一次，报告验证集最佳模型的泛化结果。
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test_metrics = evaluate_loader(model, test_loader, device)
    result = {
        "experiment": experiment,
        "best_validation_auc": best_auc,
        "test": test_metrics,
        "history": history,
    }
    with (artifact_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    print(json.dumps(test_metrics, ensure_ascii=False, indent=2))
    return checkpoint_path


def evaluate_checkpoint(checkpoint_path: str | Path) -> Dict[str, float]:
    """从 checkpoint 恢复模型，并重新计算测试集指标。"""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    device = resolve_device(config["training"]["device"])
    model = build_model(config, checkpoint["model_name"], checkpoint["feature_set"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    _, _, test_loader = make_loaders(config)
    metrics = evaluate_loader(model, test_loader, device)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    return metrics
