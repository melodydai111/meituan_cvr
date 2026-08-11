"""集中验证预处理、batch 构造和四组模型的基本张量逻辑。"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# 未安装 editable package 时，也可以直接从 src 目录运行测试。
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from conversion_prediction.dataset import SEQUENCE_COLUMNS, make_collate_fn
from conversion_prediction.models import build_model
from conversion_prediction.preprocessing import build_session_samples


MODEL_CONFIG = {
    "data": {
        "max_seq_len": 8,
        "hash_buckets": {"event": 32, "page": 16, "category": 8, "poi": 64},
    },
    "model": {
        "hidden_dim": 16,
        "num_layers": 2,
        "num_heads": 4,
        "ff_dim": 32,
        "dropout": 0.1,
    },
}


class PipelineTest(unittest.TestCase):
    def test_padding_and_tail_truncation(self):
        """长序列应保留最近行为，短序列应在右侧补零。"""

        def sample(length, label):
            values = np.arange(1, length + 1, dtype=np.int64)
            row = {name: values.copy() for name in SEQUENCE_COLUMNS}
            row.update(session_id=str(length), label=label)
            return row

        batch = make_collate_fn(3)([sample(2, 0), sample(5, 1)])
        self.assertEqual(tuple(batch["event_ids"].shape), (2, 3))
        self.assertEqual(batch["event_ids"][1].tolist(), [3, 4, 5])
        self.assertEqual(batch["mask"].sum(dim=1).tolist(), [2, 3])

    def test_model_output_shapes(self):
        """四组实验都应输出每个 session 对应的一个 logit。"""
        batch = {
            "event_ids": torch.tensor([[1, 2, 3], [4, 5, 0]]),
            "event_type_ids": torch.tensor([[1, 2, 1], [2, 1, 0]]),
            "page_ids": torch.tensor([[1, 2, 3], [2, 1, 0]]),
            "category_ids": torch.tensor([[1, 2, 3], [2, 1, 0]]),
            "poi_ids": torch.tensor([[1, 2, 3], [2, 1, 0]]),
            "stage_ids": torch.tensor([[1, 2, 3], [2, 1, 0]]),
            "time_gap_ids": torch.tensor([[1, 2, 3], [2, 1, 0]]),
            "positions": torch.tensor([[0, 1, 2], [0, 1, 2]]),
            "mask": torch.tensor([[True, True, True], [True, True, False]]),
            "lengths": torch.tensor([3, 2]),
        }
        experiments = [
            ("gru", "event"),
            ("transformer", "context"),
            ("transformer", "context_stage"),
            ("transformer", "context_stage_time"),
        ]
        for model_name, feature_set in experiments:
            with self.subTest(model=model_name, features=feature_set):
                output = build_model(MODEL_CONFIG, model_name, feature_set)(batch)
                self.assertEqual(tuple(output["logits"].shape), (2,))
                self.assertTrue(torch.isfinite(output["logits"]).all())

    def test_preprocessing_removes_order_and_splits_by_time(self):
        """ORDER 只能作为标签，不能进入模型输入序列。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []

            # 构造 20 个时间递增的 session，其中偶数 session 最后发生订单。
            for session_index in range(20):
                base = 1_700_000_000_000 + session_index * 10_000
                session_id = f"s{session_index:02d}"
                rows.extend(
                    [
                        {
                            "session_id": session_id,
                            "user_id": session_index,
                            "event_timestamp": base,
                            "event_type": "PV",
                            "event_name": "search",
                            "page_name": "home",
                            "classification_name": "首页",
                            "first_cate_name": "food",
                            "poi_id": None,
                        },
                        {
                            "session_id": session_id,
                            "user_id": session_index,
                            "event_timestamp": base + 1000,
                            "event_type": "MC",
                            "event_name": "merchant_card",
                            "page_name": "list",
                            "classification_name": "商家卡片",
                            "first_cate_name": "food",
                            "poi_id": 100 + session_index,
                        },
                    ]
                )
                if session_index % 2 == 0:
                    rows.append(
                        {
                            "session_id": session_id,
                            "user_id": session_index,
                            "event_timestamp": base + 2000,
                            "event_type": "ORDER",
                            "event_name": "_order",
                            "page_name": "submit",
                            "classification_name": "订单",
                            "first_cate_name": None,
                            "poi_id": None,
                        }
                    )

            raw_path = root / "raw.parquet"
            pd.DataFrame(rows).to_parquet(raw_path, index=False)
            config = {
                "paths": {
                    "raw_data": str(raw_path),
                    "processed_dir": str(root / "processed"),
                },
                "data": {
                    "min_seq_len": 2,
                    "row_limit": None,
                    "split": {"train": 0.7, "validation": 0.15, "test": 0.15},
                    "hash_buckets": {
                        "event": 32,
                        "page": 16,
                        "category": 8,
                        "poi": 64,
                    },
                },
            }
            output = build_session_samples(config)
            samples = pd.read_parquet(output)

            self.assertEqual(len(samples), 20)
            self.assertEqual(set(samples["split"]), {"train", "validation", "test"})
            self.assertTrue((samples["sequence_length"] == 2).all())
            self.assertTrue((samples["event_type_ids"].map(max) <= 2).all())
            lookup = pd.read_csv(root / "processed" / "event_lookup.csv")
            self.assertFalse(lookup["event_label"].str.contains("ORDER").any())


if __name__ == "__main__":
    unittest.main()
