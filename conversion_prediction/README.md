# 基于 Transformer 的本地生活消费意图与转化预测

本项目把用户 session 中的搜索、浏览、商户点击、优惠决策等行为表示成序列，在不把订单事件暴露给模型的前提下，预测该 session 是否会发生订单。项目目标不是搭建召回—排序全链路，而是用一条清晰的实验链验证：序列建模是否有效、消费阶段先验是否有效、时间间隔是否进一步改善转化预测。

## 任务定义与防泄漏约束

正样本是包含 `ORDER` 的 session，负样本是不包含 `ORDER` 的 session。正样本输入只保留首个订单时间之前的行为，`ORDER` 本身不会进入序列；否则模型只需识别订单 token 就能获得虚假的高分。切分以 session 起始时间排序后按 70%/15%/15% 划分训练、验证和测试集，而不是随机拆分事件行。预处理、模型选择和最终测试因此具有明确边界。

每个行为位置的表示为 `event + event_type + position`，上下文版本再加入 `page + category + poi`。业务增强版本继续加入消费阶段和时间间隔：输入中的消费阶段分为需求探索、商户发现和消费决策，订单事件只作为标签，不设置对应的输入 Embedding；时间间隔分为序列起点、0—1 分钟、1—5 分钟、5—30 分钟、30 分钟—2 小时和 2 小时以上。高基数 POI 使用哈希桶控制 embedding 参数规模，缺失值统一使用 0 作为 padding/unknown。

## 文件架构

```text
conversion_prediction/
├── configs/base.yaml                 # 路径、特征桶、模型与训练参数
├── data/raw/                         # 原始 CSV 和 Parquet，不提交 Git
├── data/processed/                   # 预处理后的 session 序列
├── artifacts/                        # EDA、checkpoint、指标和解释结果
├── scripts/prepare_data.py           # UTF-8 CSV 分块转换为 Parquet
├── src/conversion_prediction/
│   ├── config.py                     # 配置读取和路径解析
│   ├── preprocessing.py              # 截断 ORDER、构造特征、时间切分
│   ├── dataset.py                    # 动态 padding 与尾部截断
│   ├── models.py                     # Embedding、GRU 和 Transformer
│   ├── training.py                   # 指标、训练、早停和测试
│   ├── analysis.py                   # EDA、订单上下文和行为权重
│   └── cli.py                        # 统一命令入口
└── tests/test_pipeline.py            # 预处理、Dataset 和模型测试
```

## 安装与运行

在本目录执行：

```powershell
pip install -e .
# 仅在需要从原始 CSV 重建 Parquet 时运行下一行
python scripts/prepare_data.py --overwrite
conversion-prediction --config configs/base.yaml preprocess
conversion-prediction --config configs/base.yaml eda
```

推荐按下面四组实验训练。GRU 只使用行为 token；三组 Transformer 逐步加入上下文、消费阶段和时间间隔，使每个增量都有独立的消融依据。

```powershell
conversion-prediction --config configs/base.yaml train --model gru --features event
conversion-prediction --config configs/base.yaml train --model transformer --features context
conversion-prediction --config configs/base.yaml train --model transformer --features context_stage
conversion-prediction --config configs/base.yaml train --model transformer --features context_stage_time
```

训练会在 `artifacts/<模型>__<特征组合>/` 下保存 `best.pt` 和 `metrics.json`。测试集评估及行为权重导出方式如下：

```powershell
conversion-prediction --config configs/base.yaml evaluate artifacts/transformer__context_stage_time/best.pt
conversion-prediction --config configs/base.yaml explain artifacts/transformer__context_stage_time/best.pt
python -m unittest discover -s tests -v
```

解释文件中的 `pooling_weight` 表示 Transformer 编码后各行为对 session 表示的贡献，不等同于 Encoder 内部某一个 self-attention head 的权重。它适合比较高转化与低转化路径中的关键行为，同时避免把内部多层多头权重过度解释为因果关系。

## 当前数据说明

默认配置读取项目内 `data/raw/view_data.parquet`。当前日志约 680 万行，正式预处理会消耗一定时间和磁盘；首次调试可在 `configs/base.yaml` 中设置 `data.row_limit`，训练链路验证完成后再恢复为 `null`。

原始 CSV 与现有 Parquet 的中文字段均已核验为完整 UTF-8，Parquet 的 8 个主要文本字段中没有 Unicode 替换字符 `U+FFFD`。此前出现的乱码只是旧 Conda Python 与 PowerShell 的输出编码不一致，并非数据损坏；统一命令入口会主动把标准输出和错误输出设置为 UTF-8。消费阶段规则会直接匹配 `event_name`、`page_name`、`classification_name` 和一级品类中的中英文业务关键词。
