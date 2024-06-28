# `.\data\__init__.py`

```
# 导入数据收集器模块，包括多个特定用途的数据收集器类和函数
from .data_collator import (
    DataCollatorForLanguageModeling,  # 语言建模数据收集器
    DataCollatorForPermutationLanguageModeling,  # 排列语言建模数据收集器
    DataCollatorForSeq2Seq,  # 序列到序列数据收集器
    DataCollatorForSOP,  # SOP（句子顺序预测）数据收集器
    DataCollatorForTokenClassification,  # 标记分类数据收集器
    DataCollatorForWholeWordMask,  # 全词蒙版数据收集器
    DataCollatorWithPadding,  # 带填充功能的数据收集器
    DefaultDataCollator,  # 默认数据收集器类
    default_data_collator,  # 默认数据收集器函数
)

# 导入指标计算函数，用于 GLUE 和 XNLI 数据集的评估
from .metrics import glue_compute_metrics, xnli_compute_metrics

# 导入数据处理器和相关类、函数，用于处理输入样本和特征
from .processors import (
    DataProcessor,  # 数据处理器基类
    InputExample,  # 输入样本类
    InputFeatures,  # 输入特征类
    SingleSentenceClassificationProcessor,  # 单句分类任务处理器
    SquadExample,  # SQuAD 样本类
    SquadFeatures,  # SQuAD 特征类
    SquadV1Processor,  # SQuAD v1 处理器
    SquadV2Processor,  # SQuAD v2 处理器
    glue_convert_examples_to_features,  # 将 GLUE 样本转换为特征的函数
    glue_output_modes,  # GLUE 数据集的输出模式
    glue_processors,  # GLUE 数据集的处理器
    glue_tasks_num_labels,  # GLUE 任务对应的标签数量
    squad_convert_examples_to_features,  # 将 SQuAD 样本转换为特征的函数
    xnli_output_modes,  # XNLI 数据集的输出模式
    xnli_processors,  # XNLI 数据集的处理器
    xnli_tasks_num_labels,  # XNLI 任务对应的标签数量
)
```