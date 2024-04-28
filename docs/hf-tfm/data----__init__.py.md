# `.\transformers\data\__init__.py`

```
# 导入自定义的数据收集器模块
from .data_collator import (
    DataCollatorForLanguageModeling,  # 用于语言建模的数据收集器
    DataCollatorForPermutationLanguageModeling,  # 用于排列语言建模的数据收集器
    DataCollatorForSeq2Seq,  # 用于序列到序列模型的数据收集器
    DataCollatorForSOP,  # 用于SOP任务的数据收集器
    DataCollatorForTokenClassification,  # 用于标记分类任务的数据收集器
    DataCollatorForWholeWordMask,  # 用于全词掩码任务的数据收集器
    DataCollatorWithPadding,  # 带填充的数据收集器
    DefaultDataCollator,  # 默认数据收集器
    default_data_collator,  # 默认数据收集器的别名
)

# 导入评估指标计算函数
from .metrics import glue_compute_metrics, xnli_compute_metrics

# 导入数据处理器和相关功能
from .processors import (
    DataProcessor,  # 数据处理器基类
    InputExample,  # 输入示例类
    InputFeatures,  # 输入特征类
    SingleSentenceClassificationProcessor,  # 单句分类任务的数据处理器
    SquadExample,  # SQuAD数据集的示例类
    SquadFeatures,  # SQuAD数据集的特征类
    SquadV1Processor,  # SQuAD v1任务的数据处理器
    SquadV2Processor,  # SQuAD v2任务的数据处理器
    glue_convert_examples_to_features,  # 将GLUE任务的示例转换为特征的函数
    glue_output_modes,  # GLUE任务的输出模式字典
    glue_processors,  # GLUE任务的数据处理器字典
    glue_tasks_num_labels,  # GLUE任务的标签数量字典
    squad_convert_examples_to_features,  # 将SQuAD任务的示例转换为特征的函数
    xnli_output_modes,  # XNLI任务的输出模式字典
    xnli_processors,  # XNLI任务的数据处理器字典
    xnli_tasks_num_labels,  # XNLI任务的标签数量字典
)
```  
```