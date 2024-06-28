# `.\data\datasets\glue.py`

```
# 引入必要的模块和库
import os
import time
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Union

import torch
from filelock import FileLock
from torch.utils.data import Dataset

# 引入自定义的日志记录模块
from ...utils import logging
# 引入处理 GLUE 任务数据的相关方法和类
from ..processors.glue import glue_convert_examples_to_features, glue_output_modes, glue_processors
from ..processors.utils import InputFeatures

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)


@dataclass
class GlueDataTrainingArguments:
    """
    用于指定模型训练和评估所需数据的参数。

    使用 `HfArgumentParser` 可以将此类转换为 argparse 参数，以便能够在命令行上指定它们。
    """

    # GLUE 任务的名称，应与预处理器的键匹配
    task_name: str = field(metadata={"help": "The name of the task to train on: " + ", ".join(glue_processors.keys())})
    # 包含任务数据文件（如 .tsv 文件）的目录
    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    # 在标记化后的最大总输入序列长度，超过此长度的序列将被截断，长度不足的序列将被填充
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    # 是否覆盖缓存的训练和评估数据集
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    def __post_init__(self):
        self.task_name = self.task_name.lower()


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


class GlueDataset(Dataset):
    """
    此类将很快被一个与框架无关的方法替代。
    """

    # GLUE 数据集的参数
    args: GlueDataTrainingArguments
    # 输出模式
    output_mode: str
    # 输入特征列表
    features: List[InputFeatures]

    def __init__(
        self,
        args: GlueDataTrainingArguments,
        tokenizer: PreTrainedTokenizerBase,
        limit_length: Optional[int] = None,
        mode: Union[str, Split] = Split.train,
        cache_dir: Optional[str] = None,
    ):
        """
        初始化 GLUE 数据集。

        Args:
            args: GlueDataTrainingArguments 类的实例，包含数据集相关参数。
            tokenizer: 预训练的分词器。
            limit_length: 可选参数，限制数据长度。
            mode: 数据集模式，可以是字符串或 Split 枚举类型。
            cache_dir: 可选参数，缓存目录。
        """

        # 设置数据集参数
        self.args = args
        # 设置输出模式
        self.output_mode = glue_output_modes[args.task_name]
        # 从 GLUE 处理器中获取输入特征列表
        self.features = glue_convert_examples_to_features(
            tokenizer=tokenizer,
            examples=None,  # 此处通常传入数据示例
            max_length=args.max_seq_length,
            task=args.task_name,
            label_list=None,  # 此处通常传入标签列表
            output_mode=self.output_mode,
        )

    def __len__(self):
        """
        返回数据集中的样本数量。
        """
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        """
        获取指定索引处的输入特征。

        Args:
            i: 索引值。

        Returns:
            输入特征的实例。
        """
        return self.features[i]

    def get_labels(self):
        """
        返回数据集的标签列表。
        """
        return self.label_list
```