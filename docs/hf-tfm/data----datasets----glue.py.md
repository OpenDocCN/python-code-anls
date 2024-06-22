# `.\transformers\data\datasets\glue.py`

```py
# 导入所需的模块和库
import os
import time
import warnings
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Union

import torch
from filelock import FileLock
from torch.utils.data import Dataset

# 导入 HuggingFace 库中的相关模块和函数
from ...tokenization_utils_base import PreTrainedTokenizerBase
from ...utils import logging
from ..processors.glue import glue_convert_examples_to_features, glue_output_modes, glue_processors
from ..processors.utils import InputFeatures

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义数据训练参数的数据类
@dataclass
class GlueDataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class into argparse arguments to be able to specify them on the command
    line.
    """

    task_name: str = field(metadata={"help": "The name of the task to train on: " + ", ".join(glue_processors.keys())})
    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .tsv files (or other data files) for the task."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

    def __post_init__(self):
        self.task_name = self.task_name.lower()


# 定义数据集分割类别的枚举
class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


# 定义 GLUE 数据集类
class GlueDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    args: GlueDataTrainingArguments  # 训练参数
    output_mode: str  # 输出模式
    features: List[InputFeatures]  # 输入特征列表

    def __init__(
        self,
        args: GlueDataTrainingArguments,
        tokenizer: PreTrainedTokenizerBase,
        limit_length: Optional[int] = None,
        mode: Union[str, Split] = Split.train,
        cache_dir: Optional[str] = None,
    ):
        # 初始化数据集
        pass

    # 获取数据集的长度
    def __len__(self):
        return len(self.features)

    # 获取数据集中指定索引位置的特征
    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]

    # 获取数据集的标签
    def get_labels(self):
        return self.label_list
``` 
```