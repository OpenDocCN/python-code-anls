# `.\transformers\data\datasets\squad.py`

```
# 导入必要的库
import os  # 用于操作文件路径
import time  # 用于时间相关操作
from dataclasses import dataclass, field  # 用于定义数据类
from enum import Enum  # 用于定义枚举类型
from typing import Dict, List, Optional, Union  # 用于类型提示

import torch  # PyTorch 深度学习库
from filelock import FileLock  # 用于文件锁定，防止多个进程同时访问文件
from torch.utils.data import Dataset  # PyTorch 数据集类

# 导入 Hugging Face Transformers 库中的相关模块
from ...models.auto.modeling_auto import MODEL_FOR_QUESTION_ANSWERING_MAPPING  # 导入自动加载的模型映射
from ...tokenization_utils import PreTrainedTokenizer  # 导入预训练分词器
from ...utils import logging  # 导入日志记录模块

# 获取 logger 对象，用于记录日志
logger = logging.get_logger(__name__)

# 获取所有模型配置类的列表
MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())
# 获取所有模型类型的元组
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

# 定义用于训练数据参数的数据类
@dataclass
class SquadDataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    # 模型类型，从给定的模型类型列表中选择
    model_type: str = field(
        default=None, metadata={"help": "Model type selected in the list: " + ", ".join(MODEL_TYPES)}
    )
    # 输入数据目录，包含 SQuAD 任务的 .json 文件
    data_dir: str = field(
        default=None, metadata={"help": "The input data dir. Should contain the .json files for the SQuAD task."}
    )
    # 在分词后的最大序列长度。长于该长度将被截断，短于该长度将被填充。
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    # 文档分段时的步长
    doc_stride: int = field(
        default=128,
        metadata={"help": "When splitting up a long document into chunks, how much stride to take between chunks."},
    )
    # 问题的最大 token 数量。长于该长度的问题将被截断到此长度。
    max_query_length: int = field(
        default=64,
        metadata={
            "help": (
                "The maximum number of tokens for the question. Questions longer than this will "
                "be truncated to this length."
            )
        },
    )
    # 可生成的答案的最大长度。需要因为开始和结束预测不会相互约束。
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": (
                "The maximum length of an answer that can be generated. This is needed because the start "
                "and end predictions are not conditioned on one another."
            )
        },
    )
    # 是否覆盖已缓存的训练和评估集
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    # 是否包含负例，用于SQuAD数据集，如果为True，则SQuAD示例中可能包含没有答案的示例。
    version_2_with_negative: bool = field(
        default=False, metadata={"help": "If true, the SQuAD examples contain some that do not have an answer."}
    )
    # 预测空值的阈值。如果null_score - best_non_null大于该阈值，则预测为空。
    null_score_diff_threshold: float = field(
        default=0.0, metadata={"help": "If null_score - best_non_null is greater than the threshold predict null."}
    )
    # 保留的最佳答案数量。如果null_score - best_non_null大于阈值，则预测为空。
    n_best_size: int = field(
        default=20, metadata={"help": "If null_score - best_non_null is greater than the threshold predict null."}
    )
    # 输入的语言ID，用于语言特定的XLM模型（参见tokenization_xlm.PRETRAINED_INIT_CONFIGURATION）
    lang_id: int = field(
        default=0,
        metadata={
            "help": (
                "language id of input for language-specific xlm models (see"
                " tokenization_xlm.PRETRAINED_INIT_CONFIGURATION)"
            )
        },
    )
    # 转换示例为特征时使用的线程数
    threads: int = field(default=1, metadata={"help": "multiple threads for converting example to features"})
from enum import Enum
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
from typing import List, Optional, Union, Dict


class Split(Enum):
    train = "train"  # 训练集
    dev = "dev"      # 开发集


class SquadDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """
    # 定义Squad数据集类

    args: SquadDataTrainingArguments  # 定义参数args
    features: List[SquadFeatures]     # 定义特征列表
    mode: Split                       # 定义模式（训练或开发）
    is_language_sensitive: bool       # 是否是语言敏感的标志

    def __init__(
        self,
        args: SquadDataTrainingArguments,               # 参数args
        tokenizer: PreTrainedTokenizer,                 # 分词器
        limit_length: Optional[int] = None,             # 可选参数，长度限制
        mode: Union[str, Split] = Split.train,          # 模式，默认为训练模式
        is_language_sensitive: Optional[bool] = False,  # 可选参数，是否是语言敏感的，默认为False
        cache_dir: Optional[str] = None,                # 可选参数，缓存目录，默认为None
        dataset_format: Optional[str] = "pt",           # 可选参数，数据集格式，默认为"pt"
    ):
        pass

    def __len__(self):
        return len(self.features)  # 返回特征列表的长度

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # Convert to Tensors and build dataset
        # 将数据转换为张量并构建数据集
        feature = self.features[i]  # 获取第i个特征

        input_ids = torch.tensor(feature.input_ids, dtype=torch.long)           # 输入张量
        attention_mask = torch.tensor(feature.attention_mask, dtype=torch.long) # 注意力掩码张量
        token_type_ids = torch.tensor(feature.token_type_ids, dtype=torch.long) # 分词类型ID张量
        cls_index = torch.tensor(feature.cls_index, dtype=torch.long)           # 类别索引张量
        p_mask = torch.tensor(feature.p_mask, dtype=torch.float)                 # 掩码张量
        is_impossible = torch.tensor(feature.is_impossible, dtype=torch.float)   # 是否不可能张量

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }  # 输入字典

        if self.args.model_type in ["xlm", "roberta", "distilbert", "camembert"]:
            del inputs["token_type_ids"]  # 删除字典中的分词类型ID张量

        if self.args.model_type in ["xlnet", "xlm"]:
            inputs.update({"cls_index": cls_index, "p_mask": p_mask})  # 更新输入字典
            if self.args.version_2_with_negative:
                inputs.update({"is_impossible": is_impossible})  # 更新输入字典
            if self.is_language_sensitive:
                inputs.update({"langs": (torch.ones(input_ids.shape, dtype=torch.int64) * self.args.lang_id)})  # 更新输入字典

        if self.mode == Split.train:
            start_positions = torch.tensor(feature.start_position, dtype=torch.long)  # 开始位置张量
            end_positions = torch.tensor(feature.end_position, dtype=torch.long)      # 结束位置张量
            inputs.update({"start_positions": start_positions, "end_positions": end_positions})  # 更新输入字典

        return inputs  # 返回输入字典
```