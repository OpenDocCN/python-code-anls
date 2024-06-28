# `.\data\datasets\squad.py`

```
# 引入标准库和第三方库
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Union

# 引入PyTorch相关库
import torch
from filelock import FileLock
from torch.utils.data import Dataset

# 引入HuggingFace Transformers相关模块
from ...models.auto.modeling_auto import MODEL_FOR_QUESTION_ANSWERING_MAPPING
from ...tokenization_utils import PreTrainedTokenizer
from ...utils import logging
from ..processors.squad import SquadFeatures, SquadV1Processor, SquadV2Processor, squad_convert_examples_to_features

# 获取日志记录器
logger = logging.get_logger(__name__)

# 获取支持的模型配置类别列表
MODEL_CONFIG_CLASSES = list(MODEL_FOR_QUESTION_ANSWERING_MAPPING.keys())

# 获取支持的模型类型元组
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

# 数据类，定义用于SQuAD数据训练的参数
@dataclass
class SquadDataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    # 模型类型，默认为None，可以选择在支持列表中的任意一个
    model_type: str = field(
        default=None, metadata={"help": "Model type selected in the list: " + ", ".join(MODEL_TYPES)}
    )
    # 数据目录，包含SQuAD任务的.json文件
    data_dir: str = field(
        default=None, metadata={"help": "The input data dir. Should contain the .json files for the SQuAD task."}
    )
    # 最大输入序列长度，经过标记后的最大总输入序列长度，超出此长度将被截断，较短的将被填充
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    # 文档步幅，将长文档拆分为块时，块之间的步幅
    doc_stride: int = field(
        default=128,
        metadata={"help": "When splitting up a long document into chunks, how much stride to take between chunks."},
    )
    # 最大问题长度，问题的最大标记数，超出此长度将被截断
    max_query_length: int = field(
        default=64,
        metadata={
            "help": (
                "The maximum number of tokens for the question. Questions longer than this will "
                "be truncated to this length."
            )
        },
    )
    # 最大答案长度，可以生成的答案的最大长度，由于开始和结束预测不受彼此条件的影响
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": (
                "The maximum length of an answer that can be generated. This is needed because the start "
                "and end predictions are not conditioned on one another."
            )
        },
    )
    # 覆盖缓存，是否覆盖缓存的训练和评估集
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    version_2_with_negative: bool = field(
        default=False, metadata={"help": "If true, the SQuAD examples contain some that do not have an answer."}
    )
    # 是否启用 v2 版本的模型，支持 SQuAD 数据集中一些没有答案的情况
    null_score_diff_threshold: float = field(
        default=0.0, metadata={"help": "If null_score - best_non_null is greater than the threshold predict null."}
    )
    # 如果 null_score - best_non_null 大于该阈值，则预测为空值
    n_best_size: int = field(
        default=20, metadata={"help": "If null_score - best_non_null is greater than the threshold predict null."}
    )
    # 生成最佳预测结果的数量上限
    lang_id: int = field(
        default=0,
        metadata={
            "help": (
                "language id of input for language-specific xlm models (see"
                " tokenization_xlm.PRETRAINED_INIT_CONFIGURATION)"
            )
        },
    )
    # 语言 ID，用于特定语言的 XLM 模型输入（参见 tokenization_xlm.PRETRAINED_INIT_CONFIGURATION）
    threads: int = field(default=1, metadata={"help": "multiple threads for converting example to features"})
    # 转换示例为特征时使用的线程数
class Split(Enum):
    # 定义枚举类 Split，包含 train 和 dev 两个成员
    train = "train"
    dev = "dev"


class SquadDataset(Dataset):
    """
    This will be superseded by a framework-agnostic approach soon.
    """

    args: SquadDataTrainingArguments  # 类型注解，指定 args 为 SquadDataTrainingArguments 类型
    features: List[SquadFeatures]    # 类型注解，指定 features 为 SquadFeatures 类型的列表
    mode: Split                      # 类型注解，指定 mode 为 Split 枚举类型
    is_language_sensitive: bool      # 类型注解，指定 is_language_sensitive 为布尔类型

    def __init__(
        self,
        args: SquadDataTrainingArguments,        # 参数 args，类型为 SquadDataTrainingArguments
        tokenizer: PreTrainedTokenizer,          # 参数 tokenizer，类型为 PreTrainedTokenizer
        limit_length: Optional[int] = None,      # 可选参数 limit_length，类型为整数或 None
        mode: Union[str, Split] = Split.train,   # 参数 mode，类型为字符串或 Split 枚举，默认为 Split.train
        is_language_sensitive: Optional[bool] = False,  # 可选参数 is_language_sensitive，默认为 False
        cache_dir: Optional[str] = None,         # 可选参数 cache_dir，默认为 None
        dataset_format: Optional[str] = "pt",    # 可选参数 dataset_format，默认为 "pt"
    ):
        # 初始化方法，设置实例的属性

    def __len__(self):
        # 返回 features 列表的长度作为数据集的长度

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # 获取第 i 个样本的特征，并将其转换为包含张量的字典格式返回

        # 从 features 中取出第 i 个特征
        feature = self.features[i]

        # 将特征的各个部分转换为张量
        input_ids = torch.tensor(feature.input_ids, dtype=torch.long)
        attention_mask = torch.tensor(feature.attention_mask, dtype=torch.long)
        token_type_ids = torch.tensor(feature.token_type_ids, dtype=torch.long)
        cls_index = torch.tensor(feature.cls_index, dtype=torch.long)
        p_mask = torch.tensor(feature.p_mask, dtype=torch.float)
        is_impossible = torch.tensor(feature.is_impossible, dtype=torch.float)

        # 构建输入字典
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

        # 根据模型类型调整输入字典的内容
        if self.args.model_type in ["xlm", "roberta", "distilbert", "camembert"]:
            del inputs["token_type_ids"]

        if self.args.model_type in ["xlnet", "xlm"]:
            inputs.update({"cls_index": cls_index, "p_mask": p_mask})
            if self.args.version_2_with_negative:
                inputs.update({"is_impossible": is_impossible})
            if self.is_language_sensitive:
                inputs.update({"langs": (torch.ones(input_ids.shape, dtype=torch.int64) * self.args.lang_id)})

        # 如果模式是训练模式，添加起始位置和结束位置到输入字典中
        if self.mode == Split.train:
            start_positions = torch.tensor(feature.start_position, dtype=torch.long)
            end_positions = torch.tensor(feature.end_position, dtype=torch.long)
            inputs.update({"start_positions": start_positions, "end_positions": end_positions})

        # 返回构建好的输入字典
        return inputs
```