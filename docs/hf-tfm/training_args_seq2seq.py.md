# `.\training_args_seq2seq.py`

```py
# 导入日志模块，用于记录程序运行时的信息
import logging
# 导入数据类装饰器和字段装饰器
from dataclasses import dataclass, field
# 导入路径操作模块
from pathlib import Path
# 导入可选类型模块
from typing import Optional, Union

# 导入生成配置工具模块中的生成配置类
from .generation.configuration_utils import GenerationConfig
# 导入训练参数类
from .training_args import TrainingArguments
# 导入添加文档字符串装饰器
from .utils import add_start_docstrings

# 获取当前模块的日志记录器对象
logger = logging.getLogger(__name__)

# 定义一个数据类，继承自TrainingArguments类，并添加了额外的文档字符串
@dataclass
@add_start_docstrings(TrainingArguments.__doc__)
class Seq2SeqTrainingArguments(TrainingArguments):
    """
    Args:
        sortish_sampler (`bool`, *optional*, defaults to `False`):
            是否使用 SortishSampler。目前仅在底层数据集为 Seq2SeqDataset 时可用，
            但将来会普遍可用。

            SortishSampler 根据长度排序输入，以最小化填充大小，并在训练集中略微引入随机性。
        predict_with_generate (`bool`, *optional*, defaults to `False`):
            是否使用生成来计算生成性指标（ROUGE、BLEU）。
        generation_max_length (`int`, *optional*):
            当 `predict_with_generate=True` 时，在每次评估循环中使用的 `max_length`。
            将默认使用模型配置的 `max_length` 值。
        generation_num_beams (`int`, *optional*):
            当 `predict_with_generate=True` 时，在每次评估循环中使用的 `num_beams`。
            将默认使用模型配置的 `num_beams` 值。
        generation_config (`str` or `Path` or [`~generation.GenerationConfig`], *optional*):
            允许从 `from_pretrained` 方法加载 [`~generation.GenerationConfig`]。
            可以是以下之一：

            - 字符串，托管在 huggingface.co 模型库中的预训练模型配置的模型标识。
            - 目录路径，包含使用 [`~GenerationConfig.save_pretrained`] 方法保存的配置文件，例如 `./my_model_directory/`。
            - [`~generation.GenerationConfig`] 对象。
    """

    # 是否使用 SortishSampler 进行采样，默认为 False
    sortish_sampler: bool = field(default=False, metadata={"help": "Whether to use SortishSampler or not."})
    # 是否使用生成来计算生成性指标（ROUGE、BLEU），默认为 False
    predict_with_generate: bool = field(
        default=False, metadata={"help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)."}
    )
    generation_max_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The `max_length` to use on each evaluation loop when `predict_with_generate=True`. Will default "
                "to the `max_length` value of the model configuration."
            )
        },
    )
    generation_num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The `num_beams` to use on each evaluation loop when `predict_with_generate=True`. Will default "
                "to the `num_beams` value of the model configuration."
            )
        },
    )
    generation_config: Optional[Union[str, Path, GenerationConfig]] = field(
        default=None,
        metadata={
            "help": "Model id, file path or url pointing to a GenerationConfig json file, to use during prediction."
        },
    )


# 定义类的数据成员 generation_max_length，用于控制生成的文本长度，在预测时使用
generation_max_length: Optional[int] = field(
    default=None,
    metadata={
        "help": (
            "The `max_length` to use on each evaluation loop when `predict_with_generate=True`. Will default "
            "to the `max_length` value of the model configuration."
        )
    },
)

# 定义类的数据成员 generation_num_beams，用于控制生成的 beam search 数量，在预测时使用
generation_num_beams: Optional[int] = field(
    default=None,
    metadata={
        "help": (
            "The `num_beams` to use on each evaluation loop when `predict_with_generate=True`. Will default "
            "to the `num_beams` value of the model configuration."
        )
    },
)

# 定义类的数据成员 generation_config，用于指定生成配置的路径或对象，在预测时使用
generation_config: Optional[Union[str, Path, GenerationConfig]] = field(
    default=None,
    metadata={
        "help": "Model id, file path or url pointing to a GenerationConfig json file, to use during prediction."
    },
)



    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values and `GenerationConfig` by dictionaries (for JSON
        serialization support). It obfuscates the token values by removing their value.
        """
        # filter out fields that are defined as field(init=False)
        d = super().to_dict()
        for k, v in d.items():
            if isinstance(v, GenerationConfig):
                d[k] = v.to_dict()
        return d


# 定义一个方法 to_dict，用于将对象实例序列化为字典
def to_dict(self):
    """
    Serializes this instance while replace `Enum` by their values and `GenerationConfig` by dictionaries (for JSON
    serialization support). It obfuscates the token values by removing their value.
    """
    # 调用父类的 to_dict 方法，获取对象实例的字典表示
    d = super().to_dict()
    # 遍历字典 d 的键值对
    for k, v in d.items():
        # 如果值 v 的类型是 GenerationConfig 类的实例
        if isinstance(v, GenerationConfig):
            # 将值 v 转换为其字典表示，以支持 JSON 序列化
            d[k] = v.to_dict()
    return d
```