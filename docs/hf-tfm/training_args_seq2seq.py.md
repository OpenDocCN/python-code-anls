# `.\transformers\training_args_seq2seq.py`

```py
# 导入所需模块和库
import logging  # 导入日志模块
from dataclasses import dataclass, field  # 导入数据类相关的模块和函数
from pathlib import Path  # 导入处理文件路径的模块
from typing import Optional, Union  # 导入类型提示相关的模块

# 从generation.configuration_utils模块中导入GenerationConfig类
from .generation.configuration_utils import GenerationConfig
# 从training_args模块中导入TrainingArguments类
from .training_args import TrainingArguments
# 从utils模块中导入add_start_docstrings函数
from .utils import add_start_docstrings

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)

# 使用数据类装饰器定义Seq2SeqTrainingArguments类，继承自TrainingArguments类，并添加文档字符串
@dataclass
@add_start_docstrings(TrainingArguments.__doc__)
class Seq2SeqTrainingArguments(TrainingArguments):
    """
    Args:
        sortish_sampler (`bool`, *optional*, defaults to `False`):
            是否使用“sortish采样”。目前仅当底层数据集为Seq2SeqDataset时才可能使用，但将来会普遍可用。
            
            它根据长度对输入进行排序，以最小化填充大小，训练集有一定程度的随机性。
        predict_with_generate (`bool`, *optional*, defaults to `False`):
            是否使用generate来计算生成性指标（ROUGE、BLEU）。
        generation_max_length (`int`, *optional*):
            当`predict_with_generate=True`时，每个评估循环使用的`max_length`。将默认为模型配置的`max_length`值。
        generation_num_beams (`int`, *optional*):
            当`predict_with_generate=True`时，每个评估循环使用的`num_beams`。将默认为模型配置的`num_beams`值。
        generation_config (`str` or `Path` or [`~generation.GenerationConfig`], *optional*):
            允许从`from_pretrained`方法加载一个[`~generation.GenerationConfig`]。这可以是：

            - 一个字符串，托管在huggingface.co模型库中的预训练模型配置的*模型id*。有效的模型id可以位于根级别，例如`bert-base-uncased`，或者在用户或组织名称下的命名空间中，例如`dbmdz/bert-base-german-cased`。
            - 一个包含使用[`~GenerationConfig.save_pretrained`]方法保存的配置文件的*目录*的路径，例如`./my_model_directory/`。
            - 一个[`~generation.GenerationConfig`]对象。
    """

    # 是否使用SortishSampler
    sortish_sampler: bool = field(default=False, metadata={"help": "Whether to use SortishSampler or not."})
```  
    # 定义一个布尔类型的字段，用于指示是否使用生成模型计算生成度量指标（ROUGE，BLEU）
    predict_with_generate: bool = field(
        default=False, metadata={"help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)."}
    )
    # 定义一个可选的整数字段，用于指定在每次评估循环中使用的`max_length`值，当`predict_with_generate=True`时。将默认为模型配置的`max_length`值
    generation_max_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The `max_length` to use on each evaluation loop when `predict_with_generate=True`. Will default "
                "to the `max_length` value of the model configuration."
            )
        },
    )
    # 定义一个可选的整数字段，用于指定在每次评估循环中使用的`num_beams`值，当`predict_with_generate=True`时。将默认为模型配置的`num_beams`值
    generation_num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The `num_beams` to use on each evaluation loop when `predict_with_generate=True`. Will default "
                "to the `num_beams` value of the model configuration."
            )
        },
    )
    # 定义一个可选的字段，可以是字符串、路径或指向GenerationConfig json文件的URL，用于在预测过程中使用的模型ID、文件路径或URL
    generation_config: Optional[Union[str, Path, GenerationConfig]] = field(
        default=None,
        metadata={
            "help": "Model id, file path or url pointing to a GenerationConfig json file, to use during prediction."
        },
    )

    # 将实例序列化为字典，将`Enum`替换为其值，将`GenerationConfig`替换为字典（用于JSON序列化支持）。通过删除其值来混淆令牌值
    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values and `GenerationConfig` by dictionaries (for JSON
        serialization support). It obfuscates the token values by removing their value.
        """
        # 过滤掉定义为field(init=False)的字段
        d = super().to_dict()
        for k, v in d.items():
            if isinstance(v, GenerationConfig):
                d[k] = v.to_dict()
        return d
```