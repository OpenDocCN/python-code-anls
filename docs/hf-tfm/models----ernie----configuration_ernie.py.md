# `.\models\ernie\configuration_ernie.py`

```
# 导入所需模块
from collections import OrderedDict
from typing import Mapping

# 从 transformers 库中导入预训练配置类 PretrainedConfig
from ...configuration_utils import PretrainedConfig
# 从 transformers 库中导入 OnnxConfig 类
from ...onnx import OnnxConfig
# 从 transformers 库中导入 logging 模块
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义 ERNIE 预训练模型和配置文件的映射关系
ERNIE_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "nghuyong/ernie-1.0-base-zh": "https://huggingface.co/nghuyong/ernie-1.0-base-zh/resolve/main/config.json",
    "nghuyong/ernie-2.0-base-en": "https://huggingface.co/nghuyong/ernie-2.0-base-en/resolve/main/config.json",
    "nghuyong/ernie-2.0-large-en": "https://huggingface.co/nghuyong/ernie-2.0-large-en/resolve/main/config.json",
    "nghuyong/ernie-3.0-base-zh": "https://huggingface.co/nghuyong/ernie-3.0-base-zh/resolve/main/config.json",
    "nghuyong/ernie-3.0-medium-zh": "https://huggingface.co/nghuyong/ernie-3.0-medium-zh/resolve/main/config.json",
    "nghuyong/ernie-3.0-mini-zh": "https://huggingface.co/nghuyong/ernie-3.0-mini-zh/resolve/main/config.json",
    "nghuyong/ernie-3.0-micro-zh": "https://huggingface.co/nghuyong/ernie-3.0-micro-zh/resolve/main/config.json",
    "nghuyong/ernie-3.0-nano-zh": "https://huggingface.co/nghuyong/ernie-3.0-nano-zh/resolve/main/config.json",
    "nghuyong/ernie-gram-zh": "https://huggingface.co/nghuyong/ernie-gram-zh/resolve/main/config.json",
    "nghuyong/ernie-health-zh": "https://huggingface.co/nghuyong/ernie-health-zh/resolve/main/config.json",
}

# 定义 ERNIE 的配置类，继承自 PretrainedConfig
class ErnieConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ErnieModel`] or a [`TFErnieModel`]. It is used to
    instantiate a ERNIE model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the ERNIE
    [nghuyong/ernie-3.0-base-zh](https://huggingface.co/nghuyong/ernie-3.0-base-zh) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Examples:

    ```python
    >>> from transformers import ErnieConfig, ErnieModel

    >>> # Initializing a ERNIE nghuyong/ernie-3.0-base-zh style configuration
    >>> configuration = ErnieConfig()
    # 从 nghuyong/ernie-3.0-base-zh 风格的配置中初始化一个模型(带有随机权重)
    model = ErnieModel(configuration)
    
    # 访问模型配置
    configuration = model.config
class ErnieOnnxConfig(OnnxConfig):
    # 定义属性 inputs，返回一个字符串到整数到字符串的映射
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 如果任务是多项选择
        if self.task == "multiple-choice":
            # 动态轴等于{0: "batch", 1: "choice", 2: "sequence"}
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        else:
            # 否则，动态轴等于{0: "batch", 1: "sequence"}
            dynamic_axis = {0: "batch", 1: "sequence"}
        # 返回一个有序字典，包含输入名称和动态轴的映射关系
        return OrderedDict(
            [
                ("input_ids", dynamic_axis),
                ("attention_mask", dynamic_axis),
                ("token_type_ids", dynamic_axis),
                ("task_type_ids", dynamic_axis),
            ]
        )
```