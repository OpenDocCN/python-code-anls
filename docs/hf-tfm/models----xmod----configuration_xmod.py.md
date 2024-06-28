# `.\models\xmod\configuration_xmod.py`

```py
# 引入需要的模块和类
from collections import OrderedDict  # 导入有序字典类
from typing import Mapping  # 导入映射类型

# 从相关的模块中导入配置类和其它必要的配置
from ...configuration_utils import PretrainedConfig  # 导入预训练配置类
from ...onnx import OnnxConfig  # 导入ONNX配置类
from ...utils import logging  # 导入日志工具

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# X-MOD预训练模型配置文件映射字典，将模型名称映射到其配置文件的URL
XMOD_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/xmod-base": "https://huggingface.co/facebook/xmod-base/resolve/main/config.json",
    "facebook/xmod-large-prenorm": "https://huggingface.co/facebook/xmod-large-prenorm/resolve/main/config.json",
    "facebook/xmod-base-13-125k": "https://huggingface.co/facebook/xmod-base-13-125k/resolve/main/config.json",
    "facebook/xmod-base-30-125k": "https://huggingface.co/facebook/xmod-base-30-125k/resolve/main/config.json",
    "facebook/xmod-base-30-195k": "https://huggingface.co/facebook/xmod-base-30-195k/resolve/main/config.json",
    "facebook/xmod-base-60-125k": "https://huggingface.co/facebook/xmod-base-60-125k/resolve/main/config.json",
    "facebook/xmod-base-60-265k": "https://huggingface.co/facebook/xmod-base-60-265k/resolve/main/config.json",
    "facebook/xmod-base-75-125k": "https://huggingface.co/facebook/xmod-base-75-125k/resolve/main/config.json",
    "facebook/xmod-base-75-269k": "https://huggingface.co/facebook/xmod-base-75-269k/resolve/main/config.json",
}


class XmodConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`XmodModel`]. It is used to instantiate an X-MOD
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the
    [facebook/xmod-base](https://huggingface.co/facebook/xmod-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Examples:

    ```
    >>> from transformers import XmodConfig, XmodModel

    >>> # Initializing an X-MOD facebook/xmod-base style configuration
    >>> configuration = XmodConfig()

    >>> # Initializing a model (with random weights) from the facebook/xmod-base style configuration
    >>> model = XmodModel(configuration)

    >>> # Accessing the model configuration
    ```
    """
    pass  # 此类定义了X-MOD模型的配置，继承自PretrainedConfig类，并提供了实例化模型和控制输出的能力
    # 获取模型的配置信息
    configuration = model.config
# 从 transformers.models.roberta.configuration_roberta.RobertaOnnxConfig 复制并修改为 XmodOnnxConfig
class XmodOnnxConfig(OnnxConfig):
    
    # 定义 inputs 属性，返回一个映射，表示模型的输入格式
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 根据任务类型设置动态轴的不同配置
        if self.task == "multiple-choice":
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        else:
            dynamic_axis = {0: "batch", 1: "sequence"}
        
        # 返回有序字典，指定输入名称和对应的动态轴
        return OrderedDict(
            [
                ("input_ids", dynamic_axis),        # 输入的 token IDs
                ("attention_mask", dynamic_axis),  # 输入的注意力掩码
            ]
        )
```