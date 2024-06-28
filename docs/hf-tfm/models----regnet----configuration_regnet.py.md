# `.\models\regnet\configuration_regnet.py`

```py
# 导入所需模块和类
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取全局日志记录器
logger = logging.get_logger(__name__)

# 预训练配置文件的映射字典，将模型名称映射到配置文件的URL
REGNET_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/regnet-y-040": "https://huggingface.co/facebook/regnet-y-040/blob/main/config.json",
}

# RegNetConfig 类，继承自 PretrainedConfig 类，用于存储 RegNet 模型的配置信息
class RegNetConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`RegNetModel`]. It is used to instantiate a RegNet
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the RegNet
    [facebook/regnet-y-040](https://huggingface.co/facebook/regnet-y-040) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        embedding_size (`int`, *optional*, defaults to 64):
            Dimensionality (hidden size) for the embedding layer.
        hidden_sizes (`List[int]`, *optional*, defaults to `[256, 512, 1024, 2048]`):
            Dimensionality (hidden size) at each stage.
        depths (`List[int]`, *optional*, defaults to `[3, 4, 6, 3]`):
            Depth (number of layers) for each stage.
        layer_type (`str`, *optional*, defaults to `"y"`):
            The layer to use, it can be either `"x" or `"y"`. An `x` layer is a ResNet's BottleNeck layer with
            `reduction` fixed to `1`. While a `y` layer is a `x` but with squeeze and excitation. Please refer to the
            paper for a detailed explanation of how these layers were constructed.
        hidden_act (`str`, *optional*, defaults to `"relu"`):
            The non-linear activation function in each block. If string, `"gelu"`, `"relu"`, `"selu"` and `"gelu_new"`
            are supported.
        downsample_in_first_stage (`bool`, *optional*, defaults to `False`):
            If `True`, the first stage will downsample the inputs using a `stride` of 2.

    Example:
    ```
    >>> from transformers import RegNetConfig, RegNetModel

    >>> # Initializing a RegNet regnet-y-40 style configuration

    ```
    """
    configuration = RegNetConfig()
    # 使用 RegNetConfig 类创建一个配置对象

    model = RegNetModel(configuration)
    # 使用 RegNetModel 类基于给定的配置对象创建一个模型对象

    configuration = model.config
    # 获取模型对象的配置信息
    """

    model_type = "regnet"
    # 定义模型类型为 "regnet"

    layer_types = ["x", "y"]
    # 支持的层类型列表，包括 'x' 和 'y'

    def __init__(
        self,
        num_channels=3,
        embedding_size=32,
        hidden_sizes=[128, 192, 512, 1088],
        depths=[2, 6, 12, 2],
        groups_width=64,
        layer_type="y",
        hidden_act="relu",
        **kwargs,
    ):
        # 调用父类构造函数初始化对象
        super().__init__(**kwargs)

        # 检查给定的 layer_type 是否在支持的层类型列表中，如果不在则抛出错误
        if layer_type not in self.layer_types:
            raise ValueError(f"layer_type={layer_type} is not one of {','.join(self.layer_types)}")

        # 设置对象的各个属性值
        self.num_channels = num_channels
        self.embedding_size = embedding_size
        self.hidden_sizes = hidden_sizes
        self.depths = depths
        self.groups_width = groups_width
        self.layer_type = layer_type
        self.hidden_act = hidden_act

        # 始终在第一阶段进行下采样
        self.downsample_in_first_stage = True
    ```
```