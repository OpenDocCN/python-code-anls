# `.\transformers\models\vitmatte\configuration_vitmatte.py`

```py
# 设置文件编码为 UTF-8
# 版权声明
# 根据 Apache 许可证 2.0 版本，除非符合许可证规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"AS IS"的基础分发的，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关权限和限制的具体语言
""" VitMatte 模型配置"""

# 导入必要的库
import copy
from typing import List

# 导入配置工具和日志工具
from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto.configuration_auto import CONFIG_MAPPING

# 获取日志记录器
logger = logging.get_logger(__name__)

# 预训练模型配置文件映射
VITMATTE_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "hustvl/vitmatte-small-composition-1k": "https://huggingface.co/hustvl/vitmatte-small-composition-1k/resolve/main/config.json",
}

# VitMatte 配置类，用于存储 VitMatteForImageMatting 模型的配置
class VitMatteConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of [`VitMatteForImageMatting`]. It is used to
    instantiate a ViTMatte model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the ViTMatte
    [hustvl/vitmatte-small-composition-1k](https://huggingface.co/hustvl/vitmatte-small-composition-1k) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        backbone_config (`PretrainedConfig` or `dict`, *optional*, defaults to `VitDetConfig()`):
            The configuration of the backbone model.
        hidden_size (`int`, *optional*, defaults to 384):
            The number of input channels of the decoder.
        batch_norm_eps (`float`, *optional*, defaults to 1e-05):
            The epsilon used by the batch norm layers.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        convstream_hidden_sizes (`List[int]`, *optional*, defaults to `[48, 96, 192]`):
            The output channels of the ConvStream module.
        fusion_hidden_sizes (`List[int]`, *optional*, defaults to `[256, 128, 64, 32]`):
            The output channels of the Fusion blocks.

    Example:

    ```python
    >>> from transformers import VitMatteConfig, VitMatteForImageMatting

    >>> # Initializing a ViTMatte hustvl/vitmatte-small-composition-1k style configuration
    >>> configuration = VitMatteConfig()

    >>> # Initializing a model (with random weights) from the hustvl/vitmatte-small-composition-1k style configuration
    >>> model = VitMatteForImageMatting(configuration)
    >>> # 访问模型配置
    >>> configuration = model.config
    ```py"""

    # 设置模型类型为"vitmatte"
    model_type = "vitmatte"

    def __init__(
        self,
        backbone_config: PretrainedConfig = None,
        hidden_size: int = 384,
        batch_norm_eps: float = 1e-5,
        initializer_range: float = 0.02,
        convstream_hidden_sizes: List[int] = [48, 96, 192],
        fusion_hidden_sizes: List[int] = [256, 128, 64, 32],
        **kwargs,
    ):
        # 调用父类的初始化方法
        super().__init__(**kwargs)

        # 如果没有传入backbone_config，则使用默认的VitDet配置
        if backbone_config is None:
            logger.info("`backbone_config` is `None`. Initializing the config with the default `VitDet` backbone.")
            backbone_config = CONFIG_MAPPING["vitdet"](out_features=["stage4"])
        # 如果传入的backbone_config是字典类型，则根据model_type创建对应的配置类
        elif isinstance(backbone_config, dict):
            backbone_model_type = backbone_config.get("model_type")
            config_class = CONFIG_MAPPING[backbone_model_type]
            backbone_config = config_class.from_dict(backbone_config)

        # 初始化各个属性
        self.backbone_config = backbone_config
        self.batch_norm_eps = batch_norm_eps
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.convstream_hidden_sizes = convstream_hidden_sizes
        self.fusion_hidden_sizes = fusion_hidden_sizes

    def to_dict(self):
        """
        将实例序列化为Python字典。覆盖默认的`PretrainedConfig.to_dict`方法。返回:
            `Dict[str, any]`: 包含此配置实例所有属性的字典,
        """
        # 深拷贝实例的属性
        output = copy.deepcopy(self.__dict__)
        # 将backbone_config转换为字典形式
        output["backbone_config"] = self.backbone_config.to_dict()
        output["model_type"] = self.__class__.model_type
        return output
```