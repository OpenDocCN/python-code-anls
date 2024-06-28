# `.\models\vitmatte\configuration_vitmatte.py`

```py
# 设置文件编码方式为 utf-8
# 版权声明
# 根据 Apache License, Version 2.0 许可协议，对代码的使用和分发进行限制和规定
# 可以在 http://www.apache.org/licenses/LICENSE-2.0 获取许可协议的副本
# 在适用法律要求或书面同意的情况下，按照许可协议分发的软件基于“原样”提供，没有任何明示或暗示的担保或条件
# 请查看许可协议以了解特定语言的具体限制和条件
""" VitMatte model configuration"""

# 导入所需的模块
import copy
from typing import List

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto.configuration_auto import CONFIG_MAPPING

# 获取日志记录器
logger = logging.get_logger(__name__)

# 预训练配置的压缩档映射
VITMATTE_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "hustvl/vitmatte-small-composition-1k": "https://huggingface.co/hustvl/vitmatte-small-composition-1k/resolve/main/config.json",
}

# VitMatte 配置类，用于存储 [`VitMatteForImageMatting`] 的配置，用于实例化对应架构的 VitMatte 模型
# 实例化配置使用默认值将会产生与 ViTMatte [hustvl/vitmatte-small-composition-1k] 架构相似的配置
# 配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。查阅 [`PretrainedConfig`] 的文档以获取更多信息
class VitMatteConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of [`VitMatteForImageMatting`]. It is used to
    instantiate a ViTMatte model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the ViTMatte
    [hustvl/vitmatte-small-composition-1k](https://huggingface.co/hustvl/vitmatte-small-composition-1k) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    # 设置模型类型为 ViTMatte
    model_type = "vitmatte"

    # 初始化函数，用于创建 ViTMatteForImageMatting 类的实例
    def __init__(
        self,
        backbone_config: PretrainedConfig = None,  # 可选参数，用于指定预训练的骨干网络配置
        backbone=None,  # 可选参数，当 backbone_config 为 None 时，指定要使用的骨干网络名称
        use_pretrained_backbone=False,  # 可选参数，是否使用预训练的骨干网络权重
        use_timm_backbone=False,  # 可选参数，是否从 timm 库加载骨干网络（若为 False，则从 transformers 库加载）
        backbone_kwargs=None,  # 可选参数，传递给 AutoBackbone 的关键字参数，用于从检查点加载时指定输出索引等
        hidden_size: int = 384,  # 可选参数，解码器的输入通道数
        batch_norm_eps: float = 1e-5,  # 可选参数，批归一化层使用的 epsilon 值
        initializer_range: float = 0.02,  # 可选参数，用于初始化所有权重矩阵的截断正态分布的标准差
        convstream_hidden_sizes: List[int] = [48, 96, 192],  # 可选参数，ConvStream 模块的输出通道数列表
        fusion_hidden_sizes: List[int] = [256, 128, 64, 32],  # 可选参数，Fusion 模块的输出通道数列表
        **kwargs,  # 接受额外的关键字参数
    ):
        # 调用父类的构造方法，并传递所有的关键字参数
        super().__init__(**kwargs)

        # 如果使用预训练的骨干网络，则抛出值错误异常
        if use_pretrained_backbone:
            raise ValueError("Pretrained backbones are not supported yet.")

        # 如果同时指定了 `backbone` 和 `backbone_config`，则抛出值错误异常
        if backbone_config is not None and backbone is not None:
            raise ValueError("You can't specify both `backbone` and `backbone_config`.")

        # 如果未指定 `backbone_config` 和 `backbone`，则记录警告日志并使用默认的 `VitDet` 骨干网络配置进行初始化
        if backbone_config is None and backbone is None:
            logger.info("`backbone_config` is `None`. Initializing the config with the default `VitDet` backbone.")
            backbone_config = CONFIG_MAPPING["vitdet"](out_features=["stage4"])
        # 如果 `backbone_config` 是字典类型，则根据其 `model_type` 创建相应的配置类对象
        elif isinstance(backbone_config, dict):
            backbone_model_type = backbone_config.get("model_type")
            config_class = CONFIG_MAPPING[backbone_model_type]
            backbone_config = config_class.from_dict(backbone_config)

        # 如果同时指定了 `backbone_kwargs` 和 `backbone_config`，则抛出值错误异常
        if backbone_kwargs is not None and backbone_kwargs and backbone_config is not None:
            raise ValueError("You can't specify both `backbone_kwargs` and `backbone_config`.")

        # 设置对象的各个属性
        self.backbone_config = backbone_config
        self.backbone = backbone
        self.use_pretrained_backbone = use_pretrained_backbone
        self.use_timm_backbone = use_timm_backbone
        self.backbone_kwargs = backbone_kwargs
        self.batch_norm_eps = batch_norm_eps
        self.hidden_size = hidden_size
        self.initializer_range = initializer_range
        self.convstream_hidden_sizes = convstream_hidden_sizes
        self.fusion_hidden_sizes = fusion_hidden_sizes

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`]. Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        # 深拷贝对象的所有属性
        output = copy.deepcopy(self.__dict__)
        # 将 `backbone_config` 属性转换为字典形式
        output["backbone_config"] = self.backbone_config.to_dict()
        # 添加模型类型属性到输出字典中
        output["model_type"] = self.__class__.model_type
        return output
```