# `.\models\depth_anything\configuration_depth_anything.py`

```py
# coding=utf-8
# 引入深度学习框架的配置
# 版权声明和许可证明，保留所有权利
#
# 根据 Apache 许可证 2.0 版本，只有在符合许可证条件下才能使用此文件
# 可以通过以下链接获取许可证副本
# http://www.apache.org/licenses/LICENSE-2.0
#
# 如果没有适用法律要求或书面同意，本软件是基于“现状”分发的，
# 没有任何形式的担保或条件，无论是明示的还是暗示的
# 详见许可证内容以了解更多信息
""" DepthAnything model configuration"""

import copy

# 从相关模块导入预训练配置和日志记录工具
from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto.configuration_auto import CONFIG_MAPPING

# 获取日志记录器实例
logger = logging.get_logger(__name__)

# 预训练模型配置文件映射，将预训练模型名称映射到其配置文件的 URL
DEPTH_ANYTHING_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "LiheYoung/depth-anything-small-hf": "https://huggingface.co/LiheYoung/depth-anything-small-hf/resolve/main/config.json",
}


class DepthAnythingConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DepthAnythingModel`]. It is used to instantiate an DepthAnything
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the DepthAnything
    [LiheYoung/depth-anything-small-hf](https://huggingface.co/LiheYoung/depth-anything-small-hf) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    Args:
        backbone_config (`Union[Dict[str, Any], PretrainedConfig]`, *optional*):
            The configuration of the backbone model. Only used in case `is_hybrid` is `True` or in case you want to
            leverage the [`AutoBackbone`] API.
        backbone (`str`, *optional*):
            Name of backbone to use when `backbone_config` is `None`. If `use_pretrained_backbone` is `True`, this
            will load the corresponding pretrained weights from the timm or transformers library. If `use_pretrained_backbone`
            is `False`, this loads the backbone's config and uses that to initialize the backbone with random weights.
        use_pretrained_backbone (`bool`, *optional*, defaults to `False`):
            Whether to use pretrained weights for the backbone.
        patch_size (`int`, *optional*, defaults to 14):
            The size of the patches to extract from the backbone features.
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        reassemble_hidden_size (`int`, *optional*, defaults to 384):
            The number of input channels of the reassemble layers.
        reassemble_factors (`List[int]`, *optional*, defaults to `[4, 2, 1, 0.5]`):
            The up/downsampling factors of the reassemble layers.
        neck_hidden_sizes (`List[str]`, *optional*, defaults to `[48, 96, 192, 384]`):
            The hidden sizes to project to for the feature maps of the backbone.
        fusion_hidden_size (`int`, *optional*, defaults to 64):
            The number of channels before fusion.
        head_in_index (`int`, *optional*, defaults to -1):
            The index of the features to use in the depth estimation head.
        head_hidden_size (`int`, *optional*, defaults to 32):
            The number of output channels in the second convolution of the depth estimation head.

    Example:

    ```
    >>> from transformers import DepthAnythingConfig, DepthAnythingForDepthEstimation

    >>> # Initializing a DepthAnything small style configuration
    >>> configuration = DepthAnythingConfig()

    >>> # Initializing a model from the DepthAnything small style configuration
    >>> model = DepthAnythingForDepthEstimation(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

    # Define `model_type` as "depth_anything" for identifying the model type.
    model_type = "depth_anything"

    # Constructor for initializing the DepthAnythingForDepthEstimation class.
    def __init__(
        self,
        backbone_config=None,
        backbone=None,
        use_pretrained_backbone=False,
        patch_size=14,
        initializer_range=0.02,
        reassemble_hidden_size=384,
        reassemble_factors=[4, 2, 1, 0.5],
        neck_hidden_sizes=[48, 96, 192, 384],
        fusion_hidden_size=64,
        head_in_index=-1,
        head_hidden_size=32,
        **kwargs,
    ):
        super().__init__(**kwargs)
        # 调用父类的初始化方法，传递所有关键字参数

        if use_pretrained_backbone:
            # 如果指定使用预训练的主干网络，则抛出数值错误异常
            raise ValueError("Pretrained backbones are not supported yet.")

        if backbone_config is not None and backbone is not None:
            # 如果同时指定了 `backbone` 和 `backbone_config`，则抛出数值错误异常
            raise ValueError("You can't specify both `backbone` and `backbone_config`.")

        if backbone_config is None and backbone is None:
            # 如果未指定 `backbone_config` 和 `backbone`，记录日志并使用默认的 `Dinov2` 主干网络配置进行初始化
            logger.info("`backbone_config` is `None`. Initializing the config with the default `Dinov2` backbone.")
            backbone_config = CONFIG_MAPPING["dinov2"](
                image_size=518,
                hidden_size=384,
                num_attention_heads=6,
                out_indices=[9, 10, 11, 12],
                apply_layernorm=True,
                reshape_hidden_states=False,
            )
        elif isinstance(backbone_config, dict):
            # 如果 `backbone_config` 是字典类型，则根据字典中的 `model_type` 获取对应的配置类并从字典初始化 `backbone_config`
            backbone_model_type = backbone_config.get("model_type")
            config_class = CONFIG_MAPPING[backbone_model_type]
            backbone_config = config_class.from_dict(backbone_config)

        self.backbone_config = backbone_config
        self.backbone = backbone
        self.use_pretrained_backbone = use_pretrained_backbone
        self.reassemble_hidden_size = reassemble_hidden_size
        self.patch_size = patch_size
        self.initializer_range = initializer_range
        self.reassemble_factors = reassemble_factors
        self.neck_hidden_sizes = neck_hidden_sizes
        self.fusion_hidden_size = fusion_hidden_size
        self.head_in_index = head_in_index
        self.head_hidden_size = head_hidden_size

    def to_dict(self):
        """
        将当前实例序列化为 Python 字典。重写默认的 `PretrainedConfig.to_dict` 方法。返回:
            `Dict[str, any]`: 包含此配置实例所有属性的字典,
        """
        output = copy.deepcopy(self.__dict__)

        if output["backbone_config"] is not None:
            # 如果 `backbone_config` 不为 None，则将其转换为字典形式
            output["backbone_config"] = self.backbone_config.to_dict()

        output["model_type"] = self.__class__.model_type
        return output
```