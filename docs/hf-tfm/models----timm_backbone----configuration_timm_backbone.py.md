# `.\models\timm_backbone\configuration_timm_backbone.py`

```
# coding=utf-8
# 版权声明及许可信息，指明代码版权归 HuggingFace Inc. 团队所有，遵循 Apache License 2.0
#
# 导入必要的模块和类
from ...configuration_utils import PretrainedConfig  # 导入预训练模型配置类
from ...utils import logging  # 导入日志记录工具

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# TimmBackboneConfig 类，用于配置 timm backbone 模型
class TimmBackboneConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration for a timm backbone [`TimmBackbone`].

    It is used to instantiate a timm backbone model according to the specified arguments, defining the model.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        backbone (`str`, *optional*):
            The timm checkpoint to load.
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        features_only (`bool`, *optional*, defaults to `True`):
            Whether to output only the features or also the logits.
        use_pretrained_backbone (`bool`, *optional*, defaults to `True`):
            Whether to use a pretrained backbone.
        out_indices (`List[int]`, *optional*):
            If used as backbone, list of indices of features to output. Can be any of 0, 1, 2, etc. (depending on how
            many stages the model has). Will default to the last stage if unset.
        freeze_batch_norm_2d (`bool`, *optional*, defaults to `False`):
            Converts all `BatchNorm2d` and `SyncBatchNorm` layers of provided module into `FrozenBatchNorm2d`.

    Example:
    ```python
    >>> from transformers import TimmBackboneConfig, TimmBackbone

    >>> # Initializing a timm backbone
    >>> configuration = TimmBackboneConfig("resnet50")

    >>> # Initializing a model from the configuration
    >>> model = TimmBackbone(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "timm_backbone"

    def __init__(
        self,
        backbone=None,
        num_channels=3,
        features_only=True,
        use_pretrained_backbone=True,
        out_indices=None,
        freeze_batch_norm_2d=False,
        **kwargs,
    ):
        # 调用父类 PretrainedConfig 的初始化方法，传递参数以初始化模型配置
        super().__init__(**kwargs)
        # 设置当前实例的特定属性，用于配置 timm backbone 模型的参数
        self.backbone = backbone
        self.num_channels = num_channels
        self.features_only = features_only
        self.use_pretrained_backbone = use_pretrained_backbone
        self.out_indices = out_indices
        self.freeze_batch_norm_2d = freeze_batch_norm_2d
        ):
            # 调用父类的初始化方法，传递所有关键字参数
            super().__init__(**kwargs)
            # 设置网络的主干模型
            self.backbone = backbone
            # 设置网络的通道数
            self.num_channels = num_channels
            # 设置是否仅输出特征
            self.features_only = features_only
            # 设置是否使用预训练的主干模型
            self.use_pretrained_backbone = use_pretrained_backbone
            # 设置是否使用timm的主干模型
            self.use_timm_backbone = True
            # 设置要输出的特征索引，如果未指定则默认为最后一个索引
            self.out_indices = out_indices if out_indices is not None else (-1,)
            # 设置是否冻结2D批处理规范化层
            self.freeze_batch_norm_2d = freeze_batch_norm_2d
```