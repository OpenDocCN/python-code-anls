# `.\transformers\models\timm_backbone\configuration_timm_backbone.py`

```py
# 设置文件编码
# 版权声明
# ...
# 声明 TimmBackboneConfig 类所在模块为 Backbone models
from ...configuration_utils import PretrainedConfig  # 引入预训练配置
from ...utils import logging  # 引入日志工具

# 获取 logger
logger = logging.get_logger(__name__)

# 设置 TimmBackboneConfig 类
class TimmBackboneConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration for a timm backbone [`TimmBackbone`].

    It is used to instantiate a timm backbone model according to the specified arguments, defining the model.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        backbone (`str`, *optional`):
            The timm checkpoint to load.
        num_channels (`int`, *optional`, defaults to 3):
            The number of input channels.
        features_only (`bool`, *optional`, defaults to `True`):
            Whether to output only the features or also the logits.
        use_pretrained_backbone (`bool`, *optional`, defaults to `True`):
            Whether to use a pretrained backbone.
        out_indices (`List[int]`, *optional`):
            If used as backbone, list of indices of features to output. Can be any of 0, 1, 2, etc. (depending on how
            many stages the model has). Will default to the last stage if unset.
        freeze_batch_norm_2d (`bool`, *optional`, defaults to `False`):
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
    ```py
    """

    model_type = "timm_backbone"  # 设置模型类型为 "timm_backbone"

    def __init__(
        self,
        backbone=None,  # 设置为可选参数
        num_channels=3,  # 设置默认值为 3
        features_only=True,  # 设置默认值为 True
        use_pretrained_backbone=True,  # 设置默认值为 True
        out_indices=None,  # 设置为可选参数
        freeze_batch_norm_2d=False,  # 设置默认值为 False
        **kwargs,  # 其它关键字参数
        ):
        # 调用父类的构造方法，传入kwargs参数
        super().__init__(**kwargs)
        # 初始化backbone属性
        self.backbone = backbone
        # 初始化num_channels属性
        self.num_channels = num_channels
        # 初始化features_only属性
        self.features_only = features_only
        # 初始化use_pretrained_backbone属性
        self.use_pretrained_backbone = use_pretrained_backbone
        # 初始化use_timm_backbone属性为True
        self.use_timm_backbone = True
        # 初始化out_indices属性，如果out_indices不为空则为其值，否则为(-1,)
        self.out_indices = out_indices if out_indices is not None else (-1,)
        # 初始化freeze_batch_norm_2d属性
        self.freeze_batch_norm_2d = freeze_batch_norm_2d
```