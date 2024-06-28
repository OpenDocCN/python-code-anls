# `.\models\focalnet\configuration_focalnet.py`

```py
# coding=utf-8
# 以上为代码文件的编码声明和版权信息

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ...utils.backbone_utils import BackboneConfigMixin, get_aligned_output_features_output_indices
# 导入必要的模块和类

logger = logging.get_logger(__name__)
# 获取用于记录日志的logger对象，命名空间为当前模块

FOCALNET_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/focalnet-tiny": "https://huggingface.co/microsoft/focalnet-tiny/resolve/main/config.json",
}
# FocalNet模型预训练配置文件映射字典，指定了预训练模型及其配置文件的URL

class FocalNetConfig(BackboneConfigMixin, PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`FocalNetModel`]. It is used to instantiate a
    FocalNet model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the FocalNet
    [microsoft/focalnet-tiny](https://huggingface.co/microsoft/focalnet-tiny) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Example:

    ```
    >>> from transformers import FocalNetConfig, FocalNetModel

    >>> # Initializing a FocalNet microsoft/focalnet-tiny style configuration
    >>> configuration = FocalNetConfig()

    >>> # Initializing a model (with random weights) from the microsoft/focalnet-tiny style configuration
    >>> model = FocalNetModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """
    # FocalNetConfig类，用于存储FocalNet模型的配置信息，继承自BackboneConfigMixin和PretrainedConfig类

    model_type = "focalnet"
    # 模型类型为"focalnet"

    def __init__(
        self,
        image_size=224,
        patch_size=4,
        num_channels=3,
        embed_dim=96,
        use_conv_embed=False,
        hidden_sizes=[192, 384, 768, 768],
        depths=[2, 2, 6, 2],
        focal_levels=[2, 2, 2, 2],
        focal_windows=[3, 3, 3, 3],
        hidden_act="gelu",
        mlp_ratio=4.0,
        hidden_dropout_prob=0.0,
        drop_path_rate=0.1,
        use_layerscale=False,
        layerscale_value=1e-4,
        use_post_layernorm=False,
        use_post_layernorm_in_modulation=False,
        normalize_modulator=False,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        encoder_stride=32,
        out_features=None,
        out_indices=None,
        **kwargs,
    ):
        """
        初始化方法，用于配置FocalNet模型的各种参数和选项

        Parameters:
        - image_size (int): 输入图像的尺寸，默认为224
        - patch_size (int): 感兴趣区域（patch）的尺寸，默认为4
        - num_channels (int): 输入图像的通道数，默认为3（RGB）
        - embed_dim (int): 嵌入维度，默认为96
        - use_conv_embed (bool): 是否使用卷积进行嵌入，默认为False
        - hidden_sizes (list of int): 隐藏层的大小列表，默认为[192, 384, 768, 768]
        - depths (list of int): 各阶段的深度列表，默认为[2, 2, 6, 2]
        - focal_levels (list of int): 各阶段的聚焦级别列表，默认为[2, 2, 2, 2]
        - focal_windows (list of int): 各阶段的聚焦窗口大小列表，默认为[3, 3, 3, 3]
        - hidden_act (str): 隐藏层激活函数，默认为"gelu"
        - mlp_ratio (float): MLP扩展比例，默认为4.0
        - hidden_dropout_prob (float): 隐藏层的dropout概率，默认为0.0
        - drop_path_rate (float): drop path的概率，默认为0.1
        - use_layerscale (bool): 是否使用层标准化，默认为False
        - layerscale_value (float): 层标准化的值，默认为1e-4
        - use_post_layernorm (bool): 是否使用后层标准化，默认为False
        - use_post_layernorm_in_modulation (bool): 是否在调制中使用后层标准化，默认为False
        - normalize_modulator (bool): 是否正常化调制器，默认为False
        - initializer_range (float): 初始化范围，默认为0.02
        - layer_norm_eps (float): 层标准化的epsilon值，默认为1e-5
        - encoder_stride (int): 编码器步长，默认为32
        - out_features (None or list of int): 输出特征的索引列表，默认为None
        - out_indices (None or list of int): 输出索引的列表，默认为None
        - **kwargs: 其他参数

        Notes:
        - Parameters prefixed with 'use_' control the activation of various features in the model.
        - The defaults are set to mimic the microsoft/focalnet-tiny architecture as closely as possible.
        """
        super().__init__(**kwargs)
        # 调用父类的初始化方法，传递任意额外的关键字参数

        # 将参数存储在对象的属性中，供模型使用
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.use_conv_embed = use_conv_embed
        self.hidden_sizes = hidden_sizes
        self.depths = depths
        self.focal_levels = focal_levels
        self.focal_windows = focal_windows
        self.hidden_act = hidden_act
        self.mlp_ratio = mlp_ratio
        self.hidden_dropout_prob = hidden_dropout_prob
        self.drop_path_rate = drop_path_rate
        self.use_layerscale = use_layerscale
        self.layerscale_value = layerscale_value
        self.use_post_layernorm = use_post_layernorm
        self.use_post_layernorm_in_modulation = use_post_layernorm_in_modulation
        self.normalize_modulator = normalize_modulator
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.encoder_stride = encoder_stride
        self.out_features = out_features
        self.out_indices = out_indices
        # 初始化并设置模型配置参数的默认值和选项
        ):
            # 调用父类的初始化方法，传递所有关键字参数
            super().__init__(**kwargs)

            # 设置图像大小
            self.image_size = image_size
            # 设置补丁大小
            self.patch_size = patch_size
            # 设置通道数
            self.num_channels = num_channels
            # 设置嵌入维度
            self.embed_dim = embed_dim
            # 是否使用卷积进行嵌入
            self.use_conv_embed = use_conv_embed
            # 隐藏层大小列表
            self.hidden_sizes = hidden_sizes
            # 网络深度列表
            self.depths = depths
            # 注意力头数目
            self.focal_levels = focal_levels
            # 注意力窗口大小
            self.focal_windows = focal_windows
            # 隐藏层激活函数
            self.hidden_act = hidden_act
            # MLP比例
            self.mlp_ratio = mlp_ratio
            # 隐藏层dropout概率
            self.hidden_dropout_prob = hidden_dropout_prob
            # 路径丢弃率
            self.drop_path_rate = drop_path_rate
            # 是否使用层标准化
            self.use_layerscale = use_layerscale
            # 层标准化值
            self.layerscale_value = layerscale_value
            # 是否在模块中使用后层标准化
            self.use_post_layernorm = use_post_layernorm
            # 在调制中是否使用后层标准化
            self.use_post_layernorm_in_modulation = use_post_layernorm_in_modulation
            # 标准化调制器
            self.normalize_modulator = normalize_modulator
            # 初始化范围
            self.initializer_range = initializer_range
            # 层归一化epsilon
            self.layer_norm_eps = layer_norm_eps
            # 编码器步长
            self.encoder_stride = encoder_stride
            # 舞台名称列表，包括“stem”和各阶段的名称
            self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, len(self.depths) + 1)]
            # 获取对齐的输出特征和输出索引
            self._out_features, self._out_indices = get_aligned_output_features_output_indices(
                out_features=out_features, out_indices=out_indices, stage_names=self.stage_names
            )
```