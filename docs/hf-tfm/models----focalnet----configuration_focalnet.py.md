# `.\models\focalnet\configuration_focalnet.py`

```py
# 定义字符编码为 UTF-8
# 版权声明
# 根据 Apache 授权许可证 2.0 进行许可
# 可能不允许在没有许可证的情况下使用此文件
# 可以在以下网址获得许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是在"按原样"基础上分发的，没有任何明示或暗示的担保或条件
# 请查看许可证以了解特定语言控制权限和限制
""" FocalNet 模型配置"""

# 从配置工具和日志工具导入预训练配置和日志工具
from ...configuration_utils import PretrainedConfig
from ...utils import logging
# 从骨干网络工具导入骨干网络配置混合工具和获取对齐输出特征输出索引的函数
from ...utils.backbone_utils import BackboneConfigMixin, get_aligned_output_features_output_indices

# 获取日志记录器
logger = logging.get_logger(__name__)

# 预训练配置文件存档映射
FOCALNET_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/focalnet-tiny": "https://huggingface.co/microsoft/focalnet-tiny/resolve/main/config.json",
}

# FocalNet 配置类
class FocalNetConfig(BackboneConfigMixin, PretrainedConfig):
    r"""
    这是用于存储 [`FocalNetModel`] 配置的配置类。用于根据指定的参数实例化 FocalNet 模型，定义模型架构。
    使用默认值实例化配置将产生类似于 FocalNet [microsoft/focalnet-tiny](https://huggingface.co/microsoft/focalnet-tiny) 架构的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。阅读 [`PretrainedConfig`] 的文档以获取更多信息。

    示例:

    ```python
    >>> from transformers import FocalNetConfig, FocalNetModel

    >>> # 初始化一个 FocalNet microsoft/focalnet-tiny 风格的配置
    >>> configuration = FocalNetConfig()

    >>> # 根据 microsoft/focalnet-tiny 风格的配置初始化一个模型（带有随机权重）
    >>> model = FocalNetModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```py"""

# 模型类型为 "focalnet"
    model_type = "focalnet"

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
# 省略部分代码
        # 调用父类的初始化方法，传入kwargs参数
        super().__init__(**kwargs)

        # 设置图片大小
        self.image_size = image_size
        # 设置补丁大小
        self.patch_size = patch_size
        # 设置通道数
        self.num_channels = num_channels
        # 设置嵌入维度
        self.embed_dim = embed_dim
        # 是否使用卷积嵌入
        self.use_conv_embed = use_conv_embed
        # 设置隐藏层大小
        self.hidden_sizes = hidden_sizes
        # 设置深度
        self.depths = depths
        # 设置焦点级别
        self.focal_levels = focal_levels
        # 设置焦点窗口
        self.focal_windows = focal_windows
        # 设置隐藏层激活函数
        self.hidden_act = hidden_act
        # 设置MLP比率
        self.mlp_ratio = mlp_ratio
        # 设置隐藏层丢失概率
        self.hidden_dropout_prob = hidden_dropout_prob
        # 设置丢弃路径率
        self.drop_path_rate = drop_path_rate
        # 是否使用层缩放
        self.use_layerscale = use_layerscale
        # 设置层缩放值
        self.layerscale_value = layerscale_value
        # 是否使用后层归一化
        self.use_post_layernorm = use_post_layernorm
        # 在调制中是否使用后层归一化
        self.use_post_layernorm_in_modulation = use_post_layernorm_in_modulation
        # 是否归一化调制器
        self.normalize_modulator = normalize_modulator
        # 设置初始化范围
        self.initializer_range = initializer_range
        # 设置层归一化范围
        self.layer_norm_eps = layer_norm_eps
        # 编码器步长
        self.encoder_stride = encoder_stride
        # 设置阶段名称
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, len(self.depths) + 1)]
        # 获取输出特征和输出索引
        self._out_features, self._out_indices = get_aligned_output_features_output_indices(
            out_features=out_features, out_indices=out_indices, stage_names=self.stage_names
        )
```