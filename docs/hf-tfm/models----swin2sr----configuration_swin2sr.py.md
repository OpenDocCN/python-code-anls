# `.\models\swin2sr\configuration_swin2sr.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，版权归 HuggingFace Inc. 团队所有
#
# 根据 Apache 许可证 2.0 版本使用本文件，除非符合许可证，否则不得使用此文件
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按"原样"分发本软件
# 本软件没有任何明示或暗示的保证或条件
# 详细信息请参阅许可证

""" Swin2SR Transformer model configuration"""

# 导入预训练配置类 PretrainedConfig
from ...configuration_utils import PretrainedConfig
# 导入日志记录工具
from ...utils import logging

# 获取名为 __name__ 的日志记录器
logger = logging.get_logger(__name__)

# Swin2SR 预训练配置映射表，包含了模型名称及其配置文件的 URL
SWIN2SR_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "caidas/swin2sr-classicalsr-x2-64": (
        "https://huggingface.co/caidas/swin2sr-classicalsr-x2-64/resolve/main/config.json"
    ),
}

# Swin2SRConfig 类，用于存储 Swin2SRModel 的配置
class Swin2SRConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Swin2SRModel`]. It is used to instantiate a Swin
    Transformer v2 model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the Swin Transformer v2
    [caidas/swin2sr-classicalsr-x2-64](https://huggingface.co/caidas/swin2sr-classicalsr-x2-64) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Example:

    ```
    >>> from transformers import Swin2SRConfig, Swin2SRModel

    >>> # Initializing a Swin2SR caidas/swin2sr-classicalsr-x2-64 style configuration
    >>> configuration = Swin2SRConfig()

    >>> # Initializing a model (with random weights) from the caidas/swin2sr-classicalsr-x2-64 style configuration
    >>> model = Swin2SRModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    # 模型类型为 "swin2sr"
    model_type = "swin2sr"

    # 属性映射，将类的属性名映射到预训练模型配置中的参数名
    attribute_map = {
        "hidden_size": "embed_dim",
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
    }

    # Swin2SRConfig 类的构造函数，定义了 Swin2SR 模型的各种配置参数
    def __init__(
        self,
        image_size=64,
        patch_size=1,
        num_channels=3,
        num_channels_out=None,
        embed_dim=180,
        depths=[6, 6, 6, 6, 6, 6],
        num_heads=[6, 6, 6, 6, 6, 6],
        window_size=8,
        mlp_ratio=2.0,
        qkv_bias=True,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        drop_path_rate=0.1,
        hidden_act="gelu",
        use_absolute_embeddings=False,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        upscale=2,
        img_range=1.0,
        resi_connection="1conv",
        upsampler="pixelshuffle",
        **kwargs,
        # 调用父类的初始化方法，传入所有关键字参数
        super().__init__(**kwargs)

        # 设置模型的图像大小
        self.image_size = image_size
        # 设置每个图像块的大小
        self.patch_size = patch_size
        # 输入图像的通道数
        self.num_channels = num_channels
        # 输出图像的通道数，默认与输入通道数相同
        self.num_channels_out = num_channels if num_channels_out is None else num_channels_out
        # 嵌入维度
        self.embed_dim = embed_dim
        # 注意力层的深度列表
        self.depths = depths
        # 注意力层的数量，即深度列表的长度
        self.num_layers = len(depths)
        # 头部的数量
        self.num_heads = num_heads
        # 窗口大小
        self.window_size = window_size
        # MLP（多层感知机）扩展比例
        self.mlp_ratio = mlp_ratio
        # 是否使用查询、键、值的偏置
        self.qkv_bias = qkv_bias
        # 隐藏层的dropout率
        self.hidden_dropout_prob = hidden_dropout_prob
        # 注意力概率的dropout率
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        # 路径丢弃率
        self.drop_path_rate = drop_path_rate
        # 隐藏层的激活函数类型
        self.hidden_act = hidden_act
        # 是否使用绝对位置嵌入
        self.use_absolute_embeddings = use_absolute_embeddings
        # 层归一化的epsilon值
        self.layer_norm_eps = layer_norm_eps
        # 初始化范围
        self.initializer_range = initializer_range
        # 是否进行上采样
        self.upscale = upscale
        # 图像的像素范围
        self.img_range = img_range
        # 是否使用残差连接
        self.resi_connection = resi_connection
        # 上采样器
        self.upsampler = upsampler
```