# `.\models\glpn\configuration_glpn.py`

```py
# 设置文件编码为 UTF-8
# 版权声明：2022 KAIST 和 The HuggingFace Inc. 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）获得许可；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件根据“原样”分发，
# 没有任何形式的明示或暗示的担保或条件。
# 有关更多信息，请参阅许可证。

# 从配置工具中导入预训练配置类
from ...configuration_utils import PretrainedConfig
# 从工具包中导入日志记录模块
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义 GLPN 预训练模型配置文件的映射字典
GLPN_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    # "vinvino02/glpn-kitti" 模型的配置文件链接
    "vinvino02/glpn-kitti": "https://huggingface.co/vinvino02/glpn-kitti/resolve/main/config.json",
    # 查看所有 GLPN 模型的链接
    # https://huggingface.co/models?filter=glpn
}

# GLPN 预训练模型配置类，继承自 PretrainedConfig
class GLPNConfig(PretrainedConfig):
    r"""
    这是用于存储 [`GLPNModel`] 配置的配置类。它用于根据指定的参数实例化 GLPN 模型，定义模型架构。使用默认值实例化配置将产生与 GLPN
    [vinvino02/glpn-kitti](https://huggingface.co/vinvino02/glpn-kitti) 架构类似的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。有关更多信息，请阅读 [`PretrainedConfig`] 的文档。
    Args:
        num_channels (`int`, *optional*, defaults to 3):
            输入通道的数量。
        num_encoder_blocks (`int`, *optional*, defaults to 4):
            编码器块的数量（即 Mix Transformer 编码器中的阶段数）。
        depths (`List[int]`, *optional*, defaults to `[2, 2, 2, 2]`):
            每个编码器块中的层数。
        sr_ratios (`List[int]`, *optional*, defaults to `[8, 4, 2, 1]`):
            每个编码器块中的序列缩减比率。
        hidden_sizes (`List[int]`, *optional*, defaults to `[32, 64, 160, 256]`):
            每个编码器块的维度。
        patch_sizes (`List[int]`, *optional*, defaults to `[7, 3, 3, 3]`):
            每个编码器块之前的补丁大小。
        strides (`List[int]`, *optional*, defaults to `[4, 2, 2, 2]`):
            每个编码器块之前的步幅。
        num_attention_heads (`List[int]`, *optional*, defaults to `[1, 2, 5, 8]`):
            Transformer 编码器每个块中注意力层的注意力头数。
        mlp_ratios (`List[int]`, *optional*, defaults to `[4, 4, 4, 4]`):
            混合 FFN 中隐藏层大小相对于输入层大小的比率。
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            编码器和池化器中的非线性激活函数（函数或字符串）。如果是字符串，支持 `"gelu"`, `"relu"`, `"selu"` 和 `"gelu_new"`。
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            嵌入层、编码器和池化器中所有全连接层的丢失概率。
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            注意力概率的丢失比率。
        initializer_range (`float`, *optional*, defaults to 0.02):
            初始化所有权重矩阵的截断正态初始化器的标准差。
        drop_path_rate (`float`, *optional*, defaults to 0.1):
            随机深度中的丢失概率，用于 Transformer 编码器的块。
        layer_norm_eps (`float`, *optional*, defaults to 1e-06):
            层标准化层使用的 epsilon。
        decoder_hidden_size (`int`, *optional*, defaults to 64):
            解码器的维度。
        max_depth (`int`, *optional*, defaults to 10):
            解码器的最大深度。
        head_in_index (`int`, *optional*, defaults to -1):
            用于头部的特征的索引。

    Example:

    ```python
    >>> from transformers import GLPNModel, GLPNConfig

    >>> # 初始化 GLPN vinvino02/glpn-kitti 风格的配置
    >>> configuration = GLPNConfig()

    >>> # 根据 vinvino02/glpn-kitti 风格的配置初始化模型
    >>> model = GLPNModel(configuration)
    ```py
    # 访问模型配置
    configuration = model.config
    
    
    
    # 模型类型为 "glpn"
    model_type = "glpn"
    
    # GLPN 类的构造函数，接收多个参数作为输入
    def __init__(
        # 输入通道数量，默认为 3
        num_channels=3,
        # 编码器块数量，默认为 4
        num_encoder_blocks=4,
        # 每个编码器块的深度列表，默认为 [2, 2, 2, 2]
        depths=[2, 2, 2, 2],
        # 编码器块的空间尺度比例列表，默认为 [8, 4, 2, 1]
        sr_ratios=[8, 4, 2, 1],
        # 编码器块的隐藏层大小列表，默认为 [32, 64, 160, 256]
        hidden_sizes=[32, 64, 160, 256],
        # 编码器块的局部卷积核大小列表，默认为 [7, 3, 3, 3]
        patch_sizes=[7, 3, 3, 3],
        # 编码器块的步幅大小列表，默认为 [4, 2, 2, 2]
        strides=[4, 2, 2, 2],
        # 注意力头数量列表，默认为 [1, 2, 5, 8]
        num_attention_heads=[1, 2, 5, 8],
        # 多层感知机比例列表，默认为 [4, 4, 4, 4]
        mlp_ratios=[4, 4, 4, 4],
        # 隐层激活函数，默认为 "gelu"
        hidden_act="gelu",
        # 隐层dropout概率，默认为 0.0
        hidden_dropout_prob=0.0,
        # 注意力dropout概率，默认为 0.0
        attention_probs_dropout_prob=0.0,
        # 初始化范围，默认为 0.02
        initializer_range=0.02,
        # dropout路径概率，默认为 0.1
        drop_path_rate=0.1,
        # LayerNorm的epsilon，默认为 1e-6
        layer_norm_eps=1e-6,
        # 解码器隐层大小，默认为 64
        decoder_hidden_size=64,
        # 最大深度，默认为 10
        max_depth=10,
        # 输出头索引，默认为 -1
        head_in_index=-1,
        **kwargs,
    ):
        # 调用父类的构造函数
        super().__init__(**kwargs)
    
        # 初始化 GLPN 对象的属性
        self.num_channels = num_channels
        self.num_encoder_blocks = num_encoder_blocks
        self.depths = depths
        self.sr_ratios = sr_ratios
        self.hidden_sizes = hidden_sizes
        self.patch_sizes = patch_sizes
        self.strides = strides
        self.mlp_ratios = mlp_ratios
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.initializer_range = initializer_range
        self.drop_path_rate = drop_path_rate
        self.layer_norm_eps = layer_norm_eps
        self.decoder_hidden_size = decoder_hidden_size
        self.max_depth = max_depth
        self.head_in_index = head_in_index
```