# `.\models\nat\configuration_nat.py`

```py
# 设置文件编码为UTF-8
# 版权声明和许可证信息
#
# 根据Apache许可证版本2.0授权，除非符合许可证要求，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”分发软件，
# 没有任何明示或暗示的保证或条件。
# 有关详细信息，请参阅许可证。
""" Neighborhood Attention Transformer model configuration"""

# 从transformers库导入预训练配置类
from ...configuration_utils import PretrainedConfig
# 导入日志记录工具
from ...utils import logging
# 导入BackboneConfigMixin类和get_aligned_output_features_output_indices函数
from ...utils.backbone_utils import BackboneConfigMixin, get_aligned_output_features_output_indices

# 获取logger对象
logger = logging.get_logger(__name__)

# Nat预训练模型配置文件映射表
NAT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "shi-labs/nat-mini-in1k-224": "https://huggingface.co/shi-labs/nat-mini-in1k-224/resolve/main/config.json",
    # 查看所有Nat模型：https://huggingface.co/models?filter=nat
}

# NatConfig类，继承自BackboneConfigMixin和PretrainedConfig
class NatConfig(BackboneConfigMixin, PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`NatModel`]. It is used to instantiate a Nat model
    according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Nat
    [shi-labs/nat-mini-in1k-224](https://huggingface.co/shi-labs/nat-mini-in1k-224) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Example:

    ```
    >>> from transformers import NatConfig, NatModel

    >>> # Initializing a Nat shi-labs/nat-mini-in1k-224 style configuration
    >>> configuration = NatConfig()

    >>> # Initializing a model (with random weights) from the shi-labs/nat-mini-in1k-224 style configuration
    >>> model = NatModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    # 模型类型为"nat"
    model_type = "nat"

    # 属性映射字典，用于将外部使用的属性名映射到内部配置属性名
    attribute_map = {
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
    }

    # 初始化函数，定义了NatConfig的各种参数和默认值
    def __init__(
        self,
        patch_size=4,
        num_channels=3,
        embed_dim=64,
        depths=[3, 4, 6, 5],
        num_heads=[2, 4, 8, 16],
        kernel_size=7,
        mlp_ratio=3.0,
        qkv_bias=True,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        drop_path_rate=0.1,
        hidden_act="gelu",
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        layer_scale_init_value=0.0,
        out_features=None,
        out_indices=None,
        **kwargs,
        ):
        # 调用父类的初始化方法，传入所有关键字参数
        super().__init__(**kwargs)

        # 设置模型的补丁大小
        self.patch_size = patch_size
        # 设置输入图像的通道数
        self.num_channels = num_channels
        # 设置嵌入维度
        self.embed_dim = embed_dim
        # 设置每个层级的深度列表
        self.depths = depths
        # 计算层级的数量
        self.num_layers = len(depths)
        # 设置注意力头的数量
        self.num_heads = num_heads
        # 设置注意力机制的核心尺寸
        self.kernel_size = kernel_size
        # 设置MLP扩展比率
        self.mlp_ratio = mlp_ratio
        # 设置查询、键、值是否包含偏差
        self.qkv_bias = qkv_bias
        # 设置隐藏层的dropout概率
        self.hidden_dropout_prob = hidden_dropout_prob
        # 设置注意力概率的dropout概率
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        # 设置层级的丢弃路径率
        self.drop_path_rate = drop_path_rate
        # 设置隐藏层的激活函数
        self.hidden_act = hidden_act
        # 设置层标准化的epsilon值
        self.layer_norm_eps = layer_norm_eps
        # 设置初始化范围
        self.initializer_range = initializer_range
        # 设置隐藏尺寸，以便Nat与VisionEncoderDecoderModel一起使用
        # 这指示模型最后阶段后的通道维度
        self.hidden_size = int(embed_dim * 2 ** (len(depths) - 1))
        # 设置层尺度初始化值
        self.layer_scale_init_value = layer_scale_init_value
        # 设置阶段名称列表，包括"stem"和从"stage1"到"stageN"
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, len(depths) + 1)]
        # 获取与输出特征和输出索引对齐的特征和索引
        self._out_features, self._out_indices = get_aligned_output_features_output_indices(
            out_features=out_features, out_indices=out_indices, stage_names=self.stage_names
        )
```