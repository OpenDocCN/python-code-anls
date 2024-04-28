# `.\transformers\models\nat\configuration_nat.py`

```
# 设置文件编码为 UTF-8
# 版权声明及许可证
# 本文件受 Apache 许可证 2.0 保护
# 请在合规情况下使用本文件
# 可以在以下获取许可证副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件均是在 "按原样" 基础上分发的，没有任何明示或暗示的保证或条件
# 请查看许可证获取更多关于许可证的信息

# 导入所需的模块和函数
# logging 模块用于记录日志
# BackboneConfigMixin 用于创建神经网络的配置
# get_aligned_output_features_output_indices 用于获取对齐的输出特征的输出索引
from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ...utils.backbone_utils import BackboneConfigMixin, get_aligned_output_features_output_indices

# 使用 logging 模块获取记录器对象
logger = logging.get_logger(__name__)

# Nat 模型的预训练配置映射
NAT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "shi-labs/nat-mini-in1k-224": "https://huggingface.co/shi-labs/nat-mini-in1k-224/resolve/main/config.json",
    # 查看所有 Nat 模型 https://huggingface.co/models?filter=nat
}

# Nat 模型的配置类
class NatConfig(BackboneConfigMixin, PretrainedConfig):
    r"""
    这是用于存储[`NatModel`]配置的配置类。用于根据指定参数实例化 Nat 模型，定义模型架构。使用默认值实例化配置将返回类似于
    [shi-labs/nat-mini-in1k-224](https://huggingface.co/shi-labs/nat-mini-in1k-224) 架构的配置。

    配置对象继承自[`PretrainedConfig`]，可用于控制模型输出。阅读[`PretrainedConfig`]的文档获取更多信息。

    示例：

    ```python
    >>> from transformers import NatConfig, NatModel

    >>> # 初始化一个 Nat shi-labs/nat-mini-in1k-224 风格的配置
    >>> configuration = NatConfig()

    >>> # 从 shi-labs/nat-mini-in1k-224 风格的配置初始化一个模型（带有随机权重）
    >>> model = NatModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```"""

    # 指定模型类型为 nat
    model_type = "nat"

    # 属性映射，将配置类的属性映射到 NatModel 的属性
    attribute_map = {
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
    }

    # 配置类的初始化方法
    def __init__(
        self,
        patch_size=4,  # 补丁大小
        num_channels=3,  # 通道数
        embed_dim=64,  # 嵌入维度
        depths=[3, 4, 6, 5],  # 深度
        num_heads=[2, 4, 8, 16],  # 注意力头数
        kernel_size=7,  # 卷积核大小
        mlp_ratio=3.0,  # 多层感知器比例
        qkv_bias=True,  # Query、Key、Value 是否使用偏置
        hidden_dropout_prob=0.0,  # 隐藏层丢弃概率
        attention_probs_dropout_prob=0.0,  # 注意力概率丢弃概率
        drop_path_rate=0.1,  # DropPath 比例
        hidden_act="gelu",  # 隐藏层激活函数
        initializer_range=0.02,  # 初始化范围
        layer_norm_eps=1e-5,  # 层归一化 epsilon
        layer_scale_init_value=0.0,  # 层比例初始化值
        out_features=None,  # 输出特征
        out_indices=None,  # 输出索引
        **kwargs,  # 其他参数
        ):
        # 调用父类的初始化函数，并传入关键字参数
        super().__init__(**kwargs)

        # 设置 patch 大小
        self.patch_size = patch_size
        # 设置通道数
        self.num_channels = num_channels
        # 设置嵌入维度
        self.embed_dim = embed_dim
        # 设置层级深度
        self.depths = depths
        # 计算层级数量
        self.num_layers = len(depths)
        # 设置头数
        self.num_heads = num_heads
        # 设置卷积核大小
        self.kernel_size = kernel_size
        # 设置 MLP 比例
        self.mlp_ratio = mlp_ratio
        # 设置 QKV 是否包含偏置
        self.qkv_bias = qkv_bias
        # 设置隐藏层的 dropout 概率
        self.hidden_dropout_prob = hidden_dropout_prob
        # 设置注意力机制中 dropout 的概率
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        # 设置 dropout 的路径概率
        self.drop_path_rate = drop_path_rate
        # 设置隐藏层激活函数
        self.hidden_act = hidden_act
        # 设置层归一化的 epsilon 值
        self.layer_norm_eps = layer_norm_eps
        # 设置初始化范围
        self.initializer_range = initializer_range
        # 根据深度和嵌入维度计算出隐藏层的通道数
        self.hidden_size = int(embed_dim * 2 ** (len(depths) - 1))
        # 设置层缩放初始化值
        self.layer_scale_init_value = layer_scale_init_value
        # 根据不同阶段的名称计算出阶段名称列表
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, len(depths) + 1)]
        
        # 获取对齐的输出特征和输出索引
        self._out_features, self._out_indices = get_aligned_output_features_output_indices(
            out_features=out_features, out_indices=out_indices, stage_names=self.stage_names
        )
```