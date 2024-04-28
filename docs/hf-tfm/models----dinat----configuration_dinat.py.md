# `.\models\dinat\configuration_dinat.py`

```py
# 设置文件编码为 UTF-8
# 版权声明及许可信息
# 版权所有 2022 年 HuggingFace Inc. 团队保留所有权利。
# 根据 Apache 许可证 2.0 版本（“许可证”）获得许可;
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则本软件根据“原样”分发，
# 没有任何明示或暗示的保证或条件。
# 有关许可证的详细信息，请参阅许可证

# 从 HuggingFace 的 configuration_utils 模块中导入 PretrainedConfig 类
# 从 HuggingFace 的 logging 模块中导入 logging 函数
# 从 HuggingFace 的 backbone_utils 模块中导入 BackboneConfigMixin 和 get_aligned_output_features_output_indices 函数
from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ...utils.backbone_utils import BackboneConfigMixin, get_aligned_output_features_output_indices

# 获取 logger 对象，用于日志记录
logger = logging.get_logger(__name__)

# DINAT 预训练模型配置文件映射表，用于不同 DINAT 模型配置文件的快速访问
DINAT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "shi-labs/dinat-mini-in1k-224": "https://huggingface.co/shi-labs/dinat-mini-in1k-224/resolve/main/config.json",
    # 查看所有 Dinat 模型 https://huggingface.co/models?filter=dinat
}

# DinatConfig 类，继承自 BackboneConfigMixin 和 PretrainedConfig 类
class DinatConfig(BackboneConfigMixin, PretrainedConfig):
    r"""
    这是用于存储 [`DinatModel`] 配置的配置类。根据指定的参数实例化 Dinat 模型，定义模型架构。
    使用默认值实例化配置将生成与 Dinat [shi-labs/dinat-mini-in1k-224](https://huggingface.co/shi-labs/dinat-mini-in1k-224) 架构类似的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。阅读 [`PretrainedConfig`] 的文档以获取更多信息。

    示例:

    ```python
    >>> from transformers import DinatConfig, DinatModel

    >>> # 初始化一个 Dinat shi-labs/dinat-mini-in1k-224 风格的配置
    >>> configuration = DinatConfig()

    >>> # 从 shi-labs/dinat-mini-in1k-224 风格的配置初始化一个模型（具有随机权重）
    >>> model = DinatModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```py"""

    # 模型类型为 "dinat"
    model_type = "dinat"

    # 属性映射表，用于将一些属性名称映射到模型配置中的相应属性名称
    attribute_map = {
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
    }
    # 初始化函数，设置模型的参数和属性
    def __init__(
        self,
        patch_size=4,  # 补丁大小，默认为4
        num_channels=3,  # 通道数，默认为3
        embed_dim=64,  # 嵌入维度，默认为64
        depths=[3, 4, 6, 5],  # 不同阶段的层数，默认为[3, 4, 6, 5]
        num_heads=[2, 4, 8, 16],  # 不同阶段的多头注意力机制头数，默认为[2, 4, 8, 16]
        kernel_size=7,  # 卷积核大小，默认为7
        dilations=[[1, 8, 1], [1, 4, 1, 4], [1, 2, 1, 2, 1, 2], [1, 1, 1, 1, 1]],  # 不同阶段的卷积扩张率，默认为指定列表
        mlp_ratio=3.0,  # 多层感知机输出维度相对于输入维度倍率，默认为3.0
        qkv_bias=True,  # 是否在注意力计算中使用偏置，默认为True
        hidden_dropout_prob=0.0,  # 隐藏层的Dropout概率，默认为0.0
        attention_probs_dropout_prob=0.0,  # 注意力概率的Dropout概率，默认为0.0
        drop_path_rate=0.1,  # DropPath概率，默认为0.1
        hidden_act="gelu",  # 隐藏层激活函数，默认为gelu
        initializer_range=0.02,  # 参数初始化范围，默认为0.02
        layer_norm_eps=1e-5,  # Layer Normalization中的epsilon参数，默认为1e-5
        layer_scale_init_value=0.0,  # 层尺度初始化值，默认为0.0
        out_features=None,  # 输出特征，默认为None
        out_indices=None,  # 输出索引，默认为None
        **kwargs,  # 其他参数
    ):
        # 调用父类的初始化函数
        super().__init__(**kwargs)

        self.patch_size = patch_size  # 设置补丁大小
        self.num_channels = num_channels  # 设置通道数
        self.embed_dim = embed_dim  # 设置嵌入维度
        self.depths = depths  # 设置层数列表
        self.num_layers = len(depths)  # 计算层数
        self.num_heads = num_heads  # 设置多头注意力机制头数
        self.kernel_size = kernel_size  # 设置卷积核大小
        self.dilations = dilations  # 设置卷积扩张率
        self.mlp_ratio = mlp_ratio  # 设置多层感知机输出维度相对于输入维度倍率
        self.qkv_bias = qkv_bias  # 设置是否在注意力计算中使用偏置
        self.hidden_dropout_prob = hidden_dropout_prob  # 设置隐藏层的Dropout概率
        self.attention_probs_dropout_prob = attention_probs_dropout_prob  # 设置注意力概率的Dropout概率
        self.drop_path_rate = drop_path_rate  # 设置DropPath概率
        self.hidden_act = hidden_act  # 设置隐藏层激活函数
        self.layer_norm_eps = layer_norm_eps  # 设置Layer Normalization中的epsilon参数
        self.initializer_range = initializer_range  # 设置参数初始化范围
        # we set the hidden_size attribute in order to make Dinat work with VisionEncoderDecoderModel
        # this indicates the channel dimension after the last stage of the model
        self.hidden_size = int(embed_dim * 2 ** (len(depths) - 1))  # 计算最后一个阶段结束后的通道维度
        self.layer_scale_init_value = layer_scale_init_value  # 设置层尺度初始化值
        self.stage_names = ["stem"] + [f"stage{idx}" for idx in range(1, len(depths) + 1)]  # 设置阶段名称列表
        self._out_features, self._out_indices = get_aligned_output_features_output_indices(  # 调用函数获取对齐后的输出特征和输出索引
            out_features=out_features, out_indices=out_indices, stage_names=self.stage_names
        )
```