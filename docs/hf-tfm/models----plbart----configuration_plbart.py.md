# `.\transformers\models\plbart\configuration_plbart.py`

```
# 设置文件编码为 UTF-8
# 版权声明，版权归 UCLA NLP、Facebook AI Research Team 和 HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 版本，除非符合许可证，否则不得使用此文件
# 可以在以下网址获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”分发的，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关特定语言的权限和限制
""" PLBART 模型配置"""
# 导入所需的库
from collections import OrderedDict
from typing import Mapping

# 导入预训练配置、Onnx 配置和日志记录工具
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfigWithPast
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 预训练 PLBART 模型配置存档映射
PLBART_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "uclanlp/plbart-base": "https://huggingface.co/uclanlp/plbart-base/resolve/main/config.json",
    # 查看所有 PLBART 模型：https://huggingface.co/models?filter=plbart
}

# PLBART 配置类，继承自预训练配置类
class PLBartConfig(PretrainedConfig):
    r"""
    这是用于存储 [`PLBartModel`] 配置的配置类。根据指定的参数实例化 PLBART 模型，定义模型架构。使用默认值实例化配置将产生类似于 PLBART
    [uclanlp/plbart-base](https://huggingface.co/uclanlp/plbart-base) 架构的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。阅读 [`PretrainedConfig`] 的文档以获取更多信息。

    示例：

    ```python
    >>> from transformers import PLBartConfig, PLBartModel

    >>> # 初始化一个 PLBART uclanlp/plbart-base 风格的配置
    >>> configuration = PLBartConfig()

    >>> # 从 uclanlp/plbart-base 风格的配置初始化一个模型（带有随机权重）
    >>> model = PLBartModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```"""

    model_type = "plbart"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_attention_heads": "encoder_attention_heads", "hidden_size": "d_model"}
    # 初始化 Transformer 模型的参数
    def __init__(
        self,
        vocab_size=50005,  # 词汇表大小，默认为 50005
        max_position_embeddings=1024,  # 最大位置编码长度，默认为 1024
        encoder_layers=6,  # 编码器层数，默认为 6
        encoder_ffn_dim=3072,  # 编码器中间层维度，默认为 3072
        encoder_attention_heads=12,  # 编码器注意力头数，默认为 12
        decoder_layers=6,  # 解码器层数，默认为 6
        decoder_ffn_dim=3072,  # 解码器中间层维度，默认为 3072
        decoder_attention_heads=12,  # 解码器注意力头数，默认为 12
        encoder_layerdrop=0.0,  # 编码器层丢弃率，默认为 0.0
        decoder_layerdrop=0.0,  # 解码器层丢弃率，默认为 0.0
        use_cache=True,  # 是否使用缓存，默认为 True
        is_encoder_decoder=True,  # 是否为编码解码模型，默认为 True
        activation_function="gelu",  # 激活函数，默认为 gelu
        d_model=768,  # 模型维度，默认为 768
        dropout=0.1,  # 全连接层丢弃率，默认为 0.1
        attention_dropout=0.1,  # 注意力机制丢弃率，默认为 0.1
        activation_dropout=0.0,  # 激活函数丢弃率，默认为 0.0
        init_std=0.02,  # 初始化标准差，默认为 0.02
        classifier_dropout=0.0,  # 分类器丢弃率，默认为 0.0
        scale_embedding=True,  # 是否缩放嵌入，默认为 True，如果为 True，则缩放因子为 sqrt(d_model)
        pad_token_id=1,  # 填充标记 ID，默认为 1
        bos_token_id=0,  # 起始标记 ID，默认为 0
        eos_token_id=2,  # 结束标记 ID，默认为 2
        forced_eos_token_id=2,  # 强制结束标记 ID，默认为 2
        **kwargs,  # 其他参数
    ):
        # 初始化各个参数
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.d_model = d_model
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.classifier_dropout = classifier_dropout
        self.use_cache = use_cache
        self.num_hidden_layers = encoder_layers
        self.scale_embedding = scale_embedding  # 如果为 True，则缩放因子为 sqrt(d_model)
        # 调用父类的初始化方法
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            forced_eos_token_id=forced_eos_token_id,
            **kwargs,
        )
# 定义一个继承自OnnxConfigWithPast的PLBartOnnxConfig类
class PLBartOnnxConfig(OnnxConfigWithPast):
    # 定义一个inputs属性，返回一个有序字典，包含input_ids和attention_mask两个键值对
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                ("input_ids", {0: "batch", 1: "sequence"}),
                ("attention_mask", {0: "batch", 1: "sequence"}),
            ]
        )

    # 定义一个outputs属性，根据use_past属性的值返回不同的有序字典
    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        # 如果use_past为True，则返回包含last_hidden_state、past_keys和encoder_last_hidden_state三个键值对的有序字典
        if self.use_past:
            return OrderedDict(
                [
                    ("last_hidden_state", {0: "batch", 1: "sequence"}),
                    ("past_keys", {0: "batch", 2: "sequence"}),
                    ("encoder_last_hidden_state", {0: "batch", 1: "sequence"}),
                ]
            )
        # 如果use_past为False，则返回包含last_hidden_state和encoder_last_hidden_state两个键值对的有序字典
        else:
            return OrderedDict(
                [
                    ("last_hidden_state", {0: "batch", 1: "sequence"}),
                    ("encoder_last_hidden_state", {0: "batch", 1: "sequence"}),
                ]
            )
```