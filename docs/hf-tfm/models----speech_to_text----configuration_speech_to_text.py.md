# `.\transformers\models\speech_to_text\configuration_speech_to_text.py`

```py
# 设置脚本编码为 UTF-8
# 版权声明声明版权归 HuggingFace Inc. 团队所有
#
# 在 Apache 许可证 2.0 版本下授权
# 除非符合许可证，否则不能使用此文件
# 您可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据“原样”方式分发
# 没有任何担保或条件, 明示或暗示
# 请查看许可证以获取特定语言的权限和限制

""" Speech2Text 模型配置"""

# 从配置工具中导入预训练配置类
from ...configuration_utils import PretrainedConfig
# 从工具中导入日志记录方法
from ...utils import logging

# 获取记录日志句柄
logger = logging.get_logger(__name__)

# 预训练配置归档链接
SPEECH_TO_TEXT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/s2t-small-librispeech-asr": (
        "https://huggingface.co/facebook/s2t-small-librispeech-asr/resolve/main/config.json"
    ),
    # 在 https://huggingface.co/models?filter=speech_to_text 查看所有 Speech2Text 模型
}

# Speech2Text 配置类
class Speech2TextConfig(PretrainedConfig):
    r"""
    这是配置类，用于存储 [`Speech2TextModel`] 的配置。它用于根据指定参数实例化 Speech2Text 模型，定义模型架构。使用默认值实例化配置会产生类似于 Speech2Text [facebook/s2t-small-librispeech-asr](https://huggingface.co/facebook/s2t-small-librispeech-asr) 架构的配置。

    配置对象继承自 [`PretrainedConfig`] 并可用于控制模型输出。阅读 [`PretrainedConfig`] 的文档以获取更多信息。

    示例:

    ```python
    >>> from transformers import Speech2TextConfig, Speech2TextModel

    >>> # 初始化一个 Speech2Text s2t_transformer_s 风格的配置
    >>> configuration = Speech2TextConfig()

    >>> # 从 s2t_transformer_s 风格的配置初始化一个模型（带有随机权重）
    >>> model = Speech2TextModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```py"""

    model_type = "speech_to_text"
    keys_to_ignore_at_inference = ["past_key_values"]
    attribute_map = {"num_attention_heads": "encoder_attention_heads", "hidden_size": "d_model"}
    # 初始化方法，用于创建一个新的Transformer模型实例
    def __init__(
        self,
        vocab_size=10000,  # 词汇表大小，默认为10000
        encoder_layers=12,  # 编码器层数，默认为12
        encoder_ffn_dim=2048,  # 编码器中全连接层的维度，默认为2048
        encoder_attention_heads=4,  # 编码器中注意力头的数量，默认为4
        decoder_layers=6,  # 解码器层数，默认为6
        decoder_ffn_dim=2048,  # 解码器中全连接层的维度，默认为2048
        decoder_attention_heads=4,  # 解码器中注意力头的数量，默认为4
        encoder_layerdrop=0.0,  # 编码器中层级丢弃率，默认为0.0
        decoder_layerdrop=0.0,  # 解码器中层级丢弃率，默认为0.0
        use_cache=True,  # 是否使用缓存，默认为True
        is_encoder_decoder=True,  # 是否是编码器-解码器模型，默认为True
        activation_function="relu",  # 激活函数，默认为relu
        d_model=256,  # 模型维度，默认为256
        dropout=0.1,  # Dropout率，默认为0.1
        attention_dropout=0.0,  # 注意力机制中的Dropout率，默认为0.0
        activation_dropout=0.0,  # 激活函数中的Dropout率，默认为0.0
        init_std=0.02,  # 初始化参数的标准差，默认为0.02
        decoder_start_token_id=2,  # 解码器开始标记的ID，默认为2
        scale_embedding=True,  # 是否对嵌入进行缩放，默认为True，如果是True，缩放因子为sqrt(d_model)
        pad_token_id=1,  # 填充标记的ID，默认为1
        bos_token_id=0,  # 开始标记的ID，默认为0
        eos_token_id=2,  # 结束标记的ID，默认为2
        max_source_positions=6000,  # 最大源序列长度，默认为6000
        max_target_positions=1024,  # 最大目标序列长度，默认为1024
        num_conv_layers=2,  # 卷积层数，默认为2
        conv_kernel_sizes=(5, 5),  # 卷积核大小，默认为(5, 5)
        conv_channels=1024,  # 卷积通道数，默认为1024
        input_feat_per_channel=80,  # 每个通道的输入特征数，默认为80
        input_channels=1,  # 输入通道数，默认为1
        **kwargs,  # 其他关键字参数
    ):
        # 设置实例变量值
        self.vocab_size = vocab_size
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
        self.use_cache = use_cache
        self.num_hidden_layers = encoder_layers  # 编码器层数
        self.scale_embedding = scale_embedding  # 如果为True，则嵌入的缩放因子为sqrt(d_model)
        self.max_source_positions = max_source_positions  # 最大源序列长度
        self.max_target_positions = max_target_positions  # 最大目标序列长度
        self.num_conv_layers = num_conv_layers  # 卷积层数
        self.conv_kernel_sizes = list(conv_kernel_sizes)  # 卷积核大小列表
        self.conv_channels = conv_channels  # 卷积通道数
        self.input_feat_per_channel = input_feat_per_channel  # 每个通道的输入特征数
        self.input_channels = input_channels  # 输入通道数

        # 检查卷积模块的配置是否正确
        if len(self.conv_kernel_sizes) != self.num_conv_layers:
            raise ValueError(
                "Configuration for convolutional module is incorrect. "
                "It is required that `len(config.conv_kernel_sizes)` == `config.num_conv_layers` "
                f"but is `len(config.conv_kernel_sizes) = {len(self.conv_kernel_sizes)}`, "
                f"`config.num_conv_layers = {self.num_conv_layers}`."
            )

        # 调用父类的初始化方法
        super().__init__(
            pad_token_id=pad_token_id,  # 填充标记的ID
            bos_token_id=bos_token_id,  # 开始标记的ID
            eos_token_id=eos_token_id,  # 结束标记的ID
            is_encoder_decoder=is_encoder_decoder,  # 是否是编码器-解码器模型
            decoder_start_token_id=decoder_start_token_id,  # 解码器开始标记的ID
            **kwargs,  # 其他关键字参数
        )
```