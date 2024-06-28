# `.\models\seamless_m4t\configuration_seamless_m4t.py`

```
# 设置文件编码为 UTF-8
# 版权声明，2023 年由 HuggingFace Inc. 团队保留所有权利
#
# 根据 Apache 许可证 2.0 版本授权使用该文件
# 除非符合许可证规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按"原样"分发本软件
# 不提供任何明示或暗示的担保或条件
# 请查看许可证以获取更多详细信息
""" SeamlessM4T 模型配置"""

# 从 transformers 库导入预训练配置类 PretrainedConfig
# 从 utils 模块导入 logging 函数
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 映射预训练配置文件的 URL 地址
SEAMLESS_M4T_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/hf-seamless-m4t-medium": "https://huggingface.co/facebook/hf-seamless-m4t-medium/resolve/main/config.json",
    # 查看所有 SeamlessM4T 模型的地址：https://huggingface.co/models?filter=seamless_m4t
}

# SeamlessM4TConfig 类，继承自 PretrainedConfig 类
class SeamlessM4TConfig(PretrainedConfig):
    r"""
    这是一个配置类，用于存储 [`~SeamlessM4TModel`] 的配置。根据指定的参数实例化一个 SeamlessM4T 模型配置，
    定义模型的架构。使用默认配置实例化将产生类似于 SeamlessM4T
    ["facebook/hf-seamless-m4t-medium"](https://huggingface.co/"facebook/hf-seamless-m4t-medium") 架构的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型的输出。阅读 [`PretrainedConfig`] 的文档获取更多信息。

    ```python
    >>> from transformers import SeamlessM4TModel, SeamlessM4TConfig

    >>> # 初始化 SeamlessM4T "facebook/hf-seamless-m4t-medium" 风格的配置
    >>> configuration = SeamlessM4TConfig()

    >>> # 根据 "facebook/hf-seamless-m4t-medium" 风格的配置初始化模型
    >>> model = SeamlessM4TModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```
    """

    # 模型类型设为 "seamless_m4t"
    model_type = "seamless_m4t"
    # 初始化函数，用于创建一个新的对象实例
    def __init__(
        self,
        vocab_size=256102,  # 词汇表大小，默认为 256102
        t2u_vocab_size=10082,  # t2u 词汇表大小，默认为 10082

        # 共享配置
        hidden_size=1024,  # 隐藏层大小，默认为 1024
        initializer_range=0.02,  # 初始化范围，默认为 0.02
        layer_norm_eps=1e-5,  # 层归一化的 epsilon，默认为 1e-5
        use_cache=True,  # 是否使用缓存，默认为 True
        max_position_embeddings=1024,  # 最大位置嵌入数，默认为 1024
        is_encoder_decoder=True,  # 是否为编码器-解码器模型，默认为 True
        encoder_layerdrop=0.05,  # 编码器层的层丢弃率，默认为 0.05
        decoder_layerdrop=0.05,  # 解码器层的层丢弃率，默认为 0.05
        activation_function="relu",  # 激活函数，默认为 "relu"
        dropout=0.1,  # 普通的 dropout 率，默认为 0.1
        attention_dropout=0.1,  # 注意力 dropout 率，默认为 0.1
        activation_dropout=0.0,  # 激活函数的 dropout 率，默认为 0.0
        scale_embedding=True,  # 是否缩放嵌入，默认为 True

        # 文本编码器|解码器配置
        encoder_layers=24,  # 编码器层数，默认为 24
        encoder_ffn_dim=8192,  # 编码器 FFN 维度，默认为 8192
        encoder_attention_heads=16,  # 编码器注意力头数，默认为 16
        decoder_layers=24,  # 解码器层数，默认为 24
        decoder_ffn_dim=8192,  # 解码器 FFN 维度，默认为 8192
        decoder_attention_heads=16,  # 解码器注意力头数，默认为 16
        decoder_start_token_id=3,  # 解码器起始标记 ID，默认为 3
        max_new_tokens=256,  # 最大新标记数，默认为 256
        pad_token_id=0,  # 填充标记 ID，默认为 0
        bos_token_id=2,  # 开始标记 ID，默认为 2
        eos_token_id=3,  # 结束标记 ID，默认为 3

        # 语音编码器配置
        speech_encoder_layers=24,  # 语音编码器层数，默认为 24
        speech_encoder_attention_heads=16,  # 语音编码器注意力头数，默认为 16
        speech_encoder_intermediate_size=4096,  # 语音编码器中间层大小，默认为 4096
        speech_encoder_hidden_act="swish",  # 语音编码器隐藏层激活函数，默认为 "swish"
        speech_encoder_dropout=0.0,  # 语音编码器 dropout 率，默认为 0.0
        add_adapter=True,  # 是否添加适配器，默认为 True
        speech_encoder_layerdrop=0.1,  # 语音编码器层的层丢弃率，默认为 0.1
        feature_projection_input_dim=160,  # 特征投影输入维度，默认为 160
        num_conv_pos_embeddings=128,  # 卷积位置嵌入数，默认为 128
        num_conv_pos_embedding_groups=16,  # 卷积位置嵌入分组数，默认为 16
        adaptor_kernel_size=8,  # 适配器卷积核大小，默认为 8
        adaptor_stride=8,  # 适配器卷积步长，默认为 8
        adaptor_dropout=0.1,  # 适配器 dropout 率，默认为 0.1
        num_adapter_layers=1,  # 适配器层数，默认为 1
        position_embeddings_type="relative",  # 位置嵌入类型，默认为 "relative"
        rotary_embedding_base=10000,  # 旋转嵌入基数，默认为 10000
        max_source_positions=4096,  # 最大源位置数，默认为 4096
        conv_depthwise_kernel_size=31,  # 深度卷积核大小，默认为 31

        # t2u 配置
        t2u_bos_token_id=0,  # t2u 开始标记 ID，默认为 0
        t2u_pad_token_id=1,  # t2u 填充标记 ID，默认为 1
        t2u_eos_token_id=2,  # t2u 结束标记 ID，默认为 2
        t2u_decoder_start_token_id=2,  # t2u 解码器起始标记 ID，默认为 2
        t2u_max_new_tokens=1024,  # t2u 最大新标记数，默认为 1024
        t2u_encoder_layers=6,  # t2u 编码器层数，默认为 6
        t2u_encoder_ffn_dim=8192,  # t2u 编码器 FFN 维度，默认为 8192
        t2u_encoder_attention_heads=16,  # t2u 编码器注意力头数，默认为 16
        t2u_decoder_layers=6,  # t2u 解码器层数，默认为 6
        t2u_decoder_ffn_dim=8192,  # t2u 解码器 FFN 维度，默认为 8192
        t2u_decoder_attention_heads=16,  # t2u 解码器注意力头数，默认为 16
        t2u_max_position_embeddings=2048,  # t2u 最大位置嵌入数，默认为 2048

        # hifi-gan 语音合成器配置
        sampling_rate=16000,  # 采样率，默认为 16000
        upsample_initial_channel=512,  # 上采样初始通道数，默认为 512
        upsample_rates=[5, 4, 4, 2, 2],  # 上采样倍率列表，默认为 [5, 4, 4, 2, 2]
        upsample_kernel_sizes=[11, 8, 8, 4, 4],  # 上采样卷积核大小列表，默认为 [11, 8, 8, 4, 4]
        resblock_kernel_sizes=[3, 7, 11],  # ResBlock 卷积核大小列表，默认为 [3, 7, 11]
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],  # ResBlock 扩张大小列表，默认为 [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        leaky_relu_slope=0.1,  # Leaky ReLU 斜率，默认为 0.1

        # 特定于 Code Hifi-Gan 的配置
        unit_hifi_gan_vocab_size=10000,  # Hifi-Gan 单元词汇表大小，默认为 10000
        unit_embed_dim=1280,  # 单元嵌入维度，默认为 1280
        lang_embed_dim=256,  # 语言嵌入维度，默认为 256
        spkr_embed_dim=256,  # 说话人嵌入维度，默认为 256
        vocoder_num_langs=36,  # 语音合成器支持的语言数，默认为 36
        vocoder_num_spkrs=200,  # 语音合成器支持的说话人数，默认为 200
        variance_predictor_kernel_size=3,  # 方差预测器卷积核大小，默认为 3
        var_pred_dropout=0.5,  # 方差预测器 dropout 率，默认为 0.5
        vocoder_offset=4,  # 语音合成器偏移量，默认为 4
        **kwargs,  # 其他参数，使用字典方式接收
```