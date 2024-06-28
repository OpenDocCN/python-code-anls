# `.\models\seamless_m4t_v2\configuration_seamless_m4t_v2.py`

```
# 设置文件编码为 UTF-8，确保支持中文等多种字符集
# 版权声明和许可证信息，声明此代码的版权及其使用条款
# 导入所需的模块和类，包括预训练配置和日志记录工具
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取与模型配置文件相关的日志记录器
logger = logging.get_logger(__name__)

# 预训练模型配置文件映射，将模型名称映射到其预训练配置文件的 URL
SEAMLESS_M4T_V2_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "": "https://huggingface.co//resolve/main/config.json",
}

# SeamlessM4Tv2Config 类，用于存储 SeamlessM4Tv2 模型的配置信息
class SeamlessM4Tv2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`~SeamlessM4Tv2Model`]. It is used to instantiate
    an SeamlessM4Tv2 model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the SeamlessM4Tv2
    [""](https://huggingface.co/"") architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    ```python
    >>> from transformers import SeamlessM4Tv2Model, SeamlessM4Tv2Config

    >>> # Initializing a SeamlessM4Tv2 "" style configuration
    >>> configuration = SeamlessM4Tv2Config()

    >>> # Initializing a model from the "" style configuration
    >>> model = SeamlessM4Tv2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    
    model_type = "seamless_m4t_v2"  # 指定模型类型为 seamless_m4t_v2
    # 初始化方法，用于创建一个新的对象实例
    def __init__(
        self,
        vocab_size=256102,  # 词汇表大小，默认为 256102
        t2u_vocab_size=10082,  # T2U 词汇表大小，默认为 10082
        char_vocab_size=10943,  # 字符词汇表大小，默认为 10943

        # 共享配置项
        hidden_size=1024,  # 隐藏层大小，默认为 1024
        initializer_range=0.02,  # 初始化范围，默认为 0.02
        layer_norm_eps=1e-5,  # 层归一化的 epsilon，默认为 1e-5
        use_cache=True,  # 是否使用缓存，默认为 True
        max_position_embeddings=4096,  # 最大位置编码数，默认为 4096
        is_encoder_decoder=True,  # 是否是编码解码模型，默认为 True
        encoder_layerdrop=0.05,  # 编码器层随机丢弃率，默认为 0.05
        decoder_layerdrop=0.05,  # 解码器层随机丢弃率，默认为 0.05
        activation_function="relu",  # 激活函数，默认为 relu
        dropout=0.1,  # 普通 dropout 率，默认为 0.1
        attention_dropout=0.1,  # 注意力机制中的 dropout 率，默认为 0.1
        activation_dropout=0.0,  # 激活函数中的 dropout 率，默认为 0.0
        scale_embedding=True,  # 是否对嵌入进行缩放，默认为 True

        # 文本编码器|解码器配置
        encoder_layers=24,  # 编码器层数，默认为 24
        encoder_ffn_dim=8192,  # 编码器中 FFN 层的维度，默认为 8192
        encoder_attention_heads=16,  # 编码器中注意力头的数量，默认为 16
        decoder_layers=24,  # 解码器层数，默认为 24
        decoder_ffn_dim=8192,  # 解码器中 FFN 层的维度，默认为 8192
        decoder_attention_heads=16,  # 解码器中注意力头的数量，默认为 16
        decoder_start_token_id=3,  # 解码器起始 token 的 ID，默认为 3
        max_new_tokens=256,  # 最大新 token 数量，默认为 256
        pad_token_id=0,  # 填充 token 的 ID，默认为 0
        bos_token_id=2,  # 开始 token 的 ID，默认为 2
        eos_token_id=3,  # 结束 token 的 ID，默认为 3

        # 语音编码器配置
        speech_encoder_layers=24,  # 语音编码器层数，默认为 24
        speech_encoder_attention_heads=16,  # 语音编码器中注意力头的数量，默认为 16
        speech_encoder_intermediate_size=4096,  # 语音编码器中间层的大小，默认为 4096
        speech_encoder_hidden_act="swish",  # 语音编码器中隐藏层的激活函数，默认为 swish
        speech_encoder_dropout=0.0,  # 语音编码器中的 dropout 率，默认为 0.0
        add_adapter=True,  # 是否添加适配器，默认为 True
        speech_encoder_layerdrop=0.1,  # 语音编码器层随机丢弃率，默认为 0.1
        feature_projection_input_dim=160,  # 特征投影输入维度，默认为 160
        adaptor_kernel_size=8,  # 适配器卷积核大小，默认为 8
        adaptor_stride=8,  # 适配器卷积的步幅，默认为 8
        adaptor_dropout=0.1,  # 适配器的 dropout 率，默认为 0.1
        num_adapter_layers=1,  # 适配器层数，默认为 1
        position_embeddings_type="relative_key",  # 位置编码类型，默认为 relative_key
        conv_depthwise_kernel_size=31,  # 深度卷积核大小，默认为 31
        left_max_position_embeddings=64,  # 左侧最大位置编码数量，默认为 64
        right_max_position_embeddings=8,  # 右侧最大位置编码数量，默认为 8
        speech_encoder_chunk_size=20000,  # 语音编码器的分块大小，默认为 20000
        speech_encoder_left_chunk_num=128,  # 语音编码器左侧分块数量，默认为 128

        # T2U 配置
        t2u_bos_token_id=0,  # T2U 解码器起始 token 的 ID，默认为 0
        t2u_pad_token_id=1,  # T2U 填充 token 的 ID，默认为 1
        t2u_eos_token_id=2,  # T2U 结束 token 的 ID，默认为 2
        t2u_encoder_layers=6,  # T2U 编码器层数，默认为 6
        t2u_encoder_ffn_dim=8192,  # T2U 编码器中 FFN 层的维度，默认为 8192
        t2u_encoder_attention_heads=16,  # T2U 编码器中注意力头的数量，默认为 16
        t2u_decoder_layers=6,  # T2U 解码器层数，默认为 6
        t2u_decoder_ffn_dim=8192,  # T2U 解码器中 FFN 层的维度，默认为 8192
        t2u_decoder_attention_heads=16,  # T2U 解码器中注意力头的数量，默认为 16
        t2u_max_position_embeddings=4096,  # T2U 最大位置编码数，默认为 4096
        t2u_variance_predictor_embed_dim=1024,  # T2U 方差预测器嵌入维度，默认为 1024
        t2u_variance_predictor_hidden_dim=256,  # T2U 方差预测器隐藏层维度，默认为 256
        t2u_variance_predictor_kernel_size=3,  # T2U 方差预测器卷积核大小，默认为 3
        t2u_variance_pred_dropout=0.5,  # T2U 方差预测器的 dropout 率，默认为 0.5

        # Hifi-Gan 语音合成器配置
        sampling_rate=16000,  # 采样率，默认为 16000
        upsample_initial_channel=512,  # 上采样初始通道数，默认为 512
        upsample_rates=[5, 4, 4, 2, 2],  # 上采样倍率列表，默认为 [5, 4, 4, 2, 2]
        upsample_kernel_sizes=[11, 8, 8, 4, 4],  # 上采样卷积核大小列表，默认为 [11, 8, 8, 4, 4]
        resblock_kernel_sizes=[3, 7, 11],  # 残差块卷积核大小列表，默认为 [3, 7, 11]
        resblock_dilation_sizes=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],  # 残差块膨胀率列表，默认为 [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
        leaky_relu_slope=0.1,  # Leaky ReLU 斜率，默认为 0.1

        # 特定于 Code Hifi-Gan 的配置
        unit_hifi_gan_vocab_size=10000,  # Hifi-Gan 单元词汇表大小，默认为 10000
        unit_embed_dim=1280,  # 单元嵌入维度，默认为 1280
        lang_embed_dim=256,  # 语言嵌入维度，默认为 256
        spkr_embed_dim=256,  # 说话者嵌入维度，默认为 256
        vocoder_num_langs=36,  # 语音合成器支持的语言数量，默认为 36
```