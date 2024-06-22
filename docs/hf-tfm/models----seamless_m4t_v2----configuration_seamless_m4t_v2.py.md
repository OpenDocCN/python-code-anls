# `.\transformers\models\seamless_m4t_v2\configuration_seamless_m4t_v2.py`

```py
# 导入必要的模块和函数
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义预训练模型配置文件的下载链接映射
SEAMLESS_M4T_V2_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "": "https://huggingface.co//resolve/main/config.json",
}

# 定义 SeamlessM4Tv2Config 类，用于存储 SeamlessM4Tv2 模型的配置信息
class SeamlessM4Tv2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`~SeamlessM4Tv2Model`]. It is used to instantiate
    an SeamlessM4Tv2 model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the SeamlessM4Tv2
    [""](https://huggingface.co/"") architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    # 示例代码，展示如何使用配置类和模型
    ```python
    >>> from transformers import SeamlessM4Tv2Model, SeamlessM4Tv2Config

    >>> # 初始化一个 SeamlessM4Tv2 配置
    >>> configuration = SeamlessM4Tv2Config()

    >>> # 使用配置初始化模型
    >>> model = SeamlessM4Tv2Model(configuration)

    >>> # 访问模型的配置
    >>> configuration = model.config
    ```py"""

    # 指定模型类型
    model_type = "seamless_m4t_v2"
    # 初始化函数，用于创建一个模型实例
    def __init__(
        self,
        vocab_size=256102,  # 词汇表大小，默认为256102
        t2u_vocab_size=10082,  # T2U（文本到语音）词汇表大小，默认为10082
        char_vocab_size=10943,  # 字符词汇表大小，默认为10943
    
        # 共享配置项
        hidden_size=1024,  # 隐藏层大小，默认为1024
        initializer_range=0.02,  # 初始化范围，默认为0.02
        layer_norm_eps=1e-5,  # 层归一化的 epsilon，默认为1e-5
        use_cache=True,  # 是否使用缓存，默认为True
        max_position_embeddings=4096,  # 最大位置嵌入数，默认为4096
        is_encoder_decoder=True,  # 是否为编码-解码模型，默认为True
        encoder_layerdrop=0.05,  # 编码器层随机丢弃率，默认为0.05
        decoder_layerdrop=0.05,  # 解码器层随机丢弃率，默认为0.05
        activation_function="relu",  # 激活函数类型，默认为"relu"
        dropout=0.1,  # 普通 dropout 率，默认为0.1
        attention_dropout=0.1,  # 注意力机制 dropout 率，默认为0.1
        activation_dropout=0.0,  # 激活函数 dropout 率，默认为0.0
        scale_embedding=True,  # 是否对嵌入进行缩放，默认为True
    
        # 文本编码器|解码器配置项
        encoder_layers=24,  # 编码器层数，默认为24
        encoder_ffn_dim=8192,  # 编码器中的 FFN（Feed Forward Network）维度，默认为8192
        encoder_attention_heads=16,  # 编码器中的注意力头数，默认为16
        decoder_layers=24,  # 解码器层数，默认为24
        decoder_ffn_dim=8192,  # 解码器中的 FFN 维度，默认为8192
        decoder_attention_heads=16,  # 解码器中的注意力头数，默认为16
        decoder_start_token_id=3,  # 解码器开始 token 的 id，默认为3
        max_new_tokens=256,  # 最大新 token 数，默认为256
        pad_token_id=0,  # 填充 token 的 id，默认为0
        bos_token_id=2,  # 开始 token 的 id，默认为2
        eos_token_id=3,  # 结束 token 的 id，默认为3
    
        # 语音编码器配置项
        speech_encoder_layers=24,  # 语音编码器层数，默认为24
        speech_encoder_attention_heads=16,  # 语音编码器中的注意力头数，默认为16
        speech_encoder_intermediate_size=4096,  # 语音编码器中间层的大小，默认为4096
        speech_encoder_hidden_act="swish",  # 语音编码器隐藏层激活函数，默认为"swish"
        speech_encoder_dropout=0.0,  # 语音编码器 dropout 率，默认为0.0
        add_adapter=True,  # 是否添加适配器，默认为True
        speech_encoder_layerdrop=0.1,  # 语音编码器层随机丢弃率，默认为0.1
        feature_projection_input_dim=160,  # 特征投影的输入维度，默认为160
        adaptor_kernel_size=8,  # 适配器的卷积核大小，默认为8
        adaptor_stride=8,  # 适配器的步长，默认为8
        adaptor_dropout=0.1,  # 适配器 dropout 率，默认为0.1
        num_adapter_layers=1,  # 适配器层数，默认为1
        position_embeddings_type="relative_key",  # 位置嵌入类型，默认为"relative_key"
        conv_depthwise_kernel_size=31,  # 深度卷积的卷积核大小，默认为31
        left_max_position_embeddings=64,  # 左侧最大位置嵌入数，默认为64
        right_max_position_embeddings=8,  # 右侧最大位置嵌入数，默认为8
        speech_encoder_chunk_size=20000,  # 语音编码器的块大小，默认为20000
        speech_encoder_left_chunk_num=128,  # 语音编码器左侧块的数量，默认为128
    
        # T2U 配置项
        t2u_bos_token_id=0,  # T2U 开始 token 的 id，默认为0
        t2u_pad_token_id=1,  # T2U 填充 token 的 id，默认为1
        t2u_eos_token_id=2,  # T2U 结束 token 的 id，默认为2
        t2u_encoder_layers=6,  # T2U 编码器层数，默认为6
        t2u_encoder_ffn_dim=8192,  # T2U 编码器中的 FFN 维度，默认为8192
        t2u_encoder_attention_heads=16,  # T2U 编码器中的注意力头数，默认为16
        t2u_decoder_layers=6,  # T2U 解码器层数，默认为6
        t2u_decoder_ffn_dim=8192,  # T2U 解码器中的 FFN 维度，默认为8192
        t2u_decoder_attention_heads=16,  # T2U 解码器中的注意力头数，默认为16
        t2u_max_position_embeddings=4096,  # T2U 最大位置嵌入数，默认为4096
        t2u_variance_predictor_embed_dim=1024,  # T2U 方差预测器嵌入维度，默认为1024
        t2u_variance_predictor_hidden_dim=256,  # T2U 方差预测器隐藏层维度，默认为256
        t2u_variance_predictor_kernel_size=3,  # T2U 方差预测器卷积核大小，默认为3
        t2u_variance_pred_dropout=0.5,  # T2U 方
```