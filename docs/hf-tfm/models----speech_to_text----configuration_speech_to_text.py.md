# `.\models\speech_to_text\configuration_speech_to_text.py`

```
# 定义了 Speech2Text 模型的配置类 Speech2TextConfig，继承自 PretrainedConfig
class Speech2TextConfig(PretrainedConfig):
    # 类的文档字符串，描述了 Speech2TextConfig 的作用和用法
    r"""
    This is the configuration class to store the configuration of a [`Speech2TextModel`]. It is used to instantiate a
    Speech2Text model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the Speech2Text
    [facebook/s2t-small-librispeech-asr](https://huggingface.co/facebook/s2t-small-librispeech-asr) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Example:

    ```python
    >>> from transformers import Speech2TextConfig, Speech2TextModel

    >>> # Initializing a Speech2Text s2t_transformer_s style configuration
    >>> configuration = Speech2TextConfig()

    >>> # Initializing a model (with random weights) from the s2t_transformer_s style configuration
    >>> model = Speech2TextModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    # 模型类型为 speech_to_text
    model_type = "speech_to_text"
    # 推断过程中忽略的键列表，这里包含 "past_key_values"
    keys_to_ignore_at_inference = ["past_key_values"]
    # 属性映射字典，将 num_attention_heads 映射为 encoder_attention_heads，hidden_size 映射为 d_model
    attribute_map = {"num_attention_heads": "encoder_attention_heads", "hidden_size": "d_model"}
    # 初始化方法，用于设置模型的各种参数和配置
    def __init__(
        self,
        vocab_size=10000,  # 词汇表大小，默认为10000
        encoder_layers=12,  # 编码器层数，默认为12层
        encoder_ffn_dim=2048,  # 编码器中全连接层的维度，默认为2048
        encoder_attention_heads=4,  # 编码器中注意力头的数量，默认为4个
        decoder_layers=6,  # 解码器层数，默认为6层
        decoder_ffn_dim=2048,  # 解码器中全连接层的维度，默认为2048
        decoder_attention_heads=4,  # 解码器中注意力头的数量，默认为4个
        encoder_layerdrop=0.0,  # 编码器层的随机丢弃比例，默认为0.0（不丢弃）
        decoder_layerdrop=0.0,  # 解码器层的随机丢弃比例，默认为0.0（不丢弃）
        use_cache=True,  # 是否使用缓存，默认为True
        is_encoder_decoder=True,  # 是否为编码解码模型，默认为True
        activation_function="relu",  # 激活函数类型，默认为ReLU
        d_model=256,  # 模型的维度，默认为256
        dropout=0.1,  # 全局的Dropout比例，默认为0.1
        attention_dropout=0.0,  # 注意力机制的Dropout比例，默认为0.0（不丢弃）
        activation_dropout=0.0,  # 激活函数的Dropout比例，默认为0.0（不丢弃）
        init_std=0.02,  # 初始化参数的标准差，默认为0.02
        decoder_start_token_id=2,  # 解码器起始标记的ID，默认为2
        scale_embedding=True,  # 是否对嵌入进行缩放，默认为True
        pad_token_id=1,  # 填充标记的ID，默认为1
        bos_token_id=0,  # 开始标记的ID，默认为0
        eos_token_id=2,  # 结束标记的ID，默认为2
        max_source_positions=6000,  # 最大源序列长度，默认为6000
        max_target_positions=1024,  # 最大目标序列长度，默认为1024
        num_conv_layers=2,  # 卷积层的数量，默认为2
        conv_kernel_sizes=(5, 5),  # 卷积核大小的元组，默认为(5, 5)
        conv_channels=1024,  # 卷积通道数，默认为1024
        input_feat_per_channel=80,  # 每个通道的输入特征数，默认为80
        input_channels=1,  # 输入通道数，默认为1
        **kwargs,  # 其他参数，用于传递给父类初始化函数
    ):
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
        self.num_hidden_layers = encoder_layers  # 隐藏层的数量等同于编码器层数
        self.scale_embedding = scale_embedding  # 如果为True，则嵌入向量将按sqrt(d_model)进行缩放
        self.max_source_positions = max_source_positions
        self.max_target_positions = max_target_positions
        self.num_conv_layers = num_conv_layers
        self.conv_kernel_sizes = list(conv_kernel_sizes)  # 将卷积核大小转换为列表形式
        self.conv_channels = conv_channels
        self.input_feat_per_channel = input_feat_per_channel
        self.input_channels = input_channels

        # 检查卷积模块的配置是否正确
        if len(self.conv_kernel_sizes) != self.num_conv_layers:
            raise ValueError(
                "Configuration for convolutional module is incorrect. "
                "It is required that `len(config.conv_kernel_sizes)` == `config.num_conv_layers` "
                f"but is `len(config.conv_kernel_sizes) = {len(self.conv_kernel_sizes)}`, "
                f"`config.num_conv_layers = {self.num_conv_layers}`."
            )

        # 调用父类的初始化方法，传递必要的参数和关键字参数
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            decoder_start_token_id=decoder_start_token_id,
            **kwargs,
        )
```