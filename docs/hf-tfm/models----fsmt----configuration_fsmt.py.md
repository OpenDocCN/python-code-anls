# `.\models\fsmt\configuration_fsmt.py`

```
# 设置代码文件的编码格式为utf-8
# 版权声明
# 根据Apache 2.0许可证，许可使用本文件。详细信息请访问 http://www.apache.org/licenses/LICENSE-2.0
# 未经许可不得使用此文件
# 根据适用法律或书面一致规定，分发的软件是基于“按原样”提供的，没有任何担保或条件，无论是明示的还是暗示的
# 详细的许可声明请查看授权许可
""" FSMT 配置"""

# 导入所需模块和类
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# FSMT 预训练配置映射字典
FSMT_PRETRAINED_CONFIG_ARCHIVE_MAP = {}

# FSMT 解码器配置类
class DecoderConfig(PretrainedConfig):
    r"""
    Configuration class for FSMT's decoder specific things. note: this is a private helper class
    """

    model_type = "fsmt_decoder"

    # 初始化方法，设定默认参数
    def __init__(self, vocab_size=0, bos_token_id=0):
        super().__init__()
        self.vocab_size = vocab_size
        self.bos_token_id = bos_token_id

# FSMT 配置类
class FSMTConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`FSMTModel`]. It is used to instantiate a FSMT
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the FSMT
    [facebook/wmt19-en-ru](https://huggingface.co/facebook/wmt19-en-ru) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Examples:

    ```python
    >>> from transformers import FSMTConfig, FSMTModel

    >>> # Initializing a FSMT facebook/wmt19-en-ru style configuration
    >>> config = FSMTConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = FSMTModel(config)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "fsmt"
    attribute_map = {"num_attention_heads": "encoder_attention_heads", "hidden_size": "d_model"}

    # update the defaults from config file
    # 初始化 TransformerConfig 类
    def __init__(
        self,
        langs=["en", "de"],  # 支持的语言列表，默认为英语和德语
        src_vocab_size=42024,  # 源语言词汇表大小，默认为42024
        tgt_vocab_size=42024,  # 目标语言词汇表大小，默认为42024
        activation_function="relu",  # 激活函数类型，默认为ReLU
        d_model=1024,  # 模型的维度，默认为1024
        max_length=200,  # 输入序列的最大长度，默认为200
        max_position_embeddings=1024,  # 最大位置嵌入大小，默认为1024
        encoder_ffn_dim=4096,  # 编码器中前馈网络维度，默认为4096
        encoder_layers=12,  # 编码器层数，默认为12
        encoder_attention_heads=16,  # 编码器注意力头数，默认为16
        encoder_layerdrop=0.0,  # 编码器层间丢弃率，默认为0.0
        decoder_ffn_dim=4096,  # 解码器中前馈网络维度，默认为4096
        decoder_layers=12,  # 解码器层数，默认为12
        decoder_attention_heads=16,  # 解码器注意力头数，默认为16
        decoder_layerdrop=0.0,  # 解码器层间丢弃率，默认为0.0
        attention_dropout=0.0,  # 注意力机制的丢弃率，默认为0.0
        dropout=0.1,  # 一般的丢弃率，默认为0.1
        activation_dropout=0.0,  # 激活函数的丢弃率，默认为0.0
        init_std=0.02,  # 初始化参数的标准差，默认为0.02
        decoder_start_token_id=2,  # 解码器的起始标记ID，默认为2
        is_encoder_decoder=True,  # 指示是否为编码解码器，默认为True
        scale_embedding=True,  # 是否缩放嵌入，默认为True
        tie_word_embeddings=False,  # 是否共享词嵌入，默认为False
        num_beams=5,  # Beam搜索中的Beam大小，默认为5
        length_penalty=1.0,  # 长度惩罚超参，默认为1.0
        early_stopping=False,  # 是否启用早停策略，默认为False
        use_cache=True,  # 是否使用缓存，默认为True
        pad_token_id=1,  # 填充标记ID，默认为1
        bos_token_id=0,  # 起始标记ID，默认为0
        eos_token_id=2,  # 终止标记ID，默认为2
        forced_eos_token_id=2,  # 强制终止标记ID，默认为2
        **common_kwargs,  # 其他公共参数
    ):
        self.langs = langs  # 初始化支持的语言列表
        self.src_vocab_size = src_vocab_size  # 初始化源语言词汇表大小
        self.tgt_vocab_size = tgt_vocab_size  # 初始化目标语言词汇表大小
        self.d_model = d_model  # 初始化编码器和解码器的维度

        self.encoder_ffn_dim = encoder_ffn_dim  # 初始化编码器前馈网络维度
        self.encoder_layers = self.num_hidden_layers = encoder_layers  # 初始化编码器层数
        self.encoder_attention_heads = encoder_attention_heads  # 初始化编码器注意力头数
        self.encoder_layerdrop = encoder_layerdrop  # 初始化编码器层间丢弃率
        self.decoder_layerdrop = decoder_layerdrop  # 初始化解码器层间丢弃率
        self.decoder_ffn_dim = decoder_ffn_dim  # 初始化解码器前馈网络维度
        self.decoder_layers = decoder_layers  # 初始化解码器层数
        self.decoder_attention_heads = decoder_attention_heads  # 初始化解码器注意力头数
        self.max_position_embeddings = max_position_embeddings  # 初始化最大位置嵌入大小
        self.init_std = init_std  # 初始化参数的标准差
        self.activation_function = activation_function  # 初始化激活函数类型

        # 初始化解码器配置
        self.decoder = DecoderConfig(vocab_size=tgt_vocab_size, bos_token_id=eos_token_id)
        if "decoder" in common_kwargs:
            del common_kwargs["decoder"]

        self.scale_embedding = scale_embedding  # 初始化是否缩放嵌入

        # 初始化三种类型的丢弃率
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.dropout = dropout

        self.use_cache = use_cache  # 初始化是否使用缓存
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            decoder_start_token_id=decoder_start_token_id,
            is_encoder_decoder=is_encoder_decoder,
            tie_word_embeddings=tie_word_embeddings,
            forced_eos_token_id=forced_eos_token_id,
            max_length=max_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            early_stopping=early_stopping,
            **common_kwargs,
        )
```