# `.\models\fsmt\configuration_fsmt.py`

```py
"""
FSMT configuration
"""

from ...configuration_utils import PretrainedConfig  # 导入预训练配置类
from ...utils import logging  # 导入日志模块

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

FSMT_PRETRAINED_CONFIG_ARCHIVE_MAP = {}  # FSMT预训练模型配置存档映射为空字典

class DecoderConfig(PretrainedConfig):
    r"""
    Configuration class for FSMT's decoder specific things. note: this is a private helper class
    """
    
    model_type = "fsmt_decoder"  # 模型类型为"fsmt_decoder"

    def __init__(self, vocab_size=0, bos_token_id=0):
        super().__init__()
        self.vocab_size = vocab_size  # 初始化词汇表大小
        self.bos_token_id = bos_token_id  # 初始化起始标记ID

class FSMTConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`FSMTModel`]. It is used to instantiate a FSMT
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the FSMT
    [facebook/wmt19-en-ru](https://huggingface.co/facebook/wmt19-en-ru) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Examples:

    ```
    >>> from transformers import FSMTConfig, FSMTModel

    >>> # Initializing a FSMT facebook/wmt19-en-ru style configuration
    >>> config = FSMTConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = FSMTModel(config)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    model_type = "fsmt"  # 模型类型为"fsmt"
    attribute_map = {"num_attention_heads": "encoder_attention_heads", "hidden_size": "d_model"}

    # update the defaults from config file
    # 初始化函数，用于创建一个新的配置对象，设置各种模型参数和选项
    def __init__(
        self,
        langs=["en", "de"],  # 设置默认语言列表为英语和德语
        src_vocab_size=42024,  # 源语言词汇表大小，默认为42024
        tgt_vocab_size=42024,  # 目标语言词汇表大小，默认为42024
        activation_function="relu",  # 激活函数，默认为ReLU
        d_model=1024,  # 模型维度，同时用于编码器和解码器的嵌入维度
        max_length=200,  # 最大序列长度，默认为200
        max_position_embeddings=1024,  # 最大位置编码数，默认为1024
        encoder_ffn_dim=4096,  # 编码器中间层维度，默认为4096
        encoder_layers=12,  # 编码器层数，默认为12层
        encoder_attention_heads=16,  # 编码器注意力头数，默认为16
        encoder_layerdrop=0.0,  # 编码器层间丢弃率，默认为0.0
        decoder_ffn_dim=4096,  # 解码器中间层维度，默认为4096
        decoder_layers=12,  # 解码器层数，默认为12层
        decoder_attention_heads=16,  # 解码器注意力头数，默认为16
        decoder_layerdrop=0.0,  # 解码器层间丢弃率，默认为0.0
        attention_dropout=0.0,  # 注意力层丢弃率，默认为0.0
        dropout=0.1,  # 通用丢弃率，默认为0.1
        activation_dropout=0.0,  # 激活函数中的丢弃率，默认为0.0
        init_std=0.02,  # 初始化标准差，默认为0.02，用于参数初始化
        decoder_start_token_id=2,  # 解码器起始标记ID，默认为2
        is_encoder_decoder=True,  # 模型是否为编码解码器结构，默认为True
        scale_embedding=True,  # 是否对嵌入进行缩放，默认为True
        tie_word_embeddings=False,  # 是否绑定词嵌入，默认为False
        num_beams=5,  # Beam搜索的数量，默认为5
        length_penalty=1.0,  # 长度惩罚因子，默认为1.0
        early_stopping=False,  # 是否启用早停策略，默认为False
        use_cache=True,  # 是否使用缓存，默认为True
        pad_token_id=1,  # 填充标记ID，默认为1
        bos_token_id=0,  # 起始标记ID，默认为0
        eos_token_id=2,  # 结束标记ID，默认为2
        forced_eos_token_id=2,  # 强制结束标记ID，默认为2
        **common_kwargs,  # 其他共享关键字参数
    ):
        self.langs = langs  # 将传入的语言列表赋值给对象的langs属性
        self.src_vocab_size = src_vocab_size  # 将传入的源语言词汇表大小赋值给对象的src_vocab_size属性
        self.tgt_vocab_size = tgt_vocab_size  # 将传入的目标语言词汇表大小赋值给对象的tgt_vocab_size属性
        self.d_model = d_model  # 将传入的模型维度赋值给对象的d_model属性（编码器和解码器的嵌入维度）

        self.encoder_ffn_dim = encoder_ffn_dim  # 将传入的编码器中间层维度赋值给对象的encoder_ffn_dim属性
        self.encoder_layers = self.num_hidden_layers = encoder_layers  # 将传入的编码器层数赋值给对象的encoder_layers和num_hidden_layers属性
        self.encoder_attention_heads = encoder_attention_heads  # 将传入的编码器注意力头数赋值给对象的encoder_attention_heads属性
        self.encoder_layerdrop = encoder_layerdrop  # 将传入的编码器层间丢弃率赋值给对象的encoder_layerdrop属性
        self.decoder_layerdrop = decoder_layerdrop  # 将传入的解码器层间丢弃率赋值给对象的decoder_layerdrop属性
        self.decoder_ffn_dim = decoder_ffn_dim  # 将传入的解码器中间层维度赋值给对象的decoder_ffn_dim属性
        self.decoder_layers = decoder_layers  # 将传入的解码器层数赋值给对象的decoder_layers属性
        self.decoder_attention_heads = decoder_attention_heads  # 将传入的解码器注意力头数赋值给对象的decoder_attention_heads属性
        self.max_position_embeddings = max_position_embeddings  # 将传入的最大位置编码数赋值给对象的max_position_embeddings属性
        self.init_std = init_std  # 将传入的初始化标准差赋值给对象的init_std属性（用于参数初始化）
        self.activation_function = activation_function  # 将传入的激活函数赋值给对象的activation_function属性

        self.decoder = DecoderConfig(vocab_size=tgt_vocab_size, bos_token_id=eos_token_id)  # 创建解码器配置对象，指定词汇表大小和起始标记ID
        if "decoder" in common_kwargs:  # 如果common_kwargs中包含"decoder"键
            del common_kwargs["decoder"]  # 删除common_kwargs中的"decoder"键

        self.scale_embedding = scale_embedding  # 将传入的嵌入缩放标志赋值给对象的scale_embedding属性（如果为True，则嵌入缩放因子为sqrt(d_model)）

        self.attention_dropout = attention_dropout  # 将传入的注意力层丢弃率赋值给对象的attention_dropout属性
        self.activation_dropout = activation_dropout  # 将传入的激活函数中的丢弃率赋值给对象的activation_dropout属性
        self.dropout = dropout  # 将传入的通用丢弃率赋值给对象的dropout属性

        self.use_cache = use_cache  # 将传入的缓存使用标志赋值给对象的use_cache属性
        super().__init__(  # 调用父类的初始化方法，传入公共关键字参数和其他特定参数
            pad_token_id=pad_token_id,  # 填充标记ID
            bos_token_id=bos_token_id,  # 起始标记ID
            eos_token_id=eos_token_id,  # 结束标记ID
            decoder_start_token_id=decoder_start_token_id,  # 解码器起始标记ID
            is_encoder_decoder=is_encoder_decoder,  # 是否为编码解码器结构
            tie_word_embeddings=tie_word_embeddings,  # 是否绑定词嵌入
            forced_eos_token_id=forced_eos_token_id,  # 强制结束标记ID
            max_length=max_length,  # 最大序列长度
            num_beams=num_beams,  # Beam搜索数量
            length_penalty=length_penalty,  # 长度惩罚因子
            early_stopping=early_stopping,  # 是否启用早停策略
            **common_kwargs,  # 其他共享关键字参数
        )
```