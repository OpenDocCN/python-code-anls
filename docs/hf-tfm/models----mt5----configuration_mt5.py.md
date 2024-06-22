# `.\transformers\models\mt5\configuration_mt5.py`

```py
# 设置文件编码为utf-8
# 版权声明
# 遵循Apache License, Version 2.0的许可协议
# 可在http://www.apache.org/licenses/LICENSE-2.0获取许可协议的副本
# 除非适用法律要求或书面同意，否则按"原样"分发软件
# 没有明示或暗示的任何担保或条件。参见许可协议以获取具体的语言规定和限制
""" mT5模型配置"""
# 引入必要的包
from typing import Mapping
# 引入父类PretrainedConfig
from ...configuration_utils import PretrainedConfig
# 引入OnnxSeq2SeqConfigWithPast
from ...onnx import OnnxSeq2SeqConfigWithPast
# 引入logging
from ...utils import logging

# 获取logger对象
logger = logging.get_logger(__name__)

# 创建MT5Config类，继承自PretrainedConfig类
class MT5Config(PretrainedConfig):
    r"""
    这是用于存储[`MT5Model`]或[`TFMT5Model`]的配置的配置类。它用于根据指定的参数实例化mT5模型，
    定义模型架构。使用默认值实例化配置将生成类似于mT5[google/mt5-small](https://huggingface.co/google/mt5-small) 
    架构的配置。

    配置对象继承自[`PretrainedConfig`]，可用于控制模型输出。阅读 [`PretrainedConfig`] 的文档以获取更多信息。
    # 这是 T5 模型的配置参数定义
    Arguments:
        # 词汇表的大小,定义输入 inputs_ids 可以表示的不同标记(token)数量
        vocab_size (`int`, *optional*, defaults to 250112):
            Vocabulary size of the T5 model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`T5Model`] or [`TFT5Model`].
        # 编码器和池化层的维度大小
        d_model (`int`, *optional*, defaults to 512):
            Size of the encoder layers and the pooler layer.
        # 每个注意力头的键、查询、值的向量维度大小,通常应该等于 d_model // num_heads,但在 mt5-small 中不等于此
        d_kv (`int`, *optional*, defaults to 64):
            Size of the key, query, value projections per attention head. In the conventional context, it is typically expected that `d_kv` has to be equal to `d_model // num_heads`.
            But in the architecture of mt5-small, `d_kv` is not equal to `d_model //num_heads`. The `inner_dim` of the projection layer will be defined as `num_heads * d_kv`.
        # 每个 T5Block 中前馈子层的维度大小
        d_ff (`int`, *optional*, defaults to 1024):
            Size of the intermediate feed forward layer in each `T5Block`.
        # 编码器中隐藏层的数量
        num_layers (`int`, *optional*, defaults to 8):
            Number of hidden layers in the Transformer encoder.
        # 解码器中隐藏层的数量,如果未设置则与编码器一致
        num_decoder_layers (`int`, *optional*):
            Number of hidden layers in the Transformer decoder. Will use the same value as `num_layers` if not set.
        # 每个注意力层中注意力头的数量
        num_heads (`int`, *optional*, defaults to 6):
            Number of attention heads for each attention layer in the Transformer encoder.
        # 相对注意力中使用的存储桶的数量
        relative_attention_num_buckets (`int`, *optional*, defaults to 32):
            The number of buckets to use for each attention layer.
        # 相对注意力中较长序列的最大距离
        relative_attention_max_distance (`int`, *optional*, defaults to 128):
            The maximum distance of the longer sequences for the bucket separation.
        # 所有dropout层的比例
        dropout_rate (`float`, *optional*, defaults to 0.1):
            The ratio for all dropout layers.
        # 分类器的dropout比例
        classifier_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for classifier.
        # 层归一化的epsilon参数
        layer_norm_eps (`float`, *optional*, defaults to 1e-6):
            The epsilon used by the layer normalization layers.
        # 权重矩阵初始化的因子
        initializer_factor (`float`, *optional*, defaults to 1):
            A factor for initializing all weight matrices (should be kept to 1, used internally for initialization
            testing).
        # 前馈层的类型,可选"relu"或"gated-gelu"
        feed_forward_proj (`string`, *optional*, defaults to `"gated-gelu"`):
            Type of feed forward layer to be used. Should be one of `"relu"` or `"gated-gelu"`.
        # 是否使用缓存,用于提高推理速度
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
    
    # 模型类型为 "mt5"
    model_type = "mt5"
    # 在推理时需要忽略的键
    keys_to_ignore_at_inference = ["past_key_values"]
    # 初始化函数，设置模型参数和超参数
    def __init__(
        self,
        vocab_size=250112,  # 词汇表大小，默认为250112
        d_model=512,  # 模型维度，默认为512
        d_kv=64,  # Query和Key的维度，默认为64
        d_ff=1024,  # 前馈网络中间层的维度，默认为1024
        num_layers=8,  # 编码器和解码器层的数量，默认为8
        num_decoder_layers=None,  # 解码器层的数量，默认为None
        num_heads=6,  # 注意力头的数量，默认为6
        relative_attention_num_buckets=32,  # 相对位置编码中桶的数量，默认为32
        relative_attention_max_distance=128,  # 相对位置编码中最大距离，默认为128
        dropout_rate=0.1,  # Dropout比例，默认为0.1
        layer_norm_epsilon=1e-6,  # Layer Normalization的epsilon，默认为1e-6
        initializer_factor=1.0,  # 初始化因子，默认为1.0
        feed_forward_proj="gated-gelu",  # 前馈网络激活函数，默认为"gated-gelu"
        is_encoder_decoder=True,  # 是否为编码器-解码器结构，默认为True
        use_cache=True,  # 是否使用缓存，默认为True
        tokenizer_class="T5Tokenizer",  # 分词器类名，默认为"T5Tokenizer"
        tie_word_embeddings=False,  # 是否共享词嵌入，默认为False
        pad_token_id=0,  # 填充词的token id，默认为0
        eos_token_id=1,  # 结束词的token id，默认为1
        decoder_start_token_id=0,  # 解码器开始词的token id，默认为0
        classifier_dropout=0.0,  # 分类器的Dropout比例，默认为0.0
        **kwargs,  # 其他超参数
    ):
        # 调用父类的初始化函数
        super().__init__(
            is_encoder_decoder=is_encoder_decoder,  # 是否为编码器-解码器结构
            tokenizer_class=tokenizer_class,  # 分词器类名
            tie_word_embeddings=tie_word_embeddings,  # 是否共享词嵌入
            pad_token_id=pad_token_id,  # 填充词的token id
            eos_token_id=eos_token_id,  # 结束词的token id
            decoder_start_token_id=decoder_start_token_id,  # 解码器开始词的token id
            **kwargs,  # 其他超参数
        )
        # 设置模型参数和超参数
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.num_decoder_layers = (
            num_decoder_layers if num_decoder_layers is not None else self.num_layers
        )  # 解码器层数，默认与编码器层数一致
        self.num_heads = num_heads
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.dropout_rate = dropout_rate
        self.classifier_dropout = classifier_dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_factor = initializer_factor
        self.feed_forward_proj = feed_forward_proj
        self.use_cache = use_cache

        act_info = self.feed_forward_proj.split("-")  # 拆分前馈网络激活函数信息
        self.dense_act_fn = act_info[-1]  # 密集层激活函数
        self.is_gated_act = act_info[0] == "gated"  # 是否是门控激活函数

        # 检查前馈网络激活函数的格式
        if len(act_info) > 1 and act_info[0] != "gated" or len(act_info) > 2:
            raise ValueError(
                f"`feed_forward_proj`: {feed_forward_proj} is not a valid activation function of the dense layer. "
                "Please make sure `feed_forward_proj` is of the format `gated-{ACT_FN}` or `{ACT_FN}`, e.g. "
                "'gated-gelu' or 'relu'"
            )

        # 兼容旧版本，如果前馈网络激活函数为'gated-gelu'，则设置密集层激活函数为'gelu_new'
        if feed_forward_proj == "gated-gelu":
            self.dense_act_fn = "gelu_new"

    # 返回隐藏层维度属性
    @property
    def hidden_size(self):
        return self.d_model

    # 返回注意力头的数量属性
    @property
    def num_attention_heads(self):
        return self.num_heads

    # 返回隐藏层数量属性
    @property
    def num_hidden_layers(self):
        return self.num_layers
# 该类是一个 ONNX 模型配置类，继承自 OnnxSeq2SeqConfigWithPast
class MT5OnnxConfig(OnnxSeq2SeqConfigWithPast):
    # 定义输入的映射关系
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 定义一些常见的输入，包括 input_ids 和 attention_mask
        common_inputs = {
            "input_ids": {0: "batch", 1: "encoder_sequence"},
            "attention_mask": {0: "batch", 1: "encoder_sequence"},
        }
        # 如果使用 past 特性，则需要添加额外的输入
        if self.use_past:
            common_inputs["attention_mask"][1] = "past_encoder_sequence + sequence"
            common_inputs["decoder_input_ids"] = {0: "batch"}
            common_inputs["decoder_attention_mask"] = {0: "batch", 1: "past_decoder_sequence + sequence"}
        # 如果不使用 past 特性，则需要添加 decoder_input_ids 和 decoder_attention_mask
        else:
            common_inputs["decoder_input_ids"] = {0: "batch", 1: "decoder_sequence"}
            common_inputs["decoder_attention_mask"] = {0: "batch", 1: "decoder_sequence"}
        
        # 如果使用 past 特性，则需要填充 past_key_values 信息
        if self.use_past:
            self.fill_with_past_key_values_(common_inputs, direction="inputs")

        return common_inputs

    # 定义默认的 ONNX 操作集版本为 13
    @property
    def default_onnx_opset(self) -> int:
        return 13

    # 定义 validation 时的容忍误差
    @property
    def atol_for_validation(self) -> float:
        return 5e-4
```