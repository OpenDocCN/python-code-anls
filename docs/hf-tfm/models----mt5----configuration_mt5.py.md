# `.\models\mt5\configuration_mt5.py`

```py
# coding=utf-8
# 设置文件编码为 UTF-8

# 版权声明，指明代码版权归 The T5 Authors 和 HuggingFace Inc. 所有

# 引入 Mapping 类型用于类型提示
from typing import Mapping

# 从相关的库中导入必要的配置类和函数
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxSeq2SeqConfigWithPast
from ...utils import logging

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# 定义 mT5 模型配置类，继承自 PretrainedConfig 类
class MT5Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MT5Model`] or a [`TFMT5Model`]. It is used to
    instantiate a mT5 model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the mT5
    [google/mt5-small](https://huggingface.co/google/mt5-small) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """
    # 定义模型类型为 "mt5"
    model_type = "mt5"
    # 在推断时忽略的关键字列表，这些关键字不参与推断过程
    keys_to_ignore_at_inference = ["past_key_values"]
    # 将属性映射为模型参数，例如将 "hidden_size" 映射为 "d_model"，"num_attention_heads" 映射为 "num_heads"，"num_hidden_layers" 映射为 "num_layers"
    attribute_map = {"hidden_size": "d_model", "num_attention_heads": "num_heads", "num_hidden_layers": "num_layers"}
    # 初始化函数，设置模型的各项参数
    def __init__(
        self,
        vocab_size=250112,  # 词汇表大小，默认为250112
        d_model=512,  # 模型维度，默认为512
        d_kv=64,  # KV维度，默认为64
        d_ff=1024,  # 前馈网络的维度，默认为1024
        num_layers=8,  # 层数，默认为8
        num_decoder_layers=None,  # 解码器层数，默认为None
        num_heads=6,  # 头数，默认为6
        relative_attention_num_buckets=32,  # 相对注意力的桶数，默认为32
        relative_attention_max_distance=128,  # 相对注意力的最大距离，默认为128
        dropout_rate=0.1,  # 丢弃率，默认为0.1
        layer_norm_epsilon=1e-6,  # 层归一化的epsilon，默认为1e-6
        initializer_factor=1.0,  # 初始化因子，默认为1.0
        feed_forward_proj="gated-gelu",  # 前馈网络激活函数，默认为"gated-gelu"
        is_encoder_decoder=True,  # 是否是编码-解码模型，默认为True
        use_cache=True,  # 是否使用缓存，默认为True
        tokenizer_class="T5Tokenizer",  # 分词器的类名，默认为"T5Tokenizer"
        tie_word_embeddings=False,  # 是否绑定词嵌入，默认为False
        pad_token_id=0,  # 填充标记的ID，默认为0
        eos_token_id=1,  # 结束标记的ID，默认为1
        decoder_start_token_id=0,  # 解码器开始标记的ID，默认为0
        classifier_dropout=0.0,  # 分类器的丢弃率，默认为0.0
        **kwargs,  # 其他参数
    ):
        # 将参数赋值给对象的属性
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.num_decoder_layers = (
            num_decoder_layers if num_decoder_layers is not None else self.num_layers
        )  # 如果解码器层数不为空，则赋值为解码器层数，否则赋值为层数
        self.num_heads = num_heads
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.dropout_rate = dropout_rate
        self.classifier_dropout = classifier_dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_factor = initializer_factor
        self.feed_forward_proj = feed_forward_proj
        self.use_cache = use_cache

        # 根据feed_forward_proj设置前馈网络激活函数相关属性
        act_info = self.feed_forward_proj.split("-")
        self.dense_act_fn = act_info[-1]
        self.is_gated_act = act_info[0] == "gated"

        # 检查前馈网络激活函数是否合法
        if len(act_info) > 1 and act_info[0] != "gated" or len(act_info) > 2:
            raise ValueError(
                f"`feed_forward_proj`: {feed_forward_proj} is not a valid activation function of the dense layer. "
                "Please make sure `feed_forward_proj` is of the format `gated-{ACT_FN}` or `{ACT_FN}`, e.g. "
                "'gated-gelu' or 'relu'"
            )

        # 为了向后兼容性，如果前馈网络激活函数为"gated-gelu"，将dense_act_fn设置为"gelu_new"
        if feed_forward_proj == "gated-gelu":
            self.dense_act_fn = "gelu_new"

        # 调用父类的初始化函数
        super().__init__(
            is_encoder_decoder=is_encoder_decoder,
            tokenizer_class=tokenizer_class,
            tie_word_embeddings=tie_word_embeddings,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            decoder_start_token_id=decoder_start_token_id,
            **kwargs,
        )
# 定义一个继承自OnnxSeq2SeqConfigWithPast的MT5OnnxConfig类
class MT5OnnxConfig(OnnxSeq2SeqConfigWithPast):
    
    @property
    # 从transformers.models.t5.configuration_t5.T5OnnxConfig.inputs中复制而来
    # 返回一个映射，将输入名称映射到索引及其含义的字典
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 定义通用的输入映射，包括input_ids和attention_mask
        common_inputs = {
            "input_ids": {0: "batch", 1: "encoder_sequence"},
            "attention_mask": {0: "batch", 1: "encoder_sequence"},
        }
        # 如果使用过去的状态（self.use_past为True）
        if self.use_past:
            # 调整attention_mask的描述以包括过去的编码器序列
            common_inputs["attention_mask"][1] = "past_encoder_sequence + sequence"
            # 添加decoder_input_ids映射
            common_inputs["decoder_input_ids"] = {0: "batch"}
            # 添加decoder_attention_mask映射，包括过去的解码器序列
            common_inputs["decoder_attention_mask"] = {0: "batch", 1: "past_decoder_sequence + sequence"}
        else:
            # 如果不使用过去的状态，设置decoder_input_ids映射
            common_inputs["decoder_input_ids"] = {0: "batch", 1: "decoder_sequence"}
            # 设置decoder_attention_mask映射，仅包括当前的解码器序列
            common_inputs["decoder_attention_mask"] = {0: "batch", 1: "decoder_sequence"}

        # 如果使用过去的状态，调用self.fill_with_past_key_values_方法填充common_inputs
        if self.use_past:
            self.fill_with_past_key_values_(common_inputs, direction="inputs")

        # 返回最终的输入映射字典
        return common_inputs

    @property
    # 从transformers.models.t5.configuration_t5.T5OnnxConfig.default_onnx_opset中复制而来
    # 返回默认的ONNX操作集版本号
    def default_onnx_opset(self) -> int:
        return 13

    @property
    # 返回用于验证的绝对误差限制
    def atol_for_validation(self) -> float:
        return 5e-4
```