# `.\transformers\models\t5\configuration_t5.py`

```
# 设置编码格式为 utf-8
# 版权声明和许可证信息
# T5 模型的配置类
from typing import Mapping  # 导入 Mapping 类型

from ...configuration_utils import PretrainedConfig  # 从...路径导入 PretrainedConfig 类
from ...onnx import OnnxSeq2SeqConfigWithPast  # 从...路径导入 OnnxSeq2SeqConfigWithPast 类
from ...utils import logging  # 从...路径导入 logging 模块

# 获取 logger 实例
logger = logging.get_logger(__name__)

# T5 预训练模型配置文件的地址映射
T5_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "t5-small": "https://huggingface.co/t5-small/resolve/main/config.json",
    "t5-base": "https://huggingface.co/t5-base/resolve/main/config.json",
    "t5-large": "https://huggingface.co/t5-large/resolve/main/config.json",
    "t5-3b": "https://huggingface.co/t5-3b/resolve/main/config.json",
    "t5-11b": "https://huggingface.co/t5-11b/resolve/main/config.json",
}

# T5 配置类，用于存储 T5 模型的配置信息
class T5Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`T5Model`] or a [`TFT5Model`]. It is used to
    instantiate a T5 model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the T5
    [t5-small](https://huggingface.co/t5-small) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """
    # 定义了 T5 模型的参数配置，以下是各个参数的作用和默认取值
    
    Arguments:
        vocab_size (`int`, *optional*, defaults to 32128):
            T5 模型的词汇表大小，定义了在调用 `T5Model` 或 `TFT5Model` 时可以表示的不同 token 的数量。
        d_model (`int`, *optional*, defaults to 512):
            编码器层和池化层的大小。
        d_kv (`int`, *optional*, defaults to 64):
            每个注意力头中的 key、query、value 投影的大小。投影层的 `inner_dim` 由 `num_heads * d_kv` 定义。
        d_ff (`int`, *optional*, defaults to 2048):
            每个 `T5Block` 中中间前馈层的大小。
        num_layers (`int`, *optional*, defaults to 6):
            Transformer 编码器中的隐藏层数量。
        num_decoder_layers (`int`, *optional*):
            Transformer 解码器中的隐藏层数量。如果未设置将使用与 `num_layers` 相同的值。
        num_heads (`int`, *optional*, defaults to 8):
            Transformer 编码器中每个注意力层的注意力头数量。
        relative_attention_num_buckets (`int`, *optional*, defaults to 32):
            每个注意力层中使用的桶的数量。
        relative_attention_max_distance (`int`, *optional*, defaults to 128):
            用于桶分隔的更长序列的最大距离。
        dropout_rate (`float`, *optional*, defaults to 0.1):
            所有 dropout 层的比例。
        classifier_dropout (`float`, *optional*, defaults to 0.0):
            分类器的 dropout 比例。
        layer_norm_eps (`float`, *optional*, defaults to 1e-6):
            层归一化层使用的 epsilon 值。
        initializer_factor (`float`, *optional*, defaults to 1):
            用于初始化所有权重矩阵的因子（应保持为 1，用于内部初始化测试）。
        feed_forward_proj (`string`, *optional*, defaults to `"relu"`):
            要使用的前馈层类型。应为 `"relu"` 或 `"gated-gelu"` 之一。T5v1.1 使用 `"gated-gelu"` 前向投影。原始 T5 使用 `"relu"`。
        use_cache (`bool`, *optional*, defaults to `True`):
            模型是否应返回最后一个键/值注意力（并非所有模型都使用）。
    """
    
    # 定义 T5 模型类型为 "t5"
    model_type = "t5"
    # 在推理时忽略的键
    keys_to_ignore_at_inference = ["past_key_values"]
    # 属性映射，将 "hidden_size" 映射为 "d_model"，将 "num_attention_heads" 映射为 "num_heads"，将 "num_hidden_layers" 映射为 "num_layers"
    attribute_map = {"hidden_size": "d_model", "num_attention_heads": "num_heads", "num_hidden_layers": "num_layers"}
    # Transformer 模型的初始化方法，设置模型的各种参数和属性
    def __init__(
        self,
        vocab_size=32128,  # 词汇表大小，默认为 32128
        d_model=512,  # 模型维度，默认为 512
        d_kv=64,  # 键值映射的维度，默认为 64
        d_ff=2048,  # 前馈网络中间层的维度，默认为 2048
        num_layers=6,  # Transformer 层的数量，默认为 6
        num_decoder_layers=None,  # 解码器层数，默认与编码器层数相同
        num_heads=8,  # 注意力头的数量，默认为 8
        relative_attention_num_buckets=32,  # 相对注意力机制的桶数，默认为 32
        relative_attention_max_distance=128,  # 相对注意力机制的最大距离，默认为 128
        dropout_rate=0.1,  # 丢弃率，默认为 0.1
        layer_norm_epsilon=1e-6,  # Layer normalization 的 epsilon 值，默认为 1e-6
        initializer_factor=1.0,  # 初始化因子，默认为 1.0
        feed_forward_proj="relu",  # 前馈网络激活函数，默认为 relu
        is_encoder_decoder=True,  # 是否是编码器-解码器结构，默认为 True
        use_cache=True,  # 是否使用缓存，默认为 True
        pad_token_id=0,  # 填充标记的 ID，默认为 0
        eos_token_id=1,  # 结束标记的 ID，默认为 1
        classifier_dropout=0.0,  # 分类器的丢弃率，默认为 0.0
        **kwargs,
    ):
        # 设置模型的各个参数
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.num_layers = num_layers
        self.num_decoder_layers = (
            num_decoder_layers if num_decoder_layers is not None else self.num_layers
        )  # 默认等于编码器层数
        self.num_heads = num_heads
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.dropout_rate = dropout_rate
        self.classifier_dropout = classifier_dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_factor = initializer_factor
        self.feed_forward_proj = feed_forward_proj
        self.use_cache = use_cache
    
        # 解析前馈网络激活函数信息
        act_info = self.feed_forward_proj.split("-")
        self.dense_act_fn = act_info[-1]  # 获取激活函数名称
        self.is_gated_act = act_info[0] == "gated"  # 判断是否使用门控激活函数
    
        # 检查前馈网络激活函数的有效性
        if len(act_info) > 1 and act_info[0] != "gated" or len(act_info) > 2:
            # 抛出数值错误，提示激活函数格式无效
            raise ValueError(
                f"`feed_forward_proj`: {feed_forward_proj} is not a valid activation function of the dense layer. "
                "Please make sure `feed_forward_proj` is of the format `gated-{ACT_FN}` or `{ACT_FN}`, e.g. "
                "'gated-gelu' or 'relu'"
            )
    
        # 为了向后兼容
        if feed_forward_proj == "gated-gelu":
            self.dense_act_fn = "gelu_new"  # 更新激活函数名称为 gelu_new
    
        # 调用父类的初始化方法，设置填充标记和结束标记的 ID，以及是否是编码器-解码器结构等参数
        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            **kwargs,
        )
    ```  
# T5OnnxConfig 类继承自 OnnxSeq2SeqConfigWithPast 类
class T5OnnxConfig(OnnxSeq2SeqConfigWithPast):
    # inputs 属性是一个字典,包含模型的输入变量名及其对应的维度
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 定义通用的输入变量 
        common_inputs = {
            # 输入序列的 ID 
            "input_ids": {0: "batch", 1: "encoder_sequence"},
            # 输入序列的注意力掩码
            "attention_mask": {0: "batch", 1: "encoder_sequence"},
        }
        # 如果使用过去的信息(past)
        if self.use_past:
            # 注意力掩码的第二个维度包括过去的序列长度和当前序列长度
            common_inputs["attention_mask"][1] = "past_encoder_sequence + sequence"
            # 解码器的输入 ID
            common_inputs["decoder_input_ids"] = {0: "batch"}
            # 解码器的注意力掩码,包括过去和当前的序列长度
            common_inputs["decoder_attention_mask"] = {0: "batch", 1: "past_decoder_sequence + sequence"}
        # 如果不使用过去的信息(past)
        else:
            # 解码器的输入 ID 和注意力掩码的第二个维度是当前序列长度
            common_inputs["decoder_input_ids"] = {0: "batch", 1: "decoder_sequence"}
            common_inputs["decoder_attention_mask"] = {0: "batch", 1: "decoder_sequence"}

        # 如果使用过去的信息(past),更新 common_inputs 以包含过去的 key 和 value
        if self.use_past:
            self.fill_with_past_key_values_(common_inputs, direction="inputs")

        # 返回完整的输入变量字典
        return common_inputs

    # 默认使用的 ONNX 操作集版本
    @property
    def default_onnx_opset(self) -> int:
        return 13
```