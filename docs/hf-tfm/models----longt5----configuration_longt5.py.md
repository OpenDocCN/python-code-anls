# `.\models\longt5\configuration_longt5.py`

```
# 引入必要的模块和类，包括预训练配置类、OnnxSeq2SeqConfigWithPast 类和日志记录工具
""" LongT5 model configuration"""
from typing import Mapping  # 导入 Mapping 类型

from ...configuration_utils import PretrainedConfig  # 导入预训练配置类
from ...onnx import OnnxSeq2SeqConfigWithPast  # 导入 OnnxSeq2SeqConfigWithPast 类
from ...utils import logging  # 导入日志记录工具

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义预训练模型配置文件的下载映射字典，每个模型名称对应其配置文件的下载链接
LONGT5_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "google/long-t5-local-base": "https://huggingface.co/google/long-t5-local-base/blob/main/config.json",
    "google/long-t5-local-large": "https://huggingface.co/google/long-t5-local-large/blob/main/config.json",
    "google/long-t5-tglobal-base": "https://huggingface.co/google/long-t5-tglobal-base/blob/main/config.json",
    "google/long-t5-tglobal-large": "https://huggingface.co/google/long-t5-tglobal-large/blob/main/config.json",
}


class LongT5Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LongT5Model`] or a [`FlaxLongT5Model`]. It is
    used to instantiate a LongT5 model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the LongT5
    [google/long-t5-local-base](https://huggingface.co/google/long-t5-local-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    # 设置模型类型为 "longt5"
    model_type = "longt5"
    # 定义一个在推理过程中要忽略的键列表
    keys_to_ignore_at_inference = ["past_key_values"]
    # 定义一个将类属性名称映射到别名的字典
    attribute_map = {"hidden_size": "d_model", "num_attention_heads": "num_heads", "num_hidden_layers": "num_layers"}

    # 初始化方法，用于设置模型的各种参数和属性
    def __init__(
        self,
        vocab_size=32128,  # 词汇表大小，默认为32128
        d_model=512,  # 隐藏层大小，默认为512
        d_kv=64,  # 键值的维度，默认为64
        d_ff=2048,  # 前馈神经网络内部层的维度，默认为2048
        num_layers=6,  # 网络层数，默认为6
        num_decoder_layers=None,  # 解码器层数，默认为None，即与编码器层数相同
        num_heads=8,  # 注意力头的数量，默认为8
        local_radius=127,  # 本地注意力的半径，默认为127
        global_block_size=16,  # 全局块大小，默认为16
        relative_attention_num_buckets=32,  # 相对注意力的桶数，默认为32
        relative_attention_max_distance=128,  # 相对注意力的最大距离，默认为128
        dropout_rate=0.1,  # Dropout率，默认为0.1
        layer_norm_epsilon=1e-6,  # Layer normalization的epsilon值，默认为1e-6
        initializer_factor=1.0,  # 初始化因子，默认为1.0
        feed_forward_proj="relu",  # 前馈网络的激活函数，默认为relu
        is_encoder_decoder=True,  # 是否为编码器-解码器结构，默认为True
        encoder_attention_type="local",  # 编码器注意力的类型，默认为local
        use_cache=True,  # 是否使用缓存，默认为True
        pad_token_id=0,  # 填充标记的ID，默认为0
        eos_token_id=1,  # 结束标记的ID，默认为1
        **kwargs,  # 其他关键字参数，用于传递给父类构造函数
    ):
        # 设置对象的各种属性值
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.num_layers = num_layers
        # 如果给定的解码器层数不为None，则使用给定的值，否则使用编码器层数作为解码器层数
        self.num_decoder_layers = num_decoder_layers if num_decoder_layers is not None else self.num_layers
        self.num_heads = num_heads
        self.local_radius = local_radius
        self.global_block_size = global_block_size
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.dropout_rate = dropout_rate
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_factor = initializer_factor
        self.feed_forward_proj = feed_forward_proj
        self.encoder_attention_type = encoder_attention_type
        self.use_cache = use_cache

        # 解析前馈网络激活函数的信息，提取激活函数名称和是否为门控激活函数的标志
        act_info = self.feed_forward_proj.split("-")
        self.dense_act_fn = act_info[-1]  # 提取激活函数的名称
        self.is_gated_act = act_info[0] == "gated"  # 判断是否为门控激活函数

        # 如果激活函数信息的长度超出预期或格式不正确，则抛出值错误异常
        if len(act_info) > 1 and act_info[0] != "gated" or len(act_info) > 2:
            raise ValueError(
                f"`feed_forward_proj`: {feed_forward_proj} is not a valid activation function of the dense layer. "
                "Please make sure `feed_forward_proj` is of the format `gated-{ACT_FN}` or `{ACT_FN}`, e.g. "
                "'gated-gelu' or 'relu'"
            )

        # 对于向后兼容性，如果前馈网络激活函数设为'gated-gelu'，则更新为'gelu_new'
        if feed_forward_proj == "gated-gelu":
            self.dense_act_fn = "gelu_new"

        # 调用父类的初始化方法，传递填充标记ID、结束标记ID、是否为编码器-解码器等参数以及其他关键字参数
        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            **kwargs,
        )
# 定义一个名为 LongT5OnnxConfig 的类，继承自 OnnxSeq2SeqConfigWithPast 类
class LongT5OnnxConfig(OnnxSeq2SeqConfigWithPast):
    
    # inputs 属性，返回一个映射，描述了模型的输入结构
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 定义通用的输入格式，包括 input_ids 和 attention_mask
        common_inputs = {
            "input_ids": {0: "batch", 1: "encoder_sequence"},
            "attention_mask": {0: "batch", 1: "encoder_sequence"},
        }
        
        # 如果使用过去信息（use_past 为 True）
        if self.use_past:
            # 调整 attention_mask 的描述以包括过去编码器序列
            common_inputs["attention_mask"][1] = "past_encoder_sequence + sequence"
            # 添加 decoder_input_ids 的描述
            common_inputs["decoder_input_ids"] = {0: "batch"}
            # 添加 decoder_attention_mask 的描述，包括过去解码器序列
            common_inputs["decoder_attention_mask"] = {0: "batch", 1: "past_decoder_sequence + sequence"}
        else:
            # 如果不使用过去信息，添加普通的 decoder_input_ids 描述
            common_inputs["decoder_input_ids"] = {0: "batch", 1: "decoder_sequence"}
            # 添加普通的 decoder_attention_mask 描述
            common_inputs["decoder_attention_mask"] = {0: "batch", 1: "decoder_sequence"}

        # 如果使用过去信息，调用 fill_with_past_key_values_ 方法填充 common_inputs
        if self.use_past:
            self.fill_with_past_key_values_(common_inputs, direction="inputs")

        # 返回描述输入结构的字典 common_inputs
        return common_inputs

    # default_onnx_opset 属性，返回默认的 ONNX 运算集版本号
    @property
    def default_onnx_opset(self) -> int:
        # 返回 ONNX 运算集版本号 13
        return 13
```