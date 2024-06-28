# `.\models\t5\configuration_t5.py`

```py
# coding=utf-8
# 引入必要的模块和类
# 版权声明和许可协议
# 版权所有 2020 年，T5 作者和 HuggingFace 公司
#
# 根据 Apache 许可协议 2.0 版本（“许可证”），除非符合许可证，否则不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件根据“原样”分发，
# 没有任何形式的担保或条件，无论是明示的还是默示的。
# 有关更多信息，请参阅许可协议。
""" T5 模型配置 """
# 导入必要的类型注解
from typing import Mapping

# 从相关模块导入所需的类和函数
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxSeq2SeqConfigWithPast
from ...utils import logging

# 获取 logger 对象，用于日志记录
logger = logging.get_logger(__name__)

# 预训练配置文件的映射表，包含了不同预训练模型的配置文件 URL
T5_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "google-t5/t5-small": "https://huggingface.co/google-t5/t5-small/resolve/main/config.json",
    "google-t5/t5-base": "https://huggingface.co/google-t5/t5-base/resolve/main/config.json",
    "google-t5/t5-large": "https://huggingface.co/google-t5/t5-large/resolve/main/config.json",
    "google-t5/t5-3b": "https://huggingface.co/google-t5/t5-3b/resolve/main/config.json",
    "google-t5/t5-11b": "https://huggingface.co/google-t5/t5-11b/resolve/main/config.json",
}

# T5Config 类，继承自 PretrainedConfig 类
class T5Config(PretrainedConfig):
    r"""
    这是一个配置类，用于存储 [`T5Model`] 或 [`TFT5Model`] 的配置。它用于根据指定的参数实例化 T5 模型，定义模型架构。
    使用默认参数实例化配置对象将产生类似于 T5 [google-t5/t5-small](https://huggingface.co/google-t5/t5-small) 架构的配置。

    配置对象继承自 [`PretrainedConfig`]，可以用于控制模型输出。有关更多信息，请参阅 [`PretrainedConfig`] 的文档。
    ```
    # 模型类型设定为 "t5"
    model_type = "t5"
    
    # 推断阶段忽略的关键字列表，这里包括 "past_key_values"
    keys_to_ignore_at_inference = ["past_key_values"]
    
    # 属性映射字典，将模型属性名映射到通用命名
    attribute_map = {"hidden_size": "d_model", "num_attention_heads": "num_heads", "num_hidden_layers": "num_layers"}
    # 初始化函数，用于初始化Transformer模型的各种参数和配置
    def __init__(
        self,
        vocab_size=32128,  # 词汇表大小，默认为32128
        d_model=512,  # 模型的维度，默认为512
        d_kv=64,  # 键和值的维度，默认为64
        d_ff=2048,  # Feed Forward层的维度，默认为2048
        num_layers=6,  # Transformer层的数量，默认为6
        num_decoder_layers=None,  # 解码器层的数量，默认为num_layers的值，保持对称性
        num_heads=8,  # 多头注意力机制中的头数，默认为8
        relative_attention_num_buckets=32,  # 相对注意力机制中的桶数，默认为32
        relative_attention_max_distance=128,  # 相对注意力机制的最大距离，默认为128
        dropout_rate=0.1,  # Dropout率，默认为0.1
        layer_norm_epsilon=1e-6,  # Layer Normalization中的epsilon，默认为1e-6
        initializer_factor=1.0,  # 初始化因子，默认为1.0
        feed_forward_proj="relu",  # 前向传播投影层的激活函数，默认为'relu'
        is_encoder_decoder=True,  # 是否是编码器-解码器结构，默认为True
        use_cache=True,  # 是否使用缓存，默认为True
        pad_token_id=0,  # 填充token的ID，默认为0
        eos_token_id=1,  # EOS（句子结束）token的ID，默认为1
        classifier_dropout=0.0,  # 分类器中的Dropout率，默认为0.0
        **kwargs,  # 其他关键字参数
    ):
        self.vocab_size = vocab_size  # 初始化词汇表大小
        self.d_model = d_model  # 初始化模型的维度
        self.d_kv = d_kv  # 初始化键和值的维度
        self.d_ff = d_ff  # 初始化Feed Forward层的维度
        self.num_layers = num_layers  # 初始化Transformer层的数量
        self.num_decoder_layers = (
            num_decoder_layers if num_decoder_layers is not None else self.num_layers
        )  # 解码器层数，默认为num_layers的值，保持对称性
        self.num_heads = num_heads  # 初始化多头注意力机制中的头数
        self.relative_attention_num_buckets = relative_attention_num_buckets  # 初始化相对注意力机制中的桶数
        self.relative_attention_max_distance = relative_attention_max_distance  # 初始化相对注意力机制的最大距离
        self.dropout_rate = dropout_rate  # 初始化Dropout率
        self.classifier_dropout = classifier_dropout  # 初始化分类器中的Dropout率
        self.layer_norm_epsilon = layer_norm_epsilon  # 初始化Layer Normalization中的epsilon
        self.initializer_factor = initializer_factor  # 初始化初始化因子
        self.feed_forward_proj = feed_forward_proj  # 初始化前向传播投影层的激活函数
        self.use_cache = use_cache  # 初始化是否使用缓存

        act_info = self.feed_forward_proj.split("-")  # 拆分前向传播投影激活函数的信息
        self.dense_act_fn = act_info[-1]  # 设置密集层的激活函数
        self.is_gated_act = act_info[0] == "gated"  # 判断是否为门控激活函数

        if len(act_info) > 1 and act_info[0] != "gated" or len(act_info) > 2:
            # 如果激活函数信息长度大于1且不是'gated'或长度大于2，则引发值错误
            raise ValueError(
                f"`feed_forward_proj`: {feed_forward_proj} is not a valid activation function of the dense layer. "
                "Please make sure `feed_forward_proj` is of the format `gated-{ACT_FN}` or `{ACT_FN}`, e.g. "
                "'gated-gelu' or 'relu'"
            )

        # 为了向后兼容性
        if feed_forward_proj == "gated-gelu":
            self.dense_act_fn = "gelu_new"  # 设置密集层的激活函数为'gelu_new'

        super().__init__(
            pad_token_id=pad_token_id,  # 初始化填充token的ID
            eos_token_id=eos_token_id,  # 初始化EOS（句子结束）token的ID
            is_encoder_decoder=is_encoder_decoder,  # 初始化是否是编码器-解码器结构
            **kwargs,  # 其他关键字参数
        )
class T5OnnxConfig(OnnxSeq2SeqConfigWithPast):
    # 定义 T5 模型的配置类，继承自带过去信息的 Seq2Seq ONNX 配置类

    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 定义输入属性，返回一个映射类型，将字符串键映射到包含整数和字符串的字典

        # 常见的输入配置，包括输入 ID 和注意力掩码
        common_inputs = {
            "input_ids": {0: "batch", 1: "encoder_sequence"},
            "attention_mask": {0: "batch", 1: "encoder_sequence"},
        }

        # 如果使用过去信息
        if self.use_past:
            # 调整注意力掩码以包含过去的编码器序列信息
            common_inputs["attention_mask"][1] = "past_encoder_sequence + sequence"
            # 添加解码器输入 ID 的配置
            common_inputs["decoder_input_ids"] = {0: "batch"}
            # 添加解码器注意力掩码的配置，包括过去的解码器序列信息和当前序列信息
            common_inputs["decoder_attention_mask"] = {0: "batch", 1: "past_decoder_sequence + sequence"}
        else:
            # 添加默认的解码器输入 ID 配置
            common_inputs["decoder_input_ids"] = {0: "batch", 1: "decoder_sequence"}
            # 添加默认的解码器注意力掩码配置
            common_inputs["decoder_attention_mask"] = {0: "batch", 1: "decoder_sequence"}

        # 如果使用过去信息，调用内部方法填充键值对
        if self.use_past:
            self.fill_with_past_key_values_(common_inputs, direction="inputs")

        # 返回最终的输入配置字典
        return common_inputs

    @property
    def default_onnx_opset(self) -> int:
        # 定义默认的 ONNX 运算集版本号为 13
        return 13
```