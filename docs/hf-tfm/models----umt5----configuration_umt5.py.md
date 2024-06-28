# `.\models\umt5\configuration_umt5.py`

```
# coding=utf-8
# 定义文件编码为UTF-8

# 版权声明，版权归2023年T5作者及HuggingFace Inc.所有
#
# 根据Apache许可证2.0版发布；除非符合许可证要求，否则不得使用本文件
# 可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，本软件是基于“按原样提供”的基础分发的，
# 没有任何明示或暗示的担保或条件。请参阅许可证了解特定语言的权限和限制。
""" UMT5模型配置 """
# 导入所需模块
from typing import Mapping

# 从configuration_utils模块导入预训练配置类PretrainedConfig
from ...configuration_utils import PretrainedConfig
# 从onnx模块导入OnnxSeq2SeqConfigWithPast配置类
from ...onnx import OnnxSeq2SeqConfigWithPast
# 从utils模块导入logging工具
from ...utils import logging

# 获取logger对象
logger = logging.get_logger(__name__)

# UMT5预训练模型配置文件映射表，将模型名称映射到其配置文件的URL
UMT5_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "google/umt5-small": "https://huggingface.co/google/umt5-small/resolve/main/config.json",
    # 查看所有umt5模型，请访问https://huggingface.co/models?filter=umt5
}

# UMT5配置类，继承自PretrainedConfig
class UMT5Config(PretrainedConfig):
    r"""
    这是用于存储[`UMT5Model`]配置的配置类。根据指定的参数实例化UMT5模型，定义模型架构。
    使用默认值实例化配置将生成类似于UMT5 [google/umt5-small](https://huggingface.co/google/umt5-small) 架构的配置。

    配置对象继承自[`PretrainedConfig`]，可以用于控制模型的输出。阅读[`PretrainedConfig`]的文档以获取更多信息。
    """
    pass  # 空白的配置类，仅作为文档的说明用途
    # 模型类型为 "umt5"
    model_type = "umt5"
    
    # 推理阶段忽略的关键字列表
    keys_to_ignore_at_inference = ["past_key_values"]
    
    # 属性映射字典，将旧名称映射到新名称
    attribute_map = {"hidden_size": "d_model", "num_attention_heads": "num_heads", "num_hidden_layers": "num_layers"}
    # 初始化方法，设置模型的各种参数和默认值
    def __init__(
        self,
        vocab_size=250112,  # 词汇表大小，默认为250112
        d_model=512,  # 模型的维度，默认为512
        d_kv=64,  # 键值的维度，默认为64
        d_ff=1024,  # 前馈神经网络中间层的维度，默认为1024
        num_layers=8,  # 编码器和解码器层数，默认为8
        num_decoder_layers=None,  # 解码器层数，默认与编码器层数相同
        num_heads=6,  # 注意力头的数量，默认为6
        relative_attention_num_buckets=32,  # 相对注意力的桶数量，默认为32
        relative_attention_max_distance=128,  # 相对注意力的最大距离，默认为128
        dropout_rate=0.1,  # dropout率，默认为0.1
        layer_norm_epsilon=1e-6,  # Layer Normalization的epsilon，默认为1e-6
        initializer_factor=1.0,  # 初始化因子，默认为1.0
        feed_forward_proj="gated-gelu",  # 前馈网络的投影类型，默认为"gated-gelu"
        is_encoder_decoder=True,  # 是否为编码器-解码器模型，默认为True
        use_cache=True,  # 是否使用缓存，默认为True
        tokenizer_class="T5Tokenizer",  # Tokenizer类的名称，默认为"T5Tokenizer"
        tie_word_embeddings=True,  # 是否共享编码器和解码器的词嵌入，默认为True
        pad_token_id=0,  # 填充token的ID，默认为0
        eos_token_id=1,  # 结束token的ID，默认为1
        decoder_start_token_id=0,  # 解码器起始token的ID，默认为0
        classifier_dropout=0.0,  # 分类器的dropout率，默认为0.0
        **kwargs,  # 其他参数，用于接收未命名的关键字参数
    ):
        self.vocab_size = vocab_size  # 初始化词汇表大小
        self.d_model = d_model  # 初始化模型维度
        self.d_kv = d_kv  # 初始化键值维度
        self.d_ff = d_ff  # 初始化前馈神经网络中间层维度
        self.num_layers = num_layers  # 初始化层数
        self.num_decoder_layers = (
            num_decoder_layers if num_decoder_layers is not None else self.num_layers
        )  # 初始化解码器层数，如果未指定则与编码器层数相同
        self.num_heads = num_heads  # 初始化注意力头数量
        self.relative_attention_num_buckets = relative_attention_num_buckets  # 初始化相对注意力的桶数量
        self.relative_attention_max_distance = relative_attention_max_distance  # 初始化相对注意力的最大距离
        self.dropout_rate = dropout_rate  # 初始化dropout率
        self.classifier_dropout = classifier_dropout  # 初始化分类器的dropout率
        self.layer_norm_epsilon = layer_norm_epsilon  # 初始化Layer Normalization的epsilon
        self.initializer_factor = initializer_factor  # 初始化初始化因子
        self.feed_forward_proj = feed_forward_proj  # 初始化前馈网络的投影类型
        self.use_cache = use_cache  # 初始化是否使用缓存
    
        act_info = self.feed_forward_proj.split("-")  # 根据"-"分割前馈网络投影类型
        self.dense_act_fn = act_info[-1]  # 密集层的激活函数名称
        self.is_gated_act = act_info[0] == "gated"  # 判断是否为门控激活函数类型
    
        # 检查前馈网络投影类型是否合法
        if len(act_info) > 1 and act_info[0] != "gated" or len(act_info) > 2:
            raise ValueError(
                f"`feed_forward_proj`: {feed_forward_proj} is not a valid activation function of the dense layer. "
                "Please make sure `feed_forward_proj` is of the format `gated-{ACT_FN}` or `{ACT_FN}`, e.g. "
                "'gated-gelu' or 'relu'"
            )
    
        # 如果前馈网络投影类型为"gated-gelu"，则将密集层的激活函数名称设置为"gelu_new"
        if feed_forward_proj == "gated-gelu":
            self.dense_act_fn = "gelu_new"
    
        # 调用父类的初始化方法，传递其他参数
        super().__init__(
            is_encoder_decoder=is_encoder_decoder,
            tokenizer_class=tokenizer_class,
            tie_word_embeddings=tie_word_embeddings,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            decoder_start_token_id=decoder_start_token_id,
            **kwargs,
        )
# 定义一个继承自OnnxSeq2SeqConfigWithPast的配置类UMT5OnnxConfig，用于配置UMT5模型在ONNX中的设置

@property
# 从transformers.models.t5.configuration_t5.T5OnnxConfig.inputs中复制的代码片段，定义模型的输入格式
def inputs(self) -> Mapping[str, Mapping[int, str]]:
    # 定义通用的输入格式字典
    common_inputs = {
        "input_ids": {0: "batch", 1: "encoder_sequence"},
        "attention_mask": {0: "batch", 1: "encoder_sequence"},
    }
    # 如果使用过去信息（use_past为True），更新attention_mask的描述信息
    if self.use_past:
        common_inputs["attention_mask"][1] = "past_encoder_sequence + sequence"
        # 添加decoder_input_ids的描述信息
        common_inputs["decoder_input_ids"] = {0: "batch"}
        # 添加decoder_attention_mask的描述信息
        common_inputs["decoder_attention_mask"] = {0: "batch", 1: "past_decoder_sequence + sequence"}
    else:
        # 如果不使用过去信息，更新decoder_input_ids的描述信息
        common_inputs["decoder_input_ids"] = {0: "batch", 1: "decoder_sequence"}
        # 更新decoder_attention_mask的描述信息
        common_inputs["decoder_attention_mask"] = {0: "batch", 1: "decoder_sequence"}

    # 如果使用过去信息，调用fill_with_past_key_values_方法填充common_inputs
    if self.use_past:
        self.fill_with_past_key_values_(common_inputs, direction="inputs")

    # 返回配置好的输入格式字典
    return common_inputs

@property
# 从transformers.models.t5.configuration_t5.T5OnnxConfig.default_onnx_opset中复制的代码片段，定义默认的ONNX操作集版本号
def default_onnx_opset(self) -> int:
    # 返回ONNX操作集的版本号
    return 13

@property
# 定义一个属性atol_for_validation，返回浮点数类型的验证绝对容差
def atol_for_validation(self) -> float:
    # 返回5e-4作为验证时的绝对容差
    return 5e-4
```