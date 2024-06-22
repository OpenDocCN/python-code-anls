# `.\transformers\models\umt5\configuration_umt5.py`

```py
# 设置文件编码为 utf-8
# 版权声明，版权属于 The T5 Authors 和 HuggingFace Inc.
# 根据 Apache License, Version 2.0 许可
# 如果不符合许可条款，禁止使用该文件
# 可以在 http://www.apache.org/licenses/LICENSE-2.0 获取许可副本
# 除非适用法律要求或书面同意，否则根据许可分发的软件是基于“原样”分发的，没有任何种类的保证或条件，明示或暗示
# 请查看特定语言版本的许可证，以获取权限和限制
""" 
UMT5 模型配置"""
# 引入类型提示的 Mapping 模块
from typing import Mapping

# 从...configuration_utils 模块中引入 PretrainedConfig 类
from ...configuration_utils import PretrainedConfig
# 从...onnx 模块中引入 OnnxSeq2SeqConfigWithPast 类
from ...onnx import OnnxSeq2SeqConfigWithPast
# 从...utils 模块中引入 logging 模块
from ...utils import logging
引入日志模块并获取 logger
logger = logging.get_logger(__name__)

UMT5_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "google/umt5-small": "https://huggingface.co/google/umt5-small/resolve/main/config.json",
    # 查看所有 umt5 模型：https://huggingface.co/models?filter=umt5
}

# UMT5Config 类继承自 PretrainedConfig 类
class UMT5Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`UMT5Model`]. It is used to instantiate a UMT5
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the UMT5
    [google/umt5-small](https://huggingface.co/google/umt5-small) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    # UMT5 模型的词汇表大小，定义了可以由 `inputs_ids` 表示的不同令牌的数量
    vocab_size (`int`, *可选*, 默认为 250112):
        # 编码器层和汇聚层的尺寸
        d_model (`int`, *可选*, 默认为 512):
            # 每个注意力头的键、查询、值投影的尺寸。`d_kv` 必须等于 `d_model // num_heads`
            d_kv (`int`, *可选*, 默认为 64):
                # 每个 `UMT5Block` 中的中间前馈层的大小
                d_ff (`int`, *可选*, 默认为 1024):
                    # Transformer 编码器中的隐藏层数量
                    num_layers (`int`, *可选*, 默认为 8):
                        # Transformer 解码器中的隐藏层数量。如果未设置，则使用与 `num_layers` 相同的值
                        num_decoder_layers (`int`, *可选*):
                            # Transformer 编码器中每个注意力层的注意力头数量
                            num_heads (`int`, *可选*, 默认为 6):
                                # 每个注意力层用于分桶的桶的数量
                                relative_attention_num_buckets (`int`, *可选*, 默认为 32):
                                    # 用于桶分离的较长序列的最大距离
                                    relative_attention_max_distance (`int`, *可选*, 默认为 128):
                                        # 所有随机失活层的比率
                                        dropout_rate (`float`, *可选*, 默认为 0.1):
                                            # 类别器的失活比率
                                            classifier_dropout (`float`, *可选*, 默认为 0.0):
                                                # 层归一化层使用的 epsilon
                                                layer_norm_eps (`float`, *可选*, 默认为 1e-6):
                                                    # 用于初始化所有权重矩阵的因子（应保持为1，用于内部初始化测试）
                                                    initializer_factor (`float`, *可选*, 默认为 1):
                                                        # 要使用的前馈层的类型。应该是 `"relu"` 或 `"gated-gelu"` 之一
                                                        feed_forward_proj (`string`, *可选*, 默认为 `"gated-gelu"`):
                                                            # 模型是否应该返回最后一个键/值注意力（不被所有模型使用）
                                                            use_cache (`bool`, *可选*, 默认为 `True`):
                                                                # 模型类型为 UMT5
                                                                model_type = "umt5"
                                                                # 推理时要忽略的键
                                                                keys_to_ignore_at_inference = ["past_key_values"]
    # 初始化函数，用于设置模型的各种参数和属性
    def __init__(
        self,
        vocab_size=250112,
        d_model=512,
        d_kv=64,
        d_ff=1024,
        num_layers=8,
        num_decoder_layers=None,
        num_heads=6,
        relative_attention_num_buckets=32,
        relative_attention_max_distance=128,
        dropout_rate=0.1,
        layer_norm_epsilon=1e-6,
        initializer_factor=1.0,
        feed_forward_proj="gated-gelu",
        is_encoder_decoder=True,
        use_cache=True,
        tokenizer_class="T5Tokenizer",
        tie_word_embeddings=True,
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
        classifier_dropout=0.0,
        **kwargs,
    ):
        # 调用父类的初始化函数，设置一些通用的属性
        super().__init__(
            is_encoder_decoder=is_encoder_decoder,
            tokenizer_class=tokenizer_class,
            tie_word_embeddings=tie_word_embeddings,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            decoder_start_token_id=decoder_start_token_id,
            **kwargs,
        )
        # 设置模型的各项参数
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_kv = d_kv
        self.d_ff = d_ff
        self.num_layers = num_layers
        # 如果未指定解码器层数，则默认与编码器层数相同
        self.num_decoder_layers = (
            num_decoder_layers if num_decoder_layers is not None else self.num_layers
        )  # default = symmetry
        self.num_heads = num_heads
        self.relative_attention_num_buckets = relative_attention_num_buckets
        self.relative_attention_max_distance = relative_attention_max_distance
        self.dropout_rate = dropout_rate
        self.classifier_dropout = classifier_dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_factor = initializer_factor
        self.feed_forward_proj = feed_forward_proj

        # 解析激活函数信息
        act_info = self.feed_forward_proj.split("-")
        self.dense_act_fn = act_info[-1]
        self.is_gated_act = act_info[0] == "gated"

        # 检查激活函数格式是否正确
        if len(act_info) > 1 and act_info[0] != "gated" or len(act_info) > 2:
            raise ValueError(
                f"`feed_forward_proj`: {feed_forward_proj} is not a valid activation function of the dense layer. "
                "Please make sure `feed_forward_proj` is of the format `gated-{ACT_FN}` or `{ACT_FN}`, e.g. "
                "'gated-gelu' or 'relu'"
            )

        # 如果激活函数为"gated-gelu"，则将激活函数设为"gelu_new"
        if feed_forward_proj == "gated-gelu":
            self.dense_act_fn = "gelu_new"

    # 返回隐藏层的大小（即模型的维度）
    @property
    def hidden_size(self):
        return self.d_model

    # 返回注意力头的数量
    @property
    def num_attention_heads(self):
        return self.num_heads

    # 返回隐藏层的数量
    @property
    def num_hidden_layers(self):
        return self.num_layers
class UMT5OnnxConfig(OnnxSeq2SeqConfigWithPast):
    @property
    # 重写了父类的 inputs 属性，返回模型的输入结构
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 定义常见的输入结构
        common_inputs = {
            "input_ids": {0: "batch", 1: "encoder_sequence"},
            "attention_mask": {0: "batch", 1: "encoder_sequence"},
        }
        # 如果使用过去状态，则修改 attention_mask 和 decoder_input_ids 结构
        if self.use_past:
            common_inputs["attention_mask"][1] = "past_encoder_sequence + sequence"
            common_inputs["decoder_input_ids"] = {0: "batch"}
            common_inputs["decoder_attention_mask"] = {0: "batch", 1: "past_decoder_sequence + sequence"}
        else:
            # 否则修改 decoder_input_ids 和 decoder_attention_mask 结构
            common_inputs["decoder_input_ids"] = {0: "batch", 1: "decoder_sequence"}
            common_inputs["decoder_attention_mask"] = {0: "batch", 1: "decoder_sequence"}

        # 如果使用过去状态，则填充 inputs 结构
        if self.use_past:
            self.fill_with_past_key_values_(common_inputs, direction="inputs")

        # 返回模型的输入结构
        return common_inputs

    @property
    # 返回默认的 ONNX 操作集版本号
    def default_onnx_opset(self) -> int:
        return 13

    @property
    # 返回用于验证的容差值
    def atol_for_validation(self) -> float:
        return 5e-4
```