# `.\transformers\models\whisper\configuration_whisper.py`

```py
# 指定编码方式为 UTF-8

# 引入必要的模块和库
# 注意：以下是版权声明

# 引入 OrderedDict 用于有序字典，typing 用于类型提示
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Mapping, Optional, Union

# 引入预训练配置的基类 PretrainedConfig
from ...configuration_utils import PretrainedConfig

# 引入 ONNX 相关配置
from ...onnx import OnnxConfig, OnnxSeq2SeqConfigWithPast

# 引入日志记录工具
from ...utils import logging

# 如果是类型检查环境，则引入额外的模块
if TYPE_CHECKING:
    from ...feature_extraction_utils import FeatureExtractionMixin
    from ...tokenization_utils_base import PreTrainedTokenizerBase
    from ...utils import TensorType

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义预训练配置文件的映射
WHISPER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "openai/whisper-base": "https://huggingface.co/openai/whisper-base/resolve/main/config.json",
}

# 下面是非语音标记的列表，标记了非语音类的词汇
# fmt: off
NON_SPEECH_TOKENS = [
    1, 2, 7, 8, 9, 10, 14, 25,
    26, 27, 28, 29, 31, 58, 59, 60, 61, 62,
    63, 90, 91, 92, 93, 357, 366, 438, 532, 685,
    705, 796, 930, 1058, 1220, 1267, 1279, 1303, 1343, 1377,
    1391, 1635, 1782, 1875, 2162, 2361, 2488, 3467, 4008, 4211,
    4600, 4808, 5299, 5855, 6329, 7203, 9609, 9959, 10563, 10786,
    11420, 11709, 11907, 13163, 13697, 13700, 14808, 15306, 16410, 16791,
    17992, 19203, 19510, 20724, 22305, 22935, 27007, 30109, 30420, 33409,
    34949, 40283, 40493, 40549, 47282, 49146, 50257, 50359, 50360, 50361
]
# 定义更多的非语音标记的列表，标记了非语音类的词汇
NON_SPEECH_TOKENS_MULTI = [
    1, 2, 7, 8, 9, 10, 14, 25,
    26, 27, 28, 29, 31, 58, 59, 60, 61, 62,
    63, 90, 91, 92, 93, 359, 503, 522, 542, 873,
    893, 902, 918, 922, 931, 1350, 1853, 1982, 2460, 2627,
    3246, 3253, 3268, 3536, 3846, 3961, 4183, 4667, 6585, 6647,
    7273, 9061, 9383, 10428, 10929, 11938, 12033, 12331, 12562, 13793,
    14157, 14635, 15265, 15618, 16553, 16604, 18362, 18956, 20075, 21675,
    22520, 26130, 26161, 26435, 28279, 29464, 31650, 32302, 32470, 36865,
    42863, 47425, 49870, 50254, 50258, 50360, 50361, 50362
]
# fmt: on


# Whisper 配置类，继承自预训练配置基类 PretrainedConfig
class WhisperConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`WhisperModel`]. It is used to instantiate a
    Whisper model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Whisper
    [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    """
    documentation from [`PretrainedConfig`] for more information.

    # Whisper 模型的配置类
    Example:

    ```python
    >>> from transformers import WhisperConfig, WhisperModel

    >>> # Initializing a Whisper tiny style configuration
    >>> configuration = WhisperConfig()

    >>> # Initializing a model (with random weights) from the tiny style configuration
    >>> model = WhisperModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```py"""

    # 模型类型
    model_type = "whisper"
    # 推断时需要忽略的键
    keys_to_ignore_at_inference = ["past_key_values"]
    # 属性映射
    attribute_map = {"num_attention_heads": "encoder_attention_heads", "hidden_size": "d_model"}

    # 初始化函数，设置模型配置
    def __init__(
        self,
        vocab_size=51865,
        num_mel_bins=80,
        encoder_layers=4,
        encoder_attention_heads=6,
        decoder_layers=4,
        decoder_attention_heads=6,
        decoder_ffn_dim=1536,
        encoder_ffn_dim=1536,
        encoder_layerdrop=0.0,
        decoder_layerdrop=0.0,
        decoder_start_token_id=50257,
        use_cache=True,
        is_encoder_decoder=True,
        activation_function="gelu",
        d_model=384,
        dropout=0.0,
        attention_dropout=0.0,
        activation_dropout=0.0,
        init_std=0.02,
        scale_embedding=False,
        max_source_positions=1500,
        max_target_positions=448,
        pad_token_id=50256,
        bos_token_id=50256,
        eos_token_id=50256,
        suppress_tokens=None,
        begin_suppress_tokens=[220, 50256],
        use_weighted_layer_sum=False,
        classifier_proj_size=256,
        apply_spec_augment=False,
        mask_time_prob=0.05,
        mask_time_length=10,
        mask_time_min_masks=2,
        mask_feature_prob=0.0,
        mask_feature_length=10,
        mask_feature_min_masks=0,
        median_filter_width=7,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        # 设置词汇表大小
        self.num_mel_bins = num_mel_bins
        # 设置梅尔频谱的频道数
        self.d_model = d_model
        # 设置模型中隐藏层的大小
        self.encoder_layers = encoder_layers
        # 设置编码器层数
        self.encoder_attention_heads = encoder_attention_heads
        # 设置编码器的注意力头数
        self.decoder_layers = decoder_layers
        # 设置解码器层数
        self.decoder_attention_heads = decoder_attention_heads
        # 设置解码器的注意力头数
        self.decoder_ffn_dim = decoder_ffn_dim
        # 设置解码器的前馈网络维度
        self.encoder_ffn_dim = encoder_ffn_dim
        # 设置编码器的前馈网络维度
        self.dropout = dropout
        # 设置dropout率
        self.attention_dropout = attention_dropout
        # 设置注意力层的dropout率
        self.activation_dropout = activation_dropout
        # 设置激活函数的dropout率
        self.activation_function = activation_function
        # 设置激活函数的类型
        self.init_std = init_std
        # 设置初始化的标准差
        self.encoder_layerdrop = encoder_layerdrop
        # 设置编码器层的丢弃率
        self.decoder_layerdrop = decoder_layerdrop
        # 设置解码器层的丢弃率
        self.use_cache = use_cache
        # 是否使用缓存
        self.num_hidden_layers = encoder_layers
        # 设置隐藏层的数量为编码器层数
        self.scale_embedding = scale_embedding  # scale factor will be sqrt(d_model) if True
        # 如果为True，将使用sqrt(d_model)作为缩放因子
        self.max_source_positions = max_source_positions
        # 设置最大源序列长度
        self.max_target_positions = max_target_positions
        # 设置最大目标序列长度

        # 音频分类特定参数，其他情况可以忽略
        self.classifier_proj_size = classifier_proj_size
        # 分类器投影层大小
        self.use_weighted_layer_sum = use_weighted_layer_sum
        # 是否使用加权层求和

        # 对SpecAugment进行微调的配置参数：https://arxiv.org/abs/1904.08779
        self.apply_spec_augment = apply_spec_augment
        # 是否应用SpecAugment技术
        self.mask_time_prob = mask_time_prob
        # 时间掩码概率
        self.mask_time_length = mask_time_length
        # 时间掩码长度
        self.mask_time_min_masks = mask_time_min_masks
        # 时间掩码最小数量
        self.mask_feature_prob = mask_feature_prob
        # 特征掩码概率
        self.mask_feature_length = mask_feature_length
        # 特征掩码长度
        self.mask_feature_min_masks = mask_feature_min_masks
        # 特征掩��最小数量

        self.median_filter_width = median_filter_width
        # 中值滤波宽度

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            decoder_start_token_id=decoder_start_token_id,
            suppress_tokens=suppress_tokens,
            begin_suppress_tokens=begin_suppress_tokens,
            **kwargs,
        )
        # 调用父类的初始化函数，并传递参数
# 定义了一个名为 WhisperOnnxConfig 的类，继承自 OnnxSeq2SeqConfigWithPast 类
class WhisperOnnxConfig(OnnxSeq2SeqConfigWithPast):
    # 定义 inputs 属性，返回输入的字典结构
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 定义了通用输入的有序字典结构
        common_inputs = OrderedDict(
            [
                # 定义了输入特征的名称与维度的映射
                ("input_features", {0: "batch", 1: "feature_size", 2: "encoder_sequence"}),
            ]
        )
        # 如果使用过去的信息
        if self.use_past:
            # 向 common_inputs 中添加 decoder_input_ids 的映射，只有 batch 维度
            common_inputs["decoder_input_ids"] = {0: "batch"}
        else:
            # 向 common_inputs 中添加 decoder_input_ids 的映射，包括 batch 和 decoder_sequence 维度
            common_inputs["decoder_input_ids"] = {0: "batch", 1: "decoder_sequence"}

        # 如果使用过去的信息
        if self.use_past:
            # 填充 common_inputs 中的过去键值对信息
            self.fill_with_past_key_values_(common_inputs, direction="inputs")

        # 返回输入的字典结构
        return common_inputs

    # 生成虚拟输入的方法
    def generate_dummy_inputs(
        self,
        preprocessor: Union["PreTrainedTokenizerBase", "FeatureExtractionMixin"],
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional["TensorType"] = None,
        sampling_rate: int = 22050,
        time_duration: float = 5.0,
        frequency: int = 220,
    ) -> Mapping[str, Any]:
        # 定义虚拟输入的有序字典结构
        dummy_inputs = OrderedDict()
        # 生成编码器输入的虚拟输入
        encoder_inputs = OnnxConfig.generate_dummy_inputs(
            self,
            preprocessor=preprocessor.feature_extractor,
            batch_size=batch_size,
            framework=framework,
            sampling_rate=sampling_rate,
            time_duration=time_duration,
            frequency=frequency,
        )
        # 获取编码器输入的序列长度
        encoder_sequence_length = encoder_inputs["input_features"].shape[2]
        # 如果使用过去的信息，则将序列长度减半
        seq_length = encoder_sequence_length // 2 if self.use_past else seq_length

        # 生成解码器输入的虚拟输入
        decoder_inputs = super().generate_dummy_inputs(
            preprocessor.tokenizer, batch_size, seq_length, is_pair, framework
        )

        # 将编码器输入的 input_features 放入虚拟输入的字典中
        dummy_inputs["input_features"] = encoder_inputs.pop("input_features")
        # 将解码器输入的 decoder_input_ids 放入虚拟输入的字典中
        dummy_inputs["decoder_input_ids"] = decoder_inputs.pop("decoder_input_ids")

        # 如果解码器输入中存在过去键值对信息，则将其放入虚拟输入的字典中
        if "past_key_values" in decoder_inputs:
            dummy_inputs["past_key_values"] = decoder_inputs.pop("past_key_values")

        # 返回虚拟输入的字典结构
        return dummy_inputs

    # 定义了用于验证的绝对容差值
    @property
    def atol_for_validation(self) -> float:
        return 1e-3
```