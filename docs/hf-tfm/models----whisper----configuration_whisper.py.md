# `.\models\whisper\configuration_whisper.py`

```py
# coding=utf-8
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
# 上述为版权声明和编码声明

# 导入必要的模块和类
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Mapping, Optional, Union

# 导入预训练配置类和相关的配置
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig, OnnxSeq2SeqConfigWithPast
from ...utils import logging

# 如果是类型检查模式，则导入额外的类
if TYPE_CHECKING:
    from ...feature_extraction_utils import FeatureExtractionMixin
    from ...tokenization_utils_base import PreTrainedTokenizerBase
    from ...utils import TensorType

# 获取日志记录器
logger = logging.get_logger(__name__)

# 预训练模型配置文件映射字典
WHISPER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "openai/whisper-base": "https://huggingface.co/openai/whisper-base/resolve/main/config.json",
}

# fmt: off
# 定义非语音标记的列表
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
# 定义多模态的非语音标记的列表
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

class WhisperConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`WhisperModel`]. It is used to instantiate a
    Whisper model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Whisper
    [openai/whisper-tiny](https://huggingface.co/openai/whisper-tiny) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    rest of this class docstring for more information.
    """
    # 定义模型类型为 "whisper"
    model_type = "whisper"
    
    # 推理阶段忽略的关键字列表
    keys_to_ignore_at_inference = ["past_key_values"]
    
    # 属性映射字典，将 PretrainedConfig 中的属性映射到本地属性
    attribute_map = {"num_attention_heads": "encoder_attention_heads", "hidden_size": "d_model"}

    # 初始化函数，设置模型的各种配置参数
    def __init__(
        self,
        vocab_size=51865,  # 词汇表大小，默认为 51865
        num_mel_bins=80,  # MEL bins 数量，默认为 80
        encoder_layers=4,  # 编码器层数，默认为 4
        encoder_attention_heads=6,  # 编码器注意力头数，默认为 6
        decoder_layers=4,  # 解码器层数，默认为 4
        decoder_attention_heads=6,  # 解码器注意力头数，默认为 6
        decoder_ffn_dim=1536,  # 解码器 FFN 维度，默认为 1536
        encoder_ffn_dim=1536,  # 编码器 FFN 维度，默认为 1536
        encoder_layerdrop=0.0,  # 编码器层丢弃率，默认为 0.0
        decoder_layerdrop=0.0,  # 解码器层丢弃率，默认为 0.0
        decoder_start_token_id=50257,  # 解码器起始 token ID，默认为 50257
        use_cache=True,  # 是否使用缓存，默认为 True
        is_encoder_decoder=True,  # 是否是编码器-解码器模型，默认为 True
        activation_function="gelu",  # 激活函数类型，默认为 "gelu"
        d_model=384,  # 模型维度，默认为 384
        dropout=0.0,  # 全局 dropout 率，默认为 0.0
        attention_dropout=0.0,  # 注意力 dropout 率，默认为 0.0
        activation_dropout=0.0,  # 激活函数 dropout 率，默认为 0.0
        init_std=0.02,  # 参数初始化标准差，默认为 0.02
        scale_embedding=False,  # 是否对嵌入进行缩放，默认为 False
        max_source_positions=1500,  # 最大源序列长度，默认为 1500
        max_target_positions=448,  # 最大目标序列长度，默认为 448
        pad_token_id=50256,  # 填充 token ID，默认为 50256
        bos_token_id=50256,  # 起始 token ID，默认为 50256
        eos_token_id=50256,  # 结束 token ID，默认为 50256
        suppress_tokens=None,  # 要抑制的特定 token 列表，默认为 None
        begin_suppress_tokens=[220, 50256],  # 开始抑制的 token 列表，默认为 [220, 50256]
        use_weighted_layer_sum=False,  # 是否使用加权层求和，默认为 False
        classifier_proj_size=256,  # 分类器投影大小，默认为 256
        apply_spec_augment=False,  # 是否应用语音增强，默认为 False
        mask_time_prob=0.05,  # 时间掩码概率，默认为 0.05
        mask_time_length=10,  # 时间掩码长度，默认为 10
        mask_time_min_masks=2,  # 时间掩码最小数量，默认为 2
        mask_feature_prob=0.0,  # 特征掩码概率，默认为 0.0
        mask_feature_length=10,  # 特征掩码长度，默认为 10
        mask_feature_min_masks=0,  # 特征掩码最小数量，默认为 0
        median_filter_width=7,  # 中值滤波器宽度，默认为 7
        **kwargs,  # 其他关键字参数
        ):
        # 初始化模型参数
        self.vocab_size = vocab_size  # 词汇表大小
        self.num_mel_bins = num_mel_bins  # 梅尔频谱的频道数
        self.d_model = d_model  # 模型的维度大小
        self.encoder_layers = encoder_layers  # 编码器层数
        self.encoder_attention_heads = encoder_attention_heads  # 编码器注意力头数
        self.decoder_layers = decoder_layers  # 解码器层数
        self.decoder_attention_heads = decoder_attention_heads  # 解码器注意力头数
        self.decoder_ffn_dim = decoder_ffn_dim  # 解码器前馈网络的维度
        self.encoder_ffn_dim = encoder_ffn_dim  # 编码器前馈网络的维度
        self.dropout = dropout  # 总体dropout概率
        self.attention_dropout = attention_dropout  # 注意力机制中的dropout概率
        self.activation_dropout = activation_dropout  # 激活函数中的dropout概率
        self.activation_function = activation_function  # 激活函数类型
        self.init_std = init_std  # 参数初始化的标准差
        self.encoder_layerdrop = encoder_layerdrop  # 编码器层的LayerDrop比例
        self.decoder_layerdrop = decoder_layerdrop  # 解码器层的LayerDrop比例
        self.use_cache = use_cache  # 是否使用缓存
        self.num_hidden_layers = encoder_layers  # 隐藏层的数量（与编码器层数相同）
        self.scale_embedding = scale_embedding  # 若为True，则嵌入的缩放因子为sqrt(d_model)
        self.max_source_positions = max_source_positions  # 源序列的最大位置编码数
        self.max_target_positions = max_target_positions  # 目标序列的最大位置编码数

        # 音频分类特定参数，其他情况下可忽略
        self.classifier_proj_size = classifier_proj_size  # 分类器投影的维度
        self.use_weighted_layer_sum = use_weighted_layer_sum  # 是否使用加权层求和

        # SpecAugment的微调配置参数：https://arxiv.org/abs/1904.08779
        self.apply_spec_augment = apply_spec_augment  # 是否应用SpecAugment
        self.mask_time_prob = mask_time_prob  # 时间掩码的概率
        self.mask_time_length = mask_time_length  # 时间掩码的长度
        self.mask_time_min_masks = mask_time_min_masks  # 时间掩码的最小数量
        self.mask_feature_prob = mask_feature_prob  # 特征掩码的概率
        self.mask_feature_length = mask_feature_length  # 特征掩码的长度
        self.mask_feature_min_masks = mask_feature_min_masks  # 特征掩码的最小数量

        self.median_filter_width = median_filter_width  # 中值滤波器的宽度

        # 调用父类的初始化方法
        super().__init__(
            pad_token_id=pad_token_id,  # 填充token的ID
            bos_token_id=bos_token_id,  # 开始token的ID
            eos_token_id=eos_token_id,  # 结束token的ID
            is_encoder_decoder=is_encoder_decoder,  # 是否为编码-解码模型
            decoder_start_token_id=decoder_start_token_id,  # 解码器起始token的ID
            suppress_tokens=suppress_tokens,  # 需要抑制的token
            begin_suppress_tokens=begin_suppress_tokens,  # 开始抑制的token
            **kwargs,  # 其他关键字参数
        )
# 定义一个名为 WhisperOnnxConfig 的类，继承自 OnnxSeq2SeqConfigWithPast 类
class WhisperOnnxConfig(OnnxSeq2SeqConfigWithPast):

    # 定义一个属性方法 inputs，返回一个字典，描述模型的输入结构
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 创建一个有序字典 common_inputs，包含模型的常见输入
        common_inputs = OrderedDict(
            [
                ("input_features", {0: "batch", 1: "feature_size", 2: "encoder_sequence"}),
            ]
        )
        
        # 根据 self.use_past 的值决定是否添加 decoder_input_ids 到 common_inputs 中
        if self.use_past:
            common_inputs["decoder_input_ids"] = {0: "batch"}
        else:
            common_inputs["decoder_input_ids"] = {0: "batch", 1: "decoder_sequence"}

        # 根据 self.use_past 的值，调用 fill_with_past_key_values_ 方法填充 common_inputs
        if self.use_past:
            self.fill_with_past_key_values_(common_inputs, direction="inputs")

        # 返回描述模型输入的字典 common_inputs
        return common_inputs

    # 定义一个方法 generate_dummy_inputs，生成用于测试的虚拟输入数据
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
        # 创建一个有序字典 dummy_inputs，用于存储虚拟输入数据
        dummy_inputs = OrderedDict()
        
        # 调用 OnnxConfig 类的 generate_dummy_inputs 方法生成 encoder_inputs
        encoder_inputs = OnnxConfig.generate_dummy_inputs(
            self,
            preprocessor=preprocessor.feature_extractor,
            batch_size=batch_size,
            framework=framework,
            sampling_rate=sampling_rate,
            time_duration=time_duration,
            frequency=frequency,
        )
        
        # 计算 encoder_inputs 的 encoder_sequence_length
        encoder_sequence_length = encoder_inputs["input_features"].shape[2]
        
        # 根据 self.use_past 的值更新 seq_length
        seq_length = encoder_sequence_length // 2 if self.use_past else seq_length
        
        # 调用父类的 generate_dummy_inputs 方法生成 decoder_inputs
        decoder_inputs = super().generate_dummy_inputs(
            preprocessor.tokenizer, batch_size, seq_length, is_pair, framework
        )

        # 将 encoder_inputs 的 input_features 移动到 dummy_inputs 中
        dummy_inputs["input_features"] = encoder_inputs.pop("input_features")
        
        # 将 decoder_inputs 的 decoder_input_ids 移动到 dummy_inputs 中
        dummy_inputs["decoder_input_ids"] = decoder_inputs.pop("decoder_input_ids")

        # 如果 decoder_inputs 中有 past_key_values，则将其移动到 dummy_inputs 中
        if "past_key_values" in decoder_inputs:
            dummy_inputs["past_key_values"] = decoder_inputs.pop("past_key_values")

        # 返回包含虚拟输入数据的 dummy_inputs 字典
        return dummy_inputs

    # 定义一个属性方法 atol_for_validation，返回用于验证的容差值
    @property
    def atol_for_validation(self) -> float:
        return 1e-3
```