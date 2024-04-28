# `.\transformers\models\wav2vec2_conformer\configuration_wav2vec2_conformer.py`

```py
# 设定文件编码为utf-8
# 版权声明及许可协议
# 本版权声明适用于Fairseq团队与HuggingFace Inc.团队，所有权利均为其所有
# 根据Apache License, Version 2.0进行许可
# 除非符合许可条款，否则禁止使用本文件
# 可在以下网址获取许可协议的一份副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则依据本许可协议散发的软件是基于“原样”基础提供的
# 没有任何担保或条件，无论是明示还是含蓄的
# 请查看协议以了解具体语言管理权限和限制

""" Wav2Vec2Conformer模型配置"""

# 引入必要的库和模块
import functools
import operator

# 从预训练配置中进行引用
from ...configuration_utils import PretrainedConfig

# 引入日志记录相关工具
from ...utils import logging

# 获取全局的日志记录器
logger = logging.get_logger(__name__)

# 定义Wav2Vec2Conformer预训练配置文件地址映射
WAV2VEC2_CONFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/wav2vec2-conformer-rel-pos-large": (
        "https://huggingface.co/facebook/wav2vec2-conformer-rel-pos-large/resolve/main/config.json"
    ),
}

# Wav2Vec2Conformer配置类，用于存储[`Wav2Vec2ConformerModel`]的配置
class Wav2Vec2ConformerConfig(PretrainedConfig):
    r"""
    这是用于存储[`Wav2Vec2ConformerModel`]的配置的配置类。用于根据指定的参数实例化一个Wav2Vec2Conformer模型，
    定义模型架构。使用默认值实例化一个配置将会产生类似于Wav2Vec2Conformer[facebook/wav2vec2-conformer-rel-pos-large]
    (https://huggingface.co/facebook/wav2vec2-conformer-rel-pos-large)架构的配置。

    配置对象继承自[`PretrainedConfig`]，可用于控制模型的输出。 有关更多信息，请阅读来自[`PretrainedConfig`]的文档。

    示例:

    ```python
    >>> from transformers import Wav2Vec2ConformerConfig, Wav2Vec2ConformerModel

    >>> # 初始化一个Wav2Vec2Conformer facebook/wav2vec2-conformer-rel-pos-large风格的配置
    >>> configuration = Wav2Vec2ConformerConfig()

    >>> # 从facebook/wav2vec2-conformer-rel-pos-large风格的配置初始化一个模型(带有随机权重)
    >>> model = Wav2Vec2ConformerModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```py"""
    
    # 模型类型为wav2vec2-conformer
    model_type = "wav2vec2-conformer"
    # 初始化模型的参数
    def __init__(
        self,
        vocab_size=None,  # 词汇表大小，默认为空
        hidden_size=768,  # 隐藏层大小，默认为768
        num_hidden_layers=12,  # 隐藏层数，默认为12
        num_attention_heads=12,  # 注意力头数，默认为12
        intermediate_size=3072,  # 隐层大小，默认为3072
        hidden_act="gelu",  # 隐藏层激活函数，默认为gelu
        hidden_dropout=0.1,  # 隐藏层dropout率，默认为0.1
        activation_dropout=0.1,  # 激活函数dropout率，默认为0.1
        attention_dropout=0.1,  # 注意力dropout率，默认为0.1
        feat_proj_dropout=0.0,  # 特征投影dropout率，默认为0.0
        feat_quantizer_dropout=0.0,  # 特征量化器dropout率，默认为0.0
        final_dropout=0.1,  # 最终dropout率，默认为0.1
        layerdrop=0.1,  # 层dropout率，默认为0.1
        initializer_range=0.02,  # 初始化范围，默认为0.02
        layer_norm_eps=1e-5,  # 层归一化epsilon，默认为1e-5
        feat_extract_norm="group",  # 特征提取归一化，默认为group
        feat_extract_activation="gelu",  # 特征提取激活函数，默认为gelu
        conv_dim=(512, 512, 512, 512, 512, 512, 512),  # 卷积维度，默认为(512, 512, 512, 512, 512, 512, 512)
        conv_stride=(5, 2, 2, 2, 2, 2, 2),  # 卷积步长，默认为(5, 2, 2, 2, 2, 2, 2)
        conv_kernel=(10, 3, 3, 3, 3, 2, 2),  # 卷积核大小，默认为(10, 3, 3, 3, 3, 2, 2)
        conv_bias=False,  # 是否使用卷积偏置，默认为False
        num_conv_pos_embeddings=128,  # 卷积位置嵌入数量，默认为128
        num_conv_pos_embedding_groups=16,  # 卷积位置嵌入组数，默认为16
        apply_spec_augment=True,  # 是否应用音频数据增强，默认为True
        mask_time_prob=0.05,  # 时间掩码概率，默认为0.05
        mask_time_length=10,  # 时间掩码长度，默认为10
        mask_time_min_masks=2,  # 最小时间掩码数量，默认为2
        mask_feature_prob=0.0,  # 特征掩码概率，默认为0.0
        mask_feature_length=10,  # 特征掩码长度，默认为10
        mask_feature_min_masks=0,  # 最小特征掩码数量，默认为0
        num_codevectors_per_group=320,  # 每组编码向量数量，默认为320
        num_codevector_groups=2,  # 编码向量组数，默认为2
        contrastive_logits_temperature=0.1,  # 对比损失温度参数，默认为0.1
        num_negatives=100,  # 负样本数量，默认为100
        codevector_dim=256,  # 编码向量维度，默认为256
        proj_codevector_dim=256,  # 投影编码向量维度，默认为256
        diversity_loss_weight=0.1,  # 多样性损失权重，默认为0.1
        ctc_loss_reduction="sum",  # CTC损失缩减方式，默认为"sum"
        ctc_zero_infinity=False,  # CTC零无穷，默认为False
        use_weighted_layer_sum=False,  # 是否使用加权层求和，默认为False
        classifier_proj_size=256,  # 分类器投影大小，默认为256
        tdnn_dim=(512, 512, 512, 512, 1500),  # TDNN维度，默认为(512, 512, 512, 512, 1500)
        tdnn_kernel=(5, 3, 3, 1, 1),  # TDNN核大小，默认为(5, 3, 3, 1, 1)
        tdnn_dilation=(1, 2, 3, 1, 1),  # TDNN扩张率，默认为(1, 2, 3, 1, 1)
        xvector_output_dim=512,  # X向量输出维度，默认为512
        pad_token_id=0,  # 填充标记ID，默认为0
        bos_token_id=1,  # 起始标记ID，默认为1
        eos_token_id=2,  # 终止标记ID，默认为2
        add_adapter=False,  # 是否添加适配器，默认为False
        adapter_kernel_size=3,  # 适配器核大小，默认为3
        adapter_stride=2,  # 适配器步长，默认为2
        num_adapter_layers=3,  # 适配器层数，默认为3
        output_hidden_size=None,  # 输出隐藏层大小，默认为None
        position_embeddings_type="relative",  # 位置嵌入类型，默认为"relative"
        rotary_embedding_base=10000,  # 旋转嵌入基数，默认为10000
        max_source_positions=5000,  # 最大源位置数，默认为5000
        conv_depthwise_kernel_size=31,  # 卷积深度可分离核大小，默认为31
        conformer_conv_dropout=0.1,  # Conformer卷积dropout率，默认为0.1
        **kwargs,  # 其他关键字参数
    @property
    def inputs_to_logits_ratio(self):
        return functools.reduce(operator.mul, self.conv_stride, 1)  # 返回卷积步长的累积乘积
```