# `.\transformers\models\wav2vec2_bert\configuration_wav2vec2_bert.py`

```
# 设置文件编码为 utf-8
# 版权声明
# 2024年版权为 The Fairseq Authors 和 The HuggingFace Inc. 团队所有
# 根据 Apache License, Version 2.0 许可
# 除非符合许可证要求，否则不得使用此文件
# 您可以在以下网址获取许可证副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件以"原样"分发
# 没有任何种类的保证或条件，无论是明示的还是暗示的
# 有关特定语言的权限和限制，请参阅许可证
""" Wav2Vec2Bert 模型配置"""

# 导入模块
import functools
import operator

from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取 logger
logger = logging.get_logger(__name__)

# 预训练配置映射
WAV2VEC2_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/w2v-bert-2.0": "https://huggingface.co/facebook/w2v-bert-2.0/resolve/main/config.json",
}

# Wav2Vec2Bert 配置类
class Wav2Vec2BertConfig(PretrainedConfig):
    r"""
    这是用于存储 [`Wav2Vec2BertModel`] 配置的配置类。它用于根据指定的参数实例化 Wav2Vec2Bert 模型，
    定义模型架构。使用默认值实例化配置将产生类似于 Wav2Vec2Bert [facebook/wav2vec2-bert-rel-pos-large]
    (https://huggingface.co/facebook/wav2vec2-bert-rel-pos-large) 架构的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。阅读来自 [`PretrainedConfig`] 的文档以获取更多信息。

    示例:

    ```python
    >>> from transformers import Wav2Vec2BertConfig, Wav2Vec2BertModel

    >>> # 初始化一个 Wav2Vec2Bert facebook/wav2vec2-bert-rel-pos-large 风格的配置
    >>> configuration = Wav2Vec2BertConfig()

    >>> # 从 facebook/wav2vec2-bert-rel-pos-large 风格配置初始化一个模型（带有随机权重）
    >>> model = Wav2Vec2BertModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```"""
    
    # 模型类型为 "wav2vec2-bert"
    model_type = "wav2vec2-bert"
    # 初始化函数，设置模型的各种参数
    def __init__(
        self,
        vocab_size=None, # 词汇量大小，默认为None
        hidden_size=1024, # 隐藏层大小，默认为1024
        num_hidden_layers=24, # 隐藏层层数，默认为24
        num_attention_heads=16, # 注意力头数，默认为16
        intermediate_size=4096, # 中间层大小，默认为4096
        feature_projection_input_dim=160, # 特征投影输入维度，默认为160
        hidden_act="swish", # 隐藏层激活函数，默认为swish
        hidden_dropout=0.0, # 隐藏层dropout概率，默认为0.0
        activation_dropout=0.0, # 激活函数dropout概率，默认为0.0
        attention_dropout=0.0, # 注意力dropout概率，默认为0.0
        feat_proj_dropout=0.0, # 特征投影dropout概率，默认为0.0
        final_dropout=0.1, # 最终输出层dropout概率，默认为0.1
        layerdrop=0.1, # 层dropout概率，默认为0.1
        initializer_range=0.02, # 初始化范围，默认为0.02
        layer_norm_eps=1e-5, # 层归一化系数，默认为1e-5
        apply_spec_augment=True, # 是否使用特定数据增强，默认为True
        mask_time_prob=0.05, # 时间mask概率，默认为0.05
        mask_time_length=10, # 时间mask长度，默认为10
        mask_time_min_masks=2, # 最小时间mask个数，默认为2
        mask_feature_prob=0.0, # 特征mask概率，默认为0.0
        mask_feature_length=10, # 特征mask长度，默认为10
        mask_feature_min_masks=0, # 最小特征mask个数，默认为0
        ctc_loss_reduction="sum", # CTC损失函数减少方式，默认为"sum"
        ctc_zero_infinity=False, # CTC是否将无穷数置为零，默认为False
        use_weighted_layer_sum=False, # 是否使用加权层和，默认为False
        classifier_proj_size=768, # 分类器投影大小，默认为768
        tdnn_dim=(512, 512, 512, 512, 1500), # TDNN网络的维度，默认为(512, 512, 512, 512, 1500)
        tdnn_kernel=(5, 3, 3, 1, 1), # TDNN网络的卷积核大小，默认为(5, 3, 3, 1, 1)
        tdnn_dilation=(1, 2, 3, 1, 1), # TDNN网络的膨胀系数，默认为(1, 2, 3, 1, 1)
        xvector_output_dim=512, # X-Vector输出维度，默认为512
        pad_token_id=0, # 填充标记的ID，默认为0
        bos_token_id=1, # 开始标记的ID，默认为1
        eos_token_id=2, # 结束标记的ID，默认为2
        add_adapter=False, # 是否添加适配器，默认为False
        adapter_kernel_size=3, # 适配器的卷积核大小，默认为3
        adapter_stride=2, # 适配器的步长，默认为2
        num_adapter_layers=1, # 适配器的层数，默认为1
        adapter_act="relu", # 适配器的激活函数，默认为relu
        use_intermediate_ffn_before_adapter=False, # 在适配器��是否使用中间FFN层，默认为False
        output_hidden_size=None, # 输出隐藏层大小，默认为None
        position_embeddings_type="relative_key", # 位置嵌入的类型，默认为"relative_key"
        rotary_embedding_base=10000, # 旋转嵌入的基数，默认为10000
        max_source_positions=5000, # 最大源位置数，默认为5000
        left_max_position_embeddings=64, # 左边最大位置嵌入数，默认为64
        right_max_position_embeddings=8, # 右边最大位置嵌入数，默认为8
        conv_depthwise_kernel_size=31, # 深度卷积核大小，默认为31
        conformer_conv_dropout=0.1, # Conformer卷积层的dropout概率，默认为0.1
        **kwargs, # 其他关键字参数
    @property
    def inputs_to_logits_ratio(self): # 输入到logits的比率属性
        return functools.reduce(operator.mul, self.conv_stride, 1) # 返回输入到logits的比率
``` 
```