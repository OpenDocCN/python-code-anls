# `.\models\wav2vec2_bert\configuration_wav2vec2_bert.py`

```
# coding=utf-8
# Copyright 2024 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Wav2Vec2Bert model configuration"""


# 导入预训练配置类 PretrainedConfig 和日志工具 logging
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取logger对象，用于记录日志信息
logger = logging.get_logger(__name__)

# 定义预训练配置文件的映射字典，指定预训练模型名称和其配置文件的下载链接
WAV2VEC2_BERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/w2v-bert-2.0": "https://huggingface.co/facebook/w2v-bert-2.0/resolve/main/config.json",
}

# 定义 Wav2Vec2BertConfig 类，继承自 PretrainedConfig
class Wav2Vec2BertConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Wav2Vec2BertModel`]. It is used to
    instantiate an Wav2Vec2Bert model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the Wav2Vec2Bert
    [facebook/wav2vec2-bert-rel-pos-large](https://huggingface.co/facebook/wav2vec2-bert-rel-pos-large)
    architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Example:

    ```python
    >>> from transformers import Wav2Vec2BertConfig, Wav2Vec2BertModel

    >>> # Initializing a Wav2Vec2Bert facebook/wav2vec2-bert-rel-pos-large style configuration
    >>> configuration = Wav2Vec2BertConfig()

    >>> # Initializing a model (with random weights) from the facebook/wav2vec2-bert-rel-pos-large style configuration
    >>> model = Wav2Vec2BertModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """

    # 定义模型类型为 "wav2vec2-bert"
    model_type = "wav2vec2-bert"
    # 初始化函数，用于初始化一个类实例
    def __init__(
        self,
        vocab_size=None,  # 词汇表大小，默认为 None
        hidden_size=1024,  # 隐藏层大小，默认为 1024
        num_hidden_layers=24,  # 隐藏层数，默认为 24
        num_attention_heads=16,  # 注意力头数，默认为 16
        intermediate_size=4096,  # 中间层大小，默认为 4096
        feature_projection_input_dim=160,  # 特征投影输入维度，默认为 160
        hidden_act="swish",  # 隐藏层激活函数，默认为 "swish"
        hidden_dropout=0.0,  # 隐藏层的 dropout 概率，默认为 0.0
        activation_dropout=0.0,  # 激活函数的 dropout 概率，默认为 0.0
        attention_dropout=0.0,  # 注意力机制的 dropout 概率，默认为 0.0
        feat_proj_dropout=0.0,  # 特征投影的 dropout 概率，默认为 0.0
        final_dropout=0.1,  # 最终输出的 dropout 概率，默认为 0.1
        layerdrop=0.1,  # 层级丢弃的概率，默认为 0.1
        initializer_range=0.02,  # 初始化范围，默认为 0.02
        layer_norm_eps=1e-5,  # 层归一化的 epsilon，默认为 1e-5
        apply_spec_augment=True,  # 是否应用语音数据增强，默认为 True
        mask_time_prob=0.05,  # 时间掩码概率，默认为 0.05
        mask_time_length=10,  # 时间掩码长度，默认为 10
        mask_time_min_masks=2,  # 时间掩码的最小数量，默认为 2
        mask_feature_prob=0.0,  # 特征掩码概率，默认为 0.0
        mask_feature_length=10,  # 特征掩码长度，默认为 10
        mask_feature_min_masks=0,  # 特征掩码的最小数量，默认为 0
        ctc_loss_reduction="sum",  # CTC 损失函数的减少方式，默认为 "sum"
        ctc_zero_infinity=False,  # CTC 损失函数中是否将无限值作为零处理，默认为 False
        use_weighted_layer_sum=False,  # 是否使用加权层总和，默认为 False
        classifier_proj_size=768,  # 分类器投影大小，默认为 768
        tdnn_dim=(512, 512, 512, 512, 1500),  # TDNN 层维度，默认为 (512, 512, 512, 512, 1500)
        tdnn_kernel=(5, 3, 3, 1, 1),  # TDNN 层卷积核大小，默认为 (5, 3, 3, 1, 1)
        tdnn_dilation=(1, 2, 3, 1, 1),  # TDNN 层膨胀率，默认为 (1, 2, 3, 1, 1)
        xvector_output_dim=512,  # x-vector 输出维度，默认为 512
        pad_token_id=0,  # 填充 token 的 ID，默认为 0
        bos_token_id=1,  # 开始 token 的 ID，默认为 1
        eos_token_id=2,  # 结束 token 的 ID，默认为 2
        add_adapter=False,  # 是否添加适配器层，默认为 False
        adapter_kernel_size=3,  # 适配器层的卷积核大小，默认为 3
        adapter_stride=2,  # 适配器层的步幅，默认为 2
        num_adapter_layers=1,  # 适配器层数量，默认为 1
        adapter_act="relu",  # 适配器层的激活函数，默认为 "relu"
        use_intermediate_ffn_before_adapter=False,  # 是否在适配器层之前使用中间的 Feed Forward 层，默认为 False
        output_hidden_size=None,  # 输出的隐藏层大小，默认为 None
        position_embeddings_type="relative_key",  # 位置嵌入的类型，默认为 "relative_key"
        rotary_embedding_base=10000,  # 旋转嵌入的基础值，默认为 10000
        max_source_positions=5000,  # 最大源位置，默认为 5000
        left_max_position_embeddings=64,  # 左侧最大位置嵌入数，默认为 64
        right_max_position_embeddings=8,  # 右侧最大位置嵌入数，默认为 8
        conv_depthwise_kernel_size=31,  # 深度卷积的卷积核大小，默认为 31
        conformer_conv_dropout=0.1,  # Conformer 模型的卷积层 dropout 概率，默认为 0.1
        **kwargs,  # 其他参数，以字典形式接收
    ):
    
    @property
    # 计算输入特征到 logits 的比率
    def inputs_to_logits_ratio(self):
        # 计算 ratio 为特征投影输入维度的两倍
        ratio = self.feature_projection_input_dim * 2
        # 如果添加了适配器，则乘以适配器步幅的适配器层数量次方
        if self.add_adapter:
            ratio = ratio * (self.adapter_stride**self.num_adapter_layers)
        return ratio
```