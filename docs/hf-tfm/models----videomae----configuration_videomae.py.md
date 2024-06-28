# `.\models\videomae\configuration_videomae.py`

```
# coding=utf-8
# 上面这行指定了文件的编码格式为UTF-8，确保支持非英语字符
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
# 版权声明，版权归HuggingFace Inc.团队所有，保留所有权利
#
# Licensed under the Apache License, Version 2.0 (the "License");
# 根据 Apache License, Version 2.0 许可证进行许可，允许使用、复制、修改、合并、发布、分发、再授权和销售
# you may not use this file except in compliance with the License.
# 除非符合许可证要求，否则不得使用此文件
# You may obtain a copy of the License at
# 可以在以下网址获取许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# 除非适用法律要求或书面同意，否则不得更改
# distributed under the License is distributed on an "AS IS" BASIS,
# 根据许可证发布的软件是按“原样”提供的，没有任何形式的明示或暗示保证或条件
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 没有明示或暗示的任何保证或条件
# See the License for the specific language governing permissions and
# 详细信息请参阅许可证
# limitations under the License.
# 许可证下的限制
""" VideoMAE model configuration"""

# 从相对路径中导入预训练配置
from ...configuration_utils import PretrainedConfig
# 从相对路径中导入日志记录工具
from ...utils import logging

# 获取与当前模块相关的日志记录器
logger = logging.get_logger(__name__)

# 预训练模型配置文件的映射字典，将模型名称映射到其配置文件的URL
VIDEOMAE_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "MCG-NJU/videomae-base": "https://huggingface.co/MCG-NJU/videomae-base/resolve/main/config.json",
}

# VideoMAE模型的配置类，用于存储VideoMAEModel的配置信息
class VideoMAEConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`VideoMAEModel`]. It is used to instantiate a
    VideoMAE model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the VideoMAE
    [MCG-NJU/videomae-base](https://huggingface.co/MCG-NJU/videomae-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """
    # 定义一个类，表示一个视觉变换的模型配置
    class VisionTransformerConfig:
        def __init__(
            self,
            image_size=224,
            patch_size=16,
            num_channels=3,
            num_frames=16,
            tubelet_size=2,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            initializer_range=0.02,
            layer_norm_eps=1e-12,
            qkv_bias=True,
            use_mean_pooling=True,
            decoder_num_attention_heads=6,
            decoder_hidden_size=384,
            decoder_num_hidden_layers=4,
            decoder_intermediate_size=1536,
            norm_pix_loss=True
        ):
            # 设置图像大小
            self.image_size = image_size
            # 设置每个补丁的大小
            self.patch_size = patch_size
            # 设置输入通道数
            self.num_channels = num_channels
            # 设置每个视频中的帧数
            self.num_frames = num_frames
            # 设置管道大小
            self.tubelet_size = tubelet_size
            # 设置编码器层和汇集层的维度
            self.hidden_size = hidden_size
            # 设置Transformer编码器中的隐藏层数
            self.num_hidden_layers = num_hidden_layers
            # 设置Transformer编码器中每个注意力层的注意头数
            self.num_attention_heads = num_attention_heads
            # 设置Transformer编码器中"中间"（即前馈）层的维度
            self.intermediate_size = intermediate_size
            # 设置编码器和汇集器中的非线性激活函数
            self.hidden_act = hidden_act
            # 设置所有全连接层的dropout概率
            self.hidden_dropout_prob = hidden_dropout_prob
            # 设置注意力概率的dropout比率
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            # 设置初始化所有权重矩阵的截断正态分布的标准差
            self.initializer_range = initializer_range
            # 设置层归一化层使用的epsilon
            self.layer_norm_eps = layer_norm_eps
            # 设置是否向查询、键和值添加偏置
            self.qkv_bias = qkv_bias
            # 设置是否使用均值池化最终隐藏状态，而不是使用[CLS]标记的最终隐藏状态
            self.use_mean_pooling = use_mean_pooling
            # 设置解码器中每个注意力层的注意头数
            self.decoder_num_attention_heads = decoder_num_attention_heads
            # 设置解码器的隐藏大小
            self.decoder_hidden_size = decoder_hidden_size
            # 设置解码器中的隐藏层数
            self.decoder_num_hidden_layers = decoder_num_hidden_layers
            # 设置解码器中"中间"（即前馈）层的维度
            self.decoder_intermediate_size = decoder_intermediate_size
            # 设置是否归一化目标补丁像素
            self.norm_pix_loss = norm_pix_loss
    # 导入VideoMAEConfig和VideoMAEModel类
    >>> from transformers import VideoMAEConfig, VideoMAEModel

    # 初始化一个VideoMAE videomae-base风格的配置对象
    >>> configuration = VideoMAEConfig()

    # 根据配置对象随机初始化一个模型
    >>> model = VideoMAEModel(configuration)

    # 获取模型的配置信息
    >>> configuration = model.config
```