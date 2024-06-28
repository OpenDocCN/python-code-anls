# `.\models\vit_mae\configuration_vit_mae.py`

```
# coding=utf-8
# Copyright 2022 Facebook AI and The HuggingFace Inc. team. All rights reserved.
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
""" ViT MAE model configuration"""

# 导入必要的模块和函数
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 预训练配置文件的映射表，指定每个预训练模型对应的配置文件的 URL
VIT_MAE_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/vit-mae-base": "https://huggingface.co/facebook/vit-mae-base/resolve/main/config.json",
    # 可以查看所有 ViT MAE 模型的列表：https://huggingface.co/models?filter=vit-mae
}

# ViTMAEConfig 类，继承自 PretrainedConfig 类
class ViTMAEConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ViTMAEModel`]. It is used to instantiate an ViT
    MAE model according to the specified arguments, defining the model architecture. Instantiating a configuration with
    the defaults will yield a similar configuration to that of the ViT
    [facebook/vit-mae-base](https://huggingface.co/facebook/vit-mae-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """
    # 隐藏层的维度，包括编码器层和池化层
    Args:
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        
        # Transformer 编码器中隐藏层的数量
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        
        # Transformer 编码器中每个注意力层的注意力头数
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        
        # Transformer 编码器中"中间"（即前馈）层的维度
        intermediate_size (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the Transformer encoder.
        
        # 编码器和池化器中的非线性激活函数
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"selu"` and `"gelu_new"` are supported.
        
        # 嵌入层、编码器和池化器中所有全连接层的 dropout 概率
        hidden_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        
        # 注意力概率的 dropout 比率
        attention_probs_dropout_prob (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        
        # 初始化所有权重矩阵的截断正态分布的标准差
        initializer_range (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        
        # 层归一化层使用的 epsilon 值
        layer_norm_eps (`float`, *optional*, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        
        # 每个图像的大小（分辨率）
        image_size (`int`, *optional*, defaults to 224):
            The size (resolution) of each image.
        
        # 每个图像块（patch）的大小（分辨率）
        patch_size (`int`, *optional*, defaults to 16):
            The size (resolution) of each patch.
        
        # 输入通道的数量
        num_channels (`int`, *optional*, defaults to 3):
            The number of input channels.
        
        # 是否为查询、键和值添加偏置
        qkv_bias (`bool`, *optional*, defaults to `True`):
            Whether to add a bias to the queries, keys and values.
        
        # 解码器中每个注意力层的注意力头数
        decoder_num_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the decoder.
        
        # 解码器的维度
        decoder_hidden_size (`int`, *optional*, defaults to 512):
            Dimensionality of the decoder.
        
        # 解码器中隐藏层的数量
        decoder_num_hidden_layers (`int`, *optional*, defaults to 8):
            Number of hidden layers in the decoder.
        
        # 解码器中"中间"（即前馈）层的维度
        decoder_intermediate_size (`int`, *optional*, defaults to 2048):
            Dimensionality of the "intermediate" (i.e., feed-forward) layer in the decoder.
        
        # 输入序列中掩码标记的比例
        mask_ratio (`float`, *optional*, defaults to 0.75):
            The ratio of the number of masked tokens in the input sequence.
        
        # 是否使用归一化像素进行训练
        norm_pix_loss (`bool`, *optional*, defaults to `False`):
            Whether or not to train with normalized pixels (see Table 3 in the paper). Using normalized pixels improved
            representation quality in the experiments of the authors.
    >>> configuration = ViTMAEConfig()
    
    >>> # 初始化一个模型（带有随机权重），使用 vit-mae-base 风格的配置
    >>> model = ViTMAEModel(configuration)
    
    >>> # 访问模型的配置信息
    >>> configuration = model.config
```