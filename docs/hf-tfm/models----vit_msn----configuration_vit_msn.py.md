# `.\transformers\models\vit_msn\configuration_vit_msn.py`

```py
# coding=utf-8
# 版权 2022 年 Facebook AI 和 HuggingFace Inc. 团队保留所有权利。
#
# 根据 Apache 许可证 2.0 版（“许可证”）获得许可;
# 除非符合许可证的要求，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件
# 按“原样”分发，不提供任何形式的担保或条件，明示或暗示。
# 有关许可证的详细信息，请参阅许可证。
""" ViT MSN 模型配置"""


from ...configuration_utils import PretrainedConfig  # 导入预训练配置类
from ...utils import logging  # 导入日志工具


logger = logging.get_logger(__name__)  # 获取日志记录器

VIT_MSN_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "sayakpaul/vit-msn-base": "https://huggingface.co/sayakpaul/vit-msn-base/resolve/main/config.json",
    # 查看所有 ViT MSN 模型，请访问 https://huggingface.co/models?filter=vit_msn
}


class ViTMSNConfig(PretrainedConfig):
    r"""
    这是用于存储 [`ViTMSNModel`] 配置的配置类。根据指定的参数，定义模型架构。使用默认参数实例化配置将产生类似于 ViT
    [facebook/vit_msn_base](https://huggingface.co/facebook/vit_msn_base) 架构的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。请阅读 [`PretrainedConfig`] 的文档以获取更多信息。
    # 定义一个类变量，表示模型类型为 Vision Transformer with Multi-Scale Network (ViT-MSN)
    model_type = "vit_msn"

    # 初始化 ViT-MSN 模型的配置参数
    def __init__(
        # 编码器层和池化层的维度大小，默认为 768
        self,
        hidden_size=768,
        # Transformer 编码器中的隐藏层数量，默认为 12
        num_hidden_layers=12,
        # Transformer 编码器中每个注意力层的注意力头数量，默认为 12
        num_attention_heads=12,
        # Transformer 编码器中“中间层”（即前馈层）的维度大小，默认为 3072
        intermediate_size=3072,
        # 编码器和池化层中的非线性激活函数，默认为 "gelu"
        hidden_act="gelu",
        # 嵌入层、编码器和池化层中所有全连接层的 dropout 概率，默认为 0.0
        hidden_dropout_prob=0.0,
        # 注意力概率的 dropout 比率，默认为 0.0
        attention_probs_dropout_prob=0.0,
        # 用于初始化所有权重矩阵的截断正态分布初始化器的标准差，默认为 0.02
        initializer_range=0.02,
        # 层归一化层使用的 epsilon，默认为 1e-06
        layer_norm_eps=1e-06,
        # 每个图像的尺寸（分辨率），默认为 224
        image_size=224,
        # 每个图像块（patch）的尺寸（分辨率），默认为 16
        patch_size=16,
        # 输入通道的数量，默认为 3
        num_channels=3,
        # 是否对查询、键和值添加偏置，默认为 True
        qkv_bias=True,
        # 允许传递额外的关键字参数
        **kwargs,
    # 调用父类的构造函数，并传递参数
    super().__init__(**kwargs)
    
    # 设置隐藏层大小
    self.hidden_size = hidden_size
    
    # 设置隐藏层数量
    self.num_hidden_layers = num_hidden_layers
    
    # 设置注意力头的数量
    self.num_attention_heads = num_attention_heads
    
    # 设置中间层的大小
    self.intermediate_size = intermediate_size
    
    # 设置隐藏层的激活函数
    self.hidden_act = hidden_act
    
    # 设置隐藏层的dropout概率
    self.hidden_dropout_prob = hidden_dropout_prob
    
    # 设置注意力概率的dropout概率
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    
    # 初始化范围
    self.initializer_range = initializer_range
    
    # 设置层归一化的epsilon值
    self.layer_norm_eps = layer_norm_eps
    
    # 设置图像的大小
    self.image_size = image_size
    
    # 设置patch的大小
    self.patch_size = patch_size
    
    # 设置通道数量
    self.num_channels = num_channels
    
    # 设置qkv是否带偏置
    self.qkv_bias = qkv_bias
```