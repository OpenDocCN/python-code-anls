# `.\models\layoutlmv2\configuration_layoutlmv2.py`

```py
# coding=utf-8
# Copyright Microsoft Research and The HuggingFace Inc. team. All rights reserved.
# 标明代码文件使用UTF-8编码，版权信息声明
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
# Apache License, Version 2.0授权声明，许可信息的获取网址链接
""" LayoutLMv2 model configuration"""
# LayoutLMv2模型配置

from ...configuration_utils import PretrainedConfig
# 从transformers包中导入PretrainedConfig类
from ...utils import is_detectron2_available, logging
# 从transformers包中导入is_detectron2_available函数和logging模块

logger = logging.get_logger(__name__)
# 获取当前模块的logger对象

LAYOUTLMV2_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "layoutlmv2-base-uncased": "https://huggingface.co/microsoft/layoutlmv2-base-uncased/resolve/main/config.json",
    "layoutlmv2-large-uncased": "https://huggingface.co/microsoft/layoutlmv2-large-uncased/resolve/main/config.json",
    # See all LayoutLMv2 models at https://huggingface.co/models?filter=layoutlmv2
}
# LayoutLMv2预训练模型配置文件映射表，包含模型名称到配置文件URL的映射

# soft dependency
if is_detectron2_available():
    import detectron2
    # 如果detectron2可用，导入detectron2模块

class LayoutLMv2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LayoutLMv2Model`]. It is used to instantiate an
    LayoutLMv2 model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the LayoutLMv2
    [microsoft/layoutlmv2-base-uncased](https://huggingface.co/microsoft/layoutlmv2-base-uncased) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Example:

    ```
    >>> from transformers import LayoutLMv2Config, LayoutLMv2Model

    >>> # Initializing a LayoutLMv2 microsoft/layoutlmv2-base-uncased style configuration
    >>> configuration = LayoutLMv2Config()

    >>> # Initializing a model (with random weights) from the microsoft/layoutlmv2-base-uncased style configuration
    >>> model = LayoutLMv2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""
    # LayoutLMv2配置类，用于存储[`LayoutLMv2Model`]的配置，根据指定的参数实例化LayoutLMv2模型，定义模型架构。
    # 使用默认配置实例化将产生类似于LayoutLMv2 [microsoft/layoutlmv2-base-uncased](https://huggingface.co/microsoft/layoutlmv2-base-uncased)架构的配置。

    model_type = "layoutlmv2"
    # 模型类型为"layoutlmv2"
    # 定义一个初始化方法，用于初始化一个新的对象实例
    def __init__(
        self,
        vocab_size=30522,  # 词汇表大小，默认为30522
        hidden_size=768,  # 隐藏层大小，默认为768
        num_hidden_layers=12,  # 隐藏层的数量，默认为12
        num_attention_heads=12,  # 注意力头的数量，默认为12
        intermediate_size=3072,  # 中间层大小，默认为3072
        hidden_act="gelu",  # 隐藏层激活函数，默认为GELU
        hidden_dropout_prob=0.1,  # 隐藏层的dropout概率，默认为0.1
        attention_probs_dropout_prob=0.1,  # 注意力概率的dropout概率，默认为0.1
        max_position_embeddings=512,  # 最大位置编码数，默认为512
        type_vocab_size=2,  # 类型词汇表大小，默认为2
        initializer_range=0.02,  # 初始化范围，默认为0.02
        layer_norm_eps=1e-12,  # 层归一化的epsilon，默认为1e-12
        pad_token_id=0,  # 填充标记ID，默认为0
        max_2d_position_embeddings=1024,  # 最大二维位置编码数，默认为1024
        max_rel_pos=128,  # 最大相对位置，默认为128
        rel_pos_bins=32,  # 相对位置的bins数，默认为32
        fast_qkv=True,  # 是否使用快速的QKV计算，默认为True
        max_rel_2d_pos=256,  # 最大二维相对位置，默认为256
        rel_2d_pos_bins=64,  # 二维相对位置的bins数，默认为64
        convert_sync_batchnorm=True,  # 是否转换同步批归一化，默认为True
        image_feature_pool_shape=[7, 7, 256],  # 图像特征池形状，默认为[7, 7, 256]
        coordinate_size=128,  # 坐标大小，默认为128
        shape_size=128,  # 形状大小，默认为128
        has_relative_attention_bias=True,  # 是否具有相对注意力偏置，默认为True
        has_spatial_attention_bias=True,  # 是否具有空间注意力偏置，默认为True
        has_visual_segment_embedding=False,  # 是否具有视觉段嵌入，默认为False
        detectron2_config_args=None,  # detectron2配置参数，默认为None
        **kwargs,  # 其他关键字参数
    ):
        # 调用父类的初始化方法，传入相关参数
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            pad_token_id=pad_token_id,
            **kwargs,
        )
        # 初始化对象的特有属性
        self.max_2d_position_embeddings = max_2d_position_embeddings
        self.max_rel_pos = max_rel_pos
        self.rel_pos_bins = rel_pos_bins
        self.fast_qkv = fast_qkv
        self.max_rel_2d_pos = max_rel_2d_pos
        self.rel_2d_pos_bins = rel_2d_pos_bins
        self.convert_sync_batchnorm = convert_sync_batchnorm
        self.image_feature_pool_shape = image_feature_pool_shape
        self.coordinate_size = coordinate_size
        self.shape_size = shape_size
        self.has_relative_attention_bias = has_relative_attention_bias
        self.has_spatial_attention_bias = has_spatial_attention_bias
        self.has_visual_segment_embedding = has_visual_segment_embedding
        # 如果detectron2_config_args不是None，则使用给定的配置参数；否则使用默认的detectron2配置参数
        self.detectron2_config_args = (
            detectron2_config_args if detectron2_config_args is not None else self.get_default_detectron2_config()
        )

    @classmethod
    # 返回一个包含默认参数的字典，用于配置Detectron2模型
    def get_default_detectron2_config(self):
        return {
            "MODEL.MASK_ON": True,  # 开启模型的遮罩功能
            "MODEL.PIXEL_STD": [57.375, 57.120, 58.395],  # 图像每个通道的像素标准偏差
            "MODEL.BACKBONE.NAME": "build_resnet_fpn_backbone",  # 使用的主干网络名称
            "MODEL.FPN.IN_FEATURES": ["res2", "res3", "res4", "res5"],  # 特征金字塔网络的输入特征层
            "MODEL.ANCHOR_GENERATOR.SIZES": [[32], [64], [128], [256], [512]],  # 锚点生成器的大小
            "MODEL.RPN.IN_FEATURES": ["p2", "p3", "p4", "p5", "p6"],  # 区域生成网络的输入特征层
            "MODEL.RPN.PRE_NMS_TOPK_TRAIN": 2000,  # RPN训练时的NMS前TopK
            "MODEL.RPN.PRE_NMS_TOPK_TEST": 1000,  # RPN测试时的NMS前TopK
            "MODEL.RPN.POST_NMS_TOPK_TRAIN": 1000,  # RPN训练后的NMS后TopK
            "MODEL.POST_NMS_TOPK_TEST": 1000,  # 测试时的NMS后TopK
            "MODEL.ROI_HEADS.NAME": "StandardROIHeads",  # 区域兴趣头部的名称
            "MODEL.ROI_HEADS.NUM_CLASSES": 5,  # 区域兴趣头部的类别数量
            "MODEL.ROI_HEADS.IN_FEATURES": ["p2", "p3", "p4", "p5"],  # 区域兴趣头部的输入特征层
            "MODEL.ROI_BOX_HEAD.NAME": "FastRCNNConvFCHead",  # 区域兴趣框头部的名称
            "MODEL.ROI_BOX_HEAD.NUM_FC": 2,  # 区域兴趣框头部全连接层的数量
            "MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION": 14,  # 区域兴趣框头部的池化分辨率
            "MODEL.ROI_MASK_HEAD.NAME": "MaskRCNNConvUpsampleHead",  # 区域兴趣遮罩头部的名称
            "MODEL.ROI_MASK_HEAD.NUM_CONV": 4,  # 区域兴趣遮罩头部的卷积层数量
            "MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION": 7,  # 区域兴趣遮罩头部的池化分辨率
            "MODEL.RESNETS.DEPTH": 101,  # ResNet主干网络的深度
            "MODEL.RESNETS.SIZES": [[32], [64], [128], [256], [512]],  # ResNet主干网络中的尺寸
            "MODEL.RESNETS.ASPECT_RATIOS": [[0.5, 1.0, 2.0]],  # ResNet主干网络中的长宽比
            "MODEL.RESNETS.OUT_FEATURES": ["res2", "res3", "res4", "res5"],  # ResNet主干网络中的输出特征层
            "MODEL.RESNETS.NUM_GROUPS": 32,  # ResNet主干网络中的组数
            "MODEL.RESNETS.WIDTH_PER_GROUP": 8,  # ResNet主干网络中每组的宽度
            "MODEL.RESNETS.STRIDE_IN_1X1": False,  # ResNet主干网络中1x1卷积是否采用stride
        }

    # 返回配置好的Detectron2模型配置
    def get_detectron2_config(self):
        # 调用Detectron2库的函数获取一个空的配置对象
        detectron2_config = detectron2.config.get_cfg()
        # 遍历传入的参数字典
        for k, v in self.detectron2_config_args.items():
            # 按照点分割键，用来设置配置对象中的属性
            attributes = k.split(".")
            to_set = detectron2_config
            # 通过反射设置配置对象中的属性
            for attribute in attributes[:-1]:
                to_set = getattr(to_set, attribute)
            setattr(to_set, attributes[-1], v)

        # 返回配置好的Detectron2模型配置对象
        return detectron2_config
```