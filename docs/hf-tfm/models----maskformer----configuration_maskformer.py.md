# `.\models\maskformer\configuration_maskformer.py`

```
# coding=utf-8
# Copyright 2022 Meta Platforms, Inc.and The HuggingFace Inc. team. All rights reserved.
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
""" MaskFormer model configuration"""
from typing import Dict, Optional

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto import CONFIG_MAPPING
from ..detr import DetrConfig
from ..swin import SwinConfig


MASKFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/maskformer-swin-base-ade": (
        "https://huggingface.co/facebook/maskformer-swin-base-ade/blob/main/config.json"
    )
    # See all MaskFormer models at https://huggingface.co/models?filter=maskformer
}

# 获取全局日志记录器实例
logger = logging.get_logger(__name__)


class MaskFormerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MaskFormerModel`]. It is used to instantiate a
    MaskFormer model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the MaskFormer
    [facebook/maskformer-swin-base-ade](https://huggingface.co/facebook/maskformer-swin-base-ade) architecture trained
    on [ADE20k-150](https://huggingface.co/datasets/scene_parse_150).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Currently, MaskFormer only supports the [Swin Transformer](swin) as backbone.
    # 定义 MaskFormerConfig 类，用于配置 MaskFormerModel 模型的参数
    class MaskFormerConfig:
        # 控制掩码特征的大小，默认为 256
        mask_feature_size (`int`, *optional*, defaults to 256):
            The masks' features size, this value will also be used to specify the Feature Pyramid Network features'
            size.
        
        # 控制无物体类别的权重，默认为 0.1
        no_object_weight (`float`, *optional*, defaults to 0.1):
            Weight to apply to the null (no object) class.
        
        # 是否使用辅助损失，默认为 False
        use_auxiliary_loss(`bool`, *optional*, defaults to `False`):
            If `True` [`MaskFormerForInstanceSegmentationOutput`] will contain the auxiliary losses computed using the
            logits from each decoder's stage.
        
        # 如果未设置 backbone_config，则使用默认配置 `swin-base-patch4-window12-384` 的配置
        backbone_config (`Dict`, *optional*):
            The configuration passed to the backbone, if unset, the configuration corresponding to
            `swin-base-patch4-window12-384` will be used.
        
        # 当 backbone_config 为 None 时，使用此参数指定要使用的骨干网络名称
        backbone (`str`, *optional*):
            Name of backbone to use when `backbone_config` is `None`. If `use_pretrained_backbone` is `True`, this
            will load the corresponding pretrained weights from the timm or transformers library. If `use_pretrained_backbone`
            is `False`, this loads the backbone's config and uses that to initialize the backbone with random weights.
        
        # 是否使用预训练的骨干网络权重，默认为 False
        use_pretrained_backbone (`bool`, *optional*, `False`):
            Whether to use pretrained weights for the backbone.
        
        # 是否从 timm 库中加载 backbone，默认为 False
        use_timm_backbone (`bool`, *optional*, `False`):
            Whether to load `backbone` from the timm library. If `False`, the backbone is loaded from the transformers
            library.
        
        # 当从检查点加载时，传递给 AutoBackbone 的关键字参数，例如 `{'out_indices': (0, 1, 2, 3)}`
        backbone_kwargs (`dict`, *optional*):
            Keyword arguments to be passed to AutoBackbone when loading from a checkpoint
            e.g. `{'out_indices': (0, 1, 2, 3)}`. Cannot be specified if `backbone_config` is set.
        
        # 配置传递给变换器解码模型的参数，如果未设置，则使用 `detr-resnet-50` 的基本配置
        decoder_config (`Dict`, *optional*):
            The configuration passed to the transformer decoder model, if unset the base config for `detr-resnet-50`
            will be used.
        
        # 初始化所有权重矩阵的截断正态初始化器的标准差，默认为 0.02
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        
        # HM Attention map 模块中用于 Xavier 初始化增益的缩放因子，默认为 1
        init_xavier_std (`float`, *optional*, defaults to 1):
            The scaling factor used for the Xavier initialization gain in the HM Attention map module.
        
        # Dice 损失的权重，默认为 1.0
        dice_weight (`float`, *optional*, defaults to 1.0):
            The weight for the dice loss.
        
        # 交叉熵损失的权重，默认为 1.0
        cross_entropy_weight (`float`, *optional*, defaults to 1.0):
            The weight for the cross entropy loss.
        
        # 掩码损失的权重，默认为 20.0
        mask_weight (`float`, *optional*, defaults to 20.0):
            The weight for the mask loss.
        
        # 模型是否输出其辅助 logits，默认未指定
        output_auxiliary_logits (`bool`, *optional*):
            Should the model output its `auxiliary_logits` or not.
    
    # 当所选的骨干模型类型不在 `["swin"]` 中或解码器模型类型不在 `["detr"]` 中时，引发 `ValueError`
    Raises:
        `ValueError`:
            Raised if the backbone model type selected is not in `["swin"]` or the decoder model type selected is not
            in `["detr"]`
    
    Examples:
    
    # 从 transformers 库导入 MaskFormerConfig 和 MaskFormerModel 类
    >>> from transformers import MaskFormerConfig, MaskFormerModel
    # Initializing a MaskFormer configuration object using default values
    configuration = MaskFormerConfig()
    
    # Initializing a MaskFormerModel object with the specified configuration, initially with random weights
    model = MaskFormerModel(configuration)
    
    # Accessing the configuration of the model instance
    configuration = model.config
```