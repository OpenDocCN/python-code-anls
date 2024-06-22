# `.\models\deta\configuration_deta.py`

```py
# coding=utf-8
# Copyright 2022 SenseTime and The HuggingFace Inc. team. All rights reserved.
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
""" DETA model configuration"""


from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto import CONFIG_MAPPING


logger = logging.get_logger(__name__)

# DETA 预训练模型配置文件的下载地址映射
DETA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "ut/deta": "https://huggingface.co/ut/deta/resolve/main/config.json",
}


class DetaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DetaModel`]. It is used to instantiate a DETA
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the DETA
    [SenseTime/deformable-detr](https://huggingface.co/SenseTime/deformable-detr) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Examples:

    ```python
    >>> from transformers import DetaConfig, DetaModel

    >>> # Initializing a DETA SenseTime/deformable-detr style configuration
    >>> configuration = DetaConfig()

    >>> # Initializing a model (with random weights) from the SenseTime/deformable-detr style configuration
    >>> model = DetaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```py"""

    # 模型类型
    model_type = "deta"
    # 属性映射字典，用于将预训练模型的配置属性映射到公共属性名称
    attribute_map = {
        "hidden_size": "d_model",
        "num_attention_heads": "encoder_attention_heads",
    }
    # 初始化函数，用于初始化模型参数
    def __init__(
        # backbone_config为预训练模型的配置文件，默认为None
        backbone_config=None,
        # 查询数量，默认为900
        num_queries=900,
        # 最大位置嵌入数，默认为2048
        max_position_embeddings=2048,
        # 编码器层数，默认为6
        encoder_layers=6,
        # 编码器中FFN层维度，默认为2048
        encoder_ffn_dim=2048,
        # 编码器中注意力头数，默认为8
        encoder_attention_heads=8,
        # 解码器层数，默认为6
        decoder_layers=6,
        # 解码器中FFN层维度，默认为1024
        decoder_ffn_dim=1024,
        # 解码器中注意力头数，默认为8
        decoder_attention_heads=8,
        # 编码器层级丢弃率，默认为0.0
        encoder_layerdrop=0.0,
        # 是否是编码器-解码器结构，默认为True
        is_encoder_decoder=True,
        # 激活函数，默认为"relu"
        activation_function="relu",
        # 模型维度，默认为256
        d_model=256,
        # 丢弃率，默认为0.1
        dropout=0.1,
        # 注意力丢弃率，默认为0.0
        attention_dropout=0.0,
        # 激活函数丢弃率，默认为0.0
        activation_dropout=0.0,
        # 初始化标准差，默认为0.02
        init_std=0.02,
        # 初始化Xavier标准差，默认为1.0
        init_xavier_std=1.0,
        # 是否返回中间结果，默认为True
        return_intermediate=True,
        # 是否进行辅助损失，默认为False
        auxiliary_loss=False,
        # 位置嵌入类型，默认为"sine"
        position_embedding_type="sine",
        # 特征级别数量，默认为5
        num_feature_levels=5,
        # 编码器点数，默认为4
        encoder_n_points=4,
        # 解码器点数，默认为4
        decoder_n_points=4,
        # 是否进行两阶段模式，默认为True
        two_stage=True,
        # 两阶段模式的提议数量，默认为300
        two_stage_num_proposals=300,
        # 是否进行框盒精化，默认为True
        with_box_refine=True,
        # 是否进行第一阶段分配，默认为True
        assign_first_stage=True,
        # 是否进行第二阶段分配，默认为True
        assign_second_stage=True,
        # 类损失，默认为1
        class_cost=1,
        # 边界框损失，默认为5
        bbox_cost=5,
        # GIOU损失，默认为2
        giou_cost=2,
        # 掩码损失系数，默认为1
        mask_loss_coefficient=1,
        # Dice损失系数，默认为1
        dice_loss_coefficient=1,
        # 边界框损失系数，默认为5
        bbox_loss_coefficient=5,
        # GIOU损失系数，默认为2
        giou_loss_coefficient=2,
        # 结束系数，默认为0.1
        eos_coefficient=0.1,
        # 焦点Alpha，默认为0.25
        focal_alpha=0.25,
        # **kwargs，其他参数
        **kwargs,
    # 检查是否未提供 backbone_config
    ):
        # 如果没有提供 backbone_config，则使用默认的 ResNet backbone 进行初始化
        logger.info("`backbone_config` is `None`. Initializing the config with the default `ResNet` backbone.")
        backbone_config = CONFIG_MAPPING["resnet"](out_features=["stage2", "stage3", "stage4"])
    else:
        # 如果提供了 backbone_config，并且为字典类型，则根据 model_type 创建对应的配置类
        if isinstance(backbone_config, dict):
            backbone_model_type = backbone_config.pop("model_type")
            config_class = CONFIG_MAPPING[backbone_model_type]
            backbone_config = config_class.from_dict(backbone_config)

    # 设置模型配置参数
    self.backbone_config = backbone_config
    self.num_queries = num_queries
    self.max_position_embeddings = max_position_embeddings
    self.d_model = d_model
    self.encoder_ffn_dim = encoder_ffn_dim
    self.encoder_layers = encoder_layers
    self.encoder_attention_heads = encoder_attention_heads
    self.decoder_ffn_dim = decoder_ffn_dim
    self.decoder_layers = decoder_layers
    self.decoder_attention_heads = decoder_attention_heads
    self.dropout = dropout
    self.attention_dropout = attention_dropout
    self.activation_dropout = activation_dropout
    self.activation_function = activation_function
    self.init_std = init_std
    self.init_xavier_std = init_xavier_std
    self.encoder_layerdrop = encoder_layerdrop
    self.auxiliary_loss = auxiliary_loss
    self.position_embedding_type = position_embedding_type
    # 设置 deformable 属性
    self.num_feature_levels = num_feature_levels
    self.encoder_n_points = encoder_n_points
    self.decoder_n_points = decoder_n_points
    self.two_stage = two_stage
    self.two_stage_num_proposals = two_stage_num_proposals
    self.with_box_refine = with_box_refine
    self.assign_first_stage = assign_first_stage
    self.assign_second_stage = assign_second_stage
    # 检查两阶段模型的配置是否正确
    if two_stage is True and with_box_refine is False:
        raise ValueError("If two_stage is True, with_box_refine must be True.")
    # 设置 Hungarian matcher 相关参数
    self.class_cost = class_cost
    self.bbox_cost = bbox_cost
    self.giou_cost = giou_cost
    # 设置损失函数的系数
    self.mask_loss_coefficient = mask_loss_coefficient
    self.dice_loss_coefficient = dice_loss_coefficient
    self.bbox_loss_coefficient = bbox_loss_coefficient
    self.giou_loss_coefficient = giou_loss_coefficient
    self.eos_coefficient = eos_coefficient
    self.focal_alpha = focal_alpha
    # 调用父类初始化方法
    super().__init__(is_encoder_decoder=is_encoder_decoder, **kwargs)

@property
def num_attention_heads(self) -> int:
    # 返回 encoder_attention_heads 属性值
    return self.encoder_attention_heads

@property
def hidden_size(self) -> int:
    # 返回 d_model 属性值
    return self.d_model
```