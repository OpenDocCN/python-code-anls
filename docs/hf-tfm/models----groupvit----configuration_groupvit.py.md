# `.\models\groupvit\configuration_groupvit.py`

```py
# coding=utf-8
# 定义 Python 源文件编码为 UTF-8

# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
# 版权声明，保留所有权利

# Licensed under the Apache License, Version 2.0 (the "License");
# 根据 Apache 许可证 Version 2.0 授权

# you may not use this file except in compliance with the License.
# 除非符合许可证的要求，否则不得使用此文件

# You may obtain a copy of the License at
# 您可以在以下网址获取许可证的副本

#     http://www.apache.org/licenses/LICENSE-2.0
#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# 除非适用法律要求或书面同意，否则本软件按“原样”分发，不提供任何形式的担保或条件。
# 详见许可证，以了解特定语言的权限

""" GroupViT model configuration"""
# GroupViT 模型配置

import os
# 导入操作系统相关模块
from collections import OrderedDict
# 导入 OrderedDict 类型，用于维护键值对的插入顺序
from typing import TYPE_CHECKING, Any, Mapping, Optional, Union
# 导入类型提示相关模块

from ...configuration_utils import PretrainedConfig
# 从配置工具模块导入预训练配置类
from ...onnx import OnnxConfig
# 从 ONNX 模块导入 ONNX 配置类
from ...utils import logging
# 从工具模块导入日志模块

if TYPE_CHECKING:
    from ...processing_utils import ProcessorMixin
    # 如果是类型检查，从处理工具模块导入 ProcessorMixin 类
    from ...utils import TensorType
    # 如果是类型检查，从工具模块导入 TensorType 类型

logger = logging.get_logger(__name__)
# 获取当前模块的日志记录器对象

GROUPVIT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "nvidia/groupvit-gcc-yfcc": "https://huggingface.co/nvidia/groupvit-gcc-yfcc/resolve/main/config.json",
}
# GroupViT 预训练模型配置映射字典，将模型名称映射到其配置文件的 URL
# 当前只包含了一个模型 "nvidia/groupvit-gcc-yfcc" 对应的配置文件 URL
# https://huggingface.co/nvidia/groupvit-gcc-yfcc/resolve/main/config.json

class GroupViTTextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`GroupViTTextModel`]. It is used to instantiate an
    GroupViT model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the GroupViT
    [nvidia/groupvit-gcc-yfcc](https://huggingface.co/nvidia/groupvit-gcc-yfcc) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """
    # GroupViTTextConfig 类的文档字符串，描述了如何存储 GroupViTTextModel 的配置
    # 该配置用于实例化 GroupViT 模型，并根据指定的参数定义模型架构
    # 使用默认参数实例化配置将产生类似于 GroupViT [nvidia/groupvit-gcc-yfcc] 架构的配置
    # 配置对象继承自 PretrainedConfig，并可用于控制模型输出。详细信息请参阅 PretrainedConfig 的文档。
    # 定义一个字符串常量，表示 GroupViT 文本模型的类型
    model_type = "groupvit_text_model"
    # 初始化函数，用于创建一个新的配置对象
    def __init__(
        self,
        vocab_size=49408,
        hidden_size=256,
        intermediate_size=1024,
        num_hidden_layers=12,
        num_attention_heads=4,
        max_position_embeddings=77,
        hidden_act="quick_gelu",
        layer_norm_eps=1e-5,
        dropout=0.0,
        attention_dropout=0.0,
        initializer_range=0.02,
        initializer_factor=1.0,
        pad_token_id=1,
        bos_token_id=49406,
        eos_token_id=49407,
        **kwargs,
    ):
        # 调用父类的初始化函数，设置特殊的标记ID和其他传递的关键字参数
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        # 设置配置对象的各种属性
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.dropout = dropout
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_position_embeddings = max_position_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.initializer_factor = initializer_factor
        self.attention_dropout = attention_dropout

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        # 设置 token 的相关参数到 kwargs 中
        cls._set_token_in_kwargs(kwargs)

        # 获取配置字典和更新后的 kwargs
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # 如果配置字典中的 model_type 是 groupvit，则使用其中的 text_config
        if config_dict.get("model_type") == "groupvit":
            config_dict = config_dict["text_config"]

        # 如果存在 model_type 并且不与当前类的 model_type 相符，则发出警告
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        # 从配置字典创建配置对象并返回
        return cls.from_dict(config_dict, **kwargs)
# 继承自预训练配置类 PretrainedConfig，用于存储 GroupViTVisionModel 的配置信息
class GroupViTVisionConfig(PretrainedConfig):
    """
    这是一个配置类，用于存储 [`GroupViTVisionModel`] 的配置信息。它被用来根据指定的参数实例化一个 GroupViT 模型，
    定义模型的架构。使用默认参数实例化一个配置对象会得到与 GroupViT [nvidia/groupvit-gcc-yfcc] 架构相似的配置。

    配置对象继承自 [`PretrainedConfig`]，可以用来控制模型的输出。阅读 [`PretrainedConfig`] 的文档以获取更多信息。

    Args:
        hidden_size (`int`, *optional*, defaults to 384):
            编码器层和池化层的维度。
        intermediate_size (`int`, *optional*, defaults to 1536):
            Transformer 编码器中 "intermediate"（即前馈）层的维度。
        depths (`List[int]`, *optional*, defaults to [6, 3, 3]):
            每个编码器块中的层数。
        num_group_tokens (`List[int]`, *optional*, defaults to [64, 8, 0]):
            每个阶段的组令牌数量。
        num_output_groups (`List[int]`, *optional*, defaults to [64, 8, 8]):
            每个阶段的输出组数，0 表示没有组。
        num_attention_heads (`int`, *optional*, defaults to 6):
            Transformer 编码器中每个注意力层的注意头数。
        image_size (`int`, *optional*, defaults to 224):
            每个图像的大小（分辨率）。
        patch_size (`int`, *optional*, defaults to 16):
            每个补丁的大小（分辨率）。
        hidden_act (`str` or `function`, *optional*, defaults to `"gelu"`):
            编码器和池化器中的非线性激活函数（函数或字符串）。支持的字符串有 "gelu"、"relu"、"selu" 和 "gelu_new" "quick_gelu"。
        layer_norm_eps (`float`, *optional*, defaults to 1e-5):
            层归一化层使用的 epsilon。
        dropout (`float`, *optional*, defaults to 0.0):
            嵌入层、编码器和池化器中所有全连接层的 dropout 概率。
        attention_dropout (`float`, *optional*, defaults to 0.0):
            注意力概率的 dropout 比率。
        initializer_range (`float`, *optional*, defaults to 0.02):
            用于初始化所有权重矩阵的截断正态初始化器的标准差。
        initializer_factor (`float`, *optional*, defaults to 1.0):
            初始化所有权重矩阵的因子（应保持为 1，用于初始化测试中使用）。

    Example:

    ```
    >>> from transformers import GroupViTVisionConfig, GroupViTVisionModel
    ```
    # 初始化一个 GroupViTVisionConfig 对象，使用 nvidia/groupvit-gcc-yfcc 风格的配置
    configuration = GroupViTVisionConfig()

    # 使用上述配置初始化一个 GroupViTVisionModel 模型
    model = GroupViTVisionModel(configuration)

    # 获取模型的配置信息
    configuration = model.config
# 定义 GroupViTConfig 类，继承自 PretrainedConfig 类
class GroupViTConfig(PretrainedConfig):
    r"""
    [`GroupViTConfig`] 是用于存储 [`GroupViTModel`] 配置的配置类。它用于根据指定的参数实例化一个 GroupViT 模型，
    定义文本模型和视觉模型的配置。使用默认值实例化配置将产生与 GroupViT [nvidia/groupvit-gcc-yfcc] 架构类似的配置。

    配置对象继承自 [`PretrainedConfig`]，可以用于控制模型的输出。阅读 [`PretrainedConfig`] 的文档以获取更多信息。

    Args:
        text_config (`dict`, *optional*):
            用于初始化 [`GroupViTTextConfig`] 的配置选项字典。
        vision_config (`dict`, *optional*):
            用于初始化 [`GroupViTVisionConfig`] 的配置选项字典。
        projection_dim (`int`, *optional*, defaults to 256):
            文本和视觉投影层的维度。
        projection_intermediate_dim (`int`, *optional*, defaults to 4096):
            文本和视觉投影层中间层的维度。
        logit_scale_init_value (`float`, *optional*, defaults to 2.6592):
            *logit_scale* 参数的初始值。默认值与原始 GroupViT 实现相匹配。
        kwargs (*optional*):
            关键字参数字典。
    """

    # 模型类型为 "groupvit"
    model_type = "groupvit"

    # 构造函数，初始化 GroupViTConfig 实例
    def __init__(
        self,
        text_config=None,
        vision_config=None,
        projection_dim=256,
        projection_intermediate_dim=4096,
        logit_scale_init_value=2.6592,
        **kwargs,
    ):
        # 调用父类的构造函数，设置配置选项
        super().__init__(**kwargs)

    @classmethod
    def from_text_vision_configs(cls, text_config: GroupViTTextConfig, vision_config: GroupViTVisionConfig, **kwargs):
        r"""
        从 GroupViT 文本模型配置和 GroupViT 视觉模型配置实例化一个 [`GroupViTConfig`]（或其派生类）。

        Returns:
            [`GroupViTConfig`]: 配置对象的一个实例
        """
        # 使用传入的 text_config 和 vision_config 创建一个新的 GroupViTConfig 实例
        return cls(text_config=text_config.to_dict(), vision_config=vision_config.to_dict(), **kwargs)


class GroupViTOnnxConfig(OnnxConfig):
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 定义输入层的名称和维度映射
        return OrderedDict(
            [
                ("input_ids", {0: "batch", 1: "sequence"}),
                ("pixel_values", {0: "batch", 1: "num_channels", 2: "height", 3: "width"}),
                ("attention_mask", {0: "batch", 1: "sequence"}),
            ]
        )

    @property
    # 返回一个有序字典，包含输出标识和对应的数据结构
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                ("logits_per_image", {0: "batch"}),  # 输出标识"logits_per_image"对应值为{0: "batch"}
                ("logits_per_text", {0: "batch"}),   # 输出标识"logits_per_text"对应值为{0: "batch"}
                ("text_embeds", {0: "batch"}),        # 输出标识"text_embeds"对应值为{0: "batch"}
                ("image_embeds", {0: "batch"}),       # 输出标识"image_embeds"对应值为{0: "batch"}
            ]
        )

    # 返回用于验证的绝对误差容限值
    @property
    def atol_for_validation(self) -> float:
        return 1e-4

    # 生成虚拟的输入数据字典，结合文本和图像处理器的虚拟输入
    def generate_dummy_inputs(
        self,
        processor: "ProcessorMixin",
        batch_size: int = -1,
        seq_length: int = -1,
        framework: Optional["TensorType"] = None,
    ) -> Mapping[str, Any]:
        # 调用父类方法生成文本输入字典
        text_input_dict = super().generate_dummy_inputs(
            processor.tokenizer, batch_size=batch_size, seq_length=seq_length, framework=framework
        )
        # 调用父类方法生成图像输入字典
        image_input_dict = super().generate_dummy_inputs(
            processor.image_processor, batch_size=batch_size, framework=framework
        )
        # 合并文本和图像输入字典，返回合并后的结果
        return {**text_input_dict, **image_input_dict}

    # 返回默认的ONNX操作集版本号
    @property
    def default_onnx_opset(self) -> int:
        return 14


这些注释解释了每个方法和属性的作用，确保了代码的清晰性和可读性。
```