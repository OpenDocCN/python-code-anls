# `.\models\llava_next\configuration_llava_next.py`

```
# coding=utf-8
# 定义编码格式为 UTF-8，确保文件能正确处理各种字符
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
# 版权声明，指出此代码的版权归HuggingFace Inc.团队所有
# Licensed under the Apache License, Version 2.0 (the "License");
# 根据 Apache License, Version 2.0 许可协议授权使用此代码
# you may not use this file except in compliance with the License.
# 除非你遵守许可协议，否则不得使用此文件
# You may obtain a copy of the License at
# 你可以在以下网址获取许可协议的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# 除非适用法律要求或书面同意，否则软件
# distributed under the License is distributed on an "AS IS" BASIS,
# 根据许可协议分发的软件是基于"AS IS"的基础上分发
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 没有明示或暗示的任何保证或条件
# See the License for the specific language governing permissions and
# 请查看许可协议以了解详细的授权条款和条件
# limitations under the License.
# 限制条件在许可协议下适用

""" Llava-NeXT model configuration"""
# Llava-NeXT 模型配置

from ...configuration_utils import PretrainedConfig
# 从相对路径导入 PretrainedConfig 类
from ...utils import logging
# 从相对路径导入 logging 模块
from ..auto import CONFIG_MAPPING
# 从相对路径导入 CONFIG_MAPPING 变量


logger = logging.get_logger(__name__)
# 获取当前模块的 logger 实例

LLAVA_NEXT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "llava-hf/llava-v1.6-mistral-7b-hf": "https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf/resolve/main/config.json",
}
# 定义 LLAVA_NEXT_PRETRAINED_CONFIG_ARCHIVE_MAP 字典，映射预训练模型名到配置文件 URL

class LlavaNextConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LlavaNextForConditionalGeneration`]. It is used to instantiate an
    Llava-NeXT model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the [llava-hf/llava-v1.6-mistral-7b-hf](https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf)
    model.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """
    # LlavaNextConfig 类继承自 PretrainedConfig 类，用于存储 Llava-NeXT 模型的配置信息，指定模型的架构
    # 设置模型类型为 "llava_next"
    model_type = "llava_next"
    # 设置是否为组合模型为 False
    is_composition = False

    # 定义模型初始化方法
    def __init__(
        self,
        vision_config=None,  # 视觉部分的配置对象或字典，默认为 None
        text_config=None,    # 文本部分的配置对象或字典，默认为 None
        ignore_index=-100,   # 损失函数中的忽略索引，默认为 -100
        image_token_index=32000,  # 编码图像提示时的图像标记索引，默认为 32000
        projector_hidden_act="gelu",  # 多模态投影器使用的激活函数，默认为 "gelu"
        vision_feature_select_strategy="default",  # 从视觉主干中选择特征的策略，默认为 "default"
        vision_feature_layer=-2,    # 选择视觉特征的层索引，默认为 -2
        image_grid_pinpoints=None,  # 用于处理高分辨率图像的可能分辨率列表，默认为 None
        **kwargs,   # 其他可选关键字参数
        ):
            # 初始化类的实例变量
            self.ignore_index = ignore_index
            self.image_token_index = image_token_index
            self.projector_hidden_act = projector_hidden_act

            # 检查视觉特征选择策略是否合法
            if vision_feature_select_strategy not in ["default", "full"]:
                raise ValueError(
                    "vision_feature_select_strategy should be one of 'default', 'full'."
                    f"Got: {vision_feature_select_strategy}"
                )

            self.vision_feature_select_strategy = vision_feature_select_strategy
            self.vision_feature_layer = vision_feature_layer

            # 设置图像网格的固定标记点，默认为指定的坐标
            image_grid_pinpoints = (
                image_grid_pinpoints
                if image_grid_pinpoints is not None
                else [[336, 672], [672, 336], [672, 672], [1008, 336], [336, 1008]]
            )
            self.image_grid_pinpoints = image_grid_pinpoints

            # 根据视觉配置初始化视觉模型
            if isinstance(vision_config, dict):
                vision_config["model_type"] = (
                    vision_config["model_type"] if "model_type" in vision_config else "clip_vision_model"
                )
                vision_config = CONFIG_MAPPING[vision_config["model_type"]](**vision_config)
            elif vision_config is None:
                vision_config = CONFIG_MAPPING["clip_vision_model"](
                    intermediate_size=4096,
                    hidden_size=1024,
                    patch_size=14,
                    image_size=336,
                    num_hidden_layers=24,
                    num_attention_heads=16,
                    vocab_size=32000,
                    projection_dim=768,
                )

            self.vision_config = vision_config

            # 根据文本配置初始化文本模型
            if isinstance(text_config, dict):
                text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "llama"
                text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
            elif text_config is None:
                text_config = CONFIG_MAPPING["llama"]()

            self.text_config = text_config

            # 调用父类的初始化方法，传入额外的关键字参数
            super().__init__(**kwargs)
```