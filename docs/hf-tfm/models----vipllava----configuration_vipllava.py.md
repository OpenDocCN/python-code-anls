# `.\models\vipllava\configuration_vipllava.py`

```py
# 定义模块的版权信息和许可协议
# coding=utf-8
# Copyright 2023 Microsoft Research & University of Wisconsin-Madison and the HuggingFace Inc. team. All rights reserved.
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

""" VipLlava model configuration"""

# 引入警告模块
import warnings

# 从 transformers 包中引入预训练配置类 PretrainedConfig
from ...configuration_utils import PretrainedConfig
# 从 transformers.utils 中引入日志记录功能
from ...utils import logging
# 从 transformers.modeling_auto 中引入配置映射
from ..auto import CONFIG_MAPPING

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义预训练模型配置文件的映射
VIPLLAVA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "ybelkada/vip-llava-7b-hf": "https://huggingface.co/llava-hf/vip-llava-7b-hf/resolve/main/config.json",
}

# 定义 VipLlavaConfig 类，继承自 PretrainedConfig 类
class VipLlavaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`VipLlavaForConditionalGeneration`]. It is used to instantiate an
    VipLlava model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the VipLlava-9B.

    e.g. [ybelkada/vip-llava-7b-hf](https://huggingface.co/ybelkada/vip-llava-7b-hf)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vision_config (`VipLlavaVisionConfig`,  *optional*):
            Custom vision config or dict
        text_config (`Union[AutoConfig, dict]`, *optional*):
            The config object of the text backbone. Can be any of `LlamaConfig` or `MistralConfig`.
        ignore_index (`int`, *optional*, defaults to -100):
            The ignore index for the loss function.
        image_token_index (`int`, *optional*, defaults to 32000):
            The image token index to encode the image prompt.
        projector_hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The activation function used by the multimodal projector.
        projector_layernorm_eps (`float`, *optional*, defaults to 1e-05):
            The layer norm epsilon of the projector layernorm
        vision_feature_layers (`List[int]`, *optional*, defaults to `[-2, -5, -8, -11, 6]`):
            The list of layers to select the vision features from.

    Example:

    ```
    >>> from transformers import VipLlavaForConditionalGeneration, VipLlavaConfig, CLIPVisionConfig, LlamaConfig

    >>> # Initializing a CLIP-vision config
    >>> vision_config = CLIPVisionConfig()

    >>> # Initializing a Llama config
    >>> text_config = LlamaConfig()

    ```

    """
    # 初始化一个 VipLlava vipllava-7b 风格的配置
    >>> configuration = VipLlavaConfig(vision_config, text_config)

    # 使用 vipllava-7b 风格的配置初始化一个模型
    >>> model = VipLlavaForConditionalGeneration(configuration)

    # 获取模型的配置信息
    >>> configuration = model.config
```