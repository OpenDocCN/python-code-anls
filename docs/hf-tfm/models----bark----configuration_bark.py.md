# `.\models\bark\configuration_bark.py`

```py
# coding=utf-8
# Copyright 2023 The Suno AI Authors and The HuggingFace Inc. team. All rights reserved.
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
""" BARK model configuration"""

import os
from typing import Dict, Optional, Union

from ...configuration_utils import PretrainedConfig
from ...utils import add_start_docstrings, logging
from ..auto import CONFIG_MAPPING

# 获取名为 logging 的模块中的日志记录器对象
logger = logging.get_logger(__name__)

# BARK_PRETRAINED_CONFIG_ARCHIVE_MAP 是一个映射表，将模型名称映射到其预训练配置文件的 URL
BARK_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "suno/bark-small": "https://huggingface.co/suno/bark-small/resolve/main/config.json",
    "suno/bark": "https://huggingface.co/suno/bark/resolve/main/config.json",
}

# BARK_SUBMODELCONFIG_START_DOCSTRING 是一个多行字符串，用于说明配置类的作用和用法
BARK_SUBMODELCONFIG_START_DOCSTRING = """
    This is the configuration class to store the configuration of a [`{model}`]. It is used to instantiate the model
    according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Bark [suno/bark](https://huggingface.co/suno/bark)
    architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    # 设置块大小，定义模型可能使用的最大序列长度，默认为 1024。通常设置为较大值（例如 512、1024 或 2048），以防万一。
    block_size (`int`, *optional*, defaults to 1024):
    
    # 输入词汇表大小，用于 Bark 子模型。定义在调用 `{model}` 时可以表示的不同 token 数量。默认为 10,048，但应根据所选子模型仔细考虑。
    input_vocab_size (`int`, *optional*, defaults to 10_048):
    
    # 输出词汇表大小，用于 Bark 子模型。定义在向前传递 `{model}` 时可以表示的不同 token 数量。默认为 10,048，但应根据所选子模型仔细考虑。
    output_vocab_size (`int`, *optional*, defaults to 10_048):
    
    # 给定子模型中的隐藏层数量。默认为 12。
    num_layers (`int`, *optional*, defaults to 12):
    
    # Transformer 架构中每个注意力层的注意力头数量。默认为 12。
    num_heads (`int`, *optional*, defaults to 12):
    
    # 架构中“中间”（通常称为前馈）层的维度大小。默认为 768。
    hidden_size (`int`, *optional*, defaults to 768):
    
    # 嵌入层、编码器和池化器中所有全连接层的 dropout 概率。默认为 0.0，即不使用 dropout。
    dropout (`float`, *optional*, defaults to 0.0):
    
    # 是否在线性层和层归一化层中使用偏置。默认为 `True`。
    bias (`bool`, *optional*, defaults to `True`):
    
    # 初始化所有权重矩阵的截断正态初始化器的标准差。默认为 0.02。
    initializer_range (`float`, *optional*, defaults to 0.02):
    
    # 模型是否应返回最后的键/值注意力。并非所有模型都使用此功能。默认为 `True`。
    use_cache (`bool`, *optional*, defaults to `True`):
"""
定义了一个名为 BarkSubModelConfig 的类，继承自 PretrainedConfig。

model_type 属性指定为 "bark_module"，用于标识模型类型为 Bark 模块。
keys_to_ignore_at_inference 属性指定在推断时要忽略的键，这里包括 "past_key_values"。

attribute_map 属性是一个映射字典，将类内部属性名映射到外部使用的名称，例如将 num_attention_heads 映射为 num_heads。

__init__ 方法用于初始化类的实例，接受多个参数来设置模型配置的各个属性，如 block_size、input_vocab_size 等。

from_pretrained 方法是一个类方法，用于从预训练模型加载配置。它接受预训练模型的名称或路径，并支持设置缓存目录、强制下载等参数。

在方法内部，通过调用 cls.get_config_dict 方法获取预训练模型的配置字典。如果配置字典中的 model_type 为 "bark"，则从中提取对应的 Bark 配置。

警告日志用于提示用户，如果加载的预训练模型类型与当前类定义的模型类型不匹配，可能会导致错误。
"""
    # 获取模型的配置信息
    configuration = model.config
# 在 `BarkSubModelConfig` 的基础上定义了一个名为 `BarkSemanticConfig` 的类
class BarkSemanticConfig(BarkSubModelConfig):
    # 设定模型类型为 "semantic"
    model_type = "semantic"

# 在 `BarkSubModelConfig` 的基础上定义了一个名为 `BarkCoarseConfig` 的类
@add_start_docstrings(
    # 添加起始文档字符串，使用 `BARK_SUBMODELCONFIG_START_DOCSTRING` 格式化字符串
    BARK_SUBMODELCONFIG_START_DOCSTRING.format(config="BarkCoarseConfig", model="BarkCoarseModel"),
    """
    Example:

    ```
    >>> from transformers import BarkCoarseConfig, BarkCoarseModel

    >>> # Initializing a Bark sub-module style configuration
    >>> configuration = BarkCoarseConfig()

    >>> # Initializing a model (with random weights) from the suno/bark style configuration
    >>> model = BarkCoarseModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```""",
)
class BarkCoarseConfig(BarkSubModelConfig):
    # 设定模型类型为 "coarse_acoustics"
    model_type = "coarse_acoustics"

# 在 `BarkSubModelConfig` 的基础上定义了一个名为 `BarkFineConfig` 的类
@add_start_docstrings(
    # 添加起始文档字符串，使用 `BARK_SUBMODELCONFIG_START_DOCSTRING` 格式化字符串
    BARK_SUBMODELCONFIG_START_DOCSTRING.format(config="BarkFineConfig", model="BarkFineModel"),
    """
        n_codes_total (`int`, *optional*, defaults to 8):
            The total number of audio codebooks predicted. Used in the fine acoustics sub-model.
        n_codes_given (`int`, *optional*, defaults to 1):
            The number of audio codebooks predicted in the coarse acoustics sub-model. Used in the acoustics
            sub-models.
    Example:

    ```
    >>> from transformers import BarkFineConfig, BarkFineModel

    >>> # Initializing a Bark sub-module style configuration
    >>> configuration = BarkFineConfig()

    >>> # Initializing a model (with random weights) from the suno/bark style configuration
    >>> model = BarkFineModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```""",
)
class BarkFineConfig(BarkSubModelConfig):
    # 设定模型类型为 "fine_acoustics"
    model_type = "fine_acoustics"

    def __init__(self, tie_word_embeddings=True, n_codes_total=8, n_codes_given=1, **kwargs):
        # 初始化方法，设定了一些参数和默认值
        self.n_codes_total = n_codes_total  # 总音频码书预测数量，默认为8
        self.n_codes_given = n_codes_given  # 粗声学子模型中音频码书预测数量，默认为1

        # 调用父类的初始化方法
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

# 继承自 `PretrainedConfig` 的 `BarkConfig` 类
class BarkConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`BarkModel`]. It is used to instantiate a Bark
    model according to the specified sub-models configurations, defining the model architecture.

    Instantiating a configuration with the defaults will yield a similar configuration to that of the Bark
    [suno/bark](https://huggingface.co/suno/bark) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
    semantic_config ([`BarkSemanticConfig`], *optional*):
        Configuration of the underlying semantic sub-model.
    coarse_acoustics_config ([`BarkCoarseConfig`], *optional*):
        Configuration of the underlying coarse acoustics sub-model.
    fine_acoustics_config ([`BarkFineConfig`], *optional*):
        Configuration of the underlying fine acoustics sub-model.
    """
        codec_config ([`AutoConfig`], *optional*):
            Configuration of the underlying codec sub-model.



        model_type = "bark"



        def __init__(
            self,
            semantic_config: Dict = None,
            coarse_acoustics_config: Dict = None,
            fine_acoustics_config: Dict = None,
            codec_config: Dict = None,
            initializer_range=0.02,
            **kwargs,
        ):
            # 如果semantic_config为None，则使用默认空字典并记录日志
            if semantic_config is None:
                semantic_config = {}
                logger.info("semantic_config is None. initializing the semantic model with default values.")

            # 如果coarse_acoustics_config为None，则使用默认空字典并记录日志
            if coarse_acoustics_config is None:
                coarse_acoustics_config = {}
                logger.info("coarse_acoustics_config is None. initializing the coarse model with default values.")

            # 如果fine_acoustics_config为None，则使用默认空字典并记录日志
            if fine_acoustics_config is None:
                fine_acoustics_config = {}
                logger.info("fine_acoustics_config is None. initializing the fine model with default values.")

            # 如果codec_config为None，则使用默认空字典并记录日志
            if codec_config is None:
                codec_config = {}
                logger.info("codec_config is None. initializing the codec model with default values.")

            # 初始化各个配置对象，如果给定配置为空，则创建默认配置对象
            self.semantic_config = BarkSemanticConfig(**semantic_config)
            self.coarse_acoustics_config = BarkCoarseConfig(**coarse_acoustics_config)
            self.fine_acoustics_config = BarkFineConfig(**fine_acoustics_config)
            
            # 确定codec_model_type，如果未指定则默认为"encodec"
            codec_model_type = codec_config["model_type"] if "model_type" in codec_config else "encodec"
            self.codec_config = CONFIG_MAPPING[codec_model_type](**codec_config)

            # 设置初始化范围
            self.initializer_range = initializer_range

            super().__init__(**kwargs)



        @classmethod
        def from_sub_model_configs(
            cls,
            semantic_config: BarkSemanticConfig,
            coarse_acoustics_config: BarkCoarseConfig,
            fine_acoustics_config: BarkFineConfig,
            codec_config: PretrainedConfig,
            **kwargs,
        ):
        ):
        r"""
        从bark子模型配置中实例化一个[`BarkConfig`]（或派生类）。

        Returns:
            [`BarkConfig`]: 配置对象的一个实例
        """
        return cls(
            semantic_config=semantic_config.to_dict(),  # 将语义配置转换为字典形式
            coarse_acoustics_config=coarse_acoustics_config.to_dict(),  # 将粗略声学配置转换为字典形式
            fine_acoustics_config=fine_acoustics_config.to_dict(),  # 将精细声学配置转换为字典形式
            codec_config=codec_config.to_dict(),  # 将编解码器配置转换为字典形式
            **kwargs,  # 传递额外的关键字参数
        )
```