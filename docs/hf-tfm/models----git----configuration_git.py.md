# `.\models\git\configuration_git.py`

```
# coding=utf-8
# 上面的行声明了文件的编码格式为 UTF-8，确保文件中的中文和特殊字符能正确解析
# Copyright 2022 The HuggingFace Inc. team. All rights reserved.
# 版权声明，声明代码版权归 HuggingFace Inc. 团队所有，保留所有权利
#
# Licensed under the Apache License, Version 2.0 (the "License");
# 根据 Apache 许可证版本 2.0 进行许可，即除非符合许可证要求，否则不得使用此文件
# You may obtain a copy of the License at
# 你可以在上述链接获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 根据许可证分发的软件是基于 "AS IS" 基础分发的，没有任何形式的担保或条件
# See the License for the specific language governing permissions and
# limitations under the License.
# 查看许可证以了解详细的条款和条件
#
# Importing necessary modules
# 导入必要的模块
import os
# Importing Union type hint from typing module
# 从 typing 模块导入 Union 类型提示
from typing import Union
# Importing necessary modules from local package
# 从本地包中导入必要的模块
from ...configuration_utils import PretrainedConfig
from ...utils import logging
# Getting the logger object specific to the current module
# 获取与当前模块相关的日志记录器对象
logger = logging.get_logger(__name__)
# Mapping of pretrained model identifier to its configuration file URL
# 预训练模型标识符到配置文件 URL 的映射
GIT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/git-base": "https://huggingface.co/microsoft/git-base/resolve/main/config.json",
}


class GitVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`GitVisionModel`]. It is used to instantiate a GIT
    vision encoder according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the vision encoder of the GIT
    [microsoft/git-base](https://huggingface.co/microsoft/git-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """
    # Configuration class for GitVisionModel
    # GitVisionModel 的配置类
    # This class inherits from PretrainedConfig
    # 该类继承自 PretrainedConfig
    # It defines configuration parameters for GitVisionModel
    # 定义 GitVisionModel 的配置参数
    # Read PretrainedConfig documentation for more details
    # 详细信息请参阅 PretrainedConfig 文档
    # 模型类型标识字符串，表示这是一个 Git Vision 模型
    model_type = "git_vision_model"
    
    # GitVisionConfig 类的构造函数，用于初始化模型配置参数
    def __init__(
        self,
        hidden_size=768,  # 编码器层和池化层的维度大小，默认为768
        intermediate_size=3072,  # Transformer 编码器中间层（即前馈层）的维度大小，默认为3072
        num_hidden_layers=12,  # Transformer 编码器中的隐藏层数，默认为12
        num_attention_heads=12,  # Transformer 编码器中每个注意力层的注意头数量，默认为12
        num_channels=3,  # 图像通道数，默认为3（RGB）
        image_size=224,  # 每个图像的分辨率大小，默认为224
        patch_size=16,  # 每个图像块（patch）的大小，默认为16
        hidden_act="quick_gelu",  # 编码器和池化器中的非线性激活函数，默认为"quick_gelu"
        layer_norm_eps=1e-5,  # 层归一化层使用的 epsilon 值，默认为1e-5
        attention_dropout=0.0,  # 注意力概率的 dropout 比率，默认为0.0（不进行 dropout）
        initializer_range=0.02,  # 初始化所有权重矩阵的截断正态分布的标准差，默认为0.02
        **kwargs,  # 其他可选关键字参数
    ):
        # 调用父类的构造函数，初始化其他可能存在的关键字参数
        super().__init__(**kwargs)
    
        # 设置实例变量，将传入的参数赋值给对象的对应属性
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.initializer_range = initializer_range
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps
        self.hidden_act = hidden_act
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        # 调用类方法 _set_token_in_kwargs，将 token 设置到 kwargs 中
        cls._set_token_in_kwargs(kwargs)

        # 调用类方法 get_config_dict，获取预训练模型的配置字典和更新后的 kwargs
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # 如果配置字典中的 model_type 是 "git"，则从 vision_config 中获取配置字典
        if config_dict.get("model_type") == "git":
            config_dict = config_dict["vision_config"]

        # 如果配置字典中有 "model_type"，并且类有 model_type 属性，并且它们不相等，发出警告
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        # 调用类方法 from_dict，使用配置字典和 kwargs 创建预训练配置对象并返回
        return cls.from_dict(config_dict, **kwargs)
class GitConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`GitModel`]. It is used to instantiate a GIT model
    according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the GIT
    [microsoft/git-base](https://huggingface.co/microsoft/git-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Examples:

    ```python
    >>> from transformers import GitConfig, GitModel

    >>> # Initializing a GIT microsoft/git-base style configuration
    >>> configuration = GitConfig()

    >>> # Initializing a model (with random weights) from the microsoft/git-base style configuration
    >>> model = GitModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "git"

    def __init__(
        self,
        vision_config=None,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=6,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=1024,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=True,
        tie_word_embeddings=False,
        bos_token_id=101,
        eos_token_id=102,
        num_image_with_embedding=None,
        **kwargs,
    ):
        # 调用父类构造函数，初始化基本配置，如起始、结束、填充 token ID 等
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, pad_token_id=pad_token_id, **kwargs)

        # 如果未提供 vision_config，则使用空字典，并记录日志
        if vision_config is None:
            vision_config = {}
            logger.info("vision_config is None. initializing the GitVisionConfig with default values.")

        # 根据提供的 vision_config 创建 GitVisionConfig 实例
        self.vision_config = GitVisionConfig(**vision_config)
        # 设置模型的词汇表大小
        self.vocab_size = vocab_size
        # 设置隐藏层的大小
        self.hidden_size = hidden_size
        # 设置隐藏层的数量
        self.num_hidden_layers = num_hidden_layers
        # 设置注意力头的数量
        self.num_attention_heads = num_attention_heads
        # 设置隐藏层的激活函数
        self.hidden_act = hidden_act
        # 设置中间层的大小
        self.intermediate_size = intermediate_size
        # 设置隐藏层的 dropout 概率
        self.hidden_dropout_prob = hidden_dropout_prob
        # 设置注意力机制的 dropout 概率
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        # 设置最大位置嵌入的长度
        self.max_position_embeddings = max_position_embeddings
        # 设置初始化范围
        self.initializer_range = initializer_range
        # 设置层归一化的 epsilon 值
        self.layer_norm_eps = layer_norm_eps
        # 设置位置嵌入类型
        self.position_embedding_type = position_embedding_type
        # 设置是否使用缓存
        self.use_cache = use_cache
        # 设置是否绑定词嵌入
        self.tie_word_embeddings = tie_word_embeddings
        # 设置具有嵌入的图像数量
        self.num_image_with_embedding = num_image_with_embedding

        # 设置起始 token ID
        self.bos_token_id = bos_token_id
        # 设置结束 token ID
        self.eos_token_id = eos_token_id
```