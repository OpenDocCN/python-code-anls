# `.\models\git\configuration_git.py`

```py
# 设定文件编码为utf-8
# 版权声明
# 2022年由HuggingFace Inc.团队保留所有权利。
# 根据Apache许可证2.0版（“许可证”）许可；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律或书面同意，本软件是按“原样”分发的，
# 不提供任何明示或暗示的担保或条件。
# 有关特定语言管理权限和限制，请参阅许可证。

# 导入模块
import os
from typing import Union
# 从HuggingFace库导入预训练配置
from ...configuration_utils import PretrainedConfig
# 从HuggingFace库导入日志模块
from ...utils import logging

# 获取日志句柄
logger = logging.get_logger(__name__)

# 定义预训练配置文件路径映射
GIT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/git-base": "https://huggingface.co/microsoft/git-base/resolve/main/config.json",
}

# 创建GitVisionConfig类，用于存储GitVisionModel的配置
class GitVisionConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`GitVisionModel`]. It is used to instantiate a GIT
    vision encoder according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the vision encoder of the GIT
    [microsoft/git-base](https://huggingface.co/microsoft/git-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    # 定义 GitVisionConfig 类，包含了一系列参数，用于配置 GitVisionModel
    class GitVisionConfig:
        
        # 初始化 GitVisionConfig 类，设置各项参数的默认数值
        def __init__(
            self,
            hidden_size=768, # 编码器层和池化层的维度
            intermediate_size=3072, # Transformer 编码器中“中间”（即前馈）层的维度
            num_hidden_layers=12, # Transformer 编码器中的隐藏层数量
            num_attention_heads=12, # 每个注意力层中的注意力头数量
            num_channels=3, # 每个图像的通道数
            image_size=224, # 每个图像的大小（分辨率）
            patch_size=16, # 每个图像块的大小（分辨率）
            hidden_act="quick_gelu", # 编码器和池化层中的非线性激活函数
            layer_norm_eps=1e-5, # 层归一化层使用的 epsilon 值
            attention_dropout=0.0, # 注意力概率的 dropout 比率
            initializer_range=0.02, # 用于初始化所有权重矩阵的截断正态初始化器的标准差
            **kwargs,
        ):
            # 调用父类初始化方法
            super().__init__(**kwargs)
    
            # 初始化各个参数
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
    
        # 定义 model_type 属性，值为 "git_vision_model"
        model_type = "git_vision_model"
        
        # 定义 GitVisionConfig 类的类方法
        @classmethod
    # 使用预训练模型名称或路径创建一个新的配置对象
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> "PretrainedConfig":
        # 设置token到关键字参数中
        cls._set_token_in_kwargs(kwargs)

        # 获取配置字典，并更新关键字参数
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # 如果配置字典中的模型类型是"git"，则获取视觉配置字典
        if config_dict.get("model_type") == "git":
            config_dict = config_dict["vision_config"]

        # 如果配置字典中存在模型类型，并且类中存在模型类型属性，并且配置字典中的模型类型不等于类的模型类型，则发出警告
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        # 从配置字典中创建一个新的模型配置对象，并返回
        return cls.from_dict(config_dict, **kwargs)
# GitConfig 类是用于存储 `GitModel` 的配置的类。根据指定的参数来实例化一个 GIT 模型，定义模型的架构。
# 使用默认配置实例化一个配置类将得到与 GIT [microsoft/git-base](https://huggingface.co/microsoft/git-base) 架构相似的配置。
# 配置对象继承自  PretrainedConfig 类，可用于控制模型的输出。请阅读 PretrainedConfig 的文档获取更多信息。
class GitConfig(PretrainedConfig):
    
    model_type = "git"  # 模型类型

    # 初始化方法
    def __init__(
        self,
        vision_config=None,   # 视觉配置，默认值为 None
        vocab_size=30522,   # 词汇表大小，默认值为30522
        hidden_size=768,   # 隐藏层大小，默认值为768
        num_hidden_layers=6,   # 隐藏层数量，默认值为6
        num_attention_heads=12,   # 注意力头数量，默认值为12
        intermediate_size=3072,   # 中间层大小，默认值为3072
        hidden_act="gelu",   # 隐藏层的激活函数，默认值为"gelu"
        hidden_dropout_prob=0.1,   # 隐藏层的dropout比例，默认值为0.1
        attention_probs_dropout_prob=0.1,   # 注意力概率的dropout比例，默认值为0.1
        max_position_embeddings=1024,   # 最大位置嵌入数量，默认值为1024
        initializer_range=0.02,   # 初始化范围，默认值为0.02
        layer_norm_eps=1e-12,   # 层归一化 epsilon，默认值为1e-12
        pad_token_id=0,   # 填充标记的id，默认值为0
        position_embedding_type="absolute",   # 位置嵌入类型，默认值为"absolute"
        use_cache=True,   # 是否使用缓存，默认值为True
        tie_word_embeddings=False,   # 是否绑定词嵌入，默认值为False
        bos_token_id=101,   # 句首标记的id，默认值为101
        eos_token_id=102,   # 句尾标记的id，默认值为102
        num_image_with_embedding=None,   # 带有嵌入的图像数量，默认值为None
        **kwargs,   # 其他参数
    ):
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, pad_token_id=pad_token_id, **kwargs)   # 调用父类的初始化方法，传入句首、句尾、填充标记的id

        if vision_config is None:   # 如果视觉配置为None
            vision_config = {}   # 初始化一个空的视觉配置字典
            logger.info("vision_config is None. initializing the GitVisionConfig with default values.")   # 输出日志，视觉配置为None，使用默认值初始化GitVisionConfig

        # 根据视觉配置初始化GitVisionConfig对象
        self.vision_config = GitVisionConfig(**vision_config)
        self.vocab_size = vocab_size   # 词汇表大小
        self.hidden_size = hidden_size   # 隐藏层大小
        self.num_hidden_layers = num_hidden_layers   # 隐藏层数量
        self.num_attention_heads = num_attention_heads   # 注意力头数量
        self.hidden_act = hidden_act   # 隐藏层的激活函数
        self.intermediate_size = intermediate_size   # 中间层大小
        self.hidden_dropout_prob = hidden_dropout_prob   # 隐藏层的dropout比例
        self.attention_probs_dropout_prob = attention_probs_dropout_prob   # 注意力概率的dropout比例
        self.max_position_embeddings = max_position_embeddings   # 最大位置嵌入数量
        self.initializer_range = initializer_range   # 初始化范围
        self.layer_norm_eps = layer_norm_eps   # 层归一化 epsilon
        self.position_embedding_type = position_embedding_type   # 位置嵌入类型
        self.use_cache = use_cache   # 是否使用缓存
        self.tie_word_embeddings = tie_word_embeddings   # 是否绑定词嵌入
        self.num_image_with_embedding = num_image_with_embedding   # 带有嵌入的图像数量

        self.bos_token_id = bos_token_id   # 句首标记的id
        self.eos_token_id = eos_token_id   # 句尾标记的id
```