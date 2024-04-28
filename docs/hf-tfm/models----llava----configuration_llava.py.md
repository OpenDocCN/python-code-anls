# `.\transformers\models\llava\configuration_llava.py`

```
# 以 UTF-8 编码方式解析代码
# 版权所有 2023 年微软研究所和威斯康星大学麦迪逊分校以及 HuggingFace 公司团队. 保留所有权利.
# 根据 Apache License, Version 2.0 许可，除非符合许可要求，否则不得使用此文件.
# 您可以在以下网址获取许可的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则按 "AS IS" 基础分发软件，
# 没有任何种类的担保或条件，无论是明示的还是暗示的.
# 请查看许可协议以了解特定语言管理权限和限制
""" Llava 模型的配置类 """

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto import CONFIG_MAPPING

logger = logging.get_logger(__name__)

# Llava 预训练的配置文件映射列表
LLAVA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "llava-hf/llava-v1.5-7b": "https://huggingface.co/llava-hf/llava-v1.5-7b/resolve/main/config.json",
}


class LlavaConfig(PretrainedConfig):
    r"""
    这是用于存储 [`LlavaForConditionalGeneration`] 配置的配置类。用于根据指定的参数实例化 Llava 模型，从而定义模型架构。
    用默认值实例化配置将生成类似于 Llava-9B 的配置。

    例如 [llava-hf/llava-9b](https://huggingface.co/llava-hf/llava-9b)

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。阅读 [`PretrainedConfig`] 的文档以获取更多信息.

    Args:
        vision_config (`LlavaVisionConfig`,  *optional*):
            自定义视觉配置或字典
        text_config (`Union[AutoConfig, dict]`, *optional*):
            文本骨干的配置对象。可以是 `LlamaConfig` 或 `MistralConfig` 中的任何一个.
        ignore_index (`int`, *optional*, defaults to -100):
            损失函数的忽略索引.
        image_token_index (`int`, *optional*, defaults to 32000):
            用于编码图像提示的图像令牌索引.
        projector_hidden_act (`str`, *optional*, defaults to `"gelu"`):
            多模态投影仪使用的激活函数.
        vision_feature_select_strategy (`str`, *optional*, defaults to `"default"`):
            用于从 CLIP 骨干中选择视觉特征的特征选择策略.
        vision_feature_layer (`int`, *optional*, defaults to -2):
            选择视觉特征的图层索引.
        vocab_size (`int`, *optional*, defaults to 32000):
            Llava 模型的词汇表大小。定义了调用 [`~LlavaForConditionalGeneration`] 时可以表示的不同令牌数量.

    Example:

    ```python
    # 从transformers库中导入模型相关的类和配置
    >>> from transformers import LlavaForConditionalGeneration, LlavaConfig, CLIPVisionConfig, LlamaConfig

    # 初始化一个CLIP-vision的配置
    >>> vision_config = CLIPVisionConfig()

    # 初始化一个Llama的配置
    >>> text_config = LlamaConfig()

    # 使用CLIP-vision配置和Llama配置来初始化一个llava-1.5-7b风格的配置
    >>> configuration = LlavaConfig(vision_config, text_config)

    # 使用llava-1.5-7b风格的配置来初始化一个模型
    >>> model = LlavaForConditionalGeneration(configuration)

    # 访问模型的配置
    >>> configuration = model.config
    ```"""

    # 定义模型的类型为"llava"，组合标志为False
    model_type = "llava"
    is_composition = False

    # 初始化函数，包含模型的各种参数以及配置
    def __init__(
        self,
        vision_config=None,
        text_config=None,
        ignore_index=-100,
        image_token_index=32000,
        projector_hidden_act="gelu",
        vision_feature_select_strategy="default",
        vision_feature_layer=-2,
        vocab_size=32000,
        **kwargs,
    ):
        # 设置模型中需要用到的参数
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.projector_hidden_act = projector_hidden_act
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_layer = vision_feature_layer
        self.vocab_size = vocab_size

        # 将CLIP-vision配置存储在vision_config中
        self.vision_config = vision_config

        # 处理vision_config参数，将其转换为对应的配置类
        if isinstance(self.vision_config, dict):
            vision_config["model_type"] = (
                vision_config["model_type"] if "model_type" in vision_config else "clip_vision_model"
            )
            self.vision_config = CONFIG_MAPPING[vision_config["model_type"]](**vision_config)
        elif vision_config is None:
            self.vision_config = CONFIG_MAPPING["clip_vision_model"](
                intermediate_size=4096,
                hidden_size=1024,
                patch_size=14,
                image_size=336,
                num_hidden_layers=24,
                num_attention_heads=16,
                vocab_size=32000,
                projection_dim=768,
            )
        self.vocab_size = self.vocab_size

        # 将Llama配置存储在text_config中
        self.text_config = text_config

        # 处理text_config参数，将其转换为对应的配置类
        if isinstance(self.text_config, dict):
            text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "llama"
            self.text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
            self.vocab_size = self.text_config.vocab_size
        elif text_config is None:
            self.text_config = CONFIG_MAPPING["llama"]()

        # 调用父类的初始化方法
        super().__init__(**kwargs)
```