# `.\models\llava\configuration_llava.py`

```
# 设置文件编码为UTF-8
# 版权声明，指明了版权归属及许可信息
# 根据Apache License, Version 2.0规定，除非符合许可条件，否则不得使用此文件
# 可在以下链接获取许可协议的副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则依照"AS IS"的方式分发本软件
# 不提供任何明示或默示的担保或条件
# 查看许可协议以了解更多细节

""" Llava model configuration"""

# 引入警告模块，用于处理警告信息
import warnings

# 从transformers模块中导入预训练配置类PretrainedConfig
from ...configuration_utils import PretrainedConfig

# 从transformers.utils中导入日志模块logging
from ...utils import logging

# 从transformers.models.auto中导入配置映射字典CONFIG_MAPPING
from ..auto import CONFIG_MAPPING

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义一个映射字典，将预训练模型名称映射到其配置文件的URL
LLAVA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "llava-hf/llava-v1.5-7b": "https://huggingface.co/llava-hf/llava-v1.5-7b/resolve/main/config.json",
}

# 定义LlavaConfig类，继承自PretrainedConfig类
class LlavaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LlavaForConditionalGeneration`]. It is used to instantiate an
    Llava model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the Llava-9B.

    e.g. [llava-hf/llava-9b](https://huggingface.co/llava-hf/llava-9b)

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        vision_config (`Union[AutoConfig, dict]`,  *optional*, defaults to `CLIPVisionConfig`):
            The config object or dictionary of the vision backbone.
        text_config (`Union[AutoConfig, dict]`, *optional*, defaults to `LlamaConfig`):
            The config object or dictionary of the text backbone.
        ignore_index (`int`, *optional*, defaults to -100):
            The ignore index for the loss function.
        image_token_index (`int`, *optional*, defaults to 32000):
            The image token index to encode the image prompt.
        projector_hidden_act (`str`, *optional*, defaults to `"gelu"`):
            The activation function used by the multimodal projector.
        vision_feature_select_strategy (`str`, *optional*, defaults to `"default"`):
            The feature selection strategy used to select the vision feature from the vision backbone.
            Can be one of `"default"` or `"full"`.
        vision_feature_layer (`int`, *optional*, defaults to -2):
            The index of the layer to select the vision feature.

    Example:

    ```python
    >>> from transformers import LlavaForConditionalGeneration, LlavaConfig, CLIPVisionConfig, LlamaConfig

    >>> # Initializing a CLIP-vision config
    >>> vision_config = CLIPVisionConfig()
    ```

    注释:
    此类用于存储`LlavaForConditionalGeneration`的配置信息，根据给定的参数实例化一个Llava模型，定义模型架构。
    使用默认参数实例化配置将产生类似于Llava-9B的配置。
    配置对象继承自`PretrainedConfig`，可用于控制模型输出。阅读`PretrainedConfig`的文档获取更多信息。
    """
    # 设置模型类型为 "llava"
    model_type = "llava"
    # 是否为复合模型，默认为 False
    is_composition = False

    # 初始化函数，接收多个参数来配置模型
    def __init__(
        self,
        vision_config=None,  # 视觉配置，可以是字典或预定义配置对象，默认为 None
        text_config=None,    # 文本配置，可以是字典或预定义配置对象，默认为 None
        ignore_index=-100,    # 忽略索引，默认为 -100
        image_token_index=32000,  # 图像令牌索引，默认为 32000
        projector_hidden_act="gelu",  # 投影器隐藏层激活函数，默认为 "gelu"
        vision_feature_select_strategy="default",  # 视觉特征选择策略，默认为 "default"
        vision_feature_layer=-2,  # 视觉特征层，默认为 -2
        **kwargs,  # 其余关键字参数
    ):
        # 初始化对象的各种属性
        self.ignore_index = ignore_index
        self.image_token_index = image_token_index
        self.projector_hidden_act = projector_hidden_act

        # 检查视觉特征选择策略是否有效，否则抛出 ValueError 异常
        if vision_feature_select_strategy not in ["default", "full"]:
            raise ValueError(
                "vision_feature_select_strategy should be one of 'default', 'full'."
                f"Got: {vision_feature_select_strategy}"
            )

        # 若传入了不推荐使用的 'vocab_size' 参数，则发出警告
        if "vocab_size" in kwargs:
            warnings.warn(
                "The `vocab_size` argument is deprecated and will be removed in v4.42, since it can be inferred from the `text_config`. Passing this argument has no effect",
                FutureWarning,
            )

        # 设置视觉特征选择策略和视觉特征层
        self.vision_feature_select_strategy = vision_feature_select_strategy
        self.vision_feature_layer = vision_feature_layer

        # 根据传入的视觉配置，创建对应的配置对象，若为 None 则使用默认的 "clip_vision_model" 配置
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

        # 设置视觉配置
        self.vision_config = vision_config

        # 根据传入的文本配置，创建对应的配置对象，若为 None 则使用默认的 "llama" 配置
        if isinstance(text_config, dict):
            text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "llama"
            text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
        elif text_config is None:
            text_config = CONFIG_MAPPING["llama"]()

        # 设置文本配置和词汇表大小
        self.text_config = text_config
        self._vocab_size = self.text_config.vocab_size

        # 调用父类的初始化方法，传入剩余的关键字参数
        super().__init__(**kwargs)
    # 发出警告，提示 `vocab_size` 属性已经被废弃，在 v4.42 版本中将被移除，建议使用 `text_config.vocab_size` 替代
    warnings.warn(
        "The `vocab_size` attribute is deprecated and will be removed in v4.42, Please use `text_config.vocab_size` instead.",
        FutureWarning,
    )
    # 返回对象的 `_vocab_size` 属性值
    return self._vocab_size

    # 将对象转换为字典类型并获取输出
    def to_dict(self):
        output = super().to_dict()
        # 从输出字典中删除 `_vocab_size` 键对应的值
        output.pop("_vocab_size", None)
        return output
```