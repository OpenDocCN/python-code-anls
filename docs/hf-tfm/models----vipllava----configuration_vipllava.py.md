# `.\transformers\models\vipllava\configuration_vipllava.py`

```py
# 设置编码格式为 utf-8
# 版权声明，版权归 Microsoft Research、University of Wisconsin-Madison 和 HuggingFace Inc. 团队所有
# 以 Apache 许可证 Version 2.0 (许可证) 授权
# 你不得使用本文件，除非符合许可证规定
# 可以在以下网址获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件都是基于"AS IS"基础分发的，
# 没有任何明示或暗示的担保或条件
# 请查看许可证以获取关于特定语言的具体权限和限制
""" VipLlava 模型配置 """

# 导入必要的模块和类
from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto import CONFIG_MAPPING

# 获取日志记录器实例
logger = logging.get_logger(__name__)

# 预训练模型配置文件的下载链接映射
VIPLLAVA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "ybelkada/vip-llava-7b-hf": "https://huggingface.co/llava-hf/vip-llava-7b-hf/resolve/main/config.json",
}

# VipLlava 模型配置类，继承自 PretrainedConfig 类
class VipLlavaConfig(PretrainedConfig):
    r"""
    这是一个配置保存 `VipLlavaForConditionalGeneration` 模型的配置类。根据指定的参数实例化一个 VipLlava 模型，定义模型架构。
    使用默认值实例化配置将得到类似 VipLlava-9B 的配置。

    例如：[ybelkada/vip-llava-7b-hf](https://huggingface.co/ybelkada/vip-llava-7b-hf)

    配置对象继承自 `PretrainedConfig`，可用于控制模型输出。阅读 `PretrainedConfig` 的文档以获取更多信息。

    参数:
        vision_config (`VipLlavaVisionConfig`,  *optional*):
            自定义视觉配置或字典
        text_config (`Union[AutoConfig, dict]`, *optional*):
            文本骨干的配置对象。可以是 `LlamaConfig` 或 `MistralConfig` 中的任何一个。
        ignore_index (`int`, *optional*, defaults to -100):
            损失函数的忽略索引。
        image_token_index (`int`, *optional*, defaults to 32000):
            用于编码图像提示的图像令牌索引。
        projector_hidden_act (`str`, *optional*, defaults to `"gelu"`):
            多模态投影仪使用的激活函数。
        projector_layernorm_eps (`float`, *optional*, defaults to 1e-05):
            投影层规范层的层标准化 epsilon
        vision_feature_layers (`List[int]`, *optional*, defaults to `[-2, -5, -8, -11, 6]`):
            从中选择视觉特征的层列表。
        vocab_size (`int`, *optional*, defaults to 32000):
            VipLlava 模型的词汇量。定义 `~VipLlavaForConditionalGeneration` 调用时所传递的 `inputs_ids` 可表示的不同令牌数。

    示例:

    ```python
    # 导入所需的类
    >>> from transformers import VipLlavaForConditionalGeneration, VipLlavaConfig, CLIPVisionConfig, LlamaConfig

    # 初始化 CLIP-vision 配置
    >>> vision_config = CLIPVisionConfig()

    # 初始化 Llama 配置
    >>> text_config = LlamaConfig()

    # 初始化一个 vipllava-7b 风格的配置
    >>> configuration = VipLlavaConfig(vision_config, text_config)

    # 使用 vipllava-7b 风格的配置初始化一个模型
    >>> model = VipLlavaForConditionalGeneration(configuration)

    # 访问模型配置
    >>> configuration = model.config
    ```py

    # 模型类型
    model_type = "vipllava"
    # 是否为组合模型
    is_composition = False

    # 初始化函数
    def __init__(
        self,
        vision_config=None,
        text_config=None,
        ignore_index=-100,
        image_token_index=32000,
        projector_hidden_act="gelu",
        projector_layernorm_eps=1e-5,
        vision_feature_layers=[-2, -5, -8, -11, 6],
        vocab_size=32000,
        **kwargs,
    ):
        # 设置忽略索引
        self.ignore_index = ignore_index
        # 图像标记索引
        self.image_token_index = image_token_index
        # 投影层激活函数
        self.projector_hidden_act = projector_hidden_act
        # 投影层层归一化的 epsilon
        self.projector_layernorm_eps = projector_layernorm_eps
        # 视觉特征提取层
        self.vision_feature_layers = vision_feature_layers
        # 词汇表大小
        self.vocab_size = vocab_size

        # 视觉配置
        self.vision_config = vision_config

        # 如果传入的视觉配置是字典
        if isinstance(self.vision_config, dict):
            # 设置模型类型
            vision_config["model_type"] = (
                vision_config["model_type"] if "model_type" in vision_config else "clip_vision_model"
            )
            # 根据模型类型创建相应的配置
            self.vision_config = CONFIG_MAPPING[vision_config["model_type"]](**vision_config)
        # 如果未传入视觉配置
        elif vision_config is None:
            # 使用默认的 CLIP 视觉模型配置
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
        # 设置词汇表大小
        self.vocab_size = self.vocab_size

        # 文本配置
        self.text_config = text_config

        # 如果传入的文本配置是字典
        if isinstance(self.text_config, dict):
            # 设置模型类型
            text_config["model_type"] = text_config["model_type"] if "model_type" in text_config else "llama"
            # 根据模型类型创建相应的配置
            self.text_config = CONFIG_MAPPING[text_config["model_type"]](**text_config)
            # 更新词汇表大小
            self.vocab_size = self.text_config.vocab_size
        # 如果未传入文本配置
        elif text_config is None:
            # 使用默认的 Llama 文本模型配置
            self.text_config = CONFIG_MAPPING["llama"]()

        # 调用父类的初始化函数
        super().__init__(**kwargs)
```  
```