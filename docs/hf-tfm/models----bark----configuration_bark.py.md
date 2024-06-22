# `.\transformers\models\bark\configuration_bark.py`

```py
# 设置文件编码为 UTF-8
# 版权声明及许可证信息
# 导入必要的模块
# 配置类文件的日志记录器
""" BARK 模型配置"""

import os  # 导入操作系统模块
from typing import Dict, Optional, Union  # 导入类型提示模块

from ...configuration_utils import PretrainedConfig  # 导入预训练配置工具模块
from ...utils import add_start_docstrings, logging  # 导入辅助工具函数和日志记录模块
from ..auto import CONFIG_MAPPING  # 导入自动配置映射模块

# 获取日志记录器
logger = logging.get_logger(__name__)

# BARK 预训练配置映射字典
BARK_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "suno/bark-small": "https://huggingface.co/suno/bark-small/resolve/main/config.json",  # BARK-Small 模型配置的预训练文件映射
    "suno/bark": "https://huggingface.co/suno/bark/resolve/main/config.json",  # BARK 模型配置的预训练文件映射
}

# BARK 子模型配置的起始文档字符串
BARK_SUBMODELCONFIG_START_DOCSTRING = """
    This is the configuration class to store the configuration of a [`{model}`]. It is used to instantiate the model
    according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Bark [suno/bark](https://huggingface.co/suno/bark)
    architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


"""
    # 定义函数参数列表
    Args:
        # 定义块大小参数，默认为 1024
        block_size (`int`, *optional*, defaults to 1024):
            # 该模型可能使用的最大序列长度。通常设置为一个较大的值，以防万一（例如，512、1024或2048）。
        input_vocab_size (`int`, *optional*, defaults to 10_048):
            # Bark 子模型的词汇表大小。定义了在调用 [`{model}`] 时可以由 `inputs_ids` 表示的不同标记数量。默认为 10_048，但应根据所选子模型慎重考虑。
        output_vocab_size (`int`, *optional*, defaults to 10_048):
            # Bark 子模型的输出词汇表大小。定义了在向前传递 [`{model}`] 时可以由 `output_ids` 表示的不同标记数量。默认为 10_048，但应根据所选子模型慎重考虑。
        num_layers (`int`, *optional*, defaults to 12):
            # 给定子模型中的隐藏层数量。
        num_heads (`int`, *optional*, defaults to 12):
            # Transformer 架构中每个注意力层的注意力头数量。
        hidden_size (`int`, *optional*, defaults to 768):
            # 架构中“中间”（通常称为前馈）层的维度。
        dropout (`float`, *optional*, defaults to 0.0):
            # 嵌入层、编码器和池化器中所有全连接层的丢弃概率。
        bias (`bool`, *optional*, defaults to `True`):
            # 是否在线性层和层归一化层中使用偏置。
        initializer_range (`float`, *optional*, defaults to 0.02):
            # 用于初始化所有权重矩阵的截断正态初始化器的标准差。
        use_cache (`bool`, *optional*, defaults to `True`):
            # 模型是否应返回最后的键/值注意力（并非所有模型都使用）。
# 定义一个Bark子模型配置类，继承自PretrainedConfig
class BarkSubModelConfig(PretrainedConfig):
    # 模型类型为"bark_module"
    model_type = "bark_module"
    # 推断时需要忽略的键
    keys_to_ignore_at_inference = ["past_key_values"]

    # 属性映射字典
    attribute_map = {
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
        "vocab_size": "input_vocab_size",
        "window_size": "block_size",
    }

    # 初始化方法
    def __init__(
        self,
        block_size=1024,
        input_vocab_size=10_048,
        output_vocab_size=10_048,
        num_layers=12,
        num_heads=12,
        hidden_size=768,
        dropout=0.0,
        bias=True,  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
        initializer_range=0.02,
        use_cache=True,
        **kwargs,
    ):
        # 初始化各属性
        self.block_size = block_size
        self.input_vocab_size = input_vocab_size
        self.output_vocab_size = output_vocab_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.bias = bias
        self.use_cache = use_cache
        self.initializer_range = initializer_range

        # 调用父类的初始化方法
        super().__init__(**kwargs)

    # 从预训练模型加载配置
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Union[str, os.PathLike],
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        **kwargs,
    ) -> "PretrainedConfig":
        # 设置参数
        kwargs["cache_dir"] = cache_dir
        kwargs["force_download"] = force_download
        kwargs["local_files_only"] = local_files_only
        kwargs["revision"] = revision

        # 设置token
        cls._set_token_in_kwargs(kwargs, token)

        # 获取配置字典
        config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)

        # 如果从Bark加载，则获取Bark配置字典
        if config_dict.get("model_type") == "bark":
            config_dict = config_dict[f"{cls.model_type}_config"]

        # 如果配置字典中包含模型类型且与当前模型类型不同，则发出警告
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type}. This is not supported for all configurations of models and can yield errors."
            )

        # 从配置字典创建配置对象
        return cls.from_dict(config_dict, **kwargs)
    # 访问模型配置信息
    configuration = model.config
)
# 定义 BarkSemanticConfig 类，继承自 BarkSubModelConfig 类
class BarkSemanticConfig(BarkSubModelConfig):
    # 设置模型类型为 "semantic"
    model_type = "semantic"

# 使用 add_start_docstrings 装饰器添加文档字符串，说明 BarkCoarseConfig 类的作用和示例用法
@add_start_docstrings(
    BARK_SUBMODELCONFIG_START_DOCSTRING.format(config="BarkCoarseConfig", model="BarkCoarseModel"),
    """
    Example:

    ```python
    >>> from transformers import BarkCoarseConfig, BarkCoarseModel

    >>> # Initializing a Bark sub-module style configuration
    >>> configuration = BarkCoarseConfig()

    >>> # Initializing a model (with random weights) from the suno/bark style configuration
    >>> model = BarkCoarseModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```py""",
)
# 定义 BarkCoarseConfig 类，继承自 BarkSubModelConfig 类
class BarkCoarseConfig(BarkSubModelConfig):
    # 设置模型类型为 "coarse_acoustics"
    model_type = "coarse_acoustics"

# 使用 add_start_docstrings 装饰器添加文档字符串，说明 BarkFineConfig 类的作用、参数含义和示例用法
@add_start_docstrings(
    BARK_SUBMODELCONFIG_START_DOCSTRING.format(config="BarkFineConfig", model="BarkFineModel"),
    """
        n_codes_total (`int`, *optional*, defaults to 8):
            The total number of audio codebooks predicted. Used in the fine acoustics sub-model.
        n_codes_given (`int`, *optional*, defaults to 1):
            The number of audio codebooks predicted in the coarse acoustics sub-model. Used in the acoustics
            sub-models.
    Example:

    ```python
    >>> from transformers import BarkFineConfig, BarkFineModel

    >>> # Initializing a Bark sub-module style configuration
    >>> configuration = BarkFineConfig()

    >>> # Initializing a model (with random weights) from the suno/bark style configuration
    >>> model = BarkFineModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```py""",
)
# 定义 BarkFineConfig 类，继承自 BarkSubModelConfig 类
class BarkFineConfig(BarkSubModelConfig):
    # 设置模型类型为 "fine_acoustics"
    model_type = "fine_acoustics"

    # 定义初始化方法，包括参数 tie_word_embeddings、n_codes_total 和 n_codes_given
    def __init__(self, tie_word_embeddings=True, n_codes_total=8, n_codes_given=1, **kwargs):
        # 初始化属性 n_codes_total 和 n_codes_given
        self.n_codes_total = n_codes_total
        self.n_codes_given = n_codes_given

        # 调用父类的初始化方法，并传入参数 tie_word_embeddings 和其他关键字参数
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

# 定义 BarkConfig 类，继承自 PretrainedConfig 类
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
    # 定义了类变量 model_type，表示该模型的类型为 "bark"
    model_type = "bark"

    # 初始化方法，用于创建一个 BarkModel 实例
    def __init__(
        self,
        # 语义模型的配置信息，默认为 None
        semantic_config: Dict = None,
        # 粗糙声学模型的配置信息，默认为 None
        coarse_acoustics_config: Dict = None,
        # 细致声学模型的配置信息，默认为 None
        fine_acoustics_config: Dict = None,
        # 编解码器模型的配置信息，默认为 None
        codec_config: Dict = None,
        # 初始化参数范围，默认为 0.02
        initializer_range=0.02,
        # 其他关键字参数
        **kwargs,
    ):
        # 如果语义模型配置信息为空，则使用默认配置
        if semantic_config is None:
            semantic_config = {}
            logger.info("semantic_config is None. initializing the semantic model with default values.")

        # 如果粗糙声学模型配置信息为空，则使用默认配置
        if coarse_acoustics_config is None:
            coarse_acoustics_config = {}
            logger.info("coarse_acoustics_config is None. initializing the coarse model with default values.")

        # 如果细致声学模型配置信息为空，则使用默认配置
        if fine_acoustics_config is None:
            fine_acoustics_config = {}
            logger.info("fine_acoustics_config is None. initializing the fine model with default values.")

        # 如果编解码器模型配置信息为空，则使用默认配置
        if codec_config is None:
            codec_config = {}
            logger.info("codec_config is None. initializing the codec model with default values.")

        # 根据配置信息创建相应的对象
        self.semantic_config = BarkSemanticConfig(**semantic_config)
        self.coarse_acoustics_config = BarkCoarseConfig(**coarse_acoustics_config)
        self.fine_acoustics_config = BarkFineConfig(**fine_acoustics_config)
        # 获取编解码器模型类型
        codec_model_type = codec_config["model_type"] if "model_type" in codec_config else "encodec"
        # 根据模型类型选择相应的配置
        self.codec_config = CONFIG_MAPPING[codec_model_type](**codec_config)

        # 初始化参数范围
        self.initializer_range = initializer_range

        # 调用父类的初始化方法
        super().__init__(**kwargs)

    # 类方法，用于从各个子模型的配置创建 BarkModel 的实例
    @classmethod
    def from_sub_model_configs(
        cls,
        # 语义模型的配置信息
        semantic_config: BarkSemanticConfig,
        # 粗糙声学模型的配置信息
        coarse_acoustics_config: BarkCoarseConfig,
        # 细致声学模型的配置信息
        fine_acoustics_config: BarkFineConfig,
        # 编解码器模型的配置信息
        codec_config: PretrainedConfig,
        # 其他关键字参数
        **kwargs,
``` 
    ):  
        r"""
        从bark子模型配置实例化一个BarkConfig（或派生类）。

        Returns:
            [`BarkConfig`]: 一个配置对象的实例
        """
        return cls(
            semantic_config=semantic_config.to_dict(),  # 将semantic_config转换为字典并传入实例化参数
            coarse_acoustics_config=coarse_acoustics_config.to_dict(),  # 将coarse_acoustics_config转换为字典并传入实例化参数
            fine_acoustics_config=fine_acoustics_config.to_dict(),  # 将fine_acoustics_config转换为字典并传入实例化参数
            codec_config=codec_config.to_dict(),  # 将codec_config转换为字典并传入实例化参数
            **kwargs,  # 传入额外的关键字参数
        )
```