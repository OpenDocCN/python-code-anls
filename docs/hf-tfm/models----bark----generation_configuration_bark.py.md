# `.\transformers\models\bark\generation_configuration_bark.py`

```py
# 设置文件编码为 UTF-8
# 版权声明
# 根据 Apache 许可证 2.0 版本，除非符合许可证，否则不得使用此文件
# 可以在以下网址获取许可证副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"AS IS"的基础分发的，没有任何明示或暗示的保证或条件
# 请查看许可证以获取有关特定语言的权限和限制
""" BARK 模型生成配置"""

# 导入必要的库
import copy
from typing import Dict

# 导入生成配置工具和日志记录工具
from ...generation.configuration_utils import GenerationConfig
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义 BarkSemanticGenerationConfig 类，继承自 GenerationConfig 类
class BarkSemanticGenerationConfig(GenerationConfig):
    model_type = "semantic"

    # 初始化方法，设置各种参数的默认值
    def __init__(
        self,
        eos_token_id=10_000,
        renormalize_logits=True,
        max_new_tokens=768,
        output_scores=False,
        return_dict_in_generate=False,
        output_hidden_states=False,
        output_attentions=False,
        temperature=1.0,
        do_sample=False,
        text_encoding_offset=10_048,
        text_pad_token=129_595,
        semantic_infer_token=129_599,
        semantic_vocab_size=10_000,
        max_input_semantic_length=256,
        semantic_rate_hz=49.9,
        min_eos_p=None,
        **kwargs,
        
# 定义 BarkCoarseGenerationConfig 类，继承自 GenerationConfig 类
class BarkCoarseGenerationConfig(GenerationConfig):
    model_type = "coarse_acoustics"

    # 初始化方法，设置各种参数的默认值
    def __init__(
        self,
        renormalize_logits=True,
        output_scores=False,
        return_dict_in_generate=False,
        output_hidden_states=False,
        output_attentions=False,
        temperature=1.0,
        do_sample=False,
        coarse_semantic_pad_token=12_048,
        coarse_rate_hz=75,
        n_coarse_codebooks=2,
        coarse_infer_token=12_050,
        max_coarse_input_length=256,
        max_coarse_history: int = 630,
        sliding_window_len: int = 60,
        **kwargs,
        
# 定义 BarkFineGenerationConfig 类，继承自 GenerationConfig 类
class BarkFineGenerationConfig(GenerationConfig):
    model_type = "fine_acoustics"

    # 初始化方法，设置各种参数的默认值
    def __init__(
        self,
        temperature=1.0,
        max_fine_history_length=512,
        max_fine_input_length=1024,
        n_fine_codebooks=8,
        **kwargs,
    ):
        """Class that holds a generation configuration for [`BarkFineModel`].

        [`BarkFineModel`] is an autoencoder model, so should not usually be used for generation. However, under the
        hood, it uses `temperature` when used by [`BarkModel`]

        This configuration inherit from [`GenerationConfig`] and can be used to control the model generation. Read the
        documentation from [`GenerationConfig`] for more information.

        Args:
            temperature (`float`, *optional*):
                The value used to modulate the next token probabilities.
            max_fine_history_length (`int`, *optional*, defaults to 512):
                Max length of the fine history vector.
            max_fine_input_length (`int`, *optional*, defaults to 1024):
                Max length of fine input vector.
            n_fine_codebooks (`int`, *optional*, defaults to 8):
                Number of codebooks used.
        """
        # 调用父类构造函数，传入温度参数
        super().__init__(temperature=temperature)

        # 设置最大精细历史向量长度
        self.max_fine_history_length = max_fine_history_length
        # 设置最大精细输入向量长度
        self.max_fine_input_length = max_fine_input_length
        # 设置精细码书数量
        self.n_fine_codebooks = n_fine_codebooks

    def validate(self, **kwargs):
        """
        Overrides GenerationConfig.validate because BarkFineGenerationConfig don't use any parameters outside
        temperature.
        """
        # 重写 validate 方法，因为 BarkFineGenerationConfig 类不使用除温度外的任何参数
        # 此处不执行任何验证操作，直接 pass
        pass
class BarkGenerationConfig(GenerationConfig):
    model_type = "bark"
    is_composition = True

    # TODO (joao): nested from_dict

    def __init__(
        self,
        semantic_config: Dict = None,
        coarse_acoustics_config: Dict = None,
        fine_acoustics_config: Dict = None,
        sample_rate=24_000,
        codebook_size=1024,
        **kwargs,
    ):
        """Class that holds a generation configuration for [`BarkModel`].

        The [`BarkModel`] does not have a `generate` method, but uses this class to generate speeches with a nested
        [`BarkGenerationConfig`] which uses [`BarkSemanticGenerationConfig`],
        [`BarkCoarseGenerationConfig`], [`BarkFineGenerationConfig`].

        This configuration inherit from [`GenerationConfig`] and can be used to control the model generation. Read the
        documentation from [`GenerationConfig`] for more information.

        Args:
            semantic_config (`Dict`, *optional*):
                Semantic generation configuration.
            coarse_acoustics_config (`Dict`, *optional*):
                Coarse generation configuration.
            fine_acoustics_config (`Dict`, *optional*):
                Fine generation configuration.
            sample_rate (`int`, *optional*, defaults to 24_000):
                Sample rate.
            codebook_size (`int`, *optional*, defaults to 1024):
                Vector length for each codebook.
        """
        # 若 semantic_config 未提供，则初始化为空字典，并记录日志
        if semantic_config is None:
            semantic_config = {}
            logger.info("semantic_config is None. initializing the semantic model with default values.")

        # 若 coarse_acoustics_config 未提供，则初始化为空字典，并记录日志
        if coarse_acoustics_config is None:
            coarse_acoustics_config = {}
            logger.info("coarse_acoustics_config is None. initializing the coarse model with default values.")

        # 若 fine_acoustics_config 未提供，则初始化为空字典，并记录日志
        if fine_acoustics_config is None:
            fine_acoustics_config = {}
            logger.info("fine_acoustics_config is None. initializing the fine model with default values.")

        # 创建 BarkSemanticGenerationConfig 实例，如果提供了配置则使用提供的配置，否则使用空字典
        self.semantic_config = BarkSemanticGenerationConfig(**semantic_config)
        # 创建 BarkCoarseGenerationConfig 实例，如果提供了配置则使用提供的配置，否则使用空字典
        self.coarse_acoustics_config = BarkCoarseGenerationConfig(**coarse_acoustics_config)
        # 创建 BarkFineGenerationConfig 实例，如果提供了配置则使用提供的配置，否则使用空字典
        self.fine_acoustics_config = BarkFineGenerationConfig(**fine_acoustics_config)

        # 设置采样率和码本大小
        self.sample_rate = sample_rate
        self.codebook_size = codebook_size

    @classmethod
    def from_sub_model_configs(
        cls,
        semantic_config: BarkSemanticGenerationConfig,
        coarse_acoustics_config: BarkCoarseGenerationConfig,
        fine_acoustics_config: BarkFineGenerationConfig,
        **kwargs,
    ):
        r"""
        从 bark 子模型生成配置实例化一个 [`BarkGenerationConfig`]（或其派生类）。

        Returns:
            [`BarkGenerationConfig`]: 一个配置对象的实例
        """
        return cls(
            # 将语义配置转换为字典形式
            semantic_config=semantic_config.to_dict(),
            # 将粗略声学配置转换为字典形式
            coarse_acoustics_config=coarse_acoustics_config.to_dict(),
            # 将细致声学配置转换为字典形式
            fine_acoustics_config=fine_acoustics_config.to_dict(),
            **kwargs,
        )

    def to_dict(self):
        """
        将此实例序列化为 Python 字典。覆盖默认的 [`~PretrainedConfig.to_dict`]。

        Returns:
            `Dict[str, any]`: 构成此配置实例的所有属性的字典，
        """
        # 深度复制实例的字典形式
        output = copy.deepcopy(self.__dict__)

        # 将语义配置转换为字典形式
        output["semantic_config"] = self.semantic_config.to_dict()
        # 将粗略声学配置转换为字典形式
        output["coarse_acoustics_config"] = self.coarse_acoustics_config.to_dict()
        # 将细致声学配置转换为字典形式
        output["fine_acoustics_config"] = self.fine_acoustics_config.to_dict()

        # 添加模型类型到输出字典
        output["model_type"] = self.__class__.model_type
        return output
```