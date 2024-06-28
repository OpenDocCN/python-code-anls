# `.\models\phi\configuration_phi.py`

```
# 定义一个 Python 源码文件的编码格式为 UTF-8
# 版权声明和许可证信息，这里使用 Apache License, Version 2.0
# 导入所需的模块和类
from ...configuration_utils import PretrainedConfig  # 导入预训练配置类
from ...utils import logging  # 导入日志记录工具

# 获取全局的日志记录器
logger = logging.get_logger(__name__)

# 定义一个字典，映射预训练模型名称到其配置文件的 URL
PHI_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/phi-1": "https://huggingface.co/microsoft/phi-1/resolve/main/config.json",
    "microsoft/phi-1_5": "https://huggingface.co/microsoft/phi-1_5/resolve/main/config.json",
    "microsoft/phi-2": "https://huggingface.co/microsoft/phi-2/resolve/main/config.json",
}

# PhiConfig 类，用于存储 PhiModel 的配置信息，继承自 PretrainedConfig 类
class PhiConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`PhiModel`]. It is used to instantiate an Phi
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Phi
    [microsoft/phi-1](https://huggingface.co/microsoft/phi-1).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Example:

    ```python
    >>> from transformers import PhiModel, PhiConfig

    >>> # Initializing a Phi-1 style configuration
    >>> configuration = PhiConfig.from_pretrained("microsoft/phi-1")

    >>> # Initializing a model from the configuration
    >>> model = PhiModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```

    """
    # 模型类型为 "phi"
    model_type = "phi"
    # 推断时忽略的键列表
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=51200,
        hidden_size=2048,
        intermediate_size=8192,
        num_hidden_layers=24,
        num_attention_heads=32,
        num_key_value_heads=None,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attention_dropout=0.0,
        hidden_act="gelu_new",
        max_position_embeddings=2048,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        partial_rotary_factor=0.5,
        qk_layernorm=False,
        bos_token_id=1,
        eos_token_id=2,
        **kwargs,
    ):
        # 初始化 PhiConfig 实例，设置各种模型配置参数
        # 这些参数决定了 PhiModel 的架构和行为
        pass
        ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attention_dropout = attention_dropout
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.partial_rotary_factor = partial_rotary_factor
        self.qk_layernorm = qk_layernorm

        # 调用私有方法 _rope_scaling_validation() 进行 ROPE 缩放参数的验证
        self._rope_scaling_validation()

        # 调用父类初始化方法，传递相关参数
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    # 从 transformers.models.llama.configuration_llama.LlamaConfig._rope_scaling_validation 复制而来
    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        # 如果 rope_scaling 参数为 None，则直接返回，无需验证
        if self.rope_scaling is None:
            return

        # 如果 rope_scaling 不是字典类型或者长度不为 2，则抛出数值错误异常
        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            raise ValueError(
                "`rope_scaling` must be a dictionary with with two fields, `type` and `factor`, "
                f"got {self.rope_scaling}"
            )
        
        # 获取 rope_scaling 字典中的 type 和 factor 字段
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        
        # 如果 type 字段为 None 或者不在 ['linear', 'dynamic'] 中，则抛出数值错误异常
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
            raise ValueError(
                f"`rope_scaling`'s type field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        
        # 如果 factor 字段为 None 或者不是大于 1 的浮点数，则抛出数值错误异常
        if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor <= 1.0:
            raise ValueError(f"`rope_scaling`'s factor field must be a float > 1, got {rope_scaling_factor}")
```