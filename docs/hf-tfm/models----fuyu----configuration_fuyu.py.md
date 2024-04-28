# `.\models\fuyu\configuration_fuyu.py`

```
# 导入所需模块和类
# 配置文件的版权声明和许可协议信息
# 声明模型配置类的起始位置，用于存储 `FuyuForCausalLM` 模型的配置
from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ..auto import CONFIG_MAPPING

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# 模型名称到预训练配置文件的映射
FUYU_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "adept/fuyu-8b": "https://huggingface.co/adept/fuyu-8b/resolve/main/config.json",
}

# `FuyuConfig` 类继承自 `PretrainedConfig`，用于存储 `FuyuForCausalLM` 模型的配置
class FuyuConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`FuyuForCausalLM`]. It is used to instantiate an
    Fuyu model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the
    [adept/fuyu-8b](https://huggingface.co/adept/fuyu-8b).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    ```python
    >>> from transformers import FuyuConfig

    >>> # Initializing a Fuyu fuyu-7b style configuration
    >>> configuration = FuyuConfig()
    ```"""

    # 模型类型为 "fuyu"
    model_type = "fuyu"
    # 推断时忽略的键列表
    keys_to_ignore_at_inference = ["past_key_values"]

    # 初始化方法，用于设置模型配置的各个参数
    def __init__(
        self,
        vocab_size=262144,
        hidden_size=4096,
        intermediate_size=16384,
        num_hidden_layers=36,
        num_attention_heads=64,
        hidden_act="relu2",
        max_position_embeddings=16384,
        image_size=300,
        patch_size=30,
        num_channels=3,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=25000.0,
        rope_scaling=None,
        qk_layernorm=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        partial_rotary_factor=0.5,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        text_config=None,
        **kwargs,
    # 如果text_config为空，则使用默认值来初始化text_config
    if text_config is None:
        text_config = {
            "vocab_size": vocab_size,
            "max_position_embeddings": max_position_embeddings,
            "hidden_size": hidden_size,
            "intermediate_size": intermediate_size,
            "num_hidden_layers": num_hidden_layers,
            "num_attention_heads": num_attention_heads,
            "hidden_act": hidden_act,
            "initializer_range": initializer_range,
            "layer_norm_eps": layer_norm_eps,
            "use_cache": use_cache,
            "rope_theta": rope_theta,
            "rope_scaling": rope_scaling,
            "qk_layernorm": qk_layernorm,
            "hidden_dropout": hidden_dropout,
            "attention_dropout": attention_dropout,
            "partial_rotary_factor": partial_rotary_factor,
            "pad_token_id": pad_token_id,
            "bos_token_id": bos_token_id,
            "eos_token_id": eos_token_id,
            "tie_word_embeddings": tie_word_embeddings,
        }
        logger.info("text_config is None. initializing the text model with default values.")
    # 获取text_config中的model_type，如果不存在则使用默认值"persimmon"
    text_model_type = text_config["model_type"] if "model_type" in text_config else "persimmon"
    # 根据不同的model_type选择相应的配置信息来初始化text_config
    self.text_config = CONFIG_MAPPING[text_model_type](**text_config)

    # 初始化模型的各项参数
    self.vocab_size = vocab_size
    self.max_position_embeddings = max_position_embeddings
    self.image_size = image_size
    self.patch_size = patch_size
    self.num_channels = num_channels
    self.hidden_size = hidden_size
    self.intermediate_size = intermediate_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.hidden_act = hidden_act
    self.initializer_range = initializer_range
    self.layer_norm_eps = layer_norm_eps
    self.use_cache = use_cache
    self.rope_theta = rope_theta
    self.rope_scaling = rope_scaling
    self.qk_layernorm = qk_layernorm
    self.hidden_dropout = hidden_dropout
    self.attention_dropout = attention_dropout
    self.partial_rotary_factor = partial_rotary_factor
    # 验证rope_scaling的值是否合法
    self._rope_scaling_validation()

    # 调用父类的初始化方法，并传入相应参数
    super().__init__(
        pad_token_id=pad_token_id,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        tie_word_embeddings=tie_word_embeddings,
        **kwargs,
    )

# 被复制的方法，用于验证rope_scaling的值是否合法，暂未给出具体实现
# Copied from transformers.models.llama.configuration_llama.LlamaConfig._rope_scaling_validation
    # 对 `rope_scaling` 配置进行验证
    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        # 如果 `rope_scaling` 为 None，则直接返回
        if self.rope_scaling is None:
            return

        # 如果 `rope_scaling` 不是字典类型或长度不为2，则抛出数值错误异常
        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            raise ValueError(
                "`rope_scaling` must be a dictionary with with two fields, `type` and `factor`, "
                f"got {self.rope_scaling}"
            )
        # 获取 `rope_scaling` 中的 'type' 和 'factor' 字段值
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        # 如果 'type' 为 None 或不在 ['linear', 'dynamic'] 中，则抛出数值错误异常
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
            raise ValueError(
                f"`rope_scaling`'s type field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        # 如果 'factor' 为 None 或不是大于1的浮点数，则抛出数值错误异常
        if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor <= 1.0:
            raise ValueError(f"`rope_scaling`'s factor field must be a float > 1, got {rope_scaling_factor}")
```