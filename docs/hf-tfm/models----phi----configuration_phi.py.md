# `.\transformers\models\phi\configuration_phi.py`

```py
# 设置编码格式为 utf-8
# 版权声明及许可协议
# 配置文件信息

# 导入预训练配置和日志工具
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取日志工具的记录器
logger = logging.get_logger(__name__)

# 预训练配置文件存档映射
PHI_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/phi-1": "https://huggingface.co/microsoft/phi-1/resolve/main/config.json",
    "microsoft/phi-1_5": "https://huggingface.co/microsoft/phi-1_5/resolve/main/config.json",
    "microsoft/phi-2": "https://huggingface.co/microsoft/phi-2/resolve/main/config.json",
}

# Phi 模型配置类
class PhiConfig(PretrainedConfig):
    r"""
    存储 [`PhiModel`] 的配置的配置类。它用于根据指定的参数来实例化 Phi 模型，定义模型架构。使用默认值来实例化配置会产生类似于 Phi 预训练模型 [microsoft/phi-1](https://huggingface.co/microsoft/phi-1) 的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型的输出。阅读 [`PretrainedConfig`] 的文档以了解更多信息。

    示例:

    ```python
    >>> from transformers import PhiModel, PhiConfig

    >>> # 初始化一个 Phi-1 风格的配置
    >>> configuration = PhiConfig.from_pretrained("microsoft/phi-1")

    >>> # 根据配置初始化一个模型
    >>> model = PhiModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```py"""

    model_type = "phi"
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
    ):  # 类的构造函数定义结束
        # 设置词汇表大小
        self.vocab_size = vocab_size
        # 设置隐藏层大小
        self.hidden_size = hidden_size
        # 设置中间层大小
        self.intermediate_size = intermediate_size
        # 设置隐藏层数量
        self.num_hidden_layers = num_hidden_layers
        # 设置注意力头数量
        self.num_attention_heads = num_attention_heads

        # 如果未指定键值头数量，则默认与注意力头数量相同
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        # 设置键值头数量
        self.num_key_value_heads = num_key_value_heads
        # 设置残差连接丢弃概率
        self.resid_pdrop = resid_pdrop
        # 设置嵌入丢弃概率
        self.embd_pdrop = embd_pdrop
        # 设置注意力丢弃概率
        self.attention_dropout = attention_dropout
        # 设置隐藏层激活函数
        self.hidden_act = hidden_act
        # 设置最大位置嵌入长度
        self.max_position_embeddings = max_position_embeddings
        # 设置初始化范围
        self.initializer_range = initializer_range
        # 设置层归一化 epsilon
        self.layer_norm_eps = layer_norm_eps
        # 设置是否使用缓存
        self.use_cache = use_cache
        # 设置 ROPE 相关参数
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.partial_rotary_factor = partial_rotary_factor
        self.qk_layernorm = qk_layernorm
        # 执行 ROPE 缩放验证
        self._rope_scaling_validation()

        # 调用父类构造函数
        super().__init__(
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    # 从 transformers.models.llama.configuration_llama.LlamaConfig._rope_scaling_validation 复制过来
    # ROPE 缩放验证函数
    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        # 如果未指定 ROPE 缩放参数，则直接返回
        if self.rope_scaling is None:
            return

        # 检查 ROPE 缩放参数是否为字典且包含两个字段
        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            raise ValueError(
                "`rope_scaling` must be a dictionary with with two fields, `type` and `factor`, "
                f"got {self.rope_scaling}"
            )
        # 获取 ROPE 缩放类型和因子
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        # 检查 ROPE 缩放类型是否有效
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
            raise ValueError(
                f"`rope_scaling`'s type field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        # 检查 ROPE 缩放因子是否有效
        if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor <= 1.0:
            raise ValueError(f"`rope_scaling`'s factor field must be a float > 1, got {rope_scaling_factor}")
```