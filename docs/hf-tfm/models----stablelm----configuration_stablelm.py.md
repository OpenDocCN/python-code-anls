# `.\models\stablelm\configuration_stablelm.py`

```
# coding=utf-8
# 版权声明及许可信息
# 本文件用于定义 StableLM 模型的配置类

# 从 Transformers 库中导入预训练配置类 PretrainedConfig
from ...configuration_utils import PretrainedConfig
# 从 Transformers 库中导入日志记录工具 logging
from ...utils import logging

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 预训练模型配置文件映射表，指定不同模型的配置文件下载链接
STABLELM_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "stabilityai/stablelm-3b-4e1t": "https://huggingface.co/stabilityai/stablelm-3b-4e1t/resolve/main/config.json",
    # 查看所有 StableLM 模型的链接地址：https://huggingface.co/models?filter=stablelm
}

# 定义 StableLmConfig 类，继承自 PretrainedConfig
class StableLmConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`~StableLmModel`].
    It is used to instantiate an StableLM model according to the specified arguments, defining the model
    architecture. Instantiating a configuration with the defaults will yield a similar configuration to that of
    the StableLM [stabilityai/stablelm-3b-4e1t](https://huggingface.co/stabilityai/stablelm-3b-4e1t) architecture.

    Configuration objects inherit from  [`PretrainedConfig`] and can be used
    to control the model outputs. Read the documentation from  [`PretrainedConfig`]
    for more information.

    Example:

    ```python
    >>> from transformers import StableLmModel, StableLmConfig

    >>> # Initializing a StableLM stablelm-3b style configuration
    >>> configuration = StableLmConfig()
    ```
    """

    # 模型类型标识为 "stablelm"
    model_type = "stablelm"
    # 推断时忽略的键列表
    keys_to_ignore_at_inference = ["past_key_values"]

    # 初始化方法，设置模型配置的各项参数
    def __init__(
        self,
        vocab_size=50304,  # 词汇表大小
        intermediate_size=6912,  # 中间层大小
        hidden_size=2560,  # 隐藏层大小
        num_hidden_layers=32,  # 隐藏层层数
        num_attention_heads=32,  # 注意力头数
        num_key_value_heads=32,  # 键值头数
        hidden_act="silu",  # 隐藏层激活函数
        max_position_embeddings=4096,  # 最大位置嵌入数
        initializer_range=0.02,  # 初始化范围
        layer_norm_eps=1.0e-5,  # 层归一化 epsilon 参数
        use_cache=True,  # 是否使用缓存
        tie_word_embeddings=False,  # 是否绑定词嵌入
        rope_theta=10_000,  # 绳索 theta 参数
        rope_scaling=None,  # 绳索缩放参数
        use_qkv_bias=False,  # 是否使用 QKV 偏置
        hidden_dropout=0.0,  # 隐藏层 dropout 率
        attention_dropout=0.0,  # 注意力 dropout 率
        partial_rotary_factor=0.25,  # 部分旋转因子
        bos_token_id=0,  # 起始符号 ID
        eos_token_id=0,  # 终止符号 ID
        **kwargs,  # 其他参数
    ):
        # 调用父类的初始化方法，设置模型配置参数
        super().__init__(
            vocab_size=vocab_size,
            intermediate_size=intermediate_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            layer_norm_eps=layer_norm_eps,
            use_cache=use_cache,
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            use_qkv_bias=use_qkv_bias,
            hidden_dropout=hidden_dropout,
            attention_dropout=attention_dropout,
            partial_rotary_factor=partial_rotary_factor,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        ):
        self.vocab_size = vocab_size  # 设置对象的词汇量大小
        self.max_position_embeddings = max_position_embeddings  # 设置对象的最大位置编码长度

        self.hidden_size = hidden_size  # 设置对象的隐藏层大小
        self.intermediate_size = intermediate_size  # 设置对象的中间层大小
        self.num_hidden_layers = num_hidden_layers  # 设置对象的隐藏层数量
        self.num_attention_heads = num_attention_heads  # 设置对象的注意力头数量
        self.num_key_value_heads = num_key_value_heads  # 设置对象的键值头数量
        self.hidden_act = hidden_act  # 设置对象的隐藏层激活函数类型

        self.initializer_range = initializer_range  # 设置对象的初始化范围
        self.layer_norm_eps = layer_norm_eps  # 设置对象的层归一化 epsilon 参数
        self.use_cache = use_cache  # 设置对象是否使用缓存
        self.rope_theta = rope_theta  # 设置对象的绳子角度
        self.rope_scaling = rope_scaling  # 设置对象的绳子缩放配置
        self.use_qkv_bias = use_qkv_bias  # 设置对象是否使用查询键值的偏置
        self.hidden_dropout = hidden_dropout  # 设置对象的隐藏层dropout率
        self.attention_dropout = attention_dropout  # 设置对象的注意力dropout率
        self.partial_rotary_factor = partial_rotary_factor  # 设置对象的部分旋转因子
        self._rope_scaling_validation()  # 调用内部方法验证绳子缩放配置的有效性

        super().__init__(  # 调用父类初始化方法，传递额外的参数
            bos_token_id=bos_token_id,  # 开始标记的 token id
            eos_token_id=eos_token_id,  # 结束标记的 token id
            tie_word_embeddings=tie_word_embeddings,  # 是否共享词嵌入
            **kwargs,  # 传递任意额外的关键字参数
        )

    # 从 transformers.models.llama.configuration_llama.LlamaConfig._rope_scaling_validation 复制而来
    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        if self.rope_scaling is None:  # 如果绳子缩放配置为 None，直接返回
            return

        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:  # 如果绳子缩放配置不是字典或者长度不为2，抛出数值错误
            raise ValueError(
                "`rope_scaling` must be a dictionary with with two fields, `type` and `factor`, "
                f"got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)  # 获取绳子缩放配置的类型字段
        rope_scaling_factor = self.rope_scaling.get("factor", None)  # 获取绳子缩放配置的因子字段
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:  # 如果类型字段为空或者不在预定义的类型列表中，抛出数值错误
            raise ValueError(
                f"`rope_scaling`'s type field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor <= 1.0:  # 如果因子字段为空或者不是浮点数或者小于等于1，抛出数值错误
            raise ValueError(f"`rope_scaling`'s factor field must be a float > 1, got {rope_scaling_factor}")
```