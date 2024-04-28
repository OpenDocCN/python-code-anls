# `.\models\gpt_neox\configuration_gpt_neox.py`

```
# 模块的编码声明，指定编码为 UTF-8
# 版权声明，版权归 EleutherAI 和 The HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 版本，使用该文件需要遵守许可证的规定
# 可以在以下链接获取许可证的副本：http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则本软件按"原样"分发，不提供任何明示或暗示的担保或条件
# 有关更多信息，请参阅许可证

""" GPTNeoX 模型配置"""

# 从配置工具中导入 PretrainedConfig 类
from ...configuration_utils import PretrainedConfig
# 从工具模块中导入日志记录器
from ...utils import logging

# 获取日志记录器实例
logger = logging.get_logger(__name__)

# GPTNeoX 预训练模型配置文件映射
GPT_NEOX_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "EleutherAI/gpt-neox-20b": "https://huggingface.co/EleutherAI/gpt-neox-20b/resolve/main/config.json",
    # 在 https://huggingface.co/models?filter=gpt_neox 查看所有 GPTNeoX 模型
}

# GPTNeoX 模型配置类，继承自 PretrainedConfig 类
class GPTNeoXConfig(PretrainedConfig):
    r"""
    这是一个配置类，用于存储 [`GPTNeoXModel`] 的配置。它用于根据指定的参数实例化 GPTNeoX 模型，定义模型架构。
    使用默认参数实例化配置将产生类似于 GPTNeoX [EleutherAI/gpt-neox-20b](https://huggingface.co/EleutherAI/gpt-neox-20b) 架构的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。阅读 [`PretrainedConfig`] 的文档以获取更多信息。

    ```python
    >>> from transformers import GPTNeoXConfig, GPTNeoXModel

    >>> # 初始化 GPTNeoX gpt-neox-20b 风格的配置
    >>> configuration = GPTNeoXConfig()

    >>> # 从 gpt-neox-20b 风格的配置初始化一个模型（带有随机权重）
    >>> model = GPTNeoXModel(configuration)  # doctest: +SKIP

    >>> # 访问模型配置
    >>> configuration = model.config  # doctest: +SKIP
    ```"""

    model_type = "gpt_neox"

    # 初始化函数，定义了 GPTNeoX 模型的各种配置参数
    def __init__(
        self,
        vocab_size=50432,  # 词汇表大小
        hidden_size=6144,  # 隐藏层大小
        num_hidden_layers=44,  # 隐藏层数
        num_attention_heads=64,  # 注意力头数
        intermediate_size=24576,  # 中间层大小
        hidden_act="gelu",  # 隐藏层激活函数
        rotary_pct=0.25,  # 旋转参数的百分比
        rotary_emb_base=10000,  # 旋转嵌入的基数
        attention_dropout=0.0,  # 注意力机制的 dropout
        hidden_dropout=0.0,  # 隐藏层 dropout
        classifier_dropout=0.1,  # 分类器 dropout
        max_position_embeddings=2048,  # 最大位置嵌入
        initializer_range=0.02,  # 初始化范围
        layer_norm_eps=1e-5,  # 层归一化的 epsilon
        use_cache=True,  # 是否使用缓存
        bos_token_id=0,  # 起始符号的标识符
        eos_token_id=2,  # 结束符号的标识符
        tie_word_embeddings=False,  # 是否绑定词嵌入
        use_parallel_residual=True,  # 是否使用并行残差连接
        rope_scaling=None,  # ROPE 缩放
        attention_bias=True,  # 是否使用注意力偏置
        **kwargs,  # 其他未命名参数
    # 调用父类的初始化方法，传入开始标记 ID 和结束标记 ID，以及其他关键字参数
    super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
    # 设置词汇表大小
    self.vocab_size = vocab_size
    # 设置最大位置嵌入长度
    self.max_position_embeddings = max_position_embeddings
    # 设置隐藏层大小
    self.hidden_size = hidden_size
    # 设置隐藏层数量
    self.num_hidden_layers = num_hidden_layers
    # 设置注意力头数量
    self.num_attention_heads = num_attention_heads
    # 设置中间层大小
    self.intermediate_size = intermediate_size
    # 设置隐藏层激活函数
    self.hidden_act = hidden_act
    # 设置旋转注意力的百分比
    self.rotary_pct = rotary_pct
    # 设置旋转嵌入的基础
    self.rotary_emb_base = rotary_emb_base
    # 设置注意力机制的 dropout 率
    self.attention_dropout = attention_dropout
    # 设置隐藏层的 dropout 率
    self.hidden_dropout = hidden_dropout
    # 设置分类器的 dropout 率
    self.classifier_dropout = classifier_dropout
    # 设置初始化范围
    self.initializer_range = initializer_range
    # 设置层归一化的 epsilon 值
    self.layer_norm_eps = layer_norm_eps
    # 设置是否使用缓存
    self.use_cache = use_cache
    # 设置是否共享词嵌入
    self.tie_word_embeddings = tie_word_embeddings
    # 设置是否使用并行残差连接
    self.use_parallel_residual = use_parallel_residual
    # 设置 ROPE 缩放因子
    self.rope_scaling = rope_scaling
    # 设置注意力偏置
    self.attention_bias = attention_bias
    # 执行 ROPE 缩放验证方法
    self._rope_scaling_validation()

    # 以下为 ROPE 缩放验证方法的定义
    # Copied from transformers.models.llama.configuration_llama.LlamaConfig._rope_scaling_validation
    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        # 如果 ROPE 缩放参数为 None，则直接返回
        if self.rope_scaling is None:
            return

        # 如果 ROPE 缩放参数不是字典或字典长度不为 2，则抛出异常
        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            raise ValueError(
                "`rope_scaling` must be a dictionary with with two fields, `type` and `factor`, "
                f"got {self.rope_scaling}"
            )
        # 获取 ROPE 缩放类型和缩放因子
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        # 如果缩放类型为 None 或不在 ['linear', 'dynamic'] 中，则抛出异常
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
            raise ValueError(
                f"`rope_scaling`'s type field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        # 如果缩放因子为 None 或不是大于 1 的浮点数，则抛出异常
        if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor <= 1.0:
            raise ValueError(f"`rope_scaling`'s factor field must be a float > 1, got {rope_scaling_factor}")
```