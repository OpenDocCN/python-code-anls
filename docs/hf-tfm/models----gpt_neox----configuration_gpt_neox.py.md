# `.\models\gpt_neox\configuration_gpt_neox.py`

```
# coding=utf-8
# 版权所有 2022 EleutherAI 和 The HuggingFace Inc. 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）获得许可；
# 除非符合许可证要求，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件根据“原样”基础分发，
# 没有任何明示或暗示的担保或条件。
# 有关特定语言的权限，请参阅许可证。

""" GPTNeoX 模型配置"""

# 从 transformers 库中导入预训练配置类 PretrainedConfig
from ...configuration_utils import PretrainedConfig
# 从 transformers 库中导入日志记录工具 logging
from ...utils import logging

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义预训练配置文件映射字典，将模型名称映射到其配置文件的 URL
GPT_NEOX_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "EleutherAI/gpt-neox-20b": "https://huggingface.co/EleutherAI/gpt-neox-20b/resolve/main/config.json",
    # 查看所有 GPTNeoX 模型的列表，请访问 https://huggingface.co/models?filter=gpt_neox
}

# 定义 GPTNeoXConfig 类，继承自 PretrainedConfig 类
class GPTNeoXConfig(PretrainedConfig):
    r"""
    这是一个配置类，用于存储 [`GPTNeoXModel`] 的配置。它用于根据指定的参数实例化一个 GPTNeoX 模型，
    定义模型架构。使用默认参数实例化配置将产生类似于 GPTNeoX [EleutherAI/gpt-neox-20b]
    (https://huggingface.co/EleutherAI/gpt-neox-20b) 架构的配置。

    配置对象继承自 [`PretrainedConfig`]，可以用于控制模型的输出。更多信息请参阅 [`PretrainedConfig`] 的文档。

    ```python
    >>> from transformers import GPTNeoXConfig, GPTNeoXModel

    >>> # 初始化一个 GPTNeoX gpt-neox-20b 风格的配置
    >>> configuration = GPTNeoXConfig()

    >>> # 使用配置初始化一个（具有随机权重的）gpt-neox-20b 风格的模型
    >>> model = GPTNeoXModel(configuration)  # doctest: +SKIP

    >>> # 访问模型配置
    >>> configuration = model.config  # doctest: +SKIP
    ```
    """

    # 模型类型设为 "gpt_neox"
    model_type = "gpt_neox"

    # 初始化函数，设定模型配置的各种参数
    def __init__(
        self,
        vocab_size=50432,
        hidden_size=6144,
        num_hidden_layers=44,
        num_attention_heads=64,
        intermediate_size=24576,
        hidden_act="gelu",
        rotary_pct=0.25,
        rotary_emb_base=10000,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        classifier_dropout=0.1,
        max_position_embeddings=2048,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        use_cache=True,
        bos_token_id=0,
        eos_token_id=2,
        tie_word_embeddings=False,
        use_parallel_residual=True,
        rope_scaling=None,
        attention_bias=True,
        **kwargs,
    ):
        # 调用父类的初始化方法，传入起始标记和结束标记的 token ID，以及其他可选参数
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        # 设置词汇表大小
        self.vocab_size = vocab_size
        # 设置最大位置嵌入的长度
        self.max_position_embeddings = max_position_embeddings
        # 设置隐藏层大小
        self.hidden_size = hidden_size
        # 设置隐藏层数量
        self.num_hidden_layers = num_hidden_layers
        # 设置注意力头的数量
        self.num_attention_heads = num_attention_heads
        # 设置中间层大小
        self.intermediate_size = intermediate_size
        # 设置隐藏层的激活函数
        self.hidden_act = hidden_act
        # 设置旋转注意力的百分比
        self.rotary_pct = rotary_pct
        # 设置旋转嵌入的基数
        self.rotary_emb_base = rotary_emb_base
        # 设置注意力机制的 dropout 比例
        self.attention_dropout = attention_dropout
        # 设置隐藏层的 dropout 比例
        self.hidden_dropout = hidden_dropout
        # 设置分类器的 dropout 比例
        self.classifier_dropout = classifier_dropout
        # 设置初始化权重的范围
        self.initializer_range = initializer_range
        # 设置层归一化的 epsilon 值
        self.layer_norm_eps = layer_norm_eps
        # 设置是否使用缓存
        self.use_cache = use_cache
        # 设置是否绑定词嵌入
        self.tie_word_embeddings = tie_word_embeddings
        # 设置是否使用并行残差连接
        self.use_parallel_residual = use_parallel_residual
        # 设置注意力偏置
        self.attention_bias = attention_bias
        # 验证并调整旋转注意力的设置
        self._rope_scaling_validation()

        # 如果隐藏层大小不能被注意力头数量整除，抛出数值错误异常
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                "The hidden size is not divisble by the number of attention heads! Make sure to update them!"
            )

    # 从 transformers.models.llama.configuration_llama.LlamaConfig._rope_scaling_validation 复制而来
    def _rope_scaling_validation(self):
        """
        验证 `rope_scaling` 配置是否有效。
        """
        # 如果 `rope_scaling` 为 None，直接返回
        if self.rope_scaling is None:
            return

        # 如果 `rope_scaling` 不是字典类型或者字典长度不为 2，抛出数值错误异常
        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            raise ValueError(
                "`rope_scaling` must be a dictionary with with two fields, `type` and `factor`, "
                f"got {self.rope_scaling}"
            )
        
        # 获取 `rope_scaling` 的类型和因子
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        
        # 如果 `rope_scaling` 的类型为空或者不在 ['linear', 'dynamic'] 中，抛出数值错误异常
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
            raise ValueError(
                f"`rope_scaling`'s type field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        
        # 如果 `rope_scaling` 的因子为空或者不是大于 1 的浮点数，抛出数值错误异常
        if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor <= 1.0:
            raise ValueError(f"`rope_scaling`'s factor field must be a float > 1, got {rope_scaling_factor}")
```