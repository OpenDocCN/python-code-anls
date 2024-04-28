# `.\transformers\models\persimmon\configuration_persimmon.py`

```
# 设置文件编码为 utf-8
# 版权声明
# Adept AI 和 HuggingFace Inc. 团队，版权所有。
#
# 根据 Apache 许可证 2.0 版本 ("许可证") 获取的原始文件
# 除许可证允许外，您不得使用此文件
# 您可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则不得
# 分发的软件根据许可证"原样"分发
# 没有任何形式的保证或条件，无论明示或默示。
# 有关特定语言控制输出的条件
# 和许可证中的限制
""" Persimmon 模型配置"""

# 导入必要的模块和类
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 预训练配置映射
PERSIMMON_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "adept/persimmon-8b-base": "https://huggingface.co/adept/persimmon-8b-base/resolve/main/config.json",
}

# Persimmon 配置类，用于存储 PersimmonModel 的配置
class PersimmonConfig(PretrainedConfig):
    r"""
    此类用于存储 [`PersimmonModel`] 的配置。根据指定的参数实例化 Persimmon 模型的构架。
    使用默认值实例化配置将产生类似 [adept/persimmon-8b-base](https://huggingface.co/adept/persimmon-8b-base) 的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型输出。阅读 [`PretrainedConfig`] 的文档以了解更多信息。

    ```python
    >>> from transformers import PersimmonModel, PersimmonConfig

    >>> # 初始化一个 Persimmon 模型 persimmon-7b 风格配置
    >>> configuration = PersimmonConfig()
    ```"""

    # 模型类型为 "persimmon"
    model_type = "persimmon"
    # 推理阶段要忽略的键
    keys_to_ignore_at_inference = ["past_key_values"]

    # 初始化方法
    def __init__(
        self,
        vocab_size=262144,
        hidden_size=4096,
        intermediate_size=16384,
        num_hidden_layers=36,
        num_attention_heads=64,
        hidden_act="relu2",
        max_position_embeddings=16384,
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
        **kwargs,
    # 初始化模型配置，设置词汇量大小
    def __init__(
        self,
        vocab_size,
        # 最大位置编码长度
        max_position_embeddings,
        # 隐藏层大小
        hidden_size,
        # 中间层大小
        intermediate_size,
        # 隐藏层层数
        num_hidden_layers,
        # 注意力头的数量
        num_attention_heads,
        # 隐藏层激活函数
        hidden_act,
        # 初始化范围
        initializer_range,
        # 层归一化 epsilon
        layer_norm_eps,
        # 是否使用缓存
        use_cache,
        # ROPE（旋转位置编码）角度
        rope_theta,
        # ROPE 缩放
        rope_scaling,
        # 是否对 QK（查询键）进行层归一化
        qk_layernorm,
        # 隐藏层 dropout
        hidden_dropout,
        # 注意力 dropout
        attention_dropout,
        # 部分旋转因子
        partial_rotary_factor,
    ):
        # ROPE 缩放验证
        self.rope_scaling_validation()
        # 调用父类初始化方法，设置特殊标记的 ID 和其他参数
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    # 从 transformers.models.llama.configuration_llama.LlamaConfig._rope_scaling_validation 复制而来
    # ROPE 缩放验证
    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        # 如果没有设置 ROPE 缩放，则直接返回
        if self.rope_scaling is None:
            return

        # 如果 ROPE 缩放不是字典或者字典长度不为 2，则抛出 ValueError
        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            raise ValueError(
                "`rope_scaling` must be a dictionary with with two fields, `type` and `factor`, "
                f"got {self.rope_scaling}"
            )
        # 获取 ROPE 缩放的类型和因子
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        # 如果 ROPE 缩放类型为 None 或者不是 ['linear', 'dynamic'] 中的一种，则抛出 ValueError
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
            raise ValueError(
                f"`rope_scaling`'s type field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        # 如果 ROPE 缩放因子为 None 或者不是大于 1 的浮点数，则抛出 ValueError
        if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor <= 1.0:
            raise ValueError(f"`rope_scaling`'s factor field must be a float > 1, got {rope_scaling_factor}")
```  
```