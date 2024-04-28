# `.\transformers\models\llama\configuration_llama.py`

```
# 设定文件编码为utf-8
# 版权声明
# 基于EleutherAI的GPT-NeoX库以及该库中的GPT-NeoX和OPT实现的代码
# 经过轻微修改以适应与训练模型时使用的GPT-NeoX和OPT之间的轻微架构差异
# 根据Apache许可证2.0版许可
# 除非适用法律要求或书面同意，在许可证下，本文件不可用
# 您可以在以下网址获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则不得在许可证下分发软件是以"原样"基础分发的
# 无论是明示的还是暗示的，都没有任何保证或条件。
# 请查看特定语言的许可证以了解权限和限制
""" LLaMA 模型配置"""

# 导入必要的依赖
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 声明 LLAMA_PRETRAINED_CONFIG_ARCHIVE_MAP 变量
LLAMA_PRETRAINED_CONFIG_ARCHIVE_MAP = {}

# 创建 LlamaConfig 类，继承自 PretrainedConfig 类
class LlamaConfig(PretrainedConfig):
    r"""
    这是用于存储[`LlamaModel`]配置的配置类。它用于根据指定的参数实例化 LLaMA 模型，定义模型架构。使用默认值实例化配置将产生类似于 LLaMA-7B 的配置。
    
    配置对象继承自[`PretrainedConfig`]，可用于控制模型输出。阅读来自[`PretrainedConfig`]的文档以获取更多信息。
    
    ```python
    >>> from transformers import LlamaModel, LlamaConfig

    >>> # 初始化 LLaMA llama-7b 风格配置
    >>> configuration = LlamaConfig()

    >>> # 从 llama-7b 风格配置初始化一个模型
    >>> model = LlamaModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```"""

    # 模型类型
    model_type = "llama"
    # 推断时要忽略的关键字
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=32000,  # 词汇表大小
        hidden_size=4096,  # 隐藏层大小
        intermediate_size=11008,  # 中间层大小
        num_hidden_layers=32,  # 隐藏层的数量
        num_attention_heads=32,  # 注意力头的数量
        num_key_value_heads=None,  # 键值头的数量
        hidden_act="silu",  # 隐藏层激活函数
        max_position_embeddings=2048,  # 最大位置嵌入
        initializer_range=0.02,  # 初始化范围
        rms_norm_eps=1e-6,  # 标准均方根正则化的 epsilon
        use_cache=True,  # 是否使用缓存
        pad_token_id=None,  # 填充标记 id
        bos_token_id=1,  # 开始标记 id
        eos_token_id=2,  # 结束标记 id
        pretraining_tp=1,  # 预训练类型
        tie_word_embeddings=False,  # 是否绑定词嵌入
        rope_theta=10000.0,  # 绳索 theta
        rope_scaling=None,  # 绳索缩放
        attention_bias=False,  # 注意力偏置
        attention_dropout=0.0,  # 注意力丢弃率
        **kwargs,  # 其他参数
    ):  # 定义一个带有多个参数的初始化函数
        self.vocab_size = vocab_size  # 初始化词汇表大小
        self.max_position_embeddings = max_position_embeddings  # 初始化最大位置嵌入
        self.hidden_size = hidden_size  # 初始化隐藏层大小
        self.intermediate_size = intermediate_size  # 初始化中间层大小
        self.num_hidden_layers = num_hidden_layers  # 初始化隐藏层数
        self.num_attention_heads = num_attention_heads  # 初始化注意力头数

        # 兼容旧版本
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads  # 如果num_key_value_heads为None，则将其赋值为num_attention_heads

        self.num_key_value_heads = num_key_value_heads  # 初始化关键值头数
        self.hidden_act = hidden_act  # 初始化隐藏层激活函数
        self.initializer_range = initializer_range  # 初始化初始化器范围
        self.rms_norm_eps = rms_norm_eps  # 初始化rms_norm_eps
        self.pretraining_tp = pretraining_tp  # 初始化预训练tp
        self.use_cache = use_cache  # 初始化使用缓存
        self.rope_theta = rope_theta  # 初始化绳索theta
        self.rope_scaling = rope_scaling  # 初始化绳索缩放
        self._rope_scaling_validation()  # 调用_rope_scaling_validation方法进行验证
        self.attention_bias = attention_bias  # 初始化注意力偏置
        self.attention_dropout = attention_dropout  # 初始化注意力丢弃

        super().__init__(
            pad_token_id=pad_token_id,  # 初始化pad_token_id
            bos_token_id=bos_token_id,  # 初始化bos_token_id
            eos_token_id=eos_token_id,  # 初始化eos_token_id
            tie_word_embeddings=tie_word_embeddings,  # 初始化tie_word_embeddings
            **kwargs,  # 初始化其他参数
        )

    def _rope_scaling_validation(self):  # 定义_rope_scaling_validation方法
        """
        Validate the `rope_scaling` configuration.
        """
        if self.rope_scaling is None:  # 如果rope_scaling为空，则返回
            return

        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:  # 如果rope_scaling不是字典或长度不为2，则抛出异常
            raise ValueError(
                "`rope_scaling` must be a dictionary with with two fields, `type` and `factor`, "
                f"got {self.rope_scaling}"
            )
        rope_scaling_type = self.rope_scaling.get("type", None)  # 获取rope_scaling中的type字段
        rope_scaling_factor = self.rope_scaling.get("factor", None)  # 获取rope_scaling中的factor字段
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:  # 如果rope_scaling_type为空或不在['linear', 'dynamic']中，则抛出异常
            raise ValueError(
                f"`rope_scaling`'s type field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor <= 1.0:  # 如果rope_scaling_factor为空或不是浮点数或小于等于1.0，则抛出异常
            raise ValueError(f"`rope_scaling`'s factor field must be a float > 1, got {rope_scaling_factor}")
```