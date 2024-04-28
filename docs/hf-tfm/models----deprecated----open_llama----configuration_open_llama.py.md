# `.\models\deprecated\open_llama\configuration_open_llama.py`

```py
# 设置编码格式
# 版权声明
# 本代码基于EleutherAI的GPT-NeoX库以及该库中的GPT-NeoX和OPT实现进行修改，以适应与模型训练的Meta AI团队所使用的GPT-NeoX和OPT相比的架构上的轻微差异
# 根据Apache许可证2.0版授权
# 除非适用法律要求或书面同意，否则在遵守许可证的情况下，不得使用此文件
# 您可以在http://www.apache.org/licenses/LICENSE-2.0获取许可证的副本
# 除非适用法律要求或书面同意，否则不对软件进行任何形式的担保或条件分发。
# 请查看许可证以了解具体语言管理权限和限制

# 导入必要的库文件
from ....configuration_utils import PretrainedConfig
from ....utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义Open-Llama预训练配置文件与官方存档的映射关系
OPEN_LLAMA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "s-JoL/Open-Llama-V1": "https://huggingface.co/s-JoL/Open-Llama-V1/blob/main/config.json",
}

# OpenLlama配置类，用于存储OpenLlamaModel的配置，并用于实例化Open-Llama模型以及定义模型架构
class OpenLlamaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`OpenLlamaModel`]. It is used to instantiate an
    Open-Llama model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the
    [s-JoL/Open-Llama-V1](https://huggingface.co/s-JoL/Open-Llama-V1).
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    # 定义一个函数，用于初始化 Open-Llama 模型的配置
    Args:
        # 词汇表大小，默认为 32000，定义了在调用 OpenLlamaModel 时输入的 input_ids 可以表示的不同标记数量
        vocab_size (`int`, *optional*, defaults to 32000):
        # 隐藏表示的维度
        hidden_size (`int`, *optional*, defaults to 4096):
        # MLP 表示的维度
        intermediate_size (`int`, *optional*, defaults to 11008):
        # Transformer 编码器中的隐藏层数量
        num_hidden_layers (`int`, *optional*, defaults to 32):
        # Transformer 编码器中每个注意力层的注意力头数量
        num_attention_heads (`int`, *optional*, defaults to 32):
        # 解码器中的非线性激活函数
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
        # 模型可能使用的最大序列长度
        max_position_embeddings (`int`, *optional*, defaults to 2048):
        # 用于初始化所有权重矩阵的截断正态初始化器的标准差
        initializer_range (`float`, *optional*, defaults to 0.02):
        # rms 法归一化层使用的 epsilon
        rms_norm_eps (`float`, *optional*, defaults to 1e-12):
        # 模型是否应返回最后的键/值注意力
        use_cache (`bool`, *optional*, defaults to `True`):
        # 是否绑定权重词嵌入
        tie_word_embeddings(`bool`, *optional*, defaults to `False`):
        # RoPE 嵌入的缩放配置
        rope_scaling (`Dict`, *optional*):
            # RoPE 嵌入的缩放策略
            Dictionary containing the scaling configuration for the RoPE embeddings. Currently supports two scaling
            strategies: linear and dynamic. Their scaling factor must be a float greater than 1. The expected format is
            `{"type": strategy name, "factor": scaling factor}`. When using this flag, don't update
            `max_position_embeddings` to the expected new maximum. See the following thread for more information on how
            these scaling strategies behave:
            https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/. This is an
            experimental feature, subject to breaking API changes in future versions.
        Example:
        # 从 transformers 库中导入 OpenLlamaModel 和 OpenLlamaConfig
    ```python
    >>> from transformers import OpenLlamaModel, OpenLlamaConfig
    # 初始化一个 Open-Llama open_llama-7b 风格的配置
    >>> configuration = OpenLlamaConfig()
    # 用 open_llama-7b 风格的配置初始化一个模型
    >>> model = OpenLlamaModel(configuration)
    # 访问模型配置
    >>> configuration = model.config
    ```py 
    # 模型类型
    model_type = "open-llama"

    # 初始化函数，设置模型的各种参数
    def __init__(
        self,
        vocab_size=100000,  # 词汇表大小，默认为 100000
        hidden_size=4096,  # 隐藏层大小，默认为 4096
        intermediate_size=11008,  # 中间层大小，默认为 11008
        num_hidden_layers=32,  # 隐藏层层数，默认为 32
        num_attention_heads=32,  # 注意力头数，默认为 32
        hidden_act="silu",  # 隐藏层激活函数，默认为 silu
        max_position_embeddings=2048,  # 最大位置编码数，默认为 2048
        initializer_range=0.02,  # 初始化范围，默认为 0.02
        rms_norm_eps=1e-6,  # RMS 归一化的 epsilon，默认为 1e-6
        use_cache=True,  # 是否使用缓存，默认为 True
        pad_token_id=0,  # 填充 token 的 id，默认为 0
        bos_token_id=1,  # 起始 token 的 id，默认为 1
        eos_token_id=2,  # 结束 token 的 id，默认为 2
        tie_word_embeddings=False,  # 是否绑定词嵌入，默认为 False
        use_memory_efficient_attention=True,  # 是否使用内存高效的注意力机制，默认为 True
        hidden_dropout_prob=0.1,  # 隐藏层 dropout 概率，默认为 0.1
        attention_dropout_prob=0.1,  # 注意力 dropout 概率，默认为 0.1
        use_stable_embedding=True,  # 是否使用稳定的嵌入，默认为 True
        shared_input_output_embedding=True,  # 是否共享输入输出嵌入，默认为 True
        rope_scaling=None,  # ROPE 缩放，默认为 None
        **kwargs,
    ):
        # 初始化各参数
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        # 使用内存高效的注意力机制
        self.use_memory_efficient_attention = kwargs.pop(
            "use_memorry_efficient_attention", use_memory_efficient_attention
        )
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_dropout_prob = attention_dropout_prob
        self.use_stable_embedding = use_stable_embedding
        self.shared_input_output_embedding = shared_input_output_embedding
        self.rope_scaling = rope_scaling
        # 验证 ROPE 缩放
        self._rope_scaling_validation()

        # 调用父类的初始化函数
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    # 从 transformers.models.llama.configuration_llama.LlamaConfig._rope_scaling_validation 复制而来的方法
    def _rope_scaling_validation(self):
        """
        Validate the `rope_scaling` configuration.
        """
        # 检查是否设置了 `rope_scaling`，如果没有设置则直接返回
        if self.rope_scaling is None:
            return

        # 检查 `rope_scaling` 是否为字典且包含两个字段
        if not isinstance(self.rope_scaling, dict) or len(self.rope_scaling) != 2:
            raise ValueError(
                "`rope_scaling` must be a dictionary with with two fields, `type` and `factor`, "
                f"got {self.rope_scaling}"
            )
        
        # 获取 `rope_scaling` 字典中的 `type` 和 `factor` 字段的值
        rope_scaling_type = self.rope_scaling.get("type", None)
        rope_scaling_factor = self.rope_scaling.get("factor", None)
        
        # 检查 `type` 字段的值是否为'linear'或'dynamic'
        if rope_scaling_type is None or rope_scaling_type not in ["linear", "dynamic"]:
            raise ValueError(
                f"`rope_scaling`'s type field must be one of ['linear', 'dynamic'], got {rope_scaling_type}"
            )
        
        # 检查 `factor` 字段的值是否为浮点数且大于1
        if rope_scaling_factor is None or not isinstance(rope_scaling_factor, float) or rope_scaling_factor <= 1.0:
            raise ValueError(f"`rope_scaling`'s factor field must be a float > 1, got {rope_scaling_factor}")
```