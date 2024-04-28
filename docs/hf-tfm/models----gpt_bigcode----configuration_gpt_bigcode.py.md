# `.\models\gpt_bigcode\configuration_gpt_bigcode.py`

```
# 设置文件编码
# 版权声明
# 许可证信息
# 复制或使用此文件需符合许可证内容
# 可以在上面链接处找到许可证的副本
# 在适用法律要求或书面同意的情况下，
# 根据许可证分发的软件是基于“原样”基础分发的，
# 没有任何担保或条件，无论是明示的还是默示的。
# 请查看许可证以获取有关权限和限制的特定语言
""" GPTBigCode 配置"""

# 导入预训练配置和日志工具
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取一个日志记录器
logger = logging.get_logger(__name__)

# 预训练配置文件映射
GPT_BIGCODE_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "bigcode/gpt_bigcode-santacoder": "https://huggingface.co/bigcode/gpt_bigcode-santacoder/resolve/main/config.json",
}

# GPTBigCode 配置类
class GPTBigCodeConfig(PretrainedConfig):
    """
    这是用于存储[`GPTBigCodeModel`]的配置类。用于根据指定的参数实例化GPTBigCode模型，定义模型架构。
    使用默认实例化一个配置将产生类似于GPTBigCode[gpt_bigcode](https://huggingface.co/gpt_bigcode)架构的配置。

    配置对象继承自[`PretrainedConfig`]，可用于控制模型输出。阅读[`PretrainedConfig`]文档以获取更多信息。
    """
    # 定义 GPT-2 模型的配置类
    Args:
        # 词汇表大小，默认为 50257，定义了在调用 GPTBigCodeModel 时 `inputs_ids` 中可以表示的不同标记数量
        vocab_size (`int`, *optional*, defaults to 50257):
        # 该模型可能会被使用的最大序列长度，默认为 1024，通常设置为一个较大的值，例如 512、1024 或 2048
        n_positions (`int`, *optional*, defaults to 1024):
        # 嵌入层和隐藏状态的维度，默认为 768
        n_embd (`int`, *optional*, defaults to 768):
        # Transformer 编码器中隐藏层的数量，默认为 12
        n_layer (`int`, *optional*, defaults to 12):
        # Transformer 编码器中每个注意力层的注意力头数，默认为 12
        n_head (`int`, *optional*, defaults to 12):
        # 内部全连接层的维度，默认为 4 * n_embd，若为 None，则设为 4 倍 n_embd
        n_inner (`int`, *optional*, defaults to None):
        # 激活函数，默认为 "gelu_pytorch_tanh"，可选值为 ["relu", "silu", "gelu", "tanh", "gelu_new", "gelu_pytorch_tanh"]
        activation_function (`str`, *optional*, defaults to `"gelu_pytorch_tanh"`):
        # 嵌入层、编码器和池化器中所有全连接层的 dropout 概率，默认为 0.1
        resid_pdrop (`float`, *optional*, defaults to 0.1):
        # 嵌入层的 dropout 比率，默认为 0.1
        embd_pdrop (`float`, *optional*, defaults to 0.1):
        # 注意力的 dropout 比率，默认为 0.1
        attn_pdrop (`float`, *optional*, defaults to 0.1):
        # 层标准化层中使用的 epsilon，默认为 1e-5
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-5):
        # 用于初始化所有权重矩阵的截断正态初始化器的标准差，默认为 0.02
        initializer_range (`float`, *optional*, defaults to 0.02):
        # 是否通过除以 sqrt(hidden_size) 来缩放注意力权重，默认为 True
        scale_attn_weights (`bool`, *optional*, defaults to `True`):
        # 模型是否应返回最后一组键/值注意力，默认为 True（不是所有模型都使用）
        use_cache (`bool`, *optional*, defaults to `True`):
        # 是否在 float32 中调用融合 softmax，默认为 True
        attention_softmax_in_fp32 (`bool`, *optional*, defaults to `True`):
        # 是否在 float32 中缩放注意力 softmax，默认为 True
        scale_attention_softmax_in_fp32 (`bool`, *optional*, defaults to `True`):
        # 是否使用 Multi-Query Attion（True）还是 Multi-Head Attention（False），默认为 True
        attention_type (`bool`, *optional*, defaults to `True`):
    Example:

    ```python
    # 从 transformers 中导入 GPTBigCodeConfig 和 GPTBigCodeModel
    >>> from transformers import GPTBigCodeConfig, GPTBigCodeModel

    # 初始化一个 GPTBigCode 配置
    >>> configuration = GPTBigCodeConfig()

    # 根据配置初始化一个模型（带有随机权重）
    >>> model = GPTBigCodeModel(configuration)
    ```
    # 访问模型配置
    configuration = model.config
    
    
    
    # 定义模型类型为 "gpt_bigcode"
    model_type = "gpt_bigcode"
    # 在推断时需要忽略的键列表
    keys_to_ignore_at_inference = ["past_key_values"]
    # 属性映射字典，用于将配置属性名映射到模型属性名
    attribute_map = {
        "hidden_size": "n_embd",
        "max_position_embeddings": "n_positions",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }
    
    
    
    # 初始化方法，设置模型的各项属性
    def __init__(
        self,
        vocab_size=50257,
        n_positions=1024,
        n_embd=768,
        n_layer=12,
        n_head=12,
        n_inner=None,
        activation_function="gelu_pytorch_tanh",
        resid_pdrop=0.1,
        embd_pdrop=0.1,
        attn_pdrop=0.1,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        scale_attn_weights=True,
        use_cache=True,
        bos_token_id=50256,
        eos_token_id=50256,
        attention_softmax_in_fp32=True,
        scale_attention_softmax_in_fp32=True,
        multi_query=True,
        **kwargs,
    ):
        # 设置模型的各项属性
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_inner = n_inner
        self.activation_function = activation_function
        self.resid_pdrop = resid_pdrop
        self.embd_pdrop = embd_pdrop
        self.attn_pdrop = attn_pdrop
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.scale_attn_weights = scale_attn_weights
        self.use_cache = use_cache
        self.attention_softmax_in_fp32 = attention_softmax_in_fp32
        self.scale_attention_softmax_in_fp32 = scale_attention_softmax_in_fp32
        self.multi_query = multi_query
    
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
    
        # 调用父类的初始化方法
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
```