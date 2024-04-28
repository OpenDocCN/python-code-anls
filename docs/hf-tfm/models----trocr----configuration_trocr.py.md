# `.\transformers\models\trocr\configuration_trocr.py`

```
# 设置文件编码为 utf-8
# 版权声明，版权归 The HuggingFace Inc. 团队所有
# 根据 Apache 2.0 许可证，除非符合许可证要求，否则不得使用此文件
# 可以获取许可证的副本，访问 http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是根据"原样"分发，无任何形式的担保或条件
# 查看许可证以了解特定语言的权限和限制

""" TrOCR 模型配置"""

# 导入必要的库
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取记录器
logger = logging.get_logger(__name__)

# TrOCR 预训练配置档案映射
TROCR_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/trocr-base-handwritten": (
        "https://huggingface.co/microsoft/trocr-base-handwritten/resolve/main/config.json"
    ),
    # 查看所有 TrOCR 模型，访问 https://huggingface.co/models?filter=trocr
}

# 创建 TrOCR 配置类，用于存储 TrOCRForCausalLM 的配置
class TrOCRConfig(PretrainedConfig):
    r"""
    这是用来存储 [`TrOCRForCausalLM`] 配置的配置类。它用于根据指定参数实例化 TrOCR 模型，定义模型架构。
    使用默认配置实例化会产生类似于 TrOCR [microsoft/trocr-base-handwritten](https://huggingface.co/microsoft/trocr-base-handwritten) 架构的配置。

    配置对象继承自 [`PretrainedConfig`]，可以用于控制模型输出。查看 [`PretrainedConfig`] 的文档获取更多信息。
    """
    # 定义TrOCR模型的配置类，包括模型的各种超参数和配置项
    class TrOCRConfig:
        
        # TrOCR模型的词汇表大小，默认为50265
        vocab_size (`int`, *optional*, defaults to 50265):
            Vocabulary size of the TrOCR model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`TrOCRForCausalLM`].
        
        # 模型的隐藏层大小，默认为1024
        d_model (`int`, *optional*, defaults to 1024):
            Dimensionality of the layers and the pooler layer.
        
        # 解码器的层数，默认为12
        decoder_layers (`int`, *optional*, defaults to 12):
            Number of decoder layers.
        
        # 解码器中每个注意力层的注意力头数，默认为16
        decoder_attention_heads (`int`, *optional*, defaults to 16):
            Number of attention heads for each attention layer in the Transformer decoder.
        
        # 解码器中前馈神经网络的维度，默认为4096
        decoder_ffn_dim (`int`, *optional*, defaults to 4096):
            Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
        
        # 激活函数，默认为"gelu"
        activation_function (`str` or `function`, *optional*, defaults to `"gelu"`):
            The non-linear activation function (function or string) in the pooler. If string, `"gelu"`, `"relu"`,
            `"silu"` and `"gelu_new"` are supported.
        
        # 序列最大长度，默认为512
        max_position_embeddings (`int`, *optional*, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        
        # Dropout概率，默认为0.1，用于全连接层和pooler层
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, and pooler.
        
        # 注意力概率的Dropout概率，默认为0.0
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        
        # 激活函数的Dropout概率，默认为0.0
        activation_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for activations inside the fully connected layer.
        
        # 初始化所有权重矩阵的截断正态分布的标准差，默认为0.02
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        
        # 解码器的LayerDrop概率，默认为0.0。参考LayerDrop论文了解更多细节
        decoder_layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability for the decoder. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556)
            for more details.
        
        # 是否使用Cache，默认为True。���是所有模型都使用此功能
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        
        # 是否对词嵌入进行缩放，默认为False
        scale_embedding (`bool`, *optional*, defaults to `False`):
            Whether or not to scale the word embeddings by sqrt(d_model).
        
        # 是否使用学习得到的位置嵌入，默认为True。如果不使用，将使用正弦位置嵌入
        use_learned_position_embeddings (`bool`, *optional*, defaults to `True`):
            Whether or not to use learned position embeddings. If not, sinusoidal position embeddings will be used.
        
        # 是否在词嵌入和位置嵌入之后使用LayerNorm，默认为True
        layernorm_embedding (`bool`, *optional*, defaults to `True`):
            Whether or not to use a layernorm after the word + position embeddings.
        
        # 初始化一个TrOCR-base样式的配置
        Example:
    
        ```python
        >>> from transformers import TrOCRConfig, TrOCRForCausalLM
    
        >>> # Initializing a TrOCR-base style configuration
        >>> configuration = TrOCRConfig()
    # 从 TrOCR-base 风格的配置初始化一个模型（使用随机权重）
    model = TrOCRForCausalLM(configuration)
    
    # 访问模型配置
    configuration = model.config
    
    
    # 模型类型
    model_type = "trocr"
    # 推断时要忽略的键
    keys_to_ignore_at_inference = ["past_key_values"]
    # 属性映射
    attribute_map = {
        "num_attention_heads": "decoder_attention_heads",
        "hidden_size": "d_model",
        "num_hidden_layers": "decoder_layers",
    }
    
    # 初始化方法
    def __init__(
        self,
        vocab_size=50265,
        d_model=1024,
        decoder_layers=12,
        decoder_attention_heads=16,
        decoder_ffn_dim=4096,
        activation_function="gelu",
        max_position_embeddings=512,
        dropout=0.1,
        attention_dropout=0.0,
        activation_dropout=0.0,
        decoder_start_token_id=2,
        init_std=0.02,
        decoder_layerdrop=0.0,
        use_cache=True,
        scale_embedding=False,
        use_learned_position_embeddings=True,
        layernorm_embedding=True,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        **kwargs,
    ):
        # 设置模型属性
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.activation_function = activation_function
        self.max_position_embeddings = max_position_embeddings
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.init_std = init_std
        self.decoder_layerdrop = decoder_layerdrop
        self.use_cache = use_cache
        self.scale_embedding = scale_embedding
        self.use_learned_position_embeddings = use_learned_position_embeddings
        self.layernorm_embedding = layernorm_embedding
    
        # 调用父类初始化方法
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            decoder_start_token_id=decoder_start_token_id,
            **kwargs,
        )
```