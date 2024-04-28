# `.\transformers\models\opt\configuration_opt.py`

```
# 导入所需的模块和工具类
import logging
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义 OPT 预训练模型的配置文件 URL
OPT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    # 各种 OPT 模型的预训练配置文件 URL
    "facebook/opt-125m": "https://huggingface.co/facebook/opt-125m/blob/main/config.json",
    "facebook/opt-350m": "https://huggingface.co/facebook/opt-350m/blob/main/config.json",
    "facebook/opt-1.3b": "https://huggingface.co/facebook/opt-1.3b/blob/main/config.json",
    "facebook/opt-2.7b": "https://huggingface.co/facebook/opt-2.7b/blob/main/config.json",
    "facebook/opt-6.7b": "https://huggingface.co/facebook/opt-6.7b/blob/main/config.json",
    "facebook/opt-13b": "https://huggingface.co/facebook/opt-13b/blob/main/config.json",
}

# 定义 OPT 模型的配置类
class OPTConfig(PretrainedConfig):
    r"""
    这是一个配置类,用于存储 [`OPTModel`] 的配置信息.
    它用于根据指定的参数实例化一个 OPT 模型,定义模型的架构.
    使用默认配置实例化一个配置对象,会生成一个类似于 OPT [facebook/opt-350m] 架构的配置.
    配置对象继承自 [`PretrainedConfig`],可用于控制模型的输出.
    请查看 [`PretrainedConfig`] 的文档以了解更多信息.
    """
    # 定义了 OPT 模型的配置类，用于指定模型的各种参数
    Args:
        # 词汇量大小，默认为 50272，表示模型可以表示的不同 token 的数量
        vocab_size (`int`, *optional*, defaults to 50272):
            Vocabulary size of the OPT model. Defines the number of different tokens that can be represented by the
            `inputs_ids` passed when calling [`OPTModel`]
        # 隐藏层大小，默认为 768，表示每层和池化层的维度
        hidden_size (`int`, *optional*, defaults to 768):
            Dimensionality of the layers and the pooler layer.
        # 编码器层数，默认为 12
        num_hidden_layers (`int`, *optional*, defaults to 12):
            Number of decoder layers.
        # 解码器中“中间”（通常称为前馈）层的维度，默认为 3072
        ffn_dim (`int`, *optional*, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in decoder.
        # Transformer 解码器中每个注意力层的注意力头数，默认为 12
        num_attention_heads (`int`, *optional*, defaults to 12):
            Number of attention heads for each attention layer in the Transformer decoder.
        # 激活函数，默认为 "relu"，支持 "gelu"、"relu"、"silu" 和 "gelu_new"
        activation_function (`str` or `function`, *optional*, defaults to `"relu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
            `"relu"`, `"silu"` and `"gelu_new"` are supported.
        # 最大序列长度，默认为 2048，模型可能会用到的最大序列长度
        max_position_embeddings (`int`, *optional*, defaults to 2048):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        # 是否在注意力块之前执行层归一化，默认为 True
        do_layer_norm_before (`bool`, *optional*, defaults to `True`):
            Whether to perform layer normalization before the attention block.
        # 词嵌入投影维度，用于降维词嵌入，默认为 hidden_size
        word_embed_proj_dim (`int`, *optional*):
            `word_embed_proj_dim` can be set to down-project word embeddings, *e.g.* `opt-350m`. Defaults to
            `hidden_size`.
        # 所有全连接层的 dropout 概率，默认为 0.1
        dropout (`float`, *optional*, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        # 注意力概率的 dropout 比例，默认为 0.0
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
        # LayerDrop 概率，默认为 0.0，详细信息请参见 LayerDrop 论文
        layerdrop (`float`, *optional*, defaults to 0.0):
            The LayerDrop probability. See the [LayerDrop paper](see https://arxiv.org/abs/1909.11556) for more
            details.
        # 初始化所有权重矩阵的截断正态分布的标准差，默认为 0.02
        init_std (`float`, *optional*, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        # 模型是否应返回最后一个 key/values 注意力，默认为 True，不是所有模型都使用
        use_cache (`bool`, *optional*, defaults to `True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        # 线性层在注意力块中是否应使用偏置项，默认为 True
        enable_bias (`bool`, *optional*, defaults to `True`):
            Whether or not if the linear layers in the attention blocks should use the bias term.
        # 层标准化是否应具有可学习的参数，默认为 True
        layer_norm_elementwise_affine (`bool`, *optional*, defaults to `True`):
            Whether or not if the layer norms should have learnable parameters.

    # 示例代码，示例中展示了如何初始化一个 OPT facebook/opt-large 风格的配置类
    Example:

    ```python
    >>> from transformers import OPTConfig, OPTModel

    >>> # Initializing a OPT facebook/opt-large style configuration
    # 创建一个 OPT facebook/opt-large 风格的配置类实例
    >>> configuration = OPTConfig()
    # 根据配置初始化一个模型（带有随机权重），该配置是基于 facebook/opt-large 风格的
    model = OPTModel(configuration)

    # 访问模型配置
    configuration = model.config

    model_type = "opt"
    keys_to_ignore_at_inference = ["past_key_values"]

    # 初始化函数
    def __init__(
        # 初始化参数
        self,
        vocab_size=50272,
        hidden_size=768,
        num_hidden_layers=12,
        ffn_dim=3072,
        max_position_embeddings=2048,
        do_layer_norm_before=True,
        _remove_final_layer_norm=False,
        word_embed_proj_dim=None,
        dropout=0.1,
        attention_dropout=0.0,
        num_attention_heads=12,
        activation_function="relu",
        layerdrop=0.0,
        init_std=0.02,
        use_cache=True,
        pad_token_id=1,
        bos_token_id=2,
        eos_token_id=2,
        enable_bias=True,
        layer_norm_elementwise_affine=True,
        **kwargs,
    ):
        # 调用父类的初始化函数
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            **kwargs,
        )
        # 设置模型的各种参数
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.word_embed_proj_dim = word_embed_proj_dim if word_embed_proj_dim is not None else hidden_size
        self.ffn_dim = ffn_dim
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.layerdrop = layerdrop
        self.use_cache = use_cache
        self.do_layer_norm_before = do_layer_norm_before
        # 为了向后兼容，保持这些变量为 `True`
        self.enable_bias = enable_bias
        self.layer_norm_elementwise_affine = layer_norm_elementwise_affine

        # 注意，`_remove_final_layer_norm` 的唯一目的是为了保持向后兼容，
        # 用于在 transformers v4.20.1 之前对检查点进行微调
        # 参见 https://github.com/facebookresearch/metaseq/pull/164
        self._remove_final_layer_norm = _remove_final_layer_norm
```