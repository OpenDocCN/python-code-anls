# `.\models\reformer\configuration_reformer.py`

```
# 设置文件编码为UTF-8，确保可以正确处理中文等特殊字符
# 版权声明和许可信息，指定代码的使用权限和限制条件
# 引入预训练配置模块和日志记录工具
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取当前模块的日志记录器
logger = logging.get_logger(__name__)

# 定义预训练模型配置文件的下载链接映射
REFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "google/reformer-crime-and-punishment": (
        "https://huggingface.co/google/reformer-crime-and-punishment/resolve/main/config.json"
    ),
    "google/reformer-enwik8": "https://huggingface.co/google/reformer-enwik8/resolve/main/config.json",
}

# 定义ReformerConfig类，继承自PretrainedConfig类
class ReformerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`ReformerModel`]. It is used to instantiate a
    Reformer model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the ReFormer
    [google/reformer-crime-and-punishment](https://huggingface.co/google/reformer-crime-and-punishment) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Examples:

    ```python
    >>> from transformers import ReformerConfig, ReformerModel

    >>> # Initializing a Reformer configuration
    >>> configuration = ReformerConfig()

    >>> # Initializing a Reformer model (with random weights)
    >>> model = ReformerModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
"""

    # 模型类型为reformer，用于标识模型种类
    model_type = "reformer"
    # 在推理过程中忽略的特定键，用于控制模型输出
    keys_to_ignore_at_inference = ["past_buckets_states"]
    # 属性映射，暂未定义任何属性
    attribute_map = {}
    def __init__(
        self,
        attention_head_size=64,
        attn_layers=["local", "lsh", "local", "lsh", "local", "lsh"],
        axial_norm_std=1.0,
        axial_pos_embds=True,
        axial_pos_shape=[64, 64],
        axial_pos_embds_dim=[64, 192],
        chunk_size_lm_head=0,
        eos_token_id=2,
        feed_forward_size=512,
        hash_seed=None,
        hidden_act="relu",
        hidden_dropout_prob=0.05,
        hidden_size=256,
        initializer_range=0.02,
        is_decoder=False,
        layer_norm_eps=1e-12,
        local_num_chunks_before=1,
        local_num_chunks_after=0,
        local_attention_probs_dropout_prob=0.05,
        local_attn_chunk_length=64,
        lsh_attn_chunk_length=64,
        lsh_attention_probs_dropout_prob=0.0,
        lsh_num_chunks_before=1,
        lsh_num_chunks_after=0,
        max_position_embeddings=4096,
        num_attention_heads=12,
        num_buckets=None,
        num_hashes=1,
        pad_token_id=0,
        vocab_size=320,
        tie_word_embeddings=False,
        use_cache=True,
        classifier_dropout=None,
        **kwargs,
    ):
        # 设置对象的哈希种子
        self.hash_seed = hash_seed
        # 设置对象的词汇表大小
        self.vocab_size = vocab_size
        # 设置对象的注意力头大小
        self.attention_head_size = attention_head_size
        # 设置对象的隐藏层大小
        self.hidden_size = hidden_size
        # 设置对象的注意力头数量
        self.num_attention_heads = num_attention_heads
        # 设置对象的哈希数量
        self.num_hashes = num_hashes
        # 记录对象的注意力层总数
        self.num_hidden_layers = len(attn_layers)
        # 将桶的数量转换为元组形式（如果是列表的话）
        self.num_buckets = tuple(num_buckets) if isinstance(num_buckets, list) else num_buckets
        # 设置LSH注意力的块长度
        self.lsh_attn_chunk_length = lsh_attn_chunk_length
        # 设置局部注意力的块长度
        self.local_attn_chunk_length = local_attn_chunk_length
        # 设置LSH注意力之后的块数
        self.lsh_num_chunks_after = lsh_num_chunks_after
        # 设置LSH注意力之前的块数
        self.lsh_num_chunks_before = lsh_num_chunks_before
        # 设置局部注意力之后的块数
        self.local_num_chunks_after = local_num_chunks_after
        # 设置局部注意力之前的块数
        self.local_num_chunks_before = local_num_chunks_before
        # 设置隐藏层激活函数类型
        self.hidden_act = hidden_act
        # 设置前馈网络的大小
        self.feed_forward_size = feed_forward_size
        # 设置隐藏层的丢弃概率
        self.hidden_dropout_prob = hidden_dropout_prob
        # 设置LSH注意力的注意力概率丢弃概率
        self.lsh_attention_probs_dropout_prob = lsh_attention_probs_dropout_prob
        # 设置局部注意力的注意力概率丢弃概率
        self.local_attention_probs_dropout_prob = local_attention_probs_dropout_prob
        # 设置最大位置嵌入的长度
        self.max_position_embeddings = max_position_embeddings
        # 设置初始化器的范围
        self.initializer_range = initializer_range
        # 设置层归一化的epsilon值
        self.layer_norm_eps = layer_norm_eps
        # 设置是否使用轴向位置嵌入
        self.axial_pos_embds = axial_pos_embds
        # 设置轴向位置嵌入的形状
        self.axial_pos_shape = tuple(axial_pos_shape)
        # 设置轴向位置嵌入的维度
        self.axial_pos_embds_dim = tuple(axial_pos_embds_dim)
        # 设置轴向归一化的标准差
        self.axial_norm_std = axial_norm_std
        # 设置语言模型头部的块大小
        self.chunk_size_lm_head = chunk_size_lm_head
        # 设置注意力层的类型列表
        self.attn_layers = attn_layers
        # 设置是否使用缓存
        self.use_cache = use_cache
        # 设置分类器的丢弃率
        self.classifier_dropout = classifier_dropout
        # 调用父类的初始化方法，传入关键参数
        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            is_decoder=is_decoder,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
```