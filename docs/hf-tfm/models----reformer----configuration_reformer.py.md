# `.\transformers\models\reformer\configuration_reformer.py`

```
# 指定编码为 utf-8
# 版权声明，版权归 Trax Authors 和 HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 版本授权
# 只有在遵守许可证的情况下才能使用此文件
# 可以在以下链接获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则本软件是基于 "AS IS" 基础分发的，不附带任何明示或暗示的担保或条件
# 有关详细信息，请参阅许可证
"""Reformer 模型配置"""

# 从配置工具中导入预训练配置类
from ...configuration_utils import PretrainedConfig
# 从工具包中导入日志记录功能
from ...utils import logging

# 获取 logger 对象，用于记录日志
logger = logging.get_logger(__name__)

# 预训练配置文件的映射，映射了模型名称到预训练配置文件的 URL
REFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    # Reformer 模型 "google/reformer-crime-and-punishment" 的预训练配置文件的 URL
    "google/reformer-crime-and-punishment": (
        "https://huggingface.co/google/reformer-crime-and-punishment/resolve/main/config.json"
    ),
    # Reformer 模型 "google/reformer-enwik8" 的预训练配置文件的 URL
    "google/reformer-enwik8": "https://huggingface.co/google/reformer-enwik8/resolve/main/config.json",
}


# Reformer 配置类，继承自预训练配置类
class ReformerConfig(PretrainedConfig):
    r"""
    这是一个用于存储 [`ReformerModel`] 配置的配置类。它用于根据指定的参数实例化一个 Reformer 模型，定义了模型的体系结构。
    使用默认参数实例化一个配置对象将产生类似于 ReFormer [google/reformer-crime-and-punishment](https://huggingface.co/google/reformer-crime-and-punishment)
    架构的配置。

    配置对象继承自 [`PretrainedConfig`]，可用于控制模型的输出。阅读 [`PretrainedConfig`] 的文档以获取更多信息。

    示例:

    ```python
    >>> from transformers import ReformerConfig, ReformerModel

    >>> # 初始化一个 Reformer 配置
    >>> configuration = ReformerConfig()

    >>> # 初始化一个 Reformer 模型（带有随机权重）
    >>> model = ReformerModel(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```
    """

    # 模型类型为 "reformer"
    model_type = "reformer"
    # 推理时忽略的键列表
    keys_to_ignore_at_inference = ["past_buckets_states"]
    # 属性映射为空字典
    attribute_map = {}
    # 初始化方法，设置模型参数
    def __init__(
        # 设置注意力头大小，默认64
        self,
        attention_head_size=64,
        # 设置注意力层的类型序列，默认为["local", "lsh", "local", "lsh", "local", "lsh"]
        attn_layers=["local", "lsh", "local", "lsh", "local", "lsh"],
        # 设置轴向规范化的标准差，默认为1.0
        axial_norm_std=1.0,
        # 是否使用轴向位置嵌入，默认为True
        axial_pos_embds=True,
        # 设置轴向位置嵌入的形状，默认为[64, 64]
        axial_pos_shape=[64, 64],
        # 设置轴向位置嵌入的维度，默认为[64, 192]
        axial_pos_embds_dim=[64, 192],
        # 语言模型头的块大小，默认为0
        chunk_size_lm_head=0,
        # 结束标记的标识，默认为2
        eos_token_id=2,
        # 前馈网络的大小，默认为512
        feed_forward_size=512,
        # 哈希种子，默认为None
        hash_seed=None,
        # 隐藏层激活函数，默认为"relu"
        hidden_act="relu",
        # 隐藏层的Dropout概率，默认为0.05
        hidden_dropout_prob=0.05,
        # 隐藏层大小，默认为256
        hidden_size=256,
        # 初始化范围，默认为0.02
        initializer_range=0.02,
        # 是否为解码器，默认为False
        is_decoder=False,
        # 层归一化的epsilon值，默认为1e-12
        layer_norm_eps=1e-12,
        # local注意力之前的块数，默认为1
        local_num_chunks_before=1,
        # local注意力之后的块数，默认为0
        local_num_chunks_after=0,
        # local注意力概率的Dropout概率，默认为0.05
        local_attention_probs_dropout_prob=0.05,
        # local注意力的块长度，默认为64
        local_attn_chunk_length=64,
        # lsh注意力的块长度，默认为64
        lsh_attn_chunk_length=64,
        # lsh注意力概率的Dropout概率，默认为0.0
        lsh_attention_probs_dropout_prob=0.0,
        # lsh注意力之前的块数，默认为1
        lsh_num_chunks_before=1,
        # lsh注意力之后的块数，默认为0
        lsh_num_chunks_after=0,
        # 最大位置嵌入，默认为4096
        max_position_embeddings=4096,
        # 注意力头的数量，默认为12
        num_attention_heads=12,
        # 桶的数量，默认为None
        num_buckets=None,
        # 哈希数量，默认为1
        num_hashes=1,
        # 填充标记的标识，默认为0
        pad_token_id=0,
        # 词汇表大小，默认为320
        vocab_size=320,
        # 是否绑定词嵌入，默认为False
        tie_word_embeddings=False,
        # 是否使用缓存，默认为True
        use_cache=True,
        # 分类器的Dropout概率���默认为None
        classifier_dropout=None,
        **kwargs,
    ):
        # 设置哈希种子
        self.hash_seed = hash_seed
        # 设置词汇表大小
        self.vocab_size = vocab_size
        # 设置注意力头大小
        self.attention_head_size = attention_head_size
        # 设置隐藏层大小
        self.hidden_size = hidden_size
        # 设置注意力头的数量
        self.num_attention_heads = num_attention_heads
        # 设置哈希数量
        self.num_hashes = num_hashes
        # 设置隐藏层的层数
        self.num_hidden_layers = len(attn_layers)
        # 如果num_buckets是列表，则转换为元组，否则保持原值
        self.num_buckets = tuple(num_buckets) if isinstance(num_buckets, list) else num_buckets
        # 设置lsh注意力的块长度
        self.lsh_attn_chunk_length = lsh_attn_chunk_length
        # 设置local注意力的块长度
        self.local_attn_chunk_length = local_attn_chunk_length
        # 设置lsh注意力之后的块数
        self.lsh_num_chunks_after = lsh_num_chunks_after
        # 设置lsh注意力之前的块数
        self.lsh_num_chunks_before = lsh_num_chunks_before
        # 设置local注意力之后的块数
        self.local_num_chunks_after = local_num_chunks_after
        # 设置local注意力之前的块数
        self.local_num_chunks_before = local_num_chunks_before
        # 设置隐藏层的激活函数
        self.hidden_act = hidden_act
        # 设置前馈网络的大小
        self.feed_forward_size = feed_forward_size
        # 设置隐藏层的Dropout概率
        self.hidden_dropout_prob = hidden_dropout_prob
        # 设置lsh注意力概率的Dropout概率
        self.lsh_attention_probs_dropout_prob = lsh_attention_probs_dropout_prob
        # 设置local注意力概率的Dropout概率
        self.local_attention_probs_dropout_prob = local_attention_probs_dropout_prob
        # 设置最大位置嵌入
        self.max_position_embeddings = max_position_embeddings
        # 设置初始化范围
        self.initializer_range = initializer_range
        # 设置层归一化的epsilon值
        self.layer_norm_eps = layer_norm_eps
        # 设置是否使用轴向位置嵌入
        self.axial_pos_embds = axial_pos_embds
        # 设置轴向位置嵌入的形状
        self.axial_pos_shape = tuple(axial_pos_shape)
        # 设置轴向位置嵌入的维度
        self.axial_pos_embds_dim = tuple(axial_pos_embds_dim)
        # 设置轴向规范化的标准差
        self.axial_norm_std = axial_norm_std
        # 设置语言模型头的块大小
        self.chunk_size_lm_head = chunk_size_lm_head
        # 设置注意力层的类型序列
        self.attn_layers = attn_layers
        # 设置是否使用缓存
        self.use_cache = use_cache
        # 设置分类器的Dropout概率
        self.classifier_dropout = classifier_dropout
        # 调用父类的初始化方法，传入参数
        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            is_decoder=is_decoder,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
```