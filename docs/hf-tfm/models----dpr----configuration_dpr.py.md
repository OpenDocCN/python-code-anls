# `.\models\dpr\configuration_dpr.py`

```py
# 设置文件编码为UTF-8
# 版权所有 2010 年，DPR 作者，The Hugging Face Team.
#
# 根据 Apache 许可证 2.0 版本授权，除非符合许可证的要求，否则不得使用此文件。
# 您可以在以下网址获取许可证的副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按"原样"分发的软件，
# 没有任何明示或暗示的担保或条件。
# 请查看许可证以获取详细的权利和限制信息。
""" DPR 模型配置"""

from ...configuration_utils import PretrainedConfig  # 导入预训练模型的配置类
from ...utils import logging  # 导入日志工具


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

# 定义 DPR 预训练模型与配置文件的映射关系
DPR_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "facebook/dpr-ctx_encoder-single-nq-base": (
        "https://huggingface.co/facebook/dpr-ctx_encoder-single-nq-base/resolve/main/config.json"
    ),
    "facebook/dpr-question_encoder-single-nq-base": (
        "https://huggingface.co/facebook/dpr-question_encoder-single-nq-base/resolve/main/config.json"
    ),
    "facebook/dpr-reader-single-nq-base": (
        "https://huggingface.co/facebook/dpr-reader-single-nq-base/resolve/main/config.json"
    ),
    "facebook/dpr-ctx_encoder-multiset-base": (
        "https://huggingface.co/facebook/dpr-ctx_encoder-multiset-base/resolve/main/config.json"
    ),
    "facebook/dpr-question_encoder-multiset-base": (
        "https://huggingface.co/facebook/dpr-question_encoder-multiset-base/resolve/main/config.json"
    ),
    "facebook/dpr-reader-multiset-base": (
        "https://huggingface.co/facebook/dpr-reader-multiset-base/resolve/main/config.json"
    ),
}


class DPRConfig(PretrainedConfig):
    r"""
    [`DPRConfig`] 是用于存储 *DPRModel* 配置的配置类。

    这是用于存储 [`DPRContextEncoder`], [`DPRQuestionEncoder`] 或 [`DPRReader`] 的配置类。根据指定的参数实例化 DPR 模型组件，
    定义模型组件的架构。使用默认值实例化配置将产生类似于 DPRContextEncoder
    [facebook/dpr-ctx_encoder-single-nq-base](https://huggingface.co/facebook/dpr-ctx_encoder-single-nq-base)
    架构的配置。

    该类是 [`BertConfig`] 的子类。请查看超类以获取所有 kwargs 的文档。

    示例:

    ```
    >>> from transformers import DPRConfig, DPRContextEncoder

    >>> # 初始化 DPR facebook/dpr-ctx_encoder-single-nq-base 风格的配置
    >>> configuration = DPRConfig()

    >>> # 使用配置初始化一个模型（随机权重）来自 facebook/dpr-ctx_encoder-single-nq-base 风格的配置
    >>> model = DPRContextEncoder(configuration)

    >>> # 访问模型配置
    >>> configuration = model.config
    ```
    # 设定模型类型为 "dpr"
    model_type = "dpr"
    
    # 定义一个初始化方法，用于初始化模型参数和配置
    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        projection_dim: int = 0,
        **kwargs,
    ):
        # 调用父类的初始化方法，传入 pad_token_id 和其他关键字参数
        super().__init__(pad_token_id=pad_token_id, **kwargs)
    
        # 设置模型的各种参数
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.projection_dim = projection_dim
        self.position_embedding_type = position_embedding_type
```