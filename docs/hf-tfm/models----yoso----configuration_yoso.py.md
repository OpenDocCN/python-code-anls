# `.\transformers\models\yoso\configuration_yoso.py`

```py
# coding=utf-8
# 代码编码格式为 UTF-8
# 版权信息
# 版权属于 2022 年的 HuggingFace 公司团队，保留所有权利
# 根据 Apache 2.0 许可证授权
# 除非符合许可证要求或经书面同意，否则不得使用此文件
# 您可以获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 根据适用法律或协议约定，基于“原样”方式分发软件
# 不提供任何形式的担保或条件，无论是明示的还是暗示的
# 有关特定语言的权限和限制，请参阅许可证

""" YOSO model configuration"""

from ...configuration_utils import PretrainedConfig  # 导入 PretrainedConfig 类
from ...utils import logging  # 导入 logging 工具

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器

YOSO_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "uw-madison/yoso-4096": "https://huggingface.co/uw-madison/yoso-4096/resolve/main/config.json",
    # 查看所有 YOSO 模型 https://huggingface.co/models?filter=yoso
}


class YosoConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`YosoModel`]. It is used to instantiate an YOSO
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the YOSO
    [uw-madison/yoso-4096](https://huggingface.co/uw-madison/yoso-4096) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Example:

    ```python
    >>> from transformers import YosoConfig, YosoModel

    >>> # Initializing a YOSO uw-madison/yoso-4096 style configuration
    >>> configuration = YosoConfig()

    >>> # Initializing a model (with random weights) from the uw-madison/yoso-4096 style configuration
    >>> model = YosoModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```py"""

    model_type = "yoso"  # 模型类型为 yoso

    def __init__(
        self,
        vocab_size=50265,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=4096,
        type_vocab_size=1,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        position_embedding_type="absolute",
        use_expectation=True,
        hash_code_len=9,
        num_hash=64,
        conv_window=None,
        use_fast_hash=True,
        lsh_backward=True,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        **kwargs,
    # 调用父类的初始化方法，传入特殊符号的 ID 和其他关键参数
    def __init__(
        pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs
    ):
        # 设置词汇表大小
        self.vocab_size = vocab_size
        # 设置最大位置嵌入长度
        self.max_position_embeddings = max_position_embeddings
        # 设置隐藏层大小
        self.hidden_size = hidden_size
        # 设置隐藏层的层数
        self.num_hidden_layers = num_hidden_layers
        # 设置注意力头数
        self.num_attention_heads = num_attention_heads
        # 设置中间层大小
        self.intermediate_size = intermediate_size
        # 设置隐藏层的激活函数
        self.hidden_act = hidden_act
        # 设置隐藏层的 dropout 概率
        self.hidden_dropout_prob = hidden_dropout_prob
        # 设置注意力概率的 dropout 概率
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        # 设置初始化范围
        self.initializer_range = initializer_range
        # 设置类型词汇表大小
        self.type_vocab_size = type_vocab_size
        # 设置层标准化 epsilon
        self.layer_norm_eps = layer_norm_eps
        # 设置位置嵌入类型
        self.position_embedding_type = position_embedding_type
        # 设置是否使用期望
        self.use_expectation = use_expectation
        # 设置哈希码长度
        self.hash_code_len = hash_code_len
        # 设置哈希函数数目
        self.num_hash = num_hash
        # 设置卷积窗口大小
        self.conv_window = conv_window
        # 设置是否使用快速哈希
        self.use_fast_hash = use_fast_hash
        # 设置 LSH 是否向后传播
        self.lsh_backward = lsh_backward
```