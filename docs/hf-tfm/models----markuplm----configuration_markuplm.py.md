# `.\transformers\models\markuplm\configuration_markuplm.py`

```py
# 设置文件编码为utf-8
# 版权声明：说明代码版权归Microsoft Research Asia MarkupLM Team作者所有
#
# 根据Apache License, Version 2.0许可使用该文件，详细信息请参考许可证链接
#
# 除了依据许可证要求或经书面同意外，禁止使用此文件
# 本文件按"AS IS"方式分发，没有任何明示或暗示的担保或条件
# 请阅读许可证中关于特定语言控制权限与限制的部分
"""MarkupLM模型配置"""

# 导入必要的包和模块
from ...configuration_utils import PretrainedConfig
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 预训练模型的配置文件映射
MARKUPLM_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/markuplm-base": "https://huggingface.co/microsoft/markuplm-base/resolve/main/config.json",
    "microsoft/markuplm-large": "https://huggingface.co/microsoft/markuplm-large/resolve/main/config.json",
}

# 定义配置类，继承自PretrainedConfig
class MarkupLMConfig(PretrainedConfig):
    r"""
    该配置类用于存储[`MarkupLMModel`]的配置。根据指定的参数实例化一个MarkupLM模型，定义模型架构。使用默认值实例化一个配置将产生类似于[microsoft/markuplm-base](https://huggingface.co/microsoft/markuplm-base)架构的配置。

    配置对象继承自[`BertConfig`]，可用于控制模型输出。阅读[`BertConfig`]的文档获取更多信息。

    示例:

    ```python
    >>> from transformers import MarkupLMModel, MarkupLMConfig

    >>> # 初始化一个microsoft/markuplm-base风格的配置
    >>> 配置 = MarkupLMConfig()

    >>> # 从microsoft/markuplm-base风格的配置初始化一个模型
    >>> 模型 = MarkupLMModel(配置)

    >>> # 访问模型配置
    >>> 配置 = 模型.config
    ```py"""
    # 标记模型类型为markuplm
    model_type = "markuplm"

    # 初始化方法，定义模型的各种配置参数
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
        bos_token_id=0,
        eos_token_id=2,
        max_xpath_tag_unit_embeddings=256,
        max_xpath_subs_unit_embeddings=1024,
        tag_pad_id=216,
        subs_pad_id=1001,
        xpath_unit_hidden_size=32,
        max_depth=50,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        **kwargs,
        # 调用父类的构造函数，初始化模型参数
        super().__init__(
            pad_token_id=pad_token_id,  # padding token 的 ID
            bos_token_id=bos_token_id,  # 开始 token 的 ID
            eos_token_id=eos_token_id,  # 结束 token 的 ID
            **kwargs,  # 其他关键字参数
        )
        # 设置词汇表大小
        self.vocab_size = vocab_size
        # 设置隐藏层大小
        self.hidden_size = hidden_size
        # 设置隐藏层的数量
        self.num_hidden_layers = num_hidden_layers
        # 设置注意力头的数量
        self.num_attention_heads = num_attention_heads
        # 设置隐藏层激活函数
        self.hidden_act = hidden_act
        # 设置中间层大小
        self.intermediate_size = intermediate_size
        # 设置隐藏层的 dropout 概率
        self.hidden_dropout_prob = hidden_dropout_prob
        # 设置注意力矩阵的 dropout 概率
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        # 设置最大位置嵌入大小
        self.max_position_embeddings = max_position_embeddings
        # 设置类型词汇表大小
        self.type_vocab_size = type_vocab_size
        # 设置初始化范围
        self.initializer_range = initializer_range
        # 设置层归一化 epsilon 值
        self.layer_norm_eps = layer_norm_eps
        # 设置位置嵌入类型
        self.position_embedding_type = position_embedding_type
        # 设置是否使用缓存
        self.use_cache = use_cache
        # 设置分类器 dropout 概率
        self.classifier_dropout = classifier_dropout
        # 额外属性
        # 设置最大深度
        self.max_depth = max_depth
        # 设置最大 XPath 标签单元嵌入大小
        self.max_xpath_tag_unit_embeddings = max_xpath_tag_unit_embeddings
        # 设置最大 XPath 子单元嵌入大小
        self.max_xpath_subs_unit_embeddings = max_xpath_subs_unit_embeddings
        # 设置标签填充 ID
        self.tag_pad_id = tag_pad_id
        # 设置子单元填充 ID
        self.subs_pad_id = subs_pad_id
        # 设置 XPath 单元隐藏层大小
        self.xpath_unit_hidden_size = xpath_unit_hidden_size
```