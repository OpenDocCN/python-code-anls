# `.\transformers\models\mega\configuration_mega.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，版权归 The Mega Authors 和 The HuggingFace Inc. 团队所有
# 根据 Apache 许可证 2.0 版本，除非符合许可证，否则不得使用此文件
# 可以在以下网址获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“按原样”分发的，没有任何明示或暗示的担保或条件
# 请查看许可证以获取有关特定语言的权限和限制
""" MEGA configuration"""
# 导入所需的模块
from collections import OrderedDict
from typing import Mapping

# 导入预训练配置类
from ...configuration_utils import PretrainedConfig
# 导入 ONNX 配置类
from ...onnx import OnnxConfig
# 导入日志记录工具
from ...utils import logging

# 获取日志记录器
logger = logging.get_logger(__name__)

# 预训练模型配置文件映射
MEGA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "mnaylor/mega-base-wikitext": "https://huggingface.co/mnaylor/mega-base-wikitext/resolve/main/config.json",
}

# Mega 配置类，继承自预训练配置类
class MegaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`MegaModel`]. It is used to instantiate a Mega
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the Mega
    [mnaylor/mega-base-wikitext](https://huggingface.co/mnaylor/mega-base-wikitext) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Examples:

    ```python
    >>> from transformers import MegaConfig, MegaModel

    >>> # Initializing a Mega configuration
    >>> configuration = MegaConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = MegaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```py"""

    # 模型类型为 "mega"
    model_type = "mega"
    # 初始化模型参数
    def __init__(
        self,
        vocab_size=30522,  # 词汇表大小，默认为30522
        hidden_size=128,  # 隐藏层大小，默认为128
        num_hidden_layers=4,  # 隐藏层层数，默认为4
        intermediate_size=256,  # 中间层大小，默认为256
        ema_projection_size=16,  # EMA投影大小，默认为16
        bidirectional=True,  # 是否双向，默认为True
        shared_representation_size=64,  # 共享表示大小，默认为64
        use_chunking=False,  # 是否使用分块，默认为False
        chunk_size=-1,  # 分块大小，默认为-1
        truncation=None,  # 截断方式，默认为None
        normalize_before_mega=True,  # 是否在MEGA之前进行归一化，默认为True
        normalization_type="scalenorm",  # 归一化类型，默认为scalenorm
        norm_affine=True,  # 归一化是否可调，默认为True
        activation="silu",  # 激活函数，默认为silu
        attention_activation="softmax",  # 注意力激活函数，默认为softmax
        dropout_prob=0.1,  # 一般dropout概率，默认为0.1
        hidden_dropout_prob=0.1,  # 隐藏层dropout概率，默认为0.1
        attention_probs_dropout_prob=0.1,  # 注意力概率dropout概率，默认为0.1
        use_feature_dropout=False,  # 是否使用特征dropout，默认为False
        use_normalized_ffn=True,  # 是否使用归一化FFN，默认为True
        nffn_hidden_size=256,  # 非线性FFN隐藏层大小，默认为256
        normalize_before_ffn=True,  # 是否在FFN之前进行归一化，默认为True
        nffn_activation_dropout_prob=0.1,  # 非线性FFN激活dropout概率，默认为0.1
        max_positions=2048,  # 最大位置编码，默认为2048
        add_token_type_embeddings=False,  # 是否添加token类型嵌入，默认为False
        type_vocab_size=2,  # token类型词汇表大小，默认为2
        initializer_range=0.02,  # 初始化范围，默认为0.02
        ema_delta_alpha_range=0.2,  # EMA delta alpha范围，默认为0.2
        ema_beta_range=0.02,  # EMA beta范围，默认为0.02
        ema_gamma_omega_range=1.0,  # EMA gamma omega范围，默认为1.0
        pad_token_id=1,  # 填充token id，默认为1
        bos_token_id=0,  # 开始token id，默认为0
        eos_token_id=2,  # 结束token id，默认为2
        relative_positional_bias="rotary",  # 相对位置偏置，默认为rotary
        classifier_dropout=None,  # 分类器dropout，默认为None
        use_cache=True,  # 是否使用缓存，默认为True
        add_lm_hidden_dense_layer=True,  # 是否添加LM隐藏层密集层，默认为True
        **kwargs,  # 其他参数
        # 调用父类的构造函数，初始化模型参数
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        # 初始化模型的词汇表大小、隐藏层大小、隐藏层数、激活函数等参数
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.activation = activation
        self.attention_activation = attention_activation
        self.intermediate_size = intermediate_size
        self.ema_projection_size = ema_projection_size
        self.bidirectional = bidirectional
        self.shared_representation_size = shared_representation_size
        self.use_chunking = use_chunking
        self.chunk_size = chunk_size
        self.truncation = truncation
        self.normalize_before_mega = normalize_before_mega
        self.normalization_type = normalization_type
        self.norm_affine = norm_affine
        self.dropout_prob = dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.use_feature_dropout = use_feature_dropout
        self.use_normalized_ffn = use_normalized_ffn
        self.nffn_hidden_size = nffn_hidden_size
        self.normalize_before_ffn = normalize_before_ffn
        self.nffn_activation_dropout_prob = nffn_activation_dropout_prob
        self.max_positions = max_positions
        self.add_token_type_embeddings = add_token_type_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.ema_delta_alpha_range = ema_delta_alpha_range
        self.ema_beta_range = ema_beta_range
        self.ema_gamma_omega_range = ema_gamma_omega_range
        self.relative_positional_bias = relative_positional_bias
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout
        self.add_lm_hidden_dense_layer = add_lm_hidden_dense_layer
        # 设置注意力头数为1，虽然未使用但是Hugging Face要求必须设置
        self.num_attention_heads = 1
# 定义一个名为 MegaOnnxConfig 的类，继承自 OnnxConfig 类
class MegaOnnxConfig(OnnxConfig):
    # 定义一个名为 inputs 的属性，返回一个映射类型，键为字符串，值为映射类型，值为整数到字符串的映射
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 如果任务是多项选择
        if self.task == "multiple-choice":
            # 定义一个动态轴字典，键为整数，值为字符串，包括 batch、choice、sequence
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        else:
            # 定义一个动态轴字典，键为整数，值为字符串，包括 batch、sequence
            dynamic_axis = {0: "batch", 1: "sequence"}
        # 返回一个有序字典，包含两个键值对，键为字符串，值为动态轴字典
        return OrderedDict(
            [
                ("input_ids", dynamic_axis),
                ("attention_mask", dynamic_axis),
            ]
        )
```