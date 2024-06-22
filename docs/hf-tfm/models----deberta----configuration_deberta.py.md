# `.\models\deberta\configuration_deberta.py`

```py
# 设置文件编码为 UTF-8
# 版权声明
# 根据许可证，除非遵守许可证，否则不得使用此文件
# 可以在以下网址获取许可证副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非法律要求或书面同意，否则不得在许可下发布软件
# 分布在"按原样"基础上，没有任何种类的保证或条件，无论是明示的还是暗示的
# 请查看特定语言的许可证，以获取权限和限制
""" DeBERTa 模型配置"""

# 从 collections 模块导入 OrderedDict 类
from collections import OrderedDict
# 从 typing 模块导入类型检查相关的内容
from typing import TYPE_CHECKING, Any, Mapping, Optional, Union
# 从 ... 中导入其他模块
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging

# 如果类型检查开启，再导入下面的模块
if TYPE_CHECKING:
    from ... import FeatureExtractionMixin, PreTrainedTokenizerBase, TensorType

# 获取 logger 实例
logger = logging.get_logger(__name__)

# 设置预训练配置文件地址映射
DEBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/deberta-base": "https://huggingface.co/microsoft/deberta-base/resolve/main/config.json",
    "microsoft/deberta-large": "https://huggingface.co/microsoft/deberta-large/resolve/main/config.json",
    "microsoft/deberta-xlarge": "https://huggingface.co/microsoft/deberta-xlarge/resolve/main/config.json",
    "microsoft/deberta-base-mnli": "https://huggingface.co/microsoft/deberta-base-mnli/resolve/main/config.json",
    "microsoft/deberta-large-mnli": "https://huggingface.co/microsoft/deberta-large-mnli/resolve/main/config.json",
    "microsoft/deberta-xlarge-mnli": "https://huggingface.co/microsoft/deberta-xlarge-mnli/resolve/main/config.json",
}

# 定义 DebertaConfig 类，继承自 PretrainedConfig 类
class DebertaConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DebertaModel`] or a [`TFDebertaModel`]. It is
    used to instantiate a DeBERTa model according to the specified arguments, defining the model architecture.
    Instantiating a configuration with the defaults will yield a similar configuration to that of the DeBERTa
    [microsoft/deberta-base](https://huggingface.co/microsoft/deberta-base) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Example:

    ```python
    >>> from transformers import DebertaConfig, DebertaModel

    >>> # Initializing a DeBERTa microsoft/deberta-base style configuration
    >>> configuration = DebertaConfig()

    >>> # Initializing a model (with random weights) from the microsoft/deberta-base style configuration
    >>> model = DebertaModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```py"""

    # 模型类型为 "deberta"
    model_type = "deberta"
    # 定义一个初始化方法，用于初始化 Transformer 模型的参数
    def __init__(
        self,
        vocab_size=50265,  # 词汇表大小，默认为 50265
        hidden_size=768,  # 隐藏层大小，默认为 768
        num_hidden_layers=12,  # 隐藏层层数，默认为 12
        num_attention_heads=12,  # 注意力头数，默认为 12
        intermediate_size=3072,  # 中间层大小，默认为 3072
        hidden_act="gelu",  # 隐藏层激活函数，默认为 gelu
        hidden_dropout_prob=0.1,  # 隐藏层丢弃率，默认为 0.1
        attention_probs_dropout_prob=0.1,  # 注意力矩阵丢弃率，默认为 0.1
        max_position_embeddings=512,  # 最大位置嵌入数量，默认为 512
        type_vocab_size=0,  # 类型词汇表大小，默认为 0
        initializer_range=0.02,  # 初始化范围，默认为 0.02
        layer_norm_eps=1e-7,  # 层归一化 epsilon，默认为 1e-7
        relative_attention=False,  # 是否使用相对注意力，默认为 False
        max_relative_positions=-1,  # 最大相对位置，默认为 -1
        pad_token_id=0,  # 填充标记的 id，默认为 0
        position_biased_input=True,  # 是否使用位置偏置输入，默认为 True
        pos_att_type=None,  # 位置注意力类型，默认为 None
        pooler_dropout=0,  # 池化层丢弃率，默认为 0
        pooler_hidden_act="gelu",  # 池化层隐藏层激活函数，默认为 gelu
        **kwargs,  # 其他参数，以字典形式传递
    ):
        # 调用父类的初始化方法
        super().__init__(**kwargs)
    
        # 初始化 Transformer 模型的各个参数
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.relative_attention = relative_attention
        self.max_relative_positions = max_relative_positions
        self.pad_token_id = pad_token_id
        self.position_biased_input = position_biased_input
    
        # 兼容旧版本
        if isinstance(pos_att_type, str):
            # 将位置注意力类型转换为列表形式
            pos_att_type = [x.strip() for x in pos_att_type.lower().split("|")]
    
        # 设置位置注意力类型
        self.pos_att_type = pos_att_type
        self.vocab_size = vocab_size
        self.layer_norm_eps = layer_norm_eps
    
        # 获取池化层隐藏层大小，默认为隐藏层大小
        self.pooler_hidden_size = kwargs.get("pooler_hidden_size", hidden_size)
        self.pooler_dropout = pooler_dropout
        self.pooler_hidden_act = pooler_hidden_act
# 从 transformers.models.deberta_v2.configuration_deberta_v2.DebertaV2OnnxConfig 复制而来的类 DebertaOnnxConfig，继承自OnnxConfig
class DebertaOnnxConfig(OnnxConfig):
    # 定义 inputs 属性，返回一个字典，键为字符串，值为另一个字典
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 如果任务类型为 "multiple-choice"，则动态轴为 {0: "batch", 1: "choice", 2: "sequence"}
        if self.task == "multiple-choice":
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        else:
            # 否则动态轴为 {0: "batch", 1: "sequence"}
            dynamic_axis = {0: "batch", 1: "sequence"}
        # 如果配置的 type_vocab_size 大于 0
        if self._config.type_vocab_size > 0:
            # 返回一个有序字典，包含输入参数 "input_ids", "attention_mask", "token_type_ids"，对应动态轴
            return OrderedDict(
                [("input_ids", dynamic_axis), ("attention_mask", dynamic_axis), ("token_type_ids", dynamic_axis)]
            )
        else:
            # 否则返回一个有序字典，包含输入参数 "input_ids", "attention_mask"，对应动态轴
            return OrderedDict([("input_ids", dynamic_axis), ("attention_mask", dynamic_axis)])

    # 定义 default_onnx_opset 属性，返回一个整数
    @property
    def default_onnx_opset(self) -> int:
        # 返回 opset 版本号 12
        return 12

    # 定义 generate_dummy_inputs 方法，用于生成虚拟输入数据，返回结果为一个字典
    def generate_dummy_inputs(
        self,
        preprocessor: Union["PreTrainedTokenizerBase", "FeatureExtractionMixin"],
        batch_size: int = -1,
        seq_length: int = -1,
        num_choices: int = -1,
        is_pair: bool = False,
        framework: Optional["TensorType"] = None,
        num_channels: int = 3,
        image_width: int = 40,
        image_height: int = 40,
        tokenizer: "PreTrainedTokenizerBase" = None,
    ) -> Mapping[str, Any]:
        # 调用父类方法生成虚拟输入数据
        dummy_inputs = super().generate_dummy_inputs(preprocessor=preprocessor, framework=framework)
        # 如果配置的 type_vocab_size 为 0 并且 dummy_inputs 包含 "token_type_ids"
        if self._config.type_vocab_size == 0 and "token_type_ids" in dummy_inputs:
            # 从 dummy_inputs 中删除 "token_type_ids"
            del dummy_inputs["token_type_ids"]
        # 返回虚拟输入数据字典
        return dummy_inputs
```