# `.\models\deberta_v2\configuration_deberta_v2.py`

```py
# 设置文件编码为 utf-8
# 版权声明和许可信息
# 导入所需的模块和类型
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Mapping, Optional, Union
# 导入预训练配置类，导入onnx配置，导入日志工具
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfig
from ...utils import logging

# 如果是类型检查，则导入FeatureExtractionMixin, PreTrainedTokenizerBase, TensorType
if TYPE_CHECKING:
    from ... import FeatureExtractionMixin, PreTrainedTokenizerBase, TensorType

# 获取日志记录器
logger = logging.get_logger(__name__)

# DeBERTa-v2预训练配置的映射
DEBERTA_V2_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/deberta-v2-xlarge": "https://huggingface.co/microsoft/deberta-v2-xlarge/resolve/main/config.json",
    "microsoft/deberta-v2-xxlarge": "https://huggingface.co/microsoft/deberta-v2-xxlarge/resolve/main/config.json",
    "microsoft/deberta-v2-xlarge-mnli": (
        "https://huggingface.co/microsoft/deberta-v2-xlarge-mnli/resolve/main/config.json"
    ),
    "microsoft/deberta-v2-xxlarge-mnli": (
        "https://huggingface.co/microsoft/deberta-v2-xxlarge-mnli/resolve/main/config.json"
    ),
}

# DebertaV2Config 类，继承自 PretrainedConfig 类
class DebertaV2Config(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`DebertaV2Model`]. It is used to instantiate a
    DeBERTa-v2 model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the DeBERTa
    [microsoft/deberta-v2-xlarge](https://huggingface.co/microsoft/deberta-v2-xlarge) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Example:

    ```python
    >>> from transformers import DebertaV2Config, DebertaV2Model

    >>> # Initializing a DeBERTa-v2 microsoft/deberta-v2-xlarge style configuration
    >>> configuration = DebertaV2Config()

    >>> # Initializing a model (with random weights) from the microsoft/deberta-v2-xlarge style configuration
    >>> model = DebertaV2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```py"""

    # 模型类型为 "deberta-v2"
    model_type = "deberta-v2"
    # 这是一个 PyTorch 模型配置类的构造函数
    def __init__(
        self,
        # 词汇表大小
        vocab_size=128100,
        # 隐藏层大小
        hidden_size=1536,
        # 隐藏层数
        num_hidden_layers=24,
        # 注意力头数
        num_attention_heads=24,
        # 中间层大小
        intermediate_size=6144,
        # 激活函数
        hidden_act="gelu",
        # 隐藏层dropout概率
        hidden_dropout_prob=0.1,
        # 注意力层dropout概率
        attention_probs_dropout_prob=0.1,
        # 最大序列长度
        max_position_embeddings=512,
        # Token类型数
        type_vocab_size=0,
        # 初始化范围
        initializer_range=0.02,
        # LayerNorm的epsilon
        layer_norm_eps=1e-7,
        # 是否使用相对注意力
        relative_attention=False,
        # 最大相对位置
        max_relative_positions=-1,
        # pad token ID
        pad_token_id=0,
        # 位置编码是否作为输入
        position_biased_input=True,
        # 位置编码类型
        pos_att_type=None,
        # Pooler层dropout概率
        pooler_dropout=0,
        # Pooler层激活函数
        pooler_hidden_act="gelu",
        **kwargs,
    ):
        # 初始化父类
        super().__init__(**kwargs)
    
        # 设置模型各种属性
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
    
        # 兼容性处理
        if isinstance(pos_att_type, str):
            pos_att_type = [x.strip() for x in pos_att_type.lower().split("|")]
    
        self.pos_att_type = pos_att_type
        self.vocab_size = vocab_size
        self.layer_norm_eps = layer_norm_eps
    
        # Pooler层配置
        self.pooler_hidden_size = kwargs.get("pooler_hidden_size", hidden_size)
        self.pooler_dropout = pooler_dropout
        self.pooler_hidden_act = pooler_hidden_act
# 定义 DebertaV2OnnxConfig 类，继承自 OnnxConfig 类
class DebertaV2OnnxConfig(OnnxConfig):
    # 输入属性，返回输入名称到动态轴的映射字典
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 如果任务类型是多项选择
        if self.task == "multiple-choice":
            # 动态轴包括批次、选择和序列
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        else:
            # 动态轴包括批次和序列
            dynamic_axis = {0: "batch", 1: "sequence"}
        # 如果配置中的词汇类型数大于 0
        if self._config.type_vocab_size > 0:
            # 返回输入名称到动态轴的有序字典，包括输入 ids、注意力掩码和标记类型 ids
            return OrderedDict(
                [("input_ids", dynamic_axis), ("attention_mask", dynamic_axis), ("token_type_ids", dynamic_axis)]
            )
        else:
            # 返回输入名称到动态轴的有序字典，包括输入 ids 和注意力掩码
            return OrderedDict([("input_ids", dynamic_axis), ("attention_mask", dynamic_axis)])

    # 默认的 ONNX 操作集版本
    @property
    def default_onnx_opset(self) -> int:
        return 12

    # 生成虚拟输入
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
        # 调用父类的方法生成虚拟输入
        dummy_inputs = super().generate_dummy_inputs(preprocessor=preprocessor, framework=framework)
        # 如果配置中的词汇类型数为 0 且虚拟输入中包含 token_type_ids，则删除 token_type_ids
        if self._config.type_vocab_size == 0 and "token_type_ids" in dummy_inputs:
            del dummy_inputs["token_type_ids"]
        # 返回虚拟输入
        return dummy_inputs
```