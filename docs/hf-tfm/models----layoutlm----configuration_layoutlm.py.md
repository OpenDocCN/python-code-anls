# `.\models\layoutlm\configuration_layoutlm.py`

```py
# 导入所需的模块和类
from collections import OrderedDict  # 导入有序字典类
from typing import Any, List, Mapping, Optional  # 导入类型提示

# 导入预训练配置、预训练分词器、Onnx 配置和 PatchingSpec，以及 TensorType 类型
from ... import PretrainedConfig, PreTrainedTokenizer  
from ...onnx import OnnxConfig, PatchingSpec  
from ...utils import TensorType, is_torch_available, logging  # 导入工具函数和类

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# 预训练配置文件映射字典，映射预训练模型名称到其配置文件的 URL
LAYOUTLM_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/layoutlm-base-uncased": (
        "https://huggingface.co/microsoft/layoutlm-base-uncased/resolve/main/config.json"
    ),
    "microsoft/layoutlm-large-uncased": (
        "https://huggingface.co/microsoft/layoutlm-large-uncased/resolve/main/config.json"
    ),
}

# LayoutLMConfig 类，继承自 PretrainedConfig 类
class LayoutLMConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LayoutLMModel`]. It is used to instantiate a
    LayoutLM model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the LayoutLM
    [microsoft/layoutlm-base-uncased](https://huggingface.co/microsoft/layoutlm-base-uncased) architecture.

    Configuration objects inherit from [`BertConfig`] and can be used to control the model outputs. Read the
    documentation from [`BertConfig`] for more information.

    Examples:

    ```python
    >>> from transformers import LayoutLMConfig, LayoutLMModel

    >>> # Initializing a LayoutLM configuration
    >>> configuration = LayoutLMConfig()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = LayoutLMModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```py"""

    # 模型类型
    model_type = "layoutlm"

    # 初始化函数
    def __init__(
        self,
        vocab_size=30522,  # 词汇表大小，默认为 30522
        hidden_size=768,  # 隐藏层大小，默认为 768
        num_hidden_layers=12,  # 隐藏层的数量，默认为 12
        num_attention_heads=12,  # 注意力头的数量，默认为 12
        intermediate_size=3072,  # 中间层大小，默认为 3072
        hidden_act="gelu",  # 隐藏层激活函数，默认为 "gelu"
        hidden_dropout_prob=0.1,  # 隐藏层的 dropout 概率，默认为 0.1
        attention_probs_dropout_prob=0.1,  # 注意力机制的 dropout 概率，默认为 0.1
        max_position_embeddings=512,  # 最大位置编码数量，默认为 512
        type_vocab_size=2,  # 类型词汇表大小，默认为 2
        initializer_range=0.02,  # 初始化范围，默认为 0.02
        layer_norm_eps=1e-12,  # 层归一化 epsilon，默认为 1e-12
        pad_token_id=0,  # 填充 token 的 id，默认为 0
        position_embedding_type="absolute",  # 位置编码类型，默认为 "absolute"
        use_cache=True,  # 是否使用缓存，默认为 True
        max_2d_position_embeddings=1024,  # 最大二维位置编码数量，默认为 1024
        **kwargs,  # 其他关键字参数
        # 调用父类的初始化方法，传入pad_token_id和其他可选参数
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        # 初始化词汇表大小
        self.vocab_size = vocab_size
        # 初始化隐藏层大小
        self.hidden_size = hidden_size
        # 初始化隐藏层的数量
        self.num_hidden_layers = num_hidden_layers
        # 初始化注意力头的数量
        self.num_attention_heads = num_attention_heads
        # 初始化隐藏层的激活函数类型
        self.hidden_act = hidden_act
        # 初始化中间层大小
        self.intermediate_size = intermediate_size
        # 初始化隐藏层的dropout概率
        self.hidden_dropout_prob = hidden_dropout_prob
        # 初始化注意力概率的dropout概率
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        # 初始化最大位置嵌入数
        self.max_position_embeddings = max_position_embeddings
        # 初始化类型嵌入数
        self.type_vocab_size = type_vocab_size
        # 初始化初始化范围
        self.initializer_range = initializer_range
        # 初始化层规范化的epsilon值
        self.layer_norm_eps = layer_norm_eps
        # 初始化位置嵌入类型
        self.position_embedding_type = position_embedding_type
        # 初始化是否使用缓存
        self.use_cache = use_cache
        # 初始化最大二维位置嵌入数
        self.max_2d_position_embeddings = max_2d_position_embeddings
# 定义 LayoutLMOnnxConfig 类，继承自 OnnxConfig 类
class LayoutLMOnnxConfig(OnnxConfig):

    # 构造函数初始化方法
    def __init__(
        self,
        config: PretrainedConfig,  # 指定参数类型为 PretrainedConfig
        task: str = "default",  # 设置默认参数值为 "default"
        patching_specs: List[PatchingSpec] = None,  # 设置默认参数值为 None
    ):
        # 调用父类的初始化方法
        super().__init__(config, task=task, patching_specs=patching_specs)
        # 设置 max_2d_positions 属性值为 config.max_2d_position_embeddings - 1
        self.max_2d_positions = config.max_2d_position_embeddings - 1

    # inputs 属性的 getter 方法
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 返回一个有序字典，包含输入名称和对应索引的映射
        return OrderedDict(
            [
                ("input_ids", {0: "batch", 1: "sequence"}),
                ("bbox", {0: "batch", 1: "sequence"}),
                ("attention_mask", {0: "batch", 1: "sequence"}),
                ("token_type_ids", {0: "batch", 1: "sequence"}),
            ]
        )

    # 生成虚拟输入的方法
    def generate_dummy_inputs(
        self,
        tokenizer: PreTrainedTokenizer,  # 指定参数类型为 PreTrainedTokenizer
        batch_size: int = -1,  # 设置默认参数值为 -1
        seq_length: int = -1,  # 设置默认参数值为 -1
        is_pair: bool = False,  # 设置默认参数值为 False
        framework: Optional[TensorType] = None,  # 设置默认参数值为 None
    ) -> Mapping[str, Any]:  # 指定返回值类型为 Mapping[str, Any]

        """
        Generate inputs to provide to the ONNX exporter for the specific framework

        Args:
            tokenizer: The tokenizer associated with this model configuration
            batch_size: The batch size (int) to export the model for (-1 means dynamic axis)
            seq_length: The sequence length (int) to export the model for (-1 means dynamic axis)
            is_pair: Indicate if the input is a pair (sentence 1, sentence 2)
            framework: The framework (optional) the tokenizer will generate tensor for

        Returns:
            Mapping[str, Tensor] holding the kwargs to provide to the model's forward function
        """

        # 调用父类的 generate_dummy_inputs 方法，生成虚拟输入
        input_dict = super().generate_dummy_inputs(
            tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework
        )

        # 生成一个虚拟的 bbox
        box = [48, 84, 73, 128]

        # 如果框架不是 PyTorch，则抛出 NotImplementedError
        if not framework == TensorType.PYTORCH:
            raise NotImplementedError("Exporting LayoutLM to ONNX is currently only supported for PyTorch.")

        # 如果没有安装 PyTorch，则抛出 ValueError
        if not is_torch_available():
            raise ValueError("Cannot generate dummy inputs without PyTorch installed.")
        import torch

        # 获取 batch_size 和 seq_length
        batch_size, seq_length = input_dict["input_ids"].shape
        # 生成虚拟的 bbox，使用 torch.tensor 创建张量并进行扩充
        input_dict["bbox"] = torch.tensor([*[box] * seq_length]).tile(batch_size, 1, 1)
        return input_dict
```