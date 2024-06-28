# `.\models\longformer\configuration_longformer.py`

```py
# coding=utf-8
# Copyright 2020 The Allen Institute for AI team and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Longformer configuration"""

# 导入必要的库和模块
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, List, Mapping, Optional, Union

# 导入预训练配置类
from ...configuration_utils import PretrainedConfig
# 导入ONNX配置类
from ...onnx import OnnxConfig
# 导入TensorType和logging工具
from ...utils import TensorType, logging

# 检查类型，导入额外依赖
if TYPE_CHECKING:
    from ...onnx.config import PatchingSpec
    from ...tokenization_utils_base import PreTrainedTokenizerBase

# 获取日志记录器
logger = logging.get_logger(__name__)

# 预训练模型配置文件映射字典
LONGFORMER_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "allenai/longformer-base-4096": "https://huggingface.co/allenai/longformer-base-4096/resolve/main/config.json",
    "allenai/longformer-large-4096": "https://huggingface.co/allenai/longformer-large-4096/resolve/main/config.json",
    "allenai/longformer-large-4096-finetuned-triviaqa": (
        "https://huggingface.co/allenai/longformer-large-4096-finetuned-triviaqa/resolve/main/config.json"
    ),
    "allenai/longformer-base-4096-extra.pos.embd.only": (
        "https://huggingface.co/allenai/longformer-base-4096-extra.pos.embd.only/resolve/main/config.json"
    ),
    "allenai/longformer-large-4096-extra.pos.embd.only": (
        "https://huggingface.co/allenai/longformer-large-4096-extra.pos.embd.only/resolve/main/config.json"
    ),
}


class LongformerConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`LongformerModel`] or a [`TFLongformerModel`]. It
    is used to instantiate a Longformer model according to the specified arguments, defining the model architecture.

    This is the configuration class to store the configuration of a [`LongformerModel`]. It is used to instantiate an
    Longformer model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the LongFormer
    [allenai/longformer-base-4096](https://huggingface.co/allenai/longformer-base-4096) architecture with a sequence
    length 4,096.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    # 定义 Longformer 模型的配置类，用于配置模型的各种参数
    class LongformerConfig:
        # 构造函数，初始化 Longformer 模型的配置参数
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
            attention_window=512
        ):
            # 设置 Longformer 模型的词汇表大小
            vocab_size (`int`, *optional*, defaults to 30522):
                Vocabulary size of the Longformer model. Defines the number of different tokens that can be represented by
                the `inputs_ids` passed when calling [`LongformerModel`] or [`TFLongformerModel`].
            # 设置编码器层和池化层的隐藏单元数
            hidden_size (`int`, *optional*, defaults to 768):
                Dimensionality of the encoder layers and the pooler layer.
            # 设置 Transformer 编码器中的隐藏层数量
            num_hidden_layers (`int`, *optional*, defaults to 12):
                Number of hidden layers in the Transformer encoder.
            # 设置每个注意力层中的注意力头数
            num_attention_heads (`int`, *optional*, defaults to 12):
                Number of attention heads for each attention layer in the Transformer encoder.
            # 设置 Transformer 编码器中“中间”（通常称为前馈）层的维度
            intermediate_size (`int`, *optional*, defaults to 3072):
                Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
            # 设置编码器和池化器中的非线性激活函数
            hidden_act (`str` or `Callable`, *optional*, defaults to `"gelu"`):
                The non-linear activation function (function or string) in the encoder and pooler. If string, `"gelu"`,
                `"relu"`, `"silu"` and `"gelu_new"` are supported.
            # 设置嵌入层、编码器和池化器中所有全连接层的 dropout 概率
            hidden_dropout_prob (`float`, *optional*, defaults to 0.1):
                The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
            # 设置注意力概率的 dropout 比率
            attention_probs_dropout_prob (`float`, *optional*, defaults to 0.1):
                The dropout ratio for the attention probabilities.
            # 设置模型可能使用的最大序列长度
            max_position_embeddings (`int`, *optional*, defaults to 512):
                The maximum sequence length that this model might ever be used with. Typically set this to something large
                just in case (e.g., 512 or 1024 or 2048).
            # 设置 `token_type_ids` 的词汇表大小
            type_vocab_size (`int`, *optional*, defaults to 2):
                The vocabulary size of the `token_type_ids` passed when calling [`LongformerModel`] or
                [`TFLongformerModel`].
            # 设置所有权重矩阵初始化时的截断正态分布的标准差
            initializer_range (`float`, *optional*, defaults to 0.02):
                The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
            # 设置层归一化层使用的 epsilon 值
            layer_norm_eps (`float`, *optional*, defaults to 1e-12):
                The epsilon used by the layer normalization layers.
            # 设置每个标记周围的注意力窗口大小
            attention_window (`int` or `List[int]`, *optional*, defaults to 512):
                Size of an attention window around each token. If an `int`, use the same size for all layers. To specify a
                different window size for each layer, use a `List[int]` where `len(attention_window) == num_hidden_layers`.
    
        Example:
    
        ```
        >>> from transformers import LongformerConfig, LongformerModel
    
        >>> # Initializing a Longformer configuration
        >>> configuration = LongformerConfig()
    
        >>> # Initializing a model from the configuration
        >>> model = LongformerModel(configuration)
    
        >>> # Accessing the model configuration
        >>> configuration = model.config
        ```
        
        # 设置模型类型为 Longformer
        model_type = "longformer"
    def __init__(
        self,
        attention_window: Union[List[int], int] = 512,
        sep_token_id: int = 2,
        pad_token_id: int = 1,
        bos_token_id: int = 0,
        eos_token_id: int = 2,
        vocab_size: int = 30522,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        initializer_range: float = 0.02,
        layer_norm_eps: float = 1e-12,
        onnx_export: bool = False,
        **kwargs,
    ):
        """
        构造函数，初始化 LongformerConfig 对象。

        参数:
        - attention_window: 注意力窗口大小，可以是整数或整数列表，默认为 512
        - sep_token_id: 分隔符 token 的 ID，默认为 2
        - pad_token_id: 填充 token 的 ID，默认为 1
        - bos_token_id: 文本开始 token 的 ID，默认为 0
        - eos_token_id: 文本结束 token 的 ID，默认为 2
        - vocab_size: 词汇表大小，默认为 30522
        - hidden_size: 隐藏层大小，默认为 768
        - num_hidden_layers: 隐藏层的数量，默认为 12
        - num_attention_heads: 注意力头的数量，默认为 12
        - intermediate_size: 中间层大小，默认为 3072
        - hidden_act: 隐藏层激活函数，默认为 "gelu"
        - hidden_dropout_prob: 隐藏层的 dropout 概率，默认为 0.1
        - attention_probs_dropout_prob: 注意力概率的 dropout 概率，默认为 0.1
        - max_position_embeddings: 最大位置嵌入大小，默认为 512
        - type_vocab_size: 类型词汇表的大小，默认为 2
        - initializer_range: 初始化范围，默认为 0.02
        - layer_norm_eps: 层归一化的 epsilon 值，默认为 1e-12
        - onnx_export: 是否导出到 ONNX 格式，默认为 False
        """
        # 调用父类构造函数，初始化基类的填充 token ID 和其他关键字参数
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        # 将参数赋值给对象的属性
        self.attention_window = attention_window
        self.sep_token_id = sep_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
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
        self.onnx_export = onnx_export
# 定义一个继承自OnnxConfig的LongformerOnnxConfig类，用于配置Longformer模型的导出设置
class LongformerOnnxConfig(OnnxConfig):

    # 初始化方法，接收预训练配置、任务名称和补丁规格列表
    def __init__(self, config: "PretrainedConfig", task: str = "default", patching_specs: "List[PatchingSpec]" = None):
        # 调用父类的初始化方法
        super().__init__(config, task, patching_specs)
        # 设置onnx_export属性为True，表示要导出为ONNX格式
        config.onnx_export = True

    # inputs属性，返回一个有序字典，描述模型的输入格式
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 如果任务是多选题(multiple-choice)，动态轴设置为包含batch、choice、sequence
        if self.task == "multiple-choice":
            dynamic_axis = {0: "batch", 1: "choice", 2: "sequence"}
        else:
            # 否则动态轴设置为包含batch、sequence
            dynamic_axis = {0: "batch", 1: "sequence"}
        return OrderedDict(
            [
                ("input_ids", dynamic_axis),                 # 输入的token IDs
                ("attention_mask", dynamic_axis),           # 注意力遮罩
                ("global_attention_mask", dynamic_axis),    # 全局注意力遮罩
            ]
        )

    # outputs属性，返回一个描述模型输出格式的字典
    @property
    def outputs(self) -> Mapping[str, Mapping[int, str]]:
        # 调用父类的outputs方法获取输出字典
        outputs = super().outputs
        # 如果任务是默认任务(default)，添加额外的汇聚输出(pooler_output)
        if self.task == "default":
            outputs["pooler_output"] = {0: "batch"}
        return outputs

    # atol_for_validation属性，返回模型转换验证时的绝对误差容差
    @property
    def atol_for_validation(self) -> float:
        """
        What absolute tolerance value to use during model conversion validation.

        Returns:
            Float absolute tolerance value.
        """
        return 1e-4

    # default_onnx_opset属性，返回模型导出时所需的默认ONNX操作集版本号
    @property
    def default_onnx_opset(self) -> int:
        # 需要>=14版本支持tril运算符
        return max(super().default_onnx_opset, 14)

    # generate_dummy_inputs方法，生成用于模型导出的虚拟输入数据
    def generate_dummy_inputs(
        self,
        tokenizer: "PreTrainedTokenizerBase",
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
    ) -> Mapping[str, Any]:
        # 调用父类的generate_dummy_inputs方法生成基础输入
        inputs = super().generate_dummy_inputs(
            preprocessor=tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework
        )
        import torch

        # 设置全局注意力遮罩为与input_ids相同形状的全零张量
        inputs["global_attention_mask"] = torch.zeros_like(inputs["input_ids"])
        # 每隔一个token将全局注意力遮罩的相应位置设为1
        inputs["global_attention_mask"][:, ::2] = 1

        return inputs
```