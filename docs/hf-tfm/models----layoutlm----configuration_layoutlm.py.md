# `.\models\layoutlm\configuration_layoutlm.py`

```
# coding=utf-8
# Copyright 2010, The Microsoft Research Asia LayoutLM Team authors
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
""" LayoutLM model configuration"""
# 导入所需的模块
from collections import OrderedDict
from typing import Any, List, Mapping, Optional

# 导入预训练模型配置类和预训练分词器
from ... import PretrainedConfig, PreTrainedTokenizer
# 导入ONNX相关的配置和补丁规范
from ...onnx import OnnxConfig, PatchingSpec
# 导入工具函数：张量类型、是否有torch可用、日志记录
from ...utils import TensorType, is_torch_available, logging

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# LayoutLM预训练模型配置文件映射表，包含预训练模型的名称和对应的配置文件URL
LAYOUTLM_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "microsoft/layoutlm-base-uncased": (
        "https://huggingface.co/microsoft/layoutlm-base-uncased/resolve/main/config.json"
    ),
    "microsoft/layoutlm-large-uncased": (
        "https://huggingface.co/microsoft/layoutlm-large-uncased/resolve/main/config.json"
    ),
}

# LayoutLM配置类，继承自PretrainedConfig
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
    ```"""

    # 指定模型类型为"layoutlm"
    model_type = "layoutlm"

    # LayoutLM配置类的初始化函数，定义了各种模型参数和超参数
    def __init__(
        self,
        vocab_size=30522,  # 词汇表大小，默认为30522
        hidden_size=768,   # 隐藏层大小，默认为768
        num_hidden_layers=12,  # 隐藏层层数，默认为12
        num_attention_heads=12,  # 注意力头数，默认为12
        intermediate_size=3072,  # 中间层大小，默认为3072
        hidden_act="gelu",  # 隐藏层激活函数，默认为gelu
        hidden_dropout_prob=0.1,  # 隐藏层dropout概率，默认为0.1
        attention_probs_dropout_prob=0.1,  # 注意力概率dropout概率，默认为0.1
        max_position_embeddings=512,  # 最大位置嵌入长度，默认为512
        type_vocab_size=2,  # 类型词汇表大小，默认为2
        initializer_range=0.02,  # 初始化范围，默认为0.02
        layer_norm_eps=1e-12,  # 层归一化epsilon值，默认为1e-12
        pad_token_id=0,  # 填充token的ID，默认为0
        position_embedding_type="absolute",  # 位置嵌入类型，默认为绝对位置嵌入
        use_cache=True,  # 是否使用缓存，默认为True
        max_2d_position_embeddings=1024,  # 最大二维位置嵌入长度，默认为1024
        **kwargs,  # 其余参数，用于接收任意额外的关键字参数
        ):
        # 调用父类的初始化方法，设置填充标记ID和其他参数
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        # 设置词汇表大小
        self.vocab_size = vocab_size
        # 设置隐藏层大小
        self.hidden_size = hidden_size
        # 设置隐藏层数量
        self.num_hidden_layers = num_hidden_layers
        # 设置注意力头数量
        self.num_attention_heads = num_attention_heads
        # 设置隐藏层激活函数类型
        self.hidden_act = hidden_act
        # 设置中间层大小
        self.intermediate_size = intermediate_size
        # 设置隐藏层的dropout概率
        self.hidden_dropout_prob = hidden_dropout_prob
        # 设置注意力概率的dropout概率
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        # 设置最大位置嵌入的大小
        self.max_position_embeddings = max_position_embeddings
        # 设置类型词汇表的大小
        self.type_vocab_size = type_vocab_size
        # 设置初始化范围
        self.initializer_range = initializer_range
        # 设置层归一化的epsilon值
        self.layer_norm_eps = layer_norm_eps
        # 设置位置嵌入的类型
        self.position_embedding_type = position_embedding_type
        # 设置是否使用缓存
        self.use_cache = use_cache
        # 设置二维位置嵌入的最大值
        self.max_2d_position_embeddings = max_2d_position_embeddings
# LayoutLMOnnxConfig 类，继承自 OnnxConfig 类
class LayoutLMOnnxConfig(OnnxConfig):
    # 初始化方法
    def __init__(
        self,
        config: PretrainedConfig,  # 预训练配置对象
        task: str = "default",  # 任务名称，默认为 "default"
        patching_specs: List[PatchingSpec] = None,  # 补丁规范列表，默认为空
    ):
        # 调用父类 OnnxConfig 的初始化方法
        super().__init__(config, task=task, patching_specs=patching_specs)
        # 设置最大的二维位置嵌入数量为配置对象的最大二维位置嵌入数量减一
        self.max_2d_positions = config.max_2d_position_embeddings - 1

    # inputs 属性方法，返回一个有序字典，描述了模型的输入
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        return OrderedDict(
            [
                ("input_ids", {0: "batch", 1: "sequence"}),  # 输入的 token IDs，第一维为 batch，第二维为序列
                ("bbox", {0: "batch", 1: "sequence"}),  # 包围框信息，第一维为 batch，第二维为序列
                ("attention_mask", {0: "batch", 1: "sequence"}),  # 注意力遮罩，第一维为 batch，第二维为序列
                ("token_type_ids", {0: "batch", 1: "sequence"}),  # token 类型 IDs，第一维为 batch，第二维为序列
            ]
        )

    # generate_dummy_inputs 方法，生成用于 ONNX 导出器的虚拟输入
    def generate_dummy_inputs(
        self,
        tokenizer: PreTrainedTokenizer,  # 预训练 tokenizer 对象
        batch_size: int = -1,  # 批量大小，默认为动态轴
        seq_length: int = -1,  # 序列长度，默认为动态轴
        is_pair: bool = False,  # 是否为句对输入，默认为 False
        framework: Optional[TensorType] = None,  # 框架类型，可选的 TensorType 对象
    ) -> Mapping[str, Any]:  # 返回一个映射，包含提供给模型前向函数的参数
        """
        生成用于 ONNX 导出器的特定框架的输入

        Args:
            tokenizer: 与该模型配置关联的 tokenizer
            batch_size: 导出模型的批次大小（整数）（-1 表示动态轴）
            seq_length: 导出模型的序列长度（整数）（-1 表示动态轴）
            is_pair: 表示输入是否为句对（句子1，句子2）
            framework: tokenizer 将为其生成张量的框架（可选）

        Returns:
            Mapping[str, Tensor]，包含要提供给模型前向函数的参数
        """

        # 调用父类的 generate_dummy_inputs 方法，获取基本的输入字典
        input_dict = super().generate_dummy_inputs(
            tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework
        )

        # 生成一个虚拟的包围框
        box = [48, 84, 73, 128]

        # 如果框架不是 PyTorch，抛出 NotImplementedError
        if not framework == TensorType.PYTORCH:
            raise NotImplementedError("Exporting LayoutLM to ONNX is currently only supported for PyTorch.")

        # 如果没有安装 PyTorch，抛出 ValueError
        if not is_torch_available():
            raise ValueError("Cannot generate dummy inputs without PyTorch installed.")
        import torch

        # 获取输入中 input_ids 的批次大小和序列长度
        batch_size, seq_length = input_dict["input_ids"].shape
        # 将包围框信息转换为 PyTorch 张量，并在批次维度上进行复制
        input_dict["bbox"] = torch.tensor([*[box] * seq_length]).tile(batch_size, 1, 1)
        return input_dict
```