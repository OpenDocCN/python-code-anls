# `.\models\bloom\configuration_bloom.py`

```py
# coding=utf-8
# Copyright 2022 the Big Science Workshop and HuggingFace Inc. team.  All rights reserved.
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

""" Bloom configuration"""

from collections import OrderedDict  # 导入有序字典类
from typing import TYPE_CHECKING, Any, List, Mapping, Optional  # 导入类型提示

from packaging import version  # 导入版本号处理模块

if TYPE_CHECKING:
    from ... import PreTrainedTokenizer, TensorType  # 条件导入

from ...configuration_utils import PretrainedConfig  # 导入预训练配置基类
from ...onnx import OnnxConfigWithPast, PatchingSpec  # 导入ONNX相关配置
from ...utils import is_torch_available, logging  # 导入Torch是否可用判断和日志工具

logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器对象

BLOOM_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "bigscience/bloom": "https://huggingface.co/bigscience/bloom/resolve/main/config.json",
    "bigscience/bloom-560m": "https://huggingface.co/bigscience/bloom-560m/blob/main/config.json",
    "bigscience/bloom-1b1": "https://huggingface.co/bigscience/bloom-1b1/blob/main/config.json",
    "bigscience/bloom-1b7": "https://huggingface.co/bigscience/bloom-1b7/blob/main/config.json",
    "bigscience/bloom-3b": "https://huggingface.co/bigscience/bloom-3b/blob/main/config.json",
    "bigscience/bloom-7b1": "https://huggingface.co/bigscience/bloom-7b1/blob/main/config.json",
}

class BloomConfig(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`BloomModel`]. It is used to instantiate a Bloom
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to the Bloom architecture
    [bigscience/bloom](https://huggingface.co/bigscience/bloom).

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.
    """
    pass  # 空的配置类，用于存储Bloom模型的配置信息
    # 定义 Bloom 模型的参数
    Args:
        vocab_size (`int`, *optional*, defaults to 250880):
            Bloom 模型的词汇表大小，定义了在调用 `BloomModel` 时 `inputs_ids` 可以表示的最大不同 token 数量。
            参考 [此讨论](https://huggingface.co/bigscience/bloom/discussions/120#633d28389addb8530b406c2a) 以了解 `vocab_size` 的定义。
        hidden_size (`int`, *optional*, defaults to 64):
            嵌入和隐藏状态的维度。
        n_layer (`int`, *optional*, defaults to 2):
            Transformer 编码器中的隐藏层数量。
        n_head (`int`, *optional*, defaults to 8):
            Transformer 编码器中每个注意力层的注意力头数。
        layer_norm_epsilon (`float`, *optional*, defaults to 1e-5):
            层归一化层中使用的 epsilon 值。
        initializer_range (`float`, *optional*, defaults to 0.02):
            用于初始化所有权重矩阵的截断正态分布的标准差。
        apply_residual_connection_post_layernorm (`bool`, *optional*, defaults to `False`):
            如果启用，则在 transformer 块中使用隐藏状态的层归一化作为残差。
        hidden_dropout (`float`, *optional*, defaults to 0.1):
            应用于偏置丢弃的 dropout 率。
        attention_dropout (`float`, *optional*, defaults to 0.1):
            应用于注意力概率的 dropout 率。
        use_cache (`bool`, *optional*, defaults to `True`):
            模型是否应返回最后的 key/values 注意力（不是所有模型都使用）。
        pretraining_tp (`int`, *optional*, defaults to `1`):
            实验性功能。Megatron 预训练期间使用的张量并行性等级。请参考 [此文档](https://huggingface.co/docs/transformers/parallelism) 了解更多信息。
            此值对确保预训练结果的精确再现性至关重要。请参考 [此问题](https://github.com/pytorch/pytorch/issues/76232)。
            注意，仅在 `slow_but_exact=True` 时启用。
        slow_but_exact (`bool`, *optional*, defaults to `False`):
            实验性功能。是否使用注意力机制的缓慢但精确实现。在合并 TP 等级张量时，由于切片操作，Megatron 训练模型和我们模型之间的结果可能会略有不同。
            请参考 [此问题](https://github.com/pytorch/pytorch/issues/76232)。启用此功能可获得更准确的结果，但会增加推断的计算时间。
            一旦主模型通过 TP_rank=1 进行了精细调整，这个问题可能会在未来得到解决。
    # Importing necessary components from the transformers library
    from transformers import BloomConfig, BloomModel
    
    # Initializing a Bloom configuration object
    configuration = BloomConfig()
    
    # Initializing a Bloom model with random weights based on the configuration
    model = BloomModel(configuration)
    
    # Accessing the configuration attributes of the initialized model
    configuration = model.config
    
    model_type = "bloom"  # Setting the model type to "bloom"
    keys_to_ignore_at_inference = ["past_key_values"]  # Defining keys to ignore during inference
    
    # Mapping attributes for backward compatibility and clarity
    attribute_map = {
        "num_hidden_layers": "n_layer",  # Mapping number of hidden layers to 'n_layer'
        "num_attention_heads": "n_head",  # Mapping number of attention heads to 'n_head'
    }
    
    def __init__(
        self,
        vocab_size=250880,
        hidden_size=64,
        n_layer=2,
        n_head=8,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        use_cache=True,
        bos_token_id=1,
        eos_token_id=2,
        apply_residual_connection_post_layernorm=False,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        pretraining_tp=1,  # TP rank used when training with megatron
        slow_but_exact=False,
        **kwargs,
    ):
        # Initializing the model attributes with default values or provided kwargs
        self.vocab_size = vocab_size
        n_embed = kwargs.pop("n_embed", None)  # Handling backward compatibility with 'n_embed' kwarg
        self.hidden_size = hidden_size if n_embed is None else n_embed
        self.n_layer = n_layer
        self.n_head = n_head
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range
        self.use_cache = use_cache
        self.pretraining_tp = pretraining_tp
        self.apply_residual_connection_post_layernorm = apply_residual_connection_post_layernorm
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
    
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.slow_but_exact = slow_but_exact
    
        # Calling the superclass initializer with specific parameters
        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
# 定义 BloomOnnxConfig 类，继承自 OnnxConfigWithPast 类
class BloomOnnxConfig(OnnxConfigWithPast):
    # 设定 torch_onnx_minimum_version 属性为最低支持版本 1.12
    torch_onnx_minimum_version = version.parse("1.12")

    # 初始化方法，接收预训练配置 config，任务 task，默认补丁规格 patching_specs 和是否使用过去状态 use_past
    def __init__(
        self,
        config: PretrainedConfig,
        task: str = "default",
        patching_specs: List[PatchingSpec] = None,
        use_past: bool = False,
    ):
        # 调用父类的初始化方法，传递 config、task、patching_specs 和 use_past
        super().__init__(config, task=task, patching_specs=patching_specs, use_past=use_past)
        
        # 如果 self._config 没有定义 pad_token_id 属性，则设为默认值 0
        if not getattr(self._config, "pad_token_id", None):
            # TODO: how to do that better?
            self._config.pad_token_id = 0

    # inputs 属性，返回输入的映射关系，格式为 OrderedDict，键为字符串，值为映射关系的字典
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 创建通用的输入映射 common_inputs，包含 "input_ids" 键
        common_inputs = OrderedDict({"input_ids": {0: "batch", 1: "sequence"}})
        
        # 如果使用过去状态 self.use_past
        if self.use_past:
            # 使用 fill_with_past_key_values_ 方法填充 common_inputs，方向为 "inputs"，并反转值的形状
            self.fill_with_past_key_values_(common_inputs, direction="inputs", inverted_values_shape=True)
            # 添加 "attention_mask" 键，指定映射关系为 {0: "batch", 1: "past_sequence + sequence"}
            common_inputs["attention_mask"] = {0: "batch", 1: "past_sequence + sequence"}
        else:
            # 否则，只添加 "attention_mask" 键，映射关系为 {0: "batch", 1: "sequence"}
            common_inputs["attention_mask"] = {0: "batch", 1: "sequence"}

        # 返回最终的通用输入映射 common_inputs
        return common_inputs

    # num_layers 属性，返回配置中的层数 self._config.n_layer
    @property
    def num_layers(self) -> int:
        return self._config.n_layer

    # num_attention_heads 属性，返回配置中的注意力头数 self._config.n_head
    @property
    def num_attention_heads(self) -> int:
        return self._config.n_head

    # atol_for_validation 属性，返回用于验证的绝对容差值 1e-3
    @property
    def atol_for_validation(self) -> float:
        return 1e-3

    # generate_dummy_inputs 方法，生成虚拟输入数据
    def generate_dummy_inputs(
        self,
        tokenizer: "PreTrainedTokenizer",
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional["TensorType"] = None,
        ) -> Mapping[str, Any]:
        # 调用父类方法生成通用输入
        common_inputs = super(OnnxConfigWithPast, self).generate_dummy_inputs(
            tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework
        )

        # 按照 forward() 方法中的顺序排序输入
        ordered_inputs = OrderedDict({"input_ids": common_inputs["input_ids"]})

        # 如果需要添加过去的键（past_keys）
        if self.use_past:
            # 检查是否安装了 PyTorch，否则抛出错误
            if not is_torch_available():
                raise ValueError("Cannot generate dummy past_keys inputs without PyTorch installed.")
            else:
                import torch

                # 获取输入的批次和序列长度
                batch, seqlen = common_inputs["input_ids"].shape
                # 为 past_key_values 指定不同的长度
                past_key_values_length = seqlen + 2
                # 计算头部维度
                head_dim = self._config.hidden_size // self.num_attention_heads
                # 定义过去键和值的形状
                past_key_shape = (
                    batch * self.num_attention_heads,
                    head_dim,
                    past_key_values_length,
                )
                past_value_shape = (
                    batch * self.num_attention_heads,
                    past_key_values_length,
                    head_dim,
                )
                # 为每个层次创建零张量的 past_key_values
                ordered_inputs["past_key_values"] = [
                    (torch.zeros(past_key_shape), torch.zeros(past_value_shape)) for _ in range(self.num_layers)
                ]

        # 添加 attention_mask 到有序输入中
        ordered_inputs["attention_mask"] = common_inputs["attention_mask"]
        
        # 如果使用了 past_keys，则调整 attention_mask 的长度
        if self.use_past:
            mask_dtype = ordered_inputs["attention_mask"].dtype
            ordered_inputs["attention_mask"] = torch.cat(
                [ordered_inputs["attention_mask"], torch.ones(batch, past_key_values_length, dtype=mask_dtype)], dim=1
            )

        # 返回排序后的输入
        return ordered_inputs

    @property
    def default_onnx_opset(self) -> int:
        # 返回默认的 ONNX 操作集版本号
        return 13
```