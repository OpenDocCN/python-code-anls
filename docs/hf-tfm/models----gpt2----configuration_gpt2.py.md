# `.\models\gpt2\configuration_gpt2.py`

```
# coding=utf-8
# 设置文件编码为 UTF-8

# Copyright 2018 The OpenAI Team Authors and HuggingFace Inc. team.
# Copyright 2018, NVIDIA CORPORATION.  All rights reserved.
# 版权声明，版权归 OpenAI Team Authors 和 HuggingFace Inc. team 以及 NVIDIA CORPORATION 所有

# Licensed under the Apache License, Version 2.0 (the "License");
# 根据 Apache License, Version 2.0 授权，可以在符合许可证条件下使用本文件

# you may not use this file except in compliance with the License.
# 除非符合许可证条件，否则不得使用此文件

# You may obtain a copy of the License at
# 可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 除非适用法律要求或书面同意，否则本软件按"原样"分发，不附带任何明示或暗示的担保或条件

# See the License for the specific language governing permissions and
# limitations under the License.
# 查阅许可证了解详细的权限和限制

""" OpenAI GPT-2 configuration"""
# 引入 OpenAI GPT-2 配置模块

from collections import OrderedDict
# 导入 OrderedDict 数据结构，用于有序字典操作
from typing import Any, List, Mapping, Optional
# 导入类型提示相关的模块

from ... import PreTrainedTokenizer, TensorType, is_torch_available
# 导入其他模块或函数

from ...configuration_utils import PretrainedConfig
# 从 transformers 的配置工具模块中导入 PretrainedConfig 类
from ...onnx import OnnxConfigWithPast, PatchingSpec
# 从 transformers 的 ONNX 模块中导入特定配置
from ...utils import logging
# 导入 logging 工具模块

logger = logging.get_logger(__name__)
# 获取当前模块的日志记录器对象

GPT2_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "openai-community/gpt2": "https://huggingface.co/openai-community/gpt2/resolve/main/config.json",
    "openai-community/gpt2-medium": "https://huggingface.co/openai-community/gpt2-medium/resolve/main/config.json",
    "openai-community/gpt2-large": "https://huggingface.co/openai-community/gpt2-large/resolve/main/config.json",
    "openai-community/gpt2-xl": "https://huggingface.co/openai-community/gpt2-xl/resolve/main/config.json",
    "distilbert/distilgpt2": "https://huggingface.co/distilbert/distilgpt2/resolve/main/config.json",
}
# 定义 GPT-2 预训练模型配置文件的映射字典，包含不同模型的名称和其配置文件的 URL

class GPT2Config(PretrainedConfig):
    """
    This is the configuration class to store the configuration of a [`GPT2Model`] or a [`TFGPT2Model`]. It is used to
    instantiate a GPT-2 model according to the specified arguments, defining the model architecture. Instantiating a
    configuration with the defaults will yield a similar configuration to that of the GPT-2
    [openai-community/gpt2](https://huggingface.co/openai-community/gpt2) architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.


    Example:

    ```python
    >>> from transformers import GPT2Config, GPT2Model

    >>> # Initializing a GPT2 configuration
    >>> configuration = GPT2Config()

    >>> # Initializing a model (with random weights) from the configuration
    >>> model = GPT2Model(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```
    """
    # GPT2Config 类，用于存储 [`GPT2Model`] 或 [`TFGPT2Model`] 的配置信息。根据指定的参数实例化 GPT-2 模型，定义模型架构。
    # 使用默认配置实例化将得到与 GPT-2 [openai-community/gpt2](https://huggingface.co/openai-community/gpt2) 架构类似的配置。

    model_type = "gpt2"
    # 模型类型为 "gpt2"
    keys_to_ignore_at_inference = ["past_key_values"]
    # 推断时忽略的键名列表，在推断过程中不使用 "past_key_values" 键名
    attribute_map = {
        "hidden_size": "n_embd",
        "max_position_embeddings": "n_positions",
        "num_attention_heads": "n_head",
        "num_hidden_layers": "n_layer",
    }
    # 属性映射，将模型配置中的一些属性名称映射到新的名称，例如 "hidden_size" 映射到 "n_embd"
    # 初始化函数，用于设置 Transformer 模型的各种参数和配置
    def __init__(
        self,
        vocab_size=50257,  # 词汇表大小，默认为50257
        n_positions=1024,  # 序列长度，默认为1024
        n_embd=768,  # Embedding 向量的维度，默认为768
        n_layer=12,  # Transformer 层的数量，默认为12
        n_head=12,  # 多头注意力机制的头数，默认为12
        n_inner=None,  # Feedforward 层中间层的维度，可选参数，默认为None
        activation_function="gelu_new",  # 激活函数类型，默认为 gelu_new
        resid_pdrop=0.1,  # 残差连接中的 dropout 概率，默认为0.1
        embd_pdrop=0.1,  # Embedding 层的 dropout 概率，默认为0.1
        attn_pdrop=0.1,  # 注意力层的 dropout 概率，默认为0.1
        layer_norm_epsilon=1e-5,  # Layer normalization 中 epsilon 的值，默认为1e-5
        initializer_range=0.02,  # 参数初始化范围，默认为0.02
        summary_type="cls_index",  # 摘要生成的类型，默认为 cls_index
        summary_use_proj=True,  # 是否使用投影层来生成摘要，默认为True
        summary_activation=None,  # 摘要生成时的激活函数，默认为None
        summary_proj_to_labels=True,  # 是否将摘要投影到标签上，默认为True
        summary_first_dropout=0.1,  # 摘要生成时的第一层 dropout 概率，默认为0.1
        scale_attn_weights=True,  # 是否对注意力权重进行缩放，默认为True
        use_cache=True,  # 是否使用缓存，默认为True
        bos_token_id=50256,  # 起始 token 的 id，默认为50256
        eos_token_id=50256,  # 结束 token 的 id，默认为50256
        scale_attn_by_inverse_layer_idx=False,  # 是否按照反向层索引对注意力权重进行缩放，默认为False
        reorder_and_upcast_attn=False,  # 是否重新排序和升级注意力，默认为False
        **kwargs,  # 其他未指定的参数，使用关键字传递
    ):
        self.vocab_size = vocab_size  # 初始化词汇表大小
        self.n_positions = n_positions  # 初始化序列长度
        self.n_embd = n_embd  # 初始化 Embedding 向量维度
        self.n_layer = n_layer  # 初始化 Transformer 层的数量
        self.n_head = n_head  # 初始化注意力头数
        self.n_inner = n_inner  # 初始化 Feedforward 层中间层维度
        self.activation_function = activation_function  # 初始化激活函数类型
        self.resid_pdrop = resid_pdrop  # 初始化残差连接中的 dropout 概率
        self.embd_pdrop = embd_pdrop  # 初始化 Embedding 层的 dropout 概率
        self.attn_pdrop = attn_pdrop  # 初始化注意力层的 dropout 概率
        self.layer_norm_epsilon = layer_norm_epsilon  # 初始化 Layer normalization 中 epsilon 的值
        self.initializer_range = initializer_range  # 初始化参数初始化范围
        self.summary_type = summary_type  # 初始化摘要生成的类型
        self.summary_use_proj = summary_use_proj  # 初始化是否使用投影层来生成摘要
        self.summary_activation = summary_activation  # 初始化摘要生成时的激活函数
        self.summary_first_dropout = summary_first_dropout  # 初始化摘要生成时的第一层 dropout 概率
        self.summary_proj_to_labels = summary_proj_to_labels  # 初始化是否将摘要投影到标签上
        self.scale_attn_weights = scale_attn_weights  # 初始化是否对注意力权重进行缩放
        self.use_cache = use_cache  # 初始化是否使用缓存
        self.scale_attn_by_inverse_layer_idx = scale_attn_by_inverse_layer_idx  # 初始化是否按照反向层索引对注意力权重进行缩放
        self.reorder_and_upcast_attn = reorder_and_upcast_attn  # 初始化是否重新排序和升级注意力

        self.bos_token_id = bos_token_id  # 初始化起始 token 的 id
        self.eos_token_id = eos_token_id  # 初始化结束 token 的 id

        super().__init__(bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)  # 调用父类的初始化函数，传入起始和结束 token 的 id，以及其他未指定的参数
    # 定义一个继承自OnnxConfigWithPast的配置类，用于GPT-2模型的ONNX导出配置
    class GPT2OnnxConfig(OnnxConfigWithPast):
        # 初始化方法，接收预训练配置、任务名称、补丁规范列表和是否使用过去信息的参数
        def __init__(
            self,
            config: PretrainedConfig,
            task: str = "default",
            patching_specs: List[PatchingSpec] = None,
            use_past: bool = False,
        ):
            # 调用父类的初始化方法，传递预训练配置、任务名称、补丁规范列表和是否使用过去信息的参数
            super().__init__(config, task=task, patching_specs=patching_specs, use_past=use_past)
            # 如果配置中没有定义pad_token_id属性，则设置其默认值为0
            if not getattr(self._config, "pad_token_id", None):
                # TODO: 如何更好地处理这一情况？
                self._config.pad_token_id = 0

        # 返回输入的属性，是一个映射结构，描述了输入数据的格式
        @property
        def inputs(self) -> Mapping[str, Mapping[int, str]]:
            # 创建一个有序字典，定义常见的输入结构，包含input_ids和attention_mask
            common_inputs = OrderedDict({"input_ids": {0: "batch", 1: "sequence"}})
            # 如果使用过去信息，则在输入结构中加入past_key_values相关的描述
            if self.use_past:
                self.fill_with_past_key_values_(common_inputs, direction="inputs")
                common_inputs["attention_mask"] = {0: "batch", 1: "past_sequence + sequence"}
            else:
                common_inputs["attention_mask"] = {0: "batch", 1: "sequence"}

            return common_inputs

        # 返回层数的属性，即预训练配置中的n_layer值
        @property
        def num_layers(self) -> int:
            return self._config.n_layer

        # 返回注意力头数的属性，即预训练配置中的n_head值
        @property
        def num_attention_heads(self) -> int:
            return self._config.n_head

        # 生成虚拟输入数据的方法，用于模型推理的测试和调试
        def generate_dummy_inputs(
            self,
            tokenizer: PreTrainedTokenizer,
            batch_size: int = -1,
            seq_length: int = -1,
            is_pair: bool = False,
            framework: Optional[TensorType] = None,
        ) -> Mapping[str, Any]:
            # 调用父类的generate_dummy_inputs方法，生成通用的输入数据
            common_inputs = super(OnnxConfigWithPast, self).generate_dummy_inputs(
                tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework
            )

            # 按照forward()方法中的顺序排序输入数据
            ordered_inputs = OrderedDict({"input_ids": common_inputs["input_ids"]})

            # 如果使用过去信息，生成past_key_values的虚拟输入数据
            if self.use_past:
                if not is_torch_available():
                    # 如果没有安装PyTorch，则抛出数值错误
                    raise ValueError("Cannot generate dummy past_keys inputs without PyTorch installed.")
                else:
                    import torch

                    batch, seqlen = common_inputs["input_ids"].shape
                    # 计算past_key_values的长度，略长于当前序列长度
                    past_key_values_length = seqlen + 2
                    past_shape = (
                        batch,
                        self.num_attention_heads,
                        past_key_values_length,
                        self._config.hidden_size // self.num_attention_heads,
                    )
                    # 为每一层生成零张量作为past_key_values
                    ordered_inputs["past_key_values"] = [
                        (torch.zeros(past_shape), torch.zeros(past_shape)) for _ in range(self.num_layers)
                    ]

            # 将attention_mask加入有序的输入数据中
            ordered_inputs["attention_mask"] = common_inputs["attention_mask"]
            # 如果使用过去信息，调整attention_mask的形状以匹配past_key_values的长度
            if self.use_past:
                mask_dtype = ordered_inputs["attention_mask"].dtype
                ordered_inputs["attention_mask"] = torch.cat(
                    [ordered_inputs["attention_mask"], torch.ones(batch, past_key_values_length, dtype=mask_dtype)], dim=1
                )

            return ordered_inputs

        # 返回层数的属性，即预训练配置中的n_layer值
        @property
    # 定义一个方法，用于返回默认的 ONNX 操作集版本号，返回整数 13
    def default_onnx_opset(self) -> int:
        return 13
```