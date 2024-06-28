# `.\models\codegen\configuration_codegen.py`

```py
# coding=utf-8
# 上面是文件编码声明，指定文件编码格式为UTF-8

# Copyright 2022 Salesforce authors, The EleutherAI, and HuggingFace Teams. All rights reserved.
# 版权声明，指出代码版权归 Salesforce、The EleutherAI 和 HuggingFace Teams 所有，保留所有权利

# Licensed under the Apache License, Version 2.0 (the "License");
# 根据 Apache 许可证版本 2.0 授权，即在符合许可证的情况下可以自由使用该代码
# you may not use this file except in compliance with the License.
# 除非符合许可证要求，否则不得使用本文件

# You may obtain a copy of the License at
# 可以在以下链接获取许可证副本
#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# 除非法律要求或书面同意，否则根据许可证分发的软件是基于“原样”分发的，没有任何明示或暗示的担保或条件

# See the License for the specific language governing permissions and
# limitations under the License.
# 请查看许可证以了解特定语言的权限和限制

""" CodeGen model configuration"""
# 以下是 CodeGen 模型的配置信息

from collections import OrderedDict
# 导入 OrderedDict 类，用于创建有序字典
from typing import Any, List, Mapping, Optional
# 导入类型提示，用于声明函数参数和返回值的类型

from ... import PreTrainedTokenizer, TensorType, is_torch_available
# 导入模块和函数，包括预训练的 Tokenizer、TensorType 和 is_torch_available 函数

from ...configuration_utils import PretrainedConfig
# 从 configuration_utils 模块导入 PretrainedConfig 类，用于配置预训练模型的基本配置

from ...onnx import OnnxConfigWithPast, PatchingSpec
# 从 onnx 模块导入 OnnxConfigWithPast 和 PatchingSpec 类

from ...utils import logging
# 从 utils 模块导入 logging 模块，用于记录日志信息

logger = logging.get_logger(__name__)
# 获取当前模块的 logger 实例，用于记录模型配置相关的日志信息

CODEGEN_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "Salesforce/codegen-350M-nl": "https://huggingface.co/Salesforce/codegen-350M-nl/resolve/main/config.json",
    "Salesforce/codegen-350M-multi": "https://huggingface.co/Salesforce/codegen-350M-multi/resolve/main/config.json",
    "Salesforce/codegen-350M-mono": "https://huggingface.co/Salesforce/codegen-350M-mono/resolve/main/config.json",
    "Salesforce/codegen-2B-nl": "https://huggingface.co/Salesforce/codegen-2B-nl/resolve/main/config.json",
    "Salesforce/codegen-2B-multi": "https://huggingface.co/Salesforce/codegen-2B-multi/resolve/main/config.json",
    "Salesforce/codegen-2B-mono": "https://huggingface.co/Salesforce/codegen-2B-mono/resolve/main/config.json",
    "Salesforce/codegen-6B-nl": "https://huggingface.co/Salesforce/codegen-6B-nl/resolve/main/config.json",
    "Salesforce/codegen-6B-multi": "https://huggingface.co/Salesforce/codegen-6B-multi/resolve/main/config.json",
    "Salesforce/codegen-6B-mono": "https://huggingface.co/Salesforce/codegen-6B-mono/resolve/main/config.json",
    "Salesforce/codegen-16B-nl": "https://huggingface.co/Salesforce/codegen-16B-nl/resolve/main/config.json",
    "Salesforce/codegen-16B-multi": "https://huggingface.co/Salesforce/codegen-16B-multi/resolve/main/config.json",
    "Salesforce/codegen-16B-mono": "https://huggingface.co/Salesforce/codegen-16B-mono/resolve/main/config.json",
}
# 定义 CODEGEN_PRETRAINED_CONFIG_ARCHIVE_MAP，存储不同 CodeGen 模型的预训练配置文件的 URL 映射

class CodeGenConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`CodeGenModel`]. It is used to instantiate a
    CodeGen model according to the specified arguments, defining the model architecture. Instantiating a configuration
    with the defaults will yield a similar configuration to that of the CodeGen
    [Salesforce/codegen-2B-mono](https://huggingface.co/Salesforce/codegen-2B-mono) architecture. Configuration objects
    """
    # CodeGenConfig 类继承自 PretrainedConfig，用于存储 CodeGen 模型的配置信息

    def __init__(
        self,
        **kwargs
    ):
        # 初始化方法，接受任意关键字参数

        super().__init__(**kwargs)
        # 调用父类 PretrainedConfig 的初始化方法
    class CodeGenConfig(PretrainedConfig):
        # 继承自 `PretrainedConfig`，用于控制模型输出。更多信息请参考 `PretrainedConfig` 的文档。
        def __init__(
            self,
            vocab_size=50400,
            n_positions=2048,
            n_ctx=2048,
            n_embd=4096,
            n_layer=28,
            n_head=16,
            rotary_dim=64,
            n_inner=None,
            activation_function="gelu_new",
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
            layer_norm_epsilon=1e-05,
            initializer_range=0.02,
            use_cache=True,
            bos_token_id=50256,
            eos_token_id=50256,
            tie_word_embeddings=False,
        ):
            # 初始化方法，设定模型的各种配置参数
            self.vocab_size = vocab_size
            self.n_positions = n_positions
            self.n_ctx = n_ctx
            self.n_embd = n_embd
            self.n_layer = n_layer
            self.n_head = n_head
            self.rotary_dim = rotary_dim
            self.n_inner = n_inner if n_inner is not None else 4 * n_embd
            self.activation_function = activation_function
            self.resid_pdrop = resid_pdrop
            self.embd_pdrop = embd_pdrop
            self.attn_pdrop = attn_pdrop
            self.layer_norm_epsilon = layer_norm_epsilon
            self.initializer_range = initializer_range
            self.use_cache = use_cache
            self.bos_token_id = bos_token_id
            self.eos_token_id = eos_token_id
            self.tie_word_embeddings = tie_word_embeddings
    # 导入 transformers 库中的 CodeGenConfig 和 CodeGenModel 类
    >>> from transformers import CodeGenConfig, CodeGenModel

    # 初始化一个 CodeGenConfig 对象，用于配置 CodeGen 模型的参数
    >>> configuration = CodeGenConfig()

    # 使用上述配置初始化一个 CodeGenModel 对象，模型参数采用随机初始化
    >>> model = CodeGenModel(configuration)

    # 获取模型的配置信息并存储在 configuration 变量中
    >>> configuration = model.config
    ```
# 从 transformers.models.gpt2.configuration_gpt2.GPT2OnnxConfig 复制而来的类，继承自 OnnxConfigWithPast
class CodeGenOnnxConfig(OnnxConfigWithPast):
    
    # 初始化方法，接受一个预训练配置对象 config 和其他可选参数
    def __init__(
        self,
        config: PretrainedConfig,
        task: str = "default",
        patching_specs: List[PatchingSpec] = None,
        use_past: bool = False,
    ):
        # 调用父类 OnnxConfigWithPast 的初始化方法
        super().__init__(config, task=task, patching_specs=patching_specs, use_past=use_past)
        
        # 如果配置对象中不存在 pad_token_id 属性，则设置其为 0
        if not getattr(self._config, "pad_token_id", None):
            # TODO: how to do that better?
            self._config.pad_token_id = 0

    # 输入属性，返回一个字典，描述模型的输入结构
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 共通的输入格式，包含 input_ids 和 attention_mask
        common_inputs = OrderedDict({"input_ids": {0: "batch", 1: "sequence"}})
        
        # 如果使用过去信息，则填充 past 相关的输入格式
        if self.use_past:
            self.fill_with_past_key_values_(common_inputs, direction="inputs")
            common_inputs["attention_mask"] = {0: "batch", 1: "past_sequence + sequence"}
        else:
            # 否则只保留当前序列的 attention_mask
            common_inputs["attention_mask"] = {0: "batch", 1: "sequence"}

        return common_inputs

    # 返回模型的层数，从配置对象的 n_layer 属性获取
    @property
    def num_layers(self) -> int:
        return self._config.n_layer

    # 返回模型的注意力头数，从配置对象的 n_head 属性获取
    @property
    def num_attention_heads(self) -> int:
        return self._config.n_head

    # 生成虚拟输入数据的方法，接受 tokenizer 和其他参数
    def generate_dummy_inputs(
        self,
        tokenizer: PreTrainedTokenizer,
        batch_size: int = -1,
        seq_length: int = -1,
        is_pair: bool = False,
        framework: Optional[TensorType] = None,
        # （此处省略部分方法内容）
        ) -> Mapping[str, Any]:
        # 调用父类方法生成通用的输入字典，包括输入的文本编码、批处理大小、序列长度、是否为成对数据和框架类型
        common_inputs = super(OnnxConfigWithPast, self).generate_dummy_inputs(
            tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework
        )

        # 按照 forward() 方法中的顺序排列输入
        ordered_inputs = OrderedDict({"input_ids": common_inputs["input_ids"]})

        # 如果需要使用过去的键值（past_keys）
        if self.use_past:
            # 如果没有安装 PyTorch，则抛出数值错误
            if not is_torch_available():
                raise ValueError("Cannot generate dummy past_keys inputs without PyTorch installed.")
            else:
                import torch

                # 获取输入数据的批次大小和序列长度
                batch, seqlen = common_inputs["input_ids"].shape
                # 计算过去键值的长度，增加2以保证足够的容量
                past_key_values_length = seqlen + 2
                # 定义过去键值的形状
                past_shape = (
                    batch,
                    self.num_attention_heads,
                    past_key_values_length,
                    self._config.hidden_size // self.num_attention_heads,
                )
                # 为每一层的每个位置创建零张量，形成 past_key_values 列表
                ordered_inputs["past_key_values"] = [
                    (torch.zeros(past_shape), torch.zeros(past_shape)) for _ in range(self.num_layers)
                ]

        # 将通用的注意力掩码添加到有序输入中
        ordered_inputs["attention_mask"] = common_inputs["attention_mask"]

        # 如果需要使用过去的键值（past_keys）
        if self.use_past:
            # 获取注意力掩码的数据类型
            mask_dtype = ordered_inputs["attention_mask"].dtype
            # 将形状相同的全1张量连接到现有的注意力掩码张量后面，以扩展其长度
            ordered_inputs["attention_mask"] = torch.cat(
                [ordered_inputs["attention_mask"], torch.ones(batch, past_key_values_length, dtype=mask_dtype)], dim=1
            )

        # 返回最终的有序输入字典
        return ordered_inputs

    @property
    def default_onnx_opset(self) -> int:
        # 返回默认的 ONNX 操作集版本号
        return 13
```