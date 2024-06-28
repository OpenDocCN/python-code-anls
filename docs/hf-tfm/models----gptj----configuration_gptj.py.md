# `.\models\gptj\configuration_gptj.py`

```
# coding=utf-8
# Copyright 2021 The EleutherAI and HuggingFace Teams. All rights reserved.
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

""" GPT-J model configuration"""

# 引入 OrderedDict 类用于创建有序字典，以及其他必要的类型和模块
from collections import OrderedDict
from typing import Any, List, Mapping, Optional

# 引入 Hugging Face 库中的一些模块和函数
from ... import PreTrainedTokenizer, TensorType, is_torch_available
from ...configuration_utils import PretrainedConfig
from ...onnx import OnnxConfigWithPast, PatchingSpec
from ...utils import logging

# 获取日志记录器对象，用于记录和输出日志信息
logger = logging.get_logger(__name__)

# 定义 GPT-J 预训练模型配置文件的存档映射字典
GPTJ_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "EleutherAI/gpt-j-6B": "https://huggingface.co/EleutherAI/gpt-j-6B/resolve/main/config.json",
    # 可在 https://huggingface.co/models?filter=gpt_j 查看所有 GPT-J 模型
}

# 定义 GPTJConfig 类，用于存储 GPT-J 模型的配置信息
class GPTJConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`GPTJModel`]. It is used to instantiate a GPT-J
    model according to the specified arguments, defining the model architecture. Instantiating a configuration with the
    defaults will yield a similar configuration to that of the GPT-J
    [EleutherAI/gpt-j-6B](https://huggingface.co/EleutherAI/gpt-j-6B) architecture. Configuration objects inherit from
    [`PretrainedConfig`] and can be used to control the model outputs. Read the documentation from [`PretrainedConfig`]
    for more information.
    """
    # 定义模型类型为 GPT-J
    model_type = "gptj"
    
    # 创建属性映射字典，将配置参数名映射到对应的 GPT-J 模型配置参数名
    attribute_map = {
        "max_position_embeddings": "n_positions",     # 最大序列长度映射到 n_positions
        "hidden_size": "n_embd",                      # 隐藏大小映射到 n_embd
        "num_attention_heads": "n_head",              # 注意力头数映射到 n_head
        "num_hidden_layers": "n_layer",               # 隐藏层数映射到 n_layer
    }
    # 定义一个初始化函数，用于初始化一个Transformer模型的参数和设置
    def __init__(
        self,
        vocab_size=50400,                        # 词汇表大小，默认为50400
        n_positions=2048,                        # 最大位置编码数，默认为2048
        n_embd=4096,                             # 嵌入层维度，默认为4096
        n_layer=28,                              # Transformer层数，默认为28层
        n_head=16,                               # 自注意力机制中头数，默认为16
        rotary_dim=64,                           # 旋转注意力机制的维度，默认为64
        n_inner=None,                            # Transformer内部层的维度，默认为None
        activation_function="gelu_new",          # 激活函数类型，默认为"gelu_new"
        resid_pdrop=0.0,                          # 残差连接的dropout概率，默认为0.0
        embd_pdrop=0.0,                           # 嵌入层的dropout概率，默认为0.0
        attn_pdrop=0.0,                           # 注意力层的dropout概率，默认为0.0
        layer_norm_epsilon=1e-5,                  # Layer Norm层的epsilon，默认为1e-5
        initializer_range=0.02,                   # 参数初始化的范围，默认为0.02
        use_cache=True,                           # 是否使用缓存，默认为True
        bos_token_id=50256,                       # 开始词的token id，默认为50256
        eos_token_id=50256,                       # 结束词的token id，默认为50256
        tie_word_embeddings=False,                # 是否绑定词嵌入，默认为False
        **kwargs,                                 # 其他关键字参数
    ):
        self.vocab_size = vocab_size               # 初始化词汇表大小属性
        self.n_positions = n_positions             # 初始化最大位置编码数属性
        self.n_embd = n_embd                       # 初始化嵌入层维度属性
        self.n_layer = n_layer                     # 初始化Transformer层数属性
        self.n_head = n_head                       # 初始化自注意力机制头数属性
        self.n_inner = n_inner                     # 初始化Transformer内部层维度属性
        self.rotary_dim = rotary_dim               # 初始化旋转注意力机制维度属性
        self.activation_function = activation_function  # 初始化激活函数类型属性
        self.resid_pdrop = resid_pdrop             # 初始化残差连接的dropout概率属性
        self.embd_pdrop = embd_pdrop               # 初始化嵌入层的dropout概率属性
        self.attn_pdrop = attn_pdrop               # 初始化注意力层的dropout概率属性
        self.layer_norm_epsilon = layer_norm_epsilon  # 初始化Layer Norm层的epsilon属性
        self.initializer_range = initializer_range  # 初始化参数初始化范围属性
        self.use_cache = use_cache                 # 初始化是否使用缓存属性
    
        self.bos_token_id = bos_token_id           # 初始化开始词的token id属性
        self.eos_token_id = eos_token_id           # 初始化结束词的token id属性
    
        # 调用父类的初始化方法，传递开始词的token id、结束词的token id和是否绑定词嵌入的参数
        super().__init__(
            bos_token_id=bos_token_id, eos_token_id=eos_token_id, tie_word_embeddings=tie_word_embeddings, **kwargs
        )
# 从transformers.models.gpt2.configuration_gpt2.GPT2OnnxConfig复制而来的配置类GPTJOnnxConfig，
# 继承自OnnxConfigWithPast。
class GPTJOnnxConfig(OnnxConfigWithPast):
    
    # 初始化方法，接受以下参数：
    # - config: 预训练配置对象
    # - task: 任务名称，默认为"default"
    # - patching_specs: 补丁规格列表，可选参数，默认为None
    # - use_past: 是否使用过去键值，布尔类型，默认为False
    def __init__(
        self,
        config: PretrainedConfig,
        task: str = "default",
        patching_specs: List[PatchingSpec] = None,
        use_past: bool = False,
    ):
        # 调用父类的初始化方法，传递配置对象、任务名称、补丁规格列表和是否使用过去键值
        super().__init__(config, task=task, patching_specs=patching_specs, use_past=use_past)
        
        # 如果配置对象的pad_token_id属性不存在
        if not getattr(self._config, "pad_token_id", None):
            # 设置pad_token_id为0（默认值）
            self._config.pad_token_id = 0

    # 输入属性，返回一个字典，表示常见的输入格式
    @property
    def inputs(self) -> Mapping[str, Mapping[int, str]]:
        # 创建一个有序字典，包含输入ids的批次和序列索引
        common_inputs = OrderedDict({"input_ids": {0: "batch", 1: "sequence"}})
        
        # 如果使用过去键值
        if self.use_past:
            # 填充输入字典，包括过去键值的方向
            self.fill_with_past_key_values_(common_inputs, direction="inputs")
            # 添加注意力遮罩，考虑过去序列和当前序列
            common_inputs["attention_mask"] = {0: "batch", 1: "past_sequence + sequence"}
        else:
            # 添加默认的注意力遮罩，仅考虑当前序列
            common_inputs["attention_mask"] = {0: "batch", 1: "sequence"}

        # 返回构建好的输入字典
        return common_inputs

    # 层数属性，返回配置对象的层数
    @property
    def num_layers(self) -> int:
        return self._config.n_layer

    # 注意力头数属性，返回配置对象的注意力头数
    @property
    def num_attention_heads(self) -> int:
        return self._config.n_head

    # 生成虚拟输入方法，接受以下参数：
    # - tokenizer: 预训练分词器对象
    # - batch_size: 批次大小，整数，默认为-1
    # - seq_length: 序列长度，整数，默认为-1
    # - is_pair: 是否是成对输入，布尔类型，默认为False
    # - framework: 框架类型，可选参数，默认为None
    ) -> Mapping[str, Any]:
        # 调用父类方法生成通用的虚拟输入数据
        common_inputs = super(OnnxConfigWithPast, self).generate_dummy_inputs(
            tokenizer, batch_size=batch_size, seq_length=seq_length, is_pair=is_pair, framework=framework
        )

        # 根据模型前向方法的输入顺序，重新排序输入数据
        ordered_inputs = OrderedDict({"input_ids": common_inputs["input_ids"]})

        # 如果需要使用过去的键值（past_keys）
        if self.use_past:
            # 检查是否有安装 PyTorch，否则抛出错误
            if not is_torch_available():
                raise ValueError("Cannot generate dummy past_keys inputs without PyTorch installed.")
            else:
                import torch

                batch, seqlen = common_inputs["input_ids"].shape
                # 计算过去键值的长度，比序列长度多两个
                past_key_values_length = seqlen + 2
                past_shape = (
                    batch,
                    self.num_attention_heads,
                    past_key_values_length,
                    self._config.hidden_size // self.num_attention_heads,
                )
                # 为每个层生成空的过去键值对
                ordered_inputs["past_key_values"] = [
                    (torch.zeros(past_shape), torch.zeros(past_shape)) for _ in range(self.num_layers)
                ]

        # 将通用的注意力掩码添加到排序后的输入中
        ordered_inputs["attention_mask"] = common_inputs["attention_mask"]

        # 如果需要使用过去的键值（past_keys）
        if self.use_past:
            # 获取掩码的数据类型并为过去的键值对添加新的掩码
            mask_dtype = ordered_inputs["attention_mask"].dtype
            ordered_inputs["attention_mask"] = torch.cat(
                [ordered_inputs["attention_mask"], torch.ones(batch, past_key_values_length, dtype=mask_dtype)], dim=1
            )

        # 返回最终排序后的输入字典
        return ordered_inputs

    @property
    def default_onnx_opset(self) -> int:
        # 返回默认的 ONNX 操作集版本号
        return 13
```