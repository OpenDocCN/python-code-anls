# `.\models\vipllava\modeling_vipllava.py`

```
# coding=utf-8
# Copyright 2023 the HuggingFace Inc. team. All rights reserved.
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
""" PyTorch VipLlava model."""

# 导入必要的库和模块
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

# 导入 Hugging Face Transformers 中的预训练模型基类和其他必要组件
from ... import PreTrainedModel
from ...activations import ACT2FN
from ...cache_utils import Cache
from ...modeling_outputs import ModelOutput
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from ..auto import AutoModel, AutoModelForCausalLM

# 导入 VipLlava 模型的配置类
from .configuration_vipllava import VipLlavaConfig

# 获取 logger 对象用于日志记录
logger = logging.get_logger(__name__)

# 文档中显示的配置对象名称
_CONFIG_FOR_DOC = "VipLlavaConfig"

# 预训练模型的存档列表，包括一个示例
VIPLLAVA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "llava-hf/vip-llava-7b-hf",
    # See all VipLlava models at https://huggingface.co/models?filter=vipllava
]

# 定义一个数据类，用于表示 VipLlava 模型的自回归语言模型输出及过去状态
@dataclass
# 从 Idefics 模型中复制的类，作为 VipLlava 模型的输出基类
class VipLlavaCausalLMOutputWithPast(ModelOutput):
    """
    Base class for VipLlava causal language model (or autoregressive) outputs.
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of shape
            `(batch_size, num_heads, sequence_length, embed_size_per_head)`)

            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        image_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            Tuple of `torch.FloatTensor` (one for the output of the image embeddings, `(batch_size, num_images,
            sequence_length, hidden_size)`.

            image_hidden_states of the model produced by the vision encoder, and optionally by the perceiver
    """

    # Optional loss value for language modeling
    loss: Optional[torch.FloatTensor] = None
    # Predicted logits for each token in the batch
    logits: torch.FloatTensor = None
    # Cached key and value states for speeding up sequential decoding
    past_key_values: Optional[List[torch.FloatTensor]] = None
    # Hidden states of the model at each layer's output and optional initial embeddings
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # Attention weights after softmax, used for self-attention computation
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    # Hidden states produced by the vision encoder for image embeddings
    image_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
@add_start_docstrings(
    "The bare VipLlava Model outputting raw hidden-states without any specific head on top.",
    VIPLLAVA_START_DOCSTRING,
)
# 为 VipLlavaPreTrainedModel 类添加文档字符串，描述其作为 VipLlava 模型的基础预训练模型的输出为原始隐藏状态，没有特定的输出头部。

# 从 PreTrainedModel 类继承，定义 VipLlavaPreTrainedModel 类
class VipLlavaPreTrainedModel(PreTrainedModel):
    # 指定配置类为 VipLlavaConfig
    config_class = VipLlavaConfig
    # 模型的基础名称前缀
    base_model_prefix = "model"
    # 支持梯度检查点
    supports_gradient_checkpointing = True
    # 不分割的模块列表
    _no_split_modules = ["VipLlavaVisionAttention"]
    # 跳过键设备放置
    _skip_keys_device_placement = "past_key_values"
    # 支持 Flash Attention 2
    _supports_flash_attn_2 = True
    # 初始化模型权重的方法，用于对传入的模块进行权重初始化
    def _init_weights(self, module):
        # 注意: 这个迁移版本的 VipLlava 不适用于从头训练，只能用于推理和微调。
        # 因此，适当的初始化权重代码已经被移除。原始代码库位于 https://github.com/haotian-liu/LLaVA/tree/main/vipllava，可以用于训练目的。

        # 根据配置获取初始化标准差
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        # 如果模块具有类嵌入（class_embedding）属性，则对其进行标准正态分布初始化
        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        # 如果模块是线性层（nn.Linear）或二维卷积层（nn.Conv2d），则对权重进行标准正态分布初始化，
        # 如果有偏置，则将偏置初始化为零
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        
        # 如果模块是嵌入层（nn.Embedding），则对权重进行标准正态分布初始化，
        # 如果定义了填充索引（padding_idx），则将该索引处的权重初始化为零
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def _supports_sdpa(self):
        """
        检索语言模型的属性，检查模型是否支持 SDPA（Self-Attention with Dual Paths）。
        """
        return self.language_model._supports_sdpa
# 定义模型文档字符串，用于描述 VIPLLAVA 模型的输入
VIPLLAVA_INPUTS_DOCSTRING = r"""
"""


@add_start_docstrings(
    """The VIPLLAVA model which consists of a vision backbone and a language model.""",
    VIPLLAVA_START_DOCSTRING,
)
# 从 transformers.models.llava.modeling_llava.LlavaForConditionalGeneration 复制而来，将 LLAVA 改为 VIPLLAVA，Llava 改为 VipLlava
class VipLlavaForConditionalGeneration(VipLlavaPreTrainedModel):
    def __init__(self, config: VipLlavaConfig):
        super().__init__(config)
        # 初始化视觉塔模型，使用从配置中获取的视觉配置
        self.vision_tower = AutoModel.from_config(config.vision_config)

        # 初始化多模态投影器
        self.multi_modal_projector = VipLlavaMultiModalProjector(config)
        # 获取文本配置中的词汇表大小作为模型的词汇表大小
        self.vocab_size = config.text_config.vocab_size
        # 初始化语言模型，使用从配置中获取的文本配置和注意力实现方式
        self.language_model = AutoModelForCausalLM.from_config(
            config.text_config, attn_implementation=config._attn_implementation
        )
        # 如果配置中定义了 pad_token_id，则使用配置中的值；否则使用 -1
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        # 执行初始化后的处理
        self.post_init()

    def get_input_embeddings(self):
        # 获取语言模型的输入嵌入层
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        # 设置语言模型的输入嵌入层
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        # 获取语言模型的输出嵌入层
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        # 设置语言模型的输出嵌入层
        self.language_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        # 设置语言模型的解码器
        self.language_model.set_decoder(decoder)

    def get_decoder(self):
        # 获取语言模型的解码器
        return self.language_model.get_decoder()

    def tie_weights(self):
        # 绑定语言模型的权重
        return self.language_model.tie_weights()

    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None) -> nn.Embedding:
        # 调整语言模型的 token 嵌入层大小，并更新模型配置中的词汇表大小
        model_embeds = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds

    @add_start_docstrings_to_model_forward(VIPLLAVA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=VipLlavaCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    # 忽略复制
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layers: Optional[List[int]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        pass  # 在正式实现前，暂时占位，不执行任何操作

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, pixel_values=None, attention_mask=None, **kwargs
    ):
        pass  # 在正式实现前，暂时占位，不执行任何操作
    ):
        # 如果传入的过去键值不为 None，则处理缓存相关逻辑
        if past_key_values is not None:
            # 如果过去键值是 Cache 类型，则获取序列长度和已见标记数
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
            else:
                # 否则，假设过去键值的第一个元素的第一个维度是 token 的形状的长度
                cache_length = past_length = past_key_values[0][0].shape[2]

            # 保留未处理的 token：
            # 1 - 如果 attention_mask 的长度超过 input_ids 的长度，则处理仅作为缓存传递的情况
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - 如果 past_length 小于 input_ids 的长度，则 input_ids 包含所有输入 token，可以基于 past_length 丢弃 input_ids
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - 否则（past_length >= input_ids.shape[1]），假设 input_ids 只有未处理的 token
            elif self.config.image_token_index in input_ids:
                input_ids = input_ids[:, input_ids.shape[1] - 1 :]
            
            # 如果缓存已见 token 数超过其容量限制，那么缓存有一个大小限制。丢弃较早的 attention 值，因为它们对应的值不是输入的一部分。
            if cache_length < past_length and attention_mask is not None:
                attention_mask = attention_mask[:, -(cache_length + input_ids.shape[1]) :]

        position_ids = kwargs.get("position_ids", None)
        # 如果 attention_mask 不为 None 且 position_ids 为 None，则在批量生成时动态创建 position_ids
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # 如果传入 inputs_embeds，则仅在第一代步骤中使用它们
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # 更新 model_inputs 字典
        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
            }
        )
        return model_inputs

    # 重排序缓存的内部方法委托给语言模型的 _reorder_cache 方法
    def _reorder_cache(self, *args, **kwargs):
        return self.language_model._reorder_cache(*args, **kwargs)
```