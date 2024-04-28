# `.\transformers\models\vipllava\modeling_vipllava.py`

```
# coding=utf-8
# 版本信息和版权声明
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

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
from .configuration_vipllava import VipLlavaConfig

logger = logging.get_logger(__name__)

# VipLlava配置的文档字符串
_CONFIG_FOR_DOC = "VipLlavaConfig"

# VipLlava预训练模型的存档列表
VIPLLAVA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "llava-hf/vip-llava-7b-hf",
    # 查看所有VipLlava模型：https://huggingface.co/models?filter=vipllava
]

@dataclass
# 复制自transformers.models.idefics.modeling_idefics.IdeficsCausalLMOutputWithPast，将Idefics改为VipLlava
class VipLlavaCausalLMOutputWithPast(ModelOutput):
    """
    VipLlava因果语言模型（或自回归）输出的基类.
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

    # 损失值，对应语言建模任务中的损失值（用于下一个标记的预测）
    loss: Optional[torch.FloatTensor] = None
    # 语言建模头部的预测得分（SoftMax 之前每个词汇标记的预测分数）
    logits: torch.FloatTensor = None
    # 预先计算的隐藏状态（自注意力块中的键和值），可用于加速序列解码
    past_key_values: Optional[List[torch.FloatTensor]] = None
    # 模型各层输出的隐藏状态（如果模型有嵌入层，则包括嵌入输出和每一层的输出）
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 自注意力头部中注意力 softmax 后的注意力权重，用于计算自注意力头部的加权平均值
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 模型生成的图像嵌入的隐藏状态（由视觉编码器产生，并可由感知器可选地产生）
    image_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
# 定义一个 VipLlavaMultiModalProjector 类，继承自 nn.Module
class VipLlavaMultiModalProjector(nn.Module):
    # 初始化方法，接受一个 VipLlavaConfig 类的参数
    def __init__(self, config: VipLlavaConfig):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个 LayerNorm 层，用于归一化输入特征
        self.projector_layernorm = nn.LayerNorm(len(config.vision_feature_layers) * config.vision_config.hidden_size, eps=config.projector_layernorm_eps)

        # 创建一个线性层，进行特征映射
        self.linear_1 = nn.Linear(len(config.vision_feature_layers) * config.vision_config.hidden_size, config.text_config.hidden_size, bias=True)
        # 获取激活函数
        self.act = ACT2FN[config.projector_hidden_act]
        # 创建第二个线性层
        self.linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=True)

    # 前向传播方法
    def forward(self, hidden_states):
        # 对输入进行归一化
        hidden_states = self.projector_layernorm(hidden_states)
        # 第一个线性映射
        hidden_states = self.linear_1(hidden_states)
        # 激活函数
        hidden_states = self.act(hidden_states)
        # 第二个线性映射
        hidden_states = self.linear_2(hidden_states)
        # 返回最终输出
        return hidden_states


# 定义一个常量字符串，用于文档字符串
VIPLLAVA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`VipLlavaConfig`] or [`VipLlavaVisionConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 给类 VipLlavaPreTrainedModel 添加文档字符串
@add_start_docstrings(
    "The bare VipLlava Model outputting raw hidden-states without any specific head on top.",
    VIPLLAVA_START_DOCSTRING,
)
# 定义一个 VipLlavaPreTrainedModel 类，继承自 PreTrainedModel
# 用于加载 VipLlava 模型的权重，提供通用的方法
class VipLlavaPreTrainedModel(PreTrainedModel):
    config_class = VipLlavaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["VipLlavaVisionAttention"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    def _init_weights(self, module):
        # 定义函数_init_weights，用于初始化模型参数
        # 注意：这个移植版本的VipLlava不适用于从头训练，只适用于推理和微调
        # 所以原始代码库https://github.com/haotian-liu/LLaVA/tree/main/vipllava可以用于从头训练

        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )
        # 设置标准差std，并根据条件判断获取正确的值

        if hasattr(module, "class_embedding"):
            # 判断module是否具有class_embedding属性
            module.class_embedding.data.normal_(mean=0.0, std=std)
            # 对module的class_embedding参数进行初始化，服从正态分布，均值为0，标准差为std

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 判断module是否是nn.Linear或nn.Conv2d类型的实例
            module.weight.data.normal_(mean=0.0, std=std)
            # 对module的weight参数进行初始化，服从正态分布，均值为0，标准差为std
            if module.bias is not None:
                module.bias.data.zero_()
                # 如果module有bias参数，则将其初始化为0
        elif isinstance(module, nn.Embedding):
            # 判断module是否是nn.Embedding类型的实例
            module.weight.data.normal_(mean=0.0, std=std)
            # 对module的weight参数进行初始化，服从正态分布，均值为0，标准差为std
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
                # 如果module有padding_idx参数，则将其初始化为0

    @property
    def _supports_sdpa(self):
        """
        Retrieve language_model's attribute to check whether the model supports
        SDPA or not.
        """
        # _supports_sdpa方法的注释，用于检查模型是否支持SDPA（自注意力机制）
        return self.language_model._supports_sdpa
# VIPLLAVA_INPUTS_DOCSTRING为空字符串，用于文档字符串的输入说明
@add_start_docstrings装饰器用于添加模型的开始文档字符串
VIPLLAVA模型由视觉主干和语言模型组成
VIPLLAVA_START_DOCSTRING是VIPLLAVA模型的开始文档字符串

# 定义VipLlavaForConditionalGeneration类，继承自VipLlavaPreTrainedModel类
class VipLlavaForConditionalGeneration(VipLlavaPreTrainedModel):
    # 初始化方法，接受VipLlavaConfig类型的config参数
    def __init__(self, config: VipLlavaConfig):
        # 调用父类的初始化方法
        super().__init__(config)
        # 使用config.vision_config创建视觉塔
        self.vision_tower = AutoModel.from_config(config.vision_config)
        # 创建多模态投影器
        self.multi_modal_projector = VipLlavaMultiModalProjector(config)
        # 获取词汇表大小
        self.vocab_size = config.vocab_size
        # 创建语言模型
        self.language_model = AutoModelForCausalLM.from_config(
            config.text_config, attn_implementation=config._attn_implementation
        )
        # 如果config.pad_token_id不为None，则使用config.pad_token_id，否则使用-1
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        # 调用post_init方法

    # 获取输入嵌入
    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    # 设置输入嵌入
    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    # 获取输出嵌入
    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    # 设置输出嵌入
    def set_output_embeddings(self, new_embeddings):
        self.language_model.set_output_embeddings(new_embeddings)

    # 设置解码器
    def set_decoder(self, decoder):
        self.language_model.set_decoder(decoder)

    # 获取解码器
    def get_decoder(self):
        return self.language_model.get_decoder()

    # 绑定权重
    def tie_weights(self):
        return self.language_model.tie_weights()

    # 调整标记嵌入
    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None) -> nn.Embedding:
        # 调整模型嵌入
        model_embeds = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        # 更新词汇表大小
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        return model_embeds

    # 前向传播方法
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
    # 为生成准备输入数据
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, pixel_values=None, attention_mask=None, **kwargs
    ):
        # 如果过去的键值不为空
        if past_key_values is not None:
            # 如果过去的键值是 Cache 类型
            if isinstance(past_key_values, Cache):
                # 获取缓存长度
                cache_length = past_key_values.get_seq_length()
                # 获取已处理的 token 长度
                past_length = past_key_values.seen_tokens
            else:
                # 获取缓存长度和已处理的 token 长度
                cache_length = past_length = past_key_values[0][0].shape[2]

            # 保留未处理的 token：
            # 1 - 如果 attention_mask 的长度超过 input_ids 的长度，则说明有些输入是作为缓存的一部分传递的
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - 如果 past_length 小于 input_ids 的长度，则 input_ids 包含所有输入 token。我们可以根据 past_length 丢弃 input_ids。
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - 否则（past_length >= input_ids.shape[1]），假设 input_ids 只包含未处理的 token。
            elif self.config.image_token_index in input_ids:
                input_ids = input_ids[:, input_ids.shape[1] - 1 :]
            # 如果缓存看到的 token 数量超过其容量，则缓存有大小限制。让我们丢弃旧的 attention 值，因为它们对应的值不是输入的一部分。
            if cache_length < past_length and attention_mask is not None:
                attention_mask = attention_mask[:, -(cache_length + input_ids.shape[1]) :]

        position_ids = kwargs.get("position_ids", None)
        # 如果 attention_mask 不为空且 position_ids 为空
        if attention_mask is not None and position_ids is None:
            # 为批量生成动态创建 position_ids
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # 如果传递了 `inputs_embeds`，我们只想在第一代步骤中使用它们
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

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
    # 重新排序缓存数据的私有方法，调用语言模型对象的重新排序缓存方法
    def _reorder_cache(self, *args, **kwargs):
        return self.language_model._reorder_cache(*args, **kwargs)
```