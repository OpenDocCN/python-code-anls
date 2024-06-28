# `.\models\llava_next\modeling_llava_next.py`

```py
# coding=utf-8
# Copyright 2024 the HuggingFace Inc. team. All rights reserved.
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
""" PyTorch Llava-NeXT model."""

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from ... import PreTrainedModel
from ...activations import ACT2FN
from ...cache_utils import Cache
from ...image_processing_utils import select_best_resolution
from ...modeling_outputs import ModelOutput
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from ..auto import AutoModel, AutoModelForCausalLM
from .configuration_llava_next import LlavaNextConfig


logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "LlavaNextConfig"

LLAVA_NEXT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "llava-hf/llava-v1.6-mistral-7b-hf",
    # See all LLaVA-NeXT models at https://huggingface.co/models?filter=llava_next
]

# 定义一个函数用于计算图像预处理后的图像补丁网格形状
def get_anyres_image_grid_shape(image_size, grid_pinpoints, patch_size):
    """
    Calculate the shape of the image patch grid after the preprocessing for images of any resolution.

    Args:
        image_size (`tuple`):
            The size of the input image in the format (width, height).
        grid_pinpoints (`List`):
            A list containing possible resolutions. Each item in the list should be a tuple or list
            of the form `(height, width)`.
        patch_size (`int`):
            The size of each image patch.

    Returns:
        tuple: The shape of the image patch grid in the format (width, height).
    """
    if not isinstance(grid_pinpoints, list):
        raise ValueError("grid_pinpoints should be a list of tuples or lists")

    # 从可能的分辨率中选择最佳的分辨率
    height, width = select_best_resolution(image_size, grid_pinpoints)
    # 计算图像补丁网格的形状
    return height // patch_size, width // patch_size


# 定义一个函数用于解压经过填充和调整大小的图像的 PyTorch 张量
def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
        tensor (`torch.Tensor`):
            The image tensor, assumed to be of shape (num_channels, height, width).
        original_size (`tuple`):
            The original size of the image (height, width).

    Returns:
        `torch.Tensor`: The unpadded image tensor.
    """
    original_height, original_width = original_size
    current_height, current_width = tensor.shape[1:]

    # 计算原始图像的宽高比
    original_aspect_ratio = original_width / original_height
    # 计算当前图像的宽高比
    current_aspect_ratio = current_width / current_height

    # 检查原始图像的宽高比是否大于当前图像的宽高比
    if original_aspect_ratio > current_aspect_ratio:
        # 如果是，按照宽度比例缩放当前图像，并计算新的高度
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        # 计算垂直方向上的填充量，使得缩放后的图像居中
        padding = (current_height - new_height) // 2
        # 在垂直方向上截取不带填充的部分
        unpadded_tensor = tensor[:, padding : current_height - padding, :]
    else:
        # 如果原始图像的宽高比小于等于当前图像的宽高比，按照高度比例缩放当前图像
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        # 计算水平方向上的填充量，使得缩放后的图像居中
        padding = (current_width - new_width) // 2
        # 在水平方向上截取不带填充的部分
        unpadded_tensor = tensor[:, :, padding : current_width - padding]

    # 返回经过处理的不带填充的部分图像张量
    return unpadded_tensor
@dataclass
# 定义了一个数据类，用于表示LlavaNext模型的因果语言模型输出及其历史信息
class LlavaNextCausalLMOutputWithPast(ModelOutput):
    """
    LlavaNext因果语言模型（或自回归模型）输出的基类。

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, 当提供`labels`时返回):
            语言建模损失（用于下一个标记预测）。
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            语言建模头的预测分数（SoftMax之前的每个词汇标记的分数）。
        past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, 当传递`use_cache=True`或`config.use_cache=True`时返回):
            长度为`config.n_layers`的元组，每个元组包含2个形状为`(batch_size, num_heads, sequence_length, embed_size_per_head)`的张量。

            包含预先计算的隐藏状态（注意力块中的键和值），可用于加速顺序解码。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, 当传递`output_hidden_states=True`或`config.output_hidden_states=True`时返回):
            元组`torch.FloatTensor`（如果模型有嵌入层则为一个，每个层的输出为一个）的形状为`(batch_size, sequence_length, hidden_size)`。

            模型每一层输出的隐藏状态加上可选的初始嵌入输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, 当传递`output_attentions=True`或`config.output_attentions=True`时返回):
            元组`torch.FloatTensor`（每个层一个）的形状为`(batch_size, num_heads, sequence_length, sequence_length)`。

            自注意力头中注意力softmax后的注意力权重，用于计算自注意力头中加权平均值。
        image_hidden_states (`tuple(torch.FloatTensor)`, *optional*):
            元组`torch.FloatTensor`（图像嵌入的输出一个）的形状为`(batch_size, num_images, sequence_length, hidden_size)`。

            模型通过视觉编码器生成的图像隐藏状态，以及可选的感知器生成的图像隐藏状态。
    """
    
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[Tuple[torch.FloatTensor]] = None


# 从transformers.models.llava.modeling_llava.LlavaMultiModalProjector复制并改为LlavaNext
class LlavaNextMultiModalProjector(nn.Module):
    # 初始化函数，用于创建一个新的神经网络模型对象
    def __init__(self, config: LlavaNextConfig):
        # 调用父类（nn.Module）的初始化方法
        super().__init__()

        # 创建一个线性层，将输入特征的大小映射到文本配置中隐藏层的大小
        self.linear_1 = nn.Linear(config.vision_config.hidden_size, config.text_config.hidden_size, bias=True)
        
        # 选择激活函数，根据配置文件中指定的激活函数类型从预定义的字典中选择
        self.act = ACT2FN[config.projector_hidden_act]
        
        # 创建第二个线性层，将第一个线性层的输出映射到文本配置中隐藏层的大小
        self.linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=True)

    # 前向传播函数，定义了数据从输入到输出的流动方式
    def forward(self, image_features):
        # 第一层线性变换，将输入特征映射到文本配置中隐藏层的大小
        hidden_states = self.linear_1(image_features)
        
        # 应用选定的激活函数到第一层的输出
        hidden_states = self.act(hidden_states)
        
        # 第二层线性变换，将第一层的输出映射到文本配置中隐藏层的大小
        hidden_states = self.linear_2(hidden_states)
        
        # 返回最终的隐藏状态作为输出
        return hidden_states
# LLAVA_NEXT_START_DOCSTRING 变量，包含了关于 LLAVA-NeXT 模型的文档字符串，描述了其继承自 PreTrainedModel 的特性，
# 以及作为 PyTorch 的 nn.Module 的子类使用的相关信息和参数说明。

@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAVA_NEXT_START_DOCSTRING,
)
# 使用 add_start_docstrings 装饰器为 LlavaNextPreTrainedModel 类添加文档字符串，描述了其作为基础模型输出原始隐藏状态的特性，
# 并引用了 LLAVA_NEXT_START_DOCSTRING 中定义的模型配置和参数说明。

class LlavaNextPreTrainedModel(PreTrainedModel):
    # LlavaNextPreTrainedModel 类，继承自 PreTrainedModel 类。
    config_class = LlavaNextConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlavaNextVisionAttention"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True

    def _init_weights(self, module):
        # _init_weights 方法用于初始化模型参数，根据模型配置设置不同类型模块的权重初始化方式。
        # 在此版本的 LlavaNext 中，仅支持推断和微调，不支持从头开始训练，因此移除了原始代码中的适用于从头训练的初始化权重代码。

        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            # 如果模块具有 class_embedding 属性，则对其进行正态分布初始化，均值为 0，标准差为 std。
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 如果模块是 nn.Linear 或 nn.Conv2d 类型，则对权重进行正态分布初始化，均值为 0，标准差为 std。
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                # 如果模块有偏置项，则将偏置项初始化为 0。
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # 如果模块是 nn.Embedding 类型，则对权重进行正态分布初始化，均值为 0，标准差为 std。
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                # 如果模块有 padding_idx，则将该位置的权重初始化为 0。
                module.weight.data[module.padding_idx].zero_()

    @property
    def _supports_sdpa(self):
        """
        Retrieve language_model's attribute to check whether the model supports
        SDPA or not.
        """
        # _supports_sdpa 属性，用于检查模型是否支持 SDPA（Self-Attention based Dual-path Aggregation）。
        return self.language_model._supports_sdpa


LLAVA_NEXT_INPUTS_DOCSTRING = r"""
"""
# LLAVA_NEXT_INPUTS_DOCSTRING 变量，目前为空字符串，用于定义 LLAVA-NeXT 模型的输入文档字符串。

@add_start_docstrings(
    """The LLAVA-NeXT model which consists of a vision backbone and a language model.""",
    LLAVA_NEXT_START_DOCSTRING,
)
# 使用 add_start_docstrings 装饰器为 LLAVA-NeXT 模型添加文档字符串，描述了该模型由视觉骨干和语言模型组成的特性，
# 并引用了 LLAVA_NEXT_START_DOCSTRING 中定义的模型配置和参数说明。
    # 继承自 LlavaNextPreTrainedModel 类的条件生成模型，用于生成下一步预测的输出
    class LlavaNextForConditionalGeneration(LlavaNextPreTrainedModel):
        def __init__(self, config: LlavaNextConfig):
            # 调用父类的初始化方法，传入配置对象
            super().__init__(config)
            # 根据视觉配置创建自动模型对象
            self.vision_tower = AutoModel.from_config(config.vision_config)

            # 创建多模态投影器对象
            self.multi_modal_projector = LlavaNextMultiModalProjector(config)

            # 创建可学习的张量参数，用于图像和文本信息的融合
            self.image_newline = nn.Parameter(torch.empty(config.text_config.hidden_size, dtype=self.dtype))

            # 获取文本配置中的词汇表大小
            self.vocab_size = config.text_config.vocab_size

            # 根据文本配置创建自动语言模型对象，支持因果语言模型
            self.language_model = AutoModelForCausalLM.from_config(
                config.text_config, attn_implementation=config._attn_implementation
            )

            # 设置填充标记 ID，如果配置中未指定填充标记 ID，则设置为 -1
            self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1

            # 执行初始化后的处理逻辑
            self.post_init()

        # 从语言模型中获取输入嵌入层
        # 复制自 transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.get_input_embeddings
        def get_input_embeddings(self):
            return self.language_model.get_input_embeddings()

        # 设置语言模型的输入嵌入层
        # 复制自 transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.set_input_embeddings
        def set_input_embeddings(self, value):
            self.language_model.set_input_embeddings(value)

        # 从语言模型中获取输出嵌入层
        # 复制自 transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.get_output_embeddings
        def get_output_embeddings(self):
            return self.language_model.get_output_embeddings()

        # 设置语言模型的输出嵌入层
        # 复制自 transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.set_output_embeddings
        def set_output_embeddings(self, new_embeddings):
            self.language_model.set_output_embeddings(new_embeddings)

        # 设置语言模型的解码器
        # 复制自 transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.set_decoder
        def set_decoder(self, decoder):
            self.language_model.set_decoder(decoder)

        # 从语言模型中获取解码器
        # 复制自 transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.get_decoder
        def get_decoder(self):
            return self.language_model.get_decoder()

        # 绑定语言模型的权重
        # 复制自 transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.tie_weights
        def tie_weights(self):
            return self.language_model.tie_weights()

        # 调整语言模型的标记嵌入大小
        # 复制自 transformers.models.llava.modeling_llava.LlavaForConditionalGeneration.resize_token_embeddings
        def resize_token_embeddings(self, new_num_tokens: Optional[int] = None, pad_to_multiple_of=None) -> nn.Embedding:
            model_embeds = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
            # 更新配置中的词汇表大小
            self.config.text_config.vocab_size = model_embeds.num_embeddings
            self.vocab_size = model_embeds.num_embeddings
            return model_embeds

        # 合并输入标记 ID 与图像特征的处理逻辑，支持多模态输入
        # 复制自 transformers.models.llava.modeling_llava.LlavaForConditionalGeneration._merge_input_ids_with_image_features
        @add_start_docstrings_to_model_forward(LLAVA_NEXT_INPUTS_DOCSTRING)
        @replace_return_docstrings(output_type=LlavaNextCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    # 定义模型的前向传播方法，接受多个输入参数
    def forward(
        self,
        input_ids: torch.LongTensor = None,  # 输入的token ID张量，用于输入模型的文本序列
        pixel_values: torch.FloatTensor = None,  # 输入的像素值张量，用于输入模型的图像特征
        image_sizes: Optional[torch.LongTensor] = None,  # 可选的图像尺寸张量，用于指定输入图像的尺寸
        attention_mask: Optional[torch.Tensor] = None,  # 可选的注意力掩码张量，用于指定模型关注的位置
        position_ids: Optional[torch.LongTensor] = None,  # 可选的位置 ID 张量，用于指定输入的位置信息
        past_key_values: Optional[List[torch.FloatTensor]] = None,  # 可选的过去键值张量列表，用于缓存先前计算的键值信息
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 可选的嵌入输入张量，用于直接提供嵌入的输入
        vision_feature_layer: Optional[int] = None,  # 可选的视觉特征层索引，用于指定从哪个视觉特征层提取特征
        vision_feature_select_strategy: Optional[str] = None,  # 可选的视觉特征选择策略，用于控制视觉特征的选择方式
        labels: Optional[torch.LongTensor] = None,  # 可选的标签张量，用于计算模型的损失
        use_cache: Optional[bool] = None,  # 可选的缓存使用标志，用于控制是否使用缓存
        output_attentions: Optional[bool] = None,  # 可选的输出注意力张量标志，用于控制是否输出注意力权重
        output_hidden_states: Optional[bool] = None,  # 可选的输出隐藏状态标志，用于控制是否输出中间层的隐藏状态
        return_dict: Optional[bool] = None,  # 可选的返回字典标志，用于控制是否返回字典形式的输出
    ):
        pass  # 此处为定义方法后的代码块结尾，无具体操作，仅作为示例展示参数的定义和类型说明

    # 定义生成过程的输入准备方法，接受多个输入参数和额外关键字参数
    def prepare_inputs_for_generation(
        self,
        input_ids,  # 输入的token ID张量，用于生成过程中的输入文本序列
        past_key_values=None,  # 可选的过去键值张量，用于缓存先前计算的键值信息
        inputs_embeds=None,  # 可选的嵌入输入张量，用于直接提供嵌入的输入
        pixel_values=None,  # 可选的像素值张量，用于生成过程中的输入图像特征
        image_sizes=None,  # 可选的图像尺寸张量，用于指定生成过程中输入图像的尺寸
        attention_mask=None,  # 可选的注意力掩码张量，用于指定生成过程中模型关注的位置
        **kwargs,  # 其余的关键字参数，用于兼容可能添加的未列出参数
    ):
        pass  # 此处为定义方法后的代码块结尾，无具体操作，仅作为示例展示参数的定义和类型说明
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.
            elif self.config.image_token_index in input_ids:
                input_ids = input_ids[:, input_ids.shape[1] - 1 :]
            # If the cache has seen more tokens than it can hold, then the cache has a size limit. Let's discard the
            # older attention values, as their corresponding values are not part of the input.
            if cache_length < past_length and attention_mask is not None:
                attention_mask = attention_mask[:, -(cache_length + input_ids.shape[1]) :]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
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
                "image_sizes": image_sizes,
            }
        )
        return model_inputs

    # Copied from transformers.models.llava.modeling_llava.LlavaForConditionalGeneration._reorder_cache
    def _reorder_cache(self, *args, **kwargs):
        return self.language_model._reorder_cache(*args, **kwargs)
```