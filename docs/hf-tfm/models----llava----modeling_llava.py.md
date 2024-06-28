# `.\models\llava\modeling_llava.py`

```
# 指定编码格式为 UTF-8
# 版权声明，版权归 HuggingFace Inc. 团队所有，保留所有权利
#
# 根据 Apache 许可证 2.0 版本，除非符合许可证规定，否则不得使用本文件
# 可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按"原样"分发本软件，
# 没有任何明示或暗示的保证或条件
# 有关特定语言的权限，请参阅许可证
""" PyTorch Llava model. """

# 导入必要的库和模块
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

# 导入 HuggingFace 通用模型基类
from ... import PreTrainedModel
# 导入激活函数映射表
from ...activations import ACT2FN
# 导入缓存工具函数
from ...cache_utils import Cache
# 导入模型输出基类
from ...modeling_outputs import ModelOutput
# 导入实用工具函数：添加文档字符串、模型前向方法文档字符串、日志记录、替换返回文档字符串
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
# 导入自动加载模型和自动加载 Causal LM 模型
from ..auto import AutoModel, AutoModelForCausalLM
# 导入 Llava 模型的配置文件
from .configuration_llava import LlavaConfig

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# 用于文档的配置名
_CONFIG_FOR_DOC = "LlavaConfig"

# 预训练模型存档列表
LLAVA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "llava-hf/llava-1.5-7b-hf",
    "llava-hf/llava-1.5-13b-hf",
    "llava-hf/bakLlava-v1-hf",
    # 查看所有 Llava 模型的链接 https://huggingface.co/models?filter=llava
]

@dataclass
# 从 transformers.models.idefics.modeling_idefics.IdeficsCausalLMOutputWithPast 复制，并将 Idefics 改为 Llava
class LlavaCausalLMOutputWithPast(ModelOutput):
    """
    Llava 因果语言模型（或自回归模型）输出的基类。
    继承自 ModelOutput 类。
    包含预测输出和过去的信息。
    """
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

    # Loss值，用于语言模型的下一个标记预测的损失
    loss: Optional[torch.FloatTensor] = None
    # 预测得分，语言模型头部的预测分数（SoftMax之前的每个词汇标记的分数）
    logits: torch.FloatTensor = None
    # 过去键值对，用于加速顺序解码的预先计算的隐藏状态（自注意块中的键和值）
    past_key_values: Optional[List[torch.FloatTensor]] = None
    # 隐藏状态，每层模型输出的隐藏状态加上可选的初始嵌入输出
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 注意力权重，自注意头部中用于计算加权平均值的注意力softmax后的注意力权重
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 图像隐藏状态，视觉编码器产生的模型的图像嵌入输出
    image_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
# 定义一个名为 LlavaMultiModalProjector 的类，继承自 nn.Module 类
class LlavaMultiModalProjector(nn.Module):
    # 构造函数，初始化方法
    def __init__(self, config: LlavaConfig):
        # 调用父类的初始化方法
        super().__init__()

        # 使用线性层，将视觉配置的隐藏大小映射到文本配置的隐藏大小，并添加偏置
        self.linear_1 = nn.Linear(config.vision_config.hidden_size, config.text_config.hidden_size, bias=True)
        # 获取激活函数，并赋值给 self.act
        self.act = ACT2FN[config.projector_hidden_act]
        # 再次使用线性层，将文本配置的隐藏大小映射到文本配置的隐藏大小，并添加偏置
        self.linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=True)

    # 前向传播方法
    def forward(self, image_features):
        # 使用第一个线性层进行计算，得到隐藏状态
        hidden_states = self.linear_1(image_features)
        # 使用预定义的激活函数进行激活
        hidden_states = self.act(hidden_states)
        # 使用第二个线性层进行计算，得到最终的隐藏状态
        hidden_states = self.linear_2(hidden_states)
        # 返回计算结果作为输出
        return hidden_states


# 定义 LLAVA_START_DOCSTRING，包含一段原始文档字符串
LLAVA_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`LlavaConfig`] or [`LlavaVisionConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


# 使用装饰器 @add_start_docstrings 添加文档字符串，描述 LlavaPreTrainedModel 类的作用
@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAVA_START_DOCSTRING,
)
# 定义 LlavaPreTrainedModel 类，继承自 PreTrainedModel 类
class LlavaPreTrainedModel(PreTrainedModel):
    # 指定配置类为 LlavaConfig
    config_class = LlavaConfig
    # 基础模型前缀设为 "model"
    base_model_prefix = "model"
    # 支持梯度检查点
    supports_gradient_checkpointing = True
    # 不分割的模块列表
    _no_split_modules = ["LlavaVisionAttention"]
    # 跳过设备位置置入的键
    _skip_keys_device_placement = "past_key_values"
    # 支持 flash_attn_2 特性
    _supports_flash_attn_2 = True

    # 初始化权重的私有方法，根据配置参数进行初始化
    def _init_weights(self, module):
        # 根据配置选择合适的标准差
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        # 如果模块具有 class_embedding 属性，对其进行正态分布初始化
        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        # 根据模块的类型选择初始化方法：线性层或卷积层
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果模块是嵌入层，同样进行正态分布初始化
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
    @property
    def _supports_sdpa(self):
        """
        返回 language_model 对象的属性，用于检查模型是否支持 SDPA。
        """
        # 返回语言模型对象的 _supports_sdpa 属性
        return self.language_model._supports_sdpa
LLAVA_INPUTS_DOCSTRING = r"""
"""


@add_start_docstrings(
    """The LLAVA model which consists of a vision backbone and a language model.""",
    LLAVA_START_DOCSTRING,
)
class LlavaForConditionalGeneration(LlavaPreTrainedModel):
    def __init__(self, config: LlavaConfig):
        super().__init__(config)
        # 初始化视觉塔模型，从给定的视觉配置中创建
        self.vision_tower = AutoModel.from_config(config.vision_config)

        # 初始化多模态投影层
        self.multi_modal_projector = LlavaMultiModalProjector(config)
        
        # 设置词汇表大小
        self.vocab_size = config.text_config.vocab_size
        
        # 初始化语言模型，基于给定的文本配置和注意力机制实现
        self.language_model = AutoModelForCausalLM.from_config(
            config.text_config, attn_implementation=config._attn_implementation
        )
        
        # 设置填充标记的ID，如果没有指定则设为-1
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        
        # 执行初始化后的操作
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
        # 调整语言模型的词汇嵌入层大小
        model_embeds = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        
        # 更新配置中的词汇表大小
        self.config.text_config.vocab_size = model_embeds.num_embeddings
        self.vocab_size = model_embeds.num_embeddings
        
        # 返回调整后的嵌入层
        return model_embeds

    @add_start_docstrings_to_model_forward(LLAVA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=LlavaCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[int] = None,
        vision_feature_select_strategy: Optional[str] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        LLAVA模型的前向传播方法，接受多种输入参数并返回模型的输出。

        Args:
            input_ids (torch.LongTensor, optional): 输入的token IDs序列.
            pixel_values (torch.FloatTensor, optional): 图像的像素值.
            attention_mask (torch.Tensor, optional): 注意力掩码.
            position_ids (torch.LongTensor, optional): 位置IDs.
            past_key_values (List[torch.FloatTensor], optional): 上下文关键值.
            inputs_embeds (torch.FloatTensor, optional): 输入的嵌入向量.
            vision_feature_layer (int, optional): 视觉特征层.
            vision_feature_select_strategy (str, optional): 视觉特征选择策略.
            labels (torch.LongTensor, optional): 标签序列.
            use_cache (bool, optional): 是否使用缓存.
            output_attentions (bool, optional): 是否输出注意力.
            output_hidden_states (bool, optional): 是否输出隐藏状态.
            return_dict (bool, optional): 是否返回字典形式的输出.

        Returns:
            output (LlavaCausalLMOutputWithPast or torch.Tensor): LLAVA模型的输出，可能包含上下文关键值的信息.

        """
        # 实现LLAVA模型的前向传播逻辑，具体细节在其他方法中处理
        raise NotImplementedError

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, pixel_values=None, attention_mask=None, **kwargs
    ):
        """
        准备生成任务的输入，适用于生成文本的情景.

        Args:
            input_ids: 输入的token IDs序列.
            past_key_values: 上下文关键值.
            inputs_embeds: 输入的嵌入向量.
            pixel_values: 图像的像素值.
            attention_mask: 注意力掩码.
            **kwargs: 其他关键字参数.

        Returns:
            dict: 生成任务的输入字典.

        """
        # 实现为生成任务准备输入的逻辑，具体细节在其他方法中处理
        raise NotImplementedError
    ):
        # 如果过去的键值对不为空
        if past_key_values is not None:
            # 如果过去的键值对是Cache类型
            if isinstance(past_key_values, Cache):
                # 获取缓存中的序列长度
                cache_length = past_key_values.get_seq_length()
                # 获取已经处理的token数
                past_length = past_key_values.seen_tokens
            else:
                # 否则，从past_key_values中获取第一个元素的第一个维度的长度，作为缓存长度和已处理长度
                cache_length = past_length = past_key_values[0][0].shape[2]

            # 保留未处理的token：
            # 1 - 如果attention_mask的长度超过input_ids的长度，说明一些输入仅作为缓存的一部分传入（例如当传入input_embeds时）
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - 如果past_length小于input_ids的长度，则input_ids包含所有输入token。我们可以根据past_length丢弃input_ids。
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - 否则（past_length >= input_ids.shape[1]），假设input_ids只有未处理的token。
            elif self.config.image_token_index in input_ids:
                input_ids = input_ids[:, input_ids.shape[1] - 1 :]
            # 如果缓存已经处理的token数超过了它可以容纳的最大限制，那么缓存有一个大小限制。让我们丢弃旧的attention值，因为它们对应的值不是输入的一部分。
            if cache_length < past_length and attention_mask is not None:
                attention_mask = attention_mask[:, -(cache_length + input_ids.shape[1]) :]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # 在批处理生成时动态创建position_ids
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # 如果传入了`inputs_embeds`，我们只在第一个生成步骤中使用它们
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # 更新model_inputs字典
        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
            }
        )
        # 返回model_inputs
        return model_inputs

    # 重新排序缓存
    def _reorder_cache(self, *args, **kwargs):
        return self.language_model._reorder_cache(*args, **kwargs)
```