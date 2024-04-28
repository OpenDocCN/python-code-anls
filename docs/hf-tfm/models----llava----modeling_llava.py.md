# `.\transformers\models\llava\modeling_llava.py`

```
# 导入所需的库和模块
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
from .configuration_llava import LlavaConfig

# 创建日志记录器对象
logger = logging.get_logger(__name__)

# 用于文档的配置信息
_CONFIG_FOR_DOC = "LlavaConfig"

# Llava 预训练模型存档列表
LLAVA_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "llava-hf/llava-1.5-7b-hf",
    "llava-hf/llava-1.5-13b-hf",
    "llava-hf/bakLlava-v1-hf",
    # 查看所有 Llava 模型 https://huggingface.co/models?filter=llava
]

# 用于描述 Llava 语言模型的输出
@dataclass
# 从 transformers.models.idefics.modeling_idefics.IdeficsCausalLMOutputWithPast 复制并将 Idefics->Llava
class LlavaCausalLMOutputWithPast(ModelOutput):
    """
    Llava 因果语言模型（或自回归）输出的基类。
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

    # 定义变量loss，类型为Optional[torch.FloatTensor]，默认值为None
    loss: Optional[torch.FloatTensor] = None
    # 定义变量logits，类型为torch.FloatTensor，默认值为None
    logits: torch.FloatTensor = None
    # 定义变量past_key_values，类型为Optional[List[torch.FloatTensor]]，默认值为None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    # 定义变量hidden_states，类型为Optional[Tuple[torch.FloatTensor]]，默认值为None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 定义变量attentions，类型为Optional[Tuple[torch.FloatTensor]]，默认值为None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    # 定义变量image_hidden_states，类型为Optional[Tuple[torch.FloatTensor]]，默认值为None
    image_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
class LlavaMultiModalProjector(nn.Module):
    def __init__(self, config: LlavaConfig):
        super().__init__()

        # 创建一个线性层，输入大小为 vision_config.hidden_size，输出大小为 text_config.hidden_size
        self.linear_1 = nn.Linear(config.vision_config.hidden_size, config.text_config.hidden_size, bias=True)
        # 激活函数
        self.act = ACT2FN[config.projector_hidden_act]
        # 创建另一个线性层，输入输出大小都为 text_config.hidden_size
        self.linear_2 = nn.Linear(config.text_config.hidden_size, config.text_config.hidden_size, bias=True)

    # 前向传播函数
    def forward(self, image_features):
        # 使用第一个线性层处理图像特征
        hidden_states = self.linear_1(image_features)
        # 对处理后的结果进行激活
        hidden_states = self.act(hidden_states)
        # 使用第二个线性层处理激活后的结果
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


# LLaVA 模型的文档字符串
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


@add_start_docstrings(
    "The bare LLaMA Model outputting raw hidden-states without any specific head on top.",
    LLAVA_START_DOCSTRING,
)
# 创建 LLAvaPreTrainedModel 类，继承自 PreTrainedModel
class LlavaPreTrainedModel(PreTrainedModel):
    # 配置文件类
    config_class = LlavaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlavaVisionAttention"]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True

    # 初始化权重函数
    def _init_weights(self, module):
        # 注意：这个 LLava 的移植版本不适于从头开始训练，只用于推理和微调
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else self.config.text_config.initializer_range
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
    @property
    # 定义属性 _supports_sdpa，用于检查语言模型是否支持 SDPA
    def _supports_sdpa(self):
        """
        Retrieve language_model's attribute to check whether the model supports
        SDPA or not.
        """
        # 返回语言模型的 _supports_sdpa 属性
        return self.language_model._supports_sdpa
# LLAVA_INPUTS_DOCSTRING 的空字符串文档注释
""" 

# 添加模型整体文档注释
@add_start_docstrings(
    """The LLAVA model which consists of a vision backbone and a language model.""",
    LLAVA_START_DOCSTRING,
)

# 定义 LLAVA 训练模型类，继承自 LlavaPreTrainedModel
class LlavaForConditionalGeneration(LlavaPreTrainedModel):

    # 初始化函数
    def __init__(self, config: LlavaConfig):
        # 调用父类初始化方法
        super().__init__(config)
        
        # 创建 vision_tower，基于 config.vision_config
        self.vision_tower = AutoModel.from_config(config.vision_config)

        # 创建 multi_modal_projector
        self.multi_modal_projector = LlavaMultiModalProjector(config)
        
        # 获取词汇表大小
        self.vocab_size = config.vocab_size
        
        # 创建语言模型对象
        self.language_model = AutoModelForCausalLM.from_config(
            config.text_config, attn_implementation=config._attn_implementation
        )
        
        # 设置 pad_token_id
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        
        # 调用初始化后的函数
        self.post_init()

    # 获取输入嵌入词汇表
    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    # 设置输入嵌入词汇表
    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    # 获取输出嵌入词汇表
    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    # 设置输出嵌入词汇表
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
        
        # 返回模型嵌入
        return model_embeds

    # 前向传播方法
    @add_start_docstrings_to_model_forward(LLAVA_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=LlavaCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        ...  # 其他参数省略
    ):
    # 为生成准备输入
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, pixel_values=None, attention_mask=None, **kwargs
    ):
        # 检查是否有之前的键值存在
        if past_key_values is not None:
            # 如果之前的键值是缓存对象
            if isinstance(past_key_values, Cache):
                # 获取缓存对象的序列长度和已经看到的标记数量
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
            else:
                # 如果之前的键值不是缓存对象，则获取第一个键值的维度作为序列长度
                cache_length = past_length = past_key_values[0][0].shape[2]

            # 保留未处理的标记：
            # 1 - 如果 attention_mask 的长度超过了 input_ids 的长度，则说明有些输入是作为缓存的一部分传递的（例如，当将 input_embeds 作为输入时）
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                # 截取 input_ids 的最后一部分，保留未处理的标记
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - 如果 past_length 小于 input_ids 的长度，则说明 input_ids 包含了所有的输入标记，可以根据 past_length 丢弃一部分 input_ids
            elif past_length < input_ids.shape[1]:
                # 丢弃 input_ids 中已经处理的部分
                input_ids = input_ids[:, past_length:]
            # 3 - 否则（past_length >= input_ids.shape[1]），假设 input_ids 只包含未处理的标记
            elif self.config.image_token_index in input_ids:
                # 如果图像标记在 input_ids 中，则假设 input_ids 只包含一个未处理的标记
                input_ids = input_ids[:, input_ids.shape[1] - 1 :]
            # 如果缓存看到的标记数量超过了其容量限制，则需要丢弃旧的注意力值，因为对应的值不再是输入的一部分
            if cache_length < past_length and attention_mask is not None:
                # 截取 attention_mask 中未过时的部分
                attention_mask = attention_mask[:, -(cache_length + input_ids.shape[1]) :]

        # 获取额外的参数 position_ids
        position_ids = kwargs.get("position_ids", None)
        # 如果存在 attention_mask 但不存在 position_ids，则在批量生成时动态创建 position_ids
        if attention_mask is not None and position_ids is None:
            # 根据 attention_mask 计算 position_ids
            position_ids = attention_mask.long().cumsum(-1) - 1
            # 将 attention_mask 为 0 的位置填充为 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            # 如果存在过去的键值，则只保留最后 input_ids.shape[1] 个位置的 position_ids
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # 如果传递了 inputs_embeds，则只在第一个生成步骤中使用它们
        if inputs_embeds is not None and past_key_values is None:
            # 使用 inputs_embeds 作为模型输入
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # 使用 input_ids 作为模型输入
            model_inputs = {"input_ids": input_ids}

        # 更新模型输入字典
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

    # 重排序缓存
    def _reorder_cache(self, *args, **kwargs):
        # 调用语言模型的 _reorder_cache 方法
        return self.language_model._reorder_cache(*args, **kwargs)
```