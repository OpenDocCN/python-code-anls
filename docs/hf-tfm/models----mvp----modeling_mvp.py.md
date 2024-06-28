# `.\models\mvp\modeling_mvp.py`

```
# coding=utf-8
# Copyright 2022 The Fairseq Authors and The HuggingFace Inc. team. All rights reserved.
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
""" PyTorch MVP model."""
import copy
import math
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqQuestionAnsweringModelOutput,
    Seq2SeqSequenceClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_code_sample_docstrings,
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_mvp import MvpConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "RUCAIBox/mvp"
_CONFIG_FOR_DOC = "MvpConfig"

# Base model docstring
_EXPECTED_OUTPUT_SHAPE = [1, 8, 1024]

MVP_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "RUCAIBox/mvp",
    "RUCAIBox/mvp-data-to-text",
    "RUCAIBox/mvp-open-dialog",
    "RUCAIBox/mvp-question-answering",
    "RUCAIBox/mvp-question-generation",
    "RUCAIBox/mvp-story",
    "RUCAIBox/mvp-summarization",
    "RUCAIBox/mvp-task-dialog",
    "RUCAIBox/mtl-data-to-text",
    "RUCAIBox/mtl-multi-task",
    "RUCAIBox/mtl-open-dialog",
    "RUCAIBox/mtl-question-answering",
    "RUCAIBox/mtl-question-generation",
    "RUCAIBox/mtl-story",
    "RUCAIBox/mtl-summarization",
    # See all MVP models at https://huggingface.co/models?filter=mvp
]


# Copied from transformers.models.bart.modeling_bart.shift_tokens_right
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    
    Args:
        input_ids (torch.Tensor): Tensor of input ids.
        pad_token_id (int): The id of the padding token in the model's configuration.
        decoder_start_token_id (int): The id of the decoder's start token.
    """
    # Create a tensor of zeros with the same shape as input_ids
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    # Shift input ids one position to the right
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    # Place the decoder start token at the beginning of each sequence
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        # Raise an error if pad_token_id is not defined
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # Replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)
    # 返回经过移位处理后的输入 ID 列表
    return shifted_input_ids
# 从 transformers.models.bart.modeling_bart.BartLearnedPositionalEmbedding 复制并修改为 Mvp
class MvpLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # 如果指定了 padding_idx，则将 embedding ids 偏移 2，并相应地调整 num_embeddings
        # 其他模型没有这个 hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, input_ids: torch.Tensor, past_key_values_length: int = 0):
        """`input_ids' shape is expected to be [bsz x seqlen]."""
        
        # 获取 batch size 和序列长度
        bsz, seq_len = input_ids.shape[:2]
        # 根据序列长度生成位置信息张量，加上偏移量 self.offset
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        ).expand(bsz, -1)
        
        # 调用父类的 forward 方法来计算位置编码的 embedding
        return super().forward(positions + self.offset)


class MvpAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        # 初始化注意力机制的参数
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        # 检查 embed_dim 必须能被 num_heads 整除，否则抛出 ValueError
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        # 缩放因子，用于注意力分数计算
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        # 初始化线性变换层，用于查询、键、值和输出的线性映射
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        # 将 tensor 重塑为 [bsz, num_heads, seq_len, head_dim] 的形状
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        attn_prompt: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        # 省略了 forward 方法的具体实现部分
        pass  # 实现在其他地方
    # 初始化方法，接受一个MvpConfig类型的配置参数config
    def __init__(self, config: MvpConfig):
        # 调用父类的初始化方法
        super().__init__()
        # 设置嵌入维度为config中定义的模型维度d_model
        self.embed_dim = config.d_model
        # 创建自注意力层对象，使用MvpAttention类，设置参数包括嵌入维度、注意力头数、注意力层的dropout
        self.self_attn = MvpAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        # 创建自注意力层的LayerNorm层，输入维度为embed_dim
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # 设置全连接层的dropout概率
        self.dropout = config.dropout
        # 根据配置中的激活函数选择相应的激活函数
        self.activation_fn = ACT2FN[config.activation_function]
        # 设置激活函数的dropout概率
        self.activation_dropout = config.activation_dropout
        # 创建第一个全连接层，输入维度为embed_dim，输出维度为encoder_ffn_dim
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        # 创建第二个全连接层，输入维度为encoder_ffn_dim，输出维度为embed_dim
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        # 创建最终的LayerNorm层，输入维度为embed_dim
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            self_attn_prompt (`torch.FloatTensor`): prompt of self attention of shape
                `(2, encoder_attention_heads, pro_len, head_dim)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        # 将输入的 hidden_states 赋值给 residual，用于后续残差连接
        residual = hidden_states
        # 调用 self_attn 方法，执行自注意力计算
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            attn_prompt=self_attn_prompt,
            output_attentions=output_attentions,
        )
        # 对 hidden_states 应用 dropout
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # 执行残差连接
        hidden_states = residual + hidden_states
        # 对连接后的 hidden_states 应用 layer normalization
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # 再次使用 residual 记录当前 hidden_states，用于下一步残差连接
        residual = hidden_states
        # 通过 fc1 和激活函数执行全连接层计算
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 对 hidden_states 应用 dropout
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        # 通过 fc2 执行全连接层计算
        hidden_states = self.fc2(hidden_states)
        # 对 hidden_states 应用 dropout
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # 执行残差连接
        hidden_states = residual + hidden_states
        # 对连接后的 hidden_states 应用 layer normalization
        hidden_states = self.final_layer_norm(hidden_states)

        # 如果 hidden_states 的数据类型为 torch.float16，并且包含无穷大或 NaN 值，则进行截断处理
        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        # 将最终结果存入 outputs 中
        outputs = (hidden_states,)

        # 如果设置了 output_attentions=True，则将注意力权重 attn_weights 添加到 outputs 中一并返回
        if output_attentions:
            outputs += (attn_weights,)

        return outputs
# 定义一个名为 MvpDecoderLayer 的类，继承自 nn.Module 类，用于 MVP 模型的解码器层
class MvpDecoderLayer(nn.Module):
    def __init__(self, config: MvpConfig):
        super().__init__()
        self.embed_dim = config.d_model  # 设置编码维度为配置中的模型维度大小

        # 创建自注意力机制（self-attention）对象，用于解码器，配置包括维度、注意力头数、dropout 等
        self.self_attn = MvpAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )

        self.dropout = config.dropout  # 设置模型的全局dropout比例
        self.activation_fn = ACT2FN[config.activation_function]  # 获取激活函数
        self.activation_dropout = config.activation_dropout  # 获取激活函数的dropout比例

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)  # 对自注意力输出进行 layer normalization

        # 创建编码器-解码器注意力机制对象，配置包括维度、注意力头数、dropout 等
        self.encoder_attn = MvpAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)  # 对编码器-解码器注意力输出进行 layer normalization

        # 全连接层 1，输入维度为 embed_dim，输出维度为配置中的解码器前馈神经网络维度
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        # 全连接层 2，输入维度为配置中的解码器前馈神经网络维度，输出维度为 embed_dim
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)

        self.final_layer_norm = nn.LayerNorm(self.embed_dim)  # 对最终输出进行 layer normalization

    # 前向传播函数，接受多个输入张量，执行解码器层的前向计算
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        self_attn_prompt: Optional[torch.Tensor] = None,
        cross_attn_prompt: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ):
        # 省略前向传播函数的具体实现，因为只需要注释每个参数和输入的作用
        pass


# 定义一个名为 MvpClassificationHead 的类，继承自 nn.Module 类，用于 MVP 模型的分类头部
# 适用于句子级分类任务
class MvpClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
        self,
        input_dim: int,
        inner_dim: int,
        num_classes: int,
        pooler_dropout: float,
    ):
        super().__init__()
        # 全连接层，输入维度为 input_dim，输出维度为 inner_dim
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)  # dropout 层，使用给定的 dropout 比例
        # 输出投影层，输入维度为 inner_dim，输出维度为类别数 num_classes
        self.out_proj = nn.Linear(inner_dim, num_classes)

    # 前向传播函数，接受隐藏状态张量作为输入，执行分类头部的前向计算
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dropout(hidden_states)  # 应用 dropout 到隐藏状态
        hidden_states = self.dense(hidden_states)  # 进行全连接层计算
        hidden_states = torch.tanh(hidden_states)  # 应用 tanh 激活函数
        hidden_states = self.dropout(hidden_states)  # 再次应用 dropout
        hidden_states = self.out_proj(hidden_states)  # 输出投影到类别数维度
        return hidden_states  # 返回最终的隐藏状态


# 定义一个名为 MvpPrompt 的类，继承自 nn.Module 类，用于 MVP 模型的编码器或解码器的逐层提示
class MvpPrompt(nn.Module):
    """Layer-wise prompt for encoder or decoder."""
    # 初始化函数，用于设置模型参数和层次结构
    def __init__(self, config, num_layers, num_heads):
        super().__init__()
        # 从配置中获取提示文本长度并赋值给实例变量
        self.prompt_length = config.prompt_length
        # 将层数赋值给实例变量
        self.num_layers = num_layers
        # 将头数赋值给实例变量
        self.num_heads = num_heads
        # 计算每个头的维度
        self.head_dim = config.d_model // num_heads
        # 根据配置中的 dropout 概率创建一个 dropout 层
        self.dropout = nn.Dropout(p=config.dropout)
        # 创建一个提示文本的嵌入层，大小为 (提示文本长度, 模型维度)
        self.prompt_embedding = nn.Embedding(config.prompt_length, config.d_model)
        # 创建一个序列模块，用于处理提示文本的转换
        self.prompt_trans = nn.Sequential(
            # 线性层，将模型维度映射到中间维度
            nn.Linear(config.d_model, config.prompt_mid_dim),
            # GELU 激活函数
            nn.GELU(),
            # 再次线性映射，将中间维度映射为 (层数 * 2 * 模型维度)
            nn.Linear(config.prompt_mid_dim, num_layers * 2 * config.d_model),
        )

    # 前向传播函数，用于计算模型的输出
    def forward(self, prompt_ids: torch.Tensor) -> Tuple[torch.Tensor]:
        # 将输入的提示文本 IDs 转换为嵌入表示，并通过提示文本转换模块进行转换
        prompt = self.prompt_trans(self.prompt_embedding(prompt_ids))
        # 将转换后的结果重新形状为 (提示文本长度, 层数 * 2, 头数, 每个头的维度)
        prompt = prompt.view(self.prompt_length, self.num_layers * 2, self.num_heads, self.head_dim)
        # 对转换后的结果应用 dropout
        prompt = self.dropout(prompt)
        # 将维度重新排序，顺序为 (层数 * 2, 头数, 提示文本长度, 每个头的维度)，并按照指定维度拆分成多个张量
        prompt = prompt.permute([1, 2, 0, 3]).split(2)
        # 返回处理后的张量元组作为模型的输出
        return prompt
class MvpPreTrainedModel(PreTrainedModel):
    # 指定配置类为 MvpConfig
    config_class = MvpConfig
    # 基础模型前缀为 "model"
    base_model_prefix = "model"
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        # 初始化权重函数，根据配置中的初始标准差进行初始化
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            # 如果是线性层，使用正态分布初始化权重，偏置初始化为零
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # 如果是嵌入层，使用正态分布初始化权重，对于填充索引，将其权重初始化为零
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def dummy_inputs(self):
        # 获取虚拟输入示例，包括输入的填充标记和输入 ID
        pad_token = self.config.pad_token_id
        input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]], device=self.device)
        dummy_inputs = {
            "attention_mask": input_ids.ne(pad_token),  # 生成注意力遮罩
            "input_ids": input_ids,  # 输入 ID
        }
        return dummy_inputs


MVP_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`MvpConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

MVP_INPUTS_DOCSTRING = r"""
    Placeholder for inputs documentation.
"""

MVP_CONDITIONAL_GENERATION_EXAMPLE = r"""
    Example of summarization:

    Fine-tuning a model
    ```python
    >>> import torch
    >>> from transformers import AutoTokenizer, MvpForConditionalGeneration

    >>> tokenizer = AutoTokenizer.from_pretrained("RUCAIBox/mvp")
    >>> model = MvpForConditionalGeneration.from_pretrained("RUCAIBox/mvp")

    >>> inputs = tokenizer(
    ...     "Summarize: You may want to stick it to your boss and leave your job, but don't do it if these are your reasons.",
    ...     return_tensors="pt",
    ... )
    >>> labels = tokenizer("Bad Reasons To Quit Your Job", return_tensors="pt")["input_ids"]

    >>> loss = model(**inputs, labels=labels).loss
    >>> loss.backward()
    ```

    Inference after the model fine-tuned
    ```python
    >>> with torch.no_grad():
    ...     generated_ids = model.generate(**inputs)

    >>> generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    ```
"""

MVP_SEQUENCE_CLASSIFICATION_SAMPLE = r"""
    Example of single-label classification:

    Fine-tuning a model on `num_labels` classes
    ```python
    Placeholder for sequence classification sample.
    # 导入PyTorch库
    import torch
    # 从transformers库中导入AutoTokenizer和MvpForSequenceClassification类
    from transformers import AutoTokenizer, MvpForSequenceClassification
    
    # 设置类别数为2，示例中是一个二分类任务
    num_labels = 2
    # 使用预训练模型"RUCAIBox/mvp"初始化分词器
    tokenizer = AutoTokenizer.from_pretrained("RUCAIBox/mvp")
    # 使用预训练模型"RUCAIBox/mvp"初始化序列分类模型，并指定类别数
    model = MvpForSequenceClassification.from_pretrained("RUCAIBox/mvp", num_labels=num_labels)
    
    # 对输入文本进行分词和转换为PyTorch张量格式
    inputs = tokenizer("Classify: Hello, my dog is cute", return_tensors="pt")
    # 设置输入文本对应的真实标签
    labels = torch.tensor(1)
    
    # 使用模型进行前向传播并计算损失
    loss = model(**inputs, labels=labels).loss
    # 根据损失计算梯度
    loss.backward()
    
    # 在模型微调后进行推理
    # 禁用梯度计算
    with torch.no_grad():
        # 获取模型的输出日志概率
        logits = model(**inputs).logits
    
    # 获取预测的类别ID，即输出概率最高的类别
    predicted_class_id = logits.argmax()
"""
MVP_QUESTION_ANSWERING_SAMPLE = r"""
    Example:

    Fine-tuning a model for extrative question answering, and our model also supports generative question answering
    using `BartForConditionalGeneration`
    ```python
    >>> import torch
    >>> from transformers import AutoTokenizer, MvpForQuestionAnswering

    >>> tokenizer = AutoTokenizer.from_pretrained("RUCAIBox/mvp")
    >>> model = MvpForQuestionAnswering.from_pretrained("RUCAIBox/mvp")

    >>> inputs = tokenizer(
    ...     "Answer the following question: Who was Jim Henson? [SEP] Jim Henson was a nice puppet",
    ...     return_tensors="pt",
    ... )
    >>> target_start_index = torch.tensor([18])
    >>> target_end_index = torch.tensor([19])

    >>> loss = model(**inputs, start_positions=target_start_index, end_positions=target_end_index).loss
    >>> loss.backward()
    ```

    Inference after the model fine-tuned
    ```python
    >>> with torch.no_grad():
    ...     outputs = model(**inputs)

    >>> answer_start_index = outputs.start_logits.argmax()
    >>> answer_end_index = outputs.end_logits.argmax()

    >>> predict_answer_tokens = inputs.input_ids[0, answer_start_index : answer_end_index + 1]
    >>> predict_answer = tokenizer.decode(predict_answer_tokens)
    ```
"""


class MvpEncoder(MvpPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`MvpEncoderLayer`].

    Args:
        config: MvpConfig
        embed_tokens (nn.Embedding): output embedding
        use_prompt (bool): whether to use prompt
    """

    def __init__(
        self, config: MvpConfig, embed_tokens: Optional[nn.Embedding] = None, use_prompt: Optional[bool] = False
    ):
        super().__init__(config)

        # Dropout rate as specified in the configuration
        self.dropout = config.dropout
        # Layer dropout rate as specified in the configuration
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        # Padding index for the embeddings
        self.padding_idx = config.pad_token_id
        # Maximum position embeddings allowed
        self.max_source_positions = config.max_position_embeddings
        # Embedding scale factor
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            # Use provided embedding tokens
            self.embed_tokens = embed_tokens
        else:
            # Otherwise, create new embedding tokens
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        # Learned positional embeddings
        self.embed_positions = MvpLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )
        # List of encoder layers
        self.layers = nn.ModuleList([MvpEncoderLayer(config) for _ in range(config.encoder_layers)])
        # Layer normalization for embeddings
        self.layernorm_embedding = nn.LayerNorm(embed_dim)

        self.use_prompt = use_prompt
        if use_prompt:
            # Length of the prompt
            self.prompt_length = config.prompt_length
            # Self-attention mechanism for prompts
            self.self_attn_prompt = MvpPrompt(
                config,
                config.encoder_layers,
                config.encoder_attention_heads,
            )

        # Gradient checkpointing disabled by default
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()
    # 定义一个方法，用于获取输入的嵌入表示
    def get_input_embeddings(self):
        # 返回存储在对象中的嵌入表示
        return self.embed_tokens

    # 定义一个方法，用于设置输入的嵌入表示
    def set_input_embeddings(self, value):
        # 将传入的值赋给对象中的嵌入表示
        self.embed_tokens = value

    # 定义模型的前向传播方法
    def forward(
        self,
        input_ids: torch.LongTensor = None,  # 输入的token id张量，默认为None
        attention_mask: Optional[torch.Tensor] = None,  # 注意力遮罩张量，可选
        head_mask: Optional[torch.Tensor] = None,  # 头部遮罩张量，可选
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入的嵌入表示张量，可选
        output_attentions: Optional[bool] = None,  # 是否输出注意力张量，可选
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态张量，可选
        return_dict: Optional[bool] = None,  # 是否以字典形式返回结果，可选
# 定义一个名为 MvpDecoder 的类，继承自 MvpPreTrainedModel
class MvpDecoder(MvpPreTrainedModel):
    """
    Transformer 解码器，由 config.decoder_layers 层组成。每层是一个 `MvpDecoderLayer` 对象。

    Args:
        config: MvpConfig 对象，配置参数
        embed_tokens (nn.Embedding): 输出的嵌入层
        use_prompt (bool): 是否使用提示语
    """

    # 初始化方法，接受配置 config、嵌入层 embed_tokens 和是否使用提示语 use_prompt 作为参数
    def __init__(
        self, config: MvpConfig, embed_tokens: Optional[nn.Embedding] = None, use_prompt: Optional[bool] = False
    ):
        # 调用父类的初始化方法
        super().__init__(config)
        
        # 设置类的属性
        self.dropout = config.dropout  # dropout 概率
        self.layerdrop = config.decoder_layerdrop  # 层间 dropout 概率
        self.padding_idx = config.pad_token_id  # 填充 token 的索引
        self.max_target_positions = config.max_position_embeddings  # 最大目标位置
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0  # 嵌入层的缩放因子
        
        # 如果提供了 embed_tokens，则使用提供的，否则创建一个新的 nn.Embedding
        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        # 创建学习得到的位置嵌入层
        self.embed_positions = MvpLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )
        
        # 创建多个 MvpDecoderLayer 层，以列表形式存储在 self.layers 中
        self.layers = nn.ModuleList([MvpDecoderLayer(config) for _ in range(config.decoder_layers)])
        
        # 对嵌入层进行 LayerNorm 处理
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        # 如果 use_prompt 为 True，则创建 prompt 相关的属性
        self.use_prompt = use_prompt
        if use_prompt:
            self.prompt_length = config.prompt_length  # 提示语长度
            self.self_attn_prompt = MvpPrompt(
                config,
                config.decoder_layers,
                config.decoder_attention_heads,
            )
            self.cross_attn_prompt = MvpPrompt(
                config,
                config.decoder_layers,
                config.decoder_attention_heads,
            )

        self.gradient_checkpointing = False  # 梯度检查点，默认为 False
        
        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入层 embed_tokens
    def get_input_embeddings(self):
        return self.embed_tokens

    # 设置输入嵌入层 embed_tokens 的值
    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # 前向传播方法，接受多种输入参数，并返回输出
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 此处未完全显示，因为代码截断了一部分，应完整显示 forward 方法的实现
        ...
    # 定义存储编码器和解码器权重的键列表
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]
    
    # 模型初始化函数，接受一个MvpConfig类型的配置参数
    def __init__(self, config: MvpConfig):
        # 调用父类的初始化方法
        super().__init__(config)
    
        # 从配置中获取填充索引和词汇表大小
        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        # 根据配置决定是否使用提示（prompt）
        self.use_prompt = config.use_prompt
        # 创建一个共享的嵌入层对象，将词汇表大小、模型维度和填充索引作为参数
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)
    
        # 创建编码器和解码器对象，传入配置、共享的嵌入层对象和是否使用提示作为参数
        self.encoder = MvpEncoder(config, self.shared, config.use_prompt)
        self.decoder = MvpDecoder(config, self.shared, config.use_prompt)
    
        # 初始化权重并进行最终处理
        self.post_init()
    
    # 获取输入嵌入层对象的方法
    def get_input_embeddings(self):
        return self.shared
    
    # 设置输入嵌入层对象的方法
    def set_input_embeddings(self, value):
        self.shared = value
        # 将编码器和解码器的嵌入层对象设置为共享的嵌入层对象
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared
    
    # 获取编码器对象的方法
    def get_encoder(self):
        return self.encoder
    
    # 获取解码器对象的方法
    def get_decoder(self):
        return self.decoder
    
    # 设置轻量级调整的方法，要求必须使用提示（prompt）
    def set_lightweight_tuning(self):
        assert self.use_prompt, "If you want to use lightweight tuning, make sure that `use_prompt=True`."
    
        # 冻结整个模型的梯度
        self.requires_grad_(False)
        # 解冻编码器和解码器中的提示（prompt）自注意力机制的权重
        self.encoder.self_attn_prompt.requires_grad_(True)
        self.decoder.self_attn_prompt.requires_grad_(True)
        self.decoder.cross_attn_prompt.requires_grad_(True)
    
    # 前向传播方法，带有详细的文档字符串注释
    @add_start_docstrings_to_model_forward(MVP_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=Seq2SeqModelOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
# 添加类的文档字符串，描述这个类用于带有语言建模头部的MVP模型，适用于各种文本生成任务。
@add_start_docstrings(
    "The MVP Model with a language modeling head. Can be used for various text generation tasks.", MVP_START_DOCSTRING
)
class MvpForConditionalGeneration(MvpPreTrainedModel):
    # 定义了权重共享的关键键名列表
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight", "lm_head.weight"]

    def __init__(self, config: MvpConfig):
        super().__init__(config)
        # 根据给定配置初始化MVP模型
        self.model = MvpModel(config)
        # 注册一个缓冲区，存储最终对数偏置，形状为(1, 共享编码器的词汇表大小)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        # 初始化线性层LM头部，输入维度为config.d_model，输出维度为共享编码器的词汇表大小，无偏置
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        # 执行初始化权重和应用最终处理
        self.post_init()

    def get_encoder(self):
        # 返回模型的编码器部分
        return self.model.get_encoder()

    def get_decoder(self):
        # 返回模型的解码器部分
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int, pad_to_multiple_of: Optional[int] = None) -> nn.Embedding:
        # 调整词嵌入的大小，并且更新最终对数偏置
        new_embeddings = super().resize_token_embeddings(new_num_tokens, pad_to_multiple_of)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        # 调整最终对数偏置的大小以匹配新的词汇表大小
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        # 返回LM头部的输出词嵌入
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        # 设置LM头部的输出词嵌入
        self.lm_head = new_embeddings

    def set_lightweight_tuning(self):
        # 设置轻量级调整，即冻结LM头部的梯度更新
        self.model.set_lightweight_tuning()
        self.lm_head.requires_grad_(False)

    # 添加模型前向传播的文档字符串，使用MVP模型输入的文档字符串
    @add_start_docstrings_to_model_forward(MVP_INPUTS_DOCSTRING)
    # 替换返回值文档字符串，输出类型为Seq2SeqLMOutput，使用_CONFIG_FOR_DOC作为配置类
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    # 添加模型前向传播结束的文档字符串，使用MVP条件生成示例文档字符串
    @add_end_docstrings(MVP_CONDITIONAL_GENERATION_EXAMPLE)
    # 此方法用于执行 Transformer 模型的前向传播。
    def forward(
        self,
        input_ids: torch.LongTensor = None,  # 输入序列的 token IDs
        attention_mask: Optional[torch.Tensor] = None,  # 输入序列的注意力掩码
        decoder_input_ids: Optional[torch.LongTensor] = None,  # 解码器输入的 token IDs
        decoder_attention_mask: Optional[torch.LongTensor] = None,  # 解码器的注意力掩码
        head_mask: Optional[torch.Tensor] = None,  # 多头注意力机制的掩码
        decoder_head_mask: Optional[torch.Tensor] = None,  # 解码器的多头注意力机制的掩码
        cross_attn_head_mask: Optional[torch.Tensor] = None,  # 跨层注意力机制的多头掩码
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,  # 编码器的输出
        past_key_values: Optional[List[torch.FloatTensor]] = None,  # 过去的键值对，用于生成
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入嵌入向量
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,  # 解码器输入嵌入向量
        labels: Optional[torch.LongTensor] = None,  # 真实标签
        use_cache: Optional[bool] = None,  # 是否使用缓存
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回结果字典
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
            Returns either a tuple or a `Seq2SeqLMOutput` containing masked language model output.

        """
        # Determine whether to use the return_dict based on provided argument or default configuration
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Adjust use_cache if labels are provided, and issue a warning if necessary
        if labels is not None:
            if use_cache:
                logger.warning("The `use_cache` argument is changed to `False` since `labels` is provided.")
            use_cache = False
            # If decoder input is not provided, shift labels for decoder input preparation
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        # Pass the inputs to the model for computation
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        # Compute logits for the language model head and add final logits bias
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        # Calculate masked language model loss if labels are provided
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        # Return the appropriate output format based on return_dict flag
        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # Return Seq2SeqLMOutput containing detailed output components
        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )
    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # 如果使用了过去的键值（用于生成过程中），则裁剪decoder_input_ids
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # 一些生成方法已经只传递最后一个输入ID
            if decoder_input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 默认行为：仅保留最后一个ID
                remove_prefix_length = decoder_input_ids.shape[1] - 1

            # 裁剪掉不需要的前缀部分
            decoder_input_ids = decoder_input_ids[:, remove_prefix_length:]

        # 返回准备好的输入字典
        return {
            "input_ids": None,  # encoder_outputs已定义，不需要input_ids
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # 更改此处以避免缓存（可能是为了调试目的）
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        # 将标签右移一个位置以作为解码器的输入
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            # 缓存的交叉注意力状态无需重新排序 -> 它们始终保持不变
            reordered_past += (
                # 根据beam_idx对层的过去状态进行重新排序
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past[:2])
                + layer_past[2:],
            )
        return reordered_past
# 添加模型文档字符串，用于描述带有顶部序列分类/头部（池化输出顶部的线性层）的 MVP 模型，例如用于 GLUE 任务。
@add_start_docstrings(
    """
    Mvp model with a sequence classification/head on top (a linear layer on top of the pooled output) e.g. for GLUE
    tasks.
    """,
    MVP_START_DOCSTRING,
)
class MvpForSequenceClassification(MvpPreTrainedModel):
    # 定义权重共享的关键键列表
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config: MvpConfig, **kwargs):
        # 调用父类构造函数
        super().__init__(config, **kwargs)
        # 初始化 MvpModel
        self.model = MvpModel(config)
        # 初始化 MvpClassificationHead，用于分类任务
        self.classification_head = MvpClassificationHead(
            config.d_model,
            config.d_model,
            config.num_labels,
            config.classifier_dropout,
        )

        # 初始化权重并进行最终处理
        self.post_init()

    # 设置轻量级调优
    def set_lightweight_tuning(self):
        self.model.set_lightweight_tuning()
        # 设置分类头部的梯度为 False，以便冻结它
        self.classification_head.requires_grad_(False)

    # 添加模型前向传播的文档字符串
    @add_start_docstrings_to_model_forward(MVP_INPUTS_DOCSTRING)
    # 添加模型前向传播的结束文档字符串，用于序列分类任务的示例
    @add_end_docstrings(MVP_SEQUENCE_CLASSIFICATION_SAMPLE)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,



# 添加模型文档字符串，用于描述带有顶部抽取式问答任务的 MVP 模型，例如用于 SQuAD 的线性层输出隐藏状态以计算 `span start logits` 和 `span end logits`。
@add_start_docstrings(
    """
    MVP Model with a span classification head on top for extractive question-answering tasks like SQuAD (a linear layer
    on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    MVP_START_DOCSTRING,
)
class MvpForQuestionAnswering(MvpPreTrainedModel):
    # 定义权重共享的关键键列表
    _tied_weights_keys = ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]

    def __init__(self, config):
        # 调用父类构造函数
        super().__init__(config)

        # 设置问题回答任务的标签数量为 2
        config.num_labels = 2
        self.num_labels = config.num_labels

        # 初始化 MvpModel
        self.model = MvpModel(config)
        # 初始化用于问答任务的线性层
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并进行最终处理
        self.post_init()

    # 设置轻量级调优
    def set_lightweight_tuning(self):
        self.model.set_lightweight_tuning()
        # 设置问答任务线性层的梯度为 False，以便冻结它
        self.qa_outputs.requires_grad_(False)

    # 添加模型前向传播的文档字符串
    @add_start_docstrings_to_model_forward(MVP_INPUTS_DOCSTRING)
    # 添加模型前向传播的结束文档字符串，用于抽取式问答任务的示例
    @add_end_docstrings(MVP_QUESTION_ANSWERING_SAMPLE)
    # 定义 Transformer 模型的前向传播方法，用于执行模型推断或训练时的前向计算
    def forward(
        self,
        input_ids: torch.Tensor = None,  # 输入序列的 token IDs
        attention_mask: Optional[torch.Tensor] = None,  # 输入序列的注意力掩码，标记哪些位置是真实的 token
        decoder_input_ids: Optional[torch.LongTensor] = None,  # 解码器输入序列的 token IDs
        decoder_attention_mask: Optional[torch.LongTensor] = None,  # 解码器输入序列的注意力掩码
        head_mask: Optional[torch.Tensor] = None,  # 多头注意力机制的掩码
        decoder_head_mask: Optional[torch.Tensor] = None,  # 解码器多头注意力机制的掩码
        cross_attn_head_mask: Optional[torch.Tensor] = None,  # 跨注意力头的掩码
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,  # 编码器输出的列表，包含不同层的隐藏状态
        start_positions: Optional[torch.LongTensor] = None,  # 序列开始位置的索引
        end_positions: Optional[torch.LongTensor] = None,  # 序列结束位置的索引
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入的嵌入表示
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,  # 解码器输入的嵌入表示
        use_cache: Optional[bool] = None,  # 是否使用缓存进行推断
        output_attentions: Optional[bool] = None,  # 是否返回注意力权重
        output_hidden_states: Optional[bool] = None,  # 是否返回所有隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回字典形式的输出
        ):
# 从 transformers.models.mvp.modeling_mvp.MvpDecoderWrapper 复制，并将 Bart 改为 Mvp
class MvpDecoderWrapper(MvpPreTrainedModel):
    """
    这个包装类是一个辅助类，用于在使用 [`EncoderDecoderModel`] 框架与因果语言模型结合时正确加载预训练检查点。
    """

    def __init__(self, config):
        super().__init__(config)
        # 初始化 MvpDecoder 对象
        self.decoder = MvpDecoder(config)

    def forward(self, *args, **kwargs):
        # 调用 MvpDecoder 的 forward 方法
        return self.decoder(*args, **kwargs)


class MvpForCausalLM(MvpPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        config = copy.deepcopy(config)
        # 标记该模型为解码器，非编码器解码器结构
        config.is_decoder = True
        config.is_encoder_decoder = False
        super().__init__(config)
        # 使用 MvpDecoderWrapper 封装模型
        self.model = MvpDecoderWrapper(config)

        # 初始化语言模型头部，一个线性层
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        # 返回模型中的嵌入层
        return self.model.decoder.embed_tokens

    def set_input_embeddings(self, value):
        # 设置模型中的嵌入层
        self.model.decoder.embed_tokens = value

    def get_output_embeddings(self):
        # 返回语言模型头部的嵌入层
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        # 设置语言模型头部的嵌入层
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        # 设置解码器
        self.model.decoder = decoder

    def get_decoder(self):
        # 返回解码器
        return self.model.decoder

    def set_lightweight_tuning(self):
        # 设置轻量级调整
        self.model.set_lightweight_tuning()
        # 冻结语言模型头部的梯度
        self.lm_head.requires_grad_(False)

    @replace_return_docstrings(output_type=CausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        # 模型前向传播方法，支持参数详见函数签名
        ...

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, use_cache=None, **kwargs
    ):
        # 为生成准备输入的方法，支持参数详见函数签名
        ...
    ):
        # 如果模型作为编码器-解码器模型的解码器使用，解码器注意力遮罩在需要时动态创建
        if attention_mask is None:
            # 如果注意力遮罩为空，创建一个全为1的新张量，形状与input_ids相同
            attention_mask = input_ids.new_ones(input_ids.shape)

        if past_key_values:
            # 获取过去键值的长度（通常是序列长度）
            past_length = past_key_values[0][0].shape[2]

            # 一些生成方法已经只传递最后一个输入ID
            if input_ids.shape[1] > past_length:
                # 如果输入ID序列长度大于过去长度，计算需要移除的前缀长度
                remove_prefix_length = past_length
            else:
                # 否则，默认行为：仅保留最后一个ID
                remove_prefix_length = input_ids.shape[1] - 1

            # 从输入ID中移除前缀部分，保留剩余部分作为新的输入ID
            input_ids = input_ids[:, remove_prefix_length:]
        # 返回字典，包含模型需要的输入信息和状态信息
        return {
            "input_ids": input_ids,  # encoder_outputs is defined. input_ids not needed
            "attention_mask": attention_mask,  # 注意力遮罩，用于指定哪些位置需要关注
            "past_key_values": past_key_values,  # 过去的键值状态，用于生成模型的历史信息
            "use_cache": use_cache,  # 是否使用缓存，以提高生成效率
        }

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            # 根据beam_idx重新排列过去的状态信息
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        # 返回重新排列后的过去状态
        return reordered_past
```