# `.\models\codegen\modeling_codegen.py`

```
# coding=utf-8
# Copyright 2022 Salesforce authors, The EleutherAI, and HuggingFace Teams. All rights reserved.
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
""" PyTorch CodeGen model."""

from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_codegen import CodeGenConfig


logger = logging.get_logger(__name__)

_CHECKPOINT_FOR_DOC = "Salesforce/codegen-2B-mono"
_CONFIG_FOR_DOC = "CodeGenConfig"


CODEGEN_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "Salesforce/codegen-350M-nl",
    "Salesforce/codegen-350M-multi",
    "Salesforce/codegen-350M-mono",
    "Salesforce/codegen-2B-nl",
    "Salesforce/codegen-2B-multi",
    "Salesforce/codegen-2B-mono",
    "Salesforce/codegen-6B-nl",
    "Salesforce/codegen-6B-multi",
    "Salesforce/codegen-6B-mono",
    "Salesforce/codegen-16B-nl",
    "Salesforce/codegen-16B-multi",
    "Salesforce/codegen-16B-mono",
    # See all CodeGen models at https://huggingface.co/models?filter=codegen
]


# Copied from transformers.models.gptj.modeling_gptj.create_sinusoidal_positions
# 创建一个张量，包含给定维度和长度的正弦位置编码
def create_sinusoidal_positions(num_pos: int, dim: int) -> torch.Tensor:
    inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.int64) / dim))
    sinusoid_inp = torch.einsum("i , j -> i j", torch.arange(num_pos, dtype=torch.int64).float(), inv_freq).float()
    return torch.cat((torch.sin(sinusoid_inp), torch.cos(sinusoid_inp)), dim=1)


# Copied from transformers.models.gptj.modeling_gptj.rotate_every_two
# 旋转张量的每两个元素
def rotate_every_two(x: torch.Tensor) -> torch.Tensor:
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')


# Copied from transformers.models.gptj.modeling_gptj.apply_rotary_pos_emb
# 应用旋转位置编码到输入张量
def apply_rotary_pos_emb(tensor: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    sin = torch.repeat_interleave(sin[:, :, None, :], 2, 3)
    cos = torch.repeat_interleave(cos[:, :, None, :], 2, 3)
    return (tensor * cos) + (rotate_every_two(tensor) * sin)


class CodeGenAttention(nn.Module):
    # 初始化方法，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()

        # 从配置对象中获取最大位置嵌入数
        max_positions = config.max_position_embeddings
        # 注册一个缓冲区，用来存储因果掩码
        self.register_buffer(
            "causal_mask",
            # 创建一个下三角形状的布尔类型张量作为因果掩码
            torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool)).view(
                1, 1, max_positions, max_positions
            ),
            persistent=False,
        )

        # 定义注意力权重的 dropout 层
        self.attn_dropout = nn.Dropout(config.attn_pdrop)
        # 定义残差连接的 dropout 层
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        # 设置嵌入维度
        self.embed_dim = config.hidden_size
        # 设置注意力头的数量
        self.num_attention_heads = config.num_attention_heads
        # 计算每个注意力头的维度
        self.head_dim = self.embed_dim // self.num_attention_heads
        # 检查是否能够均分嵌入维度到每个注意力头
        if self.head_dim * self.num_attention_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_attention_heads (got `embed_dim`: {self.embed_dim} and"
                f" `num_attention_heads`: {self.num_attention_heads})."
            )
        # 缩放因子，用于缩放注意力得分
        self.scale_attn = torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32)).to(torch.get_default_dtype())
        # 定义 qkv 投影层
        self.qkv_proj = nn.Linear(self.embed_dim, self.embed_dim * 3, bias=False)

        # 定义输出投影层
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        # 如果有旋转维度，则使用它；否则使用嵌入维度
        self.rotary_dim = config.rotary_dim
        pos_embd_dim = self.rotary_dim or self.embed_dim
        # 创建正弦位置编码
        self.embed_positions = create_sinusoidal_positions(max_positions, pos_embd_dim)

    # 将张量按注意力头拆分
    def _split_heads(self, x, n_head, dim_head, mp_num):
        reshaped = x.reshape(x.shape[:-1] + (n_head // mp_num, dim_head))
        reshaped = reshaped.reshape(x.shape[:-2] + (-1,) + reshaped.shape[-1:])
        return reshaped

    # 将分开的注意力头合并回张量
    def _merge_heads(self, tensor, num_attention_heads, attn_head_size):
        """
        Merges attn_head_size dim and num_attn_heads dim into n_ctx
        """
        # 根据张量的维度进行不同的维度置换操作
        if len(tensor.shape) == 5:
            tensor = tensor.permute(0, 1, 3, 2, 4).contiguous()
        elif len(tensor.shape) == 4:
            tensor = tensor.permute(0, 2, 1, 3).contiguous()
        else:
            raise ValueError(f"Input tensor rank should be one of [4, 5], but is: {len(tensor.shape)}")
        # 重新构造张量的形状，将注意力头和注意力头大小维度合并
        new_shape = tensor.size()[:-2] + (num_attention_heads * attn_head_size,)
        return tensor.view(new_shape)

    # 注意力计算函数
    def _attn(
        self,
        query,
        key,
        value,
        attention_mask=None,
        head_mask=None,
        # 计算因果掩码，基于因果掩码缓冲区
        query_length, key_length = query.size(-2), key.size(-2)
        causal_mask = self.causal_mask[:, :, key_length - query_length : key_length, :key_length]

        # 将查询和键的数据类型转换为 float32，以避免溢出问题
        query = query.to(torch.float32)
        key = key.to(torch.float32)

        # 计算注意力权重
        attn_weights = torch.matmul(query, key.transpose(-1, -2))

        # 缩放注意力权重
        attn_weights = attn_weights / self.scale_attn

        # 设置掩码值，避免数据类型不匹配和设备不一致的错误
        mask_value = torch.finfo(attn_weights.dtype).min
        mask_value = torch.tensor(mask_value, dtype=attn_weights.dtype).to(attn_weights.device)

        # 应用因果掩码
        attn_weights = torch.where(causal_mask, attn_weights, mask_value)

        if attention_mask is not None:
            # 应用额外的注意力掩码
            attn_weights = attn_weights + attention_mask

        # 对注意力权重进行 softmax 归一化
        attn_weights = nn.Softmax(dim=-1)(attn_weights)

        # 将注意力权重转换回与值张量相同的数据类型
        attn_weights = attn_weights.to(value.dtype)

        # 应用注意力 dropout
        attn_weights = self.attn_dropout(attn_weights)

        # 如果有头部掩码，应用头部掩码
        if head_mask is not None:
            attn_weights = attn_weights * head_mask

        # 计算最终的注意力输出
        attn_output = torch.matmul(attn_weights, value)

        return attn_output, attn_weights
        ]:
        # 使用 self.qkv_proj 方法处理隐藏状态以获取查询、键、值（QKV）的投影
        qkv = self.qkv_proj(hidden_states)
        # 定义每个 TPU-v4 逻辑核的数量为 4
        # TODO(enijkamp): factor out number of logical TPU-v4 cores or make forward pass agnostic
        mp_num = 4
        # 将 QKV 张量重塑为形状为 (batch_size, seq_length, mp_num, local_dim) 的张量
        qkv_split = qkv.reshape(qkv.shape[:-1] + (mp_num, -1))

        # 计算每个局部区域的维度
        local_dim = self.head_dim * self.num_attention_heads // mp_num
        # 按照局部维度将 QKV 张量分割成查询、键、值
        query, value, key = torch.split(qkv_split, local_dim, dim=-1)
        # 将查询分割为多个头并重新排列
        query = self._split_heads(query, self.num_attention_heads, self.head_dim, mp_num=mp_num)
        # 将键分割为多个头并重新排列
        key = self._split_heads(key, self.num_attention_heads, self.head_dim, mp_num=mp_num)

        # 将值分割为多个头并重新排列
        value = self._split_heads(value, self.num_attention_heads, self.head_dim, mp_num=mp_num)
        # 对值进行维度变换，使得维度顺序变为 (batch_size, num_heads, seq_length, head_dim)
        value = value.permute(0, 2, 1, 3)

        # 获取嵌入位置信息
        embed_positions = self.embed_positions
        # 如果嵌入位置信息与位置 ID 的设备不匹配，则将嵌入位置信息移动到位置 ID 的设备上
        if embed_positions.device != position_ids.device:
            embed_positions = embed_positions.to(position_ids.device)
            self.embed_positions = embed_positions

        # 根据位置 ID 从嵌入位置信息中提取出 sin 和 cos 组成的张量 sincos
        sincos = embed_positions[position_ids]
        # 将 sincos 张量按照最后一个维度分割为 sin 和 cos 两部分
        sin, cos = torch.split(sincos, sincos.shape[-1] // 2, dim=-1)

        # 如果存在旋转维度，则分别对键和查询的旋转部分应用旋转位置编码
        if self.rotary_dim is not None:
            k_rot = key[:, :, :, : self.rotary_dim]
            k_pass = key[:, :, :, self.rotary_dim :]

            q_rot = query[:, :, :, : self.rotary_dim]
            q_pass = query[:, :, :, self.rotary_dim :]

            k_rot = apply_rotary_pos_emb(k_rot, sin, cos)
            q_rot = apply_rotary_pos_emb(q_rot, sin, cos)

            # 合并旋转后的部分和原始部分
            key = torch.cat([k_rot, k_pass], dim=-1)
            query = torch.cat([q_rot, q_pass], dim=-1)
        else:
            # 否则，对整个键和查询都应用旋转位置编码
            key = apply_rotary_pos_emb(key, sin, cos)
            query = apply_rotary_pos_emb(query, sin, cos)

        # 将键和查询的维度顺序调整为 (batch_size, num_heads, seq_length, head_dim)
        key = key.permute(0, 2, 1, 3)
        query = query.permute(0, 2, 1, 3)

        # 如果存在过去的层状态，则将过去的键和值与当前的键和值拼接起来
        if layer_past is not None:
            past_key = layer_past[0]
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-2)
            value = torch.cat((past_value, value), dim=-2)

        # 如果使用缓存，则返回用于下一步计算的 present
        if use_cache is True:
            # 注意这里的类型转换比较丑陋，但是在原始代码中，k_rot 一直是 fp32 类型
            # 参考链接：https://github.com/salesforce/CodeGen/blob/f210c3bb1216c975ad858cd4132c0fdeabf4bfc2/codegen1/jaxformer/hf/codegen/modeling_codegen.py#L38
            present = (key.to(hidden_states.dtype), value)
        else:
            present = None

        # 计算自注意力机制，得到输出和注意力权重
        attn_output, attn_weights = self._attn(query, key, value, attention_mask, head_mask)

        # 将多头自注意力机制的输出合并
        attn_output = self._merge_heads(attn_output, self.num_attention_heads, self.head_dim)
        # 使用输出投影层进行投影
        attn_output = self.out_proj(attn_output)
        # 应用残差连接的 dropout
        attn_output = self.resid_dropout(attn_output)

        # 返回输出结果以及可能的 present 和注意力权重
        outputs = (attn_output, present)
        if output_attentions:
            outputs += (attn_weights,)

        return outputs  # 返回输出元组 (attn_output, present, (attentions))
# Copied from transformers.models.gptj.modeling_gptj.GPTJMLP with GPTJ->CodeGen
class CodeGenMLP(nn.Module):
    def __init__(self, intermediate_size, config):  # in MLP: intermediate_size= 4 * embed_dim
        super().__init__()
        embed_dim = config.n_embd

        # 定义输入层全连接网络，将输入维度调整为intermediate_size
        self.fc_in = nn.Linear(embed_dim, intermediate_size)
        # 定义输出层全连接网络，将输出维度调整为embed_dim
        self.fc_out = nn.Linear(intermediate_size, embed_dim)

        # 激活函数选择，根据配置文件中的激活函数名选择对应的激活函数
        self.act = ACT2FN[config.activation_function]
        # Dropout层，根据配置中的残差概率添加dropout
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states: Optional[torch.FloatTensor]) -> torch.FloatTensor:
        # 输入数据经过输入层全连接网络
        hidden_states = self.fc_in(hidden_states)
        # 经过激活函数处理
        hidden_states = self.act(hidden_states)
        # 经过输出层全连接网络
        hidden_states = self.fc_out(hidden_states)
        # 经过dropout处理
        hidden_states = self.dropout(hidden_states)
        return hidden_states


# Copied from transformers.models.gptj.modeling_gptj.GPTJBlock with GPTJ->CodeGen
class CodeGenBlock(nn.Module):
    # Ignore copy
    def __init__(self, config):
        super().__init__()
        # 内部维度的设定，若未指定则默认为4倍的嵌入维度
        inner_dim = config.n_inner if config.n_inner is not None else 4 * config.n_embd
        # LayerNorm层，对输入进行归一化处理
        self.ln_1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        # 自注意力机制模块
        self.attn = CodeGenAttention(config)
        # 多层感知机模块
        self.mlp = CodeGenMLP(inner_dim, config)

    def forward(
        self,
        hidden_states: Optional[torch.FloatTensor],
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        # 残差连接，保存输入状态
        residual = hidden_states
        # 对输入状态进行LayerNorm归一化处理
        hidden_states = self.ln_1(hidden_states)
        # 使用自注意力机制模块进行注意力计算
        attn_outputs = self.attn(
            hidden_states=hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        # 获取自注意力机制模块的输出
        attn_output = attn_outputs[0]  # output_attn: a, present, (attentions)
        outputs = attn_outputs[1:]

        # 经过多层感知机模块处理后的隐藏状态
        feed_forward_hidden_states = self.mlp(hidden_states)
        # 经过残差连接后的隐藏状态
        hidden_states = attn_output + feed_forward_hidden_states + residual

        if use_cache:
            # 若使用缓存，则将隐藏状态作为输出的一部分
            outputs = (hidden_states,) + outputs
        else:
            # 若不使用缓存，则仅保留后续输出部分
            outputs = (hidden_states,) + outputs[1:]

        return outputs  # 返回隐藏状态、present、(注意力)
    # 设定跳过设备放置的键名，用于特定用途（如处理键值对）
    _skip_keys_device_placement = "past_key_values"

    # 初始化函数，接受任意位置参数和关键字参数
    def __init__(self, *inputs, **kwargs):
        # 调用父类的初始化方法
        super().__init__(*inputs, **kwargs)

    # 初始化模型参数函数
    def _init_weights(self, module):
        """Initialize the weights."""
        # 如果模块是线性层（nn.Linear 类型）
        if isinstance(module, (nn.Linear,)):
            # 使用正态分布初始化权重，均值为 0，标准差为模型配置中的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果模块有偏置项，则将其初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果模块是嵌入层（nn.Embedding 类型）
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化权重，均值为 0，标准差为模型配置中的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果模块设置了填充索引，将填充索引处的权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # 如果模块是层归一化层（nn.LayerNorm 类型）
        elif isinstance(module, nn.LayerNorm):
            # 将偏置项初始化为零
            module.bias.data.zero_()
            # 将权重初始化为全1
            module.weight.data.fill_(1.0)
# 生成一个长字符串，用作代码文档字符串的起始部分，描述了这个模型是一个 PyTorch 的子类，应当像普通的 PyTorch Module 一样使用。
# 提供了一个链接到 PyTorch 文档的参考。
CODEGEN_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`CodeGenConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 生成一个空字符串，用作描述代码文档字符串的输入部分，留待进一步填充和完善。
CODEGEN_INPUTS_DOCSTRING = r"""
"""
    Args:
        input_ids (`torch.LongTensor` of shape `({0})`):
            # 输入序列的token索引，用于词汇表中的标识符
            Indices of input sequence tokens in the vocabulary.
            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
            [What are input IDs?](../glossary#input-ids)

        attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
            # 遮罩，用于避免在填充token索引上执行注意力操作
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)

        token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 段token索引，指示输入的第一和第二部分
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:

            - 0 corresponds to a *sentence A* token,
            - 1 corresponds to a *sentence B* token.
            [What are token type IDs?](../glossary#token-type-ids)

        position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
            # 每个输入序列token在位置嵌入中的位置索引
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0, config.n_positions - 1]`.
            [What are position IDs?](../glossary#position-ids)

        head_mask (`torch.FloatTensor` of shape `(num_attention_heads,)` or `(n_layer, num_attention_heads)`, *optional*):
            # 用于屏蔽自注意力模块中的特定头部的遮罩
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_dim)`, *optional*):
            # 可选参数，允许直接传递嵌入表示而不是`input_ids`
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert *input_ids* indices into associated vectors than the
            model's internal embedding lookup matrix.

        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.

        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.

        return_dict (`bool`, *optional*):
            # 是否返回一个`ModelOutput`对象而不是普通元组
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""
@add_start_docstrings(
    "The bare CodeGen Model transformer outputting raw hidden-states without any specific head on top.",
    CODEGEN_START_DOCSTRING,
)
"""
class CodeGenModel(CodeGenPreTrainedModel):
    """
    Implementing a transformer model for code generation without additional task-specific heads.

    Args:
        config: The configuration class for model initialization.

    Attributes:
        embed_dim (int): Dimensionality of the embedding layer.
        vocab_size (int): Size of the vocabulary.
        wte (nn.Embedding): Embedding layer to convert input tokens to embeddings.
        drop (nn.Dropout): Dropout layer for regularization.
        h (nn.ModuleList): List of CodeGenBlock modules representing transformer layers.
        ln_f (nn.LayerNorm): Layer normalization for final layer.
        rotary_dim (int): Dimension for rotary position encodings.
        gradient_checkpointing (bool): Whether to use gradient checkpointing during training.

    Methods:
        get_input_embeddings(): Returns the input embedding layer.
        set_input_embeddings(new_embeddings): Sets new input embeddings for the model.
        forward(...): Performs forward pass through the model with various input tensors.
    """

    def __init__(self, config):
        super().__init__(config)

        self.embed_dim = config.n_embd
        self.vocab_size = config.vocab_size
        self.wte = nn.Embedding(config.vocab_size, self.embed_dim)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.h = nn.ModuleList([CodeGenBlock(config) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=config.layer_norm_epsilon)
        self.rotary_dim = min(config.rotary_dim, config.n_ctx // config.num_attention_heads)

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()


@add_start_docstrings(
    """
    The CodeGen Model transformer with a language modeling head on top.
    """,
    CODEGEN_START_DOCSTRING,
)
"""
class CodeGenForCausalLM(CodeGenPreTrainedModel):
    """
    Extended transformer model for code generation with an added language modeling head.

    Args:
        config: The configuration class for model initialization.

    Attributes:
        transformer (CodeGenModel): Instance of the base CodeGenModel transformer.
        lm_head (nn.Linear): Linear layer for language modeling predictions.

    Methods:
        get_output_embeddings(): Returns the output embedding layer.
        set_output_embeddings(new_embeddings): Sets new output embeddings for the model.
    """

    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.transformer = CodeGenModel(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, **kwargs):
        # 从 kwargs 中获取 token_type_ids，如果不存在则为 None
        token_type_ids = kwargs.get("token_type_ids", None)
        # 如果 past_key_values 不为 None，则执行以下操作
        if past_key_values:
            # 获取 past_key_values 的长度信息
            past_length = past_key_values[0][0].shape[2]

            # 如果 input_ids 的第二维度大于 past_length，则移除前缀长度为 past_length
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 否则默认行为：保留最后一个输入 ID
                remove_prefix_length = input_ids.shape[1] - 1

            # 截取 input_ids 的第二维度，保留从 remove_prefix_length 到结尾的部分
            input_ids = input_ids[:, remove_prefix_length:]
            # 如果 token_type_ids 不为 None，则截取与 input_ids 相同长度的部分
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -input_ids.shape[1] :]

        # 从 kwargs 中获取 attention_mask 和 position_ids
        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        # 如果 attention_mask 不为 None 且 position_ids 为 None，则执行以下操作
        if attention_mask is not None and position_ids is None:
            # 动态生成 position_ids 以用于批量生成
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            # 如果 past_key_values 不为 None，则截取与 input_ids 相同长度的部分
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # 返回准备好的输入字典
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
        }

    @add_start_docstrings_to_model_forward(CODEGEN_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutputWithPast,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        # 如果未指定 return_dict，则使用配置中的设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 Transformer 处理输入的各种参数，并获取输出
        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 获取 Transformer 输出中的隐藏状态
        hidden_states = transformer_outputs[0]

        # 确保在 fp16 下的采样工作正常，并且在 fp32 下计算损失，以与 mesh-tf 版本保持一致
        # 参考链接: https://github.com/EleutherAI/gpt-neo/blob/89ce74164da2fb16179106f54e2269b5da8db333/models/gpt2/gpt2.py#L179
        # 将隐藏状态传递给语言模型头部，转换为 torch.float32 类型
        lm_logits = self.lm_head(hidden_states).to(torch.float32)

        # 初始化损失为 None
        loss = None
        if labels is not None:
            # 将标签移动到正确的设备上以启用模型并行处理
            labels = labels.to(lm_logits.device)
            # 移动 logits 以便让 tokens < n 预测 n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # 展平 tokens
            loss_fct = CrossEntropyLoss()
            # 计算损失
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            # 将损失转换为与隐藏状态类型相同的类型
            loss = loss.to(hidden_states.dtype)

        # 如果不使用 return_dict，则输出的格式为元组
        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果使用 return_dict，则返回 CausalLMOutputWithPast 类型的对象，包含损失、logits 和其他 Transformer 的输出
        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    @staticmethod
    def _reorder_cache(
        past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
        """
        如果调用了 `PretrainedModel.beam_search` 或 `PretrainedModel.beam_sample`，则使用此函数重新排序 `past_key_values` 缓存。
        这是为了确保在每一代生成步骤中，`past_key_values` 与正确的 `beam_idx` 匹配。

        返回一个元组，其中包含重新排序后的 `past_key_values`，每个元素都是一个元组，每个元组包含一组 `torch.Tensor` 对象。
        """
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past_key_values
        )
```