# `.\transformers\models\clvp\modeling_clvp.py`

```
# 设置文件编码为 UTF-8
# 版权声明
# 版权所有 © 2023 年 HuggingFace 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）获得许可;
# 除非符合许可证，否则您不得使用此文件。
# 您可以在以下位置获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”分发软件，
# 没有任何明示或暗示的保证或条件。
# 有关详细信息，请参阅许可证。
#
# PyTorch CLVP 模型。

import copy  # 导入拷贝模块
import math  # 导入数学模块
from dataclasses import dataclass  # 导入数据类装饰器
from typing import Dict, Optional, Tuple, Union  # 导入类型提示

import torch  # 导入 PyTorch 库
import torch.utils.checkpoint  # 导入 PyTorch 的检查点工具
from torch import nn  # 导入 PyTorch 的神经网络模块
from torch.nn import CrossEntropyLoss  # 导入交叉熵损失函数

from ...activations import ACT2FN  # 从激活函数中导入激活函数映射
from ...generation import GenerationConfig  # 导入生成配置类
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask  # 导入注意力掩码工具函数
from ...modeling_outputs import (  # 导入模型输出类
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPooling,
    CausalLMOutputWithCrossAttentions,
)
from ...modeling_utils import PreTrainedModel, SequenceSummary  # 导入模型工具函数
from ...pytorch_utils import Conv1D  # 导入一维卷积层
from ...utils import (  # 导入工具函数
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

from .configuration_clvp import (  # 从 CLVP 配置模块中导入配置类
    ClvpConfig,
    ClvpDecoderConfig,
    ClvpEncoderConfig,
)

logger = logging.get_logger(__name__)  # 获取日志记录器

_CHECKPOINT_FOR_DOC = "susnato/clvp_dev"  # 预训练模型检查点
# CLVP 预训练模型存档列表
CLVP_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "susnato/clvp_dev",
    # 查看所有 CLVP 模型：https://huggingface.co/models?filter=clvp
]

# 从 transformers.models.clip.modeling_clip.contrastive_loss 复制的对比损失函数
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))

# 从 transformers.models.clip.modeling_clip.clip_loss 复制的 CLVP 损失函数，将 clip->clvp, image_loss->speech_loss
def clvp_loss(similarity: torch.Tensor) -> torch.Tensor:
    caption_loss = contrastive_loss(similarity)  # 计算标题损失
    speech_loss = contrastive_loss(similarity.t())  # 计算语音损失
    return (caption_loss + speech_loss) / 2.0  # 返回标题损失和语音损失的平均值

# 从 transformers.models.llama.modeling_llama.rotate_half 复制的旋转一半隐藏维度的函数
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]  # 获取输入张量的前一半
    x2 = x[..., x.shape[-1] // 2 :]  # 获取输入张量的后一半
    return torch.cat((-x2, x1), dim=-1)  # 拼接后一半和前一半，作为旋转结果

# 应用旋转位置编码到查询和键张量的函数
def apply_rotary_pos_emb(q, k, v, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors."""
    Args:
        q (`torch.Tensor`): 查询张量。
        k (`torch.Tensor`): 键张量。
        cos (`torch.Tensor`): 旋转嵌入的余弦部分。
        sin (`torch.Tensor`): 旋转嵌入的正弦部分。
        position_ids (`torch.Tensor`):
            对应于查询和键张量的标记位置索引。例如，这可以用于在使用 KV 缓存时传递偏移的位置 id。
        unsqueeze_dim (`int`, *optional*, 默认为 1):
            'unsqueeze_dim' 参数指定了要对 cos[position_ids] 和 sin[position_ids] 进行展开的维度，以便它们可以正确广播到 q 和 k 的维度。例如，注意 cos[position_ids] 和 sin[position_ids] 的形状为 [batch_size, seq_len, head_dim]。然后，如果 q 和 k 的形状为 [batch_size, heads, seq_len, head_dim]，则设置 unsqueeze_dim=1 使得 cos[position_ids] 和 sin[position_ids] 可以广播到 q 和 k 的形状。类似地，如果 q 和 k 的形状为 [batch_size, seq_len, heads, head_dim]，则设置 unsqueeze_dim=2。
    Returns:
        返回经过旋转位置嵌入的查询和键张量组成的元组。
    """
    # 根据位置 id 展开余弦和正弦部分的张量，并保持维度一致以便广播
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    # 使用旋转嵌入对查询张量进行旋转
    q_embed = (q * cos) + (rotate_half(q) * sin)
    # 使用旋转嵌入对键张量进行旋转
    k_embed = (k * cos) + (rotate_half(k) * sin)
    # 使用旋转嵌入对值张量进行旋转
    v_embed = (v * cos) + (rotate_half(v) * sin)
    # 返回旋转后的查询、键和值张量
    return q_embed, k_embed, v_embed
# 为输入的 input_ids 添加额外的 bos 和 eos 标记，并相应修改用于 ClvpConditioningEncoder 和 ClvpModelForConditionalGeneration 生成循环的 attention_mask
def _pad_extra_bos_eos_tokens(
    input_ids,
    attention_mask=None,
    pad_token_id=0,
    bos_token_id=255,
    eos_token_id=0,
    add_bos_token=True,
    add_eos_token=True,
):
    """
    This method adds extra bos and eos tokens to input_ids and accordingly modifies the attention_mask which is used in
    `ClvpConditioningEncoder` and the generation loop of the `ClvpModelForConditionalGeneration`.
    """

    # 在开头添加 bos 标记
    if add_bos_token:
        input_ids = torch.nn.functional.pad(input_ids, (1, 0), value=bos_token_id)
        attention_mask = (
            torch.nn.functional.pad(attention_mask, (1, 0), value=1) if attention_mask is not None else attention_mask
        )

    modified_input_ids = input_ids
    if add_eos_token:
        # 创建一个新的 tensor 用于存储修改后的 input_ids
        modified_input_ids = torch.zeros(
            (input_ids.shape[0], input_ids.shape[1] + 1), dtype=input_ids.dtype, device=input_ids.device
        )
        for i, each_input_id in enumerate(input_ids):
            # 定位有效标记的结束位置，然后添加 eos 标记
            if torch.isin(each_input_id, pad_token_id).sum():
                pos = torch.where(each_input_id == pad_token_id)[0].min()
                modified_input_ids[i] = torch.concatenate(
                    [each_input_id[:pos], torch.tensor([eos_token_id], device=input_ids.device), each_input_id[pos:]]
                )
            else:
                # 如果没有填充标记存在，则在末尾添加 eos
                modified_input_ids[i] = torch.nn.functional.pad(each_input_id, (0, 1), value=eos_token_id)
        attention_mask = (
            torch.nn.functional.pad(attention_mask, (1, 0), value=1) if attention_mask is not None else attention_mask
        )

    return modified_input_ids, attention_mask


@dataclass
class ClvpEncoderOutput(ModelOutput):
    """
    Base class for CLVP encoder's outputs that contains a pooling of the last hidden states as well as a projection
    output (a linear layer on top of the pooled output).
    Args:
        embeds (`torch.FloatTensor` of shape `(batch_size, output_dim)`, *optional*, returned when model is initialized with `with_projection=True`):
            The embeddings obtained by applying the projection layer to the pooler_output.
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            The hidden state of the last layer of the model.
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
            Pooled output of the `last_hidden_state`.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of
            the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`. Attentions weights after the attention softmax, used to compute the weighted average in
            the self-attention heads.
    """

    # Optional tensor for embeddings
    embeds: Optional[torch.FloatTensor] = None
    # Tensor for the hidden state of the last layer
    last_hidden_state: torch.FloatTensor = None
    # Optional tensor for pooled output
    pooler_output: Optional[torch.FloatTensor] = None
    # Optional tuple of tensors for hidden states
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # Optional tuple of tensors for attentions
    attentions: Optional[Tuple[torch.FloatTensor]] = None
# 导入必要的库
from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn
from transformers.modeling_outputs import ModelOutput
from transformers.modeling_outputs import BaseModelOutputWithPooling

# 创建一个名为ClvpOutput的数据类，用于存储Clvp模型的输出
@dataclass
class ClvpOutput(ModelOutput):
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            对比损失，用于衡量语音与文本的相似性。
        speech_ids (`torch.LongTensor`, *optional*):
            由`ClvpForCausalLM`模型生成的语音ID（或语音候选项）。
        logits_per_speech (`torch.FloatTensor` of shape `(speech_batch_size, text_batch_size)`):
            `speech_embeds`和`text_embeds`之间的缩放点积分数。 这代表了语音-文本的相似性分数。
        logits_per_text (`torch.FloatTensor` of shape `(text_batch_size, speech_batch_size)`):
            `text_embeds`和`speech_embeds`之间的缩放点积分数。 这代表了文本-语音的相似性分数。
        text_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim`):
            通过对文本编码器模型的汇聚输出应用投影层而获得的文本嵌入。
        speech_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim`):
            通过对语音编码器模型的汇聚输出应用投影层而获得的语音嵌入。
        text_model_output (`BaseModelOutputWithPooling`):
            文本编码器模型`last_hidden_state`的汇聚输出。
        speech_model_output (`BaseModelOutputWithPooling`):
            语音编码器模型`last_hidden_state`的汇聚输出。
        decoder_hidden_states (`torch.FloatTensor`, *optional*):
            解码器模型的隐藏状态。
        text_encoder_hidden_states (`torch.FloatTensor`, *optional*):
            文本编码器模型的隐藏状态。
        speech_encoder_hidden_states (`torch.FloatTensor`, *optional*):
            语音编码器模型的隐藏状态。
    """
    
    # 定义数据类的字段，用于存储模型输出的不同部分
    loss: Optional[torch.FloatTensor] = None
    speech_ids: Optional[torch.LongTensor] = None
    logits_per_speech: torch.FloatTensor = None
    logits_per_text: torch.FloatTensor = None
    text_embeds: torch.FloatTensor = None
    speech_embeds: torch.FloatTensor = None
    text_model_output: BaseModelOutputWithPooling = None
    speech_model_output: BaseModelOutputWithPooling = None
    decoder_hidden_states: torch.FloatTensor = None
    text_encoder_hidden_states: torch.FloatTensor = None
    speech_encoder_hidden_states: torch.FloatTensor = None


# 从transformers.models.llama.modeling_llama中复制ClvpRMSNorm类，并将Llama->Clvp
class ClvpRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        ClvpRMSNorm等价于T5LayerNorm
        """
        # 调用nn.Module的初始化方法
        super().__init__()
        # 创建可训练参数weight，初始值为1，用于归一化
        self.weight = nn.Parameter(torch.ones(hidden_size))
        # 设置方差的防止除零的小量
        self.variance_epsilon = eps
    # 前向传播函数，接受隐藏状态作为输入参数
    def forward(self, hidden_states):
        # 获取输入张量的数据类型
        input_dtype = hidden_states.dtype
        # 将输入张量转换为 32 位浮点类型
        hidden_states = hidden_states.to(torch.float32)
        # 计算隐藏状态的方差
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # 对隐藏状态进行归一化处理
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # 返回归一化后的隐藏状态乘以权重
        return self.weight * hidden_states.to(input_dtype)
class ClvpRotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding Class for CLVP. It was proposed in the paper 'ROFORMER: ENHANCED TRANSFORMER WITH ROTARY
    POSITION EMBEDDING', Please see https://arxiv.org/pdf/2104.09864v1.pdf .
    """

    def __init__(self, config):
        super().__init__()
        # 计算每个头的维度
        dim = max(config.projection_dim // (config.num_attention_heads * 2), 32)
        # 计算旋转位置编码的频率
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))

        # 将频率作为缓冲区注册到模型中
        self.register_buffer("inv_freq", inv_freq)
        # 缓存序列长度和旋转位置编码，以避免重复计算
        self.cached_sequence_length = None
        self.cached_rotary_positional_embedding = None

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        # 获取输入张量的序列长度
        sequence_length = hidden_states.shape[1]

        # 如果序列长度和缓存的序列长度相同，并且缓存的旋转位置编码不为空，则直接返回缓存的旋转位置编码
        if sequence_length == self.cached_sequence_length and self.cached_rotary_positional_embedding is not None:
            return self.cached_rotary_positional_embedding

        # 更新缓存的序列长度
        self.cached_sequence_length = sequence_length
        # 生成时间戳，用于计算旋转位置编码
        time_stamps = torch.arange(sequence_length, device=hidden_states.device).type_as(self.inv_freq)
        # 计算频率和时间戳的乘积，得到旋转位置编码
        freqs = torch.einsum("i,j->ij", time_stamps, self.inv_freq)
        # 将绝对位置编码和旋转位置编码拼接在一起
        embeddings = torch.cat((freqs, freqs), dim=-1)

        # 将旋转位置编码添加一个维度，并将其作为输出返回
        self.cached_rotary_positional_embedding = embeddings.unsqueeze(0)
        return self.cached_rotary_positional_embedding


class ClvpSelfAttention(nn.Module):
    """
    Multi-headed attention to combine Absolute and Rotary Positional Embeddings into a single Attention module.
    """

    def __init__(self, config):
        super().__init__()
        # 获取配置信息
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        # 检查是否可以均匀分割隐藏层维度
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        # 设置缩放因子
        self.scale = self.head_dim**-0.5
        # 设置注意力的 dropout 概率
        self.dropout = config.attention_dropout

        # 如果配置中包含最大位置编码长度，则初始化一个用于屏蔽注意力矩阵的张量
        if hasattr(config, "max_position_embeddings"):
            max_positions = config.max_position_embeddings
            bias = torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool))
            bias = bias.view(1, 1, max_positions, max_positions)
            self.register_buffer("bias", bias, persistent=False)

        # 定义用于计算键、值和查询的线性层
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.use_attention_bias)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.use_attention_bias)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.use_attention_bias)
        # 定义输出投影层
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    # Copied from transformers.models.clip.modeling_clip.CLIPAttention._shape
    # 定义私有方法_shape，用于调整输入张量的形状以匹配多头注意力机制所需的格式
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        # 调整张量形状，将其转换为(batch_size, seq_len, num_heads, head_dim)的格式
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    # 定义前向传播方法，实现自注意力机制
    def forward(
        self,
        hidden_states: torch.FloatTensor,
        rotary_pos_emb: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: Optional[bool] = False,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
class ClvpGatedLinearUnit(nn.Module):
    """
    `ClvpGatedLinearUnit` uses the second half of the `hidden_states` to act as a gate for the first half of the
    `hidden_states` which controls the flow of data from the first of the tensor.
    """

    def __init__(self, config):
        super().__init__()
        # 设置激活函数为配置中指定的隐藏层激活函数
        self.activation_fn = ACT2FN[config.hidden_act]
        # 定义一个线性变换层，将隐藏状态的维度映射为两倍的中间尺寸
        self.proj = nn.Linear(config.hidden_size, config.intermediate_size * 2)

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        # 通过线性变换层获取隐藏状态的投影结果，并将其分成两半
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
        # 返回使用门控单元激活的隐藏状态
        return hidden_states * self.activation_fn(gate)


class ClvpEncoderMLP(nn.Module):
    """
    This MLP is used in CLVP speech or text encoder models.
    """

    def __init__(self, config):
        super().__init__()
        # 初始化模型配置
        self.config = config
        # 定义第一个全连接层，使用门控线性单元作为激活函数
        self.fc1 = ClvpGatedLinearUnit(config)
        # 定义第二个全连接层，将中间尺寸映射回隐藏状态尺寸
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        # 定义一个 dropout 层
        self.dropout_layer = nn.Dropout(config.dropout)

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        # 通过第一个全连接层
        hidden_states = self.fc1(hidden_states)
        # 对结果进行 dropout 处理
        hidden_states = self.dropout_layer(hidden_states)
        # 通过第二个全连接层
        hidden_states = self.fc2(hidden_states)
        # 返回结果
        return hidden_states


class ClvpEncoderLayer(nn.Module):
    def __init__(self, config: ClvpConfig):
        super().__init__()
        # 初始化模型配置
        self.config = config
        # 定义嵌入维度
        self.embed_dim = config.hidden_size
        # 定义自注意力机制
        self.self_attn = ClvpSelfAttention(config)
        # 定义 MLP 模块
        self.mlp = ClvpEncoderMLP(config)

        # 定义输入 RMS 归一化层
        self.input_rmsnorm = ClvpRMSNorm(self.embed_dim, eps=config.layer_norm_eps)
        # 定义自注意力后 RMS 归一化层
        self.post_attention_rmsnorm = ClvpRMSNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        rotary_pos_emb: torch.FloatTensor,
        attention_mask: torch.LongTensor,
        position_ids: torch.LongTensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch, seq_len, embed_dim)`):
                input to the layer. 输入到该层的张量，形状为`(batch, seq_len, embed_dim)`。
            rotary_pos_emb (`torch.FloatTensor`):
                rotary position embeddings generated by `ClvpRotaryPositionalEmbedding` module. 由`ClvpRotaryPositionalEmbedding`模块生成的旋转位置嵌入。
            attention_mask (`torch.FloatTensor` of shape `(batch, 1, tgt_len, src_len)`):
                attention mask where padding elements are indicated by very large negative values. 注意力掩码，其中填充元素由非常大的负值表示。
            position_ids (`torch.LongTensor`):
                Denotes position ids of the input tokens. 表示输入标记的位置ID。
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail. 是否返回所有注意力层的注意力张量。有关更多详细信息，请参阅返回张量下的`attentions`。
        """
        residual = hidden_states  # 保存输入的残差连接
        
        hidden_states = self.input_rmsnorm(hidden_states)  # 输入层的 RMS 归一化

        attention_outputs = self.self_attn(
            hidden_states=hidden_states,
            rotary_pos_emb=rotary_pos_emb,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
        )  # 自注意力机制

        hidden_states = attention_outputs[0]  # 获取自注意力机制输出的隐藏状态

        hidden_states = residual + hidden_states  # 残差连接

        residual = hidden_states  # 保存残差连接

        hidden_states = self.post_attention_rmsnorm(hidden_states)  # 后注意力层的 RMS 归一化
        hidden_states = self.mlp(hidden_states)  # 多层感知机
        hidden_states = residual + hidden_states  # 残差连接

        outputs = (hidden_states,)  # 输出为隐藏状态的元组

        if output_attentions:
            outputs += (attention_outputs[-1],)  # 如果输出注意力张量，则将其添加到输出元组中

        return outputs  # 返回输出元组
# 定义一个名为ClvpDecoderMLP的类，继承自nn.Module类，用于处理GPT2模型中MLP部分的操作
class ClvpDecoderMLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size
        # 创建一个一维卷积层，输入维度为embed_dim，输出维度为intermediate_size
        self.c_fc = Conv1D(intermediate_size, embed_dim)
        # 创建一个一维卷积层，输入维度为intermediate_size，输出维度为embed_dim
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        # 根据配置中的激活函数选择对应的激活函数
        self.act = ACT2FN[config.activation_function]
        # 创建一个Dropout层，概率为config.resid_pdrop
        self.dropout = nn.Dropout(config.resid_pdrop)

    # 前向传播函数，接收隐藏状态作为输入，返回处理后的隐藏状态
    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        # 使用c_fc对隐藏状态进行卷积操作
        hidden_states = self.c_fc(hidden_states)
        # 使用激活函数对卷积后的隐藏状态进行激活
        hidden_states = self.act(hidden_states)
        # 使用c_proj对激活后的隐藏状态进行卷积操作
        hidden_states = self.c_proj(hidden_states)
        # 对卷积后的隐藏状态进行Dropout操作
        hidden_states = self.dropout(hidden_states)
        # 返回处理后的隐藏状态
        return hidden_states


# 定义一个名为ClvpDecoderLayer的类，继承自nn.Module类，用于处理GPT2模型中Decoder层的操作
class ClvpDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        # 根据配置创建内部维度inner_dim
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        # 创建LayerNorm层，用于输入层的归一化
        self.input_layernorm = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        # 创建自注意力机制层
        self.attn = ClvpSelfAttention(config)
        # 创建后自注意力机制层的归一化层
        self.post_attention_layernorm = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        # 创建MLP层
        self.mlp = ClvpDecoderMLP(inner_dim, config)

    # 前向传播函数，接收隐藏状态等多个参数，返回处理后的输出
    def forward(
        self,
        hidden_states: Optional[Tuple[torch.FloatTensor]],
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = False,
        output_attentions: Optional[bool] = False,
    ) -> Union[Tuple[torch.Tensor], Optional[Tuple[torch.Tensor, Tuple[torch.FloatTensor, ...]]]]:
        # 保存残差连接
        residual = hidden_states
        # 对输入层进行LayerNorm归一化
        hidden_states = self.input_layernorm(hidden_states)
        # 使用自注意力机制层处理隐藏状态
        attn_outputs = self.attn(
            hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        attn_output = attn_outputs[0]
        outputs = attn_outputs[1:]
        # 残差连接
        hidden_states = attn_output + residual

        # 保存残差连接
        residual = hidden_states
        # 对后自注意力机制层的输出进行LayerNorm归一化
        hidden_states = self.post_attention_layernorm(hidden_states)
        # 使用MLP层处理隐藏状态
        feed_forward_hidden_states = self.mlp(hidden_states)
        # 残差连接
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs


# 定义一个名为ClvpConditioningEncoder的类，继承自nn.Module类，用于处理特征提取器提取的log-mel频谱图和文本标记的操作
class ClvpConditioningEncoder(nn.Module):
    """
    This class processes the log-mel spectrograms(extracted by the Feature Extractor) and text tokens(produced by the
    """
    为解码器模型提供输入的前处理过程。

    首先，将每个对数梅尔频谱处理成一个单一的向量，捕捉其中的有价值特征，然后将文本标记转换为标记嵌入，并在之后添加位置嵌入。
    这两个向量被连接起来，然后传递给解码器模型。

    文本标记有助于融合“文本信息”，对数梅尔频谱用于将“语音特征”纳入生成的梅尔标记中。
    """

    def __init__(self, config: ClvpConfig):
        super().__init__()

        self.text_config = config.text_config  # 保存文本配置
        self.decoder_config = config.decoder_config  # 保存解码器配置

        self.text_token_embedding = nn.Embedding(self.text_config.vocab_size, self.decoder_config.hidden_size)  # 文本标记嵌入层
        self.text_position_embedding = nn.Embedding(  # 文本位置嵌入层
            self.decoder_config.max_text_tokens, self.decoder_config.hidden_size
        )

        self.mel_conv = nn.Conv1d(self.decoder_config.feature_size, self.decoder_config.hidden_size, kernel_size=1)  # 梅尔频谱卷积层

        # 定义每个注意力层前使用的组归一化
        num_groups = self.compute_groupnorm_groups(self.decoder_config.hidden_size)  # 计算组归一化的分组数
        self.group_norms = nn.ModuleList(
            [
                nn.GroupNorm(num_groups, self.decoder_config.hidden_size, eps=1e-5, affine=True)  # 定义组归一化层
                for _ in range(self.decoder_config.num_mel_attn_blocks)
            ]
        )

        # 定义注意力层
        self.mel_attn_blocks = nn.ModuleList(
            [ClvpSelfAttention(self.decoder_config) for _ in range(self.decoder_config.num_mel_attn_blocks)]  # 多头自注意力层
        )

        self.gradient_checkpointing = False  # 梯度检查点标志

    def compute_groupnorm_groups(self, channels: int, groups: int = 32):
        """
        计算用于 nn.GroupNorm 的 `num_groups` 的值。此逻辑来自官方的 tortoise 仓库。
        链接：https://github.com/neonbjb/tortoise-tts/blob/4003544b6ff4b68c09856e04d3eff9da26d023c2/tortoise/models/arch_util.py#L26
        """
        if channels <= 16:  # 如果通道数小于等于16
            groups = 8  # 分组数设为8
        elif channels <= 64:  # 如果通道数小于等于64
            groups = 16  # 分组数设为16
        while channels % groups != 0:  # 当通道数不能整除分组数时
            groups = int(groups / 2)  # 将分组数减半

        if groups <= 2:  # 如果分组数小于等于2
            raise ValueError(
                f"Number of groups for the GroupNorm must be greater than 2, but it is {groups}."  # 抛出数值错误异常
                f"Please consider using a different `hidden_size`"  # 提示考虑使用不同的隐藏大小
            )

        return groups  # 返回计算得到的分组数

    def forward(
        self,
        input_features: torch.FloatTensor,  # 输入特征（对数梅尔频谱）
        input_ids: Optional[torch.LongTensor] = None,  # 输入标记（文本）
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入嵌入（文本标记嵌入）
        attention_mask: Optional[torch.LongTensor] = None,  # 注意力遮罩
```   
class ClvpPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 定义一个类，用于处理权重初始化和预训练模型的下载和加载，继承自PreTrainedModel

    config_class = ClvpConfig
    # 设置配置类为ClvpConfig

    base_model_prefix = "clvp"
    # 设置基础模型前缀为"clvp"

    supports_gradient_checkpointing = True
    # 支持梯度检查点

    _skip_keys_device_placement = "past_key_values"
    # 跳过设备放置的键为"past_key_values"

    def _init_weights(self, module):
        """Initialize the weights"""
        # 初始化权重的方法

        factor = self.config.initializer_factor
        # 获取初始化因子

        if isinstance(module, nn.Embedding):
            # 如果module是nn.Embedding类型
            module.weight.data.normal_(mean=0.0, std=factor * 0.02)
            # 初始化权重数据为正态分布

        elif isinstance(module, (nn.Linear, Conv1D, nn.Conv1d)):
            # 如果module是nn.Linear、Conv1D或nn.Conv1d类型
            module.weight.data.normal_(mean=0.0, std=factor * 0.02)
            # 初始化权重数据为正态分布
            if module.bias is not None:
                module.bias.data.zero_()
                # 如果存在偏置项，则初始化为0

        elif isinstance(module, ClvpEncoderMLP):
            # 如果module是ClvpEncoderMLP类型
            factor = self.config.initializer_factor
            # 获取初始化因子
            in_proj_std = (module.config.hidden_size**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            fc_std = (2 * module.config.hidden_size) ** -0.5 * factor
            nn.init.normal_(module.fc1.proj.weight if getattr(module.fc1, "proj") else module.fc1.weight, std=fc_std)
            nn.init.normal_(module.fc2.weight, std=in_proj_std)
            # 初始化权重数据为正态分布

        elif isinstance(module, ClvpEncoder):
            # 如果module是ClvpEncoder类型
            config = self.config.text_config if hasattr(self.config, "text_config") else self.config
            factor = config.initializer_factor
            module.projection.weight.data.normal_(mean=0.0, std=factor * (config.hidden_size**-0.5))
            # 初始化权重数据为正态分布

        elif isinstance(module, ClvpConditioningEncoder):
            # 如果module是ClvpConditioningEncoder类型
            module.mel_conv.weight.data.normal_(mean=0.0, std=factor)
            module.mel_conv.bias.data.zero_()
            # 初始化权重数据为正态分布，偏置项初始化为0

        elif isinstance(module, ClvpForCausalLM):
            # 如果module是ClvpForCausalLM类型
            for name, p in module.named_parameters():
                if name == "c_proj.weight":
                    p.data.normal_(
                        mean=0.0, std=(self.config.initializer_range / math.sqrt(2 * self.config.num_hidden_layers))
                    )
            # 初始化权重数据为正态分布

        if isinstance(module, nn.LayerNorm):
            # 如果module是nn.LayerNorm类型
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
            # 初始化偏置项为0，权重为1.0


CLVP_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.
    # 参数说明：
    # config ([`ClvpConfig`]): 模型配置类，包含模型的所有参数。
    # 使用配置文件初始化模型不会加载与模型相关的权重，只加载配置信息。
    # 查看 [`~PreTrainedModel.from_pretrained`] 方法以加载模型权重。
# 定义 CLVP 输入的文档字符串，包含了各种输入参数的说明
CLVP_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional`):
            输入序列标记在词汇表中的索引。默认情况下，将忽略填充。

            可以使用 [`AutoTokenizer`] 获取索引。有关详细信息，请参阅 [`PreTrainedTokenizer.encode`] 和
            [`PreTrainedTokenizer.__call__`]。

            [什么是输入 ID？](../glossary#input-ids)
        input_features (`torch.FloatTensor` of shape `(batch_size, feature_size, time_dim)`):
            表示音频返回的对数梅尔频谱图表示的特征。由 [`ClvpFeatureExtractor`] 返回。
        conditioning_encoder_inputs_embeds (`torch.FloatTensor`, *optional`):
            `ClvpConditioningEncoder` 的 inputs_embeds。可以替代 `input_ids`。
        text_encoder_inputs_embeds (`torch.FloatTensor`, *optional`):
            用于替代 `input_ids` 的文本编码器模型的 inputs_embeds。
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional`):
            遮罩，避免在填充文本标记索引上执行注意力。遮罩值选择在 `[0, 1]`：

            - 1 表示**未遮罩**的标记，
            - 0 表示**遮罩**的标记。

            [什么是注意力遮罩？](../glossary#attention-mask)
        return_loss (`bool`, *optional`):
            是否返回对比损失。
        output_attentions (`bool`, *optional`):
            是否返回所有注意力层的注意力张量。有关更多详细信息，请参见返回张量下的 `attentions`。
        output_hidden_states (`bool`, *optional`):
            是否返回所有层的隐藏状态。有关更多详细信息，请参见返回张量下的 `hidden_states`。
        return_dict (`bool`, *optional`):
            是否返回 [`~utils.ModelOutput`] 而不是普通元组。
"""

# 定义 CLVP 解码器输入的文档字符串
CLVP_DECODER_INPUTS_DOCSTRING = r"""
"""

# 定义 ClvpEncoder 类，表示由 `config.num_hidden_layers` 个自注意力层组成的 Transformer 编码器
class ClvpEncoder(ClvpPreTrainedModel):
    """
    Transformer 编码器，由 `config.num_hidden_layers` 个自注意力层组成。每个层都是一个 [`ClvpEncoderLayer`]。

    Args:
        config: ClvpConfig
"""
    # 初始化函数，接受一个 ClvpConfig 类型的参数
    def __init__(self, config: ClvpConfig):
        # 调用父类的初始化方法
        super().__init__(config)

        # 将配置参数保存在实例中
        self.config = config
        # 创建一个词嵌入层，用于将输入的 token 编码为隐藏表示
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        # 如果配置中指定使用旋转位置编码，则创建旋转位置编码对象，否则置为 None
        self.rotary_pos_emb = ClvpRotaryPositionalEmbedding(config) if config.use_rotary_embedding else None
        # 创建多层编码器，每一层是 ClvpEncoderLayer 类型的对象
        self.layers = nn.ModuleList([ClvpEncoderLayer(config) for _ in range(config.num_hidden_layers)])

        # 创建一个用于序列摘要的对象
        self.sequence_summary = SequenceSummary(config)
        # 创建一个用于最终归一化的层
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 创建一个线性变换层，将隐藏表示投影到指定维度
        self.projection = nn.Linear(config.hidden_size, config.projection_dim, bias=False)

        # 是否使用梯度检查点
        self.gradient_checkpointing = False

        # 执行初始化后的操作
        self.post_init()

    # 获取输入嵌入层对象
    def get_input_embeddings(self):
        return self.token_embedding

    # 设置输入嵌入层对象
    def set_input_embeddings(self, value):
        self.token_embedding = value

    # 前向传播函数
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
```  
class ClvpDecoder(ClvpPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`ClvpDecoderLayer`]
    """

    def __init__(self, config):
        super().__init__(config)

        self.config = config

        # 初始化输入的词嵌入层，将词索引映射为隐藏状态的向量
        self.input_embeds_layer = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        # 初始化位置编码的词嵌入层，用于表示输入序列中每个位置的信息
        self.position_embeds_layer = nn.Embedding(self.config.max_position_embeddings, self.config.hidden_size)

        # 初始化丢弃层，用于在训练过程中进行随机丢弃以防止过拟合
        self.drop = nn.Dropout(self.config.embd_pdrop)
        # 初始化多层解码器，每个层由一个ClvpDecoderLayer组成
        self.layers = nn.ModuleList([ClvpDecoderLayer(self.config) for _ in range(self.config.num_hidden_layers)])
        # 初始化层归一化层，用于将每层的输出进行归一化处理
        self.layer_norm = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_epsilon)

        # 是否使用梯度检查点，用于节省内存，但可能会降低训练速度
        self.gradient_checkpointing = False

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        # 返回输入词嵌入层
        return self.input_embeds_layer

    def set_input_embeddings(self, new_embeddings):
        # 设置输入词嵌入层的新权重
        self.input_embeds_layer = new_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            # 裁剪模型的注意力头
            self.layers[layer].attn.prune_heads(heads)

    @add_start_docstrings_to_model_forward(CLVP_DECODER_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        Return:
        """
        raise NotImplementedError


@add_start_docstrings(
    "The bare Clvp decoder model outputting raw hidden-states without any specific head on top.",
    CLVP_START_DOCSTRING,
)
class ClvpModel(ClvpPreTrainedModel):
    def __init__(self, config: ClvpDecoderConfig):
        super().__init__(config)
        self.config = config
        # 初始化ClvpDecoder作为模型的解码器
        self.decoder = ClvpDecoder(self.config)

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        # 返回解码器的输入词嵌入层
        return self.decoder.input_embeds_layer

    def set_input_embeddings(self, value):
        # 设置解码器的输入词嵌入层的新权重
        self.decoder.input_embeds_layer = value

    def get_decoder(self):
        # 返回解码器
        return self.decoder

    @add_start_docstrings_to_model_forward(CLVP_DECODER_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        Return:
        """
        raise NotImplementedError
    # 定义一个前向传播函数，接受多个输入参数，并返回一个包含过去和交叉注意力的基础模型输出
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入的 token ID
        attention_mask: Optional[torch.FloatTensor] = None,  # 注意力掩码
        token_type_ids: Optional[torch.LongTensor] = None,  # token 类型 ID
        position_ids: Optional[torch.LongTensor] = None,  # 位置 ID
        head_mask: Optional[torch.FloatTensor] = None,  # 头部掩码
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,  # 过去的键值对
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入的嵌入向量
        use_cache: Optional[bool] = None,  # 是否使用缓存
        output_attentions: Optional[bool] = None,  # 是否输出注意力
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回字典形式的结果
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:  # 返回值的类型注解
        # 如果未指定输出注意力，则使用配置中的输出注意力设置
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果未指定输出隐藏状态，则使用配置中的输出隐藏状态设置
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果未指定使用缓存，则使用配置中的使用缓存设置
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        # 如果未指定返回字典形式的结果，则使用配置中的使用返回字典设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 解码器输出包括（解码特征，过去的键值对，解码隐藏状态，解码注意力）
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 如果不返回字典形式的结果，则直接返回解码器输出
        if not return_dict:
            return decoder_outputs

        # 返回一个包含过去和交叉注意力的基础模型输出
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
        )
# 添加开始文档字符串注释，描述 CLVP 解码模型及其语言建模头部的作用
@add_start_docstrings(
    "The CLVP decoder model with a language modelling head on top.",
    CLVP_START_DOCSTRING,
)
# 定义 ClvpForCausalLM 类，继承自 ClvpPreTrainedModel 类
class ClvpForCausalLM(ClvpPreTrainedModel):
    # 初始化函数，接受配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__(config)

        # 将配置对象存储在当前实例中
        self.config = config
        # 创建 CLVP 模型对象，传入配置对象作为参数
        self.model = ClvpModel(self.config)

        # 创建最终的归一化层，使用 LayerNorm 进行归一化，输入维度为隐藏层大小
        self.final_norm = nn.LayerNorm(self.config.hidden_size)
        # 创建语言模型头部的线性层，将隐藏状态映射到词汇表大小的向量，带有偏置
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=True)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入的方法
    def get_input_embeddings(self):
        # 返回 CLVP 模型解码器的输入嵌入层
        return self.model.decoder.input_embeds_layer

    # 设置输入嵌入的方法
    def set_input_embeddings(self, new_embeddings):
        # 将新的嵌入赋值给 CLVP 模型解码器的输入嵌入层
        self.model.decoder.input_embeds_layer = new_embeddings

    # 准备模型输入的内部方法
    def _prepare_model_inputs(
        self,
        inputs: Optional[torch.Tensor] = None,
        bos_token_id: Optional[int] = None,
        model_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ):
    # 为生成准备输入的方法
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, conditioning_embeds=None, **kwargs
```  
    ):
        # 计算输入序列的长度
        input_ids_length = input_ids.shape[-1]
        # 获取 token_type_ids 参数，如果不存在则设为 None
        token_type_ids = kwargs.get("token_type_ids", None)
        # 如果 past_key_values 存在
        if past_key_values:
            # 获取过去状态的长度
            past_length = past_key_values[0][0].shape[2]

            # 检查输入序列的长度是否大于过去状态的长度
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 如果输入序列的长度小于等于过去状态的长度，则仅保留最后一个 token
                remove_prefix_length = input_ids.shape[1] - 1

            # 从输入序列中移除前缀
            input_ids = input_ids[:, remove_prefix_length:]
            # 如果 token_type_ids 存在，则相应地调整其长度
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -input_ids.shape[1] :]

        # 获取 attention_mask 和 position_ids 参数，如果不存在则设为 None
        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        # 如果 attention_mask 存在但 position_ids 不存在
        if attention_mask is not None and position_ids is None:
            # 为批次生成创建 position_ids
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            # 如果 past_key_values 存在，则只保留最后一个位置
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            position_ids = None

        # 如果 conditioning_embeds 和 past_key_values 存在，则将 position_ids 设为输入序列的长度
        if conditioning_embeds is not None and past_key_values is not None:
            position_ids = torch.tensor([input_ids_length], dtype=torch.long, device=input_ids.device)

        # 如果 inputs_embeds 存在且 past_key_values 不存在，则将 inputs_embeds 传递给模型输入
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            # 否则，将 input_ids 传递给模型输入
            model_inputs = {"input_ids": input_ids}

        # 更新模型输入参数
        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "token_type_ids": token_type_ids,
            }
        )
        # 返回模型输入
        return model_inputs

    # 添加 CLVP_DECODER_INPUTS_DOCSTRING 中的文档字符串到模型前向方法中
    @add_start_docstrings_to_model_forward(CLVP_DECODER_INPUTS_DOCSTRING)
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
    ) -> Union[Tuple, CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """

        # 检查是否需要输出注意力权重
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 检查是否需要输出隐藏状态
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 检查是否使用缓存
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        # 检查是否返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用模型进行前向传播
        outputs = self.model(
            input_ids=input_ids,
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

        # 获取隐藏状态
        hidden_states = outputs[0]

        # 对隐藏状态进行最终归一化
        lm_logits = self.final_norm(hidden_states)
        # 将归一化后的隐藏状态传入语言模型头部
        lm_logits = self.lm_head(lm_logits)

        loss = None
        if labels is not None:
            labels = labels.to(lm_logits.device)
            # 将预测的 logits 向左移动一个位置，用于计算损失
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # 将 logits 和标签展平，用于计算交叉熵损失
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        if not return_dict:
            # 如果不返回字典，则返回损失和输出
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 返回带有注意力权重的输出
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    @staticmethod
    # 从 transformers.models.gpt2.modeling_gpt2.GPT2LMHeadModel._reorder_cache 复制而来
    def _reorder_cache(
        past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
    # 定义一个函数，用于重新排列 `past_key_values` 缓存，当调用 [`~PreTrainedModel.beam_search`] 或 [`~PreTrainedModel.beam_sample`] 时需要。这是为了确保在每一代生成步骤中将 `past_key_values` 与正确的 beam_idx 匹配。
    def reorder_past_key_values(self, past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor) -> Tuple[Tuple[torch.Tensor]]:
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        # 使用嵌套的 generator 表达式，重新排列 `past_key_values` 缓存，以匹配正确的 beam_idx
        return tuple(
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            for layer_past in past_key_values
        )
# 添加起始文档字符串，描述了 CLVP 模型的组成部分和工作原理
# 继承自 ClvpPreTrainedModel 类
class ClvpModelForConditionalGeneration(ClvpPreTrainedModel):
    # 指定配置类为 ClvpConfig
    config_class = ClvpConfig

    # 初始化方法，接受一个配置对象作为参数
    def __init__(self, config: ClvpConfig):
        # 调用父类的初始化方法
        super().__init__(config)

        # 检查文本编码器配置是否为 ClvpEncoderConfig 类型
        if not isinstance(config.text_config, ClvpEncoderConfig):
            raise ValueError(
                "config.text_config is expected to be of type `ClvpEncoderConfig` but is of type"
                f" {type(config.text_config)}."
            )

        # 检查语音编码器配置是否为 ClvpEncoderConfig 类型
        if not isinstance(config.speech_config, ClvpEncoderConfig):
            raise ValueError(
                "config.speech_config is expected to be of type `ClvpEncoderConfig` but is of type"
                f" {type(config.speech_config)}."
            )

        # 检查解码器配置是否为 ClvpDecoderConfig 类型
        if not isinstance(config.decoder_config, ClvpDecoderConfig):
            raise ValueError(
                "config.decoder_config is expected to be of type `ClvpDecoderConfig` but is of type"
                f" {type(config.decoder_config)}."
            )

        # 创建 CLVP 模型的条件编码器
        self.conditioning_encoder = ClvpConditioningEncoder(config)

        # 创建 CLVP 模型的语音解码器
        self.speech_decoder_model = ClvpForCausalLM(config.decoder_config)

        # 创建 CLVP 模型的文本编码器
        self.text_encoder_model = ClvpEncoder(config.text_config)
        # 创建 CLVP 模型的语音编码器
        self.speech_encoder_model = ClvpEncoder(config.speech_config)

        # 初始化 logit_scale 参数
        self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))

        # 初始化权重并应用最终处理
        self.post_init()

    # 从原始仓库中获取的注释
    # 链接：https://github.com/neonbjb/tortoise-tts/blob/4003544b6ff4b68c09856e04d3eff9da26d023c2/tortoise/api.py#L117
    def fix_speech_decoder_output(self, speech_ids: torch.LongTensor) -> torch.LongTensor:
        """
        This method modifies the output of the decoder model, such as replacing the `eos_token_id` and changing the
        last few tokens of each sequence.

        Args:
            speech_ids (`torch.LongTensor`):
                This refers to the output of the decoder model.
        """
        # 从第二个位置开始取，丢弃序列中的起始标记
        speech_ids = speech_ids[:, 1:]

        # 找到所有停止标记的索引位置
        stop_token_indices = torch.where(speech_ids == self.speech_decoder_model.config.eos_token_id, 1, 0)
        # 使用特定值替换停止标记的位置
        speech_ids = torch.masked_fill(speech_ids, mask=stop_token_indices.bool(), value=decoder_fixing_codes[0])

        for i, each_seq_stop_token_index in enumerate(stop_token_indices):
            # 如果该序列中没有找到停止标记，则跳过对该序列的处理
            if each_seq_stop_token_index.sum() == 0:
                continue

            # 找到当前序列中最后一个停止标记的位置
            stm = each_seq_stop_token_index.argmax()
            # 将停止标记之后的所有标记替换为特定值
            speech_ids[i, stm:] = decoder_fixing_codes[0]
            # 如果停止标记之前还有至少3个标记，则将最后3个标记替换为另一个特定值
            if stm - 3 < speech_ids.shape[1]:
                speech_ids[i, -3:] = torch.tensor(
                    [decoder_fixing_codes[1:]], device=speech_ids.device, dtype=torch.long
                )

        return speech_ids

    def get_text_features(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        text_encoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
    ) -> torch.FloatTensor:
        r"""
        This method can be used to extract text_embeds from a text. The text embeddings obtained by applying the
        projection layer to the pooled output of the CLVP text encoder model.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                [What are input IDs?](../glossary#input-ids)
            text_encoder_inputs_embeds (`torch.FloatTensor`, *optional*):
                inputs_embeds for the text encoder model passed in place of `input_ids`.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)

        Returns:
            `torch.FloatTensor` of shape `(batch_size, output_dim)`:
                The text embeddings obtained by applying the projection layer to the pooled output of the CLVP Text
                Model.

        Examples:

        ```python
        >>> from transformers import ClvpProcessor, ClvpModelForConditionalGeneration

        >>> # Define the Text
        >>> text = "This is an example text."

        >>> # Define processor and model
        >>> processor = ClvpProcessor.from_pretrained("susnato/clvp_dev")
        >>> model = ClvpModelForConditionalGeneration.from_pretrained("susnato/clvp_dev")

        >>> # Generate processor output and text embeds
        >>> processor_output = processor(text=text, return_tensors="pt")
        >>> text_embeds = model.get_text_features(input_ids=processor_output["input_ids"])
        ```
        """

        # 使用 CLVP 文本编码器模型，获取输入文本的文本嵌入
        outputs = self.text_encoder_model(
            input_ids=input_ids,
            inputs_embeds=text_encoder_inputs_embeds,
            attention_mask=attention_mask,
        )

        # 返回 CLVP 文本编码器模型的输出中的第一个元素，即文本嵌入
        return outputs[0]

    def get_speech_features(
        self,
        speech_ids: Optional[torch.LongTensor] = None,
        input_ids: Optional[torch.LongTensor] = None,
        input_features: Optional[torch.FloatTensor] = None,
        conditioning_encoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        **kwargs,
    @add_start_docstrings_to_model_forward(CLVP_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ClvpOutput, config_class=ClvpConfig)
    # 定义一个方法用于前向传播，接受多个输入参数
    def forward(
        self,
        input_ids: torch.LongTensor = None,  # 输入的 token ID
        input_features: torch.FloatTensor = None,  # 输入的特征
        conditioning_encoder_inputs_embeds: Optional[torch.FloatTensor] = None,  # 编码器输入的嵌入向量
        text_encoder_inputs_embeds: Optional[torch.FloatTensor] = None,  # 文本编码器输入的嵌入向量
        attention_mask: Optional[torch.LongTensor] = None,  # 注意力掩码
        return_loss: Optional[bool] = None,  # 是否返回损失
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        output_attentions: Optional[bool] = False,  # 是否输出注意力权重
        return_dict: Optional[bool] = None,  # 是否返回字典形式的结果
    # 禁止梯度计算
    @torch.no_grad()
    # 定义一个方法用于生成，接受多个输入参数
    def generate(
        self,
        input_ids: torch.LongTensor = None,  # 输入的 token ID
        input_features: torch.FloatTensor = None,  # 输入的特征
        attention_mask: Optional[torch.LongTensor] = None,  # 注意力掩码
        generation_config: Optional[GenerationConfig] = None,  # 生成配置
        pad_to_max_mel_tokens: Optional[int] = None,  # 填充到最大 mel token 数
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        **kwargs,  # 其他关键字参数
```