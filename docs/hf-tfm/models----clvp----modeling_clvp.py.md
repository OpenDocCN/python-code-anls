# `.\models\clvp\modeling_clvp.py`

```
# 导入必要的库和模块
import copy  # 导入 copy 模块用于复制对象
import math  # 导入 math 模块用于数学运算
from dataclasses import dataclass  # 导入 dataclass 用于定义数据类
from typing import Dict, Optional, Tuple, Union  # 导入类型提示相关模块

import torch  # 导入 PyTorch 库
import torch.utils.checkpoint  # 导入 PyTorch 的 checkpoint 模块
from torch import nn  # 导入 PyTorch 的神经网络模块
from torch.nn import CrossEntropyLoss  # 导入交叉熵损失函数

from ...activations import ACT2FN  # 导入激活函数映射
from ...generation import GenerationConfig  # 导入生成配置相关模块
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
# 导入注意力掩码相关函数
from ...modeling_outputs import (  # 导入模型输出相关类
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPooling,
    CausalLMOutputWithCrossAttentions,
)
from ...modeling_utils import PreTrainedModel, SequenceSummary  # 导入模型工具类和序列摘要类
from ...pytorch_utils import Conv1D  # 导入 PyTorch 的一维卷积类
from ...utils import (  # 导入工具函数和类
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_clvp import (  # 导入 CLVP 模型的配置类
    ClvpConfig,
    ClvpDecoderConfig,
    ClvpEncoderConfig,
)


logger = logging.get_logger(__name__)  # 获取日志记录器

_CHECKPOINT_FOR_DOC = "susnato/clvp_dev"  # 设置用于文档的检查点名称

CLVP_PRETRAINED_MODEL_ARCHIVE_LIST = [  # CLVP 预训练模型的存档列表
    "susnato/clvp_dev",
    # 查看所有 CLVP 模型：https://huggingface.co/models?filter=clvp
]


# 从 transformers.models.clip.modeling_clip.contrastive_loss 复制过来
def contrastive_loss(logits: torch.Tensor) -> torch.Tensor:
    """对比损失函数，计算交叉熵损失"""
    return nn.functional.cross_entropy(logits, torch.arange(len(logits), device=logits.device))


# 从 transformers.models.clip.modeling_clip.clip_loss 复制过来，将函数名和变量名改为 clvp_loss 和 speech_loss
def clvp_loss(similarity: torch.Tensor) -> torch.Tensor:
    """CLVP 损失函数，结合文本和语音的对比损失"""
    caption_loss = contrastive_loss(similarity)  # 计算文本部分的对比损失
    speech_loss = contrastive_loss(similarity.t())  # 计算语音部分的对比损失
    return (caption_loss + speech_loss) / 2.0  # 返回两部分损失的平均值


# 从 transformers.models.llama.modeling_llama.rotate_half 复制过来
def rotate_half(x):
    """对输入的隐藏维度的一半进行旋转"""
    x1 = x[..., : x.shape[-1] // 2]  # 取前一半的数据
    x2 = x[..., x.shape[-1] // 2 :]  # 取后一半的数据
    return torch.cat((-x2, x1), dim=-1)  # 将后一半和前一半的数据拼接并返回


def apply_rotary_pos_emb(q, k, v, cos, sin, position_ids, unsqueeze_dim=1):
    """应用旋转位置嵌入到查询和键的张量中"""
    # 这里是函数的实现部分，根据具体的旋转位置嵌入方法完成对输入张量的操作
    # 根据给定的位置索引从 cosine 和 sine 部分提取位置编码向量，并在指定维度上进行 unsqueeze 操作，以便与 q 和 k 张量的维度匹配
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)

    # 使用 Rotary Position Embedding 对查询向量 q 进行旋转编码
    q_embed = (q * cos) + (rotate_half(q) * sin)

    # 使用 Rotary Position Embedding 对键向量 k 进行旋转编码
    k_embed = (k * cos) + (rotate_half(k) * sin)

    # 使用 Rotary Position Embedding 对值向量 v 进行旋转编码
    v_embed = (v * cos) + (rotate_half(v) * sin)

    # 返回经过 Rotary Position Embedding 旋转编码后的查询、键、值向量
    return q_embed, k_embed, v_embed
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

    # 在开头添加 bos token
    if add_bos_token:
        # 使用 torch 的函数在 input_ids 前面填充一个位置，值为 bos_token_id
        input_ids = torch.nn.functional.pad(input_ids, (1, 0), value=bos_token_id)
        # 如果有 attention_mask，则在开头填充一个有效位置，值为 1
        attention_mask = (
            torch.nn.functional.pad(attention_mask, (1, 0), value=1) if attention_mask is not None else attention_mask
        )

    # 创建 modified_input_ids 变量并初始化为 input_ids
    modified_input_ids = input_ids
    # 如果要添加 eos token
    if add_eos_token:
        # 根据 input_ids 的形状创建一个扩展后的 modified_input_ids
        modified_input_ids = torch.zeros(
            (input_ids.shape[0], input_ids.shape[1] + 1), dtype=input_ids.dtype, device=input_ids.device
        )
        # 遍历每个 input_id
        for i, each_input_id in enumerate(input_ids):
            # 找到有效 token 结束的位置，然后添加 eos token
            if torch.isin(each_input_id, pad_token_id).sum():
                # 找到第一个 pad_token_id 的位置
                pos = torch.where(each_input_id == pad_token_id)[0].min()
                # 在找到的位置前后添加 eos_token_id 构成新的 modified_input_ids
                modified_input_ids[i] = torch.concatenate(
                    [each_input_id[:pos], torch.tensor([eos_token_id], device=input_ids.device), each_input_id[pos:]]
                )
            else:
                # 如果没有 pad tokens，则在结尾添加 eos token
                modified_input_ids[i] = torch.nn.functional.pad(each_input_id, (0, 1), value=eos_token_id)
        # 如果有 attention_mask，则在开头填充一个有效位置，值为 1
        attention_mask = (
            torch.nn.functional.pad(attention_mask, (1, 0), value=1) if attention_mask is not None else attention_mask
        )

    # 返回修改后的 input_ids 和 attention_mask
    return modified_input_ids, attention_mask
    # `embeds` 是一个可选参数，表示模型应用投影层到汇聚输出后得到的嵌入向量。
    embeds: Optional[torch.FloatTensor] = None
    
    # `last_hidden_state` 是必须的参数，表示模型最后一层的隐藏状态。
    last_hidden_state: torch.FloatTensor = None
    
    # `pooler_output` 是一个可选参数，表示经过汇聚层处理后得到的汇聚输出。
    pooler_output: Optional[torch.FloatTensor] = None
    
    # `hidden_states` 是一个可选参数，是一个元组，包含模型每一层的隐藏状态输出。
    # 如果模型有嵌入层，则包含嵌入层的输出，形状为 `(batch_size, sequence_length, hidden_size)`。
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    
    # `attentions` 是一个可选参数，是一个元组，包含模型每一层的注意力权重。
    # 每个元素的形状为 `(batch_size, num_heads, sequence_length, sequence_length)`，
    # 表示经过注意力 softmax 后的注意力权重，用于计算自注意力头的加权平均值。
    attentions: Optional[Tuple[torch.FloatTensor]] = None
@dataclass
class ClvpOutput(ModelOutput):
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Contrastive loss for speech-text similarity.
        speech_ids (`torch.LongTensor`, *optional*):
            speech_ids (or speech candidates) generated by the `ClvpForCausalLM` model.
        logits_per_speech (`torch.FloatTensor` of shape `(speech_batch_size, text_batch_size)`):
            The scaled dot product scores between `speech_embeds` and `text_embeds`. This represents the speech-text
            similarity scores.
        logits_per_text (`torch.FloatTensor` of shape `(text_batch_size, speech_batch_size)`):
            The scaled dot product scores between `text_embeds` and `speech_embeds`. This represents the text-speech
            similarity scores.
        text_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The text embeddings obtained by applying the projection layer to the pooled output of the text encoder
            model.
        speech_embeds (`torch.FloatTensor` of shape `(batch_size, output_dim`):
            The speech embeddings obtained by applying the projection layer to the pooled output of the speech encoder
            model.
        text_model_output (`BaseModelOutputWithPooling`):
            The pooled output of the `last_hidden_state` of the text encoder Model.
        speech_model_output (`BaseModelOutputWithPooling`):
            The pooled output of the `last_hidden_state` of the speech encoder Model.
        decoder_hidden_states (`torch.FloatTensor`, *optional*):
            The hidden states of the decoder model.
        text_encoder_hidden_states (`torch.FloatTensor`, *optional*):
            The hidden states of the text encoder model.
        speech_encoder_hidden_states (`torch.FloatTensor`, *optional*):
            The hidden states of the speech encoder model.
    """

    loss: Optional[torch.FloatTensor] = None  # 损失值，用于表示语音文本相似性的对比损失
    speech_ids: Optional[torch.LongTensor] = None  # 由`ClvpForCausalLM`模型生成的语音ID（或语音候选项）
    logits_per_speech: torch.FloatTensor = None  # `speech_embeds`和`text_embeds`之间的缩放点积得分，表示语音文本相似性
    logits_per_text: torch.FloatTensor = None  # `text_embeds`和`speech_embeds`之间的缩放点积得分，表示文本语音相似性
    text_embeds: torch.FloatTensor = None  # 通过将文本编码器模型的池化输出应用到投影层获得的文本嵌入
    speech_embeds: torch.FloatTensor = None  # 通过将语音编码器模型的池化输出应用到投影层获得的语音嵌入
    text_model_output: BaseModelOutputWithPooling = None  # 文本编码器模型最后隐藏状态的池化输出
    speech_model_output: BaseModelOutputWithPooling = None  # 语音编码器模型最后隐藏状态的池化输出
    decoder_hidden_states: torch.FloatTensor = None  # 解码器模型的隐藏状态
    text_encoder_hidden_states: torch.FloatTensor = None  # 文本编码器模型的隐藏状态
    speech_encoder_hidden_states: torch.FloatTensor = None  # 语音编码器模型的隐藏状态


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Clvp
class ClvpRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        ClvpRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))  # 归一化层的权重参数，初始化为全1
        self.variance_epsilon = eps  # 方差的小值阈值，用于数值稳定性
    # 定义前向传播函数，用于处理隐藏状态
    def forward(self, hidden_states):
        # 获取输入张量的数据类型
        input_dtype = hidden_states.dtype
        # 将隐藏状态张量转换为 float32 类型
        hidden_states = hidden_states.to(torch.float32)
        # 计算隐藏状态张量每个元素的平方，并沿着最后一个维度求平均值，保持维度不变
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        # 将隐藏状态张量按元素乘以其标准差的倒数，以标准化数据
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # 返回经过权重调整后的隐藏状态张量
        return self.weight * hidden_states.to(input_dtype)
class ClvpRotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding Class for CLVP. It was proposed in the paper 'ROFORMER: ENHANCED TRANSFORMER WITH ROTARY
    POSITION EMBEDDING', Please see https://arxiv.org/pdf/2104.09864v1.pdf .
    """

    def __init__(self, config):
        super().__init__()
        # Calculate dimension of each projection in the rotary positional embedding
        dim = max(config.projection_dim // (config.num_attention_heads * 2), 32)
        # Calculate inverse frequencies for positional encoding
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2, dtype=torch.int64).float() / dim))

        # Register inverse frequencies as a buffer tensor
        self.register_buffer("inv_freq", inv_freq)
        self.cached_sequence_length = None
        self.cached_rotary_positional_embedding = None

    def forward(self, hidden_states: torch.FloatTensor) -> torch.FloatTensor:
        # Get the length of the input sequence
        sequence_length = hidden_states.shape[1]

        # Return cached positional embeddings if sequence length matches and they are cached
        if sequence_length == self.cached_sequence_length and self.cached_rotary_positional_embedding is not None:
            return self.cached_rotary_positional_embedding

        # Cache the current sequence length
        self.cached_sequence_length = sequence_length
        # Generate timestamps for positional encoding
        time_stamps = torch.arange(sequence_length, device=hidden_states.device).type_as(self.inv_freq)
        # Compute frequencies multiplied by timestamps
        freqs = torch.einsum("i,j->ij", time_stamps, self.inv_freq)
        # Concatenate frequencies to form the rotary positional embeddings
        embeddings = torch.cat((freqs, freqs), dim=-1)

        # Cache the computed rotary positional embeddings
        self.cached_rotary_positional_embedding = embeddings.unsqueeze(0)
        return self.cached_rotary_positional_embedding


class ClvpSelfAttention(nn.Module):
    """
    Multi-headed attention to combine Absolute and Rotary Positional Embeddings into a single Attention module.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout

        if hasattr(config, "max_position_embeddings"):
            max_positions = config.max_position_embeddings
            # Create a triangular bias matrix for masking future positions
            bias = torch.tril(torch.ones((max_positions, max_positions), dtype=torch.bool))
            bias = bias.view(1, 1, max_positions, max_positions)
            # Register the bias matrix as a non-persistent buffer
            self.register_buffer("bias", bias, persistent=False)

        # Projection layers for query, key, and value
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.use_attention_bias)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.use_attention_bias)
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=config.use_attention_bias)
        # Output projection layer
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    # Copied from transformers.models.clip.modeling_clip.CLIPAttention._shape
    # 定义私有方法 `_shape`，用于调整输入张量的形状以符合注意力头的需求
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        # 调整张量的形状为 (batch_size, seq_len, num_heads, head_dim)，并交换维度 1 和 2
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    # 定义前向传播方法 `forward`
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
    """
    This class defines an encoder layer for the CLVP model, comprising self-attention mechanism and MLP for processing hidden states.
    """

    def __init__(self, config: ClvpConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        # Initialize self-attention mechanism for attending to input sequences
        self.self_attn = ClvpSelfAttention(config)
        # Initialize MLP for processing and transforming hidden states
        self.mlp = ClvpEncoderMLP(config)

        # Layer normalization for input to the self-attention mechanism
        self.input_rmsnorm = ClvpRMSNorm(self.embed_dim, eps=config.layer_norm_eps)
        # Layer normalization for output after self-attention
        self.post_attention_rmsnorm = ClvpRMSNorm(self.embed_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        rotary_pos_emb: torch.FloatTensor,
        attention_mask: torch.LongTensor,
        position_ids: torch.LongTensor,
        output_attentions: Optional[bool] = False,
    ) -> torch.FloatTensor:
        # Apply layer normalization to the input hidden states
        hidden_states = self.input_rmsnorm(hidden_states)
        # Perform self-attention on the normalized hidden states
        hidden_states, attention_weights = self.self_attn(
            hidden_states, rotary_pos_emb, attention_mask, position_ids, output_attentions
        )
        # Apply layer normalization to the output of self-attention
        hidden_states = self.post_attention_rmsnorm(hidden_states)
        # Process the normalized hidden states through the MLP
        hidden_states = self.mlp(hidden_states)
        return hidden_states
    def forward(
        hidden_states: torch.FloatTensor,
        rotary_pos_emb: torch.FloatTensor,
        attention_mask: torch.FloatTensor,
        position_ids: torch.LongTensor,
        output_attentions: bool = False
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch, seq_len, embed_dim)`):
                input to the layer.
            rotary_pos_emb (`torch.FloatTensor`):
                rotary position embeddings generated by `ClvpRotaryPositionalEmbedding` module.
            attention_mask (`torch.FloatTensor` of shape `(batch, 1, tgt_len, src_len)`):
                attention mask where padding elements are indicated by very large negative values.
            position_ids (`torch.LongTensor`):
                Denotes position ids of the input tokens.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        # 保存残差连接，以便后续使用
        residual = hidden_states
    
        # 应用输入的 RMS 归一化
        hidden_states = self.input_rmsnorm(hidden_states)
    
        # 执行自注意力机制
        attention_outputs = self.self_attn(
            hidden_states=hidden_states,
            rotary_pos_emb=rotary_pos_emb,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
        )
    
        # 从自注意力输出中提取隐藏状态
        hidden_states = attention_outputs[0]
    
        # 残差连接
        hidden_states = residual + hidden_states
    
        # 保存残差连接
        residual = hidden_states
    
        # 应用注意力后的 RMS 归一化
        hidden_states = self.post_attention_rmsnorm(hidden_states)
    
        # 应用多层感知机
        hidden_states = self.mlp(hidden_states)
    
        # 残差连接
        hidden_states = residual + hidden_states
    
        # 输出结果作为元组
        outputs = (hidden_states,)
    
        # 如果需要输出注意力权重
        if output_attentions:
            outputs += (attention_outputs[-1],)
    
        return outputs
# 从transformers.models.gpt2.modeling_gpt2.GPT2MLP复制代码，并将GPT2->ClvpDecoderMLP进行替换
class ClvpDecoderMLP(nn.Module):
    def __init__(self, intermediate_size, config):
        super().__init__()
        embed_dim = config.hidden_size
        # 创建一个一维卷积层，输入维度为embed_dim，输出维度为intermediate_size
        self.c_fc = Conv1D(intermediate_size, embed_dim)
        # 创建一个一维卷积层，输入维度为intermediate_size，输出维度为embed_dim
        self.c_proj = Conv1D(embed_dim, intermediate_size)
        # 激活函数为config.activation_function指定的函数
        self.act = ACT2FN[config.activation_function]
        # Dropout层，丢弃概率为config.resid_pdrop
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, hidden_states: Optional[Tuple[torch.FloatTensor]]) -> torch.FloatTensor:
        # 使用self.c_fc进行一维卷积操作
        hidden_states = self.c_fc(hidden_states)
        # 使用激活函数self.act处理卷积后的隐藏状态
        hidden_states = self.act(hidden_states)
        # 使用self.c_proj进行一维卷积操作
        hidden_states = self.c_proj(hidden_states)
        # 使用Dropout层处理卷积后的隐藏状态
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class ClvpDecoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_size = config.hidden_size
        # 如果config.n_inner不为None，则使用config.n_inner；否则使用4 * hidden_size
        inner_dim = config.n_inner if config.n_inner is not None else 4 * hidden_size

        # Layer normalization层，输入维度为hidden_size，epsilon为config.layer_norm_epsilon
        self.input_layernorm = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)
        # ClvpSelfAttention对象，处理self-attention相关逻辑
        self.attn = ClvpSelfAttention(config)
        # Layer normalization层，输入维度为hidden_size，epsilon为config.layer_norm_epsilon
        self.post_attention_layernorm = nn.LayerNorm(hidden_size, eps=config.layer_norm_epsilon)

        # ClvpDecoderMLP对象，处理MLP层的前向传播逻辑
        self.mlp = ClvpDecoderMLP(inner_dim, config)

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
        # Layer normalization层处理hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        # 使用self.attn进行attention计算
        attn_outputs = self.attn(
            hidden_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        # 获取attention计算结果
        attn_output = attn_outputs[0]
        outputs = attn_outputs[1:]
        # 残差连接
        hidden_states = attn_output + residual

        # 保存残差连接
        residual = hidden_states
        # Layer normalization层处理hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        # 使用self.mlp进行MLP层的前向传播计算
        feed_forward_hidden_states = self.mlp(hidden_states)
        # 残差连接
        hidden_states = residual + feed_forward_hidden_states

        if use_cache:
            outputs = (hidden_states,) + outputs
        else:
            outputs = (hidden_states,) + outputs[1:]

        return outputs


class ClvpConditioningEncoder(nn.Module):
    """
    This class processes the log-mel spectrograms(extracted by the Feature Extractor) and text tokens(produced by the
    """
    # 在这里编写该类的其他逻辑和功能
    def __init__(self, config: ClvpConfig):
        super().__init__()

        # 保存文本配置和解码器配置
        self.text_config = config.text_config
        self.decoder_config = config.decoder_config

        # 创建文本token的嵌入层，维度为（词汇表大小，隐藏层大小）
        self.text_token_embedding = nn.Embedding(self.text_config.vocab_size, self.decoder_config.hidden_size)
        # 创建文本位置嵌入层，维度为（最大文本token数，隐藏层大小）
        self.text_position_embedding = nn.Embedding(
            self.decoder_config.max_text_tokens, self.decoder_config.hidden_size
        )

        # 创建用于将mel特征向量转换成隐藏层大小的卷积层
        self.mel_conv = nn.Conv1d(self.decoder_config.feature_size, self.decoder_config.hidden_size, kernel_size=1)

        # 计算用于每个注意力层之前的GroupNorm的组数
        num_groups = self.compute_groupnorm_groups(self.decoder_config.hidden_size)
        # 创建一组GroupNorm层，每个注意力层前面有一个
        self.group_norms = nn.ModuleList(
            [
                nn.GroupNorm(num_groups, self.decoder_config.hidden_size, eps=1e-5, affine=True)
                for _ in range(self.decoder_config.num_mel_attn_blocks)
            ]
        )

        # 创建一组自注意力层模块
        self.mel_attn_blocks = nn.ModuleList(
            [ClvpSelfAttention(self.decoder_config) for _ in range(self.decoder_config.num_mel_attn_blocks)]
        )

        # 设置梯度检查点为False
        self.gradient_checkpointing = False

    def compute_groupnorm_groups(self, channels: int, groups: int = 32):
        """
        计算用于nn.GroupNorm的`num_groups`的值。这个逻辑来自于官方的tortoise repository。
        链接：https://github.com/neonbjb/tortoise-tts/blob/4003544b6ff4b68c09856e04d3eff9da26d023c2/tortoise/models/arch_util.py#L26
        """
        # 根据隐藏层大小调整分组数
        if channels <= 16:
            groups = 8
        elif channels <= 64:
            groups = 16
        # 确保分组数可以整除通道数
        while channels % groups != 0:
            groups = int(groups / 2)

        # 如果分组数小于等于2，则抛出异常
        if groups <= 2:
            raise ValueError(
                f"Number of groups for the GroupNorm must be greater than 2, but it is {groups}."
                f"Please consider using a different `hidden_size`"
            )

        return groups

    def forward(
        self,
        input_features: torch.FloatTensor,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 定义配置类，用于本模型的配置管理
    config_class = ClvpConfig
    # 模型名称前缀
    base_model_prefix = "clvp"
    # 是否支持梯度检查点
    supports_gradient_checkpointing = True
    # 需要跳过设备放置的键名
    _skip_keys_device_placement = "past_key_values"

    def _init_weights(self, module):
        """Initialize the weights"""
        # 获取初始化因子
        factor = self.config.initializer_factor
        # 如果是 Embedding 层，使用正态分布初始化权重
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=factor * 0.02)
        # 如果是 Linear 或 Conv1D 层，同时初始化权重和偏置
        elif isinstance(module, (nn.Linear, Conv1D, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=factor * 0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果是 ClvpEncoderMLP 层，根据不同的层进行不同的初始化方式
        elif isinstance(module, ClvpEncoderMLP):
            # 计算输入投影的标准差和全连接层的标准差
            in_proj_std = (module.config.hidden_size**-0.5) * ((2 * module.config.num_hidden_layers) ** -0.5) * factor
            fc_std = (2 * module.config.hidden_size) ** -0.5 * factor
            # 使用正态分布初始化权重
            nn.init.normal_(module.fc1.proj.weight if getattr(module.fc1, "proj") else module.fc1.weight, std=fc_std)
            nn.init.normal_(module.fc2.weight, std=in_proj_std)
        # 如果是 ClvpEncoder 层，根据配置初始化权重
        elif isinstance(module, ClvpEncoder):
            config = self.config.text_config if hasattr(self.config, "text_config") else self.config
            factor = config.initializer_factor
            module.projection.weight.data.normal_(mean=0.0, std=factor * (config.hidden_size**-0.5))
        # 如果是 ClvpConditioningEncoder 层，使用正态分布初始化权重和偏置
        elif isinstance(module, ClvpConditioningEncoder):
            module.mel_conv.weight.data.normal_(mean=0.0, std=factor)
            module.mel_conv.bias.data.zero_()
        # 如果是 ClvpForCausalLM 层，根据名称初始化特定参数的权重
        elif isinstance(module, ClvpForCausalLM):
            for name, p in module.named_parameters():
                if name == "c_proj.weight":
                    p.data.normal_(
                        mean=0.0, std=(self.config.initializer_range / math.sqrt(2 * self.config.num_hidden_layers))
                    )
        # 如果是 LayerNorm 层，初始化偏置为零，权重为1
        if isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


CLVP_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.
"""
    Parameters:
        config ([`ClvpConfig`]): Model configuration class with all the parameters of the model.
            # 参数：config是一个包含模型所有参数的配置类。
            # 使用配置文件初始化模型时，仅加载与模型相关的配置，并不加载模型的权重。
            # 若要加载模型权重，请查看[`~PreTrainedModel.from_pretrained`]方法。
# 定义 CLVP_INPUTS_DOCSTRING 常量，包含输入参数的文档字符串
CLVP_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        input_features (`torch.FloatTensor` of shape `(batch_size, feature_size, time_dim)`):
            Indicates log mel-spectrogram representations for audio returned by [`ClvpFeatureExtractor`].
        conditioning_encoder_inputs_embeds (`torch.FloatTensor`, *optional*):
            inputs_embeds for `ClvpConditioningEncoder`. Can be used in place of `input_ids`.
        text_encoder_inputs_embeds (`torch.FloatTensor`, *optional*):
            inputs_embeds for the text encoder model passed in place of `input_ids`.
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding text token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
        return_loss (`bool`, *optional*):
            Whether or not to return the contrastive loss.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

# 定义 CLVP_DECODER_INPUTS_DOCSTRING 常量，但是为空字符串，暂时未提供相关文档说明
CLVP_DECODER_INPUTS_DOCSTRING = r"""
"""


class ClvpEncoder(ClvpPreTrainedModel):
    """
    Transformer encoder consisting of `config.num_hidden_layers` self attention layers. Each layer is a
    [`ClvpEncoderLayer`].

    Args:
        config: ClvpConfig
    """
    # 初始化函数，接受一个配置对象作为参数
    def __init__(self, config: ClvpConfig):
        # 调用父类的初始化方法
        super().__init__(config)

        # 保存配置对象到实例变量中
        self.config = config

        # 创建一个词嵌入层对象，使用config中的词汇大小和隐藏尺寸作为参数
        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)

        # 如果配置中启用了旋转位置编码，则创建ClvpRotaryPositionalEmbedding对象，否则设为None
        self.rotary_pos_emb = ClvpRotaryPositionalEmbedding(config) if config.use_rotary_embedding else None

        # 创建一个包含多个ClvpEncoderLayer对象的模块列表，列表长度由config中的隐藏层数决定
        self.layers = nn.ModuleList([ClvpEncoderLayer(config) for _ in range(config.num_hidden_layers)])

        # 创建一个SequenceSummary对象，用于序列摘要
        self.sequence_summary = SequenceSummary(config)

        # 创建一个LayerNorm层，用于最终的归一化处理，参数为隐藏尺寸和配置中的LayerNorm epsilon值
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 创建一个线性投影层，将隐藏状态映射到投影维度，无偏置项
        self.projection = nn.Linear(config.hidden_size, config.projection_dim, bias=False)

        # 梯度检查点标志设为False
        self.gradient_checkpointing = False

        # 执行初始化后的附加操作
        self.post_init()

    # 返回token_embedding属性，即词嵌入层对象
    def get_input_embeddings(self):
        return self.token_embedding

    # 设置token_embedding属性为指定的值
    def set_input_embeddings(self, value):
        self.token_embedding = value

    # 前向传播方法
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
class ClvpDecoder(ClvpPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`ClvpDecoderLayer`]
    """

    def __init__(self, config):
        super().__init__(config)

        self.config = config  # 初始化函数，保存配置信息到实例变量

        self.input_embeds_layer = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        # 创建输入的嵌入层，根据词汇表大小和隐藏层大小进行初始化

        self.position_embeds_layer = nn.Embedding(self.config.max_position_embeddings, self.config.hidden_size)
        # 创建位置嵌入层，根据最大位置嵌入数和隐藏层大小进行初始化

        self.drop = nn.Dropout(self.config.embd_pdrop)  # 创建一个丢弃层，使用配置中的丢弃概率

        self.layers = nn.ModuleList([ClvpDecoderLayer(self.config) for _ in range(self.config.num_hidden_layers)])
        # 创建一个包含多个 ClvpDecoderLayer 的模块列表，数量由配置中的隐藏层数决定

        self.layer_norm = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_epsilon)
        # 创建一个层归一化层，使用隐藏层大小和配置中的归一化参数进行初始化

        self.gradient_checkpointing = False  # 初始化梯度检查点标志为 False

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        return self.input_embeds_layer  # 返回输入嵌入层

    def set_input_embeddings(self, new_embeddings):
        self.input_embeds_layer = new_embeddings  # 设置新的输入嵌入层

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        """
        for layer, heads in heads_to_prune.items():
            self.layers[layer].attn.prune_heads(heads)
        # 剪枝模型中的注意力头部

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
        Performs forward pass of the decoder model.

        Args:
            input_ids: Optionally provided input IDs.
            attention_mask: Optionally provided attention mask.
            token_type_ids: Optionally provided token type IDs.
            position_ids: Optionally provided position IDs.
            head_mask: Optionally provided head mask.
            past_key_values: Optionally provided past key values.
            inputs_embeds: Optionally provided input embeddings.
            use_cache: Optionally use cache.
            output_attentions: Optionally output attentions.
            output_hidden_states: Optionally output hidden states.
            return_dict: Optionally return as dictionary.

        Returns:
            Model output.
        """
        pass  # 前向传播函数声明，暂未实现具体逻辑

@add_start_docstrings(
    "The bare Clvp decoder model outputting raw hidden-states without any specific head on top.",
    CLVP_START_DOCSTRING,
)
class ClvpModel(ClvpPreTrainedModel):
    def __init__(self, config: ClvpDecoderConfig):
        super().__init__(config)
        self.config = config  # 初始化函数，保存配置信息到实例变量
        self.decoder = ClvpDecoder(self.config)  # 创建 ClvpDecoder 实例作为解码器

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        return self.decoder.input_embeds_layer  # 返回解码器的输入嵌入层

    def set_input_embeddings(self, value):
        self.decoder.input_embeds_layer = value  # 设置解码器的新输入嵌入层

    def get_decoder(self):
        return self.decoder  # 返回解码器实例

    @add_start_docstrings_to_model_forward(CLVP_DECODER_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入的token ID序列，可选的长整型张量
        attention_mask: Optional[torch.FloatTensor] = None,  # 注意力掩码张量，可选的浮点数张量
        token_type_ids: Optional[torch.LongTensor] = None,  # token类型ID张量，可选的长整型张量
        position_ids: Optional[torch.LongTensor] = None,  # 位置ID张量，可选的长整型张量
        head_mask: Optional[torch.FloatTensor] = None,  # 头部掩码张量，可选的浮点数张量
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,  # 过去的键值对元组，可选的张量元组
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入嵌入张量，可选的浮点数张量
        use_cache: Optional[bool] = None,  # 是否使用缓存，可选的布尔值
        output_attentions: Optional[bool] = None,  # 是否输出注意力，可选的布尔值
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可选的布尔值
        return_dict: Optional[bool] = None,  # 是否返回字典形式的结果，可选的布尔值
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:  # 返回值可以是元组或BaseModelOutputWithPastAndCrossAttentions对象

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果没有指定output_attentions，则使用self.config中的设置

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果没有指定output_hidden_states，则使用self.config中的设置

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        # 如果没有指定use_cache，则使用self.config中的设置

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 如果没有指定return_dict，则使用self.config中的设置

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        # 解码器的输出包括(dec_features, past_key_value, dec_hidden, dec_attn)
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

        if not return_dict:
            # 如果不返回字典形式的结果，则直接返回解码器的输出
            return decoder_outputs

        # 如果返回字典形式的结果，则构造BaseModelOutputWithPastAndCrossAttentions对象
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            hidden_states=decoder_outputs.hidden_states,
            attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
        )
@add_start_docstrings(
    "The CLVP decoder model with a language modelling head on top.",
    CLVP_START_DOCSTRING,
)
class ClvpForCausalLM(ClvpPreTrainedModel):
    # CLVPForCausalLM 类的构造函数，初始化模型配置和相关组件
    def __init__(self, config):
        super().__init__(config)

        # 存储传入的配置信息
        self.config = config
        # 使用传入的配置初始化 CLVPModel 类的实例，作为模型的主体
        self.model = ClvpModel(self.config)

        # 初始化用于最终归一化的层
        self.final_norm = nn.LayerNorm(self.config.hidden_size)
        # 初始化用于语言模型头部的线性层
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=True)

        # 调用后初始化方法，用于权重初始化和最终处理
        self.post_init()

    # 返回模型解码器的输入嵌入层
    def get_input_embeddings(self):
        return self.model.decoder.input_embeds_layer

    # 设置模型解码器的输入嵌入层
    def set_input_embeddings(self, new_embeddings):
        self.model.decoder.input_embeds_layer = new_embeddings

    # 辅助方法：准备模型的输入，接受输入张量、开始词标识符和模型关键字参数
    def _prepare_model_inputs(
        self,
        inputs: Optional[torch.Tensor] = None,
        bos_token_id: Optional[int] = None,
        model_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ):
        ...

    # 生成推断过程的输入准备方法，接受输入的标识符、过去的键值对、输入嵌入和条件嵌入等参数
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, inputs_embeds=None, conditioning_embeds=None, **kwargs
    ):
        ...
        ):
        # 计算输入的序列长度
        input_ids_length = input_ids.shape[-1]
        # 获取额外的关键字参数中的 `token_type_ids`
        token_type_ids = kwargs.get("token_type_ids", None)
        # 如果有过去的键值对 `past_key_values`
        if past_key_values:
            # 获取过去状态的长度
            past_length = past_key_values[0][0].shape[2]

            # 一些生成方法已经只传递最后一个输入 ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 默认的行为是保留最后一个 ID
                remove_prefix_length = input_ids.shape[1] - 1

            # 截取输入序列，只保留后部分
            input_ids = input_ids[:, remove_prefix_length:]
            # 如果有 `token_type_ids`，也相应地截取
            if token_type_ids is not None:
                token_type_ids = token_type_ids[:, -input_ids.shape[1] :]

        # 获取关键字参数中的 `attention_mask` 和 `position_ids`
        attention_mask = kwargs.get("attention_mask", None)
        position_ids = kwargs.get("position_ids", None)

        # 如果 `attention_mask` 不为空且 `position_ids` 为空
        if attention_mask is not None and position_ids is None:
            # 动态生成 `position_ids` 用于批量生成
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            # 如果有 `past_key_values`，则只保留最后一个位置 ID
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)
        else:
            # 否则置 `position_ids` 为空
            position_ids = None

        # 如果 `conditioning_embeds` 和 `past_key_values` 都不为空
        if conditioning_embeds is not None and past_key_values is not None:
            # 直接设置 `position_ids` 为输入序列长度的张量
            position_ids = torch.tensor([input_ids_length], dtype=torch.long, device=input_ids.device)

        # 如果传入了 `inputs_embeds`，且没有 `past_key_values`
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # 更新 `model_inputs` 字典
        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "position_ids": position_ids,
                "token_type_ids": token_type_ids,
            }
        )
        # 返回最终的 `model_inputs`
        return model_inputs

    # 添加预定义的文档字符串到模型的前向方法
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

        # 根据参数或者配置文件设置是否输出注意力权重
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 根据参数或者配置文件设置是否输出隐藏状态
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 根据参数或者配置文件设置是否使用缓存
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        # 根据参数或者配置文件设置是否返回字典格式的输出
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用模型进行预测
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

        # 获取模型的隐藏状态
        hidden_states = outputs[0]

        # 对隐藏状态进行归一化处理
        lm_logits = self.final_norm(hidden_states)
        # 应用语言模型的头部进行最终的逻辑回归计算
        lm_logits = self.lm_head(lm_logits)

        # 初始化损失值
        loss = None
        # 如果存在标签数据，则计算损失值
        if labels is not None:
            # 将标签数据移到与 lm_logits 相同的设备上
            labels = labels.to(lm_logits.device)
            # 将 logits 向左移动一个位置，用于预测下一个 token
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # 将预测值与标签展平，计算交叉熵损失
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # 如果不需要返回字典格式的输出，则按照元组格式返回结果
        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果需要返回字典格式的输出，则封装成 CausalLMOutputWithCrossAttentions 对象返回
        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    @staticmethod
    # 从 GPT2LMHeadModel._reorder_cache 复制过来的静态方法
    def _reorder_cache(
        past_key_values: Tuple[Tuple[torch.Tensor]], beam_idx: torch.Tensor
        """
        This function is used to re-order the `past_key_values` cache if [`~PreTrainedModel.beam_search`] or
        [`~PreTrainedModel.beam_sample`] is called. This is required to match `past_key_values` with the correct
        beam_idx at every generation step.
        """
        # 返回一个元组的元组，每个元组包含重新排序后的 `past_key_values` 中的每一层的状态
        return tuple(
            # 对于 `past_key_values` 中的每一层的状态，使用 `beam_idx` 来重新选择对应的状态，并移到相应设备上
            tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past)
            # 对于 `past_key_values` 中的每一层，进行上述操作并组成元组
            for layer_past in past_key_values
        )
# 为 CLVP 生成条件生成模型的类声明文档字符串，描述其包含文本编码器、语音编码器和语音解码器模型的结构和功能
@add_start_docstrings(
    "The composite CLVP model with a text encoder, speech encoder and speech decoder model."
    "The speech decoder model generates the speech_ids from the text and the text encoder and speech encoder works"
    "together to filter out the best speech_ids.",
    CLVP_START_DOCSTRING,
)
class ClvpModelForConditionalGeneration(ClvpPreTrainedModel):
    # 设置配置类为 ClvpConfig
    config_class = ClvpConfig

    # 初始化方法，接受一个 ClvpConfig 类型的参数 config
    def __init__(self, config: ClvpConfig):
        # 调用父类 ClvpPreTrainedModel 的初始化方法
        super().__init__(config)

        # 检查文本配置是否为 ClvpEncoderConfig 类型，若不是则抛出 ValueError 异常
        if not isinstance(config.text_config, ClvpEncoderConfig):
            raise ValueError(
                "config.text_config is expected to be of type `ClvpEncoderConfig` but is of type"
                f" {type(config.text_config)}."
            )

        # 检查语音配置是否为 ClvpEncoderConfig 类型，若不是则抛出 ValueError 异常
        if not isinstance(config.speech_config, ClvpEncoderConfig):
            raise ValueError(
                "config.speech_config is expected to be of type `ClvpEncoderConfig` but is of type"
                f" {type(config.speech_config)}."
            )

        # 检查解码器配置是否为 ClvpDecoderConfig 类型，若不是则抛出 ValueError 异常
        if not isinstance(config.decoder_config, ClvpDecoderConfig):
            raise ValueError(
                "config.decoder_config is expected to be of type `ClvpDecoderConfig` but is of type"
                f" {type(config.decoder_config)}."
            )

        # 创建 CLVP 条件编码器对象并赋值给 self.conditioning_encoder
        self.conditioning_encoder = ClvpConditioningEncoder(config)

        # 创建 CLVP 语音解码器模型对象并赋值给 self.speech_decoder_model
        self.speech_decoder_model = ClvpForCausalLM(config.decoder_config)

        # 创建 CLVP 文本编码器模型对象并赋值给 self.text_encoder_model
        self.text_encoder_model = ClvpEncoder(config.text_config)

        # 创建 CLVP 语音编码器模型对象并赋值给 self.speech_encoder_model
        self.speech_encoder_model = ClvpEncoder(config.speech_config)

        # 创建一个可学习参数 logit_scale，其值初始化为 config 中指定的 logit_scale_init_value
        self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))

        # 调用后处理方法，用于初始化权重和应用最终处理
        self.post_init()

    # 从原始代码库中提取的注释，指向具体代码位置的链接
    # 链接地址: https://github.com/neonbjb/tortoise-tts/blob/4003544b6ff4b68c09856e04d3eff9da26d023c2/tortoise/api.py#L117
    def fix_speech_decoder_output(self, speech_ids: torch.LongTensor) -> torch.LongTensor:
        """
        This method modifies the output of the decoder model, such as replacing the `eos_token_id` and changing the
        last few tokens of each sequence.

        Args:
            speech_ids (`torch.LongTensor`):
                This refers to the output of the decoder model.
        """
        # 获取解码器修正代码列表
        decoder_fixing_codes = self.config.decoder_config.decoder_fixing_codes
        
        # 去掉每个序列开头的第一个 token（通常是起始 token）
        speech_ids = speech_ids[:, 1:]

        # 找到所有结束 token 的索引位置
        stop_token_indices = torch.where(speech_ids == self.speech_decoder_model.config.eos_token_id, 1, 0)
        
        # 使用 decoder_fixing_codes[0] 替换所有结束 token 的位置
        speech_ids = torch.masked_fill(speech_ids, mask=stop_token_indices.bool(), value=decoder_fixing_codes[0])

        # 遍历每个序列的结束 token 索引
        for i, each_seq_stop_token_index in enumerate(stop_token_indices):
            # 如果某个序列中没有找到结束 token，则跳过对该序列的处理
            if each_seq_stop_token_index.sum() == 0:
                continue

            # 找到当前序列中第一个结束 token 的位置
            stm = each_seq_stop_token_index.argmax()
            
            # 将该位置及之后的 token 替换为 decoder_fixing_codes[0]
            speech_ids[i, stm:] = decoder_fixing_codes[0]
            
            # 如果序列长度允许，将序列末尾的最后三个 token 替换为指定的 decoder_fixing_codes[1:]
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

        # 使用 text_encoder_model 对象进行文本编码器的前向传播，生成文本嵌入
        outputs = self.text_encoder_model(
            input_ids=input_ids,
            inputs_embeds=text_encoder_inputs_embeds,
            attention_mask=attention_mask,
        )

        # 返回经过投影层处理后的文本嵌入
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
    # 定义类中的前向传播方法，接收多个输入参数
    def forward(
        self,
        input_ids: torch.LongTensor = None,  # 输入的 token IDs，数据类型为 LongTensor
        input_features: torch.FloatTensor = None,  # 输入的特征数据，数据类型为 FloatTensor
        conditioning_encoder_inputs_embeds: Optional[torch.FloatTensor] = None,  # 条件编码器输入的嵌入向量，可选的 FloatTensor
        text_encoder_inputs_embeds: Optional[torch.FloatTensor] = None,  # 文本编码器输入的嵌入向量，可选的 FloatTensor
        attention_mask: Optional[torch.LongTensor] = None,  # 注意力掩码，可选的 LongTensor
        return_loss: Optional[bool] = None,  # 是否返回损失，可选的布尔值
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可选的布尔值
        output_attentions: Optional[bool] = False,  # 是否输出注意力权重，可选的布尔值，默认为 False
        return_dict: Optional[bool] = None,  # 是否以字典形式返回结果，可选的布尔值
    ):
    
    # 使用装饰器标记此方法不会计算梯度
    @torch.no_grad()
    # 定义类中的生成方法，接收多个输入参数
    def generate(
        self,
        input_ids: torch.LongTensor = None,  # 输入的 token IDs，数据类型为 LongTensor
        input_features: torch.FloatTensor = None,  # 输入的特征数据，数据类型为 FloatTensor
        attention_mask: Optional[torch.LongTensor] = None,  # 注意力掩码，可选的 LongTensor
        generation_config: Optional[GenerationConfig] = None,  # 生成配置，可选的 GenerationConfig 对象
        pad_to_max_mel_tokens: Optional[int] = None,  # 填充到最大 mel tokens 的数量，可选的整数
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可选的布尔值
        **kwargs,  # 其他关键字参数
```