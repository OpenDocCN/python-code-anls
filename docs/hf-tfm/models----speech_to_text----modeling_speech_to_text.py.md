# `.\transformers\models\speech_to_text\modeling_speech_to_text.py`

```py
# 设置文件编码为 utf-8
# 版权声明，版权归 Fairseq 作者和 HuggingFace 公司所有
#
# 根据 Apache License, Version 2.0 进行许可
# 在遵守许可证的情况下才能使用此文件
# 可以从 http://www.apache.org/licenses/LICENSE-2.0 获取许可证的副本
#
# 在适用法律或书面同意的情况下，软件将按 "原样" 分发
# 没有任何担保或条件，无论是明示的还是隐含的
# 关于特定语言，授权的限制和
# 限制
# 请参阅许可证，了解特定语言的权限和限制
""" PyTorch 语音转文本模型。"""

import math
from typing import Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss

from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_speech_to_text import Speech2TextConfig

# 获取名为 __name__ 的日志记录器
logger = logging.get_logger(__name__)

# 用于文档的配置名称
_CONFIG_FOR_DOC = "Speech2TextConfig"

# 预训练的语音转文本模型存档列表
SPEECH_TO_TEXT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/s2t-small-librispeech-asr",
    # 查看所有的语音转文本模型 https://huggingface.co/models?filter=speech_to_text
]


# 从 transformers.models.bart.modeling_bart 中复制 shift_tokens_right 函数
def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    将输入的 token 向右移动一个位置。
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # 将标签中可能存在的 -100 值替换为 `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


class Conv1dSubsampler(nn.Module):
    """
    一维卷积子采样器：一堆 1D 卷积（沿时间维度）后跟通过门控线性单元进行的非线性激活
    """
    # 初始化方法，接收配置参数
    def __init__(self, config):
        # 调用父类初始化方法
        super(Conv1dSubsampler, self).__init__()
        # 保存配置参数
        self.config = config
        # 获取卷积层数量
        self.num_layers = config.num_conv_layers
        # 输入通道数为每个通道的特征数乘以输入通道数目
        self.in_channels = config.input_feat_per_channel * config.input_channels
        # 中间通道数为配置的卷积通道数
        self.mid_channels = config.conv_channels
        # 输出通道数为配置的模型维度
        self.out_channels = config.d_model
        # 卷积核大小为配置的卷积核大小列表
        self.kernel_sizes = config.conv_kernel_sizes
    
        # 创建一组卷积层，存储在 ModuleList 中
        self.conv_layers = nn.ModuleList(
            # 遍历卷积核大小列表，创建相应的卷积层
            nn.Conv1d(
                # 如果是第一层卷积，输入通道数为输入通道数，否则为中间通道数的一半
                self.in_channels if i == 0 else self.mid_channels // 2,
                # 如果不是最后一层卷积，输出通道数为中间通道数，否则为输出通道数的两倍
                self.mid_channels if i < self.num_layers - 1 else self.out_channels * 2,
                kernel_size=k,  # 设置卷积核大小
                stride=2,  # 设置步长为2
                padding=k // 2,  # 设置填充大小为卷积核大小的一半
            )
            for i, k in enumerate(self.kernel_sizes)  # 遍历卷积核大小列表的索引和值
        )
    
    # 前向传播方法，接收输入特征
    def forward(self, input_features):
        # 调整输入特征的维度，转置为 B x (C x D) x T 的形式
        hidden_states = input_features.transpose(1, 2).contiguous()
        # 遍历卷积层进行前向传播
        for conv in self.conv_layers:
            hidden_states = conv(hidden_states)  # 卷积操作
            hidden_states = nn.functional.glu(hidden_states, dim=1)  # GLU激活
        # 将最终输出结果的维度再次调整为 T x B x (C x D) 的形式
        hidden_states = hidden_states.transpose(1, 2).contiguous()
        # 返回最终的隐藏状态
        return hidden_states
class Speech2TextSinusoidalPositionalEmbedding(nn.Module):
    """此模块生成任意长度的正弦位置嵌入。"""

    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None):
        super().__init__()
        # 设置偏移量，默认为2
        self.offset = 2
        # 存储嵌入维度
        self.embedding_dim = embedding_dim
        # 存储填充索引，如果存在
        self.padding_idx = padding_idx
        # 生成位置嵌入权重
        self.make_weights(num_positions + self.offset, embedding_dim, padding_idx)

    def make_weights(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        # 获取嵌入权重
        emb_weights = self.get_embedding(num_embeddings, embedding_dim, padding_idx)
        # 如果已有权重参数，则将权重转换为与参数一致的类型和设备
        if hasattr(self, "weights"):
            emb_weights = emb_weights.to(dtype=self.weights.dtype, device=self.weights.device)

        # 将权重设置为不可训练的参数
        self.weights = nn.Parameter(emb_weights)
        self.weights.requires_grad = False
        # 断开权重的计算图
        self.weights.detach_()

    @staticmethod
    def get_embedding(num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        """
        构建正弦嵌入。与 tensor2tensor 中的实现匹配，但与论文 "Attention Is All You Need" 的第 3.5 节描述略有不同。
        """
        # 计算嵌入维度的一半
        half_dim = embedding_dim // 2
        # 计算规模因子
        emb = math.log(10000) / (half_dim - 1)
        # 计算缩放因子
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        # 根据嵌入维度和嵌入数创建正弦和余弦嵌入
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        # 如果嵌入维度为奇数，则添加0填充
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        # 如果存在填充索引，将相应嵌入设置为0
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        # 将嵌入转换为默认数据类型
        return emb.to(torch.get_default_dtype())

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, past_key_values_length: int = 0):
        bsz, seq_len = input_ids.size()
        # 根据输入标记ID创建位置ID，任何填充标记保持填充状态
        position_ids = self.create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length).to(
            input_ids.device
        )

        # 如果需要，扩展嵌入
        max_pos = self.padding_idx + 1 + seq_len
        if max_pos > self.weights.size(0):
            self.make_weights(max_pos + self.offset, self.embedding_dim, self.padding_idx)

        # 根据位置ID选择权重，并重塑为输入大小，返回权重
        return self.weights.index_select(0, position_ids.view(-1)).view(bsz, seq_len, -1).detach()

    def create_position_ids_from_input_ids(
        self, input_ids: torch.Tensor, padding_idx: int, past_key_values_length: Optional[int] = 0):
    ):
        """
        Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding
        symbols are ignored. This is modified from fairseq's `utils.make_positions`.

        Args:
            x: torch.Tensor x:  # 输入张量
        Returns: torch.Tensor  # 返回值是张量
        """
        # 这里的一系列转换和类型转换经过精心平衡，既可以与 ONNX 导出一起使用，也可以与 XLA 一起使用。
        mask = input_ids.ne(padding_idx).int()  # 创建一个掩码张量，将非填充符号设为1，填充符号设为0
        incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask  # 根据掩码计算递增位置索引
        return incremental_indices.long() + padding_idx  # 返回长整型张量的递增位置索引，加上填充索引作为位置编号
# 从transformers.models.bart.modeling_bart.BartAttention复制而来，将Bart->Speech2Text
class Speech2TextAttention(nn.Module):
    """来自'Attention Is All You Need'论文的多头注意力"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[Speech2TextConfig] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # 投影键
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # 投影数值
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # 投影查询
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)  # 输出投影

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
SPEECH_TO_TEXT_ATTENTION_CLASSES = {"eager": Speech2TextAttention}

# 从transformers.models.mbart.modeling_mbart.MBartEncoderLayer复制而来，将MBart->Speech2Text, MBART->SPEECH_TO_TEXT
class Speech2TextEncoderLayer(nn.Module):
    def __init__(self, config: Speech2TextConfig):
        super().__init__()
        self.embed_dim = config.d_model  # 嵌入维度

        self.self_attn = SPEECH_TO_TEXT_ATTENTION_CLASSES[config._attn_implementation](
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
            config=config,
        )   
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)  # 自注意力层归一化
        self.dropout = config.dropout  # 随机丢弃概率
        self.activation_fn = ACT2FN[config.activation_function]  # 激活函数
        self.activation_dropout = config.activation_dropout  # 激活函数随机丢弃
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)  # 全连接层1
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)  # 全连接层2
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)  # 最终归一化层
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        # 保存残差连接之前的隐藏状态
        residual = hidden_states
        # 使用 self-attention 层的 layer normalization 处理隐藏状态
        hidden_states = self.self_attn_layer_norm(hidden_states)
        # 通过 self-attention 层计算新的隐藏状态和注意力权重
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        # 对隐藏状态进行 dropout 处理
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # 将残差连接回新的隐藏状态
        hidden_states = residual + hidden_states

        # 保存残差连接之前的隐藏状态
        residual = hidden_states
        # 使用最终层的 layer normalization 处理隐藏状态
        hidden_states = self.final_layer_norm(hidden_states)
        # 使用激活函数处理全连接层 1 的结果
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        # 对隐藏状态进行激活函数的 dropout 处理
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        # 使用全连接层 2 处理隐藏状态
        hidden_states = self.fc2(hidden_states)
        # 对隐藏状态进行 dropout 处理
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        # 将残差连接回新的隐藏状态
        hidden_states = residual + hidden_states

        # 如果隐藏状态是 torch.float16 类型且包含无穷大或 NaN 值，则进行截断处理
        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        # 将处理后的隐藏状态存储到 outputs 变量中
        outputs = (hidden_states,)

        # 如果需要输出注意力权重信息，则将注意力权重加入到 outputs 中
        if output_attentions:
            outputs += (attn_weights,)

        # 返回 outputs 变量
        return outputs
# 从transformers.models.mbart.modeling_mbart.MBartDecoderLayer复制而来，将MBart改为Speech2Text，MBART改为SPEECH_TO_TEXT
class Speech2TextDecoderLayer(nn.Module):
    def __init__(self, config: Speech2TextConfig):
        super().__init__()
        self.embed_dim = config.d_model

        # 使用config中的配置参数初始化自注意力机制
        self.self_attn = SPEECH_TO_TEXT_ATTENTION_CLASSES[config._attn_implementation](
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            is_causal=True,
            config=config,
        )
        # 初始化dropout参数
        self.dropout = config.dropout
        # 根据配置参数选择激活函数
        self.activation_fn = ACT2FN[config.activation_function]
        # 初始化激活函数的dropout参数
        self.activation_dropout = config.activation_dropout

        # 初始化自注意力机制的LayerNorm
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # 使用config中的配置参数初始化编码器-解码器注意力机制
        self.encoder_attn = SPEECH_TO_TEXT_ATTENTION_CLASSES[config._attn_implementation](
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            config=config,
        )
        # 初始化编码器-解码器注意力机制的LayerNorm
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        # 初始化全连接层1
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        # 初始化全连接层2
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        # 初始化最终的LayerNorm
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
# Speech2TextPreTrainedModel的基类为PreTrainedModel
class Speech2TextPreTrainedModel(PreTrainedModel):
    # 设定配置类为Speech2TextConfig
    config_class = Speech2TextConfig
    # 指定基本模型前缀为"model"
    base_model_prefix = "model"
    # 确定主要输入名称为"input_features"
    main_input_name = "input_features"
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    # 初始化权重
    def _init_weights(self, module):
        std = self.config.init_std
        # 对于Linear和Conv1d层，初始化权重
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        # 对于Embedding层，初始化权重
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    # 获取特征提取的输出长度
    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        Computes the output length of the convolutional layers
        """
        for i in range(self.config.num_conv_layers):
            input_lengths = (input_lengths - 1) // 2 + 1

        return input_lengths
    # 获取特征向量的注意力掩码
    def _get_feature_vector_attention_mask(self, feature_vector_length, attention_mask):
        # 生成一个3D的注意力掩码，因为输入特征的形状
        # 如果是这种情况，将其转换为2D
        if len(attention_mask.shape) > 2:
            attention_mask = attention_mask[:, :, -1]

        # 获取注意力掩码之和对应的特征提取输出长度
        subsampled_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1))
        # 获取批量大小
        bsz = attention_mask.size()[0]
        # 新建一个形状为 (bsz, feature_vector_length) 的零张量，与注意力掩码的数据类型和设备相同
        attention_mask = torch.zeros(
            (bsz, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
        )

        # 这两个操作确保所有输出长度索引之前的所有值都被关注
        attention_mask[(torch.arange(bsz, device=attention_mask.device), subsampled_lengths - 1)] = 1
        # 翻转注意力掩码，累积和，再翻转，转换为长整型
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).long()
        return attention_mask
# 定义变量，存储语音转文本模型的开始文档字符串
SPEECH_TO_TEXT_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Speech2TextConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 定义变量，存储语音转文本模型的输入文档字符串
SPEECH_TO_TEXT_INPUTS_DOCSTRING = r"""
"""

# 定义类 Speech2TextEncoder，继承自 Speech2TextPreTrainedModel
class Speech2TextEncoder(Speech2TextPreTrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`Speech2TextEncoderLayer`].

    Args:
        config: Speech2TextConfig
        embed_tokens (nn.Embedding): output embedding
    """

    # 初始化方法
    def __init__(self, config: Speech2TextConfig):
        # 调用父类的初始化方法
        super().__init__(config)

        # 指定变量
        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        # 设定 embed_dim 变量
        embed_dim = config.d_model
        # 设定 padding_idx 变量
        self.padding_idx = config.pad_token_id
        # 设定 max_source_positions 变量
        self.max_source_positions = config.max_source_positions
        # 设定 embed_scale 变量
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        # 创建 Conv1dSubsampler 对象
        self.conv = Conv1dSubsampler(config)

        # 创建 Speech2TextSinusoidalPositionalEmbedding 对象
        self.embed_positions = Speech2TextSinusoidalPositionalEmbedding(
            self.max_source_positions,
            embed_dim,
            self.padding_idx,
        )
        # 创建 nn.ModuleList，存储多个 Speech2TextEncoderLayer 对象
        self.layers = nn.ModuleList([Speech2TextEncoderLayer(config) for _ in range(config.encoder_layers)])
        # 创建 nn.LayerNorm 对象
        self.layer_norm = nn.LayerNorm(config.d_model)

        # 初始化梯度检查点变量
        self.gradient_checkpointing = False
        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播方法
    def forward(
        self,
        input_features,
        attention_mask=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
class Speech2TextDecoder(Speech2TextPreTrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`Speech2TextDecoderLayer`]

    Args:
        config: Speech2TextConfig
        embed_tokens (nn.Embedding): output embedding
    """
    def __init__(self, config: Speech2TextConfig):
        # 在继承自父类的初始化方法之前，执行子类自身的初始化方法
        super().__init__(config)
        # 从配置中获取参数值并赋值给类的属性
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_target_positions
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0
    
        # 创建一个嵌入层对象，用于将输入的词索引映射为词向量
        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)
    
        # 创建一个语音到文本的正弦位置编码层对象
        self.embed_positions = Speech2TextSinusoidalPositionalEmbedding(
            self.max_target_positions,
            config.d_model,
            self.padding_idx,
        )
    
        # 使用列表推导创建一组语音到文本的解码层，并组成一个模块列表对象
        self.layers = nn.ModuleList([Speech2TextDecoderLayer(config) for _ in range(config.decoder_layers)])
    
        # 创建一个LayerNorm对象，用于归一化输入张量
        self.layer_norm = nn.LayerNorm(config.d_model)
    
        # 设置梯度检查点开关
        self.gradient_checkpointing = False
        # 初始化模型的权重并进行最后的处理
        self.post_init()
    
    def get_input_embeddings(self):
        # 返回嵌入层对象
        return self.embed_tokens
    
    def set_input_embeddings(self, value):
        # 设置嵌入层对象
        self.embed_tokens = value
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    
    
    注释：
# Speech2TextModel 类，继承自 Speech2TextPreTrainedModel 类，用于输出不带特定头部的原始隐藏状态
class Speech2TextModel(Speech2TextPreTrainedModel):
    def __init__(self, config: Speech2TextConfig):
        # 调用父类的初始化方法
        super().__init__(config)

        # 初始化编码器和解码器
        self.encoder = Speech2TextEncoder(config)
        self.decoder = Speech2TextDecoder(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入
    def get_input_embeddings(self):
        return self.decoder.embed_tokens

    # 设置输入嵌入
    def set_input_embeddings(self, value):
        self.decoder.embed_tokens = value

    # 获取编码器
    def get_encoder(self):
        return self.encoder

    # 获取解码器
    def get_decoder(self):
        return self.decoder

    # 前向传播方法
    @add_start_docstrings_to_model_forward(SPEECH_TO_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_features: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ):



# Speech2TextForConditionalGeneration 类，继承自 Speech2TextPreTrainedModel 类，用于具有语言建模头部的 Speech2Text 模型，可用于摘要
class Speech2TextForConditionalGeneration(Speech2TextPreTrainedModel):
    base_model_prefix = "model"
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Speech2TextConfig):
        # 调用父类的初始化方法
        super().__init__(config)
        
        # 初始化 Speech2TextModel
        self.model = Speech2TextModel(config)
        # 初始化语言模型头部
        self.lm_head = nn.Linear(config.d_model, self.config.vocab_size, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取编码器
    def get_encoder(self):
        return self.model.get_encoder()

    # 获取解码器
    def get_decoder(self):
        return self.model.get_decoder()

    # 获取输出嵌入
    def get_output_embeddings(self):
        return self.lm_head

    # 设置输出嵌入
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 前向传播方法
    @add_start_docstrings_to_model_forward(SPEECH_TO_TEXT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    # 定义一个 forward 方法，用于模型的前向计算
    def forward(
        self,
        input_features: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
    
    # 准备生成输入，根据输入准备模型生成需要的信息
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
        # 如果有过去的关键值，截取 decoder_input_ids 的最后一个 token
        if past_key_values is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "encoder_outputs": encoder_outputs,
            "past_key_values": past_key_values,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # 更改此处以避免缓存（可能用于调试）
        }

    # 重新排序缓存中的 past_key_values 根据 beam_idx 参数
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
```