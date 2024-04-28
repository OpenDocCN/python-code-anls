# `.\transformers\models\xglm\modeling_xglm.py`

```
# 设置文件编码为 utf-8
# 版权声明，版权归 The Fairseq Authors 和 The HuggingFace Inc. 团队所有，保留所有权利
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定或经书面同意，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 在适用法律要求或书面同意的情况下，本软件按"原样"分发，没有任何明示或暗示的担保或条件。请查看许可证以获取特定语言的权限和限制
""" PyTorch XGLM model."""

# 导入需要的库
import math
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN  # 导入激活函数
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask  # 导入处理注意力掩码的函数
from ...modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions  # 导入模型输出
from ...modeling_utils import PreTrainedModel  # 导入预训练模型的工具类
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging  # 导入工具函数和日志记录
from .configuration_xglm import XGLMConfig  # 导入 XGLM 配置类

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 用于文档的常量
_CHECKPOINT_FOR_DOC = "facebook/xglm-564M"  # 文档中的模型检查点
_CONFIG_FOR_DOC = "XGLMConfig"  # 文档中的配置文件

# 预训练模型存档列表
XGLM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/xglm-564M",
    # 查看所有 XGLM 模型列表 https://huggingface.co/models?filter=xglm
]

# XGLM 模型的文档字符串
XGLM_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`XGLMConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

XGLM_INPUTS_DOCSTRING = r"""
"""


class XGLMSinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None):
        super().__init__()
        self.offset = 2  # 偏移量
        self.embedding_dim = embedding_dim  # 嵌入维度
        self.padding_idx = padding_idx  # 填充索引
        self.make_weights(num_positions + self.offset, embedding_dim, padding_idx)  # 生成权重
    # 创建权重矩阵，用于存储嵌入层的参数
    def make_weights(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        # 调用 get_embedding 方法获取嵌入权重
        emb_weights = self.get_embedding(num_embeddings, embedding_dim, padding_idx)
        # 如果 self 对象存在 weights 属性
        if hasattr(self, "weights"):
            # 将获取的嵌入权重转换为与 self.weights 相同的数据类型和设备类型
            emb_weights = emb_weights.to(dtype=self.weights.dtype, device=self.weights.device)
        
        # 在 self 对象上注册名为 "weights" 的缓冲区
        # persistent=False 表示不会保存在模型的状态字典中
        self.register_buffer("weights", emb_weights, persistent=False)

    @staticmethod
    def get_embedding(num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        """
        Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly from the description in Section 3.5 of
        "Attention Is All You Need".
        """
        # 计算嵌入维度一半的值
        half_dim = embedding_dim // 2
        # 计算 emb 的值
        emb = math.log(10000) / (half_dim - 1)
        # 计算对数间隔后的值
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        # 计算乘法结果
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        # 拼接正弦和余弦值
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        # 如果 embedding_dim 是奇数
        if embedding_dim % 2 == 1:
            # 补零
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        # 如果存在 padding_idx，将其对应的行设为零
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        
        # 将结果转换为默认的数据类型
        return emb.to(torch.get_default_dtype())

    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor = None, past_key_values_length: int = 0):
        # 获取 position_ids 的大小
        bsz, seq_len = position_ids.size()
        # 对 position_ids 加上偏移量
        position_ids += self.offset

        # 如果 max_pos 大于 weights 的行数，调用 make_weights 方法
        max_pos = 2 + seq_len + past_key_values_length
        if max_pos > self.weights.size(0):
            self.make_weights(max_pos, self.embedding_dim, self.padding_idx)

        # 选择指定位置索引的权重，然后将结果重新组织为合适的形状
        return self.weights.index_select(0, position_ids.view(-1)).view(bsz, seq_len, self.weights.shape[-1]).detach()
class XGLMAttention(nn.Module):
    """从《Attention Is All You Need》论文中的多头注意力机制"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

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
class XGLMDecoderLayer(nn.Module):
    def __init__(self, config: XGLMConfig):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = XGLMAttention(
            embed_dim=self.embed_dim,
            num_heads=config.attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        if config.add_cross_attention:
            self.encoder_attn = XGLMAttention(
                embed_dim=self.embed_dim,
                num_heads=config.attention_heads,
                dropout=config.attention_dropout,
                is_decoder=True,
            )
            self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.ffn_dim)
        self.fc2 = nn.Linear(config.ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    # 以下代码与 transformers.models.mbart.modeling_mbart.MBartDecoderLayer.forward 相同
    # 前向传播函数定义，接收隐藏状态张量和各种可选参数
    def forward(
        # 隐藏状态张量，类型为torch.Tensor
        hidden_states: torch.Tensor,
        # 注意力掩码，类型为torch.Tensor，可选参数，默认为None
        attention_mask: Optional[torch.Tensor] = None,
        # 编码器隐藏状态，类型为torch.Tensor，可选参数，默认为None
        encoder_hidden_states: Optional[torch.Tensor] = None,
        # 编码器注意力掩码，类型为torch.Tensor，可选参数，默认为None
        encoder_attention_mask: Optional[torch.Tensor] = None,
        # 层头掩码，类型为torch.Tensor，可选参数，默认为None
        layer_head_mask: Optional[torch.Tensor] = None,
        # 交叉注意力层头掩码，类型为torch.Tensor，可选参数，默认为None
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        # 过去的键值对，类型为元组包含torch.Tensor，可选参数，默认为None
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        # 输出注意力权重，类型为bool，可选参数，默认为False
        output_attentions: Optional[bool] = False,
        # 使用缓存，类型为bool，可选参数，默认为True
        use_cache: Optional[bool] = True,
# 继承自PreTrainedModel类的XGLMPreTrainedModel类
class XGLMPreTrainedModel(PreTrainedModel):
    # 用于设置配置类
    config_class = XGLMConfig
    # 模型前缀
    base_model_prefix = "model"
    # 是否支持梯度检查点
    supports_gradient_checkpointing = True
    # 不需要进行切分模块的列表
    _no_split_modules = ["XGLMDecoderLayer"]

    # 初始化权重
    def _init_weights(self, module):
        std = self.config.init_std
        # 如果是nn.Linear类型，初始化线性层的权重和偏置
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果是nn.Embedding类型，初始化嵌入层的权重
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

# XGLMModel类，是XGLMPreTrainedModel的子类
@add_start_docstrings(
    "The bare XGLM Model transformer outputting raw hidden-states without any specific head on top.",
    XGLM_START_DOCSTRING,
)
class XGLMModel(XGLMPreTrainedModel):
    """
    Transformer decoder consisting of *config.num_layers* layers. Each layer is a [`XGLMDecoderLayer`]

    Args:
        config: XGLMConfig
        embed_tokens (nn.Embedding): output embedding
    """

    # 初始化函数
    def __init__(self, config: XGLMConfig, embed_tokens: Optional[nn.Embedding] = None):
        # 调用父类的初始化函数
        super().__init__(config)
        # 丢弃率
        self.dropout = config.dropout
        # 切分层的丢弃率
        self.layerdrop = config.layerdrop
        # 填充的索引
        self.padding_idx = config.pad_token_id
        # 目标位置的最大长度
        self.max_target_positions = config.max_position_embeddings
        # 嵌入缩放系数
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        # 如果embed_tokens不为None，则使用embed_tokens作为嵌入层
        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        # 否则使用nn.Embedding初始化嵌入层
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        # 初始化位置编码
        self.embed_positions = XGLMSinusoidalPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
            config.pad_token_id,
        )
        # 初始化解码层列表
        self.layers = nn.ModuleList([XGLMDecoderLayer(config) for _ in range(config.num_layers)])
        # 归一化层
        self.layer_norm = nn.LayerNorm(config.d_model)

        # 梯度检查��，默认为False
        self.gradient_checkpointing = False
        # 初始化权重并执行最终处理
        self.post_init()

    # 获取输入嵌入层
    def get_input_embeddings(self):
        return self.embed_tokens

    # 设置输入嵌入层
    def set_input_embeddings(self, value):
        self.embed_tokens = value

    # 对模型前向进行注释
    @add_start_docstrings_to_model_forward(XGLM_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPastAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    # 对模型进行前向传播，接收输入参数并返回输出结果
    def forward(
        # 输入的 token IDs，可选的张量，默认为 None
        input_ids: Optional[torch.Tensor] = None,
        # 注意力掩码，可选的张量，默认为 None
        attention_mask: Optional[torch.Tensor] = None,
        # 位置 IDs，可选的张量，默认为 None
        position_ids: Optional[torch.Tensor] = None,
        # 编码器隐藏状态，可选的张量，默认为 None
        encoder_hidden_states: Optional[torch.Tensor] = None,
        # 编码器注意力掩码，可选的张量，默认为 None
        encoder_attention_mask: Optional[torch.Tensor] = None,
        # 头部掩码，可选的张量，默认为 None
        head_mask: Optional[torch.Tensor] = None,
        # 跨注意力头部掩码，可选的张量，默认为 None
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        # 过去的键值对，可选的浮点数张量列表，默认为 None
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        # 输入嵌入，可选的张量，默认为 None
        inputs_embeds: Optional[torch.Tensor] = None,
        # 是否使用缓存，可选的布尔值，默认为 None
        use_cache: Optional[bool] = None,
        # 是否输出注意力权重，可选的布尔值，默认为 None
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态，可选的布尔值，默认为 None
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典形式的结果，可选的布尔值，默认为 None
        return_dict: Optional[bool] = None,
# 定义XGLMForCausalLM类，它是一个带有语言建模头的XGLM模型转换器，线性层的权重与输入嵌入层相互绑定
class XGLMForCausalLM(XGLMPreTrainedModel):
    # 基础模型前缀
    base_model_prefix = "model"
    # 被绑定的权重关键字
    _tied_weights_keys = ["lm_head.weight"]

    # 初始化函数，接受config参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__(config)
        # 实例化XGLMModel模型
        self.model = XGLMModel(config)
        # 实例化线性层，输出维度为config中的隐藏大小和词汇表大小，不使用偏置
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入层
    def get_input_embeddings(self):
        return self.model.embed_tokens

    # 设置输入嵌入层
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    # 获取输出嵌入层
    def get_output_embeddings(self):
        return self.lm_head

    # 设置输出嵌入层
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 前向传播函数，接受多个参数，包括输入ID、注意力掩码、位置ID等
    @add_start_docstrings_to_model_forward(XGLM_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutputWithCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    def forward(
        self, input_ids: torch.LongTensor, 
        attention_mask: Optional[torch.Tensor] = None, 
        encoder_hidden_states: Optional[torch.FloatTensor] = None, 
        encoder_attention_mask: Optional[torch.Tensor] = None, 
        head_mask: Optional[torch.Tensor] = None, 
        cross_attn_head_mask: Optional[torch.Tensor] = None, 
        past_key_values=None, 
        inputs_embeds: Optional[torch.FloatTensor] = None, 
        use_cache: bool = True, 
        output_attentions: Optional[bool] = None, 
        output_hidden_states: Optional[bool] = None, 
        return_dict: Optional[bool] = None
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional`):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """

        # 设置输出注意力，默认为模型设定的值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 设置输出隐藏状态，默认为模型设定的值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 设置返回字典，默认为模型设定的值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用模型进行训练
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            head_mask=head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 通过lm_head层获取logits
        logits = self.lm_head(outputs[0])

        loss = None
        if labels is not None:
            # 调整标签并在末尾添加一个填充标记
            shift_labels = labels.new_zeros(labels.shape)
            shift_labels[:, :-1] = labels[:, 1:].clone()
            shift_labels[:, -1] = self.config.pad_token_id

            # 计算交叉熵损失
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            # 返回输出
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            cross_attentions=outputs.cross_attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, use_cache=None, **kwargs
        ):
        # 如果过去的键值对不为 None，则获取过去键值对中第一个元素的 shape 第三维度的值
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # 如果输入 ID 的维度大于过去的长度，则将移除前缀的长度设为过去的长度
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 否则，默认旧的行为：仅保留最后一个 ID
                remove_prefix_length = input_ids.shape[1] - 1

            # 将输入 ID 更新为移除前缀长度后的内容
            input_ids = input_ids[:, remove_prefix_length:]

        # 获取附加参数中的 position_ids
        position_ids = kwargs.get("position_ids", None)
        # 如果存在注意力遮罩且没有位置 ID
        if attention_mask is not None and position_ids is None:
            # 为批量生成创建位置 ID
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]
        else:
            # 否则将位置 ID 设为 None
            position_ids = None
            # 如果模型作为编码器-解码器模型中的解码器使用，则在现场创建解码器注意力遮罩
            if attention_mask is None:
                attention_mask = input_ids.new_ones(input_ids.shape)
        
        # 在第一步，decoder_cached_states 为空
        return {
            "input_ids": input_ids,  # encoder_outputs 已定义，不需要 input_ids
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "use_cache": use_cache,
        }

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        # 对过去的键值对重排序
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past
```