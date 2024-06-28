# `.\models\xglm\modeling_xglm.py`

```
# 设置文件编码为UTF-8
# 版权声明和许可协议，指定了代码的使用条款
# 导入必要的库和模块
# 导入了一些特定的类和函数用于模型定义和训练
""" PyTorch XGLM model."""

# 导入数学库
import math
# 导入类型提示工具
from typing import List, Optional, Tuple, Union

# 导入PyTorch库
import torch
# 导入PyTorch中的checkpoint功能
import torch.utils.checkpoint
# 导入PyTorch中的神经网络模块
from torch import nn
# 导入PyTorch中的交叉熵损失函数
from torch.nn import CrossEntropyLoss

# 导入自定义模块和函数
from ...activations import ACT2FN
from ...modeling_attn_mask_utils import _prepare_4d_attention_mask, _prepare_4d_causal_attention_mask
from ...modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions
from ...modeling_utils import PreTrainedModel
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
# 导入XGLM模型的配置类
from .configuration_xglm import XGLMConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 预设的模型检查点和配置信息
_CHECKPOINT_FOR_DOC = "facebook/xglm-564M"
_CONFIG_FOR_DOC = "XGLMConfig"

# 预设的预训练模型列表
XGLM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/xglm-564M",
    # 查看所有XGLM模型：https://huggingface.co/models?filter=xglm
]

# XGLM模型的开始文档字符串，描述了模型的继承和参数
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

# XGLM模型的输入文档字符串，目前为空
XGLM_INPUTS_DOCSTRING = r"""
"""

# 定义一个用于生成任意长度正弦位置嵌入的模块
class XGLMSinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length."""

    def __init__(self, num_positions: int, embedding_dim: int, padding_idx: Optional[int] = None):
        super().__init__()
        # 初始偏移量
        self.offset = 2
        # 嵌入维度
        self.embedding_dim = embedding_dim
        # 填充索引，可选
        self.padding_idx = padding_idx
        # 生成位置权重
        self.make_weights(num_positions + self.offset, embedding_dim, padding_idx)
    # 定义一个方法，用于创建权重矩阵，用于位置编码或其他嵌入操作
    def make_weights(self, num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        # 调用get_embedding方法获取嵌入权重
        emb_weights = self.get_embedding(num_embeddings, embedding_dim, padding_idx)
        # 如果对象已有weights属性，将新创建的权重矩阵类型和设备与该属性相匹配
        if hasattr(self, "weights"):
            emb_weights = emb_weights.to(dtype=self.weights.dtype, device=self.weights.device)

        # 注册权重为缓冲区，不会被视为模型的参数
        self.register_buffer("weights", emb_weights, persistent=False)

    @staticmethod
    def get_embedding(num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None):
        """
        构建正弦嵌入。

        这与tensor2tensor中的实现相匹配，但与"Attention Is All You Need"第3.5节的描述略有不同。
        """
        # 计算正弦周期的半长度
        half_dim = embedding_dim // 2
        # 计算正弦函数的周期
        emb = math.log(10000) / (half_dim - 1)
        # 计算正弦嵌入
        emb = torch.exp(torch.arange(half_dim, dtype=torch.int64).float() * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.int64).float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        # 如果embedding_dim是奇数，进行零填充
        if embedding_dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        # 如果有padding_idx，则将对应位置的嵌入设置为零
        if padding_idx is not None:
            emb[padding_idx, :] = 0

        return emb.to(torch.get_default_dtype())

    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor = None, past_key_values_length: int = 0):
        # 获取位置编码的批大小和序列长度
        bsz, seq_len = position_ids.size()
        # 将位置编码偏移量加到输入的位置编码上
        position_ids += self.offset

        # 扩展嵌入权重，如果需要的话。不使用`position_ids.max()`是为了保持torch.fx的兼容性。
        max_pos = 2 + seq_len + past_key_values_length
        if max_pos > self.weights.size(0):
            self.make_weights(max_pos, self.embedding_dim, self.padding_idx)

        # 根据位置编码选择对应的权重，并调整形状以匹配输入的bsz和seq_len，并返回不可变版本
        return self.weights.index_select(0, position_ids.view(-1)).view(bsz, seq_len, self.weights.shape[-1]).detach()
class XGLMAttention(nn.Module):
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
        
        # 缩放因子，用于缩放 Q、K、V 矩阵的值
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder

        # 线性变换层，用于计算 Q、K、V 的投影
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        # 将输入的 tensor 重塑为多头注意力所需的形状
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        # 此处定义注意力层的前向传播过程
        pass  # 实际的实现应当包括 Q、K、V 的计算、注意力分数的计算以及输出的组装


class XGLMDecoderLayer(nn.Module):
    def __init__(self, config: XGLMConfig):
        super().__init__()
        self.embed_dim = config.d_model

        # 自注意力层，使用 XGLMAttention 类定义
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
            # 如果配置需要跨注意力，则定义一个额外的注意力层
            self.encoder_attn = XGLMAttention(
                embed_dim=self.embed_dim,
                num_heads=config.attention_heads,
                dropout=config.attention_dropout,
                is_decoder=True,
            )
            self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)

        # 自注意力层和全连接层后的 LayerNorm
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.ffn_dim)
        self.fc2 = nn.Linear(config.ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    # 此处缺少 forward 方法的实现，应当在复制的代码中找到并补充其实现部分
    # 定义神经网络模型的前向传播方法，接收以下参数：
    # - hidden_states: 隐藏状态的张量，通常是当前层的输出
    # - attention_mask: 可选参数，用于指定哪些位置需要被屏蔽，以避免注意力机制处理这些位置
    # - encoder_hidden_states: 可选参数，编码器的隐藏状态张量，用于注意力机制中的计算
    # - encoder_attention_mask: 可选参数，编码器的注意力掩码张量，用于编码器-解码器注意力
    # - layer_head_mask: 可选参数，多头注意力机制中每个头部的掩码，以允许或禁止特定头部的计算
    # - cross_attn_layer_head_mask: 可选参数，用于跨层注意力的头部掩码，控制不同层之间的注意力计算
    # - past_key_value: 可选参数，包含过去键值状态的元组，用于在递归解码器中重用先前计算的键值
    # - output_attentions: 可选参数，布尔值，指示是否输出注意力权重
    # - use_cache: 可选参数，布尔值，指示是否使用缓存加速解码器的计算
class XGLMPreTrainedModel(PreTrainedModel):
    # 设置配置类为 XGLMConfig
    config_class = XGLMConfig
    # 模型基本名称前缀为 "model"
    base_model_prefix = "model"
    # 支持梯度检查点
    supports_gradient_checkpointing = True
    # 不分割的模块名称列表，包括 "XGLMDecoderLayer"
    _no_split_modules = ["XGLMDecoderLayer"]

    def _init_weights(self, module):
        # 初始化权重函数
        std = self.config.init_std
        # 如果模块是线性层
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            # 如果有偏置，则初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果模块是嵌入层
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            # 如果设置了填充索引，则对应位置初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


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

    def __init__(self, config: XGLMConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        # 丢弃率
        self.dropout = config.dropout
        # 层丢弃率
        self.layerdrop = config.layerdrop
        # 填充索引
        self.padding_idx = config.pad_token_id
        # 最大目标位置
        self.max_target_positions = config.max_position_embeddings
        # 嵌入尺度
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        # 如果提供了嵌入令牌，则使用提供的；否则初始化一个新的嵌入层
        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        # 初始化位置编码
        self.embed_positions = XGLMSinusoidalPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
            config.pad_token_id,
        )
        # 创建一系列解码层
        self.layers = nn.ModuleList([XGLMDecoderLayer(config) for _ in range(config.num_layers)])
        # 层归一化
        self.layer_norm = nn.LayerNorm(config.d_model)

        # 梯度检查点设为假
        self.gradient_checkpointing = False
        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        # 获取输入嵌入层
        return self.embed_tokens

    def set_input_embeddings(self, value):
        # 设置输入嵌入层
        self.embed_tokens = value

    @add_start_docstrings_to_model_forward(XGLM_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPastAndCrossAttentions,
        config_class=_CONFIG_FOR_DOC,
    )
    # 定义模型的前向传播方法，接受多个输入参数
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,  # 输入的 token IDs，可以是 None 或者 torch.Tensor 类型
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码，用于指示哪些元素是 padding，可以是 None 或者 torch.Tensor 类型
        position_ids: Optional[torch.Tensor] = None,  # 位置编码，用于指示每个 token 的位置信息，可以是 None 或者 torch.Tensor 类型
        encoder_hidden_states: Optional[torch.Tensor] = None,  # 编码器的隐藏状态，可以是 None 或者 torch.Tensor 类型
        encoder_attention_mask: Optional[torch.Tensor] = None,  # 编码器的注意力掩码，可以是 None 或者 torch.Tensor 类型
        head_mask: Optional[torch.Tensor] = None,  # 多头注意力的掩码，可以是 None 或者 torch.Tensor 类型
        cross_attn_head_mask: Optional[torch.Tensor] = None,  # 跨注意力头的掩码，可以是 None 或者 torch.Tensor 类型
        past_key_values: Optional[List[torch.FloatTensor]] = None,  # 缓存的键值对，可以是 None 或者 List[torch.FloatTensor] 类型
        inputs_embeds: Optional[torch.Tensor] = None,  # 输入的嵌入向量，可以是 None 或者 torch.Tensor 类型
        use_cache: Optional[bool] = None,  # 是否使用缓存，可以是 None 或者 bool 类型
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，可以是 None 或者 bool 类型
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可以是 None 或者 bool 类型
        return_dict: Optional[bool] = None,  # 是否返回字典格式的输出，可以是 None 或者 bool 类型
# 使用装饰器添加文档字符串，描述了 XGLM 模型转换器，带有一个在顶部的语言建模头部的线性层（其权重与输入嵌入层相绑定）。
@add_start_docstrings(
    """
    The XGLM Model transformer with a language modeling head on top (linear layer with weights tied to the input
    embeddings).
    """,
    XGLM_START_DOCSTRING,
)
# 声明 XGLMForCausalLM 类，继承自 XGLMPreTrainedModel 类
class XGLMForCausalLM(XGLMPreTrainedModel):
    # 指定模型的前缀字符串
    base_model_prefix = "model"
    # 定义被绑定权重的键名列表
    _tied_weights_keys = ["lm_head.weight"]

    # 初始化方法，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 创建 XGLMModel 类的实例并赋值给 self.model
        self.model = XGLMModel(config)
        # 创建一个线性层用于语言建模头部，输出大小为 config.vocab_size，无偏置
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 调用后续初始化方法
        self.post_init()

    # 获取输入嵌入层的方法
    def get_input_embeddings(self):
        return self.model.embed_tokens

    # 设置输入嵌入层的方法
    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    # 获取输出嵌入层（语言建模头部）的方法
    def get_output_embeddings(self):
        return self.lm_head

    # 设置输出嵌入层（语言建模头部）的方法
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 前向传播方法，接受多个输入参数
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
        # 函数声明的参数列表未完，需要继续
    ) -> Union[Tuple[torch.Tensor], CausalLMOutputWithCrossAttentions]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """

        # 设置是否输出注意力权重
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 设置是否输出隐藏状态
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 设置是否返回字典格式的输出
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        # 将输入传递给模型进行前向传播
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

        # 通过语言模型头部生成逻辑回归结果
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
            # 如果不返回字典格式的输出，则按顺序返回元组
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        # 如果返回字典格式的输出，则构建并返回带有交叉注意力的因果语言模型输出对象
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
            # 如果传入的过去键值不为None，则获取第一个键值对应的形状的第三个元素，即长度
            if past_key_values is not None:
                past_length = past_key_values[0][0].shape[2]

                # 如果输入的input_ids的第二个维度大于过去长度，则设定要移除的前缀长度为过去长度
                if input_ids.shape[1] > past_length:
                    remove_prefix_length = past_length
                else:
                    # 否则，默认保留最后一个ID，设定要移除的前缀长度为input_ids的第二个维度减1
                    remove_prefix_length = input_ids.shape[1] - 1

                # 重新设定input_ids为去除前缀后的部分
                input_ids = input_ids[:, remove_prefix_length:]

            position_ids = kwargs.get("position_ids", None)
            # 如果存在attention_mask且position_ids为None，则动态创建position_ids用于批量生成
            if attention_mask is not None and position_ids is None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                # 如果存在过去键值，则截取最后input_ids.shape[1]列
                if past_key_values:
                    position_ids = position_ids[:, -input_ids.shape[1] :]
            else:
                position_ids = None
                # 如果模型作为编码器-解码器模型中的解码器使用，则动态创建解码器的attention_mask
                if attention_mask is None:
                    attention_mask = input_ids.new_ones(input_ids.shape)

            # 第一步，decoder_cached_states为空
            return {
                "input_ids": input_ids,  # encoder_outputs is defined. input_ids not needed
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
            }

        @staticmethod
        def _reorder_cache(past_key_values, beam_idx):
            reordered_past = ()
            # 对过去的键值进行重新排序，根据beam_idx
            for layer_past in past_key_values:
                reordered_past += (
                    # 将每一层的过去状态按照beam_idx重新排序，并且将结果组合成一个元组
                    tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
                )
            return reordered_past
```