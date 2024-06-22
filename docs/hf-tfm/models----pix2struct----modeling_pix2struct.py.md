# `.\transformers\models\pix2struct\modeling_pix2struct.py`

```py
# 设置文件编码为 UTF-8
# 版权声明
# 根据 Apache 许可证 2.0 版本，除非符合许可证，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"AS IS"的基础，
# 没有任何形式的担保或条件，无论是明示的还是暗示的
# 请查看许可证以获取特定语言的权限和限制
""" Pix2Struct 建模文件"""

# 导入所需的库
import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

# 导入自定义的激活函数映射
from ...activations import ACT2FN
# 导入模型输出相关的类
from ...modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
)
# 导入模型工具函数
from ...modeling_utils import PreTrainedModel
# 导入 PyTorch 工具函数
from ...pytorch_utils import ALL_LAYERNORM_LAYERS
# 导入工具函数
from ...utils import (
    DUMMY_INPUTS,
    DUMMY_MASK,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    is_torch_fx_proxy,
    logging,
    replace_return_docstrings,
)
# 导入 Pix2Struct 配置文件
from .configuration_pix2struct import Pix2StructConfig, Pix2StructTextConfig, Pix2StructVisionConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的配置信息
_CONFIG_FOR_DOC = "Pix2StructConfig"

# 预训练模型存档列表
PIX2STRUCT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google/pix2struct-textcaps-base",
    "google/pix2struct-textcaps-large",
    "google/pix2struct-base",
    "google/pix2struct-large",
    "google/pix2struct-ai2d-base",
    "google/pix2struct-ai2d-large",
    "google/pix2struct-widget-captioning-base",
    "google/pix2struct-widget-captioning-large",
    "google/pix2struct-screen2words-base",
    "google/pix2struct-screen2words-large",
    "google/pix2struct-docvqa-base",
    "google/pix2struct-docvqa-large",
    "google/pix2struct-ocrvqa-base",
    "google/pix2struct-ocrvqa-large",
    "google/pix2struct-chartqa-base",
    "google/pix2struct-inforgraphics-vqa-base",
    "google/pix2struct-inforgraphics-vqa-large",
    # 查看所有 Pix2StructVision 模型，请访问 https://huggingface.co/models?filter=pix2struct
]

# 从 transformers.models.t5.modeling_t5.T5LayerNorm 适配为 Pix2StructLayerNorm
class Pix2StructLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        构建一个 Pix2Struct 风格的 Layernorm 模块。无偏置和无均值减法。
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
    def forward(self, hidden_states):
        # T5使用一个只进行缩放而不进行偏移的层归一化，也被称为均方根层归一化 https://arxiv.org/abs/1910.07467，因此方差是在没有均值的情况下计算的，没有偏差。此外，我们希望确保对半精度输入的累积是在fp32中完成的

        # 计算方差，将隐藏状态转换为torch.float32类型，平方，计算均值，保持维度
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        # 对隐藏状态进行归一化
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)

        # 如果权重的数据类型是半精度，则将隐藏状态转换为相同的数据类型
        if self.weight.dtype in [torch.float16, torch.bfloat16]:
            hidden_states = hidden_states.to(self.weight.dtype)

        # 返回加权后的隐藏状态
        return self.weight * hidden_states
try:
    # 尝试导入 apex.normalization 中的 FusedRMSNorm
    from apex.normalization import FusedRMSNorm

    # 将 FusedRMSNorm 赋值给 Pix2StructLayerNorm
    Pix2StructLayerNorm = FusedRMSNorm  # noqa

    # 记录日志信息
    logger.info("Discovered apex.normalization.FusedRMSNorm - will use it instead of Pix2StructLayerNorm")
except ImportError:
    # 如果导入失败，则使用普通的 Pix2StructLayerNorm
    pass
except Exception:
    # 如果导入出现其他异常，则记录警告信息
    logger.warning("Discovered apex but it failed to load, falling back to Pix2StructLayerNorm")
    pass

# 将 Pix2StructLayerNorm 添加到 ALL_LAYERNORM_LAYERS 列表中
ALL_LAYERNORM_LAYERS.append(Pix2StructLayerNorm)


class Pix2StructVisionEmbeddings(nn.Module):
    r"""
    Construct the embeddings from patch. In `Pix2Struct` the input is different from classic Vision-transformer models.
    Here the input is a sequence of `seq_len` flattened patches that also combines padding patches (tokens). Each patch
    is represented by a vector of `hidden_size` values.
    """

    def __init__(self, config: Pix2StructConfig) -> None:
        super().__init__()
        # 线性变换，将 patch_embed_hidden_size 维度的输入映射到 hidden_size 维度
        self.patch_projection = nn.Linear(config.patch_embed_hidden_size, config.hidden_size)

        # 行索引的嵌入层
        self.row_embedder = nn.Embedding(config.seq_len, config.hidden_size)
        # 列索引的嵌入层
        self.column_embedder = nn.Embedding(config.seq_len, config.hidden_size)

        # Dropout 层
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, flattened_patches: torch.Tensor) -> torch.Tensor:
        # 将行和列索引存储在 flattened_patches 的第一和第二位置
        # flattened_patches: `batch_size`, `seq_len`, `hidden_size` + 2
        row_indices = flattened_patches[:, :, 0].long()
        col_indices = flattened_patches[:, :, 1].long()

        flattened_patches = flattened_patches[:, :, 2:]

        # 对 patches 进行线性变换
        embeddings = self.patch_projection(flattened_patches)
        row_embeddings = self.row_embedder(row_indices)
        col_embeddings = self.column_embedder(col_indices)

        # 将所有嵌入相加
        embeddings = embeddings + row_embeddings + col_embeddings

        # 对嵌入应用 Dropout
        embeddings = self.dropout(embeddings)

        return embeddings


class Pix2StructVisionAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.key_value_proj_dim = config.d_kv
        self.n_heads = config.num_attention_heads
        self.dropout = config.attention_dropout
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # 使用 Mesh TensorFlow 初始化，避免在 softmax 前进行缩放
        self.query = nn.Linear(self.hidden_size, self.inner_dim, bias=False)
        self.key = nn.Linear(self.hidden_size, self.inner_dim, bias=False)
        self.value = nn.Linear(self.hidden_size, self.inner_dim, bias=False)
        self.output = nn.Linear(self.inner_dim, self.hidden_size, bias=False)

        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        output_attentions=False,
    def forward(
        ):
            """
            Self-attention block
            """
            # Input is (batch_size, seq_length, dim)
            # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
            # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
            batch_size, seq_length = hidden_states.shape[:2]
    
            def to_projection_shape(states):
                """projection"""
                return states.contiguous().view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
    
            # get query states
            # (batch_size, n_heads, seq_length, dim_per_head)
            query_states = to_projection_shape(self.query(hidden_states))
    
            # get key/value states
            key_states = to_projection_shape(self.key(hidden_states))
            value_states = to_projection_shape(self.value(hidden_states))
    
            # compute scores
            # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9
            scores = torch.matmul(query_states, key_states.transpose(3, 2))
    
            if position_bias is None:
                position_bias = torch.zeros(
                    (1, self.n_heads, seq_length, seq_length), device=scores.device, dtype=scores.dtype
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
    
                if attention_mask is None:
                    attention_mask = torch.ones((batch_size, seq_length), device=scores.device, dtype=scores.dtype)
    
                if attention_mask.dim() == 2:
                    position_bias = position_bias + attention_mask[:, None, None, :].to(position_bias.device)
                else:
                    # (batch_size, n_heads, seq_length, key_length)
                    position_bias = position_bias + attention_mask.to(position_bias.device)
                position_bias = 1 - position_bias
    
            position_bias_masked = position_bias.masked_fill(position_bias == 1, torch.finfo(scores.dtype).min)
            scores += position_bias_masked
            scores = torch.max(scores, torch.tensor(torch.finfo(scores.dtype).min))
    
            # (batch_size, n_heads, seq_length, key_length)
            attn_weights = nn.functional.softmax(scores, dim=-1, dtype=torch.float32).type_as(scores)
    
            # (batch_size, n_heads, seq_length, key_length)
            attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    
            # Mask heads if we want to
            if layer_head_mask is not None:
                attn_weights = attn_weights * layer_head_mask
    
            attn_output = torch.matmul(attn_weights, value_states)
    
            # (batch_size, seq_length, dim)
            attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)
    
            attn_output = self.output(attn_output)
    
            outputs = (attn_output,) + (position_bias,)
    
            if output_attentions:
                outputs = outputs + (attn_weights,)
            return outputs
# 定义Pix2StructVisionMlp类，继承自nn.Module
class Pix2StructVisionMlp(nn.Module):
    # 初始化函数，接受Pix2StructVisionConfig类型的config参数
    def __init__(self, config: Pix2StructVisionConfig):
        # 调用父类的初始化函数
        super().__init__()
        # 定义线性层wi_0，输入维度为config.hidden_size，输出维度为config.d_ff，无偏置
        self.wi_0 = nn.Linear(config.hidden_size, config.d_ff, bias=False)
        # 定义线性层wi_1，输入维度为config.hidden_size，输出维度为config.d_ff，无偏置
        self.wi_1 = nn.Linear(config.hidden_size, config.d_ff, bias=False)
        # 定义线性层wo，输入维度为config.d_ff，输出维度为config.hidden_size，无偏置
        self.wo = nn.Linear(config.d_ff, config.hidden_size, bias=False)
        # 定义Dropout层，概率为config.dropout_rate
        self.dropout = nn.Dropout(config.dropout_rate)
        # 根据配置中的dense_act_fn选择激活函数
        self.act = ACT2FN[config.dense_act_fn]

    # 前向传播函数，接受hidden_states作为输入
    def forward(self, hidden_states):
        # 使用激活函数act对wi_0和hidden_states的乘积进行激活
        hidden_gelu = self.act(self.wi_0(hidden_states))
        # 计算wi_1和hidden_states的乘积
        hidden_linear = self.wi_1(hidden_states)
        # 将激活后的结果与线性结果相乘
        hidden_states = hidden_gelu * hidden_linear
        # 对结果进行Dropout
        hidden_states = self.dropout(hidden_states)

        # 为了使8位量化适用于google/flan-t5-xxl，保持wo为float32
        # 参考https://github.com/huggingface/transformers/issues/20287
        # 确保权重不是int8类型，以防用户强制将`_keep_in_fp32_modules`设置为`None`
        if (
            isinstance(self.wo.weight, torch.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != torch.int8
        ):
            # 将hidden_states转换为wo的数据类型
            hidden_states = hidden_states.to(self.wo.weight.dtype)

        # 使用wo对hidden_states进行线性变换
        hidden_states = self.wo(hidden_states)
        # 返回处理后的hidden_states
        return hidden_states


# 定义Pix2StructVisionLayer类，继承自nn.Module
class Pix2StructVisionLayer(nn.Module):
    # 初始化函数，接受Pix2StructConfig类型的config参数
    def __init__(self, config: Pix2StructConfig) -> None:
        # 调用父类的初始化函数
        super().__init__()
        # 设置chunk_size_feed_forward为config中的值
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 设置seq_len_dim为1
        self.seq_len_dim = 1
        # 初始化Pix2StructVisionAttention和Pix2StructVisionMlp
        self.attention = Pix2StructVisionAttention(config)
        self.mlp = Pix2StructVisionMlp(config)
        # 初始化前置LayerNorm层
        self.pre_mlp_layer_norm = Pix2StructLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pre_attention_layer_norm = Pix2StructLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    # 前向传播函数，接受hidden_states、attention_mask、head_mask和output_attentions作为输入
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    # 定义函数的输入和输出类型注解，可以返回一个包含两个张量或一个张量的元组
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        # 保存残差连接
        residual = hidden_states

        # 在 Pix2StructVision 中，先应用 layernorm，再进行自注意力机制
        hidden_states = self.pre_attention_layer_norm(hidden_states)

        # 使用 self.attention 进行自注意力计算
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=head_mask,
            output_attentions=output_attentions,
        )
        # 获取自注意力输出
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # 如果输出注意力权重，则添加自注意力

        # 第一个残差连接
        hidden_states = attention_output + residual

        # 在 Pix2StructVision 中，自注意力后也应用 layernorm
        layer_output = self.pre_mlp_layer_norm(hidden_states)
        # 使用 MLP 进行计算，并添加第二个残差连接
        layer_output = self.mlp(layer_output) + hidden_states

        # 将输出添加到 outputs 中
        outputs = (layer_output,) + outputs

        # 返回结果
        return outputs
class Pix2StructVisionEncoder(nn.Module):
    # 定义 Pix2StructVisionEncoder 类，继承自 nn.Module
    def __init__(self, config: Pix2StructConfig) -> None:
        # 初始化函数，接受一个 Pix2StructConfig 类型的参数 config
        super().__init__()
        # 调用父类的初始化函数
        self.config = config
        # 将参数 config 赋值给实例变量 self.config
        self.layer = nn.ModuleList([Pix2StructVisionLayer(config) for _ in range(config.num_hidden_layers)])
        # 创建 nn.ModuleList，其中包含 config.num_hidden_layers 个 Pix2StructVisionLayer 实例
        self.gradient_checkpointing = False
        # 初始化梯度检查点标志为 False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        # 定义前向传播函数，接受多个参数并返回 Union[tuple, BaseModelOutput] 类型的结果
        all_hidden_states = () if output_hidden_states else None
        # 如果 output_hidden_states 为 True，则初始化 all_hidden_states 为空元组，否则为 None
        all_self_attentions = () if output_attentions else None
        # 如果 output_attentions 为 True，则初始化 all_self_attentions 为空元组，否则为 None

        for i, layer_module in enumerate(self.layer):
            # 遍历 self.layer 中的每个层
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
                # 如果 output_hidden_states 为 True，则将 hidden_states 添加到 all_hidden_states 中

            layer_head_mask = head_mask[i] if head_mask is not None else None
            # 获取当前层的头部掩码

            if self.gradient_checkpointing and self.training:
                # 如果开启了梯度检查点且处于训练模式
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    output_attentions,
                )
                # 调用梯度检查点函数计算当前层的输出
            else:
                layer_outputs = layer_module(hidden_states, attention_mask, layer_head_mask, output_attentions)
                # 否则直接计算当前层的输出

            hidden_states = layer_outputs[0]
            # 更新 hidden_states 为当前层的输出的第一个元素

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                # 如果 output_attentions 为 True，则将当前层的注意力输出添加到 all_self_attentions 中

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
            # 如果 output_hidden_states 为 True，则将最后一个隐藏状态添加到 all_hidden_states 中

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
            # 如果不返回字典，则返回非空的 hidden_states, all_hidden_states, all_self_attentions
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
        # 返回 BaseModelOutput 对象，包含最后一个隐藏状态、所有隐藏状态和所有注意力输出

class Pix2StructPreTrainedModel(PreTrainedModel):
    # 定��� Pix2StructPreTrainedModel 类，继承自 PreTrainedModel
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    # 一个抽象类，用于处理权重初始化以及预训练模型的下载和加载的简单接口

    config_class = Pix2StructConfig
    # 设置 config_class 为 Pix2StructConfig 类

    @property
    def dummy_inputs(self):
        # 定义 dummy_inputs 属性
        input_ids = torch.tensor(DUMMY_INPUTS)
        # 创建输入张量 input_ids
        input_mask = torch.tensor(DUMMY_MASK)
        # 创建输入掩码张量 input_mask
        dummy_inputs = {
            "decoder_input_ids": input_ids,
            "input_ids": input_ids,
            "decoder_attention_mask": input_mask,
        }
        # 创建 dummy_inputs 字典
        return dummy_inputs
        # 返回 dummy_inputs

    # Copied from transformers.models.t5.modeling_t5.T5PreTrainedModel._shift_right with T5->Pix2Struct
    # 从 transformers.models.t5.modeling_t5.T5PreTrainedModel._shift_right 复制，将 T5 替换为 Pix2Struct
    # 将输入的 token ids 向右移动一位
    def _shift_right(self, input_ids):
        # 获取解码器起始 token id
        decoder_start_token_id = self.config.decoder_start_token_id
        # 获取填充 token id
        pad_token_id = self.config.pad_token_id

        # 如果解码器起始 token id未定义，则抛出数值错误
        if decoder_start_token_id is None:
            raise ValueError(
                "self.model.config.decoder_start_token_id has to be defined. In Pix2Struct it is usually set to the pad_token_id. "
                "See Pix2Struct docs for more information."
            )

        # 将输入向右移动一位
        if is_torch_fx_proxy(input_ids):
            # 对于代理对象，不支持原生的项目赋值
            shifted_input_ids = torch.full(input_ids.shape[:-1] + (1,), decoder_start_token_id)
            shifted_input_ids = torch.cat([shifted_input_ids, input_ids[..., :-1]], dim=-1)
        else:
            shifted_input_ids = input_ids.new_zeros(input_ids.shape)
            shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
            shifted_input_ids[..., 0] = decoder_start_token_id

        # 如果填充 token id未定义，则抛出数值错误
        if pad_token_id is None:
            raise ValueError("self.model.config.pad_token_id has to be defined.")
        # 将标签中可能存在的 -100 值替换为填充 token id
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        # 返回向右移动后的 token ids
        return shifted_input_ids
# 定义 Pix2StructVision 模型的文档字符串，包含模型的用途和参数说明
PIX2STRUCT_VISION_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`Pix2StructConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 定义 Pix2StructVision 模型的输入文档字符串，包含输入参数的说明
PIX2STRUCT_VISION_INPUTS_DOCSTRING = r"""
    Args:
        flattened_patches (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_channels x patch_height x patch_width)`):
            Flattened and padded pixel values. These values can be obtained using [`AutoImageProcessor`]. See
            [`Pix2StructVisionImageProcessor.__call__`] for details. Check the [original
            paper](https://arxiv.org/abs/2210.03347) (figure 5) for more details.

        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding pixel values. Mask values selected in `[0, 1]`:

        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

# 定义 Pix2StructVisionModel 类，继承自 Pix2StructPreTrainedModel
@add_start_docstrings(
    "The bare Pix2StructVision Model transformer outputting raw hidden-states without any specific head on top.",
    PIX2STRUCT_VISION_START_DOCSTRING,
)
class Pix2StructVisionModel(Pix2StructPreTrainedModel):
    # 设置模型的配置类和主要输入名称
    config_class = Pix2StructVisionConfig
    main_input_name = "flattened_patches"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Pix2StructVisionLayer"]

    # 初始化方法，接受一个 Pix2StructConfig 类型的参数
    def __init__(self, config: Pix2StructConfig):
        super().__init__(config)
        self.config = config

        # 初始化模型的嵌入层和编码器
        self.embeddings = Pix2StructVisionEmbeddings(config)
        self.encoder = Pix2StructVisionEncoder(config)

        # 初始化模型的 LayerNorm 层
        self.layernorm = Pix2StructLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 初始化权重并应用最终处理
        self.post_init()
    # 返回输入嵌入层的投影
    def get_input_embeddings(self):
        return self.embeddings.patch_projection

    # 剪枝模型的注意力头
    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        # 遍历需要剪枝的层和对应的注意力头
        for layer, heads in heads_to_prune.items():
            # 在编码器的指定层中剪枝指定的注意力头
            self.encoder.layer[layer].attention.prune_heads(heads)

    # 重写 forward 方法，添加文档字符串和返回值类型注释
    @add_start_docstrings_to_model_forward(PIX2STRUCT_VISION_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        flattened_patches: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        Returns:

        Example:

        ```python
        >>> import requests
        >>> from PIL import Image
        >>> from transformers import AutoProcessor, Pix2StructVisionModel

        >>> image_processor = AutoProcessor.from_pretrained("google/pix2struct-textcaps-base")
        >>> model = Pix2StructVisionModel.from_pretrained("google/pix2struct-textcaps-base")

        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = image_processor(images=image, return_tensors="pt")
        >>> with torch.no_grad():
        ...     outputs = model(**inputs)

        >>> last_hidden_states = outputs.last_hidden_state
        >>> list(last_hidden_states.shape)
        [1, 2048, 768]
        ```py
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if flattened_patches is None:
            raise ValueError("You have to specify flattened_patches")

        if attention_mask is None:
            # 检查`flattened_patches`中哪些位置不为0
            attention_mask = (flattened_patches.sum(dim=-1) != 0).float()

        # 如果需要，准备头部遮罩
        # head_mask中的1.0表示保留该头部
        # attention_probs的形状为bsz x n_heads x N x N
        # 输入的head_mask的形状为[num_heads]或[num_hidden_layers x num_heads]
        # 并且head_mask被转换为形状[num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(flattened_patches)

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)

        if not return_dict:
            head_outputs = (sequence_output,)
            return head_outputs + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
# 从transformers.models.t5.modeling_t5.T5DenseGatedActDense复制而来，修改为Pix2StructText，d_model修改为hidden_size
class Pix2StructTextDenseGatedActDense(nn.Module):
    def __init__(self, config: Pix2StructTextConfig):
        super().__init__()
        # 初始化线性层wi_0，输入维度为hidden_size，输出维度为d_ff，无偏置
        self.wi_0 = nn.Linear(config.hidden_size, config.d_ff, bias=False)
        # 初始化线性层wi_1，输入维度为hidden_size，输出维度为d_ff，无偏置
        self.wi_1 = nn.Linear(config.hidden_size, config.d_ff, bias=False)
        # 初始化线性层wo，输入维度为d_ff，输出维度为hidden_size，无偏置
        self.wo = nn.Linear(config.d_ff, config.hidden_size, bias=False)
        # 初始化Dropout层，使用配置中的dropout_rate
        self.dropout = nn.Dropout(config.dropout_rate)
        # 获取激活函数，根据配置中的dense_act_fn选择对应的激活函数
        self.act = ACT2FN[config.dense_act_fn]

    def forward(self, hidden_states):
        # 计算wi_0的输出并经过激活函数
        hidden_gelu = self.act(self.wi_0(hidden_states))
        # 计算wi_1的输出
        hidden_linear = self.wi_1(hidden_states)
        # 将两个输出相乘
        hidden_states = hidden_gelu * hidden_linear
        # 对结果进行Dropout
        hidden_states = self.dropout(hidden_states)

        # 为了使8位量化适用于google/flan-t5-xxl，保持self.wo为float32
        # 参考https://github.com/huggingface/transformers/issues/20287
        # 确保权重不是int8，以防用户强制`_keep_in_fp32_modules`为`None`
        if (
            isinstance(self.wo.weight, torch.Tensor)
            and hidden_states.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != torch.int8
        ):
            hidden_states = hidden_states.to(self.wo.weight.dtype)

        # 计算wo的输出
        hidden_states = self.wo(hidden_states)
        return hidden_states


class Pix2StructTextLayerFF(nn.Module):
    def __init__(self, config: Pix2StructTextConfig):
        super().__init__()
        # 初始化Pix2StructTextDenseGatedActDense层
        self.DenseReluDense = Pix2StructTextDenseGatedActDense(config)

        # 初始化LayerNorm层，输入维度为hidden_size，epsilon为配置中的layer_norm_epsilon
        self.layer_norm = Pix2StructLayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        # 初始化Dropout层，使用配置中的dropout_rate
        self.dropout = nn.Dropout(config.dropout_rate)

    # 从transformers.models.t5.modeling_t5.T5LayerFF.forward复制而来
    def forward(self, hidden_states):
        # 对输入进行LayerNorm
        forwarded_states = self.layer_norm(hidden_states)
        # 通过DenseReluDense层处理输入
        forwarded_states = self.DenseReluDense(forwarded_states)
        # 将处理后的结果与原始输入相加，并进行Dropout
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


class Pix2StructTextAttention(nn.Module):
    # 初始化函数，接受配置和是否有相对注意力偏差参数
    def __init__(self, config: Pix2StructTextConfig, has_relative_attention_bias=False):
        # 调用父类初始化函数
        super().__init__()
        # 设置是否有相对注意力偏差
        self.has_relative_attention_bias = has_relative_attention_bias
        # 设置相对注意力偏差的桶数量
        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        # 设置相对注意力偏差的最大距离
        self.relative_attention_max_distance = config.relative_attention_max_distance
        # 设置隐藏层大小
        self.hidden_size = config.hidden_size
        # 设置键值投影维度
        self.key_value_proj_dim = config.d_kv
        # 设置头部数量
        self.n_heads = config.num_heads
        # 设置丢弃概率
        self.dropout = config.dropout_rate
        # 设置内部维度
        self.inner_dim = self.n_heads * self.key_value_proj_dim

        # 使用 Mesh TensorFlow 初始化，以避免在 softmax 前进行缩放
        self.query = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.key = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.value = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.output = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # 如果存在相对注意力偏差，初始化相对注意力偏差
        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        # 初始化剪枝的头部集合
        self.pruned_heads = set()
        # 初始化梯度是否检查点
        self.gradient_checkpointing = False

    @staticmethod
    # 从 transformers.models.t5.modeling_t5.T5Attention._relative_position_bucket 复制而来
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor - 相对位置，定义为 memory_position - query_position，即从参与关注的位置到被关注位置的距离，一个 int32 张量
            bidirectional: a boolean - whether the attention is bidirectional - 一个布尔值，表示注意力是否是双向的
            num_buckets: an integer - 桶的数量，一个整数
            max_distance: an integer - 最大距离，一个整数

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets) - 一个张量，形状与 relative_position 相同，包含范围在 [0, num_buckets) 内的 int32 值
        """
        # relative_buckets 初始化为 0
        relative_buckets = 0
        # 如果是双向注意力，则将桶的数量减半，并且根据 relative_position 的正负给 relative_buckets 赋值
        if bidirectional:
            num_buckets //= 2
            relative_buckets += (relative_position > 0).to(torch.long) * num_buckets
            relative_position = torch.abs(relative_position)
        else:
            # 如果是单向注意力，则将 relative_position 变为负数
            relative_position = -torch.min(relative_position, torch.zeros_like(relative_position))
        # 现在 relative_position 范围为 [0, inf)

        # half of the buckets are for exact increments in positions
        # half of the buckets 用于准确增加位置
        max_exact = num_buckets // 2
        # 判断 relative_position 是否在 max_exact 范围内
        is_small = relative_position < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        # 另一半桶用于对数级别增大的位置，最多到 max_distance
        relative_position_if_large = max_exact + (
            torch.log(relative_position.float() / max_exact)
            / math.log(max_distance / max_exact)
            * (num_buckets - max_exact)
        ).to(torch.long)
        # 限制 relative_position_if_large 不超过 num_buckets - 1
        relative_position_if_large = torch.min(
            relative_position_if_large, torch.full_like(relative_position_if_large, num_buckets - 1)
        )

        # 根据 is_small 来选择是使用 relative_position 还是 relative_position_if_large，并更新 relative_buckets
        relative_buckets += torch.where(is_small, relative_position, relative_position_if_large)
        return relative_buckets

    # Adapted from transformers.models.t5.modeling_t5.T5Attention.compute_bias
    # 计算相对位置偏置的函数
    def compute_bias(self, query_length, key_length, device=None):
        """Compute binned relative position bias"""
        # 如果设备为空，则使用相对注意力偏置的设备
        if device is None:
            device = self.relative_attention_bias.weight.device
        # 创建表示上下文位置的张量，范围从0到query_length-1，形状为(query_length, 1)
        context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        # 创建表示记忆位置的张量，范围从0到key_length-1，形状为(1, key_length)
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        # 计算相对位置，即记忆位置减去上下文位置，形状为(query_length, key_length)
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        # 将相对位置划分为不同的区间
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=False,
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        # 获取相对位置偏置值，形状为(query_length, key_length, num_heads)
        values = self.relative_attention_bias(relative_position_bucket)
        # 转置以满足Transformer的输入格式，形状为(1, num_heads, query_length, key_length)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        # 返回相对位置偏置张量
        return values

    # 前向传播函数
    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_value=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
# 从transformers.models.t5.modeling_t5.T5LayerSelfAttention复制代码，将T5LayerNorm->Pix2StructLayerNorm，T5Attention->Pix2StructTextAttention，self.SelfAttention->self.attention，config.d_model->config.hidden_size
class Pix2StructTextLayerSelfAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        # 使用Pix2StructTextAttention替换T5Attention，创建自注意力机制层
        self.attention = Pix2StructTextAttention(config, has_relative_attention_bias=has_relative_attention_bias)
        # 使用Pix2StructLayerNorm替换T5LayerNorm，创建层归一化
        self.layer_norm = Pix2StructLayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        # 创建Dropout层
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
    ):
        # 对输入的hidden_states进行层归一化
        normed_hidden_states = self.layer_norm(hidden_states)
        # 将层归一化后的hidden_states输入到自注意力机制中进行处理
        attention_output = self.attention(
            normed_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        # 将原始hidden_states与经过Dropout的attention_output相加得到最终的输出
        hidden_states = hidden_states + self.dropout(attention_output[0])
        # 构建输出元组，如果需要输出注意力权重，则将其添加到输出中
        outputs = (hidden_states,) + attention_output[1:]  # add attentions if we output them
        return outputs


# 从transformers.models.t5.modeling_t5.T5LayerCrossAttention复制代码，将T5LayerNorm->Pix2StructLayerNorm，T5Attention->Pix2StructTextAttention，self.EncDecAttention->self.attention，config.d_model->config.hidden_size
class Pix2StructTextLayerCrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 使用Pix2StructTextAttention替换T5Attention，创建交叉注意力机制层
        self.attention = Pix2StructTextAttention(config, has_relative_attention_bias=False)
        # 使用Pix2StructLayerNorm替换T5LayerNorm，创建层归一化
        self.layer_norm = Pix2StructLayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        # 创建Dropout层
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        key_value_states,
        attention_mask=None,
        position_bias=None,
        layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        query_length=None,
        output_attentions=False,
    ):
        # 对输入的hidden_states进行层归一化
        normed_hidden_states = self.layer_norm(hidden_states)
        # 将层归一化后的hidden_states输入到交叉注意力机制中进行处理
        attention_output = self.attention(
            normed_hidden_states,
            mask=attention_mask,
            key_value_states=key_value_states,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions,
        )
        # 将原始hidden_states与经过Dropout的attention_output相加得到最终的输出
        layer_output = hidden_states + self.dropout(attention_output[0])
        # 构建输出元组，如果需要输出注意力权重，则将其添加到输出中
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs
# 定义 Pix2StructTextBlock 类，作为 Pix2Struct 模型的一个模块
class Pix2StructTextBlock(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()

        # 创建自注意力层对象，用于编码器和解码器内部的自注意力计算
        self.self_attention = Pix2StructTextLayerSelfAttention(
            config, has_relative_attention_bias=has_relative_attention_bias
        )

        # 创建编码器-解码器注意力层对象，用于编码器和解码器之间的交叉注意力计算
        self.encoder_decoder_attention = Pix2StructTextLayerCrossAttention(config)

        # 创建 MLP 层对象，用于网络的前馈神经网络计算
        self.mlp = Pix2StructTextLayerFF(config)

    # 前向传播函数，定义了模型的前向计算逻辑
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
        # 参数和返回值的文档字符串，在模型的 API 文档中使用，描述了模型输入输出的详细信息
PIX2STRUCT_START_DOCSTRING = r"""

    The Pix2Struct model was proposed in [Pix2Struct: Screenshot Parsing as Pretraining for Visual Language
    Understanding](https://arxiv.org/abs/2210.03347) by Kenton Lee, Mandar Joshi, Iulia Turc, Hexiang Hu, Fangyu Liu,
    Julian Eisenschlos, Urvashi Khandelwal, Peter Shaw, Ming-Wei Chang, Kristina Toutanova. It's an encoder decoder
    transformer pre-trained in a image-to-text setting.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config (Union[`Pix2StructConfig`, `Pix2StructTextConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# 用于描述模型接收文本输入的文档字符串
PIX2STRUCT_TEXT_INPUTS_DOCSTRING = r"""
"""

# 用于描述模型接收输入的文档字符串
PIX2STRUCT_INPUTS_DOCSTRING = r"""
"""

# 添加起始文档字符串的装饰器，用于为模型类添加额外的文档描述
@add_start_docstrings(
    "The standalone text decoder of Pix2Struct",
    PIX2STRUCT_START_DOCSTRING,
)
# 定义 Pix2StructTextModel 类，作为 Pix2Struct 模型的文本解码器
class Pix2StructTextModel(Pix2StructPreTrainedModel):
    # 模型配置类，指定模型使用的配置类
    config_class = Pix2StructTextConfig
    # 指定不需要分割的模块列表
    _no_split_modules = ["Pix2StructTextBlock"]
    # 指定需要共享权重的键列表
    _tied_weights_keys = ["lm_head.weight"]
    # 指定模型是否支持梯度检查点
    supports_gradient_checkpointing = True
    # 初始化方法，接收配置参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 创建一个词嵌入层，词汇表大小为config.vocab_size，隐藏层大小为config.hidden_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)

        # 创建一个包含多个Pix2StructTextBlock的层
        # 每个Pix2StructTextBlock代表模型的一个编码层，根据是否是第一层来设置是否有相对注意力偏置
        self.layer = nn.ModuleList(
            [Pix2StructTextBlock(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        # 创建一个Pix2StructLayerNorm层，用于最后的层归一化
        self.final_layer_norm = Pix2StructLayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        # 创建一个dropout层，用于随机丢弃输入单元
        self.dropout = nn.Dropout(config.dropout_rate)

        # 创建一个线性层，用于将隐藏状态映射到词汇表大小的向量
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 初始化权重并应用最终处理
        self.post_init()
        # 梯度检查点初始化为False
        self.gradient_checkpointing = False

    # 从缓存中重新排序键值对，用于beam搜索
    # 从transformers库中的T5PreTrainedModel类的_reorder_cache方法复制过来的实现
    def _reorder_cache(self, past_key_values, beam_idx):
        # 如果decoder过去的状态未包含在输出中，速度较慢的解码被禁用，不需要重新排序
        if past_key_values is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past_key_values

        # 重新排序decoder过去的状态
        reordered_decoder_past = ()
        for layer_past_states in past_key_values:
            # 根据beam索引重新排列每一层的过去状态
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # 需要为每个四个键/值状态设置正确的“过去”状态
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx.to(layer_past_state.device)),
                )

            # 检查重新排列后的过去状态是否与原始形状匹配
            if reordered_layer_past_states[0].shape != layer_past_states[0].shape:
                raise ValueError(
                    f"reordered_layer_past_states[0] shape {reordered_layer_past_states[0].shape} and layer_past_states[0] shape {layer_past_states[0].shape} mismatched"
                )
            # 检查重新排列后的过去状态长度是否与原始长度匹配
            if len(reordered_layer_past_states) != len(layer_past_states):
                raise ValueError(
                    f"length of reordered_layer_past_states {len(reordered_layer_past_states)} and length of layer_past_states {len(layer_past_states)} mismatched"
                )

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past

    # 获取输入词嵌入
    def get_input_embeddings(self):
        return self.embed_tokens

    # 设置输入词嵌入
    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    # 获取输出词嵌入
    def get_output_embeddings(self):
        return self.lm_head

    # 设置输出词嵌入
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    # 添加模型正向传播的文档字符串
    @add_start_docstrings_to_model_forward(PIX2STRUCT_TEXT_INPUTS_DOCSTRING)
    # 替换返回值的文档字符串
    @replace_return_docstrings(output_type=CausalLMOutputWithCrossAttentions, config_class=_CONFIG_FOR_DOC)
```  
    # 此方法用于模型的前向传播
    def forward(
        # 输入的 token IDs，类型为可选的长整型张量
        input_ids: Optional[torch.LongTensor] = None,
        # 注意力掩码，类型为可选的浮点张量
        attention_mask: Optional[torch.FloatTensor] = None,
        # 编码器隐藏状态，类型为可选的浮点张量
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        # 编码器注意力掩码，类型为可选的浮点张量
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        # 输入嵌入，类型为可选的长整型张量
        inputs_embeds: Optional[torch.LongTensor] = None,
        # 头部掩码，类型为可选的浮点张量
        head_mask: Optional[torch.FloatTensor] = None,
        # 交叉注意力头部掩码，类型为可选的张量
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        # 过去的键值对，类型为可选的元组
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        # 是否使用缓存，类型为可选的布尔值
        use_cache: Optional[bool] = None,
        # 是否输出注意力权重，类型为可选的布尔值
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态，类型为可选的布尔值
        output_hidden_states: Optional[bool] = None,
        # 标签，类型为可选的长整型张量
        labels: Optional[torch.LongTensor] = None,
        # 是否返回字典，类型为可选的布尔值
        return_dict: Optional[bool] = None,
        # 其他参数
        **kwargs,
# 这是一个继承自 Pix2StructPreTrainedModel 的条件生成模型类，具有语言建模头部，可用于序列生成任务
@add_start_docstrings(
    "A conditional generation model with a language modeling head. Can be used for sequence generation tasks.",
    PIX2STRUCT_START_DOCSTRING,
)
class Pix2StructForConditionalGeneration(Pix2StructPreTrainedModel):
    # 指定配置类为 Pix2StructConfig
    config_class = Pix2StructConfig
    # 主要输入名称为 "flattened_patches"
    main_input_name = "flattened_patches"
    # 绑定权重的键
    _tied_weights_keys = ["decoder.lm_head.weight"]

    def __init__(self, config: Pix2StructConfig):
        # 调用父类的构造函数
        super().__init__(config)
        
        # 创建视觉模型编码器
        self.encoder = Pix2StructVisionModel(config.vision_config)
        # 创建文本模型解码器
        self.decoder = Pix2StructTextModel(config.text_config)

        # 标记是否为 VQA 任务
        self.is_vqa = config.is_vqa

        # 初始化权重并应用最终处理
        self.post_init()

    # 获取输入嵌入层
    def get_input_embeddings(self):
        return self.decoder.get_input_embeddings()

    # 设置输入嵌入层
    def set_input_embeddings(self, new_embeddings):
        self.decoder.set_input_embeddings(new_embeddings)

    # 获取输出嵌入层
    def get_output_embeddings(self) -> nn.Module:
        return self.decoder.get_output_embeddings()

    # 设置输出嵌入层
    def set_output_embeddings(self, new_embeddings):
        self.decoder.set_output_embeddings(new_embeddings)

    # 调整词汇表大小
    def resize_token_embeddings(self, new_num_tokens: Optional[int] = None) -> nn.Embedding:
        model_embeds = self.decoder.resize_token_embeddings(new_num_tokens)
        # 更新配置中的词汇表大小
        self.config.text_config.vocab_size = new_num_tokens
        return model_embeds

    # 获取解码器
    def get_decoder(self):
        return self.decoder

    # 获取编码器
    def get_encoder(self):
        return self.encoder

    # 添加 docstring 到模型前向传播
    @add_start_docstrings_to_model_forward(PIX2STRUCT_INPUTS_DOCSTRING)
    # 替换返回文档
    @replace_return_docstrings(output_type=Seq2SeqModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        flattened_patches: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        decoder_inputs_embeds: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    # 为生成准备输入数据
    def prepare_inputs_for_generation(
        self,
        input_ids,
        flattened_patches: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        past_key_values=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # 如果没有提供decoder_attention_mask，则使用全1的注意力掩码
        if decoder_attention_mask is None:
            decoder_attention_mask = torch.ones_like(input_ids).to(input_ids.device)

        # 如果使用了past_key_values，则修剪decoder_input_ids
        if past_key_values is not None:
            # 获取过去的序列长度
            past_length = past_key_values[0][0].shape[2]

            # 如果input_ids的长度大于past_length，则只保留过去序列之后的部分
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # 否则，默认只保留最后一个输入ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        # 返回输入数据字典
        return {
            "flattened_patches": flattened_patches,
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }
```