# `.\transformers\models\audio_spectrogram_transformer\modeling_audio_spectrogram_transformer.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，版权归 MIT 和 The HuggingFace Inc. 团队所有
#
# 根据 Apache 许可证 2.0 版本使用此文件（"许可证"）；
# 除非遵守许可证，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按"原样"提供许可证，不提供任何形式的明示或暗示担保。
# 请参阅许可证了解特定语言下的权限和限制。
""" PyTorch 音频频谱变换器（AST）模型。"""

import math
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_audio_spectrogram_transformer import ASTConfig


logger = logging.get_logger(__name__)

# 用于文档字符串的配置信息
_CONFIG_FOR_DOC = "ASTConfig"

# 用于文档字符串的检查点信息
_CHECKPOINT_FOR_DOC = "MIT/ast-finetuned-audioset-10-10-0.4593"
_EXPECTED_OUTPUT_SHAPE = [1, 1214, 768]

# 音频分类的文档字符串信息
_SEQ_CLASS_CHECKPOINT = "MIT/ast-finetuned-audioset-10-10-0.4593"
_SEQ_CLASS_EXPECTED_OUTPUT = "'Speech'"
_SEQ_CLASS_EXPECTED_LOSS = 0.17

# 音频频谱变换器预训练模型存档列表
AUDIO_SPECTROGRAM_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    # 查看所有音频频谱变换器模型 https://huggingface.co/models?filter=ast
]


class ASTEmbeddings(nn.Module):
    """
    构建 CLS 令牌、位置和补丁嵌入。
    """

    def __init__(self, config: ASTConfig) -> None:
        super().__init__()

        # 初始化 CLS 令牌
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        # 初始化蒸馏令牌
        self.distillation_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        # 初始化补丁嵌入
        self.patch_embeddings = ASTPatchEmbeddings(config)

        # 获取位置嵌入的维度信息
        frequency_out_dimension, time_out_dimension = self.get_shape(config)
        # 计算补丁数量
        num_patches = frequency_out_dimension * time_out_dimension
        # 初始化位置嵌入
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 2, config.hidden_size))
        # 初始化 Dropout 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config
    # 获取模型输出的频率和时间维度大小，参考 Karpathy 的 cs231n 博客中关于如何计算输出维度的说明
    def get_shape(self, config):
        # 计算频率维度的输出大小
        frequency_out_dimension = (config.num_mel_bins - config.patch_size) // config.frequency_stride + 1
        # 计算时间维度的输出大小
        time_out_dimension = (config.max_length - config.patch_size) // config.time_stride + 1

        return frequency_out_dimension, time_out_dimension

    # 前向传播函数，接受输入张量并返回张量
    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        # 获取输入张量的批量大小
        batch_size = input_values.shape[0]
        # 将输入张量通过 patch_embeddings 方法进行嵌入
        embeddings = self.patch_embeddings(input_values)

        # 使用预定义的 cls_token 来扩展为当前批次大小
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # 使用预定义的 distillation_token 来扩展为当前批次大小
        distillation_tokens = self.distillation_token.expand(batch_size, -1, -1)
        # 将嵌入的张量与 cls_tokens 和 distillation_tokens 连接在一起，按指定维度连接
        embeddings = torch.cat((cls_tokens, distillation_tokens, embeddings), dim=1)
        # 将位置嵌入添加到嵌入张量中
        embeddings = embeddings + self.position_embeddings
        # 对嵌入张量进行 dropout 操作
        embeddings = self.dropout(embeddings)

        # 返回处理后的嵌入张量
        return embeddings
class ASTPatchEmbeddings(nn.Module):
    """
    This class turns `input_values` into the initial `hidden_states` (patch embeddings) of shape `(batch_size,
    seq_length, hidden_size)` to be consumed by a Transformer.
    """

    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()

        # 从配置中获取补丁大小、频率步长和时间步长
        patch_size = config.patch_size
        frequency_stride = config.frequency_stride
        time_stride = config.time_stride

        # 使用一维卷积对输入进行投影，输出为形状为 (batch_size, seq_length, hidden_size) 的补丁嵌入
        self.projection = nn.Conv2d(
            1, config.hidden_size, kernel_size=(patch_size, patch_size), stride=(frequency_stride, time_stride)
        )

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        # 在通道维度上添加一个维度
        input_values = input_values.unsqueeze(1)
        # 转置输入张量的最后两个维度
        input_values = input_values.transpose(2, 3)
        # 使用投影层对输入进行投影，并展平结果张量
        embeddings = self.projection(input_values).flatten(2).transpose(1, 2)
        # 返回嵌入张量
        return embeddings


# Copied from transformers.models.vit.modeling_vit.ViTSelfAttention with ViT->AST
class ASTSelfAttention(nn.Module):
    def __init__(self, config: ASTConfig) -> None:
        # 调用父类的初始化方法
        super().__init__()
        # 如果隐藏大小不是注意力头数的倍数且没有嵌入大小属性，则引发值错误
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        # 初始化注意力头数、注意力头大小和全部头大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 初始化查询、键和值的线性层
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        # 初始化丢弃层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        # 改变张量的形状以适应多头注意力计算
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
        ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        # 使用 query 网络层对隐藏状态进行处理
        mixed_query_layer = self.query(hidden_states)

        # 使用 key 网络层对隐藏状态进行处理，并转置以便计算注意力分数
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        # 使用 value 网络层对隐藏状态进行处理，并转置以便计算注意力分数
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        # 使用 mixed_query_layer 转置以便计算注意力分数
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # 计算原始注意力分数，即 query 和 key 的点积
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # 对注意力分数进行缩放
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # 将注意力分数归一化为概率
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # 使用 dropout 进行注意力概率的随机失活
        attention_probs = self.dropout(attention_probs)

        # 如果存在 head_mask，则对注意力概率进行掩码处理
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # 计算上下文向量，即注意力概率与 value 的乘积
        context_layer = torch.matmul(attention_probs, value_layer)

        # 调整上下文向量的维度顺序
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        # 根据是否需要输出注意力权重，返回不同的输出结果
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs
# 从transformers.models.vit.modeling_vit.ViTSelfOutput复制过来，并将ViT->AST
class ASTSelfOutput(nn.Module):
    """
    The residual connection is defined in ASTLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: ASTConfig) -> None:
        super().__init__()
        # 创建一个全连接层，将输入特征的大小映射到相同的大小
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # Dropout层，以一定概率将输入张量的元素置零，用于防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将输入张量经过全连接层
        hidden_states = self.dense(hidden_states)
        # 对全连接层的输出进行Dropout
        hidden_states = self.dropout(hidden_states)

        return hidden_states


# 从transformers.models.vit.modeling_vit.ViTAttention复制过来，并将ViT->AST
class ASTAttention(nn.Module):
    def __init__(self, config: ASTConfig) -> None:
        super().__init__()
        # 自注意力层
        self.attention = ASTSelfAttention(config)
        # 自注意力输出层
        self.output = ASTSelfOutput(config)
        # 存储被剪枝的注意力头的集合
        self.pruned_heads = set()

    def prune_heads(self, heads: Set[int]) -> None:
        if len(heads) == 0:
            return
        # 找到可剪枝的注意力头
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # 剪枝线性层
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储被剪枝的头
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        # 调用自注意力层的前向传播方法
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)

        # 将自注意力输出经过自注意力输出层
        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # 如果有输出注意力，则添加到输出中
        return outputs


# 从transformers.models.vit.modeling_vit.ViTIntermediate复制过来，并将ViT->AST
class ASTIntermediate(nn.Module):
    def __init__(self, config: ASTConfig) -> None:
        super().__init__()
        # 全连接层，将隐藏大小映射到中间大小
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            # 如果激活函数是字符串，则根据ACT2FN映射表选择对应的激活函数
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            # 否则直接使用给定的激活函数
            self.intermediate_act_fn = config.hidden_act
    # 定义一个前向传播函数，接受隐藏状态作为输入并返回处理后的隐藏状态
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 使用全连接层对隐藏状态进行处理
        hidden_states = self.dense(hidden_states)
        # 使用激活函数对处理后的隐藏状态进行处理
        hidden_states = self.intermediate_act_fn(hidden_states)

        # 返回处理后的隐藏状态
        return hidden_states
# 从transformers.models.vit.modeling_vit.ViTOutput复制代码，并将ViT更改为AST
class ASTOutput(nn.Module):
    def __init__(self, config: ASTConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = hidden_states + input_tensor

        return hidden_states


# 从transformers.models.vit.modeling_vit.ViTLayer复制代码，并将ViT更改为AST
class ASTLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: ASTConfig) -> None:
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = ASTAttention(config)
        self.intermediate = ASTIntermediate(config)
        self.output = ASTOutput(config)
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # 在AST中，self-attention之前应用layernorm
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # 如果输出注意力权重，则添加self attentions

        # 第一个残差连接
        hidden_states = attention_output + hidden_states

        # 在AST中，self-attention之后也应用layernorm
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # 第二个残差连接在这里完成
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs


# 从transformers.models.vit.modeling_vit.ViTEncoder复制代码，并将ViT更改为AST
class ASTEncoder(nn.Module):
    def __init__(self, config: ASTConfig) -> None:
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([ASTLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    # 定义一个方法，用于处理Transformer的层级结构，返回输出的隐藏状态和其他附加信息
    ) -> Union[tuple, BaseModelOutput]:
        # 如果需要输出所有隐藏状态，则初始化一个空元组，否则设置为None
        all_hidden_states = () if output_hidden_states else None
        # 如果需要输出注意力权重，则初始化一个空元组，否则设置为None
        all_self_attentions = () if output_attentions else None

        # 遍历Transformer的每一层
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出所有隐藏状态，则将当前隐藏状态添加到all_hidden_states元组中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 如果提供了头部掩码，则使用它，否则设置为None
            layer_head_mask = head_mask[i] if head_mask is not None else None

            # 如果启用了梯度检查点且正在训练阶段，则使用梯度检查点函数计算当前层的输出
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                # 否则直接调用当前层的__call__方法计算输出
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)

            # 更新隐藏状态为当前层的输出的第一个元素（即隐藏状态）
            hidden_states = layer_outputs[0]

            # 如果需要输出注意力权重，则将当前层的注意力权重添加到all_self_attentions元组中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 如果需要输出所有隐藏状态，则将最后一个隐藏状态添加到all_hidden_states元组中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不需要返回字典形式的结果，则将输出的各个部分打包成元组返回
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        # 否则，返回一个BaseModelOutput对象，包含最后的隐藏状态、所有隐藏状态和所有注意力权重
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
class ASTPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 设定配置类为 ASTConfig
    config_class = ASTConfig
    # 设定基础模型前缀为 "audio_spectrogram_transformer"
    base_model_prefix = "audio_spectrogram_transformer"
    # 设定主要输入名称为 "input_values"
    main_input_name = "input_values"
    # 设置支持梯度检查点的标志为 True
    supports_gradient_checkpointing = True

    # Copied from transformers.models.deit.modeling_deit.DeiTPreTrainedModel._init_weights
    # 从 DeiT 模型中复制的权重初始化函数
    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        # 如果模块是线性层或卷积层
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 将输入向上转换为 'fp32'，然后将其转换回所需的 'dtype'，以避免在 'half' 上出现 'trunc_normal_cpu' 未实现的问题
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32), mean=0.0, std=self.config.initializer_range
            ).to(module.weight.dtype)
            # 如果存在偏置，则将其数据清零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果模块是 LayerNorm 层
        elif isinstance(module, nn.LayerNorm):
            # 将偏置数据清零
            module.bias.data.zero_()
            # 将权重数据设置为全1
            module.weight.data.fill_(1.0)


AUDIO_SPECTROGRAM_TRANSFORMER_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`ASTConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

AUDIO_SPECTROGRAM_TRANSFORMER_INPUTS_DOCSTRING = r"""
    Args:
        input_values (`torch.FloatTensor` of shape `(batch_size, max_length, num_mel_bins)`):
            从原始音频波形提取的浮点值 mel 特征。原始音频波形可以通过将 `.flac` 或 `.wav` 音频文件加载到 `List[float]` 类型的数组或 `numpy.ndarray` 中获得，例如通过 soundfile 库 (`pip install soundfile`)。要将数组准备成 `input_features`，应使用 [`AutoFeatureExtractor`] 提取 mel 特征，填充并转换为 `torch.FloatTensor` 类型的张量。参见 [`~ASTFeatureExtractor.__call__`]

        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            用于使自注意力模块的某些头部失效的掩码。掩码值在 `[0, 1]` 中选择：

            - 1 表示头部**未被掩码**，
            - 0 表示头部**被掩码**。

        output_attentions (`bool`, *optional*):
            是否返回所有注意力层的注意力张量。有关更多详细信息，请参阅返回的张量中的 `attentions`。
        output_hidden_states (`bool`, *optional*):
            是否返回所有层的隐藏状态。有关更多详细信息，请参阅返回的张量中的 `hidden_states`。
        return_dict (`bool`, *optional*):
            是否返回 [`~utils.ModelOutput`] 而不是普通元组。
"""
Transformer 模型，输出未经特定头部处理的原始隐藏状态。
"""
class ASTModel(ASTPreTrainedModel):
    def __init__(self, config: ASTConfig) -> None:
        super().__init__(config)
        self.config = config

        # 初始化嵌入层和编码器
        self.embeddings = ASTEmbeddings(config)
        self.encoder = ASTEncoder(config)

        # 对隐藏状态应用层归一化
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self) -> ASTPatchEmbeddings:
        """
        返回输入嵌入层
        """
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        剪枝模型的注意力头部。
        heads_to_prune: 要在该层中剪枝的头部的字典 {layer_num: 要在此层中剪枝的头部列表}
        参见基类 PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> BaseModelOutputWithPooling:
        """
        前向传播函数
        """
        # 输入值：可选的输入张量
        # head_mask: 可选的头部屏蔽
        # output_attentions: 可选的输出注意力权重
        # output_hidden_states: 可选的输出隐藏状态
        # return_dict: 可选的返回字典
        # kwargs: 其他参数
    # 定义函数，接收输入值并返回模型输出
    # 返回类型为 BaseModelOutputWithPooling 或 Tuple
    def forward(
        self,
        input_values: Optional[Union[torch.Tensor, Tuple[torch.Tensor, ...]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPooling]:
        # 如果未提供 output_attentions 参数，则使用配置中的默认值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果未提供 output_hidden_states 参数，则使用配置中的默认值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果未提供 return_dict 参数，则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果未提供 input_values 参数，则抛出 ValueError 异常
        if input_values is None:
            raise ValueError("You have to specify input_values")

        # 如果需要，准备头部遮罩
        # 头部遮罩中的 1.0 表示保留该头部
        # attention_probs 的形状为 bsz x n_heads x N x N
        # 输入的 head_mask 的形状为 [num_heads] 或 [num_hidden_layers x num_heads]
        # 并且将 head_mask 转换为形状 [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # 嵌入层处理输入值
        embedding_output = self.embeddings(input_values)

        # 编码器处理嵌入输出
        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 获取序列输出
        sequence_output = encoder_outputs[0]
        # 序列输出经 Layernorm 处理
        sequence_output = self.layernorm(sequence_output)

        # 汇聚输出为序列输出的第一个和第二个元素的平均值
        pooled_output = (sequence_output[:, 0] + sequence_output[:, 1]) / 2

        # 如果不需要返回字典，则返回元组
        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        # 否则返回 BaseModelOutputWithPooling 类型的对象
        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
# 定义一个 ASTMLPHead 类，继承自 nn.Module
class ASTMLPHead(nn.Module):
    # 构造函数，接受一个 ASTConfig 类型的参数 config
    def __init__(self, config: ASTConfig):
        # 调用父类的构造函数
        super().__init__()
        # 初始化一个 LayerNorm 层，对隐藏状态进行标准化，eps 参数指定标准化时的 epsilon 值
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 如果 num_labels 大于 0，则初始化一个线性层，否则初始化一个恒等映射层
        self.dense = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

    # 前向传播方法，接受隐藏状态作为输入
    def forward(self, hidden_state):
        # 对隐藏状态进行 LayerNorm 标准化
        hidden_state = self.layernorm(hidden_state)
        # 通过线性层进行映射
        hidden_state = self.dense(hidden_state)
        # 返回处理后的隐藏状态
        return hidden_state


# 将文档字符串添加到 ASTForAudioClassification 类中，用于模型注释和文档生成
@add_start_docstrings(
    """
    带有音频分类头部的音频谱图 Transformer 模型（在汇总输出之上的线性层），例如用于 AudioSet、Speech Commands v2 等数据集。
    """,
    AUDIO_SPECTROGRAM_TRANSFORMER_START_DOCSTRING,
)
# 定义 ASTForAudioClassification 类，继承自 ASTPreTrainedModel 类
class ASTForAudioClassification(ASTPreTrainedModel):
    # 构造函数，接受一个 ASTConfig 类型的参数 config
    def __init__(self, config: ASTConfig) -> None:
        # 调用父类的构造函数
        super().__init__(config)

        # 初始化 num_labels 属性为 config.num_labels
        self.num_labels = config.num_labels
        # 初始化音频谱图 Transformer 模型
        self.audio_spectrogram_transformer = ASTModel(config)

        # 初始化分类器头部
        self.classifier = ASTMLPHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播方法，接受输入参数，并返回模型输出
    @add_start_docstrings_to_model_forward(AUDIO_SPECTROGRAM_TRANSFORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_SEQ_CLASS_CHECKPOINT,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="audio",
        expected_output=_SEQ_CLASS_EXPECTED_OUTPUT,
        expected_loss=_SEQ_CLASS_EXPECTED_LOSS,
    )
    def forward(
        self,
        input_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the audio classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 确保返回字典不为空，如果为空则使用配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将输入值传递给音频频谱转换器，并返回输出
        outputs = self.audio_spectrogram_transformer(
            input_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从输出中获取池化后的输出
        pooled_output = outputs[1]
        # 将池化后的输出传递给分类器以获得 logits
        logits = self.classifier(pooled_output)

        # 初始化损失为 None
        loss = None
        # 如果存在标签
        if labels is not None:
            # 如果问题类型未定义
            if self.config.problem_type is None:
                # 根据标签数量设置问题类型
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型计算损失
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    # 如果只有一个标签，计算均方误差损失
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                # 如果是单标签分类问题，使用交叉熵损失函数
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                # 如果是多标签分类问题，使用带 Logits 的二元交叉熵损失函数
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        # 如果不需要返回字典，则返回元组
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果需要返回字典，则返回 SequenceClassifierOutput 类型对象
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```