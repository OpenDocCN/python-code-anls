# `.\models\audio_spectrogram_transformer\modeling_audio_spectrogram_transformer.py`

```py
# 设置文件编码为 UTF-8
# 版权声明，这段代码版权属于 MIT 和 HuggingFace Inc. 团队，保留所有权利
#
# 根据 Apache License, Version 2.0 许可，除非符合许可证要求，否则不得使用此文件
# 您可以在以下网址获取许可证副本：http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，本软件是基于“原样”提供的，没有任何明示或暗示的担保或条件
# 有关详细信息，请参阅许可证

""" PyTorch Audio Spectrogram Transformer (AST) model."""
# 引入数学库
import math
# 引入类型提示
from typing import Dict, List, Optional, Set, Tuple, Union

# 引入 PyTorch 库
import torch
# 引入 PyTorch 的检查点工具
import torch.utils.checkpoint
# 引入 PyTorch 中的神经网络模块
from torch import nn
# 引入 PyTorch 中的损失函数
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 引入激活函数映射
from ...activations import ACT2FN
# 引入模型输出
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, SequenceClassifierOutput
# 引入预训练模型工具
from ...modeling_utils import PreTrainedModel
# 引入 PyTorch 实用工具
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
# 引入日志工具
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
# 引入 AST 配置
from .configuration_audio_spectrogram_transformer import ASTConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 模型配置文档字符串
_CONFIG_FOR_DOC = "ASTConfig"

# 检查点文档字符串
_CHECKPOINT_FOR_DOC = "MIT/ast-finetuned-audioset-10-10-0.4593"
# 预期输出形状文档字符串
_EXPECTED_OUTPUT_SHAPE = [1, 1214, 768]

# 音频分类检查点文档字符串
_SEQ_CLASS_CHECKPOINT = "MIT/ast-finetuned-audioset-10-10-0.4593"
# 音频分类预期输出文档字符串
_SEQ_CLASS_EXPECTED_OUTPUT = "'Speech'"
# 音频分类预期损失文档字符串
_SEQ_CLASS_EXPECTED_LOSS = 0.17

# 音频频谱变换预训练模型存档列表
AUDIO_SPECTROGRAM_TRANSFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    # 查看所有音频频谱变换模型，请访问 https://huggingface.co/models?filter=ast
]

# ASTEmbeddings 类定义，继承自 nn.Module
class ASTEmbeddings(nn.Module):
    """
    构建 CLS 标记、位置和补丁嵌入。
    """

    def __init__(self, config: ASTConfig) -> None:
        super().__init__()

        # 定义 CLS 标记参数
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        # 定义蒸馏标记参数
        self.distillation_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        # 初始化补丁嵌入
        self.patch_embeddings = ASTPatchEmbeddings(config)

        # 获取频率和时间输出维度形状
        frequency_out_dimension, time_out_dimension = self.get_shape(config)
        # 计算补丁数
        num_patches = frequency_out_dimension * time_out_dimension
        # 定义位置嵌入参数
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 2, config.hidden_size))
        # 定义 Dropout 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 保存配置
        self.config = config
    # 定义一个方法用于获取输出的形状，基于给定的配置参数
    def get_shape(self, config):
        # 根据 Karpathy 在 cs231n 博客中的方法计算频率输出的维度
        # https://cs231n.github.io/convolutional-networks/#conv
        frequency_out_dimension = (config.num_mel_bins - config.patch_size) // config.frequency_stride + 1
        # 根据 Karpathy 在 cs231n 博客中的方法计算时间输出的维度
        # https://cs231n.github.io/convolutional-networks/#conv
        time_out_dimension = (config.max_length - config.patch_size) // config.time_stride + 1

        # 返回计算得到的频率和时间输出维度
        return frequency_out_dimension, time_out_dimension

    # 定义一个前向传播方法，输入是一个 torch.Tensor，输出也是一个 torch.Tensor
    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        # 获取输入数据的批量大小
        batch_size = input_values.shape[0]
        
        # 将输入数据通过 patch_embeddings 方法进行嵌入
        embeddings = self.patch_embeddings(input_values)

        # 使用 self.cls_token 扩展成 batch_size 行，-1 列的张量
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        
        # 使用 self.distillation_token 扩展成 batch_size 行，-1 列的张量
        distillation_tokens = self.distillation_token.expand(batch_size, -1, -1)
        
        # 将 cls_tokens、distillation_tokens 和 embeddings 沿着第一维度拼接起来
        embeddings = torch.cat((cls_tokens, distillation_tokens, embeddings), dim=1)
        
        # 将位置嵌入加到 embeddings 上
        embeddings = embeddings + self.position_embeddings
        
        # 对 embeddings 进行 dropout 操作
        embeddings = self.dropout(embeddings)

        # 返回处理后的 embeddings 张量作为前向传播的输出
        return embeddings
    # ASTSelfAttention 类的构造函数，初始化自注意力机制模块
    def __init__(self, config: ASTConfig) -> None:
        super().__init__()
        # 检查隐藏大小是否可以被注意力头数整除，若不能且没有嵌入大小，则引发值错误
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        # 设置注意力头数和每个注意力头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 创建用于查询、键和值的线性变换层
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        # Dropout 层，用于注意力概率的随机失活
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    # 将输入张量转换为分数矩阵形式的函数
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        # 定义新的形状以便于注意力分数计算，并进行维度置换
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    # ASTSelfAttention 类的前向传播函数，实现自注意力机制
    def forward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    # 定义函数返回类型为一个元组，包含一个 torch.Tensor 类型的上下文层和一个 torch.Tensor 类型的注意力概率
    -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        # 使用 self.query 对隐藏状态进行查询，生成混合查询层
        mixed_query_layer = self.query(hidden_states)

        # 使用 self.key 对隐藏状态进行键的转换，并为计算注意力分数准备转置
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        
        # 使用 self.value 对隐藏状态进行值的转换，并为计算上下文层准备转置
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # 对混合查询层进行转置，为计算注意力分数准备
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # 计算原始的注意力分数，通过 query_layer 和 key_layer 的点积得到
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # 根据注意力头的大小对注意力分数进行缩放
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # 将注意力分数归一化为概率分布
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # 对注意力概率进行 dropout 操作，以防止过拟合
        attention_probs = self.dropout(attention_probs)

        # 如果给定了头部掩码，将注意力概率与头部掩码相乘，实现掩码操作
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # 计算上下文层，将注意力概率与 value_layer 相乘得到加权和
        context_layer = torch.matmul(attention_probs, value_layer)

        # 对上下文层进行维度置换和连续化操作，以便后续的形状变换
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()

        # 根据新的上下文层形状，进行视图变换，以匹配所有注意力头的输出维度
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        # 根据是否需要输出注意力权重，选择性地返回上下文层和注意力概率
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        # 返回函数的输出结果
        return outputs
# Copied from transformers.models.vit.modeling_vit.ViTSelfOutput with ViT->AST
class ASTSelfOutput(nn.Module):
    """
    The residual connection is defined in ASTLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: ASTConfig) -> None:
        super().__init__()
        # 定义一个全连接层，输入和输出大小都是 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 定义一个 Dropout 层，使用的 dropout 概率是 config.hidden_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将输入的 hidden_states 应用全连接层 self.dense
        hidden_states = self.dense(hidden_states)
        # 对应用全连接层后的 hidden_states 应用 dropout
        hidden_states = self.dropout(hidden_states)

        return hidden_states


# Copied from transformers.models.vit.modeling_vit.ViTAttention with ViT->AST
class ASTAttention(nn.Module):
    def __init__(self, config: ASTConfig) -> None:
        super().__init__()
        # 定义一个 ASTSelfAttention 层
        self.attention = ASTSelfAttention(config)
        # 定义一个 ASTSelfOutput 层
        self.output = ASTSelfOutput(config)
        # 初始化一个空的集合，用于存储被剪枝的注意力头
        self.pruned_heads = set()

    def prune_heads(self, heads: Set[int]) -> None:
        if len(heads) == 0:
            return
        # 调用外部函数 find_pruneable_heads_and_indices，找到可剪枝的头部和索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # 剪枝线性层
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储被剪枝的头部
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        # 调用 self.attention 的 forward 方法，得到 attention 的输出
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)

        # 将 attention 的输出和输入的 hidden_states 应用到 self.output 层
        attention_output = self.output(self_outputs[0], hidden_states)

        outputs = (attention_output,) + self_outputs[1:]  # 如果需要输出 attentions，则添加到 outputs 中
        return outputs


# Copied from transformers.models.vit.modeling_vit.ViTIntermediate with ViT->AST
class ASTIntermediate(nn.Module):
    def __init__(self, config: ASTConfig) -> None:
        super().__init__()
        # 定义一个全连接层，输入大小是 config.hidden_size，输出大小是 config.intermediate_size
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 根据 config.hidden_act 的类型选择相应的激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
    # 定义一个前向传播方法，接受一个名为hidden_states的张量作为输入，并返回一个张量作为输出
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将输入张量通过全连接层dense进行线性变换
        hidden_states = self.dense(hidden_states)
        # 将经过全连接层后的张量输入到激活函数intermediate_act_fn中进行非线性变换
        hidden_states = self.intermediate_act_fn(hidden_states)

        # 返回经过线性变换和非线性变换后的张量作为输出
        return hidden_states
# Copied from transformers.models.vit.modeling_vit.ViTOutput with ViT->AST
class ASTOutput(nn.Module):
    def __init__(self, config: ASTConfig) -> None:
        super().__init__()
        # 创建一个全连接层，输入维度为 config.intermediate_size，输出维度为 config.hidden_size
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建一个 Dropout 层，用于在训练过程中随机置零输入张量的元素，以防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将输入张量通过全连接层映射到新的张量 hidden_states
        hidden_states = self.dense(hidden_states)
        # 对 hidden_states 应用 Dropout 操作
        hidden_states = self.dropout(hidden_states)

        # 将 dropout 后的 hidden_states 与输入张量 input_tensor 相加，实现残差连接
        hidden_states = hidden_states + input_tensor

        return hidden_states


# Copied from transformers.models.vit.modeling_vit.ViTLayer with ViT->AST
class ASTLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: ASTConfig) -> None:
        super().__init__()
        # 设置每个前馈分块的大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 序列长度的维度
        self.seq_len_dim = 1
        # 创建自注意力机制对象
        self.attention = ASTAttention(config)
        # 创建中间层对象
        self.intermediate = ASTIntermediate(config)
        # 创建输出层对象
        self.output = ASTOutput(config)
        # 创建前层归一化对象，在隐藏大小维度上应用 LayerNorm
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建后层归一化对象，在隐藏大小维度上应用 LayerNorm
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        # 应用前层归一化后，传入自注意力层进行计算
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),
            head_mask,
            output_attentions=output_attentions,
        )
        # 获取自注意力层的输出张量
        attention_output = self_attention_outputs[0]
        # 如果输出注意力权重，则将其添加到输出元组中
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # 实现第一个残差连接
        hidden_states = attention_output + hidden_states

        # 在中间层的输出上应用后层归一化
        layer_output = self.layernorm_after(hidden_states)
        # 在中间层上应用中间层对象进行进一步处理
        layer_output = self.intermediate(layer_output)

        # 在输出层对象上执行第二个残差连接
        layer_output = self.output(layer_output, hidden_states)

        # 将层输出添加到输出元组中
        outputs = (layer_output,) + outputs

        return outputs


# Copied from transformers.models.vit.modeling_vit.ViTEncoder with ViT->AST
class ASTEncoder(nn.Module):
    def __init__(self, config: ASTConfig) -> None:
        super().__init__()
        self.config = config
        # 使用 ASTLayer 对象的列表创建层的序列
        self.layer = nn.ModuleList([ASTLayer(config) for _ in range(config.num_hidden_layers)])
        # 设置梯度检查点为 False
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        ) -> Union[tuple, BaseModelOutput]:
        # 如果不需要输出隐藏状态，则初始化空元组；否则设为None
        all_hidden_states = () if output_hidden_states else None
        # 如果不需要输出注意力权重，则初始化空元组；否则设为None
        all_self_attentions = () if output_attentions else None

        # 遍历每一个层次模块
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，则将当前隐藏状态加入到all_hidden_states中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前层的头部掩码，如果head_mask存在的话
            layer_head_mask = head_mask[i] if head_mask is not None else None

            # 如果开启梯度检查点且处于训练阶段，则使用梯度检查点函数进行前向传播
            if self.gradient_checkpointing and self.training:
                # 调用梯度检查点函数，以节省内存开销
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                # 否则直接调用当前层的前向传播函数
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)

            # 更新当前隐藏状态为当前层的输出的第一个元素
            hidden_states = layer_outputs[0]

            # 如果需要输出注意力权重，则将当前层的注意力权重加入到all_self_attentions中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 如果需要输出隐藏状态，则将最终的隐藏状态加入到all_hidden_states中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不需要返回字典格式的结果，则返回一个元组，其中包含需要返回的非None的值
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        # 否则返回一个BaseModelOutput对象，包含最终的隐藏状态、所有隐藏状态、所有注意力权重
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

    # 设置配置类为 ASTConfig
    config_class = ASTConfig
    # 设置基础模型前缀为 "audio_spectrogram_transformer"
    base_model_prefix = "audio_spectrogram_transformer"
    # 设置主输入名称为 "input_values"
    main_input_name = "input_values"
    # 启用梯度检查点支持
    supports_gradient_checkpointing = True

    # 从 transformers 库中的 DeiTPreTrainedModel 类中复制的方法
    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 将输入升级为 `fp32`，然后转回到所需的 `dtype`，以避免 `trunc_normal_cpu` 在 `half` 模式下未实现的问题
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32), mean=0.0, std=self.config.initializer_range
            ).to(module.weight.dtype)
            if module.bias is not None:
                # 如果存在偏置项，将其数据初始化为零
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            # 如果是 LayerNorm 模块，将偏置项初始化为零，权重初始化为 1.0
            module.bias.data.zero_()
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
    This class can be used as a regular PyTorch Module. Please refer to the PyTorch documentation for general usage and
    behavior details.

    Parameters:
        config (:class:`~transformers.ASTConfig`):
            The configuration class holding all parameters of this model. Initializing with a configuration file only
            initializes the model configuration; it does not load the weights. For loading weights, use the
            :meth:`~transformers.PreTrainedModel.from_pretrained` method.
"""
    Args:
        input_values (`torch.FloatTensor` of shape `(batch_size, max_length, num_mel_bins)`):
            Float values mel features extracted from the raw audio waveform. Raw audio waveform can be obtained by
            loading a `.flac` or `.wav` audio file into an array of type `List[float]` or a `numpy.ndarray`, *e.g.* via
            the soundfile library (`pip install soundfile`). To prepare the array into `input_features`, the
            [`AutoFeatureExtractor`] should be used for extracting the mel features, padding and conversion into a
            tensor of type `torch.FloatTensor`. See [`~ASTFeatureExtractor.__call__`]

输入参数 `input_values`：
- 代表形状为 `(batch_size, max_length, num_mel_bins)` 的 `torch.FloatTensor`。
- 包含从原始音频波形提取的梅尔特征的浮点值。可以通过将 `.flac` 或 `.wav` 音频文件加载到 `List[float]` 或 `numpy.ndarray` 数组中获得原始音频波形。
- 要将数组准备成 `input_features`，应使用 [`AutoFeatureExtractor`] 提取梅尔特征，进行填充并转换为 `torch.FloatTensor` 类型的张量。参见 [`~ASTFeatureExtractor.__call__`]。


        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

输入参数 `head_mask`（可选）：
- 形状为 `(num_heads,)` 或 `(num_layers, num_heads)` 的 `torch.FloatTensor`。
- 用于屏蔽自注意力模块中选定头部的掩码。掩码值在 `[0, 1]` 范围内选择：

  - 1 表示头部 **未被屏蔽**，
  - 0 表示头部 **被屏蔽**。


        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.

输入参数 `output_attentions`（可选）：
- 布尔值，指示是否返回所有注意力层的注意力张量。
- 查看返回的张量中的 `attentions` 以获取更多详细信息。


        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.

输入参数 `output_hidden_states`（可选）：
- 布尔值，指示是否返回所有层的隐藏状态。
- 查看返回的张量中的 `hidden_states` 以获取更多详细信息。


        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.

输入参数 `return_dict`（可选）：
- 布尔值，指示是否返回 [`~utils.ModelOutput`] 而不是普通元组。
"""
This class defines a transformer model for AST (Audio Spectrogram Transformer) without a specific head for output.

@add_start_docstrings(
    "The bare AST Model transformer outputting raw hidden-states without any specific head on top.",
    AUDIO_SPECTROGRAM_TRANSFORMER_START_DOCSTRING,
)
class ASTModel(ASTPreTrainedModel):
    def __init__(self, config: ASTConfig) -> None:
        super().__init__(config)
        self.config = config

        # Initialize AST embeddings and encoder based on provided configuration
        self.embeddings = ASTEmbeddings(config)
        self.encoder = ASTEncoder(config)

        # Apply layer normalization across the hidden size dimension
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self) -> ASTPatchEmbeddings:
        # Retrieve the patch embeddings used for input
        return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        Prunes heads of the model.
        
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel.
        """
        for layer, heads in heads_to_prune.items():
            # Prune specified attention heads in each encoder layer
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(AUDIO_SPECTROGRAM_TRANSFORMER_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
        modality="audio",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        input_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ):
        """
        Perform forward pass of the AST model.

        Args:
            input_values (Optional[torch.Tensor]): Input tensor to the model.
            head_mask (Optional[torch.Tensor]): Mask to nullify selected heads of the model.
            output_attentions (Optional[bool]): Whether to output attentions weights.
            output_hidden_states (Optional[bool]): Whether to output hidden states.
            return_dict (Optional[bool]): Whether to return a dictionary.

        Returns:
            BaseModelOutputWithPooling: Output with pooled representation.
        """
        ) -> Union[Tuple, BaseModelOutputWithPooling]:
        # 如果未显式指定output_attentions，则使用配置中的默认值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果未显式指定output_hidden_states，则使用配置中的默认值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果未显式指定return_dict，则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_values is None:
            # 如果输入值为None，则抛出数值错误
            raise ValueError("You have to specify input_values")

        # 准备头部掩码（如果需要）
        # 头部掩码中的1.0表示保留该头部
        # attention_probs的形状为bsz x n_heads x N x N
        # 输入的head_mask形状为[num_heads]或[num_hidden_layers x num_heads]
        # head_mask被转换为形状[num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # 将输入值传递给嵌入层进行嵌入
        embedding_output = self.embeddings(input_values)

        # 将嵌入输出传递给编码器
        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 取编码器输出的第一个元素作为序列输出
        sequence_output = encoder_outputs[0]
        # 序列输出经过LayerNorm层处理
        sequence_output = self.layernorm(sequence_output)

        # 计算池化输出，取序列输出的第一个和第二个位置的平均值
        pooled_output = (sequence_output[:, 0] + sequence_output[:, 1]) / 2

        if not return_dict:
            # 如果不要求返回字典，则返回元组形式的输出
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        # 如果要求返回字典形式的输出，则创建BaseModelOutputWithPooling对象
        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
class ASTMLPHead(nn.Module):
    def __init__(self, config: ASTConfig):
        super().__init__()
        # 初始化一个 LayerNorm 层，用于标准化隐藏状态
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 如果标签数量大于0，则使用全连接层作为分类器；否则使用恒等映射
        self.dense = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

    def forward(self, hidden_state):
        # 对隐藏状态进行 LayerNorm 标准化
        hidden_state = self.layernorm(hidden_state)
        # 应用全连接层或恒等映射，得到分类结果
        hidden_state = self.dense(hidden_state)
        return hidden_state


@add_start_docstrings(
    """
    Audio Spectrogram Transformer model with an audio classification head on top (a linear layer on top of the pooled
    output) e.g. for datasets like AudioSet, Speech Commands v2.
    """,
    AUDIO_SPECTROGRAM_TRANSFORMER_START_DOCSTRING,
)
class ASTForAudioClassification(ASTPreTrainedModel):
    def __init__(self, config: ASTConfig) -> None:
        super().__init__(config)

        self.num_labels = config.num_labels
        # 初始化 ASTModel 类，该类用于处理音频谱图的转换
        self.audio_spectrogram_transformer = ASTModel(config)

        # 分类器头部
        self.classifier = ASTMLPHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

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
        # 确定是否返回字典形式的输出，如果未指定则使用配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用音频频谱变换器处理输入数据，获取输出
        outputs = self.audio_spectrogram_transformer(
            input_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从处理后的输出中获取汇聚输出（pooled_output）
        pooled_output = outputs[1]

        # 使用分类器对汇聚输出进行分类得到 logits
        logits = self.classifier(pooled_output)

        # 初始化损失为 None
        loss = None

        # 如果提供了标签，则计算损失
        if labels is not None:
            # 如果问题类型未定义，则根据标签数据类型和标签数量确定问题类型
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型选择相应的损失函数进行计算
            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        # 如果不需要返回字典形式的输出，则返回 logits 和可能的隐藏状态
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # 如果需要返回字典形式的输出，则返回 SequenceClassifierOutput 对象
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```