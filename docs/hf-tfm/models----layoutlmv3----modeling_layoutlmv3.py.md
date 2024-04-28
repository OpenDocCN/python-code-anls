# `.\models\layoutlmv3\modeling_layoutlmv3.py`

```
# 设置编码格式为 utf-8
# 版权声明：Microsoft Research 和 HuggingFace Inc. 团队
# 根据 Apache 许可证 2.0 版本授权
# 只能在遵守许可证的情况下使用此文件
# 可以在以下网址获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则软件是基于"AS IS"基础分发的
# 没有任何明示或隐含的担保或条件
# 请查看许可证以获取特定语言的权限和限制

"""PyTorch LayoutLMv3 model."""  # PyTorch LayoutLMv3 模型

import collections  # 导入 collections 模块
import math  # 导入 math 模块
from typing import Optional, Tuple, Union  # 导入类型注解

import torch  # 导入 PyTorch 库
import torch.nn as nn  # 导入 PyTorch 中的神经网络模块
import torch.nn.functional as F  # 导入 PyTorch 中的函数操作模块
import torch.utils.checkpoint  # 导入 PyTorch 中的 checkpoint 模块
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss  # 导入 PyTorch 中的损失函数

from ...activations import ACT2FN  # 从模块中导入 ACT2FN
from ...modeling_outputs import (  # 从 modeling_outputs 模块导入输出
    BaseModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel  # 从 modeling_utils 模块导入 PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward  # 从 pytorch_utils 模块导入 apply_chunking_to_forward
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings  # 从 utils 模块导入函数和模块
from .configuration_layoutlmv3 import LayoutLMv3Config  # 从 configuration_layoutlmv3 模块导入 LayoutLMv3Config

logger = logging.get_logger(__name__)  # 获取名为 __name__ 的 logger

_CONFIG_FOR_DOC = "LayoutLMv3Config"  # 用于文档的配置变量名

LAYOUTLMV3_PRETRAINED_MODEL_ARCHIVE_LIST = [  # 预训练模型的列表
    "microsoft/layoutlmv3-base",
    "microsoft/layoutlmv3-large",
    # 查看所有 LayoutLMv3 模型 https://huggingface.co/models?filter=layoutlmv3
]

LAYOUTLMV3_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`LayoutLMv3Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

LAYOUTLMV3_MODEL_INPUTS_DOCSTRING = r"""
"""

LAYOUTLMV3_DOWNSTREAM_INPUTS_DOCSTRING = r"""
"""

class LayoutLMv3PatchEmbeddings(nn.Module):  # 定义 LayoutLMv3PatchEmbeddings 类，继承自 nn.Module
    """LayoutLMv3 image (patch) embeddings. This class also automatically interpolates the position embeddings for varying
    image sizes."""  # LayoutLMv3 图像（块）嵌入。此类还会根据不同的图像大小自动插值位置嵌入
    # 初始化函数，接受包含配置信息的参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()

        # 根据配置信息确定输入图像尺寸
        image_size = (
            config.input_size
            if isinstance(config.input_size, collections.abc.Iterable)  # 如果输入尺寸是可迭代对象
            else (config.input_size, config.input_size)  # 否则将输入尺寸扩展为二维元组
        )
        
        # 根据配置信息确定图像块尺寸
        patch_size = (
            config.patch_size
            if isinstance(config.patch_size, collections.abc.Iterable)  # 如果块尺寸是可迭代对象
            else (config.patch_size, config.patch_size)  # 否则将块尺寸扩展为二维元组
        )
        
        # 计算图像块的形状
        self.patch_shape = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])
        
        # 使用卷积层进行投影映射，实现从通道数到隐藏层大小的转换
        self.proj = nn.Conv2d(config.num_channels, config.hidden_size, kernel_size=patch_size, stride=patch_size)

    # 前向传播函数，接收像素值和位置信息作为输入
    def forward(self, pixel_values, position_embedding=None):
        # 对输入像素值进行投影映射
        embeddings = self.proj(pixel_values)

        # 如果存在位置信息嵌入
        if position_embedding is not None:
            # 插值位置嵌入到相应的大小
            position_embedding = position_embedding.view(1, self.patch_shape[0], self.patch_shape[1], -1)
            position_embedding = position_embedding.permute(0, 3, 1, 2)
            patch_height, patch_width = embeddings.shape[2], embeddings.shape[3]
            position_embedding = F.interpolate(position_embedding, size=(patch_height, patch_width), mode="bicubic")
            # 将位置嵌入加到映射结果中
            embeddings = embeddings + position_embedding

        # 对映射结果进行展平处理，并调整维度
        embeddings = embeddings.flatten(2).transpose(1, 2)
        
        # 返回处理后的向量
        return embeddings
class LayoutLMv3TextEmbeddings(nn.Module):
    """
    LayoutLMv3 text embeddings. Same as `RobertaEmbeddings` but with added spatial (layout) embeddings.
    """

    def __init__(self, config):
        # 初始化 LayoutLMv3 文本嵌入层
        super().__init__()
        # 初始化词嵌入层，使用 nn.Embedding 创建词嵌入矩阵
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 初始化 token 类型嵌入层，使用 nn.Embedding 创建 token 类型嵌入矩阵
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # 初始化 LayerNorm 层，使用 nn.LayerNorm 创建 LayerNorm 层
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化 dropout 层，使用 nn.Dropout 创建 dropout 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        # 注册 position_ids 缓冲区，用于存储位置编码的张量
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

        # 设置 padding_idx 为 config.pad_token_id，初始化位置编码嵌入层
        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

        # 初始化 x、y 方向的位置编码嵌入层，使用 nn.Embedding 创建 x、y 方向的位置编码矩阵
        self.x_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.coordinate_size)
        self.y_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.coordinate_size)
        # 初始化 h、w 方向的位置编码嵌入层，使用 nn.Embedding 创建 h、w 方向的位置编码矩阵
        self.h_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.shape_size)
        self.w_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.shape_size)

    def calculate_spatial_position_embeddings(self, bbox):
        # 计算空间位置编码嵌入
        try:
            # 获取左、上、右、下方向的位置编码
            left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
            upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
            right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
            lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
        except IndexError as e:
            raise IndexError("The `bbox` coordinate values should be within 0-1000 range.") from e

        # 计算高度、宽度方向的位置编码
        h_position_embeddings = self.h_position_embeddings(torch.clip(bbox[:, :, 3] - bbox[:, :, 1], 0, 1023))
        w_position_embeddings = self.w_position_embeddings(torch.clip(bbox[:, :, 2] - bbox[:, :, 0], 0, 1023))

        # 合并空间位置编码
        spatial_position_embeddings = torch.cat(
            [
                left_position_embeddings,
                upper_position_embeddings,
                right_position_embeddings,
                lower_position_embeddings,
                h_position_embeddings,
                w_position_embeddings,
            ],
            dim=-1,
        )
        return spatial_position_embeddings
    def create_position_ids_from_input_ids(self, input_ids, padding_idx):
        """
        根据输入的 token ids 创建位置 id。任何填充的 token 仍然保持填充状态。
        位置编号从 padding_idx+1 开始。此函数修改自 fairseq 的 `utils.make_positions`。
        """
        # 创建一个与 input_ids 形状相同的 mask，标记非填充符号为1，填充符号为0
        mask = input_ids.ne(padding_idx).int()
        # 对 mask 按列累积求和，然后将数据类型转换为 mask 的数据类型
        incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask)) * mask
        # 将位置编号转换为长整型，然后加上 padding_idx
        return incremental_indices.long() + padding_idx

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        我们直接提供了嵌入向量。无法推断哪些是填充的，因此只生成连续的位置 id。
        """
        # 获取输入嵌入向量的形状
        input_shape = inputs_embeds.size()[:-1]
        # 获取序列长度
        sequence_length = input_shape[1]

        # 生成从 padding_idx+1 到 sequence_length+padding_idx+1 的连续位置 id
        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        # 在第一维上增加一个维度，并将其扩展到与输入形状相同
        return position_ids.unsqueeze(0).expand(input_shape)

    def forward(
        self,
        input_ids=None,
        bbox=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
    ):
        # 如果未提供位置 id，则根据输入的 token ids 创建位置 id
        if position_ids is None:
            if input_ids is not None:
                position_ids = self.create_position_ids_from_input_ids(input_ids, self.padding_idx).to(
                    input_ids.device
                )
            else:
                # 如果没有提供 token ids，则根据输入的嵌入向量创建位置 id
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        # 如果提供了 token ids，则获取其形状，否则获取输入嵌入向量的形状
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        # 如果未提供 token 类型 id，则创建一个与输入形状相同的全零张量
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        # 如果未提供输入嵌入向量，则通过 word_embeddings 方法获取
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        # 获取 token 类型嵌入向量
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 将输入嵌入向量与 token 类型嵌入向量相加
        embeddings = inputs_embeds + token_type_embeddings
        # 获取位置嵌入向量
        position_embeddings = self.position_embeddings(position_ids)
        # 将位置嵌入向量与输入嵌入向量相加
        embeddings += position_embeddings

        # 计算空间位置嵌入向量
        spatial_position_embeddings = self.calculate_spatial_position_embeddings(bbox)

        # 将空间位置嵌入向量与 embeddings 相加
        embeddings = embeddings + spatial_position_embeddings

        # 应用 LayerNorm
        embeddings = self.LayerNorm(embeddings)
        # 应用 dropout
        embeddings = self.dropout(embeddings)
        # 返回 embeddings
        return embeddings
class LayoutLMv3PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 配置文件的类，用于模型初始化
    config_class = LayoutLMv3Config
    # 基础模型的前缀名称
    base_model_prefix = "layoutlmv3"

    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果是线性层或卷积层
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 初始化权重为正态分布
            # 与 TF 版本稍有不同，TF 版本使用截断正态分布进行初始化
            # 参考 https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在偏置项，初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果是嵌入层
        elif isinstance(module, nn.Embedding):
            # 初始化权重为正态分布
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在填充索引，将填充索引对应的权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # 如果是 LayerNorm 层
        elif isinstance(module, nn.LayerNorm):
            # 初始化偏置项为零
            module.bias.data.zero_()
            # 初始化权重为 1
            module.weight.data.fill_(1.0)


class LayoutLMv3SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 如果隐藏层大小不是注意力头数量的倍数，并且配置中没有嵌入大小属性
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        # 注意力头数量
        self.num_attention_heads = config.num_attention_heads
        # 每个注意力头的大小
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        # 所有头的大小
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 查询权重
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        # 键权重
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        # 值权重
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # dropout 层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        # 是否具有相对注意力偏置
        self.has_relative_attention_bias = config.has_relative_attention_bias
        # 是否具有空间注意力偏置
        self.has_spatial_attention_bias = config.has_spatial_attention_bias

    def transpose_for_scores(self, x):
        # 转换张量形状以进行注意力计算
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
```  
    def cogview_attention(self, attention_scores, alpha=32):
        """
        cogview_attention方法用于实现新的注意力计算方法，即 PB-Relax，参考论文https://arxiv.org/abs/2105.13290 Section 2.4。
        PB-Relax 是对原始的 nn.Softmax(dim=-1)(attention_scores) 的替代方案。
        注意，新的注意力概率可能导致速度变慢且略有偏差。可以使用 torch.allclose(standard_attention_probs,
        cogview_attention_probs, atol=1e-08) 进行比较，其中较小的 atol（例如，1e-08）表示更好的匹配程度。
        """
        # 对注意力分数进行缩放，以平衡训练过程中的精度和速度
        scaled_attention_scores = attention_scores / alpha
        # 计算缩放后的注意力分数的最大值，并添加一个维度以便后续广播
        max_value = scaled_attention_scores.amax(dim=(-1)).unsqueeze(-1)
        # 应用 PB-Relax 方法，重新计算注意力分数
        new_attention_scores = (scaled_attention_scores - max_value) * alpha
        # 对新的注意力分数进行 Softmax 归一化
        return nn.Softmax(dim=-1)(new_attention_scores)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        rel_pos=None,
        rel_2d_pos=None,
    ):
        # 通过 query 网络处理隐藏状态
        mixed_query_layer = self.query(hidden_states)

        # 通过 key 网络处理隐藏状态，并转置
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        
        # 通过 value 网络处理隐藏状态，并转置
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # 通过 mixed_query_layer 网络处理隐藏状态，并转置
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # 计算“query”和“key”的点积以获得原始的注意力分数。
        # 注意力分数 QT K/√d 可能明显大于输入元素，并导致溢出。
        # 将计算顺序更改为 QT(K/√d) 可以缓解该问题。（https://arxiv.org/pdf/2105.13290.pdf）
        attention_scores = torch.matmul(query_layer / math.sqrt(self.attention_head_size), key_layer.transpose(-1, -2))

        if self.has_relative_attention_bias and self.has_spatial_attention_bias:
            attention_scores += (rel_pos + rel_2d_pos) / math.sqrt(self.attention_head_size)
        elif self.has_relative_attention_bias:
            attention_scores += rel_pos / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # 应用预先计算的注意力掩码（在 RobertaModel forward() 函数中为所有层预计算）
            attention_scores = attention_scores + attention_mask

        # 将注意力分数归一化为概率。
        # 使用 CogView 论文中的技巧来稳定训练
        attention_probs = self.cogview_attention(attention_scores)

        # 实际上是丢弃了整个要关注的标记，这可能看起来有点不寻常，但是取自原始 Transformer 论文。
        attention_probs = self.dropout(attention_probs)

        # 如果需要，屏蔽头部
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # 将注意力概率与值层进行矩阵相乘，得到上下文层
        context_layer = torch.matmul(attention_probs, value_layer)

        # 对上下文层进行转置、重组形状
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # 如果需要，输出注意力概率
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs
# 定义 LayoutLMv3SelfOutput 类，继承自 nn.Module
class LayoutLMv3SelfOutput(nn.Module):
    # 初始化方法，接受配置参数 config
    def __init__(self, config):
        super().__init__()
        # 创建全连接层，输入和输出维度均为 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 创建 LayerNorm 层，输入维度为 config.hidden_size，eps 参数为 config.layer_norm_eps
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建丢弃层，丢弃概率为 config.hidden_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播方法，接受隐藏状态 hidden_states 和输入张量 input_tensor
    # 返回变换后的隐藏状态张量
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 全连接操作，将隐藏状态进行变换
        hidden_states = self.dense(hidden_states)
        # 丢弃部分神经元
        hidden_states = self.dropout(hidden_states)
        # 对变换后的隐藏状态进行 LayerNorm 和输入张量相加
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回变换后的隐藏状态张量
        return hidden_states


# 定义 LayoutLMv3Attention 类，继承自 nn.Module
class LayoutLMv3Attention(nn.Module):
    # 初始化方法，接受配置参数 config
    def __init__(self, config):
        super().__init__()
        # 创建 LayoutLMv3SelfAttention 和 LayoutLMv3SelfOutput 实例
        self.self = LayoutLMv3SelfAttention(config)
        self.output = LayoutLMv3SelfOutput(config)

    # 前向传播方法，接受隐藏状态、注意力掩码、头掩码等参数
    # 返回注意力输出
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        rel_pos=None,
        rel_2d_pos=None,
    ):
        # 使用 self 层进行自注意力计算
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions,
            rel_pos=rel_pos,
            rel_2d_pos=rel_2d_pos,
        )
        # 使用 output 层进行最终输出计算
        attention_output = self.output(self_outputs[0], hidden_states)
        # 返回输出结果
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# 定义 LayoutLMv3Layer 类，继承自 nn.Module
class LayoutLMv3Layer(nn.Module):
    # 初始化方法，接受配置参数 config
    def __init__(self, config):
        super().__init__()
        # 设置前向传播过程中的超参数和模块
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = LayoutLMv3Attention(config)
        self.intermediate = LayoutLMv3Intermediate(config)
        self.output = LayoutLMv3Output(config)

    # 前向传播方法，接受隐藏状态、注意力掩码、头掩码等参数
    # 返回层输出结果
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        rel_pos=None,
        rel_2d_pos=None,
    ):
        # 使用 attention 层计算注意力输出
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            rel_pos=rel_pos,
            rel_2d_pos=rel_2d_pos,
        )
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # 对注意力输出进行分块处理
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # 返回层输出结果
        return outputs
    # 前向传播神经网络的一部分，处理注意力输出
    def feed_forward_chunk(self, attention_output):
        # 使用中间层处理注意力输出
        intermediate_output = self.intermediate(attention_output)
        # 使用输出层处理中间层输出和注意力输出，生成最终层输出
        layer_output = self.output(intermediate_output, attention_output)
        # 返回最终层输出
        return layer_output
class LayoutLMv3Encoder(nn.Module):
    # 定义 LayoutLMv3Encoder 类，继承自 nn.Module
    def __init__(self, config):
        # 初始化方法，接收参数 config
        super().__init__()
        # 调用父类 nn.Module 的初始化方法
        self.config = config
        # 将参数 config 存储在实例变量中
        self.layer = nn.ModuleList([LayoutLMv3Layer(config) for _ in range(config.num_hidden_layers)])
        # 创建包含多个 LayoutLMv3Layer 对象的 ModuleList，其中对象个数为 config.num_hidden_layers
        self.gradient_checkpointing = False
        # 初始化渐变检查点设置为 False

        self.has_relative_attention_bias = config.has_relative_attention_bias
        # 检查是否存在相对注意力偏差
        self.has_spatial_attention_bias = config.has_spatial_attention_bias
        # 检查是否存在空间注意力偏差

        if self.has_relative_attention_bias:
            # 如果存在相对注意力偏差
            self.rel_pos_bins = config.rel_pos_bins
            # 将相对位置的桶数存储在 rel_pos_bins 中
            self.max_rel_pos = config.max_rel_pos
            # 将最大相对位置存储在 max_rel_pos 中
            self.rel_pos_bias = nn.Linear(self.rel_pos_bins, config.num_attention_heads, bias=False)
            # 创建一个线性层，用于相对位置偏差的处理

        if self.has_spatial_attention_bias:
            # 如果存在空间注意力偏差
            self.max_rel_2d_pos = config.max_rel_2d_pos
            # 将最大二维相对位置存储在 max_rel_2d_pos 中
            self.rel_2d_pos_bins = config.rel_2d_pos_bins
            # 将二维相对位置的桶数存储在 rel_2d_pos_bins 中
            self.rel_pos_x_bias = nn.Linear(self.rel_2d_pos_bins, config.num_attention_heads, bias=False)
            # 创建一个线性层，用于 x 轴上的相对位置偏差的处理
            self.rel_pos_y_bias = nn.Linear(self.rel_2d_pos_bins, config.num_attention_heads, bias=False)
            # 创建一个线性层，用于 y 轴上的相对位置偏差的处理

    def relative_position_bucket(self, relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        # 定义一个处理相对位置的方法
        ret = 0
        if bidirectional:
            num_buckets //= 2
            ret += (relative_position > 0).long() * num_buckets
            n = torch.abs(relative_position)
        else:
            n = torch.max(-relative_position, torch.zeros_like(relative_position))
        # 处理相对位置，将其转换成正值

        # 现在 n 在范围 [0, inf) 内

        # 将桶的一半用于精确位置增量
        max_exact = num_buckets // 2
        is_small = n < max_exact

        # 另一半的桶用于到 max_distance 的位置的对数增长
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).to(torch.long)
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret
        # 返回处理后的结果

    def _cal_1d_pos_emb(self, position_ids):
        # 定义一个计算一维位置嵌入的方法，接收位置信息
        rel_pos_mat = position_ids.unsqueeze(-2) - position_ids.unsqueeze(-1)
        # 计算位置信息之间的差

        rel_pos = self.relative_position_bucket(
            rel_pos_mat,
            num_buckets=self.rel_pos_bins,
            max_distance=self.max_rel_pos,
        )
        # 使用相对位置信息计算处理后的相对位置

        rel_pos = self.rel_pos_bias.weight.t()[rel_pos].permute(0, 3, 1, 2)
        # 将相对位置信息转换为相对位置偏差权重

        rel_pos = rel_pos.contiguous()
        # 使得数据在内存中连续
        return rel_pos
        # 返回相对位置信息
    #
        # 如果output_hidden_states为True，则初始化一个空元组，否则初始化为None
        all_hidden_states = () if output_hidden_states else None
        # 如果output_attentions为True，则初始化一个空元组，否则初始化为None
        all_self_attentions = () if output_attentions else None

        # 如果模型有相对位置注意力偏置，则计算一维位置嵌入
        rel_pos = self._cal_1d_pos_emb(position_ids) if self.has_relative_attention_bias else None
        # 如果模型有空间注意力偏置，则计算二维位置嵌入
        rel_2d_pos = self._cal_2d_pos_emb(bbox) if self.has_spatial_attention_bias else None

        # 遍历所有的层次模块
        for i, layer_module in enumerate(self.layer):
            # 如果output_hidden_states为True，则将当前的hidden_states加入到all_hidden_states中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前层的头部掩码
            layer_head_mask = head_mask[i] if head_mask is not None else None

            # 如果启用梯度检查点并且在训练模式下，使用梯度检查点函数来计算当前层的输出
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    output_attentions,
                    rel_pos,
                    rel_2d_pos,
                )
            # 否则直接调用当前层的模块函数来计算输出
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    output_attentions,
                    rel_pos=rel_pos,
                    rel_2d_pos=rel_2d_pos,
                )

            # 更新当前的hidden_states为当前层输出的第一个元素
            hidden_states = layer_outputs[0]
            # 如果output_attentions为True，则将当前层输出的注意力加入到all_self_attentions中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 如果output_hidden_states为True，则将最终的hidden_states加入到all_hidden_states中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果return_dict为False，则返回包含非None元素的元组
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    all_hidden_states,
                    all_self_attentions,
                ]
                if v is not None
            )
        # 否则返回包含所有结果的BaseModelOutput对象
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
# 定义 LayoutLMv3Intermediate 类，继承自 nn.Module
class LayoutLMv3Intermediate(nn.Module):
    # 初始化函数，接受一个 config 对象
    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层，输入大小为 config 中的 hidden_size，输出大小为 config 中的 intermediate_size
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 判断 config.hidden_act 是否为字符串，如果是则使用名为 config.hidden_act 对应的激活函数，否则直接使用 config.hidden_act
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    # 前向传播函数，接收隐藏状态的张量，返回处理后的张量
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将输入的隐藏状态通过全连接层处理
        hidden_states = self.dense(hidden_states)
        # 应用激活函数处理隐藏状态
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 返回处理之后的隐藏状态
        return hidden_states


# 定义 LayoutLMv3Output 类，继承自 nn.Module
class LayoutLMv3Output(nn.Module):
    # 初始化函数，接受一个 config 对象
    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层，输入大小为 config 中的 intermediate_size，输出大小为 config 中的 hidden_size
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建一个 LayerNorm 层，对隐藏状态进行归一化处理
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个 Dropout 层，用于随机置零隐藏状态
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数，接收隐藏状态的张量和输入张量，返回处理后的张量
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将输入的隐藏状态通过全连接层处理
        hidden_states = self.dense(hidden_states)
        # 随机置零处理后的隐藏状态
        hidden_states = self.dropout(hidden_states)
        # 对处理后的隐藏状态进行 LayerNorm 处理，并与输入张量相加
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回处理之后的隐藏状态
        return hidden_states

# 定义 LayoutLMv3Model 类，继承自 LayoutLMv3PreTrainedModel
@add_start_docstrings(
    "The bare LayoutLMv3 Model transformer outputting raw hidden-states without any specific head on top.",
    LAYOUTLMV3_START_DOCSTRING,
)
class LayoutLMv3Model(LayoutLMv3PreTrainedModel):
    # 初始化函��，接受一个 config 对象
    def __init__(self, config):
        super().__init__(config)
        # 设置实例的 config 属性为传入的 config 对象
        self.config = config

        # 如果 config 中包含 text_embed，创建 LayoutLMv3TextEmbeddings 对象
        if config.text_embed:
            self.embeddings = LayoutLMv3TextEmbeddings(config)

        # 如果 config 中包含 visual_embed，创建 LayoutLMv3PatchEmbeddings 对象
        if config.visual_embed:
            # 使用默认的预训练参数进行微调，当输入大小在微调中较大时，在前向传播中将对位置嵌入进行插值
            self.patch_embed = LayoutLMv3PatchEmbeddings(config)

            # 计算 size，config.input_size 除以 config.patch_size 的整数部分
            size = int(config.input_size / config.patch_size)
            # 创建一个参数化的张量，用于表示分类令牌
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
            # 创建一个参数化的位置嵌入张量
            self.pos_embed = nn.Parameter(torch.zeros(1, size * size + 1, config.hidden_size))
            # 创建一个 Dropout 层，用于位置嵌入
            self.pos_drop = nn.Dropout(p=0.0)

            # 创建一个 LayerNorm 层，对隐藏状态进行归一化
            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            # 创建一个 Dropout 层，用于隐藏状态
            self.dropout = nn.Dropout(config.hidden_dropout_prob)

            # 如果 config 中包含相对注意力偏置或空间注意力偏置，初始化视觉边界框
            if self.config.has_relative_attention_bias or self.config.has_spatial_attention_bias:
                self.init_visual_bbox(image_size=(size, size))

            # 创建一个 LayerNorm 层，对隐藏状态进行归一化
            self.norm = nn.LayerNorm(config.hidden_size, eps=1e-6)

        # 创建 LayoutLMv3Encoder 对象
        self.encoder = LayoutLMv3Encoder(config)

        # 初始化模型参数
        self.init_weights()

    # 获取输入的嵌入
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    # 设置输入的嵌入
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value
    # 对模型的注意力头进行修剪
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        # 遍历需要修剪的层和对应的头部
        for layer, heads in heads_to_prune.items():
            # 调用编码器的注意力层对象的修剪方法
            self.encoder.layer[layer].attention.prune_heads(heads)

    # 初始化视觉（patch）标记的边界框
    def init_visual_bbox(self, image_size=(14, 14), max_len=1000):
        """
        Create the bounding boxes for the visual (patch) tokens.
        """
        # 计算视觉标记的边界框的 x 坐标
        visual_bbox_x = torch.div(
            torch.arange(0, max_len * (image_size[1] + 1), max_len), image_size[1], rounding_mode="trunc"
        )
        # 计算视觉标记的边界框的 y 坐标
        visual_bbox_y = torch.div(
            torch.arange(0, max_len * (image_size[0] + 1), max_len), image_size[0], rounding_mode="trunc"
        )
        # 创建视觉标记的边界框
        visual_bbox = torch.stack(
            [
                visual_bbox_x[:-1].repeat(image_size[0], 1),
                visual_bbox_y[:-1].repeat(image_size[1], 1).transpose(0, 1),
                visual_bbox_x[1:].repeat(image_size[0], 1),
                visual_bbox_y[1:].repeat(image_size[1], 1).transpose(0, 1),
            ],
            dim=-1,
        ).view(-1, 4)

        # 创建 [CLS] 标记的边界框
        cls_token_box = torch.tensor([[0 + 1, 0 + 1, max_len - 1, max_len - 1]])
        # 将 [CLS] 标记的边界框和视觉标记的边界框合并
        self.visual_bbox = torch.cat([cls_token_box, visual_bbox], dim=0)

    # 计算视觉标记的边界框
    def calculate_visual_bbox(self, device, dtype, batch_size):
        # 将视觉标记的边界框扩展到批次维度
        visual_bbox = self.visual_bbox.repeat(batch_size, 1, 1)
        # 将视觉标记的边界框移至指定设备并转换数据类型
        visual_bbox = visual_bbox.to(device).type(dtype)
        return visual_bbox

    # 前向传播函数，用于处理图像输入
    def forward_image(self, pixel_values):
        # 使用图像输入进行补丁嵌入
        embeddings = self.patch_embed(pixel_values)

        # 添加 [CLS] 标记
        batch_size, seq_len, _ = embeddings.size()
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # 添加位置嵌入
        if self.pos_embed is not None:
            embeddings = embeddings + self.pos_embed

        embeddings = self.pos_drop(embeddings)
        embeddings = self.norm(embeddings)

        return embeddings

    # 对模型进行前向传播
    @add_start_docstrings_to_model_forward(
        LAYOUTLMV3_MODEL_INPUTS_DOCSTRING.format("batch_size, token_sequence_length")
    )
    @replace_return_docstrings(output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        bbox: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
class LayoutLMv3ClassificationHead(nn.Module):
    """
    Head for sentence-level classification tasks. Reference: RobertaClassificationHead
    """

    def __init__(self, config, pool_feature=False):
        # 初始化函数，设置分类器头部的结构
        super().__init__()
        # 是否对特征进行池化
        self.pool_feature = pool_feature
        # 如果需要对特征进行池化，使用三倍于隐藏层大小的线性层
        if pool_feature:
            self.dense = nn.Linear(config.hidden_size * 3, config.hidden_size)
        else:
            # 否则使用隐藏层大小的线性层
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 分类器的丢弃率
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        # 添加丢弃层
        self.dropout = nn.Dropout(classifier_dropout)
        # 输出投影层
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x):
        # 前向传播函数
        x = self.dropout(x)
        x = self.dense(x)
        # 使用双曲正切激活函数
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        # 返回结果
        return x


@add_start_docstrings(
    """
    LayoutLMv3 Model with a token classification head on top (a linear layer on top of the final hidden states) e.g.
    for sequence labeling (information extraction) tasks such as [FUNSD](https://guillaumejaume.github.io/FUNSD/),
    [SROIE](https://rrc.cvc.uab.es/?ch=13), [CORD](https://github.com/clovaai/cord) and
    [Kleister-NDA](https://github.com/applicaai/kleister-nda).
    """,
    LAYOUTLMV3_START_DOCSTRING,
)
class LayoutLMv3ForTokenClassification(LayoutLMv3PreTrainedModel):
    def __init__(self, config):
        # 初始化函数
        super().__init__(config)
        # 标签数量
        self.num_labels = config.num_labels

        # 初始化 LayoutLMv3 模型
        self.layoutlmv3 = LayoutLMv3Model(config)
        # 添加丢弃层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 如果标签数量小于 10，则使用线性层作为分类器
        if config.num_labels < 10:
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        else:
            # 否则使用自定义的分类器头部
            self.classifier = LayoutLMv3ClassificationHead(config, pool_feature=False)

        # 初始化权重
        self.init_weights()

    @add_start_docstrings_to_model_forward(
        LAYOUTLMV3_DOWNSTREAM_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @replace_return_docstrings(output_type=TokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        bbox: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.LongTensor] = None,
    ):
        # 前向传播函数
        # 调用 LayoutLMv3 模型进行特征提取
        outputs = self.layoutlmv3(
            input_ids=input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            pixel_values=pixel_values,
        )
        # 获取特征
        sequence_output = outputs.last_hidden_state
        # 添加丢弃层
        sequence_output = self.dropout(sequence_output)
        # 如果标签数量小于 10，直接使用线性层作为分类器
        if self.num_labels < 10:
            logits = self.classifier(sequence_output)
        else:
            # 否则使用自定义的分类器头部
            logits = self.classifier(sequence_output)
        # 构造输出
        return TokenClassifierOutput(logits=logits)
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoProcessor, AutoModelForTokenClassification
        >>> from datasets import load_dataset

        >>> processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
        >>> model = AutoModelForTokenClassification.from_pretrained("microsoft/layoutlmv3-base", num_labels=7)

        >>> dataset = load_dataset("nielsr/funsd-layoutlmv3", split="train")
        >>> example = dataset[0]
        >>> image = example["image"]
        >>> words = example["tokens"]
        >>> boxes = example["bboxes"]
        >>> word_labels = example["ner_tags"]

        >>> encoding = processor(image, words, boxes=boxes, word_labels=word_labels, return_tensors="pt")

        >>> outputs = model(**encoding)
        >>> loss = outputs.loss
        >>> logits = outputs.logits
        ```"""
        # 确定返回结果是否以字典形式返回
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将输入传递给layoutlmv3模型
        outputs = self.layoutlmv3(
            input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            pixel_values=pixel_values,
        )
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        # 获取输入序列的长度
        seq_length = input_shape[1]
        # 仅获取输出表示的文本部分
        sequence_output = outputs[0][:, :seq_length]
        # 对输出进行dropout操作
        sequence_output = self.dropout(sequence_output)
        # 通过分类器获取logits
        logits = self.classifier(sequence_output)

        loss = None
        # 如果存在标签，则计算损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 如果不返回字典形式结果
        if not return_dict:
            # 输出结果包括logits和其他信息
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 返回TokenClassifierOutput对象
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 定义LayoutLMv3ForQuestionAnswering类，用于在提取性问题回答任务（例如DocVQA）中使用具有跨度分类头的LayoutLMv3模型
@add_start_docstrings(
    """
    LayoutLMv3 Model with a span classification head on top for extractive question-answering tasks such as
    [DocVQA](https://rrc.cvc.uab.es/?ch=17) (a linear layer on top of the text part of the hidden-states output to
    compute `span start logits` and `span end logits`).
    """,
    LAYOUTLMV3_START_DOCSTRING,
)
class LayoutLMv3ForQuestionAnswering(LayoutLMv3PreTrainedModel):
    # 初始化函数
    def __init__(self, config):
        # 调用LayoutLMv3PreTrainedModel的初始化函数
        super().__init__(config)
        # 存储标签数量
        self.num_labels = config.num_labels

        # 创建LayoutLMv3模型
        self.layoutlmv3 = LayoutLMv3Model(config)
        # 创建QA输出
        self.qa_outputs = LayoutLMv3ClassificationHead(config, pool_feature=False)

        # 初始化模型权重
        self.init_weights()

    # 前向传播函数
    @add_start_docstrings_to_model_forward(
        LAYOUTLMV3_DOWNSTREAM_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @replace_return_docstrings(output_type=QuestionAnsweringModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        start_positions: Optional[torch.LongTensor] = None,
        end_positions: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        bbox: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.LongTensor] = None,
# 定义LayoutLMv3ForSequenceClassification类，用于在文档图像分类任务（如RVL-CDIP数据集）中使用具有序列分类头的LayoutLMv3模型
@add_start_docstrings(
    """
    LayoutLMv3 Model with a sequence classification head on top (a linear layer on top of the final hidden state of the
    [CLS] token) e.g. for document image classification tasks such as the
    [RVL-CDIP](https://www.cs.cmu.edu/~aharley/rvl-cdip/) dataset.
    """,
    LAYOUTLMV3_START_DOCSTRING,
)
class LayoutLMv3ForSequenceClassification(LayoutLMv3PreTrainedModel):
    # 初始化函数
    def __init__(self, config):
        # 调用LayoutLMv3PreTrainedModel的初始化函数
        super().__init__(config)
        # 存储标签数量
        self.num_labels = config.num_labels
        # 存储配置
        self.config = config
        # 创建LayoutLMv3模型
        self.layoutlmv3 = LayoutLMv3Model(config)
        # 创建分类器
        self.classifier = LayoutLMv3ClassificationHead(config, pool_feature=False)

        # 初始化模型权重
        self.init_weights()

    # 前向传播函数
    @add_start_docstrings_to_model_forward(
        LAYOUTLMV3_DOWNSTREAM_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @replace_return_docstrings(output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    # 定义 forward 方法，用于模型的前向传播
    # 输入参数包括：输入的 token IDs，可选的注意力掩码，token 类型 IDs，位置 IDs，头部掩码，输入 embeddings，标签，是否输出注意力，是否输出隐藏状态，是否返回字典形式的输出，边框框信息，像素数值
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入的 token IDs，默认为 None
        attention_mask: Optional[torch.FloatTensor] = None,  # 注意力掩码，默认为 None
        token_type_ids: Optional[torch.LongTensor] = None,  # token 类型 IDs，默认为 None
        position_ids: Optional[torch.LongTensor] = None,  # 位置 IDs，默认为 None
        head_mask: Optional[torch.FloatTensor] = None,  # 头部掩码，默认为 None
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入 embeddings，默认为 None
        labels: Optional[torch.LongTensor] = None,  # 标签，默认为 None
        output_attentions: Optional[bool] = None,  # 是否输出注意力，默认为 None
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，默认为 None
        return_dict: Optional[bool] = None,  # 是否返回字典形式的输出，默认为 None
        bbox: Optional[torch.LongTensor] = None,  # 边框框信息，默认为 None
        pixel_values: Optional[torch.LongTensor] = None,  # 像素数值，默认为 None
```