# `.\models\layoutlmv3\modeling_layoutlmv3.py`

```
# 导入必要的模块和类
import collections
import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 导入与模型相关的输出类和工具函数
from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_layoutlmv3 import LayoutLMv3Config

# 获取日志记录器
logger = logging.get_logger(__name__)

# 定义配置文件字符串
_CONFIG_FOR_DOC = "LayoutLMv3Config"

# 预训练模型列表
LAYOUTLMV3_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "microsoft/layoutlmv3-base",
    "microsoft/layoutlmv3-large",
    # See all LayoutLMv3 models at https://huggingface.co/models?filter=layoutlmv3
]

# LayoutLMv3模型的起始文档字符串
LAYOUTLMV3_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`LayoutLMv3Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# LayoutLMv3模型输入的文档字符串
LAYOUTLMV3_MODEL_INPUTS_DOCSTRING = r"""
"""

# LayoutLMv3模型下游任务输入的文档字符串
LAYOUTLMV3_DOWNSTREAM_INPUTS_DOCSTRING = r"""
"""


class LayoutLMv3PatchEmbeddings(nn.Module):
    """LayoutLMv3 image (patch) embeddings. This class also automatically interpolates the position embeddings for varying
    image sizes."""
    # 初始化函数，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()

        # 根据配置对象确定输入图像的大小，若输入大小是可迭代对象则直接使用，否则将其作为宽高相同的元组
        image_size = (
            config.input_size
            if isinstance(config.input_size, collections.abc.Iterable)
            else (config.input_size, config.input_size)
        )
        # 根据配置对象确定图像的分块大小，同样处理方式
        patch_size = (
            config.patch_size
            if isinstance(config.patch_size, collections.abc.Iterable)
            else (config.patch_size, config.patch_size)
        )
        # 计算分块的形状，即图像在每个维度上分成多少个块
        self.patch_shape = (image_size[0] // patch_size[0], image_size[1] // patch_size[1])
        
        # 使用 nn.Conv2d 创建一个卷积层对象，将输入通道数转换为隐藏层的大小，使用指定的卷积核大小和步长
        self.proj = nn.Conv2d(config.num_channels, config.hidden_size, kernel_size=patch_size, stride=patch_size)

    # 前向传播函数，接收像素值和位置嵌入作为参数
    def forward(self, pixel_values, position_embedding=None):
        # 使用定义好的卷积层进行投影，得到特征嵌入
        embeddings = self.proj(pixel_values)

        # 如果位置嵌入不为 None，则进行插值以使其与特征嵌入的尺寸相匹配
        if position_embedding is not None:
            # 将位置嵌入重塑为与分块形状相匹配的形状
            position_embedding = position_embedding.view(1, self.patch_shape[0], self.patch_shape[1], -1)
            # 将通道维度移动到正确的位置
            position_embedding = position_embedding.permute(0, 3, 1, 2)
            # 获取特征嵌入的高度和宽度
            patch_height, patch_width = embeddings.shape[2], embeddings.shape[3]
            # 使用双三次插值方法将位置嵌入插值到特征嵌入的尺寸
            position_embedding = F.interpolate(position_embedding, size=(patch_height, patch_width), mode="bicubic")
            # 将插值后的位置嵌入与特征嵌入相加
            embeddings = embeddings + position_embedding

        # 将特征嵌入展平，然后交换维度以符合预期的输出格式
        embeddings = embeddings.flatten(2).transpose(1, 2)
        # 返回最终的特征嵌入
        return embeddings
    """
    LayoutLMv3 text embeddings. Same as `RobertaEmbeddings` but with added spatial (layout) embeddings.
    """

    def __init__(self, config):
        super().__init__()
        # 定义词嵌入层，将词汇索引映射为隐藏状态向量，支持填充标记
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 定义token类型嵌入层，将token类型索引映射为隐藏状态向量
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # LayerNorm层，对隐藏状态向量进行归一化
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Dropout层，用于随机失活以减少过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # 注册缓冲区，存储位置ID向量，用于序列化和内存连续访问
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )

        # 设置填充索引，用于位置嵌入层
        self.padding_idx = config.pad_token_id
        # 位置嵌入层，将位置索引映射为隐藏状态向量，支持填充标记
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

        # x坐标位置嵌入层，将x坐标索引映射为坐标大小向量
        self.x_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.coordinate_size)
        # y坐标位置嵌入层，将y坐标索引映射为坐标大小向量
        self.y_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.coordinate_size)
        # 高度位置嵌入层，将高度索引映射为形状大小向量
        self.h_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.shape_size)
        # 宽度位置嵌入层，将宽度索引映射为形状大小向量
        self.w_position_embeddings = nn.Embedding(config.max_2d_position_embeddings, config.shape_size)

    def calculate_spatial_position_embeddings(self, bbox):
        try:
            # 获取左侧位置嵌入向量，根据bbox的x坐标第一列
            left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
            # 获取上侧位置嵌入向量，根据bbox的y坐标第二列
            upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
            # 获取右侧位置嵌入向量，根据bbox的x坐标第三列
            right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
            # 获取下侧位置嵌入向量，根据bbox的y坐标第四列
            lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
        except IndexError as e:
            # 抛出错误，提示bbox坐标值应在0-1000范围内
            raise IndexError("The `bbox` coordinate values should be within 0-1000 range.") from e

        # 计算高度位置嵌入向量，根据bbox的上下坐标差，裁剪在0-1023范围内
        h_position_embeddings = self.h_position_embeddings(torch.clip(bbox[:, :, 3] - bbox[:, :, 1], 0, 1023))
        # 计算宽度位置嵌入向量，根据bbox的左右坐标差，裁剪在0-1023范围内
        w_position_embeddings = self.w_position_embeddings(torch.clip(bbox[:, :, 2] - bbox[:, :, 0], 0, 1023))

        # 合并所有空间位置嵌入向量为一个张量，按最后一个维度连接
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
        Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding
        symbols are ignored. This is modified from fairseq's `utils.make_positions`.
        """
        # 根据输入的token ids替换非填充符号为它们的位置编号。位置编号从padding_idx+1开始。填充符号被忽略。
        # 创建一个mask张量，标记非填充符号为1，填充符号为0
        mask = input_ids.ne(padding_idx).int()
        # 使用torch.cumsum沿着dim=1的维度累积求和，生成增量索引，再乘以mask，忽略填充符号
        incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask)) * mask
        # 返回增量索引加上padding_idx后的长整型张量
        return incremental_indices.long() + padding_idx

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.
        """
        # 从直接提供的嵌入中生成顺序的位置编号，无法推断哪些是填充的
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        # 使用torch.arange生成从self.padding_idx+1开始，到sequence_length + self.padding_idx + 1结束的长整型张量
        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        # 将position_ids张量的维度扩展为与inputs_embeds相同的形状
        return position_ids.unsqueeze(0).expand(input_shape)

    def forward(
        self,
        input_ids=None,
        bbox=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
    ):
        if position_ids is None:
            if input_ids is not None:
                # 如果未提供位置ids，且提供了input_ids，则从input_ids创建位置ids，保留填充的token
                position_ids = self.create_position_ids_from_input_ids(input_ids, self.padding_idx).to(
                    input_ids.device
                )
            else:
                # 如果未提供位置ids且未提供input_ids，则从inputs_embeds创建位置ids
                position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        if token_type_ids is None:
            # 如果未提供token_type_ids，则创建一个形状与input_shape相同的零张量
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            # 如果未提供inputs_embeds，则使用self.word_embeddings从input_ids获取嵌入
            inputs_embeds = self.word_embeddings(input_ids)
        # 使用self.token_type_embeddings获取token_type_ids的嵌入
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 将输入嵌入与token类型嵌入相加
        embeddings = inputs_embeds + token_type_embeddings
        # 使用self.position_embeddings获取位置ids的嵌入
        position_embeddings = self.position_embeddings(position_ids)
        # 将位置嵌入与之前的嵌入相加
        embeddings += position_embeddings

        # 计算空间位置嵌入
        spatial_position_embeddings = self.calculate_spatial_position_embeddings(bbox)

        # 将空间位置嵌入与之前的嵌入相加
        embeddings = embeddings + spatial_position_embeddings

        # 使用LayerNorm对嵌入进行归一化处理
        embeddings = self.LayerNorm(embeddings)
        # 使用dropout对嵌入进行随机失活处理
        embeddings = self.dropout(embeddings)
        # 返回处理后的嵌入张量
        return embeddings
# LayoutLMv3PreTrainedModel 类定义，继承自 PreTrainedModel，用于处理权重初始化和预训练模型的简单接口
class LayoutLMv3PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 指定配置类为 LayoutLMv3Config
    config_class = LayoutLMv3Config
    # 基础模型前缀为 "layoutlmv3"
    base_model_prefix = "layoutlmv3"

    # 初始化模型权重的方法
    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果 module 是 nn.Linear 或 nn.Conv2d 类型
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 使用正态分布初始化权重，均值为 0.0，标准差为配置中的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在偏置项，将其初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果 module 是 nn.Embedding 类型
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化权重，均值为 0.0，标准差为配置中的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果设置了 padding_idx，将对应位置的权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # 如果 module 是 nn.LayerNorm 类型
        elif isinstance(module, nn.LayerNorm):
            # 将偏置项初始化为零
            module.bias.data.zero_()
            # 将权重初始化为 1.0
            module.weight.data.fill_(1.0)


# LayoutLMv3SelfAttention 类定义，继承自 nn.Module，实现自注意力机制
class LayoutLMv3SelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 如果 hidden_size 不能被 num_attention_heads 整除，并且没有 embedding_size 属性，则抛出 ValueError
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        # 设置注意力头的数量和每个头的尺寸
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 定义查询、键、值的线性变换层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # Dropout 层，用于注意力概率的随机丢弃
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        # 是否具有相对注意力偏置和空间注意力偏置
        self.has_relative_attention_bias = config.has_relative_attention_bias
        self.has_spatial_attention_bias = config.has_spatial_attention_bias

    def transpose_for_scores(self, x):
        # 调整 x 的形状以便计算注意力分数，从 [batch_size, seq_length, hidden_size] 转换为 [batch_size, num_attention_heads, seq_length, attention_head_size]
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    def cogview_attention(self, attention_scores, alpha=32):
        """
        https://arxiv.org/abs/2105.13290 Section 2.4 Stabilization of training: Precision Bottleneck Relaxation
        (PB-Relax). A replacement of the original nn.Softmax(dim=-1)(attention_scores). Seems the new attention_probs
        will result in a slower speed and a little bias. Can use torch.allclose(standard_attention_probs,
        cogview_attention_probs, atol=1e-08) for comparison. The smaller atol (e.g., 1e-08), the better.
        """
        # 缩放注意力分数，通过除以 alpha
        scaled_attention_scores = attention_scores / alpha
        # 计算每行中的最大值，并扩展维度以便广播
        max_value = scaled_attention_scores.amax(dim=(-1)).unsqueeze(-1)
        # 应用 PB-Relax 算法，调整注意力分数
        new_attention_scores = (scaled_attention_scores - max_value) * alpha
        # 应用 softmax 函数来获得新的注意力概率分布
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
        # 使用 self.query 对隐藏状态进行查询，获取混合查询层
        mixed_query_layer = self.query(hidden_states)

        # 使用 self.key 对隐藏状态进行键的转换，并调用 self.transpose_for_scores 进行进一步处理
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        
        # 使用 self.value 对隐藏状态进行值的转换，并调用 self.transpose_for_scores 进行进一步处理
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # 对混合查询层进行转置处理
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # 计算注意力分数，采用点积 "query" 和 "key" 得到原始注意力分数
        # 注意力分数 QT K/√d 可能明显大于输入元素，并导致溢出。
        # 将计算顺序修改为 QT(K/√d) 可以缓解这个问题。详见：https://arxiv.org/pdf/2105.13290.pdf
        attention_scores = torch.matmul(query_layer / math.sqrt(self.attention_head_size), key_layer.transpose(-1, -2))

        # 如果存在相对注意力偏置和空间注意力偏置
        if self.has_relative_attention_bias and self.has_spatial_attention_bias:
            # 添加相对位置偏置和二维相对位置偏置
            attention_scores += (rel_pos + rel_2d_pos) / math.sqrt(self.attention_head_size)
        elif self.has_relative_attention_bias:
            # 添加相对位置偏置
            attention_scores += rel_pos / math.sqrt(self.attention_head_size)

        # 如果存在注意力遮罩
        if attention_mask is not None:
            # 应用预先计算的注意力遮罩（在 RobertaModel 的 forward() 函数中进行预计算）
            attention_scores = attention_scores + attention_mask

        # 将注意力分数归一化为概率
        attention_probs = self.cogview_attention(attention_scores)

        # 使用 dropout 进行注意力概率的随机失活
        # 这实际上是删除整个待注意的令牌，这可能看起来有些不寻常，但源自原始 Transformer 论文。
        attention_probs = self.dropout(attention_probs)

        # 如果存在头部掩码
        if head_mask is not None:
            # 应用头部掩码
            attention_probs = attention_probs * head_mask

        # 计算上下文层，将注意力概率与值层进行矩阵相乘
        context_layer = torch.matmul(attention_probs, value_layer)

        # 对上下文层进行维度变换和重整，以适应全头大小
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # 如果需要输出注意力信息，则返回上下文层和注意力概率；否则仅返回上下文层
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        # 返回计算结果
        return outputs
# Copied from transformers.models.roberta.modeling_roberta.RobertaSelfOutput
class LayoutLMv3SelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)  # 初始化 LayerNorm 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # 初始化 Dropout 层

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)  # 前向传播：全连接层
        hidden_states = self.dropout(hidden_states)  # 前向传播：应用 Dropout
        hidden_states = self.LayerNorm(hidden_states + input_tensor)  # 前向传播：应用 LayerNorm
        return hidden_states


# Copied from transformers.models.layoutlmv2.modeling_layoutlmv2.LayoutLMv2Attention with LayoutLMv2->LayoutLMv3
class LayoutLMv3Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = LayoutLMv3SelfAttention(config)  # 初始化自注意力层
        self.output = LayoutLMv3SelfOutput(config)  # 初始化自注意力输出层

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        rel_pos=None,
        rel_2d_pos=None,
    ):
        self_outputs = self.self(  # 调用自注意力层
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions,
            rel_pos=rel_pos,
            rel_2d_pos=rel_2d_pos,
        )
        attention_output = self.output(self_outputs[0], hidden_states)  # 应用自注意力输出层
        outputs = (attention_output,) + self_outputs[1:]  # 如果需要输出注意力权重，则添加到输出中
        return outputs


# Copied from transformers.models.layoutlmv2.modeling_layoutlmv2.LayoutLMv2Layer with LayoutLMv2->LayoutLMv3
class LayoutLMv3Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = LayoutLMv3Attention(config)  # 初始化注意力层
        self.intermediate = LayoutLMv3Intermediate(config)  # 初始化中间层
        self.output = LayoutLMv3Output(config)  # 初始化输出层

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        rel_pos=None,
        rel_2d_pos=None,
    ):
        self_attention_outputs = self.attention(  # 调用注意力层
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            rel_pos=rel_pos,
            rel_2d_pos=rel_2d_pos,
        )
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:]  # 如果需要输出注意力权重，则添加到输出中

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )  # 应用分块机制进行前向传播
        outputs = (layer_output,) + outputs

        return outputs
    # 定义神经网络的前向传播方法，处理注意力输出
    def feed_forward_chunk(self, attention_output):
        # 使用神经网络的中间层处理注意力输出，得到中间输出
        intermediate_output = self.intermediate(attention_output)
        # 使用神经网络的输出层处理中间输出和注意力输出，得到最终层输出
        layer_output = self.output(intermediate_output, attention_output)
        # 返回最终的层输出作为本次前向传播的结果
        return layer_output
    # LayoutLMv3Encoder 类的构造函数，接收一个配置对象 config 作为参数
    def __init__(self, config):
        # 调用父类 nn.Module 的构造函数
        super().__init__()
        # 将传入的配置对象保存到实例变量中
        self.config = config
        # 创建一个包含多个 LayoutLMv3Layer 对象的模块列表，列表长度为 config.num_hidden_layers
        self.layer = nn.ModuleList([LayoutLMv3Layer(config) for _ in range(config.num_hidden_layers)])
        # 初始化梯度检查点标志为 False
        self.gradient_checkpointing = False

        # 检查是否有相对注意力偏置
        self.has_relative_attention_bias = config.has_relative_attention_bias
        # 检查是否有空间注意力偏置
        self.has_spatial_attention_bias = config.has_spatial_attention_bias

        # 如果有相对注意力偏置，则初始化相关变量
        if self.has_relative_attention_bias:
            # 从配置中获取相对位置的桶数和最大相对位置
            self.rel_pos_bins = config.rel_pos_bins
            self.max_rel_pos = config.max_rel_pos
            # 创建一个线性层，用于计算相对位置偏置
            self.rel_pos_bias = nn.Linear(self.rel_pos_bins, config.num_attention_heads, bias=False)

        # 如果有空间注意力偏置，则初始化相关变量
        if self.has_spatial_attention_bias:
            # 从配置中获取二维相对位置的最大值和桶数
            self.max_rel_2d_pos = config.max_rel_2d_pos
            self.rel_2d_pos_bins = config.rel_2d_pos_bins
            # 创建两个线性层，分别用于计算二维相对位置的 x 和 y 偏置
            self.rel_pos_x_bias = nn.Linear(self.rel_2d_pos_bins, config.num_attention_heads, bias=False)
            self.rel_pos_y_bias = nn.Linear(self.rel_2d_pos_bins, config.num_attention_heads, bias=False)

    # 计算相对位置桶的函数
    def relative_position_bucket(self, relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        # 初始化返回值为 0
        ret = 0
        # 如果是双向的，则将桶数减半，并根据相对位置的正负来选择相应的偏置
        if bidirectional:
            num_buckets //= 2
            ret += (relative_position > 0).long() * num_buckets
            n = torch.abs(relative_position)
        else:
            n = torch.max(-relative_position, torch.zeros_like(relative_position))
        # 现在 n 的范围是 [0, inf)

        # 桶的一半用于精确增量的位置
        max_exact = num_buckets // 2
        is_small = n < max_exact

        # 另一半桶用于位置在 max_distance 范围内的对数增大的位置
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).to(torch.long)
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        # 根据 is_small 条件选择使用 n 还是 val_if_large
        ret += torch.where(is_small, n, val_if_large)
        return ret

    # 计算一维位置嵌入的函数
    def _cal_1d_pos_emb(self, position_ids):
        # 计算位置 id 之间的相对位置矩阵
        rel_pos_mat = position_ids.unsqueeze(-2) - position_ids.unsqueeze(-1)

        # 使用 relative_position_bucket 函数计算相对位置的桶
        rel_pos = self.relative_position_bucket(
            rel_pos_mat,
            num_buckets=self.rel_pos_bins,
            max_distance=self.max_rel_pos,
        )
        # 获取相对位置偏置并进行相应的形状转换
        rel_pos = self.rel_pos_bias.weight.t()[rel_pos].permute(0, 3, 1, 2)
        rel_pos = rel_pos.contiguous()
        return rel_pos
    # 计算二维位置编码的方法，根据给定的边界框数据计算得到
    def _cal_2d_pos_emb(self, bbox):
        # 提取边界框中的 x 坐标和 y 坐标信息
        position_coord_x = bbox[:, :, 0]
        position_coord_y = bbox[:, :, 3]
        # 计算 x 方向和 y 方向上的相对位置矩阵
        rel_pos_x_2d_mat = position_coord_x.unsqueeze(-2) - position_coord_x.unsqueeze(-1)
        rel_pos_y_2d_mat = position_coord_y.unsqueeze(-2) - position_coord_y.unsqueeze(-1)
        # 使用预定义的函数对相对位置矩阵进行桶化处理，得到相对位置编码
        rel_pos_x = self.relative_position_bucket(
            rel_pos_x_2d_mat,
            num_buckets=self.rel_2d_pos_bins,
            max_distance=self.max_rel_2d_pos,
        )
        rel_pos_y = self.relative_position_bucket(
            rel_pos_y_2d_mat,
            num_buckets=self.rel_2d_pos_bins,
            max_distance=self.max_rel_2d_pos,
        )
        # 从权重矩阵中获取 x 和 y 方向上的位置偏置，并进行维度调整
        rel_pos_x = self.rel_pos_x_bias.weight.t()[rel_pos_x].permute(0, 3, 1, 2)
        rel_pos_y = self.rel_pos_y_bias.weight.t()[rel_pos_y].permute(0, 3, 1, 2)
        # 确保数据的连续性
        rel_pos_x = rel_pos_x.contiguous()
        rel_pos_y = rel_pos_y.contiguous()
        # 计算得到最终的二维位置编码
        rel_2d_pos = rel_pos_x + rel_pos_y
        return rel_2d_pos

    # 前向传播方法，用于模型的计算过程
    def forward(
        self,
        hidden_states,
        bbox=None,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        position_ids=None,
        patch_height=None,
        patch_width=None,
        ):
            # 如果设置了输出隐藏状态，初始化一个空元组；否则置为None
            all_hidden_states = () if output_hidden_states else None
            # 如果设置了输出注意力权重，初始化一个空元组；否则置为None
            all_self_attentions = () if output_attentions else None

            # 如果模型支持相对位置注意力偏置，则计算一维位置编码
            rel_pos = self._cal_1d_pos_emb(position_ids) if self.has_relative_attention_bias else None
            # 如果模型支持空间注意力偏置，则计算二维位置编码
            rel_2d_pos = self._cal_2d_pos_emb(bbox) if self.has_spatial_attention_bias else None

            # 遍历每一个层模块
            for i, layer_module in enumerate(self.layer):
                # 如果需要输出隐藏状态，将当前隐藏状态加入到all_hidden_states中
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                # 获取当前层的头部掩码
                layer_head_mask = head_mask[i] if head_mask is not None else None

                # 如果启用了梯度检查点且在训练阶段，则使用梯度检查点函数执行当前层的前向传播
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
                else:
                    # 否则，直接调用当前层的前向传播函数
                    layer_outputs = layer_module(
                        hidden_states,
                        attention_mask,
                        layer_head_mask,
                        output_attentions,
                        rel_pos=rel_pos,
                        rel_2d_pos=rel_2d_pos,
                    )

                # 更新隐藏状态为当前层的输出的第一个元素
                hidden_states = layer_outputs[0]
                # 如果需要输出注意力权重，将当前层的注意力权重加入到all_self_attentions中
                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)

            # 如果需要输出隐藏状态，将最终的隐藏状态加入到all_hidden_states中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 如果不需要返回字典形式的结果，返回一个元组，包括hidden_states, all_hidden_states和all_self_attentions中非None的部分
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
            # 否则，返回一个BaseModelOutput对象，包含最终的隐藏状态、所有隐藏状态和所有注意力权重
            return BaseModelOutput(
                last_hidden_state=hidden_states,
                hidden_states=all_hidden_states,
                attentions=all_self_attentions,
            )
# Copied from transformers.models.roberta.modeling_roberta.RobertaIntermediate
class LayoutLMv3Intermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 定义一个全连接层，输入大小为 config.hidden_size，输出大小为 config.intermediate_size
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 根据配置选择隐藏层激活函数，将字符串类型的激活函数映射到对应的函数上
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 输入 hidden_states 经过全连接层 dense
        hidden_states = self.dense(hidden_states)
        # 再经过选择的隐藏层激活函数 intermediate_act_fn
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.roberta.modeling_roberta.RobertaOutput
class LayoutLMv3Output(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 定义一个全连接层，输入大小为 config.intermediate_size，输出大小为 config.hidden_size
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # LayerNorm 层，对隐藏状态进行归一化，大小为 config.hidden_size，eps 为 config.layer_norm_eps
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Dropout 层，以 config.hidden_dropout_prob 的概率进行 dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 输入 hidden_states 经过全连接层 dense
        hidden_states = self.dense(hidden_states)
        # 经过 dropout
        hidden_states = self.dropout(hidden_states)
        # 输入 hidden_states 与 input_tensor 相加，然后经过 LayerNorm
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


@add_start_docstrings(
    "The bare LayoutLMv3 Model transformer outputting raw hidden-states without any specific head on top.",
    LAYOUTLMV3_START_DOCSTRING,
)
class LayoutLMv3Model(LayoutLMv3PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # 如果配置中包含文本嵌入层，则使用 LayoutLMv3TextEmbeddings 初始化 embeddings 属性
        if config.text_embed:
            self.embeddings = LayoutLMv3TextEmbeddings(config)

        # 如果配置中包含视觉嵌入层，则使用 LayoutLMv3PatchEmbeddings 初始化 patch_embed 属性
        if config.visual_embed:
            # 当输入大小较大时，根据 fine-tuning 参数调整位置嵌入，在前向传播时进行插值
            self.patch_embed = LayoutLMv3PatchEmbeddings(config)

            size = int(config.input_size / config.patch_size)
            # 初始化 cls_token 为一个全零的可训练参数
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
            # 初始化位置嵌入为一个全零的可训练参数
            self.pos_embed = nn.Parameter(torch.zeros(1, size * size + 1, config.hidden_size))
            # 初始化 pos_drop 为一个 dropout 层，概率为 0.0
            self.pos_drop = nn.Dropout(p=0.0)

            # 初始化 LayerNorm 层，对隐藏状态进行归一化，大小为 config.hidden_size，eps 为 config.layer_norm_eps
            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            # 初始化 dropout 层，以 config.hidden_dropout_prob 的概率进行 dropout
            self.dropout = nn.Dropout(config.hidden_dropout_prob)

            # 如果配置中包含相对注意力偏置或空间注意力偏置，则初始化视觉边界框
            if self.config.has_relative_attention_bias or self.config.has_spatial_attention_bias:
                self.init_visual_bbox(image_size=(size, size))

            # 初始化 norm 层，对隐藏状态进行归一化，大小为 config.hidden_size，eps 为 1e-6
            self.norm = nn.LayerNorm(config.hidden_size, eps=1e-6)

        # 使用 LayoutLMv3Encoder 初始化 encoder 属性
        self.encoder = LayoutLMv3Encoder(config)

        # 初始化模型权重
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel
        """
        # 遍历需要修剪的层和对应要修剪的注意力头
        for layer, heads in heads_to_prune.items():
            # 调用编码器中指定层的注意力模型的修剪方法
            self.encoder.layer[layer].attention.prune_heads(heads)

    def init_visual_bbox(self, image_size=(14, 14), max_len=1000):
        """
        Create the bounding boxes for the visual (patch) tokens.
        """
        # 计算视觉（patch）标记的边界框
        visual_bbox_x = torch.div(
            torch.arange(0, max_len * (image_size[1] + 1), max_len), image_size[1], rounding_mode="trunc"
        )
        visual_bbox_y = torch.div(
            torch.arange(0, max_len * (image_size[0] + 1), max_len), image_size[0], rounding_mode="trunc"
        )
        visual_bbox = torch.stack(
            [
                visual_bbox_x[:-1].repeat(image_size[0], 1),
                visual_bbox_y[:-1].repeat(image_size[1], 1).transpose(0, 1),
                visual_bbox_x[1:].repeat(image_size[0], 1),
                visual_bbox_y[1:].repeat(image_size[1], 1).transpose(0, 1),
            ],
            dim=-1,
        ).view(-1, 4)

        cls_token_box = torch.tensor([[0 + 1, 0 + 1, max_len - 1, max_len - 1]])
        # 将 [CLS] 标记的边界框与视觉标记的边界框连接起来
        self.visual_bbox = torch.cat([cls_token_box, visual_bbox], dim=0)

    def calculate_visual_bbox(self, device, dtype, batch_size):
        # 将视觉边界框重复扩展到批次中每个样本
        visual_bbox = self.visual_bbox.repeat(batch_size, 1, 1)
        # 将视觉边界框移动到指定设备上，并指定数据类型
        visual_bbox = visual_bbox.to(device).type(dtype)
        return visual_bbox

    def forward_image(self, pixel_values):
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
    ):
        """
        Override of the forward method in the parent class with specific
        docstrings added for layoutlmv3 model inputs and outputs.
        """
class LayoutLMv3ClassificationHead(nn.Module):
    """
    Head for sentence-level classification tasks. Reference: RobertaClassificationHead
    """

    def __init__(self, config, pool_feature=False):
        super().__init__()
        self.pool_feature = pool_feature
        if pool_feature:
            # 如果需要池化特征，则使用三倍的隐藏状态大小作为输入维度进行线性变换
            self.dense = nn.Linear(config.hidden_size * 3, config.hidden_size)
        else:
            # 否则使用隐藏状态大小作为输入维度进行线性变换
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 根据配置设置分类器的丢弃率，如果未指定，则使用隐藏层丢弃率
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        # 最终的线性变换层，输出维度为标签数
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x):
        # 应用丢弃操作
        x = self.dropout(x)
        # 应用线性变换
        x = self.dense(x)
        # 应用双曲正切激活函数
        x = torch.tanh(x)
        # 再次应用丢弃操作
        x = self.dropout(x)
        # 应用最终的线性变换，得到分类结果
        x = self.out_proj(x)
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
        super().__init__(config)
        self.num_labels = config.num_labels

        # 初始化 LayoutLMv3 模型
        self.layoutlmv3 = LayoutLMv3Model(config)
        # 应用丢弃操作到隐藏层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 根据标签数确定分类器的类型
        if config.num_labels < 10:
            # 如果标签数小于10，使用简单的线性分类器
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        else:
            # 否则使用复杂的分类头 LayoutLMv3ClassificationHead
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
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 根据传入的参数确定是否使用返回字典，若未指定则使用模型配置中的默认设置

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
        # 调用LayoutLMv3模型进行前向传播，传入各种输入参数

        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]
        # 根据输入的数据类型，确定输入形状的大小

        seq_length = input_shape[1]
        # 获取序列长度

        # 只取输出表示的文本部分
        sequence_output = outputs[0][:, :seq_length]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        # 对序列输出进行分类器操作，生成最终的逻辑回归输出

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        # 如果提供了标签，计算分类损失

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
        # 如果不使用返回字典，则返回一个元组

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        # 如果使用返回字典，则返回TokenClassifierOutput对象，包含损失、逻辑回归输出、隐藏状态和注意力权重
"""
LayoutLMv3 Model with a span classification head on top for extractive question-answering tasks such as
[DocVQA](https://rrc.cvc.uab.es/?ch=17) (a linear layer on top of the text part of the hidden-states output to
compute `span start logits` and `span end logits`).
"""
# 定义 LayoutLMv3 问题回答模型，包含用于抽取式问答任务的跨度分类头部
# 例如 [DocVQA](https://rrc.cvc.uab.es/?ch=17)，在隐藏状态输出的文本部分之上使用线性层计算 `span start logits` 和 `span end logits`
@add_start_docstrings(
    """
    LayoutLMv3 Model with a sequence classification head on top (a linear layer on top of the final hidden state of the
    [CLS] token) e.g. for document image classification tasks such as the
    [RVL-CDIP](https://www.cs.cmu.edu/~aharley/rvl-cdip/) dataset.
    """,
    LAYOUTLMV3_START_DOCSTRING,
)
# 定义 LayoutLMv3 序列分类模型，包含在顶部的序列分类头部
# 例如用于文档图像分类任务如 [RVL-CDIP](https://www.cs.cmu.edu/~aharley/rvl-cdip/) 数据集的线性层
class LayoutLMv3ForSequenceClassification(LayoutLMv3PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.layoutlmv3 = LayoutLMv3Model(config)  # 初始化 LayoutLMv3 模型
        self.classifier = LayoutLMv3ClassificationHead(config, pool_feature=False)  # 初始化分类头部

        self.init_weights()  # 初始化模型权重

    @add_start_docstrings_to_model_forward(
        LAYOUTLMV3_DOWNSTREAM_INPUTS_DOCSTRING.format("batch_size, sequence_length")
    )
    @replace_return_docstrings(output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    # 为模型的 forward 方法添加文档字符串，描述其输入和输出
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
    # 定义前向传播方法，用于模型的正向推断
    def forward(
        self,
        # 输入序列的 token IDs，类型为长整型张量，可选
        input_ids: Optional[torch.LongTensor] = None,
        # 注意力掩码张量，类型为浮点数张量，可选
        attention_mask: Optional[torch.FloatTensor] = None,
        # 分段 token IDs，类型为长整型张量，可选
        token_type_ids: Optional[torch.LongTensor] = None,
        # 位置 IDs，类型为长整型张量，可选
        position_ids: Optional[torch.LongTensor] = None,
        # 头部掩码，类型为浮点数张量，可选
        head_mask: Optional[torch.FloatTensor] = None,
        # 输入嵌入张量，类型为浮点数张量，可选
        inputs_embeds: Optional[torch.FloatTensor] = None,
        # 标签，类型为长整型张量，可选
        labels: Optional[torch.LongTensor] = None,
        # 是否输出注意力权重，类型为布尔值，可选
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态，类型为布尔值，可选
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典格式的输出，类型为布尔值，可选
        return_dict: Optional[bool] = None,
        # 包围框信息，类型为长整型张量，可选
        bbox: Optional[torch.LongTensor] = None,
        # 像素数值信息，类型为长整型张量，可选
        pixel_values: Optional[torch.LongTensor] = None,
```