# `.\models\vilt\modeling_vilt.py`

```
# 设置文件编码为 UTF-8
# 版权声明
# 版权所有 2022 年 NAVER AI Labs 和 HuggingFace Inc. 团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）许可；
# 除非符合许可证的规定，否则不得使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则软件
# 根据“原样”分发，不附带任何明示或暗示的担保或条件。
# 有关详细信息，请参阅许可证。
""" PyTorch ViLT 模型。"""

import collections.abc  # 导入抽象基类集合
import math  # 导入数学库
from dataclasses import dataclass  # 导入数据类装饰器
from typing import List, Optional, Tuple, Union  # 导入类型提示

import torch  # 导入 PyTorch
import torch.utils.checkpoint  # 导入 PyTorch 检查点工具
from torch import nn  # 导入 PyTorch 神经网络模块
from torch.nn import CrossEntropyLoss  # 导入交叉熵损失函数

from ...activations import ACT2FN  # 导入激活函数映射
from ...modeling_outputs import (  # 导入模型输出类
    BaseModelOutput,
    BaseModelOutputWithPooling,
    MaskedLMOutput,
    ModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from ...modeling_utils import PreTrainedModel  # 导入预训练模型工具
from ...pytorch_utils import (  # 导入 PyTorch 工具函数
    find_pruneable_heads_and_indices,
    meshgrid,
    prune_linear_layer,
)
from ...utils import (  # 导入工具函数
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_vilt import ViltConfig  # 导入 ViLT 配置

logger = logging.get_logger(__name__)  # 获取日志记录器

_CONFIG_FOR_DOC = "ViltConfig"  # 用于文档的配置类名
_CHECKPOINT_FOR_DOC = "dandelin/vilt-b32-mlm"  # 用于文档的检查点名称

VILT_PRETRAINED_MODEL_ARCHIVE_LIST = [  # ViLT 预训练模型存档列表
    "dandelin/vilt-b32-mlm",
    # 查看所有 ViLT 模型 https://huggingface.co/models?filter=vilt
]


@dataclass
class ViltForImagesAndTextClassificationOutput(ModelOutput):
    """
    [`ViltForImagesAndTextClassification`] 的输出类。
    """
    # 定义函数参数和返回类型的文档字符串，描述了该函数可以接受的参数和可能的返回值类型
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            分类（如果config.num_labels==1，则为回归）损失值。
        logits (`torch.FloatTensor` of shape `(batch_size, config.num_labels)`):
            分类（如果config.num_labels==1，则为回归）得分（SoftMax之前）。
        hidden_states (`List[tuple(torch.FloatTensor)]`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            `torch.FloatTensor`的列表（每个图像-文本对一个，每个元组包含嵌入的输出+每层输出）的元组，形状为`(batch_size, sequence_length, hidden_size)`。
            模型在每一层输出的隐藏状态加上初始嵌入的输出。
        attentions (`List[tuple(torch.FloatTensor)]`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            `torch.FloatTensor`的列表（每个图像-文本对一个，每个元组包含注意力权重的输出）的元组，形状为`(batch_size, num_heads, sequence_length, sequence_length)`。
            注意力softmax之后的注意力权重，用于计算自注意力头中的加权平均值。
    """

    # 定义可能为None的损失值变量，类型为`torch.FloatTensor`
    loss: Optional[torch.FloatTensor] = None
    # 定义必须存在的logits变量，类型为`torch.FloatTensor`
    logits: torch.FloatTensor = None
    # 定义可能为None的隐藏状态列表变量，每个元素为`torch.FloatTensor`的元组列表
    hidden_states: Optional[List[Tuple[torch.FloatTensor]]] = None
    # 定义可能为None的注意力权重列表变量，每个元素为`torch.FloatTensor`的元组列表
    attentions: Optional[List[Tuple[torch.FloatTensor]]] = None
class ViltEmbeddings(nn.Module):
    """
    Construct the text and patch embeddings.

    Text embeddings are equivalent to BERT embeddings.

    Patch embeddings are equivalent to ViT embeddings.
    """

    def __init__(self, config):
        super().__init__()

        # text embeddings
        self.text_embeddings = TextEmbeddings(config)
        # patch embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.patch_embeddings = ViltPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, config.hidden_size))
        # modality type (text/patch) embeddings
        self.token_type_embeddings = nn.Embedding(config.modality_type_vocab_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.config = config

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        pixel_values,
        pixel_mask,
        inputs_embeds,
        image_embeds,
        image_token_type_idx=1,
    ):
        # PART 1: text embeddings
        text_embeds = self.text_embeddings(
            input_ids=input_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )

        # PART 2: patch embeddings (with interpolated position encodings)
        if image_embeds is None:
            # Generate visual embeddings and masks from pixel values
            image_embeds, image_masks, patch_index = self.visual_embed(
                pixel_values, pixel_mask, max_image_length=self.config.max_image_length
            )
        else:
            # Flatten pixel masks
            image_masks = pixel_mask.flatten(1)

        # PART 3: add modality type embeddings
        # 0 indicates text, 1 indicates image, 2 is optionally used when a second image is provided (NLVR2)
        if image_token_type_idx is None:
            image_token_type_idx = 1
        # Add token type embeddings to text embeddings
        text_embeds = text_embeds + self.token_type_embeddings(
            torch.zeros_like(attention_mask, dtype=torch.long, device=text_embeds.device)
        )
        # Add token type embeddings to image embeddings
        image_embeds = image_embeds + self.token_type_embeddings(
            torch.full_like(image_masks, image_token_type_idx, dtype=torch.long, device=text_embeds.device)
        )

        # PART 4: concatenate text and image embeddings
        embeddings = torch.cat([text_embeds, image_embeds], dim=1)
        # Concatenate attention masks and image masks
        masks = torch.cat([attention_mask, image_masks], dim=1)

        return embeddings, masks
    # 初始化函数，接受一个配置参数 config
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个词嵌入层，用于将词汇索引映射为隐藏状态向量
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 创建一个位置嵌入层，用于将位置索引映射为隐藏状态向量
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # 创建一个标记类型嵌入层，用于将标记类型索引映射为隐藏状态向量
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # 创建一个 LayerNorm 层，用于标准化隐藏状态向量
        # 注意：这里 LayerNorm 的命名方式与 TensorFlow 的模型变量保持一致，以便能够加载任何 TensorFlow 的检查点文件
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个 Dropout 层，用于在训练过程中随机丢弃隐藏状态向量的部分内容，以防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # 设置位置嵌入类型，默认为绝对位置编码
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # 注册一个缓冲区，用于存储位置索引的张量，这个张量在序列化时会被导出
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        # 注册一个缓冲区，用于存储标记类型索引的张量，初始值为全零
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

    # 前向传播函数，接受多个输入参数，根据输入计算模型的输出
    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        # 如果输入的 input_ids 不为 None，则获取其形状
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            # 否则，获取 inputs_embeds 的形状（排除最后一维）
            input_shape = inputs_embeds.size()[:-1]

        # 获取序列长度，即输入数据的第二个维度大小
        seq_length = input_shape[1]

        # 如果 position_ids 为 None，则使用预先注册的位置索引张量 self.position_ids
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        # 如果 token_type_ids 为 None，则使用预先注册的标记类型索引张量 self.token_type_ids
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                # 获取并扩展预先注册的 token_type_ids 到与输入形状相匹配的张量
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                # 如果未注册 token_type_ids，则创建一个全零张量，与输入形状相匹配
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        # 如果 inputs_embeds 为 None，则通过 word_embeddings 层将 input_ids 映射为词嵌入向量
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        
        # 根据 token_type_ids 获取标记类型嵌入向量
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 将词嵌入向量和标记类型嵌入向量相加，得到最终的嵌入向量
        embeddings = inputs_embeds + token_type_embeddings
        
        # 如果位置编码方式为绝对位置编码，则添加位置嵌入向量到最终的嵌入向量中
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        
        # 对最终的嵌入向量进行 LayerNorm 标准化处理
        embeddings = self.LayerNorm(embeddings)
        # 对标准化后的向量应用 Dropout，以防止过拟合
        embeddings = self.dropout(embeddings)
        # 返回最终的嵌入向量作为模型的输出
        return embeddings
    """
    Image to Patch Embedding.
    """

    # 初始化函数，设置类的初始状态
    def __init__(self, config):
        super().__init__()
        # 从配置中获取图像大小和patch大小
        image_size, patch_size = config.image_size, config.patch_size
        # 从配置中获取通道数和隐藏层大小
        num_channels, hidden_size = config.num_channels, config.hidden_size

        # 确保image_size和patch_size是可迭代对象，若不是则转为元组
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        # 计算patch的数量
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])

        # 设置类的成员变量
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        # 使用卷积层进行投影，将图像转换为patch embeddings
        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    # 前向传播函数，定义了数据从输入到输出的流程
    def forward(self, pixel_values):
        # 获取输入张量的尺寸信息
        batch_size, num_channels, height, width = pixel_values.shape
        # 如果输入通道数与配置中的通道数不匹配，则抛出数值错误异常
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        # 确定目标数据类型为投影层权重的数据类型
        target_dtype = self.projection.weight.dtype
        # 对输入张量进行投影操作，并转换为目标数据类型
        x = self.projection(pixel_values.to(dtype=target_dtype))
        # 返回投影后的张量作为输出
        return x


class ViltSelfAttention(nn.Module):
    # 初始化函数，设置自注意力模块的初始状态
    def __init__(self, config):
        super().__init__()
        # 如果隐藏层大小不能被注意力头数整除，并且配置中没有嵌入大小属性，则抛出数值错误异常
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        # 设置注意力头数和每个头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 定义查询、键、值的线性映射层，并考虑是否使用偏置
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        # 定义dropout层，用于注意力概率的dropout
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    # 将输入张量重塑为注意力分数计算所需的形状
    def transpose_for_scores(self, x):
        # 计算新的张量形状，以便符合多头注意力的要求
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # 重新排列张量的维度，使得多头注意力能够并行计算
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    # 此方法用于前向传播，计算注意力机制中的上下文表示
    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
        # 通过self.query对隐藏状态进行查询，生成混合的查询层
        mixed_query_layer = self.query(hidden_states)

        # 通过self.key对隐藏状态进行键的变换，并进行得分计算
        key_layer = self.transpose_for_scores(self.key(hidden_states))

        # 通过self.value对隐藏状态进行值的变换，并进行得分计算
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # 对混合的查询层再进行得分计算
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # 通过点积计算"查询"和"键"之间的原始注意力分数
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # 根据注意力头的大小对注意力分数进行缩放
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # 如果提供了注意力掩码，则应用它
        if attention_mask is not None:
            # 注意力掩码是预先计算好的，适用于BertModel的forward()函数中的所有层
            attention_scores = attention_scores + attention_mask

        # 将注意力分数归一化为概率分布
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # 使用dropout函数对注意力概率进行随机失活处理
        attention_probs = self.dropout(attention_probs)

        # 如果需要，对注意力概率进行头部掩码处理
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # 计算加权和得到上下文表示层
        context_layer = torch.matmul(attention_probs, value_layer)

        # 对上下文表示层进行维度变换和重塑
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # 准备输出，根据需要包含注意力分布
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        # 返回计算结果
        return outputs
# Copied from transformers.models.vit.modeling_vit.ViTSelfOutput with ViT->Vilt
class ViltSelfOutput(nn.Module):
    """
    The residual connection is defined in ViltLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: ViltConfig) -> None:
        super().__init__()
        # 定义一个全连接层，输入和输出的维度都是 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 定义一个 dropout 层，根据给定的隐藏状态概率随机将输入置零
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将输入的隐藏状态通过全连接层映射到同一维度
        hidden_states = self.dense(hidden_states)
        # 对映射后的隐藏状态进行 dropout 操作
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class ViltAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化自注意力层和自输出层，都使用给定的配置参数
        self.attention = ViltSelfAttention(config)
        self.output = ViltSelfOutput(config)
        # 初始化一个空集合，用于存储被修剪的注意力头
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 根据给定的头部列表找到可修剪的头部和对应的索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # 修剪线性层
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储修剪的头部
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
        # 通过自注意力层处理隐藏状态和相关的掩码信息
        self_outputs = self.attention(hidden_states, attention_mask, head_mask, output_attentions)

        # 使用自输出层将注意力层的输出与原始隐藏状态相加
        attention_output = self.output(self_outputs[0], hidden_states)

        # 如果输出注意力信息，则将其添加到输出元组中
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


# Copied from transformers.models.vit.modeling_vit.ViTIntermediate with ViT->Vilt
class ViltIntermediate(nn.Module):
    def __init__(self, config: ViltConfig) -> None:
        super().__init__()
        # 定义一个全连接层，输入维度为 config.hidden_size，输出维度为 config.intermediate_size
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 如果隐藏激活函数是字符串，则根据字符串映射到相应的激活函数，否则直接使用给定的激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将输入的隐藏状态通过全连接层映射到 intermediate_size 的维度
        hidden_states = self.dense(hidden_states)
        # 将映射后的隐藏状态通过中间激活函数处理
        hidden_states = self.intermediate_act_fn(hidden_states)

        return hidden_states
# Copied from transformers.models.vit.modeling_vit.ViTOutput with ViT->Vilt
class ViltOutput(nn.Module):
    def __init__(self, config: ViltConfig) -> None:
        super().__init__()
        # 定义一个全连接层，将中间大小的特征转换为隐藏大小
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 定义一个 dropout 层，用于随机断开神经元连接，防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将输入的隐藏状态通过全连接层映射到隐藏大小的空间
        hidden_states = self.dense(hidden_states)
        # 对映射后的结果进行 dropout 处理
        hidden_states = self.dropout(hidden_states)

        # 将处理后的隐藏状态与输入张量相加作为最终输出
        hidden_states = hidden_states + input_tensor

        return hidden_states


class ViltLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config):
        super().__init__()
        # 设置用于分块前馈的块大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 序列长度的维度索引
        self.seq_len_dim = 1
        # 初始化自注意力层、中间层和输出层
        self.attention = ViltAttention(config)
        self.intermediate = ViltIntermediate(config)
        self.output = ViltOutput(config)
        # ViLT 中的 layernorm 在自注意力之前应用
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # ViLT 中的 layernorm 也在自注意力之后应用
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False):
        # 对输入的隐藏状态应用 layernorm，并传入自注意力层进行处理
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # 如果输出注意力权重，添加自注意力的输出

        # 第一个残差连接：将自注意力的输出与原始隐藏状态相加
        hidden_states = attention_output + hidden_states.to(attention_output.device)

        # 在 ViLT 中，layernorm 也在自注意力之后应用
        layer_output = self.layernorm_after(hidden_states)
        # 经过中间层的处理
        layer_output = self.intermediate(layer_output)

        # 第二个残差连接：将中间层的输出与原始隐藏状态传入输出层
        layer_output = self.output(layer_output, hidden_states)

        # 将最终层的输出添加到输出集合中
        outputs = (layer_output,) + outputs

        return outputs


class ViltEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 创建多层 ViltLayer 构成的层列表
        self.layer = nn.ModuleList([ViltLayer(config) for _ in range(config.num_hidden_layers)])
        # 默认关闭梯度检查点
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        ):
        ):
            # 如果不需要输出隐藏状态，则初始化为空元组；否则设为 None
            all_hidden_states = () if output_hidden_states else None
            # 如果不需要输出注意力权重，则初始化为空元组；否则设为 None
            all_self_attentions = () if output_attentions else None
        
            # 遍历 Transformer 模型的每一层
            for i, layer_module in enumerate(self.layer):
                # 如果需要输出隐藏状态，则累加当前隐藏状态到 all_hidden_states
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)
                
                # 获取当前层的头部遮罩，如果未提供则为 None
                layer_head_mask = head_mask[i] if head_mask is not None else None
                
                # 如果启用渐变检查点并且处于训练模式下
                if self.gradient_checkpointing and self.training:
                    # 通过渐变检查点功能调用当前层模块，获取层的输出
                    layer_outputs = self._gradient_checkpointing_func(
                        layer_module.__call__,
                        hidden_states,
                        attention_mask,
                        layer_head_mask,
                        output_attentions,
                    )
                else:
                    # 否则直接调用当前层模块，获取层的输出
                    layer_outputs = layer_module(hidden_states, attention_mask, layer_head_mask, output_attentions)
    
                # 更新隐藏状态为当前层的输出的第一个元素
                hidden_states = layer_outputs[0]
    
                # 如果需要输出注意力权重，则累加当前层的注意力权重到 all_self_attentions
                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)
    
            # 如果需要输出隐藏状态，则最后将当前隐藏状态加入 all_hidden_states
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
    
            # 如果不返回字典形式的输出，则按顺序返回隐藏状态、所有隐藏状态和所有注意力权重的非空元组
            if not return_dict:
                return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
            # 否则以 BaseModelOutput 类的形式返回结果，包含最终隐藏状态、所有隐藏状态和所有注意力权重
            return BaseModelOutput(
                last_hidden_state=hidden_states,
                hidden_states=all_hidden_states,
                attentions=all_self_attentions,
            )
class ViltPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 设置模型的配置类
    config_class = ViltConfig
    # 模型基础名称前缀
    base_model_prefix = "vilt"
    # 支持梯度检查点
    supports_gradient_checkpointing = True
    # 不需要分割的模块列表
    _no_split_modules = ["ViltEmbeddings", "ViltSelfAttention"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 如果是线性层或卷积层，使用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                # 如果有偏置，则将偏置初始化为零
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # 如果是嵌入层，使用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                # 如果有填充索引，则将对应位置的权重初始化为零
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            # 如果是LayerNorm层，将偏置初始化为零，权重初始化为1.0
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


# ViLT模型的起始文档字符串
VILT_START_DOCSTRING = r"""
    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ subclass. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`ViltConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# ViLT模型输入文档字符串（空白）
VILT_INPUTS_DOCSTRING = r"""
"""

# ViLT图像和文本分类输入文档字符串（空白）
VILT_IMAGES_AND_TEXT_CLASSIFICATION_INPUTS_DOCSTRING = r"""
"""

# 添加起始文档字符串注释到ViltModel类
@add_start_docstrings(
    "The bare ViLT Model transformer outputting raw hidden-states without any specific head on top.",
    VILT_START_DOCSTRING,
)
class ViltModel(ViltPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        # 初始化嵌入层和编码器
        self.embeddings = ViltEmbeddings(config)
        self.encoder = ViltEncoder(config)

        # LayerNorm层，用于归一化隐藏层输出
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # 如果需要添加汇聚层，则初始化汇聚器
        self.pooler = ViltPooler(config) if add_pooling_layer else None

        # 初始化权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        return self.embeddings.text_embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.text_embeddings.word_embeddings = value
    # 修剪模型的注意力头
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        # 遍历需要修剪的层和对应需要修剪的注意力头
        for layer, heads in heads_to_prune.items():
            # 对编码器中的特定层的注意力头进行修剪
            self.encoder.layer[layer].attention.prune_heads(heads)

    # 将输入参数添加到模型的文档字符串
    @add_start_docstrings_to_model_forward(VILT_INPUTS_DOCSTRING)
    # 替换返回值的文档字符串
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=_CONFIG_FOR_DOC)
    # 定义模型的前向传播方法
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        image_embeds: Optional[torch.FloatTensor] = None,
        image_token_type_idx: Optional[int] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
class ViltPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 使用全连接层进行线性变换，输入和输出维度都是 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        
        # 根据配置选择激活函数，如果配置中指定的是字符串形式的激活函数，则使用对应的函数，否则直接使用配置中的函数
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        
        # 应用 Layer Normalization 进行归一化处理，参数包括隐藏状态的维度和层归一化的 epsilon 值
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        # 将隐藏状态通过全连接层进行线性变换
        hidden_states = self.dense(hidden_states)
        
        # 应用选择的激活函数进行非线性变换
        hidden_states = self.transform_act_fn(hidden_states)
        
        # 对变换后的隐藏状态应用 Layer Normalization 进行归一化
        hidden_states = self.LayerNorm(hidden_states)
        
        return hidden_states
    # 初始化函数，用于初始化模型对象
    def __init__(self, config, weight=None):
        # 调用父类的初始化方法
        super().__init__()
        # 将配置参数保存到对象的属性中
        self.config = config
        # 创建一个 ViltPredictionHeadTransform 的实例，并保存到对象的属性中
        self.transform = ViltPredictionHeadTransform(config)
        # 创建一个线性层，用于模型的解码器，指定输入大小为 config.hidden_size，输出大小为 config.vocab_size，且没有偏置项
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # 创建一个可学习的偏置项，大小为 config.vocab_size
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        # 如果给定了预训练权重 weight，则将其赋值给解码器的权重
        if weight is not None:
            self.decoder.weight = weight

        # 为了确保偏置项能够正确地在调整 token embeddings 时被重新调整大小，需要在这里建立两者之间的链接
        self.decoder.bias = self.bias

    # 前向传播函数，接收输入 x 并返回模型的输出 x
    def forward(self, x):
        # 对输入 x 应用预测头变换
        x = self.transform(x)
        # 使用解码器对变换后的 x 进行解码
        x = self.decoder(x)
        # 返回解码后的输出 x
        return x
@add_start_docstrings(
    """
    Vilt Model transformer with a classifier head on top (a linear layer on top of the final hidden state of the [CLS]
    token) for visual question answering, e.g. for VQAv2.
    """,
    VILT_START_DOCSTRING,
)
class ViltForQuestionAnswering(ViltPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.vilt = ViltModel(config)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size * 2),  # Linear layer to expand hidden size
            nn.LayerNorm(config.hidden_size * 2),  # Layer normalization
            nn.GELU(),  # GELU activation function
            nn.Linear(config.hidden_size * 2, config.num_labels),  # Final linear layer for classification
        )

        # Initialize weights and apply final processing
        self.post_init()

    @add_start_docstrings_to_model_forward(VILT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        image_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ) -> Union[SequenceClassifierOutput, Tuple[torch.FloatTensor]]:
        r"""
        labels (`torch.FloatTensor` of shape `(batch_size, num_labels)`, *optional*):
            Labels for computing the visual question answering loss. This tensor must be either a one-hot encoding of
            all answers that are applicable for a given example in the batch, or a soft encoding indicating which
            answers are applicable, where 1.0 is the highest score.

        Returns:
            Depending on `return_dict`, returns either a `SequenceClassifierOutput` or a tuple containing logits and optionally other outputs.

        Examples:

        ```python
        >>> from transformers import ViltProcessor, ViltForQuestionAnswering
        >>> import requests
        >>> from PIL import Image

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> text = "How many cats are there?"

        >>> processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        >>> model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

        >>> # prepare inputs
        >>> encoding = processor(image, text, return_tensors="pt")

        >>> # forward pass
        >>> outputs = model(**encoding)
        >>> logits = outputs.logits
        >>> idx = logits.argmax(-1).item()
        >>> print("Predicted answer:", model.config.id2label[idx])
        Predicted answer: 2
        ```"""

        # Determine whether to use the return_dict provided or the class attribute for return settings
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Perform forward pass through the VILT model
        outputs = self.vilt(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            image_embeds=image_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Extract the pooler_output from the outputs based on return_dict setting
        pooler_output = outputs.pooler_output if return_dict else outputs[1]

        # Pass the pooler_output through the classifier layer to obtain logits
        logits = self.classifier(pooler_output)

        # Initialize loss variable
        loss = None

        # Calculate loss if labels are provided
        if labels is not None:
            # Move labels tensor to the same device as logits for compatibility
            labels = labels.to(logits.device)
            # Compute binary cross entropy loss scaled by number of labels
            loss = nn.functional.binary_cross_entropy_with_logits(logits, labels) * labels.shape[1]
            # Reference to paper or implementation where this loss scaling is discussed

        # Prepare output based on return_dict flag
        if not return_dict:
            # If return_dict is False, prepare tuple output
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # If return_dict is True, prepare SequenceClassifierOutput object
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 使用装饰器为类添加文档字符串，描述了该类的作用和功能，以及适用的应用场景（图片到文本或文本到图片的检索）
@add_start_docstrings(
    """
    Vilt Model transformer with a classifier head on top (a linear layer on top of the final hidden state of the [CLS]
    token) for image-to-text or text-to-image retrieval, e.g. MSCOCO and F30K.
    """,
    VILT_START_DOCSTRING,
)
# 定义 ViltForImageAndTextRetrieval 类，继承自 ViltPreTrainedModel
class ViltForImageAndTextRetrieval(ViltPreTrainedModel):
    
    # 初始化方法，接受一个 config 对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 创建 ViltModel 的实例，并保存到 self.vilt 属性中
        self.vilt = ViltModel(config)

        # 分类器头部，使用线性层将最终隐藏状态（[CLS] token）映射到单一输出维度
        self.rank_output = nn.Linear(config.hidden_size, 1)

        # 初始化权重并进行最终处理
        self.post_init()

    # 使用装饰器为 forward 方法添加文档字符串，描述了该方法的输入参数及其作用
    @add_start_docstrings_to_model_forward(VILT_INPUTS_DOCSTRING)
    # 使用装饰器替换返回值的文档字符串，指定输出类型为 SequenceClassifierOutput，配置类为 _CONFIG_FOR_DOC
    @replace_return_docstrings(output_type=SequenceClassifierOutput, config_class=_CONFIG_FOR_DOC)
    # forward 方法，处理模型的前向传播逻辑
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        image_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[SequenceClassifierOutput, Tuple[torch.FloatTensor]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels are currently not supported.

        Returns:
            Depending on `return_dict` flag:
                - If `return_dict` is False, returns a tuple containing `logits` and additional outputs.
                - If `return_dict` is True, returns a `SequenceClassifierOutput` object containing `loss`, `logits`, `hidden_states`, and `attentions`.

        Examples:

        ```python
        >>> from transformers import ViltProcessor, ViltForImageAndTextRetrieval
        >>> import requests
        >>> from PIL import Image

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> texts = ["An image of two cats chilling on a couch", "A football player scoring a goal"]

        >>> processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-coco")
        >>> model = ViltForImageAndTextRetrieval.from_pretrained("dandelin/vilt-b32-finetuned-coco")

        >>> # forward pass
        >>> scores = dict()
        >>> for text in texts:
        ...     # prepare inputs
        ...     encoding = processor(image, text, return_tensors="pt")
        ...     outputs = model(**encoding)
        ...     scores[text] = outputs.logits[0, :].item()
        ```
        """
        # Determine whether to use the return_dict flag or the model's default configuration
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Perform the forward pass through the VILT model
        outputs = self.vilt(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            image_embeds=image_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Select the pooler output based on whether return_dict is True or False
        pooler_output = outputs.pooler_output if return_dict else outputs[1]

        # Generate logits using the rank_output method
        logits = self.rank_output(pooler_output)

        # Initialize loss as None
        loss = None

        # Handle labels if provided (currently raises NotImplementedError)
        if labels is not None:
            # Move labels to the device where logits are located
            labels = labels.to(logits.device)
            raise NotImplementedError("Training is not yet supported.")

        # Return the output based on whether return_dict is True or False
        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # Return a SequenceClassifierOutput object containing loss, logits, hidden_states, and attentions
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
@add_start_docstrings(
    """
    Vilt Model transformer with a classifier head on top for natural language visual reasoning, e.g. NLVR2.
    """,
    VILT_IMAGES_AND_TEXT_CLASSIFICATION_INPUTS_DOCSTRING,
)
class ViltForImagesAndTextClassification(ViltPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        self.vilt = ViltModel(config)

        # Classifier head
        num_images = config.num_images
        # 定义分类器，包括线性层、LayerNorm和GELU激活函数
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size * num_images, config.hidden_size * num_images),
            nn.LayerNorm(config.hidden_size * num_images),
            nn.GELU(),
            nn.Linear(config.hidden_size * num_images, config.num_labels),
        )

        # Initialize weights and apply final processing
        # 初始化权重并进行最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(VILT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=ViltForImagesAndTextClassificationOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        image_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,



@add_start_docstrings(
    """
    ViLT Model with a token classification head on top (a linear layer on top of the final hidden-states of the text
    tokens) e.g. for Named-Entity-Recognition (NER) tasks.
    """,
    VILT_START_DOCSTRING,
)
class ViltForTokenClassification(ViltPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.num_labels = config.num_labels
        # 初始化 ViLT 模型，不添加池化层
        self.vilt = ViltModel(config, add_pooling_layer=False)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 分类器是一个线性层，输出维度为 config.num_labels
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        # 初始化权重并进行最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(VILT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TokenClassifierOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        image_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[TokenClassifierOutput, Tuple[torch.FloatTensor]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, text_sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.

        Returns:
            Either a `TokenClassifierOutput` containing loss, logits, hidden states, and attentions,
            or a tuple with logits and optional hidden states and attentions.
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Pass inputs to the VILT model for processing
        outputs = self.vilt(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            pixel_values=pixel_values,
            pixel_mask=pixel_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            image_embeds=image_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        # Determine the size of the text input
        text_input_size = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        # Apply dropout to the sequence output
        sequence_output = self.dropout(sequence_output)
        
        # Classify tokens using the classifier layer
        logits = self.classifier(sequence_output[:, :text_input_size])

        loss = None
        if labels is not None:
            # Calculate the cross-entropy loss
            loss_fct = CrossEntropyLoss()
            # Move labels to the same device as logits
            labels = labels.to(logits.device)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            # If return_dict is False, return a tuple of logits and optionally hidden states and attentions
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        # If return_dict is True, return a TokenClassifierOutput object
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```