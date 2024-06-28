# `.\models\dpt\modeling_dpt.py`

```
    """
    PyTorch DPT (Dense Prediction Transformers) model.

    This implementation is heavily inspired by OpenMMLab's implementation, found here:
    https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/models/decode_heads/dpt_head.py.
    """

# 导入必要的模块和库
import collections.abc
import math
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

# 导入工具函数和类
from ...activations import ACT2FN
from ...file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from ...modeling_outputs import BaseModelOutput, DepthEstimatorOutput, SemanticSegmenterOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import ModelOutput, logging
from ...utils.backbone_utils import load_backbone
from .configuration_dpt import DPTConfig

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 用于文档的常量和模型配置信息
_CONFIG_FOR_DOC = "DPTConfig"
_CHECKPOINT_FOR_DOC = "Intel/dpt-large"
_EXPECTED_OUTPUT_SHAPE = [1, 577, 1024]

# 预训练模型列表
DPT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "Intel/dpt-large",
    "Intel/dpt-hybrid-midas",
    # See all DPT models at https://huggingface.co/models?filter=dpt
]

@dataclass
class BaseModelOutputWithIntermediateActivations(ModelOutput):
    """
    Base class for model's outputs that also contains intermediate activations that can be used at later stages. Useful
    in the context of Vision models.

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        intermediate_activations (`tuple(torch.FloatTensor)`, *optional*):
            Intermediate activations that can be used to compute hidden states of the model at various layers.
    """
    last_hidden_states: torch.FloatTensor = None
    intermediate_activations: Optional[Tuple[torch.FloatTensor, ...]] = None


@dataclass
class BaseModelOutputWithPoolingAndIntermediateActivations(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states as well as intermediate
    """
    # 最后一个隐藏状态：模型最后一层输出的隐藏状态，形状为(batch_size, sequence_length, hidden_size)
    last_hidden_state: torch.FloatTensor = None
    
    # 汇聚器输出：经过进一步处理后，序列中第一个标记（分类标记）的最后一层隐藏状态。例如，对于BERT系列模型，这是经过线性层和tanh激活函数处理后的分类标记。
    # 线性层的权重在预训练阶段通过下一个句子预测（分类）目标进行训练。
    pooler_output: torch.FloatTensor = None
    
    # 隐藏状态：模型在每一层输出的隐藏状态的元组，如果模型有嵌入层，则包括嵌入层的输出。
    # 形状为(batch_size, sequence_length, hidden_size)。
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    
    # 注意力权重：注意力softmax后的注意力权重，用于在自注意力头中计算加权平均值。
    # 形状为(batch_size, num_heads, sequence_length, sequence_length)的元组，每个元素对应一个层。
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    
    # 中间激活：可用于计算各层模型隐藏状态的中间激活。
    intermediate_activations: Optional[Tuple[torch.FloatTensor, ...]] = None
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    def __init__(self, config, feature_size=None):
        super().__init__()
        
        # Extract configuration parameters
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        # Ensure image_size and patch_size are iterable, defaulting to tuple if not
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        
        # Calculate number of patches based on image and patch size
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])

        # Load backbone model defined by configuration
        self.backbone = load_backbone(config)
        feature_dim = self.backbone.channels[-1]

        # Ensure backbone has exactly 3 output features
        if len(self.backbone.channels) != 3:
            raise ValueError(f"Expected backbone to have 3 output features, got {len(self.backbone.channels)}")

        # Define indices of backbone stages to use for residual feature maps
        self.residual_feature_map_index = [0, 1]  # Always take the output of the first and second backbone stage

        # Determine feature size based on configuration or input feature_size
        if feature_size is None:
            feat_map_shape = config.backbone_featmap_shape
            feature_size = feat_map_shape[-2:]
            feature_dim = feat_map_shape[1]
        else:
            feature_size = (
                feature_size if isinstance(feature_size, collections.abc.Iterable) else (feature_size, feature_size)
            )
            feature_dim = self.backbone.channels[-1]

        # Store relevant configuration parameters
        self.image_size = image_size
        self.patch_size = patch_size[0]
        self.num_channels = num_channels

        # Projection layer to convert feature dimension to hidden size
        self.projection = nn.Conv2d(feature_dim, hidden_size, kernel_size=1)

        # Initialize classification token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, config.hidden_size))

    def _resize_pos_embed(self, posemb, grid_size_height, grid_size_width, start_index=1):
        """
        Resize the positional embeddings to match a specified grid size.

        Args:
            posemb (torch.Tensor): Positional embeddings tensor.
            grid_size_height (int): Target grid height.
            grid_size_width (int): Target grid width.
            start_index (int, optional): Starting index for grid reshaping. Defaults to 1.

        Returns:
            torch.Tensor: Resized positional embeddings tensor.
        """
        # Separate token and grid positional embeddings
        posemb_tok = posemb[:, :start_index]
        posemb_grid = posemb[0, start_index:]

        # Determine current grid size
        old_grid_size = int(math.sqrt(len(posemb_grid)))

        # Reshape grid embeddings and interpolate to target size
        posemb_grid = posemb_grid.reshape(1, old_grid_size, old_grid_size, -1).permute(0, 3, 1, 2)
        posemb_grid = nn.functional.interpolate(posemb_grid, size=(grid_size_height, grid_size_width), mode="bilinear")
        posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, grid_size_height * grid_size_width, -1)

        # Concatenate token and resized grid embeddings
        posemb = torch.cat([posemb_tok, posemb_grid], dim=1)

        return posemb

    def forward(
        self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False, return_dict: bool = False
    ):
        """
        Perform forward pass of the model.

        Args:
            pixel_values (torch.Tensor): Input tensor of shape `(batch_size, num_channels, height, width)`.
            interpolate_pos_encoding (bool): Whether to interpolate positional embeddings. Defaults to False.
            return_dict (bool): Whether to return output as dictionary. Defaults to False.

        Returns:
            torch.Tensor or dict: Output tensor or dictionary depending on `return_dict`.
        """
    ) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        # 获取输入张量的维度信息，分别是批量大小、通道数、高度和宽度

        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        # 如果输入的通道数与模型期望的通道数不一致，则抛出数值错误

        if not interpolate_pos_encoding:
            if height != self.image_size[0] or width != self.image_size[1]:
                raise ValueError(
                    f"Input image size ({height}*{width}) doesn't match model"
                    f" ({self.image_size[0]}*{self.image_size[1]})."
                )
        # 如果不需要插值位置编码，并且输入图像的高度或宽度与模型期望的不匹配，则抛出数值错误

        position_embeddings = self._resize_pos_embed(
            self.position_embeddings, height // self.patch_size, width // self.patch_size
        )
        # 调整位置编码的大小，以匹配当前输入图像的尺寸，将其存储在 position_embeddings 中

        backbone_output = self.backbone(pixel_values)
        # 使用预训练的骨干网络处理输入的像素值，获取骨干网络的输出

        features = backbone_output.feature_maps[-1]
        # 从骨干网络的输出中获取最后一层的特征图作为特征

        # Retrieve also the intermediate activations to use them at later stages
        output_hidden_states = [backbone_output.feature_maps[index] for index in self.residual_feature_map_index]
        # 获取中间层的激活结果，以备后续使用

        embeddings = self.projection(features).flatten(2).transpose(1, 2)
        # 将提取的特征投影到嵌入空间，并进行展平和转置操作

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # 扩展分类令牌，以匹配当前批次的大小

        embeddings = torch.cat((cls_tokens, embeddings), dim=1)
        # 将分类令牌和特征嵌入拼接在一起，形成最终的嵌入表示

        # add positional encoding to each token
        embeddings = embeddings + position_embeddings
        # 将位置编码添加到每个令牌的嵌入表示中

        if not return_dict:
            return (embeddings, output_hidden_states)
        # 如果不需要返回字典形式的输出，则直接返回嵌入表示和中间层激活结果的元组

        # Return hidden states and intermediate activations
        return BaseModelOutputWithIntermediateActivations(
            last_hidden_states=embeddings,
            intermediate_activations=output_hidden_states,
        )
        # 否则，返回包含最终隐藏状态和中间激活状态的 BaseModelOutputWithIntermediateActivations 对象
# 定义一个名为 DPTViTEmbeddings 的 nn.Module 类，用于处理 DPT-ViT 模型的嵌入部分，包括 CLS token、位置编码和图像补丁的嵌入。

class DPTViTEmbeddings(nn.Module):
    """
    Construct the CLS token, position and patch embeddings.
    构建 CLS token、位置编码和图像补丁的嵌入。

    """

    def __init__(self, config):
        super().__init__()

        # 初始化 CLS token 参数
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        # 初始化图像补丁的嵌入
        self.patch_embeddings = DPTViTPatchEmbeddings(config)
        num_patches = self.patch_embeddings.num_patches
        # 初始化位置编码的参数
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, config.hidden_size))
        # 初始化 Dropout 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 保存配置信息
        self.config = config

    def _resize_pos_embed(self, posemb, grid_size_height, grid_size_width, start_index=1):
        # 提取位置编码中的 token 部分
        posemb_tok = posemb[:, :start_index]
        # 提取位置编码中的 grid 部分
        posemb_grid = posemb[0, start_index:]

        # 获取旧的 grid 尺寸
        old_grid_size = int(math.sqrt(len(posemb_grid)))

        # 重塑 grid 部分并进行双线性插值
        posemb_grid = posemb_grid.reshape(1, old_grid_size, old_grid_size, -1).permute(0, 3, 1, 2)
        posemb_grid = nn.functional.interpolate(posemb_grid, size=(grid_size_height, grid_size_width), mode="bilinear")
        posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, grid_size_height * grid_size_width, -1)

        # 合并 token 和插值后的 grid
        posemb = torch.cat([posemb_tok, posemb_grid], dim=1)

        return posemb

    def forward(self, pixel_values, return_dict=False):
        batch_size, num_channels, height, width = pixel_values.shape

        # 可能需要插值位置编码以处理不同大小的图像
        patch_size = self.config.patch_size
        position_embeddings = self._resize_pos_embed(
            self.position_embeddings, height // patch_size, width // patch_size
        )

        # 计算图像补丁的嵌入
        embeddings = self.patch_embeddings(pixel_values)

        batch_size, seq_len, _ = embeddings.size()

        # 将 [CLS] token 添加到嵌入的补丁 token 中
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # 添加位置编码到每个 token
        embeddings = embeddings + position_embeddings

        # 应用 Dropout
        embeddings = self.dropout(embeddings)

        # 如果不需要返回字典，则返回嵌入结果元组
        if not return_dict:
            return (embeddings,)

        # 如果需要返回字典，则构建 BaseModelOutputWithIntermediateActivations 并返回
        return BaseModelOutputWithIntermediateActivations(last_hidden_states=embeddings)


class DPTViTPatchEmbeddings(nn.Module):
    """
    Image to Patch Embedding.
    图像到补丁的嵌入。

    """
    # 初始化函数，用于初始化一个类实例
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 从配置中获取图像大小和patch大小
        image_size, patch_size = config.image_size, config.patch_size
        # 从配置中获取通道数和隐藏层大小
        num_channels, hidden_size = config.num_channels, config.hidden_size

        # 如果图像大小和patch大小不是可迭代对象，则转换为元组
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        
        # 计算patch的数量
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        
        # 将相关信息保存到类实例的属性中
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        # 使用 nn.Conv2d 定义一个投影层，将输入的通道数映射到隐藏层大小
        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    # 前向传播函数，接受像素值作为输入并返回嵌入向量
    def forward(self, pixel_values):
        # 获取输入张量的批次大小、通道数、高度和宽度
        batch_size, num_channels, height, width = pixel_values.shape
        
        # 如果输入通道数与配置中设置的不符，则抛出 ValueError 异常
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        
        # 将输入张量通过投影层，然后展平、转置，得到嵌入向量
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        
        # 返回嵌入向量作为结果
        return embeddings
# 从transformers.models.vit.modeling_vit.ViTSelfAttention复制并修改为DPTViTSelfAttention
class DPTViTSelfAttention(nn.Module):
    def __init__(self, config: DPTConfig) -> None:
        super().__init__()
        # 检查hidden_size是否是attention头数的整数倍，若不是则抛出异常
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        # 设置注意力头数和每个头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 定义query、key、value线性变换层，用于构造注意力机制中的查询、键、值
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        # 定义dropout层，用于在注意力机制中应用dropout
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    # 将输入张量x转置以匹配注意力头的形状
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    # 前向传播函数，执行自注意力机制的计算过程
    def forward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        # 计算混合查询层
        mixed_query_layer = self.query(hidden_states)

        # 计算转置后的键和值张量，以便进行注意力分数计算
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # 计算原始的注意力分数，即查询与键的点积
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # 对注意力分数进行缩放
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # 对注意力分数进行softmax操作，得到注意力权重
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # 应用dropout到注意力权重上，以减少过拟合风险
        attention_probs = self.dropout(attention_probs)

        # 如果提供了head_mask，则应用到注意力权重上
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # 计算上下文层，即注意力权重与值的乘积
        context_layer = torch.matmul(attention_probs, value_layer)

        # 调整上下文层张量的形状以匹配原始输入张量的形状
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        # 如果需要输出注意力权重，将其添加到输出中
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs
# Copied from transformers.models.vit.modeling_vit.ViTSelfOutput with ViT->DPT
class DPTViTSelfOutput(nn.Module):
    """
    The residual connection is defined in DPTLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: DPTConfig) -> None:
        super().__init__()
        # 定义一个全连接层，输入和输出的维度都是 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 定义一个 dropout 层，用于在训练时随机置零输入张量的部分元素
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将输入的 hidden_states 通过全连接层 self.dense 进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对变换后的 hidden_states 应用 dropout
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class DPTViTAttention(nn.Module):
    def __init__(self, config: DPTConfig) -> None:
        super().__init__()
        # 创建 DPTViTSelfAttention 类的实例，该类负责注意力计算
        self.attention = DPTViTSelfAttention(config)
        # 创建 DPTViTSelfOutput 类的实例，该类负责自注意力输出和残差连接
        self.output = DPTViTSelfOutput(config)
        # 初始化一个集合，用于记录被修剪的注意力头的索引
        self.pruned_heads = set()

    # Copied from transformers.models.vit.modeling_vit.ViTAttention.prune_heads
    def prune_heads(self, heads: Set[int]) -> None:
        if len(heads) == 0:
            return
        # 调用 find_pruneable_heads_and_indices 函数，找到需要修剪的头部及其索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # 修剪注意力机制中的线性层
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并记录修剪过的注意力头
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    # Copied from transformers.models.vit.modeling_vit.ViTAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        # 调用 DPTViTSelfAttention 的 forward 方法进行注意力计算
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)

        # 将 self_outputs[0] 作为输入，通过 DPTViTSelfOutput 的 forward 方法进行输出计算
        attention_output = self.output(self_outputs[0], hidden_states)

        # 如果需要输出注意力权重，将它们添加到 outputs 中
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.vit.modeling_vit.ViTIntermediate with ViT->DPT
class DPTViTIntermediate(nn.Module):
    # 初始化方法，用于初始化对象
    def __init__(self, config: DPTConfig) -> None:
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个线性层，将输入特征的大小调整为中间特征的大小
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 检查隐藏激活函数是否为字符串类型，选择相应的激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    # 前向传播方法，处理输入张量并返回处理后的张量
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将输入张量通过线性层
        hidden_states = self.dense(hidden_states)
        # 应用中间激活函数到线性层的输出
        hidden_states = self.intermediate_act_fn(hidden_states)

        # 返回处理后的张量作为输出
        return hidden_states
# Copied from transformers.models.vit.modeling_vit.ViTOutput with ViT->DPT
class DPTViTOutput(nn.Module):
    def __init__(self, config: DPTConfig) -> None:
        super().__init__()
        # 定义一个全连接层，将输入维度为 config.intermediate_size 的向量映射到维度为 config.hidden_size 的向量
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 定义一个 dropout 层，以 config.hidden_dropout_prob 的概率随机将输入设置为 0，防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将输入的 hidden_states 应用全连接层
        hidden_states = self.dense(hidden_states)
        # 对全连接层的输出进行 dropout 处理
        hidden_states = self.dropout(hidden_states)

        # 将 dropout 处理后的 hidden_states 与输入的 input_tensor 相加，实现残差连接
        hidden_states = hidden_states + input_tensor

        return hidden_states


# copied from transformers.models.vit.modeling_vit.ViTLayer with ViTConfig->DPTConfig, ViTAttention->DPTViTAttention, ViTIntermediate->DPTViTIntermediate, ViTOutput->DPTViTOutput
class DPTViTLayer(nn.Module):
    """This corresponds to the Block class in the timm implementation."""

    def __init__(self, config: DPTConfig) -> None:
        super().__init__()
        # 设置用于分块前向传播的块大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 序列长度的维度设置为1，通常在处理序列时使用
        self.seq_len_dim = 1
        # 创建自注意力层对象 DPTViTAttention
        self.attention = DPTViTAttention(config)
        # 创建中间层对象 DPTViTIntermediate
        self.intermediate = DPTViTIntermediate(config)
        # 创建输出层对象 DPTViTOutput
        self.output = DPTViTOutput(config)
        # 在隐藏状态维度上应用 LayerNorm，epsilon 设置为 config.layer_norm_eps
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 在隐藏状态维度上应用 LayerNorm，epsilon 设置为 config.layer_norm_eps
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        # 在自注意力层之前应用 LayerNorm
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),
            head_mask,
            output_attentions=output_attentions,
        )
        # 获取自注意力层的输出
        attention_output = self_attention_outputs[0]
        # 如果需要输出注意力权重，则添加自注意力层的注意力权重到 outputs
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # 第一个残差连接
        hidden_states = attention_output + hidden_states

        # 在自注意力层之后应用 LayerNorm
        layer_output = self.layernorm_after(hidden_states)
        # 经过中间层处理
        layer_output = self.intermediate(layer_output)

        # 第二个残差连接在这里完成
        layer_output = self.output(layer_output, hidden_states)

        # 将本层的输出添加到 outputs
        outputs = (layer_output,) + outputs

        return outputs


# copied from transformers.models.vit.modeling_vit.ViTEncoder with ViTConfig -> DPTConfig, ViTLayer->DPTViTLayer
class DPTViTEncoder(nn.Module):
    def __init__(self, config: DPTConfig) -> None:
        super().__init__()
        self.config = config
        # 创建一个包含 config.num_hidden_layers 个 DPTViTLayer 层的 ModuleList
        self.layer = nn.ModuleList([DPTViTLayer(config) for _ in range(config.num_hidden_layers)])
        # 梯度检查点设置为 False，用于指示是否使用梯度检查点来节省内存
        self.gradient_checkpointing = False
    # 定义前向传播函数，接收隐藏状态、头部掩码、是否输出注意力、是否输出隐藏状态、是否返回字典作为参数，返回元组或BaseModelOutput对象
    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        # 如果需要输出隐藏状态，则初始化一个空元组用于存储所有的隐藏状态
        all_hidden_states = () if output_hidden_states else None
        # 如果需要输出注意力，则初始化一个空元组用于存储所有的自注意力矩阵
        all_self_attentions = () if output_attentions else None

        # 遍历所有的层次模块
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，则将当前的隐藏状态添加到all_hidden_states元组中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前层次的头部掩码
            layer_head_mask = head_mask[i] if head_mask is not None else None

            # 如果启用了梯度检查点并且处于训练状态，则使用梯度检查点函数来计算当前层的输出
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                # 否则直接调用当前层次模块计算输出
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)

            # 更新隐藏状态为当前层的输出的第一个元素
            hidden_states = layer_outputs[0]

            # 如果需要输出注意力，则将当前层的自注意力矩阵添加到all_self_attentions元组中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 如果需要输出隐藏状态，则将最终的隐藏状态添加到all_hidden_states元组中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不需要返回字典形式的输出，则将所有非空的元组打包成一个元组返回
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        # 否则，返回一个BaseModelOutput对象，包含最终的隐藏状态、所有隐藏状态和所有自注意力矩阵
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
class DPTReassembleStage(nn.Module):
    """
    This class reassembles the hidden states of the backbone into image-like feature representations at various
    resolutions.

    This happens in 3 stages:
    1. Map the N + 1 tokens to a set of N tokens, by taking into account the readout ([CLS]) token according to
       `config.readout_type`.
    2. Project the channel dimension of the hidden states according to `config.neck_hidden_sizes`.
    3. Resizing the spatial dimensions (height, width).

    Args:
        config (`[DPTConfig]`):
            Model configuration class defining the model architecture.
    """

    def __init__(self, config):
        super().__init__()

        self.config = config  # 存储模型配置信息
        self.layers = nn.ModuleList()  # 初始化层列表

        # 根据配置选择初始化 DPT-Hybrid 或者普通的 DPT 模型
        if config.is_hybrid:
            self._init_reassemble_dpt_hybrid(config)
        else:
            self._init_reassemble_dpt(config)

        self.neck_ignore_stages = config.neck_ignore_stages  # 存储忽略的 neck stages

    def _init_reassemble_dpt_hybrid(self, config):
        r"""
        For DPT-Hybrid the first 2 reassemble layers are set to `nn.Identity()`, please check the official
        implementation: https://github.com/isl-org/DPT/blob/f43ef9e08d70a752195028a51be5e1aff227b913/dpt/vit.py#L438
        for more details.
        """
        # 根据配置初始化 DPT-Hybrid 模型的各层
        for i, factor in zip(range(len(config.neck_hidden_sizes)), config.reassemble_factors):
            if i <= 1:
                self.layers.append(nn.Identity())  # 前两层设置为 nn.Identity()
            elif i > 1:
                self.layers.append(DPTReassembleLayer(config, channels=config.neck_hidden_sizes[i], factor=factor))

        # 如果 readout_type 不是 "project"，则抛出错误
        if config.readout_type != "project":
            raise ValueError(f"Readout type {config.readout_type} is not supported for DPT-Hybrid.")

        # 当使用 DPT-Hybrid 时，readout type 被设置为 "project"，在配置文件中进行了检查
        self.readout_projects = nn.ModuleList()  # 初始化 readout projects 模块列表
        hidden_size = _get_backbone_hidden_size(config)  # 获取 backbone 隐藏层大小
        for i in range(len(config.neck_hidden_sizes)):
            if i <= 1:
                self.readout_projects.append(nn.Sequential(nn.Identity()))  # 前两层设置为 nn.Identity()
            elif i > 1:
                self.readout_projects.append(
                    nn.Sequential(nn.Linear(2 * hidden_size, hidden_size), ACT2FN[config.hidden_act])
                )

    def _init_reassemble_dpt(self, config):
        # 根据配置初始化普通的 DPT 模型的各层
        for i, factor in zip(range(len(config.neck_hidden_sizes)), config.reassemble_factors):
            self.layers.append(DPTReassembleLayer(config, channels=config.neck_hidden_sizes[i], factor=factor))

        # 如果 readout_type 是 "project"，则初始化 readout projects
        if config.readout_type == "project":
            self.readout_projects = nn.ModuleList()  # 初始化 readout projects 模块列表
            hidden_size = _get_backbone_hidden_size(config)  # 获取 backbone 隐藏层大小
            for _ in range(len(config.neck_hidden_sizes)):
                self.readout_projects.append(
                    nn.Sequential(nn.Linear(2 * hidden_size, hidden_size), ACT2FN[config.hidden_act])
                )
    def forward(self, hidden_states: List[torch.Tensor], patch_height=None, patch_width=None) -> List[torch.Tensor]:
        """
        Args:
            hidden_states (`List[torch.FloatTensor]`, each of shape `(batch_size, sequence_length + 1, hidden_size)`):
                List of hidden states from the backbone.
        """
        out = []  # 用于存储每个阶段处理后的输出

        for i, hidden_state in enumerate(hidden_states):
            if i not in self.neck_ignore_stages:
                # 将隐藏状态重塑为(batch_size, num_channels, height, width)
                cls_token, hidden_state = hidden_state[:, 0], hidden_state[:, 1:]
                batch_size, sequence_length, num_channels = hidden_state.shape
                
                # 根据输入的patch_height和patch_width或者自动计算的size，重塑隐藏状态的形状
                if patch_height is not None and patch_width is not None:
                    hidden_state = hidden_state.reshape(batch_size, patch_height, patch_width, num_channels)
                else:
                    size = int(math.sqrt(sequence_length))
                    hidden_state = hidden_state.reshape(batch_size, size, size, num_channels)
                hidden_state = hidden_state.permute(0, 3, 1, 2).contiguous()  # 将维度顺序调整为(batch_size, num_channels, height, width)

                feature_shape = hidden_state.shape
                
                if self.config.readout_type == "project":
                    # 将隐藏状态展平为(batch_size, height*width, num_channels)，并调整维度顺序
                    hidden_state = hidden_state.flatten(2).permute((0, 2, 1))
                    readout = cls_token.unsqueeze(1).expand_as(hidden_state)
                    # 将读出向量连接到隐藏状态并投影
                    hidden_state = self.readout_projects[i](torch.cat((hidden_state, readout), -1))
                    # 将隐藏状态形状调整回(batch_size, num_channels, height, width)
                    hidden_state = hidden_state.permute(0, 2, 1).reshape(feature_shape)
                elif self.config.readout_type == "add":
                    # 将隐藏状态展平并加上CLS标记后，重新调整形状
                    hidden_state = hidden_state.flatten(2) + cls_token.unsqueeze(-1)
                    hidden_state = hidden_state.reshape(feature_shape)
                
                # 经过特定阶段的层处理后的隐藏状态
                hidden_state = self.layers[i](hidden_state)
                
            out.append(hidden_state)  # 将处理后的隐藏状态添加到输出列表中

        return out
# 根据配置获取骨干网络的隐藏层大小
def _get_backbone_hidden_size(config):
    # 如果配置中包含骨干网络配置，并且不是混合模式，则返回骨干网络配置的隐藏层大小
    if config.backbone_config is not None and config.is_hybrid is False:
        return config.backbone_config.hidden_size
    else:
        # 否则返回配置中的隐藏层大小
        return config.hidden_size


class DPTReassembleLayer(nn.Module):
    def __init__(self, config, channels, factor):
        super().__init__()
        # 投影层，将输入通道数为骨干网络隐藏层大小，输出通道数为指定的通道数
        hidden_size = _get_backbone_hidden_size(config)
        self.projection = nn.Conv2d(in_channels=hidden_size, out_channels=channels, kernel_size=1)

        # 根据因子进行上/下采样
        if factor > 1:
            # 上采样：使用转置卷积进行尺寸扩展
            self.resize = nn.ConvTranspose2d(channels, channels, kernel_size=factor, stride=factor, padding=0)
        elif factor == 1:
            # 不变：恒等映射
            self.resize = nn.Identity()
        elif factor < 1:
            # 下采样：使用普通卷积进行尺寸缩小
            self.resize = nn.Conv2d(channels, channels, kernel_size=3, stride=int(1 / factor), padding=1)

    def forward(self, hidden_state):
        # 将输入的隐藏状态通过投影层进行特征映射
        hidden_state = self.projection(hidden_state)
        # 将映射后的特征通过尺寸调整层进行尺寸调整
        hidden_state = self.resize(hidden_state)
        return hidden_state


class DPTFeatureFusionStage(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList()
        # 根据配置中的隐藏层大小列表创建多个特征融合层
        for _ in range(len(config.neck_hidden_sizes)):
            self.layers.append(DPTFeatureFusionLayer(config))

    def forward(self, hidden_states):
        # 反转隐藏状态列表，从最后一个开始处理
        hidden_states = hidden_states[::-1]

        fused_hidden_states = []
        # 第一个融合层仅使用最后一个隐藏状态
        fused_hidden_state = self.layers[0](hidden_states[0])
        fused_hidden_states.append(fused_hidden_state)
        # 从倒数第二层开始循环到第二层
        for hidden_state, layer in zip(hidden_states[1:], self.layers[1:]):
            # 使用当前融合层对前一个融合状态和当前隐藏状态进行融合
            fused_hidden_state = layer(fused_hidden_state, hidden_state)
            fused_hidden_states.append(fused_hidden_state)

        return fused_hidden_states


class DPTPreActResidualLayer(nn.Module):
    """
    预激活残差层，即ResidualConvUnit。

    Args:
        config (`[DPTConfig]`):
            定义模型架构的模型配置类。
    """
    # 初始化函数，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()

        # 根据配置确定是否使用批量归一化
        self.use_batch_norm = config.use_batch_norm_in_fusion_residual
        # 根据配置确定是否在融合残差中使用偏置，若配置为None则根据是否使用批量归一化来确定
        use_bias_in_fusion_residual = (
            config.use_bias_in_fusion_residual
            if config.use_bias_in_fusion_residual is not None
            else not self.use_batch_norm
        )

        # 第一个激活函数层使用ReLU激活函数
        self.activation1 = nn.ReLU()
        # 第一个卷积层，输入输出大小相同，使用指定的卷积核大小、步长和填充，并根据前面的配置确定是否使用偏置
        self.convolution1 = nn.Conv2d(
            config.fusion_hidden_size,
            config.fusion_hidden_size,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=use_bias_in_fusion_residual,
        )

        # 第二个激活函数层使用ReLU激活函数
        self.activation2 = nn.ReLU()
        # 第二个卷积层，输入输出大小相同，使用指定的卷积核大小、步长和填充，并根据前面的配置确定是否使用偏置
        self.convolution2 = nn.Conv2d(
            config.fusion_hidden_size,
            config.fusion_hidden_size,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=use_bias_in_fusion_residual,
        )

        # 如果配置中指定使用批量归一化，则创建批量归一化层
        if self.use_batch_norm:
            self.batch_norm1 = nn.BatchNorm2d(config.fusion_hidden_size)
            self.batch_norm2 = nn.BatchNorm2d(config.fusion_hidden_size)

    # 前向传播函数，接收一个张量作为输入，返回一个张量作为输出
    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        # 将输入张量作为残差保存下来
        residual = hidden_state
        # 对输入张量应用第一个ReLU激活函数
        hidden_state = self.activation1(hidden_state)

        # 将经过第一个ReLU激活函数的张量输入第一个卷积层中进行卷积操作
        hidden_state = self.convolution1(hidden_state)

        # 如果配置中指定使用批量归一化，则将经过第一个卷积层后的张量输入第一个批量归一化层中进行归一化操作
        if self.use_batch_norm:
            hidden_state = self.batch_norm1(hidden_state)

        # 对经过第一个ReLU激活函数和可能的批量归一化后的张量应用第二个ReLU激活函数
        hidden_state = self.activation2(hidden_state)
        # 将经过第二个ReLU激活函数的张量输入第二个卷积层中进行卷积操作
        hidden_state = self.convolution2(hidden_state)

        # 如果配置中指定使用批量归一化，则将经过第二个卷积层后的张量输入第二个批量归一化层中进行归一化操作
        if self.use_batch_norm:
            hidden_state = self.batch_norm2(hidden_state)

        # 将经过所有层操作后的张量与保存的残差相加作为最终的输出张量
        return hidden_state + residual
        """
        Inputs:
            hidden_state (`torch.Tensor`):
                The input tensor representing the feature maps.
            residual (`torch.Tensor`, *optional*):
                The tensor representing residual feature maps from previous stages.
                Default is `None`.
        
        Returns:
            `torch.Tensor`: The output tensor after feature fusion.
        
        Raises:
            None
        """
        if residual is not None:
            # Resize the residual tensor if its shape doesn't match `hidden_state`
            if hidden_state.shape != residual.shape:
                residual = nn.functional.interpolate(
                    residual, size=(hidden_state.shape[2], hidden_state.shape[3]), mode="bilinear", align_corners=False
                )
            # Add residual to `hidden_state` using the first residual layer
            hidden_state = hidden_state + self.residual_layer1(residual)
        
        # Process `hidden_state` through the second residual layer
        hidden_state = self.residual_layer2(hidden_state)
        
        # Upsample `hidden_state` using bilinear interpolation
        hidden_state = nn.functional.interpolate(
            hidden_state, scale_factor=2, mode="bilinear", align_corners=self.align_corners
        )
        
        # Project `hidden_state` using a 1x1 convolution
        hidden_state = self.projection(hidden_state)
        
        return hidden_state
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            # 输入参数，表示像素值的张量，形状为(batch_size, num_channels, height, width)。
            # 像素值可以通过 `AutoImageProcessor` 获取。详见 `DPTImageProcessor.__call__` 的说明。

        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            # 可选参数，用于掩盖自注意力模块中选择的头部的掩码。掩码值范围在 `[0, 1]`：

            # - 1 表示头部 **未被掩码**，
            # - 0 表示头部 **被掩码**。

        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。返回的张量中的 `attentions` 字段有更多细节。

        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。返回的张量中的 `hidden_states` 字段有更多细节。

        return_dict (`bool`, *optional*):
            # 是否返回 [`~file_utils.ModelOutput`] 而不是普通元组。
"""
定义一个 DPT 模型，该模型是 DPTPreTrainedModel 的子类，用于输出原始的隐藏状态，没有额外的特定输出头部。

:param config: 模型的配置对象，包含了模型的各种参数和设置
:param add_pooling_layer: 是否添加池化层，默认为 True
"""
class DPTModel(DPTPreTrainedModel):

    def __init__(self, config, add_pooling_layer=True):
        # 调用父类的初始化方法，传入配置对象
        super().__init__(config)
        self.config = config

        # 根据配置选择使用混合模式或者普通模式的 ViT 嵌入层
        if config.is_hybrid:
            self.embeddings = DPTViTHybridEmbeddings(config)
        else:
            self.embeddings = DPTViTEmbeddings(config)
        
        # 创建 ViT 编码器
        self.encoder = DPTViTEncoder(config)

        # LayerNorm 层，用于归一化隐藏状态向量
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # 如果指定添加池化层，则创建 DPTViTPooler 对象
        self.pooler = DPTViTPooler(config) if add_pooling_layer else None

        # 初始化模型权重并应用最终处理
        self.post_init()

    def get_input_embeddings(self):
        # 如果是混合模式，则返回整个 embeddings 对象；否则返回 patch_embeddings 属性
        if self.config.is_hybrid:
            return self.embeddings
        else:
            return self.embeddings.patch_embeddings

    def _prune_heads(self, heads_to_prune):
        """
        修剪模型中的注意力头部。

        :param heads_to_prune: 要修剪的头部的字典，格式为 {层编号: 要修剪的头部列表}
        """
        for layer, heads in heads_to_prune.items():
            # 遍历每个层中要修剪的头部列表，并调用 attention.prune_heads 进行修剪
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ):
    """
    DPT 模型的前向传播方法，处理输入数据并返回相应的输出。

    :param pixel_values: 输入的像素值张量，大小为 [batch_size, num_channels, height, width]
    :param head_mask: 可选参数，用于掩盖某些注意力头部的掩码张量
    :param output_attentions: 可选参数，是否输出注意力权重
    :param output_hidden_states: 可选参数，是否输出所有隐藏状态
    :param return_dict: 可选参数，是否以字典形式返回输出
    :return: 模型的输出，具体格式根据参数决定
    """
        ) -> Union[Tuple, BaseModelOutputWithPoolingAndIntermediateActivations]:
        # 如果输出注意力分布未指定，则使用配置中的默认设置
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果输出隐藏状态未指定，则使用配置中的默认设置
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果返回字典未指定，则使用配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 准备头部掩码（head mask），如果需要的话
        # 在头部掩码中，1.0 表示保留该头部
        # attention_probs 的形状为 bsz x n_heads x N x N
        # 输入的 head_mask 的形状为 [num_heads] 或 [num_hidden_layers x num_heads]
        # head_mask 被转换为形状 [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # 将像素值嵌入到模型的嵌入层中
        embedding_output = self.embeddings(pixel_values, return_dict=return_dict)

        # 提取嵌入层的最后隐藏状态
        embedding_last_hidden_states = embedding_output[0] if not return_dict else embedding_output.last_hidden_states

        # 将最后隐藏状态输入编码器（encoder）中进行编码
        encoder_outputs = self.encoder(
            embedding_last_hidden_states,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 提取编码器输出的序列输出
        sequence_output = encoder_outputs[0]

        # 对序列输出进行 LayerNormalization
        sequence_output = self.layernorm(sequence_output)

        # 如果存在池化器（pooler），则对序列输出进行池化
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        # 如果未使用返回字典，则返回编码器的输出和池化器的输出（如果有）
        if not return_dict:
            head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
            return head_outputs + encoder_outputs[1:] + embedding_output[1:]

        # 如果使用返回字典，则构造返回的 BaseModelOutputWithPoolingAndIntermediateActivations 对象
        return BaseModelOutputWithPoolingAndIntermediateActivations(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            intermediate_activations=embedding_output.intermediate_activations,
        )
# Copied from transformers.models.vit.modeling_vit.ViTPooler with ViT->DPT
class DPTViTPooler(nn.Module):
    def __init__(self, config: DPTConfig):
        super().__init__()
        # Initialize a linear transformation layer for pooling
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # Define activation function for the pooled output
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # Extract the hidden state corresponding to the first token for pooling
        first_token_tensor = hidden_states[:, 0]
        # Pass through linear layer for pooling
        pooled_output = self.dense(first_token_tensor)
        # Apply activation function to the pooled output
        pooled_output = self.activation(pooled_output)
        # Return the pooled output
        return pooled_output


class DPTNeck(nn.Module):
    """
    DPTNeck. A neck is a module that is normally used between the backbone and the head. It takes a list of tensors as
    input and produces another list of tensors as output. For DPT, it includes 2 stages:

    * DPTReassembleStage
    * DPTFeatureFusionStage.

    Args:
        config (dict): config dict.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Postprocessing stage: reassemble_stage handles reassembling the hidden states
        # depending on the backbone type (only required for non-hierarchical backbones)
        if config.backbone_config is not None and config.backbone_config.model_type in ["swinv2"]:
            self.reassemble_stage = None
        else:
            self.reassemble_stage = DPTReassembleStage(config)

        # Initialize a list of convolutional layers for fusion
        self.convs = nn.ModuleList()
        for channel in config.neck_hidden_sizes:
            self.convs.append(nn.Conv2d(channel, config.fusion_hidden_size, kernel_size=3, padding=1, bias=False))

        # Fusion stage: feature fusion stage for combining processed features
        self.fusion_stage = DPTFeatureFusionStage(config)

    def forward(self, hidden_states: List[torch.Tensor], patch_height=None, patch_width=None) -> List[torch.Tensor]:
        """
        Args:
            hidden_states (`List[torch.FloatTensor]`, each of shape `(batch_size, sequence_length, hidden_size)` or `(batch_size, hidden_size, height, width)`):
                List of hidden states from the backbone.
        """
        if not isinstance(hidden_states, (tuple, list)):
            raise ValueError("hidden_states should be a tuple or list of tensors")

        if len(hidden_states) != len(self.config.neck_hidden_sizes):
            raise ValueError("The number of hidden states should be equal to the number of neck hidden sizes.")

        # Postprocess hidden states if reassemble_stage is defined
        if self.reassemble_stage is not None:
            hidden_states = self.reassemble_stage(hidden_states, patch_height, patch_width)

        # Apply convolutional layers to each hidden state in hidden_states
        features = [self.convs[i](feature) for i, feature in enumerate(hidden_states)]

        # Apply fusion stage to combine processed features
        output = self.fusion_stage(features)

        # Return the fused output
        return output


class DPTDepthEstimationHead(nn.Module):
    """
    Output head head consisting of 3 convolutional layers. It progressively halves the feature dimension and upsamples
    the predictions to the input resolution after the first convolutional layer (details can be found in the paper's
    """

    # Note: The class DPTDepthEstimationHead is not fully provided in the given snippet.
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()

        # 保存传入的配置参数
        self.config = config

        # 初始化投影层为 None
        self.projection = None
        # 如果配置要求添加投影层，则创建一个卷积层作为投影层
        if config.add_projection:
            self.projection = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        # 从配置中获取特征的大小
        features = config.fusion_hidden_size
        # 定义神经网络的前向结构
        self.head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),  # 第一个卷积层
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),      # 上采样层
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),       # 第二个卷积层
            nn.ReLU(),                                                             # ReLU 激活函数
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),                   # 第三个卷积层
            nn.ReLU(),                                                             # ReLU 激活函数
        )

    def forward(self, hidden_states: List[torch.Tensor]) -> torch.Tensor:
        # 使用索引获取最后的隐藏状态
        hidden_states = hidden_states[self.config.head_in_index]

        # 如果投影层不为 None，则将隐藏状态投影到新的空间，并应用 ReLU 激活函数
        if self.projection is not None:
            hidden_states = self.projection(hidden_states)
            hidden_states = nn.ReLU()(hidden_states)

        # 通过神经网络头部处理隐藏状态，生成深度预测
        predicted_depth = self.head(hidden_states)

        # 去除预测结果中的通道维度，使得结果为二维张量
        predicted_depth = predicted_depth.squeeze(dim=1)

        # 返回预测的深度张量
        return predicted_depth
"""
DPT Model with a depth estimation head on top (consisting of 3 convolutional layers) e.g. for KITTI, NYUv2.
"""
# 导入所需的库和模块
@add_start_docstrings(
    """
    Add docstring for model initialization with DPT-specific documentation.
    """,
    DPT_START_DOCSTRING,
)
# 定义 DPTForDepthEstimation 类，继承自 DPTPreTrainedModel
class DPTForDepthEstimation(DPTPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 根据配置加载背景骨干网络或创建 DPT 模型
        self.backbone = None
        if config.backbone_config is not None and config.is_hybrid is False:
            self.backbone = load_backbone(config)
        else:
            self.dpt = DPTModel(config, add_pooling_layer=False)

        # 初始化 DPTNeck 和 DPTDepthEstimationHead
        self.neck = DPTNeck(config)
        self.head = DPTDepthEstimationHead(config)

        # 执行初始化权重和最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(DPT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=DepthEstimatorOutput, config_class=_CONFIG_FOR_DOC)
    # 实现前向传播函数，接受像素值、头部掩码、标签等参数
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        head_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
):
        """
        Perform forward pass of the DPT model for depth estimation.
        """
        # 实现模型的前向计算过程，生成深度估计输出
        # ...
        pass

"""
DPT Model with a semantic segmentation head on top e.g. for ADE20k, CityScapes.
"""
# 定义 DPTForSemanticSegmentation 类，继承自 DPTPreTrainedModel
@add_start_docstrings(
    """
    Add docstring for model initialization with DPT-specific documentation.
    """,
    DPT_START_DOCSTRING,
)
class DPTForSemanticSegmentation(DPTPreTrainedModel):
    # 初始化函数，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法，传入配置对象
        super().__init__(config)

        # 创建一个 DPTModel 的实例，不添加池化层
        self.dpt = DPTModel(config, add_pooling_layer=False)

        # 创建一个 DPTNeck 的实例作为模型的颈部（neck）
        self.neck = DPTNeck(config)

        # 创建一个 DPTSemanticSegmentationHead 的实例作为模型的主要分割头（head）
        self.head = DPTSemanticSegmentationHead(config)

        # 如果配置允许使用辅助头部（auxiliary_head），则创建一个 DPTAuxiliaryHead 的实例
        self.auxiliary_head = DPTAuxiliaryHead(config) if config.use_auxiliary_head else None

        # 执行初始化权重和最终处理
        self.post_init()

    # 前向传播函数的装饰器，添加模型输入的文档字符串
    @add_start_docstrings_to_model_forward(DPT_INPUTS_DOCSTRING)
    # 替换前向传播函数返回值的文档字符串，输出类型为 SemanticSegmenterOutput，配置类为 _CONFIG_FOR_DOC
    @replace_return_docstrings(output_type=SemanticSegmenterOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
```