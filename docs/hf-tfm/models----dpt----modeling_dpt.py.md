# `.\models\dpt\modeling_dpt.py`

```py
# 设置文件编码为utf-8
# 版权声明
# 根据 Apache 许可证版本 2.0 授权
# 除非适用法律要求或书面同意，否则软件在"原样"基础上分发，无论是表示或暗示的任何保证或条件
# 查看许可证获取更多细节

""" PyTorch DPT (Dense Prediction Transformers) model.

This implementation is heavily inspired by OpenMMLab's implementation, found here:
https://github.com/open-mmlab/mmsegmentation/blob/master/mmseg/models/decode_heads/dpt_head.py.

"""

# 导入模块
import collections.abc
import math
from dataclasses import dataclass
from typing import List, Optional, Set, Tuple, Union

# 导入torch相关模块
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

# 导入自定义模块
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

# 设置日志
logger = logging.get_logger(__name__)

# 通用文档字符串
_CONFIG_FOR_DOC = "DPTConfig"

# 基本文档字符串
_CHECKPOINT_FOR_DOC = "Intel/dpt-large"
_EXPECTED_OUTPUT_SHAPE = [1, 577, 1024]

# 预训练模型列表
DPT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "Intel/dpt-large",
    "Intel/dpt-hybrid-midas",
    # 查看所有 DPT 模型 https://huggingface.co/models?filter=dpt
]

# 声明数据类，包含模型的输出以及可用于后续阶段的中间激活
@dataclass
class BaseModelOutputWithIntermediateActivations(ModelOutput):
    """
    Base class for model's outputs that also contains intermediate activations that can be used at later stages. Useful
    in the context of Vision models.:

    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        intermediate_activations (`tuple(torch.FloatTensor)`, *optional*):
            Intermediate activations that can be used to compute hidden states of the model at various layers.
    """
    last_hidden_states: torch.FloatTensor = None
    intermediate_activations: Optional[Tuple[torch.FloatTensor]] = None

# 声明数据类，包含模型的输出、池化以及中间激活
@dataclass
class BaseModelOutputWithPoolingAndIntermediateActivations(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states as well as intermediate
    # 定义函数的参数说明文档，说明了函数所接受的参数及其类型和形状
    
    last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
        # 表示最后一个隐藏层的隐藏状态，是一个三维张量，形状为(batch_size, sequence_length, hidden_size)
    
    pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size)`):
        # 表示经过模型辅助预训练任务后的序列的第一个标记（分类标记）的最后一层隐藏状态，是一个二维张量，形状为(batch_size, hidden_size)
    
    hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
        # 表示模型在每一层的隐藏状态组成的元组，如果模型有嵌入层，则还包括嵌入层的输出，当传递`output_hidden_states=True`或`config.output_hidden_states=True`时返回
    
    attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
        # 表示注意力权重组成的元组，每层一个张量，形状为(batch_size, num_heads, sequence_length, sequence_length)，在自注意力头中用来计算加权平均值的注意力权重
    
    intermediate_activations (`tuple(torch.FloatTensor)`, *optional*):
        # 表示可以用来计算模型在各个层的隐藏状态的中间激活
    
    
    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    intermediate_activations: Optional[Tuple[torch.FloatTensor]] = None
    # 定义一个名为 DPTViTHybridEmbeddings 的类，继承自 nn.Module
    class DPTViTHybridEmbeddings(nn.Module):
        """
        This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
        `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
        Transformer.
        """

        # 初始化方法，接受配置参数 config 和特征大小 feature_size
        def __init__(self, config, feature_size=None):
            # 调用父类的初始化方法
            super().__init__()
            # 从配置中获取图像大小和分块大小
            image_size, patch_size = config.image_size, config.patch_size
            # 从配置中获取通道数和隐藏层大小
            num_channels, hidden_size = config.num_channels, config.hidden_size

            # 如果图像大小和分块大小不是迭代类型，则转换为元组类型
            image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
            patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
            # 计算分块数量
            num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])

            # 使用配置中的参数创建 AutoBackbone 对象
            self.backbone = AutoBackbone.from_config(config.backbone_config)
            # 获取最后一层特征图的通道数
            feature_dim = self.backbone.channels[-1]
            # 如果 backbone 输出的特征数量不是 3，则抛出异常
            if len(config.backbone_config.out_features) != 3:
                raise ValueError(
                    f"Expected backbone to have 3 output features, got {len(config.backbone_config.out_features)}"
                )
            # 始终采用第一和第二个 backbone 阶段的输出作为残差特征图
            self.residual_feature_map_index = [0, 1]

            # 如果特征大小为 None，则从配置中获取特征图形状
            if feature_size is None:
                feat_map_shape = config.backbone_featmap_shape
                feature_size = feat_map_shape[-2:]
                feature_dim = feat_map_shape[1]
            else:
                feature_size = (
                    feature_size if isinstance(feature_size, collections.abc.Iterable) else (feature_size, feature_size)
                )
                feature_dim = self.backbone.channels[-1]

            # 初始化图像大小、分块大小和通道数
            self.image_size = image_size
            self.patch_size = patch_size[0]
            self.num_channels = num_channels

            # 使用卷积层进行特征图映射
            self.projection = nn.Conv2d(feature_dim, hidden_size, kernel_size=1)

            # 初始化分类令牌，维度为 (1, 1, hidden_size)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
            # 初始化位置嵌入，维度为 (1, num_patches + 1, hidden_size)
            self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, config.hidden_size))

        # 重新调整位置嵌入的私有方法
        def _resize_pos_embed(self, posemb, grid_size_height, grid_size_width, start_index=1):
            # 获取位置令牌
            posemb_tok = posemb[:, :start_index]
            # 获取位置网格
            posemb_grid = posemb[0, start_index:]

            # 计算原网格大小
            old_grid_size = int(math.sqrt(len(posemb_grid)))

            # 重新调整位置网格的大小
            posemb_grid = posemb_grid.reshape(1, old_grid_size, old_grid_size, -1).permute(0, 3, 1, 2)
            posemb_grid = nn.functional.interpolate(posemb_grid, size=(grid_size_height, grid_size_width), mode="bilinear")
            posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, grid_size_height * grid_size_width, -1)

            # 合并位置令牌和位置网格
            posemb = torch.cat([posemb_tok, posemb_grid], dim=1)

            return posemb

        # 前向传播方法
        def forward(
            self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False, return_dict: bool = False
    # 定义函数的返回类型为 torch.Tensor
    ) -> torch.Tensor:
        # 获取输入张量的维度信息
        batch_size, num_channels, height, width = pixel_values.shape
        # 检查通道数是否与配置中设置的通道数匹配
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        # 如果不进行位置编码的插值
        if not interpolate_pos_encoding:
            # 检查输入图像尺寸是否与模型配置所指定的尺寸相匹配
            if height != self.image_size[0] or width != self.image_size[1]:
                raise ValueError(
                    f"Input image size ({height}*{width}) doesn't match model"
                    f" ({self.image_size[0]}*{self.image_size[1]})."
                )

        # 根据输入图像尺寸和补丁大小调整位置编码
        position_embeddings = self._resize_pos_embed(
            self.position_embeddings, height // self.patch_size, width // self.patch_size
        )

        # 使用backbone模型处理像素值
        backbone_output = self.backbone(pixel_values)

        # 获取backbone模型输出的特征图
        features = backbone_output.feature_maps[-1]

        # 保存中间激活状态以便后续使用
        output_hidden_states = [backbone_output.feature_maps[index] for index in self.residual_feature_map_index]

        # 将特征映射通过投影层，并展平、转置
        embeddings = self.projection(features).flatten(2).transpose(1, 2)

        # 创建类别令牌并与embedding连接
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # 添加位置编码到每个令牌
        embeddings = embeddings + position_embeddings

        # 如果不返回字典，则返回嵌入和中间激活状态
        if not return_dict:
            return (embeddings, output_hidden_states)

            # 返回隐藏状态和中间激活状态
        return BaseModelOutputWithIntermediateActivations(
            last_hidden_states=embeddings,
            intermediate_activations=output_hidden_states,
        )
class DPTViTEmbeddings(nn.Module):
    """
    构建CLS标记、位置编码和Patch嵌入。

    """

    def __init__(self, config):
        super().__init__()

        # 定义CLS标记，形状为(1, 1, hidden_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        # 初始化Patch嵌入对象
        self.patch_embeddings = DPTViTPatchEmbeddings(config)
        # 获取Patch数量
        num_patches = self.patch_embeddings.num_patches
        # 定义位置编码，形状为(1, num_patches + 1, hidden_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, num_patches + 1, config.hidden_size))
        # 定义Dropout层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 保存配置
        self.config = config

    def _resize_pos_embed(self, posemb, grid_size_height, grid_size_width, start_index=1):
        # 提取位置编码中的Token部分
        posemb_tok = posemb[:, :start_index]
        # 提取位置编码中的Grid部分
        posemb_grid = posemb[0, start_index:]

        # 计算旧的Grid大小
        old_grid_size = int(math.sqrt(len(posemb_grid)))

        # 重塑Grid部分
        posemb_grid = posemb_grid.reshape(1, old_grid_size, old_grid_size, -1).permute(0, 3, 1, 2)
        # 插值操作调整Grid大小
        posemb_grid = nn.functional.interpolate(posemb_grid, size=(grid_size_height, grid_size_width), mode="bilinear")
        posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, grid_size_height * grid_size_width, -1)

        # 合并Token和Grid部分
        posemb = torch.cat([posemb_tok, posemb_grid], dim=1)

        return posemb

    def forward(self, pixel_values, return_dict=False):
        batch_size, num_channels, height, width = pixel_values.shape

        # 可能会对位置编码进行插值以处理不同大小的图像
        patch_size = self.config.patch_size
        position_embeddings = self._resize_pos_embed(
            self.position_embeddings, height // patch_size, width // patch_size
        )

        # 计算Patch嵌入
        embeddings = self.patch_embeddings(pixel_values)

        batch_size, seq_len, _ = embeddings.size()

        # 将[CLS]标记添加到嵌入的Patch标记中
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # 将位置编码添加到每个标记中
        embeddings = embeddings + position_embeddings

        # 应用Dropout
        embeddings = self.dropout(embeddings)

        if not return_dict:
            return (embeddings,)

        # 返回包含中间激活的BaseModelOutputWithIntermediateActivations对象
        return BaseModelOutputWithIntermediateActivations(last_hidden_states=embeddings)


class DPTViTPatchEmbeddings(nn.Module):
    """
    图像到Patch嵌入。

    """
    # 初始化函数，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 从配置对象中获取图像大小和分块大小
        image_size, patch_size = config.image_size, config.patch_size
        # 从配置对象中获取通道数和隐藏层大小
        num_channels, hidden_size = config.num_channels, config.hidden_size

        # 如果图像大小不是可迭代对象，则将其转换为元组
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        # 如果分块大小不是可迭代对象，则将其转换为元组
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        # 计算图像分块的数量
        num_patches = (image_size[1] // patch_size[1]) * (image_size[0] // patch_size[0])
        
        # 将图像大小、分块大小、通道数和分块数量存储为对象的属性
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.num_patches = num_patches

        # 创建一个卷积层，用于将图像块投影到隐藏空间
        self.projection = nn.Conv2d(num_channels, hidden_size, kernel_size=patch_size, stride=patch_size)

    # 前向传播函数，接受像素值作为输入
    def forward(self, pixel_values):
        # 获取输入张量的维度信息
        batch_size, num_channels, height, width = pixel_values.shape
        # 检查输入的通道数是否与配置中设置的通道数匹配
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        # 将输入的像素值通过投影层并展平后转置，得到嵌入向量
        embeddings = self.projection(pixel_values).flatten(2).transpose(1, 2)
        # 返回嵌入向量
        return embeddings
# 从 transformers.models.vit.modeling_vit.ViTSelfAttention 复制代码，并将 ViT 改为 DPT
class DPTViTSelfAttention(nn.Module):
    def __init__(self, config: DPTConfig) -> None:
        super().__init__()
        # 如果隐藏大小不是注意力头数的倍数，并且没有嵌入大小的属性，则引发值错误
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                f"heads {config.num_attention_heads}."
            )

        # 设置注意力头数和每个头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 初始化查询、键、值的线性层，用于生成注意力矩阵
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        # 初始化 dropout 层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    # 将输入张量转换为注意力矩阵
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    # 前向传播函数
    def forward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        # 生成混合的查询层
        mixed_query_layer = self.query(hidden_states)

        # 生成键和值的矩阵并转换为注意力矩阵形式
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # 通过"查询"和"键"的点积来获取原始注意力分数
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # 对注意力分数进行缩放
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # 将注意力分数归一化为概率
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # 这实际上是对整个令牌进行dropout，这可能看起来有些不寻常，但是源自原始Transformer论文
        attention_probs = self.dropout(attention_probs)

        # 如果需要，掩盖头部
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # 计算上下文层
        context_layer = torch.matmul(attention_probs, value_layer)

        # 转置上下文层并调整形状
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        # 返回输出
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs
# 从transformers.models.vit.modeling_vit.ViTSelfOutput复制而来，将ViT改为DPT
class DPTViTSelfOutput(nn.Module):
    """
    The residual connection is defined in DPTLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: DPTConfig) -> None:
        super().__init__()
        # 创建一个全连接层，输入和输出大小为config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 创建一个丢弃层，使用config.hidden_dropout_prob作为丢弃概率
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 全连接层的前向传播，将hidden_states作为输入
        hidden_states = self.dense(hidden_states)
        # 使用丢弃层对全连接层的输出进行丢弃操作
        hidden_states = self.dropout(hidden_states)

        return hidden_states


class DPTViTAttention(nn.Module):
    def __init__(self, config: DPTConfig) -> None:
        super().__init__()
        # 创建DPTViTSelfAttention实例
        self.attention = DPTViTSelfAttention(config)
        # 创建DPTViTSelfOutput实例
        self.output = DPTViTSelfOutput(config)
        # 存储被剪枝的注意力头
        self.pruned_heads = set()

    # 从transformers.models.vit.modeling_vit.ViTAttention.prune_heads复制而来
    def prune_heads(self, heads: Set[int]) -> None:
        if len(heads) == 0:
            return
        # 找到可以剪枝的头并相应处理线性层
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # 剪枝线性层
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储剪枝的头
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    # 从transformers.models.vit.modeling_vit.ViTAttention.forward复制而来
    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        # 调用DPTViTSelfAttention的forward方法
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)

        # 调用DPTViTSelfOutput的forward方法
        attention_output = self.output(self_outputs[0], hidden_states)

        # 返回结果，如果需要输出attentions，则返回时一并返回attentions
        outputs = (attention_output,) + self_outputs[1:]
        return outputs


# 从transformers.models.vit.modeling_vit.ViTIntermediate复制而来，将ViT改为DPT
class DPTViTIntermediate(nn.Module):
    # 初始化方法，接受一个DPTConfig类型的参数，并且没有返回值
    def __init__(self, config: DPTConfig) -> None:
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个全连接层，输入大小为config.hidden_size，输出大小为config.intermediate_size
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 如果config.hidden_act是字符串类型，将调用ACT2FN字典中对应的激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        # 否则直接使用config.hidden_act作为激活函数
        else:
            self.intermediate_act_fn = config.hidden_act

    # 前向传播方法，接受一个torch.Tensor类型的hidden_states参数，返回一个torch.Tensor类型的结果
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将输入的hidden_states通过全连接层self.dense进行线性变换
        hidden_states = self.dense(hidden_states)
        # 将线性变换后的hidden_states通过激活函数self.intermediate_act_fn进行非线性变换
        hidden_states = self.intermediate_act_fn(hidden_states)

        # 返回经过线性和非线性变换后的hidden_states
        return hidden_states
# 从transformers.models.vit.modeling_vit.ViTOutput中复制，将ViT->DPT
class DPTViTOutput(nn.Module):
    def __init__(self, config: DPTConfig) -> None:
        super().__init__()
        # 创建一个全连接层，输入尺寸为config.intermediate_size，输出尺寸为config.hidden_size
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建一个以config.hidden_dropout_prob为dropout概率的dropout层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 全连接层的前向传播，将hidden_states映射到config.hidden_size维度
        hidden_states = self.dense(hidden_states)
        # dropout层的前向传播
        hidden_states = self.dropout(hidden_states)

        # 返回加上input_tensor后的hidden_states
        hidden_states = hidden_states + input_tensor

        return hidden_states


# 从transformers.models.vit.modeling_vit.ViTLayer中复制，将ViTConfig->DPTConfig, ViTAttention->DPTViTAttention, ViTIntermediate->DPTViTIntermediate, ViTOutput->DPTViTOutput
class DPTViTLayer(nn.Module):
    """这对应于timm实现中的Block类。"""

    def __init__(self, config: DPTConfig) -> None:
        super().__init__()
        # 设置用于feed forward的chunk大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 定义一个维度为1的序列长度
        self.seq_len_dim = 1
        # 初始化self.attention为DPTViTAttention，self.intermediate为DPTViTIntermediate，self.output为DPTViTOutput
        # 通过ViT的config初始化这些模块
        self.attention = DPTViTAttention(config)
        self.intermediate = DPTViTIntermediate(config)
        self.output = DPTViTOutput(config)
        # 初始化两个LayerNorm层
        # layernorm_before在ViT中应用于self-attention前，layernorm_after在ViT中也应用于self-attention后
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        # 使用self-attention输出更新hidden_states，返回的outputs包括self-attention输出及其它信息
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # 在ViT中，layernorm应用于self-attention前
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # 如果输出注意力权重，添加self attention

        # 第一个残差连接
        hidden_states = attention_output + hidden_states

        # 在ViT中，layernorm也应用于self-attention后
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)

        # 这里执行第二个残差连接
        layer_output = self.output(layer_output, hidden_states)

        outputs = (layer_output,) + outputs

        return outputs


# 从transformers.models.vit.modeling_vit.ViTEncoder中复制，将ViTConfig -> DPTConfig, ViTLayer->DPTViTLayer
class DPTViTEncoder(nn.Module):
    def __init__(self, config: DPTConfig) -> None:
        super().__init__()
        self.config = config
        # 创建一个由DPTViTLayer实例组成的列表，长度为config.num_hidden_layers
        self.layer = nn.ModuleList([DPTViTLayer(config) for _ in range(config.num_hidden_layers)])
        # 是否使用渐变检查点
        self.gradient_checkpointing = False
    # Transformer模型的前向传递函数，处理输入的隐藏状态并生成输出，同时支持可选的注意力和隐藏状态输出
    def forward(
        # 输入的隐藏状态张量
        hidden_states: torch.Tensor,
        # 可选的头掩码张量，控制各个注意力头
        head_mask: Optional[torch.Tensor] = None,
        # 是否输出注意力权重
        output_attentions: bool = False,
        # 是否输出隐藏状态
        output_hidden_states: bool = False,
        # 是否以字典形式返回结果
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        # 如果需要输出隐藏状态，则初始化为空元组，否则为 None
        all_hidden_states = () if output_hidden_states else None
        # 如果需要输出注意力权重，则初始化为空元组，否则为 None
        all_self_attentions = () if output_attentions else None
    
        # 遍历所有 Transformer 层
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，则将当前的隐藏状态加入 all_hidden_states
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
    
            # 从 head_mask 中获取当前层的掩码，如果 head_mask 是 None 则设为 None
            layer_head_mask = head_mask[i] if head_mask is not None else None
    
            # 如果启用了梯度检查点并且当前处于训练模式，使用梯度检查点功能
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,  # 调用当前层的模块
                    hidden_states,  # 输入隐藏状态
                    layer_head_mask,  # 当前层的掩码
                    output_attentions,  # 是否输出注意力权重
                )
            else:
                # 否则直接调用当前层模块的 forward 函数
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)
    
            # 更新隐藏状态为当前层的输出
            hidden_states = layer_outputs[0]
    
            # 如果需要输出注意力权重，则将当前层的注意力权重加入 all_self_attentions
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
    
        # 在循环结束后，如果需要输出隐藏状态，则将最终隐藏状态加入 all_hidden_states
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
    
        # 如果不需要返回字典形式，则返回包含必要输出的元组
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        
        # 否则返回 BaseModelOutput，包含最后的隐藏状态、所有隐藏状态和注意力权重
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

        self.config = config
        self.layers = nn.ModuleList()  # 初始化层列表

        # 根据是否为混合模型初始化重新组装层
        if config.is_hybrid:
            self._init_reassemble_dpt_hybrid(config)  # 调用混合模型初始化方法
        else:
            self._init_reassemble_dpt(config)  # 调用非混合模型初始化方法

        self.neck_ignore_stages = config.neck_ignore_stages  # 设置忽略的重新组装阶段

    def _init_reassemble_dpt_hybrid(self, config):
        r""" "
        For DPT-Hybrid the first 2 reassemble layers are set to `nn.Identity()`, please check the official
        implementation: https://github.com/isl-org/DPT/blob/f43ef9e08d70a752195028a51be5e1aff227b913/dpt/vit.py#L438
        for more details.
        """
        for i, factor in zip(range(len(config.neck_hidden_sizes)), config.reassemble_factors):
            if i <= 1:
                self.layers.append(nn.Identity())  # 将前两个重新组装层设置为身份映射
            elif i > 1:
                self.layers.append(DPTReassembleLayer(config, channels=config.neck_hidden_sizes[i], factor=factor))  # 添加重新组装层

        if config.readout_type != "project":
            raise ValueError(f"Readout type {config.readout_type} is not supported for DPT-Hybrid.")

        # 当使用 DPT-Hybrid 时，将读出类型设置为 "project"。在配置文件上进行健全性检查
        self.readout_projects = nn.ModuleList()  # 初始化读出层列表
        hidden_size = _get_backbone_hidden_size(config)  # 获取骨干网络的隐藏层尺寸
        for i in range(len(config.neck_hidden_sizes)):
            if i <= 1:
                self.readout_projects.append(nn.Sequential(nn.Identity()))  # 前两个读出层设置为身份映射
            elif i > 1:
                self.readout_projects.append(
                    nn.Sequential(nn.Linear(2 * hidden_size, hidden_size), ACT2FN[config.hidden_act])  # 添加读出层
                )

    def _init_reassemble_dpt(self, config):
        for i, factor in zip(range(len(config.neck_hidden_sizes)), config.reassemble_factors):
            self.layers.append(DPTReassembleLayer(config, channels=config.neck_hidden_sizes[i], factor=factor))  # 添加重新组装层

        if config.readout_type == "project":
            self.readout_projects = nn.ModuleList()  # 初始化读出层列表
            hidden_size = _get_backbone_hidden_size(config)  # 获取骨干网络的隐藏层尺寸
            for _ in range(len(config.neck_hidden_sizes)):
                self.readout_projects.append(
                    nn.Sequential(nn.Linear(2 * hidden_size, hidden_size), ACT2FN[config.hidden_act])  # 添加读出层
                )
    # 定义前向传播方法，接受隐藏状态列表和可选的补丁高度、宽度作为输入，并返回更新后的隐藏状态列表
    def forward(self, hidden_states: List[torch.Tensor], patch_height=None, patch_width=None) -> List[torch.Tensor]:
        """
        Args:
            hidden_states (`List[torch.FloatTensor]`, each of shape `(batch_size, sequence_length + 1, hidden_size)`):
                List of hidden states from the backbone.
        """
        # 初始化输出列表
        out = []

        # 遍历隐藏状态列表
        for i, hidden_state in enumerate(hidden_states):
            # 如果当前索引不在忽略的阶段列表中
            if i not in self.neck_ignore_stages:
                # 从隐藏状态中分离出类别令牌和其他隐藏状态
                cls_token, hidden_state = hidden_state[:, 0], hidden_state[:, 1:]
                # 获取批次大小、序列长度和通道数
                batch_size, sequence_length, num_channels = hidden_state.shape
                # 如果指定了补丁高度和宽度，则重新调整隐藏状态的形状
                if patch_height is not None and patch_width is not None:
                    hidden_state = hidden_state.reshape(batch_size, patch_height, patch_width, num_channels)
                else:
                    size = int(math.sqrt(sequence_length))
                    hidden_state = hidden_state.reshape(batch_size, size, size, num_channels)
                # 将通道维度置于第二个位置，并使得存储在连续内存中
                hidden_state = hidden_state.permute(0, 3, 1, 2).contiguous()

                # 记录调整后的特征形状
                feature_shape = hidden_state.shape
                
                # 根据读出类型进行相应的处理
                if self.config.readout_type == "project":
                    # 将隐藏状态展平为 (batch_size, height*width, num_channels)
                    hidden_state = hidden_state.flatten(2).permute((0, 2, 1))
                    # 将类别令牌扩展并连接到隐藏状态，并进行投影
                    readout = cls_token.unsqueeze(1).expand_as(hidden_state)
                    hidden_state = self.readout_projects[i](torch.cat((hidden_state, readout), -1))
                    # 重新调整隐藏状态的形状为 (batch_size, num_channels, height, width)
                    hidden_state = hidden_state.permute(0, 2, 1).reshape(feature_shape)
                elif self.config.readout_type == "add":
                    # 将类别令牌加到隐藏状态上
                    hidden_state = hidden_state.flatten(2) + cls_token.unsqueeze(-1)
                    hidden_state = hidden_state.reshape(feature_shape)
                # 经过读出操作后，将隐藏状态传递给相应的图层
                hidden_state = self.layers[i](hidden_state)
            # 将处理后的隐藏状态添加到输出列表中
            out.append(hidden_state)

        # 返回更新后的隐藏状态列表
        return out
def _get_backbone_hidden_size(config):
    # 如果配置中包含了骨干网络配置且不是混合模式，则返回骨干网络配置中的隐藏层大小
    if config.backbone_config is not None and config.is_hybrid is False:
        return config.backbone_config.hidden_size
    else:
        # 否则返回配置中的隐藏层大小
        return config.hidden_size


class DPTReassembleLayer(nn.Module):
    def __init__(self, config, channels, factor):
        super().__init__()
        # 投影层，用于将输入的隐藏状态转换为指定通道数
        hidden_size = _get_backbone_hidden_size(config)
        self.projection = nn.Conv2d(in_channels=hidden_size, out_channels=channels, kernel_size=1)

        # 根据因子进行上/下采样
        if factor > 1:
            # 如果因子大于1，则使用转置卷积进行上采样
            self.resize = nn.ConvTranspose2d(channels, channels, kernel_size=factor, stride=factor, padding=0)
        elif factor == 1:
            # 如果因子等于1，则保持不变
            self.resize = nn.Identity()
        elif factor < 1:
            # 如果因子小于1，则进行下采样
            # 下采样采用卷积层
            self.resize = nn.Conv2d(channels, channels, kernel_size=3, stride=int(1 / factor), padding=1)

    def forward(self, hidden_state):
        # 首先通过投影层处理隐藏状态
        hidden_state = self.projection(hidden_state)
        # 然后通过上/下采样层处理结果
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
        # 将隐藏状态列表倒序，从最后一个开始处理
        hidden_states = hidden_states[::-1]

        fused_hidden_states = []
        # 第一层只使用最后一个隐藏状态
        fused_hidden_state = self.layers[0](hidden_states[0])
        fused_hidden_states.append(fused_hidden_state)
        # 从倒数第二层开始循环处理
        for hidden_state, layer in zip(hidden_states[1:], self.layers[1:]):
            # 使用特征融合层融合隐藏状态
            fused_hidden_state = layer(fused_hidden_state, hidden_state)
            fused_hidden_states.append(fused_hidden_state)

        return fused_hidden_states


class DPTPreActResidualLayer(nn.Module):
    """
    ResidualConvUnit, pre-activate residual unit.

    Args:
        config (`[DPTConfig]`):
            Model configuration class defining the model architecture.
    """
    # 初始化函数，接收配置参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()

        # 根据配置参数决定是否使用 Batch Normalization
        self.use_batch_norm = config.use_batch_norm_in_fusion_residual
        # 根据配置参数决定是否在融合残差中使用偏置
        use_bias_in_fusion_residual = (
            config.use_bias_in_fusion_residual
            if config.use_bias_in_fusion_residual is not None
            else not self.use_batch_norm
        )

        # 初始化第一个激活函数ReLU
        self.activation1 = nn.ReLU()
        # 初始化第一个卷积层
        self.convolution1 = nn.Conv2d(
            config.fusion_hidden_size,
            config.fusion_hidden_size,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=use_bias_in_fusion_residual,
        )

        # 初始化第二个激活函数ReLU
        self.activation2 = nn.ReLU()
        # 初始化第二个卷积层
        self.convolution2 = nn.Conv2d(
            config.fusion_hidden_size,
            config.fusion_hidden_size,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=use_bias_in_fusion_residual,
        )

        # 如果使用 Batch Normalization，则初始化 Batch Normalization 层
        if self.use_batch_norm:
            self.batch_norm1 = nn.BatchNorm2d(config.fusion_hidden_size)
            self.batch_norm2 = nn.BatchNorm2d(config.fusion_hidden_size)

    # 前向传播函数，接收隐藏状态并返回计算结果
    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        # 将隐藏状态作为残差保存
        residual = hidden_state
        # 第一个激活函数ReLU
        hidden_state = self.activation1(hidden_state)

        # 第一个卷积层
        hidden_state = self.convolution1(hidden_state)

        # 如果使用 Batch Normalization，则应用 Batch Normalization
        if self.use_batch_norm:
            hidden_state = self.batch_norm1(hidden_state)

        # 第二个激活函数ReLU
        hidden_state = self.activation2(hidden_state)
        # 第二个卷积层
        hidden_state = self.convolution2(hidden_state)

        # 如果使用 Batch Normalization，则应用 Batch Normalization
        if self.use_batch_norm:
            hidden_state = self.batch_norm2(hidden_state)

        # 返回结果与原始隐藏状态的和，作为最终输出
        return hidden_state + residual
# 注释


class DPTFeatureFusionLayer(nn.Module):
    """Feature fusion layer, merges feature maps from different stages.

    Args:
        config (`[DPTConfig]`):
            Model configuration class defining the model architecture.
        align_corners (`bool`, *optional*, defaults to `True`):
            The align_corner setting for bilinear upsample.
    """
# DPTFeatureFusionLayer类用于特征融合，将不同阶段的特征映射进行融合。
# Args参数：
#     config（`[DPTConfig]`）：模型配置类，定义了模型的架构。
#     align_corners（`bool`，*可选参数*，默认为`True`）：双线性上采样的对齐角设置。

    def __init__(self, config, align_corners=True):
        super().__init__()

        self.align_corners = align_corners

        self.projection = nn.Conv2d(config.fusion_hidden_size, config.fusion_hidden_size, kernel_size=1, bias=True)

        self.residual_layer1 = DPTPreActResidualLayer(config)
        self.residual_layer2 = DPTPreActResidualLayer(config)

    def forward(self, hidden_state, residual=None):
        if residual is not None:
            if hidden_state.shape != residual.shape:
                residual = nn.functional.interpolate(
                    residual, size=(hidden_state.shape[2], hidden_state.shape[3]), mode="bilinear", align_corners=False
                )
            hidden_state = hidden_state + self.residual_layer1(residual)

        hidden_state = self.residual_layer2(hidden_state)
        hidden_state = nn.functional.interpolate(
            hidden_state, scale_factor=2, mode="bilinear", align_corners=self.align_corners
        )
        hidden_state = self.projection(hidden_state)

        return hidden_state



class DPTPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = DPTConfig
    base_model_prefix = "dpt"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Conv2d, nn.ConvTranspose2d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)



DPT_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`ViTConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

DPT_INPUTS_DOCSTRING = r"""



# 注释结束

- `DPTFeatureFusionLayer`是一个继承自`nn.Module`的类，用于特征融合，将不同阶段的特征映射进行融合。
- `DPTFeatureFusionLayer`类初始化函数`__init__`接受两个参数：`config`和`align_corners`，其中`config`为定义了模型架构的模型配置类，`align_corners`为双线性上采样的对齐角设置，默认为`True`。
- `DPTFeatureFusionLayer`类的`forward`函数接受`hidden_state`和`residual`两个参数，用于进行特征融合操作。如果`residual`不为`None`，则根据`hidden_state`和`residual`的形状进行插值操作，然后将其与`residual_layer1`的输出相加；接着将结果传递给`residual_layer2`，并进行双线性插值操作；最后使用`projection`进行卷积操作，得到融合后的特征。

- `DPTPreTrainedModel`是一个抽象类，用于处理权重初始化和预训练模型的下载和加载。
- `DPTPreTrainedModel`类的`_init_weights`函数用于初始化权重。对于`nn.Linear`、`nn.Conv2d`和`nn.ConvTranspose2d`三种类型的模块，使用正态分布进行权重初始化；对于`nn.LayerNorm`类型的模块，将偏置项置零，权重填充为1.0。
    # 定义函数参数并说明其数据类型和形状
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            像素值。像素值可以使用 [`AutoImageProcessor`] 获得。详细信息请参阅 [`DPTImageProcessor.__call__`]。
    
        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            用于使自注意力模块的部分头无效的掩码。掩码值在 `[0, 1]` 范围内：
    
            - 1 表示该头部**未被掩码**，
            - 0 表示该头部**被掩码**。
    
        output_attentions (`bool`, *optional*):
            是否返回所有注意力层的注意力张量。有关更多详细信息，请参阅返回的张量下的 `attentions`。
    
        output_hidden_states (`bool`, *optional*):
            是否返回所有层的隐藏状态。有关更多详细信息，请参阅返回的张量下的 `hidden_states`。
    
        return_dict (`bool`, *optional*):
            是否返回 [`~file_utils.ModelOutput`] 而不是普通元组。
"""
定义一个 DPT 模型，该模型输出原始隐藏状态，没有任何特定的头部。

:param config: DPT 模型的配置对象
:param add_pooling_layer: 是否在顶部添加池化层，默认为 True

# 初始化函数
def __init__(self, config, add_pooling_layer=True):
    调用父类的初始化函数
    super().__init__(config)
    设置模型的配置对象
    self.config = config

    # 如果配置为混合模式，则使用混合嵌入层，否则使用 ViT 嵌入层
    if config.is_hybrid:
        self.embeddings = DPTViTHybridEmbeddings(config)
    else:
        self.embeddings = DPTViTEmbeddings(config)
    
    # 初始化编码器
    self.encoder = DPTViTEncoder(config)

    # 初始化层归一化层
    self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    
    # 如果需要添加池化层，则初始化池化层
    self.pooler = DPTViTPooler(config) if add_pooling_layer else None

    # 初始化权重并进行最终处理
    self.post_init()

# 获取输入嵌入层
def get_input_embeddings(self):
    如果配置为混合模式，则返回嵌入层；否则返回补丁嵌入层
    if self.config.is_hybrid:
        return self.embeddings
    else:
        return self.embeddings.patch_embeddings

# 剪枝模型的注意力头
def _prune_heads(self, heads_to_prune):
    """
    对模型的注意力头进行剪枝。
    heads_to_prune: 字典，格式为 {layer_num: 在该层中要剪枝的头列表}
    参见基类 PreTrainedModel
    """
    遍历要剪枝的层和对应的头部列表
    for layer, heads in heads_to_prune.items():
        获取编码器中特定层的注意力模块，然后对指定的头部进行剪枝
        self.encoder.layer[layer].attention.prune_heads(heads)

# 模型的前向传播函数
def forward(
    self,
    pixel_values: torch.FloatTensor,
    head_mask: Optional[torch.FloatTensor] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
        ) -> Union[Tuple, BaseModelOutputWithPoolingAndIntermediateActivations]:
        # 如果未指定输出注意力矩阵，则使用配置文件中的参数
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果未指定输出隐藏状态，则使用配置文件中的参数
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果未指定返回字典，则使用配置文件中的参数
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果需要，准备头部遮罩
        # 1.0 表示保留该头部
        # attention_probs 的形状为 bsz x n_heads x N x N
        # 输入的 head_mask 的形状为 [num_heads] 或 [num_hidden_layers x num_heads]
        # 将 head_mask 转换为形状 [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # 将像素值嵌入到词嵌入空间
        embedding_output = self.embeddings(pixel_values, return_dict=return_dict)

        # 若未指定返回字典，则获取最后隐藏状态
        embedding_last_hidden_states = embedding_output[0] if not return_dict else embedding_output.last_hidden_states

        # 对嵌入表示进行编码
        encoder_outputs = self.encoder(
            embedding_last_hidden_states,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取序列输出
        sequence_output = encoder_outputs[0]
        # 对序列输出进行层归一化
        sequence_output = self.layernorm(sequence_output)
        # 如果存在池化层，则获取池化输出
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        # 如果未指定返回字典，则返回头部输出和编码器输出
        if not return_dict:
            head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
            return head_outputs + encoder_outputs[1:] + embedding_output[1:]

        # 返回带池化和中间激活的基础模型输出
        return BaseModelOutputWithPoolingAndIntermediateActivations(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            intermediate_activations=embedding_output.intermediate_activations,
        )
# 从 transformers.models.vit.modeling_vit.ViTPooler 复制代码并将 ViT->DPT
class DPTViTPooler(nn.Module):
    def __init__(self, config: DPTConfig):
        super().__init__()
        # 创建一个全连接层，将隐藏状态的维度映射到相同的维度
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # Tanh 激活函数
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # 我们通过简单地获取对应于第一个标记的隐藏状态来“池化”模型。
        first_token_tensor = hidden_states[:, 0]
        # 经过全连接层映射
        pooled_output = self.dense(first_token_tensor)
        # 应用激活函数
        pooled_output = self.activation(pooled_output)
        return pooled_output


class DPTNeck(nn.Module):
    """
    DPTNeck。一个 neck 是通常位于主干和头之间的模块。它将一个张量列表作为输入，并产生另一个张量列表作为输出。对于 DPT，它包括 2 个阶段：

    * DPTReassembleStage
    * DPTFeatureFusionStage。

    Args:
        config (dict): 配置字典。
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # 后处理：仅在非分层主干（例如 ViT、BEiT）的情况下需要
        if config.backbone_config is not None and config.backbone_config.model_type in ["swinv2"]:
            self.reassemble_stage = None
        else:
            self.reassemble_stage = DPTReassembleStage(config)

        # 使用配置中的 neck_hidden_sizes 创建卷积层
        self.convs = nn.ModuleList()
        for channel in config.neck_hidden_sizes:
            self.convs.append(nn.Conv2d(channel, config.fusion_hidden_size, kernel_size=3, padding=1, bias=False))

        # 融合阶段
        self.fusion_stage = DPTFeatureFusionStage(config)

    def forward(self, hidden_states: List[torch.Tensor], patch_height=None, patch_width=None) -> List[torch.Tensor]:
        """
        Args:
            hidden_states (`List[torch.FloatTensor]`, each of shape `(batch_size, sequence_length, hidden_size)` or `(batch_size, hidden_size, height, width)`):
                主干中的隐藏状态列表。
        """
        if not isinstance(hidden_states, (tuple, list)):
            raise ValueError("hidden_states 应该是张量的元组或列表")

        if len(hidden_states) != len(self.config.neck_hidden_sizes):
            raise ValueError("隐藏状态的数量应该等于 neck_hidden_sizes 的数量。")

        # 后处理隐藏状态
        if self.reassemble_stage is not None:
            hidden_states = self.reassemble_stage(hidden_states, patch_height, patch_width)

        # 使用卷积层对特征进行处理
        features = [self.convs[i](feature) for i, feature in enumerate(hidden_states)]

        # 融合模块
        output = self.fusion_stage(features)

        return output


class DPTDepthEstimationHead(nn.Module):
    """
    输出头部，由 3 个卷积层组成。它逐步减半特征维度，并在第一个卷积层之后将预测上采样到输入分辨率
    """

```  
    class DepthPredictionModule(nn.Module):
        """
        深度预测模块，用于生成深度图像（参考补充资料）。
        """
    
        def __init__(self, config):
            super().__init__()
    
            self.config = config  # 初始化模块配置
    
            self.projection = None  # 初始化投影层为 None
            if config.add_projection:  # 如果配置中指定添加投影层
                self.projection = nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))  # 创建投影层
    
            features = config.fusion_hidden_size  # 获取特征大小
            self.head = nn.Sequential(  # 创建网络头部序列模块
                nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),  # 第一个卷积层
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),  # 上采样
                nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),  # 第二个卷积层
                nn.ReLU(),  # 激活函数
                nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),  # 最后一个卷积层
                nn.ReLU(),  # 激活函数
            )
    
        def forward(self, hidden_states: List[torch.Tensor]) -> torch.Tensor:
            # 使用最后的特征
            hidden_states = hidden_states[self.config.head_in_index]
    
            if self.projection is not None:  # 如果存在投影层
                hidden_states = self.projection(hidden_states)  # 进行投影
                hidden_states = nn.ReLU()(hidden_states)  # 使用 ReLU 激活函数
    
            predicted_depth = self.head(hidden_states)  # 输入到头部网络
    
            predicted_depth = predicted_depth.squeeze(dim=1)  # 压缩深度维度
    
            return predicted_depth  # 返回预测的深度图像
# 使用深度估计头的 DPT 模型（包含3个卷积层），例如适用于KITTI、NYUv2数据集
@add_start_docstrings(
    """
    DPT Model with a depth estimation head on top (consisting of 3 convolutional layers) e.g. for KITTI, NYUv2.
    """,
    DPT_START_DOCSTRING,
)
class DPTForDepthEstimation(DPTPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.backbone = None
        if config.backbone_config is not None and config.is_hybrid is False:
            # 如果配置包含背景配置并且不是混合模式，则创建自动选择的背景模型
            self.backbone = AutoBackbone.from_config(config.backbone_config)
        else:
            # 否则创建 DPT 模型
            self.dpt = DPTModel(config, add_pooling_layer=False)

        # Neck（颈部网络）
        self.neck = DPTNeck(config)

        # Depth estimation head（深度估计头）
        self.head = DPTDepthEstimationHead(config)

        # Initialize weights and apply final processing（初始化权重并应用最终处理）
        self.post_init()

    @add_start_docstrings_to_model_forward(DPT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=DepthEstimatorOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: torch.FloatTensor,
        head_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        
# 语义分割头部网络
class DPTSemanticSegmentationHead(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config

        features = config.fusion_hidden_size
        # 使用一系列卷积、归一化、激活等操作构建语义分割头部网络
        self.head = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(),
            nn.Dropout(config.semantic_classifier_dropout),
            nn.Conv2d(features, config.num_labels, kernel_size=1),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
        )

    def forward(self, hidden_states: List[torch.Tensor]) -> torch.Tensor:
        # 使用最后的特征进行预测
        hidden_states = hidden_states[self.config.head_in_index]

        # 通过头部网络生成 logits
        logits = self.head(hidden_states)

        return logits


# 辅助头部网络
class DPTAuxiliaryHead(nn.Module):
    def __init__(self, config):
        super().__init__()

        features = config.fusion_hidden_size
        # 使用一系列卷积、归一化、激活等操作��建辅助头部网络
        self.head = nn.Sequential(
            nn.Conv2d(features, features, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(features),
            nn.ReLU(),
            nn.Dropout(0.1, False),
            nn.Conv2d(features, config.num_labels, kernel_size=1),
        )

    def forward(self, hidden_states):
        # 通过头部网络生成 logits
        logits = self.head(hidden_states)

        return logits


# 使用语义分割头部的 DPT 模型，例如适用于ADE20k、CityScapes数据集
@add_start_docstrings(
    """
    DPT Model with a semantic segmentation head on top e.g. for ADE20k, CityScapes.
    """,
    DPT_START_DOCSTRING,
)
class DPTForSemanticSegmentation(DPTPreTrainedModel):
    # 初始化方法，接受一个配置参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 创建 DPTModel 对象，不添加池化层
        self.dpt = DPTModel(config, add_pooling_layer=False)

        # 创建 DPTNeck 对象
        self.neck = DPTNeck(config)

        # 创建 DPTSemanticSegmentationHead 对象
        self.head = DPTSemanticSegmentationHead(config)
        # 如果配置中指定使用辅助头，则创建 DPTAuxiliaryHead 对象，否则为 None
        self.auxiliary_head = DPTAuxiliaryHead(config) if config.use_auxiliary_head else None

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播方法，接受多个参数，使用装饰器添加文档字符串
    @add_start_docstrings_to_model_forward(DPT_INPUTS_DOCSTRING)
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