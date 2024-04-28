# `.\transformers\models\vit_hybrid\modeling_vit_hybrid.py`

```
# 设置文件编码为 utf-8
# 版权声明
# 基于 Apache License, Version 2.0 开源许可协议
# 获取许可协议内容的链接
# 根据适用法律或书面协议，软件遵守 "AS IS" 基础分发，不提供任何形式的担保或条件，明示或暗示
# 有关特定语言对权限和限制的详细信息，请参阅许可协议
""" PyTorch ViT Hybrid model."""

# 导入模块
import collections.abc
import math
from typing import Dict, List, Optional, Set, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 导入自定义模块
from ...activations import ACT2FN
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ImageClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging

# 定义全局日志记录器
logger = logging.get_logger(__name__)

# 用于文档生成的配置
_CONFIG_FOR_DOC = "ViTHybridConfig"
_CHECKPOINT_FOR_DOC = "google/vit-hybrid-base-bit-384"
_EXPECTED_OUTPUT_SHAPE = [1, 197, 768]

# 图像分类相关文档
_IMAGE_CLASS_CHECKPOINT = "google/vit-hybrid-base-bit-384"
_IMAGE_CLASS_EXPECTED_OUTPUT = "tabby, tabby cat"

# 预训练模型列表
VIT_HYBRID_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "google/vit-hybrid-base-bit-384",
    # 更多 ViT Hybrid 模型参见 https://huggingface.co/models?filter=vit-hybrid
]

class ViTHybridEmbeddings(nn.Module):
    """
    构建 CLS 标记、位置和块嵌入。还可以选择是否包含掩码标记。
    """

    # 构造函数
    def __init__(self, config: ViTHybridConfig, use_mask_token: bool = False) -> None:
        super().__init__()

        # 定义学习参数 - CLS 标记
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        
        # 定义学习参数 - 掩码标记（可选）
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size)) if use_mask_token else None
        
        # 初始化块嵌入
        self.patch_embeddings = ViTHybridPatchEmbeddings(config)
        
        # 获取块数量
        num_patches = self.patch_embeddings.num_patches
        
        # 定义学习参数 - 位置嵌入
        self.position_embeddings = nn.Parameter(torch.randn(1, num_patches + 1, config.hidden_size))
        
        # 定义 dropout 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 保存配置
        self.config = config
```  
    # 该方法用于对预训练的位置编码进行插值,以便能够在更高分辨率的图像上使用该模型
    def interpolate_pos_encoding(self, embeddings: torch.Tensor, height: int, width: int) -> torch.Tensor:
        # 获取嵌入向量中的补丁数量和位置编码中的位置数量
        num_patches = embeddings.shape[1] - 1
        num_positions = self.position_embeddings.shape[1] - 1
        # 如果补丁数量和位置数量相同,并且高宽也相同,则直接返回位置编码
        if num_patches == num_positions and height == width:
            return self.position_embeddings
        # 将类别位置编码和补丁位置编码分别取出
        class_pos_embed = self.position_embeddings[:, 0]
        patch_pos_embed = self.position_embeddings[:, 1:]
        # 获取嵌入向量的特征维度
        dim = embeddings.shape[-1]
        # 根据配置的补丁大小计算出图像的高宽
        height = height // self.config.patch_size
        width = width // self.config.patch_size
        # 为了避免浮点数误差,在插值时添加一个小量
        height, width = height + 0.1, width + 0.1
        # 将补丁位置编码reshape并进行通道交换,为插值做准备
        patch_pos_embed = patch_pos_embed.reshape(1, int(math.sqrt(num_positions)), int(math.sqrt(num_positions)), dim)
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)
        # 使用双三次插值对补丁位置编码进行上采样
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            scale_factor=(height / math.sqrt(num_positions), width / math.sqrt(num_positions)),
            mode="bicubic",
            align_corners=False,
        )
        # 检查插值后的高宽是否与预期一致
        if int(height) != patch_pos_embed.shape[-2] or int(width) != patch_pos_embed.shape[-1]:
            raise ValueError(f"Invalid height or width: {height}, {width}")
        # 将补丁位置编码调整为与嵌入向量一致的形状,并与类别位置编码拼接
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
    
    # 该方法为模型的前向传播
    def forward(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        interpolate_pos_encoding: bool = False,
    ) -> torch.Tensor:
        # 获取输入张量的维度信息：批大小、通道数、高度、宽度
        batch_size, num_channels, height, width = pixel_values.shape
        # 将像素值转换为嵌入向量
        embeddings = self.patch_embeddings(pixel_values, interpolate_pos_encoding=interpolate_pos_encoding)

        # 如果存在布尔掩码，则替换对应位置的可见标记为掩码标记
        if bool_masked_pos is not None:
            # 获取嵌入向量序列的长度
            seq_length = embeddings.shape[1]
            # 生成与掩码对应的掩码标记序列
            mask_tokens = self.mask_token.expand(batch_size, seq_length, -1)
            # 创建掩码张量，并将其类型转换为掩码标记的类型
            mask = bool_masked_pos.unsqueeze(-1).type_as(mask_tokens)
            # 使用掩码替换可见标记
            embeddings = embeddings * (1.0 - mask) + mask_tokens * mask

        # 将[CLS]标记添加到嵌入的补丁标记中
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        # 拼接[CLS]标记和嵌入的补丁标记
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)

        # 为每个标记添加位置编码
        if interpolate_pos_encoding:
            # 使用插值方法为每个标记添加位置编码
            embeddings = embeddings + self.interpolate_pos_encoding(embeddings, height, width)
        else:
            # 直接将位置编码加到嵌入向量中
            embeddings = embeddings + self.position_embeddings

        # 对嵌入向量进行 dropout 操作
        embeddings = self.dropout(embeddings)

        # 返回嵌入向量
        return embeddings
# 定义一个新的类 ViTHybridPatchEmbeddings，继承自 nn.Module
class ViTHybridPatchEmbeddings(nn.Module):
    """
    This class turns `pixel_values` of shape `(batch_size, num_channels, height, width)` into the initial
    `hidden_states` (patch embeddings) of shape `(batch_size, seq_length, hidden_size)` to be consumed by a
    Transformer.
    """

    # 初始化函数，接受配置和特征大小作为参数
    def __init__(self, config, feature_size=None):
        super().__init__()
        # 从配置中获取图像大小和分块大小
        image_size, patch_size = config.image_size, config.patch_size
        num_channels, hidden_size = config.num_channels, config.hidden_size

        # 如果图像大小和分块大小不是可迭代对象，则转换为可迭代对象
        image_size = image_size if isinstance(image_size, collections.abc.Iterable) else (image_size, image_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)

        # 根据配置创建自动编码器
        self.backbone = AutoBackbone.from_config(config.backbone_config)
        # 如果不是 BIT 模型，则抛出异常
        if self.backbone.config.model_type != "bit":
            raise ValueError(f"Backbone model type {self.backbone.model_type} is not supported.")
        feature_dim = self.backbone.channels[-1]

        # 如果未提供特征大小，则根据配置获取特征映射形状和特征维度
        if feature_size is None:
            feature_map = config.backbone_featmap_shape
            feature_size = feature_map[-2:]
            feature_dim = feature_map[1]
        else:
            # 如果特征大小不是可迭代对象，则转换为可迭代对象
            feature_size = (
                feature_size if isinstance(feature_size, collections.abc.Iterable) else (feature_size, feature_size)
            )
            feature_dim = self.backbone.channels[-1]

        # 计算网格大小和块数
        self.grid_size = (feature_size[0] // patch_size[0], feature_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels

        # 定义一个卷积层，将特征映射转换为隐藏状态
        self.projection = nn.Conv2d(feature_dim, hidden_size, kernel_size=patch_size, stride=patch_size)

    # 前向传播函数，接受像素值和是否插值位置编码标志作为参数
    def forward(self, pixel_values: torch.Tensor, interpolate_pos_encoding: bool = False) -> torch.Tensor:
        _, num_channels, height, width = pixel_values.shape
        # 如果通道数不匹配，则抛出异常
        if num_channels != self.num_channels:
            raise ValueError(
                "Make sure that the channel dimension of the pixel values match with the one set in the configuration."
            )
        # 如果不插值位置编码，则检查输入图像大小是否匹配模型要求
        if not interpolate_pos_encoding:
            if height != self.image_size[0] or width != self.image_size[1]:
                raise ValueError(
                    f"Input image size {height}*{width} doesn't match model"
                    f" {self.image_size[0]}*{self.image_size[1]}."
                )

        # 使用自动编码器获取特征映射，并获取最后一个特征映射
        features = self.backbone(pixel_values).feature_maps[-1]
        # 使用投影层将特征映射转换为嵌入表示
        embeddings = self.projection(features).flatten(2).transpose(1, 2)

        # 返回嵌入表示
        return embeddings


# 从 transformers.models.vit.modeling_vit.ViTSelfAttention 复制到 ViTHybridSelfAttention
class ViTHybridSelfAttention(nn.Module):
    # 初始化函数，接受一个 ViTHybridConfig 类型的参数 config
    def __init__(self, config: ViTHybridConfig) -> None:
        # 调用父类的初始化函数
        super().__init__()
        # 如果隐藏层大小不能被注意力头数整除，且没有嵌入大小的属性
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            # 抛出值错误异常
            raise ValueError(
                # 输出错误信息，指出隐藏大小不是注意力头数的倍数
                f"The hidden size {config.hidden_size,} is not a multiple of the number of attention "
                # 输出错误信息，指出隐藏大小和注意力头数
                f"heads {config.num_attention_heads}."
            )

        # 设置注意力头数
        self.num_attention_heads = config.num_attention_heads
        # 计算每个注意力头的大小
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        # 计算所有头的总大小
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 创建查询、键、值的线性层
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=config.qkv_bias)

        # 创建丢弃层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    # 将输入张量转置以便计算注意力得分
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        # 获取新的张量形状，增加注意力头数和头大小的维度
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # 调整张量形状
        x = x.view(new_x_shape)
        # 对张量维度进行置换，以符合注意力计算的要求
        return x.permute(0, 2, 1, 3)

    # 前向传播函数，接受隐藏状态、头蒙版和是否输出注意力得分作为参数
    def forward(
        self, hidden_states, head_mask: Optional[torch.Tensor] = None, output_attentions: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        # 使用查询线性层处理隐藏状态
        mixed_query_layer = self.query(hidden_states)

        # 使用 transpose_for_scores 函数对键和值进行形状转换，并处理隐藏状态
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # 计算原始注意力得分，使用矩阵乘法实现
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # 对得分进行缩放，防止梯度爆炸
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # 对注意力得分进行归一化，得到注意力概率
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # 对注意力概率进行丢弃操作
        attention_probs = self.dropout(attention_probs)

        # 如果存在头蒙版，则应用蒙版
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # 计算上下文层，使用注意力概率和值进行加权求和
        context_layer = torch.matmul(attention_probs, value_layer)

        # 对上下文层进行形状转换，以符合输出要求
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)

        # 如果需要输出注意力得分，则返回上下文层和注意力概率；否则只返回上下文层
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs
# 从transformers.models.vit.modeling_vit.ViTSelfOutput复制并修改为ViT->ViTHybrid
class ViTHybridSelfOutput(nn.Module):
    """
    The residual connection is defined in ViTHybridLayer instead of here (as is the case with other models), due to the
    layernorm applied before each block.
    """

    def __init__(self, config: ViTHybridConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)  # 应用全连接层
        hidden_states = self.dropout(hidden_states)  # 应用dropout

        return hidden_states


# 从transformers.models.vit.modeling_vit.ViTAttention复制并修改为ViT->ViTHybrid
class ViTHybridAttention(nn.Module):
    def __init__(self, config: ViTHybridConfig) -> None:
        super().__init__()
        self.attention = ViTHybridSelfAttention(config)  # 初始化自注意力层
        self.output = ViTHybridSelfOutput(config)  # 初始化输出层
        self.pruned_heads = set()

    def prune_heads(self, heads: Set[int]) -> None:
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.attention.num_attention_heads, self.attention.attention_head_size, self.pruned_heads
        )

        # 精简线性层
        self.attention.query = prune_linear_layer(self.attention.query, index)
        self.attention.key = prune_linear_layer(self.attention.key, index)
        self.attention.value = prune_linear_layer(self.attention.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储被精简的头信息
        self.attention.num_attention_heads = self.attention.num_attention_heads - len(heads)
        self.attention.all_head_size = self.attention.attention_head_size * self.attention.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_outputs = self.attention(hidden_states, head_mask, output_attentions)  # 调用自注意力层

        attention_output = self.output(self_outputs[0], hidden_states)  # 调用输出层

        outputs = (attention_output,) + self_outputs[1:]  # 如果有需要，添加注意力值
        return outputs


# 从transformers.models.vit.modeling_vit.ViTIntermediate复制并修改为ViT->ViTHybrid
class ViTHybridIntermediate(nn.Module):
    # 初始化函数，接受一个 ViTHybridConfig 类型的参数
    def __init__(self, config: ViTHybridConfig) -> None:
        # 调用父类的初始化函数
        super().__init__()
        # 创建一个全连接层，输入尺寸为 config.hidden_size，输出尺寸为 config.intermediate_size
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 判断 config.hidden_act 是否为字符串类型，如果是则根据字符串找到对应的激活函数，保存在 self.intermediate_act_fn 中
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        # 如果 config.hidden_act 不是字符串类型，则直接使用 config.hidden_act 作为激活函数
        else:
            self.intermediate_act_fn = config.hidden_act

    # forward 函数，接受输入参数 hidden_states，并返回处理后的结果
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将输入数据通过全连接层 self.dense 处理
        hidden_states = self.dense(hidden_states)
        # 将处理后的数据通过激活函数 self.intermediate_act_fn 处理
        hidden_states = self.intermediate_act_fn(hidden_states)

        # 返回处理后的结果
        return hidden_states
# 从transformers.models.vit.modeling_vit.ViTOutput中复制代码，并将ViT改为ViTHybrid
class ViTHybridOutput(nn.Module):
    def __init__(self, config: ViTHybridConfig) -> None:
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)  # 创建一个线性层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # 创建一个dropout层

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)  # 使用线性层处理输入hidden_states
        hidden_states = self.dropout(hidden_states)  # 使用dropout层处理hidden_states

        hidden_states = hidden_states + input_tensor  # 将hidden_states和input_tensor相加

        return hidden_states  # 返回处理后的hidden_states


class ViTHybridLayer(nn.Module):
    # 这对应于timm实现中的Block类
    def __init__(self, config: ViTHybridConfig) -> None:
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward  # 设置chunk_size_feed_forward值
        self.seq_len_dim = 1  # 设置seq_len_dim值
        self.attention = ViTHybridAttention(config)  # 创建ViTHybridAttention实例
        self.intermediate = ViTHybridIntermediate(config)  # 创建ViTHybridIntermediate实例
        self.output = ViTHybridOutput(config)  # 创建ViTHybridOutput实例
        self.layernorm_before = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)  # 创建LayerNorm实例，应用在模型输出之前
        self.layernorm_after = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)  # 创建LayerNorm实例，应用在模型输出之后

    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor]]:
        self_attention_outputs = self.attention(
            self.layernorm_before(hidden_states),  # 在ViTHybrid中，在self-attention之前应用layernorm
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # 如果输出attention权重，则添加self attentions

        # 第一个残差连接
        # 为了使用`accelerate`，将hidden_states分配到正确的设备上，参考：https://github.com/huggingface/transformers/pull/20705/
        hidden_states = attention_output + hidden_states.to(attention_output.device)

        # 在ViTHybrid中，self-attention之后还应用layernorm
        layer_output = self.layernorm_after(hidden_states)
        layer_output = self.intermediate(layer_output)  # 使用ViTHybridIntermediate处理layer_output

        # 第二个残差连接
        layer_output = self.output(layer_output, hidden_states)  # 使用ViTHybridOutput处理layer_output和hidden_states

        outputs = (layer_output,) + outputs  # 将layer_output添加到outputs中

        return outputs  # 返回处理后的outputs


# 从transformers.models.vit.modeling_vit.ViTEncoder中复制代码，并将ViT改为ViTHybrid
class ViTHybridEncoder(nn.Module):
    def __init__(self, config: ViTHybridConfig) -> None:
        super().__init__()
        self.config = config  # 设置config属性
        self.layer = nn.ModuleList([ViTHybridLayer(config) for _ in range(config.num_hidden_layers)])  # 生成长度为num_hidden_layers的ViTHybridLayer列表
        self.gradient_checkpointing = False  # 设置gradient_checkpointing属性为False
    # 前向传播函数，接收隐藏状态和头部掩码等参数，输出模型的中间状态或最终结果
    def forward(
        self,
        hidden_states: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[tuple, BaseModelOutput]:
        # 如果需要输出隐藏状态，则初始化空元组，用于存储每层的隐藏状态
        all_hidden_states = () if output_hidden_states else None
        # 如果需要输出注意力矩阵，则初始化空元组，用于存储每层的注意力矩阵
        all_self_attentions = () if output_attentions else None

        # 遍历每个层的模块
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，则将当前层的隐藏状态存储到 all_hidden_states
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前层的头部掩码
            layer_head_mask = head_mask[i] if head_mask is not None else None

            # 如果启用渐变检查点且处于训练模式，则通过调用渐变检查点函数获取当前层的输出
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    layer_head_mask,
                    output_attentions,
                )
            else:
                # 否则直接调用当前层的模块得到输出
                layer_outputs = layer_module(hidden_states, layer_head_mask, output_attentions)

            # 更新隐藏状态为当前层的输出
            hidden_states = layer_outputs[0]

            # 如果需要输出注意力矩阵，则将当前层的注意力矩阵存储到 all_self_attentions
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 如果需要输出隐藏状态，则将最后一层的隐藏状态添加到 all_hidden_states
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不需要以字典形式输出结果，则返回包含隐藏状态、隐藏状态列表和注意力矩阵列表的元组
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        # 否则返回一个包含最后隐藏状态、所有隐藏状态和所有注意力矩阵的 BaseModelOutput 对象
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
# 从 transformers.models.vit.modeling_vit.ViTPreTrainedModel 复制代码并将 ViT->ViTHybrid
class ViTHybridPreTrainedModel(PreTrainedModel):
    """
    一个抽象类，处理权重初始化和一个简单的接口用于下载和加载预训练模型。
    """

    config_class = ViTHybridConfig # 使用的配置类是 ViTHybridConfig
    base_model_prefix = "vit" # 基础模型前缀是 "vit"
    main_input_name = "pixel_values" # 主输入名称是 "pixel_values"
    supports_gradient_checkpointing = True # 支持梯度检查点
    _no_split_modules = ["ViTHybridEmbeddings", "ViTHybridLayer"] # 不拆分的模块是 ["ViTHybridEmbeddings", "ViTHybridLayer"]

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """初始化权重"""
        if isinstance(module, (nn.Linear, nn.Conv2d)): # 如果模块是线性或者卷积层
            # 将输入转换为 `fp32`，并将其转换回期望的 `dtype`，以避免 `half` 模式下的 `trunc_normal_cpu` 未实现问题
            module.weight.data = nn.init.trunc_normal_(
                module.weight.data.to(torch.float32), mean=0.0, std=self.config.initializer_range
            ).to(module.weight.dtype)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm): # 如果模块是 LayerNorm
            module.bias.data.zero_() # 偏置初始化为 0
            module.weight.data.fill_(1.0) # 权重初始化为 1.0
        elif isinstance(module, ViTHybridEmbeddings): # 如果模块是 ViTHybridEmbeddings
            module.position_embeddings.data = nn.init.trunc_normal_(
                module.position_embeddings.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.position_embeddings.dtype) # 初始化位置嵌入
            module.cls_token.data = nn.init.trunc_normal_(
                module.cls_token.data.to(torch.float32),
                mean=0.0,
                std=self.config.initializer_range,
            ).to(module.cls_token.dtype) # 初始化类别标记


VIT_START_DOCSTRING = r"""
    这个模型是 PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) 的子类。将其
    作为常规的 PyTorch 模块使用，并参考 PyTorch 文档了解一切与一般使用和行为相关的事项。

    Parameters:
        config ([`ViTHybridConfig`]): 带有模型所有参数的模型配置类。
            使用配置文件初始化不会加载模型的权重，只加载配置。查看 [`~PreTrainedModel.from_pretrained`] 方法加载模型权重。
"""

VIT_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Pixel values can be obtained using [`AutoImageProcessor`]. See
            [`ViTHybridImageProcessor.__call__`] for details.

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
ViTHybridModel 类定义，继承自 ViTHybridPreTrainedModel
"""
# 从 transformers.models.vit.modeling_vit.ViTModel 复制代码，并将 ViT->ViTHybrid
class ViTHybridModel(ViTHybridPreTrainedModel):
    """
    ViTHybridModel 类的初始化方法
    """
    def __init__(self, config: ViTHybridConfig, add_pooling_layer: bool = True, use_mask_token: bool = False):
        super().__init__(config)
        self.config = config

        # 初始化 ViTHybridEmbeddings 和 ViTHybridEncoder
        self.embeddings = ViTHybridEmbeddings(config, use_mask_token=use_mask_token)
        self.encoder = ViTHybridEncoder(config)

        # 初始化 LayerNorm 和 ViTHybridPooler（如果添加了池化层）
        self.layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pooler = ViTHybridPooler(config) if add_pooling_layer else None

        # 初始化权重并应用最终处理
        self.post_init()

    """
    获取输入嵌入的方法
    """
    def get_input_embeddings(self) -> ViTHybridPatchEmbeddings:
        return self.embeddings.patch_embeddings

    """
    剪枝模型中的注意力头
    """
    def _prune_heads(self, heads_to_prune: Dict[int, List[int]]) -> None:
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    """
    ViTHybridModel 类的前向传播方法
    """
    @add_start_docstrings_to_model_forward(VIT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutputWithPooling,
        config_class=_CONFIG_FOR_DOC,
        modality="vision",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        bool_masked_pos: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ) -> Union[Tuple, BaseModelOutputWithPooling]:
        r"""
        bool_masked_pos (`torch.BoolTensor` of shape `(batch_size, num_patches)`, *optional*):
            Boolean masked positions. Indicates which patches are masked (1) and which aren't (0).
        """
        # 检查是否需要输出注意力权重
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 检查是否需要输出隐藏状态
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 检查是否需要返回字典格式的输出
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 检查是否提供了像素值
        if pixel_values is None:
            raise ValueError("You have to specify pixel_values")

        # 准备头部遮罩（head mask）
        # head_mask表示保留（1.0）和忽略（0）的头（head）
        # attention_probs的形状为batch_size x n_heads x N x N
        # 输入的head_mask形状为[num_heads]或[num_hidden_layers x num_heads]
        # head_mask被转换为[num_hidden_layers x batch x num_heads x seq_length x seq_length]的形状
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        # 如果像素值数据类型与预期不符，则调整数据类型
        expected_dtype = self.embeddings.patch_embeddings.projection.weight.dtype
        if pixel_values.dtype != expected_dtype:
            pixel_values = pixel_values.to(expected_dtype)

        # 将像素值嵌入到模型中
        embedding_output = self.embeddings(
            pixel_values, bool_masked_pos=bool_masked_pos, interpolate_pos_encoding=interpolate_pos_encoding
        )

        # 将嵌入输出传入编码器
        encoder_outputs = self.encoder(
            embedding_output,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        sequence_output = self.layernorm(sequence_output)
        # 若存在池化器，则对序列输出进行池化操作
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        # 如果不返回字典格式输出
        if not return_dict:
            head_outputs = (sequence_output, pooled_output) if pooled_output is not None else (sequence_output,)
            # 返回编码器输出
            return head_outputs + encoder_outputs[1:]

        # 返回包含池化输出和其他信息的字典格式输出
        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
# 从transformers.models.vit.modeling_vit.ViTPooler复制并将ViT->ViTHybrid
# 定义了一个继承自nn.Module的类ViTHybridPooler
class ViTHybridPooler(nn.Module):
    def __init__(self, config: ViTHybridConfig):
        super().__init__()
        # 初始化一个全连接层dense，输入输出维度都为config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 初始化激活函数Tanh
        self.activation = nn.Tanh()

    # 定义前向传播函数
    def forward(self, hidden_states):
        # 通过仅取第一个token对应的隐藏状态来"池化"模型
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

# 从transformers.models.vit.modeling_vit.ViTForImageClassification中复制并将ViT->ViTHybrid
# 定义了一个继承自ViTHybridPreTrainedModel的类ViTHybridForImageClassification
class ViTHybridForImageClassification(ViTHybridPreTrainedModel):
    def __init__(self, config: ViTHybridConfig) -> None:
        super().__init__(config)

        # 将num_labels设置为config中的num_labels值
        self.num_labels = config.num_labels
        # 初始化一个ViTHybridModel对象vit，不添加池化层
        self.vit = ViTHybridModel(config, add_pooling_layer=False)

        # 分类器头部
        # 如果配置中的num_labels大于0，则初始化一个全连接层classifier，输入维度为config.hidden_size，输出维度为config.num_labels
        # 否则初始化一个恒等映射nn.Identity()
        self.classifier = nn.Linear(config.hidden_size, config.num_labels) if config.num_labels > 0 else nn.Identity()

        # 初始化权重并应用最终处理
        self.post_init()

    # 定义前向传播函数
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        interpolate_pos_encoding: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[tuple, ImageClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional`):
            Labels for computing the image classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 设置返回字典，如果没有指定则使用配置中的设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # 使用ViT模型进行前向传播
        outputs = self.vit(
            pixel_values,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
            return_dict=return_dict,
        )

        # 获取模型输出的序列特征
        sequence_output = outputs[0]

        # 通过分类器获得分类的logits
        logits = self.classifier(sequence_output[:, 0, :])

        loss = None
        if labels is not None:
            # 将标签移动到正确的设备以启用模型并行计算
            labels = labels.to(logits.device)
            # 根据标签数据类型和类别数量确定问题类型
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据问题类型使用不同的损失函数计算损失值
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

        if not return_dict:
            output = (logits,) + outputs[1:]
            # 如果损失值不为空，则返回损失值和输出
            return ((loss,) + output) if loss is not None else output

        # 返回图像分类器的输出对象
        return ImageClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```