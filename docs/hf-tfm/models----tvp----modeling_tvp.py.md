# `.\models\tvp\modeling_tvp.py`

```py
# 设置文件编码为 UTF-8

# 版权声明，指出版权归属及许可协议
# Copyright 2023 The Intel AIA Team Authors, and HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch TVP Model"""

# 导入必要的库
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.utils.checkpoint
from torch import nn

# 导入自定义模块
from ...activations import ACT2FN
from ...file_utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ModelOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import prune_linear_layer
from ...utils import logging
from ...utils.backbone_utils import load_backbone
from .configuration_tvp import TvpConfig

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# 预训练模型的存档列表
TVP_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "Intel/tvp-base",
    "Intel/tvp-base-ANet",
    # See all Tvp models at https://huggingface.co/models?filter=tvp
]

@dataclass
# 定义 TvpVideoGroundingOutput 类，继承自 ModelOutput
class TvpVideoGroundingOutput(ModelOutput):
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`):
            Temporal-Distance IoU loss for video grounding.
        logits (`torch.FloatTensor` of shape `(batch_size, 2)`):
            Contains start_time/duration and end_time/duration. It is the time slot of the videos corresponding to the
            input texts.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of
            the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

# 定义 TvpLoss 类，继承自 nn.Module
class TvpLoss(nn.Module):
    """
    Placeholder for TvpLoss class definition.
    """
    This class computes the losses for `TvpForVideoGrounding`. The process happens in two steps: 1) we compute
    hungarian assignment between ground truth boxes and the outputs of the model 2) we supervise each pair of matched
    ground-truth / prediction (supervise class and box).

    Args:
        losses (`List[str]`):
            List of all the losses to be applied.
    """
    # 定义一个用于视频定位损失计算的类
    class TvpLossCalculator:
        
        # 初始化方法，接收损失列表并进行初始化
        def __init__(self, losses):
            super().__init__()
            # 定义损失函数映射字典
            self.loss_map = {
                "iou": self.loss_iou,
                "distance": self.loss_distance,
                "duration": self.loss_duration,
            }
            # 检查每个损失函数是否支持，若不支持则引发 ValueError 异常
            for loss in losses:
                if loss not in self.loss_map:
                    raise ValueError(f"Loss {loss} not supported")

            self.losses = losses

        # 计算 IoU 损失函数
        def loss_iou(self, start_time, end_time, candidates_start_time, candidates_end_time, duration):
            """
            Measure the intersection over union.
            """
            # 计算交集部分
            inter = torch.min(candidates_end_time, end_time) - torch.max(candidates_start_time, start_time)
            # 计算并集部分
            union = torch.max(candidates_end_time, end_time) - torch.min(candidates_start_time, start_time)
            # 计算 IoU
            iou = 1 - inter.clamp(min=0) / union

            return iou

        # 计算距离损失函数
        def loss_distance(self, start_time, end_time, candidates_start_time, candidates_end_time, duration):
            """
            Measure the distance of mid points.
            """
            # 计算候选框中点
            mid_candidates = torch.div(torch.add(candidates_start_time, candidates_end_time), 2.0)
            # 计算真实框中点
            mid_groundtruth = torch.div(torch.add(start_time, end_time), 2.0)
            # 计算中点距离差异
            distance_diff = torch.div(
                torch.max(mid_candidates, mid_groundtruth) - torch.min(mid_candidates, mid_groundtruth), duration
            ).clamp(min=0.2)

            return distance_diff

        # 计算时长损失函数
        def loss_duration(self, start_time, end_time, candidates_start_time, candidates_end_time, duration):
            """
            Measure the difference of duration.
            """
            # 计算候选框时长
            duration_candidates = torch.sub(candidates_end_time, candidates_start_time)
            # 计算真实框时长
            duration_groundtruth = torch.sub(end_time, start_time)
            # 计算时长差异
            duration_diff = torch.square(torch.div(torch.sub(duration_candidates, duration_groundtruth), duration))
            duration_diff = duration_diff.clamp(min=0.4)

            return duration_diff
    def forward(self, logits, labels):
        """
        This performs the loss computation.

        Args:
            logits (`torch.FloatTensor`):
                The output logits of head module.
            labels (`List[torch.FloatTensor]`):
                List of tensors ([start, end, duration]), which contains start time, end time of the video corresponding to the text, and also the duration.
        """
        # 从标签中解包出视频的时长、开始时间和结束时间
        duration, start_time, end_time = labels
        # 将logits乘以视频持续时间，得到候选的开始时间和结束时间
        candidates = torch.mul(logits, duration)
        # 将候选的开始时间和结束时间转换为浮点数张量
        candidates_start_time, candidates_end_time = candidates[:, 0].float(), candidates[:, 1].float()

        # 初始化损失字典
        losses_dict = {}
        # 遍历每种损失函数并计算损失值，将结果更新到损失字典中
        for loss in self.losses:
            losses_dict.update(
                {loss: self.loss_map[loss](start_time, end_time, candidates_start_time, candidates_end_time, duration)}
            )

        # 返回损失字典作为结果
        return losses_dict
class TvpVisionModel(nn.Module):
    # 定义一个视觉模型类，继承自nn.Module
    def __init__(self, config):
        super().__init__()
        # 加载指定配置的后端模型作为主干网络
        self.backbone = load_backbone(config)
        # 定义网格编码器的卷积层，用于处理特征图
        self.grid_encoder_conv = nn.Conv2d(
            config.backbone_config.hidden_sizes[-1],  # 输入通道数为主干网络的最后一个隐藏层大小
            config.hidden_size,  # 输出通道数为配置中指定的隐藏层大小
            kernel_size=3,  # 卷积核大小为3x3
            stride=1,  # 步长为1
            padding=1,  # 填充大小为1
            groups=1,  # 不使用分组卷积
            bias=False,  # 不使用偏置项
        )

    def forward(self, pixel_values):
        # 获取输入张量的形状信息
        batch_size, num_frames, num_channels, height, width = pixel_values.shape
        # 将输入张量重新排列为(batch_size * num_frames, num_channels, height, width)
        pixel_values = pixel_values.view(batch_size * num_frames, num_channels, height, width)
        # 将重新排列后的输入通过主干网络获取特征图输出，并选择第一个输出元素
        grid_feat_outputs = self.backbone(pixel_values)["feature_maps"][0]
        # 将特征图通过网格编码器的卷积层进行处理
        grid = self.grid_encoder_conv(grid_feat_outputs)
        # 对处理后的网格进行最大池化操作，核大小为2x2，步长为2
        grid = nn.functional.max_pool2d(grid, kernel_size=2, stride=2)
        # 对池化后的网格应用ReLU激活函数
        grid = nn.functional.relu(grid, inplace=True)
        # 获取处理后网格的通道数、高度和宽度信息
        new_channel, new_height, new_width = grid.shape[-3:]
        # 将网格重新排列为(batch_size, num_frames, new_channel, new_height, new_width)
        grid = grid.view(batch_size, num_frames, new_channel, new_height, new_width)
        # 将最后两个维度的顺序调整为(batch_size, num_frames, height, width, num_channels)
        grid = grid.permute(0, 1, 3, 4, 2)
        # 返回处理后的网格张量作为输出
        return grid


class TvpVisualInputEmbedding(nn.Module):
    """
    Takes input of both image and video (multi-frame)
    """

    def __init__(self, config):
        super().__init__()
        # 定义位置编码的Embedding层，用于序列的位置信息编码
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # 定义行位置编码的Embedding层，用于网格的行位置信息编码
        self.row_position_embeddings = nn.Embedding(config.max_grid_row_position_embeddings, config.hidden_size)
        # 定义列位置编码的Embedding层，用于网格的列位置信息编码
        self.col_position_embeddings = nn.Embedding(config.max_grid_col_position_embeddings, config.hidden_size)
        # 定义令牌类型编码的Embedding层，用于区分不同类型的令牌
        self.token_type_embeddings = nn.Embedding(1, config.hidden_size)
        # 定义Layer Norm层，用于归一化输入特征
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 定义Dropout层，用于在训练过程中随机丢弃部分输入特征，防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    def add_2d_positional_embeddings(self, grid):
        """
        Args:
            grid: (batch_size, height, width, hidden_dim)
        Returns:
            grid + col_position_embeddings.view(*col_shape): (batch_size, *, height, width, hidden_dim)
        """
        batch_size, height, width, hidden_dim = grid.shape

        # 添加行位置嵌入
        row_position_ids = torch.arange(height, dtype=torch.long, device=grid.device)  # (height, )
        row_position_embeddings = self.row_position_embeddings(row_position_ids)  # (height, hidden_dim)
        row_shape = (1,) * (len(grid.shape) - 3) + (height, 1, hidden_dim)  # (1, height, 1, hidden_dim)
        grid = grid + row_position_embeddings.view(*row_shape)  # 自动广播操作

        # 添加列位置嵌入
        col_position_ids = torch.arange(width, dtype=torch.long, device=grid.device)  # (width, )
        col_position_embeddings = self.col_position_embeddings(col_position_ids)  # (width, hidden_dim)
        col_shape = (batch_size, 1, width, hidden_dim)  # (1, 1, width, hidden_dim)
        return grid + col_position_embeddings.view(*col_shape)  # 自动广播操作

    def forward(self, grid):
        """
        Args:
            grid: Array of shape (batch_size, num_frames, height, width, num_channels).
                It contains processed frames extracted from videos, and is generated by Tvp image preprocessor. Note,
                num_frames can be 1

        Returns:
            embeddings: The embedding of grid with size (batch_size, height*width, num_channels)

        """
        batch_size, num_frames, height, width, num_channels = grid.shape
        # 时间平均池化，得到 (batch_size, height, width, hidden_size)
        grid = grid.mean(1)
        grid = self.add_2d_positional_embeddings(grid)
        # 图像令牌序列，得到 (batch_size, height*width, num_channels)
        visual_tokens = grid.view(batch_size, -1, num_channels)
        visual_tokens_shape = visual_tokens.shape[:-1]
        device = visual_tokens.device

        # 图像令牌类型嵌入
        token_type_ids = torch.zeros(visual_tokens_shape, dtype=torch.long, device=device)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = visual_tokens + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        # 初始化词嵌入层，将词汇表大小映射到隐藏大小，支持填充索引
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 初始化位置嵌入层，将最大位置嵌入数映射到隐藏大小
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # 初始化标记类型嵌入层，将类型词汇表大小映射到隐藏大小
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        # 初始化层归一化，对隐藏大小的张量进行归一化处理
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化丢弃层，根据隐藏丢弃概率进行随机丢弃
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        # 如果位置ID为None，则创建一个序列长度的张量作为位置ID
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        # 如果标记类型ID为None，则创建一个与输入形状相同的零张量作为标记类型ID
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # 如果输入嵌入为空，则使用输入ID获取词嵌入
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        # 获取位置嵌入和标记类型嵌入
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 计算总嵌入，包括词嵌入、位置嵌入和标记类型嵌入
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class TvpAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 如果隐藏大小不能被注意力头数整除且配置中没有嵌入大小属性，则抛出错误
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size} is not a multiple of the number of attention heads {config.num_attention_heads}"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 初始化查询、键、值线性变换层，将隐藏大小映射到注意力头大小
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        # 注意力丢弃层，根据注意力概率丢弃
        self.attn_dropout = nn.Dropout(config.attention_probs_dropout_prob)

        # 初始化全连接层和层归一化层，用于输出注意力后的隐藏大小
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化丢弃层，根据隐藏丢弃概率进行随机丢弃
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 初始化剪枝头集合，用于标识应该被剪枝的注意力头
        self.pruned_heads = set()
    # 对 self.pruned_heads 进行修剪操作，移除已经修剪过的头部
    def prune_heads(self, heads):
        # 如果 heads 长度为 0，则直接返回，不进行修剪操作
        if len(heads) == 0:
            return
        # 创建一个全为 1 的掩码，形状为 (self.num_attention_heads, self.attention_head_size)
        mask = torch.ones(self.num_attention_heads, self.attention_head_size)
        # 将 heads 转换为集合，并从中移除已经修剪过的头部
        heads = set(heads) - self.pruned_heads  
        # 遍历剩余的 heads
        for head in heads:
            # 计算比当前 head 小的已修剪头部数量，调整索引
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            # 将对应位置的掩码设为 0
            mask[head] = 0
        # 将掩码展平并获取非零元素的索引
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()

        # 对线性层进行修剪操作
        self.query = prune_linear_layer(self.query, index)
        self.key = prune_linear_layer(self.key, index)
        self.value = prune_linear_layer(self.value, index)
        self.dense = prune_linear_layer(self.dense, index, dim=1)

        # 更新超参数并存储修剪过的头部
        self.num_attention_heads = self.num_attention_heads - len(heads)
        self.all_head_size = self.attention_head_size * self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    # 重新整形操作，用于将 tensor 从形状 (batch_size * sequence_length * ...) 转换为 (batch_size * ... * num_attention_heads * attention_head_size)
    def _reshape(self, tensor: torch.Tensor, sequence_length: int, batch_size: int):
        return (
            tensor.view(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size)
            .transpose(1, 2)  # 将 sequence_length 和 num_attention_heads 这两个维度交换位置
            .contiguous()  # 确保张量的内存是连续的
        )

    # 前向传播函数
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions: Optional[bool] = None,
        ):
            # 获取隐藏状态的批量大小和序列长度
            batch_size, sequence_length = hidden_states.shape[:2]
            
            # 通过self.query对隐藏状态进行查询操作，生成混合查询层
            mixed_query_layer = self.query(hidden_states)

            # 通过self.key对隐藏状态进行键操作，生成混合键层
            mixed_key_layer = self.key(hidden_states)
            
            # 通过self.value对隐藏状态进行值操作，生成混合值层
            mixed_value_layer = self.value(hidden_states)

            # 使用私有方法self._reshape重新塑形混合查询、键、值层
            query_layer = self._reshape(mixed_query_layer, sequence_length, batch_size)
            key_layer = self._reshape(mixed_key_layer, sequence_length, batch_size)
            value_layer = self._reshape(mixed_value_layer, sequence_length, batch_size)

            # 计算"查询"和"键"之间的点积，得到原始注意力分数
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(self.attention_head_size)
            
            # 如果存在注意力遮罩，将其加到注意力分数上
            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask

            # 将注意力分数归一化为注意力概率
            attention_probs = nn.functional.softmax(attention_scores, dim=-1)

            # 对注意力概率应用注意力dropout
            attention_probs = self.attn_dropout(attention_probs)

            # 如果存在头部遮罩，将其应用到注意力概率上
            if head_mask is not None:
                attention_probs = attention_probs * head_mask

            # 计算注意力输出，将注意力概率与值层进行加权求和
            attn_output = torch.matmul(attention_probs, value_layer)
            attn_output = attn_output.transpose(1, 2).contiguous()
            
            # 重塑注意力输出的形状
            attn_output = attn_output.reshape(batch_size, sequence_length, self.all_head_size)

            # 通过self.dense对注意力输出进行全连接层操作
            attn_output = self.dense(attn_output)
            
            # 应用dropout到注意力输出上
            attn_output = self.dropout(attn_output)
            
            # 将层归一化应用到注意力输出与隐藏状态的残差上
            attn_output = self.layer_norm(attn_output + hidden_states)
            
            # 如果需要输出注意力信息，则将注意力概率加入到输出中
            outputs = (attn_output, attention_probs) if output_attentions else (attn_output,)
            return outputs
# 从 transformers.models.bert.modeling_bert.BertIntermediate 复制并修改为 TvpIntermediate
class TvpIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层，将输入大小为 config.hidden_size 转换为 config.intermediate_size
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 根据配置选择激活函数，若 config.hidden_act 是字符串则使用预定义的函数，否则直接使用配置中的函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 全连接层处理输入的 hidden_states
        hidden_states = self.dense(hidden_states)
        # 应用选择的激活函数到处理后的 hidden_states
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# 定义 TvpOutputLayer 类，包括线性层、LayerNorm 层和 Dropout 层
class TvpOutputLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层，将输入大小为 config.intermediate_size 转换为 config.hidden_size
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建一个 LayerNorm 层，标准化大小为 config.hidden_size 的输入张量
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个 Dropout 层，应用概率为 config.hidden_dropout_prob 的丢弃率
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 全连接层处理输入的 hidden_states
        hidden_states = self.dense(hidden_states)
        # 对全连接层的输出进行 Dropout 处理
        hidden_states = self.dropout(hidden_states)
        # 对加和后的结果应用 LayerNorm 处理
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        return hidden_states


# 定义 TvpEncodeLayer 类，包括 TvpAttention、TvpIntermediate 和 TvpOutputLayer 实例
class TvpEncodeLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个 TvpAttention 实例
        self.attention = TvpAttention(config)
        # 创建一个 TvpIntermediate 实例
        self.intermediate = TvpIntermediate(config)
        # 创建一个 TvpOutputLayer 实例
        self.output = TvpOutputLayer(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions: Optional[bool] = None,
    ):
        # 调用 attention 实例处理 hidden_states，并返回其输出
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # 如果输出注意力权重，则添加自注意力
        # 将 attention_output 输入到 intermediate 实例中进行处理
        intermediate_output = self.intermediate(attention_output)
        # 将 intermediate_output 和 attention_output 输入到 output 实例中进行处理
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs


# 定义 TvpEncoder 类，包括多个 TvpEncodeLayer 层和一些配置项
class TvpEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 创建一个 nn.ModuleList，其中包含 config.num_hidden_layers 个 TvpEncodeLayer 实例
        self.layer = nn.ModuleList([TvpEncodeLayer(config) for _ in range(config.num_hidden_layers)])
        # 设置梯度检查点为 False
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ):
            # 如果 return_dict 参数未指定，则使用配置中的默认值
            return_dict = return_dict if return_dict is not None else self.config.return_dict
            # 如果 output_attentions 参数未指定，则使用配置中的默认值
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            # 如果 output_hidden_states 参数未指定，则使用配置中的默认值
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            # 初始化用于存储所有层隐藏状态的元组
            all_hidden_states = ()
            # 初始化用于存储所有注意力权重的元组
            all_attentions = ()

            # 遍历每个层次的 Transformer 层
            for i, layer_module in enumerate(self.layer):
                # 如果需要输出隐藏状态
                if output_hidden_states:
                    # 将当前层的隐藏状态添加到 all_hidden_states 中
                    all_hidden_states = all_hidden_states + (hidden_states,)

                # 如果启用了梯度检查点且在训练阶段
                if self.gradient_checkpointing and self.training:
                    # 调用 _gradient_checkpointing_func 方法实现梯度检查点
                    layer_outputs = self._gradient_checkpointing_func(
                        layer_module.__call__,
                        hidden_states,
                        attention_mask,
                        (head_mask[i] if head_mask is not None else None),
                        output_attentions,
                    )
                else:
                    # 普通地调用 Transformer 层，得到层的输出
                    layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i], output_attentions)

                # 更新 hidden_states 为当前层的输出的第一个元素，即隐藏状态
                hidden_states = layer_outputs[0]
                # 如果需要输出注意力权重
                if output_attentions:
                    # 将当前层的注意力权重添加到 all_attentions 中
                    all_attentions = all_attentions + (layer_outputs[1],)

            # 添加最后一层的隐藏状态到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 如果不需要以字典形式返回结果
            if not return_dict:
                # 构造 outputs，包含最后一层的隐藏状态及可能的所有隐藏状态和注意力权重
                outputs = (hidden_states,)
                if output_hidden_states:
                    outputs = outputs + (all_hidden_states,)
                if output_attentions:
                    outputs = outputs + (all_attentions,)
                return outputs  # 返回最后一层的隐藏状态，所有隐藏状态和注意力权重的元组

            # 如果需要以 BaseModelOutput 对象形式返回结果
            return BaseModelOutput(
                last_hidden_state=hidden_states,
                hidden_states=all_hidden_states if output_hidden_states else None,
                attentions=all_attentions if output_attentions else None,
            )
# 从transformers.models.bert.modeling_bert.BertPooler复制而来，将Bert改为Tvp
class TvpPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化一个全连接层，输入输出维度都为config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 激活函数使用双曲正切函数
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 取hidden_states中每个样本的第一个token对应的隐藏状态作为池化输出
        first_token_tensor = hidden_states[:, 0]
        # 经过全连接层变换
        pooled_output = self.dense(first_token_tensor)
        # 应用激活函数
        pooled_output = self.activation(pooled_output)
        return pooled_output


class TvpPreTrainedModel(PreTrainedModel):
    """一个抽象类，用于处理权重初始化和预训练模型的下载加载的简单接口。"""

    config_class = TvpConfig  # 使用TvpConfig作为配置类
    base_model_prefix = "model"  # 基础模型前缀为"model"
    supports_gradient_checkpointing = True  # 支持梯度检查点

    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # 使用正态分布初始化权重，均值为0，标准差为self.config.initializer_range
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            # 将偏置项初始化为零，将权重初始化为1
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        if isinstance(module, nn.Linear) and module.bias is not None:
            # 如果是线性层且存在偏置项，则将偏置项初始化为零
            module.bias.data.zero_()

        if isinstance(module, nn.Conv2d):
            # 使用Kaiming正态分布初始化卷积层的权重
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                # 如果存在偏置项，则将偏置项初始化为零
                nn.init.constant_(module.bias, 0)


TVP_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`TvpConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

TVP_INPUTS_DOCSTRING = r"""
    # 定义函数签名和参数说明
    def forward(
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        attention_mask: torch.FloatTensor = None,
        head_mask: torch.FloatTensor = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True
    ):
        # 模型前向传播方法，接受输入的序列 token 索引和图像像素值作为输入
        """
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                输入序列 token 的索引，用于从词汇表中获取对应的 token。可使用 [`AutoTokenizer`] 获得。详见 [`PreTrainedTokenizer.encode`] 和 [`PreTrainedTokenizer.__call__`]。[什么是输入 ID?](../glossary#input-ids)
    
            pixel_values (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, height, width)`):
                像素值，用于表示输入图像。可使用 [`TvpImageProcessor`] 获得。详见 [`TvpImageProcessor.__call__`]。
    
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
                用于避免对填充 token 索引执行注意力操作的掩码。掩码取值为 `[0, 1]`:
                - 1 表示**不遮蔽**的 token，
                - 0 表示**遮蔽**的 token。
                [什么是注意力掩码?](../glossary#attention-mask)
    
            head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
                用于屏蔽自注意力模块中特定头部的掩码。掩码取值为 `[0, 1]`:
                - 1 表示**未遮蔽**的头部，
                - 0 表示**遮蔽**的头部。
    
            output_attentions (`bool`, *optional*):
                是否返回所有注意力层的注意力张量。详见返回的张量中的 `attentions` 字段。
    
            output_hidden_states (`bool`, *optional*):
                是否返回所有层的隐藏状态。详见返回的张量中的 `hidden_states` 字段。
    
            return_dict (`bool`, *optional*):
                是否返回 [`~utils.ModelOutput`] 而不是普通的元组。
        """
"""
Pad frames extracted from videos in the surroundings.
"""

class TvpFrameDownPadPrompter(nn.Module):
    """
    Pad frames extracted from videos only at the bottom.
    """

    def __init__(self, config):
        # 检查 `visual_prompter_apply` 是否为有效取值 ("add", "replace", "remove")
        if config.visual_prompter_apply not in ("add", "replace", "remove"):
            raise ValueError("`visual_prompter_apply` must be in (add, replace, remove)")

        super().__init__()
        # 初始化可视化提示大小、帧数、最大图像尺寸、应用方式
        self.visual_prompt_size = config.visual_prompt_size
        self.frame_num = config.frame_num
        self.max_img_size = config.max_img_size
        self.visual_prompter_apply = config.visual_prompter_apply

        # 创建用于下方填充的可训练参数
        self.pad_down = nn.Parameter(
            torch.randn([1, config.frame_num, 3, config.visual_prompt_size, config.max_img_size])
        )

    def forward(self, pixel_values):
        # 如果不是应用 "add" 方式，则创建一个全为1的掩码并设置底部为0
        if self.visual_prompter_apply != "add":
            visual_prompt_mask = torch.ones(
                [self.max_img_size, self.max_img_size], dtype=pixel_values.dtype, device=pixel_values.device
            )
            visual_prompt_mask[self.max_img_size - self.visual_prompt_size : self.max_img_size, :] = 0.0
            pixel_values *= visual_prompt_mask

        # 如果不是应用 "remove" 方式，则创建一个填充用的零张量，并在指定位置填充下方填充数据
        if self.visual_prompter_apply != "remove":
            prompt = torch.zeros(
                [pixel_values.shape[0], pixel_values.shape[1], 3, self.max_img_size, self.max_img_size],
                device=pixel_values.device,
            )
            start_point = self.max_img_size - self.visual_prompt_size
            prompt[:, :, :, start_point : self.max_img_size, :] = self.pad_down
            pixel_values += prompt.to(pixel_values.dtype)

        return pixel_values


class TvpFramePadPrompter(nn.Module):
    """
    Pad frames extracted from videos in the surroundings.
    """
    # 初始化方法，接收一个配置对象 `config`
    def __init__(self, config):
        # 检查 `visual_prompter_apply` 是否在合法取值范围内
        if config.visual_prompter_apply not in ("add", "replace", "remove"):
            raise ValueError("`visual_prompter_apply` must be in (add, replace, remove)")

        # 调用父类初始化方法
        super().__init__()

        # 初始化属性
        self.num_frames = config.num_frames  # 设置帧数
        self.max_img_size = config.max_img_size  # 设置图像最大尺寸
        self.visual_prompter_apply = config.visual_prompter_apply  # 设置视觉提示器应用模式

        # 根据配置计算基础尺寸
        self.base_size = config.max_img_size - config.visual_prompt_size * 2

        # 初始化可学习参数：上边界填充
        self.pad_up = nn.Parameter(
            torch.randn([1, config.num_frames, 3, config.visual_prompt_size, config.max_img_size])
        )
        # 初始化可学习参数：下边界填充
        self.pad_down = nn.Parameter(
            torch.randn([1, config.num_frames, 3, config.visual_prompt_size, config.max_img_size])
        )
        # 初始化可学习参数：左边界填充
        self.pad_left = nn.Parameter(
            torch.randn(
                [
                    1,
                    config.num_frames,
                    3,
                    config.max_img_size - config.visual_prompt_size * 2,
                    config.visual_prompt_size,
                ]
            )
        )
        # 初始化可学习参数：右边界填充
        self.pad_right = nn.Parameter(
            torch.randn(
                [
                    1,
                    config.num_frames,
                    3,
                    config.max_img_size - config.visual_prompt_size * 2,
                    config.visual_prompt_size,
                ]
            )
        )

    # 前向传播方法，接收输入 `pixel_values`
    def forward(self, pixel_values):
        # 检查 `visual_prompter_apply` 是否在合法取值范围内，若不在则抛出异常
        if self.visual_prompter_apply not in ("add", "remove", "replace"):
            raise ValueError(f"Invalid visual_prompter_apply value {self.visual_prompter_apply}")

        # 如果 `visual_prompter_apply` 是 "replace" 或 "remove"，则创建全为 1 的视觉提示掩码
        if self.visual_prompter_apply in ("replace", "remove"):
            visual_prompt_mask = torch.ones(
                [self.max_img_size, self.max_img_size], dtype=pixel_values.dtype, device=pixel_values.device
            )
            # 将输入 `pixel_values` 与视觉提示掩码相乘
            pixel_values *= visual_prompt_mask

        # 如果 `visual_prompter_apply` 是 "replace" 或 "add"，则进行以下操作
        if self.visual_prompter_apply in ("replace", "add"):
            # 创建全零的基础张量
            base = torch.zeros(1, self.num_frames, 3, self.base_size, self.base_size, device=pixel_values.device)
            # 拼接左右填充到基础张量上
            prompt = torch.cat([self.pad_left, base, self.pad_right], dim=4)
            # 拼接上下填充到最终的视觉提示器上
            prompt = torch.cat([self.pad_up, prompt, self.pad_down], dim=3)
            # 将视觉提示器复制并与输入 `pixel_values` 相加
            prompt = torch.cat(pixel_values.size(0) * [prompt])
            pixel_values = pixel_values + prompt.to(pixel_values.dtype)

        # 返回处理后的 `pixel_values`
        return pixel_values
# 定义了一个映射，将字符串映射到相应的 TvpFrameDownPadPrompter 或 TvpFramePadPrompter 类
TVP_PROMPTER_CLASSES_MAPPING = {
    "framedownpad": TvpFrameDownPadPrompter,
    "framepad": TvpFramePadPrompter,
}

@add_start_docstrings(
    "The bare Tvp Model transformer outputting BaseModelOutputWithPooling object without any specific head on" " top.",
    TVP_START_DOCSTRING,
)
# 定义了一个 TvpModel 类，继承自 TvpPreTrainedModel 类
class TvpModel(TvpPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        # 创建 TvpVisionModel 实例
        self.vision_model = TvpVisionModel(config)
        # 创建 TvpTextInputEmbeddings 实例
        self.embeddings = TvpTextInputEmbeddings(config)
        # 创建 TvpVisualInputEmbedding 实例
        self.visual_embeddings = TvpVisualInputEmbedding(config)
        # 创建 TvpEncoder 实例
        self.encoder = TvpEncoder(config)
        # 创建 TvpPooler 实例
        self.pooler = TvpPooler(config)
        # 创建 nn.Parameter 参数，形状为 [1, 10, hidden_size]
        self.text_prompt = nn.Parameter(torch.randn([1, 10, config.hidden_size]))
        # 创建 nn.Dropout 实例，使用给定的 hidden_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 检查 config.visual_prompter_type 是否在 TVP_PROMPTER_CLASSES_MAPPING 中，如果不在则抛出 ValueError
        if config.visual_prompter_type not in TVP_PROMPTER_CLASSES_MAPPING:
            raise ValueError("`visual_prompter_type` must be in (framedownpad, framepad)")
        # 根据 config.visual_prompter_type 创建相应的 Prompter 类实例
        self.visual_prompter = TVP_PROMPTER_CLASSES_MAPPING[config.visual_prompter_type](config)

        # 执行初始化后的处理
        self.post_init()

    def get_input_embeddings(self):
        # 返回 embeddings 的 word_embeddings 属性
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        # 设置 embeddings 的 word_embeddings 属性为给定的 value
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base class PreTrainedModel
        """
        # 遍历 heads_to_prune 中的每个项，将对应层的注意力头进行修剪
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_model_forward(TVP_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=TvpConfig)
    # 定义了 forward 方法，接受多个输入参数，并返回一个 BaseModelOutputWithPooling 对象
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
):
    # 下面是 TvpVideoGroundingHead 类的定义和注释，略去了前面的部分
    def __init__(self, config):
        # 调用父类的初始化方法，传入配置参数
        super().__init__(config)
        # 将配置参数保存到实例变量中
        self.config = config
        # 创建一个 TvpModel 的实例，并保存到实例变量中
        self.model = TvpModel(config)
        # 创建一个 TvpVideoGroundingHead 的实例，并保存到实例变量中
        self.video_grounding_head = TvpVideoGroundingHead(config)

        # 执行初始化后的自定义操作
        self.post_init()

    @add_start_docstrings_to_model_forward(TVP_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=TvpVideoGroundingOutput, config_class=TvpConfig)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        labels: Tuple[torch.Tensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        r"""
        labels (`torch.FloatTensor` of shape `(batch_size, 3)`, *optional*):
            The labels contains duration, start time, and end time of the video corresponding to the text.
        Returns:

        Examples:
        ```
        >>> import torch
        >>> from transformers import AutoConfig, AutoTokenizer, TvpForVideoGrounding

        >>> model = TvpForVideoGrounding.from_pretrained("Jiqing/tiny-random-tvp")

        >>> tokenizer = AutoTokenizer.from_pretrained("Jiqing/tiny-random-tvp")

        >>> pixel_values = torch.rand(1, 1, 3, 448, 448)
        >>> text_inputs = tokenizer("This is an example input", return_tensors="pt")
        >>> output = model(text_inputs.input_ids, pixel_values, text_inputs.attention_mask)
        ```"""
        # 如果 return_dict 为 None，则使用配置参数中的 return_dict
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        # 调用模型的 forward 方法，传入各种输入参数
        outputs = self.model(
            input_ids,
            pixel_values,
            attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 从模型输出中获取 pooler_output
        pooler_output = outputs[1]

        # 将 pooler_output 传入视频 grounding 头部模型中得到 logits
        logits = self.video_grounding_head(pooler_output)

        # 初始化 loss 为 None
        loss = None
        # 如果 labels 不为 None，则计算损失
        if labels is not None:
            # 创建损失函数对象，包括 iou, distance, duration 三种损失
            criterion = TvpLoss(["iou", "distance", "duration"])
            # 将损失函数移动到当前设备（通常是 GPU）
            criterion.to(self.device)
            # 计算损失字典
            loss_dict = criterion(logits, labels)
            # 计算加权损失总和
            loss = (
                loss_dict["iou"]
                + self.config.distance_loss_weight * loss_dict["distance"]
                + self.config.duration_loss_weight * loss_dict["duration"]
            )

        # 如果 return_dict 为 False，则返回不同类型的输出
        if not return_dict:
            # 将 logits 添加到输出中
            outputs = (logits,) + outputs[2:]
            # 如果损失不为 None，则也添加到输出中
            if loss is not None:
                outputs = (loss,) + outputs
            return outputs

        # 如果 return_dict 为 True，则构造 TvpVideoGroundingOutput 对象返回
        return TvpVideoGroundingOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```