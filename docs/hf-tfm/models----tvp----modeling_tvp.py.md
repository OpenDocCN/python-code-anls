# `.\transformers\models\tvp\modeling_tvp.py`

```py
# 指定文件编码为 UTF-8
# 版权声明
# 版权所有的 2023 年英特尔 AIA 团队作者和 HuggingFace Inc. 团队。保留所有权利。
# 根据 Apache 许可证，版本 2.0（“许可证”）授权。
# 除非按照许可证的约定，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 均按“原样”基础分发，不附带任何明示或暗示的担保或条件。
# 请查看许可证以了解特定语言控制权限和
# 许可证下的限制。
"""PyTorch TVP 模型"""

# 导入需要的库
import math
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.utils.checkpoint
from torch import nn
from ...activations import ACT2FN
from ...file_utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from ...modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ModelOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import prune_linear_layer
from ...utils import logging
from ..auto import AutoBackbone
from .configuration_tvp import TvpConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# TVP 预训练模型的存档列表
TVP_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "Intel/tvp-base",
    "Intel/tvp-base-ANet",
    # 查看所有 Tvp 模型 https://huggingface.co/models?filter=tvp
]

# TvpVideoGroundingOutput 类，继承自 ModelOutput 类
@dataclass
class TvpVideoGroundingOutput(ModelOutput):
    """
    参数:
        loss(`torch.FloatTensor` 的形状为`(1,)`，*可选*，当`return_loss` 为 `True` 时返回):
            视频定位的时间距离 IoU 损失。
        logits(`torch.FloatTensor` 的形状为`(batch_size, 2)`):
            包含开始时间/持续时间和结束时间/持续时间。这是视频与输入文本对应的时间段。
        hidden_states(`tuple(torch.FloatTensor)` 的形状为`(batch_size, sequence_length, hidden_size)`，*可选*，当传递 `output_hidden_states=True` 或 `config.output_hidden_states=True` 时返回):
            `torch.FloatTensor` 元组（如果模型有嵌入层则为嵌入层的输出 + 每个层的输出）的形状为 `(batch_size, sequence_length, hidden_size)`。在每个层的模型的隐藏状态的输出和可选的初始嵌入输出。
        attentions(`tuple(torch.FloatTensor)` 的形状为`(batch_size, num_heads, sequence_length, sequence_length)`，*可选*，当传递 `output_attentions=True` 或 `config.output_attentions=True` 时返回):
            `torch.FloatTensor` 元组（每个层一个）的形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


# TvpLoss 类，继承自 nn.Module 类
class TvpLoss(nn.Module):
    """
    # `TvpForVideoGrounding`类用于计算损失。该过程分为两步：
    # 1) 计算真实框与模型输出之间的匈牙利分配
    # 2) 监督每对匹配的真实框/预测（监督类别和框）

    def __init__(self, losses):
        # 调用父类初始化方法
        super().__init__()
        # 将损失函数名称映射到相应的损失函数方法
        self.loss_map = {
            "iou": self.loss_iou,
            "distance": self.loss_distance,
            "duration": self.loss_duration,
        }
        # 遍历损失列表
        for loss in losses:
            # 如果损失函数不在损失映射中，则抛出异常
            if loss not in self.loss_map:
                raise ValueError(f"Loss {loss} not supported")

        # 保存损失列表
        self.losses = losses

    # 计算 IoU 损失函数
    def loss_iou(self, start_time, end_time, candidates_start_time, candidates_end_time, duration):
        """
        Measure the intersection over union.
        """
        # 计算交集
        inter = torch.min(candidates_end_time, end_time) - torch.max(candidates_start_time, start_time)
        # 计算并集
        union = torch.max(candidates_end_time, end_time) - torch.min(candidates_start_time, start_time)
        # 计算 IoU
        iou = 1 - inter.clamp(min=0) / union

        return iou

    # 计算距离损失函数
    def loss_distance(self, start_time, end_time, candidates_start_time, candidates_end_time, duration):
        """
        Measure the distance of mid points.
        """
        # 计算候选框和真实框的中点
        mid_candidates = torch.div(torch.add(candidates_start_time, candidates_end_time), 2.0)
        mid_groundtruth = torch.div(torch.add(start_time, end_time), 2.0)
        # 计算中点之间的距离
        distance_diff = torch.div(
            torch.max(mid_candidates, mid_groundtruth) - torch.min(mid_candidates, mid_groundtruth), duration
        ).clamp(min=0.2)

        return distance_diff

    # 计算时长损失函数
    def loss_duration(self, start_time, end_time, candidates_start_time, candidates_end_time, duration):
        """
        Measure the difference of duration.
        """
        # 计算候选框和真实框的持续时间
        duration_candidates = torch.sub(candidates_end_time, candidates_start_time)
        duration_groundtruth = torch.sub(end_time, start_time)
        # 计算持续时间之间的差异
        duration_diff = torch.square(torch.div(torch.sub(duration_candidates, duration_groundtruth), duration))
        duration_diff = duration_diff.clamp(min=0.4)

        return duration_diff
    # 定义前向传播方法，用于计算损失值
    def forward(self, logits, labels):
        """
        This performs the loss computation.

        Args:
            logits (`torch.FloatTensor`):
                The output logits of head module.
            labels (`List[torch.FloatTensor]`):
                List of tensors ([start, end, duration]), which contains start time, end time of the video corresponding to the text, and also the duration.
        """
        # 从标签中获取视频的持续时间、开始时间和结束时间
        duration, start_time, end_time = labels
        # 将输出logits乘以视频持续时间得到候选值
        candidates = torch.mul(logits, duration)
        # 将候选值分别取出开始时间和结束时间
        candidates_start_time, candidates_end_time = candidates[:, 0].float(), candidates[:, 1].float()

        # 初始化损失字典
        losses_dict = {}
        # 遍历每种损失函数并计算对应的损失值，更新到损失字典中
        for loss in self.losses:
            losses_dict.update(
                {loss: self.loss_map[loss](start_time, end_time, candidates_start_time, candidates_end_time, duration)}
            )

        # 返回损失值字典
        return losses_dict
# 定义一个继承自 nn.Module 的 TvpVisionModel 类
class TvpVisionModel(nn.Module):
    # 初始化函数，接收一个配置参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 使用配置参数的后骨结构配置创建自动骨干网络
        self.backbone = AutoBackbone.from_config(config.backbone_config)
        # 定义一个卷积层，用于对网格编码器的特征进行处理
        self.grid_encoder_conv = nn.Conv2d(
            config.backbone_config.hidden_sizes[-1], # 输入通道数为骨干网络最后一层的隐藏单元数
            config.hidden_size, # 输出通道数为配置中设置的隐藏单元数
            kernel_size=3, # 卷积核大小为3
            stride=1, # 步长为1
            padding=1, # 填充大小为1
            groups=1, # 分组卷积数为1
            bias=False, # 不使用偏置
        )

    # 前向传播函数，接收像素值作为输入
    def forward(self, pixel_values):
        # 获取像素值的形状信息
        batch_size, num_frames, num_channels, height, width = pixel_values.shape
        # 重新调整像素值的维度
        pixel_values = pixel_values.view(batch_size * num_frames, num_channels, height, width)
        # 调用骨干网络进行前向传播，获取特征地图输出
        grid_feat_outputs = self.backbone(pixel_values)["feature_maps"][0]
        # 对特征地图进行卷积处理
        grid = self.grid_encoder_conv(grid_feat_outputs)
        # 最大池化操作
        grid = nn.functional.max_pool2d(grid, kernel_size=2, stride=2)
        # 使用 ReLU 激活函数
        grid = nn.functional.relu(grid, inplace=True)
        # 获取处理后的特征地图的形状信息
        new_channel, new_height, new_width = grid.shape[-3:]
        # 重新调整特征地图的维度
        grid = grid.view(batch_size, num_frames, new_channel, new_height, new_width)
        # 转置特征地图的维度
        grid = grid.permute(0, 1, 3, 4, 2)
        return grid

# 定义一个继承自 nn.Module 的 TvpVisualInputEmbedding 类
class TvpVisualInputEmbedding(nn.Module):
    """
    Takes input of both image and video (multi-frame)
    """

    # 初始化函数，接收一个配置参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 定义位置嵌入、行位置嵌入、列位置嵌入和标记类型嵌入
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.row_position_embeddings = nn.Embedding(config.max_grid_row_position_embeddings, config.hidden_size)
        self.col_position_embeddings = nn.Embedding(config.max_grid_col_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(1, config.hidden_size)
        # 定义层归一化和 Dropout 操作
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
```  
    def add_2d_positional_embeddings(self, grid):
        """
        Args:
            grid: (batch_size, height, width, hidden_dim)
        Returns:
            grid + col_position_embeddings.view(*col_shape): (batch_size, *, height, width, hidden_dim)
        """
        # 获取 grid 的形状
        batch_size, height, width, hidden_dim = grid.shape

        # 添加行位置嵌入
        row_position_ids = torch.arange(height, dtype=torch.long, device=grid.device)  # (height, )
        row_position_embeddings = self.row_position_embeddings(row_position_ids)  # (height, hidden_dim)
        row_shape = (1,) * (len(grid.shape) - 3) + (height, 1, hidden_dim)  # (1, height, 1, hidden_dim)
        grid = grid + row_position_embeddings.view(*row_shape)  # 自动广播

        # 添加列位置嵌入
        col_position_ids = torch.arange(width, dtype=torch.long, device=grid.device)  # (width, )
        col_position_embeddings = self.col_position_embeddings(col_position_ids)  # (width, hidden_dim)
        col_shape = (batch_size, 1, width, hidden_dim)  # (batch_size, 1, width, hidden_dim)
        return grid + col_position_embeddings.view(*col_shape)  # 自动广播

    def forward(self, grid):
        """
        Args:
            grid: Array of shape (batch_size, num_frames, height, width, num_channels).
                It contains processed frames extracted from videos, and is generated by Tvp image preprocessor. Note,
                num_frames can be 1

        Returns:
            embeddings: The embedding of grid with size (batch_size, height*width, num_channels)

        """
        # 获取 grid 的形状
        batch_size, num_frames, height, width, num_channels = grid.shape
        # 对 num_frames 进行平均池化处理，得到 (batch_size, height, width, hidden_size)
        grid = grid.mean(1)
        # 给 grid 添加 2D 位置嵌入
        grid = self.add_2d_positional_embeddings(grid)
        # 将 grid 变形为 (batch_size, height*width, num_channels)
        visual_tokens = grid.view(batch_size, -1, num_channels)
        visual_tokens_shape = visual_tokens.shape[:-1]
        device = visual_tokens.device

        # 图像 token 类型嵌入
        token_type_ids = torch.zeros(visual_tokens_shape, dtype=torch.long, device=device)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 计算嵌入
        embeddings = visual_tokens + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
```py 
class TvpTextInputEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        # 初始化文本输入嵌入层
        super().__init__()
        # 初始化词嵌入层，将词汇转换为向量表示，带有填充索引
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 初始化位置嵌入层，将位置编码转换为向量表示
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # 初始化令牌类型嵌入层，将令牌类型转换为向量表示
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        # 初始化层归一化层，用于归一化嵌入向量
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化丢弃层，用于随机丢弃嵌入向量的部分值，以减少过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        # 如果输入的词汇索引不为空，则获取输入的形状
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        # 获取序列长度
        seq_length = input_shape[1]
        # 获取设备信息
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        # 如果位置编码为空，则生成位置编码
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(input_shape)
        # 如果令牌类型为空，则生成令牌类型
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # 如果输入的嵌入向量为空，则使用词嵌入层生成嵌入向量
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        # 获取位置嵌入向量
        position_embeddings = self.position_embeddings(position_ids)
        # 获取令牌类型嵌入向量
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 将词嵌入向量、位置嵌入向量和令牌类型嵌入向量相加作为最终的嵌入向量
        embeddings = inputs_embeds + position_embeddings + token_type_embeddings
        # 对嵌入向量进行层归一化
        embeddings = self.layer_norm(embeddings)
        # 对嵌入向量进行丢弃操作
        embeddings = self.dropout(embeddings)
        # 返回嵌入向量
        return embeddings


class TvpAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 如果隐藏层大小不能被注意力头数整除，且配置中没有嵌入大小，则引发异常
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size {config.hidden_size} is not a multiple of the number of attention heads {config.num_attention_heads}"
            )

        # 设置注意力头数和每个头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 初始化查询、键、值的线性变换层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        # 初始化注意力层的丢弃操作
        self.attn_dropout = nn.Dropout(config.attention_probs_dropout_prob)

        # 初始化全连接层和层归一化层
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化注意力层的丢弃操作
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 初始化被修剪的注意力头集合
        self.pruned_heads = set()
    def prune_heads(self, heads):
        按照给定的头部信息对注意力头进行裁剪
        if len(heads) == 0:
            如果头部信息列表为空，直接返回
            return
        创建一个全为1的掩码矩阵，形状为(self.num_attention_heads, self.attention_head_size)
        heads = set(heads) - self.pruned_heads  # 转换为集合并移除已裁剪的头部信息
        遍历每个头部信息
        for head in heads:
            计算当前头部信息前面裁剪掉的头部数目，然后相应地调整头部信息的索引
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        将掩码矩阵展平，并保持连续性，然后筛选出值为1的索引
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()

        # 裁剪线性层
        对查询、键、值和稠密层进行线性层裁剪
        self.query = prune_linear_layer(self.query, index)
        self.key = prune_linear_layer(self.key, index)
        self.value = prune_linear_layer(self.value, index)
        self.dense = prune_linear_layer(self.dense, index, dim=1)

        # 更新超参数并存储已裁剪的头部信息
        更新注意力头数目和总头部尺寸
        self.num_attention_heads = self.num_attention_heads - len(heads)
        self.all_head_size = self.attention_head_size * self.num_attention_heads
        将已裁剪的头部信息添加到集合中
        self.pruned_heads = self.pruned_heads.union(heads)

    def _reshape(self, tensor: torch.Tensor, sequence_length: int, batch_size: int):
        将给定的张量重新形状，变为(batch_size, sequence_length, self.num_attention_heads, self.attention_head_size)
        然后交换维度1和维度2，并保持连续性

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions: Optional[bool] = None,
    ):
        # 获取隐藏状态的批次大小和序列长度
        batch_size, sequence_length = hidden_states.shape[:2]
        # 使用 self.query 对隐藏状态进行查询操作
        mixed_query_layer = self.query(hidden_states)

        # 使用 self.key 对隐藏状态进行关键词查询操作
        mixed_key_layer = self.key(hidden_states)
        # 使用 self.value 对隐藏状态进行数值查询操作
        mixed_value_layer = self.value(hidden_states)

        # 将查询层、关键词层、数值层重塑为指定形状
        query_layer = self._reshape(mixed_query_layer, sequence_length, batch_size)
        key_layer = self._reshape(mixed_key_layer, sequence_length, batch_size)
        value_layer = self._reshape(mixed_value_layer, sequence_length, batch_size)

        # 计算“查询”和“关键词”之间的点积，得到原始注意力分数
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # 将注意力分数归一化为概率
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # 这实际上是删除整个要参与注意力的标记，这可能有点不寻常，但是取自原始Transformer论文。
        attention_probs = self.attn_dropout(attention_probs)

        # 如果需要，对头部进行掩码操作
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # 使用注意力概率和数值层进行加权和操作
        attn_output = torch.matmul(attention_probs, value_layer)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, sequence_length, self.all_head_size)

        # 传递加权和的结果给全连接层
        attn_output = self.dense(attn_output)
        # 对全连接层的输出进行dropout操作
        attn_output = self.dropout(attn_output)
        # 注意力输出加上隐藏状态后进行 Layer Norm 操作
        attn_output = self.layer_norm(attn_output + hidden_states)
        # 如果需要输出注意力，则将注意力输出一起返回，否则只返回注意力输出
        outputs = (attn_output, attention_probs) if output_attentions else (attn_output,)
        return outputs
# Copied from transformers.models.bert.modeling_bert.BertIntermediate with Bert->Tvp

# 从 transformers.models.bert.modeling_bert.BertIntermediate 复制过来，将Bert重命名为Tvp的模型
class TvpIntermediate(nn.Module):
    # 构造函数
    def __init__(self, config):
        super().__init__()
        # 定义一个全连接层linear操作，将输入维度从config.hidden_size映射到config.intermediate_size
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 判断config.hidden_act是否为字符串类型
        if isinstance(config.hidden_act, str):
            # 如果是字符串类型，则使用字典ACT2FN中对应的激活函数作为中间层的激活函数
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            # 如果不是字符串类型，则直接使用config.hidden_act作为中间层的激活函数
            self.intermediate_act_fn = config.hidden_act

    # 前向传播函数
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将输入的hidden_states经过全连接层映射到中间层的维度
        hidden_states = self.dense(hidden_states)
        # 对映射后的结果进行激活函数处理
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 返回处理后的结果
        return hidden_states


# 定义TvpOutputLayer类
class TvpOutputLayer(nn.Module):
    # 构造函数
    def __init__(self, config):
        super().__init__()
        # 定义一个全连接层linear操作，将输入维度从config.intermediate_size映射到config.hidden_size
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # LayerNorm层，对每个样本的所有维度进行归一化操作
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Dropout操作，对输出的hidden_states中的每个元素丢弃一定概率
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将输入的hidden_states经过全连接层映射到输出维度
        hidden_states = self.dense(hidden_states)
        # 对映射后的结果进行dropout操作
        hidden_states = self.dropout(hidden_states)
        # 将dropout后的结果与输入的hidden_states相加，然后进行LayerNorm归一化
        hidden_states = self.layer_norm(hidden_states + input_tensor)
        # 返回处理后的结果
        return hidden_states


# 定义TvpEncodeLayer类
class TvpEncodeLayer(nn.Module):
    # 构造函数
    def __init__(self, config):
        super().__init__()
        # 定义一个TvpAttention的实例
        self.attention = TvpAttention(config)
        # 定义一个TvpIntermediate的实例
        self.intermediate = TvpIntermediate(config)
        # 定义一个TvpOutputLayer的实例
        self.output = TvpOutputLayer(config)

    # 前向传播函数
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions: Optional[bool] = None,
    ):
        # 调用self.attention模块进行前向传播，获取self_attention_outputs
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
        )
        # 获取self_attention_outputs的第一个元素，即attention_output
        attention_output = self_attention_outputs[0]
        # 获取self_attention_outputs的后面元素，即除了attention_output之外的其他元素，并存储在outputs变量中
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
        # 调用self.intermediate模块进行前向传播，获取intermediate_output
        intermediate_output = self.intermediate(attention_output)
        # 调用self.output模块进行前向传播，获取layer_output
        layer_output = self.output(intermediate_output, attention_output)
        # 将layer_output添加到outputs中
        outputs = (layer_output,) + outputs
        # 返回outputs
        return outputs


# 定义TvpEncoder类
class TvpEncoder(nn.Module):
    # 构造函数
    def __init__(self, config):
        super().__init__()
        # 将config赋值给self.config
        self.config = config
        # 生成一个长度为config.num_hidden_layers的列表，每个元素都是TvpEncodeLayer(config)的实例
        self.layer = nn.ModuleList([TvpEncodeLayer(config) for _ in range(config.num_hidden_layers)])
        # gradient_checkpointing默认为False
        self.gradient_checkpointing = False

    # 前向传播函数
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ):
        # 遍历self.layer中的每个TvpEncodeLayer实例，获取每个layer的输出结果
        for layer_module in self.layer:
            # 调用TvpEncodeLayer的forward函数
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                head_mask,
                output_attentions=output_attentions,
            )
            # 更新hidden_states为layer_outputs的第一个元素
            hidden_states = layer_outputs[0]

        # 如果return_dict为True，则返回一个字典类型
        if return_dict:
            return TvpBaseModelOutput(
                last_hidden_state=hidden_states,
                hidden_states=None,
                attentions=None,
            )
        # 否则，返回hidden_states
        else:
            return hidden_states
        ):
        # 设置默认返回值为self.config.return_dict，如果return_dict为None则使用默认值
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        # 设置默认输出注意力的值为self.config.output_attentions，如果output_attentions为None则使用默认值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 设置默认输出隐藏状态值为self.config.output_hidden_states，如果output_hidden_states为None则使用默认值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 初始化存储所有隐藏状态和注意力的空元组
        all_hidden_states = ()
        all_attentions = ()

        # 遍历每个层的模块
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态值
            if output_hidden_states:
                # 将当前层的隐藏状态添加到all_hidden_states元组中
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 如果使用梯度检查点并且处于训练状态下
            if self.gradient_checkpointing and self.training:
                # 调用_gradient_checkpointing_func函数执行梯度检查点
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    (head_mask[i] if head_mask is not None else None),
                    output_attentions,
                )
            else:
                # 否则，正常执行当前层的前向传播
                layer_outputs = layer_module(hidden_states, attention_mask, head_mask[i], output_attentions)

            # 更新隐藏状态为当前层的输出
            hidden_states = layer_outputs[0]
            # 如果需要输出注意力值
            if output_attentions:
                # 将当前层的注意力值添加到all_attentions元组中
                all_attentions = all_attentions + (layer_outputs[1],)

        # 添加最后一层的隐藏状态
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不需要以字典形式返回结果
        if not return_dict:
            outputs = (hidden_states,)
            # 如果需要输出所有隐藏状态
            if output_hidden_states:
                outputs = outputs + (all_hidden_states,)
            # 如果需要输出所有注意力值
            if output_attentions:
                outputs = outputs + (all_attentions,)
            # 返回结果元组，包括最后一层的隐藏状态，所有隐藏状态和所有注意力值
            return outputs  # last-layer hidden state, (all hidden states), (all attentions)

        # 以BaseModelOutput的形式返回结果，包括最后一层的隐藏状态，所有隐藏状态和所有注意力值
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states if output_hidden_states else None,
            attentions=all_attentions if output_attentions else None,
        )
# 从transformers.models.bert.modeling_bert.BertPooler中复制代码，并将Bert->Tvp
class TvpPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)  # 用于线性变换
        self.activation = nn.Tanh()  # Tanh激活函数

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 通过获取对应于第一个标记的隐藏状态来"池化"模型
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)  # 线性变换
        pooled_output = self.activation(pooled_output)  # Tanh激活函数
        return pooled_output


class TvpPreTrainedModel(PreTrainedModel):
    """处理权重初始化、预训练模型下载和加载的抽象类"""

    config_class = TvpConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """初始化权重"""
        if isinstance(module, (nn.Linear, nn.Embedding)):  # 如果是线性层或嵌入层
            # 与TF版本略有不同，TF版本使用截断正态分布进行初始化
            # 参考 https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):  # 如果是LayerNorm层
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        if isinstance(module, nn.Linear) and module.bias is not None:  # 如果是线性层并且有偏置
            module.bias.data.zero_()

        if isinstance(module, nn.Conv2d):  # 如果是二维卷积层
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")  # 使用Kaiming正态分布初始化权重
            if module.bias is not None:
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
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            # 输入序列标记在词汇表中的索引。可以使用 [`AutoTokenizer`] 获取索引。详见 [`PreTrainedTokenizer.encode`] 和 [`PreTrainedTokenizer.__call__`]。[什么是输入 ID？](../glossary#input-ids)

        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_frames, num_channels, height, width)`):
            # 像素值。可以使用 [`TvpImageProcessor`] 获取像素值。详见 [`TvpImageProcessor.__call__`]。

        attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
            # 避免在填充标记索引上执行注意力的掩码。掩码值选择在 `[0, 1]`：
            # - 1 表示**未屏蔽**的标记，
            # - 0 表示**屏蔽**的标记。
            [什么是注意力掩码？](../glossary#attention-mask)

        head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
            # 用于使自注意力模块中的部分头部失效的掩码。掩码值选择在 `[0, 1]`：
            # - 1 表示头部**未屏蔽**，
            # - 0 表示头部**屏蔽**。

        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。有关更多详细信息，请参见返回张量中的 `attentions`。

        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。有关更多详细信息，请参见返回张量中的 `hidden_states`。

        return_dict (`bool`, *optional*):
            # 是否返回 [`~utils.ModelOutput`] 而不是普通元组。
# 创建一个名为TvpFrameDownPadPrompter的类，继承自nn.Module类
class TvpFrameDownPadPrompter(nn.Module):
    """
    Pad frames extracted from videos only at the bottom.
    """

    # 初始化方法，接受一个config参数
    def __init__(self, config):
        # 检查visual_prompter_apply是否在指定范围内，如果不是则抛出数值错误
        if config.visual_prompter_apply not in ("add", "replace", "remove"):
            raise ValueError("`visual_prompter_apply` must be in (add, replace, remove)")

        # 调用父类的初始化方法
        super().__init__()
        # 初始化类的属性
        self.visual_prompt_size = config.visual_prompt_size
        self.frame_num = config.frame_num
        self.max_img_size = config.max_img_size
        self.visual_prompter_apply = config.visual_prompter_apply

        # 创建一个3维的可训练参数
        self.pad_down = nn.Parameter(
            torch.randn([1, config.frame_num, 3, config.visual_prompt_size, config.max_img_size])
        )

    # 前向传播方法，接受一个名为pixel_values的参数
    def forward(self, pixel_values):
        # 如果visual_prompter_apply不等于"add"
        if self.visual_prompter_apply != "add":
            # 创建一个全为1的视觉提示蒙版
            visual_prompt_mask = torch.ones(
                [self.max_img_size, self.max_img_size], dtype=pixel_values.dtype, device=pixel_values.device
            )
            # 将部分区域赋值为0
            visual_prompt_mask[self.max_img_size - self.visual_prompt_size : self.max_img_size, :] = 0.0
            # 对输入像素值应用蒙版
            pixel_values *= visual_prompt_mask
        # 如果visual_prompter_apply不等于"remove"
        if self.visual_prompter_apply != "remove":
            # 创建一个与pixel_values相同维度的全为0的张量
            prompt = torch.zeros(
                [pixel_values.shape[0], pixel_values.shape[1], 3, self.max_img_size, self.max_img_size],
                device=pixel_values.device,
            )
            # 计算起始点
            start_point = self.max_img_size - self.visual_prompt_size
            # 将pad_down张量添加到prompt的指定位置
            prompt[:, :, :, start_point : self.max_img_size, :] = self.pad_down
            # 将prompt转换成与pixel_values相同数据类型的张量，并加到pixel_values上
            pixel_values += prompt.to(pixel_values.dtype)
        # 返回pixel_values
        return pixel_values


# 创建一个名为TvpFramePadPrompter的类，继承自nn.Module类
class TvpFramePadPrompter(nn.Module):
    """
    Pad frames extracted from videos in the surroundings.
    """
    # 初始化方法，接受一个配置对象作为参数
    def __init__(self, config):
        # 检查配置中的视觉提示应用是否在预定义的选项中
        if config.visual_prompter_apply not in ("add", "replace", "remove"):
            # 如果不在预定义选项中，则抛出数值错误
            raise ValueError("`visual_prompter_apply` must be in (add, replace, remove)")

        # 调用父类的初始化方法
        super().__init__()
        # 设置帧数
        self.num_frames = config.num_frames
        # 设置最大图像尺寸
        self.max_img_size = config.max_img_size
        # 设置视觉提示应用方式
        self.visual_prompter_apply = config.visual_prompter_apply

        # 计算基础大小，即图像尺寸减去两个视觉提示的大小
        self.base_size = config.max_img_size - config.visual_prompt_size * 2
        # 创建上方填充参数
        self.pad_up = nn.Parameter(
            torch.randn([1, config.num_frames, 3, config.visual_prompt_size, config.max_img_size])
        )
        # 创建下方填充参数
        self.pad_down = nn.Parameter(
            torch.randn([1, config.num_frames, 3, config.visual_prompt_size, config.max_img_size])
        )
        # 创建左侧填充参数
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
        # 创建右侧填充参数
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

    # 前向传播方法，接受像素值作为输入
    def forward(self, pixel_values):
        # 检查视觉提示应用方式是否在预定义的选项中
        if self.visual_prompter_apply not in ("add", "remove", "replace"):
            # 如果不在预定义选项中，则抛出数值错误
            raise ValueError(f"Invalid visual_prompter_apply value {self.visual_prompter_apply}")
        # 如果视觉提示应用方式是替换或移除
        if self.visual_prompter_apply in ("replace", "remove"):
            # 创建全为1的视觉提示掩码
            visual_prompt_mask = torch.ones(
                [self.max_img_size, self.max_img_size], dtype=pixel_values.dtype, device=pixel_values.device
            )
            # 将像素值与视觉提示掩码相乘，以移除或替换视觉提示
            pixel_values *= visual_prompt_mask
        # 如果视觉提示应用方式是替换或添加
        if self.visual_prompter_apply in ("replace", "add"):
            # 创建基础全零张量
            base = torch.zeros(1, self.num_frames, 3, self.base_size, self.base_size, device=pixel_values.device)
            # 拼接左侧填充、基础和右侧填充，形成完整的提示
            prompt = torch.cat([self.pad_left, base, self.pad_right], dim=4)
            # 拼接上方填充、提示和下方填充，形成完整的提示
            prompt = torch.cat([self.pad_up, prompt, self.pad_down], dim=3)
            # 将提示扩展到与像素值相同的大小，并添加到像素值上
            prompt = torch.cat(pixel_values.size(0) * [prompt])
            pixel_values += prompt.to(pixel_values.dtype)
        # 返回处理后的像素值
        return pixel_values
# 将不同的字符串映射到不同的 TvpPrompter 类
TVP_PROMPTER_CLASSES_MAPPING = {
    "framedownpad": TvpFrameDownPadPrompter,
    "framepad": TvpFramePadPrompter,
}

@add_start_docstrings(
    "The bare Tvp Model transformer outputting BaseModelOutputWithPooling object without any specific head on" " top.",
    TVP_START_DOCSTRING,
)
class TvpModel(TvpPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # 初始化 TvpModel，包括配置、视觉模型、嵌入层、编码器等
        self.config = config
        self.vision_model = TvpVisionModel(config)
        self.embeddings = TvpTextInputEmbeddings(config)
        self.visual_embeddings = TvpVisualInputEmbedding(config)
        self.encoder = TvpEncoder(config)
        self.pooler = TvpPooler(config)
        self.text_prompt = nn.Parameter(torch.randn([1, 10, config.hidden_size]))
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 如果 config.visual_prompter_type 不在预先定义的映射中，则抛出数值错误
        if config.visual_prompter_type not in TVP_PROMPTER_CLASSES_MAPPING:
            raise ValueError("`visual_prompter_type` must be in (framedownpad, framepad)")
        # 根据配置选择相应的视觉提示器
        self.visual_prompter = TVP_PROMPTER_CLASSES_MAPPING[config.visual_prompter_type](config)

        self.post_init()

    # 获取输入嵌入
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    # 设置输入嵌入
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    # 剪枝模型的头部
    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # TvpModel 的前向传播
    @add_start_docstrings_to_model_forward(TVP_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutputWithPooling, config_class=TvpConfig)
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
        # 前向传播逻辑
        pass

# TvpVideoGroundingHead 类定义
class TvpVideoGroundingHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layer_0 = nn.Linear(config.hidden_size, config.hidden_size * 2)
        self.layer_1 = nn.Linear(config.hidden_size * 2, 2)
        self.activation_0 = nn.ReLU()
        self.activation_1 = nn.Sigmoid()

    # TvpVideoGroundingHead 的前向传播
    def forward(self, pooler_output):
        logits = self.activation_0(self.layer_0(pooler_output))
        logits = self.activation_1(self.layer_1(logits))
        return logits

@add_start_docstrings(
    """
    Tvp Model with a video grounding head on top computing IoU, distance, and duration loss.
    """,
    TVP_START_DOCSTRING,
)
class TvpForVideoGrounding(TvpPreTrainedModel):
    # TvpForVideoGrounding 类定义
    pass
    # 此类继承自父类，并初始化配置和模型
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 保存配置信息
        self.config = config
        # 创建 TvpModel 实例
        self.model = TvpModel(config)
        # 创建 TvpVideoGroundingHead 实例
        self.video_grounding_head = TvpVideoGroundingHead(config)
        # 调用后初始化方法
        self.post_init()
    
    # 定义模型的前向传播过程
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
        # 如果 return_dict 为 None，则使用配置信息中的默认值
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        # 通过模型获得输出
        outputs = self.model(
            input_ids,
            pixel_values,
            attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 获取池化输出
        pooler_output = outputs[1]
        # 通过视频定位头获得输出
        logits = self.video_grounding_head(pooler_output)
        # 初始化损失为 None
        loss = None
        # 如果提供了 labels，则计算损失
        if labels is not None:
            # 创建损失计算器并移动到当前设备
            criterion = TvpLoss(["iou", "distance", "duration"])
            criterion.to(self.device)
            # 计算损失字典
            loss_dict = criterion(logits, labels)
            # 根据配置信息计算加权损失
            loss = (
                loss_dict["iou"]
                + self.config.distance_loss_weight * loss_dict["distance"]
                + self.config.duration_loss_weight * loss_dict["duration"]
            )
        # 如果不需要返回字典，则返回logits和其他输出
        if not return_dict:
            outputs = (logits,) + outputs[2:]
            if loss is not None:
                outputs = (loss,) + outputs
            return outputs
        # 否则返回 TvpVideoGroundingOutput
        return TvpVideoGroundingOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```