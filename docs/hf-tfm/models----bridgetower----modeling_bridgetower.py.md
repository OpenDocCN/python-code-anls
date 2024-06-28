# `.\models\bridgetower\modeling_bridgetower.py`

```
# coding=utf-8
# 版权声明及许可证信息

"""PyTorch BridgeTower Model"""

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

# 导入自定义模块
from ...activations import ACT2FN, QuickGELUActivation
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    MaskedLMOutput,
    ModelOutput,
    SequenceClassifierOutput,
)
from ...modeling_utils import PreTrainedModel, apply_chunking_to_forward
from ...pytorch_utils import find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging, replace_return_docstrings
from .configuration_bridgetower import BridgeTowerConfig, BridgeTowerTextConfig, BridgeTowerVisionConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 提供给文档的配置、检查点和分词器
_CONFIG_FOR_DOC = "BridgeTowerConfig"
_CHECKPOINT_FOR_DOC = "BridgeTower/bridgetower-base"
_TOKENIZER_FOR_DOC = "RobertaTokenizer"

# 预训练模型存档列表
BRIDGETOWER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "BridgeTower/bridgetower-base",
    "BridgeTower/bridgetower-base-itm-mlm",
    # 查看所有的 BridgeTower 模型：https://huggingface.co/BridgeTower
]

# BridgeTower 模型的起始文档字符串
BRIDGETOWER_START_DOCSTRING = r"""
    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ subclass. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`BridgeTowerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# BridgeTower 模型输入文档字符串
BRIDGETOWER_INPUTS_DOCSTRING = r"""
"""

@dataclass
class BridgeTowerModelOutput(ModelOutput):
    """
    Output type of [`BridgeTowerModel`].
    Represents the output of the BridgeTowerModel.
    Inherits from ModelOutput defined in the modeling_outputs module.
    """
    # 定义函数参数：文本特征的隐藏状态，形状为 `(batch_size, text_sequence_length, hidden_size)`
    text_features: torch.FloatTensor = None
    
    # 定义函数参数：图像特征的隐藏状态，形状为 `(batch_size, image_sequence_length, hidden_size)`
    image_features: torch.FloatTensor = None
    
    # 定义函数参数：池化器输出，形状为 `(batch_size, hidden_size x 2)`
    # 这是文本序列和图像序列最后一层隐藏状态的分类标记（第一个标记）的连接，经过用于辅助预训练任务的进一步处理层处理后的结果
    pooler_output: torch.FloatTensor = None
    
    # 定义函数参数（可选）：隐藏状态，是一个元组 `tuple(torch.FloatTensor)`
    # 当 `output_hidden_states=True` 或者 `config.output_hidden_states=True` 时返回
    # 包含模型每层输出的 `torch.FloatTensor`，形状为 `(batch_size, sequence_length, hidden_size)`，
    # 包括模型输出每一层的隐藏状态以及可选的初始嵌入输出
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    
    # 定义函数参数（可选）：注意力权重，是一个元组 `tuple(torch.FloatTensor)`
    # 当 `output_attentions=True` 或者 `config.output_attentions=True` 时返回
    # 包含每一层注意力权重的 `torch.FloatTensor`，形状为 `(batch_size, num_heads, sequence_length, sequence_length)`
    # 这些是经过注意力 softmax 后的注意力权重，用于计算自注意力头中的加权平均值
    attentions: Optional[Tuple[torch.FloatTensor]] = None
# 使用 dataclass 装饰器定义一个数据类，表示桥塔对比学习任务的模型输出
@dataclass
class BridgeTowerContrastiveOutput(ModelOutput):
    """
    Output type of ['BridgeTowerForContrastiveLearning']

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`:
            Image-text contrastive loss. 图像与文本的对比损失值（当 `return_loss` 为 `True` 时返回）。
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
            语言建模头部的预测分数（SoftMax 前每个词汇标记的分数）。
        text_embeds (`torch.FloatTensor)`, *optional*, returned when model is initialized with `with_projection=True`):
            The text embeddings obtained by applying the projection layer to the pooler_output.
            应用投影层到池化输出后得到的文本嵌入。
        image_embeds (`torch.FloatTensor)`, *optional*, returned when model is initialized with `with_projection=True`):
            The image embeddings obtained by applying the projection layer to the pooler_output.
            应用投影层到池化输出后得到的图像嵌入。
        cross_embeds  (`torch.FloatTensor)`, *optional*, returned when model is initialized with `with_projection=True`):
            The text-image cross-modal embeddings obtained by applying the projection layer to the pooler_output.
            应用投影层到池化输出后得到的文本-图像跨模态嵌入。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states of
            the model at the output of each layer plus the optional initial embedding outputs.
            如果模型有嵌入层，输出嵌入和每一层的输出形成的元组，形状为 `(batch_size, sequence_length, hidden_size)`。
            模型每层的隐藏状态及可选的初始嵌入输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            如果传递了 `output_attentions=True` 或 `config.output_attentions=True`，返回每层的注意力分布，
            形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。
    """

    loss: Optional[torch.FloatTensor] = None  # 图像与文本的对比损失值（可选）
    logits: torch.FloatTensor = None  # 语言建模头部的预测分数
    text_embeds: Optional[Tuple[torch.FloatTensor]] = None  # 文本嵌入（可选）
    image_embeds: Optional[Tuple[torch.FloatTensor]] = None  # 图像嵌入（可选）
    cross_embeds: Optional[Tuple[torch.FloatTensor]] = None  # 文本-图像跨模态嵌入（可选）
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None  # 模型每层的隐藏状态及可选的初始嵌入输出（可选）
    attentions: Optional[Tuple[torch.FloatTensor]] = None  # 每层的注意力分布（可选）

class BridgeTowerResidualAttention(nn.Module):
    # 初始化函数，接受配置对象 config
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()

        # 创建多头注意力机制对象，配置隐藏大小和头数
        self.attn = nn.MultiheadAttention(config.hidden_size, config.hidden_size // 64)
        
        # 创建第一个 LayerNorm 层，配置隐藏大小和层归一化的 epsilon 值
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # 创建包含线性层和激活函数的模块字典
        self.mlp = nn.ModuleDict(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(config.hidden_size, config.hidden_size * 4)),  # 输入到隐藏大小乘以4的线性层
                    ("gelu", QuickGELUActivation()),  # GELU 激活函数
                    ("c_proj", nn.Linear(config.hidden_size * 4, config.hidden_size)),  # 将隐藏大小乘以4的结果线性映射回隐藏大小
                ]
            )
        )
        
        # 创建第二个 LayerNorm 层，配置隐藏大小和层归一化的 epsilon 值
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # 初始化注意力掩码为 None
        self.attn_mask = None

    # 注意力计算函数，接受隐藏状态和注意力掩码
    def attention(self, hidden_state: torch.Tensor, attention_mask: torch.Tensor):
        # 如果注意力掩码不为 None，则将其转换为布尔类型，并置于与 hidden_state 相同的设备上
        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=torch.bool, device=hidden_state.device)
        
        # 如果 self.attn_mask 不为 None，则将其转换为 hidden_state 的数据类型，并置于与 hidden_state 相同的设备上
        self.attn_mask = (
            self.attn_mask.to(dtype=hidden_state.dtype, device=hidden_state.device)
            if self.attn_mask is not None
            else None
        )
        
        # 调用多头注意力机制，传入 hidden_state 作为查询、键和值，返回注意力计算结果
        return self.attn(
            hidden_state,
            hidden_state,
            hidden_state,
            need_weights=False,
            attn_mask=self.attn_mask,
            key_padding_mask=attention_mask,
        )[0]

    # 前向传播函数，接受隐藏状态和注意力掩码，默认注意力掩码为 None
    def forward(self, hidden_state: torch.Tensor, attention_mask: torch.Tensor = None):
        # 计算残差连接的隐藏状态
        residual_state = hidden_state + self.attention(self.ln_1(hidden_state), attention_mask)
        
        # 对残差状态进行 LayerNorm
        hidden_state = self.ln_2(residual_state)
        
        # 遍历 MLP 模块字典中的每个层，并对隐藏状态进行处理
        for _, layer in self.mlp.items():
            hidden_state = layer(hidden_state)
        
        # 最终的隐藏状态是残差状态和经过 MLP 处理后的状态的和
        hidden_state = residual_state + hidden_state
        
        # 返回最终的隐藏状态
        return hidden_state
# 定义视觉Transformer模型类BridgeTowerVisionTransformer，继承自nn.Module
class BridgeTowerVisionTransformer(nn.Module):
    def __init__(self, config: BridgeTowerVisionConfig):
        super().__init__()
        # 初始化模型配置
        self.config = config
        # 设定嵌入维度为隐藏大小
        self.embed_dim = config.hidden_size

        # 图像尺寸和补丁大小从配置中获取
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        # 类别嵌入为一个可学习的参数
        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

        # 补丁嵌入为一个2D卷积层，将输入通道数转换为隐藏大小，不使用偏置
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        # 计算图像中的补丁数量和位置嵌入的数量
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1

        # 位置嵌入为一个Embedding层，其索引从0到num_positions-1，维度为embed_dim
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)

        # 注册一个缓冲区，存储位置ID张量，形状为[1, num_positions]
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        # 获取批量大小
        batch_size = pixel_values.shape[0]

        # 目标数据类型为补丁嵌入的权重类型
        target_dtype = self.patch_embedding.weight.dtype

        # 对输入的像素值进行补丁嵌入，输出形状为[*, embed_dim, width//patch_size, grid//patch_size]
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))

        # 将补丁嵌入展平并转置以适应Transformer输入的形状，形状变为[*, num_patches, embed_dim]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        # 类别嵌入扩展为(batch_size, 1, embed_dim)
        class_embeds = self.class_embedding.expand(batch_size, 1, -1)

        # 将类别嵌入和补丁嵌入连接在一起，形状为[batch_size, num_patches+1, embed_dim]
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)

        # 加上位置嵌入，使用先前注册的位置ID张量，形状为[batch_size, num_patches+1, embed_dim]
        embeddings = embeddings + self.position_embedding(self.position_ids)

        # 返回嵌入张量
        return embeddings
    # 初始化函数，接受配置参数并初始化模型的各个组件
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()

        # 创建视觉嵌入层对象，并传入配置参数
        self.embeddings = BridgeTowerVisionEmbeddings(config)
        
        # 创建 LayerNorm 层，用于在 Transformer 前后对隐藏状态进行归一化
        self.ln_pre = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # 创建 Transformer 层对象，并传入配置参数
        self.transformer = BridgeTowerTransformer(config)
        
        # 创建另一个 LayerNorm 层，用于 Transformer 结束后对隐藏状态再次归一化
        self.ln_post = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # 根据配置参数决定是否共享 LayerNorm 层
        self.share_layernorm = config.share_layernorm
        
        # 如果不共享 LayerNorm 层，则创建独立的 LayerNorm 层列表，数量与 Transformer 层数相同
        if not config.share_layernorm:
            self.ln_separate = nn.ModuleList(
                [nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) for _ in range(config.num_hidden_layers)]
            )

    # 前向传播函数，接收像素值张量和注意力掩码作为输入，返回处理后的隐藏状态张量
    def forward(self, pixel_values: torch.Tensor, attention_mask):
        # 将像素值张量传入视觉嵌入层进行处理
        hidden_states = self.embeddings(pixel_values)
        
        # 对视觉嵌入后的隐藏状态进行 LayerNorm 归一化
        hidden_states = self.ln_pre(hidden_states)
        
        # 将维度顺序从 [batch_size, seq_length, hidden_size] 调整为 [seq_length, batch_size, hidden_size]
        hidden_states = hidden_states.permute(1, 0, 2)

        # 将调整后的隐藏状态输入 Transformer 进行处理
        hidden_states = self.transformer(hidden_states, attention_mask)
        
        # 将 Transformer 输出的隐藏状态堆叠起来，形状变为 [num_hidden_layers, batch_size, hidden_size, seq_length]
        hidden_states = torch.stack(hidden_states, dim=0)
        
        # 将堆叠后的隐藏状态的维度从 [num_hidden_layers, batch_size, hidden_size, seq_length] 调整为 [num_hidden_layers, batch_size, seq_length, hidden_size]
        hidden_states = hidden_states.permute(0, 2, 1, 3)
        
        # 如果配置中指定共享 LayerNorm 层，则对输出的隐藏状态进行最终的 LayerNorm 归一化
        if self.share_layernorm:
            hidden_states = self.ln_post(hidden_states)
        else:
            # 如果不共享 LayerNorm 层，则分别对每层的隐藏状态进行独立的 LayerNorm 归一化
            hidden_states_stack = []
            for hidden_states, ln in zip(hidden_states, self.ln_separate):
                hidden_states = ln(hidden_states)
                hidden_states_stack.append(hidden_states)
            
            # 将独立归一化后的隐藏状态堆叠起来，形状为 [num_hidden_layers, batch_size, seq_length, hidden_size]
            hidden_states = torch.stack(hidden_states_stack, dim=0)
        
        # 返回最终处理后的隐藏状态张量
        return hidden_states

    # 前向传播函数的预处理部分，只包括视觉嵌入和初始 LayerNorm 归一化，返回处理后的隐藏状态张量
    def forward_pre(self, pixel_values: torch.Tensor):
        hidden_states = self.embeddings(pixel_values)
        hidden_states = self.ln_pre(hidden_states)
        hidden_states = hidden_states.permute(1, 0, 2)
        return hidden_states

    # 前向传播函数的后处理部分，接收隐藏状态张量作为输入，对其进行 LayerNorm 归一化，并返回处理后的输出张量
    def forward_post(self, hidden_state: torch.Tensor):
        # 将输入的隐藏状态张量维度从 [batch_size, seq_length, hidden_size] 调整为 [seq_length, batch_size, hidden_size]
        visual_output_post = hidden_state.permute(1, 0, 2)
        
        # 对调整后的隐藏状态进行最终的 LayerNorm 归一化处理
        visual_output_post = self.ln_post(visual_output_post)
        
        # 返回最终处理后的输出张量
        return visual_output_post
# 定义 BridgeTowerLinkTower 类，继承自 nn.Module
class BridgeTowerLinkTower(nn.Module):
    # 初始化方法，接收一个 config 对象作为参数
    def __init__(self, config):
        super().__init__()
        # 设置 link_tower_type 属性为传入 config 对象的 link_tower_type
        self.link_tower_type = config.link_tower_type
        # 设置 hidden_size 属性为传入 config 对象的 hidden_size
        self.hidden_size = config.hidden_size
        # 如果 link_tower_type 在 ["add", "scaled_add", "interpolate"] 中
        if config.link_tower_type in ["add", "scaled_add", "interpolate"]:
            # 如果 link_tower_type 是 "scaled_add"
            if config.link_tower_type == "scaled_add":
                # 创建一个可训练参数 scaled_factor，初始值为 1.0
                self.scaled_factor = nn.Parameter(torch.tensor(1.0))
            # 如果 link_tower_type 是 "interpolate"
            elif config.link_tower_type == "interpolate":
                # 创建一个可训练参数 beta，初始值为 0.5
                self.beta = nn.Parameter(torch.tensor(0.5))
            # 创建一个 LayerNorm 层，用于对 hidden_size 维度进行归一化，epsilon 值由 config 提供
            self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)
        else:
            # 如果 link_tower_type 不在支持的类型中，则抛出未实现异常
            raise NotImplementedError(f"link_tower_type {config.link_tower_type} is not implemented")

    # 前向传播方法，接收 hidden_states, cross_modal_hidden_states 和 attention_mask 作为参数
    def forward(self, hidden_states, cross_modal_hidden_states, attention_mask):
        # 根据 link_tower_type 执行不同的链接操作
        if self.link_tower_type == "add":
            # 返回 LayerNorm 应用于 hidden_states 与 cross_modal_hidden_states 相加的结果
            return self.LayerNorm(hidden_states + cross_modal_hidden_states)
        elif self.link_tower_type == "scaled_add":
            # 返回 LayerNorm 应用于 hidden_states 乘以 scaled_factor 加上 cross_modal_hidden_states 的结果
            return self.LayerNorm(hidden_states * self.scaled_factor + cross_modal_hidden_states)
        elif self.link_tower_type == "interpolate":
            # 返回 LayerNorm 应用于 hidden_states 与 (1 - beta) 相乘加上 cross_modal_hidden_states 与 beta 相乘的结果
            return self.LayerNorm(hidden_states * (1 - self.beta) + cross_modal_hidden_states * self.beta)
        else:
            # 如果 link_tower_type 不在支持的类型中，则抛出未实现异常
            raise NotImplementedError(f"link_tower_type {self.link_tower_type} is not implemented")


# 从 transformers.models.bert.modeling_bert.BertSelfOutput 复制并修改为 BridgeTowerSelfOutput
# 定义 BridgeTowerSelfOutput 类，继承自 nn.Module
class BridgeTowerSelfOutput(nn.Module):
    # 初始化方法，接收一个 config 对象作为参数
    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层 dense，输入输出维度都是 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 创建一个 LayerNorm 层，用于对 config.hidden_size 维度进行归一化，epsilon 值由 config 提供
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个 Dropout 层，丢弃概率为 config.hidden_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播方法，接收 hidden_states 和 input_tensor 作为参数
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将 hidden_states 输入到全连接层 dense 中
        hidden_states = self.dense(hidden_states)
        # 对 hidden_states 进行 Dropout 处理
        hidden_states = self.dropout(hidden_states)
        # 返回 LayerNorm 应用于 hidden_states 加上 input_tensor 的结果
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# 从 transformers.models.bert.modeling_bert.BertIntermediate 复制并修改为 BridgeTowerIntermediate
# 定义 BridgeTowerIntermediate 类，继承自 nn.Module
class BridgeTowerIntermediate(nn.Module):
    # 初始化方法，接收一个 config 对象作为参数
    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层 dense，输入维度为 config.hidden_size，输出维度为 config.intermediate_size
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 如果 config.hidden_act 是字符串类型
        if isinstance(config.hidden_act, str):
            # 根据 config.hidden_act 的值选择相应的激活函数，并赋值给 intermediate_act_fn
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            # 否则直接使用 config.hidden_act 作为激活函数
            self.intermediate_act_fn = config.hidden_act

    # 前向传播方法，接收 hidden_states 作为参数
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 将 hidden_states 输入到全连接层 dense 中
        hidden_states = self.dense(hidden_states)
        # 将全连接层的输出应用 intermediate_act_fn 激活函数后返回
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# 从 transformers.models.bert.modeling_bert.BertOutput 复制并修改为 BridgeTowerOutput
# 定义 BridgeTowerOutput 类，继承自 nn.Module
class BridgeTowerOutput(nn.Module):
    # 初始化函数，用于初始化对象
    def __init__(self, config):
        # 调用父类（nn.Module）的初始化方法
        super().__init__()
        # 创建一个全连接层，输入尺寸为config.intermediate_size，输出尺寸为config.hidden_size
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建一个 LayerNorm 层，对输入进行归一化处理，归一化维度为config.hidden_size，eps为归一化过程中的小数值偏移量
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个 Dropout 层，以config.hidden_dropout_prob的概率随机将输入置零，用于防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数，定义了数据流向和处理逻辑
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将输入的 hidden_states 经过全连接层 dense，得到新的隐藏状态
        hidden_states = self.dense(hidden_states)
        # 对新的隐藏状态进行 Dropout 处理，以防止过拟合
        hidden_states = self.dropout(hidden_states)
        # 将经过 Dropout 处理后的隐藏状态与输入的 input_tensor 相加，并经过 LayerNorm 处理，得到最终的隐藏状态
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回最终的隐藏状态作为输出
        return hidden_states
# 从 transformers.models.bert.modeling_bert.BertPooler 复制代码，将 Bert 改为 BridgeTower
class BridgeTowerPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 定义一个全连接层，将输入维度为 config.hidden_size 的向量映射到相同维度
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 定义激活函数为 Tanh
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 取出每个样本的第一个 token 对应的隐藏状态作为池化输出
        first_token_tensor = hidden_states[:, 0]
        # 将池化输出输入到全连接层中进行线性变换
        pooled_output = self.dense(first_token_tensor)
        # 使用 Tanh 激活函数处理线性变换的结果
        pooled_output = self.activation(pooled_output)
        # 返回池化后的输出
        return pooled_output


# 从 transformers.models.roberta.modeling_roberta.RobertaSelfAttention 复制代码，将 Roberta 改为 BridgeTower
class BridgeTowerSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 检查 hidden_size 是否能被 num_attention_heads 整除，若不能则抛出 ValueError
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        # 设置注意力头数和每个头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 定义 Query、Key、Value 的线性变换层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # 定义 Dropout 层，用于在计算注意力分布时进行随机置零
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        # 设置位置嵌入类型，默认为 absolute
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )

        # 若位置嵌入类型为 relative_key 或 relative_key_query，则使用距离嵌入
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        # 判断是否为解码器
        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        # 调整张量形状，以便进行多头注意力计算
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
        # 从 transformers.models.bert.modeling_bert.BertAttention 复制代码，将 Bert 改为 BridgeTower
class BridgeTowerAttention(nn.Module):
    # 初始化函数，定义注意力模块的结构
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 创建自注意力层对象
        self.self = BridgeTowerSelfAttention(config, position_embedding_type=position_embedding_type)
        # 创建输出层对象
        self.output = BridgeTowerSelfOutput(config)
        # 初始化用于记录剪枝头部的集合
        self.pruned_heads = set()

    # 剪枝头部的方法
    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 找到可剪枝头部的索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 剪枝线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并记录剪枝头部
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    # 前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        # 调用自注意力层进行前向传播
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        # 将自注意力层的输出传递给输出层进行处理
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # 如果需要输出注意力信息，则添加到输出中
        return outputs


class BridgeTowerBertCrossLayer(nn.Module):
    # 初始化函数，定义BERT跨层连接模块的结构
    def __init__(self, config):
        super().__init__()
        # 设置前向传播中的分块大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 序列长度维度
        self.seq_len_dim = 1
        # 创建注意力对象
        self.attention = BridgeTowerAttention(config)
        # 是否为解码器
        self.is_decoder = config.is_decoder
        # 是否添加交叉注意力
        self.add_cross_attention = config.add_cross_attention
        # 创建交叉注意力对象
        self.crossattention = BridgeTowerAttention(config)
        # 创建中间层对象
        self.intermediate = BridgeTowerIntermediate(config)
        # 创建输出层对象
        self.output = BridgeTowerOutput(config)

    # 前向传播函数
    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        # 使用注意力层进行前向传播
        outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        # 如果需要添加交叉注意力，则调用交叉注意力层
        if self.add_cross_attention:
            cross_attention_outputs = self.crossattention(
                outputs[0],
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
            )
            # 使用中间层和输出层处理交叉注意力的输出
            intermediate_output = self.intermediate(cross_attention_outputs[0])
            layer_output = self.output(intermediate_output, outputs[0])
            outputs = (layer_output,) + cross_attention_outputs[1:] + outputs[1:]

        return outputs
        # 如果是 decoder，decoder uni-directional self-attention 缓存的键/值元组在位置 1 和 2
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=None,
            output_attentions=output_attentions,
            past_key_value=None,
        )
        # 获取自注意力机制的输出
        attention_output = self_attention_outputs[0]

        # 如果是 decoder，在最后一个输出中包含了自注意力机制的缓存元组
        # 如果需要输出注意力权重，则添加自注意力机制的输出
        outputs = self_attention_outputs[1:]

        # 执行跨注意力机制
        cross_attention_outputs = self.crossattention(
            attention_output,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
        )
        # 获取跨注意力机制的输出
        attention_output = cross_attention_outputs[0]
        # 如果需要输出注意力权重，则添加跨注意力机制的输出（排除最后一个元素）
        outputs = outputs + cross_attention_outputs[1:-1]

        # 对注意力输出应用分块处理
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        # 将处理后的输出与先前的输出合并
        outputs = (layer_output,) + outputs

        # 返回最终的输出结果
        return outputs

    # 定义前向传播的分块处理函数
    def feed_forward_chunk(self, attention_output):
        # 通过中间层处理注意力输出
        intermediate_output = self.intermediate(attention_output)
        # 使用输出层处理中间输出和注意力输出
        layer_output = self.output(intermediate_output, attention_output)
        # 返回处理后的层输出
        return layer_output
# 定义一个名为 BridgeTowerTextLayer 的神经网络模块，继承自 nn.Module
class BridgeTowerTextLayer(nn.Module):
    # 初始化函数，接受一个 config 参数
    def __init__(self, config):
        # 调用父类 nn.Module 的初始化函数
        super().__init__()
        # 设置类的属性 chunk_size_feed_forward，从 config 中获取前馈传递的块大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 设置类的属性 seq_len_dim，指定序列长度的维度为 1
        self.seq_len_dim = 1
        # 创建一个 BridgeTowerAttention 的实例，并赋给类的属性 attention
        self.attention = BridgeTowerAttention(config)
        # 从 config 中获取是否是解码器，并赋给类的属性 is_decoder
        self.is_decoder = config.is_decoder
        # 从 config 中获取是否添加交叉注意力，如果是，则创建一个新的 BridgeTowerAttention 实例赋给类的属性 crossattention
        if self.add_cross_attention:
            if not self.is_decoder:
                # 如果不是解码器但添加了交叉注意力，抛出 ValueError 异常
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            self.crossattention = BridgeTowerAttention(config, position_embedding_type="absolute")
        # 创建一个 BridgeTowerIntermediate 的实例，并赋给类的属性 intermediate
        self.intermediate = BridgeTowerIntermediate(config)
        # 创建一个 BridgeTowerOutput 的实例，并赋给类的属性 output
        self.output = BridgeTowerOutput(config)

    # 前向传播函数，接受多个输入参数
    def forward(
        self,
        hidden_states: torch.Tensor,  # 隐藏状态张量
        attention_mask: Optional[torch.FloatTensor] = None,  # 注意力掩码张量（可选）
        head_mask: Optional[torch.FloatTensor] = None,  # 头部掩码张量（可选）
        encoder_hidden_states: Optional[torch.FloatTensor] = None,  # 编码器隐藏状态张量（可选）
        encoder_attention_mask: Optional[torch.FloatTensor] = None,  # 编码器注意力掩码张量（可选）
        past_key_value: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,  # 过去键值元组的张量（可选）
        output_attentions: Optional[bool] = False,  # 输出注意力张量的标志（可选，默认为 False）
    ) -> Tuple[torch.Tensor]:
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # Perform self-attention using the cached key/values if available
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        # if decoder, the last output is tuple of self-attn cache
        if self.is_decoder:
            # Extract all outputs except the last one as intermediate outputs
            outputs = self_attention_outputs[1:-1]
            # Retrieve the present key/value tuple for self-attention
            present_key_value = self_attention_outputs[-1]
        else:
            # Include self-attentions in outputs if we output attention weights
            outputs = self_attention_outputs[1:]

        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # cross_attn cached key/values tuple is at positions 3,4 of past_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # Perform cross-attention between decoder's self-attention output and encoder's hidden states
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            # Include cross-attentions in outputs if we output attention weights
            outputs = outputs + cross_attention_outputs[1:-1]

            # Append cross-attn cache to present_key_value tuple
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        # Apply chunking mechanism for feed-forward layer processing
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # if decoder, return the attn key/values as the last output
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        # Pass attention output through intermediate layer
        intermediate_output = self.intermediate(attention_output)
        # Apply feed-forward layer to get final layer output
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
# 从transformers.models.roberta.modeling_roberta.RobertaEncoder复制过来，将Roberta替换为BridgeTowerText
class BridgeTowerTextEncoder(nn.Module):
    # 初始化函数，设置模型配置和层列表
    def __init__(self, config):
        super().__init__()
        # 保存配置信息
        self.config = config
        # 创建包含多个BridgeTowerTextLayer的模块列表，数量为config.num_hidden_layers
        self.layer = nn.ModuleList([BridgeTowerTextLayer(config) for _ in range(config.num_hidden_layers)])
        # 是否启用梯度检查点，默认为False
        self.gradient_checkpointing = False

    # 前向传播函数，接收多个输入参数并返回多个输出
    def forward(
        self,
        hidden_states: torch.Tensor,  # 输入的隐藏状态张量
        attention_mask: Optional[torch.FloatTensor] = None,  # 可选的注意力掩码张量
        head_mask: Optional[torch.FloatTensor] = None,  # 可选的头部掩码张量
        encoder_hidden_states: Optional[torch.FloatTensor] = None,  # 可选的编码器隐藏状态张量
        encoder_attention_mask: Optional[torch.FloatTensor] = None,  # 可选的编码器注意力掩码张量
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,  # 可选的过去的键值元组
        use_cache: Optional[bool] = None,  # 可选的使用缓存标志
        output_attentions: Optional[bool] = False,  # 是否输出注意力权重，默认为False
        output_hidden_states: Optional[bool] = False,  # 是否输出隐藏状态，默认为False
        return_dict: Optional[bool] = True,  # 是否返回字典格式的输出，默认为True
        ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        # 如果不需要输出隐藏状态，则初始化为空元组；否则为 None
        all_hidden_states = () if output_hidden_states else None
        # 如果不需要输出注意力权重，则初始化为空元组；否则为 None
        all_self_attentions = () if output_attentions else None
        # 如果不需要输出跨层注意力权重或配置不支持，则初始化为空元组；否则为 None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        # 如果开启了梯度检查点且处于训练模式下
        if self.gradient_checkpointing and self.training:
            # 如果设置了 use_cache=True，则警告并强制设置为 False
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # 如果不需要使用缓存，则初始化为空元组；否则为 None
        next_decoder_cache = () if use_cache else None
        # 遍历所有的 Transformer 层
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，则添加当前层的隐藏状态到 all_hidden_states
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 如果存在头部掩码，则使用对应的掩码；否则为 None
            layer_head_mask = head_mask[i] if head_mask is not None else None
            # 如果存在历史键值，则使用对应的键值；否则为 None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # 如果开启了梯度检查点且处于训练模式下
            if self.gradient_checkpointing and self.training:
                # 使用梯度检查点函数计算层的输出
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )
            else:
                # 否则直接调用层模块计算层的输出
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            # 更新隐藏状态为当前层输出的隐藏状态
            hidden_states = layer_outputs[0]
            # 如果使用缓存，则将当前层的输出添加到下一个解码器缓存中
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            # 如果需要输出注意力权重，则将当前层的自注意力权重添加到 all_self_attentions
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                # 如果配置支持添加跨层注意力权重，则将当前层的跨层注意力权重添加到 all_cross_attentions
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # 如果需要输出隐藏状态，则添加最后一个层的隐藏状态到 all_hidden_states
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不返回字典形式的结果，则返回元组
        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        # 否则返回包含详细输出的 BaseModelOutputWithPastAndCrossAttentions 对象
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
# 从 transformers.models.roberta.modeling_roberta.RobertaEmbeddings 复制过来的类 BridgeTowerTextEmbeddings
class BridgeTowerTextEmbeddings(nn.Module):
    """
    与 BertEmbeddings 相同，但稍作调整以适应位置嵌入的索引。
    """

    # 从 transformers.models.bert.modeling_bert.BertEmbeddings.__init__ 复制而来
    def __init__(self, config):
        super().__init__()
        # 词嵌入层，用于将输入的词汇 ID 转换为对应的隐藏表示
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 位置嵌入层，用于表示单词在句子中的位置信息
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # 标记类型嵌入层，用于区分句子中不同类型的标记（如句子 A 和句子 B）
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm 名称没有改为 snake-case，以保持与 TensorFlow 模型变量名称一致，以便能够加载任何 TensorFlow 检查点文件
        # LayerNorm 层，用于归一化隐藏表示，增加训练稳定性
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Dropout 层，用于随机丢弃部分神经元的输出，防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids （1，len position emb）在内存中是连续的，并在序列化时导出
        # 位置嵌入类型，默认为绝对位置编码
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # 注册位置 ID 张量，用于嵌入层的位置编码
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        # 注册标记类型 ID 张量，用于嵌入层的标记类型编码，默认全为零
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

        # End copy
        # 填充标记 ID，用于在输入序列中表示填充位置
        self.padding_idx = config.pad_token_id
        # 重新定义位置嵌入层，指定填充位置 ID
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
        ):
            # 如果未提供位置信息，但提供了输入标记信息，则根据输入标记信息创建位置信息，
            # 所有填充标记保持填充状态。
            if position_ids is None:
                if input_ids is not None:
                    position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
                else:
                    position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

            # 如果提供了输入标记信息，则获取其形状
            if input_ids is not None:
                input_shape = input_ids.size()
            else:
                input_shape = inputs_embeds.size()[:-1]

            seq_length = input_shape[1]

            # 将 token_type_ids 设置为构造函数中注册的缓冲区，通常为全零，
            # 当其自动生成时，注册的缓冲区有助于在跟踪模型时不传递 token_type_ids，解决问题 #5664
            if token_type_ids is None:
                if hasattr(self, "token_type_ids"):
                    buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                    buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                    token_type_ids = buffered_token_type_ids_expanded
                else:
                    token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

            # 如果未提供输入嵌入信息，则使用输入标记信息获取嵌入
            if inputs_embeds is None:
                inputs_embeds = self.word_embeddings(input_ids)
            token_type_embeddings = self.token_type_embeddings(token_type_ids)

            # 计算嵌入向量
            embeddings = inputs_embeds + token_type_embeddings

            # 如果位置嵌入类型为 "absolute"，则添加位置嵌入
            if self.position_embedding_type == "absolute":
                position_embeddings = self.position_embeddings(position_ids)
                embeddings += position_embeddings

            # 应用 LayerNorm
            embeddings = self.LayerNorm(embeddings)
            # 应用 dropout
            embeddings = self.dropout(embeddings)
            # 返回嵌入向量
            return embeddings

    def create_position_ids_from_inputs_embeds(self, inputs_embeds):
        """
        直接提供嵌入向量，无法推断填充标记，因此只生成顺序位置 id。

        Args:
            inputs_embeds: torch.Tensor

        Returns: torch.Tensor
        """
        input_shape = inputs_embeds.size()[:-1]
        sequence_length = input_shape[1]

        # 创建顺序位置 id
        position_ids = torch.arange(
            self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
        )
        return position_ids.unsqueeze(0).expand(input_shape)
# Copied from transformers.models.roberta.modeling_roberta.create_position_ids_from_input_ids
def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        input_ids: torch.Tensor, input tensor containing symbol indices
        padding_idx: int, padding symbol index
        past_key_values_length: int, optional, length of past key values

    Returns:
        torch.Tensor, tensor containing position indices
    """
    # Create a mask where non-padding elements are marked as 1, padding elements as 0
    mask = input_ids.ne(padding_idx).int()
    # Calculate cumulative sum of the mask along the second dimension, type-cast to mask's type, and adjust by past_key_values_length
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    # Add padding_idx to obtain final position indices tensor
    return incremental_indices.long() + padding_idx


class BridgeTowerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = BridgeTowerConfig  # Specify the configuration class for this model
    base_model_prefix = "bridgetower"  # Prefix used for the base model's attribute names
    supports_gradient_checkpointing = False  # Indicates if gradient checkpointing is supported
    _no_split_modules = ["BridgeTowerSelfAttention", "BridgeTowerResidualAttention"]  # List of modules not to split
    _skip_keys_device_placement = "past_key_values"  # Key for skipping device placement

    def _init_weights(self, module):
        """
        Initialize weights of the given module based on its type.

        Args:
            module: nn.Module, module to initialize weights for
        """
        if isinstance(module, BridgeTowerVisionModel):
            # Initialization for vision model's transformer components
            proj_std = (module.visual.transformer.hidden_size**-0.5) * (
                (2 * module.visual.transformer.num_hidden_layers) ** -0.5
            )
            attn_std = module.visual.transformer.hidden_size**-0.5
            fc_std = (2 * module.visual.transformer.hidden_size) ** -0.5
            # Initialize weights for attention, projection, and MLP layers in transformer blocks
            for block in module.visual.transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std * self.config.initializer_factor)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std * self.config.initializer_factor)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std * self.config.initializer_factor)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std * self.config.initializer_factor)

            # Initialize weights for class and position embeddings
            nn.init.normal_(module.visual.embeddings.class_embedding, std=attn_std * self.config.initializer_factor)
            nn.init.normal_(
                module.visual.embeddings.position_embedding.weight, std=attn_std * self.config.initializer_factor
            )
        elif isinstance(module, (nn.Linear, nn.Conv2d, nn.Embedding)):
            # Initialize weights for linear, convolutional, and embedding layers
            module.weight.data.normal_(mean=0.0, std=0.05 * self.config.initializer_factor)
        elif isinstance(module, nn.LayerNorm):
            # Initialize weights for LayerNorm modules
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        if isinstance(module, nn.Linear) and module.bias is not None:
            # Set biases to zero for Linear modules if they exist
            module.bias.data.zero_()


class BridgeTowerVisionModel(BridgeTowerPreTrainedModel):
    """
    Vision model class inheriting from BridgeTowerPreTrainedModel.

    Attributes:
        config_class: Class attribute specifying the configuration class for this model.
    """

    config_class = BridgeTowerVisionConfig

    def __init__(self, config):
        """
        Initialize the vision model with the given configuration.

        Args:
            config: BridgeTowerVisionConfig, configuration instance for the model
        """
        super().__init__(config)
        self.visual = BridgeTowerVisionTransformer(config)  # Initialize vision transformer
    # 定义属性访问器，返回 self.visual.embeddings.patch_embedding.weight 的数据类型
    @property
    def dtype(self):
        return self.visual.embeddings.patch_embedding.weight.dtype
    
    # 定义前向传播方法，接收图像数据和可选的图像掩码，使用 self.dtype 设置图像数据类型后调用 self.visual 进行处理
    def forward(self, image, image_mask=None):
        return self.visual(image.type(self.dtype), image_mask)
class BridgeTowerTextModel(BridgeTowerPreTrainedModel):
    """
    
    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in *Attention is
    all you need*_ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz
    Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the `is_decoder` argument of the configuration set
    to `True`. To be used in a Seq2Seq model, the model needs to initialized with both `is_decoder` argument and
    `add_cross_attention` set to `True`; an `encoder_hidden_states` is then expected as an input to the forward pass.

    .. _*Attention is all you need*: https://arxiv.org/abs/1706.03762

    """

    config_class = BridgeTowerTextConfig  # 设置配置类为 BridgeTowerTextConfig

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)  # 调用父类构造函数初始化模型
        self.config = config  # 设置模型的配置信息

        self.embeddings = BridgeTowerTextEmbeddings(config)  # 初始化文本嵌入层
        self.encoder = BridgeTowerTextEncoder(config)  # 初始化文本编码器

        self.pooler = BridgeTowerPooler(config) if add_pooling_layer else None  # 初始化池化层，如果 add_pooling_layer 为 True

        # Initialize weights and apply final processing
        self.post_init()  # 调用后处理函数，用于初始化权重和应用最终处理

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings  # 获取输入嵌入层的词嵌入向量

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value  # 设置输入嵌入层的词嵌入向量为指定值

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)  # 剪枝模型中的注意力头部，根据给定的 heads_to_prune 字典

    # Copied from transformers.models.roberta.modeling_roberta.RobertaModel.forward
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        This method defines the forward pass for the BridgeTowerTextModel.

        Args:
            input_ids (Optional[torch.Tensor]): Indices of input tokens in the vocabulary.
            attention_mask (Optional[torch.Tensor]): Mask to avoid performing attention on padding tokens.
            token_type_ids (Optional[torch.Tensor]): Segment token indices to differentiate sentences.
            position_ids (Optional[torch.Tensor]): Indices of positions of each input token in the sequence.
            head_mask (Optional[torch.Tensor]): Mask to nullify selected heads of the self-attention modules.
            inputs_embeds (Optional[torch.Tensor]): Optional tensor of embeddings to be used as input instead of
                                                    input_ids.
            encoder_hidden_states (Optional[torch.Tensor]): Sequence of hidden states of the encoder.
            encoder_attention_mask (Optional[torch.Tensor]): Mask to avoid performing attention on encoder padding tokens.
            past_key_values (Optional[List[torch.FloatTensor]]): Cached outputs of the model to speed up sequential
                                                                decoding.
            use_cache (Optional[bool]): Whether or not to use past_key_values to speed up decoding.
            output_attentions (Optional[bool]): Whether to return attentions weights.
            output_hidden_states (Optional[bool]): Whether to return hidden states.
            return_dict (Optional[bool]): Whether to return a dict instead of a tuple.

        Returns:
            Various outputs depending on the configuration (return_dict or not).
        """
        # Actual implementation of the forward pass is expected here in the derived model classes.
        pass
    # 初始化方法，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类初始化方法，传入配置对象
        super().__init__(config)
        # 将配置对象存储为实例变量
        self.config = config
        # 从配置对象中获取视觉配置和文本配置
        vision_config = config.vision_config
        text_config = config.text_config

        # 根据配置决定是否共享跨模态变换层
        if config.share_cross_modal_transformer_layers:
            # 如果共享，创建一个线性变换层，将文本隐藏状态映射到全局隐藏状态
            self.cross_modal_text_transform = nn.Linear(text_config.hidden_size, config.hidden_size)
            # 创建一个线性变换层，将视觉隐藏状态映射到全局隐藏状态
            self.cross_modal_image_transform = nn.Linear(vision_config.hidden_size, config.hidden_size)
        else:
            # 如果不共享，创建一个模块列表，每个元素是一个线性变换层，用于每个隐藏层
            self.cross_modal_text_transform = nn.ModuleList(
                [nn.Linear(text_config.hidden_size, config.hidden_size) for _ in range(config.num_hidden_layers)]
            )
            self.cross_modal_image_transform = nn.ModuleList(
                [nn.Linear(vision_config.hidden_size, config.hidden_size) for _ in range(config.num_hidden_layers)]
            )

        # 创建一个大小为2的嵌入层，用于区分不同类型的标记（如类标记等）
        self.token_type_embeddings = nn.Embedding(2, config.hidden_size)

        # 创建视觉模型对象，使用给定的视觉配置
        self.vision_model = BridgeTowerVisionModel(vision_config)

        # 创建文本模型对象，使用给定的文本配置
        self.text_model = BridgeTowerTextModel(text_config)

        # 如果视觉配置要求不共享层归一化，且从视觉编码器初始化层归一化
        if not vision_config.share_layernorm and config.init_layernorm_from_vision_encoder:
            # 将视觉模型的后层归一化权重和偏置复制给跨模态层归一化对象
            for ln in self.vision_model.visual.cross_modal_ln_separate:
                ln.weight.data = self.vision_model.visual.ln_post.weight.data
                ln.bias.data = self.vision_model.visual.ln_post.bias.data

        # 创建文本的跨模态层对象列表，每个对象使用文本配置创建
        self.cross_modal_image_layers = nn.ModuleList(
            [BridgeTowerBertCrossLayer(text_config) for _ in range(config.num_hidden_layers)]
        )
        # 创建视觉的跨模态层对象列表，每个对象使用视觉配置创建
        self.cross_modal_text_layers = nn.ModuleList(
            [BridgeTowerBertCrossLayer(text_config) for _ in range(config.num_hidden_layers)]
        )

        # 创建跨模态文本池化器对象，使用给定的配置
        self.cross_modal_text_pooler = BridgeTowerPooler(config)
        # 创建跨模态视觉池化器对象，使用给定的配置
        self.cross_modal_image_pooler = BridgeTowerPooler(config)

        # 创建跨模态文本层归一化对象，使用给定的隐藏大小和层归一化的 epsilon 值
        self.cross_modal_text_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建跨模态视觉层归一化对象，使用给定的隐藏大小和层归一化的 epsilon 值
        self.cross_modal_image_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 根据配置决定是否共享连接塔层
        if config.share_link_tower_layers:
            # 如果共享，创建一个连接塔对象，用于文本跨模态连接
            self.cross_modal_text_link_tower = BridgeTowerLinkTower(config)
            # 创建一个连接塔对象，用于视觉跨模态连接
            self.cross_modal_image_link_tower = BridgeTowerLinkTower(config)
        else:
            # 如果不共享，创建一个模块列表，每个元素是一个连接塔对象，用于每个隐藏层的连接
            self.cross_modal_text_link_tower = nn.ModuleList(
                [BridgeTowerLinkTower(config) for _ in range(config.num_hidden_layers - 1)]
            )
            self.cross_modal_image_link_tower = nn.ModuleList(
                [BridgeTowerLinkTower(config) for _ in range(config.num_hidden_layers - 1)]
            )

        # 调用初始化后的方法，用于额外的初始化步骤
        self.post_init()

    # 获取输入嵌入层的方法，委托给文本模型的获取输入嵌入层方法
    def get_input_embeddings(self):
        return self.text_model.get_input_embeddings()

    # 设置输入嵌入层的方法，委托给文本模型的设置输入嵌入层方法
    def set_input_embeddings(self, value):
        self.text_model.set_input_embeddings(value)

    # 添加模型正向传播的文档字符串注释，使用指定的输入文档字符串模板
    @add_start_docstrings_to_model_forward(BRIDGETOWER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BridgeTowerModelOutput, config_class=_CONFIG_FOR_DOC)
    # 使用装饰器，替换该方法的返回文档字符串，指定输出类型为BridgeTowerModelOutput，配置类为_CONFIG_FOR_DOC
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
        labels: Optional[torch.LongTensor] = None,
    # 此方法定义了模型的前向传播过程，接收多个可选的输入参数，并根据装饰器指定的返回类型和配置类处理返回文档
    def get_cls_features(self, text_features, image_features):
        # 通过文本特征传递到交叉模态文本池化器，获取文本的CLS特征
        cls_features_text = self.cross_modal_text_pooler(text_features)
        # 通过图像特征传递到交叉模态图像池化器，获取图像的CLS特征
        cls_features_image = self.cross_modal_image_pooler(image_features)
        # 将文本和图像的CLS特征在最后一个维度上连接起来
        return torch.cat([cls_features_text, cls_features_image], dim=-1)
# 从 transformers.models.vilt.modeling_vilt.ViltPredictionHeadTransform 复制并改名为 BridgeTowerPredictionHeadTransform
class BridgeTowerPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层，输入和输出维度都为 config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 根据 config.hidden_act 类型选择激活函数 ACT2FN 中的对应项或直接使用给定的激活函数
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        # LayerNorm 层，对隐藏状态的每个元素进行归一化，输入维度为 config.hidden_size
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        # 全连接层的前向传播
        hidden_states = self.dense(hidden_states)
        # 应用选定的激活函数
        hidden_states = self.transform_act_fn(hidden_states)
        # LayerNorm 的前向传播
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


# 包含 MLN（掩码语言建模）头部的模型
class BridgeTowerMLMHead(nn.Module):
    def __init__(self, config, weight=None):
        super().__init__()
        self.config = config
        # BridgeTowerPredictionHeadTransform 用于处理输入特征
        self.transform = BridgeTowerPredictionHeadTransform(config)
        # 全连接层用于预测文本的词汇量大小
        self.decoder = nn.Linear(config.hidden_size, config.text_config.vocab_size, bias=False)
        # 偏置项，用于加到 decoder 输出上
        self.bias = nn.Parameter(torch.zeros(config.text_config.vocab_size))
        if weight is not None:
            # 如果提供了预训练权重，则使用这些权重
            self.decoder.weight = weight

    def forward(self, x):
        # 使用头部变换处理输入数据
        mlm_score = self.transform(x)
        # 对处理后的数据进行解码和偏置处理
        mlm_score = self.decoder(mlm_score) + self.bias
        return mlm_score


# 包含 ITM（信息主题模型）头部的模型
class BridgeTowerITMHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # 全连接层，输入维度为 hidden_size，输出为 2（用于二分类任务）
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        # 全连接层的前向传播
        itm_score = self.fc(x)
        return itm_score


# BridgeTowerForMaskedLM 是 BridgeTowerPreTrainedModel 的一个子类，用于掩码语言建模
@add_start_docstrings(
    """
    使用语言建模头部的 BridgeTower 模型，用于预训练期间的任务。
    """,
    BRIDGETOWER_START_DOCSTRING,
)
class BridgeTowerForMaskedLM(BridgeTowerPreTrainedModel):
    _tied_weights_keys = ["mlm_score.decoder.weight"]

    def __init__(self, config):
        super().__init__(config)

        # 创建 BridgeTowerModel 对象
        self.bridgetower = BridgeTowerModel(config)
        # 创建 BridgeTowerMLMHead 对象
        self.mlm_score = BridgeTowerMLMHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    def get_output_embeddings(self):
        # 返回 MLN 头部的 decoder 层
        return self.mlm_score.decoder

    def set_output_embeddings(self, new_embeddings):
        # 设置 MLN 头部的 decoder 层的权重
        self.mlm_score.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(BRIDGETOWER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=MaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    # 定义模型的前向传播方法，接收多个可选的输入参数
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入的 token IDs，类型为长整型张量，可选
        attention_mask: Optional[torch.FloatTensor] = None,  # 注意力掩码张量，类型为浮点数张量，可选
        token_type_ids: Optional[torch.LongTensor] = None,  # token 类型 IDs，类型为长整型张量，可选
        pixel_values: Optional[torch.FloatTensor] = None,  # 图像像素数值张量，类型为浮点数张量，可选
        pixel_mask: Optional[torch.LongTensor] = None,  # 图像像素掩码张量，类型为长整型张量，可选
        head_mask: Optional[torch.FloatTensor] = None,  # 头部掩码张量，类型为浮点数张量，可选
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 嵌入的输入张量，类型为浮点数张量，可选
        image_embeds: Optional[torch.FloatTensor] = None,  # 图像嵌入张量，类型为浮点数张量，可选
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，类型为布尔值，可选
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，类型为布尔值，可选
        return_dict: Optional[bool] = None,  # 是否以字典形式返回结果，类型为布尔值，可选
        labels: Optional[torch.LongTensor] = None,  # 标签张量，类型为长整型张量，可选
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 如果 return_dict 不为 None，则使用它；否则使用配置中的 use_return_dict

        outputs = self.bridgetower(
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
        # 调用 BridgeTower 模型的前向传播，传入输入数据和相关参数

        mlm_logits = self.mlm_score(outputs.text_features if return_dict else outputs[0])
        # 使用模型输出的文本特征计算 MLM (Masked Language Modeling) 的预测 logits

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # 交叉熵损失函数，用于计算损失

            labels = labels.to(mlm_logits.device)
            # 将标签移动到与 mlm_logits 相同的设备上

            masked_lm_loss = loss_fct(mlm_logits.view(-1, self.config.text_config.vocab_size), labels.view(-1))
            # 计算 MLM 损失，将 logits 和标签视图展平为二维张量进行计算

        if not return_dict:
            output = tuple(mlm_logits)
            # 如果不返回字典，则输出 MLM 的 logits 元组

            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output
            # 如果有损失，则返回损失和输出；否则只返回输出

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=mlm_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        # 返回包含损失、logits、隐藏状态和注意力的 MaskedLMOutput 对象
# 使用自定义的文档字符串为类添加注释，描述这是一个 BridgeTower 模型的变体，用于图像到文本匹配任务，其在顶部包含一个分类器头部
# （即一个线性层，放置在最终隐藏状态的 [CLS] 标记之上）。

@add_start_docstrings(
    """
    BridgeTower Model transformer with a classifier head on top (a linear layer on top of the final hidden state of the
    [CLS] token) for image-to-text matching.
    """,
    BRIDGETOWER_START_DOCSTRING,
)
class BridgeTowerForImageAndTextRetrieval(BridgeTowerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 初始化 BridgeTower 模型
        self.bridgetower = BridgeTowerModel(config)

        # 初始化 BridgeTowerITMHead 作为图像到文本匹配任务的得分头部
        self.itm_score = BridgeTowerITMHead(config.hidden_size * 2)

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(BRIDGETOWER_INPUTS_DOCSTRING)
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
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 如果 return_dict 不为 None，则使用 return_dict；否则使用 self.config.use_return_dict

        outputs = self.bridgetower(
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
        # 调用 self.bridgetower 方法，传入各种输入参数，返回模型输出结果 outputs

        pooler_output = outputs.pooler_output if return_dict else outputs[2]
        # 如果 return_dict 为 True，则使用 outputs.pooler_output；否则使用 outputs 的第三个元素作为 pooler_output

        logits = self.itm_score(pooler_output)
        # 将 pooler_output 作为输入，计算模型的 logits

        itm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # 定义交叉熵损失函数对象

            labels = labels.to(logits.device)
            # 将 labels 移动到与 logits 相同的设备上

            itm_loss = loss_fct(logits, labels)
            # 计算模型预测的 logits 与实际 labels 之间的交叉熵损失

        if not return_dict:
            output = tuple(logits)
            # 如果 return_dict 为 False，则将 logits 转换为元组形式作为 output

            return ((itm_loss,) + output) if itm_loss is not None else output
            # 如果 itm_loss 不为 None，则返回包含 itm_loss 和 output 的元组；否则只返回 output

        return SequenceClassifierOutput(
            loss=itm_loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        # 如果 return_dict 为 True，则返回一个 SequenceClassifierOutput 对象，包含 itm_loss、logits、hidden_states 和 attentions
# 定义一个自定义的 PyTorch 模型类 BridgeTowerContrastiveHead，继承自 nn.Module
class BridgeTowerContrastiveHead(nn.Module):
    def __init__(self, hidden_size, embed_size):
        super().__init__()
        # 创建一个全连接层，将输入特征维度 hidden_size 转换为 embed_size
        self.fc = nn.Linear(hidden_size, embed_size)

    # 前向传播函数，接收输入 x，通过全连接层进行线性变换后返回
    def forward(self, x):
        x = self.fc(x)
        return x


# 使用装饰器 @add_start_docstrings 和指定的文档字符串，为 BridgeTowerForContrastiveLearning 类添加说明
@add_start_docstrings(
    """
    BridgeTower Model with a image-text contrastive head on top computing image-text contrastive loss.
    """,
    BRIDGETOWER_START_DOCSTRING,
)
# 定义一个自定义的 PyTorch 模型类 BridgeTowerForContrastiveLearning，继承自 BridgeTowerPreTrainedModel
class BridgeTowerForContrastiveLearning(BridgeTowerPreTrainedModel):
    # 初始化函数，接收一个配置参数 config
    def __init__(self, config):
        # 调用父类的初始化函数，传入配置参数 config
        super().__init__(config)

        # 创建 BridgeTowerModel 类的实例，并保存在 self.bridgetower 属性中
        self.bridgetower = BridgeTowerModel(config)

        # 创建用于文本和图像对比学习的头部模块实例
        # 使用 BridgeTowerContrastiveHead 类创建 itc_text_head 和 itc_image_head 实例，
        # 分别使用配置中的 hidden_size 和 contrastive_hidden_size 参数作为输入和输出维度
        self.itc_text_head = BridgeTowerContrastiveHead(config.hidden_size, config.contrastive_hidden_size)
        self.itc_image_head = BridgeTowerContrastiveHead(config.hidden_size, config.contrastive_hidden_size)

        # 创建用于跨模态对比学习的头部模块实例
        # 使用 BridgeTowerContrastiveHead 类创建 itc_cross_modal_head 实例，
        # 使用配置中的 hidden_size * 2 和 contrastive_hidden_size 参数作为输入和输出维度
        self.itc_cross_modal_head = BridgeTowerContrastiveHead(config.hidden_size * 2, config.contrastive_hidden_size)

        # 创建一个可学习的标量参数 logit_scale，初始化值来自于配置参数 self.config.logit_scale_init_value
        self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))

        # 调用模型初始化函数
        # 在此处执行额外的初始化任务，例如权重初始化和后处理步骤
        self.post_init()

    # 前向传播函数，接收多个输入参数，根据模型需要进行计算并返回结果
    @add_start_docstrings_to_model_forward(BRIDGETOWER_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BridgeTowerContrastiveOutput, config_class=_CONFIG_FOR_DOC)
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
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True,
        return_dict: Optional[bool] = None,
        return_loss: Optional[bool] = None,
```