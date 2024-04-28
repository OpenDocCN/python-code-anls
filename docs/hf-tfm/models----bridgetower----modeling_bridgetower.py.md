# `.\transformers\models\bridgetower\modeling_bridgetower.py`

```
# coding=utf-8
# 版权声明和许可证信息，指明此代码的版权和许可证信息
# Copyright 2023 The Intel Labs Team Authors, The Microsoft Research Team Authors and HuggingFace Inc. team. All rights reserved.
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
"""PyTorch BridgeTower Model"""

import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

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
# 导入日志模块，用于记录日志信息
logger = logging.get_logger(__name__)

# 用于文档的配置、检查点、标记器
_CONFIG_FOR_DOC = "BridgeTowerConfig"
_CHECKPOINT_FOR_DOC = "BridgeTower/bridgetower-base"
_TOKENIZER_FOR_DOC = "RobertaTokenizer"

# 预训练模型归档列表
BRIDGETOWER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "BridgeTower/bridgetower-base",
    "BridgeTower/bridgetower-base-itm-mlm",
    # 查看所有 BridgeTower 模型的列表链接
    # 在 https://huggingface.co/BridgeTower
]

# BridgeTower 模型文档起始说明
BRIDGETOWER_START_DOCSTRING = r"""
    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ subclass. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`BridgeTowerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# BridgeTower 模型输入说明
BRIDGETOWER_INPUTS_DOCSTRING = r"""
"""


@dataclass
class BridgeTowerModelOutput(ModelOutput):
    """
    Output type of [`BridgeTowerModel`].
    # BridgeTowerModel 的输出类型
```  
    Args:
        text_features (`torch.FloatTensor` of shape `(batch_size, text_sequence_length, hidden_size)`):
            模型最后一层文本输出的隐藏状态序列。
        image_features (`torch.FloatTensor` of shape `(batch_size, image_sequence_length, hidden_size)`):
            模型最后一层图像输出的隐藏状态序列。
        pooler_output (`torch.FloatTensor` of shape `(batch_size, hidden_size x 2)`):
            文本和图像序列的最后一层隐藏状态的池化输出，分别是文本序列的第一个标记（分类标记）和图像序列的第一个标记，经过用于辅助预训练任务的层的进一步处理后进行连接。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            `torch.FloatTensor` 元组（如果模型具有嵌入层，则有一个用于嵌入输出的张量，加上每一层的输出）的形状为 `(batch_size, sequence_length, hidden_size)`。模型每一层的隐藏状态以及可选的初始嵌入输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            `torch.FloatTensor` 元组（每个层一个）的形状为 `(batch_size, num_heads, sequence_length, sequence_length)`。

            注意力 softmax 之后的注意力权重，用于计算自注意力头中的加权平均值。
    """

    text_features: torch.FloatTensor = None
    image_features: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
# 定义一个数据类，用于存储桥塔对比学习的输出结果
@dataclass
class BridgeTowerContrastiveOutput(ModelOutput):
    """
    Output type of ['BridgeTowerForContrastiveLearning']

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `return_loss` is `True`:
            图像文本对比损失。
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            语言建模头的预测分数（SoftMax之前的每个词汇标记的分数）。
        text_embeds (`torch.FloatTensor)`, *optional*, returned when model is initialized with `with_projection=True`):
            通过将投影层应用于池化器输出获得的文本嵌入。
        image_embeds (`torch.FloatTensor)`, *optional*, returned when model is initialized with `with_projection=True`):
            通过将投影层应用于池化器输出获得的图像嵌入。
        cross_embeds  (`torch.FloatTensor)`, *optional*, returned when model is initialized with `with_projection=True`):
            通过将投影层应用于池化器输出获得的文本-图像跨模态嵌入。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            形状为`(batch_size, sequence_length, hidden_size)`的`torch.FloatTensor`元组（一个用于嵌入层的输出，如果模型具有嵌入层，+每个层的输出的一个）。
            每层模型的隐藏状态加上可选的初始嵌入输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            形状为`(batch_size, num_heads, sequence_length, sequence_length)`的`torch.FloatTensor`元组（每个层的一个）。
    """

    # 图像文本对比损失
    loss: Optional[torch.FloatTensor] = None
    # 语言建模头的预测分数
    logits: torch.FloatTensor = None
    # 文本嵌入
    text_embeds: Optional[Tuple[torch.FloatTensor]] = None
    # 图像嵌入
    image_embeds: Optional[Tuple[torch.FloatTensor]] = None
    # 文本-图像跨模态嵌入
    cross_embeds: Optional[Tuple[torch.FloatTensor]] = None
    # 每层模型的隐藏状态
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    # 注意力
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class BridgeTowerResidualAttention(nn.Module):
    # 初始化函数，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()

        # 初始化多头注意力层，设置输入和输出维度
        self.attn = nn.MultiheadAttention(config.hidden_size, config.hidden_size // 64)
        # 初始化第一个 LayerNorm 层，对输入进行归一化
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化 MLP 模块，包含线性变换和激活函数
        self.mlp = nn.ModuleDict(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(config.hidden_size, config.hidden_size * 4)),  # 线性变换
                    ("gelu", QuickGELUActivation()),  # GELU 激活函数
                    ("c_proj", nn.Linear(config.hidden_size * 4, config.hidden_size)),  # 线性变换
                ]
            )
        )
        # 初始化第二个 LayerNorm 层，对输入进行归一化
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化注意力掩码为空
        self.attn_mask = None

    # 定义注意力函数，接受隐藏状态和注意力掩码作为输入
    def attention(self, hidden_state: torch.Tensor, attention_mask: torch.Tensor):
        # 如果存在注意力掩码，则将其转换为布尔型并移到相同设备上
        if attention_mask is not None:
            attention_mask = attention_mask.to(dtype=torch.bool, device=hidden_state.device)
        # 如果存在注意力掩码，则将其转换为与隐藏状态相同的数据类型，并移到相同设备上；否则设为 None
        self.attn_mask = (
            self.attn_mask.to(dtype=hidden_state.dtype, device=hidden_state.device)
            if self.attn_mask is not None
            else None
        )
        # 调用多头注意力层，传入隐藏状态、注意力掩码和填充掩码，返回注意力值
        return self.attn(
            hidden_state,
            hidden_state,
            hidden_state,
            need_weights=False,
            attn_mask=self.attn_mask,
            key_padding_mask=attention_mask,
        )[0]

    # 前向传播函数，接受隐藏状态和注意力掩码作为输入
    def forward(self, hidden_state: torch.Tensor, attention_mask: torch.Tensor = None):
        # 计算残差连接
        residual_state = hidden_state + self.attention(self.ln_1(hidden_state), attention_mask)
        # 进行第二个 LayerNorm 归一化
        hidden_state = self.ln_2(residual_state)
        # 遍历 MLP 模块的每一层，依次进行前向传播
        for _, layer in self.mlp.items():
            hidden_state = layer(hidden_state)
        # 计算最终的输出，加上残差连接
        hidden_state = residual_state + hidden_state
        # 返回最终的隐藏状态
        return hidden_state
# 定义一个名为 BridgeTowerTransformer 的类，继承自 nn.Module
class BridgeTowerTransformer(nn.Module):
    # 初始化方法，接受一个 config 参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 从 config 中获取 hidden_size 和 num_hidden_layers
        self.hidden_size = config.hidden_size
        self.num_hidden_layers = config.num_hidden_layers
        # 根据 config 中的 remove_last_layer 决定是否减去最后一层
        if config.remove_last_layer:
            self.resblocks = nn.ModuleList(
                [BridgeTowerResidualAttention(config) for _ in range(self.num_hidden_layers - 1)]
            )
        else:
            self.resblocks = nn.ModuleList(
                [BridgeTowerResidualAttention(config) for _ in range(self.num_hidden_layers)]
            )
        # 从 config 中获取 stop_gradient
        self.stop_gradient = config.stop_gradient

    # 前向传播方法，接受 hidden_state 和 attention_mask 两个参数
    def forward(self, hidden_state: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        # 初始化一个空列表用于存储每个 block 的 hidden_state
        hidden_states = []
        # 遍历 resblocks
        for block in self.resblocks:
            # 调用 block 的前向传播方法
            hidden_state = block(hidden_state, attention_mask)
            # 根据 stop_gradient 决定是否对 hidden_state 进行 detach 操作
            if self.stop_gradient:
                hidden_states.append(hidden_state.detach())
            else:
                hidden_states.append(hidden_state)
        # 返回所有 hidden_state
        return hidden_states


# 定义一个名为 BridgeTowerVisionEmbeddings 的类，继承自 nn.Module
class BridgeTowerVisionEmbeddings(nn.Module):
    # 初始化方法，接受一个 config 参数
    def __init__(self, config: BridgeTowerVisionConfig):
        # 调用父类的初始化方法
        super().__init__()
        # 从 config 中获取相关参数
        self.config = config
        self.embed_dim = config.hidden_size
        self.image_size = config.image_size
        self.patch_size = config.patch_size

        # 初始化 class_embedding 为一个随机张量
        self.class_embedding = nn.Parameter(torch.randn(self.embed_dim))

        # 初始化 patch_embedding 为一个卷积层
        self.patch_embedding = nn.Conv2d(
            in_channels=config.num_channels,
            out_channels=self.embed_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            bias=False,
        )

        # 计算 num_patches 和 num_positions
        self.num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = self.num_patches + 1
        # 初始化 position_embedding 为一个 Embedding 层
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        # 注册一个名为 position_ids 的缓冲张量
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)

    # 前向传播方法，接受 pixel_values 参数
    def forward(self, pixel_values: torch.FloatTensor) -> torch.Tensor:
        # 获取 batch_size 和目标数据类型
        batch_size = pixel_values.shape[0]
        target_dtype = self.patch_embedding.weight.dtype
        # 对 pixel_values 进行 patch_embedding 操作
        patch_embeds = self.patch_embedding(pixel_values.to(dtype=target_dtype))  # shape = [*, width, grid, grid]
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        # 初始化 class_embeds 为 class_embedding 的扩展
        class_embeds = self.class_embedding.expand(batch_size, 1, -1)
        # 拼接 class_embeds 和 patch_embeds
        embeddings = torch.cat([class_embeds, patch_embeds], dim=1)
        # 加上 position_embedding
        embeddings = embeddings + self.position_embedding(self.position_ids)
        # 返回 embeddings
        return embeddings


# 定义一个名为 BridgeTowerVisionTransformer 的类，继承自 nn.Module
class BridgeTowerVisionTransformer(nn.Module):
    # 初始化函数，接受配置参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()

        # 创建嵌入层对象
        self.embeddings = BridgeTowerVisionEmbeddings(config)
        # 创建预层归一化层对象
        self.ln_pre = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建Transformer对象
        self.transformer = BridgeTowerTransformer(config)
        # 创建后层归一化层对象
        self.ln_post = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 是否共享归一化层
        self.share_layernorm = config.share_layernorm
        # 如果不共享归一化层，则创建独立的归一化层列表
        if not config.share_layernorm:
            self.ln_separate = nn.ModuleList(
                [nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps) for _ in range(config.num_hidden_layers)]
            )

    # 前向传播函数，接受像素值和注意力掩码
    def forward(self, pixel_values: torch.Tensor, attention_mask):
        # 获取嵌入层的隐藏状态
        hidden_states = self.embeddings(pixel_values)
        # 对隐藏状态进行预层归一化
        hidden_states = self.ln_pre(hidden_states)
        # 调整隐藏状态的维度
        hidden_states = hidden_states.permute(1, 0, 2)

        # 使用Transformer处理隐藏状态
        hidden_states = self.transformer(hidden_states, attention_mask)
        # 将隐藏状态堆叠起来
        hidden_states = torch.stack(hidden_states, dim=0)
        # 调整隐藏状态的维度
        hidden_states = hidden_states.permute(0, 2, 1, 3)

        # 如果共享归一化层，则对隐藏状态进行后层归一化
        if self.share_layernorm:
            hidden_states = self.ln_post(hidden_states)
        # 如果不共享归一化层，则分别对每个隐藏状态进行后层归一化
        else:
            hidden_states_stack = []
            for hidden_states, ln in zip(hidden_states, self.ln_separate):
                hidden_states = ln(hidden_states)
                hidden_states_stack.append(hidden_states)
            # 将处理后的隐藏状态堆叠起来
            hidden_states = torch.stack(hidden_states_stack, dim=0)
        return hidden_states

    # 预处理函数，接受像素值
    def forward_pre(self, pixel_values: torch.Tensor):
        # 获取嵌入层的隐藏状态
        hidden_states = self.embeddings(pixel_values)
        # 对隐藏状态进行预层归一化
        hidden_states = self.ln_pre(hidden_states)
        # 调整隐藏状态的维度
        hidden_states = hidden_states.permute(1, 0, 2)
        return hidden_states

    # 后处理函数，接受隐藏状态
    def forward_post(self, hidden_state: torch.Tensor):
        # 调整隐藏状态的维度
        visual_output_post = hidden_state.permute(1, 0, 2)
        # 对隐藏状态进行后层归一化
        visual_output_post = self.ln_post(visual_output_post)
        return visual_output_post
class BridgeTowerLinkTower(nn.Module):
    # BridgeTowerLinkTower 类的初始化方法
    def __init__(self, config):
        super().__init__()
        # 设置链接塔类型
        self.link_tower_type = config.link_tower_type
        # 设置隐藏层大小
        self.hidden_size = config.hidden_size
        # 如果链接塔类型是 ["add", "scaled_add", "interpolate"] 中的一种
        if config.link_tower_type in ["add", "scaled_add", "interpolate"]:
            # 如果链接塔类型是 "scaled_add"
            if config.link_tower_type == "scaled_add":
                # 创建一个可学习的缩放因子参数
                self.scaled_factor = nn.Parameter(torch.tensor(1.0))
            # 如果链接塔类型是 "interpolate"
            elif config.link_tower_type == "interpolate":
                # 创建一个可学习的插值参数
                self.beta = nn.Parameter(torch.tensor(0.5))
            # 创建 LayerNorm 层
            self.LayerNorm = nn.LayerNorm(self.hidden_size, eps=config.layer_norm_eps)
        # 如果链接塔类型不在支持的类型中
        else:
            # 抛出未实现的错误
            raise NotImplementedError(f"link_tower_type {config.link_tower_type} is not implemented")

    # BridgeTowerLinkTower 类的前向传播方法
    def forward(self, hidden_states, cross_modal_hidden_states, attention_mask):
        # 如果链接塔类型是 "add"
        if self.link_tower_type == "add":
            # 返回 LayerNorm 层后的加法操作结果
            return self.LayerNorm(hidden_states + cross_modal_hidden_states)
        # 如果链接塔类型是 "scaled_add"
        elif self.link_tower_type == "scaled_add":
            # 返回 LayerNorm 层后的按比例缩放加法操作结果
            return self.LayerNorm(hidden_states * self.scaled_factor + cross_modal_hidden_states)
        # 如果链接塔类型是 "interpolate"
        elif self.link_tower_type == "interpolate":
            # 返回 LayerNorm 层后的插值操作结果
            return self.LayerNorm(hidden_states * (1 - self.beta) + cross_modal_hidden_states * self.beta)
        # 如果链接塔类型不在支持的类型中
        else:
            # 抛出未实现的错误
            raise NotImplementedError(f"link_tower_type {self.link_tower_type} is not implemented")


# 从 transformers.models.bert.modeling_bert.BertSelfOutput 复制并修改为 BridgeTowerSelfOutput
class BridgeTowerSelfOutput(nn.Module):
    # BridgeTowerSelfOutput 类的初始化方法
    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 创建 LayerNorm 层
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建 Dropout 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # BridgeTowerSelfOutput 类的前向传播方法
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 全连接层计算
        hidden_states = self.dense(hidden_states)
        # Dropout 层应用
        hidden_states = self.dropout(hidden_states)
        # 返回 LayerNorm 层后的加法操作结果
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# 从 transformers.models.bert.modeling_bert.BertIntermediate 复制并修改为 BridgeTowerIntermediate
class BridgeTowerIntermediate(nn.Module):
    # BridgeTowerIntermediate 类的初始化方法
    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 如果隐藏激活函数是字符串类型
        if isinstance(config.hidden_act, str):
            # 获取相应的激活函数
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        # 如果隐藏激活函数是函数类型
        else:
            # 获取相应的激活函数
            self.intermediate_act_fn = config.hidden_act

    # BridgeTowerIntermediate 类的前向传播方法
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 全连接层计算
        hidden_states = self.dense(hidden_states)
        # 激活函数应用
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# 从 transformers.models.bert.modeling_bert.BertOutput 复制并修改为 BridgeTowerOutput
class BridgeTowerOutput(nn.Module):
``` 
    # 初始化函数，用于初始化对象
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 创建一个全连接层，输入尺寸为中间尺寸，输出尺寸为隐藏尺寸
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建一个 LayerNormalization 层，输入尺寸为隐藏尺寸，epsilon 参数为配置中的 LayerNormalization epsilon 值
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个 Dropout 层，以配置中的隐藏层 Dropout 概率为参数
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播函数，接收隐藏状态和输入张量，返回隐藏状态张量
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 将隐藏状态张量传递给全连接层，进行线性变换
        hidden_states = self.dense(hidden_states)
        # 对线性变换后的隐藏状态张量进行 Dropout 操作，以减少过拟合
        hidden_states = self.dropout(hidden_states)
        # 将输入张量和经过 Dropout 后的隐藏状态张量相加，并传递给 LayerNormalization 层
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        # 返回经过 LayerNormalization 处理后的隐藏状态张量
        return hidden_states
# 从transformers.models.bert.modeling_bert.BertPooler复制而来，将Bert改为BridgeTower
class BridgeTowerPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化一个全连接层，将输入特征维度转换为隐藏层维度
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # Tanh激活函数
        self.activation = nn.Tanh()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 我们通过简单地获取第一个标记对应的隐藏状态来"池化"模型。
        first_token_tensor = hidden_states[:, 0]
        # 全连接层的计算，将第一个标记对应的隐藏状态转换为相同维度的输出
        pooled_output = self.dense(first_token_tensor)
        # 使用Tanh激活函数进行激活
        pooled_output = self.activation(pooled_output)
        return pooled_output


# 从transformers.models.roberta.modeling_roberta.RobertaSelfAttention复制而来，将Roberta改为BridgeTower
class BridgeTowerSelfAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 如果隐藏层维度不能整除注意力头数，且配置中不存在embedding_size，则报错
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        # 计算每个注意力头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 初始化查询、键、值的全连接层
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # Dropout层，用于注意力概率的dropout
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        # 位置嵌入类型，默认为绝对位置嵌入
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )
        # 如果位置嵌入类型为"relative_key"或"relative_key_query"，则初始化距离嵌入
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        # 是否为解码器
        self.is_decoder = config.is_decoder

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        # 转换张量形状以适应注意力分数计算
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
    ):
# 定义一个桥塔注意力模块的类
class BridgeTowerAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        # 继承父类构造函数
        super().__init__()
        # 创建自注意力层对象
        self.self = BridgeTowerSelfAttention(config, position_embedding_type=position_embedding_type)
        # 创建输出层对象
        self.output = BridgeTowerSelfOutput(config)
        # 存储被剪枝的注意力头的索引
        self.pruned_heads = set()

    # 剪枝函数，根据给定的注意力头集合进行剪枝
    def prune_heads(self, heads):
        # 若给定的注意力头集合为空，直接返回
        if len(heads) == 0:
            return
        # 调用剪枝工具函数找到可剪枝的头和其索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 剪枝线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并记录被剪枝的头
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    # 前向传播函数，接收输入并进行前向传播计算
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
        # 调用自注意力层进行前向传播计算
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        # 使用输出层处理自注意力层的输出结果
        attention_output = self.output(self_outputs[0], hidden_states)
        # 如果需要输出注意力矩阵，则将其添加到输出中
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        # 返回输出结果
        return outputs


# 定义一个桥塔BERT交叉层模块的类
class BridgeTowerBertCrossLayer(nn.Module):
    def __init__(self, config):
        # 继承父类构造函数
        super().__init__()
        # 设置前馈传递的块大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 序列长度维度
        self.seq_len_dim = 1
        # 创建自注意力层对象
        self.attention = BridgeTowerAttention(config)
        # 是否为解码器
        self.is_decoder = config.is_decoder
        # 是否添加交叉注意力
        self.add_cross_attention = config.add_cross_attention
        # 如果添加交叉注意力，创建交叉注意力层对象
        if self.add_cross_attention:
            self.crossattention = BridgeTowerAttention(config)
        # 创建中间层对象
        self.intermediate = BridgeTowerIntermediate(config)
        # 创建输出层对象
        self.output = BridgeTowerOutput(config)

    # 前向传播函数，接收输入并进行前向传播计算
    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    # 如果是解码器，最后一个输出是自注意力缓存的键/值元组在位置1,2
    self_attention_outputs = self.attention(
        hidden_states,
        attention_mask=attention_mask,
        head_mask=None,
        output_attentions=output_attentions,
        past_key_value=None,
    )
    attention_output = self_attention_outputs[0]

    # 如果是解码器，最后一个输出是自注意力缓存的元组
    # 如果输出注意力权重，则添加自注意力
    outputs = self_attention_outputs[1:]

    # 进行交叉注意力计算
    cross_attention_outputs = self.crossattention(
        attention_output,
        attention_mask=attention_mask,
        head_mask=head_mask,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        past_key_value=past_key_value,
        output_attentions=output_attentions,
    )
    attention_output = cross_attention_outputs[0]
    # 如果输出注意力权重，则添加交叉注意力
    outputs = outputs + cross_attention_outputs[1:-1]

    # 应用分块机制来前向传播
    layer_output = apply_chunking_to_forward(
        self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
    )
    outputs = (layer_output,) + outputs

    return outputs

# 定义前馈网络的分块函数
def feed_forward_chunk(self, attention_output):
    intermediate_output = self.intermediate(attention_output)
    layer_output = self.output(intermediate_output, attention_output)
    return layer_output
# 定义一个桥塔文本层的神经网络模块
class BridgeTowerTextLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化块大小 FeedForward
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 序列长度维度
        self.seq_len_dim = 1
        # 初始化注意力机制
        self.attention = BridgeTowerAttention(config)
        # 是否为解码器
        self.is_decoder = config.is_decoder
        # 是否添加交叉注意力
        self.add_cross_attention = config.add_cross_attention
        # 如果添加交叉注意力
        if self.add_cross_attention:
            # 如果不是解码器，则抛出错误
            if not self.is_decoder:
                raise ValueError(f"{self} should be used as a decoder model if cross attention is added")
            # 初始化交叉注意力
            self.crossattention = BridgeTowerAttention(config, position_embedding_type="absolute")
        # 初始化中间层
        self.intermediate = BridgeTowerIntermediate(config)
        # 初始化输出层
        self.output = BridgeTowerOutput(config)

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
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        # 如果过去的键/值不为空，则将decoder单向自注意力的缓存键/值元组放在位置1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # 使用self.attention进行自注意力计算
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        # 获取自注意力输出
        attention_output = self_attention_outputs[0]

        # 如果是解码器，最后一个输出是自注意力缓存的元组
        if self.is_decoder:
            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]
        else:
            outputs = self_attention_outputs[1:]  # 如果输出注意力权重，则添加自注意力
          
        cross_attn_present_key_value = None
        if self.is_decoder and encoder_hidden_states is not None:
            if not hasattr(self, "crossattention"):
                raise ValueError(
                    f"If `encoder_hidden_states` are passed, {self} has to be instantiated with cross-attention layers"
                    " by setting `config.add_cross_attention=True`"
                )

            # 交叉注意力缓存的键/值元组在过去键/值元组的位置3,4
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            # 使用crossattention进行交叉注意力计算
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                cross_attn_past_key_value,
                output_attentions,
            )
            # 获取交叉注意力输出
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:-1]  # 如果输出注意力权重，则添加交叉注意力

            # 将交叉注意力缓存添加到现在的键/值元组的位置3,4
            cross_attn_present_key_value = cross_attention_outputs[-1]
            present_key_value = present_key_value + cross_attn_present_key_value

        # 应用分块技术对前向传播进行处理
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        # 如果是解码器，将注意力键/值作为最后一个输出返回
        if self.is_decoder:
            outputs = outputs + (present_key_value,)

        return outputs

    def feed_forward_chunk(self, attention_output):
        # 使用中间层进行前向传播
        intermediate_output = self.intermediate(attention_output)
        # 使用输出层进行前向传播
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
# 从transformers.models.roberta.modeling_roberta.RobertaEncoder复制而来，将Roberta替换为BridgeTowerText
class BridgeTowerTextEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 创建包含多个BridgeTowerTextLayer的ModuleList，数量为config中指定的隐藏层数量
        self.layer = nn.ModuleList([BridgeTowerTextLayer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPastAndCrossAttentions]:
        # 初始化变量用于保存所有隐藏状态
        all_hidden_states = () if output_hidden_states else None
        # 初始化变量用于保存所有自注意力权重
        all_self_attentions = () if output_attentions else None
        # 初始化变量用于保存所有跨注意力权重
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        # 检查是否开启了梯度检查点和模型处于训练状态，若使用缓存则发出警告并关闭缓存
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # 初始化变量用于保存下一个解码器缓存
        next_decoder_cache = () if use_cache else None
        # 遍历所有解码器层
        for i, layer_module in enumerate(self.layer):
            # 如果需要输出隐藏状态，则将当前隐藏状态添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 获取当前层的头部注意力掩码和过去的键值对，如果不存在则设为 None
            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None

            # 如果开启了梯度检查点且处于训练状态，则使用梯度检查点函数调用当前层的前向传播
            if self.gradient_checkpointing and self.training:
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
            # 否则直接调用当前层的前向传播
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask,
                    layer_head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    past_key_value,
                    output_attentions,
                )

            # 更新当前隐藏状态为当前层的输出的第一个元素
            hidden_states = layer_outputs[0]
            # 如果使用缓存，则将当前层的缓存添加到 next_decoder_cache 中
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)
            # 如果需要输出注意力权重，则将当前层的自注意力权重添加到 all_self_attentions 中，
            # 并且如果模型配置中包含跨注意力，则将当前层的跨注意力权重添加到 all_cross_attentions 中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        # 如果需要输出隐藏状态，则将当前隐藏状态添加到 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果 return_dict 为 False，则返回元组形式的结果
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
        # 否则，返回字典形式的结果
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
# 从transformers.models.roberta.modeling_roberta.RobertaEmbeddings中复制代码，将Roberta改为BridgeTowerText
class BridgeTowerTextEmbeddings(nn.Module):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    # 从transformers.models.bert.modeling_bert.BertEmbeddings.__init__中复制代码
    def __init__(self, config):
        # 调用父类构造函数
        super().__init__()
        # 初始化词嵌入层，词汇量大小为config.vocab_size，隐藏层大小为config.hidden_size，padding的token ID为config.pad_token_id
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 初始化位置嵌入层，最大位置嵌入长度为config.max_position_embeddings，隐藏层大小为config.hidden_size
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # 初始化token类型嵌入层，token类型词汇量大小为config.type_vocab_size，隐藏层大小为config.hidden_size
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # self.LayerNorm没有采用蛇形命名，以保持与TensorFlow模型变量名称一致，并能够加载任何TensorFlow检查点文件
        # 初始化LayerNorm层，隐藏层大小为config.hidden_size，epsilon为config.layer_norm_eps
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化dropout层，概率为config.hidden_dropout_prob
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 如果存在position_embedding_type属性，则采用其值，否则默认为"absolute"
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        # 将位置ID张量注册为buffer，大小为(1, config.max_position_embeddings)，persistent为False
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        # 将token类型ID张量注册为buffer，大小与position_ids相同，类型为long，值全为0，persistent为False
        self.register_buffer(
            "token_type_ids", torch.zeros(self.position_ids.size(), dtype=torch.long), persistent=False
        )

        # End copy
        # 初始化padding的token ID
        self.padding_idx = config.pad_token_id
        # 初始化位置嵌入层，最大位置嵌入长度为config.max_position_embeddings，隐藏层大小为config.hidden_size，padding的token ID为self.padding_idx
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx
        )

    # 前向传播函数
    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
        ):
            # 如果没有提供位置 ids，则根据输入的 token ids 创建位置 ids。任何填充的 token 保持填充状态。
            if position_ids is None:
                if input_ids is not None:
                    # 从输入的 token ids 创建位置 ids。任何填充的 token 保持填充状态。
                    position_ids = create_position_ids_from_input_ids(input_ids, self.padding_idx, past_key_values_length)
                else:
                    # 从输入的嵌入张量创建位置 ids
                    position_ids = self.create_position_ids_from_inputs_embeds(inputs_embeds)

            if input_ids is not None:
                input_shape = input_ids.size()
            else:
                input_shape = inputs_embeds.size()[:-1]

            seq_length = input_shape[1]

            # 将 token_type_ids 设置为构造函数中注册的缓冲区，其中全部为零，通常在自动生成时出现，注册的缓冲区在没有传递 token_type_ids 时帮助用户追踪模型，解决问题 #5664
            if token_type_ids is None:
                if hasattr(self, "token_type_ids"):
                    buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                    buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                    token_type_ids = buffered_token_type_ids_expanded
                else:
                    # 使用全部为零的 token_type_ids
                    token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

            if inputs_embeds is None:
                # 如果没有提供输入的嵌入张量，则使用输入的 token ids 获取嵌入张量
                inputs_embeds = self.word_embeddings(input_ids)
            # 根据 token_type_ids 获取 token type 嵌入
            token_type_embeddings = self.token_type_embeddings(token_type_ids)

            # 将嵌入张量与 token type 嵌入相加
            embeddings = inputs_embeds + token_type_embeddings
            if self.position_embedding_type == "absolute":
                # 如果位置嵌入类型为 "absolute"，则添加位置嵌入
                position_embeddings = self.position_embeddings(position_ids)
                embeddings += position_embeddings
            # LayerNorm 归一化处理
            embeddings = self.LayerNorm(embeddings)
            # 使用 dropout 进行处理
            embeddings = self.dropout(embeddings)
            return embeddings

        def create_position_ids_from_inputs_embeds(self, inputs_embeds):
            """
            We are provided embeddings directly. We cannot infer which are padded so just generate sequential position ids.

            Args:
                inputs_embeds: torch.Tensor

            Returns: torch.Tensor
            """
            input_shape = inputs_embeds.size()[:-1]
            sequence_length = input_shape[1]

            # 生成连续的位置 ids
            position_ids = torch.arange(
                self.padding_idx + 1, sequence_length + self.padding_idx + 1, dtype=torch.long, device=inputs_embeds.device
            )
            return position_ids.unsqueeze(0).expand(input_shape)
# 从输入的 input_ids 中创建位置 id，用于替换非填充符号的位置数字，位置数字从 padding_idx+1 开始，填充符号被忽略
def create_position_ids_from_input_ids(input_ids, padding_idx, past_key_values_length=0):
    """
    Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
    are ignored. This is modified from fairseq's `utils.make_positions`.

    Args:
        x: torch.Tensor x:

    Returns: torch.Tensor
    """
    # 使用 input_ids 中不是填充符号的位置创建一个 mask
    mask = input_ids.ne(padding_idx).int()
    # 计算增量索引，将其转换为与 mask 相同类型，并加上 past_key_values_length，然后乘以 mask
    incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
    # 返回增量索引加上 padding_idx 的结果，并转换为 long 类型
    return incremental_indices.long() + padding_idx


# BridgeTowerPreTrainedModel 类，用于处理权重初始化和下载预训练模型的简单接口
class BridgeTowerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 配置类为 BridgeTowerConfig
    config_class = BridgeTowerConfig
    # 基础模型前缀为 "bridgetower"
    base_model_prefix = "bridgetower"
    # 不支持梯度检查点
    supports_gradient_checkpointing = False
    # 不需要拆分的模块列表
    _no_split_modules = ["BridgeTowerSelfAttention", "BridgeTowerResidualAttention"]
    # 跳过设备放置的键名为 "past_key_values"
    _skip_keys_device_placement = "past_key_values"

    # 初始化权重的方法
    def _init_weights(self, module):
        # 如果 module 是 BridgeTowerVisionModel 类型
        if isinstance(module, BridgeTowerVisionModel):
            # 计算各种标准差
            proj_std = (module.visual.transformer.hidden_size**-0.5) * (
                (2 * module.visual.transformer.num_hidden_layers) ** -0.5
            )
            attn_std = module.visual.transformer.hidden_size**-0.5
            fc_std = (2 * module.visual.transformer.hidden_size) ** -0.5
            # 初始化各个模块的权重
            for block in module.visual.transformer.resblocks:
                nn.init.normal_(block.attn.in_proj_weight, std=attn_std * self.config.initializer_factor)
                nn.init.normal_(block.attn.out_proj.weight, std=proj_std * self.config.initializer_factor)
                nn.init.normal_(block.mlp.c_fc.weight, std=fc_std * self.config.initializer_factor)
                nn.init.normal_(block.mlp.c_proj.weight, std=proj_std * self.config.initializer_factor)

            nn.init.normal_(module.visual.embeddings.class_embedding, std=attn_std * self.config.initializer_factor)
            nn.init.normal_(
                module.visual.embeddings.position_embedding.weight, std=attn_std * self.config.initializer_factor
            )
        # 如果 module 是 nn.Linear、nn.Conv2d 或 nn.Embedding 类型
        elif isinstance(module, (nn.Linear, nn.Conv2d, nn.Embedding)):
            # 初始化权重
            module.weight.data.normal_(mean=0.0, std=0.05 * self.config.initializer_factor)
        # 如果 module 是 nn.LayerNorm 类型
        elif isinstance(module, nn.LayerNorm):
            # 初始化偏置和权重
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

        # 如果 module 是 nn.Linear 类型且有偏置
        if isinstance(module, nn.Linear) and module.bias is not None:
            # 将偏置初始化为零
            module.bias.data.zero_()


# BridgeTowerVisionModel 类，继承自 BridgeTowerPreTrainedModel 类
class BridgeTowerVisionModel(BridgeTowerPreTrainedModel):
    # 配置类为 BridgeTowerVisionConfig
    config_class = BridgeTowerVisionConfig

    # 初始化方法
    def __init__(self, config):
        super().__init__(config)
        # 创建 BridgeTowerVisionTransformer 对象并赋值给 self.visual
        self.visual = BridgeTowerVisionTransformer(config)
    # 定义一个属性方法，用于获取嵌入层权重的数据类型
    @property
    def dtype(self):
        return self.visual.embeddings.patch_embedding.weight.dtype
    
    # 定义一个前向传播方法，接收图像和图像掩码作为输入，将图像转换为指定数据类型后传递给视觉模型
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

    config_class = BridgeTowerTextConfig

    def __init__(self, config, add_pooling_layer=True):
        # 调用父类构造函数初始化模型
        super().__init__(config)
        # 保存配置
        self.config = config

        # 创建词嵌入层
        self.embeddings = BridgeTowerTextEmbeddings(config)
        # 创建编码器
        self.encoder = BridgeTowerTextEncoder(config)

        # 如果需要添加池化层，则创建池化层
        self.pooler = BridgeTowerPooler(config) if add_pooling_layer else None

        # 初始化权重并进行最终处理
        self.post_init()

    # 获取输入的嵌入
    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    # 设置输入的嵌入
    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    # 剪枝模型的头部
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    # 从 transformers.models.roberta.modeling_roberta.RobertaModel.forward 复制而来
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
    # 初始化方法，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 将配置对象保存在实例变量中
        self.config = config
        # 从配置对象中提取视觉模型配置和文本模型配置
        vision_config = config.vision_config
        text_config = config.text_config

        # 如果配置指定共享跨模态 Transformer 层
        if config.share_cross_modal_transformer_layers:
            # 创建一个线性层，用于将文本表示转换到通用的隐藏空间
            self.cross_modal_text_transform = nn.Linear(text_config.hidden_size, config.hidden_size)
            # 创建一个线性层，用于将视觉表示转换到通用的隐藏空间
            self.cross_modal_image_transform = nn.Linear(vision_config.hidden_size, config.hidden_size)
        else:
            # 否则，为每个跨模态 Transformer 层创建一个线性层列表，用于将文本表示转换到通用的隐藏空间
            self.cross_modal_text_transform = nn.ModuleList(
                [nn.Linear(text_config.hidden_size, config.hidden_size) for _ in range(config.num_hidden_layers)]
            )
            # 为每个跨模态 Transformer 层创建一个线性层列表，用于将视觉表示转换到通用的隐藏空间
            self.cross_modal_image_transform = nn.ModuleList(
                [nn.Linear(vision_config.hidden_size, config.hidden_size) for _ in range(config.num_hidden_layers)]
            )

        # 创建一个嵌入层，用于输入类型编码（例如，视觉或文本）
        self.token_type_embeddings = nn.Embedding(2, config.hidden_size)

        # 创建视觉模型对象，使用 BridgeTowerVisionModel 类初始化
        self.vision_model = BridgeTowerVisionModel(vision_config)

        # 创建文本模型对象，使用 BridgeTowerTextModel 类初始化
        self.text_model = BridgeTowerTextModel(text_config)

        # 如果视觉模型配置未指定共享 LayerNorm，并且初始化 LayerNorm 来自视觉编码器
        if not vision_config.share_layernorm and config.init_layernorm_from_vision_encoder:
            # 将每个跨模态 Transformer 层的 LayerNorm 权重和偏置初始化为视觉编码器的 LayerNorm 权重和偏置
            for ln in self.vision_model.visual.cross_modal_ln_separate:
                ln.weight.data = self.vision_model.visual.ln_post.weight.data
                ln.bias.data = self.vision_model.visual.ln_post.bias.data

        # 创建用于跨模态图像的层列表，每层都是 BridgeTowerBertCrossLayer 类的实例
        self.cross_modal_image_layers = nn.ModuleList(
            [BridgeTowerBertCrossLayer(text_config) for _ in range(config.num_hidden_layers)]
        )
        # 创建用于跨模态文本的层列表，每层都是 BridgeTowerBertCrossLayer 类的实例
        self.cross_modal_text_layers = nn.ModuleList(
            [BridgeTowerBertCrossLayer(text_config) for _ in range(config.num_hidden_layers)]
        )

        # 创建用于图像池化的 Pooler，将隐藏状态转换为池化的图像表示
        self.cross_modal_image_pooler = BridgeTowerPooler(config)
        # 创建用于文本池化的 Pooler，将隐藏状态转换为池化的文本表示
        self.cross_modal_text_pooler = BridgeTowerPooler(config)

        # 初始化 BridgeTower 组件的 LayerNorm
        self.cross_modal_text_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.cross_modal_image_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 如果配置指定共享链接塔层
        if config.share_link_tower_layers:
            # 创建共享的文本链接塔
            self.cross_modal_text_link_tower = BridgeTowerLinkTower(config)
            # 创建共享的图像链接塔
            self.cross_modal_image_link_tower = BridgeTowerLinkTower(config)
        else:
            # 否则，为每个链接塔层创建一个链接塔列表
            self.cross_modal_text_link_tower = nn.ModuleList(
                [BridgeTowerLinkTower(config) for _ in range(config.num_hidden_layers - 1)]
            )
            self.cross_modal_image_link_tower = nn.ModuleList(
                [BridgeTowerLinkTower(config) for _ in range(config.num_hidden_layers - 1)]
            )

        # 执行后初始化操作
        self.post_init()

    # 返回文本模型的输入嵌入层
    def get_input_embeddings(self):
        return self.text_model.get_input_embeddings()

    # 设置文本模型的输入嵌入层
    def set_input_embeddings(self, value):
        self.text_model.set_input_embeddings(value)

    # 将 BRIDGETOWER_INPUTS_DOCSTRING 添加到模型前向方法
    @add_start_docstrings_to_model_forward(BRIDGETOWER_INPUTS_DOCSTRING)
    # 使用装饰器替换返回文档字符串，指定输出类型为BridgeTowerModelOutput，配置类为_CONFIG_FOR_DOC
    @replace_return_docstrings(output_type=BridgeTowerModelOutput, config_class=_CONFIG_FOR_DOC)
    # 前向传播函数，接受多个输入参数
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入的文本序列的 token IDs
        attention_mask: Optional[torch.FloatTensor] = None,  # 文本序列的注意力掩码
        token_type_ids: Optional[torch.LongTensor] = None,  # 文本序列的 token 类型 IDs
        pixel_values: Optional[torch.FloatTensor] = None,  # 图像的像素值
        pixel_mask: Optional[torch.LongTensor] = None,  # 图像的掩码
        head_mask: Optional[torch.FloatTensor] = None,  # 头部的掩码
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 嵌入的输入
        image_embeds: Optional[torch.FloatTensor] = None,  # 图像的嵌入
        image_token_type_idx: Optional[int] = None,  # 图像 token 类型的索引
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回字典形式的结果
        labels: Optional[torch.LongTensor] = None,  # 标签
    # 获取分类特征的函数，接受文本特征和图像特征作为输入
    def get_cls_features(self, text_features, image_features):
        # 使用跨模态文本池化器获取文本特征的分类特征
        cls_features_text = self.cross_modal_text_pooler(text_features)
        # 使用跨模态图像池化器获取图像特征的分类特征
        cls_features_image = self.cross_modal_image_pooler(image_features)
        # 拼接文本特征和图像特征的分类特征，沿着最后一个维度拼接
        return torch.cat([cls_features_text, cls_features_image], dim=-1)
# 从transformers.models.vilt.modeling_vilt.ViltPredictionHeadTransform复制代码，并将Vilt->BridgeTower
class BridgeTowerPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层，输入和输出维度都是config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 根据config.hidden_act的类型选择激活函数
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        # 创建LayerNorm层，输入维度是config.hidden_size
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        # 全连接层处理输入hidden_states
        hidden_states = self.dense(hidden_states)
        # 使用激活函数处理hidden_states
        hidden_states = self.transform_act_fn(hidden_states)
        # 使用LayerNorm处理hidden_states
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BridgeTowerMLMHead(nn.Module):
    def __init__(self, config, weight=None):
        super().__init__()
        self.config = config
        # 创建BridgeTowerPredictionHeadTransform实例
        self.transform = BridgeTowerPredictionHeadTransform(config)
        # 创建一个全连接层，输入维度是config.hidden_size，输出维度是config.text_config.vocab_size
        self.decoder = nn.Linear(config.hidden_size, config.text_config.vocab_size, bias=False)
        # 创建一个偏置参数
        self.bias = nn.Parameter(torch.zeros(config.text_config.vocab_size))
        # 如果提供了权重参数，则使用提供的权重
        if weight is not None:
            self.decoder.weight = weight

    def forward(self, x):
        # 使用BridgeTowerPredictionHeadTransform处理输入x
        mlm_score = self.transform(x)
        # 使用全连接层处理mlm_score并加上偏置
        mlm_score = self.decoder(mlm_score) + self.bias
        return mlm_score


class BridgeTowerITMHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # 创建一个全连接层，输入维度是hidden_size，输出维度是2
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        # 使用全连接层处理输入x
        itm_score = self.fc(x)
        return itm_score


@add_start_docstrings(
    """
    BridgeTower Model with a language modeling head on top as done during pretraining.
    """,
    BRIDGETOWER_START_DOCSTRING,
)
class BridgeTowerForMaskedLM(BridgeTowerPreTrainedModel):
    _tied_weights_keys = ["mlm_score.decoder.weight"]

    def __init__(self, config):
        super().__init__(config)

        # 创建BridgeTowerModel实例
        self.bridgetower = BridgeTowerModel(config)
        # 创建BridgeTowerMLMHead实例
        self.mlm_score = BridgeTowerMLMHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    def get_output_embeddings(self):
        return self.mlm_score.decoder

    def set_output_embeddings(self, new_embeddings):
        self.mlm_score.decoder = new_embeddings

    @add_start_docstrings_to_model_forward(BRIDGETOWER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=MaskedLMOutput, config_class=_CONFIG_FOR_DOC)
    # 正向传播函数，用于执行模型的前向推断
    def forward(
        self,
        # 输入的标识符张量，表示输入的 token 序列
        input_ids: Optional[torch.LongTensor] = None,
        # 注意力掩码张量，指定哪些 token 应该被忽略
        attention_mask: Optional[torch.FloatTensor] = None,
        # 分段标识符张量，用于区分不同的句子
        token_type_ids: Optional[torch.LongTensor] = None,
        # 图像像素值张量，用于图像输入的像素数据
        pixel_values: Optional[torch.FloatTensor] = None,
        # 图像像素掩码张量，指定哪些像素应该被忽略
        pixel_mask: Optional[torch.LongTensor] = None,
        # 头部掩码张量，指定哪些注意力头应该被忽略
        head_mask: Optional[torch.FloatTensor] = None,
        # 嵌入式输入张量，用于替代模型的输入嵌入
        inputs_embeds: Optional[torch.FloatTensor] = None,
        # 图像嵌入张量，用于替代模型的图像嵌入
        image_embeds: Optional[torch.FloatTensor] = None,
        # 是否输出注意力权重
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态
        output_hidden_states: Optional[bool] = None,
        # 是否以字典形式返回结果
        return_dict: Optional[bool] = None,
        # 标签张量，用于模型训练时的标签
        labels: Optional[torch.LongTensor] = None,
    ) -> Union[MaskedLMOutput, Tuple[torch.FloatTensor]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
        Returns:

        Examples:

        ```python
        >>> from transformers import BridgeTowerProcessor, BridgeTowerForMaskedLM
        >>> from PIL import Image
        >>> import requests

        >>> url = "http://images.cocodataset.org/val2017/000000360943.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
        >>> text = "a <mask> looking out of the window"

        >>> processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-base-itm-mlm")
        >>> model = BridgeTowerForMaskedLM.from_pretrained("BridgeTower/bridgetower-base-itm-mlm")

        >>> # prepare inputs
        >>> encoding = processor(image, text, return_tensors="pt")

        >>> # forward pass
        >>> outputs = model(**encoding)

        >>> results = processor.decode(outputs.logits.argmax(dim=-1).squeeze(0).tolist())

        >>> print(results)
        .a cat looking out of the window.
        ```"""
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 使用传入的参数或者模型配置中的设置来确定是否返回字典格式的输出
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

        mlm_logits = self.mlm_score(outputs.text_features if return_dict else outputs[0])
        # 获取 MLM 模型的输出结果
        masked_lm_loss = None
        if labels is not None:
            # 若有提供标签，则计算 MLM 损失
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            labels = labels.to(mlm_logits.device)
            masked_lm_loss = loss_fct(mlm_logits.view(-1, self.config.text_config.vocab_size), labels.view(-1))

        if not return_dict:
            # 如果不需要返回字典格式的输出，则组装并返回结果元组
            output = tuple(mlm_logits)
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        # 如果需要返回字典格式的输出，则构建 MaskedLMOutput 对象并返回
        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=mlm_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 为图像到文本匹配设计的 BridgeTower 模型转换器，其顶部有一个分类器头（在 [CLS] token 的最终隐藏状态之上的线性层）
# 这个模型继承自 BridgeTowerPreTrainedModel 类
@add_start_docstrings(
    """
    BridgeTower Model transformer with a classifier head on top (a linear layer on top of the final hidden state of the
    [CLS] token) for image-to-text matching.
    """,
    BRIDGETOWER_START_DOCSTRING,
)
class BridgeTowerForImageAndTextRetrieval(BridgeTowerPreTrainedModel):
    def __init__(self, config):
        # 调用父类构造函数，初始化模型参数
        super().__init__(config)

        # 初始化 BridgeTowerModel 模型
        self.bridgetower = BridgeTowerModel(config)

        # 初始化 BridgeTowerITMHead 模型，用于图像到文本匹配
        self.itm_score = BridgeTowerITMHead(config.hidden_size * 2)

        # 初始化权重并应用最终处理
        self.post_init()

    # 前向传播函数
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
    ) -> Union[SequenceClassifierOutput, Tuple[torch.FloatTensor]]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, 1)`, *optional*):
            Labels for computing the image-text matching loss. 0 means the pairs don't match and 1 means they match.
            The pairs with 0 will be skipped for calculation.
        Returns:

        Examples:

        ```python
        >>> from transformers import BridgeTowerProcessor, BridgeTowerForImageAndTextRetrieval
        >>> import requests
        >>> from PIL import Image

        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> texts = ["An image of two cats chilling on a couch", "A football player scoring a goal"]

        >>> processor = BridgeTowerProcessor.from_pretrained("BridgeTower/bridgetower-base-itm-mlm")
        >>> model = BridgeTowerForImageAndTextRetrieval.from_pretrained("BridgeTower/bridgetower-base-itm-mlm")

        >>> # forward pass
        >>> scores = dict()
        >>> for text in texts:
        ...     # prepare inputs
        ...     encoding = processor(image, text, return_tensors="pt")
        ...     outputs = model(**encoding)
        ...     scores[text] = outputs.logits[0, 1].item()
        ```"""
        # 设置返回字典的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 进行模型的前向传播
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

        # 如果设定了返回字典，则获取池化后的输出
        pooler_output = outputs.pooler_output if return_dict else outputs[2]

        # 计算图像-文本匹配的分数
        logits = self.itm_score(pooler_output)

        # 如果提供了标签，则计算图像-文本匹配的损失
        itm_loss = None
        if labels is not None:
            # 使用交叉熵损失函数
            loss_fct = CrossEntropyLoss()

            # 将标签移动到相同的设备上
            labels = labels.to(logits.device)
            # 计算图像-文本匹配的损失
            itm_loss = loss_fct(logits, labels)

        # 如果不要求返回字典，则返回logits或者包含损失的元组
        if not return_dict:
            output = tuple(logits)
            return ((itm_loss,) + output) if itm_loss is not None else output

        # 返回包含损失、logits、隐藏状态和注意力的对象
        return SequenceClassifierOutput(
            loss=itm_loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
class BridgeTowerContrastiveHead(nn.Module):
    # 定义一个用于图像-文本对比学习的对比头部模块
    def __init__(self, hidden_size, embed_size):
        super().__init__()
        # 创建一个全连接层，用于将隐藏层的特征映射到嵌入空间
        self.fc = nn.Linear(hidden_size, embed_size)

    def forward(self, x):
        # 将输入数据通过全连接层映射到嵌入空间
        x = self.fc(x)
        return x


@add_start_docstrings(
    """
    BridgeTower Model with a image-text contrastive head on top computing image-text contrastive loss.
    """,
    BRIDGETOWER_START_DOCSTRING,
)
class BridgeTowerForContrastiveLearning(BridgeTowerPreTrainedModel):
    # 定义一个在 BridgeTower 模型上加入图像-文本对比学习头部的模型
    def __init__(self, config):
        super().__init__(config)

        # 初始化 BridgeTower 模型
        self.bridgetower = BridgeTowerModel(config)

        # 初始化图像和文本的对比头部
        self.itc_text_head = BridgeTowerContrastiveHead(config.hidden_size, config.contrastive_hidden_size)
        self.itc_image_head = BridgeTowerContrastiveHead(config.hidden_size, config.contrastive_hidden_size)
        self.itc_cross_modal_head = BridgeTowerContrastiveHead(config.hidden_size * 2, config.contrastive_hidden_size)

        # 初始化用于缩放对数的参数
        self.logit_scale = nn.Parameter(torch.tensor(self.config.logit_scale_init_value))
        # 初始化权重并进行最终处理
        self.post_init()

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
    ):
        # 调用 BridgeTower 模型的前向传播
        return self.bridgetower(
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
            return_loss=return_loss,
        )
```