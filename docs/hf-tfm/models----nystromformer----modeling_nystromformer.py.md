# `.\models\nystromformer\modeling_nystromformer.py`

```py
# coding=utf-8
# 指定文件编码为 UTF-8

# 版权声明
# 2022 年 UW-Madison The HuggingFace Inc. 团队版权所有

# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证要求，否则不得使用此文件
# 可以从以下网址获取许可证副本：
# http://www.apache.org/licenses/LICENSE-2.0

# 在适用法律要求或书面同意的情况下，本软件按“原样”分发
# 没有任何明示或暗示的担保或条件

# 引入 PyTorch 和其他依赖
import math
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

# 引入不同的激活函数和模型输出
from ...activations import ACT2FN
from ...modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)

# 引入模型工具和 PyTorch 实用函数
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging

# 引入 Nystromformer 的配置
from .configuration_nystromformer import NystromformerConfig

# 获取 logger 对象
logger = logging.get_logger(__name__)

# 用于文档的检查点路径
_CHECKPOINT_FOR_DOC = "uw-madison/nystromformer-512"
# 用于文档的配置名称
_CONFIG_FOR_DOC = "NystromformerConfig"

# 预训练模型存档列表
NYSTROMFORMER_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "uw-madison/nystromformer-512",
    # 查看所有 Nyströmformer 模型 https://huggingface.co/models?filter=nystromformer
]


class NystromformerEmbeddings(nn.Module):
    """构建来自单词、位置和标记类型嵌入的嵌入层。"""
    # 初始化函数，接受一个配置对象 config
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 初始化词嵌入层，vocab_size 是词汇表大小，hidden_size 是隐藏层大小，padding_idx 是填充标记的索引
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 初始化位置嵌入层，max_position_embeddings 是最大位置嵌入数量，hidden_size 是隐藏层大小
        self.position_embeddings = nn.Embedding(config.max_position_embeddings + 2, config.hidden_size)
        # 初始化 token 类型嵌入层，type_vocab_size 是 token 类型数量，hidden_size 是隐藏层大小
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # 初始化 LayerNorm 层，用于归一化隐藏层的输出，eps 是归一化的 epsilon 参数
        # 注释中提到不采用蛇形命名以与 TensorFlow 模型变量名保持一致，以便加载 TensorFlow 的检查点文件
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化 Dropout 层，用于随机置零隐藏层的部分神经元，以防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # 注册缓冲区 position_ids，torch.arange 创建一个从 0 到 max_position_embeddings 的序列并扩展维度，加 2 是因为序号从 2 开始
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)) + 2, persistent=False
        )
        # 设置位置嵌入类型，默认为绝对位置编码
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        
        # 注册缓冲区 token_type_ids，初始化为全零张量，用于标识 token 的类型
        self.register_buffer(
            "token_type_ids",
            torch.zeros(self.position_ids.size(), dtype=torch.long, device=self.position_ids.device),
            persistent=False,
        )

    # 前向传播函数，接受多种输入形式的参数，并返回嵌入向量
    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None):
        # 如果传入 input_ids，则获取其形状；否则，从 inputs_embeds 获取形状
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        # 获取序列长度 seq_length
        seq_length = input_shape[1]

        # 如果未传入 position_ids，则使用预先注册的 position_ids 的前 seq_length 个位置
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        # 如果未传入 token_type_ids，则根据注册的缓冲区生成全零 token_type_ids
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        # 如果未传入 inputs_embeds，则使用 input_ids 获取词嵌入
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        
        # 获取 token 类型嵌入
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 将词嵌入和 token 类型嵌入相加得到最终嵌入向量 embeddings
        embeddings = inputs_embeds + token_type_embeddings
        
        # 如果使用绝对位置编码，则加上位置嵌入
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        
        # 应用 LayerNorm 进行归一化处理
        embeddings = self.LayerNorm(embeddings)
        # 应用 Dropout 进行随机置零处理，以防止过拟合
        embeddings = self.dropout(embeddings)
        
        # 返回最终的嵌入向量
        return embeddings
# NystromformerSelfAttention 类定义，继承自 nn.Module，用于自注意力机制
class NystromformerSelfAttention(nn.Module):
    # 初始化函数，接受配置 config 和位置嵌入类型 position_embedding_type（可选）
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 检查隐藏大小是否能被注意力头数整除，若不能且没有嵌入大小属性，则引发 ValueError
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        # 设置注意力头数和每个头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 设置 Nystromformer 特有的参数
        self.num_landmarks = config.num_landmarks  # 地标数目
        self.seq_len = config.segment_means_seq_len  # 序列长度
        self.conv_kernel_size = config.conv_kernel_size  # 卷积核大小

        # 初始化选项设定
        if config.inv_coeff_init_option:
            self.init_option = config["inv_init_coeff_option"]
        else:
            self.init_option = "original"

        # 线性变换层，用于计算查询、键、值
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        # Dropout 层，用于注意力概率的随机失活
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = position_embedding_type or getattr(
            config, "position_embedding_type", "absolute"
        )

        # 若卷积核大小不为 None，则初始化卷积层
        if self.conv_kernel_size is not None:
            self.conv = nn.Conv2d(
                in_channels=self.num_attention_heads,
                out_channels=self.num_attention_heads,
                kernel_size=(self.conv_kernel_size, 1),
                padding=(self.conv_kernel_size // 2, 0),
                bias=False,
                groups=self.num_attention_heads,
            )

    # 迭代方法近似计算 Moore-Penrose 伪逆
    def iterative_inv(self, mat, n_iter=6):
        # 创建单位矩阵并复制到与输入矩阵相同的设备上
        identity = torch.eye(mat.size(-1), device=mat.device)
        key = mat

        # 根据初始化选项选择不同的计算公式来计算初始系数矩阵 value
        if self.init_option == "original":
            # 原始实现，更保守地计算 Z_0 的系数
            value = 1 / torch.max(torch.sum(key, dim=-2)) * key.transpose(-1, -2)
        else:
            # 精确系数计算，计算 Z_0 的初始化系数，加快收敛速度
            value = 1 / torch.max(torch.sum(key, dim=-2), dim=-1).values[:, :, None, None] * key.transpose(-1, -2)

        # 迭代更新系数矩阵 value
        for _ in range(n_iter):
            key_value = torch.matmul(key, value)
            value = torch.matmul(
                0.25 * value,
                13 * identity
                - torch.matmul(key_value, 15 * identity - torch.matmul(key_value, 7 * identity - key_value)),
            )

        # 返回更新后的系数矩阵 value
        return value
    # 对输入的张量进行维度重塑，以便进行注意力分数计算
    def transpose_for_scores(self, layer):
        # 计算新的张量形状，保持前面的维度不变，最后两个维度分别为注意力头的数量和每个头的大小
        new_layer_shape = layer.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        # 将输入张量按照新的形状重塑
        layer = layer.view(*new_layer_shape)
        # 对张量进行维度置换，将注意力头的数量移到第二个维度，保证在矩阵乘法时能够正确对齐
        return layer.permute(0, 2, 1, 3)
    # 定义神经网络的前向传播函数，接受隐藏状态、注意力掩码和是否输出注意力矩阵作为参数
    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        # 计算混合查询层，通过查询函数生成
        mixed_query_layer = self.query(hidden_states)

        # 计算键层，并为得分转置以进行注意力计算
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        
        # 计算值层，并为得分转置以进行注意力计算
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # 对查询层进行得分归一化，使用平方根进行缩放
        query_layer = self.transpose_for_scores(mixed_query_layer)
        query_layer = query_layer / math.sqrt(math.sqrt(self.attention_head_size))
        
        # 对键层进行得分归一化，使用平方根进行缩放
        key_layer = key_layer / math.sqrt(math.sqrt(self.attention_head_size))

        # 如果 landmarks 的数量等于序列长度
        if self.num_landmarks == self.seq_len:
            # 计算注意力分数
            attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

            # 如果有注意力掩码，将其应用于注意力分数
            if attention_mask is not None:
                # 应用预先计算的注意力掩码（在 NystromformerModel 的 forward() 函数中计算）
                attention_scores = attention_scores + attention_mask

            # 计算注意力概率
            attention_probs = nn.functional.softmax(attention_scores, dim=-1)
            
            # 计算上下文层，通过加权值层得出
            context_layer = torch.matmul(attention_probs, value_layer)

        # 如果 landmarks 的数量不等于序列长度
        else:
            # 将查询层和键层重塑以处理 landmarks
            q_landmarks = query_layer.reshape(
                -1,
                self.num_attention_heads,
                self.num_landmarks,
                self.seq_len // self.num_landmarks,
                self.attention_head_size,
            ).mean(dim=-2)
            k_landmarks = key_layer.reshape(
                -1,
                self.num_attention_heads,
                self.num_landmarks,
                self.seq_len // self.num_landmarks,
                self.attention_head_size,
            ).mean(dim=-2)

            # 计算 kernel_1，通过查询层和键 landmarks 的转置进行 softmax
            kernel_1 = torch.nn.functional.softmax(torch.matmul(query_layer, k_landmarks.transpose(-1, -2)), dim=-1)
            
            # 计算 kernel_2，通过 landmarks 之间的注意力关系进行 softmax
            kernel_2 = torch.nn.functional.softmax(torch.matmul(q_landmarks, k_landmarks.transpose(-1, -2)), dim=-1)

            # 计算 q_landmarks 和键层之间的注意力分数
            attention_scores = torch.matmul(q_landmarks, key_layer.transpose(-1, -2))

            # 如果有注意力掩码，将其应用于注意力分数
            if attention_mask is not None:
                # 应用预先计算的注意力掩码（在 NystromformerModel 的 forward() 函数中计算）
                attention_scores = attention_scores + attention_mask

            # 计算 kernel_3，通过 landmarks 和键层之间的注意力关系进行 softmax
            kernel_3 = nn.functional.softmax(attention_scores, dim=-1)
            
            # 计算注意力概率，通过 kernel_1 和 kernel_2 的加权平均并使用 iterative_inv 函数反演
            attention_probs = torch.matmul(kernel_1, self.iterative_inv(kernel_2))
            
            # 计算新的值层，通过 kernel_3 和值层的乘积得出
            new_value_layer = torch.matmul(kernel_3, value_layer)
            
            # 计算上下文层，通过加权新值层得出
            context_layer = torch.matmul(attention_probs, new_value_layer)

        # 如果存在卷积核大小，则将卷积操作应用于上下文层
        if self.conv_kernel_size is not None:
            context_layer += self.conv(value_layer)

        # 对上下文层进行维度置换和重塑，以适应多头注意力的结构
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # 根据是否需要输出注意力矩阵，返回相应的输出结果
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        # 返回最终的输出结果
        return outputs
# Copied from transformers.models.bert.modeling_bert.BertSelfOutput
class NystromformerSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化一个全连接层，用于变换输入的隐藏状态到同样大小的输出
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 初始化 LayerNorm 层，用于规范化输出向量
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化 Dropout 层，以概率 config.hidden_dropout_prob 随机丢弃部分数据，防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 使用全连接层变换隐藏状态
        hidden_states = self.dense(hidden_states)
        # 对变换后的隐藏状态进行 Dropout
        hidden_states = self.dropout(hidden_states)
        # 对变换后的隐藏状态和输入张量 input_tensor 进行残差连接，并进行 LayerNorm 规范化
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class NystromformerAttention(nn.Module):
    def __init__(self, config, position_embedding_type=None):
        super().__init__()
        # 初始化自注意力层
        self.self = NystromformerSelfAttention(config, position_embedding_type=position_embedding_type)
        # 初始化输出层
        self.output = NystromformerSelfOutput(config)
        # 初始化一个集合，用于存储要剪枝的注意力头部
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        # 调用 find_pruneable_heads_and_indices 函数找到可以剪枝的头部索引
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 剪枝线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储已剪枝的头部
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        # 调用自注意力层的 forward 方法，获取自注意力层的输出
        self_outputs = self.self(hidden_states, attention_mask, output_attentions)
        # 将自注意力层的输出作为输入，调用输出层的 forward 方法
        attention_output = self.output(self_outputs[0], hidden_states)
        # 如果需要输出注意力权重，将它们加入到 outputs 中
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs


# Copied from transformers.models.bert.modeling_bert.BertIntermediate with Bert->Nystromformer
class NystromformerIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化一个全连接层，用于变换输入的隐藏状态到中间隐藏大小
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 根据配置选择激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 使用全连接层变换隐藏状态
        hidden_states = self.dense(hidden_states)
        # 应用中间层激活函数
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOutput with Bert->Nystromformer
# 定义一个名为 NystromformerOutput 的类，继承自 nn.Module
class NystromformerOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个线性层，将输入特征大小转换为隐藏大小
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # LayerNorm 层，对隐藏状态进行归一化处理
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Dropout 层，用于随机失活以防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # 前向传播方法
    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        # 线性层处理隐藏状态
        hidden_states = self.dense(hidden_states)
        # 使用 Dropout 层随机失活
        hidden_states = self.dropout(hidden_states)
        # LayerNorm 层归一化处理并加上输入张量
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# 定义一个名为 NystromformerLayer 的类，继承自 nn.Module
class NystromformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 定义用于分块处理前馈的块大小
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        # 序列长度的维度
        self.seq_len_dim = 1
        # NystromformerAttention 类的实例
        self.attention = NystromformerAttention(config)
        # 是否添加跨层注意力
        self.add_cross_attention = config.add_cross_attention
        # NystromformerIntermediate 类的实例
        self.intermediate = NystromformerIntermediate(config)
        # NystromformerOutput 类的实例
        self.output = NystromformerOutput(config)

    # 前向传播方法
    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        # 使用注意力机制处理隐藏状态
        self_attention_outputs = self.attention(hidden_states, attention_mask, output_attentions=output_attentions)
        attention_output = self_attention_outputs[0]

        # 如果需要输出注意力权重，添加自注意力结果到输出中
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # 对注意力输出进行分块处理
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        return outputs

    # 前馈处理的分块方法
    def feed_forward_chunk(self, attention_output):
        # 中间层处理注意力输出
        intermediate_output = self.intermediate(attention_output)
        # 输出层处理中间输出和注意力输出
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


# 定义一个名为 NystromformerEncoder 的类，继承自 nn.Module
class NystromformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 使用 NystromformerLayer 类创建指定数量的层
        self.layer = nn.ModuleList([NystromformerLayer(config) for _ in range(config.num_hidden_layers)])
        # 梯度检查点，默认为 False
        self.gradient_checkpointing = False

    # 前向传播方法
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        ):
        # 遍历每一层，依次调用其前向传播方法
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask, output_attentions=output_attentions)[0]

        return hidden_states
        ):
            # 如果不输出隐藏状态，则初始化为空元组
            all_hidden_states = () if output_hidden_states else None
            # 如果不输出注意力权重，则初始化为空元组
            all_self_attentions = () if output_attentions else None

            # 遍历每个 Transformer 层
            for i, layer_module in enumerate(self.layer):
                # 如果需要输出隐藏状态，则追加当前层的隐藏状态到 all_hidden_states 元组中
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                # 如果启用了梯度检查点且在训练阶段
                if self.gradient_checkpointing and self.training:
                    # 使用梯度检查点函数来调用当前层的前向传播
                    layer_outputs = self._gradient_checkpointing_func(
                        layer_module.__call__,
                        hidden_states,
                        attention_mask,
                        output_attentions,
                    )
                else:
                    # 否则，直接调用当前层的前向传播
                    layer_outputs = layer_module(hidden_states, attention_mask, output_attentions)

                # 更新隐藏状态为当前层的输出的第一个元素
                hidden_states = layer_outputs[0]
                # 如果需要输出注意力权重，则追加当前层的注意力权重到 all_self_attentions 元组中
                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)

            # 如果需要输出隐藏状态，则追加最终的隐藏状态到 all_hidden_states 元组中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 如果不返回字典形式的输出，则返回非空的元组，包括隐藏状态、所有隐藏状态和所有注意力权重
            if not return_dict:
                return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
            # 如果返回字典形式的输出，则创建 BaseModelOutputWithPastAndCrossAttentions 对象
            return BaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=hidden_states,
                hidden_states=all_hidden_states,
                attentions=all_self_attentions,
            )
# Copied from transformers.models.bert.modeling_bert.BertPredictionHeadTransform with Bert->Nystromformer
class NystromformerPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个全连接层，将输入特征维度变换为配置文件中指定的隐藏层大小
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # 根据配置文件中指定的激活函数类型选择相应的激活函数
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        # 创建一个 LayerNorm 层，用于规范化隐藏状态张量
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 输入隐藏状态经过全连接层变换
        hidden_states = self.dense(hidden_states)
        # 变换后的隐藏状态经过选择的激活函数
        hidden_states = self.transform_act_fn(hidden_states)
        # 变换后的隐藏状态经过 LayerNorm 规范化
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertLMPredictionHead with Bert->Nystromformer
class NystromformerLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个 NystromformerPredictionHeadTransform 对象，用于处理隐藏状态
        self.transform = NystromformerPredictionHeadTransform(config)

        # 输出权重与输入嵌入大小相同，但每个标记有一个输出偏置
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 创建一个与词汇表大小相同的偏置参数
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # 需要在两个变量之间建立链接，以便偏置在 `resize_token_embeddings` 时能正确调整大小
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        # 隐藏状态经过 NystromformerPredictionHeadTransform 处理
        hidden_states = self.transform(hidden_states)
        # 处理后的隐藏状态经过线性层变换得到预测分数
        hidden_states = self.decoder(hidden_states)
        return hidden_states


# Copied from transformers.models.bert.modeling_bert.BertOnlyMLMHead with Bert->Nystromformer
class NystromformerOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个 NystromformerLMPredictionHead 对象，用于生成 MLM 预测分数
        self.predictions = NystromformerLMPredictionHead(config)

    def forward(self, sequence_output: torch.Tensor) -> torch.Tensor:
        # 序列输出作为输入传递给预测头部生成预测分数
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class NystromformerPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 指定 NystromformerConfig 作为配置类
    config_class = NystromformerConfig
    # 指定模型前缀
    base_model_prefix = "nystromformer"
    # 支持梯度检查点
    supports_gradient_checkpointing = True
    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果 module 是线性层或者二维卷积层
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 使用正态分布初始化权重，均值为 0，标准差为配置中的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在偏置项，则将其初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果 module 是嵌入层
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化权重，均值为 0，标准差为配置中的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果定义了填充索引，则将对应的权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # 如果 module 是 LayerNorm 层
        elif isinstance(module, nn.LayerNorm):
            # 将偏置项初始化为零
            module.bias.data.zero_()
            # 将权重初始化为全1
            module.weight.data.fill_(1.0)
NYSTROMFORMER_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.
    
    Parameters:
        config ([`NystromformerConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""



NYSTROMFORMER_INPUTS_DOCSTRING = r"""
    This is a docstring describing the inputs expected by the Nystromformer model.

    Inputs:
        **input_ids** (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.
        **attention_mask** (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, optional):
            Mask to avoid performing attention on padding tokens.
        **position_ids** (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, optional):
            Indices of positions of each input sequence tokens in the position embeddings.
        **inputs_embeds** (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, optional):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful when you want more control over how to convert input tokens to embeddings before feeding them
            into the model.

    Returns:
        :obj:`torch.Tensor`: Returns tensor(s) containing the model outputs.
"""
        Args:
            input_ids (`torch.LongTensor` of shape `({0})`):
                # 输入序列标记的索引，对应词汇表中的位置。

                Indices of input sequence tokens in the vocabulary.
                Indices can be obtained using `AutoTokenizer`. See `PreTrainedTokenizer.encode` and
                `PreTrainedTokenizer.__call__` for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.FloatTensor` of shape `({0})`, *optional*):
                # 遮罩，用于在填充的标记索引上避免进行注意力计算。

                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            token_type_ids (`torch.LongTensor` of shape `({0})`, *optional*):
                # 分段标记索引，指示输入的第一和第二部分。

                Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0,
                1]`:

                - 0 corresponds to a *sentence A* token,
                - 1 corresponds to a *sentence B* token.

                [What are token type IDs?](../glossary#token-type-ids)
            position_ids (`torch.LongTensor` of shape `({0})`, *optional*):
                # 输入序列标记在位置嵌入中的位置索引。

                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
                config.max_position_embeddings - 1]`.

                [What are position IDs?](../glossary#position-ids)
            head_mask (`torch.FloatTensor` of shape `(num_heads,)` or `(num_layers, num_heads)`, *optional*):
                # 用于选择自注意力模块中屏蔽的头部的掩码。

                Mask to nullify selected heads of the self-attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            inputs_embeds (`torch.FloatTensor` of shape `({0}, hidden_size)`, *optional*):
                # 可选参数，代替传递`input_ids`，直接传递嵌入表示。

                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
                is useful if you want more control over how to convert *input_ids* indices into associated vectors than the
                model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                # 是否返回所有注意力层的注意力张量。

                Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
                tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                # 是否返回所有层的隐藏状态。

                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
                more detail.
            return_dict (`bool`, *optional*):
                # 是否返回`ModelOutput`而不是普通元组。

                Whether or not to return a `ModelOutput` instead of a plain tuple.
"""
@add_start_docstrings(
    "The bare Nyströmformer Model transformer outputting raw hidden-states without any specific head on top.",
    NYSTROMFORMER_START_DOCSTRING,
)
class NystromformerModel(NystromformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        # Initialize the Nystromformer embeddings and encoder based on the given configuration
        self.embeddings = NystromformerEmbeddings(config)
        self.encoder = NystromformerEncoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        # Retrieve the word embeddings from the Nystromformer embeddings
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        # Set new word embeddings for the Nystromformer embeddings
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            # Prune heads in each layer of the encoder
            self.encoder.layer[layer].attention.prune_heads(heads)

@add_start_docstrings(
    "Nyströmformer Model with a `language modeling` head on top.",
    NYSTROMFORMER_START_DOCSTRING
)
class NystromformerForMaskedLM(NystromformerPreTrainedModel):
    _tied_weights_keys = ["cls.predictions.decoder"]

    def __init__(self, config):
        super().__init__(config)

        # Initialize the Nystromformer model and MLM head based on the provided configuration
        self.nystromformer = NystromformerModel(config)
        self.cls = NystromformerOnlyMLMHead(config)

        # Initialize weights and apply final processing
        self.post_init()

    def get_output_embeddings(self):
        # Retrieve the output embeddings from the MLM head
        return self.cls.predictions.decoder

    def set_output_embeddings(self, new_embeddings):
        # Set new output embeddings for the MLM head
        self.cls.predictions.decoder = new_embeddings
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,  # 输入的token ids序列，可以为空
        attention_mask: Optional[torch.FloatTensor] = None,  # 注意力掩码，用于指示哪些位置是填充的
        token_type_ids: Optional[torch.LongTensor] = None,  # token类型ids，如用于区分句子A和句子B的位置
        position_ids: Optional[torch.LongTensor] = None,  # 位置ids，用于指定每个token的位置信息
        head_mask: Optional[torch.FloatTensor] = None,  # 头部掩码，用于控制每个注意力头部的选择性
        inputs_embeds: Optional[torch.FloatTensor] = None,  # 输入的嵌入向量，可以用于直接输入嵌入而不是token ids
        labels: Optional[torch.LongTensor] = None,  # 用于计算MLM损失的标签，指示哪些位置是被mask的
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态
        return_dict: Optional[bool] = None,  # 是否返回一个字典格式的输出
    ) -> Union[Tuple[torch.Tensor], MaskedLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
            config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked), the
            loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict  # 确定是否使用返回字典模式

        outputs = self.nystromformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]  # 获取模型输出的序列输出
        prediction_scores = self.cls(sequence_output)  # 使用线性层对序列输出进行预测得分计算

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # 定义交叉熵损失函数，用于计算MLM损失
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))  # 计算MLM损失

        if not return_dict:
            output = (prediction_scores,) + outputs[1:]  # 如果不使用字典返回，则组装输出元组
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output  # 返回损失和输出元组

        return MaskedLMOutput(
            loss=masked_lm_loss,  # 返回MLM损失
            logits=prediction_scores,  # 返回预测得分
            hidden_states=outputs.hidden_states,  # 返回隐藏状态
            attentions=outputs.attentions,  # 返回注意力权重
        )
# 定义一个用于序列级分类任务的模型头部
class NystromformerClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        # 线性层，输入尺寸为config.hidden_size，输出尺寸为config.hidden_size
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        # Dropout 层，以config.hidden_dropout_prob的概率随机置零输入张量的部分元素
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 输出投影层，将输入尺寸为config.hidden_size的张量线性映射到尺寸为config.num_labels的张量
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

        self.config = config

    def forward(self, features, **kwargs):
        # 提取features中的第一个位置处的张量（对应于<CLS> token）
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        # 对提取的张量施加dropout操作
        x = self.dropout(x)
        # 将张量输入到线性层中进行线性变换
        x = self.dense(x)
        # 根据config中指定的激活函数对输出进行非线性变换
        x = ACT2FN[self.config.hidden_act](x)
        # 再次对张量施加dropout操作
        x = self.dropout(x)
        # 将张量输入到输出投影层中进行线性映射，得到最终输出
        x = self.out_proj(x)
        return x


# 应用于序列分类/回归任务的Nyströmformer模型变换器，顶部带有一个线性层（汇聚输出的线性层），例如用于GLUE任务
@add_start_docstrings(
    """
    Nyströmformer Model transformer with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    NYSTROMFORMER_START_DOCSTRING,
)
class NystromformerForSequenceClassification(NystromformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        # 初始化模型的标签数量
        self.num_labels = config.num_labels
        # 初始化Nyströmformer模型
        self.nystromformer = NystromformerModel(config)
        # 初始化分类器头部
        self.classifier = NystromformerClassificationHead(config)

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(NYSTROMFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        定义了一个函数签名，表明该函数接受一些输入，并返回一个包含torch.Tensor或SequenceClassifierOutput的元组。
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            用于计算序列分类/回归损失的标签。索引应在 `[0, ..., config.num_labels - 1]` 范围内。
            如果 `config.num_labels == 1`，则计算回归损失（均方损失）；如果 `config.num_labels > 1`，则计算分类损失（交叉熵）。
        """
        # 初始化返回字典，如果return_dict为None，则根据self.config.use_return_dict确定
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用self.nystromformer进行前向传播
        outputs = self.nystromformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取序列输出
        sequence_output = outputs[0]
        # 将序列输出传入分类器得到logits
        logits = self.classifier(sequence_output)

        # 初始化损失为None
        loss = None
        # 如果labels不为None，则计算损失
        if labels is not None:
            # 如果self.config.problem_type未定义，则根据self.num_labels的值设定problem_type
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            # 根据problem_type计算不同类型的损失
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

        # 如果return_dict为False，则按顺序返回元组
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果return_dict为True，则返回SequenceClassifierOutput对象
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
"""
Nyströmformer Model with a multiple choice classification head on top (a linear layer on top of the pooled output
and a softmax) e.g. for RocStories/SWAG tasks.
"""

# 使用 Nyströmformer 模型，并在其顶部添加一个多选分类头部（即在池化输出之上的线性层和 softmax），例如用于 RocStories/SWAG 任务。

class NystromformerForMultipleChoice(NystromformerPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 初始化 Nyströmformer 模型
        self.nystromformer = NystromformerModel(config)
        
        # 初始化用于预分类的线性层
        self.pre_classifier = nn.Linear(config.hidden_size, config.hidden_size)
        
        # 初始化用于最终分类的线性层
        self.classifier = nn.Linear(config.hidden_size, 1)

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(
        NYSTROMFORMER_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
    )
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=MultipleChoiceModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    # 前向传播函数，接受一系列输入参数，并返回模型输出的多选分类结果
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # 参数和返回值的文档字符串，描述了输入和输出的具体格式
        ) -> Union[Tuple[torch.Tensor], MultipleChoiceModelOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the multiple choice classification loss. Indices should be in `[0, ...,
            num_choices-1]` where `num_choices` is the size of the second dimension of the input tensors. (See
            `input_ids` above)
        """
        # 根据 return_dict 是否为 None，确定是否使用配置中的 use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 计算 num_choices，如果 input_ids 不为 None，则为其第二维度的大小
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        # 如果 input_ids 不为 None，则重新调整其形状为 (batch_size * num_choices, seq_len)
        input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        # 如果 attention_mask 不为 None，则重新调整其形状为 (batch_size * num_choices, seq_len)
        attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        # 如果 token_type_ids 不为 None，则重新调整其形状为 (batch_size * num_choices, seq_len)
        token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        # 如果 position_ids 不为 None，则重新调整其形状为 (batch_size * num_choices, seq_len)
        position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        # 如果 inputs_embeds 不为 None，则重新调整其形状为 (batch_size * num_choices, seq_len, dim)
        inputs_embeds = (
            inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1))
            if inputs_embeds is not None
            else None
        )

        # 调用 NystromFormer 模型进行前向传播
        outputs = self.nystromformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取模型输出的隐藏状态
        hidden_state = outputs[0]  # (bs * num_choices, seq_len, dim)
        # 提取池化后的输出，仅保留每个序列的第一个标记的表示
        pooled_output = hidden_state[:, 0]  # (bs * num_choices, dim)
        # 通过预分类器进行线性变换
        pooled_output = self.pre_classifier(pooled_output)  # (bs * num_choices, dim)
        # 应用 ReLU 激活函数
        pooled_output = nn.ReLU()(pooled_output)  # (bs * num_choices, dim)
        # 通过分类器获取最终的 logits
        logits = self.classifier(pooled_output)

        # 重新调整 logits 的形状为 (batch_size, num_choices)
        reshaped_logits = logits.view(-1, num_choices)

        # 计算损失值
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)

        # 如果 return_dict 为 False，则返回扁平化的输出元组
        if not return_dict:
            output = (reshaped_logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # 如果 return_dict 为 True，则返回 MultipleChoiceModelOutput 对象
        return MultipleChoiceModelOutput(
            loss=loss,
            logits=reshaped_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 定义一个 Nyströmformer 模型，带有一个顶部的标记分类头部（隐藏状态输出上的线性层），用于命名实体识别（NER）等任务。
# 例如，用于命名实体识别（NER）任务的 Nyströmformer 模型，具有一个在隐藏状态输出之上的线性层作为标记分类头部。
@add_start_docstrings(
    """
    Nyströmformer Model with a token classification head on top (a linear layer on top of the hidden-states output)
    e.g. for Named-Entity-Recognition (NER) tasks.
    """,
    NYSTROMFORMER_START_DOCSTRING,
)
class NystromformerForTokenClassification(NystromformerPreTrainedModel):
    
    def __init__(self, config):
        super().__init__(config)
        # 从配置中获取标签数目
        self.num_labels = config.num_labels

        # 初始化 Nyströmformer 模型
        self.nystromformer = NystromformerModel(config)
        # 添加 dropout 层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # 添加线性分类器，将隐藏状态大小映射到标签数目
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(NYSTROMFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ) -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        """
        # 根据参数 `return_dict` 的值决定是否使用配置中的返回字典选项
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将输入传递给 Nystromformer 模型，并收集其输出
        outputs = self.nystromformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取模型输出中的序列输出（通常是最后一层的隐藏状态）
        sequence_output = outputs[0]

        # 对序列输出应用 dropout，用于防止过拟合
        sequence_output = self.dropout(sequence_output)
        # 将 dropout 后的输出传递给分类器，得到分类 logits
        logits = self.classifier(sequence_output)

        # 初始化损失值为 None
        loss = None
        # 如果提供了标签，则计算交叉熵损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # 将 logits 和标签展平后计算损失
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # 如果不需要返回字典格式的输出，则组装输出为元组
        if not return_dict:
            output = (logits,) + outputs[1:]  # 包括可能的额外输出
            return ((loss,) + output) if loss is not None else output

        # 如果需要返回字典格式的输出，则使用 TokenClassifierOutput 类封装输出
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 将 Nyströmformer 模型用于提取式问答任务，例如 SQuAD，顶部包含一个用于分类的线性层来计算“起始位置logits”和“结束位置logits”
@add_start_docstrings(
    """
    Nyströmformer Model with a span classification head on top for extractive question-answering tasks like SQuAD (a
    linear layers on top of the hidden-states output to compute `span start logits` and `span end logits`).
    """,
    NYSTROMFORMER_START_DOCSTRING,
)
class NystromformerForQuestionAnswering(NystromformerPreTrainedModel):
    
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        
        # 设置分类标签数为2（起始位置和结束位置）
        config.num_labels = 2
        self.num_labels = config.num_labels
        
        # 初始化 Nyströmformer 模型和用于问答的输出层
        self.nystromformer = NystromformerModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        # 初始化权重并进行最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(NYSTROMFORMER_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=QuestionAnsweringModelOutput,
        config_class=_CONFIG_FOR_DOC,
    )
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
        # 输入参数详细说明
):
    ) -> Union[Tuple[torch.Tensor], QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        # 根据 return_dict 参数确定是否返回一个字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 将输入传递给 NystromFormer 模型进行处理
        outputs = self.nystromformer(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从 NystromFormer 模型的输出中获取序列输出
        sequence_output = outputs[0]

        # 使用序列输出计算问题回答的 logits
        logits = self.qa_outputs(sequence_output)
        # 将 logits 按照最后一个维度分割成 start_logits 和 end_logits
        start_logits, end_logits = logits.split(1, dim=-1)
        # 去除多余的维度，使得 start_logits 和 end_logits 的形状变为 (batch_size, sequence_length)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        # 如果给定了 start_positions 和 end_positions，则计算损失
        if start_positions is not None and end_positions is not None:
            # 如果是多 GPU 环境，可能会有额外的维度，这里进行压缩处理
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # 将超出模型输入范围的 start/end positions 忽略掉
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            # 使用交叉熵损失函数，忽略索引为 ignored_index 的部分
            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            # 计算总损失
            total_loss = (start_loss + end_loss) / 2

        # 如果不要求返回字典形式的输出，直接返回 logits 和可能的损失
        if not return_dict:
            output = (start_logits, end_logits) + outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        # 如果要求返回 QuestionAnsweringModelOutput 对象，构建并返回
        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```