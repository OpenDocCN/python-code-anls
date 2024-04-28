# `.\models\deprecated\mctct\modeling_mctct.py`

```py
# 设定文件编码格式为utf-8
# 版权声明
# PyTorch M-CTC-T 模型
import math
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from ....activations import ACT2FN
from ....file_utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward
from ....integrations.deepspeed import is_deepspeed_zero3_enabled
from ....modeling_attn_mask_utils import _prepare_4d_attention_mask
from ....modeling_outputs import BaseModelOutput, CausalLMOutput
from ....modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)
from ....utils import logging
from .configuration_mctct import MCTCTConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 隐藏状态的起始位置
_HIDDEN_STATES_START_POSITION = 1

_CONFIG_FOR_DOC = "MCTCTConfig"
# 检查点地址
_CHECKPOINT_FOR_DOC = "speechbrain/m-ctc-t-large"
# 预期输出形状
_EXPECTED_OUTPUT_SHAPE = [1, 195, 1536]

# CTC 部分的期望输出和损失
_CTC_EXPECTED_OUTPUT = '"Mr. Quilter is the apostle of the middle classes, and we\'re glad to welcome his gospel."'
_CTC_EXPECTED_LOSS = 1885.65

# 预训练模型列表
MCTCT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "speechbrain/m-ctc-t-large",
    # 查看所有 M-CTC-T 模型 https://huggingface.co/models?filter=mctct
]

# 创建类 MCTCTConv1dSubsampler
class MCTCTConv1dSubsampler(nn.Module):
    """
    Convolutional subsampler: a stack of 1D convolution (along temporal dimension) followed by non-linear activation
    via gated linear units (https://arxiv.org/abs/1911.08460)
    """
    # 初始化函数，传入配置参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 保存配置参数
        self.config = config
        # 获取 GLU 层的维度
        self.glu_dim = config.conv_glu_dim
    
        # 创建一个以config.conv_dropout为概率丢弃输入张量的Dropout层
        self.dropout = nn.Dropout(config.conv_dropout)
    
        # 获取卷积层的数量和输入特征的通道数
        self.num_layers = config.num_conv_layers
        self.in_channels = config.input_feat_per_channel * config.input_channels
    
        # 如果有多个卷积层
        if self.num_layers > 1:
            # 如果未指定卷积通道数，抛出数值错误
            if config.conv_channels is None:
                raise ValueError(
                    "Need to specify `conv_channels` configuration in `MCTCTConfig` to use multiple convolution"
                    " layers."
                )
            # 保存卷积通道数
            self.mid_channels = config.conv_channels
        else:
            self.mid_channels = None
    
        # 获取输出特征的通道数、卷积核大小和步长
        self.out_channels = config.hidden_size * 2  # considering GLU halving
        self.kernel_size = config.conv_kernel
        self.stride = config.conv_stride
    
        # 创建一个卷积层的 ModuleList
        self.conv_layers = nn.ModuleList(
            nn.Conv1d(
                self.in_channels if i == 0 else self.mid_channels[i],
                self.mid_channels[i] if i < self.num_layers - 1 else self.out_channels,
                kernel_size=k,
                stride=self.stride[i],
                padding="valid",
            )
            for i, k in enumerate(self.kernel_size)
        )
    
    # 前向传播函数
    def forward(self, input_features):
        # 计算需要填充的大小
        padding = sum([size // 2 for size in self.kernel_size])  # (7, 7) -> (3, 3)
    
        # 在input_features的最后维度两侧填充0
        input_features = torch.nn.functional.pad(input_features, (0, 0, padding, padding), "constant", 0)
        # 交换输入特征的维度顺序
        hidden_states = input_features.transpose(1, 2).contiguous()  # -> Batch x Frame x Time
        # 遍历卷积层
        for conv in self.conv_layers:
            # 进行卷积操作
            hidden_states = conv(hidden_states)
            # 对卷积结果进行门控线性单元（GLU）激活函数操作
            hidden_states = nn.functional.glu(hidden_states, dim=self.glu_dim)
            # 对卷积结果进行随机丢弃操作
            hidden_states = self.dropout(hidden_states)
    
        # 再次交换特征的维度顺序
        hidden_states = hidden_states.transpose(1, 2).contiguous()  # -> Batch x Time x Frame
        # 返回最终的隐藏状态
        return hidden_states
class MCTCTEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        # 创建单词嵌入，使用nn.Embedding类，指定单词表大小、隐藏层大小和填充标记的索引
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        # 创建位置嵌入，使用nn.Embedding类，指定最大位置嵌入长度和隐藏层大小
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        # 创建标记类型嵌入，使用nn.Embedding类，指定标记类型表大小和隐藏层大小
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        # 创建LayerNorm层，用于归一化隐藏层的输出
        # LayerNorm的命名未采用蛇形命名法，以保持与TensorFlow模型变量名称一致，并能够加载任何TensorFlow检查点文件
        self.LayerNorm = MCTCTLayerNorm()
        # 创建丢弃层，用于在训练过程中随机丢弃部分隐藏层输出
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # 创建位置ID的缓冲区，包含从0到最大位置嵌入长度的序列，序列大小为(1, 最大位置嵌入长度)
        self.register_buffer(
            "position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)), persistent=False
        )
        # 创建标记类型ID的缓冲区，初始值全为0，与位置ID缓冲区相同大小
        self.register_buffer(
            "token_type_ids",
            torch.zeros(self.position_ids.size(), dtype=torch.long, device=self.position_ids.device),
            persistent=False,
        )

    def forward(
        self, input_features=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
        # 获取输入特征的形状
        input_shape = input_features.size() if input_features is not None else inputs_embeds.size()[:-1]

        # 获取序列长度
        seq_length = input_shape[1]

        # 如果位置ID为空，则将其设为从过去关键值长度到序列长度加上过去关键值长度的位置ID
        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # 如果标记类型ID为空
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                # 从构造函数中注册的缓冲区中获取标记类型ID，并扩展到与输入特征相同的大小
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                # 创建一个全为0的标记类型ID，与输入特征相同大小，设备和位置ID相同
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        # 如果输入嵌入为空，则使用单词嵌入层获取输入特征的嵌入
        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_features)

        # 获取标记类型嵌入
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        # 将单词嵌入和标记类型嵌入相加得到最终嵌入
        embeddings = inputs_embeds + token_type_embeddings

        # 对最终嵌入进行LayerNorm归一化处理
        embeddings = self.LayerNorm(embeddings)
        # 对归一化后的嵌入进行丢弃处理
        embeddings = self.dropout(embeddings)
        return embeddings


class MCTCTSelfAttention(nn.Module):
    # 初始化函数，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 如果隐藏层大小不是注意力头数的倍数，并且配置对象中没有嵌入大小属性，则引发值错误
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        # 初始化注意力头数和每个注意力头的大小
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.attention_head_dim
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # 初始化查询、键、值的线性层，并且不使用偏置
        self.query = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
        self.key = nn.Linear(config.hidden_size, self.all_head_size, bias=False)
        self.value = nn.Linear(config.hidden_size, self.all_head_size, bias=False)

        # 初始化丢弃层
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        # 初始化最大位置嵌入的大小和距离嵌入
        self.max_position_embeddings = config.max_position_embeddings
        self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        # 初始化是否为解码器的标志
        self.is_decoder = config.is_decoder

    # 将输入张量重塑为注意力分数张量的形状
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    # 将输入张量按照Fortran顺序重塑为指定形状
    def reshape_fortran(self, x, shape):
        if len(x.shape) > 0:
            x = x.permute(*reversed(range(len(x.shape))))
        return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))

    # 相对位置嵌入旋转操作
    def relative_position_embedding_rotate(self, scores):
        # 将张量的维度重新排列
        scores = scores.permute(0, 2, 3, 1)  # 例如 [10, 1839, 14, 4]

        # 获取张量的形状信息
        batch, hidden_state, seq_len, heads = scores.shape

        # 在第二个维度上拼接一个全零张量
        scores = torch.cat((scores, torch.zeros((batch, seq_len, seq_len, heads), device=scores.device)), dim=1)

        # 使用Fortran顺序重塑张量形状
        scores = self.reshape_fortran(scores, [batch, (hidden_state + seq_len) * seq_len, 1, heads])

        # 截取指定部分的张量
        scores = scores[:, : (seq_len + hidden_state - 1) * seq_len]

        # 使用Fortran顺序重塑张量形状
        scores = self.reshape_fortran(scores, [batch, hidden_state + seq_len - 1, seq_len, heads])

        # 获取隐藏状态的一半位置
        halfpoint = hidden_state // 2
        # 截取指定部分的张量并进行转置
        scores = scores[:, halfpoint : halfpoint + seq_len].transpose(1, 2)  # 例如 [10, 14, 14, 4]

        # 返回结果张量并进行维度重新排列
        return scores.permute(0, 3, 1, 2)

    # 前向传播函数
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        ):
        # 使用查询隐藏状态生成混合查询层
        mixed_query_layer = self.query(hidden_states)
        # 对混合查询层进行缩放处理
        mixed_query_layer = mixed_query_layer / math.sqrt(self.attention_head_size)

        # 通过Key层生成Key矩阵
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        # 通过Value层生成Value矩阵
        value_layer = self.transpose_for_scores(self.value(hidden_states))

        # 调整查询层的维度使其适用于计算注意力得分
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # 计算查询层与Key层的点积，得到原始注意力得分
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        # 相对位置嵌入
        positional_embedding = self.distance_embedding.weight
        relative_position_scores = torch.einsum("lh, bche -> bcle", positional_embedding, query_layer.transpose(2, 3))

        # 旋转相对位置得分
        relative_position_scores = self.relative_position_embedding_rotate(relative_position_scores)
        attention_scores = attention_scores + relative_position_scores

        if attention_mask is not None:
            # 将预先计算的注意力屏蔽应用于注意力得分
            attention_scores = attention_scores + attention_mask

        # 对注意力得分进行归一化处理得到注意力概率
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # 通过随机失活层增加模型的鲁棒性
        attention_probs = self.dropout(attention_probs)

        # 如果需要，屏蔽头部
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        # 将注意力概率与Value矩阵相乘得到上下文向量
        context_layer = torch.matmul(attention_probs, value_layer)

        # 对上下文向量进行维度调整
        context_layer = context_layer.permute(0, 2, 1, 3).flatten(start_dim=-2)

        # 返回输出结果，可能包含注意力权重
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        return outputs
# 定义了一个继承自nn.Module的MCTCTLayerNorm类
class MCTCTLayerNorm(nn.Module):
    def __init__(self):
        super().__init__()
        self.singleton_weight = nn.Parameter(torch.ones(1))  # 定义了一个可训练的参数，用于对hidden_states进行加权
        self.singleton_bias = nn.Parameter(torch.zeros(1))  # 定义了一个可训练的参数，用于对hidden_states进行偏置

    # 定义了前向传播函数
    def forward(self, hidden_states):
        return (hidden_states * self.singleton_weight) + self.singleton_bias


# 定义了一个继承自nn.Module的MCTCTSelfOutput类
class MCTCTSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(config.hidden_size, config.hidden_size, bias=False)  # 定义了一个全连接层，用于对hidden_states进行线性变换
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)  # 定义了一个LayerNorm层，用于对hidden_states进行归一化
        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # 定义了一个Dropout层，用于对hidden_states进行随机失活

    # 定义了前向传播函数
    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)  # 将hidden_states经过全连接层
        hidden_states = self.dropout(hidden_states)  # 对hidden_states进行随机失活
        hidden_states = self.LayerNorm(hidden_states + input_tensor)  # 将hidden_states与input_tensor相加，并进行LayerNorm处理
        return hidden_states


# 定义了一个继承自nn.Module的MCTCTAttention类
class MCTCTAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = MCTCTSelfAttention(config)  # 定义了一个MCTCTSelfAttention层
        self.output = MCTCTSelfOutput(config)  # 定义了一个MCTCTSelfOutput层
        self.pruned_heads = set()  # 定义了一个空集合，用于存储被剪枝的注意力头

    # 剪枝注意力头
    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # 剪枝线性层
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # 更新超参数并存储被剪枝的注意力头
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    # 定义了前向传播函数
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)  # 将self_outputs[0]和hidden_states输入到output层
        outputs = (attention_output,) + self_outputs[1:]  # 将attention_output和self_outputs的其他输出组合成一个tuple

        return outputs


# 定义了一个继承自nn.Module的MCTCTIntermediate类
class MCTCTIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)  # 定义了一个全连接层，用于对hidden_states进行线性变换
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]  # 将config.hidden_act映射到激活函数
        else:
            self.intermediate_act_fn = config.hidden_act
    # 前向传播函数，接收隐藏状态作为输入
    def forward(self, hidden_states):
        # 将隐藏状态传入全连接层
        hidden_states = self.dense(hidden_states)
        # 使用激活函数处理全连接层的输出
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 返回处理后的隐藏状态
        return hidden_states
class MCTCTOutput(nn.Module):
    def __init__(self, config):
        # 初始化函数，定义输出层的线性变换、Layer Norm和Dropout操作
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)  # 定义线性变换层
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)  # 定义Layer Norm层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)  # 定义Dropout层

    def forward(self, hidden_states, input_tensor):
        # 前向传播函数，将隐藏状态进行线性变换、Dropout和Layer Norm操作后返回
        hidden_states = self.dense(hidden_states)  # 线性变换
        hidden_states = self.dropout(hidden_states)  # Dropout
        hidden_states = self.LayerNorm(hidden_states + input_tensor)  # 添加输入张量并进行Layer Norm
        return hidden_states  # 返回处理后的隐藏状态


class MCTCTLayer(nn.Module):
    def __init__(self, config: MCTCTConfig):
        # 初始化函数，设置模型参数和子模块
        super().__init__()
        # 设定序列长度的维度和前馈传播的分块大小
        self.seq_len_dim = 1
        self.chunk_size_feed_forward = config.chunk_size_feed_forward

        # 创建中间层、注意力层、是否为解码器和输出层的实例
        self.intermediate = MCTCTIntermediate(config)  # 中间层
        self.attention = MCTCTAttention(config)  # 注意力层
        self.is_decoder = config.is_decoder  # 是否为解码器
        self.output = MCTCTOutput(config)  # 输出层

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
    ):
        # 前向传播函数，包括注意力操作和输出操作
        self_attention_outputs = self.attention(
            hidden_states, attention_mask, head_mask, output_attentions=output_attentions
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # 添加自注意力权重

        # 将前馈传播应用于分块操作
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )

        outputs = (layer_output,) + outputs  # 返回处理后的输出

        return outputs

    def feed_forward_chunk(self, attention_output):
        # 前馈传播的分块操作，包括中间层和输出层
        intermediate_output = self.intermediate(attention_output)  # 中间层处理
        layer_output = self.output(intermediate_output, attention_output)  # 输出层处理
        return layer_output  # 返回处理后的层


class MCTCTPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = MCTCTConfig  # 配置类
    base_model_prefix = "mctct"  # 基础模型前缀
    main_input_name = "input_features"  # 主要输入名称
    supports_gradient_checkpointing = True  # 支持梯度检查点
    def _init_weights(self, module):
        """初始化权重"""
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            # 对线性层的权重进行正态分布初始化
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                # 如果存在偏置项，则初始化为0
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # 对嵌入层的权重进行正态分布初始化
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                # 如果有padding索引，则将其对应的权重初始化为0
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            # 对层归一化的偏置项初始化为0，权重初始化为1
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, MCTCTLayerNorm):
            # 对自定义的层归一化进行初始化
            module.singleton_weight.data.fill_(1.0)
            module.singleton_bias.data.zero_()
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            # 对线性层和一维卷积层的权重进行正态分布初始化
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                # 如果存在偏置项，则初始化为0
                module.bias.data.zero_()

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor):
        """
        计算卷积层的输出长度
        """
        dilation = 1
        for _, kernel_sz, stride in zip(
            range(self.config.num_conv_layers), self.config.conv_kernel, self.config.conv_stride
        ):
            padding = kernel_sz // 2
            input_lengths = input_lengths + 2 * padding - dilation * (kernel_sz - 1) - 1
            input_lengths = torch.div(input_lengths, stride, rounding_mode="trunc") + 1

        return input_lengths

    def _get_feature_vector_attention_mask(self, feature_vector_length, attention_mask):
        # 生成3D注意力掩码，因为输入特征的形状，如果不是，则转换为2D
        if len(attention_mask.shape) > 2:
            attention_mask = attention_mask[:, :, -1]

        # 计算特征向量的长度
        subsampled_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1))
        bsz = attention_mask.size()[0]
        attention_mask = torch.zeros(
            (bsz, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
        )

        # 确保所有在输出长度索引之前的值都被关注到
        attention_mask[(torch.arange(bsz, device=attention_mask.device), subsampled_lengths - 1)] = 1
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).long()
        return attention_mask
# 模型开始部分的文档字符串，说明这个模型是 PyTorch 的子类，以及其使用方式和参数解释
MCTCT_START_DOCSTRING = r"""
    这是一个 PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) 的子类模型。
    像普通的 PyTorch 模块一样使用它，相关使用和行为请参考 PyTorch 文档。

    参数:
        config ([`MCTCTConfig`]): 模型配置类，包含模型的所有参数。
            使用配置文件初始化不会加载模型关联的权重，只是加载配置。要加载模型权重，请查看 [`~PreTrainedModel.from_pretrained`] 方法。
"""

# 输入文档字符串，描述 `forward` 函数的参数
MCTCT_INPUTS_DOCSTRING = r"""
    参数:
        input_features (`torch.LongTensor`，形状 `({0})`):
            在词汇表中的输入序列标记的索引。

            索引可以使用 [`Wav2Vec2CTCTokenizer`] 获得。请参考 [`PreTrainedTokenizer.encode`] 和
            [`PreTrainedTokenizer.__call__`] 了解详细信息。

            [什么是输入 ID？](../glossary#input-ids)
        attention_mask (`torch.FloatTensor`，形状 `({0})`，*可选*):
            避免对填充标记索引执行注意力操作的掩码。掩码值选在 `[0, 1]`：

            - 1 表示**没有被掩码**的标记，
            - 0 表示**被掩码**的标记。

            [什么是注意力掩码？](../glossary#attention-mask)
        head_mask (`torch.FloatTensor`，形状 `(num_heads,)` 或 `(num_layers, num_heads)`，*可选*):
            用于对自注意力模块的某些头进行屏蔽。掩码值选在 `[0, 1]`：

            - 1 表示该头**没有被掩码**，
            - 0 表示该头**被掩码**。
        output_attentions (`bool`，*可选*):
            是否返回所有注意力层的注意力张量。请查看返回的张量中的 `attentions` 以了解更多细节。
        output_hidden_states (`bool`，*可选*):
            是否返回所有层的隐藏状态。请查看返回的张量中的 `hidden_states` 以了解更多细节。
        return_dict (`bool`，*可选*):
            是否返回 [`~file_utils.ModelOutput`] 而不是普通的元组。
"""

# 定义一个名为 `MCTCTEncoder` 的类，继承自 `MCTCTPreTrainedModel`
class MCTCTEncoder(MCTCTPreTrainedModel):
    # 初始化函数，接收一个 `MCTCTConfig` 配置参数
    def __init__(self, config: MCTCTConfig):
        # 调用父类的初始化方法
        super().__init__(config)
        # 获取模型的隐藏层 dropout 概率
        self.hidden_dropout_prob = config.hidden_dropout_prob

        # 定义一个层规范化模块
        self.layer_norm = MCTCTLayerNorm()
        # 定义一个 1D 卷积模块，用于子采样
        self.conv = MCTCTConv1dSubsampler(config)
        # 创建一个包含多层的模块列表，每一层是 `MCTCTLayer`，数量由 `config.num_hidden_layers` 指定
        self.layers = nn.ModuleList([MCTCTLayer(config) for _ in range(config.num_hidden_layers)])

        # 设置梯度检查点为 False
        self.gradient_checkpointing = False

    # 定义 `forward` 函数，接收多个参数
    def forward(
        # 输入特征张量
        input_features: torch.Tensor,
        # 注意力掩码
        attention_mask: torch.Tensor,
        # 头掩码
        head_mask: torch.Tensor,
        # 是否输出注意力张量，默认值为 False
        output_attentions: bool = False,
        # 是否输出隐藏状态，默认值为 False
        output_hidden_states: bool = False,
        # 是否返回字典形式结果，默认值为 True
        return_dict: bool = True,
        # 装饰器，用于在函数上添加文档字符串
@add_start_docstrings(
    # 创建一个字符串，描述了M-CTC-T模型变压器输出原始隐藏状态而不带有任何特定的顶部头部
    # 引用了MCTCT_START_DOCSTRING
# 声明一个类，继承自MCTCTPreTrainedModel
class MCTCTModel(MCTCTPreTrainedModel):
    # 初始化方法
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 将参数config保存在self.config中
        self.config = config

        # 创建MCTCTEncoder对象并保存在self.encoder中
        self.encoder = MCTCTEncoder(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 定义前向传播方法
    @add_start_docstrings_to_model_forward(MCTCT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="audio",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        # 设置output_attentions的值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 设置output_hidden_states的值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 设置return_dict的值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 如果input_features为None，则引发数值错误
        if input_features is None:
            raise ValueError("You have to specify input_features.")

        # 调用self.encoder的前向传播方法
        encoder_outputs = self.encoder(
            input_features,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 获取序列输出
        sequence_output = encoder_outputs[0]

        # 如果不需要返回字典，则返回元组
        if not return_dict:
            return (sequence_output,) + encoder_outputs[1:]

        # 返回BaseModelOutput对象
        return BaseModelOutput(
            last_hidden_state=sequence_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
        
# 创建一个带有语言建模头部的MCTCT模型，用于连接主义时间分类（CTC）
@add_start_docstrings(
    """MCTCT Model with a `language modeling` head on top for Connectionist Temporal Classification (CTC).""",
    MCTCT_START_DOCSTRING,
)
class MCTCTForCTC(MCTCTPreTrainedModel):
    # 初始化函数，接受配置参数并调用父类的初始化函数
    def __init__(self, config):
        super().__init__(config)

        # 实例化 MCTCTModel 对象
        self.mctct = MCTCTModel(config)

        # 如果配置中未定义词汇表大小，则触发数值异常
        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `MCTCTForCTC.from_pretrained(..., vocab_size=vocab_size)`. "
                "or define `vocab_size` of your model's configuration."
            )
        output_hidden_size = config.hidden_size

        # 初始化线性层，输出大小为隐藏层大小，输入大小为词汇表大小
        self.ctc_head = nn.Linear(output_hidden_size, config.vocab_size)

        # 初始化权重并执行最终处理
        self.post_init()

    # 前向传播函数，处理输入特征并返回输出
    @add_start_docstrings_to_model_forward(MCTCT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_CTC_EXPECTED_OUTPUT,
        expected_loss=_CTC_EXPECTED_LOSS,
    )
    def forward(
        self,
        input_features: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutput]:
        # 参数labels是一个torch.Tensor对象，形状为(batch_size, target_length)，表示用于连接主义时间分类的标签。需要注意的是target_length必须小于等于输出logits的序列长度。标签的取值范围是[-100, 0, ..., config.vocab_size - 1]。所有取值为-100的标签将被忽略，损失只会计算取值在[0, ..., config.vocab_size - 1]的标签。
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """
        # 如果返回值是dict，则返回字典；否则返回self.config.use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 使用mctct模型处理input_features，返回outputs
        outputs = self.mctct(
            input_features,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取hidden_states
        hidden_states = outputs[0]

        # 使用ctc_head模型处理hidden_states，获取logits
        logits = self.ctc_head(hidden_states)

        # 初始化loss为None
        loss = None
        # 如果labels不为空
        if labels is not None:
            # 如果labels的最大值大于等于self.config.vocab_size，则抛出异常
            if labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

            # 从attention_mask中获取loss input_lengths
            attention_mask = (
                attention_mask
                if attention_mask is not None
                else torch.ones(input_features.shape[:-1], dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)
            # 假设填充的标记被赋值为-100，未被关注到
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # ctc_loss不支持fp16
            # 对logits进行log_softmax，将结果的维度调换
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

            # 利用torch的后端库禁用cudnn特性，���算ctc_loss
            with torch.backends.cudnn.flags(enabled=False):
                loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )

        # 如果return_dict为False，返回(logit, output)的元组；否则，返回CausalLMOutput对象
        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )
```