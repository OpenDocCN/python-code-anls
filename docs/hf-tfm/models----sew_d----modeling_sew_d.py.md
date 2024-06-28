# `.\models\sew_d\modeling_sew_d.py`

```
# coding=utf-8
# 版权所有 2021 年 ASAPP 公司和 HuggingFace 公司团队。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）授权;
# 您只能在遵守许可证的情况下使用此文件。
# 您可以在以下网址获取许可证副本：
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则本软件是基于“按现状”分发的，
# 没有任何明示或暗示的担保或条件。
# 有关特定语言权限，请参阅许可证。
""" PyTorch SEW model."""

import math
import warnings
from collections.abc import Sequence
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, LayerNorm

from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_outputs import BaseModelOutput, CausalLMOutput, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import softmax_backward_data
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_sew_d import SEWDConfig


logger = logging.get_logger(__name__)

_HIDDEN_STATES_START_POSITION = 1


# 通用文档字符串
_CONFIG_FOR_DOC = "SEWDConfig"

# 基础文档字符串
_CHECKPOINT_FOR_DOC = "asapp/sew-d-tiny-100k-ft-ls100h"
_EXPECTED_OUTPUT_SHAPE = [1, 292, 384]

# CTC（连续文本识别）文档字符串
_CTC_EXPECTED_OUTPUT = "'MISTER QUILTER IS THE APOSTIL OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL'"
_CTC_EXPECTED_LOSS = 0.21

# 音频分类文档字符串
_SEQ_CLASS_CHECKPOINT = "anton-l/sew-d-mid-400k-ft-keyword-spotting"
_SEQ_CLASS_EXPECTED_OUTPUT = "'_unknown_'"
_SEQ_CLASS_EXPECTED_LOSS = 3.16

SEW_D_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "asapp/sew-d-tiny-100k",
    "asapp/sew-d-small-100k",
    "asapp/sew-d-mid-100k",
    "asapp/sew-d-mid-k127-100k",
    "asapp/sew-d-base-100k",
    "asapp/sew-d-base-plus-100k",
    "asapp/sew-d-mid-400k",
    "asapp/sew-d-mid-k127-400k",
    "asapp/sew-d-base-plus-400k",
    # 查看所有 SEW 模型：https://huggingface.co/models?filter=sew-d
]


# 从 transformers.models.wav2vec2.modeling_wav2vec2._compute_mask_indices 复制而来
def _compute_mask_indices(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    attention_mask: Optional[torch.LongTensor] = None,
    min_masks: int = 0,
) -> np.ndarray:
    """
    计算给定形状的随机掩码间隔。用于实现“SpecAugment: ASR 的简单数据增强方法”。
    https://arxiv.org/abs/1904.08779
    注意，此方法未经优化以在 TPU 上运行，并应作为训练过程中的预处理步骤在 CPU 上运行。
    """
    # 确定掩码的数量
    num_masks = int(round(shape[0] * mask_prob))
    # 确保生成的掩码数不低于指定的最小掩码数
    num_masks = max(num_masks, min_masks)

    # 生成掩码索引
    mask_indices = np.full(shape[0], -1, dtype=np.int64)
    for i in range(num_masks):
        # 随机选择掩码的起始位置
        start = np.random.randint(0, shape[0] - mask_length + 1)
        # 标记起始位置及其后 mask_length - 1 个位置为掩码
        mask_indices[start : start + mask_length] = 1

    return mask_indices
    # 计算给定形状的掩码
    # 参数：
    #   shape: 要计算掩码的形状，应为大小为2的元组，第一个元素是批量大小，第二个元素是要跨越的轴的长度。
    #   mask_prob: 要掩盖的整个轴的百分比（介于0和1之间），将由`mask_length`长度的独立生成的掩码跨度数量计算为`mask_prob*shape[1]/mask_length`。
    #              由于重叠的存在，`mask_prob`是一个上限，实际百分比会较小。
    #   mask_length: 掩码的长度
    #   min_masks: 最小的掩码跨度数
    #   attention_mask: （右填充的）注意力掩码，独立缩短每个批次维度的特征轴。
    """
    batch_size, sequence_length = shape
    
    # 如果掩码长度小于1，抛出值错误异常
    if mask_length < 1:
        raise ValueError("`mask_length` has to be bigger than 0.")
    
    # 如果掩码长度大于序列长度，抛出值错误异常
    if mask_length > sequence_length:
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length}"
            f" and `sequence_length`: {sequence_length}`"
        )
    
    # epsilon用于概率舍入
    epsilon = np.random.rand(1).item()
    
    def compute_num_masked_span(input_length):
        """给定输入长度，计算应掩码的跨度数"""
        num_masked_span = int(mask_prob * input_length / mask_length + epsilon)
        num_masked_span = max(num_masked_span, min_masks)
    
        # 确保掩码跨度数 <= 序列长度
        if num_masked_span * mask_length > sequence_length:
            num_masked_span = sequence_length // mask_length
    
        # 确保掩码跨度数 <= 输入长度 - (掩码长度 - 1)
        if input_length - (mask_length - 1) < num_masked_span:
            num_masked_span = max(input_length - (mask_length - 1), 0)
    
        return num_masked_span
    
    # 计算批量中的掩码跨度数
    input_lengths = (
        attention_mask.sum(-1).detach().tolist()
        if attention_mask is not None
        else [sequence_length for _ in range(batch_size)]
    )
    
    # SpecAugment掩码初始化
    spec_aug_mask = np.zeros((batch_size, sequence_length), dtype=bool)
    spec_aug_mask_idxs = []
    
    # 计算在序列长度内最大的掩码跨度数
    max_num_masked_span = compute_num_masked_span(sequence_length)
    
    # 如果最大掩码跨度数为0，直接返回掩码矩阵
    if max_num_masked_span == 0:
        return spec_aug_mask
    for input_length in input_lengths:
        # 计算当前输入的被遮罩段数
        num_masked_span = compute_num_masked_span(input_length)

        # 随机选择要遮罩的索引位置
        spec_aug_mask_idx = np.random.choice(
            np.arange(input_length - (mask_length - 1)), num_masked_span, replace=False
        )

        # 选择第一个随机索引作为虚拟索引，用于在批处理中填充向量，确保所有批次具有相同的维度
        # 由于概率舍入，选择第一个样本使得向量填充两次
        if len(spec_aug_mask_idx) == 0:
            # 如果 `input_length` 严格小于 `sequence_length`，则只能发生这种情况
            # 最后一个标记必须是填充标记，可以用作虚拟掩码 ID
            dummy_mask_idx = sequence_length - 1
        else:
            dummy_mask_idx = spec_aug_mask_idx[0]

        # 将虚拟掩码索引与随机生成的掩码索引合并
        spec_aug_mask_idx = np.concatenate(
            [spec_aug_mask_idx, np.ones(max_num_masked_span - num_masked_span, dtype=np.int32) * dummy_mask_idx]
        )
        spec_aug_mask_idxs.append(spec_aug_mask_idx)

    # 将列表转换为 numpy 数组
    spec_aug_mask_idxs = np.array(spec_aug_mask_idxs)

    # 将掩码索引扩展为掩码段
    spec_aug_mask_idxs = np.broadcast_to(
        spec_aug_mask_idxs[:, :, None], (batch_size, max_num_masked_span, mask_length)
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs.reshape(batch_size, max_num_masked_span * mask_length)

    # 将起始索引添加偏移量，以创建掩码段
    offsets = np.arange(mask_length)[None, None, :]
    offsets = np.broadcast_to(offsets, (batch_size, max_num_masked_span, mask_length)).reshape(
        batch_size, max_num_masked_span * mask_length
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

    # 确保索引不超过序列长度
    if spec_aug_mask_idxs.max() > sequence_length - 1:
        spec_aug_mask_idxs[spec_aug_mask_idxs > sequence_length - 1] = sequence_length - 1

    # 在指定的索引位置上进行散布，创建掩码
    np.put_along_axis(spec_aug_mask, spec_aug_mask_idxs, 1, -1)

    # 返回最终的掩码结果
    return spec_aug_mask
# Copied from transformers.models.deberta_v2.modeling_deberta_v2.make_log_bucket_position
def make_log_bucket_position(relative_pos, bucket_size, max_position):
    # 计算相对位置的符号
    sign = torch.sign(relative_pos)
    # 计算桶的中间位置
    mid = bucket_size // 2
    # 计算绝对位置
    abs_pos = torch.where(
        (relative_pos < mid) & (relative_pos > -mid),
        torch.tensor(mid - 1).type_as(relative_pos),
        torch.abs(relative_pos),
    )
    # 计算对数位置
    log_pos = (
        torch.ceil(torch.log(abs_pos / mid) / torch.log(torch.tensor((max_position - 1) / mid)) * (mid - 1)) + mid
    )
    # 根据绝对位置是否小于等于桶的中间位置选择最终的位置
    bucket_pos = torch.where(abs_pos <= mid, relative_pos.type_as(log_pos), log_pos * sign)
    return bucket_pos


# Copied from transformers.models.deberta_v2.modeling_deberta_v2.build_relative_position
def build_relative_position(query_size, key_size, bucket_size=-1, max_position=-1, device=None):
    """
    根据查询和键构建相对位置

    假设查询的绝对位置 \\(P_q\\) 范围是 (0, query_size)，键的绝对位置 \\(P_k\\) 范围是 (0, key_size)，
    则查询到键的相对位置为 \\(R_{q \\rightarrow k} = P_q - P_k\\)

    Args:
        query_size (int): 查询的长度
        key_size (int): 键的长度
        bucket_size (int): 位置桶的大小
        max_position (int): 允许的最大绝对位置
        device (`torch.device`): 创建张量所用的设备

    Return:
        `torch.LongTensor`: 形状为 [1, query_size, key_size] 的张量
    """

    # 创建查询 ID 序列和键 ID 序列
    q_ids = torch.arange(0, query_size, device=device)
    k_ids = torch.arange(0, key_size, device=device)
    # 计算相对位置 ID
    rel_pos_ids = q_ids[:, None] - k_ids[None, :]
    # 如果指定了桶的大小和最大绝对位置，则应用对数桶化
    if bucket_size > 0 and max_position > 0:
        rel_pos_ids = make_log_bucket_position(rel_pos_ids, bucket_size, max_position)
    rel_pos_ids = rel_pos_ids.to(torch.long)
    # 限制相对位置的长度，并添加批次维度
    rel_pos_ids = rel_pos_ids[:query_size, :]
    rel_pos_ids = rel_pos_ids.unsqueeze(0)
    return rel_pos_ids


@torch.jit.script
# Copied from transformers.models.deberta.modeling_deberta.c2p_dynamic_expand
def c2p_dynamic_expand(c2p_pos, query_layer, relative_pos):
    return c2p_pos.expand([query_layer.size(0), query_layer.size(1), query_layer.size(2), relative_pos.size(-1)])


@torch.jit.script
# Copied from transformers.models.deberta.modeling_deberta.p2c_dynamic_expand
def p2c_dynamic_expand(c2p_pos, query_layer, key_layer):
    return c2p_pos.expand([query_layer.size(0), query_layer.size(1), key_layer.size(-2), key_layer.size(-2)])


@torch.jit.script
# Copied from transformers.models.deberta.modeling_deberta.pos_dynamic_expand
def pos_dynamic_expand(pos_index, p2c_att, key_layer):
    return pos_index.expand(p2c_att.size()[:2] + (pos_index.size(-2), key_layer.size(-2)))


# Copied from transformers.models.deberta.modeling_deberta.get_mask
def get_mask(input, local_context):
    if not isinstance(local_context, DropoutContext):
        # 如果 local_context 不是 DropoutContext 的实例，则使用传入的 dropout 值，否则返回空的 mask
        dropout = local_context
        mask = None
    # 如果条件不成立，执行以下操作
    else:
        # 从局部上下文中获取 dropout 参数
        dropout = local_context.dropout
        # 将 dropout 参数乘以局部上下文的缩放因子
        dropout *= local_context.scale
        # 如果局部上下文不重用掩码，则 mask 为 None；否则，从局部上下文中获取掩码
        mask = local_context.mask if local_context.reuse_mask else None

    # 如果 dropout 大于 0 并且 mask 为 None，则执行以下操作
    if dropout > 0 and mask is None:
        # 创建一个与 input 张量相同形状的随机掩码，并转换为布尔型
        mask = (1 - torch.empty_like(input).bernoulli_(1 - dropout)).to(torch.bool)

    # 如果局部上下文对象是 DropoutContext 的实例，则执行以下操作
    if isinstance(local_context, DropoutContext):
        # 如果局部上下文的掩码为 None，则将当前掩码赋值给局部上下文的掩码
        if local_context.mask is None:
            local_context.mask = mask

    # 返回计算得到的掩码和 dropout 参数
    return mask, dropout
# 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2NoLayerNormConvLayer复制代码，替换Wav2Vec2为SEWD
class SEWDNoLayerNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        # 如果layer_id大于0，则设置输入卷积维度为config.conv_dim[layer_id - 1]，否则设置为1
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        # 设置输出卷积维度为config.conv_dim[layer_id]
        self.out_conv_dim = config.conv_dim[layer_id]

        # 创建一个一维卷积层
        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],  # 使用config中的卷积核大小
            stride=config.conv_stride[layer_id],       # 使用config中的步幅大小
            bias=config.conv_bias,                     # 使用config中的偏置
        )
        # 设置激活函数为ACT2FN[config.feat_extract_activation]
        self.activation = ACT2FN[config.feat_extract_activation]

    # 定义前向传播函数
    def forward(self, hidden_states):
        # 将输入hidden_states通过卷积层self.conv
        hidden_states = self.conv(hidden_states)
        # 将卷积后的hidden_states应用激活函数self.activation
        hidden_states = self.activation(hidden_states)
        return hidden_states


# 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2LayerNormConvLayer复制代码，替换Wav2Vec2为SEWD
class SEWDLayerNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        # 如果layer_id大于0，则设置输入卷积维度为config.conv_dim[layer_id - 1]，否则设置为1
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        # 设置输出卷积维度为config.conv_dim[layer_id]
        self.out_conv_dim = config.conv_dim[layer_id]

        # 创建一个一维卷积层
        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],  # 使用config中的卷积核大小
            stride=config.conv_stride[layer_id],       # 使用config中的步幅大小
            bias=config.conv_bias,                     # 使用config中的偏置
        )
        # 创建一个LayerNorm层，对输出卷积维度进行归一化
        self.layer_norm = nn.LayerNorm(self.out_conv_dim, elementwise_affine=True)
        # 设置激活函数为ACT2FN[config.feat_extract_activation]
        self.activation = ACT2FN[config.feat_extract_activation]

    # 定义前向传播函数
    def forward(self, hidden_states):
        # 将输入hidden_states通过卷积层self.conv
        hidden_states = self.conv(hidden_states)
        
        # 将hidden_states的维度进行转置，将倒数第二维与倒数第一维交换
        hidden_states = hidden_states.transpose(-2, -1)
        # 将转置后的hidden_states通过LayerNorm层self.layer_norm进行归一化
        hidden_states = self.layer_norm(hidden_states)
        # 再次将hidden_states的维度进行转置，将倒数第二维与倒数第一维交换回来
        hidden_states = hidden_states.transpose(-2, -1)
        
        # 将归一化后的hidden_states应用激活函数self.activation
        hidden_states = self.activation(hidden_states)
        return hidden_states


# 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2GroupNormConvLayer复制代码，替换Wav2Vec2为SEWD
class SEWDGroupNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        # 如果layer_id大于0，则设置输入卷积维度为config.conv_dim[layer_id - 1]，否则设置为1
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        # 设置输出卷积维度为config.conv_dim[layer_id]
        self.out_conv_dim = config.conv_dim[layer_id]

        # 创建一个一维卷积层
        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],  # 使用config中的卷积核大小
            stride=config.conv_stride[layer_id],       # 使用config中的步幅大小
            bias=config.conv_bias,                     # 使用config中的偏置
        )
        # 设置激活函数为ACT2FN[config.feat_extract_activation]
        self.activation = ACT2FN[config.feat_extract_activation]

        # 创建一个GroupNorm层，对输出卷积维度进行分组归一化
        self.layer_norm = nn.GroupNorm(num_groups=self.out_conv_dim, num_channels=self.out_conv_dim, affine=True)

    # 定义前向传播函数
    def forward(self, hidden_states):
        # 将输入hidden_states通过卷积层self.conv
        hidden_states = self.conv(hidden_states)
        # 将卷积后的hidden_states通过GroupNorm层self.layer_norm进行归一化
        hidden_states = self.layer_norm(hidden_states)
        # 将归一化后的hidden_states应用激活函数self.activation
        hidden_states = self.activation(hidden_states)
        return hidden_states
# 从transformers.models.sew.modeling_sew.SEWPositionalConvEmbedding复制而来，修改SEW为SEWD
class SEWDPositionalConvEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 定义一个一维卷积层，用于位置编码的卷积
        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.num_conv_pos_embeddings,
            padding=config.num_conv_pos_embeddings // 2,
            groups=config.num_conv_pos_embedding_groups,
            stride=config.squeeze_factor,
        )

        # 如果启用了deepspeed的zero3功能
        if is_deepspeed_zero3_enabled():
            import deepspeed
            # 使用zero3的gathered parameters将权重进行分布式处理
            with deepspeed.zero.GatheredParameters(self.conv.weight, modifier_rank=0):
                self.conv = nn.utils.weight_norm(self.conv, name="weight", dim=2)
            # 注册卷积层的权重变量给deepspeed.zero
            deepspeed.zero.register_external_parameter(self, self.conv.weight_v)
            deepspeed.zero.register_external_parameter(self, self.conv.weight_g)
        else:
            # 对卷积层的权重进行权重归一化处理
            self.conv = nn.utils.weight_norm(self.conv, name="weight", dim=2)

        # 创建一个用于卷积后padding的层
        self.padding = SEWDSamePadLayer(config.num_conv_pos_embeddings)
        # 激活函数选择，根据配置选择不同的激活函数
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        # 进行一维卷积
        hidden_states = self.conv(hidden_states)
        # 进行padding处理
        hidden_states = self.padding(hidden_states)
        # 使用选择的激活函数进行激活
        hidden_states = self.activation(hidden_states)

        return hidden_states


# 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2SamePadLayer复制而来，修改Wav2Vec2为SEW
class SEWDSamePadLayer(nn.Module):
    def __init__(self, num_conv_pos_embeddings):
        super().__init__()
        # 根据卷积位置编码数目确定是否需要移除的padding数量
        self.num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0

    def forward(self, hidden_states):
        # 如果需要移除padding，则进行裁剪
        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, :, : -self.num_pad_remove]
        return hidden_states


# 从transformers.models.sew.modeling_sew.SEWUpsampling复制而来，修改SEW为SEWD
class SEWDUpsampling(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 定义一个线性层，用于上采样投影
        self.projection = nn.Linear(config.hidden_size, config.hidden_size * config.squeeze_factor)
        # 根据配置选择不同的激活函数
        self.activation = ACT2FN[config.feat_extract_activation]
        # 保存下采样倍数
        self.squeeze_factor = config.squeeze_factor

    def forward(self, hidden_states):
        # 进行线性投影
        hidden_states = self.projection(hidden_states)
        # 使用选择的激活函数进行激活
        hidden_states = self.activation(hidden_states)

        # 如果下采样因子大于1
        if self.squeeze_factor > 1:
            # 将嵌入通道转换为序列长度
            bsz, src_len, src_embed_dim = hidden_states.size()
            tgt_len = src_len * self.squeeze_factor
            tgt_embed_dim = src_embed_dim // self.squeeze_factor
            hidden_states = hidden_states.reshape(bsz, src_len, self.squeeze_factor, tgt_embed_dim)
            hidden_states = hidden_states.reshape(bsz, tgt_len, tgt_embed_dim)

        return hidden_states


# 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2FeatureEncoder复制而来，修改Wav2Vec2为SEWD
class SEWDFeatureEncoder(nn.Module):
    """从原始音频波形构建特征"""

    def __init__(self, config):
        super().__init__()

        # 根据配置选择特征提取的归一化方式
        if config.feat_extract_norm == "group":
            # 如果是group归一化，则创建一系列卷积层
            conv_layers = [SEWDGroupNormConvLayer(config, layer_id=0)] + [
                SEWDNoLayerNormConvLayer(config, layer_id=i + 1) for i in range(config.num_feat_extract_layers - 1)
            ]
        elif config.feat_extract_norm == "layer":
            # 如果是layer归一化，则创建一系列卷积层
            conv_layers = [SEWDLayerNormConvLayer(config, layer_id=i) for i in range(config.num_feat_extract_layers)]
        else:
            # 若配置不匹配则抛出异常
            raise ValueError(
                f"`config.feat_extract_norm` is {config.feat_extract_norm}, but has to be one of ['group', 'layer']"
            )
        
        # 将卷积层列表转换为ModuleList
        self.conv_layers = nn.ModuleList(conv_layers)
        self.gradient_checkpointing = False
        self._requires_grad = True

    def _freeze_parameters(self):
        # 冻结所有参数，使其不需要梯度更新
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def forward(self, input_values):
        # 将输入值添加一个维度，用于处理
        hidden_states = input_values[:, None]

        # 如果需要梯度并且正在训练，确保hidden_states需要梯度
        if self._requires_grad and self.training:
            hidden_states.requires_grad = True

        # 遍历所有卷积层进行前向传播
        for conv_layer in self.conv_layers:
            if self._requires_grad and self.gradient_checkpointing and self.training:
                # 如果开启了梯度检查点功能，使用梯度检查点函数进行前向传播
                hidden_states = self._gradient_checkpointing_func(
                    conv_layer.__call__,
                    hidden_states,
                )
            else:
                # 否则直接通过卷积层进行前向传播
                hidden_states = conv_layer(hidden_states)

        # 返回处理后的隐藏状态
        return hidden_states


class SEWDFeatureExtractor(SEWDFeatureEncoder):
    def __init__(self, config):
        super().__init__(config)
        # 发出警告，表明该类将被弃用并在未来版本中移除，建议使用基类替代
        warnings.warn(
            f"The class `{self.__class__.__name__}` has been depreciated "
            "and will be removed in Transformers v5. "
            f"Use `{self.__class__.__bases__[0].__name__}` instead.",
            FutureWarning,
        )


# 从transformers.models.deberta.modeling_deberta.ContextPooler复制而来
class ContextPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建线性层和稳定的dropout层
        self.dense = nn.Linear(config.pooler_hidden_size, config.pooler_hidden_size)
        self.dropout = StableDropout(config.pooler_dropout)
        self.config = config

    def forward(self, hidden_states):
        # 通过简单地获取第一个token的隐藏状态来“池化”模型
        context_token = hidden_states[:, 0]
        context_token = self.dropout(context_token)
        pooled_output = self.dense(context_token)
        pooled_output = ACT2FN[self.config.pooler_hidden_act](pooled_output)
        # 返回池化后的输出
        return pooled_output

    @property
    def output_dim(self):
        # 返回输出维度大小，与隐藏大小相同
        return self.config.hidden_size


# 从transformers.models.deberta.modeling_deberta.XSoftmax复制而来
class XSoftmax(torch.autograd.Function):
    """
    Masked Softmax which is optimized for saving memory

    Args:
        input (`torch.tensor`): The input tensor that will apply softmax.
        mask (`torch.IntTensor`):
            The mask matrix where 0 indicate that element will be ignored in the softmax calculation.
        dim (int): The dimension that will apply softmax

    Example:

    ```python
    >>> import torch
    >>> from transformers.models.deberta_v2.modeling_deberta_v2 import XSoftmax

    >>> # Make a tensor
    >>> x = torch.randn([4, 20, 100])

    >>> # Create a mask
    >>> mask = (x > 0).int()

    >>> # Specify the dimension to apply softmax
    >>> dim = -1

    >>> y = XSoftmax.apply(x, mask, dim)
    ```"""

    @staticmethod
    def forward(self, input, mask, dim):
        # 设置对象的维度属性
        self.dim = dim
        # 创建反向掩码，将输入掩码转换为布尔类型取反
        rmask = ~(mask.to(torch.bool))

        # 用最小的浮点数填充输入中的掩码位置
        output = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
        # 在指定维度上应用 softmax 函数
        output = torch.softmax(output, self.dim)
        # 将输出中掩码位置重新填充为0
        output.masked_fill_(rmask, 0)
        # 保存输出作为反向传播的一部分
        self.save_for_backward(output)
        return output

    @staticmethod
    def backward(self, grad_output):
        # 获取保存的输出张量
        (output,) = self.saved_tensors
        # 调用自定义的 softmax 反向传播函数计算输入梯度
        inputGrad = softmax_backward_data(self, grad_output, output, self.dim, output)
        return inputGrad, None, None

    @staticmethod
    def symbolic(g, self, mask, dim):
        import torch.onnx.symbolic_helper as sym_help
        from torch.onnx.symbolic_opset9 import masked_fill, softmax

        # 将掩码转换为长整型
        mask_cast_value = g.op("Cast", mask, to_i=sym_help.cast_pytorch_to_onnx["Long"])
        # 计算反向掩码，使用 ONNX 运算符
        r_mask = g.op(
            "Cast",
            g.op("Sub", g.op("Constant", value_t=torch.tensor(1, dtype=torch.int64)), mask_cast_value),
            to_i=sym_help.cast_pytorch_to_onnx["Bool"],
        )
        # 使用 ONNX 运算符对输入进行掩码填充
        output = masked_fill(
            g, self, r_mask, g.op("Constant", value_t=torch.tensor(torch.finfo(self.type().dtype()).min))
        )
        # 使用 ONNX 运算符在指定维度上应用 softmax
        output = softmax(g, output, dim)
        return masked_fill(g, output, r_mask, g.op("Constant", value_t=torch.tensor(0, dtype=torch.bool)))


# Copied from transformers.models.deberta.modeling_deberta.DropoutContext
class DropoutContext(object):
    def __init__(self):
        self.dropout = 0
        self.mask = None
        self.scale = 1
        self.reuse_mask = True


# Copied from transformers.models.deberta.modeling_deberta.XDropout
class XDropout(torch.autograd.Function):
    """Optimized dropout function to save computation and memory by using mask operation instead of multiplication."""

    @staticmethod
    def forward(ctx, input, local_ctx):
        # 调用函数获取掩码和 dropout 概率
        mask, dropout = get_mask(input, local_ctx)
        # 计算缩放比例
        ctx.scale = 1.0 / (1 - dropout)
        if dropout > 0:
            # 保存掩码用于反向传播
            ctx.save_for_backward(mask)
            # 应用掩码并乘以缩放比例
            return input.masked_fill(mask, 0) * ctx.scale
        else:
            return input
    # 定义静态方法，用于在反向传播时计算梯度
    def backward(ctx, grad_output):
        # 如果上下文中的缩放值大于1，则执行以下操作
        if ctx.scale > 1:
            # 从上下文保存的张量中获取掩码
            (mask,) = ctx.saved_tensors
            # 将梯度张量中的被掩码位置清零，并乘以缩放因子，然后返回
            return grad_output.masked_fill(mask, 0) * ctx.scale, None
        else:
            # 如果缩放值不大于1，则直接返回梯度和空值
            return grad_output, None

    # 定义静态方法，用于在符号图中生成 Dropout 操作
    def symbolic(g: torch._C.Graph, input: torch._C.Value, local_ctx: Union[float, DropoutContext]) -> torch._C.Value:
        # 导入符号化操作集12
        from torch.onnx import symbolic_opset12

        # 如果 local_ctx 是 DropoutContext 类型，则获取其中的 dropout 率
        dropout_p = local_ctx
        if isinstance(local_ctx, DropoutContext):
            dropout_p = local_ctx.dropout
        
        # 在导出过程中，稳定的 Dropout 只在训练时调用此函数
        train = True
        
        # TODO: 我们应该检查 opset_version 是否大于12，但目前没有很好的方法来执行此检查。
        # 如今，如果 opset_version < 12，导出将会因为 CheckerError 而失败。
        # 一旦 https://github.com/pytorch/pytorch/issues/78391 问题得到解决，可以像下面这样处理：
        # if opset_version < 12:
        #   return torch.onnx.symbolic_opset9.dropout(g, input, dropout_p, train)
        
        # 使用符号化操作集12中的 dropout 函数生成符号化节点
        return symbolic_opset12.dropout(g, input, dropout_p, train)
# Copied from transformers.models.deberta.modeling_deberta.StableDropout
class StableDropout(nn.Module):
    """
    Optimized dropout module for stabilizing the training

    Args:
        drop_prob (float): the dropout probabilities
    """

    def __init__(self, drop_prob):
        super().__init__()
        # 初始化稳定的dropout模块
        self.drop_prob = drop_prob  # 设置dropout概率
        self.count = 0  # 上下文堆栈计数
        self.context_stack = None  # 上下文堆栈初始化为空

    def forward(self, x):
        """
        Call the module

        Args:
            x (`torch.tensor`): The input tensor to apply dropout
        """
        if self.training and self.drop_prob > 0:
            return XDropout.apply(x, self.get_context())  # 如果处于训练状态且dropout概率大于0，则应用自定义的dropout操作
        return x  # 否则直接返回输入

    def clear_context(self):
        # 清空上下文堆栈
        self.count = 0
        self.context_stack = None

    def init_context(self, reuse_mask=True, scale=1):
        if self.context_stack is None:
            self.context_stack = []  # 如果上下文堆栈为空，则初始化为空列表
        self.count = 0  # 计数器归零
        for c in self.context_stack:
            c.reuse_mask = reuse_mask  # 设置重用掩码标志
            c.scale = scale  # 设置比例

    def get_context(self):
        if self.context_stack is not None:
            if self.count >= len(self.context_stack):
                self.context_stack.append(DropoutContext())  # 如果计数超过堆栈长度，则添加新的dropout上下文
            ctx = self.context_stack[self.count]  # 获取当前计数对应的dropout上下文
            ctx.dropout = self.drop_prob  # 设置dropout概率
            self.count += 1  # 计数器加一
            return ctx  # 返回dropout上下文
        else:
            return self.drop_prob  # 如果上下文堆栈为空，则返回dropout概率本身


# Copied from transformers.models.deberta.modeling_deberta.DebertaSelfOutput with DebertaV2->SEWD, DebertaLayerNorm->LayerNorm, hidden_dropout_prob->activation_dropout
class SEWDSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)  # 使用线性层变换隐藏状态
        self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps)  # 应用LayerNorm进行归一化
        self.dropout = StableDropout(config.activation_dropout)  # 使用稳定的dropout模块

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)  # 线性变换隐藏状态
        hidden_states = self.dropout(hidden_states)  # 应用稳定的dropout
        hidden_states = self.LayerNorm(hidden_states + input_tensor)  # 使用LayerNorm对变换后的隐藏状态进行归一化
        return hidden_states  # 返回处理后的隐藏状态


# Copied from transformers.models.deberta_v2.modeling_deberta_v2.DisentangledSelfAttention with attention_probs_dropout_prob->attention_dropout, hidden_dropout_prob->activation_dropout
class DisentangledSelfAttention(nn.Module):
    """
    Disentangled self-attention module

    Parameters:
        config (`DebertaV2Config`):
            A model config class instance with the configuration to build a new model. The schema is similar to
            *BertConfig*, for more details, please refer [`DebertaV2Config`]

    """
    # 定义类和其中的初始化方法，包含Transformer注意力机制相关参数和组件
    def __init__(self, config):
    
        # 调用基类初始化方法，默认调用具有模型特定特征的方法
        super().__init__()
    
        # 验证隐藏维度是否是 attention head 的倍数，否则会抛出错误
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
    
        # 初始化注意力头数量
        self.num_attention_heads = config.num_attention_heads
    
        _attention_head_size = config.hidden_size // config.num_attention_heads # 默认计算每个头的大小
        # 根据配置中self.attention_head_size的设置进行可能的调整
        self.attention_head_size = getattr(config, "attention_head_size", _attention_head_size)
        # 计算头数乘以每个头的大小，用于计算总头大小
        self.all_head_size = self.num_attention_heads * self.attention_head_size
            
    
        # 创建线性投影层以将输入映射到所需的输出维度
        self.query_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
        self.key_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
        self.value_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
    
        # 检查是否共享注意力键
        self.share_att_key = getattr(config, "share_att_key", False)
        # 设置注意力类型的参数列表
        self.pos_att_type = config.pos_att_type if config.pos_att_type is not None else []
        # 检查是否使用相对注意力机制
        self.relative_attention = getattr(config, "relative_attention", False)
    
        # 使用相对注意力时，将 position_buckets 和 max_relative_positions 等参数的默认值设定
        if self.relative_attention:
            self.position_buckets = getattr(config, "position_buckets", -1)
            self.max_relative_positions = getattr(config, "max_relative_positions", -1)
            # 设置 max_relative_positions 初始值为 max_position_embeddings，除非使用 position_buckets 或者其小于 1
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings
            # 计算实际的相对位置嵌入大小
            self.pos_ebd_size = self.max_relative_positions
            # 如果 position_buckets 参数已配置，调整 pos_ebd_size 大小
            if self.position_buckets > 0:
                self.pos_ebd_size = self.position_buckets
    
            # 初始化位置 dropout 层
            self.pos_dropout = StableDropout(config.activation_dropout)
    
            # 如果不共享attention键，则创建额外的线性投影层用于处理位置相关的输入
            if "c2p" in self.pos_att_type:
                self.pos_key_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
            if "p2c" in self.pos_att_type:
                self.pos_query_proj = nn.Linear(config.hidden_size, self.all_head_size)
    
        # 初始化模型的下垂dropout层
        self.dropout = StableDropout(config.attention_dropout)
    
    # 随后定义了 batch 处理数据的内部步骤 x 转换函数
    def transpose_for_scores(self, x, attention_heads):
        # 获取数据和头数维度形状
        new_x_shape = x.size()[:-1] + (attention_heads, -1)
        # 重塑数据的形状以准备在循环过程中使用
        x = x.view(new_x_shape)
        # 转置以将数据按注意力头划分
        return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))
    
    # 随后定义了前向传播方法，处理输入数据
    def forward(
        self,
        hidden_states,
        attention_mask,
        output_attentions=False,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
# Copied from transformers.models.deberta.modeling_deberta.DebertaAttention with Deberta->SEWD
class SEWDAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化自注意力层，使用SEWD版本的DisentangledSelfAttention
        self.self = DisentangledSelfAttention(config)
        # 初始化自注意力层输出层，使用SEWD版本的SEWDSelfOutput
        self.output = SEWDSelfOutput(config)
        self.config = config

    def forward(
        self,
        hidden_states,
        attention_mask,
        output_attentions=False,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
    ):
        # 执行自注意力计算，调用SEWD版本的DisentangledSelfAttention模型
        self_output = self.self(
            hidden_states,
            attention_mask,
            output_attentions,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
        )
        if output_attentions:
            self_output, att_matrix = self_output
        if query_states is None:
            query_states = hidden_states
        # 执行自注意力输出层计算，调用SEWD版本的SEWDSelfOutput模型
        attention_output = self.output(self_output, query_states)

        if output_attentions:
            return (attention_output, att_matrix)  # 返回注意力输出和注意力矩阵（如果有的话）
        else:
            return attention_output  # 返回注意力输出结果


# Copied from transformers.models.bert.modeling_bert.BertIntermediate with Bert->SEWD
class SEWDIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 使用线性层将隐藏状态的大小转换为中间状态大小
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 根据配置选择激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 使用线性层进行转换
        hidden_states = self.dense(hidden_states)
        # 应用中间激活函数
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states  # 返回转换后的中间状态


# Copied from transformers.models.deberta.modeling_deberta.DebertaOutput with DebertaLayerNorm->LayerNorm, hidden_dropout_prob->activation_dropout
class SEWDOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 使用线性层将中间状态大小转换为隐藏状态大小
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 使用SEWD版本的LayerNorm，初始化LayerNorm层
        self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps)
        # 使用SEWD版本的StableDropout，初始化稳定Dropout层
        self.dropout = StableDropout(config.activation_dropout)
        self.config = config

    def forward(self, hidden_states, input_tensor):
        # 使用线性层进行转换
        hidden_states = self.dense(hidden_states)
        # 应用稳定Dropout
        hidden_states = self.dropout(hidden_states)
        # 应用LayerNorm
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states  # 返回处理后的隐藏状态


# Copied from transformers.models.deberta.modeling_deberta.DebertaLayer with Deberta->SEWD
class SEWDLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化SEWD版本的注意力层、中间层和输出层
        self.attention = SEWDAttention(config)
        self.intermediate = SEWDIntermediate(config)
        self.output = SEWDOutput(config)
    # 定义神经网络模型中的前向传播函数，用于计算每个层的输出
    def forward(
        self,
        hidden_states,             # 输入的隐藏状态，通常是模型中前一层的输出
        attention_mask,            # 注意力掩码，指定哪些位置需要进行注意力计算
        query_states=None,         # 查询状态，用于多头注意力机制中的查询
        relative_pos=None,         # 相对位置编码，用于自注意力机制中的位置编码
        rel_embeddings=None,       # 相对位置嵌入，用于计算相对位置偏移
        output_attentions=False,   # 是否输出注意力矩阵
    ):
        # 调用注意力层计算注意力输出
        attention_output = self.attention(
            hidden_states,          # 输入的隐藏状态
            attention_mask,         # 注意力掩码
            output_attentions=output_attentions,  # 是否输出注意力矩阵的标志
            query_states=query_states,            # 查询状态
            relative_pos=relative_pos,            # 相对位置编码
            rel_embeddings=rel_embeddings,        # 相对位置嵌入
        )
        # 如果需要输出注意力矩阵，则解包注意力输出
        if output_attentions:
            attention_output, att_matrix = attention_output
        # 将注意力输出传入中间层进行处理
        intermediate_output = self.intermediate(attention_output)
        # 将中间层的输出传入输出层，生成最终层的输出
        layer_output = self.output(intermediate_output, attention_output)
        # 如果需要输出注意力矩阵，则返回输出层的输出和注意力矩阵
        if output_attentions:
            return (layer_output, att_matrix)
        else:
            # 否则，仅返回输出层的输出
            return layer_output
# Copied from transformers.models.deberta_v2.modeling_deberta_v2.ConvLayer
# 定义一个名为 ConvLayer 的类，继承自 nn.Module
class ConvLayer(nn.Module):
    # 初始化方法，接受一个 config 对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 从 config 中获取卷积核大小，默认为 3
        kernel_size = getattr(config, "conv_kernel_size", 3)
        # 从 config 中获取卷积的分组数，默认为 1
        groups = getattr(config, "conv_groups", 1)
        # 从 config 中获取卷积激活函数，默认为 "tanh"
        self.conv_act = getattr(config, "conv_act", "tanh")
        # 创建一个 1 维卷积层，输入和输出通道数都为 config.hidden_size，卷积核大小为 kernel_size
        # padding 设置为 (kernel_size - 1) // 2 保证卷积后维度不变
        # groups 参数控制分组卷积
        self.conv = nn.Conv1d(
            config.hidden_size, config.hidden_size, kernel_size, padding=(kernel_size - 1) // 2, groups=groups
        )
        # 创建一个 LayerNorm 层，输入维度为 config.hidden_size，eps 参数为 config.layer_norm_eps
        self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps)
        # 创建一个稳定 Dropout 层，dropout 概率为 config.hidden_dropout_prob
        self.dropout = StableDropout(config.hidden_dropout_prob)
        # 将 config 对象保存到当前对象的 config 属性中
        self.config = config

    # 前向传播方法，接受 hidden_states、residual_states 和 input_mask 作为输入
    def forward(self, hidden_states, residual_states, input_mask):
        # 对 hidden_states 进行维度变换，将第二维和第三维交换，然后做卷积操作
        out = self.conv(hidden_states.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        # 创建一个逻辑张量 rmask，标识 input_mask 为 0 的位置
        rmask = (1 - input_mask).bool()
        # 将 out 张量中 rmask 为 True 的位置置为 0
        out.masked_fill_(rmask.unsqueeze(-1).expand(out.size()), 0)
        # 对 out 张量应用指定的激活函数 ACT2FN[self.conv_act]，然后加上 dropout 处理
        out = ACT2FN[self.conv_act](self.dropout(out))

        # 计算 layer_norm_input，即 residual_states 和 out 的和
        layer_norm_input = residual_states + out
        # 对 layer_norm_input 应用 LayerNorm 层，然后赋值给 output
        output = self.LayerNorm(layer_norm_input).to(layer_norm_input)

        # 如果 input_mask 为 None，则直接将 output 赋值给 output_states
        if input_mask is None:
            output_states = output
        else:
            # 如果 input_mask 的维度与 layer_norm_input 的维度不同，进行维度调整
            if input_mask.dim() != layer_norm_input.dim():
                if input_mask.dim() == 4:
                    input_mask = input_mask.squeeze(1).squeeze(1)
                input_mask = input_mask.unsqueeze(2)

            # 将 input_mask 转换为与 output 相同的数据类型，并与 output 相乘，得到 output_states
            input_mask = input_mask.to(output.dtype)
            output_states = output * input_mask

        # 返回 output_states
        return output_states


# Copied from transformers.models.deberta_v2.modeling_deberta_v2.DebertaV2Encoder with DebertaV2->SEWD
# 定义一个名为 SEWDTransformerEncoder 的类，继承自 nn.Module
class SEWDTransformerEncoder(nn.Module):
    """Modified BertEncoder with relative position bias support"""
    # 初始化方法，接受一个 config 对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()

        # 创建一个包含多个 SEWDLayer 的 ModuleList，层数为 config.num_hidden_layers
        self.layer = nn.ModuleList([SEWDLayer(config) for _ in range(config.num_hidden_layers)])
        # 从 config 中获取是否支持相对位置偏置的标志，默认为 False
        self.relative_attention = getattr(config, "relative_attention", False)

        # 如果支持相对位置偏置
        if self.relative_attention:
            # 从 config 中获取最大相对位置的范围，默认为 -1
            self.max_relative_positions = getattr(config, "max_relative_positions", -1)
            # 如果最大相对位置小于 1，则设置为 config.max_position_embeddings
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings

            # 从 config 中获取位置桶的数量，默认为 -1
            self.position_buckets = getattr(config, "position_buckets", -1)
            # 计算位置嵌入的尺寸
            pos_ebd_size = self.max_relative_positions * 2

            # 如果指定了位置桶的数量，则重新计算位置嵌入的尺寸
            if self.position_buckets > 0:
                pos_ebd_size = self.position_buckets * 2

            # 创建一个 nn.Embedding 层用于存储相对位置嵌入
            self.rel_embeddings = nn.Embedding(pos_ebd_size, config.hidden_size)

        # 从 config 中获取并解析 norm_rel_ebd 字符串，设置是否使用 LayerNorm 进行相对位置嵌入的归一化
        self.norm_rel_ebd = [x.strip() for x in getattr(config, "norm_rel_ebd", "none").lower().split("|")]

        # 如果设置了 "layer_norm"，则创建一个 LayerNorm 层，用于相对位置嵌入的归一化
        if "layer_norm" in self.norm_rel_ebd:
            self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=True)

        # 如果 config 中指定了卷积核大小大于 0，则创建一个 ConvLayer
        self.conv = ConvLayer(config) if getattr(config, "conv_kernel_size", 0) > 0 else None
        # 默认关闭梯度检查点
        self.gradient_checkpointing = False
    # 返回相对位置嵌入（如果启用相对注意力机制），否则返回空值
    def get_rel_embedding(self):
        rel_embeddings = self.rel_embeddings.weight if self.relative_attention else None
        # 如果相对位置嵌入不为空，并且规范化名称包含"layer_norm"
        if rel_embeddings is not None and ("layer_norm" in self.norm_rel_ebd):
            # 对相对位置嵌入进行层标准化处理
            rel_embeddings = self.LayerNorm(rel_embeddings)
        return rel_embeddings

    # 获取注意力遮罩，根据不同维度扩展遮罩的尺寸
    def get_attention_mask(self, attention_mask):
        if attention_mask.dim() <= 2:
            # 在维度1和2上扩展注意力遮罩的尺寸
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # 生成扩展后的注意力遮罩
            attention_mask = extended_attention_mask * extended_attention_mask.squeeze(-2).unsqueeze(-1)
        elif attention_mask.dim() == 3:
            # 如果遮罩是3维的，则在维度1上进行扩展
            attention_mask = attention_mask.unsqueeze(1)
        return attention_mask

    # 获取相对位置编码，如果启用相对注意力机制且相对位置未提供，则构建相对位置编码
    def get_rel_pos(self, hidden_states, query_states=None, relative_pos=None):
        if self.relative_attention and relative_pos is None:
            # 如果启用相对注意力机制且未提供相对位置，则根据输入的大小构建相对位置编码
            q = query_states.size(-2) if query_states is not None else hidden_states.size(-2)
            relative_pos = build_relative_position(
                q,
                hidden_states.size(-2),
                bucket_size=self.position_buckets,
                max_position=self.max_relative_positions,
                device=hidden_states.device,
            )
        return relative_pos

    # 前向传播函数，接收输入的隐藏状态和注意力遮罩等参数，并返回模型的输出
    def forward(
        self,
        hidden_states,
        attention_mask,
        output_hidden_states=True,
        output_attentions=False,
        query_states=None,
        relative_pos=None,
        return_dict=True,
        ):
            # 如果注意力掩码的维度小于等于2，直接使用作为输入掩码
            if attention_mask.dim() <= 2:
                input_mask = attention_mask
            else:
                # 否则，将注意力掩码在倒数第二个维度上求和，并检查大于0的部分作为输入掩码
                input_mask = attention_mask.sum(-2) > 0
            # 获取注意力掩码，根据模型定义的方法
            attention_mask = self.get_attention_mask(attention_mask)
            # 获取相对位置编码，用于当前层的注意力计算
            relative_pos = self.get_rel_pos(hidden_states, query_states, relative_pos)

            # 初始化用于存储所有隐藏状态和注意力权重的变量，根据输出设置决定是否需要存储
            all_hidden_states = () if output_hidden_states else None
            all_attentions = () if output_attentions else None

            # 如果隐藏状态是一个序列，取第一个作为下一步的键值对
            if isinstance(hidden_states, Sequence):
                next_kv = hidden_states[0]
            else:
                next_kv = hidden_states
            # 获取相对位置编码矩阵
            rel_embeddings = self.get_rel_embedding()
            # 初始化输出状态为当前的键值对
            output_states = next_kv
            # 遍历每一层的神经网络模块
            for i, layer_module in enumerate(self.layer):
                # 如果需要输出隐藏状态，则将当前状态加入到所有隐藏状态中
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (output_states,)

                # 如果开启了梯度检查点且正在训练阶段，使用梯度检查点函数计算当前层输出状态
                if self.gradient_checkpointing and self.training:
                    output_states = self._gradient_checkpointing_func(
                        layer_module.__call__,
                        next_kv,
                        attention_mask,
                        query_states,
                        relative_pos,
                        rel_embeddings,
                        output_attentions,
                    )
                else:
                    # 否则，正常调用当前层的前向传播函数
                    output_states = layer_module(
                        next_kv,
                        attention_mask,
                        query_states=query_states,
                        relative_pos=relative_pos,
                        rel_embeddings=rel_embeddings,
                        output_attentions=output_attentions,
                    )

                # 如果需要输出注意力权重，从输出状态中提取注意力权重
                if output_attentions:
                    output_states, att_m = output_states

                # 如果是第一层且存在卷积模块，将当前隐藏状态与输入掩码传递给卷积模块
                if i == 0 and self.conv is not None:
                    output_states = self.conv(hidden_states, output_states, input_mask)

                # 如果有查询状态，更新为当前输出状态，并更新下一步的键值对
                if query_states is not None:
                    query_states = output_states
                    if isinstance(hidden_states, Sequence):
                        next_kv = hidden_states[i + 1] if i + 1 < len(self.layer) else None
                else:
                    next_kv = output_states

                # 如果需要输出注意力权重，将当前层计算得到的注意力权重加入到所有注意力中
                if output_attentions:
                    all_attentions = all_attentions + (att_m,)

            # 如果需要输出隐藏状态，将最后一层的输出状态加入到所有隐藏状态中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (output_states,)

            # 如果不需要以字典形式返回结果，则返回元组，过滤掉值为None的项
            if not return_dict:
                return tuple(v for v in [output_states, all_hidden_states, all_attentions] if v is not None)
            # 否则，以BaseModelOutput形式返回结果，包括最后隐藏状态、所有隐藏状态和所有注意力权重
            return BaseModelOutput(
                last_hidden_state=output_states, hidden_states=all_hidden_states, attentions=all_attentions
            )
# 定义 SEWDEncoder 类，继承自 nn.Module，用于实现一个自定义的编码器模型
class SEWDEncoder(nn.Module):
    # 初始化方法，接受一个 config 参数
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 初始化位置卷积嵌入层对象
        self.pos_conv_embed = SEWDPositionalConvEmbedding(config)
        # 初始化一维平均池化层
        self.pool = nn.AvgPool1d(config.squeeze_factor, config.squeeze_factor)
        # 初始化 SEWDTransformerEncoder 编码器
        self.encoder = SEWDTransformerEncoder(config)
        # 初始化 SEWDUpsampling 上采样层
        self.upsample = SEWDUpsampling(config)
        # 梯度检查点设置为 False
        self.gradient_checkpointing = False

    # 前向传播方法，接受多个参数
    def forward(
        self,
        hidden_states: torch.tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
        # 计算最大编码器长度
        max_encoder_length = hidden_states.shape[1] // self.config.squeeze_factor
        # 如果没有给定注意力掩码，则创建一个全为 1 的张量作为默认注意力掩码
        if attention_mask is None:
            attention_mask = torch.ones(
                (hidden_states.shape[0], max_encoder_length), dtype=torch.long, device=hidden_states.device
            )
        else:
            # 将注意力掩码为 False 的位置对应的隐藏状态设为 0
            hidden_states[~attention_mask.bool()] = 0.0
            # 计算输入长度并应用池化公式以获取真实的输出长度
            input_lengths = (attention_mask.long()).sum(-1)
            output_lengths = input_lengths // self.config.squeeze_factor
            # 生成注意力掩码，限制注意力范围在有效输出长度内
            attention_ids = (
                torch.arange(0, max_encoder_length, device=output_lengths.device)
                .view(1, -1)
                .expand(output_lengths.shape[0], -1)
            )
            attention_mask = (attention_ids < output_lengths.view(-1, 1)).long()

        # 记录输入时间步数
        n_input_timesteps = hidden_states.shape[1]

        # 将隐藏状态维度转置，以适应位置嵌入计算
        hidden_states = hidden_states.transpose(1, 2)
        # 计算位置嵌入
        position_embeddings = self.pos_conv_embed(hidden_states)
        # 对隐藏状态进行池化操作
        pooled_hidden_states = self.pool(hidden_states)
        # 选择较小的长度作为最终的隐藏状态长度
        min_length = min(position_embeddings.size(-1), pooled_hidden_states.size(-1))
        # 将池化后的隐藏状态和位置嵌入相加得到最终的隐藏状态表示
        hidden_states = pooled_hidden_states[..., :min_length] + position_embeddings[..., :min_length]
        # 将隐藏状态维度再次转置为输出形状
        hidden_states = hidden_states.transpose(1, 2)

        # 将最终隐藏状态传入编码器进行编码，获取编码器输出
        encoder_outputs = self.encoder(hidden_states, attention_mask, output_hidden_states, output_attentions)

        # 对编码器输出进行上采样操作
        hidden_states = self.upsample(encoder_outputs.last_hidden_state)
        # 如果上采样后的长度小于输入长度，则进行填充操作
        if hidden_states.shape[1] < n_input_timesteps:
            hidden_states = nn.functional.pad(hidden_states, (0, 0, 0, n_input_timesteps - hidden_states.shape[1]))

        # 如果 return_dict 为 False，则返回非空的元组
        if not return_dict:
            return tuple(
                v for v in [hidden_states, encoder_outputs.hidden_states, encoder_outputs.attentions] if v is not None
            )
        
        # 返回 BaseModelOutput 对象，包含最终的隐藏状态、编码器的隐藏状态和注意力权重
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


# SEWDPreTrainedModel 是一个抽象类，继承自 PreTrainedModel，用于处理权重初始化、预训练模型的下载和加载接口
class SEWDPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 定义 SEWDConfig 类作为配置类
    config_class = SEWDConfig
    # 设置基础模型前缀为 "sew-d"
    base_model_prefix = "sew-d"
    # 设置主输入名称为 "input_values"
    main_input_name = "input_values"
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """初始化权重"""
        # 如果模块是 SEWDPositionalConvEmbedding 的实例
        if isinstance(module, SEWDPositionalConvEmbedding):
            # 初始化卷积层的权重为正态分布
            nn.init.normal_(
                module.conv.weight,
                mean=0,
                std=2 * math.sqrt(1 / (module.conv.kernel_size[0] * module.conv.in_channels)),
            )
            # 初始化卷积层的偏置为常数0
            nn.init.constant_(module.conv.bias, 0)
        elif isinstance(module, nn.Linear):
            # 对线性层的权重进行初始化，使用正态分布，标准差为配置中的初始化范围
            # 这里与 TensorFlow 版本略有不同，后者使用截断正态分布进行初始化
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            # 对层归一化和分组归一化的偏置初始化为零
            module.bias.data.zero_()
            # 对层归一化和分组归一化的权重初始化为1
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            # 如果启用了 DeepSpeed Zero3
            if is_deepspeed_zero3_enabled():
                import deepspeed

                # 如果模块有权重分布，使用 GatheredParameters 进行初始化
                if hasattr(module, "weight_v") and hasattr(module, "weight_g"):
                    with deepspeed.zero.GatheredParameters([module.weight_v, module.weight_g], modifier_rank=0):
                        # 使用 Kaiming 正态分布初始化权重
                        nn.init.kaiming_normal_(module.weight.data)
                else:
                    with deepspeed.zero.GatheredParameters(module.weight, modifier_rank=0):
                        # 使用 Kaiming 正态分布初始化权重
                        nn.init.kaiming_normal_(module.weight.data)
            else:
                # 使用 Kaiming 正态分布初始化权重
                nn.init.kaiming_normal_(module.weight.data)
        elif isinstance(module, nn.Embedding):
            # 使用正态分布初始化嵌入层的权重，标准差为配置中的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果设置了填充索引，将对应索引的权重初始化为零
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

        # 如果模块是线性层或卷积层且有偏置，则将偏置初始化为零
        if isinstance(module, (nn.Linear, nn.Conv1d)) and module.bias is not None:
            module.bias.data.zero_()

    def _get_feat_extract_output_lengths(self, input_lengths: Union[torch.LongTensor, int]):
        """
        计算卷积层的输出长度
        """

        def _conv_out_length(input_length, kernel_size, stride):
            # 从 https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html 取得的一维卷积层输出长度公式
            return torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1

        # 遍历配置中的卷积核大小和步长，计算每一层卷积的输出长度
        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        # 返回最终的输入长度
        return input_lengths
    # 根据给定的特征向量长度和注意力掩码计算输出长度
    output_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)
    # 获取批次大小
    batch_size = attention_mask.shape[0]

    # 创建一个全零注意力掩码张量，形状为(batch_size, feature_vector_length)，与输入掩码相同的数据类型和设备
    attention_mask = torch.zeros(
        (batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
    )

    # 设置输出长度之前的所有位置为1，以确保这些位置被完全考虑
    attention_mask[(torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1)] = 1

    # 反转注意力掩码张量，沿着最后一个维度进行累积求和，并再次反转，最终转换为布尔类型
    attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()

    # 返回处理后的注意力掩码张量
    return attention_mask
@add_start_docstrings(
    "The bare SEW-D Model transformer outputting raw hidden-states without any specific head on top.",
    SEWD_START_DOCSTRING,
)
# 使用add_start_docstrings装饰器添加文档字符串，描述SEW-D模型输出原始隐藏状态，没有特定的输出头部
# 继承自SEWDPreTrainedModel，该类可能定义在transformers.models.sew.modeling_sew.SEWModel中，将SEW替换为SEWD，layer_norm_eps替换为feature_layer_norm_eps
class SEWDModel(SEWDPreTrainedModel):
    # 初始化方法，接受一个 SEWDConfig 类型的参数 config
    def __init__(self, config: SEWDConfig):
        # 调用父类的初始化方法，传入 config 参数
        super().__init__(config)
        # 将 config 参数保存在对象的 config 属性中
        self.config = config
        # 使用 SEWDFeatureEncoder 类根据 config 创建特征提取器对象，保存在 feature_extractor 属性中
        self.feature_extractor = SEWDFeatureEncoder(config)
        # 创建一个具有指定维度的 LayerNorm 层，eps 参数为 config 中的 feature_layer_norm_eps
        self.layer_norm = nn.LayerNorm(config.conv_dim[-1], eps=config.feature_layer_norm_eps)

        # 判断是否需要将特征向量投影到不同的维度
        self.project_features = config.conv_dim[-1] != config.hidden_size
        if self.project_features:
            # 如果需要投影特征向量，创建一个 Linear 层，将 conv_dim[-1] 维度投影到 hidden_size 维度
            self.feature_projection = nn.Linear(config.conv_dim[-1], config.hidden_size)
        # 创建一个 Dropout 层，用于特征投影的 dropout，dropout 率为 config.feat_proj_dropout
        self.feature_dropout = nn.Dropout(config.feat_proj_dropout)

        # 如果 config 中指定了 mask_time_prob 或 mask_feature_prob 大于 0.0
        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0:
            # 创建一个随机初始化的可学习参数，大小为 hidden_size
            self.masked_spec_embed = nn.Parameter(torch.FloatTensor(config.hidden_size).uniform_())

        # 使用 SEWDEncoder 类根据 config 创建编码器对象，保存在 encoder 属性中
        self.encoder = SEWDEncoder(config)

        # 调用类的后期初始化方法
        self.post_init()

    # 以下方法是从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2Model._mask_hidden_states 复制而来
    def _mask_hidden_states(
        self,
        hidden_states: torch.FloatTensor,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
    ):
        """
        Masks extracted features along time axis and/or along feature axis according to
        [SpecAugment](https://arxiv.org/abs/1904.08779).
        """

        # `config.apply_spec_augment` can set masking to False
        # 检查配置中是否禁用了 SpecAugment，如果是，则直接返回隐藏状态
        if not getattr(self.config, "apply_spec_augment", True):
            return hidden_states

        # generate indices & apply SpecAugment along time axis
        batch_size, sequence_length, hidden_size = hidden_states.size()

        if mask_time_indices is not None:
            # apply SpecAugment along time axis with given mask_time_indices
            # 使用给定的 mask_time_indices 对时间轴进行 SpecAugment
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)
        elif self.config.mask_time_prob > 0 and self.training:
            # compute mask indices if not provided and apply SpecAugment along time axis
            # 如果未提供 mask_time_indices，则计算掩码索引并沿时间轴应用 SpecAugment
            mask_time_indices = _compute_mask_indices(
                (batch_size, sequence_length),
                mask_prob=self.config.mask_time_prob,
                mask_length=self.config.mask_time_length,
                attention_mask=attention_mask,
                min_masks=self.config.mask_time_min_masks,
            )
            mask_time_indices = torch.tensor(mask_time_indices, device=hidden_states.device, dtype=torch.bool)
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)

        if self.config.mask_feature_prob > 0 and self.training:
            # generate indices & apply SpecAugment along feature axis
            # 如果训练模式且配置中开启了 mask_feature_prob，则生成索引并沿特征轴应用 SpecAugment
            mask_feature_indices = _compute_mask_indices(
                (batch_size, hidden_size),
                mask_prob=self.config.mask_feature_prob,
                mask_length=self.config.mask_feature_length,
                min_masks=self.config.mask_feature_min_masks,
            )
            mask_feature_indices = torch.tensor(mask_feature_indices, device=hidden_states.device, dtype=torch.bool)
            mask_feature_indices = mask_feature_indices[:, None].expand(-1, sequence_length, -1)
            hidden_states[mask_feature_indices] = 0

        return hidden_states

    @add_start_docstrings_to_model_forward(SEWD_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="audio",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    # 定义方法的返回类型为元组或BaseModelOutput类型
    ) -> Union[Tuple, BaseModelOutput]:
        # 如果未指定输出注意力的配置，则使用模型的默认设置
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果未指定输出隐藏状态的配置，则使用模型的默认设置
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果未指定返回字典的配置，则使用模型的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 提取特征向量
        extract_features = self.feature_extractor(input_values)
        # 调整特征向量的维度顺序
        extract_features = extract_features.transpose(1, 2)
        # 对特征向量进行层归一化处理
        extract_features = self.layer_norm(extract_features)

        # 如果需要将特征向量投影，则进行投影
        if self.project_features:
            extract_features = self.feature_projection(extract_features)
        # 对特征向量进行特征丢弃（dropout）
        hidden_states = self.feature_dropout(extract_features)

        # 如果存在注意力遮罩，则根据特征向量的形状生成相应的减少注意力遮罩
        if attention_mask is not None:
            # 计算与特征向量对应的减少注意力遮罩
            attention_mask = self._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)

        # 根据时间索引遮罩隐藏状态
        hidden_states = self._mask_hidden_states(hidden_states, mask_time_indices=mask_time_indices)

        # 将隐藏状态和其他配置传递给编码器进行处理
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从编码器输出中提取隐藏状态
        hidden_states = encoder_outputs[0]

        # 如果不要求返回字典，则返回隐藏状态和编码器输出的其他部分
        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]

        # 返回BaseModelOutput对象，包含最终隐藏状态、所有隐藏状态和注意力值
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
@add_start_docstrings(
    """SEW-D Model with a `language modeling` head on top for Connectionist Temporal Classification (CTC).""",
    SEWD_START_DOCSTRING,
)
# SEWDForCTC 类，用于在 Connectionist Temporal Classification (CTC) 上添加一个语言建模头部的 SEW-D 模型。
# 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForCTC 复制而来，将 Wav2Vec2 改为 SEWD，wav2vec2 改为 sew_d，WAV_2_VEC_2 改为 SEWD
class SEWDForCTC(SEWDPreTrainedModel):
    def __init__(self, config, target_lang: Optional[str] = None):
        super().__init__(config)

        # 初始化 SEWD 模型和 dropout 层
        self.sew_d = SEWDModel(config)
        self.dropout = nn.Dropout(config.final_dropout)

        self.target_lang = target_lang

        # 检查是否定义了语言模型头部的词汇表大小
        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `SEWDForCTC.from_pretrained(..., vocab_size=vocab_size)`. "
                "or define `vocab_size` of your model's configuration."
            )

        # 根据配置确定输出隐藏层大小
        output_hidden_size = (
            config.output_hidden_size if hasattr(config, "add_adapter") and config.add_adapter else config.hidden_size
        )

        # 初始化语言模型头部线性层
        self.lm_head = nn.Linear(output_hidden_size, config.vocab_size)

        # 初始化权重并进行最终处理
        self.post_init()

    def tie_weights(self):
        """
        This method overwrites [`~PreTrainedModel.tie_weights`] so that adapter weights can be correctly loaded when
        passing `target_lang=...` to `from_pretrained(...)`.

        This method is **not** supposed to be called by the user and is prone to be changed in the future.
        """

        # 注意，`tie_weights` 通常用于绑定输入和输出嵌入权重。在这里，我们重新定义此方法，以便在 SEWD 中正确加载适配器层，避免引入新的 API 到 `PreTrainedModel`。
        # 如果 `target_lang` 不是 None，并且配置中未定义 `adapter_attn_dim`，则会引发 ValueError。
        # 如果 `target_lang` 是 None，并且配置中定义了 `adapter_attn_dim`，则记录日志信息。
        # 如果 `target_lang` 不是 None，则强制加载指定的适配器层。
        target_lang = self.target_lang

        if target_lang is not None and getattr(self.config, "adapter_attn_dim", None) is None:
            raise ValueError(f"Cannot pass `target_lang`: {target_lang} if `config.adapter_attn_dim` is not defined.")
        elif target_lang is None and getattr(self.config, "adapter_attn_dim", None) is not None:
            logger.info("By default `target_lang` is set to 'eng'.")
        elif target_lang is not None:
            self.load_adapter(target_lang, force_load=True)
    # 调用此函数将冻结特征编码器的梯度计算，使其参数在训练期间不会更新。
    def freeze_feature_extractor(self):
        # 发出警告，提示该方法即将被弃用并在 Transformers v5 中移除，建议使用等效的 `freeze_feature_encoder` 方法。
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. "
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        # 调用 `freeze_feature_encoder` 方法冻结特征编码器。
        self.freeze_feature_encoder()

    # 调用此函数将禁用特征编码器的梯度计算，使其参数在训练期间不会更新。
    def freeze_feature_encoder(self):
        # 调用内部函数 `_freeze_parameters` 来冻结特征编码器的参数。
        self.sew_d.feature_extractor._freeze_parameters()

    # 调用此函数将禁用基础模型的梯度计算，使其参数在训练期间不会更新，只有分类头会被更新。
    def freeze_base_model(self):
        # 遍历 `self.sew_d` 的所有参数，将它们的梯度计算设为 False。
        for param in self.sew_d.parameters():
            param.requires_grad = False

    # 在模型前向传播过程中的参数注解和代码示例的添加函数修饰器
    @add_start_docstrings_to_model_forward(SEWD_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_CTC_EXPECTED_OUTPUT,
        expected_loss=_CTC_EXPECTED_LOSS,
    )
    # 模型的前向传播函数，接受输入值、注意力掩码等多个可选参数
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        ) -> Union[Tuple, CausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """

        # Determine if return_dict is explicitly provided; otherwise, use the default from model config
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Perform the forward pass through the model's sequence to sequence decoder
        outputs = self.sew_d(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Extract hidden states from the model outputs and apply dropout
        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        # Compute logits from the language model head
        logits = self.lm_head(hidden_states)

        # Initialize loss as None
        loss = None
        # Calculate loss only if labels are provided
        if labels is not None:
            # Check if any label value exceeds the vocabulary size, which would be invalid
            if labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

            # Retrieve input lengths from attention_mask, defaulting to all ones if mask is None
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            # Determine target lengths and flatten the targets tensor
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # Compute log probabilities using log_softmax for the logits
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

            # Disable cudnn for this section due to compatibility issues with fp16
            with torch.backends.cudnn.flags(enabled=False):
                # Compute the connectionist temporal classification (CTC) loss
                loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )

        # If return_dict is False, return output tuple without loss
        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        # If return_dict is True, return CausalLMOutput object with all relevant outputs
        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )
@add_start_docstrings(
    """
    SEWD Model with a sequence classification head on top (a linear layer over the pooled output) for tasks like SUPERB
    Keyword Spotting.
    """,
    SEWD_START_DOCSTRING,
)
# 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForSequenceClassification中复制而来，将Wav2Vec2改为SEWD，wav2vec2改为sew_d，WAV_2_VEC_2改为SEWD
class SEWDForSequenceClassification(SEWDPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        if hasattr(config, "add_adapter") and config.add_adapter:
            raise ValueError(
                "Sequence classification does not support the use of SEWD adapters (config.add_adapter=True)"
            )
        self.sew_d = SEWDModel(config)
        num_layers = config.num_hidden_layers + 1  # transformer layers + input embeddings
        if config.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        self.projector = nn.Linear(config.hidden_size, config.classifier_proj_size)
        self.classifier = nn.Linear(config.classifier_proj_size, config.num_labels)

        # Initialize weights and apply final processing
        self.post_init()

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameters will
        not be updated during training.
        """
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. "
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.sew_d.feature_extractor._freeze_parameters()

    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        for param in self.sew_d.parameters():
            param.requires_grad = False

    @add_start_docstrings_to_model_forward(SEWD_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_SEQ_CLASS_CHECKPOINT,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="audio",
        expected_output=_SEQ_CLASS_EXPECTED_OUTPUT,
        expected_loss=_SEQ_CLASS_EXPECTED_LOSS,
    )
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        # 设定是否返回结果的字典形式，默认根据模型配置决定
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 如果配置中指定了使用加权层求和，则输出隐藏状态
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states

        # 调用模型的前向计算
        outputs = self.sew_d(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 如果配置中指定了使用加权层求和
        if self.config.use_weighted_layer_sum:
            # 获取隐藏状态的列表
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            # 将隐藏状态堆叠起来形成张量
            hidden_states = torch.stack(hidden_states, dim=1)
            # 计算归一化的权重
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            # 对隐藏状态进行加权求和
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            # 否则直接取第一个输出作为隐藏状态
            hidden_states = outputs[0]

        # 将隐藏状态传递给投影器
        hidden_states = self.projector(hidden_states)

        # 如果没有给定注意力掩码，则对隐藏状态进行平均池化
        if attention_mask is None:
            pooled_output = hidden_states.mean(dim=1)
        else:
            # 否则根据注意力掩码获取特征向量并进行加权池化
            padding_mask = self._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)
            hidden_states[~padding_mask] = 0.0
            pooled_output = hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)

        # 将池化输出传递给分类器得到预测的 logits
        logits = self.classifier(pooled_output)

        # 计算损失，如果给定了标签
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        # 根据是否返回字典形式决定输出的结构
        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        # 返回带有损失、logits、隐藏状态和注意力的结果字典形式
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```