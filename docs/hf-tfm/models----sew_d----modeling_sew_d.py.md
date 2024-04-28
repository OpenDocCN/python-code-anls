# `.\transformers\models\sew_d\modeling_sew_d.py`

```
# 设置文件编码为 UTF-8
# 版权声明
# 根据 Apache 许可证 2.0 版本，对此文件的使用受限
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
#     http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于"AS IS"的基础分发的
# 没有任何形式的担保或条件，无论是明示的还是暗示的
# 请查看许可证以获取特定语言的权限和限制

# 导入所需的库
import math
import warnings
from collections.abc import Sequence
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, LayerNorm

# 导入所需的模块和函数
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_outputs import BaseModelOutput, CausalLMOutput, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import softmax_backward_data
from ...utils import add_code_sample_docstrings, add_start_docstrings, add_start_docstrings_to_model_forward, logging
from .configuration_sew_d import SEWDConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 隐藏状态的起始位置
_HIDDEN_STATES_START_POSITION = 1

# 通用文档字符串
_CONFIG_FOR_DOC = "SEWDConfig"

# 基本文档字符串
_CHECKPOINT_FOR_DOC = "asapp/sew-d-tiny-100k-ft-ls100h"
_EXPECTED_OUTPUT_SHAPE = [1, 292, 384]

# CTC 文档字符串
_CTC_EXPECTED_OUTPUT = "'MISTER QUILTER IS THE APOSTIL OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL'"
_CTC_EXPECTED_LOSS = 0.21

# 音频分类文档字符串
_SEQ_CLASS_CHECKPOINT = "anton-l/sew-d-mid-400k-ft-keyword-spotting"
_SEQ_CLASS_EXPECTED_OUTPUT = "'_unknown_'"
_SEQ_CLASS_EXPECTED_LOSS = 3.16

# SEW-D 预训练模型存档列表
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

# 从 transformers.models.wav2vec2.modeling_wav2vec2._compute_mask_indices 复制的函数
def _compute_mask_indices(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    attention_mask: Optional[torch.LongTensor] = None,
    min_masks: int = 0,
) -> np.ndarray:
    """
    计算给定形状的随机掩码范围。用于实现[SpecAugment: A Simple Data Augmentation Method for ASR](https://arxiv.org/abs/1904.08779)。
    请注意，此方法未经过优化，应在 CPU 上运行作为训练期间的预处理的一部分，而不是在 TPU 上运行。
    """
    Args:
        shape: 要计算掩码的形状。应该是一个大小为2的元组，其中第一个元素是批量大小，第二个元素是要跨越的轴的长度。
        mask_prob: 要掩盖的整个轴的百分比（介于0和1之间）。通过`mask_prob*shape[1]/mask_length`计算长度为`mask_length`的独立生成的掩码跨度的数量。请注意，由于重叠，`mask_prob`是一个上限，实际百分比会更小。
        mask_length: 掩码的大小
        min_masks: 掩盖跨度的最小数量
        attention_mask: 一个（右填充的）注意力掩码，独立缩短每个批量维度的特征轴。
    """
    batch_size, sequence_length = shape

    if mask_length < 1:
        raise ValueError("`mask_length`必须大于0。")

    if mask_length > sequence_length:
        raise ValueError(
            f"`mask_length`必须小于`sequence_length`，但是得到`mask_length`：{mask_length}"
            f"和`sequence_length`：{sequence_length}`"
        )

    # epsilon用于概率舍入
    epsilon = np.random.rand(1).item()

    def compute_num_masked_span(input_length):
        """给定输入长度，计算应该掩盖多少跨度"""
        num_masked_span = int(mask_prob * input_length / mask_length + epsilon)
        num_masked_span = max(num_masked_span, min_masks)

        # 确保掩盖的跨度数<=序列长度
        if num_masked_span * mask_length > sequence_length:
            num_masked_span = sequence_length // mask_length

        # 确保掩盖的跨度数也<=输入长度 - (mask_length - 1)
        if input_length - (mask_length - 1) < num_masked_span:
            num_masked_span = max(input_length - (mask_length - 1), 0)

        return num_masked_span

    # 计算批量中掩盖的跨度数
    input_lengths = (
        attention_mask.sum(-1).detach().tolist()
        if attention_mask is not None
        else [sequence_length for _ in range(batch_size)]
    )

    # SpecAugment掩码填充
    spec_aug_mask = np.zeros((batch_size, sequence_length), dtype=bool)
    spec_aug_mask_idxs = []

    max_num_masked_span = compute_num_masked_span(sequence_length)

    if max_num_masked_span == 0:
        return spec_aug_mask
    # 遍历输入长度列表
    for input_length in input_lengths:
        # 计算该输入的被遮罩的跨度数量
        num_masked_span = compute_num_masked_span(input_length)

        # 获取随机索引以进行遮罩
        spec_aug_mask_idx = np.random.choice(
            np.arange(input_length - (mask_length - 1)), num_masked_span, replace=False
        )

        # 选择第一个采样索引作为填充向量的虚拟索引，以确保由于概率舍入而使所有批次具有相同的维度
        # 选择第一个样本只是两次填充这些向量。
        if len(spec_aug_mask_idx) == 0:
            # 这种情况只会发生在`input_length`严格小于`sequence_length`的情况下，
            # 此时最后一个标记必须是填充标记，我们可以将其用作虚拟遮罩ID
            dummy_mask_idx = sequence_length - 1
        else:
            dummy_mask_idx = spec_aug_mask_idx[0]

        spec_aug_mask_idx = np.concatenate(
            [spec_aug_mask_idx, np.ones(max_num_masked_span - num_masked_span, dtype=np.int32) * dummy_mask_idx]
        )
        spec_aug_mask_idxs.append(spec_aug_mask_idx)

    spec_aug_mask_idxs = np.array(spec_aug_mask_idxs)

    # 将遮罩索引扩展为遮罩跨度
    spec_aug_mask_idxs = np.broadcast_to(
        spec_aug_mask_idxs[:, :, None], (batch_size, max_num_masked_span, mask_length)
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs.reshape(batch_size, max_num_masked_span * mask_length)

    # 添加偏移量到起始索引，使索引现在创建一个跨度
    offsets = np.arange(mask_length)[None, None, :]
    offsets = np.broadcast_to(offsets, (batch_size, max_num_masked_span, mask_length)).reshape(
        batch_size, max_num_masked_span * mask_length
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

    # 确保我们不能有大于sequence_length的索引
    if spec_aug_mask_idxs.max() > sequence_length - 1:
        spec_aug_mask_idxs[spec_aug_mask_idxs > sequence_length - 1] = sequence_length - 1

    # 将索引散布到遮罩
    np.put_along_axis(spec_aug_mask, spec_aug_mask_idxs, 1, -1)

    return spec_aug_mask
# 从transformers.models.deberta_v2.modeling_deberta_v2.make_log_bucket_position中复制代码
def make_log_bucket_position(relative_pos, bucket_size, max_position):
    # 计算相对位置的符号
    sign = torch.sign(relative_pos)
    # 计算绝对位置
    mid = bucket_size // 2
    abs_pos = torch.where(
        (relative_pos < mid) & (relative_pos > -mid),
        torch.tensor(mid - 1).type_as(relative_pos),
        torch.abs(relative_pos),
    )
    # 计算对数位置
    log_pos = (
        torch.ceil(torch.log(abs_pos / mid) / torch.log(torch.tensor((max_position - 1) / mid)) * (mid - 1)) + mid
    )
    # 计算桶位置
    bucket_pos = torch.where(abs_pos <= mid, relative_pos.type_as(log_pos), log_pos * sign)
    return bucket_pos


# 从transformers.models.deberta_v2.modeling_deberta_v2.build_relative_position中复制代码
def build_relative_position(query_size, key_size, bucket_size=-1, max_position=-1, device=None):
    """
    根据查询和键构建相对位置

    假设查询的绝对位置为(0, query_size)，键的绝对位置为(0, key_size)，则从查询到键的相对位置为R_{q \rightarrow k} = P_q - P_k

    参数:
        query_size (int): 查询的长度
        key_size (int): 键的长度
        bucket_size (int): 位置桶的大小
        max_position (int): 允许的最大绝对位置
        device (`torch.device`): 创建张量的设备

    返回:
        `torch.LongTensor`: 形状为[1, query_size, key_size]的张量
    """

    q_ids = torch.arange(0, query_size, device=device)
    k_ids = torch.arange(0, key_size, device=device)
    rel_pos_ids = q_ids[:, None] - k_ids[None, :]
    if bucket_size > 0 and max_position > 0:
        rel_pos_ids = make_log_bucket_position(rel_pos_ids, bucket_size, max_position)
    rel_pos_ids = rel_pos_ids.to(torch.long)
    rel_pos_ids = rel_pos_ids[:query_size, :]
    rel_pos_ids = rel_pos_ids.unsqueeze(0)
    return rel_pos_ids


@torch.jit.script
# 从transformers.models.deberta.modeling_deberta.c2p_dynamic_expand中复制代码
def c2p_dynamic_expand(c2p_pos, query_layer, relative_pos):
    return c2p_pos.expand([query_layer.size(0), query_layer.size(1), query_layer.size(2), relative_pos.size(-1)])


@torch.jit.script
# 从transformers.models.deberta.modeling_deberta.p2c_dynamic_expand中复制代码
def p2c_dynamic_expand(c2p_pos, query_layer, key_layer):
    return c2p_pos.expand([query_layer.size(0), query_layer.size(1), key_layer.size(-2), key_layer.size(-2)])


@torch.jit.script
# 从transformers.models.deberta.modeling_deberta.pos_dynamic_expand中复制代码
def pos_dynamic_expand(pos_index, p2c_att, key_layer):
    return pos_index.expand(p2c_att.size()[:2] + (pos_index.size(-2), key_layer.size(-2)))


# 从transformers.models.deberta.modeling_deberta.get_mask中复制代码
def get_mask(input, local_context):
    if not isinstance(local_context, DropoutContext):
        dropout = local_context
        mask = None
    else:
        # 如果不是第一次调用，从上下文中获取dropout值
        dropout = local_context.dropout
        # 将dropout值乘以缩放比例
        dropout *= local_context.scale
        # 如果不重用mask，则将mask设置为None
        mask = local_context.mask if local_context.reuse_mask else None

    # 如果dropout大于0且mask为None，则生成一个mask
    if dropout > 0 and mask is None:
        # 生成一个与输入张量相同大小的随机二值mask
        mask = (1 - torch.empty_like(input).bernoulli_(1 - dropout)).to(torch.bool)

    # 如果上下文是DropoutContext类型
    if isinstance(local_context, DropoutContext):
        # 如果上下文中的mask为None，则将生成的mask赋值给上下文中的mask
        if local_context.mask is None:
            local_context.mask = mask

    # 返回生成的mask和dropout值
    return mask, dropout
# 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2NoLayerNormConvLayer复制代码，并将Wav2Vec2更改为SEWD
class SEWDNoLayerNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        # 创建一个一维卷积层，设置输入维度、输出维度、卷积核大小、步长和偏置
        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        # 设置激活函数
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        # 对输入的隐藏状态进行卷积操作
        hidden_states = self.conv(hidden_states)
        # 使用激活函数处理卷积后的隐藏状态
        hidden_states = self.activation(hidden_states)
        return hidden_states


# 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2LayerNormConvLayer复制代码，并将Wav2Vec2更改为SEWD
class SEWDLayerNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        # 创建一个一维卷积层，设置输入维度、输出维度、卷积核大小、步长和偏置
        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        # 创建一个LayerNorm层，设置输出维度和是否使用可学习的仿射变换
        self.layer_norm = nn.LayerNorm(self.out_conv_dim, elementwise_affine=True)
        # 设置激活函数
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        # 对输入的隐藏状态进行卷积操作
        hidden_states = self.conv(hidden_states)

        # 转置隐藏状态的倒数第二和倒数第一维
        hidden_states = hidden_states.transpose(-2, -1)
        # 使用LayerNorm层处理隐藏状态
        hidden_states = self.layer_norm(hidden_states)
        # 再次转置隐藏状态的倒数第二和倒数第一维
        hidden_states = hidden_states.transpose(-2, -1)

        # 使用激活函数处理隐藏状��
        hidden_states = self.activation(hidden_states)
        return hidden_states


# 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2GroupNormConvLayer复制代码，并将Wav2Vec2更改为SEWD
class SEWDGroupNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        # 创建一个一维卷积层，设置输入维度、输出维度、卷积核大小、步长和偏置
        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        # 设置激活函数
        self.activation = ACT2FN[config.feat_extract_activation]

        # 创建一个GroupNorm层，设置组数、通道数和是否使用可学习的仿射变换
        self.layer_norm = nn.GroupNorm(num_groups=self.out_conv_dim, num_channels=self.out_conv_dim, affine=True)

    def forward(self, hidden_states):
        # 对输入的隐藏状态进行卷积操作
        hidden_states = self.conv(hidden_states)
        # 使用GroupNorm层处理隐藏状态
        hidden_states = self.layer_norm(hidden_states)
        # 使用激活函数处理隐藏状态
        hidden_states = self.activation(hidden_states)
        return hidden_states
# 从transformers.models.sew.modeling_sew.SEWPositionalConvEmbedding复制到SEWDPositionalConvEmbedding
class SEWDPositionalConvEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个一维卷积层，用于位置编码
        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.num_conv_pos_embeddings,
            padding=config.num_conv_pos_embeddings // 2,
            groups=config.num_conv_pos_embedding_groups,
            stride=config.squeeze_factor,
        )

        # 如果使用deepspeed zero3加速
        if is_deepspeed_zero3_enabled():
            import deepspeed
            # 使用deepspeed zero3对权重进行处理
            with deepspeed.zero.GatheredParameters(self.conv.weight, modifier_rank=0):
                self.conv = nn.utils.weight_norm(self.conv, name="weight", dim=2)
            # 注册外部参数
            deepspeed.zero.register_external_parameter(self, self.conv.weight_v)
            deepspeed.zero.register_external_parameter(self, self.conv.weight_g)
        else:
            # 对权重进行归一化处理
            self.conv = nn.utils.weight_norm(self.conv, name="weight", dim=2)

        # 创建一个用于填充的层
        self.padding = SEWDSamePadLayer(config.num_conv_pos_embeddings)
        # 激活函数
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        # 通过卷积层进行前向传播
        hidden_states = self.conv(hidden_states)
        # 填充
        hidden_states = self.padding(hidden_states)
        # 激活函数
        hidden_states = self.activation(hidden_states)

        return hidden_states


# 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2SamePadLayer复制到SEWDSamePadLayer
class SEWDSamePadLayer(nn.Module):
    def __init__(self, num_conv_pos_embeddings):
        super().__init__()
        # 计算需要移除的填充数量
        self.num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0

    def forward(self, hidden_states):
        # 如果需要移除填充，则进行裁剪操作
        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, :, : -self.num_pad_remove]
        return hidden_states


# 从transformers.models.sew.modeling_sew.SEWUpsampling复制到SEWDUpsampling
class SEWDUpsampling(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 线性投影层
        self.projection = nn.Linear(config.hidden_size, config.hidden_size * config.squeeze_factor)
        # 激活函数
        self.activation = ACT2FN[config.feat_extract_activation]
        self.squeeze_factor = config.squeeze_factor

    def forward(self, hidden_states):
        # 通过线性投影层进行前向传播
        hidden_states = self.projection(hidden_states)
        # 激活函数
        hidden_states = self.activation(hidden_states)

        if self.squeeze_factor > 1:
            # 将嵌入通道转换为序列长度
            bsz, src_len, src_embed_dim = hidden_states.size()
            tgt_len = src_len * self.squeeze_factor
            tgt_embed_dim = src_embed_dim // self.squeeze_factor
            hidden_states = hidden_states.reshape(bsz, src_len, self.squeeze_factor, tgt_embed_dim)
            hidden_states = hidden_states.reshape(bsz, tgt_len, tgt_embed_dim)

        return hidden_states


# 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2FeatureEncoder复制到SEWD
class SEWDFeatureEncoder(nn.Module):
    """从原始音频波形中构建特征"""

    def __init__(self, config):
        super().__init__()

        if config.feat_extract_norm == "group":
            conv_layers = [SEWDGroupNormConvLayer(config, layer_id=0)] + [
                SEWDNoLayerNormConvLayer(config, layer_id=i + 1) for i in range(config.num_feat_extract_layers - 1)
            ]
        elif config.feat_extract_norm == "layer":
            conv_layers = [SEWDLayerNormConvLayer(config, layer_id=i) for i in range(config.num_feat_extract_layers)]
        else:
            raise ValueError(
                f"`config.feat_extract_norm` is {config.feat_extract_norm}, but has to be one of ['group', 'layer']"
            )
        self.conv_layers = nn.ModuleList(conv_layers)
        self.gradient_checkpointing = False
        self._requires_grad = True

    def _freeze_parameters(self):
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def forward(self, input_values):
        hidden_states = input_values[:, None]

        # make sure hidden_states require grad for gradient_checkpointing
        if self._requires_grad and self.training:
            hidden_states.requires_grad = True

        for conv_layer in self.conv_layers:
            if self._requires_grad and self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    conv_layer.__call__,
                    hidden_states,
                )
            else:
                hidden_states = conv_layer(hidden_states)

        return hidden_states


class SEWDFeatureExtractor(SEWDFeatureEncoder):
    def __init__(self, config):
        super().__init__(config)
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
        self.dense = nn.Linear(config.pooler_hidden_size, config.pooler_hidden_size)
        self.dropout = StableDropout(config.pooler_dropout)
        self.config = config

    def forward(self, hidden_states):
        # 我们通过简单地取第一个标记对应的隐藏状态来“池化”模型。

        context_token = hidden_states[:, 0]
        context_token = self.dropout(context_token)
        pooled_output = self.dense(context_token)
        pooled_output = ACT2FN[self.config.pooler_hidden_act](pooled_output)
        return pooled_output

    @property
    def output_dim(self):
        return self.config.hidden_size


# 从transformers.models.deberta.modeling_deberta.XSoftmax复制而来，将deberta->deberta_v2
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
        # Save the dimension for later use
        self.dim = dim
        # Create a reverse mask
        rmask = ~(mask.to(torch.bool))

        # Fill masked elements with minimum value of input tensor
        output = input.masked_fill(rmask, torch.tensor(torch.finfo(input.dtype).min))
        # Apply softmax along the specified dimension
        output = torch.softmax(output, self.dim)
        # Fill masked elements with 0
        output.masked_fill_(rmask, 0)
        # Save the output for backward pass
        self.save_for_backward(output)
        return output

    @staticmethod
    def backward(self, grad_output):
        # Retrieve saved output tensor
        (output,) = self.saved_tensors
        # Calculate input gradient using softmax backward data function
        inputGrad = softmax_backward_data(self, grad_output, output, self.dim, output)
        return inputGrad, None, None

    @staticmethod
    def symbolic(g, self, mask, dim):
        import torch.onnx.symbolic_helper as sym_help
        from torch.onnx.symbolic_opset9 import masked_fill, softmax

        # Cast mask to Long type
        mask_cast_value = g.op("Cast", mask, to_i=sym_help.cast_pytorch_to_onnx["Long"])
        # Create reverse mask
        r_mask = g.op(
            "Cast",
            g.op("Sub", g.op("Constant", value_t=torch.tensor(1, dtype=torch.int64)), mask_cast_value),
            to_i=sym_help.cast_pytorch_to_onnx["Bool"],
        )
        # Fill masked elements with minimum value of input tensor
        output = masked_fill(
            g, self, r_mask, g.op("Constant", value_t=torch.tensor(torch.finfo(self.type().dtype()).min))
        )
        # Apply softmax along the specified dimension
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
        # Get mask and dropout value
        mask, dropout = get_mask(input, local_ctx)
        # Calculate scale factor
        ctx.scale = 1.0 / (1 - dropout)
        if dropout > 0:
            # Save mask for backward pass and apply dropout
            ctx.save_for_backward(mask)
            return input.masked_fill(mask, 0) * ctx.scale
        else:
            return input

    @staticmethod
    # 定义反向传播函数，根据上下文和梯度输出计算梯度
    def backward(ctx, grad_output):
        # 如果上下文中的缩放因子大于1
        if ctx.scale > 1:
            # 从保存的张量中获取掩码
            (mask,) = ctx.saved_tensors
            # 将梯度输出中的掩码位置置零，乘以缩放因子，返回结果和空梯度
            return grad_output.masked_fill(mask, 0) * ctx.scale, None
        else:
            # 如果缩放因子不大于1，直接返回梯度输出和空梯度
            return grad_output, None

    # 静态方法，用于在图中表示Dropout操作
    @staticmethod
    def symbolic(g: torch._C.Graph, input: torch._C.Value, local_ctx: Union[float, DropoutContext]) -> torch._C.Value:
        # 导入符号操作集12
        from torch.onnx import symbolic_opset12

        # 获取本地上下文中的dropout概率
        dropout_p = local_ctx
        # 如果本地上下文是DropoutContext类型，则获取其中的dropout概率
        if isinstance(local_ctx, DropoutContext):
            dropout_p = local_ctx.dropout
        # 在训练时调用StableDropout函数
        train = True
        # TODO: 我们应该检查导出时使用的opset版本是否大于12，但目前没有很好的方法来实现。
        # 如果opset版本小于12，导出将失败并显示CheckerError。
        # 一旦https://github.com/pytorch/pytorch/issues/78391问题得到解决，可以执行以下操作：
        # if opset_version < 12:
        #   return torch.onnx.symbolic_opset9.dropout(g, input, dropout_p, train)
        # 返回使用符号操作集12的dropout操作
        return symbolic_opset12.dropout(g, input, dropout_p, train)
# 从 transformers.models.deberta.modeling_deberta.StableDropout 复制过来的稳定的 Dropout 模块
class StableDropout(nn.Module):
    """
    优化的 dropout 模块，用于稳定训练

    Args:
        drop_prob (float): dropout 的概率
    """

    def __init__(self, drop_prob):
        super().__init__()
        self.drop_prob = drop_prob
        self.count = 0
        self.context_stack = None

    def forward(self, x):
        """
        调用模块

        Args:
            x (`torch.tensor`): 应用 dropout 的输入张量
        """
        if self.training and self.drop_prob > 0:
            return XDropout.apply(x, self.get_context())
        return x

    def clear_context(self):
        self.count = 0
        self.context_stack = None

    def init_context(self, reuse_mask=True, scale=1):
        if self.context_stack is None:
            self.context_stack = []
        self.count = 0
        for c in self.context_stack:
            c.reuse_mask = reuse_mask
            c.scale = scale

    def get_context(self):
        if self.context_stack is not None:
            if self.count >= len(self.context_stack):
                self.context_stack.append(DropoutContext())
            ctx = self.context_stack[self.count]
            ctx.dropout = self.drop_prob
            self.count += 1
            return ctx
        else:
            return self.drop_prob


# 从 transformers.models.deberta.modeling_deberta.DebertaSelfOutput 复制过来的 SEWD 的自我输出模块，
# 将 DebertaV2->SEWD, DebertaLayerNorm->LayerNorm, hidden_dropout_prob->activation_dropout
class SEWDSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = StableDropout(config.activation_dropout)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# 从 transformers.models.deberta_v2.modeling_deberta_v2.DisentangledSelfAttention 复制过来的解耦的自注意力模块，
# attention_probs_dropout_prob->attention_dropout, hidden_dropout_prob->activation_dropout
class DisentangledSelfAttention(nn.Module):
    """
    解耦的自注意力模块

    Parameters:
        config (`DebertaV2Config`):
            一个模型配置类实例，其中包含构建新模型所需的配置。其架构类似于 BertConfig，
            更多细节，请参考 `DebertaV2Config`
    """
    # 初始化函数，设置注意力头数、注意力头大小及相关参数
    def __init__(self, config):
        # 调用父类初始化函数
        super().__init__()
        # 检查隐藏层大小是否是注意力头数的整数倍
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )
        # 设置注意力头数
        self.num_attention_heads = config.num_attention_heads
        # 计算每个注意力头的大小
        _attention_head_size = config.hidden_size // config.num_attention_heads
        # 获取配置中的注意力头大小，如果没有设置则使用计算得到的值
        self.attention_head_size = getattr(config, "attention_head_size", _attention_head_size)
        # 计算所有头的总大小
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        # 对查询、键和值进行线性投影
        self.query_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
        self.key_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
        self.value_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)

        # 是否共享注意力键
        self.share_att_key = getattr(config, "share_att_key", False)
        # 设置位置注意力类型和相关参数
        self.pos_att_type = config.pos_att_type if config.pos_att_type is not None else []
        self.relative_attention = getattr(config, "relative_attention", False)

        if self.relative_attention:
            # 设置位置向量桶数和最大相对位置
            self.position_buckets = getattr(config, "position_buckets", -1)
            self.max_relative_positions = getattr(config, "max_relative_positions", -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings
            self.pos_ebd_size = self.max_relative_positions
            if self.position_buckets > 0:
                self.pos_ebd_size = self.position_buckets

            # 设置位置向量的稳定Dropout
            self.pos_dropout = StableDropout(config.activation_dropout)

            # 如果不共享注意力键，根据位置注意力类型，对键和查询进行线性投影
            if not self.share_att_key:
                if "c2p" in self.pos_att_type:
                    self.pos_key_proj = nn.Linear(config.hidden_size, self.all_head_size, bias=True)
                if "p2c" in self.pos_att_type:
                    self.pos_query_proj = nn.Linear(config.hidden_size, self.all_head_size)

        # 设置注意力Dropout
        self.dropout = StableDropout(config.attention_dropout)

    # 调整尺寸以便计算得分
    def transpose_for_scores(self, x, attention_heads):
        # 转换张量形状
        new_x_shape = x.size()[:-1] + (attention_heads, -1)
        x = x.view(new_x_shape)
        # 交换维度以便计算注意力得分
        return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))

    # 前向传播函数，处理隐藏状态、注意力掩码等参数
    def forward(
        self,
        hidden_states,
        attention_mask,
        output_attentions=False,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
# 定义SEWDAttention类，继承自nn.Module
class SEWDAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化DisentangledSelfAttention对象
        self.self = DisentangledSelfAttention(config)
        # 初始化SEWDSelfOutput对象
        self.output = SEWDSelfOutput(config)
        # 保存config参数
        self.config = config

    # 前向传播函数
    def forward(
        self,
        hidden_states,
        attention_mask,
        output_attentions=False,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
    ):
        # 调用self对象的前向传播函数，并保存结果
        self_output = self.self(
            hidden_states,
            attention_mask,
            output_attentions,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
        )
        # 如果output_attentions为True，则返回self_output和att_matrix
        if output_attentions:
            self_output, att_matrix = self_output
        # 如果query_states为空，则使用hidden_states代替
        if query_states is None:
            query_states = hidden_states
        # 调用output对象的前向传播函数，并返回结果
        attention_output = self.output(self_output, query_states)
        # 如果output_attentions为True，则返回attention_output和att_matrix
        if output_attentions:
            return (attention_output, att_matrix)
        else:
            return attention_output


# 定义SEWDIntermediate类，继承自nn.Module
class SEWDIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化Linear对象
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 初始化激活函数对象
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    # 前向传播函数
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 线性变换
        hidden_states = self.dense(hidden_states)
        # 使用激活函数
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


# 定义SEWDOutput类，继承自nn.Module
class SEWDOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化Linear对象
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 初始化LayerNorm对象
        self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps)
        # 初始化StableDropout对象
        self.dropout = StableDropout(config.activation_dropout)
        # ���存config参数
        self.config = config

    # 前向传播函数
    def forward(self, hidden_states, input_tensor):
        # 线性变换
        hidden_states = self.dense(hidden_states)
        # Dropout操作
        hidden_states = self.dropout(hidden_states)
        # LayerNorm操作
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


# 定义SEWDLayer类，继承自nn.Module
class SEWDLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化SEWDAttention对象
        self.attention = SEWDAttention(config)
        # 初始化SEWDIntermediate对象
        self.intermediate = SEWDIntermediate(config)
        # 初始化SEWDOutput对象
        self.output = SEWDOutput(config)
    # 定义神经网络层的前向传播函数，接收隐藏状态、注意力掩码、查询状态、相对位置、相对位置嵌入和是否输出注意力矩阵等参数
    def forward(
        self,
        hidden_states,
        attention_mask,
        query_states=None,
        relative_pos=None,
        rel_embeddings=None,
        output_attentions=False,
    ):
        # 调用注意力机制函数，传入隐藏状态、注意力掩码等参数，获取注意力输出
        attention_output = self.attention(
            hidden_states,
            attention_mask,
            output_attentions=output_attentions,
            query_states=query_states,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
        )
        # 如果需要输出注意力矩阵
        if output_attentions:
            # 将注意力输出分解为注意力输出和注意力矩阵
            attention_output, att_matrix = attention_output
        # 经过中间层处理
        intermediate_output = self.intermediate(attention_output)
        # 经过输出层处理，获取最终层输出
        layer_output = self.output(intermediate_output, attention_output)
        # 如果需要输出注意力矩阵，则返回最终层输出和注意力矩阵
        if output_attentions:
            return (layer_output, att_matrix)
        # 否则，只返回最终层输出
        else:
            return layer_output
# 从transformers.models.deberta_v2.modeling_deberta_v2.ConvLayer复制而来的类ConvLayer
class ConvLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 获取配置中的卷积核大小，默认为3
        kernel_size = getattr(config, "conv_kernel_size", 3)
        # 获取配置中的分组数，默认为1
        groups = getattr(config, "conv_groups", 1)
        # 获取配置中的卷积激活函数，默认为"tanh"
        self.conv_act = getattr(config, "conv_act", "tanh")
        # 创建1维卷积层
        self.conv = nn.Conv1d(
            config.hidden_size, config.hidden_size, kernel_size, padding=(kernel_size - 1) // 2, groups=groups
        )
        # 创建LayerNorm层
        self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps)
        # 创建稳定的Dropout层
        self.dropout = StableDropout(config.hidden_dropout_prob)
        self.config = config

    def forward(self, hidden_states, residual_states, input_mask):
        # 执行卷积操作并进行激活函数处理
        out = self.conv(hidden_states.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        # 根据输入mask，将对应位置置零
        rmask = (1 - input_mask).bool()
        out.masked_fill_(rmask.unsqueeze(-1).expand(out.size()), 0)
        # 使用指定的激活函数处理Dropout后的结果
        out = ACT2FN[self.conv_act](self.dropout(out))

        # 将输入状态和卷积结果相加得到层归一后的输出
        layer_norm_input = residual_states + out
        output = self.LayerNorm(layer_norm_input).to(layer_norm_input)

        if input_mask is None:
            output_states = output
        else:
            # 若输入mask的维度不符合要求，则处理维度
            if input_mask.dim() != layer_norm_input.dim():
                if input_mask.dim() == 4:
                    input_mask = input_mask.squeeze(1).squeeze(1)
                input_mask = input_mask.unsqueeze(2)

            input_mask = input_mask.to(output.dtype)
            # 对输出应用mask
            output_states = output * input_mask

        return output_states


# 从transformers.models.deberta_v2.modeling_deberta_v2.DebertaV2Encoder复制而来的类SEWDTransformerEncoder，将DebertaV2换成SEWD
class SEWDTransformerEncoder(nn.Module):
    """Modified BertEncoder with relative position bias support"""

    def __init__(self, config):
        super().__init__()

        # 创建一系列SEWDLayer层
        self.layer = nn.ModuleList([SEWDLayer(config) for _ in range(config.num_hidden_layers)])
        # 是否支持相对位置偏置
        self.relative_attention = getattr(config, "relative_attention", False)

        if self.relative_attention:
            # 最大相对位置���离
            self.max_relative_positions = getattr(config, "max_relative_positions", -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings

            # 相对位置的embedding大小
            self.position_buckets = getattr(config, "position_buckets", -1)
            pos_ebd_size = self.max_relative_positions * 2

            if self.position_buckets > 0:
                pos_ebd_size = self.position_buckets * 2

            # 创建相对位置的embedding
            self.rel_embeddings = nn.Embedding(pos_ebd_size, config.hidden_size)

        # 规范化相对位置embedding
        self.norm_rel_ebd = [x.strip() for x in getattr(config, "norm_rel_ebd", "none").lower().split("|")]

        if "layer_norm" in self.norm_rel_ebd:
            # 使用LayerNorm层对相对位置embedding进行处理
            self.LayerNorm = LayerNorm(config.hidden_size, config.layer_norm_eps, elementwise_affine=True)

        # 如果卷积核大小大于0，则创建卷积层
        self.conv = ConvLayer(config) if getattr(config, "conv_kernel_size", 0) > 0 else None
        self.gradient_checkpointing = False
    # 获取相对位置编码
    def get_rel_embedding(self):
        # 如果开启了相对注意力机制，则获取关系嵌入权重，否则为None
        rel_embeddings = self.rel_embeddings.weight if self.relative_attention else None
        # 如果关系嵌入权重不为空并且需要进行 layer_norm 处理
        if rel_embeddings is not None and ("layer_norm" in self.norm_rel_ebd):
            # 对关系嵌入进行 layer_norm 处理
            rel_embeddings = self.LayerNorm(rel_embeddings)
        # 返回关系嵌入权重
        return rel_embeddings

    # 获取注意力掩码
    def get_attention_mask(self, attention_mask):
        # 如果注意力掩码维度小于等于2
        if attention_mask.dim() <= 2:
            # 对注意力掩码进行扩展，添加两个维度
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # 将注意力掩码扩展成二维乘积矩阵
            attention_mask = extended_attention_mask * extended_attention_mask.squeeze(-2).unsqueeze(-1)
        # 如果注意力掩码维度为3
        elif attention_mask.dim() == 3:
            # 对注意力掩码进行维度扩展
            attention_mask = attention_mask.unsqueeze(1)
        # 返回处理后的注意力掩码
        return attention_mask

    # 获取相对位置
    def get_rel_pos(self, hidden_states, query_states=None, relative_pos=None):
        # 如果开启了相对注意力机制并且相对位置为空
        if self.relative_attention and relative_pos is None:
            # 计算相对位置
            q = query_states.size(-2) if query_states is not None else hidden_states.size(-2)
            relative_pos = build_relative_position(
                q,
                hidden_states.size(-2),
                bucket_size=self.position_buckets,
                max_position=self.max_relative_positions,
                device=hidden_states.device,
            )
        # 返回相对位置
        return relative_pos

    # 前向传播函数
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
        # 如果输入的 attention_mask 维度小于等于2，则将其赋值给 input_mask
        if attention_mask.dim() <= 2:
            input_mask = attention_mask
        else:
            # 如果输入的 attention_mask 维度大于2，则对其进行求和操作并比较是否大于0，得到一个布尔类型的 input_mask
            input_mask = attention_mask.sum(-2) > 0
        
        # 获取经过处理后的 attention_mask
        attention_mask = self.get_attention_mask(attention_mask)
        
        # 获取相对位置编码
        relative_pos = self.get_rel_pos(hidden_states, query_states, relative_pos)

        # 初始化存储所有隐藏状态的变量，如果没有设置输出隐藏状态则置为 None
        all_hidden_states = () if output_hidden_states else None
        # 初始化存储所有注意力的变量，如果没有设置输出注意力则置为 None
        all_attentions = () if output_attentions else None

        # 如果输入的 hidden_states 是一个序列
        if isinstance(hidden_states, Sequence):
            # 则将序列的第一个元素赋值给 next_kv
            next_kv = hidden_states[0]
        else:
            next_kv = hidden_states
        
        # 获取相对位置嵌入
        rel_embeddings = self.get_rel_embedding()
        
        # 初始化输出状态为 next_kv
        output_states = next_kv
        
        # 遍历每个层的模块
        for i, layer_module in enumerate(self.layer):
            # 如果设置了输出隐藏状态
            if output_hidden_states:
                # 则将 output_states 添加到 all_hidden_states 中
                all_hidden_states = all_hidden_states + (output_states,)

            # 如果设置了梯度检查点并且处于训练状态
            if self.gradient_checkpointing and self.training:
                # 使用 _gradient_checkpointing_func 进行梯度检查点计算
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
                # 否则直接调用层模块进行计算
                output_states = layer_module(
                    next_kv,
                    attention_mask,
                    query_states=query_states,
                    relative_pos=relative_pos,
                    rel_embeddings=rel_embeddings,
                    output_attentions=output_attentions,
                )

            # 如果设置了输出注意力
            if output_attentions:
                # 获取注意力矩阵并将其赋值给 output_states，att_m
                output_states, att_m = output_states

            # 如果是第一层且设置了 conv 模块
            if i == 0 and self.conv is not None:
                # 使用 conv 模块进行计算并更新 output_states
                output_states = self.conv(hidden_states, output_states, input_mask)

            # 如果有查询状态
            if query_states is not None:
                # 更新查询状态为 output_states
                query_states = output_states
                # 如果 hidden_states 是一个序列
                if isinstance(hidden_states, Sequence):
                    # 则更新 next_kv 为序列的下一个元素
                    next_kv = hidden_states[i + 1] if i + 1 < len(self.layer) else None
            else:
                # 否则更新 next_kv 为 output_states
                next_kv = output_states

            # 如果设置了输出注意力
            if output_attentions:
                # 将注意力矩阵添加到 all_attentions 中
                all_attentions = all_attentions + (att_m,)

        # 如果设置了输出隐藏状态
        if output_hidden_states:
            # 将最后的 output_states 添加到 all_hidden_states 中
            all_hidden_states = all_hidden_states + (output_states,)

        # 如果不需要返回字典
        if not return_dict:
            # 则返回一个元组，包含 output_states, all_hidden_states, all_attentions 中不为 None 的部分
            return tuple(v for v in [output_states, all_hidden_states, all_attentions] if v is not None)
        # 否则返回一个 BaseModelOutput 类的对象
        return BaseModelOutput(
            last_hidden_state=output_states, hidden_states=all_hidden_states, attentions=all_attentions
        )
class SEWDEncoder(nn.Module):
    # SEWDEncoder 类的构造函数
    def __init__(self, config):
        # 调用父类的构造函数
        super().__init__()
        # 保存配置信息
        self.config = config
        # 创建位置卷积嵌入对象
        self.pos_conv_embed = SEWDPositionalConvEmbedding(config)
        # 创建一维平均池化层
        self.pool = nn.AvgPool1d(config.squeeze_factor, config.squeeze_factor)
        # 创建 Transformer 编码器
        self.encoder = SEWDTransformerEncoder(config)
        # 创建上采样层
        self.upsample = SEWDUpsampling(config)
        # 是否启用梯度检查点
        self.gradient_checkpointing = False

    # 前向传播函数
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
        # 如果没有提供注意力掩码，则创建一个全为 1 的掩码
        if attention_mask is None:
            attention_mask = torch.ones(
                (hidden_states.shape[0], max_encoder_length), dtype=torch.long, device=hidden_states.device
            )
        else:
            # 确保填充的标记输出为 0
            hidden_states[~attention_mask.bool()] = 0.0
            # 计算输入长度
            input_lengths = (attention_mask.long()).sum(-1)
            # 应用池化公式以获取实际的输出长度
            output_lengths = input_lengths // self.config.squeeze_factor
            # 创建注意力掩码
            attention_ids = (
                torch.arange(0, max_encoder_length, device=output_lengths.device)
                .view(1, -1)
                .expand(output_lengths.shape[0], -1)
            )
            attention_mask = (attention_ids < output_lengths.view(-1, 1)).long()

        # 计算输入的时间步数
        n_input_timesteps = hidden_states.shape[1]

        # 转置隐藏状态
        hidden_states = hidden_states.transpose(1, 2)
        # 计算位置嵌入
        position_embeddings = self.pos_conv_embed(hidden_states)
        # 对隐藏状态进行池化
        pooled_hidden_states = self.pool(hidden_states)
        # 获取最小长度
        min_length = min(position_embeddings.size(-1), pooled_hidden_states.size(-1))
        # 将池化后的隐藏状态和位置嵌入相加
        hidden_states = pooled_hidden_states[..., :min_length] + position_embeddings[..., :min_length]
        # 再次转置隐藏状态
        hidden_states = hidden_states.transpose(1, 2)

        # 对隐藏状态进行 Transformer 编码
        encoder_outputs = self.encoder(hidden_states, attention_mask, output_hidden_states, output_attentions)

        # 对编码器输出进行上采样
        hidden_states = self.upsample(encoder_outputs.last_hidden_state)
        # 如果隐藏状态的长度小于输入时间步数，则进行填充
        if hidden_states.shape[1] < n_input_timesteps:
            hidden_states = nn.functional.pad(hidden_states, (0, 0, 0, n_input_timesteps - hidden_states.shape[1]))

        # 如果不返回字典，则返回元组
        if not return_dict:
            return tuple(
                v for v in [hidden_states, encoder_outputs.hidden_states, encoder_outputs.attentions] if v is not None
            )
        # 返回模型输出
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class SEWDPreTrainedModel(PreTrainedModel):
    """
    一个抽象类，处理权重初始化以及下载和加载预训练模型的简单接口。
    """

    # 配置类
    config_class = SEWDConfig
    # 基础模型前缀
    base_model_prefix = "sew-d"
    # 主输入名称
    main_input_name = "input_values"
    # 是否支持梯度检查点
    supports_gradient_checkpointing = True

    # 初始化权重的函数
    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果是 SEWDPositionalConvEmbedding 类的实例
        if isinstance(module, SEWDPositionalConvEmbedding):
            # 初始化卷积层权重
            nn.init.normal_(
                module.conv.weight,
                mean=0,
                std=2 * math.sqrt(1 / (module.conv.kernel_size[0] * module.conv.in_channels)),
            )
            # 初始化卷积层偏置
            nn.init.constant_(module.conv.bias, 0)
        # 如果是线性层
        elif isinstance(module, nn.Linear):
            # 初始化线性层权重，使用正态分布
            # 与 TF 版本稍有不同，TF 使用截断正态分布进行初始化
            # 参考 https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        # 如果是归一化层（LayerNorm 或 GroupNorm）
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            # 初始化归一化层偏置为零
            module.bias.data.zero_()
            # 初始化归一化层权重为1
            module.weight.data.fill_(1.0)
        # 如果是一维卷积层
        elif isinstance(module, nn.Conv1d):
            # 如果启用了深度速度的 Zero3
            if is_deepspeed_zero3_enabled():
                import deepspeed

                # 如果模块有权重分离的特性
                if hasattr(module, "weight_v") and hasattr(module, "weight_g"):
                    # 使用 gather 函数初始化权重
                    with deepspeed.zero.GatheredParameters([module.weight_v, module.weight_g], modifier_rank=0):
                        nn.init.kaiming_normal_(module.weight.data)
                else:
                    # 使用 gather 函数初始化权重
                    with deepspeed.zero.GatheredParameters(module.weight, modifier_rank=0):
                        nn.init.kaiming_normal_(module.weight.data)
            else:
                # 使用 Kaiming 初始化权重
                nn.init.kaiming_normal_(module.weight.data)
        # 如果是嵌入层
        elif isinstance(module, nn.Embedding):
            # 初始化嵌入层权重，使用正态分布
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果有填充索引，将填充索引处的权重初始化为0
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

        # 如果是线性层或一维卷积层且有偏置
        if isinstance(module, (nn.Linear, nn.Conv1d)) and module.bias is not None:
            # 初始化偏置为零
            module.bias.data.zero_()

    def _get_feat_extract_output_lengths(self, input_lengths: Union[torch.LongTensor, int]):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            # 从 https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html 中取得的一维卷积层输出长度公式
            return torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1

        # 遍历卷积核大小和步长，计算卷积层的输出长度
        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        # 返回计算得到的输入长度
        return input_lengths
        # 获取特征向量长度和注意力掩码，返回注意力掩码
        def _get_feature_vector_attention_mask(self, feature_vector_length: int, attention_mask: torch.LongTensor):
            # 计算输出长度
            output_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)
            # 获取批处理大小
            batch_size = attention_mask.shape[0]

            # 重新初始化注意力掩码，确保所有输出长度前的数值都被关注
            attention_mask = torch.zeros(
                (batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
            )
            # 这两个操作确保输出长度前的所有值都被关注
            attention_mask[(torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1)] = 1
            # 反转注意力掩码，累积求和，然后再次反转，最后将其转换为布尔值
            attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
            # 返回注意力掩码
            return attention_mask
# SEWD 模型的文档字符串，描述了该模型的来源、继承关系和参数说明
SEWD_START_DOCSTRING = r"""
    SEW-D was proposed in [Performance-Efficiency Trade-offs in Unsupervised Pre-training for Speech
    Recognition](https://arxiv.org/abs/2109.06870) by Felix Wu, Kwangyoun Kim, Jing Pan, Kyu Han, Kilian Q. Weinberger,
    Yoav Artzi.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving etc.).

    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`SEWDConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# SEWD 模型的输入文档字符串，描述了输入参数的含义
SEWD_INPUTS_DOCSTRING = r"""
    Args:
        input_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Float values of input raw speech waveform. Values can be obtained by loading a `.flac` or `.wav` audio file
            into an array of type `List[float]` or a `numpy.ndarray`, *e.g.* via the soundfile library (`pip install
            soundfile`). To prepare the array into `input_values`, the [`AutoProcessor`] should be used for padding and
            conversion into a tensor of type `torch.FloatTensor`. See [`Wav2Vec2Processor.__call__`] for details.
        attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing convolution and attention on padding token indices. Mask values selected in `[0,
            1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)

        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""

# 添加 SEWD 模型文档字符串的装饰器，描述了模型的基本信息和参数说明
@add_start_docstrings(
    "The bare SEW-D Model transformer outputting raw hidden-states without any specific head on top.",
    SEWD_START_DOCSTRING,
)
# 定义 SEWDModel 类，继承自 SEWDPreTrainedModel，是 SEWD 模型的主体结构
# SEW->SEWD, layer_norm_eps->feature_layer_norm_eps
class SEWDModel(SEWDPreTrainedModel):
```  
    # 初始化方法，接受一个 SEWDConfig 类型的参数，调用父类的初始化方法
    def __init__(self, config: SEWDConfig):
        # 调用父类的初始化方法，将配置信息传递给父类
        super().__init__(config)
        # 将配置信息保存到对象属性中
        self.config = config
        # 创建 SEWDFeatureEncoder 对象，用于提取特征
        self.feature_extractor = SEWDFeatureEncoder(config)
        # 创建 LayerNorm 对象，用于对特征进行归一化
        self.layer_norm = nn.LayerNorm(config.conv_dim[-1], eps=config.feature_layer_norm_eps)

        # 判断是否需要对特征进行投影
        self.project_features = config.conv_dim[-1] != config.hidden_size
        # 如果需要投影特征
        if self.project_features:
            # 创建 Linear 层，用于特征投影
            self.feature_projection = nn.Linear(config.conv_dim[-1], config.hidden_size)
        # 创建 Dropout 层，用于特征投影后的随机丢弃
        self.feature_dropout = nn.Dropout(config.feat_proj_dropout)

        # 如果配置中的 mask_time_prob 或 mask_feature_prob 大于 0，则初始化一个可学习参数
        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0:
            # 初始化一个 uniform 分布的随机张量，用作嵌入的 masked 特征
            self.masked_spec_embed = nn.Parameter(torch.FloatTensor(config.hidden_size).uniform_())

        # 创建 SEWDEncoder 对象，用于编码输入序列
        self.encoder = SEWDEncoder(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2Model._mask_hidden_states 复制而来的方法
    # 用于对隐藏状态进行遮蔽
    def _mask_hidden_states(
        self,
        hidden_states: torch.FloatTensor,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        """
        根据时间轴和/或特征轴对提取的特征进行掩码处理，根据SpecAugment的方法
        [SpecAugment](https://arxiv.org/abs/1904.08779).
        """

        # `config.apply_spec_augment`可以设置掩码为False
        if not getattr(self.config, "apply_spec_augment", True):
            # 如果不应用SpecAugment，直接返回隐藏状态
            return hidden_states

        # 生成索引并沿时间轴应用SpecAugment
        batch_size, sequence_length, hidden_size = hidden_states.size()

        if mask_time_indices is not None:
            # 使用给定的mask_time_indices沿时间轴应用SpecAugment
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)
        elif self.config.mask_time_prob > 0 and self.training:
            # 根据配置参数计算掩码索引并沿时间轴应用SpecAugment
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
            # 根据配置参数计算掩码索引并沿特征轴应用SpecAugment
            mask_feature_indices = _compute_mask_indices(
                (batch_size, hidden_size),
                mask_prob=self.config.mask_feature_prob,
                mask_length=self.config.mask_feature_length,
                min_masks=self.config.mask_feature_min_masks,
            )
            mask_feature_indices = torch.tensor(mask_feature_indices, device=hidden_states.device, dtype=torch.bool)
            mask_feature_indices = mask_feature_indices[:, None].expand(-1, sequence_length, -1) # 沿特征轴扩展掩码索引
            hidden_states[mask_feature_indices] = 0

        # 返回处理后的隐藏状态
        return hidden_states

    @add_start_docstrings_to_model_forward(SEWD_INPUTS_DOCSTRING) # 添加model_forward方法的说明文档
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=BaseModelOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="audio",
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        input_values: Optional[torch.Tensor], # 输入值
        attention_mask: Optional[torch.Tensor] = None, # 注意力掩码
        mask_time_indices: Optional[torch.FloatTensor] = None, # 时间轴掩码索引
        output_attentions: Optional[bool] = None, # 是否输出注意力权重
        output_hidden_states: Optional[bool] = None, # 是否输出隐藏状态
        return_dict: Optional[bool] = None, # 是否返回字典形式的输出
        # 设置输出注意力的选项，如果未指定则使用配置中的默认设置
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 设置输出隐藏状态的选项，如果未指定则使用配置中的默认设置
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 设置返回字典选项，如果未指定则使用配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 提取特征
        extract_features = self.feature_extractor(input_values)
        # 转置特征，将第1和第2维交换
        extract_features = extract_features.transpose(1, 2)
        # 对特征进行层归一化
        extract_features = self.layer_norm(extract_features)

        # 如果需要对特征进行投影
        if self.project_features:
            extract_features = self.feature_projection(extract_features)
        # 对特征进行特征丢弃
        hidden_states = self.feature_dropout(extract_features)

        # 如果有注意力掩码
        if attention_mask is not None:
            # 计算对应于特征向量的降维注意力掩码
            attention_mask = self._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)

        # 对隐藏状态进行掩码
        hidden_states = self._mask_hidden_states(hidden_states, mask_time_indices=mask_time_indices)

        # 对隐藏状态进行编码
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 更新隐藏状态
        hidden_states = encoder_outputs[0]

        # 如果不返回字典形式
        if not return_dict:
            # 返回隐藏状态和其他输出
            return (hidden_states,) + encoder_outputs[1:]

        # 返回基础模型输出对象
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
@add_start_docstrings(
    """SEW-D Model with a `language modeling` head on top for Connectionist Temporal Classification (CTC).""",
    SEWD_START_DOCSTRING,
)
# 定义一个 SEWDForCTC 类，该类包含一个 SEWDModel 和一个用于语言建模的头部
# SEWD_START_DOCSTRING 是从前面导入的一个文档字符串的变量
class SEWDForCTC(SEWDPreTrainedModel):
    def __init__(self, config, target_lang: Optional[str] = None):
        # 调用父类的构造函数
        super().__init__(config)

        # 初始化 SEWDModel
        self.sew_d = SEWDModel(config)
        # 初始化 dropout 层
        self.dropout = nn.Dropout(config.final_dropout)

        # 设置目标语言属性
        self.target_lang = target_lang

        # 检查配置中是否定义了词汇表大小，若未定义则引发异常
        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `SEWDForCTC.from_pretrained(..., vocab_size=vocab_size)`. "
                "or define `vocab_size` of your model's configuration."
            )
        # 计算输出隐藏大小
        output_hidden_size = (
            config.output_hidden_size if hasattr(config, "add_adapter") and config.add_adapter else config.hidden_size
        )
        # 初始化语言模型头部的线性层
        self.lm_head = nn.Linear(output_hidden_size, config.vocab_size)

        # 初始化权重并应用最终处理
        self.post_init()

    def tie_weights(self):
        """
        This method overwrites [`~PreTrainedModel.tie_weights`] so that adapter weights can be correctly loaded when
        passing `target_lang=...` to `from_pretrained(...)`.

        This method is **not** supposed to be called by the user and is prone to be changed in the future.
        """

        # 该方法覆盖了 [`~PreTrainedModel.tie_weights`] 方法，以便在传递 `target_lang=...` 到 `from_pretrained(...)` 时正确加载适配器权重。

        # 这个方法**不**应该由用户调用，并且可能在将来被更改。
        """

        # 注意，`tie_weights` 通常用于绑定输入和输出嵌入权重。该方法被重新用于正确加载 SEWD 的适配器层，
        # 这样我们就不必为 [`PreTrainedModel`] 引入新的 API。
        # 虽然有点 hacky，但 SEWD 永远不必绑定输入和输出嵌入，因此在这里重新使用这个函数是可以的。

        # 获取目标语言属性
        target_lang = self.target_lang

        # 如果目标语言不是 None，并且配置中未定义适配器的注意力维度，则引发异常
        if target_lang is not None and getattr(self.config, "adapter_attn_dim", None) is None:
            raise ValueError(f"Cannot pass `target_lang`: {target_lang} if `config.adapter_attn_dim` is not defined.")
        # 如果目标语言是 None，并且配置中定义了适配器的注意力维度，则记录信息
        elif target_lang is None and getattr(self.config, "adapter_attn_dim", None) is not None:
            logger.info("By default `target_lang` is set to 'eng'.")
        # 如果目标语言不是 None
        elif target_lang is not None:
            # 加载适配器
            self.load_adapter(target_lang, force_load=True)
    def freeze_feature_extractor(self):
        """
        调用此函数将禁用特征编码器的梯度计算，使其参数在训练期间不会被更新。
        """
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. "
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        # 调用等效的 freeze_feature_encoder 方法
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        """
        调用此函数将禁用特征编码器的梯度计算，使其参数在训练期间不会被更新。
        """
        # 冻结特征编码器的参数
        self.sew_d.feature_extractor._freeze_parameters()

    def freeze_base_model(self):
        """
        调用此函数将禁用基础模型的梯度计算，使其参数在训练期间不会被更新。只有分类头将被更新。
        """
        # 将所有模型参数的 requires_grad 设置为 False，即禁用它们的梯度计算
        for param in self.sew_d.parameters():
            param.requires_grad = False

    @add_start_docstrings_to_model_forward(SEWD_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=CausalLMOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_CTC_EXPECTED_OUTPUT,
        expected_loss=_CTC_EXPECTED_LOSS,
    )
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    def forward(
        self,
        input_values: Union[torch.Tensor, Dict[str, torch.Tensor]],
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional`):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 获取模型的输出结果
        outputs = self.sew_d(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取隐藏状态
        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        # 根据隐藏状态获取logits
        logits = self.lm_head(hidden_states)

        # 初始化损失为None
        loss = None
        # 如果存在标签
        if labels is not None:
            # 如果最大标签值超过了词汇表大小
            if labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

            # 从attention_mask中取出输入长度信息
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            # 假设填充的标记用-100表示不被注意到
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # ctc_loss不支持fp16
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

            # 禁用cudnn以防止无限值
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

        # 如果不需要返回字典
        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        # 返回CausalLMOutput对象
        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )
# 添加头部文档字符串，描述了 SEWD 模型的作用，用于序列分类任务，如 SUPERB Keyword Spotting
# 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForSequenceClassification 复制而来，修改了 Wav2Vec2 为 SEWD，wav2vec2 为 sew_d，WAV_2_VEC_2 为 SEWD
class SEWDForSequenceClassification(SEWDPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 检查是否启用 adapter，如果是，抛出数值错误
        if hasattr(config, "add_adapter") and config.add_adapter:
            raise ValueError("Sequence classification does not support the use of SEWD adapters (config.add_adapter=True)")
        
        # 创建 SEWDModel 对象
        self.sew_d = SEWDModel(config)
        # 计算层数（transformer 层 + 输入嵌入层）
        num_layers = config.num_hidden_layers + 1
        # 如果使用加权层求和，创建层权重参数
        if config.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        
        # 创建线性投影层
        self.projector = nn.Linear(config.hidden_size, config.classifier_proj_size)
        # 创建分类器层
        self.classifier = nn.Linear(config.classifier_proj_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 冻结特征提取器，禁用梯度计算，参数在训练期间不会更新
    def freeze_feature_extractor(self):
        warnings.warn("The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. Please use the equivalent `freeze_feature_encoder` method instead.", FutureWarning)
        self.freeze_feature_encoder()

    # 冻结特征编码器，禁用梯度计算，参数在训练期间不会更新
    def freeze_feature_encoder(self):
        self.sew_d.feature_extractor._freeze_parameters()

    # 冻结基础模型，禁用梯度计算，参数在训练期间不会更新，只有分类头会更新
    def freeze_base_model(self):
        for param in self.sew_d.parameters():
            param.requires_grad = False

    # 添加向前模型方法的文档字符串
    # 添加代码示例的文档字符串
    # 返回检查点、输出类型、配置类、模态（音频）、预期输出、预期损失
        def forward(
            self,
            input_values: Optional[torch.Tensor],  # 输入值的张量，可选
            attention_mask: Optional[torch.Tensor] = None,  # 注意力遮罩的张量，可选，默认为None
            output_attentions: Optional[bool] = None,  # 是否输出注意力矩阵的布尔值，可选，默认为None
            output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态的布尔值，可选，默认为None
            return_dict: Optional[bool] = None,  # 是否返回字典的布尔值，可选，默认为None
            labels: Optional[torch.Tensor] = None,  # 用于计算序列分类/回归损失的标签张量，可选，默认为None
        ) -> Union[Tuple, SequenceClassifierOutput]:  # 函数返回值的类型注释
            r"""
            labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
                config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
                `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
            """

            return_dict = return_dict if return_dict is not None else self.config.use_return_dict  # 如果return_dict不为None，则使用return_dict；否则使用self.config.use_return_dict
            output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states  # 如果self.config.use_weighted_layer_sum为True，则output_hidden_states为True；否则使用output_hidden_states

            outputs = self.sew_d(  # 调用self.sew_d方法，并传入相应参数
                input_values,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

            if self.config.use_weighted_layer_sum:  # 如果self.config.use_weighted_layer_sum为True
                hidden_states = outputs[_HIDDEN_STATES_START_POSITION]  # 获取outputs中的隐藏状态
                hidden_states = torch.stack(hidden_states, dim=1)  # 在维度1上对隐藏状态进行堆叠
                norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)  # 对self.layer_weights进行softmax
                hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)  # 计算加权和
            else:  # 如果self.config.use_weighted_layer_sum为False
                hidden_states = outputs[0]  # 获取outputs中的第一个元素作为隐藏状态

            hidden_states = self.projector(hidden_states)  # 通过self.projector对隐藏状态进行处理
            if attention_mask is None:  # 如果attention_mask为None
                pooled_output = hidden_states.mean(dim=1)  # 对隐藏状态进行平均池化
            else:  # 如果attention_mask不为None
                padding_mask = self._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)  # 获取特征向量的注意力遮罩
                hidden_states[~padding_mask] = 0.0  # 将注意力遮罩外的隐藏状态置为0
                pooled_output = hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)  # 计算池化输出

            logits = self.classifier(pooled_output)  # 使用self.classifier对池化输出进行分类预测

            loss = None  # 初始化损失值为None
            if labels is not None:  # 如果标签不为None
                loss_fct = CrossEntropyLoss()  # 初始化交叉熵损失函数
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))  # 计算损失值

            if not return_dict:  # 如果不返回字典
                output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]  # 将logits和outputs中隐藏状态起始位置之后的元素组成元组
                return ((loss,) + output) if loss is not None else output  # 如果损失值不为None，则返回包含损失值和输出的元组；否则仅返回输出

            return SequenceClassifierOutput(  # 返回序列分类器输出对象
                loss=loss,  # 损失值
                logits=logits,  # 预测值
                hidden_states=outputs.hidden_states,  # 隐藏状态
                attentions=outputs.attentions,  # 注意力矩阵
            )
```