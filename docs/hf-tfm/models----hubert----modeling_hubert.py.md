# `.\models\hubert\modeling_hubert.py`

```
# 设置文件编码为 UTF-8
# 版权声明
# 2021 年版权归 Fairseq 作者和 HuggingFace 公司团队所有。保留所有权利。
#
# 根据 Apache 许可证 2.0 版本（“许可证”）授权；
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证的副本
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 均基于“原样”分发，不附带任何明示或暗示的担保或条件。
# 请查看许可证以获取特定语言的权限和限制。
""" PyTorch Hubert 模型。"""

# 引入警告模块
import warnings
# 引入类型提示相关模块
from typing import Optional, Tuple, Union

# 引入 numpy 和 torch 模块
import numpy as np
import torch
import torch.utils.checkpoint
# 引入 nn 模块
from torch import nn
# 引入交叉熵损失函数
from torch.nn import CrossEntropyLoss

# 引入相关模块和类
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_outputs import BaseModelOutput, CausalLMOutput, SequenceClassifierOutput
from ...modeling_utils import PreTrainedModel
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
# 引入 Hubert 配置类
from .configuration_hubert import HubertConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 隐藏状态的起始位置
_HIDDEN_STATES_START_POSITION = 1

# 通用文档字符串
_CONFIG_FOR_DOC = "HubertConfig"

# 基本文档字符串
_CHECKPOINT_FOR_DOC = "facebook/hubert-large-ls960-ft"
_EXPECTED_OUTPUT_SHAPE = [1, 292, 768]

# CTC 文档字符串
_CTC_EXPECTED_OUTPUT = "'MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL'"
_CTC_EXPECTED_LOSS = 22.68

# 音频分类文档字符串
_SEQ_CLASS_CHECKPOINT = "superb/hubert-base-superb-ks"
_SEQ_CLASS_EXPECTED_OUTPUT = "'_unknown_'"
_SEQ_CLASS_EXPECTED_LOSS = 8.53

# Hubert 预训练模型列表
HUBERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/hubert-base-ls960",
    # 查看所有 Hubert 模型：https://huggingface.co/models?filter=hubert
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
    计算给定形状的随机掩码范围。用于实现 ASR 的简单数据增强方法 [SpecAugment: A Simple Data Augmentation Method for ASR](https://arxiv.org/abs/1904.08779)。
    请注意，此方法未经过优化，应在 CPU 上作为训练期间的预处理的一部分运行，而不是在 TPU 上运行。
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
# 定义一个类，继承自 nn.Module，用于实现不带 LayerNorm 的卷积层，类名为 HubertNoLayerNormConvLayer
class HubertNoLayerNormConvLayer(nn.Module):
    # 初始化方法，接受配置和层编号作为参数
    def __init__(self, config, layer_id=0):
        super().__init__()
        # 获取输入和输出卷积维度
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        # 创建一个一维卷积层，设置输入和输出维度、卷积核大小、步长和偏置
        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        # 获取激活函数
        self.activation = ACT2FN[config.feat_extract_activation]

    # 前向传播方法，接受隐藏状态作为输入
    def forward(self, hidden_states):
        # 对隐藏状态进行卷积操作
        hidden_states = self.conv(hidden_states)
        # 对卷积结果应用激活函数
        hidden_states = self.activation(hidden_states)
        # 返回处理后的隐藏状态
        return hidden_states


# 定义一个类，继承自 nn.Module，用于实现带 LayerNorm 的卷积层，类名为 HubertLayerNormConvLayer
class HubertLayerNormConvLayer(nn.Module):
    # 初始化方法，接受配置和层编号作为参数
    def __init__(self, config, layer_id=0):
        super().__init__()
        # 获取输入和输出卷积维度
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        # 创建一个一维卷积层，设置输入和输出维度、卷积核大小、步长和偏置
        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        # 创建一个 LayerNorm 层，设置输出维度和启用元素级别的仿射变换
        self.layer_norm = nn.LayerNorm(self.out_conv_dim, elementwise_affine=True)
        # 获取激活函数
        self.activation = ACT2FN[config.feat_extract_activation]

    # 前向传播方法，接受隐藏状态作为输入
    def forward(self, hidden_states):
        # 对隐藏状态进行卷积操作
        hidden_states = self.conv(hidden_states)

        # 转置隐藏状态的倒数第二和倒数第一维
        hidden_states = hidden_states.transpose(-2, -1)
        # 对转置后的隐藏状态应用 LayerNorm
        hidden_states = self.layer_norm(hidden_states)
        # 再次转置隐藏状态的倒数第二和倒数第一维
        hidden_states = hidden_states.transpose(-2, -1)

        # 对处理后的隐藏状态应用激活函数
        hidden_states = self.activation(hidden_states)
        # 返回处理后的隐藏状态
        return hidden_states


# 定义一个类，继承自 nn.Module，用于实现带 GroupNorm 的卷积层，类名为 HubertGroupNormConvLayer
class HubertGroupNormConvLayer(nn.Module):
    # 初始化方法，接受配置和层编号作为参数
    def __init__(self, config, layer_id=0):
        super().__init__()
        # 获取输入和输出卷积维度
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        # 创建一个一维卷积层，设置输入和输出维度、卷积核大小、步长和偏置
        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        # 获取激活函数
        self.activation = ACT2FN[config.feat_extract_activation]

        # 创建一个 GroupNorm 层，设置组数为输出维度，通道数为输出维度，启用仿射变换
        self.layer_norm = nn.GroupNorm(num_groups=self.out_conv_dim, num_channels=self.out_conv_dim, affine=True)

    # 前向传播方法，接受隐藏状态作为输入
    def forward(self, hidden_states):
        # 对隐藏状态进行卷积操作
        hidden_states = self.conv(hidden_states)
        # 对卷积结果应用 GroupNorm
        hidden_states = self.layer_norm(hidden_states)
        # 对处理后的隐藏状态应用激活函数
        hidden_states = self.activation(hidden_states)
        # 返回处理后的隐藏状态
        return hidden_states
# 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2PositionalConvEmbedding复制代码，并将Wav2Vec2->Hubert
class HubertPositionalConvEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个一维卷积层，用于位置编码
        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.num_conv_pos_embeddings,
            padding=config.num_conv_pos_embeddings // 2,
            groups=config.num_conv_pos_embedding_groups,
        )

        # 初始化权重归一化函数
        weight_norm = nn.utils.weight_norm
        if hasattr(nn.utils.parametrizations, "weight_norm"):
            weight_norm = nn.utils.parametrizations.weight_norm

        # 如果启用了deepspeed zero3，则使用gathered parameters和weight norm
        if is_deepspeed_zero3_enabled():
            import deepspeed

            with deepspeed.zero.GatheredParameters(self.conv.weight, modifier_rank=0):
                self.conv = weight_norm(self.conv, name="weight", dim=2)
            deepspeed.zero.register_external_parameter(self, self.conv.weight_v)
            deepspeed.zero.register_external_parameter(self, self.conv.weight_g)
        else:
            self.conv = weight_norm(self.conv, name="weight", dim=2)

        # 创建一个用于填充的层
        self.padding = HubertSamePadLayer(config.num_conv_pos_embeddings)
        # 激活函数
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        hidden_states = hidden_states.transpose(1, 2)

        hidden_states = self.conv(hidden_states)
        hidden_states = self.padding(hidden_states)
        hidden_states = self.activation(hidden_states)

        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states


# 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2SamePadLayer复制代码，并将Wav2Vec2->Hubert
class HubertSamePadLayer(nn.Module):
    def __init__(self, num_conv_pos_embeddings):
        super().__init__()
        # 计算需要移除的填充数量
        self.num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0

    def forward(self, hidden_states):
        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, :, : -self.num_pad_remove]
        return hidden_states


# 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2FeatureEncoder复制代码，并将Wav2Vec2->Hubert
class HubertFeatureEncoder(nn.Module):
    """Construct the features from raw audio waveform"""
    # 初始化函数，接受配置参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()

        # 根据配置参数中的特征提取规范选择不同的卷积层
        if config.feat_extract_norm == "group":
            # 如果特征提取规范为"group"，则创建一组带有组归一化的卷积层
            conv_layers = [HubertGroupNormConvLayer(config, layer_id=0)] + [
                HubertNoLayerNormConvLayer(config, layer_id=i + 1) for i in range(config.num_feat_extract_layers - 1)
            ]
        elif config.feat_extract_norm == "layer":
            # 如果特征提取规范为"layer"，则创建一组带有层归一化的卷积层
            conv_layers = [HubertLayerNormConvLayer(config, layer_id=i) for i in range(config.num_feat_extract_layers)]
        else:
            # 如果特征提取规范既不是"group"也不是"layer"，则抛出数值错误
            raise ValueError(
                f"`config.feat_extract_norm` is {config.feat_extract_norm}, but has to be one of ['group', 'layer']"
            )
        # 将卷积层列表转换为模块列表
        self.conv_layers = nn.ModuleList(conv_layers)
        # 初始化梯度检查点为False
        self.gradient_checkpointing = False
        # 初始化_requires_grad为True
        self._requires_grad = True

    # 冻结参数的函数
    def _freeze_parameters(self):
        # 遍历所有参数，将其requires_grad属性设置为False
        for param in self.parameters():
            param.requires_grad = False
        # 将_requires_grad属性设置为False
        self._requires_grad = False

    # 前向传播函数
    def forward(self, input_values):
        # 将输入值添加一个维度
        hidden_states = input_values[:, None]

        # 确保hidden_states需要梯度以进行梯度检查点
        if self._requires_grad and self.training:
            hidden_states.requires_grad = True

        # 遍历所有卷积层
        for conv_layer in self.conv_layers:
            # 如果_requires_grad为True且gradient_checkpointing为True且处于训练状态
            if self._requires_grad and self.gradient_checkpointing and self.training:
                # 使用梯度检查点函数对卷积层进行操作
                hidden_states = self._gradient_checkpointing_func(
                    conv_layer.__call__,
                    hidden_states,
                )
            else:
                # 否则直接对卷积层进行操作
                hidden_states = conv_layer(hidden_states)

        # 返回隐藏状态
        return hidden_states
class HubertFeatureExtractor(HubertFeatureEncoder):
    # HubertFeatureExtractor 类继承自 HubertFeatureEncoder 类
    def __init__(self, config):
        # 初始化方法，接受一个配置参数
        super().__init__(config)
        # 调用父类的初始化方法
        warnings.warn(
            f"The class `{self.__class__.__name__}` has been depreciated "
            "and will be removed in Transformers v5. "
            f"Use `{self.__class__.__bases__[0].__name__}` instead.",
            FutureWarning,
        )
        # 发出警告，提示该类已被弃用，将在 Transformers v5 中移除

class HubertFeatureProjection(nn.Module):
    # HubertFeatureProjection 类继承自 nn.Module 类
    def __init__(self, config):
        # 初始化方法，接受一个配置参数
        super().__init__()
        # 调用父类的初始化方法
        self.feat_proj_layer_norm = config.feat_proj_layer_norm
        # 从配置参数中获取特征投影层是否使用 LayerNorm
        if self.feat_proj_layer_norm:
            # 如果特征投影层使用 LayerNorm
            self.layer_norm = nn.LayerNorm(config.conv_dim[-1], eps=config.layer_norm_eps)
            # 初始化 LayerNorm 层
        self.projection = nn.Linear(config.conv_dim[-1], config.hidden_size)
        # 初始化线性投影层
        self.dropout = nn.Dropout(config.feat_proj_dropout)
        # 初始化 Dropout 层

    def forward(self, hidden_states):
        # 前向传播方法，接受隐藏状态作为输入
        if self.feat_proj_layer_norm:
            # 如果特征投影层使用 LayerNorm
            hidden_states = self.layer_norm(hidden_states)
            # 对隐藏状态进行 LayerNorm 处理
        hidden_states = self.projection(hidden_states)
        # 对隐藏状态进行线性投影
        hidden_states = self.dropout(hidden_states)
        # 对投影后的隐藏状态进行 Dropout
        return hidden_states
        # 返回处理后的隐藏状态

# Copied from transformers.models.bart.modeling_bart.BartAttention with Bart->Hubert
# 从 transformers.models.bart.modeling_bart.BartAttention 复制并将 Bart 替换为 Hubert
class HubertAttention(nn.Module):
    # HubertAttention 类继承自 nn.Module 类
    """Multi-headed attention from 'Attention Is All You Need' paper"""
    # 多头注意力机制，来源于 'Attention Is All You Need' 论文

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[HubertConfig] = None,
    ):
        # 初始化方法，接受多个参数
        super().__init__()
        # 调用父类的初始化方法
        self.embed_dim = embed_dim
        # 设置嵌入维度
        self.num_heads = num_heads
        # 设置头数
        self.dropout = dropout
        # 设置 Dropout 概率
        self.head_dim = embed_dim // num_heads
        # 计算每个头的维度
        self.config = config
        # 设置配置参数

        if (self.head_dim * num_heads) != self.embed_dim:
            # 如果头维度乘以头数不等于嵌入维度
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
            # 抛出数值错误异常
        self.scaling = self.head_dim**-0.5
        # 计算缩放因子
        self.is_decoder = is_decoder
        # 是否为解码器
        self.is_causal = is_causal
        # 是否为因果注意力

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        # 初始化键的投影层
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        # 初始化值的投影层
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        # 初始化查询的投影层
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        # 初始化输出的投影层

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        # 定义一个方法，用于调整张量的形状
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        # 调整张量的形状并返回

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        # 前向传播方法，接受多个参数
# 定义一个名为HubertFeedForward的类，继承自nn.Module类，用于实现前馈神经网络
class HubertFeedForward(nn.Module):
    # 初始化方法，接受config参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个Dropout层，用于中间层的dropout
        self.intermediate_dropout = nn.Dropout(config.activation_dropout)
        # 创建一个全连接层，将隐藏层的大小转换为中间层的大小
        self.intermediate_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 根据配置中的激活函数类型，选择对应的激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
        # 创建一个全连接层，将中间层的大小转换为隐藏层的大小
        self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建一个Dropout层，用于输出层的dropout
        self.output_dropout = nn.Dropout(config.hidden_dropout)

    # 前向传播方法，接受隐藏状态作为输入，返回处理后的隐藏状态
    def forward(self, hidden_states):
        # 中间层的全连接操作
        hidden_states = self.intermediate_dense(hidden_states)
        # 中间层的激活函数操作
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 中间层的dropout操作
        hidden_states = self.intermediate_dropout(hidden_states)

        # 输出层的全连接操作
        hidden_states = self.output_dense(hidden_states)
        # 输出层的dropout操作
        hidden_states = self.output_dropout(hidden_states)
        # 返回处理后的隐藏状态
        return hidden_states


# 定义一个名为HubertEncoderLayer的类，继承自nn.Module类，用于实现编码器层
class HubertEncoderLayer(nn.Module):
    # 初始化方法，接受config参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个HubertAttention层，用于实现注意力机制
        self.attention = HubertAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
        )
        # 创建一个Dropout层，用于隐藏层的dropout
        self.dropout = nn.Dropout(config.hidden_dropout)
        # 创建一个LayerNorm层，用于隐藏层的Layer Normalization
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个HubertFeedForward层，用于实现前馈神经网络
        self.feed_forward = HubertFeedForward(config)
        # 创建一个LayerNorm层，用于最终隐藏层的Layer Normalization
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    # 前向传播方法，接受隐藏状态、注意力掩码等参数作为输入，返回处理后的隐藏状态和可能的注意力权重
    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        # 保存注意力机制之前的隐藏状态
        attn_residual = hidden_states
        # 调用注意力机制层，获取处理后的隐藏状态、注意力权重等信息
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
        )
        # 隐藏层的dropout操作
        hidden_states = self.dropout(hidden_states)
        # 将注意力机制之前的隐藏状态与处理后的隐藏状态相加
        hidden_states = attn_residual + hidden_states

        # 隐藏层的Layer Normalization操作
        hidden_states = self.layer_norm(hidden_states)
        # 隐藏层的前馈神经网络操作
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        # 最终隐藏层的Layer Normalization操作
        hidden_states = self.final_layer_norm(hidden_states)

        # 构建输出元组
        outputs = (hidden_states,)

        # 如果需要输出注意力权重信息，则添加到输出元组中
        if output_attentions:
            outputs += (attn_weights,)

        # 返回输出元组
        return outputs


# 定义一个名为HubertAttnAdapterLayer的类，继承自nn.Module类
class HubertAttnAdapterLayer(nn.Module):
    def __init__(self, config):
        """
        Implements adapter modules directly with 3D tensor weight as parameters and without using ModuleList to speed
        up training throughput.
        """
        # 初始化 AdapterModule 类
        super().__init__()
        # 从配置中获取适配器的注意力维度和隐藏维度
        self.input_dim = config.adapter_attn_dim
        self.hidden_dim = config.hidden_size

        # 初始化 LayerNorm 层
        self.norm = nn.LayerNorm(self.hidden_dim)
        # 初始化线性层1，将隐藏维度映射到输入维度
        self.linear_1 = nn.Linear(self.hidden_dim, self.input_dim)
        # 初始化激活函数为 ReLU
        self.act_fn = nn.ReLU()
        # 初始化线性层2，将输入维度映射回隐藏维度
        self.linear_2 = nn.Linear(self.input_dim, self.hidden_dim)

    def forward(self, hidden_states: torch.FloatTensor):
        # 对隐藏状态进行 LayerNorm 处理
        hidden_states = self.norm(hidden_states)

        # 线性映射1
        hidden_states = self.linear_1(hidden_states)
        # 使用激活函数 ReLU
        hidden_states = self.act_fn(hidden_states)
        # 线性映射2
        hidden_states = self.linear_2(hidden_states)

        # 返回处理后的隐藏状态
        return hidden_states
# 定义一个名为HubertEncoderLayerStableLayerNorm的类，继承自nn.Module
class HubertEncoderLayerStableLayerNorm(nn.Module):
    # 初始化方法，接受config参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 创建HubertAttention对象，传入相关参数
        self.attention = HubertAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
        )
        # 创建Dropout层，传入隐藏层的dropout参数
        self.dropout = nn.Dropout(config.hidden_dropout)
        # 创建LayerNorm层，传入隐藏层的大小和epsilon值
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建HubertFeedForward对象，传入config参数
        self.feed_forward = HubertFeedForward(config)
        # 创建LayerNorm层，传入隐藏层的大小和epsilon值
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 如果config中存在adapter_attn_dim属性，则创建HubertAttnAdapterLayer对象，否则为None
        if getattr(config, "adapter_attn_dim", None) is not None:
            self.adapter_layer = HubertAttnAdapterLayer(config)
        else:
            self.adapter_layer = None

    # 前向传播方法，接受hidden_states、attention_mask、output_attentions等参数
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        # 保存注意力机制之前的hidden_states
        attn_residual = hidden_states
        # 对hidden_states进行LayerNorm
        hidden_states = self.layer_norm(hidden_states)
        # 调用attention方法，得到新的hidden_states、注意力权重和额外信息
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
        )
        # 对新的hidden_states进行dropout
        hidden_states = self.dropout(hidden_states)
        # 将原始hidden_states和新的hidden_states相加
        hidden_states = attn_residual + hidden_states
        # 对hidden_states进行LayerNorm和FeedForward操作
        hidden_states = hidden_states + self.feed_forward(self.final_layer_norm(hidden_states))

        # 如果存在adapter_layer，则对hidden_states进行adapter_layer操作
        if self.adapter_layer is not None:
            hidden_states = hidden_states + self.adapter_layer(hidden_states)

        # 构建输出元组
        outputs = (hidden_states,)

        # 如果output_attentions为True，则将注意力权重加入到输出元组中
        if output_attentions:
            outputs += (attn_weights,)

        # 返回输出元组
        return outputs


# 定义一个名为HubertEncoder的类，继承自nn.Module
class HubertEncoder(nn.Module):
    # 初始化方法，接受config参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 保存config参数
        self.config = config
        # 创建HubertPositionalConvEmbedding对象，传入config参数
        self.pos_conv_embed = HubertPositionalConvEmbedding(config)
        # 创建LayerNorm层，传入隐藏层的大小和epsilon值
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建Dropout层，传入隐藏层的dropout参数
        self.dropout = nn.Dropout(config.hidden_dropout)
        # 创建包含多个HubertEncoderLayer对象的ModuleList
        self.layers = nn.ModuleList([HubertEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        # 设置梯度检查点为False
        self.gradient_checkpointing = False

    # 前向传播方法，接受hidden_states、attention_mask等参数
    def forward(
        self,
        hidden_states: torch.tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        ):
        # 初始化隐藏状态和注意力矩阵的元组，根据输出设置
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None

        if attention_mask is not None:
            # 确保填充的标记输出为0
            expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
            hidden_states[~expand_attention_mask] = 0

            # 扩展注意力掩码
            attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
            attention_mask = attention_mask.expand(
                attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
            )

        # 通过位置卷积嵌入层处理隐藏状态
        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # 检查是否启用deepspeed zero3
        deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()

        # 遍历每个层
        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 添加LayerDrop（参见https://arxiv.org/abs/1909.11556）
            dropout_probability = torch.rand([])

            skip_the_layer = True if self.training and (dropout_probability < self.config.layerdrop) else False
            if not skip_the_layer or deepspeed_zero3_is_enabled:
                # 在deepspeed zero3下，所有GPU必须同步运行
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        layer.__call__,
                        hidden_states,
                        attention_mask,
                        output_attentions,
                    )
                else:
                    layer_outputs = layer(
                        hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
                    )
                hidden_states = layer_outputs[0]

            if skip_the_layer:
                layer_outputs = (None, None)

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            # 如果不返回字典，则返回非空值的元组
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        # 返回BaseModelOutput对象
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
# 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2EncoderStableLayerNorm复制代码，并将Wav2Vec2替换为Hubert
class HubertEncoderStableLayerNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 初始化位置卷积嵌入层
        self.pos_conv_embed = HubertPositionalConvEmbedding(config)
        # 初始化层归一化
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化dropout
        self.dropout = nn.Dropout(config.hidden_dropout)
        # 初始化编码层列表
        self.layers = nn.ModuleList(
            [HubertEncoderLayerStableLayerNorm(config) for _ in range(config.num_hidden_layers)]
        )
        # 是否启用梯度检查点
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        ):
            # 初始化隐藏状态和注意力矩阵的元组，根据输出设置
            all_hidden_states = () if output_hidden_states else None
            all_self_attentions = () if output_attentions else None

            if attention_mask is not None:
                # 确保填充的标记不被注意到
                expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
                hidden_states[~expand_attention_mask] = 0

                # 扩展注意力掩码
                attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
                attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
                attention_mask = attention_mask.expand(
                    attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
                )

            # 通过位置卷积嵌入层处理隐藏状态
            position_embeddings = self.pos_conv_embed(hidden_states)
            hidden_states = hidden_states + position_embeddings
            hidden_states = self.dropout(hidden_states)

            # 检查是否启用了deepspeed zero3
            deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()

            # 遍历每个层
            for layer in self.layers:
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                # 添加LayerDrop（参考https://arxiv.org/abs/1909.11556）
                dropout_probability = torch.rand([])

                skip_the_layer = True if self.training and (dropout_probability < self.config.layerdrop) else False
                if not skip_the_layer or deepspeed_zero3_is_enabled:
                    # 在deepspeed zero3下，所有GPU必须同步运行
                    if self.gradient_checkpointing and self.training:
                        layer_outputs = self._gradient_checkpointing_func(
                            layer.__call__,
                            hidden_states,
                            attention_mask,
                            output_attentions,
                        )
                    else:
                        layer_outputs = layer(
                            hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
                        )
                    hidden_states = layer_outputs[0]

                if skip_the_layer:
                    layer_outputs = (None, None)

                if output_attentions:
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)

            # 对隐藏状态进行LayerNorm
            hidden_states = self.layer_norm(hidden_states)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if not return_dict:
                # 如果不返回字典，则返回元组
                return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
            # 返回BaseModelOutput对象
            return BaseModelOutput(
                last_hidden_state=hidden_states,
                hidden_states=all_hidden_states,
                attentions=all_self_attentions,
            )
class HubertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 设置配置类为 HubertConfig
    config_class = HubertConfig
    # 设置基础模型前缀为 "hubert"
    base_model_prefix = "hubert"
    # 设置主输入名称为 "input_values"
    main_input_name = "input_values"
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果是线性层
        if isinstance(module, nn.Linear):
            # 使用正态分布初始化权重
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        # 如果是 LayerNorm 或 GroupNorm
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            # 将偏置项初始化为零
            module.bias.data.zero_()
            # 将权重初始化为 1.0
            module.weight.data.fill_(1.0)
        # 如果是 Conv1d 层
        elif isinstance(module, nn.Conv1d):
            # 如果启用了 DeepSpeed Zero3
            if is_deepspeed_zero3_enabled():
                import deepspeed

                # 如果模块有 "weight_v" 和 "weight_g" 属性
                if hasattr(module, "weight_v") and hasattr(module, "weight_g"):
                    # 使用 GatheredParameters 进行初始化
                    with deepspeed.zero.GatheredParameters([module.weight_v, module.weight_g], modifier_rank=0):
                        nn.init.kaiming_normal_(module.weight.data)
                else:
                    # 使用 GatheredParameters 进行初始化
                    with deepspeed.zero.GatheredParameters(module.weight, modifier_rank=0):
                        nn.init.kaiming_normal_(module.weight.data)
            else:
                # 使用 kaiming_normal 初始化权重
                nn.init.kaiming_normal_(module.weight.data)

        # 如果是线性层或 Conv1d 层且有偏置项
        if isinstance(module, (nn.Linear, nn.Conv1d)) and module.bias is not None:
            # 将偏置项初始化为零
            module.bias.data.zero_()

    def _get_feat_extract_output_lengths(self, input_lengths: Union[torch.LongTensor, int]):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D 卷积层输出长度公式取自官方文档
            return torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1

        # 遍历卷积核大小和步长
        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            # 计算输出长度
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        return input_lengths
    # 获取特征向量的注意力掩码，根据特征向量长度和注意力掩码
    def _get_feature_vector_attention_mask(self, feature_vector_length: int, attention_mask: torch.LongTensor):
        # 根据注意力掩码的总和计算输出长度，并转换为长整型
        output_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)
        # 获取批处理大小
        batch_size = attention_mask.shape[0]

        # 创建与注意力掩码相同形状的全零张量
        attention_mask = torch.zeros(
            (batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
        )
        # 确保输出长度之前的所有值都被关注
        attention_mask[(torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1)] = 1
        # 翻转张量，累积求和，再次翻转，并转换为布尔型
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        # 返回注意力掩码
        return attention_mask
# Hubert 模型的文档字符串，包含了模型的介绍、作者信息以及继承关系等
HUBERT_START_DOCSTRING = r"""
    Hubert was proposed in [HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden
    Units](https://arxiv.org/abs/2106.07447) by Wei-Ning Hsu, Benjamin Bolte, Yao-Hung Hubert Tsai, Kushal Lakhotia,
    Ruslan Salakhutdinov, Abdelrahman Mohamed.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving etc.).

    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`HubertConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# Hubert 模型的输入文档字符串，暂时为空
HUBERT_INPUTS_DOCSTRING = r"""
    Args:
        input_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            # 输入原始语音波形的浮点值。值可以通过加载 `.flac` 或 `.wav` 音频文件到 `List[float]` 或 `numpy.ndarray` 类型的数组中获得，
            # 例如通过 soundfile 库 (`pip install soundfile`)。要将数组准备为 `input_values`，应使用 [`AutoProcessor`] 进行填充和转换为 `torch.FloatTensor` 类型的张量。
            # 有关详细信息，请参阅 [`Wav2Vec2Processor.__call__`]。

        attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            # 避免在填充标记索引上执行卷积和注意力的掩码。掩码值选在 `[0, 1]` 之间：
            # - 1 表示**未屏蔽**的标记，
            # - 0 表示**已屏蔽**的标记。
            # [什么是注意力掩码?](../glossary#attention-mask)

            <Tip warning={true}>
            # 只有当相应的处理器具有 `config.return_attention_mask == True` 时才应传递 `attention_mask`。
            # 对于所有处理器具有 `config.return_attention_mask == False` 的模型，例如 [hubert-base](https://huggingface.co/facebook/hubert-base-ls960)，
            # 在进行批量推理时应**不**传递 `attention_mask` 以避免性能下降。对于这些模型，`input_values` 应简单地填充为 0 并在不传递 `attention_mask` 的情况下传递。
            # 请注意，这些模型还会根据 `input_values` 是否填充而产生略有不同的结果。

            </Tip>

        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。有关更多详细信息，请参阅返回的张量下的 `attentions`。

        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。有关更多详细信息，请参阅返回的张量下的 `hidden_states`。

        return_dict (`bool`, *optional*):
            # 是否返回 [`~utils.ModelOutput`] 而不是普通元组。
# 定义 HubertModel 类，继承自 HubertPreTrainedModel 类
@add_start_docstrings(
    "The bare Hubert Model transformer outputting raw hidden-states without any specific head on top.",
    HUBERT_START_DOCSTRING,
)
class HubertModel(HubertPreTrainedModel):
    # 初始化方法，接受一个 HubertConfig 类型的参数
    def __init__(self, config: HubertConfig):
        # 调用父类的初始化方法
        super().__init__(config)
        # 将传入的配置参数保存到实例中
        self.config = config
        # 创建 HubertFeatureEncoder 实例
        self.feature_extractor = HubertFeatureEncoder(config)
        # 创建 HubertFeatureProjection 实例
        self.feature_projection = HubertFeatureProjection(config)

        # 如果配置中有时间或特征掩码的概率大于0，则创建一个可学习的特征嵌入向量
        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0:
            self.masked_spec_embed = nn.Parameter(torch.FloatTensor(config.hidden_size).uniform_())

        # 根据配置选择使用稳定的层归一化或普通的编码器
        if config.do_stable_layer_norm:
            self.encoder = HubertEncoderStableLayerNorm(config)
        else:
            self.encoder = HubertEncoder(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2Model._mask_hidden_states 复制的方法
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
        # 检查配置中是否设置了 apply_spec_augment，如果为 False，则直接返回隐藏状态
        if not getattr(self.config, "apply_spec_augment", True):
            return hidden_states

        # generate indices & apply SpecAugment along time axis
        # 获取隐藏状态的形状信息
        batch_size, sequence_length, hidden_size = hidden_states.size()

        if mask_time_indices is not None:
            # apply SpecAugment along time axis with given mask_time_indices
            # 根据给定的 mask_time_indices 在时间轴上应用 SpecAugment
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)
        elif self.config.mask_time_prob > 0 and self.training:
            # 计算 mask_time_indices 并在训练时应用 SpecAugment
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
            # 计算 mask_feature_indices 并在训练时应用 SpecAugment
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

    @add_start_docstrings_to_model_forward(HUBERT_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=BaseModelOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        """

        Returns:

        Example:

        ```python
        >>> from transformers import AutoProcessor, HubertModel
        >>> from datasets import load_dataset
        >>> import soundfile as sf

        >>> processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
        >>> model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")


        >>> def map_to_array(batch):
        ...     speech, _ = sf.read(batch["file"])
        ...     batch["speech"] = speech
        ...     return batch


        >>> ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        >>> ds = ds.map(map_to_array)

        >>> input_values = processor(ds["speech"][0], return_tensors="pt").input_values  # Batch size 1
        >>> hidden_states = model(input_values).last_hidden_state
        ```"""
        # 设置输出的注意力权重，默认为 None 则使用配置中的值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 设置输出的隐藏状态，默认为 None 则使用配置中的值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 设置返回字典，默认为 None 则使用配置中的值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 提取特征
        extract_features = self.feature_extractor(input_values)
        # 转置特征维度
        extract_features = extract_features.transpose(1, 2)

        if attention_mask is not None:
            # 计算与特征向量对应的减少的注意力掩码
            attention_mask = self._get_feature_vector_attention_mask(extract_features.shape[1], attention_mask)

        # 特征投影
        hidden_states = self.feature_projection(extract_features)
        # 对隐藏状态进行掩码处理
        hidden_states = self._mask_hidden_states(hidden_states, mask_time_indices=mask_time_indices)

        # 编码器输出
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
# 为 Hubert 模型添加 CTC 语言建模头部的类
@add_start_docstrings(
    """Hubert Model with a `language modeling` head on top for Connectionist Temporal Classification (CTC).""",
    HUBERT_START_DOCSTRING,
)
# 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForCTC 复制并修改为 HubertForCTC，wav2vec2->hubert, WAV_2_VEC_2->HUBERT
class HubertForCTC(HubertPreTrainedModel):
    def __init__(self, config, target_lang: Optional[str] = None):
        # 调用父类构造函数
        super().__init__(config)

        # 初始化 Hubert 模型
        self.hubert = HubertModel(config)
        # 添加 dropout 层
        self.dropout = nn.Dropout(config.final_dropout)

        # 设置目标语言
        self.target_lang = target_lang

        # 检查配置中是否定义了词汇表大小
        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `HubertForCTC.from_pretrained(..., vocab_size=vocab_size)`. "
                "or define `vocab_size` of your model's configuration."
            )
        # 根据配置设置输出隐藏层大小
        output_hidden_size = (
            config.output_hidden_size if hasattr(config, "add_adapter") and config.add_adapter else config.hidden_size
        )
        # 添加线性层作为语言模型头部
        self.lm_head = nn.Linear(output_hidden_size, config.vocab_size)

        # 初始化权重并应用最终处理
        self.post_init()

    def tie_weights(self):
        """
        This method overwrites [`~PreTrainedModel.tie_weights`] so that adapter weights can be correctly loaded when
        passing `target_lang=...` to `from_pretrained(...)`.

        This method is **not** supposed to be called by the user and is prone to be changed in the future.
        """

        # 注意，`tie_weights` 通常用于绑定输入和输出嵌入权重。这里重新定义该方法，以便在通过 `from_pretrained(...)` 传递 `target_lang=...` 时可以正确加载适配器层的权重。
        # 这个方法**不**应该由用户调用，并且可能在将来更改。

        # 获取目标语言
        target_lang = self.target_lang

        # 如果目标语言不为空且配置中未定义 `adapter_attn_dim`，则引发错误
        if target_lang is not None and getattr(self.config, "adapter_attn_dim", None) is None:
            raise ValueError(f"Cannot pass `target_lang`: {target_lang} if `config.adapter_attn_dim` is not defined.")
        # 如果目标语言为空且配置中定义了 `adapter_attn_dim`，则记录日志
        elif target_lang is None and getattr(self.config, "adapter_attn_dim", None) is not None:
            logger.info("By default `target_lang` is set to 'eng'.")
        # 如果目标语言不为空，则加载适配器
        elif target_lang is not None:
            self.load_adapter(target_lang, force_load=True)
    # 冻结特征提取器，禁用特征编码器的梯度计算，使其参数在训练期间不会更新
    def freeze_feature_extractor(self):
        # 发出警告，提示方法`freeze_feature_extractor`已被弃用，并将在 Transformers v5 中移除
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. "
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        # 调用等效的`freeze_feature_encoder`方法
        self.freeze_feature_encoder()

    # 冻结特征编码器，禁用特征编码器的梯度计算，使其参数在训练期间不会更新
    def freeze_feature_encoder(self):
        # 冻结特征编码器的参数
        self.hubert.feature_extractor._freeze_parameters()

    # 冻结基础模型，禁用基础模型的梯度计算，使其参数在训练期间不会更新，只有分类头会更新
    def freeze_base_model(self):
        # 遍历 Hubert 模型的参数，将其梯度计算设置为 False
        for param in self.hubert.parameters():
            param.requires_grad = False

    # 前向传播函数，接受输入值、注意力掩码等参数，返回模型输出
    @add_start_docstrings_to_model_forward(HUBERT_INPUTS_DOCSTRING)
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
        ) -> Union[Tuple, CausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """

        # 设置返回字典，如果未指定则使用配置中的返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 Hubert 模型进行推理
        outputs = self.hubert(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取隐藏状态并进行 dropout 处理
        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        # 通过 lm_head 获取 logits
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            if labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

            # 从 attention_mask 中获取 loss 的 input_lengths
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            # 假设填充的标记为 -100，当不被关注时
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # ctc_loss 不支持 fp16
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                # 计算 ctc_loss
                loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )

        if not return_dict:
            # 如果不返回字典，则返回 logits 和其他输出
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        # 返回 CausalLMOutput 对象
        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )
# 使用 Hubert 模型进行序列分类，顶部有一个序列分类头（在池化输出上的线性层），用于类似 SUPERB 关键词识别的任务
@add_start_docstrings(
    """
    Hubert Model with a sequence classification head on top (a linear layer over the pooled output) for tasks like
    SUPERB Keyword Spotting.
    """,
    HUBERT_START_DOCSTRING,
)
# 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForSequenceClassification 复制而来，将 Wav2Vec2->Hubert, wav2vec2->hubert, WAV_2_VEC_2->HUBERT
class HubertForSequenceClassification(HubertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 如果配置中存在 "add_adapter" 属性并且为真，则抛出值错误
        if hasattr(config, "add_adapter") and config.add_adapter:
            raise ValueError(
                "Sequence classification does not support the use of Hubert adapters (config.add_adapter=True)"
            )
        # 创建 HubertModel 对象
        self.hubert = HubertModel(config)
        # 计算层数，包括 transformer 层和输入嵌入层
        num_layers = config.num_hidden_layers + 1
        # 如果配置中使用加权层求和，则初始化权重参数
        if config.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        # 创建线性层，用于投影到分类器投影大小
        self.projector = nn.Linear(config.hidden_size, config.classifier_proj_size)
        # 创建线性层，用于分类
        self.classifier = nn.Linear(config.classifier_proj_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 冻结特征提取器，不计算其梯度
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

    # 冻结特征编码器，不计算其梯度
    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.hubert.feature_extractor._freeze_parameters()

    # 冻结基础模型，不计算其梯度，只更新分类头
    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        for param in self.hubert.parameters():
            param.requires_grad = False

    # 添加模型前向传播的文档字符串
    @add_start_docstrings_to_model_forward(HUBERT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_SEQ_CLASS_CHECKPOINT,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="audio",
        expected_output=_SEQ_CLASS_EXPECTED_OUTPUT,
        expected_loss=_SEQ_CLASS_EXPECTED_LOSS,
    )
    # 定义一个前向传播函数，用于模型的推理过程
    def forward(
        self,
        input_values: Optional[torch.Tensor],  # 输入值，可以为空
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码，可以为空
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，可以为空
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可以为空
        return_dict: Optional[bool] = None,  # 是否返回字典形式的结果，可以为空
        labels: Optional[torch.Tensor] = None,  # 标签，可以为空
    ) -> Union[Tuple, SequenceClassifierOutput]:  # 返回值类型为元组或SequenceClassifierOutput对象

        # 如果未指定返回字典形式的结果，则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 如果使用加权层求和，则输出隐藏状态
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states

        # 调用Hubert模型进行前向传播
        outputs = self.hubert(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 如果使用加权层求和
        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]  # 获取隐藏状态
            hidden_states = torch.stack(hidden_states, dim=1)  # 在指定维度上堆叠隐藏状态
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)  # 对层权重进行softmax归一化
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)  # 加权求和隐藏状态
        else:
            hidden_states = outputs[0]  # 否则直接获取第一个输出作为隐藏状态

        hidden_states = self.projector(hidden_states)  # 投影隐藏状态
        if attention_mask is None:
            pooled_output = hidden_states.mean(dim=1)  # 如果没有注意力掩码，则对隐藏状态进行平均池化
        else:
            padding_mask = self._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)  # 获取特征向量的注意力掩码
            hidden_states[~padding_mask] = 0.0  # 将不需要的部分置零
            pooled_output = hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)  # 对隐藏状态进行池化

        logits = self.classifier(pooled_output)  # 使��分类器对池化后的隐藏状态进行分类

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # 定义交叉熵损失函数
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))  # 计算损失

        # 如果不返回字典形式的结果
        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]  # 构建输出元组
            return ((loss,) + output) if loss is not None else output  # 返回结果元组

        # 返回SequenceClassifierOutput对象
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```