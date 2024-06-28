# `.\models\hubert\modeling_hubert.py`

```py
# 设置编码格式为 UTF-8

# 版权声明及许可信息，指出代码的版权和许可条款

""" PyTorch Hubert model."""

# 导入警告模块，用于生成警告
import warnings
# 导入类型提示模块中的类型和元组
from typing import Optional, Tuple, Union

# 导入 NumPy 库
import numpy as np
# 导入 PyTorch 库
import torch
# 导入 PyTorch 中的检查点模块
import torch.utils.checkpoint
# 导入 PyTorch 中的神经网络模块
from torch import nn
# 从 PyTorch 中的损失函数模块导入交叉熵损失函数
from torch.nn import CrossEntropyLoss

# 导入特定路径中的模块
from ...activations import ACT2FN
# 导入 DeepSpeed 集成库中的函数，用于检查是否启用了 DeepSpeed zero3
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
# 导入模型输出相关的基类
from ...modeling_outputs import BaseModelOutput, CausalLMOutput, SequenceClassifierOutput
# 导入模型工具函数
from ...modeling_utils import PreTrainedModel
# 导入常用工具函数
from ...utils import (
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
# 导入 Hubert 配置文件
from .configuration_hubert import HubertConfig

# 获取日志记录器对象
logger = logging.get_logger(__name__)

# 定义隐藏状态起始位置的常量
_HIDDEN_STATES_START_POSITION = 1

# 模型配置文档字符串
_CONFIG_FOR_DOC = "HubertConfig"

# 检查点文档字符串
_CHECKPOINT_FOR_DOC = "facebook/hubert-large-ls960-ft"
# 期望的输出形状文档字符串
_EXPECTED_OUTPUT_SHAPE = [1, 292, 768]

# CTC（连续文本分类）预期输出文本字符串
_CTC_EXPECTED_OUTPUT = "'MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL'"
# CTC 预期损失值
_CTC_EXPECTED_LOSS = 22.68

# 音频分类检查点文档字符串
_SEQ_CLASS_CHECKPOINT = "superb/hubert-base-superb-ks"
# 音频分类预期输出文本字符串
_SEQ_CLASS_EXPECTED_OUTPUT = "'_unknown_'"
# 音频分类预期损失值
_SEQ_CLASS_EXPECTED_LOSS = 8.53

# Hubert 预训练模型归档列表
HUBERT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/hubert-base-ls960",
    # 可在 https://huggingface.co/models?filter=hubert 查看所有 Hubert 模型
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
    计算给定形状的随机掩码跨度。用于实现 ASR（自动语音识别）中的 SpecAugment 数据增强方法。
    注意，此方法未经过优化，应在 CPU 上作为训练期间的预处理的一部分运行，而不是在 TPU 上运行。
    """
    # 返回一个 NumPy 数组，表示随机生成的掩码跨度
    return np.ndarray
    Args:
        shape: The shape for which to compute masks. This should be of a tuple of size 2 where
               the first element is the batch size and the second element is the length of the axis to span.
        mask_prob:  The percentage of the whole axis (between 0 and 1) which will be masked. The number of
                    independently generated mask spans of length `mask_length` is computed by
                    `mask_prob*shape[1]/mask_length`. Note that due to overlaps, `mask_prob` is an upper bound and the
                    actual percentage will be smaller.
        mask_length: size of the mask
        min_masks: minimum number of masked spans
        attention_mask: A (right-padded) attention mask which independently shortens the feature axis of
                        each batch dimension.
    """
    # 解包形状参数
    batch_size, sequence_length = shape

    # 检查是否小于1
    if mask_length < 1:
        raise ValueError("`mask_length` has to be bigger than 0.")

    # 检查是否大于序列长度
    if mask_length > sequence_length:
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length}"
            f" and `sequence_length`: {sequence_length}`"
        )

    # epsilon 用于概率舍入
    epsilon = np.random.rand(1).item()

    def compute_num_masked_span(input_length):
        """Given input length, compute how many spans should be masked"""
        # 计算应该被遮罩的 span 数量
        num_masked_span = int(mask_prob * input_length / mask_length + epsilon)
        # 确保遮罩的 span 数量不低于最小要求
        num_masked_span = max(num_masked_span, min_masks)

        # 确保遮罩的 span 不超过序列长度
        if num_masked_span * mask_length > sequence_length:
            num_masked_span = sequence_length // mask_length

        # 确保遮罩的 span 不超过 input_length - (mask_length - 1)
        if input_length - (mask_length - 1) < num_masked_span:
            num_masked_span = max(input_length - (mask_length - 1), 0)

        return num_masked_span

    # 计算每个 batch 中的遮罩 span 的数量
    input_lengths = (
        attention_mask.sum(-1).detach().tolist()
        if attention_mask is not None
        else [sequence_length for _ in range(batch_size)]
    )

    # 创建用于 SpecAugment 的遮罩数组
    spec_aug_mask = np.zeros((batch_size, sequence_length), dtype=bool)
    spec_aug_mask_idxs = []

    # 计算最大允许的遮罩 span 数量
    max_num_masked_span = compute_num_masked_span(sequence_length)

    # 如果最大允许的遮罩 span 数量为 0，则直接返回空的遮罩数组
    if max_num_masked_span == 0:
        return spec_aug_mask
    # 对于每个输入长度进行循环处理
    for input_length in input_lengths:
        # 计算当前输入的被遮挡（masked）span的数量
        num_masked_span = compute_num_masked_span(input_length)

        # 随机选择要遮挡的索引位置
        spec_aug_mask_idx = np.random.choice(
            np.arange(input_length - (mask_length - 1)), num_masked_span, replace=False
        )

        # 选择第一个被抽样的索引，用作填充向量的虚拟索引
        # 确保所有批次的维度一致，因为可能存在概率舍入
        # 选择第一个样本只是将这些向量填充两次。
        if len(spec_aug_mask_idx) == 0:
            # 这种情况只能发生在`input_length`严格小于`sequence_length`的情况下，
            # 此时最后一个标记必须是填充标记，我们可以将其用作虚拟掩码ID
            dummy_mask_idx = sequence_length - 1
        else:
            dummy_mask_idx = spec_aug_mask_idx[0]

        # 将虚拟索引添加到`spec_aug_mask_idx`数组末尾，使其达到最大遮挡span数量
        spec_aug_mask_idx = np.concatenate(
            [spec_aug_mask_idx, np.ones(max_num_masked_span - num_masked_span, dtype=np.int32) * dummy_mask_idx]
        )
        spec_aug_mask_idxs.append(spec_aug_mask_idx)

    # 将列表转换为NumPy数组
    spec_aug_mask_idxs = np.array(spec_aug_mask_idxs)

    # 将遮挡的索引扩展为遮挡span
    spec_aug_mask_idxs = np.broadcast_to(
        spec_aug_mask_idxs[:, :, None], (batch_size, max_num_masked_span, mask_length)
    )
    # 将形状重新整理为(batch_size, max_num_masked_span * mask_length)
    spec_aug_mask_idxs = spec_aug_mask_idxs.reshape(batch_size, max_num_masked_span * mask_length)

    # 添加偏移量到起始索引，以创建遮挡span
    offsets = np.arange(mask_length)[None, None, :]
    offsets = np.broadcast_to(offsets, (batch_size, max_num_masked_span, mask_length)).reshape(
        batch_size, max_num_masked_span * mask_length
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

    # 确保索引不会超过序列长度
    if spec_aug_mask_idxs.max() > sequence_length - 1:
        spec_aug_mask_idxs[spec_aug_mask_idxs > sequence_length - 1] = sequence_length - 1

    # 将1散布到遮挡的索引位置
    np.put_along_axis(spec_aug_mask, spec_aug_mask_idxs, 1, -1)

    # 返回生成的遮挡mask
    return spec_aug_mask
# 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2NoLayerNormConvLayer 复制过来，将 Wav2Vec2 替换为 Hubert
class HubertNoLayerNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        # 如果 layer_id 大于 0，则使用前一个卷积层的输出维度作为输入维度，否则使用 1
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        # 使用当前层的卷积维度作为输出维度
        self.out_conv_dim = config.conv_dim[layer_id]

        # 创建一个一维卷积层
        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        # 根据配置选择激活函数
        self.activation = ACT2FN[config.feat_extract_activation]

    # 前向传播函数
    def forward(self, hidden_states):
        # 对输入的隐藏状态应用卷积操作
        hidden_states = self.conv(hidden_states)
        # 应用激活函数
        hidden_states = self.activation(hidden_states)
        return hidden_states


# 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2LayerNormConvLayer 复制过来，将 Wav2Vec2 替换为 Hubert
class HubertLayerNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        # 如果 layer_id 大于 0，则使用前一个卷积层的输出维度作为输入维度，否则使用 1
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        # 使用当前层的卷积维度作为输出维度
        self.out_conv_dim = config.conv_dim[layer_id]

        # 创建一个一维卷积层
        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        # 创建一个 LayerNorm 层，对输出维度进行归一化
        self.layer_norm = nn.LayerNorm(self.out_conv_dim, elementwise_affine=True)
        # 根据配置选择激活函数
        self.activation = ACT2FN[config.feat_extract_activation]

    # 前向传播函数
    def forward(self, hidden_states):
        # 对输入的隐藏状态应用卷积操作
        hidden_states = self.conv(hidden_states)

        # 将卷积输出的维度换位，以便于 LayerNorm 的应用
        hidden_states = hidden_states.transpose(-2, -1)
        # 应用 LayerNorm
        hidden_states = self.layer_norm(hidden_states)
        # 恢复维度的排列顺序
        hidden_states = hidden_states.transpose(-2, -1)

        # 应用激活函数
        hidden_states = self.activation(hidden_states)
        return hidden_states


# 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2GroupNormConvLayer 复制过来，将 Wav2Vec2 替换为 Hubert
class HubertGroupNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        # 如果 layer_id 大于 0，则使用前一个卷积层的输出维度作为输入维度，否则使用 1
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        # 使用当前层的卷积维度作为输出维度
        self.out_conv_dim = config.conv_dim[layer_id]

        # 创建一个一维卷积层
        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        # 根据配置选择激活函数
        self.activation = ACT2FN[config.feat_extract_activation]

        # 创建一个 GroupNorm 层，分组数为输出维度，通道数为输出维度
        self.layer_norm = nn.GroupNorm(num_groups=self.out_conv_dim, num_channels=self.out_conv_dim, affine=True)

    # 前向传播函数
    def forward(self, hidden_states):
        # 对输入的隐藏状态应用卷积操作
        hidden_states = self.conv(hidden_states)
        # 应用 GroupNorm
        hidden_states = self.layer_norm(hidden_states)
        # 应用激活函数
        hidden_states = self.activation(hidden_states)
        return hidden_states
# 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2PositionalConvEmbedding 复制代码，并将 Wav2Vec2 替换为 Hubert
class HubertPositionalConvEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 定义一维卷积层，用于位置编码
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

        # 如果启用了 DeepSpeed zero3 加速
        if is_deepspeed_zero3_enabled():
            import deepspeed

            # 使用 GatheredParameters 将权重进行分组
            with deepspeed.zero.GatheredParameters(self.conv.weight, modifier_rank=0):
                self.conv = weight_norm(self.conv, name="weight", dim=2)
            # 注册外部参数以便 DeepSpeed 管理
            deepspeed.zero.register_external_parameter(self, self.conv.weight_v)
            deepspeed.zero.register_external_parameter(self, self.conv.weight_g)
        else:
            # 否则正常进行权重归一化
            self.conv = weight_norm(self.conv, name="weight", dim=2)

        # 创建用于填充的层
        self.padding = HubertSamePadLayer(config.num_conv_pos_embeddings)
        # 选择激活函数
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        # 转置隐藏状态张量的维度
        hidden_states = hidden_states.transpose(1, 2)

        # 进行卷积操作
        hidden_states = self.conv(hidden_states)
        # 进行填充操作
        hidden_states = self.padding(hidden_states)
        # 应用激活函数
        hidden_states = self.activation(hidden_states)

        # 再次转置隐藏状态张量的维度
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states


# 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2SamePadLayer 复制代码，并将 Wav2Vec2 替换为 Hubert
class HubertSamePadLayer(nn.Module):
    def __init__(self, num_conv_pos_embeddings):
        super().__init__()
        # 根据卷积位置编码的数量确定是否需要移除填充
        self.num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0

    def forward(self, hidden_states):
        # 如果需要移除填充，则进行切片操作
        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, :, : -self.num_pad_remove]
        return hidden_states


# 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2FeatureEncoder 复制代码，并将 Wav2Vec2 替换为 Hubert
class HubertFeatureEncoder(nn.Module):
    """从原始音频波形构建特征"""

    # 构造函数留空，直接继承 nn.Module 的构造函数
    # 初始化方法，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类（superclass）的初始化方法
        super().__init__()

        # 根据配置文件中的特征提取归一化方式进行不同处理
        if config.feat_extract_norm == "group":
            # 如果是"group"方式，创建一组卷积层，第一个使用组归一化
            conv_layers = [HubertGroupNormConvLayer(config, layer_id=0)] + [
                HubertNoLayerNormConvLayer(config, layer_id=i + 1) for i in range(config.num_feat_extract_layers - 1)
            ]
        elif config.feat_extract_norm == "layer":
            # 如果是"layer"方式，创建一组卷积层，全部使用层归一化
            conv_layers = [HubertLayerNormConvLayer(config, layer_id=i) for i in range(config.num_feat_extract_layers)]
        else:
            # 如果归一化方式不是合法值，抛出数值错误异常
            raise ValueError(
                f"`config.feat_extract_norm` is {config.feat_extract_norm}, but has to be one of ['group', 'layer']"
            )

        # 将创建的卷积层列表转换为 nn.ModuleList，使其成为 nn.Module 的一部分
        self.conv_layers = nn.ModuleList(conv_layers)

        # 梯度检查点技术默认关闭
        self.gradient_checkpointing = False

        # 默认所有参数需要梯度计算
        self._requires_grad = True

    # 冻结模型参数，使其不再计算梯度
    def _freeze_parameters(self):
        # 遍历所有参数，设置其 requires_grad 属性为 False
        for param in self.parameters():
            param.requires_grad = False
        # 同时设置模型的 _requires_grad 属性为 False
        self._requires_grad = False

    # 前向传播方法
    def forward(self, input_values):
        # 将输入数据转换为二维张量
        hidden_states = input_values[:, None]

        # 如果需要梯度并且当前处于训练模式，确保 hidden_states 的 requires_grad 属性为 True
        if self._requires_grad and self.training:
            hidden_states.requires_grad = True

        # 遍历所有卷积层并逐层进行前向传播计算
        for conv_layer in self.conv_layers:
            # 如果需要梯度、开启了梯度检查点并且处于训练模式，则使用梯度检查点技术
            if self._requires_grad and self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    conv_layer.__call__,  # 调用当前卷积层的前向传播方法
                    hidden_states,        # 当前隐藏状态作为输入
                )
            else:
                # 否则，直接调用当前卷积层的前向传播方法
                hidden_states = conv_layer(hidden_states)

        # 返回最终的隐藏状态作为输出
        return hidden_states
class HubertFeatureExtractor(HubertFeatureEncoder):
    # 继承自HubertFeatureEncoder类的特征提取器类
    def __init__(self, config):
        super().__init__(config)
        # 警告：该类已被弃用，将在Transformers v5中移除，请使用`HubertFeatureEncoder`代替。
        warnings.warn(
            f"The class `{self.__class__.__name__}` has been depreciated "
            "and will be removed in Transformers v5. "
            f"Use `{self.__class__.__bases__[0].__name__}` instead.",
            FutureWarning,
        )


class HubertFeatureProjection(nn.Module):
    # Hubert特征投影模块
    def __init__(self, config):
        super().__init__()
        self.feat_proj_layer_norm = config.feat_proj_layer_norm
        if self.feat_proj_layer_norm:
            # 如果配置中包含特征投影层标准化，则初始化LayerNorm
            self.layer_norm = nn.LayerNorm(config.conv_dim[-1], eps=config.layer_norm_eps)
        # 线性映射投影到隐藏层大小
        self.projection = nn.Linear(config.conv_dim[-1], config.hidden_size)
        # 随机失活层
        self.dropout = nn.Dropout(config.feat_proj_dropout)

    def forward(self, hidden_states):
        # 非投影的隐藏状态用于量化
        if self.feat_proj_layer_norm:
            hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.projection(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


# 从transformers.models.bart.modeling_bart.BartAttention复制到HubertAttention，将Bart->Hubert
class HubertAttention(nn.Module):
    """来自'Attention Is All You Need'论文的多头注意力机制"""

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
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        # 线性映射层，用于查询、键、值和输出投影
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        # 重新塑造张量形状，以便进行多头注意力计算
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        ```
# 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2FeedForward复制到HubertFeedForward，用Hubert替换Wav2Vec2
class HubertFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.intermediate_dropout = nn.Dropout(config.activation_dropout)

        # 定义中间层全连接层，将隐藏大小转换为中间大小
        self.intermediate_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 根据配置选择隐藏层激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

        # 定义输出全连接层，将中间大小转换回隐藏大小
        self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.output_dropout = nn.Dropout(config.hidden_dropout)

    def forward(self, hidden_states):
        # 中间全连接层和激活函数
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)

        # 输出全连接层和dropout
        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states


# 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2EncoderLayer复制到HubertEncoderLayer，用Hubert替换Wav2Vec2
class HubertEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 定义注意力层，使用HubertAttention
        self.attention = HubertAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
        )
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 定义FeedForward层，使用HubertFeedForward
        self.feed_forward = HubertFeedForward(config)
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        # 记录注意力残差
        attn_residual = hidden_states
        # 执行注意力计算
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
        )
        hidden_states = self.dropout(hidden_states)
        # 添加注意力残差到隐藏状态
        hidden_states = attn_residual + hidden_states

        # 层归一化
        hidden_states = self.layer_norm(hidden_states)
        # 使用FeedForward层处理隐藏状态
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


# 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2AttnAdapterLayer复制到HubertAttnAdapterLayer，用Hubert替换Wav2Vec2
class HubertAttnAdapterLayer(nn.Module):
    def __init__(self, config):
        """
        Implements adapter modules directly with 3D tensor weight as parameters and without using ModuleList to speed
        up training throughput.
        """
        # 调用父类的初始化方法
        super().__init__()
        # 设置输入维度为配置文件中的适配器注意力维度
        self.input_dim = config.adapter_attn_dim
        # 设置隐藏维度为配置文件中的隐藏大小
        self.hidden_dim = config.hidden_size

        # 使用LayerNorm对隐藏状态进行归一化
        self.norm = nn.LayerNorm(self.hidden_dim)
        # 第一个线性层，将隐藏状态映射到适配器注意力维度
        self.linear_1 = nn.Linear(self.hidden_dim, self.input_dim)
        # 激活函数ReLU
        self.act_fn = nn.ReLU()
        # 第二个线性层，将适配器注意力维度映射回隐藏维度
        self.linear_2 = nn.Linear(self.input_dim, self.hidden_dim)

    def forward(self, hidden_states: torch.FloatTensor):
        # 对输入的隐藏状态进行LayerNorm归一化
        hidden_states = self.norm(hidden_states)

        # 第一个线性层的前向传播
        hidden_states = self.linear_1(hidden_states)
        # 应用ReLU激活函数
        hidden_states = self.act_fn(hidden_states)
        # 第二个线性层的前向传播
        hidden_states = self.linear_2(hidden_states)

        # 返回处理后的隐藏状态
        return hidden_states
# 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2EncoderLayerStableLayerNorm 复制过来，将 Wav2Vec2 替换为 Hubert
class HubertEncoderLayerStableLayerNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 定义自注意力层 HubertAttention，使用配置中的隐藏尺寸、注意力头数和注意力丢弃率，作为编码器而非解码器
        self.attention = HubertAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
        )
        # 定义 Dropout 层，使用配置中的隐藏层丢弃率
        self.dropout = nn.Dropout(config.hidden_dropout)
        # 定义 LayerNorm 层，使用配置中的隐藏尺寸和层标准化系数
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 定义前馈神经网络 HubertFeedForward
        self.feed_forward = HubertFeedForward(config)
        # 定义最终的 LayerNorm 层，使用配置中的隐藏尺寸和层标准化系数
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 如果配置中有 adapter_attn_dim 属性，则定义 HubertAttnAdapterLayer，否则为 None
        if getattr(config, "adapter_attn_dim", None) is not None:
            self.adapter_layer = HubertAttnAdapterLayer(config)
        else:
            self.adapter_layer = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        # 保存注意力残差
        attn_residual = hidden_states
        # 应用 LayerNorm 层
        hidden_states = self.layer_norm(hidden_states)
        # 应用自注意力层 HubertAttention，获取注意力权重（如果需要），输出新的隐藏状态
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
        )
        # 应用 Dropout
        hidden_states = self.dropout(hidden_states)
        # 加上注意力残差，形成新的隐藏状态
        hidden_states = attn_residual + hidden_states
        # 应用前馈神经网络，并加上最终的 LayerNorm
        hidden_states = hidden_states + self.feed_forward(self.final_layer_norm(hidden_states))

        # 如果存在 adapter_layer，则应用它
        if self.adapter_layer is not None:
            hidden_states = hidden_states + self.adapter_layer(hidden_states)

        # 输出包含最终隐藏状态的元组 outputs
        outputs = (hidden_states,)

        # 如果需要输出注意力权重，则将注意力权重加入 outputs 元组
        if output_attentions:
            outputs += (attn_weights,)

        return outputs


# 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2Encoder 复制过来，将 Wav2Vec2 替换为 Hubert
class HubertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 定义位置卷积嵌入层 HubertPositionalConvEmbedding
        self.pos_conv_embed = HubertPositionalConvEmbedding(config)
        # 定义 LayerNorm 层，使用配置中的隐藏尺寸和层标准化系数
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 定义 Dropout 层，使用配置中的隐藏层丢弃率
        self.dropout = nn.Dropout(config.hidden_dropout)
        # 定义多层 HubertEncoderLayer 层，并放入 nn.ModuleList 中
        self.layers = nn.ModuleList([HubertEncoderLayer(config) for _ in range(config.num_hidden_layers)])
        # 是否启用渐变检查点
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states: torch.tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        ):
            # 初始化隐藏状态输出，根据需要创建空元组或者None
            all_hidden_states = () if output_hidden_states else None
            # 初始化自注意力输出，根据需要创建空元组或者None
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

            # 计算位置嵌入
            position_embeddings = self.pos_conv_embed(hidden_states)
            # 将位置嵌入加到隐藏状态上
            hidden_states = hidden_states + position_embeddings
            # LayerNorm 归一化
            hidden_states = self.layer_norm(hidden_states)
            # Dropout
            hidden_states = self.dropout(hidden_states)

            # 检查是否启用了deepspeed zero3
            deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()

            # 遍历所有层进行处理
            for layer in self.layers:
                if output_hidden_states:
                    # 如果需要输出隐藏状态，则将当前层的隐藏状态添加到all_hidden_states中
                    all_hidden_states = all_hidden_states + (hidden_states,)

                # 添加LayerDrop（参见https://arxiv.org/abs/1909.11556 进行描述）
                dropout_probability = torch.rand([])

                # 根据LayerDrop的概率决定是否跳过当前层
                skip_the_layer = True if self.training and (dropout_probability < self.config.layerdrop) else False
                if not skip_the_layer or deepspeed_zero3_is_enabled:
                    # 如果不跳过当前层或者启用了deepspeed zero3，则进行前向传播
                    if self.gradient_checkpointing and self.training:
                        # 使用梯度检查点进行前向传播（checkpointing）
                        layer_outputs = self._gradient_checkpointing_func(
                            layer.__call__,
                            hidden_states,
                            attention_mask,
                            output_attentions,
                        )
                    else:
                        # 普通的前向传播
                        layer_outputs = layer(
                            hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
                        )
                    hidden_states = layer_outputs[0]

                if skip_the_layer:
                    # 如果跳过当前层，则输出设置为None
                    layer_outputs = (None, None)

                if output_attentions:
                    # 如果需要输出注意力权重，则将当前层的注意力权重添加到all_self_attentions中
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)

            if output_hidden_states:
                # 如果需要输出隐藏状态，则将最终的隐藏状态添加到all_hidden_states中
                all_hidden_states = all_hidden_states + (hidden_states,)

            if not return_dict:
                # 如果不需要返回字典形式的输出，则返回一个元组，过滤掉为None的部分
                return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
            # 如果需要返回字典形式的输出，则创建BaseModelOutput对象并返回
            return BaseModelOutput(
                last_hidden_state=hidden_states,
                hidden_states=all_hidden_states,
                attentions=all_self_attentions,
            )
# 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2EncoderStableLayerNorm 复制代码，并将 Wav2Vec2 替换为 Hubert
class HubertEncoderStableLayerNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 初始化位置编码卷积嵌入层，使用 HubertPositionalConvEmbedding 类
        self.pos_conv_embed = HubertPositionalConvEmbedding(config)
        # 初始化层归一化层，归一化隐藏状态特征向量
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化丢弃层，以减少隐藏状态特征向量中的部分信息，防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout)
        # 初始化层列表，包含 HubertEncoderLayerStableLayerNorm 类的隐藏层，数量由配置中的 num_hidden_layers 决定
        self.layers = nn.ModuleList(
            [HubertEncoderLayerStableLayerNorm(config) for _ in range(config.num_hidden_layers)]
        )
        # 梯度检查点设置为关闭状态
        self.gradient_checkpointing = False

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        # 初始化所有隐藏状态为一个空元组，如果不输出隐藏状态则为 None
        all_hidden_states = () if output_hidden_states else None
        # 初始化所有自注意力权重为一个空元组，如果不输出注意力权重则为 None
        all_self_attentions = () if output_attentions else None

        # 如果存在注意力遮罩
        if attention_mask is not None:
            # 确保填充的标记不被注意到
            expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
            hidden_states[~expand_attention_mask] = 0

            # 扩展注意力遮罩
            attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
            attention_mask = attention_mask.expand(
                attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
            )

        # 计算位置嵌入
        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.dropout(hidden_states)

        # 检查是否启用了 DeepSpeed Zero3
        deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()

        # 对每个层进行循环
        for layer in self.layers:
            # 如果输出隐藏状态，则记录当前隐藏状态
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 添加 LayerDrop（参见 https://arxiv.org/abs/1909.11556）
            dropout_probability = torch.rand([])

            # 根据 LayerDrop 的概率决定是否跳过当前层
            skip_the_layer = True if self.training and (dropout_probability < self.config.layerdrop) else False
            if not skip_the_layer or deepspeed_zero3_is_enabled:
                # 如果启用了梯度检查点和处于训练模式，则使用梯度检查点来调用当前层
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        layer.__call__,
                        hidden_states,
                        attention_mask,
                        output_attentions,
                    )
                else:
                    # 否则直接调用当前层
                    layer_outputs = layer(
                        hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
                    )
                hidden_states = layer_outputs[0]

            # 如果跳过当前层，则层输出为空
            if skip_the_layer:
                layer_outputs = (None, None)

            # 如果输出注意力权重，则记录当前层的自注意力权重
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 对最终的隐藏状态进行 Layer Norm 处理
        hidden_states = self.layer_norm(hidden_states)

        # 如果输出隐藏状态，则记录最终的隐藏状态
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不返回字典形式的结果，则按顺序返回相关结果
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        # 否则返回 Base Model Output 对象
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 指定配置类为 HubertConfig
    config_class = HubertConfig
    # 模型的前缀名为 "hubert"
    base_model_prefix = "hubert"
    # 主要输入的名称为 "input_values"
    main_input_name = "input_values"
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果是线性层，则使用正态分布初始化权重
        if isinstance(module, nn.Linear):
            # 与 TensorFlow 版本略有不同，后者使用截断正态分布进行初始化
            # 参考 https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        # 如果是 LayerNorm 或 GroupNorm，则初始化偏置为零，权重为1
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        # 如果是 1D 卷积层
        elif isinstance(module, nn.Conv1d):
            # 检查是否启用了 DeepSpeed 的 Zero3 模式
            if is_deepspeed_zero3_enabled():
                import deepspeed

                # 如果模块有 weight_v 和 weight_g 属性
                if hasattr(module, "weight_v") and hasattr(module, "weight_g"):
                    # 使用 GatheredParameters 进行初始化
                    with deepspeed.zero.GatheredParameters([module.weight_v, module.weight_g], modifier_rank=0):
                        nn.init.kaiming_normal_(module.weight.data)
                else:
                    # 使用 GatheredParameters 进行初始化
                    with deepspeed.zero.GatheredParameters(module.weight, modifier_rank=0):
                        nn.init.kaiming_normal_(module.weight.data)
            else:
                # 使用 kaiming_normal_ 方法初始化权重
                nn.init.kaiming_normal_(module.weight.data)

        # 如果是线性层或 1D 卷积层，并且有偏置，则将偏置初始化为零
        if isinstance(module, (nn.Linear, nn.Conv1d)) and module.bias is not None:
            module.bias.data.zero_()

    def _get_feat_extract_output_lengths(self, input_lengths: Union[torch.LongTensor, int]):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            # 计算 1D 卷积层的输出长度，使用公式来源于 PyTorch 文档
            return torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1

        # 遍历配置中的卷积核大小和步长
        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            # 更新输入长度为卷积层输出长度
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        return input_lengths
    # 定义一个方法，用于生成特征向量的注意力掩码
    def _get_feature_vector_attention_mask(self, feature_vector_length: int, attention_mask: torch.LongTensor):
        # 根据注意力掩码的长度信息计算输出长度，并转换为长整型
        output_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)
        # 获取当前批次的大小
        batch_size = attention_mask.shape[0]

        # 初始化一个全零的注意力掩码张量，形状为(batch_size, feature_vector_length)
        attention_mask = torch.zeros(
            (batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
        )

        # 设置注意力掩码的部分值为1，确保在输出长度之前的所有位置都被关注
        attention_mask[(torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1)] = 1

        # 翻转张量，并对每行进行累积求和，然后再次翻转，并将结果转换为布尔类型
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()

        # 返回生成的注意力掩码张量
        return attention_mask
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

HUBERT_INPUTS_DOCSTRING = r"""
    Placeholder for the documentation string describing the inputs expected by the `Hubert` model.
"""
    Args:
        input_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            # 输入的原始语音波形的浮点值。可以通过将 `.flac` 或 `.wav` 音频文件加载到 `List[float]` 或 `numpy.ndarray` 类型的数组中获得。可以使用 `soundfile` 库 (`pip install soundfile`)。使用 [`AutoProcessor`] 进行填充和转换成 `torch.FloatTensor` 类型的张量，详见 [`Wav2Vec2Processor.__call__`]。

        attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            # 遮罩，用于避免在填充标记索引上执行卷积和注意力操作。遮罩值为 `[0, 1]`：

            - 1 表示 **未被遮罩** 的标记，
            - 0 表示 **被遮罩** 的标记。

            [什么是注意力遮罩?](../glossary#attention-mask)

            <Tip warning={true}>

            只有当相应的处理器具有 `config.return_attention_mask == True` 时才应传递 `attention_mask`。对于所有处理器具有 `config.return_attention_mask == False` 的模型，例如 [hubert-base](https://huggingface.co/facebook/hubert-base-ls960)，不应传递 `attention_mask` 以避免在进行批量推理时性能下降。对于这样的模型，`input_values` 应简单地填充为 0 并且不传递 `attention_mask`。请注意，这些模型在 `input_values` 是否填充上也会产生略微不同的结果。

            </Tip>

        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量。有关更多详细信息，请参见返回的张量下的 `attentions`。

        output_hidden_states (`bool`, *optional*):
            # 是否返回所有层的隐藏状态。有关更多详细信息，请参见返回的张量下的 `hidden_states`。

        return_dict (`bool`, *optional*):
            # 是否返回 [`~utils.ModelOutput`] 而不是普通的元组。
"""
在代码中添加注释，解释每个语句的作用和功能。
"""

@add_start_docstrings(
    "The bare Hubert Model transformer outputting raw hidden-states without any specific head on top.",
    HUBERT_START_DOCSTRING,
)
class HubertModel(HubertPreTrainedModel):
    def __init__(self, config: HubertConfig):
        super().__init__(config)
        self.config = config  # 初始化模型配置
        self.feature_extractor = HubertFeatureEncoder(config)  # 使用给定配置创建特征提取器
        self.feature_projection = HubertFeatureProjection(config)  # 使用给定配置创建特征投影器

        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0:
            self.masked_spec_embed = nn.Parameter(torch.FloatTensor(config.hidden_size).uniform_())
            # 如果配置中有时间或特征掩码概率大于零，则初始化一个可学习的掩码嵌入向量

        if config.do_stable_layer_norm:
            self.encoder = HubertEncoderStableLayerNorm(config)
            # 如果配置要求稳定的层归一化，则使用稳定层归一化版本的编码器
        else:
            self.encoder = HubertEncoder(config)
            # 否则使用普通版本的编码器

        # 初始化权重并应用最终处理
        self.post_init()

    # 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2Model._mask_hidden_states复制而来
    def _mask_hidden_states(
        self,
        hidden_states: torch.FloatTensor,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        # 掩码模型的隐藏状态，可以选择性地使用时间索引或注意力掩码

        ):
        """
        对隐藏状态进行掩码操作。

        Args:
            hidden_states (torch.FloatTensor): 输入的隐藏状态张量。
            mask_time_indices (Optional[torch.FloatTensor]): 可选的时间索引掩码张量。
            attention_mask (Optional[torch.LongTensor]): 可选的注意力掩码张量。

        """
    ):
        """
        Masks extracted features along time axis and/or along feature axis according to
        [SpecAugment](https://arxiv.org/abs/1904.08779).
        """

        # `config.apply_spec_augment` can set masking to False
        # 检查配置中是否允许应用 SpecAugment，如果不允许，则直接返回隐藏状态
        if not getattr(self.config, "apply_spec_augment", True):
            return hidden_states

        # generate indices & apply SpecAugment along time axis
        batch_size, sequence_length, hidden_size = hidden_states.size()

        if mask_time_indices is not None:
            # apply SpecAugment along time axis with given mask_time_indices
            # 根据给定的 mask_time_indices 在时间轴上应用 SpecAugment
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)
        elif self.config.mask_time_prob > 0 and self.training:
            # calculate mask_time_indices if not provided explicitly
            # 如果未明确提供 mask_time_indices，则计算它
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
            # 生成索引并沿特征轴应用 SpecAugment
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
        """

        Returns a tuple containing model outputs or a BaseModelOutput.

        Example:

        ```
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

        # Initialize variables with default values if not provided
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Extract features from input_values using feature_extractor
        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)  # Transpose dimensions for further processing

        # Compute attention mask specific to feature vectors if provided
        if attention_mask is not None:
            attention_mask = self._get_feature_vector_attention_mask(extract_features.shape[1], attention_mask)

        # Project features into hidden states
        hidden_states = self.feature_projection(extract_features)

        # Mask certain time indices in hidden states if specified
        hidden_states = self._mask_hidden_states(hidden_states, mask_time_indices=mask_time_indices)

        # Encode hidden states using the encoder
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Extract last hidden states from encoder outputs
        hidden_states = encoder_outputs[0]

        # Return model outputs based on return_dict flag
        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]  # Return tuple of hidden states and additional outputs

        # Return BaseModelOutput object with specified attributes
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
@add_start_docstrings(
    """Hubert Model with a `language modeling` head on top for Connectionist Temporal Classification (CTC).""",
    HUBERT_START_DOCSTRING,
)
# 定义了一个名为 HubertForCTC 的类，继承自 HubertPreTrainedModel
# 此类实现了带有语言建模头部的 Hubert 模型，用于连接主义时间分类（CTC）任务。
class HubertForCTC(HubertPreTrainedModel):
    def __init__(self, config, target_lang: Optional[str] = None):
        super().__init__(config)

        # 初始化 Hubert 模型
        self.hubert = HubertModel(config)
        # Dropout 层
        self.dropout = nn.Dropout(config.final_dropout)

        # 可选的目标语言设定
        self.target_lang = target_lang

        # 检查配置中是否定义了词汇表大小，如果没有则抛出异常
        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `HubertForCTC.from_pretrained(..., vocab_size=vocab_size)`. "
                "or define `vocab_size` of your model's configuration."
            )

        # 根据配置定义线性层作为语言建模头部
        output_hidden_size = (
            config.output_hidden_size if hasattr(config, "add_adapter") and config.add_adapter else config.hidden_size
        )
        self.lm_head = nn.Linear(output_hidden_size, config.vocab_size)

        # 初始化权重并应用最终处理
        self.post_init()

    # 覆盖 PreTrainedModel 的 tie_weights 方法，以便在通过 from_pretrained(...) 传递 target_lang=... 时能正确加载适配器权重
    def tie_weights(self):
        """
        This method overwrites [`~PreTrainedModel.tie_weights`] so that adapter weights can be correctly loaded when
        passing `target_lang=...` to `from_pretrained(...)`.

        This method is **not** supposed to be called by the user and is prone to be changed in the future.
        """

        # 注意，tie_weights 通常用于绑定输入和输出嵌入权重。在这里重新定义它的目的是为了正确加载 Hubert 的适配器层，
        # 以便不需要引入新的 API 到 PreTrainedModel。虽然有些技巧性，但 Hubert 永远不需要绑定输入和输出嵌入，因此在这里重新用于适配器加载是可以接受的。

        # 获取目标语言
        target_lang = self.target_lang

        # 如果 target_lang 不为 None，且配置中未定义 adapter_attn_dim，则抛出异常
        if target_lang is not None and getattr(self.config, "adapter_attn_dim", None) is None:
            raise ValueError(f"Cannot pass `target_lang`: {target_lang} if `config.adapter_attn_dim` is not defined.")
        # 如果 target_lang 为 None，且配置中定义了 adapter_attn_dim，则记录日志提示用户默认 target_lang 为 'eng'
        elif target_lang is None and getattr(self.config, "adapter_attn_dim", None) is not None:
            logger.info("By default `target_lang` is set to 'eng'.")
        # 如果 target_lang 不为 None，则加载适配器
        elif target_lang is not None:
            self.load_adapter(target_lang, force_load=True)
    # 调用此函数将冻结特征编码器的梯度计算，使其在训练过程中不会更新参数。
    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        # 发出警告提示，说明函数 `freeze_feature_extractor` 将在 Transformers v5 中移除，并建议使用 `freeze_feature_encoder` 替代。
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. "
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        # 调用 `freeze_feature_encoder` 方法来实现特征编码器参数的冻结。
        self.freeze_feature_encoder()

    # 调用此函数将冻结特征编码器的梯度计算，使其在训练过程中不会更新参数。
    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        # 调用 Hubert 模型中的特征提取器的 `_freeze_parameters` 方法来冻结参数。
        self.hubert.feature_extractor._freeze_parameters()

    # 调用此函数将冻结基础模型的梯度计算，使其在训练过程中不会更新参数，仅分类头会更新。
    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        # 遍历 Hubert 模型的所有参数，并将其 `requires_grad` 设置为 False，以冻结基础模型的参数。
        for param in self.hubert.parameters():
            param.requires_grad = False

    # 重写 `forward` 方法，将其注解添加到模型的前向传播文档中，并附上代码示例的文档字符串。
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
        # 初始化返回字典，如果未指定则使用配置中的返回字典设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 Hubert 模型，获取输出结果
        outputs = self.hubert(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 从输出中获取隐藏状态，并应用 dropout
        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        # 将隐藏状态传入语言模型头部，生成预测 logits
        logits = self.lm_head(hidden_states)

        # 初始化损失为 None
        loss = None
        if labels is not None:
            # 如果标签存在，检查标签值是否超出词汇表大小，如果是则引发 ValueError
            if labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

            # 根据注意力掩码计算输入长度
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            # 假设填充的标记用 -100 填充，不被注意到时
            # 创建标签掩码以计算目标长度
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # 使用 log_softmax 计算对数概率
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

            # 禁用 cuDNN 以确保兼容性
            with torch.backends.cudnn.flags(enabled=False):
                # 计算 CTC 损失
                loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )

        # 如果不要求返回字典，则根据输出格式构建返回结果
        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        # 如果要求返回字典，则创建 CausalLMOutput 对象并返回
        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )
# 使用 Hubert 模型进行序列分类，该模型在顶部有一个用于分类的线性层（基于池化输出）
@add_start_docstrings(
    """
    Hubert Model with a sequence classification head on top (a linear layer over the pooled output) for tasks like
    SUPERB Keyword Spotting.
    """,
    HUBERT_START_DOCSTRING,
)
# 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForSequenceClassification 复制而来，将 Wav2Vec2 改为 Hubert，wav2vec2 改为 hubert，WAV_2_VEC_2 改为 HUBERT
class HubertForSequenceClassification(HubertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 如果配置中存在 `add_adapter` 属性且为 True，则抛出异常，因为序列分类不支持使用 Hubert 适配器
        if hasattr(config, "add_adapter") and config.add_adapter:
            raise ValueError(
                "Sequence classification does not support the use of Hubert adapters (config.add_adapter=True)"
            )
        # 创建 HubertModel 对象
        self.hubert = HubertModel(config)
        # 计算层数，包括变换器层和输入嵌入层
        num_layers = config.num_hidden_layers + 1
        # 如果配置指定使用加权层求和，则初始化层权重
        if config.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        # 创建投影层，将隐藏状态映射到分类器投影空间
        self.projector = nn.Linear(config.hidden_size, config.classifier_proj_size)
        # 创建分类器层，将投影后的特征映射到类别数量
        self.classifier = nn.Linear(config.classifier_proj_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 冻结特征提取器，不再更新其参数
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

    # 冻结特征编码器，不再更新其参数
    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.hubert.feature_extractor._freeze_parameters()

    # 冻结基础模型，不再更新其参数，只更新分类头
    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        for param in self.hubert.parameters():
            param.requires_grad = False

    # 将 HUBERT_INPUTS_DOCSTRING 添加到模型前向传播函数的文档字符串中
    @add_start_docstrings_to_model_forward(HUBERT_INPUTS_DOCSTRING)
    # 将代码示例的文档字符串添加到模型前向传播函数的文档字符串中，指定了检查点、输出类型、配置类、模态（audio）、预期输出和预期损失
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

        # 设置是否返回字典形式的输出结果，默认为模型配置中指定的值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 如果配置中指定使用加权层求和的隐藏状态，则设置为True，否则使用传入参数
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states

        # 使用 Hubert 模型进行前向传播，获取输出结果
        outputs = self.hubert(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 如果配置中指定使用加权层求和的隐藏状态
        if self.config.use_weighted_layer_sum:
            # 从输出结果中提取隐藏状态
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            # 将隐藏状态堆叠起来，按照加权向量进行加权求和
            hidden_states = torch.stack(hidden_states, dim=1)
            # 对加权向量进行 softmax 归一化处理
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            # 按加权向量加权求和隐藏状态
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            # 否则直接使用第一个输出的隐藏状态
            hidden_states = outputs[0]

        # 使用投影层进行映射
        hidden_states = self.projector(hidden_states)

        # 如果没有传入注意力掩码，则计算平均池化输出
        if attention_mask is None:
            pooled_output = hidden_states.mean(dim=1)
        else:
            # 否则根据注意力掩码生成填充掩码，将填充位置的隐藏状态置为0
            padding_mask = self._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)
            hidden_states[~padding_mask] = 0.0
            # 计算填充掩码后的池化输出
            pooled_output = hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)

        # 使用分类器计算 logits
        logits = self.classifier(pooled_output)

        # 初始化损失为 None
        loss = None
        # 如果传入了标签，则计算交叉熵损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        # 如果不要求返回字典形式的输出
        if not return_dict:
            # 组装输出元组，包括 logits 和隐藏状态
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            # 如果有损失，则将损失加入输出元组
            return ((loss,) + output) if loss is not None else output

        # 如果要求返回字典形式的输出结果
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```