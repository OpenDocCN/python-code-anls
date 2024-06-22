# `.\transformers\models\wav2vec2\modeling_wav2vec2.py`

```py
# 设置文件编码格式为 utf-8
# 版权声明
# 根据 Apache 许可 2.0 版本授权
# 在遵循许可证的情况下，可以使用此文件，否则不允许使用
# 可以在以下链接获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或以书面形式同意, 否则根据许可证分发的软件是基于 "如是" 基础分发的，没有任何明示或暗示的保证或条件
# 请参阅许可证以获取关于特殊语言的权限和限制
# PyTorch Wav2Vec2 模型

# 导入所需的库和模块
import math
import warnings
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss
from ...activations import ACT2FN
from ...integrations.deepspeed import is_deepspeed_zero3_enabled
from ...modeling_outputs import (
    BaseModelOutput,
    CausalLMOutput,
    MaskedLMOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
    Wav2Vec2BaseModelOutput,
    XVectorOutput,
)
from ...modeling_utils import PreTrainedModel
from ...pytorch_utils import is_torch_greater_or_equal_than_1_13
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    cached_file,
    is_safetensors_available,
    logging,
    replace_return_docstrings,
)
from .configuration_wav2vec2 import Wav2Vec2Config

# 设置适配器文件的默认名称
WAV2VEC2_ADAPTER_PT_FILE = "adapter.{}.bin"
WAV2VEC2_ADAPTER_SAFE_FILE = "adapter.{}.safetensors"

# 如果安全张量可用，导入安全加载文件函数
if is_safetensors_available():
    from safetensors.torch import load_file as safe_load_file

# 获取日志记录器
logger = logging.get_logger(__name__)

# 隐藏状态的起始位置
_HIDDEN_STATES_START_POSITION = 2

# 一般文档字符串
_CONFIG_FOR_DOC = "Wav2Vec2Config"

# 基础文档字符串
_CHECKPOINT_FOR_DOC = "facebook/wav2vec2-base-960h"
_EXPECTED_OUTPUT_SHAPE = [1, 292, 768]

# CTC 文档字符串
_CTC_EXPECTED_OUTPUT = "'MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL'"
_CTC_EXPECTED_LOSS = 53.48

# 音频类文档字符串
_SEQ_CLASS_CHECKPOINT = "superb/wav2vec2-base-superb-ks"
_SEQ_CLASS_EXPECTED_OUTPUT = "'_unknown_'"
_SEQ_CLASS_EXPECTED_LOSS = 6.54

# 帧类文档字符串
_FRAME_CLASS_CHECKPOINT = "anton-l/wav2vec2-base-superb-sd"
_FRAME_EXPECTED_OUTPUT = [0, 0]

# 说话人验证文档字符串
_XVECTOR_CHECKPOINT = "anton-l/wav2vec2-base-superb-sv"
_XVECTOR_EXPECTED_OUTPUT = 0.98

# 预训练模型存档列表
WAV_2_VEC_2_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "facebook/wav2vec2-base-960h",
    "facebook/wav2vec2-large-960h",
    "facebook/wav2vec2-large-960h-lv60",
    "facebook/wav2vec2-large-960h-lv60-self",
    # See all Wav2Vec2 models at https://huggingface.co/models?filter=wav2vec2
]

# Wav2Vec2ForPreTrainingOutput 输出类
@dataclass
class Wav2Vec2ForPreTrainingOutput(ModelOutput):
    """
    # 定义函数的输出类型及可能的隐藏状态和注意力
    Output type of [`Wav2Vec2ForPreTraining`], with potential hidden states and attentions.
    
    Args:
        loss (*optional*, returned when `sample_negative_indices` are passed, `torch.FloatTensor` of shape `(1,)`):
            Total loss as the sum of the contrastive loss (L_m) and the diversity loss (L_d) as stated in the [official
            paper](https://arxiv.org/pdf/2006.11477.pdf) . (classification) loss.
        projected_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.proj_codevector_dim)`):
            Hidden-states of the model projected to *config.proj_codevector_dim* that can be used to predict the masked
            projected quantized states.
        projected_quantized_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.proj_codevector_dim)`):
            Quantized extracted feature vectors projected to *config.proj_codevector_dim* representing the positive
            target vectors for contrastive loss.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.
    
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
    
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        contrastive_loss (*optional*, returned when `sample_negative_indices` are passed, `torch.FloatTensor` of shape `(1,)`):
            The contrastive loss (L_m) as stated in the [official paper](https://arxiv.org/pdf/2006.11477.pdf) .
        diversity_loss (*optional*, returned when `sample_negative_indices` are passed, `torch.FloatTensor` of shape `(1,)`):
            The diversity loss (L_d) as stated in the [official paper](https://arxiv.org/pdf/2006.11477.pdf) .
    """
    
    # 定义函数的输出类型
    loss: Optional[torch.FloatTensor] = None
    projected_states: torch.FloatTensor = None
    projected_quantized_states: torch.FloatTensor = None
    codevector_perplexity: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    contrastive_loss: Optional[torch.FloatTensor] = None
    diversity_loss: Optional[torch.FloatTensor] = None
# 定义一个函数，用于计算给定形状的随机遮罩范围。这个函数被用于实现 SpecAugment，一种简单的用于自动语音识别的数据增强方法。
# 需要注意的是，这个方法没有针对 TPU 进行优化，应该在 CPU 上作为训练预处理的一部分来运行。

def _compute_mask_indices(
    shape: Tuple[int, int],  # 函数输入参数，形状的元组，包含批大小和轴长度
    mask_prob: float,  # 函数输入参数，整个轴被遮罩的百分比
    mask_length: int,  # 函数输入参数，遮罩的长度
    attention_mask: Optional[torch.LongTensor] = None,  # 函数输入参数，注意力遮罩，用于短维度
    min_masks: int = 0,  # 函数输入参数，遮罩的最小数量
) -> np.ndarray:  # 函数返回类型，numpy 数组

    """
    Computes random mask spans for a given shape. Used to implement [SpecAugment: A Simple Data Augmentation Method for
    ASR](https://arxiv.org/abs/1904.08779). Note that this method is not optimized to run on TPU and should be run on
    CPU as part of the preprocessing during training.

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

    batch_size, sequence_length = shape  # 解包形状元组，获取批大小和序列长度

    if mask_length < 1:  # 如果遮罩长度小于 1，抛出 ValueError 异常
        raise ValueError("`mask_length` has to be bigger than 0.")

    if mask_length > sequence_length:  # 如果遮罩长度大于序列长度，抛出 ValueError 异常
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length}"
            f" and `sequence_length`: {sequence_length}`"
        )

    # epsilon is used for probabilistic rounding
    epsilon = np.random.rand(1).item()  # 用于概率舍入的 epsilon

    def compute_num_masked_span(input_length):
        """Given input length, compute how many spans should be masked"""
        # 给定输入长度，计算应该遮罩多少个范围
        num_masked_span = int(mask_prob * input_length / mask_length + epsilon)  # 计算遮罩的范围数量
        num_masked_span = max(num_masked_span, min_masks)  # 最小遮罩数量至少为 min_masks

        # 确保遮罩范围数量 <= 序列长度
        if num_masked_span * mask_length > sequence_length:
            num_masked_span = sequence_length // mask_length

        # 确保遮罩范围数量也 <= 输入长度 - (mask_length - 1)
        if input_length - (mask_length - 1) < num_masked_span:
            num_masked_span = max(input_length - (mask_length - 1), 0)

        return num_masked_span

    # 计算批中遮罩范围的数量
    input_lengths = (
        attention_mask.sum(-1).detach().tolist()  # 如果存在 attention_mask，则获取每个批的输入长度
        if attention_mask is not None
        else [sequence_length for _ in range(batch_size)]  # 否则，所有批的输入长度都是序列长度
    )

    # 用于 SpecAugment 的遮罩填充
    spec_aug_mask = np.zeros((batch_size, sequence_length), dtype=bool)  # 初始化一个布尔类型的遮罩矩阵
    spec_aug_mask_idxs = []  # 存储 SpecAugment 的遮罩索引

    max_num_masked_span = compute_num_masked_span(sequence_length)  # 计算序列中最大的遮罩范围数量

    if max_num_masked_span == 0:  # 如果最大遮罩范围数量为 0，则直接返回遮罩矩阵
        return spec_aug_mask
    # 遍历输入长度列表
    for input_length in input_lengths:
        # 计算当前输入需要mask的span数量
        num_masked_span = compute_num_masked_span(input_length)

        # 随机选择要进行mask的索引
        spec_aug_mask_idx = np.random.choice(
            np.arange(input_length - (mask_length - 1)), num_masked_span, replace=False
        )

        # 选择第一个采样索引作为填充向量的虚拟索引，确保所有批次的维度相同，因为概率性的四舍五入
        # 选择第一个样本只是为了两次填充这些向量
        if len(spec_aug_mask_idx) == 0:
            # 只有在`input_length`严格小于`sequence_length`时才会出现这种情况，此时最后一个标记必须是填充标记，我们可以使用它作为虚拟掩码id
            dummy_mask_idx = sequence_length - 1
        else:
            dummy_mask_idx = spec_aug_mask_idx[0]

        spec_aug_mask_idx = np.concatenate(
            [spec_aug_mask_idx, np.ones(max_num_masked_span - num_masked_span, dtype=np.int32) * dummy_mask_idx]
        )
        spec_aug_mask_idxs.append(spec_aug_mask_idx)

    spec_aug_mask_idxs = np.array(spec_aug_mask_idxs)

    # 将mask索引扩展为mask span
    spec_aug_mask_idxs = np.broadcast_to(
        spec_aug_mask_idxs[:, :, None], (batch_size, max_num_masked_span, mask_length)
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs.reshape(batch_size, max_num_masked_span * mask_length)

    # 对开始索引添加偏移量，使索引现在创建一个span
    offsets = np.arange(mask_length)[None, None, :]
    offsets = np.broadcast_to(offsets, (batch_size, max_num_masked_span, mask_length)).reshape(
        batch_size, max_num_masked_span * mask_length
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

    # 确保我们不能有大于sequence_length的索引
    if spec_aug_mask_idxs.max() > sequence_length - 1:
        spec_aug_mask_idxs[spec_aug_mask_idxs > sequence_length - 1] = sequence_length - 1

    # 分散索引以进行mask
    np.put_along_axis(spec_aug_mask, spec_aug_mask_idxs, 1, -1)

    return spec_aug_mask
def _sample_negative_indices(
    features_shape: Tuple, num_negatives: int, mask_time_indices: Optional[np.ndarray] = None
):
    """
    Sample `num_negatives` vectors from feature vectors.
    """
    # 获取特征向量的形状信息
    batch_size, sequence_length = features_shape

    # 生成正向量自身的索引，并将其重复 `num_negatives` 次
    sequence_length_range = np.arange(sequence_length)

    # 从同一个utterance中获取 `num_negatives` 个随机向量的索引
    sampled_negative_indices = np.zeros(shape=(batch_size, sequence_length, num_negatives), dtype=np.int32)

    # 将时间索引掩码转换为布尔型，如果给定则使用，否则使用全为True的数组
    mask_time_indices = (
        mask_time_indices.astype(bool) if mask_time_indices is not None else np.ones(features_shape, dtype=bool)
    )

    # 遍历每个batch中的序列
    for batch_idx in range(batch_size):
        # 计算非零时间索引的数量
        high = mask_time_indices[batch_idx].sum() - 1
        # 获取非零时间索引的值
        mapped_masked_indices = sequence_length_range[mask_time_indices[batch_idx]]

        # 创建形状为 (high + 1, num_negatives) 的特征索引数组
        feature_indices = np.broadcast_to(np.arange(high + 1)[:, None], (high + 1, num_negatives))
        # 从0到high之间随机选择 `num_negatives` 个特征索引
        sampled_indices = np.random.randint(0, high, size=(high + 1, num_negatives))
        # 避免选择相同的正向量，但保持分布均匀
        sampled_indices[sampled_indices >= feature_indices] += 1

        # 将索引映射到实际索引
        sampled_negative_indices[batch_idx][mask_time_indices[batch_idx]] = mapped_masked_indices[sampled_indices]

        # 对于批次大小进行校正
        sampled_negative_indices[batch_idx] += batch_idx * sequence_length

    return sampled_negative_indices


class Wav2Vec2NoLayerNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        # 如果layer_id大于0，则设置输入卷积维度为前一层的卷积维度，否则为1
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        # 设置输出卷积维度
        self.out_conv_dim = config.conv_dim[layer_id]

        # 定义一维卷积层
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
        # 执行前向传播
        hidden_states = self.conv(hidden_states)
        hidden_states = self.activation(hidden_states)
        return hidden_states


class Wav2Vec2LayerNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        # 如果layer_id大于0，则设置输入卷积维度为前一层的卷积维度，否则为1
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        # 设置输出卷积维度
        self.out_conv_dim = config.conv_dim[layer_id]

        # 定义一维卷积层
        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        # 定义一维卷积层后的LayerNorm层
        self.layer_norm = nn.LayerNorm(self.out_conv_dim, elementwise_affine=True)
        # 设置激活函数
        self.activation = ACT2FN[config.feat_extract_activation]
    # 定义神经网络的前向传播方法
    def forward(self, hidden_states):
        # 使用卷积层处理输入的隐藏状态
        hidden_states = self.conv(hidden_states)

        # 对隐藏状态进行转置操作，交换倒数第二个和倒数第一个维度
        hidden_states = hidden_states.transpose(-2, -1)

        # 对隐藏状态进行层归一化
        hidden_states = self.layer_norm(hidden_states)

        # 再次对隐藏状态进行转置操作，还原维度顺序
        hidden_states = hidden_states.transpose(-2, -1)

        # 使用激活函数处理隐藏状态
        hidden_states = self.activation(hidden_states)

        # 返回处理后的隐藏状态
        return hidden_states
class Wav2Vec2GroupNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        # 创建一个一维卷积层，用于特征提取
        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        self.activation = ACT2FN[config.feat_extract_activation]

        # 创建组范数层，对卷积层的输出进行归一化
        self.layer_norm = nn.GroupNorm(num_groups=self.out_conv_dim, num_channels=self.out_conv_dim, affine=True)

    def forward(self, hidden_states):
        # 通过卷积层提取特征
        hidden_states = self.conv(hidden_states)
        # 对特征进行组范数归一化
        hidden_states = self.layer_norm(hidden_states)
        # 对特征应用激活函数
        hidden_states = self.activation(hidden_states)
        return hidden_states


class Wav2Vec2PositionalConvEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个用于位置编码的卷积层
        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.num_conv_pos_embeddings,
            padding=config.num_conv_pos_embeddings // 2,
            groups=config.num_conv_pos_embedding_groups,
        )

        # 根据条件选择加权归一化函数
        weight_norm = nn.utils.weight_norm
        if hasattr(nn.utils.parametrizations, "weight_norm"):
            weight_norm = nn.utils.parametrizations.weight_norm

        # 如果开启了 Deepspeed Zero3，对卷积层的权重使用 GatheredParameters 进行处理
        if is_deepspeed_zero3_enabled():
            import deepspeed

            with deepspeed.zero.GatheredParameters(self.conv.weight, modifier_rank=0):
                self.conv = weight_norm(self.conv, name="weight", dim=2)
            deepspeed.zero.register_external_parameter(self, self.conv.weight_v)
            deepspeed.zero.register_external_parameter(self, self.conv.weight_g)
        else:
            self.conv = weight_norm(self.conv, name="weight", dim=2)

        # 创建一个用于填充的层
        self.padding = Wav2Vec2SamePadLayer(config.num_conv_pos_embeddings)
        # 获取激活函数
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        hidden_states = hidden_states.transpose(1, 2)

        # 通过卷积层提取特征
        hidden_states = self.conv(hidden_states)
        # 对卷积结果进行填��
        hidden_states = self.padding(hidden_states)
        # 对填充后的结果应用激活函数
        hidden_states = self.activation(hidden_states)

        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states


class Wav2Vec2SamePadLayer(nn.Module):
    def __init__(self, num_conv_pos_embeddings):
        super().__init__()
        # 计算要删除的填充数目，以确保长度为偶数
        self.num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0

    def forward(self, hidden_states):
        # 删除指定数量的填充，确保长度为偶数
        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, :, : -self.num_pad_remove]
        return hidden_states


class Wav2Vec2FeatureEncoder(nn.Module):
    """Construct the features from raw audio waveform"""
    # 初始化类，接收config参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
    
        # 根据config中的feat_extract_norm属性选择不同的卷积层列表
        if config.feat_extract_norm == "group":
            # 如果feat_extract_norm为"group"，创建包含Wav2Vec2GroupNormConvLayer的卷积层列表
            conv_layers = [Wav2Vec2GroupNormConvLayer(config, layer_id=0)] + [
                Wav2Vec2NoLayerNormConvLayer(config, layer_id=i + 1) for i in range(config.num_feat_extract_layers - 1)
            ]
        elif config.feat_extract_norm == "layer":
            # 如果feat_extract_norm为"layer"，创建包含Wav2Vec2LayerNormConvLayer的卷积层列表
            conv_layers = [
                Wav2Vec2LayerNormConvLayer(config, layer_id=i) for i in range(config.num_feat_extract_layers)
            ]
        else:
            # 如果feat_extract_norm属性不是'group'或'layer'，抛出数值错误
            raise ValueError(
                f"`config.feat_extract_norm` is {config.feat_extract_norm}, but has to be one of ['group', 'layer']"
            )
    
        # 将卷积层列表封装为nn.ModuleList
        self.conv_layers = nn.ModuleList(conv_layers)
        # 初始化梯度检查点开关和梯度要求
        self.gradient_checkpointing = False
        self._requires_grad = True
    
    # 冻结参数
    def _freeze_parameters(self):
        # 遍历所有参数，并设置梯度要求为False
        for param in self.parameters():
            param.requires_grad = False
        # 设置梯度要求为False
        self._requires_grad = False
    
    # 前向传播方法
    def forward(self, input_values):
        # 将输入值变形为二维，并赋值给hidden_states
        hidden_states = input_values[:, None]
    
        # 如果梯度要求为True且处于训练状态，将hidden_states的梯度要求设为True
        if self._requires_grad and self.training:
            hidden_states.requires_grad = True
    
        # 遍历卷积层，对hidden_states进行卷积操作
        for conv_layer in self.conv_layers:
            # 如果梯度要求为True且开启了梯度检查点并且处于训练状态，使用梯度检查点函数对卷积操作进行优化
            if self._requires_grad and self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    conv_layer.__call__,
                    hidden_states,
                )
            # 否则，对hidden_states进行普通的卷积操作
            else:
                hidden_states = conv_layer(hidden_states)
    
        return hidden_states
# 创建名为Wav2Vec2FeatureExtractor的类，继承自Wav2Vec2FeatureEncoder
class Wav2Vec2FeatureExtractor(Wav2Vec2FeatureEncoder):
    # 初始化方法，接受参数config
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 发出警告，提示该类已经被弃用，将在Transformers v5中移除，建议使用另一个类
        warnings.warn(
            f"The class `{self.__class__.__name__}` has been depreciated "
            "and will be removed in Transformers v5. "
            f"Use `{self.__class__.__bases__[0].__name__}` instead.",
            FutureWarning,
        )

# 创建名为Wav2Vec2FeatureProjection的类，继承自nn.Module
class Wav2Vec2FeatureProjection(nn.Module):
    # 初始化方法，接受参数config
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 创建LayerNorm层，指定输入特征的维度和epsilon值
        self.layer_norm = nn.LayerNorm(config.conv_dim[-1], eps=config.layer_norm_eps)
        # 创建线性变换层，进行特征投影，将最终的特征维度转换为config.hidden_size
        self.projection = nn.Linear(config.conv_dim[-1], config.hidden_size)
        # 创建Dropout层，用于特征投影后的结果进行随机失活
        self.dropout = nn.Dropout(config.feat_proj_dropout)

    # 前向传播方法
    def forward(self, hidden_states):
        # 需要保留非投影隐藏状态用于量化
        norm_hidden_states = self.layer_norm(hidden_states)
        # 进行特征投影
        hidden_states = self.projection(norm_hidden_states)
        # 对投影后的特征进行随机失活
        hidden_states = self.dropout(hidden_states)
        # 返回投影后的特征和非投影隐藏状态
        return hidden_states, norm_hidden_states

# 从transformers.models.bart.modeling_bart.BartAttention中复制得到Wav2Vec2Attention类
class Wav2Vec2Attention(nn.Module):
    """来自‘Attention Is All You Need’论文的多头注意力"""

    # 初始化方法，接受多个参数
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[Wav2Vec2Config] = None,
    ):
        # 调用父类的初始化方法
        super().__init__()
        # 初始化各种属性
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        # 如果embed_dim不能被num_heads整除，抛出ValueError
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        # 创建线性变换层，用于处理键、值、查询
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    # 重塑张量形状的方法
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    # 前向传播方法
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
class Wav2Vec2FeedForward(nn.Module):
# ...
    # 初始化函数，接受一个配置参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 初始化一个中间层的丢弃层
        self.intermediate_dropout = nn.Dropout(config.activation_dropout)
    
        # 初始化一个中间层的全连接层，输入维度为隐藏层大小，输出维度为中间层大小
        self.intermediate_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 如果配置中的隐藏激活函数为字符串类型，则查找对应的激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        # 否则直接使用配置中的隐藏激活函数
        else:
            self.intermediate_act_fn = config.hidden_act
    
        # 初始化一个输出层的全连接层，输入维度为中间层大小，输出维度为隐藏层大小
        self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 初始化一个输出层的丢弃层
        self.output_dropout = nn.Dropout(config.hidden_dropout)
    
    # 前向传播函数，接受隐藏层状态作为输入
    def forward(self, hidden_states):
        # 将隐藏层状态输入到中间层的全连接层中进行计算
        hidden_states = self.intermediate_dense(hidden_states)
        # 使用激活函数处理中间层的输出
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 对中间层的输出进行丢弃
        hidden_states = self.intermediate_dropout(hidden_states)
    
        # 将中间层的输出输入到输出层的全连接层中进行计算
        hidden_states = self.output_dense(hidden_states)
        # 对输出层的输出进行丢弃
        hidden_states = self.output_dropout(hidden_states)
        # 返回最终隐藏层状态
        return hidden_states
# 定义 Wav2Vec2EncoderLayer 类，用于 WAV2VEC2 模型的编码器层
class Wav2Vec2EncoderLayer(nn.Module):
    # 初始化函数，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 初始化自注意力机制对象，用于处理输入序列的注意力计算
        self.attention = Wav2Vec2Attention(
            embed_dim=config.hidden_size,  # 嵌入维度
            num_heads=config.num_attention_heads,  # 注意力头数
            dropout=config.attention_dropout,  # 注意力 dropout 概率
            is_decoder=False,  # 是否为解码器
        )
        # 初始化 dropout 层，用于在训练中随机丢弃部分神经元，防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout)
        # 初始化层归一化对象，对输入进行归一化处理
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化前馈网络对象，用于对输入进行非线性变换
        self.feed_forward = Wav2Vec2FeedForward(config)
        # 初始化最终层归一化对象，对输出进行归一化处理
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    # 前向传播函数，接受隐藏状态、注意力掩码和是否输出注意力权重作为参数
    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        # 保存自注意力之前的隐藏状态，用于残差连接
        attn_residual = hidden_states
        # 使用自注意力机制计算注意力，并返回计算结果、注意力权重以及可选的注意力权重
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
        )
        # 对计算结果进行 dropout
        hidden_states = self.dropout(hidden_states)
        # 将自注意力计算结果与残差连接起来
        hidden_states = attn_residual + hidden_states

        # 对连接结果进行层归一化
        hidden_states = self.layer_norm(hidden_states)
        # 将层归一化后的结果输入前馈网络，并将结果与原始输入相加
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        # 对输出进行最终的层归一化处理
        hidden_states = self.final_layer_norm(hidden_states)

        # 构建输出元组
        outputs = (hidden_states,)

        # 如果需要输出注意力权重，则将注意力权重添加到输出元组中
        if output_attentions:
            outputs += (attn_weights,)

        return outputs


# 定义 Wav2Vec2EncoderLayerStableLayerNorm 类，稳定的层归一化版本
class Wav2Vec2EncoderLayerStableLayerNorm(nn.Module):
    # 初始化函数，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 初始化自注意力机制对象，用于处理输入序列的注意力计算
        self.attention = Wav2Vec2Attention(
            embed_dim=config.hidden_size,  # 嵌入维度
            num_heads=config.num_attention_heads,  # 注意力头数
            dropout=config.attention_dropout,  # 注意力 dropout 概率
            is_decoder=False,  # 是否为解码器
        )
        # 初始化 dropout 层，用于在训练中随机丢弃部分神经元，防止过拟合
        self.dropout = nn.Dropout(config.hidden_dropout)
        # 初始化层归一化对象，对输入进行归一化处理
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化前馈网络对象，用于对输入进行非线性变换
        self.feed_forward = Wav2Vec2FeedForward(config)
        # 初始化最终层归一化对象，对输出进行归一化处理
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 如果配置中存在适配器注意力维度，则初始化适配器层对象，否则设置为 None
        if getattr(config, "adapter_attn_dim", None) is not None:
            self.adapter_layer = Wav2Vec2AttnAdapterLayer(config)
        else:
            self.adapter_layer = None

    # 前向传播函数，接受隐藏状态、注意力掩码和是否输出注意力权重作为参数
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        # 保存残差连接
        attn_residual = hidden_states
        # 对隐藏状态进行 layer normalization
        hidden_states = self.layer_norm(hidden_states)
        # 使用自注意力机制计算注意力权重和新的隐藏状态
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
        )
        # 对隐藏状态进行 dropout
        hidden_states = self.dropout(hidden_states)
        # 将残差连接与新的隐藏状态相加
        hidden_states = attn_residual + hidden_states
        # 对新的隐藏状态进行 layer normalization
        hidden_states = hidden_states + self.feed_forward(self.final_layer_norm(hidden_states))

        # 如果存在适配器层，将适配器层的输出与隐藏状态相加
        if self.adapter_layer is not None:
            hidden_states = hidden_states + self.adapter_layer(hidden_states)

        # 保存输出结果
        outputs = (hidden_states,)

        # 如果需要输出注意力权重，添加到输出结果中
        if output_attentions:
            outputs += (attn_weights,)

        # 返回结果
        return outputs
# Wav2Vec2Encoder 类继承于 PyTorch 的 nn.Module 类
class Wav2Vec2Encoder(nn.Module):
    # 初始化函数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 保存配置信息
        self.config = config
        # 创建位置卷积嵌入层
        self.pos_conv_embed = Wav2Vec2PositionalConvEmbedding(config)
        # 创建LayerNorm层
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建Dropout层
        self.dropout = nn.Dropout(config.hidden_dropout)
        # 创建Encoder层列表
        self.layers = nn.ModuleList([Wav2Vec2EncoderLayer(config) for _ in range(config.num_hidden_layers)])
        # 设置梯度检查点为 False
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
        all_hidden_states = () if output_hidden_states else None
        # 如果输出隐藏状态为真，则初始化一个空元组，否则为None
        all_self_attentions = () if output_attentions else None
        # 如果输出注意力权重为真，则初始化一个空元组，否则为None

        if attention_mask is not None:
            # 确保填充的标记输出为0
            expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
            # 将注意力掩码进行扩展以适应隐藏状态的形状
            hidden_states[~expand_attention_mask] = 0
            # 将填充的位置设为0

            # 扩展注意力掩码
            attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            # 将注意力掩码进行扩展以适应模型计算
            attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
            # 将注意力掩码中的1替换为较小的负数值
            attention_mask = attention_mask.expand(
                attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
            )
            # 扩展注意力掩码的形状以适应模型计算

        position_embeddings = self.pos_conv_embed(hidden_states)
        # 计算位置嵌入
        hidden_states = hidden_states + position_embeddings
        # 将位置嵌入添加到隐藏状态中
        hidden_states = self.layer_norm(hidden_states)
        # 使用层归一化对隐藏状态进行归一化
        hidden_states = self.dropout(hidden_states)
        # 使用dropout对隐藏状态进行处理

        deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()
        # 检查是否启用了DeepSpeed Zero3优化

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
                # 如果输出隐藏状态为真，则将当前隐藏状态添加到所有隐藏状态元组中

            # 添加LayerDrop（参见https://arxiv.org/abs/1909.11556）
            dropout_probability = torch.rand([])
            # 生成随机概率用于LayerDrop

            skip_the_layer = True if self.training and (dropout_probability < self.config.layerdrop) else False
            # 如果在训练模式下并且随机概率小于layerdrop概率，则跳过该层
            if not skip_the_layer or deepspeed_zero3_is_enabled:
                # 如果不跳过该层或者启用了DeepSpeed Zero3优化
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        layer.__call__,
                        hidden_states,
                        attention_mask,
                        output_attentions,
                    )
                    # 如果启用了梯度检查点且在训练模式下，则使用梯度检查点函数
                else:
                    layer_outputs = layer(
                        hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
                    )
                    # 否则直接调用该层
                hidden_states = layer_outputs[0]
                # 更新隐藏状态为当前层的输出

            if skip_the_layer:
                layer_outputs = (None, None)
                # 如果跳过了该层，则输出为None

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                # 如果输出注意力权重为真，则将当前层的注意力权重添加到所有自注意力元组中

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
            # 如果输出隐藏状态为真，则将最终隐藏状态添加到所有隐藏状态元组中

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
            # 如果不返回字典，则返回非空的隐藏状态、所有隐藏状态和所有自注意力元组
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
        # 否则返回基本模型输出对象，包含最终隐藏状态、所有隐藏状态和所有自注意力元组
# 定义一个 Wav2Vec2EncoderStableLayerNorm 类，继承自 nn.Module
class Wav2Vec2EncoderStableLayerNorm(nn.Module):
    # 初始化方法，接受 config 参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 保存传入的配置参数
        self.config = config
        # 创建 Wav2Vec2PositionalConvEmbedding 对象，并赋值给 pos_conv_embed
        self.pos_conv_embed = Wav2Vec2PositionalConvEmbedding(config)
        # 创建 LayerNorm 层，并赋值给 layer_norm，指定 hidden_size 和 layer_norm_eps
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建 Dropout 层，并赋值给 dropout，指定隐藏单元的丢弃概率
        self.dropout = nn.Dropout(config.hidden_dropout)
        # 创建包含多个 Wav2Vec2EncoderLayerStableLayerNorm 层的列表，并赋值给 layers，层数由 num_hidden_layers 决定
        self.layers = nn.ModuleList(
            [Wav2Vec2EncoderLayerStableLayerNorm(config) for _ in range(config.num_hidden_layers)]
        )
        # 初始化梯度检查点为 False
        self.gradient_checkpointing = False

    # 前向传播方法
    def forward(
        self,
        hidden_states,  # 输入的隐藏状态
        attention_mask=None,  # 注意力掩码，默认为 None
        output_attentions=False,  # 是否输出注意力权重，默认为 False
        output_hidden_states=False,  # 是否输出隐藏状态，默认为 False
        return_dict=True,  # 是否以字典形式返回结果，默认为 True
    ):
        # 如果不需要输出隐藏状态，则初始化为空元组
        all_hidden_states = () if output_hidden_states else None
        # 如果不需要输出注意力权重，则初始化为空元组
        all_self_attentions = () if output_attentions else None

        if attention_mask is not None:
            # 确保填充标记不被注意到
            expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
            hidden_states[~expand_attention_mask] = 0

            # 扩展注意力遮罩
            attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
            attention_mask = attention_mask.expand(
                attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
            )

        # 位置嵌入
        position_embeddings = self.pos_conv_embed(hidden_states)
        # 添加位置嵌入到隐藏状态
        hidden_states = hidden_states + position_embeddings
        # 对隐藏状态应用 dropout
        hidden_states = self.dropout(hidden_states)

        # 检查是否启用 DeepSpeed Zero3
        deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()

        # 遍历所有层
        for layer in self.layers:
            # 如果需要输出隐藏状态，则将隐藏状态添加到全部隐藏状态元组中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 添加层丢弃（LayerDrop）
            dropout_probability = torch.rand([])

            skip_the_layer = True if self.training and (dropout_probability < self.config.layerdrop) else False
            # 如果不跳过该层或者启用了 DeepSpeed Zero3
            if not skip_the_layer or deepspeed_zero3_is_enabled:
                if self.gradient_checkpointing and self.training:
                    # 梯度检查点
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

            # 如果需要输出注意力权重，则将该层的注意力权重添加到全部自注意力元组中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 使用 LayerNorm 对隐藏状态进行规范化
        hidden_states = self.layer_norm(hidden_states)

        # 如果需要输出隐藏状态，则将最终隐藏状态添加到全部隐藏状态元组中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不返回字典，则返回相应的元组结果，过滤掉 None 值
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        # 返回模型基本输出
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
class Wav2Vec2GumbelVectorQuantizer(nn.Module):
    """
    Vector quantization using gumbel softmax. See `[CATEGORICAL REPARAMETERIZATION WITH
    GUMBEL-SOFTMAX](https://arxiv.org/pdf/1611.01144.pdf) for more information.
    """

    def __init__(self, config):
        # 继承父类构造函数
        super().__init__()
        # 设置编码向量组的数目
        self.num_groups = config.num_codevector_groups
        # 设置每个编码向量组包含的编码向量的数目
        self.num_vars = config.num_codevectors_per_group

        # 检查编码向量维度是否能够被编码向量组数目整除
        if config.codevector_dim % self.num_groups != 0:
            raise ValueError(
                f"`config.codevector_dim {config.codevector_dim} must be divisible "
                f"by `config.num_codevector_groups` {self.num_groups} for concatenation"
            )

        # 存储码本变量（码字）的张量
        self.codevectors = nn.Parameter(
            torch.FloatTensor(1, self.num_groups * self.num_vars, config.codevector_dim // self.num_groups)
        )
        # 创建线性层，将卷积维度的最后一维投影到编码向量组和编码向量的数目
        self.weight_proj = nn.Linear(config.conv_dim[-1], self.num_groups * self.num_vars)

        # 可以通过衰减进行训练
        self.temperature = 2

    @staticmethod
    def _compute_perplexity(probs, mask=None):
        # 若输入的 mask 不为空，则将 mask 扩展成与概率张量相同的形状
        if mask is not None:
            mask_extended = mask.flatten()[:, None, None].expand(probs.shape)
            # 通过 mask 将概率张量中 mask 为 False 的位置设置为零
            probs = torch.where(mask_extended, probs, torch.zeros_like(probs))
            # 计算经过 mask 过滤后的概率的均值
            marginal_probs = probs.sum(dim=0) / mask.sum()
        else:
            # 计算概率张量的均值
            marginal_probs = probs.mean(dim=0)

        # 计算困惑度（perplexity）
        perplexity = torch.exp(-torch.sum(marginal_probs * torch.log(marginal_probs + 1e-7), dim=-1)).sum()
        # 返回困惑度
        return perplexity
    # 实现前向传播函数，输入隐藏状态和可选的时间索引遮盖
    def forward(self, hidden_states, mask_time_indices=None):
        # 获取隐藏状态的批量大小、序列长度和隐藏大小
        batch_size, sequence_length, hidden_size = hidden_states.shape

        # 将隐藏状态投影到代码向量维度
        hidden_states = self.weight_proj(hidden_states)
        hidden_states = hidden_states.view(batch_size * sequence_length * self.num_groups, -1)

        if self.training:
            # 通过 Gumbel 分布对隐藏状态进行采样，以获得代码向量的分布
            codevector_probs = nn.functional.gumbel_softmax(
                hidden_states.float(), tau=self.temperature, hard=True
            ).type_as(hidden_states)

            # 计算困惑度
            codevector_soft_dist = torch.softmax(
                hidden_states.view(batch_size * sequence_length, self.num_groups, -1).float(), dim=-1
            )
            perplexity = self._compute_perplexity(codevector_soft_dist, mask_time_indices)
        else:
            # 非训练模式下，采用 argmax 的方式确定代码向量的分布（one-hot编码）
            # 计算硬代码向量分布（one-hot编码）
            codevector_idx = hidden_states.argmax(dim=-1)
            codevector_probs = hidden_states.new_zeros(hidden_states.shape).scatter_(
                -1, codevector_idx.view(-1, 1), 1.0
            )
            codevector_probs = codevector_probs.view(batch_size * sequence_length, self.num_groups, -1)

            perplexity = self._compute_perplexity(codevector_probs, mask_time_indices)

        codevector_probs = codevector_probs.view(batch_size * sequence_length, -1)
        # 使用概率值检索代码向量
        codevectors_per_group = codevector_probs.unsqueeze(-1) * self.codevectors
        codevectors = codevectors_per_group.view(batch_size * sequence_length, self.num_groups, self.num_vars, -1)
        codevectors = codevectors.sum(-2).view(batch_size, sequence_length, -1)

        return codevectors, perplexity
class Wav2Vec2Adapter(nn.Module):
    def __init__(self, config):
        super().__init__()

        # 如果特征维度需要降维
        if config.output_hidden_size != config.hidden_size:
            # 创建线性投影层，将隐藏层维度降至输出维度
            self.proj = nn.Linear(config.hidden_size, config.output_hidden_size)
            # 创建层归一化层
            self.proj_layer_norm = nn.LayerNorm(config.output_hidden_size)
        else:
            self.proj = self.proj_layer_norm = None

        # 创建适配器层列表
        self.layers = nn.ModuleList(Wav2Vec2AdapterLayer(config) for _ in range(config.num_adapter_layers))
        # 设置层丢弃概率
        self.layerdrop = config.layerdrop

    def forward(self, hidden_states):
        # 如果存在投影层和层归一化层
        if self.proj is not None and self.proj_layer_norm is not None:
            # 对隐藏状态进行投影
            hidden_states = self.proj(hidden_states)
            # 对投影后的隐藏状态进行归一化
            hidden_states = self.proj_layer_norm(hidden_states)

        # 转置隐藏状态维度
        hidden_states = hidden_states.transpose(1, 2)

        # 遍历适配器层
        for layer in self.layers:
            # 生成层丢弃概率
            layerdrop_prob = np.random.random()
            # 如果不是训练状态或者层未被丢弃
            if not self.training or (layerdrop_prob > self.layerdrop):
                # 应用适配器层
                hidden_states = layer(hidden_states)

        # 再次转置隐藏状态维度
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states


class Wav2Vec2AdapterLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一维卷积层
        self.conv = nn.Conv1d(
            config.output_hidden_size,
            2 * config.output_hidden_size,
            config.adapter_kernel_size,
            stride=config.adapter_stride,
            padding=1,
        )

    def forward(self, hidden_states):
        # 应用一维卷积
        hidden_states = self.conv(hidden_states)
        # 使用门控线性单元激活函数
        hidden_states = nn.functional.glu(hidden_states, dim=1)

        return hidden_states


class Wav2Vec2AttnAdapterLayer(nn.Module):
    def __init__(self, config):
        """
        Implements adapter modules directly with 3D tensor weight as parameters and without using ModuleList to speed
        up training throughput.
        """
        super().__init__()
        self.input_dim = config.adapter_attn_dim
        self.hidden_dim = config.hidden_size

        # 创建层归一化层
        self.norm = nn.LayerNorm(self.hidden_dim)
        # 创建线性层1
        self.linear_1 = nn.Linear(self.hidden_dim, self.input_dim)
        # 创建激活函数
        self.act_fn = nn.ReLU()
        # 创建线性层2
        self.linear_2 = nn.Linear(self.input_dim, self.hidden_dim)

    def forward(self, hidden_states: torch.FloatTensor):
        # 归一化隐藏状态
        hidden_states = self.norm(hidden_states)

        # 应用线性层1和激活函数
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        # 应用线性层2
        hidden_states = self.linear_2(hidden_states)

        return hidden_states


class Wav2Vec2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = Wav2Vec2Config
    base_model_prefix = "wav2vec2"
    main_input_name = "input_values"
    supports_gradient_checkpointing = True
    def _init_weights(self, module):
        """Initialize the weights"""
        # 如果是 Wav2Vec2ForPreTraining 类型的模块
        if isinstance(module, Wav2Vec2ForPreTraining):
            # 重置隐藏层和查询层的参数
            module.project_hid.reset_parameters()
            module.project_q.reset_parameters()
            # 标记参数已经由 Hugging Face 初始化
            module.project_hid._is_hf_initialized = True
            module.project_q._is_hf_initialized = True
        # 如果是 Wav2Vec2GumbelVectorQuantizer 类型的模块
        elif isinstance(module, Wav2Vec2GumbelVectorQuantizer):
            # 初始化 Gumbel Softmax 相关参数
            module.weight_proj.weight.data.normal_(mean=0.0, std=1)
            module.weight_proj.bias.data.zero_()
            nn.init.uniform_(module.codevectors)
        # 如果是 Wav2Vec2PositionalConvEmbedding 类型的模块
        elif isinstance(module, Wav2Vec2PositionalConvEmbedding):
            # 初始化卷积层的参数
            nn.init.normal_(
                module.conv.weight,
                mean=0,
                std=2 * math.sqrt(1 / (module.conv.kernel_size[0] * module.conv.in_channels)),
            )
            nn.init.constant_(module.conv.bias, 0)
        # 如果是 Wav2Vec2FeatureProjection 类型的模块
        elif isinstance(module, Wav2Vec2FeatureProjection):
            # 初始化特征投影层的参数
            k = math.sqrt(1 / module.projection.in_features)
            nn.init.uniform_(module.projection.weight, a=-k, b=k)
            nn.init.uniform_(module.projection.bias, a=-k, b=k)
        # 如果是 nn.Linear 类型的模块
        elif isinstance(module, nn.Linear):
            # 初始化线性层的参数
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果有偏置项，则将其初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果是 nn.LayerNorm 或 nn.GroupNorm 类型的模块
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            # 将偏置项初始化为零，将权重初始化为 1.0
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        # 如果是 nn.Conv1d 类型的模块
        elif isinstance(module, nn.Conv1d):
            # 使用 Kaiming 正态分布初始化卷积层的参数
            nn.init.kaiming_normal_(module.weight)
            # 如果有偏置项，则将其初始化为 Kaiming 均匀分布
            if module.bias is not None:
                k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, a=-k, b=k)

    def _get_feat_extract_output_lengths(
        self, input_lengths: Union[torch.LongTensor, int], add_adapter: Optional[bool] = None
    ):
        """
        Computes the output length of the convolutional layers
        """

        add_adapter = self.config.add_adapter if add_adapter is None else add_adapter

        def _conv_out_length(input_length, kernel_size, stride):
            # 计算 1D 卷积层的输出长度
            return torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1

        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            # 根据卷积核大小和步长计算输出长度
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        if add_adapter:
            for _ in range(self.config.num_adapter_layers):
                # 如果需要添加适配器，再次计算适配器的输出长度
                input_lengths = _conv_out_length(input_lengths, 1, self.config.adapter_stride)

        return input_lengths
    # 获取特征向量注意力掩码
    def _get_feature_vector_attention_mask(
        self, feature_vector_length: int, attention_mask: torch.LongTensor, add_adapter=None
    ):
        # 计算有效的注意力掩码长度，但不改变原 attention_mask 以便在推理模式下运行
        non_padded_lengths = attention_mask.cumsum(dim=-1)[:, -1]

        # 获取特征提取输出长度，可能需要使用适配器（add_adapter）
        output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths, add_adapter=add_adapter)
        # 将输出长度转换为长整型
        output_lengths = output_lengths.to(torch.long)

        # 获取 batch 大小
        batch_size = attention_mask.shape[0]

        # 初始化新的注意力掩码，形状为 (batch_size, feature_vector_length)
        attention_mask = torch.zeros(
            (batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
        )
        # 这些操作确保在输出长度索引之前的所有值都被关注
        attention_mask[(torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1)] = 1
        # 翻转注意力掩码，计算累积和，再次翻转以得到最终的注意力掩码
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
        return attention_mask

    # 获取适配器权重
    def _get_adapters(self):
        # 如果没有定义适配器维度，则抛出错误
        if self.config.adapter_attn_dim is None:
            raise ValueError(f"{self.__class__} 没有适配器层。确保定义 `config.adapter_attn_dim`。")

        # 初始化适配器权重字典
        adapter_weights = {}
        # 遍历所有模块，获取适配器层参数
        for name, module in self.named_modules():
            if isinstance(module, Wav2Vec2AttnAdapterLayer):
                for param_name, param in module.named_parameters():
                    adapter_weights[".".join([name, param_name])] = param

        # 如果是 Wav2Vec2ForCTC 模型，还需要获取 lm_head 参数
        if isinstance(self, Wav2Vec2ForCTC):
            for name, param in self.lm_head.named_parameters():
                adapter_weights[".".join(["lm_head", name])] = param

        # 返回适配器权重
        return adapter_weights

    # 初始化适配器层
    def init_adapter_layers(self):
        """
        (重新)初始化注意力适配器层和语言模型头部，以便进行适配器微调
        """
        # 初始化注意力适配器层
        for module in self.modules():
            if isinstance(module, Wav2Vec2AttnAdapterLayer):
                self._init_weights(module)

        # 初始化语言模型头部
        if isinstance(self, Wav2Vec2ForCTC):
            self._init_weights(self.lm_head)
WAV_2_VEC_2_START_DOCSTRING = r"""
    # Wav2Vec2模型是由Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael Auli在论文[《wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations》](https://arxiv.org/abs/2006.11477)中提出的。
    
    # 该模型继承自`PreTrainedModel`类。查看父类文档以了解库实现的所有通用方法（比如下载或保存等）。
    
    # 该模型是PyTorch的[torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module)子类。可以像普通的PyTorch模块一样使用，并参考PyTorch文档以了解一切与一般用法和行为相关的事项。
    
    # 参数:
    #     config ([`Wav2Vec2Config`]): 模型的配置类，包含模型的所有参数。
    #         用配置文件初始化模型不会加载与模型关联的权重，只会加载配置。查看[`~PreTrainedModel.from_pretrained`]方法以加载模型权重。
"""


WAV_2_VEC_2_INPUTS_DOCSTRING = r"""
    # 这个地方需要继续添加注释
"""
    Args:
        input_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            输入的原始语音波形的浮点值。可以通过将 `.flac` 或 `.wav` 音频文件加载到类型为 `List[float]` 或 `numpy.ndarray` 的数组中获取值，例如通过 `soundfile` 库 (`pip install soundfile`)。要将数组准备为 `input_values`，应使用 [`AutoProcessor`] 进行填充并转换为类型为 `torch.FloatTensor` 的张量。有关详细信息，请参阅 [`Wav2Vec2Processor.__call__`]。

        attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            避免对填充标记索引执行卷积和注意力操作的掩码。掩码值选在 `[0, 1]` 之间：

            - 对于 **未屏蔽的** 标记，值为 1，
            - 对于 **已屏蔽的** 标记，值为 0。

            [什么是注意力掩码?](../glossary#attention-mask)

            <Tip warning={true}>

            只有当相应的处理器具有 `config.return_attention_mask == True` 时，才应传递 `attention_mask`。对于所有处理器具有 `config.return_attention_mask == False` 的模型，例如
            [wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base-960h)，不应传递 `attention_mask`，以避免在执行批处理推理时性能下降。对于这些模型，应仅使用 0 进行填充 `input_values` 并在不传递 `attention_mask` 的情况下传递。请注意，这些模型根据是否对 `input_values` 进行了填充，会产生略有不同的结果。

            </Tip>

        output_attentions (`bool`, *optional*):
            是否返回所有注意力层的注意力张量。有关更多详细信息，请参阅返回的张量下的 `attentions`。

        output_hidden_states (`bool`, *optional*):
            是否返回所有层的隐藏状态。有关更多详细信息，请参阅返回的张量下的 `hidden_states`。

        return_dict (`bool`, *optional*):
            是否返回 [`~utils.ModelOutput`] 而不是普通元组。
"""
The bare Wav2Vec2 Model transformer outputting raw hidden-states without any specific head on top.

这是一个裸的 Wav2Vec2 模型变压器，输出没有特定头部的原始隐藏状态。
"""
@add_start_docstrings(
    "The bare Wav2Vec2 Model transformer outputting raw hidden-states without any specific head on top.",
    WAV_2_VEC_2_START_DOCSTRING,
)
class Wav2Vec2Model(Wav2Vec2PreTrainedModel):
    def __init__(self, config: Wav2Vec2Config):
        """
        Wav2Vec2Model 构造函数，初始化模型

        Params:
            config (Wav2Vec2Config): 模型配置
        """
        super().__init__(config)
        self.config = config
        self.feature_extractor = Wav2Vec2FeatureEncoder(config)
        self.feature_projection = Wav2Vec2FeatureProjection(config)

        # model only needs masking vector if mask prob is > 0.0
        # 如果 mask  的概率大于0.0，则模型只需要遮蔽向量
        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0:
            self.masked_spec_embed = nn.Parameter(torch.FloatTensor(config.hidden_size).uniform_())

        if config.do_stable_layer_norm:
            self.encoder = Wav2Vec2EncoderStableLayerNorm(config)
        else:
            self.encoder = Wav2Vec2Encoder(config)

        self.adapter = Wav2Vec2Adapter(config) if config.add_adapter else None

        # Initialize weights and apply final processing
        # 初始化权重并应用最终处理
        self.post_init()

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameters will
        not be updated during training.

        调用此函数将禁用特征编码器的梯度计算，使其在训练过程中不会更新其参数。
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

        调用此函数将禁用特征编码器的梯度计算，使其在训练过程中不会更新其参数。
        """
        self.feature_extractor._freeze_parameters()

    def _mask_hidden_states(
        self,
        hidden_states: torch.FloatTensor,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        
        # Define function to mask hidden states based on certain indices and attention mask
        # 根据特定索引和注意力遮蔽来遮蔽隐藏状态
    ):
        """
        Masks extracted features along time axis and/or along feature axis according to
        [SpecAugment](https://arxiv.org/abs/1904.08779).
        """

        # `config.apply_spec_augment`可以将mask设置为False
        if not getattr(self.config, "apply_spec_augment", True):
            return hidden_states

        # 生成索引并沿时间轴应用SpecAugment
        batch_size, sequence_length, hidden_size = hidden_states.size()

        if mask_time_indices is not None:
            # 使用给定的mask_time_indices沿时间轴应用SpecAugment
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)
        elif self.config.mask_time_prob > 0 and self.training:
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
            # 生成索引并沿特征轴应用SpecAugment
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

    @add_start_docstrings_to_model_forward(WAV_2_VEC_2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=Wav2Vec2BaseModelOutput,
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
        ) -> Union[Tuple, Wav2Vec2BaseModelOutput]:
        # 如果输出注意力张量未指定，则使用配置中的默认值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果输出隐藏状态未指定，则使用配置中的默认值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果返回字典未指定，则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 提取输入特征
        extract_features = self.feature_extractor(input_values)
        # 转置提取的特征向量的维度
        extract_features = extract_features.transpose(1, 2)

        # 如果有注意力掩码
        if attention_mask is not None:
            # 计算与特征向量对应的减少的注意力掩码
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        # 特征投影和隐藏状态
        hidden_states, extract_features = self.feature_projection(extract_features)
        # 掩盖隐藏状态
        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )

        # 编码器输出
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 更新隐藏状态
        hidden_states = encoder_outputs[0]

        # 如果存在适配器
        if self.adapter is not None:
            # 应用适配器
            hidden_states = self.adapter(hidden_states)

        # 如果不返回字典
        if not return_dict:
            # 返回元组
            return (hidden_states, extract_features) + encoder_outputs[1:]

        # 返回 Wav2Vec2BaseModelOutput 对象
        return Wav2Vec2BaseModelOutput(
            last_hidden_state=hidden_states,
            extract_features=extract_features,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
# 使用注释说明Wav2Vec2ForPreTraining模型的功能
@add_start_docstrings("""Wav2Vec2 Model with a quantizer and `VQ` head on top.""", WAV_2_VEC_2_START_DOCSTRING)
# 初始化Wav2Vec2ForPreTraining类，继承自Wav2Vec2PreTrainedModel类
class Wav2Vec2ForPreTraining(Wav2Vec2PreTrainedModel):
    # 初始化函数，接受Wav2Vec2Config类型的参数
    def __init__(self, config: Wav2Vec2Config):
        super().__init__(config)
        # 创建Wav2Vec2Model对象
        self.wav2vec2 = Wav2Vec2Model(config)
        # 创建指定dropout比例的特征丢弃层
        self.dropout_features = nn.Dropout(config.feat_quantizer_dropout)
        
        # 创建Wav2Vec2GumbelVectorQuantizer对象
        self.quantizer = Wav2Vec2GumbelVectorQuantizer(config)
        
        # 创建隐藏层到项目化编码向量维度的线性层
        self.project_hid = nn.Linear(config.hidden_size, config.proj_codevector_dim)
        # 创建编码向量维度到项目化编码向量维度的线性层
        self.project_q = nn.Linear(config.codevector_dim, config.proj_codevector_dim)

        # 初始化权重并进行最终处理
        self.post_init()

    # 设置Gumbel softmax温度到指定值，仅在训练时需要
    def set_gumbel_temperature(self, temperature: int):
        """
        Set the Gumbel softmax temperature to a given value. Only necessary for training
        """
        self.quantizer.temperature = temperature

    # 冻结特征提取器的梯度计算，使其参数在训练时不会更新
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

    # 冻结特征提取器的梯度计算，使其参数在训练时不会更新
    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.wav2vec2.feature_extractor._freeze_parameters()

    # 计算对比损失的logits，基于余弦相似度作为距离计算
    @staticmethod
    def compute_contrastive_logits(
        target_features: torch.FloatTensor,
        negative_features: torch.FloatTensor,
        predicted_features: torch.FloatTensor,
        temperature: int = 0.1,
    ):
        """
        Compute logits for contrastive loss based using cosine similarity as the distance measure between
        `[positive_feature, negative_features]` and `[predicted_features]`. Additionally, temperature can be applied.
        """
        # 拼接正样本特征和负样本特征
        target_features = torch.cat([target_features, negative_features], dim=0)

        # 计算余弦相似度作为logits
        logits = torch.cosine_similarity(predicted_features.float(), target_features.float(), dim=-1).type_as(
            target_features
        )

        # 应用温度
        logits = logits / temperature
        return logits

    # 重写模型前向传播方法的文档字符串
    @add_start_docstrings_to_model_forward(WAV_2_VEC_2_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Wav2Vec2ForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    # 定义一个前向传播方法，用于模型推理
    def forward(
        self,
        # 输入值，可以是张量类型，可选参数
        input_values: Optional[torch.Tensor],
        # 注意力掩码，可选参数，默认为None
        attention_mask: Optional[torch.Tensor] = None,
        # 掩码时间索引，可选参数，默认为None
        mask_time_indices: Optional[torch.BoolTensor] = None,
        # 负采样的索引，可选参数，默认为None
        sampled_negative_indices: Optional[torch.BoolTensor] = None,
        # 是否输出注意力权重，可选参数，默认为None
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态，可选参数，默认为None
        output_hidden_states: Optional[bool] = None,
        # 是否返回字典，可选参数，默认为None
        return_dict: Optional[bool] = None,
# 为带有”语言模型“的Wav2Vec2模型添加注释
class Wav2Vec2ForMaskedLM(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 引发警告，该类被弃用，建议使用Wav2Vec2ForCTC
        warnings.warn(
            "The class `Wav2Vec2ForMaskedLM` is deprecated. Please use `Wav2Vec2ForCTC` instead.", FutureWarning
        )

        # 创建Wav2Vec2Model对象
        self.wav2vec2 = Wav2Vec2Model(config)
        # 创建丢弃层
        self.dropout = nn.Dropout(config.final_dropout)
        # 创建线性层
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

        # 初始化权重并应用最终处理
        self.post_init()

    # 针对model_forward函数添加注释
    @add_start_docstrings_to_model_forward(WAV_2_VEC_2_INPUTS_DOCSTRING)
    def forward(
        self,
        input_values: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, MaskedLMOutput]:
        # 如果return_dict已经定义，使用自定义的return_dict，否则使用默认的
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 运行wav2vec2模型
        outputs = self.wav2vec2(
            input_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取隐藏状态，并且使用dropout
        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)
        # 将隐藏状态输入线性层得到logits
        logits = self.lm_head(hidden_states)

        # 如果return_dict为False，则返回outputs列表，否则返回MaskedLMOutput对象
        if not return_dict:
            output = (logits,) + outputs[2:]
            return output

        return MaskedLMOutput(logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)


# 为带有CTC（Connectionist Temporal Classification）语言模型的Wav2Vec2模型添加注释
@add_start_docstrings(
    """Wav2Vec2 Model with a `language modeling` head on top for Connectionist Temporal Classification (CTC).""",
    WAV_2_VEC_2_START_DOCSTRING,
    """
        target_lang (`str`, *optional*):
            Language id of adapter weights. Adapter weights are stored in the format adapter.<lang>.safetensors or
            adapter.<lang>.bin. Only relevant when using an instance of [`Wav2Vec2ForCTC`] with adapters. Uses 'eng' by
            default.
    """,
)
class Wav2Vec2ForCTC(Wav2Vec2PreTrainedModel):
    def __init__(self, config, target_lang: Optional[str] = None):
        # 调用父类的构造函数，初始化对象
        super().__init__(config)

        # 初始化 Wav2Vec2 模型
        self.wav2vec2 = Wav2Vec2Model(config)
        # 初始化 dropout 层
        self.dropout = nn.Dropout(config.final_dropout)

        # 设置目标语言
        self.target_lang = target_lang

        # 如果配置中未定义语言模型头的词汇表大小，则引发错误
        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `Wav2Vec2ForCTC.from_pretrained(..., vocab_size=vocab_size)`. "
                "or define `vocab_size` of your model's configuration."
            )
        # 根据配置设置 LM 头的输出大小
        output_hidden_size = (
            config.output_hidden_size if hasattr(config, "add_adapter") and config.add_adapter else config.hidden_size
        )
        # 初始化 LM 头线性层
        self.lm_head = nn.Linear(output_hidden_size, config.vocab_size)

        # 初始化权重并应用最终处理
        self.post_init()

    def tie_weights(self):
        """
        该方法覆盖了 [`~PreTrainedModel.tie_weights`]，以便在向 `from_pretrained(...)` 传递 `target_lang=...` 时
        能够正确加载适配器权重。

        这个方法**不**应该被用户调用，并且可能在将来被更改。
        """

        # 注意，`tie_weights` 通常用于绑定输入和输出嵌入权重。但该方法被重新用于
        # 正确加载 Wav2Vec2 的适配器层，以便我们不必为 [`PreTrainedModel`] 引入新的 API。
        # 虽然有点取巧，但 Wav2Vec2 永远不必绑定输入和输出嵌入，所以在这里重新利用这个函数是可以的。
        target_lang = self.target_lang

        # 如果目标语言不为空，并且配置中未定义适配器的注意力维度，则引发错误
        if target_lang is not None and getattr(self.config, "adapter_attn_dim", None) is None:
            raise ValueError(f"Cannot pass `target_lang`: {target_lang} if `config.adapter_attn_dim` is not defined.")
        # 如果目标语言为空，并且配置中定义了适配器的注意力维度，则记录警告信息
        elif target_lang is None and getattr(self.config, "adapter_attn_dim", None) is not None:
            logger.info("By default `target_lang` is set to 'eng'.")
        # 如果目标语言不为空，则加载适配器
        elif target_lang is not None:
            self.load_adapter(target_lang, force_load=True)

    def freeze_feature_extractor(self):
        """
        调用此函数将禁用特征编码器的梯度计算，以便在训练过程中不更新其参数。
        """
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. "
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        self.freeze_feature_encoder()
```  
    def freeze_feature_encoder(self):
        """
        调用此函数将禁用特征编码器的梯度计算，以便在训练过程中不更新其参数。
        """
        # 冻结特征编码器的参数
        self.wav2vec2.feature_extractor._freeze_parameters()

    def freeze_base_model(self):
        """
        调用此函数将禁用基础模型的梯度计算，以便在训练过程中不更新其参数。只有分类头部将被更新。
        """
        # 遍历所有参数，并将其梯度计算设为 False
        for param in self.wav2vec2.parameters():
            param.requires_grad = False

    @add_start_docstrings_to_model_forward(WAV_2_VEC_2_INPUTS_DOCSTRING)
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
    # 定义一个函数，输入为 input_values，attention_mask，output_attentions，output_hidden_states，return_dict，labels，返回类型为 Union[Tuple, CausalLMOutput]
    def forward(
        self,
        input_values,
        attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        labels=None
    ) -> Union[Tuple, CausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional`):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """
    
        # 如果 return_dict 不为空，则返回值使用 return_dict，否则使用 self.config.use_return_dict
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
        # 将输入值传入 wav2vec2 模型中，得到输出
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    
        # 获取隐藏状态并对其进行 dropout
        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)
    
        # 通过 lm_head 获取 logits
        logits = self.lm_head(hidden_states)
    
        # 初始化 loss 为 None
        loss = None
        # 如果 labels 不为空
        if labels is not None:
            # 如果 labels 中的最大值大于等于配置的 vocab_size，则抛出数值错误
            if labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")
    
                # 从 attention_mask 中获取 loss input_lengths
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)
    
            # 假设填充的标记被填充为 -100，当没有注意到时
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)
    
            # ctc_loss 不支持 fp16
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)
    
            # 禁用 cudnn 执行 ctc_loss
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
    
        # 如果 return_dict 为 False，返回(logits,) 加上 outputs[_HIDDEN_STATES_START_POSITION:] 或 ((loss,) + output) 如果 loss 不为空，否则返回 output
        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output
    
        # 返回 CausalLMOutput 对象，包括 loss, logits, hidden_states, attentions
        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )
# 根据给定的文档字符串为Wav2Vec2模型添加特定的开始文档字符串
# 表明这是一个在顶部具有序列分类头的Wav2Vec2模型（在池化输出上的线性层），用于诸如SUPERB关键词识别之类的任务。
class Wav2Vec2ForSequenceClassification(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 如果配置中具有"add_adapter"属性并且配置中设置了 add_adapter 为 True，则引发 ValueError
        if hasattr(config, "add_adapter") and config.add_adapter:
            raise ValueError(
                "Sequence classification does not support the use of Wav2Vec2 adapters (config.add_adapter=True)"
            )
        # 创建 Wav2Vec2Model 对象
        self.wav2vec2 = Wav2Vec2Model(config)
        # 计算层数（transformer 层 + 输入的嵌入层）
        num_layers = config.num_hidden_layers + 1  
        # 如果配置中设置了use_weighted_layer_sum为True，则初始化权重矩阵
        if config.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        # 使用线性层进行投影
        self.projector = nn.Linear(config.hidden_size, config.classifier_proj_size)
        # 使用线性层进行分类
        self.classifier = nn.Linear(config.classifier_proj_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 冻结特征提取器
    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameters will
        not be updated during training.
        """
        # 引发警告，表明此方法已被弃用，并将在Transformers v5中被删除
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. "
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        # 调用等效的freeze_feature_encoder方法
        self.freeze_feature_encoder()

    # 冻结特征编码器
    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        # 禁用特征编码器的梯度计算，使其参数在训练期间不会被更新
        self.wav2vec2.feature_extractor._freeze_parameters()

    # 冻结基础模型
    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        # 禁用基础模型的梯度计算，使其参数在训练期间不会被更新，只有分类头会被更新
        for param in self.wav2vec2.parameters():
            param.requires_grad = False

    # 为模型前向传播添加开始文档字符串
    # 并根据提供的参数为模型前向传播添加代码示例文档字符串
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    # 定义带有可选标签的序列分类/回归函数，接受输入值和注意力掩码
    # 标签用于计算序列分类/回归损失，标签索引应在 [0, ..., config.num_labels-1] 范围内
    # 如果 config.num_labels == 1，则计算回归损失（均方损失），如果 config.num_labels > 1，则计算分类损失（交叉熵）
    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, SequenceClassifierOutput]:
        # 如果 return_dict 参数有指定值，就使用指定的值，否则使用模型配置文件中的 use_return_dict 参数的值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 根据配置文件中的 use_weighted_layer_sum 参数值确定是否输出隐藏状态
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states
    
        # 使用 wav2vec2 模型进行前向传播，传入输入值、注意力掩码以及其他选项
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
    
        # 如果配置文件中的 use_weighted_layer_sum 参数为 True，则对隐藏状态进行加权求和
        # 具体步骤为：将隐藏状态拼接成张量，并进行权重归一化，然后对隐藏状态和权重进行加权求和
        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = torch.stack(hidden_states, dim=1)
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        # 如果配置文件中的 use_weighted_layer_sum 参数为 False，则直接使用第一个隐藏状态
        else:
            hidden_states = outputs[0]
    
        # 通过 projector 模型将隐藏状态映射到特征向量
        hidden_states = self.projector(hidden_states)
        # 如果注意力掩码为空，则通过对隐藏状态求平均值得到汇总输出
        # 否则，根据注意力掩码对隐藏状态进行加权求和得到汇总输出
        if attention_mask is None:
            pooled_output = hidden_states.mean(dim=1)
        else:
            padding_mask = self._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)
            hidden_states[~padding_mask] = 0.0
            pooled_output = hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)
    
        # 通过 classifier 模型对汇总输出进行分类
        logits = self.classifier(pooled_output)
    
        loss = None
        # 如果标签���为空，则计算分类/回归损失
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
    
        # 如果 return_dict 为 False，则返回元组形式的输出结果
        # 否则，返回 SequenceClassifierOutput 对象形式的输出结果
        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output
    
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 使用装饰器添加模型文档字符串和首个样例代码文档字符串
@add_start_docstrings(
    """
    Wav2Vec2 Model with a frame classification head on top for tasks like Speaker Diarization.
    """,
    WAV_2_VEC_2_START_DOCSTRING,
)
class Wav2Vec2ForAudioFrameClassification(Wav2Vec2PreTrainedModel):
    # 初始化方法，接收配置参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 如果配置中包含 add_adapter 并且 add_adapter 为真，抛出数值错误
        if hasattr(config, "add_adapter") and config.add_adapter:
            raise ValueError(
                "Audio frame classification does not support the use of Wav2Vec2 adapters (config.add_adapter=True)"
            )
        # 初始化 wav2vec2 模型
        self.wav2vec2 = Wav2Vec2Model(config)
        # 计算层数：transformer 层 + 输入嵌入层
        num_layers = config.num_hidden_layers + 1
        # 如果配置中使用加权层求和，创建权重参数
        if config.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        # 创建分类器层
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # 记录分类数
        self.num_labels = config.num_labels
        # 初始化权重
        self.init_weights()

    # 冻结特征提取器，停止梯度计算，参数不会在训练期间更新
    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. "
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        self.freeze_feature_encoder()

    # 冻结特征编码器，停止梯度计算，参数不会在训练期间更新
    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.wav2vec2.feature_extractor._freeze_parameters()

    # 冻结基础模型，停止梯度计算，参数不会在训练期间更新
    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        for param in self.wav2vec2.parameters():
            param.requires_grad = False

    # 使用装饰器添加模型前向方法文档字符串和代码示例文档字符串
    @add_start_docstrings_to_model_forward(WAV_2_VEC_2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_FRAME_CLASS_CHECKPOINT,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="audio",
        expected_output=_FRAME_EXPECTED_OUTPUT,
    )
    # 前向传播方法，接收输入值、注意力遮罩、标签等参数
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    def forward(
        self,
        input_values: InputData,
        attention_mask: Optional[torch.BoolTensor] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        # 初始化返回字典是否为空，如果为空则使用模型配置的设定
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 如果模型配置要求输出加权层求和后的隐藏状态，则输出隐藏状态为真
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states

        # 使用wav2vec2模型进行前向传播，获取模型输出
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 如果模型配置要求使用加权层求和
        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            # 对隐藏状态进行堆叠操作
            hidden_states = torch.stack(hidden_states, dim=1)
            # 获取层的权重并进行softmax操作
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            hidden_states = outputs[0]

        # 将隐藏状态输入分类器得到logits
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            # 计算交叉熵损失
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), torch.argmax(labels.view(-1, self.num_labels), axis=1))

        # 如果不要求返回字典，则返回(logits, hidden_states)
        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return output

        # 返回Token分类器输出对象
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 定义自定义的AMSoftmaxLoss模型类，继承自nn.Module
class AMSoftmaxLoss(nn.Module):
    # 初始化方法，接受输入维度，标签数量，缩放参数和边界参数
    def __init__(self, input_dim, num_labels, scale=30.0, margin=0.4):
        super(AMSoftmaxLoss, self).__init__()
        # 设置缩放参数
        self.scale = scale
        # 设置边界参数
        self.margin = margin
        # 设置标签数量
        self.num_labels = num_labels
        # 创建一个可训练的参数张量，用于存储权重
        self.weight = nn.Parameter(torch.randn(input_dim, num_labels), requires_grad=True)
        # 创建一个交叉熵损失函数对象
        self.loss = nn.CrossEntropyLoss()

    # 前向传播方法，接受隐藏状态和标签
    def forward(self, hidden_states, labels):
        # 整形标签张量
        labels = labels.flatten()
        # 对权重张量进行L2范数归一化
        weight = nn.functional.normalize(self.weight, dim=0)
        # 对隐藏状态张量进行L2范数归一化
        hidden_states = nn.functional.normalize(hidden_states, dim=1)
        # 计算余弦相似度矩阵
        cos_theta = torch.mm(hidden_states, weight)
        # 计算psi值
        psi = cos_theta - self.margin

        # 创建独热编码张量
        onehot = nn.functional.one_hot(labels, self.num_labels)
        # 计算logits值
        logits = self.scale * torch.where(onehot.bool(), psi, cos_theta)
        # 计算损失
        loss = self.loss(logits, labels)

        return loss


# 定义自定义的TDNNLayer模型类，继承自nn.Module
class TDNNLayer(nn.Module):
    # 初始化方法，接受配置对象和层级ID
    def __init__(self, config, layer_id=0):
        super().__init__()
        # 设置输入卷积维度
        self.in_conv_dim = config.tdnn_dim[layer_id - 1] if layer_id > 0 else config.tdnn_dim[layer_id]
        # 设置输出卷积维度
        self.out_conv_dim = config.tdnn_dim[layer_id]
        # 设置卷积核大小
        self.kernel_size = config.tdnn_kernel[layer_id]
        # 设置膨胀率
        self.dilation = config.tdnn_dilation[layer_id]

        # 创建线性层用于卷积操作
        self.kernel = nn.Linear(self.in_conv_dim * self.kernel_size, self.out_conv_dim)
        # 创建ReLU激活函数对象
        self.activation = nn.ReLU()

    # 前向传播方法，接受隐藏状态张量
    def forward(self, hidden_states):
        # 为隐藏状态张量增加一个维度
        hidden_states = hidden_states.unsqueeze(1)
        # 对隐藏状态张量进行展开操作
        hidden_states = nn.functional.unfold(
            hidden_states,
            (self.kernel_size, self.in_conv_dim),
            stride=(1, self.in_conv_dim),
            dilation=(self.dilation, 1),
        )
        # 转置张量维度
        hidden_states = hidden_states.transpose(1, 2)
        # 通过线性层进行卷积操作
        hidden_states = self.kernel(hidden_states)

        # 通过激活函数进行激活操作
        hidden_states = self.activation(hidden_states)
        return hidden_states


# 定义Wav2Vec2ForXVector模型类，继承自Wav2Vec2PreTrainedModel
@add_start_docstrings(
    """
    Wav2Vec2 Model with an XVector feature extraction head on top for tasks like Speaker Verification.
    """,
    WAV_2_VEC_2_START_DOCSTRING,
)
class Wav2Vec2ForXVector(Wav2Vec2PreTrainedModel):
    # 初始化方法，接受配置对象
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)

        # 创建一个Wav2Vec2Model对象
        self.wav2vec2 = Wav2Vec2Model(config)
        num_layers = config.num_hidden_layers + 1  # transformer layers + input embeddings
        # 如果配置中使用加权层求和
        if config.use_weighted_layer_sum:
            # 创建一个可训练的参数张量，用于存储层级权重
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        # 创建线性层用于投影
        self.projector = nn.Linear(config.hidden_size, config.tdnn_dim[0])

        # 创建多个TDNNLayer对象，并组成模块列表
        tdnn_layers = [TDNNLayer(config, i) for i in range(len(config.tdnn_dim))]
        self.tdnn = nn.ModuleList(tdnn_layers)

        # 创建线性层用于特征提取
        self.feature_extractor = nn.Linear(config.tdnn_dim[-1] * 2, config.xvector_output_dim)
        # 创建线性层用于分类
        self.classifier = nn.Linear(config.xvector_output_dim, config.xvector_output_dim)

        # 创建AMSoftmaxLoss损失对象
        self.objective = AMSoftmaxLoss(config.xvector_output_dim, config.num_labels)

        # 初始化权重
        self.init_weights()
    # 冻结特征提取器的梯度计算，使其参数在训练期间不会更新
    def freeze_feature_extractor(self):
        # 发出警告，指出`freeze_feature_extractor`方法已被弃用，并将在Transformers v5中删除
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. "
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        # 调用等效的`freeze_feature_encoder`方法
        self.freeze_feature_encoder()

    # 冻结特征编码器的梯度计算，使其参数在训练期间不会更新
    def freeze_feature_encoder(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    # 冻结基础模型的梯度计算，使其参数在训练期间不会更新，只有分类头会更新
    def freeze_base_model(self):
        for param in self.wav2vec2.parameters():
            param.requires_grad = False

    # 计算TDNN层的输出长度
    def _get_tdnn_output_lengths(self, input_lengths: Union[torch.LongTensor, int]):
        # 计算1D卷积层的输出长度，采用来自https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html的公式
        def _conv_out_length(input_length, kernel_size, stride):
            return (input_length - kernel_size) // stride + 1

        # 遍历TDNN层的卷积核大小，计算最终的输出长度
        for kernel_size in self.config.tdnn_kernel:
            input_lengths = _conv_out_length(input_lengths, kernel_size, 1)

        return input_lengths

    # 实现`forward`函数，接受音频数据并返回模型输出
    @add_start_docstrings_to_model_forward(WAV_2_VEC_2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_XVECTOR_CHECKPOINT,
        output_type=XVectorOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="audio",
        expected_output=_XVECTOR_EXPECTED_OUTPUT,
    )
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        ) -> Union[Tuple, XVectorOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 设置返回结果是否以字典方式
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states
        # 如果使用加权层求和，则将输出的隐藏状态设置为True

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # 调用wav2vec2模型进行推理，并获取输出

        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = torch.stack(hidden_states, dim=1)
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
            # 如果使用加权层求和，则对隐藏状态进行加权层处理
        else:
            hidden_states = outputs[0]
            # 否则直接使用输出的隐藏状态

        hidden_states = self.projector(hidden_states)
        # 使用projector对隐藏状态进行处理

        for tdnn_layer in self.tdnn:
            hidden_states = tdnn_layer(hidden_states)
            # 对隐藏状态进行一系列TDNN层的处理

        # Statistic Pooling
        if attention_mask is None:
            mean_features = hidden_states.mean(dim=1)
            std_features = hidden_states.std(dim=1)
        else:
            feat_extract_output_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(dim=1))
            tdnn_output_lengths = self._get_tdnn_output_lengths(feat_extract_output_lengths)
            mean_features = []
            std_features = []
            for i, length in enumerate(tdnn_output_lengths):
                mean_features.append(hidden_states[i, :length].mean(dim=0))
                std_features.append(hidden_states[i, :length].std(dim=0))
            mean_features = torch.stack(mean_features)
            std_features = torch.stack(std_features)
        statistic_pooling = torch.cat([mean_features, std_features], dim=-1)
        # 进行统计池化的操作，计算均值和标准差

        output_embeddings = self.feature_extractor(statistic_pooling)
        logits = self.classifier(output_embeddings)
        # 对统计池化的结果进行特征提取和分类

        loss = None
        if labels is not None:
            loss = self.objective(logits, labels)
            # 如果给定了标签，则计算分类/回归损失

        if not return_dict:
            output = (logits, output_embeddings) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output
            # 如果不以字典形式返回结果，则返回logits和output_embeddings等信息

        return XVectorOutput(
            loss=loss,
            logits=logits,
            embeddings=output_embeddings,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        # 以XVectorOutput类型返回结果，包括损失、logits、嵌入、隐藏状态和注意力机制信息
```