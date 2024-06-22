# `.\transformers\models\unispeech_sat\modeling_unispeech_sat.py`

```py
# 导入所需的模块和类
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
    SequenceClassifierOutput,
    TokenClassifierOutput,
    Wav2Vec2BaseModelOutput,
    XVectorOutput,
)
from ...modeling_utils import PreTrainedModel
from ...utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from .configuration_unispeech_sat import UniSpeechSatConfig

# 获取日志记录器
logger = logging.get_logger(__name__)

# 隐藏状态的起始位置
_HIDDEN_STATES_START_POSITION = 2

# 文档用的一些常量
_CONFIG_FOR_DOC = "UniSpeechSatConfig"
_CHECKPOINT_FOR_DOC = "microsoft/unispeech-sat-base-100h-libri-ft"
_EXPECTED_OUTPUT_SHAPE = [1, 292, 768]
_CTC_EXPECTED_OUTPUT = "'MISTER QUILDER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL'"
_CTC_EXPECTED_LOSS = 39.88
_FRAME_CLASS_CHECKPOINT = "microsoft/unispeech-sat-base-plus-sd"
_FRAME_EXPECTED_OUTPUT = [0, 0]
_XVECTOR_CHECKPOINT = "microsoft/unispeech-sat-base-plus-sv"
_XVECTOR_EXPECTED_OUTPUT = 0.97

# 预训练模型列表
UNISPEECH_SAT_PRETRAINED_MODEL_ARCHIVE_LIST = [
    # See all UniSpeechSat models at https://huggingface.co/models?filter=unispeech_sat
]

# 定义一个输出类型
@dataclass
class UniSpeechSatForPreTrainingOutput(ModelOutput):
    """
    Output type of [`UniSpeechSatForPreTrainingOutput`], with potential hidden states and attentions.
    """


这段代码是 UniSpeechSat 模型的 PyTorch 实现的一部分。它主要做了以下几件事情:

1. 导入了一些必要的模块和类,包括 PyTorch、激活函数、日志记录器等。
2. 定义了一些常量,如文档用的checkpoint、输出形状等。
3. 定义了一个输出类型 `UniSpeechSatForPreTrainingOutput`。
4. 声明了一个预训练模型列表 `UNISPEECH_SAT_PRETRAINED_MODEL_ARCHIVE_LIST`。

总的来说,这个代码文件是 UniSpeechSat 模型的一部分,主要负责定义模型的输入输出和一些常量配置。
    # 这个代码块定义了一个 PyTorch 模型的输出类型，包括以下几种:
    # 1. loss: 模型训练过程中的总损失，包括对比损失和多样性损失
    # 2. logits: 模型的输出logits
    # 3. projected_states: 模型输出的隐状态投影到预设维度的结果
    # 4. projected_quantized_states: 从输入中提取的特征向量投影到预设维度的结果
    # 5. codevector_perplexity: 模型的码本困惑度指标
    # 6. hidden_states: 模型各层的隐状态输出
    # 7. attentions: 模型各层的注意力权重
        Args:
            loss (*optional*, returned when model is in train mode, `torch.FloatTensor` of shape `(1,)`):
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
        """
    
        loss: Optional[torch.FloatTensor] = None
        logits: torch.FloatTensor = None
        projected_states: torch.FloatTensor = None
        projected_quantized_states: torch.FloatTensor = None
        codevector_perplexity: torch.FloatTensor = None
        hidden_states: Optional[Tuple[torch.FloatTensor]] = None
        attentions: Optional[Tuple[torch.FloatTensor]] = None
# 从transformers.models.wav2vec2.modeling_wav2vec2._compute_mask_indices复制而来的函数
def _compute_mask_indices(
    shape: Tuple[int, int],  # 接受一个二元组作为shape参数，表示(batch_size, sequence_length)
    mask_prob: float,  # 浮点数，表示掩码概率
    mask_length: int,  # 整数，表示掩码长度
    attention_mask: Optional[torch.LongTensor] = None,  # 可选的参数，表示注意力掩码，默认为None
    min_masks: int = 0,  # 整数，表示最小的掩码数量，默认为0
) -> np.ndarray:  # 返回一个numpy数组
    """
    计算给定形状的随机掩码范围。用于实现SpecAugment: A Simple Data Augmentation Method for ASR中的方法。注意，此方法并未针对TPU进行优化，应该作为训练过程中的预处理步骤在CPU上运行。

    Args:
        shape: 用于计算掩码的形状。这应该是一个大小为2的元组，其中
               第一个元素是批量大小，第二个元素是轴的长度。
        mask_prob: 整个轴的百分比（在0和1之间）将被掩码。生成长度为`mask_length`的独立掩码范围的数量由`mask_prob*shape[1]/mask_length`计算得出。注意，由于重叠，`mask_prob`是一个上界，实际百分比会小一些。
        mask_length: 掩码的大小
        min_masks: 掩码范围的最小数量
        attention_mask: (右填充的) 注意力掩码，独立缩短每个批量维度的特征轴。
    """
    batch_size, sequence_length = shape  # 解构形状参数为批量大小和序列长度

    if mask_length < 1:  # 如果掩码长度小于1
        raise ValueError("`mask_length` has to be bigger than 0.")

    if mask_length > sequence_length:  # 如果掩码长度大于序列长度
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length}"
            f" and `sequence_length`: {sequence_length}`"
        )

    # epsilon用于概率舍入
    epsilon = np.random.rand(1).item()

    def compute_num_masked_span(input_length):
        """给定输入长度，计算应该掩码多少个区间"""
        num_masked_span = int(mask_prob * input_length / mask_length + epsilon)
        num_masked_span = max(num_masked_span, min_masks)

        # 确保掩码的区间数<=序列长度
        if num_masked_span * mask_length > sequence_length:
            num_masked_span = sequence_length // mask_length

        # 确保掩码区间数<=输入长度 - (mask_length - 1)
        if input_length - (mask_length - 1) < num_masked_span:
            num_masked_span = max(input_length - (mask_length - 1), 0)

        return num_masked_span

    # 计算批量中掩码区间的数量
    input_lengths = (
        attention_mask.sum(-1).detach().tolist()  # 如果attention_mask不为None，计算每个批量维度的输入长度之和，转换为列表
        if attention_mask is not None 
        else [sequence_length for _ in range(batch_size)]  # 否则，构建一个包含批量大小个元素的列表，每个元素初始化为序列长度
    )

    # SpecAugment掩码
    spec_aug_mask = np.zeros((batch_size, sequence_length), dtype=bool)  # 构建一个形状为(batch_size, sequence_length)的布尔类型的数组，初始化为False
    spec_aug_mask_idxs = []

    max_num_masked_span = compute_num_masked_span(sequence_length)  # 计算序列长度中的最大掩码区间数量
    # 如果最大被屏蔽跨度为0，则返回原始的spec_aug_mask
    if max_num_masked_span == 0:
        return spec_aug_mask

    # 遍历输入长度列表
    for input_length in input_lengths:
        # 计算当前输入的被屏蔽跨度数量
        num_masked_span = compute_num_masked_span(input_length)

        # 随机选择要屏蔽的索引位置
        spec_aug_mask_idx = np.random.choice(
            np.arange(input_length - (mask_length - 1)), num_masked_span, replace=False
        )

        # 选择第一个采样索引作为填充向量的虚拟索引
        # 以确保由于概率舍入而导致所有批次的维度相同
        # 选择第一个样本只是使这些向量填充两次。
        if len(spec_aug_mask_idx) == 0:
            # 这种情况只会在 `input_length` 严格小于 `sequence_length` 时发生，
            # 此时最后一个标记必须是填充标记，我们可以将其用作虚拟掩码id
            dummy_mask_idx = sequence_length - 1
        else:
            dummy_mask_idx = spec_aug_mask_idx[0]

        # 将填充的索引添加到被屏蔽索引中，以确保每个样本的索引数量相同
        spec_aug_mask_idx = np.concatenate(
            [spec_aug_mask_idx, np.ones(max_num_masked_span - num_masked_span, dtype=np.int32) * dummy_mask_idx]
        )
        spec_aug_mask_idxs.append(spec_aug_mask_idx)

    # 将索引数组转换为numpy数组
    spec_aug_mask_idxs = np.array(spec_aug_mask_idxs)

    # 将屏蔽的索引扩展为屏蔽的跨度
    spec_aug_mask_idxs = np.broadcast_to(
        spec_aug_mask_idxs[:, :, None], (batch_size, max_num_masked_span, mask_length)
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs.reshape(batch_size, max_num_masked_span * mask_length)

    # 将偏移添加到起始索引，以便索引现在创建一个跨度
    offsets = np.arange(mask_length)[None, None, :]
    offsets = np.broadcast_to(offsets, (batch_size, max_num_masked_span, mask_length)).reshape(
        batch_size, max_num_masked_span * mask_length
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

    # 确保我们的索引不会超过sequence_length
    if spec_aug_mask_idxs.max() > sequence_length - 1:
        spec_aug_mask_idxs[spec_aug_mask_idxs > sequence_length - 1] = sequence_length - 1

    # 在指定的索引位置上添加掩码
    np.put_along_axis(spec_aug_mask, spec_aug_mask_idxs, 1, -1)

    # 返回处理后的spec_aug_mask
    return spec_aug_mask
# 定义一个自定义的卷积层类 UniSpeechSatNoLayerNormConvLayer，继承自 nn.Module
class UniSpeechSatNoLayerNormConvLayer(nn.Module):
    # 初始化方法，接受配置和层级编号作为参数
    def __init__(self, config, layer_id=0):
        super().__init__()
        # 设置输入卷积维度为前一层卷积维度，如果是第一层则为1
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        # 设置输出卷积维度为当前层配置中的卷积维度
        self.out_conv_dim = config.conv_dim[layer_id]
        # 创建一个一维卷积层对象，配置包括输入维度、输出维度、卷积核大小、步长和偏置
        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        # 设置激活函数为配置中的特征提取激活函数
        self.activation = ACT2FN[config.feat_extract_activation]

    # 前向传播方法，接受隐藏状态作为输入
    def forward(self, hidden_states):
        # 对隐藏状态进行一维卷积操作
        hidden_states = self.conv(hidden_states)
        # 对卷积结果进行激活函数处理
        hidden_states = self.activation(hidden_states)
        # 返回处理后的隐藏状态
        return hidden_states


# 定义一个自定义的带层归一化的卷积层类 UniSpeechSatLayerNormConvLayer，继承自 nn.Module
class UniSpeechSatLayerNormConvLayer(nn.Module):
    # 初始化方法，接受配置和层级编号作为参数
    def __init__(self, config, layer_id=0):
        super().__init__()
        # 设置输入卷积维度为前一层卷积维度，如果是第一层则为1
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        # 设置输出卷积维度为当前层配置中的卷积维度
        self.out_conv_dim = config.conv_dim[layer_id]
        # 创建一个一维卷积层对象，配置包括输入维度、输出维度、卷积核大小、步长和偏置
        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        # 设置层归一化对象，归一化维度为输出卷积维度
        self.layer_norm = nn.LayerNorm(self.out_conv_dim, elementwise_affine=True)
        # 设置激活函数为配置中的特征提取激活函数
        self.activation = ACT2FN[config.feat_extract_activation]

    # 前向传播方法，接受隐藏状态作为输入
    def forward(self, hidden_states):
        # 对隐藏状态进行一维卷积操���
        hidden_states = self.conv(hidden_states)
        # 对卷积结果进行维度转换
        hidden_states = hidden_states.transpose(-2, -1)
        # 对转换后的卷积结果进行层归一化处理
        hidden_states = self.layer_norm(hidden_states)
        # 再次对结果进行维度转换
        hidden_states = hidden_states.transpose(-2, -1)
        # 对归一化结果进行激活函数处理
        hidden_states = self.activation(hidden_states)
        # 返回处理后的隐藏状态
        return hidden_states


# 定义一个自定义的带组归一化的卷积层类 UniSpeechSatGroupNormConvLayer，继承自 nn.Module
class UniSpeechSatGroupNormConvLayer(nn.Module):
    # 初始化方法，接受配置和层级编号作为参数
    def __init__(self, config, layer_id=0):
        super().__init__()
        # 设置输入卷积维度为前一层卷积维度，如果是第一层则为1
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        # 设置输出卷积维度为当前层配置中的卷积维度
        self.out_conv_dim = config.conv_dim[layer_id]
        # 创建一个一维卷积层对象，配置包括输入维度、输出维度、卷积核大小、步长和偏置
        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        # 设置激活函数为配置中的特征提取激活函数
        self.activation = ACT2FN[config.feat_extract_activation]
        # 设置组归一化对象，组数为输出卷积维度，通道数为输出卷积维度
        self.layer_norm = nn.GroupNorm(num_groups=self.out_conv_dim, num_channels=self.out_conv_dim, affine=True)

    # 前向传播方法，接受隐藏状态作为输入
    def forward(self, hidden_states):
        # 对隐藏状态进行一维卷积操作
        hidden_states = self.conv(hidden_states)
        # 对卷积结果进行组归一化处理
        hidden_states = self.layer_norm(hidden_states)
        # 对归一化结果进行激活函数处理
        hidden_states = self.activation(hidden_states)
        # 返回处理后的隐藏状态
        return hidden_states
# 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2PositionalConvEmbedding中复制，并将Wav2Vec2->UniSpeechSat
class UniSpeechSatPositionalConvEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 一维卷积层，用于对输入进行卷积操作
        self.conv = nn.Conv1d(
            config.hidden_size,  # 输入通道数
            config.hidden_size,  # 输出通道数
            kernel_size=config.num_conv_pos_embeddings,  # 卷积核大小
            padding=config.num_conv_pos_embeddings // 2,  # 填充大小
            groups=config.num_conv_pos_embedding_groups,  # 分组卷积参数
        )

        weight_norm = nn.utils.weight_norm
        if hasattr(nn.utils.parametrizations, "weight_norm"):
            weight_norm = nn.utils.parametrizations.weight_norm

        # 如果启用了Deepspeed ZeRO-3，使用ZeRO-3的参数并注册外部参数
        if is_deepspeed_zero3_enabled():
            import deepspeed

            with deepspeed.zero.GatheredParameters(self.conv.weight, modifier_rank=0):
                self.conv = weight_norm(self.conv, name="weight", dim=2)
            deepspeed.zero.register_external_parameter(self, self.conv.weight_v)
            deepspeed.zero.register_external_parameter(self, self.conv.weight_g)
        else:
            # 否则，在权重上应用权重归一化
            self.conv = weight_norm(self.conv, name="weight", dim=2)

        # 对输入进行填充以保持与输出相同的长度
        self.padding = UniSpeechSatSamePadLayer(config.num_conv_pos_embeddings)
        # 激活函数，根据配置选择不同的激活函数
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        # 将输入的隐藏状态进行转置，以适应卷积层的输入要求
        hidden_states = hidden_states.transpose(1, 2)

        # 输入进行卷积操作
        hidden_states = self.conv(hidden_states)
        # 对卷积后的输出进行填充操作
        hidden_states = self.padding(hidden_states)
        # 应用激活函数
        hidden_states = self.activation(hidden_states)

        # 将输出的隐藏状态进行转置，以还原原始维度
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states


# 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2SamePadLayer中复制，并将Wav2Vec2->UniSpeechSat
class UniSpeechSatSamePadLayer(nn.Module):
    def __init__(self, num_conv_pos_embeddings):
        super().__init__()
        # 计算需要移除的填充数，以保持与输入相同的长度
        self.num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0

    def forward(self, hidden_states):
        # 如果需要移除填充，则将填充部分从隐藏状态中截取掉
        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, :, : -self.num_pad_remove]
        return hidden_states


# 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2FeatureEncoder中复制，并将Wav2Vec2->UniSpeechSat
class UniSpeechSatFeatureEncoder(nn.Module):
    """从原始音频波形中构造特征"""
    # 初始化方法，接受一个配置对象作为参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()

        # 根据配置中的特征提取规范，选择使用不同类型的卷积层
        if config.feat_extract_norm == "group":
            # 如果特征提取规范是"group"，则创建一组UniSpeechSatGroupNormConvLayer对象
            conv_layers = [UniSpeechSatGroupNormConvLayer(config, layer_id=0)] + [
                UniSpeechSatNoLayerNormConvLayer(config, layer_id=i + 1)
                for i in range(config.num_feat_extract_layers - 1)
            ]
        elif config.feat_extract_norm == "layer":
            # 如果特征提取规范是"layer"，则创建一组UniSpeechSatLayerNormConvLayer对象
            conv_layers = [
                UniSpeechSatLayerNormConvLayer(config, layer_id=i) for i in range(config.num_feat_extract_layers)
            ]
        else:
            # 如果特征提取规范既不是"group"也不是"layer"，则抛出数值错误
            raise ValueError(
                f"`config.feat_extract_norm` is {config.feat_extract_norm}, but has to be one of ['group', 'layer']"
            )
        # 用创建的卷积层列表初始化ModuleList
        self.conv_layers = nn.ModuleList(conv_layers)
        self.gradient_checkpointing = False
        self._requires_grad = True

    # 冻结参数的方法
    def _freeze_parameters(self):
        # 遍历所有参数，设置requires_grad为False
        for param in self.parameters():
            param.requires_grad = False
        # 设置_requires_grad为False
        self._requires_grad = False

    # 前向传播方法
    def forward(self, input_values):
        # 将输入值转为2维的hidden_states
        hidden_states = input_values[:, None]

        # 如果_requires_grad为True且当前处于训练模式，则将hidden_states设置为可梯度
        if self._requires_grad and self.training:
            hidden_states.requires_grad = True

        # 遍历所有卷积层并进行前向传播
        for conv_layer in self.conv_layers:
            if self._requires_grad and self.gradient_checkpointing and self.training:
                # 如果_requires_grad为True，gradient_checkpointing为True，且处于训练模式，则使用gradient_checkpointing_func进行前向传播
                hidden_states = self._gradient_checkpointing_func(
                    conv_layer.__call__,
                    hidden_states,
                )
            else:
                # 否则直接使用卷积层进行前向传播
                hidden_states = conv_layer(hidden_states)

        # 返回最终的hidden_states
        return hidden_states
# 定义类 UniSpeechSatFeatureExtractor，继承自 UniSpeechSatFeatureEncoder
class UniSpeechSatFeatureExtractor(UniSpeechSatFeatureEncoder):
    # 初始化方法，接受配置参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 发出警告信息，表明这个类已弃用，未来版本会移除
        warnings.warn(
            f"The class `{self.__class__.__name__}` has been depreciated "
            "and will be removed in Transformers v5. "
            f"Use `{self.__class__.__bases__[0].__name__}` instead.",
            FutureWarning,
        )


# 定义类 UniSpeechSatFeatureProjection，继承自 nn.Module
# 这是从 transformers 模型 wav2vec2 的模型代码中复制而来，Wav2Vec2 被替换成 UniSpeechSat
class UniSpeechSatFeatureProjection(nn.Module):
    # 初始化方法，接受配置参数
    def __init__(self, config):
        # 调用 nn.Module 的初始化方法
        super().__init__()
        # 初始化层归一化，使用配置中的卷积维度的最后一个元素，以及指定的 epsilon
        self.layer_norm = nn.LayerNorm(config.conv_dim[-1], eps=config.layer_norm_eps)
        # 初始化线性投影，输入和输出维度根据配置参数定义
        self.projection = nn.Linear(config.conv_dim[-1], config.hidden_size)
        # 初始化 dropout 层，使用配置中的 dropout 概率
        self.dropout = nn.Dropout(config.feat_proj_dropout)

    # 前向传递方法，接受隐藏状态作为参数
    def forward(self, hidden_states):
        # 对隐藏状态进行层归一化
        norm_hidden_states = self.layer_norm(hidden_states)
        # 对归一化后的隐藏状态进行线性投影
        hidden_states = self.projection(norm_hidden_states)
        # 对投影后的隐藏状态应用 dropout
        hidden_states = self.dropout(hidden_states)
        # 返回投影后的隐藏状态和归一化后的隐藏状态
        return hidden_states, norm_hidden_states


# 定义类 UniSpeechSatAttention，继承自 nn.Module
# 这是从 transformers 模型 bart 的模型代码中复制而来，Bart 被替换成 UniSpeechSat
class UniSpeechSatAttention(nn.Module):
    """多头注意力机制，源自 'Attention Is All You Need' 论文"""

    # 初始化方法，接受嵌入维度、头数量、dropout 率等参数
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False,
        config: Optional[UniSpeechSatConfig] = None,
    ):
        # 调用 nn.Module 的初始化方法
        super().__init__()
        # 设置嵌入维度
        self.embed_dim = embed_dim
        # 设置多头注意力的头数量
        self.num_heads = num_heads
        # 设置 dropout 率
        self.dropout = dropout
        # 计算每个头的维度
        self.head_dim = embed_dim // num_heads
        # 保存配置
        self.config = config

        # 确保嵌入维度能被头数量整除，否则抛出错误
        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        # 计算缩放因子，用于注意力机制
        self.scaling = self.head_dim**-0.5
        # 设置是否为解码器
        self.is_decoder = is_decoder
        # 设置是否为因果关系
        self.is_causal = is_causal

        # 定义线性投影，用于键、值、查询以及输出
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    # 辅助方法，用于调整张量的形状
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        # 调整张量的形状，使其符合多头注意力的要求
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    # 前向传递方法，接受隐藏状态、键值状态、过去的键值、注意力掩码等参数
    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
# 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2FeedForward复制，并将Wav2Vec2改为UniSpeechSat
class UniSpeechSatFeedForward(nn.Module):
    # 初始化方法，接收config参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个丢弃层，丢弃率为config.activation_dropout
        self.intermediate_dropout = nn.Dropout(config.activation_dropout)
        # 创建一个全连接层，输入维度为config.hidden_size，输出维度为config.intermediate_size
        self.intermediate_dense = nn.Linear(config.hidden_size, config.intermediate_size)
        # 如果config.hidden_act为字符串类型，则使用ACT2FN字典中对应的激活函数，否则使用config.hidden_act对应的激活函数
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act
        # 创建一个全连接层，输入维度为config.intermediate_size，输出维度为config.hidden_size
        self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)
        # 创建一个丢弃层，丢弃率为config.hidden_dropout
        self.output_dropout = nn.Dropout(config.hidden_dropout)

    # 前向传播方法，接收hidden_states参数
    def forward(self, hidden_states):
        # 使用intermediate_dense层对hidden_states进行全连接操作
        hidden_states = self.intermediate_dense(hidden_states)
        # 使用激活函数对hidden_states进行激活
        hidden_states = self.intermediate_act_fn(hidden_states)
        # 使用intermediate_dropout进行丢弃操作
        hidden_states = self.intermediate_dropout(hidden_states)
        # 使用output_dense对hidden_states进行全连接操作
        hidden_states = self.output_dense(hidden_states)
        # 使用output_dropout进行丢弃操作
        hidden_states = self.output_dropout(hidden_states)
        # 返回hidden_states
        return hidden_states


# 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2EncoderLayer复制，并将Wav2Vec2改为UniSpeechSat
class UniSpeechSatEncoderLayer(nn.Module):
    # 初始化方法，接收config参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 创建一个UniSpeechSatAttention对象，参数为config中的一些设置
        self.attention = UniSpeechSatAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
        )
        # 创建一个丢弃层，丢弃率为config.hidden_dropout
        self.dropout = nn.Dropout(config.hidden_dropout)
        # 创建一个LayerNorm层，输入维度为config.hidden_size
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个UniSpeechSatFeedForward对象，参数为config
        self.feed_forward = UniSpeechSatFeedForward(config)
        # 创建一个LayerNorm层，输入维度为config.hidden_size
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    # 前向传播方法，接收hidden_states, attention_mask, output_attentions参数
    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        # 保存注意力残差
        attn_residual = hidden_states
        # 使用attention对hidden_states进行处理
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
        )
        # 使用dropout进行丢弃操作
        hidden_states = self.dropout(hidden_states)
        # 更新hidden_states
        hidden_states = attn_residual + hidden_states
        # 使用layer_norm层对hidden_states进行LayerNorm操作
        hidden_states = self.layer_norm(hidden_states)
        # 使用feed_forward对hidden_states进行处理
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        # 使用final_layer_norm对hidden_states进行LayerNorm操作
        hidden_states = self.final_layer_norm(hidden_states)

        # 输出为hidden_states
        outputs = (hidden_states,)

        # 如果output_attentions为True，则输出attn_weights
        if output_attentions:
            outputs += (attn_weights,)

        return outputs


# 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2AttnAdapterLayer复制，并将Wav2Vec2改为UniSpeechSat
class UniSpeechSatAttnAdapterLayer(nn.Module):
    # 初始化函数，接受配置参数 config
    def __init__(self, config):
        """
        Implements adapter modules directly with 3D tensor weight as parameters and without using ModuleList to speed
        up training throughput.
        """
        调用父类的初始化函数
        super().__init__()
        # 从配置参数中获取输入维度和隐藏维度
        self.input_dim = config.adapter_attn_dim
        self.hidden_dim = config.hidden_size

        # 对输入进行层归一化
        self.norm = nn.LayerNorm(self.hidden_dim)
        # 第一个线性变换，将隐藏状态维度映射到输入维度
        self.linear_1 = nn.Linear(self.hidden_dim, self.input_dim)
        # 激活函数使用 ReLU
        self.act_fn = nn.ReLU()
        # 第二个线性变换，将输入维度映射回隐藏状态维度
        self.linear_2 = nn.Linear(self.input_dim, self.hidden_dim)

    # 前向传播函数，接受隐藏状态张量作为输入
    def forward(self, hidden_states: torch.FloatTensor):
        # 对隐藏状态进行层归一化
        hidden_states = self.norm(hidden_states)

        # 第一个线性变换
        hidden_states = self.linear_1(hidden_states)
        # 使用 ReLU 激活函数
        hidden_states = self.act_fn(hidden_states)
        # 第二个线性变换
        hidden_states = self.linear_2(hidden_states)

        # 返回处理后的隐藏状态
        return hidden_states
# 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2EncoderLayerStableLayerNorm复制代码，将Wav2Vec2更改为UniSpeechSat
class UniSpeechSatEncoderLayerStableLayerNorm(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 初始化注意力层
        self.attention = UniSpeechSatAttention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
        )
        # 初始化Dropout
        self.dropout = nn.Dropout(config.hidden_dropout)
        # 初始化LayerNorm
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化前馈神经网络
        self.feed_forward = UniSpeechSatFeedForward(config)
        # 初始化最终归一化层
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 如果config中包含adapter_attn_dim，则初始化适配器层
        if getattr(config, "adapter_attn_dim", None) is not None:
            self.adapter_layer = UniSpeechSatAttnAdapterLayer(config)
        else:
            self.adapter_layer = None

    # 前向传播函数
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        attn_residual = hidden_states
        hidden_states = self.layer_norm(hidden_states)
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states
        hidden_states = hidden_states + self.feed_forward(self.final_layer_norm(hidden_states))

        if self.adapter_layer is not None:
            hidden_states = hidden_states + self.adapter_layer(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


# 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2Encoder复制代码，将Wav2Vec2更改为UniSpeechSat
class UniSpeechSatEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        # 初始化位置卷积嵌入层
        self.pos_conv_embed = UniSpeechSatPositionalConvEmbedding(config)
        # 初始化LayerNorm
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 初始化Dropout
        self.dropout = nn.Dropout(config.hidden_dropout)
        # 初始化编码器层组成的列表
        self.layers = nn.ModuleList([UniSpeechSatEncoderLayer(config) for _ in range(config.num_hidden_layers)])
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
            # 初始化用于存储所有隐藏状态的空元组，如果不输出隐藏状态则为 None
            all_hidden_states = () if output_hidden_states else None
            # 初始化用于存储所有自注意力值的空元组，如果不输出注意力值则为 None
            all_self_attentions = () if output_attentions else None

            if attention_mask is not None:
                # 确保填充的 token 输出为 0
                expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
                hidden_states[~expand_attention_mask] = 0

                # 扩展 attention_mask
                attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
                attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
                attention_mask = attention_mask.expand(
                    attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
                )

            # 通过位置卷积嵌入层对隐藏状态进行处理
            position_embeddings = self.pos_conv_embed(hidden_states)
            hidden_states = hidden_states + position_embeddings
            hidden_states = self.layer_norm(hidden_states)
            hidden_states = self.dropout(hidden_states)

            # 检查是否启用了 DeepSpeed Zero3
            deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()

            # 遍历每个层进行处理
            for layer in self.layers:
                if output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                # 添加 LayerDrop (参见 https://arxiv.org/abs/1909.11556 进行描述)
                dropout_probability = torch.rand([])

                skip_the_layer = True if self.training and (dropout_probability < self.config.layerdrop) else False
                if not skip_the_layer or deepspeed_zero3_is_enabled:
                    # 在深度学习 zero3 下所有 GPU 必须同步运行
                    if self.gradient_checkpointing and self.training:
                        # 计算使用梯度检查点拆分的层的输出
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

            # 如果不返回字典，则返回相应结果元组
            if not return_dict:
                return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
            # 返回 BaseModelOutput 类的对象
            return BaseModelOutput(
                last_hidden_state=hidden_states,
                hidden_states=all_hidden_states,
                attentions=all_self_attentions,
            )
# 定义一个新的编码器类 UniSpeechSatEncoderStableLayerNorm，继承自 nn.Module 类
class UniSpeechSatEncoderStableLayerNorm(nn.Module):
    # 初始化函数，接受一个配置参数 config
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__()
        # 将配置参数保存在当前对象中
        self.config = config
        # 创建位置卷积嵌入对象，使用 UniSpeechSatPositionalConvEmbedding 类
        self.pos_conv_embed = UniSpeechSatPositionalConvEmbedding(config)
        # 创建层归一化对象，使用 nn.LayerNorm 类，指定隐藏层大小和层归一化的 epsilon 值
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建 Dropout 对象，使用 nn.Dropout 类，指定隐藏层的 dropout 概率
        self.dropout = nn.Dropout(config.hidden_dropout)
        # 创建编码器层的列表，使用 nn.ModuleList 类，其中每个元素都是 UniSpeechSatEncoderLayerStableLayerNorm 类的实例，
        # 个数等于配置参数中的隐藏层层数 config.num_hidden_layers
        self.layers = nn.ModuleList(
            [UniSpeechSatEncoderLayerStableLayerNorm(config) for _ in range(config.num_hidden_layers)]
        )
        # 是否使用梯度检查点技术，默认为 False
        self.gradient_checkpointing = False

    # 前向传播函数，接受输入 hidden_states 和一些可选参数
    def forward(
        self,
        hidden_states,  # 输入的隐藏状态
        attention_mask=None,  # 注意力遮罩，默认为 None
        output_attentions=False,  # 是否输出注意力，默认为 False
        output_hidden_states=False,  # 是否输出隐藏状态，默认为 False
        return_dict=True,  # 是否返回字典，默认为 True
    ):
        # 如果输出所有隐藏状态，则初始化一个空元组，否则设为 None
        all_hidden_states = () if output_hidden_states else None
        # 如果输出所有自注意力矩阵，则初始化一个空元组，否则设为 None
        all_self_attentions = () if output_attentions else None

        if attention_mask is not None:
            # 确保填充的标记不被关注
            expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
            hidden_states[~expand_attention_mask] = 0

            # 扩展 attention_mask
            attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
            attention_mask = attention_mask.expand(
                attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
            )

        # 位置嵌入
        position_embeddings = self.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.dropout(hidden_states)

        # 检查是否启用了 DeepSpeed Zero3
        deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()

        # 遍历每个层
        for layer in self.layers:
            # 如果输出所有隐藏状态，则将当前隐藏状态添加到 all_hidden_states 中
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 添加层丢弃（参见 https://arxiv.org/abs/1909.11556）
            dropout_probability = torch.rand([])

            # 如果正在训练并且丢弃的概率小于配置中的 layerdrop，则跳过该层
            skip_the_layer = True if self.training and (dropout_probability < self.config.layerdrop) else False
            if not skip_the_layer or deepspeed_zero3_is_enabled:
                # 如果启用了梯度检查点并且正在训练，则使用梯度检查点函数
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

            # 如果跳过了该层，则将 layer_outputs 设为 None
            if skip_the_layer:
                layer_outputs = (None, None)

            # 如果输出所有自注意力矩阵，则将当前层的自注意力矩阵添加到 all_self_attentions 中
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 对隐藏状态进行 LayerNormalization
        hidden_states = self.layer_norm(hidden_states)

        # 如果输出所有隐藏状态，则将当前隐藏状态添加到 all_hidden_states 中
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不返回字典，则返回隐藏状态、所有隐藏状态和所有自注意力矩阵的元组
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        # 返回字典形式的结果
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
class UniSpeechSatGumbelVectorQuantizer(nn.Module):
    """
    使用 Gumbel softmax 进行向量量化。
    更多信息请参见 [CATEGORICAL REPARAMETERIZATION WITH GUMBEL-SOFTMAX](https://arxiv.org/pdf/1611.01144.pdf)。
    """

    def __init__(self, config):
        # 继承 nn.Module 类，初始化函数
        super().__init__()
        # 设置编码向量组的数量
        self.num_groups = config.num_codevector_groups
        # 设置每组编码向量的数量
        self.num_vars = config.num_codevectors_per_group

        # 若 config.codevector_dim 不能被 config.num_codevector_groups 整除，则引发 ValueError 异常
        if config.codevector_dim % self.num_groups != 0:
            raise ValueError(
                f"`config.codevector_dim {config.codevector_dim} must be divisible by `config.num_codevector_groups`"
                f" {self.num_groups} for concatenation"
            )

        # 创建编码向量的存储（码字）
        self.codevectors = nn.Parameter(
            torch.FloatTensor(1, self.num_groups * self.num_vars, config.codevector_dim // self.num_groups)
        )
        # 权重投影层，用于将隐藏层映射到编码向量的维度
        self.weight_proj = nn.Linear(config.hidden_size, self.num_groups * self.num_vars)

        # 可以为训练中的温度进行衰减
        self.temperature = 2

    @staticmethod
    def _compute_perplexity(probs, mask=None):
        # 计算每个概率的复杂度
        marginal_probs = probs.mean(dim=0)
        # 根据概率计算复杂度
        perplexity = torch.exp(-torch.sum(marginal_probs * torch.log(marginal_probs + 1e-7), dim=-1)).sum()
        return perplexity
    # 定义前向传播方法，接受隐藏状态作为输入
    def forward(self, hidden_states):
        # 获取隐藏状态张量的批大小、序列长度和隐藏单元数
        batch_size, sequence_length, hidden_size = hidden_states.shape

        # 将隐藏状态投影到码向量维度
        hidden_states = self.weight_proj(hidden_states)
        # 重塑张量形状以便后续处理
        hidden_states = hidden_states.view(batch_size * sequence_length * self.num_groups, -1)

        if self.training:
            # 在可区分的方式下通过 Gumbel-Softmax 方法采样码向量概率
            codevector_probs = nn.functional.gumbel_softmax(
                hidden_states.float(), tau=self.temperature, hard=True
            ).type_as(hidden_states)

            # 计算困惑度
            codevector_soft_dist = torch.softmax(
                hidden_states.view(batch_size * sequence_length, self.num_groups, -1).float(), dim=-1
            )
            perplexity = self._compute_perplexity(codevector_soft_dist)
        else:
            # 在不可区分的方式下取码向量概率的最大值
            # 计算硬码向量分布（one hot）
            codevector_idx = hidden_states.argmax(dim=-1)
            codevector_probs = hidden_states.new_zeros(*hidden_states.shape).scatter_(
                -1, codevector_idx.view(-1, 1), 1.0
            )
            codevector_probs = codevector_probs.view(batch_size * sequence_length, self.num_groups, -1)

            perplexity = self._compute_perplexity(codevector_probs)

        # 重塑概率张量的形状
        codevector_probs = codevector_probs.view(batch_size * sequence_length, -1)
        # 使用概率检索码向量
        codevectors_per_group = codevector_probs.unsqueeze(-1) * self.codevectors
        codevectors = codevectors_per_group.view(batch_size * sequence_length, self.num_groups, self.num_vars, -1)
        codevectors = codevectors.sum(-2).view(batch_size, sequence_length, -1)

        # 返回码向量和困惑度
        return codevectors, perplexity
class UniSpeechSatPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 设置 config 类
    config_class = UniSpeechSatConfig
    # 设置基础模型前缀
    base_model_prefix = "unispeech_sat"
    # 设置主要输入名称
    main_input_name = "input_values"
    # 支持梯度检查点
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        """Initialize the weights"""
        # 对于 Gumbel softmax 需要特殊的初始化
        if isinstance(module, UniSpeechSatGumbelVectorQuantizer):
            # 对权重进行正态分布初始化
            module.weight_proj.weight.data.normal_(mean=0.0, std=1)
            # 将偏差初始化为零
            module.weight_proj.bias.data.zero_()
            # 对代码向量进行均匀分布初始化
            nn.init.uniform_(module.codevectors)
        elif isinstance(module, UniSpeechSatPositionalConvEmbedding):
            # 使用正态分布初始化卷积权重
            nn.init.normal_(
                module.conv.weight,
                mean=0,
                std=2 * math.sqrt(1 / (module.conv.kernel_size[0] * module.conv.in_channels)),
            )
            # 将卷积层的偏差初始化为零
            nn.init.constant_(module.conv.bias, 0)
        elif isinstance(module, UniSpeechSatFeatureProjection):
            # 计算 k 以用于均匀分布初始化投影的权重和偏置
            k = math.sqrt(1 / module.projection.in_features)
            # 对投影权重进行均匀分布初始化
            nn.init.uniform_(module.projection.weight, a=-k, b=k)
            # 对投影偏置进行均匀分布初始化
            nn.init.uniform_(module.projection.bias, a=-k, b=k)
        elif isinstance(module, nn.Linear):
            # 对线性层的权重进行正态分布初始化
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

            # 如果存在偏差，则将偏差初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            # 将归一化层的偏差初始化为零
            module.bias.data.zero_()
            # 将归一化层的权重初始化为1
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv1d):
            # 使用 kaiming 正态初始化卷积层的权重
            nn.init.kaiming_normal_(module.weight)

            # 如果存在偏差，则通过计算 k 来初始化偏差
            if module.bias is not None:
                k = math.sqrt(module.groups / (module.in_channels * module.kernel_size[0]))
                nn.init.uniform_(module.bias, a=-k, b=k)

    def _get_feat_extract_output_lengths(self, input_lengths: Union[torch.LongTensor, int]):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D 卷积层输出长度公式来自 https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1

        # 遍历卷积核大小和步长，计算输出长度
        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        return input_lengths
    # 获取非填充长度，即每个序列的有效长度
    non_padded_lengths = attention_mask.cumsum(dim=-1)[:, -1]
    # 使用非填充长度获取特征提取器输出的长度，转换为长整型
    output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths).to(torch.long)
    # 获取批处理大小
    batch_size = attention_mask.shape[0]

    # 创建一个全零的注意力遮罩张量，形状为(batch_size, feature_vector_length)
    attention_mask = torch.zeros(
        (batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
    )
    # 确保在输出长度索引之前的所有值都被注意到
    attention_mask[(torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1)] = 1
    # 翻转注意力遮罩张量并进行累积和翻转，以确保在输出长度索引之前的所有值都被注意到
    attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()
    # 返回注意力遮罩张量
    return attention_mask
UNISPEECH_SAT_START_DOCSTRING = r"""
    # UniSpeechSat 是由 Alexei Baevski、Henry Zhou、Abdelrahman Mohamed 和 Michael Auli 在论文《wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations》中提出的模型。
    UniSpeechSat was proposed in [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477) by Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael Auli.

    # 这个模型继承自 PreTrainedModel。查看超类文档以获取库实现的所有模型通用方法（如下载或保存等）。
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the library implements for all its model (such as downloading or saving etc.).

    # 这个模型是 PyTorch 的 torch.nn.Module 的子类。将其用作常规的 PyTorch 模块，并参考 PyTorch 文档以获取与一般用法和行为相关的所有事项。
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and behavior.

    # 参数:
    Parameters:
        # config: UniSpeechSatConfig 类的实例，包含模型的所有参数。使用配置文件初始化不会加载与模型关联的权重，只会加载配置。查看 ~PreTrainedModel.from_pretrained 方法以加载模型权重。
        config ([`UniSpeechSatConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


UNISPEECH_SAT_INPUTS_DOCSTRING = r"""
    # 输入参数说明:
        input_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)`): 
            # 输入的原始语音波形的浮点值。可以通过将`.flac`或`.wav`音频文件加载到`List[float]`类型的数组或`numpy.ndarray`中获取值，例如可以通过soundfile库进行。 使用`AutoProcessor`对数组进行填充和转换为`torch.FloatTensor`类型的张量。详细信息请参见`Wav2Vec2Processor.__call__`。
        attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*): 
            # 用于避免在填充标记索引上执行卷积和注意力机制的mask。掩码值在`[0, 1]`范围内:
                - 1表示**未屏蔽**的标记，
                - 0表示**已屏蔽**的标记。
            # 当相应处理器具有`config.return_attention_mask == True`时，应仅传递`attention_mask`。对于所有processor的`config.return_attention_mask == False`的模型（例如[microsoft/unispeech-sat-base-100h-libri-ft](https://huggingface.co/microsoft/unispeech-sat-base-100h-libri-ft)），在进行批量推断时应**不**传递`attention_mask`以避免性能下降。 对于这些模型，应简单地使用0进行填充的`input_values`，并在不传递`attention_mask`的情况下传递。请注意，这些模型的结果也取决于`input_values`是否进行了填充。
        output_attentions (`bool`, *optional*): 
            # 是否返回所有注意层的注意力张量。有关更多详细信息，请参见返回的张量下的`attentions`。
        output_hidden_states (`bool`, *optional*): 
            # 是否返回所有层的隐藏状态。有关更多详细信息，请参见返回的张量下的`hidden_states`。
        return_dict (`bool`, *optional*): 
            # 是否返回[`~utils.ModelOutput`]而不是普通元组。
"""
Import necessary libraries and modules
- nn: neural network module
- torch: PyTorch library
- UniSpeechSatPreTrainedModel: Pre-trained model for UniSpeechSat
- UniSpeechSatConfig: Configuration class for UniSpeechSat
- UniSpeechSatFeatureEncoder: Feature encoder for UniSpeechSat
- UniSpeechSatFeatureProjection: Feature projection for UniSpeechSat
- UniSpeechSatEncoderStableLayerNorm: Encoder with stable layer normalization for UniSpeechSat
- UniSpeechSatEncoder: Encoder for UniSpeechSat
"""
@add_start_docstrings(
    "The bare UniSpeechSat Model transformer outputting raw hidden-states without any specific head on top.",
    UNISPEECH_SAT_START_DOCSTRING,
)
class UniSpeechSatModel(UniSpeechSatPreTrainedModel):
    def __init__(self, config: UniSpeechSatConfig):
        # Initialize the UniSpeechSatModel class inheriting from UniSpeechSatPreTrainedModel
        super().__init__(config)
        self.config = config
        # Initialize feature extractor using UniSpeechSatFeatureEncoder
        self.feature_extractor = UniSpeechSatFeatureEncoder(config)
        # Initialize feature projection using UniSpeechSatFeatureProjection
        self.feature_projection = UniSpeechSatFeatureProjection(config)

        # Initialize a learnable parameter for masked spectrogram embedding
        self.masked_spec_embed = nn.Parameter(torch.FloatTensor(config.hidden_size).uniform_())

        if config.do_stable_layer_norm:
            # If configuration specifies stable layer normalization, use UniSpeechSatEncoderStableLayerNorm
            self.encoder = UniSpeechSatEncoderStableLayerNorm(config)
        else:
            # Otherwise, use regular UniSpeechSatEncoder
            self.encoder = UniSpeechSatEncoder(config)

        # Initialize weights and apply final processing
        self.post_init()

    # Method to mask hidden states
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
        # 检查配置中是否关闭了SpecAugment的masking
        if not getattr(self.config, "apply_spec_augment", True):
            return hidden_states

        # generate indices & apply SpecAugment along time axis
        # 获取hidden_states的大小信息
        batch_size, sequence_length, hidden_size = hidden_states.size()

        if mask_time_indices is not None:
            # apply SpecAugment along time axis with given mask_time_indices
            # 使用给定的mask_time_indices在时间轴上应用SpecAugment
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)
        elif self.config.mask_time_prob > 0 and self.training:
            # 计算需要mask的indices并在时间轴上应用SpecAugment
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
            # 计算需要mask的indices并在特征轴上应用SpecAugment
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

    @add_start_docstrings_to_model_forward(UNISPEECH_SAT_INPUTS_DOCSTRING)
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
        # 如果未指定输出注意力权重，则使用配置中的默认值
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果未指定输出隐藏状态，则使用配置中的默认值
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果未指定返回字典，则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 提取特征
        extract_features = self.feature_extractor(input_values)
        # 转置特征
        extract_features = extract_features.transpose(1, 2)

        if attention_mask is not None:
            # 计算与特征向量对应的减少的注意力掩码
            attention_mask = self._get_feature_vector_attention_mask(extract_features.shape[1], attention_mask)

        # 特征投影
        hidden_states, extract_features = self.feature_projection(extract_features)
        # 遮蔽隐藏状态
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

        hidden_states = encoder_outputs[0]

        if not return_dict:
            # 返回隐藏状态、提取特征和编码器输出
            return (hidden_states, extract_features) + encoder_outputs[1:]

        # 返回 Wav2Vec2BaseModelOutput 对象
        return Wav2Vec2BaseModelOutput(
            last_hidden_state=hidden_states,
            extract_features=extract_features,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
# 使用装饰器添加模型的文档字符串，描述了该模型是带有量化器和 VQ 头的 UniSpeechSat 模型
@add_start_docstrings("""UniSpeechSat Model with a quantizer and `VQ` head on top.""", UNISPEECH_SAT_START_DOCSTRING)
class UniSpeechSatForPreTraining(UniSpeechSatPreTrainedModel):
    # 初始化函数，接受 UniSpeechSatConfig 类型的参数
    def __init__(self, config: UniSpeechSatConfig):
        # 调用父类的初始化函数
        super().__init__(config)
        # 创建 UniSpeechSatModel 模型
        self.unispeech_sat = UniSpeechSatModel(config)
        # 创建一个 dropout 层，用于特征的丢弃
        self.dropout_features = nn.Dropout(config.feat_quantizer_dropout)

        # 创建一个量化器对象
        self.quantizer = UniSpeechSatGumbelVectorQuantizer(config)
        # 创建一个线性层，用于将编码向量维度转换为投影编码向量维度
        self.project_q = nn.Linear(config.codevector_dim, config.proj_codevector_dim)
        # 创建一个线性层，用于将隐藏层维度转换为投影编码向量维度
        self.project_hid = nn.Linear(config.hidden_size, config.proj_codevector_dim)

        # 创建一个 dropout 层，用于最终输出的丢弃
        self.dropout = nn.Dropout(config.final_dropout)

        # 创建一个线性层，用于将隐藏层维度转换为编码向量维度
        self.speaker_proj = nn.Linear(config.hidden_size, config.codevector_dim)
        # 创建一个可学习的参数，用于存储标签嵌入向量的拼接
        self.label_embeddings_concat = nn.Parameter(torch.FloatTensor(config.num_clusters, config.codevector_dim))
        self.label_embeddings_concat.data.zero_()

        # 创建一个 LayerNorm 层，用于提取特征时的归一化
        self.layer_norm_for_extract = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 如果配置中指定了稳定的 LayerNorm，则设置为不可训练
        if self.config.do_stable_layer_norm:
            self.layer_norm_for_extract.requires_grad = False

        # 初始化权重并应用最终处理
        self.post_init()

    # 设置 Gumbel softmax 温度值的函数，仅在训练时需要
    def set_gumbel_temperature(self, temperature: int):
        self.quantizer.temperature = temperature

    # 冻结特征提取器的函数，禁用特征编码器的梯度计算，使其在训练期间不会更新参数
    def freeze_feature_extractor(self):
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. "
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        self.freeze_feature_encoder()

    # 冻结特征编码器的函数，禁用特征编码器的梯度计算，使其在训练期间不会更新参数
    def freeze_feature_encoder(self):
        self.wav2vec2.feature_extractor._freeze_parameters()

    # 静态方法，用于计算对比损失的 logits
    @staticmethod
    def compute_contrastive_logits(
        target_features: torch.FloatTensor,
        negative_features: torch.FloatTensor,
        predicted_features: torch.FloatTensor,
        temperature: int = 1,
        ):
        """
        Compute logits for contrastive loss based using cosine similarity as the distance measure between
        `[positive_feature, negative_features]` and `[predicted_features]`. Additionally, temperature can be applied.
        """
        # 将目标特征和负样本特征拼接在一起
        target_features = torch.cat([target_features, negative_features], dim=0)

        # 计算余弦相似度作为距离度量，用于计算对比损失的logits
        logits = torch.cosine_similarity(predicted_features.float(), target_features.float(), dim=-1)
        logits = logits.type_as(target_features)

        # 应用温度参数
        logits = logits / temperature
        return logits

    # 将UNISPEECH_SAT_INPUTS_DOCSTRING添加到模型前向传播函数的文档字符串中
    @add_start_docstrings_to_model_forward(UNISPEECH_SAT_INPUTS_DOCSTRING)
    # 替换返回文档字符串的类型为UniSpeechSatForPreTrainingOutput，并使用_CONFIG_FOR_DOC配置类
    @replace_return_docstrings(output_type=UniSpeechSatForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ) -> Union[Tuple, UniSpeechSatForPreTrainingOutput]:
        r"""
        Returns:

        Example:

        ```python
        >>> import torch
        >>> from transformers import AutoFeatureExtractor, UniSpeechSatForPreTraining
        >>> from transformers.models.unispeech_sat.modeling_unispeech_sat import _compute_mask_indices

        >>> feature_extractor = AutoFeatureExtractor.from_pretrained("microsoft/unispeech-sat-base")
        >>> model = UniSpeechSatForPreTraining.from_pretrained("microsoft/unispeech-sat-base")
        >>> # TODO: Add full pretraining example
        ```py"""

        # 设置返回字典，如果未指定则使用配置中的设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 UniSpeechSat 模型进行预训练
        outputs = self.unispeech_sat(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        transformer_features = outputs[0]

        # 对所有提取的特征进行量化，并投影到最终的 VQ 维度
        extract_features = self.dropout_features(outputs[1])

        # TODO(PVP) - 添加预训练逻辑并添加到测试中
        logits = extract_features
        loss = quantized_features = codevector_perplexity = None

        # 层归一化（当 `config.do_stable_layer_norm == False` 时无效）
        #        extract_features = self.layer_norm_for_extract(extract_features)
        #        quantized_features, codevector_perplexity = self.quantizer(extract_features)
        #
        # 投影量化特征两次
        #        quantized_features = self.project_q(quantized_features)
        #        quantized_features = self.project_hid(quantized_features)
        #
        #        loss = None
        #        logits = quantized_features
        if not return_dict:
            if loss is not None:
                return (loss, logits, transformer_features, quantized_features, codevector_perplexity) + outputs[2:]
            return (logits, transformer_features, quantized_features, codevector_perplexity) + outputs[2:]

        # 返回 UniSpeechSatForPreTrainingOutput 对象
        return UniSpeechSatForPreTrainingOutput(
            loss=loss,
            logits=logits,
            projected_states=transformer_features,
            projected_quantized_states=quantized_features,
            codevector_perplexity=codevector_perplexity,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 添加起始文档字符串，描述UniSpeechSat模型具有顶部的`语言建模`头用于CTC（Connectionist Temporal Classification）。
# 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForCTC复制，将Wav2Vec2->UniSpeechSat，wav2vec2->unispeech_sat，WAV_2_VEC_2->UNISPEECH_SAT
class UniSpeechSatForCTC(UniSpeechSatPreTrainedModel):
    def __init__(self, config, target_lang: Optional[str] = None):
        # 调用父类构造函数
        super().__init__(config)

        # 初始化UniSpeechSatModel和dropout层
        self.unispeech_sat = UniSpeechSatModel(config)
        self.dropout = nn.Dropout(config.final_dropout)

        # 设置目标语言，默认为'eng'
        self.target_lang = target_lang

        # 检查配置中是否定义了词汇表大小，如果没有则引发错误
        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `UniSpeechSatForCTC.from_pretrained(..., vocab_size=vocab_size)`. "
                "or define `vocab_size` of your model's configuration."
            )
        
        # 根据配置设置输出隐藏层大小
        output_hidden_size = (
            config.output_hidden_size if hasattr(config, "add_adapter") and config.add_adapter else config.hidden_size
        )
        # 初始化线性层lm_head
        self.lm_head = nn.Linear(output_hidden_size, config.vocab_size)

        # 初始化权重并应用最终处理
        self.post_init()

    def tie_weights(self):
        """
        重写`~PreTrainedModel.tie_weights`方法，以便在传递`target_lang=...`给`from_pretrained(...)`时可以正确加载适配器权重。

        此方法**不**应由用户调用，并且可能在将来更改。
        """

        # 注意，`tie_weights`通常用于绑定输入和输出嵌入权重。该方法被重新用于正确加载UniSpeechSat的适配器层，以便我们不必为`PreTrainedModel`引入新的API。
        # 虽然有点巧妙，UniSpeechSat永远不必绑定输入和输出嵌入，因此在这里重新用这个函数是可以的。
        target_lang = self.target_lang

        # 如果target_lang不为None且配置中未定义adapter_attn_dim，则引发错误
        if target_lang is not None and getattr(self.config, "adapter_attn_dim", None) is None:
            raise ValueError(f"Cannot pass `target_lang`: {target_lang} if `config.adapter_attn_dim` is not defined.")
        # 如果target_lang为None且配置中定义了adapter_attn_dim，则记录日志
        elif target_lang is None and getattr(self.config, "adapter_attn_dim", None) is not None:
            logger.info("By default `target_lang` is set to 'eng'.")
        # 如果target_lang不为None，则加载适配器
        elif target_lang is not None:
            self.load_adapter(target_lang, force_load=True)
    # 冻结特征提取器，禁用特征编码器的梯度计算，使其参数在训练期间不会更新
    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        # 发出警告，提示方法`freeze_feature_extractor`已弃用，并将在 Transformers v5 中移除
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. "
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        # 调用等效的`freeze_feature_encoder`方法
        self.freeze_feature_encoder()

    # 冻结特征编码器，禁用特征编码器的梯度计算，使其参数在训练期间不会更新
    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        # 冻结特征编码器的参数
        self.unispeech_sat.feature_extractor._freeze_parameters()

    # 冻结基础模型，禁用基础模型的梯度计算，使其参数在训练期间不会更新，只有分类头会更新
    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        # 遍历所有参数，将其梯度计算设置为False
        for param in self.unispeech_sat.parameters():
            param.requires_grad = False

    # 前向传播函数，接受输入值、注意力掩码、输出注意力、输出隐藏状态、返回字典、标签等参数
    @add_start_docstrings_to_model_forward(UNISPEECH_SAT_INPUTS_DOCSTRING)
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
        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional`):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """

        # 设置返回字典，如果未指定则使用配置中的返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 使用 unispeech_sat 模型进行前向传播
        outputs = self.unispeech_sat(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取隐藏状态并应用 dropout
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
# 使用装饰器添加模型文档字符串，描述了 UniSpeechSat 模型在顶部具有一个序列分类头的用途，用于类似 SUPERB Keyword Spotting 的任务
@add_start_docstrings(
    """
    UniSpeechSat Model with a sequence classification head on top (a linear layer over the pooled output) for tasks
    like SUPERB Keyword Spotting.
    """,
    UNISPEECH_SAT_START_DOCSTRING,
)
# 定义 UniSpeechSatForSequenceClassification 类，继承自 UniSpeechSatPreTrainedModel
class UniSpeechSatForSequenceClassification(UniSpeechSatPreTrainedModel):
    # 初始化函数，接受配置参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__(config)

        # 如果配置中存在 add_adapter 属性且为 True，则抛出数值错误
        if hasattr(config, "add_adapter") and config.add_adapter:
            raise ValueError(
                "Sequence classification does not support the use of UniSpeechSat adapters (config.add_adapter=True)"
            )
        # 创建 UniSpeechSatModel 对象
        self.unispeech_sat = UniSpeechSatModel(config)
        # 计算层数，包括 transformer 层和输入嵌入层
        num_layers = config.num_hidden_layers + 1  # transformer layers + input embeddings
        # 如果配置中使用加权层求和，则初始化层权重参数
        if config.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        # 创建线性层，用于投影到分类器的输入维度
        self.projector = nn.Linear(config.hidden_size, config.classifier_proj_size)
        # 创建线性分类器层
        self.classifier = nn.Linear(config.classifier_proj_size, config.num_labels)

        # 初始化权重并应用最终处理
        self.post_init()

    # 冻结特征提取器的梯度计算
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

    # 冻结特征编码器的梯度计算
    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.unispeech_sat.feature_extractor._freeze_parameters()

    # 冻结基础模型的梯度计算
    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        for param in self.unispeech_sat.parameters():
            param.requires_grad = False

    # 使用装饰器添加模型前向传播的文档字符串，描述了 UniSpeechSat 模型的输入
    @add_start_docstrings_to_model_forward(UNISPEECH_SAT_INPUTS_DOCSTRING)
    # 使用装饰器添加代码示例文档字符串，描述了模型的检查点、输出类型、配置类和模态
    @add_code_sample_docstrings(
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="audio",
    )
    # 从transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForSequenceClassification.forward复制代码，并将Wav2Vec2->UniSpeechSat，wav2vec2->unispeech_sat
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

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states

        # 使用UniSpeechSat模型进行前向传播
        outputs = self.unispeech_sat(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = torch.stack(hidden_states, dim=1)
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            hidden_states = outputs[0]

        # 将隐藏状态投影到新的维度
        hidden_states = self.projector(hidden_states)
        if attention_mask is None:
            pooled_output = hidden_states.mean(dim=1)
        else:
            # 获取特征向量的注意力掩码
            padding_mask = self._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)
            hidden_states[~padding_mask] = 0.0
            pooled_output = hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)

        # 通过分类器获取logits
        logits = self.classifier(pooled_output)

        loss = None
        if labels is not None:
            # 计算损失
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 添加模型文档字符串，描述 UniSpeech-SAT 模型及其用途
@add_start_docstrings(
    """
    UniSpeech-SAT Model with a frame classification head on top for tasks like Speaker Diarization.
    """,
    UNISPEECH_SAT_START_DOCSTRING,
)
# 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForAudioFrameClassification 复制代码，并修改相关名称
class UniSpeechSatForAudioFrameClassification(UniSpeechSatPreTrainedModel):
    # 初始化函数，接受配置参数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__(config)

        # 如果配置中包含 add_adapter 属性且为 True，则抛出数值错误
        if hasattr(config, "add_adapter") and config.add_adapter:
            raise ValueError(
                "Audio frame classification does not support the use of UniSpeechSat adapters (config.add_adapter=True)"
            )
        # 创建 UniSpeechSatModel 对象
        self.unispeech_sat = UniSpeechSatModel(config)
        # 计算层数，包括 transformer 层和输入嵌入层
        num_layers = config.num_hidden_layers + 1
        # 如果配置中使用加权层求和，则初始化层权重参数
        if config.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        # 创建分类器线性层
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        # 记录标签数量
        self.num_labels = config.num_labels

        # 初始化模型权重
        self.init_weights()

    # 冻结特征提取器的梯度计算
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

    # 冻结特征编码器的梯度计算
    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.unispeech_sat.feature_extractor._freeze_parameters()

    # 冻结基础模型的梯度计算
    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        # 遍历基础模型的参数，设置 requires_grad 为 False
        for param in self.unispeech_sat.parameters():
            param.requires_grad = False

    # 前向传播函数，接受输入值、注意力掩码、标签等参数
    @add_start_docstrings_to_model_forward(UNISPEECH_SAT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_FRAME_CLASS_CHECKPOINT,
        output_type=TokenClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="audio",
        expected_output=_FRAME_EXPECTED_OUTPUT,
    )
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        # 确定是否返回字典
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 根据配置决定是否输出隐藏状态
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states

        # 调用unispeech_sat模型
        outputs = self.unispeech_sat(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 如果使用加权层求和
        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = torch.stack(hidden_states, dim=1)
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            hidden_states = outputs[0]

        # 使用分类器得到logits
        logits = self.classifier(hidden_states)

        loss = None
        # 如果有标签
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), torch.argmax(labels.view(-1, self.num_labels), axis=1))

        # 如果不返回字典
        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return output

        # 返回TokenClassifierOutput对象
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 从 transformers.models.wav2vec2.modeling_wav2vec2.AMSoftmaxLoss 复制而来的类
class AMSoftmaxLoss(nn.Module):
    def __init__(self, input_dim, num_labels, scale=30.0, margin=0.4):
        super(AMSoftmaxLoss, self).__init__()
        self.scale = scale
        self.margin = margin
        self.num_labels = num_labels
        self.weight = nn.Parameter(torch.randn(input_dim, num_labels), requires_grad=True)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, hidden_states, labels):
        labels = labels.flatten()
        weight = nn.functional.normalize(self.weight, dim=0)
        hidden_states = nn.functional.normalize(hidden_states, dim=1)
        cos_theta = torch.mm(hidden_states, weight)
        psi = cos_theta - self.margin

        onehot = nn.functional.one_hot(labels, self.num_labels)
        logits = self.scale * torch.where(onehot.bool(), psi, cos_theta)
        loss = self.loss(logits, labels)

        return loss


# 从 transformers.models.wav2vec2.modeling_wav2vec2.TDNNLayer 复制而来的类
class TDNNLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.in_conv_dim = config.tdnn_dim[layer_id - 1] if layer_id > 0 else config.tdnn_dim[layer_id]
        self.out_conv_dim = config.tdnn_dim[layer_id]
        self.kernel_size = config.tdnn_kernel[layer_id]
        self.dilation = config.tdnn_dilation[layer_id]

        self.kernel = nn.Linear(self.in_conv_dim * self.kernel_size, self.out_conv_dim)
        self.activation = nn.ReLU()

    def forward(self, hidden_states):
        hidden_states = hidden_states.unsqueeze(1)
        hidden_states = nn.functional.unfold(
            hidden_states,
            (self.kernel_size, self.in_conv_dim),
            stride=(1, self.in_conv_dim),
            dilation=(self.dilation, 1),
        )
        hidden_states = hidden_states.transpose(1, 2)
        hidden_states = self.kernel(hidden_states)

        hidden_states = self.activation(hidden_states)
        return hidden_states


@add_start_docstrings(
    """
    UniSpeech-SAT Model with an XVector feature extraction head on top for tasks like Speaker Verification.
    """,
    UNISPEECH_SAT_START_DOCSTRING,
)
# 从 transformers.models.wav2vec2.modeling_wav2vec2.Wav2Vec2ForXVector 复制而来的类，将其中的 Wav2Vec2 替换为 UniSpeechSat，wav2vec2 替换为 unispeech_sat，WAV_2_VEC_2 替换为 UNISPEECH_SAT
class UniSpeechSatForXVector(UniSpeechSatPreTrainedModel):
    # 初始化函数，接受配置参数，调用父类的初始化函数
    def __init__(self, config):
        # 调用父类的初始化函数
        super().__init__(config)

        # 创建 UniSpeechSatModel 模型对象
        self.unispeech_sat = UniSpeechSatModel(config)
        # 计算层数，包括 transformer 层和输入嵌入层
        num_layers = config.num_hidden_layers + 1  
        
        # 如果配置中使用加权层求和
        if config.use_weighted_layer_sum:
            # 创建可训练参数，初始化为均匀分布
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        
        # 创建线性层，用于投影
        self.projector = nn.Linear(config.hidden_size, config.tdnn_dim[0])

        # 创建 TDNNLayer 列表
        tdnn_layers = [TDNNLayer(config, i) for i in range(len(config.tdnn_dim))]
        # 将 TDNNLayer 列表转换为模块列表
        self.tdnn = nn.ModuleList(tdnn_layers)

        # 创建线性层，用于特征提取
        self.feature_extractor = nn.Linear(config.tdnn_dim[-1] * 2, config.xvector_output_dim)
        # 创建线性层，用于分类
        self.classifier = nn.Linear(config.xvector_output_dim, config.xvector_output_dim)

        # 创建 AMSoftmaxLoss 损失函数
        self.objective = AMSoftmaxLoss(config.xvector_output_dim, config.num_labels)

        # 初始化模型参数
        self.init_weights()

    # 冻结特征提取器，不更新其参数
    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        # 发出警告信息
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. "
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        # 调用 freeze_feature_encoder 方法
        self.freeze_feature_encoder()

    # 冻结特征编码器，不更新其参数
    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        # 冻结特征编码器的参数
        self.unispeech_sat.feature_extractor._freeze_parameters()

    # 冻结基础模型，不更新其参数
    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        # 遍历基础模型的参数，设置 requires_grad 为 False
        for param in self.unispeech_sat.parameters():
            param.requires_grad = False

    # 获取 TDNN 层的输出长度
    def _get_tdnn_output_lengths(self, input_lengths: Union[torch.LongTensor, int]):
        """
        Computes the output length of the TDNN layers
        """

        # 计算 1D 卷积层的输出长度
        def _conv_out_length(input_length, kernel_size, stride):
            # 1D 卷积层输出长度的计算公式
            return (input_length - kernel_size) // stride + 1

        # 遍历 TDNN 层的卷积核大小，计算输出长度
        for kernel_size in self.config.tdnn_kernel:
            input_lengths = _conv_out_length(input_lengths, kernel_size, 1)

        return input_lengths

    # 添加模型前向传播的文档字符串和示例代码
    @add_start_docstrings_to_model_forward(UNISPEECH_SAT_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_XVECTOR_CHECKPOINT,
        output_type=XVectorOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="audio",
        expected_output=_XVECTOR_EXPECTED_OUTPUT,
    )
    # 定义一个前向传播函数，接受输入值、注意力掩码、是否输出注意力权重、是否输出隐藏状态、是否返回字典、标签等参数
    def forward(
        self,
        input_values: Optional[torch.Tensor],  # 输入值，类型为torch.Tensor，可选参数
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码，类型为torch.Tensor，可选参数，默认为None
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，类型为bool，可选参数，默认为None
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，类型为bool，可选参数，默认为None
        return_dict: Optional[bool] = None,  # 是否返回字典，类型为bool，可选参数，默认为None
        labels: Optional[torch.Tensor] = None,  # 标签，类型为torch.Tensor，可选参数，默认为None
    ) -> Union[Tuple, XVectorOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        # 设置返回字典，如果未指定则使用配置中的返回字典设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 如果使用加权层求和，则输出隐藏状态
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states

        # 调用 unispeech_sat 模型
        outputs = self.unispeech_sat(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 如果使用加权层求和
        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = torch.stack(hidden_states, dim=1)
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            hidden_states = outputs[0]

        # 投影隐藏状态
        hidden_states = self.projector(hidden_states)

        # 遍历 TDNN 层
        for tdnn_layer in self.tdnn:
            hidden_states = tdnn_layer(hidden_states)

        # 统计池化
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

        # 提取特征
        output_embeddings = self.feature_extractor(statistic_pooling)
        logits = self.classifier(output_embeddings)

        loss = None
        # 如果存在标签，则计算损失
        if labels is not None:
            loss = self.objective(logits, labels)

        # 如果不返回字典
        if not return_dict:
            output = (logits, output_embeddings) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        # 返回 XVectorOutput 对象
        return XVectorOutput(
            loss=loss,
            logits=logits,
            embeddings=output_embeddings,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```