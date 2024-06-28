# `.\models\wav2vec2\modeling_wav2vec2.py`

```py
# coding=utf-8
# Copyright 2021 The Fairseq Authors and the HuggingFace Inc. team. All rights reserved.
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
""" PyTorch Wav2Vec2 model."""

import math  # 导入数学库
import warnings  # 导入警告模块
from dataclasses import dataclass  # 导入dataclass用于创建数据类
from typing import Optional, Tuple, Union  # 导入类型提示相关的类

import numpy as np  # 导入numpy库
import torch  # 导入PyTorch库
import torch.utils.checkpoint  # 导入PyTorch的checkpoint模块
from torch import nn  # 导入PyTorch的神经网络模块
from torch.nn import CrossEntropyLoss  # 导入交叉熵损失函数

from ...activations import ACT2FN  # 从本地路径导入ACT2FN激活函数
from ...integrations.deepspeed import is_deepspeed_zero3_enabled  # 导入是否启用深度速度的标志
from ...modeling_outputs import (  # 导入各种模型输出类
    BaseModelOutput,
    CausalLMOutput,
    MaskedLMOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
    Wav2Vec2BaseModelOutput,
    XVectorOutput,
)
from ...modeling_utils import PreTrainedModel  # 导入预训练模型基类
from ...pytorch_utils import is_torch_greater_or_equal_than_1_13  # 导入是否大于或等于PyTorch 1.13版本的函数
from ...utils import (  # 导入各种工具函数和类
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    cached_file,
    is_peft_available,
    is_safetensors_available,
    logging,
    replace_return_docstrings,
)
from .configuration_wav2vec2 import Wav2Vec2Config  # 从本地路径导入Wav2Vec2的配置类


WAV2VEC2_ADAPTER_PT_FILE = "adapter.{}.bin"  # 定义适配器的PyTorch文件名格式字符串
WAV2VEC2_ADAPTER_SAFE_FILE = "adapter.{}.safetensors"  # 定义适配器的安全张量文件名格式字符串

if is_safetensors_available():  # 如果安全张量可用
    from safetensors.torch import load_file as safe_load_file  # 从安全张量库导入加载文件的函数


logger = logging.get_logger(__name__)  # 获取当前模块的日志记录器


_HIDDEN_STATES_START_POSITION = 2  # 定义隐藏状态的起始位置为2

# General docstring
_CONFIG_FOR_DOC = "Wav2Vec2Config"  # 用于文档的配置说明字符串

# Base docstring
_CHECKPOINT_FOR_DOC = "facebook/wav2vec2-base-960h"  # 用于文档的基础模型检查点说明字符串
_EXPECTED_OUTPUT_SHAPE = [1, 292, 768]  # 预期输出的形状为[1, 292, 768]

# CTC docstring
_CTC_EXPECTED_OUTPUT = "'MISTER QUILTER IS THE APOSTLE OF THE MIDDLE CLASSES AND WE ARE GLAD TO WELCOME HIS GOSPEL'"  # CTC预期输出字符串
_CTC_EXPECTED_LOSS = 53.48  # CTC预期损失值

# Audio class docstring
_SEQ_CLASS_CHECKPOINT = "superb/wav2vec2-base-superb-ks"  # 音频类检查点字符串
_SEQ_CLASS_EXPECTED_OUTPUT = "'_unknown_'"  # 音频类预期输出字符串
_SEQ_CLASS_EXPECTED_LOSS = 6.54  # 音频类预期损失值

# Frame class docstring
_FRAME_CLASS_CHECKPOINT = "anton-l/wav2vec2-base-superb-sd"  # 帧类检查点字符串
_FRAME_EXPECTED_OUTPUT = [0, 0]  # 帧类预期输出列表

# Speaker Verification docstring
_XVECTOR_CHECKPOINT = "anton-l/wav2vec2-base-superb-sv"  # 说话人验证检查点字符串
_XVECTOR_EXPECTED_OUTPUT = 0.98  # 说话人验证预期输出值


WAV_2_VEC_2_PRETRAINED_MODEL_ARCHIVE_LIST = [  # 预训练模型存档列表
    "facebook/wav2vec2-base-960h",
    "facebook/wav2vec2-large-960h",
    "facebook/wav2vec2-large-960h-lv60",
    "facebook/wav2vec2-large-960h-lv60-self",
    # See all Wav2Vec2 models at https://huggingface.co/models?filter=wav2vec2
]


@dataclass
class Wav2Vec2ForPreTrainingOutput(ModelOutput):
    """
    输出类的数据类，用于Wav2Vec2的预训练模型。
    """
    # 定义了一个包含多种类型输出的数据结构 [`Wav2Vec2ForPreTraining`]
    
    Args:
        loss (*optional*, returned when `sample_negative_indices` are passed, `torch.FloatTensor` of shape `(1,)`):
            总损失，包括对比损失 (L_m) 和多样性损失 (L_d)，详见[官方论文](https://arxiv.org/pdf/2006.11477.pdf)。
        projected_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.proj_codevector_dim)`):
            模型隐藏状态，投影到 *config.proj_codevector_dim* 维度，可用于预测掩码投影量化状态。
        projected_quantized_states (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.proj_codevector_dim)`):
            量化的提取特征向量，投影到 *config.proj_codevector_dim* 维度，代表对比损失的正目标向量。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            模型每一层的隐藏状态，包括初始嵌入输出。
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            自注意力机制 softmax 后的注意力权重，用于计算自注意力头中的加权平均值。
        contrastive_loss (*optional*, returned when `sample_negative_indices` are passed, `torch.FloatTensor` of shape `(1,)`):
            对比损失 (L_m)，详见[官方论文](https://arxiv.org/pdf/2006.11477.pdf)。
        diversity_loss (*optional*, returned when `sample_negative_indices` are passed, `torch.FloatTensor` of shape `(1,)`):
            多样性损失 (L_d)，详见[官方论文](https://arxiv.org/pdf/2006.11477.pdf)。
def _compute_mask_indices(
    shape: Tuple[int, int],
    mask_prob: float,
    mask_length: int,
    attention_mask: Optional[torch.LongTensor] = None,
    min_masks: int = 0,
) -> np.ndarray:
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
    batch_size, sequence_length = shape  # 解构 shape 元组，获取 batch_size 和 sequence_length

    if mask_length < 1:
        raise ValueError("`mask_length` has to be bigger than 0.")  # 如果 mask_length 小于 1，则抛出数值错误异常

    if mask_length > sequence_length:
        raise ValueError(
            f"`mask_length` has to be smaller than `sequence_length`, but got `mask_length`: {mask_length}"
            f" and `sequence_length`: {sequence_length}`"
        )  # 如果 mask_length 大于 sequence_length，则抛出数值错误异常，显示详细信息

    # epsilon is used for probabilistic rounding
    epsilon = np.random.rand(1).item()  # 生成一个随机浮点数 epsilon，用于概率舍入

    def compute_num_masked_span(input_length):
        """Given input length, compute how many spans should be masked"""
        num_masked_span = int(mask_prob * input_length / mask_length + epsilon)  # 计算应该遮罩的 span 数量
        num_masked_span = max(num_masked_span, min_masks)  # 确保遮罩的 span 数量不低于 min_masks

        # make sure num masked span <= sequence_length
        if num_masked_span * mask_length > sequence_length:
            num_masked_span = sequence_length // mask_length  # 确保遮罩的 span 数量不超过 sequence_length

        # make sure num_masked span is also <= input_length - (mask_length - 1)
        if input_length - (mask_length - 1) < num_masked_span:
            num_masked_span = max(input_length - (mask_length - 1), 0)  # 确保遮罩的 span 数量不超过 input_length - (mask_length - 1)

        return num_masked_span

    # compute number of masked spans in batch
    input_lengths = (
        attention_mask.sum(-1).detach().tolist()
        if attention_mask is not None
        else [sequence_length for _ in range(batch_size)]
    )  # 计算批次中每个输入的长度，如果 attention_mask 存在，则计算其和并转换为列表，否则将 sequence_length 重复 batch_size 次

    # SpecAugment mask to fill
    spec_aug_mask = np.zeros((batch_size, sequence_length), dtype=bool)  # 创建一个全零布尔类型数组，用于存储遮罩信息
    spec_aug_mask_idxs = []  # 初始化用于存储遮罩索引的列表

    max_num_masked_span = compute_num_masked_span(sequence_length)  # 计算可遮罩的最大 span 数量

    if max_num_masked_span == 0:
        return spec_aug_mask  # 如果最大遮罩 span 数量为 0，则直接返回全零的遮罩数组
    for input_length in input_lengths:
        # 计算当前输入的遮罩跨度数量
        num_masked_span = compute_num_masked_span(input_length)

        # 随机选择用于遮罩的索引
        spec_aug_mask_idx = np.random.choice(
            np.arange(input_length - (mask_length - 1)), num_masked_span, replace=False
        )

        # 选择第一个采样的索引作为填充向量的虚拟索引
        # 以确保所有批次的维度相同，由于概率舍入的影响
        # 选择第一个样本只是对这些向量进行两次填充。
        if len(spec_aug_mask_idx) == 0:
            # 这种情况只会在 `input_length` 严格小于 `sequence_length` 的情况下发生，
            # 在这种情况下，最后一个标记必须是填充标记，我们可以将其用作虚拟遮罩标识符
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

    # 将偏移量添加到起始索引，以便索引现在表示一个跨度
    offsets = np.arange(mask_length)[None, None, :]
    offsets = np.broadcast_to(offsets, (batch_size, max_num_masked_span, mask_length)).reshape(
        batch_size, max_num_masked_span * mask_length
    )
    spec_aug_mask_idxs = spec_aug_mask_idxs + offsets

    # 确保索引不超过 `sequence_length - 1`
    if spec_aug_mask_idxs.max() > sequence_length - 1:
        spec_aug_mask_idxs[spec_aug_mask_idxs > sequence_length - 1] = sequence_length - 1

    # 将索引散布到遮罩中
    np.put_along_axis(spec_aug_mask, spec_aug_mask_idxs, 1, -1)

    return spec_aug_mask
def _sample_negative_indices(
    features_shape: Tuple, num_negatives: int, mask_time_indices: Optional[np.ndarray] = None
):
    """
    Sample `num_negatives` vectors from feature vectors.
    """
    batch_size, sequence_length = features_shape

    # generate indices of the positive vectors themselves, repeat them `num_negatives` times
    # 生成正向量本身的索引，并重复它们 `num_negatives` 次
    sequence_length_range = np.arange(sequence_length)

    # get `num_negatives` random vector indices from the same utterance
    # 从同一个语句中获取 `num_negatives` 个随机向量的索引
    sampled_negative_indices = np.zeros(shape=(batch_size, sequence_length, num_negatives), dtype=np.int32)

    # Convert mask_time_indices to boolean if provided, otherwise initialize as all True
    # 如果提供了 mask_time_indices，则将其转换为布尔类型，否则初始化为全True
    mask_time_indices = (
        mask_time_indices.astype(bool) if mask_time_indices is not None else np.ones(features_shape, dtype=bool)
    )

    for batch_idx in range(batch_size):
        # Calculate the maximum valid index after masking
        # 计算经过掩码后的最大有效索引
        high = mask_time_indices[batch_idx].sum() - 1
        mapped_masked_indices = sequence_length_range[mask_time_indices[batch_idx]]

        # Create indices for all features in the batch and `num_negatives`
        # 为批次中的所有特征和 `num_negatives` 创建索引
        feature_indices = np.broadcast_to(np.arange(high + 1)[:, None], (high + 1, num_negatives))
        
        # Sample random indices avoiding the same positive vector, ensuring uniform distribution
        # 避免采样相同的正向量，同时保持均匀分布
        sampled_indices = np.random.randint(0, high, size=(high + 1, num_negatives))
        sampled_indices[sampled_indices >= feature_indices] += 1

        # Remap sampled indices to actual indices based on the mask
        # 根据掩码重新映射采样的索引到实际索引
        sampled_negative_indices[batch_idx][mask_time_indices[batch_idx]] = mapped_masked_indices[sampled_indices]

        # Correct indices for batch size
        # 校正批次大小的索引
        sampled_negative_indices[batch_idx] += batch_idx * sequence_length

    return sampled_negative_indices



class Wav2Vec2NoLayerNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        # Define a 1D convolutional layer
        # 定义一个一维卷积层
        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )

        # Activation function based on configuration
        # 基于配置选择激活函数
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        # Apply convolutional layer
        # 应用卷积层
        hidden_states = self.conv(hidden_states)

        # Apply activation function
        # 应用激活函数
        hidden_states = self.activation(hidden_states)
        return hidden_states


class Wav2Vec2LayerNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        # Define a 1D convolutional layer
        # 定义一个一维卷积层
        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )

        # Apply layer normalization to the output of the convolutional layer
        # 对卷积层的输出应用层归一化
        self.layer_norm = nn.LayerNorm(self.out_conv_dim, elementwise_affine=True)

        # Activation function based on configuration
        # 基于配置选择激活函数
        self.activation = ACT2FN[config.feat_extract_activation]
    # 定义一个前向传播方法，用于处理输入的隐藏状态数据
    def forward(self, hidden_states):
        # 使用卷积层处理隐藏状态数据，返回处理后的结果
        hidden_states = self.conv(hidden_states)
    
        # 将数据在倒数第二和倒数第一维度进行转置操作
        hidden_states = hidden_states.transpose(-2, -1)
    
        # 对转置后的数据进行层归一化处理
        hidden_states = self.layer_norm(hidden_states)
    
        # 再次将数据在倒数第二和倒数第一维度进行转置操作，恢复原始形状
        hidden_states = hidden_states.transpose(-2, -1)
    
        # 使用激活函数处理归一化后的数据，返回处理后的结果
        hidden_states = self.activation(hidden_states)
    
        # 返回最终处理后的隐藏状态数据
        return hidden_states
class Wav2Vec2GroupNormConvLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        # 从配置中获取输入和输出的卷积维度
        self.in_conv_dim = config.conv_dim[layer_id - 1] if layer_id > 0 else 1
        self.out_conv_dim = config.conv_dim[layer_id]

        # 创建一个一维卷积层
        self.conv = nn.Conv1d(
            self.in_conv_dim,
            self.out_conv_dim,
            kernel_size=config.conv_kernel[layer_id],
            stride=config.conv_stride[layer_id],
            bias=config.conv_bias,
        )
        # 获取激活函数并进行初始化
        self.activation = ACT2FN[config.feat_extract_activation]

        # 创建一维组归一化层，设置组数为输出卷积维度
        self.layer_norm = nn.GroupNorm(num_groups=self.out_conv_dim, num_channels=self.out_conv_dim, affine=True)

    def forward(self, hidden_states):
        # 执行卷积操作
        hidden_states = self.conv(hidden_states)
        # 应用组归一化
        hidden_states = self.layer_norm(hidden_states)
        # 应用激活函数
        hidden_states = self.activation(hidden_states)
        return hidden_states


class Wav2Vec2PositionalConvEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个带有位置信息的一维卷积层
        self.conv = nn.Conv1d(
            config.hidden_size,
            config.hidden_size,
            kernel_size=config.num_conv_pos_embeddings,
            padding=config.num_conv_pos_embeddings // 2,
            groups=config.num_conv_pos_embedding_groups,
        )

        # 根据条件选择是否对权重进行归一化处理
        weight_norm = nn.utils.weight_norm
        if hasattr(nn.utils.parametrizations, "weight_norm"):
            weight_norm = nn.utils.parametrizations.weight_norm

        # 根据是否启用DeepSpeed Zero3进行特殊处理
        if is_deepspeed_zero3_enabled():
            import deepspeed

            # 在Zero3环境下使用GatheredParameters进行权重聚合
            with deepspeed.zero.GatheredParameters(self.conv.weight, modifier_rank=0):
                self.conv = weight_norm(self.conv, name="weight", dim=2)
            # 注册外部参数以便DeepSpeed管理
            deepspeed.zero.register_external_parameter(self, self.conv.weight_v)
            deepspeed.zero.register_external_parameter(self, self.conv.weight_g)
        else:
            # 否则直接对权重进行归一化处理
            self.conv = weight_norm(self.conv, name="weight", dim=2)

        # 创建用于填充的同步填充层
        self.padding = Wav2Vec2SamePadLayer(config.num_conv_pos_embeddings)
        # 获取激活函数并进行初始化
        self.activation = ACT2FN[config.feat_extract_activation]

    def forward(self, hidden_states):
        # 转置隐藏状态以便卷积操作
        hidden_states = hidden_states.transpose(1, 2)

        # 执行卷积操作
        hidden_states = self.conv(hidden_states)
        # 执行填充操作
        hidden_states = self.padding(hidden_states)
        # 应用激活函数
        hidden_states = self.activation(hidden_states)

        # 再次转置隐藏状态以还原其原始形状
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states


class Wav2Vec2SamePadLayer(nn.Module):
    def __init__(self, num_conv_pos_embeddings):
        super().__init__()
        # 根据卷积位置嵌入的数量确定需要移除的填充数目
        self.num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0

    def forward(self, hidden_states):
        # 如果需要移除填充，则执行填充移除操作
        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, :, : -self.num_pad_remove]
        return hidden_states


class Wav2Vec2FeatureEncoder(nn.Module):
    """Construct the features from raw audio waveform"""
    def __init__(self, config):
        super().__init__()

        # 根据配置选择特征提取层的归一化方式
        if config.feat_extract_norm == "group":
            # 如果是"group"归一化，创建卷积层列表，第一层使用组归一化
            conv_layers = [Wav2Vec2GroupNormConvLayer(config, layer_id=0)] + [
                Wav2Vec2NoLayerNormConvLayer(config, layer_id=i + 1) for i in range(config.num_feat_extract_layers - 1)
            ]
        elif config.feat_extract_norm == "layer":
            # 如果是"layer"归一化，创建卷积层列表，每层使用层归一化
            conv_layers = [
                Wav2Vec2LayerNormConvLayer(config, layer_id=i) for i in range(config.num_feat_extract_layers)
            ]
        else:
            # 如果既不是"group"也不是"layer"，抛出值错误异常
            raise ValueError(
                f"`config.feat_extract_norm` is {config.feat_extract_norm}, but has to be one of ['group', 'layer']"
            )
        
        # 将卷积层列表转换为神经网络模块列表
        self.conv_layers = nn.ModuleList(conv_layers)
        self.gradient_checkpointing = False
        self._requires_grad = True

    def _freeze_parameters(self):
        # 冻结所有参数，使其不再需要梯度计算
        for param in self.parameters():
            param.requires_grad = False
        self._requires_grad = False

    def forward(self, input_values):
        # 将输入的特征值扩展为(batch_size, 1, sequence_length)
        hidden_states = input_values[:, None]

        # 如果需要梯度且在训练阶段，确保hidden_states需要梯度
        if self._requires_grad and self.training:
            hidden_states.requires_grad = True

        # 遍历所有卷积层并对hidden_states进行卷积操作
        for conv_layer in self.conv_layers:
            if self._requires_grad and self.gradient_checkpointing and self.training:
                # 如果需要梯度且启用了梯度检查点和在训练阶段，使用梯度检查点函数进行前向传播
                hidden_states = self._gradient_checkpointing_func(
                    conv_layer.__call__,
                    hidden_states,
                )
            else:
                # 否则直接应用卷积层进行前向传播
                hidden_states = conv_layer(hidden_states)

        return hidden_states
class Wav2Vec2FeatureExtractor(Wav2Vec2FeatureEncoder):
    # 继承自Wav2Vec2FeatureEncoder类的特征提取器类
    def __init__(self, config):
        # 调用父类构造函数初始化对象
        super().__init__(config)
        # 发出警告，提示该类已经弃用，并建议使用更换的类名
        warnings.warn(
            f"The class `{self.__class__.__name__}` has been depreciated "
            "and will be removed in Transformers v5. "
            f"Use `{self.__class__.__bases__[0].__name__}` instead.",
            FutureWarning,
        )


class Wav2Vec2FeatureProjection(nn.Module):
    # 特征投影模块类，继承自PyTorch的nn.Module
    def __init__(self, config):
        # 初始化函数
        super().__init__()
        # 使用LayerNorm层对最后一个卷积维度进行归一化
        self.layer_norm = nn.LayerNorm(config.conv_dim[-1], eps=config.layer_norm_eps)
        # 线性变换层，将卷积维度映射到隐藏层维度
        self.projection = nn.Linear(config.conv_dim[-1], config.hidden_size)
        # Dropout层，用于随机置零输入张量的部分元素，以防止过拟合
        self.dropout = nn.Dropout(config.feat_proj_dropout)

    def forward(self, hidden_states):
        # 需要保留非投影的隐藏状态用于量化
        norm_hidden_states = self.layer_norm(hidden_states)
        # 对归一化后的隐藏状态进行线性投影
        hidden_states = self.projection(norm_hidden_states)
        # 对投影后的隐藏状态进行随机置零处理
        hidden_states = self.dropout(hidden_states)
        return hidden_states, norm_hidden_states


# 从transformers.models.bart.modeling_bart.BartAttention复制，将Bart替换为Wav2Vec2
class Wav2Vec2Attention(nn.Module):
    """基于 'Attention Is All You Need' 论文的多头注意力机制"""

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
        super().__init__()
        # 初始化函数
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        self.config = config

        if (self.head_dim * num_heads) != self.embed_dim:
            # 检查embed_dim必须能被num_heads整除，否则引发错误
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        # 缩放因子，用于调整注意力计算的缩放
        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        # 线性映射层，用于查询、键、值的映射
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        # 将张量重新形状以便多头注意力机制处理
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        key_value_states: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ````
        # 初始化方法，接受配置参数 config
        def __init__(self, config):
            # 调用父类的初始化方法
            super().__init__()
            # 创建一个 Dropout 层，丢弃率由 config.activation_dropout 指定
            self.intermediate_dropout = nn.Dropout(config.activation_dropout)
    
            # 创建一个全连接层，将输入维度映射到中间层维度，输入大小为 config.hidden_size，输出大小为 config.intermediate_size
            self.intermediate_dense = nn.Linear(config.hidden_size, config.intermediate_size)
            # 根据配置的激活函数名，获取对应的激活函数，默认为一个函数对象
            if isinstance(config.hidden_act, str):
                self.intermediate_act_fn = ACT2FN[config.hidden_act]
            else:
                self.intermediate_act_fn = config.hidden_act
    
            # 创建一个全连接层，将中间层维度映射回输入维度，输入大小为 config.intermediate_size，输出大小为 config.hidden_size
            self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)
            # 创建一个 Dropout 层，丢弃率由 config.hidden_dropout 指定
            self.output_dropout = nn.Dropout(config.hidden_dropout)
    
        # 前向传播方法，定义数据流向
        def forward(self, hidden_states):
            # 将输入数据通过中间全连接层
            hidden_states = self.intermediate_dense(hidden_states)
            # 应用激活函数
            hidden_states = self.intermediate_act_fn(hidden_states)
            # 应用中间 Dropout 层
            hidden_states = self.intermediate_dropout(hidden_states)
    
            # 将数据通过输出全连接层
            hidden_states = self.output_dense(hidden_states)
            # 应用输出 Dropout 层
            hidden_states = self.output_dropout(hidden_states)
            # 返回经过处理的数据
            return hidden_states
class Wav2Vec2EncoderLayer(nn.Module):
    # Wav2Vec2 编码器层定义
    def __init__(self, config):
        super().__init__()
        # 初始化注意力层，使用配置中的隐藏大小、注意力头数和注意力丢弃率
        self.attention = Wav2Vec2Attention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
        )
        # Dropout 层，使用隐藏状态的丢弃率
        self.dropout = nn.Dropout(config.hidden_dropout)
        # LayerNorm 层，使用配置中的隐藏大小和层归一化的 epsilon
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Wav2Vec2 前向传播层，使用给定的配置
        self.feed_forward = Wav2Vec2FeedForward(config)
        # 最终的 LayerNorm 层，使用配置中的隐藏大小和层归一化的 epsilon
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    # 前向传播函数，接受隐藏状态、注意力掩码（可选）、是否输出注意力权重（可选）
    def forward(self, hidden_states, attention_mask=None, output_attentions=False):
        # 残差连接的起始点
        attn_residual = hidden_states
        # 注意力层的前向传播，获取新的隐藏状态、注意力权重（可选）、并返回的额外内容（_）
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
        )
        # 应用 Dropout 到新的隐藏状态
        hidden_states = self.dropout(hidden_states)
        # 残差连接：原始隐藏状态加上新的隐藏状态
        hidden_states = attn_residual + hidden_states

        # 应用 LayerNorm 到更新后的隐藏状态
        hidden_states = self.layer_norm(hidden_states)
        # 加上前向传播层的输出到隐藏状态
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        # 最终的 LayerNorm
        hidden_states = self.final_layer_norm(hidden_states)

        # 输出结果为一个元组，包含最终的隐藏状态
        outputs = (hidden_states,)

        # 如果输出注意力权重，则添加到输出元组中
        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class Wav2Vec2EncoderLayerStableLayerNorm(nn.Module):
    # Wav2Vec2 编码器层定义，使用稳定的 LayerNorm
    def __init__(self, config):
        super().__init__()
        # 初始化注意力层，使用配置中的隐藏大小、注意力头数和注意力丢弃率
        self.attention = Wav2Vec2Attention(
            embed_dim=config.hidden_size,
            num_heads=config.num_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=False,
        )
        # Dropout 层，使用隐藏状态的丢弃率
        self.dropout = nn.Dropout(config.hidden_dropout)
        # LayerNorm 层，使用配置中的隐藏大小和层归一化的 epsilon
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # Wav2Vec2 前向传播层，使用给定的配置
        self.feed_forward = Wav2Vec2FeedForward(config)
        # 最终的 LayerNorm 层，使用配置中的隐藏大小和层归一化的 epsilon
        self.final_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # 如果配置中有 adapter_attn_dim 属性，则初始化适配器层，否则设置为 None
        if getattr(config, "adapter_attn_dim", None) is not None:
            self.adapter_layer = Wav2Vec2AttnAdapterLayer(config)
        else:
            self.adapter_layer = None

    # 前向传播函数，接受隐藏状态、注意力掩码（可选）、是否输出注意力权重（可选）
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
    ):
        # 省略部分前向传播逻辑...
    ):
        attn_residual = hidden_states
        # 保留注意力机制之前的隐藏状态，用于残差连接
        hidden_states = self.layer_norm(hidden_states)
        # 应用层归一化到隐藏状态
        hidden_states, attn_weights, _ = self.attention(
            hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
        )
        # 执行自注意力机制，得到更新后的隐藏状态和注意力权重
        hidden_states = self.dropout(hidden_states)
        # 对更新后的隐藏状态应用 dropout
        hidden_states = attn_residual + hidden_states
        # 将残差连接的结果加回到更新后的隐藏状态中
        hidden_states = hidden_states + self.feed_forward(self.final_layer_norm(hidden_states))
        # 对经过残差连接后的隐藏状态应用前馈网络和层归一化

        if self.adapter_layer is not None:
            hidden_states = hidden_states + self.adapter_layer(hidden_states)
            # 如果存在适配器层，则将适配器层应用到隐藏状态上

        outputs = (hidden_states,)
        # 将最终的隐藏状态放入输出元组中

        if output_attentions:
            outputs += (attn_weights,)
            # 如果需要输出注意力权重，则将注意力权重也放入输出元组中

        return outputs
        # 返回包含最终隐藏状态和可能的注意力权重的输出元组
# 定义一个名为 Wav2Vec2Encoder 的神经网络模块类
class Wav2Vec2Encoder(nn.Module):
    # 初始化函数，接受一个配置参数 config
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 将传入的配置参数保存到对象的 config 属性中
        self.config = config
        # 创建一个 Wav2Vec2PositionalConvEmbedding 类的实例，并保存到 pos_conv_embed 属性中
        self.pos_conv_embed = Wav2Vec2PositionalConvEmbedding(config)
        # 创建一个具有 LayerNorm 的神经网络层，用于标准化隐藏状态的尺寸
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个 Dropout 层，用于在训练过程中随机丢弃隐藏状态的一部分，以减少过拟合风险
        self.dropout = nn.Dropout(config.hidden_dropout)
        # 创建一个包含多个 Wav2Vec2EncoderLayer 实例的列表，列表长度为配置中指定的隐藏层数量
        self.layers = nn.ModuleList([Wav2Vec2EncoderLayer(config) for _ in range(config.num_hidden_layers)])
        # 初始化梯度检查点为 False，用于后续可能的梯度检查点优化
        self.gradient_checkpointing = False

    # 正向传播方法，接收隐藏状态、注意力掩码以及其他参数，并返回计算结果
    def forward(
        self,
        hidden_states: torch.tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ):
    ):
        # 如果需要输出隐藏状态，则初始化一个空元组；否则设置为 None
        all_hidden_states = () if output_hidden_states else None
        # 如果需要输出注意力权重，则初始化一个空元组；否则设置为 None
        all_self_attentions = () if output_attentions else None

        if attention_mask is not None:
            # 确保填充的 token 输出为 0
            expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
            hidden_states[~expand_attention_mask] = 0

            # 扩展 attention_mask 的维度
            attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
            attention_mask = attention_mask.expand(
                attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
            )

        # 计算位置编码嵌入
        position_embeddings = self.pos_conv_embed(hidden_states)
        # 将位置编码嵌入加到隐藏状态中
        hidden_states = hidden_states + position_embeddings
        # Layer normalization
        hidden_states = self.layer_norm(hidden_states)
        # Dropout
        hidden_states = self.dropout(hidden_states)

        # 检查是否启用了 DeepSpeed zero3
        deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()

        # 遍历每个层
        for layer in self.layers:
            # 如果需要输出隐藏状态，则记录当前层的隐藏状态
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # 添加 LayerDrop 功能（参见 https://arxiv.org/abs/1909.11556）
            dropout_probability = torch.rand([])

            # 根据 LayerDrop 的概率决定是否跳过当前层
            skip_the_layer = True if self.training and (dropout_probability < self.config.layerdrop) else False
            # 如果不跳过当前层或者在 DeepSpeed zero3 模式下
            if not skip_the_layer or deepspeed_zero3_is_enabled:
                # 使用梯度检查点函数优化内存使用
                if self.gradient_checkpointing and self.training:
                    layer_outputs = self._gradient_checkpointing_func(
                        layer.__call__,
                        hidden_states,
                        attention_mask,
                        output_attentions,
                    )
                else:
                    # 正常执行当前层的前向传播
                    layer_outputs = layer(
                        hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
                    )
                hidden_states = layer_outputs[0]

            # 如果跳过当前层，则设置层输出为空
            if skip_the_layer:
                layer_outputs = (None, None)

            # 如果需要输出注意力权重，则记录当前层的自注意力权重
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        # 如果需要输出隐藏状态，则记录最终的隐藏状态
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # 如果不返回字典形式的结果，则返回元组形式的输出
        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
        # 返回字典形式的输出
        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )
# 定义一个名为 Wav2Vec2EncoderStableLayerNorm 的类，继承自 nn.Module
class Wav2Vec2EncoderStableLayerNorm(nn.Module):
    # 初始化方法，接受一个 config 参数
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__()
        # 将传入的 config 参数保存到实例变量中
        self.config = config
        # 创建一个 Wav2Vec2PositionalConvEmbedding 类的实例，并保存到实例变量 pos_conv_embed 中
        self.pos_conv_embed = Wav2Vec2PositionalConvEmbedding(config)
        # 创建一个 nn.LayerNorm 层，对隐藏状态进行归一化，设置归一化的隐藏状态维度和 epsilon 值
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        # 创建一个 nn.Dropout 层，用于在训练过程中随机 dropout 一部分隐藏状态
        self.dropout = nn.Dropout(config.hidden_dropout)
        # 创建一个 nn.ModuleList，其中包含 config.num_hidden_layers 个 Wav2Vec2EncoderLayerStableLayerNorm 实例
        # 每个实例代表一个编码器层
        self.layers = nn.ModuleList(
            [Wav2Vec2EncoderLayerStableLayerNorm(config) for _ in range(config.num_hidden_layers)]
        )
        # 初始化梯度检查点标志为 False
        self.gradient_checkpointing = False

    # 前向传播方法，接收隐藏状态、注意力掩码等参数
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        ):
        ):
            # 初始化空元组用于存储所有隐藏状态
            all_hidden_states = () if output_hidden_states else None
            # 初始化空元组用于存储所有自注意力机制
            all_self_attentions = () if output_attentions else None

            if attention_mask is not None:
                # 确保不对填充标记进行注意力计算
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

            # 检查是否启用了deepspeed zero3
            deepspeed_zero3_is_enabled = is_deepspeed_zero3_enabled()

            # 遍历每个层
            for layer in self.layers:
                if output_hidden_states:
                    # 如果需要输出隐藏状态，则将当前隐藏状态添加到all_hidden_states中
                    all_hidden_states = all_hidden_states + (hidden_states,)

                # 添加LayerDrop机制
                dropout_probability = torch.rand([])
                skip_the_layer = True if self.training and (dropout_probability < self.config.layerdrop) else False

                if not skip_the_layer or deepspeed_zero3_is_enabled:
                    # 如果不跳过该层或者启用了deepspeed zero3
                    if self.gradient_checkpointing and self.training:
                        # 如果启用梯度检查点并且在训练模式下
                        layer_outputs = self._gradient_checkpointing_func(
                            layer.__call__,
                            hidden_states,
                            attention_mask,
                            output_attentions,
                        )
                    else:
                        # 否则直接调用层进行前向传播
                        layer_outputs = layer(
                            hidden_states, attention_mask=attention_mask, output_attentions=output_attentions
                        )
                    hidden_states = layer_outputs[0]

                if skip_the_layer:
                    # 如果跳过了该层，则将输出置为None
                    layer_outputs = (None, None)

                if output_attentions:
                    # 如果需要输出自注意力机制，则将当前层的自注意力加入到all_self_attentions中
                    all_self_attentions = all_self_attentions + (layer_outputs[1],)

            # 对最终的隐藏状态进行层归一化处理
            hidden_states = self.layer_norm(hidden_states)

            if output_hidden_states:
                # 如果需要输出隐藏状态，则将最终的隐藏状态加入到all_hidden_states中
                all_hidden_states = all_hidden_states + (hidden_states,)

            if not return_dict:
                # 如果不需要返回字典形式的输出，则以元组形式返回相应的结果
                return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions] if v is not None)
            # 如果需要返回字典形式的输出，则构造BaseModelOutput对象返回
            return BaseModelOutput(
                last_hidden_state=hidden_states,
                hidden_states=all_hidden_states,
                attentions=all_self_attentions,
            )
    # Wav2Vec2GumbelVectorQuantizer 类，用于向量量化，使用 Gumbel softmax 进行操作。
    # 详细信息参见《CATEGORICAL REPARAMETERIZATION WITH GUMBEL-SOFTMAX》。
    class Wav2Vec2GumbelVectorQuantizer(nn.Module):
        """
        Vector quantization using gumbel softmax. See `[CATEGORICAL REPARAMETERIZATION WITH
        GUMBEL-SOFTMAX](https://arxiv.org/pdf/1611.01144.pdf) for more information.
        """

        def __init__(self, config):
            super().__init__()
            self.num_groups = config.num_codevector_groups  # 设置编码向量组的数量
            self.num_vars = config.num_codevectors_per_group  # 每个组中的编码向量数量

            if config.codevector_dim % self.num_groups != 0:
                raise ValueError(
                    f"`config.codevector_dim {config.codevector_dim} must be divisible "
                    f"by `config.num_codevector_groups` {self.num_groups} for concatenation"
                )

            # 存储码本变量（码字）的空间
            self.codevectors = nn.Parameter(
                torch.FloatTensor(1, self.num_groups * self.num_vars, config.codevector_dim // self.num_groups)
            )
            # 权重投影层，用于生成码本中每个向量的投影
            self.weight_proj = nn.Linear(config.conv_dim[-1], self.num_groups * self.num_vars)

            # 可以在训练过程中衰减的温度参数
            self.temperature = 2

        @staticmethod
        def _compute_perplexity(probs, mask=None):
            # 计算困惑度的静态方法
            if mask is not None:
                mask_extended = mask.flatten()[:, None, None].expand(probs.shape)
                probs = torch.where(mask_extended, probs, torch.zeros_like(probs))
                marginal_probs = probs.sum(dim=0) / mask.sum()
            else:
                marginal_probs = probs.mean(dim=0)

            # 根据概率计算困惑度
            perplexity = torch.exp(-torch.sum(marginal_probs * torch.log(marginal_probs + 1e-7), dim=-1)).sum()
            return perplexity
    # 定义前向传播函数，接收隐藏状态和时间掩码索引作为参数
    def forward(self, hidden_states, mask_time_indices=None):
        # 获取隐藏状态的维度信息
        batch_size, sequence_length, hidden_size = hidden_states.shape

        # 将隐藏状态投影到编码向量维度
        hidden_states = self.weight_proj(hidden_states)
        # 重塑隐藏状态的形状，以便应用于多组编码向量
        hidden_states = hidden_states.view(batch_size * sequence_length * self.num_groups, -1)

        if self.training:
            # 如果处于训练阶段，通过Gumbel softmax采样获取编码向量的概率分布（可微分方式）
            codevector_probs = nn.functional.gumbel_softmax(
                hidden_states.float(), tau=self.temperature, hard=True
            ).type_as(hidden_states)

            # 计算 perplexity（困惑度）
            codevector_soft_dist = torch.softmax(
                hidden_states.view(batch_size * sequence_length, self.num_groups, -1).float(), dim=-1
            )
            perplexity = self._compute_perplexity(codevector_soft_dist, mask_time_indices)
        else:
            # 如果处于非训练阶段，通过 argmax 获取硬编码向量分布（非可微分方式）
            codevector_idx = hidden_states.argmax(dim=-1)
            codevector_probs = hidden_states.new_zeros(hidden_states.shape).scatter_(
                -1, codevector_idx.view(-1, 1), 1.0
            )
            codevector_probs = codevector_probs.view(batch_size * sequence_length, self.num_groups, -1)

            perplexity = self._compute_perplexity(codevector_probs, mask_time_indices)

        # 将编码向量概率分布重塑为(batch_size * sequence_length, -1)的形状
        codevector_probs = codevector_probs.view(batch_size * sequence_length, -1)
        # 使用概率分布获取编码向量
        codevectors_per_group = codevector_probs.unsqueeze(-1) * self.codevectors
        # 将编码向量按组求和，重塑为(batch_size, sequence_length, -1)的形状
        codevectors = codevectors_per_group.view(batch_size * sequence_length, self.num_groups, self.num_vars, -1)
        codevectors = codevectors.sum(-2).view(batch_size, sequence_length, -1)

        # 返回编码向量和 perplexity（困惑度）
        return codevectors, perplexity
class Wav2Vec2Adapter(nn.Module):
    def __init__(self, config):
        super().__init__()

        # feature dim might need to be down-projected
        # 如果输出的隐藏状态大小不等于隐藏层大小，则需要进行降维投影
        if config.output_hidden_size != config.hidden_size:
            self.proj = nn.Linear(config.hidden_size, config.output_hidden_size)
            self.proj_layer_norm = nn.LayerNorm(config.output_hidden_size)
        else:
            self.proj = self.proj_layer_norm = None

        # 创建一个包含多个适配器层的模块列表
        self.layers = nn.ModuleList(Wav2Vec2AdapterLayer(config) for _ in range(config.num_adapter_layers))
        # 设置层次丢弃概率
        self.layerdrop = config.layerdrop

    def forward(self, hidden_states):
        # 如果需要的话，对隐藏状态进行降维投影
        if self.proj is not None and self.proj_layer_norm is not None:
            hidden_states = self.proj(hidden_states)
            hidden_states = self.proj_layer_norm(hidden_states)

        # 转置隐藏状态张量的第一和第二维度
        hidden_states = hidden_states.transpose(1, 2)

        # 遍历所有的适配器层
        for layer in self.layers:
            # 随机生成一个层丢弃的概率
            layerdrop_prob = np.random.random()
            # 如果不是训练状态或者随机数大于层丢弃概率，则应用当前层
            if not self.training or (layerdrop_prob > self.layerdrop):
                hidden_states = layer(hidden_states)

        # 再次转置隐藏状态张量的第一和第二维度
        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states


class Wav2Vec2AdapterLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 创建一个卷积层，用于适配器的卷积操作
        self.conv = nn.Conv1d(
            config.output_hidden_size,
            2 * config.output_hidden_size,
            config.adapter_kernel_size,
            stride=config.adapter_stride,
            padding=1,
        )

    def forward(self, hidden_states):
        # 对输入的隐藏状态进行卷积操作
        hidden_states = self.conv(hidden_states)
        # 应用 gated linear unit (GLU) 激活函数
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

        # 对输入的隐藏状态进行 Layer Normalization
        self.norm = nn.LayerNorm(self.hidden_dim)
        # 第一个线性层，用于将隐藏状态投影到指定的注意力适配器维度
        self.linear_1 = nn.Linear(self.hidden_dim, self.input_dim)
        self.act_fn = nn.ReLU()
        # 第二个线性层，用于将注意力适配器维度投影回隐藏状态大小
        self.linear_2 = nn.Linear(self.input_dim, self.hidden_dim)

    def forward(self, hidden_states: torch.FloatTensor):
        # 应用 Layer Normalization
        hidden_states = self.norm(hidden_states)

        # 第一个线性层投影
        hidden_states = self.linear_1(hidden_states)
        hidden_states = self.act_fn(hidden_states)
        # 第二个线性层投影
        hidden_states = self.linear_2(hidden_states)

        return hidden_states


class Wav2Vec2PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    # 使用的配置类
    config_class = Wav2Vec2Config
    # 基础模型前缀名
    base_model_prefix = "wav2vec2"
    # 主要输入名称
    main_input_name = "input_values"
    # 是否支持梯度检查点
    supports_gradient_checkpointing = True
    def _init_weights(self, module):
        """Initialize the weights"""
        # 对于 Wav2Vec2ForPreTraining 模块的最后两个线性层，使用标准的线性层初始化方法
        if isinstance(module, Wav2Vec2ForPreTraining):
            # 重置隐藏层线性变换和量化层的参数
            module.project_hid.reset_parameters()
            module.project_q.reset_parameters()
            # 设置初始化标志为 True
            module.project_hid._is_hf_initialized = True
            module.project_q._is_hf_initialized = True
        # 对于 Wav2Vec2GumbelVectorQuantizer 模块，使用特殊的初始化方法
        elif isinstance(module, Wav2Vec2GumbelVectorQuantizer):
            # 使用正态分布初始化权重矩阵，均值为 0，标准差为 1
            module.weight_proj.weight.data.normal_(mean=0.0, std=1)
            # 将偏置初始化为零
            module.weight_proj.bias.data.zero_()
            # 使用均匀分布初始化 codevectors
            nn.init.uniform_(module.codevectors)
        # 对于 Wav2Vec2PositionalConvEmbedding 模块，使用正态分布初始化卷积层参数
        elif isinstance(module, Wav2Vec2PositionalConvEmbedding):
            # 初始化卷积核权重，均值为 0，标准差为 sqrt(2 / (卷积核尺寸 * 输入通道数))
            nn.init.normal_(
                module.conv.weight,
                mean=0,
                std=2 * math.sqrt(1 / (module.conv.kernel_size[0] * module.conv.in_channels)),
            )
            # 将卷积层的偏置初始化为零
            nn.init.constant_(module.conv.bias, 0)
        # 对于 Wav2Vec2FeatureProjection 模块，使用均匀分布初始化线性投影的权重和偏置
        elif isinstance(module, Wav2Vec2FeatureProjection):
            # 计算均匀分布的上下界
            k = math.sqrt(1 / module.projection.in_features)
            # 初始化权重和偏置
            nn.init.uniform_(module.projection.weight, a=-k, b=k)
            nn.init.uniform_(module.projection.bias, a=-k, b=k)
        # 对于普通的 nn.Linear 模块，使用正态分布初始化权重，均值为 0，标准差为配置文件中的 initializer_range
        elif isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)

            # 如果存在偏置项，则将偏置初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 对于 nn.LayerNorm 或 nn.GroupNorm 模块，将偏置初始化为零，将权重初始化为 1
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        # 对于 nn.Conv1d 模块，使用 Kaiming 正态分布初始化权重
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight)

            # 如果存在偏置项，则使用均匀分布初始化偏置
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
            # 从 https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html 中获取的
            # 1D 卷积层输出长度计算公式
            return torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1

        # 遍历卷积层的内核大小和步长，计算每一层的输出长度
        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        # 如果需要添加适配器层，则对每一层应用适配器层的输出长度计算
        if add_adapter:
            for _ in range(self.config.num_adapter_layers):
                input_lengths = _conv_out_length(input_lengths, 1, self.config.adapter_stride)

        # 返回最终的输出长度
        return input_lengths
    def _get_feature_vector_attention_mask(
        self, feature_vector_length: int, attention_mask: torch.LongTensor, add_adapter=None
    ):
        # 计算非填充部分的长度，即有效长度
        non_padded_lengths = attention_mask.cumsum(dim=-1)[:, -1]

        # 获取特征提取器的输出长度
        output_lengths = self._get_feat_extract_output_lengths(non_padded_lengths, add_adapter=add_adapter)
        output_lengths = output_lengths.to(torch.long)

        # 获取批次大小
        batch_size = attention_mask.shape[0]

        # 初始化注意力掩码为全零，形状为(batch_size, feature_vector_length)
        attention_mask = torch.zeros(
            (batch_size, feature_vector_length), dtype=attention_mask.dtype, device=attention_mask.device
        )

        # 设置使得输出长度之前的所有位置都被注意到
        attention_mask[(torch.arange(attention_mask.shape[0], device=attention_mask.device), output_lengths - 1)] = 1

        # 翻转注意力掩码，累积求和，并再次翻转，最终转换为布尔值类型
        attention_mask = attention_mask.flip([-1]).cumsum(-1).flip([-1]).bool()

        # 返回处理后的注意力掩码
        return attention_mask

    def _get_adapters(self):
        # 如果配置中的 adapter_attn_dim 为 None，则抛出异常
        if self.config.adapter_attn_dim is None:
            raise ValueError(f"{self.__class__} has no adapter layers. Make sure to define `config.adapter_attn_dim`.")

        # 初始化适配器权重字典
        adapter_weights = {}

        # 遍历模型的所有模块，找出适配器层并获取其参数
        for name, module in self.named_modules():
            if isinstance(module, Wav2Vec2AttnAdapterLayer):
                for param_name, param in module.named_parameters():
                    adapter_weights[".".join([name, param_name])] = param

        # 如果模型是 Wav2Vec2ForCTC 类型，则获取 lm_head 的参数作为适配器参数
        if isinstance(self, Wav2Vec2ForCTC):
            for name, param in self.lm_head.named_parameters():
                adapter_weights[".".join(["lm_head", name])] = param

        # 返回所有适配器的权重参数字典
        return adapter_weights

    def init_adapter_layers(self):
        """
        (重新-)初始化注意力适配器层和 LM 头部，用于仅适配器微调
        """
        # 初始化注意力适配器层
        for module in self.modules():
            if isinstance(module, Wav2Vec2AttnAdapterLayer):
                self._init_weights(module)

        # 初始化语言模型头部
        if isinstance(self, Wav2Vec2ForCTC):
            self._init_weights(self.lm_head)
# WAV_2_VEC_2_START_DOCSTRING 是一个包含文档字符串的原始字符串常量，描述了 Wav2Vec2 模型的起源和相关信息。
WAV_2_VEC_2_START_DOCSTRING = r"""
    Wav2Vec2 was proposed in [wav2vec 2.0: A Framework for Self-Supervised Learning of Speech
    Representations](https://arxiv.org/abs/2006.11477) by Alexei Baevski, Henry Zhou, Abdelrahman Mohamed, Michael
    Auli.

    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving etc.).

    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`Wav2Vec2Config`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

# WAV_2_VEC_2_INPUTS_DOCSTRING 是一个空的原始字符串常量，用于接下来描述输入格式和参数信息。
WAV_2_VEC_2_INPUTS_DOCSTRING = r"""
    # 接收输入参数并解析为输入的原始语音波形张量，类型为 `torch.FloatTensor`，形状为 `(batch_size, sequence_length)`
    Args:
        input_values (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Float values of input raw speech waveform. Values can be obtained by loading a `.flac` or `.wav` audio file
            into an array of type `List[float]` or a `numpy.ndarray`, *e.g.* via the soundfile library (`pip install
            soundfile`). To prepare the array into `input_values`, the [`AutoProcessor`] should be used for padding and
            conversion into a tensor of type `torch.FloatTensor`. See [`Wav2Vec2Processor.__call__`] for details.
    
    # 可选参数，注意力遮罩张量，形状为 `(batch_size, sequence_length)`，类型为 `torch.LongTensor`
    attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        Mask to avoid performing convolution and attention on padding token indices. Mask values selected in `[0,
        1]`:
        
        - 1 for tokens that are **not masked**,
        - 0 for tokens that are **masked**.
        
        [What are attention masks?](../glossary#attention-mask)
        
        <Tip warning={true}>
        
        `attention_mask` should only be passed if the corresponding processor has `config.return_attention_mask ==
        True`. For all models whose processor has `config.return_attention_mask == False`, such as
        [wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base-960h), `attention_mask` should **not** be
        passed to avoid degraded performance when doing batched inference. For such models `input_values` should
        simply be padded with 0 and passed without `attention_mask`. Be aware that these models also yield slightly
        different results depending on whether `input_values` is padded or not.
        
        </Tip>
    
    # 可选参数，控制是否返回所有注意力层的注意力张量
    output_attentions (`bool`, *optional*):
        Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
        tensors for more detail.
    
    # 可选参数，控制是否返回所有层的隐藏状态张量
    output_hidden_states (`bool`, *optional*):
        Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
        more detail.
    
    # 可选参数，控制是否返回一个包含多个输出的 [`~utils.ModelOutput`] 对象，而不是一个普通的元组
    return_dict (`bool`, *optional*):
        Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""
Wav2Vec2Model 类，继承自 Wav2Vec2PreTrainedModel 类，表示一个基本的 Wav2Vec2 模型，输出未经特定顶层头部处理的原始隐藏状态。

@param config: Wav2Vec2Config 对象，配置当前模型的参数

初始化函数，设置模型的各个组件和参数
"""
class Wav2Vec2Model(Wav2Vec2PreTrainedModel):
    def __init__(self, config: Wav2Vec2Config):
        super().__init__(config)
        self.config = config
        self.feature_extractor = Wav2Vec2FeatureEncoder(config)  # 初始化音频特征提取器
        self.feature_projection = Wav2Vec2FeatureProjection(config)  # 初始化音频特征投影器

        # 如果配置要求，初始化遮罩向量
        if config.mask_time_prob > 0.0 or config.mask_feature_prob > 0.0:
            self.masked_spec_embed = nn.Parameter(torch.FloatTensor(config.hidden_size).uniform_())

        # 根据配置选择稳定层归一化的编码器或普通编码器
        if config.do_stable_layer_norm:
            self.encoder = Wav2Vec2EncoderStableLayerNorm(config)
        else:
            self.encoder = Wav2Vec2Encoder(config)

        # 如果配置添加适配器，则初始化适配器
        self.adapter = Wav2Vec2Adapter(config) if config.add_adapter else None

        # 初始化权重并应用最终处理
        self.post_init()

    def freeze_feature_extractor(self):
        """
        调用此函数将禁用特征编码器的梯度计算，使其参数在训练过程中不会更新。
        """
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. "
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        """
        调用此函数将禁用特征编码器的梯度计算，使其参数在训练过程中不会更新。
        """
        self.feature_extractor._freeze_parameters()

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
        if not getattr(self.config, "apply_spec_augment", True):
            return hidden_states

        # generate indices & apply SpecAugment along time axis
        batch_size, sequence_length, hidden_size = hidden_states.size()

        if mask_time_indices is not None:
            # apply SpecAugment along time axis with given mask_time_indices
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)
        elif self.config.mask_time_prob > 0 and self.training:
            # compute mask indices for time axis based on configuration parameters
            mask_time_indices = _compute_mask_indices(
                (batch_size, sequence_length),
                mask_prob=self.config.mask_time_prob,
                mask_length=self.config.mask_time_length,
                attention_mask=attention_mask,
                min_masks=self.config.mask_time_min_masks,
            )
            # convert computed indices to a Torch tensor for device compatibility and dtype
            mask_time_indices = torch.tensor(mask_time_indices, device=hidden_states.device, dtype=torch.bool)
            # apply SpecAugment along time axis using computed mask indices
            hidden_states[mask_time_indices] = self.masked_spec_embed.to(hidden_states.dtype)

        if self.config.mask_feature_prob > 0 and self.training:
            # generate indices & apply SpecAugment along feature axis
            mask_feature_indices = _compute_mask_indices(
                (batch_size, hidden_size),
                mask_prob=self.config.mask_feature_prob,
                mask_length=self.config.mask_feature_length,
                min_masks=self.config.mask_feature_min_masks,
            )
            # convert computed indices to a Torch tensor for device compatibility and dtype
            mask_feature_indices = torch.tensor(mask_feature_indices, device=hidden_states.device, dtype=torch.bool)
            # expand feature axis mask indices to match hidden_states dimensions
            mask_feature_indices = mask_feature_indices[:, None].expand(-1, sequence_length, -1)
            # apply SpecAugment along feature axis using expanded mask indices
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
        # 如果输出注意力张量未指定，则使用配置中的默认设置
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # 如果输出隐藏状态未指定，则使用配置中的默认设置
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # 如果返回字典未指定，则使用配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 提取特征向量
        extract_features = self.feature_extractor(input_values)
        # 调整特征向量的维度顺序
        extract_features = extract_features.transpose(1, 2)

        if attention_mask is not None:
            # 计算与特征向量对应的降维注意力掩码
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )

        # 投影特征向量到隐藏状态空间
        hidden_states, extract_features = self.feature_projection(extract_features)
        # 对隐藏状态进行屏蔽处理，根据时间索引和注意力掩码
        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )

        # 编码器处理
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取编码器的隐藏状态
        hidden_states = encoder_outputs[0]

        # 如果有适配器，应用适配器到隐藏状态
        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)

        # 如果不返回字典，则返回一个元组，包含隐藏状态、提取的特征向量以及可能的其他输出
        if not return_dict:
            return (hidden_states, extract_features) + encoder_outputs[1:]

        # 返回带有详细输出的 Wav2Vec2BaseModelOutput 对象
        return Wav2Vec2BaseModelOutput(
            last_hidden_state=hidden_states,
            extract_features=extract_features,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )
# 使用装饰器为 Wav2Vec2ForPreTraining 类添加文档字符串，描述该类包含量化器和 `VQ` 头部。
@add_start_docstrings("""Wav2Vec2 Model with a quantizer and `VQ` head on top.""", WAV_2_VEC_2_START_DOCSTRING)
class Wav2Vec2ForPreTraining(Wav2Vec2PreTrainedModel):
    def __init__(self, config: Wav2Vec2Config):
        # 调用父类构造函数初始化对象
        super().__init__(config)
        # 初始化 Wav2Vec2 模型
        self.wav2vec2 = Wav2Vec2Model(config)
        # 根据配置初始化特征量化器的丢弃层
        self.dropout_features = nn.Dropout(config.feat_quantizer_dropout)

        # 初始化 Gumbel 向量量化器
        self.quantizer = Wav2Vec2GumbelVectorQuantizer(config)

        # 将隐藏层的输出映射到代码向量的维度
        self.project_hid = nn.Linear(config.hidden_size, config.proj_codevector_dim)
        # 将代码向量映射到投影向量的维度
        self.project_q = nn.Linear(config.codevector_dim, config.proj_codevector_dim)

        # 初始化权重并应用最终处理
        self.post_init()

    def set_gumbel_temperature(self, temperature: int):
        """
        设置 Gumbel softmax 温度为给定值。仅在训练时需要。
        """
        self.quantizer.temperature = temperature

    def freeze_feature_extractor(self):
        """
        调用此函数将禁用特征编码器的梯度计算，使其参数在训练过程中不会被更新。
        """
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. "
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        """
        调用此函数将禁用特征编码器的梯度计算，使其参数在训练过程中不会被更新。
        """
        self.wav2vec2.feature_extractor._freeze_parameters()

    @staticmethod
    def compute_contrastive_logits(
        target_features: torch.FloatTensor,
        negative_features: torch.FloatTensor,
        predicted_features: torch.FloatTensor,
        temperature: int = 0.1,
    ):
        """
        计算对比损失的对数估计，使用余弦相似性作为 `[positive_feature, negative_features]` 和 `[predicted_features]` 之间的距离度量。
        可以应用温度调节。
        """
        # 将目标特征和负特征连接起来
        target_features = torch.cat([target_features, negative_features], dim=0)

        # 计算余弦相似性
        logits = torch.cosine_similarity(predicted_features.float(), target_features.float(), dim=-1).type_as(
            target_features
        )

        # 应用温度调节
        logits = logits / temperature
        return logits

    # 使用装饰器为 model_forward 方法添加文档字符串，根据输入的 WAV_2_VEC_2_INPUTS_DOCSTRING 描述
    @add_start_docstrings_to_model_forward(WAV_2_VEC_2_INPUTS_DOCSTRING)
    # 替换返回值文档字符串为 Wav2Vec2ForPreTrainingOutput 类型，并使用 _CONFIG_FOR_DOC 作为配置类
    @replace_return_docstrings(output_type=Wav2Vec2ForPreTrainingOutput, config_class=_CONFIG_FOR_DOC)
    # 定义一个方法 `forward`，用于模型的前向传播
    (
        self,
        input_values: Optional[torch.Tensor],  # 输入的张量数据，可选
        attention_mask: Optional[torch.Tensor] = None,  # 注意力掩码张量，可选，默认为None
        mask_time_indices: Optional[torch.BoolTensor] = None,  # 时间索引掩码张量，可选，默认为None
        sampled_negative_indices: Optional[torch.BoolTensor] = None,  # 负采样索引掩码张量，可选，默认为None
        output_attentions: Optional[bool] = None,  # 是否输出注意力权重，可选，默认为None
        output_hidden_states: Optional[bool] = None,  # 是否输出隐藏状态，可选，默认为None
        return_dict: Optional[bool] = None,  # 是否以字典形式返回结果，可选，默认为None
# 使用装饰器为类添加文档字符串，说明该类是在 Wav2Vec2 模型基础上加上语言建模头部的模型
@add_start_docstrings("""Wav2Vec2 Model with a `language modeling` head on top.""", WAV_2_VEC_2_START_DOCSTRING)
class Wav2Vec2ForMaskedLM(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 发出警告，提醒用户类 `Wav2Vec2ForMaskedLM` 已被弃用，请使用 `Wav2Vec2ForCTC` 替代
        warnings.warn(
            "The class `Wav2Vec2ForMaskedLM` is deprecated. Please use `Wav2Vec2ForCTC` instead.", FutureWarning
        )

        # 初始化 Wav2Vec2 模型和相关组件
        self.wav2vec2 = Wav2Vec2Model(config)
        self.dropout = nn.Dropout(config.final_dropout)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)

        # 初始化权重并进行最终处理
        self.post_init()

    # 使用装饰器为前向传播方法添加文档字符串，引用了 WAV_2_VEC_2_INPUTS_DOCSTRING
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
        # 如果 return_dict 未指定，则使用配置中的默认值
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 调用 Wav2Vec2 模型的前向传播，获取输出
        outputs = self.wav2vec2(
            input_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 获取隐藏状态，并对其进行 dropout 处理
        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        # 将处理后的隐藏状态传递给语言建模头部进行预测
        logits = self.lm_head(hidden_states)

        # 如果不要求返回字典，则返回一个包含 logits 和其他输出的元组
        if not return_dict:
            output = (logits,) + outputs[2:]
            return output

        # 如果要求返回字典，则返回 MaskedLMOutput 类型的对象，包含 logits、隐藏状态和注意力分布
        return MaskedLMOutput(logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions)


# 使用装饰器为类添加文档字符串，说明该类是在 Wav2Vec2 模型基础上加上用于 CTC 的语言建模头部的模型
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
        # 调用父类的构造方法初始化模型
        super().__init__(config)

        # 创建一个 Wav2Vec2 模型对象
        self.wav2vec2 = Wav2Vec2Model(config)
        # 根据配置中的最终 dropout 比率创建一个 dropout 层
        self.dropout = nn.Dropout(config.final_dropout)

        # 设置目标语言属性
        self.target_lang = target_lang

        # 检查配置中是否定义了词汇表大小，如果未定义则抛出 ValueError 异常
        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `Wav2Vec2ForCTC.from_pretrained(..., vocab_size=vocab_size)`. "
                "or define `vocab_size` of your model's configuration."
            )
        
        # 根据配置设置输出的隐藏层大小，如果配置中定义了添加适配器且有适配器，则使用配置中的隐藏层大小，否则使用常规的隐藏层大小
        output_hidden_size = (
            config.output_hidden_size if hasattr(config, "add_adapter") and config.add_adapter else config.hidden_size
        )
        # 创建线性层，用于语言模型的输出
        self.lm_head = nn.Linear(output_hidden_size, config.vocab_size)

        # 初始化权重并进行最终的处理
        self.post_init()

    def tie_weights(self):
        """
        This method overwrites [`~PreTrainedModel.tie_weights`] so that adapter weights can be correctly loaded when
        passing `target_lang=...` to `from_pretrained(...)`.

        This method is **not** supposed to be called by the user and is prone to be changed in the future.
        """
        
        # 注意，`tie_weights` 方法通常用于绑定输入和输出的嵌入权重。这里重新定义此方法，以便在通过 `from_pretrained(...)` 传递 `target_lang=...` 时能正确加载适配器层权重。
        # 这种做法虽然有些巧妙，但是由于 Wav2Vec2 模型不需要绑定输入和输出的嵌入权重，因此可以在这里重新定义此函数。
        target_lang = self.target_lang

        # 如果指定了 `target_lang` 且配置中的 `adapter_attn_dim` 未定义，则抛出 ValueError 异常
        if target_lang is not None and getattr(self.config, "adapter_attn_dim", None) is None:
            raise ValueError(f"Cannot pass `target_lang`: {target_lang} if `config.adapter_attn_dim` is not defined.")
        # 如果未指定 `target_lang` 但配置中的 `adapter_attn_dim` 定义了，则记录日志提示默认将 `target_lang` 设置为 'eng'
        elif target_lang is None and getattr(self.config, "adapter_attn_dim", None) is not None:
            logger.info("By default `target_lang` is set to 'eng'.")
        # 如果指定了 `target_lang`，则加载相应的适配器
        elif target_lang is not None:
            self.load_adapter(target_lang, force_load=True)

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        # 发出警告，指出该方法已被弃用，并将在 Transformers v5 中移除，建议使用 `freeze_feature_encoder` 方法代替。
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. "
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        # 调用 `freeze_feature_encoder` 方法来冻结特征编码器
        self.freeze_feature_encoder()
    # 冻结特征编码器的参数，使其在训练过程中不会更新梯度
    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        self.wav2vec2.feature_extractor._freeze_parameters()

    # 冻结基础模型的参数，使其在训练过程中不会更新梯度，只有分类头会被更新
    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        for param in self.wav2vec2.parameters():
            param.requires_grad = False

    # 前向传播函数，接受多个参数并返回模型输出
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
        ) -> Union[Tuple, CausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """

        # Determine if a return_dict is specified; otherwise, use the model's default setting
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Pass input_values and optional arguments to wav2vec2 model for feature extraction
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Extract hidden states from the model outputs and apply dropout
        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        # Generate logits using the language model head
        logits = self.lm_head(hidden_states)

        # Initialize loss variable
        loss = None
        if labels is not None:
            # Check if any label index exceeds the vocabulary size
            if labels.max() >= self.config.vocab_size:
                raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

            # Retrieve input_lengths based on attention_mask
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)

            # Mask out labels set to -100 and calculate target_lengths and flattened_targets
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # Apply log_softmax to logits and transpose for CTC loss computation
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

            # Disable cuDNN optimization for CTC loss computation
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

        # If return_dict is False, format output accordingly
        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        # Return output as a CausalLMOutput object when return_dict is True
        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )
@add_start_docstrings(
    """
    Wav2Vec2 Model with a sequence classification head on top (a linear layer over the pooled output) for tasks like
    SUPERB Keyword Spotting.
    """,
    WAV_2_VEC_2_START_DOCSTRING,
)
class Wav2Vec2ForSequenceClassification(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        if hasattr(config, "add_adapter") and config.add_adapter:
            raise ValueError(
                "Sequence classification does not support the use of Wav2Vec2 adapters (config.add_adapter=True)"
            )
        
        # 初始化 Wav2Vec2 模型
        self.wav2vec2 = Wav2Vec2Model(config)
        
        # 计算层数，包括 Transformer 层和输入嵌入层
        num_layers = config.num_hidden_layers + 1  # transformer layers + input embeddings
        
        # 如果配置指定使用加权层求和，则初始化权重
        if config.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        
        # 线性投影层，将隐藏状态映射到分类器投影大小
        self.projector = nn.Linear(config.hidden_size, config.classifier_proj_size)
        
        # 分类器线性层，将投影后的特征映射到标签数量的输出
        self.classifier = nn.Linear(config.classifier_proj_size, config.num_labels)

        # 初始化权重并进行最终处理
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
        # 调用底层方法冻结特征编码器的参数梯度
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        # 冻结特征编码器，禁用其参数的梯度计算
        self.wav2vec2.feature_extractor._freeze_parameters()

    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        # 冻结基础模型，禁用其所有参数的梯度计算，只允许分类头更新
        for param in self.wav2vec2.parameters():
            param.requires_grad = False

    @add_start_docstrings_to_model_forward(WAV_2_VEC_2_INPUTS_DOCSTRING)
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
        """
        Runs the forward pass of the Wav2Vec2ForSequenceClassification model.
        """
        # 省略了 forward 方法中的具体实现，但是加了装饰器和示例代码文档字符串
        ) -> Union[Tuple, SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        # 初始化返回字典，若未提供则使用配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 根据配置决定是否使用加权层的隐藏状态输出
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states

        # 调用Wav2Vec2模型进行前向传播
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 若配置中设置使用加权层求和，则进行加权求和操作
        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = torch.stack(hidden_states, dim=1)
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            # 否则直接使用模型输出的第一个隐藏状态
            hidden_states = outputs[0]

        # 将隐藏状态投影到输出空间
        hidden_states = self.projector(hidden_states)

        # 计算池化输出，根据是否提供了注意力掩码选择不同的方式
        if attention_mask is None:
            pooled_output = hidden_states.mean(dim=1)
        else:
            padding_mask = self._get_feature_vector_attention_mask(hidden_states.shape[1], attention_mask)
            hidden_states[~padding_mask] = 0.0
            pooled_output = hidden_states.sum(dim=1) / padding_mask.sum(dim=1).view(-1, 1)

        # 使用分类器预测结果
        logits = self.classifier(pooled_output)

        # 如果提供了标签，则计算损失
        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        # 根据是否使用返回字典，决定返回格式
        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        # 使用自定义的SequenceClassifierOutput类来返回结果，包括损失、预测结果、隐藏状态和注意力权重
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    """
    Wav2Vec2 Model with a frame classification head on top for tasks like Speaker Diarization.
    """

    # 继承自Wav2Vec2PreTrainedModel，用于音频帧分类任务，例如说话人辨识
    class Wav2Vec2ForAudioFrameClassification(Wav2Vec2PreTrainedModel):
        
        def __init__(self, config):
            super().__init__(config)

            # 如果配置允许使用适配器，并且添加了适配器，抛出异常
            if hasattr(config, "add_adapter") and config.add_adapter:
                raise ValueError(
                    "Audio frame classification does not support the use of Wav2Vec2 adapters (config.add_adapter=True)"
                )
            
            # 初始化Wav2Vec2模型
            self.wav2vec2 = Wav2Vec2Model(config)
            num_layers = config.num_hidden_layers + 1  # transformer layers + input embeddings
            
            # 如果配置使用加权层求和，则初始化层权重
            if config.use_weighted_layer_sum:
                self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
            
            # 初始化分类器线性层
            self.classifier = nn.Linear(config.hidden_size, config.num_labels)
            self.num_labels = config.num_labels

            # 初始化权重
            self.init_weights()

        # 弃用警告：冻结特征提取器方法，建议使用freeze_feature_encoder代替
        def freeze_feature_extractor(self):
            warnings.warn(
                "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. "
                "Please use the equivalent `freeze_feature_encoder` method instead.",
                FutureWarning,
            )
            self.freeze_feature_encoder()

        # 冻结特征编码器，禁用特征编码器参数的梯度计算，保持在训练过程中不更新
        def freeze_feature_encoder(self):
            self.wav2vec2.feature_extractor._freeze_parameters()

        # 冻结基础模型，禁用基础模型参数的梯度计算，只更新分类头部
        def freeze_base_model(self):
            for param in self.wav2vec2.parameters():
                param.requires_grad = False

        # 前向传播方法，执行模型的前向计算
        @add_start_docstrings_to_model_forward(WAV_2_VEC_2_INPUTS_DOCSTRING)
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
            ):
    ) -> Union[Tuple, TokenClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        # 确定是否需要返回字典形式的输出，若未指定则使用配置中的默认设置
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 根据配置决定是否输出隐藏状态
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states

        # 调用 wav2vec2 模型进行前向传播
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 若使用加权层求和策略，则对隐藏状态进行加权求和
        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]  # 获取隐藏状态列表
            hidden_states = torch.stack(hidden_states, dim=1)  # 在新维度上堆叠隐藏状态张量
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)  # 对层权重进行 softmax 归一化
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)  # 加权求和隐藏状态
        else:
            hidden_states = outputs[0]  # 直接使用第一个输出作为隐藏状态

        # 将隐藏状态输入分类器，生成 logits
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            # 如果提供了标签，计算交叉熵损失
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), torch.argmax(labels.view(-1, self.num_labels), axis=1))

        # 若不需要返回字典形式的输出，则返回 logits 和隐藏状态列表
        if not return_dict:
            output = (logits,) + outputs[_HIDDEN_STATES_START_POSITION:]
            return output

        # 返回 TokenClassifierOutput 对象，包括损失、logits、隐藏状态和注意力
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
# 定义 AMSoftmaxLoss 类，继承自 nn.Module
class AMSoftmaxLoss(nn.Module):
    def __init__(self, input_dim, num_labels, scale=30.0, margin=0.4):
        super(AMSoftmaxLoss, self).__init__()
        # 初始化参数 scale 和 margin
        self.scale = scale
        self.margin = margin
        self.num_labels = num_labels
        # 使用 nn.Parameter 定义可学习参数 weight
        self.weight = nn.Parameter(torch.randn(input_dim, num_labels), requires_grad=True)
        # 使用交叉熵损失函数
        self.loss = nn.CrossEntropyLoss()

    # 前向传播函数，接收 hidden_states 和 labels 作为输入
    def forward(self, hidden_states, labels):
        # 将 labels 展平以适应 CrossEntropyLoss 函数的要求
        labels = labels.flatten()
        # 对 weight 和 hidden_states 进行 L2 归一化
        weight = nn.functional.normalize(self.weight, dim=0)
        hidden_states = nn.functional.normalize(hidden_states, dim=1)
        # 计算余弦相似度 cos_theta
        cos_theta = torch.mm(hidden_states, weight)
        # 计算 AMSoftmax 中的 psi 值
        psi = cos_theta - self.margin

        # 根据 labels 生成 one-hot 编码
        onehot = nn.functional.one_hot(labels, self.num_labels)
        # 根据是否为 one-hot 中的类别，调整 logits 的值
        logits = self.scale * torch.where(onehot.bool(), psi, cos_theta)
        # 计算最终的损失值
        loss = self.loss(logits, labels)

        return loss


# 定义 TDNNLayer 类，继承自 nn.Module
class TDNNLayer(nn.Module):
    def __init__(self, config, layer_id=0):
        super().__init__()
        # 初始化 TDNN 层的参数
        self.in_conv_dim = config.tdnn_dim[layer_id - 1] if layer_id > 0 else config.tdnn_dim[layer_id]
        self.out_conv_dim = config.tdnn_dim[layer_id]
        self.kernel_size = config.tdnn_kernel[layer_id]
        self.dilation = config.tdnn_dilation[layer_id]

        # 使用 nn.Linear 定义 kernel（权重矩阵）
        self.kernel = nn.Linear(self.in_conv_dim * self.kernel_size, self.out_conv_dim)
        # 激活函数使用 ReLU
        self.activation = nn.ReLU()

    # 前向传播函数，接收 hidden_states 作为输入，返回 torch.Tensor
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 检查是否 peft 可用，如果可用，导入相关模块
        if is_peft_available():
            from peft.tuners.lora import LoraLayer

            # 如果 kernel 是 LoraLayer 类型，则发出警告
            if isinstance(self.kernel, LoraLayer):
                warnings.warn(
                    "Detected LoRA on TDNNLayer. LoRA weights won't be applied due to optimization. "
                    "You should exclude TDNNLayer from LoRA's target modules.",
                )

        # 转置 hidden_states 的维度，以便与 conv1d 函数的要求匹配
        hidden_states = hidden_states.transpose(1, 2)
        # 将 self.kernel 的权重矩阵重新视图成 conv1d 函数所需的形状，并转置维度
        weight = self.kernel.weight.view(self.out_conv_dim, self.kernel_size, self.in_conv_dim).transpose(1, 2)
        # 使用 conv1d 函数进行卷积操作
        hidden_states = nn.functional.conv1d(hidden_states, weight, self.kernel.bias, dilation=self.dilation)
        # 再次转置 hidden_states 的维度，使其与输入形状相匹配
        hidden_states = hidden_states.transpose(1, 2)

        # 应用激活函数 ReLU
        hidden_states = self.activation(hidden_states)
        # 返回处理后的 hidden_states
        return hidden_states


# 使用 add_start_docstrings 装饰器为 Wav2Vec2ForXVector 类添加文档字符串
@add_start_docstrings(
    """
    Wav2Vec2 Model with an XVector feature extraction head on top for tasks like Speaker Verification.
    """,
    WAV_2_VEC_2_START_DOCSTRING,
)
# 定义 Wav2Vec2ForXVector 类，继承自 Wav2Vec2PreTrainedModel
class Wav2Vec2ForXVector(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        # 初始化 WAV2VEC 2 模型
        self.wav2vec2 = Wav2Vec2Model(config)
        # 计算总层数，包括 Transformer 层和输入嵌入层
        num_layers = config.num_hidden_layers + 1  # transformer layers + input embeddings
        # 如果配置要使用加权层求和，则初始化层权重为均匀分布的参数
        if config.use_weighted_layer_sum:
            self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        # 初始化投影层，将隐藏状态映射到 TDNN 的第一个维度
        self.projector = nn.Linear(config.hidden_size, config.tdnn_dim[0])

        # 初始化 TDNN 层列表，根据配置文件中的维度定义
        tdnn_layers = [TDNNLayer(config, i) for i in range(len(config.tdnn_dim))]
        self.tdnn = nn.ModuleList(tdnn_layers)

        # 初始化特征提取器，将 TDNN 的最后一层输出映射到 x-vector 的输出维度
        self.feature_extractor = nn.Linear(config.tdnn_dim[-1] * 2, config.xvector_output_dim)
        # 初始化分类器，将 x-vector 的输出映射到最终的标签数量维度
        self.classifier = nn.Linear(config.xvector_output_dim, config.xvector_output_dim)

        # 初始化损失函数，使用 AMSoftmaxLoss，配置为 x-vector 的输出维度和标签数量
        self.objective = AMSoftmaxLoss(config.xvector_output_dim, config.num_labels)

        # 初始化模型权重
        self.init_weights()

    def freeze_feature_extractor(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        # 发出警告信息，提示该方法即将被弃用，并建议使用等效的 freeze_feature_encoder 方法
        warnings.warn(
            "The method `freeze_feature_extractor` is deprecated and will be removed in Transformers v5. "
            "Please use the equivalent `freeze_feature_encoder` method instead.",
            FutureWarning,
        )
        # 调用 freeze_feature_encoder 方法，冻结特征编码器的参数，停止其在训练期间的梯度计算
        self.freeze_feature_encoder()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        # 冻结特征编码器的参数，停止其在训练期间的梯度计算
        self.wav2vec2.feature_extractor._freeze_parameters()

    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        # 遍历 WAV2VEC 2 模型的所有参数，并设置其 requires_grad 属性为 False，从而停止其在训练期间的梯度计算
        for param in self.wav2vec2.parameters():
            param.requires_grad = False

    def _get_tdnn_output_lengths(self, input_lengths: Union[torch.LongTensor, int]):
        """
        Computes the output length of the TDNN layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            # 从 PyTorch 文档中获取的一维卷积层输出长度计算公式
            return (input_length - kernel_size) // stride + 1

        # 根据配置文件中的每个 TDNN 层的卷积核大小计算输出长度
        for kernel_size in self.config.tdnn_kernel:
            input_lengths = _conv_out_length(input_lengths, kernel_size, 1)

        return input_lengths

    @add_start_docstrings_to_model_forward(WAV_2_VEC_2_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        checkpoint=_XVECTOR_CHECKPOINT,
        output_type=XVectorOutput,
        config_class=_CONFIG_FOR_DOC,
        modality="audio",
        expected_output=_XVECTOR_EXPECTED_OUTPUT,
    )
    # 定义神经网络模型的前向传播方法
    def forward(
        # 输入的张量数据，可以为空
        self,
        input_values: Optional[torch.Tensor],
        # 注意力掩码，可选参数，默认为空
        attention_mask: Optional[torch.Tensor] = None,
        # 是否输出注意力权重，可选参数，默认为空
        output_attentions: Optional[bool] = None,
        # 是否输出隐藏状态，可选参数，默认为空
        output_hidden_states: Optional[bool] = None,
        # 是否返回结果的字典格式，可选参数，默认为空
        return_dict: Optional[bool] = None,
        # 标签数据，可选参数，默认为空
        labels: Optional[torch.Tensor] = None,
        ) -> Union[Tuple, XVectorOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        # 根据需要确定是否返回字典形式的输出
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # 根据配置决定是否输出隐藏状态
        output_hidden_states = True if self.config.use_weighted_layer_sum else output_hidden_states

        # 调用wav2vec2模型进行推理
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # 如果使用加权层求和，则进行加权和操作
        if self.config.use_weighted_layer_sum:
            hidden_states = outputs[_HIDDEN_STATES_START_POSITION]
            hidden_states = torch.stack(hidden_states, dim=1)
            norm_weights = nn.functional.softmax(self.layer_weights, dim=-1)
            hidden_states = (hidden_states * norm_weights.view(-1, 1, 1)).sum(dim=1)
        else:
            hidden_states = outputs[0]

        # 将隐藏状态投影到特征空间
        hidden_states = self.projector(hidden_states)

        # 遍历所有TDNN层进行特征提取
        for tdnn_layer in self.tdnn:
            hidden_states = tdnn_layer(hidden_states)

        # 统计池化
        if attention_mask is None:
            # 如果没有注意力掩码，则计算平均特征和标准差特征
            mean_features = hidden_states.mean(dim=1)
            std_features = hidden_states.std(dim=1)
        else:
            # 根据注意力掩码计算特征提取器的输出长度
            feat_extract_output_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(dim=1))
            tdnn_output_lengths = self._get_tdnn_output_lengths(feat_extract_output_lengths)
            mean_features = []
            std_features = []
            # 根据长度截取隐藏状态并计算平均特征和标准差特征
            for i, length in enumerate(tdnn_output_lengths):
                mean_features.append(hidden_states[i, :length].mean(dim=0))
                std_features.append(hidden_states[i, :length].std(dim=0))
            mean_features = torch.stack(mean_features)
            std_features = torch.stack(std_features)
        statistic_pooling = torch.cat([mean_features, std_features], dim=-1)

        # 将统计池化后的特征传入特征提取器和分类器
        output_embeddings = self.feature_extractor(statistic_pooling)
        logits = self.classifier(output_embeddings)

        # 计算损失
        loss = None
        if labels is not None:
            loss = self.objective(logits, labels)

        # 根据return_dict决定返回的输出形式
        if not return_dict:
            # 如果不返回字典，则返回一个元组
            output = (logits, output_embeddings) + outputs[_HIDDEN_STATES_START_POSITION:]
            return ((loss,) + output) if loss is not None else output

        # 如果需要返回字典形式的输出，则返回XVectorOutput对象
        return XVectorOutput(
            loss=loss,
            logits=logits,
            embeddings=output_embeddings,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```