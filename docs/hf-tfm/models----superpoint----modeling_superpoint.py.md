# `.\models\superpoint\modeling_superpoint.py`

```
# 版权声明和许可信息，指明代码归属和使用许可
# Copyright 2024 The HuggingFace Team. All rights reserved.
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

"""PyTorch SuperPoint model."""

# 引入必要的库和模块
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
from torch import nn

# 引入 transformers 库中的模块和类
from transformers import PreTrainedModel
from transformers.modeling_outputs import (
    BaseModelOutputWithNoAttention,
)
from transformers.models.superpoint.configuration_superpoint import SuperPointConfig

# 引入内部工具函数和类
from ...pytorch_utils import is_torch_greater_or_equal_than_1_13
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
)

# 获取日志记录器
logger = logging.get_logger(__name__)

# 用于文档的配置和检查点名称
_CONFIG_FOR_DOC = "SuperPointConfig"
_CHECKPOINT_FOR_DOC = "magic-leap-community/superpoint"

# 预训练模型的存档列表
SUPERPOINT_PRETRAINED_MODEL_ARCHIVE_LIST = ["magic-leap-community/superpoint"]

# 从图像中移除靠近边界的关键点的函数
def remove_keypoints_from_borders(
    keypoints: torch.Tensor, scores: torch.Tensor, border: int, height: int, width: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Removes keypoints (and their associated scores) that are too close to the border"""
    mask_h = (keypoints[:, 0] >= border) & (keypoints[:, 0] < (height - border))
    mask_w = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (width - border))
    mask = mask_h & mask_w
    return keypoints[mask], scores[mask]

# 保留具有最高分数的 k 个关键点的函数
def top_k_keypoints(keypoints: torch.Tensor, scores: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Keeps the k keypoints with highest score"""
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0)
    return keypoints[indices], scores

# 应用非最大抑制算法的函数，用于处理关键点的分数
def simple_nms(scores: torch.Tensor, nms_radius: int) -> torch.Tensor:
    """Applies non-maximum suppression on scores"""
    if nms_radius < 0:
        raise ValueError("Expected positive values for nms_radius")

    def max_pool(x):
        return nn.functional.max_pool2d(x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)

# 定义一个输出类，继承自 ModelOutput，用于描述图像中的点的信息
@dataclass
class ImagePointDescriptionOutput(ModelOutput):
    """
    # 图像关键点描述模型输出的基类。由于关键点检测的性质，关键点数量在图像之间可以不固定，这使得批处理变得复杂。
    # 在图像批处理中，将关键点的最大数量设置为关键点、分数和描述符张量的维度。掩码张量用于指示关键点、分数和描述符张量中的哪些值是关键点信息，哪些是填充。
    
    Args:
        last_hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            解码器模型最后一层输出的隐藏状态序列。
        keypoints (`torch.FloatTensor` of shape `(batch_size, num_keypoints, 2)`):
            给定图像中预测关键点的相对（x，y）坐标。
        scores (`torch.FloatTensor` of shape `(batch_size, num_keypoints)`):
            预测关键点的分数。
        descriptors (`torch.FloatTensor` of shape `(batch_size, num_keypoints, descriptor_size)`):
            预测关键点的描述符。
        mask (`torch.BoolTensor` of shape `(batch_size, num_keypoints)`):
            指示关键点、分数和描述符张量中哪些值是关键点信息的掩码。
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, 返回当 `output_hidden_states=True` 传递或 `config.output_hidden_states=True` 时):
            `torch.FloatTensor` 元组（如果模型有嵌入层，则为输出的一个 + 每个阶段的一个）。模型在每个阶段输出的隐藏状态（也称为特征图）。
    """
    
    # 最后一层隐藏状态，默认为 None
    last_hidden_state: torch.FloatTensor = None
    # 关键点坐标，默认为 None（可选）
    keypoints: Optional[torch.IntTensor] = None
    # 关键点分数，默认为 None（可选）
    scores: Optional[torch.FloatTensor] = None
    # 关键点描述符，默认为 None（可选）
    descriptors: Optional[torch.FloatTensor] = None
    # 关键点掩码，默认为 None（可选）
    mask: Optional[torch.BoolTensor] = None
    # 隐藏状态，默认为 None（可选）
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
class SuperPointConvBlock(nn.Module):
    def __init__(
        self, config: SuperPointConfig, in_channels: int, out_channels: int, add_pooling: bool = False
    ) -> None:
        super().__init__()
        # 定义第一个卷积层，输入通道数为in_channels，输出通道数为out_channels，使用3x3的卷积核
        self.conv_a = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        # 定义第二个卷积层，输入输出通道数均为out_channels，使用3x3的卷积核
        self.conv_b = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        # ReLU激活函数，inplace=True表示原地操作，节省内存
        self.relu = nn.ReLU(inplace=True)
        # 如果add_pooling为True，则定义最大池化层，池化核大小为2x2，步长为2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2) if add_pooling else None

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # 对输入进行第一层卷积和ReLU激活
        hidden_states = self.relu(self.conv_a(hidden_states))
        # 对结果进行第二层卷积和ReLU激活
        hidden_states = self.relu(self.conv_b(hidden_states))
        # 如果定义了池化层，则对结果进行最大池化操作
        if self.pool is not None:
            hidden_states = self.pool(hidden_states)
        return hidden_states


class SuperPointEncoder(nn.Module):
    """
    SuperPoint encoder module. It is made of 4 convolutional layers with ReLU activation and max pooling, reducing the
     dimensionality of the image.
    """

    def __init__(self, config: SuperPointConfig) -> None:
        super().__init__()
        # SuperPoint使用单通道图像
        self.input_dim = 1

        conv_blocks = []
        # 添加第一个卷积块，使用SuperPointConvBlock定义的卷积结构，添加了最大池化
        conv_blocks.append(
            SuperPointConvBlock(config, self.input_dim, config.encoder_hidden_sizes[0], add_pooling=True)
        )
        # 添加中间的卷积块，遍历encoder_hidden_sizes并构建多个SuperPointConvBlock实例
        for i in range(1, len(config.encoder_hidden_sizes) - 1):
            conv_blocks.append(
                SuperPointConvBlock(
                    config, config.encoder_hidden_sizes[i - 1], config.encoder_hidden_sizes[i], add_pooling=True
                )
            )
        # 添加最后一个卷积块，不添加最大池化
        conv_blocks.append(
            SuperPointConvBlock(
                config, config.encoder_hidden_sizes[-2], config.encoder_hidden_sizes[-1], add_pooling=False
            )
        )
        # 将所有卷积块封装为ModuleList
        self.conv_blocks = nn.ModuleList(conv_blocks)

    def forward(
        self,
        input,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
    ) -> Union[Tuple, BaseModelOutputWithNoAttention]:
        all_hidden_states = () if output_hidden_states else None

        # 对每个卷积块进行前向传播，保存所有隐藏状态（如果需要）
        for conv_block in self.conv_blocks:
            input = conv_block(input)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (input,)
        output = input
        # 根据return_dict返回不同的输出格式
        if not return_dict:
            return tuple(v for v in [output, all_hidden_states] if v is not None)

        return BaseModelOutputWithNoAttention(
            last_hidden_state=output,
            hidden_states=all_hidden_states,
        )


class SuperPointInterestPointDecoder(nn.Module):
    """
    The SuperPointInterestPointDecoder uses the output of the SuperPointEncoder to compute the keypoint with scores.
    """
    def __init__(self, config: SuperPointConfig) -> None:
        super().__init__()
        self.keypoint_threshold = config.keypoint_threshold
        self.max_keypoints = config.max_keypoints
        self.nms_radius = config.nms_radius
        self.border_removal_distance = config.border_removal_distance

        self.relu = nn.ReLU(inplace=True)  # 初始化 ReLU 激活函数
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # 初始化最大池化层
        self.conv_score_a = nn.Conv2d(
            config.encoder_hidden_sizes[-1],  # 输入通道数为编码器最后隐藏层的大小
            config.decoder_hidden_size,  # 输出通道数为解码器隐藏层的大小
            kernel_size=3, stride=1, padding=1,  # 使用 3x3 的卷积核，填充为1
        )
        self.conv_score_b = nn.Conv2d(
            config.decoder_hidden_size,  # 输入通道数为解码器隐藏层的大小
            config.keypoint_decoder_dim,  # 输出通道数为关键点解码器的维度大小
            kernel_size=1, stride=1, padding=0  # 使用 1x1 的卷积核，无填充
        )

    def forward(self, encoded: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scores = self._get_pixel_scores(encoded)
        keypoints, scores = self._extract_keypoints(scores)

        return keypoints, scores

    def _get_pixel_scores(self, encoded: torch.Tensor) -> torch.Tensor:
        """根据编码器输出，计算图像每个像素点的分数"""
        scores = self.relu(self.conv_score_a(encoded))  # 使用 ReLU 激活函数对卷积结果进行非线性处理
        scores = self.conv_score_b(scores)  # 继续卷积操作
        scores = nn.functional.softmax(scores, 1)[:, :-1]  # 对最后一维进行 softmax 操作，生成概率分布
        batch_size, _, height, width = scores.shape
        scores = scores.permute(0, 2, 3, 1).reshape(batch_size, height, width, 8, 8)  # 调整张量维度
        scores = scores.permute(0, 1, 3, 2, 4).reshape(batch_size, height * 8, width * 8)  # 再次调整张量维度
        scores = simple_nms(scores, self.nms_radius)  # 对分数进行简单的非极大值抑制处理
        return scores

    def _extract_keypoints(self, scores: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """根据分数提取关键点像素，用于描述符计算"""
        _, height, width = scores.shape

        # 根据分数阈值筛选关键点
        keypoints = torch.nonzero(scores[0] > self.keypoint_threshold)
        scores = scores[0][tuple(keypoints.t())]

        # 去除靠近图像边界的关键点
        keypoints, scores = remove_keypoints_from_borders(
            keypoints, scores, self.border_removal_distance, height * 8, width * 8
        )

        # 保留分数最高的 k 个关键点
        if self.max_keypoints >= 0:
            keypoints, scores = top_k_keypoints(keypoints, scores, self.max_keypoints)

        # 将 (y, x) 转换为 (x, y)
        keypoints = torch.flip(keypoints, [1]).float()

        return keypoints, scores
class SuperPointDescriptorDecoder(nn.Module):
    """
    The SuperPointDescriptorDecoder uses the outputs of both the SuperPointEncoder and the
    SuperPointInterestPointDecoder to compute the descriptors at the keypoints locations.

    The descriptors are first computed by a convolutional layer, then normalized to have a norm of 1. The descriptors
    are then interpolated at the keypoints locations.
    """

    def __init__(self, config: SuperPointConfig) -> None:
        super().__init__()

        # ReLU 激活函数，inplace=True 表示原地操作
        self.relu = nn.ReLU(inplace=True)
        # 最大池化层，kernel_size=2 表示池化核大小为 2x2，stride=2 表示步幅为 2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 第一个描述符卷积层，输入通道数为 config.encoder_hidden_sizes[-1]，输出通道数为 config.decoder_hidden_size，卷积核大小为 3x3
        self.conv_descriptor_a = nn.Conv2d(
            config.encoder_hidden_sizes[-1],
            config.decoder_hidden_size,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        # 第二个描述符卷积层，输入通道数为 config.decoder_hidden_size，输出通道数为 config.descriptor_decoder_dim，卷积核大小为 1x1
        self.conv_descriptor_b = nn.Conv2d(
            config.decoder_hidden_size,
            config.descriptor_decoder_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def forward(self, encoded: torch.Tensor, keypoints: torch.Tensor) -> torch.Tensor:
        """Based on the encoder output and the keypoints, compute the descriptors for each keypoint"""
        # 计算描述符，先经过第一个卷积层和 ReLU 激活函数，再经过第二个卷积层
        descriptors = self.conv_descriptor_b(self.relu(self.conv_descriptor_a(encoded)))
        # 对描述符进行 L2 归一化，dim=1 表示在通道维度进行归一化
        descriptors = nn.functional.normalize(descriptors, p=2, dim=1)

        # 插值计算描述符在关键点位置处的值
        descriptors = self._sample_descriptors(keypoints[None], descriptors[0][None], 8)[0]

        # 将描述符的维度从 [descriptor_dim, num_keypoints] 转置为 [num_keypoints, descriptor_dim]
        descriptors = torch.transpose(descriptors, 0, 1)

        return descriptors

    @staticmethod
    def _sample_descriptors(keypoints, descriptors, scale: int = 8) -> torch.Tensor:
        """Interpolate descriptors at keypoint locations"""
        batch_size, num_channels, height, width = descriptors.shape
        # 调整关键点位置，将其缩放并归一化到 (-1, 1) 的范围内
        keypoints = keypoints - scale / 2 + 0.5
        divisor = torch.tensor([[(width * scale - scale / 2 - 0.5), (height * scale - scale / 2 - 0.5)]])
        divisor = divisor.to(keypoints)
        keypoints /= divisor
        keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
        kwargs = {"align_corners": True} if is_torch_greater_or_equal_than_1_13 else {}
        # 使用双线性插值在描述符上进行网格采样，调整关键点位置
        keypoints = keypoints.view(batch_size, 1, -1, 2)
        descriptors = nn.functional.grid_sample(descriptors, keypoints, mode="bilinear", **kwargs)
        # 调整描述符的形状 [batch_size, descriptor_decoder_dim, num_channels, num_keypoints] -> [batch_size, descriptor_decoder_dim, num_keypoints]
        descriptors = descriptors.reshape(batch_size, num_channels, -1)
        # 对描述符进行 L2 归一化，dim=1 表示在通道维度进行归一化
        descriptors = nn.functional.normalize(descriptors, p=2, dim=1)
        return descriptors


class SuperPointPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = SuperPointConfig
    base_model_prefix = "superpoint"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = False

    def _init_weights(self, module: Union[nn.Linear, nn.Conv2d, nn.LayerNorm]) -> None:
        """Initialize the weights"""
        # 如果 module 是线性层或卷积层
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            # 使用正态分布初始化权重，均值为 0，标准差为配置中的初始化范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果存在偏置，将其初始化为零
            if module.bias is not None:
                module.bias.data.zero_()
        # 如果 module 是 LayerNorm 层
        elif isinstance(module, nn.LayerNorm):
            # 将偏置初始化为零
            module.bias.data.zero_()
            # 将权重初始化为全一
            module.weight.data.fill_(1.0)

    def extract_one_channel_pixel_values(self, pixel_values: torch.FloatTensor) -> torch.FloatTensor:
        """
        Assuming pixel_values has shape (batch_size, 3, height, width), and that all channels values are the same,
        extract the first channel value to get a tensor of shape (batch_size, 1, height, width) for SuperPoint. This is
        a workaround for the issue discussed in :
        https://github.com/huggingface/transformers/pull/25786#issuecomment-1730176446

        Args:
            pixel_values: torch.FloatTensor of shape (batch_size, 3, height, width)

        Returns:
            pixel_values: torch.FloatTensor of shape (batch_size, 1, height, width)

        """
        # 提取第一个通道的像素值，以解决超级点模型的问题
        return pixel_values[:, 0, :, :][:, None, :, :]
# 定义一个包含详细信息的 PyTorch 模型类，继承自 `torch.nn.Module`。此模型可以作为常规的 PyTorch 模块使用，有关使用和行为的所有事项请参考 PyTorch 文档。
SUPERPOINT_START_DOCSTRING = r"""
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass. Use it
    as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.

    Parameters:
        config ([`SuperPointConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
    """

# 描述模型输入文档的字符串，包括像素值、是否返回隐藏状态、是否返回字典等参数的详细信息
SUPERPOINT_INPUTS_DOCSTRING = r"""
Args:
    pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
        Pixel values. Pixel values can be obtained using [`SuperPointImageProcessor`]. See
        [`SuperPointImageProcessor.__call__`] for details.
    output_hidden_states (`bool`, *optional*):
        Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for more
        detail.
    return_dict (`bool`, *optional*):
        Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
    """

# 使用装饰器 `add_start_docstrings` 将超级点模型的输出文档字符串和起始文档字符串添加到类上
@add_start_docstrings(
    "SuperPoint model outputting keypoints and descriptors.",
    SUPERPOINT_START_DOCSTRING,
)
# 定义了一个超级点（SuperPoint）关键点检测模型类，继承自 `SuperPointPreTrainedModel`
class SuperPointForKeypointDetection(SuperPointPreTrainedModel):
    """
    SuperPoint model. It consists of a SuperPointEncoder, a SuperPointInterestPointDecoder and a
    SuperPointDescriptorDecoder. SuperPoint was proposed in `SuperPoint: Self-Supervised Interest Point Detection and
    Description <https://arxiv.org/abs/1712.07629>`__ by Daniel DeTone, Tomasz Malisiewicz, and Andrew Rabinovich. It
    is a fully convolutional neural network that extracts keypoints and descriptors from an image. It is trained in a
    self-supervised manner, using a combination of a photometric loss and a loss based on the homographic adaptation of
    keypoints. It is made of a convolutional encoder and two decoders: one for keypoints and one for descriptors.
    """

    # 初始化方法，接受一个 `SuperPointConfig` 类型的参数配置，并调用父类的初始化方法
    def __init__(self, config: SuperPointConfig) -> None:
        super().__init__(config)

        # 将配置参数保存到实例变量中
        self.config = config

        # 创建超级点编码器、关键点解码器和描述符解码器实例
        self.encoder = SuperPointEncoder(config)
        self.keypoint_decoder = SuperPointInterestPointDecoder(config)
        self.descriptor_decoder = SuperPointDescriptorDecoder(config)

        # 调用初始化后处理方法
        self.post_init()

    # 使用装饰器 `add_start_docstrings_to_model_forward` 将输入文档字符串添加到模型的 `forward` 方法
    @add_start_docstrings_to_model_forward(SUPERPOINT_INPUTS_DOCSTRING)
    # 前向传播方法，接收像素值、标签、是否返回隐藏状态和是否返回字典作为参数
    def forward(
        self,
        pixel_values: torch.FloatTensor = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # ...
```