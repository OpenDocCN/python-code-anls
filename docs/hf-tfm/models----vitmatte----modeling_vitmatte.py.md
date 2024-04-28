# `.\transformers\models\vitmatte\modeling_vitmatte.py`

```py
# 编码声明，指定源代码文件编码格式为 UTF-8
# 版权声明，版权归 HUST-VL 和 HuggingFace Inc. 团队所有
#
# 根据 Apache 许可证 2.0 版本进行许可;
# 除非符合许可证的规定，否则您不得使用此文件。
# 您可以在以下网址获取许可证副本:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则按“原样”提供的软件
# 分发，不提供任何形式的明示或暗示的担保或条件。
# 有关详细信息，请参阅许可证。
""" PyTorch ViTMatte model."""  # PyTorch ViTMatte 模型的描述

# 导入必要的库
from dataclasses import dataclass  # 导入 dataclass 装饰器，用于创建数据类
from typing import Optional, Tuple  # 导入 Optional 和 Tuple 类型提示

import torch  # 导入 PyTorch 库
from torch import nn  # 导入神经网络模块

from ... import AutoBackbone  # 导入 AutoBackbone 模块
from ...modeling_utils import PreTrainedModel  # 导入预训练模型的基类
from ...utils import (  # 导入实用工具函数和类
    ModelOutput,  # 模型输出类
    add_start_docstrings,  # 添加文档字符串的装饰器
    add_start_docstrings_to_model_forward,  # 为模型前向方法添加文档字符串的装饰器
    replace_return_docstrings,  # 替换返回文档字符串的装饰器
)
from .configuration_vitmatte import VitMatteConfig  # 导入 ViTMatte 模型的配置类


VITMATTE_PRETRAINED_MODEL_ARCHIVE_LIST = [  # ViTMatte 预训练模型的存档列表
    "hustvl/vitmatte-small-composition-1k",  # ViTMatte 小型组合模型
    # 查看所有 ViTMatte 模型: https://huggingface.co/models?filter=vitmatte
]


# 通用文档字符串
_CONFIG_FOR_DOC = "VitMatteConfig"  # 用于文档的配置类


@dataclass  # 使用 dataclass 装饰器装饰类，创建数据类
class ImageMattingOutput(ModelOutput):  # 图像抠图模型的输出类，继承自模型输出类

    """
    Class for outputs of image matting models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Loss.
        alphas (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
           Estimated alpha values.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each stage) of shape `(batch_size, sequence_length, hidden_size)`. Hidden-states
            (also called feature maps) of the model at the output of each stage.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, patch_size,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None  # 损失值，可选的 torch.FloatTensor 类型，形状为 (1,)，在提供标签时返回
    alphas: torch.FloatTensor = None  # 估计的 alpha 值，torch.FloatTensor 类型，形状为 (batch_size, num_channels, height, width)
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None  # 隐藏状态，可选的 torch.FloatTensor 元组类型，表示模型各阶段的隐藏状态
    attentions: Optional[Tuple[torch.FloatTensor]] = None  # 注意力权重，可选的 torch.FloatTensor 元组类型，表示各层的注意力权重


class VitMattePreTrainedModel(PreTrainedModel):  # ViTMatte 预训练模型的基类，继承自预训练模型基类

    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = VitMatteConfig  # 配置类为 VitMatteConfig 类
    # 主输入名称，用于指定神经网络模型的输入
    main_input_name = "pixel_values"
    
    # 指示是否支持梯度检查点，用于在训练中节省内存
    supports_gradient_checkpointing = True
    
    # 初始化神经网络模型的权重
    def _init_weights(self, module):
        # 如果当前模块是卷积层
        if isinstance(module, nn.Conv2d):
            # 初始化卷积层的权重，采用正态分布，均值为0，标准差为配置中指定的初始范围
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果卷积层有偏置项
            if module.bias is not None:
                # 将偏置项数据初始化为0
                module.bias.data.zero_()
class VitMatteBasicConv3x3(nn.Module):
    """
    Basic convolution layers including: Conv3x3, BatchNorm2d, ReLU layers.
    """
    
    def __init__(self, config, in_channels, out_channels, stride=2, padding=1):
        # 初始化方法，定义基本的卷积层，批量归一化层和ReLU激活函数层
        super().__init__()
        # 创建一个卷积层
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
            bias=False,
        )
        # 创建一个批量归一化层
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=config.batch_norm_eps)
        # 创建一个ReLU激活函数层
        self.relu = nn.ReLU()

    def forward(self, hidden_state):
        # 定义前向传播方法，包括卷积、批量归一化和ReLU激活
        hidden_state = self.conv(hidden_state)
        hidden_state = self.batch_norm(hidden_state)
        hidden_state = self.relu(hidden_state)

        return hidden_state


class VitMatteConvStream(nn.Module):
    """
    Simple ConvStream containing a series of basic conv3x3 layers to extract detail features.
    """
    
    def __init__(self, config):
        # 初始化方法，创建一个包含多个基本conv3x3层的简单ConvStream用于提取详细特征
        super().__init__()

        in_channels = config.backbone_config.num_channels
        out_channels = config.convstream_hidden_sizes

        self.convs = nn.ModuleList()
        self.conv_chans = [in_channels] + out_channels

        for i in range(len(self.conv_chans) - 1):
            in_chan_ = self.conv_chans[i]
            out_chan_ = self.conv_chans[i + 1]
            self.convs.append(VitMatteBasicConv3x3(config, in_chan_, out_chan_))

    def forward(self, pixel_values):
        # 定义前向传播方法，遍历并应用每个基本的conv3x3层
        out_dict = {"detailed_feature_map_0": pixel_values}
        embeddings = pixel_values
        for i in range(len(self.convs)):
            embeddings = self.convs[i](embeddings)
            name_ = "detailed_feature_map_" + str(i + 1)
            out_dict[name_] = embeddings

        return out_dict


class VitMatteFusionBlock(nn.Module):
    """
    Simple fusion block to fuse features from ConvStream and Plain Vision Transformer.
    """
    
    def __init__(self, config, in_channels, out_channels):
        # 初始化方法，创建一个简单的融合块用于融合来自ConvStream和Plain Vision Transformer的特征
        super().__init__()
        # 创建一个基本的conv3x3层
        self.conv = VitMatteBasicConv3x3(config, in_channels, out_channels, stride=1, padding=1)

    def forward(self, features, detailed_feature_map):
        # 定义前向传播方法，将特征上采样并与详细特征图拼接，然后应用conv3x3层
        upscaled_features = nn.functional.interpolate(features, scale_factor=2, mode="bilinear", align_corners=False)
        out = torch.cat([detailed_feature_map, upscaled_features], dim=1)
        out = self.conv(out)

        return out


class VitMatteHead(nn.Module):
    """
    Simple Matting Head, containing only conv3x3 and conv1x1 layers.
    """
    
    def __init__(self, config):
        # 初始化方法，创建一个简单的Matting Head，只包含conv3x3和conv1x1层
        super().__init__()

        in_channels = config.fusion_hidden_sizes[-1]
        mid_channels = 16

        self.matting_convs = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(True),
            nn.Conv2d(mid_channels, 1, kernel_size=1, stride=1, padding=0),
        )
``` 
    # 定义一个名为 forward 的方法，接收 hidden_state 参数
    def forward(self, hidden_state):
        # 使用 matting_convs 对 hidden_state 进行处理
        hidden_state = self.matting_convs(hidden_state)
        # 返回处理后的 hidden_state
        return hidden_state
class VitMatteDetailCaptureModule(nn.Module):
    """
    Simple and lightweight Detail Capture Module for ViT Matting.
    """

    def __init__(self, config):
        super().__init__()
        # 检查融合隐藏层大小列表与卷积流隐藏层大小列表长度是否匹配
        if len(config.fusion_hidden_sizes) != len(config.convstream_hidden_sizes) + 1:
            raise ValueError(
                "The length of fusion_hidden_sizes should be equal to the length of convstream_hidden_sizes + 1."
            )

        # 初始化模块参数
        self.config = config
        # 实例化卷积流模块
        self.convstream = VitMatteConvStream(config)
        # 记录卷积流模块输出通道数
        self.conv_chans = self.convstream.conv_chans

        # 创建融合块模块列表
        self.fusion_blocks = nn.ModuleList()
        # 计算融合通道列表
        self.fusion_channels = [config.hidden_size] + config.fusion_hidden_sizes

        # 遍历创建融合块模块
        for i in range(len(self.fusion_channels) - 1):
            # 添加融合块模块到列表
            self.fusion_blocks.append(
                VitMatteFusionBlock(
                    config=config,
                    in_channels=self.fusion_channels[i] + self.conv_chans[-(i + 1)],
                    out_channels=self.fusion_channels[i + 1],
                )
            )

        # 创建抠图头部模块
        self.matting_head = VitMatteHead(config)

    def forward(self, features, pixel_values):
        # 获取细节特征
        detail_features = self.convstream(pixel_values)
        # 遍历融合块模块，进行特征融合
        for i in range(len(self.fusion_blocks)):
            detailed_feature_map_name = "detailed_feature_map_" + str(len(self.fusion_blocks) - i - 1)
            # 将特征和细节特征输入融合块模块
            features = self.fusion_blocks[i](features, detail_features[detailed_feature_map_name])

        # 经过抠图头部模块，输出抠图结果
        alphas = torch.sigmoid(self.matting_head(features))

        return alphas


VITMATTE_START_DOCSTRING = r"""
    Parameters:
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.
        config ([`UperNetConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

VITMATTE_INPUTS_DOCSTRING = r"""



注释：
    # 参数说明：
    # pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`): 像素值。默认情况下将忽略填充。可以使用 [`AutoImageProcessor`] 来获取像素值。详情请参见 [`VitMatteImageProcessor.__call__`]。
    # output_attentions (`bool`, *optional*): 是否返回注意力层的注意力张量，如果backbone有的话。更多细节请参见返回的张量中的 `attentions`。
    # output_hidden_states (`bool`, *optional*): 是否返回backbone所有层的隐藏状态。更多细节请参见返回的张量中的 `hidden_states`。
    # return_dict (`bool`, *optional*): 是否返回一个 [`~utils.ModelOutput`] 而不是一个普通的元组。
# 定义 VitMatteForImageMatting 类，用于图像抠像的 ViTMatte 框架
@add_start_docstrings(
    """ViTMatte framework leveraging any vision backbone e.g. for ADE20k, CityScapes.""",
    VITMATTE_START_DOCSTRING,
)
class VitMatteForImageMatting(VitMattePreTrainedModel):
    # 初始化方法
    def __init__(self, config):
        # 调用父类的初始化方法
        super().__init__(config)
        # 保存配置参数
        self.config = config

        # 根据配置参数初始化视觉骨干网络
        self.backbone = AutoBackbone.from_config(config.backbone_config)
        # 初始化 VitMatteDetailCaptureModule 模块
        self.decoder = VitMatteDetailCaptureModule(config)

        # 初始化权重并应用最终处理
        self.post_init()

    # 定义 forward 方法
    @add_start_docstrings_to_model_forward(VITMATTE_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=ImageMattingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ):
        """
        labels (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional`):
            Ground truth image matting for computing the loss.
        给定的标签数据，形状为`(batch_size, height, width)`，用于计算损失

        Returns:
        返回值：

        Examples:
        例子：

        ```py
        >>> from transformers import VitMatteImageProcessor, VitMatteForImageMatting
        >>> import torch
        >>> from PIL import Image
        >>> from huggingface_hub import hf_hub_download

        >>> processor = VitMatteImageProcessor.from_pretrained("hustvl/vitmatte-small-composition-1k")
        >>> model = VitMatteForImageMatting.from_pretrained("hustvl/vitmatte-small-composition-1k")

        >>> filepath = hf_hub_download(
        ...     repo_id="hf-internal-testing/image-matting-fixtures", filename="image.png", repo_type="dataset"
        ... )
        >>> image = Image.open(filepath).convert("RGB")
        >>> filepath = hf_hub_download(
        ...     repo_id="hf-internal-testing/image-matting-fixtures", filename="trimap.png", repo_type="dataset"
        ... )
        >>> trimap = Image.open(filepath).convert("L")

        >>> # prepare image + trimap for the model
        >>> inputs = processor(images=image, trimaps=trimap, return_tensors="pt")

        >>> with torch.no_grad():
        ...     alphas = model(**inputs).alphas
        >>> print(alphas.shape)
        torch.Size([1, 1, 640, 960])
        ```"""
	返回字典，如果返回字典不为`None`则使用，否则使用类配置中的`use_return_dict`

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
	使用额外输出隐藏状态的掩模信息，如果掩模信息不为`None`则使用输出，否则使用类配置中的隐藏状态输出

        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
	如果额外输出掩模未定义则初始化为`None`，否则使用类配置中的掩模信息

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
	如果额外输出注意力未定义则初始化为`None`，否则使用类配置中的注意力输出

        outputs = self.backbone.forward_with_filtered_kwargs(
            pixel_values, output_hidden_states=output_hidden_states, output_attentions=output_attentions
        )
	使用过滤参数来执行前向计算，返回结果

        features = outputs.feature_maps[-1]
        使用上一层特征地图

        alphas = self.decoder(features, pixel_values)
	使用解码器处理特征地图和像素值，生成alpha通道图

        loss = None
        初始化损失为`None`

        if labels is not None:
            raise NotImplementedError("Training is not yet supported")
	如果存在标签数据，则抛出未实现错误

        if not return_dict:
            output = (alphas,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
	如果不返回字典，则输出数据

        return ImageMattingOutput(
            loss=loss,
            alphas=alphas,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
	返回ImageMattingOutput对象，包括损失、透明通道、隐藏状态和注意力
```