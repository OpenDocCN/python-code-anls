# `.\models\vitmatte\modeling_vitmatte.py`

```
# 设置文件编码为 UTF-8
# 版权声明及使用条款，详细说明使用限制和免责声明
# 此处定义了 PyTorch ViTMatte 模型

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

# 导入通用的模型预训练工具函数和类
from ...modeling_utils import PreTrainedModel
# 导入通用的工具函数，包括添加文档字符串等
from ...utils import (
    ModelOutput,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
# 导入特定的后端工具函数，加载背景模型
from ...utils.backbone_utils import load_backbone
# 导入 ViTMatte 模型的配置类
from .configuration_vitmatte import VitMatteConfig

# 定义预训练模型的列表
VITMATTE_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "hustvl/vitmatte-small-composition-1k",
    # 更多预训练模型列表详见 https://huggingface.co/models?filter=vitmatte
]

# 用于文档字符串的通用配置
_CONFIG_FOR_DOC = "VitMatteConfig"

@dataclass
class ImageMattingOutput(ModelOutput):
    """
    用于图像抠像模型输出的类。

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, 当 `labels` 被提供时返回):
            损失值.
        alphas (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
           估计的 alpha 通道值.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, 当 `output_hidden_states=True` 传递或当 `config.output_hidden_states=True` 时返回):
            由 `torch.FloatTensor` 组成的元组 (如果模型有嵌入层则为嵌入层的输出, 每个阶段的输出) 的形状为 `(batch_size, sequence_length, hidden_size)` 的隐藏状态
            (也称为特征映射)。
        attentions (`tuple(torch.FloatTensor)`, *optional*, 当 `output_attentions=True` 传递或当 `config.output_attentions=True` 时返回):
            由 `torch.FloatTensor` 组成的元组 (每个层一个) 的形状为 `(batch_size, num_heads, patch_size, sequence_length)` 的注意力权重。
            在注意力 softmax 后用于计算自注意力头中的加权平均值。

            注意力权重，在注意力 softmax 后用于计算自注意力头中的加权平均值。
    """

    loss: Optional[torch.FloatTensor] = None
    alphas: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class VitMattePreTrainedModel(PreTrainedModel):
    """
    一个抽象类，处理权重初始化和下载/加载预训练模型的简单接口。
    """

    # 配置类为 VitMatteConfig
    config_class = VitMatteConfig
    # 定义主要输入名称为 "pixel_values"
    main_input_name = "pixel_values"

    # 启用梯度检查点支持，设置为 True
    supports_gradient_checkpointing = True

    # 定义初始化权重函数 _init_weights，接受一个模块作为参数
    def _init_weights(self, module):
        # 如果传入的模块是 nn.Conv2d 类型
        if isinstance(module, nn.Conv2d):
            # 使用正态分布初始化该卷积层的权重，均值为 0，标准差为 self.config.initializer_range
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            # 如果模块有偏置项
            if module.bias is not None:
                # 将偏置项数据初始化为零
                module.bias.data.zero_()
class VitMatteBasicConv3x3(nn.Module):
    """
    Basic convolution layers including: Conv3x3, BatchNorm2d, ReLU layers.
    """

    def __init__(self, config, in_channels, out_channels, stride=2, padding=1):
        super().__init__()
        # 定义一个3x3卷积层，设置输入通道数、输出通道数、卷积核大小、步长和填充，不使用偏置
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=padding,
            bias=False,
        )
        # 批标准化层，设置输出通道数和epsilon值（用于数值稳定性）
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=config.batch_norm_eps)
        # ReLU激活函数
        self.relu = nn.ReLU()

    def forward(self, hidden_state):
        # 执行卷积操作
        hidden_state = self.conv(hidden_state)
        # 执行批标准化操作
        hidden_state = self.batch_norm(hidden_state)
        # 执行ReLU激活函数操作
        hidden_state = self.relu(hidden_state)

        return hidden_state


class VitMatteConvStream(nn.Module):
    """
    Simple ConvStream containing a series of basic conv3x3 layers to extract detail features.
    """

    def __init__(self, config):
        super().__init__()

        # 获取输入通道数
        in_channels = config.backbone_config.num_channels
        # 获取卷积层隐藏层尺寸列表
        out_channels = config.convstream_hidden_sizes

        self.convs = nn.ModuleList()
        self.conv_chans = [in_channels] + out_channels

        # 根据隐藏层尺寸列表创建一系列VitMatteBasicConv3x3实例
        for i in range(len(self.conv_chans) - 1):
            in_chan_ = self.conv_chans[i]
            out_chan_ = self.conv_chans[i + 1]
            self.convs.append(VitMatteBasicConv3x3(config, in_chan_, out_chan_))

    def forward(self, pixel_values):
        out_dict = {"detailed_feature_map_0": pixel_values}
        embeddings = pixel_values
        # 遍历并应用所有卷积层，将每个输出保存到字典中
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
        super().__init__()
        # 使用VitMatteBasicConv3x3创建一个融合块，设置输入通道数、输出通道数、步长和填充
        self.conv = VitMatteBasicConv3x3(config, in_channels, out_channels, stride=1, padding=1)

    def forward(self, features, detailed_feature_map):
        # 对特征进行上采样
        upscaled_features = nn.functional.interpolate(features, scale_factor=2, mode="bilinear", align_corners=False)
        # 拼接详细特征图和上采样特征
        out = torch.cat([detailed_feature_map, upscaled_features], dim=1)
        # 执行卷积操作
        out = self.conv(out)

        return out


class VitMatteHead(nn.Module):
    """
    Simple Matting Head, containing only conv3x3 and conv1x1 layers.
    """

    def __init__(self, config):
        super().__init__()

        # 获取融合块隐藏层尺寸列表的最后一个值作为输入通道数
        in_channels = config.fusion_hidden_sizes[-1]
        # 设置中间通道数为16
        mid_channels = 16

        # 创建一个简单的卷积网络序列，包含一个3x3卷积层、批标准化层和ReLU激活函数
        self.matting_convs = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(True),
            nn.Conv2d(mid_channels, 1, kernel_size=1, stride=1, padding=0),
        )
    # 定义一个方法用于正向传播，接收隐藏状态作为输入参数
    def forward(self, hidden_state):
        # 使用类内部定义的 matting_convs 层对输入的隐藏状态进行变换处理
        hidden_state = self.matting_convs(hidden_state)
        
        # 方法的返回值为经过变换后的隐藏状态
        return hidden_state
    Parameters:
    This model is a PyTorch `torch.nn.Module` sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.
        config ([`UperNetConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            # 输入的像素数值。默认情况下将忽略填充。可以使用 [`AutoImageProcessor`] 获得像素值。详见 [`VitMatteImageProcessor.__call__`]。
        output_attentions (`bool`, *optional*):
            # 是否返回所有注意力层的注意力张量，如果后端有的话。返回的张量中的 `attentions` 字段包含更多细节。
        output_hidden_states (`bool`, *optional*):
            # 是否返回后端所有层的隐藏状态。返回的张量中的 `hidden_states` 字段包含更多细节。
        return_dict (`bool`, *optional*):
            # 是否返回 [`~utils.ModelOutput`] 而不是普通元组。
"""
@add_start_docstrings(
    """ViTMatte framework leveraging any vision backbone e.g. for ADE20k, CityScapes.""",
    VITMATTE_START_DOCSTRING,
)
"""
# 使用装饰器添加类的文档字符串，描述了 ViTMatte 框架如何利用视觉骨干网络（如 ADE20k、CityScapes）进行图像抠图。

class VitMatteForImageMatting(VitMattePreTrainedModel):
    """
    派生自 VitMattePreTrainedModel 的图像抠图模型类。
    """

    def __init__(self, config):
        """
        初始化方法。

        Args:
            config (PretrainedConfig): 模型的配置对象。

        """
        super().__init__(config)
        self.config = config

        # 载入指定的视觉骨干网络
        self.backbone = load_backbone(config)
        
        # 初始化 VitMatteDetailCaptureModule 模块
        self.decoder = VitMatteDetailCaptureModule(config)

        # 初始化权重并进行最终处理
        self.post_init()

    @add_start_docstrings_to_model_forward(VITMATTE_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=ImageMattingOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
        """
        正向传播方法。

        Args:
            pixel_values (torch.Tensor, optional): 输入像素值张量。默认为 None。
            output_attentions (bool, optional): 是否输出注意力权重。默认为 None。
            output_hidden_states (bool, optional): 是否输出隐藏状态。默认为 None。
            labels (torch.Tensor, optional): 标签张量。默认为 None。
            return_dict (bool, optional): 是否返回字典格式结果。默认为 None。

        Returns:
            依据配置返回的输出类型，通常为 ImageMattingOutput 对象。

        """
        """
        labels (`torch.LongTensor` of shape `(batch_size, height, width)`, *optional*):
            Ground truth image matting for computing the loss.

        Returns:
            Returns either a tuple of tensors or an `ImageMattingOutput` object containing loss, alphas,
            hidden states, and attentions.

        Examples:

        ```python
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
        ```

        """
        # If return_dict is not provided, use the default from model configuration
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # If output_hidden_states is not provided, use the default from model configuration
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # If output_attentions is not provided, use the default from model configuration
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions

        # Forward pass through the backbone with specified arguments
        outputs = self.backbone.forward_with_filtered_kwargs(
            pixel_values, output_hidden_states=output_hidden_states, output_attentions=output_attentions
        )

        # Retrieve the feature maps from the outputs
        features = outputs.feature_maps[-1]

        # Generate alphas using the decoder with the extracted features and pixel values
        alphas = self.decoder(features, pixel_values)

        # Initialize loss variable
        loss = None

        # If labels are provided, raise NotImplementedError since training is not supported
        if labels is not None:
            raise NotImplementedError("Training is not yet supported")

        # If return_dict is False, return a tuple including alphas and other outputs
        if not return_dict:
            output = (alphas,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        # If return_dict is True, return an `ImageMattingOutput` object containing all relevant outputs
        return ImageMattingOutput(
            loss=loss,
            alphas=alphas,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
```