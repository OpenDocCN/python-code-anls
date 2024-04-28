# `.\transformers\models\upernet\modeling_upernet.py`

```
# 设置文件编码为 UTF-8
# 版权声明
# 根据 Apache 2.0 许可证的规定，对本文件的使用受到限制
# 可以在以下网址获取许可证的副本 http://www.apache.org/licenses/LICENSE-2.0
# 除非适用法律要求或经书面同意，不得使用此文件
# 根据许可证的规定，软件按"原样"分发，不附带任何形式的担保或条件，无论明示或暗示
# 请参阅许可证了解具体语言规定权限和限制
# PyTorch UperNet 模型. 基于 OpenMMLab 的实现，链接 https://github.com/open-mmlab/mmsegmentation.

from typing import List, Optional, Tuple, Union # 引入类型提示的相关模块

import torch # 引入 torch 库
from torch import nn  # 引入 torch.nn 模块
from torch.nn import CrossEntropyLoss  # 引入交叉熵损失模块

from ... import AutoBackbone # 从根目录下的 ... 文件夹中引入 AutoBackbone
from ...modeling_outputs import SemanticSegmenterOutput  # 从 ... 文件夹中引入 SemanticSegmenterOutput
from ...modeling_utils import PreTrainedModel  # 从 ... 文件夹中引入 PreTrainedModel
from ...utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings  # 从 ... 文件夹中引入相关工具函数
from .configuration_upernet import UperNetConfig  # 从当前目录下的 configuration_upernet 文件中引入 UperNetConfig 模块

# 可用的预训练 UperNet 模型列表
UPERNET_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "openmmlab/upernet-convnext-tiny",
    # 查看所有 UperNet 模型 https://huggingface.co/models?filter=upernet
]

# 通用文档字符串
_CONFIG_FOR_DOC = "UperNetConfig"


class UperNetConvModule(nn.Module):
    """
    A convolutional block that bundles conv/norm/activation layers. This block simplifies the usage of convolution
    layers, which are commonly used with a norm layer (e.g., BatchNorm) and activation layer (e.g., ReLU).
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        padding: Union[int, Tuple[int, int], str] = 0,
        bias: bool = False,
        dilation: Union[int, Tuple[int, int]] = 1,
    ) -> None:
        super().__init__()
        # 创建一个卷积层
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
            dilation=dilation,
        )
        # 创建一个批量归一化层
        self.batch_norm = nn.BatchNorm2d(out_channels)
        # 创建一个激活函数层
        self.activation = nn.ReLU()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # 应用卷积层
        output = self.conv(input)
        # 应用批量归一化层
        output = self.batch_norm(output)
        # 应用激活函数层
        output = self.activation(output)

        return output


class UperNetPyramidPoolingBlock(nn.Module):
    def __init__(self, pool_scale: int, in_channels: int, channels: int) -> None:
        super().__init__()
        # 创建自适应平均池化层
        # 创建 UperNetConvModule 实例
        self.layers = [
            nn.AdaptiveAvgPool2d(pool_scale),
            UperNetConvModule(in_channels, channels, kernel_size=1),
        ]
        for i, layer in enumerate(self.layers):
            # 将创建的层作为该模块的子模块添加
            self.add_module(str(i), layer)
    # 定义前向传播函数，输入为torch.Tensor类型，输出为torch.Tensor类型
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # 将输入赋值给隐藏状态
        hidden_state = input
        # 遍历每一层神经网络
        for layer in self.layers:
            # 调用每一层的前向传播函数，将隐藏状态作为输入
            hidden_state = layer(hidden_state)
        # 返回最终的隐藏状态
        return hidden_state
class UperNetPyramidPoolingModule(nn.Module):
    """
    Pyramid Pooling Module (PPM) used in PSPNet.

    Args:
        pool_scales (`Tuple[int]`):
            Pooling scales used in Pooling Pyramid Module.
        in_channels (`int`):
            Input channels.
        channels (`int`):
            Channels after modules, before conv_seg.
        align_corners (`bool`):
            align_corners argument of F.interpolate.
    """

    # 初始化Pyramid Pooling Module类
    def __init__(self, pool_scales: Tuple[int, ...], in_channels: int, channels: int, align_corners: bool) -> None:
        # 调用父类的初始化方法
        super().__init__()
        # 设定池化尺度，对齐角落等参数
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        self.blocks = []
        # 遍历每个池化尺度，创建对应的Pyramid Pooling Block
        for i, pool_scale in enumerate(pool_scales):
            block = UperNetPyramidPoolingBlock(pool_scale=pool_scale, in_channels=in_channels, channels=channels)
            self.blocks.append(block)
            self.add_module(str(i), block)

    # 前向传播方法
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        ppm_outs = []
        # 遍历每个Pyramid Pooling Block
        for ppm in self.blocks:
            # 获取PPM的输出
            ppm_out = ppm(x)
            # 上采样PPM输出
            upsampled_ppm_out = nn.functional.interpolate(
                ppm_out, size=x.size()[2:], mode="bilinear", align_corners=self.align_corners
            )
            ppm_outs.append(upsampled_ppm_out)
        # 返回所有PPM输出
        return ppm_outs


class UperNetHead(nn.Module):
    """
    Unified Perceptual Parsing for Scene Understanding. This head is the implementation of
    [UPerNet](https://arxiv.org/abs/1807.10221).
    """
```  
    # 定义一个 PyTorch 模型的初始化方法
    def __init__(self, config, in_channels):
        # 调用父类的初始化方法
        super().__init__()
        
        # 将配置信息保存到对象属性中
        self.config = config
        # 获取池化操作的尺度大小
        self.pool_scales = config.pool_scales  
        # 保存输入通道数
        self.in_channels = in_channels
        # 设置隐藏层通道数
        self.channels = config.hidden_size
        # 设置插值对齐方式为False
        self.align_corners = False
        # 定义一个卷积层作为分类器，输出通道数为类别数
        self.classifier = nn.Conv2d(self.channels, config.num_labels, kernel_size=1)
    
        # 创建金字塔池化模块
        self.psp_modules = UperNetPyramidPoolingModule(
            self.pool_scales,
            self.in_channels[-1],
            self.channels,
            align_corners=self.align_corners,
        )
        # 创建bottleneck模块
        self.bottleneck = UperNetConvModule(
            self.in_channels[-1] + len(self.pool_scales) * self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
        )
    
        # 创建特征金字塔网络的模块
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # 除最后一层外的所有层
            # 创建lateral卷积层
            l_conv = UperNetConvModule(in_channels, self.channels, kernel_size=1)
            # 创建FPN卷积层
            fpn_conv = UperNetConvModule(self.channels, self.channels, kernel_size=3, padding=1)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)
    
        # 创建FPN bottleneck模块
        self.fpn_bottleneck = UperNetConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            kernel_size=3,
            padding=1,
        )
    
    # 初始化模型权重的方法
    def init_weights(self):
        self.apply(self._init_weights)
    
    # 具体的权重初始化方法
    def _init_weights(self, module):
        if isinstance(module, nn.Conv2d):
            # 使用正态分布初始化卷积层权重
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                # 将偏置参数初始化为0
                module.bias.data.zero_()
    
    # 定义PSP模块的前向传播过程    
    def psp_forward(self, inputs):
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)
        return output
    # 前向传播函数，接受编码器隐藏状态作为输入，返回张量
    def forward(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:
        # 构建横向连接
        laterals = [lateral_conv(encoder_hidden_states[i]) for i, lateral_conv in enumerate(self.lateral_convs)]
        
        # 将 PSP 模块的输出添加到横向连接中
        laterals.append(self.psp_forward(encoder_hidden_states))
        
        # 构建自顶向下路径
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + nn.functional.interpolate(
                laterals[i], size=prev_shape, mode="bilinear", align_corners=self.align_corners
            )
        
        # 构建输出
        fpn_outs = [self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels - 1)]
        # 添加 PSP 特征
        fpn_outs.append(laterals[-1])
        
        # 对于自顶向下路径上的每个级别，将其上采样到与第一个级别相同的大小
        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = nn.functional.interpolate(
                fpn_outs[i], size=fpn_outs[0].shape[2:], mode="bilinear", align_corners=self.align_corners
            )
        
        # 将各个级别的特征拼接到一起
        fpn_outs = torch.cat(fpn_outs, dim=1)
        
        # 过 FPN 瓶颈层
        output = self.fpn_bottleneck(fpn_outs)
        
        # 使用分类器输出最终结果
        output = self.classifier(output)
        
        # 返回输出结果
        return output
class UperNetFCNHead(nn.Module):
    """
    Fully Convolution Networks for Semantic Segmentation. This head is the implementation of
    [FCNNet](https://arxiv.org/abs/1411.4038>).

    Args:
        config:
            Configuration.
        in_channels (int):
            Number of input channels.
        kernel_size (int):
            The kernel size for convs in the head. Default: 3.
        dilation (int):
            The dilation rate for convs in the head. Default: 1.
    """

    def __init__(
        self, config, in_index: int = 2, kernel_size: int = 3, dilation: Union[int, Tuple[int, int]] = 1
    ) -> None:
        super().__init__()

        self.config = config  # 保存传入的配置
        self.in_channels = config.auxiliary_in_channels  # 从配置中获取辅助输入的通道数
        self.channels = config.auxiliary_channels  # 从配置中获取辅助通道数
        self.num_convs = config.auxiliary_num_convs  # 从配置中获取卷积层数
        self.concat_input = config.auxiliary_concat_input  # 从配置中获取是否要将输入和卷积输出拼接
        self.in_index = in_index  # 保存传入的输入索引

        conv_padding = (kernel_size // 2) * dilation  # 计算卷积填充大小
        convs = []  # 创建空的卷积列表
        convs.append(  # 添加第一个卷积模块
            UperNetConvModule(  # 使用 UperNetConvModule 创建卷积模块
                self.in_channels, self.channels, kernel_size=kernel_size, padding=conv_padding, dilation=dilation
            )
        )
        for i in range(self.num_convs - 1):  # 根据卷积层数循环
            convs.append(  # 添加其他卷积模块
                UperNetConvModule(  # 使用 UperNetConvModule 创建卷积模块
                    self.channels, self.channels, kernel_size=kernel_size, padding=conv_padding, dilation=dilation
                )
            )
        if self.num_convs == 0:  # 如果卷积层数为0
            self.convs = nn.Identity()  # 使用 nn.Identity() 对象
        else:
            self.convs = nn.Sequential(*convs)  # 将卷积模块列表转换为序列
        if self.concat_input:  # 如果需要拼接输入和卷积输出
            self.conv_cat = UperNetConvModule(  # 使用 UperNetConvModule 创建卷积模块
                self.in_channels + self.channels, self.channels, kernel_size=kernel_size, padding=kernel_size // 2
            )

        self.classifier = nn.Conv2d(self.channels, config.num_labels, kernel_size=1)  # 创建输出分类器

    def init_weights(self):  # 初始化权重的方法
        self.apply(self._init_weights)  # 应用权重初始化方法

    def _init_weights(self, module):  # 初始化权重的私有方法
        if isinstance(module, nn.Conv2d):  # 如果是卷��层
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)  # 使用正态分布初始化权重
            if module.bias is not None:  # 如果有偏置项
                module.bias.data.zero_()  # 将偏置项初始化为0

    def forward(self, encoder_hidden_states: torch.Tensor) -> torch.Tensor:  # 正向传播方法
        # just take the relevant feature maps
        hidden_states = encoder_hidden_states[self.in_index]  # 从编码器隐藏状态中取出相关的特征图
        output = self.convs(hidden_states)  # 将特征图通过卷积层
        if self.concat_input:  # 如果需要拼接输入和卷积输出
            output = self.conv_cat(torch.cat([hidden_states, output], dim=1))  # 将输入和卷积输出拼接后通过另一卷积层
        output = self.classifier(output)  # 将拼接后的特征图通过分类器
        return output  # 返回输出的特征图


class UperNetPreTrainedModel(PreTrainedModel):  # UperNetPreTrainedModel 类继承自 PreTrainedModel 类
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = UperNetConfig  # 静态变量，保存配置类
    main_input_name = "pixel_values"  # 静态变量，保存主输入名称
    # 初始化权重函数，接受一个模块参数
    def _init_weights(self, module):
        # 如果模块是 UperNetPreTrainedModel 的实例
        if isinstance(module, UperNetPreTrainedModel):
            # 初始化模块的骨干网络权重
            module.backbone.init_weights()
            # 初始化模块的解码头权重
            module.decode_head.init_weights()
            # 如果模块有辅助头部，则初始化辅助头部权重
            if module.auxiliary_head is not None:
                module.auxiliary_head.init_weights()
    
    # 初始化权重函数
    def init_weights(self):
        # 初始化骨干网络权重
        self.backbone.init_weights()
        # 初始化解码头权重
        self.decode_head.init_weights()
        # 如果存在辅助头部，则初始化辅助头部权重
        if self.auxiliary_head is not None:
            self.auxiliary_head.init_weights()
UPERNET_START_DOCSTRING = r"""
    Parameters:
    This model is a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) sub-class. Use
    it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage and
    behavior.
        config ([`UperNetConfig`]): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""

UPERNET_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)`):
            Pixel values. Padding will be ignored by default should you provide it. Pixel values can be obtained using
            [`AutoImageProcessor`]. See [`SegformerImageProcessor.__call__`] for details.
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers in case the backbone has them. See
            `attentions` under returned tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers of the backbone. See `hidden_states` under
            returned tensors for more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    """UperNet framework leveraging any vision backbone e.g. for ADE20k, CityScapes.""",
    UPERNET_START_DOCSTRING,
)
class UperNetForSemanticSegmentation(UperNetPreTrainedModel):
    # Initialize function for the UperNetForSemanticSegmentation class
    def __init__(self, config):
        # Call the __init__ function of the parent class to initialize the model
        super().__init__(config)

        # Instantiate the backbone using the AutoBackbone class and the config's backbone_config
        self.backbone = AutoBackbone.from_config(config.backbone_config)

        # Semantic segmentation head(s)
        # Instantiate the decode head using the UperNetHead class and the backbone's channels
        self.decode_head = UperNetHead(config, in_channels=self.backbone.channels)
        # Instantiate the auxiliary head using the UperNetFCNHead class if config.use_auxiliary_head is True
        self.auxiliary_head = UperNetFCNHead(config) if config.use_auxiliary_head else None

        # Initialize weights and apply final processing
        # Call the post_init method to finalize the initialization
        self.post_init()

    # Forward function for the UperNetForSemanticSegmentation class
    @add_start_docstrings_to_model_forward(UPERNET_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @replace_return_docstrings(output_type=SemanticSegmenterOutput, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
```