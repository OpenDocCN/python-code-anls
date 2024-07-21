# `.\pytorch\benchmarks\functional_autograd_benchmark\torchvision_models.py`

```py
# 从 https://github.com/pytorch/vision 中引入代码
# 以便不需要安装 torchvision
from collections import OrderedDict

import torch
from torch import nn

from torch.jit.annotations import Dict
from torch.nn import functional as F

try:
    # 尝试导入 scipy 中的 linear_sum_assignment 函数
    from scipy.optimize import linear_sum_assignment

    # 设置 scipy_available 标志为 True，表示 scipy 可用
    scipy_available = True
except Exception:
    # 如果导入失败，则将 scipy_available 标志设置为 False
    scipy_available = False


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 卷积操作，带有填充"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 卷积操作"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            # 如果 groups 不为 1 或 base_width 不为 64，则抛出 ValueError 异常
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            # 如果 dilation 大于 1，则抛出 NotImplementedError 异常
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # 当 stride != 1 时，self.conv1 和 self.downsample 层都对输入进行降采样
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # 保存输入 x 作为恒等映射 identity
        identity = x

        # 第一个卷积层及其后续操作：卷积、批归一化、ReLU
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 第二个卷积层及其后续操作：卷积、批归一化
        out = self.conv2(out)
        out = self.bn2(out)

        # 如果存在 downsample 层，则对输入 x 进行降采样
        if self.downsample is not None:
            identity = self.downsample(x)

        # 将降采样后的 identity 与处理后的 out 相加
        out += identity
        out = self.relu(out)

        # 返回处理结果
        return out


class Bottleneck(nn.Module):
    # 在 torchvision 中，Bottleneck 类将降采样的步骤放在 3x3 卷积 (self.conv2) 中
    # 原始实现根据 "Deep residual learning for image recognition" 将步幅放在 1x1 卷积 (self.conv1) 中
    # 这种变体也称为 ResNet V1.5，并根据 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch 提高了准确性

    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super().__init__()  # 调用父类的初始化方法，确保继承关系正确
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d  # 如果规范化层未指定，则默认使用 BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups  # 计算卷积核的宽度
        # 下面的 self.conv2 和 self.downsample 层在 stride != 1 时会对输入进行下采样
        self.conv1 = conv1x1(inplanes, width)  # 创建 1x1 卷积层 self.conv1
        self.bn1 = norm_layer(width)  # 使用规范化层对 self.conv1 的输出进行规范化
        self.conv2 = conv3x3(width, width, stride, groups, dilation)  # 创建 3x3 卷积层 self.conv2
        self.bn2 = norm_layer(width)  # 使用规范化层对 self.conv2 的输出进行规范化
        self.conv3 = conv1x1(width, planes * self.expansion)  # 创建 1x1 卷积层 self.conv3
        self.bn3 = norm_layer(planes * self.expansion)  # 使用规范化层对 self.conv3 的输出进行规范化
        self.relu = nn.ReLU(inplace=True)  # 使用 ReLU 激活函数，并指定 inplace=True
        self.downsample = downsample  # 下采样函数，用于处理输入维度不匹配的情况
        self.stride = stride  # 设置步长

    def forward(self, x):
        identity = x  # 将输入赋值给 identity，用于残差连接

        out = self.conv1(x)  # 执行 self.conv1 卷积操作
        out = self.bn1(out)  # 对 self.conv1 输出进行规范化
        out = self.relu(out)  # 使用 ReLU 激活函数

        out = self.conv2(out)  # 执行 self.conv2 卷积操作
        out = self.bn2(out)  # 对 self.conv2 输出进行规范化
        out = self.relu(out)  # 使用 ReLU 激活函数

        out = self.conv3(out)  # 执行 self.conv3 卷积操作
        out = self.bn3(out)  # 对 self.conv3 输出进行规范化

        if self.downsample is not None:
            identity = self.downsample(x)  # 如果 downsample 不为 None，则对输入进行下采样

        out += identity  # 执行残差连接
        out = self.relu(out)  # 最终输出应用 ReLU 激活函数

        return out  # 返回输出张量
class ResNet(nn.Module):
    # 定义 ResNet 模型类，继承自 nn.Module
    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):
        # 初始化函数
        super().__init__()
        # 如果没有指定 norm_layer，则默认使用 nn.BatchNorm2d
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        # 初始通道数设置为 64
        self.inplanes = 64
        self.dilation = 1
        # 如果没有指定是否使用 dilated 卷积来替换步长为2的卷积，则默认不替换
        if replace_stride_with_dilation is None:
            # replace_stride_with_dilation 是一个长度为 3 的布尔型列表，表示是否替换每个阶段的步长
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            # 如果列表长度不为 3，则抛出异常
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        # 设置组数和每组的基础宽度
        self.groups = groups
        self.base_width = width_per_group
        # 第一个卷积层，输入通道为 3，输出通道为 self.inplanes，卷积核大小为 7x7，步长为 2，填充为 3，无偏置
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        # 第一个批标准化层
        self.bn1 = norm_layer(self.inplanes)
        # ReLU 激活函数，inplace=True 表示进行原地操作
        self.relu = nn.ReLU(inplace=True)
        # 最大池化层，池化核大小为 3x3，步长为 2，填充为 1
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # 构建 ResNet 的各个层
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        # 自适应平均池化层，输出大小为 1x1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 全连接层，输出维度为 num_classes
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 初始化网络中的权重参数
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 使用 Kaiming 初始化卷积层的权重
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                # 将批标准化层和组归一化层的权重初始化为常数 1，偏置初始化为 0
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # 如果指定了 zero_init_residual，则将最后一个残差分支中的 BatchNorm 权重初始化为 0
        # 这样可以让每个残差块的行为更接近于恒等映射，从而改善模型的性能
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
    # 创建网络的一个层，包括多个残差块
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        # 获取当前使用的规范化层
        norm_layer = self._norm_layer
        # 下采样设为 None
        downsample = None
        # 记录先前的扩张率
        previous_dilation = self.dilation
        # 如果需要扩张，则更新当前的扩张率并将步长设为 1
        if dilate:
            self.dilation *= stride
            stride = 1
        # 如果步长不为 1 或者输入平面数不等于输出平面数乘以块的扩展系数，则进行下采样设置
        if stride != 1 or self.inplanes != planes * block.expansion:
            # 创建一个包含 1x1 卷积和规范化层的顺序容器，作为下采样操作
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        # 创建存储各层的列表
        layers = []
        # 将第一个残差块加入列表
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        # 更新输入平面数
        self.inplanes = planes * block.expansion
        # 循环创建其余的残差块并加入列表
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        # 返回一个包含所有残差块的顺序容器作为一个层
        return nn.Sequential(*layers)

    # 网络的正向传播方法
    def _forward_impl(self, x):
        # 卷积层 1
        x = self.conv1(x)
        # 批规范化层 1
        x = self.bn1(x)
        # ReLU 激活函数
        x = self.relu(x)
        # 最大池化层
        x = self.maxpool(x)

        # 第一层残差块
        x = self.layer1(x)
        # 第二层残差块
        x = self.layer2(x)
        # 第三层残差块
        x = self.layer3(x)
        # 第四层残差块
        x = self.layer4(x)

        # 平均池化层
        x = self.avgpool(x)
        # 将特征张量展平
        x = torch.flatten(x, 1)
        # 全连接层
        x = self.fc(x)

        # 返回输出结果
        return x

    # 定义正向传播接口，调用 _forward_impl 方法
    def forward(self, x):
        return self._forward_impl(x)
# 定义一个函数 _resnet，用于构建 ResNet 模型
def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    # 创建 ResNet 模型实例
    model = ResNet(block, layers, **kwargs)
    
    # 如果 pretrained 参数为 True，则从指定 URL 加载预训练模型的状态字典
    # 并将状态字典加载到模型中
    # state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
    # model.load_state_dict(state_dict)
    
    # 返回创建的 ResNet 模型实例
    return model


# 定义一个函数 resnet18，返回一个 ResNet-18 模型实例
def resnet18(pretrained=False, progress=True, **kwargs):
    """
    ResNet-18 模型，根据指定参数返回预训练或自定义配置的模型实例。
    Args:
        pretrained (bool): 如果为 True，则返回在 ImageNet 上预训练的模型
        progress (bool): 如果为 True，则在 stderr 上显示下载进度条
    """
    # 调用 _resnet 函数，构建 ResNet-18 模型实例并返回
    return _resnet("resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


# 定义一个函数 resnet50，返回一个 ResNet-50 模型实例
def resnet50(pretrained=False, progress=True, **kwargs):
    """
    ResNet-50 模型，根据指定参数返回预训练或自定义配置的模型实例。
    Args:
        pretrained (bool): 如果为 True，则返回在 ImageNet 上预训练的模型
        progress (bool): 如果为 True，则在 stderr 上显示下载进度条
    """
    # 调用 _resnet 函数，构建 ResNet-50 模型实例并返回
    return _resnet("resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


# 定义一个继承自 nn.ModuleDict 的类 IntermediateLayerGetter
class IntermediateLayerGetter(nn.ModuleDict):
    """
    IntermediateLayerGetter 是一个模块包装器，从模型中提取中间层。
    假设模块已按照使用顺序注册到模型中。
    这意味着在 forward 方法中如果重复使用相同的 nn.Module，会导致该功能失效。
    此外，它只能查询直接分配给模型的子模块。
    Args:
        model (nn.Module): 我们要从中提取特征的模型
        return_layers (Dict[name, new_name]): 一个字典，包含要返回激活值的模块名称作为键，
            和用户指定的返回激活值名称作为值。
    Examples::
        >>> m = torchvision.models.resnet18(pretrained=True)
        >>> # 提取 layer1 和 layer3 的激活值，并分别命名为 'feat1' 和 'feat2'
        >>> new_m = torchvision.models._utils.IntermediateLayerGetter(m,
        >>>     {'layer1': 'feat1', 'layer3': 'feat2'})
        >>> out = new_m(torch.rand(1, 3, 224, 224))
        >>> print([(k, v.shape) for k, v in out.items()])
        >>>     [('feat1', torch.Size([1, 64, 56, 56])),
        >>>      ('feat2', torch.Size([1, 256, 14, 14]))]
    """

    _version = 2
    __annotations__ = {
        "return_layers": Dict[str, str],
    }
    # 初始化函数，接受一个模型和返回层字典作为参数
    def __init__(self, model, return_layers):
        # 检查传入的返回层是否都存在于模型的子模块中
        if not set(return_layers).issubset(
            [name for name, _ in model.named_children()]
        ):
            # 如果不是，则抛出数值错误异常
            raise ValueError("return_layers are not present in model")
        
        # 保存原始的返回层字典
        orig_return_layers = return_layers
        
        # 将返回层字典的键和值都转换为字符串类型
        return_layers = {str(k): str(v) for k, v in return_layers.items()}
        
        # 创建一个有序字典用于存储模型的子模块
        layers = OrderedDict()
        
        # 遍历模型的子模块
        for name, module in model.named_children():
            # 将子模块添加到layers字典中
            layers[name] = module
            
            # 如果子模块的名称在返回层字典中
            if name in return_layers:
                # 则从返回层字典中删除该名称对应的条目
                del return_layers[name]
            
            # 如果返回层字典为空
            if not return_layers:
                # 停止遍历
                break
        
        # 调用父类的初始化方法，将有序字典传递给父类
        super().__init__(layers)
        
        # 保存原始的返回层字典到对象的属性中
        self.return_layers = orig_return_layers

    # 前向传播函数，接受输入张量x作为参数
    def forward(self, x):
        # 创建一个有序字典用于存储各层的输出
        out = OrderedDict()
        
        # 遍历自定义层的子模块
        for name, module in self.items():
            # 对输入张量进行前向传播计算
            x = module(x)
            
            # 如果子模块的名称在返回层字典中
            if name in self.return_layers:
                # 获取对应的输出名称
                out_name = self.return_layers[name]
                
                # 将计算得到的输出添加到输出字典中
                out[out_name] = x
        
        # 返回所有层中指定层的输出字典
        return out
class _SimpleSegmentationModel(nn.Module):
    __constants__ = ["aux_classifier"]

    def __init__(self, backbone, classifier, aux_classifier=None):
        super().__init__()
        self.backbone = backbone  # 初始化时保存输入的背景模型
        self.classifier = classifier  # 初始化时保存输入的分类器模型
        self.aux_classifier = aux_classifier  # 初始化时保存输入的辅助分类器模型（可选）

    def forward(self, x):
        input_shape = x.shape[-2:]  # 获取输入张量的空间维度信息

        # 调用背景模型提取特征
        features = self.backbone(x)

        result = OrderedDict()  # 创建有序字典用于存储结果
        x = features["out"]  # 获取主要输出特征
        x = self.classifier(x)  # 使用分类器对主要输出进行分类
        x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)  # 对分类结果进行双线性插值
        result["out"] = x  # 将处理后的主要输出存入结果字典

        if self.aux_classifier is not None:
            x = features["aux"]  # 如果存在辅助分类器，则获取辅助输出特征
            x = self.aux_classifier(x)  # 使用辅助分类器对辅助输出进行分类
            x = F.interpolate(x, size=input_shape, mode="bilinear", align_corners=False)  # 对辅助分类结果进行双线性插值
            result["aux"] = x  # 将处理后的辅助输出存入结果字典

        return result  # 返回最终结果字典


class FCN(_SimpleSegmentationModel):
    """
    Implements a Fully-Convolutional Network for semantic segmentation.
    Args:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    pass  # FCN 类暂时没有额外的定义，继承自 _SimpleSegmentationModel


class FCNHead(nn.Sequential):
    def __init__(self, in_channels, channels):
        inter_channels = in_channels // 4  # 计算中间通道数量
        layers = [
            nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),  # 添加卷积层
            nn.BatchNorm2d(inter_channels),  # 批标准化层
            nn.ReLU(),  # ReLU 激活函数
            nn.Dropout(0.1),  # 随机失活层
            nn.Conv2d(inter_channels, channels, 1),  # 添加卷积层
        ]

        super().__init__(*layers)  # 调用父类构造函数


def _segm_resnet(name, backbone_name, num_classes, aux, pretrained_backbone=True):
    # backbone = resnet.__dict__[backbone_name](
    #     pretrained=pretrained_backbone,
    #     replace_stride_with_dilation=[False, True, True])
    # 固定使用 resnet50
    assert backbone_name == "resnet50"  # 断言确保使用的是 resnet50

    # 使用 resnet50 构建背景模型
    backbone = resnet50(
        pretrained=pretrained_backbone, replace_stride_with_dilation=[False, True, True]
    )

    return_layers = {"layer4": "out"}
    if aux:
        return_layers["layer3"] = "aux"  # 如果使用辅助分类器，则添加 layer3 到返回层

    # 使用 IntermediateLayerGetter 获取指定层的输出
    backbone = IntermediateLayerGetter(backbone, return_layers=return_layers)

    aux_classifier = None
    if aux:
        inplanes = 1024
        aux_classifier = FCNHead(inplanes, num_classes)  # 使用 FCNHead 构建辅助分类器

    model_map = {
        "fcn": (FCNHead, FCN),  # 定义模型映射表中的 FCN 类型
    }
    inplanes = 2048
    classifier = model_map[name][0](inplanes, num_classes)  # 根据 name 选择对应的分类器模型
    base_model = model_map[name][1]  # 根据 name 选择对应的基础模型

    model = base_model(backbone, classifier, aux_classifier)  # 构建最终的模型
    return model  # 返回构建的模型


def _load_model(
    arch_type, backbone, pretrained, progress, num_classes, aux_loss, **kwargs



# 解构赋值语句，从函数调用的参数中提取特定变量值
arch_type, backbone, pretrained, progress, num_classes, aux_loss, **kwargs
# arch_type: 神经网络的架构类型
# backbone: 神经网络的主干结构
# pretrained: 是否使用预训练的权重
# progress: 是否显示训练进度条
# num_classes: 分类任务中的类别数目
# aux_loss: 是否使用辅助损失函数
# **kwargs: 其他关键字参数，用于接收函数调用时传入的任意关键字参数
    if pretrained:
        # 如果使用预训练模型，设置辅助损失为True
        aux_loss = True
    # 使用_resnet函数创建一个带有ResNet-50骨干网络的模型，传递参数arch_type、backbone、num_classes、aux_loss以及其他关键字参数
    model = _segm_resnet(arch_type, backbone, num_classes, aux_loss, **kwargs)
    # 返回创建的模型对象
    return model


def fcn_resnet50(
    pretrained=False, progress=True, num_classes=21, aux_loss=None, **kwargs
):
    """Constructs a Fully-Convolutional Network model with a ResNet-50 backbone.
    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017 which
            contains the same classes as Pascal VOC
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    # 调用_load_model函数加载一个基于ResNet-50骨干网络的全卷积网络模型，传递参数pretrained、progress、num_classes、aux_loss以及其他关键字参数
    return _load_model(
        "fcn", "resnet50", pretrained, progress, num_classes, aux_loss, **kwargs
    )


# Taken from @fmassa example slides and https://github.com/facebookresearch/detr
class DETR(nn.Module):
    """
    Demo DETR implementation.

    Demo implementation of DETR in minimal number of lines, with the
    following differences wrt DETR in the paper:
    * learned positional encoding (instead of sine)
    * positional encoding is passed at input (instead of attention)
    * fc bbox predictor (instead of MLP)
    The model achieves ~40 AP on COCO val5k and runs at ~28 FPS on Tesla V100.
    Only batch size 1 supported.
    """

    def __init__(
        self,
        num_classes,
        hidden_dim=256,
        nheads=8,
        num_encoder_layers=6,
        num_decoder_layers=6,
    ):
        super().__init__()

        # create ResNet-50 backbone
        self.backbone = resnet50()
        del self.backbone.fc

        # create conversion layer
        self.conv = nn.Conv2d(2048, hidden_dim, 1)

        # create a default PyTorch transformer
        self.transformer = nn.Transformer(
            hidden_dim, nheads, num_encoder_layers, num_decoder_layers
        )

        # prediction heads, one extra class for predicting non-empty slots
        # note that in baseline DETR linear_bbox layer is 3-layer MLP
        self.linear_class = nn.Linear(hidden_dim, num_classes + 1)
        self.linear_bbox = nn.Linear(hidden_dim, 4)

        # output positional encodings (object queries)
        self.query_pos = nn.Parameter(torch.rand(100, hidden_dim))

        # spatial positional encodings
        # note that in baseline DETR we use sine positional encodings
        self.row_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
        self.col_embed = nn.Parameter(torch.rand(50, hidden_dim // 2))
    def forward(self, inputs):
        # propagate inputs through ResNet-50 up to avg-pool layer
        # 通过ResNet-50将输入数据传播至平均池化层
        x = self.backbone.conv1(inputs)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # convert from 2048 to 256 feature planes for the transformer
        # 将特征从2048维转换为256维，以供Transformer使用
        h = self.conv(x)

        # construct positional encodings
        # 构建位置编码
        H, W = h.shape[-2:]
        pos = (
            torch.cat(
                [
                    self.col_embed[:W].unsqueeze(0).repeat(H, 1, 1),
                    self.row_embed[:H].unsqueeze(1).repeat(1, W, 1),
                ],
                dim=-1,
            )
            .flatten(0, 1)
            .unsqueeze(1)
        )

        # propagate through the transformer
        # 通过Transformer传播数据
        # TODO (alband) Why this is not automatically broadcasted? (had to add the repeat)
        f = pos + 0.1 * h.flatten(2).permute(2, 0, 1)
        s = self.query_pos.unsqueeze(1)
        s = s.expand(s.size(0), inputs.size(0), s.size(2))
        h = self.transformer(f, s).transpose(0, 1)

        # finally project transformer outputs to class labels and bounding boxes
        # 最后将Transformer的输出投影到类别标签和边界框
        return {
            "pred_logits": self.linear_class(h),
            "pred_boxes": self.linear_bbox(h).sigmoid(),
        }
def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/
    The boxes should be in [x0, y0, x1, y1] format
    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()  # 检查boxes1的坐标顺序是否正确
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()  # 检查boxes2的坐标顺序是否正确
    
    # 计算boxes1和boxes2之间的IoU和并集面积
    iou, union = box_iou(boxes1, boxes2)

    # 计算左上角和右下角的坐标
    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    # 计算宽高并确保非负
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]  # 计算交集的面积

    # 返回generalized IoU计算结果
    return iou - (area - union) / area


def box_cxcywh_to_xyxy(x):
    """
    Convert bounding boxes from (center_x, center_y, width, height) format to
    (x0, y0, x1, y1) format.
    """
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_area(boxes):
    """
    Computes the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.
    Args:
        boxes (Tensor[N, 4]): boxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format
    Returns:
        area (Tensor[N]): area for each box
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    """
    Computes IoU (Intersection over Union) between two sets of boxes.
    Args:
        boxes1 (Tensor[N, 4]): bounding boxes, each box is specified in (x1, y1, x2, y2) format.
        boxes2 (Tensor[M, 4]): bounding boxes, each box is specified in (x1, y1, x2, y2) format.
    Returns:
        iou (Tensor[N, M]): IoU between each pair of boxes.
        union (Tensor[N, M]): Union area for each pair of boxes.
    """
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    # 计算交集的左上角和右下角坐标
    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    # 计算交集的宽高并确保非负
    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    # 计算并集面积
    union = area1[:, None] + area2 - inter

    # 计算IoU
    iou = inter / union
    return iou, union


def is_dist_avail_and_initialized():
    """
    Check if distributed training is available and initialized.
    """
    return False


def get_world_size():
    """
    Get the world size for distributed training.
    """
    if not is_dist_avail_and_initialized():
        return 1


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """
    Computes the precision@k for the specified values of k.
    Args:
        output (Tensor): model output
        target (Tensor): ground truth labels
        topk (tuple): tuple of integers specifying top-k precisions to compute
    Returns:
        res (list of Tensor): list containing precision@k for each k in topk
    """
    if target.numel() == 0:
        return [torch.zeros([], device=output.device)]
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class SetCriterion(nn.Module):
    """
    This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        # 调用父类的初始化方法
        super().__init__()
        # 初始化类别数
        self.num_classes = num_classes
        # 设置匹配器
        self.matcher = matcher
        # 设置损失权重字典
        self.weight_dict = weight_dict
        # 设置空对象类别的相对分类权重
        self.eos_coef = eos_coef
        # 设置要应用的所有损失函数列表
        self.losses = losses
        # 创建一个张量表示类别权重，默认所有类别权重为1，最后一个类别（空对象类别）的权重为eos_coef
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        # 将权重张量注册为缓冲区，这样它将受到模型的管理但不会被优化器更新
        self.register_buffer("empty_weight", empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        # 确保输出中包含预测的logits
        assert "pred_logits" in outputs
        # 获取预测的logits
        src_logits = outputs["pred_logits"]

        # 根据indices获取目标类别标签
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat(
            [t["labels"][J] for t, (_, J) in zip(targets, indices)]
        )
        # 创建目标类别张量，初始值为num_classes，大小与src_logits的前两个维度相同
        target_classes = torch.full(
            src_logits.shape[:2],
            self.num_classes,
            dtype=torch.int64,
            device=src_logits.device,
        )
        target_classes[idx] = target_classes_o

        # 计算交叉熵损失
        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2), target_classes, self.empty_weight
        )
        # 组成损失字典
        losses = {"loss_ce": loss_ce}

        if log:
            # 如果log为True，计算分类错误率作为损失之一
            # 注意：这可能应该是一个单独的损失，而不是在这里直接插入
            losses["class_error"] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        # 获取预测的logits
        pred_logits = outputs["pred_logits"]
        device = pred_logits.device
        # 获取目标标签的长度信息，作为目标长度张量
        tgt_lengths = torch.as_tensor(
            [len(v["labels"]) for v in targets], device=device
        )
        # 计算预测的非空盒子数量（即非空对象的数量）
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        # 计算基于L1损失的基数错误
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        # 组成损失字典
        losses = {"cardinality_error": card_err}
        return losses
    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, h, w), normalized by the image size.
        """
        # 确保输出中有预测框的键
        assert "pred_boxes" in outputs
        # 根据索引获取源排列索引
        idx = self._get_src_permutation_idx(indices)
        # 从输出中获取预测的框
        src_boxes = outputs["pred_boxes"][idx]
        # 拼接所有目标框的列表，注意目标数据结构
        target_boxes = torch.cat(
            [t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0
        )

        # 计算 L1 回归损失
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        # 初始化损失字典
        losses = {}
        # 计算并存储边界框损失
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        # 计算 GIoU 损失
        loss_giou = 1 - torch.diag(
            generalized_box_iou(
                box_cxcywh_to_xyxy(src_boxes), box_cxcywh_to_xyxy(target_boxes)
            )
        )
        # 计算并存储 GIoU 损失
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        # 返回损失字典
        return losses

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        # 确保输出中有预测掩码的键
        assert "pred_masks" in outputs

        # 获取源排列索引和目标排列索引
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)

        # 获取预测掩码
        src_masks = outputs["pred_masks"]

        # TODO 使用 valid 变量来掩盖由于填充而无效的区域在损失中
        # 从目标中获取掩码并解压缩为张量
        target_masks, valid = nested_tensor_from_tensor_list(
            [t["masks"] for t in targets]
        ).decompose()
        target_masks = target_masks.to(src_masks)

        # 根据源排列索引筛选预测掩码
        src_masks = src_masks[src_idx]
        # 将预测掩码插值至目标大小
        src_masks = interpolate(
            src_masks[:, None],
            size=target_masks.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        # 将掩码展平为一维
        src_masks = src_masks[:, 0].flatten(1)

        # 根据目标排列索引筛选目标掩码并展平为一维
        target_masks = target_masks[tgt_idx].flatten(1)

        # 初始化损失字典
        losses = {
            # 计算并存储焦点损失
            "loss_mask": sigmoid_focal_loss(
                src_masks, target_masks, num_boxes
            ),
            # 计算并存储 Dice 损失
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        # 返回损失字典
        return losses

    def _get_src_permutation_idx(self, indices):
        # 根据 indices 重新排列预测，生成批次索引和源索引
        batch_idx = torch.cat(
            [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
        )
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx
    def _get_tgt_permutation_idx(self, indices):
        # 根据给定的 indices 对目标进行重新排序
        batch_idx = torch.cat(
            [torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)]
        )
        # 将所有目标的索引拼接成一个张量
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        # 定义不同损失类型对应的损失函数映射表
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
            "masks": self.loss_masks,
        }
        # 断言所请求的损失在损失映射表中存在
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        # 调用对应损失函数计算损失并返回结果
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # 剔除辅助输出以获得主输出
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # 获取输出与目标之间的匹配关系索引
        indices = self.matcher(outputs_without_aux, targets)

        # 计算所有节点中目标框的平均数量，用于归一化
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor(
            [num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        # 如果处于分布式环境并且已初始化，进行所有节点的目标框数量全局归约
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        # 对归约后的目标框数量进行归一化处理
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # 计算所有请求的损失
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # 如果存在辅助损失，对每个中间层的输出重复上述过程
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == "masks":
                        # 中间层的 masks 损失计算成本过高，忽略此部分
                        continue
                    kwargs = {}
                    if loss == "labels":
                        # 仅在最后一层启用日志记录
                        kwargs = {"log": False}
                    # 获取对应损失函数计算损失，并按格式更新到总损失字典中
                    l_dict = self.get_loss(
                        loss, aux_outputs, targets, indices, num_boxes, **kwargs
                    )
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # 返回所有损失的字典
        return losses
# 定义一个名为 HungarianMatcher 的类，继承自 nn.Module
class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """
    
    def __init__(
        self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1
    ):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        # 调用父类的构造方法
        super().__init__()
        # 设置类内部成员变量，用于记录分类错误、边界框坐标误差和 giou 损失的相对权重
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        # 断言，确保至少有一个成本项不为零，否则抛出异常
        assert (
            cost_class != 0 or cost_bbox != 0 or cost_giou != 0
        ), "all costs cant be 0"

    @torch.no_grad()


这段代码定义了一个名为 `HungarianMatcher` 的类，用于计算网络预测结果与目标之间的匹配。类的构造方法 `__init__` 初始化了三个相对权重参数，用于计算匹配成本。
```