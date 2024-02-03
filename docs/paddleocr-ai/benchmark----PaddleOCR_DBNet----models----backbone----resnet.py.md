# `.\PaddleOCR\benchmark\PaddleOCR_DBNet\models\backbone\resnet.py`

```
# 导入 math 模块
import math
# 导入 paddle 模块
import paddle
# 从 paddle 模块中导入 nn 模块
from paddle import nn

# 将 nn.BatchNorm2D 赋值给 BatchNorm2d
BatchNorm2d = nn.BatchNorm2D

# 定义 __all__ 列表，包含一系列模型名称
__all__ = [
    'ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'deformable_resnet18', 'deformable_resnet50', 'resnet152'
]

# 定义 model_urls 字典，包含不同模型对应的下载链接
model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

# 定义 constant_init 函数，用于初始化模型参数
def constant_init(module, constant, bias=0):
    # 使用常数 constant 初始化模型参数 module.weight
    module.weight = paddle.create_parameter(
        shape=module.weight.shape,
        dtype='float32',
        default_initializer=paddle.nn.initializer.Constant(constant))
    # 如果模型参数 module 中存在 bias 属性
    if hasattr(module, 'bias'):
        # 使用常数 bias 初始化模型参数 module.bias
        module.bias = paddle.create_parameter(
            shape=module.bias.shape,
            dtype='float32',
            default_initializer=paddle.nn.initializer.Constant(bias))

# 定义 conv3x3 函数，实现 3x3 卷积操作
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    # 返回一个 2D 卷积层，使用 3x3 的卷积核，padding 为 1
    return nn.Conv2D(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias_attr=False)

# 定义 BasicBlock 类，表示 ResNet 中的基本块
class BasicBlock(nn.Layer):
    # 设置扩展系数为 1
    expansion = 1
    # 初始化 BasicBlock 类，设置输入通道数、输出通道数、步长、下采样、可选的 DCN 参数
    def __init__(self, inplanes, planes, stride=1, downsample=None, dcn=None):
        # 调用父类的初始化方法
        super(BasicBlock, self).__init__()
        # 判断是否使用 DCN
        self.with_dcn = dcn is not None
        # 创建第一个卷积层，3x3 的卷积核
        self.conv1 = conv3x3(inplanes, planes, stride)
        # 创建 BatchNorm2d 层
        self.bn1 = BatchNorm2d(planes, momentum=0.1)
        # 创建 ReLU 激活函数
        self.relu = nn.ReLU()
        # 初始化是否使用 modulated DCN
        self.with_modulated_dcn = False
        # 如果不使用 DCN，则创建普通的卷积层
        if not self.with_dcn:
            self.conv2 = nn.Conv2D(
                planes, planes, kernel_size=3, padding=1, bias_attr=False)
        # 如果使用 DCN，则创建 DeformConv2D 层
        else:
            from paddle.version.ops import DeformConv2D
            deformable_groups = dcn.get('deformable_groups', 1)
            offset_channels = 18
            self.conv2_offset = nn.Conv2D(
                planes,
                deformable_groups * offset_channels,
                kernel_size=3,
                padding=1)
            self.conv2 = DeformConv2D(
                planes, planes, kernel_size=3, padding=1, bias_attr=False)
        # 创建 BatchNorm2d 层
        self.bn2 = BatchNorm2d(planes, momentum=0.1)
        # 设置下采样和步长
        self.downsample = downsample
        self.stride = stride

    # 前向传播函数
    def forward(self, x):
        # 保存残差连接
        residual = x

        # 第一个卷积层、BatchNorm2d 层和 ReLU 激活函数
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 根据是否使用 DCN，选择不同的卷积操作
        if not self.with_dcn:
            out = self.conv2(out)
        else:
            offset = self.conv2_offset(out)
            out = self.conv2(out, offset)
        out = self.bn2(out)

        # 如果有下采样，则对残差进行下采样
        if self.downsample is not None:
            residual = self.downsample(x)

        # 残差连接和 ReLU 激活函数
        out += residual
        out = self.relu(out)

        return out
# 定义一个 Bottleneck 类，用于构建 ResNet 中的瓶颈块
class Bottleneck(nn.Layer):
    # 扩展系数，用于计算输出通道数
    expansion = 4

    # 初始化函数，定义瓶颈块的结构
    def __init__(self, inplanes, planes, stride=1, downsample=None, dcn=None):
        # 调用父类的初始化函数
        super(Bottleneck, self).__init__()
        # 判断是否使用 DCN（Deformable Convolutional Networks）
        self.with_dcn = dcn is not None
        # 第一个卷积层，1x1 卷积，用于降维
        self.conv1 = nn.Conv2D(inplanes, planes, kernel_size=1, bias_attr=False)
        # 第一个批归一化层
        self.bn1 = BatchNorm2d(planes, momentum=0.1)
        # 判断是否使用 DCN，并设置是否使用 modulated DCN
        self.with_modulated_dcn = False
        # 如果不使用 DCN，则使用普通的 3x3 卷积层
        if not self.with_dcn:
            self.conv2 = nn.Conv2D(
                planes,
                planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias_attr=False)
        # 如果使用 DCN，则使用 Deformable Convolutional Networks
        else:
            deformable_groups = dcn.get('deformable_groups', 1)
            from paddle.vision.ops import DeformConv2D
            offset_channels = 18
            # 偏移量卷积层
            self.conv2_offset = nn.Conv2D(
                planes,
                deformable_groups * offset_channels,
                stride=stride,
                kernel_size=3,
                padding=1)
            # DCN 卷积层
            self.conv2 = DeformConv2D(
                planes,
                planes,
                kernel_size=3,
                padding=1,
                stride=stride,
                bias_attr=False)
        # 第二个批归一化层
        self.bn2 = BatchNorm2d(planes, momentum=0.1)
        # 第三个卷积层，1x1 卷积，用于升维
        self.conv3 = nn.Conv2D(
            planes, planes * 4, kernel_size=1, bias_attr=False)
        # 第三个批归一化层
        self.bn3 = BatchNorm2d(planes * 4, momentum=0.1)
        # 激活函数 ReLU
        self.relu = nn.ReLU()
        # 下采样层，用于调整输入输出通道数一致
        self.downsample = downsample
        # 步长
        self.stride = stride
        # 记录是否使用 DCN
        self.dcn = dcn
        self.with_dcn = dcn is not None
    # 定义前向传播函数，接受输入 x
    def forward(self, x):
        # 保存输入 x 作为残差
        residual = x

        # 第一层卷积操作
        out = self.conv1(x)
        # 对第一层卷积结果进行批量归一化
        out = self.bn1(out)
        # 对第一层卷积结果应用激活函数 ReLU
        out = self.relu(out)

        # 判断是否使用 DCN（Deformable Convolutional Networks）
        # 如果不使用 DCN，则直接进行第二层卷积操作
        if not self.with_dcn:
            out = self.conv2(out)
        # 如果使用 DCN，则进行偏移量卷积和卷积操作
        else:
            # 计算偏移量
            offset = self.conv2_offset(out)
            # 使用偏移量进行卷积操作
            out = self.conv2(out, offset)
        # 对第二层卷积结果进行批量归一化
        out = self.bn2(out)
        # 对第二层卷积结果应用激活函数 ReLU
        out = self.relu(out)

        # 第三层卷积操作
        out = self.conv3(out)
        # 对第三层卷积结果进行批量归一化
        out = self.bn3(out)

        # 如果存在下采样操作，则对输入 x 进行下采样
        if self.downsample is not None:
            residual = self.downsample(x)

        # 将残差加到当前输出上
        out += residual
        # 对输出应用激活函数 ReLU
        out = self.relu(out)

        # 返回最终输出
        return out
# 定义 ResNet 类，继承自 nn.Layer 类
class ResNet(nn.Layer):
    # 初始化函数，接受 block、layers、in_channels 和 dcn 参数
    def __init__(self, block, layers, in_channels=3, dcn=None):
        # 将 dcn 参数赋值给 self.dcn
        self.dcn = dcn
        # 初始化 self.inplanes 为 64
        self.inplanes = 64
        # 调用父类的初始化函数
        super(ResNet, self).__init__()
        # 初始化空列表 self.out_channels
        self.out_channels = []
        # 创建第一个卷积层，输入通道数为 in_channels，输出通道数为 64，卷积核大小为 7，步长为 2，填充为 3，不使用偏置
        self.conv1 = nn.Conv2D(
            in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias_attr=False)
        # 创建 BatchNorm2d 层，输入通道数为 64，动量为 0.1
        self.bn1 = BatchNorm2d(64, momentum=0.1)
        # 创建 ReLU 激活函数层
        self.relu = nn.ReLU()
        # 创建最大池化层，池化核大小为 3，步长为 2，填充为 1
        self.maxpool = nn.MaxPool2D(kernel_size=3, stride=2, padding=1)
        # 创建第一个残差块
        self.layer1 = self._make_layer(block, 64, layers[0])
        # 创建第二个残差块
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dcn=dcn)
        # 创建第三个残差块
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dcn=dcn)
        # 创建第四个残差块
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dcn=dcn)

        # 如果存在 dcn 参数
        if self.dcn is not None:
            # 遍历所有模块
            for m in self.modules():
                # 如果是 Bottleneck 或 BasicBlock 类型的模块
                if isinstance(m, Bottleneck) or isinstance(m, BasicBlock):
                    # 如果模块具有 conv2_offset 属性
                    if hasattr(m, 'conv2_offset'):
                        # 将 conv2_offset 初始化为 0

    # 创建残差块函数，接受 block、planes、blocks、stride 和 dcn 参数
    def _make_layer(self, block, planes, blocks, stride=1, dcn=None):
        # 初始化 downsample 为 None
        downsample = None
        # 如果步长不为 1 或 self.inplanes 不等于 planes * block.expansion
        if stride != 1 or self.inplanes != planes * block.expansion:
            # 创建下采样层
            downsample = nn.Sequential(
                nn.Conv2D(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias_attr=False),
                BatchNorm2d(
                    planes * block.expansion, momentum=0.1), )

        # 初始化空列表 layers
        layers = []
        # 添加第一个残差块到 layers
        layers.append(block(self.inplanes, planes, stride, downsample, dcn=dcn))
        # 更新 self.inplanes 为 planes * block.expansion
        self.inplanes = planes * block.expansion
        # 循环创建剩余的残差块
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dcn=dcn))
        # 将 planes * block.expansion 添加到 self.out_channels
        self.out_channels.append(planes * block.expansion)
        # 返回包含所有残差块的序列
        return nn.Sequential(*layers)
    # 定义前向传播函数，接收输入 x
    def forward(self, x):
        # 使用第一个卷积层对输入 x 进行卷积操作
        x = self.conv1(x)
        # 对卷积结果进行批量归一化处理
        x = self.bn1(x)
        # 对归一化后的结果进行激活函数处理
        x = self.relu(x)
        # 对激活后的结果进行最大池化操作

        x = self.maxpool(x)

        # 将处理后的结果分别传入不同的残差块中
        x2 = self.layer1(x)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)

        # 返回不同层级的特征图
        return x2, x3, x4, x5
# 加载 Torch 模型参数到 PaddlePaddle 模型中
def load_torch_params(paddle_model, torch_patams):
    # 获取 PaddlePaddle 模型的参数字典
    paddle_params = paddle_model.state_dict()

    # 需要处理的全连接层名称
    fc_names = ['classifier']
    # 遍历 Torch 模型参数字典
    for key, torch_value in torch_patams.items():
        # 跳过 num_batches_tracked 参数
        if 'num_batches_tracked' in key:
            continue
        # 替换参数名称中的部分字符串
        key = key.replace("running_var", "_variance").replace(
            "running_mean", "_mean").replace("module.", "")
        # 将 Torch 参数转换为 NumPy 数组
        torch_value = torch_value.detach().cpu().numpy()
        # 如果参数名称在 PaddlePaddle 模型中存在
        if key in paddle_params:
            # 判断是否为全连接层的权重参数，忽略偏置参数
            flag = [i in key for i in fc_names]
            if any(flag) and "weight" in key:
                # 调整参数的维度顺序
                new_shape = [1, 0] + list(range(2, torch_value.ndim))
                print(
                    f"name: {key}, ori shape: {torch_value.shape}, new shape: {torch_value.transpose(new_shape).shape}"
                )
                torch_value = torch_value.transpose(new_shape)
            # 更新 PaddlePaddle 模型参数字典
            paddle_params[key] = torch_value
        else:
            print(f'{key} not in paddle')
    # 将更新后的参数字典设置回 PaddlePaddle 模型
    paddle_model.set_state_dict(paddle_params)


# 加载模型参数
def load_models(model, model_name):
    import torch.utils.model_zoo as model_zoo
    # 从 Torch 模型库加载参数
    torch_patams = model_zoo.load_url(model_urls[model_name])
    # 调用加载 Torch 参数的函数
    load_torch_params(model, torch_patams)


# 构建 ResNet-18 模型
def resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # 创建 ResNet-18 模型
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    # 如果需要加载预训练参数
    if pretrained:
        assert kwargs.get(
            'in_channels',
            3) == 3, 'in_channels must be 3 whem pretrained is True'
        print('load from imagenet')
        # 加载预训练参数
        load_models(model, 'resnet18')
    return model


# 构建带可变形卷积的 ResNet-18 模型
def deformable_resnet18(pretrained=True, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # 创建带可变形卷积的 ResNet-18 模型
    model = ResNet(
        BasicBlock, [2, 2, 2, 2], dcn=dict(deformable_groups=1), **kwargs)
    # 如果使用预训练模型
    if pretrained:
        # 断言输入通道数为3
        assert kwargs.get('in_channels', 3) == 3, 'in_channels must be 3 when pretrained is True'
        # 打印加载自ImageNet的信息
        print('load from imagenet')
        # 加载预训练模型的权重
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']), strict=False)
    # 返回模型
    return model
# 构建一个 ResNet-34 模型
def resnet34(pretrained=True, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # 使用 BasicBlock 构建 ResNet 模型，层数分别为 [3, 4, 6, 3]
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    # 如果需要预训练，则加载 ImageNet 预训练模型参数
    if pretrained:
        # 断言 in_channels 参数为 3，当预训练为 True 时
        assert kwargs.get(
            'in_channels',
            3) == 3, 'in_channels must be 3 whem pretrained is True'
        # 加载预训练模型参数
        model.load_state_dict(
            model_zoo.load_url(model_urls['resnet34']), strict=False)
    return model


# 构建一个 ResNet-50 模型
def resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # 使用 Bottleneck 构建 ResNet 模型，层数分别为 [3, 4, 6, 3]
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    # 如果需要预训练，则加载 ImageNet 预训练模型参数
    if pretrained:
        # 断言 in_channels 参数为 3，当预训练为 True 时
        assert kwargs.get(
            'in_channels',
            3) == 3, 'in_channels must be 3 whem pretrained is True'
        # 调用 load_models 函数加载预训练模型参数
        load_models(model, 'resnet50')
    return model


# 构建一个带有可变形卷积的 ResNet-50 模型
def deformable_resnet50(pretrained=True, **kwargs):
    """Constructs a ResNet-50 model with deformable conv.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # 使用 Bottleneck 构建 ResNet 模型，层数分别为 [3, 4, 6, 3]，并设置可变形卷积参数
    model = ResNet(
        Bottleneck, [3, 4, 6, 3], dcn=dict(deformable_groups=1), **kwargs)
    # 如果需要预训练，则加载 ImageNet 预训练模型参数
    if pretrained:
        # 断言 in_channels 参数为 3，当预训练为 True 时
        assert kwargs.get(
            'in_channels',
            3) == 3, 'in_channels must be 3 whem pretrained is True'
        # 加载预训练模型参数
        model.load_state_dict(
            model_zoo.load_url(model_urls['resnet50']), strict=False)
    return model


# 构建一个 ResNet-101 模型
def resnet101(pretrained=True, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # 使用 Bottleneck 构建 ResNet 模型，层数分别为 [3, 4, 23, 3]
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    # 如果需要预训练，则加载 ImageNet 预训练模型参数
    if pretrained:
        # 断言 in_channels 参数为 3，当预训练为 True 时
        assert kwargs.get(
            'in_channels',
            3) == 3, 'in_channels must be 3 whem pretrained is True'
        # 加载预训练模型参数
        model.load_state_dict(
            model_zoo.load_url(model_urls['resnet101']), strict=False)
    return model
# 定义 ResNet-152 模型
def resnet152(pretrained=True, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    # 使用 ResNet 类构建 ResNet-152 模型，参数为 Bottleneck 类型的残差块，每个阶段的残差块数量分别为 [3, 8, 36, 3]
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    # 如果 pretrained 为 True，则加载在 ImageNet 上预训练的模型参数
    if pretrained:
        # 断言 in_channels 参数为 3，当 pretrained 为 True 时
        assert kwargs.get(
            'in_channels',
            3) == 3, 'in_channels must be 3 whem pretrained is True'
        # 加载预训练的 ResNet-152 模型参数
        model.load_state_dict(
            model_zoo.load_url(model_urls['resnet152']), strict=False)
    # 返回构建好的模型
    return model


if __name__ == '__main__':

    # 创建一个形状为 [2, 3, 640, 640] 的全零张量
    x = paddle.zeros([2, 3, 640, 640])
    # 加载预训练的 ResNet-50 模型
    net = resnet50(pretrained=True)
    # 将输入张量 x 输入到网络中，得到输出张量 y
    y = net(x)
    # 遍历输出张量 y 中的每个元素
    for u in y:
        # 打印每个元素的形状
        print(u.shape)

    # 打印网络的输出通道数
    print(net.out_channels)
```