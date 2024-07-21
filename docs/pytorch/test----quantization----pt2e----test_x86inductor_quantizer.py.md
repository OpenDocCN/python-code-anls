# `.\pytorch\test\quantization\pt2e\test_x86inductor_quantizer.py`

```py
# Owner(s): ["oncall: quantization"]

# 导入必要的模块和库
import copy  # 导入深拷贝模块
import itertools  # 导入迭代工具模块
from enum import Enum  # 导入枚举类型支持

import torch  # 导入PyTorch库
import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq  # 导入x86感应器量化器
import torch.nn as nn  # 导入PyTorch的神经网络模块
from torch._export import capture_pre_autograd_graph  # 导入用于捕获自动求导图的函数
from torch.ao.quantization import ObserverBase  # 导入观察者基类
from torch.ao.quantization.quantize_pt2e import (  # 导入量化相关的函数和类
    convert_pt2e,
    prepare_pt2e,
    prepare_qat_pt2e,
)
from torch.ao.quantization.quantizer.x86_inductor_quantizer import (  # 导入x86感应器量化器相关的内容
    QUANT_ANNOTATION_KEY,
    X86InductorQuantizer,
)
from torch.testing._internal.common_quantization import (  # 导入用于量化测试的通用模块
    NodeSpec as ns,
    QuantizationTestCase,
    skipIfNoInductorSupport,
    skipIfNoX86,
)
from torch.testing._internal.common_quantized import override_quantized_engine  # 导入覆盖量化引擎的函数
from torch.testing._internal.common_utils import skipIfTorchDynamo  # 导入如果使用Torch Dynamo则跳过的函数


class NodePosType(Enum):
    left = 1
    right = 2
    both = 3


# 定义辅助测试模块的类
class TestHelperModules:
    # 定义包含单个Conv2d层的模块类
    class SingleConv2dModule(torch.nn.Module):
        def __init__(self, with_bn=False) -> None:
            super().__init__()
            # 定义包含Conv2d层的网络结构
            self.conv = nn.Conv2d(3, 6, (2, 2), stride=(1, 1), padding=(1, 1))
            self.bn = torch.nn.BatchNorm2d(6)  # 如果有BN层，则定义BN层
            self.with_bn = with_bn  # 是否包含BN层的标志

        def forward(self, x):
            x = self.conv(x)  # 前向传播中使用Conv2d层
            if self.with_bn:
                x = self.bn(x)  # 如果有BN层，则在前向传播中使用BN层
            return x

    # 定义包含Conv2d层、非线性操作、BN层和最大池化层的模块类
    class Conv2dUnaryModule(torch.nn.Module):
        def __init__(self, post_op, use_bias: bool = False, with_bn=False) -> None:
            super().__init__()
            # 定义包含Conv2d层的网络结构，支持使用偏置、BN层和最大池化层
            self.conv = nn.Conv2d(
                3, 6, (2, 2), stride=(1, 1), padding=(1, 1), bias=use_bias
            )
            self.post_op = post_op  # 定义非线性操作
            self.bn = torch.nn.BatchNorm2d(6)  # 如果有BN层，则定义BN层
            self.with_bn = with_bn  # 是否包含BN层的标志
            self.maxpool = torch.nn.MaxPool2d((3, 3))  # 定义最大池化层

        def forward(self, x):
            x = self.conv(x)  # 前向传播中使用Conv2d层
            if self.with_bn:
                x = self.bn(x)  # 如果有BN层，则在前向传播中使用BN层
            x = self.post_op(x)  # 前向传播中应用非线性操作
            x = self.maxpool(x)  # 前向传播中使用最大池化层
            return x
    # 定义一个名为 Conv2dAddModule 的自定义 PyTorch 模块类，用于实现卷积操作并支持加法操作
    class Conv2dAddModule(torch.nn.Module):
        # 初始化函数，设定模块的各个属性
        def __init__(
            self,
            inplace_add: bool = False,               # 控制是否原地加法操作的布尔值，默认为 False
            conv2d_type: NodePosType = NodePosType.left,  # 卷积类型，枚举类型 NodePosType 的默认值为左
            use_bias: bool = False,                 # 控制卷积层是否使用偏置的布尔值，默认为 False
            with_bn: bool = False,                  # 控制是否使用批归一化的布尔值，默认为 False
        ) -> None:
            super().__init__()
            # 创建一个卷积层对象 conv，输入通道数为 3，输出通道数为 3，卷积核大小为 3x3，步长为 1，填充为 1，是否使用偏置由 use_bias 决定
            self.conv = torch.nn.Conv2d(
                in_channels=3,
                out_channels=3,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
            )
            # 创建第二个卷积层对象 conv2，具有与 conv 相同的参数设置
            self.conv2 = torch.nn.Conv2d(
                in_channels=3,
                out_channels=3,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
            )
            # 创建一个 ReLU 激活函数对象 relu
            self.relu = nn.ReLU()
            # 将 inplace_add 参数设置为类的属性，用于控制是否进行原地加法操作
            self.inplace_add = inplace_add
            # 将 conv2d_type 参数设置为类的属性，表示卷积类型
            self.conv2d_type = conv2d_type
            # 创建一个批归一化层对象 bn，输入通道数为 3，用于在需要时应用批归一化
            self.bn = torch.nn.BatchNorm2d(3)
            # 将 with_bn 参数设置为类的属性，表示是否使用批归一化
            self.with_bn = with_bn

        # 前向传播函数，定义了模块的计算流程
        def forward(self, x):
            # 如果卷积类型为左
            if self.conv2d_type == NodePosType.left:
                # 如果 inplace_add 为 True，执行原地加法操作
                if self.inplace_add:
                    # 进行卷积操作，并将结果存储在 tmp 中
                    tmp = self.conv(x)
                    # 如果使用批归一化，将 tmp 输入到批归一化层中
                    if self.with_bn:
                        tmp = self.bn(tmp)
                    # 将 tmp 与输入 x 上的 ReLU 激活函数的输出相加，并返回结果
                    tmp += self.relu(x)
                    return tmp
                else:
                    # 执行卷积操作，并将结果存储在 tmp 中
                    tmp = self.conv(x)
                    # 如果使用批归一化，将 tmp 输入到批归一化层中
                    if self.with_bn:
                        tmp = self.bn(tmp)
                    # 将 tmp 与输入 x 上的 ReLU 激活函数的输出相加，并返回结果
                    return tmp + self.relu(x)
            # 如果卷积类型为右
            elif self.conv2d_type == NodePosType.right:
                # 如果 inplace_add 为 True，执行原地加法操作
                if self.inplace_add:
                    # 对输入 x 进行 ReLU 激活，并将结果存储在 tmp 中
                    tmp = self.relu(x)
                    # 将 tmp 与经过卷积操作的结果相加，并返回结果
                    tmp += self.conv(x)
                    return tmp
                else:
                    # 对输入 x 进行 ReLU 激活，并将结果与经过卷积操作的结果相加，并返回结果
                    return self.relu(x) + self.conv(x)
            # 如果卷积类型为双
            elif self.conv2d_type == NodePosType.both:
                # 如果 inplace_add 为 True，执行原地加法操作
                if self.inplace_add:
                    # 对输入 x 进行第一次卷积操作，并将结果存储在 tmp 中
                    tmp = self.conv(x)
                    # 对输入 x 进行第二次卷积操作，并将结果与 tmp 原地相加，并返回结果
                    tmp += self.conv2(x)
                    return tmp
                else:
                    # 对输入 x 进行第一次卷积操作，并将结果与输入 x 上的第二次卷积操作结果相加，并返回结果
                    return self.conv(x) + self.conv2(x)
    # 定义一个继承自torch.nn.Module的类Conv2dAddReLUModule，用于执行包含卷积、ReLU激活和可选的批标准化的操作
    class Conv2dAddReLUModule(torch.nn.Module):
        # 初始化方法，设置模块的参数和层
        def __init__(
            self,
            inplace_add: bool = False,                    # 是否原地添加
            conv2d_type: NodePosType = NodePosType.left,  # 卷积类型，左、右或两者
            inplace_relu: bool = False,                  # 是否原地应用ReLU
            use_bias: bool = False,                      # 是否使用偏置项
            with_bn: bool = False,                       # 是否使用批标准化
        ) -> None:
            super().__init__()
            # 第一个卷积层，输入通道数为3，输出通道数为3，卷积核大小为3x3，步长为1，填充为1，根据use_bias确定是否使用偏置
            self.conv = torch.nn.Conv2d(
                in_channels=3,
                out_channels=3,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
            )
            # 第二个卷积层，与第一个卷积层设置相同
            self.conv2 = torch.nn.Conv2d(
                in_channels=3,
                out_channels=3,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=use_bias,
            )
            # ReLU激活函数
            self.relu = nn.ReLU()
            # 是否原地应用ReLU激活
            self.relu2 = nn.ReLU(inplace=inplace_relu)
            # 批标准化层，通道数为3
            self.bn = torch.nn.BatchNorm2d(3)
            # 是否使用批标准化
            self.with_bn = with_bn

        # 前向传播方法，定义模块的计算过程
        def forward(self, x):
            # 如果卷积类型为左
            if self.conv2d_type == NodePosType.left:
                # 如果选择原地添加
                if self.inplace_add:
                    # 进行第一个卷积操作
                    tmp = self.conv(x)
                    # 如果使用批标准化，则应用批标准化
                    if self.with_bn:
                        tmp = self.bn(tmp)
                    # 原地添加ReLU激活后返回结果
                    tmp += self.relu(x)
                    return self.relu2(tmp)
                else:
                    # 进行第一个卷积操作
                    tmp = self.conv(x)
                    # 如果使用批标准化，则应用批标准化
                    if self.with_bn:
                        tmp = self.bn(tmp)
                    # 将卷积结果与ReLU激活后的结果相加，并应用ReLU激活后返回结果
                    return self.relu2(tmp + self.relu(x))
            # 如果卷积类型为右
            elif self.conv2d_type == NodePosType.right:
                # 如果选择原地添加
                if self.inplace_add:
                    # 应用ReLU激活后进行卷积操作
                    tmp = self.relu(x)
                    tmp += self.conv(x)
                    # 返回经ReLU激活后的结果
                    return self.relu2(tmp)
                else:
                    # 进行ReLU激活后的卷积操作，再经ReLU激活后返回结果
                    return self.relu2(self.relu(x) + self.conv(x))
            # 如果卷积类型为左右都有
            elif self.conv2d_type == NodePosType.both:
                # 如果选择原地添加
                if self.inplace_add:
                    # 进行第一个和第二个卷积操作后原地添加，再经ReLU激活后返回结果
                    tmp = self.conv(x)
                    tmp += self.conv2(x)
                    return self.relu2(tmp)
                else:
                    # 第一个和第二个卷积操作后相加，再经ReLU激活后返回结果
                    return self.relu2(self.conv(x) + self.conv2(x))

    # 定义一个继承自nn.Module的类Conv2dSingleOpPowModule，用于执行单一操作和平方的操作
    class Conv2dSingleOpPowModule(nn.Module):
        # 初始化方法，设置模块的参数和层
        def __init__(self, single_op):
            super().__init__()
            # 单一的卷积层，输入通道数为2，输出通道数为2，卷积核大小为1x1
            self.conv = nn.Conv2d(2, 2, 1)
            # 单一操作的参数
            self.single_op = single_op

        # 前向传播方法，定义模块的计算过程
        def forward(self, x):
            # 经过单一的卷积操作后，应用单一操作，然后进行平方操作并返回结果
            x = self.conv(x)
            x = self.single_op(x)
            return torch.pow(x, 2)
    class SerialsConv2dAddReLUModule(torch.nn.Module):
        """连续的 2 Conv2d -> Add -> ReLU 模式。"""
    
        def __init__(
            self,
        ) -> None:
            super().__init__()
            # 第一个卷积层
            self.conv = torch.nn.Conv2d(
                in_channels=3,
                out_channels=3,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            )
            # 第二个卷积层
            self.conv2 = torch.nn.Conv2d(
                in_channels=3,
                out_channels=3,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            )
            # 第三个卷积层
            self.conv3 = torch.nn.Conv2d(
                in_channels=3,
                out_channels=3,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            )
            # 第四个卷积层
            self.conv4 = torch.nn.Conv2d(
                in_channels=3,
                out_channels=3,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            )
            # ReLU 激活函数
            self.relu = nn.ReLU()
            self.relu2 = nn.ReLU()
    
        def forward(self, x):
            # 第一层卷积
            x1 = self.conv(x)
            # 第二层卷积后，加上第三层卷积的结果，然后使用 ReLU 激活函数
            res1 = self.relu(self.conv2(x1) + self.conv3(x1))
            # 第四层卷积后，加上 res1 的结果，然后使用第二个 ReLU 激活函数
            res2 = self.relu2(self.conv4(res1) + res1)
            return res2
    
    class Conv2dCatMaxpool2d(torch.nn.Module):
        def __init__(
            self,
        ):
            super().__init__()
            # 第一个卷积层
            self.conv = torch.nn.Conv2d(
                3, 16, 7, bias=True, stride=2, padding=3, dilation=1
            )
            # 第二个卷积层
            self.conv2 = torch.nn.Conv2d(
                3, 16, 7, bias=True, stride=2, padding=3, dilation=1
            )
            # ReLU 激活函数
            self.relu = torch.nn.ReLU()
            # 最大池化层
            self.maxpool = torch.nn.MaxPool2d(3, stride=2, padding=1)
            # 第三个卷积层
            self.conv3 = torch.nn.Conv2d(
                32, 32, 7, bias=True, stride=2, padding=3, dilation=1
            )
    
        def forward(self, x):
            # 第一层卷积后，使用 ReLU 激活函数
            temp1 = self.relu(self.conv(x))
            # x + 1 后，再进行第二层卷积
            temp2 = self.conv2(x + 1)
            # 将 temp1 和 temp2 连接起来
            temp3 = torch.cat((temp1, temp2), 1)
            # 对连接后的结果进行最大池化
            temp4 = self.maxpool(temp3)
            # 最后一层卷积
            temp5 = self.conv3(temp4)
            return temp5
    
    class Conv2dAvgPool2d(torch.nn.Module):
        def __init__(
            self,
        ):
            super().__init__()
            # 第一个卷积层
            self.conv = torch.nn.Conv2d(
                3, 16, 7, bias=True, stride=2, padding=3, dilation=1
            )
            # 平均池化层
            self.avgpool = torch.nn.AvgPool2d(3, stride=2, padding=1)
    
        def forward(self, x):
            # 第一层卷积后，进行平均池化
            temp1 = self.avgpool(self.conv(x))
            return temp1
    class Conv2dCatSameInputs(torch.nn.Module):
        # 定义一个包含卷积和ReLU激活的神经网络模块
        def __init__(
            self,
        ):
            super().__init__()
            # 初始化卷积层，输入通道数为3，输出通道数为16，卷积核大小为7x7，带有偏置，步长为2，填充为3，扩展率为1
            self.conv = torch.nn.Conv2d(
                3, 16, 7, bias=True, stride=2, padding=3, dilation=1
            )
            # 初始化ReLU激活函数
            self.relu = torch.nn.ReLU()
    
        def forward(self, x):
            # 对输入的特征图x进行卷积和ReLU激活
            temp1 = self.relu(self.conv(x))
            # 将两个temp1的特征图沿着通道维度拼接
            temp3 = torch.cat((temp1, temp1), 1)
            return temp3
    
    
    class Conv2dCatSingleInput(torch.nn.Module):
        # 定义一个包含卷积和ReLU激活的神经网络模块
        def __init__(
            self,
        ):
            super().__init__()
            # 初始化卷积层，输入通道数为3，输出通道数为16，卷积核大小为7x7，带有偏置，步长为2，填充为3，扩展率为1
            self.conv = torch.nn.Conv2d(
                3, 16, 7, bias=True, stride=2, padding=3, dilation=1
            )
            # 初始化ReLU激活函数
            self.relu = torch.nn.ReLU()
    
        def forward(self, x):
            # 对输入的特征图x进行卷积和ReLU激活
            temp1 = self.relu(self.conv(x))
            # 将单个temp1的特征图沿着通道维度拼接
            temp3 = torch.cat((temp1,), 1)
            return temp3
    
    
    class SingleLinearModule(torch.nn.Module):
        # 定义一个只包含线性层的神经网络模块
        def __init__(self, use_bias) -> None:
            super().__init__()
            # 初始化线性层，输入和输出维度均为4，根据use_bias参数决定是否带有偏置
            self.linear = nn.Linear(4, 4, bias=use_bias)
    
        def forward(self, x):
            # 对输入x进行线性变换
            return self.linear(x)
    
    
    class LinearUnaryModule(torch.nn.Module):
        # 定义一个包含线性层和后处理操作的神经网络模块
        def __init__(
            self, use_bias, postop, inplace_postop=False, post_op_algo="none"
        ) -> None:
            super().__init__()
            # 初始化线性层，输入和输出维度均为4，根据use_bias参数决定是否带有偏置
            self.linear = nn.Linear(4, 4, bias=use_bias)
            # 根据postop参数选择不同的后处理操作，如GELU或其他操作
            if postop == nn.GELU:
                self.postop = postop(approximate=post_op_algo)
            else:
                self.postop = postop(inplace=inplace_postop)
    
        def forward(self, x):
            # 对输入x进行线性变换后，应用选择的后处理操作
            return self.postop(self.linear(x))
    # 定义一个名为 LinearAddModule 的类，继承自 torch.nn.Module，用于线性加法模块
    class LinearAddModule(torch.nn.Module):
        # 初始化方法，设置模块的参数和层
        def __init__(
            self,
            inplace_add: bool = False,  # 是否原地加法的标志
            linear_pos: NodePosType = NodePosType.left,  # 线性层的位置，可以是左、右或两者
            use_bias: bool = False,  # 是否使用偏置
        ) -> None:
            super().__init__()  # 调用父类的初始化方法
            # 创建一个输入和输出维度为 16 的线性层，可以选择是否带偏置
            self.linear = torch.nn.Linear(
                in_features=16, out_features=16, bias=use_bias
            )
            # 创建第二个相同配置的线性层
            self.linear2 = torch.nn.Linear(
                in_features=16, out_features=16, bias=use_bias
            )
            # 创建一个 ReLU 激活函数层
            self.relu = nn.ReLU()
            self.inplace_add = inplace_add  # 保存是否原地加法的设置
            self.linear_pos = linear_pos  # 保存线性层位置的设置
    
        # 前向传播方法，定义模块如何处理输入数据 x
        def forward(self, x):
            # 如果线性层在左侧
            if self.linear_pos == NodePosType.left:
                # 如果使用原地加法
                if self.inplace_add:
                    # 计算线性层的输出，然后与 ReLU 激活后的输入 x 原地相加
                    tmp = self.linear(x)
                    tmp += self.relu(x)
                    return tmp  # 返回结果
                else:
                    # 否则，分别计算线性层和 ReLU 激活后的结果，再将它们相加
                    tmp = self.linear(x)
                    return tmp + self.relu(x)  # 返回结果
            # 如果线性层在右侧
            elif self.linear_pos == NodePosType.right:
                # 如果使用原地加法
                if self.inplace_add:
                    # 先计算 ReLU 激活后的输入，然后与线性层的输出原地相加
                    tmp = self.relu(x)
                    tmp += self.linear(x)
                    return tmp  # 返回结果
                else:
                    # 否则，将 ReLU 激活后的输入和线性层的输出相加后返回
                    return self.relu(x) + self.linear(x)  # 返回结果
            # 如果线性层在两侧
            elif self.linear_pos == NodePosType.both:
                # 如果使用原地加法
                if self.inplace_add:
                    # 先计算第一个线性层的输出，然后与第二个线性层的输出原地相加
                    tmp = self.linear(x)
                    tmp += self.linear2(x)
                    return tmp  # 返回结果
                else:
                    # 否则，将第一个线性层的输出和第二个线性层的输出相加后返回
                    return self.linear(x) + self.linear2(x)  # 返回结果
    class LinearAddReLUModule(torch.nn.Module):
        """Implements a module with linear, add, and ReLU operations based on specified positions."""
    
        def __init__(
            self,
            inplace_add: bool = False,
            linear_pos: NodePosType = NodePosType.left,
            inplace_relu: bool = False,
            use_bias: bool = False,
        ) -> None:
            super().__init__()
            # Define the first linear transformation
            self.linear = torch.nn.Linear(
                in_features=16, out_features=16, bias=use_bias
            )
            # Define the second linear transformation
            self.linear2 = torch.nn.Linear(
                in_features=16, out_features=16, bias=use_bias
            )
            # ReLU activation function
            self.relu = nn.ReLU()
            # Whether to perform inplace addition
            self.inplace_add = inplace_add
            # Specifies where the linear transformation occurs
            self.linear_pos = linear_pos
            # ReLU activation function with optional inplace parameter
            self.relu2 = nn.ReLU(inplace=inplace_relu)
    
        def forward(self, x):
            if self.linear_pos == NodePosType.left:
                if self.inplace_add:
                    # Apply first linear transformation followed by inplace addition and ReLU
                    tmp = self.linear(x)
                    tmp += self.relu(x)
                    return self.relu2(tmp)
                else:
                    # Apply first linear transformation, then add and apply ReLU
                    tmp = self.linear(x)
                    return self.relu2(tmp + self.relu(x))
            elif self.linear_pos == NodePosType.right:
                if self.inplace_add:
                    # Apply ReLU first, then perform inplace addition with second linear transformation
                    tmp = self.relu(x)
                    tmp += self.linear(x)
                    return self.relu2(tmp)
                else:
                    # Apply ReLU, then add the result of first linear transformation and second linear transformation
                    return self.relu2(self.relu(x) + self.linear(x))
            elif self.linear_pos == NodePosType.both:
                if self.inplace_add:
                    # Apply first linear transformation followed by inplace addition with second linear transformation
                    tmp = self.linear(x)
                    tmp += self.linear2(x)
                    return self.relu2(tmp)
                else:
                    # Add the results of both linear transformations and apply ReLU
                    return self.relu2(self.linear(x) + self.linear2(x))
    
    
    class SerialsLinearAddReLUModule(torch.nn.Module):
        """Implements a module consisting of serial linear, add, and ReLU operations."""
    
        def __init__(
            self,
        ) -> None:
            super().__init__()
            # Define four linear transformations
            self.linear = torch.nn.Linear(in_features=16, out_features=16, bias=True)
            self.linear2 = torch.nn.Linear(in_features=16, out_features=16, bias=True)
            self.linear3 = torch.nn.Linear(in_features=16, out_features=16, bias=True)
            self.linear4 = torch.nn.Linear(in_features=16, out_features=16, bias=True)
            # ReLU activation functions
            self.relu = nn.ReLU()
            self.relu2 = nn.ReLU()
    
        def forward(self, x):
            # Perform the sequence of operations: linear, ReLU, add, and ReLU
            x1 = self.linear(x)
            res1 = self.relu(self.linear2(x1) + self.linear3(x1))
            res2 = self.relu2(self.linear4(res1) + res1)
            return res2
    # 定义一个名为 LinearAddModule2 的 PyTorch 模块类
    class LinearAddModule2(torch.nn.Module):
        # 初始化函数，设置是否原地相加的标志 inplace_add
        def __init__(
            self,
            inplace_add: bool = False,
        ) -> None:
            super().__init__()
            # 创建一个线性层，输入和输出特征数都是 16，带有偏置
            self.linear = torch.nn.Linear(in_features=16, out_features=16, bias=True)
            # 再创建一个相同的线性层，用于后续的加法操作
            self.linear2 = torch.nn.Linear(in_features=16, out_features=16, bias=True)
            # 保存是否原地相加的标志
            self.inplace_add = inplace_add
    
        # 前向传播函数，接收输入 x
        def forward(self, x):
            # 如果指定了原地相加
            if self.inplace_add:
                # 计算第一个线性层的输出
                tmp = self.linear(x)
                # 将第二个线性层的输出原地加到第一个线性层的输出上
                tmp += self.linear2(tmp)
                # 返回原地相加后的结果
                return tmp
            else:
                # 计算第一个线性层的输出
                tmp = self.linear(x)
                # 将第二个线性层的输出加到第一个线性层的输出上，并返回结果
                return tmp + self.linear2(tmp)
    
    # 定义一个名为 Conv2dAddModule2 的 PyTorch 模块类
    class Conv2dAddModule2(torch.nn.Module):
        # 初始化函数，设置是否原地相加的标志 inplace_add
        def __init__(
            self,
            inplace_add: bool = False,
        ) -> None:
            super().__init__()
            # 创建一个二维卷积层，输入和输出通道数都是 3，卷积核大小为 3x3，步长为 1，填充为 1
            self.conv = torch.nn.Conv2d(
                in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1
            )
            # 创建另一个相同设置的二维卷积层，用于后续的加法操作
            self.conv2 = torch.nn.Conv2d(
                in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=1
            )
            # 保存是否原地相加的标志
            self.inplace_add = inplace_add
            # 创建一个批归一化层，通道数为 3
            self.bn = torch.nn.BatchNorm2d(3)
            # 创建另一个相同设置的批归一化层，用于后续的加法操作
            self.bn2 = torch.nn.BatchNorm2d(3)
    
        # 前向传播函数，接收输入 x
        def forward(self, x):
            # 如果指定了原地相加
            if self.inplace_add:
                # 对输入先进行第一个卷积和批归一化操作
                tmp = self.bn(self.conv(x))
                # 将第二个卷积和批归一化的输出原地加到第一个卷积和批归一化的输出上
                tmp += self.bn2(self.conv2(tmp))
                # 返回原地相加后的结果
                return tmp
            else:
                # 对输入先进行第一个卷积和批归一化操作
                tmp = self.bn(self.conv(x))
                # 将第二个卷积和批归一化的输出加到第一个卷积和批归一化的输出上，并返回结果
                return tmp + self.bn2(self.conv2(tmp))
    # 定义一个自注意力模块的 PyTorch 模型
    class SelfAttnLikeModule(torch.nn.Module):
        def __init__(
            self,
            input_dim,
            transpose_for_score=False,
            num_attention_heads=None,
            attention_head_size=None,
        ) -> None:
            super().__init__()
            self.input_dim = input_dim
            # 初始化查询、键、值投影层，无偏置
            self.q_proj = nn.Linear(input_dim, input_dim, bias=False)
            self.k_proj = nn.Linear(input_dim, input_dim, bias=False)
            self.v_proj = nn.Linear(input_dim, input_dim, bias=False)
            # 使用 softmax 函数进行归一化
            self.softmax = nn.Softmax(dim=-1)
            self.transpose_for_score = transpose_for_score
            if self.transpose_for_score:
                assert num_attention_heads is not None
                assert attention_head_size is not None
                # 设置注意力头的数量和每个头的大小
                self.num_attention_heads = num_attention_heads
                self.attention_head_size = attention_head_size

        # 将输入张量转置以便计算注意力分数
        def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
            new_x_shape = x.size()[:-1] + (
                self.num_attention_heads,
                self.attention_head_size,
            )
            x = x.view(new_x_shape)
            return x.permute(0, 2, 1, 3)

        # 前向传播函数
        def forward(self, x):
            # 对输入进行查询、键、值投影
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
            if self.transpose_for_score:
                # 如果需要，为查询、键、值张量进行维度转置以便计算注意力分数
                q = self.transpose_for_scores(q)
                k = self.transpose_for_scores(k)
                v = self.transpose_for_scores(v)
            # 计算注意力分数
            scores = torch.matmul(q, k.transpose(-1, -2)) / (self.input_dim**0.5)
            # 对注意力分数进行 softmax 归一化
            attention = self.softmax(scores)
            # 加权求和得到加权值
            weighted = torch.matmul(attention, v)
            return weighted
class X86InductorQuantTestCase(QuantizationTestCase):
    def _test_quantizer(
        self,
        model,
        example_inputs,
        quantizer,
        expected_node_occurrence,
        expected_node_list=None,
        is_qat=False,
        debug=False,
    ):
        # 如果是量化训练模式，将模型设置为训练模式；否则设置为评估模式
        m_eager = model.train() if is_qat else model.eval()

        # 创建模型的深层拷贝，用于捕获自动微分图
        m = copy.deepcopy(m_eager)
        m = capture_pre_autograd_graph(
            m,
            example_inputs,
        )

        # 如果是量化训练模式，直接使用 m；否则深拷贝 m
        export_model = m if is_qat else copy.deepcopy(m)
        # 根据是否是量化训练模式，选择量化准备函数
        m = prepare_qat_pt2e(m, quantizer) if is_qat else prepare_pt2e(m, quantizer)
        # 运行模型，用于校准
        m(*example_inputs)
        # 深拷贝校准后的模型
        prepare_model = copy.deepcopy(m)
        # 执行转换，将 PyTorch 模型转换为量化引擎的模型
        m = convert_pt2e(m)
        # 深拷贝转换后的模型
        convert_model = copy.deepcopy(m)
        if debug:
            # 打印可读的转换模型信息，如果 debug 标志为真
            convert_model.print_readable(True)
        # 使用示例输入运行转换后的模型，获取节点出现次数
        pt2_quant_output = m(*example_inputs)
        # 将期望的节点出现次数字典转换为适合量化引擎的格式
        node_occurrence = {
            ns.call_function(k): v for k, v in expected_node_occurrence.items()
        }
        # 如果未提供期望的节点列表，则默认为空列表
        if expected_node_list is None:
            expected_node_list = []
        # 将期望的节点列表转换为适合量化引擎的格式
        node_list = [ns.call_function(n) for n in expected_node_list]
        # 调用父类方法，检查模型的节点
        self.checkGraphModuleNodes(
            m, expected_node_occurrence=node_occurrence, expected_node_list=node_list
        )
        # 返回导出模型、校准模型、转换模型
        return export_model, prepare_model, convert_model


@skipIfNoInductorSupport
class TestQuantizePT2EX86Inductor(X86InductorQuantTestCase):
    @skipIfNoX86
    def test_conv2d(self):
        """
        Test pattern of single conv2d with X86InductorQuantizer.
        """
        # 使用 x86 量化引擎进行覆盖
        with override_quantized_engine("x86"), torch.no_grad():
            # 创建并评估单个 conv2d 的测试模块
            m = TestHelperModules.SingleConv2dModule().eval()
            example_inputs = (torch.randn(2, 3, 16, 16),)
            # 创建 x86 量化器并设置全局量化配置
            quantizer = X86InductorQuantizer().set_global(
                xiq.get_default_x86_inductor_quantization_config()
            )
            # 定义期望的节点出现次数
            node_occurrence = {
                # 输入和卷积权重的量化操作
                torch.ops.quantized_decomposed.quantize_per_tensor.default: 1,
                torch.ops.quantized_decomposed.dequantize_per_tensor.default: 1,
                # 注意：权重的量化操作已经在编译时优化
                torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
                torch.ops.quantized_decomposed.dequantize_per_channel.default: 1,
            }
            # 定义期望的节点列表
            node_list = [
                torch.ops.quantized_decomposed.quantize_per_tensor.default,
                torch.ops.quantized_decomposed.dequantize_per_tensor.default,
                torch.ops.aten.conv2d.default,
            ]
            # 调用 _test_quantizer 方法，测试量化器功能
            self._test_quantizer(
                m,
                example_inputs,
                quantizer,
                node_occurrence,
                node_list,
            )

    @skipIfNoX86
    @skipIfNoX86
    # 定义名为 test_conv2d_binary 的测试方法，用于测试带有二进制后操作（如加法）的 conv2d 模式，
    # 使用 X86InductorQuantizer 进行量化。
    def test_conv2d_binary(self):
        """
        Test pattern of conv2d with binary post ops (such as add) with X86InductorQuantizer.
        Currently, only add as binary post op is supported.
        """
        # 定义 conv2d_type_list 列表，包含 NodePosType.left 和 NodePosType.both 两种类型
        conv2d_type_list = [NodePosType.left, NodePosType.both]
        
        # 定义示例输入 example_inputs，包含一个 2x3x6x6 的随机张量
        example_inputs = (torch.randn(2, 3, 6, 6),)
        
        # 创建 X86InductorQuantizer 实例 quantizer，并设置全局量化配置
        quantizer = X86InductorQuantizer().set_global(
            xiq.get_default_x86_inductor_quantization_config()
        )
        
        # 使用 "x86" 引擎进行量化，并关闭 Torch 的梯度计算
        with override_quantized_engine("x86"), torch.no_grad():
            # 遍历 conv2d_type_list 中的每种 conv2d 类型
            for conv2d_type in conv2d_type_list:
                # 创建一个测试用的 Conv2dAddModule 实例 m，设置为评估模式
                m = TestHelperModules.Conv2dAddModule(conv2d_type=conv2d_type).eval()
                
                # 根据 conv2d_type 类型设置 node_occurrence 字典，描述节点的出现次数
                if conv2d_type != NodePosType.both:
                    node_occurrence = {
                        # conv 的输入和权重各有一个节点
                        # add 的额外输入节点一个
                        torch.ops.quantized_decomposed.quantize_per_tensor.default: 2,
                        torch.ops.quantized_decomposed.dequantize_per_tensor.default: 2,
                        # 权重的量化为每通道的常量传播
                        torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
                        torch.ops.quantized_decomposed.dequantize_per_channel.default: 1,
                    }
                else:
                    node_occurrence = {
                        # conv 的输入节点一个
                        # 另一个 conv 的输入节点一个
                        # 两个 conv 将共享相同的输入量化/反量化节点
                        # add 的额外输入节点一个
                        torch.ops.quantized_decomposed.quantize_per_tensor.default: 2,
                        torch.ops.quantized_decomposed.dequantize_per_tensor.default: 3,
                        # 权重的量化为每通道的常量传播
                        torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
                        torch.ops.quantized_decomposed.dequantize_per_channel.default: 2,
                    }
                
                # 定义 node_list 列表，包含量化和反量化节点以及 conv2d 和 add 操作节点
                node_list = [
                    torch.ops.quantized_decomposed.quantize_per_tensor.default,
                    torch.ops.quantized_decomposed.dequantize_per_tensor.default,
                    torch.ops.aten.conv2d.default,
                    torch.ops.aten.add.Tensor,
                ]
                
                # 调用 _test_quantizer 方法，传入模型 m、示例输入 example_inputs、量化器 quantizer、
                # 节点出现次数 node_occurrence 和节点列表 node_list 进行量化测试
                self._test_quantizer(
                    m,
                    example_inputs,
                    quantizer,
                    node_occurrence,
                    node_list,
                )

    # 在没有 X86 支持的情况下跳过当前测试
    @skipIfNoX86
    def test_conv2d_binary2(self):
        """
        Test Pattern:
            tmp = conv2d_1(x)
            tmp2 = conv2d_2(tmp)
            return tmp + tmp2
        Since conv2d_1 has 2 users, we should annotate conv2d_2 for binary fusion instead of conv2d_1
        """
        example_inputs = (torch.randn(2, 3, 6, 6),)
        quantizer = X86InductorQuantizer().set_global(
            xiq.get_default_x86_inductor_quantization_config()
        )
        inplace_add_list = [True, False]
        with override_quantized_engine("x86"), torch.no_grad():
            for inplace_add in inplace_add_list:
                # Create an instance of Conv2dAddModule2 for testing, setting inplace_add mode
                m = TestHelperModules.Conv2dAddModule2(inplace_add=inplace_add).eval()
                
                # Define the expected occurrences of specific nodes during quantization
                node_occurrence = {
                    torch.ops.quantized_decomposed.quantize_per_tensor.default: 2,
                    torch.ops.quantized_decomposed.dequantize_per_tensor.default: 3,
                    # quantize_per_channel for weights are const propagated
                    torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
                    torch.ops.quantized_decomposed.dequantize_per_channel.default: 2,
                }
                
                # Define the sequence of nodes expected during quantization
                node_list = [
                    torch.ops.quantized_decomposed.quantize_per_tensor.default,
                    torch.ops.quantized_decomposed.dequantize_per_tensor.default,
                    torch.ops.aten.conv2d.default,
                    torch.ops.quantized_decomposed.quantize_per_tensor.default,
                    (
                        torch.ops.aten.add_.Tensor  # Use in-place add operation if inplace_add is True
                        if inplace_add
                        else torch.ops.aten.add.Tensor  # Use regular add operation if inplace_add is False
                    ),
                ]
                
                # Call the _test_quantizer method to perform quantization testing
                self._test_quantizer(
                    m,
                    example_inputs,
                    quantizer,
                    node_occurrence,
                    node_list,
                )

    @skipIfNoX86


注释说明了每一行代码的作用，包括创建测试模块实例、定义预期的节点出现次数和顺序，以及调用量化测试方法进行测试。
    # 定义测试函数，用于测试 conv2d 与二元加法（add）和一元 relu 后操作的模式，
    # 使用 X86InductorQuantizer 进行量化。
    # 目前仅支持 add 作为二元后操作和 relu 作为一元后操作。
    def test_conv2d_binary_unary(self):
        """
        Test pattern of conv2d with binary + unary post ops (such as add + relu) with X86InductorQuantizer.
        Currently, only add as binary post op and relu as unary post op are supported.
        """
        # 定义 conv2d 的类型列表
        conv2d_type_list = [NodePosType.left, NodePosType.both]
        # 定义示例输入数据
        example_inputs = (torch.randn(2, 3, 6, 6),)
        # 创建 X86InductorQuantizer 实例，并设置全局量化配置
        quantizer = X86InductorQuantizer().set_global(
            xiq.get_default_x86_inductor_quantization_config()
        )
        
        # 使用 "x86" 引擎进行量化，进入无梯度计算环境
        with override_quantized_engine("x86"), torch.no_grad():
            # 遍历 conv2d 类型列表
            for conv2d_type in conv2d_type_list:
                # 创建 Conv2dAddReLUModule 实例，设置为评估模式
                m = TestHelperModules.Conv2dAddReLUModule(
                    conv2d_type=conv2d_type,
                ).eval()
                # 根据 conv2d 类型设置节点出现次数的字典
                if conv2d_type != NodePosType.both:
                    node_occurrence = {
                        # 输入 conv 的量化和反量化操作各一次
                        torch.ops.quantized_decomposed.quantize_per_tensor.default: 2,
                        torch.ops.quantized_decomposed.dequantize_per_tensor.default: 2,
                        # 注意：权重的量化操作是常量传播的
                        torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
                        torch.ops.quantized_decomposed.dequantize_per_channel.default: 1,
                    }
                else:
                    node_occurrence = {
                        # 第一个 conv 输入的量化和反量化操作各一次
                        # 第二个 conv 输入的量化和反量化操作各一次
                        # 两个 conv 共享相同的输入量化和反量化操作
                        torch.ops.quantized_decomposed.quantize_per_tensor.default: 2,
                        torch.ops.quantized_decomposed.dequantize_per_tensor.default: 3,
                        # 注意：权重的量化操作是常量传播的
                        torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
                        torch.ops.quantized_decomposed.dequantize_per_channel.default: 2,
                    }
                
                # 定义节点列表，包含量化、反量化、conv2d 和 add 操作
                node_list = [
                    torch.ops.quantized_decomposed.quantize_per_tensor.default,
                    torch.ops.quantized_decomposed.dequantize_per_tensor.default,
                    torch.ops.aten.conv2d.default,
                    torch.ops.aten.add.Tensor,
                ]
                
                # 调用测试量化器方法，传入模块 m、示例输入、量化器、节点出现次数字典和节点列表
                self._test_quantizer(
                    m,
                    example_inputs,
                    quantizer,
                    node_occurrence,
                    node_list,
                )

    # 标记为在没有 X86 环境时跳过测试
    @skipIfNoX86
    # 定义一个测试函数，用于测试连续的 conv2d、add 和 relu 操作的模式，使用 X86InductorQuantizer
    def test_conv2d_serials_binary_unary(self):
        """
        Test pattern of 2 following up conv2d add relu with X86InductorQuantizer.
        """
        # 使用 X86 引擎进行量化，并关闭梯度计算
        with override_quantized_engine("x86"), torch.no_grad():
            # 创建一个测试模块，包含连续的 conv2d 和 relu 操作
            m = TestHelperModules.SerialsConv2dAddReLUModule().eval()
            # 创建一个示例输入
            example_inputs = (torch.randn(2, 3, 16, 16),)
            # 创建一个 X86InductorQuantizer 实例，并设置全局量化配置
            quantizer = X86InductorQuantizer().set_global(
                xiq.get_default_x86_inductor_quantization_config()
            )
            # 定义节点出现次数的字典
            node_occurrence = {
                torch.ops.quantized_decomposed.quantize_per_tensor.default: 4,
                torch.ops.quantized_decomposed.dequantize_per_tensor.default: 6,
                # 权重的 quantize_per_channel 操作被常量传播
                torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
                torch.ops.quantized_decomposed.dequantize_per_channel.default: 4,
            }
            # 定义节点列表
            node_list = [
                torch.ops.quantized_decomposed.quantize_per_tensor.default,
                torch.ops.quantized_decomposed.dequantize_per_tensor.default,
                torch.ops.aten.conv2d.default,
                torch.ops.quantized_decomposed.quantize_per_tensor.default,
                torch.ops.quantized_decomposed.dequantize_per_tensor.default,
                torch.ops.aten.conv2d.default,
                torch.ops.aten.conv2d.default,
                torch.ops.aten.add.Tensor,
                torch.ops.aten.relu.default,
            ]
            # 调用测试量化器函数，传入模块、示例输入、量化器、节点出现次数和节点列表
            self._test_quantizer(
                m,
                example_inputs,
                quantizer,
                node_occurrence,
                node_list,
            )
    # 辅助函数，用于测试单个操作的共享观察器配方
    def _single_op_share_observer_recipe_test_helper(self, m, x, single_op):
        # 创建 X86InductorQuantizer 实例，并设置全局的默认 X86Inductor 量化配置
        quantizer = X86InductorQuantizer().set_global(
            xiq.get_default_x86_inductor_quantization_config()
        )
        # 准备示例输入数据
        example_inputs = (x,)
        # 定义节点出现次数的字典
        node_occurrence = {
            # 卷积的输入和权重各一个，最大池化的输入/输出各两个
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 3,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 3,
            # 权重的通道量化被常量传播
            torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: 1,
        }
        # 定义节点列表
        node_list = [
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.conv2d.default,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            single_op,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
        ]
        # 测试量化器
        _, prepare_model, _ = self._test_quantizer(
            m,
            example_inputs,
            quantizer,
            node_occurrence,
            node_list,
        )
        # 检查 Maxpool2d 在输入和输出处有共享观察器
        for node in prepare_model.graph.nodes:
            if node.op == "call_function" and node.target is single_op:
                single_op_node = node
                input_obs_of_single_op = getattr(
                    prepare_model, single_op_node.args[0].target
                )
                output_obs_of_single_op = getattr(
                    prepare_model, next(iter(single_op_node.users)).target
                )
            elif (
                node.op == "call_function"
                and node.target is torch.ops.aten.conv2d.default
            ):
                conv_node = node
                input_obs_of_conv = getattr(prepare_model, conv_node.args[0].target)
        # 断言输入观察器为 ObserverBase 类型
        self.assertTrue(isinstance(input_obs_of_single_op, ObserverBase))
        # 断言输出观察器为 ObserverBase 类型
        self.assertTrue(isinstance(output_obs_of_single_op, ObserverBase))
        # 断言卷积输入观察器为 ObserverBase 类型
        self.assertTrue(isinstance(input_obs_of_conv, ObserverBase))
        # 断言单个操作的输入观察器与输出观察器相同
        self.assertTrue(input_obs_of_single_op is output_obs_of_single_op)
        # 断言单个操作的输入观察器与卷积输入观察器不同
        self.assertTrue(input_obs_of_single_op is not input_obs_of_conv)

    # 如果没有 X86，跳过测试
    @skipIfNoX86
    def test_maxpool2d_recipe(self):
        r"""
        Test pattern: int8_in_int8_out_ops(maxpool) - non_quantizable op(pow)
        Since maxpool is a int8_in_int8_out_op, there is obs between maxpool and pow.
        """
        # 使用单操作共享观察器测试辅助函数，针对包含 nn.MaxPool2d(1, 1) 的 Conv2dSingleOpPowModule 进行评估
        self._single_op_share_observer_recipe_test_helper(
            TestHelperModules.Conv2dSingleOpPowModule(nn.MaxPool2d(1, 1)).eval(),
            # 创建一个形状为 (1, 2, 14, 14) 的随机张量作为输入
            torch.rand(1, 2, 14, 14),
            # 调用 torch.ops.aten.max_pool2d.default 运算
            torch.ops.aten.max_pool2d.default,
        )

    @skipIfNoX86
    def test_adaptive_avg_pool2d_recipe(self):
        r"""
        Test pattern: int8_in_int8_out_ops(adaptive_avg_pool2d) - non_quantizable op(pow)
        Since adaptive_avg_pool2d is a int8_in_int8_out_op, there is obs between adaptive_avg_pool2d and pow.
        """
        # 使用单操作共享观察器测试辅助函数，针对包含 nn.AdaptiveAvgPool2d((1, 1)) 的 Conv2dSingleOpPowModule 进行评估
        self._single_op_share_observer_recipe_test_helper(
            TestHelperModules.Conv2dSingleOpPowModule(
                nn.AdaptiveAvgPool2d((1, 1))
            ).eval(),
            # 创建一个形状为 (1, 2, 14, 14) 的随机张量作为输入
            torch.rand(1, 2, 14, 14),
            # 调用 torch.ops.aten.adaptive_avg_pool2d.default 运算
            torch.ops.aten.adaptive_avg_pool2d.default,
        )

    @skipIfNoX86
    def test_flatten_recipe(self):
        r"""
        Test pattern: int8_in_int8_out_ops(flatten) - non_quantizable op(pow)
        Since flatten is a int8_in_int8_out_op, there is obs between flatten and pow.
        """
        # 使用单操作共享观察器测试辅助函数，针对包含 lambda 函数进行 torch.flatten(x, 1) 的 Conv2dSingleOpPowModule 进行评估
        self._single_op_share_observer_recipe_test_helper(
            TestHelperModules.Conv2dSingleOpPowModule(
                lambda x: torch.flatten(x, 1)
            ).eval(),
            # 创建一个形状为 (1, 2, 14, 14) 的随机张量作为输入
            torch.rand(1, 2, 14, 14),
            # 调用 torch.ops.aten.flatten.using_ints 运算
            torch.ops.aten.flatten.using_ints,
        )
    def test_cat_recipe_same_inputs(self):
        r"""
        Test pattern: conv -> cat([input0, input0])
        Since cat has 2 input node of same tensor, they should also be with same observer.
        """
        # 创建测试模块实例，并设置为评估模式
        m = TestHelperModules.Conv2dCatSameInputs().eval()
        # 创建具有指定形状和内存格式的随机张量 x
        x = torch.randn(16, 3, 16, 16).contiguous(memory_format=torch.channels_last)
        # 创建 X86InductorQuantizer 实例
        quantizer = X86InductorQuantizer().set_global(
            xiq.get_default_x86_inductor_quantization_config()
        )
        # 定义示例输入元组
        example_inputs = (x,)
        # 定义节点出现次数的字典
        node_occurrence = {
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 3,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 3,
            # 权重的 quantize_per_channel 被常量传播
            torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: 1,
        }
        # 定义节点列表
        node_list = [
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.conv2d.default,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.cat.default,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
        ]
        # 调用 _test_quantizer 方法，获取返回的三元组结果
        _, prepare_model, _ = self._test_quantizer(
            m,
            example_inputs,
            quantizer,
            node_occurrence,
            node_list,
        )
        # 检查 Cat 操作在输入和输出处共享观察器
        for node in prepare_model.graph.nodes:
            if node.op == "call_function" and node.target == torch.ops.aten.cat.default:
                # 获取 cat 操作的输入观察器
                cat_act_obs0 = getattr(prepare_model, node.args[0][0].target)
                cat_act_obs1 = getattr(prepare_model, node.args[0][1].target)
                # 获取 cat 操作的输出观察器
                cat_out_obs = getattr(prepare_model, next(iter(node.users)).target)
        # 断言观察器类型为 ObserverBase
        self.assertTrue(isinstance(cat_act_obs0, ObserverBase))
        self.assertTrue(isinstance(cat_act_obs1, ObserverBase))
        self.assertTrue(isinstance(cat_out_obs, ObserverBase))
        # 断言 cat 操作的两个输入观察器和输出观察器是同一个对象
        self.assertTrue(cat_act_obs0 is cat_act_obs1)
        self.assertTrue(cat_act_obs0 is cat_out_obs)

    @skipIfNoX86
    def test_cat_recipe_single_input(self):
        r"""
        Test pattern: conv -> cat([input0,])
        Since cat has 1 input node, they should also be with same observer.
        """
        # 创建测试模型实例，设置为评估模式
        m = TestHelperModules.Conv2dCatSingleInput().eval()
        # 生成随机输入张量 x，形状为 (16, 3, 16, 16)，使用通道最后内存格式
        x = torch.randn(16, 3, 16, 16).contiguous(memory_format=torch.channels_last)
        # 创建 X86InductorQuantizer 实例
        quantizer = X86InductorQuantizer().set_global(
            xiq.get_default_x86_inductor_quantization_config()
        )
        # 定义示例输入元组
        example_inputs = (x,)
        # 定义节点出现次数的字典
        node_occurrence = {
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 3,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 3,
            # 权重的 quantize_per_channel 在常量传播中
            torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: 1,
        }
        # 定义节点列表
        node_list = [
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.conv2d.default,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.cat.default,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
        ]
        # 执行量化器测试，获取预处理模型
        _, prepare_model, _ = self._test_quantizer(
            m,
            example_inputs,
            quantizer,
            node_occurrence,
            node_list,
        )
        # 检查 cat 函数在输入和输出处是否共享观察器
        for node in prepare_model.graph.nodes:
            if node.op == "call_function" and node.target == torch.ops.aten.cat.default:
                # 获取第一个输入节点的观察器
                cat_act_obs0 = getattr(prepare_model, node.args[0][0].target)
                # 获取 cat 函数输出的观察器
                cat_out_obs = getattr(prepare_model, next(iter(node.users)).target)
        # 断言第一个输入节点的观察器和输出的观察器是 ObserverBase 类型
        self.assertTrue(isinstance(cat_act_obs0, ObserverBase))
        self.assertTrue(isinstance(cat_out_obs, ObserverBase))
        # 断言第一个输入节点的观察器和输出的观察器是同一个对象
        self.assertTrue(cat_act_obs0 is cat_out_obs)

    @skipIfNoX86
    def test_avg_pool2d_recipe(self):
        r"""
        Test pattern: conv -> AvgPool2d
        Since AvgPool2d is a int8_in_int8_out_op, the inputs and outputs should with same observer.
        """
        # 创建 Conv2dAvgPool2d 实例并设置为评估模式
        m = TestHelperModules.Conv2dAvgPool2d().eval()
        # 创建随机张量作为输入数据，格式为通道最后
        x = torch.randn(16, 3, 16, 16).contiguous(memory_format=torch.channels_last)
        # 创建 X86InductorQuantizer 实例，并设置默认的 x86 感知量化配置
        quantizer = X86InductorQuantizer().set_global(
            xiq.get_default_x86_inductor_quantization_config()
        )
        # 定义示例输入元组
        example_inputs = (x,)
        # 定义节点出现次数的字典，指定特定的量化节点
        node_occurrence = {
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 3,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 3,
            # 权重的 quantize_per_channel 是常量传播的
            torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: 1,
        }
        # 定义节点列表，列出了量化过程中的各种操作
        node_list = [
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.conv2d.default,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.avg_pool2d.default,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
        ]
        # 执行量化器的测试，返回量化后的模型和准备好的模型
        _, prepare_model, _ = self._test_quantizer(
            m,
            example_inputs,
            quantizer,
            node_occurrence,
            node_list,
        )
        # 遍历准备好的模型的节点
        for node in prepare_model.graph.nodes:
            # 查找 AvgPool2d 操作节点
            if (
                node.op == "call_function"
                and node.target is torch.ops.aten.avg_pool2d.default
            ):
                avgpool_node = node
                # 获取 AvgPool2d 操作节点的输入观察器
                input_obs_of_avgpool = getattr(
                    prepare_model, avgpool_node.args[0].target
                )
                # 获取 AvgPool2d 操作节点的输出观察器
                output_obs_of_avgpool = getattr(
                    prepare_model, next(iter(avgpool_node.users)).target
                )
            # 查找 Conv2d 操作节点
            elif (
                node.op == "call_function"
                and node.target is torch.ops.aten.conv2d.default
            ):
                conv_node = node
                # 获取 Conv2d 操作节点的输出观察器
                output_obs_of_conv = getattr(
                    prepare_model, next(iter(conv_node.users)).target
                )
        # 断言各观察器是 ObserverBase 类的实例
        self.assertTrue(isinstance(input_obs_of_avgpool, ObserverBase))
        self.assertTrue(isinstance(output_obs_of_avgpool, ObserverBase))
        self.assertTrue(isinstance(output_obs_of_conv, ObserverBase))
        # 断言 AvgPool2d 操作节点的输入输出观察器是相同的
        self.assertTrue(input_obs_of_avgpool is output_obs_of_avgpool)
        # 断言 Conv2d 操作节点的输出观察器和 AvgPool2d 操作节点的输入观察器是相同的
        self.assertTrue(input_obs_of_avgpool is output_obs_of_conv)

    @skipIfNoX86
    def test_linear(self):
        """
        Test pattern of single linear with X86InductorQuantizer.
        """
        # 使用 "x86" 引擎进行量化测试，关闭梯度计算
        with override_quantized_engine("x86"), torch.no_grad():
            # 对于每种是否使用偏置的情况进行测试
            for use_bias in [True, False]:
                # 创建一个测试用的单线性模型，并设置为评估模式
                m = TestHelperModules.SingleLinearModule(use_bias).eval()
                # 创建一个示例输入
                example_inputs = (torch.randn(2, 4),)
                # 创建 X86InductorQuantizer 实例，并设置全局量化配置
                quantizer = X86InductorQuantizer().set_global(
                    xiq.get_default_x86_inductor_quantization_config()
                )
                # 定义节点出现次数的期望
                node_occurrence = {
                    # 输入和权重的量化节点，以及输出的反量化节点各出现一次
                    torch.ops.quantized_decomposed.quantize_per_tensor.default: 1,
                    torch.ops.quantized_decomposed.dequantize_per_tensor.default: 1,
                    # 权重的通道量化节点被常量传播，期望出现零次
                    torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
                    torch.ops.quantized_decomposed.dequantize_per_channel.default: 1,
                }
                # 定义节点列表，包括量化、反量化以及线性运算
                node_list = [
                    torch.ops.quantized_decomposed.quantize_per_tensor.default,
                    torch.ops.quantized_decomposed.dequantize_per_tensor.default,
                    torch.ops.aten.linear.default,
                ]
                # 调用 _test_quantizer 方法，测试量化器的功能
                self._test_quantizer(
                    m,
                    example_inputs,
                    quantizer,
                    node_occurrence,
                    node_list,
                )

    @skipIfNoX86
    # 定义一个测试方法，用于测试带有一元后操作符（如 relu）的线性模式，使用 X86InductorQuantizer 进行量化。
    def test_linear_unary(self):
        """
        Test pattern of linear with unary post ops (e.g. relu) with X86InductorQuantizer.
        """
        # 定义三个列表，分别存储是否使用偏置、是否原地操作、以及使用的后操作函数（仅测试两种以节省时间）
        use_bias_list = [True, False]
        inplace_list = [True, False]
        postop_list = [nn.ReLU, nn.LeakyReLU]  # only test two to save time
        
        # 生成所有可能的组合
        cases = itertools.product(use_bias_list, inplace_list, postop_list)
        
        # 定义后操作函数映射表
        post_op_map = {
            nn.ReLU: [torch.ops.aten.relu_.default, torch.ops.aten.relu.default],
            nn.LeakyReLU: [
                torch.ops.aten.leaky_relu_.default,
                torch.ops.aten.leaky_relu.default,
            ],
        }
        
        # 设置 quantized_engine 为 x86，并禁用梯度计算
        with override_quantized_engine("x86"), torch.no_grad():
            # 遍历所有测试用例
            for use_bias, inplace, postop in cases:
                # 创建测试线性模块，设定是否使用偏置、后操作函数类型及是否原地操作
                m = TestHelperModules.LinearUnaryModule(
                    use_bias=use_bias, postop=postop, inplace_postop=inplace
                ).eval()
                
                # 准备示例输入数据
                example_inputs = (torch.randn(2, 4),)
                
                # 创建 X86InductorQuantizer 实例，并设置全局量化配置
                quantizer = X86InductorQuantizer().set_global(
                    xiq.get_default_x86_inductor_quantization_config()
                )
                
                # 定义节点发生次数字典，用于验证量化过程中节点的数量
                node_occurrence = {
                    # 输入和卷积权重各一次，relu 输出一次
                    torch.ops.quantized_decomposed.quantize_per_tensor.default: 1,
                    torch.ops.quantized_decomposed.dequantize_per_tensor.default: 1,
                    # 权重的通道量化为常数传播
                    torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
                    torch.ops.quantized_decomposed.dequantize_per_channel.default: 1,
                }
                
                # 定义节点列表，描述量化过程中的操作顺序
                node_list = [
                    torch.ops.quantized_decomposed.quantize_per_tensor.default,
                    torch.ops.quantized_decomposed.dequantize_per_tensor.default,
                    torch.ops.aten.linear.default,
                    post_op_map[postop][0 if inplace else 1],
                ]
                
                # 调用测试方法验证量化器行为是否符合预期
                self._test_quantizer(
                    m,
                    example_inputs,
                    quantizer,
                    node_occurrence,
                    node_list,
                )

    # 跳过测试，如果没有 x86 支持
    @skipIfNoX86
    def test_linear_unary_gelu(self):
        """
        Test pattern of linear with unary post ops (e.g. gelu) with X86InductorQuantizer.
        """
        # 定义使用偏置的列表和后操作函数为 GELU
        use_bias_list = [True, False]
        postop = nn.GELU
        # 后操作算法为 "none" 或 "tanh"
        post_op_algorithm = ["none", "tanh"]
        # 生成所有可能的测试用例组合
        cases = itertools.product(use_bias_list, post_op_algorithm)
        # 使用 x86 引擎进行量化计算，在无需梯度计算的上下文中执行
        with override_quantized_engine("x86"), torch.no_grad():
            # 遍历所有测试用例
            for use_bias, post_op_algo in cases:
                # 创建测试模块，使用指定的偏置、后操作函数和后操作算法，并设置为评估模式
                m = TestHelperModules.LinearUnaryModule(
                    use_bias=use_bias, postop=postop, post_op_algo=post_op_algo
                ).eval()
                # 创建示例输入
                example_inputs = (torch.randn(2, 4),)
                # 创建 x86 诱导量化器，并设置全局量化配置
                quantizer = X86InductorQuantizer().set_global(
                    xiq.get_default_x86_inductor_quantization_config()
                )
                # 定义节点出现次数的期望
                node_occurrence = {
                    # 量化输入和卷积的权重，GELU 的输出
                    torch.ops.quantized_decomposed.quantize_per_tensor.default: 1,
                    torch.ops.quantized_decomposed.dequantize_per_tensor.default: 1,
                    # 权重的通道量化在常数传播中
                    torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
                    torch.ops.quantized_decomposed.dequantize_per_channel.default: 1,
                }
                # 定义节点列表，包括默认的量化和 GELU 操作
                node_list = [
                    torch.ops.quantized_decomposed.quantize_per_tensor.default,
                    torch.ops.quantized_decomposed.dequantize_per_tensor.default,
                    torch.ops.aten.linear.default,
                    torch.ops.aten.gelu.default,
                ]
                # 调用测试量化器方法，验证量化效果
                self._test_quantizer(
                    m,
                    example_inputs,
                    quantizer,
                    node_occurrence,
                    node_list,
                )

    def _check_annotation_stat(self, gm, expected_stat_dict):
        # 检查预期的注释统计数据，以确保注释正确

        def _check_annotation(node):
            # 检查节点的注释信息，判断是否已注释以及是否是量化模式的输出
            annot = node.meta.get(QUANT_ANNOTATION_KEY, None)
            if annot is None:
                return False, False
            return annot._annotated, annot._is_output_of_quantized_pattern

        # 遍历图中的每个节点
        for node in gm.graph.nodes:
            # 如果节点的目标在期望的统计字典中
            if node.target in expected_stat_dict.keys():
                # 检查节点的注释情况
                annotated, is_quant_out = _check_annotation(node)
                # 更新预期统计字典中的值
                expected_stat_dict[node.target]["annotated"] -= annotated
                expected_stat_dict[node.target]["is_quant_out"] -= is_quant_out
        # 断言所有操作的注释统计值都为零
        for op_stat in expected_stat_dict.values():
            assert all(v == 0 for v in op_stat.values())

    @skipIfNoX86
    @skipIfNoX86
    def test_linear_binary2(self):
        """
        Test Pattern:
            tmp = linear_1(x)
            tmp2 = linear_2(tmp)
            return tmp + tmp2
        Since linear_1 has 2 users, we should annotate linear_2 for binary fusion instead of linear_1
        """
        # 准备示例输入数据
        example_inputs = (torch.randn(2, 16),)
        # 创建一个 X86InductorQuantizer 实例，并设置全局配置
        quantizer = X86InductorQuantizer().set_global(
            xiq.get_default_x86_inductor_quantization_config()
        )
        
        # 进入量化引擎为 x86 模式，关闭梯度计算
        with override_quantized_engine("x86"), torch.no_grad():
            # 遍历 inplace_add_list，测试 inplace_add 为 False 的情况
            inplace_add_list = [False]
            for inplace_add in inplace_add_list:
                # 创建测试用的 LinearAddModule2 模型，并设为评估模式
                m = TestHelperModules.LinearAddModule2(inplace_add=inplace_add).eval()
                
                # 定义节点出现次数字典，描述量化和去量化操作的出现情况
                node_occurrence = {
                    torch.ops.quantized_decomposed.quantize_per_tensor.default: 2,
                    torch.ops.quantized_decomposed.dequantize_per_tensor.default: 2,
                    torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
                    torch.ops.quantized_decomposed.dequantize_per_channel.default: 2,
                }
                
                # 定义节点列表，包括 quantize、dequantize、linear 和 add 操作
                node_list = [
                    torch.ops.quantized_decomposed.quantize_per_tensor.default,
                    torch.ops.quantized_decomposed.dequantize_per_tensor.default,
                    torch.ops.aten.linear.default,
                    torch.ops.quantized_decomposed.quantize_per_tensor.default,
                    (
                        torch.ops.aten.add_.Tensor
                        if inplace_add
                        else torch.ops.aten.add.Tensor
                    ),
                ]
                
                # 运行量化器测试，获取最后一个量化后的模型
                fq_m = self._test_quantizer(
                    m,
                    example_inputs,
                    quantizer,
                    node_occurrence,
                    node_list,
                )[-1]
                
                # 预期的注释统计信息，描述线性和加法操作的量化情况
                aten = torch.ops. aten
                add_op = aten.add_.Tensor if inplace_add else aten.add.Tensor
                expected_annotation_stat = {
                    aten.linear.default: {
                        "annotated": 2,
                        "is_quant_out": 1,
                    },
                    add_op: {"annotated": 1, "is_quant_out": 1},
                }
                
                # 检查模型的注释统计是否符合预期
                self._check_annotation_stat(fq_m, expected_annotation_stat)

    @skipIfNoX86
    @skipIfNoX86
    def test_linear_binary_unary_serials(self):
        """
        Test pattern of 2 following up linear add relu with X86InductorQuantizer.
        """
        # 使用 "x86" 引擎覆盖默认的量化引擎，并禁用梯度计算
        with override_quantized_engine("x86"), torch.no_grad():
            # 创建并评估一个测试用的 SerialsLinearAddReLUModule 模型实例
            m = TestHelperModules.SerialsLinearAddReLUModule().eval()
            # 准备一个示例输入数据元组
            example_inputs = (torch.randn(2, 16),)
            # 创建 X86InductorQuantizer 实例，并设置全局量化配置
            quantizer = X86InductorQuantizer().set_global(
                xiq.get_default_x86_inductor_quantization_config()
            )
            # 定义节点出现次数的期望字典
            node_occurrence = {
                # 对 linear_1、linear_2/3（共享）、linear_4 分别进行量化
                torch.ops.quantized_decomposed.quantize_per_tensor.default: 3,
                # 每个 linear 节点均进行去量化
                torch.ops.quantized_decomposed.dequantize_per_tensor.default: 4,
                # 权重的量化以每通道方式进行，且已经常量传播
                torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
                torch.ops.quantized_decomposed.dequantize_per_channel.default: 4,
            }
            # 定义节点列表
            node_list = [
                torch.ops.quantized_decomposed.quantize_per_tensor.default,
                torch.ops.quantized_decomposed.dequantize_per_tensor.default,
                torch.ops.aten.linear.default,
                torch.ops.quantized_decomposed.quantize_per_tensor.default,
                torch.ops.quantized_decomposed.dequantize_per_tensor.default,
                torch.ops.aten.linear.default,
                torch.ops.aten.linear.default,
                torch.ops.aten.add.Tensor,
                torch.ops.aten.relu.default,
            ]
            # 调用 _test_quantizer 方法进行量化器测试，获取最后一个返回值
            fq_m = self._test_quantizer(
                m,
                example_inputs,
                quantizer,
                node_occurrence,
                node_list,
            )[-1]
            # 预期的注释统计结果字典
            expected_annotation_stat = {
                torch.ops.aten.linear.default: {
                    "annotated": 4,
                    "is_quant_out": 2,
                },
                torch.ops.aten.add.Tensor: {"annotated": 2, "is_quant_out": 0},
                torch.ops.aten.relu.default: {"annotated": 2, "is_quant_out": 2},
            }
            # 调用 _check_annotation_stat 方法检查注释统计结果
            self._check_annotation_stat(fq_m, expected_annotation_stat)

    @skipIfTorchDynamo("very slow")
    @skipIfNoX86
    # 定义一个测试方法，用于测试 QAT 模式下的 conv2d_bn 结构，使用 X86InductorQuantizer 进行量化
    def test_qat_conv2d(self):
        """
        Test QAT pattern of conv2d_bn with X86InductorQuantizer.
        """
        # 使用 "x86" 引擎覆盖量化引擎上下文
        with override_quantized_engine("x86"):
            # 创建一个带有批归一化的单层卷积模块
            m = TestHelperModules.SingleConv2dModule(with_bn=True)
            # 准备一个示例输入
            example_inputs = (torch.randn(2, 3, 16, 16),)
            # 创建 X86InductorQuantizer 实例，并设置全局量化配置为 QAT 模式的默认配置
            quantizer = X86InductorQuantizer().set_global(
                xiq.get_default_x86_inductor_quantization_config(is_qat=True)
            )
            # 定义预期的节点发生次数字典
            node_occurrence = {
                # 卷积的输入和权重各一个节点，卷积的输出一个节点
                torch.ops.quantized_decomposed.quantize_per_tensor.default: 2,
                torch.ops.quantized_decomposed.dequantize_per_tensor.default: 2,
                # 注意: 权重的量化操作被常量传播了
                torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
                torch.ops.quantized_decomposed.dequantize_per_channel.default: 1,
                # 批归一化应当被折叠到卷积中
                torch.ops.aten._native_batch_norm_legit.default: 0,
            }
            # 定义节点顺序列表
            node_list = [
                torch.ops.quantized_decomposed.quantize_per_tensor.default,
                torch.ops.quantized_decomposed.dequantize_per_tensor.default,
                torch.ops.aten.conv2d.default,
                torch.ops.quantized_decomposed.quantize_per_tensor.default,
                torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            ]
            # 调用测试方法，验证量化器行为
            self._test_quantizer(
                m,
                example_inputs,
                quantizer,
                node_occurrence,
                node_list,
                is_qat=True,
            )

    # 标记为跳过测试，在 Torch Dynamo 环境下非常缓慢时跳过
    @skipIfTorchDynamo("very slow")
    # 标记为跳过测试，如果不是 x86 平台则跳过
    @skipIfNoX86
    # 标记为跳过测试，在 Torch Dynamo 环境下非常缓慢时跳过
    @skipIfTorchDynamo("very slow")
    # 标记为跳过测试，如果不是 x86 平台则跳过
    @skipIfNoX86
    # 定义测试函数 test_qat_conv2d_binary，用于测试使用二进制后操作（如加法）的 qat 模式 conv2d_bn，使用 X86InductorQuantizer。
    """
    Test qat pattern of conv2d_bn with binary post ops (such as add) with X86InductorQuantizer.
    Currently, only add as binary post op is supported.
    """
    # 准备示例输入数据
    example_inputs = (torch.randn(2, 3, 6, 6),)
    
    # 创建 X86InductorQuantizer 实例，并设置全局量化配置
    quantizer = X86InductorQuantizer().set_global(
        xiq.get_default_x86_inductor_quantization_config(is_qat=True)
    )
    
    # 使用 "x86" 引擎覆盖量化引擎
    with override_quantized_engine("x86"):
        # 循环测试 inplace_add 参数为 True 和 False 两种情况
        for inplace_add in [True, False]:
            # 创建 Conv2dAddModule 模块实例，传入 inplace_add 和 with_bn=True 参数
            m = TestHelperModules.Conv2dAddModule(
                inplace_add=inplace_add, with_bn=True
            )
            
            # 定义期望的节点出现次数字典
            node_occurrence = {
                # Conv2d 操作的输入和权重
                torch.ops.quantized_decomposed.quantize_per_tensor.default: 3,
                torch.ops.quantized_decomposed.dequantize_per_tensor.default: 3,
                # Add 操作的输出
                torch.ops.quantized_decomposed.add: 1 if inplace_add else 1,
                torch.ops.quantized_decomposed.dequantize_per_tensor.default: 1,
                # 权重的量化
                torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
                torch.ops.quantized_decomposed.dequantize_per_channel.default: 1,
                # BN 应该被折叠到 Conv 中
                torch.ops.aten._native_batch_norm_legit.default: 0,
            }
            
            # 定义节点列表
            node_list = [
                torch.ops.quantized_decomposed.quantize_per_tensor.default,
                torch.ops.quantized_decomposed.dequantize_per_tensor.default,
                torch.ops.aten.conv2d.default,
                torch.ops.quantized_decomposed.add if inplace_add else torch.ops.quantized_decomposed.add.Tensor,
                torch.ops.quantized_decomposed.dequantize_per_tensor.default,
                torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            ]
            
            # 调用 _test_quantizer 方法进行量化器测试
            self._test_quantizer(
                m,
                example_inputs,
                quantizer,
                node_occurrence,
                node_list,
                is_qat=True,
            )

@skipIfTorchDynamo("very slow")
@skipIfNoX86
    # 定义名为 test_qat_conv2d_binary2 的测试方法，用于测试量化感知训练（QAT）模式下的卷积操作
    def test_qat_conv2d_binary2(self):
        """
        Test qat Pattern:
            tmp = bn1(conv2d_1(x))
            tmp2 = bn2(conv2d_2(tmp))
            return tmp + tmp2
        Since conv2d_1 has 2 users, we should annotate conv2d_2 for binary fusion instead of conv2d_1
        """
        # 创建一个示例输入，是一个包含随机数据的元组
        example_inputs = (torch.randn(2, 3, 6, 6),)
        # 创建一个 X86InductorQuantizer 实例，并设置为 QAT 模式的默认配置
        quantizer = X86InductorQuantizer().set_global(
            xiq.get_default_x86_inductor_quantization_config(is_qat=True)
        )
        # inplace_add_list 包含两个布尔值，用于测试是否使用原地加法
        inplace_add_list = [True, False]
        # 进入上下文管理器，覆盖量化引擎为 "x86"，并关闭 Torch 的梯度计算
        with override_quantized_engine("x86"), torch.no_grad():
            # 遍历 inplace_add_list 中的布尔值
            for inplace_add in inplace_add_list:
                # 创建一个 TestHelperModules.Conv2dAddModule2 的实例 m，设置 inplace_add 参数
                m = TestHelperModules.Conv2dAddModule2(inplace_add=inplace_add)
                # node_occurrence 是一个字典，指定了不同操作节点的出现次数
                node_occurrence = {
                    torch.ops.quantized_decomposed.quantize_per_tensor.default: 3,
                    torch.ops.quantized_decomposed.dequantize_per_tensor.default: 4,
                    # 权重的量化操作已被常量传播
                    torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
                    torch.ops.quantized_decomposed.dequantize_per_channel.default: 2,
                    # 批量归一化应该被折叠到卷积操作中
                    torch.ops.aten._native_batch_norm_legit.default: 0,
                }
                # node_list 是一个操作节点列表，包括了量化、反量化、卷积和加法操作
                node_list = [
                    torch.ops.quantized_decomposed.quantize_per_tensor.default,
                    torch.ops.quantized_decomposed.dequantize_per_tensor.default,
                    torch.ops.aten.conv2d.default,
                    torch.ops.quantized_decomposed.quantize_per_tensor.default,
                    (
                        torch.ops.aten.add_.Tensor
                        if inplace_add
                        else torch.ops.aten.add.Tensor
                    ),
                ]
                # 调用 _test_quantizer 方法来测试量化器的行为
                self._test_quantizer(
                    m,
                    example_inputs,
                    quantizer,
                    node_occurrence,
                    node_list,
                    is_qat=True,
                )

    # 使用装饰器 @skipIfTorchDynamo("very slow")，跳过 Torch Dynamo 引擎非常慢的测试
    @skipIfTorchDynamo("very slow")
    # 使用装饰器 @skipIfNoX86，如果不支持 X86 架构，则跳过测试
    @skipIfNoX86
    # 定义一个测试函数，用于测试 QAT 模式下的 conv2d_bn 模式，其中包含二元和一元后操作（如 add + relu），使用 X86InductorQuantizer 进行量化。
    # 当前仅支持 add 作为二元后操作和 relu 作为一元后操作。
    def test_qat_conv2d_binary_unary(self):
        """
        Test QAT pattern of conv2d_bn with binary + unary post ops (such as add + relu) with X86InductorQuantizer.
        Currently, only add as binary post op and relu as unary post op are supported.
        """
        # 准备测试用例的输入数据
        example_inputs = (torch.randn(2, 3, 6, 6),)
        
        # 创建 X86InductorQuantizer 实例，并设置全局量化配置为 QAT 模式下的默认配置
        quantizer = X86InductorQuantizer().set_global(
            xiq.get_default_x86_inductor_quantization_config(is_qat=True)
        )
        
        # 使用 x86 引擎进行量化引擎的覆盖
        with override_quantized_engine("x86"):
            # 创建一个带有 BatchNorm 的 Conv2dAddReLUModule 实例
            m = TestHelperModules.Conv2dAddReLUModule(with_bn=True)
            
            # 定义期望出现的节点及其出现次数的字典
            node_occurrence = {
                # conv 输入的量化节点
                torch.ops.quantized_decomposed.quantize_per_tensor.default: 3,
                # relu 输出的量化节点
                torch.ops.quantized_decomposed.dequantize_per_tensor.default: 3,
                # add 操作的额外输入节点
                torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
                # add 输出的量化节点
                torch.ops.quantized_decomposed.dequantize_per_channel.default: 1,
                # BatchNorm 在 Conv 中被折叠
                torch.ops.aten._native_batch_norm_legit.default: 0,
            }
            
            # 定义节点列表，按照执行顺序包含相应的量化和非量化操作节点
            node_list = [
                torch.ops.quantized_decomposed.quantize_per_tensor.default,
                torch.ops.quantized_decomposed.dequantize_per_tensor.default,
                torch.ops.aten.conv2d.default,
                torch.ops.aten.add.Tensor,
                torch.ops.quantized_decomposed.quantize_per_tensor.default,
                torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            ]
            
            # 调用测试函数 _test_quantizer，测试量化器的功能
            self._test_quantizer(
                m,
                example_inputs,
                quantizer,
                node_occurrence,
                node_list,
                is_qat=True,
            )

    # 如果没有 x86 环境，则跳过测试
    @skipIfNoX86
    # 定义一个测试方法，用于测试使用 X86InductorQuantizer 进行动态量化线性模块的模式
    def test_dynamic_quant_linear(self):
        """
        Test pattern of dynamic quantization of linear with X86InductorQuantizer.
        """
        # 使用 "x86" 引擎覆盖当前的量化引擎设置，并且在进入上下文时禁用梯度计算
        with override_quantized_engine("x86"), torch.no_grad():
            # 创建一个测试用的 SelfAttnLikeModule 模块，输入维度为 64，并设置为评估模式
            m = TestHelperModules.SelfAttnLikeModule(input_dim=64).eval()
            # 创建一个示例输入，形状为 (1, 4, 64)
            example_inputs = (torch.randn(1, 4, 64),)
            # 创建一个 X86InductorQuantizer 实例，并设置全局量化配置为动态量化模式
            quantizer = X86InductorQuantizer().set_global(
                xiq.get_default_x86_inductor_quantization_config(is_dynamic=True)
            )
            # 定义节点出现次数的字典，包括 quantized_decomposed 库中的操作
            node_occurrence = {
                torch.ops.quantized_decomposed.choose_qparams.tensor: 1,
                torch.ops.quantized_decomposed.quantize_per_tensor.tensor: 1,
                torch.ops.quantized_decomposed.dequantize_per_tensor.tensor: 1,
                # 权重的 quantize_per_channel 操作被常数传播
                torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
                torch.ops.quantized_decomposed.dequantize_per_channel.default: 3,
            }
            # 定义节点列表，包括 quantized_decomposed 库中的操作和标准的线性操作
            node_list = [
                torch.ops.quantized_decomposed.choose_qparams.tensor,
                torch.ops.quantized_decomposed.quantize_per_tensor.tensor,
                torch.ops.quantized_decomposed.dequantize_per_tensor.tensor,
                torch.ops.aten.linear.default,
            ]
            # 调用内部方法 _test_quantizer，测试量化器的功能
            self._test_quantizer(
                m,
                example_inputs,
                quantizer,
                node_occurrence,
                node_list,
            )

    # 如果没有 X86 平台支持，则跳过该测试
    @skipIfNoX86
    # 定义一个测试方法，用于测试 QAT 动态量化线性模块，使用 X86InductorQuantizer。
    def test_qat_dynamic_quant_linear(self):
        """
        Test pattern of qat dynamic quantization of linear with X86InductorQuantizer.
        """
        # 用 "x86" 引擎覆盖当前的量化引擎设置，并进入无梯度计算环境
        with override_quantized_engine("x86"), torch.no_grad():
            # 创建一个 SelfAttnLikeModule 模块实例，输入维度为 64，并设置为评估模式
            m = TestHelperModules.SelfAttnLikeModule(input_dim=64).eval()
            # 创建一个示例输入，是一个形状为 (1, 4, 64) 的随机张量
            example_inputs = (torch.randn(1, 4, 64),)
            # 创建一个 X86InductorQuantizer 实例，并设置全局量化配置
            quantizer = X86InductorQuantizer().set_global(
                xiq.get_default_x86_inductor_quantization_config(
                    is_qat=True, is_dynamic=True
                )
            )
            # 定义节点出现次数的字典，指定了不同量化操作的预期出现次数
            node_occurrence = {
                torch.ops.quantized_decomposed.choose_qparams.tensor: 1,
                torch.ops.quantized_decomposed.quantize_per_tensor.tensor: 1,
                torch.ops.quantized_decomposed.dequantize_per_tensor.tensor: 1,
                # 权重的通道量化操作被常数传播
                torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
                torch.ops.quantized_decomposed.dequantize_per_channel.default: 3,
            }
            # 定义节点列表，包含了在量化流程中使用的关键节点
            node_list = [
                torch.ops.quantized_decomposed.choose_qparams.tensor,
                torch.ops.quantized_decomposed.quantize_per_tensor.tensor,
                torch.ops.quantized_decomposed.dequantize_per_tensor.tensor,
                torch.ops.aten.linear.default,
            ]
            # 调用内部方法 _test_quantizer，用于测试量化器的功能
            self._test_quantizer(
                m,
                example_inputs,
                quantizer,
                node_occurrence,
                node_list,
                is_qat=True,
            )

    # 如果当前环境不支持 x86 架构，跳过该测试
    @skipIfNoX86
    def test_set_module_name_qconfig(self):
        """Test case for quantizing a specific submodule by configuring `set_module_name_qconfig`.

        Expect that all linear layers within the submodule `sub` are quantized.
        """

        class Sub(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(5, 10)  # 创建一个线性层，输入维度为5，输出维度为10
                self.relu1 = torch.nn.ReLU(inplace=False)  # 创建一个ReLU激活函数，不覆盖原始数据
                self.linear2 = torch.nn.Linear(10, 5)  # 创建另一个线性层，输入维度为10，输出维度为5

            def forward(self, x):
                x = self.linear1(x)  # 对输入应用第一个线性层
                x = self.relu1(x)  # 对结果应用ReLU激活函数
                x = self.linear2(x)  # 对结果应用第二个线性层
                return x

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = torch.nn.Linear(5, 5)  # 创建一个线性层，输入维度和输出维度均为5
                self.sub = Sub()  # 创建一个子模块Sub的实例

            def forward(self, x):
                x = self.linear(x)  # 对输入应用线性层
                x = self.sub(x)  # 对输入应用子模块Sub
                return x

        m = M().eval()  # 创建M的实例并将其设置为评估模式
        example_inputs = (torch.randn(3, 5),)  # 创建一个示例输入，大小为(3, 5)
        # 创建一个X86InductorQuantizer的实例
        quantizer = X86InductorQuantizer()
        # 配置名为"sub"的子模块的默认量化配置
        quantizer.set_module_name_qconfig(
            "sub", xiq.get_default_x86_inductor_quantization_config()
        )
        # 定义节点出现次数的字典
        node_occurrence = {
            torch.ops.aten.linear.default: 3,  # 线性操作的默认节点出现3次
            # 对两个来自`sub`的线性层的输入进行量化和反量化
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 2,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 2,
            # 对两个来自`sub`的线性层的权重进行反量化
            torch.ops.quantized_decomposed.dequantize_per_channel.default: 2,
        }
        node_list = [
            # 第一个线性层没有量化
            torch.ops.aten.linear.default,
            # 两个`sub`中的线性层的量化/反量化对
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.linear.default,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.linear.default,
        ]
        # 调用测试量化器的测试方法
        self._test_quantizer(
            m,
            example_inputs,
            quantizer,
            node_occurrence,
            node_list,
        )

    @skipIfNoX86  # 如果没有X86支持，则跳过测试
    def test_set_module_name_qconfig_with_underscores(self) -> None:
        """Test that if a module name has an underscore, we can still quantize it."""

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # This module name has underscores, which can be part of a mangled name.
                self.foo_bar = torch.nn.Linear(2, 2)  # 创建一个名为 foo_bar 的线性层
                self.baz = torch.nn.Linear(2, 2)  # 创建一个名为 baz 的线性层

            def forward(self, x):
                return self.baz(self.foo_bar(x))  # 在 forward 方法中使用创建的线性层进行前向传播计算

        # Set global to no quantization and then default config for a specific submodule whose name includes an underscore.
        quantizer = X86InductorQuantizer()  # 创建 X86InductorQuantizer 的实例，用于量化
        quantizer.set_module_name_qconfig(
            "foo_bar", xiq.get_default_x86_inductor_quantization_config()
        )  # 设置名为 "foo_bar" 的模块使用默认的 X86InductorQuantizationConfig 进行量化配置
        example_inputs = (torch.randn(2, 2),)
        m = M().eval()  # 创建 M 类的实例，并将其设置为评估模式
        m = capture_pre_autograd_graph(m, example_inputs)  # 捕获模型在输入示例上的前向传播计算图
        m = prepare_pt2e(m, quantizer)  # 准备模型以便在量化环境中执行
        # Use a linear count instead of names because the names might change, but
        # the order should be the same.
        count = 0
        for n in m.graph.nodes:  # 遍历模型的计算图节点
            if n.op == "call_function" and n.target == torch.ops.aten.linear.default:
                # Get the weight observer to see the per-channel vs per-tensor.
                weight_observer_node = n.args[1]  # 获取与线性操作关联的第二个参数，即权重观察器节点
                if count == 0:
                    # for foo_bar.
                    self.assertEqual(
                        weight_observer_node.op,
                        "call_module",
                        f"The op of linear({count})'s weight_observer_node is {weight_observer_node.op} instead call_module",
                    )  # 断言第一个线性层的权重观察器节点的操作是 "call_module"
                    observer_instance = getattr(m, weight_observer_node.target)  # 获取权重观察器实例
                    self.assertEqual(
                        observer_instance.qscheme, torch.per_channel_symmetric
                    )  # 断言权重观察器实例的量化方案是 torch.per_channel_symmetric
                else:
                    # For baz it should have no observer at all.
                    self.assertNotEqual(
                        weight_observer_node.op,
                        "call_module",
                        f"The op of linear({count})'s weight_observer_node is {weight_observer_node.op} instead call_module",
                    )  # 断言第二个线性层的权重观察器节点的操作不是 "call_module"
                count += 1

    @skipIfNoX86
    # 定义一个测试方法，用于测试同时设置 `module_name_qconfig` 和 `module_type_qconfig` 的情况。
    # 预期结果是除了最后一个线性层外，所有线性层都不被量化。
    def test_set_module_name_and_module_type_case1(self):

        # 定义一个简单的神经网络模型类 M
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(5, 10)  # 第一个线性层，输入维度为5，输出维度为10
                self.linear2 = torch.nn.Linear(10, 5)  # 第二个线性层，输入维度为10，输出维度为5
                self.sub = torch.nn.Linear(5, 5)       # 第三个线性层，输入输出维度都为5

            def forward(self, x):
                x = self.linear1(x)
                x = self.linear2(x)
                x = self.sub(x)
                return x

        # 创建一个模型实例，并设置为评估模式
        m = M().eval()
        example_inputs = (torch.randn(3, 5),)

        # 创建一个 X86InductorQuantizer 实例
        quantizer = X86InductorQuantizer()

        # 设置模块 "sub" 使用默认配置，同时将所有类型为 `torch.nn.Linear` 的模块设置为 `None`
        quantizer.set_module_name_qconfig(
            "sub", xiq.get_default_x86_inductor_quantization_config()
        ).set_module_type_qconfig(torch.nn.Linear, None)

        # 定义期望出现的节点及其出现次数
        node_occurrence = {
            torch.ops.aten.linear.default: 3,  # 默认的线性层节点出现3次
            # 对最后一个线性层的输入进行量化和反量化
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 1,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 1,
            # 对最后一个线性层的权重进行反量化
            torch.ops.quantized_decomposed.dequantize_per_channel.default: 1,
        }

        # 定义节点列表
        node_list = [
            # 第一个和第二个线性层未被量化
            torch.ops.aten.linear.default,
            torch.ops.aten.linear.default,
            # 最后一个线性层被量化
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.linear.default,
        ]

        # 调用测试方法 _test_quantizer 进行量化器测试
        self._test_quantizer(
            m,
            example_inputs,
            quantizer,
            node_occurrence,
            node_list,
        )

    # 如果不支持X86架构，则跳过该测试
    @skipIfNoX86
    def test_set_module_name_and_module_type_case2(self):
        """Test that set `module_name_qconfig` and `module_type_qconfig` at the same time.

        Expect that all linear layers are quantized except the last one.
        """

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = torch.nn.Linear(5, 10)  # 创建一个输入维度为5，输出维度为10的线性层
                self.linear2 = torch.nn.Linear(10, 5)  # 创建一个输入维度为10，输出维度为5的线性层
                self.sub = torch.nn.Linear(5, 5)        # 创建一个输入输出维度都为5的线性层

            def forward(self, x):
                x = self.linear1(x)  # 对输入进行第一个线性层的前向传播
                x = self.linear2(x)  # 对输入进行第二个线性层的前向传播
                x = self.sub(x)      # 对输入进行子模块线性层的前向传播
                return x

        m = M().eval()  # 创建一个测试用的模型实例并设置为评估模式
        example_inputs = (torch.randn(3, 5),)  # 创建一个形状为(3, 5)的示例输入张量

        # 创建一个 x86 Inductor 类的量化器实例
        quantizer = X86InductorQuantizer()
        
        # 设置模块名为 "sub" 的量化配置为 None，并为所有类型为 torch.nn.Linear 的模块设置默认的 x86 Inductor 量化配置
        quantizer.set_module_name_qconfig("sub", None).set_module_type_qconfig(
            torch.nn.Linear, xiq.get_default_x86_inductor_quantization_config()
        )

        # 定义预期节点出现次数的字典
        node_occurrence = {
            torch.ops.aten.linear.default: 3,
            # 对第一个和第二个线性层的输入和输出进行量化和反量化
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 2,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 2,
            # 对第一个和第二个线性层的权重进行反量化
            torch.ops.quantized_decomposed.dequantize_per_channel.default: 2,
        }

        # 定义节点列表，描述量化和反量化的顺序
        node_list = [
            # 第一个线性层的量化和反量化
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.linear.default,
            # 第二个线性层的量化和反量化
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.linear.default,
            # 最后一个线性层没有被量化
            torch.ops.aten.linear.default,
        ]

        # 执行量化器的测试方法
        self._test_quantizer(
            m,
            example_inputs,
            quantizer,
            node_occurrence,
            node_list,
        )

    @skipIfNoX86
    # 定义一个测试方法，用于测试动态量化时设置特定子模块的量化配置
    def test_set_module_name_qconfig_for_dynamic_quant(self):
        """Test that quantize a specific submodule for dynamic quantization."""

        # 使用 "x86" 引擎覆盖默认量化引擎，并关闭 Torch 梯度计算上下文
        with override_quantized_engine("x86"), torch.no_grad():
            # 针对每种量化训练策略（是否使用量化感知训练）
            for is_qat in [False, True]:
                # 创建一个评估模式下的 SelfAttnLikeModule 模块实例
                m = TestHelperModules.SelfAttnLikeModule(input_dim=64).eval()
                # 创建一个示例输入
                example_inputs = (torch.randn(1, 4, 64),)
                # 设置动态量化配置，针对 x86 诱导器的默认量化配置
                dynamic_config = xiq.get_default_x86_inductor_quantization_config(
                    is_dynamic=True, is_qat=is_qat
                )
                # 创建 X86InductorQuantizer 实例，并为 q_proj 和 v_proj 设置量化配置
                quantizer = (
                    X86InductorQuantizer()
                    .set_module_name_qconfig("q_proj", dynamic_config)
                    .set_module_name_qconfig("v_proj", dynamic_config)
                )
                # 定义节点出现次数的字典，用于量化和反量化操作
                node_occurrence = {
                    # 量化和反量化输入张量
                    torch.ops.quantized_decomposed.choose_qparams.tensor: 1,
                    torch.ops.quantized_decomposed.quantize_per_tensor.tensor: 1,
                    torch.ops.quantized_decomposed.dequantize_per_tensor.tensor: 1,
                    # 反量化 q_proj 和 v_proj 的权重
                    torch.ops.quantized_decomposed.dequantize_per_channel.default: 2,
                }
                # 定义节点列表，包含量化和反量化操作，以及各子模块的线性运算
                node_list = [
                    # 量化和反量化输入张量
                    torch.ops.quantized_decomposed.choose_qparams.tensor,
                    torch.ops.quantized_decomposed.quantize_per_tensor.tensor,
                    torch.ops.quantized_decomposed.dequantize_per_tensor.tensor,
                    # q_proj
                    torch.ops.aten.linear.default,
                    # k_proj
                    torch.ops.aten.linear.default,
                    # v_proj
                    torch.ops.aten.linear.default,
                ]
                # 调用内部方法 _test_quantizer 进行量化器测试
                self._test_quantizer(
                    m,
                    example_inputs,
                    quantizer,
                    node_occurrence,
                    node_list,
                    is_qat=is_qat,
                )

    # 根据是否支持 x86 平台来跳过测试
    @skipIfNoX86
    @skipIfNoX86
    def test_set_module_name_and_module_type_with_mixed_configs(self):
        """Test that set `module_name_qconfig` and `module_type_qconfig` at the same time with mixed the configs.

        Expect that only the last linear(`sub`) is quantized using static quantization.
        """

        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                # 定义模型的三个线性层
                self.linear1 = torch.nn.Linear(5, 10)
                self.linear2 = torch.nn.Linear(10, 5)
                self.sub = torch.nn.Linear(5, 5)

            def forward(self, x):
                # 模型的前向传播
                x = self.linear1(x)
                x = self.linear2(x)
                x = self.sub(x)
                return x

        # 创建并评估模型
        m = M().eval()
        example_inputs = (torch.randn(3, 5),)

        # 设置量化器并配置模块名称的量化参数
        quantizer = X86InductorQuantizer()
        quantizer.set_module_name_qconfig(
            "sub", xiq.get_default_x86_inductor_quantization_config(is_dynamic=False)
        ).set_module_type_qconfig(
            torch.nn.Linear,
            xiq.get_default_x86_inductor_quantization_config(is_dynamic=True),
        )

        # 定义期望的节点出现次数
        node_occurrence = {
            torch.ops.aten.linear.default: 3,
            torch.ops.quantized_decomposed.quantize_per_tensor.default: 1,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default: 1,
            torch.ops.quantized_decomposed.dequantize_per_channel.default: 1,
        }

        # 定义节点列表
        node_list = [
            torch.ops.aten.linear.default,
            torch.ops.aten.linear.default,
            torch.ops.quantized_decomposed.quantize_per_tensor.default,
            torch.ops.quantized_decomposed.dequantize_per_tensor.default,
            torch.ops.aten.linear.default,
        ]

        # 调用测试量化器的私有方法
        self._test_quantizer(
            m,
            example_inputs,
            quantizer,
            node_occurrence,
            node_list,
        )

    @skipIfNoX86
    def test_filter_conv2d_recipe(self):
        """
        Test removing conv2d from default recipe of X86InductorQuantizer.
        """
        # 使用上下文管理器override_quantized_engine，设置引擎为"x86"，并关闭梯度计算
        with override_quantized_engine("x86"), torch.no_grad():
            # 创建一个Conv2dUnaryModule模型，并设置为评估模式
            m = TestHelperModules.Conv2dUnaryModule(torch.nn.ReLU(inplace=False)).eval()
            # 创建一个示例输入
            example_inputs = (torch.randn(2, 3, 16, 16),)
            # 创建X86InductorQuantizer对象，并设置全局量化配置
            quantizer = X86InductorQuantizer().set_global(
                xiq.get_default_x86_inductor_quantization_config()
            )
            # 设置torch.nn.Conv2d模块的量化配置为None
            quantizer.set_module_type_qconfig(torch.nn.Conv2d, None)
            # 定义节点出现次数的字典，这里指定了quantize和dequantize操作的默认节点和计数
            node_occurrence = {
                # 对于conv的输入和权重，一个节点
                torch.ops.quantized_decomposed.quantize_per_tensor.default: 0,
                torch.ops.quantized_decomposed.dequantize_per_tensor.default: 0,
                # 注意：权重的量化操作会被常量传播
                torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
                torch.ops.quantized_decomposed.dequantize_per_channel.default: 0,
            }
            # 定义节点列表，包含aten.conv2d和aten.relu
            node_list = [
                torch.ops.aten.conv2d.default,
                torch.ops.aten.relu.default,
            ]
            # 调用测试函数_test_quantizer，用于测试量化器
            self._test_quantizer(
                m,
                example_inputs,
                quantizer,
                node_occurrence,
                node_list,
            )

    @skipIfNoX86
    def test_filter_linear_recipe(self):
        """
        Test removing linear from default recipe of X86InductorQuantizer.
        """
        # 使用上下文管理器override_quantized_engine，设置引擎为"x86"，并关闭梯度计算
        with override_quantized_engine("x86"), torch.no_grad():
            # 创建一个LinearUnaryModule模型，并设置为评估模式，包括使用偏置和ReLU后操作
            m = TestHelperModules.LinearUnaryModule(
                use_bias=True,
                postop=nn.ReLU,
            ).eval()
            # 创建一个示例输入
            example_inputs = (torch.randn(2, 4),)
            # 创建X86InductorQuantizer对象，并设置全局量化配置
            quantizer = X86InductorQuantizer().set_global(
                xiq.get_default_x86_inductor_quantization_config()
            )
            # 设置torch.nn.functional.linear函数的量化配置为None
            quantizer.set_function_type_qconfig(torch.nn.functional.linear, None)
            # 定义节点出现次数的字典，这里指定了quantize和dequantize操作的默认节点和计数
            node_occurrence = {
                # 对于linear的输入和权重，一个节点
                torch.ops.quantized_decomposed.quantize_per_tensor.default: 0,
                torch.ops.quantized_decomposed.dequantize_per_tensor.default: 0,
                # 注意：权重的量化操作会被常量传播
                torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
                torch.ops.quantized_decomposed.dequantize_per_channel.default: 0,
            }
            # 定义节点列表，包含aten.linear和aten.relu
            node_list = [
                torch.ops.aten.linear.default,
                torch.ops.aten.relu.default,
            ]
            # 调用测试函数_test_quantizer，用于测试量化器
            self._test_quantizer(
                m,
                example_inputs,
                quantizer,
                node_occurrence,
                node_list,
            )
    # 定义一个测试函数，用于测试从 X86InductorQuantizer 的默认配方中移除 maxpool2d 操作
    def test_filter_maxpool2d_recipe(self):
        """
        Test removing maxpool2d from default recipe of X86InductorQuantizer.
        """
        # 使用 x86 引擎进行量化，并进入无梯度计算环境
        with override_quantized_engine("x86"), torch.no_grad():
            # 创建一个 Conv2dUnaryModule 实例，用于测试，设定为评估模式
            m = TestHelperModules.Conv2dUnaryModule(torch.nn.ReLU(inplace=False)).eval()
            # 创建一个示例输入
            example_inputs = (torch.randn(2, 3, 16, 16),)
            # 创建一个 X86InductorQuantizer 实例，并设置全局量化配置
            quantizer = X86InductorQuantizer().set_global(
                xiq.get_default_x86_inductor_quantization_config()
            )
            # 设置 torch.nn.functional.max_pool2d 函数的量化配置为 None
            quantizer.set_function_type_qconfig(torch.nn.functional.max_pool2d, None)
            # 定义节点发生次数的期望字典
            node_occurrence = {
                # 卷积层输入和权重的量化操作
                torch.ops.quantized_decomposed.quantize_per_tensor.default: 1,
                torch.ops.quantized_decomposed.dequantize_per_tensor.default: 1,
                # 注意：权重的量化操作在常量传播中
                torch.ops.quantized_decomposed.quantize_per_channel.default: 0,
                torch.ops.quantized_decomposed.dequantize_per_channel.default: 1,
            }
            # 定义节点列表，包含量化操作、反量化操作以及卷积和 relu 操作
            node_list = [
                torch.ops.quantized_decomposed.quantize_per_tensor.default,
                torch.ops.quantized_decomposed.dequantize_per_tensor.default,
                torch.ops.aten.conv2d.default,
                torch.ops.aten.relu.default,
                torch.ops.aten.max_pool2d.default,
            ]
            # 调用内部函数 _test_quantizer 进行量化器的测试
            self._test_quantizer(
                m,
                example_inputs,
                quantizer,
                node_occurrence,
                node_list,
            )

    # 如果没有 x86 平台支持，则跳过这个测试
    @skipIfNoX86
```