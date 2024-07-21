# `.\pytorch\torch\testing\_internal\common_pruning.py`

```py
# mypy: ignore-errors

# Owner(s): ["module: unknown"]

# 从torch.ao.pruning模块中导入BaseSparsifier基类
from torch.ao.pruning import BaseSparsifier
# 导入torch库
import torch
# 导入torch中的函数库nn.functional，并重命名为F
import torch.nn.functional as F
# 从torch中导入nn模块
from torch import nn

# 实现自定义稀疏化器类ImplementedSparsifier，继承自BaseSparsifier基类
class ImplementedSparsifier(BaseSparsifier):
    def __init__(self, **kwargs):
        # 调用父类BaseSparsifier的构造函数，传递kwargs参数
        super().__init__(defaults=kwargs)

    # 更新模块的掩码方法
    def update_mask(self, module, **kwargs):
        # 将module的第一个参数化权重的第一个掩码设置为0
        module.parametrizations.weight[0].mask[0] = 0
        # 获取并更新状态字典self.state中'linear1.weight'对应的值，增加步数计数
        linear_state = self.state['linear1.weight']
        linear_state['step_count'] = linear_state.get('step_count', 0) + 1


# 定义一个MockSparseLinear类，继承自nn.Linear类
class MockSparseLinear(nn.Linear):
    """
    This class is a MockSparseLinear class to check convert functionality.
    It is the same as a normal Linear layer, except with a different type, as
    well as an additional from_dense method.
    """
    @classmethod
    def from_dense(cls, mod):
        """
        构造方法，接受一个模型mod，返回一个新的MockSparseLinear对象
        """
        # 使用cls构造一个新的MockSparseLinear对象，继承自mod的输入和输出特征维度
        linear = cls(mod.in_features,
                     mod.out_features)
        return linear


# 定义一个函数rows_are_subset，检查subset_tensor中的所有行是否都存在于superset_tensor中
def rows_are_subset(subset_tensor, superset_tensor) -> bool:
    """
    Checks to see if all rows in subset tensor are present in the superset tensor
    """
    # 初始化索引i为0
    i = 0
    # 遍历subset_tensor中的每一行
    for row in subset_tensor:
        # 当i小于superset_tensor的长度时进行循环
        while i < len(superset_tensor):
            # 如果subset_tensor中的行与superset_tensor中的第i行不相等
            if not torch.equal(row, superset_tensor[i]):
                # 将索引i增加1
                i += 1
            else:
                # 如果相等则跳出循环
                break
        else:
            # 如果在superset_tensor中找不到相等的行，则返回False
            return False
    # 如果subset_tensor中的所有行都存在于superset_tensor中，则返回True
    return True


# 定义一个简单的神经网络模型SimpleLinear，只包含线性层，无偏置项
class SimpleLinear(nn.Module):
    r"""Model with only Linear layers without biases, some wrapped in a Sequential,
    some following the Sequential. Used to test basic pruned Linear-Linear fusion."""

    def __init__(self):
        # 调用父类nn.Module的构造函数
        super().__init__()
        # 使用nn.Sequential包装的线性层序列
        self.seq = nn.Sequential(
            nn.Linear(7, 5, bias=False),
            nn.Linear(5, 6, bias=False),
            nn.Linear(6, 4, bias=False),
        )
        # 单独的线性层self.linear1，输入和输出维度为4，无偏置项
        self.linear1 = nn.Linear(4, 4, bias=False)
        # 单独的线性层self.linear2，输入维度为4，输出维度为10，无偏置项
        self.linear2 = nn.Linear(4, 10, bias=False)

    # 前向传播方法，接受输入x，依次通过self.seq、self.linear1和self.linear2
    def forward(self, x):
        x = self.seq(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x


# 定义一个神经网络模型LinearBias，包含带偏置项的线性层和不带偏置项的线性层
class LinearBias(nn.Module):
    r"""Model with only Linear layers, alternating layers with biases,
    wrapped in a Sequential. Used to test pruned Linear-Bias-Linear fusion."""

    def __init__(self):
        # 调用父类nn.Module的构造函数
        super().__init__()
        # 使用nn.Sequential包装的线性层序列，包含带偏置项和不带偏置项的线性层交替排列
        self.seq = nn.Sequential(
            nn.Linear(7, 5, bias=True),
            nn.Linear(5, 6, bias=False),
            nn.Linear(6, 3, bias=True),
            nn.Linear(3, 3, bias=True),
            nn.Linear(3, 10, bias=False),
        )

    # 前向传播方法，接受输入x，依次通过self.seq
    def forward(self, x):
        x = self.seq(x)
        return x


# 定义一个神经网络模型LinearActivation，包含带偏置项和不带偏置项的线性层，以及激活函数
class LinearActivation(nn.Module):
    r"""Model with only Linear layers, some with bias, some in a Sequential and some following.
    Activation functions modules in between each Linear in the Sequential, and each outside layer.
    Used to test pruned Linear(Bias)-Activation-Linear fusion."""
    # 初始化函数，用于定义神经网络的结构和层次
    def __init__(self):
        # 调用父类的初始化方法
        super().__init__()
        # 定义神经网络的序列模块，依次包括线性层、激活函数ReLU、线性层、激活函数Tanh、线性层
        self.seq = nn.Sequential(
            nn.Linear(7, 5, bias=True),  # 输入维度7，输出维度5的线性层，包括偏置项
            nn.ReLU(),                   # ReLU激活函数
            nn.Linear(5, 6, bias=False), # 输入维度5，输出维度6的线性层，不包括偏置项
            nn.Tanh(),                   # Tanh激活函数
            nn.Linear(6, 4, bias=True),  # 输入维度6，输出维度4的线性层，包括偏置项
        )
        # 定义一个线性层，输入维度4，输出维度3，包括偏置项
        self.linear1 = nn.Linear(4, 3, bias=True)
        # 定义ReLU激活函数
        self.act1 = nn.ReLU()
        # 定义一个线性层，输入维度3，输出维度10，不包括偏置项
        self.linear2 = nn.Linear(3, 10, bias=False)
        # 定义Tanh激活函数
        self.act2 = nn.Tanh()

    # 前向传播函数，定义数据在神经网络中的流动过程
    def forward(self, x):
        x = self.seq(x)     # 将输入数据x经过序列模块进行处理
        x = self.linear1(x) # 经过线性层1处理
        x = self.act1(x)    # 经过ReLU激活函数处理
        x = self.linear2(x) # 经过线性层2处理
        x = self.act2(x)    # 经过Tanh激活函数处理
        return x            # 返回处理后的输出
class LinearActivationFunctional(nn.Module):
    r"""Model with only Linear layers, some with bias, some in a Sequential and some following.
    Activation functions modules in between each Linear in the Sequential, and functional
    activationals are called in between each outside layer.
    Used to test pruned Linear(Bias)-Activation-Linear fusion."""

    def __init__(self):
        super().__init__()
        # 定义包含多个线性层的序列模块
        self.seq = nn.Sequential(
            nn.Linear(7, 5, bias=True),  # 输入维度为7，输出维度为5的线性层，包含偏置
            nn.ReLU(),  # ReLU 激活函数
            nn.Linear(5, 6, bias=False),  # 输入维度为5，输出维度为6的线性层，无偏置
            nn.ReLU(),  # ReLU 激活函数
            nn.Linear(6, 4, bias=True),  # 输入维度为6，输出维度为4的线性层，包含偏置
        )
        self.linear1 = nn.Linear(4, 3, bias=True)  # 输入维度为4，输出维度为3的线性层，包含偏置
        self.linear2 = nn.Linear(3, 8, bias=False)  # 输入维度为3，输出维度为8的线性层，无偏置
        self.linear3 = nn.Linear(8, 10, bias=False)  # 输入维度为8，输出维度为10的线性层，无偏置
        self.act1 = nn.ReLU()  # ReLU 激活函数

    def forward(self, x):
        x = self.seq(x)  # 序列模块中的前向传播
        x = self.linear1(x)  # 单独的线性层1的前向传播
        x = F.relu(x)  # 使用ReLU激活函数
        x = self.linear2(x)  # 单独的线性层2的前向传播
        x = F.relu(x)  # 使用ReLU激活函数
        x = self.linear3(x)  # 单独的线性层3的前向传播
        x = F.relu(x)  # 使用ReLU激活函数
        return x


class SimpleConv2d(nn.Module):
    r"""Model with only Conv2d layers, all without bias, some in a Sequential and some following.
    Used to test pruned Conv2d-Conv2d fusion."""

    def __init__(self):
        super().__init__()
        # 定义包含多个卷积层的序列模块，全部无偏置
        self.seq = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, bias=False),  # 输入通道为1，输出通道为32的3x3卷积层，无偏置
            nn.Conv2d(32, 64, 3, 1, bias=False),  # 输入通道为32，输出通道为64的3x3卷积层，无偏置
        )
        self.conv2d1 = nn.Conv2d(64, 48, 3, 1, bias=False)  # 输入通道为64，输出通道为48的3x3卷积层，无偏置
        self.conv2d2 = nn.Conv2d(48, 52, 3, 1, bias=False)  # 输入通道为48，输出通道为52的3x3卷积层，无偏置

    def forward(self, x):
        x = self.seq(x)  # 序列模块中的前向传播
        x = self.conv2d1(x)  # 单独的卷积层1的前向传播
        x = self.conv2d2(x)  # 单独的卷积层2的前向传播
        return x


class Conv2dBias(nn.Module):
    r"""Model with only Conv2d layers, some with bias, some in a Sequential and some outside.
    Used to test pruned Conv2d-Bias-Conv2d fusion."""

    def __init__(self):
        super().__init__()
        # 定义包含多个卷积层的序列模块，部分卷积层包含偏置
        self.seq = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, bias=True),  # 输入通道为1，输出通道为32的3x3卷积层，包含偏置
            nn.Conv2d(32, 32, 3, 1, bias=True),  # 输入通道和输出通道均为32的3x3卷积层，包含偏置
            nn.Conv2d(32, 64, 3, 1, bias=False),  # 输入通道为32，输出通道为64的3x3卷积层，无偏置
        )
        self.conv2d1 = nn.Conv2d(64, 48, 3, 1, bias=True)  # 输入通道为64，输出通道为48的3x3卷积层，包含偏置
        self.conv2d2 = nn.Conv2d(48, 52, 3, 1, bias=False)  # 输入通道为48，输出通道为52的3x3卷积层，无偏置

    def forward(self, x):
        x = self.seq(x)  # 序列模块中的前向传播
        x = self.conv2d1(x)  # 单独的卷积层1的前向传播
        x = self.conv2d2(x)  # 单独的卷积层2的前向传播
        return x


class Conv2dActivation(nn.Module):
    r"""Model with only Conv2d layers, some with bias, some in a Sequential and some following.
    Activation function modules in between each Sequential layer, functional activations called
    in-between each outside layer.
    Used to test pruned Conv2d-Bias-Activation-Conv2d fusion."""
    # 初始化函数，继承父类的初始化方法
    def __init__(self):
        # 调用父类的初始化方法
        super().__init__()
        # 定义一个神经网络层序列
        self.seq = nn.Sequential(
            # 添加一个二维卷积层，输入通道数为1，输出通道数为32，卷积核大小为3，步长为1，带偏置项
            nn.Conv2d(1, 32, 3, 1, bias=True),
            # ReLU激活函数层
            nn.ReLU(),
            # 添加一个二维卷积层，输入通道数为32，输出通道数为64，卷积核大小为3，步长为1，带偏置项
            nn.Conv2d(32, 64, 3, 1, bias=True),
            # Tanh激活函数层
            nn.Tanh(),
            # 添加一个二维卷积层，输入通道数为64，输出通道数为64，卷积核大小为3，步长为1，不带偏置项
            nn.Conv2d(64, 64, 3, 1, bias=False),
            # ReLU激活函数层
            nn.ReLU(),
        )
        # 定义一个二维卷积层，输入通道数为64，输出通道数为48，卷积核大小为3，步长为1，不带偏置项
        self.conv2d1 = nn.Conv2d(64, 48, 3, 1, bias=False)
        # 定义一个二维卷积层，输入通道数为48，输出通道数为52，卷积核大小为3，步长为1，带偏置项
        self.conv2d2 = nn.Conv2d(48, 52, 3, 1, bias=True)

    # 前向传播函数
    def forward(self, x):
        # 将输入数据通过之前定义的神经网络层序列进行前向传播
        x = self.seq(x)
        # 将上一步的输出结果通过第一个二维卷积层进行前向传播
        x = self.conv2d1(x)
        # 使用ReLU激活函数处理第一个卷积层的输出
        x = F.relu(x)
        # 将上一步的输出结果通过第二个二维卷积层进行前向传播
        x = self.conv2d2(x)
        # 使用hardtanh激活函数处理第二个卷积层的输出
        x = F.hardtanh(x)
        # 返回处理后的结果
        return x
class Conv2dPadBias(nn.Module):
    r"""Model with only Conv2d layers, all with bias and some with padding > 0,
    some in a Sequential and some following. Activation function modules in between each layer.
    Used to test that bias is propagated correctly in the special case of
    pruned Conv2d-Bias-(Activation)Conv2d fusion, when the second Conv2d layer has padding > 0."""

    def __init__(self):
        super().__init__()
        # 定义包含多个 Conv2d 和激活函数的序列
        self.seq = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, padding=1, bias=True),  # 1st Conv2d layer: 1 input channel, 32 output channels, 3x3 kernel, 1 stride, 1 padding, with bias
            nn.ReLU(),  # ReLU activation
            nn.Conv2d(32, 32, 3, 1, bias=False),  # 2nd Conv2d layer: 32 input channels, 32 output channels, 3x3 kernel, 1 stride, no padding, no bias
            nn.ReLU(),  # ReLU activation
            nn.Conv2d(32, 32, 3, 1, padding=1, bias=True),  # 3rd Conv2d layer: 32 input channels, 32 output channels, 3x3 kernel, 1 stride, 1 padding, with bias
            nn.ReLU(),  # ReLU activation
            nn.Conv2d(32, 32, 3, 1, padding=1, bias=True),  # 4th Conv2d layer: 32 input channels, 32 output channels, 3x3 kernel, 1 stride, 1 padding, with bias
            nn.ReLU(),  # ReLU activation
            nn.Conv2d(32, 64, 3, 1, bias=True),  # 5th Conv2d layer: 32 input channels, 64 output channels, 3x3 kernel, 1 stride, no padding, with bias
            nn.Tanh(),  # Tanh activation
        )
        # 单独定义一个 Conv2d 层
        self.conv2d1 = nn.Conv2d(64, 48, 3, 1, padding=1, bias=True)  # 6th Conv2d layer: 64 input channels, 48 output channels, 3x3 kernel, 1 stride, 1 padding, with bias
        self.act1 = nn.ReLU()  # ReLU activation
        self.conv2d2 = nn.Conv2d(48, 52, 3, 1, padding=1, bias=True)  # 7th Conv2d layer: 48 input channels, 52 output channels, 3x3 kernel, 1 stride, 1 padding, with bias
        self.act2 = nn.Tanh()  # Tanh activation

    def forward(self, x):
        x = self.seq(x)  # 应用序列中定义的 Conv2d 层和激活函数
        x = self.conv2d1(x)  # 应用单独定义的 Conv2d1 层
        x = self.act1(x)  # 应用 ReLU 激活函数
        x = self.conv2d2(x)  # 应用单独定义的 Conv2d2 层
        x = self.act2(x)  # 应用 Tanh 激活函数
        return x


class Conv2dPool(nn.Module):
    r"""Model with only Conv2d layers, all with bias, some in a Sequential and some following.
    Activation function modules in between each layer, Pool2d modules in between each layer.
    Used to test pruned Conv2d-Pool2d-Conv2d fusion."""

    def __init__(self):
        super().__init__()
        # 定义包含多个 Conv2d 和池化层的序列
        self.seq = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=True),  # 1st Conv2d layer: 1 input channel, 32 output channels, 3x3 kernel, 1 stride, 1 padding, with bias
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),  # Max pooling with 2x2 kernel, stride 2, 1 padding
            nn.ReLU(),  # ReLU activation
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=True),  # 2nd Conv2d layer: 32 input channels, 64 output channels, 3x3 kernel, 1 stride, 1 padding, with bias
            nn.Tanh(),  # Tanh activation
            nn.AvgPool2d(kernel_size=2, stride=2, padding=1),  # Average pooling with 2x2 kernel, stride 2, 1 padding
        )
        # 单独定义一个 Conv2d 层
        self.conv2d1 = nn.Conv2d(64, 48, kernel_size=3, padding=1, bias=True)  # 3rd Conv2d layer: 64 input channels, 48 output channels, 3x3 kernel, 1 stride, 1 padding, with bias
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)  # Max pooling with 2x2 kernel, stride 2, 1 padding
        self.af1 = nn.ReLU()  # ReLU activation
        self.conv2d2 = nn.Conv2d(48, 52, kernel_size=3, padding=1, bias=True)  # 4th Conv2d layer: 48 input channels, 52 output channels, 3x3 kernel, 1 stride, 1 padding, with bias
        self.conv2d3 = nn.Conv2d(52, 52, kernel_size=3, padding=1, bias=True)  # 5th Conv2d layer: 52 input channels, 52 output channels, 3x3 kernel, 1 stride, 1 padding, with bias

    def forward(self, x):
        x = self.seq(x)  # 应用序列中定义的 Conv2d 层和池化层
        x = self.conv2d1(x)  # 应用单独定义的 Conv2d1 层
        x = self.maxpool(x)  # 应用 Max pooling 层
        x = self.af1(x)  # 应用 ReLU 激活函数
        x = self.conv2d2(x)  # 应用单独定义的 Conv2d2 层
        x = F.avg_pool2d(x, kernel_size=2, stride=2, padding=1)  # 应用平均池化
        x = F.relu(x)  # 应用 ReLU 激活函数
        x = self.conv2d3(x)  # 应用单独定义的 Conv2d3 层
        return x


class Conv2dPoolFlattenFunctional(nn.Module):
    r"""Model with Conv2d layers, all with bias, some in a Sequential and some following, and then a Pool2d
    and a functional Flatten followed by a Linear layer.
    Activation functions and Pool2ds in between each layer also.
    Used to test pruned Conv2d-Pool2d-Flatten-Linear fusion."""
    # 定义神经网络模型的初始化方法
    def __init__(self):
        super().__init__()  # 调用父类的初始化方法
        # 定义一个包含多个层的顺序容器
        self.seq = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=3, padding=1, bias=True),  # 第一个卷积层，输入通道1，输出通道3
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),      # 最大池化层，核大小为2，步幅为2
            nn.ReLU(),                                             # ReLU 激活函数
            nn.Conv2d(3, 5, kernel_size=3, padding=1, bias=True),  # 第二个卷积层，输入通道3，输出通道5
            nn.Tanh(),                                             # Tanh 激活函数
            nn.AvgPool2d(kernel_size=2, stride=2, padding=1),      # 平均池化层，核大小为2，步幅为2
        )
        self.conv2d1 = nn.Conv2d(5, 7, kernel_size=3, padding=1, bias=True)  # 单独定义的第三个卷积层，输入通道5，输出通道7
        self.af1 = nn.ReLU()  # 卷积层后的 ReLU 激活函数
        self.conv2d2 = nn.Conv2d(7, 11, kernel_size=3, padding=1, bias=True)  # 单独定义的第四个卷积层，输入通道7，输出通道11
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))  # 自适应平均池化层，输出特征图大小为 (1, 1)
        self.fc = nn.Linear(11, 13, bias=True)  # 全连接层，输入大小11，输出大小13，包含偏置项

    # 定义神经网络的前向传播方法
    def forward(self, x):
        x = self.seq(x)  # 应用序列容器中的各层到输入 x 上
        x = self.conv2d1(x)  # 应用单独定义的第三个卷积层到 x 上
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=1)  # 使用函数式接口实现最大池化
        x = self.af1(x)  # 应用卷积层后的 ReLU 激活函数到 x 上
        x = self.conv2d2(x)  # 应用单独定义的第四个卷积层到 x 上
        x = self.avg_pool(x)  # 应用自适应平均池化到 x 上，将特征图大小调整为 (1, 1)
        x = torch.flatten(x, 1)  # 使用 PyTorch 提供的 flatten 函数将 x 展平，保留批次维度
        x = self.fc(x)  # 应用全连接层到 x 上，得到最终的输出
        return x
class Conv2dPoolFlatten(nn.Module):
    r"""Model with Conv2d layers, all with bias, some in a Sequential and some following, and then a Pool2d
    and a Flatten module followed by a Linear layer.
    Activation functions and Pool2ds in between each layer also.
    Used to test pruned Conv2d-Pool2d-Flatten-Linear fusion."""

    def __init__(self):
        super().__init__()
        # 定义一个序列模块，包含多个层的连续操作
        self.seq = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=3, padding=1, bias=True),  # 一个Conv2d层，输入通道1，输出通道3
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),     # 一个MaxPool2d层，池化核大小为2x2
            nn.ReLU(),                                            # ReLU激活函数
            nn.Conv2d(3, 5, kernel_size=3, padding=1, bias=True),  # 一个Conv2d层，输入通道3，输出通道5
            nn.Tanh(),                                            # Tanh激活函数
            nn.AvgPool2d(kernel_size=2, stride=2, padding=1),     # 一个AvgPool2d层，池化核大小为2x2
        )
        self.conv2d1 = nn.Conv2d(5, 7, kernel_size=3, padding=1, bias=True)  # 单独定义一个Conv2d层，输入通道5，输出通道7
        self.af1 = nn.ReLU()                                                # 单独定义ReLU激活函数
        self.conv2d2 = nn.Conv2d(7, 11, kernel_size=3, padding=1, bias=True)  # 单独定义一个Conv2d层，输入通道7，输出通道11
        self.avg_pool = nn.AdaptiveAvgPool2d((2, 2))                         # 自适应平均池化层，输出大小为2x2
        self.flatten = nn.Flatten()                                          # Flatten层，用于将输入展平为一维向量
        self.fc = nn.Linear(44, 13, bias=True)                               # 线性层，输入大小44，输出大小13

    def forward(self, x):
        x = self.seq(x)                        # 序列模块的前向传播
        x = self.conv2d1(x)                    # 单独Conv2d层的前向传播
        x = F.max_pool2d(x, kernel_size=2, stride=2, padding=1)  # 使用函数形式的最大池化
        x = self.af1(x)                        # 单独ReLU激活函数的前向传播
        x = self.conv2d2(x)                    # 单独Conv2d层的前向传播
        x = self.avg_pool(x)                   # 自适应平均池化层的前向传播
        x = self.flatten(x)                    # Flatten层的前向传播
        x = self.fc(x)                         # 线性层的前向传播
        return x


class LSTMLinearModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a linear."""

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int
    ):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers)  # LSTM模块，输入维度input_dim，隐藏层维度hidden_dim，层数num_layers
        self.linear = nn.Linear(hidden_dim, output_dim)         # 线性层，输入大小为隐藏层维度hidden_dim，输出大小为output_dim

    def forward(self, input):
        output, hidden = self.lstm(input)   # LSTM模块的前向传播，返回输出和隐藏状态
        decoded = self.linear(output)       # 线性层的前向传播，对LSTM输出进行线性变换
        return decoded, output


class LSTMLayerNormLinearModel(nn.Module):
    """Container module with an LSTM, a LayerNorm, and a linear."""

    def __init__(
        self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int
    ):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers)  # LSTM模块，输入维度input_dim，隐藏层维度hidden_dim，层数num_layers
        self.norm = nn.LayerNorm(hidden_dim)                   # LayerNorm层，对隐藏层进行归一化
        self.linear = nn.Linear(hidden_dim, output_dim)         # 线性层，输入大小为隐藏层维度hidden_dim，输出大小为output_dim

    def forward(self, x):
        x, state = self.lstm(x)   # LSTM模块的前向传播，返回输出和状态
        x = self.norm(x)          # LayerNorm层的前向传播
        x = self.linear(x)        # 线性层的前向传播
        return x, state
```