# `.\pytorch\benchmarks\tensorexpr\conv.py`

```py
from . import benchmark
从当前目录中导入 benchmark 模块

class ConvImplBench(benchmark.Benchmark):
    继承 benchmark.Benchmark 类，表示 ConvImplBench 是 Benchmark 的子类

    def __init__(self, case, mode, device, dtype, kernel_size, N, iC, H, W, oC):
        初始化方法，接受多个参数来配置基准测试
        super().__init__(mode, device, dtype)
        调用父类的初始化方法来设置模式、设备和数据类型
        self.case = case
        将参数 case 存储为对象属性
        self.kernel_size = kernel_size
        将参数 kernel_size 存储为对象属性
        self.N = N
        将参数 N 存储为对象属性
        self.iC = iC
        将参数 iC 存储为对象属性
        self.H = H
        将参数 H 存储为对象属性
        self.W = W
        将参数 W 存储为对象属性
        self.oC = oC
        将参数 oC 存储为对象属性
        self.data = self.rand(
            [N, iC, H, W], device=device, requires_grad=self.requires_grad
        )
        使用 rand 方法生成指定形状的随机数据张量，并存储为对象属性
        if case == "conv":
            如果 case 参数为 "conv"，设置 groups 属性为 1
        elif case == "depthwise_conv":
            如果 case 参数为 "depthwise_conv"，设置 groups 属性为 iC
        else:
            如果 case 参数既不是 "conv" 也不是 "depthwise_conv"，抛出值错误异常
        self.conv = self.conv2d_layer(iC, oC, kernel_size, groups=self.groups)
        使用 conv2d_layer 方法创建一个卷积层，并将其存储为对象属性
        if device != "cpu":
            如果设备不是 CPU，则将 conv 层移动到指定设备上
            self.to_device(self.conv, device)

    def forward(self):
        前向传播方法，计算并返回卷积层在输入数据上的输出
        y = self.conv(self.data)
        调用 conv 层对输入数据进行卷积计算，并将结果存储在 y 变量中
        return y

    def config(self):
        返回配置参数的列表，包括 kernel_size, N, iC, H, W, oC

    def memory_workload(self):
        内存负载计算方法，根据模式计算并返回内存使用量的字典
        if self.mode == "fwd":
            如果模式是前向传播，设置 sol_count 和 algorithmic_count 字典的值为 1
        else:
            否则，设置 sol_count 和 algorithmic_count 字典的值为 2
        buffer_size = {
            计算输入、输出和卷积核的缓冲区大小，并存储为字典
        }
        计算 sol_size 和 algorithmic_size，并返回它们的总和
        return {"sol": sol_size, "algorithmic": algorithmic_size}

    def compute_workload(self):
        计算工作负载的方法，根据模式和参数计算并返回操作数量
        if self.mode == "fwd":
            如果模式是前向传播，设置 count 为 1
        elif self.mode == "both":
            如果模式是前向和反向传播，设置 count 为 3
        else:
            否则，抛出值错误异常
        计算卷积操作的总数，并返回它
        return op_count * count

    @staticmethod
    def default_configs():
        返回一个包含默认配置的列表，其中包括 [3, 64, 32, 128, 128, 64]

class ConvBench(ConvImplBench):
    ConvImplBench 的子类，表示基础卷积测试基准类

    def __init__(self, *args):
        初始化方法，接受可变数量的位置参数，并调用父类的初始化方法来设置 case 为 "conv"

    @staticmethod
    def module():
        静态方法，返回字符串 "conv"

class DepthwiseConvBench(ConvImplBench):
    ConvImplBench 的子类，表示深度可分离卷积测试基准类

    def __init__(self, *args):
        初始化方法，接受可变数量的位置参数，并调用父类的初始化方法来设置 case 为 "depthwise_conv"

    @staticmethod
    def module():
        静态方法，返回字符串 "depthwise_conv"

注册 ConvBench 和 DepthwiseConvBench 类到 benchmark 模块中的基准测试注册表中
```