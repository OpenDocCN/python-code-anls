# `.\pytorch\test\inductor\test_efficient_conv_bn_eval.py`

```
# Owner(s): ["module: inductor"]
# 导入必要的模块和库
import copy  # 导入 copy 模块，用于复制对象
import importlib  # 导入 importlib 模块，用于动态导入模块
import itertools  # 导入 itertools 模块，用于创建迭代器
import os  # 导入 os 模块，用于与操作系统交互
import sys  # 导入 sys 模块，用于访问系统特定的参数和功能
import unittest  # 导入 unittest 模块，用于编写和运行测试

import torch  # 导入 PyTorch 库
from torch import nn  # 从 torch 模块中导入 nn 模块

# Make the helper files in test/ importable
# 将测试目录中的辅助文件添加到导入路径中
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

from torch._dynamo.utils import counters  # 从 torch._dynamo.utils 中导入 counters 模块
from torch._inductor import config as inductor_config  # 导入 inductor_config 模块
from torch._inductor.test_case import TestCase  # 从 torch._inductor.test_case 中导入 TestCase 类

from torch.testing._internal.common_utils import IS_CI, IS_WINDOWS, TEST_WITH_ASAN  # 从 torch.testing._internal.common_utils 中导入常用工具函数

from torch.testing._internal.inductor_utils import HAS_CPU, HAS_CUDA  # 从 torch.testing._internal.inductor_utils 中导入 HAS_CPU 和 HAS_CUDA

# 如果运行环境是 Windows 且是在持续集成（CI）环境下
if IS_WINDOWS and IS_CI:
    # 输出错误消息说明 Windows CI 环境缺少 test_torchinductor 所需的依赖
    sys.stderr.write(
        "Windows CI does not have necessary dependencies for test_torchinductor yet\n"
    )
    # 如果当前脚本是主程序
    if __name__ == "__main__":
        sys.exit(0)  # 退出程序，返回状态码 0
    raise unittest.SkipTest("requires sympy/functorch/filelock")  # 抛出 SkipTest 异常，跳过当前单元测试

# 动态导入 functorch 模块
importlib.import_module("functorch")
# 动态导入 filelock 模块
importlib.import_module("filelock")

# 从 inductor.test_torchinductor 模块中导入 copy_tests 函数
from inductor.test_torchinductor import copy_tests


# 定义一个卷积操作类 ConvOp，继承自 nn.Module
class ConvOp(nn.Module):
    expected_optimization_count = 1  # 期望的优化次数为 1

    def __init__(
        self,
        conv_class,
        bn_class,
        use_bias,
        in_channels,
        out_channels,
        device,
        **kwargs,
    ):
        super().__init__()  # 调用父类的构造方法
        # 创建一个卷积层对象并指定设备
        self.conv = conv_class(in_channels, out_channels, bias=use_bias, **kwargs).to(
            device
        )
        # 创建一个 BatchNorm 层对象并指定设备
        self.bn = bn_class(out_channels).to(device)

    def forward(self, x):
        x = self.conv(x)  # 对输入 x 执行卷积操作
        return self.bn(x)  # 返回 BatchNorm 处理后的结果


# 定义一个多用户卷积操作类 MultiUserConvOp，继承自 nn.Module
class MultiUserConvOp(nn.Module):
    expected_optimization_count = 3  # 期望的优化次数为 3

    def __init__(
        self,
        conv_class,
        bn_class,
        use_bias,
        in_channels,
        out_channels,
        device,
        **kwargs,
    ):
        super().__init__()  # 调用父类的构造方法
        # 创建第一个卷积层对象并指定设备
        self.conv1 = conv_class(in_channels, out_channels, bias=use_bias, **kwargs).to(
            device
        )
        # 创建第一个 BatchNorm 层对象并指定设备
        self.bn1 = bn_class(out_channels).to(device)
        # 创建第二个卷积层对象并指定设备
        self.conv2 = conv_class(out_channels, out_channels, bias=use_bias, **kwargs).to(
            device
        )
        # 创建第二个 BatchNorm 层对象并指定设备
        self.bn2 = bn_class(out_channels).to(device)
        # 创建第三个卷积层对象并指定设备
        self.conv3 = conv_class(out_channels, out_channels, bias=use_bias, **kwargs).to(
            device
        )
        # 创建第三个 BatchNorm 层对象并指定设备
        self.bn3 = bn_class(out_channels).to(device)

    def forward(self, x):
        # 使用第一个卷积层和 BatchNorm 层
        x = self.bn1(self.conv1(input=x))
        # 对第二个卷积层执行两次前向传播，第二次不使用 efficient_conv_bn_eval 特性
        x = self.bn2(input=self.conv2(self.conv2(x)))
        # 对第三个卷积层执行前向传播，并使用 efficient_conv_bn_eval 特性
        x = self.bn3(input=self.conv3(input=x))
        x = self.bn3(x) + x  # 将第三个 BatchNorm 层处理后的结果与 x 相加
        return x


# 定义一个测试类 EfficientConvBNEvalTemplate，继承自 TestCase
class EfficientConvBNEvalTemplate(TestCase):
    @inductor_config.patch({"efficient_conv_bn_eval_fx_passes": True})
# 如果当前环境有CPU并且没有启用Torch的多进程支持
if HAS_CPU and not torch.backends.mps.is_available():

    # 定义一个名为EfficientConvBNEvalCpuTests的测试类，继承自unittest的TestCase类
    class EfficientConvBNEvalCpuTests(TestCase):
        # 设定测试类的设备为CPU
        device = "cpu"

    # 复制EfficientConvBNEvalTemplate的测试用例到EfficientConvBNEvalCpuTests，设备为CPU
    copy_tests(EfficientConvBNEvalTemplate, EfficientConvBNEvalCpuTests, "cpu")

# 如果当前环境有CUDA并且没有启用地址安全性分析（TEST_WITH_ASAN）
if HAS_CUDA and not TEST_WITH_ASAN:

    # 定义一个名为EfficientConvBNEvalCudaTests的测试类，继承自unittest的TestCase类
    class EfficientConvBNEvalCudaTests(TestCase):
        # 设定测试类的设备为CUDA
        device = "cuda"

    # 复制EfficientConvBNEvalTemplate的测试用例到EfficientConvBNEvalCudaTests，设备为CUDA
    copy_tests(EfficientConvBNEvalTemplate, EfficientConvBNEvalCudaTests, "cuda")

# 删除EfficientConvBNEvalTemplate，此处假设它是不再需要的对象或变量
del EfficientConvBNEvalTemplate

# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    # 导入torch._inductor.test_case模块中的run_tests函数
    from torch._inductor.test_case import run_tests

    # 如果当前环境有CPU或者有CUDA
    if HAS_CPU or HAS_CUDA:
        # 运行测试，需要filelock支持
        run_tests(needs="filelock")
```