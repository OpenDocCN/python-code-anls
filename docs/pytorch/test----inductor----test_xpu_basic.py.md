# `.\pytorch\test\inductor\test_xpu_basic.py`

```py
# Owner(s): ["module: inductor"]
# 导入所需的库和模块
import importlib  # 导入模块动态加载功能
import os  # 导入操作系统相关功能
import sys  # 导入系统相关功能
import unittest  # 导入单元测试框架

import torch  # 导入PyTorch深度学习库
from torch.testing._internal.common_utils import IS_CI, IS_WINDOWS  # 导入内部测试工具及系统平台信息

# 如果运行在Windows且为CI环境
if IS_WINDOWS and IS_CI:
    # 向标准错误流写入消息，说明Windows上的CI环境缺少测试所需的依赖
    sys.stderr.write(
        "Windows CI does not have necessary dependencies for test_xpu_basic yet\n"
    )
    # 如果当前脚本是主程序
    if __name__ == "__main__":
        sys.exit(0)  # 正常退出脚本
    raise unittest.SkipTest("requires sympy/functorch/filelock")  # 抛出跳过测试的异常，需要依赖sympy/functorch/filelock

importlib.import_module("filelock")  # 动态导入filelock模块

# 获取当前脚本的测试目录
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# 将测试目录添加到系统路径中
sys.path.append(pytorch_test_dir)
# 从inductor.test_torchinductor模块导入check_model_gpu和TestCase类
from inductor.test_torchinductor import check_model_gpu, TestCase


# TODO: Remove this file.
# This is a temporary test case to test the base functionality of first Intel GPU Inductor integration.
# We are working on reuse and pass the test cases in test/inductor/*  step by step.
# Will remove this file when pass full test in test/inductor/*.
# 这个文件是一个临时的测试案例，用于逐步测试第一个Intel GPU Inductor集成的基本功能。
# 我们正在逐步重用并通过test/inductor/*中的测试用例。
# 当在test/inductor/*中完全通过测试后，将删除此文件。

class XpuBasicTests(TestCase):
    common = check_model_gpu  # 设置公共属性common为check_model_gpu函数
    device = "xpu"  # 设置设备类型为"xpu"

    def test_add(self):
        def fn(a, b):
            return a + b  # 定义一个加法函数fn

        self.common(fn, (torch.rand(2, 3, 16, 16), torch.rand(2, 3, 16, 16)))  # 调用common方法进行测试

    def test_sub(self):
        def fn(a, b):
            return a - b  # 定义一个减法函数fn

        self.common(fn, (torch.rand(2, 3, 16, 16), torch.rand(2, 3, 16, 16)))  # 调用common方法进行测试

    def test_mul(self):
        def fn(a, b):
            return a * b  # 定义一个乘法函数fn

        self.common(fn, (torch.rand(2, 3, 16, 16), torch.rand(2, 3, 16, 16)))  # 调用common方法进行测试

    def test_div(self):
        def fn(a, b):
            return a / b  # 定义一个除法函数fn

        self.common(fn, (torch.rand(2, 3, 16, 16), torch.rand(2, 3, 16, 16)))  # 调用common方法进行测试


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests  # 导入测试运行函数
    from torch.testing._internal.inductor_utils import HAS_XPU  # 导入是否有XPU的判断函数

    if HAS_XPU:  # 如果有XPU
        run_tests(needs="filelock")  # 运行测试，需要filelock依赖
```