# `.\pytorch\test\inductor\test_triton_extension_backend.py`

```
# Owner(s): ["module: inductor"]
# 导入所需的库和模块
import random  # 导入随机数生成模块
import string  # 导入字符串处理模块
import sys  # 导入系统相关模块
import unittest  # 导入单元测试框架

import torch  # 导入PyTorch深度学习框架
import torch._dynamo  # 导入PyTorch的私有模块_dynamo
import torch.utils.cpp_extension  # 导入PyTorch的C++扩展工具模块

try:
    # 尝试导入Triton后端相关模块
    from extension_backends.triton.device_interface import DeviceInterface
    from extension_backends.triton.extension_codegen_backend import (
        CPUDeviceOpOverrides,
        ExtensionScheduling,
        ExtensionWrapperCodegen,
    )
except ImportError:
    # 如果导入失败，则从当前目录中导入
    from .extension_backends.triton.device_interface import DeviceInterface
    from .extension_backends.triton.extension_codegen_backend import (
        CPUDeviceOpOverrides,
        ExtensionScheduling,
        ExtensionWrapperCodegen,
    )

from torch._C import FileCheck  # 导入PyTorch的C++前端FileCheck
from torch._dynamo import device_interface  # 导入PyTorch的私有模块device_interface
from torch._inductor import metrics  # 导入PyTorch的私有模块metrics
from torch._inductor.codegen.common import (
    get_scheduling_for_device,  # 导入获取设备调度的函数
    get_wrapper_codegen_for_device,  # 导入获取设备包装代码生成器的函数
    register_backend_for_device,  # 导入注册设备后端的函数
    register_device_op_overrides,  # 导入注册设备操作重写的函数
)
from torch._inductor.utils import get_triton_code  # 导入获取Triton代码的函数
from torch.testing._internal.common_utils import IS_MACOS  # 导入判断是否为macOS的常量

try:
    try:
        # 尝试从当前目录中导入test_torchinductor模块
        from . import test_torchinductor
    except ImportError:
        # 如果导入失败，则从顶级目录中导入
        import test_torchinductor
except unittest.SkipTest:
    # 如果单元测试被跳过，则根据情况退出或抛出异常
    if __name__ == "__main__":
        sys.exit(0)
    raise

TestCase = test_torchinductor.TestCase  # 设置TestCase为test_torchinductor模块中的TestCase类


def mock_triton_hash_with_backend(*args, **kwargs):
    # 生成长度为64的随机字符串，用于模拟triton_hash_with_backend函数，
    # 因为我们没有triton后端
    return "".join(random.choices(string.ascii_uppercase + string.digits, k=64))


class TritonExtensionBackendTests(TestCase):
    """
    Test creating a backend for inductor with Triton scheduling.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()  # 调用父类的setUpClass方法

    @classmethod
    def tearDownClass(cls):
        cls._stack.close()  # 关闭堆栈
        super().tearDownClass()  # 调用父类的tearDownClass方法

    def setUp(self):
        torch._dynamo.reset()  # 重置torch._dynamo模块的状态
        super().setUp()  # 调用父类的setUp方法

    def tearDown(self):
        super().tearDown()  # 调用父类的tearDown方法
        torch._dynamo.reset()  # 重置torch._dynamo模块的状态
    # 定义一个测试方法，用于测试设备注册的功能
    def test_open_device_registration(self):
        # 注册 "cpu" 设备的后端调度器和包装器代码生成器
        register_backend_for_device("cpu", ExtensionScheduling, ExtensionWrapperCodegen)
        # 注册 "cpu" 设备的操作重写
        register_device_op_overrides("cpu", CPUDeviceOpOverrides())
        # 为 "cpu" 设备注册设备接口
        device_interface.register_interface_for_device("cpu", DeviceInterface)

        # 断言获取 "cpu" 设备的调度器是否为 ExtensionScheduling
        self.assertTrue(get_scheduling_for_device("cpu") == ExtensionScheduling)
        # 断言获取 "cpu" 设备的包装器代码生成器是否为 ExtensionWrapperCodegen
        self.assertTrue(
            get_wrapper_codegen_for_device("cpu") == ExtensionWrapperCodegen
        )
        # 断言获取 "cpu" 设备的设备接口是否为 DeviceInterface
        self.assertTrue(
            device_interface.get_interface_for_device("cpu") == DeviceInterface
        )

        # 创建一个名为 "cpu" 的 Torch 设备对象
        device = torch.device("cpu")
        # 创建一个大小为 (2, 16) 的未初始化张量，并用值 1 填充，然后将其移动到指定设备上
        x = torch.empty(2, 16).fill_(1).to(device)

        # 定义一个函数 foo，对输入张量 x 进行计算并返回结果
        def foo(x):
            return torch.sin(x) + x.min()

        # 重置度量指标
        metrics.reset()
        # 编译函数 foo，生成优化后的函数
        opt_fn = torch.compile(foo)

        # 由于没有 Triton 后端，需要模拟 triton_hash_with_backend 函数
        with unittest.mock.patch(
            "torch.utils._triton.triton_hash_with_backend",
            new=mock_triton_hash_with_backend,
        ):
            # 获取优化后函数 opt_fn 在输入数据 x 上的 Triton 代码
            code = get_triton_code(opt_fn, x)

        # 使用 FileCheck 检查生成的 Triton 代码中的特定内容
        FileCheck().check("import triton").check("@triton.jit").check(
            "tl_math.sin"
        ).check("device_str='cpu'").run(code)
# 如果当前模块是作为主程序执行（而不是被导入到其它模块中执行）
if __name__ == "__main__":
    # 从torch._inductor.test_case模块中导入run_tests函数
    from torch._inductor.test_case import run_tests
    # 从torch.testing._internal.inductor_utils模块中导入HAS_CPU常量
    from torch.testing._internal.inductor_utils import HAS_CPU
    
    # 如果HAS_CPU为真且不是在MACOS平台上
    if HAS_CPU and not IS_MACOS:
        # 执行测试函数
        run_tests()
```