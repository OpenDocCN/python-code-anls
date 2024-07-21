# `.\pytorch\test\inductor\test_extension_backend.py`

```py
# Owner(s): ["module: inductor"]

# 引入标准库和第三方库
import os
import shutil
import sys
import unittest

# 引入 PyTorch 相关模块
import torch
import torch._dynamo
import torch.utils.cpp_extension
from torch._C import FileCheck

# 尝试引入自定义的 C++ 扩展模块
try:
    from extension_backends.cpp.extension_codegen_backend import (
        ExtensionCppWrapperCodegen,
        ExtensionScheduling,
        ExtensionWrapperCodegen,
    )
except ImportError:
    from .extension_backends.cpp.extension_codegen_backend import (
        ExtensionCppWrapperCodegen,
        ExtensionScheduling,
        ExtensionWrapperCodegen,
    )

# 引入与编译器相关的配置和工具
import torch._inductor.config as config
from torch._inductor import cpu_vec_isa, metrics
from torch._inductor.codegen import cpp_utils
from torch._inductor.codegen.common import (
    get_scheduling_for_device,
    get_wrapper_codegen_for_device,
    register_backend_for_device,
)
from torch.testing._internal.common_utils import IS_FBCODE, IS_MACOS

# 尝试引入测试模块
try:
    try:
        from . import test_torchinductor
    except ImportError:
        import test_torchinductor
except unittest.SkipTest:
    # 如果是因为 unittest.SkipTest 异常，则根据条件退出程序或者抛出异常
    if __name__ == "__main__":
        sys.exit(0)
    raise

# 将 test_torchinductor 模块的函数赋值给全局变量
run_and_get_cpp_code = test_torchinductor.run_and_get_cpp_code
# 将 test_torchinductor 模块的 TestCase 类赋值给全局变量
TestCase = test_torchinductor.TestCase


# 清理构建路径的函数
def remove_build_path():
    if sys.platform == "win32":
        # Windows 下不清理构建文件夹
        return
    # 获取默认的构建根目录，并删除它
    default_build_root = torch.utils.cpp_extension.get_default_build_root()
    if os.path.exists(default_build_root):
        shutil.rmtree(default_build_root, ignore_errors=True)


# 标记为跳过测试（如果在 FBCODE 环境下）
@unittest.skipIf(IS_FBCODE, "cpp_extension doesn't work in fbcode right now")
class ExtensionBackendTests(TestCase):
    module = None

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        # 构建扩展模块
        remove_build_path()
        source_file_path = os.path.dirname(os.path.abspath(__file__))
        source_file = os.path.join(
            source_file_path, "extension_backends/cpp/extension_device.cpp"
        )
        cls.module = torch.utils.cpp_extension.load(
            name="extension_device",
            sources=[
                str(source_file),
            ],
            extra_cflags=["-g"],
            verbose=True,
        )

    @classmethod
    def tearDownClass(cls):
        cls._stack.close()
        super().tearDownClass()

        # 清理构建路径
        remove_build_path()

    def setUp(self):
        torch._dynamo.reset()
        super().setUp()

        # C++ 扩展使用相对路径，相对于此文件，所以暂时更改工作目录
        self.old_working_dir = os.getcwd()
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        assert self.module is not None

    def tearDown(self):
        super().tearDown()
        torch._dynamo.reset()

        # 恢复工作目录至 setUp 前的状态
        os.chdir(self.old_working_dir)
    # 定义测试函数，用于测试设备注册和操作
    def test_open_device_registration(self):
        # 使用 torch.utils.rename_privateuse1_backend 将设备扩展到私有使用1后端
        torch.utils.rename_privateuse1_backend("extension_device")
        # 注册设备模块到 "extension_device"
        torch._register_device_module("extension_device", self.module)

        # 为 "extension_device" 注册后端，调度器是 ExtensionScheduling，
        # 包装器代码生成器是 ExtensionWrapperCodegen，
        # C++ 包装器代码生成器是 ExtensionCppWrapperCodegen
        register_backend_for_device(
            "extension_device",
            ExtensionScheduling,
            ExtensionWrapperCodegen,
            ExtensionCppWrapperCodegen,
        )

        # 断言获取 "extension_device" 的调度器是否为 ExtensionScheduling
        self.assertTrue(
            get_scheduling_for_device("extension_device") == ExtensionScheduling
        )
        # 断言获取 "extension_device" 的包装器代码生成器是否为 ExtensionWrapperCodegen
        self.assertTrue(
            get_wrapper_codegen_for_device("extension_device")
            == ExtensionWrapperCodegen
        )
        # 断言获取 "extension_device" 的包装器代码生成器（带有 C++ 标志）是否为 ExtensionCppWrapperCodegen
        self.assertTrue(
            get_wrapper_codegen_for_device("extension_device", True)
            == ExtensionCppWrapperCodegen
        )

        # 断言 self.module 是否未调用自定义操作
        self.assertFalse(self.module.custom_op_called())
        # 获取自定义设备并分配给 device
        device = self.module.custom_device()
        # 创建 tensor x，并使用 fill_ 方法填充为 1，将其分配给 device
        x = torch.empty(2, 16).to(device=device).fill_(1)
        # 断言 self.module 是否调用了自定义操作
        self.assertTrue(self.module.custom_op_called())
        # 创建 tensor y，并使用 fill_ 方法填充为 2，将其分配给 device
        y = torch.empty(2, 16).to(device=device).fill_(2)
        # 创建 tensor z，并使用 fill_ 方法填充为 3，将其分配给 device
        z = torch.empty(2, 16).to(device=device).fill_(3)
        # 创建参考 tensor ref，并使用 fill_ 方法填充为 5
        ref = torch.empty(2, 16).fill_(5)

        # 断言 tensor x 的设备是否为 device
        self.assertTrue(x.device == device)
        # 断言 tensor y 的设备是否为 device
        self.assertTrue(y.device == device)
        # 断言 tensor z 的设备是否为 device
        self.assertTrue(z.device == device)

        # 定义函数 fn，计算 a * b + c
        def fn(a, b, c):
            return a * b + c

        # 将 "extension_device" 映射到 ATen 的私有使用1
        cpp_utils.DEVICE_TO_ATEN["extension_device"] = "at::kPrivateUse1"

        # 遍历 cpp_wrapper_flag 的值 [True, False]
        for cpp_wrapper_flag in [True, False]:
            # 使用 config.patch 设置 "cpp_wrapper" 为 cpp_wrapper_flag
            with config.patch({"cpp_wrapper": cpp_wrapper_flag}):
                # 重置度量指标
                metrics.reset()
                # 使用 torch.compile() 优化函数 fn
                opt_fn = torch.compile()(fn)
                # 运行优化后的函数 opt_fn，并获取生成的 C++ 代码
                _, code = run_and_get_cpp_code(opt_fn, x, y, z)
                # 如果 CPU 的矢量 ISA 有效，则 load_expr 为 "loadu"，否则为 " = in_ptr0[static_cast<long>(i0)];"
                if cpu_vec_isa.valid_vec_isa_list():
                    load_expr = "loadu"
                else:
                    load_expr = " = in_ptr0[static_cast<long>(i0)];"
                # 使用 FileCheck() 检查代码中是否包含 "void"、load_expr、"extension_device"
                FileCheck().check("void").check(load_expr).check(
                    "extension_device"
                ).run(code)
                # 运行优化后的函数 opt_fn，获取结果 res
                opt_fn(x, y, z)
                res = opt_fn(x, y, z)
                # 断言 res 是否与参考结果 ref 在 CPU 设备上相等
                self.assertEqual(ref, res.to(device="cpu"))
# 如果当前脚本作为主程序运行
if __name__ == "__main__":
    # 从 torch._inductor.test_case 模块导入 run_tests 函数
    from torch._inductor.test_case import run_tests
    # 从 torch.testing._internal.inductor_utils 模块导入 HAS_CPU 常量
    from torch.testing._internal.inductor_utils import HAS_CPU

    # 检查是否具有 CPU，并且不是在 macOS 系统下，并且不是在 fbcode 环境下
    if HAS_CPU and not IS_MACOS and not IS_FBCODE:
        # 运行测试，指定需要使用 "filelock" 功能
        run_tests(needs="filelock")
```