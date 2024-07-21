# `.\pytorch\test\jit\test_backend_nnapi.py`

```py
# Owner(s): ["oncall: jit"]

import os  # 导入操作系统模块
import sys  # 导入系统模块
import unittest  # 导入单元测试模块
from pathlib import Path  # 导入路径操作模块

import torch  # 导入PyTorch主模块
import torch._C  # 导入PyTorch C++扩展模块
from torch.testing._internal.common_utils import IS_FBCODE, skipIfTorchDynamo  # 导入测试工具函数

# hacky way to skip these tests in fbcode:
# during test execution in fbcode, test_nnapi is available during test discovery,
# but not during test execution. So we can't try-catch here, otherwise it'll think
# it sees tests but then fails when it tries to actually run them.
if not IS_FBCODE:
    from test_nnapi import TestNNAPI  # 导入测试NNAPI的模块
    HAS_TEST_NNAPI = True
else:
    from torch.testing._internal.common_utils import TestCase as TestNNAPI  # 导入测试用例模块
    HAS_TEST_NNAPI = False


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)  # 将测试目录添加到系统路径中，使得其中的辅助文件可以被导入

if __name__ == "__main__":
    raise RuntimeError(
        "This test file is not meant to be run directly, use:\n\n"
        "\tpython test/test_jit.py TESTNAME\n\n"
        "instead."
    )

"""
Unit Tests for Nnapi backend with delegate
Inherits most tests from TestNNAPI, which loads Android NNAPI models
without the delegate API.
"""
# First skip is needed for IS_WINDOWS or IS_MACOS to skip the tests.
torch_root = Path(__file__).resolve().parent.parent.parent
lib_path = torch_root / "build" / "lib" / "libnnapi_backend.so"


@skipIfTorchDynamo("weird py38 failures")
@unittest.skipIf(
    not os.path.exists(lib_path),
    "Skipping the test as libnnapi_backend.so was not found",
)
@unittest.skipIf(IS_FBCODE, "test_nnapi.py not found")
class TestNnapiBackend(TestNNAPI):
    def setUp(self):
        super().setUp()

        # Save default dtype
        module = torch.nn.PReLU()  # 创建PReLU模块实例
        self.default_dtype = module.weight.dtype  # 保存模块的默认数据类型
        # Change dtype to float32 (since a different unit test changed dtype to float64,
        # which is not supported by the Android NNAPI delegate)
        # Float32 should typically be the default in other files.
        torch.set_default_dtype(torch.float32)  # 设置PyTorch的默认数据类型为float32

        # Load nnapi delegate library
        torch.ops.load_library(str(lib_path))  # 加载nnapi委托库

    # Override
    def call_lowering_to_nnapi(self, traced_module, args):
        compile_spec = {"forward": {"inputs": args}}  # 定义编译规范
        return torch._C._jit_to_backend("nnapi", traced_module, compile_spec)  # 使用nnapi后端进行模块降级

    def test_tensor_input(self):
        # Lower a simple module
        args = torch.tensor([[1.0, -1.0, 2.0, -2.0]]).unsqueeze(-1).unsqueeze(-1)  # 创建输入张量
        module = torch.nn.PReLU()  # 创建PReLU模块实例
        traced = torch.jit.trace(module, args)  # 对模块进行追踪

        # Argument input is a single Tensor
        self.call_lowering_to_nnapi(traced, args)  # 调用使用单个张量作为输入的降级方法
        # Argument input is a Tensor in a list
        self.call_lowering_to_nnapi(traced, [args])  # 调用使用张量列表作为输入的降级方法

    # Test exceptions for incorrect compile specs
    # 定义名为 test_compile_spec_santiy 的测试函数，用于测试某个功能的正确性
    def test_compile_spec_santiy(self):
        # 创建一个包含张量的变量 args，张量中包含四个值，并进行维度扩展使其变成四维张量
        args = torch.tensor([[1.0, -1.0, 2.0, -2.0]]).unsqueeze(-1).unsqueeze(-1)
        # 创建一个 PReLU 模块的实例
        module = torch.nn.PReLU()
        # 使用 torch.jit.trace 对 PReLU 模块进行追踪，以便后续的 JIT 编译
        traced = torch.jit.trace(module, args)

        # 定义一个错误消息的后半部分，使用原始字符串表示法 (raw string)
        errorMsgTail = r"""
        # 缺少 "forward" 键
        compile_spec = {"backward": {"inputs": args}}
        # 使用断言检查是否引发 RuntimeError，并包含特定错误消息
        with self.assertRaisesRegex(
            RuntimeError,
            'method_compile_spec does not contain the "forward" key.' + errorMsgTail,
        ):
            # 调用 Torch 的底层函数，期望引发错误
            torch._C._jit_to_backend("nnapi", traced, compile_spec)

        # "forward" 键下没有字典
        compile_spec = {"forward": 1}
        # 使用断言检查是否引发 RuntimeError，并包含特定错误消息
        with self.assertRaisesRegex(
            RuntimeError,
            'method_compile_spec does not contain a dictionary with an "inputs" key, '
            'under it\'s "forward" key.' + errorMsgTail,
        ):
            # 调用 Torch 的底层函数，期望引发错误
            torch._C._jit_to_backend("nnapi", traced, compile_spec)

        # "forward" 键下的字典没有 "inputs" 键
        compile_spec = {"forward": {"not inputs": args}}
        # 使用断言检查是否引发 RuntimeError，并包含特定错误消息
        with self.assertRaisesRegex(
            RuntimeError,
            'method_compile_spec does not contain a dictionary with an "inputs" key, '
            'under it\'s "forward" key.' + errorMsgTail,
        ):
            # 调用 Torch 的底层函数，期望引发错误
            torch._C._jit_to_backend("nnapi", traced, compile_spec)

        # "inputs" 键下没有 Tensor 或 TensorList
        compile_spec = {"forward": {"inputs": 1}}
        # 使用断言检查是否引发 RuntimeError，并包含特定错误消息
        with self.assertRaisesRegex(
            RuntimeError,
            'method_compile_spec does not contain either a Tensor or TensorList, under it\'s "inputs" key.'
            + errorMsgTail,
        ):
            # 调用 Torch 的底层函数，期望引发错误
            torch._C._jit_to_backend("nnapi", traced, compile_spec)

        compile_spec = {"forward": {"inputs": [1]}}
        # 使用断言检查是否引发 RuntimeError，并包含特定错误消息
        with self.assertRaisesRegex(
            RuntimeError,
            'method_compile_spec does not contain either a Tensor or TensorList, under it\'s "inputs" key.'
            + errorMsgTail,
        ):
            # 调用 Torch 的底层函数，期望引发错误
            torch._C._jit_to_backend("nnapi", traced, compile_spec)
```