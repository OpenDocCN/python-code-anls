# `.\pytorch\test\test_cpp_extensions_aot.py`

```py
# Owner(s): ["module: cpp-extensions"]

# 引入必要的库和模块
import os
import re
import unittest
from itertools import repeat
from typing import get_args, get_origin, Union

# 引入 PyTorch 相关模块
import torch
import torch.backends.cudnn

# 引入 PyTorch 内部测试工具和 CUDA 相关功能
import torch.testing._internal.common_utils as common
import torch.utils.cpp_extension
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_utils import IS_WINDOWS, skipIfTorchDynamo

try:
    # 尝试导入 pytest，标记是否成功导入
    import pytest
    HAS_PYTEST = True
except ImportError as e:
    # 若导入失败，则标记为没有 pytest
    HAS_PYTEST = False

# TODO: 重写这些测试，以便可以通过 pytest 收集，而不使用 run_test.py
try:
    if HAS_PYTEST:
        # 尝试导入特定的 cpp 扩展模块，如果导入失败，则跳过测试
        cpp_extension = pytest.importorskip("torch_test_cpp_extension.cpp")
        maia_extension = pytest.importorskip("torch_test_cpp_extension.maia")
        rng_extension = pytest.importorskip("torch_test_cpp_extension.rng")
    else:
        # 若没有 pytest，则直接导入 cpp 扩展模块
        import torch_test_cpp_extension.cpp as cpp_extension
        import torch_test_cpp_extension.maia as maia_extension
        import torch_test_cpp_extension.rng as rng_extension
except ImportError as e:
    # 若导入失败，抛出运行时错误，并指导正确运行测试的方式
    raise RuntimeError(
        "test_cpp_extensions_aot.py cannot be invoked directly. Run "
        "`python run_test.py -i test_cpp_extensions_aot_ninja` instead."
    ) from e

# 标记为 Dynamo 严格测试的类
@torch.testing._internal.common_utils.markDynamoStrictTest
class TestCppExtensionAOT(common.TestCase):
    """Tests ahead-of-time cpp extensions

    NOTE: run_test.py's test_cpp_extensions_aot_ninja target
    also runs this test case, but with ninja enabled. If you are debugging
    a test failure here from the CI, check the logs for which target
    (test_cpp_extensions_aot_no_ninja vs test_cpp_extensions_aot_ninja)
    failed.
    """

    # 测试 cpp 扩展函数的功能
    def test_extension_function(self):
        x = torch.randn(4, 4)
        y = torch.randn(4, 4)
        z = cpp_extension.sigmoid_add(x, y)
        self.assertEqual(z, x.sigmoid() + y.sigmoid())
        # 测试 Pybind 对 torch.dtype 的支持转换
        self.assertEqual(
            str(torch.float32), str(cpp_extension.get_math_type(torch.half))
        )

    # 测试 cpp 扩展模块的功能
    def test_extension_module(self):
        mm = cpp_extension.MatrixMultiplier(4, 8)
        weights = torch.rand(8, 4, dtype=torch.double)
        expected = mm.get().mm(weights)
        result = mm.forward(weights)
        self.assertEqual(expected, result)

    # 测试 cpp 扩展模块的反向传播功能
    def test_backward(self):
        mm = cpp_extension.MatrixMultiplier(4, 8)
        weights = torch.rand(8, 4, dtype=torch.double, requires_grad=True)
        result = mm.forward(weights)
        result.sum().backward()
        tensor = mm.get()

        expected_weights_grad = tensor.t().mm(torch.ones([4, 4], dtype=torch.double))
        self.assertEqual(weights.grad, expected_weights_grad)

        expected_tensor_grad = torch.ones([4, 4], dtype=torch.double).mm(weights.t())
        self.assertEqual(tensor.grad, expected_tensor_grad)

    # 如果没有 CUDA 支持，则跳过这个测试
    @unittest.skipIf(not TEST_CUDA, "CUDA not found")
    def test_cuda_extension(self):
        # 导入自定义的 CUDA 扩展模块
        import torch_test_cpp_extension.cuda as cuda_extension

        # 在 CUDA 设备上创建大小为 100 的零张量
        x = torch.zeros(100, device="cuda", dtype=torch.float32)
        y = torch.zeros(100, device="cuda", dtype=torch.float32)

        # 调用 CUDA 扩展模块中的 sigmoid_add 函数并将结果移动到 CPU
        z = cuda_extension.sigmoid_add(x, y).cpu()

        # 验证结果是否为全为 1 的张量
        # 2 * sigmoid(0) = 2 * 0.5 = 1
        self.assertEqual(z, torch.ones_like(z))

    @unittest.skipIf(not torch.backends.mps.is_available(), "MPS not found")
    def test_mps_extension(self):
        # 导入自定义的 MPS 扩展模块
        import torch_test_cpp_extension.mps as mps_extension

        tensor_length = 100000
        # 在 CPU 设备上创建指定长度的随机张量
        x = torch.randn(tensor_length, device="cpu", dtype=torch.float32)
        y = torch.randn(tensor_length, device="cpu", dtype=torch.float32)

        # 调用 MPS 扩展模块中的两个函数并比较结果
        cpu_output = mps_extension.get_cpu_add_output(x, y)
        mps_output = mps_extension.get_mps_add_output(x.to("mps"), y.to("mps"))

        # 验证 MPS 输出是否与 CPU 输出相同
        self.assertEqual(cpu_output, mps_output.to("cpu"))

    @common.skipIfRocm
    @unittest.skipIf(common.IS_WINDOWS, "Windows not supported")
    @unittest.skipIf(not TEST_CUDA, "CUDA not found")
    def test_cublas_extension(self):
        # 导入自定义的 cuBLAS 扩展模块
        from torch_test_cpp_extension import cublas_extension

        # 在 CUDA 设备上创建大小为 100 的零张量
        x = torch.zeros(100, device="cuda", dtype=torch.float32)

        # 调用 cuBLAS 扩展模块中的函数并验证结果
        z = cublas_extension.noop_cublas_function(x)
        self.assertEqual(z, x)

    @common.skipIfRocm
    @unittest.skipIf(common.IS_WINDOWS, "Windows not supported")
    @unittest.skipIf(not TEST_CUDA, "CUDA not found")
    def test_cusolver_extension(self):
        # 导入自定义的 cuSolver 扩展模块
        from torch_test_cpp_extension import cusolver_extension

        # 在 CUDA 设备上创建大小为 100 的零张量
        x = torch.zeros(100, device="cuda", dtype=torch.float32)

        # 调用 cuSolver 扩展模块中的函数并验证结果
        z = cusolver_extension.noop_cusolver_function(x)
        self.assertEqual(z, x)

    @unittest.skipIf(IS_WINDOWS, "Not available on Windows")
    def test_no_python_abi_suffix_sets_the_correct_library_name(self):
        # 对于这个测试，run_test.py 将在 cpp_extensions/no_python_abi_suffix_test 文件夹中调用
        # `python setup.py install`，其中 `BuildExtension` 类设置了 `no_python_abi_suffix` 选项为 `True`。
        # 这意味着在 Python 3 中，生成的共享库在库后缀（例如 "so"）之前不会有类似 "cpython-37m-x86_64-linux-gnu" 的 ABI 后缀。
        root = os.path.join("cpp_extensions", "no_python_abi_suffix_test", "build")
        # 在目录中查找以 "so" 结尾的文件
        matches = [f for _, _, fs in os.walk(root) for f in fs if f.endswith("so")]
        # 验证是否只有一个匹配项，并且名称为 "no_python_abi_suffix_test.so"
        self.assertEqual(len(matches), 1, msg=str(matches))
        self.assertEqual(matches[0], "no_python_abi_suffix_test.so", msg=str(matches))

    def test_optional(self):
        # 调用带有可选参数的自定义 C++ 扩展函数
        has_value = cpp_extension.function_taking_optional(torch.ones(5))
        self.assertTrue(has_value)
        has_value = cpp_extension.function_taking_optional(None)
        self.assertFalse(has_value)
    # 使用 unittest 装饰器标记此测试函数，如果环境变量 USE_NINJA 的值为 "0"，则跳过此测试
    @unittest.skipIf(
        os.getenv("USE_NINJA", "0") == "0",
        "cuda extension with dlink requires ninja to build",
    )
    # 定义测试函数 test_cuda_dlink_libs，用于测试 CUDA 动态链接库
    def test_cuda_dlink_libs(self):
        # 导入 torch_test_cpp_extension 模块中的 cuda_dlink 函数
        from torch_test_cpp_extension import cuda_dlink

        # 在 CUDA 设备上生成随机张量 a 和 b
        a = torch.randn(8, dtype=torch.float, device="cuda")
        b = torch.randn(8, dtype=torch.float, device="cuda")
        # 计算参考结果，将 a 和 b 张量相加
        ref = a + b
        # 调用 cuda_dlink 模块中的 add 函数执行 CUDA 加法运算
        test = cuda_dlink.add(a, b)
        # 使用 unittest 的断言方法，验证 test 是否与 ref 相等
        self.assertEqual(test, ref)
@torch.testing._internal.common_utils.markDynamoStrictTest
class TestPybindTypeCasters(common.TestCase):
    """Pybind tests for ahead-of-time cpp extensions

    These tests verify the types returned from cpp code using custom type
    casters. By exercising pybind, we also verify that the type casters work
    properly.

    For each type caster in `torch/csrc/utils/pybind.h` we create a pybind
    function that takes no arguments and returns the type_caster type. The
    second argument to `PYBIND11_TYPE_CASTER` should be the type we expect to
    receive in python, in these tests we verify this at run-time.
    """

    @staticmethod
    def expected_return_type(func):
        """
        Our Pybind functions have a signature of the form `() -> return_type`.
        """
        # Imports needed for the `eval` below.
        from typing import List, Tuple  # noqa: F401

        return eval(re.search("-> (.*)\n", func.__doc__).group(1))
        # 解析函数文档字符串，获取函数返回类型并返回

    def check(self, func):
        val = func()
        expected = self.expected_return_type(func)
        origin = get_origin(expected)
        if origin is list:
            self.check_list(val, expected)
        elif origin is tuple:
            self.check_tuple(val, expected)
        else:
            self.assertIsInstance(val, expected)
        # 检查函数返回值类型与预期类型是否匹配，支持列表和元组类型的特殊检查

    def check_list(self, vals, expected):
        self.assertIsInstance(vals, list)
        list_type = get_args(expected)[0]
        for val in vals:
            self.assertIsInstance(val, list_type)
        # 检查返回的列表是否为列表类型，并逐个检查列表元素的类型是否符合预期

    def check_tuple(self, vals, expected):
        self.assertIsInstance(vals, tuple)
        tuple_types = get_args(expected)
        if tuple_types[1] is ...:
            tuple_types = repeat(tuple_types[0])
        for val, tuple_type in zip(vals, tuple_types):
            self.assertIsInstance(val, tuple_type)
        # 检查返回的元组是否为元组类型，并逐个检查元组元素的类型是否符合预期

    def check_union(self, funcs):
        """Special handling for Union type casters.

        A single cpp type can sometimes be cast to different types in python.
        In these cases we expect to get exactly one function per python type.
        """
        # Verify that all functions have the same return type.
        union_type = {self.expected_return_type(f) for f in funcs}
        assert len(union_type) == 1
        union_type = union_type.pop()
        self.assertIs(Union, get_origin(union_type))
        # SymInt is inconvenient to test, so don't require it
        expected_types = set(get_args(union_type)) - {torch.SymInt}
        for func in funcs:
            val = func()
            for tp in expected_types:
                if isinstance(val, tp):
                    expected_types.remove(tp)
                    break
            else:
                raise AssertionError(f"{val} is not an instance of {expected_types}")
        self.assertFalse(
            expected_types, f"Missing functions for types {expected_types}"
        )
        # 特殊处理联合类型转换器，验证所有函数的返回类型一致性，并检查返回值是否符合预期
    # 定义一个测试方法，用于测试 Python 绑定库的返回类型
    def test_pybind_return_types(self):
        # 定义要测试的函数列表，每个函数都来自 cpp_extension 模块
        functions = [
            cpp_extension.get_complex,
            cpp_extension.get_device,
            cpp_extension.get_generator,
            cpp_extension.get_intarrayref,
            cpp_extension.get_memory_format,
            cpp_extension.get_storage,
            cpp_extension.get_symfloat,
            cpp_extension.get_symintarrayref,
            cpp_extension.get_tensor,
        ]
        # 定义一个包含函数列表的列表，目前只包含一个元素，元素是获取符号整数的函数
        union_functions = [
            [cpp_extension.get_symint],
        ]
        # 遍历测试函数列表
        for func in functions:
            # 使用子测试功能，检查每个函数的返回值类型
            with self.subTest(msg=f"check {func.__name__}"):
                self.check(func)
        # 遍历联合函数列表
        for funcs in union_functions:
            # 使用子测试功能，检查联合函数组的返回值类型
            with self.subTest(msg=f"check {[f.__name__ for f in funcs]}"):
                self.check_union(funcs)
# 标记为 Dynamo 严格测试的测试类
@torch.testing._internal.common_utils.markDynamoStrictTest
class TestMAIATensor(common.TestCase):
    
    # 测试当使用未注册设备 'maia' 时是否抛出 RuntimeError 异常
    def test_unregistered(self):
        a = torch.arange(0, 10, device="cpu")
        with self.assertRaisesRegex(RuntimeError, "Could not run"):
            b = torch.arange(0, 10, device="maia")

    # 如果 TorchDynamo 不支持 'maia' 设备，则跳过测试
    @skipIfTorchDynamo("dynamo cannot model maia device")
    def test_zeros(self):
        # 创建一个空的 Tensor，在 CPU 上
        a = torch.empty(5, 5, device="cpu")
        # 断言 Tensor 的设备为 CPU
        self.assertEqual(a.device, torch.device("cpu"))

        # 创建一个空的 Tensor，在 'maia' 设备上
        b = torch.empty(5, 5, device="maia")
        # 断言 Tensor 的设备为 'maia' 设备的第一个设备
        self.assertEqual(b.device, torch.device("maia", 0))
        # 断言调用 maia_extension.get_test_int() 返回值为 0
        self.assertEqual(maia_extension.get_test_int(), 0)
        # 断言当前默认的数据类型与 b 的数据类型一致
        self.assertEqual(torch.get_default_dtype(), b.dtype)

        # 创建一个指定数据类型的 Tensor，在 'maia' 设备上
        c = torch.empty((5, 5), dtype=torch.int64, device="maia")
        # 断言调用 maia_extension.get_test_int() 返回值为 0
        self.assertEqual(maia_extension.get_test_int(), 0)
        # 断言 Tensor 的数据类型为 torch.int64
        self.assertEqual(torch.int64, c.dtype)

    # 测试在 'maia' 设备上创建带梯度的 Tensor
    def test_add(self):
        a = torch.empty(5, 5, device="maia", requires_grad=True)
        # 断言调用 maia_extension.get_test_int() 返回值为 0
        self.assertEqual(maia_extension.get_test_int(), 0)

        b = torch.empty(5, 5, device="maia")
        # 断言调用 maia_extension.get_test_int() 返回值为 0
        self.assertEqual(maia_extension.get_test_int(), 0)

        c = a + b
        # 断言调用 maia_extension.get_test_int() 返回值为 1
        self.assertEqual(maia_extension.get_test_int(), 1)

    # 测试在 'maia' 设备上使用卷积后端重写
    def test_conv_backend_override(self):
        # 为了简化测试，使用 4 维输入，避免在 _convolution 中进行 view4d 的重写
        input = torch.empty(2, 4, 10, 2, device="maia", requires_grad=True)
        weight = torch.empty(6, 4, 2, 2, device="maia", requires_grad=True)
        bias = torch.empty(6, device="maia")

        # 确保前向传播已重写
        out = torch.nn.functional.conv2d(input, weight, bias, 2, 0, 1, 1)
        # 断言调用 maia_extension.get_test_int() 返回值为 2
        self.assertEqual(maia_extension.get_test_int(), 2)
        # 断言输出的形状与输入的形状第一维相同
        self.assertEqual(out.shape[0], input.shape[0])
        # 断言输出的形状与权重的形状第一维相同
        self.assertEqual(out.shape[1], weight.shape[0])

        # 确保反向传播已重写
        # 双向反向传播分派给 _convolution_double_backward。
        # 这里不进行测试，因为涉及更多计算和重写。
        grad = torch.autograd.grad(out, input, out, create_graph=True)
        # 断言调用 maia_extension.get_test_int() 返回值为 3
        self.assertEqual(maia_extension.get_test_int(), 3)
        # 断言梯度的形状与输入的形状相同

@torch.testing._internal.common_utils.markDynamoStrictTest
class TestRNGExtension(common.TestCase):
    # 设置测试环境
    def setUp(self):
        super().setUp()

    # 如果 TorchDynamo 不支持 'maia' 设备，则跳过测试
    @skipIfTorchDynamo("https://github.com/pytorch/torchdynamo/issues/1991")
    # 定义一个测试方法，用于测试随机数生成的功能

    # 创建一个包含十个元素的张量，每个元素的值为42，数据类型为64位整数
    fourty_two = torch.full((10,), 42, dtype=torch.int64)

    # 创建一个十个元素的空张量，并用随机整数填充。断言这个张量与全42张量不相等
    t = torch.empty(10, dtype=torch.int64).random_()
    self.assertNotEqual(t, fourty_two)

    # 创建一个十个元素的空张量，并用指定的生成器生成随机整数。断言这个张量与全42张量不相等
    gen = torch.Generator(device="cpu")
    t = torch.empty(10, dtype=torch.int64).random_(generator=gen)
    self.assertNotEqual(t, fourty_two)

    # 调用一个扩展函数，断言生成器实例数量为0
    self.assertEqual(rng_extension.getInstanceCount(), 0)

    # 创建一个自定义的CPU生成器，并断言生成器实例数量为1
    gen = rng_extension.createTestCPUGenerator(42)
    self.assertEqual(rng_extension.getInstanceCount(), 1)

    # 将生成器复制给变量copy，并断言生成器实例数量仍为1
    copy = gen
    self.assertEqual(rng_extension.getInstanceCount(), 1)

    # 断言生成器gen和copy相等
    self.assertEqual(gen, copy)

    # 对copy进行身份转换，并断言生成器实例数量仍为1
    copy2 = rng_extension.identity(copy)
    self.assertEqual(rng_extension.getInstanceCount(), 1)

    # 用生成器gen生成随机整数填充张量t，并断言生成器实例数量仍为1，且张量t与全42张量相等
    t = torch.empty(10, dtype=torch.int64).random_(generator=gen)
    self.assertEqual(rng_extension.getInstanceCount(), 1)
    self.assertEqual(t, fourty_two)

    # 删除生成器gen，并断言生成器实例数量为0
    del gen
    self.assertEqual(rng_extension.getInstanceCount(), 0)

    # 删除变量copy，并断言生成器实例数量仍为0
    del copy
    self.assertEqual(rng_extension.getInstanceCount(), 0)

    # 删除变量copy2，并断言生成器实例数量为0
    del copy2
    self.assertEqual(rng_extension.getInstanceCount(), 0)
@torch.testing._internal.common_utils.markDynamoStrictTest
@unittest.skipIf(not TEST_CUDA, "CUDA not found")
class TestTorchLibrary(common.TestCase):
    # 定义测试类 TestTorchLibrary，用于测试 Torch 库功能
    def test_torch_library(self):
        # 导入 torch_test_cpp_extension.torch_library 模块，并忽略 F401 类型的 flake8 错误
        import torch_test_cpp_extension.torch_library  # noqa: F401

        # 定义函数 f，接受两个布尔类型参数 a 和 b，并调用 torch_library 模块的 logical_and 函数进行逻辑与操作
        def f(a: bool, b: bool):
            return torch.ops.torch_library.logical_and(a, b)

        # 断言调用 f 函数返回 True
        self.assertTrue(f(True, True))
        # 断言调用 f 函数返回 False
        self.assertFalse(f(True, False))
        # 断言调用 f 函数返回 False
        self.assertFalse(f(False, True))
        # 断言调用 f 函数返回 False
        self.assertFalse(f(False, False))
        
        # 使用 torch.jit.script 将函数 f 编译为 Torch 脚本 s
        s = torch.jit.script(f)
        # 断言调用 Torch 脚本 s 返回 True
        self.assertTrue(s(True, True))
        # 断言调用 Torch 脚本 s 返回 False
        self.assertFalse(s(True, False))
        # 断言调用 Torch 脚本 s 返回 False
        self.assertFalse(s(False, True))
        # 断言调用 Torch 脚本 s 返回 False
        self.assertFalse(s(False, False))
        
        # 断言字符串表示的 Torch 脚本 s 的图中包含 "torch_library::logical_and"
        self.assertIn("torch_library::logical_and", str(s.graph))


if __name__ == "__main__":
    # 运行所有的测试
    common.run_tests()
```