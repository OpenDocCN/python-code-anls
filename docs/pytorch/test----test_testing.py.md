# `.\pytorch\test\test_testing.py`

```py
# Owner(s): ["module: tests"]

# 引入所需的模块和库
import collections
import doctest
import functools
import importlib
import inspect
import itertools
import math
import os
import re
import subprocess
import sys
import unittest.mock
from typing import Any, Callable, Iterator, List, Tuple

# 引入 PyTorch 库
import torch

# 引入测试相关的功能函数和类
from torch.testing import make_tensor
from torch.testing._internal.common_utils import \
    (IS_FBCODE, IS_JETSON, IS_MACOS, IS_SANDCASTLE, IS_WINDOWS, TestCase, run_tests, slowTest,
     parametrize, subtest, instantiate_parametrized_tests, dtype_name, TEST_WITH_ROCM, decorateIf)

# 引入设备类型和相关测试基础功能
from torch.testing._internal.common_device_type import \
    (PYTORCH_TESTING_DEVICE_EXCEPT_FOR_KEY, PYTORCH_TESTING_DEVICE_ONLY_FOR_KEY, dtypes,
     get_device_type_test_bases, instantiate_device_type_tests, onlyCPU, onlyCUDA, onlyNativeDeviceTypes,
     deviceCountAtLeast, ops, expectedFailureMeta, OpDTypes)

# 引入运算符模块
import operator


# 用于测试 TestCase 方法和 torch.testing 函数的测试类
class TestTesting(TestCase):

    # 测试 assertEqual 方法是否正确处理 numpy 数组
    @dtypes(*all_types_and_complex_and(torch.bool, torch.half))
    def test_assertEqual_numpy(self, device, dtype):
        S = 10
        test_sizes = [
            (),
            (0,),
            (S,),
            (S, S),
            (0, S),
            (S, 0)]
        
        # 遍历不同的测试尺寸
        for test_size in test_sizes:
            # 创建指定设备和数据类型的张量
            a = make_tensor(test_size, dtype=dtype, device=device, low=-5, high=5)
            # 转换为 numpy 数组
            a_n = a.cpu().numpy()
            # 设置错误消息
            msg = f'size: {test_size}'
            # 使用 assertEqual 比较 numpy 数组和张量是否相等
            self.assertEqual(a_n, a, rtol=0, atol=0, msg=msg)
            self.assertEqual(a, a_n, rtol=0, atol=0, msg=msg)
            self.assertEqual(a_n, a_n, rtol=0, atol=0, msg=msg)

    # 测试 assertEqual 方法在长消息情况下的表现
    def test_assertEqual_longMessage(self):
        actual = "actual"
        expected = "expected"

        long_message = self.longMessage
        try:
            # 设置 TestCase.longMessage = False，捕获默认的错误消息
            self.longMessage = False
            try:
                self.assertEqual(actual, expected)
            except AssertionError as error:
                default_msg = str(error)
            else:
                raise AssertionError("AssertionError not raised")

            # 设置 TestCase.longMessage = True，添加额外消息并验证是否触发 AssertionError
            self.longMessage = True
            extra_msg = "sentinel"
            with self.assertRaisesRegex(AssertionError, re.escape(f"{default_msg}\n{extra_msg}")):
                self.assertEqual(actual, expected, msg=extra_msg)
        finally:
            self.longMessage = long_message
    # 辅助函数用于测试 torch.isclose 函数的多个情况
    def _isclose_helper(self, tests, device, dtype, equal_nan, atol=1e-08, rtol=1e-05):
        # 遍历每个测试用例
        for test in tests:
            # 创建两个张量 a 和 b，使用给定的设备和数据类型
            a = torch.tensor((test[0],), device=device, dtype=dtype)
            b = torch.tensor((test[1],), device=device, dtype=dtype)

            # 调用 torch.isclose 函数计算实际结果
            actual = torch.isclose(a, b, equal_nan=equal_nan, atol=atol, rtol=rtol)
            # 获取期望结果
            expected = test[2]
            # 使用断言检查实际结果是否等于期望结果
            self.assertEqual(actual.item(), expected)

    # 测试 torch.bool 类型的 torch.isclose 函数
    def test_isclose_bool(self, device):
        # 定义一组测试用例，每个元组包含 (a, b, expected)
        tests = (
            (True, True, True),
            (False, False, True),
            (True, False, False),
            (False, True, False),
        )

        # 调用辅助函数执行测试
        self._isclose_helper(tests, device, torch.bool, False)

    # 使用指定的数据类型测试 torch.isclose 函数
    @dtypes(torch.uint8,
            torch.int8, torch.int16, torch.int32, torch.int64)
    def test_isclose_integer(self, device, dtype):
        # 定义一组整数类型的测试用例
        tests = (
            (0, 0, True),
            (0, 1, False),
            (1, 0, False),
        )

        # 调用辅助函数执行测试
        self._isclose_helper(tests, device, dtype, False)

        # 针对 atol 和 rtol 进行额外的测试
        tests = [
            (0, 1, True),
            (1, 0, False),
            (1, 3, True),
        ]

        # 调用辅助函数执行测试，传入自定义的 atol 和 rtol
        self._isclose_helper(tests, device, dtype, False, atol=.5, rtol=.5)

        # 根据数据类型选择不同的测试用例
        if dtype is torch.uint8:
            tests = [
                (-1, 1, False),
                (1, -1, False)
            ]
        else:
            tests = [
                (-1, 1, True),
                (1, -1, True)
            ]

        # 调用辅助函数执行测试，传入自定义的 atol 和 rtol
        self._isclose_helper(tests, device, dtype, False, atol=1.5, rtol=.5)

    # 使用浮点数类型测试 torch.isclose 函数
    @onlyNativeDeviceTypes
    @dtypes(torch.float16, torch.float32, torch.float64)
    def test_isclose_float(self, device, dtype):
        # 定义一组浮点数类型的测试用例
        tests = (
            (0, 0, True),
            (0, -1, False),
            (float('inf'), float('inf'), True),
            (-float('inf'), float('inf'), False),
            (float('inf'), float('nan'), False),
            (float('nan'), float('nan'), False),
            (0, float('nan'), False),
            (1, 1, True),
        )

        # 调用辅助函数执行测试
        self._isclose_helper(tests, device, dtype, False)

        # 针对 atol 和 rtol 进行额外的测试
        eps = 1e-2 if dtype is torch.half else 1e-6
        tests = (
            (0, 1, True),
            (0, 1 + eps, False),
            (1, 0, False),
            (1, 3, True),
            (1 - eps, 3, False),
            (-.25, .5, True),
            (-.25 - eps, .5, False),
            (.25, -.5, True),
            (.25 + eps, -.5, False),
        )

        # 调用辅助函数执行测试，传入自定义的 atol 和 rtol
        self._isclose_helper(tests, device, dtype, False, atol=.5, rtol=.5)

        # 当 equal_nan=True 时的额外测试
        tests = (
            (0, float('nan'), False),
            (float('inf'), float('nan'), False),
            (float('nan'), float('nan'), True),
        )

        # 调用辅助函数执行测试，此时 equal_nan=True
        self._isclose_helper(tests, device, dtype, True)

    # 如果在沙堡环境中运行，则跳过此测试
    @unittest.skipIf(IS_SANDCASTLE, "Skipping because doesn't work on sandcastle")
    @dtypes(torch.complex64, torch.complex128)
    # 测试在指定复数类型上，使用 atol 或 rtol 小于零时是否会抛出异常
    #   RuntimeError
    @dtypes(torch.bool, torch.uint8,
            torch.int8, torch.int16, torch.int32, torch.int64,
            torch.float16, torch.float32, torch.float64)
    def test_isclose_atol_rtol_greater_than_zero(self, device, dtype):
        # 创建一个张量 `t`，指定设备和数据类型
        t = torch.tensor((1,), device=device, dtype=dtype)

        # 测试当 `atol` 为负数时是否会引发 RuntimeError 异常
        with self.assertRaises(RuntimeError):
            torch.isclose(t, t, atol=-1, rtol=1)
        # 测试当 `rtol` 为负数时是否会引发 RuntimeError 异常
        with self.assertRaises(RuntimeError):
            torch.isclose(t, t, atol=1, rtol=-1)
        # 测试当 `atol` 和 `rtol` 均为负数时是否会引发 RuntimeError 异常
        with self.assertRaises(RuntimeError):
            torch.isclose(t, t, atol=-1, rtol=-1)

    def test_isclose_equality_shortcut(self):
        # 对于值 >= 2**53 的整数，即使在 torch.float64 或更低精度的浮点类型中，两个相差 1 的整数也不能区分。
        # 因此，即使在 rtol == 0 和 atol == 0 的情况下，这些张量如果不作为整数比较，则被认为是相似的。
        a = torch.tensor(2 ** 53, dtype=torch.int64)
        b = a + 1

        # 验证两个整数相差 1 时，torch.isclose 是否返回 False
        self.assertFalse(torch.isclose(a, b, rtol=0, atol=0))

    @dtypes(torch.float16, torch.float32, torch.float64, torch.complex64, torch.complex128)
    def test_isclose_nan_equality_shortcut(self, device, dtype):
        # 如果数据类型是浮点类型，设置 a 和 b 为 NaN；否则设置为复数 NaN
        if dtype.is_floating_point:
            a = b = torch.nan
        else:
            a = complex(torch.nan, 0)
            b = complex(0, torch.nan)

        expected = True
        tests = [(a, b, expected)]

        # 调用辅助方法 _isclose_helper 进行 NaN 相等性测试
        self._isclose_helper(tests, device, dtype, equal_nan=True, rtol=0, atol=0)

    # The following tests (test_cuda_assert_*) are added to ensure test suite terminates early
    # when CUDA assert was thrown. Because all subsequent test will fail if that happens.
    # These tests are slow because it spawn another process to run test suite.
    # See: https://github.com/pytorch/pytorch/issues/49019
    @unittest.skipIf(TEST_WITH_ROCM, "ROCm doesn't support device side asserts")
    @onlyCUDA
    @slowTest
    def test_cuda_assert_should_stop_common_utils_test_suite(self, device):
        # 测试以确保在 CUDA 抛出断言时，common_utils.py 能够提前终止测试套件
        stderr = TestCase.runWithPytorchAPIUsageStderr("""
#!/usr/bin/env python3

import torch
from torch.testing._internal.common_utils import (TestCase, run_tests, slowTest)

class TestThatContainsCUDAAssertFailure(TestCase):
    # 继承自 TestCase 类，用于测试 CUDA 异常情况的测试用例集合

    @slowTest
    def test_throw_unrecoverable_cuda_exception(self):
        # 测试在 CUDA 设备上引发不可恢复的异常
        x = torch.rand(10, device='cuda')
        # 创建一个在 CUDA 设备上的随机张量 x

        # 引发不可恢复的 CUDA 异常，但在 CPU 上可以恢复
        y = x[torch.tensor([25])].cpu()

    @slowTest
    def test_trivial_passing_test_case_on_cpu_cuda(self):
        # 测试在 CPU 和 CUDA 上的简单通过测试案例
        x1 = torch.tensor([0., 1.], device='cuda')
        # 创建一个在 CUDA 设备上的张量 x1
        x2 = torch.tensor([0., 1.], device='cpu')
        # 创建一个在 CPU 上的张量 x2
        self.assertEqual(x1, x2)
        # 断言 x1 和 x2 相等

if __name__ == '__main__':
    run_tests()
    # 运行测试用例


``` 
    # should capture CUDA error
    self.assertIn('CUDA error: device-side assert triggered', stderr)
    # 应该捕获 CUDA 错误信息，指示设备端触发了断言

    # should run only 1 test because it throws unrecoverable error.
    self.assertIn('errors=1', stderr)
    # 因为引发了不可恢复的错误，所以应该只运行 1 个测试用例



    @unittest.skipIf(TEST_WITH_ROCM, "ROCm doesn't support device side asserts")
    @onlyCUDA
    @slowTest
    def test_cuda_assert_should_stop_common_device_type_test_suite(self, device):
        # 测试以确保 common_device_type.py 覆盖在 CUDA 设备上的早期终止
        stderr = TestCase.runWithPytorchAPIUsageStderr("""\
#!/usr/bin/env python3

import torch
from torch.testing._internal.common_utils import (TestCase, run_tests, slowTest)
from torch.testing._internal.common_device_type import instantiate_device_type_tests

class TestThatContainsCUDAAssertFailure(TestCase):

    @slowTest
    def test_throw_unrecoverable_cuda_exception(self, device):
        x = torch.rand(10, device=device)
        # 创建一个在指定设备上的随机张量 x

        # 引发不可恢复的 CUDA 异常，但在 CPU 上可以恢复
        y = x[torch.tensor([25])].cpu()

    @slowTest
    def test_trivial_passing_test_case_on_cpu_cuda(self, device):
        x1 = torch.tensor([0., 1.], device=device)
        # 创建一个在指定设备上的张量 x1
        x2 = torch.tensor([0., 1.], device='cpu')
        # 创建一个在 CPU 上的张量 x2
        self.assertEqual(x1, x2)

instantiate_device_type_tests(
    TestThatContainsCUDAAssertFailure,
    globals(),
    only_for='cuda'
)

if __name__ == '__main__':
    run_tests()
    # 运行测试用例
""")
    # 应该捕获 CUDA 错误信息，指示设备端触发了断言

    # 因为引发了不可恢复的错误，所以应该只运行 1 个测试用例
    self.assertIn('errors=1', stderr)



    @unittest.skipIf(TEST_WITH_ROCM, "ROCm doesn't support device side asserts")
    @onlyCUDA
    @slowTest
    def test_cuda_assert_should_not_stop_common_distributed_test_suite(self, device):
        # 测试以确保 common_distributed.py 覆盖不应在 CUDA 设备上提前终止
        stderr = TestCase.runWithPytorchAPIUsageStderr("""\
#!/usr/bin/env python3

import torch
from torch.testing._internal.common_utils import (run_tests, slowTest)
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_distributed import MultiProcessTestCase

class TestThatContainsCUDAAssertFailure(MultiProcessTestCase):

    @slowTest
    # 定义一个测试方法，用于测试在给定设备上抛出不可恢复的 CUDA 异常
    def test_throw_unrecoverable_cuda_exception(self, device):
        # 创建一个在指定设备上的随机张量
        x = torch.rand(10, device=device)
        # 引发不可恢复的 CUDA 异常，但在 CPU 上可以恢复
        y = x[torch.tensor([25])].cpu()

    # 带有装饰器 @slowTest 的测试方法，测试在 CPU 和 CUDA 上的简单通过测试用例
    @slowTest
    def test_trivial_passing_test_case_on_cpu_cuda(self, device):
        # 在指定设备上创建张量 x1
        x1 = torch.tensor([0., 1.], device=device)
        # 在 CPU 上创建张量 x2
        x2 = torch.tensor([0., 1.], device='cpu')
        # 断言两个张量相等
        self.assertEqual(x1, x2)
# 将包含CUDA断言失败的测试实例化为设备类型测试，并将其添加到全局命名空间中
instantiate_device_type_tests(
    TestThatContainsCUDAAssertFailure,
    globals(),
    only_for='cuda'
)

# 如果当前脚本作为主程序运行，则执行测试运行函数
if __name__ == '__main__':
    run_tests()
        test_bases_count = len(get_device_type_test_bases())
        # 获取设备类型测试基础数量
        env = dict(os.environ)
        # 复制当前环境变量到新的字典中作为测试环境
        for k in ['CI', PYTORCH_TESTING_DEVICE_ONLY_FOR_KEY, PYTORCH_TESTING_DEVICE_EXCEPT_FOR_KEY]:
            # 遍历要移除的环境变量列表
            if k in env.keys():
                # 如果环境变量存在于当前环境中，则删除它
                del env[k]
        _, stderr = TestCase.run_process_no_exception(test_filter_file_template, env=env)
        # 运行测试用例的进程，捕获标准错误输出
        self.assertIn(f'Ran {test_bases_count} test', stderr.decode('ascii'))

        # 使用只运行的设备类型设置测试环境，预期只运行一个测试
        env[PYTORCH_TESTING_DEVICE_ONLY_FOR_KEY] = 'cpu'
        _, stderr = TestCase.run_process_no_exception(test_filter_file_template, env=env)
        # 再次运行测试用例的进程，捕获标准错误输出
        self.assertIn('Ran 1 test', stderr.decode('ascii'))

        # 使用排除的设备类型设置测试环境，预期运行默认设备类型数量减一的测试
        del env[PYTORCH_TESTING_DEVICE_ONLY_FOR_KEY]
        env[PYTORCH_TESTING_DEVICE_EXCEPT_FOR_KEY] = 'cpu'
        _, stderr = TestCase.run_process_no_exception(test_filter_file_template, env=env)
        # 再次运行测试用例的进程，捕获标准错误输出
        self.assertIn(f'Ran {test_bases_count-1} test', stderr.decode('ascii'))

        # 同时设置只运行和排除的设备类型，预期会抛出异常
        env[PYTORCH_TESTING_DEVICE_ONLY_FOR_KEY] = 'cpu'
        _, stderr = TestCase.run_process_no_exception(test_filter_file_template, env=env)
        # 再次运行测试用例的进程，捕获标准错误输出
        self.assertNotIn('OK', stderr.decode('ascii'))
    .. note::

        每个不测试特定输入的测试都应该迭代此循环，以最大化覆盖率。

    Args:
        actual (Any): 实际输入。
        expected (Any): 期望输入。

    Yields:
        Callable: 返回一个部分应用了预定义位置输入的 :func:`torch.testing.assert_close` 函数。
    """
    # 对于每个由 make_assert_close_inputs 生成的输入参数组合
    for inputs in make_assert_close_inputs(actual, expected):
        # 返回一个部分应用了 torch.testing.assert_close 函数的可调用对象
        yield functools.partial(torch.testing.assert_close, *inputs)
# 定义一个名为 TestAssertClose 的测试类，继承自 TestCase 类
class TestAssertClose(TestCase):

    # 测试不同类型的 actual 和 expected，期望类型不匹配的情况下的处理
    def test_mismatching_types_subclasses(self):
        # 生成一个随机张量 actual
        actual = torch.rand(())
        # 将 actual 转换为 torch.nn.Parameter 对象作为 expected
        expected = torch.nn.Parameter(actual)

        # 遍历 assert_close_with_inputs 函数返回的可调用对象列表 fn
        for fn in assert_close_with_inputs(actual, expected):
            # 调用 fn()

    # 测试不同类型的 actual 和 expected，类型不匹配但类型相等的情况下的处理
    def test_mismatching_types_type_equality(self):
        # 生成一个空张量 actual
        actual = torch.empty(())
        # 将 actual 转换为 torch.nn.Parameter 对象作为 expected
        expected = torch.nn.Parameter(actual)

        # 遍历 assert_close_with_inputs 函数返回的可调用对象列表 fn
        for fn in assert_close_with_inputs(actual, expected):
            # 使用 self.assertRaisesRegex 检查是否抛出 TypeError 异常，异常信息包含 expected 的类型字符串
            with self.assertRaisesRegex(TypeError, str(type(expected))):
                # 调用 fn(allow_subclasses=False)

    # 测试不同类型的 actual 和 expected，类型不匹配的情况下的处理
    def test_mismatching_types(self):
        # 生成一个包含两个元素的空张量 actual
        actual = torch.empty(2)
        # 将 actual 转换为其对应的 numpy 数组作为 expected
        expected = actual.numpy()

        # 使用 itertools.product 生成 assert_close_with_inputs 返回的可调用对象列表 fn 和 allow_subclasses 参数的组合
        for fn, allow_subclasses in itertools.product(assert_close_with_inputs(actual, expected), (True, False)):
            # 使用 self.assertRaisesRegex 检查是否抛出 TypeError 异常，异常信息包含 expected 的类型字符串
            with self.assertRaisesRegex(TypeError, str(type(expected))):
                # 调用 fn(allow_subclasses=allow_subclasses)

    # 测试 actual 和 expected 为相同字符串时的处理
    def test_unknown_type(self):
        actual = "0"
        expected = "0"

        # 遍历 assert_close_with_inputs 函数返回的可调用对象列表 fn
        for fn in assert_close_with_inputs(actual, expected):
            # 使用 self.assertRaisesRegex 检查是否抛出 TypeError 异常，异常信息包含 actual 的类型字符串
            with self.assertRaisesRegex(TypeError, str(type(actual))):
                # 调用 fn()

    # 测试 actual 和 expected 的形状不匹配时的处理
    def test_mismatching_shape(self):
        # 生成一个空张量 actual
        actual = torch.empty(())
        # 复制 actual 并将其形状改变为 (1,)，作为 expected
        expected = actual.clone().reshape((1,))

        # 遍历 assert_close_with_inputs 函数返回的可调用对象列表 fn
        for fn in assert_close_with_inputs(actual, expected):
            # 使用 self.assertRaisesRegex 检查是否抛出 AssertionError 异常，异常信息包含 "shape" 字符串
            with self.assertRaisesRegex(AssertionError, "shape"):
                # 调用 fn()

    # 如果 MKLDNN 可用，则测试 actual 和 expected 使用不同布局时的处理
    @unittest.skipIf(not torch.backends.mkldnn.is_available(), reason="MKLDNN is not available.")
    def test_unknown_layout(self):
        # 生成一个包含元素的空张量 actual
        actual = torch.empty((2, 2))
        # 将 actual 转换为 MKLDNN 布局的 expected
        expected = actual.to_mkldnn()

        # 遍历 assert_close_with_inputs 函数返回的可调用对象列表 fn
        for fn in assert_close_with_inputs(actual, expected):
            # 使用 self.assertRaisesRegex 检查是否抛出 ValueError 异常，异常信息包含 "layout" 字符串
            with self.assertRaisesRegex(ValueError, "layout"):
                # 调用 fn()

    # 测试在 meta 设备上的 actual 和 expected 的处理
    def test_meta(self):
        # 生成一个在 meta 设备上的空张量 actual 和 expected
        actual = torch.empty((2, 2), device="meta")
        expected = torch.empty((2, 2), device="meta")

        # 遍历 assert_close_with_inputs 函数返回的可调用对象列表 fn
        for fn in assert_close_with_inputs(actual, expected):
            # 调用 fn()

    # 测试 actual 和 expected 使用不同布局时的处理
    def test_mismatching_layout(self):
        # 生成一个包含元素的空张量 strided
        strided = torch.empty((2, 2))
        # 将 strided 转换为稀疏 COO 布局的 sparse_coo
        sparse_coo = strided.to_sparse()
        # 将 strided 转换为稀疏 CSR 布局的 sparse_csr
        sparse_csr = strided.to_sparse_csr()

        # 遍历 itertools.combinations 生成的 sparse_coo 和 sparse_csr 的所有组合
        for actual, expected in itertools.combinations((strided, sparse_coo, sparse_csr), 2):
            # 遍历 assert_close_with_inputs 函数返回的可调用对象列表 fn
            for fn in assert_close_with_inputs(actual, expected):
                # 使用 self.assertRaisesRegex 检查是否抛出 AssertionError 异常，异常信息包含 "layout" 字符串
                with self.assertRaisesRegex(AssertionError, "layout"):
                    # 调用 fn()

    # 测试 actual 和 expected 使用不同布局时，不检查布局的处理
    def test_mismatching_layout_no_check(self):
        # 生成一个包含随机元素的空张量 strided
        strided = torch.randn((2, 2))
        # 将 strided 转换为稀疏 COO 布局的 sparse_coo
        sparse_coo = strided.to_sparse()
        # 将 strided 转换为稀疏 CSR 布局的 sparse_csr
        sparse_csr = strided.to_sparse_csr()

        # 遍历 itertools.combinations 生成的 sparse_coo 和 sparse_csr 的所有组合
        for actual, expected in itertools.combinations((strided, sparse_coo, sparse_csr), 2):
            # 遍历 assert_close_with_inputs 函数返回的可调用对象列表 fn
            for fn in assert_close_with_inputs(actual, expected):
                # 调用 fn(check_layout=False)
    # 测试函数：检查当实际值和期望值的数据类型不匹配时的情况
    def test_mismatching_dtype(self):
        # 创建一个空的张量作为实际值，数据类型为浮点型
        actual = torch.empty((), dtype=torch.float)
        # 克隆实际值并将数据类型转换为整型，作为期望值
        expected = actual.clone().to(torch.int)

        # 对于每个断言函数，使用实际值和期望值进行比较
        for fn in assert_close_with_inputs(actual, expected):
            # 检查是否抛出断言错误，并且错误消息中包含 "dtype"
            with self.assertRaisesRegex(AssertionError, "dtype"):
                fn()

    # 测试函数：检查当实际值和期望值的数据类型不匹配时（无类型检查）的情况
    def test_mismatching_dtype_no_check(self):
        # 创建一个张量填充为 1.0，数据类型为浮点型，作为实际值
        actual = torch.ones((), dtype=torch.float)
        # 克隆实际值并将数据类型转换为整型，作为期望值
        expected = actual.clone().to(torch.int)

        # 对于每个断言函数，使用实际值和期望值进行比较（无类型检查）
        for fn in assert_close_with_inputs(actual, expected):
            fn(check_dtype=False)

    # 测试函数：检查当实际值和期望值的步幅不匹配时的情况
    def test_mismatching_stride(self):
        # 创建一个空的 2x2 张量作为实际值
        actual = torch.empty((2, 2))
        # 克隆实际值并转置、重排步幅后的张量作为期望值
        expected = torch.as_strided(actual.clone().t().contiguous(), actual.shape, actual.stride()[::-1])

        # 对于每个断言函数，使用实际值和期望值进行比较
        for fn in assert_close_with_inputs(actual, expected):
            # 检查是否抛出断言错误，并且错误消息中包含 "stride"
            with self.assertRaisesRegex(AssertionError, "stride"):
                fn(check_stride=True)

    # 测试函数：检查当实际值和期望值的步幅不匹配时（无步幅检查）的情况
    def test_mismatching_stride_no_check(self):
        # 创建一个随机填充的 2x2 张量作为实际值
        actual = torch.rand((2, 2))
        # 克隆实际值并转置、重排步幅后的张量作为期望值
        expected = torch.as_strided(actual.clone().t().contiguous(), actual.shape, actual.stride()[::-1])

        # 对于每个断言函数，使用实际值和期望值进行比较（无步幅检查）
        for fn in assert_close_with_inputs(actual, expected):
            fn()

    # 测试函数：检查仅使用 rtol 时的情况
    def test_only_rtol(self):
        # 创建一个空的标量张量作为实际值和期望值
        actual = torch.empty(())
        expected = actual.clone()

        # 对于每个断言函数，使用实际值和期望值进行比较
        for fn in assert_close_with_inputs(actual, expected):
            # 检查是否抛出值错误，并且不允许设置 rtol 为 0.0
            with self.assertRaises(ValueError):
                fn(rtol=0.0)

    # 测试函数：检查仅使用 atol 时的情况
    def test_only_atol(self):
        # 创建一个空的标量张量作为实际值和期望值
        actual = torch.empty(())
        expected = actual.clone()

        # 对于每个断言函数，使用实际值和期望值进行比较
        for fn in assert_close_with_inputs(actual, expected):
            # 检查是否抛出值错误，并且不允许设置 atol 为 0.0
            with self.assertRaises(ValueError):
                fn(atol=0.0)

    # 测试函数：检查当实际值和期望值不匹配时的情况
    def test_mismatching_values(self):
        # 创建一个标量张量，实际值为 1
        actual = torch.tensor(1)
        # 创建一个标量张量，期望值为 2
        expected = torch.tensor(2)

        # 对于每个断言函数，使用实际值和期望值进行比较
        for fn in assert_close_with_inputs(actual, expected):
            # 检查是否抛出断言错误
            with self.assertRaises(AssertionError):
                fn()

    # 测试函数：检查当实际值和期望值不匹配时，使用 rtol 和 atol 的情况
    def test_mismatching_values_rtol(self):
        eps = 1e-3
        # 创建一个标量浮点型张量，实际值为 1.0
        actual = torch.tensor(1.0)
        # 创建一个比实际值略大的浮点型张量，作为期望值
        expected = torch.tensor(1.0 + eps)

        # 对于每个断言函数，使用实际值和期望值进行比较
        for fn in assert_close_with_inputs(actual, expected):
            # 检查是否抛出断言错误，并且 rtol 设置为 eps / 2，atol 设置为 0.0
            with self.assertRaises(AssertionError):
                fn(rtol=eps / 2, atol=0.0)

    # 测试函数：检查当实际值和期望值不匹配时，使用 rtol 和 atol 的情况
    def test_mismatching_values_atol(self):
        eps = 1e-3
        # 创建一个标量浮点型张量，实际值为 0.0
        actual = torch.tensor(0.0)
        # 创建一个比实际值略大的浮点型张量，作为期望值
        expected = torch.tensor(eps)

        # 对于每个断言函数，使用实际值和期望值进行比较
        for fn in assert_close_with_inputs(actual, expected):
            # 检查是否抛出断言错误，并且 rtol 设置为 0.0，atol 设置为 eps / 2
            with self.assertRaises(AssertionError):
                fn(rtol=0.0, atol=eps / 2)

    # 测试函数：检查当实际值和期望值匹配时的情况
    def test_matching(self):
        # 创建一个标量浮点型张量，实际值和期望值相同
        actual = torch.tensor(1.0)
        expected = actual.clone()

        # 使用测试框架提供的 assert_close 函数，检查实际值和期望值是否接近
        torch.testing.assert_close(actual, expected)

    # 测试函数：检查当实际值和期望值使用 rtol 时的情况
    def test_matching_rtol(self):
        eps = 1e-3
        # 创建一个标量浮点型张量，实际值为 1.0
        actual = torch.tensor(1.0)
        # 创建一个比实际值略大的浮点型张量，作为期望值
        expected = torch.tensor(1.0 + eps)

        # 对于每个断言函数，使用实际值和期望值进行比较
        for fn in assert_close_with_inputs(actual, expected):
            # 检查是否能够成功使用 rtol 进行比较，atol 设置为 0.0
            fn(rtol=eps * 2, atol=0.0)
    # 定义一个测试方法，用于测试在给定的绝对容差范围内是否匹配
    def test_matching_atol(self):
        # 设置绝对容差为 0.001
        eps = 1e-3
        # 创建一个实际值张量，值为 0.0
        actual = torch.tensor(0.0)
        # 创建一个期望值张量，值为 0.001
        expected = torch.tensor(eps)

        # 对于每个返回的函数，调用它们来进行断言检查
        for fn in assert_close_with_inputs(actual, expected):
            # 设置相对容差为 0.0，绝对容差为 0.002
            fn(rtol=0.0, atol=eps * 2)

    # TODO: 此测试设计用于检查 https://github.com/pytorch/pytorch/pull/56058 中删除的代码
    #  我们需要检查此测试是否仍然需要，或者此行为是否已默认启用。
    def test_matching_conjugate_bit(self):
        # 创建一个复数张量，对其求共轭
        actual = torch.tensor(complex(1, 1)).conj()
        # 创建一个期望的复数张量，值为 (1, -1)
        expected = torch.tensor(complex(1, -1))

        # 对于每个返回的函数，调用它们来进行断言检查
        for fn in assert_close_with_inputs(actual, expected):
            fn()

    # 测试 NaN 值的匹配情况
    def test_matching_nan(self):
        nan = float("NaN")

        # 不同的 NaN 测试对
        tests = (
            (nan, nan),
            (complex(nan, 0), complex(0, nan)),
            (complex(nan, nan), complex(nan, 0)),
            (complex(nan, nan), complex(nan, nan)),
        )

        # 对于每个测试对，对每个返回的函数，调用它们来进行断言检查
        for actual, expected in tests:
            for fn in assert_close_with_inputs(actual, expected):
                # 确保会抛出 AssertionError
                with self.assertRaises(AssertionError):
                    fn()

    # 测试 NaN 值的匹配情况，允许 NaN 与 NaN 相等
    def test_matching_nan_with_equal_nan(self):
        nan = float("NaN")

        # 不同的 NaN 测试对
        tests = (
            (nan, nan),
            (complex(nan, 0), complex(0, nan)),
            (complex(nan, nan), complex(nan, 0)),
            (complex(nan, nan), complex(nan, nan)),
        )

        # 对于每个测试对，对每个返回的函数，调用它们来进行断言检查，允许 NaN 与 NaN 相等
        for actual, expected in tests:
            for fn in assert_close_with_inputs(actual, expected):
                fn(equal_nan=True)

    # 测试将张量转换为 NumPy 数组后是否匹配
    def test_numpy(self):
        # 创建一个随机张量
        tensor = torch.rand(2, 2, dtype=torch.float32)
        # 将张量转换为 NumPy 数组
        actual = tensor.numpy()
        # 复制 NumPy 数组作为期望值
        expected = actual.copy()

        # 对于每个返回的函数，调用它们来进行断言检查
        for fn in assert_close_with_inputs(actual, expected):
            fn()

    # 测试标量值的匹配情况
    def test_scalar(self):
        # 生成一个随机整数
        number = torch.randint(10, size=()).item()
        # 遍历所有可能的实际和期望值对
        for actual, expected in itertools.product((int(number), float(number), complex(number)), repeat=2):
            # 检查实际值和期望值的数据类型是否相同
            check_dtype = type(actual) is type(expected)

            # 对于每个返回的函数，调用它们来进行断言检查，根据需要检查数据类型
            for fn in assert_close_with_inputs(actual, expected):
                fn(check_dtype=check_dtype)

    # 测试布尔类型张量的匹配情况
    def test_bool(self):
        # 创建一个包含 True 和 False 的张量
        actual = torch.tensor([True, False])
        # 复制该张量作为期望值
        expected = actual.clone()

        # 对于每个返回的函数，调用它们来进行断言检查
        for fn in assert_close_with_inputs(actual, expected):
            fn()

    # 测试 None 的匹配情况
    def test_none(self):
        # 设置实际值和期望值均为 None
        actual = expected = None

        # 对于每个返回的函数，调用它们来进行断言检查
        for fn in assert_close_with_inputs(actual, expected):
            fn()

    # 测试当实际值为 None 时的不匹配情况
    def test_none_mismatch(self):
        # 设置期望值为 None
        expected = None

        # 遍历不同的实际值情况
        for actual in (False, 0, torch.nan, torch.tensor(torch.nan)):
            # 对于每个返回的函数，调用它们来进行断言检查，预期会抛出 AssertionError
            for fn in assert_close_with_inputs(actual, expected):
                with self.assertRaises(AssertionError):
                    fn()
    # 定义一个测试方法，用于测试文档字符串示例
    def test_docstring_examples(self):
        # 创建一个 DocTestFinder 对象来查找文档测试，关闭详细输出
        finder = doctest.DocTestFinder(verbose=False)
        # 创建一个 DocTestRunner 对象来运行文档测试，设置关闭详细输出和标准化空白字符
        runner = doctest.DocTestRunner(verbose=False, optionflags=doctest.NORMALIZE_WHITESPACE)
        # 准备全局变量字典，将 torch 模块添加进去
        globs = dict(torch=torch)
        # 使用 DocTestFinder 查找 assert_close 函数的文档测试，返回的是一个列表，选取第一个元素
        doctests = finder.find(torch.testing.assert_close, globs=globs)[0]
        # 初始化一个空列表，用于存储失败的测试报告
        failures = []
        # 运行找到的文档测试，并将失败的报告添加到 failures 列表中
        runner.run(doctests, out=lambda report: failures.append(report))
        # 如果有失败的文档测试，则抛出 AssertionError，包含失败的报告信息
        if failures:
            raise AssertionError(f"Doctest found {len(failures)} failures:\n\n" + "\n".join(failures))

    # 定义一个测试方法，用于测试默认容差选择不匹配数据类型的情况
    def test_default_tolerance_selection_mismatching_dtypes(self):
        # 创建一个实际值张量，使用 torch.bfloat16 类型
        actual = torch.tensor(0.99, dtype=torch.bfloat16)
        # 创建一个期望值张量，使用 torch.float64 类型
        expected = torch.tensor(1.0, dtype=torch.float64)

        # 调用 assert_close_with_inputs 函数，获取返回的函数列表，并遍历执行每个函数，关闭类型检查
        for fn in assert_close_with_inputs(actual, expected):
            fn(check_dtype=False)

    # 定义一个自定义异常类 UnexpectedException，用于测试 assert_close 对异常的处理
    class UnexpectedException(Exception):
        """The only purpose of this exception is to test ``assert_close``'s handling of unexpected exceptions. Thus,
    the test should mock a component to raise this instead of the regular behavior. We avoid using a builtin
    exception here to avoid triggering possible handling of them.
    """
        pass

    # 使用 unittest.mock.patch 装饰器，模拟 TensorLikePair.__init__ 方法抛出 UnexpectedException 异常
    @unittest.mock.patch("torch.testing._comparison.TensorLikePair.__init__", side_effect=UnexpectedException)
    # 定义一个测试方法，测试当原始异常发生时的情况
    def test_unexpected_error_originate(self, _):
        # 创建一个实际值张量
        actual = torch.tensor(1.0)
        # 创建一个期望值张量，克隆自实际值张量
        expected = actual.clone()

        # 使用断言检查在运行 torch.testing.assert_close(actual, expected) 时是否抛出 RuntimeError 异常，并包含特定错误信息
        with self.assertRaisesRegex(RuntimeError, "unexpected exception"):
            torch.testing.assert_close(actual, expected)

    # 使用 unittest.mock.patch 装饰器，模拟 TensorLikePair.compare 方法抛出 UnexpectedException 异常
    @unittest.mock.patch("torch.testing._comparison.TensorLikePair.compare", side_effect=UnexpectedException)
    # 定义一个测试方法，测试当比较异常发生时的情况
    def test_unexpected_error_compare(self, _):
        # 创建一个实际值张量
        actual = torch.tensor(1.0)
        # 创建一个期望值张量，克隆自实际值张量
        expected = actual.clone()

        # 使用断言检查在运行 torch.testing.assert_close(actual, expected) 时是否抛出 RuntimeError 异常，并包含特定错误信息
        with self.assertRaisesRegex(RuntimeError, "unexpected exception"):
            torch.testing.assert_close(actual, expected)
class TestAssertCloseMultiDevice(TestCase):
    # 对于具有多设备的测试用例，测试不匹配设备的情况
    @deviceCountAtLeast(1)
    def test_mismatching_device(self, devices):
        # 针对所有设备的排列组合进行遍历
        for actual_device, expected_device in itertools.permutations(("cpu", *devices), 2):
            # 创建一个在指定设备上的空张量
            actual = torch.empty((), device=actual_device)
            # 克隆并移动张量到期望设备上
            expected = actual.clone().to(expected_device)
            # 对于每一个断言函数，执行断言并检查是否引发异常
            for fn in assert_close_with_inputs(actual, expected):
                with self.assertRaisesRegex(AssertionError, "device"):
                    fn()

    # 对于具有多设备的测试用例，测试不匹配设备但不检查的情况
    @deviceCountAtLeast(1)
    def test_mismatching_device_no_check(self, devices):
        # 针对所有设备的排列组合进行遍历
        for actual_device, expected_device in itertools.permutations(("cpu", *devices), 2):
            # 创建一个在指定设备上的随机张量
            actual = torch.rand((), device=actual_device)
            # 克隆并移动张量到期望设备上
            expected = actual.clone().to(expected_device)
            # 对于每一个断言函数，执行断言但不检查设备匹配
            for fn in assert_close_with_inputs(actual, expected):
                fn(check_device=False)

# 根据给定的类和全局变量实例化设备类型的测试用例，仅针对 CUDA
instantiate_device_type_tests(TestAssertCloseMultiDevice, globals(), only_for="cuda")


class TestAssertCloseErrorMessage(TestCase):
    # 测试张量和张量类对象不匹配的情况
    def test_identifier_tensor_likes(self):
        actual = torch.tensor([1, 2, 3, 4])
        expected = torch.tensor([1, 2, 5, 6])

        for fn in assert_close_with_inputs(actual, expected):
            # 断言应该引发包含“Tensor-likes”文本的 AssertionError 异常
            with self.assertRaisesRegex(AssertionError, re.escape("Tensor-likes")):
                fn()

    # 测试标量不匹配的情况
    def test_identifier_scalars(self):
        actual = 3
        expected = 5
        for fn in assert_close_with_inputs(actual, expected):
            # 断言应该引发包含“Scalars”文本的 AssertionError 异常
            with self.assertRaisesRegex(AssertionError, re.escape("Scalars")):
                fn()

    # 测试张量数值不相等的情况
    def test_not_equal(self):
        actual = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
        expected = torch.tensor([1, 2, 5, 6], dtype=torch.float32)

        for fn in assert_close_with_inputs(actual, expected):
            # 断言应该引发包含“not equal”文本的 AssertionError 异常，设置容差为0
            with self.assertRaisesRegex(AssertionError, re.escape("not equal")):
                fn(rtol=0.0, atol=0.0)

    # 测试张量数值不接近的情况
    def test_not_close(self):
        actual = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
        expected = torch.tensor([1, 2, 5, 6], dtype=torch.float32)

        for fn, (rtol, atol) in itertools.product(
            assert_close_with_inputs(actual, expected), ((1.3e-6, 0.0), (0.0, 1e-5), (1.3e-6, 1e-5))
        ):
            # 断言应该引发包含“not close”文本的 AssertionError 异常，根据不同的容差设置
            with self.assertRaisesRegex(AssertionError, re.escape("not close")):
                fn(rtol=rtol, atol=atol)

    # 测试张量的元素不匹配的情况
    def test_mismatched_elements(self):
        actual = torch.tensor([1, 2, 3, 4])
        expected = torch.tensor([1, 2, 5, 6])

        for fn in assert_close_with_inputs(actual, expected):
            # 断言应该引发包含特定文本的 AssertionError 异常，显示不匹配的元素比例
            with self.assertRaisesRegex(AssertionError, re.escape("Mismatched elements: 2 / 4 (50.0%)")):
                fn()
    # 测试绝对差异函数
    def test_abs_diff(self):
        # 创建实际和期望的张量
        actual = torch.tensor([[1, 2], [3, 4]])
        expected = torch.tensor([[1, 2], [5, 4]])

        # 对于每个返回的函数，使用assert_close_with_inputs进行断言
        for fn in assert_close_with_inputs(actual, expected):
            # 断言捕获AssertionError异常，并检查异常消息中是否包含预期的错误信息
            with self.assertRaisesRegex(AssertionError, re.escape("Greatest absolute difference: 2 at index (1, 0)")):
                fn()

    # 测试绝对差异标量函数
    def test_abs_diff_scalar(self):
        # 创建实际和期望的标量
        actual = 3
        expected = 5

        # 对于每个返回的函数，使用assert_close_with_inputs进行断言
        for fn in assert_close_with_inputs(actual, expected):
            # 断言捕获AssertionError异常，并检查异常消息中是否包含预期的错误信息
            with self.assertRaisesRegex(AssertionError, re.escape("Absolute difference: 2")):
                fn()

    # 测试相对差异函数
    def test_rel_diff(self):
        # 创建实际和期望的张量
        actual = torch.tensor([[1, 2], [3, 4]])
        expected = torch.tensor([[1, 4], [3, 4]])

        # 对于每个返回的函数，使用assert_close_with_inputs进行断言
        for fn in assert_close_with_inputs(actual, expected):
            # 断言捕获AssertionError异常，并检查异常消息中是否包含预期的错误信息
            with self.assertRaisesRegex(AssertionError, re.escape("Greatest relative difference: 0.5 at index (0, 1)")):
                fn()

    # 测试相对差异标量函数
    def test_rel_diff_scalar(self):
        # 创建实际和期望的标量
        actual = 2
        expected = 4

        # 对于每个返回的函数，使用assert_close_with_inputs进行断言
        for fn in assert_close_with_inputs(actual, expected):
            # 断言捕获AssertionError异常，并检查异常消息中是否包含预期的错误信息
            with self.assertRaisesRegex(AssertionError, re.escape("Relative difference: 0.5")):
                fn()

    # 测试零除以零的情况
    def test_zero_div_zero(self):
        # 创建实际和期望的张量
        actual = torch.tensor([1.0, 0.0])
        expected = torch.tensor([2.0, 0.0])

        # 对于每个返回的函数，使用assert_close_with_inputs进行断言
        for fn in assert_close_with_inputs(actual, expected):
            # 使用正则表达式确保错误消息中不包含'nan'，这种情况会发生在计算不匹配时
            with self.assertRaisesRegex(AssertionError, "((?!nan).)*"):
                fn()

    # 测试相对误差tolerance
    def test_rtol(self):
        rtol = 1e-3

        # 创建实际和期望的张量
        actual = torch.tensor((1, 2))
        expected = torch.tensor((2, 2))

        # 对于每个返回的函数，使用assert_close_with_inputs进行断言
        for fn in assert_close_with_inputs(actual, expected):
            # 断言捕获AssertionError异常，并检查异常消息中是否包含预期的错误信息，包括rtol值
            with self.assertRaisesRegex(AssertionError, re.escape(f"(up to {rtol} allowed)")):
                fn(rtol=rtol, atol=0.0)

    # 测试绝对误差tolerance
    def test_atol(self):
        atol = 1e-3

        # 创建实际和期望的张量
        actual = torch.tensor((1, 2))
        expected = torch.tensor((2, 2))

        # 对于每个返回的函数，使用assert_close_with_inputs进行断言
        for fn in assert_close_with_inputs(actual, expected):
            # 断言捕获AssertionError异常，并检查异常消息中是否包含预期的错误信息，包括atol值
            with self.assertRaisesRegex(AssertionError, re.escape(f"(up to {atol} allowed)")):
                fn(rtol=0.0, atol=atol)

    # 测试自定义错误消息字符串
    def test_msg_str(self):
        msg = "Custom error message!"

        # 创建实际和期望的标量
        actual = torch.tensor(1)
        expected = torch.tensor(2)

        # 对于每个返回的函数，使用assert_close_with_inputs进行断言
        for fn in assert_close_with_inputs(actual, expected):
            # 断言捕获AssertionError异常，并检查异常消息中是否包含预期的自定义错误消息
            with self.assertRaisesRegex(AssertionError, msg):
                fn(msg=msg)

    # 测试可调用的自定义错误消息
    def test_msg_callable(self):
        msg = "Custom error message"

        # 创建实际和期望的标量
        actual = torch.tensor(1)
        expected = torch.tensor(2)

        # 对于每个返回的函数，使用assert_close_with_inputs进行断言
        for fn in assert_close_with_inputs(actual, expected):
            # 断言捕获AssertionError异常，并检查异常消息中是否包含预期的自定义错误消息
            with self.assertRaisesRegex(AssertionError, msg):
                fn(msg=lambda _: msg)
# 定义一个测试类 TestAssertCloseContainer，继承自 TestCase 类
class TestAssertCloseContainer(TestCase):

    # 定义测试方法 test_sequence_mismatching_len，用于测试序列长度不匹配的情况
    def test_sequence_mismatching_len(self):
        # 设置实际值为一个空的元组
        actual = (torch.empty(()),)
        # 设置期望值为一个空的元组
        expected = ()

        # 使用断言检查是否会抛出 AssertionError 异常
        with self.assertRaises(AssertionError):
            # 调用 assert_close 函数，比较 actual 和 expected 的接近程度
            torch.testing.assert_close(actual, expected)

    # 定义测试方法 test_sequence_mismatching_values_msg，用于测试序列中值不匹配的情况
    def test_sequence_mismatching_values_msg(self):
        # 创建两个张量 t1 和 t2
        t1 = torch.tensor(1)
        t2 = torch.tensor(2)

        # 设置实际值为包含两个 t1 的元组
        actual = (t1, t1)
        # 设置期望值为包含 t1 和 t2 的元组
        expected = (t1, t2)

        # 使用断言检查是否会抛出带有指定消息的 AssertionError 异常
        with self.assertRaisesRegex(AssertionError, re.escape("item [1]")):
            # 调用 assert_close 函数，比较 actual 和 expected 的接近程度，并检查异常消息
            torch.testing.assert_close(actual, expected)

    # 定义测试方法 test_mapping_mismatching_keys，用于测试映射键不匹配的情况
    def test_mapping_mismatching_keys(self):
        # 设置实际值为包含一个键为 'a' 的空张量的字典
        actual = {"a": torch.empty(())}
        # 设置期望值为空字典
        expected = {}

        # 使用断言检查是否会抛出 AssertionError 异常
        with self.assertRaises(AssertionError):
            # 调用 assert_close 函数，比较 actual 和 expected 的接近程度
            torch.testing.assert_close(actual, expected)

    # 定义测试方法 test_mapping_mismatching_values_msg，用于测试映射值不匹配的情况
    def test_mapping_mismatching_values_msg(self):
        # 创建两个张量 t1 和 t2
        t1 = torch.tensor(1)
        t2 = torch.tensor(2)

        # 设置实际值为包含两个键 'a' 和 'b'，值分别为 t1 的字典
        actual = {"a": t1, "b": t1}
        # 设置期望值为包含键 'a' 值为 t1 和键 'b' 值为 t2 的字典
        expected = {"a": t1, "b": t2}

        # 使用断言检查是否会抛出带有指定消息的 AssertionError 异常
        with self.assertRaisesRegex(AssertionError, re.escape("item ['b']")):
            # 调用 assert_close 函数，比较 actual 和 expected 的接近程度，并检查异常消息
            torch.testing.assert_close(actual, expected)


# 定义一个测试类 TestAssertCloseSparseCOO，继承自 TestCase 类
class TestAssertCloseSparseCOO(TestCase):

    # 定义测试方法 test_matching_coalesced，用于测试匹配的压缩稀疏张量
    def test_matching_coalesced(self):
        # 设置稀疏张量的索引和值
        indices = (
            (0, 1),
            (1, 0),
        )
        values = (1, 2)
        # 创建实际值为压缩的稀疏 COO 张量
        actual = torch.sparse_coo_tensor(indices, values, size=(2, 2)).coalesce()
        # 创建期望值为实际值的克隆
        expected = actual.clone()

        # 遍历 assert_close_with_inputs 函数返回的函数列表，依次执行
        for fn in assert_close_with_inputs(actual, expected):
            fn()

    # 定义测试方法 test_matching_uncoalesced，用于测试匹配的非压缩稀疏张量
    def test_matching_uncoalesced(self):
        # 设置稀疏张量的索引和值
        indices = (
            (0, 1),
            (1, 0),
        )
        values = (1, 2)
        # 创建实际值为非压缩的稀疏 COO 张量
        actual = torch.sparse_coo_tensor(indices, values, size=(2, 2))
        # 创建期望值为实际值的克隆
        expected = actual.clone()

        # 遍历 assert_close_with_inputs 函数返回的函数列表，依次执行
        for fn in assert_close_with_inputs(actual, expected):
            fn()

    # 定义测试方法 test_mismatching_sparse_dims，用于测试稀疏 COO 张量的稀疏维度不匹配情况
    def test_mismatching_sparse_dims(self):
        # 创建一个形状为 (2, 3, 4) 的随机张量 t
        t = torch.randn(2, 3, 4)
        # 将 t 转换为压缩的稀疏 COO 张量作为实际值
        actual = t.to_sparse()
        # 将 t 在维度 2 上转换为压缩的稀疏 COO 张量作为期望值
        expected = t.to_sparse(2)

        # 遍历 assert_close_with_inputs 函数返回的函数列表，依次执行
        for fn in assert_close_with_inputs(actual, expected):
            # 使用断言检查是否会抛出带有指定消息的 AssertionError 异常
            with self.assertRaisesRegex(AssertionError, re.escape("number of sparse dimensions in sparse COO tensors")):
                fn()

    # 定义测试方法 test_mismatching_nnz，用于测试稀疏 COO 张量的非零元素数不匹配情况
    def test_mismatching_nnz(self):
        # 设置实际张量的索引和值
        actual_indices = (
            (0, 1),
            (1, 0),
        )
        actual_values = (1, 2)
        # 创建实际值为稀疏 COO 张量
        actual = torch.sparse_coo_tensor(actual_indices, actual_values, size=(2, 2))

        # 设置期望张量的索引和值
        expected_indices = (
            (0, 1, 1,),
            (1, 0, 0,),
        )
        expected_values = (1, 1, 1)
        # 创建期望值为稀疏 COO 张量
        expected = torch.sparse_coo_tensor(expected_indices, expected_values, size=(2, 2))

        # 遍历 assert_close_with_inputs 函数返回的函数列表，依次执行
        for fn in assert_close_with_inputs(actual, expected):
            # 使用断言检查是否会抛出带有指定消息的 AssertionError 异常
            with self.assertRaisesRegex(AssertionError, re.escape("number of specified values in sparse COO tensors")):
                fn()
    # 测试稀疏张量的索引不匹配时的错误消息
    def test_mismatching_indices_msg(self):
        # 实际的稀疏张量索引
        actual_indices = (
            (0, 1),
            (1, 0),
        )
        # 实际的稀疏张量数值
        actual_values = (1, 2)
        # 创建实际的稀疏 COO 张量
        actual = torch.sparse_coo_tensor(actual_indices, actual_values, size=(2, 2))
    
        # 期望的稀疏张量索引
        expected_indices = (
            (0, 1),
            (1, 1),
        )
        # 期望的稀疏张量数值
        expected_values = (1, 2)
        # 创建期望的稀疏 COO 张量
        expected = torch.sparse_coo_tensor(expected_indices, expected_values, size=(2, 2))
    
        # 对每一个断言函数应用实际和期望的稀疏张量，检查是否触发断言错误，并验证错误消息中是否包含"Sparse COO indices"
        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(AssertionError, re.escape("Sparse COO indices")):
                fn()
    
    # 测试稀疏张量的值不匹配时的错误消息
    def test_mismatching_values_msg(self):
        # 实际的稀疏张量索引
        actual_indices = (
            (0, 1),
            (1, 0),
        )
        # 实际的稀疏张量数值
        actual_values = (1, 2)
        # 创建实际的稀疏 COO 张量
        actual = torch.sparse_coo_tensor(actual_indices, actual_values, size=(2, 2))
    
        # 期望的稀疏张量索引
        expected_indices = (
            (0, 1),
            (1, 0),
        )
        # 期望的稀疏张量数值（这里故意将第二个值改为3，与实际值不匹配）
        expected_values = (1, 3)
        # 创建期望的稀疏 COO 张量
        expected = torch.sparse_coo_tensor(expected_indices, expected_values, size=(2, 2))
    
        # 对每一个断言函数应用实际和期望的稀疏张量，检查是否触发断言错误，并验证错误消息中是否包含"Sparse COO values"
        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(AssertionError, re.escape("Sparse COO values")):
                fn()
# 使用 unittest 模块中的装饰器 `skipIf`，根据条件跳过测试，条件为 `IS_FBCODE` 或 `IS_SANDCASTLE` 为真时跳过，字符串参数为跳过时的说明
@unittest.skipIf(IS_FBCODE or IS_SANDCASTLE, "Not all sandcastle jobs support CSR testing")
# 定义一个测试类 TestAssertCloseSparseCSR，继承自 TestCase 类
class TestAssertCloseSparseCSR(TestCase):
    # 定义测试方法 test_matching
    def test_matching(self):
        # 定义稀疏张量的行索引
        crow_indices = (0, 1, 2)
        # 定义稀疏张量的列索引
        col_indices = (1, 0)
        # 定义稀疏张量的值
        values = (1, 2)
        # 创建实际的稀疏 CSR 张量
        actual = torch.sparse_csr_tensor(crow_indices, col_indices, values, size=(2, 2))
        # 创建预期的稀疏 CSR 张量，复制实际的稀疏 CSR 张量
        expected = actual.clone()

        # 对于 assert_close_with_inputs 返回的每个函数 fn
        for fn in assert_close_with_inputs(actual, expected):
            # 调用 fn 函数
            fn()

    # 定义测试方法 test_mismatching_crow_indices_msg
    def test_mismatching_crow_indices_msg(self):
        # 定义实际稀疏 CSR 张量的行索引
        actual_crow_indices = (0, 1, 2)
        # 定义实际稀疏 CSR 张量的列索引
        actual_col_indices = (0, 1)
        # 定义实际稀疏 CSR 张量的值
        actual_values = (1, 2)
        # 创建实际的稀疏 CSR 张量
        actual = torch.sparse_csr_tensor(actual_crow_indices, actual_col_indices, actual_values, size=(2, 2))

        # 定义预期稀疏 CSR 张量的行索引
        expected_crow_indices = (0, 2, 2)
        # 复制实际稀疏 CSR 张量的列索引作为预期的稀疏 CSR 张量的列索引
        expected_col_indices = actual_col_indices
        # 复制实际稀疏 CSR 张量的值作为预期的稀疏 CSR 张量的值
        expected_values = actual_values
        # 创建预期的稀疏 CSR 张量
        expected = torch.sparse_csr_tensor(expected_crow_indices, expected_col_indices, expected_values, size=(2, 2))

        # 对于 assert_close_with_inputs 返回的每个函数 fn
        for fn in assert_close_with_inputs(actual, expected):
            # 使用 self.assertRaisesRegex 断言抛出 AssertionError 异常，并验证异常消息中是否包含指定文本
            with self.assertRaisesRegex(AssertionError, re.escape("Sparse CSR crow_indices")):
                fn()

    # 定义测试方法 test_mismatching_col_indices_msg
    def test_mismatching_col_indices_msg(self):
        # 定义实际稀疏 CSR 张量的行索引
        actual_crow_indices = (0, 1, 2)
        # 定义实际稀疏 CSR 张量的列索引
        actual_col_indices = (1, 0)
        # 定义实际稀疏 CSR 张量的值
        actual_values = (1, 2)
        # 创建实际的稀疏 CSR 张量
        actual = torch.sparse_csr_tensor(actual_crow_indices, actual_col_indices, actual_values, size=(2, 2))

        # 复制实际稀疏 CSR 张量的行索引作为预期的稀疏 CSR 张量的行索引
        expected_crow_indices = actual_crow_indices
        # 定义预期稀疏 CSR 张量的列索引
        expected_col_indices = (1, 1)
        # 复制实际稀疏 CSR 张量的值作为预期的稀疏 CSR 张量的值
        expected_values = actual_values
        # 创建预期的稀疏 CSR 张量
        expected = torch.sparse_csr_tensor(expected_crow_indices, expected_col_indices, expected_values, size=(2, 2))

        # 对于 assert_close_with_inputs 返回的每个函数 fn
        for fn in assert_close_with_inputs(actual, expected):
            # 使用 self.assertRaisesRegex 断言抛出 AssertionError 异常，并验证异常消息中是否包含指定文本
            with self.assertRaisesRegex(AssertionError, re.escape("Sparse CSR col_indices")):
                fn()

    # 定义测试方法 test_mismatching_values_msg
    def test_mismatching_values_msg(self):
        # 定义实际稀疏 CSR 张量的行索引
        actual_crow_indices = (0, 1, 2)
        # 定义实际稀疏 CSR 张量的列索引
        actual_col_indices = (1, 0)
        # 定义实际稀疏 CSR 张量的值
        actual_values = (1, 2)
        # 创建实际的稀疏 CSR 张量
        actual = torch.sparse_csr_tensor(actual_crow_indices, actual_col_indices, actual_values, size=(2, 2))

        # 复制实际稀疏 CSR 张量的行索引作为预期的稀疏 CSR 张量的行索引
        expected_crow_indices = actual_crow_indices
        # 复制实际稀疏 CSR 张量的列索引作为预期的稀疏 CSR 张量的列索引
        expected_col_indices = actual_col_indices
        # 定义预期稀疏 CSR 张量的值
        expected_values = (1, 3)
        # 创建预期的稀疏 CSR 张量
        expected = torch.sparse_csr_tensor(expected_crow_indices, expected_col_indices, expected_values, size=(2, 2))

        # 对于 assert_close_with_inputs 返回的每个函数 fn
        for fn in assert_close_with_inputs(actual, expected):
            # 使用 self.assertRaisesRegex 断言抛出 AssertionError 异常，并验证异常消息中是否包含指定文本
            with self.assertRaisesRegex(AssertionError, re.escape("Sparse CSR values")):
                fn()


# 使用 unittest 模块中的装饰器 `skipIf`，根据条件跳过测试，条件为 `IS_FBCODE` 或 `IS_SANDCASTLE` 为真时跳过，字符串参数为跳过时的说明
@unittest.skipIf(IS_FBCODE or IS_SANDCASTLE, "Not all sandcastle jobs support CSC testing")
# 定义一个测试类 TestAssertCloseSparseCSC，继承自 TestCase 类
class TestAssertCloseSparseCSC(TestCase):
    # 定义测试方法 test_matching
    def test_matching(self):
        # 定义稀疏张量的列索引
        ccol_indices = (0, 1, 2)
        # 定义稀疏张量的行索引
        row_indices = (1, 0)
        # 定义稀疏张量的值
        values = (1, 2)
        # 创建实际的稀疏 CSC 张量
        actual = torch.sparse_csc_tensor(ccol_indices, row_indices, values, size=(2, 2))
        # 创建预期的稀疏 CSC 张量，复制实际的稀疏 CSC 张量
        expected = actual.clone()

        # 对于 assert_close_with_inputs 返回的每个函数 fn
        for fn in assert_close_with_inputs(actual, expected):
            # 调用 fn 函数
            fn()
    # 测试稀疏张量的列压缩索引不匹配时的错误消息
    def test_mismatching_ccol_indices_msg(self):
        # 设置实际的列压缩索引、行索引和数值
        actual_ccol_indices = (0, 1, 2)
        actual_row_indices = (0, 1)
        actual_values = (1, 2)
        # 创建实际的稀疏列压缩张量
        actual = torch.sparse_csc_tensor(actual_ccol_indices, actual_row_indices, actual_values, size=(2, 2))

        # 设置预期的列压缩索引、行索引和数值
        expected_ccol_indices = (0, 2, 2)
        expected_row_indices = actual_row_indices
        expected_values = actual_values
        # 创建预期的稀疏列压缩张量
        expected = torch.sparse_csc_tensor(expected_ccol_indices, expected_row_indices, expected_values, size=(2, 2))

        # 对于每个比较函数，使用断言来验证错误消息中是否包含预期的信息
        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(AssertionError, re.escape("Sparse CSC ccol_indices")):
                fn()

    # 测试稀疏张量的行索引不匹配时的错误消息
    def test_mismatching_row_indices_msg(self):
        # 设置实际的列压缩索引、行索引和数值
        actual_ccol_indices = (0, 1, 2)
        actual_row_indices = (1, 0)
        actual_values = (1, 2)
        # 创建实际的稀疏列压缩张量
        actual = torch.sparse_csc_tensor(actual_ccol_indices, actual_row_indices, actual_values, size=(2, 2))

        # 设置预期的列压缩索引、行索引和数值
        expected_ccol_indices = actual_ccol_indices
        expected_row_indices = (1, 1)
        expected_values = actual_values
        # 创建预期的稀疏列压缩张量
        expected = torch.sparse_csc_tensor(expected_ccol_indices, expected_row_indices, expected_values, size=(2, 2))

        # 对于每个比较函数，使用断言来验证错误消息中是否包含预期的信息
        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(AssertionError, re.escape("Sparse CSC row_indices")):
                fn()

    # 测试稀疏张量的值不匹配时的错误消息
    def test_mismatching_values_msg(self):
        # 设置实际的列压缩索引、行索引和数值
        actual_ccol_indices = (0, 1, 2)
        actual_row_indices = (1, 0)
        actual_values = (1, 2)
        # 创建实际的稀疏列压缩张量
        actual = torch.sparse_csc_tensor(actual_ccol_indices, actual_row_indices, actual_values, size=(2, 2))

        # 设置预期的列压缩索引、行索引和数值
        expected_ccol_indices = actual_ccol_indices
        expected_row_indices = actual_row_indices
        expected_values = (1, 3)
        # 创建预期的稀疏列压缩张量
        expected = torch.sparse_csc_tensor(expected_ccol_indices, expected_row_indices, expected_values, size=(2, 2))

        # 对于每个比较函数，使用断言来验证错误消息中是否包含预期的信息
        for fn in assert_close_with_inputs(actual, expected):
            with self.assertRaisesRegex(AssertionError, re.escape("Sparse CSC values")):
                fn()
# 如果在 FBCODE 或 SANDCASTLE 环境下，则跳过这些测试，因为不是所有 sandcastle 作业都支持 BSR 测试
@unittest.skipIf(IS_FBCODE or IS_SANDCASTLE, "Not all sandcastle jobs support BSR testing")
class TestAssertCloseSparseBSR(TestCase):
    
    # 测试匹配情况
    def test_matching(self):
        # 定义稀疏 BSR 张量的行索引、列索引和值
        crow_indices = (0, 1, 2)
        col_indices = (1, 0)
        values = ([[1]], [[2]])
        # 创建实际的稀疏 BSR 张量
        actual = torch.sparse_bsr_tensor(crow_indices, col_indices, values, size=(2, 2))
        # 创建预期的稀疏 BSR 张量，是实际的克隆
        expected = actual.clone()

        # 对每个 assert_close_with_inputs 返回的函数进行迭代
        for fn in assert_close_with_inputs(actual, expected):
            # 执行每个函数
            fn()

    # 测试行索引不匹配的消息
    def test_mismatching_crow_indices_msg(self):
        # 定义实际的稀疏 BSR 张量的行索引、列索引和值
        actual_crow_indices = (0, 1, 2)
        actual_col_indices = (0, 1)
        actual_values = ([[1]], [[2]])
        # 创建实际的稀疏 BSR 张量
        actual = torch.sparse_bsr_tensor(actual_crow_indices, actual_col_indices, actual_values, size=(2, 2))

        # 定义预期的稀疏 BSR 张量的行索引、列索引和值
        expected_crow_indices = (0, 2, 2)
        expected_col_indices = actual_col_indices
        expected_values = actual_values
        # 创建预期的稀疏 BSR 张量
        expected = torch.sparse_bsr_tensor(expected_crow_indices, expected_col_indices, expected_values, size=(2, 2))

        # 对每个 assert_close_with_inputs 返回的函数进行迭代
        for fn in assert_close_with_inputs(actual, expected):
            # 断言函数调用时会抛出 AssertionError，并匹配特定消息
            with self.assertRaisesRegex(AssertionError, re.escape("Sparse BSR crow_indices")):
                fn()

    # 测试列索引不匹配的消息
    def test_mismatching_col_indices_msg(self):
        # 定义实际的稀疏 BSR 张量的行索引、列索引和值
        actual_crow_indices = (0, 1, 2)
        actual_col_indices = (1, 0)
        actual_values = ([[1]], [[2]])
        # 创建实际的稀疏 BSR 张量
        actual = torch.sparse_bsr_tensor(actual_crow_indices, actual_col_indices, actual_values, size=(2, 2))

        # 定义预期的稀疏 BSR 张量的行索引、列索引和值
        expected_crow_indices = actual_crow_indices
        expected_col_indices = (1, 1)
        expected_values = actual_values
        # 创建预期的稀疏 BSR 张量
        expected = torch.sparse_bsr_tensor(expected_crow_indices, expected_col_indices, expected_values, size=(2, 2))

        # 对每个 assert_close_with_inputs 返回的函数进行迭代
        for fn in assert_close_with_inputs(actual, expected):
            # 断言函数调用时会抛出 AssertionError，并匹配特定消息
            with self.assertRaisesRegex(AssertionError, re.escape("Sparse BSR col_indices")):
                fn()

    # 测试值不匹配的消息
    def test_mismatching_values_msg(self):
        # 定义实际的稀疏 BSR 张量的行索引、列索引和值
        actual_crow_indices = (0, 1, 2)
        actual_col_indices = (1, 0)
        actual_values = ([[1]], [[2]])
        # 创建实际的稀疏 BSR 张量
        actual = torch.sparse_bsr_tensor(actual_crow_indices, actual_col_indices, actual_values, size=(2, 2))

        # 定义预期的稀疏 BSR 张量的行索引、列索引和值
        expected_crow_indices = actual_crow_indices
        expected_col_indices = actual_col_indices
        expected_values = ([[1]], [[3]])
        # 创建预期的稀疏 BSR 张量
        expected = torch.sparse_bsr_tensor(expected_crow_indices, expected_col_indices, expected_values, size=(2, 2))

        # 对每个 assert_close_with_inputs 返回的函数进行迭代
        for fn in assert_close_with_inputs(actual, expected):
            # 断言函数调用时会抛出 AssertionError，并匹配特定消息
            with self.assertRaisesRegex(AssertionError, re.escape("Sparse BSR values")):
                fn()
    #`
    # 定义一个测试方法，用于验证匹配情况
    def test_matching(self):
        # 定义列压缩索引
        ccol_indices = (0, 1, 2)
        # 定义行索引
        row_indices = (1, 0)
        # 定义值
        values = ([[1]], [[2]])
        # 创建稀疏列压缩张量的实际输出
        actual = torch.sparse_bsc_tensor(ccol_indices, row_indices, values, size=(2, 2))
        # 复制实际输出以获取预期输出
        expected = actual.clone()

        # 对每一个断言测试实例，执行测试函数
        for fn in assert_close_with_inputs(actual, expected):
            fn()

    # 定义一个测试方法，用于验证列压缩索引不匹配的情况
    def test_mismatching_ccol_indices_msg(self):
        # 实际列压缩索引
        actual_ccol_indices = (0, 1, 2)
        # 实际行索引
        actual_row_indices = (0, 1)
        # 实际值
        actual_values = ([[1]], [[2]])
        # 创建稀疏列压缩张量的实际输出
        actual = torch.sparse_bsc_tensor(actual_ccol_indices, actual_row_indices, actual_values, size=(2, 2))

        # 预期列压缩索引
        expected_ccol_indices = (0, 2, 2)
        # 预期行索引与实际相同
        expected_row_indices = actual_row_indices
        # 预期值与实际相同
        expected_values = actual_values
        # 创建稀疏列压缩张量的预期输出
        expected = torch.sparse_bsc_tensor(expected_ccol_indices, expected_row_indices, expected_values, size=(2, 2))

        # 对每一个断言测试实例，执行测试函数
        for fn in assert_close_with_inputs(actual, expected):
            # 断言错误消息中包含特定信息
            with self.assertRaisesRegex(AssertionError, re.escape("Sparse BSC ccol_indices")):
                fn()

    # 定义一个测试方法，用于验证行索引不匹配的情况
    def test_mismatching_row_indices_msg(self):
        # 实际列压缩索引
        actual_ccol_indices = (0, 1, 2)
        # 实际行索引
        actual_row_indices = (1, 0)
        # 实际值
        actual_values = ([[1]], [[2]])
        # 创建稀疏列压缩张量的实际输出
        actual = torch.sparse_bsc_tensor(actual_ccol_indices, actual_row_indices, actual_values, size=(2, 2))

        # 实际列压缩索引与预期相同
        expected_ccol_indices = actual_ccol_indices
        # 预期行索引不匹配实际
        expected_row_indices = (1, 1)
        # 预期值与实际相同
        expected_values = actual_values
        # 创建稀疏列压缩张量的预期输出
        expected = torch.sparse_bsc_tensor(expected_ccol_indices, expected_row_indices, expected_values, size=(2, 2))

        # 对每一个断言测试实例，执行测试函数
        for fn in assert_close_with_inputs(actual, expected):
            # 断言错误消息中包含特定信息
            with self.assertRaisesRegex(AssertionError, re.escape("Sparse BSC row_indices")):
                fn()

    # 定义一个测试方法，用于验证值不匹配的情况
    def test_mismatching_values_msg(self):
        # 实际列压缩索引
        actual_ccol_indices = (0, 1, 2)
        # 实际行索引
        actual_row_indices = (1, 0)
        # 实际值
        actual_values = ([[1]], [[2]])
        # 创建稀疏列压缩张量的实际输出
        actual = torch.sparse_bsc_tensor(actual_ccol_indices, actual_row_indices, actual_values, size=(2, 2))

        # 实际列压缩索引与预期相同
        expected_ccol_indices = actual_ccol_indices
        # 实际行索引与预期相同
        expected_row_indices = actual_row_indices
        # 预期值不匹配实际
        expected_values = ([[1]], [[3]])
        # 创建稀疏列压缩张量的预期输出
        expected = torch.sparse_bsc_tensor(expected_ccol_indices, expected_row_indices, expected_values, size=(2, 2))

        # 对每一个断言测试实例，执行测试函数
        for fn in assert_close_with_inputs(actual, expected):
            # 断言错误消息中包含特定信息
            with self.assertRaisesRegex(AssertionError, re.escape("Sparse BSC values")):
                fn()
    # 定义一个测试类 TestAssertCloseQuantized，继承自 TestCase，用于测试量化操作的断言功能
    class TestAssertCloseQuantized(TestCase):

        # 测试不匹配的量化是否正常工作
        def test_mismatching_is_quantized(self):
            # 创建一个实际的张量 actual，未量化
            actual = torch.tensor(1.0)
            # 创建一个期望的量化张量 expected
            expected = torch.quantize_per_tensor(actual, scale=1.0, zero_point=0, dtype=torch.qint32)

            # 对每个 assert_close_with_inputs 函数返回的函数 fn 进行测试
            for fn in assert_close_with_inputs(actual, expected):
                # 断言是否抛出了 AssertionError，并检查错误信息是否包含 "is_quantized"
                with self.assertRaisesRegex(AssertionError, "is_quantized"):
                    fn()

        # 测试不匹配的量化方案（qscheme）是否正常工作
        def test_mismatching_qscheme(self):
            # 创建一个实际的张量 actual，按照 per_tensor 方式量化
            t = torch.tensor((1.0,))
            actual = torch.quantize_per_tensor(t, scale=1.0, zero_point=0, dtype=torch.qint32)
            # 创建一个期望的张量 expected，按照 per_channel 方式量化
            expected = torch.quantize_per_channel(
                t,
                scales=torch.tensor((1.0,)),
                zero_points=torch.tensor((0,)),
                axis=0,
                dtype=torch.qint32,
            )

            # 对每个 assert_close_with_inputs 函数返回的函数 fn 进行测试
            for fn in assert_close_with_inputs(actual, expected):
                # 断言是否抛出了 AssertionError，并检查错误信息是否包含 "qscheme"
                with self.assertRaisesRegex(AssertionError, "qscheme"):
                    fn()

        # 测试匹配的 per_tensor 量化是否正常工作
        def test_matching_per_tensor(self):
            # 创建一个实际的张量 actual，按照 per_tensor 方式量化
            actual = torch.quantize_per_tensor(torch.tensor(1.0), scale=1.0, zero_point=0, dtype=torch.qint32)
            # 克隆实际张量作为期望的张量 expected
            expected = actual.clone()

            # 对每个 assert_close_with_inputs 函数返回的函数 fn 进行测试
            for fn in assert_close_with_inputs(actual, expected):
                # 执行函数 fn，不应该抛出异常
                fn()

        # 测试匹配的 per_channel 量化是否正常工作
        def test_matching_per_channel(self):
            # 创建一个实际的张量 actual，按照 per_channel 方式量化
            actual = torch.quantize_per_channel(
                torch.tensor((1.0,)),
                scales=torch.tensor((1.0,)),
                zero_points=torch.tensor((0,)),
                axis=0,
                dtype=torch.qint32,
            )
            # 克隆实际张量作为期望的张量 expected
            expected = actual.clone()

            # 对每个 assert_close_with_inputs 函数返回的函数 fn 进行测试
            for fn in assert_close_with_inputs(actual, expected):
                # 执行函数 fn，不应该抛出异常
                fn()


    # 定义一个测试类 TestMakeTensor，继承自 TestCase，用于测试 make_tensor 函数的功能
    class TestMakeTensor(TestCase):
        # 支持的数据类型列表
        supported_dtypes = dtypes(
            torch.bool,
            torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64,
            torch.float16, torch.bfloat16, torch.float32, torch.float64,
            torch.complex32, torch.complex64, torch.complex128,
        )

        # 参数化测试方法 test_smoke，用于测试 make_tensor 函数是否正常工作
        @supported_dtypes
        @parametrize("shape", [tuple(), (0,), (1,), (1, 1), (2,), (2, 3), (8, 16, 32)])
        @parametrize("splat_shape", [False, True])
        def test_smoke(self, dtype, device, shape, splat_shape):
            # 调用 make_tensor 函数创建张量 t
            t = torch.testing.make_tensor(*shape if splat_shape else shape, dtype=dtype, device=device)

            # 断言 t 是 torch.Tensor 类型
            self.assertIsInstance(t, torch.Tensor)
            # 断言 t 的形状与 shape 相同
            self.assertEqual(t.shape, shape)
            # 断言 t 的数据类型与 dtype 相同
            self.assertEqual(t.dtype, dtype)
            # 断言 t 的设备与 device 相同
            self.assertEqual(t.device, torch.device(device))

        # 参数化测试方法，测试是否支持 requires_grad 参数
        @supported_dtypes
        @parametrize("requires_grad", [False, True])
    def test_requires_grad(self, dtype, device, requires_grad):
        # 使用functools.partial创建make_tensor函数，固定dtype、device和requires_grad参数
        make_tensor = functools.partial(
            torch.testing.make_tensor,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )

        # 如果不需要梯度或者dtype是浮点数或复数类型
        if not requires_grad or dtype.is_floating_point or dtype.is_complex:
            # 创建一个张量t，调用make_tensor()生成，验证其requires_grad属性是否符合预期
            t = make_tensor()
            self.assertEqual(t.requires_grad, requires_grad)
        else:
            # 否则，应该抛出ValueError异常，指出布尔和整数类型不支持requires_grad=True
            with self.assertRaisesRegex(
                    ValueError, "`requires_grad=True` is not supported for boolean and integral dtypes"
            ):
                make_tensor()

    @supported_dtypes
    @parametrize("noncontiguous", [False, True])
    @parametrize("shape", [tuple(), (0,), (1,), (1, 1), (2,), (2, 3), (8, 16, 32)])
    def test_noncontiguous(self, dtype, device, noncontiguous, shape):
        # 计算形状shape的元素个数
        numel = functools.reduce(operator.mul, shape, 1)

        # 创建一个张量t，调用torch.testing.make_tensor()生成，验证其是否连续
        t = torch.testing.make_tensor(shape, dtype=dtype, device=device, noncontiguous=noncontiguous)
        self.assertEqual(t.is_contiguous(), not noncontiguous or numel < 2)

    @supported_dtypes
    @parametrize(
        "memory_format_and_shape",
        [
            (None, (2, 3, 4)),
            (torch.contiguous_format, (2, 3, 4)),
            (torch.channels_last, (2, 3, 4, 5)),
            (torch.channels_last_3d, (2, 3, 4, 5, 6)),
            (torch.preserve_format, (2, 3, 4)),
        ],
    )
    def test_memory_format(self, dtype, device, memory_format_and_shape):
        memory_format, shape = memory_format_and_shape

        # 创建一个张量t，调用torch.testing.make_tensor()生成，验证其是否符合指定的内存格式
        t = torch.testing.make_tensor(shape, dtype=dtype, device=device, memory_format=memory_format)

        # 断言张量t在指定内存格式下是否是连续的
        self.assertTrue(
            t.is_contiguous(memory_format=torch.contiguous_format if memory_format is None else memory_format)
        )

    @supported_dtypes
    def test_noncontiguous_memory_format(self, dtype, device):
        # 应该抛出ValueError异常，说明非连续和指定内存格式是互斥的
        with self.assertRaisesRegex(ValueError, "`noncontiguous` and `memory_format` are mutually exclusive"):
            torch.testing.make_tensor(
                (2, 3, 4, 5),
                dtype=dtype,
                device=device,
                noncontiguous=True,
                memory_format=torch.channels_last,
            )

    @supported_dtypes
    def test_exclude_zero(self, dtype, device):
        # 创建一个张量t，调用torch.testing.make_tensor()生成，验证其所有元素是否不为零
        t = torch.testing.make_tensor(10_000, dtype=dtype, device=device, exclude_zero=True, low=-1, high=2)

        self.assertTrue((t != 0).all())

    @supported_dtypes
    def test_low_high_smoke(self, dtype, device):
        # 定义low_inclusive和high_exclusive的取值范围
        low_inclusive, high_exclusive = 0, 2

        # 创建一个张量t，调用torch.testing.make_tensor()生成，验证其元素是否在指定范围内
        t = torch.testing.make_tensor(10_000, dtype=dtype, device=device, low=low_inclusive, high=high_exclusive)
        if dtype.is_complex:
            t = torch.view_as_real(t)

        self.assertTrue(((t >= low_inclusive) & (t < high_exclusive)).all())

    @supported_dtypes
    # 定义一个测试方法，用于测试低和高值默认情况下的边界情况
    def test_low_high_default_smoke(self, dtype, device):
        # 根据数据类型选择低值和高值的范围
        low_inclusive, high_exclusive = {
            torch.bool: (0, 2),
            torch.uint8: (0, 10),
            **dict.fromkeys([torch.int8, torch.int16, torch.int32, torch.int64], (-9, 10)),
        }.get(dtype, (-9, 9))

        # 创建一个指定数据类型和设备的张量，低值为 low_inclusive，高值为 high_exclusive
        t = torch.testing.make_tensor(10_000, dtype=dtype, device=device, low=low_inclusive, high=high_exclusive)
        
        # 如果数据类型是复数类型，则将张量视图转换为实数视图
        if dtype.is_complex:
            t = torch.view_as_real(t)

        # 断言：所有张量元素都应在 low_inclusive 和 high_exclusive 范围内
        self.assertTrue(((t >= low_inclusive) & (t < high_exclusive)).all())

    # 参数化测试方法，测试低值大于等于高值的情况
    @parametrize("low_high", [(0, 0), (1, 0), (0, -1)])
    @parametrize("value_types", list(itertools.product([int, float], repeat=2)))
    @supported_dtypes
    def test_low_ge_high(self, dtype, device, low_high, value_types):
        # 根据 value_types 指定的类型，分别获取 low 和 high 的值
        low, high = (value_type(value) for value, value_type in zip(low_high, value_types))

        # 如果 low 等于 high，并且数据类型是浮点数或复数类型，则发出未来警告
        if low == high and (dtype.is_floating_point or dtype.is_complex):
            with self.assertWarnsRegex(
                    FutureWarning,
                    "Passing `low==high` to `torch.testing.make_tensor` for floating or complex types is deprecated",
            ):
                # 创建张量，设置 low 和 high，注意：dtype 是复数类型时设置复数值
                t = torch.testing.make_tensor(10_000, dtype=dtype, device=device, low=low, high=high)
            # 断言：生成的张量应该与期望值相等
            self.assertEqual(t, torch.full_like(t, complex(low, low) if dtype.is_complex else low))
        else:
            # 如果 low 不等于 high，则应该抛出值错误异常
            with self.assertRaisesRegex(ValueError, "`low` must be less than `high`"):
                torch.testing.make_tensor(dtype=dtype, device=device, low=low, high=high)

    # 参数化测试方法，测试 low 和 high 包含 NaN 的情况
    @supported_dtypes
    @parametrize("low_high", [(None, torch.nan), (torch.nan, None), (torch.nan, torch.nan)])
    def test_low_high_nan(self, dtype, device, low_high):
        # 获取 low 和 high 的值，这些值可能包含 NaN
        low, high = low_high

        # 断言：应该抛出值错误异常，因为 low 和 high 不能是 NaN
        with self.assertRaisesRegex(ValueError, "`low` and `high` cannot be NaN"):
            torch.testing.make_tensor(dtype=dtype, device=device, low=low, high=high)
    def test_low_high_outside_valid_range(self, dtype, device):
        # 部分应用 make_tensor 函数，使用给定的 dtype 和 device 参数
        make_tensor = functools.partial(torch.testing.make_tensor, dtype=dtype, device=device)

        # 定义函数 get_dtype_limits，根据 dtype 返回其数值类型的最小和最大值
        def get_dtype_limits(dtype):
            # 如果 dtype 是 torch.bool 类型，返回 0 和 1
            if dtype is torch.bool:
                return 0, 1

            # 否则根据 dtype 类型调用 torch.finfo 或 torch.iinfo 函数获取信息
            info = (torch.finfo if dtype.is_floating_point or dtype.is_complex else torch.iinfo)(dtype)
            # 返回数值类型的最小和最大值
            return int(info.min), int(info.max)

        # 获取当前 dtype 类型的最小值和最大值
        lowest_inclusive, highest_inclusive = get_dtype_limits(dtype)

        # 使用 assertRaisesRegex 检查下面的代码块是否会抛出 ValueError 异常
        with self.assertRaisesRegex(ValueError, ""):
            # 根据 lowest_inclusive 的值，设置 low 和 high 变量
            low, high = (-2, -1) if lowest_inclusive == 0 else (lowest_inclusive * 4, lowest_inclusive * 2)
            # 调用 make_tensor 函数，传入 low 和 high 参数
            make_tensor(low=low, high=high)

        # 使用 assertRaisesRegex 检查下面的代码块是否会抛出 ValueError 异常
        with self.assertRaisesRegex(ValueError, ""):
            # 设置 low 和 high 变量为 highest_inclusive 的两倍和四倍
            make_tensor(low=highest_inclusive * 2, high=highest_inclusive * 4)

    @dtypes(torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    def test_low_high_boolean_integral1(self, dtype, device):
        # 定义 tensor 的形状为 (10000,)
        shape = (10_000,)
        # 定义一个极小值 eps
        eps = 1e-4

        # 调用 make_tensor 函数，创建一个张量 actual
        actual = torch.testing.make_tensor(shape, dtype=dtype, device=device, low=-(1 - eps), high=1 - eps)
        # 创建一个预期的张量 expected，全部元素为 0
        expected = torch.zeros(shape, dtype=dtype, device=device)

        # 使用 assert_close 检查 actual 和 expected 是否在误差允许范围内接近
        torch.testing.assert_close(actual, expected)

    @dtypes(torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
    def test_low_high_boolean_integral2(self, dtype, device):
        # 定义 tensor 的形状为 (10000,)
        shape = (10_000,)
        # 如果 dtype 是 torch.bool 类型，设置 low 为 1
        if dtype is torch.bool:
            low = 1
        # 如果 dtype 是 torch.int64 类型，由于内部原因，设置 low 为 torch.iinfo(dtype).max - 1
        elif dtype is torch.int64:
            low = torch.iinfo(dtype).max - 1
        else:
            # 对于其他整数类型，设置 low 为 torch.iinfo(dtype).max
            low = torch.iinfo(dtype).max
        # 设置 high 为 low 的下一个值
        high = low + 1

        # 调用 make_tensor 函数，创建一个张量 actual
        actual = torch.testing.make_tensor(shape, dtype=dtype, device=device, low=low, high=high)
        # 创建一个预期的张量 expected，全部元素为 low
        expected = torch.full(shape, low, dtype=dtype, device=device)

        # 使用 assert_close 检查 actual 和 expected 是否在误差允许范围内接近
        torch.testing.assert_close(actual, expected)
# 调用函数实例化设备类型测试，使用 TestMakeTensor 类和全局变量
instantiate_device_type_tests(TestMakeTensor, globals())


def _get_test_names_for_test_class(test_cls):
    """ Convenience function to get all test names for a given test class. """
    # 获取给定测试类中所有以 'test_' 开头的测试名称
    test_names = [f'{test_cls.__name__}.{key}' for key in test_cls.__dict__
                  if key.startswith('test_')]
    return sorted(test_names)


def _get_test_funcs_for_test_class(test_cls):
    """ Convenience function to get all (test function, parametrized_name) pairs for a given test class. """
    # 获取给定测试类中所有以 'test_' 开头的测试函数及其函数名
    test_funcs = [(getattr(test_cls, key), key) for key in test_cls.__dict__ if key.startswith('test_')]
    return test_funcs


class TestTestParametrization(TestCase):
    # 定义测试类 TestTestParametrization，继承自 TestCase
    def test_default_names(self):

        class TestParametrized(TestCase):
            @parametrize("x", range(5))
            def test_default_names(self, x):
                pass

            @parametrize("x,y", [(1, 2), (2, 3), (3, 4)])
            def test_two_things_default_names(self, x, y):
                pass

        # 实例化参数化测试类 TestParametrized
        instantiate_parametrized_tests(TestParametrized)

        # 预期的测试名称列表，包含所有可能的测试函数参数组合
        expected_test_names = [
            'TestParametrized.test_default_names_x_0',
            'TestParametrized.test_default_names_x_1',
            'TestParametrized.test_default_names_x_2',
            'TestParametrized.test_default_names_x_3',
            'TestParametrized.test_default_names_x_4',
            'TestParametrized.test_two_things_default_names_x_1_y_2',
            'TestParametrized.test_two_things_default_names_x_2_y_3',
            'TestParametrized.test_two_things_default_names_x_3_y_4',
        ]
        # 获取实际的测试名称列表
        test_names = _get_test_names_for_test_class(TestParametrized)
        # 断言预期的测试名称列表与实际获取的测试名称列表相同
        self.assertEqual(expected_test_names, test_names)
    def test_name_fn(self):
        # 定义一个内部的测试类 TestParametrized，继承自 TestCase
        class TestParametrized(TestCase):
            # 参数化测试函数 test_custom_names，参数 bias 取值为 False 和 True
            @parametrize("bias", [False, True], name_fn=lambda b: 'bias' if b else 'no_bias')
            def test_custom_names(self, bias):
                pass

            # 参数化测试函数 test_three_things_composition_custom_names，分别参数化 x, y, z
            @parametrize("x", [1, 2], name_fn=str)
            @parametrize("y", [3, 4], name_fn=str)
            @parametrize("z", [5, 6], name_fn=str)
            def test_three_things_composition_custom_names(self, x, y, z):
                pass

            # 参数化测试函数 test_two_things_custom_names_alternate，参数 x, y 组合
            @parametrize("x,y", [(1, 2), (1, 3), (1, 4)], name_fn=lambda x, y: f'{x}__{y}')
            def test_two_things_custom_names_alternate(self, x, y):
                pass

        # 实例化参数化测试类 TestParametrized
        instantiate_parametrized_tests(TestParametrized)

        # 预期的测试名称列表
        expected_test_names = [
            'TestParametrized.test_custom_names_bias',
            'TestParametrized.test_custom_names_no_bias',
            'TestParametrized.test_three_things_composition_custom_names_1_3_5',
            'TestParametrized.test_three_things_composition_custom_names_1_3_6',
            'TestParametrized.test_three_things_composition_custom_names_1_4_5',
            'TestParametrized.test_three_things_composition_custom_names_1_4_6',
            'TestParametrized.test_three_things_composition_custom_names_2_3_5',
            'TestParametrized.test_three_things_composition_custom_names_2_3_6',
            'TestParametrized.test_three_things_composition_custom_names_2_4_5',
            'TestParametrized.test_three_things_composition_custom_names_2_4_6',
            'TestParametrized.test_two_things_custom_names_alternate_1__2',
            'TestParametrized.test_two_things_custom_names_alternate_1__3',
            'TestParametrized.test_two_things_custom_names_alternate_1__4',
        ]
        # 获取实际的测试名称列表
        test_names = _get_test_names_for_test_class(TestParametrized)
        # 断言预期的测试名称列表与实际的测试名称列表相等
        self.assertEqual(expected_test_names, test_names)

    def test_subtest_names(self):
        # 定义一个内部的测试类 TestParametrized，继承自 TestCase
        class TestParametrized(TestCase):
            # 参数化测试函数 test_custom_names，参数 bias 为 subtest 对象，分别命名为 'bias' 和 'no_bias'
            @parametrize("bias", [subtest(True, name='bias'),
                                  subtest(False, name='no_bias')])
            def test_custom_names(self, bias):
                pass

            # 参数化测试函数 test_two_things_custom_names，参数 x, y 为 subtest 对象，命名为 'double', 'triple', 'quadruple'
            @parametrize("x,y", [subtest((1, 2), name='double'),
                                 subtest((1, 3), name='triple'),
                                 subtest((1, 4), name='quadruple')])
            def test_two_things_custom_names(self, x, y):
                pass

        # 实例化参数化测试类 TestParametrized
        instantiate_parametrized_tests(TestParametrized)

        # 预期的测试名称列表
        expected_test_names = [
            'TestParametrized.test_custom_names_bias',
            'TestParametrized.test_custom_names_no_bias',
            'TestParametrized.test_two_things_custom_names_double',
            'TestParametrized.test_two_things_custom_names_quadruple',
            'TestParametrized.test_two_things_custom_names_triple',
        ]
        # 获取实际的测试名称列表
        test_names = _get_test_names_for_test_class(TestParametrized)
        # 断言预期的测试名称列表与实际的测试名称列表相等
        self.assertEqual(expected_test_names, test_names)
    def test_apply_param_specific_decorators(self):
        # Test that decorators can be applied on a per-param basis.

        def test_dec(func):
            # Apply a decorator by setting an attribute on the function
            func._decorator_applied = True
            return func

        class TestParametrized(TestCase):
            @parametrize("x", [subtest(1, name='one'),
                               subtest(2, name='two', decorators=[test_dec]),
                               subtest(3, name='three')])
            def test_param(self, x):
                pass

        # Instantiate parametrized tests for TestParametrized class
        instantiate_parametrized_tests(TestParametrized)

        # Retrieve test functions and their names for TestParametrized class
        for test_func, name in _get_test_funcs_for_test_class(TestParametrized):
            # Check if the decorator was applied based on test case name
            self.assertEqual(hasattr(test_func, '_decorator_applied'), name == 'test_param_two')

    def test_compose_param_specific_decorators(self):
        # Test that multiple per-param decorators compose correctly.

        def test_dec(func):
            # Apply a decorator by setting an attribute on the function
            func._decorator_applied = True
            return func

        class TestParametrized(TestCase):
            @parametrize("x", [subtest(1),
                               subtest(2, decorators=[test_dec]),
                               subtest(3)])
            @parametrize("y", [subtest(False, decorators=[test_dec]),
                               subtest(True)])
            def test_param(self, x, y):
                pass

        # Instantiate parametrized tests for TestParametrized class
        instantiate_parametrized_tests(TestParametrized)

        # Retrieve test functions and their names for TestParametrized class
        for test_func, name in _get_test_funcs_for_test_class(TestParametrized):
            # Determine if the decorator should be applied based on test case name
            should_apply = ('x_2' in name) or ('y_False' in name)
            self.assertEqual(hasattr(test_func, '_decorator_applied'), should_apply)

    def test_modules_decorator_misuse_error(self):
        # Test that @modules errors out when used with instantiate_parametrized_tests().

        class TestParametrized(TestCase):
            @modules(module_db)
            def test_modules(self, module_info):
                pass

        # Ensure that instantiating parametrized tests with modules decorator raises an error
        with self.assertRaisesRegex(RuntimeError, 'intended to be used in a device-specific context'):
            instantiate_parametrized_tests(TestParametrized)

    def test_ops_decorator_misuse_error(self):
        # Test that @ops errors out when used with instantiate_parametrized_tests().

        class TestParametrized(TestCase):
            @ops(op_db)
            def test_ops(self, module_info):
                pass

        # Ensure that instantiating parametrized tests with ops decorator raises an error
        with self.assertRaisesRegex(RuntimeError, 'intended to be used in a device-specific context'):
            instantiate_parametrized_tests(TestParametrized)
    def test_multiple_handling_of_same_param_error(self):
        # 定义一个测试方法，用于测试多个装饰器处理相同参数时的错误情况

        class TestParametrized(TestCase):
            @parametrize("x", range(3))
            @parametrize("x", range(5))
            def test_param(self, x):
                pass

        # 使用断言验证是否会抛出 RuntimeError，其中包含特定的错误信息
        with self.assertRaisesRegex(RuntimeError, 'multiple parametrization decorators'):
            instantiate_parametrized_tests(TestParametrized)

    @parametrize("x", [1, subtest(2, decorators=[unittest.expectedFailure]), 3])
    def test_subtest_expected_failure(self, x):
        # 定义一个测试方法，包含参数化的子测试，并预期其中一个会失败
        if x == 2:
            raise RuntimeError('Boom')

    @parametrize("x", [subtest(1, decorators=[unittest.expectedFailure]), 2, 3])
    @parametrize("y", [4, 5, subtest(6, decorators=[unittest.expectedFailure])])
    def test_two_things_subtest_expected_failure(self, x, y):
        # 定义一个测试方法，包含两个参数化的子测试，并预期其中一个或两个会失败
        if x == 1 or y == 6:
            raise RuntimeError('Boom')
class TestTestParametrizationDeviceType(TestCase):
    def test_unparametrized_names(self, device):
        # This test exists to protect against regressions in device / dtype test naming
        # due to parametrization logic.

        # 设定测试中使用的设备类型
        device = self.device_type

        class TestParametrized(TestCase):
            def test_device_specific(self, device):
                pass

            @dtypes(torch.float32, torch.float64)
            def test_device_dtype_specific(self, device, dtype):
                pass

        # 实例化设备类型特定的测试类，并将本地作用域的变量传递进去
        instantiate_device_type_tests(TestParametrized, locals(), only_for=device)

        # 获取特定设备类型的测试类
        device_cls = locals()[f'TestParametrized{device.upper()}']
        # 构建预期的测试名称列表，格式化字符串以包含设备和类名
        expected_test_names = [name.format(device_cls.__name__, device) for name in (
            '{}.test_device_dtype_specific_{}_float32',
            '{}.test_device_dtype_specific_{}_float64',
            '{}.test_device_specific_{}')
        ]
        # 获取实际测试类中的测试名称列表
        test_names = _get_test_names_for_test_class(device_cls)
        # 断言预期的测试名称列表与实际获取的测试名称列表相等
        self.assertEqual(expected_test_names, test_names)

    def test_empty_param_names(self, device):
        # If no param names are passed, ensure things still work without parametrization.
        # 设定测试中使用的设备类型
        device = self.device_type

        class TestParametrized(TestCase):
            @parametrize("", [])
            def test_foo(self, device):
                pass

            @parametrize("", range(5))
            def test_bar(self, device):
                pass

        # 实例化设备类型特定的测试类，并将本地作用域的变量传递进去
        instantiate_device_type_tests(TestParametrized, locals(), only_for=device)

        # 获取特定设备类型的测试类
        device_cls = locals()[f'TestParametrized{device.upper()}']
        # 构建预期的测试名称列表，格式化字符串以包含设备和类名
        expected_test_names = [name.format(device_cls.__name__, device) for name in (
            '{}.test_bar_{}',
            '{}.test_foo_{}')
        ]
        # 获取实际测试类中的测试名称列表
        test_names = _get_test_names_for_test_class(device_cls)
        # 断言预期的测试名称列表与实际获取的测试名称列表相等
        self.assertEqual(expected_test_names, test_names)

    def test_empty_param_list(self, device):
        # If no param values are passed, ensure a helpful error message is thrown.
        # In the wild, this could indicate reuse of an exhausted generator.
        # 设定测试中使用的设备类型
        device = self.device_type

        # 创建一个生成器
        generator = (a for a in range(5))

        class TestParametrized(TestCase):
            @parametrize("x", generator)
            def test_foo(self, device, x):
                pass

            # Reuse generator from first test function.
            @parametrize("y", generator)
            def test_bar(self, device, y):
                pass

        # 断言在实例化设备类型特定的测试类时，会引发特定的异常信息
        with self.assertRaisesRegex(ValueError, 'An empty arg_values was passed'):
            instantiate_device_type_tests(TestParametrized, locals(), only_for=device)
    # 定义一个测试方法，用于测试默认命名的情况，需要传入设备类型参数
    def test_default_names(self, device):
        # 设定测试设备类型
        device = self.device_type

        # 定义一个内部测试类 TestParametrized，继承自 TestCase
        class TestParametrized(TestCase):
            # 使用 parametrize 装饰器，对方法 test_default_names 进行参数化测试，参数包括 x 在 range(5) 内的取值
            @parametrize("x", range(5))
            def test_default_names(self, device, x):
                pass

            # 使用 parametrize 装饰器，对方法 test_two_things_default_names 进行参数化测试，参数包括 x,y 在给定的元组列表内的取值
            @parametrize("x,y", [(1, 2), (2, 3), (3, 4)])
            def test_two_things_default_names(self, device, x, y):
                pass

        # 实例化 TestParametrized 类的测试用例，根据当前上下文和设备类型进行筛选
        instantiate_device_type_tests(TestParametrized, locals(), only_for=device)

        # 根据设备类型动态获取 TestParametrized 类的本地引用
        device_cls = locals()[f'TestParametrized{device.upper()}']

        # 期望的测试方法名称列表，包括设备类型插入的格式化字符串
        expected_test_names = [name.format(device_cls.__name__, device) for name in (
            '{}.test_default_names_x_0_{}',
            '{}.test_default_names_x_1_{}',
            '{}.test_default_names_x_2_{}',
            '{}.test_default_names_x_3_{}',
            '{}.test_default_names_x_4_{}',
            '{}.test_two_things_default_names_x_1_y_2_{}',
            '{}.test_two_things_default_names_x_2_y_3_{}',
            '{}.test_two_things_default_names_x_3_y_4_{}')
        ]

        # 获取当前设备类型的测试方法名称列表
        test_names = _get_test_names_for_test_class(device_cls)

        # 断言期望的测试方法名称列表与实际获取的测试方法名称列表相等
        self.assertEqual(expected_test_names, test_names)

    # 定义一个测试方法，用于测试非原始类型的默认命名情况，需要传入设备类型参数
    def test_default_name_non_primitive(self, device):
        # 设定测试设备类型
        device = self.device_type

        # 定义一个内部测试类 TestParametrized，继承自 TestCase
        class TestParametrized(TestCase):
            # 使用 parametrize 装饰器，对方法 test_default_names 进行参数化测试，参数包括 x 在给定的列表内的非原始类型取值
            @parametrize("x", [1, .5, "foo", object()])
            def test_default_names(self, device, x):
                pass

            # 使用 parametrize 装饰器，对方法 test_two_things_default_names 进行参数化测试，参数包括 x,y 在给定的元组列表内的非原始类型取值
            @parametrize("x,y", [(1, object()), (object(), .5), (object(), object())])
            def test_two_things_default_names(self, device, x, y):
                pass

        # 实例化 TestParametrized 类的测试用例，根据当前上下文和设备类型进行筛选
        instantiate_device_type_tests(TestParametrized, locals(), only_for=device)

        # 根据设备类型动态获取 TestParametrized 类的本地引用
        device_cls = locals()[f'TestParametrized{device.upper()}']

        # 期望的测试方法名称列表，包括设备类型插入的格式化字符串，并按字母顺序排序
        expected_test_names = sorted(name.format(device_cls.__name__, device) for name in (
            '{}.test_default_names_x_1_{}',
            '{}.test_default_names_x_0_5_{}',
            '{}.test_default_names_x_foo_{}',
            '{}.test_default_names_x3_{}',
            '{}.test_two_things_default_names_x_1_y0_{}',
            '{}.test_two_things_default_names_x1_y_0_5_{}',
            '{}.test_two_things_default_names_x2_y2_{}')
        )

        # 获取当前设备类型的测试方法名称列表
        test_names = _get_test_names_for_test_class(device_cls)

        # 断言期望的测试方法名称列表与实际获取的测试方法名称列表相等
        self.assertEqual(expected_test_names, test_names)
    # 定义一个测试方法，参数包括设备类型
    def test_name_fn(self, device):
        # 设备类型设定为当前对象的设备类型
        device = self.device_type

        # 定义一个参数化测试类
        class TestParametrized(TestCase):
            # 使用 parametrize 装饰器，测试函数带有 bias 参数，自定义命名函数为 lambda 表达式
            @parametrize("bias", [False, True], name_fn=lambda b: 'bias' if b else 'no_bias')
            def test_custom_names(self, device, bias):
                pass

            # 使用 parametrize 装饰器，测试函数带有 x 参数，自定义命名函数为 str
            # 同时使用 parametrize 装饰器，测试函数带有 y 参数，自定义命名函数为 str
            # 同时使用 parametrize 装饰器，测试函数带有 z 参数，自定义命名函数为 str
            @parametrize("x", [1, 2], name_fn=str)
            @parametrize("y", [3, 4], name_fn=str)
            @parametrize("z", [5, 6], name_fn=str)
            def test_three_things_composition_custom_names(self, device, x, y, z):
                pass

            # 使用 parametrize 装饰器，测试函数带有 x,y 参数，自定义命名函数为 lambda 表达式
            def test_two_things_custom_names_alternate(self, device, x, y):
                pass

        # 将实例化的设备类型测试添加到当前的测试方法中
        instantiate_device_type_tests(TestParametrized, locals(), only_for=device)

        # 获取特定设备类型的测试类
        device_cls = locals()[f'TestParametrized{device.upper()}']

        # 期望的测试名称列表，使用 format 格式化字符串生成
        expected_test_names = [name.format(device_cls.__name__, device) for name in (
            '{}.test_custom_names_bias_{}',
            '{}.test_custom_names_no_bias_{}',
            '{}.test_three_things_composition_custom_names_1_3_5_{}',
            '{}.test_three_things_composition_custom_names_1_3_6_{}',
            '{}.test_three_things_composition_custom_names_1_4_5_{}',
            '{}.test_three_things_composition_custom_names_1_4_6_{}',
            '{}.test_three_things_composition_custom_names_2_3_5_{}',
            '{}.test_three_things_composition_custom_names_2_3_6_{}',
            '{}.test_three_things_composition_custom_names_2_4_5_{}',
            '{}.test_three_things_composition_custom_names_2_4_6_{}',
            '{}.test_two_things_custom_names_alternate_1__2_{}',
            '{}.test_two_things_custom_names_alternate_1__3_{}',
            '{}.test_two_things_custom_names_alternate_1__4_{}')
        ]

        # 获取测试类中的实际测试名称列表
        test_names = _get_test_names_for_test_class(device_cls)

        # 断言期望的测试名称列表和实际获取的测试名称列表是否相等
        self.assertEqual(expected_test_names, test_names)
    # 测试函数，用于验证子测试名称的生成是否正确
    def test_subtest_names(self, device):
        # 设备类型赋值
        device = self.device_type

        # 定义一个内部的测试类 TestParametrized
        class TestParametrized(TestCase):
            # 使用 parametrize 装饰器定义带参数的测试函数 test_custom_names
            @parametrize("bias", [subtest(True, name='bias'),
                                  subtest(False, name='no_bias')])
            def test_custom_names(self, device, bias):
                pass

            # 使用 parametrize 装饰器定义带多个参数的测试函数 test_two_things_custom_names
            @parametrize("x,y", [subtest((1, 2), name='double'),
                                 subtest((1, 3), name='triple'),
                                 subtest((1, 4), name='quadruple')])
            def test_two_things_custom_names(self, device, x, y):
                pass

        # 根据设备类型实例化 TestParametrized 类中的测试函数
        instantiate_device_type_tests(TestParametrized, locals(), only_for=device)

        # 根据设备类型获取对应的 TestParametrized 类
        device_cls = locals()[f'TestParametrized{device.upper()}']

        # 生成预期的测试名称列表
        expected_test_names = [name.format(device_cls.__name__, device) for name in (
            '{}.test_custom_names_bias_{}',
            '{}.test_custom_names_no_bias_{}',
            '{}.test_two_things_custom_names_double_{}',
            '{}.test_two_things_custom_names_quadruple_{}',
            '{}.test_two_things_custom_names_triple_{}')
        ]

        # 获取实际测试名称列表
        test_names = _get_test_names_for_test_class(device_cls)

        # 断言预期测试名称与实际测试名称相等
        self.assertEqual(expected_test_names, test_names)

    # 测试函数，用于验证操作组合的名称生成是否正确
    def test_ops_composition_names(self, device):
        # 设备类型赋值
        device = self.device_type

        # 定义一个内部的测试类 TestParametrized
        class TestParametrized(TestCase):
            # 使用 ops 和 parametrize 装饰器定义带参数的测试函数 test_op_parametrized
            @ops(op_db)
            @parametrize("flag", [False, True], lambda f: 'flag_enabled' if f else 'flag_disabled')
            def test_op_parametrized(self, device, dtype, op, flag):
                pass

        # 根据设备类型实例化 TestParametrized 类中的测试函数
        instantiate_device_type_tests(TestParametrized, locals(), only_for=device)

        # 根据设备类型获取对应的 TestParametrized 类
        device_cls = locals()[f'TestParametrized{device.upper()}']

        # 初始化预期测试名称列表
        expected_test_names = []

        # 遍历操作数据库中的操作
        for op in op_db:
            # 遍历操作支持的数据类型
            for dtype in op.supported_dtypes(torch.device(device).type):
                # 遍历标志部分
                for flag_part in ('flag_disabled', 'flag_enabled'):
                    # 生成预期的测试名称
                    expected_name = f'{device_cls.__name__}.test_op_parametrized_{op.formatted_name}_{flag_part}_{device}_{dtype_name(dtype)}'  # noqa: B950
                    expected_test_names.append(expected_name)

        # 获取实际测试名称列表
        test_names = _get_test_names_for_test_class(device_cls)

        # 断言预期测试名称与实际测试名称排序后相等
        self.assertEqual(sorted(expected_test_names), sorted(test_names))
    # 定义一个测试方法，测试模块组合的命名
    def test_modules_composition_names(self, device):
        # 设置测试设备类型
        device = self.device_type

        # 定义一个内部类 TestParametrized，继承自 TestCase
        class TestParametrized(TestCase):
            # 装饰器：将模块信息注入到测试方法中
            @modules(module_db)
            # 参数化装饰器：根据条件参数化测试方法
            @parametrize("flag", [False, True], lambda f: 'flag_enabled' if f else 'flag_disabled')
            # 定义参数化测试方法
            def test_module_parametrized(self, device, dtype, module_info, training, flag):
                pass

        # 实例化设备类型相关的测试用例
        instantiate_device_type_tests(TestParametrized, locals(), only_for=device)

        # 获取特定设备类型的测试类
        device_cls = locals()[f'TestParametrized{device.upper()}']
        # 初始化预期的测试方法名称列表
        expected_test_names = []

        # 遍历模块信息数据库中的每个模块信息
        for module_info in module_db:
            # 遍历每个模块信息中的数据类型
            for dtype in module_info.dtypes:
                # 遍历标志部分的两种情况：'flag_disabled' 和 'flag_enabled'
                for flag_part in ('flag_disabled', 'flag_enabled'):
                    # 根据模块信息是否有不同的训练和评估模式来确定期望的训练模式列表
                    expected_train_modes = (
                        ['train_mode', 'eval_mode'] if module_info.train_and_eval_differ else [''])
                    # 遍历期望的训练模式列表
                    for training_part in expected_train_modes:
                        # 构造预期的测试方法名称
                        expected_name = '{}.test_module_parametrized_{}{}_{}_{}_{}'.format(
                            device_cls.__name__, module_info.formatted_name,
                            '_' + training_part if len(training_part) > 0 else '',
                            flag_part, device, dtype_name(dtype))
                        # 将构造好的名称添加到预期测试方法名称列表中
                        expected_test_names.append(expected_name)

        # 获取设备类型测试类中实际存在的测试方法名称列表
        test_names = _get_test_names_for_test_class(device_cls)
        # 断言：预期的测试方法名称列表应当与实际存在的测试方法名称列表相等（无序）
        self.assertEqual(sorted(expected_test_names), sorted(test_names))
    def test_ops_decorator_applies_op_and_param_specific_decorators(self, device):
        # Test that decorators can be applied on a per-op / per-param basis.

        # 创建一个测试操作函数 test_op，用于返回输入的相反数
        def test_op(x):
            return -x

        # 创建一个测试装饰器函数 test_dec，用于标记函数是否被装饰器应用
        def test_dec(func):
            func._decorator_applied = True
            return func

        # 创建一个 OpInfo 对象 test_op_info，用于描述测试操作
        test_op_info = OpInfo(
            'test_op',  # 操作的名称
            op=test_op,  # 操作函数
            dtypes=floating_types(),  # 浮点数类型的数据类型
            sample_inputs_func=lambda _: [],  # 输入样本生成函数
            decorators=[  # 应用的装饰器信息列表
                DecorateInfo(
                    test_dec,  # 装饰器函数
                    'TestParametrized',  # 装饰器的名称
                    'test_op_param',  # 参数化测试操作的名称
                    device_type='cpu',  # 设备类型为 CPU
                    dtypes=[torch.float64],  # 数据类型为 torch.float64
                    active_if=lambda p: p['x'] == 2  # 活跃条件函数，参数 x 等于 2 时活跃
                )
            ])

        # 定义一个 TestCase 类 TestParametrized，用于包含参数化测试
        class TestParametrized(TestCase):
            @ops(op_db + [test_op_info])
            @parametrize("x", [2, 3])  # 参数化装饰器，测试参数 x 取值为 2 和 3
            def test_op_param(self, device, dtype, op, x):
                pass

            @ops(op_db + [test_op_info])
            @parametrize("y", [
                subtest(4),  # 子测试函数，测试参数 y 取值为 4
                subtest(5, decorators=[test_dec])  # 子测试函数，测试参数 y 取值为 5，应用 test_dec 装饰器
            ])
            def test_other(self, device, dtype, op, y):
                pass

            @decorateIf(test_dec, lambda p: p['dtype'] == torch.int16)
            @ops(op_db)
            def test_three(self, device, dtype, op):
                pass

        # 将设备类型设为当前对象的设备类型
        device = self.device_type
        # 实例化 TestParametrized 类的设备类型测试
        instantiate_device_type_tests(TestParametrized, locals(), only_for=device)
        # 获取设备类型的 TestParametrized 类
        device_cls = locals()[f'TestParametrized{device.upper()}']

        # 遍历设备类型的测试函数和名称
        for test_func, name in _get_test_funcs_for_test_class(device_cls):
            # 确定是否应用装饰器
            should_apply = (name == 'test_op_param_test_op_x_2_cpu_float64' or
                            ('test_other' in name and 'y_5' in name) or
                            ('test_three' in name and name.endswith('_int16')))
            # 断言测试函数是否应用了装饰器
            self.assertEqual(hasattr(test_func, '_decorator_applied'), should_apply)
    def test_modules_decorator_applies_module_and_param_specific_decorators(self, device):
        # Test that decorators can be applied on a per-module / per-param basis.

        # 定义一个测试模块 TestModule
        class TestModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.x = torch.nn.Parameter(torch.randn(3))

            def forward(self, y):
                return self.x + y

        # 定义一个装饰器函数 test_dec，用于标记被装饰的函数
        def test_dec(func):
            func._decorator_applied = True
            return func

        # 创建一个 ModuleInfo 对象，描述 TestModule，并设置装饰器
        test_module_info = ModuleInfo(
            TestModule,
            module_inputs_func=lambda _: [],
            decorators=[
                DecorateInfo(test_dec, 'TestParametrized', 'test_module_param',
                             device_type='cpu', dtypes=[torch.float64],
                             active_if=lambda p: p['x'] == 2)
            ])

        # 定义一个测试类 TestParametrized，包含多个测试方法
        class TestParametrized(TestCase):
            # 使用 modules 装饰器将 module_db 和 test_module_info 应用到测试方法
            @modules(module_db + [test_module_info])
            @parametrize("x", [2, 3])
            def test_module_param(self, device, dtype, module_info, training, x):
                pass

            # 使用 modules 装饰器将 module_db 和 test_module_info 应用到测试方法
            @modules(module_db + [test_module_info])
            @parametrize("y", [
                subtest(4),
                subtest(5, decorators=[test_dec])])
            def test_other(self, device, dtype, module_info, training, y):
                pass

            # 使用 decorateIf 装饰器将 test_dec 应用到测试方法
            @decorateIf(test_dec, lambda p: p['dtype'] == torch.float64)
            @modules(module_db)
            def test_three(self, device, dtype, module_info):
                pass

        # 获取设备类型并实例化设备相关的测试方法
        device = self.device_type
        instantiate_device_type_tests(TestParametrized, locals(), only_for=device)
        device_cls = locals()[f'TestParametrized{device.upper()}']

        # 遍历测试类中的测试方法，验证是否应用了装饰器
        for test_func, name in _get_test_funcs_for_test_class(device_cls):
            should_apply = (name == 'test_module_param_TestModule_x_2_cpu_float64' or
                            ('test_other' in name and 'y_5' in name) or
                            ('test_three' in name and name.endswith('float64')))
            self.assertEqual(hasattr(test_func, '_decorator_applied'), should_apply)
    def test_param_specific_decoration(self, device):
        # 定义一个测试装饰器函数，将 _decorator_applied 属性设置为 True
        def test_dec(func):
            func._decorator_applied = True
            return func

        # 定义一个继承自 TestCase 的参数化测试类
        class TestParametrized(TestCase):
            # 使用 decorateIf 装饰器，根据条件动态装饰测试方法
            @decorateIf(test_dec, lambda params: params["x"] == 1 and params["y"])
            # 使用 parametrize 装饰器，为 x 参数添加多个值
            @parametrize("x", range(5))
            # 使用 parametrize 装饰器，为 y 参数添加多个值
            @parametrize("y", [False, True])
            def test_param(self, x, y):
                pass

        # 获取当前设备类型
        device = self.device_type
        # 实例化设备类型相关的测试
        instantiate_device_type_tests(TestParametrized, locals(), only_for=device)
        # 根据设备类型获取相应的测试类
        device_cls = locals()[f'TestParametrized{device.upper()}']

        # 遍历测试类中的测试方法和名称
        for test_func, name in _get_test_funcs_for_test_class(device_cls):
            # 判断是否应用了装饰器，名称中包含 'test_param_x_1_y_True' 表示应用了装饰器
            should_apply = ('test_param_x_1_y_True' in name)
            self.assertEqual(hasattr(test_func, '_decorator_applied'), should_apply)

    def test_dtypes_composition_valid(self, device):
        # 测试 @parametrize 和 @dtypes 如何组合工作，其中 @parametrize 没有设置 dtype

        # 获取当前设备类型
        device = self.device_type

        # 定义一个继承自 TestCase 的参数化测试类
        class TestParametrized(TestCase):
            # 使用 @dtypes 装饰器，为测试方法添加多个 dtype
            @dtypes(torch.float32, torch.float64)
            # 使用 @parametrize 装饰器，为 x 参数添加多个值
            @parametrize("x", range(3))
            def test_parametrized(self, x, dtype):
                pass

        # 实例化设备类型相关的测试
        instantiate_device_type_tests(TestParametrized, locals(), only_for=device)

        # 根据设备类型获取相应的测试类
        device_cls = locals()[f'TestParametrized{device.upper()}']
        # 构建预期的测试方法名称列表
        expected_test_names = [name.format(device_cls.__name__, device) for name in (
            '{}.test_parametrized_x_0_{}_float32',
            '{}.test_parametrized_x_0_{}_float64',
            '{}.test_parametrized_x_1_{}_float32',
            '{}.test_parametrized_x_1_{}_float64',
            '{}.test_parametrized_x_2_{}_float32',
            '{}.test_parametrized_x_2_{}_float64')
        ]
        # 获取测试类中实际的测试方法名称列表
        test_names = _get_test_names_for_test_class(device_cls)
        # 断言预期名称和实际名称列表排序后是否相同
        self.assertEqual(sorted(expected_test_names), sorted(test_names))

    def test_dtypes_composition_invalid(self, device):
        # 测试 @dtypes 在 @parametrize 装饰器设置 dtype 时的行为

        # 获取当前设备类型
        device = self.device_type

        # 定义一个继承自 TestCase 的参数化测试类
        class TestParametrized(TestCase):
            # 使用 @dtypes 装饰器，为测试方法添加多个 dtype
            @dtypes(torch.float32, torch.float64)
            # 使用 @parametrize 装饰器，为 dtype 参数添加多个值
            @parametrize("dtype", [torch.int32, torch.int64])
            def test_parametrized(self, dtype):
                pass

        # 断言运行时是否抛出预期的异常信息，因为 @dtypes 和 @parametrize 同时尝试设置 dtype
        with self.assertRaisesRegex(RuntimeError, "handled multiple times"):
            instantiate_device_type_tests(TestParametrized, locals(), only_for=device)

        # 验证 @ops + @dtypes 在同时尝试设置 dtype 时的错误行为

        # 定义另一个继承自 TestCase 的参数化测试类
        class TestParametrized(TestCase):
            # 使用 @dtypes 装饰器，为测试方法添加多个 dtype
            @dtypes(torch.float32, torch.float64)
            # 使用 @ops 装饰器，为测试方法添加多个操作
            @ops(op_db)
            def test_parametrized(self, op, dtype):
                pass

        # 断言运行时是否抛出预期的异常信息，因为 @ops 和 @dtypes 同时尝试设置 dtype
        with self.assertRaisesRegex(RuntimeError, "handled multiple times"):
            instantiate_device_type_tests(TestParametrized, locals(), only_for=device)
    # 定义一个测试方法，用于测试相同参数错误的多个装饰器处理情况
    def test_multiple_handling_of_same_param_error(self, device):
        # 断言：期望运行时错误包含指定消息
        with self.assertRaisesRegex(RuntimeError, "handled multiple times"):
            # 实例化设备类型测试，传入 TestParametrized 类和当前作用域的局部变量
            instantiate_device_type_tests(TestParametrized, locals(), only_for=device)

    # 使用 @parametrize 装饰器定义一个测试方法，参数 x 取值为 [1, subtest(2, decorators=[unittest.expectedFailure]), 3]
    def test_subtest_expected_failure(self, device, x):
        # 如果 x 的值为 2，则抛出运行时错误 'Boom'
        if x == 2:
            raise RuntimeError('Boom')

    # 使用 @parametrize 装饰器定义一个测试方法，参数 x 取值为 [subtest(1, decorators=[unittest.expectedFailure]), 2, 3]，
    # 参数 y 取值为 [4, 5, subtest(6, decorators=[unittest.expectedFailure])]
    def test_two_things_subtest_expected_failure(self, device, x, y):
        # 如果 x 的值为 1 或者 y 的值为 6，则抛出运行时错误 'Boom'
        if x == 1 or y == 6:
            raise RuntimeError('Boom')
# 实例化带参数的测试用例，使用给定的测试类 TestTestParametrization
instantiate_parametrized_tests(TestTestParametrization)

# 实例化设备类型相关的测试用例，使用给定的测试类 TestTestParametrizationDeviceType，
# 并在全局范围内执行（globals() 返回当前全局符号表的字典）
instantiate_device_type_tests(TestTestParametrizationDeviceType, globals())

# 定义一个名为 TestImports 的测试类，继承自 TestCase
class TestImports(TestCase):

    # 定义一个类方法 _check_python_output，接收一个程序字符串作为参数并返回字符串
    @classmethod
    def _check_python_output(cls, program) -> str:
        # 使用 subprocess 模块执行命令，运行给定的程序字符串
        return subprocess.check_output(
            # 执行命令：使用系统的 Python 解释器运行程序字符串
            [sys.executable, "-W", "all", "-c", program],
            # 将 stderr 重定向到 stdout，合并所有输出到标准输出
            stderr=subprocess.STDOUT,
            # 在 Windows 系统上，默认的当前工作目录可能导致 `import torch` 失败，
            # 所以设置当前工作目录为此脚本文件所在的目录
            cwd=os.path.dirname(os.path.realpath(__file__))
        ).decode("utf-8")  # 将输出解码为 UTF-8 编码的字符串
    # 定义一个测试方法，用于检查所有 torch 内部模块是否可以被成功导入
    def test_circular_dependencies(self) -> None:
        """ Checks that all modules inside torch can be imported
        Prevents regression reported in https://github.com/pytorch/pytorch/issues/77441 """
        
        # 忽略无法导入的模块列表，这些模块有特定依赖或平台限制
        ignored_modules = ["torch.utils.tensorboard",  # 依赖于 tensorboard
                           "torch.distributed.elastic.rendezvous",  # 依赖于 etcd
                           "torch.backends._coreml",  # 依赖于 pycoreml
                           "torch.contrib.",  # 不明情况
                           "torch.testing._internal.distributed.",  # 导入失败
                           "torch.ao.pruning._experimental.",  # 依赖于 pytorch_lightning，非用户接口
                           "torch.onnx._internal.fx",  # 依赖于 onnx-script
                           "torch._inductor.runtime.triton_helpers",  # 依赖于 triton
                           "torch._inductor.codegen.cuda",  # 依赖于 cutlass
                           ]
        
        # 根据 Python 版本和操作系统平台决定是否忽略额外的模块
        if not sys.version_info >= (3, 9):
            ignored_modules.append("torch.utils.benchmark")  # 参见 https://github.com/pytorch/pytorch/issues/77801
        if IS_WINDOWS or IS_MACOS or IS_JETSON:
            # 在 Windows 上，除了 nn.api.，Distributed 应该可以导入；但在 Mac 上不行
            if IS_MACOS or IS_JETSON:
                ignored_modules.append("torch.distributed.")
            else:
                ignored_modules.append("torch.distributed.nn.api.")
                ignored_modules.append("torch.distributed.optim.")
                ignored_modules.append("torch.distributed.rpc.")
            ignored_modules.append("torch.testing._internal.dist_utils")
            # 这些模块都有传递依赖于 distributed
            ignored_modules.append("torch.nn.parallel._replicated_tensor_ddp_interop")
            ignored_modules.append("torch.testing._internal.common_fsdp")
            ignored_modules.append("torch.testing._internal.common_distributed")

        # 获取 torch 库的根目录
        torch_dir = os.path.dirname(torch.__file__)
        
        # 遍历 torch 库目录下的所有子目录和文件
        for base, folders, files in os.walk(torch_dir):
            # 计算当前目录的相对路径，并将其转换为模块名前缀形式
            prefix = os.path.relpath(base, os.path.dirname(torch_dir)).replace(os.path.sep, ".")
            
            # 遍历当前目录下的所有文件
            for f in files:
                # 如果文件不是以 .py 结尾，则跳过
                if not f.endswith(".py"):
                    continue
                
                # 构建模块名
                mod_name = f"{prefix}.{f[:-3]}" if f != "__init__.py" else prefix
                
                # 不导入可执行模块
                if f == "__main__.py":
                    continue
                
                # 如果模块名以 ignored_modules 中的任何元素开头，则跳过导入尝试
                if any(mod_name.startswith(x) for x in ignored_modules):
                    continue
                
                # 尝试导入模块，若失败则抛出异常
                try:
                    mod = importlib.import_module(mod_name)
                except Exception as e:
                    raise RuntimeError(f"Failed to import {mod_name}: {e}") from e
                
                # 断言导入的对象是一个模块
                self.assertTrue(inspect.ismodule(mod))

    @unittest.skipIf(IS_WINDOWS, "TODO enable on Windows")
    # 定义测试方法，验证懒加载模块是否真的懒加载
    def test_lazy_imports_are_lazy(self) -> None:
        # 执行 Python 命令，检查导入 torch 时所有 lazy 模块是否都未在 sys.modules 中
        out = self._check_python_output("import sys;import torch;print(all(x not in sys.modules for x in torch._lazy_modules))")
        # 断言输出结果去除空白字符后是否为 "True"
        self.assertEqual(out.strip(), "True")

    # 如果在 Windows 平台下，跳过测试，因为在 CPU 上导入 torch+CUDA 会产生警告
    @unittest.skipIf(IS_WINDOWS, "importing torch+CUDA on CPU results in warning")
    def test_no_warning_on_import(self) -> None:
        # 执行 Python 命令，检查导入 torch 是否输出空字符串
        out = self._check_python_output("import torch")
        # 断言输出结果是否为空字符串
        self.assertEqual(out, "")

    # 测试不导入 sympy 是否有效
    def test_not_import_sympy(self) -> None:
        # 执行 Python 命令，检查导入 torch 时是否确实没有导入 sympy
        out = self._check_python_output("import torch;import sys;print('sympy' not in sys.modules)")
        # 断言输出结果去除空白字符后是否为 "True"
        self.assertEqual(out.strip(), "True",
                         "PyTorch should not depend on SymPy at import time as importing SymPy is *very* slow.\n"
                         "See the beginning of the following blog post for how to profile and find which file is importing sympy:\n"
                         "https://dev-discuss.pytorch.org/t/delving-into-what-happens-when-you-import-torch/1589\n\n"
                         "If you hit this error, you may want to:\n"
                         "  - Refactor your code to avoid depending on sympy files you may not need to depend\n"
                         "  - Use TYPE_CHECKING if you are using sympy + strings if you are using sympy on type annotations\n"
                         "  - Import things that depend on SymPy locally")

    # 如果在 Windows 平台下，跳过测试，因为在 CPU 上导入 torch+CUDA 会产生警告
    @unittest.skipIf(IS_WINDOWS, "importing torch+CUDA on CPU results in warning")
    # 参数化测试，对 'path' 参数进行多次测试，分别为 'torch' 和 'functorch'
    @parametrize('path', ['torch', 'functorch'])
    def test_no_mutate_global_logging_on_import(self, path) -> None:
        # 调用 logging.basicConfig 等操作修改全局的日志状态，不应在导入 torch（或我们拥有的其他子模块）时执行这些操作，因为用户不希望如此。
        expected = 'abcdefghijklmnopqrstuvwxyz'
        commands = [
            'import logging',
            f'import {path}',
            '_logger = logging.getLogger("torch_test_testing")',
            'logging.root.addHandler(logging.StreamHandler())',
            'logging.root.setLevel(logging.INFO)',
            f'_logger.info("{expected}")'
        ]
        # 执行一系列 Python 命令，检查输出结果是否符合预期字符串
        out = self._check_python_output("; ".join(commands))
        # 断言输出结果去除空白字符后是否等于预期字符串
        self.assertEqual(out.strip(), expected)
class TestOpInfos(TestCase):
    # 定义测试类 TestOpInfos，继承自 TestCase

    def test_sample_input(self) -> None:
        # 定义测试方法 test_sample_input，无返回值

        a, b, c, d, e = (object() for _ in range(5))
        # 创建对象 a, b, c, d, e，它们都是 object 类型的实例

        # Construction with natural syntax
        s = SampleInput(a, b, c, d=d, e=e)
        # 使用自然语法进行构造
        assert s.input is a
        # 断言 s.input 等于 a
        assert s.args == (b, c)
        # 断言 s.args 等于 (b, c)
        assert s.kwargs == dict(d=d, e=e)
        # 断言 s.kwargs 等于 {'d': d, 'e': e}

        # Construction with explicit args and kwargs
        s = SampleInput(a, args=(b,), kwargs=dict(c=c, d=d, e=e))
        # 使用显式的 args 和 kwargs 进行构造
        assert s.input is a
        # 断言 s.input 等于 a
        assert s.args == (b,)
        # 断言 s.args 等于 (b,)
        assert s.kwargs == dict(c=c, d=d, e=e)
        # 断言 s.kwargs 等于 {'c': c, 'd': d, 'e': e}

        # Construction with a mixed form will error
        # 使用混合形式构造会引发错误
        with self.assertRaises(AssertionError):
            s = SampleInput(a, b, c, args=(d, e))
            # 断言构造 SampleInput(a, b, c, args=(d, e)) 会引发 AssertionError

        with self.assertRaises(AssertionError):
            s = SampleInput(a, b, c, kwargs=dict(d=d, e=e))
            # 断言构造 SampleInput(a, b, c, kwargs=dict(d=d, e=e)) 会引发 AssertionError

        with self.assertRaises(AssertionError):
            s = SampleInput(a, args=(b, c), d=d, e=e)
            # 断言构造 SampleInput(a, args=(b, c), d=d, e=e) 会引发 AssertionError

        with self.assertRaises(AssertionError):
            s = SampleInput(a, b, c=c, kwargs=dict(d=d, e=e))
            # 断言构造 SampleInput(a, b, c=c, kwargs=dict(d=d, e=e)) 会引发 AssertionError

        # Mixing metadata into "natural" construction will error
        # 将元数据混合到 "自然" 构造中会引发错误
        with self.assertRaises(AssertionError):
            s = SampleInput(a, b, name="foo")
            # 断言构造 SampleInput(a, b, name="foo") 会引发 AssertionError

        with self.assertRaises(AssertionError):
            s = SampleInput(a, b, output_process_fn_grad=lambda x: x)
            # 断言构造 SampleInput(a, b, output_process_fn_grad=lambda x: x) 会引发 AssertionError

        with self.assertRaises(AssertionError):
            s = SampleInput(a, b, broadcasts_input=True)
            # 断言构造 SampleInput(a, b, broadcasts_input=True) 会引发 AssertionError

        # But when only input is given, metadata is allowed for backward
        # compatibility
        # 但是当只给出输入时，允许使用元数据以保持向后兼容性
        s = SampleInput(a, broadcasts_input=True)
        # 构造 SampleInput(a, broadcasts_input=True)
        assert s.input is a
        # 断言 s.input 等于 a
        assert s.broadcasts_input
        # 断言 s.broadcasts_input 为真值

    def test_sample_input_metadata(self) -> None:
        # 定义测试方法 test_sample_input_metadata，无返回值

        a, b = (object() for _ in range(2))
        # 创建对象 a, b，它们都是 object 类型的实例

        s1 = SampleInput(a, b=b)
        # 使用关键字参数 b 构造 SampleInput 实例 s1

        self.assertIs(s1.output_process_fn_grad(None), None)
        # 使用 s1 调用 output_process_fn_grad(None)，断言结果为 None
        self.assertFalse(s1.broadcasts_input)
        # 断言 s1.broadcasts_input 为假值
        self.assertEqual(s1.name, "")
        # 断言 s1.name 等于 ""

        s2 = s1.with_metadata(
            output_process_fn_grad=lambda x: a,
            broadcasts_input=True,
            name="foo",
        )
        # 使用 s1 调用 with_metadata 方法添加元数据，构造实例 s2

        self.assertIs(s1, s2)
        # 断言 s1 和 s2 是同一个实例
        self.assertIs(s2.output_process_fn_grad(None), a)
        # 使用 s2 调用 output_process_fn_grad(None)，断言结果为 a
        self.assertTrue(s2.broadcasts_input)
        # 断言 s2.broadcasts_input 为真值
        self.assertEqual(s2.name, "foo")
        # 断言 s2.name 等于 "foo"


# Tests that validate the various sample generating functions on each OpInfo.
# 验证每个 OpInfo 上各种样本生成函数的测试。
class TestOpInfoSampleFunctions(TestCase):

    @ops(op_db, dtypes=OpDTypes.any_one)
    # 使用 ops 装饰器定义测试方法，接受 op_db 和 OpDTypes.any_one 作为参数
    def test_opinfo_sample_generators(self, device, dtype, op):
        # 定义测试方法 test_opinfo_sample_generators，接受 device, dtype, op 作为参数

        # Test op.sample_inputs doesn't generate multiple samples when called
        # 测试 op.sample_inputs 被调用时不会生成多个样本
        samples = op.sample_inputs(device, dtype)
        # 调用 op.sample_inputs 方法获取样本
        self.assertIsInstance(samples, Iterator)
        # 断言 samples 是 Iterator 类型的实例

    @ops([op for op in op_db if op.reference_inputs_func is not None], dtypes=OpDTypes.any_one)
    # 使用 ops 装饰器定义测试方法，接受 op_db 和 OpDTypes.any_one 作为参数
    def test_opinfo_reference_generators(self, device, dtype, op):
        # 定义测试方法 test_opinfo_reference_generators，接受 device, dtype, op 作为参数

        # Test op.reference_inputs doesn't generate multiple samples when called
        # 测试 op.reference_inputs 被调用时不会生成多个样本
        samples = op.reference_inputs(device, dtype)
        # 调用 op.reference_inputs 方法获取样本
        self.assertIsInstance(samples, Iterator)
        # 断言 samples 是 Iterator 类型的实例
    @ops([op for op in op_db if op.error_inputs_func is not None], dtypes=OpDTypes.none)
    # 使用 op_db 中具有非空 error_inputs_func 属性的操作符列表作为参数调用 ops 装饰器
    def test_opinfo_error_generators(self, device, op):
        # 测试 op.error_inputs 调用时不会生成多个输入
        # 调用 op.error_inputs 方法获取样本数据
        samples = op.error_inputs(device)
        # 断言 samples 是一个迭代器的实例
        self.assertIsInstance(samples, Iterator)
# 调用函数 instantiate_device_type_tests，将 TestOpInfoSampleFunctions 和 globals() 作为参数传递给它
instantiate_device_type_tests(TestOpInfoSampleFunctions, globals())

# 调用函数 instantiate_parametrized_tests，将 TestImports 作为参数传递给它
instantiate_parametrized_tests(TestImports)

# 检查当前模块是否作为主程序运行，如果是，则调用 run_tests() 函数
if __name__ == '__main__':
    run_tests()
```