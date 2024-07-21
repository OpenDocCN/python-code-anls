# `.\pytorch\test\test_type_info.py`

```py
# mypy: allow-untyped-defs
# Owner(s): ["module: typing"]

# 从 torch.testing._internal.common_utils 导入必要的函数和类
from torch.testing._internal.common_utils import (
    load_tests,
    run_tests,
    set_default_dtype,
    TEST_NUMPY,
    TestCase,
)

# load_tests 函数用于在 sandcastle 上自动过滤测试以进行分片。这行代码抑制了 flake 警告
load_tests = load_tests

# 导入 sys 和 unittest 模块
import sys
import unittest

# 导入 torch 库
import torch

# 如果 TEST_NUMPY 为真，则导入 numpy 库
if TEST_NUMPY:
    import numpy as np

# 定义 TestDTypeInfo 类，继承自 unittest 的 TestCase 类
class TestDTypeInfo(TestCase):
    # 定义测试函数 test_invalid_input
    def test_invalid_input(self):
        # 遍历不同的数据类型，测试是否会抛出 TypeError 异常
        for dtype in [
            torch.float16,
            torch.float32,
            torch.float64,
            torch.bfloat16,
            torch.complex64,
            torch.complex128,
            torch.bool,
        ]:
            with self.assertRaises(TypeError):
                _ = torch.iinfo(dtype)

        # 再次遍历不同的数据类型，测试是否会抛出 TypeError 和 RuntimeError 异常
        for dtype in [
            torch.int64,
            torch.int32,
            torch.int16,
            torch.int8,
            torch.uint8,
            torch.bool,
        ]:
            with self.assertRaises(TypeError):
                _ = torch.finfo(dtype)
            with self.assertRaises(RuntimeError):
                dtype.to_complex()

    # 根据 TEST_NUMPY 的真假进行条件跳过测试
    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    # 定义测试函数 test_iinfo
    def test_iinfo(self):
        # 遍历不同的整数数据类型
        for dtype in [torch.int64, torch.int32, torch.int16, torch.int8, torch.uint8]:
            # 创建 torch 张量 x，指定数据类型为当前遍历的 dtype
            x = torch.zeros((2, 2), dtype=dtype)
            # 调用 torch.iinfo 获取该数据类型的信息
            xinfo = torch.iinfo(x.dtype)
            # 将 torch 张量 x 转移到 CPU 并转换为 numpy 数组 xn
            xn = x.cpu().numpy()
            # 使用 numpy.iinfo 获取对应 numpy 数组的信息
            xninfo = np.iinfo(xn.dtype)
            # 断言 torch 和 numpy 获取的信息应该一致
            self.assertEqual(xinfo.bits, xninfo.bits)
            self.assertEqual(xinfo.max, xninfo.max)
            self.assertEqual(xinfo.min, xninfo.min)
            self.assertEqual(xinfo.dtype, xninfo.dtype)

    # 根据 TEST_NUMPY 的真假进行条件跳过测试
    @unittest.skipIf(not TEST_NUMPY, "Numpy not found")
    # 定义一个测试方法，用于验证 torch.finfo 的行为
    def test_finfo(self):
        # 遍历不同的数据类型
        for dtype in [
            torch.float16,
            torch.float32,
            torch.float64,
            torch.complex64,
            torch.complex128,
        ]:
            # 创建一个全零张量，指定数据类型为当前循环的 dtype
            x = torch.zeros((2, 2), dtype=dtype)
            # 获取该数据类型的 finfo 信息
            xinfo = torch.finfo(x.dtype)
            # 将张量转移到 CPU 并转换为 NumPy 数组
            xn = x.cpu().numpy()
            # 获取 NumPy 数组的 finfo 信息
            xninfo = np.finfo(xn.dtype)
            # 断言张量和 NumPy 数组的 finfo 信息应该一致
            self.assertEqual(xinfo.bits, xninfo.bits)
            self.assertEqual(xinfo.max, xninfo.max)
            self.assertEqual(xinfo.min, xninfo.min)
            self.assertEqual(xinfo.eps, xninfo.eps)
            self.assertEqual(xinfo.tiny, xninfo.tiny)
            self.assertEqual(xinfo.resolution, xninfo.resolution)
            self.assertEqual(xinfo.dtype, xninfo.dtype)
            # 如果当前数据类型不是复数类型，则进入下面的条件判断
            if not dtype.is_complex:
                # 设置当前数据类型为默认数据类型，并验证其 finfo 信息
                with set_default_dtype(dtype):
                    self.assertEqual(torch.finfo(dtype), torch.finfo())

        # 特殊测试用例，针对 BFloat16 类型
        x = torch.zeros((2, 2), dtype=torch.bfloat16)
        xinfo = torch.finfo(x.dtype)
        # 验证 BFloat16 类型的 finfo 信息
        self.assertEqual(xinfo.bits, 16)
        self.assertEqual(xinfo.max, 3.38953e38)
        self.assertEqual(xinfo.min, -3.38953e38)
        self.assertEqual(xinfo.eps, 0.0078125)
        self.assertEqual(xinfo.tiny, 1.17549e-38)
        self.assertEqual(xinfo.tiny, xinfo.smallest_normal)
        self.assertEqual(xinfo.resolution, 0.01)
        self.assertEqual(xinfo.dtype, "bfloat16")
        # 使用当前类型为默认数据类型，并验证其 finfo 信息
        with set_default_dtype(x.dtype):
            self.assertEqual(torch.finfo(x.dtype), torch.finfo())

        # 特殊测试用例，针对 Float8_E5M2 类型
        xinfo = torch.finfo(torch.float8_e5m2)
        # 验证 Float8_E5M2 类型的 finfo 信息
        self.assertEqual(xinfo.bits, 8)
        self.assertEqual(xinfo.max, 57344.0)
        self.assertEqual(xinfo.min, -57344.0)
        self.assertEqual(xinfo.eps, 0.25)
        self.assertEqual(xinfo.tiny, 6.10352e-05)
        self.assertEqual(xinfo.resolution, 1.0)
        self.assertEqual(xinfo.dtype, "float8_e5m2")

        # 特殊测试用例，针对 Float8_E4M3FN 类型
        xinfo = torch.finfo(torch.float8_e4m3fn)
        # 验证 Float8_E4M3FN 类型的 finfo 信息
        self.assertEqual(xinfo.bits, 8)
        self.assertEqual(xinfo.max, 448.0)
        self.assertEqual(xinfo.min, -448.0)
        self.assertEqual(xinfo.eps, 0.125)
        self.assertEqual(xinfo.tiny, 0.015625)
        self.assertEqual(xinfo.resolution, 1.0)
        self.assertEqual(xinfo.dtype, "float8_e4m3fn")

    # 定义一个测试方法，用于验证 torch.float32.to_complex() 的行为
    def test_to_complex(self):
        # 回归测试，验证是否存在内存泄漏问题
        # 如果引用计数泄漏，集合 ref_cnt 的长度将超过 3
        ref_cnt = {sys.getrefcount(torch.float32.to_complex()) for _ in range(10)}
        self.assertLess(len(ref_cnt), 3)

        # 验证不同数据类型的转换结果是否正确
        self.assertEqual(torch.float64.to_complex(), torch.complex128)
        self.assertEqual(torch.float32.to_complex(), torch.complex64)
        self.assertEqual(torch.float16.to_complex(), torch.complex32)
    def test_to_real(self):
        # 定义一个测试方法，用于验证复数到实数的转换功能
        
        # 生成一个集合，包含10次迭代中每次调用torch.cfloat.to_real()后的引用计数
        ref_cnt = {sys.getrefcount(torch.cfloat.to_real()) for _ in range(10)}
        
        # 断言集合的长度小于3，以确保引用计数未泄漏（正常情况下应为1或2）
        self.assertLess(len(ref_cnt), 3)

        # 断言torch.complex128.to_real()的返回值等于torch.double
        self.assertEqual(torch.complex128.to_real(), torch.double)
        
        # 断言torch.complex64.to_real()的返回值等于torch.float32
        self.assertEqual(torch.complex64.to_real(), torch.float32)
        
        # 断言torch.complex32.to_real()的返回值等于torch.float16
        self.assertEqual(torch.complex32.to_real(), torch.float16)
# 如果当前脚本作为主程序执行（而非被导入到其他模块），则执行以下代码块
if __name__ == "__main__":
    # 设置 TestCase 类的默认数据类型检查为启用状态
    TestCase._default_dtype_check_enabled = True
    # 运行测试函数（假设这里是一个函数或方法来执行测试）
    run_tests()
```