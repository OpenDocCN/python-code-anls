# `D:\src\scipysrc\scipy\scipy\io\_harwell_boeing\tests\test_fortran_format.py`

```
import numpy as np  # 导入 NumPy 库，通常用于科学计算

from numpy.testing import assert_equal  # 导入 NumPy 的测试模块中的断言函数 assert_equal
from pytest import raises as assert_raises  # 导入 pytest 库中的 raises 函数，并重命名为 assert_raises

from scipy.io._harwell_boeing._fortran_format_parser import (
        FortranFormatParser, IntFormat, ExpFormat, BadFortranFormat)
# 从 scipy 库中导入 FortranFormatParser 类以及相关的格式化类 IntFormat、ExpFormat 和异常类 BadFortranFormat


class TestFortranFormatParser:
    def setup_method(self):  # 定义 setup_method 方法，用于设置测试环境
        self.parser = FortranFormatParser()  # 创建 FortranFormatParser 的实例 self.parser

    def _test_equal(self, format, ref):  # 定义 _test_equal 方法，用于测试解析结果是否与预期相等
        ret = self.parser.parse(format)  # 调用 parse 方法解析 format，并保存结果到 ret
        assert_equal(ret.__dict__, ref.__dict__)  # 使用 assert_equal 检查 ret 对象与 ref 对象的属性是否相等

    def test_simple_int(self):  # 测试简单整数格式解析
        self._test_equal("(I4)", IntFormat(4))  # 调用 _test_equal 方法，测试 "(I4)" 的解析结果是否与 IntFormat(4) 相等

    def test_simple_repeated_int(self):  # 测试重复整数格式解析
        self._test_equal("(3I4)", IntFormat(4, repeat=3))  # 调用 _test_equal 方法，测试 "(3I4)" 的解析结果是否与 IntFormat(4, repeat=3) 相等

    def test_simple_exp(self):  # 测试简单指数格式解析
        self._test_equal("(E4.3)", ExpFormat(4, 3))  # 调用 _test_equal 方法，测试 "(E4.3)" 的解析结果是否与 ExpFormat(4, 3) 相等

    def test_exp_exp(self):  # 测试复合指数格式解析
        self._test_equal("(E8.3E3)", ExpFormat(8, 3, 3))  # 调用 _test_equal 方法，测试 "(E8.3E3)" 的解析结果是否与 ExpFormat(8, 3, 3) 相等

    def test_repeat_exp(self):  # 测试重复指数格式解析
        self._test_equal("(2E4.3)", ExpFormat(4, 3, repeat=2))  # 调用 _test_equal 方法，测试 "(2E4.3)" 的解析结果是否与 ExpFormat(4, 3, repeat=2) 相等

    def test_repeat_exp_exp(self):  # 测试复合重复指数格式解析
        self._test_equal("(2E8.3E3)", ExpFormat(8, 3, 3, repeat=2))  # 调用 _test_equal 方法，测试 "(2E8.3E3)" 的解析结果是否与 ExpFormat(8, 3, 3, repeat=2) 相等

    def test_wrong_formats(self):  # 测试不正确的格式字符串解析
        def _test_invalid(bad_format):  # 定义内部函数 _test_invalid，用于测试不正确的格式字符串
            assert_raises(BadFortranFormat, lambda: self.parser.parse(bad_format))  # 使用 assert_raises 检查解析不正确格式字符串时是否抛出 BadFortranFormat 异常
        _test_invalid("I4")  # 调用 _test_invalid 函数，测试解析 "I4" 是否会抛出异常
        _test_invalid("(E4)")  # 调用 _test_invalid 函数，测试解析 "(E4)" 是否会抛出异常
        _test_invalid("(E4.)")  # 调用 _test_invalid 函数，测试解析 "(E4.)" 是否会抛出异常
        _test_invalid("(E4.E3)")  # 调用 _test_invalid 函数，测试解析 "(E4.E3)" 是否会抛出异常


class TestIntFormat:
    def test_to_fortran(self):  # 测试 IntFormat 类的 to_fortran 方法
        f = [IntFormat(10), IntFormat(12, 10), IntFormat(12, 10, 3)]  # 创建 IntFormat 对象列表 f
        res = ["(I10)", "(I12.10)", "(3I12.10)"]  # 预期的格式化字符串列表 res

        for i, j in zip(f, res):  # 遍历 f 和 res
            assert_equal(i.fortran_format, j)  # 使用 assert_equal 检查 IntFormat 对象的 fortran_format 属性是否与预期相等

    def test_from_number(self):  # 测试 IntFormat 类的 from_number 方法
        f = [10, -12, 123456789]  # 创建整数列表 f
        r_f = [IntFormat(3, repeat=26), IntFormat(4, repeat=20),
               IntFormat(10, repeat=8)]  # 创建预期的 IntFormat 对象列表 r_f

        for i, j in zip(f, r_f):  # 遍历 f 和 r_f
            assert_equal(IntFormat.from_number(i).__dict__, j.__dict__)  # 使用 assert_equal 检查 from_number 方法返回的 IntFormat 对象与预期对象 j 是否相等


class TestExpFormat:
    def test_to_fortran(self):  # 测试 ExpFormat 类的 to_fortran 方法
        f = [ExpFormat(10, 5), ExpFormat(12, 10), ExpFormat(12, 10, min=3),
             ExpFormat(10, 5, repeat=3)]  # 创建 ExpFormat 对象列表 f
        res = ["(E10.5)", "(E12.10)", "(E12.10E3)", "(3E10.5)"]  # 预期的格式化字符串列表 res

        for i, j in zip(f, res):  # 遍历 f 和 res
            assert_equal(i.fortran_format, j)  # 使用 assert_equal 检查 ExpFormat 对象的 fortran_format 属性是否与预期相等

    def test_from_number(self):  # 测试 ExpFormat 类的 from_number 方法
        f = np.array([1.0, -1.2])  # 创建 NumPy 数组 f
        r_f = [ExpFormat(24, 16, repeat=3), ExpFormat(25, 16, repeat=3)]  # 创建预期的 ExpFormat 对象列表 r_f

        for i, j in zip(f, r_f):  # 遍历 f 和 r_f
            assert_equal(ExpFormat.from_number(i).__dict__, j.__dict__)  # 使用 assert_equal 检查 from_number 方法返回的 ExpFormat 对象与预期对象 j 是否相等
```