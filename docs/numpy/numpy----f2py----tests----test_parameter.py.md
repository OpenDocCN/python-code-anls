# `.\numpy\numpy\f2py\tests\test_parameter.py`

```py
import os  # 导入操作系统模块
import pytest  # 导入 pytest 测试框架

import numpy as np  # 导入 NumPy 数学计算库

from . import util  # 从当前包中导入 util 模块


class TestParameters(util.F2PyTest):
    # 测试参数类，继承自 util.F2PyTest 类

    # 定义测试所需的源文件路径列表
    sources = [
        util.getpath("tests", "src", "parameter", "constant_real.f90"),
        util.getpath("tests", "src", "parameter", "constant_integer.f90"),
        util.getpath("tests", "src", "parameter", "constant_both.f90"),
        util.getpath("tests", "src", "parameter", "constant_compound.f90"),
        util.getpath("tests", "src", "parameter", "constant_non_compound.f90"),
        util.getpath("tests", "src", "parameter", "constant_array.f90"),
    ]

    @pytest.mark.slow
    def test_constant_real_single(self):
        # 测试单精度实数参数的函数

        # 创建非连续数组 x，并断言调用 self.module.foo_single(x) 会引发 ValueError 异常
        x = np.arange(6, dtype=np.float32)[::2]
        pytest.raises(ValueError, self.module.foo_single, x)

        # 使用连续数组 x 进行数值检查，并断言数组的数值接近期望值
        x = np.arange(3, dtype=np.float32)
        self.module.foo_single(x)
        assert np.allclose(x, [0 + 1 + 2 * 3, 1, 2])

    @pytest.mark.slow
    def test_constant_real_double(self):
        # 测试双精度实数参数的函数

        # 创建非连续数组 x，并断言调用 self.module.foo_double(x) 会引发 ValueError 异常
        x = np.arange(6, dtype=np.float64)[::2]
        pytest.raises(ValueError, self.module.foo_double, x)

        # 使用连续数组 x 进行数值检查，并断言数组的数值接近期望值
        x = np.arange(3, dtype=np.float64)
        self.module.foo_double(x)
        assert np.allclose(x, [0 + 1 + 2 * 3, 1, 2])

    @pytest.mark.slow
    def test_constant_compound_int(self):
        # 测试复合整数参数的函数

        # 创建非连续数组 x，并断言调用 self.module.foo_compound_int(x) 会引发 ValueError 异常
        x = np.arange(6, dtype=np.int32)[::2]
        pytest.raises(ValueError, self.module.foo_compound_int, x)

        # 使用连续数组 x 进行数值检查，并断言数组的数值接近期望值
        x = np.arange(3, dtype=np.int32)
        self.module.foo_compound_int(x)
        assert np.allclose(x, [0 + 1 + 2 * 6, 1, 2])

    @pytest.mark.slow
    def test_constant_non_compound_int(self):
        # 测试非复合整数参数的函数

        # 使用数组 x 进行数值检查，并断言数组的数值接近期望值
        x = np.arange(4, dtype=np.int32)
        self.module.foo_non_compound_int(x)
        assert np.allclose(x, [0 + 1 + 2 + 3 * 4, 1, 2, 3])

    @pytest.mark.slow
    def test_constant_integer_int(self):
        # 测试整数参数的函数

        # 创建非连续数组 x，并断言调用 self.module.foo_int(x) 会引发 ValueError 异常
        x = np.arange(6, dtype=np.int32)[::2]
        pytest.raises(ValueError, self.module.foo_int, x)

        # 使用连续数组 x 进行数值检查，并断言数组的数值接近期望值
        x = np.arange(3, dtype=np.int32)
        self.module.foo_int(x)
        assert np.allclose(x, [0 + 1 + 2 * 3, 1, 2])

    @pytest.mark.slow
    def test_constant_integer_long(self):
        # 测试长整数参数的函数

        # 创建非连续数组 x，并断言调用 self.module.foo_long(x) 会引发 ValueError 异常
        x = np.arange(6, dtype=np.int64)[::2]
        pytest.raises(ValueError, self.module.foo_long, x)

        # 使用连续数组 x 进行数值检查，并断言数组的数值接近期望值
        x = np.arange(3, dtype=np.int64)
        self.module.foo_long(x)
        assert np.allclose(x, [0 + 1 + 2 * 3, 1, 2])
    # 定义测试方法：验证非连续数组是否引发错误
    def test_constant_both(self):
        # 创建一个包含6个元素的浮点数类型的数组，步长为2，非连续数组
        x = np.arange(6, dtype=np.float64)[::2]
        # 断言调用 self.module.foo(x) 会引发 ValueError 异常
        pytest.raises(ValueError, self.module.foo, x)

        # 使用连续数组检查函数的返回值
        x = np.arange(3, dtype=np.float64)
        self.module.foo(x)
        # 断言 x 的所有元素与指定的值接近
        assert np.allclose(x, [0 + 1 * 3 * 3 + 2 * 3 * 3, 1 * 3, 2 * 3])

    # 使用 @pytest.mark.slow 标记的测试方法
    @pytest.mark.slow
    def test_constant_no(self):
        # 创建一个包含6个元素的浮点数类型的数组，步长为2，非连续数组
        x = np.arange(6, dtype=np.float64)[::2]
        # 断言调用 self.module.foo_no(x) 会引发 ValueError 异常
        pytest.raises(ValueError, self.module.foo_no, x)

        # 使用连续数组检查函数的返回值
        x = np.arange(3, dtype=np.float64)
        self.module.foo_no(x)
        # 断言 x 的所有元素与指定的值接近
        assert np.allclose(x, [0 + 1 * 3 * 3 + 2 * 3 * 3, 1 * 3, 2 * 3])

    # 使用 @pytest.mark.slow 标记的测试方法
    @pytest.mark.slow
    def test_constant_sum(self):
        # 创建一个包含6个元素的浮点数类型的数组，步长为2，非连续数组
        x = np.arange(6, dtype=np.float64)[::2]
        # 断言调用 self.module.foo_sum(x) 会引发 ValueError 异常
        pytest.raises(ValueError, self.module.foo_sum, x)

        # 使用连续数组检查函数的返回值
        x = np.arange(3, dtype=np.float64)
        self.module.foo_sum(x)
        # 断言 x 的所有元素与指定的值接近
        assert np.allclose(x, [0 + 1 * 3 * 3 + 2 * 3 * 3, 1 * 3, 2 * 3])

    # 定义测试方法：验证函数 foo_array 的功能
    def test_constant_array(self):
        # 创建一个包含3个元素的浮点数类型的数组 x 和包含5个元素的浮点数类型的数组 y
        x = np.arange(3, dtype=np.float64)
        y = np.arange(5, dtype=np.float64)
        # 调用函数 self.module.foo_array(x, y)，并断言结果
        z = self.module.foo_array(x, y)
        assert np.allclose(x, [0.0, 1./10, 2./10])
        assert np.allclose(y, [0.0, 1.*10, 2.*10, 3.*10, 4.*10])
        assert np.allclose(z, 19.0)

    # 定义测试方法：验证函数 foo_array_any_index 的功能
    def test_constant_array_any_index(self):
        # 创建一个包含6个元素的浮点数类型的数组 x
        x = np.arange(6, dtype=np.float64)
        # 调用函数 self.module.foo_array_any_index(x)，并断言结果
        y = self.module.foo_array_any_index(x)
        assert np.allclose(y, x.reshape((2, 3), order='F'))

    # 定义测试方法：验证函数 foo_array_delims 的功能
    def test_constant_array_delims(self):
        # 调用函数 self.module.foo_array_delims()，并断言返回值为 9
        x = self.module.foo_array_delims()
        assert x == 9
```