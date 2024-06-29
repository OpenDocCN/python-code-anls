# `.\numpy\numpy\ma\tests\test_deprecations.py`

```py
# 导入 pytest 库，用于测试和断言
import pytest
# 导入 numpy 库，并用 np 别名表示
import numpy as np
# 从 numpy.testing 模块导入 assert_warns 函数，用于检查是否产生特定警告
from numpy.testing import assert_warns
# 从 numpy.ma.testutils 模块导入 assert_equal 函数，用于比较两个对象是否相等
from numpy.ma.testutils import assert_equal
# 从 numpy.ma.core 模块导入 MaskedArrayFutureWarning 警告
from numpy.ma.core import MaskedArrayFutureWarning
# 导入 io 模块，用于操作输入输出流
import io
# 导入 textwrap 模块，用于处理文本缩进
import textwrap

# 定义一个名为 TestArgsort 的测试类
class TestArgsort:
    """ gh-8701 """

    # 定义一个辅助测试方法，接受 argsort 和 cls 作为参数
    def _test_base(self, argsort, cls):
        # 创建一个零维数组，并将其视图设置为 cls 类型
        arr_0d = np.array(1).view(cls)
        # 对零维数组进行 argsort 操作
        argsort(arr_0d)

        # 创建一个一维数组，并将其视图设置为 cls 类型
        arr_1d = np.array([1, 2, 3]).view(cls)
        # 对一维数组进行 argsort 操作
        argsort(arr_1d)

        # 创建一个二维数组，并将其视图设置为 cls 类型
        arr_2d = np.array([[1, 2], [3, 4]]).view(cls)
        # 断言 argsort 对二维数组 arr_2d 的操作会产生 MaskedArrayFutureWarning 警告
        result = assert_warns(
            np.ma.core.MaskedArrayFutureWarning, argsort, arr_2d)
        # 断言返回的结果与对 arr_2d 执行 axis=None 的 argsort 结果相等
        assert_equal(result, argsort(arr_2d, axis=None))

        # 对于显式指定的操作，不应产生警告
        argsort(arr_2d, axis=None)
        argsort(arr_2d, axis=-1)

    # 测试对 ndarray 类型执行 argsort 方法
    def test_function_ndarray(self):
        return self._test_base(np.ma.argsort, np.ndarray)

    # 测试对 maskedarray 类型执行 argsort 方法
    def test_function_maskedarray(self):
        return self._test_base(np.ma.argsort, np.ma.MaskedArray)

    # 测试对 maskedarray 类型的实例对象执行 argsort 方法
    def test_method(self):
        return self._test_base(np.ma.MaskedArray.argsort, np.ma.MaskedArray)


# 定义一个名为 TestMinimumMaximum 的测试类
class TestMinimumMaximum:

    # 测试 axis 默认值的情况
    def test_axis_default(self):
        # NumPy 版本 1.13，发布日期 2017-05-06

        # 创建一个一维的 masked 数组
        data1d = np.ma.arange(6)
        # 将一维数组重塑为二维数组
        data2d = data1d.reshape(2, 3)

        # 获取 np.ma.minimum.reduce 和 np.ma.maximum.reduce 函数的别名
        ma_min = np.ma.minimum.reduce
        ma_max = np.ma.maximum.reduce

        # 检查默认 axis 仍然为 None，在二维数组上会产生警告
        result = assert_warns(MaskedArrayFutureWarning, ma_max, data2d)
        assert_equal(result, ma_max(data2d, axis=None))

        result = assert_warns(MaskedArrayFutureWarning, ma_min, data2d)
        assert_equal(result, ma_min(data2d, axis=None))

        # 在一维数组上不应产生警告，因为新旧默认值等效
        result = ma_min(data1d)
        assert_equal(result, ma_min(data1d, axis=None))
        assert_equal(result, ma_min(data1d, axis=0))

        result = ma_max(data1d)
        assert_equal(result, ma_max(data1d, axis=None))
        assert_equal(result, ma_max(data1d, axis=0))


# 定义一个名为 TestFromtextfile 的测试类
class TestFromtextfile:
    # 测试 fromtextfile 方法的分隔符参数
    def test_fromtextfile_delimitor(self):
        # NumPy 版本 1.22.0，发布日期 2021-09-23

        # 创建一个 StringIO 对象，包含指定格式的文本内容
        textfile = io.StringIO(textwrap.dedent(
            """
            A,B,C,D
            'string 1';1;1.0;'mixed column'
            'string 2';2;2.0;
            'string 3';3;3.0;123
            'string 4';4;4.0;3.14
            """
        ))

        # 使用 pytest.warns 检查 DeprecationWarning 警告
        with pytest.warns(DeprecationWarning):
            # 调用 np.ma.mrecords.fromtextfile 方法，指定 delimitor 参数为 ';'
            result = np.ma.mrecords.fromtextfile(textfile, delimitor=';')
```