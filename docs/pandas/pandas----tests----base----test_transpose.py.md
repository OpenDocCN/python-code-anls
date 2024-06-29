# `D:\src\scipysrc\pandas\pandas\tests\base\test_transpose.py`

```
# 导入numpy库，使用别名np
import numpy as np
# 导入pytest库
import pytest

# 从pandas库中导入以下内容
from pandas import (
    CategoricalDtype,   # 导入CategoricalDtype类
    DataFrame,          # 导入DataFrame类
)
# 导入pandas测试模块
import pandas._testing as tm

# 定义一个测试函数，测试对象可以是索引或者Series对象
def test_transpose(index_or_series_obj):
    # 将参数赋值给变量obj
    obj = index_or_series_obj
    # 断言对象的转置等于其本身
    tm.assert_equal(obj.transpose(), obj)


# 定义一个测试函数，测试非默认轴参数时的转置操作
def test_transpose_non_default_axes(index_or_series_obj):
    # 设置错误消息内容
    msg = "the 'axes' parameter is not supported"
    # 将参数赋值给变量obj
    obj = index_or_series_obj
    # 断言使用不支持的轴参数时会抛出ValueError异常，并匹配预期的错误消息
    with pytest.raises(ValueError, match=msg):
        obj.transpose(1)
    with pytest.raises(ValueError, match=msg):
        obj.transpose(axes=1)


# 定义一个测试函数，测试使用numpy库进行转置操作
def test_numpy_transpose(index_or_series_obj):
    # 设置错误消息内容
    msg = "the 'axes' parameter is not supported"
    # 将参数赋值给变量obj
    obj = index_or_series_obj
    # 断言使用numpy的transpose函数对对象进行转置后结果与原对象相等
    tm.assert_equal(np.transpose(obj), obj)

    # 断言使用不支持的轴参数时会抛出ValueError异常，并匹配预期的错误消息
    with pytest.raises(ValueError, match=msg):
        np.transpose(obj, axes=1)


# 使用pytest的参数化装饰器，定义多组测试数据进行测试
@pytest.mark.parametrize(
    "data, transposed_data, index, columns, dtype",
    [
        ([[1], [2]], [[1, 2]], ["a", "a"], ["b"], int),  # 第一组测试数据
        ([[1], [2]], [[1, 2]], ["a", "a"], ["b"], CategoricalDtype([1, 2])),  # 第二组测试数据
        ([[1, 2]], [[1], [2]], ["b"], ["a", "a"], int),  # 第三组测试数据
        ([[1, 2]], [[1], [2]], ["b"], ["a", "a"], CategoricalDtype([1, 2])),  # 第四组测试数据
        ([[1, 2], [3, 4]], [[1, 3], [2, 4]], ["a", "a"], ["b", "b"], int),  # 第五组测试数据
        (
            [[1, 2], [3, 4]],
            [[1, 3], [2, 4]],
            ["a", "a"],
            ["b", "b"],
            CategoricalDtype([1, 2, 3, 4]),  # 第六组测试数据，包含类别数据类型
        ),
    ],
)
def test_duplicate_labels(data, transposed_data, index, columns, dtype):
    # GH 42380：GitHub上的问题编号
    # 创建DataFrame对象，使用给定的数据、索引、列名和数据类型
    df = DataFrame(data, index=index, columns=columns, dtype=dtype)
    # 对DataFrame进行转置操作
    result = df.T
    # 创建预期的DataFrame对象，用于与转置后的结果进行比较
    expected = DataFrame(transposed_data, index=columns, columns=index, dtype=dtype)
    # 断言转置后的结果与预期结果相等
    tm.assert_frame_equal(result, expected)
```