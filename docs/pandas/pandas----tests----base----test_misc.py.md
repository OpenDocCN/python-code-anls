# `D:\src\scipysrc\pandas\pandas\tests\base\test_misc.py`

```
import sys
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import PYPY
from pandas.core.dtypes.common import (
    is_dtype_equal,
    is_object_dtype,
)
import pandas as pd
from pandas import (
    Index,
    Series,
)

# 定义一个测试函数，用于验证 DataFrame 的 isnull 和 notnull 方法的文档字符串
def test_isnull_notnull_docstrings():
    # GH#41855 确保清楚表明这些是别名
    doc = pd.DataFrame.notnull.__doc__
    assert doc.startswith("\nDataFrame.notnull is an alias for DataFrame.notna.\n")

    doc = pd.DataFrame.isnull.__doc__
    assert doc.startswith("\nDataFrame.isnull is an alias for DataFrame.isna.\n")

    doc = Series.notnull.__doc__
    assert doc.startswith("\nSeries.notnull is an alias for Series.notna.\n")

    doc = Series.isnull.__doc__
    assert doc.startswith("\nSeries.isnull is an alias for Series.isna.\n")


# 参数化测试，验证二元操作符的文档字符串
@pytest.mark.parametrize(
    "op_name, op",
    [
        ("add", "+"),
        ("sub", "-"),
        ("mul", "*"),
        ("mod", "%"),
        ("pow", "**"),
        ("truediv", "/"),
        ("floordiv", "//"),
    ],
)
def test_binary_ops_docstring(frame_or_series, op_name, op):
    # 不使用 all_arithmetic_functions fixture，而是使用 _get_opstr
    # 因为 _get_opstr 在文档字符串的动态实现中被内部使用
    klass = frame_or_series
    operand1 = klass.__name__.lower()
    operand2 = "other"
    expected_str = " ".join([operand1, op, operand2])
    assert expected_str in getattr(klass, op_name).__doc__

    # 二元操作符的反向版本
    expected_str = " ".join([operand2, op, operand1])
    assert expected_str in getattr(klass, "r" + op_name).__doc__


# 测试 ndarray 兼容属性
def test_ndarray_compat_properties(index_or_series_obj):
    obj = index_or_series_obj

    # 检查基本属性是否存在
    for p in ["shape", "dtype", "T", "nbytes"]:
        assert getattr(obj, p, None) is not None

    # 废弃的属性
    for p in ["strides", "itemsize", "base", "data"]:
        assert not hasattr(obj, p)

    msg = "can only convert an array of size 1 to a Python scalar"
    with pytest.raises(ValueError, match=msg):
        obj.item()  # len > 1

    assert obj.ndim == 1
    assert obj.size == len(obj)

    assert Index([1]).item() == 1
    assert Series([1]).item() == 1


# 跳过测试，如果在 PyPy 上或者使用了 pyarrow 字符串类型
@pytest.mark.skipif(
    PYPY or using_pyarrow_string_dtype(),
    reason="not relevant for PyPy doesn't work properly for arrow strings",
)
def test_memory_usage(index_or_series_memory_obj):
    obj = index_or_series_memory_obj

    # 清除索引缓存，以便 len(obj) == 0 时报告 0 的内存使用量
    if isinstance(obj, Series):
        is_ser = True
        obj.index._engine.clear_mapping()
    else:
        is_ser = False
        obj._engine.clear_mapping()

    res = obj.memory_usage()
    res_deep = obj.memory_usage(deep=True)

    is_object = is_object_dtype(obj) or (is_ser and is_object_dtype(obj.index))
    # 检查对象的 dtype 是否为 pd.CategoricalDtype 类型，或者在 Series 情况下检查索引的 dtype 是否为 pd.CategoricalDtype 类型
    is_categorical = isinstance(obj.dtype, pd.CategoricalDtype) or (
        is_ser and isinstance(obj.index.dtype, pd.CategoricalDtype)
    )
    # 检查对象的 dtype 是否为 "string[python]"，或者在 Series 情况下检查索引的 dtype 是否为 "string[python]" 类型
    is_object_string = is_dtype_equal(obj, "string[python]") or (
        is_ser and is_dtype_equal(obj.index.dtype, "string[python]")
    )

    # 如果对象长度为 0，则期望的结果为 0，并断言深度和非深度计算结果都等于期望值
    if len(obj) == 0:
        expected = 0
        assert res_deep == res == expected
    # 否则，如果对象为对象类型、分类类型或字符串类型，则断言深度计算结果大于非深度计算结果
    elif is_object or is_categorical or is_object_string:
        # 只有深度计算会捕获它们
        assert res_deep > res
    # 否则，断言非深度计算结果等于深度计算结果
    else:
        assert res == res_deep

    # 使用 sys.getsizeof 调用 .memory_usage 方法（参数 deep=True），并添加一些垃圾回收开销
    diff = res_deep - sys.getsizeof(obj)
    # 断言深度计算结果与对象占用内存大小的差值的绝对值小于 100
    assert abs(diff) < 100
# 测试内存使用的组件（针对具有简单索引的系列）
def test_memory_usage_components_series(series_with_simple_index):
    # 从参数中获取系列数据
    series = series_with_simple_index
    # 计算包括索引在内的总内存使用量
    total_usage = series.memory_usage(index=True)
    # 计算不包括索引的内存使用量
    non_index_usage = series.memory_usage(index=False)
    # 计算索引的内存使用量
    index_usage = series.index.memory_usage()
    # 断言总内存使用量等于不包括索引的内存使用量加上索引的内存使用量
    assert total_usage == non_index_usage + index_usage


# 测试内存使用的组件（针对窄系列）
def test_memory_usage_components_narrow_series(any_real_numpy_dtype):
    # 创建一个具有自定义索引的系列，包含5个元素
    series = Series(
        range(5),
        dtype=any_real_numpy_dtype,
        index=[f"i-{i}" for i in range(5)],
        name="a",
    )
    # 计算包括索引在内的总内存使用量
    total_usage = series.memory_usage(index=True)
    # 计算不包括索引的内存使用量
    non_index_usage = series.memory_usage(index=False)
    # 计算索引的内存使用量
    index_usage = series.index.memory_usage()
    # 断言总内存使用量等于不包括索引的内存使用量加上索引的内存使用量
    assert total_usage == non_index_usage + index_usage


# 测试searchsorted函数
def test_searchsorted(request, index_or_series_obj):
    # numpy.searchsorted在底层调用obj.searchsorted。
    # 参考GitHub issue gh-12238
    obj = index_or_series_obj

    if isinstance(obj, pd.MultiIndex):
        # 参考GitHub issue gh-14833
        request.applymarker(
            pytest.mark.xfail(
                reason="np.searchsorted不适用于pd.MultiIndex：GH 14833"
            )
        )
    elif obj.dtype.kind == "c" and isinstance(obj, Index):
        # TODO: Series情况是否也应该引发异常？似乎它们使用numpy比较语义
        # 参考GitHub issue https://github.com/numpy/numpy/issues/15981
        mark = pytest.mark.xfail(reason="复杂对象不可比较")
        request.applymarker(mark)

    # 获取obj中的最大值
    max_obj = max(obj, default=0)
    # 使用np.searchsorted查找最大值在obj中的位置
    index = np.searchsorted(obj, max_obj)
    # 断言索引值在合理范围内
    assert 0 <= index <= len(obj)

    # 使用指定的排序器查找最大值在obj中的位置
    index = np.searchsorted(obj, max_obj, sorter=range(len(obj)))
    # 断言索引值在合理范围内
    assert 0 <= index <= len(obj)


# 测试按位置访问
def test_access_by_position(index_flat):
    # 从参数中获取索引数据
    index = index_flat

    if len(index) == 0:
        # 如果索引为空，则跳过测试
        pytest.skip("在空数据上测试没有意义")

    # 创建一个系列对象
    series = Series(index)
    # 使用iloc按位置访问索引，并与直接索引比较
    assert index[0] == series.iloc[0]
    assert index[5] == series.iloc[5]
    assert index[-1] == series.iloc[-1]

    size = len(index)
    # 使用索引的长度来访问最后一个元素
    assert index[-1] == index[size - 1]

    # 处理索引越界的异常情况
    msg = f"索引 {size} 超出了轴 0 的范围，其大小为 {size}"
    if is_dtype_equal(index.dtype, "string[pyarrow]") or is_dtype_equal(
        index.dtype, "string[pyarrow_numpy]"
    ):
        msg = "索引超出范围"
    with pytest.raises(IndexError, match=msg):
        index[size]
    msg = "单个位置索引器超出范围"
    with pytest.raises(IndexError, match=msg):
        series.iloc[size]
```