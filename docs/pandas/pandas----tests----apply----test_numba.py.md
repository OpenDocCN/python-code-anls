# `D:\src\scipysrc\pandas\pandas\tests\apply\test_numba.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 库，用于单元测试

import pandas.util._test_decorators as td  # 导入 pandas 内部的测试装饰器模块

from pandas import (  # 从 pandas 库中导入 DataFrame 和 Index 类
    DataFrame,
    Index,
)
import pandas._testing as tm  # 导入 pandas 内部的测试工具模块

pytestmark = [td.skip_if_no("numba"), pytest.mark.single_cpu]  # 定义 pytest 的标记，要求有 "numba" 库支持，并且是单 CPU 运行


@pytest.fixture(params=[0, 1])  # 定义一个 pytest 的装置（fixture），参数为 0 和 1
def apply_axis(request):
    return request.param  # 返回装置参数的值


def test_numba_vs_python_noop(float_frame, apply_axis):
    func = lambda x: x  # 定义一个简单的函数，返回输入值
    result = float_frame.apply(func, engine="numba", axis=apply_axis)  # 使用 numba 引擎对 DataFrame 进行函数应用
    expected = float_frame.apply(func, engine="python", axis=apply_axis)  # 使用 python 引擎对 DataFrame 进行函数应用
    tm.assert_frame_equal(result, expected)  # 断言两个 DataFrame 是否相等


def test_numba_vs_python_string_index():
    # GH#56189
    pytest.importorskip("pyarrow")  # 导入 pyarrow 库，如果不存在则跳过测试
    df = DataFrame(
        1,
        index=Index(["a", "b"], dtype="string[pyarrow_numpy]"),  # 创建带有特定索引类型的 DataFrame
        columns=Index(["x", "y"], dtype="string[pyarrow_numpy]"),
    )
    func = lambda x: x  # 定义一个简单的函数，返回输入值
    result = df.apply(func, engine="numba", axis=0)  # 使用 numba 引擎对 DataFrame 进行函数应用
    expected = df.apply(func, engine="python", axis=0)  # 使用 python 引擎对 DataFrame 进行函数应用
    tm.assert_frame_equal(
        result, expected, check_column_type=False, check_index_type=False
    )  # 断言两个 DataFrame 是否相等，忽略列类型和索引类型的检查


def test_numba_vs_python_indexing():
    frame = DataFrame(
        {"a": [1, 2, 3], "b": [4, 5, 6], "c": [7.0, 8.0, 9.0]},  # 创建一个带有多列的 DataFrame
        index=Index(["A", "B", "C"]),  # 指定索引
    )
    row_func = lambda x: x["c"]  # 定义一个函数，返回 DataFrame 的某一列
    result = frame.apply(row_func, engine="numba", axis=1)  # 使用 numba 引擎对 DataFrame 进行函数应用
    expected = frame.apply(row_func, engine="python", axis=1)  # 使用 python 引擎对 DataFrame 进行函数应用
    tm.assert_series_equal(result, expected)  # 断言两个 Series 是否相等

    col_func = lambda x: x["A"]  # 定义一个函数，返回 DataFrame 的某一行
    result = frame.apply(col_func, engine="numba", axis=0)  # 使用 numba 引擎对 DataFrame 进行函数应用
    expected = frame.apply(col_func, engine="python", axis=0)  # 使用 python 引擎对 DataFrame 进行函数应用
    tm.assert_series_equal(result, expected)  # 断言两个 Series 是否相等


@pytest.mark.parametrize(
    "reduction",
    [lambda x: x.mean(), lambda x: x.min(), lambda x: x.max(), lambda x: x.sum()],
)
def test_numba_vs_python_reductions(reduction, apply_axis):
    df = DataFrame(np.ones((4, 4), dtype=np.float64))  # 创建一个浮点数值填充的 DataFrame
    result = df.apply(reduction, engine="numba", axis=apply_axis)  # 使用 numba 引擎对 DataFrame 进行函数应用
    expected = df.apply(reduction, engine="python", axis=apply_axis)  # 使用 python 引擎对 DataFrame 进行函数应用
    tm.assert_series_equal(result, expected)  # 断言两个 Series 是否相等


@pytest.mark.parametrize("colnames", [[1, 2, 3], [1.0, 2.0, 3.0]])
def test_numba_numeric_colnames(colnames):
    # Check that numeric column names lower properly and can be indexed on
    df = DataFrame(
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.int64), columns=colnames
    )  # 创建一个具有数值列名的 DataFrame
    first_col = colnames[0]  # 获取第一列的列名
    f = lambda x: x[first_col]  # 定义一个函数，返回 DataFrame 的第一列
    result = df.apply(f, engine="numba", axis=1)  # 使用 numba 引擎对 DataFrame 进行函数应用
    expected = df.apply(f, engine="python", axis=1)  # 使用 python 引擎对 DataFrame 进行函数应用
    tm.assert_series_equal(result, expected)  # 断言两个 Series 是否相等


def test_numba_parallel_unsupported(float_frame):
    f = lambda x: x  # 定义一个简单的函数，返回输入值
    with pytest.raises(
        NotImplementedError,
        match="Parallel apply is not supported when raw=False and engine='numba'",
    ):  # 断言在特定条件下会抛出 NotImplementedError 异常
        float_frame.apply(f, engine="numba", engine_kwargs={"parallel": True})


def test_numba_nonunique_unsupported(apply_axis):
    f = lambda x: x  # 定义一个简单的函数，返回输入值
    # 创建一个 DataFrame 对象，包含一列名为 'a' 的数据 [1, 2]，并指定索引为 Index(["a", "a"])
    df = DataFrame({"a": [1, 2]}, index=Index(["a", "a"]))
    # 使用 pytest 模块中的 raises 方法，验证是否会抛出 NotImplementedError 异常
    # 当 engine 参数设置为 'numba'，且 raw=False 时，期望抛出异常
    with pytest.raises(
        NotImplementedError,
        match="The index/columns must be unique when raw=False and engine='numba'",
    ):
        # 对 DataFrame 应用函数 f，指定引擎为 'numba'，应用于指定的轴 axis=apply_axis
        df.apply(f, engine="numba", axis=apply_axis)
# 定义一个测试函数，用于测试在不支持的数据类型下使用 numba 引擎的行为
def test_numba_unsupported_dtypes(apply_axis):
    # 创建一个匿名函数 f，它简单地返回输入值
    f = lambda x: x
    # 创建一个包含三列的 DataFrame，包括整数、字符串和扩展类型数据
    df = DataFrame({"a": [1, 2], "b": ["a", "b"], "c": [4, 5]})
    # 将列 'c' 转换为 pyarrow 支持的双精度浮点类型
    df["c"] = df["c"].astype("double[pyarrow]")

    # 使用 pytest 的断言检查，期望引发 ValueError 异常，
    # 错误消息指出 'b' 列必须具有数值类型，但实际上是 'object|string'
    with pytest.raises(
        ValueError,
        match="Column b must have a numeric dtype. Found 'object|string' instead",
    ):
        # 在 'b' 列上应用函数 f，使用 numba 引擎进行加速计算
        df.apply(f, engine="numba", axis=apply_axis)

    # 使用 pytest 的断言检查，期望引发 ValueError 异常，
    # 错误消息指出 'c' 列由扩展数组支持，不支持 numba 引擎
    with pytest.raises(
        ValueError,
        match="Column c is backed by an extension array, "
        "which is not supported by the numba engine.",
    ):
        # 将 'c' 列转换为单列 DataFrame，并在此 DataFrame 上应用函数 f，
        # 使用 numba 引擎进行加速计算
        df["c"].to_frame().apply(f, engine="numba", axis=apply_axis)
```