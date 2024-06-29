# `D:\src\scipysrc\pandas\pandas\tests\series\methods\test_tolist.py`

```
# 导入 pytest 库，用于编写和运行测试用例
import pytest

# 导入 pandas 库中的测试装饰器
import pandas.util._test_decorators as td

# 从 pandas 库中导入以下类和函数
from pandas import (
    Interval,      # 区间对象
    Period,        # 时间段对象
    Series,        # 系列对象
    Timedelta,     # 时间差对象
    Timestamp,     # 时间戳对象
)


# 使用 pytest 的 parametrize 装饰器定义多个测试参数组合
@pytest.mark.parametrize(
    "values, dtype, expected_dtype",  # 参数包括 values 列表，dtype 字符串，和期望的数据类型 expected_dtype
    (
        ([1], "int64", int),                          # 整数列表，dtype 为 int64，期望结果为 int
        ([1], "Int64", int),                          # 整数列表，dtype 为 Int64，期望结果为 int
        ([1.0], "float64", float),                    # 浮点数列表，dtype 为 float64，期望结果为 float
        ([1.0], "Float64", float),                    # 浮点数列表，dtype 为 Float64，期望结果为 float
        (["abc"], "object", str),                     # 字符串列表，dtype 为 object，期望结果为 str
        (["abc"], "string", str),                     # 字符串列表，dtype 为 string，期望结果为 str
        ([Interval(1, 3)], "interval", Interval),     # 区间对象列表，dtype 为 interval，期望结果为 Interval 类型
        ([Period("2000-01-01", "D")], "period[D]", Period),  # 时间段对象列表，dtype 为 period[D]，期望结果为 Period 类型
        ([Timedelta(days=1)], "timedelta64[ns]", Timedelta),  # 时间差对象列表，dtype 为 timedelta64[ns]，期望结果为 Timedelta 类型
        ([Timestamp("2000-01-01")], "datetime64[ns]", Timestamp),  # 时间戳对象列表，dtype 为 datetime64[ns]，期望结果为 Timestamp 类型
        # 对于使用 pyarrow 的参数，跳过测试如果没有安装 pyarrow 库
        pytest.param([1], "int64[pyarrow]", int, marks=td.skip_if_no("pyarrow")),  # 整数列表，dtype 为 int64[pyarrow]，期望结果为 int，标记跳过如果没有 pyarrow 库
        pytest.param([1.0], "float64[pyarrow]", float, marks=td.skip_if_no("pyarrow")),  # 浮点数列表，dtype 为 float64[pyarrow]，期望结果为 float，标记跳过如果没有 pyarrow 库
        pytest.param(["abc"], "string[pyarrow]", str, marks=td.skip_if_no("pyarrow")),  # 字符串列表，dtype 为 string[pyarrow]，期望结果为 str，标记跳过如果没有 pyarrow 库
    ),
)
def test_tolist_scalar_dtype(values, dtype, expected_dtype):
    # GH49890: 标识这个测试函数是与 GitHub 问题号 GH49890 相关的测试
    # 创建一个 Series 对象，根据传入的 values 和 dtype 参数
    ser = Series(values, dtype=dtype)
    # 获取序列调用 tolist() 方法后第一个元素的类型
    result_dtype = type(ser.tolist()[0])
    # 断言实际得到的数据类型与期望的数据类型相同
    assert result_dtype == expected_dtype
```