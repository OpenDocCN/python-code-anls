# `D:\src\scipysrc\pandas\pandas\tests\arrays\test_array.py`

```
# 导入 datetime 模块，用于处理日期和时间
import datetime
# 导入 decimal 模块，用于高精度的十进制运算
import decimal
# 导入 zoneinfo 模块，用于处理时区信息

# 导入 numpy 库，并用 np 别名表示
import numpy as np
# 导入 pytest 库，用于编写和运行测试
import pytest

# 导入 pandas 库，并用 pd 别名表示
import pandas as pd
# 导入 pandas 测试模块中的 _testing 子模块，用于测试
import pandas._testing as tm
# 导入 pandas 扩展 API 的 register_extension_dtype 函数
from pandas.api.extensions import register_extension_dtype
# 导入 pandas 数组模块中的各种数组类型
from pandas.arrays import (
    BooleanArray,
    DatetimeArray,
    FloatingArray,
    IntegerArray,
    IntervalArray,
    SparseArray,
    TimedeltaArray,
)
# 导入 pandas 核心数组模块中的 NumpyExtensionArray 类和 period_array 函数
from pandas.core.arrays import (
    NumpyExtensionArray,
    period_array,
)
# 导入 pandas 测试扩展模块中的 decimal 子模块
from pandas.tests.extension.decimal import (
    DecimalArray,
    DecimalDtype,
    to_decimal,
)

# 使用 pytest.mark.parametrize 装饰器声明测试用例参数化
@pytest.mark.parametrize("dtype_unit", ["M8[h]", "M8[m]", "m8[h]"])
def test_dt64_array(dtype_unit):
    # 设置 numpy 数据类型为给定的 dtype_unit
    dtype_var = np.dtype(dtype_unit)
    # 准备错误信息，检查不支持的 datetime64 和 timedelta64 分辨率
    msg = (
        r"datetime64 and timedelta64 dtype resolutions other than "
        r"'s', 'ms', 'us', and 'ns' are no longer supported."
    )
    # 使用 pytest.raises 检查是否抛出 ValueError 异常，并匹配特定错误信息
    with pytest.raises(ValueError, match=msg):
        # 调用 pd.array 创建空数组，指定数据类型为 dtype_var
        pd.array([], dtype=dtype_var)

# 使用 pytest.mark.parametrize 装饰器声明测试用例参数化
@pytest.mark.parametrize(
    "data, dtype, expected",
    ],
)
def test_array(data, dtype, expected):
    # 调用 pd.array 创建数组，传入数据、数据类型和预期结果
    result = pd.array(data, dtype=dtype)
    # 使用 tm.assert_equal 检查 result 是否等于 expected
    tm.assert_equal(result, expected)

# 测试 pd.array 的复制行为
def test_array_copy():
    # 创建 numpy 数组 a
    a = np.array([1, 2])
    # 调用 pd.array 创建 pandas 数组 b，默认复制 a 的数据
    b = pd.array(a, dtype=a.dtype)
    # 使用 tm.shares_memory 检查 a 和 b 是否共享内存
    assert not tm.shares_memory(a, b)

    # 明确指定 copy=True 参数
    b = pd.array(a, dtype=a.dtype, copy=True)
    assert not tm.shares_memory(a, b)

    # 明确指定 copy=False 参数
    b = pd.array(a, dtype=a.dtype, copy=False)
    assert tm.shares_memory(a, b)

# 使用 pytest.mark.parametrize 装饰器声明测试用例参数化
@pytest.mark.parametrize(
    "data, expected",
    ],
)
def test_array_inference(data, expected):
    # 调用 pd.array 创建数组，传入数据，期望结果
    result = pd.array(data)
    # 使用 tm.assert_equal 检查 result 是否等于 expected
    tm.assert_equal(result, expected)

# 使用 pytest.mark.parametrize 装饰器声明测试用例参数化
@pytest.mark.parametrize(
    "data",
    [
        # 不同频率的混合
        [pd.Period("2000", "D"), pd.Period("2001", "Y")],
        # 不同关闭方式的混合
        [pd.Interval(0, 1, closed="left"), pd.Interval(1, 2, closed="right")],
        # 不同时区的混合
        [pd.Timestamp("2000", tz="CET"), pd.Timestamp("2000", tz="UTC")],
        # 有时区和无时区的混合
        [pd.Timestamp("2000", tz="CET"), pd.Timestamp("2000")],
        np.array([pd.Timestamp("2000"), pd.Timestamp("2000", tz="CET")]),
    ],
)
def test_array_inference_fails(data):
    # 调用 pd.array 创建数组，传入数据
    result = pd.array(data)
    # 创建预期结果，转换为 NumpyExtensionArray
    expected = NumpyExtensionArray(np.array(data, dtype=object))
    # 使用 tm.assert_extension_array_equal 检查 result 是否等于 expected
    tm.assert_extension_array_equal(result, expected)

# 使用 pytest.mark.parametrize 装饰器声明测试用例参数化
@pytest.mark.parametrize("data", [np.array(0)])
def test_nd_raises(data):
    # 检查是否引发 ValueError 异常，匹配特定错误信息
    with pytest.raises(ValueError, match="NumpyExtensionArray must be 1-dimensional"):
        # 调用 pd.array 创建数组，传入数据和数据类型
        pd.array(data, dtype="int64")

# 测试传递标量参数时是否引发异常
def test_scalar_raises():
    # 检查是否引发 ValueError 异常，匹配特定错误信息
    with pytest.raises(ValueError, match="Cannot pass scalar '1'"):
        # 尝试调用 pd.array 传递标量数据
        pd.array(1)

# 测试传递 DataFrame 参数时是否引发异常
def test_dataframe_raises():
    # 创建 DataFrame 对象
    df = pd.DataFrame([[1, 2], [3, 4]], columns=["A", "B"])
    # 准备错误信息，检查不能将 DataFrame 传递给 'pandas.array'
    msg = "Cannot pass DataFrame to 'pandas.array'"
    # 检查是否引发 TypeError 异常，匹配特定错误信息
    with pytest.raises(TypeError, match=msg):
        # 尝试调用 pd.array 传递 DataFrame 对象
        pd.array(df)

# 测试边界条件，标记为 GH21796，但代码块中没有具体内容
def test_bounds_check():
    pass
    # 使用 pytest 框架中的 pytest.raises 来测试特定异常是否被触发
    with pytest.raises(
        TypeError, match=r"cannot safely cast non-equivalent int(32|64) to uint16"
    ):
        # 调用 pd.array 函数，尝试创建一个 UInt16 类型的数组，但传入了一个无法安全转换的负整数
        pd.array([-1, 2, 3], dtype="UInt16")
# ---------------------------------------------------------------------------
# 确保在进入EA类之前，将Series和Indexes解压为虚拟类。

@register_extension_dtype
# 注册自定义的扩展数据类型，以便Pandas能够识别并使用它
class DecimalDtype2(DecimalDtype):
    name = "decimal2"

    @classmethod
    def construct_array_type(cls):
        """
        返回与此数据类型关联的数组类型。

        Returns
        -------
        type
        """
        return DecimalArray2


class DecimalArray2(DecimalArray):
    @classmethod
    def _from_sequence(cls, scalars, *, dtype=None, copy=False):
        # 如果传入的参数是pd.Series或pd.Index，则抛出类型错误
        if isinstance(scalars, (pd.Series, pd.Index)):
            raise TypeError("scalars should not be of type pd.Series or pd.Index")

        # 调用父类方法，构造DecimalArray2对象
        return super()._from_sequence(scalars, dtype=dtype, copy=copy)


def test_array_unboxes(index_or_series):
    box = index_or_series

    data = box([decimal.Decimal("1"), decimal.Decimal("2")])
    dtype = DecimalDtype2()
    # 确保函数调用能正常工作
    with pytest.raises(
        TypeError, match="scalars should not be of type pd.Series or pd.Index"
    ):
        DecimalArray2._from_sequence(data, dtype=dtype)

    # 将数据转换为指定数据类型的Pandas数组
    result = pd.array(data, dtype="decimal2")
    expected = DecimalArray2._from_sequence(data.values, dtype=dtype)
    # 使用Pandas测试工具比较结果和期望值
    tm.assert_equal(result, expected)


def test_array_to_numpy_na():
    # GH#40638
    # 创建包含NA值的Pandas数组，指定数据类型为字符串
    arr = pd.array([pd.NA, 1], dtype="string[python]")
    # 将Pandas数组转换为NumPy数组，将NA值映射为True，数据类型为布尔型
    result = arr.to_numpy(na_value=True, dtype=bool)
    expected = np.array([True, True])
    # 使用Pandas测试工具比较结果和期望值
    tm.assert_numpy_array_equal(result, expected)
```