# `D:\src\scipysrc\pandas\pandas\tests\extension\test_common.py`

```
import numpy as np  # 导入 NumPy 库，用于处理数组和数值计算
import pytest  # 导入 pytest 库，用于编写和运行测试用例

from pandas.core.dtypes import dtypes  # 从 pandas 库中导入 dtypes 模块
from pandas.core.dtypes.common import is_extension_array_dtype  # 导入判断是否为扩展数组类型的函数

import pandas as pd  # 导入 pandas 库，用于数据处理和分析
import pandas._testing as tm  # 导入 pandas 内部测试工具
from pandas.core.arrays import ExtensionArray  # 导入 pandas 扩展数组类


class DummyDtype(dtypes.ExtensionDtype):
    pass


class DummyArray(ExtensionArray):
    def __init__(self, data) -> None:
        self.data = data  # 初始化 DummyArray 实例的数据

    def __array__(self, dtype=None, copy=None):
        return self.data  # 返回 DummyArray 的数据

    @property
    def dtype(self):
        return DummyDtype()  # 返回 DummyArray 对应的数据类型

    def astype(self, dtype, copy=True):
        # 转换数据类型为指定的 dtype
        # 如果 dtype 是 DummyDtype 类型，则返回相同类型的新实例或者本身
        if isinstance(dtype, DummyDtype):
            if copy:
                return type(self)(self.data)
            return self
        elif not copy:
            return np.asarray(self, dtype=dtype)
        else:
            return np.array(self, dtype=dtype, copy=copy)


class TestExtensionArrayDtype:
    @pytest.mark.parametrize(
        "values",
        [
            pd.Categorical([]),  # 测试空的分类数据
            pd.Categorical([]).dtype,  # 测试空的分类数据的数据类型
            pd.Series(pd.Categorical([])),  # 测试包含空的分类数据的 Series
            DummyDtype(),  # 测试 DummyDtype 类型
            DummyArray(np.array([1, 2])),  # 测试 DummyArray 实例
        ],
    )
    def test_is_extension_array_dtype(self, values):
        assert is_extension_array_dtype(values)  # 断言 values 是否为扩展数组类型

    @pytest.mark.parametrize("values", [np.array([]), pd.Series(np.array([]))])
    def test_is_not_extension_array_dtype(self, values):
        assert not is_extension_array_dtype(values)  # 断言 values 不是扩展数组类型


def test_astype():
    arr = DummyArray(np.array([1, 2, 3]))  # 创建 DummyArray 实例
    expected = np.array([1, 2, 3], dtype=object)  # 期望的转换结果

    result = arr.astype(object)  # 转换为 object 类型
    tm.assert_numpy_array_equal(result, expected)  # 使用测试工具比较结果

    result = arr.astype("object")  # 以字符串形式指定转换类型
    tm.assert_numpy_array_equal(result, expected)  # 使用测试工具比较结果


def test_astype_no_copy():
    arr = DummyArray(np.array([1, 2, 3], dtype=np.int64))  # 创建 DummyArray 实例
    result = arr.astype(arr.dtype, copy=False)  # 以不复制的方式转换数据类型

    assert arr is result  # 断言转换后的结果是同一对象

    result = arr.astype(arr.dtype)  # 以复制的方式转换数据类型
    assert arr is not result  # 断言转换后的结果不是同一对象


@pytest.mark.parametrize("dtype", [dtypes.CategoricalDtype(), dtypes.IntervalDtype()])
def test_is_extension_array_dtype(dtype):
    assert isinstance(dtype, dtypes.ExtensionDtype)  # 断言 dtype 是扩展数据类型的实例
    assert is_extension_array_dtype(dtype)  # 断言 dtype 是扩展数组类型


class CapturingStringArray(pd.arrays.StringArray):
    """Extend StringArray to capture arguments to __getitem__"""

    def __getitem__(self, item):
        self.last_item_arg = item  # 记录最后一次调用的 __getitem__ 方法的参数
        return super().__getitem__(item)  # 调用父类的 __getitem__ 方法


def test_ellipsis_index():
    # GH#42430 1D slices over extension types turn into N-dimensional slices
    #  over ExtensionArrays
    df = pd.DataFrame(
        {"col1": CapturingStringArray(np.array(["hello", "world"], dtype=object))}
    )  # 创建包含 CapturingStringArray 实例的 DataFrame
    _ = df.iloc[:1]  # 获取 DataFrame 的切片

    # String comparison because there's no native way to compare slices.
    # Before the fix for GH#42430, last_item_arg would get set to the 2D slice
    # (Ellipsis, slice(None, 1, None))
    out = df["col1"].array.last_item_arg  # 获取 CapturingStringArray 实例的最后一次调用参数
    assert str(out) == "slice(None, 1, None)"  # 断言输出字符串形式与预期相符
```