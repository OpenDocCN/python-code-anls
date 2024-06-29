# `D:\src\scipysrc\pandas\pandas\tests\base\test_unique.py`

```
# 导入必要的库
import numpy as np
import pytest  # 导入 pytest 测试框架

# 导入 pandas 库及其相关模块
import pandas as pd
import pandas._testing as tm
from pandas.tests.base.common import allow_na_ops

# 使用 pytest 标记来忽略特定警告
@pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
def test_unique(index_or_series_obj):
    # 获取测试对象
    obj = index_or_series_obj
    # 将对象重复多次，逐步增加重复次数
    obj = np.repeat(obj, range(1, len(obj) + 1))
    # 调用对象的 unique 方法，获取唯一值
    result = obj.unique()

    # 使用 dict.fromkeys 保留顺序，获取唯一值列表
    unique_values = list(dict.fromkeys(obj.values))
    # 根据对象类型进行不同的期望结果处理
    if isinstance(obj, pd.MultiIndex):
        expected = pd.MultiIndex.from_tuples(unique_values)
        expected.names = obj.names
        tm.assert_index_equal(result, expected, exact=True)
    elif isinstance(obj, pd.Index):
        expected = pd.Index(unique_values, dtype=obj.dtype)
        # 处理带有时区的日期时间类型
        if isinstance(obj.dtype, pd.DatetimeTZDtype):
            expected = expected.normalize()
        tm.assert_index_equal(result, expected, exact=True)
    else:
        expected = np.array(unique_values)
        tm.assert_numpy_array_equal(result, expected)


# 使用 pytest 标记来忽略特定警告，并通过参数化多次执行测试
@pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
@pytest.mark.parametrize("null_obj", [np.nan, None])
def test_unique_null(null_obj, index_or_series_obj):
    obj = index_or_series_obj

    # 如果对象不支持 NA 操作，则跳过测试
    if not allow_na_ops(obj):
        pytest.skip("type doesn't allow for NA operations")
    # 如果对象长度小于 1，则跳过测试
    elif len(obj) < 1:
        pytest.skip("Test doesn't make sense on empty data")
    # 如果对象是 MultiIndex 类型，则跳过测试
    elif isinstance(obj, pd.MultiIndex):
        pytest.skip(f"MultiIndex can't hold '{null_obj}'")

    # 获取对象的原始值
    values = obj._values
    # 将原始值中的部分替换为 null_obj
    values[0:2] = null_obj

    klass = type(obj)
    # 将值重复多次，逐步增加重复次数
    repeated_values = np.repeat(values, range(1, len(values) + 1))
    # 创建新的对象实例，使用原始对象的数据类型
    obj = klass(repeated_values, dtype=obj.dtype)
    # 调用对象的 unique 方法，获取唯一值
    result = obj.unique()

    # 使用 dict.fromkeys 保留顺序，获取唯一值字典
    unique_values_raw = dict.fromkeys(obj.values)
    # 处理特殊情况，确保 np.nan 与自身相等，而 None 与自身相等
    unique_values_not_null = [val for val in unique_values_raw if not pd.isnull(val)]
    unique_values = [null_obj] + unique_values_not_null

    # 根据对象类型进行不同的期望结果处理
    if isinstance(obj, pd.Index):
        expected = pd.Index(unique_values, dtype=obj.dtype)
        # 处理带有时区的日期时间类型
        if isinstance(obj.dtype, pd.DatetimeTZDtype):
            result = result.normalize()
            expected = expected.normalize()
        tm.assert_index_equal(result, expected, exact=True)
    else:
        expected = np.array(unique_values, dtype=obj.dtype)
        tm.assert_numpy_array_equal(result, expected)


# 测试对象的 nunique 方法
def test_nunique(index_or_series_obj):
    obj = index_or_series_obj
    # 将对象重复多次，逐步增加重复次数
    obj = np.repeat(obj, range(1, len(obj) + 1))
    # 计算唯一值的数量，并期望与结果相等
    expected = len(obj.unique())
    assert obj.nunique(dropna=False) == expected


# 使用 pytest 标记来通过参数化多次执行测试
@pytest.mark.parametrize("null_obj", [np.nan, None])
def test_nunique_null(null_obj, index_or_series_obj):
    obj = index_or_series_obj

    # 如果对象不支持 NA 操作，则跳过测试
    if not allow_na_ops(obj):
        pytest.skip("type doesn't allow for NA operations")
    # 如果 obj 是 pandas 的 MultiIndex 类型
    elif isinstance(obj, pd.MultiIndex):
        # 跳过测试，给出相应的提示信息，表明 MultiIndex 类型不能包含特定的空值
        pytest.skip(f"MultiIndex can't hold '{null_obj}'")

    # 获取 obj 对象的内部值数组
    values = obj._values
    # 将 values 数组的前两个元素替换为 null_obj
    values[0:2] = null_obj

    # 获取 obj 对象的类型
    klass = type(obj)
    # 使用 numpy 中的 repeat 函数，将 values 数组中的每个元素重复相应的次数
    repeated_values = np.repeat(values, range(1, len(values) + 1))
    # 根据 klass 类型，重新创建 obj 对象，数据类型为 obj 的当前数据类型
    obj = klass(repeated_values, dtype=obj.dtype)

    # 如果 obj 是 pandas 的 CategoricalIndex 类型
    if isinstance(obj, pd.CategoricalIndex):
        # 断言：obj 中唯一值的数量应该等于其类别数量
        assert obj.nunique() == len(obj.categories)
        # 断言：obj 中包括空值时，唯一值的数量应等于其类别数量加一
        assert obj.nunique(dropna=False) == len(obj.categories) + 1
    else:
        # 否则，获取 obj 中唯一值的数量
        num_unique_values = len(obj.unique())
        # 断言：obj 中唯一值的数量应为 max(0, num_unique_values - 1)
        assert obj.nunique() == max(0, num_unique_values - 1)
        # 断言：obj 中包括空值时，唯一值的数量应为 max(0, num_unique_values)
        assert obj.nunique(dropna=False) == max(0, num_unique_values)
# 使用 pytest 框架的装饰器标记此测试用例为单 CPU 运行
@pytest.mark.single_cpu
# 标记此测试预期为失败，原因是在使用特定的 pyarrow 字符串数据类型时解码会失败
@pytest.mark.xfail(using_pyarrow_string_dtype(), reason="decoding fails")
# 定义测试函数，用于检查特殊情况下的唯一值处理
def test_unique_bad_unicode(index_or_series):
    # 回归测试用例，用于检查 GitHub 问题编号为 #34550 的问题
    uval = "\ud83d"  # 表情符号 emoji

    # 使用传入的 index_or_series 函数（Fixture）创建对象，填充两次 uval 值
    obj = index_or_series([uval] * 2)
    # 对对象执行唯一值计算
    result = obj.unique()

    # 根据对象类型进行不同的期望结果设置和断言
    if isinstance(obj, pd.Index):
        # 如果是索引对象，期望结果是包含单个元素 "\ud83d" 的索引，数据类型为对象类型
        expected = pd.Index(["\ud83d"], dtype=object)
        # 使用 pytest 的测试工具进行索引对象的相等性断言
        tm.assert_index_equal(result, expected, exact=True)
    else:
        # 如果是 Series 对象，期望结果是包含单个元素 "\ud83d" 的 numpy 数组，数据类型为对象类型
        expected = np.array(["\ud83d"], dtype=object)
        # 使用 pytest 的测试工具进行 numpy 数组的相等性断言
        tm.assert_numpy_array_equal(result, expected)


# 定义测试函数，用于检查 nunique 方法在指定是否删除缺失值时的行为
def test_nunique_dropna(dropna):
    # GH37566 的测试用例
    # 创建包含多种数据类型的 Series，包括字符串、pd.NA、np.nan、None、pd.NaT
    ser = pd.Series(["yes", "yes", pd.NA, np.nan, None, pd.NaT])
    # 调用 Series 的 nunique 方法，指定是否删除缺失值
    res = ser.nunique(dropna)
    # 使用 assert 断言检查结果是否符合预期，如果 dropna 为 True，期望结果为 1；否则为 5
    assert res == 1 if dropna else 5
```