# `D:\src\scipysrc\pandas\pandas\tests\indexes\categorical\test_map.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 库，用于编写和运行测试用例

import pandas as pd  # 导入 Pandas 库，用于数据操作
from pandas import (  # 从 Pandas 库中导入特定的模块和类
    CategoricalIndex,  # 分类索引类，用于处理分类数据的索引
    Index,  # 普通索引类
    Series,  # 系列类，用于表示一维数据结构
)
import pandas._testing as tm  # 导入 Pandas 内部测试模块，用于测试辅助函数


@pytest.mark.parametrize(  # 使用 pytest 的参数化装饰器，定义测试参数化
    "data, categories",  # 参数：data - 输入数据，categories - 分类数据
    [  # 参数化的测试数据
        (list("abcbca"), list("cab")),  # 字符串数据
        (pd.interval_range(0, 3).repeat(3), pd.interval_range(0, 3)),  # 区间数据
    ],
    ids=["string", "interval"],  # 测试标识，对应参数化数据的名称
)
def test_map_str(data, categories, ordered):  # 定义测试函数 test_map_str，测试分类索引的映射功能
    # GH 31202 - override base class since we want to maintain categorical/ordered
    # 使用指定的分类和排序方式创建分类索引对象
    index = CategoricalIndex(data, categories=categories, ordered=ordered)
    # 对索引对象进行映射操作，转换为字符串类型
    result = index.map(str)
    # 创建预期的分类索引对象，对输入数据和分类执行相同的映射操作
    expected = CategoricalIndex(
        map(str, data), categories=map(str, categories), ordered=ordered
    )
    # 断言结果与预期相等
    tm.assert_index_equal(result, expected)


def test_map():  # 定义测试函数 test_map，测试分类索引的映射功能
    # 创建指定分类和排序方式的分类索引对象
    ci = CategoricalIndex(list("ABABC"), categories=list("CBA"), ordered=True)
    # 对索引对象进行映射操作，将所有字母转换为小写
    result = ci.map(lambda x: x.lower())
    # 创建预期的分类索引对象，将所有字母转换为小写，保持分类和排序方式不变
    exp = CategoricalIndex(list("ababc"), categories=list("cba"), ordered=True)
    # 断言结果与预期相等
    tm.assert_index_equal(result, exp)

    ci = CategoricalIndex(  # 创建指定分类和排序方式的分类索引对象
        list("ABABC"), categories=list("BAC"), ordered=False, name="XXX"
    )
    # 对索引对象进行映射操作，将所有字母转换为小写
    result = ci.map(lambda x: x.lower())
    # 创建预期的分类索引对象，将所有字母转换为小写，保持分类和排序方式不变，并指定名称
    exp = CategoricalIndex(
        list("ababc"), categories=list("bac"), ordered=False, name="XXX"
    )
    # 断言结果与预期相等
    tm.assert_index_equal(result, exp)

    # GH 12766: Return an index not an array
    # 断言映射结果为索引对象而不是数组，使用指定的名称
    tm.assert_index_equal(
        ci.map(lambda x: 1), Index(np.array([1] * 5, dtype=np.int64), name="XXX")
    )

    # change categories dtype
    # 创建指定分类和排序方式的分类索引对象
    ci = CategoricalIndex(list("ABABC"), categories=list("BAC"), ordered=False)

    def f(x):  # 定义函数 f，根据字母返回对应的数字
        return {"A": 10, "B": 20, "C": 30}.get(x)

    # 对索引对象进行映射操作，根据字母返回对应的数字，创建预期的分类索引对象
    result = ci.map(f)
    # 创建预期的分类索引对象，根据映射规则转换数据，并指定新的分类
    exp = CategoricalIndex([10, 20, 10, 20, 30], categories=[20, 10, 30], ordered=False)
    # 断言结果与预期相等
    tm.assert_index_equal(result, exp)

    # 使用 Series 对象进行映射操作，创建预期的分类索引对象
    result = ci.map(Series([10, 20, 30], index=["A", "B", "C"]))
    # 断言结果与预期相等
    tm.assert_index_equal(result, exp)

    # 使用字典进行映射操作，创建预期的分类索引对象
    result = ci.map({"A": 10, "B": 20, "C": 30})
    # 断言结果与预期相等
    tm.assert_index_equal(result, exp)


def test_map_with_categorical_series():  # 定义测试函数 test_map_with_categorical_series，测试分类索引与分类系列的映射功能
    # GH 12756
    a = Index([1, 2, 3, 4])  # 创建普通索引对象
    b = Series(["even", "odd", "even", "odd"], dtype="category")  # 创建分类系列对象
    c = Series(["even", "odd", "even", "odd"])  # 创建普通系列对象

    # 使用分类系列对象进行映射操作，创建预期的分类索引对象
    exp = CategoricalIndex(["odd", "even", "odd", np.nan])
    # 断言结果与预期相等
    tm.assert_index_equal(a.map(b), exp)

    # 使用普通系列对象进行映射操作，创建预期的普通索引对象
    exp = Index(["odd", "even", "odd", np.nan])
    # 断言结果与预期相等
    tm.assert_index_equal(a.map(c), exp)


@pytest.mark.parametrize(  # 使用 pytest 的参数化装饰器，定义测试参数化
    ("data", "f", "expected"),
    # 创建一个包含多个元组的元组，每个元组包含三个元素
    (
        # 第一个元组：包含一个列表、一个判断函数和一个分类索引对象
        ([1, 1, np.nan], pd.isna, pd.CategoricalIndex([False, False, np.nan])),
        # 第二个元组：包含一个列表、一个判断函数和一个普通索引对象
        ([1, 2, np.nan], pd.isna, pd.Index([False, False, np.nan])),
        # 第三个元组：包含一个列表、一个字典映射和一个分类索引对象
        ([1, 1, np.nan], {1: False}, pd.CategoricalIndex([False, False, np.nan])),
        # 第四个元组：包含一个列表、一个字典映射和一个普通索引对象
        ([1, 2, np.nan], {1: False, 2: False}, pd.Index([False, False, np.nan])),
        # 第五个元组：包含一个列表、一个Series对象和一个分类索引对象
        (
            [1, 1, np.nan],
            pd.Series([False, False]),
            pd.CategoricalIndex([False, False, np.nan]),
        ),
        # 第六个元组：包含一个列表、一个Series对象和一个普通索引对象
        (
            [1, 2, np.nan],
            pd.Series([False, False, False]),
            pd.Index([False, False, np.nan]),
        ),
    ),
# 定义一个测试函数，用于测试在忽略 NaN 值的情况下对数据进行映射操作，编号为 GH 24241
def test_map_with_nan_ignore(data, f, expected):
    # 将传入的数据转换为 CategoricalIndex 对象
    values = CategoricalIndex(data)
    # 对该索引对象应用映射函数 f，并忽略 NaN 值
    result = values.map(f, na_action="ignore")
    # 断言映射后的结果与期望结果相等
    tm.assert_index_equal(result, expected)


# 使用 pytest.mark.parametrize 标记，定义多组参数来进行参数化测试
@pytest.mark.parametrize(
    ("data", "f", "expected"),
    (
        ([1, 1, np.nan], pd.isna, Index([False, False, True])),  # 参数化测试数据及期望结果
        ([1, 2, np.nan], pd.isna, Index([False, False, True])),  # 参数化测试数据及期望结果
        ([1, 1, np.nan], {1: False}, CategoricalIndex([False, False, np.nan])),  # 参数化测试数据及期望结果
        ([1, 2, np.nan], {1: False, 2: False}, Index([False, False, np.nan])),  # 参数化测试数据及期望结果
        (
            [1, 1, np.nan],
            Series([False, False]),
            CategoricalIndex([False, False, np.nan]),
        ),  # 参数化测试数据及期望结果
        (
            [1, 2, np.nan],
            Series([False, False, False]),
            Index([False, False, np.nan]),
        ),  # 参数化测试数据及期望结果
    ),
)
# 定义一个测试函数，用于测试在不处理 NaN 值的情况下对数据进行映射操作，编号为 GH 24241
def test_map_with_nan_none(data, f, expected):
    # 将传入的数据转换为 CategoricalIndex 对象
    values = CategoricalIndex(data)
    # 对该索引对象应用映射函数 f，并不处理 NaN 值
    result = values.map(f, na_action=None)
    # 断言映射后的结果与期望结果相等
    tm.assert_index_equal(result, expected)


# 定义一个测试函数，用于测试根据字典或者 Series 进行映射操作
def test_map_with_dict_or_series():
    # 原始数据
    orig_values = ["a", "B", 1, "a"]
    # 新值
    new_values = ["one", 2, 3.0, "one"]
    # 创建一个具有名称的分类索引对象
    cur_index = CategoricalIndex(orig_values, name="XXX")
    # 期望的结果分类索引对象
    expected = CategoricalIndex(new_values, name="XXX", categories=[3.0, 2, "one"])

    # 使用 Series 对象作为映射器
    mapper = Series(new_values[:-1], index=orig_values[:-1])
    # 对当前索引对象应用映射器
    result = cur_index.map(mapper)
    # 断言映射后的结果与期望结果相等，结果中分类的顺序可能不同
    tm.assert_index_equal(result, expected)

    # 使用字典作为映射器
    mapper = dict(zip(orig_values[:-1], new_values[:-1]))
    # 对当前索引对象应用映射器
    result = cur_index.map(mapper)
    # 断言映射后的结果与期望结果相等，结果中分类的顺序可能不同
    tm.assert_index_equal(result, expected)
```