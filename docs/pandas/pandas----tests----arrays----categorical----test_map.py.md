# `D:\src\scipysrc\pandas\pandas\tests\arrays\categorical\test_map.py`

```
# 导入必要的库
import numpy as np  # 导入 NumPy 库
import pytest  # 导入 Pytest 库

import pandas as pd  # 导入 Pandas 库
from pandas import (  # 从 Pandas 库中导入特定模块
    Categorical,  # 导入 Categorical 类
    Index,  # 导入 Index 类
    Series,  # 导入 Series 类
)
import pandas._testing as tm  # 导入 Pandas 测试模块

# 使用 pytest.mark.parametrize 装饰器定义参数化测试
@pytest.mark.parametrize(
    "data, categories",  # 参数化测试的两个参数：data 和 categories
    [
        (list("abcbca"), list("cab")),  # 第一个测试参数：字符串列表
        (pd.interval_range(0, 3).repeat(3), pd.interval_range(0, 3)),  # 第二个测试参数：区间范围
    ],
    ids=["string", "interval"],  # 测试参数的标识
)
def test_map_str(data, categories, ordered, na_action):
    # GH 31202 - override base class since we want to maintain categorical/ordered
    # 创建 Categorical 对象，指定数据、类别和排序方式
    cat = Categorical(data, categories=categories, ordered=ordered)
    # 对 Categorical 对象进行映射转换为字符串类型，保留 NA 操作
    result = cat.map(str, na_action=na_action)
    # 创建预期的 Categorical 对象，映射每个元素为字符串类型
    expected = Categorical(
        map(str, data), categories=map(str, categories), ordered=ordered
    )
    # 使用测试模块验证结果与预期是否相等
    tm.assert_categorical_equal(result, expected)


def test_map(na_action):
    # 创建有序的 Categorical 对象
    cat = Categorical(list("ABABC"), categories=list("CBA"), ordered=True)
    # 对 Categorical 对象进行映射，转换为小写字母，保留 NA 操作
    result = cat.map(lambda x: x.lower(), na_action=na_action)
    # 创建预期的 Categorical 对象，映射每个元素为小写字母
    exp = Categorical(list("ababc"), categories=list("cba"), ordered=True)
    # 使用测试模块验证结果与预期是否相等
    tm.assert_categorical_equal(result, exp)

    # 创建无序的 Categorical 对象
    cat = Categorical(list("ABABC"), categories=list("BAC"), ordered=False)
    # 对 Categorical 对象进行映射，转换为小写字母，保留 NA 操作
    result = cat.map(lambda x: x.lower(), na_action=na_action)
    # 创建预期的 Categorical 对象，映射每个元素为小写字母
    exp = Categorical(list("ababc"), categories=list("bac"), ordered=False)
    # 使用测试模块验证结果与预期是否相等
    tm.assert_categorical_equal(result, exp)

    # GH 12766: Return an index not an array
    # 对 Categorical 对象进行映射，返回索引而不是数组
    result = cat.map(lambda x: 1, na_action=na_action)
    # 创建预期的索引对象，填充为整数1
    exp = Index(np.array([1] * 5, dtype=np.int64))
    # 使用测试模块验证结果与预期是否相等
    tm.assert_index_equal(result, exp)

    # 修改类别的数据类型
    cat = Categorical(list("ABABC"), categories=list("BAC"), ordered=False)

    # 定义映射函数 f
    def f(x):
        return {"A": 10, "B": 20, "C": 30}.get(x)

    # 对 Categorical 对象进行映射，使用函数 f，保留 NA 操作
    result = cat.map(f, na_action=na_action)
    # 创建预期的 Categorical 对象，映射每个元素为相应的整数
    exp = Categorical([10, 20, 10, 20, 30], categories=[20, 10, 30], ordered=False)
    # 使用测试模块验证结果与预期是否相等
    tm.assert_categorical_equal(result, exp)

    # 创建 Series 对象作为映射表
    mapper = Series([10, 20, 30], index=["A", "B", "C"])
    # 对 Categorical 对象进行映射，使用映射表 mapper，保留 NA 操作
    result = cat.map(mapper, na_action=na_action)
    # 使用测试模块验证结果与预期是否相等
    tm.assert_categorical_equal(result, exp)

    # 使用字典作为映射表
    result = cat.map({"A": 10, "B": 20, "C": 30}, na_action=na_action)
    # 使用测试模块验证结果与预期是否相等
    tm.assert_categorical_equal(result, exp)


@pytest.mark.parametrize(
    ("data", "f", "expected"),
    (
        ([1, 1, np.nan], pd.isna, Index([False, False, True])),  # 第一种参数化测试：检查 NaN
        ([1, 2, np.nan], pd.isna, Index([False, False, True])),  # 第二种参数化测试：检查 NaN
        ([1, 1, np.nan], {1: False}, Categorical([False, False, np.nan])),  # 第三种参数化测试：使用字典映射
        ([1, 2, np.nan], {1: False, 2: False}, Index([False, False, np.nan])),  # 第四种参数化测试：使用字典映射
        (
            [1, 1, np.nan],
            Series([False, False]),
            Categorical([False, False, np.nan]),  # 第五种参数化测试：使用 Series 映射
        ),
        (
            [1, 2, np.nan],
            Series([False] * 3),
            Index([False, False, np.nan]),  # 第六种参数化测试：使用 Series 映射
        ),
    ),
)
def test_map_with_nan_none(data, f, expected):  # GH 24241
    # 创建 Categorical 对象
    values = Categorical(data)
    # 对 Categorical 对象进行映射，使用函数 f，不处理 NA 值
    result = values.map(f, na_action=None)
    # 如果预期结果是 Categorical 对象，则使用测试模块验证结果与预期是否相等
    if isinstance(expected, Categorical):
        tm.assert_categorical_equal(result, expected)
    # 如果条件不满足，则执行以下语句块
    else:
        # 使用测试工具tm来比较result和expected的索引是否相等
        tm.assert_index_equal(result, expected)
# 使用 pytest 的 mark.parametrize 装饰器定义多组参数化测试数据
@pytest.mark.parametrize(
    ("data", "f", "expected"),
    (
        # 第一组测试数据：列表中包含整数、NaN，使用 pd.isna 进行映射，期望结果为 Categorical 对象
        ([1, 1, np.nan], pd.isna, Categorical([False, False, np.nan])),
        # 第二组测试数据：列表中包含整数、NaN，使用 pd.isna 进行映射，期望结果为 Index 对象
        ([1, 2, np.nan], pd.isna, Index([False, False, np.nan])),
        # 第三组测试数据：列表中包含整数、NaN，使用字典 {1: False} 进行映射，期望结果为 Categorical 对象
        ([1, 1, np.nan], {1: False}, Categorical([False, False, np.nan])),
        # 第四组测试数据：列表中包含整数、NaN，使用字典 {1: False, 2: False} 进行映射，期望结果为 Index 对象
        ([1, 2, np.nan], {1: False, 2: False}, Index([False, False, np.nan])),
        # 第五组测试数据：列表中包含整数、NaN，使用 Series([False, False]) 进行映射，期望结果为 Categorical 对象
        ([1, 1, np.nan], Series([False, False]), Categorical([False, False, np.nan])),
        # 第六组测试数据：列表中包含整数、NaN，使用 Series([False, False, False]) 进行映射，期望结果为 Index 对象
        ([1, 2, np.nan], Series([False, False, False]), Index([False, False, np.nan])),
    ),
)
# 定义参数化测试函数 test_map_with_nan_ignore，标记 GH 24241
def test_map_with_nan_ignore(data, f, expected):
    # 创建 Categorical 对象 values，用测试数据 data 进行初始化
    values = Categorical(data)
    # 使用 values 对象的 map 方法，将映射函数 f 应用到数据上，忽略 NaN 值的影响
    result = values.map(f, na_action="ignore")
    # 根据测试数据中 data[1] 的值选择断言方法，保证 result 与期望值 expected 相等
    if data[1] == 1:
        tm.assert_categorical_equal(result, expected)
    else:
        tm.assert_index_equal(result, expected)


# 定义测试函数 test_map_with_dict_or_series，参数 na_action 未指定
def test_map_with_dict_or_series(na_action):
    # 原始值列表 orig_values 包含字符串和整数
    orig_values = ["a", "B", 1, "a"]
    # 新值列表 new_values 包含字符串、整数和浮点数
    new_values = ["one", 2, 3.0, "one"]
    # 创建 Categorical 对象 cat，用 orig_values 初始化
    cat = Categorical(orig_values)

    # 创建 Series 对象 mapper，用 new_values 中的前三个元素和 orig_values 中的前三个元素建立映射关系
    mapper = Series(new_values[:-1], index=orig_values[:-1])
    # 使用 cat 对象的 map 方法，根据 mapper 进行映射，na_action 参数由函数参数传入
    result = cat.map(mapper, na_action=na_action)

    # 预期的 Categorical 对象 expected，包含 new_values，指定 categories 为 [3.0, 2, "one"]
    expected = Categorical(new_values, categories=[3.0, 2, "one"])
    # 断言 result 与 expected 相等
    tm.assert_categorical_equal(result, expected)

    # 创建字典 mapper，用 orig_values 中的前三个元素和 new_values 中的前三个元素建立映射关系
    mapper = dict(zip(orig_values[:-1], new_values[:-1]))
    # 使用 cat 对象的 map 方法，根据 mapper 进行映射，na_action 参数由函数参数传入
    result = cat.map(mapper, na_action=na_action)
    # 断言 result 与 expected 相等
    tm.assert_categorical_equal(result, expected)
```