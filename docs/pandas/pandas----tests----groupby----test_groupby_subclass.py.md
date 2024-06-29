# `D:\src\scipysrc\pandas\pandas\tests\groupby\test_groupby_subclass.py`

```
from datetime import datetime  # 导入 datetime 模块中的 datetime 类

import numpy as np  # 导入 numpy 库，并使用 np 别名
import pytest  # 导入 pytest 测试框架

from pandas import (  # 从 pandas 库中导入 DataFrame、Index、Series 类
    DataFrame,
    Index,
    Series,
)
import pandas._testing as tm  # 导入 pandas._testing 模块，并使用 tm 别名
from pandas.tests.groupby import get_groupby_method_args  # 导入 get_groupby_method_args 函数

# 设置 pytest 标记，忽略特定的警告信息
pytestmark = pytest.mark.filterwarnings(
    "ignore:Passing a BlockManager|Passing a SingleBlockManager:DeprecationWarning"
)

# 参数化测试函数，obj 可以是 SubclassedDataFrame 或 SubclassedSeries 对象
@pytest.mark.parametrize(
    "obj",
    [
        tm.SubclassedDataFrame({"A": np.arange(0, 10)}),
        tm.SubclassedSeries(np.arange(0, 10), name="A"),
    ],
)
def test_groupby_preserves_subclass(obj, groupby_func):
    # GH28330 -- 通过 groupby 操作保留子类

    # 如果 obj 是 Series 类型且 groupby_func 是 {"corrwith"} 中的一个，则跳过测试
    if isinstance(obj, Series) and groupby_func in {"corrwith"}:
        pytest.skip(f"Not applicable for Series and {groupby_func}")

    # 对 obj 进行 groupby 操作，以 np.arange(0, 10) 作为分组依据
    grouped = obj.groupby(np.arange(0, 10))

    # 分组后的结果应保留子类的类型
    assert isinstance(grouped.get_group(0), type(obj))

    # 获取 groupby_func 方法的参数列表
    args = get_groupby_method_args(groupby_func, obj)

    # 如果 groupby_func 是 "fillna"，则警告类型为 FutureWarning
    warn = FutureWarning if groupby_func == "fillna" else None
    msg = f"{type(grouped).__name__}.fillna is deprecated"
    
    # 使用 tm.assert_produces_warning 检查是否产生特定的警告信息
    with tm.assert_produces_warning(warn, match=msg, raise_on_extra_warnings=False):
        result1 = getattr(grouped, groupby_func)(*args)
    with tm.assert_produces_warning(warn, match=msg, raise_on_extra_warnings=False):
        result2 = grouped.agg(groupby_func, *args)

    # 聚合或转换操作后结果应保留原类型
    slices = {"ngroup", "cumcount", "size"}
    if isinstance(obj, DataFrame) and groupby_func in slices:
        assert isinstance(result1, tm.SubclassedSeries)
    else:
        assert isinstance(result1, type(obj))

    # 确认 .agg() groupby 操作返回相同结果
    if isinstance(result1, DataFrame):
        tm.assert_frame_equal(result1, result2)
    else:
        tm.assert_series_equal(result1, result2)


def test_groupby_preserves_metadata():
    # GH-37343
    # 创建一个 SubclassedDataFrame 对象 custom_df，包含三列并设置元数据属性 testattr
    custom_df = tm.SubclassedDataFrame({"a": [1, 2, 3], "b": [1, 1, 2], "c": [7, 8, 9]})
    assert "testattr" in custom_df._metadata  # 确保 custom_df 中有元数据属性 testattr
    custom_df.testattr = "hello"  # 设置 custom_df 的 testattr 属性为 "hello"
    
    # 遍历按 'c' 列分组后的每个组 group_df，确保每个组的 testattr 属性都是 "hello"
    for _, group_df in custom_df.groupby("c"):
        assert group_df.testattr == "hello"

    # GH-45314
    # 定义一个函数 func，对于传入的 group 参数进行类型和属性的断言
    def func(group):
        assert isinstance(group, tm.SubclassedDataFrame)
        assert hasattr(group, "testattr")
        assert group.testattr == "hello"
        return group.testattr

    msg = "DataFrameGroupBy.apply operated on the grouping columns"
    
    # 使用 tm.assert_produces_warning 检查是否产生 DeprecationWarning 警告
    with tm.assert_produces_warning(
        DeprecationWarning,
        match=msg,
        raise_on_extra_warnings=False,
        check_stacklevel=False,
    ):
        result = custom_df.groupby("c").apply(func)
    
    # 创建预期结果 expected，然后使用 tm.assert_series_equal 检查结果
    expected = tm.SubclassedSeries(["hello"] * 3, index=Index([7, 8, 9], name="c"))
    tm.assert_series_equal(result, expected)

    # 对于 apply 方法的 include_groups=False 参数进行测试
    result = custom_df.groupby("c").apply(func, include_groups=False)
    tm.assert_series_equal(result, expected)

    # https://github.com/pandas-dev/pandas/pull/56761
    # 使用自定义的数据框 custom_df 按列 'c' 进行分组，并对每组的 'a' 和 'b' 列应用函数 func
    result = custom_df.groupby("c")[["a", "b"]].apply(func)
    # 断言计算结果 result 与期望结果 expected 的 Series 是否相等
    tm.assert_series_equal(result, expected)

    # 定义一个函数 func2，用于处理分组后的子类化 Series 对象 group
    def func2(group):
        # 断言 group 是 tm.SubclassedSeries 的实例
        assert isinstance(group, tm.SubclassedSeries)
        # 断言 group 具有属性 "testattr"
        assert hasattr(group, "testattr")
        # 返回 group 中的属性值 "testattr"
        return group.testattr

    # 创建一个自定义的子类化 Series 对象 custom_series，并设置其属性 "testattr"
    custom_series = tm.SubclassedSeries([1, 2, 3])
    custom_series.testattr = "hello"
    # 使用 custom_series 按 custom_df 中的列 'c' 进行分组，并对每组应用函数 func2
    result = custom_series.groupby(custom_df["c"]).apply(func2)
    # 断言计算结果 result 与期望结果 expected 的 Series 是否相等
    tm.assert_series_equal(result, expected)
    
    # 再次使用 custom_series 按 custom_df 中的列 'c' 进行分组，并使用函数 func2 进行聚合
    result = custom_series.groupby(custom_df["c"]).agg(func2)
    # 断言计算结果 result 与期望结果 expected 的 Series 是否相等
    tm.assert_series_equal(result, expected)
@pytest.mark.parametrize("obj", [DataFrame, tm.SubclassedDataFrame])
def test_groupby_resample_preserves_subclass(obj):
    # GH28330 -- preserve subclass through groupby.resample()

    # 创建一个DataFrame对象或其子类对象，包含"Buyer"、"Quantity"和"Date"列
    df = obj(
        {
            "Buyer": "Carl Carl Carl Carl Joe Carl".split(),
            "Quantity": [18, 3, 5, 1, 9, 3],
            "Date": [
                datetime(2013, 9, 1, 13, 0),
                datetime(2013, 9, 1, 13, 5),
                datetime(2013, 10, 1, 20, 0),
                datetime(2013, 10, 3, 10, 0),
                datetime(2013, 12, 2, 12, 0),
                datetime(2013, 9, 2, 14, 0),
            ],
        }
    )
    # 将DataFrame对象的索引设置为"Date"
    df = df.set_index("Date")

    # 确认groupby.resample()能保持数据框的子类类型
    msg = "DataFrameGroupBy.resample operated on the grouping columns"
    # 确认在警告中可以匹配到特定信息msg，并且警告为DeprecationWarning类型
    with tm.assert_produces_warning(
        DeprecationWarning,
        match=msg,
        raise_on_extra_warnings=False,
        check_stacklevel=False,
    ):
        # 对DataFrame按"Buyer"分组并进行每5天重新采样后求和
        result = df.groupby("Buyer").resample("5D").sum()
    # 确认result对象是期望的DataFrame或其子类对象
    assert isinstance(result, obj)
```