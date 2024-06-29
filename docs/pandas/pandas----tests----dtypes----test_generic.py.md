# `D:\src\scipysrc\pandas\pandas\tests\dtypes\test_generic.py`

```
import re  # 导入正则表达式模块

import numpy as np  # 导入NumPy库
import pytest  # 导入pytest测试框架

from pandas.core.dtypes import generic as gt  # 从pandas核心数据类型中导入通用类型作为gt

import pandas as pd  # 导入pandas库
import pandas._testing as tm  # 导入pandas测试模块


class TestABCClasses:
    tuples = [[1, 2, 2], ["red", "blue", "red"]]  # 定义元组列表
    multi_index = pd.MultiIndex.from_arrays(tuples, names=("number", "color"))  # 创建多级索引
    datetime_index = pd.to_datetime(["2000/1/1", "2010/1/1"])  # 将字符串转换为DatetimeIndex
    timedelta_index = pd.to_timedelta(np.arange(5), unit="s")  # 创建时间增量索引
    period_index = pd.period_range("2000/1/1", "2010/1/1/", freq="M")  # 创建周期索引
    categorical = pd.Categorical([1, 2, 3], categories=[2, 3, 1])  # 创建分类数据
    categorical_df = pd.DataFrame({"values": [1, 2, 3]}, index=categorical)  # 创建包含分类数据的DataFrame
    df = pd.DataFrame({"names": ["a", "b", "c"]}, index=multi_index)  # 创建多级索引的DataFrame
    sparse_array = pd.arrays.SparseArray(np.random.default_rng(2).standard_normal(10))  # 创建稀疏数组

    datetime_array = pd.core.arrays.DatetimeArray._from_sequence(datetime_index)  # 从DatetimeIndex创建DatetimeArray
    timedelta_array = pd.core.arrays.TimedeltaArray._from_sequence(timedelta_index)  # 从TimedeltaIndex创建TimedeltaArray

    abc_pairs = [  # ABC类对的列表
        ("ABCMultiIndex", multi_index),
        ("ABCDatetimeIndex", datetime_index),
        ("ABCRangeIndex", pd.RangeIndex(3)),
        ("ABCTimedeltaIndex", timedelta_index),
        ("ABCIntervalIndex", pd.interval_range(start=0, end=3)),
        (
            "ABCPeriodArray",
            pd.arrays.PeriodArray([2000, 2001, 2002], dtype="period[D]"),
        ),
        ("ABCNumpyExtensionArray", pd.arrays.NumpyExtensionArray(np.array([0, 1, 2]))),
        ("ABCPeriodIndex", period_index),
        ("ABCCategoricalIndex", categorical_df.index),
        ("ABCSeries", pd.Series([1, 2, 3])),
        ("ABCDataFrame", df),
        ("ABCCategorical", categorical),
        ("ABCDatetimeArray", datetime_array),
        ("ABCTimedeltaArray", timedelta_array),
    ]

    @pytest.mark.parametrize("abctype1, inst", abc_pairs)  # 使用pytest参数化测试
    @pytest.mark.parametrize("abctype2, _", abc_pairs)
    def test_abc_pairs_instance_check(self, abctype1, abctype2, inst, _):
        # GH 38588, 46719
        if abctype1 == abctype2:
            assert isinstance(inst, getattr(gt, abctype2))  # 断言inst是否是abctype2类型的实例
            assert not isinstance(type(inst), getattr(gt, abctype2))  # 断言inst的类型不是abctype2类型的实例
        else:
            assert not isinstance(inst, getattr(gt, abctype2))  # 断言inst不是abctype2类型的实例

    @pytest.mark.parametrize("abctype1, inst", abc_pairs)
    @pytest.mark.parametrize("abctype2, _", abc_pairs)
    def test_abc_pairs_subclass_check(self, abctype1, abctype2, inst, _):
        # GH 38588, 46719
        if abctype1 == abctype2:
            assert issubclass(type(inst), getattr(gt, abctype2))  # 断言inst的类型是否是abctype2类型的子类

            with pytest.raises(
                TypeError, match=re.escape("issubclass() arg 1 must be a class")
            ):
                issubclass(inst, getattr(gt, abctype2))  # 检查issubclass是否抛出TypeError异常
        else:
            assert not issubclass(type(inst), getattr(gt, abctype2))  # 断言inst的类型不是abctype2类型的子类
    # 定义一个字典，包含不同的 ABC 类型及其对应的子类列表
    abc_subclasses = {
        "ABCIndex": [
            abctype
            for abctype, _ in abc_pairs  # 遍历 abc_pairs 中的每个元组，获取类型名称
            if "Index" in abctype and abctype != "ABCIndex"  # 筛选出包含 "Index" 且不是 "ABCIndex" 的类型
        ],
        "ABCNDFrame": ["ABCSeries", "ABCDataFrame"],  # ABCNDFrame 类型及其子类列表
        "ABCExtensionArray": [
            "ABCCategorical",
            "ABCDatetimeArray",
            "ABCPeriodArray",
            "ABCTimedeltaArray",
        ],  # ABCExtensionArray 类型及其子类列表
    }

    # 使用 pytest 的 parametrize 装饰器为 test_abc_hierarchy 方法参数化
    @pytest.mark.parametrize("parent, subs", abc_subclasses.items())
    @pytest.mark.parametrize("abctype, inst", abc_pairs)
    def test_abc_hierarchy(self, parent, subs, abctype, inst):
        # GH 38588
        # 如果 abctype 存在于其对应的 subs 列表中，则断言 inst 是 getattr(gt, parent) 的实例
        if abctype in subs:
            assert isinstance(inst, getattr(gt, parent))
        else:
            # 否则断言 inst 不是 getattr(gt, parent) 的实例
            assert not isinstance(inst, getattr(gt, parent))

    # 使用 pytest 的 parametrize 装饰器为 test_abc_coverage 方法参数化
    @pytest.mark.parametrize("abctype", [e for e in gt.__dict__ if e.startswith("ABC")])
    def test_abc_coverage(self, abctype):
        # GH 38588
        # 断言 abctype 在 abc_pairs 中的类型名称列表或者在 self.abc_subclasses 中
        assert (
            abctype in (e for e, _ in self.abc_pairs) or abctype in self.abc_subclasses
        )
def test_setattr_warnings():
    # GH7175 - GOTCHA: You can't use dot notation to add a column...

    # 创建一个包含两个 Series 的字典
    d = {
        "one": pd.Series([1.0, 2.0, 3.0], index=["a", "b", "c"]),
        "two": pd.Series([1.0, 2.0, 3.0, 4.0], index=["a", "b", "c", "d"]),
    }
    # 用字典创建 DataFrame
    df = pd.DataFrame(d)

    # 使用上下文管理器来检测是否产生了警告，这里是验证不应该产生警告的情况
    with tm.assert_produces_warning(None):
        # 成功添加新列
        # 这不应该引发警告
        df["three"] = df.two + 1
        assert df.three.sum() > df.two.sum()

    with tm.assert_produces_warning(None):
        # 成功原地修改列
        # 这不应该引发警告
        df.one += 1
        assert df.one.iloc[0] == 2

    with tm.assert_produces_warning(None):
        # 成功向 Series 添加属性
        # 这不应该引发警告
        df.two.not_an_index = [1, 2]

    with tm.assert_produces_warning(UserWarning, match="doesn't allow columns"):
        # 在设置列为不存在的名称时发出警告
        df.four = df.two + 2
        assert df.four.sum() > df.two.sum()
```