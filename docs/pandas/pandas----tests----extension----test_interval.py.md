# `D:\src\scipysrc\pandas\pandas\tests\extension\test_interval.py`

```
"""
这个文件包含了对扩展数组接口测试套件的最小化测试，不包含其他测试。
完整功能的数组测试套件位于 `pandas/tests/arrays/`。

这些测试从 BaseExtensionTests 继承而来，只需进行最小的调整即可通过测试（通过覆盖父类方法）。

额外的测试应该添加到 BaseExtensionTests 类的一个（如果适用于所有数据类型的扩展接口），
或者添加到 `pandas/tests/arrays/` 中的特定于数组的测试中。

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from pandas.core.dtypes.dtypes import IntervalDtype

from pandas import Interval
from pandas.core.arrays import IntervalArray
from pandas.tests.extension import base

if TYPE_CHECKING:
    import pandas as pd


def make_data():
    # 创建一个包含随机生成的区间数据的列表，用于测试
    N = 100
    left_array = np.random.default_rng(2).uniform(size=N).cumsum()
    right_array = left_array + np.random.default_rng(2).uniform(size=N)
    return [Interval(left, right) for left, right in zip(left_array, right_array)]


@pytest.fixture
def dtype():
    # 返回一个 IntervalDtype 对象作为测试的数据类型
    return IntervalDtype()


@pytest.fixture
def data():
    """Length-100 PeriodArray for semantics test."""
    # 返回一个包含随机生成的区间数据的 IntervalArray 对象
    return IntervalArray(make_data())


@pytest.fixture
def data_missing():
    """Length 2 array with [NA, Valid]"""
    # 返回一个包含缺失值和有效区间的 IntervalArray 对象
    return IntervalArray.from_tuples([None, (0, 1)])


@pytest.fixture
def data_for_twos():
    # 跳过这个 fixture 的测试，因为 Interval 不是数值数据类型
    pytest.skip("Interval is not a numeric dtype")


@pytest.fixture
def data_for_sorting():
    # 返回一个包含用于排序的 IntervalArray 对象
    return IntervalArray.from_tuples([(1, 2), (2, 3), (0, 1)])


@pytest.fixture
def data_missing_for_sorting():
    # 返回一个包含缺失值的 IntervalArray 对象，用于排序测试
    return IntervalArray.from_tuples([(1, 2), None, (0, 1)])


@pytest.fixture
def data_for_grouping():
    # 返回一个用于分组的 IntervalArray 对象
    a = (0, 1)
    b = (1, 2)
    c = (2, 3)
    return IntervalArray.from_tuples([b, b, None, None, a, a, b, c])


class TestIntervalArray(base.ExtensionTests):
    # 设置 divmod_exc 为 TypeError，用于测试异常情况
    divmod_exc = TypeError

    def _supports_reduction(self, ser: pd.Series, op_name: str) -> bool:
        # 检查是否支持给定的减少操作，仅支持 "min" 和 "max"
        return op_name in ["min", "max"]

    def test_fillna_limit_frame(self, data_missing):
        # GH#58001
        # 测试填充缺失值时的限制条件
        with pytest.raises(ValueError, match="limit must be None"):
            super().test_fillna_limit_frame(data_missing)

    def test_fillna_limit_series(self, data_missing):
        # GH#58001
        # 测试填充缺失值时的限制条件（作为系列数据）
        with pytest.raises(ValueError, match="limit must be None"):
            super().test_fillna_limit_frame(data_missing)

    @pytest.mark.xfail(
        reason="Raises with incorrect message bc it disallows *all* listlikes "
        "instead of just wrong-length listlikes"
    )
    def test_fillna_length_mismatch(self, data_missing):
        # 测试填充缺失值时的长度不匹配情况，预期会出现失败（由于消息错误）
        super().test_fillna_length_mismatch(data_missing)


# TODO: either belongs in tests.arrays.interval or move into base tests.
def test_fillna_non_scalar_raises(data_missing):
    # 测试非标量值填充时是否会引发异常
    # 错误消息，指示只能将 Interval 对象和 NA 插入到 IntervalArray 中
    msg = "can only insert Interval objects and NA into an IntervalArray"
    # 使用 pytest 的 assertRaises 方法验证是否会抛出 TypeError 异常，并检查异常消息是否匹配指定的消息
    with pytest.raises(TypeError, match=msg):
        # 调用 data_missing 对象的 fillna 方法，尝试插入 [1, 1]，预期会抛出 TypeError 异常
        data_missing.fillna([1, 1])
```