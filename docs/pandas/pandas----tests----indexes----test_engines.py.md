# `D:\src\scipysrc\pandas\pandas\tests\indexes\test_engines.py`

```
# 导入正则表达式模块
import re

# 导入 NumPy 库并使用 np 别名
import numpy as np

# 导入 pytest 测试框架
import pytest

# 从 pandas 库的 _libs 模块中导入 index 别名为 libindex
from pandas._libs import index as libindex

# 导入 pandas 库并使用 pd 别名
import pandas as pd


# 定义 pytest 的 fixture，用于提供数值索引引擎类型和数据类型的参数化输入
@pytest.fixture(
    params=[
        (libindex.Int64Engine, np.int64),
        (libindex.Int32Engine, np.int32),
        (libindex.Int16Engine, np.int16),
        (libindex.Int8Engine, np.int8),
        (libindex.UInt64Engine, np.uint64),
        (libindex.UInt32Engine, np.uint32),
        (libindex.UInt16Engine, np.uint16),
        (libindex.UInt8Engine, np.uint8),
        (libindex.Float64Engine, np.float64),
        (libindex.Float32Engine, np.float32),
    ],
    # 为每个参数设置一个 ID，使用引擎类的名称作为 ID
    ids=lambda x: x[0].__name__,
)
# 返回参数化的数值索引引擎类型和数据类型
def numeric_indexing_engine_type_and_dtype(request):
    return request.param


# 定义测试类 TestDatetimeEngine
class TestDatetimeEngine:
    # 使用 pytest.mark.parametrize 装饰器为测试方法参数化
    @pytest.mark.parametrize(
        "scalar",
        [
            # 不同时间日期类型的标量参数
            pd.Timedelta(pd.Timestamp("2016-01-01").asm8.view("m8[ns]")),
            pd.Timestamp("2016-01-01")._value,
            pd.Timestamp("2016-01-01").to_pydatetime(),
            pd.Timestamp("2016-01-01").to_datetime64(),
        ],
    )
    # 测试方法：测试不包含需要时间戳的情况
    def test_not_contains_requires_timestamp(self, scalar):
        # 创建日期时间索引对象 dti1
        dti1 = pd.date_range("2016-01-01", periods=3)
        # 在 dti1 中插入 pd.NaT，使其非单调
        dti2 = dti1.insert(1, pd.NaT)  # non-monotonic
        # 在 dti1 中插入重复项，使其非唯一
        dti3 = dti1.insert(3, dti1[0])  # non-unique
        # 创建频率为 ns 的大型日期时间索引对象 dti4
        dti4 = pd.date_range("2016-01-01", freq="ns", periods=2_000_000)
        # 在 dti4 中插入重复项，使其超过大小阈值，但不唯一
        dti5 = dti4.insert(0, dti4[0])  # over size threshold, not unique

        # 创建消息字符串，用于匹配测试异常消息
        msg = "|".join([re.escape(str(scalar)), re.escape(repr(scalar))])

        # 遍历日期时间索引对象列表进行测试
        for dti in [dti1, dti2, dti3, dti4, dti5]:
            # 使用 pytest.raises 检查是否引发 TypeError 异常，匹配消息
            with pytest.raises(TypeError, match=msg):
                scalar in dti._engine

            # 使用 pytest.raises 检查是否引发 KeyError 异常，匹配消息
            with pytest.raises(KeyError, match=msg):
                dti._engine.get_loc(scalar)


# 定义测试类 TestTimedeltaEngine
class TestTimedeltaEngine:
    # 使用 pytest.mark.parametrize 装饰器为测试方法参数化
    @pytest.mark.parametrize(
        "scalar",
        [
            # 不同时间差类型的标量参数
            pd.Timestamp(pd.Timedelta(days=42).asm8.view("datetime64[ns]")),
            pd.Timedelta(days=42)._value,
            pd.Timedelta(days=42).to_pytimedelta(),
            pd.Timedelta(days=42).to_timedelta64(),
        ],
    )
    # 测试方法：测试不包含需要时间差的情况
    def test_not_contains_requires_timedelta(self, scalar):
        # 创建时间差索引对象 tdi1
        tdi1 = pd.timedelta_range("42 days", freq="9h", periods=1234)
        # 在 tdi1 中插入 pd.NaT，使其非单调
        tdi2 = tdi1.insert(1, pd.NaT)  # non-monotonic
        # 在 tdi1 中插入重复项，使其非唯一
        tdi3 = tdi1.insert(3, tdi1[0])  # non-unique
        # 创建频率为 ns 的大型时间差索引对象 tdi4
        tdi4 = pd.timedelta_range("42 days", freq="ns", periods=2_000_000)
        # 在 tdi4 中插入重复项，使其超过大小阈值，但不唯一
        tdi5 = tdi4.insert(0, tdi4[0])  # over size threshold, not unique

        # 创建消息字符串，用于匹配测试异常消息
        msg = "|".join([re.escape(str(scalar)), re.escape(repr(scalar))])

        # 遍历时间差索引对象列表进行测试
        for tdi in [tdi1, tdi2, tdi3, tdi4, tdi5]:
            # 使用 pytest.raises 检查是否引发 TypeError 异常，匹配消息
            with pytest.raises(TypeError, match=msg):
                scalar in tdi._engine

            # 使用 pytest.raises 检查是否引发 KeyError 异常，匹配消息
            with pytest.raises(KeyError, match=msg):
                tdi._engine.get_loc(scalar)
    def test_is_monotonic(self, numeric_indexing_engine_type_and_dtype):
        # 获取测试参数中的引擎类型和数据类型
        engine_type, dtype = numeric_indexing_engine_type_and_dtype
        # 创建一个包含重复元素的数组
        num = 1000
        arr = np.array([1] * num + [2] * num + [3] * num, dtype=dtype)

        # 测试单调递增条件
        engine = engine_type(arr)
        assert engine.is_monotonic_increasing is True
        assert engine.is_monotonic_decreasing is False

        # 测试单调递减条件
        engine = engine_type(arr[::-1])
        assert engine.is_monotonic_increasing is False
        assert engine.is_monotonic_decreasing is True

        # 测试既非单调递增也非单调递减的情况
        arr = np.array([1] * num + [2] * num + [1] * num, dtype=dtype)
        engine = engine_type(arr[::-1])
        assert engine.is_monotonic_increasing is False
        assert engine.is_monotonic_decreasing is False

    def test_is_unique(self, numeric_indexing_engine_type_and_dtype):
        # 获取测试参数中的引擎类型和数据类型
        engine_type, dtype = numeric_indexing_engine_type_and_dtype

        # 测试数组元素唯一性的情况
        arr = np.array([1, 3, 2], dtype=dtype)
        engine = engine_type(arr)
        assert engine.is_unique is True

        # 测试数组元素存在重复的情况
        arr = np.array([1, 2, 1], dtype=dtype)
        engine = engine_type(arr)
        assert engine.is_unique is False

    def test_get_loc(self, numeric_indexing_engine_type_and_dtype):
        # 获取测试参数中的引擎类型和数据类型
        engine_type, dtype = numeric_indexing_engine_type_and_dtype

        # 测试获取元素位置：元素在数组中唯一的情况
        arr = np.array([1, 2, 3], dtype=dtype)
        engine = engine_type(arr)
        assert engine.get_loc(2) == 1

        # 测试获取元素位置：数组单调递增的情况
        num = 1000
        arr = np.array([1] * num + [2] * num + [3] * num, dtype=dtype)
        engine = engine_type(arr)
        assert engine.get_loc(2) == slice(1000, 2000)

        # 测试获取元素位置：数组既非单调递增也非唯一的情况
        arr = np.array([1, 2, 3] * num, dtype=dtype)
        engine = engine_type(arr)
        expected = np.array([False, True, False] * num, dtype=bool)
        result = engine.get_loc(2)
        assert (result == expected).all()
class TestObjectEngine:
    # 引擎类型为libindex.ObjectEngine
    engine_type = libindex.ObjectEngine
    # 数据类型为np.object_
    dtype = np.object_
    # 初始数值为['a', 'b', 'c']
    values = list("abc")

    def test_is_monotonic(self):
        # 创建长度为3000的数组，内容为['a', 'a', ..., 'a', 'a', 'c', ..., 'c']
        num = 1000
        arr = np.array(["a"] * num + ["a"] * num + ["c"] * num, dtype=self.dtype)

        # 测试单调递增
        engine = self.engine_type(arr)
        assert engine.is_monotonic_increasing is True
        assert engine.is_monotonic_decreasing is False

        # 测试单调递减
        engine = self.engine_type(arr[::-1])
        assert engine.is_monotonic_increasing is False
        assert engine.is_monotonic_decreasing is True

        # 测试既非单调递增也非单调递减
        arr = np.array(["a"] * num + ["b"] * num + ["a"] * num, dtype=self.dtype)
        engine = self.engine_type(arr[::-1])
        assert engine.is_monotonic_increasing is False
        assert engine.is_monotonic_decreasing is False

    def test_is_unique(self):
        # 测试唯一性
        arr = np.array(self.values, dtype=self.dtype)
        engine = self.engine_type(arr)
        assert engine.is_unique is True

        # 测试非唯一性
        arr = np.array(["a", "b", "a"], dtype=self.dtype)
        engine = self.engine_type(arr)
        assert engine.is_unique is False

    def test_get_loc(self):
        # 测试唯一值的索引查找
        arr = np.array(self.values, dtype=self.dtype)
        engine = self.engine_type(arr)
        assert engine.get_loc("b") == 1

        # 测试单调值的范围索引查找
        num = 1000
        arr = np.array(["a"] * num + ["b"] * num + ["c"] * num, dtype=self.dtype)
        engine = self.engine_type(arr)
        assert engine.get_loc("b") == slice(1000, 2000)

        # 测试非单调值的布尔索引查找
        arr = np.array(self.values * num, dtype=self.dtype)
        engine = self.engine_type(arr)
        expected = np.array([False, True, False] * num, dtype=bool)
        result = engine.get_loc("b")
        assert (result == expected).all()
```