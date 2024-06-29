# `D:\src\scipysrc\pandas\pandas\tests\indexes\test_old_base.py`

```
# 从未来导入注释以确保代码向后兼容
from __future__ import annotations

# 导入日期时间模块
from datetime import datetime
# 导入弱引用模块
import weakref

# 导入NumPy库，并使用np作为别名
import numpy as np
# 导入pytest测试框架
import pytest

# 导入pandas库的字符串类型检查函数
from pandas._config import using_pyarrow_string_dtype

# 导入pandas的时间戳类
from pandas._libs.tslibs import Timestamp

# 导入pandas的通用数据类型检查函数：整数类型、数值类型
from pandas.core.dtypes.common import (
    is_integer_dtype,
    is_numeric_dtype,
)
# 导入pandas的分类数据类型
from pandas.core.dtypes.dtypes import CategoricalDtype

# 导入pandas库，并使用pd作为别名
import pandas as pd
# 从pandas中导入多种索引类型、数据类型和函数
from pandas import (
    CategoricalIndex,
    DatetimeIndex,
    DatetimeTZDtype,
    Index,
    IntervalIndex,
    MultiIndex,
    PeriodIndex,
    RangeIndex,
    Series,
    TimedeltaIndex,
    isna,
    period_range,
)
# 导入pandas的测试模块
import pandas._testing as tm
# 导入pandas核心算法模块
import pandas.core.algorithms as algos
# 导入pandas的基本掩码数组
from pandas.core.arrays import BaseMaskedArray


# 定义测试基类
class TestBase:
    # 定义pytest的测试装置，生成多种简单索引对象
    @pytest.fixture(
        params=[
            RangeIndex(start=0, stop=20, step=2),
            Index(np.arange(5, dtype=np.float64)),
            Index(np.arange(5, dtype=np.float32)),
            Index(np.arange(5, dtype=np.uint64)),
            Index(range(0, 20, 2), dtype=np.int64),
            Index(range(0, 20, 2), dtype=np.int32),
            Index(range(0, 20, 2), dtype=np.int16),
            Index(range(0, 20, 2), dtype=np.int8),
            Index(list("abcde")),
            Index([0, "a", 1, "b", 2, "c"]),
            period_range("20130101", periods=5, freq="D"),
            TimedeltaIndex(
                [
                    "0 days 01:00:00",
                    "1 days 01:00:00",
                    "2 days 01:00:00",
                    "3 days 01:00:00",
                    "4 days 01:00:00",
                ],
                dtype="timedelta64[ns]",
                freq="D",
            ),
            DatetimeIndex(
                ["2013-01-01", "2013-01-02", "2013-01-03", "2013-01-04", "2013-01-05"],
                dtype="datetime64[ns]",
                freq="D",
            ),
            IntervalIndex.from_breaks(range(11), closed="right"),
        ]
    )
    # 返回不同参数化的简单索引对象
    def simple_index(self, request):
        return request.param

    # 测试pickle兼容性构造
    def test_pickle_compat_construction(self, simple_index):
        # 如果是RangeIndex对象，则跳过测试，因为RangeIndex是有效的构造器
        if isinstance(simple_index, RangeIndex):
            pytest.skip("RangeIndex() is a valid constructor")
        # 定义异常消息的模式，用于匹配错误信息
        msg = "|".join(
            [
                r"Index\(\.\.\.\) must be called with a collection of some "
                r"kind, None was passed",
                r"DatetimeIndex\(\) must be called with a collection of some "
                r"kind, None was passed",
                r"TimedeltaIndex\(\) must be called with a collection of some "
                r"kind, None was passed",
                r"__new__\(\) missing 1 required positional argument: 'data'",
                r"__new__\(\) takes at least 2 arguments \(1 given\)",
                r"'NoneType' object is not iterable",
            ]
        )
        # 使用pytest断言检查是否引发了预期的TypeError异常，并匹配错误消息
        with pytest.raises(TypeError, match=msg):
            type(simple_index)()
    # 测试简单索引对象的位移方法
    def test_shift(self, simple_index):
        # 如果 simple_index 是 DatetimeIndex、TimedeltaIndex 或 PeriodIndex 的实例，则跳过测试
        if isinstance(simple_index, (DatetimeIndex, TimedeltaIndex, PeriodIndex)):
            pytest.skip("Tested in test_ops/test_arithmetic")
        
        idx = simple_index  # 将 simple_index 赋值给 idx
        # 准备错误信息，指明该方法仅对 DatetimeIndex、PeriodIndex 和 TimedeltaIndex 实现
        msg = (
            f"This method is only implemented for DatetimeIndex, PeriodIndex and "
            f"TimedeltaIndex; Got type {type(idx).__name__}"
        )
        # 测试调用 idx.shift(1) 是否会抛出 NotImplementedError，并匹配指定错误信息
        with pytest.raises(NotImplementedError, match=msg):
            idx.shift(1)
        # 测试调用 idx.shift(1, 2) 是否会抛出 NotImplementedError，并匹配指定错误信息
        with pytest.raises(NotImplementedError, match=msg):
            idx.shift(1, 2)

    # 测试构造函数中的索引名是否为不可哈希类型
    def test_constructor_name_unhashable(self, simple_index):
        # GH#29069 检查索引名是否可哈希
        # 也可查看 tests.series.test_constructors 中同名的测试
        idx = simple_index  # 将 simple_index 赋值给 idx
        # 测试当 Index.name 不是可哈希类型时是否会抛出 TypeError，并匹配指定错误信息
        with pytest.raises(TypeError, match="Index.name must be a hashable type"):
            type(idx)(idx, name=[])

    # 测试创建索引时继承现有索引的名称
    def test_create_index_existing_name(self, simple_index):
        # GH11193 当传递现有索引且未指定新名称时，新索引应继承先前对象的名称
        expected = simple_index.copy()  # 复制 simple_index 到 expected
        if not isinstance(expected, MultiIndex):
            expected.name = "foo"  # 设置 expected 的名称为 "foo"
            # 创建 Index 对象，并断言结果与 expected 相等
            result = Index(expected)
            tm.assert_index_equal(result, expected)

            expected.name = "bar"  # 修改 expected 的名称为 "bar"
            # 创建 Index 对象，并断言结果与 expected 相等
            result = Index(expected, name="bar")
            tm.assert_index_equal(result, expected)
        else:
            expected.names = ["foo", "bar"]  # 设置 expected 的名称为 ["foo", "bar"]
            # 创建 Index 对象，并断言结果与指定的 MultiIndex 相等
            tm.assert_index_equal(
                result,
                Index(
                    Index(
                        [
                            ("foo", "one"),
                            ("foo", "two"),
                            ("bar", "one"),
                            ("baz", "two"),
                            ("qux", "one"),
                            ("qux", "two"),
                        ],
                        dtype="object",
                    ),
                    names=["foo", "bar"],
                ),
            )

            # 创建 Index 对象，并指定新的 names=["A", "B"]，断言结果与指定的 MultiIndex 相等
            result = Index(expected, names=["A", "B"])
            tm.assert_index_equal(
                result,
                Index(
                    Index(
                        [
                            ("foo", "one"),
                            ("foo", "two"),
                            ("bar", "one"),
                            ("baz", "two"),
                            ("qux", "one"),
                            ("qux", "two"),
                        ],
                        dtype="object",
                    ),
                    names=["A", "B"],
                ),
            )
    # 测试数字兼容性，接受一个名为 simple_index 的参数
    def test_numeric_compat(self, simple_index):
        idx = simple_index
        # 检查 simple_index 是否不是 MultiIndex 类型，如果是，跳过测试
        assert not isinstance(idx, MultiIndex)
        
        # 如果 idx 是 Index 类型，则跳过测试，因为不适用于 Index
        if type(idx) is Index:
            pytest.skip("Not applicable for Index")
        
        # 如果 simple_index 的数据类型是数值型或者是 TimedeltaIndex 类型，则跳过测试，因为在其他地方已经测试过了
        if is_numeric_dtype(simple_index.dtype) or isinstance(
            simple_index, TimedeltaIndex
        ):
            pytest.skip("Tested elsewhere.")

        # 获取 idx._data 的类型名称
        typ = type(idx._data).__name__
        # 获取 idx 对象本身的类型名称
        cls = type(idx).__name__
        
        # 创建错误消息字符串，用于检测 TypeError 异常，并匹配错误消息 lmsg
        lmsg = "|".join(
            [
                rf"unsupported operand type\(s\) for \*: '{typ}' and 'int'",
                "cannot perform (__mul__|__truediv__|__floordiv__) with "
                f"this index type: ({cls}|{typ})",
            ]
        )
        # 断言 idx * 1 会引发 TypeError 异常，且错误消息匹配 lmsg
        with pytest.raises(TypeError, match=lmsg):
            idx * 1
        
        # 创建错误消息字符串，用于检测 TypeError 异常，并匹配错误消息 rmsg
        rmsg = "|".join(
            [
                rf"unsupported operand type\(s\) for \*: 'int' and '{typ}'",
                "cannot perform (__rmul__|__rtruediv__|__rfloordiv__) with "
                f"this index type: ({cls}|{typ})",
            ]
        )
        # 断言 1 * idx 会引发 TypeError 异常，且错误消息匹配 rmsg
        with pytest.raises(TypeError, match=rmsg):
            1 * idx

        # 替换 lmsg 中的乘号为除号，创建错误消息字符串用于检测 TypeError 异常
        div_err = lmsg.replace("*", "/")
        # 断言 idx / 1 会引发 TypeError 异常，且错误消息匹配 div_err
        with pytest.raises(TypeError, match=div_err):
            idx / 1
        
        # 替换 rmsg 中的乘号为除号，创建错误消息字符串用于检测 TypeError 异常
        div_err = rmsg.replace("*", "/")
        # 断言 1 / idx 会引发 TypeError 异常，且错误消息匹配 div_err
        with pytest.raises(TypeError, match=div_err):
            1 / idx

        # 替换 lmsg 中的乘号为整除号，创建错误消息字符串用于检测 TypeError 异常
        floordiv_err = lmsg.replace("*", "//")
        # 断言 idx // 1 会引发 TypeError 异常，且错误消息匹配 floordiv_err
        with pytest.raises(TypeError, match=floordiv_err):
            idx // 1
        
        # 替换 rmsg 中的乘号为整除号，创建错误消息字符串用于检测 TypeError 异常
        floordiv_err = rmsg.replace("*", "//")
        # 断言 1 // idx 会引发 TypeError 异常，且错误消息匹配 floordiv_err
        with pytest.raises(TypeError, match=floordiv_err):
            1 // idx

    # 测试逻辑操作的兼容性，接受一个名为 simple_index 的参数
    def test_logical_compat(self, simple_index):
        # 如果 simple_index 的数据类型是 object 或者字符串类型，则跳过测试，因为在其他地方已经测试过了
        if simple_index.dtype in (object, "string"):
            pytest.skip("Tested elsewhere.")
        idx = simple_index
        
        # 如果 idx 的数据类型的种类属于 'iufcbm' 中的一种
        if idx.dtype.kind in "iufcbm":
            # 断言 idx.all() 等于 idx._values.all()
            assert idx.all() == idx._values.all()
            # 断言 idx.all() 等于 idx.to_series().all()
            assert idx.all() == idx.to_series().all()
            # 断言 idx.any() 等于 idx._values.any()
            assert idx.any() == idx._values.any()
            # 断言 idx.any() 等于 idx.to_series().any()
            assert idx.any() == idx.to_series().any()
        else:
            # 创建错误消息字符串，用于检测 TypeError 异常，匹配错误消息 msg
            msg = "does not support operation '(any|all)'"
            # 断言调用 idx.all() 会引发 TypeError 异常，且错误消息匹配 msg
            with pytest.raises(TypeError, match=msg):
                idx.all()
            # 断言调用 idx.any() 会引发 TypeError 异常，且错误消息匹配 msg
            with pytest.raises(TypeError, match=msg):
                idx.any()

    # 测试 repr 方法的往返转换，接受一个名为 simple_index 的参数
    def test_repr_roundtrip(self, simple_index):
        # 如果 simple_index 是 IntervalIndex 类型，则跳过测试，因为不支持该类型的 repr 方法
        if isinstance(simple_index, IntervalIndex):
            pytest.skip(f"Not a valid repr for {type(simple_index).__name__}")
        idx = simple_index
        # 断言 eval(repr(idx)) 等于 idx，即 repr 方法的逆过程应该还原为原始对象
        tm.assert_index_equal(eval(repr(idx)), idx)
    # 测试函数，用于验证最大序列项设置的表示形式
    def test_repr_max_seq_item_setting(self, simple_index):
        # 如果 simple_index 是 IntervalIndex 类型，则跳过测试，并给出相应的提示信息
        if isinstance(simple_index, IntervalIndex):
            pytest.skip(f"Not a valid repr for {type(simple_index).__name__}")
        
        # 将 simple_index 赋值给 idx
        idx = simple_index
        
        # 将 idx 扩展重复50次
        idx = idx.repeat(50)
        
        # 设置上下文，将 display.max_seq_items 选项设置为无限制
        with pd.option_context("display.max_seq_items", None):
            # 调用 repr 函数生成 idx 的表示形式字符串
            repr(idx)
            
            # 断言检查字符串形式的 idx 中不包含 "..."
            assert "..." not in str(idx)

    # 使用 pytest 标记，过滤掉关于 PeriodDtype[B] 废弃警告的提示
    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    def test_ensure_copied_data(self, index):
        # 检查 Index.__new__ 的 "copy" 参数是否被正确处理
        # GH12309
        init_kwargs = {}
        if isinstance(index, PeriodIndex):
            # 如果是 PeriodIndex 类型，则需要指定 "freq"
            init_kwargs["freq"] = index.freq
        elif isinstance(index, (RangeIndex, MultiIndex, CategoricalIndex)):
            # RangeIndex 不能从数据中初始化，MultiIndex 和 CategoricalIndex 分别进行单独测试
            pytest.skip(
                "RangeIndex cannot be initialized from data, "
                "MultiIndex and CategoricalIndex are tested separately"
            )
        elif index.dtype == object and index.inferred_type == "boolean":
            # 如果数据类型是对象且推断类型为布尔型，则设置 dtype
            init_kwargs["dtype"] = index.dtype

        index_type = type(index)
        # 使用指定的参数初始化一个新的 index 对象，并进行复制
        result = index_type(index.values, copy=True, **init_kwargs)
        if isinstance(index.dtype, DatetimeTZDtype):
            # 如果数据类型是 DatetimeTZDtype，则将结果进行时区本地化和转换
            result = result.tz_localize("UTC").tz_convert(index.tz)
        if isinstance(index, (DatetimeIndex, TimedeltaIndex)):
            # 如果是 DatetimeIndex 或 TimedeltaIndex，则设置频率为 None
            index = index._with_freq(None)

        # 断言 index 和 result 相等
        tm.assert_index_equal(index, result)

        if isinstance(index, PeriodIndex):
            # 如果是 PeriodIndex 类型，则从 ordinals 创建一个新的 index 对象
            result = index_type.from_ordinals(ordinals=index.asi8, **init_kwargs)
            # 断言 ordinals 数组相等
            tm.assert_numpy_array_equal(index.asi8, result.asi8, check_same="same")
        elif isinstance(index, IntervalIndex):
            # 在 test_interval.py 中进行检查，这里不做处理
            pass
        elif type(index) is Index and not isinstance(index.dtype, np.dtype):
            # 如果是普通的 Index 类型且数据类型不是 np.dtype
            result = index_type(index.values, copy=False, **init_kwargs)
            # 断言 index 和 result 相等
            tm.assert_index_equal(result, index)

            if isinstance(index._values, BaseMaskedArray):
                # 如果 index._values 是 BaseMaskedArray 类型，则检查共享内存情况
                assert np.shares_memory(index._values._data, result._values._data)
                tm.assert_numpy_array_equal(
                    index._values._data, result._values._data, check_same="same"
                )
                assert np.shares_memory(index._values._mask, result._values._mask)
                tm.assert_numpy_array_equal(
                    index._values._mask, result._values._mask, check_same="same"
                )
            elif index.dtype == "string[python]":
                # 如果数据类型是 string[python]，则检查共享内存情况
                assert np.shares_memory(index._values._ndarray, result._values._ndarray)
                tm.assert_numpy_array_equal(
                    index._values._ndarray, result._values._ndarray, check_same="same"
                )
            elif index.dtype in ("string[pyarrow]", "string[pyarrow_numpy]"):
                # 如果数据类型是 string[pyarrow] 或 string[pyarrow_numpy]，则检查共享内存情况
                assert tm.shares_memory(result._values, index._values)
            else:
                # 如果以上条件都不满足，则抛出 NotImplementedError
                raise NotImplementedError(index.dtype)
        else:
            # 否则，进行非复制方式的初始化
            result = index_type(index.values, copy=False, **init_kwargs)
            # 断言 index.values 和 result.values 相等
            tm.assert_numpy_array_equal(index.values, result.values, check_same="same")
    # 测试索引对象的内存使用情况
    def test_memory_usage(self, index):
        # 清除索引对象的映射缓存
        index._engine.clear_mapping()
        # 计算索引对象的内存使用量
        result = index.memory_usage()
        if index.empty:
            # 如果索引为空，则报告内存使用量为0
            assert result == 0
            return

        # 索引非空时，获取索引首元素的位置
        index.get_loc(index[0])
        # 计算更新后的内存使用量
        result2 = index.memory_usage()
        # 深度计算索引对象的内存使用量
        result3 = index.memory_usage(deep=True)

        # 对于非 RangeIndex 和 IntervalIndex 类型的索引
        # 以及类型为 Index 且 dtype 不是 np.dtype 的索引
        if not isinstance(index, (RangeIndex, IntervalIndex)) and not (
            type(index) is Index and not isinstance(index.dtype, np.dtype)
        ):
            # 断言更新后的内存使用量大于初始内存使用量
            assert result2 > result

        # 如果索引类型推断为 "object"
        if index.inferred_type == "object":
            # 断言深度计算的内存使用量大于更新后的内存使用量
            assert result3 > result2

    # 测试索引对象在不触发引擎的情况下的内存使用量
    def test_memory_usage_doesnt_trigger_engine(self, index):
        # 清除索引对象的缓存
        index._cache.clear()
        # 断言在缓存中不存在 "_engine" 键
        assert "_engine" not in index._cache

        # 计算不触发引擎时的内存使用量
        res_without_engine = index.memory_usage()
        # 再次断言在缓存中不存在 "_engine" 键
        assert "_engine" not in index._cache

        # 显式加载并缓存引擎
        _ = index._engine
        # 断言在缓存中存在 "_engine" 键
        assert "_engine" in index._cache

        # 计算触发引擎后的内存使用量
        res_with_engine = index.memory_usage()

        # 即使引擎被初始化为非空，空引擎不会影响结果
        # 因为 engine.sizeof() 不考虑 engine.values 的内容
        assert res_with_engine == res_without_engine

        # 如果索引对象长度为 0
        if len(index) == 0:
            # 断言不触发引擎时的内存使用量为 0
            assert res_without_engine == 0
            # 断言触发引擎时的内存使用量为 0
            assert res_with_engine == 0
        else:
            # 断言不触发引擎时的内存使用量大于 0
            assert res_without_engine > 0
            # 断言触发引擎时的内存使用量大于 0
            assert res_with_engine > 0

    # 测试索引对象的 argsort 方法
    def test_argsort(self, index):
        # 如果索引对象是 CategoricalIndex 类型，则跳过测试
        if isinstance(index, CategoricalIndex):
            pytest.skip(f"{type(self).__name__} separately tested")

        # 执行索引对象的 argsort 方法
        result = index.argsort()
        # 期望的结果是索引数组的排序结果
        expected = np.array(index).argsort()
        # 断言结果与期望一致
        tm.assert_numpy_array_equal(result, expected, check_dtype=False)

    # 测试索引对象的 numpy.argsort 方法
    def test_numpy_argsort(self, index):
        # 执行索引对象的 numpy.argsort 方法
        result = np.argsort(index)
        # 期望的结果是索引对象的 argsort 方法结果
        expected = index.argsort()
        # 断言结果与期望一致
        tm.assert_numpy_array_equal(result, expected)

        # 使用 mergesort 指定排序算法执行 numpy.argsort 方法
        result = np.argsort(index, kind="mergesort")
        # 期望的结果是索引对象的 argsort 方法结果
        expected = index.argsort(kind="mergesort")
        # 断言结果与期望一致
        tm.assert_numpy_array_equal(result, expected)

        # 下列两种类型执行 pandas 兼容性输入验证
        # 其余类型已在 pandas.core.indexes/base.py 中定义了 'values' 属性上的单独（或无）验证
        if isinstance(index, (CategoricalIndex, RangeIndex)):
            msg = "the 'axis' parameter is not supported"
            # 断言使用 axis 参数时抛出 ValueError 异常
            with pytest.raises(ValueError, match=msg):
                np.argsort(index, axis=1)

            msg = "the 'order' parameter is not supported"
            # 断言使用 order 参数时抛出 ValueError 异常
            with pytest.raises(ValueError, match=msg):
                np.argsort(index, order=("a", "b"))
    # 测试重复索引功能
    def test_repeat(self, simple_index):
        # 设置重复次数
        rep = 2
        # 复制简单索引对象
        idx = simple_index.copy()
        # 获取新索引对象的构造器
        new_index_cls = idx._constructor
        # 创建预期结果，重复索引值并设置名称
        expected = new_index_cls(idx.values.repeat(rep), name=idx.name)
        # 断言索引重复操作的结果与预期结果相等
        tm.assert_index_equal(idx.repeat(rep), expected)

        # 将 idx 设置为原始的 simple_index
        idx = simple_index
        # 创建重复数组，长度与索引相同
        rep = np.arange(len(idx))
        # 创建预期结果，重复索引值并设置名称
        expected = new_index_cls(idx.values.repeat(rep), name=idx.name)
        # 断言索引重复操作的结果与预期结果相等
        tm.assert_index_equal(idx.repeat(rep), expected)

    # 测试使用 NumPy 的重复功能
    def test_numpy_repeat(self, simple_index):
        # 设置重复次数
        rep = 2
        # 获取简单索引对象
        idx = simple_index
        # 创建预期结果，使用索引对象的重复方法
        expected = idx.repeat(rep)
        # 断言 NumPy 的重复操作结果与预期结果相等
        tm.assert_index_equal(np.repeat(idx, rep), expected)

        # 设置错误消息
        msg = "the 'axis' parameter is not supported"
        # 使用 pytest 检查是否引发 ValueError 异常，并匹配错误消息
        with pytest.raises(ValueError, match=msg):
            np.repeat(idx, rep, axis=0)

    # 测试 where 方法
    def test_where(self, listlike_box, simple_index):
        # 如果简单索引对象是区间索引或周期索引，或者是数值类型，则跳过测试
        if isinstance(simple_index, (IntervalIndex, PeriodIndex)) or is_numeric_dtype(
            simple_index.dtype
        ):
            pytest.skip("Tested elsewhere.")
        # 获取列表型对象的类
        klass = listlike_box

        # 获取简单索引对象
        idx = simple_index
        # 如果是日期时间索引或时间增量索引，移除频率信息
        if isinstance(idx, (DatetimeIndex, TimedeltaIndex)):
            idx = idx._with_freq(None)

        # 创建条件列表，全为 True
        cond = [True] * len(idx)
        # 使用 where 方法进行条件筛选
        result = idx.where(klass(cond))
        # 预期结果为原始索引对象
        expected = idx
        # 断言 where 方法的结果与预期结果相等
        tm.assert_index_equal(result, expected)

        # 创建条件列表，第一个为 False，其余为 True
        cond = [False] + [True] * len(idx[1:])
        # 创建预期结果，将第一个值设为缺失值，其余值不变
        expected = Index([idx._na_value] + idx[1:].tolist(), dtype=idx.dtype)
        # 使用 where 方法进行条件筛选
        result = idx.where(klass(cond))
        # 断言 where 方法的结果与预期结果相等
        tm.assert_index_equal(result, expected)

    # 测试插入基础功能
    def test_insert_base(self, index):
        # GH#51363
        # 获取索引的子集，从第二个到第四个元素
        trimmed = index[1:4]

        # 如果索引长度为 0，则跳过测试
        if not len(index):
            pytest.skip("Not applicable for empty index")

        # 在索引的开头插入第一个元素
        result = trimmed.insert(0, index[0])
        # 断言插入操作后的结果与预期的子集相等
        assert index[0:4].equals(result)

    # 使用 pytest 标记，根据条件跳过测试
    @pytest.mark.skipif(
        using_pyarrow_string_dtype(),
        reason="completely different behavior, tested elsewher",
    )
    # 测试插入超出界限的情况
    def test_insert_out_of_bounds(self, index):
        # TypeError/IndexError 匹配 NumPy 在这些情况下引发的异常

        # 如果索引长度大于 0，则错误类型为 TypeError
        if len(index) > 0:
            err = TypeError
        else:
            err = IndexError
        # 如果索引长度为 0，则错误消息中的 "0" 可能根据 NumPy 版本变化
        if len(index) == 0:
            msg = "index (0|0.5) is out of bounds for axis 0 with size 0"
        else:
            msg = "slice indices must be integers or None or have an __index__ method"
        # 使用 pytest 检查是否引发指定类型的异常，并匹配错误消息
        with pytest.raises(err, match=msg):
            index.insert(0.5, "foo")

        # 设置匹配模式，用于检查 IndexError 异常的错误消息
        msg = "|".join(
            [
                r"index -?\d+ is out of bounds for axis 0 with size \d+",
                "loc must be an integer between",
            ]
        )
        # 使用 pytest 检查是否引发 IndexError 异常，并匹配错误消息
        with pytest.raises(IndexError, match=msg):
            index.insert(len(index) + 1, 1)

        # 使用 pytest 检查是否引发 IndexError 异常，并匹配错误消息
        with pytest.raises(IndexError, match=msg):
            index.insert(-len(index) - 1, 1)
    # 定义一个测试方法，用于测试删除操作的基本功能
    def test_delete_base(self, index):
        # 如果索引为空，则跳过测试，并给出相应的提示信息
        if not len(index):
            pytest.skip("Not applicable for empty index")

        # 如果索引是 RangeIndex 类型的实例，则跳过测试，因为这种情况已在其他地方测试过
        if isinstance(index, RangeIndex):
            pytest.skip(f"{type(self).__name__} tested elsewhere")

        # 预期的结果是从索引中删除第一个元素后的索引
        expected = index[1:]
        # 执行删除操作，删除索引中的第一个元素
        result = index.delete(0)
        # 断言删除后的结果与预期的结果相等
        assert result.equals(expected)
        # 断言删除后的结果的名称与预期的结果的名称相等
        assert result.name == expected.name

        # 预期的结果是从索引中删除最后一个元素后的索引
        expected = index[:-1]
        # 执行删除操作，删除索引中的最后一个元素
        result = index.delete(-1)
        # 断言删除后的结果与预期的结果相等
        assert result.equals(expected)
        # 断言删除后的结果的名称与预期的结果的名称相等
        assert result.name == expected.name

        # 获取索引的长度
        length = len(index)
        # 准备索引超出范围的错误信息
        msg = f"index {length} is out of bounds for axis 0 with size {length}"
        # 断言执行删除超出索引范围的操作时会抛出 IndexError 异常，并且异常消息匹配预期的错误信息
        with pytest.raises(IndexError, match=msg):
            index.delete(length)

    # 用于测试索引对象的相等性方法
    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    def test_equals(self, index):
        # 如果索引是 IntervalIndex 类型的实例，则跳过测试，因为这种情况已在其他地方测试过
        if isinstance(index, IntervalIndex):
            pytest.skip(f"{type(index).__name__} tested elsewhere")

        # 判断索引是否为普通的 Index 类型，并且其 dtype 不是 np.dtype 类型的实例
        is_ea_idx = type(index) is Index and not isinstance(index.dtype, np.dtype)

        # 断言索引与自身相等
        assert index.equals(index)
        # 断言索引与其拷贝相等
        assert index.equals(index.copy())
        # 如果索引不是特定类型的 Index 类型，则断言索引与其转换为对象类型后相等
        if not is_ea_idx:
            assert index.equals(index.astype(object))

        # 断言索引与其列表表示形式不相等
        assert not index.equals(list(index))
        # 断言索引与其 NumPy 数组表示形式不相等
        assert not index.equals(np.array(index))

        # 如果索引不是 RangeIndex 类型，并且不是特定类型的 Index 类型，则进行以下测试
        if not isinstance(index, RangeIndex) and not is_ea_idx:
            # 创建一个 dtype 为 object 类型的索引，与原索引比较，断言它们相等
            same_values = Index(index, dtype=object)
            assert index.equals(same_values)
            assert same_values.equals(index)

        # 如果索引只有一个层级，则不测试 MultiIndex
        if index.nlevels == 1:
            assert not index.equals(Series(index))
    # 定义一个测试方法，用于检验索引相等的情况
    def test_equals_op(self, simple_index):
        # 设置测试用例编号 GH9947, GH10637

        # 复制传入的简单索引对象到 index_a
        index_a = simple_index

        # 计算索引对象 index_a 的长度
        n = len(index_a)

        # 创建一个不包含最后一个元素的切片索引 index_b
        index_b = index_a[0:-1]

        # 尝试对 index_a 进行切片并在末尾追加倒数第二到倒数第一个元素，但这不会返回一个新的列表，因为 list.append() 操作返回 None。
        index_c = index_a[0:-1].append(index_a[-2:-1])

        # 创建一个仅包含第一个元素的索引 index_d
        index_d = index_a[0:1]

        # 定义一个错误消息
        msg = "Lengths must match|could not be broadcast"

        # 断言 index_a 和 index_b 是否相等，预期抛出 ValueError 异常并匹配特定消息
        with pytest.raises(ValueError, match=msg):
            index_a == index_b

        # 创建一个预期的布尔数组，其长度为 n，所有元素为 True
        expected1 = np.array([True] * n)

        # 创建一个预期的布尔数组，其长度为 n-1，前 n-1 个元素为 True，最后一个元素为 False
        expected2 = np.array([True] * (n - 1) + [False])

        # 断言 index_a 和 index_a 是否相等，预期结果应与 expected1 相等
        tm.assert_numpy_array_equal(index_a == index_a, expected1)

        # 断言 index_a 和 index_c 是否相等，预期结果应与 expected2 相等
        tm.assert_numpy_array_equal(index_a == index_c, expected2)

        # 测试与 numpy 数组的比较
        array_a = np.array(index_a)
        array_b = np.array(index_a[0:-1])

        # 尝试对 index_a 进行切片并在末尾追加倒数第二到倒数第一个元素，但同样这里由于 list.append() 操作返回 None。
        array_c = np.array(index_a[0:-1].append(index_a[-2:-1]))

        array_d = np.array(index_a[0:1])

        # 断言 index_a 和 array_b 是否相等，预期抛出 ValueError 异常并匹配特定消息
        with pytest.raises(ValueError, match=msg):
            index_a == array_b

        # 断言 index_a 和 array_a 是否相等，预期结果应与 expected1 相等
        tm.assert_numpy_array_equal(index_a == array_a, expected1)

        # 断言 index_a 和 array_c 是否相等，预期结果应与 expected2 相等
        tm.assert_numpy_array_equal(index_a == array_c, expected2)

        # 测试与 Series 的比较
        series_a = Series(array_a)
        series_b = Series(array_b)
        series_c = Series(array_c)
        series_d = Series(array_d)

        # 断言 index_a 和 series_b 是否相等，预期抛出 ValueError 异常并匹配特定消息
        with pytest.raises(ValueError, match=msg):
            index_a == series_b

        # 断言 index_a 和 series_a 是否相等，预期结果应与 expected1 相等
        tm.assert_numpy_array_equal(index_a == series_a, expected1)

        # 断言 index_a 和 series_c 是否相等，预期结果应与 expected2 相等
        tm.assert_numpy_array_equal(index_a == series_c, expected2)

        # 长度为 1 的特殊情况
        with pytest.raises(ValueError, match="Lengths must match"):
            index_a == index_d

        with pytest.raises(ValueError, match="Lengths must match"):
            index_a == series_d

        with pytest.raises(ValueError, match="Lengths must match"):
            index_a == array_d

        # 比较两个 Series 对象，预期抛出特定异常
        msg = "Can only compare identically-labeled Series objects"
        with pytest.raises(ValueError, match=msg):
            series_a == series_d

        with pytest.raises(ValueError, match="Lengths must match"):
            series_a == array_d

        # 与标量比较应该广播；注意排除 MultiIndex，因为每个索引项都是长度为 2 的元组，因此被视为长度为 2 的数组而不是标量
        if not isinstance(index_a, MultiIndex):
            # 创建一个预期的布尔数组，其长度为 len(index_a)-2，前 len(index_a)-2 个元素为 False，倒数第二个为 True，最后一个为 False
            expected3 = np.array([False] * (len(index_a) - 2) + [True, False])

            # 假设倒数第二个元素在数据中是唯一的
            item = index_a[-2]

            # 断言 index_a 与 item 是否相等，预期结果应与 expected3 相等
            tm.assert_numpy_array_equal(index_a == item, expected3)

            # 断言 series_a 与 item 是否相等，预期结果应为 Series(expected3)
            tm.assert_series_equal(series_a == item, Series(expected3))
    # 定义一个测试方法，用于测试在给定索引上执行fillna操作
    def test_fillna(self, index):
        # GH 11343
        # 如果索引长度为0，则跳过测试，因为对空索引无意义
        if len(index) == 0:
            pytest.skip("Not relevant for empty index")
        # 如果索引的数据类型是布尔型，则跳过测试，因为布尔型索引不能包含NA值
        elif index.dtype == bool:
            pytest.skip(f"{index.dtype} cannot hold NAs")
        # 如果索引是Index类型且其数据类型是整数型，则跳过测试
        elif isinstance(index, Index) and is_integer_dtype(index.dtype):
            pytest.skip(f"Not relevant for Index with {index.dtype}")
        # 如果索引是MultiIndex类型，则进行以下测试
        elif isinstance(index, MultiIndex):
            # 复制索引以避免直接修改原始数据
            idx = index.copy(deep=True)
            msg = "isna is not defined for MultiIndex"
            # 断言调用fillna方法时会抛出NotImplementedError异常，并检查异常消息
            with pytest.raises(NotImplementedError, match=msg):
                idx.fillna(idx[0])
        # 对于其他类型的索引，执行以下测试
        else:
            # 复制索引以避免直接修改原始数据
            idx = index.copy(deep=True)
            # 使用第一个非NA值填充NA值，并保存填充后的结果
            result = idx.fillna(idx[0])
            # 断言填充后的结果与原索引相等
            tm.assert_index_equal(result, idx)
            # 断言填充后的结果与原索引不是同一个对象
            assert result is not idx

            # 准备测试填充非标量值时是否抛出TypeError异常
            msg = "'value' must be a scalar, passed: "
            # 断言调用fillna方法时会抛出TypeError异常，并检查异常消息
            with pytest.raises(TypeError, match=msg):
                idx.fillna([idx[0]])

            # 复制索引以避免直接修改原始数据
            idx = index.copy(deep=True)
            # 获取索引的内部值数组
            values = idx._values

            # 将第二个值设置为NaN，模拟存在NA值的情况
            values[1] = np.nan

            # 使用修改后的值数组创建新索引对象
            idx = type(index)(values)

            # 准备预期结果，期望第二个位置为True，其余位置为False的布尔数组
            expected = np.array([False] * len(idx), dtype=bool)
            expected[1] = True

            # 断言新索引的_isnan属性与预期结果相等
            tm.assert_numpy_array_equal(idx._isnan, expected)
            # 断言新索引包含NaN值
            assert idx.hasnans is True

    # 定义一个测试方法，用于测试索引对象上的isna和notna方法
    def test_nulls(self, index):
        # this is really a smoke test for the methods
        # as these are adequately tested for function elsewhere
        # 如果索引长度为0，则断言其isna方法返回一个空的布尔数组
        if len(index) == 0:
            tm.assert_numpy_array_equal(index.isna(), np.array([], dtype=bool))
        # 如果索引是MultiIndex类型，则进行以下测试
        elif isinstance(index, MultiIndex):
            # 复制索引以避免直接修改原始数据
            idx = index.copy()
            msg = "isna is not defined for MultiIndex"
            # 断言调用isna方法时会抛出NotImplementedError异常，并检查异常消息
            with pytest.raises(NotImplementedError, match=msg):
                idx.isna()
        # 如果索引不包含任何NA值，则断言其isna方法返回全为False的布尔数组，而notna方法返回全为True的布尔数组
        elif not index.hasnans:
            tm.assert_numpy_array_equal(index.isna(), np.zeros(len(index), dtype=bool))
            tm.assert_numpy_array_equal(index.notna(), np.ones(len(index), dtype=bool))
        # 对于其他情况，比较索引的isna方法返回的结果与调用isna函数得到的结果
        else:
            result = isna(index)
            tm.assert_numpy_array_equal(index.isna(), result)
            tm.assert_numpy_array_equal(index.notna(), ~result)

    # 定义一个测试方法，用于测试空索引的情况
    def test_empty(self, simple_index):
        # GH 15270
        # 获取simple_index并命名为idx
        idx = simple_index
        # 断言索引不为空
        assert not idx.empty
        # 断言索引切片后为空
        assert idx[:0].empty

    # 定义一个测试方法，用于测试在相同索引对象上执行join操作时的情况
    def test_join_self_unique(self, join_type, simple_index):
        # 获取simple_index并命名为idx
        idx = simple_index
        # 如果索引是唯一的，则进行以下测试
        if idx.is_unique:
            # 执行索引对象与自身的join操作，并命名为joined
            joined = idx.join(idx, how=join_type)
            # 准备预期结果，如果join_type为"outer"，则对expected执行安全排序
            expected = simple_index
            if join_type == "outer":
                expected = algos.safe_sort(expected)
            # 断言joined与expected相等
            tm.assert_index_equal(joined, expected)

    # 定义一个测试方法，用于测试在简单索引上执行map操作的情况
    def test_map(self, simple_index):
        # callable
        # 如果simple_index是TimedeltaIndex或PeriodIndex类型，则跳过测试
        if isinstance(simple_index, (TimedeltaIndex, PeriodIndex)):
            pytest.skip("Tested elsewhere.")
        # 获取simple_index并命名为idx
        idx = simple_index

        # 执行索引对象的map方法，并命名结果为result
        result = idx.map(lambda x: x)
        # 断言result与idx相等，exact参数设置为"equiv"，表示严格等价
        # RangeIndex相当于dtype为int64的相似索引
        tm.assert_index_equal(result, idx, exact="equiv")
    # 使用 pytest.mark.parametrize 装饰器，为 test_map_dictlike 方法参数化不同的 mapper 函数
    # 第一个 mapper 函数将 values 和 index 组合成字典
    # 第二个 mapper 函数使用 Series 将 values 和 index 组合成序列
    @pytest.mark.parametrize(
        "mapper",
        [
            lambda values, index: {i: e for e, i in zip(values, index)},
            lambda values, index: Series(values, index),
        ],
    )
    # 使用 pytest.mark.filterwarnings 装饰器，忽略特定警告消息
    @pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
    # 定义 test_map_dictlike 方法，接受 mapper、simple_index 和 request 作为参数
    def test_map_dictlike(self, mapper, simple_index, request):
        # 将 simple_index 赋值给 idx
        idx = simple_index
        # 如果 idx 的类型是 DatetimeIndex、TimedeltaIndex 或 PeriodIndex 中的一种，跳过测试
        if isinstance(idx, (DatetimeIndex, TimedeltaIndex, PeriodIndex)):
            pytest.skip("Tested elsewhere.")

        # 使用 mapper 函数创建 identity 对象
        identity = mapper(idx.values, idx)

        # 使用 map 方法将 identity 应用于 idx，得到结果 result
        result = idx.map(identity)
        # 使用 assert_index_equal 断言 result 与 idx 在 "equiv" 模式下相等
        tm.assert_index_equal(result, idx, exact="equiv")

        # 如果 idx 的 dtype 的 kind 属性为 'f'，将 dtype 设为 idx 的 dtype
        # 创建一个与 idx 长度相同的由 NaN 组成的 Index 对象，dtype 为之前设定的 dtype
        dtype = None
        if idx.dtype.kind == "f":
            dtype = idx.dtype

        expected = Index([np.nan] * len(idx), dtype=dtype)
        # 使用 mapper 函数将 expected 和 idx 作为参数应用于 map 方法，得到结果 result
        result = idx.map(mapper(expected, idx))
        # 使用 assert_index_equal 断言 result 与 expected 相等
        tm.assert_index_equal(result, expected)

    # 定义 test_map_str 方法，接受 simple_index 作为参数
    def test_map_str(self, simple_index):
        # 如果 simple_index 的类型是 CategoricalIndex，跳过测试
        if isinstance(simple_index, CategoricalIndex):
            pytest.skip("See test_map.py")
        # 将 simple_index 赋值给 idx
        idx = simple_index
        # 使用 map 方法将 str 函数应用于 idx，得到结果 result
        result = idx.map(str)
        # 创建一个由 idx 中每个元素转换为字符串后组成的 Index 对象，作为期望的结果 expected
        expected = Index([str(x) for x in idx])
        # 使用 assert_index_equal 断言 result 与 expected 相等
        tm.assert_index_equal(result, expected)

    # 使用 pytest.mark.parametrize 装饰器，为 test_astype_category 方法参数化不同的 copy、name 和 ordered 值
    @pytest.mark.parametrize("copy", [True, False])
    @pytest.mark.parametrize("name", [None, "foo"])
    @pytest.mark.parametrize("ordered", [True, False])
    # 定义 test_astype_category 方法，接受 copy、name、ordered 和 simple_index 作为参数
    def test_astype_category(self, copy, name, ordered, simple_index):
        # 将 simple_index 赋值给 idx
        idx = simple_index
        # 如果 name 不为 None，将 idx 重命名为 name
        if name:
            idx = idx.rename(name)

        # 创建 CategoricalDtype 对象，指定 ordered 参数
        dtype = CategoricalDtype(ordered=ordered)
        # 使用 astype 方法将 idx 转换为 dtype 类型，得到结果 result
        result = idx.astype(dtype, copy=copy)
        # 创建一个预期的 CategoricalIndex 对象，与转换后的 idx 结果相等
        expected = CategoricalIndex(idx, name=name, ordered=ordered)
        # 使用 assert_index_equal 断言 result 与 expected 相等，要求精确匹配
        tm.assert_index_equal(result, expected, exact=True)

        # 创建非标准分类类型的 CategoricalDtype 对象，使用 idx 中除最后一个元素外的唯一值列表和 ordered 参数
        dtype = CategoricalDtype(idx.unique().tolist()[:-1], ordered)
        # 再次使用 astype 方法将 idx 转换为 dtype 类型，得到结果 result
        result = idx.astype(dtype, copy=copy)
        # 创建另一个预期的 CategoricalIndex 对象，与转换后的 idx 结果相等
        expected = CategoricalIndex(idx, name=name, dtype=dtype)
        # 使用 assert_index_equal 断言 result 与 expected 相等，要求精确匹配
        tm.assert_index_equal(result, expected, exact=True)

        # 如果 ordered 为 False
        if ordered is False:
            # 使用 astype 方法将 idx 转换为 'category' 类型，得到结果 result
            result = idx.astype("category", copy=copy)
            # 创建另一个预期的 CategoricalIndex 对象，与转换后的 idx 结果相等
            expected = CategoricalIndex(idx, name=name)
            # 使用 assert_index_equal 断言 result 与 expected 相等，要求精确匹配
            tm.assert_index_equal(result, expected, exact=True)
    # 测试索引是否唯一的函数
    def test_is_unique(self, simple_index):
        # 初始化一个去重后的索引
        index = simple_index.drop_duplicates()
        # 断言索引是否唯一
        assert index.is_unique is True

        # 空索引应该是唯一的
        index_empty = index[:0]
        assert index_empty.is_unique is True

        # 测试基本的重复项
        index_dup = index.insert(0, index[0])
        # 断言索引是否不唯一
        assert index_dup.is_unique is False

        # 单个 NA 应该是唯一的
        index_na = index.insert(0, np.nan)
        assert index_na.is_unique is True

        # 多个 NA 不应该是唯一的
        index_na_dup = index_na.insert(0, np.nan)
        assert index_na_dup.is_unique is False

    @pytest.mark.arm_slow
    # 测试引擎的引用循环
    def test_engine_reference_cycle(self, simple_index):
        # GH27585
        index = simple_index.copy()
        ref = weakref.ref(index)
        index._engine  # 访问引擎
        del index
        # 断言引用已经被清除
        assert ref() is None

    # 测试2D索引的过时功能
    def test_getitem_2d_deprecated(self, simple_index):
        # GH#30588, GH#31479
        if isinstance(simple_index, IntervalIndex):
            pytest.skip("Tested elsewhere")
        idx = simple_index
        msg = "Multi-dimensional indexing|too many|only"
        # 使用 pytest 的断言检查是否抛出特定异常消息
        with pytest.raises((ValueError, IndexError), match=msg):
            idx[:, None]

        if not isinstance(idx, RangeIndex):
            # GH#44051 RangeIndex 在 2.0 版本前已经以不同的消息抛出异常
            with pytest.raises((ValueError, IndexError), match=msg):
                idx[True]
            with pytest.raises((ValueError, IndexError), match=msg):
                idx[False]
        else:
            msg = "only integers, slices"
            with pytest.raises(IndexError, match=msg):
                idx[True]
            with pytest.raises(IndexError, match=msg):
                idx[False]

    # 测试复制索引是否共享缓存
    def test_copy_shares_cache(self, simple_index):
        # GH32898, GH36840
        idx = simple_index
        idx.get_loc(idx[0])  # 填充 _cache。
        copy = idx.copy()

        # 断言复制的索引和原索引共享相同的缓存
        assert copy._cache is idx._cache

    # 测试浅复制索引是否共享缓存
    def test_shallow_copy_shares_cache(self, simple_index):
        # GH32669, GH36840
        idx = simple_index
        idx.get_loc(idx[0])  # 填充 _cache。
        shallow_copy = idx._view()

        # 断言浅复制的索引和原索引共享相同的缓存
        assert shallow_copy._cache is idx._cache

        shallow_copy = idx._shallow_copy(idx._data)
        # 断言浅复制的索引和原索引不共享相同的缓存
        assert shallow_copy._cache is not idx._cache
        # 断言浅复制的索引的缓存为空字典
        assert shallow_copy._cache == {}
    def test_index_groupby(self, simple_index):
        # 从给定的简单索引中取前五个元素
        idx = simple_index[:5]
        # 创建一个包含特定数值和NaN的numpy数组用于分组
        to_groupby = np.array([1, 2, np.nan, 2, 1])
        # 断言索引按照给定的分组键进行分组后的预期结果
        tm.assert_dict_equal(
            idx.groupby(to_groupby), {1.0: idx[[0, 4]], 2.0: idx[[1, 3]]}
        )

        # 创建一个包含日期时间索引的对象数组，其中包括了特定的日期和NaT
        to_groupby = DatetimeIndex(
            [
                datetime(2011, 11, 1),
                datetime(2011, 12, 1),
                pd.NaT,
                datetime(2011, 12, 1),
                datetime(2011, 11, 1),
            ],
            tz="UTC",
        ).values

        # 期望的分组结果，使用日期时间作为键
        ex_keys = [Timestamp("2011-11-01"), Timestamp("2011-12-01")]
        expected = {ex_keys[0]: idx[[0, 4]], ex_keys[1]: idx[[1, 3]]}
        # 断言索引按照给定的日期时间分组键进行分组后的预期结果
        tm.assert_dict_equal(idx.groupby(to_groupby), expected)

    def test_append_preserves_dtype(self, simple_index):
        # 确保索引在附加操作后保持相同的数据类型，特别是对于dtype为float32的索引
        index = simple_index
        N = len(index)

        # 执行索引的附加操作
        result = index.append(index)
        # 断言附加后的结果与原索引的数据类型相同
        assert result.dtype == index.dtype
        # 断言前半部分和后半部分的附加结果与原索引相等
        tm.assert_index_equal(result[:N], index, check_exact=True)
        tm.assert_index_equal(result[N:], index, check_exact=True)

        # 创建另一种附加结果的期望，确保附加操作正确
        alt = index.take(list(range(N)) * 2)
        tm.assert_index_equal(result, alt, check_exact=True)

    def test_inv(self, simple_index, using_infer_string):
        # 获取简单索引
        idx = simple_index

        # 根据索引的数据类型不同执行不同的逆操作
        if idx.dtype.kind in ["i", "u"]:
            # 对整数类型的索引进行位取反操作
            res = ~idx
            expected = Index(~idx.values, name=idx.name)
            # 断言逆操作的结果与预期的索引相等
            tm.assert_index_equal(res, expected)

            # 检查是否与Series的行为一致
            res2 = ~Series(idx)
            tm.assert_series_equal(res2, Series(expected))
        else:
            if idx.dtype.kind == "f":
                err = TypeError
                msg = "ufunc 'invert' not supported for the input types"
            elif using_infer_string and idx.dtype == "string":
                import pyarrow as pa

                err = pa.lib.ArrowNotImplementedError
                msg = "has no kernel"
            else:
                err = TypeError
                msg = "bad operand"
            # 使用pytest断言逆操作在不同情况下会引发预期的错误
            with pytest.raises(err, match=msg):
                ~idx

            # 检查是否与Series的行为一致
            with pytest.raises(err, match=msg):
                ~Series(idx)
class TestNumericBase:
    @pytest.fixture(
        params=[
            RangeIndex(start=0, stop=20, step=2),  # 参数化测试数据：范围索引，从0到20步长为2
            Index(np.arange(5, dtype=np.float64)),  # 参数化测试数据：浮点64位数组成的索引
            Index(np.arange(5, dtype=np.float32)),  # 参数化测试数据：浮点32位数组成的索引
            Index(np.arange(5, dtype=np.uint64)),  # 参数化测试数据：无符号64位整数数组成的索引
            Index(range(0, 20, 2), dtype=np.int64),  # 参数化测试数据：64位有符号整数范围索引，从0到20步长为2
            Index(range(0, 20, 2), dtype=np.int32),  # 参数化测试数据：32位有符号整数范围索引，从0到20步长为2
            Index(range(0, 20, 2), dtype=np.int16),  # 参数化测试数据：16位有符号整数范围索引，从0到20步长为2
            Index(range(0, 20, 2), dtype=np.int8),   # 参数化测试数据：8位有符号整数范围索引，从0到20步长为2
        ]
    )
    def simple_index(self, request):
        return request.param

    def test_constructor_unwraps_index(self, simple_index):
        if isinstance(simple_index, RangeIndex):
            pytest.skip("Tested elsewhere.")
        index_cls = type(simple_index)  # 获取索引对象的类
        dtype = simple_index.dtype  # 获取索引对象的数据类型

        idx = Index([1, 2], dtype=dtype)  # 创建一个具有指定数据类型的索引对象
        result = index_cls(idx)  # 使用索引类将idx包装起来
        expected = np.array([1, 2], dtype=idx.dtype)  # 创建预期结果数组，数据类型与idx相同
        tm.assert_numpy_array_equal(result._data, expected)  # 断言result的数据与预期数组相等

    def test_can_hold_identifiers(self, simple_index):
        idx = simple_index  # 使用简单索引作为测试数据
        key = idx[0]  # 获取索引的第一个元素作为键
        assert idx._can_hold_identifiers_and_holds_name(key) is False  # 断言索引对象不能容纳标识符并且不保持名称为key的元素

    def test_view(self, simple_index):
        if isinstance(simple_index, RangeIndex):
            pytest.skip("Tested elsewhere.")
        index_cls = type(simple_index)  # 获取索引对象的类
        dtype = simple_index.dtype  # 获取索引对象的数据类型

        idx = index_cls([], dtype=dtype, name="Foo")  # 使用指定的数据类型和名称创建空索引对象
        idx_view = idx.view()  # 创建一个视图索引
        assert idx_view.name == "Foo"  # 断言视图索引的名称为"Foo"

        idx_view = idx.view(dtype)  # 根据指定数据类型创建视图索引
        tm.assert_index_equal(idx, index_cls(idx_view, name="Foo"), exact=True)  # 断言原索引与根据视图创建的索引相等，要求精确匹配

        msg = (
            "Cannot change data-type for array of references.|"
            "Cannot change data-type for object array.|"
        )
        with pytest.raises(TypeError, match=msg):
            # GH#55709
            idx.view(index_cls)  # 尝试使用索引类创建视图索引，预期抛出TypeError异常

    def test_insert_non_na(self, simple_index):
        # GH#43921 inserting an element that we know we can hold should
        #  not change dtype or type (except for RangeIndex)
        index = simple_index  # 使用简单索引作为测试数据

        result = index.insert(0, index[0])  # 在索引的第一个位置插入第一个元素

        expected = Index([index[0]] + list(index), dtype=index.dtype)  # 创建预期结果索引对象，保持原有数据类型
        tm.assert_index_equal(result, expected, exact=True)  # 断言插入操作后的结果与预期索引对象相等，要求精确匹配

    def test_insert_na(self, nulls_fixture, simple_index):
        # GH 18295 (test missing)
        index = simple_index  # 使用简单索引作为测试数据
        na_val = nulls_fixture  # 获取空值的测试数据

        if na_val is pd.NaT:  # 如果空值是NaT（Not a Time），则预期结果索引将使用对象数据类型
            expected = Index([index[0], pd.NaT] + list(index[1:]), dtype=object)
        else:
            expected = Index([index[0], np.nan] + list(index[1:]))  # 否则，预期结果索引将包含NaN值
            # GH#43921 we preserve float dtype
            if index.dtype.kind == "f":
                expected = Index(expected, dtype=index.dtype)  # 如果原索引的数据类型是浮点数，保持预期结果索引的浮点数数据类型

        result = index.insert(1, na_val)  # 在索引的第二个位置插入空值
        tm.assert_index_equal(result, expected, exact=True)  # 断言插入操作后的结果与预期索引对象相等，要求精确匹配
    # 定义测试方法，测试索引的算术和显式类型转换
    def test_arithmetic_explicit_conversions(self, simple_index):
        # GH 8608: GitHub issue reference
        # 如果 simple_index 的类型是 RangeIndex，则创建一个范围为 0 到 4 的 RangeIndex 对象
        index_cls = type(simple_index)
        if index_cls is RangeIndex:
            idx = RangeIndex(5)
        else:
            # 否则，根据 simple_index 的类型创建一个包含整数数据的索引对象
            idx = index_cls(np.arange(5, dtype="int64"))

        # 浮点数转换
        # 创建一个包含整数数据的 numpy 数组，然后乘以 3.2 得到浮点数数组
        arr = np.arange(5, dtype="int64") * 3.2
        # 创建预期的 Index 对象，数据类型为 np.float64
        expected = Index(arr, dtype=np.float64)
        # 将 idx 中的每个元素乘以 3.2，得到新的索引对象 fidx
        fidx = idx * 3.2
        # 断言 fidx 与预期相等
        tm.assert_index_equal(fidx, expected)
        # 将 3.2 乘以 idx 中的每个元素，得到新的索引对象 fidx
        fidx = 3.2 * idx
        # 再次断言 fidx 与预期相等
        tm.assert_index_equal(fidx, expected)

        # 与 numpy 数组的互操作
        # 创建预期的 Index 对象，数据类型为 np.float64
        expected = Index(arr, dtype=np.float64)
        # 创建全零的浮点数数组 a
        a = np.zeros(5, dtype="float64")
        # 计算 fidx 减去数组 a，得到结果 result
        result = fidx - a
        # 断言 result 与预期相等
        tm.assert_index_equal(result, expected)

        # 创建预期的 Index 对象，数据类型为 np.float64
        expected = Index(-arr, dtype=np.float64)
        # 创建全零的浮点数数组 a
        a = np.zeros(5, dtype="float64")
        # 计算数组 a 减去 fidx，得到结果 result
        result = a - fidx
        # 断言 result 与预期相等
        tm.assert_index_equal(result, expected)

    # 使用 pytest 参数化装饰器，测试将索引转换为复数类型
    @pytest.mark.parametrize("complex_dtype", [np.complex64, np.complex128])
    def test_astype_to_complex(self, complex_dtype, simple_index):
        # 将 simple_index 转换为指定的复数类型 complex_dtype
        result = simple_index.astype(complex_dtype)

        # 断言结果的类型是 Index，且数据类型是 complex_dtype
        assert type(result) is Index and result.dtype == complex_dtype

    # 测试将字符串数组转换为指定类型的索引
    def test_cast_string(self, simple_index):
        # 如果 simple_index 是 RangeIndex 类型，则跳过测试
        if isinstance(simple_index, RangeIndex):
            pytest.skip("casting of strings not relevant for RangeIndex")
        # 使用 simple_index 的类型创建一个包含字符串数据的索引对象 result
        result = type(simple_index)(["0", "1", "2"], dtype=simple_index.dtype)
        # 创建预期的 Index 对象，包含整数数据
        expected = type(simple_index)([0, 1, 2], dtype=simple_index.dtype)
        # 断言 result 与预期相等
        tm.assert_index_equal(result, expected)
```