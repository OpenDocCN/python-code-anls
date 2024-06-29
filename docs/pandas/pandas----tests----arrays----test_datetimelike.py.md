# `D:\src\scipysrc\pandas\pandas\tests\arrays\test_datetimelike.py`

```
# 从未来版本导入注解支持
from __future__ import annotations

# 导入正则表达式模块
import re
# 导入警告处理模块
import warnings

# 导入第三方库 numpy，并重命名为 np
import numpy as np
# 导入 pytest 测试框架
import pytest

# 从 pandas._libs 中导入指定类和异常
from pandas._libs import (
    NaT,  # Not a Time
    OutOfBoundsDatetime,
    Timestamp,
)
# 从 pandas._libs.tslibs 中导入时间偏移量转换工具
from pandas._libs.tslibs import to_offset
# 从 pandas.compat.numpy 中导入 numpy 版本比较工具
from pandas.compat.numpy import np_version_gt2

# 从 pandas.core.dtypes.dtypes 中导入 PeriodDtype 类
from pandas.core.dtypes.dtypes import PeriodDtype

# 导入 pandas 并重命名为 pd
import pandas as pd
# 从 pandas 中导入时间索引相关的类
from pandas import (
    DatetimeIndex,
    Period,
    PeriodIndex,
    TimedeltaIndex,
)
# 导入 pandas 测试工具模块，并重命名为 tm
import pandas._testing as tm
# 从 pandas.core.arrays 中导入时间数组相关的类
from pandas.core.arrays import (
    DatetimeArray,
    NumpyExtensionArray,
    PeriodArray,
    TimedeltaArray,
)


# TODO: 更多的频率变体
@pytest.fixture(params=["D", "B", "W", "ME", "QE", "YE"])
def freqstr(request):
    """Fixture returning parametrized frequency in string format."""
    return request.param


@pytest.fixture
def period_index(freqstr):
    """
    A fixture to provide PeriodIndex objects with different frequencies.

    Most PeriodArray behavior is already tested in PeriodIndex tests,
    so here we just test that the PeriodArray behavior matches
    the PeriodIndex behavior.
    """
    # TODO: non-monotone indexes; NaTs, different start dates
    with warnings.catch_warnings():
        # suppress deprecation of Period[B]
        warnings.filterwarnings(
            "ignore", message="Period with BDay freq", category=FutureWarning
        )
        # 转换频率字符串为 PeriodDtype 对象的频率字符串
        freqstr = PeriodDtype(to_offset(freqstr))._freqstr
        # 创建一个包含指定频率的 PeriodIndex 对象
        pi = pd.period_range(start=Timestamp("2000-01-01"), periods=100, freq=freqstr)
    return pi


@pytest.fixture
def datetime_index(freqstr):
    """
    A fixture to provide DatetimeIndex objects with different frequencies.

    Most DatetimeArray behavior is already tested in DatetimeIndex tests,
    so here we just test that the DatetimeArray behavior matches
    the DatetimeIndex behavior.
    """
    # TODO: non-monotone indexes; NaTs, different start dates, timezones
    # 创建一个包含指定频率的 DatetimeIndex 对象
    dti = pd.date_range(start=Timestamp("2000-01-01"), periods=100, freq=freqstr)
    return dti


@pytest.fixture
def timedelta_index():
    """
    A fixture to provide TimedeltaIndex objects with different frequencies.

    Most TimedeltaArray behavior is already tested in TimedeltaIndex tests,
    so here we just test that the TimedeltaArray behavior matches
    the TimedeltaIndex behavior.
    """
    # TODO: flesh this out
    # 创建一个包含不同频率的 TimedeltaIndex 对象
    return TimedeltaIndex(["1 Day", "3 Hours", "NaT"])


class SharedTests:
    # 定义一个类属性 index_cls，用于存储 DatetimeIndex、PeriodIndex 或 TimedeltaIndex 类型
    index_cls: type[DatetimeIndex | PeriodIndex | TimedeltaIndex]

    @pytest.fixture
    def arr1d(self):
        """Fixture returning DatetimeArray with daily frequency."""
        # 创建一个包含每日频率的 DatetimeArray 对象
        data = np.arange(10, dtype="i8") * 24 * 3600 * 10**9
        if self.array_cls is PeriodArray:
            # 如果 array_cls 是 PeriodArray 类型，则使用每日频率创建 PeriodArray 对象
            arr = self.array_cls(data, freq="D")
        else:
            # 否则，使用每日频率创建相应类型的时间索引对象，并获取其内部数据
            arr = self.index_cls(data, freq="D")._data
        return arr
    def test_compare_len1_raises(self, arr1d):
        # 定义测试函数，验证当比较不同长度的数组时是否引发异常
        arr = arr1d
        # 使用传入的一维数组初始化 arr 变量
        idx = self.index_cls(arr)

        with pytest.raises(ValueError, match="Lengths must match"):
            # 使用 pytest 验证比较时是否引发 ValueError 异常，匹配异常信息 "Lengths must match"
            arr == arr[:1]

        # 同时测试索引类的情况，参见 GitHub 问题 #23078
        with pytest.raises(ValueError, match="Lengths must match"):
            # 使用 pytest 验证比较时是否引发 ValueError 异常，匹配异常信息 "Lengths must match"
            idx <= idx[[0]]

    @pytest.mark.parametrize(
        "result",
        [
            pd.date_range("2020", periods=3),
            pd.date_range("2020", periods=3, tz="UTC"),
            pd.timedelta_range("0 days", periods=3),
            pd.period_range("2020Q1", periods=3, freq="Q"),
        ],
    )
    def test_compare_with_Categorical(self, result):
        # 定义测试函数，验证与 Categorical 类型的比较操作
        expected = pd.Categorical(result)
        assert all(result == expected)
        assert not any(result != expected)

    @pytest.mark.parametrize("reverse", [True, False])
    @pytest.mark.parametrize("as_index", [True, False])
    def test_compare_categorical_dtype(self, arr1d, as_index, reverse, ordered):
        # 定义测试函数，验证分类数据类型的比较操作
        other = pd.Categorical(arr1d, ordered=ordered)
        if as_index:
            other = pd.CategoricalIndex(other)

        left, right = arr1d, other
        if reverse:
            left, right = right, left

        ones = np.ones(arr1d.shape, dtype=bool)
        zeros = ~ones

        result = left == right
        tm.assert_numpy_array_equal(result, ones)

        result = left != right
        tm.assert_numpy_array_equal(result, zeros)

        if not reverse and not as_index:
            # 否则，由于不是有序的，Categorical 会引发 TypeError
            # TODO: 可能应该无论如何都获得相同的行为？
            result = left < right
            tm.assert_numpy_array_equal(result, zeros)

            result = left <= right
            tm.assert_numpy_array_equal(result, ones)

            result = left > right
            tm.assert_numpy_array_equal(result, zeros)

            result = left >= right
            tm.assert_numpy_array_equal(result, ones)

    def test_take(self):
        # 定义测试函数，验证 take 方法的使用
        data = np.arange(100, dtype="i8") * 24 * 3600 * 10**9
        np.random.default_rng(2).shuffle(data)

        if self.array_cls is PeriodArray:
            arr = PeriodArray(data, dtype="period[D]")
        else:
            arr = self.index_cls(data)._data
        idx = self.index_cls._simple_new(arr)

        takers = [1, 4, 94]
        result = arr.take(takers)
        expected = idx.take(takers)

        tm.assert_index_equal(self.index_cls(result), expected)

        takers = np.array([1, 4, 94])
        result = arr.take(takers)
        expected = idx.take(takers)

        tm.assert_index_equal(self.index_cls(result), expected)

    @pytest.mark.parametrize("fill_value", [2, 2.0, Timestamp(2021, 1, 1, 12).time])
    # 测试函数，用于测试在取值操作时当填充值引发异常的情况
    def test_take_fill_raises(self, fill_value, arr1d):
        # 构造异常消息，指明期望的值类型或 NaT 类型
        msg = f"value should be a '{arr1d._scalar_type.__name__}' or 'NaT'. Got"
        # 断言在使用 allow_fill=True 且指定填充值时抛出 TypeError 异常，并匹配消息内容
        with pytest.raises(TypeError, match=msg):
            arr1d.take([0, 1], allow_fill=True, fill_value=fill_value)

    # 测试函数，用于测试在取值操作时的填充功能
    def test_take_fill(self, arr1d):
        # 将 arr1d 赋值给 arr
        arr = arr1d

        # 测试使用 None 作为填充值时的结果
        result = arr.take([-1, 1], allow_fill=True, fill_value=None)
        assert result[0] is NaT

        # 测试使用 np.nan 作为填充值时的结果
        result = arr.take([-1, 1], allow_fill=True, fill_value=np.nan)
        assert result[0] is NaT

        # 测试使用 NaT 作为填充值时的结果
        result = arr.take([-1, 1], allow_fill=True, fill_value=NaT)
        assert result[0] is NaT

    # 测试函数，用于测试在取值操作时填充字符串类型的功能
    @pytest.mark.filterwarnings(
        "ignore:Period with BDay freq is deprecated:FutureWarning"
    )
    def test_take_fill_str(self, arr1d):
        # 使用 str(arr1d[-1]) 作为填充值，与其他填充值取值方法保持一致
        result = arr1d.take([-1, 1], allow_fill=True, fill_value=str(arr1d[-1]))
        expected = arr1d[[-1, 1]]
        # 断言 result 与 expected 相等
        tm.assert_equal(result, expected)

        # 构造异常消息，指明期望的值类型或 NaT 类型
        msg = f"value should be a '{arr1d._scalar_type.__name__}' or 'NaT'. Got"
        # 断言在使用 allow_fill=True 且填充值为 "foo" 时抛出 TypeError 异常，并匹配消息内容
        with pytest.raises(TypeError, match=msg):
            arr1d.take([-1, 1], allow_fill=True, fill_value="foo")

    # 测试函数，用于测试相同类型数组的连接操作
    def test_concat_same_type(self, arr1d):
        # 将 arr1d 赋值给 arr
        arr = arr1d
        # 使用 self.index_cls 创建索引对象 idx，并在索引位置 0 插入 NaT
        idx = self.index_cls(arr)
        idx = idx.insert(0, NaT)
        # 再次将 arr1d 赋值给 arr
        arr = arr1d

        # 执行 arr 的 _concat_same_type 方法，将结果赋给 result
        result = arr._concat_same_type([arr[:-1], arr[1:], arr])
        # 将 arr 转换为对象类型 arr2
        arr2 = arr.astype(object)
        # 使用 self.index_cls 创建期望的索引对象 expected
        expected = self.index_cls(np.concatenate([arr2[:-1], arr2[1:], arr2]))

        # 断言 result 经过 self.index_cls 处理后与 expected 相等
        tm.assert_index_equal(self.index_cls(result), expected)

    # 测试函数，用于测试标量值的解包操作
    def test_unbox_scalar(self, arr1d):
        # 对 arr1d[0] 进行解包操作，将结果赋给 result
        result = arr1d._unbox_scalar(arr1d[0])
        # 获取 arr1d._ndarray 的数据类型，期望其类型为 expected
        expected = arr1d._ndarray.dtype.type
        # 断言 result 的类型为 expected
        assert isinstance(result, expected)

        # 对 NaT 进行解包操作，将结果赋给 result
        result = arr1d._unbox_scalar(NaT)
        # 断言 result 的类型为 expected
        assert isinstance(result, expected)

        # 构造异常消息，指明期望的值类型
        msg = f"'value' should be a {self.scalar_type.__name__}."
        # 断言在对 "foo" 进行解包操作时抛出 ValueError 异常，并匹配消息内容
        with pytest.raises(ValueError, match=msg):
            arr1d._unbox_scalar("foo")

    # 测试函数，用于测试与给定值兼容性检查操作
    def test_check_compatible_with(self, arr1d):
        # 检查 arr1d[0] 与 arr1d 的兼容性
        arr1d._check_compatible_with(arr1d[0])
        # 检查 arr1d[:1] 与 arr1d 的兼容性
        arr1d._check_compatible_with(arr1d[:1])
        # 检查 NaT 与 arr1d 的兼容性
        arr1d._check_compatible_with(NaT)

    # 测试函数，用于测试从字符串转换为标量值的操作
    def test_scalar_from_string(self, arr1d):
        # 将 arr1d[0] 转换为字符串后再转换回标量值，将结果赋给 result
        result = arr1d._scalar_from_string(str(arr1d[0]))
        # 断言 result 与 arr1d[0] 相等
        assert result == arr1d[0]

    # 测试函数，用于测试非法操作下的缩减操作
    def test_reduce_invalid(self, arr1d):
        # 构造异常消息，指明非法操作类型
        msg = "does not support operation 'not a method'"
        # 断言在执行非法操作时抛出 TypeError 异常，并匹配消息内容
        with pytest.raises(TypeError, match=msg):
            arr1d._reduce("not a method")

    # 使用 pytest.mark.parametrize 标记的测试函数，参数化测试填充方法为 "pad" 和 "backfill"
    # 测试fillna方法是否会修改原始数组
    def test_fillna_method_doesnt_change_orig(self, method):
        # 创建一个包含从0到9的整数数组，并将其转换为纳秒级的时间戳
        data = np.arange(10, dtype="i8") * 24 * 3600 * 10**9
        # 根据self.array_cls的类型选择不同的数组构造方法
        if self.array_cls is PeriodArray:
            arr = self.array_cls(data, dtype="period[D]")
        else:
            arr = self.array_cls._from_sequence(data)
        # 将索引为4的元素设置为NaT（Not a Time）
        arr[4] = NaT

        # 根据method参数选择填充值
        fill_value = arr[3] if method == "pad" else arr[5]

        # 调用_pad_or_backfill方法填充或向后填充数组
        result = arr._pad_or_backfill(method=method)
        # 断言填充后索引为4的元素与预期的填充值相同
        assert result[4] == fill_value

        # 检查原始数组中索引为4的元素未被修改
        assert arr[4] is NaT

    # 测试searchsorted方法
    def test_searchsorted(self):
        # 创建一个包含从0到9的整数数组，并将其转换为纳秒级的时间戳
        data = np.arange(10, dtype="i8") * 24 * 3600 * 10**9
        # 根据self.array_cls的类型选择不同的数组构造方法
        if self.array_cls is PeriodArray:
            arr = self.array_cls(data, dtype="period[D]")
        else:
            arr = self.array_cls._from_sequence(data)

        # 对标量进行搜索
        result = arr.searchsorted(arr[1])
        # 断言搜索到的位置为1
        assert result == 1

        # 使用参数side="right"进行搜索
        result = arr.searchsorted(arr[2], side="right")
        # 断言搜索到的位置为3
        assert result == 3

        # 对数组切片进行搜索
        result = arr.searchsorted(arr[1:3])
        expected = np.array([1, 2], dtype=np.intp)
        # 断言搜索结果与预期结果相同
        tm.assert_numpy_array_equal(result, expected)

        # 使用参数side="right"对数组切片进行搜索
        result = arr.searchsorted(arr[1:3], side="right")
        expected = np.array([2, 3], dtype=np.intp)
        # 断言搜索结果与预期结果相同
        tm.assert_numpy_array_equal(result, expected)

        # 根据 GH#29884，匹配numpy约定NaT是在末尾还是开头
        result = arr.searchsorted(NaT)
        # 断言搜索到的位置为10
        assert result == 10

    # 使用@pytest.mark.parametrize注解进行参数化测试
    @pytest.mark.parametrize("box", [None, "index", "series"])
    def test_searchsorted_castable_strings(self, arr1d, box, string_storage):
        # 将arr1d赋值给arr
        arr = arr1d
        # 根据box参数的值选择不同的处理方式
        if box is None:
            pass
        elif box == "index":
            # 在此处测试等效的Index.searchsorted方法
            arr = self.index_cls(arr)
        else:
            # 在此处测试等效的Series.searchsorted方法
            arr = pd.Series(arr)

        # 对标量进行搜索
        result = arr.searchsorted(str(arr[1]))
        # 断言搜索到的位置为1
        assert result == 1

        # 使用参数side="right"对标量进行搜索
        result = arr.searchsorted(str(arr[2]), side="right")
        # 断言搜索到的位置为3
        assert result == 3

        # 对字符串数组进行搜索
        result = arr.searchsorted([str(x) for x in arr[1:3]])
        expected = np.array([1, 2], dtype=np.intp)
        # 断言搜索结果与预期结果相同
        tm.assert_numpy_array_equal(result, expected)

        # 测试传入非法值时的异常情况
        with pytest.raises(
            TypeError,
            match=re.escape(
                f"value should be a '{arr1d._scalar_type.__name__}', 'NaT', "
                "or array of those. Got 'str' instead."
            ),
        ):
            arr.searchsorted("foo")

        # 测试传入字符串数组时的异常情况
        with pd.option_context("string_storage", string_storage):
            with pytest.raises(
                TypeError,
                match=re.escape(
                    f"value should be a '{arr1d._scalar_type.__name__}', 'NaT', "
                    "or array of those. Got string array instead."
                ),
            ):
                arr.searchsorted([str(arr[1]), "baz"])
    # 测试针对接近边界的索引行为
    def test_getitem_near_implementation_bounds(self):
        # 由于不同时区的边界略有不同，我们仅检查非时区敏感的情况
        i8vals = np.asarray([NaT._value + n for n in range(1, 5)], dtype="i8")
        
        # 根据实际的数组类别选择创建数组对象
        if self.array_cls is PeriodArray:
            arr = self.array_cls(i8vals, dtype="period[ns]")
        else:
            arr = self.index_cls(i8vals, freq="ns")._data
        
        # 对数组的索引访问，预期不会引发OutOfBoundsDatetime异常
        arr[0]
        
        # 将数组转换为索引对象
        index = pd.Index(arr)
        index[0]  # 预期不会引发OutOfBoundsDatetime异常
        
        # 将数组转换为序列对象
        ser = pd.Series(arr)
        ser[0]  # 预期不会引发OutOfBoundsDatetime异常

    # 测试二维数组的索引行为
    def test_getitem_2d(self, arr1d):
        # 在一维数组上进行二维切片
        expected = type(arr1d)._simple_new(
            arr1d._ndarray[:, np.newaxis], dtype=arr1d.dtype
        )
        result = arr1d[:, np.newaxis]
        tm.assert_equal(result, expected)

        # 对二维数组进行查找
        arr2d = expected
        expected = type(arr2d)._simple_new(arr2d._ndarray[:3, 0], dtype=arr2d.dtype)
        result = arr2d[:3, 0]
        tm.assert_equal(result, expected)

        # 标量查找
        result = arr2d[-1, 0]
        expected = arr1d[-1]
        assert result == expected

    # 测试二维迭代行为
    def test_iter_2d(self, arr1d):
        # 将一维数组转换为二维数据
        data2d = arr1d._ndarray[:3, np.newaxis]
        arr2d = type(arr1d)._simple_new(data2d, dtype=arr1d.dtype)
        
        # 迭代二维数组，验证每个元素为一维数组且类型、维度、dtype正确
        result = list(arr2d)
        assert len(result) == 3
        for x in result:
            assert isinstance(x, type(arr1d))
            assert x.ndim == 1
            assert x.dtype == arr1d.dtype

    # 测试二维数组的字符串表示形式
    def test_repr_2d(self, arr1d):
        # 将一维数组转换为二维数据
        data2d = arr1d._ndarray[:3, np.newaxis]
        arr2d = type(arr1d)._simple_new(data2d, dtype=arr1d.dtype)

        # 获取二维数组的字符串表示形式，并验证其与预期相符
        result = repr(arr2d)
        if isinstance(arr2d, TimedeltaArray):
            expected = (
                f"<{type(arr2d).__name__}>\n"
                "[\n"
                f"['{arr1d[0]._repr_base()}'],\n"
                f"['{arr1d[1]._repr_base()}'],\n"
                f"['{arr1d[2]._repr_base()}']\n"
                "]\n"
                f"Shape: (3, 1), dtype: {arr1d.dtype}"
            )
        else:
            expected = (
                f"<{type(arr2d).__name__}>\n"
                "[\n"
                f"['{arr1d[0]}'],\n"
                f"['{arr1d[1]}'],\n"
                f"['{arr1d[2]}']\n"
                "]\n"
                f"Shape: (3, 1), dtype: {arr1d.dtype}"
            )
        assert result == expected
    # 定义一个测试方法，用于测试设置元素的操作
    def test_setitem(self):
        # 创建一个包含从0到9的整数的NumPy数组，并将其乘以24小时、3600秒和10^9纳秒，得到时间戳数据
        data = np.arange(10, dtype="i8") * 24 * 3600 * 10**9
        
        # 根据类别选择创建不同的数组对象
        if self.array_cls is PeriodArray:
            # 如果self.array_cls是PeriodArray类，则创建PeriodArray对象
            arr = self.array_cls(data, dtype="period[D]")
        else:
            # 否则，创建一个带有频率“D”的索引类对象，并使用其内部数据
            arr = self.index_cls(data, freq="D")._data

        # 修改数组的第一个元素为第二个元素的值
        arr[0] = arr[1]
        
        # 创建期望的NumPy数组，用于验证修改操作的预期结果
        expected = np.arange(10, dtype="i8") * 24 * 3600 * 10**9
        expected[0] = expected[1]

        # 使用测试工具函数验证修改后的数组是否符合预期
        tm.assert_numpy_array_equal(arr.asi8, expected)

        # 将数组的前两个元素设置为其倒数两个元素的值
        arr[:2] = arr[-2:]
        expected[:2] = expected[-2:]
        
        # 再次使用测试工具函数验证修改后的数组是否符合新的预期
        tm.assert_numpy_array_equal(arr.asi8, expected)

    # 使用pytest的参数化装饰器，定义一个测试方法，测试对象数据类型为对象的设置操作
    @pytest.mark.parametrize(
        "box",
        [
            pd.Index,                    # Pandas的索引对象
            pd.Series,                   # Pandas的系列对象
            np.array,                    # NumPy数组
            list,                        # 列表
            NumpyExtensionArray,         # 自定义的NumPy扩展数组
        ],
    )
    def test_setitem_object_dtype(self, box, arr1d):
        # 复制一维数组并反转顺序作为预期结果
        expected = arr1d.copy()[::-1]
        
        # 如果预期结果的数据类型是日期时间相关的，则将其频率设置为None
        if expected.dtype.kind in ["m", "M"]:
            expected = expected._with_freq(None)

        # 根据不同的数据类型(box)，转换预期结果的数据值
        vals = expected
        if box is list:
            vals = list(vals)
        elif box is np.array:
            # 如果转换成np.array并且转换为对象类型，则dt64和td64会转换为整数
            vals = np.array(vals.astype(object))
        elif box is NumpyExtensionArray:
            vals = box(np.asarray(vals, dtype=object))
        else:
            vals = box(vals).astype(object)

        # 将数组设置为转换后的值
        arr1d[:] = vals

        # 使用测试工具函数验证数组是否与预期结果相等
        tm.assert_equal(arr1d, expected)

    # 定义一个测试方法，测试设置字符串元素的操作
    def test_setitem_strs(self, arr1d):
        # 检查是否能够解析标量和列表中的字符串

        # 设置列表形式的字符串
        expected = arr1d.copy()
        expected[[0, 1]] = arr1d[-2:]

        result = arr1d.copy()
        result[:2] = [str(x) for x in arr1d[-2:]]
        
        # 使用测试工具函数验证结果数组与预期数组是否相等
        tm.assert_equal(result, expected)

        # 设置标量字符串
        expected = arr1d.copy()
        expected[0] = arr1d[-1]

        result = arr1d.copy()
        result[0] = str(arr1d[-1])
        
        # 再次使用测试工具函数验证结果数组与预期数组是否相等
        tm.assert_equal(result, expected)

    # 使用pytest的参数化装饰器，定义一个测试方法，测试分类数据类型的设置操作
    @pytest.mark.parametrize("as_index", [True, False])
    def test_setitem_categorical(self, arr1d, as_index):
        # 复制一维数组并反转顺序作为预期结果
        expected = arr1d.copy()[::-1]
        
        # 如果预期结果不是PeriodArray类的实例，则设置其频率为None
        if not isinstance(expected, PeriodArray):
            expected = expected._with_freq(None)

        # 创建一个Pandas的分类对象
        cat = pd.Categorical(arr1d)
        
        # 根据参数as_index的值，将分类对象转换为分类索引对象
        if as_index:
            cat = pd.CategoricalIndex(cat)

        # 将数组设置为反转的分类对象
        arr1d[:] = cat[::-1]
        
        # 使用测试工具函数验证数组是否与预期结果相等
        tm.assert_equal(arr1d, expected)
    # 定义测试方法，验证在设置超出边界索引时是否引发 IndexError 异常
    def test_setitem_raises(self, arr1d):
        # 从 arr1d 中取出前 10 个元素创建 arr 数组
        arr = arr1d[:10]
        # 获取 arr 中第一个元素的值
        val = arr[0]

        # 断言设置超出边界索引 12 时会引发 IndexError 异常，并匹配特定错误信息
        with pytest.raises(IndexError, match="index 12 is out of bounds"):
            arr[12] = val

        # 断言设置不符合类型要求的值时会引发 TypeError 异常，并匹配特定错误信息
        with pytest.raises(TypeError, match="value should be a.* 'object'"):
            arr[0] = object()

        # 准备错误信息字符串
        msg = "cannot set using a list-like indexer with a different length"
        # 断言使用空列表作为索引器设置元素会引发 ValueError 异常，并匹配特定错误信息
        with pytest.raises(ValueError, match=msg):
            # GH#36339
            arr[[]] = [arr[1]]

        # 准备错误信息字符串
        msg = "cannot set using a slice indexer with a different length than"
        # 断言使用长度不同的切片索引器设置元素会引发 ValueError 异常，并匹配特定错误信息
        with pytest.raises(ValueError, match=msg):
            # GH#36339
            arr[1:1] = arr[:3]

    # 使用参数化装饰器标记多组测试参数，对不同的箱子（box）进行测试
    @pytest.mark.parametrize("box", [list, np.array, pd.Index, pd.Series])
    # 定义测试方法，验证在设置不合规的数值类型时是否引发 TypeError 异常
    def test_setitem_numeric_raises(self, arr1d, box):
        # 准备错误信息字符串，指明期望的数据类型
        msg = (
            f"value should be a '{arr1d._scalar_type.__name__}', "
            "'NaT', or array of those. Got"
        )

        # 断言设置不合规的 int 数组时会引发 TypeError 异常，并匹配特定错误信息
        with pytest.raises(TypeError, match=msg):
            arr1d[:2] = box([0, 1])

        # 断言设置不合规的 float 数组时会引发 TypeError 异常，并匹配特定错误信息
        with pytest.raises(TypeError, match=msg):
            arr1d[:2] = box([0.0, 1.0])

    # 定义测试方法，验证就地算术操作是否正确
    def test_inplace_arithmetic(self):
        # 准备一组数据，类型为 int64，表示以秒为单位的时间戳
        data = np.arange(10, dtype="i8") * 24 * 3600 * 10**9
        # 根据测试环境确定使用 PeriodArray 或 Index 类创建 arr 数组
        if self.array_cls is PeriodArray:
            arr = self.array_cls(data, dtype="period[D]")
        else:
            arr = self.index_cls(data, freq="D")._data

        # 预期的结果是 arr 的每个元素加上一天
        expected = arr + pd.Timedelta(days=1)
        # 执行就地加法操作，修改 arr 的每个元素加上一天
        arr += pd.Timedelta(days=1)
        # 断言 arr 的结果与预期的结果相等
        tm.assert_equal(arr, expected)

        # 预期的结果是 arr 的每个元素减去一天
        expected = arr - pd.Timedelta(days=1)
        # 执行就地减法操作，修改 arr 的每个元素减去一天
        arr -= pd.Timedelta(days=1)
        # 断言 arr 的结果与预期的结果相等
        tm.assert_equal(arr, expected)

    # 定义测试方法，验证在使用过时的整数填充参数时是否引发 TypeError 异常
    def test_shift_fill_int_deprecated(self, arr1d):
        # 准备错误信息字符串，指明填充值应该是什么类型
        with pytest.raises(TypeError, match="value should be a"):
            # 调用 shift 方法时传入整数填充值 1，预期会引发 TypeError 异常
            arr1d.shift(1, fill_value=1)
    # 定义一个测试函数，用于测试一维数组的中位数计算
    def test_median(self, arr1d):
        # 将输入的一维数组赋值给新变量 arr
        arr = arr1d
        # 如果数组长度为偶数，将 arr 截取为长度减一的子数组，使其变为奇数长度
        if len(arr) % 2 == 0:
            arr = arr[:-1]

        # 计算奇数长度数组的中位数的预期值
        expected = arr[len(arr) // 2]

        # 调用数组对象的 median() 方法计算中位数，并将结果赋给 result
        result = arr.median()
        # 断言 result 的类型与 expected 的类型相同
        assert type(result) is type(expected)
        # 断言 result 的值与 expected 的值相等
        assert result == expected

        # 将中位数位置的值设为 NaT
        arr[len(arr) // 2] = NaT
        # 如果 expected 不是 Period 类型的实例，则重新计算 expected 作为中位数
        if not isinstance(expected, Period):
            expected = arr[len(arr) // 2 - 1 : len(arr) // 2 + 2].mean()

        # 断言在不跳过 NaN 值的情况下，数组的中位数是 NaT
        assert arr.median(skipna=False) is NaT

        # 再次计算数组的中位数并断言其类型与 expected 的类型相同
        result = arr.median()
        assert type(result) is type(expected)
        # 断言计算出的中位数与 expected 的值相等
        assert result == expected

        # 断言空数组的中位数是 NaT
        assert arr[:0].median() is NaT
        # 断言在不跳过 NaN 值的情况下，空数组的中位数是 NaT
        assert arr[:0].median(skipna=False) is NaT

        # 对二维情况进行测试
        # 将一维数组 arr 重塑为二维数组 arr2，形状为 (-1, 1)
        arr2 = arr.reshape(-1, 1)

        # 计算二维数组 arr2 的全局中位数，并将结果赋给 result
        result = arr2.median(axis=None)
        # 断言 result 的类型与 expected 的类型相同
        assert type(result) is type(expected)
        # 断言计算出的全局中位数与 expected 的值相等
        assert result == expected

        # 断言在不跳过 NaN 值的情况下，计算二维数组 arr2 的全局中位数是 NaT
        assert arr2.median(axis=None, skipna=False) is NaT

        # 计算二维数组 arr2 沿 axis=0 的中位数，并将结果赋给 result
        result = arr2.median(axis=0)
        # 创建一个与 expected 相同类型的数组对象，包含单个值 expected2
        expected2 = type(arr)._from_sequence([expected], dtype=arr.dtype)
        # 使用测试工具库 tm 进行结果断言
        tm.assert_equal(result, expected2)

        # 在不跳过 NaN 值的情况下，计算二维数组 arr2 沿 axis=0 的中位数，并将结果赋给 result
        result = arr2.median(axis=0, skipna=False)
        # 创建一个与 expected2 相同类型的数组对象，包含单个值 NaT
        expected2 = type(arr)._from_sequence([NaT], dtype=arr.dtype)
        # 使用测试工具库 tm 进行结果断言
        tm.assert_equal(result, expected2)

        # 计算二维数组 arr2 沿 axis=1 的中位数，并使用测试工具库 tm 进行结果断言
        result = arr2.median(axis=1)
        tm.assert_equal(result, arr)

        # 在不跳过 NaN 值的情况下，计算二维数组 arr2 沿 axis=1 的中位数，并使用测试工具库 tm 进行结果断言
        result = arr2.median(axis=1, skipna=False)
        tm.assert_equal(result, arr)

    # 测试从整数数组创建扩展数组的函数
    def test_from_integer_array(self):
        # 创建一个包含整数 [1, 2, 3] 的 numpy 数组 arr
        arr = np.array([1, 2, 3], dtype=np.int64)
        # 使用 pandas 中的 array 函数将 arr 转换为数据类型为 "Int64" 的扩展数组对象 data
        data = pd.array(arr, dtype="Int64")

        # 根据实例的数组类别选择不同的预期结果和计算结果
        if self.array_cls is PeriodArray:
            # 如果数组类别是 PeriodArray，则使用整数数组 arr 创建类型和数据类型为 self.example_dtype 的预期结果和计算结果
            expected = self.array_cls(arr, dtype=self.example_dtype)
            result = self.array_cls(data, dtype=self.example_dtype)
        else:
            # 如果数组类别不是 PeriodArray，则使用序列 arr 创建类型和数据类型为 self.example_dtype 的预期结果和计算结果
            expected = self.array_cls._from_sequence(arr, dtype=self.example_dtype)
            result = self.array_cls._from_sequence(data, dtype=self.example_dtype)

        # 使用测试工具库 tm 断言计算结果 result 与预期结果 expected 相等
        tm.assert_extension_array_equal(result, expected)
# 创建一个测试类 TestDatetimeArray，继承自 SharedTests，用于测试日期时间索引和日期时间数组的功能
class TestDatetimeArray(SharedTests):
    # 指定索引类为 DatetimeIndex，用于测试日期时间索引
    index_cls = DatetimeIndex
    # 指定数组类为 DatetimeArray，用于测试日期时间数组
    array_cls = DatetimeArray
    # 指定标量类型为 Timestamp，用于测试时间戳的功能
    scalar_type = Timestamp
    # 定义示例的数据类型为 "M8[ns]"，表示日期时间的精度为纳秒级
    example_dtype = "M8[ns]"

    @pytest.fixture
    def arr1d(self, tz_naive_fixture, freqstr):
        """
        Fixture returning DatetimeArray with parametrized frequency and
        timezones
        """
        # 使用 pytest 的 fixture 装饰器，返回一个带有参数化频率和时区的 DatetimeArray
        tz = tz_naive_fixture
        # 创建一个日期时间索引，从 "2016-01-01 01:01:00" 开始，频率由参数 freqstr 指定，时区为 tz
        dti = pd.date_range("2016-01-01 01:01:00", periods=5, freq=freqstr, tz=tz)
        # 获取日期时间索引的底层数据
        dta = dti._data
        # 返回日期时间数组的底层数据作为 fixture
        return dta

    def test_round(self, arr1d):
        # GH#24064
        # 创建一个日期时间索引对象 dti，使用 arr1d 作为其底层数据
        dti = self.index_cls(arr1d)

        # 对日期时间索引进行舍入，将日期时间舍入到最接近的 2 分钟
        result = dti.round(freq="2min")
        # 期望的结果是每个日期时间减去 1 分钟
        expected = dti - pd.Timedelta(minutes=1)
        # 移除结果中的频率信息
        expected = expected._with_freq(None)
        # 断言索引结果与期望值相等
        tm.assert_index_equal(result, expected)

        # 获取日期时间数组的底层数据
        dta = dti._data
        # 对日期时间数组进行舍入，将日期时间舍入到最接近的 2 分钟
        result = dta.round(freq="2min")
        # 移除期望值的底层数据的频率信息
        expected = expected._data._with_freq(None)
        # 断言日期时间数组结果与期望值相等
        tm.assert_datetime_array_equal(result, expected)

    def test_array_interface(self, datetime_index):
        # 获取日期时间索引的底层数据
        arr = datetime_index._data
        # 如果 numpy 版本大于 2，则将 copy_false 设置为 None，否则设置为 False
        copy_false = None if np_version_gt2 else False

        # 默认情况下，asarray 返回相同的底层数据（对于时区无关的情况）
        result = np.asarray(arr)
        expected = arr._ndarray
        # 断言结果与期望值的身份相同
        assert result is expected
        # 断言 numpy 数组的内容相等
        tm.assert_numpy_array_equal(result, expected)
        
        # 使用 copy 参数为 False，从日期时间数组创建一个 numpy 数组
        result = np.array(arr, copy=copy_false)
        # 断言结果与期望值的身份相同
        assert result is expected
        # 断言 numpy 数组的内容相等
        tm.assert_numpy_array_equal(result, expected)

        # 指定 dtype 为 "datetime64[ns]"，返回与默认情况相同的结果
        result = np.asarray(arr, dtype="datetime64[ns]")
        # 断言结果与期望值的身份相同
        assert result is expected
        # 断言 numpy 数组的内容相等
        tm.assert_numpy_array_equal(result, expected)
        
        # 使用 copy 参数为 False，从日期时间数组创建一个 numpy 数组
        result = np.array(arr, dtype="datetime64[ns]", copy=copy_false)
        # 断言结果与期望值的身份相同
        assert result is expected
        # 断言 numpy 数组的内容相等
        tm.assert_numpy_array_equal(result, expected)
        
        # 指定 dtype 为 "datetime64[ns]"，如果 numpy 版本不大于 2
        if not np_version_gt2:
            # TODO: GH 57739
            # 断言结果不等于期望值
            assert result is not expected
        # 断言 numpy 数组的内容相等
        tm.assert_numpy_array_equal(result, expected)

        # 将 dtype 指定为 object 类型
        result = np.asarray(arr, dtype=object)
        # 从日期时间数组创建一个 object 类型的 numpy 数组
        expected = np.array(list(arr), dtype=object)
        # 断言 numpy 数组的内容相等
        tm.assert_numpy_array_equal(result, expected)

        # 将 dtype 指定为其他类型，始终会复制数据
        result = np.asarray(arr, dtype="int64")
        # 断言结果不等于 arr 的 asi8 属性
        assert result is not arr.asi8
        # 断言 arr 和 result 不共享内存
        assert not np.may_share_memory(arr, result)
        # 获取 arr.asi8 的复制品作为期望值
        expected = arr.asi8.copy()
        # 断言 numpy 数组的内容相等
        tm.assert_numpy_array_equal(result, expected)

        # 其他 dtype 类型由 numpy 处理
        for dtype in ["float64", str]:
            result = np.asarray(arr, dtype=dtype)
            expected = np.asarray(arr).astype(dtype)
            # 断言 numpy 数组的内容相等
            tm.assert_numpy_array_equal(result, expected)
    # 测试数组对象数据类型的方法
    def test_array_object_dtype(self, arr1d):
        # GH#23524
        # 将输入的一维数组赋给arr变量
        arr = arr1d
        # 使用self.index_cls类的构造方法创建一个时间索引对象dti
        dti = self.index_cls(arr1d)

        # 创建一个预期结果数组，内容为dti对象的列表形式转换为NumPy数组
        expected = np.array(list(dti))

        # 使用对象数据类型创建一个新的NumPy数组
        result = np.array(arr, dtype=object)
        # 使用tm.assert_numpy_array_equal函数比较结果数组和预期数组
        tm.assert_numpy_array_equal(result, expected)

        # 同时测试DatetimeIndex的方法
        result = np.array(dti, dtype=object)
        tm.assert_numpy_array_equal(result, expected)

    # 测试数组时区的方法
    def test_array_tz(self, arr1d):
        # GH#23524
        # 将输入的一维数组赋给arr变量
        arr = arr1d
        # 使用self.index_cls类的构造方法创建一个时间索引对象dti
        dti = self.index_cls(arr1d)
        # 根据NumPy版本选择是否复制为False
        copy_false = None if np_version_gt2 else False

        # 创建一个预期结果数组，内容为dti对象转换为'M8[ns]'数据类型的视图
        expected = dti.asi8.view("M8[ns]")
        # 使用'M8[ns]'数据类型创建一个新的NumPy数组
        result = np.array(arr, dtype="M8[ns]")
        tm.assert_numpy_array_equal(result, expected)

        # 使用'datetime64[ns]'数据类型创建一个新的NumPy数组
        result = np.array(arr, dtype="datetime64[ns]")
        tm.assert_numpy_array_equal(result, expected)

        # 检查设置copy=copy_false时是否没有进行复制操作
        result = np.array(arr, dtype="M8[ns]", copy=copy_false)
        assert result.base is expected.base
        assert result.base is not None
        result = np.array(arr, dtype="datetime64[ns]", copy=copy_false)
        assert result.base is expected.base
        assert result.base is not None

    # 测试'i8'数据类型的方法
    def test_array_i8_dtype(self, arr1d):
        # 将输入的一维数组赋给arr变量
        arr = arr1d
        # 使用self.index_cls类的构造方法创建一个时间索引对象dti
        dti = self.index_cls(arr1d)
        # 根据NumPy版本选择是否复制为False
        copy_false = None if np_version_gt2 else False

        # 创建一个预期结果数组，内容为dti对象转换为'i8'数据类型的视图
        expected = dti.asi8
        # 使用'i8'数据类型创建一个新的NumPy数组
        result = np.array(arr, dtype="i8")
        tm.assert_numpy_array_equal(result, expected)

        # 使用np.int64数据类型创建一个新的NumPy数组
        result = np.array(arr, dtype=np.int64)
        tm.assert_numpy_array_equal(result, expected)

        # 检查设置copy=copy_false时是否进行了复制操作
        result = np.array(arr, dtype="i8", copy=copy_false)
        assert result.base is not expected.base
        assert result.base is None

    # 测试从数组创建保持基础数组的方法
    def test_from_array_keeps_base(self):
        # 确保DatetimeArray._ndarray.base属性不丢失
        # 创建一个包含日期字符串的数组
        arr = np.array(["2000-01-01", "2000-01-02"], dtype="M8[ns]")
        # 使用DatetimeArray._from_sequence方法从数组创建DatetimeArray对象
        dta = DatetimeArray._from_sequence(arr)

        # 断言DatetimeArray._ndarray属性为原始数组arr
        assert dta._ndarray is arr
        # 使用空切片创建另一个DatetimeArray对象，确保其_ndarray.base属性为原始数组arr
        dta = DatetimeArray._from_sequence(arr[:0])
        assert dta._ndarray.base is arr

    # 测试从时间索引创建的方法
    def test_from_dti(self, arr1d):
        # 将输入的一维数组赋给arr变量
        arr = arr1d
        # 使用self.index_cls类的构造方法创建一个时间索引对象dti
        dti = self.index_cls(arr1d)
        # 断言时间索引对象dti转换为列表与输入数组arr转换为列表相等
        assert list(dti) == list(arr)

        # 检查Index.__new__是否正确处理DatetimeArray
        # 使用pd.Index方法创建时间索引对象dti2
        dti2 = pd.Index(arr)
        # 断言dti2对象是否为DatetimeIndex的实例，并且其转换为列表与输入数组arr转换为列表相等
        assert isinstance(dti2, DatetimeIndex)
        assert list(dti2) == list(arr)

    # 测试转换为对象数据类型的方法
    def test_astype_object(self, arr1d):
        # 将输入的一维数组赋给arr变量
        arr = arr1d
        # 使用self.index_cls类的构造方法创建一个时间索引对象dti
        dti = self.index_cls(arr1d)

        # 将数组arr转换为对象数据类型的数组asobj
        asobj = arr.astype("O")
        # 断言asobj是否为np.ndarray对象，并且其数据类型为'O'，内容与时间索引对象dti转换为列表相等
        assert isinstance(asobj, np.ndarray)
        assert asobj.dtype == "O"
        assert list(asobj) == list(dti)

    # 使用pytest.mark.filterwarnings忽略"PeriodDtype\[B\] is deprecated:FutureWarning"警告
    # 测试将日期时间索引转换为周期索引的方法
    def test_to_period(self, datetime_index, freqstr):
        # 将传入的日期时间索引存储在变量dti中
        dti = datetime_index
        # 从dti中提取底层数据数组并存储在arr中
        arr = dti._data

        # 根据频率字符串创建周期数据类型，并存储在freqstr中
        freqstr = PeriodDtype(to_offset(freqstr))._freqstr
        # 使用freqstr将dti转换为周期索引，并存储在expected中
        expected = dti.to_period(freq=freqstr)
        # 使用freqstr将arr转换为周期数组，并存储在result中
        result = arr.to_period(freq=freqstr)
        # 断言result是PeriodArray类型的对象
        assert isinstance(result, PeriodArray)

        # 使用测试工具函数验证result与expected的数据是否相等
        tm.assert_equal(result, expected._data)

    # 测试将一维数组转换为二维周期数组的方法
    def test_to_period_2d(self, arr1d):
        # 将一维数组arr1d重塑为二维数组arr2d
        arr2d = arr1d.reshape(1, -1)

        # 如果arr1d的时区信息为None，则warn为None，否则为UserWarning
        warn = None if arr1d.tz is None else UserWarning
        # 使用测试工具函数检测是否会产生警告，并匹配字符串"will drop timezone information"
        with tm.assert_produces_warning(warn, match="will drop timezone information"):
            # 使用频率字符串"D"将arr2d转换为周期数组，并存储在result中
            result = arr2d.to_period("D")
            # 使用频率字符串"D"将arr1d转换为周期数组，并重塑为二维数组，存储在expected中
            expected = arr1d.to_period("D").reshape(1, -1)
        # 使用测试工具函数验证result与expected的周期数组是否相等
        tm.assert_period_array_equal(result, expected)

    # 测试日期时间数组的布尔属性方法
    @pytest.mark.parametrize("propname", DatetimeArray._bool_ops)
    def test_bool_properties(self, arr1d, propname):
        # 将arr1d创建为self.index_cls的实例，并存储在dti中
        dti = self.index_cls(arr1d)
        # 将arr1d存储在arr中
        arr = arr1d
        # 断言dti的频率与arr的频率相同
        assert dti.freq == arr.freq

        # 使用属性propname获取arr的布尔属性结果，并存储在result中
        result = getattr(arr, propname)
        # 使用属性propname获取dti的布尔属性结果，并转换为numpy数组，存储在expected中
        expected = np.array(getattr(dti, propname), dtype=result.dtype)

        # 使用测试工具函数验证result与expected的numpy数组是否相等
        tm.assert_numpy_array_equal(result, expected)

    # 测试日期时间数组的整数属性方法
    @pytest.mark.parametrize("propname", DatetimeArray._field_ops)
    def test_int_properties(self, arr1d, propname):
        # 将arr1d创建为self.index_cls的实例，并存储在dti中
        dti = self.index_cls(arr1d)
        # 将arr1d存储在arr中
        arr = arr1d

        # 使用属性propname获取arr的整数属性结果，并存储在result中
        result = getattr(arr, propname)
        # 使用属性propname获取dti的整数属性结果，并转换为numpy数组，存储在expected中
        expected = np.array(getattr(dti, propname), dtype=result.dtype)

        # 使用测试工具函数验证result与expected的numpy数组是否相等
        tm.assert_numpy_array_equal(result, expected)
    # 定义测试函数，测试数组取值和填充操作的有效性，参数包括一维数组和固定时间戳
    def test_take_fill_valid(self, arr1d, fixed_now_ts):
        # 将传入的一维数组赋值给arr
        arr = arr1d
        # 使用self.index_cls类方法创建dti对象，传入arr1d作为参数
        dti = self.index_cls(arr1d)

        # 使用fixed_now_ts设置当前时间，并设定其时区与dti对象相同
        now = fixed_now_ts.tz_localize(dti.tz)
        # 从arr数组中获取索引为-1和1的值，允许填充，填充值为now
        result = arr.take([-1, 1], allow_fill=True, fill_value=now)
        # 断言取出的第一个元素与now相等
        assert result[0] == now

        # 准备错误信息消息，说明期望的值应为arr1d._scalar_type的类型或NaT
        msg = f"value should be a '{arr1d._scalar_type.__name__}' or 'NaT'. Got"
        # 使用pytest.raises检查是否抛出TypeError异常，且异常消息匹配msg
        with pytest.raises(TypeError, match=msg):
            # 当fill_value为now - now时，抛出TypeError异常
            arr.take([-1, 1], allow_fill=True, fill_value=now - now)

        # 同上，检查当fill_value为Period("2014Q1")时是否抛出TypeError异常
        with pytest.raises(TypeError, match=msg):
            arr.take([-1, 1], allow_fill=True, fill_value=Period("2014Q1"))

        # 如果dti对象的时区为None，则将tz赋值为None，否则为"US/Eastern"
        tz = None if dti.tz is not None else "US/Eastern"
        # 将当前时间设置为fixed_now_ts，并设定其时区为tz
        now = fixed_now_ts.tz_localize(tz)
        # 准备错误信息消息，说明不能比较时区感知与非时区感知的日期时间对象
        msg = "Cannot compare tz-naive and tz-aware datetime-like objects"
        # 检查当fill_value为now时，是否抛出TypeError异常，异常消息匹配msg
        with pytest.raises(TypeError, match=msg):
            arr.take([-1, 1], allow_fill=True, fill_value=now)

        # 获取NaT的值
        value = NaT._value
        # 准备错误信息消息，说明期望的值应为arr1d._scalar_type的类型或NaT
        msg = f"value should be a '{arr1d._scalar_type.__name__}' or 'NaT'. Got"
        # 检查当fill_value为value时，是否抛出TypeError异常，异常消息匹配msg
        with pytest.raises(TypeError, match=msg):
            arr.take([-1, 1], allow_fill=True, fill_value=value)

        # 将value设置为np.timedelta64("NaT", "ns")
        value = np.timedelta64("NaT", "ns")
        # 检查当fill_value为value时，是否抛出TypeError异常，异常消息匹配msg
        with pytest.raises(TypeError, match=msg):
            arr.take([-1, 1], allow_fill=True, fill_value=value)

        # 如果arr的时区不为None
        if arr.tz is not None:
            # GH#37356
            # 假设arr1d fixture不包含Australia/Melbourne时区
            # 将value设为fixed_now_ts在"Australia/Melbourne"时区的值
            value = fixed_now_ts.tz_localize("Australia/Melbourne")
            # 获取使用value作为fill_value时的结果
            result = arr.take([-1, 1], allow_fill=True, fill_value=value)

            # 获取使用value在arr的dtype.tz转换后作为fill_value时的期望结果
            expected = arr.take(
                [-1, 1],
                allow_fill=True,
                fill_value=value.tz_convert(arr.dtype.tz),
            )
            # 使用tm.assert_equal检查result与expected是否相等
            tm.assert_equal(result, expected)

    # 定义测试函数，测试拼接不同类型时抛出错误的情况，参数包括一维数组
    def test_concat_same_type_invalid(self, arr1d):
        # 将传入的一维数组赋值给arr
        arr = arr1d

        # 如果arr的时区为None，则将other设置为UTC时区的arr，否则为无时区的arr
        if arr.tz is None:
            other = arr.tz_localize("UTC")
        else:
            other = arr.tz_localize(None)

        # 使用pytest.raises检查是否抛出ValueError异常，且异常消息包含"to_concat must have the same"
        with pytest.raises(ValueError, match="to_concat must have the same"):
            # 调用arr的_concat_same_type方法，尝试拼接[arr, other]数组
            arr._concat_same_type([arr, other])
    # 测试函数：测试在相同类型但不同频率下连接日期时间索引
    def test_concat_same_type_different_freq(self, unit):
        # 创建一个日期范围，频率为每天 ("D")，时区为 "US/Central"，单位为给定的单位
        a = pd.date_range("2000", periods=2, freq="D", tz="US/Central", unit=unit)._data
        # 创建另一个日期范围，频率为每小时 ("h")，时区为 "US/Central"，单位为给定的单位
        b = pd.date_range("2000", periods=2, freq="h", tz="US/Central", unit=unit)._data
        # 使用 DatetimeArray 类的方法将两个日期时间数组连接起来
        result = DatetimeArray._concat_same_type([a, b])
        # 创建预期的结果数组，将日期时间对象转换为本地化为 "US/Central" 的时区，然后按给定单位调整
        expected = (
            pd.to_datetime(
                [
                    "2000-01-01 00:00:00",
                    "2000-01-02 00:00:00",
                    "2000-01-01 00:00:00",
                    "2000-01-01 01:00:00",
                ]
            )
            .tz_localize("US/Central")
            .as_unit(unit)
            ._data
        )
        # 断言检查 result 是否等于 expected
        tm.assert_datetime_array_equal(result, expected)

    # 测试函数：测试日期时间数组的 strftime 方法
    def test_strftime(self, arr1d):
        # 将输入的日期时间数组赋值给 arr
        arr = arr1d

        # 使用 strftime 方法将日期时间数组转换为指定格式的字符串数组
        result = arr.strftime("%Y %b")
        # 创建预期结果数组，使用列表推导式对每个时间戳进行格式化
        expected = np.array([ts.strftime("%Y %b") for ts in arr], dtype=object)
        # 断言检查 result 是否等于 expected
        tm.assert_numpy_array_equal(result, expected)

    # 测试函数：测试包含 NaT 的日期时间数组的 strftime 方法
    def test_strftime_nat(self):
        # 创建包含 NaT 的日期时间索引的数组
        arr = DatetimeIndex(["2019-01-01", NaT])._data

        # 使用 strftime 方法将日期时间数组转换为指定格式的字符串数组
        result = arr.strftime("%Y-%m-%d")
        # 创建预期结果数组，包含与输入数组对应的格式化日期字符串
        expected = np.array(["2019-01-01", np.nan], dtype=object)
        # 断言检查 result 是否等于 expected
        tm.assert_numpy_array_equal(result, expected)
class TestTimedeltaArray(SharedTests):
    # 设置测试类的索引类为TimedeltaIndex
    index_cls = TimedeltaIndex
    # 设置测试类的数组类为TimedeltaArray
    array_cls = TimedeltaArray
    # 设置标量类型为pd.Timedelta
    scalar_type = pd.Timedelta
    # 设置示例的数据类型为 "m8[ns]"
    example_dtype = "m8[ns]"

    def test_from_tdi(self):
        # 创建一个TimedeltaIndex对象，包含两个时间差字符串
        tdi = TimedeltaIndex(["1 Day", "3 Hours"])
        # 获取TimedeltaIndex对象的底层数据
        arr = tdi._data
        # 断言底层数据列表与原TimedeltaIndex对象的列表相同
        assert list(arr) == list(tdi)

        # 检查Index.__new__如何处理TimedeltaArray对象
        tdi2 = pd.Index(arr)
        # 断言生成的对象类型为TimedeltaIndex
        assert isinstance(tdi2, TimedeltaIndex)
        # 断言生成的对象列表与底层数据列表相同
        assert list(tdi2) == list(arr)

    def test_astype_object(self):
        # 创建一个TimedeltaIndex对象，包含两个时间差字符串
        tdi = TimedeltaIndex(["1 Day", "3 Hours"])
        # 获取TimedeltaIndex对象的底层数据
        arr = tdi._data
        # 将底层数据转换为对象数组
        asobj = arr.astype("O")
        # 断言转换后的对象类型为np.ndarray
        assert isinstance(asobj, np.ndarray)
        # 断言转换后的对象的数据类型为 "O"
        assert asobj.dtype == "O"
        # 断言转换后的对象列表与原TimedeltaIndex对象的列表相同
        assert list(asobj) == list(tdi)

    def test_to_pytimedelta(self, timedelta_index):
        # 获取时间差索引对象
        tdi = timedelta_index
        # 获取时间差索引对象的底层数据
        arr = tdi._data

        # 获取预期的Python timedelta对象数组
        expected = tdi.to_pytimedelta()
        # 获取实际的Python timedelta对象数组
        result = arr.to_pytimedelta()

        # 使用测试工具函数检查两个数组是否相等
        tm.assert_numpy_array_equal(result, expected)

    def test_total_seconds(self, timedelta_index):
        # 获取时间差索引对象
        tdi = timedelta_index
        # 获取时间差索引对象的底层数据
        arr = tdi._data

        # 获取预期的总秒数值
        expected = tdi.total_seconds()
        # 获取实际的总秒数值
        result = arr.total_seconds()

        # 使用测试工具函数检查两个数组是否相等
        tm.assert_numpy_array_equal(result, expected.values)

    @pytest.mark.parametrize("propname", TimedeltaArray._field_ops)
    def test_int_properties(self, timedelta_index, propname):
        # 获取时间差索引对象
        tdi = timedelta_index
        # 获取时间差索引对象的底层数据
        arr = tdi._data

        # 获取属性名对应的属性值数组
        result = getattr(arr, propname)
        # 获取属性名对应的预期属性值数组
        expected = np.array(getattr(tdi, propname), dtype=result.dtype)

        # 使用测试工具函数检查两个数组是否相等
        tm.assert_numpy_array_equal(result, expected)
    def test_array_interface(self, timedelta_index):
        # 获取时间增量索引对象的底层数据
        arr = timedelta_index._data
        # 根据 NumPy 的版本决定是否使用复制参数
        copy_false = None if np_version_gt2 else False

        # 默认情况下，asarray 返回相同的底层数据
        result = np.asarray(arr)
        expected = arr._ndarray
        assert result is expected
        tm.assert_numpy_array_equal(result, expected)
        result = np.array(arr, copy=copy_false)
        assert result is expected
        tm.assert_numpy_array_equal(result, expected)

        # 指定 dtype="timedelta64[ns]" 与默认情况下返回相同的结果
        result = np.asarray(arr, dtype="timedelta64[ns]")
        expected = arr._ndarray
        assert result is expected
        tm.assert_numpy_array_equal(result, expected)
        result = np.array(arr, dtype="timedelta64[ns]", copy=copy_false)
        assert result is expected
        tm.assert_numpy_array_equal(result, expected)
        result = np.array(arr, dtype="timedelta64[ns]")
        if not np_version_gt2:
            # TODO: GH 57739
            assert result is not expected
        tm.assert_numpy_array_equal(result, expected)

        # 转换为 object dtype
        result = np.asarray(arr, dtype=object)
        expected = np.array(list(arr), dtype=object)
        tm.assert_numpy_array_equal(result, expected)

        # 转换为其他 dtype 总是进行复制
        result = np.asarray(arr, dtype="int64")
        assert result is not arr.asi8
        assert not np.may_share_memory(arr, result)
        expected = arr.asi8.copy()
        tm.assert_numpy_array_equal(result, expected)

        # 其他 dtype 由 NumPy 处理
        for dtype in ["float64", str]:
            result = np.asarray(arr, dtype=dtype)
            expected = np.asarray(arr).astype(dtype)
            tm.assert_numpy_array_equal(result, expected)

    def test_take_fill_valid(self, timedelta_index, fixed_now_ts):
        # 获取时间增量索引对象和固定时间戳
        tdi = timedelta_index
        arr = tdi._data

        td1 = pd.Timedelta(days=1)
        # 使用指定索引进行取值，并允许填充为指定值 td1
        result = arr.take([-1, 1], allow_fill=True, fill_value=td1)
        assert result[0] == td1

        value = fixed_now_ts
        msg = f"value should be a '{arr._scalar_type.__name__}' or 'NaT'. Got"
        with pytest.raises(TypeError, match=msg):
            # 使用 Timestamp 类型的值作为填充值，预期会引发 TypeError 异常
            arr.take([0, 1], allow_fill=True, fill_value=value)

        value = fixed_now_ts.to_period("D")
        with pytest.raises(TypeError, match=msg):
            # 使用 Period 类型的值作为填充值，预期会引发 TypeError 异常
            arr.take([0, 1], allow_fill=True, fill_value=value)

        value = np.datetime64("NaT", "ns")
        with pytest.raises(TypeError, match=msg):
            # 如果有 NA 值，则需要适当的 dtype，预期会引发 TypeError 异常
            arr.take([-1, 1], allow_fill=True, fill_value=value)
@pytest.mark.filterwarnings(r"ignore:Period with BDay freq is deprecated:FutureWarning")
# 标记：忽略将工作日频率用于 Period 的警告信息
@pytest.mark.filterwarnings(r"ignore:PeriodDtype\[B\] is deprecated:FutureWarning")
# 标记：忽略 PeriodDtype\[B\] 类型已弃用的警告信息
class TestPeriodArray(SharedTests):
    # 测试类 TestPeriodArray 继承自 SharedTests

    index_cls = PeriodIndex
    # index_cls 设置为 PeriodIndex 类
    array_cls = PeriodArray
    # array_cls 设置为 PeriodArray 类
    scalar_type = Period
    # scalar_type 设置为 Period 类型
    example_dtype = PeriodIndex([], freq="W").dtype
    # example_dtype 设置为一个空的 PeriodIndex 对象的 dtype，频率为周（"W"）

    @pytest.fixture
    def arr1d(self, period_index):
        """
        Fixture returning DatetimeArray from parametrized PeriodIndex objects
        """
        # arr1d 作为 fixture，从参数化的 PeriodIndex 对象中返回 DatetimeArray
        return period_index._data

    def test_from_pi(self, arr1d):
        # 测试方法：从 period_index 构建 PeriodIndex
        pi = self.index_cls(arr1d)
        # 使用 index_cls 构建 PeriodIndex 对象 pi
        arr = arr1d
        # 将 arr1d 赋给 arr
        assert list(arr) == list(pi)
        # 断言：arr 和 pi 的列表表示应该相等

        # Check that Index.__new__ knows what to do with PeriodArray
        # 检查 Index.__new__ 对于 PeriodArray 的处理方式
        pi2 = pd.Index(arr)
        # 使用 pd.Index 构建索引 pi2
        assert isinstance(pi2, PeriodIndex)
        # 断言：pi2 应该是 PeriodIndex 类型
        assert list(pi2) == list(arr)
        # 断言：pi2 和 arr 的列表表示应该相等

    def test_astype_object(self, arr1d):
        # 测试方法：将数组转换为 object 类型
        pi = self.index_cls(arr1d)
        # 使用 index_cls 构建 PeriodIndex 对象 pi
        arr = arr1d
        # 将 arr1d 赋给 arr
        asobj = arr.astype("O")
        # 将 arr 转换为 object 类型，赋给 asobj
        assert isinstance(asobj, np.ndarray)
        # 断言：asobj 应该是 numpy.ndarray 类型
        assert asobj.dtype == "O"
        # 断言：asobj 的 dtype 应该是 "O"
        assert list(asobj) == list(pi)
        # 断言：asobj 和 pi 的列表表示应该相等

    def test_take_fill_valid(self, arr1d):
        # 测试方法：测试 take 方法的填充有效性
        arr = arr1d
        # 将 arr1d 赋给 arr

        value = NaT._value
        # value 设置为 NaT 的值
        msg = f"value should be a '{arr1d._scalar_type.__name__}' or 'NaT'. Got"
        # msg 设置为错误消息模板，显示当前值应为 arr1d._scalar_type 或 NaT，实际值为何

        with pytest.raises(TypeError, match=msg):
            # 使用 pytest 断言应该抛出 TypeError，并匹配 msg 的消息
            # require NaT, not iNaT, as it could be confused with an integer
            # 要求 NaT，而不是 iNaT，因为后者可能与整数混淆
            arr.take([-1, 1], allow_fill=True, fill_value=value)

        value = np.timedelta64("NaT", "ns")
        # value 设置为表示 NaT 的 np.timedelta64 对象
        with pytest.raises(TypeError, match=msg):
            # 使用 pytest 断言应该抛出 TypeError，并匹配 msg 的消息
            # require appropriate-dtype if we have a NA value
            # 如果有 NA 值，则需要适当的 dtype
            arr.take([-1, 1], allow_fill=True, fill_value=value)

    @pytest.mark.parametrize("how", ["S", "E"])
    # 参数化标记：how 参数取值为 "S" 和 "E"
    def test_to_timestamp(self, how, arr1d):
        # 测试方法：测试 to_timestamp 方法
        pi = self.index_cls(arr1d)
        # 使用 index_cls 构建 PeriodIndex 对象 pi
        arr = arr1d
        # 将 arr1d 赋给 arr

        expected = DatetimeIndex(pi.to_timestamp(how=how))._data
        # expected 设置为 pi 转换为时间戳后的 DatetimeIndex 对象的数据部分
        result = arr.to_timestamp(how=how)
        # 使用 arr 的 to_timestamp 方法得到结果 result
        assert isinstance(result, DatetimeArray)
        # 断言：result 应该是 DatetimeArray 类型

        tm.assert_equal(result, expected)
        # 使用 tm.assert_equal 检查 result 和 expected 是否相等

    def test_to_timestamp_roundtrip_bday(self):
        # 测试方法：测试工作日往返转换
        # Case where infer_freq inside would choose "D" instead of "B"
        # 在此处，infer_freq 内部会选择 "D" 而不是 "B"

        dta = pd.date_range("2021-10-18", periods=3, freq="B")._data
        # dta 设置为从 2021-10-18 开始的 3 个工作日范围的日期数组的数据部分
        parr = dta.to_period()
        # 将 dta 转换为 PeriodArray 对象 parr
        result = parr.to_timestamp()
        # 将 parr 转换回时间戳，结果赋给 result
        assert result.freq == "B"
        # 断言：result 的频率应该是 "B"
        tm.assert_extension_array_equal(result, dta)
        # 使用 tm.assert_extension_array_equal 检查 result 和 dta 是否相等

        dta2 = dta[::2]
        # dta2 设置为 dta 的每隔一个元素的切片
        parr2 = dta2.to_period()
        # 将 dta2 转换为 PeriodArray 对象 parr2
        result2 = parr2.to_timestamp()
        # 将 parr2 转换回时间戳，结果赋给 result2
        assert result2.freq == "2B"
        # 断言：result2 的频率应该是 "2B"
        tm.assert_extension_array_equal(result2, dta2)
        # 使用 tm.assert_extension_array_equal 检查 result2 和 dta2 是否相等

        parr3 = dta.to_period("2B")
        # 将 dta 转换为频率为 "2B" 的 PeriodArray 对象 parr3
        result3 = parr3.to_timestamp()
        # 将 parr3 转换回时间戳，结果赋给 result3
        assert result3.freq == "B"
        # 断言：result3 的频率应该是 "B"
        tm.assert_extension_array_equal(result3, dta)
        # 使用 tm.assert_extension_array_equal 检查 result3 和 dta 是否相等
    def test_to_timestamp_out_of_bounds(self):
        # 测试超出边界情况的时间戳转换
        # 创建一个包含三个时间段的PeriodIndex，从"1500"年开始，频率为每年一次
        pi = pd.period_range("1500", freq="Y", periods=3)
        # 出现超出边界的纳秒时间戳异常时，应该抛出OutOfBoundsDatetime异常，并且匹配指定的错误消息
        msg = "Out of bounds nanosecond timestamp: 1500-01-01 00:00:00"
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            pi.to_timestamp()

        # 对PeriodIndex的数据部分进行相同的测试
        with pytest.raises(OutOfBoundsDatetime, match=msg):
            pi._data.to_timestamp()

    @pytest.mark.parametrize("propname", PeriodArray._bool_ops)
    def test_bool_properties(self, arr1d, propname):
        # 在这种情况下，_bool_ops 只是 `is_leap_year`
        # 创建一个self.index_cls类型的对象pi，使用arr1d作为参数
        pi = self.index_cls(arr1d)
        arr = arr1d

        # 获取属性propname在arr上的值
        result = getattr(arr, propname)
        # 获取属性propname在pi上的值，并转换为NumPy数组
        expected = np.array(getattr(pi, propname))

        # 断言两个NumPy数组是否相等
        tm.assert_numpy_array_equal(result, expected)

    @pytest.mark.parametrize("propname", PeriodArray._field_ops)
    def test_int_properties(self, arr1d, propname):
        # 创建一个self.index_cls类型的对象pi，使用arr1d作为参数
        pi = self.index_cls(arr1d)
        arr = arr1d

        # 获取属性propname在arr上的值
        result = getattr(arr, propname)
        # 获取属性propname在pi上的值，并转换为NumPy数组
        expected = np.array(getattr(pi, propname))

        # 断言两个NumPy数组是否相等
        tm.assert_numpy_array_equal(result, expected)

    def test_array_interface(self, arr1d):
        arr = arr1d

        # 默认的asarray返回对象数组
        result = np.asarray(arr)
        # 将arr转换为包含对象的NumPy数组
        expected = np.array(list(arr), dtype=object)
        # 断言两个NumPy数组是否相等
        tm.assert_numpy_array_equal(result, expected)

        # 转换为对象类型的NumPy数组（与默认相同）
        result = np.asarray(arr, dtype=object)
        # 断言两个NumPy数组是否相等
        tm.assert_numpy_array_equal(result, expected)

        # 转换为int64类型的NumPy数组
        result = np.asarray(arr, dtype="int64")
        # 断言两个NumPy数组是否相等
        tm.assert_numpy_array_equal(result, arr.asi8)

        # 尝试转换为其他dtype时应该抛出TypeError异常，匹配指定的错误消息
        msg = r"float\(\) argument must be a string or a( real)? number, not 'Period'"
        with pytest.raises(TypeError, match=msg):
            np.asarray(arr, dtype="float64")

        # 转换为字符串类型S20的NumPy数组
        result = np.asarray(arr, dtype="S20")
        # 将arr转换为S20字符串类型的NumPy数组，并断言它们是否相等
        expected = np.asarray(arr).astype("S20")
        tm.assert_numpy_array_equal(result, expected)

    def test_strftime(self, arr1d):
        arr = arr1d

        # 对arr中每个Period对象调用strftime方法，格式化为"%Y"形式的年份字符串
        result = arr.strftime("%Y")
        # 创建一个期望的字符串数组，包含每个Period对象的"%Y"格式化结果
        expected = np.array([per.strftime("%Y") for per in arr], dtype=object)
        # 断言两个NumPy数组是否相等
        tm.assert_numpy_array_equal(result, expected)

    def test_strftime_nat(self):
        # GH 29578
        # 创建一个包含Period对象的PeriodArray，其中包括一个日期和NaT（Not a Time）值
        arr = PeriodArray(PeriodIndex(["2019-01-01", NaT], dtype="period[D]"))

        # 对PeriodArray调用strftime方法，格式化为"%Y-%m-%d"形式的字符串
        result = arr.strftime("%Y-%m-%d")
        # 创建一个期望的字符串数组，包含每个Period对象的"%Y-%m-%d"格式化结果
        expected = np.array(["2019-01-01", np.nan], dtype=object)
        # 断言两个NumPy数组是否相等
        tm.assert_numpy_array_equal(result, expected)
# 使用 pytest 模块的 mark.parametrize 装饰器，为 test_casting_nat_setitem_array 函数设置多个参数化测试用例
@pytest.mark.parametrize(
    "arr,casting_nats",
    [
        # 设置参数化测试用例：TimedeltaIndex 对象的 _data 属性作为 arr，casting_nats 包含 NaT 和 np.timedelta64("NaT", "ns")
        (
            TimedeltaIndex(["1 Day", "3 Hours", "NaT"])._data,
            (NaT, np.timedelta64("NaT", "ns")),
        ),
        # 设置参数化测试用例：pd.date_range 的 _data 属性作为 arr，casting_nats 包含 NaT 和 np.datetime64("NaT", "ns")
        (
            pd.date_range("2000-01-01", periods=3, freq="D")._data,
            (NaT, np.datetime64("NaT", "ns")),
        ),
        # 设置参数化测试用例：pd.period_range 的 _data 属性作为 arr，casting_nats 只包含 NaT
        (pd.period_range("2000-01-01", periods=3, freq="D")._data, (NaT,)),
    ],
    # 使用 lambda 函数为每个参数化测试用例设置一个唯一的 ID
    ids=lambda x: type(x).__name__,
)
# 定义测试函数 test_casting_nat_setitem_array，接收参数 arr 和 casting_nats
def test_casting_nat_setitem_array(arr, casting_nats):
    # 期望结果是根据 arr 构建的一个新对象，其中第一个元素被替换为 NaT
    expected = type(arr)._from_sequence([NaT, arr[1], arr[2]], dtype=arr.dtype)

    # 遍历 casting_nats 中的每个元素 nat
    for nat in casting_nats:
        # 创建 arr 的副本
        arr = arr.copy()
        # 将 arr 的第一个元素替换为 nat
        arr[0] = nat
        # 使用测试框架的函数 tm.assert_equal 检查 arr 是否等于期望的结果 expected
        tm.assert_equal(arr, expected)


# 使用 pytest 模块的 mark.parametrize 装饰器，为 test_invalid_nat_setitem_array 函数设置多个参数化测试用例
@pytest.mark.parametrize(
    "arr,non_casting_nats",
    [
        # 设置参数化测试用例：TimedeltaIndex 对象的 _data 属性作为 arr，non_casting_nats 包含 np.datetime64("NaT", "ns") 和 NaT._value
        (
            TimedeltaIndex(["1 Day", "3 Hours", "NaT"])._data,
            (np.datetime64("NaT", "ns"), NaT._value),
        ),
        # 设置参数化测试用例：pd.date_range 的 _data 属性作为 arr，non_casting_nats 包含 np.timedelta64("NaT", "ns") 和 NaT._value
        (
            pd.date_range("2000-01-01", periods=3, freq="D")._data,
            (np.timedelta64("NaT", "ns"), NaT._value),
        ),
        # 设置参数化测试用例：pd.period_range 的 _data 属性作为 arr，non_casting_nats 包含 np.datetime64("NaT", "ns")、np.timedelta64("NaT", "ns") 和 NaT._value
        (
            pd.period_range("2000-01-01", periods=3, freq="D")._data,
            (np.datetime64("NaT", "ns"), np.timedelta64("NaT", "ns"), NaT._value),
        ),
    ],
    # 使用 lambda 函数为每个参数化测试用例设置一个唯一的 ID
    ids=lambda x: type(x).__name__,
)
# 定义测试函数 test_invalid_nat_setitem_array，接收参数 arr 和 non_casting_nats
def test_invalid_nat_setitem_array(arr, non_casting_nats):
    # 错误消息的内容
    msg = (
        "value should be a '(Timestamp|Timedelta|Period)', 'NaT', or array of those. "
        "Got '(timedelta64|datetime64|int)' instead."
    )

    # 遍历 non_casting_nats 中的每个元素 nat
    for nat in non_casting_nats:
        # 使用 pytest.raises 检查是否抛出 TypeError 异常，并匹配指定的错误消息
        with pytest.raises(TypeError, match=msg):
            # 尝试将 arr 的第一个元素替换为 nat
            arr[0] = nat


# 使用 pytest.mark.parametrize 装饰器，为 test_to_numpy_extra 函数设置参数化测试用例
@pytest.mark.parametrize(
    "arr",
    [
        # 设置参数化测试用例：pd.date_range("2000", periods=4) 的 array 属性
        pd.date_range("2000", periods=4).array,
        # 设置参数化测试用例：pd.timedelta_range("2000", periods=4) 的 array 属性
        pd.timedelta_range("2000", periods=4).array,
    ],
)
# 定义测试函数 test_to_numpy_extra，接收参数 arr
def test_to_numpy_extra(arr):
    # 将 arr 的第一个元素替换为 NaT
    arr[0] = NaT
    # 创建 arr 的副本 original
    original = arr.copy()

    # 调用 arr 的 to_numpy 方法，将结果存储在 result 中
    result = arr.to_numpy()
    # 使用 assert 断言检查 result 的第一个元素是否为 NaN
    assert np.isnan(result[0])

    # 调用 arr 的 to_numpy 方法，指定 dtype 为 "int64"，将结果存储在 result 中
    result = arr.to_numpy(dtype="int64")
    # 使用 assert 断言检查 result 的第一个元素是否等于 -9223372036854775808
    assert result[0] == -9223372036854775808

    # 调用 arr 的 to_numpy 方法，指定 dtype 为 "int64" 和 na_value 为 0，将结果存储在 result 中
    result = arr.to_numpy(dtype="int64", na_value=0)
    # 使用 assert 断言检查 result 的第一个元素是否等于 0
    assert result[0] == 0

    # 调用 arr 的 to_numpy 方法，指定 na_value 为 arr[1].to_numpy()，将结果存储在 result 中
    result = arr.to_numpy(na_value=arr[1].to_numpy())
    # 使用 assert 断言检查 result 的第一个元素是否等于 result 的第二个元素
    assert result[0] == result[1]

    # 调用 arr 的 to_numpy 方法，指定 na_value 为 arr[1].to_numpy(copy=False)，将结果存储在 result 中
    result = arr.to_numpy(na_value=arr[1].to_numpy(copy=False))
    # 使用 assert 断言检查 result 的第一个元素是否等于 result 的第二个元素
    assert result[0] == result[1]

    # 使用测试框架的函数 tm.assert_equal 检查 arr 是否等于副本 original
    tm.assert_equal(arr, original)


# 使用 pytest.mark.parametrize 装饰器，为 test_searchsorted_datetimelike_with_listlike 函数设置参数化测试用例
@pytest.mark.parametrize("as_index", [True, False])
@pytest.mark.parametrize(
    "values",
    [
        # 设置参数化测试用例：pd.to_datetime(["2020-01-01", "2020-02-01"])
        pd.to_datetime(["2020-01-01", "2020-02-01"]),
        # 设置参数化测试用例：pd.to_timedelta([1, 2], unit="D")
        pd.to_timedelta([1, 2], unit="D"),
        # 设置参数化测试用例：PeriodIndex(["2020-01-01", "2020-02-01"], freq="D")
        PeriodIndex(["2020-01-01", "2020-02-01"], freq="D"),
    ],
)
@pytest.mark.parametrize(
    "klass",
    [
        list,
        np.array,
        pd.array,
        pd.Series,
        pd.Index,
        pd.Categorical,
        pd.CategoricalIndex,
    ],
)
# 定义测试函数 test_searchsorted_datetimelike_with_listlike，接收参数 as_index, values 和 klass
def test_searchsorted_datetimelike_with_listlike(values, klass, as_index):
    # 如果不使用索引（as_index 为 False），将 values 转换为其 _data 属性的值
    if not as_index:
        values = values._data

    # 调用 values 的 searchsorted 方法，使用 klass(values) 作为参数，并将结果存储在 result 中
    # 创建一个 NumPy 数组，内容为 [0, 1]，数据类型与 result 数组相同
    expected = np.array([0, 1], dtype=result.dtype)
    
    # 使用测试框架中的函数检查 result 数组是否与 expected 数组相等
    tm.assert_numpy_array_equal(result, expected)
@pytest.mark.parametrize(  # 使用 pytest 的 parametrize 装饰器，用于多组参数化测试数据
    "values",  # 第一个参数 values
    [  # 参数值列表开始
        pd.to_datetime(["2020-01-01", "2020-02-01"]),  # 调用 pandas 的 to_datetime 函数，将日期字符串转换为 datetime 对象数组
        pd.to_timedelta([1, 2], unit="D"),  # 调用 pandas 的 to_timedelta 函数，创建以天为单位的时间增量数组
        PeriodIndex(["2020-01-01", "2020-02-01"], freq="D"),  # 创建 pandas 的 PeriodIndex 对象，频率为每日
    ],  # 参数值列表结束
)
@pytest.mark.parametrize(  # 使用 pytest 的 parametrize 装饰器，用于多组参数化测试数据
    "arg",  # 第二个参数 arg
    [[1, 2], ["a", "b"], [Timestamp("2020-01-01", tz="Europe/London")] * 2]  # 参数值列表开始，包含不同类型的数据作为参数
)
def test_searchsorted_datetimelike_with_listlike_invalid_dtype(values, arg):
    # https://github.com/pandas-dev/pandas/issues/32762
    msg = "[Unexpected type|Cannot compare]"  # 设置期望的错误信息模式
    with pytest.raises(TypeError, match=msg):  # 使用 pytest.raises 检测是否抛出指定类型的异常，并匹配期望的错误信息模式
        values.searchsorted(arg)  # 调用 values 的 searchsorted 方法进行测试


@pytest.mark.parametrize("klass", [list, tuple, np.array, pd.Series])
def test_period_index_construction_from_strings(klass):
    # https://github.com/pandas-dev/pandas/issues/26109
    strings = ["2020Q1", "2020Q2"] * 2  # 创建一个包含多个季度字符串的列表
    data = klass(strings)  # 使用 klass 构造函数将字符串列表转换为指定类型的数据结构
    result = PeriodIndex(data, freq="Q")  # 使用 PeriodIndex 构造函数创建季度频率的 PeriodIndex 对象
    expected = PeriodIndex([Period(s) for s in strings])  # 创建期望的 PeriodIndex 对象，每个元素是从字符串创建的 Period 对象
    tm.assert_index_equal(result, expected)  # 使用 pandas 的 assert_index_equal 函数检查 result 和 expected 是否相等


@pytest.mark.parametrize("dtype", ["M8[ns]", "m8[ns]"])
def test_from_pandas_array(dtype):
    # GH#24615
    data = np.array([1, 2, 3], dtype=dtype)  # 创建一个指定 dtype 的 numpy 数组
    arr = NumpyExtensionArray(data)  # 使用 NumpyExtensionArray 将 numpy 数组封装成扩展数组对象

    cls = {"M8[ns]": DatetimeArray, "m8[ns]": TimedeltaArray}[dtype]  # 根据 dtype 选择相应的类

    result = cls._from_sequence(arr, dtype=dtype)  # 调用相应类的 _from_sequence 方法，从扩展数组创建对象
    expected = cls._from_sequence(data, dtype=dtype)  # 从原始数据创建期望的对象
    tm.assert_extension_array_equal(result, expected)  # 使用 pandas 的 assert_extension_array_equal 函数检查扩展数组对象是否相等

    func = {"M8[ns]": pd.to_datetime, "m8[ns]": pd.to_timedelta}[dtype]  # 根据 dtype 选择相应的函数
    result = func(arr).array  # 使用选择的函数处理扩展数组对象
    expected = func(data).array  # 使用选择的函数处理原始数据
    tm.assert_equal(result, expected)  # 使用 pandas 的 assert_equal 函数检查处理结果是否相等

    # Let's check the Indexes while we're here
    idx_cls = {"M8[ns]": DatetimeIndex, "m8[ns]": TimedeltaIndex}[dtype]  # 根据 dtype 选择相应的索引类
    result = idx_cls(arr)  # 使用选择的索引类创建索引对象
    expected = idx_cls(data)  # 使用选择的索引类从原始数据创建期望的索引对象
    tm.assert_index_equal(result, expected)  # 使用 pandas 的 assert_index_equal 函数检查索引对象是否相等
```