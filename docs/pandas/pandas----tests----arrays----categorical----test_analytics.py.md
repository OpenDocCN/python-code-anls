# `D:\src\scipysrc\pandas\pandas\tests\arrays\categorical\test_analytics.py`

```
import re  # 导入正则表达式模块
import sys  # 导入系统相关的模块

import numpy as np  # 导入 NumPy 数学计算库
import pytest  # 导入 pytest 测试框架

from pandas.compat import PYPY  # 从 pandas 兼容模块中导入 PYPY 变量

from pandas import (  # 从 pandas 中导入多个类和函数
    Categorical,
    CategoricalDtype,
    DataFrame,
    Index,
    NaT,
    Series,
    date_range,
)
import pandas._testing as tm  # 导入 pandas 测试模块
from pandas.api.types import is_scalar  # 从 pandas API 类型模块导入 is_scalar 函数


class TestCategoricalAnalytics:
    @pytest.mark.parametrize("aggregation", ["min", "max"])
    def test_min_max_not_ordered_raises(self, aggregation):
        # unordered cats have no min/max
        cat = Categorical(["a", "b", "c", "d"], ordered=False)  # 创建一个无序分类对象 cat
        msg = f"Categorical is not ordered for operation {aggregation}"  # 错误信息字符串
        agg_func = getattr(cat, aggregation)  # 获取分类对象 cat 上的聚合函数

        with pytest.raises(TypeError, match=msg):  # 检查是否抛出预期的 TypeError 异常
            agg_func()

        ufunc = np.minimum if aggregation == "min" else np.maximum  # 根据聚合类型选择最小或最大函数
        with pytest.raises(TypeError, match=msg):  # 检查是否抛出预期的 TypeError 异常
            ufunc.reduce(cat)

    def test_min_max_ordered(self, index_or_series_or_array):
        cat = Categorical(["a", "b", "c", "d"], ordered=True)  # 创建一个有序分类对象 cat
        obj = index_or_series_or_array(cat)  # 使用传入的索引、Series 或数组创建对象 obj
        _min = obj.min()  # 计算 obj 中的最小值
        _max = obj.max()  # 计算 obj 中的最大值
        assert _min == "a"  # 断言最小值为 "a"
        assert _max == "d"  # 断言最大值为 "d"

        assert np.minimum.reduce(obj) == "a"  # 使用 NumPy 计算 obj 的最小值，预期为 "a"
        assert np.maximum.reduce(obj) == "d"  # 使用 NumPy 计算 obj 的最大值，预期为 "d"
        # TODO: raises if we pass axis=0  (on Index and Categorical, not Series)

        cat = Categorical(
            ["a", "b", "c", "d"], categories=["d", "c", "b", "a"], ordered=True
        )  # 创建一个有序分类对象 cat，指定自定义的顺序
        obj = index_or_series_or_array(cat)  # 使用传入的索引、Series 或数组创建对象 obj
        _min = obj.min()  # 计算 obj 中的最小值
        _max = obj.max()  # 计算 obj 中的最大值
        assert _min == "d"  # 断言最小值为 "d"
        assert _max == "a"  # 断言最大值为 "a"
        assert np.minimum.reduce(obj) == "d"  # 使用 NumPy 计算 obj 的最小值，预期为 "d"
        assert np.maximum.reduce(obj) == "a"  # 使用 NumPy 计算 obj 的最大值，预期为 "a"

    def test_min_max_reduce(self):
        # GH52788
        cat = Categorical(["a", "b", "c", "d"], ordered=True)  # 创建一个有序分类对象 cat
        df = DataFrame(cat)  # 使用分类对象创建 DataFrame 对象

        result_max = df.agg("max")  # 对 DataFrame 执行最大值聚合操作
        expected_max = Series(Categorical(["d"], dtype=cat.dtype))  # 期望的最大值 Series 对象
        tm.assert_series_equal(result_max, expected_max)  # 使用测试工具比较结果和期望的最大值 Series 对象

        result_min = df.agg("min")  # 对 DataFrame 执行最小值聚合操作
        expected_min = Series(Categorical(["a"], dtype=cat.dtype))  # 期望的最小值 Series 对象
        tm.assert_series_equal(result_min, expected_min)  # 使用测试工具比较结果和期望的最小值 Series 对象

    @pytest.mark.parametrize(
        "categories,expected",
        [
            (list("ABC"), np.nan),
            ([1, 2, 3], np.nan),
            pytest.param(
                Series(date_range("2020-01-01", periods=3), dtype="category"),
                NaT,
                marks=pytest.mark.xfail(
                    reason="https://github.com/pandas-dev/pandas/issues/29962"
                ),
            ),
        ],
    )
    @pytest.mark.parametrize("aggregation", ["min", "max"])
    def test_min_max_ordered_empty(self, categories, expected, aggregation):
        # GH 30227
        cat = Categorical([], categories=categories, ordered=True)  # 创建一个有序空分类对象 cat

        agg_func = getattr(cat, aggregation)  # 获取分类对象 cat 上的聚合函数
        result = agg_func()  # 执行聚合函数
        assert result is expected  # 断言结果与预期相同
    @pytest.mark.parametrize(
        "values, categories",
        [(["a", "b", "c", np.nan], list("cba")), ([1, 2, 3, np.nan], [3, 2, 1])],
    )
    @pytest.mark.parametrize("function", ["min", "max"])
    def test_min_max_with_nan(self, values, categories, function, skipna):
        # GH 25303
        # 创建一个分类变量对象，用于测试包含 NaN 值的最小值和最大值函数
        cat = Categorical(values, categories=categories, ordered=True)
        # 调用 getattr 函数执行指定的最小值或最大值函数，并传入 skipna 参数
        result = getattr(cat, function)(skipna=skipna)

        # 如果 skipna 参数为 False，则断言结果应为 np.nan
        if skipna is False:
            assert result is np.nan
        else:
            # 否则根据 function 参数确定期望的结果值
            expected = categories[0] if function == "min" else categories[2]
            assert result == expected

    @pytest.mark.parametrize("function", ["min", "max"])
    def test_min_max_only_nan(self, function, skipna):
        # https://github.com/pandas-dev/pandas/issues/33450
        # 创建一个只包含 NaN 值的分类变量对象，用于测试最小值和最大值函数的处理
        cat = Categorical([np.nan], categories=[1, 2], ordered=True)
        # 调用 getattr 函数执行指定的最小值或最大值函数，并传入 skipna 参数
        result = getattr(cat, function)(skipna=skipna)
        # 断言结果应为 np.nan
        assert result is np.nan

    @pytest.mark.parametrize("method", ["min", "max"])
    def test_numeric_only_min_max_raises(self, method):
        # GH 25303
        # 创建一个包含 NaN 和数值的分类变量对象，用于测试仅支持数值的最小值和最大值函数
        cat = Categorical(
            [np.nan, 1, 2, np.nan], categories=[5, 4, 3, 2, 1], ordered=True
        )
        # 使用 pytest.raises 检查调用最小值或最大值函数时，是否会引发 TypeError 异常
        with pytest.raises(TypeError, match=".* got an unexpected keyword"):
            getattr(cat, method)(numeric_only=True)

    @pytest.mark.parametrize("method", ["min", "max"])
    def test_numpy_min_max_raises(self, method):
        # 创建一个非有序的分类变量对象，用于测试使用 NumPy 函数执行最小值和最大值操作时的异常情况
        cat = Categorical(["a", "b", "c", "b"], ordered=False)
        # 准备一个异常消息，说明分类变量不是有序的
        msg = (
            f"Categorical is not ordered for operation {method}\n"
            "you can use .as_ordered() to change the Categorical to an ordered one"
        )
        # 使用 pytest.raises 检查调用 NumPy 函数时是否会引发 TypeError 异常，并匹配预期的消息
        method = getattr(np, method)
        with pytest.raises(TypeError, match=re.escape(msg)):
            method(cat)

    @pytest.mark.parametrize("kwarg", ["axis", "out", "keepdims"])
    @pytest.mark.parametrize("method", ["min", "max"])
    def test_numpy_min_max_unsupported_kwargs_raises(self, method, kwarg):
        # 创建一个有序的分类变量对象，用于测试在 NumPy 函数中使用不支持的关键字参数时的异常情况
        cat = Categorical(["a", "b", "c", "b"], ordered=True)
        # 准备一个异常消息，说明不支持该参数
        msg = (
            f"the '{kwarg}' parameter is not supported in the pandas implementation "
            f"of {method}"
        )
        # 对于 axis 参数，使用特定的消息格式进行匹配
        if kwarg == "axis":
            msg = r"`axis` must be fewer than the number of dimensions \(1\)"
        # 准备 kwargs 字典，包含要测试的关键字参数
        kwargs = {kwarg: 42}
        # 使用 pytest.raises 检查调用 NumPy 函数时是否会引发 ValueError 异常，并匹配预期的消息
        method = getattr(np, method)
        with pytest.raises(ValueError, match=msg):
            method(cat, **kwargs)

    @pytest.mark.parametrize("method, expected", [("min", "a"), ("max", "c")])
    def test_numpy_min_max_axis_equals_none(self, method, expected):
        # 创建一个有序的分类变量对象，用于测试在 NumPy 函数中 axis 参数为 None 时的最小值和最大值函数
        cat = Categorical(["a", "b", "c", "b"], ordered=True)
        # 调用 getattr 函数执行指定的最小值或最大值函数，并传入 axis 参数为 None
        method = getattr(np, method)
        result = method(cat, axis=None)
        # 断言结果与期望的值相等
        assert result == expected
    @pytest.mark.parametrize(
        "values,categories,exp_mode",
        [  # 使用 pytest 的 parametrize 装饰器，定义多组参数化测试数据
            ([1, 1, 2, 4, 5, 5, 5], [5, 4, 3, 2, 1], [5]),  # 第一组测试数据
            ([1, 1, 1, 4, 5, 5, 5], [5, 4, 3, 2, 1], [5, 1]),  # 第二组测试数据
            ([1, 2, 3, 4, 5], [5, 4, 3, 2, 1], [5, 4, 3, 2, 1]),  # 第三组测试数据
            ([np.nan, np.nan, np.nan, 4, 5], [5, 4, 3, 2, 1], [5, 4]),  # 第四组测试数据
            ([np.nan, np.nan, np.nan, 4, 5, 4], [5, 4, 3, 2, 1], [4]),  # 第五组测试数据
            ([np.nan, np.nan, 4, 5, 4], [5, 4, 3, 2, 1], [4]),  # 第六组测试数据
        ],
    )
    def test_mode(self, values, categories, exp_mode):
        # 测试函数：验证 Categorical 对象的 mode 方法返回的结果是否符合预期
        cat = Categorical(values, categories=categories, ordered=True)
        res = Series(cat).mode()._values
        exp = Categorical(exp_mode, categories=categories, ordered=True)
        tm.assert_categorical_equal(res, exp)

    def test_searchsorted(self, ordered):
        # 测试函数：验证 Categorical 对象的 searchsorted 方法返回的搜索结果是否正确

        # 创建一个有序的 Categorical 对象
        cat = Categorical(
            ["cheese", "milk", "apple", "bread", "bread"],
            categories=["cheese", "milk", "apple", "bread"],
            ordered=ordered,
        )
        ser = Series(cat)

        # 测试搜索单个元素的情况，默认 side='left'
        res_cat = cat.searchsorted("apple")
        assert res_cat == 2
        assert is_scalar(res_cat)

        res_ser = ser.searchsorted("apple")
        assert res_ser == 2
        assert is_scalar(res_ser)

        # 测试搜索单个元素数组的情况，默认 side='left'
        res_cat = cat.searchsorted(["bread"])
        res_ser = ser.searchsorted(["bread"])
        exp = np.array([3], dtype=np.intp)
        tm.assert_numpy_array_equal(res_cat, exp)
        tm.assert_numpy_array_equal(res_ser, exp)

        # 测试搜索多个元素数组的情况，指定 side='right'
        res_cat = cat.searchsorted(["apple", "bread"], side="right")
        res_ser = ser.searchsorted(["apple", "bread"], side="right")
        exp = np.array([3, 5], dtype=np.intp)
        tm.assert_numpy_array_equal(res_cat, exp)
        tm.assert_numpy_array_equal(res_ser, exp)

        # 测试搜索一个不在 Categorical 中的值，应抛出 TypeError 异常
        with pytest.raises(TypeError, match="cucumber"):
            cat.searchsorted("cucumber")
        with pytest.raises(TypeError, match="cucumber"):
            ser.searchsorted("cucumber")

        # 测试搜索多个值中有一个不在 Categorical 中的值，应抛出 TypeError 异常
        msg = (
            "Cannot setitem on a Categorical with a new category, "
            "set the categories first"
        )
        with pytest.raises(TypeError, match=msg):
            cat.searchsorted(["bread", "cucumber"])
        with pytest.raises(TypeError, match=msg):
            ser.searchsorted(["bread", "cucumber"])
    def test_unique(self, ordered):
        # GH38140
        # 创建指定顺序的分类数据类型
        dtype = CategoricalDtype(["a", "b", "c"], ordered=ordered)

        # 创建具有指定数据类型的分类数据对象
        cat = Categorical(["a", "b", "c"], dtype=dtype)
        # 获取分类对象的唯一值
        res = cat.unique()
        # 断言唯一值与原分类对象相同
        tm.assert_categorical_equal(res, cat)

        # 创建具有重复值的分类数据对象
        cat = Categorical(["a", "b", "a", "a"], dtype=dtype)
        # 获取分类对象的唯一值
        res = cat.unique()
        # 断言唯一值与预期分类对象相同
        tm.assert_categorical_equal(res, Categorical(["a", "b"], dtype=dtype))

        # 创建具有多个重复值的分类数据对象
        cat = Categorical(["c", "a", "b", "a", "a"], dtype=dtype)
        # 获取分类对象的唯一值
        res = cat.unique()
        # 创建预期的分类对象
        exp_cat = Categorical(["c", "a", "b"], dtype=dtype)
        # 断言唯一值与预期分类对象相同
        tm.assert_categorical_equal(res, exp_cat)

        # 创建包含 NaN 值的分类数据对象，确保 NaN 被移除
        cat = Categorical(["b", np.nan, "b", np.nan, "a"], dtype=dtype)
        # 获取分类对象的唯一值
        res = cat.unique()
        # 创建预期的分类对象，确保 NaN 被保留
        exp_cat = Categorical(["b", np.nan, "a"], dtype=dtype)
        # 断言唯一值与预期分类对象相同
        tm.assert_categorical_equal(res, exp_cat)

    def test_unique_index_series(self, ordered):
        # GH38140
        # 创建指定顺序的分类数据类型
        dtype = CategoricalDtype([3, 2, 1], ordered=ordered)

        # 创建具有整数值的分类数据对象
        c = Categorical([3, 1, 2, 2, 1], dtype=dtype)
        # 获取分类对象的唯一值，按出现顺序排序
        exp = Categorical([3, 1, 2], dtype=dtype)
        tm.assert_categorical_equal(c.unique(), exp)

        # 断言索引对象的唯一值与预期索引对象相同
        tm.assert_index_equal(Index(c).unique(), Index(exp))
        # 断言序列对象的唯一值与预期分类对象相同
        tm.assert_categorical_equal(Series(c).unique(), exp)

        # 创建具有重复整数值的分类数据对象
        c = Categorical([1, 1, 2, 2], dtype=dtype)
        exp = Categorical([1, 2], dtype=dtype)
        tm.assert_categorical_equal(c.unique(), exp)
        tm.assert_index_equal(Index(c).unique(), Index(exp))
        tm.assert_categorical_equal(Series(c).unique(), exp)

    def test_shift(self):
        # GH 9416
        # 创建具有字符串值的分类数据对象
        cat = Categorical(["a", "b", "c", "d", "a"])

        # 向前位移分类数据对象
        sp1 = cat.shift(1)
        xp1 = Categorical([np.nan, "a", "b", "c", "d"])
        tm.assert_categorical_equal(sp1, xp1)
        # 断言前向位移后的分类数据与原数据相同
        tm.assert_categorical_equal(cat[:-1], sp1[1:])

        # 向后位移分类数据对象
        sn2 = cat.shift(-2)
        xp2 = Categorical(
            ["c", "d", "a", np.nan, np.nan], categories=["a", "b", "c", "d"]
        )
        tm.assert_categorical_equal(sn2, xp2)
        # 断言后向位移后的分类数据与原数据相同
        tm.assert_categorical_equal(cat[2:], sn2[:-2])

        # 零位移分类数据对象
        tm.assert_categorical_equal(cat, cat.shift(0))

    def test_nbytes(self):
        # 创建具有整数值的分类数据对象
        cat = Categorical([1, 2, 3])
        # 预期的字节大小计算结果
        exp = 3 + 3 * 8  # 3 int8s for values + 3 int64s for categories
        # 断言分类对象的字节大小与预期结果相同
        assert cat.nbytes == exp
    def test_memory_usage(self):
        # 创建一个包含整数 1, 2, 3 的分类变量
        cat = Categorical([1, 2, 3])

        # .categories 是一个索引，因此内存使用量包括哈希表
        assert 0 < cat.nbytes <= cat.memory_usage()
        assert 0 < cat.nbytes <= cat.memory_usage(deep=True)

        # 创建一个包含字符串 "foo", "foo", "bar" 的分类变量
        cat = Categorical(["foo", "foo", "bar"])
        # 深层内存使用量应大于 nbytes，因为字符串会占用更多内存
        assert cat.memory_usage(deep=True) > cat.nbytes

        if not PYPY:
            # PYPY 平台下不执行以下代码块
            # sys.getsizeof 会调用 memory_usage(deep=True)，并增加一些 GC 开销
            diff = cat.memory_usage(deep=True) - sys.getsizeof(cat)
            assert abs(diff) < 100

    def test_map(self):
        # 创建一个包含字符 "ABABC" 的分类变量，指定类别和排序方式
        c = Categorical(list("ABABC"), categories=list("CBA"), ordered=True)
        # 对分类变量进行 map 操作，将每个元素转换为小写字母
        result = c.map(lambda x: x.lower(), na_action=None)
        # 创建期望的分类变量，将字符 "ABABC" 转换为小写并保持类别和排序方式
        exp = Categorical(list("ababc"), categories=list("cba"), ordered=True)
        # 断言两个分类变量相等
        tm.assert_categorical_equal(result, exp)

        # 创建一个包含字符 "ABABC" 的分类变量，指定类别但无序
        c = Categorical(list("ABABC"), categories=list("ABC"), ordered=False)
        # 对分类变量进行 map 操作，将每个元素转换为小写字母
        result = c.map(lambda x: x.lower(), na_action=None)
        # 创建期望的分类变量，将字符 "ABABC" 转换为小写并保持类别但无序
        exp = Categorical(list("ababc"), categories=list("abc"), ordered=False)
        # 断言两个分类变量相等
        tm.assert_categorical_equal(result, exp)

        # 对分类变量进行 map 操作，将每个元素映射为整数 1
        result = c.map(lambda x: 1, na_action=None)
        # 断言结果是一个索引而不是数组
        # GH 12766: 返回一个索引而不是数组
        tm.assert_index_equal(result, Index(np.array([1] * 5, dtype=np.int64)))

    @pytest.mark.parametrize("value", [1, "True", [1, 2, 3], 5.0])
    def test_validate_inplace_raises(self, value):
        # 创建一个包含字符 "A", "B", "B", "C", "A" 的分类变量
        cat = Categorical(["A", "B", "B", "C", "A"])
        # 准备错误消息，指示 inplace 参数应该是布尔类型，而不是给定类型的值
        msg = (
            'For argument "inplace" expected type bool, '
            f"received type {type(value).__name__}"
        )

        # 使用 pytest 断言捕获 ValueError 异常，并匹配预期的错误消息
        with pytest.raises(ValueError, match=msg):
            cat.sort_values(inplace=value)

    def test_quantile_empty(self):
        # 确保在结果代码中有正确的 itemsize
        # 创建一个包含字符 "A", "B" 的分类变量
        cat = Categorical(["A", "B"])
        # 创建一个浮点数索引
        idx = Index([0.0, 0.5])
        # 对空分类变量进行 _quantile 操作，使用线性插值
        result = cat[:0]._quantile(idx, interpolation="linear")
        # 断言结果的 _codes 属性的数据类型为 np.int8
        assert result._codes.dtype == np.int8

        # 创建期望的分类变量，使用 allow_fill=True
        expected = cat.take([-1, -1], allow_fill=True)
        # 断言扩展数组的相等性
        tm.assert_extension_array_equal(result, expected)
```