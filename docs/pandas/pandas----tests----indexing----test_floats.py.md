# `D:\src\scipysrc\pandas\pandas\tests\indexing\test_floats.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 Pytest 库，用于单元测试

from pandas import (  # 从 Pandas 库中导入以下对象和函数
    DataFrame,  # 数据框架对象
    Index,  # 索引对象
    RangeIndex,  # 范围索引对象
    Series,  # 系列对象
    date_range,  # 日期范围生成函数
    period_range,  # 时期范围生成函数
    timedelta_range,  # 时间差范围生成函数
)
import pandas._testing as tm  # 导入 Pandas 内部测试工具模块


def gen_obj(klass, index):
    """
    根据类别和索引生成对象

    Parameters:
    klass : class
        类别，可以是 Series 或 DataFrame
    index : Index
        索引对象

    Returns:
    obj : object
        根据类别和索引生成的对象
    """
    if klass is Series:
        obj = Series(np.arange(len(index)), index=index)
    else:
        obj = DataFrame(
            np.random.default_rng(2).standard_normal((len(index), len(index))),
            index=index,
            columns=index,
        )
    return obj


class TestFloatIndexers:
    def check(self, result, original, indexer, getitem):
        """
        比较器，用于比较结果

        Parameters:
        result : object
            测试结果
        original : object
            原始对象
        indexer : object
            索引器
        getitem : bool
            是否使用 getitem 方法
        """
        if isinstance(original, Series):
            expected = original.iloc[indexer]
        elif getitem:
            expected = original.iloc[:, indexer]
        else:
            expected = original.iloc[indexer]

        tm.assert_almost_equal(result, expected)

    @pytest.mark.parametrize(
        "index",
        [
            Index(list("abcde")),
            Index(list("abcde"), dtype="category"),
            date_range("2020-01-01", periods=5),
            timedelta_range("1 day", periods=5),
            period_range("2020-01-01", periods=5),
        ],
    )
    def test_scalar_non_numeric(self, index, frame_or_series, indexer_sl):
        """
        测试非数值标量的索引行为

        Parameters:
        index : Index
            索引对象
        frame_or_series : class
            类别，可以是 Series 或 DataFrame
        indexer_sl : function
            索引器函数
        """
        # GH 4892
        # float_indexers should raise exceptions
        # on appropriate Index types & accessors
        # 测试 float_indexers 在适当的索引类型和访问器上是否会引发异常

        s = gen_obj(frame_or_series, index)

        # getting
        with pytest.raises(KeyError, match="^3.0$"):
            indexer_sl(s)[3.0]

        # contains
        assert 3.0 not in s

        s2 = s.copy()
        indexer_sl(s2)[3.0] = 10

        if indexer_sl is tm.setitem:
            assert 3.0 in s2.axes[-1]
        elif indexer_sl is tm.loc:
            assert 3.0 in s2.axes[0]
        else:
            assert 3.0 not in s2.axes[0]
            assert 3.0 not in s2.axes[-1]

    @pytest.mark.parametrize(
        "index",
        [
            Index(list("abcde")),
            Index(list("abcde"), dtype="category"),
            date_range("2020-01-01", periods=5),
            timedelta_range("1 day", periods=5),
            period_range("2020-01-01", periods=5),
        ],
    )
    def test_scalar_non_numeric_series_fallback(self, index):
        """
        测试非数值标量索引在 Series 中的回退行为

        Parameters:
        index : Index
            索引对象
        """
        # starting in 3.0, integer keys are always treated as labels, no longer
        # fall back to positional.
        # 从 3.0 开始，整数键始终被视为标签，不再回退到位置索引。

        s = Series(np.arange(len(index)), index=index)

        with pytest.raises(KeyError, match="3"):
            s[3]
        with pytest.raises(KeyError, match="^3.0$"):
            s[3.0]
    def test_scalar_with_mixed(self, indexer_sl):
        # 创建一个包含整数的 Series 对象，指定索引为字符串
        s2 = Series([1, 2, 3], index=["a", "b", "c"])
        # 创建一个包含整数的 Series 对象，其中索引包括字符串和浮点数
        s3 = Series([1, 2, 3], index=["a", "b", 1.5])

        # 在纯字符串索引中使用无效的索引器进行查找，预期引发 KeyError 异常
        with pytest.raises(KeyError, match="^1.0$"):
            indexer_sl(s2)[1.0]

        # 在纯字符串索引中使用无效的索引器进行查找，预期引发 KeyError 异常，使用正则表达式匹配
        with pytest.raises(KeyError, match=r"^1\.0$"):
            indexer_sl(s2)[1.0]

        # 使用有效的索引器从 s2 中获取索引为 'b' 的结果
        result = indexer_sl(s2)["b"]
        expected = 2
        assert result == expected

        # 混合索引，因此需要使用标签索引
        # 进行索引操作
        with pytest.raises(KeyError, match="^1.0$"):
            indexer_sl(s3)[1.0]

        # 如果索引器不是 tm.loc，则表明在版本 3.0 中，__getitem__ 不再回退到位置索引
        if indexer_sl is not tm.loc:
            with pytest.raises(KeyError, match="^1$"):
                s3[1]

        # 在混合索引中使用无效的索引器进行查找，预期引发 KeyError 异常，使用正则表达式匹配
        with pytest.raises(KeyError, match=r"^1\.0$"):
            indexer_sl(s3)[1.0]

        # 使用有效的索引器从 s3 中获取索引为 1.5 的结果
        result = indexer_sl(s3)[1.5]
        expected = 3
        assert result == expected

    @pytest.mark.parametrize(
        "index", [Index(np.arange(5), dtype=np.int64), RangeIndex(5)]
    )
    def test_scalar_integer(self, index, frame_or_series, indexer_sl):
        getitem = indexer_sl is not tm.loc

        # 测试在整数索引上如何使用标量浮点数索引器进行工作

        # 使用整数索引 i 初始化 obj
        i = index
        obj = gen_obj(frame_or_series, i)

        # 将浮点数索引强制转换为整数索引，获取索引为 3.0 的结果
        result = indexer_sl(obj)[3.0]
        self.check(result, obj, 3, getitem)

        if isinstance(obj, Series):
            # 如果 obj 是 Series 类型，则定义比较函数为 compare，断言相等性
            def compare(x, y):
                assert x == y

            expected = 100
        else:
            # 否则比较函数为 tm.assert_series_equal，如果使用 getitem，则期望的结果为 Series(100, index=range(len(obj)), name=3)
            compare = tm.assert_series_equal
            if getitem:
                expected = Series(100, index=range(len(obj)), name=3)
            else:
                expected = Series(100.0, index=range(len(obj)), name=3)

        # 复制 obj 到 s2
        s2 = obj.copy()
        # 使用索引器在 s2 中的索引 3.0 处赋值为 100
        indexer_sl(s2)[3.0] = 100

        # 使用索引器从 s2 中获取索引为 3.0 的结果
        result = indexer_sl(s2)[3.0]
        compare(result, expected)

        # 使用索引器从 s2 中获取索引为 3 的结果
        result = indexer_sl(s2)[3]
        compare(result, expected)

    @pytest.mark.parametrize(
        "index", [Index(np.arange(5), dtype=np.int64), RangeIndex(5)]
    )
    def test_scalar_integer_contains_float(self, index, frame_or_series):
        # contains
        # 整数索引
        obj = gen_obj(frame_or_series, index)

        # 断言 3.0 是否在 obj 中
        assert 3.0 in obj
    # 定义一个测试方法，用于测试标量浮点数索引器在浮点数索引上的工作
    def test_scalar_float(self, frame_or_series):
        # 创建一个浮点数索引对象
        index = Index(np.arange(5.0))
        # 生成一个对象，使用给定的 frame_or_series 和浮点数索引对象 index
        s = gen_obj(frame_or_series, index)

        # 断言除了 iloc 外的所有操作都正常
        # 获取当前索引的值作为索引器
        indexer = index[3]
        for idxr in [tm.loc, tm.setitem]:
            # 检查是否为获取操作
            getitem = idxr is not tm.loc

            # 获取操作
            result = idxr(s)[indexer]
            # 使用自定义方法检查结果
            self.check(result, s, 3, getitem)

            # 设置操作
            s2 = s.copy()
            result = idxr(s2)[indexer]
            # 使用自定义方法检查结果
            self.check(result, s, 3, getitem)

            # 试图使用随机浮点数进行访问会引发 KeyError 异常
            with pytest.raises(KeyError, match=r"^3\.5$"):
                idxr(s)[3.5]

        # 断言浮点数 3.0 在 s 中
        assert 3.0 in s

        # iloc 操作成功，使用整数索引
        expected = s.iloc[3]
        s2 = s.copy()

        # 设置 iloc 操作
        s2.iloc[3] = expected
        # 获取设置后的结果
        result = s2.iloc[3]
        # 使用自定义方法检查结果
        self.check(result, s, 3, False)

    @pytest.mark.parametrize(
        "index",
        [
            Index(list("abcde"), dtype=object),
            date_range("2020-01-01", periods=5),
            timedelta_range("1 day", periods=5),
            period_range("2020-01-01", periods=5),
        ],
    )
    @pytest.mark.parametrize("idx", [slice(3.0, 4), slice(3, 4.0), slice(3.0, 4.0)])
    def test_slice_non_numeric(self, index, idx, frame_or_series, indexer_sli):
        # GH 4892
        # float_indexers should raise exceptions
        # on appropriate Index types & accessors

        # 使用给定的 frame_or_series 和索引对象 index 生成一个对象 s
        s = gen_obj(frame_or_series, index)

        # 获取或设置项
        if indexer_sli is tm.iloc:
            msg = (
                # 如果是 iloc 索引器，指示在具有这些浮点数索引器的特定索引类型和访问器上无法进行位置索引
                "cannot do positional indexing "
                rf"on {type(index).__name__} with these indexers \[(3|4)\.0\] of "
                "type float"
            )
        else:
            msg = (
                # 否则指示在具有这些浮点或整数索引器的特定索引类型上无法进行切片索引
                "cannot do slice indexing "
                rf"on {type(index).__name__} with these indexers "
                r"\[(3|4)(\.0)?\] "
                r"of type (float|int)"
            )
        # 断言在特定的 TypeError 异常情况下，匹配指定的错误消息
        with pytest.raises(TypeError, match=msg):
            indexer_sli(s)[idx]

        # 设置项
        if indexer_sli is tm.iloc:
            # 否则保持与上述相同的消息
            msg = "slice indices must be integers or None or have an __index__ method"
        # 断言在特定的 TypeError 异常情况下，匹配指定的错误消息
        with pytest.raises(TypeError, match=msg):
            indexer_sli(s)[idx] = 0
    def test_slice_integer(self):
        # same as above, but for Integer based indexes
        # these coerce to a like integer
        # oob indicates if we are out of bounds
        # of positional indexing
        for index, oob in [
            (Index(np.arange(5, dtype=np.int64)), False),  # Initialize index as Integer-based with no out-of-bounds
            (RangeIndex(5), False),  # Initialize index as RangeIndex with no out-of-bounds
            (Index(np.arange(5, dtype=np.int64) + 10), True),  # Initialize index as Integer-based with out-of-bounds
        ]:
            # s is an in-range index
            s = Series(range(5), index=index)  # Create a Pandas Series with specified index

            # getitem
            for idx in [slice(3.0, 4), slice(3, 4.0), slice(3.0, 4.0)]:
                result = s.loc[idx]  # Retrieve elements using label-based indexing

                # these are all label indexing
                # except getitem which is positional
                # empty
                if oob:
                    indexer = slice(0, 0)  # If out-of-bounds, set empty slice
                else:
                    indexer = slice(3, 5)  # Otherwise, set a valid slice

                self.check(result, s, indexer, False)  # Check against expected result

            # getitem out-of-bounds
            for idx in [slice(-6, 6), slice(-6.0, 6.0)]:
                result = s.loc[idx]  # Retrieve elements using label-based indexing

                # these are all label indexing
                # except getitem which is positional
                # empty
                if oob:
                    indexer = slice(0, 0)  # If out-of-bounds, set empty slice
                else:
                    indexer = slice(-6, 6)  # Otherwise, set a valid slice

                self.check(result, s, indexer, False)  # Check against expected result

            # positional indexing
            msg = (
                "cannot do slice indexing "
                rf"on {type(index).__name__} with these indexers \[-6\.0\] of "
                "type float"
            )
            with pytest.raises(TypeError, match=msg):
                s[slice(-6.0, 6.0)]  # Attempt positional indexing with float type indexer

            # getitem odd floats
            for idx, res1 in [
                (slice(2.5, 4), slice(3, 5)),
                (slice(2, 3.5), slice(2, 4)),
                (slice(2.5, 3.5), slice(3, 4)),
            ]:
                result = s.loc[idx]  # Retrieve elements using label-based indexing
                if oob:
                    res = slice(0, 0)  # If out-of-bounds, set empty slice
                else:
                    res = res1  # Otherwise, set a valid slice

                self.check(result, s, res, False)  # Check against expected result

                # positional indexing
                msg = (
                    "cannot do slice indexing "
                    rf"on {type(index).__name__} with these indexers \[(2|3)\.5\] of "
                    "type float"
                )
                with pytest.raises(TypeError, match=msg):
                    s[idx]  # Attempt positional indexing with float type indexer

    @pytest.mark.parametrize("idx", [slice(2, 4.0), slice(2.0, 4), slice(2.0, 4.0)])
    # 定义一个测试方法，用于测试整数位置索引
    def test_integer_positional_indexing(self, idx):
        """确保对整数索引进行位置索引时会引发错误"""
        # 创建一个 Series 对象，索引为整数范围
        s = Series(range(2, 6), index=range(2, 6))

        # 使用普通的位置索引获取数据
        result = s[2:4]
        # 使用 iloc 进行相同的位置索引获取数据
        expected = s.iloc[2:4]
        tm.assert_series_equal(result, expected)

        # 设置一个类变量 klass 为 RangeIndex
        klass = RangeIndex
        # 构造一个错误消息，指示不能在 RangeIndex 上使用浮点类型的索引器
        msg = (
            "cannot do (slice|positional) indexing "
            rf"on {klass.__name__} with these indexers \[(2|4)\.0\] of "
            "type float"
        )
        # 使用 pytest 检查是否引发了 TypeError，并匹配预期的错误消息
        with pytest.raises(TypeError, match=msg):
            s[idx]
        with pytest.raises(TypeError, match=msg):
            s.iloc[idx]

    @pytest.mark.parametrize(
        "index", [Index(np.arange(5), dtype=np.int64), RangeIndex(5)]
    )
    # 定义一个测试方法，用于测试 DataFrame 的整数切片索引
    def test_slice_integer_frame_getitem(self, index):
        # 类似于上面的测试，但是针对 DataFrame 的获取项目维度
        # 创建一个 DataFrame，其中包含随机正态分布的数据，使用不同的索引类型
        s = DataFrame(np.random.default_rng(2).standard_normal((5, 2)), index=index)

        # 使用切片索引
        for idx in [slice(0.0, 1), slice(0, 1.0), slice(0.0, 1.0)]:
            result = s.loc[idx]
            indexer = slice(0, 2)
            self.check(result, s, indexer, False)

            # 对于位置索引
            msg = (
                "cannot do slice indexing "
                rf"on {type(index).__name__} with these indexers \[(0|1)\.0\] of "
                "type float"
            )
            with pytest.raises(TypeError, match=msg):
                s[idx]

        # 处理超出边界的切片索引
        for idx in [slice(-10, 10), slice(-10.0, 10.0)]:
            result = s.loc[idx]
            self.check(result, s, slice(-10, 10), True)

        # 对于位置索引
        msg = (
            "cannot do slice indexing "
            rf"on {type(index).__name__} with these indexers \[-10\.0\] of "
            "type float"
        )
        with pytest.raises(TypeError, match=msg):
            s[slice(-10.0, 10.0)]

        # 处理奇怪的浮点数切片索引
        for idx, res in [
            (slice(0.5, 1), slice(1, 2)),
            (slice(0, 0.5), slice(0, 1)),
            (slice(0.5, 1.5), slice(1, 2)),
        ]:
            result = s.loc[idx]
            self.check(result, s, res, False)

            # 对于位置索引
            msg = (
                "cannot do slice indexing "
                rf"on {type(index).__name__} with these indexers \[0\.5\] of "
                "type float"
            )
            with pytest.raises(TypeError, match=msg):
                s[idx]

    @pytest.mark.parametrize("idx", [slice(3.0, 4), slice(3, 4.0), slice(3.0, 4.0)])
    @pytest.mark.parametrize(
        "index", [Index(np.arange(5), dtype=np.int64), RangeIndex(5)]
    )
    def test_float_slice_getitem_with_integer_index_raises(self, idx, index):
        # 定义一个测试函数，测试当使用整数索引时，DataFrame 的切片取值是否会引发异常

        # 创建一个 DataFrame 对象，包含随机生成的标准正态分布数据，使用给定的索引
        s = DataFrame(np.random.default_rng(2).standard_normal((5, 2)), index=index)

        # 复制 DataFrame 对象
        sc = s.copy()
        # 使用 loc 方法设置指定索引处的值为 0
        sc.loc[idx] = 0
        # 获取设置后的值，将其展平为一维数组
        result = sc.loc[idx].values.ravel()
        # 断言结果中的所有元素是否都为 0
        assert (result == 0).all()

        # 使用位置索引方式进行测试
        # 准备异常消息，用于说明无法在特定类型的索引器上进行切片索引
        msg = (
            "cannot do slice indexing "
            rf"on {type(index).__name__} with these indexers \[(3|4)\.0\] of "
            "type float"
        )
        # 使用 pytest 检查是否会引发 TypeError 异常，并匹配预期的异常消息
        with pytest.raises(TypeError, match=msg):
            s[idx] = 0

        with pytest.raises(TypeError, match=msg):
            s[idx]

    @pytest.mark.parametrize("idx", [slice(3.0, 4), slice(3, 4.0), slice(3.0, 4.0)])
    def test_slice_float(self, idx, frame_or_series, indexer_sl):
        # 定义一个测试函数，测试当使用浮点数切片索引时的情况

        # 创建一个索引对象，包含从 0.1 开始的五个浮点数
        index = Index(np.arange(5.0)) + 0.1
        # 使用给定的数据类型生成一个对象（DataFrame 或 Series），并使用上述索引
        s = gen_obj(frame_or_series, index)

        # 期望的结果是从 s 中提取的切片
        expected = s.iloc[3:4]

        # 获取使用给定索引器处理后的切片结果
        result = indexer_sl(s)[idx]
        # 断言结果的类型与原始对象的类型相同
        assert isinstance(result, type(s))
        # 使用测试框架中的工具方法检查结果是否与期望值相等
        tm.assert_equal(result, expected)

        # 设置值操作
        s2 = s.copy()
        # 使用给定的索引器设置指定索引处的值为 0
        indexer_sl(s2)[idx] = 0
        # 获取设置后的值，将其展平为一维数组
        result = indexer_sl(s2)[idx].values.ravel()
        # 断言结果中的所有元素是否都为 0
        assert (result == 0).all()

    def test_floating_index_doc_example(self):
        # 测试浮点数索引的文档示例

        # 创建一个索引对象，包含浮点数
        index = Index([1.5, 2, 3, 4.5, 5])
        # 创建一个 Series 对象，使用给定的索引和值
        s = Series(range(5), index=index)
        # 断言使用整数索引访问 s 中的元素是否为预期值
        assert s[3] == 2
        # 断言使用 loc 方法的索引访问 s 中的元素是否为预期值
        assert s.loc[3] == 2
        # 断言使用 iloc 方法的索引访问 s 中的元素是否为预期值
        assert s.iloc[3] == 3
    # 定义一个测试方法，测试与浮点索引相关的情况
    def test_floating_misc(self, indexer_sl):
        # related 236
        # scalar/slicing of a float index
        # 创建一个 Series 对象，使用 numpy.arange 生成的数组作为数据，索引使用 np.arange(5) * 2.5，数据类型为 np.int64
        s = Series(np.arange(5), index=np.arange(5) * 2.5, dtype=np.int64)

        # 基于标签进行切片
        result = indexer_sl(s)[1.0:3.0]
        expected = Series(1, index=[2.5])
        # 断言切片后的结果与期望的 Series 相等
        tm.assert_series_equal(result, expected)

        # 精确索引查找
        result = indexer_sl(s)[5.0]
        # 断言查找结果为 2
        assert result == 2

        result = indexer_sl(s)[5]
        # 断言查找结果为 2
        assert result == 2

        # 当值未找到时抛出 KeyError（没有任何回退）
        
        # 标量整数索引，预期抛出 KeyError，错误信息匹配正则表达式 "^4$"
        with pytest.raises(KeyError, match=r"^4$"):
            indexer_sl(s)[4]

        # 浮点数和整数索引可以正确创建条目（作为 NaN）
        # 使用不同的浮点数和整数索引测试
        expected = Series([2, 0], index=Index([5.0, 0.0], dtype=np.float64))
        for fancy_idx in [[5.0, 0.0], np.array([5.0, 0.0])]:  # float
            tm.assert_series_equal(indexer_sl(s)[fancy_idx], expected)

        expected = Series([2, 0], index=Index([5, 0], dtype="float64"))
        for fancy_idx in [[5, 0], np.array([5, 0])]:
            tm.assert_series_equal(indexer_sl(s)[fancy_idx], expected)

        # 所有切片操作应该返回相同的结果，因为它们切片的是相同的区间
        result2 = indexer_sl(s)[2.0:5.0]
        result3 = indexer_sl(s)[2.0:5]
        result4 = indexer_sl(s)[2.1:5]
        tm.assert_series_equal(result2, result3)
        tm.assert_series_equal(result2, result4)

        # 列表选择
        result1 = indexer_sl(s)[[0.0, 5, 10]]
        result2 = s.iloc[[0, 2, 4]]
        # 断言列表选择后的结果与预期的 Series 相等
        tm.assert_series_equal(result1, result2)

        # 预期抛出 KeyError，错误信息为 "not in index"
        with pytest.raises(KeyError, match="not in index"):
            indexer_sl(s)[[1.6, 5, 10]]

        with pytest.raises(KeyError, match="not in index"):
            indexer_sl(s)[[0, 1, 2]]

        result = indexer_sl(s)[[2.5, 5]]
        tm.assert_series_equal(result, Series([1, 2], index=[2.5, 5.0]))

        result = indexer_sl(s)[[2.5]]
        tm.assert_series_equal(result, Series([1], index=[2.5]))
```