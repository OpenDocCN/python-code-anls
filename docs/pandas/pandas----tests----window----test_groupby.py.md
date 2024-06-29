# `D:\src\scipysrc\pandas\pandas\tests\window\test_groupby.py`

```
import numpy as np  # 导入NumPy库，用于数值计算
import pytest  # 导入pytest库，用于单元测试

from pandas import (  # 从pandas库中导入以下模块：
    DataFrame,  # 数据框架
    DatetimeIndex,  # 时间日期索引
    Index,  # 索引
    MultiIndex,  # 多级索引
    Series,  # 系列数据结构
    Timestamp,  # 时间戳
    date_range,  # 日期范围生成器
    to_datetime,  # 将对象转换为日期时间类型
)
import pandas._testing as tm  # 导入pandas测试模块作为tm别名
from pandas.api.indexers import BaseIndexer  # 从pandas库中的api.indexers模块导入BaseIndexer类
from pandas.core.groupby.groupby import get_groupby  # 从pandas库中的core.groupby.groupby模块导入get_groupby函数


@pytest.fixture  # 使用pytest装饰器定义测试夹具
def times_frame():  # 定义名为times_frame的测试数据框架夹具
    """Frame for testing times argument in EWM groupby."""  # 文档字符串，描述此测试数据框架的用途
    return DataFrame(  # 返回一个数据框架对象，包含以下列：
        {
            "A": ["a", "b", "c", "a", "b", "c", "a", "b", "c", "a"],  # 字符串列'A'
            "B": [0, 0, 0, 1, 1, 1, 2, 2, 2, 3],  # 整数列'B'
            "C": to_datetime(  # 日期时间类型列'C'，包含指定的日期字符串转换而来
                [
                    "2020-01-01",
                    "2020-01-01",
                    "2020-01-01",
                    "2020-01-02",
                    "2020-01-10",
                    "2020-01-22",
                    "2020-01-03",
                    "2020-01-23",
                    "2020-01-23",
                    "2020-01-04",
                ]
            ),
        }
    )


@pytest.fixture  # 使用pytest装饰器定义测试夹具
def roll_frame():  # 定义名为roll_frame的测试数据框架夹具
    return DataFrame({"A": [1] * 20 + [2] * 12 + [3] * 8, "B": np.arange(40)})  # 返回一个数据框架对象，包含两列：'A'列和'B'列


class TestRolling:  # 定义测试类TestRolling
    def test_groupby_unsupported_argument(self, roll_frame):  # 测试方法：测试groupby方法不支持的参数
        msg = r"groupby\(\) got an unexpected keyword argument 'foo'"  # 错误消息字符串，描述了不支持的参数情况
        with pytest.raises(TypeError, match=msg):  # 使用pytest的断言检查是否抛出TypeError异常，且异常消息符合msg字符串
            roll_frame.groupby("A", foo=1)  # 调用数据框架的groupby方法，传入参数'A'和不支持的参数'foo'

    def test_getitem(self, roll_frame):  # 测试方法：测试数据框架的getitem操作
        g = roll_frame.groupby("A")  # 对数据框架按照'A'列进行分组，返回分组对象g
        g_mutated = get_groupby(roll_frame, by="A")  # 使用get_groupby函数对数据框架进行'A'列的分组，返回分组对象g_mutated

        expected = g_mutated.B.apply(lambda x: x.rolling(2).mean())  # 使用g_mutated的B列，对每个分组应用滚动窗口为2的均值函数，得到期望的结果

        result = g.rolling(2).mean().B  # 对g对象应用滚动窗口为2的均值函数，并获取B列，得到结果
        tm.assert_series_equal(result, expected)  # 使用tm模块的assert_series_equal函数比较结果和期望，断言相等

        result = g.rolling(2).B.mean()  # 对g对象的B列应用滚动窗口为2的均值函数，得到结果
        tm.assert_series_equal(result, expected)  # 使用tm模块的assert_series_equal函数比较结果和期望，断言相等

        result = g.B.rolling(2).mean()  # 对g对象的B列应用滚动窗口为2的均值函数，得到结果
        tm.assert_series_equal(result, expected)  # 使用tm模块的assert_series_equal函数比较结果和期望，断言相等

        result = roll_frame.B.groupby(roll_frame.A).rolling(2).mean()  # 对数据框架按照A列分组后的B列，应用滚动窗口为2的均值函数，得到结果
        tm.assert_series_equal(result, expected)  # 使用tm模块的assert_series_equal函数比较结果和期望，断言相等

    def test_getitem_multiple(self, roll_frame):  # 测试方法：测试数据框架的getitem操作（多次）
        # GH 13174
        g = roll_frame.groupby("A")  # 对数据框架按照'A'列进行分组，返回分组对象g
        r = g.rolling(2, min_periods=0)  # 对g对象应用滚动窗口为2的滚动对象r，设置最小周期为0

        g_mutated = get_groupby(roll_frame, by="A")  # 使用get_groupby函数对数据框架进行'A'列的分组，返回分组对象g_mutated
        expected = g_mutated.B.apply(lambda x: x.rolling(2, min_periods=0).count())  # 使用g_mutated的B列，对每个分组应用滚动窗口为2的计数函数，得到期望的结果

        result = r.B.count()  # 对r对象的B列应用计数函数，得到结果
        tm.assert_series_equal(result, expected)  # 使用tm模块的assert_series_equal函数比较结果和期望，断言相等

        result = r.B.count()  # 对r对象的B列应用计数函数，得到结果
        tm.assert_series_equal(result, expected)  # 使用tm模块的assert_series_equal函数比较结果和期望，断言相等

    @pytest.mark.parametrize(  # 使用pytest的参数化标记，指定参数化测试的参数'f'
        "f",  # 参数名称为'f'
        [  # 参数列表，包含以下字符串元素：
            "sum",
            "mean",
            "min",
            "max",
            "count",
            "kurt",
            "skew",
        ],
    )
    # 定义一个测试方法，用于测试滚动窗口操作
    def test_rolling(self, f, roll_frame):
        # 根据列"A"对数据进行分组，不创建分组键
        g = roll_frame.groupby("A", group_keys=False)
        # 创建一个滚动窗口对象，窗口大小为4
        r = g.rolling(window=4)

        # 调用滚动窗口对象r的方法f，计算滚动窗口函数的结果
        result = getattr(r, f)()
        # 准备一个警告消息，用于断言是否产生特定的警告
        msg = "DataFrameGroupBy.apply operated on the grouping columns"
        # 断言在调用过程中是否产生了DeprecationWarning警告，且警告消息匹配msg
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            # 对分组g应用一个lambda函数，计算滚动窗口函数的结果
            expected = g.apply(lambda x: getattr(x.rolling(4), f)())
        # 由于groupby.apply不会丢弃分组列"A"，所以需要删除该列
        expected = expected.drop("A", axis=1)
        # GH 39732：准备预期的索引，包含"A"列和范围为40的整数
        expected_index = MultiIndex.from_arrays([roll_frame["A"], range(40)])
        # 将预期结果的索引设置为expected_index
        expected.index = expected_index
        # 使用断言方法tm.assert_frame_equal来比较结果和预期的DataFrame
        tm.assert_frame_equal(result, expected)

    # 使用pytest.mark.parametrize注释的参数f，进行参数化测试
    @pytest.mark.parametrize("f", ["std", "var"])
    # 定义一个测试方法，测试带有自由度修正的滚动窗口操作
    def test_rolling_ddof(self, f, roll_frame):
        # 根据列"A"对数据进行分组，不创建分组键
        g = roll_frame.groupby("A", group_keys=False)
        # 创建一个滚动窗口对象，窗口大小为4
        r = g.rolling(window=4)

        # 调用滚动窗口对象r的方法f，计算滚动窗口函数的结果，使用ddof=1进行自由度修正
        result = getattr(r, f)(ddof=1)
        # 准备一个警告消息，用于断言是否产生特定的警告
        msg = "DataFrameGroupBy.apply operated on the grouping columns"
        # 断言在调用过程中是否产生了DeprecationWarning警告，且警告消息匹配msg
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            # 对分组g应用一个lambda函数，计算滚动窗口函数的结果，使用ddof=1进行自由度修正
            expected = g.apply(lambda x: getattr(x.rolling(4), f)(ddof=1))
        # 由于groupby.apply不会丢弃分组列"A"，所以需要删除该列
        expected = expected.drop("A", axis=1)
        # GH 39732：准备预期的索引，包含"A"列和范围为40的整数
        expected_index = MultiIndex.from_arrays([roll_frame["A"], range(40)])
        # 将预期结果的索引设置为expected_index
        expected.index = expected_index
        # 使用断言方法tm.assert_frame_equal来比较结果和预期的DataFrame
        tm.assert_frame_equal(result, expected)

    # 使用pytest.mark.parametrize注释的参数interpolation，进行参数化测试
    @pytest.mark.parametrize(
        "interpolation", ["linear", "lower", "higher", "midpoint", "nearest"]
    )
    # 定义一个测试方法，测试滚动窗口的分位数操作
    def test_rolling_quantile(self, interpolation, roll_frame):
        # 根据列"A"对数据进行分组，不创建分组键
        g = roll_frame.groupby("A", group_keys=False)
        # 创建一个滚动窗口对象，窗口大小为4
        r = g.rolling(window=4)

        # 调用滚动窗口对象r的quantile方法，计算分位数为0.4的结果，指定插值方法为interpolation
        result = r.quantile(0.4, interpolation=interpolation)
        # 准备一个警告消息，用于断言是否产生特定的警告
        msg = "DataFrameGroupBy.apply operated on the grouping columns"
        # 断言在调用过程中是否产生了DeprecationWarning警告，且警告消息匹配msg
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            # 对分组g应用一个lambda函数，计算滚动窗口quantile的结果，指定插值方法为interpolation
            expected = g.apply(
                lambda x: x.rolling(4).quantile(0.4, interpolation=interpolation)
            )
        # 由于groupby.apply不会丢弃分组列"A"，所以需要删除该列
        expected = expected.drop("A", axis=1)
        # GH 39732：准备预期的索引，包含"A"列和范围为40的整数
        expected_index = MultiIndex.from_arrays([roll_frame["A"], range(40)])
        # 将预期结果的索引设置为expected_index
        expected.index = expected_index
        # 使用断言方法tm.assert_frame_equal来比较结果和预期的DataFrame
        tm.assert_frame_equal(result, expected)
    # 定义一个测试函数，用于测试滚动相关性和协方差，其他数据与分组大小相同
    def test_rolling_corr_cov_other_same_size_as_groups(self, f, expected_val):
        # 创建一个 DataFrame 包含三列：'value'列为0到9的整数，'idx1'列为[1]*5 + [2]*5，'idx2'列为[1, 2, 3, 4, 5]*2，并设置多重索引['idx1', 'idx2']
        df = DataFrame(
            {"value": range(10), "idx1": [1] * 5 + [2] * 5, "idx2": [1, 2, 3, 4, 5] * 2}
        ).set_index(["idx1", "idx2"])
        # 创建另一个 DataFrame 包含两列：'value'列为0到4的整数，'idx2'列为[1, 2, 3, 4, 5]，并设置索引为'idx2'
        other = DataFrame({"value": range(5), "idx2": [1, 2, 3, 4, 5]}).set_index(
            "idx2"
        )
        # 调用 groupby(level=0).rolling(2) 后再调用 f 方法（如corr或cov），并传入 other DataFrame 进行计算，返回结果
        result = getattr(df.groupby(level=0).rolling(2), f)(other)
        # 期望的数据列表，包含 np.nan 和预期值 expected_val 的重复值
        expected_data = ([np.nan] + [expected_val] * 4) * 2
        # 创建期望结果的 DataFrame，包含'value'列，使用 MultiIndex 来指定复杂索引
        expected = DataFrame(
            expected_data,
            columns=["value"],
            index=MultiIndex.from_arrays(
                [
                    [1] * 5 + [2] * 5,
                    [1] * 5 + [2] * 5,
                    list(range(1, 6)) * 2,
                ],
                names=["idx1", "idx1", "idx2"],
            ),
        )
        # 使用 assert_frame_equal 来比较 result 和 expected 的内容是否一致
        tm.assert_frame_equal(result, expected)

    # 使用 pytest.mark.parametrize 来定义参数化测试，测试滚动相关性和协方差，其他数据与分组大小不同
    @pytest.mark.parametrize("f", ["corr", "cov"])
    def test_rolling_corr_cov_other_diff_size_as_groups(self, f, roll_frame):
        # 对 roll_frame 根据 'A' 列进行分组
        g = roll_frame.groupby("A")
        # 对分组后的结果应用 rolling(window=4) 操作
        r = g.rolling(window=4)
        # 调用 getattr(r, f)(roll_frame) 来计算滚动相关性或协方差，返回结果
        result = getattr(r, f)(roll_frame)
        
        # 定义一个函数 func(x)，内部调用 getattr(x.rolling(4), f)(roll_frame) 来计算滚动相关性或协方差
        def func(x):
            return getattr(x.rolling(4), f)(roll_frame)
        
        # 执行 groupby.apply(func) 操作，并使用 assert_produces_warning 检查是否产生了 DeprecationWarning 警告
        msg = "DataFrameGroupBy.apply operated on the grouping columns"
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            expected = g.apply(func)
        # GH 39591: 分组的列应该都是 np.nan
        expected["A"] = np.nan
        # 使用 assert_frame_equal 检查 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

    # 使用 pytest.mark.parametrize 来定义参数化测试，测试滚动相关性和协方差，以及在成对计算时的情况
    @pytest.mark.parametrize("f", ["corr", "cov"])
    def test_rolling_corr_cov_pairwise(self, f, roll_frame):
        # 对 roll_frame 根据 'A' 列进行分组
        g = roll_frame.groupby("A")
        # 对分组后的结果应用 rolling(window=4) 操作
        r = g.rolling(window=4)
        # 调用 getattr(r.B, f)(pairwise=True) 来计算 B 列的滚动相关性或协方差，返回结果为 Series
        result = getattr(r.B, f)(pairwise=True)
        
        # 定义一个函数 func(x)，内部调用 getattr(x.B.rolling(4), f)(pairwise=True) 来计算 B 列的滚动相关性或协方差
        def func(x):
            return getattr(x.B.rolling(4), f)(pairwise=True)
        
        # 执行 groupby.apply(func) 操作，并使用 assert_produces_warning 检查是否产生了 DeprecationWarning 警告
        msg = "DataFrameGroupBy.apply operated on the grouping columns"
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            expected = g.apply(func)
        # 使用 assert_series_equal 检查 result 和 expected 是否相等
        tm.assert_series_equal(result, expected)

    # 参数化测试，测试协方差和相关性函数的期望值
    @pytest.mark.parametrize(
        "func, expected_values",
        [("cov", [[1.0, 1.0], [1.0, 4.0]]), ("corr", [[1.0, 0.5], [0.5, 1.0]])],
    )
    # 定义一个测试函数，用于测试滚动相关性和协方差的计算，不要求顺序
    def test_rolling_corr_cov_unordered(self, func, expected_values):
        # 创建一个包含三列的数据帧，列 'a' 是分组键，列 'b' 和 'c' 是数值列
        df = DataFrame(
            {
                "a": ["g1", "g2", "g1", "g1"],
                "b": [0, 0, 1, 2],
                "c": [2, 0, 6, 4],
            }
        )
        # 按列 'a' 分组，并创建一个滚动对象，窗口大小为 3
        rol = df.groupby("a").rolling(3)
        # 调用传入的函数（func），计算滚动结果
        result = getattr(rol, func)()
        # 根据预期值构建一个数据帧，用于和结果比较
        expected = DataFrame(
            {
                "b": 4 * [np.nan] + expected_values[0] + 2 * [np.nan],
                "c": 4 * [np.nan] + expected_values[1] + 2 * [np.nan],
            },
            index=MultiIndex.from_tuples(
                [
                    ("g1", 0, "b"),
                    ("g1", 0, "c"),
                    ("g1", 2, "b"),
                    ("g1", 2, "c"),
                    ("g1", 3, "b"),
                    ("g1", 3, "c"),
                    ("g2", 1, "b"),
                    ("g2", 1, "c"),
                ],
                names=["a", None, None],
            ),
        )
        # 断言滚动计算的结果和预期结果相等
        tm.assert_frame_equal(result, expected)

    # 定义一个测试函数，测试滚动应用
    def test_rolling_apply(self, raw, roll_frame):
        # 按列 'A' 分组，不显示分组键
        g = roll_frame.groupby("A", group_keys=False)
        # 对分组后的结果创建一个滚动对象，窗口大小为 4
        r = g.rolling(window=4)

        # 使用 lambda 函数对滚动对象应用求和操作，raw 参数控制是否原始计算
        result = r.apply(lambda x: x.sum(), raw=raw)
        # 设置警告信息
        msg = "DataFrameGroupBy.apply operated on the grouping columns"
        # 断言应该产生 DeprecationWarning 警告并匹配给定消息
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            # 期望的结果是对分组后的数据帧应用双重滚动应用求和操作
            expected = g.apply(lambda x: x.rolling(4).apply(lambda y: y.sum(), raw=raw))
        # 从预期结果中删除列 'A'
        expected = expected.drop("A", axis=1)
        # 设置预期的索引，由 roll_frame 的列 'A' 和范围为 40 的索引数组构成
        expected_index = MultiIndex.from_arrays([roll_frame["A"], range(40)])
        expected.index = expected_index
        # 断言滚动应用的结果和预期结果相等
        tm.assert_frame_equal(result, expected)

    # 定义一个测试函数，测试滚动应用的可变性
    def test_rolling_apply_mutability(self):
        # 创建一个包含两列 'A' 和 'B' 的数据帧
        df = DataFrame({"A": ["foo"] * 3 + ["bar"] * 3, "B": [1] * 6})
        # 按列 'A' 分组
        g = df.groupby("A")

        # 创建一个多级索引，第一级是列 'A' 的值，第二级是范围为 0 到 5 的整数
        mi = MultiIndex.from_tuples(
            [("bar", 3), ("bar", 4), ("bar", 5), ("foo", 0), ("foo", 1), ("foo", 2)]
        )
        # 将第一级索引命名为 'A'，第二级索引为无名称
        mi.names = ["A", None]
        # 创建预期的数据帧，包含 'B' 列，索引为 mi
        expected = DataFrame([np.nan, 2.0, 2.0] * 2, columns=["B"], index=mi)

        # 对分组后的结果创建一个滚动对象，窗口大小为 2，并应用求和操作
        result = g.rolling(window=2).sum()
        # 断言滚动求和的结果和预期结果相等
        tm.assert_frame_equal(result, expected)

        # 对分组后的结果调用求和函数
        g.sum()

        # 再次确认没有发生突变
        result = g.rolling(window=2).sum()
        # 断言滚动求和的结果和预期结果相等
        tm.assert_frame_equal(result, expected)
    def test_groupby_rolling(self, expected_value, raw_value):
        # 定义一个测试函数，用于测试 groupby 和 rolling 的功能

        def isnumpyarray(x):
            # 检查 x 是否是 numpy 数组，返回 1 或 0
            return int(isinstance(x, np.ndarray))

        # 创建一个 DataFrame 包含 id 和 value 列
        df = DataFrame({"id": [1, 1, 1], "value": [1, 2, 3]})
        
        # 对 df 按 id 分组，然后对 value 列进行 rolling 操作，应用 isnumpyarray 函数
        result = df.groupby("id").value.rolling(1).apply(isnumpyarray, raw=raw_value)
        
        # 创建预期的 Series 对象，包含预期的值，使用 MultiIndex 设置索引
        expected = Series(
            [expected_value] * 3,
            index=MultiIndex.from_tuples(((1, 0), (1, 1), (1, 2)), names=["id", None]),
            name="value",
        )
        
        # 使用 pytest 的 assert_series_equal 函数比较 result 和 expected 是否相等
        tm.assert_series_equal(result, expected)

    def test_groupby_rolling_center_center(self):
        # 定义一个测试函数，用于测试带 center=True 和 window=3 的 groupby 和 rolling 功能
        
        # 创建一个 Series 对象，包含整数范围 1 到 5
        series = Series(range(1, 6))
        
        # 对 series 进行分组并进行 rolling 平均计算，窗口大小为 3，居中计算
        result = series.groupby(series).rolling(center=True, window=3).mean()
        
        # 创建预期的 Series 对象，包含 NaN 值，使用 MultiIndex 设置索引
        expected = Series(
            [np.nan] * 5,
            index=MultiIndex.from_tuples(((1, 0), (2, 1), (3, 2), (4, 3), (5, 4))),
        )
        
        # 使用 pytest 的 assert_series_equal 函数比较 result 和 expected 是否相等
        tm.assert_series_equal(result, expected)

        # 创建另一个 Series 对象，包含整数范围 1 到 4
        series = Series(range(1, 5))
        
        # 对 series 进行分组并进行 rolling 平均计算，窗口大小为 3，居中计算
        result = series.groupby(series).rolling(center=True, window=3).mean()
        
        # 创建预期的 Series 对象，包含 NaN 值，使用 MultiIndex 设置索引
        expected = Series(
            [np.nan] * 4,
            index=MultiIndex.from_tuples(((1, 0), (2, 1), (3, 2), (4, 3))),
        )
        
        # 使用 pytest 的 assert_series_equal 函数比较 result 和 expected 是否相等
        tm.assert_series_equal(result, expected)

        # 创建一个 DataFrame 包含列 'a' 和 'b'，'a' 列包含 'a' 和 'b' 的值，'b' 列包含整数范围 0 到 10
        df = DataFrame({"a": ["a"] * 5 + ["b"] * 6, "b": range(11)})
        
        # 对 df 按 'a' 列进行分组并进行 rolling 平均计算，窗口大小为 3，居中计算
        result = df.groupby("a").rolling(center=True, window=3).mean()
        
        # 创建预期的 DataFrame 对象，包含 NaN 值，使用 MultiIndex 设置索引和列名
        expected = DataFrame(
            [np.nan, 1, 2, 3, np.nan, np.nan, 6, 7, 8, 9, np.nan],
            index=MultiIndex.from_tuples(
                (
                    ("a", 0),
                    ("a", 1),
                    ("a", 2),
                    ("a", 3),
                    ("a", 4),
                    ("b", 5),
                    ("b", 6),
                    ("b", 7),
                    ("b", 8),
                    ("b", 9),
                    ("b", 10),
                ),
                names=["a", None],
            ),
            columns=["b"],
        )
        
        # 使用 pytest 的 assert_frame_equal 函数比较 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

        # 创建另一个 DataFrame 包含列 'a' 和 'b'，'a' 列包含 'a' 和 'b' 的值，'b' 列包含整数范围 0 到 9
        df = DataFrame({"a": ["a"] * 5 + ["b"] * 5, "b": range(10)})
        
        # 对 df 按 'a' 列进行分组并进行 rolling 平均计算，窗口大小为 3，居中计算
        result = df.groupby("a").rolling(center=True, window=3).mean()
        
        # 创建预期的 DataFrame 对象，包含 NaN 值，使用 MultiIndex 设置索引和列名
        expected = DataFrame(
            [np.nan, 1, 2, 3, np.nan, np.nan, 6, 7, 8, np.nan],
            index=MultiIndex.from_tuples(
                (
                    ("a", 0),
                    ("a", 1),
                    ("a", 2),
                    ("a", 3),
                    ("a", 4),
                    ("b", 5),
                    ("b", 6),
                    ("b", 7),
                    ("b", 8),
                    ("b", 9),
                ),
                names=["a", None],
            ),
            columns=["b"],
        )
        
        # 使用 pytest 的 assert_frame_equal 函数比较 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)
    # 定义一个测试函数，测试 groupby 和 rolling center 参数的使用
    def test_groupby_rolling_center_on(self):
        # GH 37141
        # 创建一个 DataFrame 包含 Date、gb 和 value 列
        df = DataFrame(
            data={
                "Date": date_range("2020-01-01", "2020-01-10"),
                "gb": ["group_1"] * 6 + ["group_2"] * 4,
                "value": range(10),
            }
        )
        # 对 DataFrame 按 gb 列分组，然后在 Date 列上应用 rolling 方法，计算 value 列的均值
        result = (
            df.groupby("gb")
            .rolling(6, on="Date", center=True, min_periods=1)
            .value.mean()
        )
        # 创建一个 MultiIndex，用 gb 和 Date 列作为索引名
        mi = MultiIndex.from_arrays([df["gb"], df["Date"]], names=["gb", "Date"])
        # 创建一个预期的 Series，包含计算好的均值，以及对应的索引
        expected = Series(
            [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 7.0, 7.5, 7.5, 7.5],
            name="value",
            index=mi,
        )
        # 使用 pytest 的 assert_series_equal 函数检查 result 和 expected 是否相等
        tm.assert_series_equal(result, expected)

    # 参数化测试函数，测试 groupby 和 rolling center、min_periods 参数的使用
    @pytest.mark.parametrize("min_periods", [5, 4, 3])
    def test_groupby_rolling_center_min_periods(self, min_periods):
        # GH 36040
        # 创建一个包含 group 和 data 列的 DataFrame
        df = DataFrame({"group": ["A"] * 10 + ["B"] * 10, "data": range(20)})

        window_size = 5
        # 对 DataFrame 按 group 列分组，然后在 data 列上应用 rolling 方法，计算均值
        result = (
            df.groupby("group")
            .rolling(window_size, center=True, min_periods=min_periods)
            .mean()
        )
        # 重置索引，只保留 group 和 data 列
        result = result.reset_index()[["group", "data"]]

        # 创建预期的均值列表，分别对应 group A 和 group B
        grp_A_mean = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 7.5, 8.0]
        grp_B_mean = [x + 10.0 for x in grp_A_mean]

        # 计算 NaN 值的数量，用于 window_size 为 5 的情况
        num_nans = max(0, min_periods - 3)
        nans = [np.nan] * num_nans
        # 创建预期的 DataFrame，包含 group 和 data 列，以及对应的均值数据
        grp_A_expected = nans + grp_A_mean[num_nans : 10 - num_nans] + nans
        grp_B_expected = nans + grp_B_mean[num_nans : 10 - num_nans] + nans
        expected = DataFrame(
            {"group": ["A"] * 10 + ["B"] * 10, "data": grp_A_expected + grp_B_expected}
        )
        # 使用 pytest 的 assert_frame_equal 函数检查 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

    # 测试函数，测试 groupby 和 rolling 方法在子选择数据上的应用
    def test_groupby_subselect_rolling(self):
        # GH 35486
        # 创建一个包含 a、b 和 c 列的 DataFrame
        df = DataFrame(
            {"a": [1, 2, 3, 2], "b": [4.0, 2.0, 3.0, 1.0], "c": [10, 20, 30, 20]}
        )
        # 对 DataFrame 按 a 列分组，然后在 b 列上应用 rolling 方法，计算最大值
        result = df.groupby("a")[["b"]].rolling(2).max()
        # 创建预期的 DataFrame，包含计算好的最大值，以及对应的 MultiIndex 索引
        expected = DataFrame(
            [np.nan, np.nan, 2.0, np.nan],
            columns=["b"],
            index=MultiIndex.from_tuples(
                ((1, 0), (2, 1), (2, 3), (3, 2)), names=["a", None]
            ),
        )
        # 使用 pytest 的 assert_frame_equal 函数检查 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

        # 对 DataFrame 按 a 列分组，然后在 b 列上应用 rolling 方法，计算最大值
        result = df.groupby("a")["b"].rolling(2).max()
        # 创建预期的 Series，包含计算好的最大值，以及对应的 MultiIndex 索引
        expected = Series(
            [np.nan, np.nan, 2.0, np.nan],
            index=MultiIndex.from_tuples(
                ((1, 0), (2, 1), (2, 3), (3, 2)), names=["a", None]
            ),
            name="b",
        )
        # 使用 pytest 的 assert_series_equal 函数检查 result 和 expected 是否相等
        tm.assert_series_equal(result, expected)
    def test_groupby_rolling_custom_indexer(self):
        # GH 35557
        # 定义一个自定义索引器类 SimpleIndexer，继承自 BaseIndexer
        class SimpleIndexer(BaseIndexer):
            # 实现获取滚动窗口边界的方法
            def get_window_bounds(
                self,
                num_values=0,
                min_periods=None,
                center=None,
                closed=None,
                step=None,
            ):
                # 设置最小期数为窗口大小，如果未指定则为 0
                min_periods = self.window_size if min_periods is None else 0
                # 计算窗口的结束位置
                end = np.arange(num_values, dtype=np.int64) + 1
                # 计算窗口的起始位置
                start = end - self.window_size
                # 将起始位置小于 0 的值设置为最小期数
                start[start < 0] = min_periods
                # 返回计算得到的窗口起始位置和结束位置
                return start, end

        # 创建一个 DataFrame 对象 df，包含一列名为 'a' 的数据，重复五次，并指定不同的索引
        df = DataFrame(
            {"a": [1.0, 2.0, 3.0, 4.0, 5.0] * 3}, index=[0] * 5 + [1] * 5 + [2] * 5
        )
        # 对 DataFrame 按索引分组，并使用自定义索引器 SimpleIndexer 创建滚动窗口，计算滚动和
        result = (
            df.groupby(df.index)
            .rolling(SimpleIndexer(window_size=3), min_periods=1)
            .sum()
        )
        # 创建预期结果，对 DataFrame 按索引分组，使用内置方法 rolling 创建滚动窗口，计算滚动和
        expected = df.groupby(df.index).rolling(window=3, min_periods=1).sum()
        # 使用测试工具函数检查 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

    def test_groupby_rolling_subset_with_closed(self):
        # GH 35549
        # 创建 DataFrame 对象 df，包含多列数据和一列日期数据
        df = DataFrame(
            {
                "column1": range(8),
                "column2": range(8),
                "group": ["A"] * 4 + ["B"] * 4,
                "date": [
                    Timestamp(date)
                    for date in ["2019-01-01", "2019-01-01", "2019-01-02", "2019-01-02"]
                ]
                * 2,
            }
        )
        # 对 DataFrame 按 'group' 列分组，使用日期 'date' 创建 1 天的滚动窗口，并指定闭合方式为左闭合，计算 'column1' 列的和
        result = (
            df.groupby("group").rolling("1D", on="date", closed="left")["column1"].sum()
        )
        # 创建预期结果，为 Series 对象，包含按 'group' 和 'date' 分层索引的 'column1' 列的和
        expected = Series(
            [np.nan, np.nan, 1.0, 1.0, np.nan, np.nan, 9.0, 9.0],
            index=MultiIndex.from_frame(
                df[["group", "date"]],
                names=["group", "date"],
            ),
            name="column1",
        )
        # 使用测试工具函数检查 result 和 expected 是否相等
        tm.assert_series_equal(result, expected)

    def test_groupby_subset_rolling_subset_with_closed(self):
        # GH 35549
        # 创建 DataFrame 对象 df，包含多列数据和一列日期数据
        df = DataFrame(
            {
                "column1": range(8),
                "column2": range(8),
                "group": ["A"] * 4 + ["B"] * 4,
                "date": [
                    Timestamp(date)
                    for date in ["2019-01-01", "2019-01-01", "2019-01-02", "2019-01-02"]
                ]
                * 2,
            }
        )

        # 对 DataFrame 按 'group' 列分组，选取 'column1' 和 'date' 列，使用日期 'date' 创建 1 天的滚动窗口，并指定闭合方式为左闭合，计算 'column1' 列的和
        result = (
            df.groupby("group")[["column1", "date"]]
            .rolling("1D", on="date", closed="left")["column1"]
            .sum()
        )
        # 创建预期结果，为 Series 对象，包含按 'group' 和 'date' 分层索引的 'column1' 列的和
        expected = Series(
            [np.nan, np.nan, 1.0, 1.0, np.nan, np.nan, 9.0, 9.0],
            index=MultiIndex.from_frame(
                df[["group", "date"]],
                names=["group", "date"],
            ),
            name="column1",
        )
        # 使用测试工具函数检查 result 和 expected 是否相等
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize("func", ["max", "min"])
    def test_groupby_rolling_index_changed(self, func):
        # GH: #36018 nlevels of MultiIndex changed
        # 创建一个包含三个元素的Series，具有MultiIndex作为索引，名为'a'
        ds = Series(
            [1, 2, 2],
            index=MultiIndex.from_tuples(
                [("a", "x"), ("a", "y"), ("c", "z")], names=["1", "2"]
            ),
            name="a",
        )

        # 对Series进行分组并进行滚动计算，应用指定的函数func
        result = getattr(ds.groupby(ds).rolling(2), func)()
        
        # 创建预期的Series，具有MultiIndex作为索引，名为'a'
        expected = Series(
            [np.nan, np.nan, 2.0],
            index=MultiIndex.from_tuples(
                [(1, "a", "x"), (2, "a", "y"), (2, "c", "z")], names=["a", "1", "2"]
            ),
            name="a",
        )
        
        # 断言计算结果与预期结果相等
        tm.assert_series_equal(result, expected)

    def test_groupby_rolling_empty_frame(self):
        # GH 36197
        # 创建一个空的DataFrame，列名为's1'
        expected = DataFrame({"s1": []})
        
        # 对DataFrame进行按's1'列分组，并进行滚动窗口大小为1的求和操作
        result = expected.groupby("s1").rolling(window=1).sum()
        
        # 删除DataFrame中的's1'列
        expected = expected.drop(columns="s1")
        
        # 设置预期的索引，使用MultiIndex，其中包含空的Index对象
        expected.index = MultiIndex.from_product(
            [Index([], dtype="float64"), Index([], dtype="int64")], names=["s1", None]
        )
        
        # 断言计算结果与预期结果相等
        tm.assert_frame_equal(result, expected)

        # 创建一个空的DataFrame，列名为's1'和's2'
        expected = DataFrame({"s1": [], "s2": []})
        
        # 对DataFrame进行按's1'和's2'列分组，并进行滚动窗口大小为1的求和操作
        result = expected.groupby(["s1", "s2"]).rolling(window=1).sum()
        
        # 删除DataFrame中的's1'和's2'列
        expected = expected.drop(columns=["s1", "s2"])
        
        # 设置预期的索引，使用MultiIndex，其中包含空的Index对象
        expected.index = MultiIndex.from_product(
            [
                Index([], dtype="float64"),
                Index([], dtype="float64"),
                Index([], dtype="int64"),
            ],
            names=["s1", "s2", None],
        )
        
        # 断言计算结果与预期结果相等
        tm.assert_frame_equal(result, expected)
    def test_groupby_rolling_string_index(self):
        # GH: 36727
        # 创建一个DataFrame对象，包含索引、分组和时间戳
        df = DataFrame(
            [
                ["A", "group_1", Timestamp(2019, 1, 1, 9)],
                ["B", "group_1", Timestamp(2019, 1, 2, 9)],
                ["Z", "group_2", Timestamp(2019, 1, 3, 9)],
                ["H", "group_1", Timestamp(2019, 1, 6, 9)],
                ["E", "group_2", Timestamp(2019, 1, 20, 9)],
            ],
            columns=["index", "group", "eventTime"],
        ).set_index("index")
        
        # 根据分组字段"group"对DataFrame进行分组
        groups = df.groupby("group")
        
        # 在DataFrame中添加一个名为"count_to_date"的列，表示到目前为止每个组内的累计计数
        df["count_to_date"] = groups.cumcount()
        
        # 创建一个在"eventTime"列上滚动窗口为"10d"的滚动对象
        rolling_groups = groups.rolling("10d", on="eventTime")
        
        # 对滚动窗口应用一个函数，计算每个窗口内的行数，并将结果存储在result中
        result = rolling_groups.apply(lambda df: df.shape[0])
        
        # 创建一个期望的DataFrame，包含预期的结果，设置组和索引作为MultiIndex
        expected = DataFrame(
            [
                ["A", "group_1", Timestamp(2019, 1, 1, 9), 1.0],
                ["B", "group_1", Timestamp(2019, 1, 2, 9), 2.0],
                ["H", "group_1", Timestamp(2019, 1, 6, 9), 3.0],
                ["Z", "group_2", Timestamp(2019, 1, 3, 9), 1.0],
                ["E", "group_2", Timestamp(2019, 1, 20, 9), 1.0],
            ],
            columns=["index", "group", "eventTime", "count_to_date"],
        ).set_index(["group", "index"])
        
        # 使用assert_frame_equal函数比较结果和期望值是否相等
        tm.assert_frame_equal(result, expected)

    def test_groupby_rolling_no_sort(self):
        # GH 36889
        # 创建一个DataFrame对象，包含名为"foo"和"bar"的列
        result = (
            DataFrame({"foo": [2, 1], "bar": [2, 1]})
            # 根据"foo"列进行分组，但不进行排序
            .groupby("foo", sort=False)
            # 在每个分组上创建大小为1的滚动窗口，并计算每个窗口的最小值
            .rolling(1)
            .min()
        )
        
        # 创建一个期望的DataFrame，包含预期的结果，删除"foo"列
        expected = DataFrame(
            np.array([[2.0, 2.0], [1.0, 1.0]]),
            columns=["foo", "bar"],
            index=MultiIndex.from_tuples([(2, 0), (1, 1)], names=["foo", None]),
        )
        
        # 删除"foo"列后，使用assert_frame_equal函数比较结果和期望值是否相等
        tm.assert_frame_equal(result, expected)

    def test_groupby_rolling_count_closed_on(self, unit):
        # GH 35869
        # 创建一个DataFrame对象，包含列"column1"、"column2"、"group"和"date"
        df = DataFrame(
            {
                "column1": range(6),
                "column2": range(6),
                "group": 3 * ["A", "B"],
                "date": date_range(end="20190101", periods=6, unit=unit),
            }
        )
        
        # 根据"group"列进行分组
        result = (
            df.groupby("group")
            # 在"date"列上创建大小为"3d"的滚动窗口，左闭合，计算"column1"列每个窗口内的计数
            .rolling("3d", on="date", closed="left")["column1"]
            .count()
        )
        
        # 创建一个DatetimeIndex对象，表示日期范围
        dti = DatetimeIndex(
            [
                "2018-12-27",
                "2018-12-29",
                "2018-12-31",
                "2018-12-28",
                "2018-12-30",
                "2019-01-01",
            ],
            dtype=f"M8[{unit}]",
        )
        
        # 创建一个MultiIndex对象，包含"group"和"date"作为索引
        mi = MultiIndex.from_arrays(
            [
                ["A", "A", "A", "B", "B", "B"],
                dti,
            ],
            names=["group", "date"],
        )
        
        # 创建一个期望的Series对象，包含预期的结果
        expected = Series(
            [np.nan, 1.0, 1.0, np.nan, 1.0, 1.0],
            name="column1",
            index=mi,
        )
        
        # 使用assert_series_equal函数比较结果和期望值是否相等
        tm.assert_series_equal(result, expected)
    @pytest.mark.parametrize(
        ("func", "kwargs"),
        [("rolling", {"window": 2, "min_periods": 1}), ("expanding", {})],
    )
    # 使用 pytest 的参数化装饰器，定义多个测试参数组合
    def test_groupby_rolling_sem(self, func, kwargs):
        # GH: 26476
        # 创建一个 DataFrame 对象，包含指定数据和列名
        df = DataFrame(
            [["a", 1], ["a", 2], ["b", 1], ["b", 2], ["b", 3]], columns=["a", "b"]
        )
        # 对 DataFrame 按 'a' 列分组，然后应用指定的滚动函数并计算标准误差
        result = getattr(df.groupby("a"), func)(**kwargs).sem()
        # 创建预期的 DataFrame 对象，指定数据和索引
        expected = DataFrame(
            {"a": [np.nan] * 5, "b": [np.nan, 0.70711, np.nan, 0.70711, 0.70711]},
            index=MultiIndex.from_tuples(
                [("a", 0), ("a", 1), ("b", 2), ("b", 3), ("b", 4)], names=["a", None]
            ),
        )
        # GH 32262
        # 移除预期 DataFrame 的 'a' 列
        expected = expected.drop(columns="a")
        # 使用 pytest 的断言函数比较结果和预期值是否相等
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        ("rollings", "key"), [({"on": "a"}, "a"), ({"on": None}, "index")]
    )
    # 使用 pytest 的参数化装饰器，定义多个测试参数组合
    def test_groupby_rolling_nans_in_index(self, rollings, key):
        # GH: 34617
        # 创建一个 DataFrame 对象，包含指定数据和列名
        df = DataFrame(
            {
                "a": to_datetime(["2020-06-01 12:00", "2020-06-01 14:00", np.nan]),
                "b": [1, 2, 3],
                "c": [1, 1, 1],
            }
        )
        # 如果 key 是 'index'，则将 DataFrame 根据 'a' 列设置为索引
        if key == "index":
            df = df.set_index("a")
        # 使用 pytest 的断言函数检查是否抛出指定异常，验证是否包含 'NaT' 值
        with pytest.raises(ValueError, match=f"{key} values must not have NaT"):
            df.groupby("c").rolling("60min", **rollings)

    @pytest.mark.parametrize("group_keys", [True, False])
    # 使用 pytest 的参数化装饰器，定义多个测试参数组合
    def test_groupby_rolling_group_keys(self, group_keys):
        # GH 37641
        # GH 38523: GH 37641 actually was not a bug.
        # group_keys 只适用于直接应用 groupby.apply 方法
        # 创建一个多级索引的 Series 对象，包含指定数据和索引
        arrays = [["val1", "val1", "val2"], ["val1", "val1", "val2"]]
        index = MultiIndex.from_arrays(arrays, names=("idx1", "idx2"))

        s = Series([1, 2, 3], index=index)
        # 根据指定的键分组，并应用滚动窗口计算均值
        result = s.groupby(["idx1", "idx2"], group_keys=group_keys).rolling(1).mean()
        # 创建预期的 Series 对象，指定数据和多级索引
        expected = Series(
            [1.0, 2.0, 3.0],
            index=MultiIndex.from_tuples(
                [
                    ("val1", "val1", "val1", "val1"),
                    ("val1", "val1", "val1", "val1"),
                    ("val2", "val2", "val2", "val2"),
                ],
                names=["idx1", "idx2", "idx1", "idx2"],
            ),
        )
        # 使用 pytest 的断言函数比较结果和预期值是否相等
        tm.assert_series_equal(result, expected)
    def test_groupby_rolling_index_level_and_column_label(self):
        # 测试：按照指定索引级别和列标签进行分组滚动计算

        # 创建多级索引
        arrays = [["val1", "val1", "val2"], ["val1", "val1", "val2"]]
        index = MultiIndex.from_arrays(arrays, names=("idx1", "idx2"))

        # 创建 DataFrame
        df = DataFrame({"A": [1, 1, 2], "B": range(3)}, index=index)

        # 对 DataFrame 进行分组并进行滚动计算
        result = df.groupby(["idx1", "A"]).rolling(1).mean()

        # 期望的结果 DataFrame，包含指定的数据和多级索引
        expected = DataFrame(
            {"B": [0.0, 1.0, 2.0]},
            index=MultiIndex.from_tuples(
                [
                    ("val1", 1, "val1", "val1"),
                    ("val1", 1, "val1", "val1"),
                    ("val2", 2, "val2", "val2"),
                ],
                names=["idx1", "A", "idx1", "idx2"],
            ),
        )

        # 使用测试框架检查结果是否与期望相符
        tm.assert_frame_equal(result, expected)

    def test_groupby_rolling_resulting_multiindex(self):
        # 测试：检查结果多级索引的创建情况，包括不同的案例

        # 创建 DataFrame
        df = DataFrame({"a": np.arange(8.0), "b": [1, 2] * 4})

        # 对 DataFrame 进行分组并进行滚动计算
        result = df.groupby("b").rolling(3).mean()

        # 期望的索引
        expected_index = MultiIndex.from_tuples(
            [(1, 0), (1, 2), (1, 4), (1, 6), (2, 1), (2, 3), (2, 5), (2, 7)],
            names=["b", None],
        )

        # 使用测试框架检查结果的索引是否与期望相符
        tm.assert_index_equal(result.index, expected_index)

    def test_groupby_rolling_resulting_multiindex2(self):
        # 测试：按两个列进行分组，检查结果的三级多级索引的创建情况

        # 创建 DataFrame
        df = DataFrame({"a": np.arange(12.0), "b": [1, 2] * 6, "c": [1, 2, 3, 4] * 3})

        # 对 DataFrame 进行分组并进行滚动求和计算
        result = df.groupby(["b", "c"]).rolling(2).sum()

        # 期望的索引
        expected_index = MultiIndex.from_tuples(
            [
                (1, 1, 0),
                (1, 1, 4),
                (1, 1, 8),
                (1, 3, 2),
                (1, 3, 6),
                (1, 3, 10),
                (2, 2, 1),
                (2, 2, 5),
                (2, 2, 9),
                (2, 4, 3),
                (2, 4, 7),
                (2, 4, 11),
            ],
            names=["b", "c", None],
        )

        # 使用测试框架检查结果的索引是否与期望相符
        tm.assert_index_equal(result.index, expected_index)

    def test_groupby_rolling_resulting_multiindex3(self):
        # 测试：在具有二级多级索引的 DataFrame 上进行单级分组，检查结果的三级多级索引的创建情况

        # 创建 DataFrame
        df = DataFrame({"a": np.arange(8.0), "b": [1, 2] * 4, "c": [1, 2, 3, 4] * 2})

        # 将 "c" 列设置为二级索引
        df = df.set_index("c", append=True)

        # 对 DataFrame 进行分组并进行滚动计算
        result = df.groupby("b").rolling(3).mean()

        # 期望的索引
        expected_index = MultiIndex.from_tuples(
            [
                (1, 0, 1),
                (1, 2, 3),
                (1, 4, 1),
                (1, 6, 3),
                (2, 1, 2),
                (2, 3, 4),
                (2, 5, 2),
                (2, 7, 4),
            ],
            names=["b", None, "c"],
        )

        # 使用测试框架检查结果的索引是否与期望相符
        tm.assert_index_equal(result.index, expected_index, exact="equiv")
    def test_groupby_rolling_object_doesnt_affect_groupby_apply(self, roll_frame):
        # GH 39732
        # 使用 "A" 列进行分组，不在结果中包含组键列
        g = roll_frame.groupby("A", group_keys=False)
        # 设置警告消息，用于断言 DataFrameGroupBy.apply 操作在分组列上
        msg = "DataFrameGroupBy.apply operated on the grouping columns"
        # 断言警告消息为 DeprecationWarning 类型，并匹配特定消息
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            # 对分组后的数据应用滚动窗口为 4 的滑动求和，并获取索引
            expected = g.apply(lambda x: x.rolling(4).sum()).index
        # 创建一个滚动窗口对象，窗口大小为 4
        _ = g.rolling(window=4)
        # 再次设置警告消息，用于断言 DataFrameGroupBy.apply 操作在分组列上
        msg = "DataFrameGroupBy.apply operated on the grouping columns"
        # 断言警告消息为 DeprecationWarning 类型，并匹配特定消息
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            # 对分组后的数据应用滚动窗口为 4 的滑动求和，并获取索引
            result = g.apply(lambda x: x.rolling(4).sum()).index
        # 断言结果的索引与预期相等
        tm.assert_index_equal(result, expected)

    @pytest.mark.parametrize(
        ("window", "min_periods", "closed", "expected"),
        [
            # 参数化测试用例，测试不同的窗口、最小期数和闭合方式
            (2, 0, "left", [None, 0.0, 1.0, 1.0, None, 0.0, 1.0, 1.0]),
            (2, 2, "left", [None, None, 1.0, 1.0, None, None, 1.0, 1.0]),
            (4, 4, "left", [None, None, None, None, None, None, None, None]),
            (4, 4, "right", [None, None, None, 5.0, None, None, None, 5.0]),
        ],
    )
    def test_groupby_rolling_var(self, window, min_periods, closed, expected):
        # 创建包含 1 到 8 的 DataFrame 对象
        df = DataFrame([1, 2, 3, 4, 5, 6, 7, 8])
        # 对 DataFrame 按照列 [1, 2, 1, 2, 1, 2, 1, 2] 进行分组，并应用滚动窗口和方差计算
        result = (
            df.groupby([1, 2, 1, 2, 1, 2, 1, 2])
            .rolling(window=window, min_periods=min_periods, closed=closed)
            .var(0)
        )
        # 创建预期的 DataFrame 结果，包含特定的浮点数数组和多级索引
        expected_result = DataFrame(
            np.array(expected, dtype="float64"),
            index=MultiIndex(
                levels=[np.array([1, 2]), [0, 1, 2, 3, 4, 5, 6, 7]],
                codes=[[0, 0, 0, 0, 1, 1, 1, 1], [0, 2, 4, 6, 1, 3, 5, 7]],
            ),
        )
        # 断言计算结果与预期结果相等
        tm.assert_frame_equal(result, expected_result)

    @pytest.mark.parametrize(
        "columns", [MultiIndex.from_tuples([("A", ""), ("B", "C")]), ["A", "B"]]
    )
    def test_by_column_not_in_values(self, columns):
        # GH 32262
        # 创建一个 DataFrame 对象，包含指定的数据和列名
        df = DataFrame([[1, 0]] * 20 + [[2, 0]] * 12 + [[3, 0]] * 8, columns=columns)
        # 对 DataFrame 按照 "A" 列进行分组
        g = df.groupby("A")
        # 备份原始的分组对象
        original_obj = g.obj.copy(deep=True)
        # 创建一个窗口大小为 4 的滚动窗口对象
        r = g.rolling(4)
        # 对滚动窗口对象应用求和操作
        result = r.sum()
        # 断言结果中不包含 "A" 列
        assert "A" not in result.columns
        # 断言分组对象与原始对象相等
        tm.assert_frame_equal(g.obj, original_obj)
    def test_groupby_level(self):
        # 定义一个包含多个数组的列表，每个数组表示不同的组合
        arrays = [
            ["Falcon", "Falcon", "Parrot", "Parrot"],
            ["Captive", "Wild", "Captive", "Wild"],
        ]
        # 从给定的数组创建一个多级索引对象，并指定各级索引的名称
        index = MultiIndex.from_arrays(arrays, names=("Animal", "Type"))
        # 创建一个数据帧，包含"Max Speed"列，以及使用之前创建的多级索引
        df = DataFrame({"Max Speed": [390.0, 350.0, 30.0, 20.0]}, index=index)
        # 对数据帧按照第一级索引进行分组，然后对"Max Speed"列执行滚动窗口求和操作
        result = df.groupby(level=0)["Max Speed"].rolling(2).sum()
        # 创建预期结果的序列，包含与测试数据帧相同的索引结构和名称
        expected = Series(
            [np.nan, 740.0, np.nan, 50.0],
            index=MultiIndex.from_tuples(
                [
                    ("Falcon", "Falcon", "Captive"),
                    ("Falcon", "Falcon", "Wild"),
                    ("Parrot", "Parrot", "Captive"),
                    ("Parrot", "Parrot", "Wild"),
                ],
                names=["Animal", "Animal", "Type"],
            ),
            name="Max Speed",
        )
        # 使用测试工具验证计算结果与预期结果是否一致
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "by, expected_data",
        [
            [["id"], {"num": [100.0, 150.0, 150.0, 200.0]}],
            [
                ["id", "index"],
                {
                    "date": [
                        Timestamp("2018-01-01"),
                        Timestamp("2018-01-02"),
                        Timestamp("2018-01-01"),
                        Timestamp("2018-01-02"),
                    ],
                    "num": [100.0, 200.0, 150.0, 250.0],
                },
            ],
        ],
    )
    def test_as_index_false(self, by, expected_data, unit):
        # GH 39433
        # 定义一个包含测试数据的列表，每个元素表示一行数据
        data = [
            ["A", "2018-01-01", 100.0],
            ["A", "2018-01-02", 200.0],
            ["B", "2018-01-01", 150.0],
            ["B", "2018-01-02", 250.0],
        ]
        # 创建数据帧，指定列名为"id", "date", "num"，并将"date"列转换为指定的日期时间单位
        df = DataFrame(data, columns=["id", "date", "num"])
        df["date"] = df["date"].astype(f"M8[{unit}]")
        # 将指定列作为索引设置给数据帧
        df = df.set_index(["date"])

        # 根据参数列表中指定的列属性，构建分组依据的列表
        gp_by = [getattr(df, attr) for attr in by]
        # 对数据帧按照分组依据进行分组，并对滚动窗口内的数据执行均值计算
        result = (
            df.groupby(gp_by, as_index=False).rolling(window=2, min_periods=1).mean()
        )

        # 创建预期的数据帧，包含指定的列和索引
        expected = {"id": ["A", "A", "B", "B"]}
        expected.update(expected_data)
        expected = DataFrame(
            expected,
            index=df.index,
        )
        # 如果预期数据包含"date"列，则将其转换为指定的日期时间单位
        if "date" in expected_data:
            expected["date"] = expected["date"].astype(f"M8[{unit}]")
        # 使用测试工具验证计算结果与预期结果是否一致
        tm.assert_frame_equal(result, expected)
    def test_nan_and_zero_endpoints(self, any_int_numpy_dtype):
        # 测试函数，用于检验处理 NaN 和零端点的情况
        # https://github.com/twosigma/pandas/issues/53

        # 根据输入的任意整数类型 numpy dtype 创建对应的类型对象
        typ = np.dtype(any_int_numpy_dtype).type
        # 设定数组大小为 1000
        size = 1000
        # 创建一个包含 size 个 typ(0) 元素的数组
        idx = np.repeat(typ(0), size)
        # 将数组的最后一个元素设置为 1
        idx[-1] = 1

        # 设定一个很大的值
        val = 5e25
        # 创建一个包含 size 个 val 元素的数组
        arr = np.repeat(val, size)
        # 将数组的第一个元素设置为 NaN
        arr[0] = np.nan
        # 将数组的最后一个元素设置为 0
        arr[-1] = 0

        # 创建 DataFrame 对象，其中包含两列：'index' 和 'adl2'
        df = DataFrame(
            {
                "index": idx,
                "adl2": arr,
            }
        ).set_index("index")

        # 对 DataFrame 进行分组，按照 'index' 列进行分组，计算 'adl2' 列的滚动均值
        result = df.groupby("index")["adl2"].rolling(window=10, min_periods=1).mean()

        # 创建预期的 Series 对象，包含与 arr 相同的数据，并使用 MultiIndex 进行索引
        expected = Series(
            arr,
            name="adl2",
            index=MultiIndex.from_arrays(
                [
                    Index([0] * 999 + [1], dtype=typ, name="index"),
                    Index([0] * 999 + [1], dtype=typ, name="index"),
                ],
            ),
        )

        # 使用测试框架检查 result 和 expected 是否相等
        tm.assert_series_equal(result, expected)

    def test_groupby_rolling_non_monotonic(self):
        # 测试函数，用于检验非单调情况下的分组滚动操作
        # GH 43909

        # 创建一个打乱顺序的列表
        shuffled = [3, 0, 1, 2]
        # 设置时间间隔为 1000
        sec = 1_000
        # 创建 DataFrame 对象，包含 't', 'x', 'c' 列，'t' 列为时间戳
        df = DataFrame(
            [{"t": Timestamp(2 * x * sec), "x": x + 1, "c": 42} for x in shuffled]
        )
        # 使用 pytest 框架断言应该引发 ValueError 异常，且异常消息匹配正则表达式 ".* must be monotonic"
        with pytest.raises(ValueError, match=r".* must be monotonic"):
            # 对 DataFrame 按照 'c' 列进行分组，然后执行滚动窗口操作，按照 't' 列进行滚动
            df.groupby("c").rolling(on="t", window="3s")

    def test_groupby_monotonic(self):
        # 测试函数，用于检验单调情况下的分组滚动操作
        # GH 15130
        # 我们在分组时不需要验证单调性

        # GH 43909，我们应该在这里引发一个错误以匹配非分组滚动的行为。

        # 创建包含数据的列表
        data = [
            ["David", "1/1/2015", 100],
            ["David", "1/5/2015", 500],
            ["David", "5/30/2015", 50],
            ["David", "7/25/2015", 50],
            ["Ryan", "1/4/2014", 100],
            ["Ryan", "1/19/2015", 500],
            ["Ryan", "3/31/2016", 50],
            ["Joe", "7/1/2015", 100],
            ["Joe", "9/9/2015", 500],
            ["Joe", "10/15/2015", 50],
        ]

        # 创建 DataFrame 对象，指定列名为 ["name", "date", "amount"]
        df = DataFrame(data=data, columns=["name", "date", "amount"])
        # 将 'date' 列转换为日期时间格式
        df["date"] = to_datetime(df["date"])
        # 根据 'date' 列对 DataFrame 进行排序
        df = df.sort_values("date")

        # 创建一个警告消息，指出 DataFrameGroupBy.apply 操作作用于分组列上
        msg = "DataFrameGroupBy.apply operated on the grouping columns"
        # 使用测试框架检查是否产生 DeprecationWarning 警告，且警告消息与正则表达式匹配
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            # 创建预期的 Series 对象，对 'amount' 列执行滚动窗口操作，并使用 '180D' 时间跨度
            expected = (
                df.set_index("date")
                .groupby("name")
                .apply(lambda x: x.rolling("180D")["amount"].sum())
            )
        # 对 DataFrame 按照 'name' 列进行分组，然后对 'amount' 列执行滚动窗口操作，时间跨度为 '180D'
        result = df.groupby("name").rolling("180D", on="date")["amount"].sum()
        # 使用测试框架检查 result 和 expected 是否相等
        tm.assert_series_equal(result, expected)
    def test_datelike_on_monotonic_within_each_group(self):
        # 测试在每个分组内日期类数据是否单调递增（类似于 GH 13966，由 #15130 关闭，由 #15175 修复）

        # 被 43909 取代
        # GH 46061: 如果日期类数据在每个分组内是单调递增的，则通过
        dates = date_range(start="2016-01-01 09:30:00", periods=20, freq="s")
        # 创建包含不同分组的 DataFrame
        df = DataFrame(
            {
                "A": [1] * 20 + [2] * 12 + [3] * 8,
                "B": np.concatenate((dates, dates)),
                "C": np.arange(40),
            }
        )

        msg = "DataFrameGroupBy.apply operated on the grouping columns"
        # 断言产生警告信息
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            # 期望结果是对每个分组在时间窗口内进行滚动平均
            expected = (
                df.set_index("B")
                .groupby("A")
                .apply(lambda x: x.rolling("4s")["C"].mean())
            )
        # 对每个分组在时间窗口内进行滚动平均，计算结果
        result = df.groupby("A").rolling("4s", on="B").C.mean()
        # 断言结果是否与期望一致
        tm.assert_series_equal(result, expected)

    def test_datelike_on_not_monotonic_within_each_group(self):
        # GH 46061
        # 创建包含不同分组的 DataFrame，其中日期类数据在每个分组内不是单调递增的
        df = DataFrame(
            {
                "A": [1] * 3 + [2] * 3,
                "B": [Timestamp(year, 1, 1) for year in [2020, 2021, 2019]] * 2,
                "C": range(6),
            }
        )
        # 断言引发值错误，要求每个分组内的日期类数据必须是单调递增的
        with pytest.raises(ValueError, match="Each group within B must be monotonic."):
            df.groupby("A").rolling("365D", on="B")
    # 定义一个测试类 TestExpanding，用于测试扩展操作
    class TestExpanding:
        # 定义一个 pytest fixture，返回一个包含特定数据的 DataFrame 对象
        @pytest.fixture
        def frame(self):
            return DataFrame({"A": [1] * 20 + [2] * 12 + [3] * 8, "B": np.arange(40)})

        # 使用 pytest.mark.parametrize 装饰器指定多个参数化测试函数，每个测试函数测试不同的函数操作
        @pytest.mark.parametrize(
            "f", ["sum", "mean", "min", "max", "count", "kurt", "skew"]
        )
        # 定义测试函数 test_expanding，测试 DataFrameGroupBy 对象的扩展操作
        def test_expanding(self, f, frame):
            # 根据列"A"对 DataFrame 进行分组，不保留分组键
            g = frame.groupby("A", group_keys=False)
            # 获取扩展对象
            r = g.expanding()

            # 调用扩展对象的特定函数（如 sum、mean 等）
            result = getattr(r, f)()
            # 准备警告消息，用于检查是否触发特定警告
            msg = "DataFrameGroupBy.apply operated on the grouping columns"
            # 使用 assert_produces_warning 检查是否产生特定类型的警告
            with tm.assert_produces_warning(DeprecationWarning, match=msg):
                # 通过 apply 函数对每个分组应用扩展对象的特定函数
                expected = g.apply(lambda x: getattr(x.expanding(), f)())
            # 由于 groupby.apply 不会删除分组列"A"，需要手动删除
            expected = expected.drop("A", axis=1)
            # 准备预期结果的索引，结合原始 DataFrame 的"A"列和索引范围
            expected_index = MultiIndex.from_arrays([frame["A"], range(40)])
            expected.index = expected_index
            # 使用 assert_frame_equal 检查结果是否与预期一致
            tm.assert_frame_equal(result, expected)

        # 另一个参数化测试函数，测试带有自由度设置的扩展函数操作
        @pytest.mark.parametrize("f", ["std", "var"])
        def test_expanding_ddof(self, f, frame):
            g = frame.groupby("A", group_keys=False)
            r = g.expanding()

            result = getattr(r, f)(ddof=0)
            msg = "DataFrameGroupBy.apply operated on the grouping columns"
            with tm.assert_produces_warning(DeprecationWarning, match=msg):
                expected = g.apply(lambda x: getattr(x.expanding(), f)(ddof=0))
            expected = expected.drop("A", axis=1)
            expected_index = MultiIndex.from_arrays([frame["A"], range(40)])
            expected.index = expected_index
            tm.assert_frame_equal(result, expected)

        # 另一个参数化测试函数，测试扩展对象的分位数操作
        @pytest.mark.parametrize(
            "interpolation", ["linear", "lower", "higher", "midpoint", "nearest"]
        )
        def test_expanding_quantile(self, interpolation, frame):
            g = frame.groupby("A", group_keys=False)
            r = g.expanding()

            # 调用扩展对象的分位数计算函数
            result = r.quantile(0.4, interpolation=interpolation)
            msg = "DataFrameGroupBy.apply operated on the grouping columns"
            with tm.assert_produces_warning(DeprecationWarning, match=msg):
                expected = g.apply(
                    lambda x: x.expanding().quantile(0.4, interpolation=interpolation)
                )
            expected = expected.drop("A", axis=1)
            expected_index = MultiIndex.from_arrays([frame["A"], range(40)])
            expected.index = expected_index
            tm.assert_frame_equal(result, expected)

        # 未完全注释的参数化测试函数，测试扩展对象的相关性和协方差计算
        @pytest.mark.parametrize("f", ["corr", "cov"])
    # 定义一个测试函数，用于测试 DataFrameGroupBy 对象的 expending 方法对协方差和相关系数的计算
    def test_expanding_corr_cov(self, f, frame):
        # 按照列 "A" 进行分组，返回一个 GroupBy 对象
        g = frame.groupby("A")
        # 创建一个 ExpandingGroupby 对象，用于扩展操作
        r = g.expanding()

        # 调用 r 对象的指定方法（如 corr 或 cov），对整个 DataFrame 进行计算
        result = getattr(r, f)(frame)

        # 定义一个函数 func_0，对每个分组的 ExpandingGroupby 对象调用指定方法（如 corr 或 cov）
        def func_0(x):
            return getattr(x.expanding(), f)(frame)

        # 用 groupby.apply 方法调用 func_0 函数，对每个分组进行操作，生成期望的结果
        msg = "DataFrameGroupBy.apply operated on the grouping columns"
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            expected = g.apply(func_0)

        # 处理特定情况：当窗口内全部为 NaN 时，确保返回 NaN 而非 1
        null_idx = list(range(20, 61)) + list(range(72, 113))
        expected.iloc[null_idx, 1] = np.nan

        # 处理特定情况：确保分组列 "A" 全为 NaN
        expected["A"] = np.nan

        # 比较计算结果与期望结果是否一致
        tm.assert_frame_equal(result, expected)

        # 对 B 列进行类似的扩展操作，计算相关系数或协方差
        result = getattr(r.B, f)(pairwise=True)

        # 定义一个函数 func_1，对每个分组的 B 列的 ExpandingGroupby 对象调用指定方法（如 corr 或 cov）
        def func_1(x):
            return getattr(x.B.expanding(), f)(pairwise=True)

        # 用 groupby.apply 方法调用 func_1 函数，对每个分组的 B 列进行操作，生成期望的结果
        msg = "DataFrameGroupBy.apply operated on the grouping columns"
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            expected = g.apply(func_1)

        # 比较计算结果与期望结果是否一致
        tm.assert_series_equal(result, expected)

    # 定义一个测试函数，用于测试 DataFrameGroupBy 对象的 expending 方法对 apply 函数的支持
    def test_expanding_apply(self, raw, frame):
        # 按照列 "A" 进行分组，返回一个 GroupBy 对象，不显示分组键
        g = frame.groupby("A", group_keys=False)
        # 创建一个 ExpandingGroupby 对象，用于扩展操作
        r = g.expanding()

        # 使用 apply 方法对扩展窗口内的数据应用 lambda 函数（如求和）
        result = r.apply(lambda x: x.sum(), raw=raw)

        # 出现警告时的消息内容
        msg = "DataFrameGroupBy.apply operated on the grouping columns"
        with tm.assert_produces_warning(DeprecationWarning, match=msg):
            # 使用 groupby.apply 方法，对每个分组的 ExpandingGroupby 对象应用 lambda 函数（如求和）
            expected = g.apply(
                lambda x: x.expanding().apply(lambda y: y.sum(), raw=raw)
            )

        # 移除结果中的分组列 "A"
        expected = expected.drop("A", axis=1)

        # 处理特定情况：修复 GH 39732 的问题，重新设置期望结果的索引
        expected_index = MultiIndex.from_arrays([frame["A"], range(40)])
        expected.index = expected_index

        # 比较计算结果与期望结果是否一致
        tm.assert_frame_equal(result, expected)
class TestEWM:
    @pytest.mark.parametrize(
        "method, expected_data",
        [
            ["mean", [0.0, 0.6666666666666666, 1.4285714285714286, 2.2666666666666666]],
            ["std", [np.nan, 0.707107, 0.963624, 1.177164]],
            ["var", [np.nan, 0.5, 0.9285714285714286, 1.3857142857142857]],
        ],
    )
    def test_methods(self, method, expected_data):
        # GH 16037
        # 创建包含四行数据的 DataFrame，列"A"中全为"a"，列"B"从0到3
        df = DataFrame({"A": ["a"] * 4, "B": range(4)})
        # 对 DataFrame 按列"A"分组，并对每组进行指数加权移动平均(ewm)，然后取指定方法(method)的结果
        result = getattr(df.groupby("A").ewm(com=1.0), method)()
        # 创建期望的 DataFrame，包含预期的数据和索引
        expected = DataFrame(
            {"B": expected_data},
            index=MultiIndex.from_tuples(
                [
                    ("a", 0),
                    ("a", 1),
                    ("a", 2),
                    ("a", 3),
                ],
                names=["A", None],
            ),
        )
        # 使用测试框架比较结果和预期值的 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

    @pytest.mark.parametrize(
        "method, expected_data",
        [["corr", [np.nan, 1.0, 1.0, 1]], ["cov", [np.nan, 0.5, 0.928571, 1.385714]]],
    )
    def test_pairwise_methods(self, method, expected_data):
        # GH 16037
        # 创建包含四行数据的 DataFrame，列"A"中全为"a"，列"B"从0到3
        df = DataFrame({"A": ["a"] * 4, "B": range(4)})
        # 对 DataFrame 按列"A"分组，并对每组进行指数加权移动平均(ewm)，然后取指定方法(method)的结果
        result = getattr(df.groupby("A").ewm(com=1.0), method)()
        # 创建期望的 DataFrame，包含预期的数据和索引
        expected = DataFrame(
            {"B": expected_data},
            index=MultiIndex.from_tuples(
                [
                    ("a", 0, "B"),
                    ("a", 1, "B"),
                    ("a", 2, "B"),
                    ("a", 3, "B"),
                ],
                names=["A", None, None],
            ),
        )
        # 使用测试框架比较结果和预期值的 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

        # 重新计算预期值，对每个"A"组的"B"列应用指定方法(method)
        expected = df.groupby("A")[["B"]].apply(
            lambda x: getattr(x.ewm(com=1.0), method)()
        )
        # 再次使用测试框架比较结果和新的预期值的 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)

    def test_times(self, times_frame):
        # GH 40951
        halflife = "23 days"
        # GH#42738
        # 从 times_frame 中取出列"C"，并从 DataFrame 中删除
        times = times_frame.pop("C")
        # 对 times_frame 按"A"列分组，并对每组数据应用指数加权移动平均(ewm)，计算均值
        result = times_frame.groupby("A").ewm(halflife=halflife, times=times).mean()
        # 创建期望的 DataFrame，包含预期的数据和索引
        expected = DataFrame(
            {
                "B": [
                    0.0,
                    0.507534,
                    1.020088,
                    1.537661,
                    0.0,
                    0.567395,
                    1.221209,
                    0.0,
                    0.653141,
                    1.195003,
                ]
            },
            index=MultiIndex.from_tuples(
                [
                    ("a", 0),
                    ("a", 3),
                    ("a", 6),
                    ("a", 9),
                    ("b", 1),
                    ("b", 4),
                    ("b", 7),
                    ("c", 2),
                    ("c", 5),
                    ("c", 8),
                ],
                names=["A", None],
            ),
        )
        # 使用测试框架比较结果和预期值的 DataFrame 是否相等
        tm.assert_frame_equal(result, expected)
    # 测试函数，验证 times_frame 中 "C" 列弹出并存储在 times 变量中
    def test_times_array(self, times_frame):
        # GH 40951
        # 指定指数加权移动平均的半衰期
        halflife = "23 days"
        # 弹出 times_frame 中的 "C" 列，并将其存储在 times 变量中
        times = times_frame.pop("C")
        # 根据 "A" 列对 times_frame 进行分组
        gb = times_frame.groupby("A")
        # 对分组后的数据应用指数加权移动平均，使用指定的半衰期和时间点 times
        result = gb.ewm(halflife=halflife, times=times).mean()
        # 期望的结果，对分组后的数据应用指数加权移动平均，使用指定的半衰期和 times 数组的值
        expected = gb.ewm(halflife=halflife, times=times.values).mean()
        # 使用测试工具验证 result 和 expected 的数据帧是否相等
        tm.assert_frame_equal(result, expected)

    # 测试函数，验证切片操作后对象不可变性
    def test_dont_mutate_obj_after_slicing(self):
        # GH 43355
        # 创建包含"id"、"timestamp"、"y" 列的数据帧
        df = DataFrame(
            {
                "id": ["a", "a", "b", "b", "b"],
                "timestamp": date_range("2021-9-1", periods=5, freq="h"),
                "y": range(5),
            }
        )
        # 根据"id"列对数据帧进行分组，并对"timestamp"列使用1小时的滚动窗口
        grp = df.groupby("id").rolling("1h", on="timestamp")
        # 计算每个分组的滚动窗口内的数据数量
        result = grp.count()
        # 期望的结果数据帧，包含"timestamp"和"y"列，以及多级索引
        expected_df = DataFrame(
            {
                "timestamp": date_range("2021-9-1", periods=5, freq="h"),
                "y": [1.0] * 5,
            },
            index=MultiIndex.from_arrays(
                [["a", "a", "b", "b", "b"], list(range(5))], names=["id", None]
            ),
        )
        # 使用测试工具验证 result 和 expected_df 的数据帧是否相等
        tm.assert_frame_equal(result, expected_df)

        # 计算分组后的"y"列在滚动窗口内的数据数量
        result = grp["y"].count()
        # 期望的结果序列，包含"y"列的数据和多级索引
        expected_series = Series(
            [1.0] * 5,
            index=MultiIndex.from_arrays(
                [
                    ["a", "a", "b", "b", "b"],
                    date_range("2021-9-1", periods=5, freq="h"),
                ],
                names=["id", "timestamp"],
            ),
            name="y",
        )
        # 使用测试工具验证 result 和 expected_series 的序列是否相等
        tm.assert_series_equal(result, expected_series)
        # 这是关键的测试点，验证再次计算滚动窗口内的数据数量是否和前面的结果相等
        result = grp.count()
        # 使用测试工具验证 result 和 expected_df 的数据帧是否相等
        tm.assert_frame_equal(result, expected_df)
# 定义一个测试函数，用于验证在索引中只有单个整数的情况下的滚动相关性计算
def test_rolling_corr_with_single_integer_in_index():
    # GH 44078: 这个测试用例对应 GitHub 上的 issue 44078
    # 创建一个 DataFrame，其中包含一列元组 (1,) 和一列整数 [4, 5, 6]
    df = DataFrame({"a": [(1,), (1,), (1,)], "b": [4, 5, 6]})
    # 根据列 "a" 进行分组
    gb = df.groupby(["a"])
    # 计算滚动窗口为 2 的相关系数，other 参数为整个 DataFrame df
    result = gb.rolling(2).corr(other=df)
    # 创建一个多级索引，包含元组 ((1,), 0), ((1,), 1), ((1,), 2)，命名为 ["a", None]
    index = MultiIndex.from_tuples([((1,), 0), ((1,), 1), ((1,), 2)], names=["a", None])
    # 创建期望的 DataFrame，包含列 "a" 的 NaN 值和列 "b" 的相关系数值
    expected = DataFrame(
        {"a": [np.nan, np.nan, np.nan], "b": [np.nan, 1.0, 1.0]}, index=index
    )
    # 使用测试框架中的函数验证 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)


# 定义一个测试函数，用于验证在索引中包含元组的情况下的滚动相关性计算
def test_rolling_corr_with_tuples_in_index():
    # GH 44078: 这个测试用例对应 GitHub 上的 issue 44078
    # 创建一个 DataFrame，其中包含一列元组 (1, 2) 和一列整数 [4, 5, 6]
    df = DataFrame(
        {
            "a": [(1, 2), (1, 2), (1, 2)],
            "b": [4, 5, 6],
        }
    )
    # 根据列 "a" 进行分组
    gb = df.groupby(["a"])
    # 计算滚动窗口为 2 的相关系数，other 参数为整个 DataFrame df
    result = gb.rolling(2).corr(other=df)
    # 创建一个多级索引，包含元组 ((1, 2), 0), ((1, 2), 1), ((1, 2), 2)，命名为 ["a", None]
    index = MultiIndex.from_tuples(
        [((1, 2), 0), ((1, 2), 1), ((1, 2), 2)], names=["a", None]
    )
    # 创建期望的 DataFrame，包含列 "a" 的 NaN 值和列 "b" 的相关系数值
    expected = DataFrame(
        {"a": [np.nan, np.nan, np.nan], "b": [np.nan, 1.0, 1.0]}, index=index
    )
    # 使用测试框架中的函数验证 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)
```