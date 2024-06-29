# `D:\src\scipysrc\pandas\pandas\tests\generic\test_duplicate_labels.py`

```
"""Tests dealing with the NDFrame.allows_duplicates."""

import operator  # 导入 operator 模块

import numpy as np  # 导入 NumPy 库
import pytest  # 导入 pytest 测试框架

import pandas as pd  # 导入 Pandas 库
import pandas._testing as tm  # 导入 Pandas 测试模块

not_implemented = pytest.mark.xfail(reason="Not implemented.")  # 定义 pytest 的标记 xfail，表示未实现的测试

# ----------------------------------------------------------------------------
# Preservation

class TestPreserves:  # 定义测试类 TestPreserves

    @pytest.mark.parametrize(
        "cls, data",
        [
            (pd.Series, np.array([])),
            (pd.Series, [1, 2]),
            (pd.DataFrame, {}),
            (pd.DataFrame, {"A": [1, 2]}),
        ],
    )
    def test_construction_ok(self, cls, data):
        result = cls(data)  # 使用给定数据 data 构造 cls 对象
        assert result.flags.allows_duplicate_labels is True  # 断言对象的 allows_duplicate_labels 标志为 True

        result = cls(data).set_flags(allows_duplicate_labels=False)  # 使用 set_flags 方法设置 allows_duplicate_labels 标志为 False
        assert result.flags.allows_duplicate_labels is False  # 断言对象的 allows_duplicate_labels 标志为 False

    @pytest.mark.parametrize(
        "func",
        [
            operator.itemgetter(["a"]),  # 使用 operator.itemgetter 获取指定索引的元素
            operator.methodcaller("add", 1),  # 使用 operator.methodcaller 调用对象的 add 方法，并传递参数 1
            operator.methodcaller("rename", str.upper),  # 使用 operator.methodcaller 调用对象的 rename 方法，并传递函数 str.upper
            operator.methodcaller("rename", "name"),  # 使用 operator.methodcaller 调用对象的 rename 方法，并传递字符串参数 "name"
            operator.methodcaller("abs"),  # 使用 operator.methodcaller 调用对象的 abs 方法
            np.abs,  # 使用 NumPy 的 abs 函数
        ],
    )
    def test_preserved_series(self, func):
        s = pd.Series([0, 1], index=["a", "b"]).set_flags(allows_duplicate_labels=False)  # 创建 Series 对象并设置 allows_duplicate_labels 标志为 False
        assert func(s).flags.allows_duplicate_labels is False  # 断言调用 func 函数后的对象的 allows_duplicate_labels 标志为 False

    @pytest.mark.parametrize("index", [["a", "b", "c"], ["a", "b"]])
    # TODO: frame
    @not_implemented  # 使用 xfail 标记暂未实现的测试
    def test_align(self, index):
        other = pd.Series(0, index=index)  # 创建具有指定索引的 Series 对象
        s = pd.Series([0, 1], index=["a", "b"]).set_flags(allows_duplicate_labels=False)  # 创建 Series 对象并设置 allows_duplicate_labels 标志为 False
        a, b = s.align(other)  # 对象与其他对象对齐
        assert a.flags.allows_duplicate_labels is False  # 断言对齐后的结果对象的 allows_duplicate_labels 标志为 False
        assert b.flags.allows_duplicate_labels is False  # 断言对齐后的结果对象的 allows_duplicate_labels 标志为 False

    def test_preserved_frame(self):
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]}, index=["a", "b"]).set_flags(
            allows_duplicate_labels=False  # 创建 DataFrame 对象并设置 allows_duplicate_labels 标志为 False
        )
        assert df.loc[["a"]].flags.allows_duplicate_labels is False  # 断言 DataFrame 的 loc 子集的 allows_duplicate_labels 标志为 False
        assert df.loc[:, ["A", "B"]].flags.allows_duplicate_labels is False  # 断言 DataFrame 的 loc 子集的 allows_duplicate_labels 标志为 False

    def test_to_frame(self):
        ser = pd.Series(dtype=float).set_flags(allows_duplicate_labels=False)  # 创建 Series 对象并设置 allows_duplicate_labels 标志为 False
        assert ser.to_frame().flags.allows_duplicate_labels is False  # 断言转换为 DataFrame 后的对象的 allows_duplicate_labels 标志为 False

    @pytest.mark.parametrize("func", ["add", "sub"])
    @pytest.mark.parametrize("frame", [False, True])
    @pytest.mark.parametrize("other", [1, pd.Series([1, 2], name="A")])
    def test_binops(self, func, other, frame):
        df = pd.Series([1, 2], name="A", index=["a", "b"]).set_flags(
            allows_duplicate_labels=False  # 创建 Series 对象并设置 allows_duplicate_labels 标志为 False
        )
        if frame:
            df = df.to_frame()  # 如果 frame 为 True，转换为 DataFrame 对象
        if isinstance(other, pd.Series) and frame:
            other = other.to_frame()  # 如果 other 是 Series 对象且 frame 为 True，将 other 转换为 DataFrame 对象
        func = operator.methodcaller(func, other)  # 使用 operator.methodcaller 调用指定的函数 func，并传递 other 参数
        assert df.flags.allows_duplicate_labels is False  # 断言对象的 allows_duplicate_labels 标志为 False
        assert func(df).flags.allows_duplicate_labels is False  # 断言调用 func 后的结果对象的 allows_duplicate_labels 标志为 False
    # 定义测试方法 test_preserve_getitem，测试 DataFrame 对象的行为
    def test_preserve_getitem(self):
        # 创建一个包含一列数据的 DataFrame，并设置不允许重复标签的标志
        df = pd.DataFrame({"A": [1, 2]}).set_flags(allows_duplicate_labels=False)
        # 断言访问列索引的结果不允许重复标签
        assert df[["A"]].flags.allows_duplicate_labels is False
        # 断言访问列的结果不允许重复标签
        assert df["A"].flags.allows_duplicate_labels is False
        # 断言通过 loc 访问行的结果不允许重复标签
        assert df.loc[0].flags.allows_duplicate_labels is False
        # 断言通过 loc 访问单行 DataFrame 的结果不允许重复标签
        assert df.loc[[0]].flags.allows_duplicate_labels is False
        # 断言通过 loc 访问行和列的结果不允许重复标签
        assert df.loc[0, ["A"]].flags.allows_duplicate_labels is False

    @pytest.mark.parametrize(
        "objs, kwargs",
        [
            # Series
            (
                # 创建两个具有不同索引的 Series 对象
                [
                    pd.Series(1, index=["a", "b"]),
                    pd.Series(2, index=["c", "d"]),
                ],
                {},  # 不带额外参数
            ),
            (
                # 创建两个具有相同索引的 Series 对象，忽略索引重置参数
                [
                    pd.Series(1, index=["a", "b"]),
                    pd.Series(2, index=["a", "b"]),
                ],
                {"ignore_index": True},
            ),
            (
                # 创建两个具有相同索引的 Series 对象，指定轴向参数
                [
                    pd.Series(1, index=["a", "b"]),
                    pd.Series(2, index=["a", "b"]),
                ],
                {"axis": 1},
            ),
            # DataFrame
            (
                # 创建两个具有不同索引的 DataFrame 对象
                [
                    pd.DataFrame({"A": [1, 2]}, index=["a", "b"]),
                    pd.DataFrame({"A": [1, 2]}, index=["c", "d"]),
                ],
                {},  # 不带额外参数
            ),
            (
                # 创建两个具有相同索引的 DataFrame 对象，忽略索引重置参数
                [
                    pd.DataFrame({"A": [1, 2]}, index=["a", "b"]),
                    pd.DataFrame({"A": [1, 2]}, index=["a", "b"]),
                ],
                {"ignore_index": True},
            ),
            (
                # 创建两个具有不同列的 DataFrame 对象，指定轴向参数
                [
                    pd.DataFrame({"A": [1, 2]}, index=["a", "b"]),
                    pd.DataFrame({"B": [1, 2]}, index=["a", "b"]),
                ],
                {"axis": 1},
            ),
            # Series / DataFrame
            (
                # 创建一个包含 DataFrame 和 Series 的对象列表，指定轴向参数
                [
                    pd.DataFrame({"A": [1, 2]}, index=["a", "b"]),
                    pd.Series([1, 2], index=["a", "b"], name="B"),
                ],
                {"axis": 1},
            ),
        ],
    )
    # 定义测试方法 test_concat，测试 pandas.concat 方法的行为
    def test_concat(self, objs, kwargs):
        # 对输入的对象列表中的每个对象设置不允许重复标签的标志
        objs = [x.set_flags(allows_duplicate_labels=False) for x in objs]
        # 调用 pandas.concat 进行对象合并操作，使用传入的参数
        result = pd.concat(objs, **kwargs)
        # 断言合并结果的标志中不允许重复标签
        assert result.flags.allows_duplicate_labels is False
    @pytest.mark.parametrize(
        "left, right, expected",
        [  # 参数化测试数据集合
            # Case 1: False, False, False
            pytest.param(
                pd.DataFrame({"A": [0, 1]}, index=["a", "b"]).set_flags(
                    allows_duplicate_labels=False
                ),  # 创建左侧 DataFrame，禁止重复标签
                pd.DataFrame({"B": [0, 1]}, index=["a", "d"]).set_flags(
                    allows_duplicate_labels=False
                ),  # 创建右侧 DataFrame，禁止重复标签
                False,  # 期望结果为 False
                marks=not_implemented,  # 使用未实现的标记
            ),
            # Case 2: False, True, False
            pytest.param(
                pd.DataFrame({"A": [0, 1]}, index=["a", "b"]).set_flags(
                    allows_duplicate_labels=False
                ),  # 创建左侧 DataFrame，禁止重复标签
                pd.DataFrame({"B": [0, 1]}, index=["a", "d"]),  # 创建右侧 DataFrame
                False,  # 期望结果为 False
                marks=not_implemented,  # 使用未实现的标记
            ),
            # Case 3: True, True, True
            (
                pd.DataFrame({"A": [0, 1]}, index=["a", "b"]),  # 创建左侧 DataFrame
                pd.DataFrame({"B": [0, 1]}, index=["a", "d"]),  # 创建右侧 DataFrame
                True,  # 期望结果为 True
            ),
        ],
    )
    def test_merge(self, left, right, expected):
        result = pd.merge(left, right, left_index=True, right_index=True)  # 执行合并操作
        assert result.flags.allows_duplicate_labels is expected  # 断言合并后的标签是否符合预期

    @not_implemented
    def test_groupby(self):
        # XXX: This is under tested
        # TODO:
        #  - apply
        #  - transform
        #  - Should passing a grouper that disallows duplicates propagate?
        df = pd.DataFrame({"A": [1, 2, 3]}).set_flags(allows_duplicate_labels=False)  # 创建 DataFrame，禁止重复标签
        result = df.groupby([0, 0, 1]).agg("count")  # 执行分组和聚合操作
        assert result.flags.allows_duplicate_labels is False  # 断言结果标签是否符合预期

    @pytest.mark.parametrize("frame", [True, False])
    @not_implemented
    def test_window(self, frame):
        df = pd.Series(
            1,
            index=pd.date_range("2000", periods=12),
            name="A",
            allows_duplicate_labels=False,
        )  # 创建 Series，禁止重复标签
        if frame:
            df = df.to_frame()  # 如果 frame 为 True，则转换为 DataFrame
        assert df.rolling(3).mean().flags.allows_duplicate_labels is False  # 断言滚动均值后的标签是否符合预期
        assert df.ewm(3).mean().flags.allows_duplicate_labels is False  # 断言指数加权移动平均后的标签是否符合预期
        assert df.expanding(3).mean().flags.allows_duplicate_labels is False  # 断言扩展窗口均值后的标签是否符合预期
# ----------------------------------------------------------------------------
# Raises

class TestRaises:
    @pytest.mark.parametrize(
        "cls, axes",
        [
            # 测试 pd.Series，设置索引有重复的情况
            (pd.Series, {"index": ["a", "a"], "dtype": float}),
            # 测试 pd.DataFrame，设置索引有重复的情况
            (pd.DataFrame, {"index": ["a", "a"]}),
            # 测试 pd.DataFrame，设置索引和列名都有重复的情况
            (pd.DataFrame, {"index": ["a", "a"], "columns": ["b", "b"]}),
            # 测试 pd.DataFrame，设置列名有重复的情况
            (pd.DataFrame, {"columns": ["b", "b"]}),
        ],
    )
    def test_set_flags_with_duplicates(self, cls, axes):
        # 创建指定类型和参数的对象
        result = cls(**axes)
        # 断言对象允许重复标签
        assert result.flags.allows_duplicate_labels is True

        # 准备错误消息
        msg = "Index has duplicates."
        # 断言在设置不允许重复标签时会抛出 DuplicateLabelError 异常，并匹配指定消息
        with pytest.raises(pd.errors.DuplicateLabelError, match=msg):
            cls(**axes).set_flags(allows_duplicate_labels=False)

    @pytest.mark.parametrize(
        "data",
        [
            # 测试 pd.Series，设置索引有重复的情况
            pd.Series(index=[0, 0], dtype=float),
            # 测试 pd.DataFrame，设置索引有重复的情况
            pd.DataFrame(index=[0, 0]),
            # 测试 pd.DataFrame，设置列名有重复的情况
            pd.DataFrame(columns=[0, 0]),
        ],
    )
    def test_setting_allows_duplicate_labels_raises(self, data):
        # 准备错误消息
        msg = "Index has duplicates."
        # 断言在设置不允许重复标签时会抛出 DuplicateLabelError 异常，并匹配指定消息
        with pytest.raises(pd.errors.DuplicateLabelError, match=msg):
            data.flags.allows_duplicate_labels = False

        # 断言对象允许重复标签
        assert data.flags.allows_duplicate_labels is True

    def test_series_raises(self):
        # 创建具有重复索引的 pd.Series 对象
        a = pd.Series(0, index=["a", "b"])
        # 创建另一个具有重复索引的 pd.Series 对象，并设置不允许重复标签
        b = pd.Series([0, 1], index=["a", "b"]).set_flags(allows_duplicate_labels=False)
        # 准备错误消息
        msg = "Index has duplicates."
        # 断言在连接包含重复索引的两个 Series 时会抛出 DuplicateLabelError 异常，并匹配指定消息
        with pytest.raises(pd.errors.DuplicateLabelError, match=msg):
            pd.concat([a, b])

    @pytest.mark.parametrize(
        "getter, target",
        [
            # 测试通过列名获取元素时，列名有重复的情况
            (operator.itemgetter(["A", "A"]), None),
            # loc 操作，通过索引获取元素时，索引有重复的情况
            (operator.itemgetter(["a", "a"]), "loc"),
            # loc 操作，通过索引和列名同时获取元素时，索引和列名都有重复的情况
            pytest.param(operator.itemgetter(("a", ["A", "A"])), "loc"),
            # loc 操作，通过索引和列名同时获取元素时，索引和列名都有重复的情况
            (operator.itemgetter((["a", "a"], "A")), "loc"),
            # iloc 操作，通过索引位置获取元素时，索引位置有重复的情况
            (operator.itemgetter([0, 0]), "iloc"),
            # iloc 操作，通过索引位置和列名同时获取元素时，索引位置和列名都有重复的情况
            pytest.param(operator.itemgetter((0, [0, 0])), "iloc"),
            # iloc 操作，通过索引位置和列名同时获取元素时，索引位置和列名都有重复的情况
            pytest.param(operator.itemgetter(([0, 0], 0)), "iloc"),
        ],
    )
    def test_getitem_raises(self, getter, target):
        # 创建具有重复索引的 pd.DataFrame 对象，并设置不允许重复标签
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]}, index=["a", "b"]).set_flags(
            allows_duplicate_labels=False
        )
        if target:
            # 根据目标字符串选择获取器对象 df, df.loc, 或 df.iloc
            target = getattr(df, target)
        else:
            target = df

        # 准备错误消息
        msg = "Index has duplicates."
        # 断言在使用指定的获取器获取元素时会抛出 DuplicateLabelError 异常，并匹配指定消息
        with pytest.raises(pd.errors.DuplicateLabelError, match=msg):
            getter(target)

    def test_concat_raises(self):
        # 创建包含具有重复索引的 Series 对象列表，并设置不允许重复标签
        objs = [
            pd.Series(1, index=[0, 1], name="a"),
            pd.Series(2, index=[0, 1], name="a"),
        ]
        objs = [x.set_flags(allows_duplicate_labels=False) for x in objs]
        # 准备错误消息
        msg = "Index has duplicates."
        # 断言在连接具有重复索引的 Series 对象时会抛出 DuplicateLabelError 异常，并匹配指定消息
        with pytest.raises(pd.errors.DuplicateLabelError, match=msg):
            pd.concat(objs, axis=1)

    @not_implemented
    # 定义一个测试函数，测试合并操作是否会引发异常
    def test_merge_raises(self):
        # 创建 DataFrame a，包含一列数据"A"，索引为["a", "b", "c"]，并设置不允许重复标签
        a = pd.DataFrame({"A": [0, 1, 2]}, index=["a", "b", "c"]).set_flags(
            allows_duplicate_labels=False
        )
        # 创建 DataFrame b，包含一列数据"B"，索引为["a", "b", "b"]
        b = pd.DataFrame({"B": [0, 1, 2]}, index=["a", "b", "b"])
        # 定义错误信息
        msg = "Index has duplicates."
        # 使用 pytest 检查是否会引发 DuplicateLabelError 异常，并匹配错误信息
        with pytest.raises(pd.errors.DuplicateLabelError, match=msg):
            # 尝试将 DataFrame a 和 DataFrame b 按索引合并
            pd.merge(a, b, left_index=True, right_index=True)
# 使用 pytest 的 mark.parametrize 装饰器为 test_raises_basic 函数添加多个参数化测试用例
@pytest.mark.parametrize(
    "idx",
    [
        pd.Index([1, 1]),  # 创建一个整数类型的重复索引
        pd.Index(["a", "a"]),  # 创建一个字符串类型的重复索引
        pd.Index([1.1, 1.1]),  # 创建一个浮点数类型的重复索引
        pd.PeriodIndex([pd.Period("2000", "D")] * 2),  # 创建一个周期索引，每个周期是一天
        pd.DatetimeIndex([pd.Timestamp("2000")] * 2),  # 创建一个日期时间索引，都是2000年的日期
        pd.TimedeltaIndex([pd.Timedelta("1D")] * 2),  # 创建一个时间增量索引，每个增量是一天
        pd.CategoricalIndex(["a", "a"]),  # 创建一个分类类型的重复索引
        pd.IntervalIndex([pd.Interval(0, 1)] * 2),  # 创建一个区间索引，区间范围是 [0, 1]
        pd.MultiIndex.from_tuples([("a", 1), ("a", 1)]),  # 创建一个多级索引，包含重复的元组
    ],
    # 将测试用例的参数命名为其类型的名称（例如，pd.Index）
    ids=lambda x: type(x).__name__,
)
def test_raises_basic(idx):
    # 准备用于测试的错误消息
    msg = "Index has duplicates."
    
    # 测试在创建 Series 时是否会引发 DuplicateLabelError 异常
    with pytest.raises(pd.errors.DuplicateLabelError, match=msg):
        pd.Series(1, index=idx).set_flags(allows_duplicate_labels=False)

    # 测试在创建 DataFrame 时是否会引发 DuplicateLabelError 异常
    with pytest.raises(pd.errors.DuplicateLabelError, match=msg):
        pd.DataFrame({"A": [1, 1]}, index=idx).set_flags(allows_duplicate_labels=False)

    # 测试在创建具有指定列的 DataFrame 时是否会引发 ValueError 异常
    with pytest.raises(ValueError, match=msg):
        pd.DataFrame([[1, 2]], columns=idx).set_flags(allows_duplicate_labels=False)


# 测试 Index 对象的 _format_duplicate_message 方法
def test_format_duplicate_labels_message():
    idx = pd.Index(["a", "b", "a", "b", "c"])
    result = idx._format_duplicate_message()
    expected = pd.DataFrame(
        {"positions": [[0, 2], [1, 3]]}, index=pd.Index(["a", "b"], name="label")
    )
    tm.assert_frame_equal(result, expected)


# 测试 MultiIndex 对象的 _format_duplicate_message 方法
def test_format_duplicate_labels_message_multi():
    idx = pd.MultiIndex.from_product([["A"], ["a", "b", "a", "b", "c"]])
    result = idx._format_duplicate_message()
    expected = pd.DataFrame(
        {"positions": [[0, 2], [1, 3]]},
        index=pd.MultiIndex.from_product([["A"], ["a", "b"]]),
    )
    tm.assert_frame_equal(result, expected)


# 测试在向 DataFrame 插入数据时是否会引发 ValueError 异常
def test_dataframe_insert_raises():
    df = pd.DataFrame({"A": [1, 2]}).set_flags(allows_duplicate_labels=False)
    msg = "Cannot specify"
    with pytest.raises(ValueError, match=msg):
        df.insert(0, "A", [3, 4], allow_duplicates=True)


# 使用 pytest 的 mark.parametrize 装饰器为 test_inplace_raises 函数添加多个参数化测试用例
@pytest.mark.parametrize(
    "method, frame_only",
    [
        # 准备调用 set_index 方法，并设置 inplace=True 参数
        (operator.methodcaller("set_index", "A", inplace=True), True),
        # 准备调用 reset_index 方法，并设置 inplace=True 参数
        (operator.methodcaller("reset_index", inplace=True), True),
        # 准备调用 rename 方法，并传递 lambda 函数和 inplace=True 参数
        (operator.methodcaller("rename", lambda x: x, inplace=True), False),
    ],
)
def test_inplace_raises(method, frame_only):
    # 创建一个 DataFrame 对象，并设置 allows_duplicate_labels=False
    df = pd.DataFrame({"A": [0, 0], "B": [1, 2]}).set_flags(
        allows_duplicate_labels=False
    )
    # 创建一个 Series 对象，并设置 allows_duplicate_labels=False
    s = df["A"]
    s.flags.allows_duplicate_labels = False
    msg = "Cannot specify"

    # 测试在调用指定方法时是否会引发 ValueError 异常
    with pytest.raises(ValueError, match=msg):
        method(df)
    if not frame_only:
        with pytest.raises(ValueError, match=msg):
            method(s)


# 测试 Series 和 DataFrame 对象的 pickle 反序列化功能
def test_pickle():
    # 创建一个 Series 对象，并设置 allows_duplicate_labels=False
    a = pd.Series([1, 2]).set_flags(allows_duplicate_labels=False)
    # 测试 round_trip_pickle 方法对 Series 对象的序列化和反序列化是否保持一致性
    b = tm.round_trip_pickle(a)
    tm.assert_series_equal(a, b)

    # 创建一个 DataFrame 对象，并设置 allows_duplicate_labels=False
    a = pd.DataFrame({"A": []}).set_flags(allows_duplicate_labels=False)
    # 测试 round_trip_pickle 方法对 DataFrame 对象的序列化和反序列化是否保持一致性
    b = tm.round_trip_pickle(a)
    tm.assert_frame_equal(a, b)
```