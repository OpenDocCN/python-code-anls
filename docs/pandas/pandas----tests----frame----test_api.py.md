# `D:\src\scipysrc\pandas\pandas\tests\frame\test_api.py`

```
# 从 copy 模块中导入 deepcopy 函数
from copy import deepcopy
# 导入 inspect 模块，用于检查对象
import inspect
# 导入 pydoc 模块，用于生成文档
import pydoc

# 导入 numpy 库，并使用别名 np
import numpy as np
# 导入 pytest 库
import pytest

# 从 pandas._config 模块中导入 using_pyarrow_string_dtype
from pandas._config import using_pyarrow_string_dtype
# 从 pandas._config.config 模块中导入 option_context
from pandas._config.config import option_context

# 导入 pandas 库，并使用别名 pd
import pandas as pd
# 从 pandas 库中导入 DataFrame, Series, date_range, timedelta_range 函数
from pandas import (
    DataFrame,
    Series,
    date_range,
    timedelta_range,
)
# 导入 pandas._testing 模块，并使用别名 tm
import pandas._testing as tm

# 定义 TestDataFrameMisc 类
class TestDataFrameMisc:
    # 定义 test_getitem_pop_assign_name 方法，接受 float_frame 参数
    def test_getitem_pop_assign_name(self, float_frame):
        # 获取 float_frame 的 "A" 列，并赋值给 s
        s = float_frame["A"]
        # 断言 s 的名称为 "A"
        assert s.name == "A"

        # 从 float_frame 中弹出 "A" 列，并赋值给 s
        s = float_frame.pop("A")
        # 断言 s 的名称为 "A"
        assert s.name == "A"

        # 获取 float_frame 的 "B" 列，并赋值给 s
        s = float_frame.loc[:, "B"]
        # 断言 s 的名称为 "B"
        assert s.name == "B"

        # 对 s 进行切片，并赋值给 s2
        s2 = s.loc[:]
        # 断言 s2 的名称为 "B"
        assert s2.name == "B"

    # 定义 test_get_axis 方法，接受 float_frame 参数
    def test_get_axis(self, float_frame):
        # 将 float_frame 赋值给 f
        f = float_frame
        # 断言获取轴号 0 的结果为 0
        assert f._get_axis_number(0) == 0
        # 断言获取轴号 1 的结果为 1
        assert f._get_axis_number(1) == 1
        # 断言获取轴号为 "index" 的结果为 0
        assert f._get_axis_number("index") == 0
        # 断言获取轴号为 "rows" 的结果为 0
        assert f._get_axis_number("rows") == 0
        # 断言获取轴号为 "columns" 的结果为 1
        assert f._get_axis_number("columns") == 1

        # 断言获取轴名 0 的结果为 "index"
        assert f._get_axis_name(0) == "index"
        # 断言获取轴名 1 的结果为 "columns"
        assert f._get_axis_name(1) == "columns"
        # 断言获取轴名为 "index" 的结果为 "index"
        assert f._get_axis_name("index") == "index"
        # 断言获取轴名为 "rows" 的结果为 "index"
        assert f._get_axis_name("rows") == "index"
        # 断言获取轴名为 "columns" 的结果为 "columns"
        assert f._get_axis_name("columns") == "columns"

        # 断言获取轴号 0 对应的轴对象为 f.index
        assert f._get_axis(0) is f.index
        # 断言获取轴号 1 对应的轴对象为 f.columns
        assert f._get_axis(1) is f.columns

        # 使用 pytest 断言引发 ValueError 异常，且异常信息匹配 "No axis named"
        with pytest.raises(ValueError, match="No axis named"):
            f._get_axis_number(2)

        # 使用 pytest 断言引发 ValueError 异常，且异常信息匹配 "No axis.*foo"
        with pytest.raises(ValueError, match="No axis.*foo"):
            f._get_axis_name("foo")

        # 使用 pytest 断言引发 ValueError 异常，且异常信息匹配 "No axis.*None"
        with pytest.raises(ValueError, match="No axis.*None"):
            f._get_axis_name(None)

        # 使用 pytest 断言引发 ValueError 异常，且异常信息匹配 "No axis named"
        with pytest.raises(ValueError, match="No axis named"):
            f._get_axis_number(None)

    # 定义 test_column_contains_raises 方法，接受 float_frame 参数
    def test_column_contains_raises(self, float_frame):
        # 使用 pytest 断言引发 TypeError 异常，且异常信息匹配 "unhashable type: 'Index'"
        with pytest.raises(TypeError, match="unhashable type: 'Index'"):
            float_frame.columns in float_frame

    # 定义 test_tab_completion 方法
    def test_tab_completion(self):
        # 创建一个 DataFrame，其列是标识符，将其赋值给 df
        df = DataFrame([list("abcd"), list("efgh")], columns=list("ABCD"))
        # 遍历列名为 "ABCD" 的列表，断言每个键在 df 的 dir() 方法结果中
        for key in list("ABCD"):
            assert key in dir(df)
        # 断言 df.__getitem__("A") 返回的对象是 Series 类型的实例
        assert isinstance(df.__getitem__("A"), Series)

        # 创建一个 MultiIndex 的 DataFrame，将其赋值给 df
        df = DataFrame(
            [list("abcd"), list("efgh")],
            columns=pd.MultiIndex.from_tuples(list(zip("ABCD", "EFGH"))),
        )
        # 遍历列名为 "ABCD" 的列表，断言每个键在 df 的 dir() 方法结果中
        for key in list("ABCD"):
            assert key in dir(df)
        # 遍历列名为 "EFGH" 的列表，断言每个键不在 df 的 dir() 方法结果中
        for key in list("EFGH"):
            assert key not in dir(df)
        # 断言 df.__getitem__("A") 返回的对象是 DataFrame 类型的实例
        assert isinstance(df.__getitem__("A"), DataFrame)
    def test_display_max_dir_items(self):
        # 设置 display.max_dir_items 增加 __dir__ 中的列数
        columns = ["a" + str(i) for i in range(420)]
        values = [range(420), range(420)]
        df = DataFrame(values, columns=columns)

        # display.max_dir_items 的默认值为 100
        assert "a99" in dir(df)
        assert "a100" not in dir(df)

        # 使用 option_context 设置 display.max_dir_items 为 300
        with option_context("display.max_dir_items", 300):
            df = DataFrame(values, columns=columns)
            assert "a299" in dir(df)
            assert "a300" not in dir(df)

        # 使用 option_context 设置 display.max_dir_items 为 None
        with option_context("display.max_dir_items", None):
            df = DataFrame(values, columns=columns)
            assert "a419" in dir(df)

    def test_not_hashable(self):
        empty_frame = DataFrame()

        df = DataFrame([1])
        msg = "unhashable type: 'DataFrame'"
        
        # 测试 DataFrame 类型的对象不可哈希
        with pytest.raises(TypeError, match=msg):
            hash(df)
        with pytest.raises(TypeError, match=msg):
            hash(empty_frame)

    @pytest.mark.xfail(using_pyarrow_string_dtype(), reason="surrogates not allowed")
    def test_column_name_contains_unicode_surrogate(self):
        # GH 25509
        colname = "\ud83d"
        
        # 创建一个包含 Unicode 代理项的列名的 DataFrame 对象
        df = DataFrame({colname: []})
        
        # 检查是否不会崩溃
        assert colname not in dir(df)
        assert df.columns[0] == colname

    def test_new_empty_index(self):
        df1 = DataFrame(np.random.default_rng(2).standard_normal((0, 3)))
        df2 = DataFrame(np.random.default_rng(2).standard_normal((0, 3)))
        
        # 设置一个 DataFrame 对象的索引名为 "foo"
        df1.index.name = "foo"
        assert df2.index.name is None

    def test_get_agg_axis(self, float_frame):
        cols = float_frame._get_agg_axis(0)
        
        # 检索 DataFrame 对象的聚合轴 0
        assert cols is float_frame.columns

        idx = float_frame._get_agg_axis(1)
        
        # 检索 DataFrame 对象的聚合轴 1
        assert idx is float_frame.index

        msg = r"Axis must be 0 or 1 \(got 2\)"
        
        # 测试当传入不支持的轴时是否引发 ValueError
        with pytest.raises(ValueError, match=msg):
            float_frame._get_agg_axis(2)

    def test_empty(self, float_frame, float_string_frame):
        empty_frame = DataFrame()
        
        # 检查空的 DataFrame 对象
        assert empty_frame.empty

        assert not float_frame.empty
        assert not float_string_frame.empty

        # 边界情况
        df = DataFrame({"A": [1.0, 2.0, 3.0], "B": ["a", "b", "c"]}, index=np.arange(3))
        del df["A"]
        
        # 检查非空 DataFrame 对象
        assert not df.empty

    def test_len(self, float_frame):
        assert len(float_frame) == len(float_frame.index)

        # 单块边界情况
        arr = float_frame[["A", "B"]].values
        expected = float_frame.reindex(columns=["A", "B"]).values
        
        # 检查 DataFrame 对象的长度以及索引重新排序后的数据
        tm.assert_almost_equal(arr, expected)

    def test_axis_aliases(self, float_frame):
        f = float_frame

        # 常规名称
        expected = f.sum(axis=0)
        result = f.sum(axis="index")
        
        # 检查轴别名 "index" 的聚合操作结果
        tm.assert_series_equal(result, expected)

        expected = f.sum(axis=1)
        result = f.sum(axis="columns")
        
        # 检查轴别名 "columns" 的聚合操作结果
        tm.assert_series_equal(result, expected)
    # 测试类方法 `test_class_axis`
    def test_class_axis(self):
        # GH 18147，测试用例编号
        # 检查 `DataFrame.index` 的文档字符串非空
        assert pydoc.getdoc(DataFrame.index)
        # 检查 `DataFrame.columns` 的文档字符串非空
        assert pydoc.getdoc(DataFrame.columns)

    # 测试方法 `test_series_put_names`
    def test_series_put_names(self, float_string_frame):
        # 从 `float_string_frame` 获取 Series 对象
        series = float_string_frame._series
        # 遍历 Series 中的键值对
        for k, v in series.items():
            # 断言每个 Series 中的值的名称等于其键
            assert v.name == k

    # 测试方法 `test_empty_nonzero`
    def test_empty_nonzero(self):
        # 创建一个包含 [1, 2, 3] 的 DataFrame 对象
        df = DataFrame([1, 2, 3])
        # 断言 DataFrame 不为空
        assert not df.empty
        # 创建一个只有索引和列的 DataFrame 对象
        df = DataFrame(index=[1], columns=[1])
        # 断言 DataFrame 不为空
        assert not df.empty
        # 创建一个带有索引 ["a", "b"] 和列 ["c", "d"] 的 DataFrame 对象，然后删除 NaN 值
        df = DataFrame(index=["a", "b"], columns=["c", "d"]).dropna()
        # 断言 DataFrame 为空
        assert df.empty
        # 断言 DataFrame 转置后为空
        assert df.T.empty

    # 使用 pytest.mark.parametrize 装饰的测试方法 `test_empty_like`
    @pytest.mark.parametrize(
        "df",
        [
            DataFrame(),  # 创建一个空的 DataFrame 对象
            DataFrame(index=[1]),  # 创建一个带有索引 [1] 的 DataFrame 对象
            DataFrame(columns=[1]),  # 创建一个带有列 [1] 的 DataFrame 对象
            DataFrame({1: []}),  # 创建一个列名为 1 且数据为空列表的 DataFrame 对象
        ],
    )
    def test_empty_like(self, df):
        # 断言 DataFrame 为空
        assert df.empty
        # 断言 DataFrame 转置后为空
        assert df.T.empty

    # 测试方法 `test_with_datetimelikes`
    def test_with_datetimelikes(self):
        # 创建一个包含日期范围和时间间隔范围的 DataFrame 对象
        df = DataFrame(
            {
                "A": date_range("20130101", periods=10),
                "B": timedelta_range("1 day", periods=10),
            }
        )
        # 对 DataFrame 进行转置
        t = df.T

        # 计算转置后每列的数据类型并统计它们的数量
        result = t.dtypes.value_counts()
        # 期望的结果是一个包含对象类型数据计数的 Series 对象
        expected = Series({np.dtype("object"): 10}, name="count")
        # 断言结果与期望相等
        tm.assert_series_equal(result, expected)

    # 测试方法 `test_deepcopy`
    def test_deepcopy(self, float_frame):
        # 对传入的 `float_frame` 对象进行深拷贝
        cp = deepcopy(float_frame)
        # 修改拷贝后的 DataFrame 中第一行 "A" 列的值为 10
        cp.loc[0, "A"] = 10
        # 断言原始的 `float_frame` 与拷贝后的 `cp` 不相等
        assert not float_frame.equals(cp)
    def test_inplace_return_self(self):
        # GH 1893

        data = DataFrame(
            {"a": ["foo", "bar", "baz", "qux"], "b": [0, 0, 1, 1], "c": [1, 2, 3, 4]}
        )

        def _check_f(base, f):
            # 调用函数 f，并检查返回结果是否为 None
            result = f(base)
            assert result is None

        # -----DataFrame-----

        # set_index
        # 定义 lambda 函数 f，用于在 DataFrame 上调用 set_index 方法，inplace=True 表示原地操作
        f = lambda x: x.set_index("a", inplace=True)
        _check_f(data.copy(), f)

        # reset_index
        # 定义 lambda 函数 f，用于在 DataFrame 上调用 reset_index 方法，inplace=True 表示原地操作
        f = lambda x: x.reset_index(inplace=True)
        _check_f(data.set_index("a"), f)

        # drop_duplicates
        # 定义 lambda 函数 f，用于在 DataFrame 上调用 drop_duplicates 方法，inplace=True 表示原地操作
        f = lambda x: x.drop_duplicates(inplace=True)
        _check_f(data.copy(), f)

        # sort
        # 定义 lambda 函数 f，用于在 DataFrame 上调用 sort_values 方法，inplace=True 表示原地操作
        f = lambda x: x.sort_values("b", inplace=True)
        _check_f(data.copy(), f)

        # sort_index
        # 定义 lambda 函数 f，用于在 DataFrame 上调用 sort_index 方法，inplace=True 表示原地操作
        f = lambda x: x.sort_index(inplace=True)
        _check_f(data.copy(), f)

        # fillna
        # 定义 lambda 函数 f，用于在 DataFrame 上调用 fillna 方法，inplace=True 表示原地操作
        f = lambda x: x.fillna(0, inplace=True)
        _check_f(data.copy(), f)

        # replace
        # 定义 lambda 函数 f，用于在 DataFrame 上调用 replace 方法，inplace=True 表示原地操作
        f = lambda x: x.replace(1, 0, inplace=True)
        _check_f(data.copy(), f)

        # rename
        # 定义 lambda 函数 f，用于在 DataFrame 上调用 rename 方法，inplace=True 表示原地操作
        f = lambda x: x.rename({1: "foo"}, inplace=True)
        _check_f(data.copy(), f)

        # -----Series-----
        d = data.copy()["c"]

        # reset_index
        # 定义 lambda 函数 f，用于在 Series 上调用 reset_index 方法，inplace=True 表示原地操作，drop=True 表示丢弃原索引
        f = lambda x: x.reset_index(inplace=True, drop=True)
        _check_f(data.set_index("a")["c"], f)

        # fillna
        # 定义 lambda 函数 f，用于在 Series 上调用 fillna 方法，inplace=True 表示原地操作
        f = lambda x: x.fillna(0, inplace=True)
        _check_f(d.copy(), f)

        # replace
        # 定义 lambda 函数 f，用于在 Series 上调用 replace 方法，inplace=True 表示原地操作
        f = lambda x: x.replace(1, 0, inplace=True)
        _check_f(d.copy(), f)

        # rename
        # 定义 lambda 函数 f，用于在 Series 上调用 rename 方法，inplace=True 表示原地操作
        f = lambda x: x.rename({1: "foo"}, inplace=True)
        _check_f(d.copy(), f)
    def test_set_flags(
        self,
        allows_duplicate_labels,
        frame_or_series,
    ):
        # 创建一个 DataFrame 对象，包含一列名为 "A" 的数据 [1, 2]
        obj = DataFrame({"A": [1, 2]})
        # 初始化 key 为元组 (0, 0)
        key = (0, 0)
        # 如果 frame_or_series 是 Series 类型，则将 obj 赋值为 "A" 列的 Series
        if frame_or_series is Series:
            obj = obj["A"]
            key = 0

        # 调用 set_flags 方法设置对象的标志，并保存结果
        result = obj.set_flags(allows_duplicate_labels=allows_duplicate_labels)

        # 如果 allows_duplicate_labels 为 None，则断言结果的 allows_duplicate_labels 属性为 True
        if allows_duplicate_labels is None:
            # 当未提供 allows_duplicate_labels 时，不更新属性
            assert result.flags.allows_duplicate_labels is True
        else:
            # 否则断言结果的 allows_duplicate_labels 属性与参数 allows_duplicate_labels 相同
            assert result.flags.allows_duplicate_labels is allows_duplicate_labels

        # 断言 obj 和 result 不是同一个对象（即进行了复制操作）
        assert obj is not result

        # 断言 obj 的 allows_duplicate_labels 属性为 True（未发生变异）
        assert obj.flags.allows_duplicate_labels is True

        # 但是数据未被复制
        if frame_or_series is Series:
            # 如果 frame_or_series 是 Series，则断言 obj.values 和 result.values 可能共享内存
            assert np.may_share_memory(obj.values, result.values)
        else:
            # 否则断言 obj["A"].values 和 result["A"].values 可能共享内存
            assert np.may_share_memory(obj["A"].values, result["A"].values)

        # 修改 result 的指定索引位置的值为 0，并断言 obj 相同位置的值仍为 1
        result.iloc[key] = 0
        assert obj.iloc[key] == 1

        # 现在进行复制操作
        result = obj.set_flags(allows_duplicate_labels=allows_duplicate_labels)
        # 修改 result 的指定索引位置的值为 10，并断言 obj 相同位置的值仍为 1
        result.iloc[key] = 10
        assert obj.iloc[key] == 1
```