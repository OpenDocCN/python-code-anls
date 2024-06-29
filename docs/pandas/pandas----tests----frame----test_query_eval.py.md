# `D:\src\scipysrc\pandas\pandas\tests\frame\test_query_eval.py`

```
import operator  # 导入operator模块，用于操作符的函数形式

import numpy as np  # 导入numpy库，并简写为np
import pytest  # 导入pytest库

from pandas.errors import (  # 从pandas.errors中导入以下错误类
    NumExprClobberingError,  # 当NumExpr被覆盖时的错误
    UndefinedVariableError,  # 未定义变量时的错误
)
import pandas.util._test_decorators as td  # 导入pandas.util._test_decorators模块，并简写为td

import pandas as pd  # 导入pandas库，并简写为pd
from pandas import (  # 从pandas库中导入以下对象
    DataFrame,  # 数据框对象
    Index,  # 索引对象
    MultiIndex,  # 多重索引对象
    Series,  # 系列对象
    date_range,  # 日期范围函数
)
import pandas._testing as tm  # 导入pandas._testing模块，并简写为tm
from pandas.core.computation.check import NUMEXPR_INSTALLED  # 从pandas.core.computation.check中导入NUMEXPR_INSTALLED常量

@pytest.fixture(params=["python", "pandas"], ids=lambda x: x)
def parser(request):
    return request.param  # 根据参数请求返回参数值

@pytest.fixture(
    params=["python", pytest.param("numexpr", marks=td.skip_if_no("numexpr"))],
    ids=lambda x: x,
)
def engine(request):
    return request.param  # 根据参数请求返回参数值

def skip_if_no_pandas_parser(parser):
    if parser != "pandas":
        pytest.skip(f"cannot evaluate with parser={parser}")  # 如果parser不是'pandas'，则跳过测试

class TestCompat:
    @pytest.fixture
    def df(self):
        return DataFrame({"A": [1, 2, 3]})  # 创建一个DataFrame对象

    @pytest.fixture
    def expected1(self, df):
        return df[df.A > 0]  # 返回符合条件的DataFrame子集

    @pytest.fixture
    def expected2(self, df):
        return df.A + 1  # 返回Series对象，包含每个元素加1后的结果

    def test_query_default(self, df, expected1, expected2):
        # GH 12749
        # this should always work, whether NUMEXPR_INSTALLED or not
        result = df.query("A>0")  # 使用query方法过滤DataFrame
        tm.assert_frame_equal(result, expected1)  # 断言结果与预期DataFrame相等
        result = df.eval("A+1")  # 使用eval方法计算表达式
        tm.assert_series_equal(result, expected2)  # 断言结果与预期Series相等

    def test_query_None(self, df, expected1, expected2):
        result = df.query("A>0", engine=None)  # 使用python引擎过滤DataFrame
        tm.assert_frame_equal(result, expected1)  # 断言结果与预期DataFrame相等
        result = df.eval("A+1", engine=None)  # 使用python引擎计算表达式
        tm.assert_series_equal(result, expected2)  # 断言结果与预期Series相等

    def test_query_python(self, df, expected1, expected2):
        result = df.query("A>0", engine="python")  # 使用python引擎过滤DataFrame
        tm.assert_frame_equal(result, expected1)  # 断言结果与预期DataFrame相等
        result = df.eval("A+1", engine="python")  # 使用python引擎计算表达式
        tm.assert_series_equal(result, expected2)  # 断言结果与预期Series相等

    def test_query_numexpr(self, df, expected1, expected2):
        if NUMEXPR_INSTALLED:
            result = df.query("A>0", engine="numexpr")  # 使用numexpr引擎过滤DataFrame
            tm.assert_frame_equal(result, expected1)  # 断言结果与预期DataFrame相等
            result = df.eval("A+1", engine="numexpr")  # 使用numexpr引擎计算表达式
            tm.assert_series_equal(result, expected2)  # 断言结果与预期Series相等
        else:
            msg = (
                r"'numexpr' is not installed or an unsupported version. "
                r"Cannot use engine='numexpr' for query/eval if 'numexpr' is "
                r"not installed"
            )
            with pytest.raises(ImportError, match=msg):
                df.query("A>0", engine="numexpr")  # 断言引发ImportError异常
            with pytest.raises(ImportError, match=msg):
                df.eval("A+1", engine="numexpr")  # 断言引发ImportError异常

class TestDataFrameEval:
    # smaller hits python, larger hits numexpr
    @pytest.mark.parametrize("n", [4, 4000])  # 参数化测试n为4和4000
    @pytest.mark.parametrize(
        "op_str,op,rop",
        [
            ("+", "__add__", "__radd__"),  # 加法运算符及其对应的特殊方法
            ("-", "__sub__", "__rsub__"),  # 减法运算符及其对应的特殊方法
            ("*", "__mul__", "__rmul__"),  # 乘法运算符及其对应的特殊方法
            ("/", "__truediv__", "__rtruediv__"),  # 除法运算符及其对应的特殊方法
        ],
    )
    def test_ops(self, op_str, op, rop, n):
        # 测试运算符和反向运算符在评估中的应用
        # GH7198

        # 创建一个具有指定索引和列的 DataFrame，所有值初始化为1
        df = DataFrame(1, index=range(n), columns=list("abcd"))
        
        # 将第一行数据设置为2
        df.iloc[0] = 2
        
        # 计算 DataFrame 的均值
        m = df.mean()

        # 创建一个新的 DataFrame，使用 m 的值重复 n 次，并按指定列名设置列
        base = DataFrame(
            np.tile(m.values, n).reshape(n, -1), columns=list("abcd")
        )

        # 通过字符串动态评估表达式，生成期望结果
        expected = eval(f"base {op_str} df")

        # 使用字符串运算符进行运算，并验证结果是否与期望一致
        result = eval(f"m {op_str} df")
        tm.assert_frame_equal(result, expected)

        # 对于可交换的运算符，如加法和乘法
        if op in ["+", "*"]:
            # 使用 getattr 调用对应的运算符方法，并验证结果是否与期望一致
            result = getattr(df, op)(m)
            tm.assert_frame_equal(result, expected)

        # 对于不可交换的运算符，如减法和除法
        elif op in ["-", "/"]:
            # 使用 getattr 调用对应的反向运算符方法，并验证结果是否与期望一致
            result = getattr(df, rop)(m)
            tm.assert_frame_equal(result, expected)

    def test_dataframe_sub_numexpr_path(self):
        # GH7192: 需要大量的行来确保使用 numexpr 路径
        df = DataFrame({"A": np.random.default_rng(2).standard_normal(25000)})
        
        # 将前5行设置为 NaN
        df.iloc[0:5] = np.nan
        
        # 生成预期结果，检查前25行中的非 NaN 值
        expected = 1 - np.isnan(df.iloc[0:25])
        
        # 计算整个 DataFrame 中非 NaN 值，并检查前25行结果是否与预期一致
        result = (1 - np.isnan(df)).iloc[0:25]
        tm.assert_frame_equal(result, expected)

    def test_query_non_str(self):
        # GH 11485
        df = DataFrame({"A": [1, 2, 3], "B": ["a", "b", "b"]})

        # 准备错误消息，确保表达式必须是字符串以便被评估
        msg = "expr must be a string to be evaluated"
        
        # 使用 pytest 检查 lambda 表达式作为参数时是否抛出 ValueError 异常
        with pytest.raises(ValueError, match=msg):
            df.query(lambda x: x.B == "b")

        # 使用 pytest 检查整数作为参数时是否抛出 ValueError 异常
        with pytest.raises(ValueError, match=msg):
            df.query(111)

    def test_query_empty_string(self):
        # GH 13139
        df = DataFrame({"A": [1, 2, 3]})

        # 准备错误消息，确保表达式不能为空字符串
        msg = "expr cannot be an empty string"
        
        # 使用 pytest 检查空字符串作为参数时是否抛出 ValueError 异常
        with pytest.raises(ValueError, match=msg):
            df.query("")

    def test_eval_resolvers_as_list(self):
        # GH 14095
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 2)), columns=list("ab")
        )
        dict1 = {"a": 1}
        dict2 = {"b": 2}
        
        # 使用 resolvers 列表作为参数进行列计算，并验证结果是否与预期一致
        assert df.eval("a + b", resolvers=[dict1, dict2]) == dict1["a"] + dict2["b"]
        
        # 使用 pd.eval 函数进行同样的列计算，并验证结果是否与预期一致
        assert pd.eval("a + b", resolvers=[dict1, dict2]) == dict1["a"] + dict2["b"]

    def test_eval_resolvers_combined(self):
        # GH 34966
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 2)), columns=list("ab")
        )
        dict1 = {"c": 2}

        # 同时使用输入和默认的索引/列解析器进行计算
        result = df.eval("a + b * c", resolvers=[dict1])

        # 生成预期结果，进行列计算，并验证结果是否与预期一致
        expected = df["a"] + df["b"] * dict1["c"]
        tm.assert_series_equal(result, expected)

    def test_eval_object_dtype_binop(self):
        # GH#24883
        df = DataFrame({"a1": ["Y", "N"]})
        
        # 使用 eval 方法计算布尔表达式并创建新列，验证结果是否与期望一致
        res = df.eval("c = ((a1 == 'Y') & True)")
        
        # 生成预期结果的 DataFrame，并验证结果是否与预期一致
        expected = DataFrame({"a1": ["Y", "N"], "c": [True, False]})
        tm.assert_frame_equal(res, expected)
    # 测试使用 NumPy 的 eval 函数
    def test_using_numpy(self, engine, parser):
        # 检查是否有 Pandas 解析器，如果没有则跳过测试
        skip_if_no_pandas_parser(parser)
        # 创建一个包含单个列的 Series，并转换为 DataFrame
        df = Series([0.2, 1.5, 2.8], name="a").to_frame()
        # 使用 eval 函数计算表达式 "@np.floor(a)"，返回计算结果
        res = df.eval("@np.floor(a)", engine=engine, parser=parser)
        # 使用 NumPy 的 floor 函数计算 DataFrame 列 "a" 的各元素的下限值
        expected = np.floor(df["a"])
        # 检查计算结果与预期结果是否相等
        tm.assert_series_equal(expected, res)

    # 测试简单的 eval 函数使用
    def test_eval_simple(self, engine, parser):
        # 创建一个包含单个列的 Series，并转换为 DataFrame
        df = Series([0.2, 1.5, 2.8], name="a").to_frame()
        # 使用 eval 函数计算表达式 "a"，返回计算结果
        res = df.eval("a", engine=engine, parser=parser)
        # 从 DataFrame 中获取列 "a" 作为预期结果
        expected = df["a"]
        # 检查计算结果与预期结果是否相等
        tm.assert_series_equal(expected, res)

    # 测试扩展数组的 eval 函数使用
    def test_extension_array_eval(self, engine, parser):
        # 创建一个包含扩展数组的 DataFrame
        df = DataFrame({"a": pd.array([1, 2, 3]), "b": pd.array([4, 5, 6])})
        # 使用 eval 函数计算表达式 "a / b"，返回计算结果
        result = df.eval("a / b", engine=engine, parser=parser)
        # 创建一个预期的 Series，包含计算结果的预期值
        expected = Series([0.25, 0.40, 0.50])
        # 检查计算结果与预期结果是否相等
        tm.assert_series_equal(result, expected)
class TestDataFrameQueryWithMultiIndex:
    # 定义测试类，用于测试带有命名多级索引的数据框查询功能
    def test_query_with_named_multiindex(self, parser, engine):
        # 如果没有安装 Pandas 解析器，跳过测试
        skip_if_no_pandas_parser(parser)
        
        # 创建随机选择的颜色和食物数组
        a = np.random.default_rng(2).choice(["red", "green"], size=10)
        b = np.random.default_rng(2).choice(["eggs", "ham"], size=10)
        
        # 使用颜色和食物数组创建命名的多级索引
        index = MultiIndex.from_arrays([a, b], names=["color", "food"])
        
        # 创建一个随机标准正态分布的数据框，使用上述多级索引作为索引
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 2)), index=index)
        
        # 创建一个系列，其值为数据框索引中颜色级别的数组，索引与数据框相同，命名为"color"
        ind = Series(
            df.index.get_level_values("color").values, index=index, name="color"
        )

        # 测试相等性条件
        res1 = df.query('color == "red"', parser=parser, engine=engine)
        res2 = df.query('"red" == color', parser=parser, engine=engine)
        exp = df[ind == "red"]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

        # 测试不等性条件
        res1 = df.query('color != "red"', parser=parser, engine=engine)
        res2 = df.query('"red" != color', parser=parser, engine=engine)
        exp = df[ind != "red"]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

        # 测试列表相等性条件（实际上是集合成员资格）
        res1 = df.query('color == ["red"]', parser=parser, engine=engine)
        res2 = df.query('["red"] == color', parser=parser, engine=engine)
        exp = df[ind.isin(["red"])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

        # 测试列表不相等性条件
        res1 = df.query('color != ["red"]', parser=parser, engine=engine)
        res2 = df.query('["red"] != color', parser=parser, engine=engine)
        exp = df[~ind.isin(["red"])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

        # 测试 in 和 not in 操作
        res1 = df.query('["red"] in color', parser=parser, engine=engine)
        res2 = df.query('"red" in color', parser=parser, engine=engine)
        exp = df[ind.isin(["red"])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)

        res1 = df.query('["red"] not in color', parser=parser, engine=engine)
        res2 = df.query('"red" not in color', parser=parser, engine=engine)
        exp = df[~ind.isin(["red"])]
        tm.assert_frame_equal(res1, exp)
        tm.assert_frame_equal(res2, exp)
    # 测试在部分命名的多级索引上执行查询功能
    def test_query_with_partially_named_multiindex(self, parser, engine):
        # 如果没有 Pandas 解析器，则跳过测试
        skip_if_no_pandas_parser(parser)
        
        # 生成随机的颜色和序号数据
        a = np.random.default_rng(2).choice(["red", "green"], size=10)
        b = np.arange(10)
        
        # 创建多级索引对象，其中第一级无名称，第二级命名为 "rating"
        index = MultiIndex.from_arrays([a, b])
        index.names = [None, "rating"]
        
        # 创建一个 DataFrame，其中包含随机正态分布的数据，使用前面创建的索引
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 2)), index=index)
        
        # 执行基于查询字符串 "rating == 1" 的查询操作，使用给定的解析器和引擎
        res = df.query("rating == 1", parser=parser, engine=engine)
        
        # 提取 "rating" 级别的数据作为 Series 对象
        ind = Series(
            df.index.get_level_values("rating").values, index=index, name="rating"
        )
        
        # 期望的结果是根据条件选取的 DataFrame 切片
        exp = df[ind == 1]
        
        # 断言查询结果与期望结果相等
        tm.assert_frame_equal(res, exp)

        # 执行基于查询字符串 "rating != 1" 的查询操作，使用给定的解析器和引擎
        res = df.query("rating != 1", parser=parser, engine=engine)
        
        # 再次提取 "rating" 级别的数据作为 Series 对象
        ind = Series(
            df.index.get_level_values("rating").values, index=index, name="rating"
        )
        
        # 期望的结果是根据条件选取的 DataFrame 切片
        exp = df[ind != 1]
        
        # 断言查询结果与期望结果相等
        tm.assert_frame_equal(res, exp)

        # 执行基于查询字符串 'ilevel_0 == "red"' 的查询操作，使用给定的解析器和引擎
        res = df.query('ilevel_0 == "red"', parser=parser, engine=engine)
        
        # 提取第一级别索引的数据作为 Series 对象
        ind = Series(df.index.get_level_values(0).values, index=index)
        
        # 期望的结果是根据条件选取的 DataFrame 切片
        exp = df[ind == "red"]
        
        # 断言查询结果与期望结果相等
        tm.assert_frame_equal(res, exp)

        # 执行基于查询字符串 'ilevel_0 != "red"' 的查询操作，使用给定的解析器和引擎
        res = df.query('ilevel_0 != "red"', parser=parser, engine=engine)
        
        # 再次提取第一级别索引的数据作为 Series 对象
        ind = Series(df.index.get_level_values(0).values, index=index)
        
        # 期望的结果是根据条件选取的 DataFrame 切片
        exp = df[ind != "red"]
        
        # 断言查询结果与期望结果相等
        tm.assert_frame_equal(res, exp)

    # 测试获取多级索引的索引解析器
    def test_query_multiindex_get_index_resolvers(self):
        # 创建一个包含全为1的 DataFrame，其索引为多级索引，命名为 "spam" 和 "eggs"
        df = DataFrame(
            np.ones((10, 3)),
            index=MultiIndex.from_arrays(
                [range(10) for _ in range(2)], names=["spam", "eggs"]
            ),
        )
        
        # 获取 DataFrame 的索引解析器
        resolvers = df._get_index_resolvers()

        # 定义一个函数，将指定级别的多级索引转换为 Series 对象
        def to_series(mi, level):
            level_values = mi.get_level_values(level)
            s = level_values.to_series()
            s.index = mi
            return s

        # 将 DataFrame 的列转换为 Series 对象
        col_series = df.columns.to_series()
        
        # 期望的索引解析结果，包括 "index"、"columns" 和每个索引级别的 Series 对象
        expected = {
            "index": df.index,
            "columns": col_series,
            "spam": to_series(df.index, "spam"),
            "eggs": to_series(df.index, "eggs"),
            "clevel_0": col_series,  # clevel_0 是 DataFrame 列的 Series 对象
        }
        
        # 遍历每个解析器结果，断言类型为 Series 或 Index，并与期望结果比较
        for k, v in resolvers.items():
            if isinstance(v, Index):
                assert v.is_(expected[k])
            elif isinstance(v, Series):
                tm.assert_series_equal(v, expected[k])
            else:
                raise AssertionError("object must be a Series or Index")
# 使用 pytest 的装饰器跳过，如果没有安装名为 "numexpr" 的库
@td.skip_if_no("numexpr")
class TestDataFrameQueryNumExprPandas:
    # 返回测试使用的引擎名称 "numexpr"
    @pytest.fixture
    def engine(self):
        return "numexpr"

    # 返回测试使用的解析器名称 "pandas"
    @pytest.fixture
    def parser(self):
        return "pandas"

    # 测试通过属性访问日期查询功能
    def test_date_query_with_attribute_access(self, engine, parser):
        # 如果解析器不是 pandas，则跳过测试
        skip_if_no_pandas_parser(parser)
        # 创建一个包含随机数据的 DataFrame
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
        # 添加名为 "dates1", "dates2", "dates3" 的日期列
        df["dates1"] = date_range("1/1/2012", periods=5)
        df["dates2"] = date_range("1/1/2013", periods=5)
        df["dates3"] = date_range("1/1/2014", periods=5)
        # 执行日期查询并返回结果
        res = df.query(
            "@df.dates1 < 20130101 < @df.dates3", engine=engine, parser=parser
        )
        # 预期的查询结果
        expec = df[(df.dates1 < "20130101") & ("20130101" < df.dates3)]
        # 断言结果是否相等
        tm.assert_frame_equal(res, expec)

    # 测试不通过属性访问日期查询功能
    def test_date_query_no_attribute_access(self, engine, parser):
        # 创建一个包含随机数据的 DataFrame
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
        # 添加名为 "dates1", "dates2", "dates3" 的日期列
        df["dates1"] = date_range("1/1/2012", periods=5)
        df["dates2"] = date_range("1/1/2013", periods=5)
        df["dates3"] = date_range("1/1/2014", periods=5)
        # 执行日期查询并返回结果
        res = df.query("dates1 < 20130101 < dates3", engine=engine, parser=parser)
        # 预期的查询结果
        expec = df[(df.dates1 < "20130101") & ("20130101" < df.dates3)]
        # 断言结果是否相等
        tm.assert_frame_equal(res, expec)

    # 测试包含 NaT 的日期查询功能
    def test_date_query_with_NaT(self, engine, parser):
        n = 10
        # 创建一个包含随机数据和 NaT 值的 DataFrame
        df = DataFrame(np.random.default_rng(2).standard_normal((n, 3)))
        df["dates1"] = date_range("1/1/2012", periods=n)
        df["dates2"] = date_range("1/1/2013", periods=n)
        df["dates3"] = date_range("1/1/2014", periods=n)
        df.loc[np.random.default_rng(2).random(n) > 0.5, "dates1"] = pd.NaT
        df.loc[np.random.default_rng(2).random(n) > 0.5, "dates3"] = pd.NaT
        # 执行日期查询并返回结果
        res = df.query("dates1 < 20130101 < dates3", engine=engine, parser=parser)
        # 预期的查询结果
        expec = df[(df.dates1 < "20130101") & ("20130101" < df.dates3)]
        # 断言结果是否相等
        tm.assert_frame_equal(res, expec)

    # 测试日期索引查询功能
    def test_date_index_query(self, engine, parser):
        n = 10
        # 创建一个包含随机数据的 DataFrame
        df = DataFrame(np.random.default_rng(2).standard_normal((n, 3)))
        df["dates1"] = date_range("1/1/2012", periods=n)
        df["dates3"] = date_range("1/1/2014", periods=n)
        # 将 "dates1" 列设置为索引，并就地修改 DataFrame
        return_value = df.set_index("dates1", inplace=True, drop=True)
        # 断言设置索引的返回值为 None
        assert return_value is None
        # 执行日期索引查询并返回结果
        res = df.query("index < 20130101 < dates3", engine=engine, parser=parser)
        # 预期的查询结果
        expec = df[(df.index < "20130101") & ("20130101" < df.dates3)]
        # 断言结果是否相等
        tm.assert_frame_equal(res, expec)
    # 定义测试函数，用于测试带有 pd.NaT 的日期索引查询
    def test_date_index_query_with_NaT(self, engine, parser):
        # 生成长度为 n 的随机数据框，使用对象类型以避免在设置 pd.NaT 时的隐式类型转换
        df = DataFrame(np.random.default_rng(2).standard_normal((n, 3))).astype(
            {0: object}
        )
        # 添加日期列 dates1 和 dates3
        df["dates1"] = date_range("1/1/2012", periods=n)
        df["dates3"] = date_range("1/1/2014", periods=n)
        # 将第一个元素设置为 pd.NaT
        df.iloc[0, 0] = pd.NaT
        # 设置索引为 dates1，并在原地丢弃原始索引列
        return_value = df.set_index("dates1", inplace=True, drop=True)
        assert return_value is None
        # 执行查询，筛选出符合条件的数据框 res
        res = df.query("index < 20130101 < dates3", engine=engine, parser=parser)
        # 期望的结果数据框 expec，使用布尔索引进行筛选
        expec = df[(df.index < "20130101") & ("20130101" < df.dates3)]
        # 断言两个数据框是否相等
        tm.assert_frame_equal(res, expec)

    # 定义测试函数，用于测试带有 pd.NaT 的日期索引查询，并处理重复值
    def test_date_index_query_with_NaT_duplicates(self, engine, parser):
        # 创建包含日期列 dates1 和 dates3 的字典
        d = {}
        d["dates1"] = date_range("1/1/2012", periods=n)
        d["dates3"] = date_range("1/1/2014", periods=n)
        # 根据字典创建数据框 df
        df = DataFrame(d)
        # 根据随机生成的布尔数组，将部分 dates1 列的值设置为 pd.NaT
        df.loc[np.random.default_rng(2).random(n) > 0.5, "dates1"] = pd.NaT
        # 设置索引为 dates1，并在原地丢弃原始索引列
        return_value = df.set_index("dates1", inplace=True, drop=True)
        assert return_value is None
        # 执行查询，筛选出符合条件的数据框 res
        res = df.query("dates1 < 20130101 < dates3", engine=engine, parser=parser)
        # 期望的结果数据框 expec，使用布尔索引进行筛选
        expec = df[(df.index.to_series() < "20130101") & ("20130101" < df.dates3)]
        # 断言两个数据框是否相等
        tm.assert_frame_equal(res, expec)

    # 定义测试函数，用于测试包含非日期的列与日期列的比较
    def test_date_query_with_non_date(self, engine, parser):
        # 创建包含日期列 dates 和非日期列 nondate 的数据框 df
        df = DataFrame(
            {"dates": date_range("1/1/2012", periods=n), "nondate": np.arange(n)}
        )
        # 执行查询，比较日期列 dates 和非日期列 nondate 的值是否相等
        result = df.query("dates == nondate", parser=parser, engine=engine)
        # 断言查询结果长度为 0，即日期列和非日期列不可能相等
        assert len(result) == 0
        # 执行查询，比较日期列 dates 和非日期列 nondate 的值是否不相等
        result = df.query("dates != nondate", parser=parser, engine=engine)
        # 断言查询结果与原数据框 df 相等
        tm.assert_frame_equal(result, df)
        # 循环检查使用不支持的比较运算符（<, >, <=, >=）时是否会引发 TypeError 异常
        msg = r"Invalid comparison between dtype=datetime64\[ns\] and ndarray"
        for op in ["<", ">", "<=", ">="]:
            with pytest.raises(TypeError, match=msg):
                df.query(f"dates {op} nondate", parser=parser, engine=engine)

    # 定义测试函数，用于测试查询语法错误
    def test_query_syntax_error(self, engine, parser):
        # 创建包含列名 "i", "+", "r" 的数据框 df
        df = DataFrame({"i": range(10), "+": range(3, 13), "r": range(4, 14)})
        # 定义预期的语法错误信息
        msg = "invalid syntax"
        # 断言执行查询语法错误时会抛出 SyntaxError 异常，并且异常信息符合预期
        with pytest.raises(SyntaxError, match=msg):
            df.query("i - +", engine=engine, parser=parser)
    # 测试查询范围函数，使用指定的引擎和解析器
    def test_query_scope(self, engine, parser):
        # 如果解析器不是 Pandas 解析器，则跳过该测试
        skip_if_no_pandas_parser(parser)

        # 创建一个包含随机标准正态分布数据的 DataFrame，包括两列 'a' 和 'b'
        df = DataFrame(
            np.random.default_rng(2).standard_normal((20, 2)), columns=list("ab")
        )

        # 定义局部变量 a 和 b，并标记其未使用的警告
        a, b = 1, 2  # noqa: F841

        # 使用 df.query 方法查询 DataFrame 中满足条件 'a > b' 的行
        res = df.query("a > b", engine=engine, parser=parser)
        # 创建预期的 DataFrame，其中包含满足条件 'df.a > df.b' 的行
        expected = df[df.a > df.b]
        # 断言查询结果与预期结果相等
        tm.assert_frame_equal(res, expected)

        # 使用 @ 符号引用局部变量 a，查询条件为 '@a > b'
        res = df.query("@a > b", engine=engine, parser=parser)
        # 创建预期的 DataFrame，其中包含满足条件 'a > df.b' 的行
        expected = df[a > df.b]
        # 断言查询结果与预期结果相等
        tm.assert_frame_equal(res, expected)

        # 在查询中引用了未定义的局部变量 'c'，预期引发 UndefinedVariableError 异常
        with pytest.raises(
            UndefinedVariableError, match="local variable 'c' is not defined"
        ):
            df.query("@a > b > @c", engine=engine, parser=parser)

        # 在查询中引用了不存在的列名 'c'，预期引发 UndefinedVariableError 异常
        with pytest.raises(UndefinedVariableError, match="name 'c' is not defined"):
            df.query("@a > b > c", engine=engine, parser=parser)

    # 测试查询不捕获局部变量的函数，使用指定的引擎和解析器
    def test_query_doesnt_pickup_local(self, engine, parser):
        # 定义变量 n 和 m，并将它们设置为相同的值 10
        n = m = 10
        # 创建一个包含随机整数数据的 DataFrame，包括三列 'a', 'b', 'c'
        df = DataFrame(
            np.random.default_rng(2).integers(m, size=(n, 3)), columns=list("abc")
        )

        # 在查询中引用了未定义的函数 'sin'，预期引发 UndefinedVariableError 异常
        with pytest.raises(UndefinedVariableError, match="name 'sin' is not defined"):
            df.query("sin > 5", engine=engine, parser=parser)

    # 测试查询内置函数的函数，使用指定的引擎和解析器
    def test_query_builtin(self, engine, parser):
        # 定义变量 n 和 m，并将它们设置为相同的值 10
        n = m = 10
        # 创建一个包含随机整数数据的 DataFrame，包括三列 'a', 'b', 'c'
        df = DataFrame(
            np.random.default_rng(2).integers(m, size=(n, 3)), columns=list("abc")
        )

        # 将 DataFrame 的索引名称设置为 'sin'
        df.index.name = "sin"
        # 定义异常消息的部分内容
        msg = "Variables in expression.+"
        # 在查询中引用了 DataFrame 索引 'sin'，预期引发 NumExprClobberingError 异常
        with pytest.raises(NumExprClobberingError, match=msg):
            df.query("sin > 5", engine=engine, parser=parser)

    # 测试查询函数，使用指定的引擎和解析器
    def test_query(self, engine, parser):
        # 创建一个包含随机标准正态分布数据的 DataFrame，包括三列 'a', 'b', 'c'
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 3)), columns=["a", "b", "c"]
        )

        # 断言查询结果与预期结果相等，查询条件为 'a < b'
        tm.assert_frame_equal(
            df.query("a < b", engine=engine, parser=parser), df[df.a < df.b]
        )
        # 断言查询结果与预期结果相等，查询条件为 'a + b > b * c'
        tm.assert_frame_equal(
            df.query("a + b > b * c", engine=engine, parser=parser),
            df[df.a + df.b > df.b * df.c],
        )

    # 测试带有名称索引的查询函数，使用指定的引擎和解析器
    def test_query_index_with_name(self, engine, parser):
        # 创建一个包含随机整数数据的 DataFrame，包括三列 'a', 'b', 'c'，和一个名称为 'blob' 的索引
        df = DataFrame(
            np.random.default_rng(2).integers(10, size=(10, 3)),
            index=Index(range(10), name="blob"),
            columns=["a", "b", "c"],
        )

        # 使用查询条件 '(blob < 5) & (a < b)' 查询 DataFrame
        res = df.query("(blob < 5) & (a < b)", engine=engine, parser=parser)
        # 创建预期的 DataFrame，包含满足条件 '(df.index < 5) & (df.a < df.b)' 的行
        expec = df[(df.index < 5) & (df.a < df.b)]
        # 断言查询结果与预期结果相等
        tm.assert_frame_equal(res, expec)

        # 使用查询条件 'blob < b' 查询 DataFrame
        res = df.query("blob < b", engine=engine, parser=parser)
        # 创建预期的 DataFrame，包含满足条件 'df.index < df.b' 的行
        expec = df[df.index < df.b]
        # 断言查询结果与预期结果相等
        tm.assert_frame_equal(res, expec)
    # 测试在没有指定列名的情况下查询索引，生成一个包含随机整数的 DataFrame
    def test_query_index_without_name(self, engine, parser):
        df = DataFrame(
            np.random.default_rng(2).integers(10, size=(10, 3)),
            index=range(10),
            columns=["a", "b", "c"],
        )

        # 使用查询表达式"index < b"从 DataFrame 中筛选符合条件的行，结果保存在 res 中
        res = df.query("index < b", engine=engine, parser=parser)
        # 期望的结果是通过普通索引和列名筛选得到的 DataFrame，保存在 expec 中
        expec = df[df.index < df.b]
        # 断言 res 和 expec 是否相等
        tm.assert_frame_equal(res, expec)

        # 对一个标量进行查询测试
        res = df.query("index < 5", engine=engine, parser=parser)
        expec = df[df.index < 5]
        tm.assert_frame_equal(res, expec)

    # 测试嵌套作用域
    def test_nested_scope(self, engine, parser):
        # 如果没有 Pandas 解析器，则跳过测试
        skip_if_no_pandas_parser(parser)

        # 创建两个随机标准正态分布的 DataFrame
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
        df2 = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
        # 预期的 DataFrame 是两个条件同时满足的结果
        expected = df[(df > 0) & (df2 > 0)]

        # 使用查询表达式"(@df > 0) & (@df2 > 0)"进行查询，结果保存在 result 中
        result = df.query("(@df > 0) & (@df2 > 0)", engine=engine, parser=parser)
        # 断言 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

        # 使用 pd.eval 函数执行查询"df[df > 0 and df2 > 0]"，结果保存在 result 中
        result = pd.eval("df[df > 0 and df2 > 0]", engine=engine, parser=parser)
        # 断言 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

        # 使用 pd.eval 函数执行查询"df[df > 0 and df2 > 0 and df[df > 0] > 0]"，结果保存在 result 中
        result = pd.eval(
            "df[df > 0 and df2 > 0 and df[df > 0] > 0]", engine=engine, parser=parser
        )
        expected = df[(df > 0) & (df2 > 0) & (df[df > 0] > 0)]
        # 断言 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

        # 使用 pd.eval 函数执行查询"df[(df>0) & (df2>0)]"，结果保存在 result 中
        result = pd.eval("df[(df>0) & (df2>0)]", engine=engine, parser=parser)
        # 期望的结果是通过 query 方法查询得到的结果，保存在 expected 中
        expected = df.query("(@df>0) & (@df2>0)", engine=engine, parser=parser)
        # 断言 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

    # 测试在局部作用域中引发异常
    def test_nested_raises_on_local_self_reference(self, engine, parser):
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))

        # 由于 df 是局部变量，无法直接引用自身，因此必须使用 @ 符号
        with pytest.raises(UndefinedVariableError, match="name 'df' is not defined"):
            df.query("df > 0", engine=engine, parser=parser)

    # 测试局部语法
    def test_local_syntax(self, engine, parser):
        # 如果没有 Pandas 解析器，则跳过测试
        skip_if_no_pandas_parser(parser)

        # 创建一个包含随机标准正态分布数据的 DataFrame，列名为字母表前10个字母
        df = DataFrame(
            np.random.default_rng(2).standard_normal((100, 10)),
            columns=list("abcdefghij"),
        )
        b = 1
        # 期望的结果是通过普通索引和标量比较得到的 DataFrame，保存在 expect 中
        expect = df[df.a < b]
        # 使用查询表达式"a < @b"进行查询，结果保存在 result 中
        result = df.query("a < @b", engine=engine, parser=parser)
        # 断言 result 和 expect 是否相等
        tm.assert_frame_equal(result, expect)

        # 期望的结果是通过普通索引和列名比较得到的 DataFrame，保存在 expect 中
        expect = df[df.a < df.b]
        # 使用查询表达式"a < b"进行查询，结果保存在 result 中
        result = df.query("a < b", engine=engine, parser=parser)
        # 断言 result 和 expect 是否相等
        tm.assert_frame_equal(result, expect)

    # 测试链式比较和 in 操作符
    def test_chained_cmp_and_in(self, engine, parser):
        # 如果没有 Pandas 解析器，则跳过测试
        skip_if_no_pandas_parser(parser)
        cols = list("abc")
        # 创建一个包含随机标准正态分布数据的 DataFrame，列名为"abc"
        df = DataFrame(
            np.random.default_rng(2).standard_normal((100, len(cols))), columns=cols
        )
        # 使用查询表达式"a < b < c and a not in b not in c"进行查询，结果保存在 res 中
        res = df.query(
            "a < b < c and a not in b not in c", engine=engine, parser=parser
        )
        # 期望的结果是通过普通索引和列比较得到的 DataFrame，保存在 expec 中
        ind = (df.a < df.b) & (df.b < df.c) & ~df.b.isin(df.a) & ~df.c.isin(df.b)
        expec = df[ind]
        # 断言 res 和 expec 是否相等
        tm.assert_frame_equal(res, expec)
    # 测试本地变量在查询中使用带有 'in' 操作符的情况
    def test_local_variable_with_in(self, engine, parser):
        # 如果没有 Pandas 解析器，则跳过测试
        skip_if_no_pandas_parser(parser)
        
        # 创建 Series 'a'，包含随机整数数据
        a = Series(np.random.default_rng(2).integers(3, size=15), name="a")
        # 创建 Series 'b'，包含随机整数数据
        b = Series(np.random.default_rng(2).integers(10, size=15), name="b")
        # 创建 DataFrame 'df'，包含 'a' 和 'b' 两列
        df = DataFrame({"a": a, "b": b})

        # 根据条件选择出预期的 DataFrame 行
        expected = df.loc[(df.b - 1).isin(a)]
        # 使用 query 方法查询符合条件的 DataFrame 行，并设置引擎和解析器
        result = df.query("b - 1 in a", engine=engine, parser=parser)
        # 断言查询结果与预期结果相等
        tm.assert_frame_equal(expected, result)

        # 重新创建 Series 'b'，包含新的随机整数数据
        b = Series(np.random.default_rng(2).integers(10, size=15), name="b")
        # 根据条件选择出预期的 DataFrame 行
        expected = df.loc[(b - 1).isin(a)]
        # 使用 query 方法查询符合条件的 DataFrame 行，@b 表示使用局部变量 'b'
        result = df.query("@b - 1 in a", engine=engine, parser=parser)
        # 断言查询结果与预期结果相等
        tm.assert_frame_equal(expected, result)

    # 测试在字符串内部使用 '@' 符号
    def test_at_inside_string(self, engine, parser):
        # 如果没有 Pandas 解析器，则跳过测试
        skip_if_no_pandas_parser(parser)
        
        # 定义变量 'c'，用于引发 noqa: F841 告警
        c = 1  # noqa: F841
        # 创建包含字符串数据的 DataFrame 'df'
        df = DataFrame({"a": ["a", "a", "b", "b", "@c", "@c"]})
        # 使用 query 方法查询符合条件的 DataFrame 行，'@c' 表示字符串 '@c'
        result = df.query('a == "@c"', engine=engine, parser=parser)
        # 根据条件选择出预期的 DataFrame 行
        expected = df[df.a == "@c"]
        # 断言查询结果与预期结果相等
        tm.assert_frame_equal(result, expected)

    # 测试查询中使用未定义的本地变量
    def test_query_undefined_local(self):
        # 获取引擎和解析器实例
        engine, parser = self.engine, self.parser
        # 如果没有 Pandas 解析器，则跳过测试
        skip_if_no_pandas_parser(parser)

        # 创建包含随机数据的 DataFrame 'df'
        df = DataFrame(np.random.default_rng(2).random((10, 2)), columns=list("ab"))
        # 使用 pytest 检查是否引发 UndefinedVariableError 异常，匹配异常信息
        with pytest.raises(
            UndefinedVariableError, match="local variable 'c' is not defined"
        ):
            # 使用 query 方法查询符合条件的 DataFrame 行，引用了未定义的变量 'c'
            df.query("a == @c", engine=engine, parser=parser)

    # 测试在查询中使用索引解析器来筛选数据
    def test_index_resolvers_come_after_columns_with_the_same_name(
        self, engine, parser
    ):
        # 定义变量 'n'，用于引发 noqa: F841 告警
        n = 1  # noqa: F841
        # 创建包含索引 'a' 和随机数据的 DataFrame 'df'
        a = np.r_[20:101:20]
        df = DataFrame(
            {"index": a, "b": np.random.default_rng(2).standard_normal(a.size)}
        )
        # 设置 DataFrame 的索引名称为 'index'
        df.index.name = "index"

        # 使用 query 方法查询符合条件的 DataFrame 行
        result = df.query("index > 5", engine=engine, parser=parser)
        # 根据条件选择出预期的 DataFrame 行
        expected = df[df["index"] > 5]
        # 断言查询结果与预期结果相等
        tm.assert_frame_equal(result, expected)

        # 创建包含索引 'a' 和随机数据的新 DataFrame 'df'
        df = DataFrame(
            {"index": a, "b": np.random.default_rng(2).standard_normal(a.size)}
        )
        # 使用 query 方法查询符合条件的 DataFrame 行
        result = df.query("ilevel_0 > 5", engine=engine, parser=parser)
        # 根据条件选择出预期的 DataFrame 行
        expected = df.loc[df.index[df.index > 5]]
        # 断言查询结果与预期结果相等
        tm.assert_frame_equal(result, expected)

        # 创建包含索引 'a' 和随机数据的新 DataFrame 'df'
        df = DataFrame({"a": a, "b": np.random.default_rng(2).standard_normal(a.size)})
        # 设置 DataFrame 的索引名称为 'a'
        df.index.name = "a"

        # 使用 query 方法查询符合条件的 DataFrame 行
        result = df.query("a > 5", engine=engine, parser=parser)
        # 根据条件选择出预期的 DataFrame 行
        expected = df[df.a > 5]
        # 断言查询结果与预期结果相等
        tm.assert_frame_equal(result, expected)

        # 使用 query 方法查询符合条件的 DataFrame 行
        result = df.query("index > 5", engine=engine, parser=parser)
        # 根据条件选择出预期的 DataFrame 行
        expected = df.loc[df.index[df.index > 5]]
        # 断言查询结果与预期结果相等
        tm.assert_frame_equal(result, expected)

    # 使用 pytest 参数化标记，测试不同的操作符和函数
    @pytest.mark.parametrize("op, f", [["==", operator.eq], ["!=", operator.ne]])
    def test_inf(self, op, f, engine, parser):
        # 定义测试函数，接受操作符、函数、引擎和解析器作为参数

        n = 10
        # 设置数据集大小

        df = DataFrame(
            {
                "a": np.random.default_rng(2).random(n),
                "b": np.random.default_rng(2).random(n),
            }
        )
        # 创建包含随机数的DataFrame，列名为'a'和'b'

        df.loc[::2, 0] = np.inf
        # 将DataFrame中每隔一行的第一列设置为正无穷

        q = f"a {op} inf"
        # 构造查询表达式，例如"a op inf"

        expected = df[f(df.a, np.inf)]
        # 计算预期的查询结果，使用函数f比较DataFrame的列'a'和np.inf

        result = df.query(q, engine=engine, parser=parser)
        # 使用查询表达式q在DataFrame上进行查询操作，使用指定的引擎和解析器

        tm.assert_frame_equal(result, expected)
        # 断言查询结果与预期结果相等

    def test_check_tz_aware_index_query(self, tz_aware_fixture):
        # 测试处理时区感知索引的查询功能

        tz = tz_aware_fixture
        # 使用给定的时区感知fixture

        df_index = date_range(
            start="2019-01-01", freq="1d", periods=10, tz=tz, name="time"
        )
        # 创建一个包含日期索引的DataFrame，日期从2019-01-01开始，每天一条记录，共10天，带有指定时区，索引名为'time'

        expected = DataFrame(index=df_index)
        # 创建预期结果DataFrame，只包含日期索引

        df = DataFrame(index=df_index)
        # 创建一个与预期相同的DataFrame，同样带有日期索引

        result = df.query('"2018-01-03 00:00:00+00" < time')
        # 在DataFrame上执行查询操作，查找大于指定日期时间的记录

        tm.assert_frame_equal(result, expected)
        # 断言查询结果与预期结果相等

        expected = DataFrame(df_index)
        # 创建一个新的预期结果DataFrame，将df_index直接作为数据

        result = df.reset_index().query('"2018-01-03 00:00:00+00" < time')
        # 将DataFrame重置索引后再执行查询操作，查找大于指定日期时间的记录

        tm.assert_frame_equal(result, expected)
        # 断言查询结果与预期结果相等

    def test_method_calls_in_query(self, engine, parser):
        # 测试查询中的方法调用功能

        n = 10
        # 设置数据集大小

        df = DataFrame(
            {
                "a": 2 * np.random.default_rng(2).random(n),
                "b": np.random.default_rng(2).random(n),
            }
        )
        # 创建包含随机数的DataFrame，列名为'a'和'b'

        expected = df[df["a"].astype("int") == 0]
        # 计算预期的查询结果，使用DataFrame列'a'转换为整数后与0比较

        result = df.query("a.astype('int') == 0", engine=engine, parser=parser)
        # 使用查询表达式在DataFrame上执行查询操作，使用指定的引擎和解析器

        tm.assert_frame_equal(result, expected)
        # 断言查询结果与预期结果相等

        df = DataFrame(
            {
                "a": np.where(
                    np.random.default_rng(2).random(n) < 0.5,
                    np.nan,
                    np.random.default_rng(2).standard_normal(n),
                ),
                "b": np.random.default_rng(2).standard_normal(n),
            }
        )
        # 创建包含随机数的DataFrame，列名为'a'和'b'，其中'a'的部分值可能为NaN

        expected = df[df["a"].notnull()]
        # 计算预期的查询结果，筛选出列'a'中不为NaN的记录

        result = df.query("a.notnull()", engine=engine, parser=parser)
        # 使用查询表达式在DataFrame上执行查询操作，使用指定的引擎和解析器

        tm.assert_frame_equal(result, expected)
        # 断言查询结果与预期结果相等
# 如果未安装 numexpr 库，则跳过测试用例
@td.skip_if_no("numexpr")
class TestDataFrameQueryNumExprPython(TestDataFrameQueryNumExprPandas):
    # 定义 fixture，返回测试引擎为 "numexpr"
    @pytest.fixture
    def engine(self):
        return "numexpr"

    # 定义 fixture，返回解析器为 "python"
    @pytest.fixture
    def parser(self):
        return "python"

    # 测试日期查询，不涉及属性访问
    def test_date_query_no_attribute_access(self, engine, parser):
        # 创建一个 DataFrame，包含随机数据和日期列
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
        df["dates1"] = date_range("1/1/2012", periods=5)
        df["dates2"] = date_range("1/1/2013", periods=5)
        df["dates3"] = date_range("1/1/2014", periods=5)
        # 执行查询操作，使用指定的引擎和解析器
        res = df.query(
            "(dates1 < 20130101) & (20130101 < dates3)", engine=engine, parser=parser
        )
        # 期望的查询结果
        expec = df[(df.dates1 < "20130101") & ("20130101" < df.dates3)]
        # 断言结果与期望是否相等
        tm.assert_frame_equal(res, expec)

    # 测试日期查询，包含 NaT 值情况
    def test_date_query_with_NaT(self, engine, parser):
        n = 10
        df = DataFrame(np.random.default_rng(2).standard_normal((n, 3)))
        df["dates1"] = date_range("1/1/2012", periods=n)
        df["dates2"] = date_range("1/1/2013", periods=n)
        df["dates3"] = date_range("1/1/2014", periods=n)
        # 随机设置部分日期为 NaT
        df.loc[np.random.default_rng(2).random(n) > 0.5, "dates1"] = pd.NaT
        df.loc[np.random.default_rng(2).random(n) > 0.5, "dates3"] = pd.NaT
        # 执行查询操作，使用指定的引擎和解析器
        res = df.query(
            "(dates1 < 20130101) & (20130101 < dates3)", engine=engine, parser=parser
        )
        # 期望的查询结果
        expec = df[(df.dates1 < "20130101") & ("20130101" < df.dates3)]
        # 断言结果与期望是否相等
        tm.assert_frame_equal(res, expec)

    # 测试使用日期作为索引的查询
    def test_date_index_query(self, engine, parser):
        n = 10
        df = DataFrame(np.random.default_rng(2).standard_normal((n, 3)))
        df["dates1"] = date_range("1/1/2012", periods=n)
        df["dates3"] = date_range("1/1/2014", periods=n)
        # 将 dates1 列设置为索引
        return_value = df.set_index("dates1", inplace=True, drop=True)
        assert return_value is None
        # 执行查询操作，使用指定的引擎和解析器
        res = df.query(
            "(index < 20130101) & (20130101 < dates3)", engine=engine, parser=parser
        )
        # 期望的查询结果
        expec = df[(df.index < "20130101") & ("20130101" < df.dates3)]
        # 断言结果与期望是否相等
        tm.assert_frame_equal(res, expec)

    # 测试使用日期作为索引的查询，包含 NaT 值情况
    def test_date_index_query_with_NaT(self, engine, parser):
        n = 10
        # 将第一列强制转换为对象类型，避免将 pd.NaT 隐式转换
        df = DataFrame(np.random.default_rng(2).standard_normal((n, 3))).astype(
            {0: object}
        )
        df["dates1"] = date_range("1/1/2012", periods=n)
        df["dates3"] = date_range("1/1/2014", periods=n)
        df.iloc[0, 0] = pd.NaT
        # 将 dates1 列设置为索引
        return_value = df.set_index("dates1", inplace=True, drop=True)
        assert return_value is None
        # 执行查询操作，使用指定的引擎和解析器
        res = df.query(
            "(index < 20130101) & (20130101 < dates3)", engine=engine, parser=parser
        )
        # 期望的查询结果
        expec = df[(df.index < "20130101") & ("20130101" < df.dates3)]
        # 断言结果与期望是否相等
        tm.assert_frame_equal(res, expec)
    # 测试带有 NaT（Not a Time）重复项的日期索引查询功能
    def test_date_index_query_with_NaT_duplicates(self, engine, parser):
        # 创建一个包含随机标准正态分布数据的 DataFrame，形状为 (n, 3)
        n = 10
        df = DataFrame(np.random.default_rng(2).standard_normal((n, 3)))
        # 在 DataFrame 中添加名为 "dates1" 的日期范围列，从 '1/1/2012' 开始
        df["dates1"] = date_range("1/1/2012", periods=n)
        # 添加名为 "dates3" 的日期范围列，从 '1/1/2014' 开始
        df["dates3"] = date_range("1/1/2014", periods=n)
        # 随机将 "dates1" 列中超过 0.5 的行设置为 NaT
        df.loc[np.random.default_rng(2).random(n) > 0.5, "dates1"] = pd.NaT
        # 将 "dates1" 列设置为索引，并且在原地操作并丢弃旧索引
        return_value = df.set_index("dates1", inplace=True, drop=True)
        assert return_value is None
        # 断言引发 NotImplementedError 异常，且异常消息需包含 "'BoolOp' nodes are not implemented"
        msg = r"'BoolOp' nodes are not implemented"
        with pytest.raises(NotImplementedError, match=msg):
            # 使用引擎和解析器执行查询操作
            df.query("index < 20130101 < dates3", engine=engine, parser=parser)

    # 测试嵌套作用域
    def test_nested_scope(self, engine, parser):
        # 烟雾测试
        x = 1  # noqa: F841
        # 使用引擎和解析器执行表达式 "x + 1"，并将结果存储在 result 中
        result = pd.eval("x + 1", engine=engine, parser=parser)
        assert result == 2

        # 创建两个包含随机标准正态分布数据的 DataFrame，形状均为 (5, 3)
        df = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))
        df2 = DataFrame(np.random.default_rng(2).standard_normal((5, 3)))

        # 断言引发 SyntaxError 异常，且异常消息需包含 "The '@' prefix is only supported by the pandas parser"
        msg = r"The '@' prefix is only supported by the pandas parser"
        with pytest.raises(SyntaxError, match=msg):
            # 使用引擎和解析器执行查询操作，查询中使用了 '@' 前缀
            df.query("(@df>0) & (@df2>0)", engine=engine, parser=parser)

        # 断言引发 UndefinedVariableError 异常，且异常消息需包含 "name 'df' is not defined"
        with pytest.raises(UndefinedVariableError, match="name 'df' is not defined"):
            # 使用引擎和解析器执行查询操作，但查询中使用了未定义的变量 'df'
            df.query("(df>0) & (df2>0)", engine=engine, parser=parser)

        # 期望的结果是一个包含满足条件的 DataFrame 子集
        expected = df[(df > 0) & (df2 > 0)]
        # 使用引擎和解析器执行表达式 "df[(df > 0) & (df2 > 0)]"，并将结果存储在 result 中
        result = pd.eval("df[(df > 0) & (df2 > 0)]", engine=engine, parser=parser)
        # 断言 result 和 expected 相等
        tm.assert_frame_equal(expected, result)

        # 期望的结果是一个包含满足条件的 DataFrame 子集
        expected = df[(df > 0) & (df2 > 0) & (df[df > 0] > 0)]
        # 使用引擎和解析器执行表达式 "df[(df > 0) & (df2 > 0) & (df[df > 0] > 0)]"，并将结果存储在 result 中
        result = pd.eval(
            "df[(df > 0) & (df2 > 0) & (df[df > 0] > 0)]", engine=engine, parser=parser
        )
        # 断言 result 和 expected 相等
        tm.assert_frame_equal(expected, result)

    # 测试带有最小和最大列的 numexpr 查询
    def test_query_numexpr_with_min_and_max_columns(self):
        # 创建一个包含两列 "min" 和 "max" 的 DataFrame
        df = DataFrame({"min": [1, 2, 3], "max": [4, 5, 6]})
        # 断言引发 NumExprClobberingError 异常，且异常消息需包含 "Variables in expression "\(min\) == \(1\)" overlap with builtins: \('min'\)"
        regex_to_match = (
            r"Variables in expression \"\(min\) == \(1\)\" "
            r"overlap with builtins: \('min'\)"
        )
        with pytest.raises(NumExprClobberingError, match=regex_to_match):
            # 使用引擎和解析器执行查询操作，查询条件中涉及到 "min == 1"
            df.query("min == 1")

        # 断言引发 NumExprClobberingError 异常，且异常消息需包含 "Variables in expression "\(max\) == \(1\)" overlap with builtins: \('max'\)"
        regex_to_match = (
            r"Variables in expression \"\(max\) == \(1\)\" "
            r"overlap with builtins: \('max'\)"
        )
        with pytest.raises(NumExprClobberingError, match=regex_to_match):
            # 使用引擎和解析器执行查询操作，查询条件中涉及到 "max == 1"
            df.query("max == 1")
class TestDataFrameQueryStrings:
    # 定义测试类 `TestDataFrameQueryStrings`
    def test_str_query_method(self, parser, engine):
        # 测试方法 `test_str_query_method`，接收 `parser` 和 `engine` 作为参数

        # 创建一个包含随机数的 DataFrame，列名为 `b` 和 `strings`
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 1)), columns=["b"])
        df["strings"] = Series(list("aabbccddee"))

        # 期望的结果 DataFrame，选择 `strings` 列中值为 "a" 的行
        expect = df[df.strings == "a"]

        # 如果 parser 不是 "pandas"，则执行以下逻辑
        if parser != "pandas":
            # 设置要比较的列名和列表
            col = "strings"
            lst = '"a"'

            # 创建左右操作数列表
            lhs = [col] * 2 + [lst] * 2
            rhs = lhs[::-1]

            # 设置比较操作符和相关消息
            eq, ne = "==", "!="
            ops = 2 * ([eq] + [ne])
            msg = r"'(Not)?In' nodes are not implemented"

            # 遍历操作数，预期会抛出 NotImplementedError 异常，消息为 `msg`
            for lh, op_, rh in zip(lhs, ops, rhs):
                ex = f"{lh} {op_} {rh}"
                with pytest.raises(NotImplementedError, match=msg):
                    df.query(
                        ex,
                        engine=engine,
                        parser=parser,
                        local_dict={"strings": df.strings},
                    )
        else:
            # 否则，执行以下逻辑（parser 是 "pandas"）

            # 查询语句，比较 "a" 是否等于 `strings` 列中的值，使用给定的 engine 和 parser
            res = df.query('"a" == strings', engine=engine, parser=parser)
            tm.assert_frame_equal(res, expect)

            # 查询语句，比较 `strings` 列中的值是否等于 "a"，使用给定的 engine 和 parser
            res = df.query('strings == "a"', engine=engine, parser=parser)
            tm.assert_frame_equal(res, expect)

            # 使用 `isin` 函数，比较 `strings` 列中的值是否在 ["a"] 中，与期望结果比较
            tm.assert_frame_equal(res, df[df.strings.isin(["a"])])

            # 更新期望结果，选择 `strings` 列中不等于 "a" 的行
            expect = df[df.strings != "a"]

            # 查询语句，比较 `strings` 列中的值是否不等于 "a"，与期望结果比较
            res = df.query('strings != "a"', engine=engine, parser=parser)
            tm.assert_frame_equal(res, expect)

            # 查询语句，比较 "a" 是否不等于 `strings` 列中的值，与期望结果比较
            tm.assert_frame_equal(res, df[~df.strings.isin(["a"])])
    # 定义一个测试方法，测试字符串列表查询功能
    def test_str_list_query_method(self, parser, engine):
        # 创建一个包含随机标准正态分布数据的 DataFrame，包含列名为 "b"
        df = DataFrame(np.random.default_rng(2).standard_normal((10, 1)), columns=["b"])
        # 在 DataFrame 中添加名为 "strings" 的列，并赋值为 ['a', 'a', 'b', 'b', 'c', 'c', 'd', 'd', 'e', 'e']
        df["strings"] = Series(list("aabbccddee"))
        # 期望的 DataFrame，包含了字符串列 "strings" 中值为 'a' 或 'b' 的行
        expect = df[df.strings.isin(["a", "b"])]

        # 如果不是用 Pandas 解析器进行测试
        if parser != "pandas":
            # 定义列名和字符串列表
            col = "strings"
            lst = '["a", "b"]'

            # 创建左右操作数列表和对应的操作符列表
            lhs = [col] * 2 + [lst] * 2
            rhs = lhs[::-1]
            eq, ne = "==", "!="
            ops = 2 * ([eq] + [ne])
            msg = r"'(Not)?In' nodes are not implemented"

            # 遍历左右操作数和操作符，构建查询表达式并验证抛出预期异常
            for lh, ops_, rh in zip(lhs, ops, rhs):
                ex = f"{lh} {ops_} {rh}"
                with pytest.raises(NotImplementedError, match=msg):
                    df.query(ex, engine=engine, parser=parser)
        else:
            # 使用 Pandas 解析器时，进行查询和预期结果比较
            res = df.query('strings == ["a", "b"]', engine=engine, parser=parser)
            tm.assert_frame_equal(res, expect)

            res = df.query('["a", "b"] == strings', engine=engine, parser=parser)
            tm.assert_frame_equal(res, expect)

            # 更新期望结果为不包含字符串列 "strings" 中值为 'a' 或 'b' 的行
            expect = df[~df.strings.isin(["a", "b"])]

            res = df.query('strings != ["a", "b"]', engine=engine, parser=parser)
            tm.assert_frame_equal(res, expect)

            res = df.query('["a", "b"] != strings', engine=engine, parser=parser)
            tm.assert_frame_equal(res, expect)

    # 定义测试字符串列查询功能的方法，考虑不同的解析器和引擎
    def test_query_with_string_columns(self, parser, engine):
        # 创建一个包含列 "a", "b", "c", "d" 的 DataFrame，其中 "a" 和 "b" 包含字符串列表和随机整数数据
        df = DataFrame(
            {
                "a": list("aaaabbbbcccc"),
                "b": list("aabbccddeeff"),
                "c": np.random.default_rng(2).integers(5, size=12),
                "d": np.random.default_rng(2).integers(9, size=12),
            }
        )
        
        # 如果解析器为 Pandas
        if parser == "pandas":
            # 执行带有 'a in b' 查询，并验证结果与预期是否一致
            res = df.query("a in b", parser=parser, engine=engine)
            expec = df[df.a.isin(df.b)]
            tm.assert_frame_equal(res, expec)

            # 执行带有 'a in b and c < d' 查询，并验证结果与预期是否一致
            res = df.query("a in b and c < d", parser=parser, engine=engine)
            expec = df[df.a.isin(df.b) & (df.c < df.d)]
            tm.assert_frame_equal(res, expec)
        else:
            # 当解析器不是 Pandas 时，验证是否抛出预期的异常信息
            msg = r"'(Not)?In' nodes are not implemented"
            with pytest.raises(NotImplementedError, match=msg):
                df.query("a in b", parser=parser, engine=engine)

            msg = r"'BoolOp' nodes are not implemented"
            with pytest.raises(NotImplementedError, match=msg):
                df.query("a in b and c < d", parser=parser, engine=engine)
    def test_object_array_eq_ne(self, parser, engine, using_infer_string):
        # 创建一个测试数据框 df，包含四列：a, b, c, d
        df = DataFrame(
            {
                "a": list("aaaabbbbcccc"),
                "b": list("aabbccddeeff"),
                "c": np.random.default_rng(2).integers(5, size=12),
                "d": np.random.default_rng(2).integers(9, size=12),
            }
        )
        # 根据条件选择是否设置 RuntimeWarning
        warning = RuntimeWarning if using_infer_string and engine == "numexpr" else None
        # 使用上下文管理器确保查询操作会产生警告
        with tm.assert_produces_warning(warning):
            # 执行查询，比较列 a 和列 b 是否相等
            res = df.query("a == b", parser=parser, engine=engine)
        # 期望的结果是 df 中满足 a == b 条件的行
        exp = df[df.a == df.b]
        # 断言查询结果与期望结果相等
        tm.assert_frame_equal(res, exp)

        with tm.assert_produces_warning(warning):
            # 执行查询，比较列 a 和列 b 是否不相等
            res = df.query("a != b", parser=parser, engine=engine)
        # 期望的结果是 df 中满足 a != b 条件的行
        exp = df[df.a != df.b]
        # 断言查询结果与期望结果相等
        tm.assert_frame_equal(res, exp)

    def test_query_with_nested_strings(self, parser, engine):
        # 如果没有指定的解析器，跳过测试
        skip_if_no_pandas_parser(parser)
        # 创建一个包含事件字符串的列表
        events = [
            f"page {n} {act}" for n in range(1, 4) for act in ["load", "exit"]
        ] * 2
        # 创建两个时间戳范围
        stamps1 = date_range("2014-01-01 0:00:01", freq="30s", periods=6)
        stamps2 = date_range("2014-02-01 1:00:01", freq="30s", periods=6)
        # 创建数据框 df，包含 id, event, timestamp 列
        df = DataFrame(
            {
                "id": np.arange(1, 7).repeat(2),
                "event": events,
                "timestamp": stamps1.append(stamps2),
            }
        )

        # 期望的结果是 df 中 event 列等于 '"page 1 load"' 的行
        expected = df[df.event == '"page 1 load"']
        # 执行查询，查找包含 '"page 1 load"' 字符串的 event 列
        res = df.query("""'"page 1 load"' in event""", parser=parser, engine=engine)
        # 断言查询结果与期望结果相等
        tm.assert_frame_equal(expected, res)

    def test_query_with_nested_special_character(self, parser, engine):
        # 如果没有指定的解析器，跳过测试
        skip_if_no_pandas_parser(parser)
        # 创建一个包含特殊字符的数据框 df
        df = DataFrame({"a": ["a", "b", "test & test"], "b": [1, 2, 3]})
        # 执行查询，查找包含特定字符串 'test & test' 的行
        res = df.query('a == "test & test"', parser=parser, engine=engine)
        # 期望的结果是 df 中列 a 等于 'test & test' 的行
        expec = df[df.a == "test & test"]
        # 断言查询结果与期望结果相等
        tm.assert_frame_equal(res, expec)

    @pytest.mark.parametrize(
        "op, func",
        [
            ["<", operator.lt],
            [">", operator.gt],
            ["<=", operator.le],
            [">=", operator.ge],
        ],
    )
    def test_query_lex_compare_strings(
        self, parser, engine, op, func, using_infer_string
    ):
        # 创建两个 Series 对象 a 和 b
        a = Series(np.random.default_rng(2).choice(list("abcde"), 20))
        b = Series(np.arange(a.size))
        # 创建数据框 df，包含 X 和 Y 两列
        df = DataFrame({"X": a, "Y": b})

        # 根据条件选择是否设置 RuntimeWarning
        warning = RuntimeWarning if using_infer_string and engine == "numexpr" else None
        # 使用上下文管理器确保查询操作会产生警告
        with tm.assert_produces_warning(warning):
            # 执行查询，比较列 X 是否小于 'd'
            res = df.query(f'X {op} "d"', engine=engine, parser=parser)
        # 期望的结果是 df 中满足 X 与 'd' 比较条件的行
        expected = df[func(df.X, "d")]
        # 断言查询结果与期望结果相等
        tm.assert_frame_equal(res, expected)
    # 定义一个测试方法，用于测试查询单个元素的布尔运算
    def test_query_single_element_booleans(self, parser, engine):
        # 定义数据列名
        columns = "bid", "bidsize", "ask", "asksize"
        # 生成随机的布尔类型数据，形状为(1, 列数)，并转换为布尔型
        data = np.random.default_rng(2).integers(2, size=(1, len(columns))).astype(bool)
        # 创建 DataFrame 对象，用生成的数据和列名
        df = DataFrame(data, columns=columns)
        # 对 DataFrame 进行查询，使用引擎和解析器来处理查询字符串"bid & ask"
        res = df.query("bid & ask", engine=engine, parser=parser)
        # 生成期望的 DataFrame，包含符合条件的行
        expected = df[df.bid & df.ask]
        # 使用断言检查查询结果和期望结果是否相等
        tm.assert_frame_equal(res, expected)

    # 定义一个测试方法，用于测试查询包含字符串标量变量的情况
    def test_query_string_scalar_variable(self, parser, engine):
        # 如果没有 pandas 解析器，则跳过该测试
        skip_if_no_pandas_parser(parser)
        # 创建 DataFrame，包含标记和价格的数据
        df = DataFrame(
            {
                "Symbol": ["BUD US", "BUD US", "IBM US", "IBM US"],
                "Price": [109.70, 109.72, 183.30, 183.35],
            }
        )
        # 生成预期的 DataFrame，包含符合条件的行（Symbol == "BUD US"）
        e = df[df.Symbol == "BUD US"]
        # 设置符号变量为 "BUD US"，并标注为 noqa: F841，表示变量未使用
        symb = "BUD US"  # noqa: F841
        # 对 DataFrame 进行查询，使用引擎和解析器来处理查询字符串"Symbol == @symb"
        r = df.query("Symbol == @symb", parser=parser, engine=engine)
        # 使用断言检查查询结果和期望结果是否相等
        tm.assert_frame_equal(e, r)

    # 使用 pytest 的参数化装饰器，定义一个测试方法，用于测试查询包含空元素的字符串情况
    @pytest.mark.parametrize(
        "in_list",
        [
            [None, "asdf", "ghjk"],
            ["asdf", None, "ghjk"],
            ["asdf", "ghjk", None],
            [None, None, "asdf"],
            ["asdf", None, None],
            [None, None, None],
        ],
    )
    def test_query_string_null_elements(self, in_list):
        # 设置解析器和引擎
        parser = "pandas"
        engine = "python"
        # 生成预期的字典，包含符合条件的值为 "asdf" 的元素
        expected = {i: value for i, value in enumerate(in_list) if value == "asdf"}

        # 创建预期的 DataFrame，将预期字典放入列名为"a"的 DataFrame 中，数据类型为字符串
        df_expected = DataFrame({"a": expected}, dtype="string")
        # 将索引类型转换为 int64 类型
        df_expected.index = df_expected.index.astype("int64")
        
        # 创建 DataFrame，将输入列表放入列名为"a"的 DataFrame 中，数据类型为字符串
        df = DataFrame({"a": in_list}, dtype="string")
        
        # 对 DataFrame 进行查询，查询条件为"a == 'asdf'"，使用解析器和引擎来处理
        res1 = df.query("a == 'asdf'", parser=parser, engine=engine)
        # 生成期望的 DataFrame，包含符合条件的行
        res2 = df[df["a"] == "asdf"]
        # 对 DataFrame 进行查询，查询条件为"a <= 'asdf'"，使用解析器和引擎来处理
        res3 = df.query("a <= 'asdf'", parser=parser, engine=engine)
        
        # 使用断言检查查询结果和期望结果是否相等
        tm.assert_frame_equal(res1, df_expected)
        tm.assert_frame_equal(res1, res2)
        tm.assert_frame_equal(res1, res3)
        tm.assert_frame_equal(res2, res3)
class TestDataFrameEvalWithFrame:
    # 定义测试类 `TestDataFrameEvalWithFrame`
    
    @pytest.fixture
    def frame(self):
        # 定义测试用例的 Fixture 函数 `frame`，返回一个 DataFrame 对象
        return DataFrame(
            np.random.default_rng(2).standard_normal((10, 3)), columns=list("abc")
        )

    def test_simple_expr(self, frame, parser, engine):
        # 测试简单表达式计算
        res = frame.eval("a + b", engine=engine, parser=parser)
        # 期望的结果是列 `a` 和 `b` 的求和
        expect = frame.a + frame.b
        tm.assert_series_equal(res, expect)

    def test_bool_arith_expr(self, frame, parser, engine):
        # 测试带有布尔条件的表达式计算
        res = frame.eval("a[a < 1] + b", engine=engine, parser=parser)
        # 期望的结果是满足条件 `a < 1` 的行的列 `a` 和 `b` 的求和
        expect = frame.a[frame.a < 1] + frame.b
        tm.assert_series_equal(res, expect)

    @pytest.mark.parametrize("op", ["+", "-", "*", "/"])
    def test_invalid_type_for_operator_raises(self, parser, engine, op):
        # 测试不支持的操作符引发异常
        df = DataFrame({"a": [1, 2], "b": ["c", "d"]})
        msg = r"unsupported operand type\(s\) for .+: '.+' and '.+'|Cannot"

        with pytest.raises(TypeError, match=msg):
            df.eval(f"a {op} b", engine=engine, parser=parser)


class TestDataFrameQueryBacktickQuoting:
    # 定义测试类 `TestDataFrameQueryBacktickQuoting`

    @pytest.fixture
    def df(self):
        """
        Yields a dataframe with strings that may or may not need escaping
        by backticks. The last two columns cannot be escaped by backticks
        and should raise a ValueError.
        """
        # 返回一个包含字符串列，可能需要使用反引号转义的 DataFrame 对象
        return DataFrame(
            {
                "A": [1, 2, 3],
                "B B": [3, 2, 1],
                "C C": [4, 5, 6],
                "C  C": [7, 4, 3],
                "C_C": [8, 9, 10],
                "D_D D": [11, 1, 101],
                "E.E": [6, 3, 5],
                "F-F": [8, 1, 10],
                "1e1": [2, 4, 8],
                "def": [10, 11, 2],
                "A (x)": [4, 1, 3],
                "B(x)": [1, 1, 5],
                "B (x)": [2, 7, 4],
                "  &^ :!€$?(} >    <++*''  ": [2, 5, 6],
                "": [10, 11, 1],
                " A": [4, 7, 9],
                "  ": [1, 2, 1],
                "it's": [6, 3, 1],
                "that's": [9, 1, 8],
                "☺": [8, 7, 6],
                "foo#bar": [2, 4, 5],
                1: [5, 7, 9],
            }
        )

    def test_single_backtick_variable_query(self, df):
        # 测试使用单个反引号的变量查询
        res = df.query("1 < `B B`")
        expect = df[1 < df["B B"]]
        tm.assert_frame_equal(res, expect)

    def test_two_backtick_variables_query(self, df):
        # 测试使用两个反引号的变量查询
        res = df.query("1 < `B B` and 4 < `C C`")
        expect = df[(1 < df["B B"]) & (4 < df["C C"])]
        tm.assert_frame_equal(res, expect)

    def test_single_backtick_variable_expr(self, df):
        # 测试使用单个反引号的变量表达式
        res = df.eval("A + `B B`")
        expect = df["A"] + df["B B"]
        tm.assert_series_equal(res, expect)

    def test_two_backtick_variables_expr(self, df):
        # 测试使用两个反引号的变量表达式
        res = df.eval("`B B` + `C C`")
        expect = df["B B"] + df["C C"]
        tm.assert_series_equal(res, expect)
    # 测试已经使用下划线命名的变量，使用 Pandas 的 eval 方法进行表达式计算
    def test_already_underscore_variable(self, df):
        # 执行表达式 "`C_C` + A"，计算结果
        res = df.eval("`C_C` + A")
        # 期望的计算结果是 df["C_C"] + df["A"]
        expect = df["C_C"] + df["A"]
        # 使用 Pandas 的 assert_series_equal 方法比较结果 res 和 expect 是否相等
        tm.assert_series_equal(res, expect)

    # 测试相同名称但使用不同的下划线方式的变量
    def test_same_name_but_underscores(self, df):
        # 执行表达式 "C_C + `C C`"，计算结果
        res = df.eval("C_C + `C C`")
        # 期望的计算结果是 df["C_C"] + df["C C"]
        expect = df["C_C"] + df["C C"]
        # 使用 Pandas 的 assert_series_equal 方法比较结果 res 和 expect 是否相等
        tm.assert_series_equal(res, expect)

    # 测试混合使用下划线和空格的变量名
    def test_mixed_underscores_and_spaces(self, df):
        # 执行表达式 "A + `D_D D`"，计算结果
        res = df.eval("A + `D_D D`")
        # 期望的计算结果是 df["A"] + df["D_D D"]
        expect = df["A"] + df["D_D D"]
        # 使用 Pandas 的 assert_series_equal 方法比较结果 res 和 expect 是否相等
        tm.assert_series_equal(res, expect)

    # 测试使用反引号括起来且不含空格的变量名
    def test_backtick_quote_name_with_no_spaces(self, df):
        # 执行表达式 "A + `C_C`"，计算结果
        res = df.eval("A + `C_C`")
        # 期望的计算结果是 df["A"] + df["C_C"]
        expect = df["A"] + df["C_C"]
        # 使用 Pandas 的 assert_series_equal 方法比较结果 res 和 expect 是否相等
        tm.assert_series_equal(res, expect)

    # 测试包含特殊字符的变量名
    def test_special_characters(self, df):
        # 执行表达式 "`E.E` + `F-F` - A"，计算结果
        res = df.eval("`E.E` + `F-F` - A")
        # 期望的计算结果是 df["E.E"] + df["F-F"] - df["A"]
        expect = df["E.E"] + df["F-F"] - df["A"]
        # 使用 Pandas 的 assert_series_equal 方法比较结果 res 和 expect 是否相等
        tm.assert_series_equal(res, expect)

    # 测试变量名以数字开头的情况
    def test_start_with_digit(self, df):
        # 执行表达式 "A + `1e1`"，计算结果
        res = df.eval("A + `1e1`")
        # 期望的计算结果是 df["A"] + df["1e1"]
        expect = df["A"] + df["1e1"]
        # 使用 Pandas 的 assert_series_equal 方法比较结果 res 和 expect 是否相等
        tm.assert_series_equal(res, expect)

    # 测试变量名是 Python 关键字的情况
    def test_keyword(self, df):
        # 执行表达式 "A + `def`"，计算结果
        res = df.eval("A + `def`")
        # 期望的计算结果是 df["A"] + df["def"]
        expect = df["A"] + df["def"]
        # 使用 Pandas 的 assert_series_equal 方法比较结果 res 和 expect 是否相等
        tm.assert_series_equal(res, expect)

    # 测试不需要使用反引号的情况
    def test_unneeded_quoting(self, df):
        # 使用 Pandas 的 query 方法查询表达式 "`A` > 2"
        res = df.query("`A` > 2")
        # 期望的查询结果是 df[df["A"] > 2]
        expect = df[df["A"] > 2]
        # 使用 Pandas 的 assert_frame_equal 方法比较结果 res 和 expect 是否相等
        tm.assert_frame_equal(res, expect)

    # 测试变量名包含括号的情况
    def test_parenthesis(self, df):
        # 使用 Pandas 的 query 方法查询表达式 "`A (x)` > 2"
        res = df.query("`A (x)` > 2")
        # 期望的查询结果是 df[df["A (x)"] > 2]
        expect = df[df["A (x)"] > 2]
        # 使用 Pandas 的 assert_frame_equal 方法比较结果 res 和 expect 是否相等
        tm.assert_frame_equal(res, expect)

    # 测试变量名为空字符串的情况
    def test_empty_string(self, df):
        # 使用 Pandas 的 query 方法查询表达式 "`` > 5"
        res = df.query("`` > 5")
        # 期望的查询结果是 df[df[""] > 5]
        expect = df[df[""] > 5]
        # 使用 Pandas 的 assert_frame_equal 方法比较结果 res 和 expect 是否相等
        tm.assert_frame_equal(res, expect)

    # 测试变量名包含多个空格的情况
    def test_multiple_spaces(self, df):
        # 使用 Pandas 的 query 方法查询表达式 "`C  C` > 5"
        res = df.query("`C  C` > 5")
        # 期望的查询结果是 df[df["C  C"] > 5]
        expect = df[df["C  C"] > 5]
        # 使用 Pandas 的 assert_frame_equal 方法比较结果 res 和 expect 是否相等
        tm.assert_frame_equal(res, expect)

    # 测试变量名以空格开头的情况
    def test_start_with_spaces(self, df):
        # 执行表达式 "` A` + `  `"，计算结果
        res = df.eval("` A` + `  `")
        # 期望的计算结果是 df[" A"] + df["  "]
        expect = df[" A"] + df["  "]
        # 使用 Pandas 的 assert_series_equal 方法比较结果 res 和 expect 是否相等
        tm.assert_series_equal(res, expect)

    # 测试包含大量运算符和空格的变量名
    def test_lots_of_operators_string(self, df):
        # 使用 Pandas 的 query 方法查询表达式 "`  &^ :!€$?(} >    <++*''  ` > 4"
        res = df.query("`  &^ :!€$?(} >    <++*''  ` > 4")
        # 期望的查询结果是 df[df["  &^ :!€$?(} >    <++*''  "] > 4]
        expect = df[df["  &^ :!€$?(} >    <++*''  "] > 4]
        # 使用 Pandas 的 assert_frame_equal 方法比较结果 res 和 expect 是否相等
        tm.assert_frame_equal(res, expect)

    # 测试访问不存在的属性时是否会引发异常
    def test_missing_attribute(self, df):
        # 预期的错误消息
        message = "module 'pandas' has no attribute 'thing'"
        # 使用 pytest 的 pytest.raises 方法检查是否会抛出 AttributeError 异常，并且错误消息匹配 message
        with pytest.raises(AttributeError, match=message):
            df.eval("@pd.thing")

    # 测试引号使用不正确的情况是否会引发异常
    def test_failing_quote(self, df):
        # 预期的错误消息模式，包含特定的错误信息
        msg = r"(Could not convert ).*( to a valid Python identifier.)"
        # 使用 pytest 的 pytest.raises 方法检查是否会抛出 SyntaxError 异常，并且错误消息匹配 msg
        with pytest.raises(SyntaxError, match=msg):
            df.query("`it's` > `that's`")

    # 测试变量名包含超出范围的字符时是否会引发异常
    def test_failing_character_outside_range(self, df):
        # 预期的错误消息模式，包含特定的错误信息
        msg = r"(Could not convert ).*( to a valid Python identifier.)"
        # 使用 pytest 的 pytest.raises 方法检查是否会抛出 SyntaxError 异常，并且错误消息匹配 msg
        with pytest.raises(Syntax
    # 定义测试函数，测试调用非命名表达式时的行为
    def test_call_non_named_expression(self, df):
        """
        Only attributes and variables ('named functions') can be called.
        .__call__() is not an allowed attribute because that would allow
        calling anything.
        https://github.com/pandas-dev/pandas/pull/32460
        """

        # 定义一个简单的函数 func，接受任意参数并返回 1
        def func(*_):
            return 1

        # 将 func 函数放入列表 funcs 中，用 noqa: F841 禁止未使用的变量警告
        funcs = [func]  # noqa: F841

        # 使用 df.eval() 执行表达式 "@func()"，调用名为 func 的函数
        df.eval("@func()")

        # 使用 pytest 的断言，预期会抛出 TypeError 异常，并匹配特定的错误信息
        with pytest.raises(TypeError, match="Only named functions are supported"):
            # 使用 df.eval() 调用 funcs 列表中第一个函数，预期抛出异常
            df.eval("@funcs[0]()")

        # 使用 pytest 的断言，预期会抛出 TypeError 异常，并匹配特定的错误信息
        with pytest.raises(TypeError, match="Only named functions are supported"):
            # 尝试使用 funcs 列表中第一个函数的 __call__() 方法，预期抛出异常
            df.eval("@funcs[0].__call__()")

    # 测试处理任意数值类型和 Arrow 类型的数据帧的情况
    def test_ea_dtypes(self, any_numeric_ea_and_arrow_dtype):
        # 创建一个 DataFrame 对象 df，包含两行两列的整数数据，列名为 "a" 和 "b"，指定数据类型为 any_numeric_ea_and_arrow_dtype
        df = DataFrame(
            [[1, 2], [3, 4]], columns=["a", "b"], dtype=any_numeric_ea_and_arrow_dtype
        )
        # 如果 NUMEXPR_INSTALLED 为真，设置警告为 RuntimeWarning，否则为 None
        warning = RuntimeWarning if NUMEXPR_INSTALLED else None
        # 使用 tm.assert_produces_warning 检查是否产生警告
        with tm.assert_produces_warning(warning):
            # 执行 df.eval() 计算表达式 "c = b - a"，将结果赋给 result
            result = df.eval("c = b - a")
        # 创建一个预期的 DataFrame 对象 expected，与 result 结构和数据类型一致
        expected = DataFrame(
            [[1, 2, 1], [3, 4, 1]],
            columns=["a", "b", "c"],
            dtype=any_numeric_ea_and_arrow_dtype,
        )
        # 使用 tm.assert_frame_equal 检查 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

    # 测试处理数值类型和标量操作的情况
    def test_ea_dtypes_and_scalar(self):
        # 创建一个 DataFrame 对象 df，包含两行两列的整数数据，列名为 "a" 和 "b"，数据类型为 "Float64"
        df = DataFrame([[1, 2], [3, 4]], columns=["a", "b"], dtype="Float64")
        # 如果 NUMEXPR_INSTALLED 为真，设置警告为 RuntimeWarning，否则为 None
        warning = RuntimeWarning if NUMEXPR_INSTALLED else None
        # 使用 tm.assert_produces_warning 检查是否产生警告
        with tm.assert_produces_warning(warning):
            # 执行 df.eval() 计算表达式 "c = b - 1"，将结果赋给 result
            result = df.eval("c = b - 1")
        # 创建一个预期的 DataFrame 对象 expected，与 result 结构和数据类型一致
        expected = DataFrame(
            [[1, 2, 1], [3, 4, 3]], columns=["a", "b", "c"], dtype="Float64"
        )
        # 使用 tm.assert_frame_equal 检查 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

    # 测试处理任意数值类型和 Arrow 类型的数据帧，以及标量操作的情况
    def test_ea_dtypes_and_scalar_operation(self, any_numeric_ea_and_arrow_dtype):
        # 创建一个 DataFrame 对象 df，包含两行两列的整数数据，列名为 "a" 和 "b"，指定数据类型为 any_numeric_ea_and_arrow_dtype
        df = DataFrame(
            [[1, 2], [3, 4]], columns=["a", "b"], dtype=any_numeric_ea_and_arrow_dtype
        )
        # 执行 df.eval() 计算表达式 "c = 2 - 1"，将结果赋给 result
        result = df.eval("c = 2 - 1")
        # 创建一个预期的 DataFrame 对象 expected，与 result 结构和数据类型一致
        expected = DataFrame(
            {
                "a": Series([1, 3], dtype=any_numeric_ea_and_arrow_dtype),
                "b": Series([2, 4], dtype=any_numeric_ea_and_arrow_dtype),
                "c": Series([1, 1], dtype=result["c"].dtype),
            }
        )
        # 使用 tm.assert_frame_equal 检查 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

    # 使用 pytest.mark.parametrize 对 dtype 参数进行参数化的测试
    @pytest.mark.parametrize("dtype", ["int64", "Int64", "int64[pyarrow]"])
    def test_query_ea_dtypes(self, dtype):
        # 如果 dtype 是 "int64[pyarrow]"，则检查是否可以导入 "pyarrow"，否则跳过测试
        if dtype == "int64[pyarrow]":
            pytest.importorskip("pyarrow")
        # 创建一个 DataFrame 对象 df，包含一列名为 "a" 的 Series，数据类型为 dtype 所指定的类型
        df = DataFrame({"a": Series([1, 2], dtype=dtype)})
        # 创建一个集合 ref 包含元素 2，用 noqa: F841 禁止未使用的变量警告
        ref = {2}  # noqa: F841
        # 如果 dtype 是 "Int64" 并且 NUMEXPR_INSTALLED 为真，则设置警告为 RuntimeWarning，否则为 None
        warning = RuntimeWarning if dtype == "Int64" and NUMEXPR_INSTALLED else None
        # 使用 tm.assert_produces_warning 检查是否产生警告
        with tm.assert_produces_warning(warning):
            # 执行 df.query() 查询表达式 "a in @ref"，将结果赋给 result
            result = df.query("a in @ref")
        # 创建一个预期的 DataFrame 对象 expected，与 result 结构和数据类型一致
        expected = DataFrame({"a": Series([2], dtype=dtype, index=[1])})
        # 使用 tm.assert_frame_equal 检查 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

    # 使用 pytest.mark.parametrize 对 engine 参数进行参数化的测试
    @pytest.mark.parametrize("engine", ["python", "numexpr"])
    # 使用 pytest 的参数化装饰器，定义测试函数的多个参数化方式
    @pytest.mark.parametrize("dtype", ["int64", "Int64", "int64[pyarrow]"])
    # 定义测试方法，用于测试查询操作的相等性比较
    def test_query_ea_equality_comparison(self, dtype, engine):
        # 设置警告类型为 RuntimeWarning（如果引擎是 numexpr），否则为 None
        warning = RuntimeWarning if engine == "numexpr" else None
        # 如果引擎为 numexpr 但未安装 numexpr，跳过测试
        if engine == "numexpr" and not NUMEXPR_INSTALLED:
            pytest.skip("numexpr not installed")
        # 如果 dtype 是 "int64[pyarrow]"，确保导入 pyarrow 库
        if dtype == "int64[pyarrow]":
            pytest.importorskip("pyarrow")
        # 创建包含两列的 DataFrame 对象 df，其中一列是 Nullable 整数，另一列根据参数化的 dtype 决定
        df = DataFrame(
            {"A": Series([1, 1, 2], dtype="Int64"), "B": Series([1, 2, 2], dtype=dtype)}
        )
        # 使用 pytest 的 assert_produces_warning 上下文管理器，检查是否产生指定类型的警告
        with tm.assert_produces_warning(warning):
            # 执行查询操作，比较列 A 和列 B 的值是否相等，使用指定的引擎
            result = df.query("A == B", engine=engine)
        # 创建预期结果的 DataFrame 对象 expected，与 df 的结构相同，但值可能不同
        expected = DataFrame(
            {
                "A": Series([1, 2], dtype="Int64", index=[0, 2]),
                "B": Series([1, 2], dtype=dtype, index=[0, 2]),
            }
        )
        # 使用 pytest 的 assert_frame_equal 断言函数，比较 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)

    # 定义测试方法，用于测试对象列中是否包含所有的 NaT（Not a Time）
    def test_all_nat_in_object(self):
        # 生成当前时间的 Timestamp 对象 now，设定为 UTC 时区（不使用的警告 F841 可以忽略）
        now = pd.Timestamp.now("UTC")  # noqa: F841
        # 创建包含一列 a 的 DataFrame 对象 df，该列使用 pd.to_datetime 转换为日期时间对象，类型为 object
        df = DataFrame({"a": pd.to_datetime([None, None], utc=True)}, dtype=object)
        # 执行查询操作，找出列 a 中大于 @now 时间的值
        result = df.query("a > @now")
        # 创建预期结果的 DataFrame 对象 expected，该对象只包含一列 a，且内容为空列表
        expected = DataFrame({"a": []}, dtype=object)
        # 使用 pytest 的 assert_frame_equal 断言函数，比较 result 和 expected 是否相等
        tm.assert_frame_equal(result, expected)
```