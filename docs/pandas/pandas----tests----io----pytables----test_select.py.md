# `D:\src\scipysrc\pandas\pandas\tests\io\pytables\test_select.py`

```
# 导入所需的库
import numpy as np
import pytest

# 导入 pandas 库中的特定模块和类
from pandas._libs.tslibs import Timestamp
from pandas.compat import PY312

# 导入 pandas 库，并选择性地导入特定的函数和类
import pandas as pd
from pandas import (
    DataFrame,
    HDFStore,
    Index,
    MultiIndex,
    Series,
    _testing as tm,
    bdate_range,
    concat,
    date_range,
    isna,
    read_hdf,
)
# 导入 pandas 测试用例中的工具函数
from pandas.tests.io.pytables.common import (
    _maybe_remove,
    ensure_clean_store,
)

# 导入 pandas 库中的 Term 类
from pandas.io.pytables import Term

# 设置 pytest 的标记，指定单 CPU 运行
pytestmark = pytest.mark.single_cpu


def test_select_columns_in_where(setup_path):
    # GH 6169
    # 当在 `where` 参数中传递 columns 时，重新创建多级索引

    # 创建一个多级索引
    index = MultiIndex(
        levels=[["foo", "bar", "baz", "qux"], ["one", "two", "three"]],
        codes=[[0, 0, 0, 1, 1, 2, 2, 3, 3, 3], [0, 1, 2, 0, 1, 1, 2, 0, 1, 2]],
        names=["foo_name", "bar_name"],
    )

    # 使用 DataFrame 创建一个数据框
    df = DataFrame(
        np.random.default_rng(2).standard_normal((10, 3)),
        index=index,
        columns=["A", "B", "C"],
    )

    # 在一个干净的存储路径下，确保存储环境清洁
    with ensure_clean_store(setup_path) as store:
        # 将 DataFrame 存储到 HDF5 文件中，格式为表格格式
        store.put("df", df, format="table")
        # 期望的结果是 df 中的 "A" 列
        expected = df[["A"]]

        # 使用测试工具函数验证存储中选择的结果与期望的结果相等
        tm.assert_frame_equal(store.select("df", columns=["A"]), expected)

        # 使用测试工具函数验证存储中选择的结果与期望的结果相等，
        # 这里使用了字符串表达式指定要选择的列 "A"
        tm.assert_frame_equal(store.select("df", where="columns=['A']"), expected)

    # 使用 Series 创建一个序列
    s = Series(np.random.default_rng(2).standard_normal(10), index=index, name="A")
    with ensure_clean_store(setup_path) as store:
        # 将 Series 存储到 HDF5 文件中，格式为表格格式
        store.put("s", s, format="table")
        # 使用测试工具函数验证存储中选择的结果与期望的结果相等，
        # 这里使用了字符串表达式指定要选择的列 "A"
        tm.assert_series_equal(store.select("s", where="columns=['A']"), s)


def test_select_with_dups(setup_path):
    # 创建一个包含重复列名的 DataFrame，所有列具有相同的数据类型
    df = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)), columns=["A", "A", "B", "B"]
    )
    # 设置索引为日期时间索引
    df.index = date_range("20130101 9:30", periods=10, freq="min")

    with ensure_clean_store(setup_path) as store:
        # 将 DataFrame 附加到 HDF5 存储中
        store.append("df", df)

        # 从存储中选择数据框，并与期望的结果进行比较，通过块进行比较
        result = store.select("df")
        expected = df
        tm.assert_frame_equal(result, expected, by_blocks=True)

        # 从存储中选择数据框的特定列，并与期望的结果进行比较，通过块进行比较
        result = store.select("df", columns=df.columns)
        expected = df
        tm.assert_frame_equal(result, expected, by_blocks=True)

        # 从存储中选择数据框的特定列，并与期望的结果进行比较
        result = store.select("df", columns=["A"])
        expected = df.loc[:, ["A"]]
        tm.assert_frame_equal(result, expected)

    # 创建一个包含不同数据类型的 DataFrame，其中包含重复列名
    df = concat(
        [
            DataFrame(
                np.random.default_rng(2).standard_normal((10, 4)),
                columns=["A", "A", "B", "B"],
            ),
            DataFrame(
                np.random.default_rng(2).integers(0, 10, size=20).reshape(10, 2),
                columns=["A", "C"],
            ),
        ],
        axis=1,
    )
    # 设置索引为日期时间索引
    df.index = date_range("20130101 9:30", periods=10, freq="min")
    # 使用 ensure_clean_store 上下文管理器来确保存储路径 setup_path 是干净的
    with ensure_clean_store(setup_path) as store:
        # 将 DataFrame df 附加到存储器中，命名为 "df"
        store.append("df", df)
    
        # 从存储器中选择命名为 "df" 的数据，将结果赋给 result
        result = store.select("df")
        # 期望的结果是 DataFrame df
        expected = df
        # 使用 by_blocks=True 比较两个 DataFrame 是否相等
        tm.assert_frame_equal(result, expected, by_blocks=True)
    
        # 从存储器中选择命名为 "df" 的数据，并只选择指定列（df.columns）
        result = store.select("df", columns=df.columns)
        # 期望的结果是 DataFrame df
        expected = df
        # 使用 by_blocks=True 比较两个 DataFrame 是否相等
        tm.assert_frame_equal(result, expected, by_blocks=True)
    
        # 期望的结果是 DataFrame df 中的 "A" 列数据
        expected = df.loc[:, ["A"]]
        # 从存储器中选择命名为 "df" 的数据，并只选择指定列 ["A"]
        result = store.select("df", columns=["A"])
        # 使用 by_blocks=True 比较两个 DataFrame 是否相等
        tm.assert_frame_equal(result, expected, by_blocks=True)
    
        # 期望的结果是 DataFrame df 中的 "B" 和 "A" 列数据
        expected = df.loc[:, ["B", "A"]]
        # 从存储器中选择命名为 "df" 的数据，并只选择指定列 ["B", "A"]
        result = store.select("df", columns=["B", "A"])
        # 使用 by_blocks=True 比较两个 DataFrame 是否相等
        tm.assert_frame_equal(result, expected, by_blocks=True)
    
    # 在索引和列上都有重复的情况下
    # 使用 ensure_clean_store 上下文管理器来确保存储路径 setup_path 是干净的
    with ensure_clean_store(setup_path) as store:
        # 将 DataFrame df 两次附加到存储器中，命名为 "df"
        store.append("df", df)
        store.append("df", df)
    
        # 期望的结果是 DataFrame df 中的 "B" 和 "A" 列数据
        expected = df.loc[:, ["B", "A"]]
        # 将期望的结果两次连接起来
        expected = concat([expected, expected])
        # 从存储器中选择命名为 "df" 的数据，并只选择指定列 ["B", "A"]
        result = store.select("df", columns=["B", "A"])
        # 使用 by_blocks=True 比较两个 DataFrame 是否相等
        tm.assert_frame_equal(result, expected, by_blocks=True)
# 定义名为 test_select 的测试函数，接受一个 setup_path 参数
def test_select(setup_path):
    # 使用 ensure_clean_store 上下文管理器，确保 store 被清理
    with ensure_clean_store(setup_path) as store:
        # 创建一个 DataFrame 对象 df，包含随机生成的数据
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )
        # 可能删除 store 中的 "df" 键对应的数据
        _maybe_remove(store, "df")
        # 将 DataFrame df 添加到 store 中，键为 "df"
        store.append("df", df)
        # 从 store 中选择 "df" 键对应的数据，仅保留 "A" 和 "B" 列
        result = store.select("df", columns=["A", "B"])
        # 创建期望的 DataFrame 对象 expected，仅保留 "A" 和 "B" 列
        expected = df.reindex(columns=["A", "B"])
        # 使用 assert_frame_equal 检查 result 和 expected 是否相等
        tm.assert_frame_equal(expected, result)

        # 等效的方式
        # 重新选择 "df" 键对应的数据，使用字符串指定需要的列
        result = store.select("df", [("columns=['A', 'B']")])
        # 创建期望的 DataFrame 对象 expected，仅保留 "A" 和 "B" 列
        expected = df.reindex(columns=["A", "B"])
        # 使用 assert_frame_equal 检查 result 和 expected 是否相等
        tm.assert_frame_equal(expected, result)

        # 将 DataFrame df 添加到 store 中，并指定 "A" 列作为数据列
        store.append("df", df, data_columns=["A"])
        # 从 store 中选择 "df" 键对应的数据，仅保留 "A" 和 "B" 列，并满足 "A > 0" 的条件
        result = store.select("df", ["A > 0"], columns=["A", "B"])
        # 创建期望的 DataFrame 对象 expected，仅保留 "A" 和 "B" 列，并满足 "A > 0" 的条件
        expected = df[df.A > 0].reindex(columns=["A", "B"])
        # 使用 assert_frame_equal 检查 result 和 expected 是否相等
        tm.assert_frame_equal(expected, result)

        # 将 DataFrame df 添加到 store 中，并将所有列指定为数据列
        store.append("df", df, data_columns=True)
        # 从 store 中选择 "df" 键对应的数据，仅保留 "A" 和 "B" 列，并满足 "A > 0" 的条件
        result = store.select("df", ["A > 0"], columns=["A", "B"])
        # 创建期望的 DataFrame 对象 expected，仅保留 "A" 和 "B" 列，并满足 "A > 0" 的条件
        expected = df[df.A > 0].reindex(columns=["A", "B"])
        # 使用 assert_frame_equal 检查 result 和 expected 是否相等
        tm.assert_frame_equal(expected, result)

        # 将 DataFrame df 添加到 store 中，并指定 "A" 列作为数据列
        store.append("df", df, data_columns=["A"])
        # 从 store 中选择 "df" 键对应的数据，仅保留 "C" 和 "D" 列，并满足 "A > 0" 的条件
        result = store.select("df", ["A > 0"], columns=["C", "D"])
        # 创建期望的 DataFrame 对象 expected，仅保留 "C" 和 "D" 列，并满足 "A > 0" 的条件
        expected = df[df.A > 0].reindex(columns=["C", "D"])
        # 使用 assert_frame_equal 检查 result 和 expected 是否相等
        tm.assert_frame_equal(expected, result)
    # 使用 ensure_clean_store 函数确保在 setup_path 下创建一个干净的存储对象，并通过 with 上下文管理器使用它
    with ensure_clean_store(setup_path) as store:
        # 创建一个包含时间戳数据列的 DataFrame，时间范围从 '2012-01-01' 开始，持续 300 个时间段
        df = DataFrame(
            {
                "ts": bdate_range("2012-01-01", periods=300),
                "A": np.random.default_rng(2).standard_normal(300),
            }
        )
        # 如果存储中存在名为 "df" 的数据集合，则尝试删除它
        _maybe_remove(store, "df")
        # 将 DataFrame df 添加到 store 中，指定数据列为 ["ts", "A"]
        store.append("df", df, data_columns=["ts", "A"])

        # 从 store 中选择 "df" 数据集中时间戳大于等于 '2012-02-01' 的数据
        result = store.select("df", "ts>=Timestamp('2012-02-01')")
        # 创建预期的 DataFrame，选择 df 中时间戳大于等于 '2012-02-01' 的行
        expected = df[df.ts >= Timestamp("2012-02-01")]
        # 使用 tm.assert_frame_equal 函数比较预期结果和实际结果
        tm.assert_frame_equal(expected, result)

        # 创建一个包含两列随机标准正态分布数据的 DataFrame
        df = DataFrame(
            np.random.default_rng(2).standard_normal((5, 2)), columns=["A", "B"]
        )
        # 在 df 中添加一个名为 "object" 的列，大部分为 "foo"，部分行为 "bar"
        df["object"] = "foo"
        df.loc[4:5, "object"] = "bar"
        # 根据列 "A" 的值大于 0 创建一个布尔列 "boolv"
        df["boolv"] = df["A"] > 0
        # 如果存储中存在名为 "df" 的数据集合，则尝试删除它
        _maybe_remove(store, "df")
        # 将 DataFrame df 添加到 store 中，并自动识别数据列
        store.append("df", df, data_columns=True)

        # 创建预期结果，选择 df 中 boolv 列为 True 的行，并重新索引为 ["A", "boolv"] 列
        expected = df[df.boolv == True].reindex(columns=["A", "boolv"])  # noqa: E712
        # 遍历 True 的不同表示形式，从 store 中选择 boolv 列为该值的数据，并比较预期结果
        for v in [True, "true", 1]:
            result = store.select("df", f"boolv == {v}", columns=["A", "boolv"])
            tm.assert_frame_equal(expected, result)

        # 创建预期结果，选择 df 中 boolv 列为 False 的行，并重新索引为 ["A", "boolv"] 列
        expected = df[df.boolv == False].reindex(columns=["A", "boolv"])  # noqa: E712
        # 遍历 False 的不同表示形式，从 store 中选择 boolv 列为该值的数据，并比较预期结果
        for v in [False, "false", 0]:
            result = store.select("df", f"boolv == {v}", columns=["A", "boolv"])
            tm.assert_frame_equal(expected, result)

        # 创建一个包含两列随机数的 DataFrame，使用默认整数索引
        df = DataFrame(
            {
                "A": np.random.default_rng(2).random(20),
                "B": np.random.default_rng(2).random(20),
            }
        )
        # 如果存储中存在名为 "df_int" 的数据集合，则尝试删除它
        _maybe_remove(store, "df_int")
        # 将 DataFrame df 添加到 store 中
        store.append("df_int", df)
        # 从 store 中选择 "df_int" 数据集中索引小于 10 的数据，并选择列 "A"
        result = store.select("df_int", "index<10 and columns=['A']")
        # 创建预期结果，选择 df 中索引在前 10 行的数据，并选择列 "A"
        expected = df.reindex(index=list(df.index)[0:10], columns=["A"])
        # 使用 tm.assert_frame_equal 函数比较预期结果和实际结果
        tm.assert_frame_equal(expected, result)

        # 创建一个包含三列随机数的 DataFrame，其中包括浮点索引列 "index"
        df = DataFrame(
            {
                "A": np.random.default_rng(2).random(20),
                "B": np.random.default_rng(2).random(20),
                "index": np.arange(20, dtype="f8"),
            }
        )
        # 如果存储中存在名为 "df_float" 的数据集合，则尝试删除它
        _maybe_remove(store, "df_float")
        # 将 DataFrame df 添加到 store 中
        store.append("df_float", df)
        # 从 store 中选择 "df_float" 数据集中索引小于 10.0 的数据，并选择列 "A"
        result = store.select("df_float", "index<10.0 and columns=['A']")
        # 创建预期结果，选择 df 中索引在前 10 行的数据，并选择列 "A"
        expected = df.reindex(index=list(df.index)[0:10], columns=["A"])
        # 使用 tm.assert_frame_equal 函数比较预期结果和实际结果
        tm.assert_frame_equal(expected, result)
    # 使用 ensure_clean_store 函数确保在指定路径 setup_path 下的存储环境干净
    with ensure_clean_store(setup_path) as store:
        # 创建一个包含浮点数列和整数列的 DataFrame 对象 df
        # "cols" 列包含整数 0 到 10，"values" 列包含浮点数 0.0 到 10.0
        df = DataFrame({"cols": range(11), "values": range(11)}, dtype="float64")
        # 将 "cols" 列中的整数值加上 10，并转换为字符串
        df["cols"] = (df["cols"] + 10).apply(str)

        # 将 DataFrame df 存储到 store 中，以名称 "df1"，并启用数据列索引
        store.append("df1", df, data_columns=True)
        # 从 store 中选择名为 "df1" 的数据，其中 "values" 列的值大于 2.0
        result = store.select("df1", where="values>2.0")
        # 从原始 DataFrame df 中选取 "values" 列中大于 2.0 的行，作为预期结果
        expected = df[df["values"] > 2.0]
        # 断言从 store 中选择的结果与预期结果相等
        tm.assert_frame_equal(expected, result)

        # 在 DataFrame df 的第一行设置为 NaN
        df.iloc[0] = np.nan
        # 更新预期结果为 DataFrame df 中 "values" 列大于 2.0 的行
        expected = df[df["values"] > 2.0]

        # 将更新后的 DataFrame df 存储到 store 中，名称为 "df2"，禁用索引
        store.append("df2", df, data_columns=True, index=False)
        # 从 store 中选择名为 "df2" 的数据，其中 "values" 列的值大于 2.0
        result = store.select("df2", where="values>2.0")
        # 断言从 store 中选择的结果与预期结果相等
        tm.assert_frame_equal(expected, result)

        # 以下代码段是关于 PyTables/PyTables 问题 #282 的注释和示例代码，
        # 由于问题未解决，这部分代码被注释掉

        # 在 DataFrame df 的第二行设置为 NaN
        df.iloc[1] = np.nan
        # 更新预期结果为 DataFrame df 中 "values" 列大于 2.0 的行
        expected = df[df["values"] > 2.0]

        # 将更新后的 DataFrame df 存储到 store 中，名称为 "df4"，启用数据列索引
        store.append("df4", df, data_columns=True)
        # 从 store 中选择名为 "df4" 的数据，其中 "values" 列的值大于 2.0
        result = store.select("df4", where="values>2.0")
        # 断言从 store 中选择的结果与预期结果相等
        tm.assert_frame_equal(expected, result)

    # 使用 ensure_clean_store 函数确保在指定路径 setup_path 下的存储环境干净
    # 这部分代码段是关于 GH 11283 的测试用例
    with ensure_clean_store(setup_path) as store:
        # 创建一个包含特定数据的 DataFrame df
        # 包括四列数据，命名为 'A', 'B', 'C', 'D'，并设置自定义索引
        df = DataFrame(
            1.1 * np.arange(120).reshape((30, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=Index([f"i-{i}" for i in range(30)], dtype=object),
        )

        # 更新预期结果为 DataFrame df 中 'A' 列大于 0 的行
        expected = df[df["A"] > 0]

        # 将 DataFrame df 存储到 store 中，名称为 "df"，并启用数据列索引
        store.append("df", df, data_columns=True)

        # 应用 pytest.mark.xfail 标记来预期 PY312 中的 AST 改变引发 ValueError 异常
        request.applymarker(
            pytest.mark.xfail(
                PY312,
                reason="AST change in PY312",
                raises=ValueError,
            )
        )

        # 定义一个 numpy 标量 np_zero，并使用它进行比较选择
        np_zero = np.float64(0)  # noqa: F841
        # 从 store 中选择名为 "df" 的数据，其中 'A' 列的值大于 np_zero
        result = store.select("df", where=["A>np_zero"])
        # 断言从 store 中选择的结果与预期结果相等
        tm.assert_frame_equal(expected, result)
# 使用给定的路径设置来测试数据存储中的选择操作
def test_select_with_many_inputs(setup_path):
    # 确保在每次运行测试前，存储都是空的
    with ensure_clean_store(setup_path) as store:
        # 创建一个包含时间序列、随机数据、序号和用户标识的DataFrame
        df = DataFrame(
            {
                "ts": bdate_range("2012-01-01", periods=300),
                "A": np.random.default_rng(2).standard_normal(300),
                "B": range(300),
                "users": ["a"] * 50
                + ["b"] * 50
                + ["c"] * 100
                + [f"a{i:03d}" for i in range(100)],
            }
        )
        # 如果存在名为'df'的数据表，先尝试移除它
        _maybe_remove(store, "df")
        # 将DataFrame添加到存储中，指定数据列为'ts'、'A'、'B'和'users'
        store.append("df", df, data_columns=["ts", "A", "B", "users"])

        # 普通选择操作，选择时间戳大于等于'2012-02-01'的数据
        result = store.select("df", "ts>=Timestamp('2012-02-01')")
        expected = df[df.ts >= Timestamp("2012-02-01")]
        tm.assert_frame_equal(expected, result)

        # 使用小选择器，选择时间戳大于等于'2012-02-01'且用户为'a'、'b'或'c'的数据
        result = store.select("df", "ts>=Timestamp('2012-02-01') & users=['a','b','c']")
        expected = df[
            (df.ts >= Timestamp("2012-02-01")) & df.users.isin(["a", "b", "c"])
        ]
        tm.assert_frame_equal(expected, result)

        # 使用大选择器选择器，选择时间戳大于等于'2012-02-01'且用户在指定列表中的数据
        selector = ["a", "b", "c"] + [f"a{i:03d}" for i in range(60)]
        result = store.select("df", "ts>=Timestamp('2012-02-01') and users=selector")
        expected = df[(df.ts >= Timestamp("2012-02-01")) & df.users.isin(selector)]
        tm.assert_frame_equal(expected, result)

        # 使用序号选择器，选择序号在指定范围内的数据
        selector = range(100, 200)
        result = store.select("df", "B=selector")
        expected = df[df.B.isin(selector)]
        tm.assert_frame_equal(expected, result)
        assert len(result) == 100

        # 使用索引选择器，选择索引在给定序列中的数据
        selector = Index(df.ts[0:100].values)
        result = store.select("df", "ts=selector")
        expected = df[df.ts.isin(selector.values)]
        tm.assert_frame_equal(expected, result)
        assert len(result) == 100


# 测试迭代器模式下的数据选择
def test_select_iterator(tmp_path, setup_path):
    # 单表模式
    with ensure_clean_store(setup_path) as store:
        # 创建一个包含随机数据的DataFrame，并将其添加到存储中
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )
        _maybe_remove(store, "df")
        store.append("df", df)

        # 从存储中选择整个表格
        expected = store.select("df")

        # 使用迭代器模式选择数据，将结果连接成一个DataFrame
        results = list(store.select("df", iterator=True))
        result = concat(results)
        tm.assert_frame_equal(expected, result)

        # 使用分块大小为2的模式选择数据，并确保结果正确连接
        results = list(store.select("df", chunksize=2))
        assert len(results) == 5
        result = concat(results)
        tm.assert_frame_equal(expected, result)

        # 再次使用分块大小为2的模式选择数据，确保结果与预期一致
        results = list(store.select("df", chunksize=2))
        result = concat(results)
        tm.assert_frame_equal(result, expected)

    # 使用临时路径设置和给定的路径设置创建一个DataFrame
    path = tmp_path / setup_path

    df = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=date_range("2000-01-01", periods=10, freq="B"),
    )
    df.to_hdf(path, key="df_non_table")
    # 将 DataFrame 保存为 HDF 文件，使用 "df_non_table" 作为键名

    msg = "can only use an iterator or chunksize on a table"
    # 定义错误消息，用于检测是否引发特定类型的异常

    with pytest.raises(TypeError, match=msg):
        # 使用 pytest 检测是否引发指定类型的异常，并匹配错误消息
        read_hdf(path, "df_non_table", chunksize=2)
        # 调用 read_hdf 函数，尝试以 chunksize=2 的方式读取 "df_non_table"

    with pytest.raises(TypeError, match=msg):
        # 使用 pytest 检测是否引发指定类型的异常，并匹配错误消息
        read_hdf(path, "df_non_table", iterator=True)
        # 调用 read_hdf 函数，尝试以 iterator=True 的方式读取 "df_non_table"

    path = tmp_path / setup_path
    # 设置路径为临时路径 tmp_path 与 setup_path 的组合

    df = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=date_range("2000-01-01", periods=10, freq="B"),
    )
    # 创建一个 DataFrame，包含随机数据，列为 ["A", "B", "C", "D"]，行索引为工作日频率的日期范围

    df.to_hdf(path, key="df", format="table")
    # 将 DataFrame 以表格格式保存到 HDF 文件，使用 "df" 作为键名

    results = list(read_hdf(path, "df", chunksize=2))
    # 使用 read_hdf 函数以 chunksize=2 的方式读取 HDF 文件中的 "df" 数据，返回结果列表
    result = concat(results)
    # 将结果列表中的数据拼接成一个 DataFrame

    assert len(results) == 5
    # 断言结果列表的长度为 5
    tm.assert_frame_equal(result, df)
    # 使用 tm.assert_frame_equal 断言拼接后的结果与原始 DataFrame df 相等
    tm.assert_frame_equal(result, read_hdf(path, "df"))
    # 使用 tm.assert_frame_equal 断言拼接后的结果与再次读取的 HDF 数据相等

    # multiple

    with ensure_clean_store(setup_path) as store:
        # 使用 ensure_clean_store 函数确保存储的干净状态，并使用 setup_path 作为参数命名存储

        df1 = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )
        # 创建一个 DataFrame df1，包含随机数据，列为 ["A", "B", "C", "D"]，行索引为工作日频率的日期范围

        store.append("df1", df1, data_columns=True)
        # 将 df1 追加到存储中，使用 "df1" 作为键名，并启用数据列特性

        df2 = df1.copy().rename(columns="{}_2".format)
        # 复制 df1 并重命名列名为 "{}_2" 的格式化字符串

        df2["foo"] = "bar"
        # 添加新列 "foo" 并赋值为 "bar"

        store.append("df2", df2)
        # 将 df2 追加到存储中，使用 "df2" 作为键名

        df = concat([df1, df2], axis=1)
        # 沿着列轴拼接 df1 和 df2，形成一个新的 DataFrame df

        # full selection
        expected = store.select_as_multiple(["df1", "df2"], selector="df1")
        # 从存储中选择 "df1" 和 "df2" 的数据，selector 参数指定选择 "df1" 的数据

        results = list(
            store.select_as_multiple(["df1", "df2"], selector="df1", chunksize=2)
        )
        # 使用 chunksize=2 的方式从存储中选择 "df1" 的数据，并返回结果列表
        result = concat(results)
        # 将结果列表中的数据拼接成一个 DataFrame

        tm.assert_frame_equal(expected, result)
        # 使用 tm.assert_frame_equal 断言预期结果与拼接后的结果相等
# 定义测试函数，测试选择迭代器完整性功能，以解决 GitHub 问题 8014
def test_select_iterator_complete_8014(setup_path):
    # GH 8014
    # 使用迭代器和where子句
    chunksize = 1e4  # 设置块大小为1万条记录

    # 不使用迭代器
    with ensure_clean_store(setup_path) as store:
        # 创建预期的DataFrame对象，包含100064行、4列的标准正态分布随机数
        expected = DataFrame(
            np.random.default_rng(2).standard_normal((100064, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=100064, freq="s"),
        )
        _maybe_remove(store, "df")  # 如果存在"df"，则尝试删除
        store.append("df", expected)  # 将预期数据追加到存储中的"df"表

        beg_dt = expected.index[0]  # 获取数据框的第一个日期时间索引
        end_dt = expected.index[-1]  # 获取数据框的最后一个日期时间索引

        # 选择不使用迭代器和where子句的情况，验证其正常工作
        result = store.select("df")
        tm.assert_frame_equal(expected, result)  # 断言预期结果与实际结果一致

        # 选择不使用迭代器和where子句的情况，单个范围条件（起始日期时间），验证其正常工作
        where = f"index >= '{beg_dt}'"
        result = store.select("df", where=where)
        tm.assert_frame_equal(expected, result)  # 断言预期结果与实际结果一致

        # 选择不使用迭代器和where子句的情况，单个范围条件（结束日期时间），验证其正常工作
        where = f"index <= '{end_dt}'"
        result = store.select("df", where=where)
        tm.assert_frame_equal(expected, result)  # 断言预期结果与实际结果一致

        # 选择不使用迭代器和where子句的情况，包含的范围条件，验证其正常工作
        where = f"index >= '{beg_dt}' & index <= '{end_dt}'"
        result = store.select("df", where=where)
        tm.assert_frame_equal(expected, result)  # 断言预期结果与实际结果一致

    # 使用迭代器，并选择全部范围
    with ensure_clean_store(setup_path) as store:
        # 创建预期的DataFrame对象，包含100064行、4列的标准正态分布随机数
        expected = DataFrame(
            np.random.default_rng(2).standard_normal((100064, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=100064, freq="s"),
        )
        _maybe_remove(store, "df")  # 如果存在"df"，则尝试删除
        store.append("df", expected)  # 将预期数据追加到存储中的"df"表

        beg_dt = expected.index[0]  # 获取数据框的第一个日期时间索引
        end_dt = expected.index[-1]  # 获取数据框的最后一个日期时间索引

        # 选择使用迭代器和不使用where子句的情况，验证其正常工作
        results = list(store.select("df", chunksize=chunksize))
        result = concat(results)
        tm.assert_frame_equal(expected, result)  # 断言预期结果与实际结果一致

        # 选择使用迭代器和单个范围条件（起始日期时间），验证其正常工作
        where = f"index >= '{beg_dt}'"
        results = list(store.select("df", where=where, chunksize=chunksize))
        result = concat(results)
        tm.assert_frame_equal(expected, result)  # 断言预期结果与实际结果一致

        # 选择使用迭代器和单个范围条件（结束日期时间），验证其正常工作
        where = f"index <= '{end_dt}'"
        results = list(store.select("df", where=where, chunksize=chunksize))
        result = concat(results)
        tm.assert_frame_equal(expected, result)  # 断言预期结果与实际结果一致

        # 选择使用迭代器和包含范围条件，验证其正常工作
        where = f"index >= '{beg_dt}' & index <= '{end_dt}'"
        results = list(store.select("df", where=where, chunksize=chunksize))
        result = concat(results)
        tm.assert_frame_equal(expected, result)  # 断言预期结果与实际结果一致


def test_select_iterator_non_complete_8014(setup_path):
    # GH 8014
    # 测试未完成的选择迭代器功能，这部分的代码待补充
    # 使用迭代器和 where 子句，定义块大小为 10000
    chunksize = 1e4

    # 使用 ensure_clean_store 上下文管理器，确保存储路径 setup_path 清洁
    with ensure_clean_store(setup_path) as store:
        # 创建一个预期的 DataFrame，包含标准正态分布的随机数据，4列，100064行
        expected = DataFrame(
            np.random.default_rng(2).standard_normal((100064, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=100064, freq="s"),
        )
        # 如果存在名称为 "df" 的数据表，可能会移除它
        _maybe_remove(store, "df")
        # 将预期的 DataFrame 写入存储中，表名称为 "df"
        store.append("df", expected)

        # 设置开始时间为预期 DataFrame 的第一个索引时间
        beg_dt = expected.index[1]
        # 设置结束时间为预期 DataFrame 的倒数第二个索引时间
        end_dt = expected.index[-2]

        # 使用迭代器和 where 子句选择数据，条件为索引大于等于 beg_dt 的数据块，块大小为 chunksize
        where = f"index >= '{beg_dt}'"
        results = list(store.select("df", where=where, chunksize=chunksize))
        # 合并结果为一个 DataFrame
        result = concat(results)
        # 从预期的 DataFrame 中选择索引大于等于 beg_dt 的数据块
        rexpected = expected[expected.index >= beg_dt]
        # 断言合并的结果与预期结果相等
        tm.assert_frame_equal(rexpected, result)

        # 使用迭代器和 where 子句选择数据，条件为索引小于等于 end_dt 的数据块，块大小为 chunksize
        where = f"index <= '{end_dt}'"
        results = list(store.select("df", where=where, chunksize=chunksize))
        # 合并结果为一个 DataFrame
        result = concat(results)
        # 从预期的 DataFrame 中选择索引小于等于 end_dt 的数据块
        rexpected = expected[expected.index <= end_dt]
        # 断言合并的结果与预期结果相等
        tm.assert_frame_equal(rexpected, result)

        # 使用迭代器和 where 子句选择数据，条件为索引在 beg_dt 和 end_dt 之间的数据块，块大小为 chunksize
        where = f"index >= '{beg_dt}' & index <= '{end_dt}'"
        results = list(store.select("df", where=where, chunksize=chunksize))
        # 合并结果为一个 DataFrame
        result = concat(results)
        # 从预期的 DataFrame 中选择索引在 beg_dt 和 end_dt 之间的数据块
        rexpected = expected[(expected.index >= beg_dt) & (expected.index <= end_dt)]
        # 断言合并的结果与预期结果相等
        tm.assert_frame_equal(rexpected, result)

    # 使用迭代器和空 where 子句选择数据，预期结果长度为 0
    with ensure_clean_store(setup_path) as store:
        # 创建一个预期的 DataFrame，包含标准正态分布的随机数据，4列，100064行
        expected = DataFrame(
            np.random.default_rng(2).standard_normal((100064, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=100064, freq="s"),
        )
        # 如果存在名称为 "df" 的数据表，可能会移除它
        _maybe_remove(store, "df")
        # 将预期的 DataFrame 写入存储中，表名称为 "df"
        store.append("df", expected)

        # 设置结束时间为预期 DataFrame 的最后一个索引时间
        end_dt = expected.index[-1]

        # 使用迭代器和 where 子句选择数据，条件为索引大于 end_dt 的数据块，块大小为 chunksize
        where = f"index > '{end_dt}'"
        results = list(store.select("df", where=where, chunksize=chunksize))
        # 断言结果列表长度为 0
        assert 0 == len(results)
# 定义一个测试函数，用于测试使用迭代器和条件语句可以返回多个空数据帧的情况
def test_select_iterator_many_empty_frames(setup_path):
    # GH 8014
    # 使用迭代器和 where 子句可能返回许多空数据帧。
    chunksize = 10_000

    # 使用 ensure_clean_store 函数创建一个上下文管理器，确保存储环境干净
    with ensure_clean_store(setup_path) as store:
        # 创建一个预期的 DataFrame 对象，包含随机生成的数据，形状为 (100064, 4)
        expected = DataFrame(
            np.random.default_rng(2).standard_normal((100064, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=100064, freq="s"),
        )
        # 确保在存储中移除 "df"，并将新的 DataFrame 对象添加到存储中
        _maybe_remove(store, "df")
        store.append("df", expected)

        # 获取预期 DataFrame 的起始和结束时间
        beg_dt = expected.index[0]
        end_dt = expected.index[chunksize - 1]

        # 使用迭代器和 where 子句进行选择，单个条件，选择范围的起始部分
        where = f"index >= '{beg_dt}'"
        results = list(store.select("df", where=where, chunksize=chunksize))
        result = concat(results)
        rexpected = expected[expected.index >= beg_dt]
        tm.assert_frame_equal(rexpected, result)

        # 使用迭代器和 where 子句进行选择，单个条件，选择范围的结束部分
        where = f"index <= '{end_dt}'"
        results = list(store.select("df", where=where, chunksize=chunksize))

        # 确保结果的数量为 1
        assert len(results) == 1
        result = concat(results)
        rexpected = expected[expected.index <= end_dt]
        tm.assert_frame_equal(rexpected, result)

        # 使用迭代器和 where 子句进行选择，包含范围的条件
        where = f"index >= '{beg_dt}' & index <= '{end_dt}'"
        results = list(store.select("df", where=where, chunksize=chunksize))

        # 确保结果的数量为 1
        assert len(results) == 1
        result = concat(results)
        rexpected = expected[(expected.index >= beg_dt) & (expected.index <= end_dt)]
        tm.assert_frame_equal(rexpected, result)

        # 使用迭代器和 where 子句进行选择，选择 *nothing* 的条件
        #
        # 为了与 Python 习惯一致，建议这种情况应返回 []，例如 `for e in []: print True` 永远不会打印 True。
        where = f"index <= '{beg_dt}' & index >= '{end_dt}'"
        results = list(store.select("df", where=where, chunksize=chunksize))

        # 确保结果的数量为 0
        assert len(results) == 0


# 定义一个测试函数，用于测试 DataFrame 的选择功能
def test_frame_select(setup_path, request):
    df = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=date_range("2000-01-01", periods=10, freq="B"),
    )
    # 使用 ensure_clean_store 函数创建一个临时存储，并在退出代码块时确保其被清理
    with ensure_clean_store(setup_path) as store:
        # 将 DataFrame df 存储到临时存储中，使用表格格式
        store.put("frame", df, format="table")
        # 计算 DataFrame df 的中间日期
        date = df.index[len(df) // 2]

        # 创建一个 Term 对象 crit1，表示条件 index>=date
        crit1 = Term("index>=date")
        # 断言 crit1 的环境中的 scope 字典中的 date 键对应于计算得到的 date 值
        assert crit1.env.scope["date"] == date

        # 创建两个字符串条件 crit2 和 crit3
        crit2 = "columns=['A', 'D']"
        crit3 = "columns=A"

        # 应用 pytest 的 xfail 标记到 request 上，标记针对 PY312 的 AST 变更预期会抛出 TypeError
        request.applymarker(
            pytest.mark.xfail(
                PY312,
                reason="AST change in PY312",
                raises=TypeError,
            )
        )

        # 从存储中选择 "frame"，使用 crit1 和 crit2 条件进行选择
        result = store.select("frame", [crit1, crit2])
        # 从原始 DataFrame df 中提取预期的数据，即从 date 开始的列 "A" 和 "D"
        expected = df.loc[date:, ["A", "D"]]
        # 断言 result 和 expected 的 DataFrame 相等性
        tm.assert_frame_equal(result, expected)

        # 从存储中选择 "frame"，使用 crit3 条件进行选择
        result = store.select("frame", [crit3])
        # 从原始 DataFrame df 中提取预期的数据，即列 "A"
        expected = df.loc[:, ["A"]]
        # 断言 result 和 expected 的 DataFrame 相等性
        tm.assert_frame_equal(result, expected)

        # 创建一个新的 DataFrame df，包含随机生成的数据
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )
        # 将新的 DataFrame df 追加到存储中，存储键为 "df_time"
        store.append("df_time", df)
        # 设置异常消息字符串
        msg = "day is out of range for month: 0"
        # 使用 pytest 的 raises 断言捕获 ValueError 异常，并匹配异常消息 msg
        with pytest.raises(ValueError, match=msg):
            # 在存储中选择 "df_time"，使用条件 "index>0"
            store.select("df_time", "index>0")

        # 以下代码段被注释掉，表明未将 DataFrame "frame" 以表格格式写入存储中时无法选择
        # store['frame'] = df
        # 使用 pytest 的 raises 断言捕获 ValueError 异常
        # with pytest.raises(ValueError):
        #     store.select('frame', [crit1, crit2])
# 定义一个测试函数，用于复杂条件下的数据框选择测试
def test_frame_select_complex(setup_path):
    # select via complex criteria

    # 创建一个包含随机数据的数据框，列名为['A', 'B', 'C', 'D']，索引从2000-01-01开始，频率为工作日，共10行
    df = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=date_range("2000-01-01", periods=10, freq="B"),
    )
    # 在数据框中添加名为'string'的新列，初始值为'foo'
    df["string"] = "foo"
    # 设置索引0到3的行的'string'列值为'bar'
    df.loc[df.index[0:4], "string"] = "bar"

    # 使用setup_path确保存储环境的干净状态，store是环境的上下文管理器
    with ensure_clean_store(setup_path) as store:
        # 将数据框df存储到store中，使用表格格式，并指定'string'列作为数据列
        store.put("df", df, format="table", data_columns=["string"])

        # 空条件查询，使用'index>df.index[3] & string="bar"'作为查询条件
        result = store.select("df", 'index>df.index[3] & string="bar"')
        # 预期结果为df中满足条件'index>df.index[3] & string=="bar"'的子集
        expected = df.loc[(df.index > df.index[3]) & (df.string == "bar")]
        tm.assert_frame_equal(result, expected)

        # 查询条件为'index>df.index[3] & string="foo"'
        result = store.select("df", 'index>df.index[3] & string="foo"')
        # 预期结果为df中满足条件'index>df.index[3] & string=="foo"'的子集
        expected = df.loc[(df.index > df.index[3]) & (df.string == "foo")]
        tm.assert_frame_equal(result, expected)

        # 或查询条件，'index>df.index[3] | string="bar"'
        result = store.select("df", 'index>df.index[3] | string="bar"')
        # 预期结果为df中满足条件'index>df.index[3] | string=="bar"'的子集
        expected = df.loc[(df.index > df.index[3]) | (df.string == "bar")]
        tm.assert_frame_equal(result, expected)

        # 复杂条件查询，'(index>df.index[3] & index<=df.index[6]) | string="bar"'
        result = store.select(
            "df", '(index>df.index[3] & index<=df.index[6]) | string="bar"'
        )
        # 预期结果为df中满足条件'((index>df.index[3] & index<=df.index[6]) | string=="bar")'的子集
        expected = df.loc[
            ((df.index > df.index[3]) & (df.index <= df.index[6]))
            | (df.string == "bar")
        ]
        tm.assert_frame_equal(result, expected)

        # 反向查询条件，'string!="bar"'
        result = store.select("df", 'string!="bar"')
        # 预期结果为df中满足条件'string!="bar"'的子集
        expected = df.loc[df.string != "bar"]
        tm.assert_frame_equal(result, expected)

        # 对于不支持的条件（如invert），抛出异常
        msg = "cannot use an invert condition when passing to numexpr"
        with pytest.raises(NotImplementedError, match=msg):
            store.select("df", '~(string="bar")')

        # 支持在过滤器中使用反向条件
        result = store.select("df", "~(columns=['A','B'])")
        # 预期结果为df中除了列'A'和'B'之外的所有列
        expected = df.loc[:, df.columns.difference(["A", "B"])]
        tm.assert_frame_equal(result, expected)

        # 使用'in'操作符，'index>df.index[3] & columns in ['A','B']'
        result = store.select("df", "index>df.index[3] & columns in ['A','B']")
        # 预期结果为df中满足条件'index>df.index[3]'并且包含列'A'和'B'的子集
        expected = df.loc[df.index > df.index[3]].reindex(columns=["A", "B"])
        tm.assert_frame_equal(result, expected)


# 定义第二个测试函数，用于复杂条件下的数据框选择测试2
def test_frame_select_complex2(tmp_path):
    # 设置参数文件路径和历史数据文件路径
    pp = tmp_path / "params.hdf"
    hh = tmp_path / "hist.hdf"

    # 使用非平凡的选择条件创建参数数据框
    params = DataFrame({"A": [1, 1, 2, 2, 3]})
    # 将参数数据框params存储为HDF文件，key为'df'，格式为表格，'A'列作为数据列
    params.to_hdf(pp, key="df", mode="w", format="table", data_columns=["A"])

    # 使用指定条件从params中读取数据
    selection = read_hdf(pp, "df", where="A=[2,3]")

    # 创建包含随机数据的历史数据框hist，列名为'data'，索引为多级索引(l1, l2)
    hist = DataFrame(
        np.random.default_rng(2).standard_normal((25, 1)),
        columns=["data"],
        index=MultiIndex.from_tuples(
            [(i, j) for i in range(5) for j in range(5)], names=["l1", "l2"]
        ),
    )

    # 将历史数据框hist存储为HDF文件，key为'df'，格式为表格
    hist.to_hdf(hh, key="df", mode="w", format="table")

    # 从历史数据中读取符合条件'l1=[2, 3, 4]'的数据
    expected = read_hdf(hh, "df", where="l1=[2, 3, 4]")

    # 作用域中使用类似列表的对象l0
    l0 = selection.index.tolist()  # noqa: F841
    # 使用 HDFStore 打开指定的 HDF 文件，使用 'hh' 作为文件句柄
    with HDFStore(hh) as store:
        # 从 HDFStore 中选择 'df' 表，条件是 'l1=l0'，并将结果存储在 result 变量中
        result = store.select("df", where="l1=l0")
        # 使用 tm.assert_frame_equal 检查 result 是否与期望的结果相等

    # 调用 read_hdf 函数，读取 HDF 文件中 'df' 表，条件为 'l1=l0'，将结果存储在 result 变量中
    result = read_hdf(hh, "df", where="l1=l0")
    # 使用 tm.assert_frame_equal 检查 result 是否与期望的结果相等

    # 获取 selection 对象的索引，存储在 index 变量中（忽略 F841 警告）
    index = selection.index  # noqa: F841
    # 调用 read_hdf 函数，读取 HDF 文件中 'df' 表，条件为 'l1=index'，将结果存储在 result 变量中
    result = read_hdf(hh, "df", where="l1=index")
    # 使用 tm.assert_frame_equal 检查 result 是否与期望的结果相等

    # 调用 read_hdf 函数，读取 HDF 文件中 'df' 表，条件为 'l1=selection.index'，将结果存储在 result 变量中
    result = read_hdf(hh, "df", where="l1=selection.index")
    # 使用 tm.assert_frame_equal 检查 result 是否与期望的结果相等

    # 调用 read_hdf 函数，读取 HDF 文件中 'df' 表，条件为 'l1=selection.index.tolist()'，将结果存储在 result 变量中
    result = read_hdf(hh, "df", where="l1=selection.index.tolist()")
    # 使用 tm.assert_frame_equal 检查 result 是否与期望的结果相等

    # 调用 read_hdf 函数，读取 HDF 文件中 'df' 表，条件为 'l1=list(selection.index)'，将结果存储在 result 变量中
    result = read_hdf(hh, "df", where="l1=list(selection.index)")
    # 使用 tm.assert_frame_equal 检查 result 是否与期望的结果相等

    # 在带有索引的作用域内使用 HDFStore 打开指定的 HDF 文件，使用 'hh' 作为文件句柄
    with HDFStore(hh) as store:
        # 从 HDFStore 中选择 'df' 表，条件是 'l1=index'，并将结果存储在 result 变量中
        result = store.select("df", where="l1=index")
        # 使用 tm.assert_frame_equal 检查 result 是否与期望的结果相等

        # 从 HDFStore 中选择 'df' 表，条件是 'l1=selection.index'，并将结果存储在 result 变量中
        result = store.select("df", where="l1=selection.index")
        # 使用 tm.assert_frame_equal 检查 result 是否与期望的结果相等

        # 从 HDFStore 中选择 'df' 表，条件是 'l1=selection.index.tolist()'，并将结果存储在 result 变量中
        result = store.select("df", where="l1=selection.index.tolist()")
        # 使用 tm.assert_frame_equal 检查 result 是否与期望的结果相等

        # 从 HDFStore 中选择 'df' 表，条件是 'l1=list(selection.index)'，并将结果存储在 result 变量中
        result = store.select("df", where="l1=list(selection.index)")
        # 使用 tm.assert_frame_equal 检查 result 是否与期望的结果相等
# 测试无效过滤器功能的函数
def test_invalid_filtering(setup_path):
    # 创建一个包含随机数据的 DataFrame，列为 A、B、C、D，索引为从 '2000-01-01' 开始的 10 个工作日频率日期
    df = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=date_range("2000-01-01", periods=10, freq="B"),
    )

    # 使用 ensure_clean_store 函数确保环境清洁并创建一个数据存储对象
    with ensure_clean_store(setup_path) as store:
        # 将 DataFrame 存储到数据存储对象中，格式为表格格式
        store.put("df", df, format="table")

        # 设置错误消息字符串
        msg = "unable to collapse Joint Filters"

        # 使用 pytest 的 assertRaises 检测是否抛出 NotImplementedError 异常，匹配错误消息字符串
        with pytest.raises(NotImplementedError, match=msg):
            store.select("df", "columns=['A'] | columns=['B']")

        # 同样使用 pytest 的 assertRaises 检测是否抛出 NotImplementedError 异常，匹配错误消息字符串
        with pytest.raises(NotImplementedError, match=msg):
            store.select("df", "columns=['A','B'] & columns=['C']")


# 测试字符串选择功能的函数，针对 GH 2973 提出的问题
def test_string_select(setup_path):
    # 使用 ensure_clean_store 函数确保环境清洁并创建一个数据存储对象
    with ensure_clean_store(setup_path) as store:
        # 创建一个包含随机数据的 DataFrame，列为 A、B、C、D，索引为从 '2000-01-01' 开始的 10 个工作日频率日期
        df = DataFrame(
            np.random.default_rng(2).standard_normal((10, 4)),
            columns=Index(list("ABCD"), dtype=object),
            index=date_range("2000-01-01", periods=10, freq="B"),
        )

        # 在 DataFrame 中添加一列 'x'，大部分值为 'none'，部分值为空字符串
        df["x"] = "none"
        df.loc[df.index[2:7], "x"] = ""

        # 将 DataFrame 存储到数据存储对象中，命名为 'df'，并指定数据列为 'x'
        store.append("df", df, data_columns=["x"])

        # 使用 store.select 方法选择 'df' 中 'x=none' 的数据，与预期结果比较
        result = store.select("df", "x=none")
        expected = df[df.x == "none"]
        tm.assert_frame_equal(result, expected)

        # 使用 store.select 方法选择 'df' 中 'x!=none' 的数据，与预期结果比较
        result = store.select("df", "x!=none")
        expected = df[df.x != "none"]
        tm.assert_frame_equal(result, expected)

        # 创建 DataFrame 'df2'，复制 'df'，并将部分 'x' 值为空字符串替换为 NaN
        df2 = df.copy()
        df2.loc[df2.x == "", "x"] = np.nan

        # 将 DataFrame 'df2' 存储到数据存储对象中，命名为 'df2'，并指定数据列为 'x'
        store.append("df2", df2, data_columns=["x"])

        # 使用 store.select 方法选择 'df2' 中 'x!=none' 的数据，与预期结果比较
        result = store.select("df2", "x!=none")
        expected = df2[pd.isna(df2.x)]
        tm.assert_frame_equal(result, expected)

        # 在 DataFrame 中添加一列 'int'，大部分值为 1，部分值为 2
        df["int"] = 1
        df.loc[df.index[2:7], "int"] = 2

        # 将 DataFrame 'df' 存储到数据存储对象中，命名为 'df3'，并指定数据列为 'int'
        store.append("df3", df, data_columns=["int"])

        # 使用 store.select 方法选择 'df3' 中 'int=2' 的数据，与预期结果比较
        result = store.select("df3", "int=2")
        expected = df[df.int == 2]
        tm.assert_frame_equal(result, expected)

        # 使用 store.select 方法选择 'df3' 中 'int!=2' 的数据，与预期结果比较
        result = store.select("df3", "int!=2")
        expected = df[df.int != 2]
        tm.assert_frame_equal(result, expected)


# 测试多个数据集选择功能的函数
def test_select_as_multiple(setup_path):
    # 创建一个包含随机数据的 DataFrame，列为 A、B、C、D，索引为从 '2000-01-01' 开始的 10 个工作日频率日期
    df1 = DataFrame(
        np.random.default_rng(2).standard_normal((10, 4)),
        columns=Index(list("ABCD"), dtype=object),
        index=date_range("2000-01-01", periods=10, freq="B"),
    )

    # 复制 df1，并将列名重命名为 A_2、B_2、C_2、D_2，同时添加一列 'foo' 值为 'bar'
    df2 = df1.copy().rename(columns="{}_2".format)
    df2["foo"] = "bar"
    # 使用 ensure_clean_store 函数确保在 setup_path 上的存储环境是干净的
    with ensure_clean_store(setup_path) as store:
        msg = "keys must be a list/tuple"
        
        # 断言 TypeError 异常被正确地触发，并且错误消息为 "keys must be a list/tuple"
        with pytest.raises(TypeError, match=msg):
            # 调用 store.select_as_multiple 方法，预期会抛出 TypeError 异常
            # where 参数指定条件，selector 参数指定选择器名称
            store.select_as_multiple(None, where=["A>0", "B>0"], selector="df1")

        # 向存储中追加 df1 表，指定数据列为 ["A", "B"]
        store.append("df1", df1, data_columns=["A", "B"])
        
        # 向存储中追加 df2 表
        store.append("df2", df2)

        # 再次测试 TypeError 异常被正确地触发，消息为 "keys must be a list/tuple"
        with pytest.raises(TypeError, match=msg):
            store.select_as_multiple(None, where=["A>0", "B>0"], selector="df1")

        # 测试传入无效的键 [None] 时是否能正确触发 TypeError 异常
        with pytest.raises(TypeError, match=msg):
            store.select_as_multiple([None], where=["A>0", "B>0"], selector="df1")

        # 测试选择不存在的键 df3 时是否能正确触发 KeyError 异常
        msg = "'No object named df3 in the file'"
        with pytest.raises(KeyError, match=msg):
            store.select_as_multiple(
                ["df1", "df3"], where=["A>0", "B>0"], selector="df1"
            )

        # 测试选择不存在的键 df3 时是否能正确触发 KeyError 异常
        with pytest.raises(KeyError, match=msg):
            store.select_as_multiple(["df3"], where=["A>0", "B>0"], selector="df1")

        # 测试选择不存在的键 df4 时是否能正确触发 KeyError 异常
        with pytest.raises(KeyError, match="'No object named df4 in the file'"):
            store.select_as_multiple(
                ["df1", "df2"], where=["A>0", "B>0"], selector="df4"
            )

        # 测试默认选择，与通过 select_as_multiple 返回的结果进行比较
        result = store.select("df1", ["A>0", "B>0"])
        expected = store.select_as_multiple(
            ["df1"], where=["A>0", "B>0"], selector="df1"
        )
        tm.assert_frame_equal(result, expected)

        # 再次测试默认选择，与通过 select_as_multiple 返回的结果进行比较
        expected = store.select_as_multiple("df1", where=["A>0", "B>0"], selector="df1")
        tm.assert_frame_equal(result, expected)

        # 测试多表选择，与预期结果进行比较
        result = store.select_as_multiple(
            ["df1", "df2"], where=["A>0", "B>0"], selector="df1"
        )
        expected = concat([df1, df2], axis=1)
        expected = expected[(expected.A > 0) & (expected.B > 0)]
        tm.assert_frame_equal(result, expected, check_freq=False)

        # FIXME: 2021-01-20 某些构建中，此处失败，freq 为 None vs 4B

        # 使用不同的选择器测试多表选择，与预期结果进行比较
        result = store.select_as_multiple(
            ["df1", "df2"], where="index>df2.index[4]", selector="df2"
        )
        expected = concat([df1, df2], axis=1)
        expected = expected[5:]
        tm.assert_frame_equal(result, expected)

        # 测试行数不匹配时是否正确触发 ValueError 异常
        df3 = df1.copy().head(2)
        store.append("df3", df3)
        msg = "all tables must have exactly the same nrows!"
        with pytest.raises(ValueError, match=msg):
            store.select_as_multiple(
                ["df1", "df3"], where=["A>0", "B>0"], selector="df1"
            )
# 测试函数，用于验证修复了 Bug 4858 的情况下的 NaN 选择问题
def test_nan_selection_bug_4858(setup_path):
    # 在清理后的存储环境中进行操作
    with ensure_clean_store(setup_path) as store:
        # 创建一个包含两列的 DataFrame，其中一列为浮点数类型
        df = DataFrame({"cols": range(6), "values": range(6)}, dtype="float64")
        # 将 "cols" 列中的值加上 10，并转换为字符串
        df["cols"] = (df["cols"] + 10).apply(str)
        # 将第一行第一列设置为 NaN
        df.iloc[0] = np.nan

        # 预期结果的 DataFrame，包含特定条件下的行和列
        expected = DataFrame(
            {"cols": ["13.0", "14.0", "15.0"], "values": [3.0, 4.0, 5.0]},
            index=[3, 4, 5],
        )

        # 将 DataFrame 写入存储，特定列作为索引
        store.append("df", df, data_columns=True, index=["cols"])
        # 从存储中选择满足条件的数据集
        result = store.select("df", where="values>2.0")
        # 断言结果与预期结果相等
        tm.assert_frame_equal(result, expected)


# 测试函数，验证带有嵌套特殊字符的查询
def test_query_with_nested_special_character(setup_path):
    # 创建一个包含两列的 DataFrame，其中一列包含特殊字符
    df = DataFrame(
        {
            "a": ["a", "a", "c", "b", "test & test", "c", "b", "e"],
            "b": [1, 2, 3, 4, 5, 6, 7, 8],
        }
    )
    # 期望的结果 DataFrame，选择包含特定字符串的行
    expected = df[df.a == "test & test"]
    # 在清理后的存储环境中进行操作
    with ensure_clean_store(setup_path) as store:
        # 将 DataFrame 写入存储，指定格式和数据列
        store.append("test", df, format="table", data_columns=True)
        # 从存储中选择满足特定条件的数据集
        result = store.select("test", 'a = "test & test"')
    # 断言结果与预期结果相等
    tm.assert_frame_equal(expected, result)


# 测试函数，验证长浮点数文字的查询情况
def test_query_long_float_literal(setup_path):
    # 创建一个包含单列的 DataFrame，其中包含长浮点数文字
    df = DataFrame({"A": [1000000000.0009, 1000000000.0011, 1000000000.0015]})

    # 在清理后的存储环境中进行操作
    with ensure_clean_store(setup_path) as store:
        # 将 DataFrame 写入存储，指定格式和数据列
        store.append("test", df, format="table", data_columns=True)

        # 设置一个截断值，选择小于该值的数据
        cutoff = 1000000000.0006
        result = store.select("test", f"A < {cutoff:.4f}")
        # 断言结果为空
        assert result.empty

        # 设置另一个截断值，选择大于该值的数据
        cutoff = 1000000000.0010
        result = store.select("test", f"A > {cutoff:.4f}")
        # 期望的结果 DataFrame，选择特定行
        expected = df.loc[[1, 2], :]
        # 断言结果与预期结果相等
        tm.assert_frame_equal(expected, result)

        # 设置一个精确值，选择等于该值的数据
        exact = 1000000000.0011
        result = store.select("test", f"A == {exact:.4f}")
        # 期望的结果 DataFrame，选择特定行
        expected = df.loc[[1], :]
        # 断言结果与预期结果相等
        tm.assert_frame_equal(expected, result)


# 测试函数，验证列类型比较的查询情况
def test_query_compare_column_type(setup_path):
    # 创建一个包含多列的 DataFrame，包括日期、实数日期、浮点数和整数列
    df = DataFrame(
        {
            "date": ["2014-01-01", "2014-01-02"],
            "real_date": date_range("2014-01-01", periods=2),
            "float": [1.1, 1.2],
            "int": [1, 2],
        },
        columns=["date", "real_date", "float", "int"],
    )
    # 使用 ensure_clean_store 函数确保 setup_path 目录的存储处于干净状态，并将其作为 store 对象使用
    with ensure_clean_store(setup_path) as store:
        # 向 store 对象中添加名为 "test" 的数据框 df，使用表格格式存储，并启用数据列索引
        store.append("test", df, format="table", data_columns=True)

        # 创建时间戳对象 ts，但未被使用
        ts = Timestamp("2014-01-01")  # noqa: F841
        # 从 store 中选择名为 "test" 的数据，条件为 real_date 大于 ts，返回结果赋值给 result
        result = store.select("test", where="real_date > ts")
        # 从数据框 df 中选择索引为 1 的行，构建预期结果 expected
        expected = df.loc[[1], :]
        # 使用测试框架确保 expected 与 result 相等
        tm.assert_frame_equal(expected, result)

        # 遍历操作符列表 ["<", ">", "=="]
        for op in ["<", ">", "=="]:
            # 对于非字符串类型 v，尝试将其与字符串列比较将导致失败
            for v in [2.1, True, Timestamp("2014-01-01"), pd.Timedelta(1, "s")]:
                # 构建查询字符串，比较 date 列与 v 的值
                query = f"date {op} v"
                # 构建错误消息，指示无法将类型为 type(v) 的 v 与字符串列进行比较
                msg = f"Cannot compare {v} of type {type(v)} to string column"
                # 使用 pytest 框架断言会抛出 TypeError 异常，并匹配特定错误消息
                with pytest.raises(TypeError, match=msg):
                    store.select("test", where=query)

            # 对于字符串类型 v，尝试将其与其它列比较，需要能够转换为对应的数据类型
            v = "a"
            for col in ["int", "float", "real_date"]:
                # 构建查询字符串，比较指定列 col 与 v 的值
                query = f"{col} {op} v"
                if col == "real_date":
                    # 如果列为 real_date，构建特定的错误消息
                    msg = 'Given date string "a" not likely a datetime'
                else:
                    # 其它列则指示无法将字符串转换为对应类型
                    msg = "could not convert string to"
                # 使用 pytest 框架断言会抛出 ValueError 异常，并匹配特定错误消息
                with pytest.raises(ValueError, match=msg):
                    store.select("test", where=query)

            # 遍历字符串类型列表 ["1", "1.1", "2014-01-01"] 与列名列表 ["int", "float", "real_date"]
            for v, col in zip(
                ["1", "1.1", "2014-01-01"], ["int", "float", "real_date"]
            ):
                # 构建查询字符串，比较列 col 与 v 的值
                query = f"{col} {op} v"
                # 从 store 中选择符合查询条件的数据，并将结果赋值给 result
                result = store.select("test", where=query)

                # 根据操作符 op 确定预期的数据框 expected
                if op == "==":
                    expected = df.loc[[0], :]
                elif op == ">":
                    expected = df.loc[[1], :]
                else:
                    expected = df.loc[[], :]
                # 使用测试框架确保 expected 与 result 相等
                tm.assert_frame_equal(expected, result)
# 使用 pytest 的 parametrize 装饰器，为测试函数 test_select_empty_where 提供多组参数化测试数据
@pytest.mark.parametrize("where", ["", (), (None,), [], [None]])
def test_select_empty_where(tmp_path, where):
    # 注释标识：GH26610

    # 创建一个包含 [1, 2, 3] 的 DataFrame 对象
    df = DataFrame([1, 2, 3])
    # 在临时路径下创建 HDF5 文件路径对象
    path = tmp_path / "empty_where.h5"
    
    # 使用 HDFStore 打开创建的 HDF5 文件
    with HDFStore(path) as store:
        # 将 DataFrame 对象 df 存储在 HDFStore 中，键名为 "df"，数据表类型为 "t"
        store.put("df", df, "t")
        # 调用 read_hdf 函数从 HDFStore 中读取数据，根据 where 参数指定条件
        result = read_hdf(store, "df", where=where)
        # 使用 tm.assert_frame_equal 断言 result 和 df 相等
        tm.assert_frame_equal(result, df)


# 定义测试函数 test_select_large_integer，测试处理大整数的情况
def test_select_large_integer(tmp_path):
    # 创建 HDF5 文件路径对象
    path = tmp_path / "large_int.h5"

    # 创建一个包含大整数的 DataFrame 对象
    df = DataFrame(
        zip(
            ["a", "b", "c", "d"],
            [-9223372036854775801, -9223372036854775802, -9223372036854775803, 123],
        ),
        columns=["x", "y"],
    )
    # 初始化结果变量为 None
    result = None
    # 使用 HDFStore 打开创建的 HDF5 文件
    with HDFStore(path) as s:
        # 将 DataFrame 对象 df 追加到 HDFStore 中的数据表 "data"，启用数据列和禁用索引
        s.append("data", df, data_columns=True, index=False)
        # 使用 s.select 选择数据表 "data" 中符合条件 "y==-9223372036854775801" 的行，并获取其 "y" 列的第一个值
        result = s.select("data", where="y==-9223372036854775801").get("y").get(0)
    # 期望的结果为 df["y"] 的第一个值
    expected = df["y"][0]

    # 使用断言验证期望的结果和实际结果是否相等
    assert expected == result
```