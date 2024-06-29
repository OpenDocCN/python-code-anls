# `D:\src\scipysrc\pandas\pandas\tests\groupby\aggregate\test_numba.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 库，用于编写和运行测试用例

from pandas.errors import NumbaUtilError  # 从 pandas 库中导入 NumbaUtilError 错误类

from pandas import (  # 导入 pandas 库中的多个模块和类
    DataFrame,  # 数据帧类，用于操作二维表格数据
    Index,  # 索引类，用于标识数据帧的索引
    NamedAgg,  # 命名聚合类，用于指定聚合操作的名称
    Series,  # 系列类，用于操作一维数据
    option_context,  # 上下文选项类，用于临时设置选项
)
import pandas._testing as tm  # 导入 pandas 内部的测试工具模块

pytestmark = pytest.mark.single_cpu  # 使用 pytest.mark.single_cpu 标记测试用例


def test_correct_function_signature():
    pytest.importorskip("numba")  # 确保 numba 库存在，否则跳过测试

    def incorrect_function(x):
        return sum(x) * 2.7  # 不正确的函数实现，将每个元素乘以 2.7 并求和

    # 创建包含键和数据列的数据帧
    data = DataFrame(
        {"key": ["a", "a", "b", "b", "a"], "data": [1.0, 2.0, 3.0, 4.0, 5.0]},
        columns=["key", "data"],
    )

    # 断言调用 agg 函数时会引发 NumbaUtilError 错误，并检查错误消息
    with pytest.raises(NumbaUtilError, match="The first 2"):
        data.groupby("key").agg(incorrect_function, engine="numba")

    with pytest.raises(NumbaUtilError, match="The first 2"):
        data.groupby("key")["data"].agg(incorrect_function, engine="numba")


def test_check_nopython_kwargs():
    pytest.importorskip("numba")  # 确保 numba 库存在，否则跳过测试

    def incorrect_function(values, index):
        return sum(values) * 2.7  # 不正确的函数实现，将每个元素乘以 2.7 并求和

    # 创建包含键和数据列的数据帧
    data = DataFrame(
        {"key": ["a", "a", "b", "b", "a"], "data": [1.0, 2.0, 3.0, 4.0, 5.0]},
        columns=["key", "data"],
    )

    # 断言调用 agg 函数时会引发 NumbaUtilError 错误，并检查错误消息
    with pytest.raises(NumbaUtilError, match="numba does not support"):
        data.groupby("key").agg(incorrect_function, engine="numba", a=1)

    with pytest.raises(NumbaUtilError, match="numba does not support"):
        data.groupby("key")["data"].agg(incorrect_function, engine="numba", a=1)


@pytest.mark.filterwarnings("ignore")
# Filter warnings when parallel=True and the function can't be parallelized by Numba
@pytest.mark.parametrize("jit", [True, False])
def test_numba_vs_cython(jit, frame_or_series, nogil, parallel, nopython, as_index):
    pytest.importorskip("numba")  # 确保 numba 库存在，否则跳过测试

    def func_numba(values, index):
        return np.mean(values) * 2.7  # 计算数值序列的均值并乘以 2.7

    if jit:
        # 如果启用 jit 参数，使用 numba 库对 func_numba 进行即时编译
        import numba
        func_numba = numba.jit(func_numba)

    # 创建包含键和数据列的数据帧
    data = DataFrame(
        {0: ["a", "a", "b", "b", "a"], 1: [1.0, 2.0, 3.0, 4.0, 5.0]}, columns=[0, 1]
    )
    
    # 根据条件选择组类型（DataFrame 或 Series）
    engine_kwargs = {"nogil": nogil, "parallel": parallel, "nopython": nopython}
    grouped = data.groupby(0, as_index=as_index)
    if frame_or_series is Series:
        grouped = grouped[1]

    # 调用 agg 函数并比较使用 numba 引擎和 cython 引擎的结果
    result = grouped.agg(func_numba, engine="numba", engine_kwargs=engine_kwargs)
    expected = grouped.agg(lambda x: np.mean(x) * 2.7, engine="cython")
    
    # 断言结果与预期相等
    tm.assert_equal(result, expected)


@pytest.mark.filterwarnings("ignore")
# Filter warnings when parallel=True and the function can't be parallelized by Numba
@pytest.mark.parametrize("jit", [True, False])
def test_cache(jit, frame_or_series, nogil, parallel, nopython):
    # 测试函数是否正确缓存，即使切换了不同的函数实现
    pytest.importorskip("numba")  # 确保 numba 库存在，否则跳过测试

    def func_1(values, index):
        return np.mean(values) - 3.4  # 计算数值序列的均值并减去 3.4

    def func_2(values, index):
        return np.mean(values) * 2.7  # 计算数值序列的均值并乘以 2.7

    if jit:
        import numba
        func_1 = numba.jit(func_1)  # 对 func_1 进行即时编译
        func_2 = numba.jit(func_2)  # 对 func_2 进行即时编译
    # 创建一个 DataFrame 对象，包含两列数据，列名为 0 和 1
    data = DataFrame(
        {0: ["a", "a", "b", "b", "a"], 1: [1.0, 2.0, 3.0, 4.0, 5.0]}, columns=[0, 1]
    )
    # 根据传入的参数构建引擎相关的关键字参数字典
    engine_kwargs = {"nogil": nogil, "parallel": parallel, "nopython": nopython}
    # 根据列 0 对数据进行分组
    grouped = data.groupby(0)
    # 如果 frame_or_series 是 Series 类型，则重新赋值 grouped 变量
    if frame_or_series is Series:
        grouped = grouped[1]

    # 对分组后的数据应用 func_1 函数进行聚合，使用 numba 引擎和指定的引擎关键字参数
    result = grouped.agg(func_1, engine="numba", engine_kwargs=engine_kwargs)
    # 对分组后的数据应用匿名函数进行聚合，计算每组数据的均值减去 3.4，使用 cython 引擎
    expected = grouped.agg(lambda x: np.mean(x) - 3.4, engine="cython")
    # 使用测试框架中的 assert_equal 函数比较 result 和 expected 是否相等
    tm.assert_equal(result, expected)

    # 将 func_2 添加到缓存中
    result = grouped.agg(func_2, engine="numba", engine_kwargs=engine_kwargs)
    # 对分组后的数据应用匿名函数进行聚合，计算每组数据的均值乘以 2.7，使用 cython 引擎
    expected = grouped.agg(lambda x: np.mean(x) * 2.7, engine="cython")
    # 使用测试框架中的 assert_equal 函数比较 result 和 expected 是否相等
    tm.assert_equal(result, expected)

    # 重新测试 func_1，这次应该使用缓存中的结果
    result = grouped.agg(func_1, engine="numba", engine_kwargs=engine_kwargs)
    # 对分组后的数据应用匿名函数进行聚合，计算每组数据的均值减去 3.4，使用 cython 引擎
    expected = grouped.agg(lambda x: np.mean(x) - 3.4, engine="cython")
    # 使用测试框架中的 assert_equal 函数比较 result 和 expected 是否相等
    tm.assert_equal(result, expected)
# 测试使用全局配置
def test_use_global_config():
    # 导入 pytest 并跳过如果未安装 numba 模块
    pytest.importorskip("numba")

    # 定义一个函数 func_1，计算 values 的平均值减去 3.4
    def func_1(values, index):
        return np.mean(values) - 3.4

    # 创建一个 DataFrame 对象 data，包含两列，0列为字符串列表，1列为浮点数列表
    data = DataFrame(
        {0: ["a", "a", "b", "b", "a"], 1: [1.0, 2.0, 3.0, 4.0, 5.0]}, columns=[0, 1]
    )
    # 按第0列对 data 进行分组
    grouped = data.groupby(0)
    # 使用 numba 引擎计算 grouped 的聚合结果，应用 func_1 函数
    expected = grouped.agg(func_1, engine="numba")
    # 在 compute.use_numba 设置为 True 的上下文中，使用 None 引擎计算 grouped 的聚合结果
    with option_context("compute.use_numba", True):
        result = grouped.agg(func_1, engine=None)
    # 断言期望结果和实际结果相等
    tm.assert_frame_equal(expected, result)


# 参数化测试，比较 numba 引擎和 cython 引擎在多函数聚合上的性能
@pytest.mark.parametrize(
    "agg_kwargs",
    [
        {"func": ["min", "max"]},
        {"func": "min"},
        {"func": {1: ["min", "max"], 2: "sum"}},
        {"bmin": NamedAgg(column=1, aggfunc="min")},
    ],
)
def test_multifunc_numba_vs_cython_frame(agg_kwargs):
    # 导入 pytest 并跳过如果未安装 numba 模块
    pytest.importorskip("numba")
    # 创建一个 DataFrame 对象 data，包含三列，0列为字符串列表，1列为浮点数列表，2列为整数列表
    data = DataFrame(
        {
            0: ["a", "a", "b", "b", "a"],
            1: [1.0, 2.0, 3.0, 4.0, 5.0],
            2: [1, 2, 3, 4, 5],
        },
        columns=[0, 1, 2],
    )
    # 按第0列对 data 进行分组
    grouped = data.groupby(0)
    # 使用 numba 引擎计算 grouped 的聚合结果，根据参数 agg_kwargs 指定的函数
    result = grouped.agg(**agg_kwargs, engine="numba")
    # 使用 cython 引擎计算 grouped 的聚合结果，根据参数 agg_kwargs 指定的函数
    expected = grouped.agg(**agg_kwargs, engine="cython")
    # 断言期望结果和实际结果相等
    tm.assert_frame_equal(result, expected)


# 参数化测试，比较 numba 引擎和 cython 引擎在用户定义函数上的性能
@pytest.mark.parametrize(
    "agg_kwargs,expected_func",
    [
        ({"func": lambda values, index: values.sum()}, "sum"),
        # FIXME
        # 标记预期失败的测试用例，尚未支持的功能，会在 nopython 管道中失败
        pytest.param(
            {
                "func": [
                    lambda values, index: values.sum(),
                    lambda values, index: values.min(),
                ]
            },
            ["sum", "min"],
            marks=pytest.mark.xfail(
                reason="This doesn't work yet! Fails in nopython pipeline!"
            ),
        ),
    ],
)
def test_multifunc_numba_udf_frame(agg_kwargs, expected_func):
    # 导入 pytest 并跳过如果未安装 numba 模块
    pytest.importorskip("numba")
    # 创建一个 DataFrame 对象 data，包含三列，0列为字符串列表，1列为浮点数列表，2列为整数列表
    data = DataFrame(
        {
            0: ["a", "a", "b", "b", "a"],
            1: [1.0, 2.0, 3.0, 4.0, 5.0],
            2: [1, 2, 3, 4, 5],
        },
        columns=[0, 1, 2],
    )
    # 按第0列对 data 进行分组
    grouped = data.groupby(0)
    # 使用 numba 引擎计算 grouped 的聚合结果，根据参数 agg_kwargs 指定的函数
    result = grouped.agg(**agg_kwargs, engine="numba")
    # 使用 cython 引擎计算 grouped 的聚合结果，根据预期的函数名 expected_func
    expected = grouped.agg(expected_func, engine="cython")
    # 断言期望结果和实际结果相等，检查数据类型可以移除，如果解决了 GH 44952 的问题
    tm.assert_frame_equal(result, expected, check_dtype=False)


# 参数化测试，比较 numba 引擎和 cython 引擎在 Series 对象上的多函数聚合性能
@pytest.mark.parametrize(
    "agg_kwargs",
    [{"func": ["min", "max"]}, {"func": "min"}, {"min_val": "min", "max_val": "max"}],
)
def test_multifunc_numba_vs_cython_series(agg_kwargs):
    # 导入 pytest 并跳过如果未安装 numba 模块
    pytest.importorskip("numba")
    # 创建一个 Series 对象 data，包含浮点数列表和对应的标签列表
    labels = ["a", "a", "b", "b", "a"]
    data = Series([1.0, 2.0, 3.0, 4.0, 5.0])
    # 按 labels 列对 data 进行分组
    grouped = data.groupby(labels)
    # 将 engine 设置为 numba，使用 agg_kwargs 指定的函数计算 grouped 的聚合结果
    agg_kwargs["engine"] = "numba"
    result = grouped.agg(**agg_kwargs)
    # 将 engine 设置为 cython，使用 agg_kwargs 指定的函数计算 grouped 的聚合结果
    agg_kwargs["engine"] = "cython"
    expected = grouped.agg(**agg_kwargs)
    # 如果期望结果是 DataFrame，则断言两个 DataFrame 相等，否则断言两个 Series 相等
    if isinstance(expected, DataFrame):
        tm.assert_frame_equal(result, expected)
    else:
        tm.assert_series_equal(result, expected)
@pytest.mark.parametrize(
    "data,agg_kwargs",
    [
        (Series([1.0, 2.0, 3.0, 4.0, 5.0]), {"func": ["min", "max"]}),  # 测试用例1：单个Series，包含"min"和"max"聚合函数
        (Series([1.0, 2.0, 3.0, 4.0, 5.0]), {"func": "min"}),  # 测试用例2：单个Series，仅包含"min"聚合函数
        (
            DataFrame(
                {1: [1.0, 2.0, 3.0, 4.0, 5.0], 2: [1, 2, 3, 4, 5]}, columns=[1, 2]
            ),
            {"func": ["min", "max"]},  # 测试用例3：DataFrame，包含"min"和"max"聚合函数
        ),
        (
            DataFrame(
                {1: [1.0, 2.0, 3.0, 4.0, 5.0], 2: [1, 2, 3, 4, 5]}, columns=[1, 2]
            ),
            {"func": "min"},  # 测试用例4：DataFrame，仅包含"min"聚合函数
        ),
        (
            DataFrame(
                {1: [1.0, 2.0, 3.0, 4.0, 5.0], 2: [1, 2, 3, 4, 5]}, columns=[1, 2]
            ),
            {"func": {1: ["min", "max"], 2: "sum"}},  # 测试用例5：DataFrame，列1包含"min"和"max"，列2包含"sum"聚合函数
        ),
        (
            DataFrame(
                {1: [1.0, 2.0, 3.0, 4.0, 5.0], 2: [1, 2, 3, 4, 5]}, columns=[1, 2]
            ),
            {"min_col": NamedAgg(column=1, aggfunc="min")},  # 测试用例6：DataFrame，使用NamedAgg进行聚合
        ),
    ],
)
def test_multifunc_numba_kwarg_propagation(data, agg_kwargs):
    pytest.importorskip("numba")  # 导入numba库，如果不存在则跳过测试

    labels = ["a", "a", "b", "b", "a"]
    grouped = data.groupby(labels)  # 根据标签分组数据
    result = grouped.agg(**agg_kwargs, engine="numba", engine_kwargs={"parallel": True})  # 使用numba引擎聚合数据
    expected = grouped.agg(**agg_kwargs, engine="numba")  # 期望结果使用numba引擎聚合数据

    if isinstance(expected, DataFrame):
        tm.assert_frame_equal(result, expected)  # 如果期望是DataFrame，则比较两者是否相等
    else:
        tm.assert_series_equal(result, expected)  # 如果期望是Series，则比较两者是否相等


def test_args_not_cached():
    # GH 41647
    pytest.importorskip("numba")  # 导入numba库，如果不存在则跳过测试

    def sum_last(values, index, n):
        return values[-n:].sum()  # 返回最后n个值的和

    df = DataFrame({"id": [0, 0, 1, 1], "x": [1, 1, 1, 1]})
    grouped_x = df.groupby("id")["x"]  # 根据"id"列分组数据"x"
    result = grouped_x.agg(sum_last, 1, engine="numba")  # 使用numba引擎对分组数据应用sum_last函数
    expected = Series([1.0] * 2, name="x", index=Index([0, 1], name="id"))
    tm.assert_series_equal(result, expected)  # 比较结果是否符合期望

    result = grouped_x.agg(sum_last, 2, engine="numba")  # 再次使用numba引擎对分组数据应用sum_last函数
    expected = Series([2.0] * 2, name="x", index=Index([0, 1], name="id"))
    tm.assert_series_equal(result, expected)  # 比较结果是否符合期望


def test_index_data_correctly_passed():
    # GH 43133
    pytest.importorskip("numba")  # 导入numba库，如果不存在则跳过测试

    def f(values, index):
        return np.mean(index)  # 返回索引的均值

    df = DataFrame({"group": ["A", "A", "B"], "v": [4, 5, 6]}, index=[-1, -2, -3])
    result = df.groupby("group").aggregate(f, engine="numba")  # 使用numba引擎对分组数据应用f函数
    expected = DataFrame(
        [-1.5, -3.0], columns=["v"], index=Index(["A", "B"], name="group")
    )
    tm.assert_frame_equal(result, expected)  # 比较结果是否符合期望


def test_engine_kwargs_not_cached():
    # If the user passes a different set of engine_kwargs don't return the same
    # jitted function
    pytest.importorskip("numba")  # 导入numba库，如果不存在则跳过测试

    nogil = True
    parallel = False
    nopython = True

    def func_kwargs(values, index):
        return nogil + parallel + nopython  # 返回nogil、parallel和nopython的和

    engine_kwargs = {"nopython": nopython, "nogil": nogil, "parallel": parallel}
    df = DataFrame({"value": [0, 0, 0]})
    result = df.groupby(level=0).aggregate(
        func_kwargs, engine="numba", engine_kwargs=engine_kwargs
    )  # 使用numba引擎和指定的engine_kwargs对数据应用func_kwargs函数
    )
    # 创建一个包含值为2.0的"value"列的DataFrame，并赋给变量expected
    expected = DataFrame({"value": [2.0, 2.0, 2.0]})
    # 使用pandas的tm模块中的assert_frame_equal函数比较result和expected两个DataFrame是否相等

    nogil = False
    # 设置变量nogil为False，表示不使用Numba的gil释放
    engine_kwargs = {"nopython": nopython, "nogil": nogil, "parallel": parallel}
    # 创建一个包含了nopython、nogil和parallel参数的字典，并赋给engine_kwargs变量
    result = df.groupby(level=0).aggregate(
        func_kwargs, engine="numba", engine_kwargs=engine_kwargs
    )
    # 对DataFrame df按照索引级别0进行分组聚合，使用Numba作为引擎，并传递engine_kwargs作为引擎的关键字参数
    expected = DataFrame({"value": [1.0, 1.0, 1.0]})
    # 创建一个包含值为1.0的"value"列的DataFrame，并赋给变量expected
    tm.assert_frame_equal(result, expected)
    # 使用pandas的tm模块中的assert_frame_equal函数比较result和expected两个DataFrame是否相等
@pytest.mark.filterwarnings("ignore")
# 标记该测试函数忽略所有警告信息
def test_multiindex_one_key(nogil, parallel, nopython):
    # 导入 numba 库，如果不存在则跳过测试
    pytest.importorskip("numba")

    def numba_func(values, index):
        # 示例函数，返回固定值 1
        return 1

    # 创建一个包含单个 MultiIndex 的 DataFrame
    df = DataFrame([{"A": 1, "B": 2, "C": 3}]).set_index(["A", "B"])
    # 设置引擎参数字典
    engine_kwargs = {"nopython": nopython, "nogil": nogil, "parallel": parallel}
    # 对 DataFrame 进行分组并应用 numba_func 函数进行聚合
    result = df.groupby("A").agg(
        numba_func, engine="numba", engine_kwargs=engine_kwargs
    )
    # 创建预期的 DataFrame 结果，用于比较
    expected = DataFrame([1.0], index=Index([1], name="A"), columns=["C"])
    # 使用测试框架的函数来比较 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)


def test_multiindex_multi_key_not_supported(nogil, parallel, nopython):
    # 导入 numba 库，如果不存在则跳过测试
    pytest.importorskip("numba")

    def numba_func(values, index):
        # 示例函数，返回固定值 1
        return 1

    # 创建一个包含单个 MultiIndex 的 DataFrame
    df = DataFrame([{"A": 1, "B": 2, "C": 3}]).set_index(["A", "B"])
    # 设置引擎参数字典
    engine_kwargs = {"nopython": nopython, "nogil": nogil, "parallel": parallel}
    # 使用 pytest 来确保对于多个分组标签的聚合会引发 NotImplementedError 异常
    with pytest.raises(NotImplementedError, match="more than 1 grouping labels"):
        df.groupby(["A", "B"]).agg(
            numba_func, engine="numba", engine_kwargs=engine_kwargs
        )


def test_multilabel_numba_vs_cython(numba_supported_reductions):
    # 导入 numba 库，如果不存在则跳过测试
    pytest.importorskip("numba")
    # 获取 numba_supported_reductions 参数中的函数和参数字典
    reduction, kwargs = numba_supported_reductions
    # 创建一个包含多个标签的 DataFrame
    df = DataFrame(
        {
            "A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
            "B": ["one", "one", "two", "three", "two", "two", "one", "three"],
            "C": np.random.default_rng(2).standard_normal(8),
            "D": np.random.default_rng(2).standard_normal(8),
        }
    )
    # 对 DataFrame 进行分组
    gb = df.groupby(["A", "B"])
    # 使用 numba 引擎聚合数据
    res_agg = gb.agg(reduction, engine="numba", **kwargs)
    # 使用 cython 引擎聚合数据，作为预期结果
    expected_agg = gb.agg(reduction, engine="cython", **kwargs)
    # 使用测试框架的函数来比较 res_agg 和 expected_agg 是否相等
    tm.assert_frame_equal(res_agg, expected_agg)
    # 测试直接调用聚合函数是否也正常工作
    direct_res = getattr(gb, reduction)(engine="numba", **kwargs)
    direct_expected = getattr(gb, reduction)(engine="cython", **kwargs)
    # 使用测试框架的函数来比较直接调用的结果
    tm.assert_frame_equal(direct_res, direct_expected)


def test_multilabel_udf_numba_vs_cython():
    # 导入 numba 库，如果不存在则跳过测试
    pytest.importorskip("numba")
    # 创建一个包含多个标签的 DataFrame
    df = DataFrame(
        {
            "A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
            "B": ["one", "one", "two", "three", "two", "two", "one", "three"],
            "C": np.random.default_rng(2).standard_normal(8),
            "D": np.random.default_rng(2).standard_normal(8),
        }
    )
    # 对 DataFrame 进行分组
    gb = df.groupby(["A", "B"])
    # 使用 numba 引擎聚合数据，通过 lambda 函数计算最小值
    result = gb.agg(lambda values, index: values.min(), engine="numba")
    # 使用 cython 引擎聚合数据，通过 lambda 函数计算最小值，作为预期结果
    expected = gb.agg(lambda x: x.min(), engine="cython")
    # 使用测试框架的函数来比较 result 和 expected 是否相等
    tm.assert_frame_equal(result, expected)
```