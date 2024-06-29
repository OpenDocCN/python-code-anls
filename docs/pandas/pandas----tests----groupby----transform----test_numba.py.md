# `D:\src\scipysrc\pandas\pandas\tests\groupby\transform\test_numba.py`

```
import numpy as np  # 导入 NumPy 库，用于数值计算
import pytest  # 导入 pytest 库，用于单元测试

from pandas.errors import NumbaUtilError  # 从 pandas.errors 模块中导入 NumbaUtilError 错误类

from pandas import (  # 从 pandas 库中导入 DataFrame、Series 和 option_context
    DataFrame,
    Series,
    option_context,
)
import pandas._testing as tm  # 导入 pandas 内部测试模块 pandas._testing as tm

pytestmark = pytest.mark.single_cpu  # 将 pytestmark 设置为 pytest.mark.single_cpu 标记


def test_correct_function_signature():
    pytest.importorskip("numba")  # 如果没有 numba 库则跳过测试

    def incorrect_function(x):
        return x + 1  # 定义一个错误的函数，将参数 x 加 1

    data = DataFrame(  # 创建一个 DataFrame 对象 data
        {"key": ["a", "a", "b", "b", "a"], "data": [1.0, 2.0, 3.0, 4.0, 5.0]},  # 设置 DataFrame 的数据和列
        columns=["key", "data"],  # 指定列名
    )
    with pytest.raises(NumbaUtilError, match="The first 2"):  # 捕获 NumbaUtilError 异常，匹配错误信息 "The first 2"
        data.groupby("key").transform(incorrect_function, engine="numba")

    with pytest.raises(NumbaUtilError, match="The first 2"):  # 同上，但是针对 "data" 列
        data.groupby("key")["data"].transform(incorrect_function, engine="numba")


def test_check_nopython_kwargs():
    pytest.importorskip("numba")  # 如果没有 numba 库则跳过测试

    def incorrect_function(values, index):
        return values + 1  # 定义一个错误的函数，将 values 参数加 1

    data = DataFrame(  # 创建一个 DataFrame 对象 data
        {"key": ["a", "a", "b", "b", "a"], "data": [1.0, 2.0, 3.0, 4.0, 5.0]},  # 设置 DataFrame 的数据和列
        columns=["key", "data"],  # 指定列名
    )
    with pytest.raises(NumbaUtilError, match="numba does not support"):  # 捕获 NumbaUtilError 异常，匹配错误信息 "numba does not support"
        data.groupby("key").transform(incorrect_function, engine="numba", a=1)

    with pytest.raises(NumbaUtilError, match="numba does not support"):  # 同上，但是针对 "data" 列
        data.groupby("key")["data"].transform(incorrect_function, engine="numba", a=1)


@pytest.mark.filterwarnings("ignore")
# 当 parallel=True 时，如果 Numba 无法并行化该函数，则忽略警告
@pytest.mark.parametrize("jit", [True, False])
def test_numba_vs_cython(jit, frame_or_series, nogil, parallel, nopython, as_index):
    pytest.importorskip("numba")  # 如果没有 numba 库则跳过测试

    def func(values, index):
        return values + 1  # 定义一个函数，将 values 参数加 1

    if jit:
        # 如果 jit 为 True，则测试接受的 jit 函数
        import numba

        func = numba.jit(func)  # 使用 numba.jit 对 func 进行即时编译

    data = DataFrame(  # 创建一个 DataFrame 对象 data
        {0: ["a", "a", "b", "b", "a"], 1: [1.0, 2.0, 3.0, 4.0, 5.0]},  # 设置 DataFrame 的数据和列
        columns=[0, 1],  # 指定列名
    )
    engine_kwargs = {"nogil": nogil, "parallel": parallel, "nopython": nopython}  # 定义引擎参数字典
    grouped = data.groupby(0, as_index=as_index)  # 根据列 0 对数据分组，并根据 as_index 参数选择是否作为索引

    if frame_or_series is Series:
        grouped = grouped[1]  # 如果 frame_or_series 是 Series，则选取第二个分组

    result = grouped.transform(func, engine="numba", engine_kwargs=engine_kwargs)  # 使用 numba 引擎和引擎参数 engine_kwargs 对分组进行转换
    expected = grouped.transform(lambda x: x + 1, engine="cython")  # 使用 cython 引擎对分组进行转换，预期结果应为每个值加 1

    tm.assert_equal(result, expected)  # 使用 pandas 内部测试模块进行结果比较


@pytest.mark.filterwarnings("ignore")
# 当 parallel=True 时，如果 Numba 无法并行化该函数，则忽略警告
@pytest.mark.parametrize("jit", [True, False])
def test_cache(jit, frame_or_series, nogil, parallel, nopython):
    # 测试函数是否能正确缓存，如果我们切换函数
    pytest.importorskip("numba")  # 如果没有 numba 库则跳过测试

    def func_1(values, index):
        return values + 1  # 定义第一个函数，将 values 参数加 1

    def func_2(values, index):
        return values * 5  # 定义第二个函数，将 values 参数乘以 5

    if jit:
        import numba

        func_1 = numba.jit(func_1)  # 如果 jit 为 True，则对 func_1 进行即时编译
        func_2 = numba.jit(func_2)  # 如果 jit 为 True，则对 func_2 进行即时编译

    data = DataFrame(  # 创建一个 DataFrame 对象 data
        {0: ["a", "a", "b", "b", "a"], 1: [1.0, 2.0, 3.0, 4.0, 5.0]},  # 设置 DataFrame 的数据和列
        columns=[0, 1],  # 指定列名
    )
    # 定义引擎参数的字典，包括是否使用全局解释器锁 (nogil)、是否并行执行 (parallel)、是否使用无Python模式 (nopython)
    engine_kwargs = {"nogil": nogil, "parallel": parallel, "nopython": nopython}
    
    # 根据第一列对数据进行分组
    grouped = data.groupby(0)
    
    # 如果frame_or_series是Series类型，则重新赋值grouped为其第二列数据
    if frame_or_series is Series:
        grouped = grouped[1]
    
    # 使用Numba引擎和指定的引擎参数对分组后的数据应用func_1函数进行转换，将结果赋给result
    result = grouped.transform(func_1, engine="numba", engine_kwargs=engine_kwargs)
    
    # 使用Cython引擎对分组后的数据应用lambda函数进行转换，将结果赋给expected
    expected = grouped.transform(lambda x: x + 1, engine="cython")
    
    # 断言result与expected是否相等
    tm.assert_equal(result, expected)
    
    # 使用Numba引擎和指定的引擎参数对分组后的数据应用func_2函数进行转换，将结果赋给result
    result = grouped.transform(func_2, engine="numba", engine_kwargs=engine_kwargs)
    
    # 使用Cython引擎对分组后的数据应用lambda函数进行转换，将结果赋给expected
    expected = grouped.transform(lambda x: x * 5, engine="cython")
    
    # 断言result与expected是否相等
    tm.assert_equal(result, expected)
    
    # 再次测试func_1函数，期望此次调用使用缓存
    result = grouped.transform(func_1, engine="numba", engine_kwargs=engine_kwargs)
    
    # 使用Cython引擎对分组后的数据应用lambda函数进行转换，将结果赋给expected
    expected = grouped.transform(lambda x: x + 1, engine="cython")
    
    # 断言result与expected是否相等
    tm.assert_equal(result, expected)
# 测试函数，验证在全局配置下的使用
def test_use_global_config():
    # 导入 pytest 并跳过如果缺少 numba 库
    pytest.importorskip("numba")

    # 定义一个简单的函数，对传入的值加一
    def func_1(values, index):
        return values + 1

    # 创建一个 DataFrame 对象，包含两列数据，分别是字符串和浮点数
    data = DataFrame(
        {0: ["a", "a", "b", "b", "a"], 1: [1.0, 2.0, 3.0, 4.0, 5.0]}, columns=[0, 1]
    )
    # 对数据按照第一列进行分组
    grouped = data.groupby(0)
    # 对分组后的数据应用 func_1 函数进行转换，使用 numba 引擎
    expected = grouped.transform(func_1, engine="numba")
    # 在使用 numba 引擎的上下文中再次应用 func_1 函数
    with option_context("compute.use_numba", True):
        result = grouped.transform(func_1, engine=None)
    # 验证预期结果与实际结果是否相等
    tm.assert_frame_equal(expected, result)


# TODO: Test more than just reductions (e.g. actually test transformations once we have
# 测试字符串比较 cython 引擎和 numba 引擎性能
@pytest.mark.parametrize(
    "agg_func", [["min", "max"], "min", {"B": ["min", "max"], "C": "sum"}]
)
def test_string_cython_vs_numba(agg_func, numba_supported_reductions):
    # 导入 pytest 并跳过如果缺少 numba 库
    pytest.importorskip("numba")
    # 从参数中获取 numba_supported_reductions
    agg_func, kwargs = numba_supported_reductions
    # 创建一个 DataFrame 对象，包含两列数据，分别是字符串和浮点数
    data = DataFrame(
        {0: ["a", "a", "b", "b", "a"], 1: [1.0, 2.0, 3.0, 4.0, 5.0]}, columns=[0, 1]
    )
    # 对数据按照第一列进行分组
    grouped = data.groupby(0)

    # 使用 numba 引擎进行转换，并验证结果与 cython 引擎的结果是否一致
    result = grouped.transform(agg_func, engine="numba", **kwargs)
    expected = grouped.transform(agg_func, engine="cython", **kwargs)
    tm.assert_frame_equal(result, expected)

    # 对分组后的第二列数据使用 numba 引擎进行转换，并验证结果与 cython 引擎的结果是否一致
    result = grouped[1].transform(agg_func, engine="numba", **kwargs)
    expected = grouped[1].transform(agg_func, engine="cython", **kwargs)
    tm.assert_series_equal(result, expected)


# 测试参数不被缓存的情况
def test_args_not_cached():
    # GH 41647
    # 导入 pytest 并跳过如果缺少 numba 库
    pytest.importorskip("numba")

    # 定义一个函数，计算最后 n 个元素的和
    def sum_last(values, index, n):
        return values[-n:].sum()

    # 创建一个 DataFrame 对象，包含两列数据，一列是 id，一列是 x
    df = DataFrame({"id": [0, 0, 1, 1], "x": [1, 1, 1, 1]})
    # 对数据按照 id 列进行分组，并选取 x 列
    grouped_x = df.groupby("id")["x"]
    # 使用 numba 引擎对分组后的数据进行转换，并验证结果是否与预期一致
    result = grouped_x.transform(sum_last, 1, engine="numba")
    expected = Series([1.0] * 4, name="x")
    tm.assert_series_equal(result, expected)

    # 使用 numba 引擎对分组后的数据进行转换，更改 n 的值，并验证结果是否与预期一致
    result = grouped_x.transform(sum_last, 2, engine="numba")
    expected = Series([2.0] * 4, name="x")
    tm.assert_series_equal(result, expected)


# 测试正确传递索引数据
def test_index_data_correctly_passed():
    # GH 43133
    # 导入 pytest 并跳过如果缺少 numba 库
    pytest.importorskip("numba")

    # 定义一个函数，返回索引减去 1 的结果
    def f(values, index):
        return index - 1

    # 创建一个 DataFrame 对象，包含两列数据，一列是分组标签 group，一列是数据 v，同时定义了索引
    df = DataFrame({"group": ["A", "A", "B"], "v": [4, 5, 6]}, index=[-1, -2, -3])
    # 使用 numba 引擎对分组后的数据进行转换，并验证结果是否与预期一致
    result = df.groupby("group").transform(f, engine="numba")
    expected = DataFrame([-2.0, -3.0, -4.0], columns=["v"], index=[-1, -2, -3])
    tm.assert_frame_equal(result, expected)


# 测试索引顺序的一致性保留
def test_index_order_consistency_preserved():
    # GH 57069
    # 导入 pytest 并跳过如果缺少 numba 库
    pytest.importorskip("numba")

    # 定义一个函数，直接返回值
    def f(values, index):
        return values

    # 创建一个 DataFrame 对象，包含两列数据，一列是 vals，一列是 group，并定义了逆序的索引
    df = DataFrame(
        {"vals": [0.0, 1.0, 2.0, 3.0], "group": [0, 1, 0, 1]}, index=range(3, -1, -1)
    )
    # 使用 numba 引擎对分组后的 vals 列数据进行转换，并验证结果是否与预期一致
    result = df.groupby("group")["vals"].transform(f, engine="numba")
    expected = Series([0.0, 1.0, 2.0, 3.0], index=range(3, -1, -1), name="vals")
    tm.assert_series_equal(result, expected)


# 测试引擎参数不被缓存的情况
def test_engine_kwargs_not_cached():
    # 如果用户传递了不同的引擎参数，不应返回相同的 jitted 函数
    pytest.importorskip("numba")
    nogil = True
    parallel = False
    nopython = True
    # 定义接受关键字参数的函数，返回值为 nogil + parallel + nopython 的结果
    def func_kwargs(values, index):
        return nogil + parallel + nopython

    # 定义引擎参数字典，包含 nopython、nogil 和 parallel 参数
    engine_kwargs = {"nopython": nopython, "nogil": nogil, "parallel": parallel}
    
    # 创建一个 DataFrame，包含一个名为 "value" 的列，初始值为 [0, 0, 0]
    df = DataFrame({"value": [0, 0, 0]})
    
    # 对 DataFrame 进行分组并应用 transform 函数，使用 numba 引擎和给定的引擎参数
    result = df.groupby(level=0).transform(
        func_kwargs, engine="numba", engine_kwargs=engine_kwargs
    )
    
    # 创建预期的 DataFrame，包含一个名为 "value" 的列，值为 [2.0, 2.0, 2.0]
    expected = DataFrame({"value": [2.0, 2.0, 2.0]})
    
    # 断言结果 DataFrame 和预期 DataFrame 相等
    tm.assert_frame_equal(result, expected)

    # 将 nogil 设置为 False
    nogil = False
    
    # 更新引擎参数字典，将 nogil 参数更新为 False
    engine_kwargs = {"nopython": nopython, "nogil": nogil, "parallel": parallel}
    
    # 再次对 DataFrame 进行分组并应用 transform 函数，使用 numba 引擎和更新后的引擎参数
    result = df.groupby(level=0).transform(
        func_kwargs, engine="numba", engine_kwargs=engine_kwargs
    )
    
    # 创建新的预期 DataFrame，包含一个名为 "value" 的列，值为 [1.0, 1.0, 1.0]
    expected = DataFrame({"value": [1.0, 1.0, 1.0]})
    
    # 断言更新后的结果 DataFrame 和新的预期 DataFrame 相等
    tm.assert_frame_equal(result, expected)
@pytest.mark.filterwarnings("ignore")
# 标记：忽略所有警告，用于测试函数
def test_multiindex_one_key(nogil, parallel, nopython):
    # 引入pytest并检查是否能导入numba库，如果不能则跳过测试
    pytest.importorskip("numba")

    # 定义一个使用numba的函数
    def numba_func(values, index):
        return 1

    # 创建一个包含单一索引的DataFrame
    df = DataFrame([{"A": 1, "B": 2, "C": 3}]).set_index(["A", "B"])

    # 设置引擎的关键字参数
    engine_kwargs = {"nopython": nopython, "nogil": nogil, "parallel": parallel}

    # 对DataFrame进行分组并应用transform操作，使用numba引擎执行
    result = df.groupby("A").transform(
        numba_func, engine="numba", engine_kwargs=engine_kwargs
    )

    # 期望的结果DataFrame
    expected = DataFrame([{"A": 1, "B": 2, "C": 1.0}]).set_index(["A", "B"])

    # 断言两个DataFrame是否相等
    tm.assert_frame_equal(result, expected)


# 定义测试函数，用于测试多重索引多个键不支持的情况
def test_multiindex_multi_key_not_supported(nogil, parallel, nopython):
    # 引入pytest并检查是否能导入numba库，如果不能则跳过测试
    pytest.importorskip("numba")

    # 定义一个使用numba的函数
    def numba_func(values, index):
        return 1

    # 创建一个包含单一索引的DataFrame
    df = DataFrame([{"A": 1, "B": 2, "C": 3}]).set_index(["A", "B"])

    # 设置引擎的关键字参数
    engine_kwargs = {"nopython": nopython, "nogil": nogil, "parallel": parallel}

    # 使用pytest来确保某个特定异常被抛出
    with pytest.raises(NotImplementedError, match="more than 1 grouping labels"):
        df.groupby(["A", "B"]).transform(
            numba_func, engine="numba", engine_kwargs=engine_kwargs
        )


# 定义测试函数，比较numba和cython引擎在多标签下的聚合行为
def test_multilabel_numba_vs_cython(numba_supported_reductions):
    # 引入pytest并检查是否能导入numba库，如果不能则跳过测试
    pytest.importorskip("numba")

    # 从输入参数中获取numba支持的聚合函数和其关键字参数
    reduction, kwargs = numba_supported_reductions

    # 创建一个包含多标签的DataFrame
    df = DataFrame(
        {
            "A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
            "B": ["one", "one", "two", "three", "two", "two", "one", "three"],
            "C": np.random.default_rng(2).standard_normal(8),
            "D": np.random.default_rng(2).standard_normal(8),
        }
    )

    # 对DataFrame进行分组
    gb = df.groupby(["A", "B"])

    # 使用numba引擎执行transform操作
    res_agg = gb.transform(reduction, engine="numba", **kwargs)

    # 使用cython引擎执行transform操作
    expected_agg = gb.transform(reduction, engine="cython", **kwargs)

    # 断言两个DataFrame是否相等
    tm.assert_frame_equal(res_agg, expected_agg)


# 定义测试函数，比较numba和cython引擎在自定义函数下的行为
def test_multilabel_udf_numba_vs_cython():
    # 引入pytest并检查是否能导入numba库，如果不能则跳过测试
    pytest.importorskip("numba")

    # 创建一个包含多标签的DataFrame
    df = DataFrame(
        {
            "A": ["foo", "bar", "foo", "bar", "foo", "bar", "foo", "foo"],
            "B": ["one", "one", "two", "three", "two", "two", "one", "three"],
            "C": np.random.default_rng(2).standard_normal(8),
            "D": np.random.default_rng(2).standard_normal(8),
        }
    )

    # 对DataFrame进行分组
    gb = df.groupby(["A", "B"])

    # 使用numba引擎执行transform操作，传入自定义函数作为参数
    result = gb.transform(
        lambda values, index: (values - values.min()) / (values.max() - values.min()),
        engine="numba",
    )

    # 使用cython引擎执行transform操作，传入自定义函数作为参数
    expected = gb.transform(
        lambda x: (x - x.min()) / (x.max() - x.min()), engine="cython"
    )

    # 断言两个DataFrame是否相等
    tm.assert_frame_equal(result, expected)
```