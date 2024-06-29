# `D:\src\scipysrc\pandas\pandas\tests\window\test_numba.py`

```
import numpy as np
import pytest

# 从 pandas.errors 中导入 NumbaUtilError 异常类
from pandas.errors import NumbaUtilError
# 导入 pandas.util._test_decorators 模块，重命名为 td
import pandas.util._test_decorators as td

# 从 pandas 中导入 DataFrame, Series, option_context, to_datetime 函数
from pandas import (
    DataFrame,
    Series,
    option_context,
    to_datetime,
)
# 导入 pandas._testing 模块，重命名为 tm
import pandas._testing as tm

# 给当前模块添加 pytestmark 标记，表示只在单CPU上运行
pytestmark = pytest.mark.single_cpu

# 用于参数化的 pytest fixture，参数为 "single" 和 "table"
@pytest.fixture(params=["single", "table"])
def method(request):
    """method keyword in rolling/expanding/ewm constructor"""
    return request.param

# 用于参数化的 pytest fixture，提供一组算术操作符和参数字典
@pytest.fixture(
    params=[
        ["sum", {}],
        ["mean", {}],
        ["median", {}],
        ["max", {}],
        ["min", {}],
        ["var", {}],
        ["var", {"ddof": 0}],
        ["std", {}],
        ["std", {"ddof": 0}],
    ]
)
def arithmetic_numba_supported_operators(request):
    return request.param

# 标记，如果没有安装 numba，则跳过这个测试
@td.skip_if_no("numba")
# 在这个类中忽略警告信息
@pytest.mark.filterwarnings("ignore")
# 定义一个测试类 TestEngine
class TestEngine:
    # 参数化测试方法 test_numba_vs_cython_apply，参数为 jit=True/False
    @pytest.mark.parametrize("jit", [True, False])
    def test_numba_vs_cython_apply(self, jit, nogil, parallel, nopython, center, step):
        # 定义一个函数 f，接受一个数组 x 和可变数量的参数 args
        def f(x, *args):
            arg_sum = 0
            # 将 args 中的参数累加到 arg_sum 中
            for arg in args:
                arg_sum += arg
            # 返回 x 的平均值加上 arg_sum
            return np.mean(x) + arg_sum

        # 如果 jit=True，则使用 numba 对函数 f 进行编译优化
        if jit:
            import numba
            f = numba.jit(f)

        # 定义引擎参数字典
        engine_kwargs = {"nogil": nogil, "parallel": parallel, "nopython": nopython}
        args = (2,)  # 定义参数 args 为元组 (2,)

        s = Series(range(10))  # 创建一个 Series 对象
        # 使用 numba 引擎计算滚动函数，raw=True 表示原始数据输入
        result = s.rolling(2, center=center, step=step).apply(
            f, args=args, engine="numba", engine_kwargs=engine_kwargs, raw=True
        )
        # 使用 cython 引擎计算滚动函数，raw=True 表示原始数据输入
        expected = s.rolling(2, center=center, step=step).apply(
            f, engine="cython", args=args, raw=True
        )
        # 断言两个 Series 对象相等
        tm.assert_series_equal(result, expected)

    # 参数化测试方法 test_numba_min_periods
    def test_numba_min_periods(self):
        # GH 58868
        # 定义一个函数 last_row，参数为 x，断言 x 的长度为 3，返回 x 的最后一个元素
        def last_row(x):
            assert len(x) == 3
            return x[-1]

        # 创建一个 DataFrame 对象 df
        df = DataFrame([[1, 2], [3, 4], [5, 6], [7, 8]])

        # 使用 numba 引擎计算滚动函数，设置最小期数为 3
        result = df.rolling(3, method="table", min_periods=3).apply(
            last_row, raw=True, engine="numba"
        )

        # 期望的结果 DataFrame 对象
        expected = DataFrame([[np.nan, np.nan], [np.nan, np.nan], [5, 6], [7, 8]])
        # 断言两个 DataFrame 对象相等
        tm.assert_frame_equal(result, expected)

    # 参数化测试方法 test_numba_vs_cython_rolling_methods，参数化 data
    @pytest.mark.parametrize(
        "data",
        [
            DataFrame(np.eye(5)),
            DataFrame(
                [
                    [5, 7, 7, 7, np.nan, np.inf, 4, 3, 3, 3],
                    [5, 7, 7, 7, np.nan, np.inf, 7, 3, 3, 3],
                    [np.nan, np.nan, 5, 6, 7, 5, 5, 5, 5, 5],
                ]
            ).T,
            Series(range(5), name="foo"),
            Series([20, 10, 10, np.inf, 1, 1, 2, 3]),
            Series([20, 10, 10, np.nan, 10, 1, 2, 3]),
        ],
    )
    def test_numba_vs_cython_rolling_methods(
        self,
        data,
        nogil,
        parallel,
        nopython,
        arithmetic_numba_supported_operators,
        step,
    ):
    ):
        # 从参数中获取支持的算术运算方法和关键字参数
        method, kwargs = arithmetic_numba_supported_operators

        # 设置引擎参数字典，包括 nogil, parallel, nopython 参数
        engine_kwargs = {"nogil": nogil, "parallel": parallel, "nopython": nopython}

        # 使用 rolling 方法创建一个窗口对象，窗口大小为 3，步长为 step
        roll = data.rolling(3, step=step)
        # 调用窗口对象的特定方法，使用 numba 引擎执行，传入引擎参数和其他关键字参数
        result = getattr(roll, method)(
            engine="numba", engine_kwargs=engine_kwargs, **kwargs
        )
        # 调用窗口对象的特定方法，使用 cython 引擎执行，传入其他关键字参数
        expected = getattr(roll, method)(engine="cython", **kwargs)
        # 使用测试框架的方法验证结果和预期相等
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize(
        "data", [DataFrame(np.eye(5)), Series(range(5), name="foo")]
    )
    def test_numba_vs_cython_expanding_methods(
        self, data, nogil, parallel, nopython, arithmetic_numba_supported_operators
    ):
        # 从参数中获取支持的算术运算方法和关键字参数
        method, kwargs = arithmetic_numba_supported_operators

        # 设置引擎参数字典，包括 nogil, parallel, nopython 参数
        engine_kwargs = {"nogil": nogil, "parallel": parallel, "nopython": nopython}

        # 创建一个 DataFrame 对象，使用单位矩阵进行初始化
        data = DataFrame(np.eye(5))
        # 使用 expanding 方法创建一个展开对象
        expand = data.expanding()
        # 调用展开对象的特定方法，使用 numba 引擎执行，传入引擎参数和其他关键字参数
        result = getattr(expand, method)(
            engine="numba", engine_kwargs=engine_kwargs, **kwargs
        )
        # 调用展开对象的特定方法，使用 cython 引擎执行，传入其他关键字参数
        expected = getattr(expand, method)(engine="cython", **kwargs)
        # 使用测试框架的方法验证结果和预期相等
        tm.assert_equal(result, expected)

    @pytest.mark.parametrize("jit", [True, False])
    def test_cache_apply(self, jit, nogil, parallel, nopython, step):
        # 测试函数是否正确缓存，根据是否启用 JIT 编译来切换函数

        # 定义函数 func_1，计算输入数据的平均值再加 4
        def func_1(x):
            return np.mean(x) + 4

        # 定义函数 func_2，计算输入数据的标准差再乘以 5
        def func_2(x):
            return np.std(x) * 5

        # 如果启用 JIT 编译，则使用 numba 模块对 func_1 和 func_2 进行编译
        if jit:
            import numba

            func_1 = numba.jit(func_1)
            func_2 = numba.jit(func_2)

        # 设置引擎参数字典，包括 nogil, parallel, nopython 参数
        engine_kwargs = {"nogil": nogil, "parallel": parallel, "nopython": nopython}

        # 使用 Series 对象创建一个滚动对象，窗口大小为 2，步长为 step
        roll = Series(range(10)).rolling(2, step=step)
        # 调用滚动对象的 apply 方法，应用 func_1 函数，使用 numba 引擎执行，传入引擎参数和其他关键字参数，原始数据传入 raw 参数
        result = roll.apply(
            func_1, engine="numba", engine_kwargs=engine_kwargs, raw=True
        )
        # 调用滚动对象的 apply 方法，应用 func_1 函数，使用 cython 引擎执行，原始数据传入 raw 参数
        expected = roll.apply(func_1, engine="cython", raw=True)
        # 使用测试框架的方法验证结果和预期相等
        tm.assert_series_equal(result, expected)

        # 再次调用滚动对象的 apply 方法，应用 func_2 函数，使用 numba 引擎执行，传入引擎参数和其他关键字参数，原始数据传入 raw 参数
        result = roll.apply(
            func_2, engine="numba", engine_kwargs=engine_kwargs, raw=True
        )
        # 再次调用滚动对象的 apply 方法，应用 func_2 函数，使用 cython 引擎执行，原始数据传入 raw 参数
        expected = roll.apply(func_2, engine="cython", raw=True)
        # 使用测试框架的方法验证结果和预期相等
        tm.assert_series_equal(result, expected)
        # 这次运行应该使用缓存的 func_1 函数
        result = roll.apply(
            func_1, engine="numba", engine_kwargs=engine_kwargs, raw=True
        )
        # 再次调用滚动对象的 apply 方法，应用 func_1 函数，使用 cython 引擎执行，原始数据传入 raw 参数
        expected = roll.apply(func_1, engine="cython", raw=True)
        # 使用测试框架的方法验证结果和预期相等
        tm.assert_series_equal(result, expected)

    @pytest.mark.parametrize(
        "window,window_kwargs",
        [
            ["rolling", {"window": 3, "min_periods": 0}],
            ["expanding", {}],
        ],
    )
    def test_dont_cache_args(
        self, window, window_kwargs, nogil, parallel, nopython, method
        # 这是一个参数化测试函数，用于测试不缓存参数的情况
    ):
        # GH 42287
        # 定义一个名为 add 的函数，用于将 values 数组的总和与 x 相加
        def add(values, x):
            return np.sum(values) + x

        # 设置引擎参数字典，包括是否禁用 Python 解释器、是否允许并行处理
        engine_kwargs = {"nopython": nopython, "nogil": nogil, "parallel": parallel}
        # 创建一个包含初始值为 0 的 DataFrame 对象
        df = DataFrame({"value": [0, 0, 0]})
        # 调用滚动窗口函数，并应用之前定义的 add 函数，传入原始数据、指定引擎为 numba，并传入额外参数 1
        result = getattr(df, window)(method=method, **window_kwargs).apply(
            add, raw=True, engine="numba", engine_kwargs=engine_kwargs, args=(1,)
        )
        # 创建一个期望的 DataFrame 对象，其中每个值都为 1.0
        expected = DataFrame({"value": [1.0, 1.0, 1.0]})
        # 断言结果 DataFrame 与期望 DataFrame 相等
        tm.assert_frame_equal(result, expected)

        # 重新调用滚动窗口函数，应用 add 函数和其他参数，此时传入参数 2
        result = getattr(df, window)(method=method, **window_kwargs).apply(
            add, raw=True, engine="numba", engine_kwargs=engine_kwargs, args=(2,)
        )
        # 创建另一个期望的 DataFrame 对象，其中每个值都为 2.0
        expected = DataFrame({"value": [2.0, 2.0, 2.0]})
        # 再次断言结果 DataFrame 与期望 DataFrame 相等
        tm.assert_frame_equal(result, expected)

    def test_dont_cache_engine_kwargs(self):
        # 如果用户传递了不同的 engine_kwargs，则不应返回相同的编译函数
        nogil = False
        parallel = True
        nopython = True

        # 定义一个函数 func，返回 nogil、parallel 和 nopython 的总和
        def func(x):
            return nogil + parallel + nopython

        # 设置引擎参数字典，包括是否禁用 Python 解释器、是否允许并行处理
        engine_kwargs = {"nopython": nopython, "nogil": nogil, "parallel": parallel}
        # 创建一个包含初始值为 0 的 DataFrame 对象
        df = DataFrame({"value": [0, 0, 0]})
        # 调用滚动窗口函数，并应用之前定义的 func 函数，传入原始数据、指定引擎为 numba，并传入 engine_kwargs
        result = df.rolling(1).apply(
            func, raw=True, engine="numba", engine_kwargs=engine_kwargs
        )
        # 创建一个期望的 DataFrame 对象，其中每个值为 2.0，因为 nogil、parallel 和 nopython 均为 True
        expected = DataFrame({"value": [2.0, 2.0, 2.0]})
        # 断言结果 DataFrame 与期望 DataFrame 相等
        tm.assert_frame_equal(result, expected)

        # 将 parallel 设置为 False，更新引擎参数字典
        parallel = False
        engine_kwargs = {"nopython": nopython, "nogil": nogil, "parallel": parallel}
        # 再次调用滚动窗口函数，并应用 func 函数，传入相应参数
        result = df.rolling(1).apply(
            func, raw=True, engine="numba", engine_kwargs=engine_kwargs
        )
        # 创建另一个期望的 DataFrame 对象，其中每个值为 1.0，因为 parallel 现在为 False
        expected = DataFrame({"value": [1.0, 1.0, 1.0]})
        # 再次断言结果 DataFrame 与期望 DataFrame 相等
        tm.assert_frame_equal(result, expected)
# 装饰器，用于跳过测试（如果缺少"numba"）
@td.skip_if_no("numba")
# 定义测试类 TestEWM
class TestEWM:
    
    # 使用 pytest 的参数化装饰器，对 test_invalid_engine 方法进行参数化测试
    @pytest.mark.parametrize(
        "grouper", [lambda x: x, lambda x: x.groupby("A")], ids=["None", "groupby"]
    )
    @pytest.mark.parametrize("method", ["mean", "sum"])
    # 定义测试方法 test_invalid_engine，测试无效的引擎参数
    def test_invalid_engine(self, grouper, method):
        # 创建 DataFrame 对象 df
        df = DataFrame({"A": ["a", "b", "a", "b"], "B": range(4)})
        # 使用 pytest 的上下文管理器，期望抛出 ValueError 异常，且异常信息包含"engine must be either"
        with pytest.raises(ValueError, match="engine must be either"):
            # 获取 grouper(df).ewm(com=1.0) 对象，并调用其中的 method 方法，设置 engine="foo"
            getattr(grouper(df).ewm(com=1.0), method)(engine="foo")

    # 使用 pytest 的参数化装饰器，对 test_invalid_engine_kwargs 方法进行参数化测试
    @pytest.mark.parametrize(
        "grouper", [lambda x: x, lambda x: x.groupby("A")], ids=["None", "groupby"]
    )
    @pytest.mark.parametrize("method", ["mean", "sum"])
    # 定义测试方法 test_invalid_engine_kwargs，测试无效的引擎参数和引擎关键字参数
    def test_invalid_engine_kwargs(self, grouper, method):
        # 创建 DataFrame 对象 df
        df = DataFrame({"A": ["a", "b", "a", "b"], "B": range(4)})
        # 使用 pytest 的上下文管理器，期望抛出 ValueError 异常，且异常信息包含"cython engine does not"
        with pytest.raises(ValueError, match="cython engine does not"):
            # 获取 grouper(df).ewm(com=1.0) 对象，并调用其中的 method 方法，设置 engine="cython" 和 engine_kwargs={"nopython": True}
            getattr(grouper(df).ewm(com=1.0), method)(
                engine="cython", engine_kwargs={"nopython": True}
            )

    # 使用 pytest 的参数化装饰器，对 test_cython_vs_numba 方法进行参数化测试
    @pytest.mark.parametrize("grouper", ["None", "groupby"])
    @pytest.mark.parametrize("method", ["mean", "sum"])
    # 定义测试方法 test_cython_vs_numba，测试 cython 引擎与 numba 引擎的比较
    def test_cython_vs_numba(
        self, grouper, method, nogil, parallel, nopython, ignore_na, adjust
    ):
        # 创建 DataFrame 对象 df
        df = DataFrame({"B": range(4)})
        # 根据 grouper 参数，定义 grouper 函数
        if grouper == "None":
            grouper = lambda x: x
        else:
            df["A"] = ["a", "b", "a", "b"]
            grouper = lambda x: x.groupby("A")
        # 如果 method 为 "sum"，则设置 adjust=True
        if method == "sum":
            adjust = True
        # 创建 ewm 对象，根据 grouper(df).ewm 的调用参数
        ewm = grouper(df).ewm(com=1.0, adjust=adjust, ignore_na=ignore_na)

        # 定义引擎关键字参数
        engine_kwargs = {"nogil": nogil, "parallel": parallel, "nopython": nopython}
        # 调用 getattr(ewm, method)(...)，设置 engine="numba" 和 engine_kwargs=engine_kwargs
        result = getattr(ewm, method)(engine="numba", engine_kwargs=engine_kwargs)
        # 调用 getattr(ewm, method)(...)，设置 engine="cython"
        expected = getattr(ewm, method)(engine="cython")

        # 使用 tm.assert_frame_equal 检查 result 和 expected 的一致性
        tm.assert_frame_equal(result, expected)

    # 使用 pytest 的参数化装饰器，对 test_cython_vs_numba_times 方法进行参数化测试
    @pytest.mark.parametrize("grouper", ["None", "groupby"])
    # 定义测试方法 test_cython_vs_numba_times，测试 cython 引擎与 numba 引擎的比较（带有时间参数）
    def test_cython_vs_numba_times(self, grouper, nogil, parallel, nopython, ignore_na):
        # GH 40951

        # 创建 DataFrame 对象 df
        df = DataFrame({"B": [0, 0, 1, 1, 2, 2]})
        # 根据 grouper 参数，定义 grouper 函数
        if grouper == "None":
            grouper = lambda x: x
        else:
            grouper = lambda x: x.groupby("A")
            df["A"] = ["a", "b", "a", "b", "b", "a"]

        # 定义半衰期参数 halflife 和时间参数 times
        halflife = "23 days"
        times = to_datetime(
            [
                "2020-01-01",
                "2020-01-01",
                "2020-01-02",
                "2020-01-10",
                "2020-02-23",
                "2020-01-03",
            ]
        )
        # 创建 ewm 对象，根据 grouper(df).ewm 的调用参数
        ewm = grouper(df).ewm(
            halflife=halflife, adjust=True, ignore_na=ignore_na, times=times
        )

        # 定义引擎关键字参数
        engine_kwargs = {"nogil": nogil, "parallel": parallel, "nopython": nopython}

        # 调用 ewm.mean(...)，设置 engine="numba" 和 engine_kwargs=engine_kwargs
        result = ewm.mean(engine="numba", engine_kwargs=engine_kwargs)
        # 调用 ewm.mean(...)，设置 engine="cython"
        expected = ewm.mean(engine="cython")

        # 使用 tm.assert_frame_equal 检查 result 和 expected 的一致性
        tm.assert_frame_equal(result, expected)


# 装饰器，用于跳过测试（如果缺少"numba"）
@td.skip_if_no("numba")
# 定义函数 test_use_global_config，测试使用全局配置
def test_use_global_config():
    # 定义函数 f，计算平均值并加上常数 2
    def f(x):
        return np.mean(x) + 2

    # 创建 Series 对象 s，包含整数范围为 0 到 9
    s = Series(range(10))
    # 使用上下文管理器设置选项 "compute.use_numba" 为 True，表明希望使用 Numba 进行计算加速
    with option_context("compute.use_numba", True):
        # 对序列 s 进行滚动窗口为2的滚动操作，并应用函数 f
        result = s.rolling(2).apply(f, engine=None, raw=True)
    # 期望的结果是使用 Numba 引擎对序列 s 进行相同的滚动操作
    expected = s.rolling(2).apply(f, engine="numba", raw=True)
    # 使用测试工具比较两个序列，确保它们相等
    tm.assert_series_equal(expected, result)
# 如果没有安装 "numba" 模块，则跳过此测试
@td.skip_if_no("numba")
# 测试在使用 "numba" 引擎时传递无效关键字参数是否引发 NumbaUtilError 异常
def test_invalid_kwargs_nopython():
    # 使用 pytest.raises 断言捕获 NumbaUtilError 异常，匹配特定错误消息
    with pytest.raises(
        NumbaUtilError, match="numba does not support keyword-only arguments"
    ):
        # 创建 Series 对象，并在其上执行 rolling 和 apply 操作
        Series(range(1)).rolling(1).apply(
            lambda x: x, kwargs={"a": 1}, engine="numba", raw=True
        )


# 如果没有安装 "numba" 模块，则跳过此测试
@td.skip_if_no("numba")
# 标记此类测试为慢速测试，并在运行时忽略警告
@pytest.mark.slow
@pytest.mark.filterwarnings("ignore")
# 当 parallel=True 且函数无法由 Numba 并行化时，忽略警告
# Filter warnings when parallel=True and the function can't be parallelized by Numba
class TestTableMethod:
    # 测试在使用 "table" 方法时，对 Series 对象执行方法时是否引发 ValueError 异常
    def test_table_series_valueerror(self):
        # 定义一个简单的函数 f，用于在 Series 对象上执行 rolling 和 apply 操作
        def f(x):
            return np.sum(x, axis=0) + 1

        # 使用 pytest.raises 断言捕获 ValueError 异常，匹配特定错误消息
        with pytest.raises(
            ValueError, match="method='table' not applicable for Series objects."
        ):
            # 创建 Series 对象，并在其上执行 rolling 和 apply 操作
            Series(range(1)).rolling(1, method="table").apply(
                f, engine="numba", raw=True
            )

    # 测试在使用 "table" 方法时，对 DataFrame 对象执行各种 rolling 方法的正确性
    def test_table_method_rolling_methods(
        self,
        nogil,
        parallel,
        nopython,
        arithmetic_numba_supported_operators,
        step,
    ):
        method, kwargs = arithmetic_numba_supported_operators

        # 定义引擎参数字典
        engine_kwargs = {"nogil": nogil, "parallel": parallel, "nopython": nopython}

        # 创建一个 DataFrame 对象
        df = DataFrame(np.eye(3))
        
        # 创建一个滚动窗口对象 roll_table，使用 "table" 方法
        roll_table = df.rolling(2, method="table", min_periods=0, step=step)
        
        # 如果方法是 "var" 或 "std"，则预期抛出 NotImplementedError 异常
        if method in ("var", "std"):
            with pytest.raises(NotImplementedError, match=f"{method} not supported"):
                # 调用 roll_table 对象的特定方法，使用 numba 引擎和给定的参数
                getattr(roll_table, method)(
                    engine_kwargs=engine_kwargs, engine="numba", **kwargs
                )
        else:
            # 创建一个单行窗口对象 roll_single，使用 "single" 方法
            roll_single = df.rolling(2, method="single", min_periods=0, step=step)
            
            # 调用 roll_table 对象的特定方法，使用 numba 引擎和给定的参数，并比较结果
            result = getattr(roll_table, method)(
                engine_kwargs=engine_kwargs, engine="numba", **kwargs
            )
            # 预期的结果，使用 roll_single 对象的相同方法和参数
            expected = getattr(roll_single, method)(
                engine_kwargs=engine_kwargs, engine="numba", **kwargs
            )
            # 使用 assert_frame_equal 检查结果和预期是否相等
            tm.assert_frame_equal(result, expected)

    # 测试在使用 "table" 方法时，对 DataFrame 对象执行 apply 操作的正确性
    def test_table_method_rolling_apply(self, nogil, parallel, nopython, step):
        # 定义引擎参数字典
        engine_kwargs = {"nogil": nogil, "parallel": parallel, "nopython": nopython}

        # 定义一个简单的函数 f，用于在 DataFrame 对象上执行 rolling 和 apply 操作
        def f(x):
            return np.sum(x, axis=0) + 1

        # 创建一个 DataFrame 对象
        df = DataFrame(np.eye(3))
        
        # 调用 rolling 和 apply 操作，使用 "table" 方法和 numba 引擎，并比较结果
        result = df.rolling(2, method="table", min_periods=0, step=step).apply(
            f, raw=True, engine_kwargs=engine_kwargs, engine="numba"
        )
        # 预期的结果，使用 "single" 方法和相同的参数
        expected = df.rolling(2, method="single", min_periods=0, step=step).apply(
            f, raw=True, engine_kwargs=engine_kwargs, engine="numba"
        )
        # 使用 assert_frame_equal 检查结果和预期是否相等
        tm.assert_frame_equal(result, expected)
    # 定义一个测试方法，用于测试 rolling 方法中的 weighted_mean 函数
    def test_table_method_rolling_weighted_mean(self, step):
        # 定义一个计算加权平均的函数
        def weighted_mean(x):
            # 创建一个全为 1 的数组，将前两列乘以权重后的结果加入数组
            arr = np.ones((1, x.shape[1]))
            arr[:, :2] = (x[:, :2] * x[:, 2]).sum(axis=0) / x[:, 2].sum()
            return arr

        # 创建一个测试数据框
        df = DataFrame([[1, 2, 0.6], [2, 3, 0.4], [3, 4, 0.2], [4, 5, 0.7]])
        # 使用 rolling 方法应用 weighted_mean 函数，生成结果
        result = df.rolling(2, method="table", min_periods=0, step=step).apply(
            weighted_mean, raw=True, engine="numba"
        )
        # 创建预期结果数据框
        expected = DataFrame(
            [
                [1.0, 2.0, 1.0],
                [1.8, 2.0, 1.0],
                [3.333333, 2.333333, 1.0],
                [1.555556, 7, 1.0],
            ]
        )[::step]
        # 使用测试框架检查结果是否符合预期
        tm.assert_frame_equal(result, expected)

    # 定义一个测试方法，用于测试 expanding 方法中的 apply 函数
    def test_table_method_expanding_apply(self, nogil, parallel, nopython):
        # 定义一个简单的函数，对输入数组进行求和并加一
        def f(x):
            return np.sum(x, axis=0) + 1

        # 创建一个测试数据框
        df = DataFrame(np.eye(3))
        # 使用 expanding 方法应用函数 f，生成结果
        result = df.expanding(method="table").apply(
            f, raw=True, engine_kwargs={"nogil": nogil, "parallel": parallel, "nopython": nopython}, engine="numba"
        )
        # 创建使用 expanding 方法应用函数 f 的预期结果数据框
        expected = df.expanding(method="single").apply(
            f, raw=True, engine_kwargs={"nogil": nogil, "parallel": parallel, "nopython": nopython}, engine="numba"
        )
        # 使用测试框架检查结果是否符合预期
        tm.assert_frame_equal(result, expected)

    # 定义一个测试方法，用于测试 expanding 方法中的方法调用
    def test_table_method_expanding_methods(
        self, nogil, parallel, nopython, arithmetic_numba_supported_operators
    ):
        # 从测试参数中获取支持的算术运算方法和参数
        method, kwargs = arithmetic_numba_supported_operators

        engine_kwargs = {"nogil": nogil, "parallel": parallel, "nopython": nopython}

        # 创建一个测试数据框
        df = DataFrame(np.eye(3))
        # 使用 expanding 方法创建一个扩展对象
        expand_table = df.expanding(method="table")
        # 如果方法是 "var" 或 "std"，则预期引发 NotImplementedError
        if method in ("var", "std"):
            with pytest.raises(NotImplementedError, match=f"{method} not supported"):
                getattr(expand_table, method)(
                    engine_kwargs=engine_kwargs, engine="numba", **kwargs
                )
        else:
            # 否则，获取单独 expanding 方法的预期结果
            expand_single = df.expanding(method="single")
            result = getattr(expand_table, method)(
                engine_kwargs=engine_kwargs, engine="numba", **kwargs
            )
            expected = getattr(expand_single, method)(
                engine_kwargs=engine_kwargs, engine="numba", **kwargs
            )
            # 使用测试框架检查结果是否符合预期
            tm.assert_frame_equal(result, expected)

    # 定义一个参数化测试方法，测试 ewm 方法中的不同数据和方法
    @pytest.mark.parametrize("data", [np.eye(3), np.ones((2, 3)), np.ones((3, 2))])
    @pytest.mark.parametrize("method", ["mean", "sum"])
    def test_table_method_ewm(self, data, method, nogil, parallel, nopython):
        engine_kwargs = {"nogil": nogil, "parallel": parallel, "nopython": nopython}

        # 创建一个测试数据框
        df = DataFrame(data)

        # 使用 ewm 方法应用指定方法，生成结果
        result = getattr(df.ewm(com=1, method="table"), method)(
            engine_kwargs=engine_kwargs, engine="numba"
        )
        # 创建使用单独 ewm 方法应用指定方法的预期结果数据框
        expected = getattr(df.ewm(com=1, method="single"), method)(
            engine_kwargs=engine_kwargs, engine="numba"
        )
        # 使用测试框架检查结果是否符合预期
        tm.assert_frame_equal(result, expected)
# 在测试函数上使用装饰器，仅在 "numba" 模块存在时才运行该测试，否则跳过
@td.skip_if_no("numba")
# 定义一个测试函数，用于验证在指定条件下是否会产生警告
def test_npfunc_no_warnings():
    # 创建一个包含一列数据的 DataFrame
    df = DataFrame({"col1": [1, 2, 3, 4, 5]})
    # 使用上下文管理器确保在执行以下代码块期间不会产生警告
    with tm.assert_produces_warning(False):
        # 对 DataFrame 的 col1 列执行滚动窗口计算，应用 np.prod 函数
        # 使用 Numba 引擎进行原始计算，这可能会导致性能提升
        df.col1.rolling(2).apply(np.prod, raw=True, engine="numba")
```