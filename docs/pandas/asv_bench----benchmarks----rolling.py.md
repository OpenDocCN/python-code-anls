# `D:\src\scipysrc\pandas\asv_bench\benchmarks\rolling.py`

```
# 导入警告模块，用于捕获警告信息
import warnings

# 导入 NumPy 库并使用 np 别名
import numpy as np

# 导入 Pandas 库并使用 pd 别名
import pandas as pd


# 定义 Methods 类
class Methods:
    # 定义参数元组 params，包含不同方法需要的参数组合
    params = (
        ["DataFrame", "Series"],  # 数据结构：DataFrame 或 Series
        [("rolling", {"window": 10}), ("rolling", {"window": 1000}), ("expanding", {})],  # 滚动窗口设置
        ["int", "float"],  # 数据类型：整数或浮点数
        ["median", "mean", "max", "min", "std", "count", "skew", "kurt", "sum", "sem"],  # 方法：中位数、均值、最大值、最小值等
    )
    # 参数名称列表
    param_names = ["constructor", "window_kwargs", "dtype", "method"]

    # 设置方法，初始化对象
    def setup(self, constructor, window_kwargs, dtype, method):
        N = 10**5  # 数据量
        window, kwargs = window_kwargs  # 解构 window_kwargs 元组
        arr = (100 * np.random.random(N)).astype(dtype)  # 生成随机数组，指定数据类型
        obj = getattr(pd, constructor)(arr)  # 根据构造函数名构造 Pandas 对象
        self.window = getattr(obj, window)(**kwargs)  # 获取滚动窗口对象

    # 测试方法的执行时间
    def time_method(self, constructor, window_kwargs, dtype, method):
        getattr(self.window, method)()  # 调用窗口对象的特定方法

    # 测试方法的内存使用峰值
    def peakmem_method(self, constructor, window_kwargs, dtype, method):
        getattr(self.window, method)()  # 调用窗口对象的特定方法


# 定义 Apply 类
class Apply:
    # 定义参数元组 params，包含不同方法需要的参数组合
    params = (
        ["DataFrame", "Series"],  # 数据结构：DataFrame 或 Series
        [3, 300],  # 滚动窗口大小
        ["int", "float"],  # 数据类型：整数或浮点数
        [sum, np.sum, lambda x: np.sum(x) + 5],  # 应用的函数：求和、NumPy 求和、自定义函数
        [True, False],  # 是否使用原始数据
    )
    # 参数名称列表
    param_names = ["constructor", "window", "dtype", "function", "raw"]

    # 设置方法，初始化对象
    def setup(self, constructor, window, dtype, function, raw):
        N = 10**3  # 数据量
        arr = (100 * np.random.random(N)).astype(dtype)  # 生成随机数组，指定数据类型
        self.roll = getattr(pd, constructor)(arr).rolling(window)  # 创建滚动窗口对象

    # 测试滚动应用函数的执行时间
    def time_rolling(self, constructor, window, dtype, function, raw):
        self.roll.apply(function, raw=raw)  # 应用函数于滚动窗口


# 定义 NumbaEngineMethods 类
class NumbaEngineMethods:
    # 定义参数元组 params，包含不同方法需要的参数组合
    params = (
        ["DataFrame", "Series"],  # 数据结构：DataFrame 或 Series
        ["int", "float"],  # 数据类型：整数或浮点数
        [("rolling", {"window": 10}), ("expanding", {})],  # 滚动窗口设置
        ["sum", "max", "min", "median", "mean", "var", "std"],  # 方法：求和、最大值、最小值等
        [True, False],  # 是否并行执行
        [None, 100],  # 列数，若有的话
    )
    # 参数名称列表
    param_names = [
        "constructor",
        "dtype",
        "window_kwargs",
        "method",
        "parallel",
        "cols",
    ]

    # 设置方法，初始化对象
    def setup(self, constructor, dtype, window_kwargs, method, parallel, cols):
        N = 10**3  # 数据量
        window, kwargs = window_kwargs  # 解构 window_kwargs 元组
        shape = (N, cols) if cols is not None and constructor != "Series" else N  # 数据形状
        arr = (100 * np.random.random(shape)).astype(dtype)  # 生成随机数组，指定数据类型
        data = getattr(pd, constructor)(arr)  # 根据构造函数名构造 Pandas 对象

        # 预热缓存
        with warnings.catch_warnings(record=True):
            # 捕获可能的警告信息，如并行不适用于 1D 数据的情况
            self.window = getattr(data, window)(**kwargs)  # 获取滚动窗口对象
            getattr(self.window, method)(
                engine="numba", engine_kwargs={"parallel": parallel}  # 使用 Numba 引擎执行特定方法
            )

    # 测试方法的性能
    def test_method(self, constructor, dtype, window_kwargs, method, parallel, cols):
        with warnings.catch_warnings(record=True):
            getattr(self.window, method)(
                engine="numba", engine_kwargs={"parallel": parallel}  # 使用 Numba 引擎执行特定方法
            )
    # 参数列表，每个元素为一个参数的可能取值列表
    params = (
        ["DataFrame", "Series"],             # 第一个参数：数据结构的类型，可以是DataFrame或Series
        ["int", "float"],                    # 第二个参数：数据类型，可以是整数或浮点数
        [("rolling", {"window": 10}), ("expanding", {})],  # 第三个参数：窗口操作类型及其参数，rolling窗口大小为10，expanding无额外参数
        [np.sum, lambda x: np.sum(x) + 5],   # 第四个参数：函数对象或lambda表达式，用于数据操作
        [True, False],                       # 第五个参数：并行计算的布尔值，True或False
        [None, 100],                         # 第六个参数：列数或者空值（表示默认值）
    )
    # 参数名称列表，与params中的元素一一对应
    param_names = [
        "constructor",                      # 第一个参数的名称：数据结构类型
        "dtype",                            # 第二个参数的名称：数据类型
        "window_kwargs",                    # 第三个参数的名称：窗口操作参数
        "function",                         # 第四个参数的名称：操作函数
        "parallel",                         # 第五个参数的名称：并行计算标志
        "cols",                             # 第六个参数的名称：列数
    ]

    # 初始化方法，用于设置测试环境
    def setup(self, constructor, dtype, window_kwargs, function, parallel, cols):
        N = 10**3                              # 数据点数量
        window, kwargs = window_kwargs         # 解包窗口操作类型及其参数
        shape = (N, cols) if cols is not None and constructor != "Series" else N  # 根据条件确定数据形状
        arr = (100 * np.random.random(shape)).astype(dtype)  # 生成指定类型和形状的随机数组
        data = getattr(pd, constructor)(arr)   # 根据构造器类型，生成对应的pandas数据结构

        # 预热缓存
        with warnings.catch_warnings(record=True):
            # 捕获可能的警告信息，例如1D数据情况下不适用parallel=True
            self.window = getattr(data, window)(**kwargs)  # 根据窗口类型和参数执行操作，并将结果存储在self.window中
            self.window.apply(
                function, raw=True, engine="numba", engine_kwargs={"parallel": parallel}  # 应用指定函数及其参数到窗口对象上
            )

    # 测试方法，用于执行测试操作
    def test_method(self, constructor, dtype, window_kwargs, function, parallel, cols):
        with warnings.catch_warnings(record=True):
            self.window.apply(
                function, raw=True, engine="numba", engine_kwargs={"parallel": parallel}  # 应用指定函数及其参数到窗口对象上
            )
class EWMMethods:
    # 定义参数元组，包含数据结构类型、参数设置和数据类型
    params = (
        ["DataFrame", "Series"],  # 第一个元素：数据结构类型为 DataFrame 或 Series
        [
            ({"halflife": 10}, "mean"),    # 参数设置为 {"halflife": 10}，方法为 "mean"
            ({"halflife": 10}, "std"),     # 参数设置为 {"halflife": 10}，方法为 "std"
            ({"halflife": 1000}, "mean"),  # 参数设置为 {"halflife": 1000}，方法为 "mean"
            ({"halflife": 1000}, "std"),   # 参数设置为 {"halflife": 1000}，方法为 "std"
            (
                {
                    "halflife": "1 Day",
                    "times": pd.date_range("1900", periods=10**5, freq="23s"),
                },
                "mean",
            ),  # 参数设置包含 "halflife" 和 "times"，方法为 "mean"
        ],
        ["int", "float"],  # 第三个元素：数据类型为 int 或 float
    )
    # 参数名称列表
    param_names = ["constructor", "kwargs_method", "dtype"]

    # 初始化方法，设置数据结构、参数方法和数据类型
    def setup(self, constructor, kwargs_method, dtype):
        N = 10**5
        kwargs, method = kwargs_method
        arr = (100 * np.random.random(N)).astype(dtype)
        self.method = method  # 设置实例变量 method
        self.ewm = getattr(pd, constructor)(arr).ewm(**kwargs)  # 创建指数加权移动平均对象

    # 测试指数加权移动平均方法的执行时间
    def time_ewm(self, constructor, kwargs_method, dtype):
        getattr(self.ewm, self.method)()  # 调用指定方法


class VariableWindowMethods(Methods):
    # 定义参数元组，包含数据结构类型、窗口长度、数据类型和统计方法
    params = (
        ["DataFrame", "Series"],  # 第一个元素：数据结构类型为 DataFrame 或 Series
        ["50s", "1h", "1d"],      # 第二个元素：窗口长度为 "50s", "1h" 或 "1d"
        ["int", "float"],        # 第三个元素：数据类型为 int 或 float
        ["median", "mean", "max", "min", "std", "count", "skew", "kurt", "sum", "sem"],  # 第四个元素：统计方法
    )
    # 参数名称列表
    param_names = ["constructor", "window", "dtype", "method"]

    # 初始化方法，设置数据结构、窗口长度、数据类型和统计方法
    def setup(self, constructor, window, dtype, method):
        N = 10**5
        arr = (100 * np.random.random(N)).astype(dtype)
        index = pd.date_range("2017-01-01", periods=N, freq="5s")
        self.window = getattr(pd, constructor)(arr, index=index).rolling(window)  # 创建滚动窗口对象


class Pairwise:
    # 定义参数元组，包含窗口设置、方法和是否成对操作
    params = (
        [({"window": 10}, "rolling"), ({"window": 1000}, "rolling"), ({}, "expanding")],  # 第一个元素：窗口设置和方法
        ["corr", "cov"],   # 第二个元素：方法为 "corr" 或 "cov"
        [True, False],     # 第三个元素：是否成对操作，True 或 False
    )
    # 参数名称列表
    param_names = ["window_kwargs", "method", "pairwise"]

    # 初始化方法，设置窗口设置、方法和是否成对操作
    def setup(self, kwargs_window, method, pairwise):
        N = 10**4
        n_groups = 20
        kwargs, window = kwargs_window
        groups = [i for _ in range(N // n_groups) for i in range(n_groups)]
        arr = np.random.random(N)
        self.df = pd.DataFrame(arr)  # 创建数据帧
        self.window = getattr(self.df, window)(**kwargs)  # 创建滚动窗口对象
        self.window_group = getattr(
            pd.DataFrame({"A": groups, "B": arr}).groupby("A"), window
        )(**kwargs)  # 对分组后的数据帧应用滚动窗口

    # 测试成对方法的执行时间
    def time_pairwise(self, kwargs_window, method, pairwise):
        getattr(self.window, method)(self.df, pairwise=pairwise)  # 调用成对方法

    # 测试分组后的方法的执行时间
    def time_groupby(self, kwargs_window, method, pairwise):
        getattr(self.window_group, method)(self.df, pairwise=pairwise)  # 调用分组后的方法


class Quantile:
    # 定义参数元组，包含数据结构类型、窗口长度、数据类型、分位数和插值方式
    params = (
        ["DataFrame", "Series"],  # 第一个元素：数据结构类型为 DataFrame 或 Series
        [10, 1000],               # 第二个元素：窗口长度为 10 或 1000
        ["int", "float"],         # 第三个元素：数据类型为 int 或 float
        [0, 0.5, 1],              # 第四个元素：分位数为 0, 0.5 或 1
        ["linear", "nearest", "lower", "higher", "midpoint"],  # 第五个元素：插值方式
    )
    # 参数名称列表
    param_names = ["constructor", "window", "dtype", "percentile"]

    # 初始化方法，设置数据结构、窗口长度、数据类型、分位数和插值方式
    def setup(self, constructor, window, dtype, percentile, interpolation):
        N = 10**5
        arr = np.random.random(N).astype(dtype)
        self.roll = getattr(pd, constructor)(arr).rolling(window)  # 创建滚动窗口对象
    # 定义一个方法 `time_quantile`，接受参数 `constructor`, `window`, `dtype`, `percentile`, `interpolation`
    def time_quantile(self, constructor, window, dtype, percentile, interpolation):
        # 调用 `roll` 对象的 `quantile` 方法，计算给定百分位数的分位数
        self.roll.quantile(percentile, interpolation=interpolation)
# 定义 Rank 类，用于进行数据排名操作
class Rank:
    # 定义参数组合列表，包含不同数据类型和选项的组合
    params = (
        ["DataFrame", "Series"],  # 数据结构类型
        [10, 1000],               # 滚动窗口大小
        ["int", "float"],         # 数据类型
        [True, False],            # 百分位数选项
        [True, False],            # 升序降序选项
        ["min", "max", "average"],# 排名方法选项
    )
    # 定义参数名列表
    param_names = [
        "constructor",    # 数据结构构造函数名
        "window",         # 滚动窗口大小
        "dtype",          # 数据类型
        "percentile",     # 百分位数选项
        "ascending",      # 升序降序选项
        "method",         # 排名方法选项
    ]

    # 设置方法，初始化数据和滚动对象
    def setup(self, constructor, window, dtype, percentile, ascending, method):
        N = 10**5
        arr = np.random.random(N).astype(dtype)
        self.roll = getattr(pd, constructor)(arr).rolling(window)

    # 执行时间测量的排名方法
    def time_rank(self, constructor, window, dtype, percentile, ascending, method):
        self.roll.rank(pct=percentile, ascending=ascending, method=method)


# 定义 PeakMemFixedWindowMinMax 类，执行固定窗口的极值操作
class PeakMemFixedWindowMinMax:
    # 定义操作选项列表，包含极小值和极大值
    params = ["min", "max"]

    # 设置方法，初始化数据和滚动对象
    def setup(self, operation):
        N = 10**6
        arr = np.random.random(N)
        self.roll = pd.Series(arr).rolling(2)

    # 执行峰值内存测量的固定窗口操作方法
    def peakmem_fixed(self, operation):
        for x in range(5):
            getattr(self.roll, operation)()


# 定义 ForwardWindowMethods 类，执行前向窗口方法操作
class ForwardWindowMethods:
    # 定义参数组合列表，包含不同数据类型、窗口大小和方法的组合
    params = (
        ["DataFrame", "Series"],  # 数据结构类型
        [10, 1000],               # 窗口大小
        ["int", "float"],         # 数据类型
        ["median", "mean", "max", "min", "kurt", "sum"],  # 操作方法选项
    )
    # 定义参数名列表
    param_names = ["constructor", "window_size", "dtype", "method"]

    # 设置方法，初始化数据和滚动对象
    def setup(self, constructor, window_size, dtype, method):
        N = 10**5
        arr = np.random.random(N).astype(dtype)
        indexer = pd.api.indexers.FixedForwardWindowIndexer(window_size=window_size)
        self.roll = getattr(pd, constructor)(arr).rolling(window=indexer)

    # 执行时间测量的滚动方法操作
    def time_rolling(self, constructor, window_size, dtype, method):
        getattr(self.roll, method)()

    # 执行峰值内存测量的滚动方法操作
    def peakmem_rolling(self, constructor, window_size, dtype, method):
        getattr(self.roll, method)()


# 定义 Groupby 类，执行分组操作
class Groupby:
    # 定义参数组合列表，包含不同方法和窗口选项的组合
    params = (
        ["sum", "median", "mean", "max", "min", "kurt", "sum"],  # 聚合方法选项
        [("rolling", {"window": 2}), ("rolling", {"window": "30s"}), ("expanding", {})],  # 窗口类型和参数字典
    )

    # 设置方法，初始化数据和分组窗口对象
    def setup(self, method, window_kwargs):
        N = 1000
        window, kwargs = window_kwargs
        df = pd.DataFrame(
            {
                "A": [str(i) for i in range(N)] * 10,
                "B": list(range(N)) * 10,
            }
        )
        if isinstance(kwargs.get("window", None), str):
            df.index = pd.date_range(start="1900-01-01", freq="1min", periods=N * 10)
        self.groupby_window = getattr(df.groupby("A"), window)(**kwargs)

    # 执行时间测量的分组方法操作
    def time_method(self, method, window_kwargs):
        getattr(self.groupby_window, method)()


# 定义 GroupbyLargeGroups 类，执行大规模分组操作
class GroupbyLargeGroups:
    # 设置方法，初始化数据框
    def setup(self):
        N = 100000
        self.df = pd.DataFrame({"A": [1, 2] * (N // 2), "B": np.random.randn(N)})
    # 定义一个方法 `time_rolling_multiindex_creation`，用于处理时间序列数据，并生成多级索引
    def time_rolling_multiindex_creation(self):
        # 对 DataFrame `self.df` 按列 "A" 进行分组
        # 对每个分组执行滚动窗口操作，窗口大小为3，并计算均值
        self.df.groupby("A").rolling(3).mean()
# 定义一个名为 GroupbyEWM 的类，用于执行基于分组的指数加权移动平均等操作
class GroupbyEWM:
    # 定义参数列表，包括 "var", "std", "cov", "corr"
    params = ["var", "std", "cov", "corr"]
    # 定义参数名称列表，仅包含 "method"
    param_names = ["method"]

    # 设置方法，初始化数据框 df 包含列"A"和"B"，每列包含50个数值
    def setup(self, method):
        df = pd.DataFrame({"A": range(50), "B": range(50)})
        # 根据"A"列分组并进行指数加权移动平均，设置 com 参数为 1.0
        self.gb_ewm = df.groupby("A").ewm(com=1.0)

    # 测试分组方法的执行时间，使用 getattr 动态调用 self.gb_ewm 对象的指定方法（method）
    def time_groupby_method(self, method):
        getattr(self.gb_ewm, method)()


# 定义一个名为 GroupbyEWMEngine 的类，用于执行基于分组的均值操作
class GroupbyEWMEngine:
    # 定义参数列表，包括 "cython", "numba"
    params = ["cython", "numba"]
    # 定义参数名称列表，仅包含 "engine"
    param_names = ["engine"]

    # 设置方法，初始化数据框 df 包含列"A"和"B"，每列包含50个数值
    def setup(self, engine):
        df = pd.DataFrame({"A": range(50), "B": range(50)})
        # 根据"A"列分组并进行指数加权移动平均，设置 com 参数为 1.0
        self.gb_ewm = df.groupby("A").ewm(com=1.0)

    # 测试分组均值方法的执行时间，调用 self.gb_ewm 对象的均值方法，指定引擎为参数 engine
    def time_groupby_mean(self, engine):
        self.gb_ewm.mean(engine=engine)


# 定义一个名为 table_method_func 的函数，接受参数 x 并返回对 x 求和并加 1 的结果
def table_method_func(x):
    return np.sum(x, axis=0) + 1


# 定义一个名为 TableMethod 的类，用于测试数据框的滚动和指数加权均值方法
class TableMethod:
    # 定义参数列表，包括 "single", "table"
    params = ["single", "table"]
    # 定义参数名称列表，仅包含 "method"
    param_names = ["method"]

    # 设置方法，在实例化时生成一个10行1000列的随机数数据框 self.df
    def setup(self, method):
        self.df = pd.DataFrame(np.random.randn(10, 1000))

    # 测试 apply 方法的执行时间，对 self.df 执行滚动窗口操作，应用 table_method_func 函数，原始数据标记为 True，指定引擎为 "numba"
    def time_apply(self, method):
        self.df.rolling(2, method=method).apply(
            table_method_func, raw=True, engine="numba"
        )

    # 测试 ewm 方法的执行时间，对 self.df 执行指数加权均值，设置 com 参数为 1，指定引擎为 "numba"
    def time_ewm_mean(self, method):
        self.df.ewm(1, method=method).mean(engine="numba")


# 导入 pandas_vb_common 模块中的 setup 函数，忽略 isort 排序警告
from .pandas_vb_common import setup  # noqa: F401 isort:skip
```