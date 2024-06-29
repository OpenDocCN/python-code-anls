# `D:\src\scipysrc\pandas\asv_bench\benchmarks\algos\isin.py`

```
# 导入 NumPy 库，简写为 np
import numpy as np

# 从 pandas 库中导入以下对象：
# Categorical：用于处理分类数据的对象
# Index：用于创建索引对象的类
# NaT：表示缺失的时间戳值
# Series：用于处理一维数据的类
# date_range：用于生成日期范围的函数
from pandas import (
    Categorical,
    Index,
    NaT,
    Series,
    date_range,
)

# 定义一个名为 IsIn 的类
class IsIn:
    # 定义一个参数列表，包含各种数据类型的字符串表示
    params = [
        "int64",
        "uint64",
        "object",
        "Int64",
        "boolean",
        "bool",
        "datetime64[ns]",
        "category[object]",
        "category[int]",
        "str",
        "string[python]",
        "string[pyarrow]",
    ]
    # 定义参数名称列表，只包含一个元素："dtype"
    param_names = ["dtype"]

    # 设置方法，根据传入的数据类型初始化数据
    def setup(self, dtype):
        N = 10000

        # 初始化一个长度为 2 的列表，元素为 NaT.to_datetime64() 的结果
        self.mismatched = [NaT.to_datetime64()] * 2

        # 根据数据类型初始化 Series 和 values
        if dtype in ["boolean", "bool"]:
            # 生成随机整数序列，转换为指定的布尔类型的 Series
            self.series = Series(np.random.randint(0, 2, N)).astype(dtype)
            # 初始化 values 为布尔值列表
            self.values = [True, False]

        elif dtype == "datetime64[ns]":
            # 生成一个日期时间序列 dti
            dti = date_range(start="2015-10-26", end="2016-01-01", freq="50s")
            # 将 dti 转换为 Series
            self.series = Series(dti)
            # 初始化 values 为 series 的每第三个值组成的列表
            self.values = self.series._values[::3]
            # 更新 mismatched 列表为 [1, 2]
            self.mismatched = [1, 2]

        elif dtype in ["category[object]", "category[int]"]:
            n = 5 * 10**5
            sample_size = 100

            # 生成随机整数数组 arr
            arr = list(np.random.randint(0, n // 10, size=n))
            if dtype == "category[object]":
                # 如果 dtype 是 "category[object]"，则将 arr 转换为带有前缀的字符串数组
                arr = [f"s{i:04d}" for i in arr]

            # 初始化 values 为 arr 的随机选取的 sample_size 个值
            self.values = np.random.choice(arr, sample_size)
            # 将 arr 转换为 category 类型的 Series
            self.series = Series(arr).astype("category")

        elif dtype in ["str", "string[python]", "string[pyarrow]"]:
            try:
                # 创建一个索引对象，包含以 "i-" 开头的字符串，将其转换为指定类型的 Series
                self.series = Series(
                    Index([f"i-{i}" for i in range(N)], dtype=object)._values,
                    dtype=dtype,
                )
            except ImportError as err:
                # 如果导入错误，抛出 NotImplementedError 异常
                raise NotImplementedError from err
            # 初始化 values 为 series 的前两个值组成的列表
            self.values = list(self.series[:2])

        else:
            # 如果 dtype 不在以上类型中，生成随机整数序列，转换为指定类型的 Series
            self.series = Series(np.random.randint(1, 10, N)).astype(dtype)
            # 初始化 values 为 [1, 2]
            self.values = [1, 2]

        # 将 values 转换为 Categorical 类型
        self.cat_values = Categorical(self.values)

    # 测试 isin 方法的性能，传入 values 参数
    def time_isin(self, dtype):
        self.series.isin(self.values)

    # 测试 isin 方法对 Categorical 类型的性能，传入 cat_values 参数
    def time_isin_categorical(self, dtype):
        self.series.isin(self.cat_values)

    # 测试 isin 方法对空列表的性能
    def time_isin_empty(self, dtype):
        self.series.isin([])

    # 测试 isin 方法对与 series 数据类型不匹配的 values 的性能，传入 mismatched 参数
    def time_isin_mismatched_dtype(self, dtype):
        self.series.isin(self.mismatched)


# 定义一个名为 IsinAlmostFullWithRandomInt 的类
class IsinAlmostFullWithRandomInt:
    # 参数列表包含三个参数：
    # - 包含数据类型的列表
    # - 一个整数范围
    # - 标题字符串列表
    params = [
        [np.float64, np.int64, np.uint64, np.object_],
        range(10, 21),
        ["inside", "outside"],
    ]
    # 参数名称列表包含三个元素："dtype", "exponent", "title"
    param_names = ["dtype", "exponent", "title"]
    # 根据传入的数据类型、指数和标题设置参数 M
    M = 3 * 2 ** (exponent - 2)

    # 0.77-被占用桶的最大份额，这里是一个注释，解释了下面一行代码的作用
    self.series = Series(np.random.randint(0, M, M)).astype(dtype)

    # 生成一个包含随机整数的数组，用作值
    values = np.random.randint(0, M, M).astype(dtype)

    # 根据标题决定如何处理值数组
    if title == "inside":
        self.values = values
    elif title == "outside":
        self.values = values + M
    else:
        # 如果标题不是 "inside" 或 "outside"，则引发值错误异常
        raise ValueError(title)

    # Series 对象调用 isin 方法，用来检查元素是否在 values 数组中
    self.series.isin(self.values)
class IsinWithRandomFloat:
    # 定义参数组合：数据类型，大小，标题
    params = [
        [np.float64, np.object_],
        [
            1_300,
            2_000,
            7_000,
            8_000,
            70_000,
            80_000,
            750_000,
            900_000,
        ],
        ["inside", "outside"],
    ]
    # 参数名称列表
    param_names = ["dtype", "size", "title"]

    # 初始化设置方法，根据参数设置随机值和系列对象
    def setup(self, dtype, size, title):
        # 生成指定大小的随机数值
        self.values = np.random.rand(size)
        # 创建系列对象并转换为指定数据类型
        self.series = Series(self.values).astype(dtype)
        # 打乱随机值顺序
        np.random.shuffle(self.values)

        # 如果标题为"outside"，则对随机值进行调整
        if title == "outside":
            self.values = self.values + 0.1

    # 测试方法：计算isin函数的执行时间
    def time_isin(self, dtype, size, title):
        self.series.isin(self.values)


class IsinWithArangeSorted:
    # 定义参数组合：数据类型，大小
    params = [
        [np.float64, np.int64, np.uint64, np.object_],
        [
            1_000,
            2_000,
            8_000,
            100_000,
            1_000_000,
        ],
    ]
    # 参数名称列表
    param_names = ["dtype", "size"]

    # 初始化设置方法，创建按顺序排列的系列对象和数值对象
    def setup(self, dtype, size):
        # 创建按顺序排列的系列对象
        self.series = Series(np.arange(size)).astype(dtype)
        # 创建按顺序排列的数值对象
        self.values = np.arange(size).astype(dtype)

    # 测试方法：计算isin函数的执行时间
    def time_isin(self, dtype, size):
        self.series.isin(self.values)


class IsinWithArange:
    # 定义参数组合：数据类型，M（大小），偏移因子
    params = [
        [np.float64, np.int64, np.uint64, np.object_],
        [
            1_000,
            2_000,
            8_000,
        ],
        [-2, 0, 2],
    ]
    # 参数名称列表
    param_names = ["dtype", "M", "offset_factor"]

    # 初始化设置方法，根据参数设置随机系列对象和数值对象
    def setup(self, dtype, M, offset_factor):
        # 根据偏移因子计算偏移量
        offset = int(M * offset_factor)
        # 创建随机整数系列对象
        tmp = Series(np.random.randint(offset, M + offset, 10**6))
        self.series = tmp.astype(dtype)
        # 创建按顺序排列的数值对象
        self.values = np.arange(M).astype(dtype)

    # 测试方法：计算isin函数的执行时间
    def time_isin(self, dtype, M, offset_factor):
        self.series.isin(self.values)


class IsInFloat64:
    # 定义参数组合：数据类型，标题
    params = [
        [np.float64, "Float64"],
        ["many_different_values", "few_different_values", "only_nans_values"],
    ]
    # 参数名称列表
    param_names = ["dtype", "title"]

    # 初始化设置方法，根据标题设置不同的数值对象
    def setup(self, dtype, title):
        N_many = 10**5
        N_few = 10**6
        # 创建包含1和2的系列对象，数据类型由标题决定
        self.series = Series([1, 2], dtype=dtype)

        # 根据标题选择创建不同的数值对象
        if title == "many_different_values":
            # 创建包含大量不同值的数值对象
            self.values = np.arange(N_many, dtype=np.float64)
        elif title == "few_different_values":
            # 创建全为0的大数值对象
            self.values = np.zeros(N_few, dtype=np.float64)
        elif title == "only_nans_values":
            # 创建全为NaN的大数值对象
            self.values = np.full(N_few, np.nan, dtype=np.float64)
        else:
            raise ValueError(title)

    # 测试方法：计算isin函数的执行时间
    def time_isin(self, dtype, title):
        self.series.isin(self.values)


class IsInForObjects:
    """
    A subset of the cartesian product of cases have special motivations:

    "nans" x "nans"
        if nan-objects are different objects,
        this has the potential to trigger O(n^2) running time
    """
    # 空类，仅包含文档字符串作为注释
    pass
    """
    "short" x "long"
        运行时间由预处理阶段主导

    "long" x "short"
        运行时间由查找操作主导

    "long" x "long"
        没有明显的主导部分

    "long_floats" x "long_floats"
        由于存在 NaN，浮点数具有特殊性
        没有明显的主导部分
    """

    # 定义可选的变体类型
    variants = ["nans", "short", "long", "long_floats"]

    # 参数组合
    params = [variants, variants]
    # 参数名称
    param_names = ["series_type", "vals_type"]

    # 设置方法，在运行基准测试前执行
    def setup(self, series_type, vals_type):
        # 大数据量级
        N_many = 10**5

        # 根据 series_type 初始化不同类型的序列数据
        if series_type == "nans":
            ser_vals = np.full(10**4, np.nan)
        elif series_type == "short":
            ser_vals = np.arange(2)
        elif series_type == "long":
            ser_vals = np.arange(N_many)
        elif series_type == "long_floats":
            ser_vals = np.arange(N_many, dtype=np.float64)

        # 使用初始化的序列数据创建 Pandas Series 对象，并转换为对象类型
        self.series = Series(ser_vals).astype(object)

        # 根据 vals_type 初始化不同类型的值数据
        if vals_type == "nans":
            values = np.full(10**4, np.nan)
        elif vals_type == "short":
            values = np.arange(2)
        elif vals_type == "long":
            values = np.arange(N_many)
        elif vals_type == "long_floats":
            values = np.arange(N_many, dtype=np.float64)

        # 转换值数据为对象类型
        self.values = values.astype(object)

    # 基准测试方法，测量 isin 方法的运行时间
    def time_isin(self, series_type, vals_type):
        # 调用 Pandas Series 的 isin 方法，传入值数据作为参数
        self.series.isin(self.values)
# 定义一个类 IsInLongSeriesLookUpDominates，用于测试 Series 对象的 isin 方法性能在不同参数下的表现
class IsInLongSeriesLookUpDominates:
    # 参数列表，包含数据类型、最大数值、序列类型
    params = [
        ["int64", "int32", "float64", "float32", "object", "Int64", "Float64"],
        [5, 1000],
        ["random_hits", "random_misses", "monotone_hits", "monotone_misses"],
    ]
    # 参数名称列表
    param_names = ["dtype", "MaxNumber", "series_type"]

    # 设置方法，根据不同的参数类型和序列类型生成测试数据
    def setup(self, dtype, MaxNumber, series_type):
        # 设定序列长度 N
        N = 10**7

        # 根据序列类型生成不同类型的数组
        if series_type == "random_hits":
            array = np.random.randint(0, MaxNumber, N)
        if series_type == "random_misses":
            array = np.random.randint(0, MaxNumber, N) + MaxNumber
        if series_type == "monotone_hits":
            array = np.repeat(np.arange(MaxNumber), N // MaxNumber)
        if series_type == "monotone_misses":
            array = np.arange(N) + MaxNumber

        # 将生成的数组转换为 Series 对象，并设置其数据类型为 dtype
        self.series = Series(array).astype(dtype)

        # 生成值数组，数据类型为 dtype 的小写形式
        self.values = np.arange(MaxNumber).astype(dtype.lower())

    # 计时方法，测试 Series 对象的 isin 方法在给定参数下的性能
    def time_isin(self, dtypes, MaxNumber, series_type):
        self.series.isin(self.values)


# 定义一个类 IsInLongSeriesValuesDominate，用于测试 Series 对象的 isin 方法在值占主导地位时的性能
class IsInLongSeriesValuesDominate:
    # 参数列表，包含数据类型和序列类型
    params = [
        ["int64", "int32", "float64", "float32", "object", "Int64", "Float64"],
        ["random", "monotone"],
    ]
    # 参数名称列表
    param_names = ["dtype", "series_type"]

    # 设置方法，根据不同的数据类型和序列类型生成测试数据
    def setup(self, dtype, series_type):
        # 设定序列长度 N
        N = 10**7

        # 根据序列类型生成不同类型的值数组
        if series_type == "random":
            vals = np.random.randint(0, 10 * N, N)
        if series_type == "monotone":
            vals = np.arange(N)

        # 将生成的值数组转换为数据类型为 dtype 的形式
        self.values = vals.astype(dtype.lower())

        # 设定序列长度为 M
        M = 10**6 + 1
        # 生成序列，包含从 0 到 M-1 的整数，并将其数据类型设定为 dtype
        self.series = Series(np.arange(M)).astype(dtype)

    # 计时方法，测试 Series 对象的 isin 方法在给定参数下的性能
    def time_isin(self, dtypes, series_type):
        self.series.isin(self.values)


# 定义一个类 IsInWithLongTupples，用于测试 Series 对象的 isin 方法在包含长元组时的性能
class IsInWithLongTupples:
    # 设置方法，生成包含重复长元组的 Series 对象和值列表
    def setup(self):
        # 创建一个长度为 1000 的元组 t
        t = tuple(range(1000))
        # 创建一个 Series 对象，包含 1000 个 t 元组
        self.series = Series([t] * 1000)
        # 创建一个值列表，包含元组 t
        self.values = [t]

    # 计时方法，测试 Series 对象的 isin 方法在给定参数下的性能
    def time_isin(self):
        self.series.isin(self.values)


# 定义一个类 IsInIndexes，用于测试 Series 对象的 isin 方法在不同类型索引下的性能
class IsInIndexes:
    # 设置方法，生成不同类型的索引和随机数序列的 Series 对象
    def setup(self):
        # 创建一个标准范围索引，包含从 0 到 999 的整数
        self.range_idx = Index(range(1000))
        # 创建一个列表形式的索引，包含从 0 到 999 的整数
        self.index = Index(list(range(1000)))
        # 创建一个包含 1000 个随机整数的 Series 对象，整数范围为 [0, 99999]
        self.series = Series(np.random.randint(100_000, size=1000))

    # 计时方法，测试 Series 对象的 isin 方法在标准范围索引下的性能
    def time_isin_range_index(self):
        self.series.isin(self.range_idx)

    # 计时方法，测试 Series 对象的 isin 方法在列表形式索引下的性能
    def time_isin_index(self):
        self.series.isin(self.index)
```