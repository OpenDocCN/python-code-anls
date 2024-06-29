# `D:\src\scipysrc\pandas\asv_bench\benchmarks\algorithms.py`

```
from importlib import import_module  # 导入 import_module 函数，用于动态导入模块

import numpy as np  # 导入 numpy 库，并使用 np 别名

import pandas as pd  # 导入 pandas 库，并使用 pd 别名

for imp in ["pandas.util", "pandas.tools.hashing"]:
    try:
        hashing = import_module(imp)  # 尝试动态导入指定的模块路径
        break  # 如果成功导入，则跳出循环
    except (ImportError, TypeError, ValueError):
        pass  # 如果导入失败，则捕获异常并继续循环

class Factorize:
    params = [
        [True, False],  # 布尔型参数列表
        [True, False],  # 布尔型参数列表
        [
            "int64",
            "uint64",
            "float64",
            "object",
            "object_str",
            "datetime64[ns]",
            "datetime64[ns, tz]",
            "Int64",
            "boolean",
            "string[pyarrow]",
        ],  # 数据类型参数列表
    ]
    param_names = ["unique", "sort", "dtype"]  # 参数名称列表

    def setup(self, unique, sort, dtype):
        N = 10**5  # 数据数量

        if dtype in ["int64", "uint64", "Int64", "object"]:
            data = pd.Index(np.arange(N), dtype=dtype)  # 根据不同数据类型创建数据
        elif dtype == "float64":
            data = pd.Index(np.random.randn(N), dtype=dtype)
        elif dtype == "boolean":
            data = pd.array(np.random.randint(0, 2, N), dtype=dtype)
        elif dtype == "datetime64[ns]":
            data = pd.date_range("2011-01-01", freq="h", periods=N)
        elif dtype == "datetime64[ns, tz]":
            data = pd.date_range("2011-01-01", freq="h", periods=N, tz="Asia/Tokyo")
        elif dtype == "object_str":
            data = pd.Index([f"i-{i}" for i in range(N)], dtype=object)
        elif dtype == "string[pyarrow]":
            data = pd.array(
                pd.Index([f"i-{i}" for i in range(N)], dtype=object),
                dtype="string[pyarrow]",
            )
        else:
            raise NotImplementedError  # 抛出未实现错误，处理未知数据类型

        if not unique:
            data = data.repeat(5)  # 如果不要求唯一值，则重复数据
        self.data = data  # 将生成的数据存储到对象属性中

    def time_factorize(self, unique, sort, dtype):
        pd.factorize(self.data, sort=sort)  # 使用 pandas 的 factorize 函数进行因子化操作

    def peakmem_factorize(self, unique, sort, dtype):
        pd.factorize(self.data, sort=sort)  # 使用 pandas 的 factorize 函数进行因子化操作


class Duplicated:
    params = [
        [True, False],  # 布尔型参数列表
        ["first", "last", False],  # 重复值处理方式列表
        [
            "int64",
            "uint64",
            "float64",
            "string",
            "datetime64[ns]",
            "datetime64[ns, tz]",
            "timestamp[ms][pyarrow]",
            "duration[s][pyarrow]",
        ],  # 数据类型参数列表
    ]
    param_names = ["unique", "keep", "dtype"]  # 参数名称列表
    # 设置函数，用于初始化数据索引
    def setup(self, unique, keep, dtype):
        # 设定数据量级
        N = 10**5
        # 根据数据类型选择不同的数据创建方式
        if dtype in ["int64", "uint64"]:
            data = pd.Index(np.arange(N), dtype=dtype)
        elif dtype == "float64":
            data = pd.Index(np.random.randn(N), dtype="float64")
        elif dtype == "string":
            data = pd.Index([f"i-{i}" for i in range(N)], dtype=object)
        elif dtype == "datetime64[ns]":
            data = pd.date_range("2011-01-01", freq="h", periods=N)
        elif dtype == "datetime64[ns, tz]":
            data = pd.date_range("2011-01-01", freq="h", periods=N, tz="Asia/Tokyo")
        elif dtype in ["timestamp[ms][pyarrow]", "duration[s][pyarrow]"]:
            data = pd.Index(np.arange(N), dtype=dtype)
        else:
            # 抛出未实现错误，如果遇到未知的数据类型
            raise NotImplementedError
        # 如果不需要唯一值，数据重复5次
        if not unique:
            data = data.repeat(5)
        # 将数据索引存储在实例变量中
        self.idx = data
        # 缓存 is_unique 属性
        self.idx.is_unique

    # 时间重复检查函数
    def time_duplicated(self, unique, keep, dtype):
        # 调用 Pandas 的 duplicated 方法检查重复值
        self.idx.duplicated(keep=keep)
class DuplicatedMaskedArray:
    # 定义参数组合列表
    params = [
        [True, False],
        ["first", "last", False],
        ["Int64", "Float64"],
    ]
    # 定义参数名称列表
    param_names = ["unique", "keep", "dtype"]

    # 设置方法，初始化数据
    def setup(self, unique, keep, dtype):
        # 创建一个长度为 10^5 的 Series 对象，包含整数范围内的数据
        N = 10**5
        data = pd.Series(np.arange(N), dtype=dtype)
        # 将每隔 100 个元素设为 NA（缺失值）
        data[list(range(1, N, 100))] = pd.NA
        # 如果 unique 不为 True，则重复每个元素五次
        if not unique:
            data = data.repeat(5)
        # 将处理后的 Series 对象赋值给实例变量 ser
        self.ser = data
        # 缓存 is_unique 属性
        self.ser.is_unique

    # 计时方法，计算重复值
    def time_duplicated(self, unique, keep, dtype):
        self.ser.duplicated(keep=keep)


class Hashing:
    # 设置缓存方法，生成包含多种数据类型的 DataFrame
    def setup_cache(self):
        N = 10**5
        # 创建包含多种数据类型的 DataFrame
        df = pd.DataFrame(
            {
                "strings": pd.Series(
                    pd.Index([f"i-{i}" for i in range(10000)], dtype=object).take(
                        np.random.randint(0, 10000, size=N)
                    )
                ),
                "floats": np.random.randn(N),
                "ints": np.arange(N),
                "dates": pd.date_range("20110101", freq="s", periods=N),
                "timedeltas": pd.timedelta_range("1 day", freq="s", periods=N),
            }
        )
        # 将 strings 列转换为 category 类型
        df["categories"] = df["strings"].astype("category")
        # 将第 10 到 20 行设为 NaN
        df.iloc[10:20] = np.nan
        return df

    # 计时方法，对 DataFrame 执行哈希操作
    def time_frame(self, df):
        hashing.hash_pandas_object(df)

    # 计时方法，对整数序列执行哈希操作
    def time_series_int(self, df):
        hashing.hash_pandas_object(df["ints"])

    # 计时方法，对字符串序列执行哈希操作
    def time_series_string(self, df):
        hashing.hash_pandas_object(df["strings"])

    # 计时方法，对浮点数序列执行哈希操作
    def time_series_float(self, df):
        hashing.hash_pandas_object(df["floats"])

    # 计时方法，对分类数据序列执行哈希操作
    def time_series_categorical(self, df):
        hashing.hash_pandas_object(df["categories"])

    # 计时方法，对时间间隔序列执行哈希操作
    def time_series_timedeltas(self, df):
        hashing.hash_pandas_object(df["timedeltas"])

    # 计时方法，对日期序列执行哈希操作
    def time_series_dates(self, df):
        hashing.hash_pandas_object(df["dates"])


class Quantile:
    # 定义参数组合列表
    params = [
        [0, 0.5, 1],
        ["linear", "nearest", "lower", "higher", "midpoint"],
        ["float64", "int64", "uint64"],
    ]
    # 定义参数名称列表
    param_names = ["quantile", "interpolation", "dtype"]

    # 设置方法，初始化数据
    def setup(self, quantile, interpolation, dtype):
        N = 10**5
        # 根据 dtype 类型生成不同的数据
        if dtype in ["int64", "uint64"]:
            data = np.arange(N, dtype=dtype)
        elif dtype == "float64":
            data = np.random.randn(N)
        else:
            raise NotImplementedError
        # 将数据重复五次，并创建为 Series 对象
        self.ser = pd.Series(data.repeat(5))

    # 计时方法，计算分位数
    def time_quantile(self, quantile, interpolation, dtype):
        self.ser.quantile(quantile, interpolation=interpolation)


class SortIntegerArray:
    # 定义参数列表
    params = [10**3, 10**5]

    # 设置方法，初始化数据
    def setup(self, N):
        # 创建一个包含浮点数的数组，第 40 个元素设为 NaN
        data = np.arange(N, dtype=float)
        data[40] = np.nan
        # 将数据创建为 Int64 类型的数组对象
        self.array = pd.array(data, dtype="Int64")

    # 计时方法，对数组执行 argsort 操作
    def time_argsort(self, N):
        self.array.argsort()


from .pandas_vb_common import setup  # noqa: F401 isort:skip
```