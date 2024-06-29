# `D:\src\scipysrc\pandas\asv_bench\benchmarks\series_methods.py`

```
from datetime import datetime  # 导入datetime模块，用于处理日期时间

import numpy as np  # 导入numpy库，用于数值计算

from pandas import (  # 从pandas库中导入以下模块：
    NA,  # 用于表示缺失值的常量
    Index,  # 索引对象
    NaT,  # 用于表示缺失日期时间的常量
    Series,  # Series数据结构
    date_range,  # 用于生成日期范围的函数
)


class SeriesConstructor:  # 定义SeriesConstructor类，用于创建Series对象
    def setup(self):  # 设置方法，初始化数据
        self.idx = date_range(  # 生成日期范围，从2015年10月26日到2016年1月1日，间隔50秒
            start=datetime(2015, 10, 26), end=datetime(2016, 1, 1), freq="50s"
        )
        self.data = dict(zip(self.idx, range(len(self.idx))))  # 创建日期到整数的字典映射
        self.array = np.array([1, 2, 3])  # 创建一个numpy数组
        self.idx2 = Index(["a", "b", "c"])  # 创建索引对象

    def time_constructor_dict(self):  # 创建基于字典的Series对象
        Series(data=self.data, index=self.idx)

    def time_constructor_no_data(self):  # 创建不带数据的Series对象
        Series(data=None, index=self.idx)


class ToFrame:  # 定义ToFrame类，用于数据框转换
    params = [["int64", "datetime64[ns]", "category", "Int64"], [None, "foo"]]  # 参数化设置
    param_names = ["dtype", "name"]  # 参数名设置

    def setup(self, dtype, name):  # 设置方法，初始化数据
        arr = np.arange(10**5)  # 创建一个包含10^5个元素的数组
        ser = Series(arr, dtype=dtype)  # 使用指定数据类型创建Series对象
        self.ser = ser

    def time_to_frame(self, dtype, name):  # 将Series对象转换为数据框
        self.ser.to_frame(name)


class NSort:  # 定义NSort类，用于排序操作
    params = ["first", "last", "all"]  # 参数化设置
    param_names = ["keep"]  # 参数名设置

    def setup(self, keep):  # 设置方法，初始化数据
        self.s = Series(np.random.randint(1, 10, 100000))  # 创建包含随机整数的Series对象

    def time_nlargest(self, keep):  # 计算Series对象中最大的前3个元素
        self.s.nlargest(3, keep=keep)

    def time_nsmallest(self, keep):  # 计算Series对象中最小的前3个元素
        self.s.nsmallest(3, keep=keep)


class Dropna:  # 定义Dropna类，用于处理缺失值
    params = ["int", "datetime"]  # 参数化设置
    param_names = ["dtype"]  # 参数名设置

    def setup(self, dtype):  # 设置方法，初始化数据
        N = 10**6  # 数据量级
        data = {
            "int": np.random.randint(1, 10, N),  # 整数数据
            "datetime": date_range("2000-01-01", freq="s", periods=N),  # 日期时间数据
        }
        self.s = Series(data[dtype])  # 根据指定类型选择数据
        if dtype == "datetime":
            self.s[np.random.randint(1, N, 100)] = NaT  # 随机将一些日期时间设置为缺失值

    def time_dropna(self, dtype):  # 删除Series对象中的缺失值
        self.s.dropna()


class Fillna:  # 定义Fillna类，用于填充缺失值
    params = [  # 参数化设置
        [
            "datetime64[ns]",  # 日期时间类型
            "float32",  # 单精度浮点数类型
            "float64",  # 双精度浮点数类型
            "Float64",  # Pandas中的浮点数类型
            "Int64",  # 整数类型
            "int64[pyarrow]",  # Apache Arrow中的整数类型
            "string",  # 字符串类型
            "string[pyarrow]",  # Apache Arrow中的字符串类型
        ],
    ]
    param_names = ["dtype"]  # 参数名设置

    def setup(self, dtype):  # 设置方法，初始化数据
        N = 10**6  # 数据量级
        if dtype == "datetime64[ns]":
            data = date_range("2000-01-01", freq="s", periods=N)  # 生成日期时间序列
            na_value = NaT  # 缺失日期时间值
        elif dtype in ("float64", "Float64"):
            data = np.random.randn(N)  # 生成随机双精度浮点数
            na_value = np.nan  # 缺失浮点数值
        elif dtype in ("Int64", "int64[pyarrow]"):
            data = np.arange(N)  # 生成整数序列
            na_value = NA  # 缺失整数值
        elif dtype in ("string", "string[pyarrow]"):
            data = np.array([str(i) * 5 for i in range(N)], dtype=object)  # 生成字符串序列
            na_value = NA  # 缺失字符串值
        else:
            raise NotImplementedError  # 抛出未实现错误
        fill_value = data[0]  # 填充值为数据的第一个元素
        ser = Series(data, dtype=dtype)  # 创建指定类型的Series对象
        ser[::2] = na_value  # 将部分数据设置为缺失值
        self.ser = ser  # 保存Series对象
        self.fill_value = fill_value  # 保存填充值

    def time_fillna(self, dtype):  # 填充Series对象中的缺失值
        self.ser.fillna(value=self.fill_value)

    def time_ffill(self, dtype):  # 前向填充Series对象中的缺失值
        self.ser.ffill()

    def time_bfill(self, dtype):  # 后向填充Series对象中的缺失值
        self.ser.bfill()


class SearchSorted:  # 定义SearchSorted类，用于搜索排序后的位置
    # 设定目标时间为0.2秒，用于性能测试时的参考时间
    goal_time = 0.2
    # 定义参数列表，包括各种数据类型的字符串表示
    params = [
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float16",
        "float32",
        "float64",
        "str",
    ]
    # 参数名称列表，目前只包含一个元素："dtype"
    param_names = ["dtype"]
    
    # 定义性能测试类，包含设置数据和搜索操作的方法
    def setup(self, dtype):
        # 设置数据大小为10万，包括一部分值为1，一部分值为2，一部分值为3，并转换为指定类型的数据
        N = 10**5
        data = np.array([1] * N + [2] * N + [3] * N).astype(dtype)
        # 创建一个 pandas Series 对象，用设置好的数据
        self.s = Series(data)
    
    def time_searchsorted(self, dtype):
        # 根据数据类型选择搜索的关键字，如果数据类型是字符串，则关键字为 "2"，否则为整数 2
        key = "2" if dtype == "str" else 2
        # 执行搜索操作，返回搜索结果的索引位置
        self.s.searchsorted(key)
class Map:
    # 定义类变量 params，包含多个参数组合
    params = (
        ["dict", "Series", "lambda"],
        ["object", "category", "int"],
        [None, "ignore"],
    )
    # 定义类变量 param_names，包含参数名列表
    param_names = ["mapper", "dtype", "na_action"]

    # 设置函数，根据参数初始化映射数据和随机数序列
    def setup(self, mapper, dtype, na_action):
        # 设置映射数据的大小
        map_size = 1000
        # 创建一个 Series 对象，包含 map_size 个元素，数据类型为 dtype
        map_data = Series(map_size - np.arange(map_size), dtype=dtype)

        # 根据 mapper 参数构建映射关系
        if mapper == "Series":
            self.map_data = map_data
        elif mapper == "dict":
            # 将 Series 对象转换为字典
            self.map_data = map_data.to_dict()
        elif mapper == "lambda":
            # 将 Series 对象转换为字典，并创建一个 lambda 函数
            map_dict = map_data.to_dict()
            self.map_data = lambda x: map_dict[x]
        else:
            # 抛出未实现的错误，如果传入了未知的 mapper 参数
            raise NotImplementedError

        # 创建一个包含随机整数的 Series 对象，大小为 10000
        self.s = Series(np.random.randint(0, map_size, 10000), dtype=dtype)

    # 时间测量函数，调用 Series 对象的 map 方法
    def time_map(self, mapper, dtype, na_action):
        self.s.map(self.map_data, na_action=na_action)


class Clip:
    # 定义类变量 params，包含多个 n 值
    params = [50, 1000, 10**5]
    # 定义类变量 param_names，包含参数名列表
    param_names = ["n"]

    # 设置函数，根据参数初始化随机数序列
    def setup(self, n):
        self.s = Series(np.random.randn(n))

    # 时间测量函数，调用 Series 对象的 clip 方法
    def time_clip(self, n):
        self.s.clip(0, 1)


class ClipDt:
    # 设置函数，初始化日期范围和 Series 对象
    def setup(self):
        # 创建一个包含 100000 个时间戳的日期范围
        dr = date_range("20220101", periods=100_000, freq="s", tz="UTC")
        # 从日期范围中选择前 1000 个时间戳并重复 100 次
        self.clipper_dt = dr[0:1_000].repeat(100)
        # 创建一个包含全部时间戳的 Series 对象
        self.s = Series(dr)

    # 时间测量函数，调用 Series 对象的 clip 方法
    def time_clip(self):
        self.s.clip(upper=self.clipper_dt)


class ValueCounts:
    # 定义类变量 params，包含多个 N 值和数据类型 dtype
    params = [[10**3, 10**4, 10**5], ["int", "uint", "float", "object"]]
    # 定义类变量 param_names，包含参数名列表
    param_names = ["N", "dtype"]

    # 设置函数，根据参数初始化包含随机数的 Series 对象
    def setup(self, N, dtype):
        self.s = Series(np.random.randint(0, N, size=10 * N)).astype(dtype)

    # 时间测量函数，调用 Series 对象的 value_counts 方法
    def time_value_counts(self, N, dtype):
        self.s.value_counts()


class ValueCountsEA:
    # 定义类变量 params，包含多个 N 值和 dropna 参数
    params = [[10**3, 10**4, 10**5], [True, False]]
    # 定义类变量 param_names，包含参数名列表
    param_names = ["N", "dropna"]

    # 设置函数，根据参数初始化包含随机数和缺失值的 Series 对象
    def setup(self, N, dropna):
        self.s = Series(np.random.randint(0, N, size=10 * N), dtype="Int64")
        self.s.loc[1] = NA

    # 时间测量函数，调用 Series 对象的 value_counts 方法
    def time_value_counts(self, N, dropna):
        self.s.value_counts(dropna=dropna)


class ValueCountsObjectDropNAFalse:
    # 定义类变量 params，包含多个 N 值
    params = [10**3, 10**4, 10**5]
    # 定义类变量 param_names，包含参数名列表
    param_names = ["N"]

    # 设置函数，根据参数初始化包含随机数的对象类型 Series 对象
    def setup(self, N):
        self.s = Series(np.random.randint(0, N, size=10 * N)).astype("object")

    # 时间测量函数，调用 Series 对象的 value_counts 方法，且 dropna 参数设为 False
    def time_value_counts(self, N):
        self.s.value_counts(dropna=False)


class Mode:
    # 定义类变量 params，包含多个 N 值和数据类型 dtype
    params = [[10**3, 10**4, 10**5], ["int", "uint", "float", "object"]]
    # 定义类变量 param_names，包含参数名列表
    param_names = ["N", "dtype"]

    # 设置函数，根据参数初始化包含随机数的 Series 对象
    def setup(self, N, dtype):
        self.s = Series(np.random.randint(0, N, size=10 * N)).astype(dtype)

    # 时间测量函数，调用 Series 对象的 mode 方法
    def time_mode(self, N, dtype):
        self.s.mode()


class ModeObjectDropNAFalse:
    # 定义类变量 params，包含多个 N 值
    params = [10**3, 10**4, 10**5]
    # 定义类变量 param_names，包含参数名列表
    param_names = ["N"]

    # 设置函数，根据参数初始化包含随机数的对象类型 Series 对象
    def setup(self, N):
        self.s = Series(np.random.randint(0, N, size=10 * N)).astype("object")

    # 时间测量函数，调用 Series 对象的 mode 方法，且 dropna 参数设为 False
    def time_mode(self, N):
        self.s.mode(dropna=False)


class Dir:
    # 设置函数，初始化包含对象字符串索引的 Series 对象
    def setup(self):
        # 创建一个包含对象类型索引的 Series 对象，索引为 "i-0" 到 "i-9999"
        self.s = Series(index=Index([f"i-{i}" for i in range(10000)], dtype=object))

    # 时间测量函数，调用 dir 函数
    def time_dir_strings(self):
        dir(self.s)


class SeriesGetattr:
    # 该类暂无需要添加的代码注释，因为不包含任何方法或参数
    pass
    # 在这个方法中，设置一个名为 `s` 的 Pandas Series 对象，包含了从 "2012-01-01" 开始、频率为每秒钟、总共一百万个时间点的索引。
    def setup(self):
        self.s = Series(1, index=date_range("2012-01-01", freq="s", periods=10**6))
    
    # 调用 `getattr` 函数，尝试从 `self.s` 中获取属性 `"a"` 的值，如果属性不存在则返回 None。
    def time_series_datetimeindex_repr(self):
        getattr(self.s, "a", None)
# 定义一个名为 All 的类，用于测试 Series 对象中的 all() 方法的性能
class All:
    # 参数组合，包括不同的 N（数据长度）、case（快速或慢速）、dtype（数据类型）
    params = [[10**3, 10**6], ["fast", "slow"], ["bool", "boolean"]]
    # 参数名称，分别对应 N、case、dtype
    param_names = ["N", "case", "dtype"]

    # 设置方法，在测试运行前准备数据
    def setup(self, N, case, dtype):
        # 根据 case 是否为 "fast" 决定 val 的值，用于创建 Series 对象
        val = case != "fast"
        self.s = Series([val] * N, dtype=dtype)

    # 测试 all() 方法的运行时间
    def time_all(self, N, case, dtype):
        self.s.all()


# 定义一个名为 Any 的类，用于测试 Series 对象中的 any() 方法的性能
class Any:
    # 参数组合，包括不同的 N（数据长度）、case（快速或慢速）、dtype（数据类型）
    params = [[10**3, 10**6], ["fast", "slow"], ["bool", "boolean"]]
    # 参数名称，分别对应 N、case、dtype
    param_names = ["N", "case", "dtype"]

    # 设置方法，在测试运行前准备数据
    def setup(self, N, case, dtype):
        # 根据 case 是否为 "fast" 决定 val 的值，用于创建 Series 对象
        val = case == "fast"
        self.s = Series([val] * N, dtype=dtype)

    # 测试 any() 方法的运行时间
    def time_any(self, N, case, dtype):
        self.s.any()


# 定义一个名为 NanOps 的类，用于测试 Series 对象中各种操作方法的性能
class NanOps:
    # 参数组合，包括各种操作方法、不同的 N（数据长度）、不同的 dtype（数据类型）
    params = [
        [
            "var",
            "mean",
            "median",
            "max",
            "min",
            "sum",
            "std",
            "sem",
            "argmax",
            "skew",
            "kurt",
            "prod",
        ],
        [10**3, 10**6],
        ["int8", "int32", "int64", "float64", "Int64", "boolean"],
    ]
    # 参数名称，分别对应操作方法、N、dtype
    param_names = ["func", "N", "dtype"]

    # 设置方法，在测试运行前准备数据
    def setup(self, func, N, dtype):
        # 如果操作方法为 "argmax"，且 dtype 为 "Int64" 或 "boolean"，抛出 NotImplementedError
        if func == "argmax" and dtype in {"Int64", "boolean"}:
            raise NotImplementedError
        # 创建一个包含 N 个元素的 Series 对象，数据类型为指定的 dtype
        self.s = Series(np.ones(N), dtype=dtype)
        # 获取 Series 对象的指定操作方法，如 var、mean、median 等
        self.func = getattr(self.s, func)

    # 测试各种操作方法的运行时间
    def time_func(self, func, N, dtype):
        self.func()


# 定义一个名为 Rank 的类，用于测试 Series 对象中的 rank() 方法的性能
class Rank:
    # 参数组合，只包括不同的 dtype（数据类型）
    param_names = ["dtype"]
    params = [
        ["int", "uint", "float", "object"],
    ]

    # 设置方法，在测试运行前准备数据
    def setup(self, dtype):
        # 创建一个包含 100000 个随机整数的 Series 对象，数据类型为指定的 dtype
        self.s = Series(np.random.randint(0, 1000, size=100000), dtype=dtype)

    # 测试 rank() 方法的运行时间
    def time_rank(self, dtype):
        self.s.rank()


# 定义一个名为 Iter 的类，用于测试 Series 对象的迭代性能
class Iter:
    # 参数组合，包括不同的 dtype（数据类型）
    param_names = ["dtype"]
    params = [
        "bool",
        "boolean",
        "int64",
        "Int64",
        "float64",
        "Float64",
        "datetime64[ns]",
    ]

    # 设置方法，在测试运行前准备数据
    def setup(self, dtype):
        N = 10**5
        # 根据 dtype 创建不同类型的 Series 对象，并填充数据
        if dtype in ["bool", "boolean"]:
            data = np.repeat([True, False], N // 2)
        elif dtype in ["int64", "Int64"]:
            data = np.arange(N)
        elif dtype in ["float64", "Float64"]:
            data = np.random.randn(N)
        elif dtype == "datetime64[ns]":
            data = date_range("2000-01-01", freq="s", periods=N)
        else:
            raise NotImplementedError
        
        # 创建 Series 对象，数据类型为指定的 dtype
        self.s = Series(data, dtype=dtype)

    # 测试 Series 对象的迭代性能
    def time_iter(self, dtype):
        for v in self.s:
            pass


# 定义一个名为 ToNumpy 的类，用于测试 Series 对象的 to_numpy() 方法的性能
class ToNumpy:
    # 设置方法，在测试运行前准备数据
    def setup(self):
        N = 1_000_000
        # 创建一个包含 1000000 个随机数的 Series 对象
        self.ser = Series(
            np.random.randn(
                N,
            )
        )

    # 测试 to_numpy() 方法的运行时间
    def time_to_numpy(self):
        self.ser.to_numpy()

    # 测试带有复制操作的 to_numpy() 方法的运行时间
    def time_to_numpy_double_copy(self):
        self.ser.to_numpy(dtype="float64", copy=True)

    # 测试带有复制操作的 to_numpy() 方法的运行时间
    def time_to_numpy_copy(self):
        self.ser.to_numpy(copy=True)

    # 测试带有 NaN 值的 to_numpy() 方法的运行时间
    def time_to_numpy_float_with_nan(self):
        self.ser.to_numpy(dtype="float64", na_value=np.nan)


# 定义一个名为 Replace 的类，用于测试 Series 对象的 replace() 方法的性能
class Replace:
    # 参数名称，指定要替换的数目
    param_names = ["num_to_replace"]
    # 定义一个包含两个整数元素的列表 params，分别是 100 和 1000
    params = [100, 1000]

    # 设置方法，用于初始化测试环境
    def setup(self, num_to_replace):
        # 设定常量 N 为 1000000，用于生成随机数数组
        N = 1_000_000
        # 生成一个包含 N 个随机数的 NumPy 数组 self.arr
        self.arr = np.random.randn(N)
        # 将 self.arr 复制到 self.arr1
        self.arr1 = self.arr.copy()
        # 对 self.arr1 进行乱序操作
        np.random.shuffle(self.arr1)
        # 创建一个 Pandas Series 对象 self.ser，其数据源自 self.arr
        self.ser = Series(self.arr)

        # 从 self.arr 中随机选择 num_to_replace 个元素，构成列表 self.to_replace_list
        self.to_replace_list = np.random.choice(self.arr, num_to_replace)
        # 从 self.arr1 中随机选择 num_to_replace 个元素，构成列表 self.values_list
        self.values_list = np.random.choice(self.arr1, num_to_replace)

        # 创建一个字典 self.replace_dict，将 self.to_replace_list 的元素作为键，self.values_list 的元素作为值
        self.replace_dict = dict(zip(self.to_replace_list, self.values_list))

    # 计时方法，用于测试字典替换操作的时间性能
    def time_replace_dict(self, num_to_replace):
        # 调用 Pandas Series 的 replace 方法，使用 self.replace_dict 进行替换
        self.ser.replace(self.replace_dict)

    # 记录内存峰值方法，用于测试字典替换操作的内存使用情况
    def peakmem_replace_dict(self, num_to_replace):
        # 调用 Pandas Series 的 replace 方法，使用 self.replace_dict 进行替换
        self.ser.replace(self.replace_dict)

    # 计时方法，用于测试列表替换操作的时间性能
    def time_replace_list(self, num_to_replace):
        # 调用 Pandas Series 的 replace 方法，使用 self.to_replace_list 和 self.values_list 进行替换
        self.ser.replace(self.to_replace_list, self.values_list)

    # 记录内存峰值方法，用于测试列表替换操作的内存使用情况
    def peakmem_replace_list(self, num_to_replace):
        # 调用 Pandas Series 的 replace 方法，使用 self.to_replace_list 和 self.values_list 进行替换
        self.ser.replace(self.to_replace_list, self.values_list)
# 从当前目录下的 pandas_vb_common 模块中导入 setup 函数
# 使用 noqa: F401 isort:skip 来告知 linter 工具忽略 F401 错误（未使用的导入）和 isort 排序检查
from .pandas_vb_common import setup  # noqa: F401 isort:skip
```