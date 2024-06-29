# `D:\src\scipysrc\pandas\asv_bench\benchmarks\indexing_engines.py`

```
"""
Benchmarks in this file depend mostly on code in _libs/

We have to created masked arrays to test the masked engine though. The
array is unpacked on the Cython level.

If a PR does not edit anything in _libs, it is very unlikely that benchmarks
in this file will be affected.
"""

# 导入所需的库
import numpy as np

# 从 pandas._libs 中导入 index 模块
from pandas._libs import index as libindex

# 从 pandas.core.arrays 中导入 BaseMaskedArray 类
from pandas.core.arrays import BaseMaskedArray


# 函数：获取所有的数值引擎
def _get_numeric_engines():
    # 定义数值引擎的名称和对应的数据类型
    engine_names = [
        ("Int64Engine", np.int64),
        ("Int32Engine", np.int32),
        ("Int16Engine", np.int16),
        ("Int8Engine", np.int8),
        ("UInt64Engine", np.uint64),
        ("UInt32Engine", np.uint32),
        ("UInt16engine", np.uint16),
        ("UInt8Engine", np.uint8),
        ("Float64Engine", np.float64),
        ("Float32Engine", np.float32),
    ]
    # 返回存在于 libindex 中的引擎名称和数据类型的列表
    return [
        (getattr(libindex, engine_name), dtype)
        for engine_name, dtype in engine_names
        if hasattr(libindex, engine_name)
    ]


# 函数：获取所有的带掩码引擎
def _get_masked_engines():
    # 定义带掩码引擎的名称和对应的数据类型
    engine_names = [
        ("MaskedInt64Engine", "Int64"),
        ("MaskedInt32Engine", "Int32"),
        ("MaskedInt16Engine", "Int16"),
        ("MaskedInt8Engine", "Int8"),
        ("MaskedUInt64Engine", "UInt64"),
        ("MaskedUInt32Engine", "UInt32"),
        ("MaskedUInt16engine", "UInt16"),
        ("MaskedUInt8Engine", "UInt8"),
        ("MaskedFloat64Engine", "Float64"),
        ("MaskedFloat32Engine", "Float32"),
    ]
    # 返回存在于 libindex 中的带掩码引擎名称和数据类型的列表
    return [
        (getattr(libindex, engine_name), dtype)
        for engine_name, dtype in engine_names
        if hasattr(libindex, engine_name)
    ]


# 类：数值引擎索引
class NumericEngineIndexing:
    # 参数定义
    params = [
        _get_numeric_engines(),          # 数值引擎及其数据类型
        ["monotonic_incr", "monotonic_decr", "non_monotonic"],  # 索引类型
        [True, False],                   # 是否唯一
        [10**5, 2 * 10**6],              # 数据量 N
    ]
    param_names = ["engine_and_dtype", "index_type", "unique", "N"]  # 参数名称

    # 设置方法：初始化测试环境
    def setup(self, engine_and_dtype, index_type, unique, N):
        engine, dtype = engine_and_dtype

        # 根据索引类型生成不同的数组
        if index_type == "monotonic_incr":
            if unique:
                arr = np.arange(N * 3, dtype=dtype)
            else:
                arr = np.array([1, 2, 3], dtype=dtype).repeat(N)
        elif index_type == "monotonic_decr":
            if unique:
                arr = np.arange(N * 3, dtype=dtype)[::-1]
            else:
                arr = np.array([3, 2, 1], dtype=dtype).repeat(N)
        else:
            assert index_type == "non_monotonic"
            if unique:
                arr = np.empty(N * 3, dtype=dtype)
                arr[:N] = np.arange(N * 2, N * 3, dtype=dtype)
                arr[N:] = np.arange(N * 2, dtype=dtype)
            else:
                arr = np.array([1, 2, 3], dtype=dtype).repeat(N)

        # 使用引擎创建数据对象
        self.data = engine(arr)
        
        # 在计时时，通过获取位置信息来避免填充映射等操作
        self.data.get_loc(2)

        # 设置中间键和早期键
        self.key_middle = arr[len(arr) // 2]
        self.key_early = arr[2]
    # 定义一个方法用于从数据中获取特定键的位置
    def time_get_loc(self, engine_and_dtype, index_type, unique, N):
        # 调用数据对象的方法，获取指定键在数据中的位置
        self.data.get_loc(self.key_early)

    # 定义另一个方法用于从数据中获取接近中间位置的键的位置
    def time_get_loc_near_middle(self, engine_and_dtype, index_type, unique, N):
        # 当在一个范围的中间位置进行搜索时，searchsorted 的性能可能与在端点附近时不同
        # 调用数据对象的方法，获取接近中间位置的键在数据中的位置
        self.data.get_loc(self.key_middle)
# 定义一个名为 MaskedNumericEngineIndexing 的类，用于测试不同类型的索引引擎性能
class MaskedNumericEngineIndexing:
    # 参数化的测试参数列表，包括调用 _get_masked_engines() 函数返回的结果
    params = [
        _get_masked_engines(),  # 获取掩码引擎列表作为第一个参数
        ["monotonic_incr", "monotonic_decr", "non_monotonic"],  # 索引类型列表
        [True, False],  # 是否唯一的布尔值列表
        [10**5, 2 * 10**6],  # N 值列表，其中 2e6 超过了 SIZE_CUTOFF
    ]
    # 参数名称列表
    param_names = ["engine_and_dtype", "index_type", "unique", "N"]

    # 设置函数，用于初始化测试环境和数据
    def setup(self, engine_and_dtype, index_type, unique, N):
        # 解包引擎和数据类型
        engine, dtype = engine_and_dtype
        # 将数据类型转换为小写
        dtype = dtype.lower()

        # 根据索引类型初始化数组 arr 和掩码 mask
        if index_type == "monotonic_incr":
            if unique:
                # 如果唯一，创建递增的数组
                arr = np.arange(N * 3, dtype=dtype)
            else:
                # 如果不唯一，重复 [1, 2, 3] 构成的数组 N 次
                arr = np.array([1, 2, 3], dtype=dtype).repeat(N)
            # 初始化全零掩码数组
            mask = np.zeros(N * 3, dtype=np.bool_)
        elif index_type == "monotonic_decr":
            if unique:
                # 如果唯一，创建递减的数组
                arr = np.arange(N * 3, dtype=dtype)[::-1]
            else:
                # 如果不唯一，重复 [3, 2, 1] 构成的数组 N 次
                arr = np.array([3, 2, 1], dtype=dtype).repeat(N)
            # 初始化全零掩码数组
            mask = np.zeros(N * 3, dtype=np.bool_)
        else:
            # 当索引类型为非单调时
            assert index_type == "non_monotonic"
            if unique:
                # 如果唯一，构造特定的非单调数组
                arr = np.zeros(N * 3, dtype=dtype)
                arr[:N] = np.arange(N * 2, N * 3, dtype=dtype)
                arr[N:] = np.arange(N * 2, dtype=dtype)
            else:
                # 如果不唯一，重复 [1, 2, 3] 构成的数组 N 次
                arr = np.array([1, 2, 3], dtype=dtype).repeat(N)
            # 初始化全零掩码数组，并将最后一个元素设置为 True
            mask = np.zeros(N * 3, dtype=np.bool_)
            mask[-1] = True

        # 使用 BaseMaskedArray 类构造掩码数组并传入引擎，保存在 self.data 中
        self.data = engine(BaseMaskedArray(arr, mask))
        
        # 在计时时，避免填充映射等操作，直接调用 get_loc(2)
        self.data.get_loc(2)

        # 计算数组中间元素和第二个元素的值并保存在实例中
        self.key_middle = arr[len(arr) // 2]
        self.key_early = arr[2]

    # 测试 get_loc 方法的性能
    def time_get_loc(self, engine_and_dtype, index_type, unique, N):
        # 调用 get_loc 方法测试关键字 self.key_early 的性能
        self.data.get_loc(self.key_early)

    # 测试在接近数组中间位置和末尾位置时，searchsorted 方法的性能是否有所不同
    def time_get_loc_near_middle(self, engine_and_dtype, index_type, unique, N):
        # 调用 get_loc 方法测试关键字 self.key_middle 的性能
        # 在接近数组中间位置时，searchsorted 的性能可能会不同于接近端点位置时
        self.data.get_loc(self.key_middle)


# 定义一个名为 ObjectEngineIndexing 的类，用于测试对象类型数组的索引引擎性能
class ObjectEngineIndexing:
    # 参数化的测试参数列表，只包含索引类型
    params = [("monotonic_incr", "monotonic_decr", "non_monotonic")]
    # 参数名称列表
    param_names = ["index_type"]

    # 设置函数，用于初始化测试环境和数据
    def setup(self, index_type):
        N = 10**5
        # 创建一个包含 N 个 'a'、N 个 'b' 和 N 个 'c' 的列表
        values = list("a" * N + "b" * N + "c" * N)
        # 根据索引类型选择特定类型的对象数组 arr
        arr = {
            "monotonic_incr": np.array(values, dtype=object),  # 递增数组
            "monotonic_decr": np.array(list(reversed(values)), dtype=object),  # 递减数组
            "non_monotonic": np.array(list("abc") * N, dtype=object),  # 非单调数组
        }[index_type]

        # 使用 ObjectEngine 类构造对象类型的引擎，并保存在 self.data 中
        self.data = libindex.ObjectEngine(arr)
        
        # 在计时时，避免填充映射等操作，直接调用 get_loc("b")
        self.data.get_loc("b")

    # 测试 get_loc 方法的性能
    def time_get_loc(self, index_type):
        # 调用 get_loc 方法测试关键字 "b" 的性能
        self.data.get_loc("b")
```