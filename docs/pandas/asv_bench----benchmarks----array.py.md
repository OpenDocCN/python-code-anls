# `D:\src\scipysrc\pandas\asv_bench\benchmarks\array.py`

```
import numpy as np  # 导入 NumPy 库，用于处理数值数组

import pandas as pd  # 导入 Pandas 库，用于数据分析和处理


class BooleanArray:
    def setup(self):
        self.values_bool = np.array([True, False, True, False])  # 创建布尔类型的 NumPy 数组
        self.values_float = np.array([1.0, 0.0, 1.0, 0.0])  # 创建浮点数类型的 NumPy 数组
        self.values_integer = np.array([1, 0, 1, 0])  # 创建整数类型的 NumPy 数组
        self.values_integer_like = [1, 0, 1, 0]  # 创建类整数的 Python 列表
        self.data = np.array([True, False, True, False])  # 创建布尔类型的 NumPy 数组
        self.mask = np.array([False, False, True, False])  # 创建布尔类型的 NumPy 数组

    def time_constructor(self):
        pd.arrays.BooleanArray(self.data, self.mask)  # 使用 Pandas 构造布尔类型数组对象

    def time_from_bool_array(self):
        pd.array(self.values_bool, dtype="boolean")  # 将布尔类型的 NumPy 数组转换为 Pandas 的布尔类型数组

    def time_from_integer_array(self):
        pd.array(self.values_integer, dtype="boolean")  # 将整数类型的 NumPy 数组转换为 Pandas 的布尔类型数组

    def time_from_integer_like(self):
        pd.array(self.values_integer_like, dtype="boolean")  # 将类整数的 Python 列表转换为 Pandas 的布尔类型数组

    def time_from_float_array(self):
        pd.array(self.values_float, dtype="boolean")  # 将浮点数类型的 NumPy 数组转换为 Pandas 的布尔类型数组


class IntegerArray:
    def setup(self):
        N = 250_000
        self.values_integer = np.tile(np.array([1, 0, 1, 0]), N)  # 使用 NumPy 的 tile 函数复制数组以创建大数组
        self.data = np.tile(np.array([1, 2, 3, 4], dtype="int64"), N)  # 使用 NumPy 的 tile 函数复制数组以创建大数组
        self.mask = np.tile(np.array([False, False, True, False]), N)  # 使用 NumPy 的 tile 函数复制数组以创建大数组

    def time_constructor(self):
        pd.arrays.IntegerArray(self.data, self.mask)  # 使用 Pandas 构造整数类型数组对象

    def time_from_integer_array(self):
        pd.array(self.values_integer, dtype="Int64")  # 将整数类型的 NumPy 数组转换为 Pandas 的整数类型数组


class IntervalArray:
    def setup(self):
        N = 10_000
        self.tuples = [(i, i + 1) for i in range(N)]  # 创建包含元组的列表，每个元组代表一个区间

    def time_from_tuples(self):
        pd.arrays.IntervalArray.from_tuples(self.tuples)  # 使用 Pandas 从元组列表创建区间数组对象


class StringArray:
    def setup(self):
        N = 100_000
        values = np.array([str(i) for i in range(N)], dtype=object)  # 创建包含字符串的 NumPy 数组
        self.values_obj = np.array(values, dtype="object")  # 创建包含字符串的 NumPy 数组
        self.values_str = np.array(values, dtype="U")  # 创建 Unicode 字符串类型的 NumPy 数组
        self.values_list = values.tolist()  # 将 NumPy 数组转换为 Python 列表

    def time_from_np_object_array(self):
        pd.array(self.values_obj, dtype="string")  # 将对象类型的 NumPy 数组转换为 Pandas 的字符串类型数组

    def time_from_np_str_array(self):
        pd.array(self.values_str, dtype="string")  # 将 Unicode 字符串类型的 NumPy 数组转换为 Pandas 的字符串类型数组

    def time_from_list(self):
        pd.array(self.values_list, dtype="string")  # 将 Python 列表转换为 Pandas 的字符串类型数组


class ArrowStringArray:
    params = [False, True]
    param_names = ["multiple_chunks"]

    def setup(self, multiple_chunks):
        try:
            import pyarrow as pa  # 尝试导入 PyArrow 库，用于处理箭头格式的数据
        except ImportError as err:
            raise NotImplementedError from err
        strings = np.array([str(i) for i in range(10_000)], dtype=object)  # 创建包含字符串的 NumPy 数组
        if multiple_chunks:
            chunks = [strings[i : i + 100] for i in range(0, len(strings), 100)]  # 将字符串数组切分成多个块
            self.array = pd.arrays.ArrowStringArray(pa.chunked_array(chunks))  # 使用 PyArrow 创建 ArrowStringArray 对象
        else:
            self.array = pd.arrays.ArrowStringArray(pa.array(strings))  # 使用 PyArrow 创建 ArrowStringArray 对象

    def time_setitem(self, multiple_chunks):
        for i in range(200):
            self.array[i] = "foo"  # 修改 ArrowStringArray 中的元素为字符串 "foo"

    def time_setitem_list(self, multiple_chunks):
        indexer = list(range(50)) + list(range(-1000, 0, 50))
        self.array[indexer] = ["foo"] * len(indexer)  # 使用列表修改 ArrowStringArray 中的部分元素为字符串 "foo"
    # 定义一个方法 `time_setitem_slice`，接收参数 `multiple_chunks`
    def time_setitem_slice(self, multiple_chunks):
        # 使用切片操作，每隔10个元素设置值为 "foo"
        self.array[::10] = "foo"
    
    # 定义一个方法 `time_setitem_null_slice`，接收参数 `multiple_chunks`
    def time_setitem_null_slice(self, multiple_chunks):
        # 使用空切片操作，将整个数组设置值为 "foo"
        self.array[:] = "foo"
    
    # 定义一个方法 `time_tolist`，接收参数 `multiple_chunks`
    def time_tolist(self, multiple_chunks):
        # 将数组转换为列表
        self.array.tolist()
# 定义一个自定义的 Arrow 扩展数组类
class ArrowExtensionArray:
    # 定义参数列表，包括数据类型和是否包含缺失值两个部分
    params = [
        [
            "boolean[pyarrow]",        # 布尔类型数组（使用 PyArrow）
            "float64[pyarrow]",        # 浮点数类型数组（使用 PyArrow）
            "int64[pyarrow]",          # 整数类型数组（使用 PyArrow）
            "string[pyarrow]",         # 字符串类型数组（使用 PyArrow）
            "timestamp[ns][pyarrow]",  # 时间戳类型数组（使用 PyArrow）
        ],
        [False, True],  # 是否包含缺失值的选项：False 表示不包含，True 表示包含
    ]
    param_names = ["dtype", "hasna"]  # 参数名列表：数据类型和是否包含缺失值

    # 初始化方法，在测试之前生成测试数据
    def setup(self, dtype, hasna):
        N = 100_000  # 生成数据的数量
        # 根据数据类型生成相应的测试数据
        if dtype == "boolean[pyarrow]":
            data = np.random.choice([True, False], N, replace=True)
        elif dtype == "float64[pyarrow]":
            data = np.random.randn(N)
        elif dtype == "int64[pyarrow]":
            data = np.arange(N)
        elif dtype == "string[pyarrow]":
            data = np.array([str(i) for i in range(N)], dtype=object)
        elif dtype == "timestamp[ns][pyarrow]":
            data = pd.date_range("2000-01-01", freq="s", periods=N)
        else:
            raise NotImplementedError("Unsupported dtype")  # 如果遇到未支持的数据类型，则抛出异常

        # 使用 Pandas 创建 Arrow 扩展数组对象
        arr = pd.array(data, dtype=dtype)
        # 如果 hasna 为 True，则将数组的偶数索引位置设为缺失值 NA
        if hasna:
            arr[::2] = pd.NA
        self.arr = arr  # 将生成的数组存储到对象的属性中

    # 测试方法：将 Arrow 扩展数组转换为 NumPy 数组的性能测试
    def time_to_numpy(self, dtype, hasna):
        self.arr.to_numpy()  # 转换为 NumPy 数组并返回
```