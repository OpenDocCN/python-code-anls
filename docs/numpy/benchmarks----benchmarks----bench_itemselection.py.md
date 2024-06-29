# `.\numpy\benchmarks\benchmarks\bench_itemselection.py`

```
# 从common模块中导入Benchmark类和TYPES1变量
from .common import Benchmark, TYPES1

# 导入numpy库并将其命名为np
import numpy as np

# 定义Take类，继承Benchmark类
class Take(Benchmark):
    # 定义params列表，包含三个参数列表的组合
    params = [
        [(1000, 1), (2, 1000, 1), (1000, 3)],  # 不同的数组形状
        ["raise", "wrap", "clip"],           # 不同的模式
        TYPES1 + ["O", "i,O"]                # 不同的数据类型
    ]
    # 定义param_names列表，列出各个参数列表的名称
    param_names = ["shape", "mode", "dtype"]

    # 定义setup方法，初始化测试环境
    def setup(self, shape, mode, dtype):
        # 创建一个形状为shape，数据类型为dtype的全1数组，并将其赋值给self.arr
        self.arr = np.ones(shape, dtype)
        # 创建一个包含1000个元素的数组，赋值给self.indices
        self.indices = np.arange(1000)

    # 定义time_contiguous方法，测试连续取值操作的性能
    def time_contiguous(self, shape, mode, dtype):
        # 在指定轴上使用给定的取值模式(mode)，从self.arr中按照self.indices取值
        self.arr.take(self.indices, axis=-2, mode=mode)


# 定义PutMask类，继承Benchmark类
class PutMask(Benchmark):
    # 定义params列表，包含两个参数列表的组合
    params = [
        [True, False],            # 布尔参数，指示值是否标量
        TYPES1 + ["O", "i,O"]     # 不同的数据类型
    ]
    # 定义param_names列表，列出各个参数列表的名称
    param_names = ["values_is_scalar", "dtype"]

    # 定义setup方法，初始化测试环境
    def setup(self, values_is_scalar, dtype):
        # 根据values_is_scalar的值选择性地创建标量或者1000个元素的数组，并赋值给self.vals
        if values_is_scalar:
            self.vals = np.array(1., dtype=dtype)
        else:
            self.vals = np.ones(1000, dtype=dtype)

        # 创建一个长度为1000的全1数组，并赋值给self.arr
        self.arr = np.ones(1000, dtype=dtype)

        # 创建一个长度为1000的布尔数组，所有元素为True，并赋值给self.dense_mask
        self.dense_mask = np.ones(1000, dtype="bool")
        # 创建一个长度为1000的布尔数组，所有元素为False，并赋值给self.sparse_mask
        self.sparse_mask = np.zeros(1000, dtype="bool")

    # 定义time_dense方法，测试稠密掩码操作的性能
    def time_dense(self, values_is_scalar, dtype):
        # 使用np.putmask函数，根据self.dense_mask对self.arr应用self.vals
        np.putmask(self.arr, self.dense_mask, self.vals)

    # 定义time_sparse方法，测试稀疏掩码操作的性能
    def time_sparse(self, values_is_scalar, dtype):
        # 使用np.putmask函数，根据self.sparse_mask对self.arr应用self.vals
        np.putmask(self.arr, self.sparse_mask, self.vals)


# 定义Put类，继承Benchmark类
class Put(Benchmark):
    # 定义params列表，包含两个参数列表的组合
    params = [
        [True, False],            # 布尔参数，指示值是否标量
        TYPES1 + ["O", "i,O"]     # 不同的数据类型
    ]
    # 定义param_names列表，列出各个参数列表的名称
    param_names = ["values_is_scalar", "dtype"]

    # 定义setup方法，初始化测试环境
    def setup(self, values_is_scalar, dtype):
        # 根据values_is_scalar的值选择性地创建标量或者1000个元素的数组，并赋值给self.vals
        if values_is_scalar:
            self.vals = np.array(1., dtype=dtype)
        else:
            self.vals = np.ones(1000, dtype=dtype)

        # 创建一个长度为1000的全1数组，并赋值给self.arr
        self.arr = np.ones(1000, dtype=dtype)
        # 创建一个长度为1000的整数数组，值从0到999，并赋值给self.indx
        self.indx = np.arange(1000, dtype=np.intp)

    # 定义time_ordered方法，测试有序放置操作的性能
    def time_ordered(self, values_is_scalar, dtype):
        # 使用np.put函数，根据self.indx将self.vals放置到self.arr中
        np.put(self.arr, self.indx, self.vals)
```