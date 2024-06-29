# `.\numpy\benchmarks\benchmarks\bench_array_coercion.py`

```
# 从.common模块导入Benchmark类，用于性能基准测试
from .common import Benchmark
# 导入NumPy库，用于数组操作
import numpy as np

# 定义ArrayCoercionSmall类，继承Benchmark类，用于数组类型转换的详细性能基准测试
class ArrayCoercionSmall(Benchmark):
    # 参数列表，包含多种类型的数组或类数组对象，用于不同的性能测试
    params = [[range(3), [1], 1, np.array([5], dtype=np.int64), np.int64(5)]]
    # 参数名列表，描述params中各参数的含义
    param_names = ['array_like']
    # 定义一个np.int64类型的数据类型对象
    int64 = np.dtype(np.int64)

    # 测试函数：测试在使用无效关键字参数时调用np.array(array_like)的性能
    def time_array_invalid_kwarg(self, array_like):
        try:
            np.array(array_like, ndmin="not-integer")
        except TypeError:
            pass

    # 测试函数：测试调用np.array(array_like)的性能
    def time_array(self, array_like):
        np.array(array_like)

    # 测试函数：测试在使用非关键字参数（如dtype=self.int64）时调用np.array(array_like)的性能
    def time_array_dtype_not_kwargs(self, array_like):
        np.array(array_like, self.int64)

    # 测试函数：测试在使用copy=None参数时调用np.array(array_like)的性能
    def time_array_no_copy(self, array_like):
        np.array(array_like, copy=None)

    # 测试函数：测试在使用subok=True参数时调用np.array(array_like)的性能
    def time_array_subok(self, array_like):
        np.array(array_like, subok=True)

    # 测试函数：测试在使用多个关键字参数时调用np.array(array_like)的性能
    def time_array_all_kwargs(self, array_like):
        np.array(array_like, dtype=self.int64, copy=None, order="F",
                 subok=False, ndmin=2)

    # 测试函数：测试调用np.asarray(array_like)的性能
    def time_asarray(self, array_like):
        np.asarray(array_like)

    # 测试函数：测试在使用dtype=self.int64参数时调用np.asarray(array_like)的性能
    def time_asarray_dtype(self, array_like):
        np.asarray(array_like, dtype=self.int64)

    # 测试函数：测试在使用dtype=self.int64和order="F"参数时调用np.asarray(array_like)的性能
    def time_asarray_dtype_order(self, array_like):
        np.asarray(array_like, dtype=self.int64, order="F")

    # 测试函数：测试调用np.asanyarray(array_like)的性能
    def time_asanyarray(self, array_like):
        np.asanyarray(array_like)

    # 测试函数：测试在使用dtype=self.int64参数时调用np.asanyarray(array_like)的性能
    def time_asanyarray_dtype(self, array_like):
        np.asanyarray(array_like, dtype=self.int64)

    # 测试函数：测试在使用dtype=self.int64和order="F"参数时调用np.asanyarray(array_like)的性能
    def time_asanyarray_dtype_order(self, array_like):
        np.asanyarray(array_like, dtype=self.int64, order="F")

    # 测试函数：测试调用np.ascontiguousarray(array_like)的性能
    def time_ascontiguousarray(self, array_like):
        np.ascontiguousarray(array_like)
```