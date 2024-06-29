# `.\numpy\benchmarks\benchmarks\bench_overrides.py`

```
# 导入Benchmark类，用于性能基准测试
from .common import Benchmark

try:
    # 尝试导入新版Numpy的array_function_dispatch
    from numpy._core.overrides import array_function_dispatch
except ImportError:
    # 如果导入失败，定义一个兼容旧版Numpy的array_function_dispatch函数
    def array_function_dispatch(*args, **kwargs):
        def wrap(*args, **kwargs):
            return None
        return wrap

# 导入Numpy库并使用np作为别名
import numpy as np


def _broadcast_to_dispatcher(array, shape, subok=None):
    return (array,)


# 使用array_function_dispatch装饰器将_broadcast_to_dispatcher注册为mock_broadcast_to的分发器
@array_function_dispatch(_broadcast_to_dispatcher)
def mock_broadcast_to(array, shape, subok=False):
    pass


def _concatenate_dispatcher(arrays, axis=None, out=None):
    if out is not None:
        # 如果提供了输出参数out，则将其追加到arrays列表中
        arrays = list(arrays)
        arrays.append(out)
    return arrays


# 使用array_function_dispatch装饰器将_concatenate_dispatcher注册为mock_concatenate的分发器
@array_function_dispatch(_concatenate_dispatcher)
def mock_concatenate(arrays, axis=0, out=None):
    pass


# 定义一个DuckArray类，用于模拟Numpy数组的行为
class DuckArray:
    def __array_function__(self, func, types, args, kwargs):
        pass


# ArrayFunction类继承自Benchmark类，用于对数组操作的性能进行基准测试
class ArrayFunction(Benchmark):

    def setup(self):
        # 初始化各种类型的数组用于测试
        self.numpy_array = np.array(1)
        self.numpy_arrays = [np.array(1), np.array(2)]
        self.many_arrays = 500 * self.numpy_arrays
        self.duck_array = DuckArray()
        self.duck_arrays = [DuckArray(), DuckArray()]
        self.mixed_arrays = [np.array(1), DuckArray()]

    # 测试mock_broadcast_to函数对Numpy数组的调用性能
    def time_mock_broadcast_to_numpy(self):
        mock_broadcast_to(self.numpy_array, ())

    # 测试mock_broadcast_to函数对DuckArray对象的调用性能
    def time_mock_broadcast_to_duck(self):
        mock_broadcast_to(self.duck_array, ())

    # 测试mock_concatenate函数对Numpy数组列表的调用性能
    def time_mock_concatenate_numpy(self):
        mock_concatenate(self.numpy_arrays, axis=0)

    # 测试mock_concatenate函数对大量Numpy数组列表的调用性能
    def time_mock_concatenate_many(self):
        mock_concatenate(self.many_arrays, axis=0)

    # 测试mock_concatenate函数对DuckArray对象列表的调用性能
    def time_mock_concatenate_duck(self):
        mock_concatenate(self.duck_arrays, axis=0)

    # 测试mock_concatenate函数对混合类型数组列表的调用性能
    def time_mock_concatenate_mixed(self):
        mock_concatenate(self.mixed_arrays, axis=0)
```