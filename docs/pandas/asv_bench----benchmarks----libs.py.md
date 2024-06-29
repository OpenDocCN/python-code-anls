# `D:\src\scipysrc\pandas\asv_bench\benchmarks\libs.py`

```
"""
pandas/_libs 目录中代码的基准测试，不包括 pandas/_libs/tslibs 目录。

如果一个 PR 没有编辑 _libs/ 中的任何内容，那么基准测试可能不会受到影响。
"""

# 导入 NumPy 库
import numpy as np

# 从 pandas._libs.lib 中导入特定函数
from pandas._libs.lib import (
    infer_dtype,
    is_list_like,
    is_scalar,
)

# 从 pandas 中导入常用对象
from pandas import (
    NA,
    Index,
    NaT,
)

# 从当前目录中的 pandas_vb_common 模块中导入 lib
from .pandas_vb_common import lib

# 尝试从 pandas.util 中导入 cache_readonly 函数，如果失败则从 pandas.util.decorators 导入
try:
    from pandas.util import cache_readonly
except ImportError:
    from pandas.util.decorators import cache_readonly


# TODO: 与 pd._testing 中的某些内容共享？
# 标量对象列表
scalars = [
    0,
    1.0,
    1 + 2j,
    True,
    "foo",
    b"bar",
    None,
    np.datetime64(123, "ns"),
    np.timedelta64(123, "ns"),
    NaT,
    NA,
]

# 零维数组列表
zero_dims = [np.array("123")]

# 类似列表对象列表
listlikes = [
    np.array([1, 2, 3]),
    {0: 1},
    {1, 2, 3},
    [1, 2, 3],
    (1, 2, 3),
]


class ScalarListLike:
    # 参数为标量对象、零维数组和类似列表对象的组合
    params = scalars + zero_dims + listlikes

    # 测试 is_list_like 函数的运行时间
    def time_is_list_like(self, param):
        is_list_like(param)

    # 测试 is_scalar 函数的运行时间
    def time_is_scalar(self, param):
        is_scalar(param)


class FastZip:
    # 设置初始化环境
    def setup(self):
        N = 10000
        K = 10
        # 创建 key1 和 key2 索引对象，并重复 K 次
        key1 = Index([f"i-{i}" for i in range(N)], dtype=object).values.repeat(K)
        key2 = Index([f"i-{i}" for i in range(N)], dtype=object).values.repeat(K)
        # 创建包含 key1、key2 和随机数据的列数组
        col_array = np.vstack([key1, key2, np.random.randn(N * K)])
        # 创建 col_array 的副本 col_array2，并将前 10000 列设置为 NaN
        col_array2 = col_array.copy()
        col_array2[:, :10000] = np.nan
        # 将 col_array 转换为列表并存储在 col_array_list 中
        self.col_array_list = list(col_array)

    # 测试 lib.fast_zip 函数的运行时间
    def time_lib_fast_zip(self):
        lib.fast_zip(self.col_array_list)


class InferDtype:
    # 参数名称为 dtype
    param_names = ["dtype"]
    # 数据字典包含多种数据类型的数据示例
    data_dict = {
        "np-object": np.array([1] * 100000, dtype="O"),
        "py-object": [1] * 100000,
        "np-null": np.array([1] * 50000 + [np.nan] * 50000),
        "py-null": [1] * 50000 + [None] * 50000,
        "np-int": np.array([1] * 100000, dtype=int),
        "np-floating": np.array([1.0] * 100000, dtype=float),
        "empty": [],
        "bytes": [b"a"] * 100000,
    }
    # 参数为数据字典的键列表
    params = list(data_dict.keys())

    # 测试 infer_dtype 函数在 skipna=True 时的运行时间
    def time_infer_dtype_skipna(self, dtype):
        infer_dtype(self.data_dict[dtype], skipna=True)

    # 测试 infer_dtype 函数在 skipna=False 时的运行时间
    def time_infer_dtype(self, dtype):
        infer_dtype(self.data_dict[dtype], skipna=False)


class CacheReadonly:
    # 设置初始化环境
    def setup(self):
        # 定义一个 Foo 类，其中 prop 属性使用 cache_readonly 装饰器装饰
        class Foo:
            @cache_readonly
            def prop(self):
                return 5

        # 创建 Foo 类的实例对象
        self.obj = Foo()

    # 测试 self.obj.prop 的访问时间
    def time_cache_readonly(self):
        self.obj.prop
```