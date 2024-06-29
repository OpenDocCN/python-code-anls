# `.\numpy\benchmarks\benchmarks\bench_creation.py`

```py
# 从当前包中导入Benchmark类、TYPES1常量和get_squares_函数
from .common import Benchmark, TYPES1, get_squares_

# 导入NumPy库，并使用np作为别名
import numpy as np

# Benchmark类的子类，用于测量meshgrid生成的性能
class MeshGrid(Benchmark):
    """ Benchmark meshgrid generation
    """
    # 参数化设置：
    # size：两个整数列表 [16, 32]
    # ndims：三个整数列表 [2, 3, 4]
    # ind：两个字符串列表 ['ij', 'xy']
    # ndtype：TYPES1常量
    params = [[16, 32],
              [2, 3, 4],
              ['ij', 'xy'], TYPES1]
    # 参数名称：
    # size - size参数的名称
    # ndims - ndims参数的名称
    # ind - ind参数的名称
    # ndtype - ndtype参数的名称
    param_names = ['size', 'ndims', 'ind', 'ndtype']
    # 设置超时时间为10秒
    timeout = 10

    # 初始化函数，设置网格维度数组
    def setup(self, size, ndims, ind, ndtype):
        # 使用种子值1864768776创建随机状态对象rnd
        rnd = np.random.RandomState(1864768776)
        # 生成ndims个网格维度数组，每个数组长度为size，元素类型为ndtype
        self.grid_dims = [(rnd.random_sample(size)).astype(ndtype) for
                          x in range(ndims)]

    # 测量meshgrid函数运行时间的函数
    def time_meshgrid(self, size, ndims, ind, ndtype):
        # 调用NumPy的meshgrid函数生成网格
        np.meshgrid(*self.grid_dims, indexing=ind)


# Benchmark类的子类，用于测量创建函数的性能
class Create(Benchmark):
    """ Benchmark for creation functions
    """
    # 参数化设置：
    # shape：三个不同形状的列表 [16, 512, (32, 32)]
    # npdtypes：TYPES1常量
    params = [[16, 512, (32, 32)],
              TYPES1]
    # 参数名称：
    # shape - shape参数的名称
    # npdtypes - npdtypes参数的名称
    param_names = ['shape', 'npdtypes']
    # 设置超时时间为10秒
    timeout = 10

    # 初始化函数，获取方形值并设置xarg变量
    def setup(self, shape, npdtypes):
        # 调用get_squares_函数获取方形值字典
        values = get_squares_()
        # 从字典中获取npdtypes对应的方形值数组，并将其作为xarg保存
        self.xarg = values.get(npdtypes)[0]

    # 测量使用np.full函数创建数组的运行时间
    def time_full(self, shape, npdtypes):
        # 使用NumPy的full函数创建指定形状、填充值和数据类型的数组
        np.full(shape, self.xarg[1], dtype=npdtypes)

    # 测量使用np.full_like函数创建数组的运行时间
    def time_full_like(self, shape, npdtypes):
        # 使用NumPy的full_like函数根据现有数组创建形状相同、填充值相同的新数组
        np.full_like(self.xarg, self.xarg[0])

    # 测量使用np.ones函数创建数组的运行时间
    def time_ones(self, shape, npdtypes):
        # 使用NumPy的ones函数创建指定形状、数据类型的全1数组
        np.ones(shape, dtype=npdtypes)

    # 测量使用np.ones_like函数创建数组的运行时间
    def time_ones_like(self, shape, npdtypes):
        # 使用NumPy的ones_like函数根据现有数组创建形状相同、元素值为1的新数组
        np.ones_like(self.xarg)

    # 测量使用np.zeros函数创建数组的运行时间
    def time_zeros(self, shape, npdtypes):
        # 使用NumPy的zeros函数创建指定形状、数据类型的全0数组
        np.zeros(shape, dtype=npdtypes)

    # 测量使用np.zeros_like函数创建数组的运行时间
    def time_zeros_like(self, shape, npdtypes):
        # 使用NumPy的zeros_like函数根据现有数组创建形状相同、元素值为0的新数组
        np.zeros_like(self.xarg)

    # 测量使用np.empty函数创建数组的运行时间
    def time_empty(self, shape, npdtypes):
        # 使用NumPy的empty函数创建指定形状、数据类型的未初始化数组
        np.empty(shape, dtype=npdtypes)

    # 测量使用np.empty_like函数创建数组的运行时间
    def time_empty_like(self, shape, npdtypes):
        # 使用NumPy的empty_like函数根据现有数组创建形状相同、未初始化的新数组
        np.empty_like(self.xarg)


# Benchmark类的子类，用于测量从DLPack创建数组的性能
class UfuncsFromDLP(Benchmark):
    """ Benchmark for creation functions
    """
    # 参数化设置：
    # shape：四个不同形状的列表 [16, 32, (16, 16), (64, 64)]
    # npdtypes：TYPES1常量
    params = [[16, 32, (16, 16), (64, 64)],
              TYPES1]
    # 参数名称：
    # shape - shape参数的名称
    # npdtypes - npdtypes参数的名称
    param_names = ['shape', 'npdtypes']
    # 设置超时时间为10秒
    timeout = 10

    # 初始化函数，获取方形值并设置xarg变量
    def setup(self, shape, npdtypes):
        # 调用get_squares_函数获取方形值字典
        values = get_squares_()
        # 从字典中获取npdtypes对应的方形值数组，并将其作为xarg保存
        self.xarg = values.get(npdtypes)[0]

    # 测量使用np.from_dlpack函数从DLPack数据结构创建数组的运行时间
    def time_from_dlpack(self, shape, npdtypes):
        # 使用NumPy的from_dlpack函数从DLPack数据结构创建数组
        np.from_dlpack(self.xarg)
```