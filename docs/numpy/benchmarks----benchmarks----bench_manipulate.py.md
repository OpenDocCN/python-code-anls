# `.\numpy\benchmarks\benchmarks\bench_manipulate.py`

```
from .common import Benchmark, get_squares_, TYPES1, DLPACK_TYPES
# 导入必要的模块和函数

import numpy as np
# 导入 NumPy 库

from collections import deque
# 导入 deque 数据结构，用于特定的维度操作

class BroadcastArrays(Benchmark):
    # 声明 BroadcastArrays 类，继承自 Benchmark 基类

    params = [[(16, 32), (128, 256), (512, 1024)],
              TYPES1]
    # 参数化设置：数组形状和数据类型的组合

    param_names = ['shape', 'ndtype']
    # 参数名称

    timeout = 10
    # 设置超时时间为 10 秒

    def setup(self, shape, ndtype):
        # 设置方法，用于初始化测试环境
        self.xarg = np.random.ranf(shape[0]*shape[1]).reshape(shape)
        # 创建随机数据数组，并根据指定形状进行重塑
        self.xarg = self.xarg.astype(ndtype)
        # 将数组类型转换为指定的数据类型
        if ndtype.startswith('complex'):
            self.xarg += np.random.ranf(1)*1j
            # 如果数据类型以 'complex' 开头，则添加一个虚数部分

    def time_broadcast_arrays(self, shape, ndtype):
        # 测试方法：broadcast_arrays
        np.broadcast_arrays(self.xarg, np.ones(1))
        # 使用 np.broadcast_arrays 函数广播数组


class BroadcastArraysTo(Benchmark):
    # 声明 BroadcastArraysTo 类，继承自 Benchmark 基类

    params = [[16, 64, 512],
              TYPES1]
    # 参数化设置：数组大小和数据类型的组合

    param_names = ['size', 'ndtype']
    # 参数名称

    timeout = 10
    # 设置超时时间为 10 秒

    def setup(self, size, ndtype):
        # 设置方法，用于初始化测试环境
        self.rng = np.random.default_rng()
        # 创建随机数生成器对象
        self.xarg = self.rng.random(size)
        # 生成指定大小的随机数数组
        self.xarg = self.xarg.astype(ndtype)
        # 将数组类型转换为指定的数据类型
        if ndtype.startswith('complex'):
            self.xarg += self.rng.random(1)*1j
            # 如果数据类型以 'complex' 开头，则添加一个虚数部分

    def time_broadcast_to(self, size, ndtype):
        # 测试方法：broadcast_to
        np.broadcast_to(self.xarg, (size, size))
        # 使用 np.broadcast_to 函数进行广播操作


class ConcatenateStackArrays(Benchmark):
    # 声明 ConcatenateStackArrays 类，继承自 Benchmark 基类

    params = [[(16, 32), (32, 64)],
              [2, 5],
              TYPES1]
    # 参数化设置：数组形状、堆叠数量和数据类型的组合

    param_names = ['shape', 'narrays', 'ndtype']
    # 参数名称

    timeout = 10
    # 设置超时时间为 10 秒

    def setup(self, shape, narrays, ndtype):
        # 设置方法，用于初始化测试环境
        self.xarg = [np.random.ranf(shape[0]*shape[1]).reshape(shape)
                     for x in range(narrays)]
        # 创建包含多个随机数组的列表
        self.xarg = [x.astype(ndtype) for x in self.xarg]
        # 将列表中的每个数组转换为指定的数据类型
        if ndtype.startswith('complex'):
            [x + np.random.ranf(1)*1j for x in self.xarg]
            # 如果数据类型以 'complex' 开头，则为每个数组添加一个虚数部分

    def time_concatenate_ax0(self, shape, narrays, ndtype):
        # 测试方法：concatenate_ax0
        np.concatenate(self.xarg, axis=0)
        # 使用 np.concatenate 函数在 axis=0 上堆叠数组

    def time_concatenate_ax1(self, shape, narrays, ndtype):
        # 测试方法：concatenate_ax1
        np.concatenate(self.xarg, axis=1)
        # 使用 np.concatenate 函数在 axis=1 上堆叠数组

    def time_stack_ax0(self, shape, narrays, ndtype):
        # 测试方法：stack_ax0
        np.stack(self.xarg, axis=0)
        # 使用 np.stack 函数在 axis=0 上堆叠数组

    def time_stack_ax1(self, shape, narrays, ndtype):
        # 测试方法：stack_ax1
        np.stack(self.xarg, axis=1)
        # 使用 np.stack 函数在 axis=1 上堆叠数组


class ConcatenateNestedArrays(ConcatenateStackArrays):
    # 声明 ConcatenateNestedArrays 类，继承自 ConcatenateStackArrays 类

    # Large number of small arrays to test GIL (non-)release
    params = [[(1, 1)], [1000, 100000], TYPES1]
    # 参数化设置：小数组数量大以测试 GIL（非）释放

class DimsManipulations(Benchmark):
    # 声明 DimsManipulations 类，继承自 Benchmark 基类

    params = [
        [(2, 1, 4), (2, 1), (5, 2, 3, 1)],
    ]
    # 参数化设置：数组形状的组合

    param_names = ['shape']
    # 参数名称

    timeout = 10
    # 设置超时时间为 10 秒

    def setup(self, shape):
        # 设置方法，用于初始化测试环境
        self.xarg = np.ones(shape=shape)
        # 创建元素全为 1 的数组，并使用指定形状
        self.reshaped = deque(shape)
        # 创建 deque 对象，存储形状信息
        self.reshaped.rotate(1)
        # 将 deque 中的元素循环移动一个位置
        self.reshaped = tuple(self.reshaped)
        # 将 deque 转换为元组形式

    def time_expand_dims(self, shape):
        # 测试方法：expand_dims
        np.expand_dims(self.xarg, axis=1)
        # 使用 np.expand_dims 函数在 axis=1 上扩展数组的维度

    def time_expand_dims_neg(self, shape):
        # 测试方法：expand_dims_neg
        np.expand_dims(self.xarg, axis=-1)
        # 使用 np.expand_dims 函数在 axis=-1 上扩展数组的维度

    def time_squeeze_dims(self, shape):
        # 测试方法：squeeze_dims
        np.squeeze(self.xarg)
        # 使用 np.squeeze 函数去除数组中的单维度

    def time_flip_all(self, shape):
        # 测试方法：flip_all
        np.flip(self.xarg, axis=None)
        # 使用 np.flip 函数反转数组的所有元素

    def time_flip_one(self, shape):
        # 测试方法：flip_one
        np.flip(self.xarg, axis=1)
        # 使用 np.flip 函数反转数组的第二个维度（axis=1）

    def time_flip_neg(self, shape):
        # 测试方法：flip_neg
        np.flip(self.xarg, axis=-1)
        # 使用 np.flip 函数反转数组的最后一个维度（axis=-1）
    # 定义一个方法 `time_moveaxis`，用于执行 `moveaxis` 操作
    def time_moveaxis(self, shape):
        # 使用 NumPy 的 `moveaxis` 函数，将数组 `self.xarg` 的轴从 [0, 1] 调整为 [-1, -2]，但未对 `self.xarg` 进行任何操作
        np.moveaxis(self.xarg, [0, 1], [-1, -2])
    
    # 定义一个方法 `time_roll`，用于执行 `roll` 操作
    def time_roll(self, shape):
        # 使用 NumPy 的 `roll` 函数，将数组 `self.xarg` 向左滚动 3 个位置，但未对 `self.xarg` 进行任何操作
        np.roll(self.xarg, 3)
    
    # 定义一个方法 `time_reshape`，用于执行 `reshape` 操作
    def time_reshape(self, shape):
        # 使用 NumPy 的 `reshape` 函数，将数组 `self.xarg` 重新塑形为 `self.reshaped` 的形状，但未对 `self.xarg` 进行任何操作
        np.reshape(self.xarg, self.reshaped)
```