# `.\numpy\benchmarks\benchmarks\bench_shape_base.py`

```py
# 导入自定义的Benchmark类，该类用于性能基准测试
from .common import Benchmark

# 导入NumPy库，并将其命名为np，用于处理数组和矩阵运算
import numpy as np


# 定义Block类，继承Benchmark类，用于进行块状数组操作的性能测试
class Block(Benchmark):
    # 参数列表，指定不同的块大小进行性能测试
    params = [1, 10, 100]
    # 参数名称，描述params列表中各个参数的含义
    param_names = ['size']

    # 初始化方法，在每次性能测试前调用，设置测试所需的各种数组和矩阵
    def setup(self, n):
        # 创建一个2*n × 2*n的全1二维数组a_2d
        self.a_2d = np.ones((2 * n, 2 * n))
        # 创建一个长度为2*n的全1一维数组b_1d
        self.b_1d = np.ones(2 * n)
        # 创建一个2*n × 2*n的数组b_2d，每个元素是对应位置a_2d元素的两倍
        self.b_2d = 2 * self.a_2d

        # 创建一个长度为3*n的全1一维数组a
        self.a = np.ones(3 * n)
        # 创建一个长度为3*n的全1一维数组b
        self.b = np.ones(3 * n)

        # 创建一个1*n × 3*n的全1二维数组one_2d
        self.one_2d = np.ones((1 * n, 3 * n))
        # 创建一个1*n × 3*n的全1二维数组two_2d
        self.two_2d = np.ones((1 * n, 3 * n))
        # 创建一个1*n × 6*n的全1二维数组three_2d
        self.three_2d = np.ones((1 * n, 6 * n))
        # 创建一个长度为6*n的全1一维数组four_1d
        self.four_1d = np.ones(6 * n)
        # 创建一个长度为1*n的全1一维数组five_0d
        self.five_0d = np.ones(1 * n)
        # 创建一个长度为5*n的全1一维数组six_1d
        self.six_1d = np.ones(5 * n)
        
        # 避免使用np.zeros懒惰分配可能在基准测试期间导致的页面错误
        # 创建一个2*n × 6*n的全0二维数组zero_2d
        self.zero_2d = np.full((2 * n, 6 * n), 0)

        # 创建一个长度为3*n的全1一维数组one
        self.one = np.ones(3 * n)
        # 创建一个3 × 3*n的全2二维数组two
        self.two = 2 * np.ones((3, 3 * n))
        # 创建一个长度为3*n的全3一维数组three
        self.three = 3 * np.ones(3 * n)
        # 创建一个长度为3*n的全4一维数组four
        self.four = 4 * np.ones(3 * n)
        # 创建一个长度为1*n的全5一维数组five
        self.five = 5 * np.ones(1 * n)
        # 创建一个长度为5*n的全6一维数组six
        self.six = 6 * np.ones(5 * n)
        
        # 避免使用np.zeros懒惰分配可能在基准测试期间导致的页面错误
        # 创建一个2*n × 6*n的全0二维数组zero
        self.zero = np.full((2 * n, 6 * n), 0)

    # 定义简单的行主要方式块操作的性能测试方法
    def time_block_simple_row_wise(self, n):
        np.block([self.a_2d, self.b_2d])

    # 定义简单的列主要方式块操作的性能测试方法
    def time_block_simple_column_wise(self, n):
        np.block([[self.a_2d], [self.b_2d]])

    # 定义复杂块操作的性能测试方法
    def time_block_complicated(self, n):
        np.block([[self.one_2d, self.two_2d],
                  [self.three_2d],
                  [self.four_1d],
                  [self.five_0d, self.six_1d],
                  [self.zero_2d]])

    # 定义嵌套块操作的性能测试方法
    def time_nested(self, n):
        np.block([
            [
                np.block([
                   [self.one],
                   [self.three],
                   [self.four]
                ]),
                self.two
            ],
            [self.five, self.six],
            [self.zero]
        ])

    # 定义不使用列表的块操作性能测试方法
    def time_no_lists(self, n):
        np.block(1)  # 测试np.block函数对标量1的处理
        np.block(np.eye(3 * n))  # 测试np.block函数对3*n维单位矩阵的处理


# 定义Block2D类，继承Benchmark类，用于二维块操作的性能测试
class Block2D(Benchmark):
    # 参数列表，指定不同的形状、数据类型和块数进行性能测试
    params = [[(16, 16), (64, 64), (256, 256), (1024, 1024)],
              ['uint8', 'uint16', 'uint32', 'uint64'],
              [(2, 2), (4, 4)]]
    # 参数名称，描述params列表中各个参数的含义
    param_names = ['shape', 'dtype', 'n_chunks']

    # 初始化方法，在每次性能测试前调用，设置测试所需的块列表
    def setup(self, shape, dtype, n_chunks):
        # 创建一个二维块列表block_list，用于性能测试
        self.block_list = [
             [np.full(shape=[s//n_chunk for s, n_chunk in zip(shape, n_chunks)],
                     fill_value=1, dtype=dtype) for _ in range(n_chunks[1])]
            for _ in range(n_chunks[0])
        ]

    # 定义二维块操作的性能测试方法
    def time_block2d(self, shape, dtype, n_chunks):
        np.block(self.block_list)


# 定义Block3D类，继承Benchmark类，用于三维块操作的性能测试
class Block3D(Benchmark):
    """This benchmark concatenates an array of size ``(5n)^3``"""
    # 参数列表，指定不同的大小和操作模式进行性能测试
    params = [[1, 10, 100],
              ['block', 'copy']]
    param_names = ['n', 'mode']

    def setup(self, n, mode):
        # Slow setup method: hence separated from the others above
        # 初始化一个 3D 的数组，每个维度长度为 2*n，元素值为 1
        self.a000 = np.ones((2 * n, 2 * n, 2 * n), int) * 1

        # 初始化一个 3D 的数组，每个维度长度为 3*n，除了元素值为 2 外其余元素为 1
        self.a100 = np.ones((3 * n, 2 * n, 2 * n), int) * 2
        # 初始化一个 3D 的数组，每个维度长度为 2*n，除了元素值为 3 外其余元素为 1
        self.a010 = np.ones((2 * n, 3 * n, 2 * n), int) * 3
        # 初始化一个 3D 的数组，每个维度长度为 2*n，除了元素值为 4 外其余元素为 1
        self.a001 = np.ones((2 * n, 2 * n, 3 * n), int) * 4

        # 初始化一个 3D 的数组，每个维度长度为 2*n 或 3*n，部分元素值为 5，其余为 1
        self.a011 = np.ones((2 * n, 3 * n, 3 * n), int) * 5
        self.a101 = np.ones((3 * n, 2 * n, 3 * n), int) * 6
        self.a110 = np.ones((3 * n, 3 * n, 2 * n), int) * 7

        # 初始化一个 3D 的数组，每个维度长度为 3*n，元素值为 8
        self.a111 = np.ones((3 * n, 3 * n, 3 * n), int) * 8

        # 构建一个包含两个 2x2x2 维度的块状数组
        self.block = [
            [
                [self.a000, self.a001],
                [self.a010, self.a011],
            ],
            [
                [self.a100, self.a101],
                [self.a110, self.a111],
            ]
        ]
        # 将 self.block 中的所有数组放入一个列表中
        self.arr_list = [a
                         for two_d in self.block
                         for one_d in two_d
                         for a in one_d]

    def time_3d(self, n, mode):
        # 根据 mode 参数选择执行不同的操作
        if mode == 'block':
            # 如果 mode 是 'block'，调用 numpy 的块状数组函数 np.block
            np.block(self.block)
        else:  # mode == 'copy'
            # 如果 mode 是 'copy'，对 self.arr_list 中的所有数组执行复制操作
            [arr.copy() for arr in self.arr_list]

    # 为了向后兼容，保留旧的基准测试名称
    time_3d.benchmark_name = "bench_shape_base.Block.time_3d"
# 继承自Benchmark类，用于评估Kronecker乘积的性能
class Kron(Benchmark):
    """Benchmarks for Kronecker product of two arrays"""

    # 初始化方法，设置大数组、大矩阵和标量
    def setup(self):
        self.large_arr = np.random.random((10,) * 4)  # 创建一个形状为(10, 10, 10, 10)的随机数组
        self.large_mat = np.asmatrix(np.random.random((100, 100)))  # 创建一个形状为(100, 100)的随机矩阵
        self.scalar = 7  # 设置一个标量值为7

    # 评估大数组的Kronecker乘积的性能
    def time_arr_kron(self):
        np.kron(self.large_arr, self.large_arr)

    # 评估大数组与标量的Kronecker乘积的性能
    def time_scalar_kron(self):
        np.kron(self.large_arr, self.scalar)

    # 评估大矩阵的Kronecker乘积的性能
    def time_mat_kron(self):
        np.kron(self.large_mat, self.large_mat)

# 继承自Benchmark类，用于评估np.atleast_1d函数的性能
class AtLeast1D(Benchmark):
    """Benchmarks for np.atleast_1d"""

    # 初始化方法，设置数组和零维浮点数
    def setup(self):
        self.x = np.array([1, 2, 3])  # 创建一个形状为(3,)的数组
        self.zero_d = np.float64(1.)  # 创建一个零维浮点数

    # 评估np.atleast_1d函数将多个数组至少转换为一维数组的性能
    def time_atleast_1d(self):
        np.atleast_1d(self.x, self.x, self.x)

    # 评估np.atleast_1d函数将多个零维对象至少转换为一维数组的性能
    def time_atleast_1d_reshape(self):
        np.atleast_1d(self.zero_d, self.zero_d, self.zero_d)

    # 评估np.atleast_1d函数将单个数组至少转换为一维数组的性能
    def time_atleast_1d_single_argument(self):
        np.atleast_1d(self.x)
```