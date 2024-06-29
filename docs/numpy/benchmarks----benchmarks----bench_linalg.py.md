# `.\numpy\benchmarks\benchmarks\bench_linalg.py`

```
# 导入必要的模块和函数
from .common import Benchmark, get_squares_, get_indexes_rand, TYPES1
import numpy as np

# 定义一个继承自Benchmark类的新类Eindot，用于测试矩阵运算的性能
class Eindot(Benchmark):
    
    # 设置测试环境，在每个测试函数执行前调用
    def setup(self):
        # 创建并初始化各种大小的NumPy数组
        self.a = np.arange(60000.0).reshape(150, 400)
        self.ac = self.a.copy()
        self.at = self.a.T
        self.atc = self.a.T.copy()
        self.b = np.arange(240000.0).reshape(400, 600)
        self.c = np.arange(600)
        self.d = np.arange(400)

        self.a3 = np.arange(480000.).reshape(60, 80, 100)
        self.b3 = np.arange(192000.).reshape(80, 60, 40)

    # 定义矩阵乘法测试函数，计算 self.a 和 self.b 的乘积
    def time_dot_a_b(self):
        np.dot(self.a, self.b)

    # 定义多层次的矩阵乘法测试函数，计算 np.dot(self.b, self.c) 的结果再与 self.d 点乘
    def time_dot_d_dot_b_c(self):
        np.dot(self.d, np.dot(self.b, self.c))

    # 定义矩阵转置后的乘法测试函数，计算 self.a 和 self.at 的乘积
    def time_dot_trans_a_at(self):
        np.dot(self.a, self.at)

    # 定义矩阵和其转置副本的乘法测试函数，计算 self.a 和 self.atc 的乘积
    def time_dot_trans_a_atc(self):
        np.dot(self.a, self.atc)

    # 定义转置矩阵和原始矩阵的乘法测试函数，计算 self.at 和 self.a 的乘积
    def time_dot_trans_at_a(self):
        np.dot(self.at, self.a)

    # 定义转置副本矩阵和原始矩阵的乘法测试函数，计算 self.atc 和 self.a 的乘积
    def time_dot_trans_atc_a(self):
        np.dot(self.atc, self.a)

    # 定义einsum函数的测试函数，计算 'i,ij,j' 的乘积，其中 self.d 与 self.b 和 self.c 的组合
    def time_einsum_i_ij_j(self):
        np.einsum('i,ij,j', self.d, self.b, self.c)

    # 定义einsum函数的测试函数，计算 'ij,jk' 的乘积，其中 self.a 和 self.b 的乘积
    def time_einsum_ij_jk_a_b(self):
        np.einsum('ij,jk', self.a, self.b)

    # 定义einsum函数的测试函数，计算 'ijk,jil->kl' 的乘积，其中 self.a3 和 self.b3 的组合
    def time_einsum_ijk_jil_kl(self):
        np.einsum('ijk,jil->kl', self.a3, self.b3)

    # 定义内积计算函数的测试函数，计算 self.a 和 self.a 的内积
    def time_inner_trans_a_a(self):
        np.inner(self.a, self.a)

    # 定义内积计算函数的测试函数，计算 self.a 和 self.ac 的内积
    def time_inner_trans_a_ac(self):
        np.inner(self.a, self.ac)

    # 定义矩阵乘法函数的测试函数，计算 self.a 和 self.b 的乘积
    def time_matmul_a_b(self):
        np.matmul(self.a, self.b)

    # 定义多层次的矩阵乘法函数的测试函数，计算 np.matmul(self.b, self.c) 的结果再与 self.d 点乘
    def time_matmul_d_matmul_b_c(self):
        np.matmul(self.d, np.matmul(self.b, self.c))

    # 定义矩阵转置后的乘法函数的测试函数，计算 self.a 和 self.at 的乘积
    def time_matmul_trans_a_at(self):
        np.matmul(self.a, self.at)

    # 定义矩阵和其转置副本的乘法函数的测试函数，计算 self.a 和 self.atc 的乘积
    def time_matmul_trans_a_atc(self):
        np.matmul(self.a, self.atc)

    # 定义转置矩阵和原始矩阵的乘法函数的测试函数，计算 self.at 和 self.a 的乘积
    def time_matmul_trans_at_a(self):
        np.matmul(self.at, self.a)

    # 定义转置副本矩阵和原始矩阵的乘法函数的测试函数，计算 self.atc 和 self.a 的乘积
    def time_matmul_trans_atc_a(self):
        np.matmul(self.atc, self.a)

    # 定义tensordot函数的测试函数，计算 axes=([1, 0], [0, 1]) 的 self.a3 和 self.b3 的乘积
    def time_tensordot_a_b_axes_1_0_0_1(self):
        np.tensordot(self.a3, self.b3, axes=([1, 0], [0, 1]))

# 定义一个继承自Benchmark类的新类Linalg，用于测试线性代数运算的性能
class Linalg(Benchmark):
    # 使用类型TYPES1的集合减去'float16'类型后作为参数
    params = sorted(list(set(TYPES1) - set(['float16'])))
    param_names = ['dtype']

    # 设置测试环境，在每个测试函数执行前调用
    def setup(self, typename):
        # 忽略所有的NumPy错误
        np.seterr(all='ignore')
        # 获取指定类型的方阵并赋值给 self.a
        self.a = get_squares_()[typename]

    # 定义奇异值分解测试函数，计算 self.a 的奇异值分解
    def time_svd(self, typename):
        np.linalg.svd(self.a)

    # 定义伪逆矩阵测试函数，计算 self.a 的伪逆矩阵
    def time_pinv(self, typename):
        np.linalg.pinv(self.a)

    # 定义行列式计算测试函数，计算 self.a 的行列式
    def time_det(self, typename):
        np.linalg.det(self.a)

# 定义一个继承自Benchmark类的新类LinalgNorm，用于测试线性代数中的范数计算性能
class LinalgNorm(Benchmark):
    # 使用类型TYPES1作为参数
    params = TYPES1
    param_names = ['dtype']

    # 设置测试环境，在每个测试函数执行前调用
    def setup(self, typename):
        # 获取指定类型的方阵并赋值给 self.a
        self.a = get_squares_()[typename]

    # 定义范数计算测试函数，计算 self.a 的范数
    def time_norm(self, typename):
        np.linalg.norm(self.a)

# 定义一个继承自Benchmark类的新类LinalgSmallArrays，用于测试小数组的线性代数运算性能
class LinalgSmallArrays(Benchmark):
    """ Test overhead of linalg methods for small arrays """

    # 设置测试环境，在每个测试函数执行前调用
    def setup(self):
        # 创建大小为5的一维数组和大小为5x5的二维数组
        self.array_5 = np.arange(5.)
        self.array_5_5 = np.reshape(np.arange(25.), (5, 5))

    # 定义小数组的范数计算测试函数，计算 self.array_5 的范数
    def time_norm_small_array(self):
        np.linalg.norm(self.array_5)

    # 定义小数组的行列式计算测试函数，计算 self.array_5_5 的行列式
    def time_det_small_array(self):
        np.linalg.det(self.array_5_5)
    # 定义一个方法 `setup`，用于初始化对象的 `a` 和 `b` 属性
    def setup(self):
        # 使用 `get_squares_()` 函数获取返回结果中 'float64' 对应的值，将其赋给对象的属性 `a`
        self.a = get_squares_()['float64']
        # 使用 `get_indexes_rand()` 函数获取随机索引，并转换为 `np.float64` 类型的数组，取前 100 个元素，赋给对象的属性 `b`

    # 定义一个方法 `time_numpy_linalg_lstsq_a__b_float64`，用于测试 `np.linalg.lstsq` 函数的性能
    def time_numpy_linalg_lstsq_a__b_float64(self):
        # 调用 `np.linalg.lstsq` 函数，传入对象属性 `a` 和 `b` 作为参数，并设置 `rcond=-1`
        np.linalg.lstsq(self.a, self.b, rcond=-1)
# 定义一个继承自Benchmark类的Einsum类，用于进行基准测试
class Einsum(Benchmark):
    # 参数名称列表
    param_names = ['dtype']
    # 参数取值列表，包括np.float32和np.float64两种数据类型
    params = [[np.float32, np.float64]]
    
    # 初始化方法，设置不同数据类型下的各种测试数据
    def setup(self, dtype):
        # 生成长度为600的一维数组，数据类型为dtype
        self.one_dim_small = np.arange(600, dtype=dtype)
        # 生成长度为3000的一维数组，数据类型为dtype
        self.one_dim = np.arange(3000, dtype=dtype)
        # 生成长度为480000的一维数组，数据类型为dtype
        self.one_dim_big = np.arange(480000, dtype=dtype)
        # 生成形状为(30, 40)的二维数组，数据类型为dtype
        self.two_dim_small = np.arange(1200, dtype=dtype).reshape(30, 40)
        # 生成形状为(400, 600)的二维数组，数据类型为dtype
        self.two_dim = np.arange(240000, dtype=dtype).reshape(400, 600)
        # 生成形状为(10, 100, 10)的三维数组，数据类型为dtype
        self.three_dim_small = np.arange(10000, dtype=dtype).reshape(10, 100, 10)
        # 生成形状为(20, 30, 40)的三维数组，数据类型为dtype
        self.three_dim = np.arange(24000, dtype=dtype).reshape(20, 30, 40)
        
        # 非连续数组
        # 生成步长为2，长度为80的一维数组，数据类型为dtype
        self.non_contiguous_dim1_small = np.arange(1, 80, 2, dtype=dtype)
        # 生成步长为2，长度为4000的一维数组，数据类型为dtype
        self.non_contiguous_dim1 = np.arange(1, 4000, 2, dtype=dtype)
        # 生成步长为2，形状为(30, 40)的二维数组，数据类型为dtype
        self.non_contiguous_dim2 = np.arange(1, 2400, 2, dtype=dtype).reshape(30, 40)
        # 生成步长为2，形状为(20, 30, 40)的三维数组，数据类型为dtype
        self.non_contiguous_dim3 = np.arange(1, 48000, 2, dtype=dtype).reshape(20, 30, 40)

    # 使用np.einsum进行外积计算，触发sum_of_products_contig_stride0_outcontig_two
    def time_einsum_outer(self, dtype):
        np.einsum("i,j", self.one_dim, self.one_dim, optimize=True)

    # 使用np.einsum进行矩阵乘法计算，触发sum_of_products_contig_two
    def time_einsum_multiply(self, dtype):
        np.einsum("..., ...", self.two_dim_small, self.three_dim, optimize=True)

    # 使用np.einsum进行求和和乘法计算，触发sum_of_products_contig_stride0_outstride0_two
    def time_einsum_sum_mul(self, dtype):
        np.einsum(",i...->", 300, self.three_dim_small, optimize=True)

    # 使用np.einsum进行求和和乘法计算，触发sum_of_products_stride0_contig_outstride0_two
    def time_einsum_sum_mul2(self, dtype):
        np.einsum("i...,->", self.three_dim_small, 300, optimize=True)

    # 使用np.einsum进行标量乘法计算，触发sum_of_products_stride0_contig_outcontig_two
    def time_einsum_mul(self, dtype):
        np.einsum("i,->i", self.one_dim_big, 300, optimize=True)

    # 使用np.einsum进行矩阵乘法计算，触发contig_contig_outstride0_two
    def time_einsum_contig_contig(self, dtype):
        np.einsum("ji,i->", self.two_dim, self.one_dim_small, optimize=True)

    # 使用np.einsum进行求和计算，触发sum_of_products_contig_outstride0_one
    def time_einsum_contig_outstride0(self, dtype):
        np.einsum("i->", self.one_dim_big, optimize=True)

    # 使用np.einsum进行外积计算，处理非连续数组
    def time_einsum_noncon_outer(self, dtype):
        np.einsum("i,j", self.non_contiguous_dim1, self.non_contiguous_dim1, optimize=True)

    # 使用np.einsum进行矩阵乘法计算，处理非连续数组
    def time_einsum_noncon_multiply(self, dtype):
        np.einsum("..., ...", self.non_contiguous_dim2, self.non_contiguous_dim3, optimize=True)

    # 使用np.einsum进行求和和乘法计算，处理非连续数组
    def time_einsum_noncon_sum_mul(self, dtype):
        np.einsum(",i...->", 300, self.non_contiguous_dim3, optimize=True)

    # 使用np.einsum进行求和和乘法计算，处理非连续数组
    def time_einsum_noncon_sum_mul2(self, dtype):
        np.einsum("i...,->", self.non_contiguous_dim3, 300, optimize=True)
    # 使用 NumPy 的 einsum 函数计算指定的乘积，优化计算以提高效率
    def time_einsum_noncon_mul(self, dtype):
        # 执行 einsum 操作，计算非连续数组 self.non_contiguous_dim1 与标量 300 的逐元素乘积
        np.einsum("i,->i", self.non_contiguous_dim1, 300, optimize=True)

    # contig_contig_outstride0_two: 非连续数组
    def time_einsum_noncon_contig_contig(self, dtype):
        # 执行 einsum 操作，计算非连续数组 self.non_contiguous_dim2 和 self.non_contiguous_dim1_small 的乘积的总和
        np.einsum("ji,i->", self.non_contiguous_dim2, self.non_contiguous_dim1_small, optimize=True)

    # sum_of_products_contig_outstride0_one：非连续数组
    def time_einsum_noncon_contig_outstride0(self, dtype):
        # 执行 einsum 操作，计算非连续数组 self.non_contiguous_dim1 所有元素的总和
        np.einsum("i->", self.non_contiguous_dim1, optimize=True)
class LinAlgTransposeVdot(Benchmark):
    # 继承自 Benchmark 类，用于性能测试
    # 参数设置：矩阵形状和数据类型
    params = [[(16, 16), (32, 32),
               (64, 64)], TYPES1]
    param_names = ['shape', 'npdtypes']

    def setup(self, shape, npdtypes):
        # 初始化第一个随机矩阵，并按照给定形状重新调整其结构
        self.xarg = np.random.uniform(-1, 1, np.dot(*shape)).reshape(shape)
        # 将第一个随机矩阵转换为指定的数据类型
        self.xarg = self.xarg.astype(npdtypes)
        # 初始化第二个随机矩阵，并按照给定形状重新调整其结构
        self.x2arg = np.random.uniform(-1, 1, np.dot(*shape)).reshape(shape)
        # 将第二个随机矩阵转换为指定的数据类型
        self.x2arg = self.x2arg.astype(npdtypes)
        # 如果数据类型以 'complex' 开头，则将第一个和第二个矩阵转换为复数类型
        if npdtypes.startswith('complex'):
            self.xarg += self.xarg.T*1j
            self.x2arg += self.x2arg.T*1j

    def time_transpose(self, shape, npdtypes):
        # 测试 np.transpose() 函数的执行时间，对第一个随机矩阵进行转置操作
        np.transpose(self.xarg)

    def time_vdot(self, shape, npdtypes):
        # 测试 np.vdot() 函数的执行时间，计算第一个和第二个随机矩阵的向量点积
        np.vdot(self.xarg, self.x2arg)
```