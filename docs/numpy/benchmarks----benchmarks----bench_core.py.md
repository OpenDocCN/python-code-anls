# `.\numpy\benchmarks\benchmarks\bench_core.py`

```py
# 从.common模块导入Benchmark类
from .common import Benchmark

# 导入NumPy库并使用np作为别名
import numpy as np

# 定义Core类，继承Benchmark类
class Core(Benchmark):
    
    # 设置初始化方法
    def setup(self):
        # 创建长度为100的整数范围对象并赋值给self.l100
        self.l100 = range(100)
        # 创建长度为50的整数范围对象并赋值给self.l50
        self.l50 = range(50)
        # 创建包含1000个浮点数的列表并赋值给self.float_l1000
        self.float_l1000 = [float(i) for i in range(1000)]
        # 使用NumPy创建包含1000个np.float64类型浮点数的列表并赋值给self.float64_l1000
        self.float64_l1000 = [np.float64(i) for i in range(1000)]
        # 创建包含1000个整数的列表并赋值给self.int_l1000
        self.int_l1000 = list(range(1000))
        # 创建包含两个长度为1000的NumPy数组的列表并赋值给self.l
        self.l = [np.arange(1000), np.arange(1000)]
        # 创建self.l中每个数组的内存视图组成的列表并赋值给self.l_view
        self.l_view = [memoryview(a) for a in self.l]
        # 创建一个全为1的10x10的NumPy数组并赋值给self.l10x10
        self.l10x10 = np.ones((10, 10))
        # 创建np.float64类型的NumPy数据类型对象并赋值给self.float64_dtype
        self.float64_dtype = np.dtype(np.float64)

    # 定义time_array_1方法
    def time_array_1(self):
        # 创建包含单个元素1的NumPy数组
        np.array(1)

    # 定义time_array_empty方法
    def time_array_empty(self):
        # 创建空的NumPy数组
        np.array([])

    # 定义time_array_l1方法
    def time_array_l1(self):
        # 使用self.l100创建包含单个元素1的NumPy数组
        np.array([1])

    # 定义time_array_l100方法
    def time_array_l100(self):
        # 使用self.l100创建NumPy数组
        np.array(self.l100)

    # 定义time_array_float_l1000方法
    def time_array_float_l1000(self):
        # 使用self.float_l1000创建NumPy数组
        np.array(self.float_l1000)

    # 定义time_array_float_l1000_dtype方法
    def time_array_float_l1000_dtype(self):
        # 使用self.float_l1000和self.float64_dtype创建NumPy数组
        np.array(self.float_l1000, dtype=self.float64_dtype)

    # 定义time_array_float64_l1000方法
    def time_array_float64_l1000(self):
        # 使用self.float64_l1000创建NumPy数组
        np.array(self.float64_l1000)

    # 定义time_array_int_l1000方法
    def time_array_int_l1000(self):
        # 使用self.int_l1000创建NumPy数组
        np.array(self.int_l1000)

    # 定义time_array_l方法
    def time_array_l(self):
        # 使用self.l创建NumPy数组
        np.array(self.l)

    # 定义time_array_l_view方法
    def time_array_l_view(self):
        # 使用self.l_view创建NumPy数组
        np.array(self.l_view)

    # 定义time_can_cast方法
    def time_can_cast(self):
        # 检查是否可以将self.l10x10转换为self.float64_dtype类型
        np.can_cast(self.l10x10, self.float64_dtype)

    # 定义time_can_cast_same_kind方法
    def time_can_cast_same_kind(self):
        # 在相同种类转换下检查是否可以将self.l10x10转换为self.float64_dtype类型
        np.can_cast(self.l10x10, self.float64_dtype, casting="same_kind")

    # 定义time_vstack_l方法
    def time_vstack_l(self):
        # 垂直堆叠self.l中的NumPy数组
        np.vstack(self.l)

    # 定义time_hstack_l方法
    def time_hstack_l(self):
        # 水平堆叠self.l中的NumPy数组
        np.hstack(self.l)

    # 定义time_dstack_l方法
    def time_dstack_l(self):
        # 深度堆叠self.l中的NumPy数组
        np.dstack(self.l)

    # 定义time_arange_100方法
    def time_arange_100(self):
        # 创建包含100个元素的NumPy数组
        np.arange(100)

    # 定义time_zeros_100方法
    def time_zeros_100(self):
        # 创建包含100个0的NumPy数组
        np.zeros(100)

    # 定义time_ones_100方法
    def time_ones_100(self):
        # 创建包含100个1的NumPy数组
        np.ones(100)

    # 定义time_empty_100方法
    def time_empty_100(self):
        # 创建长度为100的空NumPy数组
        np.empty(100)

    # 定义time_empty_like方法
    def time_empty_like(self):
        # 创建与self.l10x10具有相同形状的空NumPy数组
        np.empty_like(self.l10x10)

    # 定义time_eye_100方法
    def time_eye_100(self):
        # 创建100x100的单位矩阵（对角线元素为1，其余为0）
        np.eye(100)

    # 定义time_identity_100方法
    def time_identity_100(self):
        # 创建100x100的单位矩阵
        np.identity(100)

    # 定义time_eye_3000方法
    def time_eye_3000(self):
        # 创建3000x3000的单位矩阵
        np.eye(3000)

    # 定义time_identity_3000方法
    def time_identity_3000(self):
        # 创建3000x3000的单位矩阵
        np.identity(3000)

    # 定义time_diag_l100方法
    def time_diag_l100(self):
        # 创建包含self.l100对角线元素的对角矩阵
        np.diag(self.l100)

    # 定义time_diagflat_l100方法
    def time_diagflat_l100(self):
        # 创建包含self.l100扁平化对角元素的数组
        np.diagflat(self.l100)

    # 定义time_diagflat_l50_l50方法
    def time_diagflat_l50_l50(self):
        # 创建包含两个self.l50的对角矩阵的扁平化数组
        np.diagflat([self.l50, self.l50])

    # 定义time_triu_l10x10方法
    def time_triu_l10x10(self):
        # 返回self.l10x10的上三角部分
        np.triu(self.l10x10)

    # 定义time_tril_l10x10方法
    def time_tril_l10x10(self):
        # 返回self.l10x10的下三角部分
        np.tril(self.l10x10)

    # 定义time_triu_indices_500方法
    def time_triu_indices_500(self):
        # 返回一个包含500个元素的上三角形状的索引数组
        np.triu_indices(500)

    # 定义time_tril_indices_500方法
    def time_tril_indices_500(self):
        # 返回一个包含500个元素的下三角形状的索引数组
        np.tril_indices(500)


# 定义Temporaries类，继承Benchmark类
class Temporaries(Benchmark):
    
    # 设置初始化方法
    def setup(self):
        # 创建包含50000个1的NumPy数组并赋值给self.amid
        self.amid = np.ones(50000)
        # 创建包含50000个1的NumPy数组并赋值给self.bmid
        self.bmid = np.ones(50000)
        # 创建包含1000000个1的NumPy数组并赋值给self.alarge
        self.alarge = np.ones(1000000)
        # 创建包含1000000个1的NumPy数组并赋值给self.blarge
        self.blarge = np.ones(1000000)

    # 定义time_mid方法
    def time_mid(self):
        # 计算self
    # 定义一个参数列表，包含三个子列表，每个子列表代表一组参数
    params = [[50, 1000, int(1e5)],
              [10, 100, 1000, int(1e4)],
              ['valid', 'same', 'full']]
    
    # 定义一个参数名称列表，对应于参数列表中的每个参数
    param_names = ['size1', 'size2', 'mode']
    
    # 定义一个类，包含三个方法用于性能测试
    class ClassName:
        # 设置方法，用于初始化数据
        def setup(self, size1, size2, mode):
            # 创建一个等间隔的数组，范围在0到1之间，元素个数为size1
            self.x1 = np.linspace(0, 1, num=size1)
            # 创建一个余弦函数的数组，元素个数为size2
            self.x2 = np.cos(np.linspace(0, 2*np.pi, num=size2))
    
        # 性能测试方法，用于测试np.correlate函数的运行时间
        def time_correlate(self, size1, size2, mode):
            # 调用np.correlate函数，计算x1和x2的相关性，使用指定的mode参数
            np.correlate(self.x1, self.x2, mode=mode)
    
        # 性能测试方法，用于测试np.convolve函数的运行时间
        def time_convolve(self, size1, size2, mode):
            # 调用np.convolve函数，计算x1和x2的卷积，使用指定的mode参数
            np.convolve(self.x1, self.x2, mode=mode)
class CountNonzero(Benchmark):
    # 参数名列表，包括numaxes（轴数）、size（大小）、dtype（数据类型）
    param_names = ['numaxes', 'size', 'dtype']
    # 参数的取值范围：numaxes为1, 2, 3；size为100, 10000, 1000000；dtype为bool, np.int8, np.int16, np.int32, np.int64, str, object
    params = [
        [1, 2, 3],
        [100, 10000, 1000000],
        [bool, np.int8, np.int16, np.int32, np.int64, str, object]
    ]

    # 设置函数，初始化测试所需的数据
    def setup(self, numaxes, size, dtype):
        # 创建一个数组，形状为numaxes * size，并填充0到(numaxes * size - 1)的值
        self.x = np.arange(numaxes * size).reshape(numaxes, size)
        # 将数组元素对3取模后转换为指定的数据类型dtype
        self.x = (self.x % 3).astype(dtype)

    # 测试函数：计算数组self.x中非零元素的数量
    def time_count_nonzero(self, numaxes, size, dtype):
        np.count_nonzero(self.x)

    # 测试函数：计算数组self.x沿最后一个轴（self.x.ndim - 1）中非零元素的数量
    def time_count_nonzero_axis(self, numaxes, size, dtype):
        np.count_nonzero(self.x, axis=self.x.ndim - 1)

    # 测试函数：如果数组self.x的维度大于等于2，计算数组沿倒数第二个和最后一个轴中非零元素的数量
    def time_count_nonzero_multi_axis(self, numaxes, size, dtype):
        if self.x.ndim >= 2:
            np.count_nonzero(self.x, axis=(
                self.x.ndim - 1, self.x.ndim - 2))


class PackBits(Benchmark):
    # 参数名列表，只有dtype（数据类型）
    param_names = ['dtype']
    # 参数的取值范围：dtype为bool或者np.uintp
    params = [[bool, np.uintp]]
    
    # 设置函数，初始化测试所需的数据
    def setup(self, dtype):
        # 创建一个长度为10000的全为1的数组，数据类型为dtype
        self.d = np.ones(10000, dtype=dtype)
        # 创建一个形状为(200, 1000)的全为1的数组，数据类型为dtype
        self.d2 = np.ones((200, 1000), dtype=dtype)

    # 测试函数：对数组self.d进行packbits压缩
    def time_packbits(self, dtype):
        np.packbits(self.d)

    # 测试函数：对数组self.d进行packbits压缩，使用小端字节顺序
    def time_packbits_little(self, dtype):
        np.packbits(self.d, bitorder="little")

    # 测试函数：对数组self.d2按axis=0进行packbits压缩
    def time_packbits_axis0(self, dtype):
        np.packbits(self.d2, axis=0)

    # 测试函数：对数组self.d2按axis=1进行packbits压缩
    def time_packbits_axis1(self, dtype):
        np.packbits(self.d2, axis=1)


class UnpackBits(Benchmark):
    # 设置函数，初始化测试所需的数据
    def setup(self):
        # 创建一个长度为10000的全为1的数组，数据类型为uint8
        self.d = np.ones(10000, dtype=np.uint8)
        # 创建一个形状为(200, 1000)的全为1的数组，数据类型为uint8
        self.d2 = np.ones((200, 1000), dtype=np.uint8)

    # 测试函数：对数组self.d进行unpackbits解压
    def time_unpackbits(self):
        np.unpackbits(self.d)

    # 测试函数：对数组self.d进行unpackbits解压，使用小端字节顺序
    def time_unpackbits_little(self):
        np.unpackbits(self.d, bitorder="little")

    # 测试函数：对数组self.d2按axis=0进行unpackbits解压
    def time_unpackbits_axis0(self):
        np.unpackbits(self.d2, axis=0)

    # 测试函数：对数组self.d2按axis=1进行unpackbits解压
    def time_unpackbits_axis1(self):
        np.unpackbits(self.d2, axis=1)

    # 测试函数：对数组self.d2按axis=1进行unpackbits解压，使用小端字节顺序
    def time_unpackbits_axis1_little(self):
        np.unpackbits(self.d2, bitorder="little", axis=1)


class Indices(Benchmark):
    # 测试函数：生成一个形状为(1000, 500)的索引数组
    def time_indices(self):
        np.indices((1000, 500))


class StatsMethods(Benchmark):
    # 参数名列表，包括dtype（数据类型）和size（大小）
    param_names = ['dtype', 'size']
    # 参数的取值范围：dtype为'int64', 'uint64', 'float32', 'float64', 'complex64', 'bool_'；size为100或10000
    params = [['int64', 'uint64', 'float32', 'float64',
               'complex64', 'bool_'],
              [100, 10000]]

    # 设置函数，初始化测试所需的数据
    def setup(self, dtype, size):
        # 创建一个长度为size的全为1的数组，数据类型为dtype
        self.data = np.ones(size, dtype=dtype)
        # 如果数据类型dtype以'complex'开头，则创建一个随机复数数组
        if dtype.startswith('complex'):
            self.data = np.random.randn(size) + 1j * np.random.randn(size)

    # 测试函数：计算数组self.data的最小值
    def time_min(self, dtype, size):
        self.data.min()

    # 测试函数：计算数组self.data的最大值
    def time_max(self, dtype, size):
        self.data.max()

    # 测试函数：计算数组self.data的均值
    def time_mean(self, dtype, size):
        self.data.mean()

    # 测试函数：计算数组self.data的标准差
    def time_std(self, dtype, size):
        self.data.std()

    # 测试函数：计算数组self.data的乘积
    def time_prod(self, dtype, size):
        self.data.prod()

    # 测试函数：计算数组self.data的方差
    def time_var(self, dtype, size):
        self.data.var()

    # 测试函数：计算数组self.data的总和
    def time_sum(self, dtype, size):
        self.data.sum()


class NumPyChar(Benchmark):
    # 这里省略了具体的实现，需要根据实际情况添加注释
    # 设置函数，初始化类的实例变量
    def setup(self):
        # 创建包含两个长字符串的 NumPy 数组 A
        self.A = np.array([100*'x', 100*'y'])
        # 创建包含 1000 个 'aa' 字符串的 NumPy 数组 B
        self.B = np.array(1000 * ['aa'])

        # 创建包含三个字符串的 NumPy 数组 C，每个字符串都很长
        self.C = np.array([100*'x' + 'z', 100*'y' + 'z' + 'y', 100*'x'])
        # 创建包含 2000 个字符串的 NumPy 数组 D，一半是 'ab'，一半是 'ac'
        self.D = np.array(1000 * ['ab'] + 1000 * ['ac'])

    # 测试 isalpha 函数对于 A 数组的性能
    def time_isalpha_small_list_big_string(self):
        np.char.isalpha(self.A)

    # 测试 isalpha 函数对于 B 数组的性能
    def time_isalpha_big_list_small_string(self):
        np.char.isalpha(self.B)

    # 测试 add 函数对于 A 数组的性能
    def time_add_small_list_big_string(self):
        np.char.add(self.A, self.A)

    # 测试 add 函数对于 B 数组的性能
    def time_add_big_list_small_string(self):
        np.char.add(self.B, self.B)

    # 测试 find 函数对于 C 数组的性能
    def time_find_small_list_big_string(self):
        np.char.find(self.C, 'z')

    # 测试 find 函数对于 D 数组的性能
    def time_find_big_list_small_string(self):
        np.char.find(self.D, 'b')

    # 测试 startswith 函数对于 A 数组的性能
    def time_startswith_small_list_big_string(self):
        np.char.startswith(self.A, 'x')

    # 测试 startswith 函数对于 B 数组的性能
    def time_startswith_big_list_small_string(self):
        np.char.startswith(self.B, 'a')
```