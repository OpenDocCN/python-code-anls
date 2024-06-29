# `.\numpy\benchmarks\benchmarks\bench_indexing.py`

```py
# 从common模块导入Benchmark类和其他函数和变量
from .common import (
    Benchmark, get_square_, get_indexes_, get_indexes_rand_, TYPES1)

# 导入os.path模块中的join函数，并重命名为pjoin
from os.path import join as pjoin
# 导入shutil模块，用于文件和目录操作
import shutil
# 导入numpy模块中的memmap、float32和array函数
from numpy import memmap, float32, array
# 导入numpy模块，并重命名为np
import numpy as np
# 导入tempfile模块中的mkdtemp函数，用于创建临时目录
from tempfile import mkdtemp

# 定义Benchmark的子类Indexing，用于测试索引操作的性能
class Indexing(Benchmark):
    # 参数列表，包括TYPES1扩展后的dtype、索引方式、选择器和操作符
    params = [TYPES1 + ["object", "O,i"],
              ["indexes_", "indexes_rand_"],
              ['I', ':,I', 'np.ix_(I, I)'],
              ['', '=1']]
    # 参数名称列表
    param_names = ['dtype', 'indexes', 'sel', 'op']

    # 初始化方法，根据参数设置选择器并动态创建函数
    def setup(self, dtype, indexes, sel, op):
        # 将选择器中的'I'替换为具体的索引方式
        sel = sel.replace('I', indexes)

        # 定义命名空间，包括获取数组函数、numpy模块、固定和随机索引函数
        ns = {'a': get_square_(dtype),
              'np': np,
              'indexes_': get_indexes_(),
              'indexes_rand_': get_indexes_rand_()}

        # 定义函数代码字符串
        code = "def run():\n    a[%s]%s"
        # 将选择器和操作符插入到函数代码字符串中
        code = code % (sel, op)

        # 在命名空间中执行函数代码，生成run函数
        exec(code, ns)
        # 将生成的run函数赋值给实例变量self.func
        self.func = ns['run']

    # 测试运行函数执行时间的方法
    def time_op(self, dtype, indexes, sel, op):
        self.func()

# 定义Benchmark的子类IndexingWith1DArr，用于测试使用1维数组的索引性能
class IndexingWith1DArr(Benchmark):
    # 参数列表，包括不同形状和类型的数组
    params = [
        [(1000,), (1000, 1), (1000, 2), (2, 1000, 1), (1000, 3)],
        TYPES1 + ["O", "i,O"]]
    # 参数名称列表
    param_names = ["shape", "dtype"]

    # 初始化方法，根据参数创建指定形状和类型的数组，并设置索引
    def setup(self, shape, dtype):
        self.arr = np.ones(shape, dtype)
        self.index = np.arange(1000)
        # 如果数组是3维的，则设置索引为第二维的所有元素
        if len(shape) == 3:
            self.index = (slice(None), self.index)

    # 测试按顺序获取数组元素的方法
    def time_getitem_ordered(self, shape, dtype):
        self.arr[self.index]

    # 测试按顺序设置数组元素的方法
    def time_setitem_ordered(self, shape, dtype):
        self.arr[self.index] = 0

# 定义Benchmark的子类ScalarIndexing，用于测试标量索引操作的性能
class ScalarIndexing(Benchmark):
    # 参数列表，包括标量索引的维度
    params = [[0, 1, 2]]
    # 参数名称列表
    param_names = ["ndim"]

    # 初始化方法，创建指定维度的全1数组
    def setup(self, ndim):
        self.array = np.ones((5,) * ndim)

    # 测试索引操作的执行时间
    def time_index(self, ndim):
        # 使用标量索引访问数组元素，并计时执行时间
        arr = self.array
        indx = (1,) * ndim
        for i in range(100):
            arr[indx]

    # 测试赋值操作的执行时间
    def time_assign(self, ndim):
        # 使用标量索引赋值，并计时执行时间
        arr = self.array
        indx = (1,) * ndim
        for i in range(100):
            arr[indx] = 5.

    # 测试赋值操作可能涉及的类型转换的执行时间
    def time_assign_cast(self, ndim):
        # 使用标量索引赋值，可能涉及类型转换，并计时执行时间
        arr = self.array
        indx = (1,) * ndim
        val = np.int16(43)
        for i in range(100):
            arr[indx] = val

# 定义Benchmark的子类IndexingSeparate，用于测试内存映射文件的切片和花式索引操作的性能
class IndexingSeparate(Benchmark):
    # 初始化方法，创建临时目录和内存映射文件，并设置花式索引
    def setup(self):
        self.tmp_dir = mkdtemp()
        self.fp = memmap(pjoin(self.tmp_dir, 'tmp.dat'),
                         dtype=float32, mode='w+', shape=(50, 60))
        self.indexes = array([3, 4, 6, 10, 20])

    # 清理方法，删除内存映射文件和临时目录
    def teardown(self):
        del self.fp
        shutil.rmtree(self.tmp_dir)

    # 测试切片操作的执行时间
    def time_mmap_slicing(self):
        for i in range(1000):
            self.fp[5:10]

    # 测试花式索引操作的执行时间
    def time_mmap_fancy_indexing(self):
        for i in range(1000):
            self.fp[self.indexes]

# 定义Benchmark的子类IndexingStructured0D，用于测试0维结构化索引的性能
class IndexingStructured0D(Benchmark):
    # 设置自定义数据类型，包含一个名为 'a' 的字段，每个字段是一个长度为 256 的 float32 数组
    self.dt = np.dtype([('a', 'f4', 256)])

    # 创建一个空的 ndarray A，数据类型为 self.dt
    self.A = np.zeros((), self.dt)
    # 通过复制创建一个 ndarray B，数据类型与 A 相同
    self.B = self.A.copy()

    # 创建一个长度为 1 的 ndarray，元素类型为 self.dt，并取出第一个元素赋给 a
    self.a = np.zeros(1, self.dt)[0]
    # 通过复制创建一个 ndarray b，数据类型与 a 相同
    self.b = self.a.copy()

# 将 A 数组的 'a' 字段的所有元素复制到 B 数组的 'a' 字段
def time_array_slice(self):
    self.B['a'][:] = self.A['a']

# 将 A 数组的 'a' 字段的所有元素直接赋给 B 数组的 'a' 字段
def time_array_all(self):
    self.B['a'] = self.A['a']

# 将 a 数组的 'a' 字段的所有元素复制到 b 数组的 'a' 字段
def time_scalar_slice(self):
    self.b['a'][:] = self.a['a']

# 将 a 数组的 'a' 字段的所有元素直接赋给 b 数组的 'a' 字段
def time_scalar_all(self):
    self.b['a'] = self.a['a']
# 定义一个继承自Benchmark类的FlatIterIndexing类，用于性能基准测试
class FlatIterIndexing(Benchmark):
    # 设置方法，在每个性能测试之前初始化数据
    def setup(self):
        # 创建一个形状为(200, 50000)的全1数组，并赋值给self.a
        self.a = np.ones((200, 50000))
        # 创建一个长度为200*50000的全True布尔数组，并赋值给self.m_all
        self.m_all = np.repeat(True, 200 * 50000)
        # 复制self.m_all到self.m_half，并将self.m_half中偶数索引位置设置为False
        self.m_half = np.copy(self.m_all)
        self.m_half[::2] = False
        # 创建一个长度为200*50000的全False布尔数组，并赋值给self.m_none
        self.m_none = np.repeat(False, 200 * 50000)

    # 定义性能测试方法，用于测试使用self.m_none进行布尔索引时的性能
    def time_flat_bool_index_none(self):
        # 使用布尔索引self.m_none访问self.a的扁平化视图
        self.a.flat[self.m_none]

    # 定义性能测试方法，用于测试使用self.m_half进行布尔索引时的性能
    def time_flat_bool_index_half(self):
        # 使用布尔索引self.m_half访问self.a的扁平化视图
        self.a.flat[self.m_half]

    # 定义性能测试方法，用于测试使用self.m_all进行布尔索引时的性能
    def time_flat_bool_index_all(self):
        # 使用布尔索引self.m_all访问self.a的扁平化视图
        self.a.flat[self.m_all]
```