# `D:\src\scipysrc\scipy\scipy\io\tests\test_mmio.py`

```
# 导入必要的库和模块
from tempfile import mkdtemp  # 导入创建临时目录的函数
import os  # 导入操作系统相关功能的模块
import io  # 导入用于处理 IO 操作的模块
import shutil  # 导入文件和目录操作相关的模块
import textwrap  # 导入用于格式化文本的模块

import numpy as np  # 导入 NumPy 库并使用别名 np
from numpy import array, transpose, pi  # 从 NumPy 中导入特定的函数和常数
from numpy.testing import (assert_equal, assert_allclose,  # 导入 NumPy 测试功能
                           assert_array_equal, assert_array_almost_equal)
import pytest  # 导入 pytest 测试框架
from pytest import raises as assert_raises  # 导入 pytest 中的 raises 函数

import scipy.sparse  # 导入 SciPy 中的稀疏矩阵功能
import scipy.io._mmio  # 导入 SciPy 中的矩阵市场格式 IO 模块
import scipy.io._fast_matrix_market as fmm  # 导入快速矩阵市场格式 IO 模块

# 定义参数化测试的参数
parametrize_args = [('integer', 'int'), ('unsigned-integer', 'uint')]

# 使用 pytest 的 fixture 机制，在模块级别运行测试，同时传入两种实现（_mmio 和 _fast_matrix_market）
@pytest.fixture(scope='module', params=(scipy.io._mmio, fmm), autouse=True)
def implementations(request):
    global mminfo  # 全局变量：用于获取矩阵市场文件信息的函数
    global mmread  # 全局变量：用于从矩阵市场文件读取数据的函数
    global mmwrite  # 全局变量：用于向矩阵市场文件写入数据的函数
    mminfo = request.param.mminfo  # 获取当前参数化的实现的 mminfo 函数
    mmread = request.param.mmread  # 获取当前参数化的实现的 mmread 函数
    mmwrite = request.param.mmwrite  # 获取当前参数化的实现的 mmwrite 函数


class TestMMIOArray:
    def setup_method(self):
        # 创建临时目录用于测试
        self.tmpdir = mkdtemp()
        # 在临时目录中创建测试文件的路径
        self.fn = os.path.join(self.tmpdir, 'testfile.mtx')

    def teardown_method(self):
        # 清理临时目录及其内容
        shutil.rmtree(self.tmpdir)

    def check(self, a, info):
        # 将数组 a 写入到测试文件中
        mmwrite(self.fn, a)
        # 检查写入后的文件信息是否符合预期
        assert_equal(mminfo(self.fn), info)
        # 从文件中读取数据并检查与原始数组 a 是否几乎相等
        b = mmread(self.fn)
        assert_array_almost_equal(a, b)

    def check_exact(self, a, info):
        # 将数组 a 精确地写入到测试文件中
        mmwrite(self.fn, a)
        # 检查写入后的文件信息是否符合预期
        assert_equal(mminfo(self.fn), info)
        # 从文件中读取数据并检查与原始数组 a 是否完全相等
        b = mmread(self.fn)
        assert_equal(a, b)

    @pytest.mark.parametrize('typeval, dtype', parametrize_args)
    def test_simple_integer(self, typeval, dtype):
        # 测试简单整数数组的写入和读取，检查文件信息和数组内容是否符合预期
        self.check_exact(array([[1, 2], [3, 4]], dtype=dtype),
                         (2, 2, 4, 'array', typeval, 'general'))

    @pytest.mark.parametrize('typeval, dtype', parametrize_args)
    def test_32bit_integer(self, typeval, dtype):
        # 测试32位整数数组的写入和读取，检查文件信息和数组内容是否符合预期
        a = array([[2**31-1, 2**31-2], [2**31-3, 2**31-4]], dtype=dtype)
        self.check_exact(a, (2, 2, 4, 'array', typeval, 'general'))

    def test_64bit_integer(self):
        # 测试64位有符号整数数组的写入和读取，检查是否引发溢出异常或数组内容是否符合预期
        a = array([[2**31, 2**32], [2**63-2, 2**63-1]], dtype=np.int64)
        if (np.intp(0).itemsize < 8) and mmwrite == scipy.io._mmio.mmwrite:
            assert_raises(OverflowError, mmwrite, self.fn, a)
        else:
            self.check_exact(a, (2, 2, 4, 'array', 'integer', 'general'))

    def test_64bit_unsigned_integer(self):
        # 测试64位无符号整数数组的写入和读取，检查文件信息和数组内容是否符合预期
        a = array([[2**31, 2**32], [2**64-2, 2**64-1]], dtype=np.uint64)
        self.check_exact(a, (2, 2, 4, 'array', 'unsigned-integer', 'general'))

    @pytest.mark.parametrize('typeval, dtype', parametrize_args)
    def test_simple_upper_triangle_integer(self, typeval, dtype):
        # 测试简单上三角整数数组的写入和读取，检查文件信息和数组内容是否符合预期
        self.check_exact(array([[0, 1], [0, 0]], dtype=dtype),
                         (2, 2, 4, 'array', typeval, 'general'))

    @pytest.mark.parametrize('typeval, dtype', parametrize_args)
    def test_simple_lower_triangle_integer(self, typeval, dtype):
        # 测试简单下三角整数数组的写入和读取，检查文件信息和数组内容是否符合预期
        self.check_exact(array([[0, 0], [1, 0]], dtype=dtype),
                         (2, 2, 4, 'array', typeval, 'general'))
    # 使用 pytest 的参数化标记，为测试函数 test_simple_rectangular_integer 注入多组参数进行测试
    @pytest.mark.parametrize('typeval, dtype', parametrize_args)
    def test_simple_rectangular_integer(self, typeval, dtype):
        # 调用 self.check 方法验证输入数组和预期元组的一致性
        self.check_exact(array([[1, 2, 3], [4, 5, 6]], dtype=dtype),
                         (2, 3, 6, 'array', typeval, 'general'))

    # 测试简单的浮点型矩形数组
    def test_simple_rectangular_float(self):
        # 调用 self.check 方法验证输入数组和预期元组的一致性
        self.check([[1, 2], [3.5, 4], [5, 6]],
                   (3, 2, 6, 'array', 'real', 'general'))

    # 测试简单的浮点型数组
    def test_simple_float(self):
        # 调用 self.check 方法验证输入数组和预期元组的一致性
        self.check([[1, 2], [3, 4.0]],
                   (2, 2, 4, 'array', 'real', 'general'))

    # 测试简单的复数型数组
    def test_simple_complex(self):
        # 调用 self.check 方法验证输入数组和预期元组的一致性
        self.check([[1, 2], [3, 4j]],
                   (2, 2, 4, 'array', 'complex', 'general'))

    # 使用 pytest 的参数化标记，为测试函数 test_simple_symmetric_integer 注入多组参数进行测试
    @pytest.mark.parametrize('typeval, dtype', parametrize_args)
    def test_simple_symmetric_integer(self, typeval, dtype):
        # 调用 self.check_exact 方法验证输入数组和预期元组的一致性
        self.check_exact(array([[1, 2], [2, 4]], dtype=dtype),
                         (2, 2, 4, 'array', typeval, 'symmetric'))

    # 测试简单的整型反对称数组
    def test_simple_skew_symmetric_integer(self):
        # 调用 self.check_exact 方法验证输入数组和预期元组的一致性
        self.check_exact([[0, 2], [-2, 0]],
                         (2, 2, 4, 'array', 'integer', 'skew-symmetric'))

    # 测试简单的浮点型反对称数组
    def test_simple_skew_symmetric_float(self):
        # 调用 self.check 方法验证输入数组和预期元组的一致性
        self.check(array([[0, 2], [-2.0, 0.0]], 'f'),
                   (2, 2, 4, 'array', 'real', 'skew-symmetric'))

    # 测试简单的复数埃尔米特数组
    def test_simple_hermitian_complex(self):
        # 调用 self.check 方法验证输入数组和预期元组的一致性
        self.check([[1, 2+3j], [2-3j, 4]],
                   (2, 2, 4, 'array', 'complex', 'hermitian'))

    # 测试随机生成的浮点型对称数组
    def test_random_symmetric_float(self):
        # 随机生成大小为 (20, 20) 的数组
        sz = (20, 20)
        a = np.random.random(sz)
        # 将数组与其转置相加，确保得到对称数组
        a = a + transpose(a)
        # 调用 self.check 方法验证输入数组和预期元组的一致性
        self.check(a, (20, 20, 400, 'array', 'real', 'symmetric'))

    # 测试随机生成的浮点型矩形数组
    def test_random_rectangular_float(self):
        # 随机生成大小为 (20, 15) 的数组
        sz = (20, 15)
        a = np.random.random(sz)
        # 调用 self.check 方法验证输入数组和预期元组的一致性
        self.check(a, (20, 15, 300, 'array', 'real', 'general'))

    # 使用 pytest 的标记，指定在这个测试中失败不会立即停止，最多失败 10 次
    @pytest.mark.fail_slow(10)
    def test_bad_number_of_array_header_fields(self):
        # 定义一个含有错误数量字段的 MatrixMarket 格式的字符串
        s = """\
            %%MatrixMarket matrix array real general
              3  3 999
            1.0
            2.0
            3.0
            4.0
            5.0
            6.0
            7.0
            8.0
            9.0
            """
        # 将字符串去除额外的缩进，并编码为 ASCII 格式
        text = textwrap.dedent(s).encode('ascii')
        # 使用 pytest 的断言来验证 mmread 是否会抛出预期的 ValueError 异常
        with pytest.raises(ValueError, match='not of length 2'):
            scipy.io.mmread(io.BytesIO(text))

    # 测试非反对称整型数组（GitHub issue #13634）
    def test_gh13634_non_skew_symmetric_int(self):
        # 调用 self.check_exact 方法验证输入数组和预期元组的一致性
        self.check_exact(array([[1, 2], [-2, 99]], dtype=np.int32),
                         (2, 2, 4, 'array', 'integer', 'general'))

    # 测试非反对称浮点型数组（GitHub issue #13634）
    def test_gh13634_non_skew_symmetric_float(self):
        # 调用 self.check 方法验证输入数组和预期元组的一致性
        self.check(array([[1, 2], [-2, 99.]], dtype=np.float32),
                   (2, 2, 4, 'array', 'real', 'general'))
# 继承自 TestMMIOArray 类的 TestMMIOSparseCSR 类，用于测试稀疏 CSR 格式的矩阵存储和读取功能
class TestMMIOSparseCSR(TestMMIOArray):

    # 设置测试方法的准备工作，创建临时目录和测试文件路径
    def setup_method(self):
        self.tmpdir = mkdtemp()
        self.fn = os.path.join(self.tmpdir, 'testfile.mtx')

    # 设置测试方法的清理工作，删除临时目录及其下所有内容
    def teardown_method(self):
        shutil.rmtree(self.tmpdir)

    # 检查方法，用于测试稀疏矩阵的写入、信息读取和读取后的数据一致性验证
    def check(self, a, info):
        mmwrite(self.fn, a)
        assert_equal(mminfo(self.fn), info)
        b = mmread(self.fn)
        assert_array_almost_equal(a.toarray(), b.toarray())

    # 精确检查方法，用于测试稀疏矩阵的写入、信息读取和读取后的数据完全一致性验证
    def check_exact(self, a, info):
        mmwrite(self.fn, a)
        assert_equal(mminfo(self.fn), info)
        b = mmread(self.fn)
        assert_equal(a.toarray(), b.toarray())

    # 参数化测试方法，测试稀疏 CSR 矩阵中的简单整数类型数据
    @pytest.mark.parametrize('typeval, dtype', parametrize_args)
    def test_simple_integer(self, typeval, dtype):
        self.check_exact(scipy.sparse.csr_matrix([[1, 2], [3, 4]], dtype=dtype),
                         (2, 2, 4, 'coordinate', typeval, 'general'))

    # 测试 32 位整数数据的稀疏 CSR 矩阵写入和读取
    def test_32bit_integer(self):
        a = scipy.sparse.csr_matrix(array([[2**31-1, -2**31+2],
                                           [2**31-3, 2**31-4]],
                                          dtype=np.int32))
        self.check_exact(a, (2, 2, 4, 'coordinate', 'integer', 'general'))

    # 测试 64 位整数数据的稀疏 CSR 矩阵写入和读取，如果整数溢出则抛出异常
    def test_64bit_integer(self):
        a = scipy.sparse.csr_matrix(array([[2**32+1, 2**32+1],
                                           [-2**63+2, 2**63-2]],
                                          dtype=np.int64))
        if (np.intp(0).itemsize < 8) and mmwrite == scipy.io._mmio.mmwrite:
            assert_raises(OverflowError, mmwrite, self.fn, a)
        else:
            self.check_exact(a, (2, 2, 4, 'coordinate', 'integer', 'general'))

    # 测试 32 位无符号整数数据的稀疏 CSR 矩阵写入和读取
    def test_32bit_unsigned_integer(self):
        a = scipy.sparse.csr_matrix(array([[2**31-1, 2**31-2],
                                           [2**31-3, 2**31-4]],
                                          dtype=np.uint32))
        self.check_exact(a, (2, 2, 4, 'coordinate', 'unsigned-integer', 'general'))

    # 测试 64 位无符号整数数据的稀疏 CSR 矩阵写入和读取
    def test_64bit_unsigned_integer(self):
        a = scipy.sparse.csr_matrix(array([[2**32+1, 2**32+1],
                                           [2**64-2, 2**64-1]],
                                          dtype=np.uint64))
        self.check_exact(a, (2, 2, 4, 'coordinate', 'unsigned-integer', 'general'))

    # 参数化测试方法，测试稀疏 CSR 矩阵中的上三角整数类型数据
    @pytest.mark.parametrize('typeval, dtype', parametrize_args)
    def test_simple_upper_triangle_integer(self, typeval, dtype):
        self.check_exact(scipy.sparse.csr_matrix([[0, 1], [0, 0]], dtype=dtype),
                         (2, 2, 1, 'coordinate', typeval, 'general'))

    # 参数化测试方法，测试稀疏 CSR 矩阵中的下三角整数类型数据
    @pytest.mark.parametrize('typeval, dtype', parametrize_args)
    def test_simple_lower_triangle_integer(self, typeval, dtype):
        self.check_exact(scipy.sparse.csr_matrix([[0, 0], [1, 0]], dtype=dtype),
                         (2, 2, 1, 'coordinate', typeval, 'general'))
    # 定义一个测试方法，用于测试稀疏矩阵的形状、值类型等特性
    def test_simple_rectangular_integer(self, typeval, dtype):
        # 调用自定义的检查方法，验证给定的稀疏矩阵是否符合预期
        self.check_exact(scipy.sparse.csr_matrix([[1, 2, 3], [4, 5, 6]], dtype=dtype),
                         (2, 3, 6, 'coordinate', typeval, 'general'))

    # 定义一个测试方法，测试包含浮点数的简单矩阵
    def test_simple_rectangular_float(self):
        # 调用自定义的检查方法，验证给定的稀疏矩阵是否符合预期
        self.check(scipy.sparse.csr_matrix([[1, 2], [3.5, 4], [5, 6]]),
                   (3, 2, 6, 'coordinate', 'real', 'general'))

    # 定义一个测试方法，测试包含浮点数的简单矩阵
    def test_simple_float(self):
        # 调用自定义的检查方法，验证给定的稀疏矩阵是否符合预期
        self.check(scipy.sparse.csr_matrix([[1, 2], [3, 4.0]]),
                   (2, 2, 4, 'coordinate', 'real', 'general'))

    # 定义一个测试方法，测试包含复数的简单矩阵
    def test_simple_complex(self):
        # 调用自定义的检查方法，验证给定的稀疏矩阵是否符合预期
        self.check(scipy.sparse.csr_matrix([[1, 2], [3, 4j]]),
                   (2, 2, 4, 'coordinate', 'complex', 'general'))

    # 使用 pytest 的参数化功能，定义一个测试方法，测试对称整数矩阵
    @pytest.mark.parametrize('typeval, dtype', parametrize_args)
    def test_simple_symmetric_integer(self, typeval, dtype):
        # 调用自定义的检查方法，验证给定的稀疏对称矩阵是否符合预期
        self.check_exact(scipy.sparse.csr_matrix([[1, 2], [2, 4]], dtype=dtype),
                         (2, 2, 3, 'coordinate', typeval, 'symmetric'))

    # 定义一个测试方法，测试包含整数的反对称矩阵
    def test_simple_skew_symmetric_integer(self):
        # 调用自定义的检查方法，验证给定的稀疏反对称矩阵是否符合预期
        self.check_exact(scipy.sparse.csr_matrix([[0, 2], [-2, 0]]),
                         (2, 2, 1, 'coordinate', 'integer', 'skew-symmetric'))

    # 定义一个测试方法，测试包含浮点数的反对称矩阵
    def test_simple_skew_symmetric_float(self):
        # 调用自定义的检查方法，验证给定的稀疏反对称矩阵是否符合预期
        self.check(scipy.sparse.csr_matrix(array([[0, 2], [-2.0, 0]], 'f')),
                   (2, 2, 1, 'coordinate', 'real', 'skew-symmetric'))

    # 定义一个测试方法，测试包含复数的共轭对称矩阵
    def test_simple_hermitian_complex(self):
        # 调用自定义的检查方法，验证给定的稀疏共轭对称矩阵是否符合预期
        self.check(scipy.sparse.csr_matrix([[1, 2+3j], [2-3j, 4]]),
                   (2, 2, 3, 'coordinate', 'complex', 'hermitian'))

    # 定义一个测试方法，测试随机生成的对称浮点数矩阵
    def test_random_symmetric_float(self):
        sz = (20, 20)
        a = np.random.random(sz)
        a = a + transpose(a)
        a = scipy.sparse.csr_matrix(a)
        # 调用自定义的检查方法，验证给定的稀疏对称矩阵是否符合预期
        self.check(a, (20, 20, 210, 'coordinate', 'real', 'symmetric'))

    # 定义一个测试方法，测试随机生成的矩形浮点数矩阵
    def test_random_rectangular_float(self):
        sz = (20, 15)
        a = np.random.random(sz)
        a = scipy.sparse.csr_matrix(a)
        # 调用自定义的检查方法，验证给定的稀疏矩阵是否符合预期
        self.check(a, (20, 15, 300, 'coordinate', 'real', 'general'))

    # 定义一个测试方法，测试稀疏模式矩阵
    def test_simple_pattern(self):
        a = scipy.sparse.csr_matrix([[0, 1.5], [3.0, 2.5]])
        p = np.zeros_like(a.toarray())
        p[a.toarray() > 0] = 1
        info = (2, 2, 3, 'coordinate', 'pattern', 'general')
        mmwrite(self.fn, a, field='pattern')
        # 验证写入的稀疏模式矩阵文件是否与预期信息一致
        assert_equal(mminfo(self.fn), info)
        # 读取稀疏模式矩阵文件，并验证其内容与预期的稀疏模式矩阵是否一致
        b = mmread(self.fn)
        assert_array_almost_equal(p, b.toarray())

    # 定义一个测试方法，测试非反对称整数矩阵
    def test_gh13634_non_skew_symmetric_int(self):
        a = scipy.sparse.csr_matrix([[1, 2], [-2, 99]], dtype=np.int32)
        # 调用自定义的检查方法，验证给定的稀疏矩阵是否符合预期
        self.check_exact(a, (2, 2, 4, 'coordinate', 'integer', 'general'))

    # 定义一个测试方法，测试非反对称浮点数矩阵
    def test_gh13634_non_skew_symmetric_float(self):
        a = scipy.sparse.csr_matrix([[1, 2], [-2, 99.]], dtype=np.float32)
        # 调用自定义的检查方法，验证给定的稀疏矩阵是否符合预期
        self.check(a, (2, 2, 4, 'coordinate', 'real', 'general'))
#`
# 定义一个包含 MatrixMarket 格式的 32 位整数稠密矩阵的字符串
_32bit_integer_dense_example = '''\
%%MatrixMarket matrix array integer general
2  2
2147483647
2147483646
2147483647
2147483646
'''

# 定义一个包含 MatrixMarket 格式的 32 位整数稀疏矩阵的字符串
_32bit_integer_sparse_example = '''\
%%MatrixMarket matrix coordinate integer symmetric
2  2  2
1  1  2147483647
2  2  2147483646
'''

# 定义一个包含 MatrixMarket 格式的 64 位整数稠密矩阵的字符串
_64bit_integer_dense_example = '''\
%%MatrixMarket matrix array integer general
2  2
          2147483648
-9223372036854775806
         -2147483648
 9223372036854775807
'''

# 定义一个包含 MatrixMarket 格式的 64 位整数稀疏矩阵（一般形式）的字符串
_64bit_integer_sparse_general_example = '''\
%%MatrixMarket matrix coordinate integer general
2  2  3
1  1  2147483648
1  2  9223372036854775807
2  2  9223372036854775807
'''

# 定义一个包含 MatrixMarket 格式的 64 位整数稀疏矩阵（对称形式）的字符串
_64bit_integer_sparse_symmetric_example = '''\
%%MatrixMarket matrix coordinate integer symmetric
2  2  3
1  1  2147483648
1  2  -9223372036854775807
2  2  9223372036854775807
'''

# 定义一个包含 MatrixMarket 格式的 64 位整数稀疏矩阵（反对称形式）的字符串
_64bit_integer_sparse_skew_example = '''\
%%MatrixMarket matrix coordinate integer skew-symmetric
2  2  3
1  1  2147483648
1  2  -9223372036854775807
2  2  9223372036854775807
'''

# 定义一个包含 MatrixMarket 格式的超过 64 位整数稠密矩阵的字符串
_over64bit_integer_dense_example = '''\
%%MatrixMarket matrix array integer general
2  2
         2147483648
9223372036854775807
         2147483648
9223372036854775808
'''

# 定义一个包含 MatrixMarket 格式的超过 64 位整数稀疏矩阵的字符串
_over64bit_integer_sparse_example = '''\
%%MatrixMarket matrix coordinate integer symmetric
2  2  2
1  1  2147483648
2  2  19223372036854775808
'''

# 定义一个测试类 TestMMIOReadLargeIntegers，用于测试 MatrixMarket 文件的读取
class TestMMIOReadLargeIntegers:
    # 初始化方法，在每个测试方法运行前执行
    def setup_method(self):
        # 创建一个临时目录用于存储测试文件
        self.tmpdir = mkdtemp()
        # 创建测试文件的完整路径
        self.fn = os.path.join(self.tmpdir, 'testfile.mtx')

    # 清理方法，在每个测试方法运行后执行
    def teardown_method(self):
        # 删除临时目录及其中的所有文件
        shutil.rmtree(self.tmpdir)

    # 检查文件读取函数的辅助方法
    def check_read(self, example, a, info, dense, over32, over64):
        # 将 example 字符串写入到测试文件中
        with open(self.fn, 'w') as f:
            f.write(example)
        # 检查文件的元数据是否与预期匹配
        assert_equal(mminfo(self.fn), info)
        # 根据条件判断是否抛出溢出错误
        if ((over32 and (np.intp(0).itemsize < 8) and mmwrite == scipy.io._mmio.mmwrite)
            or over64):
            assert_raises(OverflowError, mmread, self.fn)
        else:
            # 读取矩阵数据
            b = mmread(self.fn)
            # 将稀疏矩阵转换为数组形式
            if not dense:
                b = b.toarray()
            # 验证读取的数据是否与预期一致
            assert_equal(a, b)

    # 测试读取 32 位整数稠密矩阵
    def test_read_32bit_integer_dense(self):
        a = array([[2**31-1, 2**31-1],
                   [2**31-2, 2**31-2]], dtype=np.int64)
        # 调用 check_read 方法进行测试
        self.check_read(_32bit_integer_dense_example,
                        a,
                        (2, 2, 4, 'array', 'integer', 'general'),
                        dense=True,
                        over32=False,
                        over64=False)

    # 测试读取 32 位整数稀疏矩阵
    def test_read_32bit_integer_sparse(self):
        a = array([[2**31-1, 0],
                   [0, 2**31-2]], dtype=np.int64)
        # 调用 check_read 方法进行测试
        self.check_read(_32bit_integer_sparse_example,
                        a,
                        (2, 2, 2, 'coordinate', 'integer', 'symmetric'),
                        dense=False,
                        over32=False,
                        over64=False)
    # 测试读取包含64位整数的稠密数组
    def test_read_64bit_integer_dense(self):
        # 创建包含64位整数的NumPy数组
        a = array([[2**31, -2**31],
                   [-2**63+2, 2**63-1]], dtype=np.int64)
        # 使用自定义检查函数检查读取结果
        self.check_read(_64bit_integer_dense_example,
                        a,
                        (2, 2, 4, 'array', 'integer', 'general'),
                        dense=True,
                        over32=True,
                        over64=False)
    
    # 测试读取包含64位整数的稀疏一般数组
    def test_read_64bit_integer_sparse_general(self):
        # 创建包含64位整数的稀疏一般NumPy数组
        a = array([[2**31, 2**63-1],
                   [0, 2**63-1]], dtype=np.int64)
        # 使用自定义检查函数检查读取结果
        self.check_read(_64bit_integer_sparse_general_example,
                        a,
                        (2, 2, 3, 'coordinate', 'integer', 'general'),
                        dense=False,
                        over32=True,
                        over64=False)
    
    # 测试读取包含64位整数的稀疏对称数组
    def test_read_64bit_integer_sparse_symmetric(self):
        # 创建包含64位整数的稀疏对称NumPy数组
        a = array([[2**31, -2**63+1],
                   [-2**63+1, 2**63-1]], dtype=np.int64)
        # 使用自定义检查函数检查读取结果
        self.check_read(_64bit_integer_sparse_symmetric_example,
                        a,
                        (2, 2, 3, 'coordinate', 'integer', 'symmetric'),
                        dense=False,
                        over32=True,
                        over64=False)
    
    # 测试读取包含64位整数的稀疏偏斜对称数组
    def test_read_64bit_integer_sparse_skew(self):
        # 创建包含64位整数的稀疏偏斜对称NumPy数组
        a = array([[2**31, -2**63+1],
                   [2**63-1, 2**63-1]], dtype=np.int64)
        # 使用自定义检查函数检查读取结果
        self.check_read(_64bit_integer_sparse_skew_example,
                        a,
                        (2, 2, 3, 'coordinate', 'integer', 'skew-symmetric'),
                        dense=False,
                        over32=True,
                        over64=False)
    
    # 测试读取超过64位整数的稠密数组
    def test_read_over64bit_integer_dense(self):
        # 使用自定义检查函数检查读取结果，期望结果为None
        self.check_read(_over64bit_integer_dense_example,
                        None,
                        (2, 2, 4, 'array', 'integer', 'general'),
                        dense=True,
                        over32=True,
                        over64=True)
    
    # 测试读取超过64位整数的稀疏数组
    def test_read_over64bit_integer_sparse(self):
        # 使用自定义检查函数检查读取结果，期望结果为None
        self.check_read(_over64bit_integer_sparse_example,
                        None,
                        (2, 2, 2, 'coordinate', 'integer', 'symmetric'),
                        dense=False,
                        over32=True,
                        over64=True)
# 定义一个多行字符串，包含一个通用的稀疏矩阵示例，使用 Matrix Market 格式
_general_example = '''\
%%MatrixMarket matrix coordinate real general
%=================================================================================
%
% This ASCII file represents a sparse MxN matrix with L
% nonzeros in the following Matrix Market format:
%
% +----------------------------------------------+
% |%%MatrixMarket matrix coordinate real general | <--- 标题行
% |%                                             | <--+
% |% comments                                    |    |-- 0 或多个注释行
% |%                                             | <--+
% |    M  N  L                                   | <--- 行数、列数、条目数
% |    I1  J1  A(I1, J1)                         | <--+
% |    I2  J2  A(I2, J2)                         |    |
% |    I3  J3  A(I3, J3)                         |    |-- L 行数据
% |        . . .                                 |    |
% |    IL JL  A(IL, JL)                          | <--+
% +----------------------------------------------+
%
% Indices are 1-based, i.e. A(1,1) is the first element.
%
%=================================================================================
  5  5  8
    1     1   1.000e+00
    2     2   1.050e+01
    3     3   1.500e-02
    1     4   6.000e+00
    4     2   2.505e+02
    4     4  -2.800e+02
    4     5   3.332e+01
    5     5   1.200e+01
'''

# 定义一个多行字符串，包含一个 Hermitian 复数矩阵的示例，使用 Matrix Market 格式
_hermitian_example = '''\
%%MatrixMarket matrix coordinate complex hermitian
  5  5  7
    1     1     1.0      0
    2     2    10.5      0
    4     2   250.5     22.22
    3     3     1.5e-2   0
    4     4    -2.8e2    0
    5     5    12.       0
    5     4     0       33.32
'''

# 定义一个多行字符串，包含一个实数反对称矩阵的示例，使用 Matrix Market 格式
_skew_example = '''\
%%MatrixMarket matrix coordinate real skew-symmetric
  5  5  7
    1     1     1.0
    2     2    10.5
    4     2   250.5
    3     3     1.5e-2
    4     4    -2.8e2
    5     5    12.
    5     4     0
'''

# 定义一个多行字符串，包含一个实数对称矩阵的示例，使用 Matrix Market 格式
_symmetric_example = '''\
%%MatrixMarket matrix coordinate real symmetric
  5  5  7
    1     1     1.0
    2     2    10.5
    4     2   250.5
    3     3     1.5e-2
    4     4    -2.8e2
    5     5    12.
    5     4     8
'''

# 定义一个多行字符串，包含一个对称模式矩阵的示例，使用 Matrix Market 格式
_symmetric_pattern_example = '''\
%%MatrixMarket matrix coordinate pattern symmetric
  5  5  7
    1     1
    2     2
    4     2
    3     3
    4     4
    5     5
    5     4
'''

# 定义一个多行字符串，包含一个稀疏矩阵示例，使用 Matrix Market 格式，其中包含空行
_empty_lines_example = '''\
%%MatrixMarket  MATRIX    Coordinate    Real General

   5  5         8

1 1  1.0
2 2       10.5
3 3             1.5e-2
4 4                     -2.8E2
5 5                              12.
     1      4      6
     4      2      250.5
     4      5      33.32
'''

# 定义一个测试类，用于测试 Matrix Market 格式的输入输出，包含设置方法和清理方法
class TestMMIOCoordinate:
    def setup_method(self):
        # 创建临时目录
        self.tmpdir = mkdtemp()
        # 设置文件名为临时目录下的 testfile.mtx
        self.fn = os.path.join(self.tmpdir, 'testfile.mtx')

    def teardown_method(self):
        # 删除临时目录及其内容
        shutil.rmtree(self.tmpdir)
    # 定义一个用于测试读取功能的方法，接受参数example, a, info
    def check_read(self, example, a, info):
        # 打开文件以写入example内容
        f = open(self.fn, 'w')
        # 写入example内容到文件
        f.write(example)
        # 关闭文件
        f.close()
        # 断言文件信息是否与预期的info相等
        assert_equal(mminfo(self.fn), info)
        # 从文件中读取稀疏矩阵数据并转换为稠密数组，并赋给变量b
        b = mmread(self.fn).toarray()
        # 断言读取的数组b与预期数组a近似相等
        assert_array_almost_equal(a, b)

    # 测试通用读取功能
    def test_read_general(self):
        # 预期的稀疏矩阵a
        a = [[1, 0, 0, 6, 0],
             [0, 10.5, 0, 0, 0],
             [0, 0, .015, 0, 0],
             [0, 250.5, 0, -280, 33.32],
             [0, 0, 0, 0, 12]]
        # 调用check_read方法，传入通用示例_example, 矩阵a以及预期的元组info
        self.check_read(_general_example, a,
                        (5, 5, 8, 'coordinate', 'real', 'general'))

    # 测试读取Hermitian对称矩阵功能
    def test_read_hermitian(self):
        # 预期的Hermitian对称矩阵a
        a = [[1, 0, 0, 0, 0],
             [0, 10.5, 0, 250.5 - 22.22j, 0],
             [0, 0, .015, 0, 0],
             [0, 250.5 + 22.22j, 0, -280, -33.32j],
             [0, 0, 0, 33.32j, 12]]
        # 调用check_read方法，传入Hermitian示例_example, 矩阵a以及预期的元组info
        self.check_read(_hermitian_example, a,
                        (5, 5, 7, 'coordinate', 'complex', 'hermitian'))

    # 测试读取Skew对称矩阵功能
    def test_read_skew(self):
        # 预期的Skew对称矩阵a
        a = [[1, 0, 0, 0, 0],
             [0, 10.5, 0, -250.5, 0],
             [0, 0, .015, 0, 0],
             [0, 250.5, 0, -280, 0],
             [0, 0, 0, 0, 12]]
        # 调用check_read方法，传入Skew示例_example, 矩阵a以及预期的元组info
        self.check_read(_skew_example, a,
                        (5, 5, 7, 'coordinate', 'real', 'skew-symmetric'))

    # 测试读取对称矩阵功能
    def test_read_symmetric(self):
        # 预期的对称矩阵a
        a = [[1, 0, 0, 0, 0],
             [0, 10.5, 0, 250.5, 0],
             [0, 0, .015, 0, 0],
             [0, 250.5, 0, -280, 8],
             [0, 0, 0, 8, 12]]
        # 调用check_read方法，传入对称矩阵示例_example, 矩阵a以及预期的元组info
        self.check_read(_symmetric_example, a,
                        (5, 5, 7, 'coordinate', 'real', 'symmetric'))

    # 测试读取对称模式矩阵功能
    def test_read_symmetric_pattern(self):
        # 预期的对称模式矩阵a
        a = [[1, 0, 0, 0, 0],
             [0, 1, 0, 1, 0],
             [0, 0, 1, 0, 0],
             [0, 1, 0, 1, 1],
             [0, 0, 0, 1, 1]]
        # 调用check_read方法，传入对称模式矩阵示例_example, 矩阵a以及预期的元组info
        self.check_read(_symmetric_pattern_example, a,
                        (5, 5, 7, 'coordinate', 'pattern', 'symmetric'))

    # 测试读取空行功能
    def test_read_empty_lines(self):
        # 预期的稀疏矩阵a，与test_read_general类似
        a = [[1, 0, 0, 6, 0],
             [0, 10.5, 0, 0, 0],
             [0, 0, .015, 0, 0],
             [0, 250.5, 0, -280, 33.32],
             [0, 0, 0, 0, 12]]
        # 调用check_read方法，传入空行示例_example, 矩阵a以及预期的元组info
        self.check_read(_empty_lines_example, a,
                        (5, 5, 8, 'coordinate', 'real', 'general'))

    # 测试空写入和读取功能
    def test_empty_write_read(self):
        # 创建一个空的稀疏矩阵b
        b = scipy.sparse.coo_matrix((10, 10))
        # 将稀疏矩阵b写入文件self.fn
        mmwrite(self.fn, b)

        # 断言文件信息是否与预期的元组相等
        assert_equal(mminfo(self.fn),
                     (10, 10, 0, 'coordinate', 'real', 'symmetric'))
        # 将矩阵b转换为稠密数组a
        a = b.toarray()
        # 从文件中读取稀疏矩阵数据并转换为稠密数组，并赋给变量b
        b = mmread(self.fn).toarray()
        # 断言读取的数组b与预期数组a近似相等
        assert_array_almost_equal(a, b)
    def test_bzip2_py3(self):
        # 测试修复＃2152是否有效
        try:
            # 尝试导入bz2模块，因为有时Python编译时未包含该模块
            import bz2
        except ImportError:
            # 如果导入失败，则退出函数
            return
        
        # 创建稀疏矩阵的行索引
        I = array([0, 0, 1, 2, 3, 3, 3, 4])
        # 创建稀疏矩阵的列索引
        J = array([0, 3, 1, 2, 1, 3, 4, 4])
        # 创建稀疏矩阵的值
        V = array([1.0, 6.0, 10.5, 0.015, 250.5, -280.0, 33.32, 12.0])

        # 创建稀疏矩阵COO格式对象
        b = scipy.sparse.coo_matrix((V, (I, J)), shape=(5, 5))

        # 将稀疏矩阵写入文件
        mmwrite(self.fn, b)

        # 创建压缩后文件名
        fn_bzip2 = "%s.bz2" % self.fn
        # 以二进制只读方式打开文件
        with open(self.fn, 'rb') as f_in:
            # 创建BZ2文件对象
            f_out = bz2.BZ2File(fn_bzip2, 'wb')
            # 将输入文件内容写入BZ2文件中
            f_out.write(f_in.read())
            # 关闭BZ2文件
            f_out.close()

        # 从压缩文件中读取稀疏矩阵并转换为数组
        a = mmread(fn_bzip2).toarray()
        # 断言压缩文件中读取的数组与原始稀疏矩阵数组近似相等
        assert_array_almost_equal(a, b.toarray())

    def test_gzip_py3(self):
        # 测试修复＃2152是否有效
        try:
            # 尝试导入gzip模块，因为有时Python安装未包含该模块
            import gzip
        except ImportError:
            # 如果导入失败，则退出函数
            return
        
        # 创建稀疏矩阵的行索引
        I = array([0, 0, 1, 2, 3, 3, 3, 4])
        # 创建稀疏矩阵的列索引
        J = array([0, 3, 1, 2, 1, 3, 4, 4])
        # 创建稀疏矩阵的值
        V = array([1.0, 6.0, 10.5, 0.015, 250.5, -280.0, 33.32, 12.0])

        # 创建稀疏矩阵COO格式对象
        b = scipy.sparse.coo_matrix((V, (I, J)), shape=(5, 5))

        # 将稀疏矩阵写入文件
        mmwrite(self.fn, b)

        # 创建压缩后文件名
        fn_gzip = "%s.gz" % self.fn
        # 以二进制只读方式打开文件
        with open(self.fn, 'rb') as f_in:
            # 创建Gzip文件对象
            f_out = gzip.open(fn_gzip, 'wb')
            # 将输入文件内容写入Gzip文件中
            f_out.write(f_in.read())
            # 关闭Gzip文件
            f_out.close()

        # 从压缩文件中读取稀疏矩阵并转换为数组
        a = mmread(fn_gzip).toarray()
        # 断言压缩文件中读取的数组与原始稀疏矩阵数组近似相等
        assert_array_almost_equal(a, b.toarray())

    def test_real_write_read(self):
        # 创建稀疏矩阵的行索引
        I = array([0, 0, 1, 2, 3, 3, 3, 4])
        # 创建稀疏矩阵的列索引
        J = array([0, 3, 1, 2, 1, 3, 4, 4])
        # 创建稀疏矩阵的值
        V = array([1.0, 6.0, 10.5, 0.015, 250.5, -280.0, 33.32, 12.0])

        # 创建稀疏矩阵COO格式对象
        b = scipy.sparse.coo_matrix((V, (I, J)), shape=(5, 5))

        # 将稀疏矩阵写入文件
        mmwrite(self.fn, b)

        # 断言读取的文件信息与预期信息相等
        assert_equal(mminfo(self.fn),
                     (5, 5, 8, 'coordinate', 'real', 'general'))
        # 将原始稀疏矩阵转换为数组
        a = b.toarray()
        # 从文件中读取稀疏矩阵并转换为数组
        b = mmread(self.fn).toarray()
        # 断言从文件中读取的数组与原始稀疏矩阵数组相等
        assert_array_almost_equal(a, b)

    def test_complex_write_read(self):
        # 创建稀疏矩阵的行索引
        I = array([0, 0, 1, 2, 3, 3, 3, 4])
        # 创建稀疏矩阵的列索引
        J = array([0, 3, 1, 2, 1, 3, 4, 4])
        # 创建稀疏矩阵的复数值
        V = array([1.0 + 3j, 6.0 + 2j, 10.50 + 0.9j, 0.015 - 4.4j,
                   250.5 + 0j, -280.0 + 5j, 33.32 + 6.4j, 12.00 + 0.8j])

        # 创建稀疏矩阵COO格式对象
        b = scipy.sparse.coo_matrix((V, (I, J)), shape=(5, 5))

        # 将稀疏矩阵写入文件
        mmwrite(self.fn, b)

        # 断言读取的文件信息与预期信息相等
        assert_equal(mminfo(self.fn),
                     (5, 5, 8, 'coordinate', 'complex', 'general'))
        # 将原始稀疏矩阵转换为数组
        a = b.toarray()
        # 从文件中读取稀疏矩阵并转换为数组
        b = mmread(self.fn).toarray()
        # 断言从文件中读取的数组与原始稀疏矩阵数组相等
        assert_array_almost_equal(a, b)
    # 定义测试稀疏格式的方法，使用 tmp_path 参数作为 pytest 的固定装置，它会处理清理工作
    def test_sparse_formats(self, tmp_path):
        # 创建一个临时目录 `sparse_formats` 在 tmp_path 下
        tmpdir = tmp_path / 'sparse_formats'
        tmpdir.mkdir()

        # 初始化稀疏矩阵的行索引和列索引
        I = array([0, 0, 1, 2, 3, 3, 3, 4])
        J = array([0, 3, 1, 2, 1, 3, 4, 4])

        # 第一个稀疏矩阵的数值
        V = array([1.0, 6.0, 10.5, 0.015, 250.5, -280.0, 33.32, 12.0])
        # 使用 scipy.sparse.coo_matrix 创建稀疏 COO 格式矩阵，并添加到 mats 列表中
        mats.append(scipy.sparse.coo_matrix((V, (I, J)), shape=(5, 5)))

        # 第二个稀疏矩阵的数值（包含复数）
        V = array([1.0 + 3j, 6.0 + 2j, 10.50 + 0.9j, 0.015 + -4.4j,
                   250.5 + 0j, -280.0 + 5j, 33.32 + 6.4j, 12.00 + 0.8j])
        # 使用 scipy.sparse.coo_matrix 创建稀疏 COO 格式矩阵，并添加到 mats 列表中
        mats.append(scipy.sparse.coo_matrix((V, (I, J)), shape=(5, 5)))

        # 遍历 mats 列表中的每个稀疏矩阵
        for mat in mats:
            # 期望的稀疏矩阵转换为稠密数组
            expected = mat.toarray()
            # 遍历格式列表 ['csr', 'csc', 'coo']
            for fmt in ['csr', 'csc', 'coo']:
                # 构造文件名，将当前格式的稀疏矩阵写入到临时目录中
                fname = tmpdir / (fmt + '.mtx')
                mmwrite(fname, mat.asformat(fmt))
                # 读取写入的稀疏矩阵文件并转换为稠密数组
                result = mmread(fname).toarray()
                # 断言结果与期望一致
                assert_array_almost_equal(result, expected)

    # 定义测试精度的方法
    def test_precision(self):
        # 测试值列表，包括 pi 和 10 的负幂次方
        test_values = [pi] + [10**(i) for i in range(0, -10, -1)]
        # 测试精度范围
        test_precisions = range(1, 10)
        # 遍历测试值
        for value in test_values:
            # 遍历测试精度
            for precision in test_precisions:
                # 根据测试精度构造稀疏对角矩阵
                n = 10**precision + 1
                A = scipy.sparse.dok_matrix((n, n))
                A[n-1, n-1] = value
                # 将矩阵以指定精度写入文件并重新读取
                mmwrite(self.fn, A, precision=precision)
                A = scipy.io.mmread(self.fn)
                # 检查矩阵的正确输入
                assert_array_equal(A.row, [n-1])
                assert_array_equal(A.col, [n-1])
                assert_allclose(A.data, [float('%.dg' % precision % value)])

    # 定义测试坏的坐标头字段数的方法
    def test_bad_number_of_coordinate_header_fields(self):
        # 定义一个包含不正确的坐标头字段数的矩阵市场格式字符串
        s = """\
            %%MatrixMarket matrix coordinate real general
              5  5  8 999
                1     1   1.000e+00
                2     2   1.050e+01
                3     3   1.500e-02
                1     4   6.000e+00
                4     2   2.505e+02
                4     4  -2.800e+02
                4     5   3.332e+01
                5     5   1.200e+01
            """
        # 去除字符串的缩进并转换为 ASCII 编码
        text = textwrap.dedent(s).encode('ascii')
        # 使用 pytest 的断言检查 ValueError 异常并验证消息包含 'not of length 3'
        with pytest.raises(ValueError, match='not of length 3'):
            scipy.io.mmread(io.BytesIO(text))
# 定义测试函数 test_gh11389，用于测试 mmread 函数处理特定格式输入的情况
def test_gh11389():
    # 创建包含特定格式内容的字符串流对象，传递给 mmread 函数处理
    mmread(io.StringIO("%%MatrixMarket matrix coordinate complex symmetric\n"
                       " 1 1 1\n"
                       "1 1 -2.1846000000000e+02  0.0000000000000e+00"))

# 定义测试函数 test_gh18123，用于测试 mmread 函数处理文件读取的情况
def test_gh18123(tmp_path):
    # 定义包含矩阵 Market 格式内容的列表
    lines = [" %%MatrixMarket matrix coordinate real general\n",
             "5 5 3\n",
             "2 3 1.0\n",
             "3 4 2.0\n",
             "3 5 3.0\n"]
    # 创建临时文件路径对象
    test_file = tmp_path / "test.mtx"
    # 将内容写入临时文件
    with open(test_file, "w") as f:
        f.writelines(lines)
    # 调用 mmread 函数处理临时文件
    mmread(test_file)

# 定义测试函数 test_threadpoolctl，用于测试 threadpoolctl 的并发控制功能
def test_threadpoolctl():
    try:
        # 尝试导入 threadpoolctl 库
        import threadpoolctl
        # 检查 threadpoolctl 版本是否支持 register 属性
        if not hasattr(threadpoolctl, "register"):
            # 若不支持，则跳过测试并给出相应信息
            pytest.skip("threadpoolctl too old")
            return
    except ImportError:
        # 若导入失败，则跳过测试并给出相应信息
        pytest.skip("no threadpoolctl")
        return

    # 使用 threadpoolctl 控制并发线程数上限为 4
    with threadpoolctl.threadpool_limits(limits=4):
        # 断言并发度是否为 4
        assert_equal(fmm.PARALLELISM, 4)

    # 使用 threadpoolctl 控制并发线程数上限为 2，限制应用于 scipy 用户 API
    with threadpoolctl.threadpool_limits(limits=2, user_api='scipy'):
        # 断言并发度是否为 2
        assert_equal(fmm.PARALLELISM, 2)
```