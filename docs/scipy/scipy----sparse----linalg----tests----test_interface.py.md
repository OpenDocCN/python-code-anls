# `D:\src\scipysrc\scipy\scipy\sparse\linalg\tests\test_interface.py`

```
"""Test functions for the sparse.linalg._interface module
"""

# 从 functools 模块导入 partial 函数，用于创建 partial 函数应用
from functools import partial
# 从 itertools 模块导入 product 函数，用于迭代生成 Cartesian product
from itertools import product
# 导入 operator 模块，用于操作符操作
import operator
# 从 pytest 模块导入 raises 函数并重命名为 assert_raises，用于断言异常
from pytest import raises as assert_raises, warns
# 从 numpy.testing 模块导入 assert_ 和 assert_equal 函数，用于断言相等
from numpy.testing import assert_, assert_equal

# 导入 numpy 和 scipy.sparse 模块
import numpy as np
import scipy.sparse as sparse

# 导入 scipy.sparse.linalg._interface 模块并重命名为 interface
import scipy.sparse.linalg._interface as interface
# 从 scipy.sparse._sputils 模块导入 matrix 函数
from scipy.sparse._sputils import matrix


class TestLinearOperator:
    def setup_method(self):
        # 初始化测试用例的矩阵 A, B, C
        self.A = np.array([[1,2,3],
                           [4,5,6]])
        self.B = np.array([[1,2],
                           [3,4],
                           [5,6]])
        self.C = np.array([[1,2],
                           [3,4]])

    def test_matmul(self):
        # 定义字典 D 包含 LinearOperator 的初始化参数
        D = {'shape': self.A.shape,
             'matvec': lambda x: np.dot(self.A, x).reshape(self.A.shape[0]),
             'rmatvec': lambda x: np.dot(self.A.T.conj(),
                                         x).reshape(self.A.shape[1]),
             'rmatmat': lambda x: np.dot(self.A.T.conj(), x),
             'matmat': lambda x: np.dot(self.A, x)}
        # 创建 LinearOperator 对象 A，使用 D 中的参数
        A = interface.LinearOperator(**D)
        
        # 定义测试用例中的矩阵 B 和向量 b
        B = np.array([[1 + 1j, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])
        b = B[0]

        # 进行 matmul 操作的断言
        assert_equal(operator.matmul(A, b), A * b)
        assert_equal(operator.matmul(A, b.reshape(-1, 1)), A * b.reshape(-1, 1))
        assert_equal(operator.matmul(A, B), A * B)
        assert_equal(operator.matmul(b, A.H), b * A.H)
        assert_equal(operator.matmul(b.reshape(1, -1), A.H), b.reshape(1, -1) * A.H)
        assert_equal(operator.matmul(B, A.H), B * A.H)
        
        # 断言异常情况
        assert_raises(ValueError, operator.matmul, A, 2)
        assert_raises(ValueError, operator.matmul, 2, A)


class TestAsLinearOperator:
    # 测试基本功能的单元测试
    def test_basic(self):

        # 遍历测试用例
        for M, A_array in self.cases:
            # 将M转换为线性操作符A
            A = interface.aslinearoperator(M)
            # 获取A的形状
            M, N = A.shape

            # 定义不同类型的输入向量和矩阵
            xs = [np.array([1, 2, 3]),               # 一维数组
                  np.array([[1], [2], [3]])]         # 二维数组
            ys = [np.array([1, 2]),                  # 一维数组
                  np.array([[1], [2]])]              # 二维数组

            # 如果A的数据类型是复数（complex128），添加额外的测试用例
            if A.dtype == np.complex128:
                xs += [np.array([1, 2j, 3j]),        # 复数一维数组
                       np.array([[1], [2j], [3j]])]  # 复数二维数组
                ys += [np.array([1, 2j]),            # 复数一维数组
                       np.array([[1], [2j]])]        # 复数二维数组

            # 定义一个二维数组
            x2 = np.array([[1, 4], [2, 5], [3, 6]])

            # 遍历输入向量和矩阵进行断言测试
            for x in xs:
                assert_equal(A.matvec(x), A_array.dot(x))   # 断言A与A_array.dot(x)的向量乘积相等
                assert_equal(A * x, A_array.dot(x))         # 断言A与A_array.dot(x)的向量乘积相等

            # 断言A与A_array.dot(x2)的矩阵乘积相等
            assert_equal(A.matmat(x2), A_array.dot(x2))
            # 断言A与A_array.dot(x2)的矩阵乘积相等
            assert_equal(A * x2, A_array.dot(x2))

            # 遍历输入向量和矩阵进行右乘转置向量的断言测试
            for y in ys:
                assert_equal(A.rmatvec(y), A_array.T.conj().dot(y))   # 断言A的右乘转置向量与A_array.T.conj().dot(y)相等
                assert_equal(A.T.matvec(y), A_array.T.dot(y))         # 断言A的转置向量乘积与A_array.T.dot(y)相等
                assert_equal(A.H.matvec(y), A_array.T.conj().dot(y))  # 断言A的共轭转置向量乘积与A_array.T.conj().dot(y)相等

            # 遍历输入向量和矩阵进行右乘转置矩阵的断言测试
            for y in ys:
                if y.ndim < 2:
                    continue
                assert_equal(A.rmatmat(y), A_array.T.conj().dot(y))   # 断言A的右乘转置矩阵与A_array.T.conj().dot(y)相等
                assert_equal(A.T.matmat(y), A_array.T.dot(y))         # 断言A的转置矩阵乘积与A_array.T.dot(y)相等
                assert_equal(A.H.matmat(y), A_array.T.conj().dot(y))  # 断言A的共轭转置矩阵乘积与A_array.T.conj().dot(y)相等

            # 如果M具有dtype属性，则断言A的dtype与M的dtype相等
            if hasattr(M, 'dtype'):
                assert_equal(A.dtype, M.dtype)

            # 断言A具有'args'属性
            assert_(hasattr(A, 'args'))

    # 测试dot方法的单元测试
    def test_dot(self):

        # 遍历测试用例
        for M, A_array in self.cases:
            # 将M转换为线性操作符A
            A = interface.aslinearoperator(M)
            # 获取A的形状
            M, N = A.shape

            # 定义不同类型的输入向量和矩阵
            x0 = np.array([1, 2, 3])         # 一维数组
            x1 = np.array([[1], [2], [3]])   # 二维数组
            x2 = np.array([[1, 4], [2, 5], [3, 6]])   # 二维数组

            # 断言A与A_array.dot(x0)的乘积相等
            assert_equal(A.dot(x0), A_array.dot(x0))
            # 断言A与A_array.dot(x1)的乘积相等
            assert_equal(A.dot(x1), A_array.dot(x1))
            # 断言A与A_array.dot(x2)的乘积相等
            assert_equal(A.dot(x2), A_array.dot(x2))
def test_repr():
    # 创建一个 shape 为 (1, 1) 的线性操作符 A，其作用是将输入向量映射为常数 1
    A = interface.LinearOperator(shape=(1, 1), matvec=lambda x: 1)
    # 获取 A 的字符串表示形式
    repr_A = repr(A)
    # 断言字符串 "unspecified dtype" 不在 repr_A 中，否则抛出异常并输出 repr_A
    assert_('unspecified dtype' not in repr_A, repr_A)


def test_identity():
    # 创建一个 3x3 的单位矩阵操作符 ident
    ident = interface.IdentityOperator((3, 3))
    # 断言 ident 作用于向量 [1, 2, 3] 后得到 [1, 2, 3]
    assert_equal(ident * [1, 2, 3], [1, 2, 3])
    # 断言 ident 作用于矩阵 np.arange(9).reshape(3, 3) 后展平得到 np.arange(9)
    assert_equal(ident.dot(np.arange(9).reshape(3, 3)).ravel(), np.arange(9))
    # 断言 ident 作用于长度不符合预期的向量 [1, 2, 3, 4] 时抛出 ValueError 异常
    assert_raises(ValueError, ident.matvec, [1, 2, 3, 4])


def test_attributes():
    # 将 np.arange(16).reshape(4, 4) 转换为线性操作符 A
    A = interface.aslinearoperator(np.arange(16).reshape(4, 4))

    def always_four_ones(x):
        # 将输入 x 转换为 numpy 数组
        x = np.asarray(x)
        # 断言 x 的形状是 (3,) 或者 (3, 1)
        assert_(x.shape == (3,) or x.shape == (3, 1))
        # 返回一个全为 1 的长度为 4 的向量
        return np.ones(4)

    # 创建一个形状为 (4, 3) 的线性操作符 B，其作用是将输入向量映射为全为 1 的向量
    B = interface.LinearOperator(shape=(4, 3), matvec=always_four_ones)

    # 对于列表中的每个操作符 op
    for op in [A, B, A * B, A.H, A + A, B + B, A**4]:
        # 断言 op 具有属性 "dtype"
        assert_(hasattr(op, "dtype"))
        # 断言 op 具有属性 "shape"
        assert_(hasattr(op, "shape"))
        # 断言 op 具有属性 "_matvec"


def matvec(x):
    """ Needed for test_pickle as local functions are not pickleable """
    # 返回一个全为 0 的长度为 3 的向量，用于测试 pickle
    return np.zeros(3)


def test_pickle():
    import pickle

    # 对于所有 pickle 协议编号
    for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
        # 创建一个使用 matvec 函数作为 matvec 的线性操作符 A
        A = interface.LinearOperator((3, 3), matvec)
        # 将 A 序列化为字符串 s，使用当前协议编号
        s = pickle.dumps(A, protocol=protocol)
        # 从字符串 s 反序列化出操作符 B
        B = pickle.loads(s)

        # 断言 A 和 B 的每个属性相等
        for k in A.__dict__:
            assert_equal(getattr(A, k), getattr(B, k))


def test_inheritance():
    # 定义一个空的线性操作符类 Empty，继承自 interface.LinearOperator
    class Empty(interface.LinearOperator):
        pass

    # 断言 Empty 的实例化会引发 TypeError
    with warns(RuntimeWarning, match="should implement at least"):
        assert_raises(TypeError, Empty)

    # 定义一个 Identity 类，继承自 interface.LinearOperator
    class Identity(interface.LinearOperator):
        def __init__(self, n):
            super().__init__(dtype=None, shape=(n, n))

        def _matvec(self, x):
            return x

    # 创建一个 3x3 的 Identity 实例 id3
    id3 = Identity(3)
    # 断言 id3 作用于向量 [1, 2, 3] 后得到 [1, 2, 3]
    assert_equal(id3.matvec([1, 2, 3]), [1, 2, 3])
    # 断言调用 id3 的未实现方法 rmatvec 会引发 NotImplementedError 异常
    assert_raises(NotImplementedError, id3.rmatvec, [4, 5, 6])

    # 定义一个 MatmatOnly 类，继承自 interface.LinearOperator
    class MatmatOnly(interface.LinearOperator):
        def __init__(self, A):
            super().__init__(A.dtype, A.shape)
            self.A = A

        def _matmat(self, x):
            return self.A.dot(x)

    # 创建一个 MatmatOnly 实例 mm，其内部使用随机生成的 5x3 矩阵作为输入矩阵 A
    mm = MatmatOnly(np.random.randn(5, 3))
    # 断言 mm 作用于长度为 3 的随机向量后得到的向量形状为 (5,)
    assert_equal(mm.matvec(np.random.randn(3)).shape, (5,))


def test_dtypes_of_operator_sum():
    # 创建一个复数随机矩阵 mat_complex 和一个实数随机矩阵 mat_real
    mat_complex = np.random.rand(2,2) + 1j * np.random.rand(2,2)
    mat_real = np.random.rand(2,2)

    # 将 mat_complex 和 mat_real 转换为线性操作符
    complex_operator = interface.aslinearoperator(mat_complex)
    real_operator = interface.aslinearoperator(mat_real)

    # 计算两个复数操作符的和 sum_complex 和两个实数操作符的和 sum_real
    sum_complex = complex_operator + complex_operator
    sum_real = real_operator + real_operator

    # 断言 sum_real 的数据类型为 np.float64
    assert_equal(sum_real.dtype, np.float64)
    # 断言 sum_complex 的数据类型为 np.complex128


def test_no_double_init():
    call_count = [0]

    def matvec(v):
        call_count[0] += 1
        return v

    # 实例化一个线性操作符，应该仅调用一次 matvec 函数来确定操作符的数据类型
    interface.LinearOperator((2, 2), matvec=matvec)
    # 断言 matvec 函数确实仅被调用了一次
    assert_equal(call_count[0], 1)


def test_adjoint_conjugate():
    X = np.array([[1j]])
    # 将 X 转换为线性操作符 A
    A = interface.aslinearoperator(X)
    # 将复数单位虚部乘以向量 A，得到复数数组 B
    B = 1j * A

    # 将复数单位虚部乘以向量 X，得到复数数组 Y
    Y = 1j * X

    # 创建一个包含单个元素 1 的 NumPy 数组 v
    v = np.array([1])

    # 断言 B 乘以向量 v 的结果与 Y 乘以向量 v 的结果相等
    assert_equal(B.dot(v), Y.dot(v))

    # 断言 B 的共轭转置乘以向量 v 的结果与 Y 的转置的共轭乘以向量 v 的结果相等
    assert_equal(B.H.dot(v), Y.T.conj().dot(v))
# 定义一个测试函数，用于测试 ndim 属性
def test_ndim():
    # 创建一个包含单个元素的二维 NumPy 数组
    X = np.array([[1]])
    # 调用接口函数将数组 X 转换为线性操作对象 A
    A = interface.aslinearoperator(X)
    # 断言线性操作对象 A 的维度为 2
    assert_equal(A.ndim, 2)

# 定义一个测试函数，用于测试转置操作不进行共轭的情况
def test_transpose_noconjugate():
    # 创建一个包含复数 1j 的二维 NumPy 数组
    X = np.array([[1j]])
    # 调用接口函数将数组 X 转换为线性操作对象 A
    A = interface.aslinearoperator(X)

    # 对线性操作对象 A 进行数乘
    B = 1j * A
    # 对原始数组 X 进行数乘
    Y = 1j * X

    # 创建一个包含单个元素的一维 NumPy 数组
    v = np.array([1])

    # 断言 B 与 Y 对向量 v 的乘积结果相等
    assert_equal(B.dot(v), Y.dot(v))
    # 断言 B 的转置乘向量 v 等于 Y 的转置乘向量 v
    assert_equal(B.T.dot(v), Y.T.dot(v))

# 定义一个测试函数，用于测试稀疏矩阵乘法中的异常情况
def test_sparse_matmat_exception():
    # 创建一个指定形状的线性操作对象 A，其 matvec 方法是恒等映射
    A = interface.LinearOperator((2, 2), matvec=lambda x: x)
    # 创建一个 2x2 的稀疏单位矩阵 B
    B = sparse.identity(2)
    # 定义用于异常断言的错误消息
    msg = "Unable to multiply a LinearOperator with a sparse matrix."
    # 断言在 A 与 B 相乘时抛出 TypeError 异常，并且异常消息匹配预期消息
    with assert_raises(TypeError, match=msg):
        A @ B
    # 断言在 B 与 A 相乘时抛出 TypeError 异常，并且异常消息匹配预期消息
    with assert_raises(TypeError, match=msg):
        B @ A
    # 断言在 A 与 4x4 的单位矩阵相乘时抛出 ValueError 异常
    with assert_raises(ValueError):
        A @ np.identity(4)
    # 断言在 4x4 的单位矩阵与 A 相乘时抛出 ValueError 异常
    with assert_raises(ValueError):
        np.identity(4) @ A
```