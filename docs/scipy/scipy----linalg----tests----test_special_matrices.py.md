# `D:\src\scipysrc\scipy\scipy\linalg\tests\test_special_matrices.py`

```
import pytest  # 导入 pytest 测试框架
import numpy as np  # 导入 NumPy 库并使用别名 np
from numpy import arange, array, eye, copy, sqrt  # 导入特定函数和对象
from numpy.testing import (assert_equal, assert_array_equal,  # 导入测试函数
                           assert_array_almost_equal, assert_allclose)
from pytest import raises as assert_raises  # 导入 pytest 的 raises 函数并使用别名 assert_raises

from scipy.fft import fft  # 导入 scipy 的 FFT 函数
from scipy.special import comb  # 导入 scipy 的组合函数
from scipy.linalg import (toeplitz, hankel, circulant, hadamard, leslie, dft,  # 导入 scipy 的线性代数函数
                          companion, kron, block_diag,
                          helmert, hilbert, invhilbert, pascal, invpascal,
                          fiedler, fiedler_companion, eigvals,
                          convolution_matrix)
from numpy.linalg import cond  # 导入 NumPy 的条件数计算函数


class TestToeplitz:  # 定义测试类 TestToeplitz

    def test_basic(self):  # 定义测试方法 test_basic
        y = toeplitz([1, 2, 3])  # 调用 toeplitz 函数，生成 Toeplitz 矩阵
        assert_array_equal(y, [[1, 2, 3], [2, 1, 2], [3, 2, 1]])  # 断言结果与预期相等
        y = toeplitz([1, 2, 3], [1, 4, 5])  # 使用指定的第一列和第一行参数创建 Toeplitz 矩阵
        assert_array_equal(y, [[1, 4, 5], [2, 1, 4], [3, 2, 1]])  # 断言结果与预期相等

    def test_complex_01(self):  # 定义测试方法 test_complex_01
        data = (1.0 + arange(3.0)) * (1.0 + 1.0j)  # 创建复数数组
        x = copy(data)  # 复制数组 data 到 x
        t = toeplitz(x)  # 创建复数 Toeplitz 矩阵
        # 调用 toeplitz 不应该改变 x 的值。
        assert_array_equal(x, data)  # 断言 x 的值与 data 相等
        # 根据文档字符串，x 应该是矩阵 t 的第一列。
        col0 = t[:, 0]  # 获取矩阵 t 的第一列
        assert_array_equal(col0, data)  # 断言矩阵 t 的第一列与 data 相等
        assert_array_equal(t[0, 1:], data[1:].conj())  # 断言矩阵 t 的第一行的其余部分与 data 的共轭相等

    def test_scalar_00(self):  # 定义测试方法 test_scalar_00
        """Scalar arguments still produce a 2D array."""
        t = toeplitz(10)  # 使用标量参数创建 Toeplitz 矩阵
        assert_array_equal(t, [[10]])  # 断言结果为预期的二维数组
        t = toeplitz(10, 20)  # 使用两个标量参数创建 Toeplitz 矩阵
        assert_array_equal(t, [[10]])  # 断言结果为预期的二维数组

    def test_scalar_01(self):  # 定义测试方法 test_scalar_01
        c = array([1, 2, 3])  # 创建数组 c
        t = toeplitz(c, 1)  # 使用数组 c 和标量参数创建 Toeplitz 矩阵
        assert_array_equal(t, [[1], [2], [3]])  # 断言结果与预期相等

    def test_scalar_02(self):  # 定义测试方法 test_scalar_02
        c = array([1, 2, 3])  # 创建数组 c
        t = toeplitz(c, array(1))  # 使用数组 c 和数组参数创建 Toeplitz 矩阵
        assert_array_equal(t, [[1], [2], [3]])  # 断言结果与预期相等

    def test_scalar_03(self):  # 定义测试方法 test_scalar_03
        c = array([1, 2, 3])  # 创建数组 c
        t = toeplitz(c, array([1]))  # 使用数组 c 和数组参数创建 Toeplitz 矩阵
        assert_array_equal(t, [[1], [2], [3]])  # 断言结果与预期相等

    def test_scalar_04(self):  # 定义测试方法 test_scalar_04
        r = array([10, 2, 3])  # 创建数组 r
        t = toeplitz(1, r)  # 使用标量参数和数组参数创建 Toeplitz 矩阵
        assert_array_equal(t, [[1, 2, 3]])  # 断言结果与预期相等


class TestHankel:  # 定义测试类 TestHankel

    def test_basic(self):  # 定义测试方法 test_basic
        y = hankel([1, 2, 3])  # 创建 Hankel 矩阵
        assert_array_equal(y, [[1, 2, 3], [2, 3, 0], [3, 0, 0]])  # 断言结果与预期相等
        y = hankel([1, 2, 3], [3, 4, 5])  # 使用指定的第一列和第一行参数创建 Hankel 矩阵
        assert_array_equal(y, [[1, 2, 3], [2, 3, 4], [3, 4, 5]])  # 断言结果与预期相等


class TestCirculant:  # 定义测试类 TestCirculant

    def test_basic(self):  # 定义测试方法 test_basic
        y = circulant([1, 2, 3])  # 创建循环矩阵
        assert_array_equal(y, [[1, 3, 2], [2, 1, 3], [3, 2, 1]])  # 断言结果与预期相等


class TestHadamard:  # 定义测试类 TestHadamard

    def test_basic(self):  # 定义测试方法 test_basic

        y = hadamard(1)  # 创建 Hadamard 矩阵
        assert_array_equal(y, [[1]])  # 断言结果与预期相等

        y = hadamard(2, dtype=float)  # 创建 Hadamard 矩阵，指定数据类型为浮点数
        assert_array_equal(y, [[1.0, 1.0], [1.0, -1.0]])  # 断言结果与预期相等

        y = hadamard(4)  # 创建 Hadamard 矩阵
        assert_array_equal(y, [[1, 1, 1, 1],
                               [1, -1, 1, -1],
                               [1, 1, -1, -1],
                               [1, -1, -1, 1]])  # 断言结果与预期相等

        assert_raises(ValueError, hadamard, 0)  # 断言调用 hadamard(0) 会抛出 ValueError
        assert_raises(ValueError, hadamard, 5)  # 断言调用 hadamard(5) 会抛出 ValueError
class TestLeslie:
    def test_bad_shapes(self):
        # 检查传入的 Leslie 矩阵和向量是否符合形状要求，应该引发 ValueError 异常
        assert_raises(ValueError, leslie, [[1, 1], [2, 2]], [3, 4, 5])
        assert_raises(ValueError, leslie, [3, 4, 5], [[1, 1], [2, 2]])
        assert_raises(ValueError, leslie, [1, 2], [1, 2])
        assert_raises(ValueError, leslie, [1], [])

    def test_basic(self):
        # 测试基本的 Leslie 矩阵生成功能
        a = leslie([1, 2, 3], [0.25, 0.5])
        expected = array([[1.0, 2.0, 3.0],
                          [0.25, 0.0, 0.0],
                          [0.0, 0.5, 0.0]])
        assert_array_equal(a, expected)


class TestCompanion:
    def test_bad_shapes(self):
        # 检查传入的 Companion 矩阵是否符合形状要求，应该引发 ValueError 异常
        assert_raises(ValueError, companion, [[1, 1], [2, 2]])
        assert_raises(ValueError, companion, [0, 4, 5])
        assert_raises(ValueError, companion, [1])
        assert_raises(ValueError, companion, [])

    def test_basic(self):
        # 测试基本的 Companion 矩阵生成功能
        c = companion([1, 2, 3])
        expected = array([
            [-2.0, -3.0],
            [1.0, 0.0]])
        assert_array_equal(c, expected)

        c = companion([2.0, 5.0, -10.0])
        expected = array([
            [-2.5, 5.0],
            [1.0, 0.0]])
        assert_array_equal(c, expected)


class TestBlockDiag:
    def test_basic(self):
        # 测试基本的 block_diag 函数，生成一个块对角矩阵
        x = block_diag(eye(2), [[1, 2], [3, 4], [5, 6]], [[1, 2, 3]])
        assert_array_equal(x, [[1, 0, 0, 0, 0, 0, 0],
                               [0, 1, 0, 0, 0, 0, 0],
                               [0, 0, 1, 2, 0, 0, 0],
                               [0, 0, 3, 4, 0, 0, 0],
                               [0, 0, 5, 6, 0, 0, 0],
                               [0, 0, 0, 0, 1, 2, 3]])

    def test_dtype(self):
        # 测试 block_diag 函数对不同类型数据的处理，期望生成的矩阵数据类型正确
        x = block_diag([[1.5]])
        assert_equal(x.dtype, float)

        x = block_diag([[True]])
        assert_equal(x.dtype, bool)

    def test_mixed_dtypes(self):
        # 测试 block_diag 函数对混合数据类型的处理
        actual = block_diag([[1]], [[1j]])
        desired = np.array([[1, 0], [0, 1j]])
        assert_array_equal(actual, desired)

    def test_scalar_and_1d_args(self):
        # 测试 block_diag 函数对标量和一维数组的处理
        a = block_diag(1)
        assert_equal(a.shape, (1, 1))
        assert_array_equal(a, [[1]])

        a = block_diag([2, 3], 4)
        assert_array_equal(a, [[2, 3, 0], [0, 0, 4]])

    def test_bad_arg(self):
        # 测试 block_diag 函数对不符合要求的参数的处理，应该引发 ValueError 异常
        assert_raises(ValueError, block_diag, [[[1]]])

    def test_no_args(self):
        # 测试 block_diag 函数在没有输入参数时的行为，期望生成一个空矩阵
        a = block_diag()
        assert_equal(a.ndim, 2)
        assert_equal(a.nbytes, 0)

    def test_empty_matrix_arg(self):
        # regression test for gh-4596: 检查 block_diag 函数对空矩阵输入的处理
        # 当存在空矩阵作为参数时，不再忽略，而是视为形状为 (1, 0) 的矩阵
        a = block_diag([[1, 0], [0, 1]],
                       [],
                       [[2, 3], [4, 5], [6, 7]])
        assert_array_equal(a, [[1, 0, 0, 0],
                               [0, 1, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 2, 3],
                               [0, 0, 4, 5],
                               [0, 0, 6, 7]])
    def test_zerosized_matrix_arg(self):
        # 测试用例：检查对于零大小矩阵输入的结果形状，即形状为 (0, n) 或 (n, 0) 的矩阵。
        # 注意：[[]] 的形状为 (1, 0)
        # 创建一个对角矩阵 a，包含以下块矩阵：
        a = block_diag([[1, 0], [0, 1]],        # 2x2 矩阵，左上角块
                       [[]],                    # 空矩阵，1x0 形状
                       [[2, 3], [4, 5], [6, 7]], # 3x2 矩阵，右上角块
                       np.zeros([0, 2], dtype='int32'))  # 形状为 (0, 2) 的零矩阵
        # 断言矩阵 a 的结果与预期相等
        assert_array_equal(a, [[1, 0, 0, 0, 0, 0],
                               [0, 1, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0],
                               [0, 0, 2, 3, 0, 0],
                               [0, 0, 4, 5, 0, 0],
                               [0, 0, 6, 7, 0, 0]])
class TestKron:

    def test_basic(self):
        # 使用 kron 函数计算两个数组的 Kronecker 乘积，并验证结果是否与期望值相等
        a = kron(array([[1, 2], [3, 4]]), array([[1, 1, 1]]))
        assert_array_equal(a, array([[1, 1, 1, 2, 2, 2],
                                     [3, 3, 3, 4, 4, 4]]))

        # 定义两个矩阵 m1 和 m2
        m1 = array([[1, 2], [3, 4]])
        m2 = array([[10], [11]])
        # 计算它们的 Kronecker 乘积，并验证结果是否与期望值相等
        a = kron(m1, m2)
        expected = array([[10, 20],
                          [11, 22],
                          [30, 40],
                          [33, 44]])
        assert_array_equal(a, expected)

    def test_empty(self):
        # 创建一个空的 numpy 数组 m1 和 m2
        m1 = np.empty((0, 2))
        m2 = np.empty((1, 3))
        # 计算它们的 Kronecker 乘积，并验证结果是否与预期的空数组相等
        a = kron(m1, m2)
        assert_allclose(a, np.empty((0, 6)))


class TestHelmert:

    def test_orthogonality(self):
        # 对于从 1 到 6 的每个数字 n
        for n in range(1, 7):
            # 生成 Helmert 矩阵 H，并验证其与单位矩阵 Id 的转置乘积是否在指定的误差范围内
            H = helmert(n, full=True)
            Id = np.eye(n)
            assert_allclose(H.dot(H.T), Id, atol=1e-12)
            assert_allclose(H.T.dot(H), Id, atol=1e-12)

    def test_subspace(self):
        # 对于从 2 到 6 的每个数字 n
        for n in range(2, 7):
            # 生成完整 Helmert 矩阵 H_full 和部分 Helmert 矩阵 H_partial
            H_full = helmert(n, full=True)
            H_partial = helmert(n)
            # 对于 H_full 和 H_partial 的每列 U
            for U in H_full[1:, :].T, H_partial.T:
                # 计算压缩矩阵 C，并验证 U 乘以其转置是否在指定的误差范围内等于 C
                C = np.eye(n) - np.full((n, n), 1 / n)
                assert_allclose(U.dot(U.T), C)
                assert_allclose(U.T.dot(U), np.eye(n-1), atol=1e-12)


class TestHilbert:

    def test_basic(self):
        # 验证 Hilbert 矩阵的生成是否与预期的精度相等
        h3 = array([[1.0, 1/2., 1/3.],
                    [1/2., 1/3., 1/4.],
                    [1/3., 1/4., 1/5.]])
        assert_array_almost_equal(hilbert(3), h3)

        # 验证生成单元素 Hilbert 矩阵是否与预期相等
        assert_array_equal(hilbert(1), [[1.0]])

        # 生成零维 Hilbert 矩阵，并验证其形状是否符合预期
        h0 = hilbert(0)
        assert_equal(h0.shape, (0, 0))


class TestInvHilbert:

    def test_inverse(self):
        # 对于从 1 到 9 的每个数字 n
        for n in range(1, 10):
            # 生成 Hilbert 矩阵 a 和其逆矩阵 b
            a = hilbert(n)
            b = invhilbert(n)
            # 计算条件数 c，并验证 a 乘以 b 是否接近单位矩阵，考虑到 Hilbert 矩阵逐渐恶化的条件
            c = cond(a)
            assert_allclose(a.dot(b), eye(n), atol=1e-15*c, rtol=1e-15*c)


class TestPascal:

    cases = [
        (1, array([[1]]), array([[1]])),
        (2, array([[1, 1],
                   [1, 2]]),
            array([[1, 0],
                   [1, 1]])),
        (3, array([[1, 1, 1],
                   [1, 2, 3],
                   [1, 3, 6]]),
            array([[1, 0, 0],
                   [1, 1, 0],
                   [1, 2, 1]])),
        (4, array([[1, 1, 1, 1],
                   [1, 2, 3, 4],
                   [1, 3, 6, 10],
                   [1, 4, 10, 20]]),
            array([[1, 0, 0, 0],
                   [1, 1, 0, 0],
                   [1, 2, 1, 0],
                   [1, 3, 3, 1]])),
    ]
    # 定义一个方法，用于检查生成的帕斯卡三角形是否符合预期
    def check_case(self, n, sym, low):
        # 断言生成的帕斯卡三角形与预期的对称版本相等
        assert_array_equal(pascal(n), sym)
        # 断言生成的帕斯卡三角形与预期的下三角形版本相等
        assert_array_equal(pascal(n, kind='lower'), low)
        # 断言生成的帕斯卡三角形与预期的上三角形版本相等
        assert_array_equal(pascal(n, kind='upper'), low.T)
        # 断言生成的帕斯卡三角形（近似值）与预期的对称版本相等
        assert_array_almost_equal(pascal(n, exact=False), sym)
        # 断言生成的帕斯卡三角形（近似值）与预期的下三角形版本相等
        assert_array_almost_equal(pascal(n, exact=False, kind='lower'), low)
        # 断言生成的帕斯卡三角形（近似值）与预期的上三角形版本相等
        assert_array_almost_equal(pascal(n, exact=False, kind='upper'), low.T)

    # 定义一个方法，用于测试多组帕斯卡三角形的生成情况
    def test_cases(self):
        # 对于每组测试用例，执行帕斯卡三角形生成检查
        for n, sym, low in self.cases:
            self.check_case(n, sym, low)

    # 定义一个方法，用于测试生成较大规模的帕斯卡三角形
    def test_big(self):
        # 生成一个规模为 50 的帕斯卡三角形
        p = pascal(50)
        # 断言帕斯卡三角形最后一个元素的值与组合数的计算结果相等
        assert p[-1, -1] == comb(98, 49, exact=True)

    # 定义一个方法，用于测试帕斯卡三角形在边界情况下的行为
    def test_threshold(self):
        # 回归测试：早期版本的 `pascal` 函数在 n=35 时返回 np.uint64 类型数组，
        # 但该数据类型无法容纳 p[-1, -1] 的值。第二个 assert_equal 断言会失败，
        # 因为 p[-1, -1] 溢出了。
        p = pascal(34)
        assert_equal(2*p.item(-1, -2), p.item(-1, -1), err_msg="n = 34")
        p = pascal(35)
        assert_equal(2.*p.item(-1, -2), 1.*p.item(-1, -1), err_msg="n = 35")
def test_invpascal():
    # 定义内部函数，用于验证 invpascal 函数的输出结果
    def check_invpascal(n, kind, exact):
        # 调用 invpascal 函数，获取其返回值 ip
        ip = invpascal(n, kind=kind, exact=exact)
        # 调用 pascal 函数，获取其返回值 p
        p = pascal(n, kind=kind, exact=exact)
        
        # 矩阵相乘 ip 和 p，验证是否得到单位矩阵
        # 由于 dtype 不同可能会导致精度问题，使用对象数组进行乘法运算
        e = ip.astype(object).dot(p.astype(object))
        
        # 断言 e 和单位矩阵相等，若不等则输出错误信息
        assert_array_equal(e, eye(n), err_msg="n=%d  kind=%r exact=%r" % (n, kind, exact))

    # 定义三种不同的 Pascal 矩阵类型
    kinds = ['symmetric', 'lower', 'upper']

    # 对不同的 n 值进行测试
    ns = [1, 2, 5, 18]
    for n in ns:
        for kind in kinds:
            for exact in [True, False]:
                # 调用 check_invpascal 函数进行测试
                check_invpascal(n, kind, exact)

    # 对较大的 n 值进行额外测试
    ns = [19, 34, 35, 50]
    for n in ns:
        for kind in kinds:
            # 仅测试 exact=True 的情况
            check_invpascal(n, kind, True)


def test_dft():
    # 测试 DFT 函数对于 n=2 的情况
    m = dft(2)
    expected = array([[1.0, 1.0], [1.0, -1.0]])
    assert_array_almost_equal(m, expected)
    
    # 测试 DFT 函数对于 n=2 和 scale='n' 的情况
    m = dft(2, scale='n')
    assert_array_almost_equal(m, expected/2.0)
    
    # 测试 DFT 函数对于 n=2 和 scale='sqrtn' 的情况
    m = dft(2, scale='sqrtn')
    assert_array_almost_equal(m, expected/sqrt(2.0))

    # 测试 DFT 函数对于向量 x 的结果
    x = array([0, 1, 2, 3, 4, 5, 0, 1])
    m = dft(8)
    mx = m.dot(x)
    fx = fft(x)
    assert_array_almost_equal(mx, fx)


def test_fiedler():
    # 测试空列表情况下的 fiedler 函数输出
    f = fiedler([])
    assert_equal(f.size, 0)
    
    # 测试包含单个元素列表情况下的 fiedler 函数输出
    f = fiedler([123.])
    assert_array_equal(f, np.array([[0.]]))
    
    # 测试一般情况下的 fiedler 函数输出
    f = fiedler(np.arange(1, 7))
    des = np.array([[0, 1, 2, 3, 4, 5],
                    [1, 0, 1, 2, 3, 4],
                    [2, 1, 0, 1, 2, 3],
                    [3, 2, 1, 0, 1, 2],
                    [4, 3, 2, 1, 0, 1],
                    [5, 4, 3, 2, 1, 0]])
    assert_array_equal(f, des)


def test_fiedler_companion():
    # 测试空列表情况下的 fiedler_companion 函数输出
    fc = fiedler_companion([])
    assert_equal(fc.size, 0)
    
    # 测试只有一个元素的列表情况下的 fiedler_companion 函数输出
    fc = fiedler_companion([1.])
    assert_equal(fc.size, 0)
    
    # 测试包含多个元素的列表情况下的 fiedler_companion 函数输出
    fc = fiedler_companion([1., 2.])
    assert_array_equal(fc, np.array([[-2.]]))
    
    # 测试异常情况下的 fiedler_companion 函数输出
    fc = fiedler_companion([1e-12, 2., 3.])
    assert_array_almost_equal(fc, companion([1e-12, 2., 3.]))
    with assert_raises(ValueError):
        fiedler_companion([0, 1, 2])
    
    # 测试特定情况下的 fiedler_companion 函数输出
    fc = fiedler_companion([1., -16., 86., -176., 105.])
    assert_array_almost_equal(eigvals(fc),
                              np.array([7., 5., 3., 1.]))


class TestConvolutionMatrix:
    """
    Test convolution_matrix vs. numpy.convolve for various parameters.
    """

    def create_vector(self, n, cpx):
        """Make a complex or real test vector of length n."""
        # 创建一个长度为 n 的复数或实数测试向量
        x = np.linspace(-2.5, 2.2, n)
        if cpx:
            x = x + 1j*np.linspace(-1.5, 3.1, n)
        return x
    # 测试函数，用于检查当 n 不是正整数时是否引发 ValueError 异常
    def test_bad_n(self):
        # n must be a positive integer
        with pytest.raises(ValueError, match='n must be a positive integer'):
            convolution_matrix([1, 2, 3], 0)

    # 测试函数，用于检查当第一个参数不是一维数组时是否引发 ValueError 异常
    def test_bad_first_arg(self):
        # first arg must be a 1d array, otherwise ValueError
        with pytest.raises(ValueError, match='one-dimensional'):
            convolution_matrix(1, 4)

    # 测试函数，用于检查当第一个参数为空数组时是否引发 ValueError 异常
    def test_empty_first_arg(self):
        # first arg must have at least one value
        with pytest.raises(ValueError, match=r'len\(a\)'):
            convolution_matrix([], 4)

    # 测试函数，用于检查当模式参数不在 ('full', 'valid', 'same') 中时是否引发 ValueError 异常
    def test_bad_mode(self):
        # mode must be in ('full', 'valid', 'same')
        with pytest.raises(ValueError, match='mode.*must be one of'):
            convolution_matrix((1, 1), 4, mode='invalid argument')

    # 使用参数化测试，与 NumPy 的 convolve 函数对比结果
    @pytest.mark.parametrize('cpx', [False, True])
    @pytest.mark.parametrize('na', [1, 2, 9])
    @pytest.mark.parametrize('nv', [1, 2, 9])
    @pytest.mark.parametrize('mode', [None, 'full', 'valid', 'same'])
    def test_against_numpy_convolve(self, cpx, na, nv, mode):
        # 创建长度为 na 的复数或实数向量 a
        a = self.create_vector(na, cpx)
        # 创建长度为 nv 的复数或实数向量 v
        v = self.create_vector(nv, cpx)
        
        # 如果 mode 为 None，则使用默认的完整模式进行卷积运算
        if mode is None:
            y1 = np.convolve(v, a)
            A = convolution_matrix(a, nv)
        else:
            # 使用指定的模式进行卷积运算
            y1 = np.convolve(v, a, mode)
            A = convolution_matrix(a, nv, mode)
        
        # 计算 A @ v 的结果
        y2 = A @ v
        
        # 断言 y1 和 y2 的值近似相等
        assert_array_almost_equal(y1, y2)
```