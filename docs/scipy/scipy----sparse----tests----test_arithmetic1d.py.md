# `D:\src\scipysrc\scipy\scipy\sparse\tests\test_arithmetic1d.py`

```
"""Test of 1D arithmetic operations"""

# 引入 pytest 库，用于测试框架
import pytest

# 引入 numpy 库，并从中引入必要的函数和模块
import numpy as np
from numpy.testing import assert_equal, assert_allclose

# 引入 scipy.sparse 库中的稀疏数组类型
from scipy.sparse import coo_array, csr_array
# 引入 scipy.sparse 库中的工具函数
from scipy.sparse._sputils import isscalarlike

# 定义一个列表，包含两种稀疏数组的创建函数
spcreators = [coo_array, csr_array]
# 定义一个列表，包含三种数学数据类型
math_dtypes = [np.int64, np.float64, np.complex128]

# 将输入转换为 ndarray 类型或者标量类型
def toarray(a):
    if isinstance(a, np.ndarray) or isscalarlike(a):
        return a
    return a.toarray()

# 定义一个 pytest 的 fixture，返回一个 1 维浮点数数组
@pytest.fixture
def dat1d():
    return np.array([3, 0, 1, 0], 'd')

# 定义一个 pytest 的 fixture，返回不同数据类型的稀疏数组和对应的数据
@pytest.fixture
def datsp_math_dtypes(dat1d):
    # 生成不同数据类型下的 dat1d 数组
    dat_dtypes = {dtype: dat1d.astype(dtype) for dtype in math_dtypes}
    # 返回字典，包含每种稀疏数组类型下不同数据类型的数据
    return {
        sp: [(dtype, dat, sp(dat)) for dtype, dat in dat_dtypes.items()]
        for sp in spcreators
    }

# 使用 pytest 的 parametrize 装饰器，对 spcreators 列表中的每个函数进行测试
@pytest.mark.parametrize("spcreator", spcreators)
class TestArithmetic1D:
    # 测试空数组的算术操作
    def test_empty_arithmetic(self, spcreator):
        shape = (5,)
        # 遍历不同的数据类型
        for mytype in [
            np.dtype('int32'),
            np.dtype('float32'),
            np.dtype('float64'),
            np.dtype('complex64'),
            np.dtype('complex128'),
        ]:
            # 创建稀疏数组 a，指定数据类型为 mytype
            a = spcreator(shape, dtype=mytype)
            # 执行加法操作
            b = a + a
            # 执行乘法操作
            c = 2 * a
            # 断言点乘结果为 ndarray 类型
            assert isinstance(a @ a.tocsr(), np.ndarray)
            assert isinstance(a @ a.tocoo(), np.ndarray)
            # 对 a、b、c 中的每个元素进行点乘运算，并断言结果与对应的 ndarray 的点乘结果一致
            for m in [a, b, c]:
                assert m @ m == a.toarray() @ a.toarray()
                # 断言 m 的数据类型与指定的 mytype 一致
                assert m.dtype == mytype
                # 断言将 m 转换为 ndarray 后的数据类型与指定的 mytype 一致
                assert toarray(m).dtype == mytype

    # 测试绝对值函数
    def test_abs(self, spcreator):
        A = np.array([-1, 0, 17, 0, -5, 0, 1, -4, 0, 0, 0, 0], 'd')
        # 断言稀疏数组的绝对值与对应 ndarray 的绝对值相等
        assert_equal(abs(A), abs(spcreator(A)).toarray())

    # 测试四舍五入函数
    def test_round(self, spcreator):
        A = np.array([-1.35, 0.56, 17.25, -5.98], 'd')
        Asp = spcreator(A)
        # 断言稀疏数组的四舍五入结果与对应 ndarray 的四舍五入结果相等
        assert_equal(np.around(A, decimals=1), round(Asp, ndigits=1).toarray())

    # 测试元素级幂函数
    def test_elementwise_power(self, spcreator):
        A = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4], 'd')
        Asp = spcreator(A)
        # 断言稀疏数组的元素级幂运算结果与对应 ndarray 的元素级幂运算结果相等
        assert_equal(np.power(A, 2), Asp.power(2).toarray())

        # 对于元素级幂函数，输入必须是标量
        with pytest.raises(NotImplementedError, match='input is not scalar'):
            spcreator(A).power(A)

    # 测试实部函数
    def test_real(self, spcreator):
        D = np.array([1 + 3j, 2 - 4j])
        A = spcreator(D)
        # 断言稀疏数组的实部与对应 ndarray 的实部相等
        assert_equal(A.real.toarray(), D.real)

    # 测试虚部函数
    def test_imag(self, spcreator):
        D = np.array([1 + 3j, 2 - 4j])
        A = spcreator(D)
        # 断言稀疏数组的虚部与对应 ndarray 的虚部相等
        assert_equal(A.imag.toarray(), D.imag)

    # 测试标量乘法
    def test_mul_scalar(self, spcreator, datsp_math_dtypes):
        for dtype, dat, datsp in datsp_math_dtypes[spcreator]:
            # 断言稀疏数组乘以标量的结果与对应 ndarray 乘以标量的结果相等
            assert_equal(dat * 2, (datsp * 2).toarray())
            assert_equal(dat * 17.3, (datsp * 17.3).toarray())

    # 测试右乘标量
    def test_rmul_scalar(self, spcreator, datsp_math_dtypes):
        for dtype, dat, datsp in datsp_math_dtypes[spcreator]:
            # 断言标量乘以稀疏数组的结果与标量乘以对应 ndarray 的结果相等
            assert_equal(2 * dat, (2 * datsp).toarray())
            assert_equal(17.3 * dat, (17.3 * datsp).toarray())
    # 测试稀疏矩阵的减法操作
    def test_sub(self, spcreator, datsp_math_dtypes):
        # 遍历给定的稀疏矩阵和数据对
        for dtype, dat, datsp in datsp_math_dtypes[spcreator]:
            if dtype == np.dtype('bool'):
                # 如果数据类型是布尔型，说明布尔数组减法在版本1.9.0中已弃用，跳过此次循环
                continue

            # 断言稀疏矩阵减去自身的结果等于全零数组
            assert_equal((datsp - datsp).toarray(), np.zeros(4))
            # 断言稀疏矩阵减去0的结果等于原始数据dat
            assert_equal((datsp - 0).toarray(), dat)

            # 创建一个稀疏矩阵A，包含给定的数据并指定数据类型为双精度浮点型
            A = spcreator([1, -4, 0, 2], dtype='d')
            # 断言稀疏矩阵减去A的结果等于原始数据dat减去A的稀疏矩阵表示的结果
            assert_equal((datsp - A).toarray(), dat - A.toarray())
            # 断言A减去稀疏矩阵datsp的结果等于A的稀疏矩阵表示减去原始数据dat的结果
            assert_equal((A - datsp).toarray(), A.toarray() - dat)

            # 测试广播特性，将稀疏矩阵datsp转换为数组后减去dat中的第一个元素
            assert_equal(datsp.toarray() - dat[0], dat - dat[0])

    # 测试稀疏矩阵加0的操作
    def test_add0(self, spcreator, datsp_math_dtypes):
        # 遍历给定的稀疏矩阵和数据对
        for dtype, dat, datsp in datsp_math_dtypes[spcreator]:
            # 将0加到稀疏矩阵datsp上，断言结果等于原始数据dat
            assert_equal((datsp + 0).toarray(), dat)
            # 使用sum函数对稀疏矩阵datsp进行乘积求和，断言结果与相应的密集数据dat的乘积求和结果接近
            sumS = sum([k * datsp for k in range(1, 3)])
            sumD = sum([k * dat for k in range(1, 3)])
            assert_allclose(sumS.toarray(), sumD)

    # 测试稀疏矩阵的逐元素乘法
    def test_elementwise_multiply(self, spcreator):
        # 测试实数与实数的逐元素乘法
        A = np.array([4, 0, 9])
        B = np.array([0, 7, -1])
        Asp = spcreator(A)
        Bsp = spcreator(B)
        assert_allclose(Asp.multiply(Bsp).toarray(), A * B)  # 稀疏矩阵与稀疏矩阵的逐元素乘法
        assert_allclose(Asp.multiply(B).toarray(), A * B)  # 稀疏矩阵与密集数组的逐元素乘法

        # 测试复数与复数的逐元素乘法
        C = np.array([1 - 2j, 0 + 5j, -1 + 0j])
        D = np.array([5 + 2j, 7 - 3j, -2 + 1j])
        Csp = spcreator(C)
        Dsp = spcreator(D)
        assert_allclose(Csp.multiply(Dsp).toarray(), C * D)  # 稀疏矩阵与稀疏矩阵的逐元素乘法
        assert_allclose(Csp.multiply(D).toarray(), C * D)  # 稀疏矩阵与密集数组的逐元素乘法

        # 测试实数与复数的逐元素乘法
        assert_allclose(Asp.multiply(Dsp).toarray(), A * D)  # 稀疏矩阵与稀疏矩阵的逐元素乘法
        assert_allclose(Asp.multiply(D).toarray(), A * D)  # 稀疏矩阵与密集数组的逐元素乘法
    # 定义一个测试方法，用于测试元素级乘法的广播功能
    def test_elementwise_multiply_broadcast(self, spcreator):
        # 创建一维数组 A，B，C，D，E，F，G，H，J，K，L
        A = np.array([4])
        B = np.array([[-9]])
        C = np.array([1, -1, 0])
        D = np.array([[7, 9, -9]])
        E = np.array([[3], [2], [1]])
        F = np.array([[8, 6, 3], [-4, 3, 2], [6, 6, 6]])
        G = [1, 2, 3]
        H = np.ones((3, 4))
        J = H.T
        K = np.array([[0]])
        L = np.array([[[1, 2], [0, 1]]])

        # 对于无法转换为稀疏矩阵的数组（A，C，L），不进行处理
        Asp = spcreator(A)
        Csp = spcreator(C)
        Gsp = spcreator(G)

        # 创建二维数组的稀疏矩阵
        Bsp = spcreator(B)
        Dsp = spcreator(D)
        Esp = spcreator(E)
        Fsp = spcreator(F)
        Hsp = spcreator(H)
        Hspp = spcreator(H[0, None])
        Jsp = spcreator(J)
        Jspp = spcreator(J[:, 0, None])
        Ksp = spcreator(K)

        # 创建包含所有数组的列表
        matrices = [A, B, C, D, E, F, G, H, J, K, L]
        # 创建包含所有稀疏矩阵的列表
        spmatrices = [Asp, Bsp, Csp, Dsp, Esp, Fsp, Gsp, Hsp, Hspp, Jsp, Jspp, Ksp]
        # 创建包含一维稀疏矩阵的列表
        sp1dmatrices = [Asp, Csp, Gsp]

        # 对于稀疏/稀疏矩阵的乘法
        for i in sp1dmatrices:
            for j in spmatrices:
                try:
                    # 尝试计算稀疏矩阵的密集乘积
                    dense_mult = i.toarray() * j.toarray()
                except ValueError:
                    # 如果形状不一致，预期引发 ValueError 异常
                    with pytest.raises(ValueError, match='inconsistent shapes'):
                        i.multiply(j)
                    continue
                # 计算稀疏矩阵的乘积
                sp_mult = i.multiply(j)
                # 使用 assert_allclose 检查稀疏乘积的结果是否接近于密集乘积
                assert_allclose(sp_mult.toarray(), dense_mult)

        # 对于稀疏/密集矩阵的乘法
        for i in sp1dmatrices:
            for j in matrices:
                try:
                    # 尝试计算稀疏矩阵的密集乘积
                    dense_mult = i.toarray() * j
                except TypeError:
                    continue
                except ValueError:
                    # 如果形状不一致，预期引发 ValueError 异常
                    matchme = 'broadcast together|inconsistent shapes'
                    with pytest.raises(ValueError, match=matchme):
                        i.multiply(j)
                    continue
                # 计算稀疏矩阵的乘积
                sp_mult = i.multiply(j)
                # 使用 assert_allclose 检查稀疏乘积的结果是否接近于密集乘积
                assert_allclose(toarray(sp_mult), dense_mult)
    # 测试元素级除法功能
    def test_elementwise_divide(self, spcreator, dat1d):
        # 使用提供的 spcreator 函数创建稀疏数组 datsp
        datsp = spcreator(dat1d)
        # 预期的结果数组，包含了 NaN 值
        expected = np.array([1, np.nan, 1, np.nan])
        # 对稀疏数组进行元素级除法操作
        actual = datsp / datsp
        # 使用 assert_array_equal 进行比较，处理 NaN 值
        np.testing.assert_array_equal(actual, expected)

        # 使用不同的分母 denom 进行除法操作
        denom = spcreator([1, 0, 0, 4], dtype='d')
        expected = [3, np.nan, np.inf, 0]
        np.testing.assert_array_equal(datsp / denom, expected)

        # 复数情况下的除法操作
        A = np.array([1 - 2j, 0 + 5j, -1 + 0j])
        B = np.array([5 + 2j, 7 - 3j, -2 + 1j])
        Asp = spcreator(A)
        Bsp = spcreator(B)
        # 使用 assert_allclose 检查稀疏数组除法结果与普通数组除法结果的近似性
        assert_allclose(Asp / Bsp, A / B)

        # 整数情况下的除法操作
        A = np.array([1, 2, 3])
        B = np.array([0, 1, 2])
        Asp = spcreator(A)
        Bsp = spcreator(B)
        # 忽略除法时的除以零警告
        with np.errstate(divide='ignore'):
            assert_equal(Asp / Bsp, A / B)

        # 不匹配稀疏模式的除法操作
        A = np.array([0, 1])
        B = np.array([1, 0])
        Asp = spcreator(A)
        Bsp = spcreator(B)
        # 忽略除法和无效值警告
        with np.errstate(divide='ignore', invalid='ignore'):
            assert_equal(Asp / Bsp, A / B)
    def test_matmul(self, spcreator):
        # 使用 spcreator 函数创建稀疏矩阵 Msp，参数为 [2, 0, 3.0]
        Msp = spcreator([2, 0, 3.0])
        # 使用 spcreator 函数创建稀疏矩阵 B，参数为包含数组 [[0, 1], [1, 0], [0, 2]] 的 np.array，数据类型为双精度浮点数 ('d')
        B = spcreator(np.array([[0, 1], [1, 0], [0, 2]], 'd'))
        # 创建列向量 col，数据为 [[1, 2, 3]] 的转置
        col = np.array([[1, 2, 3]]).T

        # 检查稀疏矩阵 Msp 与稠密列向量 col 的矩阵乘法结果是否接近
        assert_allclose(Msp @ col, Msp.toarray() @ col)

        # 检查稀疏矩阵 Msp 与稀疏矩阵 B 的矩阵乘法结果是否接近，以及稀疏矩阵 Msp 与稠密矩阵 B 的矩阵乘法结果是否接近
        assert_allclose((Msp @ B).toarray(), (Msp @ B).toarray())
        assert_allclose(Msp.toarray() @ B, (Msp @ B).toarray())
        assert_allclose(Msp @ B.toarray(), (Msp @ B).toarray())

        # 检查稀疏矩阵 Msp 与稠密向量 V 的乘法结果是否接近，以及稀疏矩阵 Msp 与稀疏向量 Vsp 的乘法结果是否接近
        V = np.array([0, 0, 1])
        assert_allclose(Msp @ V, Msp.toarray() @ V)

        Vsp = spcreator(V)
        # 对稀疏矩阵 Msp 与稀疏向量 Vsp 进行乘法运算，并检查结果类型是否为 np.ndarray
        Msp_Vsp = Msp @ Vsp
        assert isinstance(Msp_Vsp, np.ndarray)
        assert Msp_Vsp.shape == ()

        # 检查输出结果为零维 np.ndarray
        assert_allclose(np.array(3), Msp_Vsp)
        assert_allclose(np.array(3), Msp.toarray() @ Vsp)
        assert_allclose(np.array(3), Msp @ Vsp.toarray())
        assert_allclose(np.array(3), Msp.toarray() @ Vsp.toarray())

        # 检查矩阵与标量相乘时是否引发错误
        with pytest.raises(ValueError, match='Scalar operands are not allowed'):
            Msp @ 1
        with pytest.raises(ValueError, match='Scalar operands are not allowed'):
            1 @ Msp
    def test_size_zero_matrix_arithmetic(self, spcreator):
        # 测试对形状为 0、(1, 0)、(0, 3) 等的基本矩阵算术操作

        # 创建一个空的 NumPy 数组
        mat = np.array([])
        # 将数组重塑为一维数组，长度为 0
        a = mat.reshape(0)
        # 将数组重塑为二维数组，形状为 (1, 0)
        d = mat.reshape((1, 0))
        # 创建一个 5x5 全 1 矩阵
        f = np.ones([5, 5])

        # 使用 spcreator 函数创建稀疏矩阵对象 asp 和 dsp
        asp = spcreator(a)
        dsp = spcreator(d)

        # 测试加法操作，预期会引发 ValueError 异常，异常信息包含 'inconsistent shapes'
        with pytest.raises(ValueError, match='inconsistent shapes'):
            asp.__add__(dsp)

        # 测试矩阵乘积操作，验证稀疏矩阵 asp 的 dot 方法结果是否等于对应的 NumPy 数组 a 的 dot 乘积结果
        assert_equal(asp.dot(asp), np.dot(a, a))

        # 测试矩阵乘积操作，预期会引发 ValueError 异常，异常信息包含 'dimension mismatch'
        with pytest.raises(ValueError, match='dimension mismatch'):
            asp.dot(f)

        # 测试元素级乘法操作，验证稀疏矩阵 asp 的 multiply 方法结果是否等于对应的 NumPy 数组 a 的元素级乘积结果
        assert_equal(asp.multiply(asp).toarray(), np.multiply(a, a))

        # 测试元素级乘法操作，验证稀疏矩阵 asp 与数组 a 的元素级乘积结果是否相等
        assert_equal(asp.multiply(a).toarray(), np.multiply(a, a))

        # 测试元素级乘法操作，验证稀疏矩阵 asp 与标量值 6 的元素级乘积结果是否等于对应的 NumPy 数组 a 与 6 的元素级乘积结果
        assert_equal(asp.multiply(6).toarray(), np.multiply(a, 6))

        # 测试元素级乘法操作，预期会引发 ValueError 异常，异常信息包含 'inconsistent shapes'
        with pytest.raises(ValueError, match='inconsistent shapes'):
            asp.multiply(f)

        # 测试加法操作，验证稀疏矩阵 asp 的 __add__ 方法结果是否等于对应的 NumPy 数组 a 的加法结果
        assert_equal(asp.__add__(asp).toarray(), a.__add__(a))
```