# `D:\src\scipysrc\scipy\scipy\sparse\tests\test_common1d.py`

```
"""Test of 1D aspects of sparse array classes"""

# 导入必要的库和模块
import pytest  # 导入pytest库用于测试

import numpy as np  # 导入NumPy库，并用np作为别名
from numpy.testing import assert_equal, assert_allclose  # 导入NumPy测试模块中的断言函数

# 导入稀疏矩阵相关类
from scipy.sparse import (
        bsr_array, csc_array, dia_array, lil_array,
        coo_array, csr_array, dok_array,
    )

# 导入稀疏矩阵相关函数和工具
from scipy.sparse._sputils import supported_dtypes, matrix
from scipy._lib._util import ComplexWarning

# 忽略复数警告
sup_complex = np.testing.suppress_warnings()
sup_complex.filter(ComplexWarning)

# 定义稀疏矩阵的创建函数列表和数学数据类型列表
spcreators = [coo_array, csr_array, dok_array]
math_dtypes = [np.int64, np.float64, np.complex128]


@pytest.fixture
def dat1d():
    """返回一个1维的NumPy数组作为测试数据"""
    return np.array([3, 0, 1, 0], 'd')


@pytest.fixture
def datsp_math_dtypes(dat1d):
    """返回各种数学数据类型下的1维数据的稀疏矩阵表示"""
    dat_dtypes = {dtype: dat1d.astype(dtype) for dtype in math_dtypes}
    return {
        spcreator: [(dtype, dat, spcreator(dat)) for dtype, dat in dat_dtypes.items()]
        for spcreator in spcreators
    }


# 测试不支持1维输入的稀疏矩阵初始化
@pytest.mark.parametrize("spcreator", [bsr_array, csc_array, dia_array, lil_array])
def test_no_1d_support_in_init(spcreator):
    """测试初始化时不支持1维输入的稀疏矩阵"""
    with pytest.raises(ValueError, match="arrays don't support 1D input"):
        spcreator([0, 1, 2, 3])


# 主测试类
@pytest.mark.parametrize("spcreator", spcreators)
class TestCommon1D:
    """测试1维稀疏格式共享的常见功能"""

    def test_create_empty(self, spcreator):
        """测试创建空稀疏矩阵"""
        assert_equal(spcreator((3,)).toarray(), np.zeros(3))
        assert_equal(spcreator((3,)).nnz, 0)
        assert_equal(spcreator((3,)).count_nonzero(), 0)

    def test_invalid_shapes(self, spcreator):
        """测试不支持的形状"""
        with pytest.raises(ValueError, match='elements cannot be negative'):
            spcreator((-3,))

    def test_repr(self, spcreator, dat1d):
        """测试稀疏矩阵的repr表示"""
        repr(spcreator(dat1d))

    def test_str(self, spcreator, dat1d):
        """测试稀疏矩阵的str表示"""
        str(spcreator(dat1d))

    def test_neg(self, spcreator):
        """测试稀疏矩阵的取负操作"""
        A = np.array([-1, 0, 17, 0, -5, 0, 1, -4, 0, 0, 0, 0], 'd')
        assert_equal(-A, (-spcreator(A)).toarray())

    def test_1d_supported_init(self, spcreator):
        """测试支持1维输入的稀疏矩阵初始化"""
        A = spcreator([0, 1, 2, 3])
        assert A.ndim == 1

    def test_reshape_1d_tofrom_row_or_column(self, spcreator):
        """测试从1维到2维和从2维到1维的重塑操作"""
        # 添加一个维度，从1维到2维
        x = spcreator([1, 0, 7, 0, 0, 0, 0, -3, 0, 0, 0, 5])
        y = x.reshape(1, 12)
        desired = [[1, 0, 7, 0, 0, 0, 0, -3, 0, 0, 0, 5]]
        assert_equal(y.toarray(), desired)

        # 去除大小为1的维度，从2维到1维
        x = spcreator(desired)
        y = x.reshape(12)
        assert_equal(y.toarray(), desired[0])
        y2 = x.reshape((12,))
        assert y.shape == y2.shape

        # 将一个2维列转换为1维，从2维到1维
        y = x.T.reshape(12)
        assert_equal(y.toarray(), desired[0])
    def test_reshape(self, spcreator):
        # 创建稀疏向量对象，并初始化为 [1, 0, 7, 0, 0, 0, 0, -3, 0, 0, 0, 5]
        x = spcreator([1, 0, 7, 0, 0, 0, 0, -3, 0, 0, 0, 5])
        
        # 对稀疏向量进行形状重塑为 (4, 3)
        y = x.reshape((4, 3))
        
        # 预期结果的二维列表
        desired = [[1, 0, 7], [0, 0, 0], [0, -3, 0], [0, 0, 5]]
        
        # 断言重塑后的稀疏向量的密集表示等于预期结果
        assert_equal(y.toarray(), desired)
        
        # 对稀疏向量进行形状重塑为 (12,)，预期返回结果是原始向量本身
        y = x.reshape((12,))
        assert y is x
        
        # 对稀疏向量进行形状重塑为 12，预期结果与原始向量的密集表示相同
        y = x.reshape(12)
        assert_equal(y.toarray(), x.toarray())

    def test_sum(self, spcreator):
        # 设置随机种子
        np.random.seed(1234)
        
        # 创建不同类型的数据数组
        dat_1 = np.array([0, 1, 2, 3, -4, 5, -6, 7, 9])
        dat_2 = np.random.rand(5)
        dat_3 = np.array([])
        dat_4 = np.zeros((40,))
        arrays = [dat_1, dat_2, dat_3, dat_4]
        
        # 遍历数组进行测试
        for dat in arrays:
            # 使用 spcreator 创建稀疏向量
            datsp = spcreator(dat)
            
            # 忽略运算过程中的溢出警告
            with np.errstate(over='ignore'):
                # 断言稀疏向量的总和是一个标量
                assert np.isscalar(datsp.sum())
                
                # 断言稀疏向量总和的全等性
                assert_allclose(dat.sum(), datsp.sum())
                
                # 断言稀疏向量沿着各个轴的总和的全等性
                assert_allclose(dat.sum(axis=None), datsp.sum(axis=None))
                assert_allclose(dat.sum(axis=0), datsp.sum(axis=0))
                assert_allclose(dat.sum(axis=-1), datsp.sum(axis=-1))
        
        # 测试 `out` 参数
        datsp.sum(axis=0, out=np.zeros(()))

    def test_sum_invalid_params(self, spcreator):
        # 创建一个大小错误的 `out` 数组
        out = np.zeros((3,))
        
        # 创建一个普通的数据数组
        dat = np.array([0, 1, 2])
        
        # 使用 spcreator 创建稀疏向量
        datsp = spcreator(dat)
        
        # 使用 pytest 断言来测试不合法的参数
        with pytest.raises(ValueError, match='axis must be None, -1 or 0'):
            datsp.sum(axis=1)
        with pytest.raises(TypeError, match='Tuples are not accepted'):
            datsp.sum(axis=(0, 1))
        with pytest.raises(TypeError, match='axis must be an integer'):
            datsp.sum(axis=1.5)
        with pytest.raises(ValueError, match='dimensions do not match'):
            datsp.sum(axis=0, out=out)

    def test_numpy_sum(self, spcreator):
        # 创建一个普通的数据数组
        dat = np.array([0, 1, 2])
        
        # 使用 spcreator 创建稀疏向量
        datsp = spcreator(dat)
        
        # 计算普通数组和稀疏向量的总和，断言它们的全等性
        dat_sum = np.sum(dat)
        datsp_sum = np.sum(datsp)
        assert_allclose(dat_sum, datsp_sum)

    def test_mean(self, spcreator):
        # 创建一个普通的数据数组
        dat = np.array([0, 1, 2])
        
        # 使用 spcreator 创建稀疏向量
        datsp = spcreator(dat)
        
        # 断言稀疏向量的均值与普通数组的均值的全等性
        assert_allclose(dat.mean(), datsp.mean())
        
        # 断言稀疏向量的均值是一个标量
        assert np.isscalar(datsp.mean(axis=None))
        
        # 断言稀疏向量沿着各个轴的均值的全等性
        assert_allclose(dat.mean(axis=None), datsp.mean(axis=None))
        assert_allclose(dat.mean(axis=0), datsp.mean(axis=0))
        assert_allclose(dat.mean(axis=-1), datsp.mean(axis=-1))
        
        # 使用 pytest 断言来测试不合法的参数
        with pytest.raises(ValueError, match='axis'):
            datsp.mean(axis=1)
        with pytest.raises(ValueError, match='axis'):
            datsp.mean(axis=-2)
    # 测试函数，用于验证特定情况下抛出异常
    def test_mean_invalid_params(self, spcreator):
        # 创建一个形状为 (1, 3) 的全零数组
        out = np.asarray(np.zeros((1, 3)))
        # 创建一个包含特定数据的二维数组
        dat = np.array([[0, 1, 2], [3, -4, 5], [-6, 7, 9]])

        # 使用 spcreator 函数处理 dat 数组，返回稀疏矩阵
        datsp = spcreator(dat)
        
        # 测试块，验证是否抛出 ValueError 异常并检查异常消息
        with pytest.raises(ValueError, match='axis out of range'):
            # 调用稀疏矩阵的 mean 方法，传入 axis=3 参数
            datsp.mean(axis=3)
        
        # 测试块，验证是否抛出 TypeError 异常并检查异常消息
        with pytest.raises(TypeError, match='Tuples are not accepted'):
            # 调用稀疏矩阵的 mean 方法，传入 axis=(0, 1) 参数
            datsp.mean(axis=(0, 1))
        
        # 测试块，验证是否抛出 TypeError 异常并检查异常消息
        with pytest.raises(TypeError, match='axis must be an integer'):
            # 调用稀疏矩阵的 mean 方法，传入 axis=1.5 参数
            datsp.mean(axis=1.5)
        
        # 测试块，验证是否抛出 ValueError 异常并检查异常消息
        with pytest.raises(ValueError, match='dimensions do not match'):
            # 调用稀疏矩阵的 mean 方法，传入 axis=1 和 out 参数
            datsp.mean(axis=1, out=out)

    # 测试函数，验证在不同数据类型下稀疏矩阵与原始数组的 sum 方法结果一致
    def test_sum_dtype(self, spcreator):
        # 创建一个包含特定数据的一维数组
        dat = np.array([0, 1, 2])
        # 使用 spcreator 函数处理 dat 数组，返回稀疏矩阵
        datsp = spcreator(dat)

        # 遍历支持的数据类型列表
        for dtype in supported_dtypes:
            # 调用原始数组的 sum 方法，指定数据类型为当前迭代的 dtype
            dat_sum = dat.sum(dtype=dtype)
            # 调用稀疏矩阵的 sum 方法，指定数据类型为当前迭代的 dtype
            datsp_sum = datsp.sum(dtype=dtype)

            # 断言两者的结果在数值上接近
            assert_allclose(dat_sum, datsp_sum)
            # 断言两者的数据类型相同
            assert_equal(dat_sum.dtype, datsp_sum.dtype)

    # 测试函数，验证在不同数据类型下稀疏矩阵与原始数组的 mean 方法结果一致
    def test_mean_dtype(self, spcreator):
        # 创建一个包含特定数据的一维数组
        dat = np.array([0, 1, 2])
        # 使用 spcreator 函数处理 dat 数组，返回稀疏矩阵
        datsp = spcreator(dat)

        # 遍历支持的数据类型列表
        for dtype in supported_dtypes:
            # 调用原始数组的 mean 方法，指定数据类型为当前迭代的 dtype
            dat_mean = dat.mean(dtype=dtype)
            # 调用稀疏矩阵的 mean 方法，指定数据类型为当前迭代的 dtype
            datsp_mean = datsp.mean(dtype=dtype)

            # 断言两者的结果在数值上接近
            assert_allclose(dat_mean, datsp_mean)
            # 断言两者的数据类型相同
            assert_equal(dat_mean.dtype, datsp_mean.dtype)

    # 测试函数，验证稀疏矩阵与原始数组的 mean 方法在使用 out 参数时的一致性
    def test_mean_out(self, spcreator):
        # 创建一个包含特定数据的一维数组
        dat = np.array([0, 1, 2])
        # 使用 spcreator 函数处理 dat 数组，返回稀疏矩阵
        datsp = spcreator(dat)

        # 创建两个数组，用于接收 mean 方法的输出
        dat_out = np.array([0])
        datsp_out = np.array([0])

        # 调用原始数组的 mean 方法，指定输出数组和 keepdims=True 参数
        dat.mean(out=dat_out, keepdims=True)
        # 调用稀疏矩阵的 mean 方法，指定输出数组
        datsp.mean(out=datsp_out)
        # 断言两者的结果在数值上接近
        assert_allclose(dat_out, datsp_out)

        # 调用原始数组的 mean 方法，指定 axis=0, out 和 keepdims=True 参数
        dat.mean(axis=0, out=dat_out, keepdims=True)
        # 调用稀疏矩阵的 mean 方法，指定 axis=0 和 out 参数
        datsp.mean(axis=0, out=datsp_out)
        # 断言两者的结果在数值上接近
        assert_allclose(dat_out, datsp_out)

    # 测试函数，验证稀疏矩阵与原始数组的 mean 方法结果与 numpy 中的 mean 函数结果一致
    def test_numpy_mean(self, spcreator):
        # 创建一个包含特定数据的一维数组
        dat = np.array([0, 1, 2])
        # 使用 spcreator 函数处理 dat 数组，返回稀疏矩阵
        datsp = spcreator(dat)

        # 调用 numpy 中的 mean 函数计算原始数组的平均值
        dat_mean = np.mean(dat)
        # 调用 numpy 中的 mean 函数计算稀疏矩阵的平均值
        datsp_mean = np.mean(datsp)

        # 断言两者的结果在数值上接近
        assert_allclose(dat_mean, datsp_mean)
        # 断言两者的数据类型相同
        assert_equal(dat_mean.dtype, datsp_mean.dtype)

    # 用于创建稀疏矩阵的装饰器函数，后续测试均基于该函数创建的稀疏矩阵
    @sup_complex
    def test_from_array(self, spcreator):
        # 创建一个包含特定数据的一维数组
        A = np.array([2, 3, 4])
        # 断言稀疏矩阵与原始数组在调用 toarray 方法后结果一致
        assert_equal(spcreator(A).toarray(), A)

        # 创建一个包含复数数据的一维数组
        A = np.array([1.0 + 3j, 0, -1])
        # 断言稀疏矩阵与原始数组在调用 toarray 方法后结果一致
        assert_equal(spcreator(A).toarray(), A)
        # 断言稀疏矩阵与转换为 int16 数据类型后的原始数组结果一致
        assert_equal(spcreator(A, dtype='int16').toarray(), A.astype('int16'))

    # 用于创建稀疏矩阵的装饰器函数，后续测试均基于该函数创建的稀疏矩阵
    @sup_complex
    def test_from_list(self, spcreator):
        # 创建一个包含特定数据的列表
        A = [2, 3, 4]
        # 断言稀疏矩阵与原始数组在调用 toarray 方法后结果一致
        assert_equal(spcreator(A).toarray(), A)

        # 创建一个包含复数数据的列表
        A = [1.0 + 3j, 0, -1]
        # 断言稀疏矩阵与转换为 numpy 数组后的原始数组结果一致
        assert_equal(spcreator(A).toarray(), np.array(A))
        # 断言
    # 测试稀疏矩阵创建函数的行为，使用给定的 spcreator 函数
    def test_from_sparse(self, spcreator):
        # 创建一个包含整数的 NumPy 数组 D
        D = np.array([1, 0, 0])
        # 使用 coo_array 函数将 D 转换为稀疏矩阵 S
        S = coo_array(D)
        # 断言使用 spcreator 函数将 S 转换为稠密数组后与 D 相等
        assert_equal(spcreator(S).toarray(), D)
        # 将 D 直接作为参数传递给 spcreator 函数，再将结果转换为稠密数组，断言与 D 相等
        S = spcreator(D)
        assert_equal(spcreator(S).toarray(), D)

        # 创建一个包含复数的 NumPy 数组 D
        D = np.array([1.0 + 3j, 0, -1])
        # 使用 coo_array 函数将 D 转换为稀疏矩阵 S
        S = coo_array(D)
        # 断言使用 spcreator 函数将 S 转换为稠密数组后与 D 相等
        assert_equal(spcreator(S).toarray(), D)
        # 断言使用 spcreator 函数将 S 转换为 int16 类型的稠密数组后与 D 转换为 int16 类型后相等
        assert_equal(spcreator(S, dtype='int16').toarray(), D.astype('int16'))
        # 将 D 直接作为参数传递给 spcreator 函数，再将结果转换为稠密数组，断言与 D 相等
        S = spcreator(D)
        # 断言使用 spcreator 函数将 S 转换为稠密数组后与 D 相等
        assert_equal(spcreator(S).toarray(), D)
        # 断言使用 spcreator 函数将 S 转换为 int16 类型的稠密数组后与 D 转换为 int16 类型后相等
        assert_equal(spcreator(S, dtype='int16').toarray(), D.astype('int16'))

    # 测试稀疏矩阵的 toarray 方法，使用给定的 spcreator 函数和 dat1d 参数
    def test_toarray(self, spcreator, dat1d):
        # 使用 spcreator 函数创建稀疏矩阵 datsp
        datsp = spcreator(dat1d)
        # 检查默认情况下的 C 或 F 连续性
        chk = datsp.toarray()
        assert_equal(chk, dat1d)
        assert chk.flags.c_contiguous == chk.flags.f_contiguous

        # 检查指定为 C 连续性的情况
        chk = datsp.toarray(order='C')
        assert_equal(chk, dat1d)
        assert chk.flags.c_contiguous
        assert chk.flags.f_contiguous

        # 检查指定为 F 连续性的情况
        chk = datsp.toarray(order='F')
        assert_equal(chk, dat1d)
        assert chk.flags.c_contiguous
        assert chk.flags.f_contiguous

        # 检查使用输出参数时的情况
        out = np.zeros(datsp.shape, dtype=datsp.dtype)
        datsp.toarray(out=out)
        assert_equal(out, dat1d)

        # 在不初始化为零的情况下检查
        out[...] = 1.0
        datsp.toarray(out=out)
        assert_equal(out, dat1d)

        # np.dot 在稀疏矩阵上不适用（除非是标量），所以这里测试 dat1d 是否与 datsp.toarray() 匹配
        a = np.array([1.0, 2.0, 3.0, 4.0])
        dense_dot_dense = np.dot(a, dat1d)
        check = np.dot(a, datsp.toarray())
        assert_equal(dense_dot_dense, check)

        b = np.array([1.0, 2.0, 3.0, 4.0])
        dense_dot_dense = np.dot(dat1d, b)
        check = np.dot(datsp.toarray(), b)
        assert_equal(dense_dot_dense, check)

        # 检查布尔数据的工作情况
        spbool = spcreator(dat1d, dtype=bool)
        arrbool = dat1d.astype(bool)
        assert_equal(spbool.toarray(), arrbool)

    # 测试稀疏矩阵的加法操作，使用给定的 spcreator 函数和 datsp_math_dtypes 参数
    def test_add(self, spcreator, datsp_math_dtypes):
        # 遍历 datsp_math_dtypes[spcreator] 中的每种数据类型、数据和稀疏矩阵
        for dtype, dat, datsp in datsp_math_dtypes[spcreator]:
            # 复制数据 dat 到 a
            a = dat.copy()
            a[0] = 2.0
            # 将 datsp 赋值给 b
            b = datsp
            # 执行加法操作 c = b + a，并断言结果与 b.toarray() + a 相等
            c = b + a
            assert_equal(c, b.toarray() + a)

            # 测试广播特性
            # 注意：不能将非零标量加到稀疏矩阵上。可以将长度为1的数组加上
            c = b + a[0:1]
            assert_equal(c, b.toarray() + a[0])

    # 测试稀疏矩阵的右加操作，使用给定的 spcreator 函数和 datsp_math_dtypes 参数
    def test_radd(self, spcreator, datsp_math_dtypes):
        # 遍历 datsp_math_dtypes[spcreator] 中的每种数据类型、数据和稀疏矩阵
        for dtype, dat, datsp in datsp_math_dtypes[spcreator]:
            # 复制数据 dat 到 a
            a = dat.copy()
            a[0] = 2.0
            # 将 datsp 赋值给 b
            b = datsp
            # 执行右加操作 c = a + b，并断言结果与 a + b.toarray() 相等
            c = a + b
            assert_equal(c, a + b.toarray())
    def test_rsub(self, spcreator, datsp_math_dtypes):
        # 遍历数据类型及其对应的数据和稀疏矩阵
        for dtype, dat, datsp in datsp_math_dtypes[spcreator]:
            # 如果数据类型是布尔型，跳过（布尔数组的减法在1.9.0版本中已废弃）
            if dtype == np.dtype('bool'):
                continue
            
            # 断言 dat - datsp 的结果为 [0, 0, 0, 0]
            assert_equal((dat - datsp), [0, 0, 0, 0])
            # 断言 datsp - dat 的结果为 [0, 0, 0, 0]
            assert_equal((datsp - dat), [0, 0, 0, 0])
            # 断言 0 - datsp 的结果为 -dat 的稀疏数组形式
            assert_equal((0 - datsp).toarray(), -dat)

            # 创建稀疏矩阵 A，数据为 [1, -4, 0, 2]，数据类型为双精度浮点型
            A = spcreator([1, -4, 0, 2], dtype='d')
            # 断言 dat - A 的结果与 dat - A.toarray() 的结果相等
            assert_equal((dat - A), dat - A.toarray())
            # 断言 A - dat 的结果与 A.toarray() - dat 的结果相等
            assert_equal((A - dat), A.toarray() - dat)
            # 断言 A.toarray() - datsp 的结果与 A.toarray() - dat 的结果相等
            assert_equal(A.toarray() - datsp, A.toarray() - dat)
            # 断言 datsp - A.toarray() 的结果与 dat - A.toarray() 的结果相等
            assert_equal(datsp - A.toarray(), dat - A.toarray())

            # 测试广播特性
            assert_equal(dat[:1] - datsp, dat[:1] - dat)

    def test_matvec(self, spcreator):
        # 创建数组 A，数据为 [2, 0, 3.0]
        A = np.array([2, 0, 3.0])
        # 创建稀疏矩阵 Asp，基于数组 A
        Asp = spcreator(A)
        # 创建列向量 col，数据为 [[1], [2], [3]]
        col = np.array([[1, 2, 3]]).T

        # 断言稀疏矩阵 Asp 与列向量 col 的乘积近似等于稀疏矩阵 Asp.toarray() 与列向量 col 的乘积
        assert_allclose(Asp @ col, Asp.toarray() @ col)

        # 断言 A 与 [1, 2, 3] 的点积的形状为 ()
        assert (A @ np.array([1, 2, 3])).shape == ()
        # 断言稀疏矩阵 Asp 与 [1, 2, 3] 的乘积结果为 11
        assert Asp @ np.array([1, 2, 3]) == 11
        # 断言稀疏矩阵 Asp 与 [1, 2, 3] 的乘积的形状为 ()
        assert (Asp @ np.array([1, 2, 3])).shape == ()
        # 断言稀疏矩阵 Asp 与 [[1], [2], [3]] 的乘积的形状为 ()
        assert (Asp @ np.array([[1], [2], [3]])).shape == ()
        
        # 检查结果类型
        assert isinstance(Asp @ matrix([[1, 2, 3]]).T, np.ndarray)
        # 断言稀疏矩阵 Asp 与 [[1, 2, 3]] 的转置的乘积的形状为 ()
        assert (Asp @ np.array([[1, 2, 3]]).T).shape == ()

        # 确保对于不正确的维度会引发异常
        bad_vecs = [np.array([1, 2]), np.array([1, 2, 3, 4]), np.array([[1], [2]])]
        for x in bad_vecs:
            with pytest.raises(ValueError, match='dimension mismatch'):
                Asp.__matmul__(x)

        # 稀疏矩阵产品与数组产品的当前关系
        dot_result = np.dot(Asp.toarray(), [1, 2, 3])
        # 断言稀疏矩阵 Asp 与 [1, 2, 3] 的乘积近似等于点积结果 dot_result
        assert_allclose(Asp @ np.array([1, 2, 3]), dot_result)
        # 断言稀疏矩阵 Asp 与 [[1], [2], [3]] 的乘积近似等于点积结果 dot_result 的转置
        assert_allclose(Asp @ [[1], [2], [3]], dot_result.T)

    def test_rmatvec(self, spcreator, dat1d):
        # 创建稀疏矩阵 M，基于一维数据 dat1d
        M = spcreator(dat1d)
        # 断言 [1, 2, 3, 4] 与 M 的乘积近似等于 [1, 2, 3, 4] 与 M.toarray() 的点积
        assert_allclose([1, 2, 3, 4] @ M, np.dot([1, 2, 3, 4], M.toarray()))
        # 创建行向量 row，数据为 [[1, 2, 3, 4]]
        row = np.array([[1, 2, 3, 4]])
        # 断言行向量 row 与 M 的乘积近似等于行向量 row 与 M.toarray() 的点积
        assert_allclose(row @ M, row @ M.toarray())

    def test_transpose(self, spcreator, dat1d):
        # 遍历数据 dat1d 和空数组 []
        for A in [dat1d, np.array([])]:
            # 创建稀疏矩阵 B，基于数组 A
            B = spcreator(A)
            # 断言稀疏矩阵 B 的稀疏表示与数组 A 相等
            assert_equal(B.toarray(), A)
            # 断言稀疏矩阵 B 的转置的稀疏表示与数组 A 相等
            assert_equal(B.transpose().toarray(), A)
            # 断言稀疏矩阵 B 的数据类型与数组 A 的数据类型相等
            assert_equal(B.dtype, A.dtype)

    def test_add_dense_to_sparse(self, spcreator, datsp_math_dtypes):
        # 遍历数据类型及其对应的数据和稀疏矩阵
        for dtype, dat, datsp in datsp_math_dtypes[spcreator]:
            # 计算 dat + datsp 的结果并断言其与 dat + dat 的结果相等
            sum1 = dat + datsp
            assert_equal(sum1, dat + dat)
            # 计算 datsp + dat 的结果并断言其与 dat + dat 的结果相等
            sum2 = datsp + dat
            assert_equal(sum2, dat + dat)
    # 定义测试方法 test_iterator，用于测试迭代器 __iter__ 是否与 NumPy 兼容
    def test_iterator(self, spcreator):
        # 创建一个 NumPy 数组 B，包含从 0 到 4 的整数
        B = np.arange(5)
        # 使用 spcreator 函数（稀疏矩阵创建函数）创建稀疏矩阵 A
        A = spcreator(B)

        # 如果稀疏矩阵 A 的格式不在 ['coo', 'dia', 'bsr'] 中
        if A.format not in ['coo', 'dia', 'bsr']:
            # 对 A 和 B 进行并行迭代，检查每对元素是否相等
            for x, y in zip(A, B):
                assert_equal(x, y)

    # 定义测试方法 test_resize，用于测试稀疏矩阵的 resize(shape) 方法
    def test_resize(self, spcreator):
        # 创建一个 NumPy 数组 D，包含整数 [1, 0, 3, 4]
        D = np.array([1, 0, 3, 4])
        # 使用 spcreator 函数创建稀疏矩阵 S
        S = spcreator(D)

        # 断言调整稀疏矩阵 S 的形状为 (3,)，并且返回值为 None
        assert S.resize((3,)) is None
        # 断言将稀疏矩阵 S 转换为密集数组后的值与预期值 [1, 0, 3] 相等
        assert_equal(S.toarray(), [1, 0, 3])
        # 将稀疏矩阵 S 的形状调整为 (5,)
        S.resize((5,))
        # 断言将稀疏矩阵 S 转换为密集数组后的值与预期值 [1, 0, 3, 0, 0] 相等
        assert_equal(S.toarray(), [1, 0, 3, 0, 0])
```