# `D:\src\scipysrc\scipy\scipy\interpolate\tests\test_rbfinterp.py`

```
# 导入 pickle 模块，用于序列化和反序列化 Python 对象
import pickle
# 导入 pytest 模块，用于编写测试用例
import pytest
# 导入 numpy 模块，并指定别名 np
import numpy as np
# 导入 LinAlgError 异常类，用于处理线性代数相关的错误
from numpy.linalg import LinAlgError
# 导入 assert_allclose 函数，用于比较两个数组是否近似相等
from numpy.testing import assert_allclose
# 导入 Halton 类，用于生成 Halton 序列
from scipy.stats.qmc import Halton
# 导入 cKDTree 类，用于构建 KD 树
from scipy.spatial import cKDTree
# 导入 _rbfinterp 模块中的相关函数和类
from scipy.interpolate._rbfinterp import (
    _AVAILABLE, _SCALE_INVARIANT, _NAME_TO_MIN_DEGREE, _monomial_powers,
    RBFInterpolator
    )
# 导入 _rbfinterp_pythran 模块
from scipy.interpolate import _rbfinterp_pythran


def _vandermonde(x, degree):
    # 返回一个矩阵，其中包含在指定次数下评估的 x 的多项式
    powers = _monomial_powers(x.shape[1], degree)
    return _rbfinterp_pythran._polynomial_matrix(x, powers)


def _1d_test_function(x):
    # Wahba 的 "Spline Models for Observational Data" 中使用的测试函数
    # 定义域约为 (0, 3)，值域约为 (-1.0, 0.2)
    x = x[:, 0]
    y = 4.26*(np.exp(-x) - 4*np.exp(-2*x) + 3*np.exp(-3*x))
    return y


def _2d_test_function(x):
    # Franke 的测试函数
    # 定义域约为 (0, 1) X (0, 1)，值域约为 (0.0, 1.2)
    x1, x2 = x[:, 0], x[:, 1]
    term1 = 0.75 * np.exp(-(9*x1-2)**2/4 - (9*x2-2)**2/4)
    term2 = 0.75 * np.exp(-(9*x1+1)**2/49 - (9*x2+1)/10)
    term3 = 0.5 * np.exp(-(9*x1-7)**2/4 - (9*x2-3)**2/4)
    term4 = -0.2 * np.exp(-(9*x1-4)**2 - (9*x2-7)**2)
    y = term1 + term2 + term3 + term4
    return y


def _is_conditionally_positive_definite(kernel, m):
    # 测试核函数是否是 m 阶条件正定的
    # 参考 Fasshauer 的 "Meshfree Approximation Methods with MATLAB" 第 7 章
    nx = 10
    ntests = 100
    for ndim in [1, 2, 3, 4, 5]:
        # 使用 Halton 序列生成样本点，避免样本点过于接近，导致矩阵奇异
        seq = Halton(ndim, scramble=False, seed=np.random.RandomState())
        for _ in range(ntests):
            x = 2*seq.random(nx) - 1
            A = _rbfinterp_pythran._kernel_matrix(x, kernel)
            P = _vandermonde(x, m - 1)
            Q, R = np.linalg.qr(P, mode='complete')
            # Q2 形成一个基，跨越 P.T.dot(x) = 0 的空间。将 A 投影到这个空间，然后使用 Cholesky 分解检查是否是正定的
            Q2 = Q[:, P.shape[1]:]
            B = Q2.T.dot(A).dot(Q2)
            try:
                np.linalg.cholesky(B)
            except np.linalg.LinAlgError:
                return False

    return True


# 对 parametrize 参数进行排序，以避免并行化问题
# 详情请参考：https://github.com/pytest-dev/pytest-xdist/issues/432
@pytest.mark.parametrize('kernel', sorted(_AVAILABLE))
def test_conditionally_positive_definite(kernel):
    # 测试 _AVAILABLE 中的每个核函数是否是 _NAME_TO_MIN_DEGREE 中指定阶数的条件正定的
    # 这是平滑 RBF 插值器在一般情况下是良定的必要条件
    # 根据 kernel 从 _NAME_TO_MIN_DEGREE 字典获取最小度数，如果找不到则默认为 -1，然后加 1
    m = _NAME_TO_MIN_DEGREE.get(kernel, -1) + 1
    # 断言 kernel 是否是条件上正定的，使用最小度数 m 进行检查
    assert _is_conditionally_positive_definite(kernel, m)
    @pytest.mark.parametrize('kernel', sorted(_SCALE_INVARIANT))
    def test_scale_invariance_1d(self, kernel):
        # 针对一维情况，验证_SCALE_INVARIANT中的函数在平滑参数为0时对形状参数不敏感。
        seq = Halton(1, scramble=False, seed=np.random.RandomState())
        # 使用Halton序列生成长度为50的随机数作为x坐标
        x = 3*seq.random(50)
        # 使用一维测试函数生成与x对应的y值
        y = _1d_test_function(x)
        # 使用Halton序列生成长度为50的随机数作为插值点x坐标
        xitp = 3*seq.random(50)
        # 分别对epsilon=1.0和epsilon=2.0的情况建立插值模型，并计算插值结果
        yitp1 = self.build(x, y, epsilon=1.0, kernel=kernel)(xitp)
        yitp2 = self.build(x, y, epsilon=2.0, kernel=kernel)(xitp)
        # 断言两次插值结果的近似程度，允许的误差为1e-8
        assert_allclose(yitp1, yitp2, atol=1e-8)

    @pytest.mark.parametrize('kernel', sorted(_SCALE_INVARIANT))
    def test_scale_invariance_2d(self, kernel):
        # 针对二维情况，验证_SCALE_INVARIANT中的函数在平滑参数为0时对形状参数不敏感。
        seq = Halton(2, scramble=False, seed=np.random.RandomState())
        # 使用Halton序列生成长度为100的随机数作为x坐标
        x = seq.random(100)
        # 使用二维测试函数生成与x对应的y值
        y = _2d_test_function(x)
        # 使用Halton序列生成长度为100的随机数作为插值点x坐标
        xitp = seq.random(100)
        # 分别对epsilon=1.0和epsilon=2.0的情况建立插值模型，并计算插值结果
        yitp1 = self.build(x, y, epsilon=1.0, kernel=kernel)(xitp)
        yitp2 = self.build(x, y, epsilon=2.0, kernel=kernel)(xitp)
        # 断言两次插值结果的近似程度，允许的误差为1e-8
        assert_allclose(yitp1, yitp2, atol=1e-8)

    @pytest.mark.parametrize('kernel', sorted(_AVAILABLE))
    def test_extreme_domains(self, kernel):
        # 确保插值器在非常大或非常小的域上保持数值稳定性。
        seq = Halton(2, scramble=False, seed=np.random.RandomState())
        scale = 1e50
        shift = 1e55

        # 使用Halton序列生成长度为100的随机数作为x坐标
        x = seq.random(100)
        # 使用二维测试函数生成与x对应的y值
        y = _2d_test_function(x)
        # 使用Halton序列生成长度为100的随机数作为插值点x坐标
        xitp = seq.random(100)

        if kernel in _SCALE_INVARIANT:
            # 对于_SCALE_INVARIANT中的核函数，分别计算两次插值结果
            yitp1 = self.build(x, y, kernel=kernel)(xitp)
            yitp2 = self.build(
                x*scale + shift, y,
                kernel=kernel
                )(xitp*scale + shift)
        else:
            # 对于不在_SCALE_INVARIANT中的核函数，使用固定的epsilon=5.0进行插值
            yitp1 = self.build(x, y, epsilon=5.0, kernel=kernel)(xitp)
            yitp2 = self.build(
                x*scale + shift, y,
                epsilon=5.0/scale,
                kernel=kernel
                )(xitp*scale + shift)

        # 断言两次插值结果的近似程度，允许的误差为1e-8
        assert_allclose(yitp1, yitp2, atol=1e-8)

    def test_polynomial_reproduction(self):
        # 如果观测数据来自多项式，则插值器应该能够精确复现该多项式，前提是degree足够高。
        rng = np.random.RandomState(0)
        seq = Halton(2, scramble=False, seed=rng)
        degree = 3

        # 使用Halton序列生成长度为50的随机数作为x坐标
        x = seq.random(50)
        # 使用Halton序列生成长度为50的随机数作为插值点x坐标
        xitp = seq.random(50)

        # 构建Vandermonde矩阵P和插值点Pitp
        P = _vandermonde(x, degree)
        Pitp = _vandermonde(xitp, degree)

        # 生成多项式系数，使用正态分布随机数
        poly_coeffs = rng.normal(0.0, 1.0, P.shape[1])

        # 计算原始数据y和插值结果yitp1、yitp2
        y = P.dot(poly_coeffs)
        yitp1 = Pitp.dot(poly_coeffs)
        yitp2 = self.build(x, y, degree=degree)(xitp)

        # 断言插值结果yitp1和yitp2的近似程度，允许的误差为1e-8
        assert_allclose(yitp1, yitp2, atol=1e-8)
    # 定义一个单元测试方法，用于测试数据分块处理
    def test_chunking(self, monkeypatch):
        # 如果观察数据来自多项式，插值器应当能够完全重现该多项式，前提是 `degree` 足够高。
        
        # 创建一个随机数生成器，用于生成随机数序列
        rng = np.random.RandomState(0)
        # 使用 Halton 序列生成器创建序列，使用默认的种子和禁用混淆
        seq = Halton(2, scramble=False, seed=rng)
        # 设置多项式的阶数
        degree = 3

        # 设置一个大的 N 值，以测试 RBFInterpolator 的数据分块功能
        largeN = 1000 + 33
        
        # 生成长度为 50 的随机数序列 x
        x = seq.random(50)
        # 生成长度为 largeN 的随机数序列 xitp
        xitp = seq.random(largeN)

        # 计算 x 对应的 Vandermonde 矩阵 P
        P = _vandermonde(x, degree)
        # 计算 xitp 对应的 Vandermonde 矩阵 Pitp
        Pitp = _vandermonde(xitp, degree)

        # 生成服从正态分布的多项式系数
        poly_coeffs = rng.normal(0.0, 1.0, P.shape[1])

        # 计算多项式在 x 上的值 y
        y = P.dot(poly_coeffs)
        # 计算多项式在 xitp 上的值 yitp1
        yitp1 = Pitp.dot(poly_coeffs)
        
        # 使用 build 方法构建插值器 interp
        interp = self.build(x, y, degree=degree)
        # 获取 interp 的 _chunk_evaluator 属性
        ce_real = interp._chunk_evaluator

        # 定义一个新的 _chunk_evaluator 函数，增加了内存预算参数 memory_budget=100
        def _chunk_evaluator(*args, **kwargs):
            kwargs.update(memory_budget=100)
            return ce_real(*args, **kwargs)

        # 使用 monkeypatch 替换 interp 的 _chunk_evaluator 方法为新定义的 _chunk_evaluator 函数
        monkeypatch.setattr(interp, '_chunk_evaluator', _chunk_evaluator)
        
        # 在 xitp 上进行插值计算得到 yitp2
        yitp2 = interp(xitp)
        
        # 断言 yitp1 与 yitp2 在容差 1e-8 内相等
        assert_allclose(yitp1, yitp2, atol=1e-8)

    # 定义一个单元测试方法，用于测试向量数据的插值
    def test_vector_data(self):
        # 确保对向量场进行插值与分别插值每个分量是相同的
        
        # 使用 Halton 序列生成器创建序列，使用默认的种子和随机数生成器
        seq = Halton(2, scramble=False, seed=np.random.RandomState())

        # 生成长度为 100 的随机数序列 x
        x = seq.random(100)
        # 生成长度为 100 的随机数序列 xitp
        xitp = seq.random(100)

        # 计算在 x 上的二维测试函数的值，并将结果存储在 y 中
        y = np.array([_2d_test_function(x),
                      _2d_test_function(x[:, ::-1])]).T

        # 分别使用 build 方法构建插值器对 y 进行插值，得到 yitp1
        yitp1 = self.build(x, y)(xitp)
        # 使用 build 方法构建插值器对 y 的第一列进行插值，得到 yitp2
        yitp2 = self.build(x, y[:, 0])(xitp)
        # 使用 build 方法构建插值器对 y 的第二列进行插值，得到 yitp3
        yitp3 = self.build(x, y[:, 1])(xitp)

        # 断言 yitp1 的第一列与 yitp2 相等
        assert_allclose(yitp1[:, 0], yitp2)
        # 断言 yitp1 的第二列与 yitp3 相等
        assert_allclose(yitp1[:, 1], yitp3)

    # 定义一个单元测试方法，用于测试复数数据的插值
    def test_complex_data(self):
        # 确保对复数输入进行插值与分别插值其实部和虚部是相同的
        
        # 使用 Halton 序列生成器创建序列，使用默认的种子和随机数生成器
        seq = Halton(2, scramble=False, seed=np.random.RandomState())

        # 生成长度为 100 的随机数序列 x
        x = seq.random(100)
        # 生成长度为 100 的随机数序列 xitp
        xitp = seq.random(100)

        # 计算在 x 上的二维测试函数的值，并加上复数单位j乘以 x[:, ::-1] 上的二维测试函数的值，存储在 y 中
        y = _2d_test_function(x) + 1j*_2d_test_function(x[:, ::-1])

        # 使用 build 方法构建插值器对 y 进行插值，得到 yitp1
        yitp1 = self.build(x, y)(xitp)
        # 使用 build 方法构建插值器对 y 的实部进行插值，得到 yitp2
        yitp2 = self.build(x, y.real)(xitp)
        # 使用 build 方法构建插值器对 y 的虚部进行插值，得到 yitp3
        yitp3 = self.build(x, y.imag)(xitp)

        # 断言 yitp1 的实部与 yitp2 相等
        assert_allclose(yitp1.real, yitp2)
        # 断言 yitp1 的虚部与 yitp3 相等
        assert_allclose(yitp1.imag, yitp3)

    # 定义一个单元测试方法，用于测试一维插值的误差
    @pytest.mark.parametrize('kernel', sorted(_AVAILABLE))
    def test_interpolation_misfit_1d(self, kernel):
        # 确保每个核函数在默认的 `degree` 和适当的 `epsilon` 下在一维插值中表现良好
        
        # 使用 Halton 序列生成器创建序列，使用默认的种子和随机数生成器
        seq = Halton(1, scramble=False, seed=np.random.RandomState())

        # 生成长度为 50 的随机数序列 x，并乘以 3
        x = 3*seq.random(50)
        # 生成长度为 50 的随机数序列 xitp，并乘以 3
        xitp = 3*seq.random(50)

        # 计算在 x 上的一维测试函数的值 y
        y = _1d_test_function(x)
        # 计算在 xitp 上的一维测试函数的真实值 ytrue
        ytrue = _1d_test_function(xitp)
        # 使用 build 方法构建插值器对 y 进行插值，得到 yitp
        yitp = self.build(x, y, epsilon=5.0, kernel=kernel)(xitp)

        # 计算均方误差 mse
        mse = np.mean((yitp - ytrue)**2)
        
        # 断言 mse 小于 1.0e-4
        assert mse < 1.0e-4
    # 测试二维插值误差，确保每个核函数在默认 `degree` 和适当的 `epsilon` 下在二维插值中表现良好。
    def test_interpolation_misfit_2d(self, kernel):
        seq = Halton(2, scramble=False, seed=np.random.RandomState())

        # 生成随机的观测点和评估点
        x = seq.random(100)
        xitp = seq.random(100)

        # 获取观测点和评估点对应的真实函数值
        y = _2d_test_function(x)
        ytrue = _2d_test_function(xitp)

        # 使用指定的核函数进行插值
        yitp = self.build(x, y, epsilon=5.0, kernel=kernel)(xitp)

        # 计算均方误差（MSE）
        mse = np.mean((yitp - ytrue)**2)

        # 断言均方误差小于指定阈值
        assert mse < 2.0e-4

    # 使用参数化测试，确保每个核函数都能找到足够去噪的平滑参数。
    @pytest.mark.parametrize('kernel', sorted(_AVAILABLE))
    def test_smoothing_misfit(self, kernel):
        rng = np.random.RandomState(0)
        seq = Halton(1, scramble=False, seed=rng)

        noise = 0.2
        rmse_tol = 0.1
        smoothing_range = 10**np.linspace(-4, 1, 20)

        x = 3*seq.random(100)

        # 生成带有噪声的观测点
        y = _1d_test_function(x) + rng.normal(0.0, noise, (100,))
        ytrue = _1d_test_function(x)

        rmse_within_tol = False

        # 在平滑参数范围内搜索合适的平滑值
        for smoothing in smoothing_range:
            ysmooth = self.build(
                x, y,
                epsilon=1.0,
                smoothing=smoothing,
                kernel=kernel)(x)

            # 计算均方根误差（RMSE）
            rmse = np.sqrt(np.mean((ysmooth - ytrue)**2))

            # 如果 RMSE 小于指定容差，则标记为找到合适的平滑参数
            if rmse < rmse_tol:
                rmse_within_tol = True
                break

        # 断言找到了符合误差容差的平滑参数
        assert rmse_within_tol

    # 测试使用数组作为 `smoothing` 参数，以减少已知异常值的权重。
    def test_array_smoothing(self):
        rng = np.random.RandomState(0)
        seq = Halton(1, scramble=False, seed=rng)
        degree = 2

        x = seq.random(50)

        # 生成带有多项式拟合和异常值的观测点
        P = _vandermonde(x, degree)
        poly_coeffs = rng.normal(0.0, 1.0, P.shape[1])
        y = P.dot(poly_coeffs)
        y_with_outlier = np.copy(y)
        y_with_outlier[10] += 1.0

        # 创建一个与观测点维度相同的平滑参数数组，对异常值施加更大的平滑值
        smoothing = np.zeros((50,))
        smoothing[10] = 1000.0

        # 使用指定的平滑参数进行插值
        yitp = self.build(x, y_with_outlier, smoothing=smoothing)(x)

        # 断言能够几乎精确地重现未损坏的数据
        assert_allclose(yitp, y, atol=1e-4)

    # 测试当观测点和评估点具有不同维度时是否会引发 ValueError 异常。
    def test_inconsistent_x_dimensions_error(self):
        y = Halton(2, scramble=False, seed=np.random.RandomState()).random(10)
        d = _2d_test_function(y)
        x = Halton(1, scramble=False, seed=np.random.RandomState()).random(10)
        match = 'Expected the second axis of `x`'

        # 使用 pytest 的断言检查是否引发了预期的 ValueError 异常
        with pytest.raises(ValueError, match=match):
            self.build(y, d)(x)

    # 测试当观测数据长度与函数长度不一致时是否会引发 ValueError 异常。
    def test_inconsistent_d_length_error(self):
        y = np.linspace(0, 1, 5)[:, None]
        d = np.zeros(1)
        match = 'Expected the first axis of `d`'

        # 使用 pytest 的断言检查是否引发了预期的 ValueError 异常
        with pytest.raises(ValueError, match=match):
            self.build(y, d)
    # 测试函数，验证 `y` 不是二维数组时是否会引发错误
    def test_y_not_2d_error(self):
        # 创建一个长度为5的等间隔数组 `y`
        y = np.linspace(0, 1, 5)
        # 创建一个长度为5的全零数组 `d`
        d = np.zeros(5)
        # 设定匹配字符串，用于验证错误信息
        match = '`y` must be a 2-dimensional array.'
        # 使用 pytest 检查是否引发 ValueError，并验证错误信息是否匹配
        with pytest.raises(ValueError, match=match):
            # 调用被测试的 `build` 方法，传入参数 `y` 和 `d`
            self.build(y, d)

    # 测试函数，验证 `smoothing` 的长度不一致时是否会引发错误
    def test_inconsistent_smoothing_length_error(self):
        # 创建一个长度为5的列向量 `y`
        y = np.linspace(0, 1, 5)[:, None]
        # 创建一个长度为5的全零数组 `d`
        d = np.zeros(5)
        # 创建一个长度为1的全一数组 `smoothing`
        smoothing = np.ones(1)
        # 设定匹配字符串，用于验证错误信息
        match = 'Expected `smoothing` to be'
        # 使用 pytest 检查是否引发 ValueError，并验证错误信息是否匹配
        with pytest.raises(ValueError, match=match):
            # 调用被测试的 `build` 方法，传入参数 `y`、`d` 和 `smoothing`
            self.build(y, d, smoothing=smoothing)

    # 测试函数，验证 `kernel` 名称无效时是否会引发错误
    def test_invalid_kernel_name_error(self):
        # 创建一个长度为5的列向量 `y`
        y = np.linspace(0, 1, 5)[:, None]
        # 创建一个长度为5的全零数组 `d`
        d = np.zeros(5)
        # 设定匹配字符串，用于验证错误信息
        match = '`kernel` must be one of'
        # 使用 pytest 检查是否引发 ValueError，并验证错误信息是否匹配
        with pytest.raises(ValueError, match=match):
            # 调用被测试的 `build` 方法，传入参数 `y`、`d` 和无效的 `kernel` 名称
            self.build(y, d, kernel='test')

    # 测试函数，验证未指定 `epsilon` 时是否会引发错误
    def test_epsilon_not_specified_error(self):
        # 创建一个长度为5的列向量 `y`
        y = np.linspace(0, 1, 5)[:, None]
        # 创建一个长度为5的全零数组 `d`
        d = np.zeros(5)
        # 遍历 `_AVAILABLE` 列表中的 `kernel`
        for kernel in _AVAILABLE:
            # 如果当前 `kernel` 在 `_SCALE_INVARIANT` 中，则跳过
            if kernel in _SCALE_INVARIANT:
                continue
            # 设定匹配字符串，用于验证错误信息
            match = '`epsilon` must be specified'
            # 使用 pytest 检查是否引发 ValueError，并验证错误信息是否匹配
            with pytest.raises(ValueError, match=match):
                # 调用被测试的 `build` 方法，传入参数 `y`、`d` 和当前 `kernel`
                self.build(y, d, kernel=kernel)

    # 测试函数，验证 `x` 不是二维数组时是否会引发错误
    def test_x_not_2d_error(self):
        # 创建一个长度为5的列向量 `y`
        y = np.linspace(0, 1, 5)[:, None]
        # 创建一个长度为5的等间隔数组 `x`
        x = np.linspace(0, 1, 5)
        # 创建一个长度为5的全零数组 `d`
        d = np.zeros(5)
        # 设定匹配字符串，用于验证错误信息
        match = '`x` must be a 2-dimensional array.'
        # 使用 pytest 检查是否引发 ValueError，并验证错误信息是否匹配
        with pytest.raises(ValueError, match=match):
            # 调用被测试的 `build` 方法，传入参数 `y`、`d` 和 `x`
            self.build(y, d)(x)

    # 测试函数，验证观测数不足时是否会引发错误
    def test_not_enough_observations_error(self):
        # 创建一个长度为1的列向量 `y`
        y = np.linspace(0, 1, 1)[:, None]
        # 创建一个长度为1的全零数组 `d`
        d = np.zeros(1)
        # 设定匹配字符串，用于验证错误信息
        match = 'At least 2 data points are required'
        # 使用 pytest 检查是否引发 ValueError，并验证错误信息是否匹配
        with pytest.raises(ValueError, match=match):
            # 调用被测试的 `build` 方法，传入参数 `y`、`d` 和特定 `kernel` 名称
            self.build(y, d, kernel='thin_plate_spline')

    # 测试函数，验证 `degree` 参数过低时是否会产生警告
    def test_degree_warning(self):
        # 创建一个长度为5的列向量 `y`
        y = np.linspace(0, 1, 5)[:, None]
        # 创建一个长度为5的全零数组 `d`
        d = np.zeros(5)
        # 遍历 `_NAME_TO_MIN_DEGREE` 字典中的 `kernel` 和 `deg`
        for kernel, deg in _NAME_TO_MIN_DEGREE.items():
            # 仅测试最小度数不为0的 `kernel`
            if deg >= 1:
                # 设定匹配字符串，用于验证警告信息
                match = f'`degree` should not be below {deg}'
                # 使用 pytest 检查是否引发 Warning，并验证警告信息是否匹配
                with pytest.warns(Warning, match=match):
                    # 调用被测试的 `build` 方法，传入参数 `y`、`d`、`epsilon`、`kernel` 和 `degree`
                    self.build(y, d, epsilon=1.0, kernel=kernel, degree=deg-1)

    # 测试函数，验证 `degree` 参数为-1时是否被接受而不产生警告
    def test_minus_one_degree(self):
        # 创建一个长度为5的列向量 `y`
        y = np.linspace(0, 1, 5)[:, None]
        # 创建一个长度为5的全零数组 `d`
        d = np.zeros(5)
        # 遍历 `_NAME_TO_MIN_DEGREE` 字典中的 `kernel` 和 `_`
        for kernel, _ in _NAME_TO_MIN_DEGREE.items():
            # 调用被测试的 `build` 方法，传入参数 `y`、`d`、`epsilon`、`kernel` 和 `degree`
            self.build(y, d, epsilon=1.0, kernel=kernel, degree=-1)

    # 测试函数，验证 `kernel` 为 "thin_plate_spline" 时观测数据为二维且共线时是否会引发错误
    def test_rank_error(self):
        # 创建一个包含三个点的二维数组 `y`
        y = np.array([[2.0, 0.0], [1.0, 0.0], [0.0, 0.0]])
        # 创建一个长度为3的全零数组 `d`
        d = np.array([0.0, 0.0, 0.0])
        # 设定匹配字符串，用于验证错误信息
        match = 'does not have full column rank'
        # 使用 pytest 检查是否引发 LinAlgError，并验证错误信息是否匹配
        with pytest.raises(LinAlgError, match=match):
            # 调用被测试的 `build` 方法，传入参数 `y`、`d` 和 `kernel`
            self.build(y, d, kernel='thin_plate_spline')(y)
    # 定义单点测试函数
    def test_single_point(self):
        # 确保即使只有一个点（在1、2和3维度中），插值仍然有效。
        for dim in [1, 2, 3]:
            # 创建一个维度为dim的零向量
            y = np.zeros((1, dim))
            # 创建一个长度为1的全一向量
            d = np.ones((1,))
            # 使用线性核函数建立插值器，并对y进行插值
            f = self.build(y, d, kernel='linear')(y)
            # 断言插值结果与d相近
            assert_allclose(d, f)

    # 定义可序列化测试函数
    def test_pickleable(self):
        # 确保我们能够对插值器进行序列化和反序列化而不影响其行为。
        # 创建一个Halton序列对象，指定种子和不进行混淆
        seq = Halton(1, scramble=False, seed=np.random.RandomState(2305982309))

        # 生成长度为50的随机数列x和xitp
        x = 3*seq.random(50)
        xitp = 3*seq.random(50)

        # 计算x处的测试函数值
        y = _1d_test_function(x)

        # 使用输入x和y建立插值器
        interp = self.build(x, y)

        # 对xitp进行插值，获得yitp1和yitp2
        yitp1 = interp(xitp)
        yitp2 = pickle.loads(pickle.dumps(interp))(xitp)

        # 断言yitp1和yitp2在数值上相近
        assert_allclose(yitp1, yitp2, atol=1e-16)
class TestRBFInterpolatorNeighborsNone(_TestRBFInterpolator):
    # Test class for RBFInterpolator with no nearest neighbors specified.
    
    def build(self, *args, **kwargs):
        # Method to instantiate RBFInterpolator object.
        return RBFInterpolator(*args, **kwargs)

    def test_smoothing_limit_1d(self):
        # Test case for 1-dimensional interpolation with large smoothing parameter.
        # The interpolant should approach a least squares fit of a polynomial with the specified degree.
        seq = Halton(1, scramble=False, seed=np.random.RandomState())

        degree = 3
        smoothing = 1e8

        x = 3*seq.random(50)
        xitp = 3*seq.random(50)

        y = _1d_test_function(x)

        yitp1 = self.build(
            x, y,
            degree=degree,
            smoothing=smoothing
            )(xitp)

        P = _vandermonde(x, degree)
        Pitp = _vandermonde(xitp, degree)
        yitp2 = Pitp.dot(np.linalg.lstsq(P, y, rcond=None)[0])

        assert_allclose(yitp1, yitp2, atol=1e-8)

    def test_smoothing_limit_2d(self):
        # Test case for 2-dimensional interpolation with large smoothing parameter.
        # The interpolant should approach a least squares fit of a polynomial with the specified degree.
        seq = Halton(2, scramble=False, seed=np.random.RandomState())

        degree = 3
        smoothing = 1e8

        x = seq.random(100)
        xitp = seq.random(100)

        y = _2d_test_function(x)

        yitp1 = self.build(
            x, y,
            degree=degree,
            smoothing=smoothing
            )(xitp)

        P = _vandermonde(x, degree)
        Pitp = _vandermonde(xitp, degree)
        yitp2 = Pitp.dot(np.linalg.lstsq(P, y, rcond=None)[0])

        assert_allclose(yitp1, yitp2, atol=1e-8)


class TestRBFInterpolatorNeighbors20(_TestRBFInterpolator):
    # Test class for RBFInterpolator using 20 nearest neighbors.
    def build(self, *args, **kwargs):
        # Method to instantiate RBFInterpolator object with 20 nearest neighbors.
        return RBFInterpolator(*args, **kwargs, neighbors=20)

    def test_equivalent_to_rbf_interpolator(self):
        # Test case to ensure results are equivalent to RBFInterpolator with 20 nearest neighbors.
        seq = Halton(2, scramble=False, seed=np.random.RandomState())

        x = seq.random(100)
        xitp = seq.random(100)

        y = _2d_test_function(x)

        yitp1 = self.build(x, y)(xitp)

        yitp2 = []
        tree = cKDTree(x)
        for xi in xitp:
            _, nbr = tree.query(xi, 20)
            yitp2.append(RBFInterpolator(x[nbr], y[nbr])(xi[None])[0])

        assert_allclose(yitp1, yitp2, atol=1e-8)


class TestRBFInterpolatorNeighborsInf(TestRBFInterpolatorNeighborsNone):
    # Test class for RBFInterpolator using neighbors=np.inf.
    # This should give exactly the same results as neighbors=None, but it will be slower.
    def build(self, *args, **kwargs):
        # Method to instantiate RBFInterpolator object with neighbors set to np.inf.
        return RBFInterpolator(*args, **kwargs, neighbors=np.inf)

    def test_equivalent_to_rbf_interpolator(self):
        # Test case to ensure results are equivalent to RBFInterpolator with neighbors set to np.inf.
        seq = Halton(1, scramble=False, seed=np.random.RandomState())

        x = 3*seq.random(50)
        xitp = 3*seq.random(50)

        y = _1d_test_function(x)
        yitp1 = self.build(x, y)(xitp)
        yitp2 = RBFInterpolator(x, y)(xitp)

        assert_allclose(yitp1, yitp2, atol=1e-8)
```