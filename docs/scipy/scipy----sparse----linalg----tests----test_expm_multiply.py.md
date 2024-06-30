# `D:\src\scipysrc\scipy\scipy\sparse\linalg\tests\test_expm_multiply.py`

```
"""Test functions for the sparse.linalg._expm_multiply module."""
# 导入所需的模块和函数
from functools import partial
from itertools import product

import numpy as np
import pytest
from numpy.testing import (assert_allclose, assert_, assert_equal,
                           suppress_warnings)
from scipy.sparse import SparseEfficiencyWarning
from scipy.sparse.linalg import aslinearoperator
import scipy.linalg
from scipy.sparse.linalg import expm as sp_expm
from scipy.sparse.linalg._expm_multiply import (_theta, _compute_p_max,
        _onenormest_matrix_power, expm_multiply, _expm_multiply_simple,
        _expm_multiply_interval)
from scipy._lib._util import np_long


IMPRECISE = {np.single, np.csingle}
REAL_DTYPES = {np.intc, np_long, np.longlong,
               np.float32, np.float64, np.longdouble}
COMPLEX_DTYPES = {np.complex64, np.complex128, np.clongdouble}
# 使用排序的列表确保测试的顺序固定
DTYPES = sorted(REAL_DTYPES ^ COMPLEX_DTYPES, key=str)


def estimated(func):
    """If trace is estimated, it should warn.

    We warn that estimation of trace might impact performance.
    All result have to be correct nevertheless!

    """
    def wrapped(*args, **kwds):
        # 使用 pytest 的 warns 上下文管理器确保函数调用时会触发 UserWarning
        with pytest.warns(UserWarning,
                          match="Trace of LinearOperator not available"):
            return func(*args, **kwds)
    return wrapped


def less_than_or_close(a, b):
    # 判断 a 是否小于等于 b 或者它们是否在数值上相近
    return np.allclose(a, b) or (a < b)


class TestExpmActionSimple:
    """
    These tests do not consider the case of multiple time steps in one call.
    """

    def test_theta_monotonicity(self):
        # 对 _theta 字典中的值进行排序，并检查其单调性
        pairs = sorted(_theta.items())
        for (m_a, theta_a), (m_b, theta_b) in zip(pairs[:-1], pairs[1:]):
            assert_(theta_a < theta_b)

    def test_p_max_default(self):
        # 测试默认情况下 _compute_p_max 函数的返回值
        m_max = 55
        expected_p_max = 8
        observed_p_max = _compute_p_max(m_max)
        assert_equal(observed_p_max, expected_p_max)

    def test_p_max_range(self):
        # 测试 _compute_p_max 函数在不同 m_max 值下的范围
        for m_max in range(1, 55+1):
            p_max = _compute_p_max(m_max)
            # 检查 p_max*(p_max - 1) 是否小于等于 m_max + 1
            assert_(p_max*(p_max - 1) <= m_max + 1)
            p_too_big = p_max + 1
            # 检查 p_too_big*(p_too_big - 1) 是否大于 m_max + 1
            assert_(p_too_big*(p_too_big - 1) > m_max + 1)

    def test_onenormest_matrix_power(self):
        # 随机生成矩阵 A，并测试 _onenormest_matrix_power 函数的估算结果与精确结果的关系
        np.random.seed(1234)
        n = 40
        nsamples = 10
        for i in range(nsamples):
            A = scipy.linalg.inv(np.random.randn(n, n))
            for p in range(4):
                if not p:
                    M = np.identity(n)
                else:
                    M = np.dot(M, A)
                estimated = _onenormest_matrix_power(A, p)
                exact = np.linalg.norm(M, 1)
                # 断言估算值是否小于等于精确值的三倍，并且精确值是否小于等于估算值的三倍
                assert_(less_than_or_close(estimated, exact))
                assert_(less_than_or_close(exact, 3*estimated))
    # 定义一个测试函数，用于测试 expm_multiply 函数的行为
    def test_expm_multiply(self):
        # 设置随机种子，以便结果可重现
        np.random.seed(1234)
        # 定义矩阵 A 的大小
        n = 40
        # 定义矩阵 B 的列数
        k = 3
        # 定义样本数
        nsamples = 10
        # 循环进行样本测试
        for i in range(nsamples):
            # 生成一个随机的逆矩阵 A
            A = scipy.linalg.inv(np.random.randn(n, n))
            # 生成一个随机矩阵 B
            B = np.random.randn(n, k)
            # 使用 expm_multiply 计算结果
            observed = expm_multiply(A, B)
            # 使用 sp_expm 计算期望结果
            expected = np.dot(sp_expm(A), B)
            # 断言观察结果与期望结果的近似程度
            assert_allclose(observed, expected)
            # 使用估计函数测试 expm_multiply 的结果
            observed = estimated(expm_multiply)(aslinearoperator(A), B)
            # 断言估计结果与期望结果的近似程度
            assert_allclose(observed, expected)
            # 计算矩阵 A 的迹
            traceA = np.trace(A)
            # 使用 expm_multiply 计算结果，包括矩阵 A 的迹
            observed = expm_multiply(aslinearoperator(A), B, traceA=traceA)
            # 断言观察结果与期望结果的近似程度
            assert_allclose(observed, expected)

    # 定义一个测试函数，用于测试 expm_multiply 函数与向量乘法的行为
    def test_matrix_vector_multiply(self):
        # 设置随机种子，以便结果可重现
        np.random.seed(1234)
        # 定义矩阵 A 的大小
        n = 40
        # 定义样本数
        nsamples = 10
        # 循环进行样本测试
        for i in range(nsamples):
            # 生成一个随机的逆矩阵 A
            A = scipy.linalg.inv(np.random.randn(n, n))
            # 生成一个随机向量 v
            v = np.random.randn(n)
            # 使用 expm_multiply 计算结果
            observed = expm_multiply(A, v)
            # 使用 sp_expm 计算期望结果
            expected = np.dot(sp_expm(A), v)
            # 断言观察结果与期望结果的近似程度
            assert_allclose(observed, expected)
            # 使用估计函数测试 expm_multiply 的结果
            observed = estimated(expm_multiply)(aslinearoperator(A), v)
            # 断言估计结果与期望结果的近似程度
            assert_allclose(observed, expected)

    # 定义一个测试函数，用于测试 _expm_multiply_simple 函数在不同时间点的行为
    def test_scaled_expm_multiply(self):
        # 设置随机种子，以便结果可重现
        np.random.seed(1234)
        # 定义矩阵 A 的大小
        n = 40
        # 定义矩阵 B 的列数
        k = 3
        # 定义样本数
        nsamples = 10
        # 使用 product 函数生成时间点和样本数的组合
        for i, t in product(range(nsamples), [0.2, 1.0, 1.5]):
            # 在计算时忽略无效值错误
            with np.errstate(invalid='ignore'):
                # 生成一个随机的逆矩阵 A
                A = scipy.linalg.inv(np.random.randn(n, n))
                # 生成一个随机矩阵 B
                B = np.random.randn(n, k)
                # 使用 _expm_multiply_simple 计算结果，带有时间参数 t
                observed = _expm_multiply_simple(A, B, t=t)
                # 使用 sp_expm 计算期望结果，带有时间参数 tA
                expected = np.dot(sp_expm(t*A), B)
                # 断言观察结果与期望结果的近似程度
                assert_allclose(observed, expected)
                # 使用估计函数测试 _expm_multiply_simple 的结果，带有时间参数 t
                observed = estimated(_expm_multiply_simple)(
                    aslinearoperator(A), B, t=t
                )
                # 断言估计结果与期望结果的近似程度
                assert_allclose(observed, expected)

    # 定义一个测试函数，用于测试 _expm_multiply_simple 函数在单个时间点的行为
    def test_scaled_expm_multiply_single_timepoint(self):
        # 设置随机种子，以便结果可重现
        np.random.seed(1234)
        # 定义时间点 t
        t = 0.1
        # 定义矩阵 A 的大小
        n = 5
        # 定义矩阵 B 的列数
        k = 2
        # 生成一个随机矩阵 A
        A = np.random.randn(n, n)
        # 生成一个随机矩阵 B
        B = np.random.randn(n, k)
        # 使用 _expm_multiply_simple 计算结果，带有时间参数 t
        observed = _expm_multiply_simple(A, B, t=t)
        # 使用 sp_expm 计算期望结果，带有时间参数 tA
        expected = sp_expm(t*A).dot(B)
        # 断言观察结果与期望结果的近似程度
        assert_allclose(observed, expected)
        # 使用估计函数测试 _expm_multiply_simple 的结果，带有时间参数 t
        observed = estimated(_expm_multiply_simple)(
            aslinearoperator(A), B, t=t
        )
        # 断言估计结果与期望结果的近似程度
        assert_allclose(observed, expected)
    `
        # 测试稀疏矩阵的指数矩阵乘法
        def test_sparse_expm_multiply(self):
            # 设置随机种子，确保结果可复现
            np.random.seed(1234)
            n = 40  # 矩阵维度
            k = 3   # 乘法维度
            nsamples = 10  # 测试样本数量
            # 循环进行测试样本的生成与验证
            for i in range(nsamples):
                # 生成一个稀疏矩阵 A，密度为 0.05
                A = scipy.sparse.rand(n, n, density=0.05)
                # 生成一个随机矩阵 B，维度为 n x k
                B = np.random.randn(n, k)
                # 计算稀疏矩阵 A 与矩阵 B 的指数矩阵乘法结果
                observed = expm_multiply(A, B)
                # 使用上下文管理器忽略警告，针对特定警告进行过滤
                with suppress_warnings() as sup:
                    sup.filter(SparseEfficiencyWarning,
                               "splu converted its input to CSC format")
                    sup.filter(SparseEfficiencyWarning,
                               "spsolve is more efficient when sparse b is in the"
                               " CSC matrix format")
                    # 计算稀疏矩阵指数函数与矩阵乘法的期望结果
                    expected = sp_expm(A).dot(B)
                # 断言计算结果与期望结果相似
                assert_allclose(observed, expected)
                # 使用估算器对稀疏矩阵 A 转换为线性算子，并计算其指数矩阵乘法结果
                observed = estimated(expm_multiply)(aslinearoperator(A), B)
                # 断言计算结果与期望结果相似
                assert_allclose(observed, expected)
    
        # 测试复数矩阵的指数矩阵乘法
        def test_complex(self):
            # 定义一个复数矩阵 A
            A = np.array([
                [1j, 1j],
                [0, 1j]], dtype=complex)
            # 定义一个复数向量 B
            B = np.array([1j, 1j])
            # 计算复数矩阵 A 与向量 B 的指数矩阵乘法结果
            observed = expm_multiply(A, B)
            # 定义期望结果向量，基于矩阵指数函数的计算
            expected = np.array([
                1j * np.exp(1j) + 1j * (1j*np.cos(1) - np.sin(1)),
                1j * np.exp(1j)], dtype=complex)
            # 断言计算结果与期望结果相似
            assert_allclose(observed, expected)
            # 使用估算器对复数矩阵 A 转换为线性算子，并计算其指数矩阵乘法结果
            observed = estimated(expm_multiply)(aslinearoperator(A), B)
            # 断言计算结果与期望结果相似
            assert_allclose(observed, expected)
    # 定义测试类 TestExpmActionInterval
class TestExpmActionInterval:

    # 使用 pytest 的装饰器标记该方法为 "fail_slow"，最大失败次数为 20 次
    @pytest.mark.fail_slow(20)
    # 定义测试方法 test_sparse_expm_multiply_interval
    def test_sparse_expm_multiply_interval(self):
        # 设置随机数种子为 1234，确保可重复性
        np.random.seed(1234)
        # 设置起始值、终止值、步数和特征值数量
        start = 0.1
        stop = 3.2
        n = 40
        k = 3
        endpoint = True
        # 对于指定的三组数字执行以下循环
        for num in (14, 13, 2):
            # 创建稀疏随机矩阵 A，密度为 0.05
            A = scipy.sparse.rand(n, n, density=0.05)
            # 创建随机矩阵 B 和向量 v
            B = np.random.randn(n, k)
            v = np.random.randn(n)
            # 对于目标 B 和 v 执行以下循环
            for target in (B, v):
                # 使用 expm_multiply 函数计算 A 和 target 的乘积 X
                X = expm_multiply(A, target, start=start, stop=stop,
                                  num=num, endpoint=endpoint)
                # 生成均匀间隔的样本点，用于后续比较
                samples = np.linspace(start=start, stop=stop,
                                      num=num, endpoint=endpoint)
                # 在忽略警告环境中执行以下循环
                with suppress_warnings() as sup:
                    # 过滤 SparseEfficiencyWarning 类型的警告
                    sup.filter(SparseEfficiencyWarning,
                               "splu converted its input to CSC format")
                    sup.filter(SparseEfficiencyWarning,
                               "spsolve is more efficient when sparse b is in"
                               " the CSC matrix format")
                    # 对于每个解决方案和时间点 t 执行断言比较
                    for solution, t in zip(X, samples):
                        assert_allclose(solution, sp_expm(t*A).dot(target))

    # 使用 pytest 的装饰器标记该方法为 "fail_slow"，最大失败次数为 20 次
    @pytest.mark.fail_slow(20)
    # 定义测试方法 test_expm_multiply_interval_vector
    def test_expm_multiply_interval_vector(self):
        # 设置随机数种子为 1234，确保可重复性
        np.random.seed(1234)
        # 设置起始值、终止值和是否包含终点的间隔参数
        interval = {'start': 0.1, 'stop': 3.2, 'endpoint': True}
        # 对于每个 num 和 n 的组合执行以下循环
        for num, n in product([14, 13, 2], [1, 2, 5, 20, 40]):
            # 创建 n x n 的随机逆矩阵 A
            A = scipy.linalg.inv(np.random.randn(n, n))
            # 创建随机向量 v
            v = np.random.randn(n)
            # 生成均匀间隔的样本点，用于后续比较
            samples = np.linspace(num=num, **interval)
            # 使用 expm_multiply 函数计算 A 和 v 的乘积 X
            X = expm_multiply(A, v, num=num, **interval)
            # 对于每个解决方案和时间点 t 执行断言比较
            for solution, t in zip(X, samples):
                assert_allclose(solution, sp_expm(t*A).dot(v))
            # 使用估计函数 estimated(expm_multiply) 计算线性操作数 A 和 v 的乘积 Xguess
            Xguess = estimated(expm_multiply)(aslinearoperator(A), v,
                                              num=num, **interval)
            # 使用给定的迹值计算线性操作数 A 和 v 的乘积 Xgiven
            Xgiven = expm_multiply(aslinearoperator(A), v, num=num, **interval,
                                   traceA=np.trace(A))
            # 使用错误的迹值计算线性操作数 A 和 v 的乘积 Xwrong，测试其鲁棒性
            Xwrong = expm_multiply(aslinearoperator(A), v, num=num, **interval,
                                   traceA=np.trace(A)*5)
            # 对于每个解决方案和时间点 t 执行断言比较
            for sol_guess, sol_given, sol_wrong, t in zip(Xguess, Xgiven,
                                                          Xwrong, samples):
                correct = sp_expm(t*A).dot(v)
                assert_allclose(sol_guess, correct)
                assert_allclose(sol_given, correct)
                assert_allclose(sol_wrong, correct)

    # 使用 pytest 的装饰器标记该方法为 "fail_slow"，最大失败次数为 20 次
    @pytest.mark.fail_slow(20)
    # 定义一个测试方法，用于测试 expm_multiply 函数在不同参数和数据类型下的行为
    def test_expm_multiply_interval_matrix(self):
        # 设定随机种子以确保可重复性
        np.random.seed(1234)
        # 定义一个时间间隔字典
        interval = {'start': 0.1, 'stop': 3.2, 'endpoint': True}
        # 遍历参数组合的笛卡尔积
        for num, n, k in product([14, 13, 2], [1, 2, 5, 20, 40], [1, 2]):
            # 生成随机的 n x n 反矩阵 A
            A = scipy.linalg.inv(np.random.randn(n, n))
            # 生成 n x k 的随机矩阵 B
            B = np.random.randn(n, k)
            # 在指定时间间隔内生成 num 个等间隔的样本点
            samples = np.linspace(num=num, **interval)
            # 调用 expm_multiply 函数计算结果 X
            X = expm_multiply(A, B, num=num, **interval)
            # 遍历 X 中的解和样本点，验证解是否满足条件
            for solution, t in zip(X, samples):
                assert_allclose(solution, sp_expm(t * A).dot(B))
            # 使用 estimated 函数对 expm_multiply 进行估计
            X = estimated(expm_multiply)(aslinearoperator(A), B, num=num,
                                         **interval)
            # 再次验证 X 中的解是否满足条件
            for solution, t in zip(X, samples):
                assert_allclose(solution, sp_expm(t * A).dot(B))

    # 定义一个测试方法，用于测试 expm_multiply 函数在稀疏矩阵和不同数据类型下的行为
    def test_sparse_expm_multiply_interval_dtypes(self):
        # 测试 A 和 B 均为整数类型的情况
        A = scipy.sparse.diags(np.arange(5), format='csr', dtype=int)
        B = np.ones(5, dtype=int)
        Aexpm = scipy.sparse.diags(np.exp(np.arange(5)), format='csr')
        assert_allclose(expm_multiply(A, B, 0, 1)[-1], Aexpm.dot(B))

        # 测试 A 为复数类型，B 为整数类型的情况
        A = scipy.sparse.diags(-1j * np.arange(5), format='csr', dtype=complex)
        B = np.ones(5, dtype=int)
        Aexpm = scipy.sparse.diags(np.exp(-1j * np.arange(5)), format='csr')
        assert_allclose(expm_multiply(A, B, 0, 1)[-1], Aexpm.dot(B))

        # 测试 A 为整数类型，B 为复数类型的情况
        A = scipy.sparse.diags(np.arange(5), format='csr', dtype=int)
        B = np.full(5, 1j, dtype=complex)
        Aexpm = scipy.sparse.diags(np.exp(np.arange(5)), format='csr')
        assert_allclose(expm_multiply(A, B, 0, 1)[-1], Aexpm.dot(B))

    # 定义测试方法，用于测试 expm_multiply 函数在具体状态下的行为
    def test_expm_multiply_interval_status_0(self):
        # 调用辅助方法 _help_test_specific_expm_interval_status，状态值为 0
        self._help_test_specific_expm_interval_status(0)

    # 定义测试方法，用于测试 expm_multiply 函数在具体状态下的行为
    def test_expm_multiply_interval_status_1(self):
        # 调用辅助方法 _help_test_specific_expm_interval_status，状态值为 1
        self._help_test_specific_expm_interval_status(1)

    # 定义测试方法，用于测试 expm_multiply 函数在具体状态下的行为
    def test_expm_multiply_interval_status_2(self):
        # 调用辅助方法 _help_test_specific_expm_interval_status，状态值为 2
        self._help_test_specific_expm_interval_status(2)
    # 定义一个测试函数，用于检查特定状态下的指数矩阵乘法的区间计算
    def _help_test_specific_expm_interval_status(self, target_status):
        # 设置随机数种子，以便结果可重现
        np.random.seed(1234)
        # 初始化区间计算的起始点、终止点和采样点数量
        start = 0.1
        stop = 3.2
        num = 13
        # 是否包含终点
        endpoint = True
        # 设置矩阵的行数和列数
        n = 5
        k = 2
        # 重复实验次数
        nrepeats = 10
        # 记录成功匹配目标状态的次数
        nsuccesses = 0
        # 循环执行多次实验
        for num in [14, 13, 2] * nrepeats:
            # 随机生成 n x n 和 n x k 大小的矩阵 A 和 B
            A = np.random.randn(n, n)
            B = np.random.randn(n, k)
            # 调用 _expm_multiply_interval 函数计算区间内的指数矩阵乘法
            status = _expm_multiply_interval(A, B,
                    start=start, stop=stop, num=num, endpoint=endpoint,
                    status_only=True)
            # 检查计算结果是否与目标状态匹配
            if status == target_status:
                # 如果匹配，再次调用函数获取详细计算结果 X
                X, status = _expm_multiply_interval(A, B,
                        start=start, stop=stop, num=num, endpoint=endpoint,
                        status_only=False)
                # 断言 X 的形状为 (num, n, k)
                assert_equal(X.shape, (num, n, k))
                # 生成从 start 到 stop 的 num 个均匀分布的样本点
                samples = np.linspace(start=start, stop=stop,
                        num=num, endpoint=endpoint)
                # 验证每个解决方案是否满足指定精度要求
                for solution, t in zip(X, samples):
                    assert_allclose(solution, sp_expm(t*A).dot(B))
                # 成功匹配目标状态的次数加一
                nsuccesses += 1
        # 如果没有找到任何匹配目标状态的区间，抛出异常
        if not nsuccesses:
            msg = 'failed to find a status-' + str(target_status) + ' interval'
            raise Exception(msg)
# 使用 pytest 的 parametrize 装饰器为 test_expm_multiply_dtype 函数提供多组参数组合
@pytest.mark.parametrize("dtype_a", DTYPES)
@pytest.mark.parametrize("dtype_b", DTYPES)
@pytest.mark.parametrize("b_is_matrix", [False, True])
def test_expm_multiply_dtype(dtype_a, dtype_b, b_is_matrix):
    """Make sure `expm_multiply` handles all numerical dtypes correctly."""
    
    # 根据 IMPRECISE 中定义的数据类型是否在 dtype_a 或 dtype_b 中，选择性地设置 assert_allclose_ 函数
    assert_allclose_ = (partial(assert_allclose, rtol=1.2e-3, atol=1e-5)
                        if {dtype_a, dtype_b} & IMPRECISE else assert_allclose)
    
    # 使用 numpy 的默认随机数生成器创建 rng 对象
    rng = np.random.default_rng(1234)
    
    # 定义测试数据维度
    n = 7
    b_shape = (n, 3) if b_is_matrix else (n, )
    
    # 根据 dtype_a 是否在 REAL_DTYPES 中选择 A 的类型和生成方法
    if dtype_a in REAL_DTYPES:
        A = scipy.linalg.inv(rng.random([n, n])).astype(dtype_a)
    else:
        A = scipy.linalg.inv(
            rng.random([n, n]) + 1j*rng.random([n, n])
        ).astype(dtype_a)
    
    # 根据 dtype_b 是否在 REAL_DTYPES 中选择 B 的类型和生成方法
    if dtype_b in REAL_DTYPES:
        B = (2*rng.random(b_shape)).astype(dtype_b)
    else:
        B = (rng.random(b_shape) + 1j*rng.random(b_shape)).astype(dtype_b)

    # 单次应用
    sol_mat = expm_multiply(A, B)
    sol_op = estimated(expm_multiply)(aslinearoperator(A), B)
    direct_sol = np.dot(sp_expm(A), B)
    
    # 断言 sol_mat 和 direct_sol 的近似程度
    assert_allclose_(sol_mat, direct_sol)
    assert_allclose_(sol_op, direct_sol)
    
    # 使用 traceA=np.trace(A) 调用 expm_multiply
    sol_op = expm_multiply(aslinearoperator(A), B, traceA=np.trace(A))
    assert_allclose_(sol_op, direct_sol)

    # 对于时间点
    interval = {'start': 0.1, 'stop': 3.2, 'num': 13, 'endpoint': True}
    samples = np.linspace(**interval)
    
    # 使用不同的时间点调用 expm_multiply
    X_mat = expm_multiply(A, B, **interval)
    X_op = estimated(expm_multiply)(aslinearoperator(A), B, **interval)
    
    # 遍历结果并断言每个时间点的 sol_mat 和 sol_op 与 direct_sol 的近似程度
    for sol_mat, sol_op, t in zip(X_mat, X_op, samples):
        direct_sol = sp_expm(t*A).dot(B)
        assert_allclose_(sol_mat, direct_sol)
        assert_allclose_(sol_op, direct_sol)
```