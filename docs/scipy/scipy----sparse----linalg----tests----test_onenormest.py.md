# `D:\src\scipysrc\scipy\scipy\sparse\linalg\tests\test_onenormest.py`

```
"""Test functions for the sparse.linalg._onenormest module
"""

# 导入所需的库和模块
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_
import pytest
import scipy.linalg
import scipy.sparse.linalg
from scipy.sparse.linalg._onenormest import _onenormest_core, _algorithm_2_2

# 定义一个自定义的线性操作符类，继承自scipy.sparse.linalg.LinearOperator
class MatrixProductOperator(scipy.sparse.linalg.LinearOperator):
    """
    This is purely for onenormest testing.
    """

    # 初始化方法，接受两个矩阵 A 和 B
    def __init__(self, A, B):
        # 检查 A 和 B 是否是二维数组
        if A.ndim != 2 or B.ndim != 2:
            raise ValueError('expected ndarrays representing matrices')
        # 检查 A 和 B 的形状是否兼容
        if A.shape[1] != B.shape[0]:
            raise ValueError('incompatible shapes')
        self.A = A
        self.B = B
        self.ndim = 2
        self.shape = (A.shape[0], B.shape[1])

    # 矩阵向量乘法操作
    def _matvec(self, x):
        return np.dot(self.A, np.dot(self.B, x))

    # 反向矩阵向量乘法操作
    def _rmatvec(self, x):
        return np.dot(np.dot(x, self.A), self.B)

    # 矩阵乘法操作
    def _matmat(self, X):
        return np.dot(self.A, np.dot(self.B, X))

    # 返回转置操作
    @property
    def T(self):
        return MatrixProductOperator(self.B.T, self.A.T)


# 测试类 TestOnenormest
class TestOnenormest:

    # 标记为慢速测试
    @pytest.mark.xslow
    def test_onenormest_table_3_t_2(self):
        # 此测试如果计算机性能较慢，可能需要数秒钟
        # 由于是随机性质，所以公差可能会过于严格

        # 设置随机种子
        np.random.seed(1234)

        # 定义变量 t、n、itmax 和 nsamples
        t = 2
        n = 100
        itmax = 5
        nsamples = 5000

        # 初始化存储观测值和期望值的列表
        observed = []
        expected = []
        nmult_list = []
        nresample_list = []

        # 循环进行多次采样
        for i in range(nsamples):
            # 生成随机矩阵 A，并求其逆矩阵
            A = scipy.linalg.inv(np.random.randn(n, n))

            # 调用 _onenormest_core 函数进行估计
            est, v, w, nmults, nresamples = _onenormest_core(A, A.T, t, itmax)

            # 将估计值、期望值以及其他统计量存入相应的列表中
            observed.append(est)
            expected.append(scipy.linalg.norm(A, 1))
            nmult_list.append(nmults)
            nresample_list.append(nresamples)

        # 转换为 numpy 数组
        observed = np.array(observed, dtype=float)
        expected = np.array(expected, dtype=float)

        # 计算相对误差
        relative_errors = np.abs(observed - expected) / expected

        # 检查平均低估比率
        underestimation_ratio = observed / expected
        assert_(0.99 < np.mean(underestimation_ratio) < 1.0)

        # 检查最大和平均所需的列重新采样次数
        assert_equal(np.max(nresample_list), 2)
        assert_(0.05 < np.mean(nresample_list) < 0.2)

        # 检查准确计算的规范比例
        nexact = np.count_nonzero(relative_errors < 1e-14)
        proportion_exact = nexact / float(nsamples)
        assert_(0.9 < proportion_exact < 0.95)

        # 检查平均矩阵*向量乘法次数
        assert_(3.5 < np.mean(nmult_list) < 4.5)

    # 标记为慢速测试
    @pytest.mark.xslow
    def test_onenormest_table_4_t_7(self):
        # 如果你的计算机像我的一样慢，这段代码可能需要几秒钟执行。
        # 由于其是随机的，所以容差可能过于严格。
        np.random.seed(1234)  # 使用种子1234设置随机数生成器的种子
        t = 7  # 设置 t 的值为 7
        n = 100  # 设置 n 的值为 100
        itmax = 5  # 设置 itmax 的值为 5
        nsamples = 5000  # 设置 nsamples 的值为 5000
        observed = []  # 初始化空列表 observed，用于存储观察到的估计值
        expected = []  # 初始化空列表 expected，用于存储期望值
        nmult_list = []  # 初始化空列表 nmult_list，用于存储矩阵乘向量的次数
        nresample_list = []  # 初始化空列表 nresample_list，用于存储重采样次数
        for i in range(nsamples):  # 开始循环，执行 nsamples 次
            A = np.random.randint(-1, 2, size=(n, n))  # 生成一个 n x n 的随机整数矩阵 A
            est, v, w, nmults, nresamples = _onenormest_core(A, A.T, t, itmax)  # 调用 _onenormest_core 函数进行估计
            observed.append(est)  # 将估计值 est 添加到 observed 列表中
            expected.append(scipy.linalg.norm(A, 1))  # 将 A 的 1-范数作为期望值添加到 expected 列表中
            nmult_list.append(nmults)  # 将矩阵乘向量次数 nmults 添加到 nmult_list 列表中
            nresample_list.append(nresamples)  # 将重采样次数 nresamples 添加到 nresample_list 列表中
        observed = np.array(observed, dtype=float)  # 将 observed 转换为 numpy 数组，数据类型为 float
        expected = np.array(expected, dtype=float)  # 将 expected 转换为 numpy 数组，数据类型为 float
        relative_errors = np.abs(observed - expected) / expected  # 计算相对误差

        # 检查平均低估比率
        underestimation_ratio = observed / expected  # 计算低估比率
        assert_(0.90 < np.mean(underestimation_ratio) < 0.99)  # 断言平均低估比率在0.90和0.99之间

        # 检查所需的列重采样
        assert_equal(np.max(nresample_list), 0)  # 断言重采样列表中的最大值为0

        # 检查精确计算的范数比例
        nexact = np.count_nonzero(relative_errors < 1e-14)  # 计算相对误差小于1e-14的精确计数
        proportion_exact = nexact / float(nsamples)  # 计算精确比例
        assert_(0.15 < proportion_exact < 0.25)  # 断言精确比例在0.15和0.25之间

        # 检查平均矩阵乘向量乘法次数
        assert_(3.5 < np.mean(nmult_list) < 4.5)  # 断言平均矩阵乘向量乘法次数在3.5和4.5之间

    def test_onenormest_table_5_t_1(self):
        # "请注意，此处没有随机性，因此 t=1 时只有一个估计值"
        t = 1  # 设置 t 的值为 1
        n = 100  # 设置 n 的值为 100
        itmax = 5  # 设置 itmax 的值为 5
        alpha = 1 - 1e-6  # 设置 alpha 的值为 1 - 1e-6
        A = -scipy.linalg.inv(np.identity(n) + alpha*np.eye(n, k=1))  # 计算 A 矩阵
        first_col = np.array([1] + [0]*(n-1))  # 创建第一列
        first_row = np.array([(-alpha)**i for i in range(n)])  # 创建第一行
        B = -scipy.linalg.toeplitz(first_col, first_row)  # 创建 B 矩阵
        assert_allclose(A, B)  # 断言 A 和 B 在数值上接近

        # 调用 _onenormest_core 函数进行估计
        est, v, w, nmults, nresamples = _onenormest_core(B, B.T, t, itmax)
        exact_value = scipy.linalg.norm(B, 1)  # 计算 B 的 1-范数作为精确值
        underest_ratio = est / exact_value  # 计算估计比率
        assert_allclose(underest_ratio, 0.05, rtol=1e-4)  # 断言估计比率在0.05左右，相对误差小于1e-4
        assert_equal(nmults, 11)  # 断言矩阵乘向量乘法次数为11
        assert_equal(nresamples, 0)  # 断言重采样次数为0

        # 检查非下划线版本的 onenormest 函数
        est_plain = scipy.sparse.linalg.onenormest(B, t=t, itmax=itmax)  # 调用稀疏矩阵的 onenormest 函数
        assert_allclose(est, est_plain)  # 断言两种估计值在数值上接近

    @pytest.mark.xslow
    # 定义一个测试方法，用于验证单范数估计在特定条件下的表现
    def test_onenormest_table_6_t_1(self):
        # TODO 这个测试似乎给出了与表格匹配的估计结果，
        # TODO 尽管在处理单范数估计中的复数时并未尝试。
        # 这段代码会花费较长时间，如果你的电脑像我的一样较慢。
        # 由于其随机性，公差可能过于严格。
        
        # 设置随机种子，以便结果可复现
        np.random.seed(1234)
        # 初始化参数
        t = 1  # t 值设为 1
        n = 100  # 矩阵维度设为 100
        itmax = 5  # 最大迭代次数设为 5
        nsamples = 5000  # 样本数设为 5000
        observed = []  # 观察到的单范数估计值列表
        expected = []  # 期望的单范数值列表
        nmult_list = []  # 矩阵乘法次数列表
        nresample_list = []  # 列重采样次数列表
        
        # 开始循环进行多次样本计算
        for i in range(nsamples):
            # 生成具有复数部分的随机矩阵 A_inv
            A_inv = np.random.rand(n, n) + 1j * np.random.rand(n, n)
            # 计算 A_inv 的逆矩阵 A
            A = scipy.linalg.inv(A_inv)
            # 调用 _onenormest_core 方法进行单范数估计
            est, v, w, nmults, nresamples = _onenormest_core(A, A.T, t, itmax)
            # 将估计值、期望值、矩阵乘法次数、列重采样次数加入对应列表
            observed.append(est)
            expected.append(scipy.linalg.norm(A, 1))
            nmult_list.append(nmults)
            nresample_list.append(nresamples)
        
        # 将列表转换为 NumPy 数组
        observed = np.array(observed, dtype=float)
        expected = np.array(expected, dtype=float)
        
        # 计算相对误差
        relative_errors = np.abs(observed - expected) / expected
        
        # 检查平均低估比率
        underestimation_ratio = observed / expected
        underestimation_ratio_mean = np.mean(underestimation_ratio)
        assert_(0.90 < underestimation_ratio_mean < 0.99)
        
        # 检查所需的列重采样次数
        max_nresamples = np.max(nresample_list)
        assert_equal(max_nresamples, 0)
        
        # 检查准确计算的单范数比例
        nexact = np.count_nonzero(relative_errors < 1e-14)
        proportion_exact = nexact / float(nsamples)
        assert_(0.7 < proportion_exact < 0.8)
        
        # 检查平均矩阵*向量乘法次数
        mean_nmult = np.mean(nmult_list)
        assert_(4 < mean_nmult < 5)

    # 辅助方法，用于慢速计算矩阵乘积的一范数
    def _help_product_norm_slow(self, A, B):
        # 使用 NumPy 的 dot 方法计算矩阵乘积 C
        C = np.dot(A, B)
        # 返回矩阵 C 的一范数
        return scipy.linalg.norm(C, 1)

    # 辅助方法，用于快速计算矩阵乘积的一范数
    def _help_product_norm_fast(self, A, B):
        # 设置 t 和 itmax 的值
        t = 2
        itmax = 5
        # 使用 MatrixProductOperator 构建操作符 D
        D = MatrixProductOperator(A, B)
        # 调用 _onenormest_core 方法进行单范数估计
        est, v, w, nmults, nresamples = _onenormest_core(D, D.T, t, itmax)
        # 返回估计的一范数值
        return est

    # 标记为慢速测试
    @pytest.mark.slow
    # 定义一个测试方法，用于验证线性操作符的单范数估计
    def test_onenormest_linear_operator(self):
        # 通过其乘积 A B 定义一个矩阵。
        # 根据 A 和 B 的形状，可能很容易将这个乘积与一个小矩阵相乘，
        # 但逐个检查乘积的所有条目可能会很麻烦。
        
        # 设置随机种子，以便结果可复现
        np.random.seed(1234)
        # 初始化参数
        n = 6000  # 矩阵维度设为 6000
        k = 3  # 矩阵维度设为 3
        A = np.random.randn(n, k)  # 生成随机矩阵 A
        B = np.random.randn(k, n)  # 生成随机矩阵 B
        
        # 调用 _help_product_norm_fast 方法进行快速计算的单范数估计
        fast_estimate = self._help_product_norm_fast(A, B)
        # 调用 _help_product_norm_slow 方法进行慢速计算的单范数估计
        exact_value = self._help_product_norm_slow(A, B)
        
        # 断言快速估计值应小于等于精确值的三倍
        assert_(fast_estimate <= exact_value <= 3 * fast_estimate,
                f'fast: {fast_estimate:g}\nexact:{exact_value:g}')
    # 定义一个测试方法，验证函数返回值的正确性
    def test_returns(self):
        # 设定随机数种子，确保结果可重复
        np.random.seed(1234)
        # 生成一个稀疏随机矩阵 A，大小为 50x50，稀疏度为 0.1
        A = scipy.sparse.rand(50, 50, 0.1)

        # 计算 A 矩阵的 1-范数（所有列的绝对值之和的最大值）
        s0 = scipy.linalg.norm(A.toarray(), 1)
        # 使用 sparse.linalg.onenormest 计算 A 矩阵的 1-范数，并返回 1-范数值 s1 和特征向量 v
        s1, v = scipy.sparse.linalg.onenormest(A, compute_v=True)
        # 使用 sparse.linalg.onenormest 计算 A 矩阵的 1-范数，并返回 1-范数值 s2 和特征向量 w
        s2, w = scipy.sparse.linalg.onenormest(A, compute_w=True)
        # 使用 sparse.linalg.onenormest 计算 A 矩阵的 1-范数，并返回 1-范数值 s3、特征向量 v2 和特征向量 w2
        s3, v2, w2 = scipy.sparse.linalg.onenormest(A, compute_w=True, compute_v=True)

        # 断言 s1 与 s0 在指定的相对误差范围内相等
        assert_allclose(s1, s0, rtol=1e-9)
        # 断言 A 矩阵乘以特征向量 v 的结果的 1-范数与 s0 乘以 v 的 1-范数的结果在指定相对误差范围内相等
        assert_allclose(np.linalg.norm(A.dot(v), 1), s0*np.linalg.norm(v, 1), rtol=1e-9)
        # 断言 A 矩阵乘以特征向量 v 的结果与特征向量 w 在指定相对误差范围内相等
        assert_allclose(A.dot(v), w, rtol=1e-9)
class TestAlgorithm_2_2:

    def test_randn_inv(self):
        np.random.seed(1234)  # 设置随机种子为1234，保证随机数可复现性
        n = 20  # 初始化变量 n 为20，表示矩阵的大小为 n x n
        nsamples = 100  # 设定采样次数为100次
        for i in range(nsamples):
            # 从整数范围 [1, 4) 中均匀随机选择一个整数 t
            t = np.random.randint(1, 4)

            # 从整数范围 [10, 41) 中均匀随机选择一个整数 n
            n = np.random.randint(10, 41)

            # 生成一个 n x n 大小的矩阵，其元素为从标准正态分布中随机抽取的数值，并计算其逆矩阵
            A = scipy.linalg.inv(np.random.randn(n, n))

            # 调用 _algorithm_2_2 函数，传入 A 和 A 的转置矩阵 A.T，以及之前随机选择的整数 t
            g, ind = _algorithm_2_2(A, A.T, t)
```