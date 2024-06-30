# `D:\src\scipysrc\scipy\scipy\linalg\tests\test_solvers.py`

```
import os  # 导入操作系统模块
import numpy as np  # 导入NumPy库

from numpy.testing import assert_array_almost_equal, assert_allclose  # 导入NumPy测试相关函数
import pytest  # 导入pytest测试框架
from pytest import raises as assert_raises  # 导入pytest中的raises函数，重命名为assert_raises

from scipy.linalg import solve_sylvester  # 导入SciPy线性代数模块中的solve_sylvester函数
from scipy.linalg import solve_continuous_lyapunov, solve_discrete_lyapunov  # 导入SciPy线性代数模块中的Lyapunov方程求解函数
from scipy.linalg import solve_continuous_are, solve_discrete_are  # 导入SciPy线性代数模块中的Riccati方程求解函数
from scipy.linalg import block_diag, solve, LinAlgError  # 导入SciPy线性代数模块中的其他函数和异常类
from scipy.sparse._sputils import matrix  # 导入SciPy稀疏矩阵相关模块中的matrix函数


def _load_data(name):
    """
    Load npz data file under data/
    Returns a copy of the data, rather than keeping the npz file open.
    """
    filename = os.path.join(os.path.abspath(os.path.dirname(__file__)),  # 构建数据文件的完整路径
                            'data', name)
    with np.load(filename) as f:  # 使用NumPy加载数据文件
        return dict(f.items())  # 将加载的数据作为字典返回


class TestSolveLyapunov:
    
    def test_continuous_squareness_and_shape(self):
        nsq = np.ones((3, 2))  # 创建一个3x2的全1数组
        sq = np.eye(3)  # 创建一个3x3的单位矩阵
        assert_raises(ValueError, solve_continuous_lyapunov, nsq, sq)  # 断言调用solve_continuous_lyapunov会抛出ValueError异常
        assert_raises(ValueError, solve_continuous_lyapunov, sq, nsq)  # 断言调用solve_continuous_lyapunov会抛出ValueError异常
        assert_raises(ValueError, solve_continuous_lyapunov, sq, np.eye(2))  # 断言调用solve_continuous_lyapunov会抛出ValueError异常

    def check_continuous_case(self, a, q):
        x = solve_continuous_lyapunov(a, q)  # 解连续Lyapunov方程
        assert_array_almost_equal(  # 断言计算结果是否与期望值近似相等
                          np.dot(a, x) + np.dot(x, a.conj().transpose()), q)

    def check_discrete_case(self, a, q, method=None):
        x = solve_discrete_lyapunov(a, q, method=method)  # 解离散Lyapunov方程
        assert_array_almost_equal(  # 断言计算结果是否与期望值近似相等
                      np.dot(np.dot(a, x), a.conj().transpose()) - x, -1.0*q)

    def test_cases(self):
        for case in self.cases:  # 遍历测试用例列表
            self.check_continuous_case(case[0], case[1])  # 调用连续情况测试函数
            self.check_discrete_case(case[0], case[1])  # 调用离散情况测试函数
            self.check_discrete_case(case[0], case[1], method='direct')  # 调用离散情况测试函数（指定方法为'direct'）
            self.check_discrete_case(case[0], case[1], method='bilinear')  # 调用离散情况测试函数（指定方法为'bilinear')


class TestSolveContinuousAre:
    mat6 = _load_data('carex_6_data.npz')  # 加载名称为'carex_6_data.npz'的数据文件
    mat15 = _load_data('carex_15_data.npz')  # 加载名称为'carex_15_data.npz'的数据文件
    mat18 = _load_data('carex_18_data.npz')  # 加载名称为'carex_18_data.npz'的数据文件
    mat19 = _load_data('carex_19_data.npz')  # 加载名称为'carex_19_data.npz'的数据文件
    mat20 = _load_data('carex_20_data.npz')  # 加载名称为'carex_20_data.npz'的数据文件
    
    # 定义用于测试的最小精度要求列表，每个条目表示解x插入方程后与零矩阵的小数点位数要求
    # res = array([[8e-3,1e-16],[1e-16,1e-20]]) --> min_decimal[k] = 2
    # 如果测试失败，使用"None"表示该条目
    min_decimal = (14, 12, 13, 14, 11, 6, None, 5, 7, 14, 14,
                   None, 9, 14, 13, 14, None, 12, None, None)

    @pytest.mark.parametrize("j, case", enumerate(cases))  # 参数化测试用例
    # 定义一个测试方法，用于验证是否满足连续代数Riccati方程 0 = XA + A'X - XB(R)^{-1}B'X + Q
    def test_solve_continuous_are(self, j, case):
        """Checks if 0 = XA + A'X - XB(R)^{-1} B'X + Q is true"""
        # 从测试用例中获取参数 a, b, q, r, knownfailure
        a, b, q, r, knownfailure = case
        # 如果 knownfailure 为 True，则标记测试为预期失败状态
        if knownfailure:
            pytest.xfail(reason=knownfailure)

        # 获取测试精度
        dec = self.min_decimal[j]
        # 调用 solve_continuous_are 函数解决连续时间的代数Riccati方程，返回结果矩阵 X
        x = solve_continuous_are(a, b, q, r)
        # 计算 Riccati 方程左边的部分：XA + A'X + Q
        res = x @ a + a.conj().T @ x + q
        # 计算 Riccati 方程右边的部分：XB(R)^{-1}B'X
        out_fact = x @ b
        res -= out_fact @ solve(np.atleast_2d(r), out_fact.conj().T)
        # 使用 assert_array_almost_equal 函数验证 res 是否接近零矩阵，精度为 dec
        assert_array_almost_equal(res, np.zeros_like(res), decimal=dec)
# 定义一个测试类 TestSolveDiscreteAre，用于测试离散时间代数Riccati方程的解算函数
class TestSolveDiscreteAre:

    # 定义最小精度要求，每个元素表示在方程解 x 插入到方程中时，结果与零矩阵的小数位数匹配
    # 如果测试失败，对应的条目使用 "None"
    min_decimal = (12, 14, 13, 14, 13, 16, 18, 14, 14, 13,
                   14, 13, 13, 14, 12, 2, 4, 6, 10)
    
    # 计算每个最小精度对应的最大容差
    max_tol = [1.5 * 10**-ind for ind in min_decimal]

    # gh-18012 提升到 OpenBLAS 后的放宽容差
    max_tol[11] = 2.5e-13

    # gh-20335 中针对 Cirrus 上 Linux-aarch64 构建的放宽容差
    max_tol[15] = 2.0e-2

    # gh-20335 中针对 Ubuntu Jammy 上 OpenBLAS 3.20 的放宽容差，对于 OpenBLAS 3.26 不需要提升
    max_tol[16] = 2.0e-4

    # 使用 pytest.mark.parametrize 注册参数化测试用例
    @pytest.mark.parametrize("j, case", enumerate(cases))
    def test_solve_discrete_are(self, j, case):
        """检查是否满足 X = A'XA-(A'XB)(R+B'XB)^-1(B'XA)+Q 的条件"""
        # 从参数中提取测试数据 a, b, q, r, knownfailure
        a, b, q, r, knownfailure = case
        
        # 如果测试失败，标记为 xfail
        if knownfailure:
            pytest.xfail(reason=knownfailure)

        # 获取当前测试序号 j 对应的最大容差
        atol = self.max_tol[j]

        # 调用 solve_discrete_are 函数计算方程的解 x
        x = solve_discrete_are(a, b, q, r)
        
        # 计算 b 的共轭转置
        bH = b.conj().T
        
        # 计算 xa 和 xb
        xa, xb = x @ a, x @ b
        
        # 计算方程左侧的结果 res
        res = a.conj().T @ xa - x + q
        res -= a.conj().T @ xb @ (solve(r + bH @ xb, bH) @ xa)
        
        # 使用 assert_allclose 检查 res 是否接近零矩阵，容差为 atol
        assert_allclose(res, np.zeros_like(res), atol=atol)

    def test_infeasible(self):
        # 从 https://arxiv.org/abs/1505.04861v1 中提取一个不可行的例子
        A = np.triu(np.ones((3, 3)))
        A[0, 1] = -1
        B = np.array([[1, 1, 0], [0, 0, 1]]).T
        Q = np.full_like(A, -2) + np.diag([8, -1, -1.9])
        R = np.diag([-10, 0.1])
        
        # 使用 assert_raises 检查 solve_continuous_are 函数是否抛出 LinAlgError 异常
        assert_raises(LinAlgError, solve_continuous_are, A, B, Q, R)
    cases = [
        # 定义测试用例列表，每个元素是一个元组，包含多个输入参数和一个可选的错误信息
        # 第一个测试用例
        (np.array([[2.769230e-01, 8.234578e-01, 9.502220e-01],
                   [4.617139e-02, 6.948286e-01, 3.444608e-02],
                   [9.713178e-02, 3.170995e-01, 4.387444e-01]]),  # 参数 a
         np.array([[3.815585e-01, 1.868726e-01],                   # 参数 b
                   [7.655168e-01, 4.897644e-01],
                   [7.951999e-01, 4.455862e-01]]),
         np.eye(3),                                               # 参数 q
         np.eye(2),                                               # 参数 r
         np.array([[6.463130e-01, 2.760251e-01, 1.626117e-01],     # 参数 e
                   [7.093648e-01, 6.797027e-01, 1.189977e-01],
                   [7.546867e-01, 6.550980e-01, 4.983641e-01]]),
         np.zeros((3, 2)),                                         # 参数 s
         None),                                                    # 可能的错误信息

        # 第二个测试用例，与第一个测试用例不同的地方在于参数 s
        (np.array([[2.769230e-01, 8.234578e-01, 9.502220e-01],
                   [4.617139e-02, 6.948286e-01, 3.444608e-02],
                   [9.713178e-02, 3.170995e-01, 4.387444e-01]]),
         np.array([[3.815585e-01, 1.868726e-01],
                   [7.655168e-01, 4.897644e-01],
                   [7.951999e-01, 4.455862e-01]]),
         np.eye(3),
         np.eye(2),
         np.array([[6.463130e-01, 2.760251e-01, 1.626117e-01],
                   [7.093648e-01, 6.797027e-01, 1.189977e-01],
                   [7.546867e-01, 6.550980e-01, 4.983641e-01]]),
         np.ones((3, 2)),                                          # 参数 s 不同
         None)
    ]

    # 定义最小的小数位数，用于测试精度
    min_decimal = (10, 10)

    def _test_factory(case, dec):
        """检查是否满足特定方程式 X = A'XA-(A'XB)(R+B'XB)^-1(B'XA)+Q)"""
        # 解析测试用例的参数
        a, b, q, r, e, s, knownfailure = case
        
        # 如果存在已知的失败情况，则标记为 xfail
        if knownfailure:
            pytest.xfail(reason=knownfailure)

        # 调用 solve_continuous_are 函数计算结果 x
        x = solve_continuous_are(a, b, q, r, e, s)
        
        # 计算方程式的左侧部分 res
        res = a.conj().T.dot(x.dot(e)) + e.conj().T.dot(x.dot(a)) + q
        
        # 计算方程式的输出项 out_fact
        out_fact = e.conj().T.dot(x).dot(b) + s
        
        # 调用 solve 函数求解并更新 res
        res -= out_fact.dot(solve(np.atleast_2d(r), out_fact.conj().T))
        
        # 使用 assert 检查 res 是否接近零矩阵
        assert_array_almost_equal(res, np.zeros_like(res), decimal=dec)

    # 遍历测试用例列表，并依次调用 _test_factory 函数
    for ind, case in enumerate(cases):
        _test_factory(case, min_decimal[ind])
# 定义一个测试函数，用于测试 solve_discrete_are 函数的泛化离散代数里程碑解决方案。
def test_solve_generalized_discrete_are():
    # 加载日期为 20170120 的数据文件 gendare_20170120_data.npz 的数据
    mat20170120 = _load_data('gendare_20170120_data.npz')

    # 定义多个测试案例
    cases = [
        # 第一个案例：两个随机示例，只有 s 项不同
        (
            np.array([[2.769230e-01, 8.234578e-01, 9.502220e-01],
                      [4.617139e-02, 6.948286e-01, 3.444608e-02],
                      [9.713178e-02, 3.170995e-01, 4.387444e-01]]),
            np.array([[3.815585e-01, 1.868726e-01],
                      [7.655168e-01, 4.897644e-01],
                      [7.951999e-01, 4.455862e-01]]),
            np.eye(3),
            np.eye(2),
            np.array([[6.463130e-01, 2.760251e-01, 1.626117e-01],
                      [7.093648e-01, 6.797027e-01, 1.189977e-01],
                      [7.546867e-01, 6.550980e-01, 4.983641e-01]]),
            np.zeros((3, 2)),
            None
        ),
        # 第二个案例：与第一个案例相似，但 s 项为全 1 矩阵
        (
            np.array([[2.769230e-01, 8.234578e-01, 9.502220e-01],
                      [4.617139e-02, 6.948286e-01, 3.444608e-02],
                      [9.713178e-02, 3.170995e-01, 4.387444e-01]]),
            np.array([[3.815585e-01, 1.868726e-01],
                      [7.655168e-01, 4.897644e-01],
                      [7.951999e-01, 4.455862e-01]]),
            np.eye(3),
            np.eye(2),
            np.array([[6.463130e-01, 2.760251e-01, 1.626117e-01],
                      [7.093648e-01, 6.797027e-01, 1.189977e-01],
                      [7.546867e-01, 6.550980e-01, 4.983641e-01]]),
            np.ones((3, 2)),
            None
        ),
        # 第三个案例：根据用户报告，进行测试，其中 E 为 None，但提供了 S
        (
            mat20170120['A'],
            mat20170120['B'],
            mat20170120['Q'],
            mat20170120['R'],
            None,
            mat20170120['S'],
            None
        )
    ]

    # 定义精度容差
    max_atol = (1.5e-11, 1.5e-11, 3.5e-16)

    # 定义内部测试函数，用于每个案例的测试
    def _test_factory(case, atol):
        """检查是否满足 X = A'XA-(A'XB)(R+B'XB)^-1(B'XA)+Q) 的条件"""
        a, b, q, r, e, s, knownfailure = case
        
        # 如果有已知的失败情况，标记为 pytest 的 xfail
        if knownfailure:
            pytest.xfail(reason=knownfailure)
        
        # 调用 solve_discrete_are 函数计算 X
        x = solve_discrete_are(a, b, q, r, e, s)
        
        # 如果 E 是 None，则设为单位矩阵
        if e is None:
            e = np.eye(a.shape[0])
        
        # 如果 S 是 None，则设为与 b 维度相同的全零矩阵
        if s is None:
            s = np.zeros_like(b)
        
        # 计算结果 res
        res = a.conj().T.dot(x.dot(a)) - e.conj().T.dot(x.dot(e)) + q
        res -= (a.conj().T.dot(x.dot(b)) + s).dot(
                    solve(r+b.conj().T.dot(x.dot(b)),
                          (b.conj().T.dot(x.dot(a)) + s.conj().T)
                          )
                )
        
        # 使用 assert_allclose 检查 res 是否接近全零矩阵
        assert_allclose(res, np.zeros_like(res), atol=atol)

    # 对每个案例调用 _test_factory 函数进行测试
    for ind, case in enumerate(cases):
        _test_factory(case, max_atol[ind])
    def test_square_shape():
        nsq = np.ones((3, 2))  # 创建一个 3x2 的全一数组 nsq
        sq = np.eye(3)  # 创建一个 3x3 的单位矩阵 sq
        for x in (solve_continuous_are, solve_discrete_are):
            assert_raises(ValueError, x, nsq, 1, 1, 1)  # 调用 x 函数，预期抛出 ValueError 异常
            assert_raises(ValueError, x, sq, sq, nsq, 1)  # 调用 x 函数，预期抛出 ValueError 异常
            assert_raises(ValueError, x, sq, sq, sq, nsq)  # 调用 x 函数，预期抛出 ValueError 异常
            assert_raises(ValueError, x, sq, sq, sq, sq, nsq)  # 调用 x 函数，预期抛出 ValueError 异常

    def test_compatible_sizes():
        nsq = np.ones((3, 2))  # 创建一个 3x2 的全一数组 nsq
        sq = np.eye(4)  # 创建一个 4x4 的单位矩阵 sq
        for x in (solve_continuous_are, solve_discrete_are):
            assert_raises(ValueError, x, sq, nsq, 1, 1)  # 调用 x 函数，预期抛出 ValueError 异常
            assert_raises(ValueError, x, sq, sq, sq, sq, sq, nsq)  # 调用 x 函数，预期抛出 ValueError 异常
            assert_raises(ValueError, x, sq, sq, np.eye(3), sq)  # 调用 x 函数，预期抛出 ValueError 异常
            assert_raises(ValueError, x, sq, sq, sq, np.eye(3))  # 调用 x 函数，预期抛出 ValueError 异常
            assert_raises(ValueError, x, sq, sq, sq, sq, np.eye(3))  # 调用 x 函数，预期抛出 ValueError 异常

    def test_symmetry():
        nsym = np.arange(9).reshape(3, 3)  # 创建一个 3x3 的数组 nsym
        sym = np.eye(3)  # 创建一个 3x3 的单位矩阵 sym
        for x in (solve_continuous_are, solve_discrete_are):
            assert_raises(ValueError, x, sym, sym, nsym, sym)  # 调用 x 函数，预期抛出 ValueError 异常
            assert_raises(ValueError, x, sym, sym, sym, nsym)  # 调用 x 函数，预期抛出 ValueError 异常

    def test_singularity():
        sing = np.full((3, 3), 1e12)  # 创建一个 3x3 的所有元素为 1e12 的数组 sing
        sing[2, 2] -= 1  # 修改 sing 数组的特定位置值
        sq = np.eye(3)  # 创建一个 3x3 的单位矩阵 sq
        for x in (solve_continuous_are, solve_discrete_are):
            assert_raises(ValueError, x, sq, sq, sq, sq, sing)  # 调用 x 函数，预期抛出 ValueError 异常

        assert_raises(ValueError, solve_continuous_are, sq, sq, sq, sing)  # 调用 solve_continuous_are 函数，预期抛出 ValueError 异常

    def test_finiteness():
        nm = np.full((2, 2), np.nan)  # 创建一个 2x2 的所有元素为 NaN 的数组 nm
        sq = np.eye(2)  # 创建一个 2x2 的单位矩阵 sq
        for x in (solve_continuous_are, solve_discrete_are):
            assert_raises(ValueError, x, nm, sq, sq, sq)  # 调用 x 函数，预期抛出 ValueError 异常
            assert_raises(ValueError, x, sq, nm, sq, sq)  # 调用 x 函数，预期抛出 ValueError 异常
            assert_raises(ValueError, x, sq, sq, nm, sq)  # 调用 x 函数，预期抛出 ValueError 异常
            assert_raises(ValueError, x, sq, sq, sq, nm)  # 调用 x 函数，预期抛出 ValueError 异常
            assert_raises(ValueError, x, sq, sq, sq, sq, nm)  # 调用 x 函数，预期抛出 ValueError 异常
            assert_raises(ValueError, x, sq, sq, sq, sq, sq, nm)  # 调用 x 函数，预期抛出 ValueError 异常
class TestSolveSylvester:

    cases = [
        # a, b, c all real.
        (np.array([[1, 2], [0, 4]]),
         np.array([[5, 6], [0, 8]]),
         np.array([[9, 10], [11, 12]])),
        # a, b, c all real, 4x4. a and b have non-trival 2x2 blocks in their
        # quasi-triangular form.
        (np.array([[1.0, 0, 0, 0],
                   [0, 1.0, 2.0, 0.0],
                   [0, 0, 3.0, -4],
                   [0, 0, 2, 5]]),
         np.array([[2.0, 0, 0, 1.0],
                   [0, 1.0, 0.0, 0.0],
                   [0, 0, 1.0, -1],
                   [0, 0, 1, 1]]),
         np.array([[1.0, 0, 0, 0],
                   [0, 1.0, 0, 0],
                   [0, 0, 1.0, 0],
                   [0, 0, 0, 1.0]])),
        # a, b, c all complex.
        (np.array([[1.0+1j, 2.0], [3.0-4.0j, 5.0]]),
         np.array([[-1.0, 2j], [3.0, 4.0]]),
         np.array([[2.0-2j, 2.0+2j], [-1.0-1j, 2.0]])),
        # a and b real; c complex.
        (np.array([[1.0, 2.0], [3.0, 5.0]]),
         np.array([[-1.0, 0], [3.0, 4.0]]),
         np.array([[2.0-2j, 2.0+2j], [-1.0-1j, 2.0]])),
        # a and c complex; b real.
        (np.array([[1.0+1j, 2.0], [3.0-4.0j, 5.0]]),
         np.array([[-1.0, 0], [3.0, 4.0]]),
         np.array([[2.0-2j, 2.0+2j], [-1.0-1j, 2.0]])),
        # a complex; b and c real.
        (np.array([[1.0+1j, 2.0], [3.0-4.0j, 5.0]]),
         np.array([[-1.0, 0], [3.0, 4.0]]),
         np.array([[2.0, 2.0], [-1.0, 2.0]])),
        # not square matrices, real
        (np.array([[8, 1, 6], [3, 5, 7], [4, 9, 2]]),
         np.array([[2, 3], [4, 5]]),
         np.array([[1, 2], [3, 4], [5, 6]])),
        # not square matrices, complex
        (np.array([[8, 1j, 6+2j], [3, 5, 7], [4, 9, 2]]),
         np.array([[2, 3], [4, 5-1j]]),
         np.array([[1, 2j], [3, 4j], [5j, 6+7j]])),
    ]

    # 检查解决 Sylvester 方程的测试用例
    def check_case(self, a, b, c):
        # 调用 solve_sylvester 函数求解 Sylvester 方程
        x = solve_sylvester(a, b, c)
        # 断言解 x 满足 Sylvester 方程的定义
        assert_array_almost_equal(np.dot(a, x) + np.dot(x, b), c)

    # 测试所有预定义的测试用例
    def test_cases(self):
        for case in self.cases:
            self.check_case(case[0], case[1], case[2])

    # 测试特殊情况：矩阵 a 是单位矩阵，b 是 1x1 矩阵，c 是 2 维向量
    def test_trivial(self):
        a = np.array([[1.0, 0.0], [0.0, 1.0]])
        b = np.array([[1.0]])
        c = np.array([2.0, 2.0]).reshape(-1, 1)
        x = solve_sylvester(a, b, c)
        # 断言解 x 与预期值相近
        assert_array_almost_equal(x, np.array([1.0, 1.0]).reshape(-1, 1))
```