# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_eigen\tests\test_svds.py`

```
# 导入正则表达式模块
import re
# 导入深拷贝函数
import copy
# 导入NumPy库，并使用np作为别名
import numpy as np

# 从NumPy的测试模块中导入断言函数和数组相等函数
from numpy.testing import assert_allclose, assert_equal, assert_array_equal
# 导入pytest测试框架
import pytest

# 从SciPy线性代数模块中导入奇异值分解（SVD）和零空间计算函数
from scipy.linalg import svd, null_space
# 从SciPy稀疏矩阵模块中导入压缩稀疏列矩阵、判断是否稀疏、对角稀疏矩阵和随机稀疏矩阵函数
from scipy.sparse import csc_matrix, issparse, spdiags, random
# 从SciPy稀疏矩阵线性代数模块中导入线性算子和将对象转换为线性算子函数
from scipy.sparse.linalg import LinearOperator, aslinearoperator
# 从SciPy稀疏矩阵线性代数模块中导入奇异值分解函数
from scipy.sparse.linalg import svds
# 从SciPy稀疏矩阵线性代数模块中导入ARPACK奇异值分解无收敛异常类
from scipy.sparse.linalg._eigen.arpack import ArpackNoConvergence


# --- Helper Functions / Classes ---


def sorted_svd(m, k, which='LM'):
    # 对给定的密集矩阵m进行奇异值分解，并按照特定方式排序后返回奇异向量/值
    # 如果输入矩阵m是稀疏矩阵，则转换为密集矩阵
    if issparse(m):
        m = m.toarray()
    # 执行奇异值分解，返回奇异向量u、奇异值s和右奇异向量vh
    u, s, vh = svd(m)
    # 根据which参数指定的方式排序奇异值，选取前k个或后k个
    if which == 'LM':
        ii = np.argsort(s)[-k:]
    elif which == 'SM':
        ii = np.argsort(s)[:k]
    else:
        raise ValueError(f"unknown which={which!r}")

    return u[:, ii], s[ii], vh[ii]


def _check_svds(A, k, u, s, vh, which="LM", check_usvh_A=False,
                check_svd=True, atol=1e-10, rtol=1e-7):
    n, m = A.shape

    # 检查u、s、vh的形状是否符合预期
    assert_equal(u.shape, (n, k))
    assert_equal(s.shape, (k,))
    assert_equal(vh.shape, (k, m))

    # 检查是否能够通过u、s、vh重构原始矩阵A
    A_rebuilt = (u*s).dot(vh)
    assert_equal(A_rebuilt.shape, A.shape)
    # 如果需要检查重构矩阵与原始矩阵A的接近程度，则进行检查
    if check_usvh_A:
        assert_allclose(A_rebuilt, A, atol=atol, rtol=rtol)

    # 检查u是否为半正交矩阵
    uh_u = np.dot(u.T.conj(), u)
    assert_equal(uh_u.shape, (k, k))
    assert_allclose(uh_u, np.identity(k), atol=atol, rtol=rtol)

    # 检查vh是否为半正交矩阵
    vh_v = np.dot(vh, vh.T.conj())
    assert_equal(vh_v.shape, (k, k))
    assert_allclose(vh_v, np.identity(k), atol=atol, rtol=rtol)

    # 检查稀疏矩阵奇异值分解结果与密集矩阵奇异值分解结果的一致性
    if check_svd:
        u2, s2, vh2 = sorted_svd(A, k, which)
        assert_allclose(np.abs(u), np.abs(u2), atol=atol, rtol=rtol)
        assert_allclose(s, s2, atol=atol, rtol=rtol)
        assert_allclose(np.abs(vh), np.abs(vh2), atol=atol, rtol=rtol)


def _check_svds_n(A, k, u, s, vh, which="LM", check_res=True,
                  check_svd=True, atol=1e-10, rtol=1e-7):
    n, m = A.shape

    # 检查u、s、vh的形状是否符合预期
    assert_equal(u.shape, (n, k))
    assert_equal(s.shape, (k,))
    assert_equal(vh.shape, (k, m))

    # 检查u是否为半正交矩阵
    uh_u = np.dot(u.T.conj(), u)
    assert_equal(uh_u.shape, (k, k))
    error = np.sum(np.abs(uh_u - np.identity(k))) / (k * k)
    assert_allclose(error, 0.0, atol=atol, rtol=rtol)

    # 检查vh是否为半正交矩阵
    vh_v = np.dot(vh, vh.T.conj())
    assert_equal(vh_v.shape, (k, k))
    error = np.sum(np.abs(vh_v - np.identity(k))) / (k * k)
    assert_allclose(error, 0.0, atol=atol, rtol=rtol)

    # 检查残差是否符合预期
    # 如果 check_res 为真，则进行以下检查
    if check_res:
        # 计算残差 ru，等于 A 的转置的共轭乘以 u 减去 vh 的转置的共轭乘以 s
        ru = A.T.conj() @ u - vh.T.conj() * s
        # 计算 ru 的绝对值之和除以元素个数 n*k，得到平均残差 rus
        rus = np.sum(np.abs(ru)) / (n * k)
        # 计算残差 rvh，等于 A 乘以 vh 的转置的共轭减去 u 乘以 s
        rvh = A @ vh.T.conj() - u * s
        # 计算 rvh 的绝对值之和除以元素个数 m*k，得到平均残差 rvhs
        rvhs = np.sum(np.abs(rvh)) / (m * k)
        # 使用 assert_allclose 函数检查 rus 是否接近于 0，允许的误差为 atol 和 rtol
        assert_allclose(rus, 0.0, atol=atol, rtol=rtol)
        # 使用 assert_allclose 函数检查 rvhs 是否接近于 0，允许的误差为 atol 和 rtol

        assert_allclose(rvhs, 0.0, atol=atol, rtol=rtol)

    # 检查 scipy.sparse.linalg.svds 是否接近 scipy.linalg.svd
    if check_svd:
        # 使用 sorted_svd 函数计算 A 的前 k 个奇异值分解的结果 u2, s2, vh2
        u2, s2, vh2 = sorted_svd(A, k, which)
        # 使用 assert_allclose 函数检查 s 和 s2 是否接近，允许的误差为 atol 和 rtol
        assert_allclose(s, s2, atol=atol, rtol=rtol)
        # 重建 A 的 SVD 结果 A_rebuilt_svd
        A_rebuilt_svd = (u2*s2).dot(vh2)
        # 重建 A 的结果 A_rebuilt
        A_rebuilt = (u*s).dot(vh)
        # 使用 assert_equal 函数检查 A_rebuilt 的形状是否与 A 相同
        assert_equal(A_rebuilt.shape, A.shape)
        # 计算重建误差 error，为重建结果之差的绝对值之和除以 k*k
        error = np.sum(np.abs(A_rebuilt_svd - A_rebuilt)) / (k * k)
        # 使用 assert_allclose 函数检查 error 是否接近于 0，允许的误差为 atol 和 rtol
        assert_allclose(error, 0.0, atol=atol, rtol=rtol)
# 定义一个继承自 LinearOperator 的类 CheckingLinearOperator
class CheckingLinearOperator(LinearOperator):
    
    # 初始化方法，接受参数 A 作为线性操作对象
    def __init__(self, A):
        # 将参数 A 赋值给实例变量 self.A
        self.A = A
        # 设置实例变量 self.dtype 为 A 的数据类型
        self.dtype = A.dtype
        # 设置实例变量 self.shape 为 A 的形状
        self.shape = A.shape

    # 定义矩阵向量乘法方法
    def _matvec(self, x):
        # 断言条件，确保 x 的最大形状等于其大小
        assert_equal(max(x.shape), np.size(x))
        # 返回 A 乘以 x 的结果
        return self.A.dot(x)

    # 定义矩阵转置向量乘法方法
    def _rmatvec(self, x):
        # 断言条件，确保 x 的最大形状等于其大小
        assert_equal(max(x.shape), np.size(x))
        # 返回 A 的共轭转置与 x 的乘积的结果
        return self.A.T.conjugate().dot(x)


# --- Test Input Validation ---
# 测试输入参数 `k` 和 `which` 的有效性
# 需要改进其他参数的输入验证检查

# 定义测试类 SVDSCommonTests
class SVDSCommonTests:

    solver = None

    # 一些输入验证测试可能仅运行一次，例如当 solver=None 时

    # 空数组异常消息
    _A_empty_msg = "`A` must not be empty."
    # 数组数据类型异常消息
    _A_dtype_msg = "`A` must be of floating or complex floating data type"
    # 类型不理解异常消息
    _A_type_msg = "type not understood"
    # 数组维度异常消息
    _A_ndim_msg = "array must have ndim <= 2"
    # 输入验证参数列表
    _A_validation_inputs = [
        (np.asarray([[]]), ValueError, _A_empty_msg),
        (np.asarray([[1, 2], [3, 4]]), ValueError, _A_dtype_msg),
        ("hi", TypeError, _A_type_msg),
        (np.asarray([[[1., 2.], [3., 4.]]]), ValueError, _A_ndim_msg)]

    # 使用 pytest 的参数化标记来测试参数 A 的输入验证
    @pytest.mark.parametrize("args", _A_validation_inputs)
    def test_svds_input_validation_A(self, args):
        # 从参数中获取 A、错误类型和消息
        A, error_type, message = args
        # 使用 pytest 的断言检查，确保在调用 svds 时会引发特定类型和消息的异常
        with pytest.raises(error_type, match=message):
            svds(A, k=1, solver=self.solver)

    # 使用 pytest 的参数化标记来测试参数 k 的输入验证
    @pytest.mark.parametrize("k", [-1, 0, 3, 4, 5, 1.5, "1"])
    def test_svds_input_validation_k_1(self, k):
        # 创建一个随机数生成器对象 rng
        rng = np.random.default_rng(0)
        # 生成一个随机矩阵 A，形状为 (4, 3)
        A = rng.random((4, 3))

        # 如果 solver 是 'propack' 并且 k 等于 3，则 propack 可以执行完整的奇异值分解
        if self.solver == 'propack' and k == 3:
            # 调用 svds 方法，传入参数 A、k、solver 和 random_state=0，并获取结果 res
            res = svds(A, k=k, solver=self.solver, random_state=0)
            # 检查结果的有效性，验证 usvh 和 A 的正确性，并进行奇异值分解检查
            _check_svds(A, k, *res, check_usvh_A=True, check_svd=True)
            return

        # 否则，验证 k 的类型和值的异常消息
        message = ("`k` must be an integer satisfying")
        with pytest.raises(ValueError, match=message):
            svds(A, k=k, solver=self.solver)

    # 测试方法，验证 k 参数的输入验证
    def test_svds_input_validation_k_2(self):
        # 当 k 无法转换为整数时，期望获得合理的堆栈跟踪信息
        message = "int() argument must be a"
        # 使用 pytest 的断言检查，确保在调用 svds 时会引发特定类型和消息的异常
        with pytest.raises(TypeError, match=re.escape(message)):
            svds(np.eye(10), k=[], solver=self.solver)

        message = "invalid literal for int()"
        # 使用 pytest 的断言检查，确保在调用 svds 时会引发特定类型和消息的异常
        with pytest.raises(ValueError, match=message):
            svds(np.eye(10), k="hi", solver=self.solver)

    # 使用 pytest 的参数化标记来测试参数 tol 的输入验证
    @pytest.mark.parametrize("tol", (-1, np.inf, np.nan))
    def test_svds_input_validation_tol_1(self, tol):
        # 非负浮点数值异常消息
        message = "`tol` must be a non-negative floating point value."
        # 使用 pytest 的断言检查，确保在调用 svds 时会引发特定类型和消息的异常
        with pytest.raises(ValueError, match=message):
            svds(np.eye(10), tol=tol, solver=self.solver)

    # 参数化标记测试参数 tol 的输入验证
    @pytest.mark.parametrize("tol", ([], 'hi'))
    # 针对 svds 函数的输入验证，测试当 tol 参数不合法时是否抛出 TypeError 异常
    def test_svds_input_validation_tol_2(self, tol):
        # 设置预期的错误信息
        message = "'<' not supported between instances"
        # 使用 pytest 的上下文管理器检查是否抛出 TypeError 异常，并验证错误信息是否匹配
        with pytest.raises(TypeError, match=message):
            svds(np.eye(10), tol=tol, solver=self.solver)

    # 使用参数化测试验证 svds 函数在不同 which 参数下是否抛出 ValueError 异常
    @pytest.mark.parametrize("which", ('LA', 'SA', 'ekki', 0))
    def test_svds_input_validation_which(self, which):
        # 用于回归测试的说明，解释了 GitHub 上的问题链接及相关背景
        # https://github.com/scipy/scipy/issues/4590
        # 函数未检查特定的特征值类型，可能返回意外的值。
        with pytest.raises(ValueError, match="`which` must be in"):
            svds(np.eye(10), which=which, solver=self.solver)

    # 使用参数化测试验证 svds 函数在不同 transpose 和 n 参数下的输入验证
    @pytest.mark.parametrize("transpose", (True, False))
    @pytest.mark.parametrize("n", range(4, 9))
    def test_svds_input_validation_v0_1(self, transpose, n):
        # 创建一个随机数生成器
        rng = np.random.default_rng(0)
        # 生成一个随机矩阵 A
        A = rng.random((5, 7))
        # 生成一个随机向量 v0
        v0 = rng.random(n)
        # 如果 transpose 为 True，则转置矩阵 A
        if transpose:
            A = A.T
        # 设置错误信息
        message = "`v0` must have shape"
        # 计算所需的 v0 长度
        required_length = (A.shape[0] if self.solver == 'propack'
                           else min(A.shape))
        # 如果 v0 的长度不符合要求，则期望引发 ValueError 异常
        if n != required_length:
            with pytest.raises(ValueError, match=message):
                svds(A, k=2, v0=v0, solver=self.solver)

    # 验证 svds 函数在 v0 参数不合法时是否抛出 ValueError 异常
    def test_svds_input_validation_v0_2(self):
        # 创建一个全为 1 的矩阵 A
        A = np.ones((10, 10))
        # 创建一个形状不符合要求的向量 v0
        v0 = np.ones((1, 10))
        # 设置错误信息
        message = "`v0` must have shape"
        # 验证是否抛出 ValueError 异常，并检查错误信息是否匹配
        with pytest.raises(ValueError, match=message):
            svds(A, k=1, v0=v0, solver=self.solver)

    # 使用参数化测试验证 svds 函数在不同 v0 类型下的输入验证
    @pytest.mark.parametrize("v0", ("hi", 1, np.ones(10, dtype=int)))
    def test_svds_input_validation_v0_3(self, v0):
        # 创建一个全为 1 的矩阵 A
        A = np.ones((10, 10))
        # 设置错误信息
        message = "`v0` must be of floating or complex floating data type."
        # 验证是否抛出 ValueError 异常，并检查错误信息是否匹配
        with pytest.raises(ValueError, match=message):
            svds(A, k=1, v0=v0, solver=self.solver)

    # 使用参数化测试验证 svds 函数在不同 maxiter 参数下的输入验证
    @pytest.mark.parametrize("maxiter", (-1, 0, 5.5))
    def test_svds_input_validation_maxiter_1(self, maxiter):
        # 设置错误信息
        message = ("`maxiter` must be a positive integer.")
        # 验证是否抛出 ValueError 异常，并检查错误信息是否匹配
        with pytest.raises(ValueError, match=message):
            svds(np.eye(10), maxiter=maxiter, solver=self.solver)

    # 验证 svds 函数在 maxiter 参数不合法时是否抛出 TypeError 和 ValueError 异常
    def test_svds_input_validation_maxiter_2(self):
        # 设置错误信息
        message = "int() argument must be a"
        # 验证是否抛出 TypeError 异常，并检查错误信息是否匹配
        with pytest.raises(TypeError, match=re.escape(message)):
            svds(np.eye(10), maxiter=[], solver=self.solver)

        # 设置错误信息
        message = "invalid literal for int()"
        # 验证是否抛出 ValueError 异常，并检查错误信息是否匹配
        with pytest.raises(ValueError, match=message):
            svds(np.eye(10), maxiter="hi", solver=self.solver)

    # 使用参数化测试验证 svds 函数在不同 rsv 参数下的输入验证
    @pytest.mark.parametrize("rsv", ('ekki', 10))
    # 定义一个测试方法，用于验证 svds 函数对 `return_singular_vectors` 参数的输入验证
    def test_svds_input_validation_return_singular_vectors(self, rsv):
        # 设置错误消息，用于在参数验证失败时匹配异常信息
        message = "`return_singular_vectors` must be in"
        # 使用 pytest 断言来检查是否抛出 ValueError 异常，并匹配指定的错误消息
        with pytest.raises(ValueError, match=message):
            # 调用 svds 函数进行特征分解，验证 `return_singular_vectors` 参数
            svds(np.eye(10), return_singular_vectors=rsv, solver=self.solver)

    # --- Test Parameters ---

    @pytest.mark.parametrize("k", [3, 5])
    @pytest.mark.parametrize("which", ["LM", "SM"])
    # 定义一个测试方法，用于验证 svds 函数的 `k` 和 `which` 参数
    def test_svds_parameter_k_which(self, k, which):
        # 检查 `k` 参数设置特征值/特征向量的返回数量
        # 同时检查 `which` 参数设置返回最大或最小特征值
        rng = np.random.default_rng(0)
        A = rng.random((10, 10))
        # 如果使用 'lobpcg' 求解器，预期会触发 UserWarning 警告
        if self.solver == 'lobpcg':
            with pytest.warns(UserWarning, match="The problem size"):
                res = svds(A, k=k, which=which, solver=self.solver,
                           random_state=0)
        else:
            # 调用 svds 函数进行特征分解，验证 `k` 和 `which` 参数
            res = svds(A, k=k, which=which, solver=self.solver,
                       random_state=0)
        # 调用辅助函数 _check_svds，检查 svds 函数的返回结果与预期是否一致
        _check_svds(A, k, *res, which=which, atol=1e-9, rtol=2e-13)

    @pytest.mark.filterwarnings("ignore:Exited",
                                reason="Ignore LOBPCG early exit.")
    # 为了简化，使用循环而不是 parametrize 来定义测试方法
    def test_svds_parameter_tol(self):
        # 验证 `tol` 参数对求解器精度的影响，通过解决具有不同 `tol` 的问题并比较特征值
        n = 100  # 矩阵大小
        k = 3    # 要检查的特征值数量

        # 生成一个随机的、稀疏的矩阵
        # 当矩阵过小时效果不明显
        rng = np.random.default_rng(0)
        A = rng.random((n, n))
        A[A > .1] = 0
        A = A @ A.T

        _, s, _ = svd(A)  # 计算基准特征值

        # 定义一个函数来计算 `tol` 的误差效果
        A = csc_matrix(A)

        def err(tol):
            _, s2, _ = svds(A, k=k, v0=np.ones(n), maxiter=1000,
                            solver=self.solver, tol=tol, random_state=0)
            return np.linalg.norm((s2 - s[k-1::-1])/s[k-1::-1])

        tols = [1e-4, 1e-2, 1e0]  # 要检查的容差级别
        # 对于 'arpack' 和 'propack' 求解器，准确性会有离散步进
        accuracies = {'propack': [1e-12, 1e-6, 1e-4],
                      'arpack': [2.5e-15, 1e-10, 1e-10],
                      'lobpcg': [2e-12, 4e-2, 2]}

        # 针对每个 `tol` 和对应的精度级别进行检查
        for tol, accuracy in zip(tols, accuracies[self.solver]):
            # 计算误差并断言误差应小于指定的精度水平
            error = err(tol)
            assert error < accuracy
    def test_svd_v0(self):
        # 检查 `v0` 参数对解的影响
        n = 100
        k = 1
        # 如果 k != 1，LOBPCG 需要更多初始向量，这些向量是通过 random_state 生成的，
        # 所以 k >= 2 时测试无法通过。
        # 对于一些 `n` 的其他值，使用不同的 v0 不会引发 AssertionError，这是合理的。

        rng = np.random.default_rng(0)
        A = rng.random((n, n))

        # 使用相同的 v0，解是相同的且准确的
        v0a = rng.random(n)
        res1a = svds(A, k, v0=v0a, solver=self.solver, random_state=0)
        res2a = svds(A, k, v0=v0a, solver=self.solver, random_state=1)
        for idx in range(3):
            assert_allclose(res1a[idx], res2a[idx], rtol=1e-15, atol=2e-16)
        _check_svds(A, k, *res1a)

        # 使用相同的 v0，解是相同的且准确的
        v0b = rng.random(n)
        res1b = svds(A, k, v0=v0b, solver=self.solver, random_state=2)
        res2b = svds(A, k, v0=v0b, solver=self.solver, random_state=3)
        for idx in range(3):
            assert_allclose(res1b[idx], res2b[idx], rtol=1e-15, atol=2e-16)
        _check_svds(A, k, *res1b)

        # 使用不同的 v0，解可能在数值上有所不同
        message = "Arrays are not equal"
        with pytest.raises(AssertionError, match=message):
            assert_equal(res1a, res1b)

    def test_svd_random_state(self):
        # 检查 `random_state` 参数对解的影响
        # 确实，选择 `n` 和 `k` 是为了确保所有求解器都通过所有这些检查。
        # 这是一项艰巨的任务，因为LOBPCG不愿达到所需的精度，而ARPACK经常对不同的 v0 返回相同的奇异值/向量。
        n = 100
        k = 1

        rng = np.random.default_rng(0)
        A = rng.random((n, n))

        # 使用相同的 random_state，解是相同的且准确的
        res1a = svds(A, k, solver=self.solver, random_state=0)
        res2a = svds(A, k, solver=self.solver, random_state=0)
        for idx in range(3):
            assert_allclose(res1a[idx], res2a[idx], rtol=1e-15, atol=2e-16)
        _check_svds(A, k, *res1a)

        # 使用相同的 random_state，解是相同的且准确的
        res1b = svds(A, k, solver=self.solver, random_state=1)
        res2b = svds(A, k, solver=self.solver, random_state=1)
        for idx in range(3):
            assert_allclose(res1b[idx], res2b[idx], rtol=1e-15, atol=2e-16)
        _check_svds(A, k, *res1b)

        # 使用不同的 random_state，解可能在数值上有所不同
        message = "Arrays are not equal"
        with pytest.raises(AssertionError, match=message):
            assert_equal(res1a, res1b)
    @pytest.mark.parametrize("random_state", (0, 1,
                                              np.random.RandomState(0),
                                              np.random.default_rng(0)))
    def test_svd_random_state_2(self, random_state):
        n = 100
        k = 1

        rng = np.random.default_rng(0)  # 使用种子0创建一个随机数生成器实例
        A = rng.random((n, n))  # 使用rng生成一个n x n的随机矩阵A

        random_state_2 = copy.deepcopy(random_state)  # 深拷贝random_state，得到random_state_2

        # 使用相同的random_state，得到的解是相同的且精确的
        res1a = svds(A, k, solver=self.solver, random_state=random_state)  # 运行svds函数，使用给定的random_state
        res2a = svds(A, k, solver=self.solver, random_state=random_state_2)  # 再次运行svds函数，使用相同的random_state_2
        for idx in range(3):
            assert_allclose(res1a[idx], res2a[idx], rtol=1e-15, atol=2e-16)  # 断言res1a和res2a的前三个返回值非常接近
        _check_svds(A, k, *res1a)  # 调用_check_svds函数，验证svds的结果是否符合预期

    @pytest.mark.parametrize("random_state", (None,
                                              np.random.RandomState(0),
                                              np.random.default_rng(0)))
    @pytest.mark.filterwarnings("ignore:Exited",
                                reason="Ignore LOBPCG early exit.")
    def test_svd_random_state_3(self, random_state):
        n = 100
        k = 5

        rng = np.random.default_rng(0)  # 使用种子0创建一个随机数生成器实例
        A = rng.random((n, n))  # 使用rng生成一个n x n的随机矩阵A

        random_state = copy.deepcopy(random_state)  # 深拷贝random_state，得到一个新的随机状态对象

        # 使用不同的random_state，得到的解是精确的，但不一定是相同的
        res1a = svds(A, k, solver=self.solver, random_state=random_state, maxiter=1000)  # 运行svds函数，使用给定的random_state
        res2a = svds(A, k, solver=self.solver, random_state=random_state, maxiter=1000)  # 再次运行svds函数，使用相同的random_state
        _check_svds(A, k, *res1a, atol=2e-7)  # 调用_check_svds函数，验证svds的结果是否符合预期，设置容错参数
        _check_svds(A, k, *res2a, atol=2e-7)  # 再次调用_check_svds函数，验证第二次svds的结果是否符合预期，设置容错参数

        message = "Arrays are not equal"
        with pytest.raises(AssertionError, match=message):
            assert_equal(res1a, res2a)  # 使用断言确保res1a和res2a是相等的，否则抛出AssertionError

    @pytest.mark.filterwarnings("ignore:Exited postprocessing")
    def test_svd_maxiter(self):
        # 测试函数：test_svd_maxiter，用于验证最大迭代次数的功能

        # 创建一个 9x9 的对角矩阵 A，元素为 0 到 8 的浮点数
        A = np.diag(np.arange(9)).astype(np.float64)
        
        # 设置要计算的奇异值个数 k
        k = 1
        
        # 调用 sorted_svd 函数，返回奇异值分解的结果 u, s, vh
        u, s, vh = sorted_svd(A, k)
        
        # 默认使用默认的 maxiter 参数
        maxiter = None

        # 根据不同的求解器类型进行条件判断
        if self.solver == 'arpack':
            # 如果求解器是 ARPACK，验证在 maxiter=1 时是否抛出 ArpackNoConvergence 错误
            message = "ARPACK error -1: No convergence"
            with pytest.raises(ArpackNoConvergence, match=message):
                svds(A, k, ncv=3, maxiter=1, solver=self.solver)
        elif self.solver == 'lobpcg':
            # 如果求解器是 LOBPCG，设置较高的 maxiter 值为 30，确保测试通过
            maxiter = 30
            with pytest.warns(UserWarning, match="Exited at iteration"):
                svds(A, k, maxiter=1, solver=self.solver)
        elif self.solver == 'propack':
            # 如果求解器是 PROPACK，验证在 maxiter=1 时是否抛出 np.linalg.LinAlgError 错误
            message = "k=1 singular triplets did not converge within"
            with pytest.raises(np.linalg.LinAlgError, match=message):
                svds(A, k, maxiter=1, solver=self.solver)

        # 调用 svds 函数，使用给定的求解器和 maxiter 参数计算奇异值分解，结果存储在 ud, sd, vhd 中
        ud, sd, vhd = svds(A, k, solver=self.solver, maxiter=maxiter,
                           random_state=0)
        
        # 调用 _check_svds 函数，验证计算结果的正确性
        _check_svds(A, k, ud, sd, vhd, atol=1e-8)
        
        # 检查计算得到的奇异向量的绝对值是否与预期的 u 的绝对值在指定的误差范围内相等
        assert_allclose(np.abs(ud), np.abs(u), atol=1e-8)
        
        # 检查计算得到的右奇异向量的绝对值是否与预期的 vh 的绝对值在指定的误差范围内相等
        assert_allclose(np.abs(vhd), np.abs(vh), atol=1e-8)
        
        # 检查计算得到的奇异值的绝对值是否与预期的 s 的绝对值在指定的误差范围内相等
        assert_allclose(np.abs(sd), np.abs(s), atol=1e-9)
    # 定义一个测试函数，用于测试简单的奇异值分解（SVD）情况
    def test_svd_simple(self, A, k, real, transpose, lo_type):
        # 将输入的 A 转换为 NumPy 数组
        A = np.asarray(A)
        # 如果指定要求实部，则取 A 的实部
        A = np.real(A) if real else A
        # 如果指定要求转置，则将 A 进行转置操作
        A = A.T if transpose else A
        # 使用指定的 lo_type 对 A 进行处理，返回处理后的结果 A2
        A2 = lo_type(A)

        # 如果 k 大于 A 的最小维度，则跳过测试并提示错误信息
        if k > min(A.shape):
            pytest.skip("`k` cannot be greater than `min(A.shape)`")
        # 如果不是使用 'propack' 解决器，并且 k 大于等于 A 的最小维度，则跳过测试
        if self.solver != 'propack' and k >= min(A.shape):
            pytest.skip("Only PROPACK supports complete SVD")
        # 如果使用 'arpack' 解决器，并且 A 的数据类型不是实数，并且 k 恰好等于 A 的最小维度减1，则跳过测试
        if self.solver == 'arpack' and not real and k == min(A.shape) - 1:
            pytest.skip("#16725")

        # 设置默认的绝对误差阈值
        atol = 3e-10
        # 如果使用 'propack' 解决器，则调整绝对误差阈值以避免在特定平台上的测试失败
        if self.solver == 'propack':
            atol = 3e-9  # 否则在 Linux aarch64 上测试会失败（参见 gh-19855）

        # 如果使用 'lobpcg' 解决器，测试过程中会产生 UserWarning 警告信息
        if self.solver == 'lobpcg':
            with pytest.warns(UserWarning, match="The problem size"):
                # 执行 SVD，返回 u, s, vh
                u, s, vh = svds(A2, k, solver=self.solver, random_state=0)
        else:
            # 执行 SVD，返回 u, s, vh
            u, s, vh = svds(A2, k, solver=self.solver, random_state=0)
        
        # 对计算得到的 SVD 结果进行验证
        _check_svds(A, k, u, s, vh, atol=atol)

    # 定义一个元组，包含多组不同的形状值
    SHAPES = ((100, 100), (100, 101), (101, 100))

    # 使用参数化测试进行测试过程中的警告信息过滤
    @pytest.mark.filterwarnings("ignore:Exited at iteration")
    @pytest.mark.filterwarnings("ignore:Exited postprocessing")
    # 使用参数化测试来测试不同的形状和数据类型组合
    @pytest.mark.parametrize("shape", SHAPES)
    # ARPACK 只支持 dtype 为 float、complex 或 np.float32
    @pytest.mark.parametrize("dtype", (float, complex, np.float32))
    def test_small_sigma_sparse(self, shape, dtype):
        # 设置 solver 为当前对象的 solver
        solver = self.solver
        # 随机数生成器，种子为 0
        rng = np.random.default_rng(0)
        k = 5
        # 从稀疏随机矩阵生成 S
        (m, n) = shape
        S = random(m, n, density=0.1, random_state=rng)
        # 如果数据类型为复数，则生成虚数部分为 1j 的随机矩阵
        if dtype == complex:
            S = + 1j * random(m, n, density=0.1, random_state=rng)
        # 生成对角线为 e 的对角矩阵，并与 S 相乘
        e = np.ones(m)
        e[0:5] *= 1e1 ** np.arange(-5, 0, 1)
        S = spdiags(e, 0, m, m) @ S
        S = S.astype(dtype)
        # 执行 SVD，返回 u, s, vh
        u, s, vh = svds(S, k, which='SM', solver=solver, maxiter=1000,
                        random_state=0)
        # 验证计算得到的 SVD 结果
        c_svd = False  # 部分 SVD 可能与完全 SVD 不同
        _check_svds_n(S, k, u, s, vh, which="SM", check_svd=c_svd, atol=2e-1)

    # --- Test Edge Cases ---
    # 检查一些边缘情况。

    # 使用参数化测试来测试不同的形状和数据类型组合
    @pytest.mark.parametrize("shape", ((6, 5), (5, 5), (5, 6)))
    @pytest.mark.parametrize("dtype", (float, complex))
    def test_svd_LM_ones_matrix(self, shape, dtype):
        # 检查在 LM 模式下，svds 能够处理矩阵秩小于 k 的情况。
        k = 3
        n, m = shape
        A = np.ones((n, m), dtype=dtype)  # 创建一个全为1的矩阵 A，形状为 (n, m)，数据类型为 dtype

        if self.solver == 'lobpcg':
            # 如果求解器为 'lobpcg'，预期会有 UserWarning 提示信息
            with pytest.warns(UserWarning, match="The problem size"):
                U, s, VH = svds(A, k, solver=self.solver, random_state=0)
        else:
            # 否则直接调用 svds 函数求解
            U, s, VH = svds(A, k, solver=self.solver, random_state=0)

        # 检查 svds 返回结果的一致性和正确性
        _check_svds(A, k, U, s, VH, check_usvh_A=True, check_svd=False)

        # 检查最大奇异值是否接近 sqrt(n*m)
        # 并且其他奇异值是否被强制为零
        assert_allclose(np.max(s), np.sqrt(n*m))
        s = np.array(sorted(s)[:-1]) + 1
        z = np.ones_like(s)
        assert_allclose(s, z)

    @pytest.mark.filterwarnings("ignore:k >= N - 1",
                                reason="needed to demonstrate #16725")
    @pytest.mark.parametrize("shape", ((3, 4), (4, 4), (4, 3), (4, 2)))
    @pytest.mark.parametrize("dtype", (float, complex))
    def test_zero_matrix(self, shape, dtype):
        # 检查 svds 能够处理全零矩阵的情况；
        # 参考 https://github.com/scipy/scipy/issues/3452/
        # shape = (4, 2) 是因为它是报告问题的特定案例
        k = 1
        n, m = shape
        A = np.zeros((n, m), dtype=dtype)  # 创建一个全零矩阵 A，形状为 (n, m)，数据类型为 dtype

        if (self.solver == 'arpack' and dtype is complex
                and k == min(A.shape) - 1):
            pytest.skip("#16725")

        if self.solver == 'propack':
            pytest.skip("PROPACK failures unrelated to PR #16712")

        if self.solver == 'lobpcg':
            # 如果求解器为 'lobpcg'，预期会有 UserWarning 提示信息
            with pytest.warns(UserWarning, match="The problem size"):
                U, s, VH = svds(A, k, solver=self.solver, random_state=0)
        else:
            # 否则直接调用 svds 函数求解
            U, s, VH = svds(A, k, solver=self.solver, random_state=0)

        # 检查 svds 返回结果的一致性和正确性
        _check_svds(A, k, U, s, VH, check_usvh_A=True, check_svd=False)

        # 检查奇异值是否全部为零
        assert_array_equal(s, 0)

    @pytest.mark.parametrize("shape", ((20, 20), (20, 21), (21, 20)))
    # ARPACK 仅支持 dtype 为 float、complex 或 np.float32
    @pytest.mark.parametrize("dtype", (float, complex, np.float32))
    @pytest.mark.filterwarnings("ignore:Exited",
                                reason="Ignore LOBPCG early exit.")
    # 定义一个测试方法，测试在给定形状和数据类型下的小 Sigma 情况
    def test_small_sigma(self, shape, dtype):
        # 使用指定的随机种子创建一个随机数生成器对象
        rng = np.random.default_rng(179847540)
        # 生成一个指定形状和数据类型的随机数组，并转换为指定数据类型
        A = rng.random(shape).astype(dtype)
        # 对矩阵 A 进行奇异值分解，得到左奇异向量 u、奇异值 s 和右奇异向量 vh
        u, _, vh = svd(A, full_matrices=False)
        # 根据数据类型确定参数 e 的值
        if dtype == np.float32:
            e = 10.0
        else:
            e = 100.0
        # 计算 e 的负幂次方，并转换为指定数据类型
        t = e**(-np.arange(len(vh))).astype(dtype)
        # 更新矩阵 A 为 u * t @ vh
        A = (u*t).dot(vh)
        # 设置参数 k 的值为 4
        k = 4
        # 使用 svds 函数对更新后的矩阵 A 进行截断奇异值分解，得到截断后的 u、s、vh
        u, s, vh = svds(A, k, solver=self.solver, maxiter=100, random_state=0)
        # 计算 s 中大于 0 的奇异值的数量
        t = np.sum(s > 0)
        # 使用 assert_equal 断言 t 等于 k
        assert_equal(t, k)
        # 调用 _check_svds_n 函数，验证截断奇异值分解结果满足特定条件
        _check_svds_n(A, k, u, s, vh, atol=1e-3, rtol=1e0, check_svd=False)

    # 使用 ARPACK 求解器支持的数据类型只能是 float、complex 或 np.float32
    @pytest.mark.filterwarnings("ignore:The problem size")
    @pytest.mark.parametrize("dtype", (float, complex, np.float32))
    # 定义另一个测试方法，测试在给定数据类型下的小 Sigma 情况
    def test_small_sigma2(self, dtype):
        # 使用指定的随机种子创建一个随机数生成器对象
        rng = np.random.default_rng(179847540)
        # 创建一个大小为 10x10 的奇异矩阵，具有 4 维零空间
        dim = 4
        size = 10
        x = rng.random((size, size-dim))
        y = x[:, :dim] * rng.random(dim)
        mat = np.hstack((x, y))
        mat = mat.astype(dtype)

        # 计算矩阵 mat 的零空间
        nz = null_space(mat)
        # 使用 assert_equal 断言零空间的列数等于 dim
        assert_equal(nz.shape[1], dim)

        # 设置容差参数 atol 和 rtol，以通过 np.float32 的测试
        # 使用稠密矩阵奇异值分解
        u, s, vh = svd(mat)
        # 断言 s 的最小值接近于 0
        assert_allclose(s[-dim:], 0, atol=1e-6, rtol=1e0)
        # 断言 mat @ vh 的最小右奇异向量接近于 0
        assert_allclose(mat @ vh[-dim:, :].T, 0, atol=1e-6, rtol=1e0)

        # 断言最小奇异值接近于 0，使用稀疏矩阵奇异值分解
        sp_mat = csc_matrix(mat)
        su, ss, svh = svds(sp_mat, k=dim, which='SM', solver=self.solver,
                           random_state=0)
        # 断言最小的 dim 个奇异值接近于 0
        assert_allclose(ss, 0, atol=1e-5, rtol=1e0)
        # 断言通过 svds 计算的最小奇异向量在零空间中接近于 0
        n, m = mat.shape
        if n < m:  # 否则在某些库中可能会导致断言失败，原因不明
            assert_allclose(sp_mat.transpose() @ su, 0, atol=1e-5, rtol=1e0)
        assert_allclose(sp_mat @ svh.T, 0, atol=1e-5, rtol=1e0)
# --- Perform tests with each solver ---

# 定义测试类 Test_SVDS_once，用于测试 svds 函数的输入验证，只有一个参数 solver
class Test_SVDS_once:
    
    # 使用 pytest 的 parametrize 装饰器，参数化 solver，允许输入 ['ekki', object]
    @pytest.mark.parametrize("solver", ['ekki', object])
    def test_svds_input_validation_solver(self, solver):
        # 验证错误信息中应包含字符串 "solver must be one of"
        message = "solver must be one of"
        # 断言调用 svds 函数时会抛出 ValueError 异常，异常信息与 message 匹配
        with pytest.raises(ValueError, match=message):
            svds(np.ones((3, 4)), k=2, solver=solver)


# 定义测试类 Test_SVDS_ARPACK，继承 SVDSCommonTests 类
class Test_SVDS_ARPACK(SVDSCommonTests):
    
    # 在每个测试方法执行前调用，设置 self.solver 为 'arpack'
    def setup_method(self):
        self.solver = 'arpack'

    # 使用 pytest 的 parametrize 装饰器，参数化 ncv，范围为 [-1, 7] + [4.5, "5"]
    @pytest.mark.parametrize("ncv", list(range(-1, 8)) + [4.5, "5"])
    def test_svds_input_validation_ncv_1(self, ncv):
        # 创建随机数生成器 rng
        rng = np.random.default_rng(0)
        # 生成一个随机矩阵 A，维度为 (6, 7)
        A = rng.random((6, 7))
        # 设置 k 的值为 3
        k = 3
        # 如果 ncv 的值为 4 或 5
        if ncv in {4, 5}:
            # 调用 svds 函数进行部分奇异值分解，使用参数 ncv, k, solver=self.solver, random_state=0
            u, s, vh = svds(A, k=k, ncv=ncv, solver=self.solver, random_state=0)
            # 检查 svds 分解结果的有效性，调用 _check_svds 函数
            _check_svds(A, k, u, s, vh)
        else:
            # 如果 ncv 不是 4 或 5，抛出 ValueError 异常，异常信息中包含 "ncv must be an integer satisfying"
            message = ("`ncv` must be an integer satisfying")
            with pytest.raises(ValueError, match=message):
                svds(A, k=k, ncv=ncv, solver=self.solver)

    # 定义测试方法 test_svds_input_validation_ncv_2，不接受任何参数
    def test_svds_input_validation_ncv_2(self):
        # 在 `ncv` 无法转换为整数时，期望捕获 TypeError 异常，异常信息中包含 "int() argument must be a"
        message = "int() argument must be a"
        with pytest.raises(TypeError, match=re.escape(message)):
            svds(np.eye(10), ncv=[], solver=self.solver)

        # 在 `ncv` 的值为非法字符串时，期望捕获 ValueError 异常，异常信息中包含 "invalid literal for int()"
        message = "invalid literal for int()"
        with pytest.raises(ValueError, match=message):
            svds(np.eye(10), ncv="hi", solver=self.solver)

    # I can't see a robust relationship between `ncv` and relevant outputs
    # (e.g. accuracy, time), so no test of the parameter.


# 定义测试类 Test_SVDS_LOBPCG，继承 SVDSCommonTests 类
class Test_SVDS_LOBPCG(SVDSCommonTests):
    
    # 在每个测试方法执行前调用，设置 self.solver 为 'lobpcg'
    def setup_method(self):
        self.solver = 'lobpcg'


# 定义测试类 Test_SVDS_PROPACK，继承 SVDSCommonTests 类
class Test_SVDS_PROPACK(SVDSCommonTests):
    
    # 在每个测试方法执行前调用，设置 self.solver 为 'propack'
    def setup_method(self):
        self.solver = 'propack'

    # 定义测试方法 test_svd_LM_ones_matrix，用于测试全为 1 的矩阵的奇异值分解
    def test_svd_LM_ones_matrix(self):
        # 设置期望失败的消息字符串，标记该测试为预期失败
        message = ("PROPACK does not return orthonormal singular vectors "
                   "associated with zero singular values.")
        pytest.xfail(message)

    # 定义测试方法 test_svd_LM_zeros_matrix，用于测试全为 0 的矩阵的奇异值分解
    def test_svd_LM_zeros_matrix(self):
        # 设置期望失败的消息字符串，标记该测试为预期失败
        message = ("PROPACK does not return orthonormal singular vectors "
                   "associated with zero singular values.")
        pytest.xfail(message)
```