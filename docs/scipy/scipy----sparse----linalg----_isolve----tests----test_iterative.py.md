# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_isolve\tests\test_iterative.py`

```
# 导入所需的库和模块
import itertools  # 导入 itertools 模块，用于生成迭代器的函数
import platform   # 导入 platform 模块，用于访问平台特定的系统信息
import sys        # 导入 sys 模块，提供了访问与 Python 解释器及其环境有关的变量与函数
import pytest     # 导入 pytest 模块，用于编写和运行测试用例

import numpy as np                    # 导入 NumPy 库，并简化命名为 np
from numpy.testing import assert_array_equal, assert_allclose  # 导入 NumPy 测试函数
from numpy import zeros, arange, array, ones, eye, iscomplexobj  # 导入 NumPy 函数和对象
from numpy.linalg import norm         # 导入 NumPy 线性代数模块中的 norm 函数

from scipy.sparse import spdiags, csr_matrix, kronsum  # 导入 SciPy 稀疏矩阵处理相关函数和类

from scipy.sparse.linalg import LinearOperator, aslinearoperator  # 导入 SciPy 稀疏线性代数相关函数和类
from scipy.sparse.linalg._isolve import (bicg, bicgstab, cg, cgs,  # 导入 SciPy 稀疏线性代数迭代求解器函数
                                         gcrotmk, gmres, lgmres,
                                         minres, qmr, tfqmr)

# TODO check that method preserve shape and type
# TODO test both preconditioner methods

# solvers 列表包含了所有要测试的求解器函数
_SOLVERS = [bicg, bicgstab, cg, cgs, gcrotmk, gmres, lgmres,
            minres, qmr, tfqmr]

# CB_TYPE_FILTER 用于过滤回调类型未指定的警告信息

# 创建参数化的 fixture，便于在测试中重复使用
@pytest.fixture(params=_SOLVERS, scope="session")
def solver(request):
    """
    Fixture for all solvers in scipy.sparse.linalg._isolve
    """
    return request.param


# 定义 Case 类，表示单个测试用例
class Case:
    def __init__(self, name, A, b=None, skip=None, nonconvergence=None):
        self.name = name
        self.A = A
        # 如果 b 未指定，则默认为 A 的行数范围的浮点数数组
        if b is None:
            self.b = arange(A.shape[0], dtype=float)
        else:
            self.b = b
        # 如果 skip 未指定，则设为空列表
        if skip is None:
            self.skip = []
        else:
            self.skip = skip
        # 如果 nonconvergence 未指定，则设为空列表
        if nonconvergence is None:
            self.nonconvergence = []
        else:
            self.nonconvergence = nonconvergence


# 定义 SingleTest 类，表示单个迭代测试
class SingleTest:
    def __init__(self, A, b, solver, casename, convergence=True):
        self.A = A
        self.b = b
        self.solver = solver
        self.name = casename + '-' + solver.__name__
        self.convergence = convergence

    def __repr__(self):
        return f"<{self.name}>"


# 定义 IterativeParams 类，用于生成测试用例
class IterativeParams:
    def generate_tests(self):
        # 生成应用跳过的测试用例
        tests = []
        for case in self.cases:
            for solver in _SOLVERS:
                if (solver in case.skip):
                    continue
                if solver in case.nonconvergence:
                    tests += [SingleTest(case.A, case.b, solver, case.name,
                                         convergence=False)]
                else:
                    tests += [SingleTest(case.A, case.b, solver, case.name)]
        return tests


# 从 IterativeParams 实例生成测试用例列表 cases
cases = IterativeParams().generate_tests()

# 创建参数化 fixture，用于在模块范围内重复使用测试用例
@pytest.fixture(params=cases, ids=[x.name for x in cases], scope="module")
def case(request):
    """
    Fixture for all cases in IterativeParams
    """
    return request.param


# 定义测试函数 test_maxiter，用于测试最大迭代次数
def test_maxiter(case):
    if not case.convergence:
        pytest.skip("Solver - Breakdown case, see gh-8829")
    A = case.A
    rtol = 1e-12

    b = case.b
    x0 = 0 * b

    residuals = []

    # 定义回调函数 callback，用于记录残差
    def callback(x):
        residuals.append(norm(b - case.A * x))
    # 如果使用的求解器是 gmres
    if case.solver == gmres:
        # 在运行时，用 pytest 捕获 DeprecationWarning，匹配特定的 CB_TYPE_FILTER 字符串
        with pytest.warns(DeprecationWarning, match=CB_TYPE_FILTER):
            # 调用 gmres 求解器求解线性方程组 A * x = b
            # 返回解 x 和求解信息 info，最大迭代次数为 1，回调函数为 callback
            x, info = case.solver(A, b, x0=x0, rtol=rtol, maxiter=1, callback=callback)
    else:
        # 如果不是 gmres 求解器，直接调用求解器求解线性方程组 A * x = b
        # 返回解 x 和求解信息 info，最大迭代次数为 1，回调函数为 callback
        x, info = case.solver(A, b, x0=x0, rtol=rtol, maxiter=1, callback=callback)

    # 断言残差列表的长度为 1
    assert len(residuals) == 1
    # 断言求解信息为 1
    assert info == 1
# 定义测试函数，用于验证求解器的收敛性和预处理器的效果
def test_convergence(case):
    # 从参数 case 中获取矩阵 A
    A = case.A

    # 根据 A 的数据类型确定相对误差容限 rtol
    if A.dtype.char in "dD":
        rtol = 1e-8
    else:
        rtol = 1e-2

    # 从参数 case 中获取向量 b 和定义初始解 x0
    b = case.b
    x0 = 0 * b

    # 调用 case 中的求解器 solver，解方程 A @ x = b，返回解 x 和求解信息 info
    x, info = case.solver(A, b, x0=x0, rtol=rtol)

    # 断言初始解 x0 没有被覆写
    assert_array_equal(x0, 0 * b)

    # 如果 case 表示应该收敛
    if case.convergence:
        # 断言求解信息为 0，即收敛成功
        assert info == 0
        # 断言解 x 满足残差限制条件
        assert norm(A @ x - b) <= norm(b) * rtol
    else:
        # 如果 case 表示不应该收敛，断言求解信息不为 0，即收敛失败
        assert info != 0
        # 断言解 x 满足残差限制条件
        assert norm(A @ x - b) <= norm(b)


# 定义测试函数，用于验证特定条件下的预处理器效果
def test_precond_dummy(case):
    # 如果不应该收敛，跳过测试
    if not case.convergence:
        pytest.skip("Solver - Breakdown case, see gh-8829")

    # 设置相对误差容限 rtol
    rtol = 1e-8

    # 定义简单的单位预处理器函数 identity
    def identity(b, which=None):
        """trivial preconditioner"""
        return b

    # 从参数 case 中获取矩阵 A
    A = case.A

    # 获取矩阵 A 的形状信息 M, N
    M, N = A.shape

    # 确保矩阵 A 的对角元素非零，以便计算逆对角阵
    diagOfA = A.diagonal()
    if np.count_nonzero(diagOfA) == len(diagOfA):
        spdiags([1.0 / diagOfA], [0], M, N)

    # 从参数 case 中获取向量 b 和定义初始解 x0
    b = case.b
    x0 = 0 * b

    # 创建线性操作器 precond，使用 identity 作为其前后向乘函数
    precond = LinearOperator(A.shape, identity, rmatvec=identity)

    # 根据求解器类型选择相应的参数传递给求解函数 solver
    if case.solver is qmr:
        x, info = case.solver(A, b, M1=precond, M2=precond, x0=x0, rtol=rtol)
    else:
        x, info = case.solver(A, b, M=precond, x0=x0, rtol=rtol)

    # 断言求解信息为 0，即收敛成功
    assert info == 0
    # 断言解 x 满足残差限制条件
    assert norm(A @ x - b) <= norm(b) * rtol

    # 将矩阵 A 转换为线性操作器，设置前后向乘函数为 identity
    A = aslinearoperator(A)
    A.psolve = identity
    A.rpsolve = identity

    # 使用修改后的 A 进行求解，验证仍然满足收敛条件
    x, info = case.solver(A, b, x0=x0, rtol=rtol)
    # 断言求解信息为 0，即收敛成功
    assert info == 0
    # 断言解 x 满足残差限制条件
    assert norm(A @ x - b) <= norm(b) * rtol


# 对于 poisson1d 和 poisson2d 案例的特定测试
@pytest.mark.fail_slow(10)
@pytest.mark.parametrize('case', [x for x in IterativeParams().cases
                                  if x.name in ('poisson1d', 'poisson2d')],
                         ids=['poisson1d', 'poisson2d'])
def test_precond_inverse(case):
    for solver in _SOLVERS:
        # 遍历可用的求解器列表
        
        if solver in case.skip or solver is qmr:
            # 如果求解器在跳过列表中，或者等于 qmr，则跳过当前循环
            continue

        rtol = 1e-8
        # 设置残差的相对容差

        def inverse(b, which=None):
            """inverse preconditioner"""
            # 定义逆预处理函数，使用案例中的矩阵 A
            A = case.A
            if not isinstance(A, np.ndarray):
                A = A.toarray()
            return np.linalg.solve(A, b)
            # 返回矩阵 A 对向量 b 的求解结果

        def rinverse(b, which=None):
            """inverse preconditioner"""
            # 定义逆转置预处理函数，使用案例中的矩阵 A
            A = case.A
            if not isinstance(A, np.ndarray):
                A = A.toarray()
            return np.linalg.solve(A.T, b)
            # 返回矩阵 A 转置对向量 b 的求解结果

        matvec_count = [0]
        # 初始化矩阵向量乘法计数器

        def matvec(b):
            # 定义矩阵向量乘法函数
            matvec_count[0] += 1
            return case.A @ b
            # 返回矩阵 A 乘以向量 b 的结果

        def rmatvec(b):
            # 定义转置矩阵向量乘法函数
            matvec_count[0] += 1
            return case.A.T @ b
            # 返回矩阵 A 转置乘以向量 b 的结果

        b = case.b
        # 设置向量 b 为案例的右侧向量
        x0 = 0 * b
        # 初始化初始解向量 x0 为零向量

        A = LinearOperator(case.A.shape, matvec, rmatvec=rmatvec)
        # 创建线性算子 A，使用案例中的矩阵 A 和定义的 matvec、rmatvec 函数

        precond = LinearOperator(case.A.shape, inverse, rmatvec=rinverse)
        # 创建预处理算子 precond，使用案例中的矩阵 A 和定义的 inverse、rinverse 函数

        # Solve with preconditioner
        # 使用预处理器求解
        matvec_count = [0]
        # 重置矩阵向量乘法计数器
        x, info = solver(A, b, M=precond, x0=x0, rtol=rtol)
        # 使用当前求解器 solver，线性算子 A，右侧向量 b，预处理器 M=precond，初始解 x0，残差容差 rtol 求解线性方程组

        assert info == 0
        # 断言求解返回的信息值为 0，表示成功求解

        assert norm(case.A @ x - b) <= norm(b) * rtol
        # 断言求解后的解 x 满足残差的相对容差要求

        # Solution should be nearly instant
        # 求解过程应当几乎瞬时完成
        assert matvec_count[0] <= 3
        # 断言矩阵向量乘法的次数不超过 3 次
# 测试特定求解器的绝对容差设置，历史上它们没有使用绝对容差，所以修复它不太紧急。
def test_atol(solver):
    # 如果求解器是 minres 或 tfqmr，则跳过测试并显示相应的信息
    if solver in (minres, tfqmr):
        pytest.skip("TODO: Add atol to minres/tfqmr")

    # 使用随机数生成器创建一个随机种子，生成一个随机的对称正定矩阵 A 和一个随机向量 b
    rng = np.random.default_rng(168441431005389)
    A = rng.uniform(size=[10, 10])
    A = A @ A.T + 10*np.eye(10)
    b = 1e3 * rng.uniform(size=10)

    # 计算向量 b 的范数
    b_norm = np.linalg.norm(b)

    # 定义绝对容差的一系列值，从指数 -9 到 2
    tols = np.r_[0, np.logspace(-9, 2, 7), np.inf]

    # 检查不同预处理器条件下的影响
    M0 = rng.standard_normal(size=(10, 10))
    M0 = M0 @ M0.T
    Ms = [None, 1e-6 * M0, 1e6 * M0]

    # 使用 itertools.product 遍历所有 Ms、rtol 和 atol 的组合
    for M, rtol, atol in itertools.product(Ms, tols, tols):
        # 跳过 rtol 和 atol 都为 0 的情况
        if rtol == 0 and atol == 0:
            continue

        # 对于 qmr 求解器，需要对 M 进行处理
        if solver is qmr:
            if M is not None:
                M = aslinearoperator(M)
                M2 = aslinearoperator(np.eye(10))
            else:
                M2 = None
            # 调用求解器求解线性系统
            x, info = solver(A, b, M1=M, M2=M2, rtol=rtol, atol=atol)
        else:
            # 调用求解器求解线性系统
            x, info = solver(A, b, M=M, rtol=rtol, atol=atol)

        # 断言求解成功
        assert info == 0

        # 计算残差向量，并计算其范数
        residual = A @ x - b
        err = np.linalg.norm(residual)

        # 计算新的绝对容差值 atol2
        atol2 = rtol * b_norm

        # 增加 1.00025 的调整因子，因为在 s390x 上，err 稍微超过 atol 的情况（见 gh-17839）
        assert err <= 1.00025 * max(atol, atol2)


# 测试零右手边的情况
def test_zero_rhs(solver):
    # 使用随机数生成器创建随机种子，生成一个随机的对称正定矩阵 A
    rng = np.random.default_rng(1684414984100503)
    A = rng.random(size=[10, 10])
    A = A @ A.T + 10 * np.eye(10)

    # 设置右手边向量 b 为零向量
    b = np.zeros(10)

    # 定义绝对容差的一系列值，从指数 -10 到 2
    tols = np.r_[np.logspace(-10, 2, 7)]

    # 遍历所有绝对容差的值
    for tol in tols:
        # 求解线性系统，期望解 x 为零向量
        x, info = solver(A, b, rtol=tol)
        assert info == 0
        assert_allclose(x, 0., atol=1e-15)

        # 使用初始值为全1向量的情况下，求解线性系统，期望解 x 为零向量
        x, info = solver(A, b, rtol=tol, x0=np.ones(10))
        assert info == 0
        assert_allclose(x, 0., atol=tol)

        # 如果求解器不是 minres，测试更多的情况
        if solver is not minres:
            # 使用初始值为全1向量，且绝对容差为0的情况下，求解线性系统，期望解 x 为零向量
            x, info = solver(A, b, rtol=tol, atol=0, x0=np.ones(10))
            if info == 0:
                assert_allclose(x, 0)

            # 使用绝对容差为 tol 的情况下，求解线性系统，期望解 x 为零向量
            x, info = solver(A, b, rtol=tol, atol=tol)
            assert info == 0
            assert_allclose(x, 0, atol=1e-300)

            # 使用绝对容差为0的情况下，求解线性系统，期望解 x 为零向量
            x, info = solver(A, b, rtol=tol, atol=0)
            assert info == 0
            assert_allclose(x, 0, atol=1e-300)
    # 如果 solver 是 gmres，并且平台是 'aarch64'，并且 Python 版本是 3.9.x，
    # 则标记为预期失败，原因是 gh-13019
    if (solver is gmres and platform.machine() == 'aarch64'
            and sys.version_info[1] == 9):
        pytest.xfail(reason="gh-13019")
    
    # 如果 solver 是 lgmres，并且平台不在 ['x86_64', 'x86', 'aarch64', 'arm64'] 中，
    # 则标记为预期失败，原因是在 ppc64le、ppc64 和 riscv64 上失败，参见 gh-17839
    if (solver is lgmres and
            platform.machine() not in ['x86_64', 'x86', 'aarch64', 'arm64']):
        pytest.xfail(reason="fails on at least ppc64le, ppc64 and riscv64")

    # 定义一个 4x4 的 numpy 数组 A，包含特定的数值
    A = np.array([[-0.1112795288033378, 0, 0, 0.16127952880333685],
                  [0, -0.13627952880333782 + 6.283185307179586j, 0, 0],
                  [0, 0, -0.13627952880333782 - 6.283185307179586j, 0],
                  [0.1112795288033368, 0j, 0j, -0.16127952880333785]])
    
    # 创建一个长度为 4 的 numpy 数组 v，所有元素为 1
    v = np.ones(4)
    
    # 初始化一个变量 best_error 为无穷大
    best_error = np.inf

    # 设定 slack_tol 变量，根据平台决定不同的值
    # 原注释提到根据平台是 'aarch64' 还是其他情况设定不同的值，但这里直接设为 9
    slack_tol = 9

    # 迭代 maxiter 从 1 到 19
    for maxiter in range(1, 20):
        # 调用 solver 函数求解线性系统 A @ x = v
        x, info = solver(A, v, maxiter=maxiter, rtol=1e-8, atol=0)

        # 如果求解成功 (info == 0)，则检查误差是否在允许的范围内
        if info == 0:
            assert norm(A @ x - v) <= 1e-8 * norm(v)

        # 计算当前误差
        error = np.linalg.norm(A @ x - v)
        
        # 更新最小误差
        best_error = min(best_error, error)

        # 使用 slack_tol 值检查误差是否在允许的范围内
        assert error <= slack_tol * best_error
# 测试求解器在简单问题上的工作
def test_x0_working(solver):
    # 使用固定种子生成随机数生成器
    rng = np.random.default_rng(1685363802304750)
    n = 10
    # 生成随机的 n x n 矩阵 A
    A = rng.random(size=[n, n])
    A = A @ A.T  # 计算 A 的转置与自身的乘积
    # 生成随机向量 b 和 x0
    b = rng.random(n)
    x0 = rng.random(n)

    # 根据求解器选择设置关键字参数字典
    if solver is minres:
        kw = dict(rtol=1e-6)
    else:
        kw = dict(atol=0, rtol=1e-6)

    # 调用求解器求解线性方程 Ax = b，返回解 x 和信息
    x, info = solver(A, b, **kw)
    assert info == 0
    # 验证解的精度是否满足要求
    assert norm(A @ x - b) <= 1e-6 * norm(b)

    # 使用指定初始向量 x0 调用求解器求解线性方程 Ax = b，返回解 x 和信息
    x, info = solver(A, b, x0=x0, **kw)
    assert info == 0
    # 验证解的精度是否满足更严格的要求
    assert norm(A @ x - b) <= 3e-6 * norm(b)


# 测试使用 x0='Mb' 的情况
def test_x0_equals_Mb(case):
    # 在特定条件下跳过测试
    if (case.solver is bicgstab) and (case.name == 'nonsymposdef-bicgstab'):
        pytest.skip("Solver fails due to numerical noise "
                    "on some architectures (see gh-15533).")
    if case.solver is tfqmr:
        pytest.skip("Solver does not support x0='Mb'")

    A = case.A
    b = case.b
    x0 = 'Mb'
    rtol = 1e-8
    # 使用 x0='Mb' 调用求解器求解线性方程 Ax = b，返回解 x 和信息
    x, info = case.solver(A, b, x0=x0, rtol=rtol)

    # 确保 x0 没有被覆盖
    assert_array_equal(x0, 'Mb')
    # 验证解的精度是否满足要求
    assert info == 0
    assert norm(A @ x - b) <= rtol * norm(b)


# 测试使用初始向量 x0=rhs 求解问题
def test_x0_solves_problem_exactly(solver):
    # 见 gh-19948
    mat = np.eye(2)
    rhs = np.array([-1., -1.])

    # 使用初始向量 x0=rhs 调用求解器求解线性方程 Ax = rhs，返回解 sol 和信息
    sol, info = solver(mat, rhs, x0=rhs)
    # 确保解等于右手边 rhs
    assert_allclose(sol, rhs)
    # 验证解的信息为成功（info == 0）
    assert info == 0


# 测试 TFQMR 求解器的输出
@pytest.mark.parametrize('case', IterativeParams().cases)
def test_show(case, capsys):
    # 定义一个简单的回调函数 cb
    def cb(x):
        pass

    # 调用 TFQMR 求解器，显示求解过程并捕获输出
    x, info = tfqmr(case.A, case.b, callback=cb, show=True)
    out, err = capsys.readouterr()

    if case.name == "sym-nonpd":
        # 某些情况下没有日志输出
        exp = ""
    elif case.name in ("nonsymposdef", "nonsymposdef-F"):
        # 非对称且正定情况下的输出信息
        exp = "TFQMR: Linear solve not converged due to reach MAXIT iterations"
    else:
        # 其他情况下的输出信息
        exp = "TFQMR: Linear solve converged due to reach TOL iterations"

    # 验证输出以特定字符串开头，并且错误输出为空
    assert out.startswith(exp)
    assert err == ""


# 测试未指定 x0 时的错误处理
def test_positional_error(solver):
    # 来自 test_x0_working
    rng = np.random.default_rng(1685363802304750)
    n = 10
    A = rng.random(size=[n, n])
    A = A @ A.T
    b = rng.random(n)
    x0 = rng.random(n)
    # 使用 pytest 引发 TypeError 异常，因为 x0 参数未指定名称
    with pytest.raises(TypeError):
        solver(A, b, x0, 1e-5)


# 测试无效的 atol 参数处理
@pytest.mark.parametrize("atol", ["legacy", None, -1])
def test_invalid_atol(solver, atol):
    if solver == minres:
        pytest.skip("minres has no `atol` argument")
    # 来自 test_x0_working
    rng = np.random.default_rng(1685363802304750)
    n = 10
    A = rng.random(size=[n, n])
    A = A @ A.T
    b = rng.random(n)
    x0 = rng.random(n)
    # 使用 pytest 引发 ValueError 异常，因为 atol 参数无效
    with pytest.raises(ValueError):
        solver(A, b, x0, atol=atol)


# QMR 求解器的单元测试类
class TestQMR:
    @pytest.mark.filterwarnings('ignore::scipy.sparse.SparseEfficiencyWarning')
    def test_leftright_precond(self):
        """Check that QMR works with left and right preconditioners"""
        # 导入必要的库函数
        from scipy.sparse.linalg._dsolve import splu
        from scipy.sparse.linalg._interface import LinearOperator

        # 定义矩阵维度
        n = 100

        # 创建对角线上的数据数组
        dat = ones(n)
        # 构建稀疏矩阵 A
        A = spdiags([-2 * dat, 4 * dat, -dat], [-1, 0, 1], n, n)
        # 构建右侧向量 b
        b = arange(n, dtype='d')

        # 构建左预条件矩阵 L 和右预条件矩阵 U
        L = spdiags([-dat / 2, dat], [-1, 0], n, n)
        U = spdiags([4 * dat, -dat], [0, 1], n, n)
        # 使用 LU 分解求解 L 和 U 的逆
        L_solver = splu(L)
        U_solver = splu(U)

        # 定义解 L_solve 和 U_solve 的函数
        def L_solve(b):
            return L_solver.solve(b)

        def U_solve(b):
            return U_solver.solve(b)

        # 定义解 L_solve 的转置的函数 LT_solve 和 U_solve 的转置的函数 UT_solve
        def LT_solve(b):
            return L_solver.solve(b, 'T')

        def UT_solve(b):
            return U_solver.solve(b, 'T')

        # 创建 LinearOperator 对象 M1 和 M2，用于传递预条件器给 QMR 方法
        M1 = LinearOperator((n, n), matvec=L_solve, rmatvec=LT_solve)
        M2 = LinearOperator((n, n), matvec=U_solve, rmatvec=UT_solve)

        # 设置相对容差
        rtol = 1e-8
        # 调用 QMR 方法求解方程 A @ x = b，使用 M1 和 M2 作为预条件器
        x, info = qmr(A, b, rtol=rtol, maxiter=15, M1=M1, M2=M2)

        # 断言求解成功
        assert info == 0
        # 断言残差满足容差要求
        assert norm(A @ x - b) <= rtol * norm(b)
class TestGMRES:
    # GMRES 测试类
    def test_basic(self):
        # 基本测试：构造 Vandermonde 矩阵 A 和零向量 b
        A = np.vander(np.arange(10) + 1)[:, ::-1]
        b = np.zeros(10)
        b[0] = 1

        # 运行 GMRES 算法，限定重启次数和最大迭代次数
        x_gm, err = gmres(A, b, restart=5, maxiter=1)

        # 断言：验证结果 x_gm 的第一个元素接近于 0.359，相对误差容忍度为 1e-2
        assert_allclose(x_gm[0], 0.359, rtol=1e-2)

    @pytest.mark.filterwarnings(f"ignore:{CB_TYPE_FILTER}:DeprecationWarning")
    # 回调函数测试，忽略特定的警告类型
    def test_callback(self):

        def store_residual(r, rvec):
            # 存储残差函数，将残差 r 存入 rvec 中合适的位置
            rvec[rvec.nonzero()[0].max() + 1] = r

        # 定义稀疏矩阵 A 和单位向量 b
        A = csr_matrix(array([[-2, 1, 0, 0, 0, 0],
                              [1, -2, 1, 0, 0, 0],
                              [0, 1, -2, 1, 0, 0],
                              [0, 0, 1, -2, 1, 0],
                              [0, 0, 0, 1, -2, 1],
                              [0, 0, 0, 0, 1, -2]]))
        b = ones((A.shape[0],))
        maxiter = 1
        rvec = zeros(maxiter + 1)
        rvec[0] = 1.0

        def callback(r):
            # 回调函数，返回存储残差的函数 store_residual
            return store_residual(r, rvec)

        # 运行 GMRES 算法，指定初值、相对误差容忍度、最大迭代次数和回调函数
        x, flag = gmres(A, b, x0=zeros(A.shape[0]), rtol=1e-16,
                        maxiter=maxiter, callback=callback)

        # 预期输出（来自 SciPy 1.0.0），验证残差向量 rvec
        assert_allclose(rvec, array([1.0, 0.81649658092772603]), rtol=1e-10)

        # 测试预条件回调函数
        M = 1e-3 * np.eye(A.shape[0])
        rvec = zeros(maxiter + 1)
        rvec[0] = 1.0
        x, flag = gmres(A, b, M=M, rtol=1e-16, maxiter=maxiter,
                        callback=callback)

        # 预期输出（来自 SciPy 1.0.0），预条件化后的残差向量 rvec
        assert_allclose(rvec, array([1.0, 1e-3 * 0.81649658092772603]),
                        rtol=1e-10)

    def test_abi(self):
        # 检查 GMRES 对复杂参数不会导致段错误
        A = eye(2)
        b = ones(2)
        r_x, r_info = gmres(A, b)
        r_x = r_x.astype(complex)
        x, info = gmres(A.astype(complex), b.astype(complex))

        # 断言：x 是复数对象
        assert iscomplexobj(x)
        # 断言：验证 r_x 与 x 接近
        assert_allclose(r_x, x)
        # 断言：信息 r_info 与 info 相等
        assert r_info == info

    @pytest.mark.fail_slow(10)
    # 绝对容差测试，标记为失败缓慢
    def test_atol_legacy(self):

        A = eye(2)
        b = ones(2)
        x, info = gmres(A, b, rtol=1e-5)

        # 断言：验证 A @ x 与 b 的范数小于等于相对容差的乘积
        assert np.linalg.norm(A @ x - b) <= 1e-5 * np.linalg.norm(b)
        # 断言：验证 x 与 b 接近，绝对容差为 0，相对容差为 1e-8
        assert_allclose(x, b, atol=0, rtol=1e-8)

        rndm = np.random.RandomState(12345)
        A = rndm.rand(30, 30)
        b = 1e-6 * ones(30)
        x, info = gmres(A, b, rtol=1e-7, restart=20)

        # 断言：验证 A @ x 与 b 的范数大于 1e-7
        assert np.linalg.norm(A @ x - b) > 1e-7

        A = eye(2)
        b = 1e-10 * ones(2)
        x, info = gmres(A, b, rtol=1e-8, atol=0)

        # 断言：验证 A @ x 与 b 的范数小于等于 1e-8 的乘积
        assert np.linalg.norm(A @ x - b) <= 1e-8 * np.linalg.norm(b)
    def test_defective_precond_breakdown(self):
        # Breakdown due to defective preconditioner

        # 创建一个3x3的单位矩阵
        M = np.eye(3)
        # 将第三行第三列元素设为0，导致预条件器有缺陷
        M[2, 2] = 0

        # 创建向量b和初始向量x
        b = np.array([0, 1, 1])
        x = np.array([1, 0, 0])
        # 创建对角矩阵A
        A = np.diag([2, 3, 4])

        # 使用gmres求解线性方程组Ax=b，指定预条件矩阵M
        x, info = gmres(A, b, x0=x, M=M, rtol=1e-15, atol=0)

        # 确保解x中没有NaN值
        assert not np.isnan(x).any()
        # 如果info为0，确保解的精度符合要求
        if info == 0:
            assert np.linalg.norm(A @ x - b) <= 1e-15 * np.linalg.norm(b)

        # 确保解x在M的零空间之外是正确的
        assert_allclose(M @ (A @ x), M @ b)

    def test_defective_matrix_breakdown(self):
        # Breakdown due to defective matrix

        # 创建一个有缺陷的矩阵A
        A = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
        # 创建向量b
        b = np.array([1, 0, 1])
        # 设置解的相对容差
        rtol = 1e-8
        # 使用gmres求解线性方程组Ax=b
        x, info = gmres(A, b, rtol=rtol, atol=0)

        # 确保解x中没有NaN值
        assert not np.isnan(x).any()
        # 如果info为0，确保解的精度符合要求
        if info == 0:
            assert np.linalg.norm(A @ x - b) <= rtol * np.linalg.norm(b)

        # 确保解x在A的零空间之外是正确的
        assert_allclose(A @ (A @ x), A @ b)

    @pytest.mark.filterwarnings(f"ignore:{CB_TYPE_FILTER}:DeprecationWarning")
    def test_callback_type(self):
        # The legacy callback type changes meaning of 'maxiter'

        # 设置随机种子
        np.random.seed(1)
        # 创建一个20x20的随机矩阵A和长度为20的随机向量b
        A = np.random.rand(20, 20)
        b = np.random.rand(20)

        # 定义一个计数器列表cb_count
        cb_count = [0]

        # 定义两个回调函数
        def pr_norm_cb(r):
            cb_count[0] += 1
            assert isinstance(r, float)

        def x_cb(x):
            cb_count[0] += 1
            assert isinstance(x, np.ndarray)

        # 使用gmres求解线性方程组Ax=b，限制最大迭代次数为2，并设置回调函数pr_norm_cb
        cb_count = [0]
        x, info = gmres(A, b, rtol=1e-6, atol=0, callback=pr_norm_cb,
                        maxiter=2, restart=50)
        # 确保返回的info为2，表示达到最大迭代次数
        assert info == 2
        # 确保回调函数pr_norm_cb被调用了2次
        assert cb_count[0] == 2

        # 使用gmres求解线性方程组Ax=b，限制最大迭代次数为2，并设置回调函数pr_norm_cb和回调类型为'legacy'
        cb_count = [0]
        x, info = gmres(A, b, rtol=1e-6, atol=0, callback=pr_norm_cb,
                        maxiter=2, restart=50, callback_type='legacy')
        # 确保返回的info为2，表示达到最大迭代次数
        assert info == 2
        # 确保回调函数pr_norm_cb被调用了2次
        assert cb_count[0] == 2

        # 使用gmres求解线性方程组Ax=b，限制最大迭代次数为2，并设置回调函数pr_norm_cb和回调类型为'pr_norm'
        cb_count = [0]
        x, info = gmres(A, b, rtol=1e-6, atol=0, callback=pr_norm_cb,
                        maxiter=2, restart=50, callback_type='pr_norm')
        # 确保返回的info为0，表示成功收敛
        assert info == 0
        # 确保回调函数pr_norm_cb被调用次数大于2
        assert cb_count[0] > 2

        # 使用gmres求解线性方程组Ax=b，限制最大迭代次数为2，并设置回调函数x_cb和回调类型为'x'
        cb_count = [0]
        x, info = gmres(A, b, rtol=1e-6, atol=0, callback=x_cb, maxiter=2,
                        restart=50, callback_type='x')
        # 确保返回的info为0，表示成功收敛
        assert info == 0
        # 确保回调函数x_cb被调用了1次
        assert cb_count[0] == 1
    def test_callback_x_monotonic(self):
        # 定义单元测试方法：验证 callback_type='x' 时，误差递减单调性
        np.random.seed(1)
        # 生成一个 20x20 的随机矩阵 A，并加上单位矩阵
        A = np.random.rand(20, 20) + np.eye(20)
        # 生成长度为 20 的随机向量 b
        b = np.random.rand(20)

        # 初始化一个列表，包含无穷大作为初始残差
        prev_r = [np.inf]
        # 初始化计数器
        count = [0]

        # 定义回调函数 x_cb，计算当前解 x 的残差，并断言其不大于上一步的残差
        def x_cb(x):
            r = np.linalg.norm(A @ x - b)
            assert r <= prev_r[0]
            prev_r[0] = r  # 更新 prev_r[0] 为当前残差 r
            count[0] += 1  # 每调用一次 x_cb，计数器加一

        # 调用 gmres 方法求解线性方程组 Ax=b
        x, info = gmres(A, b, rtol=1e-6, atol=0, callback=x_cb, maxiter=20,
                        restart=10, callback_type='x')
        # 断言求解返回的信息 info 等于最大迭代次数 20
        assert info == 20
        # 断言 x_cb 被调用的次数等于迭代次数 20
        assert count[0] == 20
```