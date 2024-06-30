# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_eigen\arpack\tests\test_arpack.py`

```
# 程序的用法说明字符串，描述如何在本地运行测试
__usage__ = """
To run tests locally:
  python tests/test_arpack.py [-l<int>] [-v<int>]

"""

# 导入线程和迭代工具
import threading
import itertools

# 导入 NumPy 库并使用简称 np
import numpy as np

# 导入 NumPy 测试模块中的断言函数和警告抑制功能
from numpy.testing import assert_allclose, assert_equal, suppress_warnings

# 导入 pytest 的 raises 断言函数，并使用简称 assert_raises
from pytest import raises as assert_raises
import pytest

# 导入 NumPy 的 dot、conj 和 random 函数
from numpy import dot, conj, random

# 导入 SciPy 线性代数模块中的 eig 和 eigh 函数
from scipy.linalg import eig, eigh

# 导入 SciPy 稀疏矩阵模块中的 csc_matrix、csr_matrix、diags 和 rand 函数
from scipy.sparse import csc_matrix, csr_matrix, diags, rand

# 导入 SciPy 稀疏矩阵线性代数模块中的 LinearOperator 和 aslinearoperator 函数
from scipy.sparse.linalg import LinearOperator, aslinearoperator

# 导入 SciPy 稀疏矩阵 ARPACK 模块中的特定函数和异常
from scipy.sparse.linalg._eigen.arpack import (eigs, eigsh, arpack,
                                              ArpackNoConvergence)

# 导入 SciPy 的内部库中的垃圾回收相关函数和标志
from scipy._lib._gcutils import assert_deallocated, IS_PYPY

# 测试精度设置，用于不同数据类型的 ARPACK 特征值问题
_ndigits = {'f': 3, 'd': 11, 'F': 3, 'D': 11}


def _get_test_tolerance(type_char, mattype=None, D_type=None, which=None):
    """
    根据测试需求返回适当的容差值：

    Parameters
    ----------
    type_char : {'f', 'd', 'F', 'D'}
        ARPACK 特征值问题中的数据类型字符
    mattype : {csr_matrix, aslinearoperator, asarray}, optional
        线性操作类型

    Returns
    -------
    tol
        传递给 ARPACK 例程的容差
    rtol
        输出的相对容差
    atol
        输出的绝对容差

    """

    # 初始化相对容差
    rtol = {'f': 3000 * np.finfo(np.float32).eps,
            'F': 3000 * np.finfo(np.float32).eps,
            'd': 2000 * np.finfo(np.float64).eps,
            'D': 2000 * np.finfo(np.float64).eps}[type_char]

    # 初始化绝对容差为相对容差的值
    atol = rtol
    tol = 0

    # 如果 mattype 是 aslinearoperator 并且 type_char 是 'f' 或 'F'
    if mattype is aslinearoperator and type_char in ('f', 'F'):
        # 单精度迭代方法: 错误更大，提高 ARPACK 的容差以确保迭代方法收敛
        tol = 30 * np.finfo(np.float32).eps
        rtol *= 5

    # 如果 mattype 是 csr_matrix 并且 type_char 是 'f' 或 'F'
    if mattype is csr_matrix and type_char in ('f', 'F'):
        # 稀疏矩阵单精度: 错误更大
        rtol *= 5

    # 如果 which 是 ('LM', 'SM', 'LA') 并且 D_type.name 是 "gen-hermitian-Mc"
    if (
        which in ('LM', 'SM', 'LA')
        and D_type.name == "gen-hermitian-Mc"
    ):
        # 如果 type_char 是 'F'
        if type_char == 'F':
            # 增加容差以覆盖 PR 14798 中缺失的情况
            rtol *= 5

        # 如果 type_char 是 'D'
        if type_char == 'D':
            # 增加更多的容差以覆盖 PR 14798 中缺失的情况
            rtol *= 10
            atol *= 10

    return tol, rtol, atol


def generate_matrix(N, complex_=False, hermitian=False,
                    pos_definite=False, sparse=False):
    """
    生成一个随机矩阵：

    Parameters
    ----------
    N : int
        矩阵的大小
    complex_ : bool, optional
        是否生成复数矩阵，默认为 False
    hermitian : bool, optional
        是否生成 Hermitian 矩阵，默认为 False
    pos_definite : bool, optional
        是否生成正定矩阵，默认为 False
    sparse : bool, optional
        是否生成稀疏矩阵，默认为 False

    Returns
    -------
    M : ndarray
        生成的随机矩阵

    """

    # 生成随机的 N x N 矩阵
    M = np.random.random((N, N))

    # 如果 complex_ 为 True，生成复数矩阵
    if complex_:
        M = M + 1j * np.random.random((N, N))

    return M
    # 如果需要生成 Hermitian 矩阵
    if hermitian:
        # 如果需要生成正定矩阵
        if pos_definite:
            # 如果需要生成稀疏矩阵
            if sparse:
                # 创建一个包含0到N-1的整数数组
                i = np.arange(N)
                # 从0到N-1中随机选择N-2个整数作为列索引
                j = np.random.randint(N, size=N-2)
                # 创建索引网格
                i, j = np.meshgrid(i, j)
                # 将M中对应索引的元素设为0
                M[i, j] = 0
            # 计算 M 的共轭转置的乘积，得到 Hermitian 矩阵
            M = np.dot(M.conj(), M.T)
        else:
            # 计算 M 的共轭转置的乘积，得到 Hermitian 矩阵
            M = np.dot(M.conj(), M.T)
            # 如果需要生成稀疏矩阵
            if sparse:
                # 随机生成N*N//4个索引对 i, j
                i = np.random.randint(N, size=N * N // 4)
                j = np.random.randint(N, size=N * N // 4)
                # 找出对角线上的元素索引并进行调整
                ind = np.nonzero(i == j)
                j[ind] = (j[ind] + 1) % N
                # 将M中对应索引的元素设为0
                M[i, j] = 0
                M[j, i] = 0
    # 如果不需要生成 Hermitian 矩阵
    else:
        # 如果需要生成稀疏矩阵
        if sparse:
            # 随机生成N*N//2个索引对 i, j
            i = np.random.randint(N, size=N * N // 2)
            j = np.random.randint(N, size=N * N // 2)
            # 将M中对应索引的元素设为0
            M[i, j] = 0
    # 返回生成的矩阵 M
    return M
# 生成一个随机 N × N 的矩阵 M
def generate_matrix_symmetric(N, pos_definite=False, sparse=False):
    M = np.random.random((N, N))

    # 将 M 变换为对称矩阵的形式
    M = 0.5 * (M + M.T)

    # 如果要求正定性，则将单位矩阵乘以 N 加到 M 上
    if pos_definite:
        Id = N * np.eye(N)
        if sparse:
            M = csr_matrix(M)
        M += Id
    else:
        # 如果要求稀疏矩阵，则将 M 转换为 CSR 格式
        if sparse:
            M = csr_matrix(M)

    return M


# 检查两个数组是否近似相等，或者其复共轭接近相等
def assert_allclose_cc(actual, desired, **kw):
    try:
        assert_allclose(actual, desired, **kw)
    except AssertionError:
        # 如果失败，则检查实际值与期望值的复共轭是否近似相等
        assert_allclose(actual, conj(desired), **kw)


# 根据给定的特征值数组，使用指定的条件对特征值排序并返回索引
def argsort_which(eigenvalues, typ, k, which,
                  sigma=None, OPpart=None, mode=None):
    """Return sorted indices of eigenvalues using the "which" keyword
    from eigs and eigsh"""

    # 根据 sigma 的有无，计算重新调整后的特征值
    if sigma is None:
        reval = np.round(eigenvalues, decimals=_ndigits[typ])
    else:
        if mode is None or mode == 'normal':
            if OPpart is None:
                reval = 1. / (eigenvalues - sigma)
            elif OPpart == 'r':
                reval = 0.5 * (1. / (eigenvalues - sigma)
                               + 1. / (eigenvalues - np.conj(sigma)))
            elif OPpart == 'i':
                reval = -0.5j * (1. / (eigenvalues - sigma)
                                 - 1. / (eigenvalues - np.conj(sigma)))
        elif mode == 'cayley':
            reval = (eigenvalues + sigma) / (eigenvalues - sigma)
        elif mode == 'buckling':
            reval = eigenvalues / (eigenvalues - sigma)
        else:
            raise ValueError("mode='%s' not recognized" % mode)

        reval = np.round(reval, decimals=_ndigits[typ])

    # 根据 which 关键字选择排序的方式
    if which in ['LM', 'SM']:
        ind = np.argsort(abs(reval))
    elif which in ['LR', 'SR', 'LA', 'SA', 'BE']:
        ind = np.argsort(np.real(reval))
    elif which in ['LI', 'SI']:
        if typ.islower():
            ind = np.argsort(abs(np.imag(reval)))
        else:
            ind = np.argsort(np.imag(reval))
    else:
        raise ValueError("which='%s' is unrecognized" % which)

    # 根据 which 关键字返回特征值索引的前 k 个或后 k 个
    if which in ['LM', 'LA', 'LR', 'LI']:
        return ind[-k:]
    elif which in ['SM', 'SA', 'SR', 'SI']:
        return ind[:k]
    elif which == 'BE':
        return np.concatenate((ind[:k//2], ind[k//2-k:]))


# 根据给定的参数和特征值计算特征向量和特征值
def eval_evec(symmetric, d, typ, k, which, v0=None, sigma=None,
              mattype=np.asarray, OPpart=None, mode='normal'):
    general = ('bmat' in d)

    # 根据矩阵是否对称选择特征值求解的函数
    if symmetric:
        eigs_func = eigsh
    else:
        eigs_func = eigs

    # 如果指定了 general 参数，则输出特定的错误信息
    if general:
        err = ("error for {}:general, typ={}, which={}, sigma={}, "
               "mattype={}, OPpart={}, mode={}".format(eigs_func.__name__,
                                                   typ, which, sigma,
                                                   mattype.__name__,
                                                   OPpart, mode))
    # 否则，生成错误消息字符串，指明特征值计算函数的名称及其他参数信息
    err = ("error for {}:standard, typ={}, which={}, sigma={}, "
           "mattype={}, OPpart={}, mode={}".format(eigs_func.__name__,
                                               typ, which, sigma,
                                               mattype.__name__,
                                               OPpart, mode))

    # 将矩阵数据转换为指定类型
    a = d['mat'].astype(typ)
    # 根据指定的矩阵类型创建对应类型的矩阵对象
    ac = mattype(a)

    # 如果是一般化特征值问题，处理矩阵 b
    if general:
        # 将矩阵 b 数据转换为指定类型
        b = d['bmat'].astype(typ)
        # 根据指定的矩阵类型创建对应类型的矩阵对象
        bc = mattype(b)

    # 获取精确的特征值
    exact_eval = d['eval'].astype(typ.upper())
    # 根据指定条件对精确特征值进行排序，返回排序后的索引
    ind = argsort_which(exact_eval, typ, k, which,
                        sigma, OPpart, mode)
    exact_eval = exact_eval[ind]

    # 计算 ARPACK 方法得到的特征值
    kwargs = dict(which=which, v0=v0, sigma=sigma)
    if eigs_func is eigsh:
        kwargs['mode'] = mode
    else:
        kwargs['OPpart'] = OPpart

    # 计算适当的容差
    kwargs['tol'], rtol, atol = _get_test_tolerance(typ, mattype, d, which)
    # 在罕见情况下，ARPACK 方法可能返回符合要求的特征值和特征向量，但不一定是请求的哪一种。
    # 这是 Krylov 方法固有的特性，不应视为错误。如果出现这种情况，会尝试重新计算（最多几次）。
    ntries = 0
    while ntries < 5:
        # 求解特征值问题
        if general:
            try:
                eigenvalues, evec = eigs_func(ac, k, bc, **kwargs)
            except ArpackNoConvergence:
                # 如果没有收敛，增加最大迭代次数并重新尝试求解
                kwargs['maxiter'] = 20*a.shape[0]
                eigenvalues, evec = eigs_func(ac, k, bc, **kwargs)
        else:
            try:
                eigenvalues, evec = eigs_func(ac, k, **kwargs)
            except ArpackNoConvergence:
                # 如果没有收敛，增加最大迭代次数并重新尝试求解
                kwargs['maxiter'] = 20*a.shape[0]
                eigenvalues, evec = eigs_func(ac, k, **kwargs)

        # 根据指定条件对计算得到的特征值进行排序，返回排序后的索引
        ind = argsort_which(eigenvalues, typ, k, which,
                            sigma, OPpart, mode)
        eigenvalues = eigenvalues[ind]
        evec = evec[:, ind]

        try:
            # 检查特征值是否符合要求
            assert_allclose_cc(eigenvalues, exact_eval, rtol=rtol, atol=atol,
                               err_msg=err)
            check_evecs = True
        except AssertionError:
            # 如果特征值不符合要求，标记检查特征向量为假，并增加尝试次数
            check_evecs = False
            ntries += 1

        if check_evecs:
            # 检查特征向量
            LHS = np.dot(a, evec)
            if general:
                RHS = eigenvalues * np.dot(b, evec)
            else:
                RHS = eigenvalues * evec

            # 检查左右两边是否接近
            assert_allclose(LHS, RHS, rtol=rtol, atol=atol, err_msg=err)
            break

    # 最终检查特征值是否符合要求
    assert_allclose_cc(eigenvalues, exact_eval, rtol=rtol, atol=atol, err_msg=err)
# 定义一个继承自 dict 的类，用于具有自定义字符串表示的字典对象
class DictWithRepr(dict):
    def __init__(self, name):
        self.name = name

    # 返回对象的字符串表示，格式为对象名字的尖括号形式
    def __repr__(self):
        return "<%s>" % self.name


# 定义一个类，用于存储对称参数
class SymmetricParams:
    def __init__(self):
        # 设置默认的特征值计算方法为 eigsh
        self.eigs = eigsh
        # 设置可接受的特征值类型列表
        self.which = ['LM', 'SM', 'LA', 'SA', 'BE']
        # 设置可接受的矩阵类型列表
        self.mattypes = [csr_matrix, aslinearoperator, np.asarray]
        # 设置 sigma 参数对应的模式字典
        self.sigmas_modes = {None: ['normal'],
                             0.5: ['normal', 'buckling', 'cayley']}

        # 生成矩阵数据
        # 这些矩阵应该都是 float32 类型，以保证在 float32 和 float64 下的特征值一致性
        N = 6
        np.random.seed(2300)
        # 生成实对称矩阵 Ar
        Ar = generate_matrix(N, hermitian=True,
                             pos_definite=True).astype('f').astype('d')
        # 生成实对称矩阵 M
        M = generate_matrix(N, hermitian=True,
                            pos_definite=True).astype('f').astype('d')
        # 生成复数域下的对称矩阵 Ac
        Ac = generate_matrix(N, hermitian=True, pos_definite=True,
                             complex_=True).astype('F').astype('D')
        # 生成复数域下的对称矩阵 Mc
        Mc = generate_matrix(N, hermitian=True, pos_definite=True,
                             complex_=True).astype('F').astype('D')
        # 随机生成初始向量 v0
        v0 = np.random.random(N)

        # 标准对称问题
        SS = DictWithRepr("std-symmetric")
        SS['mat'] = Ar
        SS['v0'] = v0
        # 计算 Ar 的特征值
        SS['eval'] = eigh(SS['mat'], eigvals_only=True)

        # 一般对称问题
        GS = DictWithRepr("gen-symmetric")
        GS['mat'] = Ar
        GS['bmat'] = M
        GS['v0'] = v0
        # 计算 GS['mat'] 和 GS['bmat'] 的特征值
        GS['eval'] = eigh(GS['mat'], GS['bmat'], eigvals_only=True)

        # 标准厄米特问题
        SH = DictWithRepr("std-hermitian")
        SH['mat'] = Ac
        SH['v0'] = v0
        # 计算 Ac 的特征值
        SH['eval'] = eigh(SH['mat'], eigvals_only=True)

        # 一般厄米特问题
        GH = DictWithRepr("gen-hermitian")
        GH['mat'] = Ac
        GH['bmat'] = M
        GH['v0'] = v0
        # 计算 GH['mat'] 和 GH['bmat'] 的特征值
        GH['eval'] = eigh(GH['mat'], GH['bmat'], eigvals_only=True)

        # 一般厄米特问题，其中 M 也是厄米特矩阵
        GHc = DictWithRepr("gen-hermitian-Mc")
        GHc['mat'] = Ac
        GHc['bmat'] = Mc
        GHc['v0'] = v0
        # 计算 GHc['mat'] 和 GHc['bmat'] 的特征值
        GHc['eval'] = eigh(GHc['mat'], GHc['bmat'], eigvals_only=True)

        # 存储实对称测试案例列表
        self.real_test_cases = [SS, GS]
        # 存储复数域下的测试案例列表
        self.complex_test_cases = [SH, GH, GHc]
    # 初始化函数，用于设置对象的初始状态
    def __init__(self):
        # 初始化特征值求解器对象
        self.eigs = eigs
        # 指定要使用的特征值求解方式，包括 'LM', 'LR', 'LI'
        self.which = ['LM', 'LR', 'LI']  # , 'SM', 'LR', 'SR', 'LI', 'SI']
        # 定义矩阵类型列表，包括稀疏矩阵、线性操作器和数组
        self.mattypes = [csr_matrix, aslinearoperator, np.asarray]
        # 设置 sigma 参数对应的特征值求解结果集合
        self.sigmas_OPparts = {None: [None],
                               0.1: ['r'],
                               0.1 + 0.1j: ['r', 'i']}

        # 生成矩阵
        # 所有矩阵应该是 float32 类型，以确保在 float32 和 float64 下特征值相同
        N = 6
        np.random.seed(2300)
        # 生成标准实非对称问题的矩阵 Ar，并转换为 float64 类型
        Ar = generate_matrix(N).astype('f').astype('d')
        # 生成对称正定问题的矩阵 M，并转换为 float64 类型
        M = generate_matrix(N, hermitian=True, pos_definite=True).astype('f').astype('d')
        # 生成复杂数矩阵 Ac，并转换为复数 float64 类型
        Ac = generate_matrix(N, complex_=True).astype('F').astype('D')
        # 生成随机向量 v0
        v0 = np.random.random(N)

        # 标准实非对称问题
        SNR = DictWithRepr("std-real-nonsym")
        SNR['mat'] = Ar
        SNR['v0'] = v0
        # 计算 SNR['mat'] 的特征值，仅返回特征值，不计算特征向量
        SNR['eval'] = eig(SNR['mat'], left=False, right=False)

        # 一般实非对称问题
        GNR = DictWithRepr("gen-real-nonsym")
        GNR['mat'] = Ar
        GNR['bmat'] = M
        GNR['v0'] = v0
        # 计算 GNR['mat'] 的特征值，使用 GNR['bmat'] 作为广义特征值求解的辅助矩阵，仅返回特征值
        GNR['eval'] = eig(GNR['mat'], GNR['bmat'], left=False, right=False)

        # 标准复杂非对称问题
        SNC = DictWithRepr("std-cmplx-nonsym")
        SNC['mat'] = Ac
        SNC['v0'] = v0
        # 计算 SNC['mat'] 的特征值，仅返回特征值
        SNC['eval'] = eig(SNC['mat'], left=False, right=False)

        # 一般复杂非对称问题
        GNC = DictWithRepr("gen-cmplx-nonsym")
        GNC['mat'] = Ac
        GNC['bmat'] = M
        GNC['v0'] = v0
        # 计算 GNC['mat'] 的特征值，使用 GNC['bmat'] 作为广义特征值求解的辅助矩阵，仅返回特征值
        GNC['eval'] = eig(GNC['mat'], GNC['bmat'], left=False, right=False)

        # 存储实非对称问题的测试用例
        self.real_test_cases = [SNR, GNR]
        # 存储复杂非对称问题的测试用例
        self.complex_test_cases = [SNC, GNC]
# 测试对称模式的函数
def test_symmetric_modes():
    # 创建对称参数对象
    params = SymmetricParams()
    # 设置特定的 k 值
    k = 2
    # 指示是否为对称矩阵的标志
    symmetric = True
    # 遍历实数测试用例
    for D in params.real_test_cases:
        # 遍历数据类型 'f' 和 'd'
        for typ in 'fd':
            # 遍历参数中的 'which' 变量
            for which in params.which:
                # 遍历矩阵类型
                for mattype in params.mattypes:
                    # 遍历 sigma 和 modes 的组合
                    for (sigma, modes) in params.sigmas_modes.items():
                        # 遍历 modes
                        for mode in modes:
                            # 调用 eval_evec 函数进行计算
                            eval_evec(symmetric, D, typ, k, which,
                                      None, sigma, mattype, None, mode)


# 测试厄米模式的函数
def test_hermitian_modes():
    # 创建对称参数对象
    params = SymmetricParams()
    # 设置特定的 k 值
    k = 2
    # 指示是否为对称矩阵的标志
    symmetric = True
    # 遍历复数测试用例
    for D in params.complex_test_cases:
        # 遍历数据类型 'F' 和 'D'
        for typ in 'FD':
            # 遍历参数中的 'which' 变量
            for which in params.which:
                # 如果 which 为 'BE' 则跳过当前循环
                if which == 'BE':
                    continue  # BE invalid for complex
                # 遍历矩阵类型
                for mattype in params.mattypes:
                    # 遍历 sigma 的值
                    for sigma in params.sigmas_modes:
                        # 调用 eval_evec 函数进行计算
                        eval_evec(symmetric, D, typ, k, which,
                                  None, sigma, mattype)


# 测试对称模式的起始向量的函数
def test_symmetric_starting_vector():
    # 创建对称参数对象
    params = SymmetricParams()
    # 指示是否为对称矩阵的标志
    symmetric = True
    # 遍历指定的 k 值列表
    for k in [1, 2, 3, 4, 5]:
        # 遍历实数测试用例
        for D in params.real_test_cases:
            # 遍历数据类型 'f' 和 'd'
            for typ in 'fd':
                # 随机生成长度与 D['v0'] 相同的随机数组作为起始向量
                v0 = random.rand(len(D['v0'])).astype(typ)
                # 调用 eval_evec 函数进行计算
                eval_evec(symmetric, D, typ, k, 'LM', v0)


# 测试对称模式无收敛的函数
def test_symmetric_no_convergence():
    # 设置随机数种子
    np.random.seed(1234)
    # 生成一个大小为 30 的厄米矩阵，要求是正定的
    m = generate_matrix(30, hermitian=True, pos_definite=True)
    # 获取测试容差的值
    tol, rtol, atol = _get_test_tolerance('d')
    try:
        # 调用 eigsh 函数计算特征值和特征向量
        w, v = eigsh(m, 4, which='LM', v0=m[:, 0], maxiter=5, tol=tol, ncv=9)
        # 如果没有出现 ArpackNoConvergence 异常，则抛出断言错误
        raise AssertionError("Spurious no-error exit")
    except ArpackNoConvergence as err:
        # 获取未收敛时的特征值个数 k
        k = len(err.eigenvalues)
        # 如果 k 小于等于 0，则抛出断言错误
        if k <= 0:
            raise AssertionError("Spurious no-eigenvalues-found case") from err
        # 否则获取特征值和特征向量
        w, v = err.eigenvalues, err.eigenvectors
        # 使用 assert_allclose 函数验证特征值方程是否成立
        assert_allclose(dot(m, v), w * v, rtol=rtol, atol=atol)


# 测试实数非对称模式的函数
def test_real_nonsymmetric_modes():
    # 创建非对称参数对象
    params = NonSymmetricParams()
    # 设置特定的 k 值
    k = 2
    # 指示是否为对称矩阵的标志
    symmetric = False
    # 遍历实数测试用例
    for D in params.real_test_cases:
        # 遍历数据类型 'f' 和 'd'
        for typ in 'fd':
            # 遍历参数中的 'which' 变量
            for which in params.which:
                # 遍历矩阵类型
                for mattype in params.mattypes:
                    # 遍历 sigma 和 OPparts 的组合
                    for sigma, OPparts in params.sigmas_OPparts.items():
                        # 遍历 OPparts
                        for OPpart in OPparts:
                            # 调用 eval_evec 函数进行计算
                            eval_evec(symmetric, D, typ, k, which,
                                      None, sigma, mattype, OPpart)


# 测试复数非对称模式的函数
def test_complex_nonsymmetric_modes():
    # 创建非对称参数对象
    params = NonSymmetricParams()
    # 设置特定的 k 值
    k = 2
    # 指示是否为对称矩阵的标志
    symmetric = False
    # 遍历复数测试用例
    for D in params.complex_test_cases:
        # 遍历数据类型 'D' 和 'F'
        for typ in 'DF':
            # 遍历参数中的 'which' 变量
            for which in params.which:
                # 遍历矩阵类型
                for mattype in params.mattypes:
                    # 遍历 sigma 的值
                    for sigma in params.sigmas_OPparts:
                        # 调用 eval_evec 函数进行计算
                        eval_evec(symmetric, D, typ, k, which,
                                  None, sigma, mattype)


# 测试标准的非对称模式起始向量的函数
def test_standard_nonsymmetric_starting_vector():
    # 创建非对称参数对象
    params = NonSymmetricParams()
    # 设置 sigma 为 None
    sigma = None
    # 指示是否为对称矩阵的标志
    symmetric = False
    # 对于给定的参数组合进行多层嵌套循环，遍历参数列表和测试用例
    for k in [1, 2, 3, 4]:
        # 遍历复杂测试用例列表中的每个测试用例
        for d in params.complex_test_cases:
            # 遍历数据字典中的矩阵 'mat' 
            for typ in 'FD':
                # 从当前测试用例中获取矩阵 'mat'
                A = d['mat']
                # 获取矩阵 'A' 的维度
                n = A.shape[0]
                # 创建一个随机数数组作为初始向量 'v0'，并指定数据类型 'typ'
                v0 = random.rand(n).astype(typ)
                # 调用特定函数进行特定操作，传入相关参数进行计算
                eval_evec(symmetric, d, typ, k, "LM", v0, sigma)
# 定义一个测试函数，用于测试非对称起始向量的一般情况
def test_general_nonsymmetric_starting_vector():
    # 创建非对称参数对象
    params = NonSymmetricParams()
    # 初始设定 sigma 为 None
    sigma = None
    # 设定 symmetric 为 False
    symmetric = False
    # 遍历复杂测试用例列表
    for k in [1, 2, 3, 4]:
        # 遍历参数对象的复杂测试用例
        for d in params.complex_test_cases:
            # 遍历类型 'FD'
            for typ in 'FD':
                # 从测试用例中获取矩阵 A
                A = d['mat']
                # 获取矩阵 A 的维度
                n = A.shape[0]
                # 创建随机起始向量 v0，数据类型为 typ
                v0 = random.rand(n).astype(typ)
                # 调用 eval_evec 函数进行特征值-特征向量计算
                eval_evec(symmetric, d, typ, k, "LM", v0, sigma)


# 定义一个测试函数，用于测试非对称矩阵没有收敛的情况
def test_standard_nonsymmetric_no_convergence():
    # 设定随机种子
    np.random.seed(1234)
    # 生成一个复杂矩阵 m，大小为 30x30
    m = generate_matrix(30, complex_=True)
    # 获取测试容差值
    tol, rtol, atol = _get_test_tolerance('d')
    try:
        # 调用 eigs 函数计算前 4 个特征值和特征向量
        w, v = eigs(m, 4, which='LM', v0=m[:, 0], maxiter=5, tol=tol)
        # 如果出现意外的无错误退出，则引发 AssertionError
        raise AssertionError("Spurious no-error exit")
    except ArpackNoConvergence as err:
        # 获取未收敛时的特征值数量
        k = len(err.eigenvalues)
        # 如果特征值数量小于等于 0，则引发 AssertionError
        if k <= 0:
            raise AssertionError("Spurious no-eigenvalues-found case") from err
        # 否则，获取特征值和特征向量
        w, v = err.eigenvalues, err.eigenvectors
        # 对每个特征值和特征向量进行检查
        for ww, vv in zip(w, v.T):
            # 使用 assert_allclose 函数检查特征值特征向量关系
            assert_allclose(dot(m, vv), ww * vv, rtol=rtol, atol=atol)


# 定义一个测试函数，用于测试特征值计算中矩阵 A 不是方阵的情况
def test_eigen_bad_shapes():
    # 创建一个非方阵矩阵 A
    A = csc_matrix(np.zeros((2, 3)))
    # 使用 assert_raises 函数检查是否引发 ValueError 异常
    assert_raises(ValueError, eigs, A)


# 定义一个测试函数，用于测试特征值计算中的错误关键字参数
def test_eigen_bad_kwargs():
    # 创建一个方阵矩阵 A
    A = csc_matrix(np.zeros((8, 8)))
    # 使用 assert_raises 函数检查是否引发 ValueError 异常
    assert_raises(ValueError, eigs, A, which='XX')


# 定义一个测试函数，用于测试特定的 arpack 崩溃问题（ticket 1459）
def test_ticket_1459_arpack_crash():
    # 遍历浮点数数据类型列表
    for dtype in [np.float32, np.float64]:
        # 设置矩阵的维度和特征值数量
        N = 6
        k = 2
        # 设置随机种子并生成随机浮点矩阵 A
        np.random.seed(2301)
        A = np.random.random((N, N)).astype(dtype)
        # 设置特定的起始向量 v0
        v0 = np.array([-0.71063568258907849895, -0.83185111795729227424,
                       -0.34365925382227402451, 0.46122533684552280420,
                       -0.58001341115969040629, -0.78844877570084292984e-01],
                      dtype=dtype)
        # 调用 evals, evecs = eigs(A, k, v0=v0) 函数进行特征值计算
        evals, evecs = eigs(A, k, v0=v0)


# 标记为跳过测试条件为 IS_PYPY 的并行线程测试函数
@pytest.mark.skipif(IS_PYPY, reason="Test not meaningful on PyPy")
def test_parallel_threads():
    # 初始化结果列表
    results = []
    # 生成一个长度为 50 的随机向量 v0
    v0 = np.random.rand(50)
    # 定义一个函数 worker，用于执行特定的任务
    def worker():
        # 创建一个带有指定对角线值和偏移的稀疏矩阵 x
        x = diags([1, -2, 1], [-1, 0, 1], shape=(50, 50))
        # 计算稀疏矩阵 x 的前三个特征值和对应的特征向量
        w, v = eigs(x, k=3, v0=v0)
        # 将计算得到的特征值 w 添加到结果列表 results 中
        results.append(w)
    
        # 使用 Lanczos 方法计算稀疏矩阵 x 的前三个特征值和对应的特征向量
        w, v = eigsh(x, k=3, v0=v0)
        # 将计算得到的特征值 w 添加到结果列表 results 中
        results.append(w)
    
    # 创建包含 10 个 worker 函数的线程列表
    threads = [threading.Thread(target=worker) for k in range(10)]
    # 启动所有线程
    for t in threads:
        t.start()
    # 等待所有线程执行完成
    for t in threads:
        t.join()
    
    # 单独执行一次 worker 函数
    worker()
    
    # 对结果列表中的每个结果 r，检查其是否与结果列表中最后一个结果相近
    for r in results:
        assert_allclose(r, results[-1])
def test_reentering():
    # Just some linear operator that calls eigs recursively
    # 定义一个线性算子，递归调用 eigs 函数
    def A_matvec(x):
        # 创建一个带有对角元素的稀疏矩阵
        x = diags([1, -2, 1], [-1, 0, 1], shape=(50, 50))
        # 调用 eigs 函数求解特征值和特征向量
        w, v = eigs(x, k=1)
        # 返回特征向量除以特征值的第一个值
        return v / w[0]
    # 创建一个线性算子 A
    A = LinearOperator(matvec=A_matvec, dtype=float, shape=(50, 50))

    # 由于 Fortran 代码不支持重入，所以这里会触发 RuntimeError（优雅地失败，不会崩溃）
    assert_raises(RuntimeError, eigs, A, k=1)
    assert_raises(RuntimeError, eigsh, A, k=1)


def test_regression_arpackng_1315():
    # 检查是否存在 arpack-ng/#1315 问题
    # 改编自 arpack-ng/TESTS/bug_1315_single.c
    # 如果这里失败了，那么安装的 ARPACK 库可能存在问题。

    for dtype in [np.float32, np.float64]:
        np.random.seed(1234)

        # 创建一个对角矩阵 A
        w0 = np.arange(1, 1000+1).astype(dtype)
        A = diags([w0], [0], shape=(1000, 1000))

        # 创建一个随机向量 v0
        v0 = np.random.rand(1000).astype(dtype)
        # 调用 eigs 函数求解 A 的部分特征值和特征向量
        w, v = eigs(A, k=9, ncv=2*9+1, which="LM", v0=v0)

        # 断言近似相等，排序后的部分特征值与 w0 的后 9 个元素
        assert_allclose(np.sort(w), np.sort(w0[-9:]),
                        rtol=1e-4)


def test_eigs_for_k_greater():
    # 测试当 k 超出限制时的 eigs() 函数
    A_sparse = diags([1, -2, 1], [-1, 0, 1], shape=(4, 4))  # 稀疏矩阵
    A = generate_matrix(4, sparse=False)
    M_dense = np.random.random((4, 4))
    M_sparse = generate_matrix(4, sparse=True)
    M_linop = aslinearoperator(M_dense)
    eig_tuple1 = eig(A, b=M_dense)
    eig_tuple2 = eig(A, b=M_sparse)

    with suppress_warnings() as sup:
        sup.filter(RuntimeWarning)

        # 断言调用 eigs() 函数时，参数 k=3 时的结果与 eig() 的结果相等
        assert_equal(eigs(A, M=M_dense, k=3), eig_tuple1)
        assert_equal(eigs(A, M=M_dense, k=4), eig_tuple1)
        assert_equal(eigs(A, M=M_dense, k=5), eig_tuple1)
        assert_equal(eigs(A, M=M_sparse, k=5), eig_tuple2)

        # 当 M 是 LinearOperator 时，预期会触发 TypeError
        assert_raises(TypeError, eigs, A, M=M_linop, k=3)

        # 测试不同类型的 'A'，预期会触发 TypeError
        assert_raises(TypeError, eigs, aslinearoperator(A), k=3)
        assert_raises(TypeError, eigs, A_sparse, k=3)


def test_eigsh_for_k_greater():
    # 测试当 k 超出限制时的 eigsh() 函数
    A_sparse = diags([1, -2, 1], [-1, 0, 1], shape=(4, 4))  # 稀疏矩阵
    A = generate_matrix(4, sparse=False)
    M_dense = generate_matrix_symmetric(4, pos_definite=True)
    M_sparse = generate_matrix_symmetric(4, pos_definite=True, sparse=True)
    M_linop = aslinearoperator(M_dense)
    eig_tuple1 = eigh(A, b=M_dense)
    eig_tuple2 = eigh(A, b=M_sparse)

    with suppress_warnings() as sup:
        sup.filter(RuntimeWarning)

        # 断言调用 eigsh() 函数时，参数 k=4 和 k=5 时的结果与 eigh() 的结果相等
        assert_equal(eigsh(A, M=M_dense, k=4), eig_tuple1)
        assert_equal(eigsh(A, M=M_dense, k=5), eig_tuple1)
        assert_equal(eigsh(A, M=M_sparse, k=5), eig_tuple2)

        # 当 M 是 LinearOperator 时，预期会触发 TypeError
        assert_raises(TypeError, eigsh, A, M=M_linop, k=4)

        # 测试不同类型的 'A'，预期会触发 TypeError
        assert_raises(TypeError, eigsh, aslinearoperator(A), k=4)
        assert_raises(TypeError, eigsh, A_sparse, M=M_dense, k=4)


def test_real_eigs_real_k_subset():
    np.random.seed(1)

    n = 10
    # 生成一个随机稀疏矩阵A，大小为n x n，密度为0.5
    A = rand(n, n, density=0.5)
    # 将矩阵A中的所有元素乘以2
    A.data *= 2
    # 将矩阵A中的所有元素减去1
    A.data -= 1
    
    # 创建一个大小为n的全1向量v0
    v0 = np.ones(n)
    
    # 定义要测试的不同模式和数据类型
    whichs = ['LM', 'SM', 'LR', 'SR', 'LI', 'SI']
    dtypes = [np.float32, np.float64]
    
    # 使用itertools的product函数迭代所有whichs、sigma、dtypes的组合
    for which, sigma, dtype in itertools.product(whichs, [None, 0, 5], dtypes):
        # 初始化空的数组prev_w，数据类型为dtype
        prev_w = np.array([], dtype=dtype)
        # 获取dtype的机器精度
        eps = np.finfo(dtype).eps
        # 对于k从1到8进行迭代
        for k in range(1, 9):
            # 计算矩阵A的前k个特征值和对应的特征向量
            w, z = eigs(A.astype(dtype), k=k, which=which, sigma=sigma,
                        v0=v0.astype(dtype), tol=0)
            # 断言矩阵A乘以特征向量z减去特征值w乘以z的模为0，允许误差为sqrt(eps)
            assert_allclose(np.linalg.norm(A.dot(z) - z * w), 0, atol=np.sqrt(eps))
    
            # 检查第k个特征值集合是否是第k+1个特征值集合的子集
            dist = abs(prev_w[:,None] - w).min(axis=1)
            assert_allclose(dist, 0, atol=np.sqrt(eps))
    
            prev_w = w
    
            # 检查特征值的排序顺序
            if sigma is None:
                d = w
            else:
                d = 1 / (w - sigma)
    
            if which == 'LM':
                # 对于'LM'模式，ARPACK算法保证排序顺序，但对于其他模式排序顺序未明确定义
                assert np.all(np.diff(abs(d)) <= 1e-6)
```