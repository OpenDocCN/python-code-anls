# `D:\src\scipysrc\scipy\scipy\sparse\linalg\_eigen\arpack\arpack.py`

```
# 包含 ARPACK 的一些入口点
# ------------------------
# ARPACK 的入口点有：
# - (s,d)seupd：单精度和双精度对称矩阵
# - (s,d,c,z)neupd：单精度、双精度、复数和双精度复数的一般矩阵
# 此包装器将 *neupd（一般矩阵）接口放在 eigs() 中，
# *seupd（对称矩阵）接口放在 eigsh() 中。
# 没有专门用于复数 Hermite 矩阵的接口。
# 要找到复数 Hermite 矩阵的特征值，可以使用 eigsh()，
# 但 eigsh() 简单地调用 eigs() 并返回获得的特征值的实部。

# 返回的特征值数量和复数特征值
# -----------------------------
# ARPACK 的非对称实数和双精度接口 (s,d)naupd 返回特征值和特征向量，
# 存储在实数（float、double）数组中。
# 由于特征值和特征向量通常是复数的，ARPACK 将实部和虚部放在连续的实数数组条目中。
# 此包装器将实数条目放入复数数据类型中，并尝试返回请求的特征值和特征向量。

# 求解器模式
# -----------
# ARPACK 可以通过提供一个位移（sigma）和一个求解器来处理特征值的位移和位移逆运算。

import numpy as np
import warnings
from scipy.sparse.linalg._interface import aslinearoperator, LinearOperator
from scipy.sparse import eye, issparse
from scipy.linalg import eig, eigh, lu_factor, lu_solve
from scipy.sparse._sputils import (
    convert_pydata_sparse_to_scipy, isdense, is_pydata_spmatrix,
)
from scipy.sparse.linalg import gmres, splu
from scipy._lib._util import _aligned_zeros
from scipy._lib._threadsafety import ReentrancyLock

from . import _arpack
arpack_int = _arpack.timing.nbx.dtype

__docformat__ = "restructuredtext en"

__all__ = ['eigs', 'eigsh', 'ArpackError', 'ArpackNoConvergence']

# 数值类型转换表
_type_conv = {'f': 's', 'd': 'd', 'F': 'c', 'D': 'z'}
# 数字保留小数位数
_ndigits = {'f': 5, 'd': 12, 'F': 5, 'D': 12}

# DNAUPD 错误码及其含义
DNAUPD_ERRORS = {
    0: "Normal exit.",
    1: "Maximum number of iterations taken. "
       "All possible eigenvalues of OP has been found. IPARAM(5) "
       "returns the number of wanted converged Ritz values.",
    2: "No longer an informational error. Deprecated starting "
       "with release 2 of ARPACK.",
    3: "No shifts could be applied during a cycle of the "
       "Implicitly restarted Arnoldi iteration. One possibility "
       "is to increase the size of NCV relative to NEV. ",
    -1: "N must be positive.",
    -2: "NEV must be positive.",
    -3: "NCV-NEV >= 2 and less than or equal to N.",
    -4: "The maximum number of Arnoldi update iterations allowed "
        "must be greater than zero.",
    -5: " WHICH must be one of 'LM', 'SM', 'LR', 'SR', 'LI', 'SI'",
    -6: "BMAT must be one of 'I' or 'G'.",
}
    # 错误码 -7：私有工作数组 WORKL 的长度不足。
    -7: "Length of private work array WORKL is not sufficient.",
    
    # 错误码 -8：从 LAPACK 特征值计算中返回错误。
    -8: "Error return from LAPACK eigenvalue calculation;",
    
    # 错误码 -9：起始向量为零。
    -9: "Starting vector is zero.",
    
    # 错误码 -10：IPARAM(7) 必须是 1、2、3 或 4。
    -10: "IPARAM(7) must be 1,2,3,4.",
    
    # 错误码 -11：IPARAM(7) = 1 且 BMAT = 'G' 不兼容。
    -11: "IPARAM(7) = 1 and BMAT = 'G' are incompatible.",
    
    # 错误码 -12：IPARAM(1) 必须等于 0 或 1。
    -12: "IPARAM(1) must be equal to 0 or 1.",
    
    # 错误码 -13：NEV 和 WHICH = 'BE' 不兼容。
    -13: "NEV and WHICH = 'BE' are incompatible.",
    
    # 错误码 -9999：无法构建 Arnoldi 分解。
    # IPARAM(5) 返回当前 Arnoldi 分解的大小。
    # 用户建议检查是否分配了足够的工作空间和数组存储空间。
    -9999: "Could not build an Arnoldi factorization. "
           "IPARAM(5) returns the size of the current Arnoldi "
           "factorization. The user is advised to check that "
           "enough workspace and array storage has been allocated."
}

# 将 DNAUPD_ERRORS 赋值给 SNAUPD_ERRORS，它们指向同一个对象
SNAUPD_ERRORS = DNAUPD_ERRORS

# 复制 DNAUPD_ERRORS 对象，赋值给 ZNAUPD_ERRORS
ZNAUPD_ERRORS = DNAUPD_ERRORS.copy()
# 在 ZNAUPD_ERRORS 字典中添加键为 -10 的条目，对应值为特定错误消息
ZNAUPD_ERRORS[-10] = "IPARAM(7) must be 1,2,3."

# 将 ZNAUPD_ERRORS 赋值给 CNAUPD_ERRORS，它们指向同一个对象
CNAUPD_ERRORS = ZNAUPD_ERRORS

# 定义 DSAUPD_ERRORS 字典，包含整数到错误消息的映射
DSAUPD_ERRORS = {
    0: "Normal exit.",
    1: "Maximum number of iterations taken. "
       "All possible eigenvalues of OP has been found.",
    2: "No longer an informational error. Deprecated starting with "
       "release 2 of ARPACK.",
    3: "No shifts could be applied during a cycle of the Implicitly "
       "restarted Arnoldi iteration. One possibility is to increase "
       "the size of NCV relative to NEV. ",
    -1: "N must be positive.",
    -2: "NEV must be positive.",
    -3: "NCV must be greater than NEV and less than or equal to N.",
    -4: "The maximum number of Arnoldi update iterations allowed "
        "must be greater than zero.",
    -5: "WHICH must be one of 'LM', 'SM', 'LA', 'SA' or 'BE'.",
    -6: "BMAT must be one of 'I' or 'G'.",
    -7: "Length of private work array WORKL is not sufficient.",
    -8: "Error return from trid. eigenvalue calculation; "
        "Informational error from LAPACK routine dsteqr .",
    -9: "Starting vector is zero.",
    -10: "IPARAM(7) must be 1,2,3,4,5.",
    -11: "IPARAM(7) = 1 and BMAT = 'G' are incompatible.",
    -12: "IPARAM(1) must be equal to 0 or 1.",
    -13: "NEV and WHICH = 'BE' are incompatible. ",
    -9999: "Could not build an Arnoldi factorization. "
           "IPARAM(5) returns the size of the current Arnoldi "
           "factorization. The user is advised to check that "
           "enough workspace and array storage has been allocated.",
}

# 将 DSAUPD_ERRORS 赋值给 SSAUPD_ERRORS，它们指向同一个对象
SSAUPD_ERRORS = DSAUPD_ERRORS

# 定义 DNEUPD_ERRORS 字典，包含整数到错误消息的映射
DNEUPD_ERRORS = {
    0: "Normal exit.",
    1: "The Schur form computed by LAPACK routine dlahqr "
       "could not be reordered by LAPACK routine dtrsen. "
       "Re-enter subroutine dneupd  with IPARAM(5)NCV and "
       "increase the size of the arrays DR and DI to have "
       "dimension at least dimension NCV and allocate at least NCV "
       "columns for Z. NOTE: Not necessary if Z and V share "
       "the same space. Please notify the authors if this error"
       "occurs.",
    -1: "N must be positive.",
    -2: "NEV must be positive.",
    -3: "NCV-NEV >= 2 and less than or equal to N.",
    -5: "WHICH must be one of 'LM', 'SM', 'LR', 'SR', 'LI', 'SI'",
    -6: "BMAT must be one of 'I' or 'G'.",
    -7: "Length of private work WORKL array is not sufficient.",
    -8: "Error return from calculation of a real Schur form. "
        "Informational error from LAPACK routine dlahqr .",
    -9: "Error return from calculation of eigenvectors. "
        "Informational error from LAPACK routine dtrevc.",
    -10: "IPARAM(7) must be 1,2,3,4.",
    -11: "IPARAM(7) = 1 and BMAT = 'G' are incompatible.",
    -12: "HOWMNY = 'S' not yet implemented",
    -13: "HOWMNY must be one of 'A' or 'P' if RVEC = .true.",
    -14: "DNAUPD  did not find any eigenvalues to sufficient "
         "accuracy.",
}
    -15: """
         "DNEUPD got a different count of the number of converged "
         "Ritz values than DNAUPD got.  This indicates the user "
         "probably made an error in passing data from DNAUPD to "
         "DNEUPD or that the data was modified before entering "
         "DNEUPD"
         """



# 第-15行：警告信息字符串，指出DNEUPD和DNAUPD在收敛的瑞兹值数量上有不同的计数。
# 这表明用户可能在从DNAUPD传递数据到DNEUPD时出现错误，或者在输入DNEUPD之前修改了数据。
}

# 复制 DNEUPD_ERRORS 字典并赋值给 SNEUPD_ERRORS，保留原有内容
SNEUPD_ERRORS = DNEUPD_ERRORS.copy()

# 添加自定义错误消息，键为 1
SNEUPD_ERRORS[1] = ("The Schur form computed by LAPACK routine slahqr "
                    "could not be reordered by LAPACK routine strsen . "
                    "Re-enter subroutine dneupd  with IPARAM(5)=NCV and "
                    "increase the size of the arrays DR and DI to have "
                    "dimension at least dimension NCV and allocate at least "
                    "NCV columns for Z. NOTE: Not necessary if Z and V share "
                    "the same space. Please notify the authors if this error "
                    "occurs.")

# 添加自定义错误消息，键为 -14
SNEUPD_ERRORS[-14] = ("SNAUPD did not find any eigenvalues to sufficient "
                      "accuracy.")

# 添加自定义错误消息，键为 -15
SNEUPD_ERRORS[-15] = ("SNEUPD got a different count of the number of "
                      "converged Ritz values than SNAUPD got.  This indicates "
                      "the user probably made an error in passing data from "
                      "SNAUPD to SNEUPD or that the data was modified before "
                      "entering SNEUPD")
# 定义一个字典，用于存储 ZNEUPD 的错误码及其对应的错误描述
ZNEUPD_ERRORS = {
    0: "Normal exit.",
    1: "The Schur form computed by LAPACK routine csheqr "
       "could not be reordered by LAPACK routine ztrsen. "
       "Re-enter subroutine zneupd with IPARAM(5)=NCV and "
       "increase the size of the array D to have "
       "dimension at least dimension NCV and allocate at least "
       "NCV columns for Z. NOTE: Not necessary if Z and V share "
       "the same space. Please notify the authors if this error "
       "occurs.",
    -1: "N must be positive.",
    -2: "NEV must be positive.",
    -3: "NCV-NEV >= 1 and less than or equal to N.",
    -5: "WHICH must be one of 'LM', 'SM', 'LR', 'SR', 'LI', 'SI'",
    -6: "BMAT must be one of 'I' or 'G'.",
    -7: "Length of private work WORKL array is not sufficient.",
    -8: "Error return from LAPACK eigenvalue calculation. "
        "This should never happened.",
    -9: "Error return from calculation of eigenvectors. "
        "Informational error from LAPACK routine ztrevc.",
    -10: "IPARAM(7) must be 1,2,3",
    -11: "IPARAM(7) = 1 and BMAT = 'G' are incompatible.",
    -12: "HOWMNY = 'S' not yet implemented",
    -13: "HOWMNY must be one of 'A' or 'P' if RVEC = .true.",
    -14: "ZNAUPD did not find any eigenvalues to sufficient "
         "accuracy.",
    -15: "ZNEUPD got a different count of the number of "
         "converged Ritz values than ZNAUPD got.  This "
         "indicates the user probably made an error in passing "
         "data from ZNAUPD to ZNEUPD or that the data was "
         "modified before entering ZNEUPD"
}

# 通过复制 ZNEUPD_ERRORS 字典来创建一个新的字典 CNEUPD_ERRORS，
# 并修改其中两个特定错误码的描述信息
CNEUPD_ERRORS = ZNEUPD_ERRORS.copy()
CNEUPD_ERRORS[-14] = ("CNAUPD did not find any eigenvalues to sufficient "
                      "accuracy.")
CNEUPD_ERRORS[-15] = ("CNEUPD got a different count of the number of "
                      "converged Ritz values than CNAUPD got.  This indicates "
                      "the user probably made an error in passing data from "
                      "CNAUPD to CNEUPD or that the data was modified before "
                      "entering CNEUPD")

# 定义一个字典，用于存储 DSEUPD 的错误码及其对应的错误描述
DSEUPD_ERRORS = {
    0: "Normal exit.",
    -1: "N must be positive.",
    -2: "NEV must be positive.",
    -3: "NCV must be greater than NEV and less than or equal to N.",
    -5: "WHICH must be one of 'LM', 'SM', 'LA', 'SA' or 'BE'.",
    -6: "BMAT must be one of 'I' or 'G'.",
    -7: "Length of private work WORKL array is not sufficient.",
    -8: ("Error return from trid. eigenvalue calculation; "
         "Information error from LAPACK routine dsteqr."),
    -9: "Starting vector is zero.",
    -10: "IPARAM(7) must be 1,2,3,4,5.",
}
    # 错误代码 -11: "IPARAM(7) = 1 and BMAT = 'G' are incompatible."
        # IPARAM(7) 等于 1 和 BMAT 等于 'G' 是不兼容的情况。
    
    # 错误代码 -12: "NEV and WHICH = 'BE' are incompatible."
        # NEV 和 WHICH 等于 'BE' 是不兼容的情况。
    
    # 错误代码 -14: "DSAUPD  did not find any eigenvalues to sufficient accuracy."
        # DSAUPD 没有找到足够精度的特征值。
    
    # 错误代码 -15: "HOWMNY must be one of 'A' or 'S' if RVEC = .true."
        # 如果 RVEC 为真，则 HOWMNY 必须是 'A' 或 'S' 中的一个。
    
    # 错误代码 -16: "HOWMNY = 'S' not yet implemented"
        # HOWMNY = 'S' 还没有实现。
    
    # 错误代码 -17: ("DSEUPD  got a different count of the number of converged "
    #           "Ritz values than DSAUPD  got.  This indicates the user "
    #           "probably made an error in passing data from DSAUPD  to "
    #           "DSEUPD  or that the data was modified before entering  "
    #           "DSEUPD.")
        # DSEUPD 得到的收敛瑞兹值数量与 DSAUPD 得到的不同。这表明用户可能在从 DSAUPD 传递数据给 DSEUPD 时出错，或者数据在进入 DSEUPD 之前被修改了。
}

# 复制 DSEUPD_ERRORS 到 SSEUPD_ERRORS
SSEUPD_ERRORS = DSEUPD_ERRORS.copy()

# 设置 SSEUPD_ERRORS[-14] 的错误消息
SSEUPD_ERRORS[-14] = ("SSAUPD  did not find any eigenvalues "
                      "to sufficient accuracy.")

# 设置 SSEUPD_ERRORS[-17] 的错误消息
SSEUPD_ERRORS[-17] = ("SSEUPD  got a different count of the number of "
                      "converged "
                      "Ritz values than SSAUPD  got.  This indicates the user "
                      "probably made an error in passing data from SSAUPD  to "
                      "SSEUPD  or that the data was modified before entering  "
                      "SSEUPD.")

# 定义 _SAUPD_ERRORS 字典，包含 'd' 和 's' 错误信息
_SAUPD_ERRORS = {'d': DSAUPD_ERRORS,
                 's': SSAUPD_ERRORS}

# 定义 _NAUPD_ERRORS 字典，包含 'd', 's', 'z', 'c' 错误信息
_NAUPD_ERRORS = {'d': DNAUPD_ERRORS,
                 's': SNAUPD_ERRORS,
                 'z': ZNAUPD_ERRORS,
                 'c': CNAUPD_ERRORS}

# 定义 _SEUPD_ERRORS 字典，包含 'd' 和 's' 错误信息
_SEUPD_ERRORS = {'d': DSEUPD_ERRORS,
                 's': SSEUPD_ERRORS}

# 定义 _NEUPD_ERRORS 字典，包含 'd', 's', 'z', 'c' 错误信息
_NEUPD_ERRORS = {'d': DNEUPD_ERRORS,
                 's': SNEUPD_ERRORS,
                 'z': ZNEUPD_ERRORS,
                 'c': CNEUPD_ERRORS}

# _SEUPD_WHICH 参数的有效值列表
_SEUPD_WHICH = ['LM', 'SM', 'LA', 'SA', 'BE']

# _NEUPD_WHICH 参数的有效值列表
_NEUPD_WHICH = ['LM', 'SM', 'LR', 'SR', 'LI', 'SI']


class ArpackError(RuntimeError):
    """
    ARPACK error
    """

    def __init__(self, info, infodict=_NAUPD_ERRORS):
        # 根据 info 获取错误消息，如果未知则返回 "Unknown error"
        msg = infodict.get(info, "Unknown error")
        # 调用父类的初始化方法，设置错误消息
        RuntimeError.__init__(self, "ARPACK error %d: %s" % (info, msg))


class ArpackNoConvergence(ArpackError):
    """
    ARPACK iteration did not converge

    Attributes
    ----------
    eigenvalues : ndarray
        Partial result. Converged eigenvalues.
    eigenvectors : ndarray
        Partial result. Converged eigenvectors.

    """

    def __init__(self, msg, eigenvalues, eigenvectors):
        # 调用父类的初始化方法，设置错误消息为 msg
        ArpackError.__init__(self, -1, {-1: msg})
        # 设置 eigenvalues 和 eigenvectors 属性
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors


def choose_ncv(k):
    """
    Choose number of lanczos vectors based on target number
    of singular/eigen values and vectors to compute, k.
    """
    # 计算并返回 lanczos 向量的数量
    return max(2 * k + 1, 20)


class _ArpackParams:
    # 在此类中定义 ARPACK 参数相关的其他内容
    # 初始化方法，接受多个参数，并设置默认值
    def __init__(self, n, k, tp, mode=1, sigma=None,
                 ncv=None, v0=None, maxiter=None, which="LM", tol=0):
        # 检查 k 是否为正数，否则抛出值错误异常
        if k <= 0:
            raise ValueError("k must be positive, k=%d" % k)

        # 如果 maxiter 为 None，则设为 n 的 10 倍
        if maxiter is None:
            maxiter = n * 10
        # 检查 maxiter 是否为正数，否则抛出值错误异常
        if maxiter <= 0:
            raise ValueError("maxiter must be positive, maxiter=%d" % maxiter)

        # 检查 tp 是否为 'f', 'd', 'F', 'D' 中的一个，否则抛出值错误异常
        if tp not in 'fdFD':
            raise ValueError("matrix type must be 'f', 'd', 'F', or 'D'")

        # 如果 v0 不为 None，则使用其创建一个数组副本，否则创建一个类型为 tp 的 n 长度的零数组
        if v0 is not None:
            # ARPACK 会覆盖其初始残差，这里创建一个副本
            self.resid = np.array(v0, copy=True)
            info = 1
        else:
            # ARPACK 将使用随机的初始向量
            self.resid = np.zeros(n, tp)
            info = 0

        # 如果 sigma 为 None，则设为 0，否则设为给定的值
        if sigma is None:
            # sigma 没有被使用
            self.sigma = 0
        else:
            self.sigma = sigma

        # 如果 ncv 为 None，则根据 k 选择一个默认值
        if ncv is None:
            ncv = choose_ncv(k)
        # ncv 取 ncv 和 n 之间的较小值
        ncv = min(ncv, n)

        # 创建一个 n 行 ncv 列的零数组，用于存储 Ritz 向量
        self.v = np.zeros((n, ncv), tp)  # holds Ritz vectors
        # 创建一个包含 11 个 arpack_int 类型元素的零数组
        self.iparam = np.zeros(11, arpack_int)

        # 设置求解器的模式和参数
        ishfts = 1
        self.mode = mode
        self.iparam[0] = ishfts
        self.iparam[2] = maxiter
        self.iparam[3] = 1
        self.iparam[6] = mode

        # 设置对象的各种属性值
        self.n = n
        self.tol = tol
        self.k = k
        self.maxiter = maxiter
        self.ncv = ncv
        self.which = which
        self.tp = tp
        self.info = info

        # 初始化收敛状态为 False，迭代操作的状态为 0
        self.converged = False
        self.ido = 0

    # 抛出无收敛异常
    def _raise_no_convergence(self):
        msg = "No convergence (%d iterations, %d/%d eigenvectors converged)"
        k_ok = self.iparam[4]
        num_iter = self.iparam[2]
        try:
            ev, vec = self.extract(True)
        except ArpackError as err:
            msg = f"{msg} [{err}]"
            ev = np.zeros((0,))
            vec = np.zeros((self.n, 0))
            k_ok = 0
        # 抛出 ArpackNoConvergence 异常，包含迭代次数、成功收敛的特征值和特征向量
        raise ArpackNoConvergence(msg % (num_iter, k_ok, self.k), ev, vec)
# 继承自 _ArpackParams 的 _SymmetricArpackParams 类，实现了 ARPACK 迭代方法和结果提取方法

class _SymmetricArpackParams(_ArpackParams):
    def iterate(self):
        # 调用 ARPACK 求解器进行一次迭代，更新各参数和状态
        self.ido, self.tol, self.resid, self.v, self.iparam, self.ipntr, self.info = \
            self._arpack_solver(self.ido, self.bmat, self.which, self.k,
                                self.tol, self.resid, self.v, self.iparam,
                                self.ipntr, self.workd, self.workl, self.info)

        # 切片操作，获取对应的工作向量段
        xslice = slice(self.ipntr[0] - 1, self.ipntr[0] - 1 + self.n)
        yslice = slice(self.ipntr[1] - 1, self.ipntr[1] - 1 + self.n)
        if self.ido == -1:
            # 初始化操作
            self.workd[yslice] = self.OP(self.workd[xslice])
        elif self.ido == 1:
            # 计算 y = Op*x
            if self.mode == 1:
                self.workd[yslice] = self.OP(self.workd[xslice])
            elif self.mode == 2:
                # 处理不对称模式下的操作
                self.workd[xslice] = self.OPb(self.workd[xslice])
                self.workd[yslice] = self.OPa(self.workd[xslice])
            elif self.mode == 5:
                # 处理广义特征值问题中的操作
                Bxslice = slice(self.ipntr[2] - 1, self.ipntr[2] - 1 + self.n)
                Ax = self.A_matvec(self.workd[xslice])
                self.workd[yslice] = self.OPa(Ax + (self.sigma *
                                                    self.workd[Bxslice]))
            else:
                # 处理其他模式下的操作
                Bxslice = slice(self.ipntr[2] - 1, self.ipntr[2] - 1 + self.n)
                self.workd[yslice] = self.OPa(self.workd[Bxslice])
        elif self.ido == 2:
            # 计算 y = B*x
            self.workd[yslice] = self.B(self.workd[xslice])
        elif self.ido == 3:
            # ARPACK 请求用户进行移位操作，抛出异常
            raise ValueError("ARPACK requested user shifts.  Assure ISHIFT==0")
        else:
            # 收敛状态标志
            self.converged = True

            if self.info == 0:
                pass
            elif self.info == 1:
                # 若 info 状态指示未收敛，抛出异常
                self._raise_no_convergence()
            else:
                # 其他 info 状态下抛出 ARPACK 错误
                raise ArpackError(self.info, infodict=self.iterate_infodict)

    def extract(self, return_eigenvectors):
        # 提取特征值和特征向量
        rvec = return_eigenvectors
        ierr = 0
        howmny = 'A'  # 返回所有特征向量
        sselect = np.zeros(self.ncv, 'int')  # 未使用的选择数组
        # 调用 ARPACK 提取方法，获取特征值和特征向量
        d, z, ierr = self._arpack_extract(rvec, howmny, sselect, self.sigma,
                                          self.bmat, self.which, self.k,
                                          self.tol, self.resid, self.v,
                                          self.iparam[0:7], self.ipntr,
                                          self.workd[0:2 * self.n],
                                          self.workl, ierr)
        if ierr != 0:
            # 若提取过程中出现错误，抛出 ARPACK 错误
            raise ArpackError(ierr, infodict=self.extract_infodict)
        k_ok = self.iparam[4]
        d = d[:k_ok]
        z = z[:, :k_ok]

        if return_eigenvectors:
            # 若需返回特征向量，则返回特征值和特征向量
            return d, z
        else:
            # 若仅需返回特征值，则返回特征值
            return d
    def iterate(self):
        # 如果 self.tp 是 'f' 或者 'd'，使用特定参数调用 _arpack_solver 方法
        if self.tp in 'fd':
            results = self._arpack_solver(self.ido, self.bmat, self.which, self.k,
                                          self.tol, self.resid, self.v, self.iparam,
                                          self.ipntr, self.workd, self.workl, self.info)
        else:
            # 否则使用另一组参数调用 _arpack_solver 方法
            results = self._arpack_solver(self.ido, self.bmat, self.which, self.k,
                                          self.tol, self.resid, self.v, self.iparam,
                                          self.ipntr, self.workd, self.workl,
                                          self.rwork, self.info)
        
        # 解包 results 到相应的实例变量
        self.ido, self.tol, self.resid, self.v, \
            self.iparam, self.ipntr, self.info = results

        # 根据 ipntr 的指示，创建对应的切片对象
        xslice = slice(self.ipntr[0] - 1, self.ipntr[0] - 1 + self.n)
        yslice = slice(self.ipntr[1] - 1, self.ipntr[1] - 1 + self.n)
        
        # 根据 ido 的值执行不同的操作
        if self.ido == -1:
            # 初始化操作
            self.workd[yslice] = self.OP(self.workd[xslice])
        elif self.ido == 1:
            # 计算 y = Op*x
            if self.mode in (1, 2):
                self.workd[yslice] = self.OP(self.workd[xslice])
            else:
                # 使用另一种操作符 OPa 计算
                Bxslice = slice(self.ipntr[2] - 1, self.ipntr[2] - 1 + self.n)
                self.workd[yslice] = self.OPa(self.workd[Bxslice])
        elif self.ido == 2:
            # 使用操作符 B 计算
            self.workd[yslice] = self.B(self.workd[xslice])
        elif self.ido == 3:
            # 如果 ido 为 3，则抛出异常
            raise ValueError("ARPACK requested user shifts.  Assure ISHIFT==0")
        else:
            # 其他情况下，标记收敛为 True
            self.converged = True

            if self.info == 0:
                # 如果 info 为 0，表示正常结束，不做任何操作
                pass
            elif self.info == 1:
                # 如果 info 为 1，调用 _raise_no_convergence 方法
                self._raise_no_convergence()
            else:
                # 否则，抛出 ArpackError 异常
                raise ArpackError(self.info, infodict=self.iterate_infodict)
# 将输入参数 m 转换为线性操作器对象，并确保其具有 dtype 属性
def _aslinearoperator_with_dtype(m):
    m = aslinearoperator(m)
    # 如果 m 没有 dtype 属性，则创建一个形状与 m 第二维度大小相同的零向量 x，并设置 m 的 dtype 为 (m * x) 的 dtype
    if not hasattr(m, 'dtype'):
        x = np.zeros(m.shape[1])
        m.dtype = (m * x).dtype
    return m


class SpLuInv(LinearOperator):
    """
    SpLuInv:
       使用 M 的稀疏 LU 分解重复解决 M*x=b 的辅助类
    """

    def __init__(self, M):
        # 使用 scipy.sparse.linalg.splu 对象初始化 SpLuInv 实例的 M_lu 属性
        self.M_lu = splu(M)
        # 设置实例的形状和数据类型属性为 M 的形状和数据类型
        self.shape = M.shape
        self.dtype = M.dtype
        # 设置实例的 isreal 属性为 True，如果 M 的数据类型不是复数类型
        self.isreal = not np.issubdtype(self.dtype, np.complexfloating)

    def _matvec(self, x):
        # 将输入向量 x 转换为 ndarray 对象
        x = np.asarray(x)
        # 如果实例的数据类型为实数，并且 x 的数据类型为复数浮点型，则分别求解实部和虚部，并返回复数结果
        if self.isreal and np.issubdtype(x.dtype, np.complexfloating):
            return (self.M_lu.solve(np.real(x).astype(self.dtype))
                    + 1j * self.M_lu.solve(np.imag(x).astype(self.dtype)))
        else:
            # 否则，直接求解 M_lu * x 并返回结果
            return self.M_lu.solve(x.astype(self.dtype))


class LuInv(LinearOperator):
    """
    LuInv:
       使用 M 的 LU 分解重复解决 M*x=b 的辅助类
    """

    def __init__(self, M):
        # 使用 scipy.linalg.lu_factor 对象初始化 LuInv 实例的 M_lu 属性
        self.M_lu = lu_factor(M)
        # 设置实例的形状和数据类型属性为 M 的形状和数据类型
        self.shape = M.shape
        self.dtype = M.dtype

    def _matvec(self, x):
        # 使用 scipy.linalg.lu_solve 求解线性方程组 M_lu * x = b，并返回结果
        return lu_solve(self.M_lu, x)


def gmres_loose(A, b, tol):
    """
    使用宽松终止条件的 GMRES 方法。
    """
    # 将输入向量 b 转换为 ndarray 对象
    b = np.asarray(b)
    # 计算最小容忍度 min_tol，以确保 GMRES 方法的相对容忍度不小于机器精度的平方根乘以 b 的大小
    min_tol = 1000 * np.sqrt(b.size) * np.finfo(b.dtype).eps
    # 调用 scipy.sparse.linalg.gmres 方法解决线性方程组 A * x = b，使用给定的相对容忍度和绝对容忍度
    return gmres(A, b, rtol=max(tol, min_tol), atol=0)


class IterInv(LinearOperator):
    """
    IterInv:
       使用迭代方法重复解决 M*x=b 的辅助类
    """

    def __init__(self, M, ifunc=gmres_loose, tol=0):
        self.M = M
        # 如果 M 具有 dtype 属性，则将实例的 dtype 设置为 M 的数据类型；否则，创建一个与 M 第二维度大小相同的零向量 x，并设置实例的 dtype 为 (M * x) 的数据类型
        if hasattr(M, 'dtype'):
            self.dtype = M.dtype
        else:
            x = np.zeros(M.shape[1])
            self.dtype = (M * x).dtype
        # 设置实例的形状属性为 M 的形状
        self.shape = M.shape

        # 当 tol <= 0 时，将实例的 tol 属性设置为机器精度的两倍
        if tol <= 0:
            # 当 tol=0 时，ARPACK 使用由 LAPACK 的 _LAMCH 函数计算的机器容忍度
            tol = 2 * np.finfo(self.dtype).eps
        self.ifunc = ifunc  # 设置实例的 ifunc 属性为输入的解线性方程组的函数
        self.tol = tol  # 设置实例的 tol 属性为输入的容忍度

    def _matvec(self, x):
        # 使用 ifunc 方法求解线性方程组 M * x = b，返回解 b 和信息 info
        b, info = self.ifunc(self.M, x, tol=self.tol)
        # 如果 info 不为 0，则抛出 ValueError 异常
        if info != 0:
            raise ValueError("Error in inverting M: function "
                             "%s did not converge (info = %i)."
                             % (self.ifunc.__name__, info))
        # 返回求解结果 b
        return b


class IterOpInv(LinearOperator):
    """
    IterOpInv:
       使用迭代方法重复解决 [A-sigma*M]*x = b 的辅助类
    """

    # 在这里添加 __init__ 方法的注释
    # 初始化函数，接受矩阵 A, M 和参数 sigma，并可选地接受求解函数 ifunc 和容差 tol
    def __init__(self, A, M, sigma, ifunc=gmres_loose, tol=0):
        # 将参数 A 赋给实例变量 self.A
        self.A = A
        # 将参数 M 赋给实例变量 self.M
        self.M = M
        # 将参数 sigma 赋给实例变量 self.sigma

        # 定义一个用于乘法运算的函数 mult_func，根据 M 是否为 None 使用不同的方式计算
        def mult_func(x):
            return A.matvec(x) - sigma * M.matvec(x)

        # 定义另一个用于乘法运算的函数 mult_func_M_None，当 M 为 None 时使用简化的计算方式
        def mult_func_M_None(x):
            return A.matvec(x) - sigma * x

        # 创建一个零向量 x，其长度为 A 的列数
        x = np.zeros(A.shape[1])
        
        # 根据 M 的值选择合适的乘法函数和数据类型
        if M is None:
            # 如果 M 为 None，则使用 mult_func_M_None 的返回类型作为数据类型
            dtype = mult_func_M_None(x).dtype
            # 创建一个线性操作对象 OP，使用 mult_func_M_None 作为乘法函数
            self.OP = LinearOperator(self.A.shape,
                                     mult_func_M_None,
                                     dtype=dtype)
        else:
            # 否则使用 mult_func 的返回类型作为数据类型
            dtype = mult_func(x).dtype
            # 创建一个线性操作对象 OP，使用 mult_func 作为乘法函数
            self.OP = LinearOperator(self.A.shape,
                                     mult_func,
                                     dtype=dtype)
        
        # 设置实例变量 shape 为 A 的形状
        self.shape = A.shape

        # 如果传入的 tol 小于等于 0，则设置 tol 为机器精度的两倍
        if tol <= 0:
            # 当 tol=0 时，ARPACK 使用 LAPACK 的 _LAMCH 函数计算的机器精度
            tol = 2 * np.finfo(self.OP.dtype).eps
        
        # 将 ifunc 和 tol 分别赋给实例变量 ifunc 和 tol
        self.ifunc = ifunc
        self.tol = tol

    # 定义矩阵向量乘法函数 _matvec，使用 ifunc 求解器求解线性方程组
    def _matvec(self, x):
        # 调用 ifunc 求解器求解线性方程组 OP * x = b，返回 b 和 info
        b, info = self.ifunc(self.OP, x, tol=self.tol)
        # 如果求解未收敛（info != 0），抛出值错误异常
        if info != 0:
            raise ValueError("Error in inverting [A-sigma*M]: function "
                             "%s did not converge (info = %i)."
                             % (self.ifunc.__name__, info))
        # 返回求解得到的向量 b
        return b

    # 定义属性函数 dtype，返回线性操作对象 OP 的数据类型
    @property
    def dtype(self):
        return self.OP.dtype
# 定义函数以将稀疏矩阵快速转换为 CSC 格式（如果可能则进行转置）
def _fast_spmatrix_to_csc(A, hermitian=False):
    if (A.format == "csr" and hermitian
            and not np.issubdtype(A.dtype, np.complexfloating)):
        # 如果输入矩阵 A 的格式是 CSR 并且要求共轭转置，并且 A 的数据类型不是复数浮点型，则返回 A 的转置
        return A.T
    elif is_pydata_spmatrix(A):
        # 如果输入矩阵 A 是 PyData 稀疏矩阵，则无需转换，直接返回 A
        return A
    else:
        # 否则将 A 转换为 CSC 格式并返回
        return A.tocsc()


# 定义函数以获取稀疏或稠密矩阵 M 的逆乘向量函数
def get_inv_matvec(M, hermitian=False, tol=0):
    if isdense(M):
        # 如果 M 是稠密矩阵，则返回 LuInv(M).matvec 函数
        return LuInv(M).matvec
    elif issparse(M) or is_pydata_spmatrix(M):
        # 如果 M 是稀疏矩阵或 PyData 稀疏矩阵，则先将 M 转换为 CSC 格式，然后返回 SpLuInv(M).matvec 函数
        M = _fast_spmatrix_to_csc(M, hermitian=hermitian)
        return SpLuInv(M).matvec
    else:
        # 否则返回 IterInv(M, tol=tol).matvec 函数
        return IterInv(M, tol=tol).matvec


# 定义函数以获取矩阵 A 的逆乘向量函数
def get_OPinv_matvec(A, M, sigma, hermitian=False, tol=0):
    if sigma == 0:
        # 如果 sigma 为 0，则返回 get_inv_matvec(A, hermitian=hermitian, tol=tol) 函数
        return get_inv_matvec(A, hermitian=hermitian, tol=tol)

    if M is None:
        # 如果 M 为 None，则表示 M 是单位矩阵
        if isdense(A):
            # 如果 A 是稠密矩阵
            if (np.issubdtype(A.dtype, np.complexfloating)
                    or np.imag(sigma) == 0):
                # 如果 A 的数据类型是复数浮点型，或者 sigma 的虚部为 0，则创建 A 的副本
                A = np.copy(A)
            else:
                # 否则将 A 转换为复数浮点型
                A = A + 0j
            # 对角线上的元素减去 sigma
            A.flat[::A.shape[1] + 1] -= sigma
            # 返回 LuInv(A).matvec 函数
            return LuInv(A).matvec
        elif issparse(A) or is_pydata_spmatrix(A):
            # 如果 A 是稀疏矩阵或 PyData 稀疏矩阵，则构建 A - sigma * 单位矩阵，并将其转换为 CSC 格式
            A = A - sigma * eye(A.shape[0])
            A = _fast_spmatrix_to_csc(A, hermitian=hermitian)
            # 返回 SpLuInv(A).matvec 函数
            return SpLuInv(A).matvec
        else:
            # 否则返回 IterOpInv(_aslinearoperator_with_dtype(A), M, sigma, tol=tol).matvec 函数
            return IterOpInv(_aslinearoperator_with_dtype(A),
                             M, sigma, tol=tol).matvec
    else:
        # 如果 M 不为 None
        if ((not isdense(A) and not issparse(A) and not is_pydata_spmatrix(A)) or
                (not isdense(M) and not issparse(M) and not is_pydata_spmatrix(A))):
            # 如果 A 和 M 都不是稠密矩阵、稀疏矩阵或 PyData 稀疏矩阵，则返回 IterOpInv(_aslinearoperator_with_dtype(A), _aslinearoperator_with_dtype(M), sigma, tol=tol).matvec 函数
            return IterOpInv(_aslinearoperator_with_dtype(A),
                             _aslinearoperator_with_dtype(M),
                             sigma, tol=tol).matvec
        elif isdense(A) or isdense(M):
            # 如果 A 或 M 是稠密矩阵，则返回 LuInv(A - sigma * M).matvec 函数
            return LuInv(A - sigma * M).matvec
        else:
            # 否则构建 OP = A - sigma * M，并将其转换为 CSC 格式，然后返回 SpLuInv(OP).matvec 函数
            OP = A - sigma * M
            OP = _fast_spmatrix_to_csc(OP, hermitian=hermitian)
            return SpLuInv(OP).matvec


# 定义 ARPACK 锁和重新进入检查
_ARPACK_LOCK = ReentrancyLock("Nested calls to eigs/eighs not allowed: ARPACK is not re-entrant")
    A : ndarray, sparse matrix or LinearOperator
        # A 是一个数组、稀疏矩阵或线性操作器，表示运算 ``A @ x``，其中 A 是一个实数或复数方阵。

    k : int, optional
        # 所需的特征值和特征向量的数量。`k` 必须小于 N-1。无法计算矩阵的所有特征向量。

    M : ndarray, sparse matrix or LinearOperator, optional
        # 表示广义特征值问题中 M@x 运算的数组、稀疏矩阵或线性操作器

            A @ x = w * M @ x.

        # 如果 A 是实数，则 M 必须表示一个实对称矩阵；如果 A 是复数，则 M 必须表示一个复 Hermite 矩阵。
        # 为获得最佳结果，M 的数据类型应与 A 的数据类型相同。
        # 此外：

            如果 `sigma` 为 None，则 M 是正定的。

            如果指定了 sigma，则 M 是正半定的。

        # 如果 sigma 为 None，则 eigs 需要一个操作器来计算线性方程的解 ``M @ x = b``。
        # 对于显式矩阵 M，内部使用 (稀疏) LU 分解来进行计算，对于一般线性操作器，则使用迭代求解器。
        # 另外，用户可以提供矩阵或操作器 Minv，给出 ``x = Minv @ b = M^-1 @ b``。

    sigma : real or complex, optional
        # 使用移位反转模式寻找接近 sigma 的特征值。这需要一个操作器来计算线性系统的解
        # ``[A - sigma * M] @ x = b``，其中如果未指定，则 M 是单位矩阵。对于显式矩阵 A 和 M，
        # 内部使用 (稀疏) LU 分解来进行计算，对于一般线性操作器，则使用迭代求解器。
        # 另外，用户可以提供矩阵或操作器 OPinv，给出 ``x = OPinv @ b = [A - sigma * M]^-1 @ b``。
        # 对于实矩阵 A，移位反转可以在虚数模式或实数模式下执行，由参数 OPpart（'r' 或 'i'）指定。
        # 注意，当指定 sigma 时，关键字 'which'（下文）指的是移位后的特征值 ``w'[i]``，其中：

            如果 A 是实数且 OPpart == 'r'（默认），
              ``w'[i] = 1/2 * [1/(w[i]-sigma) + 1/(w[i]-conj(sigma))]``。

            如果 A 是实数且 OPpart == 'i'，
              ``w'[i] = 1/2i * [1/(w[i]-sigma) - 1/(w[i]-conj(sigma))]``。

            如果 A 是复数，则 ``w'[i] = 1/(w[i]-sigma)``。

    v0 : ndarray, optional
        # 迭代的起始向量。默认值为随机向量。

    ncv : int, optional
        # 生成的 Lanczos 向量的数量，`ncv` 必须大于 `k`；建议 ``ncv > 2*k``。
        # 默认值为 ``min(n, max(2*k + 1, 20))``
    # 将输入的稀疏矩阵 A 转换为 SciPy 稀疏矩阵格式
    A = convert_pydata_sparse_to_scipy(A)
    # 将输入的稀疏矩阵 M 转换为 SciPy 稀疏矩阵格式
    M = convert_pydata_sparse_to_scipy(M)
    # 如果转换后的矩阵 A 不是方阵，则抛出数值错误异常
    if A.shape[0] != A.shape[1]:
        raise ValueError(f'expected square matrix (shape={A.shape})')
    # 如果矩阵 M 不为 None，则进行以下判断和警告
    if M is not None:
        # 检查 M 的形状是否与 A 的形状相同，如果不同则引发 ValueError 异常
        if M.shape != A.shape:
            raise ValueError(f'wrong M dimensions {M.shape}, should be {A.shape}')
        # 检查 M 的数据类型是否与 A 的数据类型相同（忽略大小写），如果不同则发出警告
        if np.dtype(M.dtype).char.lower() != np.dtype(A.dtype).char.lower():
            warnings.warn('M does not have the same type precision as A. '
                          'This may adversely affect ARPACK convergence',
                          stacklevel=2)

    # 获取矩阵 A 的行数
    n = A.shape[0]

    # 检查所需的特征值数量 k 是否合理
    if k <= 0:
        raise ValueError("k=%d must be greater than 0." % k)

    # 如果 k 大于等于矩阵 A 的行数减一，发出运行时警告并尝试使用 scipy.linalg.eig
    if k >= n - 1:
        warnings.warn("k >= N - 1 for N * N square matrix. "
                      "Attempting to use scipy.linalg.eig instead.",
                      RuntimeWarning, stacklevel=2)

        # 对于稀疏矩阵 A，无法直接使用 scipy.linalg.eig，需要先转换为稠密数组
        if issparse(A):
            raise TypeError("Cannot use scipy.linalg.eig for sparse A with "
                            "k >= N - 1. Use scipy.linalg.eig(A.toarray()) or"
                            " reduce k.")
        # 对于 LinearOperator 类型的矩阵 A 或 M，同样无法直接使用 scipy.linalg.eig
        if isinstance(A, LinearOperator):
            raise TypeError("Cannot use scipy.linalg.eig for LinearOperator "
                            "A with k >= N - 1.")
        # 对于 LinearOperator 类型的矩阵 M，同样无法直接使用 scipy.linalg.eig
        if isinstance(M, LinearOperator):
            raise TypeError("Cannot use scipy.linalg.eig for LinearOperator "
                            "M with k >= N - 1.")

        # 返回使用 scipy.linalg.eig 处理后的结果，包括特征值和可能的特征向量
        return eig(A, b=M, right=return_eigenvectors)

    # 如果未指定特征值问题的 sigma 值
    if sigma is None:
        # 获取 A 的线性操作对象
        matvec = _aslinearoperator_with_dtype(A).matvec

        # 如果 OPinv 不为 None，则引发 ValueError 异常
        if OPinv is not None:
            raise ValueError("OPinv should not be specified "
                             "with sigma = None.")
        # 如果 OPpart 不为 None，则引发 ValueError 异常
        if OPpart is not None:
            raise ValueError("OPpart should not be specified with "
                             "sigma = None or complex A")

        # 如果 M 也为 None，则说明是标准特征值问题
        if M is None:
            mode = 1  # 标记为标准特征值问题
            M_matvec = None  # M 的线性操作函数为空
            Minv_matvec = None  # Minv 的线性操作函数为空
            # 如果 Minv 不为 None，则引发 ValueError 异常
            if Minv is not None:
                raise ValueError("Minv should not be "
                                 "specified with M = None.")
        else:
            # 否则，说明是一般化特征值问题
            mode = 2  # 标记为一般化特征值问题
            # 如果 Minv 为 None，则通过函数获取其逆的线性操作函数
            if Minv is None:
                Minv_matvec = get_inv_matvec(M, hermitian=True, tol=tol)
            else:
                Minv = _aslinearoperator_with_dtype(Minv)
                Minv_matvec = Minv.matvec
            # 获取 M 的线性操作对象
            M_matvec = _aslinearoperator_with_dtype(M).matvec
    else:
        # 如果 sigma 不为 None，则进入 shift-invert 模式
        if np.issubdtype(A.dtype, np.complexfloating):
            # 如果 A 的数据类型是复数类型
            if OPpart is not None:
                # 如果 OPpart 不为 None，则抛出数值错误
                raise ValueError("OPpart should not be specified "
                                 "with sigma=None or complex A")
            # mode 设为 3
            mode = 3
        elif OPpart is None or OPpart.lower() == 'r':
            # 如果 OPpart 为 None 或者为 'r'（实数部分），mode 设为 3
            mode = 3
        elif OPpart.lower() == 'i':
            # 如果 OPpart 为 'i'（虚数部分）
            if np.imag(sigma) == 0:
                # 如果 sigma 的虚部为 0，则抛出数值错误
                raise ValueError("OPpart cannot be 'i' if sigma is real")
            # mode 设为 4
            mode = 4
        else:
            # 其他情况下，抛出数值错误，要求 OPpart 必须是 ('r', 'i') 中的一个
            raise ValueError("OPpart must be one of ('r','i')")

        # 获取 A 的线性操作对象的 matvec 方法
        matvec = _aslinearoperator_with_dtype(A).matvec

        # 如果 Minv 不为 None，则抛出数值错误，不应同时指定 Minv 和 sigma
        if Minv is not None:
            raise ValueError("Minv should not be specified when sigma is")

        # 如果 OPinv 为 None，则获取一个特定设置的 OPinv 的 matvec 方法
        if OPinv is None:
            Minv_matvec = get_OPinv_matvec(A, M, sigma,
                                           hermitian=False, tol=tol)
        else:
            # 否则，将 OPinv 转换为线性操作对象，并获取其 matvec 方法
            OPinv = _aslinearoperator_with_dtype(OPinv)
            Minv_matvec = OPinv.matvec

        # 如果 M 为 None，则 M_matvec 设为 None；否则获取 M 的 matvec 方法
        if M is None:
            M_matvec = None
        else:
            M_matvec = _aslinearoperator_with_dtype(M).matvec

    # 创建一个 _UnsymmetricArpackParams 对象，传入必要的参数
    params = _UnsymmetricArpackParams(n, k, A.dtype.char, matvec, mode,
                                      M_matvec, Minv_matvec, sigma,
                                      ncv, v0, maxiter, which, tol)

    # 使用 _ARPACK_LOCK 进行并发控制
    with _ARPACK_LOCK:
        # 在 params 没有收敛之前循环迭代
        while not params.converged:
            params.iterate()

        # 返回提取的结果，可能包括特征向量
        return params.extract(return_eigenvectors)
# 导入所需模块和库，包括NumPy和SciPy的线性代数模块
import numpy as np
from scipy.sparse.linalg import LinearOperator, eigs

# 定义函数 eigsh，用于计算实对称方阵或复共轭厄米方阵 A 的前 k 个特征值和特征向量
def eigsh(A, k=6, M=None, sigma=None, which='LM', v0=None,
          ncv=None, maxiter=None, tol=0, return_eigenvectors=True,
          Minv=None, OPinv=None, mode='normal'):
    """
    Find k eigenvalues and eigenvectors of the real symmetric square matrix
    or complex Hermitian matrix A.

    Solves ``A @ x[i] = w[i] * x[i]``, the standard eigenvalue problem for
    w[i] eigenvalues with corresponding eigenvectors x[i].

    If M is specified, solves ``A @ x[i] = w[i] * M @ x[i]``, the
    generalized eigenvalue problem for w[i] eigenvalues
    with corresponding eigenvectors x[i].

    Note that there is no specialized routine for the case when A is a complex
    Hermitian matrix. In this case, ``eigsh()`` will call ``eigs()`` and return the
    real parts of the eigenvalues thus obtained.

    Parameters
    ----------
    A : ndarray, sparse matrix or LinearOperator
        A square operator representing the operation ``A @ x``, where ``A`` is
        real symmetric or complex Hermitian. For buckling mode (see below)
        ``A`` must additionally be positive-definite.
    k : int, optional
        The number of eigenvalues and eigenvectors desired.
        `k` must be smaller than N. It is not possible to compute all
        eigenvectors of a matrix.

    Returns
    -------
    w : array
        Array of k eigenvalues.
    v : array
        An array representing the `k` eigenvectors.  The column ``v[:, i]`` is
        the eigenvector corresponding to the eigenvalue ``w[i]``.

    Other Parameters
    ----------------
    M : An N x N matrix, array, sparse matrix, or linear operator representing
        the operation ``M @ x`` for the generalized eigenvalue problem

            A @ x = w * M @ x.

        M must represent a real symmetric matrix if A is real, and must
        represent a complex Hermitian matrix if A is complex. For best
        results, the data type of M should be the same as that of A.
        Additionally:

            If sigma is None, M is symmetric positive definite.

            If sigma is specified, M is symmetric positive semi-definite.

            In buckling mode, M is symmetric indefinite.

        If sigma is None, eigsh requires an operator to compute the solution
        of the linear equation ``M @ x = b``. This is done internally via a
        (sparse) LU decomposition for an explicit matrix M, or via an
        iterative solver for a general linear operator.  Alternatively,
        the user can supply the matrix or operator Minv, which gives
        ``x = Minv @ b = M^-1 @ b``.
    """
    # 如果 M 参数为 None，则将 A 转换为 LinearOperator
    if M is None:
        M = LinearOperator(A.shape, matvec=lambda v: A.dot(v))

    # 如果需要返回特征向量，则设置 v0 为 None 或者使用用户指定的 v0
    if return_eigenvectors:
        if v0 is None:
            v0 = np.random.rand(min(A.shape[0], k))
        else:
            v0 = v0.astype(A.dtype)

    # 使用 SciPy 的 eigs 函数求解特征值问题
    w, v = eigs(A=A, k=k, M=M, sigma=sigma, which=which, v0=v0,
                ncv=ncv, maxiter=maxiter, tol=tol, return_eigenvectors=return_eigenvectors,
                Minv=Minv, OPinv=OPinv, mode=mode)

    # 返回计算得到的特征值 w 和特征向量 v
    return w, v.real
    sigma : real
        # 按照 shift-invert 模式查找接近 sigma 的特征值。这要求一个操作符来计算线性系统的解
        # ``[A - sigma * M] x = b``，其中如果未指定，M 是单位矩阵。对于显式矩阵 A 和 M，
        # 这内部通过稀疏 LU 分解来计算；如果 A 或 M 是一般线性操作符，则通过迭代求解器计算。
        # 用户也可以提供矩阵或操作符 OPinv，这样会得到 ``x = OPinv @ b = [A - sigma * M]^-1 @ b``。
        # 注意，当指定了 sigma 时，关键字 'which' 指的是偏移特征值 ``w'[i]``，其中：
        #
        # - 如果 mode == 'normal'，``w'[i] = 1 / (w[i] - sigma)``。
        # - 如果 mode == 'cayley'，``w'[i] = (w[i] + sigma) / (w[i] - sigma)``。
        # - 如果 mode == 'buckling'，``w'[i] = w[i] / (w[i] - sigma)``。
        #
        # （详见下面 'mode' 的进一步讨论）
    v0 : ndarray, optional
        # 迭代的起始向量。默认为随机生成。
    ncv : int, optional
        # 生成的 Lanczos 向量数目。ncv 必须大于 k 并小于 n；
        # 建议 ``ncv > 2*k``。
        # 默认值为 ``min(n, max(2*k + 1, 20))``
    which : str ['LM' | 'SM' | 'LA' | 'SA' | 'BE']
        # 如果 A 是复 Hermite 矩阵，则 'BE' 是无效的。
        # 指定要找到的前 k 个特征向量和特征值的类型：
        #
        # - 'LM' : 最大（按幅值）特征值。
        # - 'SM' : 最小（按幅值）特征值。
        # - 'LA' : 最大（代数）特征值。
        # - 'SA' : 最小（代数）特征值。
        # - 'BE' : 谱的两端各取 k/2 个特征值。
        #
        # 当 k 是奇数时，从高端返回一个更多（k/2+1）个特征值。
        # 当 sigma != None 时，'which' 指的是偏移特征值 ``w'[i]``
        # （详见上面 'sigma' 的讨论）。通常 ARPACK 更适合找到大值而不是小值。
        # 如果需要小特征值，考虑使用 shift-invert 模式以获得更好的性能。
    maxiter : int, optional
        # 允许的 Arnoldi 更新迭代的最大次数。默认为 ``n*10``。
    tol : float
        # 特征值的相对精度（停止准则）。默认值 0 表示机器精度。
    Minv : N x N 矩阵、数组、稀疏矩阵或线性操作符
        # 参见上述 M 的注释。
    OPinv : N x N 矩阵、数组、稀疏矩阵或线性操作符
        # 参见上述 sigma 的注释。
    return_eigenvectors : bool
        # 是否返回特征向量（True），以及特征值。此值确定特征值排序的顺序。
        # 排序顺序也取决于 `which` 变量的值。

            For which = 'LM' or 'SA':
                # 如果 `return_eigenvectors` 为 True，则按照代数值对特征值进行排序。
                # 如果 `return_eigenvectors` 为 False，则按照绝对值对特征值进行排序。

            For which = 'BE' or 'LA':
                # 特征值总是按照代数值进行排序。

            For which = 'SM':
                # 如果 `return_eigenvectors` 为 True，则按照代数值对特征值进行排序。
                # 如果 `return_eigenvectors` 为 False，则按照绝对值递减的顺序对特征值进行排序。

    mode : string ['normal' | 'buckling' | 'cayley']
        # 指定用于shift-invert模式的策略。此参数仅适用于实值A和sigma != None的情况。
        # 对于shift-invert模式，ARPACK内部解决特征值问题
        # ``OP @ x'[i] = w'[i] * B @ x'[i]``
        # 并将得到的里兹向量x'[i]和里兹值w'[i]转换为问题
        # ``A @ x[i] = w[i] * M @ x[i]`` 的期望特征向量和特征值。
        # 模式如下：

            'normal' :
                # OP = [A - sigma * M]^-1 @ M,
                # B = M,
                # w'[i] = 1 / (w[i] - sigma)

            'buckling' :
                # OP = [A - sigma * M]^-1 @ A,
                # B = A,
                # w'[i] = w[i] / (w[i] - sigma)

            'cayley' :
                # OP = [A - sigma * M]^-1 @ [A + sigma * M],
                # B = M,
                # w'[i] = (w[i] + sigma) / (w[i] - sigma)

        # 模式的选择会影响由关键字 'which' 选择的特征值，并可能影响收敛稳定性（参见[2]进行讨论）。

    Raises
    ------
    ArpackNoConvergence
        # 当未达到请求的收敛性时引发此异常。

        # 可以在异常对象的 `eigenvalues` 和 `eigenvectors` 属性中找到当前收敛的特征值和特征向量。

    See Also
    --------
    eigs : 用于一般（非对称）矩阵A的特征值和特征向量
    svds : 矩阵A的奇异值分解

    Notes
    -----
    # 此函数是对ARPACK [1]_ SSEUPD和DSEUPD函数的包装，
    # 这些函数使用隐式重启Lanczos方法来找到特征值和特征向量[2]_。

    References
    ----------
    .. [1] ARPACK Software, https://github.com/opencollab/arpack-ng
    .. [2] R. B. Lehoucq, D. C. Sorensen, and C. Yang,  ARPACK USERS GUIDE:
       Solution of Large Scale Eigenvalue Problems by Implicitly Restarted
       Arnoldi Methods. SIAM, Philadelphia, PA, 1998.

    Examples
    --------
    >>> import numpy as np
    # 导入从 scipy.sparse.linalg 模块中的 eigsh 函数
    from scipy.sparse.linalg import eigsh
    # 创建一个 13x13 的单位矩阵
    identity = np.eye(13)
    # 使用 eigsh 函数计算单位矩阵的前 6 个特征值和特征向量
    eigenvalues, eigenvectors = eigsh(identity, k=6)
    # 打印特征值数组
    eigenvalues
    # 打印特征向量数组的形状
    eigenvectors.shape

    """
    # 复埃尔米特矩阵应使用 eigs 函数求解
    if np.issubdtype(A.dtype, np.complexfloating):
        # 如果 A 的数据类型是复数类型
        if mode != 'normal':
            # 如果模式不是 'normal'，抛出异常
            raise ValueError("mode=%s cannot be used with "
                             "complex matrix A" % mode)
        if which == 'BE':
            # 如果 which 参数为 'BE'，抛出异常
            raise ValueError("which='BE' cannot be used with complex matrix A")
        elif which == 'LA':
            # 如果 which 参数为 'LA'，将其转换为 'LR'
            which = 'LR'
        elif which == 'SA':
            # 如果 which 参数为 'SA'，将其转换为 'SR'
            which = 'SR'
        # 调用 eigs 函数求解特征值问题
        ret = eigs(A, k, M=M, sigma=sigma, which=which, v0=v0,
                   ncv=ncv, maxiter=maxiter, tol=tol,
                   return_eigenvectors=return_eigenvectors, Minv=Minv,
                   OPinv=OPinv)

        if return_eigenvectors:
            # 如果需要返回特征向量，则返回实部的特征值和特征向量
            return ret[0].real, ret[1]
        else:
            # 否则，只返回实部的特征值
            return ret.real

    # 如果 A 不是方阵，则抛出异常
    if A.shape[0] != A.shape[1]:
        raise ValueError(f'expected square matrix (shape={A.shape})')
    # 如果 M 不为 None，则检查其维度和数据类型是否与 A 相匹配
    if M is not None:
        if M.shape != A.shape:
            raise ValueError(f'wrong M dimensions {M.shape}, should be {A.shape}')
        if np.dtype(M.dtype).char.lower() != np.dtype(A.dtype).char.lower():
            # 如果 M 的数据类型精度与 A 不匹配，则发出警告
            warnings.warn('M does not have the same type precision as A. '
                          'This may adversely affect ARPACK convergence',
                          stacklevel=2)

    # 获取矩阵 A 的维度
    n = A.shape[0]

    # 如果 k 小于等于 0，则抛出异常
    if k <= 0:
        raise ValueError("k must be greater than 0.")

    # 如果 k 大于等于矩阵的维度 n，则发出警告
    if k >= n:
        warnings.warn("k >= N for N * N square matrix. "
                      "Attempting to use scipy.linalg.eigh instead.",
                      RuntimeWarning, stacklevel=2)

        # 如果 A 是稀疏矩阵，则抛出类型错误
        if issparse(A):
            raise TypeError("Cannot use scipy.linalg.eigh for sparse A with "
                            "k >= N. Use scipy.linalg.eigh(A.toarray()) or"
                            " reduce k.")
        # 如果 A 是线性操作符，则抛出类型错误
        if isinstance(A, LinearOperator):
            raise TypeError("Cannot use scipy.linalg.eigh for LinearOperator "
                            "A with k >= N.")
        # 如果 M 是线性操作符，则抛出类型错误
        if isinstance(M, LinearOperator):
            raise TypeError("Cannot use scipy.linalg.eigh for LinearOperator "
                            "M with k >= N.")

        # 返回使用 scipy.linalg.eigh 求解的结果，仅返回特征值或者不返回特征向量
        return eigh(A, b=M, eigvals_only=not return_eigenvectors)
    # 如果 sigma 参数为 None，则进行以下操作
    if sigma is None:
        # 将 A 转换为具有指定数据类型的线性操作对象
        A = _aslinearoperator_with_dtype(A)
        # 获取 A 对象的 matvec 方法，用于矩阵向量乘法
        matvec = A.matvec

        # 如果 OPinv 参数不为空，则引发数值错误
        if OPinv is not None:
            raise ValueError("OPinv should not be specified "
                             "with sigma = None.")
        # 如果 M 参数为空，则为标准特征值问题
        if M is None:
            mode = 1  # 设置模式为1，表示标准特征值问题
            M_matvec = None
            Minv_matvec = None
            # 如果 Minv 参数不为空，则引发数值错误
            if Minv is not None:
                raise ValueError("Minv should not be "
                                 "specified with M = None.")
        else:
            # 如果 M 参数不为空，则为一般特征值问题
            mode = 2  # 设置模式为2，表示一般特征值问题
            # 如果 Minv 参数为空，则计算其逆的 matvec 方法
            if Minv is None:
                Minv_matvec = get_inv_matvec(M, hermitian=True, tol=tol)
            else:
                # 将 Minv 转换为具有指定数据类型的线性操作对象，并获取其 matvec 方法
                Minv = _aslinearoperator_with_dtype(Minv)
                Minv_matvec = Minv.matvec
            # 获取 M 的 matvec 方法
            M_matvec = _aslinearoperator_with_dtype(M).matvec
    else:
        # 如果 sigma 参数不为 None，则进入移位反转模式
        if Minv is not None:
            # 如果 Minv 参数不为空，则引发数值错误
            raise ValueError("Minv should not be specified when sigma is")

        # 普通模式
        if mode == 'normal':
            mode = 3  # 设置模式为3，表示普通模式
            matvec = None
            # 如果 OPinv 参数为空，则计算 OPinv 的 matvec 方法
            if OPinv is None:
                Minv_matvec = get_OPinv_matvec(A, M, sigma,
                                               hermitian=True, tol=tol)
            else:
                # 将 OPinv 转换为具有指定数据类型的线性操作对象，并获取其 matvec 方法
                OPinv = _aslinearoperator_with_dtype(OPinv)
                Minv_matvec = OPinv.matvec
            # 如果 M 参数为空，则设置 M_matvec 为 None
            if M is None:
                M_matvec = None
            else:
                # 将 M 转换为具有指定数据类型的线性操作对象，并获取其 matvec 方法
                M = _aslinearoperator_with_dtype(M)
                M_matvec = M.matvec

        # 屈曲模式
        elif mode == 'buckling':
            mode = 4  # 设置模式为4，表示屈曲模式
            # 如果 OPinv 参数为空，则计算 OPinv 的 matvec 方法
            if OPinv is None:
                Minv_matvec = get_OPinv_matvec(A, M, sigma,
                                               hermitian=True, tol=tol)
            else:
                # 将 OPinv 转换为具有指定数据类型的线性操作对象，并获取其 matvec 方法
                Minv_matvec = _aslinearoperator_with_dtype(OPinv).matvec
            # 获取 A 的 matvec 方法
            matvec = _aslinearoperator_with_dtype(A).matvec
            # 设置 M_matvec 为 None
            M_matvec = None

        # 凯莱变换模式
        elif mode == 'cayley':
            mode = 5  # 设置模式为5，表示凯莱变换模式
            # 获取 A 的 matvec 方法
            matvec = _aslinearoperator_with_dtype(A).matvec
            # 如果 OPinv 参数为空，则计算 OPinv 的 matvec 方法
            if OPinv is None:
                Minv_matvec = get_OPinv_matvec(A, M, sigma,
                                               hermitian=True, tol=tol)
            else:
                # 将 OPinv 转换为具有指定数据类型的线性操作对象，并获取其 matvec 方法
                Minv_matvec = _aslinearoperator_with_dtype(OPinv).matvec
            # 如果 M 参数为空，则设置 M_matvec 为 None
            if M is None:
                M_matvec = None
            else:
                # 将 M 转换为具有指定数据类型的线性操作对象，并获取其 matvec 方法
                M_matvec = _aslinearoperator_with_dtype(M).matvec

        # 未识别的模式
        else:
            # 引发数值错误，指示未识别的模式
            raise ValueError("unrecognized mode '%s'" % mode)

    # 创建 _SymmetricArpackParams 对象，用于保存所有参数信息
    params = _SymmetricArpackParams(n, k, A.dtype.char, matvec, mode,
                                    M_matvec, Minv_matvec, sigma,
                                    ncv, v0, maxiter, which, tol)
    # 使用 `_ARPACK_LOCK` 上的锁来确保线程安全，防止多线程同时访问 ARPACK 算法库
    with _ARPACK_LOCK:
        # 在参数对象 `params` 还未收敛之前循环执行迭代过程
        while not params.converged:
            # 调用参数对象的 `iterate()` 方法执行一次迭代
            params.iterate()
    
        # 根据参数 `return_eigenvectors` 的值，从参数对象 `params` 中提取特征值或特征向量并返回
        return params.extract(return_eigenvectors)
```