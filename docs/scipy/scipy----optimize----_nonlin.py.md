# `D:\src\scipysrc\scipy\scipy\optimize\_nonlin.py`

```
# 导入模块和包，包括标准库和SciPy库中的函数和异常
import inspect  # 导入 inspect 模块，用于获取对象的信息
import sys  # 导入 sys 模块，提供对 Python 解释器的访问
import warnings  # 导入 warnings 模块，用于警告处理

import numpy as np  # 导入 NumPy 库，并将其命名为 np
from numpy import asarray, dot, vdot  # 从 NumPy 中导入特定函数

from scipy.linalg import norm, solve, inv, qr, svd, LinAlgError  # 从 SciPy 的 linalg 子模块导入特定函数和异常
import scipy.sparse.linalg  # 导入 SciPy 的 sparse.linalg 子模块
import scipy.sparse  # 导入 SciPy 的 sparse 子模块
from scipy.linalg import get_blas_funcs  # 从 SciPy 的 linalg 子模块导入特定函数
from scipy._lib._util import copy_if_needed  # 导入 SciPy 的 _lib._util 模块中的函数
from scipy._lib._util import getfullargspec_no_self as _getfullargspec  # 导入 SciPy 的 _lib._util 模块中的函数，并重命名

from ._linesearch import scalar_search_wolfe1, scalar_search_armijo  # 从当前包的 _linesearch 模块导入特定函数

__all__ = [  # 定义 __all__ 变量，指定模块公开的接口
    'broyden1', 'broyden2', 'anderson', 'linearmixing',
    'diagbroyden', 'excitingmixing', 'newton_krylov',
    'BroydenFirst', 'KrylovJacobian', 'InverseJacobian', 'NoConvergence'
]

#------------------------------------------------------------------------------
# Utility functions
#------------------------------------------------------------------------------

class NoConvergence(Exception):
    """Exception raised when nonlinear solver fails to converge within the specified
    `maxiter`."""
    pass  # 定义一个自定义异常类 NoConvergence，用于非线性求解器无法在指定的最大迭代次数内收敛时抛出异常

def maxnorm(x):
    return np.absolute(x).max()  # 计算向量 x 的最大范数

def _as_inexact(x):
    """Return `x` as an array, of either floats or complex floats"""
    x = asarray(x)  # 将输入 x 转换为 NumPy 数组
    if not np.issubdtype(x.dtype, np.inexact):
        return asarray(x, dtype=np.float64)  # 如果 x 的数据类型不是浮点数或复数，转换为浮点数数组
    return x  # 否则直接返回 x

def _array_like(x, x0):
    """Return ndarray `x` as same array subclass and shape as `x0`"""
    x = np.reshape(x, np.shape(x0))  # 将 x 重塑为与 x0 相同形状的 ndarray
    wrap = getattr(x0, '__array_wrap__', x.__array_wrap__)  # 获取 x0 或 x 的数组包装器函数
    return wrap(x)  # 使用包装器函数包装并返回 x

def _safe_norm(v):
    if not np.isfinite(v).all():  # 如果向量 v 中存在非有限值
        return np.array(np.inf)  # 返回包含无穷大的数组
    return norm(v)  # 否则计算向量 v 的范数并返回
    tol_norm : function(vector) -> scalar, optional
        # 容忍范数函数，用于收敛性检查。默认为最大范数。
        Norm to use in convergence check. Default is the maximum norm.
    line_search : {None, 'armijo' (default), 'wolfe'}, optional
        # 线搜索类型，用于确定在由雅可比逼近给出的方向上的步长。默认为 'armijo'。
        Which type of a line search to use to determine the step size in the
        direction given by the Jacobian approximation. Defaults to 'armijo'.
    callback : function, optional
        # 可选的回调函数。每次迭代时调用，形式为 ``callback(x, f)``，其中 `x` 是当前解，`f` 是相应的残差。
        Optional callback function. It is called on every iteration as
        ``callback(x, f)`` where `x` is the current solution and `f`
        the corresponding residual.

    Returns
    -------
    sol : ndarray
        # 包含最终解的数组（与 `x0` 的相似数组类型）。
        An array (of similar array type as `x0`) containing the final solution.

    Raises
    ------
    NoConvergence
        # 当未找到解决方案时引发异常。
        When a solution was not found.
def _set_doc(obj):
    # 如果对象有文档字符串，则使用给定的文档部分来格式化它
    if obj.__doc__:
        obj.__doc__ = obj.__doc__ % _doc_parts


def nonlin_solve(F, x0, jacobian='krylov', iter=None, verbose=False,
                 maxiter=None, f_tol=None, f_rtol=None, x_tol=None, x_rtol=None,
                 tol_norm=None, line_search='armijo', callback=None,
                 full_output=False, raise_exception=True):
    """
    Find a root of a function, in a way suitable for large-scale problems.

    Parameters
    ----------
    %(params_basic)s
    jacobian : Jacobian
        A Jacobian approximation: `Jacobian` object or something that
        `asjacobian` can transform to one. Alternatively, a string specifying
        which of the builtin Jacobian approximations to use:

            krylov, broyden1, broyden2, anderson
            diagbroyden, linearmixing, excitingmixing

    %(params_extra)s
    full_output : bool
        If true, returns a dictionary `info` containing convergence
        information.
    raise_exception : bool
        If True, a `NoConvergence` exception is raise if no solution is found.

    See Also
    --------
    asjacobian, Jacobian

    Notes
    -----
    This algorithm implements the inexact Newton method, with
    backtracking or full line searches. Several Jacobian
    approximations are available, including Krylov and Quasi-Newton
    methods.

    References
    ----------
    .. [KIM] C. T. Kelley, \"Iterative Methods for Linear and Nonlinear
       Equations\". Society for Industrial and Applied Mathematics. (1995)
       https://archive.siam.org/books/kelley/fr16/

    """
    # 不能使用默认参数，因为它从调用函数中明确传递为 None，所以在这里设置它
    tol_norm = maxnorm if tol_norm is None else tol_norm
    # 设置终止条件对象
    condition = TerminationCondition(f_tol=f_tol, f_rtol=f_rtol,
                                     x_tol=x_tol, x_rtol=x_rtol,
                                     iter=iter, norm=tol_norm)

    # 将 x0 转换为浮点数类型
    x0 = _as_inexact(x0)

    # 定义函数 func，返回展平后的 F(z) 的结果
    def func(z):
        return _as_inexact(F(_array_like(z, x0))).flatten()

    # 将 x 展平
    x = x0.flatten()

    # 初始化 dx 为具有无限大值的与 x 相同形状的数组
    dx = np.full_like(x, np.inf)

    # 计算 F(x) 的值
    Fx = func(x)

    # 计算 F(x) 的范数
    Fx_norm = norm(Fx)

    # 转换 jacobian 参数为 Jacobian 对象
    jacobian = asjacobian(jacobian)
    # 设置 jacobian 对象的初始状态
    jacobian.setup(x.copy(), Fx, func)

    # 如果 maxiter 为 None，则根据 x 的大小设置其值
    if maxiter is None:
        if iter is not None:
            maxiter = iter + 1
        else:
            maxiter = 100*(x.size+1)

    # 将 line_search 参数转换为 'armijo'，如果 line_search 为 True
    if line_search is True:
        line_search = 'armijo'
    # 如果 line_search 为 False，则设置为 None
    elif line_search is False:
        line_search = None

    # 如果 line_search 不在 (None, 'armijo', 'wolfe') 中，则引发 ValueError 异常
    if line_search not in (None, 'armijo', 'wolfe'):
        raise ValueError("Invalid line search")

    # 设置解算器的容差选择参数
    gamma = 0.9
    eta_max = 0.9999
    eta_treshold = 0.1
    eta = 1e-3
    # 迭代最大次数的范围
    for n in range(maxiter):
        # 检查当前状态是否满足停止条件
        status = condition.check(Fx, x, dx)
        if status:
            break

        # 计算容差，用于 scipy.sparse.linalg.* 算法
        tol = min(eta, eta*Fx_norm)
        # 解线性方程组 jacobian.solve，并设定容差
        dx = -jacobian.solve(Fx, tol=tol)

        # 如果计算出的方向向量为零向量，则抛出异常
        if norm(dx) == 0:
            raise ValueError("Jacobian inversion yielded zero vector. "
                             "This indicates a bug in the Jacobian "
                             "approximation.")

        # 进行线搜索或者牛顿步
        if line_search:
            # 进行非线性线搜索
            s, x, Fx, Fx_norm_new = _nonlin_line_search(func, x, Fx, dx,
                                                        line_search)
        else:
            # 直接进行牛顿步
            s = 1.0
            x = x + dx
            Fx = func(x)
            Fx_norm_new = norm(Fx)

        # 更新雅可比矩阵
        jacobian.update(x.copy(), Fx)

        # 如果有回调函数，则调用回调函数
        if callback:
            callback(x, Fx)

        # 根据不精确方法调整强制参数
        eta_A = gamma * Fx_norm_new**2 / Fx_norm**2
        if gamma * eta**2 < eta_treshold:
            eta = min(eta_max, eta_A)
        else:
            eta = min(eta_max, max(eta_A, gamma*eta**2))

        # 更新 Fx_norm
        Fx_norm = Fx_norm_new

        # 打印当前状态信息
        if verbose:
            sys.stdout.write("%d:  |F(x)| = %g; step %g\n" % (
                n, tol_norm(Fx), s))
            sys.stdout.flush()
    else:
        # 如果达到最大迭代次数且需要抛出异常，则抛出异常
        if raise_exception:
            raise NoConvergence(_array_like(x, x0))
        else:
            status = 2

    # 如果需要返回完整输出
    if full_output:
        # 构建返回信息字典
        info = {'nit': condition.iteration,
                'fun': Fx,
                'status': status,
                'success': status == 1,
                'message': {1: 'A solution was found at the specified '
                               'tolerance.',
                            2: 'The maximum number of iterations allowed '
                               'has been reached.'
                            }[status]
                }
        # 返回结果和信息字典
        return _array_like(x, x0), info
    else:
        # 只返回结果
        return _array_like(x, x0)
# 设置文档字符串为非线性求解器的文档
_set_doc(nonlin_solve)

# 定义非线性搜索函数，用于优化问题中的非线性优化
def _nonlin_line_search(func, x, Fx, dx, search_type='armijo', rdiff=1e-8,
                        smin=1e-2):
    # 临时存储搜索过程中的步长、函数值和目标函数的平方范数
    tmp_s = [0]
    tmp_Fx = [Fx]
    tmp_phi = [norm(Fx)**2]
    # 计算搜索方向向量的范数与当前解向量的范数比值
    s_norm = norm(x) / norm(dx)

    # 定义目标函数的平方范数
    def phi(s, store=True):
        # 如果步长与之前保存的步长相同，则直接返回之前计算的目标函数的平方范数
        if s == tmp_s[0]:
            return tmp_phi[0]
        # 计算新步长对应点的函数值，并返回其平方范数
        xt = x + s*dx
        v = func(xt)
        p = _safe_norm(v)**2
        if store:
            tmp_s[0] = s
            tmp_phi[0] = p
            tmp_Fx[0] = v
        return p

    # 定义目标函数在步长 s 处的导数
    def derphi(s):
        # 计算微小增量 ds，并返回目标函数的导数估计
        ds = (abs(s) + s_norm + 1) * rdiff
        return (phi(s+ds, store=False) - phi(s)) / ds

    # 根据指定的搜索类型选择不同的步长搜索方法
    if search_type == 'wolfe':
        # 使用 Wolfe 条件进行步长搜索
        s, phi1, phi0 = scalar_search_wolfe1(phi, derphi, tmp_phi[0],
                                             xtol=1e-2, amin=smin)
    elif search_type == 'armijo':
        # 使用 Armijo 条件进行步长搜索
        s, phi1 = scalar_search_armijo(phi, tmp_phi[0], -tmp_phi[0],
                                       amin=smin)

    # 如果未找到合适的步长，则记录警告信息，并使用完整的牛顿步长
    if s is None:
        # XXX: 未找到合适的步长。使用完整的牛顿步长，并希望取得最佳结果。
        s = 1.0

    # 更新当前解向量 x，并计算相应的函数值 Fx 和其范数
    x = x + s*dx
    if s == tmp_s[0]:
        Fx = tmp_Fx[0]
    else:
        Fx = func(x)
    Fx_norm = norm(Fx)

    # 返回更新后的步长 s、解向量 x、函数值 Fx 和其范数 Fx_norm
    return s, x, Fx, Fx_norm


class TerminationCondition:
    """
    迭代终止条件。满足以下条件时终止迭代：

    - |F| < f_rtol*|F_0|，且
    - |F| < f_tol

    以及

    - |dx| < x_rtol*|x|，且
    - |dx| < x_tol

    """
    def __init__(self, f_tol=None, f_rtol=None, x_tol=None, x_rtol=None,
                 iter=None, norm=maxnorm):

        # 如果未指定 f_tol，则设定为浮点数精度的三分之一次方
        if f_tol is None:
            f_tol = np.finfo(np.float64).eps ** (1./3)
        # 如果未指定 f_rtol，则设定为无穷大
        if f_rtol is None:
            f_rtol = np.inf
        # 如果未指定 x_tol，则设定为无穷大
        if x_tol is None:
            x_tol = np.inf
        # 如果未指定 x_rtol，则设定为无穷大
        if x_rtol is None:
            x_rtol = np.inf

        # 初始化迭代终止条件的各个属性
        self.x_tol = x_tol
        self.x_rtol = x_rtol
        self.f_tol = f_tol
        self.f_rtol = f_rtol

        self.norm = norm

        self.iter = iter

        self.f0_norm = None
        self.iteration = 0

    # 检查迭代是否应终止的方法
    def check(self, f, x, dx):
        # 增加迭代次数计数
        self.iteration += 1
        # 计算当前函数值、解向量和搜索方向的范数
        f_norm = self.norm(f)
        x_norm = self.norm(x)
        dx_norm = self.norm(dx)

        # 如果初次迭代，则记录初始函数值的范数
        if self.f0_norm is None:
            self.f0_norm = f_norm

        # 如果当前函数值的范数为零，则返回 1 表示终止迭代
        if f_norm == 0:
            return 1

        # 如果设置了最大迭代次数 iter，则检查是否达到
        if self.iter is not None:
            # 与 SciPy 0.6.0 版本的向后兼容性
            return 2 * (self.iteration > self.iter)

        # 检查迭代是否应终止的条件
        return int((f_norm <= self.f_tol
                    and f_norm/self.f_rtol <= self.f0_norm)
                   and (dx_norm <= self.x_tol
                        and dx_norm/self.x_rtol <= x_norm))


#------------------------------------------------------------------------------
# 通用雅可比矩阵近似
#------------------------------------------------------------------------------

class Jacobian:
    """
    雅可比矩阵或其近似的通用接口。
    """
    """
    The optional methods come useful when implementing trust region
    etc., algorithms that often require evaluating transposes of the
    Jacobian.

    Methods
    -------
    solve
        Returns J^-1 * v
    update
        Updates Jacobian to point `x` (where the function has residual `Fx`)

    matvec : optional
        Returns J * v
    rmatvec : optional
        Returns A^H * v
    rsolve : optional
        Returns A^-H * v
    matmat : optional
        Returns A * V, where V is a dense matrix with dimensions (N,K).
    todense : optional
        Form the dense Jacobian matrix. Necessary for dense trust region
        algorithms, and useful for testing.

    Attributes
    ----------
    shape
        Matrix dimensions (M, N)
    dtype
        Data type of the matrix.
    func : callable, optional
        Function the Jacobian corresponds to

    """

    # 初始化方法，接受关键字参数
    def __init__(self, **kw):
        # 允许的属性名列表
        names = ["solve", "update", "matvec", "rmatvec", "rsolve",
                 "matmat", "todense", "shape", "dtype"]
        # 遍历传入的关键字参数
        for name, value in kw.items():
            # 如果属性名不在允许的列表中，则抛出异常
            if name not in names:
                raise ValueError("Unknown keyword argument %s" % name)
            # 如果值不为None，则设置实例的属性
            if value is not None:
                setattr(self, name, kw[name])

        # 如果实例具有'todense'属性，定义一个特殊方法__array__
        if hasattr(self, "todense"):
            def __array__(self, dtype=None, copy=None):
                # 如果dtype不为None，则抛出异常
                if dtype is not None:
                    raise ValueError(f"`dtype` must be None, was {dtype}")
                # 调用实例的todense方法并返回结果
                return self.todense()

    # 返回一个InverseJacobian对象
    def aspreconditioner(self):
        return InverseJacobian(self)

    # 解决方法，抛出未实现错误
    def solve(self, v, tol=0):
        raise NotImplementedError

    # 更新方法，更新Jacobian至点x处（函数具有残差Fx）
    def update(self, x, F):
        pass

    # 设置方法，设置函数、形状和数据类型属性
    def setup(self, x, F, func):
        self.func = func
        self.shape = (F.size, x.size)
        self.dtype = F.dtype
        # 如果setup方法是Jacobian类自带的方法，则在第一个点上调用update
        if self.__class__.setup is Jacobian.setup:
            self.update(x, F)
class InverseJacobian:
    # 初始化函数，接收一个 Jacobian 对象作为参数
    def __init__(self, jacobian):
        # 将传入的 jacobian 对象赋值给实例变量 self.jacobian
        self.jacobian = jacobian
        # 将 jacobian 对象的 solve 方法赋值给实例变量 self.matvec
        self.matvec = jacobian.solve
        # 将 jacobian 对象的 update 方法赋值给实例变量 self.update
        self.update = jacobian.update
        # 如果 jacobian 对象有 'setup' 属性，将其赋值给实例变量 self.setup
        if hasattr(jacobian, 'setup'):
            self.setup = jacobian.setup
        # 如果 jacobian 对象有 'rsolve' 属性，将其赋值给实例变量 self.rmatvec
        if hasattr(jacobian, 'rsolve'):
            self.rmatvec = jacobian.rsolve

    # 返回实例变量 self.jacobian 的 shape 属性
    @property
    def shape(self):
        return self.jacobian.shape

    # 返回实例变量 self.jacobian 的 dtype 属性
    @property
    def dtype(self):
        return self.jacobian.dtype


# 将给定对象转换为适合作为 Jacobian 使用的对象
def asjacobian(J):
    spsolve = scipy.sparse.linalg.spsolve
    # 如果 J 是 Jacobian 类的实例，则直接返回 J
    if isinstance(J, Jacobian):
        return J
    # 如果 J 是 Jacobian 类的子类，则返回一个新的 J 实例
    elif inspect.isclass(J) and issubclass(J, Jacobian):
        return J()
    # 如果 J 是 numpy 数组
    elif isinstance(J, np.ndarray):
        # 如果数组维度大于 2，则抛出异常
        if J.ndim > 2:
            raise ValueError('array must have rank <= 2')
        # 将 J 至少转换为二维数组
        J = np.atleast_2d(np.asarray(J))
        # 如果数组不是方阵，则抛出异常
        if J.shape[0] != J.shape[1]:
            raise ValueError('array must be square')

        # 返回一个新的 Jacobian 对象，使用传入的矩阵 J
        return Jacobian(matvec=lambda v: dot(J, v),
                        rmatvec=lambda v: dot(J.conj().T, v),
                        solve=lambda v, tol=0: solve(J, v),
                        rsolve=lambda v, tol=0: solve(J.conj().T, v),
                        dtype=J.dtype, shape=J.shape)
    # 如果 J 是稀疏矩阵
    elif scipy.sparse.issparse(J):
        # 如果稀疏矩阵不是方阵，则抛出异常
        if J.shape[0] != J.shape[1]:
            raise ValueError('matrix must be square')
        # 返回一个新的 Jacobian 对象，使用传入的稀疏矩阵 J
        return Jacobian(matvec=lambda v: J @ v,
                        rmatvec=lambda v: J.conj().T @ v,
                        solve=lambda v, tol=0: spsolve(J, v),
                        rsolve=lambda v, tol=0: spsolve(J.conj().T, v),
                        dtype=J.dtype, shape=J.shape)
    # 如果 J 具有 'shape'、'dtype' 和 'solve' 属性
    elif hasattr(J, 'shape') and hasattr(J, 'dtype') and hasattr(J, 'solve'):
        # 返回一个新的 Jacobian 对象，使用 J 对象的相关属性和方法
        return Jacobian(matvec=getattr(J, 'matvec'),
                        rmatvec=getattr(J, 'rmatvec'),
                        solve=J.solve,
                        rsolve=getattr(J, 'rsolve'),
                        update=getattr(J, 'update'),
                        setup=getattr(J, 'setup'),
                        dtype=J.dtype,
                        shape=J.shape)
    elif callable(J):
        # 如果 J 是可调用的，则假定它是一个函数 J(x)，返回雅可比矩阵
        # 定义一个名为 Jac 的子类，继承自 Jacobian
        class Jac(Jacobian):
            # 更新方法，将当前 x 存储到实例中
            def update(self, x, F):
                self.x = x

            # 解决线性系统方法，使用 J(x) 的结果进行求解
            def solve(self, v, tol=0):
                m = J(self.x)
                if isinstance(m, np.ndarray):
                    return solve(m, v)
                elif scipy.sparse.issparse(m):
                    return spsolve(m, v)
                else:
                    raise ValueError("Unknown matrix type")

            # 矩阵向量乘法方法，根据 J(x) 的类型进行乘法运算
            def matvec(self, v):
                m = J(self.x)
                if isinstance(m, np.ndarray):
                    return dot(m, v)
                elif scipy.sparse.issparse(m):
                    return m @ v
                else:
                    raise ValueError("Unknown matrix type")

            # 右侧解线性系统方法，使用 J(x) 的共轭转置进行求解
            def rsolve(self, v, tol=0):
                m = J(self.x)
                if isinstance(m, np.ndarray):
                    return solve(m.conj().T, v)
                elif scipy.sparse.issparse(m):
                    return spsolve(m.conj().T, v)
                else:
                    raise ValueError("Unknown matrix type")

            # 右侧矩阵向量乘法方法，使用 J(x) 的共轭转置进行乘法运算
            def rmatvec(self, v):
                m = J(self.x)
                if isinstance(m, np.ndarray):
                    return dot(m.conj().T, v)
                elif scipy.sparse.issparse(m):
                    return m.conj().T @ v
                else:
                    raise ValueError("Unknown matrix type")
        
        # 返回 Jac 类的实例
        return Jac()
    
    elif isinstance(J, str):
        # 如果 J 是字符串类型，则根据字典映射返回相应的 Jacobian 类
        return dict(broyden1=BroydenFirst,
                    broyden2=BroydenSecond,
                    anderson=Anderson,
                    diagbroyden=DiagBroyden,
                    linearmixing=LinearMixing,
                    excitingmixing=ExcitingMixing,
                    krylov=KrylovJacobian)[J]()
    
    else:
        # 若不满足以上条件，则抛出类型错误异常
        raise TypeError('Cannot convert object to a Jacobian')
#------------------------------------------------------------------------------
# Broyden
#------------------------------------------------------------------------------

class GenericBroyden(Jacobian):
    # GenericBroyden 类继承自 Jacobian 类
    def setup(self, x0, f0, func):
        # 调用父类的 setup 方法，设置初始值
        Jacobian.setup(self, x0, f0, func)
        # 初始化上一次迭代的函数值和变量值
        self.last_f = f0
        self.last_x = x0

        # 如果已定义 self.alpha 且为 None，则自动调整初始雅可比矩阵参数
        # 除非已经猜测到解决方案
        if hasattr(self, 'alpha') and self.alpha is None:
            # 计算初始雅可比矩阵参数的自动缩放
            normf0 = norm(f0)
            if normf0:
                self.alpha = 0.5 * max(norm(x0), 1) / normf0
            else:
                self.alpha = 1.0

    def _update(self, x, f, dx, df, dx_norm, df_norm):
        # 抽象方法，用于更新算法状态
        raise NotImplementedError

    def update(self, x, f):
        # 更新函数，计算变量和函数的增量，调用 _update 方法更新状态
        df = f - self.last_f
        dx = x - self.last_x
        self._update(x, f, dx, df, norm(dx), norm(df))
        # 更新上一次迭代的函数值和变量值
        self.last_f = f
        self.last_x = x


class LowRankMatrix:
    r"""
    A matrix represented as

    .. math:: \alpha I + \sum_{n=0}^{n=M} c_n d_n^\dagger

    However, if the rank of the matrix reaches the dimension of the vectors,
    full matrix representation will be used thereon.

    """
    
    def __init__(self, alpha, n, dtype):
        # 初始化 LowRankMatrix 类的实例
        self.alpha = alpha  # 初始化参数 alpha
        self.cs = []        # 初始化系数列表 cs
        self.ds = []        # 初始化向量列表 ds
        self.n = n          # 初始化矩阵维度 n
        self.dtype = dtype  # 初始化数据类型 dtype
        self.collapsed = None  # 初始化 collapsed 属性为 None

    @staticmethod
    def _matvec(v, alpha, cs, ds):
        # 静态方法：计算向量 v 经过矩阵 M 后的结果 M * v
        axpy, scal, dotc = get_blas_funcs(['axpy', 'scal', 'dotc'],
                                          cs[:1] + [v])
        w = alpha * v  # 计算 alpha * v
        for c, d in zip(cs, ds):
            a = dotc(d, v)  # 计算向量 d 和 v 的内积
            w = axpy(c, w, w.size, a)  # 计算 axpy(c, w, w.size, a)
        return w

    @staticmethod
    def _solve(v, alpha, cs, ds):
        """Evaluate w = M^-1 v"""
        if len(cs) == 0:
            return v / alpha  # 如果 cs 列表为空，则返回 v / alpha

        # 计算求解 M^-1 * v 的过程

        axpy, dotc = get_blas_funcs(['axpy', 'dotc'], cs[:1] + [v])

        c0 = cs[0]
        A = alpha * np.identity(len(cs), dtype=c0.dtype)  # 创建单位矩阵 alpha * I
        for i, d in enumerate(ds):
            for j, c in enumerate(cs):
                A[i,j] += dotc(d, c)  # 更新 A 矩阵的元素

        q = np.zeros(len(cs), dtype=c0.dtype)
        for j, d in enumerate(ds):
            q[j] = dotc(d, v)  # 计算向量 d 和 v 的内积
        q /= alpha
        q = solve(A, q)  # 解线性方程 A * q = q

        w = v / alpha
        for c, qc in zip(cs, q):
            w = axpy(c, w, w.size, -qc)  # 计算 axpy(c, w, w.size, -qc)

        return w

    def matvec(self, v):
        """Evaluate w = M v"""
        if self.collapsed is not None:
            return np.dot(self.collapsed, v)  # 如果 collapsed 不为 None，返回 collapsed 矩阵与 v 的乘积
        return LowRankMatrix._matvec(v, self.alpha, self.cs, self.ds)  # 调用 _matvec 方法计算 M * v

    def rmatvec(self, v):
        """Evaluate w = M^H v"""
        if self.collapsed is not None:
            return np.dot(self.collapsed.T.conj(), v)  # 如果 collapsed 不为 None，返回 collapsed 转置共轭矩阵与 v 的乘积
        return LowRankMatrix._matvec(v, np.conj(self.alpha), self.ds, self.cs)  # 调用 _matvec 方法计算 M^H * v
    # 解决线性系统 Mx = v，其中 M 是一个低秩矩阵，v 是给定的向量
    def solve(self, v, tol=0):
        """Evaluate w = M^-1 v"""
        # 如果矩阵已经被折叠（collapsed），则使用折叠后的矩阵进行求解
        if self.collapsed is not None:
            return solve(self.collapsed, v)
        # 否则调用 LowRankMatrix 类的 _solve 方法进行求解
        return LowRankMatrix._solve(v, self.alpha, self.cs, self.ds)

    # 解决共轭转置线性系统 M^H x = v，其中 M 是一个低秩矩阵，v 是给定的向量
    def rsolve(self, v, tol=0):
        """Evaluate w = M^-H v"""
        # 如果矩阵已经被折叠（collapsed），则使用折叠后的矩阵进行求解
        if self.collapsed is not None:
            return solve(self.collapsed.T.conj(), v)
        # 否则调用 LowRankMatrix 类的 _solve 方法进行求解，注意其中的复共轭操作
        return LowRankMatrix._solve(v, np.conj(self.alpha), self.ds, self.cs)

    # 向低秩矩阵添加新的列向量 c 和行向量 d
    def append(self, c, d):
        # 如果矩阵已经被折叠（collapsed），则直接对折叠后的矩阵进行更新
        if self.collapsed is not None:
            self.collapsed += c[:,None] * d[None,:].conj()
            return

        # 否则将新的列向量 c 和行向量 d 添加到 cs 和 ds 列表中
        self.cs.append(c)
        self.ds.append(d)

        # 如果 cs 列表的长度超过了列向量 c 的大小，则执行矩阵的折叠操作
        if len(self.cs) > c.size:
            self.collapse()

    # 将低秩矩阵表示为数组，支持指定数据类型和复制选项
    def __array__(self, dtype=None, copy=None):
        # 如果指定了 dtype，则发出警告，因为 dtype 应当为 None
        if dtype is not None:
            warnings.warn("LowRankMatrix is scipy-internal code, `dtype` "
                          f"should only be None but was {dtype} (not handled)",
                          stacklevel=3)
        # 如果指定了 copy，则发出警告，因为 copy 应当为 None
        if copy is not None:
            warnings.warn("LowRankMatrix is scipy-internal code, `copy` "
                          f"should only be None but was {copy} (not handled)",
                          stacklevel=3)
        
        # 如果矩阵已经被折叠（collapsed），则返回折叠后的矩阵
        if self.collapsed is not None:
            return self.collapsed

        # 否则构建一个包含所有列向量和行向量贡献的矩阵 Gm
        Gm = self.alpha*np.identity(self.n, dtype=self.dtype)
        for c, d in zip(self.cs, self.ds):
            Gm += c[:,None]*d[None,:].conj()
        return Gm

    # 折叠低秩矩阵为完全秩的矩阵
    def collapse(self):
        """Collapse the low-rank matrix to a full-rank one."""
        # 将当前的低秩矩阵折叠成一个完全秩的矩阵，并存储在 collapsed 属性中
        self.collapsed = np.array(self, copy=copy_if_needed)
        self.cs = None  # 清空 cs 列表
        self.ds = None  # 清空 ds 列表
        self.alpha = None  # 将 alpha 属性置为 None

    # 通过删除所有向量来减少矩阵的秩
    def restart_reduce(self, rank):
        """
        Reduce the rank of the matrix by dropping all vectors.
        """
        # 如果矩阵已经被折叠（collapsed），则无需操作
        if self.collapsed is not None:
            return
        assert rank > 0
        # 如果 cs 列表的长度超过了指定的秩 rank，则删除所有列向量和行向量
        if len(self.cs) > rank:
            del self.cs[:]
            del self.ds[:]

    # 通过删除最旧的向量来减少矩阵的秩
    def simple_reduce(self, rank):
        """
        Reduce the rank of the matrix by dropping oldest vectors.
        """
        # 如果矩阵已经被折叠（collapsed），则无需操作
        if self.collapsed is not None:
            return
        assert rank > 0
        # 当 cs 列表的长度超过指定的秩 rank 时，循环删除最旧的列向量和行向量，直至满足秩的要求
        while len(self.cs) > rank:
            del self.cs[0]
            del self.ds[0]
    # 如果已经进行了矩阵收缩，则无需再次执行
    if self.collapsed is not None:
        return

    # 设置矩阵的最大降维后的秩
    p = max_rank

    # 如果指定了要保留的奇异值分解（SVD）分量数，则使用指定值，否则默认为 p - 2
    if to_retain is not None:
        q = to_retain
    else:
        q = p - 2

    # 如果存在列向量集合（self.cs），将 p 限制为它们的最小长度
    if self.cs:
        p = min(p, len(self.cs[0]))

    # 确保 q 在 [0, p-1] 范围内
    q = max(0, min(q, p-1))

    # 矩阵的行数
    m = len(self.cs)

    # 如果行数小于 p，无需操作，直接返回
    if m < p:
        # nothing to do
        return

    # 将列向量集合转换为 NumPy 数组并进行转置
    C = np.array(self.cs).T
    D = np.array(self.ds).T

    # 对 D 应用 QR 分解，经济模式
    D, R = qr(D, mode='economic')

    # 对 C 应用 R 的共轭转置
    C = dot(C, R.T.conj())

    # 对 C 进行奇异值分解，得到 U、S、WH
    U, S, WH = svd(C, full_matrices=False)

    # 修正 C 和 D
    C = dot(C, inv(WH))
    D = dot(D, WH.T.conj())

    # 仅保留前 q 个列向量
    for k in range(q):
        self.cs[k] = C[:,k].copy()
        self.ds[k] = D[:,k].copy()

    # 删除多余的列向量
    del self.cs[q:]
    del self.ds[q:]
# 将'broyden_params'键添加到_doc_parts字典中，其值是包含Broyden方法参数说明的多行字符串
_doc_parts['broyden_params'] = """
    alpha : float, optional
        Initial guess for the Jacobian is ``(-1/alpha)``.
    reduction_method : str or tuple, optional
        Method used in ensuring that the rank of the Broyden matrix
        stays low. Can either be a string giving the name of the method,
        or a tuple of the form ``(method, param1, param2, ...)``
        that gives the name of the method and values for additional parameters.

        Methods available:

            - ``restart``: drop all matrix columns. Has no extra parameters.
            - ``simple``: drop oldest matrix column. Has no extra parameters.
            - ``svd``: keep only the most significant SVD components.
              Takes an extra parameter, ``to_retain``, which determines the
              number of SVD components to retain when rank reduction is done.
              Default is ``max_rank - 2``.

    max_rank : int, optional
        Maximum rank for the Broyden matrix.
        Default is infinity (i.e., no rank reduction).
    """.strip()

# 定义BroydenFirst类，它继承自GenericBroyden类，用于使用Broyden的第一雅可比近似方法寻找函数的根
class BroydenFirst(GenericBroyden):
    # 文档字符串，描述使用Broyden的第一雅可比近似方法寻找函数根的算法
    r"""
    Find a root of a function, using Broyden's first Jacobian approximation.

    This method is also known as "Broyden's good method".

    Parameters
    ----------
    %(params_basic)s
    %(broyden_params)s  # 引用_doc_parts字典中'broyden_params'的值，将Broyden方法的参数说明插入此处
    %(params_extra)s

    See Also
    --------
    root : Interface to root finding algorithms for multivariate
           functions. See ``method='broyden1'`` in particular.

    Notes
    -----
    This algorithm implements the inverse Jacobian Quasi-Newton update

    .. math:: H_+ = H + (dx - H df) dx^\dagger H / ( dx^\dagger H df)

    which corresponds to Broyden's first Jacobian update

    .. math:: J_+ = J + (df - J dx) dx^\dagger / dx^\dagger dx


    References
    ----------
    .. [1] B.A. van der Rotten, PhD thesis,
       "A limited memory Broyden method to solve high-dimensional
       systems of nonlinear equations". Mathematisch Instituut,
       Universiteit Leiden, The Netherlands (2003).

       https://web.archive.org/web/20161022015821/http://www.math.leidenuniv.nl/scripties/Rotten.pdf

    Examples
    --------
    The following functions define a system of nonlinear equations

    >>> def fun(x):
    ...     return [x[0]  + 0.5 * (x[0] - x[1])**3 - 1.0,
    ...             0.5 * (x[1] - x[0])**3 + x[1]]

    A solution can be obtained as follows.

    >>> from scipy import optimize
    >>> sol = optimize.broyden1(fun, [0, 0])
    >>> sol
    array([0.84116396, 0.15883641])

    """
    def __init__(self, alpha=None, reduction_method='restart', max_rank=None):
        # 调用父类 GenericBroyden 的初始化方法
        GenericBroyden.__init__(self)
        # 设置 alpha 参数
        self.alpha = alpha
        # 初始化 Gm 为 None
        self.Gm = None

        # 如果 max_rank 未指定，则设为无穷大
        if max_rank is None:
            max_rank = np.inf
        # 设置最大秩参数
        self.max_rank = max_rank

        # 根据 reduction_method 类型确定 reduce_params
        if isinstance(reduction_method, str):
            reduce_params = ()
        else:
            reduce_params = reduction_method[1:]
            reduction_method = reduction_method[0]
        # 调整 reduce_params 大小
        reduce_params = (max_rank - 1,) + reduce_params

        # 根据 reduction_method 类型设置 _reduce 方法
        if reduction_method == 'svd':
            self._reduce = lambda: self.Gm.svd_reduce(*reduce_params)
        elif reduction_method == 'simple':
            self._reduce = lambda: self.Gm.simple_reduce(*reduce_params)
        elif reduction_method == 'restart':
            self._reduce = lambda: self.Gm.restart_reduce(*reduce_params)
        else:
            # 抛出异常，如果 reduction_method 不支持
            raise ValueError("Unknown rank reduction method '%s'" %
                             reduction_method)

    def setup(self, x, F, func):
        # 调用父类 GenericBroyden 的 setup 方法
        GenericBroyden.setup(self, x, F, func)
        # 初始化 Gm 为 LowRankMatrix 对象
        self.Gm = LowRankMatrix(-self.alpha, self.shape[0], self.dtype)

    def todense(self):
        # 返回 Gm 的逆矩阵
        return inv(self.Gm)

    def solve(self, f, tol=0):
        # 计算 Gm 对 f 的乘积
        r = self.Gm.matvec(f)
        # 如果结果不是有限的，则表示奇异，重新设置雅可比矩阵近似
        if not np.isfinite(r).all():
            self.setup(self.last_x, self.last_f, self.func)
            return self.Gm.matvec(f)
        # 返回计算结果
        return r

    def matvec(self, f):
        # 调用 Gm 的 solve 方法
        return self.Gm.solve(f)

    def rsolve(self, f, tol=0):
        # 调用 Gm 的 rmatvec 方法
        return self.Gm.rmatvec(f)

    def rmatvec(self, f):
        # 调用 Gm 的 rsolve 方法
        return self.Gm.rsolve(f)

    def _update(self, x, f, dx, df, dx_norm, df_norm):
        # 执行降秩操作以保持割线条件
        self._reduce()

        # 计算 Gm 的右乘向量
        v = self.Gm.rmatvec(dx)
        # 计算 dx - Gm 乘以 df 的结果
        c = dx - self.Gm.matvec(df)
        # 计算新的向量 d
        d = v / vdot(df, v)

        # 向 Gm 添加新的向量对
        self.Gm.append(c, d)
class BroydenSecond(BroydenFirst):
    """
    Find a root of a function, using Broyden\'s second Jacobian approximation.

    This method is also known as \"Broyden's bad method\".

    Parameters
    ----------
    %(params_basic)s
        基本参数，通常包括函数、初始点等
    %(broyden_params)s
        Broyden 方法的特定参数，如迭代步长等
    %(params_extra)s
        额外参数，例如数值稳定性参数等

    See Also
    --------
    root : Interface to root finding algorithms for multivariate
           functions. See ``method='broyden2'`` in particular.
        多元函数根查找算法的接口，特别是使用 ``method='broyden2'``

    Notes
    -----
    This algorithm implements the inverse Jacobian Quasi-Newton update

    .. math:: H_+ = H + (dx - H df) df^\\dagger / ( df^\\dagger df)

    corresponding to Broyden's second method.
        实现反 Jacobian 的拟牛顿更新，对应于 Broyden 的第二方法

    References
    ----------
    .. [1] B.A. van der Rotten, PhD thesis,
       \"A limited memory Broyden method to solve high-dimensional
       systems of nonlinear equations\". Mathematisch Instituut,
       Universiteit Leiden, The Netherlands (2003).
       高维非线性方程组解的有限内存 Broyden 方法的博士论文参考

       https://web.archive.org/web/20161022015821/http://www.math.leidenuniv.nl/scripties/Rotten.pdf

    Examples
    --------
    The following functions define a system of nonlinear equations

    >>> def fun(x):
    ...     return [x[0]  + 0.5 * (x[0] - x[1])**3 - 1.0,
    ...             0.5 * (x[1] - x[0])**3 + x[1]]
        定义了一个非线性方程组的函数示例

    A solution can be obtained as follows.

    >>> from scipy import optimize
    >>> sol = optimize.broyden2(fun, [0, 0])
    >>> sol
    array([0.84116365, 0.15883529])
        可以如下获得解

    """

    def _update(self, x, f, dx, df, dx_norm, df_norm):
        self._reduce()  # reduce first to preserve secant condition
        减少历史记录以保持割线条件

        v = df
        c = dx - self.Gm.matvec(df)
        d = v / df_norm**2
        self.Gm.append(c, d)
    """

    # Note:
    #
    # Anderson method maintains a rank M approximation of the inverse Jacobian,
    # 
    #     J^-1 v ~ -v*alpha + (dX + alpha dF) A^-1 dF^H v
    #     A      = W + dF^H dF
    #     W      = w0^2 diag(dF^H dF)
    #
    # so that for w0 = 0 the secant condition applies for last M iterates, i.e.,
    #
    #     J^-1 df_j = dx_j
    #
    # for all j = 0 ... M-1.
    #
    # Moreover, (from Sherman-Morrison-Woodbury formula)
    #
    #    J v ~ [ b I - b^2 C (I + b dF^H A^-1 C)^-1 dF^H ] v
    #    C   = (dX + alpha dF) A^-1
    #    b   = -1/alpha
    #
    # and after simplification
    #
    #    J v ~ -v/alpha + (dX/alpha + dF) (dF^H dX - alpha W)^-1 dF^H v
    #

    # 初始化函数，设置 Anderson 方法的参数
    def __init__(self, alpha=None, w0=0.01, M=5):
        # 调用父类构造函数
        GenericBroyden.__init__(self)
        # 设置 alpha 参数
        self.alpha = alpha
        # 设置 M 参数
        self.M = M
        # 初始化 dx 列表，用于存储 dx 向量
        self.dx = []
        # 初始化 df 列表，用于存储 df 向量
        self.df = []
        # 初始化 gamma 变量
        self.gamma = None
        # 设置 w0 参数
        self.w0 = w0

    # 解算函数，根据 Anderson 方法求解方程
    def solve(self, f, tol=0):
        # 计算 dx 向量
        dx = -self.alpha*f

        # 获取已存储的 dx 向量个数
        n = len(self.dx)
        # 若 dx 向量个数为 0，则返回当前计算的 dx 向量
        if n == 0:
            return dx

        # 初始化 df_f 数组，用于存储 df 和 f 的内积
        df_f = np.empty(n, dtype=f.dtype)
        # 计算 df 和 f 的内积
        for k in range(n):
            df_f[k] = vdot(self.df[k], f)

        try:
            # 使用已存储的矩阵 a 求解 gamma
            gamma = solve(self.a, df_f)
        except LinAlgError:
            # 如果出现线性代数错误，重置 Jacobian 近似
            del self.dx[:]
            del self.df[:]
            return dx

        # 根据 Anderson 方法更新 dx 向量
        for m in range(n):
            dx += gamma[m]*(self.dx[m] + self.alpha*self.df[m])
        return dx

    # 矩阵向量乘法函数，根据 Anderson 方法计算 matvec
    def matvec(self, f):
        # 计算 dx 向量
        dx = -f/self.alpha

        # 获取已存储的 dx 向量个数
        n = len(self.dx)
        # 若 dx 向量个数为 0，则返回当前计算的 dx 向量
        if n == 0:
            return dx

        # 初始化 df_f 数组，用于存储 df 和 f 的内积
        df_f = np.empty(n, dtype=f.dtype)
        # 计算 df 和 f 的内积
        for k in range(n):
            df_f[k] = vdot(self.df[k], f)

        # 初始化 b 矩阵
        b = np.empty((n, n), dtype=f.dtype)
        # 计算 b 矩阵的元素
        for i in range(n):
            for j in range(n):
                b[i,j] = vdot(self.df[i], self.dx[j])
                # 如果 i == j 且 w0 不为 0，则进行额外计算
                if i == j and self.w0 != 0:
                    b[i,j] -= vdot(self.df[i], self.df[i])*self.w0**2*self.alpha
        # 使用已存储的矩阵 b 求解 gamma
        gamma = solve(b, df_f)

        # 根据 Anderson 方法更新 dx 向量
        for m in range(n):
            dx += gamma[m]*(self.df[m] + self.dx[m]/self.alpha)
        return dx

    # 更新函数，根据 Anderson 方法更新 Jacobian 近似矩阵 a
    def _update(self, x, f, dx, df, dx_norm, df_norm):
        # 如果 M 为 0，则直接返回
        if self.M == 0:
            return

        # 将 dx 和 df 添加到存储列表中
        self.dx.append(dx)
        self.df.append(df)

        # 若存储列表长度超过 M，则删除最旧的元素
        while len(self.dx) > self.M:
            self.dx.pop(0)
            self.df.pop(0)

        # 获取已存储的 dx 向量个数
        n = len(self.dx)
        # 初始化 a 矩阵
        a = np.zeros((n, n), dtype=f.dtype)

        # 计算 a 矩阵的元素
        for i in range(n):
            for j in range(i, n):
                # 如果 i == j，则计算额外的项 wd
                if i == j:
                    wd = self.w0**2
                else:
                    wd = 0
                a[i,j] = (1+wd)*vdot(self.df[i], self.df[j])

        # 将 a 矩阵转换为上三角形式
        a += np.triu(a, 1).T.conj()
        self.a = a
#------------------------------------------------------------------------------
# Simple iterations
#------------------------------------------------------------------------------


class DiagBroyden(GenericBroyden):
    """
    Find a root of a function, using diagonal Broyden Jacobian approximation.

    The Jacobian approximation is derived from previous iterations, by
    retaining only the diagonal of Broyden matrices.

    .. warning::

       This algorithm may be useful for specific problems, but whether
       it will work may depend strongly on the problem.

    Parameters
    ----------
    %(params_basic)s
    alpha : float, optional
        Initial guess for the Jacobian is (-1/alpha).
    %(params_extra)s

    See Also
    --------
    root : Interface to root finding algorithms for multivariate
           functions. See ``method='diagbroyden'`` in particular.

    Examples
    --------
    The following functions define a system of nonlinear equations

    >>> def fun(x):
    ...     return [x[0]  + 0.5 * (x[0] - x[1])**3 - 1.0,
    ...             0.5 * (x[1] - x[0])**3 + x[1]]

    A solution can be obtained as follows.

    >>> from scipy import optimize
    >>> sol = optimize.diagbroyden(fun, [0, 0])
    >>> sol
    array([0.84116403, 0.15883384])

    """

    def __init__(self, alpha=None):
        # 调用父类的初始化方法
        GenericBroyden.__init__(self)
        # 设置 alpha 属性
        self.alpha = alpha

    def setup(self, x, F, func):
        # 调用父类的 setup 方法，初始化 x、F 和 func
        GenericBroyden.setup(self, x, F, func)
        # 初始化对角线向量 d，使用 self.alpha 计算初始值
        self.d = np.full((self.shape[0],), 1 / self.alpha, dtype=self.dtype)

    def solve(self, f, tol=0):
        # 返回解向量，使用对角线向量 d 计算
        return -f / self.d

    def matvec(self, f):
        # 返回矩阵向量乘积，使用对角线向量 d 计算
        return -f * self.d

    def rsolve(self, f, tol=0):
        # 返回右乘逆矩阵后的解向量，使用对角线向量 d 计算
        return -f / self.d.conj()

    def rmatvec(self, f):
        # 返回右乘逆矩阵后的矩阵向量乘积，使用对角线向量 d 计算
        return -f * self.d.conj()

    def todense(self):
        # 返回对角矩阵表示的密集矩阵，使用对角线向量 d 计算
        return np.diag(-self.d)

    def _update(self, x, f, dx, df, dx_norm, df_norm):
        # 更新对角线向量 d，使用 Broyden 方法更新
        self.d -= (df + self.d*dx)*dx/dx_norm**2


class LinearMixing(GenericBroyden):
    """
    Find a root of a function, using a scalar Jacobian approximation.

    .. warning::

       This algorithm may be useful for specific problems, but whether
       it will work may depend strongly on the problem.

    Parameters
    ----------
    %(params_basic)s
    alpha : float, optional
        The Jacobian approximation is (-1/alpha).
    %(params_extra)s

    See Also
    --------
    root : Interface to root finding algorithms for multivariate
           functions. See ``method='linearmixing'`` in particular.

    """

    def __init__(self, alpha=None):
        # 调用父类的初始化方法
        GenericBroyden.__init__(self)
        # 设置 alpha 属性
        self.alpha = alpha

    def solve(self, f, tol=0):
        # 返回解向量，使用标量 alpha 计算
        return -f*self.alpha

    def matvec(self, f):
        # 返回矩阵向量乘积，使用标量 alpha 计算
        return -f/self.alpha

    def rsolve(self, f, tol=0):
        # 返回右乘逆矩阵后的解向量，使用标量 alpha 计算
        return -f*np.conj(self.alpha)

    def rmatvec(self, f):
        # 返回右乘逆矩阵后的矩阵向量乘积，使用标量 alpha 计算
        return -f/np.conj(self.alpha)

    def todense(self):
        # 返回对角矩阵表示的密集矩阵，使用标量 alpha 计算
        return np.diag(np.full(self.shape[0], -1/self.alpha))
    # 定义一个私有方法 `_update`，用于执行某种更新操作，接受多个参数
    def _update(self, x, f, dx, df, dx_norm, df_norm):
        # 这里是方法的实现部分，当前未进行具体实现，使用 `pass` 占位
        pass
class ExcitingMixing(GenericBroyden):
    """
    Find a root of a function, using a tuned diagonal Jacobian approximation.

    The Jacobian matrix is diagonal and is tuned on each iteration.

    .. warning::

       This algorithm may be useful for specific problems, but whether
       it will work may depend strongly on the problem.

    See Also
    --------
    root : Interface to root finding algorithms for multivariate
           functions. See ``method='excitingmixing'`` in particular.

    Parameters
    ----------
    %(params_basic)s
    alpha : float, optional
        Initial Jacobian approximation is (-1/alpha).
    alphamax : float, optional
        The entries of the diagonal Jacobian are kept in the range
        ``[alpha, alphamax]``.
    %(params_extra)s
    """

    def __init__(self, alpha=None, alphamax=1.0):
        # 调用父类构造函数，初始化基类 GenericBroyden
        GenericBroyden.__init__(self)
        # 设置初始的 alpha 和 alphamax 值
        self.alpha = alpha
        self.alphamax = alphamax
        self.beta = None  # 初始化 beta 为 None

    def setup(self, x, F, func):
        # 调用父类的 setup 方法，设置初始状态
        GenericBroyden.setup(self, x, F, func)
        # 使用 alpha 初始化一个全为 alpha 的数组作为 beta
        self.beta = np.full((self.shape[0],), self.alpha, dtype=self.dtype)

    def solve(self, f, tol=0):
        # 返回计算的解决方案，使用当前 beta 对 f 进行加权处理
        return -f*self.beta

    def matvec(self, f):
        # 返回使用当前 beta 对 f 进行加权处理的向量
        return -f/self.beta

    def rsolve(self, f, tol=0):
        # 返回右侧的解决方案，使用当前 beta 的共轭进行加权处理
        return -f*self.beta.conj()

    def rmatvec(self, f):
        # 返回使用当前 beta 的共轭进行加权处理的向量
        return -f/self.beta.conj()

    def todense(self):
        # 返回一个对角线矩阵，矩阵元素为 -1/self.beta
        return np.diag(-1/self.beta)

    def _update(self, x, f, dx, df, dx_norm, df_norm):
        # 根据 f 和上一次 f 的符号变化来更新 beta
        incr = f*self.last_f > 0
        self.beta[incr] += self.alpha
        self.beta[~incr] = self.alpha
        np.clip(self.beta, 0, self.alphamax, out=self.beta)

#------------------------------------------------------------------------------
# Iterative/Krylov approximated Jacobians
#------------------------------------------------------------------------------

class KrylovJacobian(Jacobian):
    r"""
    Find a root of a function, using Krylov approximation for inverse Jacobian.

    This method is suitable for solving large-scale problems.

    Parameters
    ----------
    %(params_basic)s
    rdiff : float, optional
        Relative step size to use in numerical differentiation.
    method : str or callable, optional
        Krylov method to use to approximate the Jacobian.  Can be a string,
        or a function implementing the same interface as the iterative
        solvers in `scipy.sparse.linalg`. If a string, needs to be one of:
        ``'lgmres'``, ``'gmres'``, ``'bicgstab'``, ``'cgs'``, ``'minres'``,
        ``'tfqmr'``.

        The default is `scipy.sparse.linalg.lgmres`.
    inner_maxiter : int, optional
        Parameter to pass to the "inner" Krylov solver: maximum number of
        iterations. Iteration will stop after maxiter steps even if the
        specified tolerance has not been achieved.
    """
    # 内部 Krylov 迭代的预条件器，可以是 LinearOperator 或者 InverseJacobian
    inner_M : LinearOperator or InverseJacobian
        Preconditioner for the inner Krylov iteration.
        
        # 可以使用反向雅各比作为（自适应的）预条件器，例如：
        Note that you can use also inverse Jacobians as (adaptive)
        preconditioners. For example,
        
        >>> from scipy.optimize import BroydenFirst, KrylovJacobian
        >>> from scipy.optimize import InverseJacobian
        >>> jac = BroydenFirst()
        >>> kjac = KrylovJacobian(inner_M=InverseJacobian(jac))
        
        # 如果预条件器有名为 'update' 的方法，将在每个非线性步骤后调用，
        # 参数为 x 表示当前点，f 表示当前函数值。
        If the preconditioner has a method named 'update', it will be called
        as ``update(x, f)`` after each nonlinear step, with ``x`` giving
        the current point, and ``f`` the current function value.
        
    # LGMRES 非线性迭代过程中保留的子空间大小
    outer_k : int, optional
        Size of the subspace kept across LGMRES nonlinear iterations.
        See `scipy.sparse.linalg.lgmres` for details.
        
    # 内部 Krylov 求解器的关键字参数
    inner_kwargs : kwargs
        Keyword parameters for the "inner" Krylov solver
        (defined with `method`). Parameter names must start with
        the `inner_` prefix which will be stripped before passing on
        the inner method. See, e.g., `scipy.sparse.linalg.gmres` for details.
        
    # %(params_extra)s 的额外参数
    %(params_extra)s

    # 参见
    # root: 多元函数的根查找算法接口，特别是 ``method='krylov'``。
    # scipy.sparse.linalg.gmres
    # scipy.sparse.linalg.lgmres
    See Also
    --------
    root : Interface to root finding algorithms for multivariate
           functions. See ``method='krylov'`` in particular.
    scipy.sparse.linalg.gmres
    scipy.sparse.linalg.lgmres

    # 注释
    Notes
    -----
    This function implements a Newton-Krylov solver. The basic idea is
    to compute the inverse of the Jacobian with an iterative Krylov
    method. These methods require only evaluating the Jacobian-vector
    products, which are conveniently approximated by a finite difference:

    .. math:: J v \approx (f(x + \omega*v/|v|) - f(x)) / \omega

    Due to the use of iterative matrix inverses, these methods can
    deal with large nonlinear problems.

    SciPy's `scipy.sparse.linalg` module offers a selection of Krylov
    solvers to choose from. The default here is `lgmres`, which is a
    variant of restarted GMRES iteration that reuses some of the
    information obtained in the previous Newton steps to invert
    Jacobians in subsequent steps.

    For a review on Newton-Krylov methods, see for example [1]_,
    and for the LGMRES sparse inverse method, see [2]_.

    # 参考文献
    References
    ----------
    .. [1] C. T. Kelley, Solving Nonlinear Equations with Newton's Method,
           SIAM, pp.57-83, 2003.
           :doi:`10.1137/1.9780898718898.ch3`
    .. [2] D.A. Knoll and D.E. Keyes, J. Comp. Phys. 193, 357 (2004).
           :doi:`10.1016/j.jcp.2003.08.010`
    .. [3] A.H. Baker and E.R. Jessup and T. Manteuffel,
           SIAM J. Matrix Anal. Appl. 26, 962 (2005).
           :doi:`10.1137/S0895479803422014`

    # 示例
    Examples
    --------
    The following functions define a system of nonlinear equations

    >>> def fun(x):
    ...     return [x[0] + 0.5 * x[1] - 1.0,
    ...             0.5 * (x[1] - x[0]) ** 2]

    A solution can be obtained as follows.

    >>> from scipy import optimize
    # 调用 optimize 模块中的 newton_krylov 函数，求解给定函数 fun 的 Newton-Krylov 方法的解
    >>> sol = optimize.newton_krylov(fun, [0, 0])
    # 打印求解结果 sol
    >>> sol
    # 打印结果为一个包含两个元素的数组
    array([0.66731771, 0.66536458])

    """

    # 初始化函数，设定参数和选项
    def __init__(self, rdiff=None, method='lgmres', inner_maxiter=20,
                 inner_M=None, outer_k=10, **kw):
        # 设置内部参数 preconditioner 为 inner_M
        self.preconditioner = inner_M
        # 设置参数 rdiff 为 self.rdiff
        self.rdiff = rdiff
        # 根据给定的 method 字符串选择相应的稀疏线性代数求解方法，或者直接使用用户提供的可调用方法
        self.method = dict(
            bicgstab=scipy.sparse.linalg.bicgstab,
            gmres=scipy.sparse.linalg.gmres,
            lgmres=scipy.sparse.linalg.lgmres,
            cgs=scipy.sparse.linalg.cgs,
            minres=scipy.sparse.linalg.minres,
            tfqmr=scipy.sparse.linalg.tfqmr,
            ).get(method, method)

        # 设置方法的关键字参数
        self.method_kw = dict(maxiter=inner_maxiter, M=self.preconditioner)

        # 如果选择的方法是 GMRES，则修改相关参数以将其用于牛顿法的外部迭代
        if self.method is scipy.sparse.linalg.gmres:
            self.method_kw['restart'] = inner_maxiter
            self.method_kw['maxiter'] = 1
            self.method_kw.setdefault('atol', 0)
        # 如果选择的方法是 GCROTMK、BICGSTAB 或 CGS，设置相应的公共参数
        elif self.method in (scipy.sparse.linalg.gcrotmk,
                             scipy.sparse.linalg.bicgstab,
                             scipy.sparse.linalg.cgs):
            self.method_kw.setdefault('atol', 0)
        # 如果选择的方法是 LGMRES，则设定特定于 LGMRES 的参数和选项
        elif self.method is scipy.sparse.linalg.lgmres:
            self.method_kw['outer_k'] = outer_k
            self.method_kw['maxiter'] = 1
            self.method_kw.setdefault('outer_v', [])
            self.method_kw.setdefault('prepend_outer_v', True)
            self.method_kw.setdefault('store_outer_Av', False)
            self.method_kw.setdefault('atol', 0)

        # 处理其他用户提供的关键字参数
        for key, value in kw.items():
            # 如果关键字参数不是以 'inner_' 开头，抛出错误
            if not key.startswith('inner_'):
                raise ValueError("Unknown parameter %s" % key)
            self.method_kw[key[6:]] = value

    # 更新差分步长的内部方法
    def _update_diff_step(self):
        # 计算当前状态向量 x0 和函数值 f0 的最大绝对值
        mx = abs(self.x0).max()
        mf = abs(self.f0).max()
        # 根据相对差分限制系数 rdiff，更新差分步长 omega
        self.omega = self.rdiff * max(1, mx) / max(1, mf)

    # 稀疏矩阵向量乘法的内部方法
    def matvec(self, v):
        # 计算向量 v 的范数 nv
        nv = norm(v)
        # 如果 v 的范数为零，返回全零向量
        if nv == 0:
            return 0*v
        # 计算缩放因子 sc 和残差 r
        sc = self.omega / nv
        r = (self.func(self.x0 + sc*v) - self.f0) / sc
        # 如果 r 中存在非有限值而 v 中的所有值都是有限的，则抛出值错误
        if not np.all(np.isfinite(r)) and np.all(np.isfinite(v)):
            raise ValueError('Function returned non-finite results')
        # 返回计算得到的残差向量 r
        return r
    # 解决方程或优化问题的方法，根据设定选择不同的参数
    def solve(self, rhs, tol=0):
        # 如果方法关键字参数中包含 'rtol'，则使用设定的方法和关键字参数进行求解
        if 'rtol' in self.method_kw:
            sol, info = self.method(self.op, rhs, **self.method_kw)
        else:
            # 否则，使用指定的方法、右手边(rhs)和公差(tol)进行求解
            sol, info = self.method(self.op, rhs, rtol=tol, **self.method_kw)
        # 返回解
        return sol

    # 更新对象的初始点和函数值，然后更新差分步长
    def update(self, x, f):
        self.x0 = x
        self.f0 = f
        self._update_diff_step()

        # 如果存在预处理器，并且预处理器具有 'update' 方法，更新预处理器
        if self.preconditioner is not None:
            if hasattr(self.preconditioner, 'update'):
                self.preconditioner.update(x, f)

    # 设置对象的初始点、函数值和功能函数，然后配置雅可比矩阵，并将对象转换为线性操作器
    def setup(self, x, f, func):
        Jacobian.setup(self, x, f, func)  # 调用父类 Jacobian 的 setup 方法进行初始化
        self.x0 = x
        self.f0 = f
        self.op = scipy.sparse.linalg.aslinearoperator(self)  # 将对象转换为线性操作器

        # 如果没有设置相对差分，则设定默认的相对差分步长
        if self.rdiff is None:
            self.rdiff = np.finfo(x.dtype).eps ** (1./2)

        self._update_diff_step()  # 更新差分步长

        # 如果存在预处理器，并且预处理器具有 'setup' 方法，初始化预处理器
        if self.preconditioner is not None:
            if hasattr(self.preconditioner, 'setup'):
                self.preconditioner.setup(x, f, func)
#------------------------------------------------------------------------------
# Wrapper functions
#------------------------------------------------------------------------------

# 构建一个求解器包装函数，使用给定的名称和近似雅可比矩阵
def _nonlin_wrapper(name, jac):
    """
    Construct a solver wrapper with given name and Jacobian approx.

    It inspects the keyword arguments of ``jac.__init__``, and allows to
    use the same arguments in the wrapper function, in addition to the
    keyword arguments of `nonlin_solve`

    """
    # 获取传入 jac 对象的 __init__ 方法的参数签名
    signature = _getfullargspec(jac.__init__)
    args, varargs, varkw, defaults, kwonlyargs, kwdefaults, _ = signature
    # 将默认参数与其名称组成键值对列表
    kwargs = list(zip(args[-len(defaults):], defaults))
    # 构建关键字参数的字符串表示形式
    kw_str = ", ".join([f"{k}={v!r}" for k, v in kwargs])
    if kw_str:
        kw_str = ", " + kw_str
    # 构建传递给 jac 对象的关键字参数的名称映射字符串
    kwkw_str = ", ".join([f"{k}={k}" for k, v in kwargs])
    if kwkw_str:
        kwkw_str = kwkw_str + ", "
    # 如果有仅限关键字参数，则抛出异常
    if kwonlyargs:
        raise ValueError('Unexpected signature %s' % signature)

    # 构建包装函数的字符串表示形式，使其关键字参数在 pydoc.help 等文档中可见
    wrapper = """
def %(name)s(F, xin, iter=None %(kw)s, verbose=False, maxiter=None,
             f_tol=None, f_rtol=None, x_tol=None, x_rtol=None,
             tol_norm=None, line_search='armijo', callback=None, **kw):
    jac = %(jac)s(%(kwkw)s **kw)
    return nonlin_solve(F, xin, jac, iter, verbose, maxiter,
                        f_tol, f_rtol, x_tol, x_rtol, tol_norm, line_search,
                        callback)
"""

    # 根据参数填充包装函数模板
    wrapper = wrapper % dict(name=name, kw=kw_str, jac=jac.__name__,
                             kwkw=kwkw_str)
    ns = {}
    ns.update(globals())
    # 在当前全局命名空间中执行包装函数字符串
    exec(wrapper, ns)
    # 从命名空间中获取生成的函数对象
    func = ns[name]
    # 设置函数文档字符串为 jac 对象的文档字符串
    func.__doc__ = jac.__doc__
    # 设置函数的其他文档属性
    _set_doc(func)
    # 返回生成的函数对象
    return func


# 创建特定求解器的包装函数，每个函数使用不同的近似雅可比矩阵类
broyden1 = _nonlin_wrapper('broyden1', BroydenFirst)
broyden2 = _nonlin_wrapper('broyden2', BroydenSecond)
anderson = _nonlin_wrapper('anderson', Anderson)
linearmixing = _nonlin_wrapper('linearmixing', LinearMixing)
diagbroyden = _nonlin_wrapper('diagbroyden', DiagBroyden)
excitingmixing = _nonlin_wrapper('excitingmixing', ExcitingMixing)
newton_krylov = _nonlin_wrapper('newton_krylov', KrylovJacobian)
```