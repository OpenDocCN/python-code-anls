# `D:\src\scipysrc\scipy\scipy\optimize\_differentiable_functions.py`

```
import numpy as np
import scipy.sparse as sps
from ._numdiff import approx_derivative, group_columns
from ._hessian_update_strategy import HessianUpdateStrategy
from scipy.sparse.linalg import LinearOperator
from scipy._lib._array_api import atleast_nd, array_namespace

# 用于数值微分的方法列表
FD_METHODS = ('2-point', '3-point', 'cs')

# 包装函数，用于包装用户定义的目标函数，记录调用次数
def _wrapper_fun(fun, args=()):
    ncalls = [0]

    def wrapped(x):
        ncalls[0] += 1
        # 发送副本以防用户可能会覆盖它。
        # 覆盖会导致未定义的行为，因为 fun(self.x) 会改变 self.x，两者不再关联。
        fx = fun(np.copy(x), *args)
        # 确保函数返回一个真正的标量值
        if not np.isscalar(fx):
            try:
                fx = np.asarray(fx).item()
            except (TypeError, ValueError) as e:
                raise ValueError(
                    "The user-provided objective function "
                    "must return a scalar value."
                ) from e
        return fx
    return wrapped, ncalls

# 包装函数，用于包装梯度函数或者数值微分方法
def _wrapper_grad(grad, fun=None, args=(), finite_diff_options=None):
    ncalls = [0]

    if callable(grad):
        def wrapped(x, **kwds):
            # kwds 存在是为了使函数具有与数值微分变体相同的签名
            ncalls[0] += 1
            return np.atleast_1d(grad(np.copy(x), *args))
        return wrapped, ncalls

    elif grad in FD_METHODS:
        def wrapped1(x, f0=None):
            ncalls[0] += 1
            return approx_derivative(
                fun, x, f0=f0, **finite_diff_options
            )

        return wrapped1, ncalls

# 包装函数，用于包装 Hessian 矩阵函数或者数值微分方法
def _wrapper_hess(hess, grad=None, x0=None, args=(), finite_diff_options=None):
    if callable(hess):
        H = hess(np.copy(x0), *args)
        ncalls = [1]

        if sps.issparse(H):
            def wrapped(x, **kwds):
                ncalls[0] += 1
                return sps.csr_matrix(hess(np.copy(x), *args))

            H = sps.csr_matrix(H)

        elif isinstance(H, LinearOperator):
            def wrapped(x, **kwds):
                ncalls[0] += 1
                return hess(np.copy(x), *args)

        else:  # dense
            def wrapped(x, **kwds):
                ncalls[0] += 1
                return np.atleast_2d(np.asarray(hess(np.copy(x), *args)))

            H = np.atleast_2d(np.asarray(H))

        return wrapped, ncalls, H
    elif hess in FD_METHODS:
        ncalls = [0]

        def wrapped1(x, f0=None):
            return approx_derivative(
                grad, x, f0=f0, **finite_diff_options
            )

        return wrapped1, ncalls, None

# 表示一个标量函数及其导数的类
class ScalarFunction:
    """Scalar function and its derivatives.

    This class defines a scalar function F: R^n->R and methods for
    computing or approximating its first and second derivatives.

    Parameters
    ----------
    # 定义参数 `fun`，应为可调用对象，用于评估标量函数。必须是形如 `fun(x, *args)` 的函数，
    # 其中 `x` 是一个一维数组的参数，`args` 是一个包含完全指定函数的额外固定参数的元组。应返回一个标量。
    fun : callable
        
    # 提供用于评估 `fun` 的初始变量集。应为大小为 `(n,)` 的实数数组，其中 `n` 是独立变量的数量。
    x0 : array-like
        
    # 可选的额外固定参数，用于完全指定标量函数。
    args : tuple, optional
        
    # 计算梯度向量的方法。
    # 如果是可调用的，应该是一个返回梯度向量的函数：
    # ``grad(x, *args) -> array_like, shape (n,)``
    # 其中 `x` 是形状为 `(n,)` 的数组，`args` 是固定参数的元组。
    # 或者可以使用关键字 `{'2-point', '3-point', 'cs'}` 选择有限差分方案来数值估计梯度，
    # 这些有限差分方案符合指定的 `bounds`。
    grad : {callable, '2-point', '3-point', 'cs'}
        
    # 计算 Hessian 矩阵的方法。
    # 如果是可调用的，应该返回 Hessian 矩阵：
    # ``hess(x, *args) -> {LinearOperator, spmatrix, array}, (n, n)``
    # 其中 `x` 是 `(n,)` 形状的 ndarray，`args` 是固定参数的元组。
    # 或者可以使用关键字 `{'2-point', '3-point', 'cs'}` 选择数值估计的有限差分方案。
    # 或者，可以使用实现了 `HessianUpdateStrategy` 接口的对象来近似 Hessian 矩阵。
    # 当通过有限差分估计梯度时，无法使用 `{'2-point', '3-point', 'cs'}` 选项估计 Hessian，
    # 需要使用其中的一种拟牛顿策略来估计。
    hess : {callable, '2-point', '3-point', 'cs', HessianUpdateStrategy}
        
    # 用于使用的相对步长。绝对步长计算为 ``h = finite_diff_rel_step * sign(x0) * max(1, abs(x0))``，
    # 可能会调整以适应边界。对于 ``method='3-point'``，`h` 的符号将被忽略。
    # 如果为 None，则会自动选择 finite_diff_rel_step。
    finite_diff_rel_step : None or array_like
        
    # 独立变量的下限和上限。默认为没有边界，即 `(-np.inf, np.inf)`。
    # 每个边界必须与 `x0` 的大小匹配或者是一个标量。在后一种情况下，所有变量的边界将相同。
    # 用于限制函数评估的范围。
    finite_diff_bounds : tuple of array_like
    epsilon : None or array_like, optional
        Absolute step size to use, possibly adjusted to fit into the bounds.
        For ``method='3-point'`` the sign of `epsilon` is ignored. By default
        relative steps are used, only if ``epsilon is not None`` are absolute
        steps used.

    Notes
    -----
    This class implements a memoization logic. There are methods `fun`,
    `grad`, hess` and corresponding attributes `f`, `g` and `H`. The following
    things should be considered:

        1. Use only public methods `fun`, `grad` and `hess`.
        2. After one of the methods is called, the corresponding attribute
           will be set. However, a subsequent call with a different argument
           of *any* of the methods may overwrite the attribute.
    """
    @property
    def nfev(self):
        # 返回属性 `_nfev` 的第一个元素，表示函数评估的次数
        return self._nfev[0]

    @property
    def ngev(self):
        # 返回属性 `_ngev` 的第一个元素，表示梯度评估的次数
        return self._ngev[0]

    @property
    def nhev(self):
        # 返回属性 `_nhev` 的第一个元素，表示黑塞矩阵评估的次数
        return self._nhev[0]

    def _update_x(self, x):
        # 如果 `_orig_hess` 是 `HessianUpdateStrategy` 的实例
        if isinstance(self._orig_hess, HessianUpdateStrategy):
            # 更新梯度信息
            self._update_grad()
            # 将当前的 `self.x` 和 `self.g` 存储为之前的值
            self.x_prev = self.x
            self.g_prev = self.g
            # 确保 `self.x` 是 `x` 的副本，避免存储引用以保证记忆化正确工作
            _x = atleast_nd(x, ndim=1, xp=self.xp)
            self.x = self.xp.astype(_x, self.x_dtype)
            self.f_updated = False
            self.g_updated = False
            self.H_updated = False
            # 更新黑塞矩阵信息
            self._update_hess()
        else:
            # 确保 `self.x` 是 `x` 的副本，避免存储引用以保证记忆化正确工作
            _x = atleast_nd(x, ndim=1, xp=self.xp)
            self.x = self.xp.astype(_x, self.x_dtype)
            self.f_updated = False
            self.g_updated = False
            self.H_updated = False

    def _update_fun(self):
        # 如果 `self.f` 没有更新过
        if not self.f_updated:
            # 计算函数值 `fx`
            fx = self._wrapped_fun(self.x)
            # 如果 `fx` 比 `_lowest_f` 更低，则更新最低函数值和对应的 `self.x`
            if fx < self._lowest_f:
                self._lowest_x = self.x
                self._lowest_f = fx

            self.f = fx
            self.f_updated = True

    def _update_grad(self):
        # 如果 `self.g` 没有更新过
        if not self.g_updated:
            # 如果 `_orig_grad` 在 `FD_METHODS` 中
            if self._orig_grad in FD_METHODS:
                self._update_fun()
            # 计算梯度 `self.g`
            self.g = self._wrapped_grad(self.x, f0=self.f)
            self.g_updated = True

    def _update_hess(self):
        # 如果 `self.H` 没有更新过
        if not self.H_updated:
            # 如果 `_orig_hess` 在 `FD_METHODS` 中
            if self._orig_hess in FD_METHODS:
                self._update_grad()
                # 计算黑塞矩阵 `self.H`
                self.H = self._wrapped_hess(self.x, f0=self.g)
            # 如果 `_orig_hess` 是 `HessianUpdateStrategy` 的实例
            elif isinstance(self._orig_hess, HessianUpdateStrategy):
                self._update_grad()
                # 更新黑塞矩阵 `self.H`
                self.H.update(self.x - self.x_prev, self.g - self.g_prev)
            else:       # 否则，假设 `self._orig_hess` 是可调用对象
                # 计算黑塞矩阵 `self.H`
                self.H = self._wrapped_hess(self.x)

            self.H_updated = True
    # 定义一个方法 `fun`，计算给定输入 `x` 的函数值，并更新对象的状态
    def fun(self, x):
        # 如果输入的 `x` 与对象当前保存的 `x` 不相等，则更新对象的 `x`
        if not np.array_equal(x, self.x):
            self._update_x(x)
        # 更新函数值 `f` 的计算结果
        self._update_fun()
        # 返回计算得到的函数值 `f`
        return self.f

    # 定义一个方法 `grad`，计算给定输入 `x` 的梯度，并更新对象的状态
    def grad(self, x):
        # 如果输入的 `x` 与对象当前保存的 `x` 不相等，则更新对象的 `x`
        if not np.array_equal(x, self.x):
            self._update_x(x)
        # 更新梯度 `g` 的计算结果
        self._update_grad()
        # 返回计算得到的梯度 `g`
        return self.g

    # 定义一个方法 `hess`，计算给定输入 `x` 的黑塞矩阵，并更新对象的状态
    def hess(self, x):
        # 如果输入的 `x` 与对象当前保存的 `x` 不相等，则更新对象的 `x`
        if not np.array_equal(x, self.x):
            self._update_x(x)
        # 更新黑塞矩阵 `H` 的计算结果
        self._update_hess()
        # 返回计算得到的黑塞矩阵 `H`
        return self.H

    # 定义一个方法 `fun_and_grad`，计算给定输入 `x` 的函数值和梯度，并更新对象的状态
    def fun_and_grad(self, x):
        # 如果输入的 `x` 与对象当前保存的 `x` 不相等，则更新对象的 `x`
        if not np.array_equal(x, self.x):
            self._update_x(x)
        # 更新函数值 `f` 的计算结果
        self._update_fun()
        # 更新梯度 `g` 的计算结果
        self._update_grad()
        # 返回计算得到的函数值 `f` 和梯度 `g`
        return self.f, self.g
class VectorFunction:
    """Vector function and its derivatives.

    This class defines a vector function F: R^n->R^m and methods for
    computing or approximating its first and second derivatives.

    Notes
    -----
    This class implements a memoization logic. There are methods `fun`,
    `jac`, hess` and corresponding attributes `f`, `J` and `H`. The following
    things should be considered:

        1. Use only public methods `fun`, `jac` and `hess`.
        2. After one of the methods is called, the corresponding attribute
           will be set. However, a subsequent call with a different argument
           of *any* of the methods may overwrite the attribute.
    """
    def _update_v(self, v):
        if not np.array_equal(v, self.v):
            self.v = v
            self.H_updated = False

    def _update_x(self, x):
        if not np.array_equal(x, self.x):
            self._update_x_impl(x)

    def _update_fun(self):
        if not self.f_updated:
            self._update_fun_impl()
            self.f_updated = True

    def _update_jac(self):
        if not self.J_updated:
            self._update_jac_impl()
            self.J_updated = True

    def _update_hess(self):
        if not self.H_updated:
            self._update_hess_impl()
            self.H_updated = True

    def fun(self, x):
        # Update internal state with new input vector x
        self._update_x(x)
        # Ensure function value is up-to-date
        self._update_fun()
        # Return cached function value
        return self.f

    def jac(self, x):
        # Update internal state with new input vector x
        self._update_x(x)
        # Ensure Jacobian matrix is up-to-date
        self._update_jac()
        # Return cached Jacobian matrix
        return self.J

    def hess(self, x, v):
        # Update internal state with new input vector v for Hessian calculation
        # v should be updated before x.
        self._update_v(v)
        # Update internal state with new input vector x
        self._update_x(x)
        # Ensure Hessian matrix is up-to-date
        self._update_hess()
        # Return cached Hessian matrix
        return self.H


class LinearVectorFunction:
    """Linear vector function and its derivatives.

    Defines a linear function F = A x, where x is N-D vector and
    A is m-by-n matrix. The Jacobian is constant and equals to A. The Hessian
    is identically zero and it is returned as a csr matrix.
    """
    def __init__(self, A, x0, sparse_jacobian):
        # Determine if A is sparse and assign Jacobian accordingly
        if sparse_jacobian or sparse_jacobian is None and sps.issparse(A):
            self.J = sps.csr_matrix(A)
            self.sparse_jacobian = True
        elif sps.issparse(A):
            self.J = A.toarray()
            self.sparse_jacobian = False
        else:
            # Convert A to a 2-dimensional ndarray
            # np.asarray makes sure A is ndarray and not matrix
            self.J = np.atleast_2d(np.asarray(A))
            self.sparse_jacobian = False

        # Dimensions of the Jacobian matrix
        self.m, self.n = self.J.shape

        # Initialize the input vector x and its properties
        self.xp = xp = array_namespace(x0)
        _x = atleast_nd(x0, ndim=1, xp=xp)
        _dtype = xp.float64
        if xp.isdtype(_x.dtype, "real floating"):
            _dtype = _x.dtype
        # Promote _x to floating point dtype
        self.x = xp.astype(_x, _dtype)
        self.x_dtype = _dtype

        # Compute the function value F(x) = A*x
        self.f = self.J.dot(self.x)
        self.f_updated = True

        # Initialize the vector v and Hessian matrix H
        self.v = np.zeros(self.m, dtype=float)
        self.H = sps.csr_matrix((self.n, self.n))  # Hessian matrix is zero
    # 更新内部状态变量 self.x，仅当输入 x 与当前 self.x 不相等时更新
    def _update_x(self, x):
        # 检查输入的 x 是否与当前的 self.x 相等
        if not np.array_equal(x, self.x):
            # 将输入 x 转换成至少是一维的数组，并使用 self.xp 来处理数组
            _x = atleast_nd(x, ndim=1, xp=self.xp)
            # 将转换后的数组 _x 强制类型转换为 self.x_dtype，并赋值给 self.x
            self.x = self.xp.astype(_x, self.x_dtype)
            # 将标志位 self.f_updated 置为 False，表示函数值需要更新
            self.f_updated = False

    # 计算函数在给定输入 x 下的函数值，并缓存结果
    def fun(self, x):
        # 更新 self.x
        self._update_x(x)
        # 如果函数值未更新过，则重新计算函数值 self.f，并标记 self.f_updated 为 True
        if not self.f_updated:
            self.f = self.J.dot(x)
            self.f_updated = True
        # 返回函数值 self.f
        return self.f

    # 返回预先计算好的雅可比矩阵 self.J
    def jac(self, x):
        # 更新 self.x
        self._update_x(x)
        # 直接返回雅可比矩阵 self.J
        return self.J

    # 返回预先计算好的黑塞矩阵 self.H，并更新内部状态变量 self.v
    def hess(self, x, v):
        # 更新 self.x
        self._update_x(x)
        # 将输入的向量 v 赋给 self.v
        self.v = v
        # 直接返回黑塞矩阵 self.H
        return self.H
class IdentityVectorFunction(LinearVectorFunction):
    """Identity vector function and its derivatives.

    The Jacobian is the identity matrix, returned as a dense array when
    `sparse_jacobian=False` and as a csr matrix otherwise. The Hessian is
    identically zero and it is returned as a csr matrix.
    """

    # 定义身份向量函数类，继承自线性向量函数类 LinearVectorFunction

    def __init__(self, x0, sparse_jacobian):
        # 初始化函数，接受初始向量 x0 和稀疏雅可比矩阵标志 sparse_jacobian

        n = len(x0)
        # 计算初始向量 x0 的长度 n

        if sparse_jacobian or sparse_jacobian is None:
            # 如果稀疏雅可比矩阵标志为真或为 None，则使用稀疏格式的单位矩阵
            A = sps.eye(n, format='csr')
            sparse_jacobian = True
        else:
            # 否则，使用密集格式的单位矩阵
            A = np.eye(n)
            sparse_jacobian = False

        # 调用父类 LinearVectorFunction 的构造函数，传入 A 矩阵、初始向量 x0 和稀疏雅可比矩阵标志
        super().__init__(A, x0, sparse_jacobian)
```