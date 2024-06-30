# `D:\src\scipysrc\scipy\scipy\signal\_lti_conversion.py`

```
"""
ltisys -- a collection of functions to convert linear time invariant systems
from one representation to another.
"""

# 导入必要的库
import numpy as np
from numpy import (r_, eye, atleast_2d, poly, dot,
                   asarray, zeros, array, outer)
from scipy import linalg

# 导入本地模块中的函数
from ._filter_design import tf2zpk, zpk2tf, normalize

# 定义模块的公共接口，仅包括以下函数
__all__ = ['tf2ss', 'abcd_normalize', 'ss2tf', 'zpk2ss', 'ss2zpk',
           'cont2discrete']

# 定义函数，将传递的分子和分母多项式转换为状态空间表示
def tf2ss(num, den):
    r"""Transfer function to state-space representation.

    Parameters
    ----------
    num, den : array_like
        Sequences representing the coefficients of the numerator and
        denominator polynomials, in order of descending degree. The
        denominator needs to be at least as long as the numerator.

    Returns
    -------
    A, B, C, D : ndarray
        State space representation of the system, in controller canonical
        form.

    Examples
    --------
    Convert the transfer function:

    .. math:: H(s) = \frac{s^2 + 3s + 3}{s^2 + 2s + 1}

    >>> num = [1, 3, 3]
    >>> den = [1, 2, 1]

    to the state-space representation:

    .. math::

        \dot{\textbf{x}}(t) =
        \begin{bmatrix} -2 & -1 \\ 1 & 0 \end{bmatrix} \textbf{x}(t) +
        \begin{bmatrix} 1 \\ 0 \end{bmatrix} \textbf{u}(t) \\

        \textbf{y}(t) = \begin{bmatrix} 1 & 2 \end{bmatrix} \textbf{x}(t) +
        \begin{bmatrix} 1 \end{bmatrix} \textbf{u}(t)

    >>> from scipy.signal import tf2ss
    >>> A, B, C, D = tf2ss(num, den)
    >>> A
    array([[-2., -1.],
           [ 1.,  0.]])
    >>> B
    array([[ 1.],
           [ 0.]])
    >>> C
    array([[ 1.,  2.]])
    >>> D
    array([[ 1.]])

    Notes
    -----
    Controller canonical state-space representation is used where states
    are derived from the transfer function's coefficients, ensuring
    compatibility with state-space operations.
    """
    # 控制规范的状态空间表示。
    # 如果 M+1 = len(num) 且 K+1 = len(den)，则必须有 M <= K
    # 状态通过断言 X(s) = U(s) / D(s) 来找到
    # 然后 Y(s) = N(s) * X(s)
    #
    # A, B, C 和 D 随之而来。

    # 规范化传入的分子和分母多项式
    num, den = normalize(num, den)   # 去除零项，检查数组
    nn = len(num.shape)
    if nn == 1:
        num = asarray([num], num.dtype)
    M = num.shape[1]
    K = len(den)
    if M > K:
        msg = "Improper transfer function. `num` is longer than `den`."
        raise ValueError(msg)
    if M == 0 or K == 0:  # 空系统
        return (array([], float), array([], float), array([], float),
                array([], float))

    # 将分子填充到与分母相同的列数
    num = np.hstack((np.zeros((num.shape[0], K - M), dtype=num.dtype), num))

    if num.shape[-1] > 0:
        D = atleast_2d(num[:, 0])
    else:
        # 如果没有非零的 D 矩阵，仍应分配一个非零形状，以便可以使用 'ss2tf' 等函数操作它
        D = array([[0]], float)
    # 如果 K 等于 1，执行以下操作：
    if K == 1:
        # 将 D 重塑为与 num.shape 相同的形状
        D = D.reshape(num.shape)

        # 返回四个元组：一个 1x1 的零矩阵，一个 1x(D.shape[1]) 的零矩阵，
        # 一个 (D.shape[0])x1 的零矩阵，以及 D 本身
        return (zeros((1, 1)), zeros((1, D.shape[1])),
                zeros((D.shape[0], 1)), D)

    # 计算 frow，它是 den[1:] 的负数组成的一维数组
    frow = -array([den[1:]])
    # 构建 A 矩阵，包含 frow 和一个 (K-2)x(K-1) 的单位矩阵
    A = r_[frow, eye(K - 2, K - 1)]
    # 构建 B 矩阵，一个 (K-1)x1 的单位矩阵
    B = eye(K - 1, 1)
    # 构建 C 矩阵，num[:, 1:] 减去 num[:, 0] 与 den[1:] 的外积
    C = num[:, 1:] - outer(num[:, 0], den[1:])
    # 将 D 重塑为 (C.shape[0])x(B.shape[1]) 的形状
    D = D.reshape((C.shape[0], B.shape[1]))

    # 返回 A, B, C, D 四个矩阵
    return A, B, C, D
def _none_to_empty_2d(arg):
    # 如果参数为 None，则返回一个形状为 (0, 0) 的零数组
    if arg is None:
        return zeros((0, 0))
    else:
        return arg


def _atleast_2d_or_none(arg):
    # 如果参数不为 None，则将其转换为至少二维数组
    if arg is not None:
        return atleast_2d(arg)


def _shape_or_none(M):
    # 如果参数不为 None，则返回其形状；否则返回一个形状为 (None, None) 的元组
    if M is not None:
        return M.shape
    else:
        return (None,) * 2


def _choice_not_none(*args):
    # 遍历参数列表，返回第一个不为 None 的参数
    for arg in args:
        if arg is not None:
            return arg


def _restore(M, shape):
    # 如果 M 的形状为 (0, 0)，则返回一个指定形状的零数组；否则检查形状是否相符，如果不符则抛出 ValueError 异常
    if M.shape == (0, 0):
        return zeros(shape)
    else:
        if M.shape != shape:
            raise ValueError("The input arrays have incompatible shapes.")
        return M


def abcd_normalize(A=None, B=None, C=None, D=None):
    """Check state-space matrices and ensure they are 2-D.

    If enough information on the system is provided, that is, enough
    properly-shaped arrays are passed to the function, the missing ones
    are built from this information, ensuring the correct number of
    rows and columns. Otherwise a ValueError is raised.

    Parameters
    ----------
    A, B, C, D : array_like, optional
        State-space matrices. All of them are None (missing) by default.
        See `ss2tf` for format.

    Returns
    -------
    A, B, C, D : array
        Properly shaped state-space matrices.

    Raises
    ------
    ValueError
        If not enough information on the system was provided.

    """
    # 将 A, B, C, D 中每个非 None 的参数转换为至少二维数组
    A, B, C, D = map(_atleast_2d_or_none, (A, B, C, D))

    # 获取每个数组的形状或返回 (None, None) 的元组
    MA, NA = _shape_or_none(A)
    MB, NB = _shape_or_none(B)
    MC, NC = _shape_or_none(C)
    MD, ND = _shape_or_none(D)

    # 选择非 None 的 MA, MB, NC 中的值作为 p
    p = _choice_not_none(MA, MB, NC)
    # 选择非 None 的 NB, ND 中的值作为 q
    q = _choice_not_none(NB, ND)
    # 选择非 None 的 MC, MD 中的值作为 r
    r = _choice_not_none(MC, MD)
    # 如果 p, q, r 中有任何一个为 None，则抛出 ValueError 异常
    if p is None or q is None or r is None:
        raise ValueError("Not enough information on the system.")

    # 将 A, B, C, D 中每个 None 转换为形状为 (0, 0) 的零数组
    A, B, C, D = map(_none_to_empty_2d, (A, B, C, D))
    # 恢复 A, B, C, D 的形状，确保与给定的 p, q, r 相符
    A = _restore(A, (p, p))
    B = _restore(B, (p, q))
    C = _restore(C, (r, p))
    D = _restore(D, (r, q))

    return A, B, C, D
    # 转换状态空间表示为传递函数：

    # 系统状态方程
    \dot{\textbf{x}}(t) =
    \begin{bmatrix} -2 & -1 \\ 1 & 0 \end{bmatrix} \textbf{x}(t) +
    \begin{bmatrix} 1 \\ 0 \end{bmatrix} \textbf{u}(t)

    # 输出方程
    \textbf{y}(t) = \begin{bmatrix} 1 & 2 \end{bmatrix} \textbf{x}(t) +
    \begin{bmatrix} 1 \end{bmatrix} \textbf{u}(t)

>>> A = [[-2, -1], [1, 0]]
>>> B = [[1], [0]]  # 二维列向量
>>> C = [[1, 2]]    # 二维行向量
>>> D = 1

# 将状态空间表示转换为传递函数表示
>>> from scipy.signal import ss2tf
>>> ss2tf(A, B, C, D)
(array([[1., 3., 3.]]), array([ 1.,  2.,  1.]))


"""
# 传递函数为 C (sI - A)**(-1) B + D

# 检查参数的一致性，并将它们都转换为二维数组
A, B, C, D = abcd_normalize(A, B, C, D)

nout, nin = D.shape
# 如果输入序号大于等于输入的数量，则抛出数值错误
if input >= nin:
    raise ValueError("System does not have the input specified.")

# 如果系统是多输入多输出，则转换为单输入多输出系统
B = B[:, input:input + 1]
D = D[:, input:input + 1]

try:
    # 计算特征多项式的分母
    den = poly(A)
except ValueError:
    den = 1

# 如果 B 和 C 均为空，则返回直流增益和传递函数的分母
if (B.size == 0) and (C.size == 0):
    num = np.ravel(D)
    if (D.size == 0) and (A.size == 0):
        den = []
    return num, den

# 计算传递函数的分子
num_states = A.shape[0]
type_test = A[:, 0] + B[:, 0] + C[0, :] + D + 0.0
num = np.empty((nout, num_states + 1), type_test.dtype)
for k in range(nout):
    Ck = atleast_2d(C[k, :])
    num[k] = poly(A - dot(B, Ck)) + (D[k] - 1) * den

return num, den
def cont2discrete(system, dt, method="zoh", alpha=None):
    """
    Transform a continuous to a discrete state-space system.

    Parameters
    ----------
    system : a tuple describing the system or an instance of `lti`
        The following gives the number of elements in the tuple and
        the interpretation:

            * 1: (instance of `lti`)
            * 2: (num, den)
            * 3: (zeros, poles, gain)
            * 4: (A, B, C, D)

    dt : float
        The discretization time step.
    method : str, optional
        Which method to use:

            * gbt: generalized bilinear transformation
            * bilinear: Tustin's approximation ("gbt" with alpha=0.5)
            * euler: Euler (or forward differencing) method ("gbt" with alpha=0)
            * backward_diff: Backwards differencing ("gbt" with alpha=1.0)
            * zoh: zero-order hold (default)
            * foh: first-order hold (*versionadded: 1.3.0*)
            * impulse: equivalent impulse response (*versionadded: 1.3.0*)

    alpha : float within [0, 1], optional
        The generalized bilinear transformation weighting parameter, which
        should only be specified with method="gbt", and is ignored otherwise

    Returns
    -------
    sysd : tuple containing the discrete system
        Based on the input type, the output will be of the form

        * (num, den, dt)   for transfer function input
        * (zeros, poles, gain, dt)   for zeros-poles-gain input
        * (A, B, C, D, dt) for state-space system input

    Notes
    -----
    By default, the routine uses a Zero-Order Hold (zoh) method to perform
    the transformation. Alternatively, a generalized bilinear transformation
    may be used, which includes the common Tustin's bilinear approximation,

    """
    # 根据输入的系统描述和指定的离散化方法，将连续时间状态空间系统转换为离散时间状态空间系统
    pass
    """
    If the input `system` is of specific lengths, convert continuous-time systems to discrete-time using various methods.
    
    Parameters
    ----------
    system : tuple
        A tuple representing the continuous-time system in different forms:
        - Length 1: Transfer function (tf)
        - Length 2: Numerator and denominator arrays of a transfer function (tf)
        - Length 3: Zero-pole-gain (zpk) representation
        - Length 4: State-space (ss) representation
    
    dt : float
        Sampling interval (time step) for discretization.
    
    method : str, optional
        Method used for discretization:
        - 'zoh': Zero-Order Hold method
        - 'bilinear': Generalized bilinear approximation
        - 'euler': Euler's method technique
        - 'backward_diff': Backwards differencing technique
        - 'foh': First-Order Hold method
        - 'impulse': Impulse invariance method
    
    alpha : float, optional
        Parameter used in certain discretization methods.
    
    Returns
    -------
    tuple
        Depending on the length of `system`, returns the discrete-time system in different representations:
        - Length 1: Returns as a discrete transfer function (tf) with the sampling interval appended.
        - Length 2: Converts from tf to state-space (ss) and then returns as a discrete transfer function (tf) with the sampling interval appended.
        - Length 3: Converts from zpk to state-space (ss) and then returns as a discrete zero-pole-gain (zpk) representation with the sampling interval appended.
        - Length 4: Raises a ValueError since state-space (ss) representation is expected in the fourth argument.
    
    Raises
    ------
    ValueError
        If the length of `system` is not 1, 2, 3, or 4.
    
    Notes
    -----
    The methods for discretization are based on specific references:
    - 'zoh': Zero-Order Hold method based on [1]_.
    - 'bilinear': Generalized bilinear approximation based on [2]_ and [3]_.
    - 'foh': First-Order Hold method based on [4]_.
    
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Discretization#Discretization_of_linear_state_space_models
    .. [2] http://techteach.no/publications/discretetime_signals_systems/discrete.pdf
    .. [3] G. Zhang, X. Chen, and T. Chen, Digital redesign via the generalized bilinear transformation,
           Int. J. Control, vol. 82, no. 4, pp. 741-754, 2009. (https://www.mypolyuweb.hk/~magzhang/Research/ZCC09_IJC.pdf)
    .. [4] G. F. Franklin, J. D. Powell, and M. L. Workman, Digital control of dynamic systems,
           3rd ed. Menlo Park, Calif: Addison-Wesley, pp. 204-206, 1998.
    
    Examples
    --------
    Example of transforming a continuous state-space system to a discrete one using various methods:
    >>> import numpy as np
    >>> from scipy.signal import cont2discrete, lti, dstep
    >>> A = np.array([[0, 1],[-10., -3]])
    >>> B = np.array([[0],[10.]])
    >>> C = np.array([[1., 0]])
    >>> D = np.array([[0.]])
    >>> l_system = lti(A, B, C, D)
    >>> dt = 0.1
    >>> for method in ['zoh', 'bilinear', 'euler', 'backward_diff', 'foh', 'impulse']:
    ...    d_system = cont2discrete((A, B, C, D), dt, method=method)
    ...    s, x_d = dstep(d_system)
    ...    # Plot the step response of each discrete system method
    ...    plt.step(s, np.squeeze(x_d), label=method, where='post')
    >>> plt.legend(loc='best')
    >>> plt.show()
    """
    if len(system) == 1:
        # Convert single transfer function to discrete-time representation
        return system.to_discrete()
    if len(system) == 2:
        # Convert numerator-denominator representation to discrete-time transfer function
        sysd = cont2discrete(tf2ss(system[0], system[1]), dt, method=method, alpha=alpha)
        return ss2tf(sysd[0], sysd[1], sysd[2], sysd[3]) + (dt,)
    elif len(system) == 3:
        # Convert zero-pole-gain representation to discrete-time zero-pole-gain
        sysd = cont2discrete(zpk2ss(system[0], system[1], system[2]), dt, method=method, alpha=alpha)
        return ss2zpk(sysd[0], sysd[1], sysd[2], sysd[3]) + (dt,)
    elif len(system) == 4:
        # Handle state-space representation directly
        a, b, c, d = system
    else:
        # Raise error if the length of `system` does not match expected forms
        raise ValueError("First argument must either be a tuple of 2 (tf), 3 (zpk), or 4 (ss) arrays.")
    ```python`
        if method == 'gbt':
            # 如果使用的是广义双线性变换 (gbt) 方法，则需要检查 alpha 参数
            if alpha is None:
                # 如果 alpha 参数未指定，则抛出数值错误
                raise ValueError("Alpha parameter must be specified for the "
                                 "generalized bilinear transform (gbt) method")
            elif alpha < 0 or alpha > 1:
                # 如果 alpha 参数超出范围 [0,1]，则抛出数值错误
                raise ValueError("Alpha parameter must be within the interval "
                                 "[0,1] for the gbt method")
    
            # 计算状态空间方程的离散化矩阵 ad, bd, cd, dd
            # ima 是单位矩阵减去 alpha * dt * a
            ima = np.eye(a.shape[0]) - alpha*dt*a
            # 计算 ad 矩阵
            ad = linalg.solve(ima, np.eye(a.shape[0]) + (1.0-alpha)*dt*a)
            # 计算 bd 矩阵
            bd = linalg.solve(ima, dt*b)
    
            # 计算 cd 矩阵
            cd = linalg.solve(ima.transpose(), c.transpose())
            cd = cd.transpose()
            # 计算 dd 矩阵
            dd = d + alpha*np.dot(c, bd)
    
        elif method == 'bilinear' or method == 'tustin':
            # 如果使用双线性变换或 Tustin 方法，则返回其离散化结果
            return cont2discrete(system, dt, method="gbt", alpha=0.5)
    
        elif method == 'euler' or method == 'forward_diff':
            # 如果使用欧拉方法或前向差分法，则返回其离散化结果
            return cont2discrete(system, dt, method="gbt", alpha=0.0)
    
        elif method == 'backward_diff':
            # 如果使用后向差分法，则返回其离散化结果
            return cont2discrete(system, dt, method="gbt", alpha=1.0)
    
        elif method == 'zoh':
            # 如果使用零阶保持法 (zoh) 方法，则构建指数矩阵
            em_upper = np.hstack((a, b))
            em_lower = np.hstack((np.zeros((b.shape[1], a.shape[0])),
                                  np.zeros((b.shape[1], b.shape[1]))))
            em = np.vstack((em_upper, em_lower))
            # 计算指数矩阵的指数函数
            ms = linalg.expm(dt * em)
    
            # 取指数矩阵的上半部分作为结果
            ms = ms[:a.shape[0], :]
    
            # 分别得到 ad 和 bd 矩阵
            ad = ms[:, 0:a.shape[1]]
            bd = ms[:, a.shape[1]:]
    
            # cd 和 dd 矩阵直接复制
            cd = c
            dd = d
    
        elif method == 'foh':
            # 如果使用一阶保持法 (foh) 方法，则构建指数矩阵
            n = a.shape[0]
            m = b.shape[1]
            em_upper = linalg.block_diag(np.block([a, b]) * dt, np.eye(m))
            em_lower = zeros((m, n + 2 * m))
            em = np.block([[em_upper], [em_lower]])
    
            # 计算指数矩阵的指数函数
            ms = linalg.expm(em)
    
            # 分块取出所需的矩阵块
            ms11 = ms[:n, 0:n]
            ms12 = ms[:n, n:n + m]
            ms13 = ms[:n, n + m:]
    
            # 计算 ad, bd, cd, dd 矩阵
            ad = ms11
            bd = ms12 - ms13 + ms11 @ ms13
            cd = c
            dd = d + c @ ms13
    
        elif method == 'impulse':
            # 如果使用冲激响应方法，则检查 d 矩阵是否为零
            if not np.allclose(d, 0):
                # 如果 d 矩阵不全为零，则抛出数值错误
                raise ValueError("Impulse method is only applicable "
                                 "to strictly proper systems")
    
            # 计算 ad, bd, cd, dd 矩阵
            ad = linalg.expm(a * dt)
            bd = ad @ b * dt
            cd = c
            dd = c @ b * dt
    
        else:
            # 如果方法未知，则抛出数值错误
            raise ValueError("Unknown transformation method '%s'" % method)
    
        # 返回计算得到的 ad, bd, cd, dd 矩阵以及离散化步长 dt
        return ad, bd, cd, dd, dt
```