# `D:\src\scipysrc\scipy\scipy\integrate\_ode.py`

```
# Authors: Pearu Peterson, Pauli Virtanen, John Travers
"""
First-order ODE integrators.

User-friendly interface to various numerical integrators for solving a
system of first order ODEs with prescribed initial conditions::

    d y(t)[i]
    ---------  = f(t,y(t))[i],
       d t

    y(t=0)[i] = y0[i],

where::

    i = 0, ..., len(y0) - 1

class ode
---------

A generic interface class to numeric integrators. It has the following
methods::

    integrator = ode(f, jac=None)
    integrator = integrator.set_integrator(name, **params)
    integrator = integrator.set_initial_value(y0, t0=0.0)
    integrator = integrator.set_f_params(*args)
    integrator = integrator.set_jac_params(*args)
    y1 = integrator.integrate(t1, step=False, relax=False)
    flag = integrator.successful()

class complex_ode
-----------------

This class has the same generic interface as ode, except it can handle complex
f, y and Jacobians by transparently translating them into the equivalent
real-valued system. It supports the real-valued solvers (i.e., not zvode) and is
an alternative to ode with the zvode solver, sometimes performing better.
"""
# XXX: Integrators must have:
# ===========================
# cvode - C version of vode and vodpk with many improvements.
#   Get it from http://www.netlib.org/ode/cvode.tar.gz.
#   To wrap cvode to Python, one must write the extension module by
#   hand. Its interface is too much 'advanced C' that using f2py
#   would be too complicated (or impossible).
#
# How to define a new integrator:
# ===============================
#
# class myodeint(IntegratorBase):
#
#     runner = <odeint function> or None
#
#     def __init__(self,...):                           # required
#         <initialize>
#
#     def reset(self,n,has_jac):                        # optional
#         # n - the size of the problem (number of equations)
#         # has_jac - whether user has supplied its own routine for Jacobian
#         <allocate memory,initialize further>
#
#     def run(self,f,jac,y0,t0,t1,f_params,jac_params): # required
#         # this method is called to integrate from t=t0 to t=t1
#         # with initial condition y0. f and jac are user-supplied functions
#         # that define the problem. f_params,jac_params are additional
#         # arguments
#         # to these functions.
#         <calculate y1>
#         if <calculation was unsuccessful>:
#             self.success = 0
#         return t1,y1
#
#     # In addition, one can define step() and run_relax() methods (they
#     # take the same arguments as run()) if the integrator can support
#     # these features (see IntegratorBase doc strings).
#
# if myodeint.runner:
#     IntegratorBase.integrator_classes.append(myodeint)

__all__ = ['ode', 'complex_ode']

import re                          # 导入正则表达式模块
import warnings                    # 导入警告模块

from numpy import asarray, array, zeros, isscalar, real, imag, vstack  # 从 numpy 中导入所需函数和类型

from . import _vode                # 导入模块 _vode
from . import _dop                 # 导入模块 _dop
from . import _lsoda               # 导入模块 _lsoda

_dop_int_dtype = _dop.types.intvar.dtype  # 从 _dop 模块中获取 intvar 类型的数据类型
# 将_vode.types.intvar.dtype赋值给_vode_int_dtype
_vode_int_dtype = _vode.types.intvar.dtype
# 将_lsoda.types.intvar.dtype赋值给_lsoda_int_dtype
_lsoda_int_dtype = _lsoda.types.intvar.dtype


# ------------------------------------------------------------------------------
# 用户界面
# ------------------------------------------------------------------------------


class ode:
    """
    一个通用的数值积分器接口类。

    解决方程组 :math:`y'(t) = f(t,y)`，可选 ``jac = df/dy``。

    *注意*: ``f(t, y, ...)`` 的前两个参数与 `scipy.integrate.odeint` 使用的系统定义函数的参数顺序相反。

    Parameters
    ----------
    f : callable ``f(t, y, *f_args)``
        微分方程的右侧。t 是标量，``y.shape == (n,)``。
        ``f_args`` 通过调用 ``set_f_params(*args)`` 设置。
        `f` 应返回标量、数组或列表（不是元组）。
    jac : callable ``jac(t, y, *jac_args)``, 可选
        右侧的雅可比矩阵，``jac[i,j] = d f[i] / d y[j]``。
        ``jac_args`` 通过调用 ``set_jac_params(*args)`` 设置。

    Attributes
    ----------
    t : float
        当前时间。
    y : ndarray
        当前变量值。

    See also
    --------
    odeint : 基于 ODEPACK 中的 lsoda 的更简单接口的积分器
    quad : 用于找到曲线下的面积

    Notes
    -----
    下面列出了可用的积分器。可以使用 `set_integrator` 方法选择其中一个。
    """

    def __init__(self, f, jac=None):
        # 初始化方法，接受 f 和 jac 参数
        self.f = f
        self.jac = jac
        self.t = None  # 当前时间初始化为 None
        self.y = None  # 当前变量值初始化为 None

    def set_integrator(self, name, **integrator_params):
        """
        选择数值积分器。

        Parameters
        ----------
        name : str
            积分器的名称。
        integrator_params : dict
            积分器的参数。

        Returns
        -------
        None
        """
        pass

    def set_f_params(self, *args):
        """
        设置 f 函数的参数。

        Parameters
        ----------
        *args : tuple
            f 函数的参数。

        Returns
        -------
        None
        """
        pass

    def set_jac_params(self, *args):
        """
        设置 jac 函数的参数。

        Parameters
        ----------
        *args : tuple
            jac 函数的参数。

        Returns
        -------
        None
        """
        pass


注释：
    "vode"

        Real-valued Variable-coefficient Ordinary Differential Equation
        solver, with fixed-leading-coefficient implementation. It provides
        implicit Adams method (for non-stiff problems) and a method based on
        backward differentiation formulas (BDF) (for stiff problems).

        Source: http://www.netlib.org/ode/vode.f

        .. warning::

           This integrator is not re-entrant. You cannot have two `ode`
           instances using the "vode" integrator at the same time.

        This integrator accepts the following parameters in `set_integrator`
        method of the `ode` class:

        - atol : float or sequence
          absolute tolerance for solution
        - rtol : float or sequence
          relative tolerance for solution
        - lband : None or int
          Lower bandwidth of the Jacobian matrix if it's banded
        - uband : None or int
          Upper bandwidth of the Jacobian matrix if it's banded
          Jacobian band width, jac[i,j] != 0 for i-lband <= j <= i+uband.
          Setting these requires your jac routine to return the jacobian
          in packed format, jac_packed[i-j+uband, j] = jac[i,j]. The
          dimension of the matrix must be (lband+uband+1, len(y)).
        - method: 'adams' or 'bdf'
          Which solver to use, Adams (non-stiff) or BDF (stiff)
        - with_jacobian : bool
          This option is only considered when the user has not supplied a
          Jacobian function and has not indicated (by setting either band)
          that the Jacobian is banded. In this case, `with_jacobian` specifies
          whether the iteration method of the ODE solver's correction step is
          chord iteration with an internally generated full Jacobian or
          functional iteration with no Jacobian.
        - nsteps : int
          Maximum number of (internally defined) steps allowed during one
          call to the solver.
        - first_step : float
          Initial step size for the integrator
        - min_step : float
          Minimum allowed step size
        - max_step : float
          Maximum allowed step size
          Limits for the step sizes used by the integrator.
        - order : int
          Maximum order used by the integrator,
          order <= 12 for Adams, <= 5 for BDF.
    "zvode"

        Complex-valued Variable-coefficient Ordinary Differential Equation
        solver, with fixed-leading-coefficient implementation. It provides
        implicit Adams method (for non-stiff problems) and a method based on
        backward differentiation formulas (BDF) (for stiff problems).

        Source: http://www.netlib.org/ode/zvode.f

        .. warning::

           This integrator is not re-entrant. You cannot have two `ode`
           instances using the "zvode" integrator at the same time.

        This integrator accepts the same parameters in `set_integrator`
        as the "vode" solver.

        .. note::

            When using ZVODE for a stiff system, it should only be used for
            the case in which the function f is analytic, that is, when each f(i)
            is an analytic function of each y(j). Analyticity means that the
            partial derivative df(i)/dy(j) is a unique complex number, and this
            fact is critical in the way ZVODE solves the dense or banded linear
            systems that arise in the stiff case. For a complex stiff ODE system
            in which f is not analytic, ZVODE is likely to have convergence
            failures, and for this problem one should instead use DVODE on the
            equivalent real system (in the real and imaginary parts of y).


注释：

# 描述 "zvode" 解算器的性质和用途，以及一些注意事项和要求
"zvode"

    复杂值变系数常微分方程求解器，采用固定主导系数实现。它提供了隐式阿当方法（用于非刚性问题）和基于后向差分公式（BDF）的方法（用于刚性问题）。

    Source: http://www.netlib.org/ode/zvode.f

    .. warning::

       此积分器不可重入。不能同时使用两个使用 "zvode" 积分器的 `ode` 实例。

    此积分器接受与 "vode" 解算器相同的参数在 `set_integrator` 中。

    .. note::

        当在刚性系统中使用 ZVODE 时，应仅在函数 f 是解析的情况下使用，即当每个 f(i) 是每个 y(j) 的解析函数时。解析性意味着偏导数 df(i)/dy(j) 是唯一的复数，这一事实对 ZVODE 解决在刚性情况下出现的密集或带状线性系统至关重要。对于一个复杂的刚性ODE系统，其中 f 不是解析的，ZVODE 可能会出现收敛失败，对于这种问题，应该使用 DVODE 处理等效的实数系统（y 的实部和虚部）。
    # "lsoda" 是一个实现了变系数常微分方程求解器，包含固定前导系数的实现。它提供了在隐式亚当方法（用于非刚性问题）和基于向后差分公式（BDF）的方法（用于刚性问题）之间自动切换。
    
    # 参考来源: http://www.netlib.org/odepack
    
    # .. warning::
    #    此积分器不是可重入的。你不能同时使用两个 `ode` 实例来使用 "lsoda" 积分器。
    
    # 此积分器在 `ode` 类的 `set_integrator` 方法中接受以下参数:
    
    # - atol : float 或者 sequence
    #   解的绝对容忍度
    # - rtol : float 或者 sequence
    #   解的相对容忍度
    # - lband : None 或者 int
    # - uband : None 或者 int
    #   雅可比矩阵的带宽，对于 i-lband <= j <= i+uband，jac[i,j] != 0。
    #   设置这些参数需要你的 jac 程序以打包格式返回雅可比矩阵，jac_packed[i-j+uband, j] = jac[i,j]。
    # - with_jacobian : bool
    #   *未使用*
    # - nsteps : int
    #   求解器在一次调用中允许的最大步数（内部定义的步数）
    # - first_step : float
    # - min_step : float
    # - max_step : float
    #   积分器使用的步长限制
    # - max_order_ns : int
    #   非刚性情况下使用的最大阶数（默认为12）
    # - max_order_s : int
    #   刚性情况下使用的最大阶数（默认为5）
    # - max_hnil : int
    #   报告步长过小的最大消息数（t + h = t）（默认为0）
    # - ixpr : int
    #   是否在方法切换时生成额外的打印输出（默认为False）
    "dopri5"

        This is an explicit runge-kutta method of order (4)5 due to Dormand &
        Prince (with stepsize control and dense output).

        Authors:

            E. Hairer and G. Wanner
            Universite de Geneve, Dept. de Mathematiques
            CH-1211 Geneve 24, Switzerland
            e-mail:  ernst.hairer@math.unige.ch, gerhard.wanner@math.unige.ch

        This code is described in [HNW93]_.

        This integrator accepts the following parameters in set_integrator()
        method of the ode class:

        - atol : float or sequence
          absolute tolerance for solution
        - rtol : float or sequence
          relative tolerance for solution
        - nsteps : int
          Maximum number of (internally defined) steps allowed during one
          call to the solver.
        - first_step : float
        - max_step : float
        - safety : float
          Safety factor on new step selection (default 0.9)
        - ifactor : float
        - dfactor : float
          Maximum factor to increase/decrease step size by in one step
        - beta : float
          Beta parameter for stabilised step size control.
        - verbosity : int
          Switch for printing messages (< 0 for no messages).

    "dop853"

        This is an explicit runge-kutta method of order 8(5,3) due to Dormand
        & Prince (with stepsize control and dense output).

        Options and references the same as "dopri5".
    # 初始化函数，用于设置对象的刚性属性和初始条件
    def __init__(self, f, jac=None):
        # 刚性属性初始化为0
        self.stiff = 0
        # 函数 f 的赋值
        self.f = f
        # Jacobian 矩阵的赋值，可选参数
        self.jac = jac
        # 函数 f 的参数，初始为空元组
        self.f_params = ()
        # Jacobian 矩阵的参数，初始为空元组
        self.jac_params = ()
        # 积分器的状态变量，初始为空列表
        self._y = []

    @property
    # 属性装饰器，用于访问私有变量 _y
    def y(self):
        return self._y

    # 设置初始条件的方法
    def set_initial_value(self, y, t=0.0):
        """Set initial conditions y(t) = y."""
        # 如果 y 是标量，则转换为列表
        if isscalar(y):
            y = [y]
        # 获取当前 _y 的长度
        n_prev = len(self._y)
        # 如果之前没有设置过初始条件，则尝试设置第一个可用的积分器
        if not n_prev:
            self.set_integrator('')  # find first available integrator
        # 将 y 转换为 numpy 数组，并根据积分器的标量类型进行处理
        self._y = asarray(y, self._integrator.scalar)
        # 设置当前时间为 t
        self.t = t
        # 重置积分器，以便更新积分器状态和参数
        self._integrator.reset(len(self._y), self.jac is not None)
        return self

    # 设置积分器的方法
    def set_integrator(self, name, **integrator_params):
        """
        Set integrator by name.

        Parameters
        ----------
        name : str
            Name of the integrator.
        **integrator_params
            Additional parameters for the integrator.
        """
        # 查找指定名称的积分器
        integrator = find_integrator(name)
        # 如果找不到对应的积分器
        if integrator is None:
            # 发出警告，指出找不到或不可用的积分器名称
            message = f'No integrator name match with {name!r} or is not available.'
            warnings.warn(message, stacklevel=2)
        else:
            # 使用找到的积分器及其参数进行初始化
            self._integrator = integrator(**integrator_params)
            # 如果当前没有设置过初始条件，则初始化 _y 和时间 t
            if not len(self._y):
                self.t = 0.0
                self._y = array([0.0], self._integrator.scalar)
            # 重置积分器，以便更新积分器状态和参数
            self._integrator.reset(len(self._y), self.jac is not None)
        return self
    def integrate(self, t, step=False, relax=False):
        """Find y=y(t), set y as an initial condition, and return y.

        Parameters
        ----------
        t : float
            The endpoint of the integration step.
        step : bool
            If True, and if the integrator supports the step method,
            then perform a single integration step and return.
            This parameter is provided in order to expose internals of
            the implementation, and should not be changed from its default
            value in most cases.
        relax : bool
            If True and if the integrator supports the run_relax method,
            then integrate until t_1 >= t and return. ``relax`` is not
            referenced if ``step=True``.
            This parameter is provided in order to expose internals of the implementation,
            and should not be changed from its default value in most cases.

        Returns
        -------
        y : float
            The integrated value at t
        """
        # Determine the method to use for integration based on parameters
        if step and self._integrator.supports_step:
            mth = self._integrator.step  # Assign the step method of the integrator
        elif relax and self._integrator.supports_run_relax:
            mth = self._integrator.run_relax  # Assign the run_relax method of the integrator
        else:
            mth = self._integrator.run  # Default to the run method of the integrator

        try:
            # Perform the integration using the selected method
            self._y, self.t = mth(self.f, self.jac or (lambda: None),
                                  self._y, self.t, t,
                                  self.f_params, self.jac_params)
        except SystemError as e:
            # Handle case where the integrator function returns a tuple, which is not expected
            raise ValueError(
                'Function to integrate must not return a tuple.'
            ) from e

        return self._y  # Return the integrated value y

    def successful(self):
        """Check if integration was successful."""
        try:
            self._integrator
        except AttributeError:
            self.set_integrator('')  # If _integrator is not set, initialize it with an empty string
        return self._integrator.success == 1  # Return True if integration was successful, False otherwise

    def set_f_params(self, *args):
        """Set extra parameters for user-supplied function f."""
        self.f_params = args  # Set the parameters for function f
        return self  # Return self for method chaining

    def set_jac_params(self, *args):
        """Set extra parameters for user-supplied function jac."""
        self.jac_params = args  # Set the parameters for function jac
        return self  # Return self for method chaining
    # 设置在每个成功的积分步骤中调用的可调用对象。
    def set_solout(self, solout):
        """
        Set callable to be called at every successful integration step.

        Parameters
        ----------
        solout : callable
            ``solout(t, y)`` is called at each internal integrator step,
            t is a scalar providing the current independent position
            y is the current solution ``y.shape == (n,)``
            solout should return -1 to stop integration
            otherwise it should return None or 0

        """
        # 检查积分器是否支持设置 solout 回调函数
        if self._integrator.supports_solout:
            # 将 solout 回调函数设置给积分器对象
            self._integrator.set_solout(solout)
            # 如果当前状态向量 _y 不为 None，则重置积分器状态
            if self._y is not None:
                self._integrator.reset(len(self._y), self.jac is not None)
        else:
            # 如果积分器不支持 solout 回调函数，则抛出 ValueError 异常
            raise ValueError("selected integrator does not support solout,"
                             " choose another one")
def _transform_banded_jac(bjac):
    """
    Convert a real matrix of the form (for example)

        [0 0 A B]        [0 0 0 B]
        [0 0 C D]        [0 0 A D]
        [E F G H]   to   [0 F C H]
        [I J K L]        [E J G L]
                         [I 0 K 0]

    That is, every other column is shifted up one.
    """
    # 创建一个新的矩阵，行数比原矩阵多一行，列数与原矩阵相同
    newjac = zeros((bjac.shape[0] + 1, bjac.shape[1]))
    # 将原矩阵每隔一列的元素复制到新矩阵的每隔一列
    newjac[1:, ::2] = bjac[:, ::2]
    # 将原矩阵每隔一列的下移一位元素复制到新矩阵的每隔一列
    newjac[:-1, 1::2] = bjac[:, 1::2]
    return newjac


class complex_ode(ode):
    """
    A wrapper of ode for complex systems.

    This functions similarly as `ode`, but re-maps a complex-valued
    equation system to a real-valued one before using the integrators.

    Parameters
    ----------
    f : callable ``f(t, y, *f_args)``
        Rhs of the equation. t is a scalar, ``y.shape == (n,)``.
        ``f_args`` is set by calling ``set_f_params(*args)``.
    jac : callable ``jac(t, y, *jac_args)``
        Jacobian of the rhs, ``jac[i,j] = d f[i] / d y[j]``.
        ``jac_args`` is set by calling ``set_f_params(*args)``.

    Attributes
    ----------
    t : float
        Current time.
    y : ndarray
        Current variable values.

    Examples
    --------
    For usage examples, see `ode`.

    """

    def __init__(self, f, jac=None):
        self.cf = f
        self.cjac = jac
        if jac is None:
            # 如果未提供雅可比矩阵函数，使用内部的复数到实数映射函数
            ode.__init__(self, self._wrap, None)
        else:
            # 如果提供了雅可比矩阵函数，使用内部的复数到实数映射函数和复数雅可比矩阵映射函数
            ode.__init__(self, self._wrap, self._wrap_jac)

    def _wrap(self, t, y, *f_args):
        # 计算复数形式的右手边函数的实部和虚部
        f = self.cf(*((t, y[::2] + 1j * y[1::2]) + f_args))
        # self.tmp 是一个包含交错实部和虚部的实数数组
        self.tmp[::2] = real(f)
        self.tmp[1::2] = imag(f)
        return self.tmp

    def _wrap_jac(self, t, y, *jac_args):
        # jac 是用户定义函数计算的复数雅可比矩阵
        jac = self.cjac(*((t, y[::2] + 1j * y[1::2]) + jac_args))

        # jac_tmp 是复数雅可比矩阵的实数版本。每个复数元素变成了一个2x2块
        #     [2 -3]
        #     [3  2]
        jac_tmp = zeros((2 * jac.shape[0], 2 * jac.shape[1]))
        jac_tmp[1::2, 1::2] = jac_tmp[::2, ::2] = real(jac)
        jac_tmp[1::2, ::2] = imag(jac)
        jac_tmp[::2, 1::2] = -jac_tmp[1::2, ::2]

        ml = getattr(self._integrator, 'ml', None)
        mu = getattr(self._integrator, 'mu', None)
        if ml is not None or mu is not None:
            # 雅可比矩阵是带状的。用户的雅可比函数以打包格式计算复数雅可比矩阵。
            # 对应的实数版本需要将每隔一列上移。
            jac_tmp = _transform_banded_jac(jac_tmp)

        return jac_tmp

    @property
    def y(self):
        # 返回当前变量值的实部和虚部的交错数组
        return self._y[::2] + 1j * self._y[1::2]
    def set_integrator(self, name, **integrator_params):
        """
        Set integrator by name.

        Parameters
        ----------
        name : str
            Name of the integrator
        **integrator_params
            Additional parameters for the integrator.
        """
        if name == 'zvode':
            raise ValueError("zvode must be used with ode, not complex_ode")

        # 获取用户提供的带宽参数
        lband = integrator_params.get('lband')
        uband = integrator_params.get('uband')
        if lband is not None or uband is not None:
            # 如果用户提供了带宽参数，则将复杂雅可比矩阵的带宽转换为对应实数雅可比矩阵的带宽
            integrator_params['lband'] = 2 * (lband or 0) + 1
            integrator_params['uband'] = 2 * (uband or 0) + 1

        # 调用ode模块的set_integrator方法设置积分器，并传递参数
        return ode.set_integrator(self, name, **integrator_params)

    def set_initial_value(self, y, t=0.0):
        """Set initial conditions y(t) = y."""
        # 将y转换为数组形式
        y = asarray(y)
        # 创建一个临时数组，用于存储实部和虚部交替存放的y值
        self.tmp = zeros(y.size * 2, 'float')
        self.tmp[::2] = real(y)  # 将实部存入临时数组的偶数索引位置
        self.tmp[1::2] = imag(y)  # 将虚部存入临时数组的奇数索引位置
        # 调用ode模块的set_initial_value方法设置初始值，并传递临时数组及时间参数t
        return ode.set_initial_value(self, self.tmp, t)

    def integrate(self, t, step=False, relax=False):
        """Find y=y(t), set y as an initial condition, and return y.

        Parameters
        ----------
        t : float
            The endpoint of the integration step.
        step : bool
            If True, and if the integrator supports the step method,
            then perform a single integration step and return.
            This parameter is provided in order to expose internals of
            the implementation, and should not be changed from its default
            value in most cases.
        relax : bool
            If True and if the integrator supports the run_relax method,
            then integrate until t_1 >= t and return. ``relax`` is not
            referenced if ``step=True``.
            This parameter is provided in order to expose internals of
            the implementation, and should not be changed from its default
            value in most cases.

        Returns
        -------
        y : float
            The integrated value at t
        """
        # 调用ode模块的integrate方法进行积分计算，并获取结果y
        y = ode.integrate(self, t, step, relax)
        # 将复合数组转换为复数数组并返回，实部部分加上虚部乘以虚数单位
        return y[::2] + 1j * y[1::2]
    def set_solout(self, solout):
        """
        Set callable to be called at every successful integration step.

        Parameters
        ----------
        solout : callable
            ``solout(t, y)`` is called at each internal integrator step,
            t is a scalar providing the current independent position
            y is the current solution ``y.shape == (n,)``
            solout should return -1 to stop integration
            otherwise it should return None or 0

        """
        # 检查当前积分器是否支持设置 solout 回调函数
        if self._integrator.supports_solout:
            # 若支持，则将 solout 回调函数设置给积分器，使用复数参数
            self._integrator.set_solout(solout, complex=True)
        else:
            # 若不支持，则抛出类型错误异常，提示选择另一个积分器
            raise TypeError("selected integrator does not support solouta, "
                            "choose another one")
# ------------------------------------------------------------------------------
# ODE integrators
# ------------------------------------------------------------------------------

# 根据给定的名称查找并返回对应的积分器类
def find_integrator(name):
    for cl in IntegratorBase.integrator_classes:
        # 使用正则表达式忽略大小写地匹配积分器类的名称
        if re.match(name, cl.__name__, re.I):
            return cl
    # 如果找不到匹配的积分器类，则返回None
    return None


class IntegratorConcurrencyError(RuntimeError):
    """
    Failure due to concurrent usage of an integrator that can be used
    only for a single problem at a time.

    """

    def __init__(self, name):
        # 根据传入的积分器类名构造异常消息
        msg = ("Integrator `%s` can be used to solve only a single problem "
               "at a time. If you want to integrate multiple problems, "
               "consider using a different integrator "
               "(see `ode.set_integrator`)") % name
        # 调用父类的初始化方法，将消息传递给 RuntimeError
        RuntimeError.__init__(self, msg)


class IntegratorBase:
    # runner is None => integrator is not available
    runner = None
    # success==1 if integrator was called successfully
    success = None
    # istate > 0 means success, istate < 0 means failure
    istate = None
    supports_run_relax = None
    supports_step = None
    supports_solout = False
    integrator_classes = []  # 存储所有已注册的积分器类
    scalar = float  # 标量类型，默认为浮点数

    def acquire_new_handle(self):
        # 一些积分器具有内部状态（如古老的Fortran...），因此一次只能使用一个实例。
        # 我们跟踪这一点，并在尝试并发使用时失败。
        # 增加全局活跃句柄计数器，确保每个实例都有唯一的句柄
        self.__class__.active_global_handle += 1
        # 将当前句柄分配给实例的handle属性
        self.handle = self.__class__.active_global_handle

    def check_handle(self):
        # 检查当前实例的句柄是否与全局活跃句柄计数器相匹配
        if self.handle is not self.__class__.active_global_handle:
            # 如果不匹配，则抛出积分器并发使用错误
            raise IntegratorConcurrencyError(self.__class__.__name__)

    def reset(self, n, has_jac):
        """Prepare integrator for call: allocate memory, set flags, etc.
        n - number of equations.
        has_jac - if user has supplied function for evaluating Jacobian.
        """
        # 准备积分器以进行调用：分配内存、设置标志等
        # n - 方程数量
        # has_jac - 用户是否提供了评估雅可比矩阵函数

    def run(self, f, jac, y0, t0, t1, f_params, jac_params):
        """Integrate from t=t0 to t=t1 using y0 as an initial condition.
        Return 2-tuple (y1,t1) where y1 is the result and t=t1
        defines the stoppage coordinate of the result.
        """
        # 积分从t=t0到t=t1，使用y0作为初始条件进行积分
        # 返回一个二元组(y1, t1)，其中y1是结果，t=t1定义了结果的停止坐标
        raise NotImplementedError('all integrators must define '
                                  'run(f, jac, t0, t1, y0, f_params, jac_params)')

    def step(self, f, jac, y0, t0, t1, f_params, jac_params):
        """Make one integration step and return (y1,t1)."""
        # 进行一步积分并返回(y1, t1)
        raise NotImplementedError('%s does not support step() method' %
                                  self.__class__.__name__)

    def run_relax(self, f, jac, y0, t0, t1, f_params, jac_params):
        """Integrate from t=t0 to t>=t1 and return (y1,t)."""
        # 从t=t0积分到t>=t1并返回(y1, t)
        raise NotImplementedError('%s does not support run_relax() method' %
                                  self.__class__.__name__)

    # XXX: __str__ method for getting visual state of the integrator
    # 用于获取积分器可视化状态的__str__方法
# 定义一个内部函数 `_vode_banded_jac_wrapper`，用于包装带有带状结构雅可比矩阵的函数，将雅可比矩阵在底部填充 `ml` 行零。
def _vode_banded_jac_wrapper(jacfunc, ml, jac_params):
    """
    Wrap a banded Jacobian function with a function that pads
    the Jacobian with `ml` rows of zeros.
    """

    # 定义内部函数 `jac_wrapper`，接受时间 `t` 和状态向量 `y` 作为参数
    def jac_wrapper(t, y):
        # 调用传入的 `jacfunc` 函数计算雅可比矩阵，并转换为 NumPy 数组
        jac = asarray(jacfunc(t, y, *jac_params))
        # 在雅可比矩阵 `jac` 底部填充 `ml` 行零，得到 `padded_jac`
        padded_jac = vstack((jac, zeros((ml, jac.shape[1]))))
        # 返回填充后的雅可比矩阵 `padded_jac`
        return padded_jac

    # 返回内部函数 `jac_wrapper` 作为结果
    return jac_wrapper


# 定义一个类 `vode`，继承自 `IntegratorBase`
class vode(IntegratorBase):
    # 获取 `_vode` 模块中的 `dvode` 属性，并赋值给类变量 `runner`
    runner = getattr(_vode, 'dvode', None)

    # 定义一个消息字典，将整数码映射到相应的消息文本
    messages = {
        -1: 'Excess work done on this call. (Perhaps wrong MF.)',
        -2: 'Excess accuracy requested. (Tolerances too small.)',
        -3: 'Illegal input detected. (See printed message.)',
        -4: 'Repeated error test failures. (Check all input.)',
        -5: 'Repeated convergence failures. (Perhaps bad'
            ' Jacobian supplied or wrong choice of MF or tolerances.)',
        -6: 'Error weight became zero during problem. (Solution'
            ' component i vanished, and ATOL or ATOL(i) = 0.)'
    }

    # 设置支持的属性 `supports_run_relax` 和 `supports_step` 的初始值
    supports_run_relax = 1
    supports_step = 1
    active_global_handle = 0

    # 构造函数 `__init__`，初始化 `vode` 类的实例
    def __init__(self,
                 method='adams',  # 集成方法，默认为 'adams'
                 with_jacobian=False,  # 是否使用雅可比矩阵，默认为 False
                 rtol=1e-6, atol=1e-12,  # 相对容差和绝对容差的初始值
                 lband=None, uband=None,  # 下带和上带的初始值
                 order=12,  # 集成方法的阶数，默认为 12
                 nsteps=500,  # 最大步数，默认为 500
                 max_step=0.0,  # 最大步长，默认为无限制
                 min_step=0.0,  # 最小步长，默认为无限制
                 first_step=0.0,  # 首步长，默认由求解器确定
                 ):

        # 根据传入的 `method` 参数选择集成方法
        if re.match(method, r'adams', re.I):
            self.meth = 1  # Adams 方法对应数字 1
        elif re.match(method, r'bdf', re.I):
            self.meth = 2  # BDF 方法对应数字 2
        else:
            # 如果 `method` 不匹配 'adams' 或 'bdf'，抛出 ValueError 异常
            raise ValueError('Unknown integration method %s' % method)

        # 设置对象的其他属性
        self.with_jacobian = with_jacobian
        self.rtol = rtol
        self.atol = atol
        self.mu = uband
        self.ml = lband
        self.order = order
        self.nsteps = nsteps
        self.max_step = max_step
        self.min_step = min_step
        self.first_step = first_step
        self.success = 1  # 表示初始化成功的标志
        self.initialized = False  # 表示对象是否已经初始化的标志
    def _determine_mf_and_set_bands(self, has_jac):
        """
        Determine the `MF` parameter (Method Flag) for the Fortran subroutine `dvode`.

        In the Fortran code, the legal values of `MF` are:
            10, 11, 12, 13, 14, 15, 20, 21, 22, 23, 24, 25,
            -11, -12, -14, -15, -21, -22, -24, -25
        but this Python wrapper does not use negative values.

        Returns

            mf  = 10*self.meth + miter

        self.meth is the linear multistep method:
            self.meth == 1:  method="adams"
            self.meth == 2:  method="bdf"

        miter is the correction iteration method:
            miter == 0:  Functional iteration; no Jacobian involved.
            miter == 1:  Chord iteration with user-supplied full Jacobian.
            miter == 2:  Chord iteration with internally computed full Jacobian.
            miter == 3:  Chord iteration with internally computed diagonal Jacobian.
            miter == 4:  Chord iteration with user-supplied banded Jacobian.
            miter == 5:  Chord iteration with internally computed banded Jacobian.

        Side effects: If either self.mu or self.ml is not None and the other is None,
        then the one that is None is set to 0.
        """

        # Determine if the Jacobian matrix is banded
        jac_is_banded = self.mu is not None or self.ml is not None

        # Adjust self.mu and self.ml if one is None and the other is not None
        if jac_is_banded:
            if self.mu is None:
                self.mu = 0
            if self.ml is None:
                self.ml = 0

        # Determine the value of miter based on whether a Jacobian function is provided
        if has_jac:
            if jac_is_banded:
                miter = 4  # Chord iteration with user-supplied banded Jacobian.
            else:
                miter = 1  # Chord iteration with user-supplied full Jacobian.
        else:
            if jac_is_banded:
                if self.ml == self.mu == 0:
                    miter = 3  # Chord iteration with internal diagonal Jacobian.
                else:
                    miter = 5  # Chord iteration with internal banded Jacobian.
            else:
                if self.with_jacobian:
                    miter = 2  # Chord iteration with internal full Jacobian.
                else:
                    miter = 0  # Functional iteration; no Jacobian involved.

        # Calculate the final value of MF based on self.meth and miter
        mf = 10 * self.meth + miter
        return mf
    # 重置方法，根据输入的参数n和has_jac确定mf，然后计算lrw和liw，设置rwork和iwork数组，初始化call_args等属性
    def reset(self, n, has_jac):
        mf = self._determine_mf_and_set_bands(has_jac)

        # 根据mf的不同值计算lrw的长度
        if mf == 10:
            lrw = 20 + 16 * n
        elif mf in [11, 12]:
            lrw = 22 + 16 * n + 2 * n * n
        elif mf == 13:
            lrw = 22 + 17 * n
        elif mf in [14, 15]:
            lrw = 22 + 18 * n + (3 * self.ml + 2 * self.mu) * n
        elif mf == 20:
            lrw = 20 + 9 * n
        elif mf in [21, 22]:
            lrw = 22 + 9 * n + 2 * n * n
        elif mf == 23:
            lrw = 22 + 10 * n
        elif mf in [24, 25]:
            lrw = 22 + 11 * n + (3 * self.ml + 2 * self.mu) * n
        else:
            raise ValueError('Unexpected mf=%s' % mf)

        # 根据mf % 10的值确定liw的长度
        if mf % 10 in [0, 3]:
            liw = 30
        else:
            liw = 30 + n

        # 初始化rwork数组，并设置其部分固定值
        rwork = zeros((lrw,), float)
        rwork[4] = self.first_step
        rwork[5] = self.max_step
        rwork[6] = self.min_step
        self.rwork = rwork

        # 初始化iwork数组，并设置其部分固定值
        iwork = zeros((liw,), _vode_int_dtype)
        if self.ml is not None:
            iwork[0] = self.ml
        if self.mu is not None:
            iwork[1] = self.mu
        iwork[4] = self.order
        iwork[5] = self.nsteps
        iwork[6] = 2  # mxhnil
        self.iwork = iwork

        # 设置call_args列表，包括一些初始化的参数
        self.call_args = [self.rtol, self.atol, 1, 1,
                          self.rwork, self.iwork, mf]
        self.success = 1
        self.initialized = False

    # 运行方法，调用runner方法来执行ODE求解，处理异常状态，返回计算结果y1和t
    def run(self, f, jac, y0, t0, t1, f_params, jac_params):
        if self.initialized:
            self.check_handle()  # 检查句柄是否有效
        else:
            self.initialized = True
            self.acquire_new_handle()  # 获取新的句柄

        # 如果ml存在且大于0，则使用_vode_banded_jac_wrapper包装jac函数
        if self.ml is not None and self.ml > 0:
            # 带状雅可比矩阵的情况，包装用户提供的jac函数
            jac = _vode_banded_jac_wrapper(jac, self.ml, jac_params)

        # 构建参数元组，包括f, jac, y0, t0, t1等以及self.call_args等属性
        args = ((f, jac, y0, t0, t1) + tuple(self.call_args) +
                (f_params, jac_params))
        # 调用runner方法执行计算
        y1, t, istate = self.runner(*args)
        self.istate = istate
        if istate < 0:
            # 如果istate小于0，发出警告
            unexpected_istate_msg = f'Unexpected istate={istate:d}'
            warnings.warn('{:s}: {:s}'.format(self.__class__.__name__,
                          self.messages.get(istate, unexpected_istate_msg)),
                          stacklevel=2)
            self.success = 0
        else:
            # 如果istate为正数，则更新self.call_args的第3个元素为2
            self.call_args[3] = 2  # upgrade istate from 1 to 2
            self.istate = 2
        return y1, t

    # 单步计算方法，先将itask设为2，然后调用run方法执行计算，最后将itask还原
    def step(self, *args):
        itask = self.call_args[2]
        self.call_args[2] = 2
        r = self.run(*args)
        self.call_args[2] = itask
        return r

    # 松弛运行方法，先将itask设为3，然后调用run方法执行计算，最后将itask还原
    def run_relax(self, *args):
        itask = self.call_args[2]
        self.call_args[2] = 3
        r = self.run(*args)
        self.call_args[2] = itask
        return r
# 如果 vode.runner 不是 None，则将 vode 类型的积分器添加到 IntegratorBase 类的 integrator_classes 列表中
if vode.runner is not None:
    IntegratorBase.integrator_classes.append(vode)

# 定义 zvode 类，继承自 vode 类
class zvode(vode):
    # 获取 _vode 模块中的 zvode 属性，如果不存在则为 None
    runner = getattr(_vode, 'zvode', None)

    # 支持运行放松算法
    supports_run_relax = 1
    # 支持单步运算
    supports_step = 1
    # 标量类型为复数
    scalar = complex
    # 激活全局处理句柄
    active_global_handle = 0

    # 重置方法，根据给定参数 n 和是否有雅各比矩阵 has_jac 来设定 mf 值，并计算 lzw 的值
    def reset(self, n, has_jac):
        # 确定 mf 值并设置带状存储矩阵
        mf = self._determine_mf_and_set_bands(has_jac)

        # 根据不同的 mf 值计算 lzw 的大小
        if mf in (10,):
            lzw = 15 * n
        elif mf in (11, 12):
            lzw = 15 * n + 2 * n ** 2
        elif mf in (-11, -12):
            lzw = 15 * n + n ** 2
        elif mf in (13,):
            lzw = 16 * n
        elif mf in (14, 15):
            lzw = 17 * n + (3 * self.ml + 2 * self.mu) * n
        elif mf in (-14, -15):
            lzw = 16 * n + (2 * self.ml + self.mu) * n
        elif mf in (20,):
            lzw = 8 * n
        elif mf in (21, 22):
            lzw = 8 * n + 2 * n ** 2
        elif mf in (-21, -22):
            lzw = 8 * n + n ** 2
        elif mf in (23,):
            lzw = 9 * n
        elif mf in (24, 25):
            lzw = 10 * n + (3 * self.ml + 2 * self.mu) * n
        elif mf in (-24, -25):
            lzw = 9 * n + (2 * self.ml + self.mu) * n

        # 计算 lrw 的大小
        lrw = 20 + n

        # 根据 mf 的值计算 liw 的大小
        if mf % 10 in (0, 3):
            liw = 30
        else:
            liw = 30 + n

        # 初始化复数数组 zwork
        zwork = zeros((lzw,), complex)
        self.zwork = zwork

        # 初始化浮点数数组 rwork，设置其中的一些参数
        rwork = zeros((lrw,), float)
        rwork[4] = self.first_step
        rwork[5] = self.max_step
        rwork[6] = self.min_step
        self.rwork = rwork

        # 初始化整数数组 iwork，设置其中的一些参数
        iwork = zeros((liw,), _vode_int_dtype)
        if self.ml is not None:
            iwork[0] = self.ml
        if self.mu is not None:
            iwork[1] = self.mu
        iwork[4] = self.order
        iwork[5] = self.nsteps
        iwork[6] = 2  # mxhnil
        self.iwork = iwork

        # 设置调用参数列表
        self.call_args = [self.rtol, self.atol, 1, 1,
                          self.zwork, self.rwork, self.iwork, mf]
        # 设置成功标志
        self.success = 1
        # 初始化标志为 False
        self.initialized = False

# 如果 zvode.runner 不是 None，则将 zvode 类型的积分器添加到 IntegratorBase 类的 integrator_classes 列表中
if zvode.runner is not None:
    IntegratorBase.integrator_classes.append(zvode)

# 定义 dopri5 类，继承自 IntegratorBase 类
class dopri5(IntegratorBase):
    # 获取 _dop 模块中的 dopri5 属性，如果不存在则为 None
    runner = getattr(_dop, 'dopri5', None)
    # 设置名称为 'dopri5'
    name = 'dopri5'
    # 支持 solout 方法
    supports_solout = True

    # 定义消息字典，用于返回不同计算结果的消息
    messages = {1: 'computation successful',
                2: 'computation successful (interrupted by solout)',
                -1: 'input is not consistent',
                -2: 'larger nsteps is needed',
                -3: 'step size becomes too small',
                -4: 'problem is probably stiff (interrupted)',
                }
    # 初始化函数，设置默认参数和属性
    def __init__(self,
                 rtol=1e-6, atol=1e-12,  # 相对和绝对容差
                 nsteps=500,  # 最大步数
                 max_step=0.0,  # 最大步长
                 first_step=0.0,  # 初始步长，由求解器确定
                 safety=0.9,  # 安全系数
                 ifactor=10.0,  # 增长因子
                 dfactor=0.2,  # 减小因子
                 beta=0.0,  # 未使用
                 method=None,  # 未使用
                 verbosity=-1,  # 负数时无消息输出
                 ):
        self.rtol = rtol  # 设置相对容差
        self.atol = atol  # 设置绝对容差
        self.nsteps = nsteps  # 设置最大步数
        self.max_step = max_step  # 设置最大步长
        self.first_step = first_step  # 设置初始步长
        self.safety = safety  # 设置安全系数
        self.ifactor = ifactor  # 设置增长因子
        self.dfactor = dfactor  # 设置减小因子
        self.beta = beta  # 设置未使用参数
        self.verbosity = verbosity  # 设置消息输出级别
        self.success = 1  # 初始化成功标志为1
        self.set_solout(None)  # 调用设置回调函数的方法并传入None作为参数
    
    # 设置回调函数的方法
    def set_solout(self, solout, complex=False):
        self.solout = solout  # 设置回调函数
        self.solout_cmplx = complex  # 设置是否处理复数
        if solout is None:
            self.iout = 0  # 如果回调函数为空，则设置输出标志为0
        else:
            self.iout = 1  # 否则设置输出标志为1
    
    # 重置方法，重新初始化工作数组和整数工作数组
    def reset(self, n, has_jac):
        work = zeros((8 * n + 21,), float)  # 初始化工作数组
        work[1] = self.safety  # 设置工作数组的第一个元素为安全系数
        work[2] = self.dfactor  # 设置工作数组的第二个元素为减小因子
        work[3] = self.ifactor  # 设置工作数组的第三个元素为增长因子
        work[4] = self.beta  # 设置工作数组的第四个元素为beta
        work[5] = self.max_step  # 设置工作数组的第五个元素为最大步长
        work[6] = self.first_step  # 设置工作数组的第六个元素为初始步长
        self.work = work  # 将工作数组存储到对象的属性中
        iwork = zeros((21,), _dop_int_dtype)  # 初始化整数工作数组
        iwork[0] = self.nsteps  # 设置整数工作数组的第一个元素为最大步数
        iwork[2] = self.verbosity  # 设置整数工作数组的第三个元素为消息输出级别
        self.iwork = iwork  # 将整数工作数组存储到对象的属性中
        self.call_args = [self.rtol, self.atol, self._solout,
                          self.iout, self.work, self.iwork]  # 设置调用参数列表
        self.success = 1  # 重置成功标志为1
    
    # 运行方法，执行求解器的运行，并返回结果
    def run(self, f, jac, y0, t0, t1, f_params, jac_params):
        x, y, iwork, istate = self.runner(*((f, t0, y0, t1) +
                                          tuple(self.call_args) + (f_params,)))  # 调用runner方法并获取返回值
        self.istate = istate  # 存储状态码
        if istate < 0:  # 如果状态码小于0
            unexpected_istate_msg = f'Unexpected istate={istate:d}'
            warnings.warn('{:s}: {:s}'.format(self.__class__.__name__,
                          self.messages.get(istate, unexpected_istate_msg)),
                          stacklevel=2)  # 发出警告
            self.success = 0  # 设置成功标志为0
        return y, x  # 返回结果
    
    # 默认的回调函数，根据设置调用回调或返回1
    def _solout(self, nr, xold, x, y, nd, icomp, con):
        if self.solout is not None:  # 如果回调函数不为空
            if self.solout_cmplx:  # 如果需要处理复数
                y = y[::2] + 1j * y[1::2]  # 将y转换为复数形式
            return self.solout(x, y)  # 调用回调函数并返回结果
        else:
            return 1  # 否则返回1
# 如果 dopri5.runner 不为 None，则将 dopri5 添加到 IntegratorBase.integrator_classes 列表中
if dopri5.runner is not None:
    IntegratorBase.integrator_classes.append(dopri5)

# 定义 dop853 类，继承自 dopri5 类
class dop853(dopri5):
    # 获取 _dop 模块中的 dop853 运行器，如果不存在则为 None
    runner = getattr(_dop, 'dop853', None)
    # 设置名称为 'dop853'
    name = 'dop853'

    # 初始化方法，接受多个参数设置默认的积分器参数
    def __init__(self,
                 rtol=1e-6, atol=1e-12,
                 nsteps=500,
                 max_step=0.0,
                 first_step=0.0,  # 由求解器决定
                 safety=0.9,
                 ifactor=6.0,
                 dfactor=0.3,
                 beta=0.0,
                 method=None,
                 verbosity=-1,  # 如果为负数则不输出信息
                 ):
        # 调用父类 dopri5 的初始化方法，设置积分器参数
        super().__init__(rtol, atol, nsteps, max_step, first_step, safety,
                         ifactor, dfactor, beta, method, verbosity)

    # 重置方法，初始化工作空间和整数工作空间，设置参数
    def reset(self, n, has_jac):
        # 创建大小为 (11 * n + 21,) 的浮点数数组，并初始化
        work = zeros((11 * n + 21,), float)
        work[1] = self.safety
        work[2] = self.dfactor
        work[3] = self.ifactor
        work[4] = self.beta
        work[5] = self.max_step
        work[6] = self.first_step
        self.work = work
        
        # 创建大小为 (21,) 的整数数组，并初始化
        iwork = zeros((21,), _dop_int_dtype)
        iwork[0] = self.nsteps
        iwork[2] = self.verbosity
        self.iwork = iwork
        
        # 设置调用参数列表
        self.call_args = [self.rtol, self.atol, self._solout,
                          self.iout, self.work, self.iwork]
        # 设置成功标志为 1
        self.success = 1

# 如果 dop853.runner 不为 None，则将 dop853 添加到 IntegratorBase.integrator_classes 列表中
if dop853.runner is not None:
    IntegratorBase.integrator_classes.append(dop853)

# 定义 lsoda 类，继承自 IntegratorBase 类
class lsoda(IntegratorBase):
    # 获取 _lsoda 模块中的 lsoda 运行器，如果不存在则为 None
    runner = getattr(_lsoda, 'lsoda', None)
    # 设置全局活跃处理标志为 0
    active_global_handle = 0

    # 消息字典，映射整数到对应的消息字符串
    messages = {
        2: "Integration successful.",
        -1: "Excess work done on this call (perhaps wrong Dfun type).",
        -2: "Excess accuracy requested (tolerances too small).",
        -3: "Illegal input detected (internal error).",
        -4: "Repeated error test failures (internal error).",
        -5: "Repeated convergence failures (perhaps bad Jacobian or tolerances).",
        -6: "Error weight became zero during problem.",
        -7: "Internal workspace insufficient to finish (internal error)."
    }

    # 初始化方法，接受多个参数设置 lsoda 积分器的默认参数
    def __init__(self,
                 with_jacobian=False,
                 rtol=1e-6, atol=1e-12,
                 lband=None, uband=None,
                 nsteps=500,
                 max_step=0.0,  # 对应无限
                 min_step=0.0,
                 first_step=0.0,  # 由求解器决定
                 ixpr=0,
                 max_hnil=0,
                 max_order_ns=12,
                 max_order_s=5,
                 method=None
                 ):

        # 设置是否使用雅可比矩阵
        self.with_jacobian = with_jacobian
        # 设置相对误差和绝对误差容忍度
        self.rtol = rtol
        self.atol = atol
        # 设置上三带和下三带矩阵
        self.mu = uband
        self.ml = lband

        # 设置非刚性模式和刚性模式的最大积分阶数
        self.max_order_ns = max_order_ns
        self.max_order_s = max_order_s
        # 设置最大步数和最小步数
        self.nsteps = nsteps
        self.max_step = max_step
        self.min_step = min_step
        # 设置初始步长
        self.first_step = first_step
        # 设置输出控制标志
        self.ixpr = ixpr
        # 设置最大步长未变化次数
        self.max_hnil = max_hnil
        # 设置成功标志为 1
        self.success = 1

        # 初始化未完成标志
        self.initialized = False
    # 重置方法，设置积分器参数并初始化
    def reset(self, n, has_jac):
        # 计算用于 Fortran 子例程 dvode 的参数。
        if has_jac:
            # 如果有雅可比矩阵
            if self.mu is None and self.ml is None:
                jt = 1
            else:
                if self.mu is None:
                    self.mu = 0
                if self.ml is None:
                    self.ml = 0
                jt = 4
        else:
            # 如果没有雅可比矩阵
            if self.mu is None and self.ml is None:
                jt = 2
            else:
                if self.mu is None:
                    self.mu = 0
                if self.ml is None:
                    self.ml = 0
                jt = 5
        # 计算所需的工作空间大小
        lrn = 20 + (self.max_order_ns + 4) * n
        if jt in [1, 2]:
            lrs = 22 + (self.max_order_s + 4) * n + n * n
        elif jt in [4, 5]:
            lrs = 22 + (self.max_order_s + 5 + 2 * self.ml + self.mu) * n
        else:
            raise ValueError('Unexpected jt=%s' % jt)
        lrw = max(lrn, lrs)
        liw = 20 + n
        # 初始化工作数组和整型数组
        rwork = zeros((lrw,), float)
        rwork[4] = self.first_step
        rwork[5] = self.max_step
        rwork[6] = self.min_step
        self.rwork = rwork
        iwork = zeros((liw,), _lsoda_int_dtype)
        if self.ml is not None:
            iwork[0] = self.ml
        if self.mu is not None:
            iwork[1] = self.mu
        iwork[4] = self.ixpr
        iwork[5] = self.nsteps
        iwork[6] = self.max_hnil
        iwork[7] = self.max_order_ns
        iwork[8] = self.max_order_s
        self.iwork = iwork
        # 设置调用参数列表
        self.call_args = [self.rtol, self.atol, 1, 1,
                          self.rwork, self.iwork, jt]
        self.success = 1
        self.initialized = False

    # 运行积分器进行仿真
    def run(self, f, jac, y0, t0, t1, f_params, jac_params):
        if self.initialized:
            self.check_handle()
        else:
            self.initialized = True
            self.acquire_new_handle()
        # 准备调用参数列表
        args = [f, y0, t0, t1] + self.call_args[:-1] + \
               [jac, self.call_args[-1], f_params, 0, jac_params]
        # 调用积分器运行函数
        y1, t, istate = self.runner(*args)
        self.istate = istate
        # 处理返回的状态 istate
        if istate < 0:
            # 如果 istate 小于零，发出警告
            unexpected_istate_msg = f'Unexpected istate={istate:d}'
            warnings.warn('{:s}: {:s}'.format(self.__class__.__name__,
                          self.messages.get(istate, unexpected_istate_msg)),
                          stacklevel=2)
            self.success = 0
        else:
            # 如果 istate 大于等于零，设置 success 标志为成功
            self.call_args[3] = 2  # 升级 istate 从 1 到 2
            self.istate = 2
        return y1, t

    # 单步运行积分器
    def step(self, *args):
        itask = self.call_args[2]
        self.call_args[2] = 2
        r = self.run(*args)
        self.call_args[2] = itask
        return r

    # 运行积分器进行松弛计算
    def run_relax(self, *args):
        itask = self.call_args[2]
        self.call_args[2] = 3
        r = self.run(*args)
        self.call_args[2] = itask
        return r
# 如果 lsoda.runner 存在（即不为 None 或者 False），则执行以下操作
if lsoda.runner:
    # 将 lsoda 添加到 IntegratorBase 类的 integrator_classes 列表中
    IntegratorBase.integrator_classes.append(lsoda)
```