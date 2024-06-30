# `D:\src\scipysrc\scipy\scipy\integrate\_quadpack_py.py`

```
# 导入必要的模块和库
import sys
import warnings
from functools import partial

# 导入 _quadpack 模块，这是一个 C 编写的 Fortran QUADPACK 库的接口
from . import _quadpack

# 导入 NumPy 库，并将其命名为 np，用于数值计算
import numpy as np

# 指定可以从本模块导入的公共接口
__all__ = ["quad", "dblquad", "tplquad", "nquad", "IntegrationWarning"]


class IntegrationWarning(UserWarning):
    """
    对积分过程中可能出现的问题发出警告。
    """
    pass


def quad(func, a, b, args=(), full_output=0, epsabs=1.49e-8, epsrel=1.49e-8,
         limit=50, points=None, weight=None, wvar=None, wopts=None, maxp1=50,
         limlst=50, complex_func=False):
    """
    计算定积分。

    使用 Fortran 库 QUADPACK 中的技术，对 func 从 `a` 到 `b` （可能是无穷区间）进行积分。

    Parameters
    ----------
    func : {function, scipy.LowLevelCallable}
        要积分的 Python 函数或方法。如果 `func` 接受多个参数，则沿着第一个参数对应的轴进行积分。

        如果用户希望提高积分性能，则 `func` 可以是一个 `scipy.LowLevelCallable`，具有以下签名之一：

            double func(double x)
            double func(double x, void *user_data)
            double func(int n, double *xx)
            double func(int n, double *xx, void *user_data)

        `user_data` 是包含在 `scipy.LowLevelCallable` 中的数据。在带有 `xx` 的调用形式中，
        `n` 是包含 `xx[0] == x` 和 `args` 参数中的其余项目的数组长度。

        此外，某些 ctypes 调用签名也支持向后兼容性，但不应在新代码中使用。
    a : float
        积分下限（使用 -numpy.inf 表示负无穷）。
    b : float
        积分上限（使用 numpy.inf 表示正无穷）。
    args : tuple, optional
        传递给 `func` 的额外参数。
    full_output : int, optional
        非零以返回积分信息的字典。
        如果非零，还会抑制警告消息，并将消息附加到输出元组中。
    complex_func : bool, optional
        指示函数 `func` 的返回类型是实数（`complex_func=False`：默认）还是复数（`complex_func=True`）。
        在两种情况下，函数的参数都是实数。
        如果 full_output 也是非零，则在带有键 "real output" 和 "imag output" 的字典中返回实部和虚部的 `infodict`、`message` 和 `explain`。

    Returns
    -------
    y : float
        func 从 `a` 到 `b` 的积分值。
    abserr : float
        结果的绝对误差估计。
    infodict : dict
        包含附加信息的字典。
    message
        收敛消息。
    explain
        Appended only with 'cos' or 'sin' weighting and infinite
        integration limits, it contains an explanation of the codes in
        infodict['ierlst']

    Other Parameters
    ----------------
    epsabs : float or int, optional
        Absolute error tolerance. Default is 1.49e-8. `quad` tries to obtain
        an accuracy of ``abs(i-result) <= max(epsabs, epsrel*abs(i))``
        where ``i`` = integral of `func` from `a` to `b`, and ``result`` is the
        numerical approximation. See `epsrel` below.
    epsrel : float or int, optional
        Relative error tolerance. Default is 1.49e-8.
        If ``epsabs <= 0``, `epsrel` must be greater than both 5e-29
        and ``50 * (machine epsilon)``. See `epsabs` above.
    limit : float or int, optional
        An upper bound on the number of subintervals used in the adaptive
        algorithm.
    points : (sequence of floats,ints), optional
        A sequence of break points in the bounded integration interval
        where local difficulties of the integrand may occur (e.g.,
        singularities, discontinuities). The sequence does not have
        to be sorted. Note that this option cannot be used in conjunction
        with ``weight``.
    weight : float or int, optional
        String indicating weighting function. Full explanation for this
        and the remaining arguments can be found below.
    wvar : optional
        Variables for use with weighting functions.
    wopts : optional
        Optional input for reusing Chebyshev moments.
    maxp1 : float or int, optional
        An upper bound on the number of Chebyshev moments.
    limlst : int, optional
        Upper bound on the number of cycles (>=3) for use with a sinusoidal
        weighting and an infinite end-point.

    See Also
    --------
    dblquad : double integral
    tplquad : triple integral
    nquad : n-dimensional integrals (uses `quad` recursively)
    fixed_quad : fixed-order Gaussian quadrature
    simpson : integrator for sampled data
    romb : integrator for sampled data
    scipy.special : for coefficients and roots of orthogonal polynomials

    Notes
    -----
    For valid results, the integral must converge; behavior for divergent
    integrals is not guaranteed.

    **Extra information for quad() inputs and outputs**

    If full_output is non-zero, then the third output argument
    (infodict) is a dictionary with entries as tabulated below. For
    infinite limits, the range is transformed to (0,1) and the
    optional outputs are given with respect to this transformed range.
    Let M be the input argument limit and let K be infodict['last'].
    The entries are:

    'neval'
        The number of function evaluations.
    'last'
        The number, K, of subintervals produced in the subdivision process.
    'alist'
        A rank-1 array of length M, the first K elements of which are the
        left end points of the subintervals in the partition of the
        integration range.
    'blist'
        A rank-1 array of length M, the first K elements of which are the
        right end points of the subintervals.
    'rlist'
        A rank-1 array of length M, the first K elements of which are the
        integral approximations on the subintervals.
    'elist'
        A rank-1 array of length M, the first K elements of which are the
        moduli of the absolute error estimates on the subintervals.
    'iord'
        A rank-1 integer array of length M, the first L elements of
        which are pointers to the error estimates over the subintervals
        with ``L=K`` if ``K<=M/2+2`` or ``L=M+1-K`` otherwise. Let I be the
        sequence ``infodict['iord']`` and let E be the sequence
        ``infodict['elist']``.  Then ``E[I[1]], ..., E[I[L]]`` forms a
        decreasing sequence.

        If the input argument points is provided (i.e., it is not None),
        the following additional outputs are placed in the output
        dictionary. Assume the points sequence is of length P.

    'pts'
        A rank-1 array of length P+2 containing the integration limits
        and the break points of the intervals in ascending order.
        This is an array giving the subintervals over which integration
        will occur.
    'level'
        A rank-1 integer array of length M (=limit), containing the
        subdivision levels of the subintervals, i.e., if (aa,bb) is a
        subinterval of ``(pts[1], pts[2])`` where ``pts[0]`` and ``pts[2]``
        are adjacent elements of ``infodict['pts']``, then (aa,bb) has level l
        if ``|bb-aa| = |pts[2]-pts[1]| * 2**(-l)``.
    'ndin'
        A rank-1 integer array of length P+2. After the first integration
        over the intervals (pts[1], pts[2]), the error estimates over some
        of the intervals may have been increased artificially in order to
        put their subdivision forward. This array has ones in slots
        corresponding to the subintervals for which this happens.

        **Weighting the integrand**

        The input variables, *weight* and *wvar*, are used to weight the
        integrand by a select list of functions. Different integration
        methods are used to compute the integral with these weighting
        functions, and these do not support specifying break points. The
        possible values of weight and the corresponding weighting functions are.

        ==========  ===================================   =====================
        ``weight``  Weight function used                  ``wvar``
        ==========  ===================================   =====================
        'cos'       cos(w*x)                              wvar = w
        'sin'       sin(w*x)                              wvar = w
        'alg'       g(x) = ((x-a)**alpha)*((b-x)**beta)   wvar = (alpha, beta)
    # Weighting functions and corresponding integration formulas:
    'alg-loga'  g(x)*log(x-a)                         wvar = (alpha, beta)
    'alg-logb'  g(x)*log(b-x)                         wvar = (alpha, beta)
    'alg-log'   g(x)*log(x-a)*log(b-x)                wvar = (alpha, beta)
    'cauchy'    1/(x-c)                               wvar = c
    ==========  ===================================   =====================

    # Explanation of 'wvar':
    # - For 'alg-loga', 'alg-logb', and 'alg-log' weighting functions, wvar represents a tuple (alpha, beta).
    #   - alpha and beta are parameters depending on the specific weighting function chosen.
    #   - 'alg-loga': g(x) * log(x-a), where a is one of the integration limits.
    #   - 'alg-logb': g(x) * log(b-x), where b is the other integration limit.
    #   - 'alg-log': g(x) * log(x-a) * log(b-x), combining both limits.
    # - For 'cauchy' weighting, wvar is simply the parameter c.
    #   - 'cauchy': 1/(x-c), where c is a constant parameter.

    wvar holds the parameter w, (alpha, beta), or c depending on the weight
    selected. In these expressions, a and b are the integration limits.

    For the 'cos' and 'sin' weighting, additional inputs and outputs are
    available.

    For finite integration limits, the integration is performed using a
    Clenshaw-Curtis method which uses Chebyshev moments. For repeated
    calculations, these moments are saved in the output dictionary:

    'momcom'
        The maximum level of Chebyshev moments that have been computed,
        i.e., if ``M_c`` is ``infodict['momcom']`` then the moments have been
        computed for intervals of length ``|b-a| * 2**(-l)``,
        ``l=0,1,...,M_c``.
    'nnlog'
        A rank-1 integer array of length M(=limit), containing the
        subdivision levels of the subintervals, i.e., an element of this
        array is equal to l if the corresponding subinterval is
        ``|b-a|* 2**(-l)``.
    'chebmo'
        A rank-2 array of shape (25, maxp1) containing the computed
        Chebyshev moments. These can be passed on to an integration
        over the same interval by passing this array as the second
        element of the sequence wopts and passing infodict['momcom'] as
        the first element.

    If one of the integration limits is infinite, then a Fourier integral is
    computed (assuming w neq 0). If full_output is 1 and a numerical error
    is encountered, besides the error message attached to the output tuple,
    a dictionary is also appended to the output tuple which translates the
    error codes in the array ``info['ierlst']`` to English messages. The
    output information dictionary contains the following entries instead of
    'last', 'alist', 'blist', 'rlist', and 'elist':

    'lst'
        The number of subintervals needed for the integration (call it ``K_f``).
    'rslst'
        A rank-1 array of length M_f=limlst, whose first ``K_f`` elements
        contain the integral contribution over the interval
        ``(a+(k-1)c, a+kc)`` where ``c = (2*floor(|w|) + 1) * pi / |w|``
        and ``k=1,2,...,K_f``.
    'erlst'
        A rank-1 array of length ``M_f`` containing the error estimate
        corresponding to the interval in the same position in
        ``infodict['rslist']``.
    'ierlst'
        A rank-1 integer array of length ``M_f`` containing an error flag
        corresponding to the interval in the same position in
        ``infodict['rslist']``.  See the explanation dictionary (last entry
        in the output tuple) for the meaning of the codes.


    **Details of QUADPACK level routines**
    # `quad` calls routines from the FORTRAN library QUADPACK. This section
    # provides details on the conditions for each routine to be called and a
    # short description of each routine. The routine called depends on
    # `weight`, `points` and the integration limits `a` and `b`.
    
    # ================  ==============  ==========  =====================
    # QUADPACK routine  `weight`        `points`    infinite bounds
    # ================  ==============  ==========  =====================
    # qagse             None            No          No
    # qagie             None            No          Yes
    # qagpe             None            Yes         No
    # qawoe             'sin', 'cos'    No          No
    # qawfe             'sin', 'cos'    No          either `a` or `b`
    # qawse             'alg*'          No          No
    # qawce             'cauchy'        No          No
    # ================  ==============  ==========  =====================
    
    # The following provides a short description from [1]_ for each
    # routine.
    
    # qagse
    #     is an integrator based on globally adaptive interval
    #     subdivision in connection with extrapolation, which will
    #     eliminate the effects of integrand singularities of
    #     several types.
    
    # qagie
    #     handles integration over infinite intervals. The infinite range is
    #     mapped onto a finite interval and subsequently the same strategy as
    #     in ``QAGS`` is applied.
    
    # qagpe
    #     serves the same purposes as QAGS, but also allows the
    #     user to provide explicit information about the location
    #     and type of trouble-spots i.e. the abscissae of internal
    #     singularities, discontinuities and other difficulties of
    #     the integrand function.
    
    # qawoe
    #     is an integrator for the evaluation of
    #     :math:`\\int^b_a \\cos(\\omega x)f(x)dx` or
    #     :math:`\\int^b_a \\sin(\\omega x)f(x)dx`
    #     over a finite interval [a,b], where :math:`\\omega` and :math:`f`
    #     are specified by the user. The rule evaluation component is based
    #     on the modified Clenshaw-Curtis technique
    
    #     An adaptive subdivision scheme is used in connection
    #     with an extrapolation procedure, which is a modification
    #     of that in ``QAGS`` and allows the algorithm to deal with
    #     singularities in :math:`f(x)`.
    
    # qawfe
    #     calculates the Fourier transform
    #     :math:`\\int^\\infty_a \\cos(\\omega x)f(x)dx` or
    #     :math:`\\int^\\infty_a \\sin(\\omega x)f(x)dx`
    #     for user-provided :math:`\\omega` and :math:`f`. The procedure of
    #     ``QAWO`` is applied on successive finite intervals, and convergence
    #     acceleration by means of the :math:`\\varepsilon`-algorithm is applied
    #     to the series of integral approximations.
    qawse
        approximate :math:`\\int^b_a w(x)f(x)dx`, with :math:`a < b` where
        :math:`w(x) = (x-a)^{\\alpha}(b-x)^{\\beta}v(x)` with
        :math:`\\alpha,\\beta > -1`, where :math:`v(x)` may be one of the
        following functions: :math:`1`, :math:`\\log(x-a)`, :math:`\\log(b-x)`,
        :math:`\\log(x-a)\\log(b-x)`.

        The user specifies :math:`\\alpha`, :math:`\\beta` and the type of the
        function :math:`v`. A globally adaptive subdivision strategy is
        applied, with modified Clenshaw-Curtis integration on those
        subintervals which contain `a` or `b`.



    qawce
        compute :math:`\\int^b_a f(x) / (x-c)dx` where the integral must be
        interpreted as a Cauchy principal value integral, for user specified
        :math:`c` and :math:`f`. The strategy is globally adaptive. Modified
        Clenshaw-Curtis integration is used on those intervals containing the
        point :math:`x = c`.



    **Integration of Complex Function of a Real Variable**

    A complex valued function, :math:`f`, of a real variable can be written as
    :math:`f = g + ih`.  Similarly, the integral of :math:`f` can be
    written as

    .. math::
        \\int_a^b f(x) dx = \\int_a^b g(x) dx + i\\int_a^b h(x) dx

    assuming that the integrals of :math:`g` and :math:`h` exist
    over the interval :math:`[a,b]` [2]_. Therefore, ``quad`` integrates
    complex-valued functions by integrating the real and imaginary components
    separately.



    References
    ----------

    .. [1] Piessens, Robert; de Doncker-Kapenga, Elise;
           Überhuber, Christoph W.; Kahaner, David (1983).
           QUADPACK: A subroutine package for automatic integration.
           Springer-Verlag.
           ISBN 978-3-540-12553-2.

    .. [2] McCullough, Thomas; Phillips, Keith (1973).
           Foundations of Analysis in the Complex Plane.
           Holt Rinehart Winston.
           ISBN 0-03-086370-8



    Examples
    --------
    Calculate :math:`\\int^4_0 x^2 dx` and compare with an analytic result

    >>> from scipy import integrate
    >>> import numpy as np
    >>> x2 = lambda x: x**2
    >>> integrate.quad(x2, 0, 4)
    (21.333333333333332, 2.3684757858670003e-13)
    >>> print(4**3 / 3.)  # analytical result
    21.3333333333

    Calculate :math:`\\int^\\infty_0 e^{-x} dx`

    >>> invexp = lambda x: np.exp(-x)
    >>> integrate.quad(invexp, 0, np.inf)
    (1.0, 5.842605999138044e-11)

    Calculate :math:`\\int^1_0 a x \\,dx` for :math:`a = 1, 3`

    >>> f = lambda x, a: a*x
    >>> y, err = integrate.quad(f, 0, 1, args=(1,))
    >>> y
    0.5
    >>> y, err = integrate.quad(f, 0, 1, args=(3,))
    >>> y
    1.5



    Calculate :math:`\\int^1_0 x^2 + y^2 dx` with ctypes, holding
    y parameter as 1::

        testlib.c =>
            double func(int n, double args[n]){
                return args[0]*args[0] + args[1]*args[1];}
        compile to library testlib.*
    """
    Import the 'integrate' function from the scipy library for numerical integration.
    Import the 'ctypes' module for interacting with C-compatible libraries.
    Load a dynamic link library (DLL) located at the specified absolute path.
    Ensure 'func' from the loaded library returns a double and accepts an integer and a double as arguments.
    Perform numerical integration using the 'quad' function from scipy.integrate with 'lib.func' over the interval [0, 1].
    Print the difference between the analytical and numerical integration results for 1 - 0.
    Expected result is 1.3333333333333333.

    Be aware of potential inaccuracies when integrating sharp features compared to the interval size using 'quad'.

    >>> Define a lambda function 'y' that returns 1 if x <= 0 else 0.
    >>> Integrate 'y' over the interval [-1, 1] using 'quad'.
    Result: (1.0, 1.1102230246251565e-14)
    >>> Integrate 'y' over the interval [-1, 100] using 'quad'.
    Result: (1.0000000002199108, 1.0189464580163188e-08)
    >>> Integrate 'y' over the interval [-1, 10000] using 'quad'.
    Result: (0.0, 0.0)
    """

    # Ensure 'args' is a tuple even if initially provided as a single element.
    if not isinstance(args, tuple):
        args = (args,)

    # Determine the correct integration limits: ensure 'a' < 'b'.
    flip, a, b = b < a, min(a, b), max(a, b)

    # If 'complex_func' is True, separate 'func' into real and imaginary parts for integration.
    if complex_func:
        def imfunc(x, *args):
            return func(x, *args).imag

        def refunc(x, *args):
            return func(x, *args).real

        # Perform integration on the real and imaginary parts separately.
        re_retval = quad(refunc, a, b, args, full_output, epsabs,
                         epsrel, limit, points, weight, wvar, wopts,
                         maxp1, limlst, complex_func=False)
        im_retval = quad(imfunc, a, b, args, full_output, epsabs,
                         epsrel, limit, points, weight, wvar, wopts,
                         maxp1, limlst, complex_func=False)

        # Combine results into a complex number and error estimate.
        integral = re_retval[0] + 1j * im_retval[0]
        error_estimate = re_retval[1] + 1j * im_retval[1]
        retval = integral, error_estimate

        # Include detailed messages if 'full_output' is True.
        if full_output:
            msgexp = {}
            msgexp["real"] = re_retval[2:]
            msgexp["imag"] = im_retval[2:]
            retval = retval + (msgexp,)

        return retval

    # If 'weight' is not specified, use standard numerical integration.
    if weight is None:
        retval = _quad(func, a, b, args, full_output, epsabs, epsrel, limit,
                       points)
    else:
        # Handle integration with specified weights.
        if points is not None:
            msg = ("Break points cannot be specified when using weighted integrand.\n"
                   "Continuing, ignoring specified points.")
            warnings.warn(msg, IntegrationWarning, stacklevel=2)
        retval = _quad_weight(func, a, b, args, full_output, epsabs, epsrel,
                              limlst, limit, maxp1, weight, wvar, wopts)

    # Adjust result if integration limits were flipped.
    if flip:
        retval = (-retval[0],) + retval[1:]

    # Return integration result excluding the error code if integration was successful.
    ier = retval[-1]
    if ier == 0:
        return retval[:-1]
    # 定义一个字典 msgs，用于存储不同错误代码对应的错误消息
    msgs = {
        80: "A Python error occurred possibly while calling the function.",
        1: f"The maximum number of subdivisions ({limit}) has been achieved.\n  "
           f"If increasing the limit yields no improvement it is advised to "
           f"analyze \n  the integrand in order to determine the difficulties.  "
           f"If the position of a \n  local difficulty can be determined "
           f"(singularity, discontinuity) one will \n  probably gain from "
           f"splitting up the interval and calling the integrator \n  on the "
           f"subranges.  Perhaps a special-purpose integrator should be used.",
        2: "The occurrence of roundoff error is detected, which prevents \n  "
           "the requested tolerance from being achieved.  "
           "The error may be \n  underestimated.",
        3: "Extremely bad integrand behavior occurs at some points of the\n  "
           "integration interval.",
        4: "The algorithm does not converge.  Roundoff error is detected\n  "
           "in the extrapolation table.  It is assumed that the requested "
           "tolerance\n  cannot be achieved, and that the returned result "
           "(if full_output = 1) is \n  the best which can be obtained.",
        5: "The integral is probably divergent, or slowly convergent.",
        6: "The input is invalid.",
        7: "Abnormal termination of the routine.  The estimates for result\n  "
           "and error are less reliable.  It is assumed that the requested "
           "accuracy\n  has not been achieved.",
        'unknown': "Unknown error."
    }
    # 检查权重是否在['cos', 'sin']中，并且上下限a和b是否为无穷大
    if weight in ['cos','sin'] and (b == np.inf or a == -np.inf):
        # 将不同的错误消息分配给msgs字典中的不同索引
        msgs[1] = (
            "The maximum number of cycles allowed has been achieved., e.e.\n  of "
            "subintervals (a+(k-1)c, a+kc) where c = (2*int(abs(omega)+1))\n  "
            "*pi/abs(omega), for k = 1, 2, ..., lst.  "
            "One can allow more cycles by increasing the value of limlst.  "
            "Look at info['ierlst'] with full_output=1."
        )
        msgs[4] = (
            "The extrapolation table constructed for convergence acceleration\n  of "
            "the series formed by the integral contributions over the cycles, \n  does "
            "not converge to within the requested accuracy.  "
            "Look at \n  info['ierlst'] with full_output=1."
        )
        msgs[7] = (
            "Bad integrand behavior occurs within one or more of the cycles.\n  "
            "Location and type of the difficulty involved can be determined from \n  "
            "the vector info['ierlist'] obtained with full_output=1."
        )
        # 解释不同错误代码对应的含义
        explain = {1: "The maximum number of subdivisions (= limit) has been \n  "
                      "achieved on this cycle.",
                   2: "The occurrence of roundoff error is detected and prevents\n  "
                      "the tolerance imposed on this cycle from being achieved.",
                   3: "Extremely bad integrand behavior occurs at some points of\n  "
                      "this cycle.",
                   4: "The integral over this cycle does not converge (to within the "
                      "required accuracy) due to roundoff in the extrapolation "
                      "procedure invoked on this cycle.  It is assumed that the result "
                      "on this interval is the best which can be obtained.",
                   5: "The integral over this cycle is probably divergent or "
                      "slowly convergent."}

    try:
        # 尝试获取与错误代码对应的消息
        msg = msgs[ier]
    except KeyError:
        # 如果错误代码未定义，则使用默认的未知消息
        msg = msgs['unknown']

    # 如果错误代码在[1,2,3,4,5,7]中
    if ier in [1,2,3,4,5,7]:
        # 如果需要完整的输出
        if full_output:
            # 如果权重在['cos', 'sin']中，并且上下限a和b为无穷大
            if weight in ['cos', 'sin'] and (b == np.inf or a == -np.inf):
                # 返回结果除最后一个元素外的所有内容，附加错误消息和解释
                return retval[:-1] + (msg, explain)
            else:
                # 返回结果除最后一个元素外的所有内容，附加错误消息
                return retval[:-1] + (msg,)
        else:
            # 如果不需要完整输出，发出警告，并返回结果除最后一个元素外的所有内容
            warnings.warn(msg, IntegrationWarning, stacklevel=2)
            return retval[:-1]
    elif ier == 6:  # 当 QUADPACK 抛出 ier=6 时的取证决策树
        if epsabs <= 0:  # 较小的误差容限 - 适用于所有方法
            if epsrel < max(50 * sys.float_info.epsilon, 5e-29):
                msg = ("If 'epsabs'<=0, 'epsrel' must be greater than both"
                       " 5e-29 and 50*(machine epsilon).")
            elif weight in ['sin', 'cos'] and (abs(a) + abs(b) == np.inf):
                msg = ("Sine or cosine weighted integrals with infinite domain"
                       " must have 'epsabs'>0.")

        elif weight is None:
            if points is None:  # 对于 QAGSE/QAGIE
                msg = ("Invalid 'limit' argument. There must be"
                       " at least one subinterval")
            else:  # 对于 QAGPE
                if not (min(a, b) <= min(points) <= max(points) <= max(a, b)):
                    msg = ("All break points in 'points' must lie within the"
                           " integration limits.")
                elif len(points) >= limit:
                    msg = (f"Number of break points ({len(points):d}) "
                           f"must be less than subinterval limit ({limit:d})")

        else:
            if maxp1 < 1:
                msg = "Chebyshev moment limit maxp1 must be >=1."

            elif weight in ('cos', 'sin') and abs(a+b) == np.inf:  # 对于 QAWFE
                msg = "Cycle limit limlst must be >=3."

            elif weight.startswith('alg'):  # 对于 QAWSE
                if min(wvar) < -1:
                    msg = "wvar parameters (alpha, beta) must both be >= -1."
                if b < a:
                    msg = "Integration limits a, b must satistfy a<b."

            elif weight == 'cauchy' and wvar in (a, b):
                msg = ("Parameter 'wvar' must not equal"
                       " integration limits 'a' or 'b'.")
    
    raise ValueError(msg)


注释：
# 定义一个函数 `_quad`，用于数值积分，采用了 adaptive quadrature 方法
def _quad(func, a, b, args, full_output, epsabs, epsrel, limit, points):
    # 初始化变量 infbounds 为 0
    infbounds = 0
    
    # 根据 a 和 b 的值判断积分区间类型
    if (b != np.inf and a != -np.inf):
        pass   # 标准积分区间
    elif (b == np.inf and a != -np.inf):
        infbounds = 1
        bound = a   # 设置边界为 a
    elif (b == np.inf and a == -np.inf):
        infbounds = 2
        bound = 0   # 忽略此情况下的边界设置
    elif (b != np.inf and a == -np.inf):
        infbounds = -1
        bound = b   # 设置边界为 b
    else:
        # 抛出运行时错误，如果无法处理无穷的比较
        raise RuntimeError("Infinity comparisons don't work for you.")
    
    # 如果 points 为 None
    if points is None:
        # 根据 infbounds 的值选择合适的积分函数进行计算
        if infbounds == 0:
            return _quadpack._qagse(func, a, b, args, full_output, epsabs, epsrel, limit)
        else:
            return _quadpack._qagie(func, bound, infbounds, args, full_output,
                                    epsabs, epsrel, limit)
    else:
        # 如果 infbounds 不为 0，则不能使用分段积分
        if infbounds != 0:
            raise ValueError("Infinity inputs cannot be used with break points.")
        else:
            # 在特定点强制执行函数的计算
            the_points = np.unique(points)
            the_points = the_points[a < the_points]
            the_points = the_points[the_points < b]
            the_points = np.concatenate((the_points, (0., 0.)))
            return _quadpack._qagpe(func, a, b, the_points, args, full_output,
                                    epsabs, epsrel, limit)


# 定义一个函数 `_quad_weight`，用于带权重的数值积分
def _quad_weight(func, a, b, args, full_output, epsabs, epsrel,
                 limlst, limit, maxp1, weight, wvar, wopts):
    # 如果 weight 参数不是预定义的几种权重函数，则抛出值错误
    if weight not in ['cos', 'sin', 'alg', 'alg-loga', 'alg-logb', 'alg-log', 'cauchy']:
        raise ValueError("%s not a recognized weighting function." % weight)
    
    # 根据 weight 的值选择相应的权重函数编号
    strdict = {'cos': 1, 'sin': 2, 'alg': 1, 'alg-loga': 2, 'alg-logb': 3, 'alg-log': 4}
    # 检查 weight 是否在 ['cos', 'sin'] 中
    if weight in ['cos','sin']:
        # 根据 weight 获取对应的积分方法字符串
        integr = strdict[weight]
        # 如果上下限 a 和 b 都不是正无穷或负无穷，则为有限积分区间
        if (b != np.inf and a != -np.inf):  # finite limits
            # 如果没有预先计算的切比雪夫矩，使用 _qawoe 函数进行积分
            if wopts is None:         # no precomputed Chebyshev moments
                return _quadpack._qawoe(func, a, b, wvar, integr, args, full_output,
                                        epsabs, epsrel, limit, maxp1, 1)
            else:                     # 如果有预先计算的切比雪夫矩，使用 _qawoe 函数进行积分
                momcom = wopts[0]
                chebcom = wopts[1]
                return _quadpack._qawoe(func, a, b, wvar, integr, args,
                                        full_output, epsabs, epsrel, limit, maxp1, 2,
                                        momcom, chebcom)

        # 如果上限 b 是正无穷而下限 a 不是负无穷，则为 (a, +∞) 区间积分
        elif (b == np.inf and a != -np.inf):
            return _quadpack._qawfe(func, a, wvar, integr, args, full_output,
                                    epsabs, limlst, limit, maxp1)
        # 如果上限 b 不是正无穷而下限 a 是负无穷，则需要重映射函数和区间
        elif (b != np.inf and a == -np.inf):  # remap function and interval
            # 根据 weight 的值选择不同的重映射函数
            if weight == 'cos':
                # 定义重映射函数 thefunc(x, *myargs)，将 x 映射为 -x
                def thefunc(x, *myargs):
                    y = -x
                    func = myargs[0]
                    myargs = (y,) + myargs[1:]
                    return func(*myargs)
            else:
                # 定义重映射函数 thefunc(x, *myargs)，将 x 映射为 -x，并对 func 返回值取负
                def thefunc(x, *myargs):
                    y = -x
                    func = myargs[0]
                    myargs = (y,) + myargs[1:]
                    return -func(*myargs)
            # 将 func 添加到参数列表中
            args = (func,) + args
            # 使用重映射后的函数进行积分，区间为 (-b, +∞)
            return _quadpack._qawfe(thefunc, -b, wvar, integr, args,
                                    full_output, epsabs, limlst, limit, maxp1)
        else:
            # 如果不满足以上条件，则抛出异常
            raise ValueError("Cannot integrate with this weight from -Inf to +Inf.")
    else:
        # 如果 weight 不在 ['cos', 'sin'] 中，则需要检查是否存在无穷限制
        if a in [-np.inf, np.inf] or b in [-np.inf, np.inf]:
            message = "Cannot integrate with this weight over an infinite interval."
            # 抛出异常，说明不能在无穷区间上进行积分
            raise ValueError(message)

        # 如果 weight 以 'alg' 开头，则使用 _qawse 函数进行积分
        if weight.startswith('alg'):
            integr = strdict[weight]
            return _quadpack._qawse(func, a, b, wvar, integr, args,
                                    full_output, epsabs, epsrel, limit)
        else:  # 如果 weight 是 'cauchy'，则使用 _qawce 函数进行积分
            return _quadpack._qawce(func, a, b, wvar, args, full_output,
                                    epsabs, epsrel, limit)
# 定义函数 dblquad，用于计算二重积分

"""
Compute a double integral.

Return the double (definite) integral of ``func(y, x)`` from ``x = a..b``
and ``y = gfun(x)..hfun(x)``.

Parameters
----------
func : callable
    A Python function or method of at least two variables: y must be the
    first argument and x the second argument.
a, b : float
    The limits of integration in x: `a` < `b`
gfun : callable or float
    The lower boundary curve in y which is a function taking a single
    floating point argument (x) and returning a floating point result
    or a float indicating a constant boundary curve.
hfun : callable or float
    The upper boundary curve in y (same requirements as `gfun`).
args : sequence, optional
    Extra arguments to pass to `func`.
epsabs : float, optional
    Absolute tolerance passed directly to the inner 1-D quadrature
    integration. Default is 1.49e-8. ``dblquad`` tries to obtain
    an accuracy of ``abs(i-result) <= max(epsabs, epsrel*abs(i))``
    where ``i`` = inner integral of ``func(y, x)`` from ``gfun(x)``
    to ``hfun(x)``, and ``result`` is the numerical approximation.
    See `epsrel` below.
epsrel : float, optional
    Relative tolerance of the inner 1-D integrals. Default is 1.49e-8.
    If ``epsabs <= 0``, `epsrel` must be greater than both 5e-29
    and ``50 * (machine epsilon)``. See `epsabs` above.

Returns
-------
y : float
    The resultant integral.
abserr : float
    An estimate of the error.

See Also
--------
quad : single integral
tplquad : triple integral
nquad : N-dimensional integrals
fixed_quad : fixed-order Gaussian quadrature
simpson : integrator for sampled data
romb : integrator for sampled data
scipy.special : for coefficients and roots of orthogonal polynomials

Notes
-----
For valid results, the integral must converge; behavior for divergent
integrals is not guaranteed.

**Details of QUADPACK level routines**

`quad` calls routines from the FORTRAN library QUADPACK. This section
provides details on the conditions for each routine to be called and a
short description of each routine. For each level of integration, ``qagse``
is used for finite limits or ``qagie`` is used if either limit (or both!)
are infinite. The following provides a short description from [1]_ for each
routine.

qagse
    is an integrator based on globally adaptive interval
    subdivision in connection with extrapolation, which will
    eliminate the effects of integrand singularities of
    several types.
qagie
    handles integration over infinite intervals. The infinite range is
    mapped onto a finite interval and subsequently the same strategy as
    in ``QAGS`` is applied.

References
----------

"""
    # 定义一个函数 temp_ranges，用于生成函数的积分范围列表
    def temp_ranges(*args):
        # 如果 gfun 是可调用的，则使用 gfun(args[0])，否则直接使用 gfun
        return [gfun(args[0]) if callable(gfun) else gfun,
                # 如果 hfun 是可调用的，则使用 hfun(args[0])，否则直接使用 hfun
                hfun(args[0]) if callable(hfun) else hfun]
    
    # 返回 nquad 函数的结果，用于计算多重积分
    return nquad(func, [temp_ranges, [a, b]], args=args,
            # 设置积分的绝对误差和相对误差
            opts={"epsabs": epsabs, "epsrel": epsrel})
# 定义一个函数，计算三重定积分。
def tplquad(func, a, b, gfun, hfun, qfun, rfun, args=(), epsabs=1.49e-8,
            epsrel=1.49e-8):
    """
    Compute a triple (definite) integral.

    Return the triple integral of ``func(z, y, x)`` from ``x = a..b``,
    ``y = gfun(x)..hfun(x)``, and ``z = qfun(x,y)..rfun(x,y)``.

    Parameters
    ----------
    func : function
        A Python function or method of at least three variables in the
        order (z, y, x).
    a, b : float
        The limits of integration in x: `a` < `b`
    gfun : function or float
        The lower boundary curve in y which is a function taking a single
        floating point argument (x) and returning a floating point result
        or a float indicating a constant boundary curve.
    hfun : function or float
        The upper boundary curve in y (same requirements as `gfun`).
    qfun : function or float
        The lower boundary surface in z.  It must be a function that takes
        two floats in the order (x, y) and returns a float or a float
        indicating a constant boundary surface.
    rfun : function or float
        The upper boundary surface in z. (Same requirements as `qfun`.)
    args : tuple, optional
        Extra arguments to pass to `func`.
    epsabs : float, optional
        Absolute tolerance passed directly to the innermost 1-D quadrature
        integration. Default is 1.49e-8.
    epsrel : float, optional
        Relative tolerance of the innermost 1-D integrals. Default is 1.49e-8.

    Returns
    -------
    y : float
        The resultant integral.
    abserr : float
        An estimate of the error.

    See Also
    --------
    quad : Adaptive quadrature using QUADPACK
    fixed_quad : Fixed-order Gaussian quadrature
    dblquad : Double integrals
    nquad : N-dimensional integrals
    romb : Integrators for sampled data
    simpson : Integrators for sampled data
    scipy.special : For coefficients and roots of orthogonal polynomials

    Notes
    -----
    For valid results, the integral must converge; behavior for divergent
    integrals is not guaranteed.

    **Details of QUADPACK level routines**

    `quad` calls routines from the FORTRAN library QUADPACK. This section
    provides details on the conditions for each routine to be called and a
    short description of each routine. For each level of integration, ``qagse``
    is used for finite limits or ``qagie`` is used, if either limit (or both!)
    are infinite. The following provides a short description from [1]_ for each
    routine.

    qagse
        is an integrator based on globally adaptive interval
        subdivision in connection with extrapolation, which will
        eliminate the effects of integrand singularities of
        several types.
    qagie
        handles integration over infinite intervals. The infinite range is
        mapped onto a finite interval and subsequently the same strategy as
        in ``QAGS`` is applied.

    References
    ----------

    """
    """
    # f(z, y, x)
    # qfun/rfun(x, y)
    # gfun/hfun(x)
    # nquad will hand (y, x, t0, ...) to ranges0
    # nquad will hand (x, t0, ...) to ranges1
    # Only qfun / rfun is different API...
    
    定义一个函数 ranges0，接受任意数量的参数 *args，并返回一个列表，
    其中第一个元素是 qfun(args[1], args[0]) 或者 qfun，取决于 qfun 是否可调用，
    第二个元素是 rfun(args[1], args[0]) 或者 rfun，取决于 rfun 是否可调用。
    
    定义一个函数 ranges1，接受任意数量的参数 *args，并返回一个列表，
    其中第一个元素是 gfun(args[0]) 或者 gfun，取决于 gfun 是否可调用，
    第二个元素是 hfun(args[0]) 或者 hfun，取决于 hfun 是否可调用。
    
    ranges 是一个包含 ranges0、ranges1 和列表 [a, b] 的列表。
    
    返回 nquad 的结果，调用 func 函数，使用 ranges 和 args 作为参数，
    并传递 opts={"epsabs": epsabs, "epsrel": epsrel} 作为选项。
    """
    def ranges0(*args):
        return [qfun(args[1], args[0]) if callable(qfun) else qfun,
                rfun(args[1], args[0]) if callable(rfun) else rfun]
    
    def ranges1(*args):
        return [gfun(args[0]) if callable(gfun) else gfun,
                hfun(args[0]) if callable(hfun) else hfun]
    
    ranges = [ranges0, ranges1, [a, b]]
    return nquad(func, ranges, args=args,
            opts={"epsabs": epsabs, "epsrel": epsrel})
# 定义一个函数 nquad，用于多变量积分
def nquad(func, ranges, args=None, opts=None, full_output=False):
    r"""
    Integration over multiple variables.

    Wraps `quad` to enable integration over multiple variables.
    Various options allow improved integration of discontinuous functions, as
    well as the use of weighted integration, and generally finer control of the
    integration process.

    Parameters
    ----------
    func : {callable, scipy.LowLevelCallable}
        The function to be integrated. Has arguments of ``x0, ... xn``,
        ``t0, ... tm``, where integration is carried out over ``x0, ... xn``,
        which must be floats.  Where ``t0, ... tm`` are extra arguments
        passed in args.
        Function signature should be ``func(x0, x1, ..., xn, t0, t1, ..., tm)``.
        Integration is carried out in order.  That is, integration over ``x0``
        is the innermost integral, and ``xn`` is the outermost.

        If the user desires improved integration performance, then `f` may
        be a `scipy.LowLevelCallable` with one of the signatures::

            double func(int n, double *xx)
            double func(int n, double *xx, void *user_data)

        where ``n`` is the number of variables and args.  The ``xx`` array
        contains the coordinates and extra arguments. ``user_data`` is the data
        contained in the `scipy.LowLevelCallable`.
    ranges : iterable object
        Each element of ranges may be either a sequence  of 2 numbers, or else
        a callable that returns such a sequence. ``ranges[0]`` corresponds to
        integration over x0, and so on. If an element of ranges is a callable,
        then it will be called with all of the integration arguments available,
        as well as any parametric arguments. e.g., if
        ``func = f(x0, x1, x2, t0, t1)``, then ``ranges[0]`` may be defined as
        either ``(a, b)`` or else as ``(a, b) = range0(x1, x2, t0, t1)``.
    args : iterable object, optional
        Additional arguments ``t0, ... tn``, required by ``func``, ``ranges``,
        and ``opts``.
    opts : iterable object or dict, optional
        Options to be passed to `quad`. May be empty, a dict, or
        a sequence of dicts or functions that return a dict. If empty, the
        default options from scipy.integrate.quad are used. If a dict, the same
        options are used for all levels of integraion. If a sequence, then each
        element of the sequence corresponds to a particular integration. e.g.,
        ``opts[0]`` corresponds to integration over ``x0``, and so on. If a
        callable, the signature must be the same as for ``ranges``. The
        available options together with their default values are:

          - epsabs = 1.49e-08
          - epsrel = 1.49e-08
          - limit  = 50
          - points = None
          - weight = None
          - wvar   = None
          - wopts  = None

        For more information on these options, see `quad`.
    full_output : bool, optional
        If True, return a tuple with the final integration value and
        information about the integrator.

    """
    full_output : bool, optional
        是否输出完整的结果。这是从 scipy.integrate.quad 中部分实现的 `full_output` 参数。
        当调用 nquad 时，设置 `full_output=True` 可以获取积分函数评估的次数 `neval`。

    Returns
    -------
    result : float
        积分的结果，即数值积分的值。
    abserr : float
        各种积分结果的绝对误差估计的最大值。
    out_dict : dict, optional
        包含积分额外信息的字典，可选返回。

    See Also
    --------
    quad : 一维数值积分
    dblquad, tplquad : 二重和三重积分
    fixed_quad : 固定阶数的高斯积分

    Notes
    -----
    对于有效的结果，积分必须收敛；对于发散的积分，不保证其行为。

    **QUADPACK 级别例程的详细信息**

    `nquad` 调用来自 FORTRAN 库 QUADPACK 的例程。本部分提供每个例程被调用的条件及其简要描述。
    所调用的例程取决于 `weight`、`points` 和积分限制 `a` 和 `b`。

    ================  ==============  ==========  =====================
    QUADPACK 例程      `weight`        `points`    无限边界
    ================  ==============  ==========  =====================
    qagse             None            否          否
    qagie             None            否          是
    qagpe             None            是          否
    qawoe             'sin', 'cos'    否          否
    qawfe             'sin', 'cos'    否          `a` 或 `b` 之一
    qawse             'alg*'          否          否
    qawce             'cauchy'        否          否
    ================  ==============  ==========  =====================

    下面从 [1]_ 提供了每个例程的简要描述。

    qagse
        基于全局自适应区间分割和外推的积分器，可消除多种类型的积分奇异性的影响。
    qagie
        处理无限区间上的积分。将无限范围映射到有限区间，然后应用与 "QAGS" 相同的策略。
    qagpe
        与 QAGS 相同，但允许用户提供关于麻烦点（例如内部奇异点、不连续点和积分函数其他困难处）的位置和类型的显式信息。
    qawoe
        is an integrator for the evaluation of
        :math:`\int^b_a \cos(\omega x)f(x)dx` or
        :math:`\int^b_a \sin(\omega x)f(x)dx`
        over a finite interval [a,b], where :math:`\omega` and :math:`f`
        are specified by the user. The rule evaluation component is based
        on the modified Clenshaw-Curtis technique

        An adaptive subdivision scheme is used in connection
        with an extrapolation procedure, which is a modification
        of that in ``QAGS`` and allows the algorithm to deal with
        singularities in :math:`f(x)`.


    qawfe
        calculates the Fourier transform
        :math:`\int^\infty_a \cos(\omega x)f(x)dx` or
        :math:`\int^\infty_a \sin(\omega x)f(x)dx`
        for user-provided :math:`\omega` and :math:`f`. The procedure of
        ``QAWO`` is applied on successive finite intervals, and convergence
        acceleration by means of the :math:`\varepsilon`-algorithm is applied
        to the series of integral approximations.


    qawse
        approximate :math:`\int^b_a w(x)f(x)dx`, with :math:`a < b` where
        :math:`w(x) = (x-a)^{\alpha}(b-x)^{\beta}v(x)` with
        :math:`\alpha,\beta > -1`, where :math:`v(x)` may be one of the
        following functions: :math:`1`, :math:`\log(x-a)`, :math:`\log(b-x)`,
        :math:`\log(x-a)\log(b-x)`.

        The user specifies :math:`\alpha`, :math:`\beta` and the type of the
        function :math:`v`. A globally adaptive subdivision strategy is
        applied, with modified Clenshaw-Curtis integration on those
        subintervals which contain `a` or `b`.


    qawce
        compute :math:`\int^b_a f(x) / (x-c)dx` where the integral must be
        interpreted as a Cauchy principal value integral, for user specified
        :math:`c` and :math:`f`. The strategy is globally adaptive. Modified
        Clenshaw-Curtis integration is used on those intervals containing the
        point :math:`x = c`.
    # 根据给定的函数和积分范围进行多重积分计算，并返回详细输出
    >>> integrate.nquad(func, [[0,1], [-1,1], [.13,.8], [-.15,1]],
    ...                 opts=[opts0,{},{},{}], full_output=True)

    Calculate

    .. math::

        \int^{t_0+t_1+1}_{t_0+t_1-1}
        \int^{x_2+t_0^2 t_1^3+1}_{x_2+t_0^2 t_1^3-1}
        \int^{t_0 x_1+t_1 x_2+1}_{t_0 x_1+t_1 x_2-1}
        f(x_0,x_1, x_2,t_0,t_1)
        \,dx_0 \,dx_1 \,dx_2,

    where

    .. math::

        f(x_0, x_1, x_2, t_0, t_1) = \begin{cases}
          x_0 x_2^2 + \sin{x_1}+2 & (x_0+t_1 x_1-t_0 > 0) \\
          x_0 x_2^2 +\sin{x_1}+1 & (x_0+t_1 x_1-t_0 \leq 0)
        \end{cases}

    and :math:`(t_0, t_1) = (0, 1)` .

    # 定义一个新的函数 func2，接受五个参数并返回计算结果
    >>> def func2(x0, x1, x2, t0, t1):
    ...     return x0*x2**2 + np.sin(x1) + 1 + (1 if x0+t1*x1-t0>0 else 0)
    
    # 定义 lim0 函数，返回 x1 和 x2 的限制范围列表
    >>> def lim0(x1, x2, t0, t1):
    ...     return [t0*x1 + t1*x2 - 1, t0*x1 + t1*x2 + 1]
    
    # 定义 lim1 函数，返回 x2 的限制范围列表
    >>> def lim1(x2, t0, t1):
    ...     return [x2 + t0**2*t1**3 - 1, x2 + t0**2*t1**3 + 1]
    
    # 定义 lim2 函数，返回 t0 和 t1 的限制范围列表
    >>> def lim2(t0, t1):
    ...     return [t0 + t1 - 1, t0 + t1 + 1]
    
    # 定义 opts0 函数，返回带有 'points' 键的选项字典
    >>> def opts0(x1, x2, t0, t1):
    ...     return {'points' : [t0 - t1*x1]}
    
    # 定义 opts1 函数，返回空字典作为选项
    >>> def opts1(x2, t0, t1):
    ...     return {}
    
    # 定义 opts2 函数，返回空字典作为选项
    >>> def opts2(t0, t1):
    ...     return {}
    
    # 使用 _NQuad 类执行多重积分计算并返回结果
    return _NQuad(func, ranges, opts, full_output).integrate(*args)
class _RangeFunc:
    def __init__(self, range_):
        # 初始化函数，存储传入的范围值
        self.range_ = range_

    def __call__(self, *args):
        """Return stored value.

        *args needed because range_ can be float or func, and is called with
        variable number of parameters.
        """
        # 调用对象时返回存储的范围值
        return self.range_


class _OptFunc:
    def __init__(self, opt):
        # 初始化函数，存储传入的选项字典
        self.opt = opt

    def __call__(self, *args):
        """Return stored dict."""
        # 调用对象时返回存储的选项字典
        return self.opt


class _NQuad:
    def __init__(self, func, ranges, opts, full_output):
        # 初始化函数
        self.abserr = 0
        self.func = func
        self.ranges = ranges
        self.opts = opts
        self.maxdepth = len(ranges)
        self.full_output = full_output
        if self.full_output:
            # 如果需要完整输出，则初始化输出字典，记录评估次数
            self.out_dict = {'neval': 0}

    def integrate(self, *args, **kwargs):
        # 多维积分函数
        depth = kwargs.pop('depth', 0)
        if kwargs:
            raise ValueError('unexpected kwargs')

        # 获取当前深度的积分范围和选项
        ind = -(depth + 1)
        fn_range = self.ranges[ind]
        low, high = fn_range(*args)
        fn_opt = self.opts[ind]
        opt = dict(fn_opt(*args))

        if 'points' in opt:
            # 如果选项中有 'points'，则筛选出落在积分范围内的点
            opt['points'] = [x for x in opt['points'] if low <= x <= high]
        if depth + 1 == self.maxdepth:
            f = self.func
        else:
            f = partial(self.integrate, depth=depth+1)
        # 执行积分计算
        quad_r = quad(f, low, high, args=args, full_output=self.full_output,
                      **opt)
        value = quad_r[0]
        abserr = quad_r[1]
        if self.full_output:
            infodict = quad_r[2]
            # 如果需要完整输出，更新评估次数
            if depth + 1 == self.maxdepth:
                self.out_dict['neval'] += infodict['neval']
        self.abserr = max(self.abserr, abserr)
        if depth > 0:
            return value
        else:
            # 返回多维积分的最终结果及误差
            if self.full_output:
                return value, self.abserr, self.out_dict
            else:
                return value, self.abserr
```