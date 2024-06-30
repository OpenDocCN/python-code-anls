# `D:\src\scipysrc\scipy\scipy\odr\_odrpack.py`

```
"""
Python wrappers for Orthogonal Distance Regression (ODRPACK).

Notes
=====

* Array formats -- FORTRAN stores its arrays in memory column first, i.e., an
  array element A(i, j, k) will be next to A(i+1, j, k). In C and, consequently,
  NumPy, arrays are stored row first: A[i, j, k] is next to A[i, j, k+1]. For
  efficiency and convenience, the input and output arrays of the fitting
  function (and its Jacobians) are passed to FORTRAN without transposition.
  Therefore, where the ODRPACK documentation says that the X array is of shape
  (N, M), it will be passed to the Python function as an array of shape (M, N).
  If M==1, the 1-D case, then nothing matters; if M>1, then your
  Python functions will be dealing with arrays that are indexed in reverse of
  the ODRPACK documentation. No real issue, but watch out for your indexing of
  the Jacobians: the i,jth elements (@f_i/@x_j) evaluated at the nth
  observation will be returned as jacd[j, i, n]. Except for the Jacobians, it
  really is easier to deal with x[0] and x[1] than x[:,0] and x[:,1]. Of course,
  you can always use the transpose() function from SciPy explicitly.

* Examples -- See the accompanying file test/test.py for examples of how to set
  up fits of your own. Some are taken from the User's Guide; some are from
  other sources.

* Models -- Some common models are instantiated in the accompanying module
  models.py . Contributions are welcome.

Credits
=======

* Thanks to Arnold Moene and Gerard Vermeulen for fixing some killer bugs.

Robert Kern
robert.kern@gmail.com

"""

import os  # 导入标准库 os

import numpy as np  # 导入第三方库 numpy
from warnings import warn  # 从标准库 warnings 中导入 warn 函数
from scipy.odr import __odrpack  # 从 scipy.odr 模块导入 __odrpack

__all__ = ['odr', 'OdrWarning', 'OdrError', 'OdrStop',
           'Data', 'RealData', 'Model', 'Output', 'ODR',
           'odr_error', 'odr_stop']

odr = __odrpack.odr  # 设置变量 odr 为 __odrpack 模块的 odr 属性


class OdrWarning(UserWarning):
    """
    Warning indicating that the data passed into
    ODR will cause problems when passed into 'odr'
    that the user should be aware of.
    """
    pass


class OdrError(Exception):
    """
    Exception indicating an error in fitting.

    This is raised by `~scipy.odr.odr` if an error occurs during fitting.
    """
    pass


class OdrStop(Exception):
    """
    Exception stopping fitting.

    You can raise this exception in your objective function to tell
    `~scipy.odr.odr` to stop fitting.
    """
    pass


# Backwards compatibility
odr_error = OdrError  # 设置 odr_error 为 OdrError 类
odr_stop = OdrStop  # 设置 odr_stop 为 OdrStop 类

__odrpack._set_exceptions(OdrError, OdrStop)  # 调用 __odrpack 模块的 _set_exceptions 函数，并传入 OdrError 和 OdrStop 类作为参数


def _conv(obj, dtype=None):
    """ Convert an object to the preferred form for input to the odr routine.
    """
    if obj is None:
        return obj  # 如果 obj 为 None，则返回 obj 本身
    else:
        if dtype is None:
            obj = np.asarray(obj)  # 将 obj 转换为 numpy 数组
        else:
            obj = np.asarray(obj, dtype)  # 将 obj 转换为指定 dtype 的 numpy 数组
        if obj.shape == ():
            # Scalar.
            return obj.dtype.type(obj)  # 返回标量对象的类型
        else:
            return obj  # 返回转换后的 numpy 数组


def _report_error(info):
    """ Interprets the return code of the odr routine.
    """
    Parameters
    ----------
    info : int
        odr例程的返回代码。

    Returns
    -------
    problems : list(str)
        包含odr()例程停止原因的消息列表。
    """

    # 根据info的值确定停止原因
    stopreason = ('空白',
                  '平方和收敛',
                  '参数收敛',
                  '平方和和参数均收敛',
                  '达到迭代限制')[info % 5]

    if info >= 5:
        # 可能有问题的结果或者致命错误

        # 解析info以获取问题指示符I
        I = (info//10000 % 10,
             info//1000 % 10,
             info//100 % 10,
             info//10 % 10,
             info % 10)
        problems = []

        # 根据I的值判断具体的问题类型并添加到problems列表中
        if I[0] == 0:
            if I[1] != 0:
                problems.append('导数可能不正确')
            if I[2] != 0:
                problems.append('回调函数中发生错误')
            if I[3] != 0:
                problems.append('解处问题不是满秩的')
            problems.append(stopreason)
        elif I[0] == 1:
            if I[1] != 0:
                problems.append('N < 1')
            if I[2] != 0:
                problems.append('M < 1')
            if I[3] != 0:
                problems.append('NP < 1 或者 NP > N')
            if I[4] != 0:
                problems.append('NQ < 1')
        elif I[0] == 2:
            if I[1] != 0:
                problems.append('LDY 和/或 LDX 不正确')
            if I[2] != 0:
                problems.append('LDWE, LD2WE, LDWD 和/或 LD2WD 不正确')
            if I[3] != 0:
                problems.append('LDIFX, LDSTPD 和/或 LDSCLD 不正确')
            if I[4] != 0:
                problems.append('LWORK 和/或 LIWORK 太小')
        elif I[0] == 3:
            if I[1] != 0:
                problems.append('STPB 和/或 STPD 不正确')
            if I[2] != 0:
                problems.append('SCLB 和/或 SCLD 不正确')
            if I[3] != 0:
                problems.append('WE 不正确')
            if I[4] != 0:
                problems.append('WD 不正确')
        elif I[0] == 4:
            problems.append('导数计算错误')
        elif I[0] == 5:
            problems.append('回调函数中发生错误')
        elif I[0] == 6:
            problems.append('检测到数值错误')

        return problems

    else:
        return [stopreason]
class Data:
    """
    The data to fit.

    Parameters
    ----------
    x : array_like
        Observed data for the independent variable of the regression
    y : array_like, optional
        If array-like, observed data for the dependent variable of the
        regression. A scalar input implies that the model to be used on
        the data is implicit.
    we : array_like, optional
        Weighting matrix for observations of the response variable.
        - If scalar: Uniform weight for all observations.
        - If 1D array (length q): Diagonal of covariant weighting matrix.
        - If 1D array (length n): Weight for each observation's response.
        - If 2D array (shape q x q): Full covariant weighting matrix.
        - If 2D array (shape q x n): Diagonal covariant matrix for each observation.
        - If 3D array (shape q x q x n): Full covariant matrix for each observation.
        For implicit fit, a positive scalar is used.
    wd : array_like, optional
        Weighting matrix for observations of the input variable.
        - If scalar: Uniform weight for all observations.
        - If 1D array (length m): Diagonal of covariant weighting matrix.
        - If 1D array (length n): Weight for each observation's input.
        - If 2D array (shape m x m): Full covariant weighting matrix.
        - If 2D array (shape m x n): Diagonal covariant matrix for each observation.
        - If 3D array (shape m x m x n): Full covariant matrix for each observation.
        If wd = 0, identity matrix is used.
    """
    """
    fix : array_like of ints, optional
        The `fix` argument is an array that specifies which observations in `x`
        are fixed during the fitting process. A value of 0 fixes an observation,
        while a positive value makes it free to vary.
    meta : dict, optional
        A dictionary used to store arbitrary metadata associated with the data.

    Notes
    -----
    Each argument (`x`, `y`, `we`, `wd`, `fix`, `meta`) corresponds to an attribute
    of the instance of the same name.
    The structures of `x` and `y` are described in the docstring of the `Model`
    class.
    If `y` is an integer, the `Data` instance is restricted to fitting implicit
    models where the response dimensionality matches the specified integer `y`.

    The `we` argument weights the influence of deviations in the response variable
    on the fit, while `wd` weights deviations in the input variable. These arguments
    are structured with the n'th dimensional axis first to facilitate handling of
    multidimensional inputs and responses. They leverage the structured arguments
    feature of ODRPACK for flexibility in fitting options.
    See the ODRPACK User's Guide for a detailed explanation of how these weights
    affect the fitting algorithm. Generally, higher weights indicate more significant
    impact of deviations at corresponding data points on the fit.

    """

    def __init__(self, x, y=None, we=None, wd=None, fix=None, meta=None):
        """
        Initialize a `Data` instance with input data and optional parameters.

        Parameters
        ----------
        x : array_like
            Input data to be fitted.
        y : array_like or None, optional
            Response data to be fitted. If `y` is None, it implies fitting with
            implicit models where the dimensionality of the response matches the
            data.
        we : array_like or None, optional
            Weighting for response variable deviations in the fitting process.
        wd : array_like or None, optional
            Weighting for input variable deviations in the fitting process.
        fix : array_like of ints or None, optional
            Array specifying which observations in `x` are fixed during the fitting.
        meta : dict or None, optional
            Dictionary to store additional metadata associated with the data.

        Raises
        ------
        ValueError
            If `x` is not an `ndarray`.

        """
        # Convert input data to appropriate format
        self.x = _conv(x)

        # Check if `x` is not a numpy ndarray
        if not isinstance(self.x, np.ndarray):
            raise ValueError("Expected an 'ndarray' of data for 'x', "
                             f"but instead got data of type '{type(self.x).__name__}'")

        # Convert `y`, `we`, `wd`, `fix` to appropriate formats
        self.y = _conv(y)
        self.we = _conv(we)
        self.wd = _conv(wd)
        self.fix = _conv(fix)

        # Initialize metadata dictionary; if `meta` is None, initialize as empty dict
        self.meta = {} if meta is None else meta

    def set_meta(self, **kwds):
        """
        Update the metadata dictionary with the provided keywords and associated data.

        Parameters
        ----------
        **kwds
            Arbitrary keyword arguments where each key-value pair updates the metadata.

        Examples
        --------
        ::

            data.set_meta(lab="Ph 7; Lab 26", title="Ag110 + Ag108 Decay")

        """
        self.meta.update(kwds)

    def __getattr__(self, attr):
        """
        Dispatch attribute access to the metadata dictionary.

        Parameters
        ----------
        attr : str
            Attribute name to access.

        Returns
        -------
        object
            Value associated with the attribute in the metadata.

        Raises
        ------
        AttributeError
            If the requested attribute `attr` is not found in the metadata.

        """
        if attr in self.meta:
            return self.meta[attr]
        else:
            raise AttributeError("'%s' not in metadata" % attr)
# 定义一个名为 RealData 的类，继承自 Data 类
class RealData(Data):
    """
    The data, with weightings as actual standard deviations and/or
    covariances.

    Parameters
    ----------
    x : array_like
        Observed data for the independent variable of the regression
    y : array_like, optional
        If array-like, observed data for the dependent variable of the
        regression. A scalar input implies that the model to be used on
        the data is implicit.
    sx : array_like, optional
        Standard deviations of `x`.
        `sx` are standard deviations of `x` and are converted to weights by
        dividing 1.0 by their squares.
    sy : array_like, optional
        Standard deviations of `y`.
        `sy` are standard deviations of `y` and are converted to weights by
        dividing 1.0 by their squares.
    covx : array_like, optional
        Covariance of `x`
        `covx` is an array of covariance matrices of `x` and are converted to
        weights by performing a matrix inversion on each observation's
        covariance matrix.
    covy : array_like, optional
        Covariance of `y`
        `covy` is an array of covariance matrices and are converted to
        weights by performing a matrix inversion on each observation's
        covariance matrix.
    fix : array_like, optional
        The argument and member fix is the same as Data.fix and ODR.ifixx:
        It is an array of integers with the same shape as `x` that
        determines which input observations are treated as fixed. One can
        use a sequence of length m (the dimensionality of the input
        observations) to fix some dimensions for all observations. A value
        of 0 fixes the observation, a value > 0 makes it free.
    meta : dict, optional
        Free-form dictionary for metadata.

    Notes
    -----
    The weights `wd` and `we` are computed from provided values as follows:

    `sx` and `sy` are converted to weights by dividing 1.0 by their squares.
    For example, ``wd = 1./np.power(`sx`, 2)``.

    `covx` and `covy` are arrays of covariance matrices and are converted to
    weights by performing a matrix inversion on each observation's covariance
    matrix. For example, ``we[i] = np.linalg.inv(covy[i])``.

    These arguments follow the same structured argument conventions as wd and
    we only restricted by their natures: `sx` and `sy` can't be rank-3, but
    `covx` and `covy` can be.

    Only set *either* `sx` or `covx` (not both). Setting both will raise an
    exception. Same with `sy` and `covy`.

    """
    def __init__(self, x, y=None, sx=None, sy=None, covx=None, covy=None,
                 fix=None, meta=None):
        # 检查输入参数，确保不同时设置了 sx 和 covx
        if (sx is not None) and (covx is not None):
            raise ValueError("cannot set both sx and covx")
        # 检查输入参数，确保不同时设置了 sy 和 covy
        if (sy is not None) and (covy is not None):
            raise ValueError("cannot set both sy and covy")

        # 设置用于 __getattr__ 的标志位
        self._ga_flags = {}
        # 根据输入的参数设置权重标志位
        if sx is not None:
            self._ga_flags['wd'] = 'sx'
        else:
            self._ga_flags['wd'] = 'covx'
        if sy is not None:
            self._ga_flags['we'] = 'sy'
        else:
            self._ga_flags['we'] = 'covy'

        # 转换并存储输入的 x 值
        self.x = _conv(x)

        # 如果 x 不是 ndarray 类型，则抛出异常
        if not isinstance(self.x, np.ndarray):
            raise ValueError("Expected an 'ndarray' of data for 'x', "
                              f"but instead got data of type '{type(self.x).__name__}'")

        # 转换并存储输入的 y 值
        self.y = _conv(y)
        # 转换并存储输入的 sx 值
        self.sx = _conv(sx)
        # 转换并存储输入的 sy 值
        self.sy = _conv(sy)
        # 转换并存储输入的 covx 值
        self.covx = _conv(covx)
        # 转换并存储输入的 covy 值
        self.covy = _conv(covy)
        # 转换并存储输入的 fix 值
        self.fix = _conv(fix)
        # 初始化元数据字典，如果未提供则为空字典
        self.meta = {} if meta is None else meta

    def _sd2wt(self, sd):
        """ Convert standard deviation to weights.
        """
        # 将标准差转换为权重
        return 1./np.power(sd, 2)

    def _cov2wt(self, cov):
        """ Convert covariance matrix(-ices) to weights.
        """
        # 导入 inv 函数用于矩阵求逆
        from scipy.linalg import inv

        # 如果矩阵是二维的，则直接求逆
        if len(cov.shape) == 2:
            return inv(cov)
        else:
            # 否则初始化一个全零数组用于存储每个矩阵的逆
            weights = np.zeros(cov.shape, float)

            # 对于每个矩阵，求其逆并存储到 weights 中
            for i in range(cov.shape[-1]):  # n
                weights[:,:,i] = inv(cov[:,:,i])

            return weights

    def __getattr__(self, attr):
        # 定义属性与处理函数的查找表
        lookup_tbl = {('wd', 'sx'): (self._sd2wt, self.sx),
                      ('wd', 'covx'): (self._cov2wt, self.covx),
                      ('we', 'sy'): (self._sd2wt, self.sy),
                      ('we', 'covy'): (self._cov2wt, self.covy)}

        # 如果请求的属性不在 ('wd', 'we') 中，则检查是否在元数据中，否则抛出异常
        if attr not in ('wd', 'we'):
            if attr in self.meta:
                return self.meta[attr]
            else:
                raise AttributeError("'%s' not in metadata" % attr)
        else:
            # 根据属性查找对应的处理函数和参数
            func, arg = lookup_tbl[(attr, self._ga_flags[attr])]

            # 如果参数不为 None，则调用处理函数并返回结果，否则返回 None
            if arg is not None:
                return func(*(arg,))
            else:
                return None
class Model:
    """
    The Model class stores information about the function you wish to fit.

    It stores the function itself, at the least, and optionally stores
    functions which compute the Jacobians used during fitting. Also, one
    can provide a function that will provide reasonable starting values
    for the fit parameters possibly given the set of data.

    Parameters
    ----------
    fcn : function
          fcn(beta, x) --> y
          主要函数，接受参数 beta 和 x，返回 y。

    fjacb : function
          Jacobian of fcn wrt the fit parameters beta.
          对于参数 beta 的 Jacobian 矩阵函数。

          fjacb(beta, x) --> @f_i(x,B)/@B_j

    fjacd : function
          Jacobian of fcn wrt the (possibly multidimensional) input
          variable.
          对于输入变量 x 的 Jacobian 矩阵函数。

          fjacd(beta, x) --> @f_i(x,B)/@x_j

    extra_args : tuple, optional
          If specified, `extra_args` should be a tuple of extra
          arguments to pass to `fcn`, `fjacb`, and `fjacd`. Each will be called
          by `apply(fcn, (beta, x) + extra_args)`
          可选的额外参数元组，传递给 `fcn`, `fjacb`, 和 `fjacd`。

    estimate : array_like of rank-1
          Provides estimates of the fit parameters from the data
          根据数据提供的拟合参数估计。

          estimate(data) --> estbeta

    implicit : boolean
          If TRUE, specifies that the model
          is implicit; i.e `fcn(beta, x)` ~= 0 and there is no y data to fit
          against
          是否是隐式模型的布尔值。如果为 TRUE，表示模型是隐式的，即 `fcn(beta, x)` ~ 0，
          并且没有 y 数据进行拟合。

    meta : dict, optional
          freeform dictionary of metadata for the model
          模型的元数据字典，可选项。

    Notes
    -----
    Note that the `fcn`, `fjacb`, and `fjacd` operate on NumPy arrays and
    return a NumPy array. The `estimate` object takes an instance of the
    Data class.
    注意 `fcn`, `fjacb`, 和 `fjacd` 都操作 NumPy 数组并返回 NumPy 数组。
    `estimate` 对象接受 Data 类的实例。

    Here are the rules for the shapes of the argument and return
    arrays of the callback functions:

    `x`
        if the input data is single-dimensional, then `x` is rank-1
        array; i.e., ``x = array([1, 2, 3, ...]); x.shape = (n,)``
        If the input data is multi-dimensional, then `x` is a rank-2 array;
        i.e., ``x = array([[1, 2, ...], [2, 4, ...]]); x.shape = (m, n)``.
        In all cases, it has the same shape as the input data array passed to
        `~scipy.odr.odr`. `m` is the dimensionality of the input data,
        `n` is the number of observations.
        输入数据 x 的形状规则描述。

    `y`
        if the response variable is single-dimensional, then `y` is a
        rank-1 array, i.e., ``y = array([2, 4, ...]); y.shape = (n,)``.
        If the response variable is multi-dimensional, then `y` is a rank-2
        array, i.e., ``y = array([[2, 4, ...], [3, 6, ...]]); y.shape =
        (q, n)`` where `q` is the dimensionality of the response variable.
        输出数据 y 的形状规则描述。

    `beta`
        rank-1 array of length `p` where `p` is the number of parameters;
        i.e. ``beta = array([B_1, B_2, ..., B_p])``
        参数 beta 的形状规则描述。

    `fjacb`
        if the response variable is multi-dimensional, then the
        return array's shape is ``(q, p, n)`` such that ``fjacb(x,beta)[l,k,i] =
        d f_l(X,B)/d B_k`` evaluated at the ith data point.  If ``q == 1``, then
        the return array is only rank-2 and with shape ``(p, n)``.
        Jacobian 矩阵函数 fjacb 返回的数组形状规则描述。
    """
    """
    fjacd
        与 fjacb 类似，但返回数组的形状为 ``(q, m, n)``
        这里 ``fjacd(x,beta)[l,j,i] = d f_l(X,B)/d X_j`` 在第 i 个数据点上。
        如果 ``q == 1``，则返回数组的形状为 ``(m, n)``。
        如果 ``m == 1``，形状为 ``(q, n)``。
        如果 `m == q == 1`，形状为 ``(n,)``。
    """

    def __init__(self, fcn, fjacb=None, fjacd=None,
                 extra_args=None, estimate=None, implicit=0, meta=None):
        """
        初始化函数对象。

        Parameters
        ----------
        fcn : callable
            主函数对象，用于计算目标函数。
        fjacb : callable, optional
            Jacobian 函数对象，计算目标函数的雅可比矩阵。
        fjacd : callable, optional
            另一个类型的 Jacobian 函数对象，计算目标函数的雅可比矩阵。
        extra_args : tuple, optional
            额外的参数元组，传递给函数对象。
        estimate : object, optional
            估计对象，用于估计优化问题的初始点。
        implicit : int, optional
            隐式标志，指示是否处理隐式函数。
        meta : dict, optional
            元数据字典，用于存储额外的元信息。

        Notes
        -----
        若提供了额外参数 `extra_args`，则将其转换为元组形式。
        """
        
        self.fcn = fcn
        self.fjacb = fjacb
        self.fjacd = fjacd

        if extra_args is not None:
            extra_args = tuple(extra_args)

        self.extra_args = extra_args
        self.estimate = estimate
        self.implicit = implicit
        self.meta = meta if meta is not None else {}

    def set_meta(self, **kwds):
        """ 
        更新元数据字典，使用提供的关键字和数据。

        Examples
        --------
        set_meta(name="Exponential", equation="y = a exp(b x) + c")
        """
        self.meta.update(kwds)

    def __getattr__(self, attr):
        """ 
        将属性访问分派给元数据字典。

        Parameters
        ----------
        attr : str
            属性名称。

        Returns
        -------
        object
            元数据字典中与属性名称匹配的值。

        Raises
        ------
        AttributeError
            如果属性名称在元数据字典中不存在。

        """
        if attr in self.meta:
            return self.meta[attr]
        else:
            raise AttributeError("'%s' not in metadata" % attr)
class Output:
    """
    The Output class stores the output of an ODR run.

    Attributes
    ----------
    beta : ndarray
        Estimated parameter values, of shape (q,).
    sd_beta : ndarray
        Standard deviations of the estimated parameters, of shape (p,).
    cov_beta : ndarray
        Covariance matrix of the estimated parameters, of shape (p,p).
        Note that this `cov_beta` is not scaled by the residual variance 
        `res_var`, whereas `sd_beta` is. This means 
        ``np.sqrt(np.diag(output.cov_beta * output.res_var))`` is the same 
        result as `output.sd_beta`.
    delta : ndarray, optional
        Array of estimated errors in input variables, of same shape as `x`.
    eps : ndarray, optional
        Array of estimated errors in response variables, of same shape as `y`.
    xplus : ndarray, optional
        Array of ``x + delta``.
    y : ndarray, optional
        Array ``y = fcn(x + delta)``.
    res_var : float, optional
        Residual variance.
    sum_square : float, optional
        Sum of squares error.
    sum_square_delta : float, optional
        Sum of squares of delta error.
    sum_square_eps : float, optional
        Sum of squares of eps error.
    inv_condnum : float, optional
        Inverse condition number (cf. ODRPACK UG p. 77).
    rel_error : float, optional
        Relative error in function values computed within fcn.
    work : ndarray, optional
        Final work array.
    work_ind : dict, optional
        Indices into work for drawing out values (cf. ODRPACK UG p. 83).
    info : int, optional
        Reason for returning, as output by ODRPACK (cf. ODRPACK UG p. 38).
    stopreason : list of str, optional
        `info` interpreted into English.

    Notes
    -----
    Takes one argument for initialization, the return value from the
    function `~scipy.odr.odr`. The attributes listed as "optional" above are
    only present if `~scipy.odr.odr` was run with ``full_output=1``.

    """

    def __init__(self, output):
        # Assign estimated parameter values from ODR output
        self.beta = output[0]
        # Assign standard deviations of estimated parameters from ODR output
        self.sd_beta = output[1]
        # Assign covariance matrix of estimated parameters from ODR output
        self.cov_beta = output[2]

        # Check if full output was provided
        if len(output) == 4:
            # If full output, update object attributes with additional output details
            self.__dict__.update(output[3])
            # Generate human-readable stop reasons based on 'info'
            self.stopreason = _report_error(self.info)

    def pprint(self):
        """ Pretty-print important results.
        """
        # Print estimated parameter values
        print('Beta:', self.beta)
        # Print standard errors of estimated parameters
        print('Beta Std Error:', self.sd_beta)
        # Print covariance matrix of estimated parameters
        print('Beta Covariance:', self.cov_beta)
        
        # Check if 'info' attribute exists (indicating full output)
        if hasattr(self, 'info'):
            # Print residual variance
            print('Residual Variance:', self.res_var)
            # Print inverse condition number
            print('Inverse Condition #:', self.inv_condnum)
            # Print reasons for halting
            print('Reason(s) for Halting:')
            for r in self.stopreason:
                print('  %s' % r)


class ODR:
    """
    The ODR class gathers all information and coordinates the running of the
    main fitting routine.

    Members of instances of the ODR class have the same names as the arguments
    to the initialization routine.

    Parameters
    """
    # 数据：Data 类的实例
    #     Data 类的一个实例对象
    data : Data class instance
    # 模型：Model 类的实例
    #     Model 类的一个实例对象
    model : Model class instance

    # 其他参数
    # ----------------
    # beta0 : rank-1 的 array_like
    #     初始参数值的一维序列。如果 model 提供了 "estimate" 函数来估计这些值，则可选。
    beta0 : array_like of rank-1
    # delta0 : rank-1 浮点数的 array_like，可选
    #     用于保存输入变量误差初始值的双精度浮点数数组。必须与 data.x 的形状相同。
    delta0 : array_like of floats of rank-1, optional
    # ifixb : rank-1 的 int 数组，可选
    #     与 beta0 长度相同的整数序列，确定哪些参数被固定。值为 0 表示固定参数，大于 0 表示自由参数。
    ifixb : array_like of ints of rank-1, optional
    # ifixx : shape 与 data.x 相同的 int 数组，可选
    #     一个与 data.x 形状相同的整数数组，确定哪些输入观测被视为固定。可以使用长度为 m（输入观测的维度）的序列来固定所有观测的某些维度。
    #     值为 0 表示固定观测，大于 0 表示自由观测。
    ifixx : array_like of ints with same shape as data.x, optional
    # job : int，可选
    #     告诉 ODRPACK 要执行的任务的整数。如果必须在此处设置值，请参阅 ODRPACK 用户指南第 31 页。建议使用初始化后的 set_job 方法以获得更可读的界面。
    job : int, optional
    # iprint : int，可选
    #     告诉 ODRPACK 要打印什么信息的整数。如果必须在此处设置值，请参阅 ODRPACK 用户指南第 33-34 页。建议使用初始化后的 set_iprint 方法以获得更可读的界面。
    iprint : int, optional
    # errfile : str，可选
    #     打印 ODRPACK 错误信息的文件名字符串。如果文件已经存在，将抛出错误。可以使用 overwrite 参数来防止这种情况。*不要自行打开此文件！*
    errfile : str, optional
    # rptfile : str，可选
    #     打印 ODRPACK 摘要信息的文件名字符串。如果文件已经存在，将抛出错误。可以使用 overwrite 参数来防止这种情况。*不要自行打开此文件！*
    rptfile : str, optional
    # ndigit : int，可选
    #     指定计算函数中可靠数字的数量的整数。默认值是计算机上双精度计算的最小值 eps 的平方根，其中 eps 是使得 1 + eps > 1 的最小值。
    ndigit : int, optional
    # taufac : float，可选
    #     指定初始信任区域的浮点数。默认值为 1。初始信任区域等于 taufac 乘以第一个计算的 Gauss-Newton 步长的长度。taufac 必须小于 1。
    taufac : float, optional
    # sstol : float，可选
    #     指定基于残差平方和相对变化收敛的公差。默认值为 eps**(1/2)，其中 eps 是使得计算机上双精度计算满足 1 + eps > 1 的最小值。sstol 必须小于 1。
    sstol : float, optional
    partol : float, optional
        float specifying the tolerance for convergence based on the relative
        change in the estimated parameters. The default value is eps**(2/3) for
        explicit models and ``eps**(1/3)`` for implicit models. partol must be less
        than 1.
    maxit : int, optional
        integer specifying the maximum number of iterations to perform. For
        first runs, maxit is the total number of iterations performed and
        defaults to 50. For restarts, maxit is the number of additional
        iterations to perform and defaults to 10.
    stpb : array_like, optional
        sequence (``len(stpb) == len(beta0)``) of relative step sizes to compute
        finite difference derivatives wrt the parameters.
    stpd : optional
        array (``stpd.shape == data.x.shape`` or ``stpd.shape == (m,)``) of relative
        step sizes to compute finite difference derivatives wrt the input
        variable errors. If stpd is a rank-1 array with length m (the
        dimensionality of the input variable), then the values are broadcast to
        all observations.
    sclb : array_like, optional
        sequence (``len(stpb) == len(beta0)``) of scaling factors for the
        parameters. The purpose of these scaling factors are to scale all of
        the parameters to around unity. Normally appropriate scaling factors
        are computed if this argument is not specified. Specify them yourself
        if the automatic procedure goes awry.
    scld : array_like, optional
        array (scld.shape == data.x.shape or scld.shape == (m,)) of scaling
        factors for the *errors* in the input variables. Again, these factors
        are automatically computed if you do not provide them. If scld.shape ==
        (m,), then the scaling factors are broadcast to all observations.
    work : ndarray, optional
        array to hold the double-valued working data for ODRPACK. When
        restarting, takes the value of self.output.work.
    iwork : ndarray, optional
        array to hold the integer-valued working data for ODRPACK. When
        restarting, takes the value of self.output.iwork.
    overwrite : bool, optional
        If it is True, output files defined by `errfile` and `rptfile` are
        overwritten. The default is False.

    Attributes
    ----------
    data : Data
        The data for this fit
    model : Model
        The model used in fit
    output : Output
        An instance if the Output class containing all of the returned
        data from an invocation of ODR.run() or ODR.restart()

    """
    # 初始化函数，接受多个参数并进行初始化操作
    def __init__(self, data, model, beta0=None, delta0=None, ifixb=None,
        ifixx=None, job=None, iprint=None, errfile=None, rptfile=None,
        ndigit=None, taufac=None, sstol=None, partol=None, maxit=None,
        stpb=None, stpd=None, sclb=None, scld=None, work=None, iwork=None,
        overwrite=False):

        # 将传入的数据和模型赋值给对象的属性
        self.data = data
        self.model = model

        # 如果未提供 beta0，则根据模型的估计值进行初始化
        if beta0 is None:
            if self.model.estimate is not None:
                self.beta0 = _conv(self.model.estimate(self.data))
            else:
                # 如果模型没有提供估计方法，则抛出数值错误
                raise ValueError(
                  "must specify beta0 or provide an estimator with the model"
                )
        else:
            # 使用提供的 beta0 值进行初始化
            self.beta0 = _conv(beta0)

        # 如果未提供 ifixx 并且数据对象包含 fix 属性，则使用其值
        if ifixx is None and data.fix is not None:
            ifixx = data.fix

        # 如果 overwrite 参数为 True，则删除指定的输出文件
        if overwrite:
            # 如果提供了报告文件名并且文件存在，则删除
            if rptfile is not None and os.path.exists(rptfile):
                os.remove(rptfile)
            # 如果提供了错误文件名并且文件存在，则删除
            if errfile is not None and os.path.exists(errfile):
                os.remove(errfile)

        # 使用 _conv 函数将 delta0 转换后赋值给对象的属性
        self.delta0 = _conv(delta0)
        
        # 根据 FORTRAN 的规范将 ifixx 和 ifixb 转换为 32 位整数
        # 在一些 FORTRAN 编译器上可能需要特别注意
        self.ifixx = _conv(ifixx, dtype=np.int32)
        self.ifixb = _conv(ifixb, dtype=np.int32)

        # 将传入的 job、iprint、errfile、rptfile、ndigit、taufac、sstol、partol、
        # maxit、stpb、stpd、sclb、scld、work、iwork 转换后赋值给对象的属性
        self.job = job
        self.iprint = iprint
        self.errfile = errfile
        self.rptfile = rptfile
        self.ndigit = ndigit
        self.taufac = taufac
        self.sstol = sstol
        self.partol = partol
        self.maxit = maxit
        self.stpb = _conv(stpb)
        self.stpd = _conv(stpd)
        self.sclb = _conv(sclb)
        self.scld = _conv(scld)
        self.work = _conv(work)
        self.iwork = _conv(iwork)

        # 初始化一个输出属性为 None
        self.output = None

        # 调用对象的内部方法进行检查操作
        self._check()
    # 定义一个方法 _check，用于检查输入数据的一致性，但不检查内置函数 odr 将会检查的内容。
    def _check(self):
        """ Check the inputs for consistency, but don't bother checking things
        that the builtin function odr will check.
        """
        
        # 将 self.data.x 的形状转换为列表并赋值给 x_s
        x_s = list(self.data.x.shape)

        # 如果 self.data.y 是一个 numpy 数组
        if isinstance(self.data.y, np.ndarray):
            # 将 self.data.y 的形状转换为列表并赋值给 y_s
            y_s = list(self.data.y.shape)
            # 如果模型是隐式模型，则抛出 OdrError 异常
            if self.model.implicit:
                raise OdrError("an implicit model cannot use response data")
        else:
            # 对于没有 numpy 数组的情况，如果模型是隐式模型
            y_s = [self.data.y, x_s[-1]]
            # 如果模型不是隐式模型，则抛出 OdrError 异常
            if not self.model.implicit:
                raise OdrError("an explicit model needs response data")
            # 设置作业类型为 1
            self.set_job(fit_type=1)

        # 检查观测数量是否匹配
        if x_s[-1] != y_s[-1]:
            raise OdrError("number of observations do not match")

        # n 是观测数量
        n = x_s[-1]

        # 如果 x_s 的长度为 2，则 m 是 x_s 的第一个元素，否则为 1
        if len(x_s) == 2:
            m = x_s[0]
        else:
            m = 1
        # 如果 y_s 的长度为 2，则 q 是 y_s 的第一个元素，否则为 1
        if len(y_s) == 2:
            q = y_s[0]
        else:
            q = 1

        # p 是 self.beta0 的长度
        p = len(self.beta0)

        # 允许的输出数组形状列表
        fcn_perms = [(q, n)]
        fjacd_perms = [(q, m, n)]
        fjacb_perms = [(q, p, n)]

        # 根据 q, m, p 的不同值，可能会添加额外的输出数组形状
        if q == 1:
            fcn_perms.append((n,))
            fjacd_perms.append((m, n))
            fjacb_perms.append((p, n))
        if m == 1:
            fjacd_perms.append((q, n))
        if p == 1:
            fjacb_perms.append((q, n))
        if m == q == 1:
            fjacd_perms.append((n,))
        if p == q == 1:
            fjacb_perms.append((n,))

        # 尝试评估提供的函数以确保它们提供合理的输出

        # 参数列表包括 self.beta0 和 self.data.x
        arglist = (self.beta0, self.data.x)
        # 如果模型有额外的参数，则将其添加到参数列表中
        if self.model.extra_args is not None:
            arglist = arglist + self.model.extra_args
        # 调用 self.model.fcn 函数，并将结果保存在 res 变量中
        res = self.model.fcn(*arglist)

        # 检查 self.model.fcn 的输出形状是否在允许的形状列表中
        if res.shape not in fcn_perms:
            print(res.shape)
            print(fcn_perms)
            raise OdrError("fcn does not output %s-shaped array" % y_s)

        # 如果 self.model.fjacd 不为空
        if self.model.fjacd is not None:
            # 调用 self.model.fjacd 函数，并将结果保存在 res 变量中
            res = self.model.fjacd(*arglist)
            # 检查 self.model.fjacd 的输出形状是否在允许的形状列表中
            if res.shape not in fjacd_perms:
                raise OdrError(
                    "fjacd does not output %s-shaped array" % repr((q, m, n)))
        # 如果 self.model.fjacb 不为空
        if self.model.fjacb is not None:
            # 调用 self.model.fjacb 函数，并将结果保存在 res 变量中
            res = self.model.fjacb(*arglist)
            # 检查 self.model.fjacb 的输出形状是否在允许的形状列表中
            if res.shape not in fjacb_perms:
                raise OdrError(
                    "fjacb does not output %s-shaped array" % repr((q, p, n)))

        # 检查 delta0 的形状是否与 self.data.x 的形状相匹配
        if self.delta0 is not None and self.delta0.shape != self.data.x.shape:
            raise OdrError(
                "delta0 is not a %s-shaped array" % repr(self.data.x.shape))

        # 如果 self.data.x 的大小为 0，则发出警告
        if self.data.x.size == 0:
            warn("Empty data detected for ODR instance. "
                 "Do not expect any fitting to occur",
                 OdrWarning, stacklevel=3)
    def _gen_work(self):
        """ Generate a suitable work array if one does not already exist.
        """
        # 获取数据向量 x 的最后一个维度大小
        n = self.data.x.shape[-1]
        # 获取 beta0 数组的第一个维度大小
        p = self.beta0.shape[0]

        # 检查数据向量 x 的维度是否为二维
        if len(self.data.x.shape) == 2:
            m = self.data.x.shape[0]
        else:
            m = 1

        # 根据模型是否隐式指定确定 q 的值
        if self.model.implicit:
            q = self.data.y
        elif len(self.data.y.shape) == 2:
            q = self.data.y.shape[0]
        else:
            q = 1

        # 根据数据中权重矩阵 we 的情况确定 ldwe 和 ld2we 的值
        if self.data.we is None:
            ldwe = ld2we = 1
        elif len(self.data.we.shape) == 3:
            ld2we, ldwe = self.data.we.shape[1:]
        else:
            we = self.data.we
            ldwe = 1
            ld2we = 1
            # 根据权重矩阵 we 的维度和 q 的值调整 ldwe 和 ld2we
            if we.ndim == 1 and q == 1:
                ldwe = n
            elif we.ndim == 2:
                if we.shape == (q, q):
                    ld2we = q
                elif we.shape == (q, n):
                    ldwe = n

        # 根据作业号 job 的最后一位判断使用 ODR 还是 OLS 的计算公式
        if self.job % 10 < 2:
            # ODR 情况下的工作数组长度计算
            lwork = (18 + 11*p + p*p + m + m*m + 4*n*q + 6*n*m + 2*n*q*p +
                     2*n*q*m + q*q + 5*q + q*(p+m) + ldwe*ld2we*q)
        else:
            # OLS 情况下的工作数组长度计算
            lwork = (18 + 11*p + p*p + m + m*m + 4*n*q + 2*n*m + 2*n*q*p +
                     5*q + q*(p+m) + ldwe*ld2we*q)

        # 检查现有的工作数组是否已存在且符合要求，如果是则直接返回
        if isinstance(self.work, np.ndarray) and self.work.shape == (lwork,)\
                and self.work.dtype.str.endswith('f8'):
            # 现有的数组已经适用
            return
        else:
            # 创建一个新的 float 类型的工作数组
            self.work = np.zeros((lwork,), float)
    def set_job(self, fit_type=None, deriv=None, var_calc=None,
        del_init=None, restart=None):
        """
        设置 "job" 参数，希望能够理解。

        如果参数未指定，则保持其原值。类初始化时，默认所有选项均为 0。

        Parameters
        ----------
        fit_type : {0, 1, 2} int
            0 -> 显式 ODR

            1 -> 隐式 ODR

            2 -> 普通最小二乘法
        deriv : {0, 1, 2, 3} int
            0 -> 前向有限差分

            1 -> 中心有限差分

            2 -> 用户提供的导数（雅可比矩阵），并由 ODRPACK 检查结果

            3 -> 用户提供的导数，不进行检查
        var_calc : {0, 1, 2} int
            0 -> 计算渐近协方差矩阵和拟合参数的不确定性（V_B, s_B），使用在最终解处重新计算的导数

            1 -> 使用最后一次迭代的导数计算 V_B 和 s_B

            2 -> 不计算 V_B 和 s_B
        del_init : {0, 1} int
            0 -> 初始输入变量偏移设为 0

            1 -> 用户在变量 "work" 中提供初始偏移
        restart : {0, 1} int
            0 -> 拟合不是重新启动

            1 -> 拟合是重新启动

        Notes
        -----
        允许的值与 ODRPACK 用户指南第31页上给出的值不同，唯一的区别在于不能指定大于每个变量的最后一个值。

        如果没有提供计算雅可比矩阵的函数，拟合过程将把 deriv 设置为 0，即默认使用有限差分。要自己初始化输入变量偏移，请将 del_init 设置为 1，并正确地将偏移放入 "work" 变量中。

        """

        # 如果 self.job 为 None，则初始化为 [0, 0, 0, 0, 0]
        if self.job is None:
            job_l = [0, 0, 0, 0, 0]
        else:
            # 否则解析 self.job 中的每个位数到列表 job_l 中
            job_l = [self.job // 10000 % 10,
                     self.job // 1000 % 10,
                     self.job // 100 % 10,
                     self.job // 10 % 10,
                     self.job % 10]

        # 根据参数设置 job_l 中对应位置的值
        if fit_type in (0, 1, 2):
            job_l[4] = fit_type
        if deriv in (0, 1, 2, 3):
            job_l[3] = deriv
        if var_calc in (0, 1, 2):
            job_l[2] = var_calc
        if del_init in (0, 1):
            job_l[1] = del_init
        if restart in (0, 1):
            job_l[0] = restart

        # 将 job_l 转换回整数形式，并设置为 self.job
        self.job = (job_l[0]*10000 + job_l[1]*1000 +
                    job_l[2]*100 + job_l[3]*10 + job_l[4])
    def run(self):
        """ Run the fitting routine with all of the information given and with ``full_output=1``.

        Returns
        -------
        output : Output instance
            This object is also assigned to the attribute .output .
        """  # noqa: E501

        # 准备参数元组和关键字参数字典
        args = (self.model.fcn, self.beta0, self.data.y, self.data.x)
        kwds = {'full_output': 1}
        kwd_l = ['ifixx', 'ifixb', 'job', 'iprint', 'errfile', 'rptfile',
                 'ndigit', 'taufac', 'sstol', 'partol', 'maxit', 'stpb',
                 'stpd', 'sclb', 'scld', 'work', 'iwork']

        # 如果有提供 delta0 并且 fit 不是重新启动
        if self.delta0 is not None and (self.job // 10000) % 10 == 0:
            # 生成工作参数
            self._gen_work()

            # 将 delta0 扁平化并存入工作参数数组
            d0 = np.ravel(self.delta0)
            self.work[:len(d0)] = d0

        # 从其他对象显式设置关键字参数
        if self.model.fjacb is not None:
            kwds['fjacb'] = self.model.fjacb
        if self.model.fjacd is not None:
            kwds['fjacd'] = self.model.fjacd
        if self.data.we is not None:
            kwds['we'] = self.data.we
        if self.data.wd is not None:
            kwds['wd'] = self.data.wd
        if self.model.extra_args is not None:
            kwds['extra_args'] = self.model.extra_args

        # 从 self 的成员隐式设置关键字参数
        for attr in kwd_l:
            obj = getattr(self, attr)
            if obj is not None:
                kwds[attr] = obj

        # 运行 ODR 拟合，并将结果赋给 self.output
        self.output = Output(odr(*args, **kwds))

        # 返回拟合结果对象
        return self.output

    def restart(self, iter=None):
        """ Restarts the run with iter more iterations.

        Parameters
        ----------
        iter : int, optional
            ODRPACK's default for the number of new iterations is 10.

        Returns
        -------
        output : Output instance
            This object is also assigned to the attribute .output .
        """

        # 如果 self.output 为 None，则无法重新启动
        if self.output is None:
            raise OdrError("cannot restart: run() has not been called before")

        # 设置为重新启动模式
        self.set_job(restart=1)

        # 将 self.output 的工作和工作参数赋给当前对象
        self.work = self.output.work
        self.iwork = self.output.iwork

        # 设置最大迭代次数为 iter
        self.maxit = iter

        # 运行拟合并返回结果
        return self.run()
```