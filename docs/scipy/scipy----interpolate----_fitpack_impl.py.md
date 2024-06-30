# `D:\src\scipysrc\scipy\scipy\interpolate\_fitpack_impl.py`

```
"""
fitpack (dierckx in netlib) --- A Python-C wrapper to FITPACK (by P. Dierckx).
        FITPACK is a collection of FORTRAN programs for curve and surface
        fitting with splines and tensor product splines.

See
 https://web.archive.org/web/20010524124604/http://www.cs.kuleuven.ac.be:80/cwis/research/nalag/research/topics/fitpack.html
or
 http://www.netlib.org/dierckx/

Copyright 2002 Pearu Peterson all rights reserved,
Pearu Peterson <pearu@cens.ioc.ee>
Permission to use, modify, and distribute this software is given under the
terms of the SciPy (BSD style) license. See LICENSE.txt that came with
this distribution for specifics.

NO WARRANTY IS EXPRESSED OR IMPLIED.  USE AT YOUR OWN RISK.

TODO: Make interfaces to the following fitpack functions:
    For univariate splines: cocosp, concon, fourco, insert
    For bivariate splines: profil, regrid, parsur, surev
"""

__all__ = ['splrep', 'splprep', 'splev', 'splint', 'sproot', 'spalde',
           'bisplrep', 'bisplev', 'insert', 'splder', 'splantider']

import warnings              # 导入警告模块
import numpy as np           # 导入 NumPy 库，并使用 np 别名
from . import _fitpack        # 导入当前目录下的 _fitpack 模块
from numpy import (atleast_1d, array, ones, zeros, sqrt, ravel, transpose,
                   empty, iinfo, asarray)   # 从 NumPy 中导入多个函数和类

# Try to replace _fitpack interface with
#  f2py-generated version
from . import _dfitpack as dfitpack   # 导入当前目录下的 _dfitpack 模块，并使用 dfitpack 别名


dfitpack_int = dfitpack.types.intvar.dtype   # 设置 dfitpack_int 为 dfitpack 模块中 intvar 的数据类型


def _int_overflow(x, exception, msg=None):
    """Cast the value to an dfitpack_int and raise an OverflowError if the value
    cannot fit.
    """
    if x > iinfo(dfitpack_int).max:   # 如果 x 大于 dfitpack_int 的最大值
        if msg is None:
            msg = f'{x!r} cannot fit into an {dfitpack_int!r}'   # 如果未提供消息，则创建一条新消息
        raise exception(msg)   # 抛出异常
    return dfitpack_int.type(x)   # 返回 x 转换为 dfitpack_int 后的值


_iermess = {
    0: ["The spline has a residual sum of squares fp such that "
        "abs(fp-s)/s<=0.001", None],   # 拟合样条的残差平方和满足条件
    -1: ["The spline is an interpolating spline (fp=0)", None],   # 样条是插值样条
    -2: ["The spline is weighted least-squares polynomial of degree k.\n"
         "fp gives the upper bound fp0 for the smoothing factor s", None],   # 样条是加权最小二乘多项式

    1: ["The required storage space exceeds the available storage space.\n"
        "Probable causes: data (x,y) size is too small or smoothing parameter"
        "\ns is too small (fp>s).", ValueError],   # 所需存储空间超出可用空间

    2: ["A theoretically impossible result when finding a smoothing spline\n"
        "with fp = s. Probable cause: s too small. (abs(fp-s)/s>0.001)",
        ValueError],   # 寻找平滑样条时出现理论上不可能的结果

    3: ["The maximal number of iterations (20) allowed for finding smoothing\n"
        "spline with fp=s has been reached. Probable cause: s too small.\n"
        "(abs(fp-s)/s>0.001)", ValueError],   # 达到寻找平滑样条的最大迭代次数

    10: ["Error on input data", ValueError],   # 输入数据错误

    'unknown': ["An error occurred", TypeError]   # 发生未知错误
}

_iermess2 = {
    0: ["The spline has a residual sum of squares fp such that "
        "abs(fp-s)/s<=0.001", None],   # 拟合样条的残差平方和满足条件
    -1: ["The spline is an interpolating spline (fp=0)", None],   # 样条是插值样条
    # 错误代码与对应的错误说明列表，每个条目包含错误代码、错误说明文本和可能的异常类型
    -2: ["The spline is weighted least-squares polynomial of degree kx and ky."
         "\nfp gives the upper bound fp0 for the smoothing factor s", None],
        # 键 -2: 描述加权最小二乘多项式的阶数 kx 和 ky
        # 值：说明 fp 是平滑因子 s 的上界 fp0
    
    -3: ["Warning. The coefficients of the spline have been computed as the\n"
         "minimal norm least-squares solution of a rank deficient system.",
         None],
        # 键 -3: 警告，样条插值的系数是通过计算秩亏系统的最小范数最小二乘解得到的
        # 值：无异常类型，仅作为警告信息
    
    1: ["The required storage space exceeds the available storage space.\n"
        "Probable causes: nxest or nyest too small or s is too small. (fp>s)",
        ValueError],
        # 键 1: 所需存储空间超过可用存储空间
        # 值：ValueError 异常，可能的原因包括 nxest 或 nyest 过小或 s 过小（fp>s）
    
    2: ["A theoretically impossible result when finding a smoothing spline\n"
        "with fp = s. Probable causes: s too small or badly chosen eps.\n"
        "(abs(fp-s)/s>0.001)", ValueError],
        # 键 2: 在找到平滑样条时出现理论上不可能的结果，其中 fp = s
        # 值：ValueError 异常，可能的原因包括 s 过小或选择的 eps 不合适（abs(fp-s)/s>0.001）
    
    3: ["The maximal number of iterations (20) allowed for finding smoothing\n"
        "spline with fp=s has been reached. Probable cause: s too small.\n"
        "(abs(fp-s)/s>0.001)", ValueError],
        # 键 3: 达到了查找平滑样条的允许的最大迭代次数（20），其中 fp=s
        # 值：ValueError 异常，可能的原因是 s 过小（abs(fp-s)/s>0.001）
    
    4: ["No more knots can be added because the number of B-spline\n"
        "coefficients already exceeds the number of data points m.\n"
        "Probable causes: either s or m too small. (fp>s)", ValueError],
        # 键 4: 不能再添加结点，因为 B 样条系数的数量已经超过数据点数 m
        # 值：ValueError 异常，可能的原因包括 s 或 m 过小（fp>s）
    
    5: ["No more knots can be added because the additional knot would\n"
        "coincide with an old one. Probable cause: s too small or too large\n"
        "a weight to an inaccurate data point. (fp>s)", ValueError],
        # 键 5: 不能再添加结点，因为额外的结点会与旧结点重合
        # 值：ValueError 异常，可能的原因包括 s 过小或过大，或对不准确数据点赋予了过大权重（fp>s）
    
    10: ["Error on input data", ValueError],
        # 键 10: 输入数据错误
        # 值：ValueError 异常
    
    11: ["rwrk2 too small, i.e., there is not enough workspace for computing\n"
         "the minimal least-squares solution of a rank deficient system of\n"
         "linear equations.", ValueError],
        # 键 11: rwrk2 太小，即没有足够的工作空间来计算秩亏系统的最小二乘解
        # 值：ValueError 异常
    
    'unknown': ["An error occurred", TypeError]
        # 键 'unknown': 发生了未知错误
        # 值：TypeError 异常
}

# 初始化一个缓存字典 `_parcur_cache`，包含一些初始的空数组和默认值
_parcur_cache = {'t': array([], float), 'wrk': array([], float),
                 'iwrk': array([], dfitpack_int), 'u': array([], float),
                 'ub': 0, 'ue': 1}


# 定义函数 `splprep`，用于样条插值的预处理
def splprep(x, w=None, u=None, ub=None, ue=None, k=3, task=0, s=None, t=None,
            full_output=0, nest=None, per=0, quiet=1):
    # 查看 `_fitpack_py/splprep` 的文档字符串以获取更多信息

    # 如果 `task` 小于等于 0，则重新设置 `_parcur_cache`
    if task <= 0:
        _parcur_cache = {'t': array([], float), 'wrk': array([], float),
                         'iwrk': array([], dfitpack_int), 'u': array([], float),
                         'ub': 0, 'ue': 1}

    # 将输入 `x` 至少转换为一维数组
    x = atleast_1d(x)
    # 确定输入数组的维数 `idim` 和长度 `m`
    idim, m = x.shape

    # 如果 `per` 标志为真，则检查是否需要将周期性边界条件应用于输入数据
    if per:
        for i in range(idim):
            if x[i][0] != x[i][-1]:
                # 如果不安静模式打开，则警告用户
                if not quiet:
                    warnings.warn(RuntimeWarning('Setting x[%d][%d]=x[%d][0]' %
                                                 (i, m, i)),
                                  stacklevel=2)
                # 设置周期性边界条件
                x[i][-1] = x[i][0]

    # 检查输入维数 `idim` 必须在 0 到 11 之间
    if not 0 < idim < 11:
        raise TypeError('0 < idim < 11 must hold')

    # 如果未提供 `w`，则创建一个长度为 `m` 的全一数组
    if w is None:
        w = ones(m, float)
    else:
        w = atleast_1d(w)

    # 如果 `u` 存在，则设置 `_parcur_cache` 中的 `u`，并相应地设置 `ub` 和 `ue`
    ipar = (u is not None)
    if ipar:
        _parcur_cache['u'] = u
        if ub is None:
            _parcur_cache['ub'] = u[0]
        else:
            _parcur_cache['ub'] = ub
        if ue is None:
            _parcur_cache['ue'] = u[-1]
        else:
            _parcur_cache['ue'] = ue
    else:
        # 否则，在 `_parcur_cache` 中设置 `u` 为长度为 `m` 的零数组
        _parcur_cache['u'] = zeros(m, float)

    # 检查样条阶数 `k` 必须在 1 到 5 之间
    if not (1 <= k <= 5):
        raise TypeError('1 <= k= %d <=5 must hold' % k)

    # 检查 `task` 必须在 -1、0 或 1 之间
    if not (-1 <= task <= 1):
        raise TypeError('task must be -1, 0 or 1')

    # 检查输入维度的匹配性
    if (not len(w) == m) or (ipar == 1 and (not len(u) == m)):
        raise TypeError('Mismatch of input dimensions')

    # 如果未提供 `s`，则根据默认公式计算其值
    if s is None:
        s = m - sqrt(2*m)

    # 如果 `t` 未提供且 `task` 为 -1，则引发异常
    if t is None and task == -1:
        raise TypeError('Knots must be given for task=-1')

    # 如果提供了 `t`，则将其至少转换为一维数组
    if t is not None:
        _parcur_cache['t'] = atleast_1d(t)

    # 获取 `_parcur_cache['t']` 的长度 `n`
    n = len(_parcur_cache['t'])

    # 如果 `task` 为 -1 且 `n` 小于 2*k + 2，则引发异常
    if task == -1 and n < 2*k + 2:
        raise TypeError('There must be at least 2*k+2 knots for task=-1')

    # 检查 `m` 必须大于 `k`
    if m <= k:
        raise TypeError('m > k must hold')

    # 如果未提供 `nest`，则计算默认值
    if nest is None:
        nest = m + 2*k

    # 如果 `task` 大于等于 0 且 `s` 等于 0，或者 `nest` 小于 0，则进行适当调整
    if (task >= 0 and s == 0) or (nest < 0):
        if per:
            nest = m + 2*k
        else:
            nest = m + k + 1

    # 确保 `nest` 至少为 2*k + 3
    nest = max(nest, 2*k + 3)

    # 从 `_parcur_cache` 中获取缓存的 `u`, `ub`, `ue`, `t`, `wrk`, `iwrk` 的值
    u = _parcur_cache['u']
    ub = _parcur_cache['ub']
    ue = _parcur_cache['ue']
    t = _parcur_cache['t']
    wrk = _parcur_cache['wrk']
    iwrk = _parcur_cache['iwrk']

    # 调用 `_fitpack._parcur` 函数进行样条插值计算，并获取返回的结果
    t, c, o = _fitpack._parcur(ravel(transpose(x)), w, u, ub, ue, k,
                               task, ipar, s, t, nest, wrk, iwrk, per)

    # 更新 `_parcur_cache` 中的 `u`, `ub`, `ue`, `t`, `wrk`, `iwrk` 的值
    _parcur_cache['u'] = o['u']
    _parcur_cache['ub'] = o['ub']
    _parcur_cache['ue'] = o['ue']
    _parcur_cache['t'] = t
    _parcur_cache['wrk'] = o['wrk']
    _parcur_cache['iwrk'] = o['iwrk']

    # 获取结果中的 `ier` 和 `fp` 的值
    ier = o['ier']
    fp = o['fp']

    # 获取 `t` 的长度
    n = len(t)

    # 重新获取 `u` 的值
    u = o['u']

    # 将系数矩阵 `c` 重塑为正确的形状
    c.shape = idim, n - k - 1

    # 返回结果 `tcku`，其中包含 knots, coefficients, and degree
    tcku = [t, list(c), k], u
    # 如果 ier 小于等于 0 并且非安静模式，则发出运行时警告，指明相关参数信息
    if ier <= 0 and not quiet:
        warnings.warn(RuntimeWarning(_iermess[ier][0] +
                                     "\tk=%d n=%d m=%d fp=%f s=%f" %
                                     (k, len(t), m, fp, s)),
                      stacklevel=2)
    
    # 如果 ier 大于 0 并且非完整输出模式，则根据不同的 ier 值发出警告
    if ier > 0 and not full_output:
        if ier in [1, 2, 3]:
            # 根据 ier 值选择对应的警告信息进行发出
            warnings.warn(RuntimeWarning(_iermess[ier][0]), stacklevel=2)
        else:
            try:
                # 抛出特定类型的错误，根据 ier 值选择错误类型
                raise _iermess[ier][1](_iermess[ier][0])
            except KeyError as e:
                # 如果 ier 对应的错误类型未知，则抛出默认的未知错误类型
                raise _iermess['unknown'][1](_iermess['unknown'][0]) from e
    
    # 如果 full_output 为真，则返回 tcku, fp, ier, 对应的信息文本
    if full_output:
        try:
            # 尝试返回 tcku, fp, ier 和相应的信息文本
            return tcku, fp, ier, _iermess[ier][0]
        except KeyError:
            # 如果 ier 对应的信息文本未知，则返回 tcku, fp, ier 和默认的未知信息文本
            return tcku, fp, ier, _iermess['unknown'][0]
    else:
        # 如果不是完整输出模式，则只返回 tcku
        return tcku
_curfit_cache = {'t': array([], float), 'wrk': array([], float),
                 'iwrk': array([], dfitpack_int)}

初始化一个全局变量 `_curfit_cache`，用于缓存一些计算中间结果，包括空数组和类型为 `dfitpack_int` 的数组。


def splrep(x, y, w=None, xb=None, xe=None, k=3, task=0, s=None, t=None,
           full_output=0, per=0, quiet=1):

定义一个函数 `splrep`，用于进行样条插值拟合。


    # see the docstring of `_fitpack_py/splrep`
    if task <= 0:
        _curfit_cache = {}

如果 `task` 小于等于 0，清空 `_curfit_cache` 缓存，准备重新计算。


    x, y = map(atleast_1d, [x, y])

确保 `x` 和 `y` 至少是一维数组。


    m = len(x)

获取数组 `x` 的长度。


    if w is None:
        w = ones(m, float)
        if s is None:
            s = 0.0
    else:
        w = atleast_1d(w)
        if s is None:
            s = m - sqrt(2*m)

处理权重数组 `w`，如果未提供则初始化为全一数组，如果 `s` 未提供则计算默认值。


    if not len(w) == m:
        raise TypeError('len(w)=%d is not equal to m=%d' % (len(w), m))

检查权重数组 `w` 的长度是否与 `x` 数组长度一致。


    if (m != len(y)) or (m != len(w)):
        raise TypeError('Lengths of the first three arguments (x,y,w) must '
                        'be equal')

检查 `x`、`y` 和 `w` 的长度是否相等。


    if not (1 <= k <= 5):
        raise TypeError('Given degree of the spline (k=%d) is not supported. '
                        '(1<=k<=5)' % k)

检查样条插值的阶数 `k` 是否在支持范围内。


    if m <= k:
        raise TypeError('m > k must hold')

确保数据点数大于样条插值阶数 `k`。


    if xb is None:
        xb = x[0]
    if xe is None:
        xe = x[-1]

如果未提供插值区间的端点 `xb` 和 `xe`，则默认取 `x` 数组的第一个和最后一个元素。


    if not (-1 <= task <= 1):
        raise TypeError('task must be -1, 0 or 1')

确保 `task` 的取值在预期范围内。


    if t is not None:
        task = -1

如果提供了节点 `t`，则将 `task` 设置为 `-1`。


    if task == -1:
        if t is None:
            raise TypeError('Knots must be given for task=-1')
        numknots = len(t)
        _curfit_cache['t'] = empty((numknots + 2*k + 2,), float)
        _curfit_cache['t'][k+1:-k-1] = t
        nest = len(_curfit_cache['t'])

如果 `task` 为 `-1`，根据给定的节点 `t`，计算节点数组并存储在 `_curfit_cache['t']` 中。


    elif task == 0:
        if per:
            nest = max(m + 2*k, 2*k + 3)
        else:
            nest = max(m + k + 1, 2*k + 3)
        t = empty((nest,), float)
        _curfit_cache['t'] = t

如果 `task` 为 `0`，根据是否周期性 `per`，计算节点数量并存储在 `_curfit_cache['t']` 中。


    if task <= 0:
        if per:
            _curfit_cache['wrk'] = empty((m*(k + 1) + nest*(8 + 5*k),), float)
        else:
            _curfit_cache['wrk'] = empty((m*(k + 1) + nest*(7 + 3*k),), float)
        _curfit_cache['iwrk'] = empty((nest,), dfitpack_int)

如果 `task` 小于等于 `0`，根据情况初始化工作空间和整数工作数组。


    try:
        t = _curfit_cache['t']
        wrk = _curfit_cache['wrk']
        iwrk = _curfit_cache['iwrk']
    except KeyError as e:
        raise TypeError("must call with task=1 only after"
                        " call with task=0,-1") from e

尝试获取缓存中的 `t`、`wrk` 和 `iwrk`，如果缓存中不存在则抛出异常提示。


    if not per:
        n, c, fp, ier = dfitpack.curfit(task, x, y, w, t, wrk, iwrk,
                                        xb, xe, k, s)
    else:
        n, c, fp, ier = dfitpack.percur(task, x, y, w, t, wrk, iwrk, k, s)

根据是否周期性调用不同的拟合函数进行样条插值计算。


    tck = (t[:n], c[:n], k)

生成插值结果的三元组 `tck`，包括节点、系数和阶数。


    if ier <= 0 and not quiet:
        _mess = (_iermess[ier][0] + "\tk=%d n=%d m=%d fp=%f s=%f" %
                 (k, len(t), m, fp, s))
        warnings.warn(RuntimeWarning(_mess), stacklevel=2)

如果计算出错且非安静模式，则发出运行时警告。


    if ier > 0 and not full_output:
        if ier in [1, 2, 3]:
            warnings.warn(RuntimeWarning(_iermess[ier][0]), stacklevel=2)
        else:
            try:
                raise _iermess[ier][1](_iermess[ier][0])
            except KeyError as e:
                raise _iermess['unknown'][1](_iermess['unknown'][0]) from e

如果计算出错且非完整输出模式，则发出相应警告或异常。
    # 如果需要完整的输出（full_output=True）则执行以下代码块
    if full_output:
        # 尝试返回 tck, fp, ier, 以及对应的错误消息或描述
        try:
            return tck, fp, ier, _iermess[ier][0]
        # 如果在 _iermess 中找不到对应的错误消息，则返回默认的 'unknown' 错误消息
        except KeyError:
            return tck, fp, ier, _iermess['unknown'][0]
    else:
        # 否则，只返回 tck
        return tck
# 根据给定节点和系数确定的 B 样条曲线上计算点的值或导数
def splev(x, tck, der=0, ext=0):
    # 从输入参数 tck 中解包出节点 t、系数 c 和阶数 k
    t, c, k = tck
    try:
        # 尝试访问 c 的第一个元素的第一个元素，以检测是否为参数化表示
        c[0][0]
        parametric = True
    except Exception:
        # 如果访问失败，说明不是参数化表示
        parametric = False

    if parametric:
        # 如果是参数化表示，递归调用 splev 函数计算结果
        return list(map(lambda c, x=x, t=t, k=k, der=der:
                        splev(x, [t, c, k], der, ext), c))
    else:
        # 如果不是参数化表示，检查导数 der 是否在有效范围内
        if not (0 <= der <= k):
            raise ValueError("0<=der=%d<=k=%d must hold" % (der, k))
        # 检查 ext 是否在允许的范围内
        if ext not in (0, 1, 2, 3):
            raise ValueError("ext = %s not in (0, 1, 2, 3) " % ext)

        # 将输入参数 x 转换为数组，并确保其至少是一维的
        x = asarray(x)
        shape = x.shape
        x = atleast_1d(x).ravel()

        # 根据导数 der 的不同选择调用不同的底层函数计算结果
        if der == 0:
            y, ier = dfitpack.splev(t, c, k, x, ext)
        else:
            y, ier = dfitpack.splder(t, c, k, x, der, ext)

        # 处理底层计算函数返回的错误码 ier
        if ier == 10:
            raise ValueError("Invalid input data")
        if ier == 1:
            raise ValueError("Found x value not in the domain")
        if ier:
            raise TypeError("An error occurred")

        # 将计算得到的结果 y 重新整形成原始输入 x 的形状，并返回
        return y.reshape(shape)


# 对 B 样条曲线在给定区间 [a, b] 上进行积分计算
def splint(a, b, tck, full_output=0):
    # 从输入参数 tck 中解包出节点 t、系数 c 和阶数 k
    t, c, k = tck
    try:
        # 尝试访问 c 的第一个元素的第一个元素，以检测是否为参数化表示
        c[0][0]
        parametric = True
    except Exception:
        # 如果访问失败，说明不是参数化表示
        parametric = False

    if parametric:
        # 如果是参数化表示，递归调用 splint 函数计算结果
        return list(map(lambda c, a=a, b=b, t=t, k=k:
                        splint(a, b, [t, c, k]), c))
    else:
        # 调用底层函数 splint 计算 B 样条曲线在 [a, b] 区间上的积分
        aint, wrk = dfitpack.splint(t, c, k, a, b)
        if full_output:
            return aint, wrk
        else:
            return aint


# 计算给定 B 样条曲线的根（零点）
def sproot(tck, mest=10):
    # 从输入参数 tck 中解包出节点 t、系数 c 和阶数 k
    t, c, k = tck
    # 检查阶数 k 是否为 3，因为 sproot 只对立方样条（k=3）有效
    if k != 3:
        raise ValueError("sproot works only for cubic (k=3) splines")

    try:
        # 尝试访问 c 的第一个元素的第一个元素，以检测是否为参数化表示
        c[0][0]
        parametric = True
    except Exception:
        # 如果访问失败，说明不是参数化表示
        parametric = False

    if parametric:
        # 如果是参数化表示，递归调用 sproot 函数计算结果
        return list(map(lambda c, t=t, k=k, mest=mest:
                        sproot([t, c, k], mest), c))
    else:
        # 检查节点数是否大于等于 8
        if len(t) < 8:
            raise TypeError("The number of knots %d>=8" % len(t))
        
        # 调用底层函数 sproot 计算 B 样条曲线的根
        z, m, ier = dfitpack.sproot(t, c, mest)

        # 根据底层函数返回的错误码 ier 处理不同的错误情况
        if ier == 10:
            raise TypeError("Invalid input data. "
                            "t1<=..<=t4<t5<..<tn-3<=..<=tn must hold.")
        if ier == 0:
            return z[:m]
        if ier == 1:
            warnings.warn(RuntimeWarning("The number of zeros exceeds mest"),
                          stacklevel=2)
            return z[:m]
        raise TypeError("Unknown error")


# 计算给定 B 样条曲线在某一点的值及其导数
def spalde(x, tck):
    # 从输入参数 tck 中解包出节点 t、系数 c 和阶数 k
    t, c, k = tck
    try:
        # 尝试访问 c 的第一个元素的第一个元素，以检测是否为参数化表示
        c[0][0]
        parametric = True
    except Exception:
        # 如果访问失败，说明不是参数化表示
        parametric = False

    if parametric:
        # 如果是参数化表示，递归调用 spalde 函数计算结果
        return list(map(lambda c, x=x, t=t, k=k:
                        spalde(x, [t, c, k]), c))
    else:
        # 如果输入的 x 不是数组，则转换为至少是一维的数组
        x = atleast_1d(x)
        # 如果 x 的长度大于1，对 x 中的每个元素应用 spalde 函数，并返回结果列表
        if len(x) > 1:
            return list(map(lambda x, tck=tck: spalde(x, tck), x))
        # 调用 dfitpack.spalde 函数计算导数，返回导数 d 和错误标志 ier
        d, ier = dfitpack.spalde(t, c, k+1, x[0])
        # 如果计算成功，返回导数 d
        if ier == 0:
            return d
        # 如果输入数据不合法，抛出 TypeError 异常
        if ier == 10:
            raise TypeError("Invalid input data. t(k)<=x<=t(n-k+1) must hold.")
        # 如果出现未知错误，抛出 TypeError 异常
        raise TypeError("Unknown error")
# 定义一个缓存字典，用于存储插值计算过程中的临时数据
_surfit_cache = {'tx': array([], float), 'ty': array([], float),
                 'wrk': array([], float), 'iwrk': array([], dfitpack_int)}


def bisplrep(x, y, z, w=None, xb=None, xe=None, yb=None, ye=None,
             kx=3, ky=3, task=0, s=None, eps=1e-16, tx=None, ty=None,
             full_output=0, nxest=None, nyest=None, quiet=1):
    """
    Find a bivariate B-spline representation of a surface.

    给定一组数据点 (x[i], y[i], z[i]) 表示一个二维曲面 z=f(x,y)，计算该曲面的 B 样条表示。
    基于 FITPACK 中的 SURFIT 程序。

    Parameters
    ----------
    x, y, z : ndarray
        数据点的一维数组，分别表示 x, y, z 坐标。
    w : ndarray, optional
        权重数组。默认为 ``w=np.ones(len(x))``。
    xb, xe : float, optional
        在 `x` 上的逼近区间的端点。
        默认为 ``xb = x.min(), xe=x.max()``。
    yb, ye : float, optional
        在 `y` 上的逼近区间的端点。
        默认为 ``yb=y.min(), ye = y.max()``。
    kx, ky : int, optional
        样条的阶数 (1 <= kx, ky <= 5)。
        推荐使用三阶样条 (kx=ky=3)。
    task : int, optional
        控制任务的参数：
        如果 task=0，则在 x 和 y 上找到节点和给定平滑因子 s 下的系数。
        如果 task=1，则为另一个平滑因子 s 找到节点和系数。必须先调用 bisplrep(task=0) 或 bisplrep(task=1)。
        如果 task=-1，则为给定节点 tx, ty 找到系数。
    s : float, optional
        非负的平滑因子。如果权重对应于 z 误差的标准差的倒数，则在范围 ``(m-sqrt(2*m),m+sqrt(2*m))`` 内选择一个合适的 s 值，
        其中 m=len(x)。
    eps : float, optional
        用于确定超定线性方程组有效秩的阈值 (0 < eps < 1)。
        `eps` 不太可能需要更改。
    tx, ty : ndarray, optional
        用于 task=-1 时的样条节点的一维数组。
    full_output : int, optional
        非零以返回可选的输出。
    nxest, nyest : int, optional
        节点总数的过估计。如果为 None，则计算方法为
        ``nxest = max(kx+sqrt(m/2),2*kx+3)``,
        ``nyest = max(ky+sqrt(m/2),2*ky+3)``。
    quiet : int, optional
        非零以抑制消息的打印输出。

    Returns
    -------
    tck : array_like
        包含样条节点 (tx, ty) 和系数 (c) 的列表 [tx, ty, c, kx, ky]，表示二维 B 样条曲面的表示，
        以及样条的阶数。
    fp : ndarray
        样条逼近的加权残差平方和。
    """
    ier : int
        An integer flag indicating success or failure of the spline fitting process.
        Success is indicated if ier <= 0. If ier is in [1, 2, 3], an error occurred but was not raised.
        Otherwise, an error is raised.

    msg : str
        A message corresponding to the integer flag ier, providing additional information about the outcome.

    See Also
    --------
    splprep, splrep, splint, sproot, splev
    UnivariateSpline, BivariateSpline

    Notes
    -----
    See `bisplev` to evaluate the value of the B-spline given its tck representation.

    If the input data dimensions have incommensurate units and vary significantly in scale,
    the resulting interpolant may exhibit numerical artifacts. Consider normalizing or rescaling
    the data before performing interpolation.

    References
    ----------
    .. [1] Dierckx P.: An algorithm for surface fitting with spline functions
       Ima J. Numer. Anal. 1 (1981) 267-283.
    .. [2] Dierckx P.: An algorithm for surface fitting with spline functions
       report tw50, Dept. Computer Science, K.U.Leuven, 1980.
    .. [3] Dierckx P.: Curve and surface fitting with splines, Monographs on
       Numerical Analysis, Oxford University Press, 1993.

    Examples
    --------
    Examples are provided in the tutorial on 2D spline interpolation.
    Ref: https://docs.scipy.org/doc/scipy/reference/tutorial/interpolate.html#id10

    """
    x, y, z = map(ravel, [x, y, z])  # Ensure x, y, z are flattened to 1-dimensional arrays.
    m = len(x)  # Determine the length of array x.

    # Check if the lengths of x, y, and z are equal.
    if not (m == len(y) == len(z)):
        raise TypeError('len(x)==len(y)==len(z) must hold.')

    # Initialize weights array w if it is not provided.
    if w is None:
        w = ones(m, float)
    else:
        w = atleast_1d(w)  # Ensure w is at least 1-dimensional.

    # Check if the length of w matches the length of x.
    if not len(w) == m:
        raise TypeError('len(w)=%d is not equal to m=%d' % (len(w), m))

    # Set minimum values for xb, xe, yb, ye if they are not provided.
    if xb is None:
        xb = x.min()
    if xe is None:
        xe = x.max()
    if yb is None:
        yb = y.min()
    if ye is None:
        ye = y.max()

    # Check if task is within acceptable range.
    if not (-1 <= task <= 1):
        raise TypeError('task must be -1, 0 or 1')

    # Set default value for s if it is not provided.
    if s is None:
        s = m - sqrt(2 * m)

    # Check requirements for tx if task is -1.
    if tx is None and task == -1:
        raise TypeError('Knots_x must be given for task=-1')

    # Cache knots_x if tx is provided.
    if tx is not None:
        _surfit_cache['tx'] = atleast_1d(tx)

    # Determine the length of cached knots_x.
    nx = len(_surfit_cache['tx'])

    # Check requirements for ty if task is -1.
    if ty is None and task == -1:
        raise TypeError('Knots_y must be given for task=-1')

    # Cache knots_y if ty is provided.
    if ty is not None:
        _surfit_cache['ty'] = atleast_1d(ty)

    # Determine the length of cached knots_y.
    ny = len(_surfit_cache['ty'])

    # Check minimum number of knots_x for task=-1.
    if task == -1 and nx < 2 * kx + 2:
        raise TypeError('There must be at least 2*kx+2 knots_x for task=-1')

    # Check minimum number of knots_y for task=-1.
    if task == -1 and ny < 2 * ky + 2:
        raise TypeError('There must be at least 2*ky+2 knots_y for task=-1')

    # Check if the given spline degrees kx and ky are supported.
    if not ((1 <= kx <= 5) and (1 <= ky <= 5)):
        raise TypeError('Given degree of the spline (kx, ky=%d,%d) is not supported. (1<=k<=5)' % (kx, ky))

    # Check the condition m >= (kx+1)(ky+1) for spline fitting.
    if m < (kx + 1) * (ky + 1):
        raise TypeError('m >= (kx+1)(ky+1) must hold')

    # Estimate default values for nxest and nyest if they are not provided.
    if nxest is None:
        nxest = int(kx + sqrt(m / 2))
    if nyest is None:
        nyest = int(ky + sqrt(m / 2))

    # Ensure nxest and nyest meet minimum requirements.
    nxest, nyest = max(nxest, 2 * kx + 3), max(nyest, 2 * ky + 3)
    # 如果任务标识大于等于0并且s为0，则计算nxest和nyest
    if task >= 0 and s == 0:
        nxest = int(kx + sqrt(3*m))  # 计算nxest
        nyest = int(ky + sqrt(3*m))  # 计算nyest
    
    # 如果任务标识为-1，则将tx和ty放入_surfit_cache缓存
    if task == -1:
        _surfit_cache['tx'] = atleast_1d(tx)
        _surfit_cache['ty'] = atleast_1d(ty)
    
    # 从_surfit_cache缓存中获取tx、ty和wrk
    tx, ty = _surfit_cache['tx'], _surfit_cache['ty']
    wrk = _surfit_cache['wrk']
    
    # 计算u和v
    u = nxest - kx - 1
    v = nyest - ky - 1
    
    # 计算km和ne
    km = max(kx, ky) + 1
    ne = max(nxest, nyest)
    
    # 计算bx和by
    bx, by = kx*v + ky + 1, ky*u + kx + 1
    
    # 确保b1和b2是按顺序最小和最大的
    if bx > by:
        b1, b2 = by, by + u - kx
    else:
        b1, b2 = bx, bx + v - ky
    
    # 错误消息
    msg = "Too many data points to interpolate"
    
    # 计算lwrk1和lwrk2，处理可能的溢出
    lwrk1 = _int_overflow(u*v*(2 + b1 + b2) +
                          2*(u + v + km*(m + ne) + ne - kx - ky) + b2 + 1,
                          OverflowError,
                          msg=msg)
    lwrk2 = _int_overflow(u*v*(b2 + 1) + b2, OverflowError, msg=msg)
    
    # 调用_fitpack._surfit进行曲面拟合，返回tx、ty、c、o
    tx, ty, c, o = _fitpack._surfit(x, y, z, w, xb, xe, yb, ye, kx, ky,
                                    task, s, eps, tx, ty, nxest, nyest,
                                    wrk, lwrk1, lwrk2)
    
    # 将结果缓存到_curfit_cache
    _curfit_cache['tx'] = tx
    _curfit_cache['ty'] = ty
    _curfit_cache['wrk'] = o['wrk']
    
    # 从o中获取ier和fp
    ier, fp = o['ier'], o['fp']
    
    # 构建tck列表
    tck = [tx, ty, c, kx, ky]
    
    # 计算ierm，限制在[-3, 11]之间
    ierm = min(11, max(-3, ier))
    
    # 如果ierm小于等于0且不安静模式，则发出运行时警告
    if ierm <= 0 and not quiet:
        _mess = (_iermess2[ierm][0] +
                 "\tkx,ky=%d,%d nx,ny=%d,%d m=%d fp=%f s=%f" %
                 (kx, ky, len(tx), len(ty), m, fp, s))
        warnings.warn(RuntimeWarning(_mess), stacklevel=2)
    
    # 如果ierm大于0且不返回完整输出，则根据ier发出警告
    if ierm > 0 and not full_output:
        if ier in [1, 2, 3, 4, 5]:
            _mess = ("\n\tkx,ky=%d,%d nx,ny=%d,%d m=%d fp=%f s=%f" %
                     (kx, ky, len(tx), len(ty), m, fp, s))
            warnings.warn(RuntimeWarning(_iermess2[ierm][0] + _mess), stacklevel=2)
        else:
            try:
                raise _iermess2[ierm][1](_iermess2[ierm][0])
            except KeyError as e:
                raise _iermess2['unknown'][1](_iermess2['unknown'][0]) from e
    
    # 如果要求完整输出，则返回tck、fp、ier和相关消息
    if full_output:
        try:
            return tck, fp, ier, _iermess2[ierm][0]
        except KeyError:
            return tck, fp, ier, _iermess2['unknown'][0]
    else:
        # 否则，只返回tck
        return tck
# 计算二元 B-样条及其导数的值。

# 返回一个二维数组，其中包含在由排列组合的 x 和 y 构成的点处评估的样条函数值（或样条导数值）。
# 在特殊情况下，如果 x 或 y 或两者都是浮点数，则返回一个数组或仅一个浮点数。
# 基于 FITPACK 中的 BISPEV 和 PARDER 函数。

# Parameters 参数说明：
# x, y : ndarray
#     指定用于评估样条或其导数的定义域的一维数组。
# tck : tuple
#     由 `bisplrep` 返回的长度为 5 的序列，包含结点位置、系数和样条的阶数:
#     [tx, ty, c, kx, ky].
# dx, dy : int, optional
#     分别为 `x` 和 `y` 的偏导数阶数。

# Returns 返回值：
# vals : ndarray
#     在由 `x` 和 `y` 的排列组合形成的集合上评估的 B-样条或其导数。

# See Also 参见：
# splprep, splrep, splint, sproot, splev
# UnivariateSpline, BivariateSpline

# Notes 注释：
#     参见 `bisplrep` 以生成 `tck` 表示。

# References 参考文献：
# .. [1] Dierckx P. : An algorithm for surface fitting
#    with spline functions
#    Ima J. Numer. Anal. 1 (1981) 267-283.
# .. [2] Dierckx P. : An algorithm for surface fitting
#    with spline functions
#    report tw50, Dept. Computer Science,K.U.Leuven, 1980.
# .. [3] Dierckx P. : Curve and surface fitting with splines,
#    Monographs on Numerical Analysis, Oxford University Press, 1993.

# Examples 示例：
# 示例可以在教程中找到 :ref:`in the tutorial <tutorial-interpolate_2d_spline>`.

def bisplev(x, y, tck, dx=0, dy=0):
    tx, ty, c, kx, ky = tck
    # 检查偏导数的阶数是否在有效范围内
    if not (0 <= dx < kx):
        raise ValueError("0 <= dx = %d < kx = %d must hold" % (dx, kx))
    if not (0 <= dy < ky):
        raise ValueError("0 <= dy = %d < ky = %d must hold" % (dy, ky))
    # 将 x 和 y 转换为至少为一维的数组
    x, y = map(atleast_1d, [x, y])
    # 检查 x 和 y 是否为一维数组
    if (len(x.shape) != 1) or (len(y.shape) != 1):
        raise ValueError("First two entries should be rank-1 arrays.")

    msg = "Too many data points to interpolate."
    # 检查是否可能导致内存溢出
    _int_overflow(x.size * y.size, MemoryError, msg=msg)

    if dx != 0 or dy != 0:
        # 检查是否可能导致内存溢出
        _int_overflow((tx.size - kx - 1)*(ty.size - ky - 1),
                      MemoryError, msg=msg)
        # 调用 FITPACK 库中的 parder 函数计算偏导数
        z, ier = dfitpack.parder(tx, ty, c, kx, ky, dx, dy, x, y)
    else:
        # 调用 FITPACK 库中的 bispev 函数计算 B-样条
        z, ier = dfitpack.bispev(tx, ty, c, kx, ky, x, y)

    # 处理可能的错误情况
    if ier == 10:
        raise ValueError("Invalid input data")
    if ier:
        raise TypeError("An error occurred")
    # 调整 z 的形状为 len(x) x len(y)
    z.shape = len(x), len(y)
    # 根据 z 的形状返回相应结果
    if len(z) > 1:
        return z
    if len(z[0]) > 1:
        return z[0]
    return z[0][0]
    tx, ty, c, kx, ky = tck
    # 从参数 tck 中解包出 spline 曲线的关键信息：节点位置 tx, ty，系数 c，以及曲线的阶数 kx, ky

    return dfitpack.dblint(tx, ty, c, kx, ky, xa, xb, ya, yb)
    # 调用 dfitpack.dblint 函数计算二维样条曲线在指定区间 [xa, xb] x [ya, yb] 上的二重积分，并返回结果
def insert(x, tck, m=1, per=0):
    # 解释：从传入的 tck 中解包出 t, c, k
    t, c, k = tck
    
    try:
        # 尝试访问 c 的第一个元素的第一个元素，以检测是否为参数化样条曲线
        c[0][0]
        parametric = True
    except Exception:
        # 如果访问失败，则不是参数化样条曲线
        parametric = False
    
    if parametric:
        # 如果是参数化样条曲线，则递归调用 insert 函数计算每个参数组合的插值
        cc = []
        for c_vals in c:
            tt, cc_val, kk = insert(x, [t, c_vals, k], m)
            cc.append(cc_val)
        return (tt, cc, kk)
    else:
        # 如果不是参数化样条曲线，则调用 _fitpack._insert 函数进行插值计算
        tt, cc, ier = _fitpack._insert(per, t, c, k, x, m)
        
        # 检查插值过程中的错误码，若为 10，则抛出数值错误异常；若不为 0，则抛出类型错误异常
        if ier == 10:
            raise ValueError("Invalid input data")
        if ier:
            raise TypeError("An error occurred")
        
        return (tt, cc, k)


def splder(tck, n=1):
    # 解释：从传入的 tck 中解包出 t, c, k
    t, c, k = tck
    
    if n < 0:
        # 如果 n 小于 0，则调用 splantider 函数计算相反次数的反导数
        return splantider(tck, -n)

    if n > k:
        # 如果 n 大于 k（样条的阶数），则抛出值错误异常
        raise ValueError(f"Order of derivative (n = {n!r}) must be <= "
                         f"order of spline (k = {tck[2]!r})")

    # 为 c 数组的尾部维度添加额外的轴
    sh = (slice(None),) + ((None,)*len(c.shape[1:]))

    with np.errstate(invalid='raise', divide='raise'):
        try:
            for j in range(n):
                # 计算导数公式中的分母
                dt = t[k+1:-1] - t[1:-k-1]
                dt = dt[sh]
                # 计算新的系数
                c = (c[1:-1-k] - c[:-2-k]) * k / dt
                # 将系数数组填充至与节点数目相同的大小（FITPACK 约定）
                c = np.r_[c, np.zeros((k,) + c.shape[1:])]
                # 调整节点数组
                t = t[1:-1]
                k -= 1
        except FloatingPointError as e:
            # 若出现浮点数错误，则抛出值错误异常
            raise ValueError(("The spline has internal repeated knots "
                              "and is not differentiable %d times") % n) from e

    return t, c, k


def splantider(tck, n=1):
    # 解释：从传入的 tck 中解包出 t, c, k
    t, c, k = tck

    if n < 0:
        # 如果 n 小于 0，则调用 splder 函数计算相反次数的导数
        return splder(tck, -n)

    # 为 c 数组的尾部维度添加额外的轴
    sh = (slice(None),) + (None,)*len(c.shape[1:])

    for j in range(n):
        # 这是 splder 函数操作的逆过程。

        # 计算反导数公式中的乘数
        dt = t[k+1:] - t[:-k-1]
        dt = dt[sh]
        # 计算新的系数
        c = np.cumsum(c[:-k-1] * dt, axis=0) / (k + 1)
        c = np.r_[np.zeros((1,) + c.shape[1:]),
                  c,
                  [c[-1]] * (k+2)]
        # 新的节点数组
        t = np.r_[t[0], t, t[-1]]
        k += 1

    return t, c, k
```