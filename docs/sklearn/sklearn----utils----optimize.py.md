# `D:\src\scipysrc\scikit-learn\sklearn\utils\optimize.py`

```
"""
Our own implementation of the Newton algorithm

Unlike the scipy.optimize version, this version of the Newton conjugate
gradient solver uses only one function call to retrieve the
func value, the gradient value and a callable for the Hessian matvec
product. If the function call is very expensive (e.g. for logistic
regression with large design matrix), this approach gives very
significant speedups.
"""

# This is a modified file from scipy.optimize
# Original authors: Travis Oliphant, Eric Jones
# Modifications by Gael Varoquaux, Mathieu Blondel and Tom Dupre la Tour
# SPDX-License-Identifier: BSD-3-Clause

import warnings  # 导入警告模块

import numpy as np  # 导入 NumPy 库
import scipy  # 导入 SciPy 库

from ..exceptions import ConvergenceWarning  # 从当前包中导入收敛警告异常
from .fixes import line_search_wolfe1, line_search_wolfe2  # 从当前包的 fixes 模块中导入两个函数


class _LineSearchError(RuntimeError):
    pass


def _line_search_wolfe12(
    f, fprime, xk, pk, gfk, old_fval, old_old_fval, verbose=0, **kwargs
):
    """
    Same as line_search_wolfe1, but fall back to line_search_wolfe2 if
    suitable step length is not found, and raise an exception if a
    suitable step length is not found.

    Raises
    ------
    _LineSearchError
        If no suitable step size is found.

    """
    is_verbose = verbose >= 2  # 判断是否开启详细输出模式
    eps = 16 * np.finfo(np.asarray(old_fval).dtype).eps  # 计算机器精度的放大量
    if is_verbose:
        print("  Line Search")  # 输出信息：线性搜索
        print(f"    eps=16 * finfo.eps={eps}")  # 输出信息：eps 的值
        print("    try line search wolfe1")  # 输出信息：尝试使用 wolfe1 进行线性搜索

    ret = line_search_wolfe1(f, fprime, xk, pk, gfk, old_fval, old_old_fval, **kwargs)  # 调用 line_search_wolfe1 函数进行线性搜索

    if is_verbose:
        _not_ = "not " if ret[0] is None else ""  # 判断线性搜索是否成功
        print("    wolfe1 line search was " + _not_ + "successful")  # 输出信息：显示 wolfe1 线性搜索是否成功
    if ret[0] is None:
        # 如果返回结果的第一个元素为 None，执行以下逻辑：
        
        # 从 kwargs 中获取参数 args，如果未提供则使用空元组
        args = kwargs.get("args", tuple())
        
        # 计算在 xk + pk 处的函数值
        fval = f(xk + pk, *args)
        
        # 计算一个极小的损失值，用于处理机器精度附近的相对损失差异
        tiny_loss = np.abs(old_fval * eps)
        
        # 计算当前损失改进与旧损失值的差
        loss_improvement = fval - old_fval
        
        # 检查损失改进的绝对值是否小于等于极小损失值
        check = np.abs(loss_improvement) <= tiny_loss
        
        # 如果开启了详细输出，打印损失改进情况的信息
        if is_verbose:
            print(
                "    check loss |improvement| <= eps * |loss_old|:"
                f" {np.abs(loss_improvement)} <= {tiny_loss} {check}"
            )
        
        # 如果检查通过，继续下一步条件判断
        if check:
            # 2.1 检查绝对梯度之和作为备用条件。
            
            # 计算当前梯度的 L1 范数
            sum_abs_grad_old = scipy.linalg.norm(gfk, ord=1)
            
            # 计算在 xk + pk 处的梯度
            grad = fprime(xk + pk, *args)
            
            # 计算新梯度的 L1 范数
            sum_abs_grad = scipy.linalg.norm(grad, ord=1)
            
            # 检查新梯度的 L1 范数是否小于旧梯度的 L1 范数
            check = sum_abs_grad < sum_abs_grad_old
            
            # 如果开启了详细输出，打印梯度比较信息
            if is_verbose:
                print(
                    "    check sum(|gradient|) < sum(|gradient_old|): "
                    f"{sum_abs_grad} < {sum_abs_grad_old} {check}"
                )
            
            # 如果检查通过，更新返回结果 ret
            if check:
                ret = (
                    1.0,              # 步长大小
                    ret[1] + 1,       # 函数评估次数加一
                    ret[2] + 1,       # 梯度评估次数加一
                    fval,             # 新函数值
                    old_fval,         # 旧函数值
                    grad,             # 新梯度
                )

    if ret[0] is None:
        # 如果返回结果的第一个元素仍为 None，说明线搜索失败，尝试使用不同的方法
        
        # 如果开启了详细输出，打印尝试使用其他方法的信息
        if is_verbose:
            print("    last resort: try line search wolfe2")
        
        # 调用 line_search_wolfe2 函数进行进一步的线搜索
        ret = line_search_wolfe2(
            f, fprime, xk, pk, gfk, old_fval, old_old_fval, **kwargs
        )
        
        # 如果开启了详细输出，打印线搜索结果信息
        if is_verbose:
            _not_ = "not " if ret[0] is None else ""
            print("    wolfe2 line search was " + _not_ + "successful")

    if ret[0] is None:
        # 如果返回结果的第一个元素仍为 None，抛出 _LineSearchError 异常
        
        raise _LineSearchError()

    return ret
def _cg(fhess_p, fgrad, maxiter, tol, verbose=0):
    """
    Solve iteratively the linear system 'fhess_p . xsupi = fgrad'
    with a conjugate gradient descent.

    Parameters
    ----------
    fhess_p : callable
        Function that takes the gradient as a parameter and returns the
        matrix product of the Hessian and gradient.

    fgrad : ndarray of shape (n_features,) or (n_features + 1,)
        Gradient vector.

    maxiter : int
        Number of CG iterations.

    tol : float
        Stopping criterion.

    Returns
    -------
    xsupi : ndarray of shape (n_features,) or (n_features + 1,)
        Estimated solution.
    """
    eps = 16 * np.finfo(np.float64).eps
    xsupi = np.zeros(len(fgrad), dtype=fgrad.dtype)
    ri = np.copy(fgrad)  # residual = fgrad - fhess_p @ xsupi
    psupi = -ri
    i = 0
    dri0 = np.dot(ri, ri)
    # We also keep track of |p_i|^2.
    psupi_norm2 = dri0
    is_verbose = verbose >= 2

    while i <= maxiter:
        if np.sum(np.abs(ri)) <= tol:
            if is_verbose:
                print(
                    f"  Inner CG solver iteration {i} stopped with\n"
                    f"    sum(|residuals|) <= tol: {np.sum(np.abs(ri))} <= {tol}"
                )
            break

        Ap = fhess_p(psupi)
        # check curvature
        curv = np.dot(psupi, Ap)
        if 0 <= curv <= eps * psupi_norm2:
            # See https://arxiv.org/abs/1803.02924, Algo 1 Capped Conjugate Gradient.
            if is_verbose:
                print(
                    f"  Inner CG solver iteration {i} stopped with\n"
                    f"    tiny_|p| = eps * ||p||^2, eps = {eps}, "
                    f"squred L2 norm ||p||^2 = {psupi_norm2}\n"
                    f"    curvature <= tiny_|p|: {curv} <= {eps * psupi_norm2}"
                )
            break
        elif curv < 0:
            if i > 0:
                if is_verbose:
                    print(
                        f"  Inner CG solver iteration {i} stopped with negative "
                        f"curvature, curvature = {curv}"
                    )
                break
            else:
                # fall back to steepest descent direction
                xsupi += dri0 / curv * psupi
                if is_verbose:
                    print("  Inner CG solver iteration 0 fell back to steepest descent")
                break
        alphai = dri0 / curv
        xsupi += alphai * psupi
        ri += alphai * Ap
        dri1 = np.dot(ri, ri)
        betai = dri1 / dri0
        psupi = -ri + betai * psupi
        # We use  |p_i|^2 = |r_i|^2 + beta_i^2 |p_{i-1}|^2
        psupi_norm2 = dri1 + betai**2 * psupi_norm2
        i = i + 1
        dri0 = dri1  # update np.dot(ri,ri) for next time.
    if is_verbose and i > maxiter:
        print(
            f"  Inner CG solver stopped reaching maxiter={i - 1} with "
            f"sum(|residuals|) = {np.sum(np.abs(ri))}"
        )
    return xsupi


注释：

# 定义了一个使用共轭梯度法解决线性系统的函数
def _cg(fhess_p, fgrad, maxiter, tol, verbose=0):
    eps = 16 * np.finfo(np.float64).eps  # 机器精度的16倍
    xsupi = np.zeros(len(fgrad), dtype=fgrad.dtype)  # 初始化解向量为零向量
    ri = np.copy(fgrad)  # 残差向量，初始为梯度向量 fgrad
    psupi = -ri  # 搜索方向，初始为负梯度方向
    i = 0  # 迭代计数器
    dri0 = np.dot(ri, ri)  # 初始残差的平方范数
    psupi_norm2 = dri0  # 初始搜索方向的平方范数
    is_verbose = verbose >= 2  # 是否需要详细输出迭代过程

    while i <= maxiter:  # 开始迭代
        if np.sum(np.abs(ri)) <= tol:  # 判断是否达到停止条件
            if is_verbose:
                print(
                    f"  Inner CG solver iteration {i} stopped with\n"
                    f"    sum(|residuals|) <= tol: {np.sum(np.abs(ri))} <= {tol}"
                )
            break

        Ap = fhess_p(psupi)  # 计算 Hessian 和梯度的乘积 Ap
        curv = np.dot(psupi, Ap)  # 计算当前搜索方向的曲率
        if 0 <= curv <= eps * psupi_norm2:  # 判断是否满足收敛条件
            if is_verbose:
                print(
                    f"  Inner CG solver iteration {i} stopped with\n"
                    f"    tiny_|p| = eps * ||p||^2, eps = {eps}, "
                    f"squred L2 norm ||p||^2 = {psupi_norm2}\n"
                    f"    curvature <= tiny_|p|: {curv} <= {eps * psupi_norm2}"
                )
            break
        elif curv < 0:  # 如果曲率为负数
            if i > 0:
                if is_verbose:
                    print(
                        f"  Inner CG solver iteration {i} stopped with negative "
                        f"curvature, curvature = {curv}"
                    )
                break
            else:
                # 如果是第一次迭代，回退到最陡下降方向
                xsupi += dri0 / curv * psupi
                if is_verbose:
                    print("  Inner CG solver iteration 0 fell back to steepest descent")
                break
        alphai = dri0 / curv  # 计算步长
        xsupi += alphai * psupi  # 更新解向量
        ri += alphai * Ap  # 更新残差向量
        dri1 = np.dot(ri, ri)  # 更新残差的平方范数
        betai = dri1 / dri0  # 计算 beta_i
        psupi = -ri + betai * psupi  # 更新搜索方向
        psupi_norm2 = dri1 + betai**2 * psupi_norm2  # 更新搜索方向的平方范数
        i = i + 1  # 迭代计数器加一
        dri0 = dri1  # 更新下一次迭代的初始残差范数
    if is_verbose and i > maxiter:  # 如果迭代达到最大次数仍未收敛
        print(
            f"  Inner CG solver stopped reaching maxiter={i - 1} with "
            f"sum(|residuals|) = {np.sum(np.abs(ri))}"
        )
    return xsupi  # 返回估计的解向量
    grad,                     # 梯度函数
    x0,                       # 初始变量值
    args=(),                  # 非变量参数，默认为空元组
    tol=1e-4,                 # 公差，控制算法收敛的精度
    maxiter=100,              # 最大迭代次数
    maxinner=200,             # 最大内部循环次数
    line_search=True,         # 是否进行线搜索
    warn=True,                # 是否输出警告信息
    verbose=0,                # 详细程度，控制输出信息的数量
    """
    Minimization of scalar function of one or more variables using the
    Newton-CG algorithm.

    Parameters
    ----------
    grad_hess : callable
        Should return the gradient and a callable returning the matvec product
        of the Hessian.

    func : callable
        Should return the value of the function.

    grad : callable
        Should return the function value and the gradient. This is used
        by the linesearch functions.

    x0 : array of float
        Initial guess.

    args : tuple, default=()
        Arguments passed to func_grad_hess, func and grad.

    tol : float, default=1e-4
        Stopping criterion. The iteration will stop when
        ``max{|g_i | i = 1, ..., n} <= tol``
        where ``g_i`` is the i-th component of the gradient.

    maxiter : int, default=100
        Number of Newton iterations.

    maxinner : int, default=200
        Number of CG iterations.

    line_search : bool, default=True
        Whether to use a line search or not.

    warn : bool, default=True
        Whether to warn when didn't converge.

    verbose : int, default=0
        Verbosity level. Controls the amount of output during optimization.

    Returns
    -------
    xk : ndarray of float
        Estimated minimum.
    """
    # 将 x0 转换为一维数组
    x0 = np.asarray(x0).flatten()
    # 复制初始值 x0 到 xk
    xk = np.copy(x0)
    # 迭代次数计数器
    k = 0

    # 如果开启了线搜索
    if line_search:
        # 计算初始点的函数值
        old_fval = func(x0, *args)
        old_old_fval = None
    else:
        old_fval = 0

    # 是否打印详细信息
    is_verbose = verbose > 0

    # 外层循环：Newton 迭代
    while k < maxiter:
        # 计算梯度和 Hessian 矢量乘积
        fgrad, fhess_p = grad_hess(xk, *args)

        # 计算梯度绝对值
        absgrad = np.abs(fgrad)
        # 计算最大梯度绝对值
        max_absgrad = np.max(absgrad)
        # 判断是否满足收敛条件
        check = max_absgrad <= tol
        # 如果需要打印详细信息，则输出当前迭代状态
        if is_verbose:
            print(f"Newton-CG iter = {k}")
            print("  Check Convergence")
            print(f"    max |gradient| <= tol: {max_absgrad} <= {tol} {check}")
        # 如果满足收敛条件，则跳出迭代
        if check:
            break

        # 计算梯度绝对值之和
        maggrad = np.sum(absgrad)
        # 计算 eta 参数
        eta = min([0.5, np.sqrt(maggrad)])
        # 计算终止条件
        termcond = eta * maggrad

        # 内层循环：通过共轭梯度法求解 Newton 更新，避免求逆 Hessian
        xsupi = _cg(fhess_p, fgrad, maxiter=maxinner, tol=termcond, verbose=verbose)

        # 设置默认的步长因子
        alphak = 1.0

        # 如果开启了线搜索
        if line_search:
            try:
                # 进行 Wolfe 线搜索，找到合适的步长
                alphak, fc, gc, old_fval, old_old_fval, gfkp1 = _line_search_wolfe12(
                    func,
                    grad,
                    xk,
                    xsupi,
                    fgrad,
                    old_fval,
                    old_old_fval,
                    verbose=verbose,
                    args=args,
                )
            except _LineSearchError:
                # 如果线搜索失败，则发出警告并跳出迭代
                warnings.warn("Line Search failed")
                break

        # 更新 xk
        xk += alphak * xsupi  # 如有需要，进行类型转换

        # 更新迭代次数
        k += 1
    # 如果开启警告并且迭代次数超过最大迭代次数，发出警告信息
    if warn and k >= maxiter:
        warnings.warn(
            (
                f"newton-cg failed to converge at loss = {old_fval}. Increase the"
                " number of iterations."
            ),
            ConvergenceWarning,
        )
    # 如果设置了详细输出标志，并且求解器收敛到指定损失值
    elif is_verbose:
        # 打印求解器成功收敛的损失值信息
        print(f"  Solver did converge at loss = {old_fval}.")
    # 返回计算出的解和迭代次数
    return xk, k
# 检查优化结果以确保收敛成功

def _check_optimize_result(solver, result, max_iter=None, extra_warning_msg=None):
    """Check the OptimizeResult for successful convergence

    Parameters
    ----------
    solver : str
       Solver name. Currently only `lbfgs` is supported.

    result : OptimizeResult
       Result of the scipy.optimize.minimize function.

    max_iter : int, default=None
       Expected maximum number of iterations.

    extra_warning_msg : str, default=None
        Extra warning message.

    Returns
    -------
    n_iter : int
       Number of iterations.
    """
    # 处理 scipy 和 scikit-learn 的求解器名称
    if solver == "lbfgs":
        # 如果优化未成功收敛
        if result.status != 0:
            try:
                # 在 scipy >= 1.6.0 中，消息已经解码
                result_message = result.message.decode("latin1")
            except AttributeError:
                result_message = result.message
            # 构建警告消息
            warning_msg = (
                "{} failed to converge (status={}):\n{}.\n\n"
                "Increase the number of iterations (max_iter) "
                "or scale the data as shown in:\n"
                "    https://scikit-learn.org/stable/modules/"
                "preprocessing.html"
            ).format(solver, result.status, result_message)
            # 如果有额外的警告消息，则添加到警告信息中
            if extra_warning_msg is not None:
                warning_msg += "\n" + extra_warning_msg
            # 发出警告
            warnings.warn(warning_msg, ConvergenceWarning, stacklevel=2)
        # 如果定义了最大迭代次数
        if max_iter is not None:
            # 在 scipy <= 1.0.0 中，lbfgs 的 nit 可能会超过 max_iter
            # 参见 https://github.com/scipy/scipy/issues/7854
            n_iter_i = min(result.nit, max_iter)
        else:
            n_iter_i = result.nit
    else:
        # 如果求解器不是 "lbfgs"，则引发未实现的错误
        raise NotImplementedError

    # 返回迭代次数
    return n_iter_i
```