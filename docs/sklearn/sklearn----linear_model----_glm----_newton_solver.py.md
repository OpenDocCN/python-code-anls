# `D:\src\scipysrc\scikit-learn\sklearn\linear_model\_glm\_newton_solver.py`

```
# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause
"""
Newton solver for Generalized Linear Models
"""

import warnings
from abc import ABC, abstractmethod

import numpy as np
import scipy.linalg
import scipy.optimize

from ..._loss.loss import HalfSquaredError
from ...exceptions import ConvergenceWarning
from ...utils.optimize import _check_optimize_result
from .._linear_loss import LinearModelLoss


class NewtonSolver(ABC):
    """Newton solver for GLMs.

    This class implements Newton/2nd-order optimization routines for GLMs. Each Newton
    iteration aims at finding the Newton step which is done by the inner solver. With
    Hessian H, gradient g and coefficients coef, one step solves:

        H @ coef_newton = -g

    For our GLM / LinearModelLoss, we have gradient g and Hessian H:

        g = X.T @ loss.gradient + l2_reg_strength * coef
        H = X.T @ diag(loss.hessian) @ X + l2_reg_strength * identity

    Backtracking line search updates coef = coef_old + t * coef_newton for some t in
    (0, 1].

    This is a base class, actual implementations (child classes) may deviate from the
    above pattern and use structure specific tricks.

    Usage pattern:
        - initialize solver: sol = NewtonSolver(...)
        - solve the problem: sol.solve(X, y, sample_weight)

    References
    ----------
    - Jorge Nocedal, Stephen J. Wright. (2006) "Numerical Optimization"
      2nd edition
      https://doi.org/10.1007/978-0-387-40065-5

    - Stephen P. Boyd, Lieven Vandenberghe. (2004) "Convex Optimization."
      Cambridge University Press, 2004.
      https://web.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf

    Parameters
    ----------
    coef : ndarray of shape (n_dof,), (n_classes, n_dof) or (n_classes * n_dof,)
        Initial coefficients of a linear model.
        If shape (n_classes * n_dof,), the classes of one feature are contiguous,
        i.e. one reconstructs the 2d-array via
        coef.reshape((n_classes, -1), order="F").

    linear_loss : LinearModelLoss
        The loss to be minimized.

    l2_reg_strength : float, default=0.0
        L2 regularization strength.

    tol : float, default=1e-4
        The optimization problem is solved when each of the following condition is
        fulfilled:
        1. maximum |gradient| <= tol
        2. Newton decrement d: 1/2 * d^2 <= tol

    max_iter : int, default=100
        Maximum number of Newton steps allowed.

    n_threads : int, default=1
        Number of OpenMP threads to use for the computation of the Hessian and gradient
        of the loss function.

    Attributes
    ----------
    coef_old : ndarray of shape coef.shape
        Coefficient of previous iteration.

    coef_newton : ndarray of shape coef.shape
        Newton step.

    gradient : ndarray of shape coef.shape
        Gradient of the loss w.r.t. the coefficients.

    gradient_old : ndarray of shape coef.shape
        Gradient of previous iteration.
"""
    loss_value : float
        # 目标函数的值，即损失加上惩罚项。

    loss_value_old : float
        # 上一次迭代的目标函数值。

    raw_prediction : ndarray of shape (n_samples,) or (n_samples, n_classes)
        # 原始预测结果数组，形状可以是 (样本数,) 或者 (样本数, 类别数)。

    converged : bool
        # 求解器收敛的指示器。

    iteration : int
        # 牛顿步数，即调用 inner_solve 的次数。

    use_fallback_lbfgs_solve : bool
        # 如果设置为 True，在收敛出现问题时，求解器将调用 LBFGS 来完成优化过程。

    gradient_times_newton : float
        # gradient @ coef_newton，在 inner_solve 中设置，并由 line_search 使用。
        # 如果牛顿步是一个下降方向，这个值是负数。
    """

    def __init__(
        self,
        *,
        coef,
        linear_loss=LinearModelLoss(base_loss=HalfSquaredError(), fit_intercept=True),
        l2_reg_strength=0.0,
        tol=1e-4,
        max_iter=100,
        n_threads=1,
        verbose=0,
    ):
        # 初始化方法，设置对象的初始属性。
        self.coef = coef
        self.linear_loss = linear_loss
        self.l2_reg_strength = l2_reg_strength
        self.tol = tol
        self.max_iter = max_iter
        self.n_threads = n_threads
        self.verbose = verbose

    def setup(self, X, y, sample_weight):
        """Precomputations

        If None, initializes:
            - self.coef
        Sets:
            - self.raw_prediction
            - self.loss_value
        """
        # 预计算方法，初始化 coef，设置 raw_prediction 和 loss_value 属性。
        _, _, self.raw_prediction = self.linear_loss.weight_intercept_raw(self.coef, X)
        self.loss_value = self.linear_loss.loss(
            coef=self.coef,
            X=X,
            y=y,
            sample_weight=sample_weight,
            l2_reg_strength=self.l2_reg_strength,
            n_threads=self.n_threads,
            raw_prediction=self.raw_prediction,
        )

    @abstractmethod
    def update_gradient_hessian(self, X, y, sample_weight):
        """Update gradient and Hessian."""
        # 更新梯度和 Hessian 矩阵的抽象方法。

    @abstractmethod
    def inner_solve(self, X, y, sample_weight):
        """Compute Newton step.

        Sets:
            - self.coef_newton
            - self.gradient_times_newton
        """
        # 计算牛顿步的抽象方法，设置 self.coef_newton 和 self.gradient_times_newton 属性。
    def fallback_lbfgs_solve(self, X, y, sample_weight):
        """Fallback solver in case of emergency.

        If a solver detects convergence problems, it may fall back to this method in
        the hope to exit with success instead of raising an error.

        Sets:
            - self.coef: Updated coefficient vector after optimization
            - self.converged: Boolean flag indicating convergence status
        """
        # 使用L-BFGS-B方法进行优化，最大迭代次数为self.max_iter，jac=True表示提供了梯度函数
        opt_res = scipy.optimize.minimize(
            self.linear_loss.loss_gradient,  # 优化的损失梯度函数
            self.coef,  # 初始系数
            method="L-BFGS-B",  # 优化方法
            jac=True,  # 表示提供了梯度函数
            options={
                "maxiter": self.max_iter,  # 最大迭代次数
                "maxls": 50,  # 最大线搜索次数，默认为20
                "iprint": self.verbose - 1,  # 控制输出信息详细程度
                "gtol": self.tol,  # 梯度范数的容忍度
                "ftol": 64 * np.finfo(np.float64).eps,  # 函数值变化的容忍度
            },
            args=(X, y, sample_weight, self.l2_reg_strength, self.n_threads),  # 传入参数
        )
        # 检查优化结果并更新迭代次数
        self.n_iter_ = _check_optimize_result("lbfgs", opt_res)
        # 更新系数
        self.coef = opt_res.x
        # 设置收敛状态，优化成功则为True
        self.converged = opt_res.status == 0

    def check_convergence(self, X, y, sample_weight):
        """Check for convergence.

        Sets self.converged.
        """
        if self.verbose:
            print("  Check Convergence")
        # 注意：使用最大梯度绝对值作为收敛判断标准，这种方法可能不太好
        # coef_step = self.coef - self.coef_old
        # check = np.max(np.abs(coef_step) / np.maximum(1, np.abs(self.coef_old)))

        # 1. Criterion: maximum |gradient| <= tol
        #    The gradient was already updated in line_search()
        # 计算当前梯度的最大绝对值
        check = np.max(np.abs(self.gradient))
        if self.verbose:
            print(f"    1. max |gradient| {check} <= {self.tol}")
        # 如果梯度的最大绝对值大于tol，则返回，表示未收敛
        if check > self.tol:
            return

        # 2. Criterion: For Newton decrement d, check 1/2 * d^2 <= tol
        #       d = sqrt(grad @ hessian^-1 @ grad)
        #         = sqrt(coef_newton @ hessian @ coef_newton)
        #    参见Boyd，Vanderberghe (2009)的"Convex Optimization"第9.5.1章节。
        # 计算Newton减量的平方
        d2 = self.coef_newton @ self.hessian @ self.coef_newton
        if self.verbose:
            print(f"    2. Newton decrement {0.5 * d2} <= {self.tol}")
        # 如果Newton减量的一半大于tol，则返回，表示未收敛
        if 0.5 * d2 > self.tol:
            return

        if self.verbose:
            # 输出当前损失值，用于确认优化是否收敛
            loss_value = self.linear_loss.loss(
                coef=self.coef,
                X=X,
                y=y,
                sample_weight=sample_weight,
                l2_reg_strength=self.l2_reg_strength,
                n_threads=self.n_threads,
            )
            print(f"  Solver did converge at loss = {loss_value}.")
        # 设置收敛状态为True
        self.converged = True

    def finalize(self, X, y, sample_weight):
        """Finalize the solvers results.

        Some solvers may need this, others not.
        """
        # 这个方法没有实际操作，留空作为占位符
        pass
    # NewtonCholeskySolver 类，继承自 NewtonSolver，用于基于 Cholesky 分解的 Newton 求解器
    class NewtonCholeskySolver(NewtonSolver):
        
        # 初始化方法，设置求解器参数和属性
        """Cholesky based Newton solver.
        
        Inner solver for finding the Newton step H w_newton = -g uses Cholesky based linear
        solver.
        """
        def setup(self, X, y, sample_weight):
            # 调用父类 NewtonSolver 的 setup 方法，设置数据和权重
            super().setup(X=X, y=y, sample_weight=sample_weight)
            # 确定自由度数量为特征矩阵 X 的列数
            n_dof = X.shape[1]
            # 如果使用了 fit_intercept，则自由度数量加一
            if self.linear_loss.fit_intercept:
                n_dof += 1
            # 初始化梯度数组，与系数数组相同的形状
            self.gradient = np.empty_like(self.coef)
            # 初始化 Hessian 矩阵，形状为 n_dof x n_dof，与系数数组相同的形状
            self.hessian = np.empty_like(self.coef, shape=(n_dof, n_dof))

        # 更新梯度和 Hessian 矩阵的方法
        def update_gradient_hessian(self, X, y, sample_weight):
            # 调用 linear_loss 的 gradient_hessian 方法获取梯度和 Hessian 矩阵，
            # 并记录可能的警告信息到 hessian_warning 中
            _, _, self.hessian_warning = self.linear_loss.gradient_hessian(
                coef=self.coef,
                X=X,
                y=y,
                sample_weight=sample_weight,
                l2_reg_strength=self.l2_reg_strength,
                n_threads=self.n_threads,
                gradient_out=self.gradient,
                hessian_out=self.hessian,
                raw_prediction=self.raw_prediction,  # this was updated in line_search
            )
```