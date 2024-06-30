# `D:\src\scipysrc\scikit-learn\sklearn\linear_model\_linear_loss.py`

```
"""
Loss functions for linear models with raw_prediction = X @ coef
"""

# 导入必要的库
import numpy as np  # 导入 NumPy 库
from scipy import sparse  # 导入 SciPy 库中的稀疏矩阵支持

# 导入外部自定义模块
from ..utils.extmath import squared_norm  # 从上级目录中的 utils.extmath 模块导入 squared_norm 函数


class LinearModelLoss:
    """General class for loss functions with raw_prediction = X @ coef + intercept.

    Note that raw_prediction is also known as linear predictor.

    The loss is the average of per sample losses and includes a term for L2
    regularization::

        loss = 1 / s_sum * sum_i s_i loss(y_i, X_i @ coef + intercept)
               + 1/2 * l2_reg_strength * ||coef||_2^2

    with sample weights s_i=1 if sample_weight=None and s_sum=sum_i s_i.

    Gradient and hessian, for simplicity without intercept, are::

        gradient = 1 / s_sum * X.T @ loss.gradient + l2_reg_strength * coef
        hessian = 1 / s_sum * X.T @ diag(loss.hessian) @ X
                  + l2_reg_strength * identity

    Conventions:
        if fit_intercept:
            n_dof =  n_features + 1
        else:
            n_dof = n_features

        if base_loss.is_multiclass:
            coef.shape = (n_classes, n_dof) or ravelled (n_classes * n_dof,)
        else:
            coef.shape = (n_dof,)

        The intercept term is at the end of the coef array:
        if base_loss.is_multiclass:
            if coef.shape (n_classes, n_dof):
                intercept = coef[:, -1]
            if coef.shape (n_classes * n_dof,)
                intercept = coef[n_features::n_dof] = coef[(n_dof-1)::n_dof]
            intercept.shape = (n_classes,)
        else:
            intercept = coef[-1]

    Note: If coef has shape (n_classes * n_dof,), the 2d-array can be reconstructed as

        coef.reshape((n_classes, -1), order="F")

    The option order="F" makes coef[:, i] contiguous. This, in turn, makes the
    coefficients without intercept, coef[:, :-1], contiguous and speeds up
    matrix-vector computations.

    Note: If the average loss per sample is wanted instead of the sum of the loss per
    sample, one can simply use a rescaled sample_weight such that
    sum(sample_weight) = 1.

    Parameters
    ----------
    base_loss : instance of class BaseLoss from sklearn._loss.
        The loss function to be optimized.
    fit_intercept : bool
        Whether to fit an intercept term in the model.
    """

    def __init__(self, base_loss, fit_intercept):
        self.base_loss = base_loss  # 初始化损失函数对象
        self.fit_intercept = fit_intercept  # 初始化是否拟合截距参数
    def init_zero_coef(self, X, dtype=None):
        """Allocate coef of correct shape with zeros.

        Parameters:
        -----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.
        dtype : data-type, default=None
            Overrides the data type of coef. With dtype=None, coef will have the same
            dtype as X.

        Returns
        -------
        coef : ndarray of shape (n_dof,) or (n_classes, n_dof)
            Coefficients of a linear model.
        """
        # 获取特征数
        n_features = X.shape[1]
        # 获取类别数
        n_classes = self.base_loss.n_classes
        # 如果需要拟合截距，增加一个自由度
        if self.fit_intercept:
            n_dof = n_features + 1
        else:
            n_dof = n_features
        # 如果是多类别问题，创建一个形状为 (n_classes, n_dof) 的全零数组
        if self.base_loss.is_multiclass:
            coef = np.zeros_like(X, shape=(n_classes, n_dof), dtype=dtype, order="F")
        else:
            # 否则创建形状为 (n_dof,) 的全零数组
            coef = np.zeros_like(X, shape=n_dof, dtype=dtype)
        # 返回初始化后的 coef 数组
        return coef

    def weight_intercept(self, coef):
        """Helper function to get coefficients and intercept.

        Parameters
        ----------
        coef : ndarray of shape (n_dof,), (n_classes, n_dof) or (n_classes * n_dof,)
            Coefficients of a linear model.
            If shape (n_classes * n_dof,), the classes of one feature are contiguous,
            i.e. one reconstructs the 2d-array via
            coef.reshape((n_classes, -1), order="F").

        Returns
        -------
        weights : ndarray of shape (n_features,) or (n_classes, n_features)
            Coefficients without intercept term.
        intercept : float or ndarray of shape (n_classes,)
            Intercept terms.
        """
        # 如果不是多类别问题
        if not self.base_loss.is_multiclass:
            # 如果拟合截距，最后一个元素作为截距，其余作为权重
            if self.fit_intercept:
                intercept = coef[-1]
                weights = coef[:-1]
            else:
                # 否则截距为 0，所有元素作为权重
                intercept = 0.0
                weights = coef
        else:
            # reshape 到 (n_classes, n_dof)
            if coef.ndim == 1:
                weights = coef.reshape((self.base_loss.n_classes, -1), order="F")
            else:
                weights = coef
            # 如果拟合截距，每行的最后一个元素作为截距，其余作为权重
            if self.fit_intercept:
                intercept = weights[:, -1]
                weights = weights[:, :-1]
            else:
                # 否则截距为 0
                intercept = 0.0

        return weights, intercept
    def weight_intercept_raw(self, coef, X):
        """Helper function to get coefficients, intercept and raw_prediction.

        Parameters
        ----------
        coef : ndarray of shape (n_dof,), (n_classes, n_dof) or (n_classes * n_dof,)
            Coefficients of a linear model.
            If shape (n_classes * n_dof,), the classes of one feature are contiguous,
            i.e. one reconstructs the 2d-array via
            coef.reshape((n_classes, -1), order="F").
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.

        Returns
        -------
        weights : ndarray of shape (n_features,) or (n_classes, n_features)
            Coefficients without intercept term.
        intercept : float or ndarray of shape (n_classes,)
            Intercept terms.
        raw_prediction : ndarray of shape (n_samples,) or (n_samples, n_classes)
            Raw predictions computed from the linear model.
        """
        weights, intercept = self.weight_intercept(coef)

        if not self.base_loss.is_multiclass:
            # Calculate raw prediction for non-multiclass case
            raw_prediction = X @ weights + intercept
        else:
            # For multiclass, weights has shape (n_classes, n_dof)
            # Compute raw prediction using matrix multiplication
            raw_prediction = X @ weights.T + intercept  # ndarray, likely C-contiguous

        return weights, intercept, raw_prediction

    def l2_penalty(self, weights, l2_reg_strength):
        """Compute L2 penalty term l2_reg_strength/2 *||w||_2^2.

        Parameters
        ----------
        weights : ndarray
            Coefficients of the linear model.
        l2_reg_strength : float
            L2 regularization strength.

        Returns
        -------
        l2_penalty_term : float
            L2 regularization penalty term.
        """
        norm2_w = weights @ weights if weights.ndim == 1 else squared_norm(weights)
        # Calculate L2 penalty term
        return 0.5 * l2_reg_strength * norm2_w

    def loss(
        self,
        coef,
        X,
        y,
        sample_weight=None,
        l2_reg_strength=0.0,
        n_threads=1,
        raw_prediction=None,
    ):
        """Compute the loss function of the linear model.

        Parameters
        ----------
        coef : ndarray
            Coefficients of the linear model.
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.
        y : ndarray of shape (n_samples,)
            Target values (labels).
        sample_weight : ndarray of shape (n_samples,), optional
            Sample weights.
        l2_reg_strength : float, optional, default=0.0
            L2 regularization strength.
        n_threads : int, optional, default=1
            Number of threads to use.
        raw_prediction : ndarray of shape (n_samples,) or (n_samples, n_classes), optional
            Raw predictions computed from the linear model.

        Returns
        -------
        loss_value : float
            Computed loss value.
        """
    ):
        """Compute the loss as weighted average over point-wise losses.

        Parameters
        ----------
        coef : ndarray of shape (n_dof,), (n_classes, n_dof) or (n_classes * n_dof,)
            Coefficients of a linear model.
            If shape (n_classes * n_dof,), the classes of one feature are contiguous,
            i.e. one reconstructs the 2d-array via
            coef.reshape((n_classes, -1), order="F").
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.
        y : contiguous array of shape (n_samples,)
            Observed, true target values.
        sample_weight : None or contiguous array of shape (n_samples,), default=None
            Sample weights.
        l2_reg_strength : float, default=0.0
            L2 regularization strength
        n_threads : int, default=1
            Number of OpenMP threads to use.
        raw_prediction : C-contiguous array of shape (n_samples,) or array of \
            shape (n_samples, n_classes)
            Raw prediction values (in link space). If provided, these are used. If
            None, then raw_prediction = X @ coef + intercept is calculated.

        Returns
        -------
        loss : float
            Weighted average of losses per sample, plus penalty.
        """
        if raw_prediction is None:
            # 如果没有提供原始预测值，则通过权重和截距计算原始预测值
            weights, intercept, raw_prediction = self.weight_intercept_raw(coef, X)
        else:
            # 如果提供了原始预测值，则通过权重和截距计算权重
            weights, intercept = self.weight_intercept(coef)

        # 计算损失，使用基础损失函数计算每个样本的损失
        loss = self.base_loss.loss(
            y_true=y,
            raw_prediction=raw_prediction,
            sample_weight=None,  # 这里sample_weight被设为None，可能需要根据具体情况修改
            n_threads=n_threads,
        )
        # 计算加权平均损失
        loss = np.average(loss, weights=sample_weight)

        # 返回加上L2正则化惩罚项的最终损失值
        return loss + self.l2_penalty(weights, l2_reg_strength)

    def loss_gradient(
        self,
        coef,
        X,
        y,
        sample_weight=None,
        l2_reg_strength=0.0,
        n_threads=1,
        raw_prediction=None,
    def _loss_and_grad(
            """
            Computes the sum of loss and gradient w.r.t. coef.
    
            Parameters
            ----------
            coef : ndarray of shape (n_dof,), (n_classes, n_dof) or (n_classes * n_dof,)
                Coefficients of a linear model.
                If shape (n_classes * n_dof,), the classes of one feature are contiguous,
                i.e. one reconstructs the 2d-array via
                coef.reshape((n_classes, -1), order="F").
            X : {array-like, sparse matrix} of shape (n_samples, n_features)
                Training data.
            y : contiguous array of shape (n_samples,)
                Observed, true target values.
            sample_weight : None or contiguous array of shape (n_samples,), default=None
                Sample weights.
            l2_reg_strength : float, default=0.0
                L2 regularization strength
            n_threads : int, default=1
                Number of OpenMP threads to use.
            raw_prediction : C-contiguous array of shape (n_samples,) or array of \
                shape (n_samples, n_classes)
                Raw prediction values (in link space). If provided, these are used. If
                None, then raw_prediction = X @ coef + intercept is calculated.
    
            Returns
            -------
            loss : float
                Weighted average of losses per sample, plus penalty.
    
            gradient : ndarray of shape coef.shape
                 The gradient of the loss.
            """
            (n_samples, n_features), n_classes = X.shape, self.base_loss.n_classes
            n_dof = n_features + int(self.fit_intercept)
    
            if raw_prediction is None:
                weights, intercept, raw_prediction = self.weight_intercept_raw(coef, X)
            else:
                weights, intercept = self.weight_intercept(coef)
    
            # Compute loss and pointwise gradient using base_loss object
            loss, grad_pointwise = self.base_loss.loss_gradient(
                y_true=y,
                raw_prediction=raw_prediction,
                sample_weight=sample_weight,
                n_threads=n_threads,
            )
    
            # Compute sum of sample weights if present
            sw_sum = n_samples if sample_weight is None else np.sum(sample_weight)
            # Calculate average loss per sample and add L2 regularization penalty
            loss = loss.sum() / sw_sum
            loss += self.l2_penalty(weights, l2_reg_strength)
    
            # Normalize gradient by sum of sample weights
            grad_pointwise /= sw_sum
    
            if not self.base_loss.is_multiclass:
                # If not multiclass, compute gradient for each feature and intercept
                grad = np.empty_like(coef, dtype=weights.dtype)
                grad[:n_features] = X.T @ grad_pointwise + l2_reg_strength * weights
                if self.fit_intercept:
                    grad[-1] = grad_pointwise.sum()
            else:
                # If multiclass, compute gradient for each class and feature, and intercept
                grad = np.empty((n_classes, n_dof), dtype=weights.dtype, order="F")
                grad[:, :n_features] = grad_pointwise.T @ X + l2_reg_strength * weights
                if self.fit_intercept:
                    grad[:, -1] = grad_pointwise.sum(axis=0)
                if coef.ndim == 1:
                    grad = grad.ravel(order="F")
    
            return loss, grad
    def gradient(
        self,
        coef,
        X,
        y,
        sample_weight=None,
        l2_reg_strength=0.0,
        n_threads=1,
        raw_prediction=None,
    ):
        """Computes the gradient w.r.t. coef.

        Parameters
        ----------
        coef : ndarray of shape (n_dof,), (n_classes, n_dof) or (n_classes * n_dof,)
            Coefficients of a linear model.
            If shape (n_classes * n_dof,), the classes of one feature are contiguous,
            i.e. one reconstructs the 2d-array via
            coef.reshape((n_classes, -1), order="F").
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training data.
        y : contiguous array of shape (n_samples,)
            Observed, true target values.
        sample_weight : None or contiguous array of shape (n_samples,), default=None
            Sample weights.
        l2_reg_strength : float, default=0.0
            L2 regularization strength
        n_threads : int, default=1
            Number of OpenMP threads to use.
        raw_prediction : C-contiguous array of shape (n_samples,) or array of \
            shape (n_samples, n_classes)
            Raw prediction values (in link space). If provided, these are used. If
            None, then raw_prediction = X @ coef + intercept is calculated.

        Returns
        -------
        gradient : ndarray of shape coef.shape
             The gradient of the loss.
        """
        # Determine the dimensions of the input data and coefficients
        (n_samples, n_features), n_classes = X.shape, self.base_loss.n_classes
        n_dof = n_features + int(self.fit_intercept)

        # Calculate weights, intercept, and raw_prediction if raw_prediction is None
        if raw_prediction is None:
            weights, intercept, raw_prediction = self.weight_intercept_raw(coef, X)
        else:
            weights, intercept = self.weight_intercept(coef)

        # Compute the pointwise gradient of the base loss function
        grad_pointwise = self.base_loss.gradient(
            y_true=y,
            raw_prediction=raw_prediction,
            sample_weight=sample_weight,
            n_threads=n_threads,
        )

        # Calculate sum of sample weights if sample_weight is provided
        sw_sum = n_samples if sample_weight is None else np.sum(sample_weight)
        grad_pointwise /= sw_sum  # Normalize gradient by sample weight sum

        # Compute gradient based on whether it's a multiclass or not
        if not self.base_loss.is_multiclass:
            grad = np.empty_like(coef, dtype=weights.dtype)
            grad[:n_features] = X.T @ grad_pointwise + l2_reg_strength * weights
            if self.fit_intercept:
                grad[-1] = grad_pointwise.sum()  # Include intercept gradient
            return grad
        else:
            grad = np.empty((n_classes, n_dof), dtype=weights.dtype, order="F")
            # gradient.shape = (n_classes, n_dof)
            grad[:, :n_features] = grad_pointwise.T @ X + l2_reg_strength * weights
            if self.fit_intercept:
                grad[:, -1] = grad_pointwise.sum(axis=0)  # Include intercept gradient
            if coef.ndim == 1:
                return grad.ravel(order="F")  # Return flattened gradient
            else:
                return grad  # Return gradient for multiclass case
    # 计算梯度和黑塞矩阵的方法，用于训练模型中的参数
    def gradient_hessian(
        self,
        coef,
        X,
        y,
        sample_weight=None,
        l2_reg_strength=0.0,
        n_threads=1,
        gradient_out=None,
        hessian_out=None,
        raw_prediction=None,
    ):
        # 在给定的数据集(X, y)上计算梯度和黑塞矩阵
        # coef: 当前模型的参数向量
        # X: 训练数据集的特征矩阵
        # y: 训练数据集的标签向量
        # sample_weight: 样本权重，默认为None
        # l2_reg_strength: L2 正则化的强度，默认为0.0
        # n_threads: 并行计算的线程数，默认为1
        # gradient_out: 输出参数梯度的数组，如果为None则不输出
        # hessian_out: 输出黑塞矩阵的数组，如果为None则不输出
        # raw_prediction: 原始预测结果的数组，如果为None则不输出
        pass  # 方法主体未提供，使用 pass 占位符
    
    # 计算梯度和黑塞矩阵乘积的方法，用于训练模型中的参数
    def gradient_hessian_product(
        self,
        coef,
        X,
        y,
        sample_weight=None,
        l2_reg_strength=0.0,
        n_threads=1
    ):
        # 在给定的数据集(X, y)上计算梯度和黑塞矩阵的乘积
        # coef: 当前模型的参数向量
        # X: 训练数据集的特征矩阵
        # y: 训练数据集的标签向量
        # sample_weight: 样本权重，默认为None
        # l2_reg_strength: L2 正则化的强度，默认为0.0
        # n_threads: 并行计算的线程数，默认为1
        pass  # 方法主体未提供，使用 pass 占位符
```