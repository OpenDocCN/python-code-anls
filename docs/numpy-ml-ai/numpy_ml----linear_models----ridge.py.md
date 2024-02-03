# `numpy-ml\numpy_ml\linear_models\ridge.py`

```
# 导入 numpy 库
import numpy as np

# 定义 RidgeRegression 类，用于执行岭回归
class RidgeRegression:
    # 初始化 Ridge 回归模型对象
    def __init__(self, alpha=1, fit_intercept=True):
        # Ridge 回归模型通过正规方程最大似然估计
        r"""
        A ridge regression model with maximum likelihood fit via the normal
        equations.

        Notes
        -----
        Ridge regression is a biased estimator for linear models which adds an
        additional penalty proportional to the L2-norm of the model
        coefficients to the standard mean-squared-error loss:

        .. math::

            \mathcal{L}_{Ridge} = (\mathbf{y} - \mathbf{X} \beta)^\top
                (\mathbf{y} - \mathbf{X} \beta) + \alpha ||\beta||_2^2

        where :math:`\alpha` is a weight controlling the severity of the
        penalty.

        Given data matrix **X** and target vector **y**, the maximum-likelihood
        estimate for ridge coefficients, :math:`\beta`, is:

        .. math::

            \hat{\beta} =
                \left(\mathbf{X}^\top \mathbf{X} + \alpha \mathbf{I} \right)^{-1}
                    \mathbf{X}^\top \mathbf{y}

        It turns out that this estimate for :math:`\beta` also corresponds to
        the MAP estimate if we assume a multivariate Gaussian prior on the
        model coefficients, assuming that the data matrix **X** has been
        standardized and the target values **y** centered at 0:

        .. math::

            \beta \sim \mathcal{N}\left(\mathbf{0}, \frac{1}{2M} \mathbf{I}\right)

        Parameters
        ----------
        alpha : float
            L2 regularization coefficient. Larger values correspond to larger
            penalty on the L2 norm of the model coefficients. Default is 1.
        fit_intercept : bool
            Whether to fit an additional intercept term. Default is True.

        Attributes
        ----------
        beta : :py:class:`ndarray <numpy.ndarray>` of shape `(M, K)` or None
            Fitted model coefficients.
        """
        # 初始化模型系数和超参数
        self.beta = None
        self.alpha = alpha
        self.fit_intercept = fit_intercept
    def fit(self, X, y):
        """
        Fit the regression coefficients via maximum likelihood.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(N, M)`
            A dataset consisting of `N` examples, each of dimension `M`.
        y : :py:class:`ndarray <numpy.ndarray>` of shape `(N, K)`
            The targets for each of the `N` examples in `X`, where each target
            has dimension `K`.

        Returns
        -------
        self : :class:`RidgeRegression <numpy_ml.linear_models.RidgeRegression>` instance
        """  # noqa: E501
        # 如果需要拟合截距，将 X 转换为设计矩阵
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        # 创建正则化矩阵 A
        A = self.alpha * np.eye(X.shape[1])
        # 计算伪逆矩阵
        pseudo_inverse = np.linalg.inv(X.T @ X + A) @ X.T
        # 计算回归系数 beta
        self.beta = pseudo_inverse @ y
        # 返回 self 对象
        return self

    def predict(self, X):
        """
        Use the trained model to generate predictions on a new collection of
        data points.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(Z, M)`
            A dataset consisting of `Z` new examples, each of dimension `M`.

        Returns
        -------
        y_pred : :py:class:`ndarray <numpy.ndarray>` of shape `(Z, K)`
            The model predictions for the items in `X`.
        """
        # 如果需要拟合截距，将 X 转换为设计矩阵
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        # 返回预测结果
        return np.dot(X, self.beta)
```