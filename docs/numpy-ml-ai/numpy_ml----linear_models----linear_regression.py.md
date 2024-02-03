# `numpy-ml\numpy_ml\linear_models\linear_regression.py`

```
# 导入 numpy 库
import numpy as np

# 定义线性回归类
class LinearRegression:
    # 初始化加权线性最小二乘回归模型
    def __init__(self, fit_intercept=True):
        r"""
        A weighted linear least-squares regression model.

        Notes
        -----
        在加权线性最小二乘回归中，一个实值目标向量 **y** 被建模为协变量 **X** 和模型系数 :math:`\beta` 的线性组合：

        .. math::

            y_i = \beta^\top \mathbf{x}_i + \epsilon_i

        在这个方程中 :math:`\epsilon_i \sim \mathcal{N}(0, \sigma^2_i)` 是与示例 :math:`i` 相关的误差项，而 :math:`\sigma^2_i` 是相应示例的方差。

        在这个模型下，回归系数 :math:`\beta` 的最大似然估计为：

        .. math::

            \hat{\beta} = \Sigma^{-1} \mathbf{X}^\top \mathbf{Wy}

        其中 :math:`\Sigma^{-1} = (\mathbf{X}^\top \mathbf{WX})^{-1}`，**W** 是一个权重对角矩阵，每个条目与相应测量的方差成反比。当 **W** 是单位矩阵时，示例被等权重处理，模型简化为标准线性最小二乘 [2]_。

        参考资料
        ----------
        .. [1] https://en.wikipedia.org/wiki/Weighted_least_squares
        .. [2] https://en.wikipedia.org/wiki/General_linear_model

        参数
        ----------
        fit_intercept : bool
            是否在模型系数之外拟合一个截距项。默认为 True。

        属性
        ----------
        beta : :py:class:`ndarray <numpy.ndarray>` of shape `(M, K)` or None
            拟合的模型系数。
        sigma_inv : :py:class:`ndarray <numpy.ndarray>` of shape `(N, N)` or None
            数据协方差矩阵的逆矩阵。
        """
        # 初始化模型系数和数据协方差矩阵的逆矩阵
        self.beta = None
        self.sigma_inv = None
        # 设置是否拟合截距项
        self.fit_intercept = fit_intercept

        # 标记模型是否已经拟合
        self._is_fit = False
    # 对单个样本进行Sherman-Morrison更新
    def _update1D(self, x, y, w):
        """Sherman-Morrison update for a single example"""
        beta, S_inv = self.beta, self.sigma_inv

        # 如果拟合截距，将x转换为设计向量
        if self.fit_intercept:
            x = np.c_[np.diag(w), x]

        # 通过Sherman-Morrison更新协方差矩阵的逆
        S_inv -= (S_inv @ x.T @ x @ S_inv) / (1 + x @ S_inv @ x.T)

        # 更新模型系数
        beta += S_inv @ x.T @ (y - x @ beta)

    # 对多个样本进行Woodbury更新
    def _update2D(self, X, y, W):
        """Woodbury update for multiple examples"""
        beta, S_inv = self.beta, self.sigma_inv

        # 如果拟合截距，将X转换为设计矩阵
        if self.fit_intercept:
            X = np.c_[np.diag(W), X]

        I = np.eye(X.shape[0])  # noqa: E741

        # 通过Woodbury恒等式更新协方差矩阵的逆
        S_inv -= S_inv @ X.T @ np.linalg.pinv(I + X @ S_inv @ X.T) @ X @ S_inv

        # 更新模型系数
        beta += S_inv @ X.T @ (y - X @ beta)
    def fit(self, X, y, weights=None):
        r"""
        Fit regression coefficients via maximum likelihood.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(N, M)`
            A dataset consisting of `N` examples, each of dimension `M`.
        y : :py:class:`ndarray <numpy.ndarray>` of shape `(N, K)`
            The targets for each of the `N` examples in `X`, where each target
            has dimension `K`.
        weights : :py:class:`ndarray <numpy.ndarray>` of shape `(N,)` or None
            Weights associated with the examples in `X`. Examples
            with larger weights exert greater influence on model fit.  When
            `y` is a vector (i.e., `K = 1`), weights should be set to the
            reciporical of the variance for each measurement (i.e., :math:`w_i
            = 1/\sigma^2_i`). When `K > 1`, it is assumed that all columns of
            `y` share the same weight :math:`w_i`. If None, examples are
            weighted equally, resulting in the standard linear least squares
            update.  Default is None.

        Returns
        -------
        self : :class:`LinearRegression <numpy_ml.linear_models.LinearRegression>` instance
        """  # noqa: E501
        N = X.shape[0]

        # 如果 weights 为 None，则将权重设置为全为 1 的数组
        weights = np.ones(N) if weights is None else np.atleast_1d(weights)
        # 如果 weights 不是一维数组，则将其转换为一维数组
        weights = np.squeeze(weights) if weights.size > 1 else weights
        err_str = f"weights must have shape ({N},) but got {weights.shape}"
        # 检查权重数组的形状是否为 (N,)，如果不是则抛出异常
        assert weights.shape == (N,), err_str

        # 根据权重调整 X 和 y 的值
        W = np.diag(np.sqrt(weights))
        X, y = W @ X, W @ y

        # 如果需要拟合截距项，则将 X 转换为设计矩阵
        if self.fit_intercept:
            X = np.c_[np.sqrt(weights), X]

        # 计算最大似然估计的回归系数
        self.sigma_inv = np.linalg.pinv(X.T @ X)
        self.beta = np.atleast_2d(self.sigma_inv @ X.T @ y)

        self._is_fit = True
        return self
    # 使用训练好的模型在新的数据集上生成预测结果

    # 将参数 X 转换为设计矩阵，如果我们正在拟合一个截距
    if self.fit_intercept:
        X = np.c_[np.ones(X.shape[0]), X]
    
    # 返回 X 与模型参数 beta 的矩阵乘积作为预测结果
    return X @ self.beta
```