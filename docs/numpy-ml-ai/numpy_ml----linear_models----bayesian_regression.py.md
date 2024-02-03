# `numpy-ml\numpy_ml\linear_models\bayesian_regression.py`

```py
# 导入必要的库：numpy 和 scipy.stats
import numpy as np
import scipy.stats as stats

# 从自定义的测试模块中导入 is_number 和 is_symmetric_positive_definite 函数
from numpy_ml.utils.testing import is_number, is_symmetric_positive_definite

# 定义一个类 BayesianLinearRegressionUnknownVariance，用于贝叶斯线性回归模型
class BayesianLinearRegressionUnknownVariance:
    def fit(self, X, y):
        """
        Compute the posterior over model parameters using the data in `X` and
        `y`.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(N, M)`
            A dataset consisting of `N` examples, each of dimension `M`.
        y : :py:class:`ndarray <numpy.ndarray>` of shape `(N, K)`
            The targets for each of the `N` examples in `X`, where each target
            has dimension `K`.

        Returns
        -------
        self : :class:`BayesianLinearRegressionUnknownVariance<numpy_ml.linear_models.BayesianLinearRegressionUnknownVariance>` instance
        """  # noqa: E501
        # 如果需要拟合截距，将 X 转换为设计矩阵
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        N, M = X.shape
        alpha, beta, V, mu = self.alpha, self.beta, self.V, self.mu

        if is_number(V):
            # 如果 V 是数字，将其转换为对角矩阵
            V *= np.eye(M)

        if is_number(mu):
            # 如果 mu 是数字，将其转换为长度为 M 的数组
            mu *= np.ones(M)

        # 计算 sigma
        I = np.eye(N)  # noqa: E741
        a = y - (X @ mu)
        b = np.linalg.inv(X @ V @ X.T + I)
        c = y - (X @ mu)

        shape = N + alpha
        # 计算 sigma
        sigma = (1 / shape) * (alpha * beta ** 2 + a @ b @ c)
        scale = sigma ** 2

        # sigma 是 sigma^2 的逆 Gamma 先验的众数
        sigma = scale / (shape - 1)

        # 计算均值
        V_inv = np.linalg.inv(V)
        L = np.linalg.inv(V_inv + X.T @ X)
        R = V_inv @ mu + X.T @ y

        mu = L @ R
        cov = L * sigma

        # 计算 sigma^2 和 b 的后验分布
        self.posterior = {
            "sigma**2": stats.distributions.invgamma(a=shape, scale=scale),
            "b | sigma**2": stats.multivariate_normal(mean=mu, cov=cov),
        }
        return self
    # 预测目标与输入数据关联的最大后验概率（MAP）预测
    def predict(self, X):
        """
        返回与`X`相关联的目标的MAP预测。

        参数
        ----------
        X：形状为`(Z, M)`的 :py:class:`ndarray <numpy.ndarray>`
            由`Z`个新示例组成的数据集，每个示例的维度为`M`。

        返回
        -------
        y_pred：形状为`(Z, K)`的 :py:class:`ndarray <numpy.ndarray>`
            `X`中项目的模型预测。
        """
        # 如果拟合截距，则将X转换为设计矩阵
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        # 创建单位矩阵I，形状为X的行数
        I = np.eye(X.shape[0])  # noqa: E741
        # 计算mu，即X与后验分布的均值的乘积
        mu = X @ self.posterior["b | sigma**2"].mean
        # 计算cov，即X与后验分布的协方差矩阵的乘积再加上单位矩阵I
        cov = X @ self.posterior["b | sigma**2"].cov @ X.T + I

        # MAP估计的y对应于后验预测的均值
        self.posterior_predictive = stats.multivariate_normal(mu, cov)
        # 返回mu作为预测结果
        return mu
class BayesianLinearRegressionKnownVariance:
    def fit(self, X, y):
        """
        Compute the posterior over model parameters using the data in `X` and
        `y`.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(N, M)`
            A dataset consisting of `N` examples, each of dimension `M`.
        y : :py:class:`ndarray <numpy.ndarray>` of shape `(N, K)`
            The targets for each of the `N` examples in `X`, where each target
            has dimension `K`.
        """
        # 如果需要拟合截距，将X转换为设计矩阵
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        N, M = X.shape

        # 如果V是数字，将其转换为M维单位矩阵
        if is_number(self.V):
            self.V *= np.eye(M)

        # 如果mu是数字，将其转换为M维全1向量
        if is_number(self.mu):
            self.mu *= np.ones(M)

        V = self.V
        mu = self.mu
        sigma = self.sigma

        # 计算V的逆矩阵
        V_inv = np.linalg.inv(V)
        # 计算L矩阵
        L = np.linalg.inv(V_inv + X.T @ X)
        # 计算R矩阵
        R = V_inv @ mu + X.T @ y

        # 计算后验均值mu
        mu = L @ R
        # 计算协方差矩阵cov
        cov = L * sigma ** 2

        # 在给定sigma的条件下，计算b的后验分布
        self.posterior["b"] = stats.multivariate_normal(mu, cov)
    def predict(self, X):
        """
        Return the MAP prediction for the targets associated with `X`.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(Z, M)`
            A dataset consisting of `Z` new examples, each of dimension `M`.

        Returns
        -------
        y_pred : :py:class:`ndarray <numpy.ndarray>` of shape `(Z, K)`
            The MAP predictions for the targets associated with the items in
            `X`.
        """
        # 如果需要拟合截距，将 X 转换为设计矩阵
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        # 创建单位矩阵 I，形状与 X 的行数相同
        I = np.eye(X.shape[0])  # noqa: E741
        # 计算 mu，使用 X 与后验均值的矩阵乘法
        mu = X @ self.posterior["b"].mean
        # 计算 cov，使用 X 与后验协方差的矩阵乘法，加上单位矩阵 I
        cov = X @ self.posterior["b"].cov @ X.T + I

        # MAP 预测 y 对应于高斯后验预测分布的均值/众数
        # 将 mu 和 cov 作为参数创建多元正态分布对象
        self.posterior_predictive = stats.multivariate_normal(mu, cov)
        # 返回 mu 作为预测结果
        return mu
```