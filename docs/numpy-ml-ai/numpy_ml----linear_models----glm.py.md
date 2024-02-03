# `numpy-ml\numpy_ml\linear_models\glm.py`

```
# 导入 numpy 库
import numpy as np

# 导入线性回归模型
from numpy_ml.linear_models.linear_regression import LinearRegression

# 定义一个极小值
eps = np.finfo(float).eps

# 不同链接函数的字典
_GLM_LINKS = {
    "logit": {
        # 对数几率链接函数
        "link": lambda mu: np.log((mu + eps) / (1 - mu + eps)),
        "inv_link": lambda eta: 1.0 / (1.0 + np.exp(-eta)),
        "link_prime": lambda x: (1 / (x + eps)) + (1 / (1 - x + eps)),
        "theta": lambda mu: np.log((mu + eps) / (1 - mu + eps)),
        "phi": lambda x: np.ones(x.shape[0]),
        "a": lambda phi: phi,
        "b": lambda theta: np.log(1 + np.exp(theta)),
        "p": 1,
        "b_prime": lambda theta: np.exp(theta) / (1 + np.exp(theta)),
        "b_prime2": lambda theta: np.exp(theta) / ((1 + np.exp(theta)) ** 2),
    },
    "identity": {
        # 恒等链接函数
        "link": lambda mu: mu,
        "inv_link": lambda eta: eta,
        "link_prime": lambda x: np.ones_like(x),
        "theta": lambda mu: mu,
        "phi": lambda x: np.var(x, axis=0),
        "a": lambda phi: phi,
        "b": lambda theta: 0.5 * theta ** 2,
        "p": 1,
        "b_prime": lambda theta: theta,
        "b_prime2": lambda theta: np.ones_like(theta),
    },
    "log": {
        # 对数链接函数
        "link": lambda mu: np.log(mu + eps),
        "inv_link": lambda eta: np.exp(eta),
        "link_prime": lambda x: 1 / (x + eps),
        "theta": lambda mu: np.log(mu + eps),
        "phi": lambda x: np.ones(x.shape[0]),
        "a": lambda phi: phi,
        "p": 1,
        "b": lambda theta: np.exp(theta),
        "b_prime": lambda theta: np.exp(theta),
        "b_prime2": lambda theta: np.exp(theta),
    },
}

# 广义线性模型类
class GeneralizedLinearModel:
    def fit(self, X, y):
        """
        Find the maximum likelihood GLM coefficients via IRLS.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(N, M)`
            A dataset consisting of `N` examples, each of dimension `M`.
        y : :py:class:`ndarray <numpy.ndarray>` of shape `(N,)`
            The targets for each of the `N` examples in `X`.

        Returns
        -------
        self : :class:`GeneralizedLinearModel <numpy_ml.linear_models.GeneralizedLinearModel>` instance
        """  # noqa: E501
        # 将 y 压缩为一维数组
        y = np.squeeze(y)
        # 断言 y 的维度为 1
        assert y.ndim == 1

        # 获取 X 的形状
        N, M = X.shape
        # 获取 GLM 链接函数
        L = _GLM_LINKS[self.link]

        # 初始化参数的起始值
        mu = np.ones_like(y) * np.mean(y)
        eta = L["link"](mu)
        theta = L["theta"](mu)

        # 如果拟合截距，将 X 转换为设计矩阵
        if self.fit_intercept:
            X = np.c_[np.ones(N), X]

        # 通过 IRLS 进行 GLM 拟合
        i = 0
        diff, beta = np.inf, np.inf
        while diff > (self.tol * M):
            if i > self.max_iter:
                print("Warning: Model did not converge")
                break

            # 计算一阶泰勒近似
            z = eta + (y - mu) * L["link_prime"](mu)
            w = L["p"] / (L["b_prime2"](theta) * L["link_prime"](mu) ** 2)

            # 对 z 执行加权最小二乘
            wlr = LinearRegression(fit_intercept=False)
            beta_new = wlr.fit(X, z, weights=w).beta.ravel()

            eta = X @ beta_new
            mu = L["inv_link"](eta)
            theta = L["theta"](mu)

            diff = np.linalg.norm(beta - beta_new, ord=1)
            beta = beta_new
            i += 1

        self.beta = beta
        self._is_fit = True
        return self
    # 使用训练好的模型为数据集 X 中的数据点生成分布均值预测
    def predict(self, X):
        r"""
        Use the trained model to generate predictions for the distribution
        means, :math:`\mu`, associated with the collection of data points in
        **X**.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(Z, M)`
            A dataset consisting of `Z` new examples, each of dimension `M`.

        Returns
        -------
        mu_pred : :py:class:`ndarray <numpy.ndarray>` of shape `(Z,)`
            The model predictions for the expected value of the target
            associated with each item in `X`.
        """
        # 检查模型是否已经拟合
        assert self._is_fit, "Must call `fit` before generating predictions"
        # 从链接函数字典中获取链接函数
        L = _GLM_LINKS[self.link]

        # 如果使用截距，将 X 转换为设计矩阵
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        # 计算模型预测的均值
        mu_pred = L["inv_link"](X @ self.beta)
        # 将结果展平为一维数组并返回
        return mu_pred.ravel()
```