# `numpy-ml\numpy_ml\linear_models\logistic.py`

```
"""Logistic regression module"""
import numpy as np


class LogisticRegression:
    def fit(self, X, y, lr=0.01, tol=1e-7, max_iter=1e7):
        """
        Fit the regression coefficients via gradient descent on the negative
        log likelihood.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(N, M)`
            A dataset consisting of `N` examples, each of dimension `M`.
        y : :py:class:`ndarray <numpy.ndarray>` of shape `(N,)`
            The binary targets for each of the `N` examples in `X`.
        lr : float
            The gradient descent learning rate. Default is 1e-7.
        max_iter : float
            The maximum number of iterations to run the gradient descent
            solver. Default is 1e7.
        """
        # convert X to a design matrix if we're fitting an intercept
        # 如果需要拟合截距，则将 X 转换为设计矩阵
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]

        # initialize the previous loss to infinity
        l_prev = np.inf
        # initialize beta with random values
        self.beta = np.random.rand(X.shape[1])
        # iterate for a maximum number of iterations
        for _ in range(int(max_iter)):
            # calculate the predicted values using the sigmoid function
            y_pred = _sigmoid(X @ self.beta)
            # calculate the negative log likelihood loss
            loss = self._NLL(X, y, y_pred)
            # check if the improvement in loss is less than the tolerance
            if l_prev - loss < tol:
                return
            # update the previous loss
            l_prev = loss
            # update the beta coefficients using the gradient descent
            self.beta -= lr * self._NLL_grad(X, y, y_pred)
    # 计算当前模型下目标的惩罚负对数似然
    def _NLL(self, X, y, y_pred):
        # 获取样本数量和特征数量
        N, M = X.shape
        # 获取当前模型的 beta 和 gamma 参数
        beta, gamma = self.beta, self.gamma
        # 根据惩罚类型确定阶数
        order = 2 if self.penalty == "l2" else 1
        # 计算 beta 的 L2 或 L1 范数
        norm_beta = np.linalg.norm(beta, ord=order)

        # 计算负对数似然
        nll = -np.log(y_pred[y == 1]).sum() - np.log(1 - y_pred[y == 0]).sum()
        # 计算惩罚项
        penalty = (gamma / 2) * norm_beta ** 2 if order == 2 else gamma * norm_beta
        # 返回标准化后的惩罚负对数似然
        return (penalty + nll) / N

    # 计算惩罚负对数似然对 beta 的梯度
    def _NLL_grad(self, X, y, y_pred):
        """Gradient of the penalized negative log likelihood wrt beta"""
        # 获取样本数量和特征数量
        N, M = X.shape
        # 获取惩罚类型、beta 和 gamma 参数
        p, beta, gamma = self.penalty, self.beta, self.gamma
        # 计算惩罚项的梯度
        d_penalty = gamma * beta if p == "l2" else gamma * np.sign(beta)
        # 返回梯度
        return -((y - y_pred) @ X + d_penalty) / N

    # 使用训练好的模型在新数据集上生成预测概率
    def predict(self, X):
        """
        Use the trained model to generate prediction probabilities on a new
        collection of data points.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(Z, M)`
            A dataset consisting of `Z` new examples, each of dimension `M`.

        Returns
        -------
        y_pred : :py:class:`ndarray <numpy.ndarray>` of shape `(Z,)`
            The model prediction probabilities for the items in `X`.
        """
        # 如果拟合截距，则将 X 转换为设计矩阵
        if self.fit_intercept:
            X = np.c_[np.ones(X.shape[0]), X]
        # 返回预测概率
        return _sigmoid(X @ self.beta)
# 定义一个函数，计算 logistic sigmoid 函数的值
def _sigmoid(x):
    # logistic sigmoid 函数的定义：1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))
```