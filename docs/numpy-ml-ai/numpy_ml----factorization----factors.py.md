# `numpy-ml\numpy_ml\factorization\factors.py`

```py
# 包含了用于近似矩阵分解的算法
from copy import deepcopy
# 导入 numpy 库并重命名为 np
import numpy as np

# 定义 VanillaALS 类
class VanillaALS:
    # 返回当前模型参数的字典
    @property
    def parameters(self):
        return {"W": self.W, "H": self.H}

    # 返回模型超参数的字典
    @property
    def hyperparameters(self):
        return {
            "id": "ALSFactor",
            "K": self.K,
            "tol": self.tol,
            "alpha": self.alpha,
            "max_iter": self.max_iter,
        }

    # 初始化因子矩阵，可以选择随机初始化或者传入已有的矩阵
    def _init_factor_matrices(self, X, W=None, H=None):
        N, M = X.shape
        scale = np.sqrt(X.mean() / self.K)
        # 如果未传入 W，则随机初始化
        self.W = np.random.rand(N, self.K) * scale if W is None else W
        # 如果未传入 H，则随机初始化
        self.H = np.random.rand(self.K, M) * scale if H is None else H

        assert self.W.shape == (N, self.K)
        assert self.H.shape == (self.K, M)

    # 计算正则化的 Frobenius 损失
    def _loss(self, X, Xhat):
        alpha, W, H = self.alpha, self.W, self.H
        # 定义计算平方 Frobenius 范数的函数
        sq_fnorm = lambda x: np.sum(x ** 2)  # noqa: E731
        return sq_fnorm(X - Xhat) + alpha * (sq_fnorm(W) + sq_fnorm(H))

    # 执行 ALS 更新
    def _update_factor(self, X, A):
        T1 = np.linalg.inv(A.T @ A + self.alpha * np.eye(self.K))
        return X @ A @ T1
    # 将数据矩阵分解为两个低秩因子，使用ALS算法
    def fit(self, X, W=None, H=None, n_initializations=10, verbose=False):
        """
        Factor a data matrix into two low rank factors via ALS.

        Parameters
        ----------
        X : numpy array of shape `(N, M)`
            The data matrix to factor.
        W : numpy array of shape `(N, K)` or None
            An initial value for the `W` factor matrix. If None, initialize `W`
            randomly. Default is None.
        H : numpy array of shape `(K, M)` or None
            An initial value for the `H` factor matrix. If None, initialize `H`
            randomly. Default is None.
        n_initializations : int
            Number of re-initializations of the algorithm to perform before
            taking the answer with the lowest reconstruction error. This value
            is ignored and set to 1 if both `W` and `H` are not None. Default
            is 10.
        verbose : bool
            Whether to print the loss at each iteration. Default is False.
        """
        # 如果W和H都不为None，则将n_initializations设置为1
        if W is not None and H is not None:
            n_initializations = 1

        # 初始化最佳损失值为正无穷
        best_loss = np.inf
        # 进行n_initializations次初始化
        for f in range(n_initializations):
            # 如果verbose为True，则打印初始化信息
            if verbose:
                print("\nINITIALIZATION {}".format(f + 1))

            # 调用_fit方法进行拟合，得到新的W、H和损失值
            new_W, new_H, loss = self._fit(X, W, H, verbose)

            # 如果当前损失值小于等于最佳损失值，则更新最佳损失值和最佳的W、H
            if loss <= best_loss:
                best_loss = loss
                best_W, best_H = deepcopy(new_W), deepcopy(new_H)

        # 将最佳的W、H赋值给模型的W、H
        self.W, self.H = best_W, best_H

        # 如果verbose为True，则打印最终损失值
        if verbose:
            print("\nFINAL LOSS: {}".format(best_loss))
    # 在模型拟合过程中，更新因子矩阵 W 和 H
    def _fit(self, X, W, H, verbose):
        # 初始化因子矩阵 W 和 H
        self._init_factor_matrices(X, W, H)
        # 获取初始化后的因子矩阵 W 和 H
        W, H = self.W, self.H

        # 迭代更新因子矩阵，直到达到最大迭代次数
        for i in range(self.max_iter):
            # 更新因子矩阵 W
            W = self._update_factor(X, H.T)
            # 更新因子矩阵 H
            H = self._update_factor(X.T, W).T

            # 计算损失函数值
            loss = self._loss(X, W @ H)

            # 如果 verbose 为 True，则打印当前迭代的损失值
            if verbose:
                print("[Iter {}] Loss: {:.8f}".format(i + 1, loss))

            # 如果损失值小于等于设定的阈值 tol，则提前结束迭代
            if loss <= self.tol:
                break

        # 返回更新后的因子矩阵 W 和 H，以及最终的损失值
        return W, H, loss
# 定义一个 NMF 类，用于执行非负矩阵分解
class NMF:
    # 返回当前模型参数的字典
    @property
    def parameters(self):
        """Return a dictionary of the current model parameters"""
        return {"W": self.W, "H": self.H}

    # 返回模型超参数的字典
    @property
    def hyperparameters(self):
        """Return a dictionary of the model hyperparameters"""
        return {
            "id": "NMF",
            "K": self.K,
            "tol": self.tol,
            "max_iter": self.max_iter,
        }

    # 初始化因子矩阵，使用普通 ALS 算法
    def _init_factor_matrices(self, X, W, H):
        """Initialize the factor matrices using vanilla ALS"""
        ALS = None
        N, M = X.shape

        # 如果 W 未定义，则使用 ALS 初始化
        if W is None:
            ALS = VanillaALS(self.K, alpha=0, max_iter=200)
            ALS.fit(X, verbose=False)
            W = ALS.W / np.linalg.norm(ALS.W, axis=0)

        # 如果 H 未定义，则随机初始化，或者使用 ALS 初始化
        if H is None:
            H = np.abs(np.random.rand(self.K, M)) if ALS is None else ALS.H

        # 断言确保矩阵形状正确
        assert W.shape == (N, self.K)
        assert H.shape == (self.K, M)

        # 将初始化后的 W 和 H 赋值给类属性
        self.H = H
        self.W = W

    # 计算重构损失函数，即 X 和 Xhat 之间的最小二乘损失
    def _loss(self, X, Xhat):
        """Return the least-squares reconstruction loss between X and Xhat"""
        return np.sum((X - Xhat) ** 2)

    # 更新 H 矩阵，使用快速 HALS 算法
    def _update_H(self, X, W, H):
        """Perform the fast HALS update for H"""
        eps = np.finfo(float).eps
        XtW = X.T @ W  # dim: (M, K)
        WtW = W.T @ W  # dim: (K, K)

        # 针对每个 K 值进行更新
        for k in range(self.K):
            H[k, :] += XtW[:, k] - H.T @ WtW[:, k]
            H[k, :] = np.clip(H[k, :], eps, np.inf)  # 强制非负性
        return H
    def _update_W(self, X, W, H):
        """Perform the fast HALS update for W"""
        eps = np.finfo(float).eps
        XHt = X @ H.T  # 计算 X 与 H 转置的乘积，维度为 (N, K)
        HHt = H @ H.T  # 计算 H 与 H 转置的乘积，维度为 (K, K)

        for k in range(self.K):
            # 更新 W 矩阵的第 k 列
            W[:, k] = W[:, k] * HHt[k, k] + XHt[:, k] - W @ HHt[:, k]
            # 强制非负性，将小于 eps 的值设为 eps，大于 np.inf 的值设为 np.inf
            W[:, k] = np.clip(W[:, k], eps, np.inf)

            # 重新归一化新的列
            n = np.linalg.norm(W[:, k])
            W[:, k] /= n if n > 0 else 1.0
        return W

    def _fit(self, X, W, H, verbose):
        self._init_factor_matrices(X, W, H)

        W, H = self.W, self.H
        for i in range(self.max_iter):
            H = self._update_H(X, W, H)
            W = self._update_W(X, W, H)
            loss = self._loss(X, W @ H)

            if verbose:
                print("[Iter {}] Loss: {:.8f}".format(i + 1, loss))

            if loss <= self.tol:
                break
        return W, H, loss
```