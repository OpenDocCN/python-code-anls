# `numpy-ml\numpy_ml\gmm\gmm.py`

```py
# 导入必要的库
import numpy as np

# 导入自定义的辅助函数
from numpy_ml.utils.misc import logsumexp, log_gaussian_pdf

# 定义一个高斯混合模型类
class GMM(object):
    def __init__(self, C=3, seed=None):
        """
        通过期望最大化算法训练的高斯混合模型。

        参数
        ----------
        C : int
            GMM 中的簇数/混合成分数量。默认为 3。
        seed : int
            随机数生成器的种子。默认为 None。

        属性
        ----------
        N : int
            训练数据集中的示例数量。
        d : int
            训练数据集中每个示例的维度。
        pi : :py:class:`ndarray <numpy.ndarray>` of shape `(C,)`
            簇先验。
        Q : :py:class:`ndarray <numpy.ndarray>` of shape `(N, C)`
            变分分布 `q(T)`。
        mu : :py:class:`ndarray <numpy.ndarray>` of shape `(C, d)`
            簇均值。
        sigma : :py:class:`ndarray <numpy.ndarray>` of shape `(C, d, d)`
            簇协方差矩阵。
        """
        # 初始化 ELBO 和参数字典
        self.elbo = None
        self.parameters = {}
        # 初始化超参数字典
        self.hyperparameters = {
            "C": C,
            "seed": seed,
        }

        # 是否已拟合的标志
        self.is_fit = False

        # 如果有指定种子，则设置随机数种子
        if seed:
            np.random.seed(seed)

    def _initialize_params(self, X):
        """随机初始化起始 GMM 参数。"""
        # 获取训练数据集的示例数量和维度
        N, d = X.shape
        C = self.hyperparameters["C"]

        # 随机生成 C 个随机数
        rr = np.random.rand(C)

        # 初始化参数字典
        self.parameters = {
            "pi": rr / rr.sum(),  # 簇先验
            "Q": np.zeros((N, C)),  # 变分分布 q(T)
            "mu": np.random.uniform(-5, 10, C * d).reshape(C, d),  # 簇均值
            "sigma": np.array([np.eye(d) for _ in range(C)]),  # 簇协方差矩阵
        }

        # 重置 ELBO 和拟合标志
        self.elbo = None
        self.is_fit = False
    # 计算当前 GMM 参数下的 LLB（对数似然下界）
    def likelihood_lower_bound(self, X):
        """Compute the LLB under the current GMM parameters."""
        # 获取样本数量
        N = X.shape[0]
        # 获取模型参数
        P = self.parameters
        # 获取超参数中的聚类数
        C = self.hyperparameters["C"]
        # 获取模型参数中的 pi, Q, mu, sigma
        pi, Q, mu, sigma = P["pi"], P["Q"], P["mu"], P["sigma"]

        # 设置一个很小的数，用于避免除零错误
        eps = np.finfo(float).eps
        # 初始化两个期望值
        expec1, expec2 = 0.0, 0.0
        # 遍历每个样本
        for i in range(N):
            x_i = X[i]

            # 遍历每个聚类
            for c in range(C):
                # 获取当前聚类的 pi, Q, mu, sigma
                pi_k = pi[c]
                z_nk = Q[i, c]
                mu_k = mu[c, :]
                sigma_k = sigma[c, :, :]

                # 计算当前聚类的对数 pi 和对数概率密度
                log_pi_k = np.log(pi_k + eps)
                log_p_x_i = log_gaussian_pdf(x_i, mu_k, sigma_k)
                # 计算当前样本在当前聚类下的概率
                prob = z_nk * (log_p_x_i + log_pi_k)

                # 更新期望值
                expec1 += prob
                expec2 += z_nk * np.log(z_nk + eps)

        # 计算损失值
        loss = expec1 - expec2
        # 返回损失值
        return loss
    def fit(self, X, max_iter=100, tol=1e-3, verbose=False):
        """
        Fit the parameters of the GMM on some training data.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(N, d)`
            A collection of `N` training data points, each with dimension `d`.
        max_iter : int
            The maximum number of EM updates to perform before terminating
            training. Default is 100.
        tol : float
            The convergence tolerance. Training is terminated if the difference
            in VLB between the current and previous iteration is less than
            `tol`. Default is 1e-3.
        verbose : bool
            Whether to print the VLB at each training iteration. Default is
            False.

        Returns
        -------
        success : {0, -1}
            Whether training terminated without incident (0) or one of the
            mixture components collapsed and training was halted prematurely
            (-1).
        """
        # 初始化变量 prev_vlb 为负无穷
        prev_vlb = -np.inf
        # 初始化 GMM 参数
        self._initialize_params(X)

        # 迭代进行 EM 算法更新参数
        for _iter in range(max_iter):
            try:
                # E 步：计算后验概率
                self._E_step(X)
                # M 步：更新参数
                self._M_step(X)
                # 计算变分下界
                vlb = self.likelihood_lower_bound(X)

                # 如果 verbose 为 True，则打印当前迭代的变分下界
                if verbose:
                    print(f"{_iter + 1}. Lower bound: {vlb}")

                # 判断是否收敛
                converged = _iter > 0 and np.abs(vlb - prev_vlb) <= tol
                # 如果变分下界为 NaN 或者已经收敛，则跳出循环
                if np.isnan(vlb) or converged:
                    break

                # 更新 prev_vlb
                prev_vlb = vlb

            except np.linalg.LinAlgError:
                # 捕获线性代数错误，表示某个混合成分崩溃
                print("Singular matrix: components collapsed")
                return -1

        # 将最终的变分下界赋值给 elbo，并标记已拟合完成
        self.elbo = vlb
        self.is_fit = True
        return 0
    def predict(self, X, soft_labels=True):
        """
        Return the log probability of each data point in `X` under each
        mixture components.

        Parameters
        ----------
        X : :py:class:`ndarray <numpy.ndarray>` of shape `(M, d)`
            A collection of `M` data points, each with dimension `d`.
        soft_labels : bool
            If True, return the log probabilities of the M data points in X
            under each mixture component. If False, return only the ID of the
            most probable mixture. Default is True.

        Returns
        -------
        y : :py:class:`ndarray <numpy.ndarray>` of shape `(M, C)` or `(M,)`
            If `soft_labels` is True, `y` is a 2D array where index (i,j) gives
            the log probability of the `i` th data point under the `j` th
            mixture component. If `soft_labels` is False, `y` is a 1D array
            where the `i` th index contains the ID of the most probable mixture
            component.
        """
        # 检查模型是否已经拟合
        assert self.is_fit, "Must call the `.fit` method before making predictions"

        # 获取模型参数和混合成分数量
        P = self.parameters
        C = self.hyperparameters["C"]
        mu, sigma = P["mu"], P["sigma"]

        # 初始化结果列表
        y = []
        # 遍历数据集中的每个数据点
        for x_i in X:
            # 计算每个数据点在每个混合成分下的对数概率
            cprobs = [log_gaussian_pdf(x_i, mu[c, :], sigma[c, :, :]) for c in range(C)]

            # 如果不需要软标签
            if not soft_labels:
                # 将最可能的混合成分的索引添加到结果列表中
                y.append(np.argmax(cprobs))
            else:
                # 将每个数据点在每个混合成分下的对数概率添加到结果列表中
                y.append(cprobs)

        # 将结果列表转换为 numpy 数组并返回
        return np.array(y)
    # 执行 E 步骤，更新隐藏变量 Q
    def _E_step(self, X):
        # 获取模型参数和超参数
        P = self.parameters
        C = self.hyperparameters["C"]
        pi, Q, mu, sigma = P["pi"], P["Q"], P["mu"], P["sigma"]

        # 遍历数据集 X 中的每个样本
        for i, x_i in enumerate(X):
            # 存储分母值的列表
            denom_vals = []
            # 遍历每个聚类中心
            for c in range(C):
                pi_c = pi[c]
                mu_c = mu[c, :]
                sigma_c = sigma[c, :, :]

                # 计算对数概率值
                log_pi_c = np.log(pi_c)
                log_p_x_i = log_gaussian_pdf(x_i, mu_c, sigma_c)

                # 计算分母值并添加到列表中
                denom_vals.append(log_p_x_i + log_pi_c)

            # 计算对数分母值
            log_denom = logsumexp(denom_vals)
            # 计算隐藏变量 Q
            q_i = np.exp([num - log_denom for num in denom_vals])
            # 确保 Q 的每行和为 1
            np.testing.assert_allclose(np.sum(q_i), 1, err_msg="{}".format(np.sum(q_i)))

            Q[i, :] = q_i

    # 执行 M 步骤，更新模型参数
    def _M_step(self, X):
        N, d = X.shape
        P = self.parameters
        C = self.hyperparameters["C"]
        pi, Q, mu, sigma = P["pi"], P["Q"], P["mu"], P["sigma"]

        # 计算每个聚类的总权重
        denoms = np.sum(Q, axis=0)

        # 更新聚类先验概率
        pi = denoms / N

        # 更新聚类均值
        nums_mu = [np.dot(Q[:, c], X) for c in range(C)]
        for ix, (num, den) in enumerate(zip(nums_mu, denoms)):
            mu[ix, :] = num / den if den > 0 else np.zeros_like(num)

        # 更新聚类协方差矩阵
        for c in range(C):
            mu_c = mu[c, :]
            n_c = denoms[c]

            outer = np.zeros((d, d))
            for i in range(N):
                wic = Q[i, c]
                xi = X[i, :]
                outer += wic * np.outer(xi - mu_c, xi - mu_c)

            outer = outer / n_c if n_c > 0 else outer
            sigma[c, :, :] = outer

        # 确保聚类先验概率之和为 1
        np.testing.assert_allclose(np.sum(pi), 1, err_msg="{}".format(np.sum(pi)))
```