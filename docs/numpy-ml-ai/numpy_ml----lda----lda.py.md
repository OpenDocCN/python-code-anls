# `numpy-ml\numpy_ml\lda\lda.py`

```py
import numpy as np
from scipy.special import digamma, polygamma, gammaln

class LDA(object):
    def __init__(self, T=10):
        """
        Vanilla (non-smoothed) LDA model trained using variational EM.
        Generates maximum-likelihood estimates for model paramters
        `alpha` and `beta`.

        Parameters
        ----------
        T : int
            Number of topics

        Attributes
        ----------
        D : int
            Number of documents
        N : list of length `D`
            Number of words in each document
        V : int
            Number of unique word tokens across all documents
        phi : :py:class:`ndarray <numpy.ndarray>` of shape `(D, N[d], T)`
            Variational approximation to word-topic distribution
        gamma : :py:class:`ndarray <numpy.ndarray>` of shape `(D, T)`
            Variational approximation to document-topic distribution
        alpha : :py:class:`ndarray <numpy.ndarray>` of shape `(1, T)`
            Parameter for the Dirichlet prior on the document-topic distribution
        beta  : :py:class:`ndarray <numpy.ndarray>` of shape `(V, T)`
            Word-topic distribution
        """
        # 初始化LDA模型，设置主题数T
        self.T = T

    def _maximize_phi(self):
        """
        Optimize variational parameter phi
        ϕ_{t, n} ∝ β_{t, w_n}  e^( Ψ(γ_t) )
        """
        # 获取模型参数
        D = self.D
        N = self.N
        T = self.T

        # 获取模型变量
        phi = self.phi
        beta = self.beta
        gamma = self.gamma
        corpus = self.corpus

        # 遍历文档
        for d in range(D):
            # 遍历文档中的单词
            for n in range(N[d]):
                # 遍历主题
                for t in range(T):
                    # 获取单词索引
                    w_n = int(corpus[d][n])
                    # 计算phi的值
                    phi[d][n, t] = beta[w_n, t] * np.exp(digamma(gamma[d, t]))

                # 对主题进行归一化
                phi[d][n, :] = phi[d][n, :] / np.sum(phi[d][n, :])
        return phi
    # 最大化变分参数 gamma
    def _maximize_gamma(self):
        """
        Optimize variational parameter gamma
        γ_t = α_t + \sum_{n=1}^{N_d} ϕ_{t, n}
        """
        # 获取文档数量
        D = self.D
        # 获取文档-主题分布
        phi = self.phi
        # 获取先验参数 alpha
        alpha = self.alpha

        # 计算 gamma
        gamma = np.tile(alpha, (D, 1)) + np.array(
            list(map(lambda x: np.sum(x, axis=0), phi))
        )
        return gamma

    # 最大化模型参数 beta
    def _maximize_beta(self):
        """
        Optimize model parameter beta
        β_{t, n} ∝ \sum_{d=1}^D \sum_{i=1}^{N_d} ϕ_{d, t, n} [ i = n]
        """
        # 获取主题数量
        T = self.T
        # 获取词汇表大小
        V = self.V

        # 获取文档-主题分布
        phi = self.phi
        # 获取模型参数 beta
        beta = self.beta
        # 获取语料库
        corpus = self.corpus

        # 遍历词汇表中的每个词
        for n in range(V):
            # 构建与 phi 相同形状的二进制掩码 [i == n]
            mask = [np.tile((doc == n), (T, 1)).T for doc in corpus]
            # 更新 beta
            beta[n, :] = np.sum(
                np.array(list(map(lambda x: np.sum(x, axis=0), phi * mask))), axis=0
            )

        # 对每个主题进行归一化
        for t in range(T):
            beta[:, t] = beta[:, t] / np.sum(beta[:, t])

        return beta
    # 使用 Blei 的 O(n) 牛顿-拉普拉斯修改来优化 alpha，针对具有特殊结构的 Hessian 矩阵
    def _maximize_alpha(self, max_iters=1000, tol=0.1):
        # 获取文档数和主题数
        D = self.D
        T = self.T

        # 获取当前的 alpha 和 gamma
        alpha = self.alpha
        gamma = self.gamma

        # 迭代最大次数
        for _ in range(max_iters):
            # 保存旧的 alpha
            alpha_old = alpha

            # 计算梯度
            g = D * (digamma(np.sum(alpha)) - digamma(alpha)) + np.sum(
                digamma(gamma) - np.tile(digamma(np.sum(gamma, axis=1)), (T, 1)).T,
                axis=0,
            )

            # 计算 Hessian 对角线分量
            h = -D * polygamma(1, alpha)

            # 计算 Hessian 常数分量
            z = D * polygamma(1, np.sum(alpha))

            # 计算常数
            c = np.sum(g / h) / (z ** (-1.0) + np.sum(h ** (-1.0)))

            # 更新 alpha
            alpha = alpha - (g - c) / h

            # 检查收敛性
            if np.sqrt(np.mean(np.square(alpha - alpha_old))) < tol:
                break

        return alpha

    # E 步：最大化变分参数 γ 和 ϕ 以最大化 VLB
    def _E_step(self):
        # 最大化 ϕ
        self.phi = self._maximize_phi()
        # 最大化 γ
        self.gamma = self._maximize_gamma()

    # M 步：最大化模型参数 α 和 β 以最大化 VLB
    def _M_step(self):
        # 最大化 β
        self.beta = self._maximize_beta()
        # 最大化 alpha
        self.alpha = self._maximize_alpha()
    def VLB(self):
        """
        Return the variational lower bound associated with the current model
        parameters.
        """
        # 获取当前模型参数
        phi = self.phi
        alpha = self.alpha
        beta = self.beta
        gamma = self.gamma
        corpus = self.corpus

        D = self.D
        T = self.T
        N = self.N

        a, b, c, _d = 0, 0, 0, 0
        # 遍历文档
        for d in range(D):
            # 计算 a 部分
            a += (
                gammaln(np.sum(alpha))
                - np.sum(gammaln(alpha))
                + np.sum([(alpha[t] - 1) * dg(gamma, d, t) for t in range(T)])
            )

            _d += (
                gammaln(np.sum(gamma[d, :]))
                - np.sum(gammaln(gamma[d, :]))
                + np.sum([(gamma[d, t] - 1) * dg(gamma, d, t) for t in range(T)])
            )

            # 遍历文档中的词
            for n in range(N[d]):
                w_n = int(corpus[d][n])

                # 计算 b 部分
                b += np.sum([phi[d][n, t] * dg(gamma, d, t) for t in range(T)])
                # 计算 c 部分
                c += np.sum([phi[d][n, t] * np.log(beta[w_n, t]) for t in range(T)])
                # 计算 _d 部分
                _d += np.sum([phi[d][n, t] * np.log(phi[d][n, t]) for t in range(T)])

        # 返回变分下界
        return a + b + c - _d

    def initialize_parameters(self):
        """
        Provide reasonable initializations for model and variational parameters.
        """
        T = self.T
        V = self.V
        N = self.N
        D = self.D

        # 初始化模型参数
        self.alpha = 100 * np.random.dirichlet(10 * np.ones(T), 1)[0]
        self.beta = np.random.dirichlet(np.ones(V), T).T

        # 初始化变分参数
        self.phi = np.array([1 / T * np.ones([N[d], T]) for d in range(D)])
        self.gamma = np.tile(self.alpha, (D, 1)) + np.tile(N / T, (T, 1)).T
    def train(self, corpus, verbose=False, max_iter=1000, tol=5):
        """
        Train the LDA model on a corpus of documents (bags of words).

        Parameters
        ----------
        corpus : list of length `D`
            A list of lists, with each sublist containing the tokenized text of
            a single document.
        verbose : bool
            Whether to print the VLB at each training iteration. Default is
            True.
        max_iter : int
            The maximum number of training iterations to perform before
            breaking. Default is 1000.
        tol : int
            Break the training loop if the difference betwen the VLB on the
            current iteration and the previous iteration is less than `tol`.
            Default is 5.
        """
        # 设置文档数量 D 为输入语料库的长度
        self.D = len(corpus)
        # 设置词汇量 V 为语料库中所有不同词的数量
        self.V = len(set(np.concatenate(corpus)))
        # 计算每个文档的词数，存储在 N 中
        self.N = np.array([len(d) for d in corpus])
        # 存储输入的语料库
        self.corpus = corpus

        # 初始化模型参数
        self.initialize_parameters()
        # 初始化变分下界值为负无穷
        vlb = -np.inf

        # 迭代训练模型
        for i in range(max_iter):
            # 保存上一次迭代的变分下界值
            old_vlb = vlb

            # E 步：更新变分参数
            self._E_step()
            # M 步：更新模型参数
            self._M_step()

            # 计算当前迭代的变分下界值
            vlb = self.VLB()
            # 计算当前迭代变分下界值与上一次迭代的差值
            delta = vlb - old_vlb

            # 如果 verbose 为 True，则打印当前迭代的变分下界值和差值
            if verbose:
                print("Iteration {}: {:.3f} (delta: {:.2f})".format(i + 1, vlb, delta))

            # 如果变分下界值的变化小于设定的阈值 tol，则结束训练
            if delta < tol:
                break
# 定义一个函数，计算 Dirichlet 分布中随机变量 X_t 的期望对数值，其中 X_t ~ Dirichlet
def dg(gamma, d, t):
    """
    E[log X_t] where X_t ~ Dir
    """
    # 返回 digamma(gamma[d, t]) 减去 digamma(np.sum(gamma[d, :])) 的结果
    return digamma(gamma[d, t]) - digamma(np.sum(gamma[d, :]))
```