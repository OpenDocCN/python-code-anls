# `numpy-ml\numpy_ml\hmm\hmm.py`

```py
# 隐马尔可夫模型模块
import numpy as np
# 导入 logsumexp 函数
from numpy_ml.utils.misc import logsumexp

# 定义 MultinomialHMM 类
class MultinomialHMM:
    # 从 HMM 中生成一个序列
    def generate(self, n_steps, latent_state_types, obs_types):
        """
        从 HMM 中采样一个序列。

        参数
        ----------
        n_steps : int
            生成序列的长度
        latent_state_types : :py:class:`ndarray <numpy.ndarray>` of shape `(N,)`
            潜在状态的标签集合
        obs_types : :py:class:`ndarray <numpy.ndarray>` of shape `(V,)`
            观测值的标签集合

        返回
        -------
        states : :py:class:`ndarray <numpy.ndarray>` of shape `(n_steps,)`
            采样的潜在状态。
        emissions : :py:class:`ndarray <numpy.ndarray>` of shape `(n_steps,)`
            采样的观测值。
        """
        # 获取 HMM 参数
        P = self.parameters
        A, B, pi = P["A"], P["B"], P["pi"]

        # 采样初始潜在状态
        s = np.random.multinomial(1, pi).argmax()
        states = [latent_state_types[s]]

        # 生成给定潜在状态的观测值
        v = np.random.multinomial(1, B[s, :]).argmax()
        emissions = [obs_types[v]]

        # 采样潜在状态转移，重复此过程
        for i in range(n_steps - 1):
            s = np.random.multinomial(1, A[s, :]).argmax()
            states.append(latent_state_types[s])

            v = np.random.multinomial(1, B[s, :]).argmax()
            emissions.append(obs_types[v])

        return np.array(states), np.array(emissions)
    # 初始化模型参数
    def _initialize_parameters(self):
        # 获取参数字典
        P = self.parameters
        # 获取模型参数 A, B, pi
        A, B, pi = P["A"], P["B"], P["pi"]
        # 获取派生变量 N, V
        N, V = self.derived_variables["N"], self.derived_variables["V"]

        # 对隐藏状态的先验进行均匀初始化
        if pi is None:
            pi = np.ones(N)
            pi = pi / pi.sum()

        # 对转移矩阵 A 进行均匀初始化
        if A is None:
            A = np.ones((N, N))
            A = A / A.sum(axis=1)[:, None]

        # 对观测概率矩阵 B 进行随机初始化
        if B is None:
            B = np.random.rand(N, V)
            B = B / B.sum(axis=1)[:, None]

        # 更新模型参数字典中的 A, B, pi
        P["A"], P["B"], P["pi"] = A, B, pi

    # 拟合模型
    def fit(
        self,
        O,
        latent_state_types,
        observation_types,
        pi=None,
        tol=1e-5,
        verbose=False,
```