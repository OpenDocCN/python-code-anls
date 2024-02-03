# `numpy-ml\numpy_ml\bandits\bandits.py`

```
# 导入必要的模块
from abc import ABC, abstractmethod
# 导入 numpy 库
import numpy as np
# 导入自定义的测试函数
from numpy_ml.utils.testing import random_one_hot_matrix, is_number

# 定义一个抽象类 Bandit，表示多臂赌博机环境
class Bandit(ABC):
    # 初始化函数，接受奖励和奖励概率列表以及上下文信息
    def __init__(self, rewards, reward_probs, context=None):
        # 断言奖励和奖励概率列表长度相同
        assert len(rewards) == len(reward_probs)
        # 初始化步数为 0
        self.step = 0
        # 记录赌博机臂数
        self.n_arms = len(rewards)

        super().__init__()

    # 返回对象的字符串表示形式
    def __repr__(self):
        """A string representation for the bandit"""
        # 获取超参数字典
        HP = self.hyperparameters
        # 格式化超参数字典为字符串
        params = ", ".join(["{}={}".format(k, v) for (k, v) in HP.items() if k != "id"])
        return "{}({})".format(HP["id"], params)

    # 返回超参数字典
    @property
    def hyperparameters(self):
        """A dictionary of the bandit hyperparameters"""
        return {}

    # 抽象方法，返回最优策略下的期望奖励
    @abstractmethod
    def oracle_payoff(self, context=None):
        """
        Return the expected reward for an optimal agent.

        Parameters
        ----------
        context : :py:class:`ndarray <numpy.ndarray>` of shape `(D, K)` or None
            The current context matrix for each of the bandit arms, if
            applicable. Default is None.

        Returns
        -------
        optimal_rwd : float
            The expected reward under an optimal policy.
        """
        pass
    # 从给定臂的收益分布中“拉取”（即采样）一个样本
    def pull(self, arm_id, context=None):
        """
        "Pull" (i.e., sample from) a given arm's payoff distribution.

        Parameters
        ----------
        arm_id : int
            The integer ID of the arm to sample from
        context : :py:class:`ndarray <numpy.ndarray>` of shape `(D,)` or None
            The context vector for the current timestep if this is a contextual
            bandit. Otherwise, this argument is unused and defaults to None.

        Returns
        -------
        reward : float
            The reward sampled from the given arm's payoff distribution
        """
        # 确保臂的ID小于臂的总数
        assert arm_id < self.n_arms

        # 增加步数计数器
        self.step += 1
        # 调用内部方法_pull来实际执行拉取操作
        return self._pull(arm_id, context)

    # 重置赌博机的步数和动作计数器为零
    def reset(self):
        """Reset the bandit step and action counters to zero."""
        self.step = 0

    # 抽象方法，用于实际执行拉取操作，需要在子类中实现
    @abstractmethod
    def _pull(self, arm_id):
        pass
class MultinomialBandit(Bandit):
    # 多项式赌博机类，每个臂与不同的多项式支付分布相关联
    def __init__(self, payoffs, payoff_probs):
        """
        A multi-armed bandit where each arm is associated with a different
        multinomial payoff distribution.

        Parameters
        ----------
        payoffs : ragged list of length `K`
            The payoff values for each of the `n` bandits. ``payoffs[k][i]``
            holds the `i` th payoff value for arm `k`.
        payoff_probs : ragged list of length `K`
            A list of the probabilities associated with each of the payoff
            values in ``payoffs``. ``payoff_probs[k][i]`` holds the probability
            of payoff index `i` for arm `k`.
        """
        # 调用父类的构造函数
        super().__init__(payoffs, payoff_probs)

        # 检查每个臂的支付值和概率列表长度是否相等
        for r, rp in zip(payoffs, payoff_probs):
            assert len(r) == len(rp)
            np.testing.assert_almost_equal(sum(rp), 1.0)

        # 将支付值和概率列表转换为 NumPy 数组
        payoffs = np.array([np.array(x) for x in payoffs])
        payoff_probs = np.array([np.array(x) for x in payoff_probs])

        # 初始化实例变量
        self.payoffs = payoffs
        self.payoff_probs = payoff_probs
        self.arm_evs = np.array([sum(p * v) for p, v in zip(payoff_probs, payoffs)])
        self.best_ev = np.max(self.arm_evs)
        self.best_arm = np.argmax(self.arm_evs)

    @property
    def hyperparameters(self):
        """A dictionary of the bandit hyperparameters"""
        # 返回赌博机的超参数字典
        return {
            "id": "MultinomialBandit",
            "payoffs": self.payoffs,
            "payoff_probs": self.payoff_probs,
        }
    # 定义一个方法，用于返回最优策略下的预期奖励和最优臂
    def oracle_payoff(self, context=None):
        """
        Return the expected reward for an optimal agent.
    
        Parameters
        ----------
        context : :py:class:`ndarray <numpy.ndarray>` of shape `(D, K)` or None
            Unused. Default is None.
    
        Returns
        -------
        optimal_rwd : float
            The expected reward under an optimal policy.
        optimal_arm : float
            The arm ID with the largest expected reward.
        """
        # 返回最优奖励和最优臂
        return self.best_ev, self.best_arm
    
    # 定义一个私有方法，用于选择特定臂的奖励
    def _pull(self, arm_id, context):
        # 获取特定臂的奖励
        payoffs = self.payoffs[arm_id]
        # 获取特定臂的奖励概率
        probs = self.payoff_probs[arm_id]
        # 根据奖励概率随机选择奖励
        return np.random.choice(payoffs, p=probs)
class BernoulliBandit(Bandit):
    def __init__(self, payoff_probs):
        """
        A multi-armed bandit where each arm is associated with an independent
        Bernoulli payoff distribution.

        Parameters
        ----------
        payoff_probs : list of length `K`
            A list of the payoff probability for each arm. ``payoff_probs[k]``
            holds the probability of payoff for arm `k`.
        """
        # 初始化每个臂的奖励为1
        payoffs = [1] * len(payoff_probs)
        # 调用父类的初始化方法，传入奖励和概率
        super().__init__(payoffs, payoff_probs)

        # 检查每个臂的概率是否在0到1之间
        for p in payoff_probs:
            assert p >= 0 and p <= 1

        # 将奖励和概率转换为NumPy数组
        self.payoffs = np.array(payoffs)
        self.payoff_probs = np.array(payoff_probs)

        # 计算每个臂的期望值和最佳臂的期望值
        self.arm_evs = self.payoff_probs
        self.best_ev = np.max(self.arm_evs)
        self.best_arm = np.argmax(self.arm_evs)

    @property
    def hyperparameters(self):
        """A dictionary of the bandit hyperparameters"""
        # 返回包含超参数的字典
        return {
            "id": "BernoulliBandit",
            "payoff_probs": self.payoff_probs,
        }

    def oracle_payoff(self, context=None):
        """
        Return the expected reward for an optimal agent.

        Parameters
        ----------
        context : :py:class:`ndarray <numpy.ndarray>` of shape `(D, K)` or None
            Unused. Default is None.

        Returns
        -------
        optimal_rwd : float
            The expected reward under an optimal policy.
        optimal_arm : float
            The arm ID with the largest expected reward.
        """
        # 返回最佳臂的期望值和最佳臂的ID
        return self.best_ev, self.best_arm

    def _pull(self, arm_id, context):
        # 模拟拉动臂的动作，根据概率返回奖励
        return int(np.random.rand() <= self.payoff_probs[arm_id])


class GaussianBandit(Bandit):
    def __init__(self, payoff_dists, payoff_probs):
        """
        初始化函数，创建一个类似于BernoulliBandit的多臂赌博机，但每个臂的奖励值不是固定的1，而是从独立的高斯随机变量中抽样得到。

        Parameters
        ----------
        payoff_dists : 长度为`K`的2元组列表
            每个臂的奖励值分布的参数。具体来说，``payoffs[k]``是与臂`k`关联的奖励值高斯分布的均值和方差的元组。
        payoff_probs : 长度为`n`的列表
            每个奖励值在``payoffs``中的概率列表。``payoff_probs[k]``保存了臂`k`的奖励概率。
        """
        super().__init__(payoff_dists, payoff_probs)

        for (mean, var), rp in zip(payoff_dists, payoff_probs):
            assert var > 0
            assert np.testing.assert_almost_equal(sum(rp), 1.0)

        self.payoff_dists = payoff_dists
        self.payoff_probs = payoff_probs
        self.arm_evs = np.array([mu for (mu, var) in payoff_dists])
        self.best_ev = np.max(self.arm_evs)
        self.best_arm = np.argmax(self.arm_evs)

    @property
    def hyperparameters(self):
        """返回赌博机的超参数字典"""
        return {
            "id": "GaussianBandit",
            "payoff_dists": self.payoff_dists,
            "payoff_probs": self.payoff_probs,
        }

    def _pull(self, arm_id, context):
        mean, var = self.payoff_dists[arm_id]

        reward = 0
        if np.random.rand() < self.payoff_probs[arm_id]:
            reward = np.random.normal(mean, var)

        return reward
    # 定义一个方法，用于返回最佳策略下的预期奖励
    def oracle_payoff(self, context=None):
        """
        Return the expected reward for an optimal agent.

        Parameters
        ----------
        context : :py:class:`ndarray <numpy.ndarray>` of shape `(D, K)` or None
            Unused. Default is None.

        Returns
        -------
        optimal_rwd : float
            The expected reward under an optimal policy.
        optimal_arm : float
            The arm ID with the largest expected reward.
        """
        # 返回最佳策略下的预期奖励和最佳臂的ID
        return self.best_ev, self.best_arm
class ShortestPathBandit(Bandit):
    # 最短路径赌博机类，继承自Bandit类
    def __init__(self, G, start_vertex, end_vertex):
        """
        A weighted graph shortest path problem formulated as a multi-armed
        bandit.

        Notes
        -----
        Each arm corresponds to a valid path through the graph from start to
        end vertex. The agent's goal is to find the path that minimizes the
        expected sum of the weights on the edges it traverses.

        Parameters
        ----------
        G : :class:`Graph <numpy_ml.utils.graphs.Graph>` instance
            A weighted graph object. Weights can be fixed or probabilistic.
        start_vertex : int
            The index of the path's start vertex in the graph
        end_vertex : int
            The index of the path's end vertex in the graph
        """
        # 初始化函数，接受图G、起始顶点start_vertex和结束顶点end_vertex作为参数
        self.G = G
        self.end_vertex = end_vertex
        self.adj_dict = G.to_adj_dict()
        self.start_vertex = start_vertex
        self.paths = G.all_paths(start_vertex, end_vertex)

        self.arm_evs = self._calc_arm_evs()
        self.best_ev = np.max(self.arm_evs)
        self.best_arm = np.argmax(self.arm_evs)

        placeholder = [None] * len(self.paths)
        # 调用父类的初始化函数，传入占位符列表作为参数
        super().__init__(placeholder, placeholder)

    @property
    def hyperparameters(self):
        """A dictionary of the bandit hyperparameters"""
        # 返回赌博机的超参数字典
        return {
            "id": "ShortestPathBandit",
            "G": self.G,
            "end_vertex": self.end_vertex,
            "start_vertex": self.start_vertex,
        }
    # 返回最佳策略下的预期奖励和最佳臂的 ID
    def oracle_payoff(self, context=None):
        """
        Return the expected reward for an optimal agent.

        Parameters
        ----------
        context : :py:class:`ndarray <numpy.ndarray>` of shape `(D, K)` or None
            Unused. Default is None.

        Returns
        -------
        optimal_rwd : float
            The expected reward under an optimal policy.
        optimal_arm : float
            The arm ID with the largest expected reward.
        """
        # 返回最佳策略下的预期奖励和最佳臂的 ID
        return self.best_ev, self.best_arm

    # 计算每个臂的预期奖励
    def _calc_arm_evs(self):
        # 获取顶点映射函数
        I2V = self.G.get_vertex
        # 初始化预期奖励数组
        evs = np.zeros(len(self.paths))
        # 遍历所有路径
        for p_ix, path in enumerate(self.paths):
            # 遍历路径中的每个顶点
            for ix, v_i in enumerate(path[:-1]):
                # 获取当前边的权重
                e = [e for e in self.adj_dict[v_i] if e.to == I2V(path[ix + 1])][0]
                # 累加边的权重到预期奖励中
                evs[p_ix] -= e.weight
        return evs

    # 拉动指定臂
    def _pull(self, arm_id, context):
        # 初始化奖励
        reward = 0
        # 获取顶点映射函数
        I2V = self.G.get_vertex
        # 获取指定臂对应的路径
        path = self.paths[arm_id]
        # 遍历路径中的每个顶点
        for ix, v_i in enumerate(path[:-1]):
            # 获取当前边的权重
            e = [e for e in self.adj_dict[v_i] if e.to == I2V(path[ix + 1])][0]
            # 累加边的权重到奖励中
            reward -= e.weight
        return reward
class ContextualBernoulliBandit(Bandit):
    # 定义一个上下文版本的 BernoulliBandit 类，其中每个二进制上下文特征与独立的伯努利支付分布相关联
    def __init__(self, context_probs):
        """
        A contextual version of :class:`BernoulliBandit` where each binary
        context feature is associated with an independent Bernoulli payoff
        distribution.

        Parameters
        ----------
        context_probs : :py:class:`ndarray <numpy.ndarray>` of shape `(D, K)`
            A matrix of the payoff probabilities associated with each of the
            `D` context features, for each of the `K` arms. Index `(i, j)`
            contains the probability of payoff for arm `j` under context `i`.
        """
        # 获取上下文概率矩阵的形状
        D, K = context_probs.shape

        # 使用一个虚拟占位符变量来初始化 Bandit 超类
        placeholder = [None] * K
        super().__init__(placeholder, placeholder)

        # 设置上下文概率矩阵和臂的期望值
        self.context_probs = context_probs
        self.arm_evs = self.context_probs
        self.best_evs = self.arm_evs.max(axis=1)
        self.best_arms = self.arm_evs.argmax(axis=1)

    @property
    def hyperparameters(self):
        """A dictionary of the bandit hyperparameters"""
        # 返回一个包含 bandit 超参数的字典
        return {
            "id": "ContextualBernoulliBandit",
            "context_probs": self.context_probs,
        }

    def get_context(self):
        """
        Sample a random one-hot context vector. This vector will be the same
        for all arms.

        Returns
        -------
        context : :py:class:`ndarray <numpy.ndarray>` of shape `(D, K)`
            A random `D`-dimensional one-hot context vector repeated for each
            of the `K` bandit arms.
        """
        # 获取上下文概率矩阵的形状
        D, K = self.context_probs.shape
        # 创建一个全零矩阵作为上下文向量
        context = np.zeros((D, K))
        # 在随机选择的维度上设置为 1，生成一个随机的 one-hot 上下文向量
        context[np.random.choice(D), :] = 1
        return random_one_hot_matrix(1, D).ravel()
    # 计算在给定上下文下，最优策略的预期奖励和最优臂
    def oracle_payoff(self, context):
        """
        Return the expected reward for an optimal agent.

        Parameters
        ----------
        context : :py:class:`ndarray <numpy.ndarray>` of shape `(D, K)` or None
            The current context matrix for each of the bandit arms.

        Returns
        -------
        optimal_rwd : float
            The expected reward under an optimal policy.
        optimal_arm : float
            The arm ID with the largest expected reward.
        """
        # 获取上下文矩阵中第一列的最大值所在的索引
        context_id = context[:, 0].argmax()
        # 返回最优策略下的预期奖励和最优臂
        return self.best_evs[context_id], self.best_arms[context_id]

    # 执行拉动动作，返回奖励
    def _pull(self, arm_id, context):
        # 获取上下文概率矩阵的维度
        D, K = self.context_probs.shape
        # 计算选择的臂的概率
        arm_probs = context[:, arm_id] @ self.context_probs
        # 生成服从该臂概率分布的奖励
        arm_rwds = (np.random.rand(K) <= arm_probs).astype(int)
        # 返回选择臂的奖励
        return arm_rwds[arm_id]
class ContextualLinearBandit(Bandit):
    def __init__(self, K, D, payoff_variance=1):
        r"""
        A contextual linear multi-armed bandit.

        Notes
        -----
        In a contextual linear bandit the expected payoff of an arm :math:`a
        \in \mathcal{A}` at time `t` is a linear combination of its context
        vector :math:`\mathbf{x}_{t,a}` with a coefficient vector
        :math:`\theta_a`:

        .. math::

            \mathbb{E}[r_{t, a} \mid \mathbf{x}_{t, a}] = \mathbf{x}_{t,a}^\top \theta_a

        In this implementation, the arm coefficient vectors :math:`\theta` are
        initialized independently from a uniform distribution on the interval
        [-1, 1], and the specific reward at timestep `t` is normally
        distributed:

        .. math::

            r_{t, a} \mid \mathbf{x}_{t, a} \sim
                \mathcal{N}(\mathbf{x}_{t,a}^\top \theta_a, \sigma_a^2)

        Parameters
        ----------
        K : int
            The number of bandit arms
        D : int
            The dimensionality of the context vectors
        payoff_variance : float or :py:class:`ndarray <numpy.ndarray>` of shape `(K,)`
            The variance of the random noise in the arm payoffs. If a float,
            the variance is assumed to be equal for each arm. Default is 1.
        """
        # 如果 payoff_variance 是一个数字，则将其扩展为长度为 K 的列表
        if is_number(payoff_variance):
            payoff_variance = [payoff_variance] * K

        # 确保 payoff_variance 的长度等于 K
        assert len(payoff_variance) == K
        # 确保 payoff_variance 中的值都大于 0
        assert all(v > 0 for v in payoff_variance)

        # 初始化 K、D 和 payoff_variance 属性
        self.K = K
        self.D = D
        self.payoff_variance = payoff_variance

        # 使用占位符变量初始化 Bandit 超类
        placeholder = [None] * K
        super().__init__(placeholder, placeholder)

        # 初始化 theta 矩阵
        self.thetas = np.random.uniform(-1, 1, size=(D, K))
        self.thetas /= np.linalg.norm(self.thetas, 2)

    @property
    def hyperparameters(self):
        """返回一个字典，包含赌博机的超参数"""
        return {
            "id": "ContextualLinearBandit",
            "K": self.K,
            "D": self.D,
            "payoff_variance": self.payoff_variance,
        }

    @property
    def parameters(self):
        """返回当前赌博机参数的字典"""
        return {"thetas": self.thetas}

    def get_context(self):
        """
        从多元标准正态分布中抽样每个臂的上下文向量。

        Returns
        -------
        context : :py:class:`ndarray <numpy.ndarray>` of shape `(D, K)`
            对于每个 `K` 赌博机臂，从标准正态分布中抽样的 `D` 维上下文向量。
        """
        return np.random.normal(size=(self.D, self.K))

    def oracle_payoff(self, context):
        """
        返回最佳策略下的预期奖励。

        Parameters
        ----------
        context : :py:class:`ndarray <numpy.ndarray>` of shape `(D, K)` or None
            每个赌博机臂的当前上下文矩阵，如果适用。默认为 None。

        Returns
        -------
        optimal_rwd : float
            最佳策略下的预期奖励。
        optimal_arm : float
            具有最大预期奖励的臂的 ID。
        """
        best_arm = np.argmax(self.arm_evs)
        return self.arm_evs[best_arm], best_arm

    def _pull(self, arm_id, context):
        K, thetas = self.K, self.thetas
        self._noise = np.random.normal(scale=self.payoff_variance, size=self.K)
        self.arm_evs = np.array([context[:, k] @ thetas[:, k] for k in range(K)])
        return (self.arm_evs + self._noise)[arm_id]
```