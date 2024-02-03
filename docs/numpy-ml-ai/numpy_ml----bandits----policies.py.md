# `numpy-ml\numpy_ml\bandits\policies.py`

```
# 一个包含各种多臂赌博问题探索策略的模块
from abc import ABC, abstractmethod
from collections import defaultdict
import numpy as np
from ..utils.testing import is_number

# 定义一个多臂赌博策略的基类
class BanditPolicyBase(ABC):
    def __init__(self):
        """一个简单的多臂赌博策略基类"""
        # 初始化步数为0，估计值为空字典，未初始化标志为False
        self.step = 0
        self.ev_estimates = {}
        self.is_initialized = False
        super().__init__()

    def __repr__(self):
        """返回策略的字符串表示"""
        HP = self.hyperparameters
        params = ", ".join(["{}={}".format(k, v) for (k, v) in HP.items() if k != "id"])
        return "{}({})".format(HP["id"], params)

    @property
    def hyperparameters(self):
        """包含策略超参数的字典"""
        pass

    @property
    def parameters(self):
        """包含当前策略参数的字典"""
        pass

    def act(self, bandit, context=None):
        """
        选择一个臂并从其收益分布中采样。

        Parameters
        ----------
        bandit : :class:`Bandit <numpy_ml.bandits.bandits.Bandit>` 实例
            要操作的多臂赌博机
        context : :py:class:`ndarray <numpy.ndarray>` 形状为 `(D,)` 或 None
            如果与上下文多臂赌博机交互，则为当前时间步的上下文向量。否则，此参数未使用。默认为None。

        Returns
        -------
        rwd : float
            拉动 ``arm_id`` 后收到的奖励。
        arm_id : int
            生成 ``rwd`` 的被拉动的臂。
        """
        if not self.is_initialized:
            self._initialize_params(bandit)

        arm_id = self._select_arm(bandit, context)
        rwd = self._pull_arm(bandit, arm_id, context)
        self._update_params(arm_id, rwd, context)
        return rwd, arm_id
    # 重置策略参数和计数器到初始状态
    def reset(self):
        """Reset the policy parameters and counters to their initial states."""
        self.step = 0
        self._reset_params()
        self.is_initialized = False

    # 执行一个摇臂动作并返回收到的奖励
    def _pull_arm(self, bandit, arm_id, context):
        """Execute a bandit action and return the received reward."""
        self.step += 1
        return bandit.pull(arm_id, context)

    # 根据当前上下文选择一个摇臂
    @abstractmethod
    def _select_arm(self, bandit, context):
        """Select an arm based on the current context"""
        pass

    # 在交互后更新策略参数
    @abstractmethod
    def _update_params(self, bandit, context):
        """Update the policy parameters after an interaction"""
        pass

    # 初始化依赖于摇臂环境信息的任何策略特定参数
    @abstractmethod
    def _initialize_params(self, bandit):
        """
        Initialize any policy-specific parameters that depend on information
        from the bandit environment.
        """
        pass

    # 重置任何模型特定参数，在公共的 `self.reset()` 方法中调用
    @abstractmethod
    def _reset_params(self):
        """
        Reset any model-specific parameters. This gets called within the
        public `self.reset()` method.
        """
        pass
class EpsilonGreedy(BanditPolicyBase):
    # 初始化函数，创建一个 epsilon-greedy 策略对象
    def __init__(self, epsilon=0.05, ev_prior=0.5):
        r"""
        An epsilon-greedy policy for multi-armed bandit problems.

        Notes
        -----
        Epsilon-greedy policies greedily select the arm with the highest
        expected payoff with probability :math:`1-\epsilon`, and selects an arm
        uniformly at random with probability :math:`\epsilon`:

        .. math::

            P(a) = \left\{
                 \begin{array}{lr}
                   \epsilon / N + (1 - \epsilon) &\text{if }
                        a = \arg \max_{a' \in \mathcal{A}}
                            \mathbb{E}_{q_{\hat{\theta}}}[r \mid a']\\
                   \epsilon / N &\text{otherwise}
                 \end{array}
               \right.

        where :math:`N = |\mathcal{A}|` is the number of arms,
        :math:`q_{\hat{\theta}}` is the estimate of the arm payoff
        distribution under current model parameters :math:`\hat{\theta}`, and
        :math:`\mathbb{E}_{q_{\hat{\theta}}}[r \mid a']` is the expected
        reward under :math:`q_{\hat{\theta}}` of receiving reward `r` after
        taking action :math:`a'`.

        Parameters
        ----------
        epsilon : float in [0, 1]
            The probability of taking a random action. Default is 0.05.
        ev_prior : float
            The starting expected payoff for each arm before any data has been
            observed. Default is 0.5.
        """
        # 调用父类的初始化函数
        super().__init__()
        # 设置 epsilon 参数
        self.epsilon = epsilon
        # 设置 ev_prior 参数
        self.ev_prior = ev_prior
        # 创建一个默认值为 0 的字典，用于记录每个臂被拉动的次数
        self.pull_counts = defaultdict(lambda: 0)

    @property
    # 返回当前策略的参数字典
    def parameters(self):
        """A dictionary containing the current policy parameters"""
        return {"ev_estimates": self.ev_estimates}

    @property
    # 返回包含策略超参数的字典
    def hyperparameters(self):
        """A dictionary containing the policy hyperparameters"""
        return {
            "id": "EpsilonGreedy",
            "epsilon": self.epsilon,
            "ev_prior": self.ev_prior,
        }

    # 初始化依赖于bandit环境信息的策略特定参数
    def _initialize_params(self, bandit):
        """
        Initialize any policy-specific parameters that depend on information
        from the bandit environment.
        """
        # 初始化每个臂的估计值为先验值
        self.ev_estimates = {i: self.ev_prior for i in range(bandit.n_arms)}
        # 标记已初始化
        self.is_initialized = True

    # 选择臂的方法
    def _select_arm(self, bandit, context=None):
        # 根据epsilon贪心策略选择臂
        if np.random.rand() < self.epsilon:
            arm_id = np.random.choice(bandit.n_arms)
        else:
            # 选择估计值最大的臂
            ests = self.ev_estimates
            (arm_id, _) = max(ests.items(), key=lambda x: x[1])
        return arm_id

    # 更新参数的方法
    def _update_params(self, arm_id, reward, context=None):
        # 获取估计值和拉动次数
        E, C = self.ev_estimates, self.pull_counts
        # 更新拉动次数
        C[arm_id] += 1
        # 更新估计值
        E[arm_id] += (reward - E[arm_id]) / (C[arm_id])

    # 重置参数的方法
    def _reset_params(self):
        """
        Reset any model-specific parameters. This gets called within the
        public `self.reset()` method.
        """
        # 重置估计值和拉动次数
        self.ev_estimates = {}
        self.pull_counts = defaultdict(lambda: 0)
class UCB1(BanditPolicyBase):
    # 初始化UCB1算法的参数，包括置信度参数C和初始期望值ev_prior
    def __init__(self, C=1, ev_prior=0.5):
        r"""
        A UCB1 policy for multi-armed bandit problems.

        Notes
        -----
        The UCB1 algorithm [*]_ guarantees the cumulative regret is bounded by log
        `t`, where `t` is the current timestep. To make this guarantee UCB1
        assumes all arm payoffs are between 0 and 1.

        Under UCB1, the upper confidence bound on the expected value for
        pulling arm `a` at timestep `t` is:

        .. math::

            \text{UCB}(a, t) = \text{EV}_t(a) + C \sqrt{\frac{2 \log t}{N_t(a)}}

        where :math:`\text{EV}_t(a)` is the average of the rewards recieved so
        far from pulling arm `a`, `C` is a free parameter controlling the
        "optimism" of the confidence upper bound for :math:`\text{UCB}(a, t)`
        (for logarithmic regret bounds, `C` must equal 1), and :math:`N_t(a)`
        is the number of times arm `a` has been pulled during the previous `t -
        1` timesteps.

        References
        ----------
        .. [*] Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). Finite-time
           analysis of the multiarmed bandit problem. *Machine Learning,
           47(2)*.

        Parameters
        ----------
        C : float in (0, +infinity)
            A confidence/optimisim parameter affecting the degree of
            exploration, where larger values encourage greater exploration. The
            UCB1 algorithm assumes `C=1`. Default is 1.
        ev_prior : float
            The starting expected value for each arm before any data has been
            observed. Default is 0.5.
        """
        # 设置参数C和ev_prior
        self.C = C
        self.ev_prior = ev_prior
        # 调用父类的初始化方法
        super().__init__()

    @property
    def parameters(self):
        """A dictionary containing the current policy parameters"""
        # 返回包含当前策略参数的字典
        return {"ev_estimates": self.ev_estimates}

    @property
    # 返回包含策略超参数的字典
    def hyperparameters(self):
        """A dictionary containing the policy hyperparameters"""
        return {
            "C": self.C,
            "id": "UCB1",
            "ev_prior": self.ev_prior,
        }

    # 初始化依赖于赌博环境信息的任何特定于策略的参数
    def _initialize_params(self, bandit):
        """
        Initialize any policy-specific parameters that depend on information
        from the bandit environment.
        """
        self.ev_estimates = {i: self.ev_prior for i in range(bandit.n_arms)}
        self.is_initialized = True

    # 选择臂，计算每个臂的得分并返回得分最高的臂
    def _select_arm(self, bandit, context=None):
        # 添加 eps 以避免在每个臂的第一次拉动时出现除零错误
        eps = np.finfo(float).eps
        N, T = bandit.n_arms, self.step + 1
        E, C = self.ev_estimates, self.pull_counts
        scores = [E[a] + self.C * np.sqrt(np.log(T) / (C[a] + eps)) for a in range(N)]
        return np.argmax(scores)

    # 更新参数，根据臂的奖励更新估计值和计数
    def _update_params(self, arm_id, reward, context=None):
        E, C = self.ev_estimates, self.pull_counts
        C[arm_id] += 1
        E[arm_id] += (reward - E[arm_id]) / (C[arm_id])

    # 重置模型特定的参数，在公共的 reset 方法中调用
    def _reset_params(self):
        """
        Reset any model-specific parameters. This gets called within the
        public :method:`reset` method.
        """
        self.ev_estimates = {}
        self.pull_counts = defaultdict(lambda: 0)
class ThompsonSamplingBetaBinomial(BanditPolicyBase):
    # 返回当前策略参数的字典
    @property
    def parameters(self):
        """A dictionary containing the current policy parameters"""
        return {
            "ev_estimates": self.ev_estimates,
            "alphas": self.alphas,
            "betas": self.betas,
        }

    # 返回策略超参数的字典
    @property
    def hyperparameters(self):
        """A dictionary containing the policy hyperparameters"""
        return {
            "id": "ThompsonSamplingBetaBinomial",
            "alpha": self.alpha,
            "beta": self.beta,
        }

    # 初始化参数
    def _initialize_params(self, bandit):
        bhp = bandit.hyperparameters
        fstr = "ThompsonSamplingBetaBinomial only defined for BernoulliBandit, got: {}"
        assert bhp["id"] == "BernoulliBandit", fstr.format(bhp["id"])

        # 初始化模型先验
        if is_number(self.alpha):
            self.alphas = [self.alpha] * bandit.n_arms
        if is_number(self.beta):
            self.betas = [self.beta] * bandit.n_arms
        assert len(self.alphas) == len(self.betas) == bandit.n_arms

        # 为每个手臂计算期望值估计
        self.ev_estimates = {i: self._map_estimate(i, 1) for i in range(bandit.n_arms)}
        self.is_initialized = True

    # 选择手臂
    def _select_arm(self, bandit, context):
        if not self.is_initialized:
            self._initialize_prior(bandit)

        # 从当前模型后验中抽取一个样本
        posterior_sample = np.random.beta(self.alphas, self.betas)

        # 基于这个样本贪婪地选择一个动作
        return np.argmax(posterior_sample)

    # 更新参数
    def _update_params(self, arm_id, rwd, context):
        """
        Compute the parameters of the Beta posterior, P(payoff prob | rwd),
        for arm `arm_id`.
        """
        # 更新 Beta 后验的参数，P(获胜概率 | 收益)
        self.alphas[arm_id] += rwd
        self.betas[arm_id] += 1 - rwd
        self.ev_estimates[arm_id] = self._map_estimate(arm_id, rwd)
    # 计算当前臂的概率支付概率的 MAP 估计值
    def _map_estimate(self, arm_id, rwd):
        """Compute the current MAP estimate for an arm's payoff probability"""
        # 获取当前臂的 alpha 和 beta 参数
        A, B = self.alphas, self.betas
        # 根据 alpha 和 beta 参数的取值情况计算 MAP 估计值
        if A[arm_id] > 1 and B[arm_id] > 1:
            map_payoff_prob = (A[arm_id] - 1) / (A[arm_id] + B[arm_id] - 2)
        elif A[arm_id] < 1 and B[arm_id] < 1:
            map_payoff_prob = rwd  # 0 or 1 equally likely, make a guess
        elif A[arm_id] <= 1 and B[arm_id] > 1:
            map_payoff_prob = 0
        elif A[arm_id] > 1 and B[arm_id] <= 1:
            map_payoff_prob = 1
        else:
            map_payoff_prob = 0.5
        return map_payoff_prob

    # 重置模型特定的参数，该方法在公共的 `self.reset()` 方法中被调用
    def _reset_params(self):
        """
        Reset any model-specific parameters. This gets called within the
        public `self.reset()` method.
        """
        # 重置 alpha 和 beta 参数列表
        self.alphas, self.betas = [], []
        # 重置期望值估计字典
        self.ev_estimates = {}
class LinUCB(BanditPolicyBase):
    def __init__(self, alpha=1):
        """
        A disjoint linear UCB policy [*]_ for contextual linear bandits.

        Notes
        -----
        LinUCB is only defined for :class:`ContextualLinearBandit <numpy_ml.bandits.ContextualLinearBandit>` environments.

        References
        ----------
        .. [*] Li, L., Chu, W., Langford, J., & Schapire, R. (2010). A
           contextual-bandit approach to personalized news article
           recommendation. In *Proceedings of the 19th International Conference
           on World Wide Web*, 661-670.

        Parameters
        ----------
        alpha : float
            A confidence/optimisim parameter affecting the amount of
            exploration. Default is 1.
        """  # noqa
        # 调用父类的构造函数
        super().__init__()

        # 初始化参数 alpha
        self.alpha = alpha
        # 初始化参数 A 和 b 为空列表
        self.A, self.b = [], []
        # 初始化标志位为 False
        self.is_initialized = False

    @property
    def parameters(self):
        """A dictionary containing the current policy parameters"""
        # 返回当前策略参数的字典
        return {"ev_estimates": self.ev_estimates, "A": self.A, "b": self.b}

    @property
    def hyperparameters(self):
        """A dictionary containing the policy hyperparameters"""
        # 返回策略的超参数字典
        return {
            "id": "LinUCB",
            "alpha": self.alpha,
        }

    def _initialize_params(self, bandit):
        """
        Initialize any policy-specific parameters that depend on information
        from the bandit environment.
        """
        # 获取环境的超参数
        bhp = bandit.hyperparameters
        # 检查环境是否为 ContextualLinearBandit 类型
        fstr = "LinUCB only defined for contextual linear bandits, got: {}"
        assert bhp["id"] == "ContextualLinearBandit", fstr.format(bhp["id"])

        # 初始化参数 A 和 b
        self.A, self.b = [], []
        # 根据环境的臂数初始化 A 和 b
        for _ in range(bandit.n_arms):
            self.A.append(np.eye(bandit.D))
            self.b.append(np.zeros(bandit.D))

        # 设置初始化标志位为 True
        self.is_initialized = True
    # 选择最优的臂，根据每个臂的概率计算
    def _select_arm(self, bandit, context):
        # 初始化概率列表
        probs = []
        # 遍历每个臂
        for a in range(bandit.n_arms):
            # 获取当前臂的上下文信息、参数 A 和 b
            C, A, b = context[:, a], self.A[a], self.b[a]
            # 计算参数 A 的逆矩阵
            A_inv = np.linalg.inv(A)
            # 计算参数 theta_hat
            theta_hat = A_inv @ b
            # 计算当前臂的概率 p
            p = theta_hat @ C + self.alpha * np.sqrt(C.T @ A_inv @ C)

            # 将当前臂的概率添加到概率列表中
            probs.append(p)
        # 返回概率最大的臂的索引
        return np.argmax(probs)

    # 更新参数 A 和 b
    def _update_params(self, arm_id, rwd, context):
        """Compute the parameters for A and b."""
        # 更新参数 A
        self.A[arm_id] += context[:, arm_id] @ context[:, arm_id].T
        # 更新参数 b
        self.b[arm_id] += rwd * context[:, arm_id]

    # 重置模型特定的参数
    def _reset_params(self):
        """
        Reset any model-specific parameters. This gets called within the
        public `self.reset()` method.
        """
        # 重置参数 A 和 b
        self.A, self.b = [], []
        # 重置预估值字典
        self.ev_estimates = {}
```