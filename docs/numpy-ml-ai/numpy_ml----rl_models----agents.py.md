# `numpy-ml\numpy_ml\rl_models\agents.py`

```py
# 引入必要的库
from abc import ABC, abstractmethod
from collections import defaultdict
import numpy as np

# 从自定义的 rl_utils 模块中引入 EnvModel, env_stats, tile_state_space
from .rl_utils import EnvModel, env_stats, tile_state_space
# 从自定义的 data_structures 模块中引入 Dict
from ..utils.data_structures import Dict

# 定义一个抽象基类 AgentBase
class AgentBase(ABC):
    # 初始化 AgentBase 类
    def __init__(self, env):
        super().__init__()
        self.env = env
        self.parameters = {}
        self.hyperparameters = {}
        self.derived_variables = {}
        self.env_info = env_stats(env)

    # 创建观测和动作的映射字典
    def _create_2num_dicts(self, obs_encoder=None, act_encoder=None):
        E = self.env_info
        n_states = np.prod(E["n_obs_per_dim"])
        n_actions = np.prod(E["n_actions_per_dim"])

        # 创建动作到标量的字典和标量到动作的字典
        self._num2action = Dict()
        self._action2num = Dict(act_encoder)
        if n_actions != np.inf:
            self._action2num = {act: i for i, act in enumerate(E["action_ids"])}
            self._num2action = {i: act for act, i in self._action2num.items()}

        # 创建观测到标量的字典和标量到观测的字典
        self._num2obs = Dict()
        self._obs2num = Dict(obs_encoder)
        if n_states != np.inf:
            self._obs2num = {act: i for i, act in enumerate(E["obs_ids"])}
            self._num2obs = {i: act for act, i in self._obs2num.items()}

    # 清空历史记录
    def flush_history(self):
        """Clear the episode history"""
        for k, v in self.episode_history.items():
            self.episode_history[k] = []

    # 抽象方法，根据当前观测生成动作
    @abstractmethod
    def act(self, obs):
        """Generate an action given the current observation"""
        raise NotImplementedError

    # 抽象方法，采取贪婪策略
    @abstractmethod
    def greedy_policy(self, **kwargs):
        """
        Take a greedy action.

        Returns
        -------
        total_reward : float
            The total reward on the episode.
        n_steps : float
            The total number of steps taken on the episode.
        """
        raise NotImplementedError

    @abstractmethod
    # 定义一个方法，用于运行 agent 在一个单独的 episode 上
    def run_episode(self, max_steps, render=False):
        """
        Run the agent on a single episode.
    
        Parameters
        ----------
        max_steps : int
            The maximum number of steps to run an episode
        render : bool
            Whether to render the episode during training
    
        Returns
        -------
        reward : float
            The total reward on the episode, averaged over the theta samples.
        steps : float
            The total number of steps taken on the episode, averaged over the
            theta samples.
        """
        # 抛出未实现的错误，需要在子类中实现该方法
        raise NotImplementedError
    
    # 定义一个抽象方法，用于更新 agent 的参数根据当前 episode 上获得的奖励
    @abstractmethod
    def update(self):
        r"""
        Update the agent parameters according to the rewards accrued on the
        current episode.
    
        Returns
        -------
        avg_reward : float
            The average reward earned by the best `retain_prcnt` theta samples
            on the current episode.
        """
        # 抛出未实现的错误，需要在子类中实现该方法
        raise NotImplementedError
class CrossEntropyAgent(AgentBase):
    # 定义交叉熵代理类，继承自AgentBase基类

    def _init_params(self):
        # 初始化参数方法

        E = self.env_info
        # 获取环境信息
        assert not E["continuous_actions"], "Action space must be discrete"
        # 断言动作空间必须是离散的

        self._create_2num_dicts()
        # 调用私有方法创建两个数字字典

        b_len = np.prod(E["n_actions_per_dim"])
        # 计算动作维度的乘积作为b_len
        W_len = b_len * np.prod(E["obs_dim"])
        # 计算观测维度的乘积与b_len相乘作为W_len
        theta_dim = b_len + W_len
        # 计算theta的维度

        # init mean and variance for mv gaussian with dimensions theta_dim
        # 初始化维度为theta_dim的多变量高斯分布的均值和方差
        theta_mean = np.random.rand(theta_dim)
        # 生成theta_dim维度的随机均值
        theta_var = np.ones(theta_dim)
        # 生成theta_dim维度的方差为1的数组

        self.parameters = {"theta_mean": theta_mean, "theta_var": theta_var}
        # 设置参数字典包含均值和方差

        self.derived_variables = {
            "b_len": b_len,
            "W_len": W_len,
            "W_samples": [],
            "b_samples": [],
            "episode_num": 0,
            "cumulative_rewards": [],
        }
        # 设置派生变量字典包含b_len、W_len、W_samples、b_samples、episode_num和cumulative_rewards

        self.hyperparameters = {
            "agent": "CrossEntropyAgent",
            "retain_prcnt": self.retain_prcnt,
            "n_samples_per_episode": self.n_samples_per_episode,
        }
        # 设置超参数字典包含代理名称、保留百分比和每个episode的样本数

        self.episode_history = {"rewards": [], "state_actions": []}
        # 设置episode历史字典包含奖励和状态动作对
    def act(self, obs):
        r"""
        Generate actions according to a softmax policy.

        Notes
        -----
        The softmax policy assumes that the pmf over actions in state :math:`x_t` is
        given by:

        .. math::

            \pi(a | x^{(t)}) = \text{softmax}(
                \text{obs}^{(t)} \cdot \mathbf{W}_i^{(t)} + \mathbf{b}_i^{(t)} )

        where :math:`\mathbf{W}` is a learned weight matrix, `obs` is the observation
        at timestep `t`, and **b** is a learned bias vector.

        Parameters
        ----------
        obs : int or :py:class:`ndarray <numpy.ndarray>`
            An observation from the environment.

        Returns
        -------
        action : int, float, or :py:class:`ndarray <numpy.ndarray>`
            An action sampled from the distribution over actions defined by the
            softmax policy.
        """
        E, P = self.env_info, self.parameters
        W, b = P["W"], P["b"]

        s = self._obs2num[obs]
        s = np.array([s]) if E["obs_dim"] == 1 else s

        # compute softmax
        # 计算 softmax 分布的分子部分
        Z = s.T @ W + b
        # 对分子部分进行指数化，减去最大值以防止数值不稳定
        e_Z = np.exp(Z - np.max(Z, axis=-1, keepdims=True))
        # 计算 softmax 分布
        action_probs = e_Z / e_Z.sum(axis=-1, keepdims=True)

        # sample action
        # 从 softmax 分布中采样一个动作
        a = np.random.multinomial(1, action_probs).argmax()
        # 返回对应动作的编号
        return self._num2action[a]
    # 运行智能体在单个 episode 上的操作

    def run_episode(self, max_steps, render=False):
        """
        Run the agent on a single episode.

        Parameters
        ----------
        max_steps : int
            The maximum number of steps to run an episode
        render : bool
            Whether to render the episode during training

        Returns
        -------
        reward : float
            The total reward on the episode, averaged over the theta samples.
        steps : float
            The total number of steps taken on the episode, averaged over the
            theta samples.
        """
        
        # 从 theta 样本中采样
        self._sample_thetas()

        # 获取环境信息和派生变量
        E, D = self.env_info, self.derived_variables
        n_actions = np.prod(E["n_actions_per_dim"])
        W_len, obs_dim = D["W_len"], E["obs_dim"]
        steps, rewards = [], []

        # 遍历 theta 样本
        for theta in D["theta_samples"]:
            W = theta[:W_len].reshape(obs_dim, n_actions)
            b = theta[W_len:]

            # 运行 episode，获取总奖励和步数
            total_rwd, n_steps = self._episode(W, b, max_steps, render)
            rewards.append(total_rwd)
            steps.append(n_steps)

        # 返回当前 episode 所有样本的平均奖励和平均步数
        D["episode_num"] += 1
        D["cumulative_rewards"] = rewards
        return np.mean(D["cumulative_rewards"]), np.mean(steps)
    def _episode(self, W, b, max_steps, render):
        """
        Run the agent for an episode.

        Parameters
        ----------
        W : :py:class:`ndarray <numpy.ndarray>` of shape `(obs_dim, n_actions)`
            The weights for the softmax policy.
        b : :py:class:`ndarray <numpy.ndarray>` of shape `(bias_len, )`
            The bias for the softmax policy.
        max_steps : int
            The maximum number of steps to run the episode.
        render : bool
            Whether to render the episode during training.

        Returns
        -------
        reward : float
            The total reward on the episode.
        steps : float
            The total number of steps taken on the episode.
        """
        # 初始化奖励列表和状态-动作对列表
        rwds, sa = [], []
        # 获取当前 episode 的历史记录
        H = self.episode_history
        # 初始化总奖励和步数
        total_reward, n_steps = 0.0, 1
        # 重置环境并获取初始观察
        obs = self.env.reset()

        # 更新策略参数
        self.parameters["W"] = W
        self.parameters["b"] = b

        # 循环执行每一步
        for i in range(max_steps):
            # 如果需要渲染环境，则进行渲染
            if render:
                self.env.render()

            # 增加步数计数
            n_steps += 1
            # 根据当前观察选择动作
            action = self.act(obs)
            # 将观察和动作转换为数字编码
            s, a = self._obs2num[obs], self._action2num[action]
            sa.append((s, a))

            # 执行动作，获取下一个观察和奖励
            obs, reward, done, _ = self.env.step(action)
            rwds.append(reward)
            total_reward += reward

            # 如果 episode 结束，则跳出循环
            if done:
                break

        # 将奖励列表和状态-动作对列表添加到历史记录中
        H["rewards"].append(rwds)
        H["state_actions"].append(sa)
        # 返回总奖励和步数
        return total_reward, n_steps
    # 更新 mu 和 Sigma，根据当前 episode 中获得的奖励
    def update(self):
        # 获取派生变量和参数
        D, P = self.derived_variables, self.parameters
        # 计算需要保留的样本数量
        n_retain = int(self.retain_prcnt * self.n_samples_per_episode)

        # 对每个 theta 样本的累积奖励进行排序，从大到小
        sorted_y_val_idxs = np.argsort(D["cumulative_rewards"])[::-1]
        top_idxs = sorted_y_val_idxs[:n_retain]

        # 使用最佳 theta 值更新 theta_mean 和 theta_var
        P["theta_mean"] = np.mean(D["theta_samples"][top_idxs], axis=0)
        P["theta_var"] = np.var(D["theta_samples"][top_idxs], axis=0)

    # 从具有均值为 theta_mean 和协方差为 diag(theta_var) 的多元高斯分布中采样 n_samples_per_episode 个 theta
    def _sample_thetas(self):
        P, N = self.parameters, self.n_samples_per_episode
        Mu, Sigma = P["theta_mean"], np.diag(P["theta_var"])
        # 从多元高斯分布中生成样本
        samples = np.random.multivariate_normal(Mu, Sigma, N)
        # 将生成的样本保存在派生变量中
        self.derived_variables["theta_samples"] = samples
    # 定义一个贪婪策略函数，使用当前代理参数执行
    def greedy_policy(self, max_steps, render=True):
        """
        Execute a greedy policy using the current agent parameters.

        Parameters
        ----------
        max_steps : int
            The maximum number of steps to run the episode.
        render : bool
            Whether to render the episode during execution.

        Returns
        -------
        total_reward : float
            The total reward on the episode.
        n_steps : float
            The total number of steps taken on the episode.
        """
        # 获取环境信息、派生变量和参数
        E, D, P = self.env_info, self.derived_variables, self.parameters
        # 获取参数中的均值和方差
        Mu, Sigma = P["theta_mean"], np.diag(P["theta_var"])
        # 从多元正态分布中采样一个样本
        sample = np.random.multivariate_normal(Mu, Sigma, 1)

        # 获取权重矩阵的长度和观测维度
        W_len, obs_dim = D["W_len"], E["obs_dim"]
        # 计算动作空间的维度
        n_actions = np.prod(E["n_actions_per_dim"])

        # 从样本中提取权重矩阵和偏置向量
        W = sample[0, :W_len].reshape(obs_dim, n_actions)
        b = sample[0, W_len:]
        # 执行一个 episode，返回总奖励和步数
        total_reward, n_steps = self._episode(W, b, max_steps, render)
        # 返回总奖励和步数
        return total_reward, n_steps
class MonteCarloAgent(AgentBase):
    # 定义一个 Monte-Carlo 学习代理类，继承自 AgentBase 类
    def __init__(self, env, off_policy=False, temporal_discount=0.9, epsilon=0.1):
        """
        A Monte-Carlo learning agent trained using either first-visit Monte
        Carlo updates (on-policy) or incremental weighted importance sampling
        (off-policy).

        Parameters
        ----------
        env : :class:`gym.wrappers` or :class:`gym.envs` instance
            The environment to run the agent on.
        off_policy : bool
            Whether to use a behavior policy separate from the target policy
            during training. If False, use the same epsilon-soft policy for
            both behavior and target policies. Default is False.
        temporal_discount : float between [0, 1]
            The discount factor used for downweighting future rewards. Smaller
            values result in greater discounting of future rewards. Default is
            0.9.
        epsilon : float between [0, 1]
            The epsilon value in the epsilon-soft policy. Larger values
            encourage greater exploration during training. Default is 0.1.
        """
        # 初始化 MonteCarloAgent 类的实例
        super().__init__(env)

        # 设置 epsilon 值
        self.epsilon = epsilon
        # 设置是否使用 off-policy
        self.off_policy = off_policy
        # 设置时间折扣因子
        self.temporal_discount = temporal_discount

        # 初始化参数
        self._init_params()
    # 初始化参数
    def _init_params(self):
        # 获取环境信息
        E = self.env_info
        # 确保动作空间是离散的
        assert not E["continuous_actions"], "Action space must be discrete"
        # 确保观察空间是离散的
        assert not E["continuous_observations"], "Observation space must be discrete"

        # 计算状态数量
        n_states = np.prod(E["n_obs_per_dim"])
        # 计算动作数量
        n_actions = np.prod(E["n_actions_per_dim"])

        # 创建状态和动作的映射字典
        self._create_2num_dicts()

        # 行为策略是随机的，epsilon-soft策略
        self.behavior_policy = self.target_policy = self._epsilon_soft_policy
        # 如果是离策略学习
        if self.off_policy:
            # 初始化C矩阵
            self.parameters["C"] = np.zeros((n_states, n_actions))

            # 目标策略是确定性的，贪婪策略
            self.target_policy = self._greedy

        # 初始化Q函数
        self.parameters["Q"] = np.random.rand(n_states, n_actions)

        # 初始化每个状态-动作对的回报对象
        self.derived_variables = {
            "returns": {(s, a): [] for s in range(n_states) for a in range(n_actions)},
            "episode_num": 0,
        }

        # 设置超参数
        self.hyperparameters = {
            "agent": "MonteCarloAgent",
            "epsilon": self.epsilon,
            "off_policy": self.off_policy,
            "temporal_discount": self.temporal_discount,
        }

        # 初始化历史记录
        self.episode_history = {"state_actions": [], "rewards": []}
    # 定义一个贪婪行为策略函数，用于在离策略为真时使用

    def _greedy(self, s, a=None):
        """
        A greedy behavior policy.

        Notes
        -----
        Only used when off-policy is True.

        Parameters
        ----------
        s : int, float, or tuple
            The state number for the current observation, as returned by
            ``self._obs2num[obs]``.
        a : int, float, or tuple
            The action number in the current state, as returned by
            ``self._action2num[obs]``. If None, sample an action from the action
            probabilities in state `s`, otherwise, return the probability of
            action `a` under the greedy policy. Default is None.

        Returns
        -------
        action : int, float, or :py:class:`ndarray <numpy.ndarray>`
            If `a` is None, this is an action sampled from the distribution
            over actions defined by the greedy policy. If `a` is not
            None, this is the probability of `a` under the greedy policy.
        """
        
        # 根据状态 s 对应的 Q 值，找到最大值对应的动作
        a_star = self.parameters["Q"][s, :].argmax()
        
        # 如果 a 为 None，则从贪婪策略中的动作概率分布中随机选择一个动作
        if a is None:
            out = self._num2action[a_star]
        # 如果 a 不为 None，则返回 a 在贪婪策略下的概率
        else:
            out = 1 if a == a_star else 0
        
        # 返回结果
        return out
    # 更新 Q 函数，使用基于策略的首次访问蒙特卡洛更新
    def _on_policy_update(self):
        r"""
        Update the `Q` function using an on-policy first-visit Monte Carlo
        update.

        Notes
        -----
        The on-policy first-visit Monte Carlo update is

        .. math::

            Q'(s, a) \leftarrow
                \text{avg}(\text{reward following first visit to } (s, a)
                \text{ across all episodes})

        RL agents seek to learn action values conditional on subsequent optimal
        behavior, but they need to behave non-optimally in order to explore all
        actions (to find the optimal actions).

        The on-policy approach is a compromise -- it learns action values not
        for the optimal policy, but for a *near*-optimal policy that still
        explores (the epsilon-soft policy).
        """
        # 获取派生变量、参数和历史记录
        D, P, HS = self.derived_variables, self.parameters, self.episode_history

        # 获取历史记录中的奖励和状态-动作对
        ep_rewards = HS["rewards"]
        sa_tuples = set(HS["state_actions"])

        # 找到每个状态-动作对第一次出现的位置
        locs = [HS["state_actions"].index(sa) for sa in sa_tuples]
        # 计算每个状态-动作对的累积回报
        cumulative_returns = [np.sum(ep_rewards[i:]) for i in locs]

        # 使用首次访问回报的平均值更新 Q 值
        for (s, a), cr in zip(sa_tuples, cumulative_returns):
            # 将首次访问回报添加到返回值列表中
            D["returns"][(s, a)].append(cr)
            # 更新 Q 值为返回值列表的平均值
            P["Q"][s, a] = np.mean(D["returns"][(s, a)])
    def _off_policy_update(self):
        """
        Update `Q` using weighted importance sampling.

        Notes
        -----
        In importance sampling updates, we account for the fact that we are
        updating a different policy from the one we used to generate behavior
        by weighting the accumulated rewards by the ratio of the probability of
        the trajectory under the target policy versus its probability under
        the behavior policies. This is known as the importance sampling weight.

        In weighted importance sampling, we scale the accumulated rewards for a
        trajectory by their importance sampling weight, then take the
        *weighted* average using the importance sampling weight. This weighted
        average then becomes the value for the trajectory.

            W   = importance sampling weight
            G_t = total discounted reward from time t until episode end
            C_n = sum of importance weights for the first n rewards

        This algorithm converges to Q* in the limit.
        """
        P = self.parameters
        HS = self.episode_history
        ep_rewards = HS["rewards"]
        T = len(ep_rewards)

        G, W = 0.0, 1.0
        # 从最后一个时间步开始向前遍历
        for t in reversed(range(T)):
            s, a = HS["state_actions"][t]
            # 计算从时间步 t 开始到结束的总折扣奖励
            G = self.temporal_discount * G + ep_rewards[t]
            # 更新状态动作对 (s, a) 的重要性权重和
            P["C"][s, a] += W

            # 使用加权重要性采样更新 Q(s, a)
            P["Q"][s, a] += (W / P["C"][s, a]) * (G - P["Q"][s, a])

            # 将重要性采样比率乘以当前权重
            W *= self.target_policy(s, a) / self.behavior_policy(s, a)

            # 如果权重为零，则终止循环
            if W == 0.0:
                break
    # 定义一个方法，用于执行行为策略，生成训练过程中的动作
    def act(self, obs):
        r"""
        Execute the behavior policy--an :math:`\epsilon`-soft policy used to
        generate actions during training.

        Parameters
        ----------
        obs : int, float, or :py:class:`ndarray <numpy.ndarray>` as returned by ``env.step(action)``
            An observation from the environment.

        Returns
        -------
        action : int, float, or :py:class:`ndarray <numpy.ndarray>`
            An action sampled from the distribution over actions defined by the
            epsilon-soft policy.
        """  # noqa: E501
        # 将观察值转换为数字
        s = self._obs2num[obs]
        # 调用行为策略方法，返回动作
        return self.behavior_policy(s)

    # 运行一个单独的 episode
    def run_episode(self, max_steps, render=False):
        """
        Run the agent on a single episode.

        Parameters
        ----------
        max_steps : int
            The maximum number of steps to run an episode.
        render : bool
            Whether to render the episode during training.

        Returns
        -------
        reward : float
            The total reward on the episode.
        steps : float
            The number of steps taken on the episode.
        """
        # 获取派生变量
        D = self.derived_variables
        # 运行 episode，获取总奖励和步数
        total_rwd, n_steps = self._episode(max_steps, render)

        # 更新 episode 数量
        D["episode_num"] += 1
        # 返回总奖励和步数
        return total_rwd, n_steps
    def _episode(self, max_steps, render):
        """
        Execute agent on an episode.

        Parameters
        ----------
        max_steps : int
            The maximum number of steps to run the episode.
        render : bool
            Whether to render the episode during training.

        Returns
        -------
        reward : float
            The total reward on the episode.
        steps : float
            The number of steps taken on the episode.
        """
        # 重置环境并获取初始观察值
        obs = self.env.reset()
        # 获取当前 episode 的历史记录
        HS = self.episode_history
        # 初始化总奖励和步数
        total_reward, n_steps = 0.0, 0

        # 循环执行每一步直到达到最大步数
        for i in range(max_steps):
            # 如果需要渲染，则显示环境
            if render:
                self.env.render()

            # 增加步数计数
            n_steps += 1
            # 根据当前观察值选择动作
            action = self.act(obs)

            # 将观察值和动作转换为数字
            s = self._obs2num[obs]
            a = self._action2num[action]

            # 存储 (状态, 动作) 元组
            HS["state_actions"].append((s, a))

            # 执行动作并获取奖励等信息
            obs, reward, done, info = self.env.step(action)

            # 记录奖励
            HS["rewards"].append(reward)
            total_reward += reward

            # 如果 episode 结束，则跳出循环
            if done:
                break

        # 返回总奖励和步数
        return total_reward, n_steps

    def update(self):
        """
        Update the parameters of the model following the completion of an
        episode. Flush the episode history after the update is complete.
        """
        # 获取超参数
        H = self.hyperparameters
        # 如果是离线策略更新，则调用离线策略更新方法
        if H["off_policy"]:
            self._off_policy_update()
        else:
            # 否则调用在线策略更新方法
            self._on_policy_update()

        # 清空 episode 历史记录
        self.flush_history()
    def greedy_policy(self, max_steps, render=True):
        """
        Execute a greedy policy using the current agent parameters.

        Parameters
        ----------
        max_steps : int
            The maximum number of steps to run the episode.
        render : bool
            Whether to render the episode during execution.

        Returns
        -------
        total_reward : float
            The total reward on the episode.
        n_steps : float
            The total number of steps taken on the episode.
        """
        # 获取当前的 episode 历史记录
        H = self.episode_history
        # 重置环境并获取初始观察值
        obs = self.env.reset()

        # 初始化总奖励和步数
        total_reward, n_steps = 0.0, 0
        # 循环执行最大步数
        for i in range(max_steps):
            # 如果需要渲染环境，则进行渲染
            if render:
                self.env.render()

            # 增加步数计数
            n_steps += 1
            # 根据当前观察值执行贪婪策略选择动作
            action = self._greedy(obs)

            # 将观察值和动作转换为数字表示
            s = self._obs2num[obs]
            a = self._action2num[action]

            # 存储 (状态, 动作) 元组
            H["state_actions"].append((s, a))

            # 执行动作
            obs, reward, done, info = self.env.step(action)

            # 记录奖励
            H["rewards"].append(reward)
            total_reward += reward

            # 如果 episode 结束，则跳出循环
            if done:
                break

        # 返回总奖励和步数
        return total_reward, n_steps
# 定义一个 TemporalDifferenceAgent 类，继承自 AgentBase 类
class TemporalDifferenceAgent(AgentBase):
    # 初始化函数，接受环境、学习率、探索率、瓦片数、观测最大值、观测最小值、网格维度、是否离线策略、时间折扣等参数
    def __init__(
        self,
        env,
        lr=0.4,
        epsilon=0.1,
        n_tilings=8,
        obs_max=None,
        obs_min=None,
        grid_dims=[8, 8],
        off_policy=False,
        temporal_discount=0.99,
    ):
    # 初始化参数函数
    def _init_params(self):
        # 获取环境信息
        E = self.env_info
        # 断言动作空间必须是离散的
        assert not E["continuous_actions"], "Action space must be discrete"

        obs_encoder = None
        # 如果观测空间是连续的
        if E["continuous_observations"]:
            # 对观测空间进行编码
            obs_encoder, _ = tile_state_space(
                self.env,
                self.env_info,
                self.n_tilings,
                state_action=False,
                obs_max=self.obs_max,
                obs_min=self.obs_min,
                grid_size=self.grid_dims,
            )

        # 创建观测空间到数字的字典
        self._create_2num_dicts(obs_encoder=obs_encoder)

        # 行为策略是随机的，epsilon-soft 策略
        self.behavior_policy = self.target_policy = self._epsilon_soft_policy
        # 如果是离线策略
        if self.off_policy:
            # 目标策略是确定性的，贪婪策略
            self.target_policy = self._greedy

        # 初始化 Q 函数
        self.parameters["Q"] = defaultdict(np.random.rand)

        # 初始化每个状态-动作对的回报对象
        self.derived_variables = {"episode_num": 0}

        # 超参数
        self.hyperparameters = {
            "agent": "TemporalDifferenceAgent",
            "lr": self.lr,
            "obs_max": self.obs_max,
            "obs_min": self.obs_min,
            "epsilon": self.epsilon,
            "n_tilings": self.n_tilings,
            "grid_dims": self.grid_dims,
            "off_policy": self.off_policy,
            "temporal_discount": self.temporal_discount,
        }

        # 记录每一集的历史数据
        self.episode_history = {"state_actions": [], "rewards": []}
    def run_episode(self, max_steps, render=False):
        """
        Run the agent on a single episode without updating the priority queue
        or performing backups.

        Parameters
        ----------
        max_steps : int
            The maximum number of steps to run an episode
        render : bool
            Whether to render the episode during training

        Returns
        -------
        reward : float
            The total reward on the episode, averaged over the theta samples.
        steps : float
            The total number of steps taken on the episode, averaged over the
            theta samples.
        """
        # 调用 _episode 方法运行一个单独的 episode，不更新优先级队列或执行备份
        return self._episode(max_steps, render, update=False)

    def train_episode(self, max_steps, render=False):
        """
        Train the agent on a single episode.

        Parameters
        ----------
        max_steps : int
            The maximum number of steps to run an episode.
        render : bool
            Whether to render the episode during training.

        Returns
        -------
        reward : float
            The total reward on the episode.
        steps : float
            The number of steps taken on the episode.
        """
        # 获取派生变量
        D = self.derived_variables
        # 调用 _episode 方法训练一个单独的 episode
        total_rwd, n_steps = self._episode(max_steps, render, update=True)

        # 更新 episode_num
        D["episode_num"] += 1

        return total_rwd, n_steps
    # 定义一个方法，用于运行或训练智能体在一个 episode 上
    def _episode(self, max_steps, render, update=True):
        """
        Run or train the agent on an episode.

        Parameters
        ----------
        max_steps : int
            The maximum number of steps to run the episode.
        render : bool
            Whether to render the episode during training.
        update : bool
            Whether to perform the Q function backups after each step. Default
            is True.

        Returns
        -------
        reward : float
            The total reward on the episode.
        steps : float
            The number of steps taken on the episode.
        """
        # 清空 episode 历史记录
        self.flush_history()

        # 重置环境并获取初始观察
        obs = self.env.reset()
        HS = self.episode_history

        # 根据当前观察选择动作
        action = self.act(obs)
        s = self._obs2num[obs]
        a = self._action2num[action]

        # 存储初始的 (状态, 动作) 元组
        HS["state_actions"].append((s, a))

        total_reward, n_steps = 0.0, 0
        for i in range(max_steps):
            if render:
                self.env.render()

            # 执行动作
            obs, reward, done, info = self.env.step(action)
            n_steps += 1

            # 记录奖励
            HS["rewards"].append(reward)
            total_reward += reward

            # 生成下一个状态和动作
            action = self.act(obs)
            s_ = self._obs2num[obs] if not done else None
            a_ = self._action2num[action]

            # 存储下一个 (状态, 动作) 元组
            HS["state_actions"].append((s_, a_))

            # 如果需要更新 Q 函数，则执行更新
            if update:
                self.update()

            # 如果 episode 结束，则跳出循环
            if done:
                break

        # 返回总奖励和步数
        return total_reward, n_steps
    def _greedy(self, s, a=None):
        """
        A greedy behavior policy. Only used when off-policy is true.

        Parameters
        ----------
        s : int, float, or tuple
            The state number for the current observation, as returned by
            ``self._obs2num[obs]``
        a : int, float, or tuple
            The action number in the current state, as returned by
            ``self._action2num[obs]``. If None, sample an action from the
            action probabilities in state `s`, otherwise, return the
            probability of action `a` under the greedy policy. Default is None.

        Returns
        -------
        If `a` is None:
        action : int, float, or :py:class:`ndarray <numpy.ndarray>` as returned by ``self._num2action``
            If `a` is None, returns an action sampled from the distribution
            over actions defined by the greedy policy.

        If `a` is not None:
        action_prob : float in range [0, 1]
            If `a` is not None, returns the probability of `a` under the
            greedy policy.
        """  # noqa: E501
        # 获取参数和环境信息
        P, E = self.parameters, self.env_info
        # 计算动作空间的总数
        n_actions = np.prod(E["n_actions_per_dim"])
        # 找到在当前状态下使得 Q 值最大的动作
        a_star = np.argmax([P["Q"][(s, aa)] for aa in range(n_actions)])
        # 如果 a 为 None，则从贪婪策略定义的动作分布中随机选择一个动作
        if a is None:
            out = self._num2action[a_star]
        # 如果 a 不为 None，则返回在贪婪策略下动作 a 的概率
        else:
            out = 1 if a == a_star else 0
        return out
    def _on_policy_update(self, s, a, r, s_, a_):
        """
        Update the Q function using the expected SARSA on-policy TD(0) update:

            Q[s, a] <- Q[s, a] + lr * [
                r + temporal_discount * E[Q[s', a'] | s'] - Q[s, a]
            ]

        where

            E[ Q[s', a'] | s'] is the expected value of the Q function over all
            a_ given that we're in state s' under the current policy

        NB. the expected SARSA update can be used for both on- and off-policy
        methods. In an off-policy context, if the target policy is greedy and
        the expectation is taken wrt. the target policy then the expected SARSA
        update is exactly Q-learning.

        Parameters
        ----------
        s : int as returned by `self._obs2num`
            The id for the state/observation at timestep t-1
        a : int as returned by `self._action2num`
            The id for the action taken at timestep t-1
        r : float
            The reward after taking action `a` in state `s` at timestep t-1
        s_ : int as returned by `self._obs2num`
            The id for the state/observation at timestep t
        a_ : int as returned by `self._action2num`
            The id for the action taken at timestep t
        """
        Q, E, pi = self.parameters["Q"], self.env_info, self.behavior_policy

        # TODO: this assumes that all actions are available in each state
        n_actions = np.prod(E["n_actions_per_dim"])

        # compute the expected value of Q(s', a') given that we are in state s'
        E_Q = np.sum([pi(s_, aa) * Q[(s_, aa)] for aa in range(n_actions)]) if s_ else 0

        # perform the expected SARSA TD(0) update
        qsa = Q[(s, a)]
        Q[(s, a)] = qsa + self.lr * (r + self.temporal_discount * E_Q - qsa)
    def _off_policy_update(self, s, a, r, s_):
        """
        Update the `Q` function using the TD(0) Q-learning update:

            Q[s, a] <- Q[s, a] + lr * (
                r + temporal_discount * max_a { Q[s', a] } - Q[s, a]
            )

        Parameters
        ----------
        s : int as returned by `self._obs2num`
            The id for the state/observation at timestep `t-1`
        a : int as returned by `self._action2num`
            The id for the action taken at timestep `t-1`
        r : float
            The reward after taking action `a` in state `s` at timestep `t-1`
        s_ : int as returned by `self._obs2num`
            The id for the state/observation at timestep `t`
        """
        Q, E = self.parameters["Q"], self.env_info
        n_actions = np.prod(E["n_actions_per_dim"])

        qsa = Q[(s, a)]
        Qs_ = [Q[(s_, aa)] for aa in range(n_actions)] if s_ else [0]
        Q[(s, a)] = qsa + self.lr * (r + self.temporal_discount * np.max(Qs_) - qsa)

    def update(self):
        """Update the parameters of the model online after each new state-action."""
        H, HS = self.hyperparameters, self.episode_history
        (s, a), r = HS["state_actions"][-2], HS["rewards"][-1]
        s_, a_ = HS["state_actions"][-1]

        if H["off_policy"]:
            # 如果是离线策略更新，则调用_off_policy_update函数
            self._off_policy_update(s, a, r, s_)
        else:
            # 如果是在线策略更新，则调用_on_policy_update函数
            self._on_policy_update(s, a, r, s_, a_)
    # 定义一个方法，执行行为策略--一个用于在训练期间生成动作的 :math:`\epsilon`-soft 策略
    def act(self, obs):
        r"""
        Execute the behavior policy--an :math:`\epsilon`-soft policy used to
        generate actions during training.

        Parameters
        ----------
        obs : int, float, or :py:class:`ndarray <numpy.ndarray>` as returned by ``env.step(action)``
            An observation from the environment.

        Returns
        -------
        action : int, float, or :py:class:`ndarray <numpy.ndarray>`
            An action sampled from the distribution over actions defined by the
            epsilon-soft policy.
        """  # noqa: E501
        # 将观察值转换为数字
        s = self._obs2num[obs]
        # 调用行为策略方法，返回动作
        return self.behavior_policy(s)

    # 定义一个方法，执行一个确定性贪婪策略，使用当前代理参数
    def greedy_policy(self, max_steps, render=True):
        """
        Execute a deterministic greedy policy using the current agent
        parameters.

        Parameters
        ----------
        max_steps : int
            The maximum number of steps to run the episode.
        render : bool
            Whether to render the episode during execution.

        Returns
        -------
        total_reward : float
            The total reward on the episode.
        n_steps : float
            The total number of steps taken on the episode.
        """
        # 清空历史记录
        self.flush_history()

        # 获取环境的初始观察值
        H = self.episode_history
        obs = self.env.reset()

        total_reward, n_steps = 0.0, 0
        # 循环执行最大步数
        for i in range(max_steps):
            # 如果需要渲染环境，则渲染
            if render:
                self.env.render()

            # 将观察值转换为数字
            s = self._obs2num[obs]
            # 使用贪婪策略选择动作
            action = self._greedy(s)
            # 将动作转换为数字
            a = self._action2num[action]

            # 存储 (状态, 动作) 元组
            H["state_actions"].append((s, a))

            # 执行动作
            obs, reward, done, info = self.env.step(action)
            n_steps += 1

            # 记录奖励
            H["rewards"].append(reward)
            total_reward += reward

            # 如果完成了一个 episode，则跳出循环
            if done:
                break

        return total_reward, n_steps
# 定义一个名为DynaAgent的类，继承自AgentBase类
class DynaAgent(AgentBase):
    # 初始化方法，接受多个参数
    def __init__(
        self,
        env,  # 环境对象
        lr=0.4,  # 学习率，默认值为0.4
        epsilon=0.1,  # ε-greedy策略中的ε值，默认为0.1
        n_tilings=8,  # 瓦片编码中的瓦片数量，默认为8
        obs_max=None,  # 观测值的最大值，默认为None
        obs_min=None,  # 观测值的最小值，默认为None
        q_plus=False,  # 是否使用Q+学习算法，默认为False
        grid_dims=[8, 8],  # 网格维度，默认为[8, 8]
        explore_weight=0.05,  # 探索权重，默认为0.05
        temporal_discount=0.9,  # 时间折扣因子，默认为0.9
        n_simulated_actions=50,  # 模拟动作的数量，默认为50
    # 初始化参数
    def _init_params(self):
        # 获取环境信息
        E = self.env_info
        # 确保动作空间是离散的
        assert not E["continuous_actions"], "Action space must be discrete"

        # 初始化观测编码器
        obs_encoder = None
        # 如果观测是连续的
        if E["continuous_observations"]:
            # 对状态空间进行切片
            obs_encoder, _ = tile_state_space(
                self.env,
                self.env_info,
                self.n_tilings,
                state_action=False,
                obs_max=self.obs_max,
                obs_min=self.obs_min,
                grid_size=self.grid_dims,
            )

        # 创建状态编码器和动作编码器的字典
        self._create_2num_dicts(obs_encoder=obs_encoder)
        # 设置行为策略和目标策略为 epsilon-soft 策略
        self.behavior_policy = self.target_policy = self._epsilon_soft_policy

        # 初始化 Q 函数和模型
        self.parameters["Q"] = defaultdict(np.random.rand)
        self.parameters["model"] = EnvModel()

        # 初始化每个状态-动作对的返回对象
        self.derived_variables = {
            "episode_num": 0,
            "sweep_queue": {},
            "visited": set(),
            "steps_since_last_visit": defaultdict(lambda: 0),
        }

        # 如果使用 Q+ 算法
        if self.q_plus:
            self.derived_variables["steps_since_last_visit"] = defaultdict(
                np.random.rand,
            )

        # 设置超参数
        self.hyperparameters = {
            "agent": "DynaAgent",
            "lr": self.lr,
            "q_plus": self.q_plus,
            "obs_max": self.obs_max,
            "obs_min": self.obs_min,
            "epsilon": self.epsilon,
            "n_tilings": self.n_tilings,
            "grid_dims": self.grid_dims,
            "explore_weight": self.explore_weight,
            "temporal_discount": self.temporal_discount,
            "n_simulated_actions": self.n_simulated_actions,
        }

        # 初始化每一集的历史记录
        self.episode_history = {"state_actions": [], "rewards": []}
    # 执行行为策略--一个用于在训练期间生成动作的ε-soft策略
    def act(self, obs):
        # 将环境返回的观测转换为数字形式
        s = self._obs2num[obs]
        # 从由ε-soft策略定义的动作分布中采样一个动作
        return self.behavior_policy(s)
    # 定义一个贪婪的行为策略函数
    def _greedy(self, s, a=None):
        """
        A greedy behavior policy.

        Parameters
        ----------
        s : int, float, or tuple
            The state number for the current observation, as returned by
            self._obs2num[obs]
        a : int, float, or tuple
            The action number in the current state, as returned by
            self._action2num[obs]. If None, sample an action from the action
            probabilities in state s, otherwise, return the probability of
            action `a` under the greedy policy. Default is None.

        Returns
        -------
        If `a` is None:
        action : int, float, or :py:class:`ndarray <numpy.ndarray>` as returned by :meth:`_num2action`
            If `a` is None, returns an action sampled from the distribution
            over actions defined by the greedy policy.

        If `a` is not None:
        action_prob : float in range [0, 1]
            If `a` is not None, returns the probability of `a` under the
            greedy policy.
        """  # noqa: E501
        # 获取环境信息和 Q 值
        E, Q = self.env_info, self.parameters["Q"]
        # 计算动作空间的总数
        n_actions = np.prod(E["n_actions_per_dim"])
        # 找到在当前状态下使 Q 值最大的动作
        a_star = np.argmax([Q[(s, aa)] for aa in range(n_actions)])
        # 如果 a 为 None，则从贪婪策略定义的动作分布中随机选择一个动作
        if a is None:
            out = self._num2action[a_star]
        # 如果 a 不为 None，则返回 a 在贪婪策略下的概率
        else:
            out = 1 if a == a_star else 0
        return out
    def update(self):
        """
        Update the priority queue with the most recent (state, action) pair and
        perform random-sample one-step tabular Q-planning.

        Notes
        -----
        The planning algorithm uses a priority queue to retrieve the
        state-action pairs from the agent's history which will result in the
        largest change to its `Q`-value if backed up. When the first pair in
        the queue is backed up, the effect on each of its predecessor pairs is
        computed. If the predecessor's priority is greater than a small
        threshold the pair is added to the queue and the process is repeated
        until either the queue is empty or we exceed `n_simulated_actions`
        updates.
        """
        # 获取最近的 (state, action) 对
        s, a = self.episode_history["state_actions"][-1]
        # 更新优先级队列
        self._update_queue(s, a)
        # 模拟行为
        self._simulate_behavior()

    def _update_queue(self, s, a):
        """
        Update the priority queue by calculating the priority for (s, a) and
        inserting it into the queue if it exceeds a fixed (small) threshold.

        Parameters
        ----------
        s : int as returned by `self._obs2num`
            The id for the state/observation
        a : int as returned by `self._action2num`
            The id for the action taken from state `s`
        """
        # 获取派生变量中的优先级队列
        sweep_queue = self.derived_variables["sweep_queue"]

        # TODO: what's a good threshold here?
        # 计算 (s, a) 的优先级
        priority = self._calc_priority(s, a)
        # 如果优先级大于等于 0.001，则插入到优先级队列中
        if priority >= 0.001:
            if (s, a) in sweep_queue:
                sweep_queue[(s, a)] = max(priority, sweep_queue[(s, a)])
            else:
                sweep_queue[(s, a)] = priority
    def _calc_priority(self, s, a):
        """
        计算状态动作对 (s, a) 的“优先级”。优先级 P 定义为：

            P = sum_{s_} p(s_) * abs(r + temporal_discount * max_a {Q[s_, a]} - Q[s, a])

        这对应于 TD(0) Q-learning 对 (s, a) 的绝对值大小的备份。

        Parameters
        ----------
        s : int as returned by `self._obs2num`
            状态/观察的 id
        a : int as returned by `self._action2num`
            从状态 `s` 中采取的动作的 id

        Returns
        -------
        priority : float
            (s, a) 的全备份 TD(0) Q-learning 更新的绝对值大小
        """
        priority = 0.0
        E = self.env_info
        Q = self.parameters["Q"]
        env_model = self.parameters["model"]
        n_actions = np.prod(E["n_actions_per_dim"])

        outcome_probs = env_model.outcome_probs(s, a)
        for (r, s_), p_rs_ in outcome_probs:
            max_q = np.max([Q[(s_, aa)] for aa in range(n_actions)])
            P = p_rs_ * (r + self.temporal_discount * max_q - Q[(s, a)])
            priority += np.abs(P)
        return priority
    def _simulate_behavior(self):
        """
        Perform random-sample one-step tabular Q-planning with prioritized
        sweeping.

        Notes
        -----
        This approach uses a priority queue to retrieve the state-action pairs
        from the agent's history with largest change to their Q-values if
        backed up. When the first pair in the queue is backed up, the effect on
        each of its predecessor pairs is computed. If the predecessor's
        priority is greater than a small threshold the pair is added to the
        queue and the process is repeated until either the queue is empty or we
        have exceeded a `n_simulated_actions` updates.
        """
        # 获取环境模型和优先级队列
        env_model = self.parameters["model"]
        sweep_queue = self.derived_variables["sweep_queue"]
        # 进行一定次数的模拟行为
        for _ in range(self.n_simulated_actions):
            # 如果队列为空，则结束模拟
            if len(sweep_queue) == 0:
                break

            # 从队列中选择具有最大更新（优先级）的（s, a）对
            sq_items = list(sweep_queue.items())
            (s_sim, a_sim), _ = sorted(sq_items, key=lambda x: x[1], reverse=True)[0]

            # 从队列中删除条目
            del sweep_queue[(s_sim, a_sim)]

            # 使用完全备份版本的TD(0) Q-learning更新为（s_sim, a_sim）更新Q函数
            self._update(s_sim, a_sim)

            # 获取导致s_sim的所有(_s, _a)对（即s_sim的前导状态）
            pairs = env_model.state_action_pairs_leading_to_outcome(s_sim)

            # 如果前导状态的优先级超过阈值，则将其添加到队列中
            for (_s, _a) in pairs:
                self._update_queue(_s, _a)
    def _update(self, s, a):
        """
        Update Q using a full-backup version of the TD(0) Q-learning update:

            Q(s, a) = Q(s, a) + lr *
                sum_{r, s'} [
                    p(r, s' | s, a) * (r + gamma * max_a { Q(s', a) } - Q(s, a))
                ]

        Parameters
        ----------
        s : int as returned by ``self._obs2num``
            The id for the state/observation
        a : int as returned by ``self._action2num``
            The id for the action taken from state `s`
        """
        # 初始化更新值为0
        update = 0.0
        # 获取环境模型、环境信息、派生变量和Q值
        env_model = self.parameters["model"]
        E, D, Q = self.env_info, self.derived_variables, self.parameters["Q"]
        # 计算动作空间的大小
        n_actions = np.prod(E["n_actions_per_dim"])

        # 从模型中采样奖励
        outcome_probs = env_model.outcome_probs(s, a)
        for (r, s_), p_rs_ in outcome_probs:
            # 如果启用Q+算法，根据上次访问时间给奖励加上一个“奖励”
            if self.q_plus:
                r += self.explore_weight * np.sqrt(D["steps_since_last_visit"][(s, a)])

            # 计算下一个状态的最大Q值
            max_q = np.max([Q[(s_, a_)] for a_ in range(n_actions)])
            # 更新值根据TD(0) Q-learning更新公式计算
            update += p_rs_ * (r + self.temporal_discount * max_q - Q[(s, a)])

        # 更新Q值
        Q[(s, a)] += self.lr * update

    def run_episode(self, max_steps, render=False):
        """
        Run the agent on a single episode without performing `Q`-function
        backups.

        Parameters
        ----------
        max_steps : int
            The maximum number of steps to run an episode.
        render : bool
            Whether to render the episode during training.

        Returns
        -------
        reward : float
            The total reward on the episode.
        steps : float
            The number of steps taken on the episode.
        """
        # 运行一个不执行Q函数备份的单个episode
        return self._episode(max_steps, render, update=False)
    def train_episode(self, max_steps, render=False):
        """
        Train the agent on a single episode.

        Parameters
        ----------
        max_steps : int
            The maximum number of steps to run an episode.
        render : bool
            Whether to render the episode during training.

        Returns
        -------
        reward : float
            The total reward on the episode.
        steps : float
            The number of steps taken on the episode.
        """
        # 获取派生变量
        D = self.derived_variables
        # 在一个 episode 上运行 _episode 方法，返回总奖励和步数
        total_rwd, n_steps = self._episode(max_steps, render, update=True)
        # 增加 episode_num 计数
        D["episode_num"] += 1
        # 返回总奖励和步数
        return total_rwd, n_steps

    def greedy_policy(self, max_steps, render=True):
        """
        Execute a deterministic greedy policy using the current agent
        parameters.

        Parameters
        ----------
        max_steps : int
            The maximum number of steps to run the episode.
        render : bool
            Whether to render the episode during execution.

        Returns
        -------
        total_reward : float
            The total reward on the episode.
        n_steps : float
            The total number of steps taken on the episode.
        """
        # 清空历史记录
        self.flush_history()

        # 获取 episode_history
        H = self.episode_history
        # 重置环境并获取初始观察
        obs = self.env.reset()

        total_reward, n_steps = 0.0, 0
        for i in range(max_steps):
            # 如果需要渲染，显示环境
            if render:
                self.env.render()

            # 将观察转换为数字
            s = self._obs2num[obs]
            # 使用贪婪策略选择动作
            action = self._greedy(s)
            # 将动作转换为数字
            a = self._action2num[action]

            # 存储 (状态, 动作) 元组
            H["state_actions"].append((s, a))

            # 执行动作
            obs, reward, done, info = self.env.step(action)
            n_steps += 1

            # 记录奖励
            H["rewards"].append(reward)
            total_reward += reward

            # 如果 episode 结束，跳出循环
            if done:
                break

        # 返回总奖励和步数
        return total_reward, n_steps
```