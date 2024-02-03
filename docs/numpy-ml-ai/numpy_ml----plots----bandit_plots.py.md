# `numpy-ml\numpy_ml\plots\bandit_plots.py`

```py
# 导入必要的库
from collections import namedtuple

import numpy as np

# 导入多臂赌博机相关的类和函数
from numpy_ml.bandits import (
    MultinomialBandit,
    BernoulliBandit,
    ShortestPathBandit,
    ContextualLinearBandit,
)
from numpy_ml.bandits.trainer import BanditTrainer
from numpy_ml.bandits.policies import (
    EpsilonGreedy,
    UCB1,
    ThompsonSamplingBetaBinomial,
    LinUCB,
)
from numpy_ml.utils.graphs import random_DAG, DiGraph, Edge


# 生成一个随机的多项式多臂赌博机环境
def random_multinomial_mab(n_arms=10, n_choices_per_arm=5, reward_range=[0, 1]):
    """Generate a random multinomial multi-armed bandit environemt"""
    payoffs = []
    payoff_probs = []
    lo, hi = reward_range
    for a in range(n_arms):
        p = np.random.uniform(size=n_choices_per_arm)
        p = p / p.sum()
        r = np.random.uniform(low=lo, high=hi, size=n_choices_per_arm)

        payoffs.append(list(r))
        payoff_probs.append(list(p))

    return MultinomialBandit(payoffs, payoff_probs)


# 生成一个随机的伯努利多臂赌博机环境
def random_bernoulli_mab(n_arms=10):
    """Generate a random Bernoulli multi-armed bandit environemt"""
    p = np.random.uniform(size=n_arms)
    payoff_probs = p / p.sum()
    return BernoulliBandit(payoff_probs)


# 在一个随机的多项式多臂赌博机环境上评估 epsilon-greedy 策略
def plot_epsilon_greedy_multinomial_payoff():
    """
    Evaluate an epsilon-greedy policy on a random multinomial bandit
    problem
    """
    np.random.seed(12345)
    N = np.random.randint(2, 30)  # n arms
    K = np.random.randint(2, 10)  # n payoffs / arm
    ep_length = 1

    rrange = [0, 1]
    n_duplicates = 5
    n_episodes = 5000

    mab = random_multinomial_mab(N, K, rrange)
    policy = EpsilonGreedy(epsilon=0.05, ev_prior=rrange[1] / 2)
    policy = BanditTrainer().train(policy, mab, ep_length, n_episodes, n_duplicates)


# 在一个多项式多臂赌博机环境上评估 UCB1 策略
def plot_ucb1_multinomial_payoff():
    """Evaluate the UCB1 policy on a multinomial bandit environment"""
    np.random.seed(12345)
    N = np.random.randint(2, 30)  # n arms
    K = np.random.randint(2, 10)  # n payoffs / arm
    # 设置每个回合的长度为1
    ep_length = 1

    # 设置探索参数C为1，奖励范围为[0, 1]，重复次数为5，总回合数为5000
    C = 1
    rrange = [0, 1]
    n_duplicates = 5
    n_episodes = 5000

    # 使用随机多项式分布生成多臂赌博机
    mab = random_multinomial_mab(N, K, rrange)
    # 使用UCB1算法初始化策略，设置探索参数C和先验期望值为奖励范围的上限的一半
    policy = UCB1(C=C, ev_prior=rrange[1] / 2)
    # 使用BanditTrainer训练策略，传入策略、多臂赌博机、回合长度、总回合数和重复次数
    policy = BanditTrainer().train(policy, mab, ep_length, n_episodes, n_duplicates)
def plot_thompson_sampling_beta_binomial_payoff():
    """
    Evaluate the ThompsonSamplingBetaBinomial policy on a random Bernoulli
    multi-armed bandit.
    """
    # 设置随机种子
    np.random.seed(12345)
    # 随机生成臂的数量
    N = np.random.randint(2, 30)  # n arms
    ep_length = 1

    n_duplicates = 5
    n_episodes = 5000

    # 创建随机伯努利多臂老虎机
    mab = random_bernoulli_mab(N)
    # 创建 ThompsonSamplingBetaBinomial 策略
    policy = ThompsonSamplingBetaBinomial(alpha=1, beta=1)
    # 训练策略
    policy = BanditTrainer().train(policy, mab, ep_length, n_episodes, n_duplicates)


def plot_lin_ucb():
    """Plot the linUCB policy on a contextual linear bandit problem"""
    # 设置随机种子
    np.random.seed(12345)
    ep_length = 1
    # 随机生成 K 和 D
    K = np.random.randint(2, 25)
    D = np.random.randint(2, 10)

    n_duplicates = 5
    n_episodes = 5000

    # 创建上下文线性老虎机
    cmab = ContextualLinearBandit(K, D, 1)
    # 创建 LinUCB 策略
    policy = LinUCB(alpha=1)
    # 训练策略
    policy = BanditTrainer().train(policy, cmab, ep_length, n_episodes, n_duplicates)


def plot_ucb1_gaussian_shortest_path():
    """
    Plot the UCB1 policy on a graph shortest path problem each edge weight
    drawn from an independent univariate Gaussian
    """
    # 设置随机种子
    np.random.seed(12345)

    ep_length = 1
    n_duplicates = 5
    n_episodes = 5000
    p = np.random.rand()
    n_vertices = np.random.randint(5, 15)

    Gaussian = namedtuple("Gaussian", ["mean", "variance", "EV", "sample"])

    # create randomly-weighted edges
    print("Building graph")
    E = []
    G = random_DAG(n_vertices, p)
    V = G.vertices
    for e in G.edges:
        mean, var = np.random.uniform(0, 1), np.random.uniform(0, 1)
        w = lambda: np.random.normal(mean, var)  # noqa: E731
        rv = Gaussian(mean, var, mean, w)
        E.append(Edge(e.fr, e.to, rv))

    G = DiGraph(V, E)
    while not G.path_exists(V[0], V[-1]):
        print("Skipping")
        idx = np.random.randint(0, len(V))
        V[idx], V[-1] = V[-1], V[idx]

    # 创建最短路径老虎机
    mab = ShortestPathBandit(G, V[0], V[-1])
    # 创建 UCB1 策略
    policy = UCB1(C=1, ev_prior=0.5)
    # 使用BanditTrainer类的train方法对策略进行训练，传入策略、多臂赌博机、每个episode的长度、总共的episode数以及重复次数作为参数
    policy = BanditTrainer().train(policy, mab, ep_length, n_episodes, n_duplicates)
# 定义一个函数用于比较不同策略在相同赌博机问题上的表现
def plot_comparison():
    # 设置随机种子
    np.random.seed(1234)
    # 设置每个回合的长度
    ep_length = 1
    # 设置赌博机的数量
    K = 10

    # 设置重复次数
    n_duplicates = 5
    # 设置回合数
    n_episodes = 5000

    # 创建一个随机伯努利赌博机
    cmab = random_bernoulli_mab(n_arms=K)
    # 创建三种不同的策略
    policy1 = EpsilonGreedy(epsilon=0.05, ev_prior=0.5)
    policy2 = UCB1(C=1, ev_prior=0.5)
    policy3 = ThompsonSamplingBetaBinomial(alpha=1, beta=1)
    policies = [policy1, policy2, policy3]

    # 使用BanditTrainer类的compare方法比较不同策略的表现
    BanditTrainer().compare(
        policies, cmab, ep_length, n_episodes, n_duplicates,
    )
```