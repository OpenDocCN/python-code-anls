# `numpy-ml\numpy_ml\plots\rl_plots.py`

```
# 禁用 flake8 检查
# 导入 gym 库
import gym

# 导入 Trainer 类
from numpy_ml.rl_models.trainer import Trainer
# 导入不同的智能体类
from numpy_ml.rl_models.agents import (
    CrossEntropyAgent,
    MonteCarloAgent,
    TemporalDifferenceAgent,
    DynaAgent,
)

# 测试交叉熵智能体
def test_cross_entropy_agent():
    seed = 12345
    max_steps = 300
    n_episodes = 50
    retain_prcnt = 0.2
    n_samples_per_episode = 500
    # 创建 LunarLander-v2 环境
    env = gym.make("LunarLander-v2")

    # 创建交叉熵智能体
    agent = CrossEntropyAgent(env, n_samples_per_episode, retain_prcnt)
    trainer = Trainer(agent, env)
    # 训练智能体
    trainer.train(
        n_episodes, max_steps, seed=seed, plot=True, verbose=True, render_every=None
    )

# 测试蒙特卡洛智能体
def test_monte_carlo_agent():
    seed = 12345
    max_steps = 300
    n_episodes = 10000

    epsilon = 0.05
    off_policy = True
    smooth_factor = 0.001
    temporal_discount = 0.95
    # 创建 Copy-v0 环境
    env = gym.make("Copy-v0")

    # 创建蒙特卡洛智能体
    agent = MonteCarloAgent(env, off_policy, temporal_discount, epsilon)
    trainer = Trainer(agent, env)
    # 训练智能体
    trainer.train(
        n_episodes,
        max_steps,
        seed=seed,
        plot=True,
        verbose=True,
        render_every=None,
        smooth_factor=smooth_factor,
    )

# 测试时序差分智能体
def test_temporal_difference_agent():
    seed = 12345
    max_steps = 200
    n_episodes = 5000

    lr = 0.4
    n_tilings = 10
    epsilon = 0.10
    off_policy = True
    grid_dims = [100, 100]
    smooth_factor = 0.005
    temporal_discount = 0.999
    # 创建 LunarLander-v2 环境
    env = gym.make("LunarLander-v2")
    obs_max = 1
    obs_min = -1

    # 创建时序差分智能体
    agent = TemporalDifferenceAgent(
        env,
        lr=lr,
        obs_max=obs_max,
        obs_min=obs_min,
        epsilon=epsilon,
        n_tilings=n_tilings,
        grid_dims=grid_dims,
        off_policy=off_policy,
        temporal_discount=temporal_discount,
    )

    trainer = Trainer(agent, env)
    # 训练智能体
    trainer.train(
        n_episodes,
        max_steps,
        seed=seed,
        plot=True,
        verbose=True,
        render_every=None,
        smooth_factor=smooth_factor,
    )

# 测试 Dyna 智能体
def test_dyna_agent():
    seed = 12345
    # 设置最大步数
    max_steps = 200
    # 设置训练的总回合数
    n_episodes = 150

    # 学习率
    lr = 0.4
    # 是否使用 Q+ 算法
    q_plus = False
    # 瓦片编码的数量
    n_tilings = 10
    # ε-贪心策略中的 ε 值
    epsilon = 0.10
    # 网格维度
    grid_dims = [10, 10]
    # 平滑因子
    smooth_factor = 0.01
    # 时间折扣因子
    temporal_discount = 0.99
    # 探索权重
    explore_weight = 0.05
    # 模拟动作的数量
    n_simulated_actions = 25

    # 观测值的最大值和最小值
    obs_max, obs_min = 1, -1
    # 创建环境
    env = gym.make("Taxi-v2")

    # 创建 DynaAgent 对象
    agent = DynaAgent(
        env,
        lr=lr,
        q_plus=q_plus,
        obs_max=obs_max,
        obs_min=obs_min,
        epsilon=epsilon,
        n_tilings=n_tilings,
        grid_dims=grid_dims,
        explore_weight=explore_weight,
        temporal_discount=temporal_discount,
        n_simulated_actions=n_simulated_actions,
    )

    # 创建 Trainer 对象
    trainer = Trainer(agent, env)
    # 开始训练
    trainer.train(
        n_episodes,
        max_steps,
        seed=seed,
        plot=True,
        verbose=True,
        render_every=None,
        smooth_factor=smooth_factor,
    )
```