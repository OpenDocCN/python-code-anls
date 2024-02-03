# `numpy-ml\numpy_ml\rl_models\trainer.py`

```py
from time import time
import numpy as np

# 定义一个 Trainer 类，用于方便地进行 agent 的训练和评估
class Trainer(object):
    def __init__(self, agent, env):
        """
        An object to facilitate agent training and evaluation.

        Parameters
        ----------
        agent : :class:`AgentBase` instance
            The agent to train.
        env : ``gym.wrappers`` or ``gym.envs`` instance
            The environment to run the agent on.
        """
        # 初始化 Trainer 对象，设置 agent 和 env 属性
        self.env = env
        self.agent = agent
        # 初始化 rewards 字典，用于存储训练过程中的奖励和相关信息
        self.rewards = {"total": [], "smooth_total": [], "n_steps": [], "duration": []}

    def _train_episode(self, max_steps, render_every=None):
        # 记录当前时间
        t0 = time()
        if "train_episode" in dir(self.agent):
            # 如果 agent 中有 train_episode 方法，则在线训练更新
            reward, n_steps = self.agent.train_episode(max_steps)
        else:
            # 如果 agent 中没有 train_episode 方法，则离线训练更新
            reward, n_steps = self.agent.run_episode(max_steps)
            # 更新 agent
            self.agent.update()
        # 计算训练时长
        duration = time() - t0
        return reward, duration, n_steps

    def train(
        self,
        n_episodes,
        max_steps,
        seed=None,
        plot=True,
        verbose=True,
        render_every=None,
        smooth_factor=0.05,
    def plot_rewards(self, rwd_greedy):
        """
        Plot the cumulative reward per episode as a function of episode number.

        Notes
        -----
        Saves plot to the file ``./img/<agent>-<env>.png``

        Parameters
        ----------
        rwd_greedy : float
            The cumulative reward earned with a final execution of a greedy
            target policy.
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            # 设置 seaborn 库的样式为白色
            sns.set_style("white")
            # 设置 seaborn 库的上下文为 notebook，字体大小为 1
            sns.set_context("notebook", font_scale=1)
        except:
            fstr = "Error importing `matplotlib` and `seaborn` -- plotting functionality is disabled"
            # 如果导入 matplotlib 和 seaborn 失败，则抛出 ImportError 异常
            raise ImportError(fstr)

        # 获取累积奖励数据
        R = self.rewards
        # 创建图形和轴对象
        fig, ax = plt.subplots()
        # 创建 x 轴数据，表示每一轮的序号
        x = np.arange(len(R["total"]))
        # 创建 y 轴数据，表示平滑后的累积奖励
        y = R["smooth_total"]
        # 创建 y_raw 轴数据，表示原始的累积奖励
        y_raw = R["total"]

        # 绘制平滑后的累积奖励曲线
        ax.plot(x, y, label="smoothed")
        # 绘制原始的累积奖励曲线，透明度为 0.5
        ax.plot(x, y_raw, alpha=0.5, label="raw")
        # 添加一条虚线，表示最终贪婪策略的累积奖励
        ax.axhline(y=rwd_greedy, xmin=min(x), xmax=max(x), ls=":", label="final greedy")
        # 添加图例
        ax.legend()
        # 移除图形的上边界和右边界
        sns.despine()

        # 获取环境名称和智能体名称
        env = self.agent.env_info["id"]
        agent = self.agent.hyperparameters["agent"]

        # 设置 x 轴标签为 "Episode"
        ax.set_xlabel("Episode")
        # 设置 y 轴标签为 "Cumulative reward"
        ax.set_ylabel("Cumulative reward")
        # 设置图形标题为智能体名称和环境名称的组合
        ax.set_title("{} on '{}'".format(agent, env))
        # 保存图形到文件 img/<agent>-<env>.png
        plt.savefig("img/{}-{}.png".format(agent, env))
        # 关闭所有图形
        plt.close("all")
```