# `numpy-ml\numpy_ml\bandits\trainer.py`

```py
# 为执行和比较多臂赌博（MAB）策略而创建的训练器/运行器对象
import warnings
import os.path as op
from collections import defaultdict

import numpy as np

# 导入自定义的依赖警告类
from numpy_ml.utils.testing import DependencyWarning

# 尝试导入 matplotlib 库，如果导入失败则禁用绘图功能
try:
    import matplotlib.pyplot as plt

    _PLOTTING = True
except ImportError:
    fstr = "Cannot import matplotlib. Plotting functionality disabled."
    warnings.warn(fstr, DependencyWarning)
    _PLOTTING = False


# 返回包含 `trainer.py` 脚本的目录
def get_scriptdir():
    return op.dirname(op.realpath(__file__))


# 计算策略对期望臂收益的估计与真实期望收益之间的均方误差
def mse(bandit, policy):
    if not hasattr(policy, "ev_estimates") or len(policy.ev_estimates) == 0:
        return np.nan

    se = []
    evs = bandit.arm_evs
    ests = sorted(policy.ev_estimates.items(), key=lambda x: x[0])
    for ix, (est, ev) in enumerate(zip(ests, evs)):
        se.append((est[1] - ev) ** 2)
    return np.mean(se)


# 计算前一个值和当前值的简单加权平均
def smooth(prev, cur, weight):
    r"""
    Compute a simple weighted average of the previous and current value.

    Notes
    -----
    The smoothed value at timestep `t`, :math:`\tilde{X}_t` is calculated as

    .. math::

        \tilde{X}_t = \epsilon \tilde{X}_{t-1} + (1 - \epsilon) X_t

    where :math:`X_t` is the value at timestep `t`, :math:`\tilde{X}_{t-1}` is
    the value of the smoothed signal at timestep `t-1`, and :math:`\epsilon` is
    the smoothing weight.

    Parameters
    ----------
    prev : float or :py:class:`ndarray <numpy.ndarray>` of shape `(N,)`
        The value of the smoothed signal at the immediately preceding
        timestep.
    cur : float or :py:class:`ndarray <numpy.ndarray>` of shape `(N,)`
        The value of the signal at the current timestep
    # weight: 平滑权重，可以是浮点数或形状为`(N,)`的数组
    #         值越接近0，平滑效果越弱；值越接近1，平滑效果越强
    #         如果weight是一个数组，每个维度将被解释为一个单独的平滑权重，对应于`cur`中的相应维度

    # 返回值
    # smoothed: 平滑后的信号，可以是浮点数或形状为`(N,)`的数组
    #           返回平滑后的信号
    """
    返回加权平滑后的结果
    """
    return weight * prev + (1 - weight) * cur
class BanditTrainer:
    def __init__(self):
        """
        初始化 BanditTrainer 类的实例，用于多臂赌博机的训练、比较和评估。
        """
        self.logs = {}

    def compare(
        self,
        policies,
        bandit,
        n_trials,
        n_duplicates,
        plot=True,
        seed=None,
        smooth_weight=0.999,
        out_dir=None,
    """
    比较不同策略在给定赌博机上的性能，包括进行多次试验、绘制图表等。
    """

    def train(
        self,
        policy,
        bandit,
        n_trials,
        n_duplicates,
        plot=True,
        axes=None,
        verbose=True,
        print_every=100,
        smooth_weight=0.999,
        out_dir=None,
    """
    在给定赌博机上训练指定策略，包括进行多次试验、绘制图表等。
    """

    def _train_step(self, bandit, policy):
        """
        私有方法，执行单步训练，返回奖励、选择的臂、理想情况下的奖励和臂。
        """
        P, B = policy, bandit
        C = B.get_context() if hasattr(B, "get_context") else None
        rwd, arm = P.act(B, C)
        oracle_rwd, oracle_arm = B.oracle_payoff(C)
        return rwd, arm, oracle_rwd, oracle_arm
    # 初始化日志记录，用于存储训练过程中的数据
    def init_logs(self, policies):
        """
        Initialize the episode logs.

        Notes
        -----
        Training logs are represented as a nested set of dictionaries with the
        following structure:

            log[model_id][metric][trial_number][duplicate_number]

        For example, ``logs['model1']['regret'][3][1]`` holds the regret value
        accrued on the 3rd trial of the 2nd duplicate run for model1.

        Available fields are 'regret', 'cregret' (cumulative regret), 'reward',
        'mse' (mean-squared error between estimated arm EVs and the true EVs),
        'optimal_arm', 'selected_arm', and 'optimal_reward'.
        """
        # 如果输入的策略不是列表，则转换为列表
        if not isinstance(policies, list):
            policies = [policies]

        # 初始化日志记录字典，包含不同策略的不同指标数据
        self.logs = {
            str(p): {
                "mse": defaultdict(lambda: []),
                "regret": defaultdict(lambda: []),
                "reward": defaultdict(lambda: []),
                "cregret": defaultdict(lambda: []),
                "optimal_arm": defaultdict(lambda: []),
                "selected_arm": defaultdict(lambda: []),
                "optimal_reward": defaultdict(lambda: []),
            }
            for p in policies
        }

    # 打印运行摘要信息，包括估计值与真实值的均方误差和遗憾值
    def _print_run_summary(self, bandit, policy, regret):
        # 如果策略没有估计值或估计值为空，则返回空
        if not hasattr(policy, "ev_estimates") or len(policy.ev_estimates) == 0:
            return None

        # 获取臂的真实值和估计值
        evs, se = bandit.arm_evs, []
        fstr = "Arm {}: {:.4f} v. {:.4f}"
        ests = sorted(policy.ev_estimates.items(), key=lambda x: x[0])
        print("\n\nEstimated vs. Real EV\n" + "-" * 21)
        # 打印每个臂的估计值和真实值
        for ix, (est, ev) in enumerate(zip(ests, evs)):
            print(fstr.format(ix + 1, est[1], ev))
            se.append((est[1] - ev) ** 2)
        fstr = "\nFinal MSE: {:.4f}\nFinal Regret: {:.4f}\n\n"
        # 打印最终的均方误差和遗憾值
        print(fstr.format(np.mean(se), regret))
    # 绘制奖励图表，包括平滑处理、对比最优奖励、绘制曲线等
    def _plot_reward(self, optimal_rwd, policy, smooth_weight, axes=None, out_dir=None):
        # 获取指定策略的日志数据
        L = self.logs[str(policy)]
        # 计算平滑后的指标数据
        smds = self._smoothed_metrics(policy, optimal_rwd, smooth_weight)

        # 如果未提供绘图坐标轴，则创建新的图表和子图
        if axes is None:
            fig, [ax1, ax2] = plt.subplots(1, 2)
        else:
            # 如果提供了绘图坐标轴，则确保长度为2
            assert len(axes) == 2
            ax1, ax2 = axes

        # 生成实验次数的范围
        e_ids = range(1, len(L["reward"]) + 1)
        # 定义绘图参数列表
        plot_params = [[ax1, ax2], ["reward", "cregret"], ["b", "r"], [optimal_rwd, 0]]

        # 遍历绘图参数列表，绘制曲线
        for (ax, m, c, opt) in zip(*plot_params):
            avg, std = "sm_{}_avg sm_{}_std".format(m, m).split()
            ax.plot(e_ids, smds[avg], color=c)
            ax.axhline(opt, 0, 1, color=c, ls="--")
            ax.fill_between(
                e_ids,
                smds[avg] + smds[std],
                smds[avg] - smds[std],
                color=c,
                alpha=0.25,
            )
            ax.set_xlabel("Trial")
            # 设置 Y 轴标签
            m = "Cumulative Regret" if m == "cregret" else m
            ax.set_ylabel("Smoothed Avg. {}".format(m.title())

            # 如果未提供绘图坐标轴，则设置纵横比例
            if axes is None:
                ax.set_aspect(np.diff(ax.get_xlim()) / np.diff(ax.get_ylim()))

            # 如果提供了绘图坐标轴，则设置子图标题
            if axes is not None:
                ax.set_title(str(policy))

        # 如果未提供绘图坐标轴，则设置整体标题和布局
        if axes is None:
            fig.suptitle(str(policy))
            fig.tight_layout()

            # 如果指定输出目录，则保存图表
            if out_dir is not None:
                bid = policy.hyperparameters["id"]
                plt.savefig(op.join(out_dir, f"{bid}.png"), dpi=300)
            # 显示图表
            plt.show()
        # 返回绘图坐标轴
        return ax1, ax2
    # 计算平滑后的指标数据
    def _smoothed_metrics(self, policy, optimal_rwd, smooth_weight):
        # 获取指定策略的日志数据
        L = self.logs[str(policy)]

        # 预先分配平滑后的数据结构
        smds = {}
        for m in L.keys():
            # 如果是"selections"则跳过
            if m == "selections":
                continue

            # 初始化平均值的平滑数据
            smds["sm_{}_avg".format(m)] = np.zeros(len(L["reward"]))
            smds["sm_{}_avg".format(m)][0] = np.mean(L[m][1])

            # 初始化标准差的平滑数据
            smds["sm_{}_std".format(m)] = np.zeros(len(L["reward"]))
            smds["sm_{}_std".format(m)][0] = np.std(L[m][1])

        # 复制原始数据到平滑数据
        smoothed = {m: L[m][1] for m in L.keys()}
        # 从第二个元素开始遍历数据
        for e_id in range(2, len(L["reward"]) + 1):
            for m in L.keys():
                # 如果是"selections"则跳过
                if m == "selections":
                    continue
                # 获取前一个和当前的数据
                prev, cur = smoothed[m], L[m][e_id]
                # 对数据进行平滑处理
                smoothed[m] = [smooth(p, c, smooth_weight) for p, c in zip(prev, cur)]
                # 计算平均值和标准差并存储到平滑数据结构中
                smds["sm_{}_avg".format(m)][e_id - 1] = np.mean(smoothed[m])
                smds["sm_{}_std".format(m)][e_id - 1] = np.std(smoothed[m])
        # 返回平滑后的数据结构
        return smds
```