# `numpy-ml\numpy_ml\neural_nets\schedulers\schedulers.py`

```py
from copy import deepcopy
from abc import ABC, abstractmethod

import numpy as np

from math import erf

# 定义一个函数，计算从具有均值`mean`和方差`var`的一维高斯分布中随机抽取的值小于或等于`x`的概率
def gaussian_cdf(x, mean, var):
    eps = np.finfo(float).eps
    x_scaled = (x - mean) / np.sqrt(var + eps)
    return (1 + erf(x_scaled / np.sqrt(2))) / 2

# 定义一个抽象基类，用于所有调度器对象的基类
class SchedulerBase(ABC):
    def __init__(self):
        """Abstract base class for all Scheduler objects."""
        self.hyperparameters = {}

    def __call__(self, step=None, cur_loss=None):
        return self.learning_rate(step=step, cur_loss=cur_loss)

    def copy(self):
        """Return a copy of the current object."""
        return deepcopy(self)

    def set_params(self, hparam_dict):
        """Set the scheduler hyperparameters from a dictionary."""
        if hparam_dict is not None:
            for k, v in hparam_dict.items():
                if k in self.hyperparameters:
                    self.hyperparameters[k] = v

    @abstractmethod
    def learning_rate(self, step=None):
        raise NotImplementedError

# 定义一个常数调度器类，继承自SchedulerBase
class ConstantScheduler(SchedulerBase):
    def __init__(self, lr=0.01, **kwargs):
        """
        Returns a fixed learning rate, regardless of the current step.

        Parameters
        ----------
        initial_lr : float
            The learning rate. Default is 0.01
        """
        super().__init__()
        self.lr = lr
        self.hyperparameters = {"id": "ConstantScheduler", "lr": self.lr}

    def __str__(self):
        return "ConstantScheduler(lr={})".format(self.lr)

    def learning_rate(self, **kwargs):
        """
        Return the current learning rate.

        Returns
        -------
        lr : float
            The learning rate
        """
        return self.lr

# 定义一个指数调度器类，继承自SchedulerBase
class ExponentialScheduler(SchedulerBase:
    # 初始化指数学习率调度器
    def __init__(
        self, initial_lr=0.01, stage_length=500, staircase=False, decay=0.1, **kwargs
    ):
        """
        An exponential learning rate scheduler.

        Notes
        -----
        The exponential scheduler decays the learning rate by `decay` every
        `stage_length` steps, starting from `initial_lr`::

            learning_rate = initial_lr * decay ** curr_stage

        where::

            curr_stage = step / stage_length          if staircase = False
            curr_stage = floor(step / stage_length)   if staircase = True

        Parameters
        ----------
        initial_lr : float
            The learning rate at the first step. Default is 0.01.
        stage_length : int
            The length of each stage, in steps. Default is 500.
        staircase : bool
            If True, only adjusts the learning rate at the stage transitions,
            producing a step-like decay schedule. If False, adjusts the
            learning rate after each step, creating a smooth decay schedule.
            Default is False.
        decay : float
            The amount to decay the learning rate at each new stage. Default is
            0.1.
        """
        # 调用父类的初始化方法
        super().__init__()
        # 设置学习率衰减值
        self.decay = decay
        # 设置是否阶梯式调整学习率
        self.staircase = staircase
        # 设置初始学习率
        self.initial_lr = initial_lr
        # 设置每个阶段的长度
        self.stage_length = stage_length
        # 设置超参数字典
        self.hyperparameters = {
            "id": "StepScheduler",
            "decay": self.decay,
            "staircase": self.staircase,
            "initial_lr": self.initial_lr,
            "stage_length": self.stage_length,
        }

    # 返回调度器的字符串表示
    def __str__(self):
        return "ExponentialScheduler(initial_lr={}, stage_length={}, staircase={}, decay={})".format(
            self.initial_lr, self.stage_length, self.staircase, self.decay
        )
    # 定义一个方法，根据步数返回当前的学习率
    def learning_rate(self, step, **kwargs):
        """
        Return the current learning rate as a function of `step`.

        Parameters
        ----------
        step : int
            The current step number.

        Returns
        -------
        lr : float
            The learning rate for the current step.
        """
        # 计算当前阶段，即步数除以阶段长度
        cur_stage = step / self.stage_length
        # 如果采用阶梯式学习率衰减，则取当前阶段的下限值
        if self.staircase:
            cur_stage = np.floor(cur_stage)
        # 返回当前步数对应的学习率，根据初始学习率和衰减率计算
        return self.initial_lr * self.decay ** cur_stage
class NoamScheduler(SchedulerBase):
    def __init__(self, model_dim=512, scale_factor=1, warmup_steps=4000, **kwargs):
        """
        The Noam learning rate scheduler, originally used in conjunction with
        the Adam optimizer in [1].

        Notes
        -----
        The Noam scheduler increases the learning rate linearly for the first
        `warmup_steps` steps, and decreases it thereafter proportionally to the
        inverse square root of the step number::

            lr = scale_factor * ( (model_dim ** (-0.5)) * adj_step )
            adj_step = min(step_num ** (-0.5), step_num * warmup_steps ** (-1.5))

        References
        ----------
        .. [1] Vaswani et al. (2017) "Attention is all you need". *31st
           Conference on Neural Information Processing Systems*,
           https://arxiv.org/pdf/1706.03762.pdf

        Parameters
        ----------
        model_dim : int
            The number of units in the layer output. Default is 512.
        scale_factor : float
            A fixed coefficient for rescaling the final learning rate. Default
            is 1.
        warmup_steps : int
            The number of steps in the warmup stage of training. Default is
            4000.
        """
        # 调用父类的构造函数
        super().__init__()
        # 初始化 NoamScheduler 的属性
        self.model_dim = model_dim
        self.scale_factor = scale_factor
        self.warmup_steps = warmup_steps
        self.hyperparameters = {
            "id": "NoamScheduler",
            "model_dim": self.model_dim,
            "scale_factor": self.scale_factor,
            "warmup_steps": self.warmup_steps,
        }

    def __str__(self):
        # 返回 NoamScheduler 对象的字符串表示形式
        return "NoamScheduler(model_dim={}, scale_factor={}, warmup_steps={})".format(
            self.model_dim, self.scale_factor, self.warmup_steps
        )
    # 定义学习率函数，根据当前步数和额外参数计算学习率
    def learning_rate(self, step, **kwargs):
        # 获取预热步数和模型维度
        warmup, d_model = self.warmup_steps, self.model_dim
        # 根据论文提出的公式计算新的学习率
        new_lr = d_model ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
        # 返回经过缩放因子调整后的新学习率
        return self.scale_factor * new_lr
# 定义 KingScheduler 类，继承自 SchedulerBase 类
class KingScheduler(SchedulerBase):
    # 初始化方法，设置学习率初始值、耐心值、衰减率等参数
    def __init__(self, initial_lr=0.01, patience=1000, decay=0.99, **kwargs):
        """
        The Davis King / DLib learning rate scheduler.

        Notes
        -----
        The KingScheduler computes the probability that the slope of the OLS
        fit to the loss history is negative. If the probability that it is
        negative is less than 51% over the last `patience` steps, the scheduler
        exponentially decreases the current learning rate by `decay`.

        References
        ----------
        .. [1] King, D. (2018). "Automatic learning rate scheduling that really
           works". http://blog.dlib.net/2018/02/automatic-learning-rate-scheduling-that.html

        Parameters
        ----------
        initial_lr : float
            The learning rate to begin at. Default is 0.01.
        patience : int
            Amount of time to maintain the current learning rate without a
            decrease in loss before adjustment. Default is 1000.
        decay : float
            The amount to decay the learning rate at each new stage. Default is
            0.99.
        """
        # 调用父类的初始化方法
        super().__init__()
        # 设置衰减率、耐心值、学习率初始值等属性
        self.decay = decay
        self.patience = patience
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        # 计算历史记录的最大长度
        self.max_history = np.ceil(1.1 * (patience + 1)).astype(int)

        # 初始化损失历史记录和超参数字典
        self.loss_history = []
        self.hyperparameters = {
            "id": "KingScheduler",
            "decay": self.decay,
            "patience": self.patience,
            "initial_lr": self.initial_lr,
        }

    # 定义 __str__ 方法，返回 KingScheduler 对象的字符串表示
    def __str__(self):
        return "KingScheduler(initial_lr={}, patience={}, decay={})".format(
            self.initial_lr, self.patience, self.decay
        )
    # 返回最大的时间步数，其中`P(loss is decreasing) < 0.51`
    def _steps_without_decrease(self, robust=False, check_all=False):
        """
        Returns the maximum number of timesteps for which `P(loss is decreasing)
        < 0.51`.

        Parameters
        ----------
        robust : bool
            If `robust=True`, first filter out the largest 10% of the loss
            values to remove transient spikes in the loss due to, e.g., a few
            bad minibatches. Default is False.
        check_all : bool
            If False, returns the maximum number of timesteps for which P(loss
            is decreasing) < 0.51. If True, only checks whether the number of
            timesteps for which P(loss is decreasing) < 0.51 is equal to
            ``self.patience``. The former provides more information but is
            significantly more computationally expensive.  Default is False.

        Returns
        -------
        steps_without_decrease: int
            The maximum number of steps back in loss_history for which P(loss
            is decreasing) < 0.51.
        """
        # 将损失历史转换为 NumPy 数组
        lh = np.array(self.loss_history)

        # 如果 robust 为 True，则过滤掉损失值中最大的 10%，以消除由于一些坏的小批次导致的损失暂时性波动
        if robust:
            thresh = np.quantile(lh, 0.9)
            lh = np.array([i for i in lh if i <= thresh])

        # 获取损失历史的长度
        N = len(lh)
        steps_without_decrease = 0
        # 如果 check_all 为 True，则遍历损失历史，找到 P(loss is decreasing) < 0.51 的最大时间步数
        if check_all:
            for i in reversed(range(N - 2)):
                if self._p_decreasing(lh, i) < 0.51:
                    steps_without_decrease = N - i
        # 如果 check_all 为 False，则只检查 P(loss is decreasing) < 0.51 的时间步数是否等于 self.patience
        else:
            i = max(0, N - self.patience - 1)
            if self._p_decreasing(lh, i) < 0.51:
                steps_without_decrease = N - i
        return steps_without_decrease
    def _p_decreasing(self, loss_history, i):
        """
        Compute the probability that the slope of the OLS fit to the loss
        history is negative.

        Parameters
        ----------
        loss_history : numpy array of shape (N,)
            The sequence of loss values for the previous `N` minibatches.
        i : int
            Compute P(Slope < 0) beginning at index i in `history`.

        Returns
        ------
        p_decreasing : float
            The probability that the slope of the OLS fit to loss_history is
            less than or equal to 0.
        """
        # 从索引 i 开始截取 loss_history，得到截取后的 loss 数组
        loss = loss_history[i:]
        # 获取截取后的 loss 数组的长度
        N = len(loss)

        # 对 loss 数组执行最小二乘法（OLS），计算斜率均值
        X = np.c_[np.ones(N), np.arange(i, len(loss_history))]
        intercept, s_mean = np.linalg.inv(X.T @ X) @ X.T @ loss
        # 计算预测的 loss 值
        loss_pred = s_mean * X[:, 1] + intercept

        # 计算我们的 loss 预测的方差，并用此计算（无偏）斜率方差的估计值
        loss_var = 1 / (N - 2) * np.sum((loss - loss_pred) ** 2)
        s_var = (12 * loss_var) / (N ** 3 - N)

        # 计算从以 s_mean 和 s_var 为参数的高斯分布中随机抽取的样本小于或等于 0 的概率
        p_decreasing = gaussian_cdf(0, s_mean, s_var)
        return p_decreasing
    # 计算当前步数和损失值对应的更新后学习率

    def learning_rate(self, step, cur_loss):
        """
        Compute the updated learning rate for the current step and loss.

        Parameters
        ----------
        step : int
            The current step number. Unused.
            当前步数，未使用
        cur_loss : float
            The loss at the current step.
            当前步数的损失值

        Returns
        -------
        lr : float
            The learning rate for the current step.
            当前步数的学习率
        """
        # 如果当前损失值为空，则抛出数值错误
        if cur_loss is None:
            raise ValueError("cur_loss must be a float, but got {}".format(cur_loss))

        # 如果没有属性"max_history"，则初始化为1.1倍的(patience + 1)的向上取整值
        if not hasattr(self, "max_history"):
            self.max_history = np.ceil(1.1 * (self.patience + 1)).astype(int)
        patience, max_history = self.patience, self.max_history

        # 将当前损失值添加到损失历史记录中
        self.loss_history.append(cur_loss)
        # 如果损失历史记录长度小于patience，则返回当前学习率
        if len(self.loss_history) < patience:
            return self.current_lr
        # 保留最近的max_history个损失值
        self.loss_history = self.loss_history[-max_history:]

        # 如果损失值连续patience步没有减小，则降低学习率
        if (
            self._steps_without_decrease() > patience
            and self._steps_without_decrease(robust=True) > patience
        ):
            self.current_lr *= self.decay

        return self.current_lr
```