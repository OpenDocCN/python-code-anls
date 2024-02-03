# `.\PaddleOCR\ppocr\optimizer\learning_rate.py`

```py
# 版权声明，版权归 PaddlePaddle 作者所有
#
# 根据 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 可以在以下网址获取许可证的副本
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件
# 均按“原样”分发，不附带任何担保或条件，无论是明示还是暗示的
# 请查看许可证以获取有关权限和限制的详细信息

# 导入未来的绝对导入、除法、打印函数和 Unicode 字符串
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# 从 Paddle 的优化器中导入学习率
from paddle.optimizer import lr
# 从当前目录下的 lr_scheduler 模块中导入 CyclicalCosineDecay、OneCycleDecay、TwoStepCosineDecay 类
from .lr_scheduler import CyclicalCosineDecay, OneCycleDecay, TwoStepCosineDecay

# 定义 Linear 类
class Linear(object):
    """
    Linear learning rate decay
    Args:
        lr (float): The initial learning rate. It is a python float number.
        epochs(int): The decay step size. It determines the decay cycle.
        end_lr(float, optional): The minimum final learning rate. Default: 0.0001.
        power(float, optional): Power of polynomial. Default: 1.0.
        last_epoch (int, optional):  The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
    """

    # 初始化 Linear 类
    def __init__(self,
                 learning_rate,
                 epochs,
                 step_each_epoch,
                 end_lr=0.0,
                 power=1.0,
                 warmup_epoch=0,
                 last_epoch=-1,
                 **kwargs):
        # 调用父类的初始化方法
        super(Linear, self).__init__()
        # 设置初始学习率
        self.learning_rate = learning_rate
        # 计算总的训练步数
        self.epochs = epochs * step_each_epoch
        # 设置最终学习率
        self.end_lr = end_lr
        # 设置多项式的幂次
        self.power = power
        # 设置最后一个周期的索引
        self.last_epoch = last_epoch
        # 计算热身训练的步数
        self.warmup_epoch = round(warmup_epoch * step_each_epoch)
    # 定义一个 __call__ 方法，用于调用对象
    def __call__(self):
        # 使用多项式衰减方式设置学习率
        learning_rate = lr.PolynomialDecay(
            learning_rate=self.learning_rate,
            decay_steps=self.epochs,
            end_lr=self.end_lr,
            power=self.power,
            last_epoch=self.last_epoch)
        # 如果存在预热周期
        if self.warmup_epoch > 0:
            # 使用线性预热方式设置学习率
            learning_rate = lr.LinearWarmup(
                learning_rate=learning_rate,
                warmup_steps=self.warmup_epoch,
                start_lr=0.0,
                end_lr=self.learning_rate,
                last_epoch=self.last_epoch)
        # 返回学习率对象
        return learning_rate
class Cosine(object):
    """
    Cosine learning rate decay
    lr = 0.05 * (math.cos(epoch * (math.pi / epochs)) + 1)
    Args:
        lr(float): initial learning rate
        step_each_epoch(int): steps each epoch
        epochs(int): total training epochs
        last_epoch (int, optional):  The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
    """

    def __init__(self,
                 learning_rate,
                 step_each_epoch,
                 epochs,
                 warmup_epoch=0,
                 last_epoch=-1,
                 **kwargs):
        super(Cosine, self).__init__()
        self.learning_rate = learning_rate
        self.T_max = step_each_epoch * epochs
        self.last_epoch = last_epoch
        self.warmup_epoch = round(warmup_epoch * step_each_epoch)

    def __call__(self):
        learning_rate = lr.CosineAnnealingDecay(
            learning_rate=self.learning_rate,
            T_max=self.T_max,
            last_epoch=self.last_epoch)
        if self.warmup_epoch > 0:
            learning_rate = lr.LinearWarmup(
                learning_rate=learning_rate,
                warmup_steps=self.warmup_epoch,
                start_lr=0.0,
                end_lr=self.learning_rate,
                last_epoch=self.last_epoch)
        return learning_rate


class Step(object):
    """
    Piecewise learning rate decay
    Args:
        step_each_epoch(int): steps each epoch
        learning_rate (float): The initial learning rate. It is a python float number.
        step_size (int): the interval to update.
        gamma (float, optional): The Ratio that the learning rate will be reduced. ``new_lr = origin_lr * gamma`` .
            It should be less than 1.0. Default: 0.1.
        last_epoch (int, optional):  The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
    """
    # 初始化函数，设置学习率、步长、每个epoch的步数、衰减因子、热身epoch数、上一个epoch数等参数
    def __init__(self,
                 learning_rate,
                 step_size,
                 step_each_epoch,
                 gamma,
                 warmup_epoch=0,
                 last_epoch=-1,
                 **kwargs):
        # 调用父类的初始化函数
        super(Step, self).__init__()
        # 计算总的步数
        self.step_size = step_each_epoch * step_size
        # 设置学习率
        self.learning_rate = learning_rate
        # 设置衰减因子
        self.gamma = gamma
        # 设置上一个epoch数
        self.last_epoch = last_epoch
        # 计算热身epoch数
        self.warmup_epoch = round(warmup_epoch * step_each_epoch)

    # 调用函数，返回学习率
    def __call__(self):
        # 创建StepDecay学习率衰减对象
        learning_rate = lr.StepDecay(
            learning_rate=self.learning_rate,
            step_size=self.step_size,
            gamma=self.gamma,
            last_epoch=self.last_epoch)
        # 如果有热身epoch，则创建LinearWarmup对象
        if self.warmup_epoch > 0:
            learning_rate = lr.LinearWarmup(
                learning_rate=learning_rate,
                warmup_steps=self.warmup_epoch,
                start_lr=0.0,
                end_lr=self.learning_rate,
                last_epoch=self.last_epoch)
        # 返回学习率对象
        return learning_rate
class Piecewise(object):
    """
    Piecewise learning rate decay
    Args:
        boundaries(list): A list of steps numbers. The type of element in the list is python int.
        values(list): A list of learning rate values that will be picked during different epoch boundaries.
            The type of element in the list is python float.
        last_epoch (int, optional):  The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
    """

    def __init__(self,
                 step_each_epoch,
                 decay_epochs,
                 values,
                 warmup_epoch=0,
                 last_epoch=-1,
                 **kwargs):
        super(Piecewise, self).__init__()
        # 初始化 Piecewise 类，设置学习率衰减的边界和对应的学习率值
        self.boundaries = [step_each_epoch * e for e in decay_epochs]
        self.values = values
        self.last_epoch = last_epoch
        self.warmup_epoch = round(warmup_epoch * step_each_epoch)

    def __call__(self):
        # 调用 Piecewise 类时返回学习率衰减策略
        learning_rate = lr.PiecewiseDecay(
            boundaries=self.boundaries,
            values=self.values,
            last_epoch=self.last_epoch)
        # 如果设置了 warmup_epoch，则使用线性预热策略
        if self.warmup_epoch > 0:
            learning_rate = lr.LinearWarmup(
                learning_rate=learning_rate,
                warmup_steps=self.warmup_epoch,
                start_lr=0.0,
                end_lr=self.values[0],
                last_epoch=self.last_epoch)
        return learning_rate


class CyclicalCosine(object):
    """
    Cyclical cosine learning rate decay
    Args:
        learning_rate(float): initial learning rate
        step_each_epoch(int): steps each epoch
        epochs(int): total training epochs
        cycle(int): period of the cosine learning rate
        last_epoch (int, optional):  The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
    """
    # 初始化函数，接受学习率、每个周期的步数、总周期数、循环次数、热身周期数、上一个周期数等参数
    def __init__(self,
                 learning_rate,
                 step_each_epoch,
                 epochs,
                 cycle,
                 warmup_epoch=0,
                 last_epoch=-1,
                 **kwargs):
        # 调用父类的初始化函数
        super(CyclicalCosine, self).__init__()
        # 设置学习率
        self.learning_rate = learning_rate
        # 计算总步数
        self.T_max = step_each_epoch * epochs
        # 设置上一个周期数
        self.last_epoch = last_epoch
        # 计算热身周期的步数
        self.warmup_epoch = round(warmup_epoch * step_each_epoch)
        # 计算循环周期的步数
        self.cycle = round(cycle * step_each_epoch)

    # 调用函数
    def __call__(self):
        # 创建一个CyclicalCosineDecay对象，传入学习率、总步数、循环周期、上一个周期数
        learning_rate = CyclicalCosineDecay(
            learning_rate=self.learning_rate,
            T_max=self.T_max,
            cycle=self.cycle,
            last_epoch=self.last_epoch)
        # 如果热身周期大于0
        if self.warmup_epoch > 0:
            # 创建一个LinearWarmup对象，传入学习率、热身步数、起始学习率、结束学习率、上一个周期数
            learning_rate = lr.LinearWarmup(
                learning_rate=learning_rate,
                warmup_steps=self.warmup_epoch,
                start_lr=0.0,
                end_lr=self.learning_rate,
                last_epoch=self.last_epoch)
        # 返回学习率
        return learning_rate
class OneCycle(object):
    """
    One Cycle learning rate decay
    Args:
        max_lr(float): Upper learning rate boundaries
        epochs(int): total training epochs
        step_each_epoch(int): steps each epoch
        anneal_strategy(str): {‘cos’, ‘linear’} Specifies the annealing strategy: “cos” for cosine annealing, “linear” for linear annealing. 
            Default: ‘cos’
        three_phase(bool): If True, use a third phase of the schedule to annihilate the learning rate according to ‘final_div_factor’ 
            instead of modifying the second phase (the first two phases will be symmetrical about the step indicated by ‘pct_start’).
        last_epoch (int, optional):  The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
    """

    def __init__(self,
                 max_lr,
                 epochs,
                 step_each_epoch,
                 anneal_strategy='cos',
                 three_phase=False,
                 warmup_epoch=0,
                 last_epoch=-1,
                 **kwargs):
        # 初始化 OneCycle 类
        super(OneCycle, self).__init__()
        # 设置最大学习率上限
        self.max_lr = max_lr
        # 设置总训练周期数
        self.epochs = epochs
        # 设置每个周期的步数
        self.steps_per_epoch = step_each_epoch
        # 设置退火策略，默认为余弦退火
        self.anneal_strategy = anneal_strategy
        # 设置是否使用三阶段学习率退火
        self.three_phase = three_phase
        # 设置最后一个周期的索引，默认为初始学习率
        self.last_epoch = last_epoch
        # 设置预热周期的步数
        self.warmup_epoch = round(warmup_epoch * step_each_epoch)
    # 定义一个 __call__ 方法，用于返回学习率调度器对象
    def __call__(self):
        # 创建一个 OneCycleDecay 学习率调度器对象
        learning_rate = OneCycleDecay(
            max_lr=self.max_lr,
            epochs=self.epochs,
            steps_per_epoch=self.steps_per_epoch,
            anneal_strategy=self.anneal_strategy,
            three_phase=self.three_phase,
            last_epoch=self.last_epoch)
        # 如果存在预热周期，使用 LinearWarmup 进行学习率预热
        if self.warmup_epoch > 0:
            learning_rate = lr.LinearWarmup(
                learning_rate=learning_rate,
                warmup_steps=self.warmup_epoch,
                start_lr=0.0,
                end_lr=self.max_lr,
                last_epoch=self.last_epoch)
        # 返回学习率调度器对象
        return learning_rate
# 定义一个常数学习率衰减类
class Const(object):
    """
    Const learning rate decay
    Args:
        learning_rate(float): 初始学习率
        step_each_epoch(int): 每个 epoch 的步数
        last_epoch (int, optional): 上一个 epoch 的索引。可以设置为重新开始训练。默认值为 -1，表示初始学习率。
    """

    def __init__(self,
                 learning_rate,
                 step_each_epoch,
                 warmup_epoch=0,
                 last_epoch=-1,
                 **kwargs):
        super(Const, self).__init__()
        self.learning_rate = learning_rate
        self.last_epoch = last_epoch
        self.warmup_epoch = round(warmup_epoch * step_each_epoch)

    def __call__(self):
        learning_rate = self.learning_rate
        if self.warmup_epoch > 0:
            learning_rate = lr.LinearWarmup(
                learning_rate=learning_rate,
                warmup_steps=self.warmup_epoch,
                start_lr=0.0,
                end_lr=self.learning_rate,
                last_epoch=self.last_epoch)
        return learning_rate


# 定义一个学习率衰减类
class DecayLearningRate(object):
    """
    DecayLearningRate learning rate decay
    new_lr = (lr - end_lr) * (1 - epoch/decay_steps)**power + end_lr
    Args:
        learning_rate(float): 初始学习率
        step_each_epoch(int): 每个 epoch 的步数
        epochs(int): 总训练 epochs
        factor(float): 多项式的幂，应大于 0.0 以获得学习率衰减。默认值为 0.9
        end_lr(float): 最小的最终学习率。默认值为 0.0。
    """

    def __init__(self,
                 learning_rate,
                 step_each_epoch,
                 epochs,
                 factor=0.9,
                 end_lr=0,
                 **kwargs):
        super(DecayLearningRate, self).__init__()
        self.learning_rate = learning_rate
        self.epochs = epochs + 1
        self.factor = factor
        self.end_lr = 0
        self.decay_steps = step_each_epoch * epochs
    # 定义一个特殊方法 __call__，用于对象的调用
    def __call__(self):
        # 使用多项式衰减方式生成学习率
        learning_rate = lr.PolynomialDecay(
            learning_rate=self.learning_rate,  # 初始学习率
            decay_steps=self.decay_steps,  # 衰减步数
            power=self.factor,  # 多项式幂次
            end_lr=self.end_lr)  # 最终学习率
        # 返回生成的学习率
        return learning_rate
class MultiStepDecay(object):
    """
    Piecewise learning rate decay
    Args:
        step_each_epoch(int): steps each epoch
        learning_rate (float): The initial learning rate. It is a python float number.
        step_size (int): the interval to update.
        gamma (float, optional): The Ratio that the learning rate will be reduced. ``new_lr = origin_lr * gamma`` .
            It should be less than 1.0. Default: 0.1.
        last_epoch (int, optional):  The index of last epoch. Can be set to restart training. Default: -1, means initial learning rate.
    """

    def __init__(self,
                 learning_rate,
                 milestones,
                 step_each_epoch,
                 gamma,
                 warmup_epoch=0,
                 last_epoch=-1,
                 **kwargs):
        # 初始化函数，设置学习率衰减的参数
        super(MultiStepDecay, self).__init__()
        # 计算每个里程碑对应的步数
        self.milestones = [step_each_epoch * e for e in milestones]
        # 设置初始学习率
        self.learning_rate = learning_rate
        # 设置学习率衰减比例
        self.gamma = gamma
        # 设置上一个周期的索引
        self.last_epoch = last_epoch
        # 设置热身训练周期
        self.warmup_epoch = round(warmup_epoch * step_each_epoch)

    def __call__(self):
        # 调用函数，返回学习率衰减策略
        learning_rate = lr.MultiStepDecay(
            learning_rate=self.learning_rate,
            milestones=self.milestones,
            gamma=self.gamma,
            last_epoch=self.last_epoch)
        # 如果有热身训练周期
        if self.warmup_epoch > 0:
            # 使用线性热身训练策略
            learning_rate = lr.LinearWarmup(
                learning_rate=learning_rate,
                warmup_steps=self.warmup_epoch,
                start_lr=0.0,
                end_lr=self.learning_rate,
                last_epoch=self.last_epoch)
        # 返回学习率
        return learning_rate


class TwoStepCosine(object):
    """
    Cosine learning rate decay
    lr = 0.05 * (math.cos(epoch * (math.pi / epochs)) + 1)
    """
    Args:
        lr(float): 初始学习率
        step_each_epoch(int): 每个 epoch 的步数
        epochs(int): 总训练 epochs
        last_epoch (int, optional): 上一个 epoch 的索引。可以设置为重新开始训练。默认值为 -1，表示初始学习率。
    """

    def __init__(self,
                 learning_rate,
                 step_each_epoch,
                 epochs,
                 warmup_epoch=0,
                 last_epoch=-1,
                 **kwargs):
        super(TwoStepCosine, self).__init__()
        self.learning_rate = learning_rate
        self.T_max1 = step_each_epoch * 200
        self.T_max2 = step_each_epoch * epochs
        self.last_epoch = last_epoch
        self.warmup_epoch = round(warmup_epoch * step_each_epoch)

    def __call__(self):
        learning_rate = TwoStepCosineDecay(
            learning_rate=self.learning_rate,
            T_max1=self.T_max1,
            T_max2=self.T_max2,
            last_epoch=self.last_epoch)
        if self.warmup_epoch > 0:
            learning_rate = lr.LinearWarmup(
                learning_rate=learning_rate,
                warmup_steps=self.warmup_epoch,
                start_lr=0.0,
                end_lr=self.learning_rate,
                last_epoch=self.last_epoch)
        return learning_rate
```