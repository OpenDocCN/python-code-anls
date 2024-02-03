# `.\PaddleOCR\ppocr\optimizer\lr_scheduler.py`

```
# 版权声明
#
# 基于 Apache 许可证 2.0 版本授权
# 除非符合许可证的规定，否则不得使用此文件
# 您可以在以下网址获取许可证的副本
# http://www.apache.org/licenses/LICENSE-2.0
#
# 除非适用法律要求或书面同意，否则根据许可证分发的软件是基于“原样”分发的，
# 没有任何明示或暗示的保证或条件
# 请查看许可证以获取特定语言的权限和限制

import math
from paddle.optimizer.lr import LRScheduler

# 定义 CyclicalCosineDecay 类，继承自 LRScheduler 类
class CyclicalCosineDecay(LRScheduler):
    def __init__(self,
                 learning_rate,
                 T_max,
                 cycle=1,
                 last_epoch=-1,
                 eta_min=0.0,
                 verbose=False):
        """
        Cyclical cosine learning rate decay
        A learning rate which can be referred in https://arxiv.org/pdf/2012.12645.pdf
        Args:
            learning rate(float): 学习率
            T_max(int): 最大的 epoch 数
            cycle(int): 余弦衰减的周期
            last_epoch (int, optional): 上一个 epoch 的索引。可以设置为重新开始训练。默认值为 -1，表示初始学习率。
            eta_min(float): 训练过程中的最小学习率
            verbose(bool): 是否打印每个 epoch 的学习率
        """
        # 调用父类的构造函数
        super(CyclicalCosineDecay, self).__init__(learning_rate, last_epoch,
                                                  verbose)
        # 设置余弦衰减的周期
        self.cycle = cycle
        # 设置训练过程中的最小学习率
        self.eta_min = eta_min

    # 获取当前 epoch 对应的学习率
    def get_lr(self):
        # 如果是第一个 epoch，则返回基础学习率
        if self.last_epoch == 0:
            return self.base_lr
        # 计算当前 epoch 在周期内的相对位置
        reletive_epoch = self.last_epoch % self.cycle
        # 根据余弦函数计算学习率
        lr = self.eta_min + 0.5 * (self.base_lr - self.eta_min) * \
                (1 + math.cos(math.pi * reletive_epoch / self.cycle))
        return lr
# 定义一个名为 OneCycleDecay 的类，继承自 LRScheduler
class OneCycleDecay(LRScheduler):
    """
    One Cycle learning rate decay
    A learning rate which can be referred in https://arxiv.org/abs/1708.07120
    Code refered in https://pytorch.org/docs/stable/_modules/torch/optim/lr_scheduler.html#OneCycleLR
    """
    
    # 定义一个私有方法 _annealing_cos，用于计算余弦退火的学习率
    def _annealing_cos(self, start, end, pct):
        "Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."
        # 计算余弦值并加上1
        cos_out = math.cos(math.pi * pct) + 1
        # 返回根据余弦值计算的学习率
        return end + (start - end) / 2.0 * cos_out
    
    # 定义一个私有方法 _annealing_linear，用于计算线性退火的学习率
    def _annealing_linear(self, start, end, pct):
        "Linearly anneal from `start` to `end` as pct goes from 0.0 to 1.0."
        # 根据线性关系计算学习率
        return (end - start) * pct + start
    
    # 定义一个获取学习率的方法
    def get_lr(self):
        # 初始化计算得到的学习率为0
        computed_lr = 0.0
        # 获取当前步数
        step_num = self.last_epoch
        
        # 如果当前步数大于总步数，则抛出异常
        if step_num > self.total_steps:
            raise ValueError(
                "Tried to step {} times. The specified number of total steps is {}"
                .format(step_num + 1, self.total_steps))
        
        # 初始化起始步数为0
        start_step = 0
        # 遍历学习率调度阶段
        for i, phase in enumerate(self._schedule_phases):
            # 获取当前阶段的结束步数
            end_step = phase['end_step']
            # 如果当前步数小于等于结束步数或者已经是最后一个阶段，则计算学习率
            if step_num <= end_step or i == len(self._schedule_phases) - 1:
                # 计算当前阶段的百分比
                pct = (step_num - start_step) / (end_step - start_step)
                # 根据退火函数计算学习率
                computed_lr = self.anneal_func(phase['start_lr'],
                                               phase['end_lr'], pct)
                break
            # 更新起始步数为当前阶段的结束步数
            start_step = phase['end_step']
        
        # 返回计算得到的学习率
        return computed_lr


# 定义一个名为 TwoStepCosineDecay 的类，继承自 LRScheduler
class TwoStepCosineDecay(LRScheduler):
    # 初始化函数，接受学习率、T_max1、T_max2、eta_min等参数
    def __init__(self,
                 learning_rate,
                 T_max1,
                 T_max2,
                 eta_min=0,
                 last_epoch=-1,
                 verbose=False):
        # 检查T_max1是否为整数类型，如果不是则抛出类型错误
        if not isinstance(T_max1, int):
            raise TypeError(
                "The type of 'T_max1' in 'CosineAnnealingDecay' must be 'int', but received %s."
                % type(T_max1))
        # 检查T_max2是否为整数类型，如果不是则抛出类型错误
        if not isinstance(T_max2, int):
            raise TypeError(
                "The type of 'T_max2' in 'CosineAnnealingDecay' must be 'int', but received %s."
                % type(T_max2))
        # 检查eta_min是否为浮点数或整数类型，如果不是则抛出类型错误
        if not isinstance(eta_min, (float, int)):
            raise TypeError(
                "The type of 'eta_min' in 'CosineAnnealingDecay' must be 'float, int', but received %s."
                % type(eta_min))
        # 断言T_max1为正整数
        assert T_max1 > 0 and isinstance(
            T_max1, int), " 'T_max1' must be a positive integer."
        # 断言T_max2为正整数
        assert T_max2 > 0 and isinstance(
            T_max2, int), " 'T_max1' must be a positive integer."
        # 初始化T_max1和T_max2
        self.T_max1 = T_max1
        self.T_max2 = T_max2
        # 初始化eta_min为浮点数
        self.eta_min = float(eta_min)
        # 调用父类的初始化函数
        super(TwoStepCosineDecay, self).__init__(learning_rate, last_epoch,
                                                 verbose)
    # 获取当前学习率
    def get_lr(self):
        # 如果当前 epoch 小于等于 T_max1
        if self.last_epoch <= self.T_max1:
            # 如果当前 epoch 为 0，返回基础学习率
            if self.last_epoch == 0:
                return self.base_lr
            # 如果当前 epoch 满足条件，根据余弦退火计算学习率
            elif (self.last_epoch - 1 - self.T_max1) % (2 * self.T_max1) == 0:
                return self.last_lr + (self.base_lr - self.eta_min) * (
                    1 - math.cos(math.pi / self.T_max1)) / 2
            # 其他情况，根据余弦退火计算学习率
            return (1 + math.cos(math.pi * self.last_epoch / self.T_max1)) / (
                1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max1)) * (
                    self.last_lr - self.eta_min) + self.eta_min
        # 如果当前 epoch 大于 T_max1
        else:
            # 如果当前 epoch 满足条件，根据余弦退火计算学习率
            if (self.last_epoch - 1 - self.T_max2) % (2 * self.T_max2) == 0:
                return self.last_lr + (self.base_lr - self.eta_min) * (
                    1 - math.cos(math.pi / self.T_max2)) / 2
            # 其他情况，根据余弦退火计算学习率
            return (1 + math.cos(math.pi * self.last_epoch / self.T_max2)) / (
                1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max2)) * (
                    self.last_lr - self.eta_min) + self.eta_min

    # 获取闭式计算的学习率
    def _get_closed_form_lr(self):
        # 如果当前 epoch 小于等于 T_max1，根据闭式计算公式计算学习率
        if self.last_epoch <= self.T_max1:
            return self.eta_min + (self.base_lr - self.eta_min) * (1 + math.cos(
                math.pi * self.last_epoch / self.T_max1)) / 2
        # 如果当前 epoch 大于 T_max1，根据闭式计算公式计算学习率
        else:
            return self.eta_min + (self.base_lr - self.eta_min) * (1 + math.cos(
                math.pi * self.last_epoch / self.T_max2)) / 2
```