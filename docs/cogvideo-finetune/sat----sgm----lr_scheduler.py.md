# `.\cogvideo-finetune\sat\sgm\lr_scheduler.py`

```py
# 导入 NumPy 库，通常用于数值计算
import numpy as np


class LambdaWarmUpCosineScheduler:
    """
    note: use with a base_lr of 1.0
    """  # 类的文档字符串，说明使用基础学习率为 1.0

    def __init__(  # 构造函数，初始化调度器
        self,
        warm_up_steps,  # 预热步骤数
        lr_min,  # 最小学习率
        lr_max,  # 最大学习率
        lr_start,  # 起始学习率
        max_decay_steps,  # 最大衰减步骤数
        verbosity_interval=0,  # 输出间隔
    ):
        # 将输入参数赋值给实例变量
        self.lr_warm_up_steps = warm_up_steps
        self.lr_start = lr_start
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.lr_max_decay_steps = max_decay_steps
        self.last_lr = 0.0  # 初始化最近的学习率为 0
        self.verbosity_interval = verbosity_interval  # 设定输出间隔

    def schedule(self, n, **kwargs):  # 学习率调度函数
        if self.verbosity_interval > 0:  # 检查是否需要输出信息
            if n % self.verbosity_interval == 0:  # 如果当前步骤满足输出条件
                print(f"current step: {n}, recent lr-multiplier: {self.last_lr}")  # 输出当前步骤和学习率
        if n < self.lr_warm_up_steps:  # 如果当前步骤在预热阶段
            # 计算线性增加的学习率
            lr = (self.lr_max - self.lr_start) / self.lr_warm_up_steps * n + self.lr_start
            self.last_lr = lr  # 更新最近的学习率
            return lr  # 返回计算的学习率
        else:  # 如果已过预热阶段
            # 计算归一化的时间参数 t
            t = (n - self.lr_warm_up_steps) / (self.lr_max_decay_steps - self.lr_warm_up_steps)
            t = min(t, 1.0)  # 确保 t 不超过 1.0
            # 计算余弦衰减的学习率
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + np.cos(t * np.pi))
            self.last_lr = lr  # 更新最近的学习率
            return lr  # 返回计算的学习率

    def __call__(self, n, **kwargs):  # 使得类实例可调用
        return self.schedule(n, **kwargs)  # 调用调度函数


class LambdaWarmUpCosineScheduler2:
    """
    supports repeated iterations, configurable via lists
    note: use with a base_lr of 1.0.
    """  # 类的文档字符串，说明支持重复迭代且通过列表配置，基础学习率为 1.0

    def __init__(self, warm_up_steps, f_min, f_max, f_start, cycle_lengths, verbosity_interval=0):  # 构造函数
        # 检查所有输入列表长度是否一致
        assert len(warm_up_steps) == len(f_min) == len(f_max) == len(f_start) == len(cycle_lengths)
        # 将输入参数赋值给实例变量
        self.lr_warm_up_steps = warm_up_steps
        self.f_start = f_start
        self.f_min = f_min
        self.f_max = f_max
        self.cycle_lengths = cycle_lengths
        # 计算循环长度的累积和
        self.cum_cycles = np.cumsum([0] + list(self.cycle_lengths))
        self.last_f = 0.0  # 初始化最近的函数值为 0
        self.verbosity_interval = verbosity_interval  # 设定输出间隔

    def find_in_interval(self, n):  # 查找当前步骤所在的周期
        interval = 0  # 初始化周期计数
        for cl in self.cum_cycles[1:]:  # 遍历所有累积周期
            if n <= cl:  # 如果当前步骤在当前周期内
                return interval  # 返回周期索引
            interval += 1  # 递增周期计数

    def schedule(self, n, **kwargs):  # 学习率调度函数
        cycle = self.find_in_interval(n)  # 查找当前步骤所在的周期
        n = n - self.cum_cycles[cycle]  # 计算当前步骤在周期内的相对步骤
        if self.verbosity_interval > 0:  # 检查是否需要输出信息
            if n % self.verbosity_interval == 0:  # 如果当前步骤满足输出条件
                print(f"current step: {n}, recent lr-multiplier: {self.last_f}, " f"current cycle {cycle}")  # 输出信息
        if n < self.lr_warm_up_steps[cycle]:  # 如果当前步骤在预热阶段
            # 计算线性增加的函数值
            f = (self.f_max[cycle] - self.f_start[cycle]) / self.lr_warm_up_steps[cycle] * n + self.f_start[cycle]
            self.last_f = f  # 更新最近的函数值
            return f  # 返回计算的函数值
        else:  # 如果已过预热阶段
            # 计算归一化的时间参数 t
            t = (n - self.lr_warm_up_steps[cycle]) / (self.cycle_lengths[cycle] - self.lr_warm_up_steps[cycle])
            t = min(t, 1.0)  # 确保 t 不超过 1.0
            # 计算余弦衰减的函数值
            f = self.f_min[cycle] + 0.5 * (self.f_max[cycle] - self.f_min[cycle]) * (1 + np.cos(t * np.pi))
            self.last_f = f  # 更新最近的函数值
            return f  # 返回计算的函数值
    # 定义可调用对象的魔术方法，接收一个参数 n 和可变关键字参数 kwargs
        def __call__(self, n, **kwargs):
            # 调用该对象的 schedule 方法，传入参数 n 和可变关键字参数 kwargs，并返回其结果
            return self.schedule(n, **kwargs)
# 定义一个新的类 LambdaLinearScheduler，继承自 LambdaWarmUpCosineScheduler2
class LambdaLinearScheduler(LambdaWarmUpCosineScheduler2):
    # 定义调度方法，接受当前步数 n 和额外参数 kwargs
    def schedule(self, n, **kwargs):
        # 找到 n 在累积周期中的区间
        cycle = self.find_in_interval(n)
        # 计算当前步数 n，减去已完成的累计周期
        n = n - self.cum_cycles[cycle]
        # 如果设置了详细输出间隔，则打印当前状态
        if self.verbosity_interval > 0:
            if n % self.verbosity_interval == 0:
                print(f"current step: {n}, recent lr-multiplier: {self.last_f}, " f"current cycle {cycle}")

        # 如果当前步数小于热身步数，计算线性增加的函数值
        if n < self.lr_warm_up_steps[cycle]:
            f = (self.f_max[cycle] - self.f_start[cycle]) / self.lr_warm_up_steps[cycle] * n + self.f_start[cycle]
            # 更新最近的函数值
            self.last_f = f
            # 返回计算得到的函数值
            return f
        else:
            # 在热身步数之后，计算函数值为周期长度的线性递减
            f = (
                self.f_min[cycle]
                + (self.f_max[cycle] - self.f_min[cycle])
                * (self.cycle_lengths[cycle] - n)
                / (self.cycle_lengths[cycle])
            )
            # 更新最近的函数值
            self.last_f = f
            # 返回计算得到的函数值
            return f
```