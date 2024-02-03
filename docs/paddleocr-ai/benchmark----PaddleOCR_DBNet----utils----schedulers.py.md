# `.\PaddleOCR\benchmark\PaddleOCR_DBNet\utils\schedulers.py`

```
# 从 paddle.optimizer 模块中导入 lr
from paddle.optimizer import lr
# 导入 logging 模块
import logging
# 定义 Polynomial 类
__all__ = ['Polynomial']

class Polynomial(object):
    """
    Polynomial learning rate decay
    Args:
        learning_rate (float): 初始学习率。它是一个 Python 浮点数。
        epochs(int): 衰减周期大小。它确定了衰减周期，当 by_epoch 设置为 true 时，它将变为 epochs=epochs*step_each_epoch。
        step_each_epoch: 每个周期中的所有步骤。
        end_lr(float, optional): 最小最终学习率。默认值为 0.0001。
        power(float, optional): 多项式的幂。默认值为 1.0。
        warmup_epoch(int): LinearWarmup 的周期数。默认值为 0，当 by_epoch 设置为 true 时，它将变为 warmup_epoch=warmup_epoch*step_each_epoch。
        warmup_start_lr(float): 热身阶段的初始学习率。默认值为 0.0。
        last_epoch (int, optional): 最后一个周期的索引。可以设置为重新开始训练。默认值为 -1，表示初始学习率。
        by_epoch: 参数设置是基于周期还是迭代，当设置为 true 时，epochs 和 warmup_epoch 将自动乘以 step_each_epoch。默认值为 True
    """
    # 初始化函数，设置学习率、训练周期数、每个周期的步数、结束学习率、幂次方、热身周期数、热身起始学习率、最后周期数、按周期计算标志位
    def __init__(self,
                 learning_rate,
                 epochs,
                 step_each_epoch,
                 end_lr=0.0,
                 power=1.0,
                 warmup_epoch=0,
                 warmup_start_lr=0.0,
                 last_epoch=-1,
                 by_epoch=True,
                 **kwargs):
        # 调用父类的初始化函数
        super().__init__()
        # 如果热身周期数大于等于总周期数，则发出警告并将热身周期数设置为总周期数
        if warmup_epoch >= epochs:
            msg = f"When using warm up, the value of \"epochs\" must be greater than value of \"Optimizer.lr.warmup_epoch\". The value of \"Optimizer.lr.warmup_epoch\" has been set to {epochs}."
            logging.warning(msg)
            warmup_epoch = epochs
        # 设置各个参数
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.end_lr = end_lr
        self.power = power
        self.last_epoch = last_epoch
        self.warmup_epoch = warmup_epoch
        self.warmup_start_lr = warmup_start_lr

        # 如果按周期计算学习率，则将总周期数和热身周期数乘以每个周期的步数
        if by_epoch:
            self.epochs *= step_each_epoch
            self.warmup_epoch = int(self.warmup_epoch * step_each_epoch)

    # 调用函数，计算学习率
    def __call__(self):
        # 根据多项式衰减计算学习率
        learning_rate = lr.PolynomialDecay(
            learning_rate=self.learning_rate,
            decay_steps=self.epochs,
            end_lr=self.end_lr,
            power=self.power,
            last_epoch=self.
            last_epoch) if self.epochs > 0 else self.learning_rate
        # 如果有热身周期，则根据线性热身计算学习率
        if self.warmup_epoch > 0:
            learning_rate = lr.LinearWarmup(
                learning_rate=learning_rate,
                warmup_steps=self.warmup_epoch,
                start_lr=self.warmup_start_lr,
                end_lr=self.learning_rate,
                last_epoch=self.last_epoch)
        # 返回计算得到的学习率
        return learning_rate
```