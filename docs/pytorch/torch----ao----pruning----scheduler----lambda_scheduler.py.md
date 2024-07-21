# `.\pytorch\torch\ao\pruning\scheduler\lambda_scheduler.py`

```
# mypy: allow-untyped-defs
# 引入警告模块，用于显示警告信息
import warnings

# 导入基础调度器类
from .base_scheduler import BaseScheduler

# 导出模块中的 LambdaSL 类
__all__ = ["LambdaSL"]

class LambdaSL(BaseScheduler):
    """Sets the sparsity level of each parameter group to the final sl
    times a given function. When last_epoch=-1, sets initial sl as zero.
    Args:
        sparsifier (BaseSparsifier): Wrapped sparsifier.
        sl_lambda (function or list): A function which computes a multiplicative
            factor given an integer parameter epoch, or a list of such
            functions, one for each group in sparsifier.param_groups.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    Example:
        >>> # Assuming sparsifier has two groups.
        >>> lambda1 = lambda epoch: epoch // 30
        >>> lambda2 = lambda epoch: 0.95 ** epoch
        >>> # xdoctest: +SKIP
        >>> scheduler = LambdaSL(sparsifier, sl_lambda=[lambda1, lambda2])
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(self, sparsifier, sl_lambda, last_epoch=-1, verbose=False):
        # 初始化方法，接受稀疏化器、sl_lambda 函数或函数列表、最后周期索引和详细输出标志
        self.sparsifier = sparsifier

        # 如果 sl_lambda 不是列表或元组，则复制为列表并重复 sparsifier.groups 数量次
        if not isinstance(sl_lambda, list) and not isinstance(sl_lambda, tuple):
            self.sl_lambdas = [sl_lambda] * len(sparsifier.groups)
        else:
            # 否则，检查 sl_lambda 的长度与 sparsifier.groups 的长度是否相同
            if len(sl_lambda) != len(sparsifier.groups):
                raise ValueError(f"Expected {len(sparsifier.groups)} lr_lambdas, but got {len(sl_lambda)}")
            self.sl_lambdas = list(sl_lambda)

        # 调用父类的初始化方法
        super().__init__(sparsifier, last_epoch, verbose)

    def get_sl(self):
        # 获取当前稀疏度级别的方法
        if not self._get_sl_called_within_step:
            # 如果不在步骤内调用该方法，则发出警告提示
            warnings.warn(
                "To get the last sparsity level computed by the scheduler, "
                "please use `get_last_sl()`.")
        # 返回每个参数组的最终稀疏度级别，通过每个 sl_lambda 函数计算得到，并与基础稀疏度级别相乘
        return [base_sl * lmbda(self.last_epoch)
                for lmbda, base_sl in zip(self.sl_lambdas, self.base_sl)]
```