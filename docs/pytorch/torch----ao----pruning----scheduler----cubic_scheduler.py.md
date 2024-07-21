# `.\pytorch\torch\ao\pruning\scheduler\cubic_scheduler.py`

```
# mypy: allow-untyped-defs
# 引入警告模块，用于管理警告信息
import warnings

# 从当前目录下的base_scheduler模块中导入BaseScheduler类
from .base_scheduler import BaseScheduler

# 将CubicSL类添加到公开接口中，允许其他模块导入
__all__ = ["CubicSL"]

# 定义一个函数，用于将x限制在lo和hi之间
def _clamp(x, lo, hi):
    return max(lo, min(hi, x))

# 定义CubicSL类，继承自BaseScheduler类
class CubicSL(BaseScheduler):
    r"""Sets the sparsity level of each parameter group to the final sl
    plus a given exponential function.

    .. math::

        s_i = s_f + (s_0 - s_f) \cdot \left( 1 - \frac{t - t_0}{n\Delta t} \right)^3

    where :math:`s_i` is the sparsity at epoch :math:`t`, :math;`s_f` is the final
    sparsity level, :math:`f(i)` is the function to be applied to the current epoch
    :math:`t`, initial epoch :math:`t_0`, and final epoch :math:`t_f`.
    :math:`\Delta t` is used to control how often the update of the sparsity level
    happens. By default,

    Args:
        sparsifier (BaseSparsifier): Wrapped sparsifier.
        init_sl (int, list): Initial level of sparsity
        init_t (int, list): Initial step, when pruning starts
        delta_t (int, list): Pruning frequency
        total_t (int, list): Total number of pruning steps
        initially_zero (bool, list): If True, sets the level of sparsity to 0
            before init_t (:math:`t_0`). Otherwise, the sparsity level before
            init_t (:math:`t_0`) is set to init_sl(:math:`s_0`)
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    """
    
    # 初始化函数，接受多个参数来配置稀疏性级别的调度
    def __init__(self,
                 sparsifier,
                 init_sl=0.0,
                 init_t=0,
                 delta_t=10,
                 total_t=100,
                 initially_zero=False,
                 last_epoch=-1,
                 verbose=False
                 ):
        # 调用父类的初始化方法，传入sparsifier、last_epoch和verbose参数
        self.sparsifier = sparsifier

        # 调用_make_sure_a_list方法，确保init_sl是一个列表形式
        self.init_sl = self._make_sure_a_list(init_sl)
        # 同样地，确保init_t是一个列表形式
        self.init_t = self._make_sure_a_list(init_t)
        # 确保delta_t是一个列表形式
        self.delta_t = self._make_sure_a_list(delta_t)
        # 确保total_t是一个列表形式
        self.total_t = self._make_sure_a_list(total_t)

        # 确保initially_zero是一个列表形式
        self.initially_zero = self._make_sure_a_list(initially_zero)

        # 调用父类BaseScheduler的初始化方法，传入sparsifier和verbose参数
        super().__init__(sparsifier, last_epoch, verbose)

    # 静态方法，用于将x限制在lo和hi之间
    @staticmethod
    def sparsity_compute_fn(s_0, s_f, t, t_0, dt, n, initially_zero=False):
        r""""Computes the current level of sparsity.

        Based on https://arxiv.org/pdf/1710.01878.pdf

        Args:
            s_0: Initial level of sparsity, :math:`s_i`
            s_f: Target level of sparsity, :math:`s_f`
            t: Current step, :math:`t`
            t_0: Initial step, :math:`t_0`
            dt: Pruning frequency, :math:`\Delta T`
            n: Pruning steps, :math:`n`
            initially_zero: Sets the level of sparsity to 0 before t_0.
                If False, sets to s_0

        Returns:
            The sparsity level :math:`s_t` at the current step :math:`t`
        """
        # 如果 initially_zero 为 True 并且当前步 t 小于初始步 t_0，则返回 0
        if initially_zero and t < t_0:
            return 0
        # 根据论文中的公式计算当前步 t 的稀疏度 s_t
        s_t = s_f + (s_0 - s_f) * (1.0 - (t - t_0) / (dt * n)) ** 3
        # 使用 _clamp 函数确保稀疏度 s_t 在 s_0 和 s_f 之间
        s_t = _clamp(s_t, s_0, s_f)
        return s_t

    def get_sl(self):
        # 如果 get_sl 不在步骤中被调用，则发出警告信息
        if not self._get_sl_called_within_step:
            warnings.warn(
                "To get the last sparsity level computed by the scheduler, "
                "please use `get_last_sl()`.")
        # 返回一个列表，其中每个元素是通过 sparsity_compute_fn 计算得到的稀疏度水平
        return [
            self.sparsity_compute_fn(
                s_0=initial_sparsity,
                s_f=final_sparsity,
                t=self.last_epoch,
                t_0=initial_epoch,
                dt=delta_epoch,
                n=interval_epochs,
                initially_zero=initially_zero
            ) for initial_sparsity, final_sparsity, initial_epoch, delta_epoch, interval_epochs, initially_zero in
            zip(
                self.init_sl,    # 初始稀疏度列表
                self.base_sl,    # 目标稀疏度列表
                self.init_t,     # 初始步数列表
                self.delta_t,    # 裁剪频率列表
                self.total_t,    # 裁剪步数列表
                self.initially_zero  # 是否从零开始的标志列表
            )
        ]
```