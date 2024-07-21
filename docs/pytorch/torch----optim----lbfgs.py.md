# `.\pytorch\torch\optim\lbfgs.py`

```py
# mypy: allow-untyped-defs
# 引入类型声明模块，允许未经类型定义的函数
from typing import Optional

# 引入PyTorch模块
import torch
# 从本地目录中的optimizer模块导入Optimizer和ParamsT
from .optimizer import Optimizer, ParamsT

# 定义本模块中公开的类名称
__all__ = ["LBFGS"]


def _cubic_interpolate(x1, f1, g1, x2, f2, g2, bounds=None):
    # 源自https://github.com/torch/optim/blob/master/polyinterp.lua的代码
    # 计算插值区域的边界
    if bounds is not None:
        xmin_bound, xmax_bound = bounds
    else:
        xmin_bound, xmax_bound = (x1, x2) if x1 <= x2 else (x2, x1)

    # 最常见情况的代码：对两点进行立方插值
    # 使用两点的函数值和导数值进行插值
    # 在x2为较远点的情况下的解决方案：
    #   d1 = g1 + g2 - 3*(f1-f2)/(x1-x2);
    #   d2 = sqrt(d1^2 - g1*g2);
    #   min_pos = x2 - (x2 - x1)*((g2 + d2 - d1)/(g2 - g1 + 2*d2));
    #   t_new = min(max(min_pos,xmin_bound),xmax_bound);
    d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2)
    d2_square = d1**2 - g1 * g2
    if d2_square >= 0:
        d2 = d2_square.sqrt()
        if x1 <= x2:
            min_pos = x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2))
        else:
            min_pos = x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2))
        return min(max(min_pos, xmin_bound), xmax_bound)
    else:
        return (xmin_bound + xmax_bound) / 2.0


def _strong_wolfe(
    obj_func, x, t, d, f, g, gtd, c1=1e-4, c2=0.9, tolerance_change=1e-9, max_ls=25
):
    # 源自https://github.com/torch/optim/blob/master/lswolfe.lua的代码
    # 计算方向向量d的最大绝对值
    d_norm = d.abs().max()
    # 克隆梯度g，使用内存格式为torch.contiguous_format
    g = g.clone(memory_format=torch.contiguous_format)
    # 使用初始步长评估目标函数和梯度
    f_new, g_new = obj_func(x, t, d)
    ls_func_evals = 1
    gtd_new = g_new.dot(d)

    # 寻找满足Wolfe准则的区间
    t_prev, f_prev, g_prev, gtd_prev = 0, f, g, gtd
    done = False
    ls_iter = 0
    # 当前线搜索迭代次数小于最大允许次数时，执行循环
    while ls_iter < max_ls:
        # 检查条件
        if f_new > (f + c1 * t * gtd) or (ls_iter > 1 and f_new >= f_prev):
            # 如果 Armijo条件不满足或者非第一次迭代且函数值不下降，则更新 bracket 和相关变量后中断循环
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        if abs(gtd_new) <= -c2 * gtd:
            # 如果 Wolfe条件满足，则更新 bracket 和相关变量后中断循环
            bracket = [t]
            bracket_f = [f_new]
            bracket_g = [g_new]
            done = True
            break

        if gtd_new >= 0:
            # 如果搜索方向导数非负，则更新 bracket 和相关变量后中断循环
            bracket = [t_prev, t]
            bracket_f = [f_prev, f_new]
            bracket_g = [g_prev, g_new.clone(memory_format=torch.contiguous_format)]
            bracket_gtd = [gtd_prev, gtd_new]
            break

        # 插值步骤
        min_step = t + 0.01 * (t - t_prev)
        max_step = t * 10
        tmp = t
        t = _cubic_interpolate(
            t_prev, f_prev, gtd_prev, t, f_new, gtd_new, bounds=(min_step, max_step)
        )

        # 更新变量为下一步准备
        t_prev = tmp
        f_prev = f_new
        g_prev = g_new.clone(memory_format=torch.contiguous_format)
        gtd_prev = gtd_new
        f_new, g_new = obj_func(x, t, d)  # 调用目标函数求得新的函数值和梯度
        ls_func_evals += 1  # 记录函数评估次数
        gtd_new = g_new.dot(d)  # 计算新的搜索方向导数
        ls_iter += 1  # 更新迭代计数器

    # 如果达到最大迭代次数仍未找到合适点
    if ls_iter == max_ls:
        bracket = [0, t]
        bracket_f = [f, f_new]
        bracket_g = [g, g_new]

    # 缩放阶段：此时已找到满足条件的点或其周围的 bracket。我们继续细化 bracket 直至找到精确满足条件的点
    insuf_progress = False

    # 在 bracket 中找到高点和低点的位置
    low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[-1] else (1, 0)  # type: ignore[possibly-undefined]
    # 当尚未完成且未达到最大迭代次数时执行循环
    while not done and ls_iter < max_ls:
        # 如果线搜索区间非常小
        if abs(bracket[1] - bracket[0]) * d_norm < tolerance_change:  # type: ignore[possibly-undefined]
            break

        # 计算新的试探值
        t = _cubic_interpolate(
            bracket[0],
            bracket_f[0],
            bracket_gtd[0],  # type: ignore[possibly-undefined]
            bracket[1],
            bracket_f[1],
            bracket_gtd[1],
        )

        # 检查是否取得足够的进展：
        # 如果`t`接近边界，则标记为进展不足，
        # 如果：
        #   + 在上一步中进展不足，或者
        #   + `t`处于边界之一，
        # 则将`t`移动到距离最近边界点`0.1 * len(bracket)`的位置。
        eps = 0.1 * (max(bracket) - min(bracket))
        if min(max(bracket) - t, t - min(bracket)) < eps:
            # 插值接近边界
            if insuf_progress or t >= max(bracket) or t <= min(bracket):
                # 在距离边界0.1的位置处评估
                if abs(t - max(bracket)) < abs(t - min(bracket)):
                    t = max(bracket) - eps
                else:
                    t = min(bracket) + eps
                insuf_progress = False
            else:
                insuf_progress = True
        else:
            insuf_progress = False

        # 评估新点
        f_new, g_new = obj_func(x, t, d)
        ls_func_evals += 1
        gtd_new = g_new.dot(d)
        ls_iter += 1

        # 如果不满足Armijo条件或者不比最低点更低
        if f_new > (f + c1 * t * gtd) or f_new >= bracket_f[low_pos]:
            bracket[high_pos] = t
            bracket_f[high_pos] = f_new
            bracket_g[high_pos] = g_new.clone(memory_format=torch.contiguous_format)  # type: ignore[possibly-undefined]
            bracket_gtd[high_pos] = gtd_new
            low_pos, high_pos = (0, 1) if bracket_f[0] <= bracket_f[1] else (1, 0)
        else:
            # 如果满足Wolfe条件
            if abs(gtd_new) <= -c2 * gtd:
                done = True
            # 如果gtd_new与(bracket[high_pos] - bracket[low_pos])同号
            elif gtd_new * (bracket[high_pos] - bracket[low_pos]) >= 0:
                bracket[high_pos] = bracket[low_pos]
                bracket_f[high_pos] = bracket_f[low_pos]
                bracket_g[high_pos] = bracket_g[low_pos]  # type: ignore[possibly-undefined]
                bracket_gtd[high_pos] = bracket_gtd[low_pos]

            bracket[low_pos] = t
            bracket_f[low_pos] = f_new
            bracket_g[low_pos] = g_new.clone(memory_format=torch.contiguous_format)  # type: ignore[possibly-undefined]
            bracket_gtd[low_pos] = gtd_new

    # 返回结果
    t = bracket[low_pos]  # type: ignore[possibly-undefined]
    f_new = bracket_f[low_pos]
    # 从列表 bracket_g 中根据索引 low_pos 获取元素赋值给变量 g_new
    # type: ignore[possibly-undefined] 表示忽略可能未定义的类型检查警告
    g_new = bracket_g[low_pos]  # type: ignore[possibly-undefined]
    # 返回变量 f_new, g_new, t, ls_func_evals，作为函数的结果
    return f_new, g_new, t, ls_func_evals
# 定义了一个 LBFGS 类，实现 L-BFGS 算法，继承自 Optimizer 类
class LBFGS(Optimizer):
    """Implements L-BFGS algorithm.

    Heavily inspired by `minFunc
    <https://www.cs.ubc.ca/~schmidtm/Software/minFunc.html>`_.

    .. warning::
        This optimizer doesn't support per-parameter options and parameter
        groups (there can be only one).

    .. warning::
        Right now all parameters have to be on a single device. This will be
        improved in the future.

    .. note::
        This is a very memory intensive optimizer (it requires additional
        ``param_bytes * (history_size + 1)`` bytes). If it doesn't fit in memory
        try reducing the history size, or use a different algorithm.

    Args:
        params (iterable): iterable of parameters to optimize. Parameters must be real.
        lr (float): learning rate (default: 1)
        max_iter (int): maximal number of iterations per optimization step
            (default: 20)
        max_eval (int): maximal number of function evaluations per optimization
            step (default: max_iter * 1.25).
        tolerance_grad (float): termination tolerance on first order optimality
            (default: 1e-7).
        tolerance_change (float): termination tolerance on function
            value/parameter changes (default: 1e-9).
        history_size (int): update history size (default: 100).
        line_search_fn (str): either 'strong_wolfe' or None (default: None).
    """

    # LBFGS 类的构造函数，初始化优化器的参数和默认值
    def __init__(
        self,
        params: ParamsT,
        lr: float = 1,
        max_iter: int = 20,
        max_eval: Optional[int] = None,
        tolerance_grad: float = 1e-7,
        tolerance_change: float = 1e-9,
        history_size: int = 100,
        line_search_fn: Optional[str] = None,
    ):
        # 如果未指定 max_eval，则根据 max_iter 计算默认值
        if max_eval is None:
            max_eval = max_iter * 5 // 4
        # 设置默认参数字典
        defaults = dict(
            lr=lr,
            max_iter=max_iter,
            max_eval=max_eval,
            tolerance_grad=tolerance_grad,
            tolerance_change=tolerance_change,
            history_size=history_size,
            line_search_fn=line_search_fn,
        )
        # 调用父类 Optimizer 的构造函数，传递参数和默认值
        super().__init__(params, defaults)

        # 检查参数组的数量，LBFGS 只支持单一参数组
        if len(self.param_groups) != 1:
            raise ValueError(
                "LBFGS doesn't support per-parameter options " "(parameter groups)"
            )

        # 获取优化的参数列表
        self._params = self.param_groups[0]["params"]
        # 用于缓存计算的参数数量
        self._numel_cache = None

    # 计算当前优化参数的总元素数量
    def _numel(self):
        if self._numel_cache is None:
            # 如果缓存为空，则计算所有参数的元素数量
            self._numel_cache = sum(
                2 * p.numel() if torch.is_complex(p) else p.numel()
                for p in self._params
            )

        return self._numel_cache
    # 收集所有参数的梯度并展平成一个一维张量
    def _gather_flat_grad(self):
        views = []
        for p in self._params:
            if p.grad is None:
                # 如果参数的梯度为None，则创建一个与参数大小相同的零张量
                view = p.new(p.numel()).zero_()
            elif p.grad.is_sparse:
                # 如果参数的梯度是稀疏张量，则将其转换为稠密张量并展平成一维
                view = p.grad.to_dense().view(-1)
            else:
                # 否则，直接展平参数的梯度成一维张量
                view = p.grad.view(-1)
            if torch.is_complex(view):
                # 如果视图是复数张量，则将其视为实数张量并展平成一维
                view = torch.view_as_real(view).view(-1)
            views.append(view)
        return torch.cat(views, 0)

    # 给参数添加梯度乘以步长的更新
    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            if torch.is_complex(p):
                # 如果参数是复数张量，则将其视为实数张量
                p = torch.view_as_real(p)
            numel = p.numel()
            # 使用视图避免废弃的逐点语义，将更新添加到参数上
            p.add_(update[offset : offset + numel].view_as(p), alpha=step_size)
            offset += numel
        assert offset == self._numel()

    # 克隆参数列表并保证克隆张量的内存格式是连续的
    def _clone_param(self):
        return [p.clone(memory_format=torch.contiguous_format) for p in self._params]

    # 设置参数为给定的参数数据
    def _set_param(self, params_data):
        for p, pdata in zip(self._params, params_data):
            # 将参数的数据复制为给定数据
            p.copy_(pdata)

    # 在给定方向上评估闭包函数的损失和梯度
    def _directional_evaluate(self, closure, x, t, d):
        # 添加在方向d上乘以步长t的梯度更新到参数上
        self._add_grad(t, d)
        # 计算闭包函数的损失值并转换为浮点数
        loss = float(closure())
        # 收集当前参数的展平梯度
        flat_grad = self._gather_flat_grad()
        # 设置参数为给定的参数向量x
        self._set_param(x)
        return loss, flat_grad

    # 禁用梯度计算上下文管理器，确保不会在此上下文中计算梯度
    @torch.no_grad()
```