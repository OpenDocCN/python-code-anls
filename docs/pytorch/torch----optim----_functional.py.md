# `.\pytorch\torch\optim\_functional.py`

```py
# mypy: allow-untyped-defs
# 导入所需的库和模块
r"""Functional interface."""  # 模块文档字符串，描述了本模块的功能
import math  # 导入 math 库，用于数学运算
from typing import List  # 导入 List 类型提示

from torch import Tensor  # 从 torch 库中导入 Tensor 类型

# 下面导入了各种优化算法，例如 adadelta, adagrad 等，类型提示被忽略
from .adadelta import adadelta  # type: ignore[attr-defined]  # noqa: F401
from .adagrad import _make_sparse, adagrad  # type: ignore[attr-defined]  # noqa: F401
from .adam import adam  # type: ignore[attr-defined]  # noqa: F401
from .adamax import adamax  # type: ignore[attr-defined]  # noqa: F401
from .adamw import adamw  # type: ignore[attr-defined]  # noqa: F401
from .asgd import asgd  # type: ignore[attr-defined]  # noqa: F401
from .nadam import nadam  # type: ignore[attr-defined]  # noqa: F401
from .radam import radam  # type: ignore[attr-defined]  # noqa: F401
from .rmsprop import rmsprop  # type: ignore[attr-defined]  # noqa: F401
from .rprop import rprop  # type: ignore[attr-defined]  # noqa: F401
from .sgd import sgd  # type: ignore[attr-defined]  # noqa: F401

# TODO: use foreach API in optim._functional to do all the computation
# 提示性注释，建议在 optim._functional 中使用 foreach API 来执行所有计算

# 定义 Sparse Adam 算法的函数接口
def sparse_adam(
    params: List[Tensor],  # 参数列表，每个参数是一个 Tensor
    grads: List[Tensor],  # 梯度列表，对应每个参数的梯度 Tensor
    exp_avgs: List[Tensor],  # 指数加权平均值列表
    exp_avg_sqs: List[Tensor],  # 指数加权平方平均值列表
    state_steps: List[int],  # 参数步数列表
    *,
    eps: float,  # 用于数值稳定性的小常数
    beta1: float,  # 第一时刻估计的指数衰减率
    beta2: float,  # 第二时刻估计的指数衰减率
    lr: float,  # 学习率
    maximize: bool,  # 是否最大化目标
):
    r"""Functional API that performs Sparse Adam algorithm computation.

    See :class:`~torch.optim.SparseAdam` for details.
    """
    for i, param in enumerate(params):
        # 获取当前参数的梯度
        grad = grads[i]
        # 如果是最大化优化，则取相反数
        grad = grad if not maximize else -grad
        # 对梯度进行合并操作，确保索引唯一性
        grad = grad.coalesce()  # 更新是非线性的，所以索引必须唯一

        # 获取梯度的索引和数值
        grad_indices = grad._indices()
        grad_values = grad._values()

        # 如果梯度值为空，则跳过更新
        if grad_values.numel() == 0:
            # 跳过空梯度的更新
            continue

        # 获取参数的大小
        size = grad.size()

        # 获取指数加权平均数和平方加权平均数
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]

        # 定义生成稀疏张量的函数
        def make_sparse(values):
            constructor = grad.new
            # 如果梯度索引或者数值的维度为0，则返回空的构造器
            if grad_indices.dim() == 0 or values.dim() == 0:
                return constructor().resize_as_(grad)
            # 否则返回由 grad_indices 和 values 构造的稀疏张量
            return constructor(grad_indices, values, size)

        # 更新指数加权平均数和平方加权平均数
        # 更新公式为 old <- b * old + (1 - b) * new
        old_exp_avg_values = exp_avg.sparse_mask(grad)._values()
        exp_avg_update_values = grad_values.sub(old_exp_avg_values).mul_(1 - beta1)
        exp_avg.add_(make_sparse(exp_avg_update_values))

        old_exp_avg_sq_values = exp_avg_sq.sparse_mask(grad)._values()
        exp_avg_sq_update_values = (
            grad_values.pow(2).sub_(old_exp_avg_sq_values).mul_(1 - beta2)
        )
        exp_avg_sq.add_(make_sparse(exp_avg_sq_update_values))

        # 计算更新公式中的分子和分母
        numer = exp_avg_update_values.add_(old_exp_avg_values)
        exp_avg_sq_update_values.add_(old_exp_avg_sq_values)
        denom = exp_avg_sq_update_values.sqrt_().add_(eps)
        del exp_avg_update_values, exp_avg_sq_update_values

        # 计算偏置修正项
        bias_correction1 = 1 - beta1**step
        bias_correction2 = 1 - beta2**step

        # 计算步长
        step_size = lr * math.sqrt(bias_correction2) / bias_correction1

        # 更新参数
        param.add_(make_sparse(-step_size * numer.div_(denom)))
```