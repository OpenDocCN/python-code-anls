# `.\pytorch\torch\optim\sparse_adam.py`

```py
# 设置类型提示允许未定义函数
mypy: allow-untyped-defs
# 导入需要的模块和类型
from typing import List, Tuple
# 导入 torch 模块及相关内容
import torch
from torch import Tensor
# 导入自定义的 _functional 模块
from . import _functional as F
# 导入自定义 optimizer 模块中的 _maximize_doc、Optimizer 和 ParamsT
from .optimizer import _maximize_doc, Optimizer, ParamsT

# 暴露 SparseAdam 类
__all__ = ["SparseAdam"]


# 定义 SparseAdam 类，继承自 Optimizer 类
class SparseAdam(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        maximize: bool = False,
    ):
        # 检查学习率是否有效
        if not 0.0 < lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        # 检查 epsilon 值是否有效
        if not 0.0 < eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        # 检查 beta 参数是否有效
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        # 检查 beta 参数是否有效
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        # 设置默认参数
        defaults = dict(lr=lr, betas=betas, eps=eps, maximize=maximize)
        # 调用父类 Optimizer 的初始化方法
        super().__init__(params, defaults)

        # 初始化存储稀疏参数和复杂参数的列表
        sparse_params = []
        complex_params = []
        # 遍历参数组
        for index, param_group in enumerate(self.param_groups):
            # 断言参数组是字典类型
            assert isinstance(
                param_group, dict
            ), f"param_groups must be a list of dicts, but got {type(param_group)}"
            # 遍历参数并判断是否为稀疏或复杂参数
            for d_index, d_param in enumerate(param_group["params"]):
                if d_param.is_sparse:
                    sparse_params.append([index, d_index])
                if d_param.is_complex():
                    complex_params.append([index, d_index])
        # 抛出异常，稀疏参数不支持 SparseAdam
        if sparse_params:
            raise ValueError(
                f"Sparse params at indices {sparse_params}: SparseAdam requires dense parameter tensors"
            )
        # 抛出异常，复杂参数不支持 SparseAdam
        if complex_params:
            raise ValueError(
                f"Complex params at indices {complex_params}: SparseAdam does not support complex parameters"
            )

    # 使用 torch.no_grad() 修饰
    @torch.no_grad()
    def step(self, closure=None):
        """执行单个优化步骤。

        Args:
            closure (Callable, optional): 一个闭包，重新评估模型并返回损失。
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # 遍历每个参数组
        for group in self.param_groups:
            params_with_grad: List[Tensor] = []
            grads: List[Tensor] = []
            exp_avgs: List[Tensor] = []
            exp_avg_sqs: List[Tensor] = []
            state_steps: List[int] = []
            beta1, beta2 = group["betas"]
            maximize = group.get("maximize", False)

            # 遍历参数组中的每个参数
            for p in group["params"]:
                if p.grad is not None:
                    params_with_grad.append(p)
                    # 检查梯度是否为稀疏张量
                    if not p.grad.is_sparse:
                        raise RuntimeError(
                            "SparseAdam does not support dense gradients, please consider Adam instead"
                        )
                    grads.append(p.grad)

                    state = self.state[p]

                    # 状态初始化
                    if len(state) == 0:
                        state["step"] = 0
                        # 梯度值的指数移动平均
                        state["exp_avg"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                        # 梯度值平方的指数移动平均
                        state["exp_avg_sq"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )

                    exp_avgs.append(state["exp_avg"])
                    exp_avg_sqs.append(state["exp_avg_sq"])

                    # 更新每个参数组的步骤数
                    state["step"] += 1
                    # 记录更新后的步骤数
                    state_steps.append(state["step"])

            # 调用 F.sparse_adam 进行优化步骤
            F.sparse_adam(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                state_steps,
                eps=group["eps"],
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                maximize=maximize,
            )

        return loss
# 将 SparseAdam 类的文档字符串设为一个包含详细说明的原始格式字面字符串
SparseAdam.__doc__ = rf"""SparseAdam implements a masked version of the Adam algorithm
    suitable for sparse gradients. Currently, due to implementation constraints (explained
    below), SparseAdam is only intended for a narrow subset of use cases, specifically
    parameters of a dense layout with gradients of a sparse layout. This occurs in a
    special case where the module backwards produces grads already in a sparse layout.
    One example NN module that behaves as such is ``nn.Embedding(sparse=True)``.

    SparseAdam approximates the Adam algorithm by masking out the parameter and moment
    updates corresponding to the zero values in the gradients. Whereas the Adam algorithm
    will update the first moment, the second moment, and the parameters based on all values
    of the gradients, SparseAdam only updates the moments and parameters corresponding
    to the non-zero values of the gradients.

    A simplified way of thinking about the `intended` implementation is as such:

    1. Create a mask of the non-zero values in the sparse gradients. For example,
       if your gradient looks like [0, 5, 0, 0, 9], the mask would be [0, 1, 0, 0, 1].
    2. Apply this mask over the running moments and do computation on only the
       non-zero values.
    3. Apply this mask over the parameters and only apply an update on non-zero values.

    In actuality, we use sparse layout Tensors to optimize this approximation, which means the
    more gradients that are masked by not being materialized, the more performant the optimization.
    Since we rely on using sparse layout tensors, we infer that any materialized value in the
    sparse layout is non-zero and we do NOT actually verify that all values are not zero!
    It is important to not conflate a semantically sparse tensor (a tensor where many
    of its values are zeros) with a sparse layout tensor (a tensor where ``.is_sparse``
    returns ``True``). The SparseAdam approximation is intended for `semantically` sparse
    tensors and the sparse layout is only a implementation detail. A clearer implementation
    would be to use MaskedTensors, but those are experimental.


    .. note::

        If you suspect your gradients are semantically sparse (but do not have sparse
        layout), this variant may not be the best for you. Ideally, you want to avoid
        materializing anything that is suspected to be sparse in the first place, since
        needing to convert all your grads from dense layout to sparse layout may outweigh
        the performance gain. Here, using Adam may be the best alternative, unless you
        can easily rig up your module to output sparse grads similar to
        ``nn.Embedding(sparse=True)``. If you insist on converting your grads, you can do
        so by manually overriding your parameters' ``.grad`` fields with their sparse
        equivalents before calling ``.step()``.
"""
    Args:
        params (iterable): 用于优化的参数迭代器或定义参数组的字典
        lr (float, optional): 学习率（默认值: 1e-3）
        betas (Tuple[float, float], optional): 用于计算梯度及其平方的运行平均值的系数（默认值: (0.9, 0.999)）
        eps (float, optional): 添加到分母以提高数值稳定性的项（默认值: 1e-8）
        {_maximize_doc}
    
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
```