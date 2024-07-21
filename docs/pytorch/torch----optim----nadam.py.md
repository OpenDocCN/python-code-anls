# `.\pytorch\torch\optim\nadam.py`

```
# mypy: allow-untyped-defs
# 引入NAdam算法的实现
r"""Implementation for the NAdam algorithm."""
from typing import cast, List, Optional, Tuple, Union

import torch
from torch import Tensor
# 从本地的optimizer模块导入所需函数和类
from .optimizer import (
    _capturable_doc,
    _default_to_fused_or_foreach,
    _differentiable_doc,
    _disable_dynamo_if_unsupported,
    _dispatch_sqrt,
    _foreach_doc,
    _get_capturable_supported_devices,
    _get_scalar_dtype,
    _get_value,
    _maximize_doc,
    _stack_if_compiling,
    _use_grad_for_differentiable,
    _view_as_real,
    Optimizer,
    ParamsT,
)

__all__ = ["NAdam", "nadam"]


class NAdam(Optimizer):  # noqa: D101
    def __init__(
        self,
        params: ParamsT,
        lr: float = 2e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        momentum_decay: float = 4e-3,
        decoupled_weight_decay: bool = False,
        *,
        foreach: Optional[bool] = None,
        maximize: bool = False,
        capturable: bool = False,
        differentiable: bool = False,
    ):  # noqa: D107
        # 检查学习率lr是否合法
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        # 检查epsilon值eps是否合法
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        # 检查beta参数betas[0]是否在合法范围内
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        # 检查beta参数betas[1]是否在合法范围内
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        # 检查weight_decay是否合法
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        # 检查momentum_decay是否合法
        if not 0.0 <= momentum_decay:
            raise ValueError(f"Invalid momentum_decay value: {momentum_decay}")
        
        # 设置默认参数字典
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            momentum_decay=momentum_decay,
            decoupled_weight_decay=decoupled_weight_decay,
            maximize=maximize,
            foreach=foreach,
            capturable=capturable,
            differentiable=differentiable,
        )
        # 调用父类Optimizer的初始化方法，传入参数和默认值字典
        super().__init__(params, defaults)
    def __setstate__(self, state):  # noqa: D105
        # 调用父类的 __setstate__ 方法，将状态恢复到对象中
        super().__setstate__(state)
        
        # 遍历优化器的参数组列表
        for group in self.param_groups:
            # 设置默认参数，如果不存在则使用指定的默认值
            group.setdefault("maximize", False)
            group.setdefault("foreach", None)
            group.setdefault("capturable", False)
            group.setdefault("differentiable", False)
            group.setdefault("decoupled_weight_decay", False)
            
            # 遍历当前参数组中的参数列表
            for p in group["params"]:
                # 获取当前参数的状态
                p_state = self.state.get(p, [])
                
                # 如果参数状态不为空
                if len(p_state) != 0:
                    # 如果当前步骤（step）不是张量，则将其转换为指定设备上的张量
                    if not torch.is_tensor(p_state["step"]):
                        step_val = float(p_state["step"])
                        p_state["step"] = (
                            torch.tensor(
                                step_val, dtype=_get_scalar_dtype(), device=p.device
                            )
                            if group["capturable"]
                            else torch.tensor(step_val, dtype=_get_scalar_dtype())
                        )
                    
                    # 如果当前的 mu_product 不是张量，则将其转换为指定设备上的张量
                    if not torch.is_tensor(p_state["mu_product"]):
                        mu_prod_val = p_state["mu_product"]
                        p_state["mu_product"] = (
                            torch.tensor(
                                mu_prod_val, dtype=_get_scalar_dtype(), device=p.device
                            )
                            if group["capturable"]
                            else torch.tensor(mu_prod_val, dtype=_get_scalar_dtype())
                        )

    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        exp_avgs,
        exp_avg_sqs,
        mu_products,
        state_steps,
        ):
        # 初始化一个标志变量，用于标识是否存在复数类型的张量参数
        has_complex = False
        # 遍历参数组中的每一个参数
        for p in group["params"]:
            # 检查参数是否有梯度
            if p.grad is not None:
                # 检查参数是否是复数类型
                has_complex |= torch.is_complex(p)
                # 将具有梯度的参数添加到列表中
                params_with_grad.append(p)
                # 如果参数的梯度是稀疏的，则抛出错误
                if p.grad.is_sparse:
                    raise RuntimeError("NAdam does not support sparse gradients")
                # 将参数的梯度添加到梯度列表中
                grads.append(p.grad)

                # 获取参数对应的状态字典
                state = self.state[p]
                # 懒惰状态初始化
                if len(state) == 0:
                    # 初始化参数的步数 `step` 和 `mu_product`
                    # 如果 capturable 是 False，则故意将它们放在 CPU 上
                    # 因为在 CUDA 和 XLA 上进行核启动是昂贵的
                    state["step"] = (
                        torch.zeros((), dtype=_get_scalar_dtype(), device=p.device)
                        if group["capturable"]
                        else torch.tensor(0.0, dtype=_get_scalar_dtype())
                    )
                    state["mu_product"] = (
                        torch.ones((), dtype=_get_scalar_dtype(), device=p.device)
                        if group["capturable"]
                        else torch.tensor(1.0, dtype=_get_scalar_dtype())
                    )
                    # 初始化梯度值的指数移动平均值 `exp_avg`
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # 初始化梯度平方值的指数移动平均值 `exp_avg_sq`
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                # 将各种状态的引用添加到对应的列表中
                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])
                mu_products.append(state["mu_product"])
                state_steps.append(state["step"])
        # 返回是否存在复数类型参数的标志变量
        return has_complex

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        # 执行单步优化操作的方法

        # 检查 CUDA 图形捕获的健康状态
        self._cuda_graph_capture_health_check()

        # 初始化损失为 None
        loss = None
        # 如果传入了闭包函数 closure
        if closure is not None:
            # 启用梯度计算上下文管理器
            with torch.enable_grad():
                # 重新评估模型并计算损失
                loss = closure()

        # 遍历参数组
        for group in self.param_groups:
            # 初始化空列表以存储有梯度的参数、梯度值等
            params_with_grad: List[Tensor] = []
            grads: List[Tensor] = []
            exp_avgs: List[Tensor] = []
            exp_avg_sqs: List[Tensor] = []
            mu_products: List[Tensor] = []
            state_steps: List[Tensor] = []
            # 从参数组中获取 beta1 和 beta2
            beta1, beta2 = cast(Tuple[float, float], group["betas"])

            # 初始化参数组，获取是否包含复数参数的信息
            has_complex = self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                mu_products,
                state_steps,
            )

            # 调用 NAdam 优化器的方法，执行优化步骤
            nadam(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                mu_products,
                state_steps,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                momentum_decay=group["momentum_decay"],
                eps=group["eps"],
                maximize=group["maximize"],
                decoupled_weight_decay=group["decoupled_weight_decay"],
                foreach=group["foreach"],
                capturable=group["capturable"],
                differentiable=group["differentiable"],
                has_complex=has_complex,
            )

        # 返回损失值（如果有的话）
        return loss
# 将 NAdam 类的文档字符串设置为指定的内容，描述了 NAdam 算法的实现细节和参数说明
NAdam.__doc__ = (
    r"""Implements NAdam algorithm.

    For further details regarding the algorithm we refer to `Incorporating Nesterov Momentum into Adam`_.
    """
    + rf"""
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 2e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        momentum_decay (float, optional): momentum momentum_decay (default: 4e-3)
        decoupled_weight_decay (bool, optional): whether to use decoupled weight
            decay as in AdamW to obtain NAdamW (default: False)
        {_foreach_doc}
        {_maximize_doc}
        {_capturable_doc}
        {_differentiable_doc}

    .. _Incorporating Nesterov Momentum into Adam:
        https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101

    """
)


# 定义了一个名为 _single_tensor_nadam 的函数，实现了 NAdam 算法的优化步骤
def _single_tensor_nadam(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    mu_products: List[Tensor],
    state_steps: List[Tensor],
    *,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    momentum_decay: float,
    eps: float,
    decoupled_weight_decay: bool,
    maximize: bool,
    capturable: bool,
    differentiable: bool,
    has_complex: bool,
):
    # 遍历参数列表及其梯度，以及相关状态变量
    for i, param in enumerate(params):
        # 根据是否最大化来确定梯度的方向
        grad = grads[i] if not maximize else -grads[i]
        # 获取指数移动平均的第一阶段结果
        exp_avg = exp_avgs[i]
        # 获取指数移动平均的第二阶段结果
        exp_avg_sq = exp_avg_sqs[i]
        # 获取动量乘积
        mu_product = mu_products[i]
        # 获取当前步骤数
        step_t = state_steps[i]

        # 如果参数是复数，则将其视为实数处理
        if torch.is_complex(param):
            param = torch.view_as_real(param)
            grad = torch.view_as_real(grad)
            exp_avg = torch.view_as_real(exp_avg)
            exp_avg_sq = torch.view_as_real(exp_avg_sq)

        # 如果不是在编译过程中，并且支持捕获，则进行捕获支持设备的检查
        if not torch._utils.is_compiling() and capturable:
            capturable_supported_devices = _get_capturable_supported_devices()
            assert (
                param.device.type == mu_product.device.type == step_t.device.type
                and param.device.type in capturable_supported_devices
            ), (
                f"If capturable=True, params, mu_products and state_steps must be "
                f"on supported devices: {capturable_supported_devices}."
            )

        # 更新步骤数
        step_t += 1

        # 如果支持捕获，则使用当前步骤数作为步数
        if capturable:
            step = step_t
        else:
            step = _get_value(step_t)

        # 计算偏置修正项2
        bias_correction2 = 1 - beta2**step

        # 如果有权重衰减，则根据设置执行不同的处理方式
        if weight_decay != 0:
            if decoupled_weight_decay:
                # 执行独立的权重衰减步骤
                param.mul_(1 - lr * weight_decay)
            else:
                # 将权重衰减项添加到梯度中
                grad = grad.add(param, alpha=weight_decay)

        # 计算动量缓存 \mu^{t} 和 \mu^{t+1}
        mu = beta1 * (1.0 - 0.5 * (0.96 ** (step * momentum_decay)))
        mu_next = beta1 * (1.0 - 0.5 * (0.96 ** ((step + 1) * momentum_decay)))

        # 更新动量乘积
        mu_product *= mu

        # 衰减第一和第二动量的平均系数
        exp_avg.lerp_(grad, 1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        denom = exp_avg_sq.div(bias_correction2).sqrt()

        # 如果需要可微分或者支持捕获
        if differentiable or capturable:
            denom = denom.add(eps)
            # 更新 grad 和 exp_avg 直接不使用 addcdiv 的标量"value"参数，使得自动求导跟踪这些操作
            mu_product_next = mu_product * mu_next
            grad = grad * (-lr * (1.0 - mu) / (1.0 - mu_product))
            exp_avg = exp_avg * (-lr * mu_next / (1.0 - mu_product_next))
            param.addcdiv_(grad, denom)
            param.addcdiv_(exp_avg, denom)
        else:
            mu_product_next = _get_value(mu_product) * mu_next
            denom.add_(eps)
            param.addcdiv_(
                grad, denom, value=(-lr * (1.0 - mu) / (1.0 - _get_value(mu_product)))
            )
            param.addcdiv_(
                exp_avg, denom, value=(-lr * mu_next) / (1.0 - mu_product_next)
            )
def _multi_tensor_nadam(
    params: List[Tensor],  # 参数列表，包含模型参数张量
    grads: List[Tensor],  # 梯度列表，包含对应模型参数的梯度张量
    exp_avgs: List[Tensor],  # 指数平均值列表，用于动量更新
    exp_avg_sqs: List[Tensor],  # 指数平方平均值列表，用于二阶动量更新
    mu_products: List[Tensor],  # 动量乘积列表，用于NAdam算法中的自适应动量
    state_steps: List[Tensor],  # 状态步骤列表，用于更新步数
    *,
    beta1: float,  # NAdam算法的动量衰减率
    beta2: float,  # NAdam算法的二阶动量衰减率
    lr: float,  # 学习率
    weight_decay: float,  # 权重衰减率
    momentum_decay: float,  # 动量衰减率
    eps: float,  # 数值稳定性参数
    decoupled_weight_decay: bool,  # 是否使用解耦的权重衰减
    maximize: bool,  # 是否最大化优化目标
    capturable: bool,  # 是否支持捕获
    differentiable: bool,  # 是否支持自动求导
    has_complex: bool,  # 模型是否包含复杂类型数据
):
    if len(params) == 0:  # 如果参数列表为空，则直接返回
        return

    assert not differentiable, "_foreach ops don't support autograd"  # 断言不支持自动求导，因为 _foreach 操作不支持

    # If compiling, the compiler will handle cudagraph checks, see note [torch.compile x capturable]
    if not torch._utils.is_compiling() and capturable:  # 如果不在编译状态且支持捕获
        capturable_supported_devices = _get_capturable_supported_devices(
            supports_xla=False
        )
        assert all(
            p.device.type == mp.device.type == step.device.type
            and p.device.type in capturable_supported_devices
            for p, mp, step in zip(params, mu_products, state_steps)
        ), f"If capturable=True, params, mu_products, and state_steps must be on supported devices: {capturable_supported_devices}."

    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, exp_avgs, exp_avg_sqs, mu_products, state_steps]
    )
    for (
        grouped_params,  # 按设备和数据类型分组的参数列表
        grouped_grads,  # 按设备和数据类型分组的梯度列表
        grouped_exp_avgs,  # 按设备和数据类型分组的指数平均值列表
        grouped_exp_avg_sqs,  # 按设备和数据类型分组的指数平方平均值列表
        grouped_mu_products,  # 按设备和数据类型分组的动量乘积列表
        grouped_state_steps,  # 按设备和数据类型分组的状态步骤列表
@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_nadam)
def nadam(
    params: List[Tensor],  # 参数列表，包含模型参数张量
    grads: List[Tensor],  # 梯度列表，包含对应模型参数的梯度张量
    exp_avgs: List[Tensor],  # 指数平均值列表，用于动量更新
    exp_avg_sqs: List[Tensor],  # 指数平方平均值列表，用于二阶动量更新
    mu_products: List[Tensor],  # 动量乘积列表，用于NAdam算法中的自适应动量
    state_steps: List[Tensor],  # 状态步骤列表，用于更新步数
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    decoupled_weight_decay: bool = False,  # 是否使用解耦的权重衰减，默认为False
    foreach: Optional[bool] = None,  # 是否使用 foreach 操作，默认为None
    capturable: bool = False,  # 是否支持捕获，默认为False
    differentiable: bool = False,  # 是否支持自动求导，默认为False
    has_complex: bool = False,  # 模型是否包含复杂类型数据，默认为False
    maximize: bool = False,  # 是否最大化优化目标，默认为False
    *,
    beta1: float,  # NAdam算法的动量衰减率
    beta2: float,  # NAdam算法的二阶动量衰减率
    lr: float,  # 学习率
    weight_decay: float,  # 权重衰减率
    momentum_decay: float,  # 动量衰减率
    eps: float,  # 数值稳定性参数
):
    r"""Functional API that performs NAdam algorithm computation.

    See :class:`~torch.optim.NAdam` for details.
    """
    if not all(isinstance(t, torch.Tensor) for t in state_steps):
        raise RuntimeError(
            "API has changed, `state_steps` argument must contain a list of singleton tensors"
        )

    if not all(isinstance(t, torch.Tensor) for t in mu_products):
        raise RuntimeError(
            "API has changed, `mu_products` argument must contain a list of singleton tensors"
        )

    if foreach is None:
        _, foreach = _default_to_fused_or_foreach(
            params, differentiable, use_fused=False
        )
    # 如果使用了 foreach 且当前处于 Torch 脚本模式，则抛出运行时错误，因为 foreach 优化器不支持 Torch 脚本。
    if foreach and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with foreach optimizers")

    # 如果使用了 foreach 且当前不处于 Torch 脚本模式，则选择多张量版本的 Nadam 优化函数。
    # 否则选择单张量版本的 Nadam 优化函数。
    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_nadam
    else:
        func = _single_tensor_nadam

    # 调用选定的 Nadam 优化函数，传入参数列表和关键字参数。
    func(
        params,             # 待优化的参数列表
        grads,              # 参数的梯度列表
        exp_avgs,           # 参数的指数移动平均值列表
        exp_avg_sqs,        # 参数的指数移动平均平方值列表
        mu_products,        # 参数的 mu 乘积列表
        state_steps,        # 参数的状态步数列表
        beta1=beta1,        # Nadam 优化算法的 beta1 参数
        beta2=beta2,        # Nadam 优化算法的 beta2 参数
        lr=lr,              # 学习率
        weight_decay=weight_decay,  # 权重衰减（L2 正则化）参数
        momentum_decay=momentum_decay,  # 动量衰减参数
        maximize=maximize,  # 是否最大化优化目标
        decoupled_weight_decay=decoupled_weight_decay,  # 是否使用解耦的权重衰减
        eps=eps,            # 数值稳定性参数
        capturable=capturable,  # 是否支持捕获状态
        differentiable=differentiable,  # 参数是否可微分
        has_complex=has_complex,  # 参数是否包含复数类型
    )
```