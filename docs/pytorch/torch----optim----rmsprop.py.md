# `.\pytorch\torch\optim\rmsprop.py`

```
# mypy: allow-untyped-defs
r"""Implementation for the RMSprop algorithm."""
from typing import List, Optional

import torch
from torch import Tensor
from .optimizer import (
    _capturable_doc,
    _default_to_fused_or_foreach,
    _differentiable_doc,
    _disable_dynamo_if_unsupported,
    _foreach_doc,
    _get_capturable_supported_devices,
    _get_scalar_dtype,
    _maximize_doc,
    _use_grad_for_differentiable,
    _view_as_real,
    Optimizer,
    ParamsT,
)

__all__ = ["RMSprop", "rmsprop"]


class RMSprop(Optimizer):  # noqa: D101
    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-2,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0,
        momentum: float = 0,
        centered=False,
        capturable=False,
        foreach: Optional[bool] = None,
        maximize: bool = False,
        differentiable: bool = False,
    ):  # noqa: D107
        # 检查并确保参数值在有效范围内，否则抛出异常
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= momentum:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= alpha:
            raise ValueError(f"Invalid alpha value: {alpha}")

        # 初始化默认参数字典
        defaults = dict(
            lr=lr,
            momentum=momentum,
            alpha=alpha,
            eps=eps,
            centered=centered,
            weight_decay=weight_decay,
            capturable=capturable,
            foreach=foreach,
            maximize=maximize,
            differentiable=differentiable,
        )
        # 调用父类的初始化方法
        super().__init__(params, defaults)

    def __setstate__(self, state):  # noqa: D105
        # 恢复对象状态
        super().__setstate__(state)
        for group in self.param_groups:
            # 设置默认组属性，确保存在并设置正确的数据类型
            group.setdefault("momentum", 0)
            group.setdefault("centered", False)
            group.setdefault("foreach", None)
            group.setdefault("maximize", False)
            group.setdefault("differentiable", False)
            group.setdefault("capturable", False)
            for p in group["params"]:
                # 获取参数状态并检查步数是否为张量，若不是则转换为正确的数据类型
                p_state = self.state.get(p, [])
                if len(p_state) != 0 and not torch.is_tensor(p_state["step"]):
                    step_val = float(p_state["step"])
                    p_state["step"] = (
                        torch.tensor(
                            step_val, dtype=_get_scalar_dtype(), device=p.device
                        )
                        if group["capturable"]
                        else torch.tensor(step_val, dtype=_get_scalar_dtype())
                    )

    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        square_avgs,
        momentum_buffer_list,
        grad_avgs,
        state_steps,
        ):
            # 初始化标志位，指示是否存在复数张量
            has_complex = False
            # 遍历参数组中的每一个参数
            for p in group["params"]:
                # 如果梯度为None，则跳过当前参数
                if p.grad is None:
                    continue
                # 检查当前参数是否包含复数元素，并更新标志位
                has_complex |= torch.is_complex(p)
                # 将具有梯度的参数添加到列表中
                params_with_grad.append(p)

                # 如果梯度是稀疏的，则抛出运行时错误
                if p.grad.is_sparse:
                    raise RuntimeError("RMSprop does not support sparse gradients")
                # 将当前参数的梯度添加到梯度列表中
                grads.append(p.grad)

                # 获取当前参数的状态字典
                state = self.state[p]

                # 状态初始化
                if len(state) == 0:
                    # 初始化步数状态，根据参数是否可以捕获选择设备
                    state["step"] = (
                        torch.zeros((), dtype=_get_scalar_dtype(), device=p.device)
                        if group["capturable"]
                        else torch.zeros((), dtype=_get_scalar_dtype())
                    )
                    # 初始化平方平均状态，保留原始内存格式
                    state["square_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # 如果设置了动量，初始化动量缓冲状态
                    if group["momentum"] > 0:
                        state["momentum_buffer"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                    # 如果设置了中心化参数，初始化梯度平均状态
                    if group["centered"]:
                        state["grad_avg"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                # 将当前参数的平方平均状态添加到列表中
                square_avgs.append(state["square_avg"])
                # 将当前参数的步数状态添加到列表中
                state_steps.append(state["step"])

                # 如果设置了动量，将当前参数的动量缓冲状态添加到列表中
                if group["momentum"] > 0:
                    momentum_buffer_list.append(state["momentum_buffer"])
                # 如果设置了中心化参数，将当前参数的梯度平均状态添加到列表中
                if group["centered"]:
                    grad_avgs.append(state["grad_avg"])

            # 返回是否存在复数张量的标志位
            return has_complex

        # 使用装饰器处理不同iable的梯度
        @_use_grad_for_differentiable
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        # 检查 CUDA 图形捕获状态
        self._cuda_graph_capture_health_check()

        # 初始化损失为 None
        loss = None
        # 如果有传入闭包函数，则在启用梯度计算的环境下重新评估模型并计算损失
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # 遍历每个参数组
        for group in self.param_groups:
            # 初始化用于存储参数梯度、梯度、平方平均、梯度平均、动量缓冲列表和状态步骤的空列表
            params_with_grad: List[Tensor] = []
            grads: List[Tensor] = []
            square_avgs: List[Tensor] = []
            grad_avgs: List[Tensor] = []
            momentum_buffer_list: List[Tensor] = []
            state_steps: List[Tensor] = []

            # 初始化参数组，获取参数梯度等相关信息，并返回是否包含复数的标志
            has_complex = self._init_group(
                group,
                params_with_grad,
                grads,
                square_avgs,
                momentum_buffer_list,
                grad_avgs,
                state_steps,
            )

            # 调用 RMSprop 优化器方法，更新参数
            rmsprop(
                params_with_grad,
                grads,
                square_avgs,
                grad_avgs,
                momentum_buffer_list,
                state_steps,
                lr=group["lr"],
                alpha=group["alpha"],
                eps=group["eps"],
                weight_decay=group["weight_decay"],
                momentum=group["momentum"],
                centered=group["centered"],
                foreach=group["foreach"],
                maximize=group["maximize"],
                differentiable=group["differentiable"],
                capturable=group["capturable"],
                has_complex=has_complex,
            )

        # 返回损失值
        return loss
# 将 RMSprop 类的 __doc__ 属性设置为 RMSprop 算法的文档字符串
RMSprop.__doc__ = (
    r"""Implements RMSprop algorithm.

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \alpha \text{ (alpha)},\: \gamma \text{ (lr)},
                \: \theta_0 \text{ (params)}, \: f(\theta) \text{ (objective)}                   \\
            &\hspace{13mm}   \lambda \text{ (weight decay)},\: \mu \text{ (momentum)},\: centered\\
            &\textbf{initialize} : v_0 \leftarrow 0 \text{ (square average)}, \:
                \textbf{b}_0 \leftarrow 0 \text{ (buffer)}, \: g^{ave}_0 \leftarrow 0     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{5mm}if \: \lambda \neq 0                                                    \\
            &\hspace{10mm} g_t \leftarrow g_t + \lambda  \theta_{t-1}                            \\
            &\hspace{5mm}v_t           \leftarrow   \alpha v_{t-1} + (1 - \alpha) g^2_t
                \hspace{8mm}                                                                     \\
            &\hspace{5mm} \tilde{v_t} \leftarrow v_t                                             \\
            &\hspace{5mm}if \: centered                                                          \\
            &\hspace{10mm} g^{ave}_t \leftarrow g^{ave}_{t-1} \alpha + (1-\alpha) g_t            \\
            &\hspace{10mm} \tilde{v_t} \leftarrow \tilde{v_t} -  \big(g^{ave}_{t} \big)^2        \\
            &\hspace{5mm}if \: \mu > 0                                                           \\
            &\hspace{10mm} \textbf{b}_t\leftarrow \mu \textbf{b}_{t-1} +
                g_t/ \big(\sqrt{\tilde{v_t}} +  \epsilon \big)                                   \\
            &\hspace{10mm} \theta_t \leftarrow \theta_{t-1} - \gamma \textbf{b}_t                \\
            &\hspace{5mm} else                                                                   \\
            &\hspace{10mm}\theta_t      \leftarrow   \theta_{t-1} -
                \gamma  g_t/ \big(\sqrt{\tilde{v_t}} + \epsilon \big)  \hspace{3mm}              \\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    For further details regarding the algorithm we refer to
    `lecture notes <https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf>`_ by G. Hinton.
    and centered version `Generating Sequences
"""
)
    # 此处是一个长字符串的文档字符串，用来描述 RMSProp 优化器的实现细节和参数说明。
    # 文档字符串包含了多个段落，介绍了算法的工作原理以及每个参数的作用和默认值。
    # 它还提到了 TensorFlow 中的一些实现细节，如梯度平均值开平方和添加 epsilon 的顺序。
    # 最后使用了 f-string（rf""）来格式化参数部分的文本内容，其中包含了几个其他的文档字符串变量（_foreach_doc, _maximize_doc, _capturable_doc, _differentiable_doc）。
    """
    + rf"""
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        momentum (float, optional): momentum factor (default: 0)
        alpha (float, optional): smoothing constant (default: 0.99)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        centered (bool, optional) : if ``True``, compute the centered RMSProp,
            the gradient is normalized by an estimation of its variance
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        {_foreach_doc}  # 参数说明中可能包含的文档字符串变量之一
        {_maximize_doc}  # 参数说明中可能包含的文档字符串变量之一
        {_capturable_doc}  # 参数说明中可能包含的文档字符串变量之一
        {_differentiable_doc}  # 参数说明中可能包含的文档字符串变量之一
    
    """
    )


def _single_tensor_rmsprop(
    params: List[Tensor],                       # 参数列表，包含待更新的张量
    grads: List[Tensor],                        # 梯度列表，对应每个参数的梯度张量
    square_avgs: List[Tensor],                  # 平方平均列表，用于存储平方梯度的移动平均
    grad_avgs: List[Tensor],                    # 梯度平均列表，用于存储梯度的移动平均
    momentum_buffer_list: List[Tensor],          # 动量缓存列表，用于存储动量更新的缓存张量
    state_steps: List[Tensor],                  # 状态步数列表，用于记录参数更新的步数
    *,
    lr: float,                                  # 学习率
    alpha: float,                               # 移动平均系数
    eps: float,                                 # 用于数值稳定性的小常数
    weight_decay: float,                        # 权重衰减（L2正则化）的权重
    momentum: float,                            # 动量参数
    centered: bool,                             # 是否使用中心化的RMSProp
    maximize: bool,                             # 是否最大化优化目标（取负梯度）
    differentiable: bool,                       # 是否支持微分
    capturable: bool,                           # 是否支持捕获
    has_complex: bool,                          # 是否有复数类型的参数
):
    for i, param in enumerate(params):          # 遍历参数列表，同时获取索引和参数张量
        step = state_steps[i]                   # 获取当前参数的步数状态

        # 如果不是编译模式且支持捕获
        if not torch._utils.is_compiling() and capturable:
            # 获取支持捕获的设备列表
            capturable_supported_devices = _get_capturable_supported_devices()
            # 断言参数张量和步数状态张量在支持捕获的设备上
            assert (
                param.device.type == step.device.type
                and param.device.type in capturable_supported_devices
            ), f"If capturable=True, params and state_steps must be on supported devices: {capturable_supported_devices}."

        grad = grads[i]                         # 获取当前参数的梯度张量
        grad = grad if not maximize else -grad  # 如果需要最大化优化目标，则取负梯度
        square_avg = square_avgs[i]             # 获取当前参数的平方平均张量

        step += 1                               # 更新步数

        if weight_decay != 0:
            # 加上权重衰减的梯度
            grad = grad.add(param, alpha=weight_decay)

        is_complex_param = torch.is_complex(param)  # 判断当前参数是否是复数类型
        if is_complex_param:
            # 如果是复数类型的参数，将参数、梯度和平方平均张量视为实数处理
            param = torch.view_as_real(param)
            grad = torch.view_as_real(grad)
            square_avg = torch.view_as_real(square_avg)

        # 更新平方平均张量
        square_avg.mul_(alpha).addcmul_(grad, grad, value=1 - alpha)

        if centered:
            # 如果使用中心化的RMSProp
            grad_avg = grad_avgs[i]
            if is_complex_param:
                grad_avg = torch.view_as_real(grad_avg)
            grad_avg.lerp_(grad, 1 - alpha)
            avg = square_avg.addcmul(grad_avg, grad_avg, value=-1).sqrt_()
        else:
            avg = square_avg.sqrt()             # 否则直接对平方平均张量开平方

        if differentiable:
            avg = avg.add(eps)                  # 如果支持微分，加上小常数eps
        else:
            avg = avg.add_(eps)                 # 否则原地加上小常数eps

        if momentum > 0:
            buf = momentum_buffer_list[i]
            if is_complex_param:
                buf = torch.view_as_real(buf)
            buf.mul_(momentum).addcdiv_(grad, avg)  # 更新动量缓存
            param.add_(buf, alpha=-lr)          # 使用动量更新参数
        else:
            param.addcdiv_(grad, avg, value=-lr)  # 直接使用更新参数的公式


def _multi_tensor_rmsprop(
    params: List[Tensor],
    grads: List[Tensor],
    square_avgs: List[Tensor],
    grad_avgs: List[Tensor],
    momentum_buffer_list: List[Tensor],
    state_steps: List[Tensor],
    *,
    lr: float,
    alpha: float,
    eps: float,
    weight_decay: float,
    momentum: float,
    centered: bool,
    maximize: bool,
    differentiable: bool,
    capturable: bool,
    has_complex: bool,
):
    if len(params) == 0:
        return

    assert not differentiable, "_foreach ops don't support autograd"

    # 如果正在编译，编译器将处理 cudagraph 检查，详见注释 [torch.compile x capturable]
    # 如果不是在编译 Torch 模型且 capturable=True，则进行以下条件检查
    if not torch._utils.is_compiling() and capturable:
        # 获取支持捕获的设备列表
        capturable_supported_devices = _get_capturable_supported_devices()
        # 断言所有参数和状态步骤都在支持的设备上，并且设备类型一致
        assert all(
            p.device.type == step.device.type
            and p.device.type in capturable_supported_devices
            for p, step in zip(params, state_steps)
        ), f"If capturable=True, params and state_steps must be on supported devices: {capturable_supported_devices}."

    # 将一组张量按照设备和数据类型分组
    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, square_avgs, grad_avgs, momentum_buffer_list, state_steps]
    )
    # 遍历分组后的张量集合
    for (
        (
            grouped_params,
            grouped_grads,
            grouped_square_avgs,
            grouped_grad_avgs,
            grouped_momentum_buffer_list,
            grouped_state_steps,
        )
# 用装饰器禁用 Dynamo 如果不支持指定的单一张量函数 `_single_tensor_rmsprop`
@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_rmsprop)
# RMSProp 算法的函数接口，接受多个张量作为参数进行优化
def rmsprop(
    params: List[Tensor],  # 待优化的张量列表
    grads: List[Tensor],  # 对应张量的梯度列表
    square_avgs: List[Tensor],  # 平方平均值的张量列表
    grad_avgs: List[Tensor],  # 梯度平均值的张量列表
    momentum_buffer_list: List[Tensor],  # 动量缓存的张量列表
    state_steps: List[Tensor],  # 状态步数的张量列表
    # 下面的参数是具有默认值的关键字参数，由于 torchscript 问题 #70627 的影响，当前不支持
    # 暂时作为关键字参数来设置，因为 torch/distributed/optim 编译了函数 API
    foreach: Optional[bool] = None,  # 是否应用于每个张量的布尔值
    maximize: bool = False,  # 是否最大化目标函数的布尔值
    differentiable: bool = False,  # 是否支持微分的布尔值
    capturable: bool = False,  # 是否可捕获的布尔值
    has_complex: bool = False,  # 是否具有复杂性的布尔值
    *,
    lr: float,  # 学习率的浮点数
    alpha: float,  # 移动平均系数的浮点数
    eps: float,  # 为了数值稳定性而添加到分母中的小值的浮点数
    weight_decay: float,  # 权重衰减的浮点数
    momentum: float,  # 动量因子的浮点数
    centered: bool,  # 是否使用中心化的布尔值
):
    r"""Functional API that performs rmsprop algorithm computation.

    See :class:`~torch.optim.RMSProp` for details.
    """
    # 这个检查在编译时较慢，因此我们在此跳过它
    # 如果严格需要，我们可以在 Dynamo 中将此检查添加回去
    if not torch._utils.is_compiling() and not all(
        isinstance(t, torch.Tensor) for t in state_steps
    ):
        raise RuntimeError(
            "API has changed, `state_steps` argument must contain a list of singleton tensors"
        )

    # 如果 foreach 未指定，根据默认设置为融合或分别进行选择
    if foreach is None:
        _, foreach = _default_to_fused_or_foreach(
            params, differentiable, use_fused=False
        )

    # 如果 foreach 为真且当前正在脚本化，则抛出运行时错误
    if foreach and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with foreach optimizers")

    # 根据 foreach 的值选择相应的优化函数
    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_rmsprop
    else:
        func = _single_tensor_rmsprop

    # 调用选定的优化函数来执行 RMSProp 算法
    func(
        params,
        grads,
        square_avgs,
        grad_avgs,
        momentum_buffer_list,
        state_steps,
        lr=lr,
        alpha=alpha,
        eps=eps,
        weight_decay=weight_decay,
        momentum=momentum,
        centered=centered,
        maximize=maximize,
        capturable=capturable,
        differentiable=differentiable,
        has_complex=has_complex,
    )
```