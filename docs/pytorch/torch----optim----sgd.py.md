# `.\pytorch\torch\optim\sgd.py`

```py
# mypy: allow-untyped-defs
r"""Implementation for Stochastic Gradient Descent optimizer."""
# 导入需要的模块和类型
from typing import List, Optional

import torch
from torch import Tensor
from torch.utils._foreach_utils import _get_fused_kernels_supported_devices
# 导入优化器基类
from .optimizer import (
    _default_to_fused_or_foreach,
    _differentiable_doc,
    _foreach_doc,
    _fused_doc,
    _maximize_doc,
    _use_grad_for_differentiable,
    DeviceDict,
    Optimizer,
)

# 公开的类和函数
__all__ = ["SGD", "sgd"]


class SGD(Optimizer):  # noqa: D101
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov=False,
        *,
        maximize: bool = False,
        foreach: Optional[bool] = None,
        differentiable: bool = False,
        fused: Optional[bool] = None,
    ):  # noqa: D107
        # 检查学习率、动量和权重衰减是否为非负值
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        # 初始化默认参数字典
        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            maximize=maximize,
            foreach=foreach,
            differentiable=differentiable,
            fused=fused,
        )
        # 如果使用 Nesterov 动量，要求必须设置 momentum 并且 dampening 应为 0
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        # 调用父类的初始化方法
        super().__init__(params, defaults)

        # 如果使用 fused 梯度更新
        if fused:
            # 允许步骤支持自动混合精度缩放
            self._step_supports_amp_scaling = True

            # 获取支持 fused 内核的设备列表
            fused_supported_devices = _get_fused_kernels_supported_devices()
            # 检查所有参数组中的参数是否都是浮点数张量，并且设备在支持的设备列表中
            if not all(
                p.device.type in fused_supported_devices and torch.is_floating_point(p)
                for pg in self.param_groups
                for p in pg["params"]
            ):
                raise RuntimeError(
                    "`fused=True` requires all the params to be floating point Tensors of "
                    f"supported devices: {fused_supported_devices}."
                )
            # 如果设置了 differentiable，则抛出错误，因为 fused 不支持 differentiable
            if differentiable:
                raise RuntimeError("`fused` does not support `differentiable`")
            # 如果设置了 foreach，则抛出错误，因为 fused 和 foreach 不能同时为 True
            if foreach:
                raise RuntimeError("`fused` and `foreach` cannot be `True` together.")

    # 设置状态的反序列化方法
    def __setstate__(self, state):  # noqa: D105
        super().__setstate__(state)
        # 确保每个参数组都有默认的 nesterov、maximize、foreach、differentiable 和 fused 属性
        for group in self.param_groups:
            group.setdefault("nesterov", False)
            group.setdefault("maximize", False)
            group.setdefault("foreach", None)
            group.setdefault("differentiable", False)
            group.setdefault("fused", False)
    # 初始化一个参数组的优化过程，收集参数、梯度以及动量缓冲区列表
    def _init_group(self, group, params, grads, momentum_buffer_list):
        # 初始化稀疏梯度标志为 False
        has_sparse_grad = False

        # 遍历该参数组中的每个参数
        for p in group["params"]:
            # 如果参数存在梯度
            if p.grad is not None:
                # 将参数添加到参数列表中
                params.append(p)
                # 将参数的梯度添加到梯度列表中
                grads.append(p.grad)
                # 如果参数的梯度是稀疏的，则将稀疏梯度标志设置为 True
                if p.grad.is_sparse:
                    has_sparse_grad = True

                # 如果参数组中设置了动量，则获取参数对应的动量缓冲区并添加到缓冲区列表中
                if group["momentum"] != 0:
                    state = self.state[p]
                    momentum_buffer_list.append(state.get("momentum_buffer"))

        # 返回是否存在稀疏梯度的标志
        return has_sparse_grad

    # 使用装饰器指定的梯度计算方法进行优化步骤
    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        # 如果提供了 closure 函数，则在启用梯度的情况下执行该函数获取损失值
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # 遍历每个参数组
        for group in self.param_groups:
            # 初始化参数列表、梯度列表和动量缓冲区列表
            params: List[Tensor] = []
            grads: List[Tensor] = []
            momentum_buffer_list: List[Optional[Tensor]] = []

            # 初始化该参数组的优化过程，获取是否存在稀疏梯度的信息
            has_sparse_grad = self._init_group(
                group, params, grads, momentum_buffer_list
            )

            # 调用 SGD 进行参数更新
            sgd(
                params,
                grads,
                momentum_buffer_list,
                weight_decay=group["weight_decay"],
                momentum=group["momentum"],
                lr=group["lr"],
                dampening=group["dampening"],
                nesterov=group["nesterov"],
                maximize=group["maximize"],
                has_sparse_grad=has_sparse_grad,
                foreach=group["foreach"],
                fused=group["fused"],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
            )

            # 如果参数组中设置了动量，则更新状态中的动量缓冲区
            if group["momentum"] != 0:
                # 更新状态中的动量缓冲区
                for p, momentum_buffer in zip(params, momentum_buffer_list):
                    state = self.state[p]
                    state["momentum_buffer"] = momentum_buffer

        # 返回损失值
        return loss
# 将 SGD 类的文档字符串设置为包含随机梯度下降算法的详细说明，包括动量选项

SGD.__doc__ = (
    r"""Implements stochastic gradient descent (optionally with momentum).

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{ (lr)}, \: \theta_0 \text{ (params)}, \: f(\theta)
                \text{ (objective)}, \: \lambda \text{ (weight decay)},                          \\
            &\hspace{13mm} \:\mu \text{ (momentum)}, \:\tau \text{ (dampening)},
            \:\textit{ nesterov,}\:\textit{ maximize}                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{5mm}\textbf{if} \: \lambda \neq 0                                           \\
            &\hspace{10mm} g_t \leftarrow g_t + \lambda  \theta_{t-1}                            \\
            &\hspace{5mm}\textbf{if} \: \mu \neq 0                                               \\
            &\hspace{10mm}\textbf{if} \: t > 1                                                   \\
            &\hspace{15mm} \textbf{b}_t \leftarrow \mu \textbf{b}_{t-1} + (1-\tau) g_t           \\
            &\hspace{10mm}\textbf{else}                                                          \\
            &\hspace{15mm} \textbf{b}_t \leftarrow g_t                                           \\
            &\hspace{10mm}\textbf{if} \: \textit{nesterov}                                       \\
            &\hspace{15mm} g_t \leftarrow g_{t} + \mu \textbf{b}_t                             \\
            &\hspace{10mm}\textbf{else}                                                   \\[-1.ex]
            &\hspace{15mm} g_t  \leftarrow  \textbf{b}_t                                         \\
            &\hspace{5mm}\textbf{if} \: \textit{maximize}                                          \\
            &\hspace{10mm}\theta_t \leftarrow \theta_{t-1} + \gamma g_t                   \\[-1.ex]
            &\hspace{5mm}\textbf{else}                                                    \\[-1.ex]
            &\hspace{10mm}\theta_t \leftarrow \theta_{t-1} - \gamma g_t                   \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    Nesterov momentum is based on the formula from
    `On the importance of initialization and momentum in deep learning`__.
    """
    + rf"""
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)  # 学习率，默认为0.001
        momentum (float, optional): momentum factor (default: 0)  # 动量因子，默认为0
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)  # 权重衰减（L2惩罚项），默认为0
        dampening (float, optional): dampening for momentum (default: 0)  # 动量的阻尼效果，默认为0
        nesterov (bool, optional): enables Nesterov momentum (default: False)  # 是否启用 Nesterov 动量，默认为False
        {_maximize_doc}  # 最大化文档占位符
        {_foreach_doc}  # foreach文档占位符
        {_differentiable_doc}  # 可微分文档占位符
        {_fused_doc}  # 融合文档占位符
    """
    + r"""

    Example:
        >>> # xdoctest: +SKIP
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)  # 创建一个SGD优化器，设置学习率为0.1，动量为0.9
        >>> optimizer.zero_grad()  # 清零梯度
        >>> loss_fn(model(input), target).backward()  # 计算损失函数，反向传播
        >>> optimizer.step()  # 执行优化步骤

    __ http://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf  # 外部链接，指向Hinton的动量论文

    .. note::
        The implementation of SGD with Momentum/Nesterov subtly differs from
        Sutskever et al. and implementations in some other frameworks.

        Considering the specific case of Momentum, the update can be written as

        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + g_{t+1}, \\
                p_{t+1} & = p_{t} - \text{lr} * v_{t+1},
            \end{aligned}

        where :math:`p`, :math:`g`, :math:`v` and :math:`\mu` denote the
        parameters, gradient, velocity, and momentum respectively.

        This is in contrast to Sutskever et al. and
        other frameworks which employ an update of the form

        .. math::
            \begin{aligned}
                v_{t+1} & = \mu * v_{t} + \text{lr} * g_{t+1}, \\
                p_{t+1} & = p_{t} - v_{t+1}.
            \end{aligned}

        The Nesterov version is analogously modified.

        Moreover, the initial value of the momentum buffer is set to the
        gradient value at the first step. This is in contrast to some other
        frameworks that initialize it to all zeros.
    """
# SGD算法的函数定义，用于优化神经网络参数
def sgd(
    params: List[Tensor],  # 参数列表，每个参数是一个张量
    d_p_list: List[Tensor],  # 梯度列表，每个梯度对应一个参数张量
    momentum_buffer_list: List[Optional[Tensor]],  # 动量缓存列表，每个元素可选地是一个张量
    # 由于函数编译时支持的默认值不支持带有默认值的命名参数，因此暂时将其作为关键字参数
    has_sparse_grad: bool = False,  # 是否具有稀疏梯度，默认为False
    foreach: Optional[bool] = None,  # 是否使用foreach优化，默认为None
    fused: Optional[bool] = None,  # 是否使用融合优化，默认为None
    grad_scale: Optional[Tensor] = None,  # 梯度缩放，默认为None
    found_inf: Optional[Tensor] = None,  # 是否发现无穷大，默认为None
    *,
    weight_decay: float,  # 权重衰减参数
    momentum: float,  # 动量参数
    lr: float,  # 学习率
    dampening: float,  # 动量的抑制系数
    nesterov: bool,  # 是否使用Nesterov动量，默认为False
    maximize: bool,  # 是否进行最大化优化，默认为False
):
    r"""Functional API that performs SGD algorithm computation.

    See :class:`~torch.optim.SGD` for details.
    """
    # 如果foreach和fused都没有被用户指定，则按照默认情况处理
    # 注意：在torch.jit.scripting()下，必须明确指出if语句，因为JIT无法处理可选参数或复杂的条件表达式
    if foreach is None and fused is None:
        if not torch.jit.is_scripting():
            # 根据情况选择使用融合优化或foreach优化
            fused, foreach = _default_to_fused_or_foreach(
                params, differentiable=False, use_fused=False
            )
        else:
            foreach = False
            fused = False
    if foreach is None:
        foreach = False
    if fused is None:
        fused = False

    # 在torch.jit.scripting()下，不支持foreach优化和融合优化
    if foreach and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with foreach optimizers")
    if fused and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with fused optimizers")

    # 根据是否使用foreach或融合优化，选择对应的优化函数
    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_sgd
    elif fused and not torch.jit.is_scripting():
        func = _fused_sgd
    else:
        func = _single_tensor_sgd

    # 调用选择的优化函数，执行SGD算法
    func(
        params,
        d_p_list,
        momentum_buffer_list,
        weight_decay=weight_decay,
        momentum=momentum,
        lr=lr,
        dampening=dampening,
        nesterov=nesterov,
        has_sparse_grad=has_sparse_grad,
        maximize=maximize,
        grad_scale=grad_scale,
        found_inf=found_inf,
    )


# 单张量SGD算法实现
def _single_tensor_sgd(
    params: List[Tensor],  # 参数列表，每个参数是一个张量
    grads: List[Tensor],  # 梯度列表，每个梯度对应一个参数张量
    momentum_buffer_list: List[Optional[Tensor]],  # 动量缓存列表，每个元素可选地是一个张量
    grad_scale: Optional[Tensor],  # 梯度缩放，默认为None
    found_inf: Optional[Tensor],  # 是否发现无穷大，默认为None
    *,
    weight_decay: float,  # 权重衰减参数
    momentum: float,  # 动量参数
    lr: float,  # 学习率
    dampening: float,  # 动量的抑制系数
    nesterov: bool,  # 是否使用Nesterov动量，默认为False
    maximize: bool,  # 是否进行最大化优化，默认为False
    has_sparse_grad: bool,  # 是否具有稀疏梯度
):
    assert grad_scale is None and found_inf is None
    # 使用 enumerate 遍历 params 列表，获取索引 i 和对应的参数 param
    for i, param in enumerate(params):
        # 根据 maximize 参数确定是否取相反数，计算梯度 grad
        grad = grads[i] if not maximize else -grads[i]

        # 如果设置了 weight_decay，对梯度 grad 进行 L2 正则化
        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # 如果设置了 momentum，执行动量更新
        if momentum != 0:
            # 获取动量缓存 buf
            buf = momentum_buffer_list[i]

            # 如果 buf 为 None，初始化为 grad 的克隆，并且取消梯度追踪
            if buf is None:
                buf = torch.clone(grad).detach()
                momentum_buffer_list[i] = buf
            else:
                # 否则，根据动量和阻尼系数更新 buf
                buf.mul_(momentum).add_(grad, alpha=1 - dampening)

            # 如果采用 Nesterov 动量，更新梯度 grad
            if nesterov:
                grad = grad.add(buf, alpha=momentum)
            else:
                # 否则，使用 buf 作为梯度 grad
                grad = buf

        # 使用学习率 lr 对参数 param 进行更新
        param.add_(grad, alpha=-lr)
# 定义一个函数，实现多个张量的随机梯度下降 (SGD) 更新
def _multi_tensor_sgd(
    params: List[Tensor],  # 参数列表，包含待更新的张量
    grads: List[Tensor],  # 梯度列表，对应于每个参数张量的梯度
    momentum_buffer_list: List[Optional[Tensor]],  # 动量缓冲列表，与每个参数张量对应
    grad_scale: Optional[Tensor],  # 梯度缩放因子，这里断言为None
    found_inf: Optional[Tensor],  # 是否发现无穷大值的标志，这里断言为None
    *,
    weight_decay: float,  # 权重衰减参数
    momentum: float,  # 动量参数
    lr: float,  # 学习率
    dampening: float,  # 阻尼参数
    nesterov: bool,  # 是否使用 Nesterov 动量
    maximize: bool,  # 是否最大化优化目标（针对优化器设计）
    has_sparse_grad: bool,  # 是否有稀疏梯度
):
    assert grad_scale is None and found_inf is None  # 断言梯度缩放因子和无穷大标志为None

    if len(params) == 0:  # 如果参数列表为空，则直接返回
        return

    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, momentum_buffer_list], with_indices=True  # 使用优化器的方法将参数、梯度和动量缓冲分组，并包含其索引信息
    )
    for (
        device_params,  # 按设备分组后的参数张量列表
        device_grads,  # 按设备分组后的梯度列表
        device_momentum_buffer_list,  # 按设备分组后的动量缓冲列表
        ), indices in grouped_tensors.values():  # 遍历grouped_tensors中的索引和值
        device_has_sparse_grad = has_sparse_grad and any(  # 检查是否存在稀疏梯度
            grad.is_sparse for grad in device_grads
        )

        if maximize:
            device_grads = torch._foreach_neg(device_grads)  # 如果需要最大化，则对梯度取反

        if weight_decay != 0:
            # 如果有权重衰减，则重用已分配的中间内存（device_grads）
            if maximize:
                torch._foreach_add_(device_grads, device_params, alpha=weight_decay)  # 对梯度应用权重衰减
            else:
                device_grads = torch._foreach_add(  # 对梯度应用权重衰减
                    device_grads, device_params, alpha=weight_decay
                )

        if momentum != 0:
            bufs = []

            all_states_with_momentum_buffer = True
            for i in range(len(device_momentum_buffer_list)):
                if device_momentum_buffer_list[i] is None:
                    all_states_with_momentum_buffer = False
                    break
                else:
                    bufs.append(device_momentum_buffer_list[i])

            if all_states_with_momentum_buffer:
                torch._foreach_mul_(bufs, momentum)  # 对动量缓冲区进行乘法操作
                torch._foreach_add_(bufs, device_grads, alpha=1 - dampening)  # 对动量缓冲区和梯度应用衰减操作
            else:
                bufs = []
                for i in range(len(device_momentum_buffer_list)):
                    if device_momentum_buffer_list[i] is None:
                        buf = device_momentum_buffer_list[i] = momentum_buffer_list[  # 创建动量缓冲区并复制梯度
                            indices[i]
                        ] = torch.clone(device_grads[i]).detach()
                    else:
                        buf = device_momentum_buffer_list[i]
                        buf.mul_(momentum).add_(device_grads[i], alpha=1 - dampening)

                    bufs.append(buf)

            if nesterov:
                torch._foreach_add_(device_grads, bufs, alpha=momentum)  # 如果使用Nesterov动量，则对梯度和动量缓冲区进行加法操作
            else:
                device_grads = bufs  # 否则更新梯度为动量缓冲区

        if not device_has_sparse_grad:
            # 处理lr为张量时的内部item()调用
            if isinstance(lr, torch.Tensor) and torch._utils.is_compiling():
                grads_x_lr = torch._foreach_mul(device_grads, -lr)  # 计算梯度与lr的乘积
                torch._foreach_add_(device_params, grads_x_lr)  # 对参数应用乘积结果
            else:
                torch._foreach_add_(device_params, device_grads, alpha=-lr)  # 对参数应用梯度的加权和
        else:
            # foreach API不支持稀疏操作，使用传统的稠密操作
            for i in range(len(device_params)):
                device_params[i].add_(device_grads[i], alpha=-lr)  # 对参数应用稠密梯度的加权和
# 定义了一个函数 `_fused_sgd`，用于实现融合的 SGD（随机梯度下降）优化器操作
def _fused_sgd(
    params: List[Tensor],  # 参数列表，包含需要优化的张量
    grads: List[Tensor],  # 梯度列表，对应每个参数的梯度张量
    momentum_buffer_list: List[Optional[Tensor]],  # 动量缓存列表，用于存储动量的张量
    grad_scale: Optional[Tensor],  # 梯度缩放因子，可选的张量
    found_inf: Optional[Tensor],  # 是否发现无穷大的张量，可选的
    *,
    weight_decay: float,  # 权重衰减参数
    momentum: float,  # 动量参数
    lr: float,  # 学习率
    dampening: float,  # 阻尼系数
    nesterov: bool,  # 是否使用 Nesterov 动量
    maximize: bool,  # 是否最大化优化目标（一般为最小化）
    has_sparse_grad: bool,  # 是否存在稀疏梯度（不支持稀疏梯度）
) -> None:
    # 如果参数列表为空，则直接返回，不执行任何操作
    if not params:
        return
    # 如果存在稀疏梯度，则抛出异常，因为 `_fused_sgd` 不支持稀疏梯度
    if has_sparse_grad:
        raise RuntimeError("`_fused_sgd` does not support sparse gradients")

    # 创建用于存储梯度缩放因子和发现无穷大的设备字典
    grad_scale_dict: DeviceDict = (
        {grad_scale.device: grad_scale} if grad_scale is not None else {}
    )
    found_inf_dict: DeviceDict = (
        {found_inf.device: found_inf} if found_inf is not None else {}
    )

    # 判断是否没有动量缓存
    no_momentum_buffer = momentum == 0
    # 判断是否是第一步（所有动量缓存都为 None 并且不是没有动量缓存的情况）
    is_first_step = (
        all(t is None for t in momentum_buffer_list) and not no_momentum_buffer
    )
    # 如果是第一步，则初始化每个参数的动量缓存
    if is_first_step:
        for i, g in enumerate(grads):
            momentum_buffer_list[i] = torch.empty_like(g)

    # 按设备和数据类型对参数、梯度和动量缓存列表进行分组
    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, momentum_buffer_list], with_indices=False  # type: ignore[list-item]
    )
    # 遍历分组后的张量组合
    for (device, _), (
        (device_params, device_grads, device_momentum_buffer_list),
        _,
    ) in grouped_tensors.items():
        # 初始化设备上的梯度缩放因子和发现无穷大的标记
        device_grad_scale, device_found_inf = None, None
        # 如果梯度缩放因子不为 None，则将其加入到对应设备的字典中
        if grad_scale is not None:
            device_grad_scale = grad_scale_dict.setdefault(
                device, grad_scale.to(device)
            )
        # 如果发现无穷大的标记不为 None，则将其加入到对应设备的字典中
        if found_inf_dict is not None and found_inf is not None:
            device_found_inf = found_inf_dict.setdefault(device, found_inf.to(device))
        
        # 调用底层的融合 SGD 操作
        torch._fused_sgd_(
            device_params,  # 设备上的参数列表
            device_grads,  # 设备上的梯度列表
            [] if no_momentum_buffer else device_momentum_buffer_list,  # 设备上的动量缓存列表（如果没有动量缓存则为空列表）
            weight_decay=weight_decay,  # 权重衰减参数
            momentum=momentum,  # 动量参数
            lr=lr,  # 学习率
            dampening=dampening,  # 阻尼系数
            nesterov=nesterov,  # 是否使用 Nesterov 动量
            maximize=maximize,  # 是否最大化优化目标
            is_first_step=is_first_step,  # 是否第一步
            grad_scale=device_grad_scale,  # 梯度缩放因子
            found_inf=device_found_inf,  # 发现无穷大的标记
        )
```