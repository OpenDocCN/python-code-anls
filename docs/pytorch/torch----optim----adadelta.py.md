# `.\pytorch\torch\optim\adadelta.py`

```py
# mypy: allow-untyped-defs
# 引入必要的类型和函数
from typing import Any, Dict, List, Optional

# 引入PyTorch库
import torch
from torch import Tensor

# 从本地导入优化器相关模块和函数
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

# 公开的类和函数列表
__all__ = ["Adadelta", "adadelta"]


class Adadelta(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float = 1.0,
        rho: float = 0.9,
        eps: float = 1e-6,
        weight_decay: float = 0,
        foreach: Optional[bool] = None,
        *,
        capturable: bool = False,
        maximize: bool = False,
        differentiable: bool = False,
    ):
        # 检查并设置学习率、rho参数、epsilon值、权重衰减的有效性
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= rho <= 1.0:
            raise ValueError(f"Invalid rho value: {rho}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        # 设置默认参数字典
        defaults = dict(
            lr=lr,
            rho=rho,
            eps=eps,
            weight_decay=weight_decay,
            maximize=maximize,
            capturable=capturable,
            foreach=foreach,
            differentiable=differentiable,
        )
        # 调用父类的初始化方法
        super().__init__(params, defaults)

    def __setstate__(self, state):
        # 调用父类的状态设置方法
        super().__setstate__(state)
        # 对每个参数组设置默认值或从状态中恢复
        for group in self.param_groups:
            group.setdefault("foreach", None)
            group.setdefault("maximize", False)
            group.setdefault("differentiable", False)
            group.setdefault("capturable", False)
            for p in group["params"]:
                # 获取参数状态，确保步数是Tensor类型
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
        group: Dict[str, Any],
        params_with_grad: List[Tensor],
        grads: List[Tensor],
        square_avgs: List[Tensor],
        acc_deltas: List[Tensor],
        state_steps: List[Tensor],
        *,
        foreach: Optional[bool] = None,
        maximize: bool = False,
        differentiable: bool = False,
        capturable: bool = False,
    ):
        # 初始化参数组，根据传入参数和状态初始化各项
        pass  # Placeholder, 实际方法功能未在提供的代码段中定义
        ):
            # 初始化一个布尔变量，表示是否存在复数类型的张量
            has_complex = False
            # 声明变量 p，用于迭代 group["params"] 中的张量参数
            p: Tensor
            # 遍历 group["params"] 中的每个参数 p
            for p in group["params"]:
                # 如果当前参数 p 的梯度为 None，则跳过当前循环
                if p.grad is None:
                    continue
                # 检查当前参数 p 是否包含复数类型数据，并更新 has_complex 变量
                has_complex |= torch.is_complex(p)
                # 将当前参数 p 加入 params_with_grad 列表中
                params_with_grad.append(p)
                # 如果当前参数 p 的梯度为稀疏张量，则抛出 RuntimeError 异常
                if p.grad.is_sparse:
                    raise RuntimeError("Adadelta does not support sparse gradients")
                # 将当前参数 p 的梯度添加到 grads 列表中
                grads.append(p.grad)

                # 获取当前参数 p 对应的优化状态信息，如果未初始化则进行初始化
                state = self.state[p]

                # 惰性状态初始化
                if len(state) == 0:
                    # 如果 group["capturable"] 为 True，则在设备 p.device 上创建零张量
                    state["step"] = (
                        torch.zeros((), dtype=_get_scalar_dtype(), device=p.device)
                        if group["capturable"]
                        else torch.zeros((), dtype=_get_scalar_dtype())
                    )

                    # 初始化 state["square_avg"] 和 state["acc_delta"] 为与 p 相同形状的零张量
                    state["square_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["acc_delta"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                # 将 state["square_avg"], state["acc_delta"] 和 state["step"] 添加到相应的列表中
                square_avgs.append(state["square_avg"])
                acc_deltas.append(state["acc_delta"])
                state_steps.append(state["step"])

            # 返回是否存在复数类型的张量的布尔值
            return has_complex

        # 使用 _use_grad_for_differentiable 进行修饰
    def step(self, closure=None):
        """执行一步优化操作。

        Args:
            closure (Callable, optional): 重新评估模型并返回损失的闭包函数。
        """
        # 执行 CUDA 图形捕捉健康检查
        self._cuda_graph_capture_health_check()

        # 初始化损失为 None
        loss = None
        # 如果有闭包函数，则在启用梯度的情况下重新评估损失
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # 遍历每个参数组
        for group in self.param_groups:
            # 初始化各个列表
            params_with_grad: List[Tensor] = []
            grads: List[Tensor] = []
            square_avgs: List[Tensor] = []
            acc_deltas: List[Tensor] = []
            state_steps: List[Tensor] = []

            # 解包参数组的各项值
            (
                lr,
                rho,
                eps,
                weight_decay,
                foreach,
                maximize,
                differentiable,
                capturable,
            ) = (
                group["lr"],
                group["rho"],
                group["eps"],
                group["weight_decay"],
                group["foreach"],
                group["maximize"],
                group["differentiable"],
                group["capturable"],
            )

            # 初始化参数组的状态，收集具有梯度的参数
            has_complex = self._init_group(
                group, params_with_grad, grads, square_avgs, acc_deltas, state_steps
            )

            # 调用 Adadelta 优化器进行优化步骤
            adadelta(
                params_with_grad,
                grads,
                square_avgs,
                acc_deltas,
                state_steps,
                lr=lr,
                rho=rho,
                eps=eps,
                weight_decay=weight_decay,
                foreach=foreach,
                maximize=maximize,
                differentiable=differentiable,
                capturable=capturable,
                has_complex=has_complex,
            )

        # 返回损失值
        return loss
# 将 Adadelta 类的文档字符串设置为实现 Adadelta 算法的详细说明
Adadelta.__doc__ = (
    r"""Implements Adadelta algorithm.

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{ (lr)}, \: \theta_0 \text{ (params)},
                \: f(\theta) \text{ (objective)}, \: \rho \text{ (decay)},
                \: \lambda \text{ (weight decay)}                                                \\
            &\textbf{initialize} :  v_0  \leftarrow 0 \: \text{ (square avg)},
                \: u_0 \leftarrow 0 \: \text{ (accumulate variables)}                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{5mm}if \: \lambda \neq 0                                                    \\
            &\hspace{10mm} g_t \leftarrow g_t + \lambda  \theta_{t-1}                            \\
            &\hspace{5mm} v_t      \leftarrow v_{t-1} \rho + g^2_t (1 - \rho)                    \\
            &\hspace{5mm}\Delta x_t    \leftarrow   \frac{\sqrt{u_{t-1} +
                \epsilon }}{ \sqrt{v_t + \epsilon}  }g_t \hspace{21mm}                           \\
            &\hspace{5mm} u_t  \leftarrow   u_{t-1}  \rho +
                 \Delta x^2_t  (1 - \rho)                                                        \\
            &\hspace{5mm}\theta_t      \leftarrow   \theta_{t-1} - \gamma  \Delta x_t            \\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    For further details regarding the algorithm we refer to `ADADELTA: An Adaptive Learning Rate Method`_.
    """
    + rf"""
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        rho (float, optional): coefficient used for computing a running average
            of squared gradients (default: 0.9). A higher value of `rho` will
            result in a slower average, which can be helpful for preventing
            oscillations in the learning process.
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-6).
        lr (float, optional): coefficient that scale delta before it is applied
            to the parameters (default: 1.0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        {_foreach_doc}
        {_capturable_doc}
        {_maximize_doc}
        {_differentiable_doc}
"""
)
   `
    .. _ADADELTA\: An Adaptive Learning Rate Method:
        https://arxiv.org/abs/1212.5701


Explanation:
)

def _single_tensor_adadelta(
    params: List[Tensor],
    grads: List[Tensor],
    square_avgs: List[Tensor],
    acc_deltas: List[Tensor],
    state_steps: List[Tensor],
    *,
    lr: float,
    rho: float,
    eps: float,
    weight_decay: float,
    maximize: bool,
    differentiable: bool,
    capturable: bool,
    has_complex: bool,
):
    # 如果在编译阶段，并且 capturable=True，编译器将处理 cudagraph 检查，参见注释 [torch.compile x capturable]
    if not torch._utils.is_compiling() and capturable:
        # 获取支持的 capturable 设备列表，排除支持 XLA
        capturable_supported_devices = _get_capturable_supported_devices(
            supports_xla=False
        )
        # 断言所有 params 和 state_steps 的设备类型与支持的 capturable 设备类型一致
        assert all(
            p.device.type == step.device.type
            and p.device.type in capturable_supported_devices
            for p, step in zip(params, state_steps)
        ), f"If capturable=True, params and state_steps must be on supported devices: {capturable_supported_devices}."

    for param, grad, square_avg, acc_delta, step in zip(
        params, grads, square_avgs, acc_deltas, state_steps
    ):
        # 更新步数
        step += 1
        # 如果需要最大化目标，则取相反数
        grad = grad if not maximize else -grad

        # 如果有 weight_decay，则加上对应的惩罚项
        if weight_decay != 0:
            grad = grad.add(param, alpha=weight_decay)

        # 如果 param 是复数类型，则将 square_avg、acc_delta 和 grad 转换为实数视图
        if torch.is_complex(param):
            square_avg = torch.view_as_real(square_avg)
            acc_delta = torch.view_as_real(acc_delta)
            grad = torch.view_as_real(grad)

        # 更新 square_avg
        square_avg.mul_(rho).addcmul_(grad, grad, value=1 - rho)
        # 计算标准差
        std = square_avg.add(eps).sqrt_()
        # 计算 delta
        delta = acc_delta.add(eps).sqrt_()
        if differentiable:
            delta = delta.clone()
        delta.div_(std).mul_(grad)
        # 更新 acc_delta
        acc_delta.mul_(rho).addcmul_(delta, delta, value=1 - rho)

        # 如果 param 是复数类型，则将 delta 转换为复数视图
        if torch.is_complex(param):
            delta = torch.view_as_complex(delta)
        # 更新 param 参数
        param.add_(delta, alpha=-lr)


def _multi_tensor_adadelta(
    params: List[Tensor],
    grads: List[Tensor],
    square_avgs: List[Tensor],
    acc_deltas: List[Tensor],
    state_steps: List[Tensor],
    *,
    lr: float,
    rho: float,
    eps: float,
    weight_decay: float,
    maximize: bool,
    differentiable: bool,
    capturable: bool,
    has_complex: bool,
):
    # 断言 _foreach 操作不支持自动微分
    assert not differentiable, "_foreach ops don't support autograd"

    # 如果在编译阶段，并且 capturable=True，编译器将处理 cudagraph 检查，参见注释 [torch.compile x capturable]
    if not torch._utils.is_compiling() and capturable:
        # 获取支持的 capturable 设备列表，排除支持 XLA
        capturable_supported_devices = _get_capturable_supported_devices(
            supports_xla=False
        )
        # 断言所有 params 和 state_steps 的设备类型与支持的 capturable 设备类型一致
        assert all(
            p.device.type == step.device.type
            and p.device.type in capturable_supported_devices
            for p, step in zip(params, state_steps)
        ), f"If capturable=True, params and state_steps must be on supported devices: {capturable_supported_devices}."

    if len(params) == 0:
        return

    # 根据设备和数据类型对 params、grads、square_avgs、acc_deltas、state_steps 进行分组
    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, square_avgs, acc_deltas, state_steps]
    )
    for (
        device_params,
        device_grads,
        device_square_avgs,
        device_acc_deltas,
        device_state_steps,
    ), _ in grouped_tensors.values():
        # 如果包含复数，则将参数、梯度、平方平均值和累积增量视为实数处理
        if has_complex:
            _view_as_real(
                device_params, device_grads, device_square_avgs, device_acc_deltas
            )

        # 更新步骤
        # 如果步骤在 CPU 上，则 foreach 将退回到较慢的路径，这是一个 for 循环，重复调用 t.add(1)
        # 1 将被反复封装为一个 Tensor，这比现在封装它一次要慢，alpha 是必需的，以确保我们进入正确的重载。
        if device_state_steps[0].is_cpu:
            torch._foreach_add_(
                device_state_steps, torch.tensor(1.0, device="cpu"), alpha=1.0
            )
        else:
            torch._foreach_add_(device_state_steps, 1)

        # 如果进行最大化操作，则对梯度取负
        if maximize:
            device_grads = torch._foreach_neg(device_grads)  # type: ignore[assignment]

        # 如果有权重衰减，则使用已分配给最大化的中间内存（device_grads）
        if weight_decay != 0:
            if maximize:
                torch._foreach_add_(device_grads, device_params, alpha=weight_decay)
            else:
                device_grads = torch._foreach_add(  # type: ignore[assignment]
                    device_grads, device_params, alpha=weight_decay
                )

        # 对平方平均值进行乘法操作
        torch._foreach_mul_(device_square_avgs, rho)
        # 对平方平均值和梯度进行加权相加
        torch._foreach_addcmul_(
            device_square_avgs, device_grads, device_grads, value=1 - rho
        )

        # 对标准差进行加法操作
        std = torch._foreach_add(device_square_avgs, eps)
        # 对标准差进行平方根操作
        torch._foreach_sqrt_(std)

        # 对累积增量进行加法操作
        deltas = torch._foreach_add(device_acc_deltas, eps)
        # 对累积增量进行平方根操作
        torch._foreach_sqrt_(deltas)
        # 对累积增量除以标准差
        torch._foreach_div_(deltas, std)
        # 对累积增量乘以梯度
        torch._foreach_mul_(deltas, device_grads)

        # 对累积增量乘以 rho
        torch._foreach_mul_(device_acc_deltas, rho)
        # 对累积增量和 deltas 进行加权相加
        torch._foreach_addcmul_(device_acc_deltas, deltas, deltas, value=1 - rho)

        # 如果可以捕获并且学习率是 Tensor 类型，则对 deltas 进行乘法操作
        if capturable and isinstance(lr, torch.Tensor):
            torch._foreach_mul_(deltas, -lr)
            # 对参数进行加法操作
            torch._foreach_add_(device_params, deltas)
        else:
            # 否则，对参数进行加法操作，乘以 -lr
            torch._foreach_add_(device_params, deltas, alpha=-lr)
# 使用装饰器禁用 Dynamo 如果不支持，使用 _single_tensor_adadelta 作为单张量函数
@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_adadelta)
# Adadelta 算法的功能性 API，接受多个张量参数和设置参数
def adadelta(
    params: List[Tensor],  # 参数张量列表
    grads: List[Tensor],  # 梯度张量列表
    square_avgs: List[Tensor],  # 平方平均数张量列表
    acc_deltas: List[Tensor],  # 累积增量张量列表
    state_steps: List[Tensor],  # 状态步骤张量列表

    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    capturable: bool = False,  # 可捕获的标志，默认为 False
    foreach: Optional[bool] = None,  # 可选的 foreach 参数，默认为 None
    differentiable: bool = False,  # 可微的标志，默认为 False
    has_complex: bool = False,  # 是否包含复杂数标志，默认为 False
    *,  # 以下参数为强制关键字参数
    lr: float,  # 学习率参数
    rho: float,  # rho 参数
    eps: float,  # eps 参数
    weight_decay: float,  # 权重衰减参数
    maximize: bool,  # 是否最大化标志

):
    r"""Functional API that performs Adadelta algorithm computation.

    See :class:`~torch.optim.Adadelta` for details.
    """

    # this check is slow during compilation, so we skip it
    # if it's strictly needed we can add this check back in dynamo
    # 检查是否在编译时检测到错误的状态步骤参数，如果是，抛出 RuntimeError
    if not torch._utils.is_compiling() and not all(
        isinstance(t, torch.Tensor) for t in state_steps
    ):
        raise RuntimeError(
            "API has changed, `state_steps` argument must contain a list of singleton tensors"
        )

    # We still respect when the user inputs False for foreach.
    # 根据 foreach 参数决定是否使用融合优化器或 foreach 方式执行
    if foreach is None:
        _, foreach = _default_to_fused_or_foreach(
            params, differentiable, use_fused=False
        )

    # 如果使用 foreach 并且在 torch.jit.script 模式下，抛出 RuntimeError
    if foreach and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with foreach optimizers")

    # 根据 foreach 参数选择使用单张量或多张量的 Adadelta 函数实现
    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_adadelta
    else:
        func = _single_tensor_adadelta

    # 调用选定的 Adadelta 函数，执行优化算法
    func(
        params,
        grads,
        square_avgs,
        acc_deltas,
        state_steps,
        lr=lr,
        rho=rho,
        eps=eps,
        weight_decay=weight_decay,
        maximize=maximize,
        differentiable=differentiable,
        capturable=capturable,
        has_complex=has_complex,
    )
```