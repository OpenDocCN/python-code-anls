# `.\pytorch\torch\optim\adamax.py`

```
# mypy: allow-untyped-defs
# 引入类型提示需要的模块
from typing import List, Optional, Tuple, Union

# 引入 PyTorch 框架
import torch
from torch import Tensor

# 从本地模块中导入优化器基类
from .optimizer import (
    _capturable_doc,
    _default_to_fused_or_foreach,
    _differentiable_doc,
    _disable_dynamo_if_unsupported,
    _foreach_doc,
    _get_capturable_supported_devices,
    _get_scalar_dtype,
    _get_value,
    _maximize_doc,
    _use_grad_for_differentiable,
    _view_as_real,
    Optimizer,
    ParamsT,
)

# 指定模块的公开接口
__all__ = ["Adamax", "adamax"]


# Adamax 优化器类，继承自 Optimizer 基类
class Adamax(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float = 2e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0,
        foreach: Optional[bool] = None,
        *,
        maximize: bool = False,
        differentiable: bool = False,
        capturable: bool = False,
    ):
        # 检查学习率参数是否有效
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        # 检查 epsilon 参数是否有效
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        # 检查 beta 参数是否在有效范围内
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        # 检查 weight_decay 参数是否有效
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        # 初始化默认参数字典
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            foreach=foreach,
            maximize=maximize,
            differentiable=differentiable,
            capturable=capturable,
        )
        # 调用父类的构造方法进行初始化
        super().__init__(params, defaults)

    # 用于反序列化对象状态的特殊方法
    def __setstate__(self, state):
        # 调用父类的反序列化方法
        super().__setstate__(state)
        # 对每个参数组设置默认值或根据状态设置步骤的张量表示
        for group in self.param_groups:
            group.setdefault("foreach", None)
            group.setdefault("maximize", False)
            group.setdefault("differentiable", False)
            group.setdefault("capturable", False)
            for p in group["params"]:
                p_state = self.state.get(p, [])
                # 如果状态中的步骤不是张量，则转换为张量表示
                if len(p_state) != 0 and not torch.is_tensor(p_state["step"]):
                    step_val = float(p_state["step"])
                    p_state["step"] = (
                        torch.tensor(
                            step_val, dtype=_get_scalar_dtype(), device=p.device
                        )
                        if group["capturable"]
                        else torch.tensor(step_val, dtype=_get_scalar_dtype())
                    )

    # 初始化参数组的方法
    def _init_group(
        self, group, params_with_grad, grads, exp_avgs, exp_infs, state_steps
    ):
    ):
        # 初始化标志，表示是否存在复数张量
        has_complex = False
        # 遍历优化器参数组中的每个参数
        for p in group["params"]:
            # 如果参数梯度为None，跳过当前参数
            if p.grad is None:
                continue
            # 检查当前参数是否为复数张量，并更新标志
            has_complex |= torch.is_complex(p)
            # 将具有梯度的参数添加到列表中
            params_with_grad.append(p)
            # 如果参数的梯度是稀疏的，抛出运行时错误
            if p.grad.is_sparse:
                raise RuntimeError("Adamax does not support sparse gradients")
            # 将参数的梯度添加到梯度列表中
            grads.append(p.grad)

            # 获取当前参数的状态字典
            state = self.state[p]

            # 如果状态字典为空，进行状态初始化
            if len(state) == 0:
                # 初始化步数
                state["step"] = (
                    torch.zeros((), dtype=_get_scalar_dtype(), device=p.device)
                    if group["capturable"]
                    else torch.tensor(0.0, dtype=_get_scalar_dtype())
                )
                # 初始化一阶矩估计
                state["exp_avg"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )
                # 初始化无穷范数估计
                state["exp_inf"] = torch.zeros_like(
                    p, memory_format=torch.preserve_format
                )

            # 将一阶矩估计、无穷范数估计和步数添加到对应的列表中
            exp_avgs.append(state["exp_avg"])
            exp_infs.append(state["exp_inf"])
            state_steps.append(state["step"])

        # 返回是否存在复数张量的标志
        return has_complex

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        # 检查CUDA图捕获的健康状态
        self._cuda_graph_capture_health_check()

        # 初始化损失值
        loss = None
        # 如果提供了闭包函数，则重新计算模型并获取损失值
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # 遍历优化器参数组
        for group in self.param_groups:
            # 初始化用于存储参数、梯度、一阶矩估计、无穷范数估计和步数的列表
            params_with_grad: List[Tensor] = []
            grads: List[Tensor] = []
            exp_avgs: List[Tensor] = []
            exp_infs: List[Tensor] = []
            state_steps: List[Tensor] = []

            # 获取当前参数组的超参数
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            lr = group["lr"]
            weight_decay = group["weight_decay"]
            foreach = group["foreach"]
            maximize = group["maximize"]
            differentiable = group["differentiable"]
            capturable = group["capturable"]

            # 初始化当前参数组的状态并获取是否存在复数张量的标志
            has_complex = self._init_group(
                group, params_with_grad, grads, exp_avgs, exp_infs, state_steps
            )

            # 执行Adamax优化步骤
            adamax(
                params_with_grad,
                grads,
                exp_avgs,
                exp_infs,
                state_steps,
                eps=eps,
                beta1=beta1,
                beta2=beta2,
                lr=lr,
                weight_decay=weight_decay,
                foreach=foreach,
                maximize=maximize,
                differentiable=differentiable,
                capturable=capturable,
                has_complex=has_complex,
            )

        # 返回损失值
        return loss
# 将 Adamax 算法的文档字符串赋值给 Adamax 类的 __doc__ 属性，用于说明该算法的实现细节和参数含义
Adamax.__doc__ = (
    r"""Implements Adamax algorithm (a variant of Adam based on infinity norm).

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{ (lr)}, \beta_1, \beta_2
                \text{ (betas)},\theta_0 \text{ (params)},f(\theta) \text{ (objective)},
                \: \lambda \text{ (weight decay)},                                                \\
            &\hspace{13mm}    \epsilon \text{ (epsilon)}                                          \\
            &\textbf{initialize} :  m_0 \leftarrow 0 \text{ ( first moment)},
                u_0 \leftarrow 0 \text{ ( infinity norm)}                                 \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{5mm}if \: \lambda \neq 0                                                    \\
            &\hspace{10mm} g_t \leftarrow g_t + \lambda  \theta_{t-1}                            \\
            &\hspace{5mm}m_t      \leftarrow   \beta_1 m_{t-1} + (1 - \beta_1) g_t               \\
            &\hspace{5mm}u_t      \leftarrow   \mathrm{max}(\beta_2 u_{t-1}, |g_{t}|+\epsilon)   \\
            &\hspace{5mm}\theta_t \leftarrow \theta_{t-1} - \frac{\gamma m_t}{(1-\beta^t_1) u_t} \\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    For further details regarding the algorithm we refer to `Adam: A Method for Stochastic Optimization`_.
    """
    + rf"""
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 2e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        {_foreach_doc}
        {_maximize_doc}
        {_differentiable_doc}
        {_capturable_doc}

    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980

    """
)

# 定义一个内部函数 _single_tensor_adamax，用于执行单个张量的 Adamax 算法优化步骤
def _single_tensor_adamax(
    params: List[Tensor],  # 参数列表，包含需要优化的张量
    grads: List[Tensor],   # 梯度列表，包含对应张量的梯度
    exp_avgs: List[Tensor],  # 梯度指数加权移动平均值列表
    exp_infs: List[Tensor],  # 梯度绝对值的指数加权移动平均值列表
    state_steps: List[Tensor],  # 状态步骤列表，记录当前步骤数
    *,
    eps: float,        # 数值稳定性参数 epsilon
    beta1: float,      # 梯度的一阶矩估计的指数衰减率
    beta2: float,      # 梯度的无穷范数的指数衰减率
    lr: float,         # 学习率
    weight_decay: float,
    # 权重衰减（正则化）的系数，通常用来防止模型过拟合，减少参数的值，float 表示浮点数类型

    maximize: bool,
    # 是否最大化目标函数的标志，bool 表示布尔类型，用来控制优化算法的方向

    differentiable: bool,
    # 目标函数是否可微的标志，bool 表示布尔类型，用来指示目标函数是否能够应用梯度下降等算法进行优化

    capturable: bool,
    # 是否允许捕获异常的标志，bool 表示布尔类型，用来决定是否在运行时捕获和处理异常

    has_complex: bool,
    # 是否包含复杂结构的标志，bool 表示布尔类型，用来标识程序中是否存在复杂的数据结构或逻辑
    assert not differentiable, "_foreach ops don't support autograd"
    # 确保不支持自动求导，因为 _foreach 操作不支持自动求导

    if len(params) == 0:
        return
    # 如果参数列表为空，则直接返回

    # If compiling, the compiler will handle cudagraph checks, see note [torch.compile x capturable]
    # 如果在编译中，编译器会处理 cudagraph 检查，参见注释 [torch.compile x capturable]
    # 检查是否处于编译状态且允许捕获，若不是则继续
    if not torch._utils.is_compiling() and capturable:
        # 获取支持的可捕获设备列表（排除XLA支持）
        capturable_supported_devices = _get_capturable_supported_devices(
            supports_xla=False
        )
        # 断言所有参数和状态步骤在支持的设备上，并且设备类型相同
        assert all(
            p.device.type == step.device.type
            and p.device.type in capturable_supported_devices
            for p, step in zip(params, state_steps)
        ), f"If capturable=True, params and state_steps must be on supported devices: {capturable_supported_devices}."

    # 将参数、梯度、指数平均值、指数信息等按设备和数据类型分组
    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, exp_avgs, exp_infs, state_steps]
    )
    for (
        grouped_params,
        grouped_grads,
        grouped_exp_avgs,
        grouped_exp_infs,
        grouped_state_steps,
        ), _ in grouped_tensors.values():
        # 遍历分组张量的值，对每个张量进行操作
        if has_complex:
            # 如果存在复杂情况，则调用视为实部的函数
            _view_as_real(
                grouped_params, grouped_grads, grouped_exp_avgs, grouped_exp_infs
            )

        if maximize:
            # 如果需要最大化，则对梯度进行负数化处理
            grouped_grads = torch._foreach_neg(grouped_grads)  # type: ignore[assignment]

        # 更新步骤
        # 如果步骤在 CPU 上，则 foreach 将退化为慢路径，即使用循环调用 t.add(1)。
        # 1 将会被反复包装成张量，而直接包装一次会更快。alpha 参数确保我们调用正确的重载。
        if grouped_state_steps[0].is_cpu:
            torch._foreach_add_(
                grouped_state_steps, torch.tensor(1.0, device="cpu"), alpha=1.0
            )
        else:
            torch._foreach_add_(grouped_state_steps, 1)

        if weight_decay != 0:
            if maximize:
                # 重复使用为最大化而已经分配的中间内存（grouped_grads）
                torch._foreach_add_(grouped_grads, grouped_params, alpha=weight_decay)
            else:
                grouped_grads = torch._foreach_add(  # type: ignore[assignment]
                    grouped_grads, grouped_params, alpha=weight_decay
                )

        # 更新偏置的第一个矩估计
        torch._foreach_lerp_(grouped_exp_avgs, grouped_grads, 1 - beta1)

        # 更新指数加权的无穷范数
        torch._foreach_mul_(grouped_exp_infs, beta2)

        # 在这种情况下，我们需要引入梯度的一个副本，因为之前没有引入过
        if not maximize and weight_decay == 0:
            grouped_grads = torch._foreach_abs(grouped_grads)  # type: ignore[assignment]
        else:
            torch._foreach_abs_(grouped_grads)

        torch._foreach_add_(grouped_grads, eps)
        torch._foreach_maximum_(grouped_exp_infs, grouped_grads)

        bias_corrections: Union[Tuple[Tensor, ...], List[Tensor]]
        if capturable:
            # 如果可以捕获，则计算偏置修正
            bias_corrections = torch._foreach_pow(beta1, grouped_state_steps)
            # foreach_sub 不允许将标量作为第一个参数
            torch._foreach_sub_(bias_corrections, 1)
            torch._foreach_div_(bias_corrections, lr)

            denom = torch._foreach_mul(grouped_exp_infs, bias_corrections)
            torch._foreach_addcdiv_(grouped_params, grouped_exp_avgs, denom)
        else:
            # 如果无法捕获，则按照备选计算偏置修正和步长
            bias_corrections = [
                1 - beta1 ** _get_value(step) for step in grouped_state_steps
            ]
            step_size = [(_get_value(lr) / bc) * -1 for bc in bias_corrections]
            torch._foreach_addcdiv_(
                grouped_params, grouped_exp_avgs, grouped_exp_infs, step_size
            )
# 禁用 Dynamo 如果不支持，使用单个张量函数 _single_tensor_adamax 来执行
@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_adamax)
# Adamax 算法的函数实现，使用函数式 API 进行计算
def adamax(
    params: List[Tensor],  # 参数列表，每个元素是一个张量
    grads: List[Tensor],   # 梯度列表，每个元素是一个张量
    exp_avgs: List[Tensor],    # 指数加权平均值列表，每个元素是一个张量
    exp_infs: List[Tensor],    # 指数加权无穷范数列表，每个元素是一个张量
    state_steps: List[Tensor],  # 状态步骤列表，每个元素是一个张量
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    foreach: Optional[bool] = None,  # 可选的布尔值参数，默认为 None
    maximize: bool = False,   # 是否进行最大化操作，默认为 False
    differentiable: bool = False,    # 是否可微分，默认为 False
    capturable: bool = False,    # 是否可捕获，默认为 False
    has_complex: bool = False,   # 是否包含复杂数，默认为 False
    *,  # 之后的参数均为关键字参数
    eps: float,     # Adamax 算法的 epsilon 参数
    beta1: float,   # Adamax 算法的 beta1 参数
    beta2: float,   # Adamax 算法的 beta2 参数
    lr: float,      # Adamax 算法的学习率参数
    weight_decay: float,    # 权重衰减参数
):
    r"""Functional API that performs adamax algorithm computation.

    See :class:`~torch.optim.Adamax` for details.
    """

    # 如果不处于编译状态且 state_steps 列表中不全是 torch.Tensor 类型，抛出运行时错误
    if not torch._utils.is_compiling() and not all(
        isinstance(t, torch.Tensor) for t in state_steps
    ):
        raise RuntimeError(
            "API has changed, `state_steps` argument must contain a list of singleton tensors"
        )

    # 如果 foreach 参数为 None，则通过 _default_to_fused_or_foreach 函数设定默认值
    if foreach is None:
        _, foreach = _default_to_fused_or_foreach(
            params, differentiable, use_fused=False
        )

    # 如果 foreach 为 True 且当前处于 torch.jit.script 编译状态，抛出运行时错误
    if foreach and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with foreach optimizers")

    # 如果 foreach 为 True 且不处于 torch.jit.script 编译状态，选择 _multi_tensor_adamax 函数
    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_adamax
    else:
        func = _single_tensor_adamax

    # 调用选择的函数 func 来执行 Adamax 算法计算
    func(
        params,
        grads,
        exp_avgs,
        exp_infs,
        state_steps,
        eps=eps,
        beta1=beta1,
        beta2=beta2,
        lr=lr,
        weight_decay=weight_decay,
        maximize=maximize,
        differentiable=differentiable,
        has_complex=has_complex,
        capturable=capturable,
    )
```