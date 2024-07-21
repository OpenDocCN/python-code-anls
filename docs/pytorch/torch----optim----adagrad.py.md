# `.\pytorch\torch\optim\adagrad.py`

```py
# mypy: allow-untyped-defs
# 引入类型声明
from typing import List, Optional

# 引入 PyTorch 库
import torch
from torch import Tensor
from torch.utils._foreach_utils import _get_fused_kernels_supported_devices
# 导入自定义优化器模块
from .optimizer import (
    _default_to_fused_or_foreach,
    _differentiable_doc,
    _foreach_doc,
    _get_scalar_dtype,
    _get_value,
    _maximize_doc,
    _use_grad_for_differentiable,
    _view_as_real,
    Optimizer,
    ParamsT,
)

# 导出的类和函数列表
__all__ = ["Adagrad", "adagrad"]


class Adagrad(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-2,  # 学习率，默认为 0.01
        lr_decay: float = 0,  # 学习率衰减，默认为 0
        weight_decay: float = 0,  # 权重衰减，默认为 0
        initial_accumulator_value: float = 0,  # 初始累加器值，默认为 0
        eps: float = 1e-10,  # 用于数值稳定性的小值，默认为 1e-10
        foreach: Optional[bool] = None,  # 是否使用 foreach 操作，默认为 None
        *,
        maximize: bool = False,  # 是否最大化优化目标，默认为 False
        differentiable: bool = False,  # 是否允许不可导操作，默认为 False
        fused: Optional[bool] = None,  # 是否使用融合操作，默认为 None
        ):
            # 检查学习率是否非负，否则引发值错误异常
            if not 0.0 <= lr:
                raise ValueError(f"Invalid learning rate: {lr}")
            # 检查学习率衰减是否非负，否则引发值错误异常
            if not 0.0 <= lr_decay:
                raise ValueError(f"Invalid lr_decay value: {lr_decay}")
            # 检查权重衰减是否非负，否则引发值错误异常
            if not 0.0 <= weight_decay:
                raise ValueError(f"Invalid weight_decay value: {weight_decay}")
            # 检查初始累加器值是否非负，否则引发值错误异常
            if not 0.0 <= initial_accumulator_value:
                raise ValueError(
                    f"Invalid initial_accumulator_value value: {initial_accumulator_value}"
                )
            # 检查 epsilon 是否非负，否则引发值错误异常
            if not 0.0 <= eps:
                raise ValueError(f"Invalid epsilon value: {eps}")

        # 构建默认参数字典
        defaults = dict(
            lr=lr,
            lr_decay=lr_decay,
            eps=eps,
            weight_decay=weight_decay,
            initial_accumulator_value=initial_accumulator_value,
            foreach=foreach,
            maximize=maximize,
            differentiable=differentiable,
            fused=fused,
        )
        # 调用父类的构造方法，传入参数列表和默认参数字典
        super().__init__(params, defaults)

        # 如果启用了 fused
        if fused:
            # 如果同时启用了 differentiable，抛出运行时错误
            if differentiable:
                raise RuntimeError("`fused` does not support `differentiable`")
            # 设置步骤支持 amp 缩放
            self._step_supports_amp_scaling = True
            # 获取支持 fused 内核的设备列表
            fused_supported_devices = _get_fused_kernels_supported_devices()
            # 移除不支持的 CUDA 设备
            fused_supported_devices.remove("cuda")
            # 检查所有参数组中的参数是否符合要求
            if not all(
                p.device.type in fused_supported_devices and torch.is_floating_point(p)
                for pg in self.param_groups
                for p in pg["params"]
            ):
                raise RuntimeError(
                    "`fused=True` requires all the params to be floating point Tensors of "
                    f"supported devices: {fused_supported_devices}."
                )
            # 如果 foreach 同时为真，抛出运行时错误
            if foreach:
                raise RuntimeError("`fused` and `foreach` cannot be `True` together.")

        # 遍历每个参数组
        for group in self.param_groups:
            # 遍历当前参数组的参数
            for p in group["params"]:
                # 获取当前参数的状态字典
                state = self.state[p]
                # 如果使用了 fused
                if group["fused"]:
                    # 初始化步骤计数器为零张量
                    state["step"] = (
                        torch.zeros(
                            (),
                            dtype=_get_scalar_dtype(is_fused=group["fused"]),
                            device=p.device,
                        )
                    )
                else:
                    # 初始化步骤计数器为浮点数零张量
                    state["step"] = torch.tensor(0.0, dtype=_get_scalar_dtype())
                
                # 根据参数类型初始化累加器值
                init_value = (
                    complex(initial_accumulator_value, initial_accumulator_value)
                    if torch.is_complex(p)
                    else initial_accumulator_value
                )
                # 使用初始值填充累加器
                state["sum"] = torch.full_like(
                    p, init_value, memory_format=torch.preserve_format
                )
    # 定义 __setstate__ 方法，用于反序列化对象状态
    def __setstate__(self, state):
        # 调用父类的 __setstate__ 方法，继承并设置对象状态
        super().__setstate__(state)
        
        # 初始化变量 fused，用于存储参数组中的 "fused" 值，默认为 None
        fused = None
        
        # 遍历参数组 self.param_groups
        for group in self.param_groups:
            # 设置参数组中的默认属性值
            group.setdefault("foreach", None)  # 设置 "foreach" 默认值为 None
            group.setdefault("maximize", False)  # 设置 "maximize" 默认值为 False
            group.setdefault("differentiable", False)  # 设置 "differentiable" 默认值为 False
            
            # 获取参数组中的 "fused" 属性值，并赋给 fused 变量
            fused = group.setdefault("fused", None)

        # 获取对象状态中的所有值
        state_values = list(self.state.values())
        
        # 检查第一个状态值是否包含名为 "step" 的张量
        step_is_tensor = (len(state_values) != 0) and torch.is_tensor(
            state_values[0]["step"]
        )
        
        # 如果 step 不是张量，则将所有状态值中的 "step" 转换为张量
        if not step_is_tensor:
            for s in state_values:
                s["step"] = torch.tensor(
                    float(s["step"]), dtype=_get_scalar_dtype(is_fused=fused)
                )

    # 将参数组中的状态 "sum" 与共享内存关联
    def share_memory(self):
        for group in self.param_groups:
            for p in group["params"]:
                # 获取参数 p 对应的状态
                state = self.state[p]
                # 将状态中的 "sum" 与共享内存关联
                state["sum"].share_memory_()

    # 初始化参数组的方法，收集具有梯度的参数及其状态
    def _init_group(self, group, params_with_grad, grads, state_sums, state_steps):
        # 初始化标志位，用于检测稀疏梯度和复数张量
        has_sparse_grad, has_complex = False, False
        
        # 遍历参数组中的参数列表 "params"
        for p in group["params"]:
            # 检查参数 p 是否具有梯度
            if p.grad is not None:
                # 更新稀疏梯度和复数张量的标志位
                has_sparse_grad |= p.grad.is_sparse
                has_complex |= torch.is_complex(p)
                
                # 将具有梯度的参数 p 添加到 params_with_grad 列表中
                params_with_grad.append(p)
                
                # 将参数 p 的梯度添加到 grads 列表中
                grads.append(p.grad)
                
                # 获取参数 p 对应的状态
                state = self.state[p]
                
                # 将状态中的 "sum" 添加到 state_sums 列表中
                state_sums.append(state["sum"])
                
                # 将状态中的 "step" 添加到 state_steps 列表中
                state_steps.append(state["step"])

        # 返回是否具有稀疏梯度和复数张量的标志位
        return has_sparse_grad, has_complex

    # 使用梯度进行不同iable操作的装饰器
    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        # 初始化损失为None
        loss = None

        # 如果有传入闭包函数，则在启用梯度计算的上下文中重新评估模型并计算损失
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # 遍历优化器参数组
        for group in self.param_groups:
            # 初始化存储梯度相关信息的列表
            params_with_grad: List[Tensor] = []
            grads: List[Tensor] = []
            state_sums: List[Tensor] = []
            state_steps: List[Tensor] = []

            # 初始化当前参数组的状态，获取是否有稀疏梯度和复杂梯度的标志
            has_sparse_grad, has_complex = self._init_group(
                group, params_with_grad, grads, state_sums, state_steps
            )

            # 调用Adagrad优化器更新函数
            adagrad(
                params_with_grad,
                grads,
                state_sums,
                state_steps,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                lr_decay=group["lr_decay"],
                eps=group["eps"],
                has_sparse_grad=has_sparse_grad,
                foreach=group["foreach"],
                maximize=group["maximize"],
                differentiable=group["differentiable"],
                has_complex=has_complex,
                fused=group["fused"],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
            )

        # 返回最后计算的损失
        return loss
# 设置 Adagrad 类的文档字符串，详细描述了 Adagrad 算法的实现细节和数学表达式
Adagrad.__doc__ = (
    r"""Implements Adagrad algorithm.

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{ (lr)}, \: \theta_0 \text{ (params)}, \: f(\theta)
                \text{ (objective)}, \: \lambda \text{ (weight decay)},                          \\
            &\hspace{12mm}    \tau \text{ (initial accumulator value)}, \: \eta\text{ (lr decay)}\\
            &\textbf{initialize} :  state\_sum_0 \leftarrow \tau                          \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{5mm} \tilde{\gamma}    \leftarrow \gamma / (1 +(t-1) \eta)                  \\
            &\hspace{5mm} \textbf{if} \: \lambda \neq 0                                          \\
            &\hspace{10mm} g_t \leftarrow g_t + \lambda \theta_{t-1}                             \\
            &\hspace{5mm}state\_sum_t  \leftarrow  state\_sum_{t-1} + g^2_t                      \\
            &\hspace{5mm}\theta_t \leftarrow
                \theta_{t-1}- \tilde{\gamma} \frac{g_t}{\sqrt{state\_sum_t}+\epsilon}            \\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    For further details regarding the algorithm we refer to `Adaptive Subgradient Methods for Online Learning
    and Stochastic Optimization`_.
    """
    + rf"""
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        lr_decay (float, optional): learning rate decay (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        initial_accumulator_value (float, optional): initial value of the
            sum of squares of gradients (default: 0)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-10)
        {_foreach_doc}
        {_maximize_doc}
        {_differentiable_doc}
        fused (bool, optional): whether the fused implementation (CPU only) is used.
            Currently, `torch.float64`, `torch.float32`, `torch.float16`, and `torch.bfloat16`
            are supported. (default: None). Please note that the fused implementations does not
            support sparse or complex gradients.
    """
)
    .. _Adaptive Subgradient Methods for Online Learning and Stochastic
        Optimization: http://jmlr.org/papers/v12/duchi11a.html
# 定义 Adagrad 优化算法的函数接口，用于更新模型参数
def adagrad(
    params: List[Tensor],  # 参数列表，每个元素是一个张量，表示模型的参数
    grads: List[Tensor],   # 梯度列表，每个元素是一个张量，表示对应参数的梯度
    state_sums: List[Tensor],  # 状态和列表，每个元素是一个张量，用于存储参数的状态和信息
    state_steps: List[Tensor],  # 状态步骤列表，每个元素是一个张量，用于更新参数状态的步骤
    fused: Optional[bool] = None,  # 是否使用融合版本的 Adagrad 算法，默认为 None
    grad_scale: Optional[Tensor] = None,  # 梯度缩放因子，默认为 None
    found_inf: Optional[Tensor] = None,  # 是否发现梯度中有无穷值，默认为 None
    # 以下参数使用关键字参数（kwonly）方式传递，不支持函数编译时的默认参数的问题
    has_sparse_grad: bool = False,  # 是否有稀疏梯度，默认为 False
    foreach: Optional[bool] = None,  # 是否使用 foreach 模式，默认为 None
    differentiable: bool = False,  # 是否支持不同iable，默认为 False
    has_complex: bool = False,  # 是否有复杂数据类型，默认为 False
    *,
    lr: float,  # 学习率
    weight_decay: float,  # 权重衰减
    lr_decay: float,  # 学习率衰减
    eps: float,  # 用于数值稳定性的小值
    maximize: bool,  # 是否最大化优化目标
):
    r"""Functional API that performs Adagrad algorithm computation.

    See :class:`~torch.optim.Adagrad` for details.
    """

    if not all(isinstance(t, torch.Tensor) for t in state_steps):
        # 检查 state_steps 是否都是张量类型，否则抛出运行时错误
        raise RuntimeError(
            "API has changed, `state_steps` argument must contain a list of singleton tensors"
        )

    # 根据用户输入的 foreach 和 fused 参数，决定是否使用融合版本或者 foreach 模式
    if fused is None and foreach is None:
        _, foreach = _default_to_fused_or_foreach(
            params, differentiable, use_fused=False
        )

    # 如果 fused 参数未指定，设为 False
    if fused is None:
        fused = False
    # 如果 foreach 参数未指定，设为 False
    if foreach is None:
        foreach = False

    # 如果正在使用 foreach 优化器且当前处于 TorchScript 编译状态，抛出运行时错误
    if foreach and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with foreach optimizers")
    # 如果正在使用融合优化器且当前处于 TorchScript 编译状态，抛出运行时错误
    if fused and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with fused optimizers")

    # 根据使用情况选择对应的优化器函数实现
    if fused and not torch.jit.is_scripting():
        func = _fused_adagrad
    elif foreach and not torch.jit.is_scripting():
        func = _multi_tensor_adagrad
    else:
        func = _single_tensor_adagrad

    # 调用选择的优化器函数，执行 Adagrad 算法
    func(
        params,
        grads,
        state_sums,
        state_steps,
        lr=lr,
        weight_decay=weight_decay,
        lr_decay=lr_decay,
        eps=eps,
        has_sparse_grad=has_sparse_grad,
        maximize=maximize,
        differentiable=differentiable,
        has_complex=has_complex,
        grad_scale=grad_scale,
        found_inf=found_inf,
    )


# 创建稀疏张量的辅助函数，用于处理稀疏梯度
def _make_sparse(grad, grad_indices, values):
    size = grad.size()
    return torch.sparse_coo_tensor(grad_indices, values, size)


# 单张量模式的 Adagrad 实现函数
def _single_tensor_adagrad(
    params: List[Tensor],
    grads: List[Tensor],
    state_sums: List[Tensor],
    state_steps: List[Tensor],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    lr: float,
    weight_decay: float,
    lr_decay: float,
    eps: float,
    has_sparse_grad: bool,
    maximize: bool,
    differentiable: bool,
    has_complex: bool,
):
    # 断言确保 grad_scale 和 found_inf 都为 None
    assert grad_scale is None and found_inf is None
    # 对每个参数 param、梯度 grad、状态和步数 state_sum、步骤数 step_t 进行迭代
    for param, grad, state_sum, step_t in zip(params, grads, state_sums, state_steps):
        # 更新步数 step_t
        step_t += 1
        # 获取步数的实际值
        step = _get_value(step_t)
        # 如果需要最大化，则将梯度取负
        grad = grad if not maximize else -grad

        # 如果有权重衰减，则对梯度进行添加操作
        if weight_decay != 0:
            # 如果梯度是稀疏的，则引发 RuntimeError
            if grad.is_sparse:
                raise RuntimeError(
                    "weight_decay option is not compatible with sparse gradients"
                )
            # 将权重衰减应用到梯度上
            grad = grad.add(param, alpha=weight_decay)

        # 计算当前的学习率 clr
        clr = lr / (1 + (step - 1) * lr_decay)

        # 如果梯度是稀疏的，则处理稀疏情况
        if grad.is_sparse:
            # 使稀疏梯度的更新非线性化，因此索引必须是唯一的
            grad = grad.coalesce()
            grad_indices = grad._indices()
            grad_values = grad._values()

            # 更新 state_sum 以反映稀疏梯度的平方和
            state_sum.add_(_make_sparse(grad, grad_indices, grad_values.pow(2)))
            # 计算标准差 std，并添加一个小的 epsilon 以避免除以零
            std = state_sum.sparse_mask(grad)._values().sqrt_().add_(eps)
            # 更新参数 param，考虑稀疏梯度的影响
            param.add_(
                _make_sparse(grad, grad_indices, grad_values / std), alpha=-clr
            )
        else:
            # 检查参数是否复数类型
            is_complex = torch.is_complex(param)
            if is_complex:
                # 如果参数是复数，将梯度和状态转换为实数视图
                grad = torch.view_as_real(grad)
                state_sum = torch.view_as_real(state_sum)
                param = torch.view_as_real(param)
            # 更新 state_sum，使用梯度的平方乘以系数 1
            state_sum.addcmul_(grad, grad, value=1)
            # 计算标准差 std，根据是否可微分来确定是否需要添加 epsilon
            if differentiable:
                std = state_sum.sqrt() + eps
            else:
                std = state_sum.sqrt().add_(eps)
            # 更新参数 param，考虑标准差 std 的影响
            param.addcdiv_(grad, std, value=-clr)
            # 如果参数是复数类型，则将其转换回复数类型
            if is_complex:
                param = torch.view_as_complex(param)
                state_sum = torch.view_as_complex(state_sum)
def _multi_tensor_adagrad(
    params: List[Tensor],  # 参数列表，包含需要更新的张量
    grads: List[Tensor],  # 梯度列表，对应参数列表的梯度
    state_sums: List[Tensor],  # 状态和列表，用于保存梯度平方和的状态
    state_steps: List[Tensor],  # 状态步数列表，记录参数更新步数的状态
    grad_scale: Optional[Tensor],  # 梯度比例，可选的张量
    found_inf: Optional[Tensor],  # 发现无穷大的标记，可选的张量
    *,
    lr: float,  # 学习率，用于控制参数更新的步长
    weight_decay: float,  # 权重衰减率，控制参数的稀疏性和过拟合
    lr_decay: float,  # 学习率衰减率，控制学习率的变化趋势
    eps: float,  # 用于数值稳定性的小值，防止除零错误
    has_sparse_grad: bool,  # 是否具有稀疏梯度，影响梯度更新的方式
    maximize: bool,  # 是否最大化目标函数，影响参数更新的方向
    differentiable: bool,  # 是否可微分，如果为False，不支持自动求导操作
    has_complex: bool,  # 是否包含复数，影响梯度计算的复杂性
):
    assert not differentiable, "_foreach ops don't support autograd"  # 断言，确保不支持自动求导操作

    assert grad_scale is None and found_inf is None  # 断言，确保梯度比例和无穷大标记为空

    # Foreach functions will throw errors if given empty lists
    if len(params) == 0:  # 如果参数列表为空
        return  # 直接返回，避免处理空列表导致的错误

    grouped_tensorlists = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, state_sums, state_steps]
    )
    for (
        device_params,  # 设备上的参数列表
        device_grads,  # 设备上的梯度列表
        device_state_sums,  # 设备上的状态和列表
        device_state_steps,  # 设备上的状态步数列表
        # 注：此处省略了对分组张量列表中每个元素的注释，因为可以从名称推断其含义
        ), _ in grouped_tensorlists.values():
        # 检查设备梯度是否包含稀疏梯度
        device_has_sparse_grad = has_sparse_grad and any(
            grad.is_sparse for grad in device_grads
        )

        # 如果设备有稀疏梯度，则调用单个张量的Adagrad更新函数
        if device_has_sparse_grad:
            _single_tensor_adagrad(
                device_params,
                device_grads,
                device_state_sums,
                device_state_steps,
                lr=lr,
                weight_decay=weight_decay,
                lr_decay=lr_decay,
                eps=eps,
                has_sparse_grad=True,
                maximize=maximize,
                differentiable=differentiable,
                has_complex=has_complex,
                grad_scale=grad_scale,
                found_inf=found_inf,
            )
            continue

        # 处理复数参数
        if has_complex:
            _view_as_real(device_params, device_grads, device_state_sums)

        # 如果需要最大化，则将设备梯度取反
        if maximize:
            device_grads = torch._foreach_neg(device_grads)  # type: ignore[assignment]

        # 更新步骤
        # 如果步骤在CPU上，则foreach将回退到慢路径，即通过for循环调用t.add(1)。这时1将会被反复包装成一个张量，比我们现在直接包装一次慢。
        if device_state_steps[0].is_cpu:
            torch._foreach_add_(
                device_state_steps, torch.tensor(1.0, device="cpu"), alpha=1.0
            )
        else:
            torch._foreach_add_(device_state_steps, 1)

        # 如果有权重衰减，则使用已经分配给最大化的中间内存（device_grads）重复利用
        if weight_decay != 0:
            if maximize:
                torch._foreach_add_(device_grads, device_params, alpha=weight_decay)
            else:
                device_grads = torch._foreach_add(  # type: ignore[assignment]
                    device_grads, device_params, alpha=weight_decay
                )

        # 计算负的lr/(1 + (步数 - 1) * lr_decay)，用于更新参数
        minus_clr = [
            -lr / (1 + (_get_value(step) - 1) * lr_decay) for step in device_state_steps
        ]

        # 计算标准差并加上eps
        torch._foreach_addcmul_(device_state_sums, device_grads, device_grads, value=1)
        std = torch._foreach_sqrt(device_state_sums)
        torch._foreach_add_(std, eps)

        # 如果有权重衰减或者需要最大化，则重复利用中间内存（device_grads）
        if weight_decay != 0 or maximize:
            torch._foreach_mul_(device_grads, minus_clr)
            numerator = device_grads
        else:
            numerator = torch._foreach_mul(device_grads, minus_clr)  # type: ignore[assignment]

        # 执行参数更新
        torch._foreach_addcdiv_(device_params, numerator, std)
# 定义私有函数 _fused_adagrad，接受多个参数作为输入，并不返回任何值
def _fused_adagrad(
    params: List[Tensor],  # 参数列表，包含张量
    grads: List[Tensor],  # 梯度列表，包含张量
    state_sums: List[Tensor],  # 状态和列表，包含张量
    state_steps: List[Tensor],  # 状态步数列表，包含张量
    grad_scale: Optional[Tensor],  # 可选的梯度缩放张量
    found_inf: Optional[Tensor],  # 可选的发现无穷张量
    *,
    lr: float,  # 学习率，浮点数
    weight_decay: float,  # 权重衰减，浮点数
    lr_decay: float,  # 学习率衰减，浮点数
    eps: float,  # 一个很小的数，用于数值稳定性，浮点数
    has_sparse_grad: bool,  # 是否有稀疏梯度，布尔值
    maximize: bool,  # 是否最大化优化目标，布尔值
    differentiable: bool,  # 是否可微，布尔值
    has_complex: bool,  # 是否有复杂参数，布尔值
) -> None:  # 函数没有返回值

    # 如果参数列表为空，则直接返回，不执行后续代码
    if not params:
        return

    # 如果有稀疏梯度或复杂参数，抛出运行时错误
    if has_sparse_grad or has_complex:
        raise RuntimeError("`fused` does not support sparse grad or complex param")

    # 如果设置了可微参数为 True，抛出运行时错误
    if differentiable:
        raise RuntimeError(
            "adagrad with fused=True does not support differentiable=True"
        )

    # 创建梯度缩放字典，如果梯度缩放不为 None，则以设备为键，张量为值
    grad_scale_dict = (
        {grad_scale.device: grad_scale} if grad_scale is not None else None
    )

    # 创建发现无穷字典，如果发现无穷不为 None，则以设备为键，张量为值
    found_inf_dict = {found_inf.device: found_inf} if found_inf is not None else None

    # 按设备和数据类型对张量进行分组，返回分组后的字典
    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, state_sums, state_steps]
    )

    # 遍历分组后的张量字典
    for (device, _), (
        (
            device_params,
            device_grads,
            device_state_sums,
            device_state_steps,
        ),
        _,
    ) in grouped_tensors.items():
        # 初始化设备梯度缩放和发现无穷为 None
        device_grad_scale, device_found_inf = None, None
        
        # 如果梯度缩放不为 None 并且梯度缩放字典不为 None
        if grad_scale is not None and grad_scale_dict is not None:
            # 如果当前设备不在梯度缩放字典中，则将梯度缩放张量移到当前设备
            if device not in grad_scale_dict:
                grad_scale_dict[device] = grad_scale.to(device, non_blocking=True)  # type: ignore[index]
            device_grad_scale = grad_scale_dict[device]  # type: ignore[index]
        
        # 如果发现无穷不为 None 并且发现无穷字典不为 None
        if found_inf is not None and found_inf_dict is not None:
            # 如果当前设备不在发现无穷字典中，则将发现无穷张量移到当前设备
            if found_inf not in found_inf_dict:
                found_inf_dict[device] = found_inf.to(device, non_blocking=True)  # type: ignore[index]
            device_found_inf = found_inf_dict[device]  # type: ignore[index]
        
        # 对设备状态步数列表中的每个元素加1
        torch._foreach_add_(device_state_steps, 1)
        
        # 调用底层的融合的 Adagrad 算法函数
        torch._fused_adagrad_(
            device_params,
            device_grads,
            device_state_sums,
            device_state_steps,
            lr=lr,
            lr_decay=lr_decay,
            weight_decay=weight_decay,
            eps=eps,
            maximize=maximize,
            grad_scale=device_grad_scale,
            found_inf=device_found_inf,
        )
        
        # 如果发现无穷不为 None，则将其从设备状态步数列表中减去
        if device_found_inf is not None:
            torch._foreach_sub_(
                device_state_steps, [device_found_inf] * len(device_state_steps)
            )
```