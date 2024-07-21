# `.\pytorch\torch\optim\rprop.py`

```
# mypy: allow-untyped-defs
r"""Implementation for the Resilient backpropagation."""
from typing import List, Optional, Tuple  # 导入必要的类型注解

import torch  # 导入PyTorch库
from torch import Tensor  # 导入Tensor类型
from .optimizer import (  # 导入自定义的优化器相关模块
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

__all__ = ["Rprop", "rprop"]  # 模块公开的类和函数列表


class Rprop(Optimizer):  # 定义Rprop优化器类，继承自Optimizer类
    def __init__(  # 初始化函数
        self,
        params: ParamsT,  # 参数列表
        lr: float = 1e-2,  # 学习率，默认为0.01
        etas: Tuple[float, float] = (0.5, 1.2),  # eta参数元组，默认为(0.5, 1.2)
        step_sizes: Tuple[float, float] = (1e-6, 50),  # 步长尺寸参数元组，默认为(1e-6, 50)
        *,
        capturable: bool = False,  # 是否可捕获的标志，默认为False
        foreach: Optional[bool] = None,  # 是否应用于每个参数的标志，默认为None
        maximize: bool = False,  # 是否最大化优化的标志，默认为False
        differentiable: bool = False,  # 是否可微分的标志，默认为False
    ):  # 参数初始化注释
        if not 0.0 <= lr:  # 检查学习率是否非负
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 < etas[0] < 1.0 < etas[1]:  # 检查eta参数是否有效
            raise ValueError(f"Invalid eta values: {etas[0]}, {etas[1]}")

        defaults = dict(  # 初始化默认参数字典
            lr=lr,
            etas=etas,
            step_sizes=step_sizes,
            foreach=foreach,
            maximize=maximize,
            differentiable=differentiable,
            capturable=capturable,
        )
        super().__init__(params, defaults)  # 调用父类Optimizer的初始化函数

    def __setstate__(self, state):  # 重载__setstate__方法
        super().__setstate__(state)  # 调用父类的__setstate__方法
        for group in self.param_groups:  # 遍历参数组
            group.setdefault("foreach", None)  # 设置默认参数组属性foreach为None
            group.setdefault("maximize", False)  # 设置默认参数组属性maximize为False
            group.setdefault("differentiable", False)  # 设置默认参数组属性differentiable为False
            group.setdefault("capturable", False)  # 设置默认参数组属性capturable为False
            for p in group["params"]:  # 遍历参数组中的每个参数
                p_state = self.state.get(p, [])  # 获取参数p的状态信息，默认为空列表
                if len(p_state) != 0 and not torch.is_tensor(p_state["step"]):  # 如果状态信息不为空且步长值不是Tensor
                    step_val = float(p_state["step"])  # 将步长值转换为浮点数
                    p_state["step"] = (  # 更新步长值为Tensor类型
                        torch.tensor(
                            step_val, dtype=_get_scalar_dtype(), device=p.device
                        ) if group["capturable"]  # 如果capturable为True，则使用指定设备创建Tensor
                        else torch.tensor(step_val, dtype=_get_scalar_dtype())
                    )
    # 初始化一个参数组，将参数、梯度、上一步的状态、步长大小等信息添加到各自的列表中
    def _init_group(self, group, params, grads, prevs, step_sizes, state_steps):
        # 检查该参数组中是否存在复数类型的张量
        has_complex = False
        for p in group["params"]:
            # 如果参数没有梯度，则跳过
            if p.grad is None:
                continue
            # 判断参数是否为复数类型
            has_complex |= torch.is_complex(p)
            # 将参数添加到参数列表中
            params.append(p)
            # 获取参数的梯度
            grad = p.grad
            # 如果梯度是稀疏的，则抛出异常，因为 Rprop 不支持稀疏梯度
            if grad.is_sparse:
                raise RuntimeError("Rprop does not support sparse gradients")

            # 将梯度添加到梯度列表中
            grads.append(grad)
            # 获取参数对应的状态字典
            state = self.state[p]

            # 状态初始化
            if len(state) == 0:
                # 初始化步数为零张量，根据参数是否可捕获而确定设备
                state["step"] = (
                    torch.zeros((), dtype=_get_scalar_dtype(), device=p.device)
                    if group["capturable"]
                    else torch.zeros((), dtype=_get_scalar_dtype())
                )

                # 初始化上一步的值为与参数相同形状的零张量，保留参数的存储格式
                state["prev"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                # 如果参数是复数类型，步长大小初始化为复数形式的学习率
                if p.dtype.is_complex:
                    # 复数应被视为两个独立的实数，因此步长大小的虚部不应为零
                    state["step_size"] = torch.full_like(
                        grad, complex(group["lr"], group["lr"])
                    )
                else:
                    # 否则，步长大小初始化为与梯度相同形状的学习率张量
                    state["step_size"] = torch.full_like(grad, group["lr"])

            # 将上一步的值、步长大小和步数添加到各自的列表中
            prevs.append(state["prev"])
            step_sizes.append(state["step_size"])
            state_steps.append(state["step"])

        # 返回是否存在复数类型参数的布尔值
        return has_complex

    # 应用梯度以用于可微分操作
    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        # 执行单个优化步骤的方法

        self._cuda_graph_capture_health_check()
        # 检查 CUDA 图形捕获健康状态

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                # 如果有闭包函数，启用梯度计算，并重新评估模型，获取损失值

        for group in self.param_groups:
            params: List[Tensor] = []
            grads: List[Tensor] = []
            prevs: List[Tensor] = []
            step_sizes: List[Tensor] = []
            state_steps: List[Tensor] = []

            etaminus, etaplus = group["etas"]
            step_size_min, step_size_max = group["step_sizes"]
            foreach = group["foreach"]
            maximize = group["maximize"]

            has_complex = self._init_group(
                group, params, grads, prevs, step_sizes, state_steps
            )
            # 初始化当前参数组，获取相关参数和状态

            rprop(
                params,
                grads,
                prevs,
                step_sizes,
                state_steps,
                step_size_min=step_size_min,
                step_size_max=step_size_max,
                etaminus=etaminus,
                etaplus=etaplus,
                foreach=foreach,
                maximize=maximize,
                differentiable=group["differentiable"],
                capturable=group["capturable"],
                has_complex=has_complex,
            )
            # 调用 Rprop 算法进行参数优化

        return loss
        # 返回损失值
# 设置 Rprop 类的文档字符串，描述实现 resilient backpropagation 算法的详细信息
Rprop.__doc__ = (
    r"""Implements the resilient backpropagation algorithm.

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \theta_0 \in \mathbf{R}^d \text{ (params)},f(\theta)
                \text{ (objective)},                                                             \\
            &\hspace{13mm}      \eta_{+/-} \text{ (etaplus, etaminus)}, \Gamma_{max/min}
                \text{ (step sizes)}                                                             \\
            &\textbf{initialize} :   g^0_{prev} \leftarrow 0,
                \: \eta_0 \leftarrow \text{lr (learning rate)}                                   \\
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\
            &\hspace{5mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})           \\
            &\hspace{5mm} \textbf{for} \text{  } i = 0, 1, \ldots, d-1 \: \mathbf{do}            \\
            &\hspace{10mm}  \textbf{if} \:   g^i_{prev} g^i_t  > 0                               \\
            &\hspace{15mm}  \eta^i_t \leftarrow \mathrm{min}(\eta^i_{t-1} \eta_{+},
                \Gamma_{max})                                                                    \\
            &\hspace{10mm}  \textbf{else if}  \:  g^i_{prev} g^i_t < 0                           \\
            &\hspace{15mm}  \eta^i_t \leftarrow \mathrm{max}(\eta^i_{t-1} \eta_{-},
                \Gamma_{min})                                                                    \\
            &\hspace{15mm}  g^i_t \leftarrow 0                                                   \\
            &\hspace{10mm}  \textbf{else}  \:                                                    \\
            &\hspace{15mm}  \eta^i_t \leftarrow \eta^i_{t-1}                                     \\
            &\hspace{5mm}\theta_t \leftarrow \theta_{t-1}- \eta_t \mathrm{sign}(g_t)             \\
            &\hspace{5mm}g_{prev} \leftarrow  g_t                                                \\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    For further details regarding the algorithm we refer to the paper
    `A Direct Adaptive Method for Faster Backpropagation Learning: The RPROP Algorithm
    <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.21.1417>`_.
    """
    + rf"""  # 添加原始字符串格式
    Args:
        params (iterable): 用于优化的参数迭代器或定义参数组的字典
        lr (float, optional): 学习率 (默认值: 1e-2)
        etas (Tuple[float, float], optional): 一对 (etaminus, etaplus)，它们是乘法增加和减少因子 (默认值: (0.5, 1.2))
        step_sizes (Tuple[float, float], optional): 一对最小和最大允许步长 (默认值: (1e-6, 50))
        {_foreach_doc}  # 循环中的每个参数更新函数的文档字符串
        {_capturable_doc}  # 可捕获函数中捕获器的文档字符串
        {_maximize_doc}  # 最大化函数的文档字符串
        {_differentiable_doc}  # 可微分函数的文档字符串
# 函数用于单个张量的Rprop算法更新
def _single_tensor_rprop(
    params: List[Tensor],            # 参数列表，包含需要更新的张量
    grads: List[Tensor],             # 梯度列表，对应参数列表中每个张量的梯度
    prevs: List[Tensor],             # 上一步的梯度乘积列表
    step_sizes: List[Tensor],        # 步长大小列表
    state_steps: List[Tensor],       # 状态步骤列表
    *,
    step_size_min: float,            # 步长的最小值
    step_size_max: float,            # 步长的最大值
    etaminus: float,                 # 负梯度乘积的因子
    etaplus: float,                  # 正梯度乘积的因子
    maximize: bool,                  # 是否最大化目标（调整梯度符号）
    capturable: bool,                # 是否支持捕获操作
    differentiable: bool,            # 是否区分可微性
    has_complex: bool,               # 是否包含复数张量
):
    for i, param in enumerate(params):
        grad = grads[i]
        grad = grad if not maximize else -grad  # 根据最大化标志调整梯度的方向
        prev = prevs[i]
        step_size = step_sizes[i]
        step = state_steps[i]

        # 如果不在编译过程中，并且支持捕获操作，则进行设备类型检查
        if not torch._utils.is_compiling() and capturable:
            capturable_supported_devices = _get_capturable_supported_devices()
            assert (
                param.device.type == step.device.type
                and param.device.type in capturable_supported_devices
            ), f"If capturable=True, params and state_steps must be on supported devices: {capturable_supported_devices}."

        step += 1  # 更新步骤计数器

        # 如果参数是复数张量，将其视为实部处理
        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            prev = torch.view_as_real(prev)
            param = torch.view_as_real(param)
            step_size = torch.view_as_real(step_size)

        # 根据是否区分可微性，计算梯度的符号
        if differentiable:
            sign = grad.mul(prev.clone()).sign()
        else:
            sign = grad.mul(prev).sign()

        # 如果支持捕获操作，根据符号调整梯度
        if capturable:
            sign.copy_(torch.where(sign.gt(0), etaplus, sign))
            sign.copy_(torch.where(sign.lt(0), etaminus, sign))
            sign.copy_(torch.where(sign.eq(0), 1, sign))
        else:
            sign[sign.gt(0)] = etaplus
            sign[sign.lt(0)] = etaminus
            sign[sign.eq(0)] = 1

        # 使用符号更新步长，并限制在设定的最小和最大值之间
        step_size.mul_(sign).clamp_(step_size_min, step_size_max)

        # 对于符号为etaminus的情况，将梯度置为0
        grad = grad.clone(memory_format=torch.preserve_format)
        if capturable:
            grad.copy_(torch.where(sign.eq(etaminus), 0, grad))
        else:
            grad[sign.eq(etaminus)] = 0

        # 使用梯度符号更新参数
        param.addcmul_(grad.sign(), step_size, value=-1)
        prev.copy_(grad)  # 更新上一步梯度乘积


# 函数用于多个张量的Rprop算法更新
def _multi_tensor_rprop(
    params: List[Tensor],            # 参数列表，包含需要更新的张量
    grads: List[Tensor],             # 梯度列表，对应参数列表中每个张量的梯度
    prevs: List[Tensor],             # 上一步的梯度乘积列表
    step_sizes: List[Tensor],        # 步长大小列表
    state_steps: List[Tensor],       # 状态步骤列表
    *,
    step_size_min: float,            # 步长的最小值
    step_size_max: float,            # 步长的最大值
    etaminus: float,                 # 负梯度乘积的因子
    etaplus: float,                  # 正梯度乘积的因子
    maximize: bool,                  # 是否最大化目标（调整梯度符号）
    capturable: bool,                # 是否支持捕获操作
    differentiable: bool,            # 是否区分可微性
    has_complex: bool,               # 是否包含复数张量
):
    if len(params) == 0:
        return

    assert not differentiable, "_foreach ops don't support autograd"

    # 如果不在编译过程中，并且支持捕获操作，则进行设备类型检查
    # 编译时，编译器将处理cudagraph检查，详见注释 [torch.compile x capturable]
    # 如果当前没有正在编译的 Torch 代码，并且 capturable=True，则进行以下检查
    if not torch._utils.is_compiling() and capturable:
        # 获取支持捕获的设备列表
        capturable_supported_devices = _get_capturable_supported_devices()
        # 断言所有参数和状态步骤在支持捕获的设备上，并且设备类型一致
        assert all(
            p.device.type == step.device.type
            and p.device.type in capturable_supported_devices
            for p, step in zip(params, state_steps)
        ), f"If capturable=True, params and state_steps must be on supported devices: {capturable_supported_devices}."
    
    # 将多个列表中的张量按设备和数据类型分组
    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, prevs, step_sizes, state_steps]
    )
    for (
        grouped_params,
        grouped_grads,
        grouped_prevs,
        grouped_step_sizes,
        grouped_state_steps,
        # Update steps
        # If steps are on CPU, foreach will fall back to the slow path, which is a for-loop calling t.add(1) over
        # and over. 1 will then be wrapped into a Tensor over and over again, which is slower than if we just
        # wrapped it once now. The alpha is required to assure we go to the right overload.
        # 如果步骤在 CPU 上，则 foreach 将退回到慢速路径，这是一个 for 循环调用 t.add(1) 多次。
        # 1 将会被包装成一个 Tensor，这比我们现在只包装一次要慢。alpha 是必需的，以确保我们使用正确的重载。
        if grouped_state_steps[0].is_cpu:
            torch._foreach_add_(
                grouped_state_steps, torch.tensor(1.0, device="cpu"), alpha=1.0
            )
        else:
            torch._foreach_add_(grouped_state_steps, 1)

        # Handle complex params
        # 处理复杂参数
        if has_complex:
            _view_as_real(
                grouped_params, grouped_grads, grouped_prevs, grouped_step_sizes
            )

        signs = torch._foreach_mul(grouped_grads, grouped_prevs)
        if maximize:
            torch._foreach_neg_(signs)

        # At the end of the step, grouped_prevs will contain the current grads, so we reuse
        # grouped_prevs memory instead of creating a new buffer, but, for clarity, we reassign
        # to keep referring to the buffer as grouped_grads.
        # 在步骤结束时，grouped_prevs 将包含当前的梯度，因此我们重用 grouped_prevs 的内存而不是创建新的缓冲区，
        # 但为了清晰起见，我们重新分配以保持将缓冲区称为 grouped_grads。

        torch._foreach_copy_(grouped_prevs, grouped_grads)
        if maximize:
            torch._foreach_neg_(grouped_prevs)
        grouped_grads = grouped_prevs

        torch._foreach_sign_(signs)

        # If capturable, adjust signs using etaplus, etaminus
        # 如果可捕获，使用 etaplus 和 etaminus 调整符号
        if capturable:
            for sign in signs:
                sign.copy_(torch.where(sign.gt(0), etaplus, sign))
                sign.copy_(torch.where(sign.lt(0), etaminus, sign))
                sign.copy_(torch.where(sign.eq(0), 1, sign))
        else:
            for sign in signs:
                sign[sign.gt(0)] = etaplus
                sign[sign.lt(0)] = etaminus
                sign[sign.eq(0)] = 1

        # Update step sizes with step size updates
        # 使用步长更新更新步长大小
        torch._foreach_mul_(grouped_step_sizes, signs)
        for step_size in grouped_step_sizes:
            step_size.clamp_(step_size_min, step_size_max)

        # Set gradients to zero where signs are etaminus
        # 当符号为 etaminus 时，将梯度设置为零
        grouped_grads = list(grouped_grads)
        for i in range(len(grouped_grads)):
            grouped_grads[i].copy_(
                torch.where(signs[i].eq(etaminus), 0, grouped_grads[i])
            )

        # Explicitly delete signs to save memory as it's not used beyond this point
        # 明确删除 signs 以节省内存，因为它在此后不再使用
        del signs

        # Update parameters using the gradient signs and step sizes
        # 使用梯度符号和步长大小更新参数
        grad_signs = [grad.sign() for grad in grouped_grads]
        torch._foreach_addcmul_(
            grouped_params, grad_signs, grouped_step_sizes, value=-1
        )

        # Note: grouped_prevs has effectively become grouped_grads at this point,
        # though we still refer to it as grouped_prevs for clarity.
        # 注意：在这一点上，grouped_prevs 实际上已经成为 grouped_grads，
        # 尽管为了清晰起见我们仍然称其为 grouped_prevs。
# 使用装饰器禁用 Dynamo 如果不支持的话，并调用 _single_tensor_rprop 函数
@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_rprop)
# 定义 rprop 函数，接受多个参数，执行 rprop 算法的功能 API
def rprop(
    params: List[Tensor],                      # 参数列表，包含张量
    grads: List[Tensor],                       # 梯度列表，包含张量
    prevs: List[Tensor],                       # 前一步参数列表，包含张量
    step_sizes: List[Tensor],                  # 步长大小列表，包含张量
    state_steps: List[Tensor],                 # 状态步数列表，包含张量
    # 由于 torchscript 问题 #70627，带有默认值的关键字参数在函数编译时不被支持
    # 目前暂时将其设置为关键字参数，因为 torch/distributed/optim 中的函数 API 由 torch 编译
    foreach: Optional[bool] = None,            # 可选的 foreach 参数，默认为 None
    capturable: bool = False,                  # 是否可捕获的布尔值，默认为 False
    maximize: bool = False,                    # 是否最大化的布尔值，默认为 False
    differentiable: bool = False,              # 是否可微分的布尔值，默认为 False
    has_complex: bool = False,                 # 是否包含复杂数的布尔值，默认为 False
    *,
    step_size_min: float,                      # 最小步长大小，浮点数
    step_size_max: float,                      # 最大步长大小，浮点数
    etaminus: float,                           # etaminus 参数，浮点数
    etaplus: float,                            # etaplus 参数，浮点数
):
    r"""Functional API that performs rprop algorithm computation.

    See :class:`~torch.optim.Rprop` for details.
    """
    # 如果不是在编译时，且 state_steps 中不全是 torch.Tensor 类型，抛出 RuntimeError
    if not torch._utils.is_compiling() and not all(
        isinstance(t, torch.Tensor) for t in state_steps
    ):
        raise RuntimeError(
            "API has changed, `state_steps` argument must contain a list of singleton tensors"
        )

    # 如果 foreach 为 None，则调用 _default_to_fused_or_foreach 函数获取默认的 foreach 值
    if foreach is None:
        _, foreach = _default_to_fused_or_foreach(
            params, differentiable, use_fused=False
        )

    # 如果 foreach 为 True 且正在进行 torch.jit 编译，则抛出 RuntimeError
    if foreach and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with foreach optimizers")

    # 根据 foreach 的值选择合适的函数进行 rprop 算法的计算
    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_rprop
    else:
        func = _single_tensor_rprop

    # 调用选定的 func 函数执行 rprop 算法，传入所有必要的参数和关键字参数
    func(
        params,
        grads,
        prevs,
        step_sizes,
        state_steps,
        step_size_min=step_size_min,
        step_size_max=step_size_max,
        etaminus=etaminus,
        etaplus=etaplus,
        capturable=capturable,
        maximize=maximize,
        differentiable=differentiable,
        has_complex=has_complex,
    )
```