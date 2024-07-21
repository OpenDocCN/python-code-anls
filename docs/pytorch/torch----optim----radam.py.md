# `.\pytorch\torch\optim\radam.py`

```py
# mypy: allow-untyped-defs
r"""Implementation for the RAdam algorithm."""
# 引入必要的模块和类型声明
from typing import cast, List, Optional, Tuple, Union

import torch  # 导入 PyTorch 库
from torch import Tensor  # 导入 Tensor 类型

# 导入优化器基类和相关辅助函数
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
    _use_grad_for_differentiable,
    _view_as_real,
    Optimizer,
    ParamsT,
)

__all__ = ["RAdam", "radam"]  # 模块中公开的类和函数名称列表


class RAdam(Optimizer):  # RAdam 类继承自 Optimizer 类
    def __init__(
        self,
        params: ParamsT,  # 参数列表
        lr: float = 1e-3,  # 学习率，默认为 0.001
        betas: Tuple[float, float] = (0.9, 0.999),  # Adam 中的 beta 参数，默认为 (0.9, 0.999)
        eps: float = 1e-8,  # 用于数值稳定性的 epsilon 值，默认为 1e-8
        weight_decay: float = 0,  # 权重衰减（L2 正则化）的参数，默认为 0
        decoupled_weight_decay: bool = False,  # 是否使用解耦的权重衰减，默认为 False
        *,
        foreach: Optional[bool] = None,  # 是否使用 foreach 操作的标志，可选参数
        maximize: bool = False,  # 是否最大化优化目标，默认为 False
        capturable: bool = False,  # 是否支持捕获优化状态，默认为 False
        differentiable: bool = False,  # 是否可微分，默认为 False
    ):  # 初始化方法
        # 参数有效性检查
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        # 设置默认参数字典
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            maximize=maximize,
            foreach=foreach,
            capturable=capturable,
            decoupled_weight_decay=decoupled_weight_decay,
            differentiable=differentiable,
        )
        # 调用父类的初始化方法
        super().__init__(params, defaults)

    def __setstate__(self, state):  # 对象状态设置方法
        super().__setstate__(state)
        # 针对每个参数组设置默认值
        for group in self.param_groups:
            group.setdefault("foreach", None)
            group.setdefault("maximize", False)
            group.setdefault("differentiable", False)
            group.setdefault("decoupled_weight_decay", False)
            group.setdefault("capturable", False)
            # 遍历参数组中的参数
            for p in group["params"]:
                p_state = self.state.get(p, [])  # 获取参数的状态信息
                # 如果状态信息不为空且步数不是 Tensor 类型，则转换为 Tensor 类型
                if len(p_state) != 0 and not torch.is_tensor(p_state["step"]):
                    step_val = float(p_state["step"])
                    # 根据 capturable 参数决定是否在特定设备上创建 Tensor
                    p_state["step"] = (
                        torch.tensor(
                            step_val, dtype=_get_scalar_dtype(), device=p.device
                        )
                        if group["capturable"]
                        else torch.tensor(step_val, dtype=_get_scalar_dtype())
                    )

    def _init_group(
        self, group, params_with_grad, grads, exp_avgs, exp_avg_sqs, state_steps
    ):
        # 省略部分实现，初始化参数组的方法
        pass
    ):
        has_complex = False
        # 检查参数组中是否存在梯度，并标记是否包含复数类型的张量
        for p in group["params"]:
            if p.grad is not None:
                has_complex |= torch.is_complex(p)
                # 将具有梯度的参数添加到列表中
                params_with_grad.append(p)
                # 如果梯度是稀疏的，抛出运行时错误
                if p.grad.is_sparse:
                    raise RuntimeError("RAdam does not support sparse gradients")
                # 将参数的梯度添加到梯度列表中
                grads.append(p.grad)

                # 获取参数对应的状态信息，进行延迟状态初始化
                state = self.state[p]
                if len(state) == 0:
                    # 初始化步数状态
                    state["step"] = (
                        torch.zeros((), dtype=_get_scalar_dtype(), device=p.device)
                        if group["capturable"]
                        else torch.tensor(0.0, dtype=_get_scalar_dtype())
                    )
                    # 梯度值的指数移动平均
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # 梯度平方值的指数移动平均
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                # 将指数移动平均值和步数状态添加到对应列表中
                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])
                state_steps.append(state["step"])

        # 返回是否存在复数类型张量的标志
        return has_complex

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        # 检查 CUDA 图捕获的健康状态
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            # 启用梯度计算，并执行 closure 函数获取损失值
            with torch.enable_grad():
                loss = closure()

        # 遍历每个参数组
        for group in self.param_groups:
            params_with_grad: List[Tensor] = []
            grads: List[Tensor] = []
            exp_avgs: List[Tensor] = []
            exp_avg_sqs: List[Tensor] = []
            state_steps: List[Tensor] = []
            beta1, beta2 = cast(Tuple[float, float], group["betas"])

            # 初始化参数组，并获取是否存在复数类型张量的标志
            has_complex = self._init_group(
                group, params_with_grad, grads, exp_avgs, exp_avg_sqs, state_steps
            )

            # 调用 RAdam 优化器执行优化步骤
            radam(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                state_steps,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                maximize=group["maximize"],
                foreach=group["foreach"],
                capturable=group["capturable"],
                differentiable=group["differentiable"],
                decoupled_weight_decay=group["decoupled_weight_decay"],
                has_complex=has_complex,
            )

        # 返回损失值
        return loss
# 设置 RAdam 类的文档字符串，描述 RAdam 算法的实现细节和参数选项
RAdam.__doc__ = (
    r"""Implements RAdam algorithm.

    For further details regarding the algorithm we refer to `On the variance of the adaptive learning rate and beyond`_.

    This implementation provides an option to use either the original weight_decay implementation as in Adam
    (where the weight_decay is applied to the gradient) or the one from AdamW (where weight_decay is applied
    to the weight) through the decoupled_weight_decay option. When decoupled_weight_decay is set to False
    (default), it uses the original Adam style weight decay, otherwise, it uses the AdamW style which
    corresponds more closely to the `author's implementation`_ in the RAdam paper. Further information
    about decoupled weight decay can be found in `Decoupled Weight Decay Regularization`_.

    """
    + rf"""
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        decoupled_weight_decay (bool, optional): whether to use decoupled weight
            decay as in AdamW to obtain RAdamW (default: False)
        {_foreach_doc}
        {_maximize_doc}
        {_differentiable_doc}
        {_capturable_doc}

    .. _On the variance of the adaptive learning rate and beyond:
        https://arxiv.org/abs/1908.03265
    .. _author's implementation:
        https://github.com/LiyuanLucasLiu/RAdam
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101

    """
)

# 定义 _single_tensor_radam 函数，实现单个张量的 RAdam 算法优化
def _single_tensor_radam(
    params: List[Tensor],  # 参数列表，包含待优化的张量
    grads: List[Tensor],   # 梯度列表，包含对应参数的梯度张量
    exp_avgs: List[Tensor],  # 梯度一阶矩的指数加权平均列表
    exp_avg_sqs: List[Tensor],  # 梯度二阶矩的指数加权平均列表
    state_steps: List[Tensor],  # 步数计数器列表
    *,
    beta1: float,  # 梯度一阶矩估计的指数衰减率
    beta2: float,  # 梯度二阶矩估计的指数衰减率
    lr: float,     # 学习率
    weight_decay: float,  # 权重衰减（L2正则化）项
    eps: float,    # 为了数值稳定性而添加到分母的项
    decoupled_weight_decay: bool,  # 是否使用解耦的权重衰减方式
    differentiable: bool,  # 是否支持自动微分
    maximize: bool,  # 是否最大化目标函数（通常用于对抗性训练）
    capturable: bool,  # 是否可以捕获图形（编译时的特性）
    has_complex: bool,  # 参数是否包含复数类型
):
# 如果参数列表为空，则直接返回
    if len(params) == 0:
        return

    # 确保不支持自动微分，因为 _foreach 操作不支持自动微分
    assert not differentiable, "_foreach ops don't support autograd"

    # 如果在编译时，编译器会处理 cudagraph 检查，参见注释 [torch.compile x capturable]
    # 检查是否处于非编译状态且支持捕获
    if not torch._utils.is_compiling() and capturable:
        # 获取支持捕获的设备列表（排除支持 XLA 的设备）
        capturable_supported_devices = _get_capturable_supported_devices(
            supports_xla=False
        )
        # 断言所有参数和状态步骤在支持捕获的设备上，并且设备类型一致
        assert all(
            p.device.type == step.device.type
            and p.device.type in capturable_supported_devices
            for p, step in zip(params, state_steps)
        ), f"If capturable=True, params and state_steps must be on supported devices: {capturable_supported_devices}."

    # 将参数、梯度、指数平均值、平方平均值和状态步骤按设备和数据类型分组
    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, exp_avgs, exp_avg_sqs, state_steps]
    )
    # 遍历分组后的参数、梯度、指数平均值、平方平均值和状态步骤
    for (
        grouped_params,
        grouped_grads,
        grouped_exp_avgs,
        grouped_exp_avg_sqs,
        grouped_state_steps,
# 使用装饰器禁用不支持的 Dynamo 优化器，使用单张量函数 _single_tensor_radam
def radam(
    params: List[Tensor],  # 参数列表，包含张量
    grads: List[Tensor],  # 梯度列表，包含张量
    exp_avgs: List[Tensor],  # 指数平均列表，包含张量
    exp_avg_sqs: List[Tensor],  # 平方指数平均列表，包含张量
    state_steps: List[Tensor],  # 状态步骤列表，包含张量

    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    decoupled_weight_decay: bool = False,  # 解耦权重衰减，默认为 False
    foreach: Optional[bool] = None,  # 可选参数 foreach，默认为 None
    differentiable: bool = False,  # 可微分标志，默认为 False
    capturable: bool = False,  # 可捕获标志，默认为 False
    has_complex: bool = False,  # 是否包含复数标志，默认为 False
    maximize: bool = False,  # 最大化标志，默认为 False
    *,  # 之后的参数为强制命名参数
    beta1: float,  # beta1 参数，浮点数
    beta2: float,  # beta2 参数，浮点数
    lr: float,  # 学习率参数，浮点数
    weight_decay: float,  # 权重衰减参数，浮点数
    eps: float,  # eps 参数，浮点数
):
    r"""Functional API that performs RAdam algorithm computation.

    See :class:`~torch.optim.RAdam` for details.
    """
    # 如果 state_steps 中有任何非 Tensor 类型的对象，则抛出运行时错误
    if not all(isinstance(t, torch.Tensor) for t in state_steps):
        raise RuntimeError(
            "API has changed, `state_steps` argument must contain a list of singleton tensors"
        )

    # 如果 foreach 参数为 None，则根据默认设置选择 fused 或 foreach 方法
    if foreach is None:
        _, foreach = _default_to_fused_or_foreach(
            params, differentiable, use_fused=False
        )

    # 如果 foreach 为 True 并且当前为 TorchScript 编译模式，则抛出运行时错误
    if foreach and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with foreach optimizers")

    # 根据 foreach 参数选择相应的优化函数
    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_radam
    else:
        func = _single_tensor_radam

    # 调用选定的优化函数 func 执行 RAdam 算法的计算
    func(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        state_steps,
        beta1=beta1,
        beta2=beta2,
        lr=lr,
        weight_decay=weight_decay,
        eps=eps,
        maximize=maximize,
        decoupled_weight_decay=decoupled_weight_decay,
        differentiable=differentiable,
        capturable=capturable,
        has_complex=has_complex,
    )
```