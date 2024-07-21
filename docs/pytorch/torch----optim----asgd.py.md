# `.\pytorch\torch\optim\asgd.py`

```
# mypy: allow-untyped-defs
# 导入所需模块和类型声明
from typing import List, Optional, Tuple, Union

# 导入 PyTorch 相关模块和类
import torch
from torch import Tensor

# 从当前目录下的 optimizer 模块导入所需函数和类
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

# 指定导出的类和函数名
__all__ = ["ASGD", "asgd"]


# ASGD 类继承自 Optimizer 类
class ASGD(Optimizer):
    # 初始化方法
    def __init__(
        self,
        params: ParamsT,
        lr: float = 1e-2,
        lambd: float = 1e-4,
        alpha: float = 0.75,
        t0: float = 1e6,
        weight_decay: float = 0,
        foreach: Optional[bool] = None,
        maximize: bool = False,
        differentiable: bool = False,
        capturable: bool = False,
    ):
        # 检查学习率和权重衰减值是否合法
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        # 设定默认参数字典
        defaults = dict(
            lr=lr,
            lambd=lambd,
            alpha=alpha,
            t0=t0,
            weight_decay=weight_decay,
            foreach=foreach,
            maximize=maximize,
            differentiable=differentiable,
            capturable=capturable,
        )
        # 调用父类 Optimizer 的初始化方法
        super().__init__(params, defaults)

    # 设置状态恢复方法
    def __setstate__(self, state):
        # 调用父类 Optimizer 的状态恢复方法
        super().__setstate__(state)
        # 遍历参数组
        for group in self.param_groups:
            # 设置默认值或者获取已有值
            group.setdefault("foreach", None)
            group.setdefault("maximize", False)
            group.setdefault("differentiable", False)
            group.setdefault("capturable", False)
            # 遍历参数张量
            for p in group["params"]:
                # 获取参数状态列表
                p_state = self.state.get(p, [])
                # 如果状态列表不为空
                if len(p_state) != 0:
                    # 如果步数不是张量，则转换为张量类型
                    if not torch.is_tensor(p_state["step"]):
                        step_val = float(p_state["step"])
                        p_state["step"] = torch.tensor(
                            step_val, dtype=_get_scalar_dtype(), device=p.device
                        )
                    # 如果 eta 不是张量，则转换为张量类型
                    if not torch.is_tensor(p_state["eta"]):
                        p_state["eta"] = torch.tensor(
                            p_state["eta"], dtype=_get_scalar_dtype(), device=p.device
                        )
                    # 如果 mu 不是张量，则转换为张量类型
                    if not torch.is_tensor(p_state["mu"]):
                        p_state["mu"] = torch.tensor(
                            p_state["mu"], dtype=_get_scalar_dtype(), device=p.device
                        )
    def _init_group(self, group, params_with_grad, grads, mus, axs, etas, state_steps):
        # 初始化一个标志位，用于检查是否存在复数张量
        has_complex = False
        # 遍历组中的参数列表
        for p in group["params"]:
            # 检查参数是否有梯度
            if p.grad is not None:
                # 检查参数是否为复数类型
                has_complex |= torch.is_complex(p)
                # 将带有梯度的参数添加到列表中
                params_with_grad.append(p)
                # 检查梯度是否为稀疏张量，若是则抛出异常
                if p.grad.is_sparse:
                    raise RuntimeError("ASGD does not support sparse gradients")
                # 将参数的梯度添加到梯度列表中
                grads.append(p.grad)

                # 获取参数的状态字典
                state = self.state[p]
                # 如果状态字典为空，进行初始化
                if len(state) == 0:
                    # 初始化步数状态为零张量
                    state["step"] = torch.zeros(
                        (), device=p.device, dtype=_get_scalar_dtype()
                    )
                    # 初始化学习率因子状态为指定的学习率值
                    state["eta"] = (
                        torch.as_tensor(
                            group["lr"], device=p.device, dtype=_get_scalar_dtype()
                        )
                        .clone()
                        .detach()
                    )
                    # 初始化动量因子状态为单位张量
                    state["mu"] = torch.ones(
                        (), device=p.device, dtype=_get_scalar_dtype()
                    )
                    # 初始化ax状态为与参数p相同格式的零张量
                    state["ax"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                # 将mu、ax、eta、step状态添加到对应的列表中
                mus.append(state["mu"])
                axs.append(state["ax"])
                etas.append(state["eta"])
                state_steps.append(state["step"])
        # 返回是否存在复数张量的标志位
        return has_complex

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """执行单个优化步骤。

        Args:
            closure (Callable, optional): 一个重新评估模型并返回损失的闭包。
        """
        # 执行CUDA图捕获健康检查
        self._cuda_graph_capture_health_check()

        # 初始化损失为None
        loss = None
        # 如果有闭包，则启用梯度计算，并执行闭包以获取损失
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # 遍历所有参数组
        for group in self.param_groups:
            # 初始化参数组相关的列表
            params_with_grad: List[Tensor] = []
            grads: List[Tensor] = []
            mus: List[Tensor] = []
            axs: List[Tensor] = []
            etas: List[Tensor] = []
            state_steps: List[Tensor] = []

            # 初始化组中的参数组和状态，获取是否存在复数张量的标志
            has_complex = self._init_group(
                group, params_with_grad, grads, mus, axs, etas, state_steps
            )

            # 调用ASGD优化函数，传递相关参数
            asgd(
                params_with_grad,
                grads,
                axs,
                mus,
                etas,
                state_steps,
                lambd=group["lambd"],
                lr=group["lr"],
                t0=group["t0"],
                alpha=group["alpha"],
                weight_decay=group["weight_decay"],
                foreach=group["foreach"],
                maximize=group["maximize"],
                differentiable=group["differentiable"],
                capturable=group["capturable"],
                has_complex=has_complex,
            )

        # 返回损失值
        return loss
# 将 ASGD 类的文档字符串设置为描述实现了平均随机梯度下降算法的信息。
# 文档字符串包含参数列表和相关的参数说明。
ASGD.__doc__ = rf"""Implements Averaged Stochastic Gradient Descent.

    It has been proposed in `Acceleration of stochastic approximation by
    averaging`_.

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        lambd (float, optional): decay term (default: 1e-4)
        alpha (float, optional): power for eta update (default: 0.75)
        t0 (float, optional): point at which to start averaging (default: 1e6)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        {_foreach_doc}
        {_maximize_doc}
        {_differentiable_doc}
        {_capturable_doc}

    .. _Acceleration of stochastic approximation by averaging:
        https://dl.acm.org/citation.cfm?id=131098

    """

# 定义了一个名为 _single_tensor_asgd 的函数，实现单个张量的 ASGD 算法。
# 函数参数包括参数列表、梯度列表、轴列表、缩放因子列表、学习率更新步骤列表等。
def _single_tensor_asgd(
    params: List[Tensor],
    grads: List[Tensor],
    axs: List[Tensor],
    mus: List[Tensor],
    etas: List[Tensor],
    state_steps: List[Tensor],
    *,
    lambd: float,
    lr: float,
    t0: float,
    alpha: float,
    weight_decay: float,
    maximize: bool,
    differentiable: bool,
    capturable: bool,
    has_complex: bool,
):
        # 遍历参数列表及其索引
        for i, param in enumerate(params):
            # 获取当前参数对应的梯度
            grad = grads[i]
            # 如果需要最大化，则将梯度取负
            grad = grad if not maximize else -grad
            # 获取当前参数对应的动量值
            mu = mus[i]
            # 获取当前参数对应的动量轴
            ax = axs[i]
            # 获取当前参数对应的学习率衰减因子
            eta = etas[i]
            # 获取当前参数对应的步数状态
            step_t = state_steps[i]

            # 如果不是在编译阶段，并且支持捕获功能
            if not torch._utils.is_compiling() and capturable:
                # 获取支持捕获功能的设备列表
                capturable_supported_devices = _get_capturable_supported_devices()
                # 检查参数及其相关变量是否在支持的设备上
                assert (
                    param.device.type
                    == mu.device.type
                    == eta.device.type
                    == step_t.device.type
                    and param.device.type in capturable_supported_devices
                ), (
                    f"If capturable=True, params, mus, etas, and state_steps must be "
                    f"on supported devices: {capturable_supported_devices}."
                )

            # 如果参数是复数类型，则转换为实部处理
            if torch.is_complex(param):
                grad = torch.view_as_real(grad)
                param = torch.view_as_real(param)
                ax = torch.view_as_real(ax)

            # 更新步数
            step_t += 1

            # 如果有权重衰减，加入到梯度中
            if weight_decay != 0:
                grad = grad.add(param, alpha=weight_decay)

            # 如果支持捕获功能，应用参数更新公式
            if capturable:
                param.mul_(1 - lambd * eta)
                param.addcmul_(grad, eta, value=-1)  # 更新参数
            else:
                eta_value = _get_value(eta)
                param.mul_(1 - lambd * eta_value)  # 衰减项
                param.add_(grad, alpha=-eta_value)  # 更新参数

            # 计算加权平均值
            if capturable or mu.item() != 1:
                ax.add_(param.sub(ax).mul_(mu))
            else:
                ax.copy_(param)

            # 如果支持捕获功能，更新学习率衰减因子和动量值
            if capturable:
                eta.copy_(lr / ((1 + lambd * lr * step_t) ** alpha))
                mu.copy_(1 / torch.maximum(step_t - t0, torch.ones_like(step_t)))
            else:
                step = _get_value(step_t)
                new_eta = torch.as_tensor(lr / ((1 + lambd * lr * step) ** alpha))
                eta.copy_(new_eta)
                new_mu = torch.as_tensor(1 / max(1, step - t0))
                mu.copy_(new_mu)
def _multi_tensor_asgd(
    params: List[Tensor],                    # 参数列表，包含张量
    grads: List[Tensor],                     # 梯度列表，对应参数列表的梯度
    axs: List[Tensor],                       # 轴列表，用于计算参数更新的轴向
    mus: List[Tensor],                       # 动量列表，存储每个参数的动量信息
    etas: List[Tensor],                      # 学习率列表，每个参数的学习率
    state_steps: List[Tensor],               # 状态步骤列表，记录每个参数的状态步数
    *,
    lambd: float,                            # lambda参数，影响正则化的强度
    lr: float,                               # 学习率
    t0: float,                               # 衰减率的初始值
    alpha: float,                            # 稳定的衰减率
    weight_decay: float,                     # 权重衰减参数
    maximize: bool,                          # 是否最大化优化目标
    differentiable: bool,                    # 是否支持自动求导
    capturable: bool,                        # 是否支持捕获
    has_complex: bool,                       # 是否包含复杂结构
):
    if len(params) == 0:                     # 如果参数列表为空，则直接返回
        return

    assert not differentiable, "_foreach ops don't support autograd"  # 断言不支持自动求导，因为 _foreach 操作不支持自动求导

    # If compiling, the compiler will handle cudagraph checks, see note [torch.compile x capturable]
    if not torch._utils.is_compiling() and capturable:
        capturable_supported_devices = _get_capturable_supported_devices(
            supports_xla=False
        )
        assert all(
            p.device.type == mu.device.type == eta.device.type == step.device.type
            and p.device.type in capturable_supported_devices
            for p, mu, eta, step in zip(params, mus, etas, state_steps)
        ), f"If capturable=True, params, mus, etas, and state_steps must be on supported devices: {capturable_supported_devices}."

    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, axs, mus, etas, state_steps]  # 将参数、梯度、轴、动量、学习率、状态步骤分组
    )
    for (device, _), (
        (
            grouped_params,
            grouped_grads,
            grouped_axs,
            grouped_mus,
            grouped_etas,
            grouped_state_steps,
        ),
        _
@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_asgd)
def asgd(
    params: List[Tensor],                    # 参数列表，包含张量
    grads: List[Tensor],                     # 梯度列表，对应参数列表的梯度
    axs: List[Tensor],                       # 轴列表，用于计算参数更新的轴向
    mus: List[Tensor],                       # 动量列表，存储每个参数的动量信息
    etas: List[Tensor],                      # 学习率列表，每个参数的学习率
    state_steps: List[Tensor],               # 状态步骤列表，记录每个参数的状态步数
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    foreach: Optional[bool] = None,          # 可选参数，用于指定是否采用 foreach 操作
    maximize: bool = False,                  # 是否最大化优化目标，默认为 False
    differentiable: bool = False,            # 是否支持自动求导，默认为 False
    capturable: bool = False,                # 是否支持捕获，默认为 False
    has_complex: bool = False,               # 是否包含复杂结构，默认为 False
    *,
    lambd: float,                            # lambda参数，影响正则化的强度
    lr: float,                               # 学习率
    t0: float,                               # 衰减率的初始值
    alpha: float,                            # 稳定的衰减率
    weight_decay: float,                     # 权重衰减参数
):
    r"""Functional API that performs asgd algorithm computation.

    See :class:`~torch.optim.ASGD` for details.
    """
    if foreach is None:
        _, foreach = _default_to_fused_or_foreach(
            params, differentiable, use_fused=False
        )

    if foreach and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with foreach optimizers")

    if foreach and not torch.jit.is_scripting():
        func = _multi_tensor_asgd            # 如果使用 foreach 且未在 torch script 模式下，使用 _multi_tensor_asgd 函数
    else:
        func = _single_tensor_asgd           # 否则使用 _single_tensor_asgd 函数
    # 调用函数 `func`，传入以下参数：
    # - params: 参数列表
    # - grads: 梯度信息
    # - axs: 轴信息
    # - mus: μ 参数
    # - etas: η 参数
    # - state_steps: 状态步长信息
    # - lambd: 可选参数，λ 的值，默认值由变量 lambd 提供
    # - lr: 可选参数，学习率，默认值由变量 lr 提供
    # - t0: 可选参数，t0 的值，默认值由变量 t0 提供
    # - alpha: 可选参数，α 的值，默认值由变量 alpha 提供
    # - weight_decay: 可选参数，权重衰减，默认值由变量 weight_decay 提供
    # - maximize: 可选参数，最大化标志，默认值由变量 maximize 提供
    # - differentiable: 可选参数，可微标志，默认值由变量 differentiable 提供
    # - capturable: 可选参数，可捕获标志，默认值由变量 capturable 提供
    # - has_complex: 可选参数，复杂性标志，默认值由变量 has_complex 提供
```