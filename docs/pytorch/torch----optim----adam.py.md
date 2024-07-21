# `.\pytorch\torch\optim\adam.py`

```py
# 指定不检查未类型化的函数定义
# 从 typing 模块导入所需的类型提示
from typing import List, Optional, Tuple, Union

# 导入 PyTorch 库
import torch
# 从 torch 库中导入 Tensor 类型
from torch import Tensor
# 导入 _get_fused_kernels_supported_devices 函数，用于获取支持的设备
from torch.utils._foreach_utils import _get_fused_kernels_supported_devices
# 从当前包中导入 optimizer 模块
from .optimizer import (
    # 导入 optimizer 模块中的各个函数和类
    _capturable_doc,
    _default_to_fused_or_foreach,
    _differentiable_doc,
    _disable_dynamo_if_unsupported,
    _dispatch_sqrt,
    _foreach_doc,
    _fused_doc,
    _get_capturable_supported_devices,
    _get_scalar_dtype,
    _get_value,
    _maximize_doc,
    _stack_if_compiling,
    _use_grad_for_differentiable,
    _view_as_real,
    DeviceDict,
    Optimizer,
    ParamsT,
)

# 指定可以被外部导入的类和函数列表
__all__ = ["Adam", "adam"]


class Adam(Optimizer):
    # Adam 优化器的初始化方法
    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor] = 1e-3,  # 学习率，默认为 0.001
        betas: Tuple[float, float] = (0.9, 0.999),  # β 参数，默认为 (0.9, 0.999)
        eps: float = 1e-8,  # 用于数值稳定性的小值，默认为 1e-8
        weight_decay: float = 0,  # 权重衰减，默认为 0
        amsgrad: bool = False,  # 是否使用 AMSGrad 算法，默认为 False
        *,
        foreach: Optional[bool] = None,  # 是否使用 foreach 优化，默认为 None
        maximize: bool = False,  # 是否进行最大化优化，默认为 False
        capturable: bool = False,  # 是否可捕获，默认为 False
        differentiable: bool = False,  # 是否可微分，默认为 False
        fused: Optional[bool] = None,  # 是否使用融合优化，默认为 None
        ):
            # 检查学习率是否在合法范围内（大于等于0）
            if not 0.0 <= lr:
                raise ValueError(f"Invalid learning rate: {lr}")
            # 如果学习率是一个张量，并且同时设置了 foreach=True 和 capturable=False，则抛出异常
            if isinstance(lr, Tensor) and foreach and not capturable:
                raise ValueError(
                    "lr as a Tensor is not supported for capturable=False and foreach=True"
                )
            # 检查 epsilon 是否在合法范围内（大于等于0）
            if not 0.0 <= eps:
                raise ValueError(f"Invalid epsilon value: {eps}")
            # 检查 beta 参数列表的第一个参数是否在合法范围内（大于等于0且小于1）
            if not 0.0 <= betas[0] < 1.0:
                raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
            # 检查 beta 参数列表的第二个参数是否在合法范围内（大于等于0且小于1）
            if not 0.0 <= betas[1] < 1.0:
                raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
            # 检查 weight_decay 是否在合法范围内（大于等于0）
            if not 0.0 <= weight_decay:
                raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        # 构建参数的默认字典
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            maximize=maximize,
            foreach=foreach,
            capturable=capturable,
            differentiable=differentiable,
            fused=fused,
        )
        # 调用父类的初始化方法，传入参数和默认字典
        super().__init__(params, defaults)

        # 如果设置了 fused=True，则进一步检查和设置
        if fused:
            # 如果同时设置了 fused=True 和 differentiable=True，则抛出异常
            if differentiable:
                raise RuntimeError("`fused` does not support `differentiable`")
            # 允许使用 AMP 缩放（Automatic Mixed Precision）处理，设置支持 AMP 缩放
            self._step_supports_amp_scaling = True
            # TODO(crcrpar): [low prec params & their higher prec copy]
            # 支持在 FP16/BF16 模型参数上使用 AMP，需要在更高精度下进行参数复制以进行更新数学运算，以减少信息损失。
            # 获取支持融合内核的设备列表
            fused_supported_devices = _get_fused_kernels_supported_devices()
            # 检查所有参数组中的所有参数是否都是浮点数张量，并且位于支持的设备上
            if not all(
                p.device.type in fused_supported_devices and torch.is_floating_point(p)
                for pg in self.param_groups
                for p in pg["params"]
            ):
                raise RuntimeError(
                    "`fused=True` requires all the params to be floating point Tensors of "
                    f"supported devices: {fused_supported_devices}."
                )
            # 如果设置了 foreach=True，则抛出异常，因为 fused 和 foreach 不能同时为 True
            if foreach:
                raise RuntimeError("`fused` and `foreach` cannot be `True` together.")
    # 调用父类的 __setstate__ 方法，恢复模型状态
    def __setstate__(self, state):
        super().__setstate__(state)
        # 遍历优化器的参数组
        for group in self.param_groups:
            # 设置默认参数 'amsgrad' 为 False
            group.setdefault("amsgrad", False)
            # 设置默认参数 'maximize' 为 False
            group.setdefault("maximize", False)
            # 设置默认参数 'foreach' 为 None
            group.setdefault("foreach", None)
            # 设置默认参数 'capturable' 为 False
            group.setdefault("capturable", False)
            # 设置默认参数 'differentiable' 为 False
            group.setdefault("differentiable", False)
            # 获取并设置参数组的 'fused' 值，如果没有则为 None
            fused = group.setdefault("fused", None)
            # 遍历参数组中的参数
            for p in group["params"]:
                # 获取参数 p 的状态信息
                p_state = self.state.get(p, [])
                # 如果 p_state 非空且其 'step' 不是 Tensor 类型，则进行处理
                if len(p_state) != 0 and not torch.is_tensor(p_state["step"]):
                    # 获取 'step' 值，并转换为浮点数
                    step_val = float(p_state["step"])
                    # 根据 'capturable' 或 'fused' 决定是否创建带有特定数据类型和设备的 Tensor
                    p_state["step"] = (
                        torch.tensor(
                            step_val,
                            dtype=_get_scalar_dtype(is_fused=fused),
                            device=p.device,
                        )
                        if group["capturable"] or group["fused"]
                        else torch.tensor(step_val, dtype=_get_scalar_dtype())
                    )

    # 初始化参数组的方法
    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        ):
        # 初始化一个布尔变量，用于判断是否存在复数类型的张量参数
        has_complex = False
        # 遍历参数组中的每个参数
        for p in group["params"]:
            # 如果参数的梯度不为None
            if p.grad is not None:
                # 检查参数是否包含复数类型的数据
                has_complex |= torch.is_complex(p)
                # 将具有梯度的参数添加到列表中
                params_with_grad.append(p)
                # 如果参数的梯度是稀疏的，抛出运行时异常
                if p.grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )
                # 将参数的梯度添加到梯度列表中
                grads.append(p.grad)

                # 获取参数的状态信息
                state = self.state[p]
                # 惰性状态初始化
                if len(state) == 0:
                    # 注意(crcrpar): [特殊设备托管步骤]
                    # 如果capturable和fused都关闭，则故意将`step`托管在CPU上。
                    # 这是因为在CUDA和XLA上进行内核启动是昂贵的。
                    state["step"] = (
                        torch.zeros(
                            (),
                            dtype=_get_scalar_dtype(is_fused=group["fused"]),
                            device=p.device,
                        )
                        if group["capturable"] or group["fused"]
                        else torch.tensor(0.0, dtype=_get_scalar_dtype())
                    )
                    # 参数梯度的指数移动平均值
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # 参数梯度平方的指数移动平均值
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # 如果使用AMSGrad，则维护所有梯度平方指数移动平均值的最大值
                    if group["amsgrad"]:
                        state["max_exp_avg_sq"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )

                # 将参数的指数移动平均值和平方指数移动平均值添加到列表中
                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])

                # 如果使用AMSGrad，则将最大梯度平方指数移动平均值添加到列表中
                if group["amsgrad"]:
                    max_exp_avg_sqs.append(state["max_exp_avg_sq"])
                # 如果在可微模式下，且`step`需要梯度，抛出运行时异常
                if group["differentiable"] and state["step"].requires_grad:
                    raise RuntimeError(
                        "`requires_grad` is not supported for `step` in differentiable mode"
                    )

                # 如果是foreach模式且lr是张量类型，且capturable为False时抛出异常
                if (
                    group["foreach"]
                    and torch.is_tensor(group["lr"])
                    and not group["capturable"]
                ):
                    raise RuntimeError(
                        "lr as a Tensor is not supported for capturable=False and foreach=True"
                    )

                # 将参数状态中的步数添加到列表中
                state_steps.append(state["step"])
        # 返回是否存在复数类型参数的布尔值
        return has_complex

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        # 执行单步优化操作

        # 检查 CUDA 图捕获的健康状态
        self._cuda_graph_capture_health_check()

        # 初始化 loss 变量为 None
        loss = None
        # 如果提供了 closure 函数，则在上下文中启用梯度计算，并执行 closure 函数获取损失值
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # 遍历每个参数组
        for group in self.param_groups:
            # 初始化存储参数梯度、梯度、一阶矩估计、二阶矩估计等列表
            params_with_grad: List[Tensor] = []
            grads: List[Tensor] = []
            exp_avgs: List[Tensor] = []
            exp_avg_sqs: List[Tensor] = []
            max_exp_avg_sqs: List[Tensor] = []
            state_steps: List[Tensor] = []
            beta1, beta2 = group["betas"]

            # 初始化参数组并检查是否存在复数类型的参数
            has_complex = self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
            )

            # 调用优化函数 Adam 进行参数更新
            adam(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=group["amsgrad"],
                has_complex=has_complex,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                maximize=group["maximize"],
                foreach=group["foreach"],
                capturable=group["capturable"],
                differentiable=group["differentiable"],
                fused=group["fused"],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
            )

        # 返回损失值
        return loss
# 设置 Adam 类的文档字符串，描述其实现了 Adam 算法。
Adam.__doc__ = (
    r"""Implements Adam algorithm.
    # 定义 Adam 优化算法的函数
    def adam_optimizer(lr, betas, params, objective, weight_decay, amsgrad, maximize):
        # 初始化动量 m 和二阶矩 v
        m_0 <- 0  # 第一时刻的动量
        v_0 <- 0  # 第一时刻的二阶矩
        \widehat{v_0}^{max} <- 0  # 最大的二阶矩估计
        
        # 进入主循环
        for t=1 to ... do
            # 根据 maximize 参数确定梯度方向
            if maximize:
                g_t <- -\nabla_{\theta} f_t (\theta_{t-1})
            else:
                g_t <- \nabla_{\theta} f_t (\theta_{t-1})
            
            # 如果有 weight_decay，则添加正则化项
            if \lambda \neq 0
                g_t <- g_t + \lambda \theta_{t-1}
            
            # 更新动量 m_t 和二阶矩 v_t
            m_t <- \beta_1 m_{t-1} + (1 - \beta_1) g_t
            v_t <- \beta_2 v_{t-1} + (1-\beta_2) g^2_t
            
            # 计算偏差修正后的一阶矩估计 \widehat{m_t} 和二阶矩估计 \widehat{v_t}
            \widehat{m_t} <- m_t / (1-\beta_1^t)
            \widehat{v_t} <- v_t / (1-\beta_2^t)
            
            # 如果使用 amsgrad，则更新最大的二阶矩估计 \widehat{v_t}^{max}
            if amsgrad
                \widehat{v_t}^{max} <- \mathrm{max}(\widehat{v_t}^{max}, \widehat{v_t})
            
            # 根据是否使用 amsgrad 更新参数 \theta_t
            if amsgrad
                \theta_t <- \theta_{t-1} - \gamma \widehat{m_t} / (\sqrt{\widehat{v_t}^{max}} + \epsilon)
            else
                \theta_t <- \theta_{t-1} - \gamma \widehat{m_t} / (\sqrt{\widehat{v_t}} + \epsilon)
        
        # 返回优化后的参数 \theta_t
        return \theta_t
    """
    For further details regarding the algorithm we refer to `Adam: A Method for Stochastic Optimization`_.
    """
    
    # 添加额外的文档字符串，用于描述 Adam 优化算法的详细信息和参数选项
    + rf"""
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, Tensor, optional): learning rate (default: 1e-3). A tensor LR
            is not yet supported for all our implementations. Please use a float
            LR if you are not also specifying fused=True or capturable=True.
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (bool, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
        {_foreach_doc}
        {_maximize_doc}
        {_capturable_doc}
        {_differentiable_doc}
        {_fused_doc}
    .. Note::
        A prototype implementation of Adam and AdamW for MPS supports `torch.float32` and `torch.float16`.
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    
    """
def _single_tensor_adam(
    params: List[Tensor],               # 参数列表，包含模型参数张量
    grads: List[Tensor],                # 梯度列表，包含参数张量对应的梯度张量
    exp_avgs: List[Tensor],             # 指数加权平均值列表，用于优化算法中的动量项
    exp_avg_sqs: List[Tensor],          # 指数加权平方平均值列表，用于优化算法中的RMSprop项
    max_exp_avg_sqs: List[Tensor],      # 最大的指数加权平方平均值列表，用于AMSGrad优化算法
    state_steps: List[Tensor],          # 状态步数列表，用于跟踪优化步骤
    grad_scale: Optional[Tensor],       # 梯度缩放因子，一般为None
    found_inf: Optional[Tensor],        # 发现的无穷值，一般为None
    *,
    amsgrad: bool,                      # 是否使用AMSGrad优化算法
    has_complex: bool,                  # 是否具有复数类型，这是为了保持一致性
    beta1: float,                       # Adam优化算法的动量参数
    beta2: float,                       # Adam优化算法的RMSprop参数
    lr: Union[float, Tensor],           # 学习率，可以是浮点数或张量
    weight_decay: float,                # 权重衰减（L2正则化）参数
    eps: float,                         # 用于数值稳定性的小常数
    maximize: bool,                     # 是否最大化优化目标
    capturable: bool,                   # 是否支持捕获运算（capturable）
    differentiable: bool,               # 是否支持自动微分
):
    assert grad_scale is None and found_inf is None  # 断言确保 grad_scale 和 found_inf 均为None

    if torch.jit.is_scripting():
        # 由于Torch JIT在处理下面的操作时有重载，能够处理浮点数和张量学习率，因此这里断言它是浮点数，因为大多数使用JIT的人都使用浮点数
        assert isinstance(lr, float)



def _multi_tensor_adam(
    params: List[Tensor],               # 参数列表，包含模型参数张量
    grads: List[Tensor],                # 梯度列表，包含参数张量对应的梯度张量
    exp_avgs: List[Tensor],             # 指数加权平均值列表，用于优化算法中的动量项
    exp_avg_sqs: List[Tensor],          # 指数加权平方平均值列表，用于优化算法中的RMSprop项
    max_exp_avg_sqs: List[Tensor],      # 最大的指数加权平方平均值列表，用于AMSGrad优化算法
    state_steps: List[Tensor],          # 状态步数列表，用于跟踪优化步骤
    grad_scale: Optional[Tensor],       # 梯度缩放因子，一般为None
    found_inf: Optional[Tensor],        # 发现的无穷值，一般为None
    *,
    amsgrad: bool,                      # 是否使用AMSGrad优化算法
    has_complex: bool,                  # 是否具有复数类型，这是为了保持一致性
    beta1: float,                       # Adam优化算法的动量参数
    beta2: float,                       # Adam优化算法的RMSprop参数
    lr: Union[float, Tensor],           # 学习率，可以是浮点数或张量
    weight_decay: float,                # 权重衰减（L2正则化）参数
    eps: float,                         # 用于数值稳定性的小常数
    maximize: bool,                     # 是否最大化优化目标
    capturable: bool,                   # 是否支持捕获运算（capturable）
    differentiable: bool,               # 是否支持自动微分
):
    if len(params) == 0:
        return

    if isinstance(lr, Tensor) and not capturable:
        raise RuntimeError(
            "lr as a Tensor is not supported for capturable=False and foreach=True"
        )

    # 如果正在编译，编译器将处理cudagraph检查，参见说明 [torch.compile x capturable]
    if not torch._utils.is_compiling() and capturable:
        capturable_supported_devices = _get_capturable_supported_devices(
            supports_xla=False
        )
        # 断言所有参数和状态步数位于支持的设备上，如果capturable=True，则必须在支持的设备上
        assert all(
            p.device.type == step.device.type
            and p.device.type in capturable_supported_devices
            for p, step in zip(params, state_steps)
        ), f"If capturable=True, params and state_steps must be on supported devices: {capturable_supported_devices}."

    assert grad_scale is None and found_inf is None  # 断言确保 grad_scale 和 found_inf 均为None

    assert not differentiable, "_foreach ops don't support autograd"

    # 将张量按设备和数据类型分组，以便优化器处理
    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps]
    )
    for (
        device_params,
        device_grads,
        device_exp_avgs,
        device_exp_avg_sqs,
        device_max_exp_avg_sqs,
        device_state_steps,



def _fused_adam(
    params: List[Tensor],               # 参数列表，包含模型参数张量
    grads: List[Tensor],                # 梯度列表，包含参数张量对应的梯度张量
    exp_avgs: List[Tensor],             # 指数加权平均值列表，用于优化算法中的动量项
    exp_avg_sqs: List[Tensor],          # 指数加权平方平均值列表，用于优化算法中的RMSprop项
    max_exp_avg_sqs: List[Tensor],      # 最大的指数加权平方平均值列表，用于AMSGrad优化算法
    state_steps: List[Tensor],          # 状态步数列表，用于跟踪优化步骤
    grad_scale: Optional[Tensor],       # 梯度缩放因子，一般为None
    found_inf: Optional[Tensor],        # 发现的无穷值，一般为None
    *,
    amsgrad: bool,                      # 是否使用AMSGrad优化算法
    has_complex: bool,                  # 是否具有复数类型，这是为了保持一致性
    beta1: float,                       # Adam优化算法的动量参数
    beta2: float,                       # Adam优化算法的RMSprop参数
    lr: Union[float, Tensor],           # 学习率，可以是浮点数或张量
):
    weight_decay: float,  # 权重衰减参数，用于控制模型训练时的正则化
    eps: float,  # 用于控制优化算法的数值稳定性
    maximize: bool,  # 是否最大化目标函数，用于优化算法
    capturable: bool,  # 是否可捕获，用于保持一致性
    differentiable: bool,  # 是否可微分，用于指示目标函数是否可微分
def adam(
    params: List[Tensor],
    grads: List[Tensor],
    exp_avgs: List[Tensor],
    exp_avg_sqs: List[Tensor],
    max_exp_avg_sqs: List[Tensor],
    state_steps: List[Tensor],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    # 定义了 Adam 优化器的函数，接受多个张量列表作为参数

) -> None:
    # 如果 params 列表为空，则直接返回，不进行任何操作
    if not params:
        return
    # 如果 differentiable 参数为 True，则抛出 RuntimeError 异常
    if differentiable:
        raise RuntimeError("Adam with fused=True does not support differentiable=True")

    # 创建一个字典 grad_scale_dict，用于存储梯度缩放因子，按设备分类存储
    grad_scale_dict: DeviceDict = (
        {grad_scale.device: grad_scale} if grad_scale is not None else {}
    )
    # 创建一个字典 found_inf_dict，用于存储是否发现无穷大值的标志，按设备分类存储
    found_inf_dict: DeviceDict = (
        {found_inf.device: found_inf} if found_inf is not None else {}
    )

    # 只有当 lr 是 Tensor 类型并且在 CUDA 上时，才将其作为字典 lr_dict 的一部分，否则为 None
    lr_dict: Optional[DeviceDict] = (
        {lr.device: lr} if isinstance(lr, Tensor) and str(lr.device) != "cpu" else None
    )
    
    # 将 params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps 六个列表根据设备和数据类型进行分组
    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps]
    )
    
    # 遍历 grouped_tensors 中的每个设备及其对应的张量列表
    for (device, _), (
        (
            device_params,
            device_grads,
            device_exp_avgs,
            device_exp_avg_sqs,
            device_max_exp_avg_sqs,
            device_state_steps,
        ),
        _,
    ) in grouped_tensors.items():
        # 如果设备类型为 "mps"，则进行以下断言和检查
        if device.type == "mps":  # type: ignore[union-attr]
            assert found_inf is None and grad_scale is None
            assert not isinstance(lr, Tensor)

        # 初始化设备相关的 grad_scale 和 found_inf 为 None
        device_grad_scale, device_found_inf = None, None
        # 如果 grad_scale 不为 None，则将其加入 grad_scale_dict 中，如果已存在则更新
        if grad_scale is not None:
            device_grad_scale = grad_scale_dict.setdefault(
                device, grad_scale.to(device, non_blocking=True)
            )
        # 如果 found_inf 不为 None，则将其加入 found_inf_dict 中，如果已存在则更新
        if found_inf is not None:
            device_found_inf = found_inf_dict.setdefault(
                device, found_inf.to(device, non_blocking=True)
            )
        # 如果 lr_dict 不为 None 并且设备不在 lr_dict 中，则将 lr 转移到当前设备
        if lr_dict is not None and device not in lr_dict:
            lr_dict[device] = lr.to(device=device, non_blocking=True)  # type: ignore[union-attr]
            lr = lr_dict[device]
        # 对 device_state_steps 中的每个张量执行加一操作
        torch._foreach_add_(device_state_steps, 1)
        # 调用底层的融合 Adam 优化算法函数，传入各类参数
        torch._fused_adam_(
            device_params,
            device_grads,
            device_exp_avgs,
            device_exp_avg_sqs,
            device_max_exp_avg_sqs,
            device_state_steps,
            amsgrad=amsgrad,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            weight_decay=weight_decay,
            eps=eps,
            maximize=maximize,
            grad_scale=device_grad_scale,
            found_inf=device_found_inf,
        )
        # 如果 device_found_inf 不为 None，则对 device_state_steps 中的每个张量执行减法操作
        if device_found_inf is not None:
            torch._foreach_sub_(
                device_state_steps, [device_found_inf] * len(device_state_steps)
            )
    foreach: Optional[bool] = None,  
    # 可选的布尔类型参数，用于某种用途，初始值为None

    capturable: bool = False,         
    # 布尔类型参数，指示是否可以捕获某些状态，初始值为False

    differentiable: bool = False,     
    # 布尔类型参数，指示对象是否可微分，初始值为False

    fused: Optional[bool] = None,     
    # 可选的布尔类型参数，用于某种融合操作，初始值为None

    grad_scale: Optional[Tensor] = None,  
    # 可选的张量类型参数，用于梯度缩放，初始值为None

    found_inf: Optional[Tensor] = None,    
    # 可选的张量类型参数，表示是否找到了无穷大值，初始值为None

    has_complex: bool = False,       
    # 布尔类型参数，指示对象是否具有复数部分，初始值为False

    *,  # 星号表示后面的参数必须使用关键字传递

    amsgrad: bool,                   
    # 布尔类型参数，指示是否使用AMSGrad优化算法

    beta1: float,                    
    # 浮点数参数，表示优化算法中的beta1超参数

    beta2: float,                    
    # 浮点数参数，表示优化算法中的beta2超参数

    lr: Union[float, Tensor],       
    # 联合类型参数，可以是浮点数或张量类型，表示学习率

    weight_decay: float,             
    # 浮点数参数，表示优化算法中的权重衰减

    eps: float,                      
    # 浮点数参数，表示优化算法中的epsilon值

    maximize: bool,                  
    # 布尔类型参数，指示是否最大化优化目标
    r"""Functional API that performs Adam algorithm computation.

    See :class:`~torch.optim.Adam` for details.
    """
    # 如果用户没有显式指定 fused 或 foreach，则默认使用非融合版本，并传递 False 给 use_fused。
    # 这不是错误——我们希望在将融合实现设为默认之前，给融合实现一个铺垫时间，即使它通常更快。
    if fused is None and foreach is None:
        # 通过 _default_to_fused_or_foreach 函数确定应当使用 fused 还是 foreach
        _, foreach = _default_to_fused_or_foreach(
            params, differentiable, use_fused=False
        )
        # 如果 foreach 为 True，并且 lr 是 Tensor 类型，并且 capturable=False，则将 foreach 设为 False。
        if foreach and isinstance(lr, Tensor) and not capturable:
            foreach = False
    # 如果用户没有显式指定 fused，则将其设为 False。
    if fused is None:
        fused = False
    # 如果用户没有显式指定 foreach，则将其设为 False。
    if foreach is None:
        foreach = False

    # 编译时这个检查很慢，因此我们跳过它。如果确实需要，我们可以在 dynamo 中重新添加这个检查。
    if not torch._utils.is_compiling() and not all(
        isinstance(t, torch.Tensor) for t in state_steps
    ):
        # 如果 state_steps 参数不是由单个张量组成的列表，则引发 RuntimeError。
        raise RuntimeError(
            "API has changed, `state_steps` argument must contain a list of singleton tensors"
        )

    # 如果 foreach 为 True 且正在使用 torch.jit.script，抛出 RuntimeError。
    if foreach and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with foreach optimizers")
    # 如果 fused 为 True 且正在使用 torch.jit.script，抛出 RuntimeError。
    if fused and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with fused optimizers")

    # 根据用户指定的优化策略选择相应的优化函数
    if fused and not torch.jit.is_scripting():
        func = _fused_adam
    elif foreach and not torch.jit.is_scripting():
        func = _multi_tensor_adam
    else:
        func = _single_tensor_adam

    # 调用选定的优化函数来执行 Adam 算法的计算
    func(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        amsgrad=amsgrad,
        has_complex=has_complex,
        beta1=beta1,
        beta2=beta2,
        lr=lr,
        weight_decay=weight_decay,
        eps=eps,
        maximize=maximize,
        capturable=capturable,
        differentiable=differentiable,
        grad_scale=grad_scale,
        found_inf=found_inf,
    )
```