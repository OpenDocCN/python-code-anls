# `.\pytorch\torch\optim\adamw.py`

```
# 声明不允许使用未类型化的函数定义（用于类型检查）
# 导入必要的类型和模块
from typing import cast, List, Optional, Tuple, Union

# 导入 PyTorch 相关模块
import torch
from torch import Tensor
from torch.utils._foreach_utils import _get_fused_kernels_supported_devices

# 导入自定义的优化器及其相关函数和类
from .optimizer import (
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

# 指定 AdamW 类在模块外部可见的成员列表
__all__ = ["AdamW", "adamw"]

# 定义 AdamW 类，继承自 Optimizer 类
class AdamW(Optimizer):
    # 构造函数，初始化 AdamW 类实例
    def __init__(
        self,
        params: ParamsT,                   # 参数类型为 ParamsT 类型
        lr: Union[float, Tensor] = 1e-3,   # 学习率，默认为 0.001
        betas: Tuple[float, float] = (0.9, 0.999),  # beta 参数元组，默认为 (0.9, 0.999)
        eps: float = 1e-8,                 # epsilon 参数，默认为 1e-8
        weight_decay: float = 1e-2,        # 权重衰减参数，默认为 0.01
        amsgrad: bool = False,             # 是否使用 AMSGrad，默认为 False
        *,
        maximize: bool = False,            # 是否最大化目标，默认为 False
        foreach: Optional[bool] = None,    # 是否使用 foreach 循环，默认为 None
        capturable: bool = False,          # 是否支持捕获，默认为 False
        differentiable: bool = False,      # 是否可微分，默认为 False
        fused: Optional[bool] = None,      # 是否融合操作，默认为 None
    # 参数验证和设置默认参数字典
    if (
        not 0.0 <= lr:  # 检查学习率 lr 是否在有效范围内
    ):
        raise ValueError(f"Invalid learning rate: {lr}")  # 如果不在范围内，抛出数值错误异常
    if isinstance(lr, Tensor) and foreach and not capturable:
        raise ValueError(
            "lr as a Tensor is not supported for capturable=False and foreach=True"
        )  # 如果 lr 是 Tensor 类型且 foreach=True 且 capturable=False，抛出数值错误异常
    if not 0.0 <= eps:  # 检查 epsilon 值 eps 是否在有效范围内
        raise ValueError(f"Invalid epsilon value: {eps}")  # 如果不在范围内，抛出数值错误异常
    if not 0.0 <= betas[0] < 1.0:  # 检查 beta 参数列表中索引为 0 的值是否在有效范围内
        raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")  # 如果不在范围内，抛出数值错误异常
    if not 0.0 <= betas[1] < 1.0:  # 检查 beta 参数列表中索引为 1 的值是否在有效范围内
        raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")  # 如果不在范围内，抛出数值错误异常
    if not 0.0 <= weight_decay:  # 检查 weight_decay 值是否在有效范围内
        raise ValueError(f"Invalid weight_decay value: {weight_decay}")  # 如果不在范围内，抛出数值错误异常
    
    # 设置默认参数字典
    defaults = dict(
        lr=lr,  # 学习率
        betas=betas,  # beta 参数列表
        eps=eps,  # epsilon 值
        weight_decay=weight_decay,  # 权重衰减
        amsgrad=amsgrad,  # 是否使用 amsgrad
        foreach=foreach,  # 是否为 foreach 模式
        maximize=maximize,  # 是否最大化优化目标
        capturable=capturable,  # 是否支持捕获状态
        differentiable=differentiable,  # 是否可微分
        fused=fused,  # 是否使用融合优化
    )
    
    # 调用父类的初始化方法，传入参数和默认参数字典
    super().__init__(params, defaults)
    
    if fused:
        if differentiable:
            raise RuntimeError("`fused` does not support `differentiable`")  # 如果使用融合优化且 differentiable=True，则抛出运行时错误
        self._step_supports_amp_scaling = True
        # TODO(crcrpar): [low prec params & their higher prec copy]
        # Suppor AMP with FP16/BF16 model params which would need
        # higher prec copy of params to do update math in higher prec to
        # alleviate the loss of information.
        # 支持使用 FP16/BF16 模型参数的混合精度训练，这需要对参数进行更高精度的复制以进行数学更新，以减少信息损失。
        fused_supported_devices = _get_fused_kernels_supported_devices()
        # 检查所有参数组中的所有参数是否都支持融合优化
        if not all(
            p.device.type in fused_supported_devices and torch.is_floating_point(p)
            for pg in self.param_groups  # 遍历参数组
            for p in pg["params"]  # 遍历参数组中的参数列表
        ):
            raise RuntimeError(
                "`fused=True` requires all the params to be floating point Tensors of "
                f"supported devices: {fused_supported_devices}."
            )  # 如果不是所有参数都是支持的浮点数张量，则抛出运行时错误
        if foreach:
            raise RuntimeError("`fused` and `foreach` cannot be `True` together.")  # 如果 fused=True 且 foreach=True，则抛出运行时错误
    # 重写父类的 __setstate__ 方法，设置对象的状态
    def __setstate__(self, state):
        # 调用父类的 __setstate__ 方法，传入状态参数，完成基础状态的设置
        super().__setstate__(state)
        # 遍历优化器的参数组列表
        for group in self.param_groups:
            # 设置默认参数 'amsgrad' 为 False，表示不使用amsgrad算法
            group.setdefault("amsgrad", False)
            # 设置默认参数 'maximize' 为 False，表示不最大化优化目标
            group.setdefault("maximize", False)
            # 设置默认参数 'foreach' 为 None，用于指定参数组的迭代操作
            group.setdefault("foreach", None)
            # 设置默认参数 'capturable' 为 False，表示不可捕获的
            group.setdefault("capturable", False)
            # 设置默认参数 'differentiable' 为 False，表示参数不可微分
            group.setdefault("differentiable", False)
            # 设置默认参数 'fused' 为 None，用于指定参数组的融合状态
            fused = group.setdefault("fused", None)
            # 遍历当前参数组中的参数列表
            for p in group["params"]:
                # 获取当前参数 p 的状态列表 p_state
                p_state = self.state.get(p, [])
                # 如果 p_state 非空且 p_state["step"] 不是 torch 的张量
                if len(p_state) != 0 and not torch.is_tensor(p_state["step"]):
                    # 获取 p_state["step"] 的浮点值，并将其转换为 torch 张量类型
                    step_val = float(p_state["step"])
                    # 如果 group["capturable"] 或 group["fused"] 为 True
                    if group["capturable"] or group["fused"]:
                        # 创建指定设备上的 torch 张量，指定数据类型为 _get_scalar_dtype(is_fused=fused)
                        p_state["step"] = (
                            torch.tensor(
                                step_val,
                                dtype=_get_scalar_dtype(is_fused=fused),
                                device=p.device,
                            )
                        )
                    else:
                        # 创建默认设备上的 torch 张量，数据类型为 _get_scalar_dtype()
                        p_state["step"] = torch.tensor(step_val, dtype=_get_scalar_dtype())
        ):
            # 初始化一个标志，用于检查是否有复数张量
            has_complex = False
            # 遍历参数组中的每个参数
            for p in group["params"]:
                # 如果梯度为None，跳过该参数
                if p.grad is None:
                    continue
                # 检查参数是否包含复数部分
                has_complex |= torch.is_complex(p)
                # 将具有梯度的参数添加到列表中
                params_with_grad.append(p)
                # 如果梯度为稀疏张量，则抛出运行时错误
                if p.grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")
                # 将参数的梯度添加到梯度列表中
                grads.append(p.grad)

                # 获取参数对应的状态字典
                state = self.state[p]

                # 状态初始化
                if len(state) == 0:
                    # 如果capturable和fused都未开启，则在CPU上存储`step`
                    # 这是因为在CUDA和XLA上进行核启动非常昂贵
                    state["step"] = (
                        torch.zeros(
                            (),
                            dtype=_get_scalar_dtype(is_fused=group["fused"]),
                            device=p.device,
                        )
                        if group["capturable"] or group["fused"]
                        else torch.tensor(0.0, dtype=_get_scalar_dtype())
                    )
                    # 梯度值的指数移动平均
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # 梯度值平方的指数移动平均
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # 如果开启了amsgrad，则维护所有梯度平方移动平均的最大值
                    if amsgrad:
                        state["max_exp_avg_sq"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )

                # 将状态字典中的梯度平均值添加到列表中
                exp_avgs.append(state["exp_avg"])
                # 将状态字典中的梯度平方平均值添加到列表中
                exp_avg_sqs.append(state["exp_avg_sq"])

                # 如果开启了amsgrad，则将状态字典中的最大梯度平方平均值添加到列表中
                if group["amsgrad"]:
                    max_exp_avg_sqs.append(state["max_exp_avg_sq"])
                # 如果在可微模式下，且状态字典中的`step`需要梯度，则抛出运行时错误
                if group["differentiable"] and state["step"].requires_grad:
                    raise RuntimeError(
                        "`requires_grad` is not supported for `step` in differentiable mode"
                    )

                # 如果开启了foreach并且lr是一个张量，并且capturable为False，则抛出运行时错误
                if (
                    group["foreach"]
                    and isinstance(group["lr"], Tensor)
                    and not group["capturable"]
                ):
                    raise RuntimeError(
                        "lr as a Tensor is not supported for capturable=False and foreach=True"
                    )

                # 将状态字典中的`step`添加到列表中
                state_steps.append(state["step"])
            # 返回是否存在复数张量的标志
            return has_complex

        @_use_grad_for_differentiable
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        # 执行 CUDA 图形捕获健康检查
        self._cuda_graph_capture_health_check()

        # 初始化损失值为 None
        loss = None
        # 如果有传入闭包函数，则在启用梯度的环境中重新评估模型并计算损失
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # 遍历优化器的参数组
        for group in self.param_groups:
            # 初始化存储参数梯度、梯度、指数移动平均值等列表
            params_with_grad: List[Tensor] = []
            grads: List[Tensor] = []
            exp_avgs: List[Tensor] = []
            exp_avg_sqs: List[Tensor] = []
            max_exp_avg_sqs: List[Tensor] = []
            state_steps: List[Tensor] = []
            # 获取当前参数组的 amsgrad 标志
            amsgrad: bool = group["amsgrad"]
            # 获取当前参数组的 beta1 和 beta2 参数
            beta1, beta2 = cast(Tuple[float, float], group["betas"])

            # 初始化当前参数组，并获取是否存在复数参数的标志
            has_complex = self._init_group(
                group,
                params_with_grad,
                grads,
                amsgrad,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
            )

            # 调用 AdamW 优化函数来执行优化步骤
            adamw(
                params_with_grad,
                grads,
                exp_avgs,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=amsgrad,
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
                has_complex=has_complex,
            )

        # 返回损失值
        return loss
# 将 AdamW 类的文档字符串设置为 AdamW 算法的实现说明
AdamW.__doc__ = (
    r"""Implements AdamW algorithm.
    # 定义一个优化算法，通常是针对深度学习中的参数优化问题
    # 使用Adam优化器，结合了动量(momentum)和自适应学习率(adaptive learning rate)的特性
    
    # 输入参数说明：
    # - lr: 学习率，控制参数更新的步长
    # - betas: 两个动量的参数 (beta1, beta2)
    # - params: 待优化的参数 (theta0)
    # - objective: 目标函数，计算损失的函数 (f(theta))
    # - epsilon: 防止除零的小常数
    # - weight decay: 权重衰减的系数 (lambda)
    # - amsgrad: 是否使用AMSGrad算法进行优化
    # - maximize: 是否是最大化目标函数（默认为最小化）
    
    # 初始化变量说明：
    # - m0: 第一时刻的动量 (first moment)，初始化为0
    # - v0: 第二时刻的动量 (second moment)，初始化为0
    # - max_v0: AMSGrad算法中的变量，初始化为0
    
    # 算法流程：
    # - 通过梯度的反向传播计算当前参数的梯度 g_t
    # - 根据是否最大化目标函数，调整梯度方向
    # - 更新参数 theta_t：
    #   - 计算动量 m_t 和 v_t
    #   - 计算带偏置修正的一阶和二阶矩 (m_hat_t, v_hat_t)
    #   - 如果使用AMSGrad，更新最大的二阶矩 (max_v_t)
    #   - 最终更新参数 theta_t，根据优化器的公式进行更新
    # - 返回优化后的参数 theta_t
    # 这部分代码是一个多行字符串，用于文档化Adam优化算法的详细参数和链接
    """
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
        weight_decay (float, optional): weight decay coefficient (default: 1e-2)
        amsgrad (bool, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
        {_maximize_doc}
        {_foreach_doc}
        {_capturable_doc}
        {_differentiable_doc}
        {_fused_doc}
    .. Note::
        A prototype implementation of Adam and AdamW for MPS supports `torch.float32` and `torch.float16`.
    .. _Decoupled Weight Decay Regularization:
        https://arxiv.org/abs/1711.05101
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    
    """
def _single_tensor_adamw(
    params: List[Tensor],                          # 参数列表，包含需要更新的张量
    grads: List[Tensor],                           # 梯度列表，对应参数列表的梯度
    exp_avgs: List[Tensor],                        # 指数加权平均值列表，用于动量更新
    exp_avg_sqs: List[Tensor],                     # 指数加权平方平均值列表，用于AdaGrad更新
    max_exp_avg_sqs: List[Tensor],                 # 最大指数加权平方平均值列表，用于AMSGrad更新
    state_steps: List[Tensor],                     # 状态步数列表，用于动量和步数更新
    grad_scale: Optional[Tensor],                  # 梯度缩放因子，如果存在则必须为None
    found_inf: Optional[Tensor],                   # 是否发现梯度中有无穷值，如果存在则必须为None
    *,
    amsgrad: bool,                                  # 是否使用AMSGrad优化
    beta1: float,                                   # 第一动量的指数衰减率
    beta2: float,                                   # 第二动量的指数衰减率
    lr: Union[Tensor, float],                       # 学习率，可以是Tensor或float
    weight_decay: float,                            # 权重衰减参数
    eps: float,                                     # 用于数值稳定性的小常数
    maximize: bool,                                 # 是否最大化优化目标
    capturable: bool,                               # 是否支持捕获张量
    differentiable: bool,                           # 是否支持自动求导
    has_complex: bool,                              # 是否处理复数类型
):
    assert grad_scale is None and found_inf is None   # 断言：梯度缩放因子和无穷值检查必须为None

    if torch.jit.is_scripting():
        # 如果正在使用JIT编译，这个断言是因为JIT不能智能地识别下面的操作
        # 可以处理float和Tensor类型的lr，所以我们断言lr应该是float类型
        assert isinstance(lr, float)



def _multi_tensor_adamw(
    params: List[Tensor],                          # 参数列表，包含需要更新的张量
    grads: List[Tensor],                           # 梯度列表，对应参数列表的梯度
    exp_avgs: List[Tensor],                        # 指数加权平均值列表，用于动量更新
    exp_avg_sqs: List[Tensor],                     # 指数加权平方平均值列表，用于AdaGrad更新
    max_exp_avg_sqs: List[Tensor],                 # 最大指数加权平方平均值列表，用于AMSGrad更新
    state_steps: List[Tensor],                     # 状态步数列表，用于动量和步数更新
    grad_scale: Optional[Tensor],                  # 梯度缩放因子，如果存在则必须为None
    found_inf: Optional[Tensor],                   # 是否发现梯度中有无穷值，如果存在则必须为None
    *,
    amsgrad: bool,                                  # 是否使用AMSGrad优化
    beta1: float,                                   # 第一动量的指数衰减率
    beta2: float,                                   # 第二动量的指数衰减率
    lr: Union[Tensor, float],                       # 学习率，可以是Tensor或float
    weight_decay: float,                            # 权重衰减参数
    eps: float,                                     # 用于数值稳定性的小常数
    maximize: bool,                                 # 是否最大化优化目标
    capturable: bool,                               # 是否支持捕获张量
    differentiable: bool,                           # 是否支持自动求导
    has_complex: bool,                              # 是否处理复数类型
):
    if len(params) == 0:
        return

    if isinstance(lr, Tensor) and not capturable:
        # 如果lr是Tensor类型并且不支持捕获，则抛出异常
        raise RuntimeError(
            "lr as a Tensor is not supported for capturable=False and foreach=True"
        )

    # 如果正在编译，编译器将处理cudagraph检查，参见注释 [torch.compile x capturable]
    if not torch._utils.is_compiling() and capturable:
        # 如果不是在编译阶段且支持捕获，则进行设备支持检查
        capturable_supported_devices = _get_capturable_supported_devices(
            supports_xla=False
        )
        assert all(
            p.device.type == step.device.type
            and p.device.type in capturable_supported_devices
            for p, step in zip(params, state_steps)
        ), f"If capturable=True, params and state_steps must be on supported devices: {capturable_supported_devices}."

    assert not differentiable, "_foreach ops don't support autograd"

    assert grad_scale is None and found_inf is None   # 断言：梯度缩放因子和无穷值检查必须为None

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



def _fused_adamw(
    params: List[Tensor],                          # 参数列表，包含需要更新的张量
    grads: List[Tensor],                           # 梯度列表，对应参数列表的梯度
    exp_avgs: List[Tensor],                        # 指数加权平均值列表，用于动量更新
    exp_avg_sqs: List[Tensor],                     # 指数加权平方平均值列表，用于AdaGrad更新
    max_exp_avg_sqs: List[Tensor],                 # 最大指数加权平方平均值列表，用于AMSGrad更新
    state_steps: List[Tensor],                     # 状态步数列表，用于动量和步数更新
    grad_scale: Optional[Tensor],                  # 梯度缩放因子，如果存在则必须为None
    found_inf: Optional[Tensor],                   # 是否发现梯度中有无穷值，如果存在则必须为None
    *,
    amsgrad: bool,                                  # 是否使用AMSGrad优化
    beta1: float,                                   # 第一动量的指数衰减率
    beta2: float,                                   # 第二动量的指数衰减率
    lr: Union[Tensor, float],                       # 学习率，可以是Tensor或float
    weight_decay: float,                            # 权重衰减参数
    eps: float,                                     # 用于数值稳定性的小常数
    maximize: bool,                                 # 是否最大化优化目标
):
    capturable: bool,  # 是否支持捕获功能，保持一致性需要
    differentiable: bool,  # 是否可微分
    has_complex: bool,  # 是否包含复杂功能，保持一致性需要
# 定义 AdamW 优化器的函数签名和类型注解
def adamw(
    params: List[Tensor],                    # 参数列表，包含模型的各个参数张量
    grads: List[Tensor],                     # 梯度列表，包含相应参数的梯度张量
    exp_avgs: List[Tensor],                  # 指数移动平均列表，用于动量优化
    exp_avg_sqs: List[Tensor],               # 指数移动平方平均列表，用于二阶矩优化
    max_exp_avg_sqs: List[Tensor],           # 最大指数移动平方平均列表，用于 AMSGrad
    state_steps: List[Tensor],               # 状态步数列表，用于记录优化步数
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
) -> None:
    # 如果参数列表为空，则直接返回，不进行优化操作
    if not params:
        return
    
    # 如果 differentiable 参数为 True，则抛出异常，因为融合 Adam 不支持 differentiable=True
    if differentiable:
        raise RuntimeError("Adam with fused=True does not support differentiable=True")

    # 根据梯度缩放系数 grad_scale 构建设备相关的字典，以设备为 key，梯度缩放系数为 value
    grad_scale_dict: DeviceDict = (
        {grad_scale.device: grad_scale} if grad_scale is not None else {}
    )
    
    # 根据发现无穷大值 found_inf 构建设备相关的字典，以设备为 key，无穷大值标志为 value
    found_inf_dict: DeviceDict = (
        {found_inf.device: found_inf} if found_inf is not None else {}
    )

    # 如果学习率 lr 是 Tensor 类型且在 CUDA 设备上，则构建设备相关的字典，以设备为 key，学习率为 value
    lr_dict: Optional[DeviceDict] = (
        {lr.device: lr} if isinstance(lr, Tensor) and str(lr.device) != "cpu" else None
    )

    # 将各个张量按设备和数据类型分组，返回一个字典，键为设备和数据类型的元组，值为对应的张量列表
    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps]
    )
    
    # 遍历分组后的张量字典
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
        # 如果设备类型是 "mps"，通常表示多 GPU 设备
        if device.type == "mps":  # type: ignore[union-attr]
            # 断言 found_inf 和 grad_scale 都为 None，并且 lr 不是 Tensor 类型
            assert found_inf is None and grad_scale is None
            assert not isinstance(lr, Tensor)

        # 初始化设备相关的梯度缩放和无穷大值标志为 None
        device_grad_scale, device_found_inf = None, None
        
        # 如果 grad_scale 不为 None，则将其设置为 grad_scale_dict 的值，如果不存在则创建
        if grad_scale is not None:
            device_grad_scale = grad_scale_dict.setdefault(
                device, grad_scale.to(device, non_blocking=True)
            )
        
        # 如果 found_inf 不为 None，则将其设置为 found_inf_dict 的值，如果不存在则创建
        if found_inf is not None:
            device_found_inf = found_inf_dict.setdefault(
                device, found_inf.to(device, non_blocking=True)
            )
        
        # 如果 lr_dict 不为 None 并且设备不在 lr_dict 中，则将 lr 转移到设备上并设置到 lr_dict 中
        if lr_dict is not None and device not in lr_dict:
            lr = lr_dict.setdefault(
                device, lr.to(device=device, non_blocking=True)  # type: ignore[union-attr]
            )
        
        # 对 device_state_steps 中的每个张量执行加一操作
        torch._foreach_add_(device_state_steps, 1)
        
        # 调用底层的融合 AdamW 函数进行优化
        torch._fused_adamw_(
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
        
        # 如果 device_found_inf 不为 None，则对 device_state_steps 执行逐元素减操作
        if device_found_inf is not None:
            torch._foreach_sub_(
                device_state_steps, [device_found_inf] * len(device_state_steps)
            )
    foreach: Optional[bool] = None,
    # foreach 参数，可选的布尔类型，默认为 None

    capturable: bool = False,
    # capturable 参数，布尔类型，默认为 False

    differentiable: bool = False,
    # differentiable 参数，布尔类型，默认为 False

    fused: Optional[bool] = None,
    # fused 参数，可选的布尔类型，默认为 None

    grad_scale: Optional[Tensor] = None,
    # grad_scale 参数，可选的张量类型，默认为 None

    found_inf: Optional[Tensor] = None,
    # found_inf 参数，可选的张量类型，默认为 None

    has_complex: bool = False,
    # has_complex 参数，布尔类型，默认为 False

    *,
    # 星号表示以下参数是强制使用关键字传递

    amsgrad: bool,
    # amsgrad 参数，布尔类型，必须通过关键字传递

    beta1: float,
    # beta1 参数，浮点数类型，必须通过关键字传递

    beta2: float,
    # beta2 参数，浮点数类型，必须通过关键字传递

    lr: Union[float, Tensor],
    # lr 参数，可以是浮点数或张量类型，必须通过关键字传递

    weight_decay: float,
    # weight_decay 参数，浮点数类型，必须通过关键字传递

    eps: float,
    # eps 参数，浮点数类型，必须通过关键字传递

    maximize: bool,
    # maximize 参数，布尔类型，必须通过关键字传递
    r"""Functional API that performs AdamW algorithm computation.

    See :class:`~torch.optim.AdamW` for details.
    """
    # 检查是否处于编译状态，并且 state_steps 中的所有元素是否都是 torch.Tensor 类型
    if not torch._utils.is_compiling() and not all(
        isinstance(t, torch.Tensor) for t in state_steps
    ):
        # 如果不符合条件，抛出运行时错误提示
        raise RuntimeError(
            "API has changed, `state_steps` argument must contain a list of singleton tensors"
        )

    # 根据用户输入的 fused 和 foreach 参数来设置默认值
    # 如果 fused 和 foreach 都未指定，则根据参数 params, differentiable 来选择使用 fused 还是 foreach
    if fused is None and foreach is None:
        _, foreach = _default_to_fused_or_foreach(
            params, differentiable, use_fused=False
        )
        # 对于不支持的情况，当 lr 是 Tensor 类型且 capturable=False 时，不使用 foreach
        if foreach and isinstance(lr, Tensor) and not capturable:
            foreach = False
    # 如果 fused 未指定，则默认为 False
    if fused is None:
        fused = False
    # 如果 foreach 未指定，则默认为 False
    if foreach is None:
        foreach = False

    # 如果用户启用了 foreach 且正在使用 torch.jit.script，则抛出运行时错误
    if foreach and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with foreach optimizers")
    # 如果用户启用了 fused 且正在使用 torch.jit.script，则抛出运行时错误
    if fused and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with fused optimizers")

    # 根据用户选择的 fused 和 foreach 来选择具体的优化算法函数
    if fused and not torch.jit.is_scripting():
        func = _fused_adamw
    elif foreach and not torch.jit.is_scripting():
        func = _multi_tensor_adamw
    else:
        func = _single_tensor_adamw

    # 调用选定的优化算法函数 func，传入各种参数进行优化计算
    func(
        params,
        grads,
        exp_avgs,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        amsgrad=amsgrad,
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
        has_complex=has_complex,
    )
```