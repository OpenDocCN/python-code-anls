# `.\pytorch\torch\optim\swa_utils.py`

```py
# mypy: allow-untyped-defs
r"""Implementation for Stochastic Weight Averaging implementation."""

# 引入 itertools、math、warnings 等常用库
import itertools
import math
import warnings

# 从 copy 模块中导入 deepcopy 函数
from copy import deepcopy

# 导入类型提示相关的模块和类型
from typing import Any, Callable, Iterable, List, Literal, Optional, Tuple, Union

# 导入 PyTorch 相关模块
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim.lr_scheduler import _format_param, LRScheduler
from torch.utils._foreach_utils import _get_foreach_kernels_supported_devices
from .optimizer import Optimizer

# 指定导出的模块成员列表
__all__ = [
    "AveragedModel",
    "update_bn",
    "SWALR",
    "get_ema_multi_avg_fn",
    "get_swa_multi_avg_fn",
    "get_ema_avg_fn",
    "get_swa_avg_fn",
]

# 从 torch.utils._foreach_utils 导入 _group_tensors_by_device_and_dtype 函数
from torch.utils._foreach_utils import _group_tensors_by_device_and_dtype

# 定义 PARAM_LIST 类型为 Tensor 元组或列表
PARAM_LIST = Union[Tuple[Tensor, ...], List[Tensor]]


def get_ema_multi_avg_fn(decay=0.999):
    """Get the function applying exponential moving average (EMA) across multiple params."""
    
    # 定义一个函数 ema_update，用于在多个参数上应用指数移动平均 (EMA)
    @torch.no_grad()
    def ema_update(ema_param_list: PARAM_LIST, current_param_list: PARAM_LIST, _):
        # 对于浮点数和复数类型的参数，使用 torch._foreach_lerp_ 函数进行线性插值操作
        if torch.is_floating_point(ema_param_list[0]) or torch.is_complex(
            ema_param_list[0]
        ):
            torch._foreach_lerp_(ema_param_list, current_param_list, 1 - decay)
        else:
            # 对于其他类型的参数，逐个参数进行 EMA 更新
            for p_ema, p_model in zip(ema_param_list, current_param_list):
                p_ema.copy_(p_ema * decay + p_model * (1 - decay))

    return ema_update


def get_swa_multi_avg_fn():
    """Get the function applying stochastic weight average (SWA) across multiple params."""
    
    # 定义一个函数 swa_update，用于在多个参数上应用随机权重平均 (SWA)
    @torch.no_grad()
    def swa_update(
        averaged_param_list: PARAM_LIST,
        current_param_list: PARAM_LIST,
        num_averaged: Union[Tensor, int],
    ):
        # 对于浮点数和复数类型的参数，使用 torch._foreach_lerp_ 函数进行线性插值操作
        if torch.is_floating_point(averaged_param_list[0]) or torch.is_complex(
            averaged_param_list[0]
        ):
            torch._foreach_lerp_(
                averaged_param_list, current_param_list, 1 / (num_averaged + 1)
            )
        else:
            # 对于其他类型的参数，计算参数差异并进行 SWA 更新
            diffs = torch._foreach_sub(current_param_list, averaged_param_list)
            if isinstance(num_averaged, Tensor):
                torch._foreach_addcdiv_(
                    averaged_param_list,
                    diffs,
                    [num_averaged + 1] * len(averaged_param_list),
                )
            else:
                torch._foreach_add_(
                    averaged_param_list, diffs, alpha=1.0 / (num_averaged + 1)
                )

    return swa_update


def get_ema_avg_fn(decay=0.999):
    """Get the function applying exponential moving average (EMA) across a single param."""
    
    # 定义一个函数 ema_update，用于在单个参数上应用指数移动平均 (EMA)
    @torch.no_grad()
    def ema_update(ema_param: Tensor, current_param: Tensor, num_averaged):
        return decay * ema_param + (1 - decay) * current_param

    return ema_update


def get_swa_avg_fn():
    """Get the function applying stochastic weight average (SWA) across a single param."""

    # 此函数暂未实现，预计用于在单个参数上应用随机权重平均 (SWA)，但未给出具体实现
    pass
    # 使用 @torch.no_grad() 装饰器，确保在此函数中不会计算梯度
    @torch.no_grad()
    # 定义名为 swa_update 的函数，用于更新平均参数
    def swa_update(
        averaged_param: Tensor, current_param: Tensor, num_averaged: Union[Tensor, int]
    ):
        # 计算平均参数的更新值，以实现平均参数的滑动更新
        return averaged_param + (current_param - averaged_param) / (num_averaged + 1)
    
    # 返回 swa_update 函数的引用，用于在其他地方调用
    return swa_update
# 定义一个名为 AveragedModel 的类，继承自 torch.nn.Module
class AveragedModel(Module):
    # 文档字符串，说明该类实现了用于随机权重平均（SWA）和指数移动平均（EMA）的平均模型。
    # 提供了对 SWA 和 EMA 的简要介绍及其应用背景文献引用

    # 初始化方法，创建一个给定模型的副本，并可在指定设备上存储
    def __init__(self, model, device=None, avg_fn=None, multi_avg_fn=None, use_buffers=False):
        # 调用父类初始化方法
        super(AveragedModel, self).__init__()
        # 将提供的模型存储在 self.model 属性中
        self.model = model
        # 如果提供了设备，则将 averaged model 存储在该设备上
        self.device = device
        # 指定参数更新的平均函数，如果为 None 则使用等权重平均
        self.avg_fn = avg_fn
        # 指定参数就地更新的平均函数，如果为 None 则使用等权重平均
        self.multi_avg_fn = multi_avg_fn
        # 是否计算模型参数和缓冲区的运行平均，默认为 False
        self.use_buffers = use_buffers

    # 示例用法，展示如何使用 AveragedModel 类
    def example_function(self):
        # 示例代码，展示了如何在训练过程中使用 SWA
        pass

    # 更新模型参数的方法，用提供的模型的参数更新 averaged model 的参数
    def update_parameters(self, model):
        pass

# 在代码末尾未包含额外的注释，因为示例中未涉及额外代码部分
    n_averaged: Tensor



    # 定义类中的成员变量 `n_averaged`，表示已经平均过的次数，类型为 Tensor



    def __init__(
        self,
        model: Module,
        device: Optional[Union[int, torch.device]] = None,
        avg_fn: Optional[Callable[[Tensor, Tensor, Union[Tensor, int]], Tensor]] = None,
        multi_avg_fn: Optional[
            Callable[[PARAM_LIST, PARAM_LIST, Union[Tensor, int]], None]
        ] = None,
        use_buffers=False,



    # 初始化方法，创建一个新的 AveragedModel 对象
    # 参数:
    #   - model: 要平均的模型，类型为 Module
    #   - device: 可选参数，指定计算设备，可以是整数或 torch.device 对象
    #   - avg_fn: 可选参数，用于计算参数平均值的函数，接受三个参数 (当前参数, 平均参数, 更新步数)，返回平均后的参数
    #   - multi_avg_fn: 可选参数，用于多参数列表的平均函数，接受三个参数 (当前参数列表, 平均参数列表, 更新步数)，无返回值
    #   - use_buffers: 是否使用缓冲区来更新统计信息，默认为 False
    ):  # noqa: D107
        # 调用父类的构造函数（初始化方法）
        super().__init__()
        # 断言语句，确保 avg_fn 和 multi_avg_fn 只能提供一个
        assert (
            avg_fn is None or multi_avg_fn is None
        ), "Only one of avg_fn and multi_avg_fn should be provided"
        # 深拷贝传入的模型对象
        self.module = deepcopy(model)
        # 如果指定了设备，则将模型移动到对应设备上
        if device is not None:
            self.module = self.module.to(device)
        # 注册一个缓冲区，用于存储累积平均数，初始值为0
        self.register_buffer(
            "n_averaged", torch.tensor(0, dtype=torch.long, device=device)
        )
        # 设置平均函数和多重平均函数
        self.avg_fn = avg_fn
        self.multi_avg_fn = multi_avg_fn
        # 是否使用缓冲区的标志位
        self.use_buffers = use_buffers

    def forward(self, *args, **kwargs):
        """Forward pass."""
        # 调用模型的前向传播方法
        return self.module(*args, **kwargs)
    def update_parameters(self, model: Module):
        """Update model parameters."""
        # 获取当前对象的参数和缓冲区，根据 self.use_buffers 决定使用 self.module 还是 self 的参数
        self_param = (
            itertools.chain(self.module.parameters(), self.module.buffers())
            if self.use_buffers
            else self.parameters()
        )
        # 获取传入模型的参数和缓冲区，根据 self.use_buffers 决定使用传入模型的参数
        model_param = (
            itertools.chain(model.parameters(), model.buffers())
            if self.use_buffers
            else model.parameters()
        )
        # 初始化用于存储分离后参数的列表
        self_param_detached: List[Optional[Tensor]] = []
        model_param_detached: List[Optional[Tensor]] = []
        # 遍历当前对象和传入模型的参数，并进行分离和设备转换
        for p_averaged, p_model in zip(self_param, model_param):
            # 将传入模型的参数分离并转移到当前对象参数所在的设备上
            p_model_ = p_model.detach().to(p_averaged.device)
            # 分离当前对象的参数并添加到列表中
            self_param_detached.append(p_averaged.detach())
            # 添加分离后的传入模型参数到列表中
            model_param_detached.append(p_model_)
            # 如果 n_averaged 为 0，则直接将传入模型的参数复制给当前对象的参数
            if self.n_averaged == 0:
                p_averaged.detach().copy_(p_model_)

        # 如果 n_averaged 大于 0，则根据 multi_avg_fn 或 avg_fn 将分离后的参数进行平均化处理
        if self.n_averaged > 0:
            if self.multi_avg_fn is not None or self.avg_fn is None:
                # 将分离后的参数按设备和数据类型分组
                grouped_tensors = _group_tensors_by_device_and_dtype(
                    [self_param_detached, model_param_detached]
                )
                # 遍历分组后的参数并应用多平均函数或单平均函数
                for (device, _), (
                    [self_params, model_params],
                    _,
                ) in grouped_tensors.items():
                    if self.multi_avg_fn:
                        self.multi_avg_fn(
                            self_params, model_params, self.n_averaged.to(device)  # type: ignore[arg-type]
                        )
                    elif (
                        device is not None
                        and device.type in _get_foreach_kernels_supported_devices()
                    ):
                        multi_avg_fn = get_swa_multi_avg_fn()
                        multi_avg_fn(
                            self_params, model_params, self.n_averaged.to(device)
                        )
                    else:
                        avg_fn = get_swa_avg_fn()
                        n_averaged = self.n_averaged.to(device)
                        # 遍历参数并应用单平均函数
                        for p_averaged, p_model in zip(self_params, model_params):  # type: ignore[assignment]
                            p_averaged.copy_(avg_fn(p_averaged, p_model, n_averaged))
            else:
                # 如果有单平均函数，则应用单平均函数处理分离后的参数
                for p_averaged, p_model in zip(  # type: ignore[assignment]
                    self_param_detached, model_param_detached
                ):
                    n_averaged = self.n_averaged.to(p_averaged.device)
                    p_averaged.detach().copy_(
                        self.avg_fn(p_averaged.detach(), p_model, n_averaged)
                    )

        # 如果不使用缓冲区，将当前对象的缓冲区与传入模型的缓冲区保持同步
        if not self.use_buffers:
            # 如果不对缓冲区应用运行平均化，保持缓冲区与源模型同步
            for b_swa, b_model in zip(self.module.buffers(), model.buffers()):
                b_swa.detach().copy_(b_model.detach().to(b_swa.device))
        # 增加 n_averaged 计数
        self.n_averaged += 1
@torch.no_grad()
def update_bn(
    loader: Iterable[Any],
    model: Module,
    device: Optional[Union[int, torch.device]] = None,
):
    r"""Update BatchNorm running_mean, running_var buffers in the model.

    It performs one pass over data in `loader` to estimate the activation
    statistics for BatchNorm layers in the model.
    Args:
        loader (torch.utils.data.DataLoader): dataset loader to compute the
            activation statistics on. Each data batch should be either a
            tensor, or a list/tuple whose first element is a tensor
            containing data.
        model (torch.nn.Module): model for which we seek to update BatchNorm
            statistics.
        device (torch.device, optional): If set, data will be transferred to
            :attr:`device` before being passed into :attr:`model`.

    Example:
        >>> # xdoctest: +SKIP("Undefined variables")
        >>> loader, model = ...
        >>> torch.optim.swa_utils.update_bn(loader, model)

    .. note::
        The `update_bn` utility assumes that each data batch in :attr:`loader`
        is either a tensor or a list or tuple of tensors; in the latter case it
        is assumed that :meth:`model.forward()` should be called on the first
        element of the list or tuple corresponding to the data batch.
    """
    # 初始化一个空字典用于保存 BatchNorm 层的动量参数
    momenta = {}
    # 遍历模型中的每个模块
    for module in model.modules():
        # 如果模块是 BatchNorm 层
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            # 重置 BatchNorm 层的运行统计数据
            module.reset_running_stats()
            # 将当前 BatchNorm 层的动量参数保存到 momenta 字典中
            momenta[module] = module.momentum

    # 如果没有找到任何 BatchNorm 层，直接返回
    if not momenta:
        return

    # 保存模型当前的训练状态，并将模型设置为训练模式
    was_training = model.training
    model.train()
    # 将所有 BatchNorm 层的动量参数设置为 None
    for module in momenta.keys():
        module.momentum = None

    # 遍历数据加载器中的每个输入数据批次
    for input in loader:
        # 如果输入是一个列表或元组，取其第一个元素作为数据张量
        if isinstance(input, (list, tuple)):
            input = input[0]
        # 如果指定了设备，将输入数据移动到该设备上
        if device is not None:
            input = input.to(device)

        # 将输入数据传递给模型进行前向计算
        model(input)

    # 恢复每个 BatchNorm 层的原始动量参数
    for bn_module in momenta.keys():
        bn_module.momentum = momenta[bn_module]
    # 恢复模型之前的训练状态
    model.train(was_training)


class SWALR(LRScheduler):
    r"""Anneals the learning rate in each parameter group to a fixed value.

    This learning rate scheduler is meant to be used with Stochastic Weight
    Averaging (SWA) method (see `torch.optim.swa_utils.AveragedModel`).

    Args:
        optimizer (torch.optim.Optimizer): wrapped optimizer
        swa_lrs (float or list): the learning rate value for all param groups
            together or separately for each group.
        annealing_epochs (int): number of epochs in the annealing phase
            (default: 10)
        annealing_strategy (str): "cos" or "linear"; specifies the annealing
            strategy: "cos" for cosine annealing, "linear" for linear annealing
            (default: "cos")
        last_epoch (int): the index of the last epoch (default: -1)

    The :class:`SWALR` scheduler can be used together with other
    schedulers to switch to a constant learning rate late in the training
    as in the example below.
    """
    实现基于优化器的 SWA 调度器，参考 Averaging Weights Leads to Wider Optima and Better Generalization
    
    Args:
        optimizer (Optimizer): 模型优化器
        swa_lr (float): SWA 学习率
        anneal_epochs (int, optional): 衰减周期，默认为 10
        anneal_strategy (Literal["cos", "linear"], optional): 衰减策略，默认为 "cos"
        last_epoch (int, optional): 上一个周期，默认为 -1
        
    Raises:
        ValueError: 如果 anneal_strategy 不是 "cos" 或 "linear"
        ValueError: 如果 anneal_epochs 不是正整数
    
    Attributes:
        anneal_func (function): 衰减函数，根据 anneal_strategy 选择 _cosine_anneal 或 _linear_anneal
        anneal_epochs (int): 衰减周期数

    Notes:
        这个类实现了基于 SWA 的学习率调度器，根据给定的优化器和参数设置 SWA 相关的参数，并提供了不同的衰减策略选项。
        更多细节可以参考原文：Averaging Weights Leads to Wider Optima and Better Generalization。
        
        论文链接：https://arxiv.org/abs/1803.05407
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        swa_lr: float,
        anneal_epochs=10,
        anneal_strategy: Literal["cos", "linear"] = "cos",
        last_epoch=-1,
    ):  # noqa: D107
        """
        初始化方法
        
        Args:
            optimizer (Optimizer): 模型优化器
            swa_lr (float): SWA 学习率
            anneal_epochs (int, optional): 衰减周期，默认为 10
            anneal_strategy (Literal["cos", "linear"], optional): 衰减策略，默认为 "cos"
            last_epoch (int, optional): 上一个周期，默认为 -1
            
        Raises:
            ValueError: 如果 anneal_strategy 不是 "cos" 或 "linear"
            ValueError: 如果 anneal_epochs 不是正整数
        """
        # 使用 _format_param 函数为每个参数组设置 swa_lr
        swa_lrs = _format_param("swa_lr", optimizer, swa_lr)
        for swa_lr, group in zip(swa_lrs, optimizer.param_groups):
            group["swa_lr"] = swa_lr
        
        # 检查并设置衰减策略
        if anneal_strategy not in ["cos", "linear"]:
            raise ValueError(
                "anneal_strategy must by one of 'cos' or 'linear', "
                f"instead got {anneal_strategy}"
            )
        elif anneal_strategy == "cos":
            self.anneal_func = self._cosine_anneal
        elif anneal_strategy == "linear":
            self.anneal_func = self._linear_anneal
        
        # 检查并设置衰减周期数
        if not isinstance(anneal_epochs, int) or anneal_epochs < 0:
            raise ValueError(
                f"anneal_epochs must be equal or greater than 0, got {anneal_epochs}"
            )
        self.anneal_epochs = anneal_epochs
        
        # 调用父类的初始化方法
        super().__init__(optimizer, last_epoch)

    @staticmethod
    def _linear_anneal(t):
        """
        线性衰减函数
        
        Args:
            t (float): 当前时间步
        
        Returns:
            float: 衰减后的值
        """
        return t

    @staticmethod
    def _cosine_anneal(t):
        """
        余弦衰减函数
        
        Args:
            t (float): 当前时间步
        
        Returns:
            float: 衰减后的值
        """
        return (1 - math.cos(math.pi * t)) / 2

    @staticmethod
    def _get_initial_lr(lr, swa_lr, alpha):
        """
        计算初始学习率
        
        Args:
            lr (float): 初始学习率
            swa_lr (float): SWA 学习率
            alpha (float): 衰减率
            
        Returns:
            float: 初始学习率
        """
        if alpha == 1:
            return swa_lr
        return (lr - alpha * swa_lr) / (1 - alpha)
    def get_lr(self):
        """获取学习率。"""
        # `_get_lr_called_within_step` 只在 `_enable_get_lr_call` 启用时可用，
        # 因此这里忽略类型错误。详见 `LRScheduler.step()` 获取更多细节。
        if not self._get_lr_called_within_step:  # type: ignore[attr-defined]
            warnings.warn(
                "要获取调度器计算的最后一个学习率，请使用 `get_last_lr()`。",
                UserWarning,
            )
        # 在 `LRScheduler._initial_step()` 中设置
        step = self._step_count - 1  # type: ignore[attr-defined]
        if self.anneal_epochs == 0:
            step = max(1, step)
        prev_t = max(0, min(1, (step - 1) / max(1, self.anneal_epochs)))
        prev_alpha = self.anneal_func(prev_t)
        prev_lrs = [
            self._get_initial_lr(group["lr"], group["swa_lr"], prev_alpha)
            for group in self.optimizer.param_groups
        ]
        t = max(0, min(1, step / max(1, self.anneal_epochs)))
        alpha = self.anneal_func(t)
        # 返回更新后的学习率列表
        return [
            group["swa_lr"] * alpha + lr * (1 - alpha)
            for group, lr in zip(self.optimizer.param_groups, prev_lrs)
        ]
```