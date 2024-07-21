# `.\pytorch\torch\optim\lr_scheduler.py`

```py
# mypy: allow-untyped-defs
r"""Learning Rate Scheduler."""
# 导入需要的模块和类
import math
import types
import warnings
from bisect import bisect_right
from collections import Counter
from functools import partial
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    SupportsFloat,
    TypedDict,
    Union,
)
# 引入弱引用类 ref
from weakref import ref

# 导入 torch 库中的 inf 和 Tensor 类
from torch import inf, Tensor

# 从当前包中导入 optimizer 模块
from .optimizer import Optimizer

# 定义公开的类和函数列表
__all__ = [
    "LambdaLR",
    "MultiplicativeLR",
    "StepLR",
    "MultiStepLR",
    "ConstantLR",
    "LinearLR",
    "ExponentialLR",
    "SequentialLR",
    "CosineAnnealingLR",
    "ChainedScheduler",
    "ReduceLROnPlateau",
    "CyclicLR",
    "CosineAnnealingWarmRestarts",
    "OneCycleLR",
    "PolynomialLR",
    "LRScheduler",
]

# 提示信息，关于 epoch 参数的弃用警告
EPOCH_DEPRECATION_WARNING = (
    "The epoch parameter in `scheduler.step()` was not necessary and is being "
    "deprecated where possible. Please use `scheduler.step()` to step the "
    "scheduler. During the deprecation, if epoch is different from None, the "
    "closed form is used instead of the new chainable form, where available. "
    "Please open an issue if you are unable to replicate your use case: "
    "https://github.com/pytorch/pytorch/issues/new/choose."
)


def _check_verbose_deprecated_warning(verbose):
    """Raise a warning when verbose is not the default value."""
    # 如果 verbose 不是默认值 "deprecated"，则发出警告
    if verbose != "deprecated":
        warnings.warn(
            "The verbose parameter is deprecated. Please use get_last_lr() "
            "to access the learning rate.",
            UserWarning,
        )
        return verbose
    return False


def _format_param(name: str, optimizer: Optimizer, param):
    """Return correctly formatted lr/momentum for each param group."""

    def _copy(_param):
        return _param.clone() if isinstance(_param, Tensor) else _param

    # 检查 param 是否为列表或元组，如果是，则长度必须与 param_groups 相同
    if isinstance(param, (list, tuple)):
        if len(param) != len(optimizer.param_groups):
            raise ValueError(
                f"{name} must have the same length as optimizer.param_groups. "
                f"{name} has {len(param)} values, param_groups has {len(optimizer.param_groups)}."
            )
    else:
        # 如果 param 不是列表或元组，则复制为与 param_groups 长度相同的列表
        param = [param] * len(optimizer.param_groups)

    # 使用 _copy 函数对每个 param 进行处理并返回列表
    return list(map(_copy, param))


class LRScheduler:
    r"""Adjusts the learning rate during optimization."""

    # 初始化时的变量设置
    _get_lr_called_within_step: bool = False

    def __init__(
        self, optimizer: Optimizer, last_epoch=-1, verbose="deprecated"
    ):  # noqa: D107
        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"{type(optimizer).__name__} is not an Optimizer")
        self.optimizer = optimizer

        # Initialize epoch and base learning rates
        if last_epoch == -1:
            for group in optimizer.param_groups:
                initial_lr = group["lr"]
                if isinstance(initial_lr, Tensor):
                    initial_lr = initial_lr.clone()
                group.setdefault("initial_lr", initial_lr)
        else:
            for i, group in enumerate(optimizer.param_groups):
                if "initial_lr" not in group:
                    raise KeyError(
                        "param 'initial_lr' is not specified "
                        f"in param_groups[{i}] when resuming an optimizer"
                    )
        self.base_lrs: List[float] = [
            group["initial_lr"] for group in optimizer.param_groups
        ]
        self.last_epoch = last_epoch

        # Following https://github.com/pytorch/pytorch/issues/20124
        # We would like to ensure that `lr_scheduler.step()` is called after
        # `optimizer.step()`
        def patch_track_step_called(opt: Optimizer):
            if hasattr(opt.step, "_wrapped_by_lr_sched"):
                # we've already patched
                return opt.step

            def wrap_step(step_fn):
                opt_ref = ref(self.optimizer)
                func = step_fn.__func__

                def wrapper(*args, **kwargs):
                    opt = opt_ref()
                    opt._opt_called = True  # type: ignore[union-attr]
                    return func.__get__(opt, opt.__class__)(*args, **kwargs)

                wrapper._wrapped_by_lr_sched = True  # type: ignore[attr-defined]
                return wrapper

            opt.step = wrap_step(opt.step)  # type: ignore[method-assign]

        # Call the function to patch the optimizer's step method
        patch_track_step_called(self.optimizer)

        # Check and set verbose mode, handling deprecation warnings
        self.verbose = _check_verbose_deprecated_warning(verbose)

        # Initialize the step counts and perform an initial step
        self._initial_step()

    def _initial_step(self):
        """Initialize step counts and perform a step."""
        self._step_count = 0
        self.step()

    def state_dict(self):
        """Return the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {
            key: value for key, value in self.__dict__.items() if key != "optimizer"
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load the scheduler's state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict)

    def get_last_lr(self) -> List[float]:
        """Return last computed learning rate by current scheduler."""
        return self._last_lr
    # 返回一个列表，其中包含使用调度器的链式形式计算的学习率
    def get_lr(self) -> List[float]:
        """Compute learning rate using chainable form of the scheduler."""
        # 抛出未实现错误，暂未实现此方法
        raise NotImplementedError

    # 显示当前的学习率
    def print_lr(
        self,
        is_verbose: bool,
        group: Dict[str, Any],
        lr: float,
        epoch: Optional[int] = None,
    ):
        """Display the current learning rate.

        .. deprecated:: 2.4
            ``print_lr()`` is deprecated. Please use ``get_last_lr()`` to access the
            learning rate.
        """
        # 发出警告，提醒用户该方法已被弃用
        warnings.warn(
            "`LRScheduler.print_lr()` is being deprecated. To fetch the learning rate, "
            "please use `get_last_lr()` instead. For more details, "
            "see https://github.com/pytorch/pytorch/issues/99270.",
            UserWarning,
        )
        if is_verbose:
            if epoch is None:
                # 如果没有提供 epoch 参数，则打印调整学习率组到指定学习率的消息
                print(f"Adjusting learning rate of group {group} to {lr:.4e}.")
            else:
                # 根据 epoch 参数的类型打印相应格式的消息，指示调整学习率组到指定学习率
                epoch_str = ("%.2f" if isinstance(epoch, float) else "%.5d") % epoch
                print(
                    f"Epoch {epoch_str}: adjusting learning rate of group {group} to {lr:.4e}."
                )
    def step(self, epoch: Optional[int] = None):
        """Perform a step."""
        # 如果步数为1，检查是否存在老版本的模式
        # https://github.com/pytorch/pytorch/issues/20124
        if self._step_count == 1:
            # 如果 optimizer.step() 没有被 lr_scheduler.step() 包装，则发出警告
            if not hasattr(self.optimizer.step, "_wrapped_by_lr_sched"):
                warnings.warn(
                    "Seems like `optimizer.step()` has been overridden after learning rate scheduler "
                    "initialization. Please, make sure to call `optimizer.step()` before "
                    "`lr_scheduler.step()`. See more details at "
                    "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate",
                    UserWarning,
                )

            # 如果 optimizer.step() 之前没有调用过 lr_scheduler.step()，则发出警告
            elif not getattr(self.optimizer, "_opt_called", False):
                warnings.warn(
                    "Detected call of `lr_scheduler.step()` before `optimizer.step()`. "
                    "In PyTorch 1.1.0 and later, you should call them in the opposite order: "
                    "`optimizer.step()` before `lr_scheduler.step()`.  Failure to do this "
                    "will result in PyTorch skipping the first value of the learning rate schedule. "
                    "See more details at "
                    "https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate",
                    UserWarning,
                )
        
        # 增加步数计数
        self._step_count += 1

        # 启用获取当前学习率的上下文管理器
        with _enable_get_lr_call(self):
            # 如果 epoch 为 None，则递增最后一个 epoch，并获取当前学习率
            if epoch is None:
                self.last_epoch += 1
                values = self.get_lr()
            else:
                # 发出关于 epoch 弃用的警告
                warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
                self.last_epoch = epoch
                # 如果存在 _get_closed_form_lr 方法，则使用其结果，否则获取当前学习率
                if hasattr(self, "_get_closed_form_lr"):
                    values = cast(List[float], self._get_closed_form_lr())
                else:
                    values = self.get_lr()

        # 更新每个参数组的学习率
        for i, data in enumerate(zip(self.optimizer.param_groups, values)):
            param_group, lr = data
            if isinstance(param_group["lr"], Tensor):
                # 如果学习率是 Tensor 类型，更新为 lr 的值
                lr_val = lr.item() if isinstance(lr, Tensor) else lr  # type: ignore[attr-defined]
                param_group["lr"].fill_(lr_val)
            else:
                # 否则，直接更新为 lr 的值
                param_group["lr"] = lr

        # 更新最后一个学习率的列表
        self._last_lr: List[float] = [
            group["lr"] for group in self.optimizer.param_groups
        ]
# 当 lr_scheduler._get_lr_called_within_step 为 False 时发出警告信息
def _warn_get_lr_called_within_step(lr_scheduler: LRScheduler):
    if not lr_scheduler._get_lr_called_within_step:
        warnings.warn(
            "To get the last learning rate computed by the scheduler, "
            "please use `get_last_lr()`.",
            UserWarning,
            stacklevel=2,
        )

# 包含 _LRScheduler 以保持向后兼容性
# 使用子类化而不是赋值的方式，因为我们希望 _LRScheduler 的 __name__ 保持 _LRScheduler（赋值会使其变为 LRScheduler）。
class _LRScheduler(LRScheduler):
    pass

class _enable_get_lr_call:
    def __init__(self, o: LRScheduler):
        self.o = o

    # 进入上下文时将 o._get_lr_called_within_step 设置为 True
    def __enter__(self):
        self.o._get_lr_called_within_step = True
        return self

    # 离开上下文时将 o._get_lr_called_within_step 设置为 False
    def __exit__(self, type, value, traceback):
        self.o._get_lr_called_within_step = False

class LambdaLR(LRScheduler):
    """设置初始学习率。

    每个参数组的学习率设置为初始 lr 乘以给定的函数。当 last_epoch=-1 时，将初始 lr 设置为 lr。

    Args:
        optimizer (Optimizer): 封装的优化器。
        lr_lambda (function or list): 给定一个整数参数 epoch，计算乘法因子的函数，或者是这样的函数列表，每个组在 optimizer.param_groups 中有一个。
        last_epoch (int): 上一个 epoch 的索引。默认值：-1。
        verbose (bool | str): 如果 ``True``，则对每次更新打印一条消息到标准输出。默认值：``False``。

            .. deprecated:: 2.2
                ``verbose`` 已被弃用。请使用 ``get_last_lr()`` 来获取学习率。

    Example:
        >>> # xdoctest: +SKIP
        >>> # 假设优化器有两个组。
        >>> lambda1 = lambda epoch: epoch // 30
        >>> lambda2 = lambda epoch: 0.95 ** epoch
        >>> scheduler = LambdaLR(optimizer, lr_lambda=[lambda1, lambda2])
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(
        self,
        optimizer: Optimizer,
        lr_lambda: Union[Callable[[int], float], List[Callable[[int], float]]],
        last_epoch=-1,
        verbose="deprecated",
    ):  # noqa: D107
        self.optimizer = optimizer

        # 如果 lr_lambda 不是列表或元组，则将其复制成 optimizer.param_groups 的长度
        self.lr_lambdas: List[Callable[[int], float]]
        if not isinstance(lr_lambda, list) and not isinstance(lr_lambda, tuple):
            self.lr_lambdas = [lr_lambda] * len(optimizer.param_groups)
        else:
            if len(lr_lambda) != len(optimizer.param_groups):
                raise ValueError(
                    f"Expected {len(optimizer.param_groups)} lr_lambdas, but got {len(lr_lambda)}"
                )
            self.lr_lambdas = list(lr_lambda)
        super().__init__(optimizer, last_epoch, verbose)
    def state_dict(self):
        """Return the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        The learning rate lambda functions will only be saved if they are callable objects
        and not if they are functions or lambdas.

        When saving or loading the scheduler, please make sure to also save or load the state of the optimizer.
        """
        # 创建一个空的状态字典
        state_dict = {
            key: value
            for key, value in self.__dict__.items()
            if key not in ("optimizer", "lr_lambdas")
        }
        # 将 lr_lambdas 初始化为与其长度相同的空列表
        state_dict["lr_lambdas"] = [None] * len(self.lr_lambdas)

        # 遍历 lr_lambdas 列表中的每个函数对象，将可调用对象的字典形式保存到 state_dict 中
        for idx, fn in enumerate(self.lr_lambdas):
            if not isinstance(fn, types.FunctionType):
                state_dict["lr_lambdas"][idx] = fn.__dict__.copy()

        # 返回状态字典
        return state_dict

    def load_state_dict(self, state_dict):
        """Load the scheduler's state.

        When saving or loading the scheduler, please make sure to also save or load the state of the optimizer.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # 从 state_dict 中取出 lr_lambdas
        lr_lambdas = state_dict.pop("lr_lambdas")
        # 更新调度器对象的状态，排除了 lr_lambdas
        self.__dict__.update(state_dict)
        # 恢复 state_dict 中的 lr_lambdas 键，以防止副作用
        # https://github.com/pytorch/pytorch/issues/32756
        state_dict["lr_lambdas"] = lr_lambdas

        # 遍历 lr_lambdas 列表中的每个函数对象，将其字典形式的内容更新回原始的 lr_lambdas 中的对象
        for idx, fn in enumerate(lr_lambdas):
            if fn is not None:
                self.lr_lambdas[idx].__dict__.update(fn)

    def get_lr(self):
        """Compute learning rate."""
        # 警告函数，提示在 step 函数内部调用 get_lr
        _warn_get_lr_called_within_step(self)

        # 返回基础学习率与对应 lambda 函数在当前 epoch 下的计算结果的乘积列表
        return [
            base_lr * lmbda(self.last_epoch)
            for lmbda, base_lr in zip(self.lr_lambdas, self.base_lrs)
        ]
class MultiplicativeLR(LRScheduler):
    """Multiply the learning rate of each parameter group by the factor given in the specified function.

    When last_epoch=-1, set initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        lr_lambda (function or list): A function which computes a multiplicative
            factor given an integer parameter epoch, or a list of such
            functions, one for each group in optimizer.param_groups.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool | str): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

            .. deprecated:: 2.2
                ``verbose`` is deprecated. Please use ``get_last_lr()`` to access the
                learning rate.

    Example:
        >>> # xdoctest: +SKIP
        >>> lmbda = lambda epoch: 0.95
        >>> scheduler = MultiplicativeLR(optimizer, lr_lambda=lmbda)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(
        self,
        optimizer: Optimizer,
        lr_lambda: Union[Callable[[int], float], List[Callable[[int], float]]],
        last_epoch=-1,
        verbose="deprecated",
    ):  # noqa: D107
        # 初始化函数，设置优化器对象
        self.optimizer = optimizer

        # 根据传入的 lr_lambda 参数类型，确定学习率 lambda 函数列表
        self.lr_lambdas: List[Callable[[int], float]]
        if not isinstance(lr_lambda, list) and not isinstance(lr_lambda, tuple):
            # 如果 lr_lambda 是单个函数，则复制到每个参数组
            self.lr_lambdas = [lr_lambda] * len(optimizer.param_groups)
        else:
            # 如果 lr_lambda 是列表或元组，则检查长度是否与参数组数量一致
            if len(lr_lambda) != len(optimizer.param_groups):
                raise ValueError(
                    f"Expected {len(optimizer.param_groups)} lr_lambdas, but got {len(lr_lambda)}"
                )
            self.lr_lambdas = list(lr_lambda)

        # 调用父类的初始化方法
        super().__init__(optimizer, last_epoch, verbose)

    def state_dict(self):
        """Return the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        The learning rate lambda functions will only be saved if they are callable objects
        and not if they are functions or lambdas.
        """
        # 创建一个状态字典，包含除了优化器和 lr_lambdas 之外的所有 self.__dict__ 变量
        state_dict = {
            key: value
            for key, value in self.__dict__.items()
            if key not in ("optimizer", "lr_lambdas")
        }
        state_dict["lr_lambdas"] = [None] * len(self.lr_lambdas)

        # 将每个学习率 lambda 函数保存到状态字典中（只有当它们是可调用对象而不是函数或 lambda 时才保存）
        for idx, fn in enumerate(self.lr_lambdas):
            if not isinstance(fn, types.FunctionType):
                state_dict["lr_lambdas"][idx] = fn.__dict__.copy()

        return state_dict
    def load_state_dict(self, state_dict):
        """Load the scheduler's state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # 从 state_dict 中取出 lr_lambdas，并移除该键
        lr_lambdas = state_dict.pop("lr_lambdas")
        # 使用 state_dict 更新当前对象的属性
        self.__dict__.update(state_dict)
        # 恢复 lr_lambdas 到 state_dict 中，以防止副作用
        # 参考：https://github.com/pytorch/pytorch/issues/32756
        state_dict["lr_lambdas"] = lr_lambdas

        # 遍历 lr_lambdas 列表的每个元素
        for idx, fn in enumerate(lr_lambdas):
            # 如果 fn 不为 None，则更新 self.lr_lambdas[idx] 的属性
            if fn is not None:
                self.lr_lambdas[idx].__dict__.update(fn)

    def get_lr(self):
        """Compute the learning rate of each parameter group."""
        # 警告：在 step 函数内调用 get_lr 函数
        _warn_get_lr_called_within_step(self)

        # 如果当前 epoch 大于 0
        if self.last_epoch > 0:
            # 返回每个参数组的学习率乘以对应的 lr_lambdas 函数在当前 epoch 的返回值
            return [
                group["lr"] * lmbda(self.last_epoch)
                for lmbda, group in zip(self.lr_lambdas, self.optimizer.param_groups)
            ]
        else:
            # 返回每个参数组的学习率
            return [group["lr"] for group in self.optimizer.param_groups]
# 定义一个学习率调度器 StepLR，继承自 LRScheduler 类
class StepLR(LRScheduler):
    """Decays the learning rate of each parameter group by gamma every step_size epochs.

    Notice that such decay can happen simultaneously with other changes to the learning rate
    from outside this scheduler. When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.  # 优化器对象，被包装后的优化器
        step_size (int): Period of learning rate decay.  # 学习率衰减的周期
        gamma (float): Multiplicative factor of learning rate decay.
            Default: 0.1.  # 学习率衰减的乘法因子，默认为 0.1
        last_epoch (int): The index of last epoch. Default: -1.  # 最后一个 epoch 的索引，默认为 -1
        verbose (bool | str): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

            .. deprecated:: 2.2
                ``verbose`` is deprecated. Please use ``get_last_lr()`` to access the
                learning rate.
                # verbose 参数已经废弃，请使用 get_last_lr() 方法访问学习率

    Example:
        >>> # xdoctest: +SKIP
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.05     if epoch < 30
        >>> # lr = 0.005    if 30 <= epoch < 60
        >>> # lr = 0.0005   if 60 <= epoch < 90
        >>> # ...
        >>> scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
        # 示例代码，展示了如何使用 StepLR 调度器来调整学习率

    def __init__(
        self,
        optimizer: Optimizer,
        step_size: int,
        gamma=0.1,
        last_epoch=-1,
        verbose="deprecated",
    ):  # noqa: D107
        # 初始化函数，设置学习率衰减的步长 step_size、乘法因子 gamma
        self.step_size = step_size
        self.gamma = gamma
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        """Compute the learning rate of each parameter group."""
        _warn_get_lr_called_within_step(self)
        # 计算每个参数组的学习率

        # 如果是第一个 epoch 或者当前 epoch 不是 step_size 的整数倍
        if (self.last_epoch == 0) or (self.last_epoch % self.step_size != 0):
            return [group["lr"] for group in self.optimizer.param_groups]
            # 返回当前优化器所有参数组的学习率列表
        return [group["lr"] * self.gamma for group in self.optimizer.param_groups]
        # 否则返回每个参数组学习率乘以 gamma 后的列表

    def _get_closed_form_lr(self):
        # 返回一个列表，包含每个参数组基础学习率乘以 gamma 的 last_epoch // step_size 次方
        return [
            base_lr * self.gamma ** (self.last_epoch // self.step_size)
            for base_lr in self.base_lrs
        ]


class MultiStepLR(LRScheduler):
    """Decays the learning rate of each parameter group by gamma once the number of epoch reaches one of the milestones.

    Notice that such decay can happen simultaneously with other changes to the learning rate
    from outside this scheduler. When last_epoch=-1, sets initial lr as lr.
    """
    # 多步学习率调度器，当 epoch 数到达里程碑时，每个参数组的学习率按 gamma 衰减
    # 注意，此类衰减可以与外部对学习率的其他更改同时发生。当 last_epoch=-1 时，初始 lr 为 lr。
    def __init__(
        self,
        optimizer: Optimizer,
        milestones: Iterable[int],
        gamma=0.1,
        last_epoch=-1,
        verbose="deprecated",
    ):  # noqa: D107
        """
        初始化方法，设置学习率调度器的参数。

        Args:
            optimizer (Optimizer): 被包装的优化器对象。
            milestones (Iterable[int]): 里程碑的列表，必须是递增的。
            gamma (float): 学习率衰减的乘法因子，默认为 0.1。
            last_epoch (int): 上一个 epoch 的索引，默认为 -1。
            verbose (str): 冗余参数，已弃用。

                .. deprecated:: 2.2
                    `verbose` 已弃用，请使用 `get_last_lr()` 来访问学习率。
        """
        self.milestones = Counter(milestones)
        self.gamma = gamma
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        """
        计算每个参数组的学习率。

        Returns:
            List[float]: 每个参数组的学习率列表。
        """
        _warn_get_lr_called_within_step(self)

        if self.last_epoch not in self.milestones:
            return [group["lr"] for group in self.optimizer.param_groups]
        return [
            group["lr"] * self.gamma ** self.milestones[self.last_epoch]
            for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self):
        """
        计算闭式形式下的学习率。

        Returns:
            List[float]: 每个参数组的学习率列表。
        """
        milestones = sorted(self.milestones.elements())
        return [
            base_lr * self.gamma ** bisect_right(milestones, self.last_epoch)
            for base_lr in self.base_lrs
        ]
class ConstantLR(LRScheduler):
    """Multiply the learning rate of each parameter group by a small constant factor.

    The multiplication is done until the number of epoch reaches a pre-defined milestone: total_iters.
    Notice that such multiplication of the small constant factor can
    happen simultaneously with other changes to the learning rate from outside this scheduler.
    When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        factor (float): The number we multiply learning rate until the milestone. Default: 1./3.
        total_iters (int): The number of steps that the scheduler multiplies the learning rate by the factor.
            Default: 5.
        last_epoch (int): The index of the last epoch. Default: -1.
        verbose (bool | str): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

            .. deprecated:: 2.2
                ``verbose`` is deprecated. Please use ``get_last_lr()`` to access the
                learning rate.

    Example:
        >>> # xdoctest: +SKIP
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.025   if epoch == 0
        >>> # lr = 0.025   if epoch == 1
        >>> # lr = 0.025   if epoch == 2
        >>> # lr = 0.025   if epoch == 3
        >>> # lr = 0.05    if epoch >= 4
        >>> scheduler = ConstantLR(optimizer, factor=0.5, total_iters=4)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(
        self,
        optimizer: Optimizer,
        factor=1.0 / 3,
        total_iters=5,
        last_epoch=-1,
        verbose="deprecated",
    ):  # noqa: D107
        """
        Initialize ConstantLR scheduler.

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            factor (float): Multiplicative factor applied to learning rate. Default: 1./3.
            total_iters (int): Number of steps to apply the factor. Default: 5.
            last_epoch (int): Index of the last epoch. Default: -1.
            verbose (bool | str): Verbosity mode. Default: "deprecated".
        """
        if factor > 1.0 or factor < 0:
            raise ValueError(
                "Constant multiplicative factor expected to be between 0 and 1."
            )

        self.factor = factor
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        """Compute the learning rate of each parameter group."""
        _warn_get_lr_called_within_step(self)

        if self.last_epoch == 0:
            # Multiply the initial learning rate by the factor for the first epoch
            return [group["lr"] * self.factor for group in self.optimizer.param_groups]

        if self.last_epoch != self.total_iters:
            # Keep the current learning rate unchanged for epochs between 1 and total_iters - 1
            return [group["lr"] for group in self.optimizer.param_groups]

        # In the final epoch (total_iters), revert the learning rate by dividing by the factor
        return [
            group["lr"] * (1.0 / self.factor) for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self):
        """
        Compute learning rates based on a closed-form expression.

        Returns:
            list: Updated learning rates for each parameter group.
        """
        return [
            base_lr
            * (self.factor + (self.last_epoch >= self.total_iters) * (1 - self.factor))
            for base_lr in self.base_lrs
        ]


class LinearLR(LRScheduler):
    """Decays the learning rate of each parameter group by linearly changing small multiplicative factor.
    
    This class is currently incomplete and needs further implementation.
    """
    """
    The `LinearLR` scheduler adjusts the learning rate linearly over a specified number of epochs (`total_iters`).
    
    The multiplication is done until the number of epoch reaches a pre-defined milestone: total_iters.
    Notice that such decay can happen simultaneously with other changes to the learning rate
    from outside this scheduler. When last_epoch=-1, sets initial lr as lr.
    
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        start_factor (float): The number we multiply learning rate in the first epoch.
            The multiplication factor changes towards `end_factor` in the following epochs.
            Default: 1./3.
        end_factor (float): The number we multiply learning rate at the end of linear changing
            process. Default: 1.0.
        total_iters (int): The number of iterations that multiplicative factor reaches to 1.
            Default: 5.
        last_epoch (int): The index of the last epoch. Default: -1.
        verbose (bool | str): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    
            .. deprecated:: 2.2
                ``verbose`` is deprecated. Please use ``get_last_lr()`` to access the
                learning rate.
    
    Example:
        >>> # xdoctest: +SKIP
        >>> # Assuming optimizer uses lr = 0.05 for all groups
        >>> # lr = 0.025    if epoch == 0
        >>> # lr = 0.03125  if epoch == 1
        >>> # lr = 0.0375   if epoch == 2
        >>> # lr = 0.04375  if epoch == 3
        >>> # lr = 0.05    if epoch >= 4
        >>> scheduler = LinearLR(optimizer, start_factor=0.5, total_iters=4)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """
    def __init__(
        self,
        optimizer: Optimizer,
        start_factor=1.0 / 3,
        end_factor=1.0,
        total_iters=5,
        last_epoch=-1,
        verbose="deprecated",
    ):  # noqa: D107
        """
        Initialize the LinearLR scheduler with specified parameters.
    
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            start_factor (float): The initial multiplicative factor for the learning rate.
                It should be greater than 0 and less or equal to 1.
            end_factor (float): The final multiplicative factor for the learning rate.
                It should be between 0 and 1.
            total_iters (int): The total number of iterations over which the multiplicative
                factor reaches 1.
            last_epoch (int): The index of the last epoch. Default is -1.
            verbose (bool | str): Deprecated option for printing messages to stdout.
    
        Raises:
            ValueError: If `start_factor` or `end_factor` are outside their expected ranges.
    
        Notes:
            This scheduler adjusts the learning rate linearly from `start_factor` to `end_factor`
            over `total_iters` iterations.
        """
        if start_factor > 1.0 or start_factor <= 0:
            raise ValueError(
                "Starting multiplicative factor expected to be greater than 0 and less or equal to 1."
            )
    
        if end_factor > 1.0 or end_factor < 0:
            raise ValueError(
                "Ending multiplicative factor expected to be between 0 and 1."
            )
    
        # Initialize with provided parameters
        self.start_factor = start_factor
        self.end_factor = end_factor
        self.total_iters = total_iters
    
        # Call superclass constructor to initialize the optimizer and other inherited attributes
        super().__init__(optimizer, last_epoch, verbose)
    # 计算学习率的函数
    def get_lr(self):
        """Compute the learning rate."""
        # 调用警告函数，提示在 step 方法中调用了 get_lr 函数
        _warn_get_lr_called_within_step(self)

        # 如果当前 epoch 是第一次迭代，则返回每个参数组的起始学习率乘以起始因子
        if self.last_epoch == 0:
            return [
                group["lr"] * self.start_factor for group in self.optimizer.param_groups
            ]

        # 如果当前 epoch 大于总迭代次数，则返回每个参数组的当前学习率
        if self.last_epoch > self.total_iters:
            return [group["lr"] for group in self.optimizer.param_groups]

        # 计算当前 epoch 对应的学习率，使用线性插值计算方式
        return [
            group["lr"]
            * (
                1.0
                + (self.end_factor - self.start_factor)
                / (
                    self.total_iters * self.start_factor
                    + (self.last_epoch - 1) * (self.end_factor - self.start_factor)
                )
            )
            for group in self.optimizer.param_groups
        ]

    # 计算封闭形式的学习率函数
    def _get_closed_form_lr(self):
        # 对每个基础学习率进行计算，使用封闭形式的计算方式
        return [
            base_lr
            * (
                self.start_factor
                + (self.end_factor - self.start_factor)
                * min(self.total_iters, self.last_epoch)
                / self.total_iters
            )
            for base_lr in self.base_lrs
        ]
class ExponentialLR(LRScheduler):
    """Decays the learning rate of each parameter group by gamma every epoch.

    When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        gamma (float): Multiplicative factor of learning rate decay.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool | str): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

            .. deprecated:: 2.2
                ``verbose`` is deprecated. Please use ``get_last_lr()`` to access the
                learning rate.
    """

    def __init__(
        self, optimizer: Optimizer, gamma: float, last_epoch=-1, verbose="deprecated"
    ):  # noqa: D107
        # 初始化 ExponentialLR 类
        self.gamma = gamma
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        """Compute the learning rate of each parameter group."""
        _warn_get_lr_called_within_step(self)

        # 如果当前是第一个 epoch，返回每个参数组的当前学习率
        if self.last_epoch == 0:
            return [group["lr"] for group in self.optimizer.param_groups]
        # 否则，根据 gamma 计算每个参数组的新学习率
        return [group["lr"] * self.gamma for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        # 返回一个列表，其中每个学习率按指数衰减公式计算
        return [base_lr * self.gamma**self.last_epoch for base_lr in self.base_lrs]


class SequentialLR(LRScheduler):
    """Contains a list of schedulers expected to be called sequentially during the optimization process.

    Specifically, the schedulers will be called according to the milestone points, which should provide exact
    intervals by which each scheduler should be called at a given epoch.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        schedulers (list): List of chained schedulers.
        milestones (list): List of integers that reflects milestone points.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool | str): Does nothing.

            .. deprecated:: 2.2
                ``verbose`` is deprecated. Please use ``get_last_lr()`` to access the
                learning rate.

    Example:
        >>> # xdoctest: +SKIP
        >>> # Assuming optimizer uses lr = 1. for all groups
        >>> # lr = 0.1     if epoch == 0
        >>> # lr = 0.1     if epoch == 1
        >>> # lr = 0.9     if epoch == 2
        >>> # lr = 0.81    if epoch == 3
        >>> # lr = 0.729   if epoch == 4
        >>> scheduler1 = ConstantLR(optimizer, factor=0.1, total_iters=2)
        >>> scheduler2 = ExponentialLR(optimizer, gamma=0.9)
        >>> scheduler = SequentialLR(optimizer, schedulers=[scheduler1, scheduler2], milestones=[2])
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(
        self,
        optimizer: Optimizer,
        schedulers: List[LRScheduler],
        milestones: List[int],
        last_epoch=-1,
        verbose="deprecated",
    ):
        # 初始化 SequentialLR 类
        self.schedulers = schedulers
        self.milestones = milestones
        super().__init__(optimizer, last_epoch, verbose)
    ):  # noqa: D107
        # 如果提供的调度器列表为空，则抛出数值错误异常
        if len(schedulers) < 1:
            raise ValueError(
                f"{self.__class__.__name__} expects at least one scheduler, but got no scheduler."
            )

        # 遍历调度器列表，检查每个调度器是否具有 `optimizer` 属性
        for scheduler_idx, scheduler in enumerate(schedulers):
            if not hasattr(scheduler, "optimizer"):
                raise TypeError(
                    f"{self.__class__.__name__} at index {scheduler_idx} should have `optimizer` as its attribute."
                )
            # 如果调度器是 `ReduceLROnPlateau` 类型，则抛出数值错误异常
            if isinstance(scheduler, ReduceLROnPlateau):
                raise ValueError(
                    f"{self.__class__.__name__} does not support `ReduceLROnPlateau` scheduler as it "
                    "requires additional kwargs to be specified when calling `step`, "
                    f"but got one at index {scheduler_idx} in the given schedulers sequence."
                )
            # 如果调度器的优化器与主优化器不匹配，则抛出数值错误异常
            if optimizer != scheduler.optimizer:
                raise ValueError(
                    f"{self.__class__.__name__} expects all schedulers to belong to the same optimizer, but "
                    f"got scheduler {scheduler.__class__.__name__} at index {scheduler_idx} has {scheduler.optimizer}, "
                    f"which is different from {optimizer.__class__.__name__}."
                )

        # 如果提供的里程碑列表长度与调度器数量不符合预期，则抛出数值错误异常
        if len(milestones) != len(schedulers) - 1:
            raise ValueError(
                "Sequential Schedulers expects number of schedulers provided to be one more "
                f"than the number of milestone points, but got number of schedulers {len(schedulers)} and the "
                f"number of milestones to be equal to {len(milestones)}"
            )

        # 检查并提示关于 verbose 参数过时警告
        _check_verbose_deprecated_warning(verbose)

        # 将传入的调度器列表和里程碑列表分别赋值给对象属性
        self._schedulers = schedulers
        self._milestones = milestones

        # 将 last_epoch 初始化为提供的 last_epoch 参数值加一
        self.last_epoch = last_epoch + 1

        # 将主优化器赋值给对象属性 optimizer
        self.optimizer = optimizer

        # 将所有参数组的学习率重置为初始学习率
        for group in self.optimizer.param_groups:
            group["lr"] = group["initial_lr"]

        # 将所有调度器的 last_epoch 属性减一，以撤销上次的步骤
        for scheduler in self._schedulers:
            scheduler.last_epoch -= 1

        # 对第一个调度器执行初始步骤
        self._schedulers[0]._initial_step()

        # 获取第一个调度器的最后学习率并赋值给对象属性 _last_lr
        self._last_lr = schedulers[0].get_last_lr()

    def step(self):
        """Perform a step."""
        # 每调用一次 step 方法，将 last_epoch 属性加一
        self.last_epoch += 1

        # 使用二分查找找到当前 epoch 对应的调度器索引
        idx = bisect_right(self._milestones, self.last_epoch)

        # 获取当前 epoch 对应的调度器对象
        scheduler = self._schedulers[idx]

        # 如果当前 epoch 是一个里程碑点，调用调度器的 step 方法并传入参数 0
        if idx > 0 and self._milestones[idx - 1] == self.last_epoch:
            scheduler.step(0)
        else:
            # 否则，调用调度器的 step 方法
            scheduler.step()

        # 获取当前调度器的最后学习率并赋值给对象属性 _last_lr
        self._last_lr = scheduler.get_last_lr()
    def state_dict(self):
        """
        Return the state of the scheduler as a :class:`dict`.

        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        The wrapped scheduler states will also be saved.
        """
        # 创建一个空字典 state_dict 来保存调度器的状态信息
        state_dict = {
            key: value
            for key, value in self.__dict__.items()
            if key not in ("optimizer", "_schedulers")
        }
        # 初始化 _schedulers 列表为与 self._schedulers 相同长度的空列表
        state_dict["_schedulers"] = [None] * len(self._schedulers)

        # 遍历 self._schedulers 列表，保存每个调度器的状态到 state_dict["_schedulers"]
        for idx, s in enumerate(self._schedulers):
            state_dict["_schedulers"][idx] = s.state_dict()

        # 返回保存完整状态信息的 state_dict
        return state_dict

    def load_state_dict(self, state_dict):
        """
        Load the scheduler's state.

        Args:
            state_dict (dict): scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        # 从 state_dict 中弹出 _schedulers 列表并保存到 _schedulers 变量
        _schedulers = state_dict.pop("_schedulers")
        # 更新调度器对象的 __dict__，将 state_dict 中的所有项加载进来
        self.__dict__.update(state_dict)
        # 恢复 _schedulers 到 state_dict 以避免副作用
        # https://github.com/pytorch/pytorch/issues/32756
        state_dict["_schedulers"] = _schedulers

        # 遍历 _schedulers 列表，使用每个项的 load_state_dict 方法加载状态
        for idx, s in enumerate(_schedulers):
            self._schedulers[idx].load_state_dict(s)
class PolynomialLR(LRScheduler):
    """Decays the learning rate of each parameter group using a polynomial function in the given total_iters.

    When last_epoch=-1, sets initial lr as lr.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        total_iters (int): The number of steps that the scheduler decays the learning rate. Default: 5.
        power (float): The power of the polynomial. Default: 1.0.
        verbose (bool | str): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

            .. deprecated:: 2.2
                ``verbose`` is deprecated. Please use ``get_last_lr()`` to access the
                learning rate.

    Example:
        >>> # xdoctest: +SKIP("undefined vars")
        >>> # Assuming optimizer uses lr = 0.001 for all groups
        >>> # lr = 0.001     if epoch == 0
        >>> # lr = 0.00075   if epoch == 1
        >>> # lr = 0.00050   if epoch == 2
        >>> # lr = 0.00025   if epoch == 3
        >>> # lr = 0.0       if epoch >= 4
        >>> scheduler = PolynomialLR(optimizer, total_iters=4, power=1.0)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(
        self,
        optimizer: Optimizer,
        total_iters=5,
        power=1.0,
        last_epoch=-1,
        verbose="deprecated",
    ):  # noqa: D107
        # 初始化多项式学习率调度器
        self.total_iters = total_iters  # 设置总的迭代次数
        self.power = power  # 设置多项式的幂次
        super().__init__(optimizer, last_epoch, verbose)  # 调用父类构造函数进行初始化

    def get_lr(self):
        """Compute the learning rate."""
        _warn_get_lr_called_within_step(self)  # 调用警告函数，提示在step函数内调用get_lr()

        if self.last_epoch == 0 or self.last_epoch > self.total_iters:
            return [group["lr"] for group in self.optimizer.param_groups]

        decay_factor = (
            (1.0 - self.last_epoch / self.total_iters)
            / (1.0 - (self.last_epoch - 1) / self.total_iters)
        ) ** self.power  # 计算学习率衰减因子
        return [group["lr"] * decay_factor for group in self.optimizer.param_groups]  # 返回每个参数组的学习率列表

    def _get_closed_form_lr(self):
        # 计算闭式解下的学习率
        return [
            (
                base_lr
                * (1.0 - min(self.total_iters, self.last_epoch) / self.total_iters)
                ** self.power
            )
            for base_lr in self.base_lrs
        ]


class CosineAnnealingLR(LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing schedule.

    The :math:`\eta_{max}` is set to the initial lr and
    :math:`T_{cur}` is the number of epochs since the last restart in SGDR:
    """
    When last_epoch=-1, sets initial lr as lr. Notice that because the schedule
    is defined recursively, the learning rate can be simultaneously modified
    outside this scheduler by other operators. If the learning rate is set
    solely by this scheduler, the learning rate at each step becomes:
    
    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{max}}\pi\right)\right)
    
    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_. Note that this only
    implements the cosine annealing part of SGDR, and not the restarts.
    
    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_max (int): Maximum number of iterations.
        eta_min (float): Minimum learning rate. Default: 0.
        last_epoch (int): The index of last epoch. Default: -1.
        verbose (bool | str): If ``True``, prints a message to stdout for
            each update. Default: ``False``.
    
            .. deprecated:: 2.2
                ``verbose`` is deprecated. Please use ``get_last_lr()`` to access the
                learning rate.
    
    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """
    
    def __init__(
        self,
        optimizer: Optimizer,
        T_max: int,
        eta_min=0,
        last_epoch=-1,
        verbose="deprecated",
    ):  # noqa: D107
        """
        Initialize the Cosine Annealing with Warm Restarts scheduler.
    
        Args:
            optimizer (Optimizer): Wrapped optimizer.
            T_max (int): Maximum number of iterations.
            eta_min (float): Minimum learning rate. Default: 0.
            last_epoch (int): The index of last epoch. Default: -1.
            verbose (bool | str): If ``True``, prints a message to stdout for
                each update. Default: ``False``.
    
                .. deprecated:: 2.2
                    ``verbose`` is deprecated. Please use ``get_last_lr()`` to access the
                    learning rate.
        """
        self.T_max = T_max  # 设置最大迭代次数
        self.eta_min = eta_min  # 设置最小学习率
        super().__init__(optimizer, last_epoch, verbose)  # 调用父类初始化方法
    # 获取每个参数组的学习率
    def get_lr(self):
        """Retrieve the learning rate of each parameter group."""
        # 调用一个警告函数，用于检查在步骤内调用获取学习率的情况
        _warn_get_lr_called_within_step(self)

        # 如果当前是第一个 epoch
        if self.last_epoch == 0:
            # 返回每个参数组的学习率列表
            return [group["lr"] for group in self.optimizer.param_groups]
        # 如果当前步骤计数为1且不是第一个 epoch
        elif self._step_count == 1 and self.last_epoch > 0:
            # 根据余弦退火公式计算学习率列表
            return [
                self.eta_min
                + (base_lr - self.eta_min)
                * (1 + math.cos((self.last_epoch) * math.pi / self.T_max))
                / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        # 如果上一个 epoch 的索引满足余弦退火重启条件
        elif (self.last_epoch - 1 - self.T_max) % (2 * self.T_max) == 0:
            # 根据余弦退火公式计算学习率列表
            return [
                group["lr"]
                + (base_lr - self.eta_min) * (1 - math.cos(math.pi / self.T_max)) / 2
                for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups)
            ]
        
        # 默认情况下，根据余弦退火公式计算学习率列表
        return [
            (1 + math.cos(math.pi * self.last_epoch / self.T_max))
            / (1 + math.cos(math.pi * (self.last_epoch - 1) / self.T_max))
            * (group["lr"] - self.eta_min)
            + self.eta_min
            for group in self.optimizer.param_groups
        ]

    # 获取闭式形式的学习率列表
    def _get_closed_form_lr(self):
        return [
            self.eta_min
            + (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * self.last_epoch / self.T_max))
            / 2
            for base_lr in self.base_lrs
        ]
class ChainedScheduler(LRScheduler):
    """Chains a list of learning rate schedulers.

    Takes in a sequence of chainable learning rate schedulers and calls their
    step() functions consecutively in just one call to step().

    Args:
        schedulers (sequence): sequence of chained schedulers.
        optimizer (Optimizer, optional): Wrapped optimizer. Default: None.

    Example:
        >>> # xdoctest: +SKIP
        >>> # Assuming optimizer uses lr = 1. for all groups
        >>> # lr = 0.09     if epoch == 0
        >>> # lr = 0.081    if epoch == 1
        >>> # lr = 0.729    if epoch == 2
        >>> # lr = 0.6561   if epoch == 3
        >>> # lr = 0.59049  if epoch >= 4
        >>> scheduler1 = ConstantLR(optimizer, factor=0.1, total_iters=2)
        >>> scheduler2 = ExponentialLR(optimizer, gamma=0.9)
        >>> scheduler = ChainedScheduler([scheduler1, scheduler2], optimizer=optimizer)
        >>> for epoch in range(100):
        >>>     train(...)
        >>>     validate(...)
        >>>     scheduler.step()
    """

    def __init__(
        self, schedulers: Sequence[LRScheduler], optimizer: Optional[Optimizer] = None
    ):  # noqa: D107
        """
        Initialize the ChainedScheduler instance.

        Args:
            schedulers (Sequence[LRScheduler]): List of chained learning rate schedulers.
            optimizer (Optimizer, optional): Wrapped optimizer. Defaults to None.

        Raises:
            ValueError: If no schedulers are provided.
            TypeError: If any scheduler does not have 'optimizer' as an attribute.
            ValueError: If ReduceLROnPlateau scheduler is included, which is not supported.
            ValueError: If schedulers belong to different optimizers.
        """
        if len(schedulers) < 1:
            raise ValueError(
                f"{self.__class__.__name__} expects at least one scheduler to be chained, but got no scheduler."
            )

        optimizer = optimizer or schedulers[0].optimizer
        for scheduler_idx, scheduler in enumerate(schedulers):
            if not hasattr(scheduler, "optimizer"):
                raise TypeError(
                    f"{self.__class__.__name__} at index {scheduler_idx} should have `optimizer` as its attribute."
                )
            if isinstance(scheduler, ReduceLROnPlateau):
                raise ValueError(
                    f"{self.__class__.__name__} does not support `ReduceLROnPlateau` scheduler as it "
                    "requires additional kwargs to be specified when calling `step`, "
                    f"but got one at index {scheduler_idx} in the given schedulers sequence."
                )
            if optimizer != scheduler.optimizer:
                raise ValueError(
                    f"{self.__class__.__name__} expects all schedulers to belong to the same optimizer, but "
                    f"got scheduler {scheduler.__class__.__name__} at index {scheduler_idx} has {scheduler.optimizer}, "
                    f"which is different from {optimizer.__class__.__name__}."
                )
        self._schedulers = schedulers
        self.optimizer = optimizer
        self._last_lr = [
            group["lr"] for group in self._schedulers[-1].optimizer.param_groups
        ]

    def step(self):
        """Perform a step by calling step() on each chained scheduler."""
        for scheduler in self._schedulers:
            scheduler.step()
        self._last_lr = [
            group["lr"] for group in self._schedulers[-1].optimizer.param_groups
        ]
    # 返回调度器的状态作为一个字典

    state_dict = {
        key: value
        for key, value in self.__dict__.items()  # 遍历对象的所有属性和对应的值
        if key not in ("optimizer", "_schedulers")  # 排除 optimizer 和 _schedulers 属性
    }

    # 初始化 _schedulers 为一个长度与 self._schedulers 相同的空列表
    state_dict["_schedulers"] = [None] * len(self._schedulers)

    # 遍历 self._schedulers 列表，将每个调度器对象的状态字典存入 state_dict["_schedulers"]
    for idx, s in enumerate(self._schedulers):
        state_dict["_schedulers"][idx] = s.state_dict()

    # 返回构建好的状态字典
    return state_dict

    # 加载调度器的状态

    # 弹出 _schedulers 并赋值给局部变量 _schedulers
    _schedulers = state_dict.pop("_schedulers")

    # 更新调度器对象的属性字典，以加载新的状态信息
    self.__dict__.update(state_dict)

    # 恢复 _schedulers 到 state_dict，防止副作用
    # 参考链接：https://github.com/pytorch/pytorch/issues/32756
    state_dict["_schedulers"] = _schedulers

    # 遍历 _schedulers 列表，调用每个调度器对象的 load_state_dict 方法，恢复其状态
    for idx, s in enumerate(_schedulers):
        self._schedulers[idx].load_state_dict(s)
# 定义一个学习率调度器，用于在指标停止改善时减小学习率。

class ReduceLROnPlateau(LRScheduler):
    """Reduce learning rate when a metric has stopped improving.

    Models often benefit from reducing the learning rate by a factor
    of 2-10 once learning stagnates. This scheduler reads a metrics
    quantity and if no improvement is seen for a 'patience' number
    of epochs, the learning rate is reduced.

    Args:
        optimizer (Optimizer): Wrapped optimizer.  # 传入的优化器对象
        mode (str): One of `min`, `max`. In `min` mode, lr will
            be reduced when the quantity monitored has stopped
            decreasing; in `max` mode it will be reduced when the
            quantity monitored has stopped increasing. Default: 'min'.
        factor (float): Factor by which the learning rate will be
            reduced. new_lr = lr * factor. Default: 0.1.  # 学习率减小的因子
        patience (int): The number of allowed epochs with no improvement after
            which the learning rate will be reduced.
            For example, consider the case of having no patience (`patience = 0`).
            In the first epoch, a baseline is established and is always considered good as there's no previous baseline.
            In the second epoch, if the performance is worse than the baseline,
            we have what is considered an intolerable epoch.
            Since the count of intolerable epochs (1) is greater than the patience level (0),
            the learning rate is reduced at the end of this epoch.
            From the third epoch onwards, the learning rate continues to be reduced at the end of each epoch
            if the performance is worse than the baseline. If the performance improves or remains the same,
            the learning rate is not adjusted.
            Default: 10.  # 允许在未见改善后减小学习率的最大轮次
        threshold (float): Threshold for measuring the new optimum,
            to only focus on significant changes. Default: 1e-4.  # 用于衡量新最优值的阈值
        threshold_mode (str): One of `rel`, `abs`. In `rel` mode,
            dynamic_threshold = best * ( 1 + threshold ) in 'max'
            mode or best * ( 1 - threshold ) in `min` mode.
            In `abs` mode, dynamic_threshold = best + threshold in
            `max` mode or best - threshold in `min` mode. Default: 'rel'.  # 动态阈值的计算模式
        cooldown (int): Number of epochs to wait before resuming
            normal operation after lr has been reduced. Default: 0.  # 在减小学习率后恢复正常操作之前等待的轮次数
        min_lr (float or list): A scalar or a list of scalars. A
            lower bound on the learning rate of all param groups
            or each group respectively. Default: 0.  # 所有参数组或各个组学习率的下限
        eps (float): Minimal decay applied to lr. If the difference
            between new and old lr is smaller than eps, the update is
            ignored. Default: 1e-8.  # 应用于学习率的最小衰减
        verbose (bool | str): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

            .. deprecated:: 2.2
                ``verbose`` is deprecated. Please use ``get_last_lr()`` to access the
                learning rate.
    """
    def __init__(
        self,
        optimizer: Optimizer,
        mode: Literal["min", "max"] = "min",
        factor=0.1,
        patience=10,
        threshold=1e-4,
        threshold_mode: Literal["rel", "abs"] = "rel",
        cooldown=0,
        min_lr: Union[List[float], float] = 0,
        eps=1e-8,
        verbose="deprecated",
    ):  # noqa: D107
        # 检查 factor 是否小于 1.0，否则抛出数值错误
        if factor >= 1.0:
            raise ValueError("Factor should be < 1.0.")
        self.factor = factor

        # 将优化器对象连接到调度器实例
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"{type(optimizer).__name__} is not an Optimizer")
        self.optimizer = optimizer

        # 处理最小学习率，如果是列表或元组，则检查长度是否与参数组数匹配
        if isinstance(min_lr, (list, tuple)):
            if len(min_lr) != len(optimizer.param_groups):
                raise ValueError(
                    f"expected {len(optimizer.param_groups)} min_lrs, got {len(min_lr)}"
                )
            self.min_lrs = list(min_lr)
        else:
            self.min_lrs = [min_lr] * len(optimizer.param_groups)

        # 设置耐心值
        self.patience = patience

        # 处理 verbose 参数，检查是否过时
        self.verbose = _check_verbose_deprecated_warning(verbose)

        # 设置冷却时间和计数器
        self.cooldown = cooldown
        self.cooldown_counter = 0

        # 设置模式（最小或最大值）、阈值和阈值模式（相对或绝对）
        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode

        # 初始化最好的值和坏的周期数计数器
        self.best: float
        self.num_bad_epochs: int
        self.mode_worse: float  # 选择模式下的较差值
        self.eps = eps

        # 最后的周期数初始化为0
        self.last_epoch = 0

        # 记录初始学习率，用于后续的比较
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

        # 初始化 _is_better 函数，用于比较当前值和最佳值
        self._init_is_better(
            mode=mode, threshold=threshold, threshold_mode=threshold_mode
        )

        # 调用内部方法 _reset() 来初始化 num_bad_epochs 计数和 cooldown 计数器
        self._reset()
    # 定义一个方法 `step`，用于执行调度器的一步操作，接受一个指标 `metrics` 和可选的 `epoch` 参数
    def step(self, metrics: SupportsFloat, epoch=None):  # type: ignore[override]
        """Perform a step."""
        # 将 `metrics` 转换为浮点数，以防它是零维张量
        current = float(metrics)
        # 如果 `epoch` 未指定，则默认为上一个 epoch 加 1
        if epoch is None:
            epoch = self.last_epoch + 1
        else:
            # 如果指定了 `epoch`，发出警告，表示该参数已过时
            warnings.warn(EPOCH_DEPRECATION_WARNING, UserWarning)
        # 更新 `last_epoch` 为当前 epoch
        self.last_epoch = epoch

        # 如果当前指标比历史最佳指标更好，则更新 `best` 和重置 `num_bad_epochs`
        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            # 否则增加 `num_bad_epochs`
            self.num_bad_epochs += 1

        # 如果处于冷却期，则减少冷却计数器，并重置 `num_bad_epochs`，忽略冷却期内的坏 epoch
        if self.in_cooldown:
            self.cooldown_counter -= 1
            self.num_bad_epochs = 0  # 忽略冷却期内的坏 epoch

        # 如果坏 epoch 的数量超过了容忍的上限 `patience`，则减小学习率 `_reduce_lr`，并设置冷却期计数器
        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.cooldown_counter = self.cooldown
            self.num_bad_epochs = 0

        # 更新 `_last_lr` 列表，包含当前所有参数组的学习率
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

    # 减小学习率的私有方法，根据当前 epoch 更新参数组的学习率
    def _reduce_lr(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            old_lr = float(param_group["lr"])
            new_lr = max(old_lr * self.factor, self.min_lrs[i])
            # 如果计算出的新学习率与旧学习率相差大于阈值 `eps`，则更新学习率
            if old_lr - new_lr > self.eps:
                param_group["lr"] = new_lr

    # 判断当前是否处于冷却期的属性方法
    @property
    def in_cooldown(self):  # noqa: D102
        return self.cooldown_counter > 0

    # 判断新指标 `a` 是否比当前最佳指标 `best` 更好的方法
    def is_better(self, a, best):  # noqa: D102
        if self.mode == "min" and self.threshold_mode == "rel":
            rel_epsilon = 1.0 - self.threshold
            return a < best * rel_epsilon

        elif self.mode == "min" and self.threshold_mode == "abs":
            return a < best - self.threshold

        elif self.mode == "max" and self.threshold_mode == "rel":
            rel_epsilon = self.threshold + 1.0
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self.threshold

    # 初始化判断是否更好的方法，根据模式和阈值模式设定 `mode`、`threshold` 和 `threshold_mode`
    def _init_is_better(self, mode, threshold, threshold_mode):
        if mode not in {"min", "max"}:
            raise ValueError("mode " + mode + " is unknown!")
        if threshold_mode not in {"rel", "abs"}:
            raise ValueError("threshold mode " + threshold_mode + " is unknown!")

        if mode == "min":
            self.mode_worse = inf
        else:  # mode == 'max':
            self.mode_worse = -inf

        self.mode = mode
        self.threshold = threshold
        self.threshold_mode = threshold_mode

    # 返回调度器的状态字典，忽略 `optimizer` 参数
    def state_dict(self):  # noqa: D102
        return {
            key: value for key, value in self.__dict__.items() if key != "optimizer"
        }

    # 加载调度器的状态字典，更新实例变量，并重新初始化判断是否更好的方法
    def load_state_dict(self, state_dict):
        """Load the scheduler's state."""
        self.__dict__.update(state_dict)
        self._init_is_better(
            mode=self.mode, threshold=self.threshold, threshold_mode=self.threshold_mode
        )
class CyclicLR(LRScheduler):
    r"""Sets the learning rate of each parameter group according to cyclical learning rate policy (CLR).

    The policy cycles the learning rate between two boundaries with a constant frequency,
    as detailed in the paper `Cyclical Learning Rates for Training Neural Networks`_.
    The distance between the two boundaries can be scaled on a per-iteration
    or per-cycle basis.

    Cyclical learning rate policy changes the learning rate after every batch.
    `step` should be called after a batch has been used for training.

    This class has three built-in policies, as put forth in the paper:

    * "triangular": A basic triangular cycle without amplitude scaling.
    * "triangular2": A basic triangular cycle that scales initial amplitude by half each cycle.
    * "exp_range": A cycle that scales initial amplitude by :math:`\text{gamma}^{\text{cycle iterations}}`
      at each cycle iteration.

    This implementation was adapted from the github repo: `bckenstler/CLR`_

    Example:
        >>> # xdoctest: +SKIP
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.01, max_lr=0.1)
        >>> data_loader = torch.utils.data.DataLoader(...)
        >>> for epoch in range(10):
        >>>     for batch in data_loader:
        >>>         train_batch(...)
        >>>         scheduler.step()

    .. _Cyclical Learning Rates for Training Neural Networks: https://arxiv.org/abs/1506.01186
    .. _bckenstler/CLR: https://github.com/bckenstler/CLR
    """

    def __init__(
        self,
        optimizer: Optimizer,
        base_lr: Union[float, List[float]],
        max_lr: Union[float, List[float]],
        step_size_up=2000,
        step_size_down: Optional[int] = None,
        mode: Literal["triangular", "triangular2", "exp_range"] = "triangular",
        gamma=1.0,
        scale_fn: Optional[Callable[[float], float]] = None,
        scale_mode: Literal["cycle", "iterations"] = "cycle",
        cycle_momentum=True,
        base_momentum=0.8,
        max_momentum=0.9,
        last_epoch=-1,
        verbose="deprecated",
    ):
        # 初始化函数，设置循环学习率调度器的参数和选项
        # optimizer: 优化器，用于更新参数
        # base_lr: 初始学习率或学习率下界，可以是单个值或多个值的列表
        # max_lr: 最大学习率或学习率上界，可以是单个值或多个值的列表
        # step_size_up: 升序步长，即在周期的前多少个步骤内逐渐增加学习率
        # step_size_down: 降序步长，如果设置，则指定在周期的后多少个步骤内逐渐减小学习率
        # mode: 学习率调度模式，可以是"triangular", "triangular2"或"exp_range"
        # gamma: 当模式为"exp_range"时的缩放因子
        # scale_fn: 自定义缩放函数，根据当前迭代数返回学习率缩放因子
        # scale_mode: 缩放模式，可以是"cycle"或"iterations"
        # cycle_momentum: 是否对动量进行周期性调整
        # base_momentum: 初始动量值
        # max_momentum: 最大动量值
        # last_epoch: 上一个周期的索引，默认为-1，即从头开始
        # verbose: 提示信息输出设置，默认为"deprecated"
        
        # 调用父类的初始化方法，传递 optimizer 参数
        super(CyclicLR, self).__init__(optimizer, last_epoch)

        # 存储传入的参数到对象属性中
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size_up = step_size_up
        self.step_size_down = step_size_down
        self.mode = mode
        self.gamma = gamma
        self.scale_fn = scale_fn
        self.scale_mode = scale_mode
        self.cycle_momentum = cycle_momentum
        self.base_momentum = base_momentum
        self.max_momentum = max_momentum
        self.verbose = verbose

        # 如果调度模式为"triangular2"，将初始学习率放大到两倍的上界
        if self.mode == "triangular2":
            self.scale_fn = lambda x: 1 / (2.0 ** (x - 1))
            if self.scale_mode == "cycle":
                self.scale_fn = lambda x: 1 / (2.0 ** (x - 1))

        # 初始化动量值
        if self.cycle_momentum:
            self.base_momentums = list(map(lambda group: group['momentum'], optimizer.param_groups))

    def get_lr(self):
        # 根据当前迭代数和设置的模式计算当前学习率
        cycle_iter = self.last_epoch % (self.step_size_up + self.step_size_down)
        if cycle_iter < self.step_size_up:
            # 上升阶段，计算学习率从基础值到最大值的变化
            pct = cycle_iter / self.step_size_up
            return [base_lr + pct * (max_lr - base_lr) for base_lr, max_lr in zip(self.base_lr, self.max_lr)]
        else:
            # 下降阶段，计算学习率从最大值到基础值的变化
            pct = (cycle_iter - self.step_size_up) / self.step_size_down
            return [max_lr - pct * (max_lr - base_lr) for base_lr, max_lr in zip(self.base_lr, self.max_lr)]

    def get_momentum(self):
        # 如果设置了循环动量，根据当前迭代数计算动量值的变化
        if not self.cycle_momentum:
            return [self.base_momentum] * len(self.optimizer.param_groups)

        cycle_iter = self.last_epoch % (self.step_size_up + self.step_size_down)
        if cycle_iter < self.step_size_up:
            # 上升阶段，动量从最大值逐渐减小到基础值
            pct = cycle_iter / self.step_size_up
            return [self.max_momentum - pct * (self.max_momentum - self.base_momentum) for _ in self.optimizer.param_groups]
        else:
            # 下降阶段，动量从基础值逐渐增加到最大值
            pct = (cycle_iter - self.step_size_up) / self.step_size_down
            return [self.base_momentum + pct * (self.max_momentum - self.base_momentum) for _ in self.optimizer.param_groups]

    def step(self, epoch=None):
        # 更新学习率和动量值到优化器的参数组中
        # epoch: 当前周期数，用于更新学习率和动量值
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        # 更新学习率
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

        # 更新动量
        if self.cycle_momentum:
            for param_group, momentum in zip(self.optimizer.param_groups, self.get_momentum()):
                param_group['momentum'] = momentum

        # 打印警告信息，提示 verbose 参数已弃用
        if self.verbose == "deprecated":
            warnings.warn("The 'verbose' parameter is deprecated and has no effect.")
        ):  # noqa: D107
        # Attach optimizer
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"{type(optimizer).__name__} is not an Optimizer")
        self.optimizer = optimizer

        # Format base learning rates for the optimizer
        base_lrs = _format_param("base_lr", optimizer, base_lr)

        # Initialize learning rates if last_epoch == -1
        if last_epoch == -1:
            for lr, group in zip(base_lrs, optimizer.param_groups):
                if isinstance(group["lr"], Tensor):
                    lr_val = lr.item() if isinstance(lr, Tensor) else lr
                    group["lr"].fill_(lr_val)
                else:
                    group["lr"] = lr

        # Format max learning rates for the optimizer
        self.max_lrs = _format_param("max_lr", optimizer, max_lr)

        # Convert step sizes to float and calculate total size and step ratio
        step_size_up = float(step_size_up)
        step_size_down = (
            float(step_size_down) if step_size_down is not None else step_size_up
        )
        self.total_size = step_size_up + step_size_down
        self.step_ratio = step_size_up / self.total_size

        # Validate mode and scale_fn
        if mode not in ["triangular", "triangular2", "exp_range"] and scale_fn is None:
            raise ValueError("mode is invalid and scale_fn is None")

        # Assign mode and gamma
        self.mode = mode
        self.gamma = gamma

        # Initialize scale_fn references
        self._scale_fn_ref: Callable[[float], float]
        self._scale_fn_custom = scale_fn
        self.scale_mode = scale_mode
        self._init_scale_fn()

        # Handle cycle momentum if enabled
        self.cycle_momentum = cycle_momentum
        if cycle_momentum:
            if (
                "momentum" not in optimizer.defaults
                and "betas" not in optimizer.defaults
            ):
                raise ValueError(
                    "optimizer must support momentum or beta1 with `cycle_momentum` option enabled"
                )

            # Check if beta1 is used
            self.use_beta1 = "betas" in self.optimizer.defaults

            # Format base and max momentums for the optimizer
            self.base_momentums = _format_param(
                "base_momentum", optimizer, base_momentum
            )
            self.max_momentums = _format_param("max_momentum", optimizer, max_momentum)

            # Initialize momentums if last_epoch == -1
            if last_epoch == -1:
                for m_momentum, b_momentum, group in zip(
                    self.max_momentums, self.base_momentums, optimizer.param_groups
                ):
                    if self.use_beta1:
                        group["betas"] = (m_momentum, *group["betas"][1:])
                    else:
                        group["momentum"] = m_momentum
                    group["max_momentum"] = m_momentum
                    group["base_momentum"] = b_momentum

        # Call super constructor to initialize base class
        super().__init__(optimizer, last_epoch, verbose)
        self.base_lrs = base_lrs
    def _init_scale_fn(self):
        # 如果自定义缩放函数不为空，则直接返回，不进行后续操作
        if self._scale_fn_custom is not None:
            return
        # 如果模式是 "triangular"，则设置缩放函数为 _triangular_scale_fn，并将缩放模式设置为 "cycle"
        if self.mode == "triangular":
            self._scale_fn_ref = self._triangular_scale_fn
            self.scale_mode = "cycle"
        # 如果模式是 "triangular2"，则设置缩放函数为 _triangular2_scale_fn，并将缩放模式设置为 "cycle"
        elif self.mode == "triangular2":
            self._scale_fn_ref = self._triangular2_scale_fn
            self.scale_mode = "cycle"
        # 如果模式是 "exp_range"，则设置缩放函数为 _exp_range_scale_fn，并使用 partial 设置 gamma 参数，将缩放模式设置为 "iterations"
        elif self.mode == "exp_range":
            self._scale_fn_ref = partial(self._exp_range_scale_fn, self.gamma)
            self.scale_mode = "iterations"

    def scale_fn(self, x) -> float:
        """Get the scaling policy."""
        # 如果自定义缩放函数不为空，则调用自定义缩放函数
        if self._scale_fn_custom is not None:
            return self._scale_fn_custom(x)
        else:
            return self._scale_fn_ref(x)  # 调用静态方法作为缩放函数

    @staticmethod
    def _triangular_scale_fn(x: float) -> float:
        # 返回固定的缩放因子 1.0
        return 1.0

    @staticmethod
    def _triangular2_scale_fn(x: float) -> float:
        # 返回根据 x 计算的缩放因子，用于 "triangular2" 模式下的学习率调整
        return 1 / (2.0 ** (x - 1))

    @staticmethod
    def _exp_range_scale_fn(gamma: float, x: float) -> float:
        # 根据指数范围缩放函数的参数 gamma 和 x 计算缩放因子
        return gamma ** x

    def get_lr(self):
        """Calculate the learning rate at batch index.

        This function treats `self.last_epoch` as the last batch index.

        If `self.cycle_momentum` is ``True``, this function has a side effect of
        updating the optimizer's momentum.
        """
        # 检查是否在 step 方法内部调用 get_lr 函数，如果是则发出警告
        _warn_get_lr_called_within_step(self)

        # 计算当前周期数
        cycle = math.floor(1 + self.last_epoch / self.total_size)
        # 计算当前周期内的进度 x
        x = 1.0 + self.last_epoch / self.total_size - cycle
        # 根据当前进度 x 计算缩放因子 scale_factor
        if x <= self.step_ratio:
            scale_factor = x / self.step_ratio
        else:
            scale_factor = (x - 1) / (self.step_ratio - 1)

        lrs = []
        # 遍历基础学习率和最大学习率列表，计算当前学习率
        for base_lr, max_lr in zip(self.base_lrs, self.max_lrs):
            base_height = (max_lr - base_lr) * scale_factor
            # 根据缩放模式选择对应的缩放函数计算学习率
            if self.scale_mode == "cycle":
                lr = base_lr + base_height * self.scale_fn(cycle)
            else:
                lr = base_lr + base_height * self.scale_fn(self.last_epoch)
            lrs.append(lr)

        # 如果启用循环动量调整
        if self.cycle_momentum:
            momentums = []
            # 遍历基础动量和最大动量列表，计算当前动量
            for base_momentum, max_momentum in zip(
                self.base_momentums, self.max_momentums
            ):
                base_height = (max_momentum - base_momentum) * scale_factor
                # 根据缩放模式选择对应的缩放函数计算动量
                if self.scale_mode == "cycle":
                    momentum = max_momentum - base_height * self.scale_fn(cycle)
                else:
                    momentum = max_momentum - base_height * self.scale_fn(
                        self.last_epoch
                    )
                momentums.append(momentum)
            # 更新优化器参数组的动量
            for param_group, momentum in zip(self.optimizer.param_groups, momentums):
                if self.use_beta1:
                    param_group["betas"] = (momentum, *param_group["betas"][1:])
                else:
                    param_group["momentum"] = momentum

        return lrs
    # 返回当前对象的状态字典，继承自父类的状态字典
    def state_dict(self):  # noqa: D102
        # 调用父类的state_dict方法获取初始状态字典
        state = super().state_dict()
        # 移除"_scale_fn_ref"属性，因为它是一个weakref.WeakMethod，无法被pickle化
        state.pop("_scale_fn_ref", None)
        # 弹出"_scale_fn_custom"属性，并保存到fn变量中
        fn = state.pop("_scale_fn_custom")
        # 将"_scale_fn_custom"设为None
        state["_scale_fn_custom"] = None
        # 如果fn不为None且不是types.FunctionType类型，则保存"_scale_fn_custom"属性的__dict__
        if fn is not None and not isinstance(fn, types.FunctionType):
            # 只有当fn是可调用对象时才保存"_scale_fn_custom"
            state["_scale_fn_custom"] = fn.__dict__.copy()

        # 返回更新后的状态字典
        return state

    # 加载给定的状态字典到调度器中
    def load_state_dict(self, state_dict):
        """Load the scheduler's state."""
        # 弹出"_scale_fn_custom"属性并保存到fn变量中
        fn = state_dict.pop("_scale_fn_custom")
        # 调用父类的load_state_dict方法加载其余状态字典
        super().load_state_dict(state_dict)
        # 如果fn不为None，则更新self._scale_fn_custom的__dict__
        if fn is not None:
            self._scale_fn_custom.__dict__.update(fn)
        # 初始化_scale_fn方法
        self._init_scale_fn()
class CosineAnnealingWarmRestarts(LRScheduler):
    r"""Set the learning rate of each parameter group using a cosine annealing schedule.

    The :math:`\eta_{max}` is set to the initial lr, :math:`T_{cur}`
    is the number of epochs since the last restart and :math:`T_{i}` is the number
    of epochs between two warm restarts in SGDR:

    .. math::
        \eta_t = \eta_{min} + \frac{1}{2}(\eta_{max} - \eta_{min})\left(1 +
        \cos\left(\frac{T_{cur}}{T_{i}}\pi\right)\right)

    When :math:`T_{cur}=T_{i}`, set :math:`\eta_t = \eta_{min}`.
    When :math:`T_{cur}=0` after restart, set :math:`\eta_t=\eta_{max}`.

    It has been proposed in
    `SGDR: Stochastic Gradient Descent with Warm Restarts`_.

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        T_0 (int): Number of iterations until the first restart.
        T_mult (int, optional): A factor by which :math:`T_{i}` increases after a restart. Default: 1.
        eta_min (float, optional): Minimum learning rate. Default: 0.
        last_epoch (int, optional): The index of the last epoch. Default: -1.
        verbose (bool | str): If ``True``, prints a message to stdout for
            each update. Default: ``False``.

            .. deprecated:: 2.2
                ``verbose`` is deprecated. Please use ``get_last_lr()`` to access the
                learning rate.

    .. _SGDR\: Stochastic Gradient Descent with Warm Restarts:
        https://arxiv.org/abs/1608.03983
    """

    def __init__(
        self,
        optimizer: Optimizer,
        T_0: int,
        T_mult=1,
        eta_min=0,
        last_epoch=-1,
        verbose="deprecated",
    ):  # noqa: D107
        """Initialize the scheduler.

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            T_0 (int): Number of epochs until the first restart.
            T_mult (int, optional): Multiplicative factor by which T_i increases after a restart. Default: 1.
            eta_min (float, optional): Minimum learning rate. Default: 0.
            last_epoch (int, optional): Index of the last epoch. Default: -1.
            verbose (str): Verbosity option ('deprecated' or 'None'). Default: 'deprecated'.

        Raises:
            ValueError: If T_0 <= 0 or not an integer, T_mult < 1 or not an integer, or eta_min not a float or int.
        """
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError(f"Expected positive integer T_0, but got {T_0}")
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError(f"Expected integer T_mult >= 1, but got {T_mult}")
        if not isinstance(eta_min, (float, int)):
            raise ValueError(
                f"Expected float or int eta_min, but got {eta_min} of type {type(eta_min)}"
            )
        self.T_0 = T_0
        self.T_i = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.T_cur = last_epoch
        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        """Compute the learning rate at the current epoch."""
        _warn_get_lr_called_within_step(self)

        return [
            self.eta_min
            + (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * self.T_cur / self.T_i))
            / 2
            for base_lr in self.base_lrs
        ]
    def step(self, epoch=None):
        """Step could be called after every batch update.

        Example:
            >>> # xdoctest: +SKIP("Undefined vars")
            >>> scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
            >>> iters = len(dataloader)
            >>> for epoch in range(20):
            >>>     for i, sample in enumerate(dataloader):
            >>>         inputs, labels = sample['inputs'], sample['labels']
            >>>         optimizer.zero_grad()
            >>>         outputs = net(inputs)
            >>>         loss = criterion(outputs, labels)
            >>>         loss.backward()
            >>>         optimizer.step()
            >>>         scheduler.step(epoch + i / iters)

        This function can be called in an interleaved way.

        Example:
            >>> # xdoctest: +SKIP("Undefined vars")
            >>> scheduler = CosineAnnealingWarmRestarts(optimizer, T_0, T_mult)
            >>> for epoch in range(20):
            >>>     scheduler.step()
            >>> scheduler.step(26)
            >>> scheduler.step() # scheduler.step(27), instead of scheduler(20)
        """
        # 如果未指定 epoch 且上一轮 epoch 小于 0，则将 epoch 设为 0
        if epoch is None and self.last_epoch < 0:
            epoch = 0

        # 如果 epoch 为 None，则更新 epoch 为上一轮 epoch 加 1，并更新 T_cur 和 T_i
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.T_cur = self.T_cur - self.T_i
                self.T_i = self.T_i * self.T_mult
        else:
            # 如果 epoch 小于 0，则抛出异常
            if epoch < 0:
                raise ValueError(f"Expected non-negative epoch, but got {epoch}")
            # 如果 epoch 大于等于 T_0，则根据 T_mult 计算 T_cur 和 T_i
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                else:
                    n = int(
                        math.log(
                            (epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult
                        )
                    )
                    self.T_cur = epoch - self.T_0 * (self.T_mult**n - 1) / (
                        self.T_mult - 1
                    )
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
        # 更新 last_epoch 为当前 epoch 的整数部分
        self.last_epoch = math.floor(epoch)

        # 更新优化器中各参数组的学习率
        with _enable_get_lr_call(self):
            for i, data in enumerate(zip(self.optimizer.param_groups, self.get_lr())):
                param_group, lr = data
                param_group["lr"] = lr

        # 更新 _last_lr 为当前优化器各参数组的学习率列表
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]
class _SchedulePhase(TypedDict):
    end_step: float  # 定义_SchedulePhase字典结构的一个字段end_step，类型为float
    start_lr: str  # 定义_SchedulePhase字典结构的一个字段start_lr，类型为str
    end_lr: str  # 定义_SchedulePhase字典结构的一个字段end_lr，类型为str
    start_momentum: str  # 定义_SchedulePhase字典结构的一个字段start_momentum，类型为str
    end_momentum: str  # 定义_SchedulePhase字典结构的一个字段end_momentum，类型为str


class OneCycleLR(LRScheduler):
    r"""Sets the learning rate of each parameter group according to the 1cycle learning rate policy.

    The 1cycle policy anneals the learning rate from an initial learning rate to some maximum
    learning rate and then from that maximum learning rate to some minimum learning rate much
    lower than the initial learning rate.
    This policy was initially described in the paper `Super-Convergence:
    Very Fast Training of Neural Networks Using Large Learning Rates`_.

    The 1cycle learning rate policy changes the learning rate after every batch.
    `step` should be called after a batch has been used for training.

    This scheduler is not chainable.

    Note also that the total number of steps in the cycle can be determined in one
    of two ways (listed in order of precedence):

    #. A value for total_steps is explicitly provided.
    #. A number of epochs (epochs) and a number of steps per epoch
       (steps_per_epoch) are provided.
       In this case, the number of total steps is inferred by
       total_steps = epochs * steps_per_epoch

    You must either provide a value for total_steps or provide a value for both
    epochs and steps_per_epoch.

    The default behaviour of this scheduler follows the fastai implementation of 1cycle, which
    claims that "unpublished work has shown even better results by using only two phases". To
    mimic the behaviour of the original paper instead, set ``three_phase=True``.

    Example:
        >>> # xdoctest: +SKIP
        >>> data_loader = torch.utils.data.DataLoader(...)
        >>> optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        >>> scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(data_loader), epochs=10)
        >>> for epoch in range(10):
        >>>     for batch in data_loader:
        >>>         train_batch(...)
        >>>         optimizer.step()
        >>>         scheduler.step()


    .. _Super-Convergence\: Very Fast Training of Neural Networks Using Large Learning Rates:
        https://arxiv.org/abs/1708.07120
    """

    def __init__(
        self,
        optimizer: Optimizer,
        max_lr: Union[float, List[float]],  # 最大学习率，可以是单个值或列表
        total_steps: Optional[int] = None,  # 总步数（可选），如果未提供，则由epochs和steps_per_epoch推断
        epochs: Optional[int] = None,  # 训练周期数（可选）
        steps_per_epoch: Optional[int] = None,  # 每个周期的步数（可选）
        pct_start=0.3,  # 开始学习率变化的百分比
        anneal_strategy: Literal["cos", "linear"] = "cos",  # 学习率退火策略，可以是"cos"或"linear"
        cycle_momentum=True,  # 是否使用动量循环
        base_momentum: Union[float, List[float]] = 0.85,  # 基础动量，可以是单个值或列表
        max_momentum: Union[float, List[float]] = 0.95,  # 最大动量，可以是单个值或列表
        div_factor=25.0,  # 学习率最大值相对于初始学习率的倍数
        final_div_factor=1e4,  # 学习率最小值相对于初始学习率的倍数
        three_phase=False,  # 是否使用三阶段策略
        last_epoch=-1,  # 上一个周期的索引，默认为-1表示从头开始
        verbose="deprecated",  # 日志记录方式，默认为弃用模式
    # 定义一个内部方法 `_anneal_func`，接受任意位置参数和关键字参数
    def _anneal_func(self, *args, **kwargs):
        # 检查对象是否有属性 `_anneal_func_type`
        if hasattr(self, "_anneal_func_type"):
            # 如果 `_anneal_func_type` 属性的值为 "cos"，调用 `_annealing_cos` 方法
            if self._anneal_func_type == "cos":
                return self._annealing_cos(*args, **kwargs)
            # 如果 `_anneal_func_type` 属性的值为 "linear"，调用 `_annealing_linear` 方法
            elif self._anneal_func_type == "linear":
                return self._annealing_linear(*args, **kwargs)
            else:
                # 抛出值错误异常，指明未知的 `_anneal_func_type`
                raise ValueError(f"Unknown _anneal_func_type: {self._anneal_func_type}")
        else:
            # 兼容性处理，调用 `anneal_func` 方法
            return self.anneal_func(*args, **kwargs)  # type: ignore[attr-defined]

    @staticmethod
    # 定义静态方法 `_annealing_cos`，实现余弦退火从 `start` 到 `end` 的过程，`pct` 从 0.0 到 1.0
    def _annealing_cos(start, end, pct):
        """Cosine anneal from `start` to `end` as pct goes from 0.0 to 1.0."""
        cos_out = math.cos(math.pi * pct) + 1
        return end + (start - end) / 2.0 * cos_out

    @staticmethod
    # 定义静态方法 `_annealing_linear`，实现线性退火从 `start` 到 `end` 的过程，`pct` 从 0.0 到 1.0
    def _annealing_linear(start, end, pct):
        """Linearly anneal from `start` to `end` as pct goes from 0.0 to 1.0."""
        return (end - start) * pct + start

    # 定义方法 `get_lr`，计算每个参数组的学习率
    def get_lr(self):
        """Compute the learning rate of each parameter group."""
        # 警告方法 `_warn_get_lr_called_within_step` 在步骤内部被调用
        _warn_get_lr_called_within_step(self)

        # 初始化学习率列表 `lrs` 和步骤编号 `step_num`
        lrs = []
        step_num = self.last_epoch

        # 如果步骤编号大于总步数 `total_steps`，抛出值错误异常
        if step_num > self.total_steps:
            raise ValueError(
                f"Tried to step {step_num} times. The specified number of total steps is {self.total_steps}"  # noqa: UP032
            )

        # 遍历优化器的参数组
        for group in self.optimizer.param_groups:
            start_step = 0.0
            # 遍历调度阶段 `_schedule_phases`
            for i, phase in enumerate(self._schedule_phases):
                end_step = phase["end_step"]
                # 如果当前步数小于等于结束步数或者是最后一个阶段，则计算百分比 `pct`
                if step_num <= end_step or i == len(self._schedule_phases) - 1:
                    pct = (step_num - start_step) / (end_step - start_step)
                    # 使用 `_anneal_func` 方法计算学习率 `computed_lr`
                    computed_lr = self._anneal_func(
                        group[phase["start_lr"]], group[phase["end_lr"]], pct
                    )
                    # 如果循环动量为真，计算动量 `computed_momentum`
                    if self.cycle_momentum:
                        computed_momentum = self._anneal_func(
                            group[phase["start_momentum"]],
                            group[phase["end_momentum"]],
                            pct,
                        )
                    break
                start_step = phase["end_step"]

            # 将计算得到的学习率 `computed_lr` 添加到学习率列表 `lrs` 中
            lrs.append(computed_lr)  # type: ignore[possibly-undefined]
            # 如果循环动量为真，根据 `use_beta1` 设置优化器参数组的动量 `momentum` 或者 `betas`
            if self.cycle_momentum:
                if self.use_beta1:
                    group["betas"] = (computed_momentum, *group["betas"][1:])  # type: ignore[possibly-undefined]
                else:
                    group[
                        "momentum"
                    ] = computed_momentum  # type: ignore[possibly-undefined]

        # 返回学习率列表 `lrs`
        return lrs
```