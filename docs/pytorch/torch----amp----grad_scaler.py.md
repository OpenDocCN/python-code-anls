# `.\pytorch\torch\amp\grad_scaler.py`

```py
# 允许未标注的函数类型
# 从未来导入的__future__模块，使得当前代码支持对类型注解的处理
from __future__ import annotations

# 导入inspect模块，用于获取对象信息
import inspect
# 导入warnings模块，用于处理警告
import warnings
# 导入collections模块中的abc和defaultdict类
from collections import abc, defaultdict
# 导入Enum类，用于创建枚举类型
from enum import Enum
# 导入typing模块中的各种类型
from typing import Any, cast, Dict, Iterable, List, Optional, overload, Tuple, Union

# 导入torch模块
import torch

# __all__列表，指定可以通过from ... import * 导入的模块成员
__all__ = ["OptState", "GradScaler"]

# _MultiDeviceReplicator类定义
class _MultiDeviceReplicator:
    """Lazily serves copies of a tensor to requested devices.

    Copies are cached per-device.
    """

    def __init__(self, master_tensor: torch.Tensor) -> None:
        # 初始化函数，传入主要的张量对象作为参数
        self.master = master_tensor
        # _per_device_tensors字典，存储每个设备对应的张量副本
        self._per_device_tensors: Dict[torch.device, torch.Tensor] = {}

    def get(self, device: torch.device) -> torch.Tensor:
        # get方法，根据设备获取对应的张量副本
        retval = self._per_device_tensors.get(device, None)
        if retval is None:
            # 如果当前设备没有缓存的副本，则创建并缓存
            retval = self.master.to(device=device, non_blocking=True, copy=True)
            self._per_device_tensors[device] = retval
        return retval

# 定义OptState枚举类，表示优化器状态
class OptState(Enum):
    READY = 0
    UNSCALED = 1
    STEPPED = 2

# _refresh_per_optimizer_state函数，返回一个包含优化器状态和空字典的字典
def _refresh_per_optimizer_state() -> Dict[str, Any]:
    return {"stage": OptState.READY, "found_inf_per_device": {}}

# GradScaler类定义
class GradScaler:
    """An instance ``scaler`` of :class:`GradScaler`.

    Helps perform the steps of gradient scaling
    conveniently.

    * ``scaler.scale(loss)`` multiplies a given loss by ``scaler``'s current scale factor.
    * ``scaler.step(optimizer)`` safely unscales gradients and calls ``optimizer.step()``.
    * ``scaler.update()`` updates ``scaler``'s scale factor.

    Example::

        # Creates a GradScaler once at the beginning of training.
        scaler = GradScaler()

        for epoch in epochs:
            for input, target in data:
                optimizer.zero_grad()
                output = model(input)
                loss = loss_fn(output, target)

                # Scales loss.  Calls backward() on scaled loss to create scaled gradients.
                scaler.scale(loss).backward()

                # scaler.step() first unscales gradients of the optimizer's params.
                # If gradients don't contain infs/NaNs, optimizer.step() is then called,
                # otherwise, optimizer.step() is skipped.
                scaler.step(optimizer)

                # Updates the scale for next iteration.
                scaler.update()

    See the :ref:`Automatic Mixed Precision examples<amp-examples>` for usage
    (along with autocasting) in more complex cases like gradient clipping, gradient accumulation, gradient penalty,
    and multiple losses/optimizers.
    """
    """
    ``scaler`` dynamically estimates the scale factor each iteration.  To minimize gradient underflow,
    a large scale factor should be used.  However, ``float16`` values can "overflow" (become inf or NaN) if
    the scale factor is too large.  Therefore, the optimal scale factor is the largest factor that can be used
    without incurring inf or NaN gradient values.
    ``scaler`` approximates the optimal scale factor over time by checking the gradients for infs and NaNs during every
    ``scaler.step(optimizer)`` (or optional separate ``scaler.unscale_(optimizer)``, see :meth:`unscale_`).

    * If infs/NaNs are found, ``scaler.step(optimizer)`` skips the underlying ``optimizer.step()`` (so the params
      themselves remain uncorrupted) and ``update()`` multiplies the scale by ``backoff_factor``.

    * If no infs/NaNs are found, ``scaler.step(optimizer)`` runs the underlying ``optimizer.step()`` as usual.
      If ``growth_interval`` unskipped iterations occur consecutively, ``update()`` multiplies the scale by
      ``growth_factor``.

    The scale factor often causes infs/NaNs to appear in gradients for the first few iterations as its
    value calibrates.  ``scaler.step`` will skip the underlying ``optimizer.step()`` for these
    iterations.  After that, step skipping should occur rarely (once every few hundred or thousand iterations).

    Args:
        device (str, optional, default="cuda"): Device type to use. Possible values are: 'cuda' and 'cpu'.
            The type is the same as the `type` attribute of a :class:`torch.device`.
            Thus, you may obtain the device type of a tensor using `Tensor.device.type`.
        init_scale (float, optional, default=2.**16):  Initial scale factor.
        growth_factor (float, optional, default=2.0):  Factor by which the scale is multiplied during
            :meth:`update` if no inf/NaN gradients occur for ``growth_interval`` consecutive iterations.
        backoff_factor (float, optional, default=0.5):  Factor by which the scale is multiplied during
            :meth:`update` if inf/NaN gradients occur in an iteration.
        growth_interval (int, optional, default=2000):  Number of consecutive iterations without inf/NaN gradients
            that must occur for the scale to be multiplied by ``growth_factor``.
        enabled (bool, optional):  If ``False``, disables gradient scaling. :meth:`step` simply
            invokes the underlying ``optimizer.step()``, and other methods become no-ops.
            Default: ``True``
    """

    # 构造函数初始化方法，用于创建一个名为 scaler 的对象，进行动态的梯度缩放
    def __init__(
        self,
        device: str = "cuda",  # 初始化参数：指定设备类型，默认为 CUDA
        init_scale: float = 2.0**16,  # 初始化参数：初始缩放因子，默认为 2^16
        growth_factor: float = 2.0,  # 初始化参数：增长因子，默认为 2.0
        backoff_factor: float = 0.5,  # 初始化参数：回退因子，默认为 0.5
        growth_interval: int = 2000,  # 初始化参数：增长间隔，默认为 2000
        enabled: bool = True,  # 初始化参数：是否启用梯度缩放，默认为 True
   `
# 定义一个GradScaler类，用于自动调节梯度的缩放
class GradScaler:
    # 初始化方法，接受设备类型和是否启用GradScaler
    def __init__(
        self,
        device: str,
        enabled: bool,
        init_scale: float = 1.0,
        growth_factor: float = 2.0,
        backoff_factor: float = 0.5,
        growth_interval: int = 2000,
    ) -> None:
        self._device = device  # 存储设备类型
        self._enabled = enabled  # 是否启用GradScaler

        # 如果设备为cuda且启用GradScaler但CUDA不可用时，发出警告并禁用GradScaler
        if self._device == "cuda":
            if enabled and torch.cuda.amp.common.amp_definitely_not_available():
                warnings.warn(
                    "torch.cuda.amp.GradScaler is enabled, but CUDA is not available.  Disabling."
                )
                self._enabled = False

        # 如果GradScaler被启用，则进行以下断言
        if self._enabled:
            assert growth_factor > 1.0, "The growth factor must be > 1.0."
            assert backoff_factor < 1.0, "The backoff factor must be < 1.0."

            self._init_scale = init_scale  # 初始缩放比例
            # self._scale将在首次调用scale()时惰性初始化
            self._scale: Optional[torch.Tensor] = None
            self._growth_factor = growth_factor  # 缩放因子
            self._backoff_factor = backoff_factor  # 回退因子
            self._growth_interval = growth_interval  # 生长间隔
            self._init_growth_tracker = 0  # 初始生长追踪器
            # self._growth_tracker将在首次调用scale()时惰性初始化
            self._growth_tracker: Optional[torch.Tensor] = None
            # self._per_optimizer_states将存储每个优化器的状态字典
            self._per_optimizer_states: Dict[int, Dict[str, Any]] = defaultdict(
                _refresh_per_optimizer_state
            )

    # 检查_scale和_growth_tracker是否已经初始化，用于在调用某些方法时进行检查
    def _check_scale_growth_tracker(
        self, funcname: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        fix = "This may indicate your script did not use scaler.scale(loss or outputs) earlier in the iteration."
        assert self._scale is not None, (
            f"Attempted {funcname} but _scale is None.  " + fix
        )
        assert self._growth_tracker is not None, (
            f"Attempted {funcname} but _growth_tracker is None.  " + fix
        )
        return (self._scale, self._growth_tracker)

    # 惰性初始化_scale和_growth_tracker
    def _lazy_init_scale_growth_tracker(self, dev: torch.device) -> None:
        assert self._growth_tracker is None, "_growth_tracker initialized before _scale"
        # 使用给定的初始值初始化_scale和_growth_tracker
        self._scalef scale(self, outputs: torch.Tensor) -> torch.Tensor:
        ...

    @overload
    def scale(self, outputs: List[torch.Tensor]) -> List[torch.Tensor]:
        ...

    @overload
    def scale(self, outputs: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
        ...

    @overload
    def scale(self, outputs: Iterable[torch.Tensor]) -> Iterable[torch.Tensor]:
        ...

    def scale(
        self,
        outputs: Union[torch.Tensor, Iterable[torch.Tensor]],
        # scale 方法的重载，接受单个张量或张量的可迭代对象作为输入
    ) -> Union[torch.Tensor, Iterable[torch.Tensor]]:
        """
        Multiplies ('scales') a tensor or list of tensors by the scale factor.

        Returns scaled outputs.  If this instance of :class:`GradScaler` is not enabled, outputs are returned
        unmodified.

        Args:
            outputs (Tensor or iterable of Tensors):  Outputs to scale.
        """
        # 如果未启用GradScaler实例，则直接返回未修改的outputs
        if not self._enabled:
            return outputs

        # 对于单个张量的简化处理
        if isinstance(outputs, torch.Tensor):
            # 如果尚未初始化_scale增长追踪器，则初始化它
            if self._scale is None:
                self._lazy_init_scale_growth_tracker(outputs.device)
            # 确保_scale不为None，然后对输出进行缩放操作
            assert self._scale is not None
            return outputs * self._scale.to(device=outputs.device, non_blocking=True)

        # 处理多个输出的复杂情况
        stash: List[
            _MultiDeviceReplicator
        ] = []  # 用于保存可以被apply_scale函数覆盖的引用

        def apply_scale(val: Union[torch.Tensor, Iterable[torch.Tensor]]):
            # 对于单个张量的情况
            if isinstance(val, torch.Tensor):
                # 如果stash列表为空，则进行初始化操作
                if len(stash) == 0:
                    # 如果尚未初始化_scale增长追踪器，则初始化它
                    if self._scale is None:
                        self._lazy_init_scale_growth_tracker(val.device)
                    assert self._scale is not None
                    # 将_MultiDeviceReplicator对象添加到stash列表中
                    stash.append(_MultiDeviceReplicator(self._scale))
                # 返回缩放后的张量
                return val * stash[0].get(val.device)
            
            # 对于可迭代对象的情况
            if isinstance(val, abc.Iterable):
                # 递归地对每个元素应用缩放函数
                iterable = map(apply_scale, val)
                # 如果原始输入是列表或元组，则返回相同类型的结果
                if isinstance(val, (list, tuple)):
                    return type(val)(iterable)
                # 否则返回迭代器对象
                return iterable
            
            # 如果既不是张量也不是可迭代对象，则抛出数值错误异常
            raise ValueError("outputs must be a Tensor or an iterable of Tensors")

        # 应用缩放函数到outputs，并返回结果
        return apply_scale(outputs)

    def _unscale_grads_(
        self,
        optimizer: torch.optim.Optimizer,
        inv_scale: torch.Tensor,
        found_inf: torch.Tensor,
        allow_fp16: bool,
        ) -> Dict[torch.device, torch.Tensor]:
        # 创建 _MultiDeviceReplicator 对象，用于存储每个设备上的反比例尺
        per_device_inv_scale = _MultiDeviceReplicator(inv_scale)
        # 创建 _MultiDeviceReplicator 对象，用于标记每个设备上是否存在无限值
        per_device_found_inf = _MultiDeviceReplicator(found_inf)

        # 设置 _amp_foreach_non_finite_check_and_unscale_ 函数的准备工作，将梯度按设备和数据类型进行分割
        # 由于可能有大量的梯度，我们希望只遍历一次
        # 但是事先不知道它们的设备或数据类型

        # 使用 defaultdict 嵌套字典的方式初始化 per_device_and_dtype_grads 字典
        # 每个设备对应一个字典，字典的值是一个列表，用于存储不同数据类型的梯度张量
        per_device_and_dtype_grads: Dict[
            torch.device, Dict[torch.dtype, List[torch.Tensor]]
        ] = defaultdict(lambda: defaultdict(list))

        # 进入 torch 的无梯度环境
        with torch.no_grad():
            # 遍历优化器的参数组
            for group in optimizer.param_groups:
                for param in group["params"]:
                    assert isinstance(param, torch.Tensor)
                    # 如果梯度为 None，则跳过
                    if param.grad is None:
                        continue
                    # 如果不允许使用 FP16 并且梯度的数据类型是 torch.float16，则抛出异常
                    if (not allow_fp16) and param.grad.dtype == torch.float16:
                        raise ValueError("Attempting to unscale FP16 gradients.")
                    # 如果梯度是稀疏的
                    if param.grad.is_sparse:
                        # 如果稀疏梯度的数据类型是 torch.float16，则调用 coalesce() 方法进行压缩处理
                        if param.grad.dtype is torch.float16:
                            param.grad = param.grad.coalesce()
                        # 获取需要反比例尺的值
                        to_unscale = param.grad._values()
                    else:
                        # 否则直接获取需要反比例尺的梯度
                        to_unscale = param.grad

                    # 将梯度按设备和数据类型存储到 per_device_and_dtype_grads 字典中的对应位置
                    per_device_and_dtype_grads[to_unscale.device][
                        to_unscale.dtype
                    ].append(to_unscale)

            # 遍历 per_device_and_dtype_grads 字典中的每个设备和数据类型的梯度
            for device, per_dtype_grads in per_device_and_dtype_grads.items():
                for grads in per_dtype_grads.values():
                    # 调用 _amp_foreach_non_finite_check_and_unscale_ 函数对每个梯度进行非有限检查和反比例尺操作
                    torch._amp_foreach_non_finite_check_and_unscale_(
                        grads,
                        per_device_found_inf.get(device),
                        per_device_inv_scale.get(device),
                    )

        # 返回 per_device_found_inf._per_device_tensors 结果
        return per_device_found_inf._per_device_tensors
    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        """
        Divides ("unscales") the optimizer's gradient tensors by the scale factor.

        :meth:`unscale_` is optional, serving cases where you need to
        :ref:`modify or inspect gradients<working-with-unscaled-gradients>`
        between the backward pass(es) and :meth:`step`.
        If :meth:`unscale_` is not called explicitly, gradients will be unscaled automatically during :meth:`step`.

        Simple example, using :meth:`unscale_` to enable clipping of unscaled gradients::

            ...
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(optimizer)
            scaler.update()

        Args:
            optimizer (torch.optim.Optimizer): Optimizer that owns the gradients to be unscaled.

        .. note::
            :meth:`unscale_` does not incur a CPU-GPU sync.

        .. warning::
            :meth:`unscale_` should only be called once per optimizer per :meth:`step` call,
            and only after all gradients for that optimizer's assigned parameters have been accumulated.
            Calling :meth:`unscale_` twice for a given optimizer between each :meth:`step` triggers a RuntimeError.

        .. warning::
            :meth:`unscale_` may unscale sparse gradients out of place, replacing the ``.grad`` attribute.
        """
        # 如果未启用梯度缩放，则直接返回
        if not self._enabled:
            return

        # 检查是否已经调用过unscale_方法，以确保每次只调用一次
        self._check_scale_growth_tracker("unscale_")

        # 获取特定优化器的状态信息
        optimizer_state = self._per_optimizer_states[id(optimizer)]

        # 如果已经在上次更新后调用了unscale_，则会触发RuntimeError
        if optimizer_state["stage"] is OptState.UNSCALED:
            raise RuntimeError(
                "unscale_() has already been called on this optimizer since the last update()."
            )
        # 如果在调用step()之后再次调用unscale_，则会触发RuntimeError
        elif optimizer_state["stage"] is OptState.STEPPED:
            raise RuntimeError("unscale_() is being called after step().")

        # FP32的除法在某些编译选项下可能不精确，因此我们在FP64中进行倒数操作
        assert self._scale is not None
        inv_scale = self._scale.double().reciprocal().float()

        # 创建一个用于记录是否找到无穷大梯度的张量，初始化为0
        found_inf = torch.full((), 0.0, dtype=torch.float32, device=self._scale.device)

        # 调用内部方法_unscale_grads_来执行梯度的unscale操作
        optimizer_state["found_inf_per_device"] = self._unscale_grads_(
            optimizer, inv_scale, found_inf, False
        )

        # 标记优化器状态为UNSCALED，表示已经进行了unscale操作
        optimizer_state["stage"] = OptState.UNSCALED

    def _maybe_opt_step(
        self,
        optimizer: torch.optim.Optimizer,
        optimizer_state: Dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> Optional[float]:
        """
        Optionally performs an optimizer step based on whether gradients contain infinite values.

        Args:
            optimizer (torch.optim.Optimizer): Optimizer instance.
            optimizer_state (Dict[str, Any]): State information associated with the optimizer.
            *args: Variable length argument list.
            **kwargs: Keyword arguments.

        Returns:
            Optional[float]: If gradients do not contain infinite values, returns the optimizer's step result.

        """
        # 初始化返回值为None
        retval: Optional[float] = None

        # 如果未发现任何设备上存在无穷大梯度，则执行优化器的step操作
        if not sum(v.item() for v in optimizer_state["found_inf_per_device"].values()):
            retval = optimizer.step(*args, **kwargs)

        # 返回step操作的结果，如果未执行step，则返回None
        return retval

    def step(
        self, optimizer: torch.optim.Optimizer, *args: Any, **kwargs: Any
    def _get_scale_async(self) -> Optional[torch.Tensor]:
        """Return the asynchronous scale value as a Torch tensor or None."""
        return self._scale



    def get_scale(self) -> float:
        """Return a Python float containing the current scale, or 1.0 if scaling is disabled.

        .. warning::
            :meth:`get_scale` incurs a CPU-GPU sync.
        """
        if self._enabled:
            return (
                self._init_scale  # 返回初始化时的比例
                if (scale := self._get_scale_async()) is None  # 如果异步比例为空，则返回初始化比例
                else cast(float, scale.item())  # 否则将异步比例转换为float并返回
            )
        return 1.0  # 如果未启用，则返回默认比例1.0



    def get_growth_factor(self) -> float:
        r"""Return a Python float containing the scale growth factor."""
        return self._growth_factor  # 返回增长因子



    def set_growth_factor(self, new_factor: float) -> None:
        r"""Set a new scale growth factor.

        Args:
            new_scale (float):  Value to use as the new scale growth factor.
        """
        self._growth_factor = new_factor  # 设置新的增长因子



    def get_backoff_factor(self) -> float:
        r"""Return a Python float containing the scale backoff factor."""
        return self._backoff_factor  # 返回退避因子



    def set_backoff_factor(self, new_factor: float) -> None:
        r"""Set a new scale backoff factor.

        Args:
            new_scale (float):  Value to use as the new scale backoff factor.
        """
        self._backoff_factor = new_factor  # 设置新的退避因子



    def get_growth_interval(self) -> int:
        r"""Return a Python int containing the growth interval."""
        return self._growth_interval  # 返回增长间隔



    def set_growth_interval(self, new_interval: int) -> None:
        r"""Set a new growth interval.

        Args:
            new_interval (int):  Value to use as the new growth interval.
        """
        self._growth_interval = new_interval  # 设置新的增长间隔



    def _get_growth_tracker(self) -> int:
        """Return the growth tracker value as an integer or 0 if not enabled."""
        if self._enabled:
            return (
                self._init_growth_tracker  # 返回初始化时的增长追踪器
                if self._growth_tracker is None  # 如果增长追踪器为空，则返回初始化值
                else cast(int, self._growth_tracker.item())  # 否则将增长追踪器转换为int并返回
            )
        return 0  # 如果未启用，则返回0



    def is_enabled(self) -> bool:
        r"""Return a bool indicating whether this instance is enabled."""
        return self._enabled  # 返回实例是否启用的布尔值
    def state_dict(self) -> Dict[str, Any]:
        r"""Return the state of the scaler as a :class:`dict`.

        It contains five entries:

        * ``"scale"`` - a Python float containing the current scale
        * ``"growth_factor"`` - a Python float containing the current growth factor
        * ``"backoff_factor"`` - a Python float containing the current backoff factor
        * ``"growth_interval"`` - a Python int containing the current growth interval
        * ``"_growth_tracker"`` - a Python int containing the number of recent consecutive unskipped steps.

        If this instance is not enabled, returns an empty dict.

        .. note::
           If you wish to checkpoint the scaler's state after a particular iteration, :meth:`state_dict`
           should be called after :meth:`update`.
        """
        # 如果实例被启用，则返回当前标量的状态作为字典
        if self._enabled:
            return {
                "scale": self.get_scale(),                  # 获取当前缩放比例
                "growth_factor": self._growth_factor,       # 获取当前生长因子
                "backoff_factor": self._backoff_factor,     # 获取当前回退因子
                "growth_interval": self._growth_interval,   # 获取当前生长间隔
                "_growth_tracker": self._get_growth_tracker(),  # 获取最近连续未跳过步骤的数量
            }
        # 如果实例未启用，则返回空字典
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        r"""Load the scaler state.

        If this instance is disabled, :meth:`load_state_dict` is a no-op.

        Args:
           state_dict(dict): scaler state.  Should be an object returned from a call to :meth:`state_dict`.
        """
        # 如果实例未启用，则无操作
        if not self._enabled:
            return

        # 如果状态字典为空，则引发运行时错误
        if len(state_dict) == 0:
            raise RuntimeError(
                "The source state dict is empty, possibly because it was saved "
                "from a disabled instance of GradScaler."
            )

        # 加载缩放比例
        self._init_scale = cast(float, state_dict["scale"])
        if self._scale is not None:
            self._scale.fill_(state_dict["scale"])  # 填充当前缩放比例
        # 加载生长因子、回退因子、生长间隔和生长追踪器
        self._growth_factor = cast(float, state_dict["growth_factor"])
        self._backoff_factor = cast(float, state_dict["backoff_factor"])
        self._growth_interval = cast(int, state_dict["growth_interval"])
        self._init_growth_tracker = cast(int, state_dict["_growth_tracker"])
        if self._growth_tracker is not None:
            self._growth_tracker.fill_(state_dict["_growth_tracker"])  # 填充最近连续未跳过步骤的数量
    # 返回对象的状态字典，用于对象的序列化
    def __getstate__(self) -> Dict[str, Any]:
        # 复制当前对象的字典形式状态
        state = self.__dict__.copy()
        # 如果启用了梯度缩放
        if self._enabled:
            # 断言当前优化器状态列表为空，即在迭代开始或者在 scaler.update() 结束后才能进行序列化
            assert len(self._per_optimizer_states) == 0, (
                "A GradScaler instance may only be pickled at the beginning "
                "of an iteration, or at the end after scaler.update()."
            )
            # 通过方法获取当前梯度缩放比例的初始值，并存入状态字典中
            state["_init_scale"] = self.get_scale()
            # 获取增长追踪器的初始状态，并存入状态字典中
            state["_init_growth_tracker"] = self._get_growth_tracker()
            # 将 _scale 和 _growth_tracker 张量设置为 None，以避免直接序列化引发警告
            state["_scale"] = None
            state["_growth_tracker"] = None
        # 返回最终状态字典
        return state

    # 恢复对象的状态
    def __setstate__(self, state: Dict[str, Any]) -> None:
        # 更新对象的字典形式状态
        self.__dict__.update(state)

    # 检查每个设备上的无穷值情况，并返回结果字典
    def _check_inf_per_device(self, optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
        # 获取当前缩放比例和增长追踪器状态
        _scale, _ = self._check_scale_growth_tracker("_check_inf_per_device")

        # 创建虚拟的逆缩放比例张量和找到无穷值的张量，初始化为全1和全0
        dummy_inv_scale = torch.full((), 1.0, dtype=torch.float32, device=_scale.device)
        found_inf = torch.full((), 0.0, dtype=torch.float32, device=_scale.device)

        # 更新当前优化器状态中的 "found_inf_per_device" 条目，调用 _unscale_grads_ 方法来更新
        self._per_optimizer_states[id(optimizer)][
            "found_inf_per_device"
        ] = self._unscale_grads_(optimizer, dummy_inv_scale, found_inf, True)

        # 返回当前优化器状态中的 "found_inf_per_device" 条目
        return self._per_optimizer_states[id(optimizer)]["found_inf_per_device"]

    # 返回特定优化器的 "found_inf_per_device" 条目
    def _found_inf_per_device(self, optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
        return self._per_optimizer_states[id(optimizer)]["found_inf_per_device"]
```