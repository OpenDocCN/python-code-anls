# `.\pytorch\torch\distributed\fsdp\sharded_grad_scaler.py`

```py
# mypy: allow-untyped-defs
# 导入日志模块
import logging
# 导入抽象基类及默认字典
from collections import abc, defaultdict
# 导入类型提示相关模块
from typing import Any, Dict, Iterable, List, Optional, overload, Sequence, Tuple, Union

# 导入PyTorch相关模块
import torch
# 导入PyTorch分布式相关模块
import torch.distributed as dist
# 导入PyTorch自动混合精度相关模块
from torch.amp.grad_scaler import _MultiDeviceReplicator, GradScaler, OptState
# 导入PyTorch分布式C10d进程组模块
from torch.distributed.distributed_c10d import ProcessGroup

# 获取当前模块的日志记录器
logger = logging.getLogger(__name__)


def _refresh_per_optimizer_state() -> Dict[str, Any]:
    """
    返回一个包含优化器状态的字典，初始状态为READY，且找到的每设备无穷大（inf）值为空字典。
    """
    return {"stage": OptState.READY, "found_inf_per_device": {}}


def _is_supported_device(tensor: torch.Tensor) -> bool:
    """
    检查给定的张量是否支持的设备类型，支持CUDA设备或者设备类型为"cpu"、"xla"、"hpu"或者私有使用的第一个后端名称。
    """
    return tensor.is_cuda or tensor.device.type in (
        "xla",
        "cpu",
        "hpu",
        torch._C._get_privateuse1_backend_name(),
    )


class _GeneralMultiDeviceReplicator(_MultiDeviceReplicator):
    """
    _MultiDeviceReplicator的扩展类，用于按需为请求的设备提供张量。
    支持扩展到支持"cpu"作为设备。
    """

    def __init__(self, master_tensor: torch.Tensor) -> None:
        """
        初始化方法，要求master_tensor必须是支持的设备类型。
        """
        assert _is_supported_device(master_tensor)
        self.master = master_tensor
        self._per_device_tensors: Dict[torch.device, torch.Tensor] = {}


class ShardedGradScaler(GradScaler):
    """
    ShardedGradScaler能够以分片感知的方式执行梯度缩放。它扩展自GradScaler的功能：
    * 支持PyTorch DDP和FSDP实现
    * 支持CPU离载张量（如在完全分片数据并行[FSDP]中使用）
    * 支持FSDP返回的自定义混合精度损失数据类型（fp16、bf16）
    * 在任何torch.device上同步缩放梯度张量的inf/nan（张量放置的位置）跨节点

    示例::

        # 在训练开始时创建一个ShardedGradScaler。
        scaler = ShardedGradScaler()

        for epoch in epochs:
            for input, target in data:
                optimizer.zero_grad()
                output = model(input)
                loss = loss_fn(output, target)

                # 缩放损失。调用backward()在缩放后的损失上以创建缩放后的梯度。
                scaler.scale(loss).backward()

                # scaler.step()首先对优化器参数的梯度进行反缩放。
                # 如果梯度不包含inf/nan，则调用optimizer.step()，
                # 否则跳过optimizer.step()。
                scaler.step(optimizer)

                # 更新下一轮迭代的缩放比例。
                scaler.update()

    参见:class:`GradScaler`以了解缩放/反缩放的解释和更多用例。
    """
    pass  # 类定义结束，无需进一步注释
    """
    Initialize a GradientScaler object.
    
    Args:
        init_scale (float, optional, default=2.**16):  Initial scale factor.
            Specifies the initial scaling factor applied to gradients.
        growth_factor (float, optional, default=2.0):  Factor by which the scale is multiplied during
            :meth:`update` if no inf/NaN gradients occur for ``growth_interval`` consecutive iterations.
            Determines how much the scale should grow when gradients are stable.
        backoff_factor (float, optional, default=0.5):  Factor by which the scale is multiplied during
            :meth:`update` if inf/NaN gradients occur in an iteration.
            Specifies the reduction factor when gradients become unstable.
        growth_interval (int, optional, default=2000):  Number of consecutive iterations without inf/NaN gradients
            that must occur for the scale to be multiplied by ``growth_factor``.
            Defines how many stable iterations are required before scaling up.
        enabled (bool, optional):  If ``False``, disables gradient scaling. :meth:`step` simply
            invokes the underlying ``optimizer.step()``, and other methods become no-ops.
            Default: ``True``
            Determines if gradient scaling functionality is active.
        process_group (ProcessGroup, optional, default=torch.distributed.group.WORLD):
            process group for sharding
            Specifies the distributed process group for synchronization.
    
    """
    
    def __init__(
        self,
        device: str = "cuda",
        init_scale: float = 2.0**16,
        backoff_factor: float = 0.5,
        growth_factor: float = 2.0,
        growth_interval: int = 2000,
        enabled: bool = True,
        process_group: Optional[ProcessGroup] = dist.group.WORLD,
    ) -> None:
        """
        Initialize the GradientScaler object with specified parameters.
    
        Args:
            device (str, optional): Specifies the device to be used ('cuda' by default).
            init_scale (float, optional): Initial scaling factor for gradients.
            backoff_factor (float, optional): Factor by which the scale is reduced on encountering inf/NaN gradients.
            growth_factor (float, optional): Factor by which the scale is increased during stable gradient phases.
            growth_interval (int, optional): Number of stable iterations before scaling up.
            enabled (bool, optional): Flag to enable/disable gradient scaling.
            process_group (ProcessGroup, optional): Distributed process group for synchronization.
        """
        super().__init__(
            device,
            init_scale=init_scale,
            backoff_factor=backoff_factor,
            growth_factor=growth_factor,
            growth_interval=growth_interval,
            enabled=enabled,
        )
        if self._enabled:
            self.process_group = process_group
            self._per_optimizer_states = defaultdict(_refresh_per_optimizer_state)
    
    @overload
    def scale(self, outputs: torch.Tensor) -> torch.Tensor:
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
        self, outputs: Union[torch.Tensor, Iterable[torch.Tensor]]
        ) -> Union[torch.Tensor, Iterable[torch.Tensor]]:
        """
        Scale gradients based on the type of output tensors.
    
        Args:
            outputs (Union[torch.Tensor, Iterable[torch.Tensor]]): Gradients to be scaled.
    
        Returns:
            Union[torch.Tensor, Iterable[torch.Tensor]]: Scaled gradients.
        """
    ) -> Union[torch.Tensor, Iterable[torch.Tensor]]:
        # 如果自动混合精度未启用，则直接返回输出
        if not self._enabled:
            return outputs

        # 如果输出是单个张量，则进行缩放处理
        if isinstance(outputs, torch.Tensor):
            # 确保输出张量位于支持的设备上
            assert _is_supported_device(outputs)
            
            # 如果尚未初始化缩放参数，则初始化
            if self._scale is None:
                self._lazy_init_scale_growth_tracker(outputs.device)
            
            # 确保缩放参数已存在
            assert self._scale is not None
            
            # 缩放输出张量，并保持返回值的 dtype 与输出张量一致
            scaled_output = outputs * self._scale.to(
                device=outputs.device, non_blocking=True
            )
            return scaled_output.type(outputs.dtype)

        # outputs 是一个可迭代对象时的处理
        stash: List[_GeneralMultiDeviceReplicator] = []

        def apply_scale(val: Union[torch.Tensor, Iterable[torch.Tensor]]):
            # 如果是张量，则进行缩放处理
            if isinstance(val, torch.Tensor):
                assert _is_supported_device(val)
                
                # 如果 stash 中尚无缩放对象，则初始化
                if len(stash) == 0:
                    if self._scale is None:
                        self._lazy_init_scale_growth_tracker(val.device)
                    assert self._scale is not None
                    stash.append(_GeneralMultiDeviceReplicator(self._scale))
                
                # 根据设备获取缩放比例，并进行缩放
                scaled_val = val * stash[0].get(val.device)
                
                # 确保返回值的 dtype 与输入张量一致
                return scaled_val.type(val.dtype)
            
            # 如果是可迭代对象，则递归应用缩放函数
            if isinstance(val, abc.Iterable):
                iterator = map(apply_scale, val)
                
                # 如果原始对象是 list 或 tuple，则返回相同类型的对象
                if isinstance(val, (list, tuple)):
                    return type(val)(iterator)
                
                # 否则返回迭代器对象
                return iterator
            
            # 如果既不是张量也不是可迭代对象，则抛出异常
            raise ValueError("outputs must be a Tensor or an iterable of Tensors")

        # 应用缩放函数到输出对象，并返回处理后的结果
        return apply_scale(outputs)

    def _foreach_non_finite_check_and_unscale_cpu_(
        self,
        grads: Sequence[torch.Tensor],
        found_inf: torch.Tensor,
        inv_scale: torch.Tensor,
    ) -> None:
        # 如果梯度列表为空，则直接返回
        if len(grads) == 0:
            return
        # 确保inv_scale是一个包含一个元素的张量
        assert inv_scale.numel() == 1, "inv_scale must be a 1-element tensor."
        # 确保found_inf是一个包含一个元素的张量
        assert found_inf.numel() == 1, "found_inf must be a 1-element tensor."

        # 遍历梯度列表
        for grad in grads:
            # 如果梯度不在CPU上
            if grad.device.type != "cpu":
                # 记录错误日志，显示当前梯度所在的设备类型
                logger.error(
                    "tensor device is %s but was expected to be ``cpu``",
                    grad.device,
                )
                # 抛出数值错误，说明梯度不在CPU上
                raise ValueError(
                    "Gradients were found on a non-CPU device when"
                    " expected to be on CPU."
                )
            # 如果梯度中包含无穷大或者NaN
            if (
                torch.isinf(grad).any().item() is True
                or torch.isnan(grad).any().item() is True
            ):
                # 将found_inf的数据设置为1.0的张量，并结束循环
                found_inf.data = torch.tensor([1.0])
                break
            else:
                # 否则，将梯度数据乘以inv_scale的数值
                grad.data *= inv_scale.item()

    def _unscale_grads_(
        self,
        optimizer: torch.optim.Optimizer,
        inv_scale: torch.Tensor,
        found_inf: torch.Tensor,
        allow_fp16: bool = True,
    # 如果未启用自动混合精度，则直接返回
    if not self._enabled:
        return

    # 检查和记录 scale growth 的状态
    self._check_scale_growth_tracker("unscale_")

    # 获取当前优化器的状态
    optimizer_state = self._per_optimizer_states[id(optimizer)]

    # 如果优化器状态已经是 UNSCALED，则抛出异常
    if optimizer_state["stage"] is OptState.UNSCALED:
        raise RuntimeError(
            "unscale_() has already been called on this optimizer since the last update()."
        )
    # 如果优化器状态是 STEPPED，则抛出异常
    elif optimizer_state["stage"] is OptState.STEPPED:
        raise RuntimeError("unscale_() is being called after step().")

    # 使用双精度计算反比例以避免 FP32 精度问题
    assert self._scale is not None
    inv_scale = self._scale.double().reciprocal().float()

    # 创建一个张量来标记是否找到无穷大值，初始值为 0.0
    found_inf = torch.full(
        (1,), 0.0, dtype=torch.float32, device=self._scale.device
    )

    # 调用 _unscale_grads_ 方法来处理梯度的反向操作，并记录找到的无穷大情况
    optimizer_state["found_inf_per_device"] = self._unscale_grads_(
        optimizer, inv_scale, found_inf, True
    )

    # 将优化器状态设置为 UNSCALED
    optimizer_state["stage"] = OptState.UNSCALED

    # 同步检测到的无穷大值跨多个设备
    optimizer_state = self._per_optimizer_states[id(optimizer)]
    works = []
    found_inf_on_cpus = []
    found_inf_on_devices = []

    # 遍历每个设备上的找到的无穷大值，并进行全局同步
    for found_inf in optimizer_state["found_inf_per_device"].values():
        if self._device != "cpu" and found_inf.device.type == "cpu":
            found_inf_on_cpus.append(found_inf)
            found_inf_on_device = found_inf.to(self._device)
            found_inf_on_devices.append(found_inf_on_device)
            works.append(
                dist.all_reduce(
                    found_inf_on_device, async_op=True, group=self.process_group
                )
            )
        else:
            works.append(
                dist.all_reduce(found_inf, async_op=True, group=self.process_group)
            )

    # 等待所有全局同步操作完成
    for work in works:
        work.wait()

    # 如果在 CPU 上找到了无穷大值，则复制到对应设备上
    if found_inf_on_cpus:
        torch._foreach_copy_(found_inf_on_cpus, found_inf_on_devices)


def _amp_update_scale_cpu_(self, found_inf: torch.Tensor) -> None:
    """
    如果 found_inf 是 1.0（True），则将 scale 乘以 backoff_factor，并将 growth_tracker 设置为零。
    否则，当达到增长间隔时，将 scale 乘以增长因子。
    """
    assert self._scale is not None and self._growth_tracker is not None

    # 如果 found_inf 大于等于 1.0，则根据 backoff_factor 调整 scale，并将 growth_tracker 清零
    if found_inf.item() >= 1.0:
        self._scale *= self._backoff_factor
        self._growth_tracker.fill_(0)
    else:
        # 否则，增加成功计数器，如果达到增长间隔，则根据 growth_factor 调整 scale，并清零 growth_tracker
        successful = self._growth_tracker + 1
        if successful == self._growth_interval:
            self._scale *= self._growth_factor
            self._growth_tracker.fill_(0)
        else:
            self._growth_tracker = successful
```