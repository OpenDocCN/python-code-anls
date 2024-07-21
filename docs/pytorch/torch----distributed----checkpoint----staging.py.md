# `.\pytorch\torch\distributed\checkpoint\staging.py`

```py
# 从 typing 模块导入 Optional 和 runtime_checkable 装饰器
from typing import Optional, runtime_checkable
# 从 typing_extensions 模块导入 Protocol 类
from typing_extensions import Protocol
# 从 torch.distributed._state_dict_utils 模块导入特定函数
from torch.distributed._state_dict_utils import (
    _copy_state_dict,
    _create_cpu_state_dict,
    _offload_state_dict_to_cpu,
)
# 从 torch.distributed.checkpoint.metadata 模块导入 STATE_DICT_TYPE 常量
from torch.distributed.checkpoint.metadata import STATE_DICT_TYPE

# 定义导出的所有符号（symbols）
__all__ = ["AsyncStager", "BlockingAsyncStager"]


@runtime_checkable
# 定义 AsyncStager 协议，用于定制和扩展 dcp.async_save 的行为
class AsyncStager(Protocol):
    """
    This protocol is meant to provide customization and extensibility for dcp.async_save, allowing users
    to customize how data is staged previous to executing the usual dcp.save path in parallel.
    The expected order of operations (concretely defined in `torch.distributed.state_dict_saver.async_save`)
    is the following:

    1. AsyncStager.stage_data(state_dict):
        This call gives the AsyncStager the opportunity to 'stage'
        the state_dict. The expectation and purpose of staging in this context is to create a "training-safe"
        representation of the state dict, meaning that any updates to module data after staging is complete
        should not be reflected in the state dict returned from this method. For example, in the default
        case a copy of the entire state dict is created on CPU RAM and returned here, allowing users
        to continue training without risking changes to data which is being serialized.

    2. dcp.save is called on the state_dict returned from stage in parallel. This call is responsible
        for serializing the state_dict and writing it to storage.

    3. If AsyncStager.should_synchronize_after_execute is True, this method will be called immediately after
        the serialization thread starts and before returning from dcp.async_save. If this is set to False,
        the assumption is the user has defined a custom synchronization point for the the purpose of further
        optimizing save latency in the training loop (for example, by overlapping staging with the
        forward/backward pass), and it is the responsibility of the user to call `AsyncStager.synchronize_staging`
        at the appropriate time.

    """

    # 默认为 True，因为通常情况下是同步进行阶段
    _synchronize_after_execute: bool = True

    @property
    def should_synchronize_after_execute(self) -> bool:
        """
        Whether to synchronize after executing the stage.
        """
        # 返回是否在执行阶段后进行同步的标志位
        return self._synchronize_after_execute

    def stage(self, state_dict: STATE_DICT_TYPE) -> STATE_DICT_TYPE:
        """
        Returns a "staged" copy of `state_dict`. The expectation of the staged copy is that it is
        inoculated from any updates incurred after the stage call is complete.
        """
        # 抛出未实现的错误，要求子类实现 stage 方法
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement stage method"
        )
    # 定义一个方法 `synchronize_staging`，用于同步暂存区的状态。
    # 如果 `stage` 操作在某种方式上是异步的，调用这个方法来确保暂存操作已完成，
    # 并且可以安全地开始修改原始的 `state_dict`。
    def synchronize_staging(self) -> None:
        """
        In the case `stage` is async in some way, this method should be called to ensure staging
        is complete and it is safe to begin modifying the original `state_dict`
        """
        pass
# 定义一个继承自 AsyncStager 的 BlockingAsyncStager 类，用于在 CPU 内存上进行状态字典的分段操作，且在复制完成前会阻塞。
# 此实现还提供了使用固定内存优化分段延迟的选项。
# 注意：在本例中，synchronize_staging 是一个空操作。

# 默认为 True，因为常见情况是同步分段
_synchronize_after_execute: bool = False

def __init__(
    self,
    cache_staged_state_dict: bool = False,
    type_check: bool = False,
):
    """
    初始化 BlockingAsyncStager。

    Args:
        cache_staged_state_dict: 是否缓存分段后的状态字典。此选项会降低分段延迟，但会增加内存使用。如果此参数设置为 True，则期望
            该分段器被维护并重复使用于多次 dcp.async_save 调用。默认为 False。
        type_check: 是否在 cpu_offload 过程中执行类型检查。默认为 False。
    """
    self.cache_staged_state_dict = cache_staged_state_dict
    self.type_check = type_check
    self.state_dict_cache: Optional[STATE_DICT_TYPE] = None

def stage(self, state_dict: STATE_DICT_TYPE) -> STATE_DICT_TYPE:
    """
    返回 `state_dict` 在 CPU 上的副本。
    """
    if not self.cache_staged_state_dict:
        return _offload_state_dict_to_cpu(state_dict, type_check=self.type_check)

    if self.state_dict_cache is None:
        self.state_dict_cache = _create_cpu_state_dict(state_dict, pin_memory=True)
    return _copy_state_dict(state_dict, self.state_dict_cache)

def synchronize_staging(self) -> None:
    """
    空操作函数，因为分段是阻塞的。
    """
    pass
```