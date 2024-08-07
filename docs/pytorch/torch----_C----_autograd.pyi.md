# `.\pytorch\torch\_C\_autograd.pyi`

```py
# mypy: allow-untyped-defs
# 从枚举模块导入枚举类型和任意类型、可调用类型
from enum import Enum
from typing import Any, Callable

# 导入 PyTorch 深度学习框架
import torch
# 从 torch._C._profiler 模块导入各种类型和类
from torch._C._profiler import (
    _ProfilerEvent,            # 导入 _ProfilerEvent 类
    ActiveProfilerType,        # 导入 ActiveProfilerType 类
    ProfilerActivity,          # 导入 ProfilerActivity 类
    ProfilerConfig,            # 导入 ProfilerConfig 类
)

# torch/csrc/autograd/init.cpp 中定义的设备类型枚举
class DeviceType(Enum):
    CPU = ...                   # CPU 设备类型
    CUDA = ...                  # CUDA 设备类型
    XPU = ...                   # XPU 设备类型
    MKLDNN = ...                # MKLDNN 设备类型
    OPENGL = ...                # OPENGL 设备类型
    OPENCL = ...                # OPENCL 设备类型
    IDEEP = ...                 # IDEEP 设备类型
    HIP = ...                   # HIP 设备类型
    FPGA = ...                  # FPGA 设备类型
    MAIA = ...                  # MAIA 设备类型
    XLA = ...                   # XLA 设备类型
    MTIA = ...                  # MTIA 设备类型
    MPS = ...                   # MPS 设备类型
    HPU = ...                   # HPU 设备类型
    Meta = ...                  # Meta 设备类型
    Vulkan = ...                # Vulkan 设备类型
    Metal = ...                 # Metal 设备类型
    PrivateUse1 = ...           # 私有使用1 设备类型

# 分析器事件类
class ProfilerEvent:
    def cpu_elapsed_us(self, other: ProfilerEvent) -> float: ...
    def cpu_memory_usage(self) -> int: ...
    def cuda_elapsed_us(self, other: ProfilerEvent) -> float: ...
    def privateuse1_elapsed_us(self, other: ProfilerEvent) -> float: ...
    def cuda_memory_usage(self) -> int: ...
    def device(self) -> int: ...
    def handle(self) -> int: ...
    def has_cuda(self) -> bool: ...
    def is_remote(self) -> bool: ...
    def kind(self) -> int: ...
    def name(self) -> str: ...
    def node_id(self) -> int: ...
    def sequence_nr(self) -> int: ...
    def shapes(self) -> list[list[int]]: ...
    def thread_id(self) -> int: ...
    def flops(self) -> float: ...
    def is_async(self) -> bool: ...

# Kineto 事件类
class _KinetoEvent:
    def name(self) -> str: ...
    def device_index(self) -> int: ...
    def device_resource_id(self) -> int: ...
    def start_ns(self) -> int: ...
    def end_ns(self) -> int: ...
    def duration_ns(self) -> int: ...
    def is_async(self) -> bool: ...
    def linked_correlation_id(self) -> int: ...
    def shapes(self) -> list[list[int]]: ...
    def dtypes(self) -> list[str]: ...
    def concrete_inputs(self) -> list[Any]: ...
    def device_type(self) -> DeviceType: ...  # 返回设备类型的枚举
    def start_thread_id(self) -> int: ...
    def end_thread_id(self) -> int: ...
    def correlation_id(self) -> int: ...
    def fwd_thread_id(self) -> int: ...
    def stack(self) -> list[str]: ...
    def scope(self) -> int: ...
    def sequence_nr(self) -> int: ...
    def flops(self) -> int: ...
    def cuda_elapsed_us(self) -> int: ...
    def privateuse1_elapsed_us(self) -> int: ...

# 分析器结果类
class _ProfilerResult:
    def events(self) -> list[_KinetoEvent]: ...               # 返回 Kineto 事件列表
    def legacy_events(self) -> list[list[ProfilerEvent]]: ...  # 返回遗留事件列表
    def save(self, path: str) -> None: ...                    # 将结果保存到指定路径
    def experimental_event_tree(self) -> list[_ProfilerEvent]: ...  # 返回实验性事件树
    def trace_start_ns(self) -> int: ...                      # 返回跟踪起始时间戳

class SavedTensor: ...  # 保存的张量类

# 启用分析器函数，接受配置和活动集作为参数，无返回值
def _enable_profiler(
    config: ProfilerConfig,
    activities: set[ProfilerActivity],
) -> None: ...

# 准备分析器函数，接受配置和活动集作为参数，无返回值
def _prepare_profiler(
    config: ProfilerConfig,
    activities: set[ProfilerActivity],
) -> None: ...

# 禁用分析器函数，返回 _ProfilerResult 结果对象
def _disable_profiler() -> _ProfilerResult: ...

# 检查分析器是否启用函数，返回布尔值
def _profiler_enabled() -> bool: ...

# 添加元数据 JSON 函数，接受键和值作为参数，无返回值
def _add_metadata_json(key: str, value: str) -> None: ...

# Kineto 步骤函数，无参数和返回值
def _kineto_step() -> None: ...

# 获取序列号函数，返回整数
def _get_sequence_nr() -> int: ...

# 检查 Kineto 是否可用函数，返回布尔值
def kineto_available() -> bool: ...
# 定义一个函数 `_record_function_with_args_enter`，用于记录带有参数的函数进入事件
def _record_function_with_args_enter(name: str, *args) -> torch.Tensor:
    ...

# 定义一个函数 `_record_function_with_args_exit`，用于记录带有参数的函数退出事件
def _record_function_with_args_exit(handle: torch.Tensor) -> None:
    ...

# 定义一个函数 `_supported_activities`，返回支持的分析器活动的集合
def _supported_activities() -> set[ProfilerActivity]:
    ...

# 定义一个函数 `_enable_record_function`，用于启用或禁用记录函数
def _enable_record_function(enable: bool) -> None:
    ...

# 定义一个函数 `_set_empty_test_observer`，设置一个空的测试观察者
def _set_empty_test_observer(is_global: bool, sampling_prob: float) -> None:
    ...

# 定义一个函数 `_push_saved_tensors_default_hooks`，推入默认的保存张量钩子
def _push_saved_tensors_default_hooks(
    pack_hook: Callable[[torch.Tensor], Any],
    unpack_hook: Callable[[Any], torch.Tensor],
) -> None:
    ...

# 定义一个函数 `_pop_saved_tensors_default_hooks`，弹出默认的保存张量钩子
def _pop_saved_tensors_default_hooks() -> None:
    ...

# 定义一个函数 `_unsafe_set_version_counter`，不安全地设置张量的版本计数器
def _unsafe_set_version_counter(t: torch.Tensor, prev_version: int) -> None:
    ...

# 定义一个函数 `_enable_profiler_legacy`，启用旧版性能分析器
def _enable_profiler_legacy(config: ProfilerConfig) -> None:
    ...

# 定义一个函数 `_disable_profiler_legacy`，禁用旧版性能分析器，并返回事件列表的列表
def _disable_profiler_legacy() -> list[list[ProfilerEvent]]:
    ...

# 定义一个函数 `_profiler_type`，返回活动性能分析器类型
def _profiler_type() -> ActiveProfilerType:
    ...

# 定义一个函数 `_saved_tensors_hooks_enable`，启用保存张量的钩子
def _saved_tensors_hooks_enable() -> None:
    ...

# 定义一个函数 `_saved_tensors_hooks_disable`，禁用保存张量的钩子，并接收一个消息字符串参数
def _saved_tensors_hooks_disable(message: str) -> None:
    ...

# 定义一个函数 `_saved_tensors_hooks_get_disabled_error_message`，获取禁用保存张量钩子时的错误消息
def _saved_tensors_hooks_get_disabled_error_message() -> str | None:
    ...

# 定义一个函数 `_saved_tensors_hooks_set_tracing`，设置保存张量钩子是否追踪的状态，并返回当前状态
def _saved_tensors_hooks_set_tracing(is_tracing: bool) -> bool:
    ...

# 定义一个枚举类 `CreationMeta`，定义了张量创建元信息的不同模式
class CreationMeta(Enum):
    DEFAULT = ...
    IN_CUSTOM_FUNCTION = ...
    MULTI_OUTPUT_NODE = ...
    NO_GRAD_MODE = ...
    INFERENCE_MODE = ...

# 定义一个函数 `_set_creation_meta`，设置张量的创建元信息
def _set_creation_meta(t: torch.Tensor, creation_meta: CreationMeta) -> None:
    ...

# 定义一个函数 `_get_creation_meta`，获取张量的创建元信息
def _get_creation_meta(t: torch.Tensor) -> CreationMeta:
    ...
```