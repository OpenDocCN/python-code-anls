# `.\pytorch\torch\_C\_profiler.pyi`

```py
# 导入必要的模块
from enum import Enum  # 导入枚举类型的支持
from typing import Any, Literal  # 引入类型提示中的通用和文字类型
from typing_extensions import TypeAlias  # 导入类型别名支持

from torch._C import device, dtype, layout  # 从torch._C模块导入device、dtype和layout

# 定义在torch/csrc/profiler/python/init.cpp中的记录作用域
class RecordScope(Enum):
    FUNCTION = ...  # 函数作用域
    BACKWARD_FUNCTION = ...  # 反向函数作用域
    TORCHSCRIPT_FUNCTION = ...  # TorchScript函数作用域
    KERNEL_FUNCTION_DTYPE = ...  # 内核函数数据类型作用域
    CUSTOM_CLASS = ...  # 自定义类作用域
    BUILD_FEATURE = ...  # 构建特性作用域
    LITE_INTERPRETER = ...  # 轻量级解释器作用域
    USER_SCOPE = ...  # 用户作用域
    STATIC_RUNTIME_OP = ...  # 静态运行时操作作用域
    STATIC_RUNTIME_MODEL = ...  # 静态运行时模型作用域

# 定义性能分析器状态
class ProfilerState(Enum):
    Disable = ...  # 禁用状态
    CPU = ...  # CPU状态
    CUDA = ...  # CUDA状态
    NVTX = ...  # NVIDIA Tools Extension状态
    ITT = ...  # Intel Trace Analyzer and Collector状态
    KINETO = ...  # Kineto性能分析器状态
    KINETO_GPU_FALLBACK = ...  # Kineto GPU回退状态
    KINETO_PRIVATEUSE1_FALLBACK = ...  # Kineto私有用途1回退状态
    KINETO_PRIVATEUSE1 = ...  # Kineto私有用途1状态

# 定义激活性能分析器类型
class ActiveProfilerType(Enum):
    NONE = ...  # 无激活分析器
    LEGACY = ...  # 旧版性能分析器
    KINETO = ...  # Kineto性能分析器
    NVTX = ...  # NVIDIA Tools Extension性能分析器
    ITT = ...  # Intel Trace Tools性能分析器

# 定义性能分析活动类型
class ProfilerActivity(Enum):
    CPU = ...  # CPU性能分析活动
    CUDA = ...  # CUDA性能分析活动
    XPU = ...  # XPU性能分析活动
    MTIA = ...  # MTIA性能分析活动
    PrivateUse1 = ...  # 私有用途1性能分析活动

# 定义事件类型
class _EventType(Enum):
    TorchOp = ...  # Torch操作事件
    Backend = ...  # 后端事件
    Allocation = ...  # 分配事件
    OutOfMemory = ...  # 内存耗尽事件
    PyCall = ...  # Python调用事件
    PyCCall = ...  # C调用Python事件
    Kineto = ...  # Kineto事件

# 定义实验性配置类
class _ExperimentalConfig:
    def __init__(
        self,
        profiler_metrics: list[str] = ...,  # 性能分析器度量列表
        profiler_measure_per_kernel: bool = ...,  # 每个内核度量性能分析器
        verbose: bool = ...,  # 冗长输出标志
        performance_events: list[str] = ...,  # 性能事件列表
        enable_cuda_sync_events: bool = ...,  # 启用CUDA同步事件
    ) -> None: ...

# 定义性能分析器配置类
class ProfilerConfig:
    def __init__(
        self,
        state: ProfilerState,  # 性能分析器状态
        report_input_shapes: bool,  # 报告输入形状标志
        profile_memory: bool,  # 分析内存标志
        with_stack: bool,  # 带堆栈标志
        with_flops: bool,  # 带浮点操作数标志
        with_modules: bool,  # 带模块标志
        experimental_config: _ExperimentalConfig,  # 实验性配置
    ) -> None: ...

# 定义性能分析器事件类
class _ProfilerEvent:
    start_tid: int  # 开始线程ID
    start_time_ns: int  # 开始时间（纳秒）
    children: list[_ProfilerEvent]  # 子事件列表

    # TODO(robieta): remove in favor of `self.typed`
    extra_fields: _ExtraFields_TorchOp | _ExtraFields_Backend | _ExtraFields_Allocation | _ExtraFields_OutOfMemory | _ExtraFields_PyCall | _ExtraFields_PyCCall | _ExtraFields_Kineto

    @property
    def typed(
        self,
    ) -> (
        tuple[Literal[_EventType.TorchOp], _ExtraFields_TorchOp]
        | tuple[Literal[_EventType.Backend], _ExtraFields_Backend]
        | tuple[Literal[_EventType.Allocation], _ExtraFields_Allocation]
        | tuple[Literal[_EventType.OutOfMemory], _ExtraFields_OutOfMemory]
        | tuple[Literal[_EventType.PyCall], _ExtraFields_PyCall]
        | tuple[Literal[_EventType.PyCCall], _ExtraFields_PyCCall]
        | tuple[Literal[_EventType.Kineto], _ExtraFields_Kineto]
    ): ...  # 返回事件类型和额外字段的类型元组

    @property
    def name(self) -> str: ...  # 返回事件名称

    @property
    def tag(self) -> _EventType: ...  # 返回事件标签

    @property
    def id(self) -> int: ...  # 返回事件ID

    @property
    def parent(self) -> _ProfilerEvent | None: ...  # 返回父事件或空

    @property
    def correlation_id(self) -> int: ...  # 返回相关ID

    @property
    def end_time_ns(self) -> int: ...  # 返回结束时间（纳秒）

    @property
    def duration_time_ns(self) -> int: ...  # 返回持续时间（纳秒）

# 定义张量元数据类
class _TensorMetadata:
    impl_ptr: int | None  # 实现指针或空
    storage_data_ptr: int | None  # 存储数据指针或空
    # 定义一个属性 `id`，类型为 `int` 或者 `None`
    id: int | None

    # 定义一个名为 `allocation_id` 的属性，返回类型为 `int` 或者 `None`
    @property
    def allocation_id(self) -> int | None: ...

    # 定义一个名为 `layout` 的属性，返回类型为 `layout`
    @property
    def layout(self) -> layout: ...

    # 定义一个名为 `device` 的属性，返回类型为 `device`
    @property
    def device(self) -> device: ...

    # 定义一个名为 `dtype` 的属性，返回类型为 `dtype`
    @property
    def dtype(self) -> dtype: ...

    # 定义一个名为 `sizes` 的属性，返回类型为 `list[int]`
    @property
    def sizes(self) -> list[int]: ...

    # 定义一个名为 `strides` 的属性，返回类型为 `list[int]`
    @property
    def strides(self) -> list[int]: ...
# 定义一个标量类型别名，可以是 int、float、bool 或 complex 中的一种
Scalar: TypeAlias = int | float | bool | complex
# 定义输入参数的类型别名，可以是 _TensorMetadata 对象、_TensorMetadata 对象的列表、标量或 None
Input: TypeAlias = _TensorMetadata | list[_TensorMetadata] | Scalar | None

# 定义 _ExtraFields_TorchOp 类
class _ExtraFields_TorchOp:
    # 属性：操作名称，序列号，是否允许 TF32 CUBLAS
    name: str
    sequence_number: int
    allow_tf32_cublas: bool

    # @property 装饰器定义 inputs 属性，返回输入参数列表，每个参数可以是 Input 类型
    @property
    def inputs(self) -> list[Input]: ...
    
    # @property 装饰器定义 scope 属性，返回记录作用域的对象
    @property
    def scope(self) -> RecordScope: ...

# 定义 _ExtraFields_Backend 类
class _ExtraFields_Backend: ...

# 定义 _ExtraFields_Allocation 类
class _ExtraFields_Allocation:
    # 属性：指针，分配 ID（可选），分配大小，总共分配大小，总保留大小
    ptr: int
    id: int | None
    alloc_size: int
    total_allocated: int
    total_reserved: int

    # @property 装饰器定义 allocation_id 属性，返回分配 ID 或 None
    @property
    def allocation_id(self) -> int | None: ...
    
    # @property 装饰器定义 device 属性，返回设备信息
    @property
    def device(self) -> device: ...

# 定义 _ExtraFields_OutOfMemory 类
class _ExtraFields_OutOfMemory: ...

# 定义 _PyFrameState 类
class _PyFrameState:
    # 属性：行号，函数名称
    line_number: int
    function_name: str

    # @property 装饰器定义 file_name 属性，返回文件名
    @property
    def file_name(self) -> str: ...

# 定义 _NNModuleInfo 类
class _NNModuleInfo:
    # @property 装饰器定义 self_ptr 属性，返回 self 指针
    @property
    def self_ptr(self) -> int: ...
    
    # @property 装饰器定义 cls_ptr 属性，返回类指针
    @property
    def cls_ptr(self) -> int: ...
    
    # @property 装饰器定义 cls_name 属性，返回类名称
    @property
    def cls_name(self) -> str: ...
    
    # @property 装饰器定义 parameters 属性，返回参数列表，每个参数包含名称、_TensorMetadata 类型、可选的 _TensorMetadata 类型
    @property
    def parameters(
        self,
    ) -> list[tuple[str, _TensorMetadata, _TensorMetadata | None]]: ...

# 定义 _OptimizerInfo 类
class _OptimizerInfo:
    # @property 装饰器定义 parameters 属性，返回参数列表，每个参数包含 Parameter、梯度（如果存在）、优化器状态信息的元组
    @property
    def parameters(
        self,
    ) -> list[
        tuple[
            _TensorMetadata,
            _TensorMetadata | None,
            list[tuple[str, _TensorMetadata]],
        ]
    ]: ...

# 定义 _ExtraFields_PyCCall 类
class _ExtraFields_PyCCall:
    # @property 装饰器定义 caller 属性，返回 Python 调用帧状态对象
    @property
    def caller(self) -> _PyFrameState: ...

# 定义 _ExtraFields_PyCall 类
class _ExtraFields_PyCall:
    # @property 装饰器定义 callsite 属性，返回 Python 调用点帧状态对象
    @property
    def callsite(self) -> _PyFrameState: ...
    
    # @property 装饰器定义 caller 属性，返回 Python 调用帧状态对象
    @property
    def caller(self) -> _PyFrameState: ...
    
    # @property 装饰器定义 module 属性，返回神经网络模块信息对象或 None
    @property
    def module(self) -> _NNModuleInfo | None: ...
    
    # @property 装饰器定义 optimizer 属性，返回优化器信息对象或 None
    @property
    def optimizer(self) -> _OptimizerInfo | None: ...

# 定义 _ExtraFields_Kineto 类
class _ExtraFields_Kineto: ...

# 定义 _add_execution_trace_observer 函数，添加执行追踪观察者
def _add_execution_trace_observer(output_file_path: str) -> bool: ...

# 定义 _remove_execution_trace_observer 函数，移除执行追踪观察者
def _remove_execution_trace_observer() -> None: ...

# 定义 _enable_execution_trace_observer 函数，启用执行追踪观察者
def _enable_execution_trace_observer() -> None: ...

# 定义 _disable_execution_trace_observer 函数，禁用执行追踪观察者
def _disable_execution_trace_observer() -> None: ...

# 定义 _set_record_concrete_inputs_enabled_val 函数，设置记录具体输入值是否启用
def _set_record_concrete_inputs_enabled_val(val: bool) -> None: ...

# 定义 _set_fwd_bwd_enabled_val 函数，设置前向和后向是否启用
def _set_fwd_bwd_enabled_val(val: bool) -> None: ...

# 定义 _set_cuda_sync_enabled_val 函数，设置 CUDA 同步是否启用
def _set_cuda_sync_enabled_val(val: bool) -> None: ...

# 定义 CapturedTraceback 类，用于捕获的回溯信息
class CapturedTraceback: ...

# 定义 gather_traceback 函数，用于收集回溯信息，包括 Python、脚本、C++ 的信息
def gather_traceback(python: bool, script: bool, cpp: bool) -> CapturedTraceback: ...

# 定义 symbolize_tracebacks 函数，用于符号化回溯信息列表，每个回溯信息有 name、filename、line 字段
# 返回一个列表，每个元素都是一个字典列表，包含符号化后的信息
def symbolize_tracebacks(
    to_symbolize: list[CapturedTraceback],
) -> list[list[dict[str, str]]]: ...

# 定义 _RecordFunctionFast 类，用于快速记录函数调用信息
class _RecordFunctionFast:
    # 初始化函数，接受名称、输入值列表或元组或 None、关键字参数字典或 None
    def __init__(
        self,
        name: str,
        input_values: list | tuple | None = None,
        keyword_values: dict | None = None,
    ) -> None: ...
    
    # 进入上下文时调用的方法
    def __enter__(self) -> None: ...
    
    # 退出上下文时调用的方法
    def __exit__(self, *args: Any) -> None: ...
```