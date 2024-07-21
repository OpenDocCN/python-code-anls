# `.\pytorch\torch\cuda\_sanitizer.py`

```py
# mypy: allow-untyped-defs
r"""
This module introduces CUDA Sanitizer, a tool for detecting synchronization errors between kernels ran on different streams.

It stores information on accesses to tensors to determine if they are synchronized
or not. When enabled in a python program and a possible data race is detected, a
detailed warning will be printed and the program will exit.

It can be enabled either by importing this module and calling
:func:`enable_cuda_sanitizer()` or by exporting the ``TORCH_CUDA_SANITIZER``
environment variable.
"""

import enum  # 导入枚举类型模块
import functools  # 导入函数工具模块
import inspect  # 导入检查模块
import io  # 导入IO模块
import logging  # 导入日志模块
import sys  # 导入系统相关模块
import textwrap  # 导入文本包装模块
import traceback  # 导入异常跟踪模块
from dataclasses import dataclass, field  # 导入数据类和域
from typing import Any, Dict, Iterator, List, Optional, Set, Tuple, TypeVar  # 导入类型注解相关

import torch  # 导入PyTorch模块
import torch.cuda._gpu_trace as gpu_trace  # 导入CUDA GPU跟踪模块
from torch.utils import _pytree as pytree  # 导入私有树模块
from torch.utils._python_dispatch import TorchDispatchMode  # 导入Torch分发模式


DEFAULT_STREAM_ID = 0  # 默认流ID为0

TK = TypeVar("TK")  # 类型变量TK
TVa = TypeVar("TVa")  # 类型变量TVa
TVb = TypeVar("TVb")  # 类型变量TVb

DataPtr = int  # 数据指针类型为整数
StreamId = int  # 流ID类型为整数
EventId = int  # 事件ID类型为整数
SeqNum = int  # 顺序号类型为整数

logger = logging.getLogger(__name__)  # 获取当前模块的日志记录器


class AccessType(enum.Enum):
    READ = enum.auto()  # 读取访问类型
    WRITE = enum.auto()  # 写入访问类型

    def __str__(self):
        return "reading from" if self is AccessType.READ else "writing to"


@dataclass
class Access:
    r"""Stores information about a single access to a tensor by a kernel.

    Args:
        type: either AccessType.READ or AccessType.Write.
        seq_num: the sequential number of the kernel performing the access.
        stream: the stream id of the stream executing the kernel.
        operator: the schema of the launched kernel, which lists the
            arguments and return type.
        aliases: the arguments in the schema this access corresponds to.
        is_output: Whether the tensor was an output of the kernel.
        stack_trace: the stack summary object captured during access.
    """

    type: AccessType  # 访问类型，可以是读或写
    seq_num: SeqNum  # 执行访问的内核的顺序号
    stream: StreamId  # 执行内核的流ID
    operator: str  # 启动内核的架构，列出参数和返回类型
    aliases: List[str]  # 与此访问对应的架构参数
    is_output: bool  # 张量是否是内核的输出
    stack_trace: traceback.StackSummary  # 访问期间捕获的堆栈摘要信息


class SynchronizationError(Exception):
    """Base class for errors detected by CUDA Sanitizer."""

    pass


class UnsynchronizedAccessError(SynchronizationError):
    """Stores information about two unsynchronized accesses to one data pointer."""

    def __init__(
        self,
        data_ptr: DataPtr,
        allocation_stack_trace: Optional[traceback.StackSummary],
        current_access: Access,
        previous_access: Access,
    ):
        self.data_ptr = data_ptr  # 数据指针
        self.allocation_stack_trace = allocation_stack_trace  # 分配的堆栈跟踪信息
        self.current_access = current_access  # 当前访问信息
        self.previous_access = previous_access  # 先前的访问信息
    # 定义对象的字符串表示形式，用于打印对象的详细信息
    def __str__(self):
        
        # 定义格式化访问信息的内部函数
        def format_access(access: Access):
            # 将访问操作符和类型写入消息
            message.write(f"{access.operator}\n{access.type}")
            # 如果有别名，则将其作为参数添加到消息中
            if access.aliases:
                message.write(" argument(s) " + ", ".join(access.aliases))
                # 如果是输出访问，则在消息中指出输出
                if access.is_output:
                    message.write(", and to")
            # 如果是输出访问，则在消息中指出输出
            if access.is_output:
                message.write(" the output")
            # 写入堆栈跟踪信息
            message.write(
                f"\nWith stack trace:\n{''.join(access.stack_trace.format())}\n"
            )

        # 使用字符串缓冲区，生成对象的字符串表示形式
        with io.StringIO() as message:
            # 写入带有缩进的消息头部
            message.write(
                textwrap.dedent(
                    f"""\
                    ============================
                    CSAN detected a possible data race on tensor with data pointer {self.data_ptr}
                    Access by stream {self.current_access.stream} during kernel:
                    """
                )
            )
            # 格式化当前访问的信息
            format_access(self.current_access)

            # 写入先前访问的信息
            message.write(
                f"Previous access by stream {self.previous_access.stream} during kernel:\n"
            )
            # 格式化先前访问的信息
            format_access(self.previous_access)

            # 如果有分配堆栈跟踪信息，则添加到消息中
            if self.allocation_stack_trace:
                message.write(
                    "Tensor was allocated with stack trace:\n"
                    f"{''.join(self.allocation_stack_trace.format())}"
                )
            else:
                # 如果没有分配堆栈跟踪信息，则指出未找到
                message.write("Trace for tensor allocation not found.")
            # 返回生成的消息文本
            return message.getvalue()
class CUDASanitizerErrors(Exception):
    """CUDA Sanitizer报告的错误的包装类。"""

    def __init__(self, errors: List[SynchronizationError]):
        self.errors = errors

    def __str__(self):
        return f"detected {len(self.errors)} errors"


@dataclass
class TensorInfo:
    r"""存储有关单个张量及其最近访问的信息。

    Args:
        allocation_stack_trace: 在张量分配期间捕获的堆栈摘要对象。如果分配未被CSAN捕获，则可以为“None”。
        reads: 自上次写入以来对张量的读取访问列表。
        write: 对张量的最后一次写入访问。
    """

    allocation_stack_trace: Optional[traceback.StackSummary]
    reads: List[Access] = field(default_factory=list)
    write: Optional[Access] = None


class _TensorsAccessed:
    def __init__(self):
        self.accesses: Dict[DataPtr, TensorInfo] = {}

    def ensure_tensor_exists(self, data_ptr: DataPtr) -> None:
        if data_ptr not in self.accesses:
            logger.info(
                "Found tensor with pointer: %s, but no matching tensor "
                "allocation in the trace. Backfilling the trace now. "
                "Perhaps the sanitizer was enabled after some torch operations?",
                data_ptr,
            )
            self.create_tensor(data_ptr, None)

    def ensure_tensor_does_not_exist(self, data_ptr: DataPtr) -> None:
        if data_ptr in self.accesses:
            logger.info(
                "Found duplicate tensor allocation in the trace for tensor with "
                "pointer: %s. Assuming the trace for tensor deallocation "
                "wasn't caught and backfilling it now. "
                "Perhaps the sanitizer was enabled after some torch operations?",
                data_ptr,
            )
            self.delete_tensor(data_ptr)

    def create_tensor(
        self, data_ptr: DataPtr, stack_trace: Optional[traceback.StackSummary]
    ) -> None:
        """创建一个新的张量并将其添加到访问字典中。"""
        self.accesses[data_ptr] = TensorInfo(stack_trace)

    def delete_tensor(self, data_ptr: DataPtr) -> None:
        """从访问字典中删除指定的张量。"""
        del self.accesses[data_ptr]

    def were_there_reads_since_last_write(self, data_ptr: DataPtr) -> bool:
        """检查自上次写入以来是否有对张量的读取访问。"""
        return True if self.accesses[data_ptr].reads else False

    def get_allocation_stack_trace(
        self, data_ptr: DataPtr
    ) -> Optional[traceback.StackSummary]:
        """获取指定张量的分配时堆栈摘要。"""
        return self.accesses[data_ptr].allocation_stack_trace

    def get_write(self, data_ptr: DataPtr) -> Optional[Access]:
        """获取指定张量的最后一次写入访问。"""
        return self.accesses[data_ptr].write

    def get_reads(self, data_ptr: DataPtr) -> List[Access]:
        """获取指定张量的所有读取访问列表。"""
        return self.accesses[data_ptr].reads

    def add_read(self, data_ptr: DataPtr, access: Access) -> None:
        """向指定张量的读取访问列表中添加新的访问。"""
        self.accesses[data_ptr].reads.append(access)
    # 定义一个方法 `set_write`，用于设置数据指针的写入访问权限
    def set_write(self, data_ptr: DataPtr, access: Access) -> None:
        # 设置给定数据指针的写入权限
        self.accesses[data_ptr].write = access
        # 将给定数据指针的读取权限列表清空
        self.accesses[data_ptr].reads = []
# 定义一个名为 StreamSynchronizations 的类，用于管理流同步状态的记录和操作
class StreamSynchronizations:
    
    # 初始化方法，设置初始同步状态字典和主机同步状态字典，并创建默认流
    def __init__(self):
        self.current_sync_states: Dict[StreamId, Dict[StreamId, SeqNum]] = {}
        self.recorded_sync_states: Dict[EventId, Dict[StreamId, SeqNum]] = {}
        self.host_sync_state: Dict[StreamId, SeqNum] = {}
        self.create_stream(DEFAULT_STREAM_ID)

    # 确保指定流存在的内部方法，如果不存在则创建，同时记录日志
    def _ensure_stream_exists(self, stream: StreamId) -> None:
        if stream not in self.current_sync_states:
            logger.info(
                "Found Stream with id: %s, but no matching stream "
                "creation in the trace. Backfilling the trace now. "
                "Perhaps the sanitizer was enabled after some torch operations?",
                stream,
            )
            self.create_stream(stream)

    # 确保指定事件存在的内部方法，如果不存在则创建，同时记录日志
    def _ensure_event_exists(self, event: EventId) -> None:
        if event not in self.recorded_sync_states:
            logger.info(
                "Found Event with id: %s, but no matching event "
                "creation in the trace. Backfilling the trace now. "
                "Perhaps the sanitizer was enabled after some torch operations?",
                event,
            )
            self.create_event(event)

    # 确保指定事件不存在的内部方法，如果存在则删除，同时记录日志
    def _ensure_event_does_not_exist(self, event: EventId) -> None:
        if event in self.recorded_sync_states:
            logger.info(
                "Found duplicate event creation in the trace for event with "
                "id: %s. Assuming the trace for event deletion wasn't caught "
                "and backfilling it now. "
                "Perhaps the sanitizer was enabled after some torch operations?",
                event,
            )
            self.delete_event(event)

    # 创建新流的方法，如果流已存在则记录日志，否则初始化主机同步状态并复制到当前同步状态字典
    def create_stream(self, stream: StreamId) -> None:
        if stream in self.current_sync_states:
            logger.info(
                "Found duplicate Stream creation in the trace for Stream with "
                "id: %s. PyTorch Streams are only created once, so this "
                "trace entry is ignored.",
                stream,
            )
        else:
            self.host_sync_state[stream] = 0
            self.current_sync_states[stream] = self.host_sync_state.copy()

    # 创建新事件的方法，首先确保事件不存在，然后将其添加到记录的同步状态字典中
    def create_event(self, event: EventId) -> None:
        self._ensure_event_does_not_exist(event)
        self.recorded_sync_states[event] = {}

    # 删除事件的方法，首先确保事件存在，然后从记录的同步状态字典中删除
    def delete_event(self, event: EventId) -> None:
        self._ensure_event_exists(event)
        del self.recorded_sync_states[event]

    # 更新指定流的序列号的方法，确保流存在后，更新当前同步状态字典中该流的序列号
    def update_seq_num(self, stream: StreamId, seq_num: SeqNum) -> None:
        self._ensure_stream_exists(stream)
        self.current_sync_states[stream][stream] = seq_num

    # 记录事件的状态的方法，确保事件和流均存在后，将当前同步状态字典中指定流的复制添加到记录的同步状态字典中
    def record_state(self, event: EventId, stream: StreamId) -> None:
        self._ensure_event_exists(event)
        self._ensure_stream_exists(stream)
        self.recorded_sync_states[event] = self.current_sync_states[stream].copy()

    # 内部方法，用于等待另一个状态的同步，接受两个同步状态字典作为参数
    # 更新状态字典，确保每个流的状态号按最大值更新
    def _state_wait_for_other(self, state: SyncState, other: SyncState) -> None:
        for stream, seq_num in other.items():
            state[stream] = max(state.get(stream, -1), seq_num)

    # 让特定流等待指定事件的发生
    def stream_wait_for_event(self, stream: StreamId, event: EventId) -> None:
        self._ensure_stream_exists(stream)  # 确保流存在
        self._ensure_event_exists(event)    # 确保事件存在
        self._state_wait_for_other(
            self.current_sync_states[stream], self.recorded_sync_states[event]
        )

    # 让所有流等待指定事件的发生
    def all_streams_wait_for_event(self, event: EventId) -> None:
        self._ensure_event_exists(event)    # 确保事件存在
        for stream in self.current_sync_states.keys():
            self.stream_wait_for_event(stream, event)

        self._state_wait_for_other(
            self.host_sync_state, self.recorded_sync_states[event]
        )

    # 让所有流等待指定流的状态更新
    def all_streams_wait_for_stream(self, stream: StreamId) -> None:
        self._ensure_stream_exists(stream)  # 确保流存在
        for state in self.current_sync_states.values():
            self._state_wait_for_other(state, self.current_sync_states[stream])

        self._state_wait_for_other(
            self.host_sync_state, self.current_sync_states[stream]
        )

    # 同步所有流的状态到主机状态
    def sync_all_streams(self) -> None:
        for stream, state in self.current_sync_states.items():
            self.host_sync_state[stream] = state[stream]

        for state in self.current_sync_states.values():
            self._state_wait_for_other(state, self.host_sync_state)

    # 检查当前流的序号是否在另一流之后
    def is_ordered_after(
        self, current_stream: StreamId, seq_num: SeqNum, other_stream: StreamId
    ) -> bool:
        self._ensure_stream_exists(current_stream)  # 确保当前流存在
        self._ensure_stream_exists(other_stream)    # 确保另一流存在
        return seq_num <= self.current_sync_states[current_stream].get(other_stream, -1)
class EventHandler:
    """Analyzes CSAN trace for synchronization errors.

    Stores information on each stream's synchronizations with other streams as well
    as tensor accesses to determine whether a given kernel launch might cause a
    data race.
    """

    def __init__(self):
        # 初始化 _TensorsAccessed 实例，用于跟踪张量的访问情况
        self.tensors_accessed = _TensorsAccessed()
        # 初始化 StreamSynchronizations 实例，用于跟踪流之间的同步操作
        self.syncs = StreamSynchronizations()
        # 初始化序列号为 0，用于标识事件处理器中的顺序
        self.seq_num: SeqNum = 0

    def _handle_kernel_launch(
        self,
        stream: StreamId,
        read_only: Set[DataPtr],
        read_write: Set[DataPtr],
        outputs: Set[DataPtr],
        operator: str,
        tensor_aliases: Dict[int, List[str]],
    ) -> List[SynchronizationError]:
        def check_conflict(
            data_ptr: DataPtr, current_access: Access, previous_access: Optional[Access]
        ) -> None:
            # 检查是否存在冲突的访问，如果前一个访问为None，则不需要检查
            if previous_access is None:
                return
            # 检查当前访问是否在前一个访问之后发生，如果不是，则添加未同步访问错误
            if not self.syncs.is_ordered_after(
                current_access.stream, previous_access.seq_num, previous_access.stream
            ):
                error_list.append(
                    UnsynchronizedAccessError(
                        data_ptr,
                        self.tensors_accessed.get_allocation_stack_trace(data_ptr),
                        current_access,
                        previous_access,
                    )
                )

        error_list: List[SynchronizationError] = []
        # 增加序列号，用于标识访问顺序的增加
        self.seq_num += 1
        # 更新同步对象的序列号
        self.syncs.update_seq_num(stream, self.seq_num)
        # 提取当前堆栈的追踪信息，由于其顺序是反向的，需要反转
        stack_trace = traceback.StackSummary.extract(
            traceback.walk_stack(inspect.currentframe()), lookup_lines=False
        )
        # 反转堆栈追踪，以便按照调用顺序记录
        stack_trace.reverse()

        # 对只读数据进行访问处理
        for data_ptr in read_only:
            # 确保数据指针存在于被访问的张量中
            self.tensors_accessed.ensure_tensor_exists(data_ptr)
            # 创建当前访问的Access对象，标记为读取操作
            current_access = Access(
                AccessType.READ,
                self.seq_num,
                stream,
                operator,
                tensor_aliases[data_ptr],
                data_ptr in outputs,
                stack_trace,
            )
            # 检查当前读取是否与之前的写操作存在冲突
            check_conflict(
                data_ptr, current_access, self.tensors_accessed.get_write(data_ptr)
            )
            # 将当前读取操作记录到被访问的张量中
            self.tensors_accessed.add_read(data_ptr, current_access)

        # 对读写数据进行访问处理
        for data_ptr in read_write:
            # 确保数据指针存在于被访问的张量中
            self.tensors_accessed.ensure_tensor_exists(data_ptr)
            # 创建当前访问的Access对象，标记为写入操作
            current_access = Access(
                AccessType.WRITE,
                self.seq_num,
                stream,
                operator,
                tensor_aliases[data_ptr],
                data_ptr in outputs,
                stack_trace,
            )
            # 如果在上次写入之后存在读取操作，则检查每次读取操作是否与当前写入操作存在冲突
            if self.tensors_accessed.were_there_reads_since_last_write(data_ptr):
                for previous_access in self.tensors_accessed.get_reads(data_ptr):
                    check_conflict(data_ptr, current_access, previous_access)
            else:
                # 否则，检查当前写入操作是否与之前的写操作存在冲突
                check_conflict(
                    data_ptr, current_access, self.tensors_accessed.get_write(data_ptr)
                )
            # 记录当前写入操作到被访问的张量中
            self.tensors_accessed.set_write(data_ptr, current_access)

        # 返回检测到的所有同步错误列表
        return error_list

    def _handle_event_creation(self, event: EventId) -> None:
        # 创建事件处理函数，将事件添加到同步对象中
        self.syncs.create_event(event)

    def _handle_event_deletion(self, event: EventId) -> None:
        # 删除事件处理函数，将事件从同步对象中移除
        self.syncs.delete_event(event)

    def _handle_event_record(self, event: EventId, stream: StreamId) -> None:
        # 记录事件处理函数，记录事件在指定流中的状态
        self.syncs.record_state(event, stream)
    # 处理等待事件的方法，根据给定的事件和流 ID 进行同步
    def _handle_event_wait(self, event: EventId, stream: StreamId) -> None:
        self.syncs.stream_wait_for_event(stream, event)

    # 处理内存分配的方法，确保数据指针不存在，生成调用堆栈追踪并创建张量
    def _handle_memory_allocation(self, data_ptr: DataPtr) -> None:
        self.tensors_accessed.ensure_tensor_does_not_exist(data_ptr)
        # 从当前堆栈中提取调用堆栈摘要，不包括源代码行号查找
        stack_trace = traceback.StackSummary.extract(
            traceback.walk_stack(inspect.currentframe()), lookup_lines=False
        )
        # 由于生成的堆栈追踪是逆序的，因此需要反转顺序
        stack_trace.reverse()
        self.tensors_accessed.create_tensor(
            data_ptr,
            stack_trace,
        )

    # 处理内存释放的方法，确保数据指针存在，删除张量
    def _handle_memory_deallocation(self, data_ptr: DataPtr) -> None:
        self.tensors_accessed.ensure_tensor_exists(data_ptr)
        self.tensors_accessed.delete_tensor(data_ptr)

    # 处理流创建的方法，创建新的数据流
    def _handle_stream_creation(self, stream: StreamId) -> None:
        self.syncs.create_stream(stream)

    # 处理设备同步的方法，同步所有的数据流
    def _handle_device_synchronization(self) -> None:
        self.syncs.sync_all_streams()

    # 处理流同步的方法，使所有数据流等待特定数据流的同步
    def _handle_stream_synchronization(self, stream: StreamId) -> None:
        self.syncs.all_streams_wait_for_stream(stream)

    # 处理事件同步的方法，使所有数据流等待特定事件的同步
    def _handle_event_synchronization(self, event: EventId) -> None:
        self.syncs.all_streams_wait_for_event(event)
# 定义一个函数，接受两个泛型字典参数 a 和 b，并生成元组的迭代器，包含键、a 中对应值和 b 中对应值
def zip_by_key(a: Dict[TK, TVa], b: Dict[TK, TVb]) -> Iterator[Tuple[TK, TVa, TVb]]:
    for arg, value in a.items():
        # 如果 a 的键存在于 b 中，则生成包含键、a 的值和 b 的值的元组
        if arg in b:
            yield arg, value, b[arg]

# 定义一个函数，接受 torch 函数模式 schema、位置参数 args 和关键字参数 kwargs，并生成元组的迭代器，包含 torch.Argument 和对应的值
def zip_arguments(
    schema: torch.FunctionSchema, args: Tuple[Any, ...], kwargs: Dict[str, Any]
) -> Iterator[Tuple[torch.Argument, Any]]:
    # 获取与位置参数对应的模式参数
    schema_args = schema.arguments[: len(args)]
    # 创建一个字典，将模式参数的名称映射到其本身
    schema_kwargs = {arg.name: arg for arg in schema.arguments[len(args) :]}

    # 生成位置参数与模式参数的元组迭代器
    yield from zip(schema_args, args)

    # 使用 zip_by_key 函数生成关键字参数与模式关键字参数的元组迭代器
    for _, argument, value in zip_by_key(schema_kwargs, kwargs):
        yield (argument, value)

# 定义一个类 ArgumentHandler
class ArgumentHandler:
    def __init__(self):
        # 初始化四个属性，分别用于记录读取的数据指针集合、写入的数据指针集合、张量别名字典和输出数据指针集合
        self.dataptrs_read: Set[DataPtr] = set()
        self.dataptrs_written: Set[DataPtr] = set()
        self.tensor_aliases: Dict[DataPtr, List[str]] = dict()
        self.outputs: Set[DataPtr] = set()

    # 定义一个私有方法 _handle_argument，用于处理单个参数
    def _handle_argument(
        self,
        value: Any,
        is_write: bool,
        name: Optional[str] = None,
        is_output: bool = False,
    ) -> None:
        # 如果 value 是 torch.Tensor 类型且在 CUDA 上
        if isinstance(value, torch.Tensor) and value.is_cuda:
            # 获取数据指针
            data_ptr = value.data_ptr()
            # 根据参数 is_write 决定将数据指针添加到读取或写入集合中
            if is_write:
                self.dataptrs_written.add(data_ptr)
            else:
                self.dataptrs_read.add(data_ptr)

            # 如果数据指针不在 tensor_aliases 字典中，则设为一个空列表
            self.tensor_aliases.setdefault(data_ptr, [])
            # 如果参数 name 不为 None，则将其添加到 tensor_aliases 中对应数据指针的列表中
            if name is not None:
                self.tensor_aliases[data_ptr].append(name)
            # 如果是输出参数，则将数据指针添加到 outputs 集合中
            if is_output:
                self.outputs.add(data_ptr)

    # 定义方法 parse_inputs，解析输入参数
    def parse_inputs(
        self,
        schema: torch.FunctionSchema,
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> None:
        # 使用 zip_arguments 函数生成参数与值的元组迭代器，并根据每个参数调用 _handle_argument 处理
        for argument, value in zip_arguments(schema, args, kwargs):
            # 判断是否为写入参数，并进行相应处理
            is_write = argument.alias_info is not None and argument.alias_info.is_write
            pytree.tree_map_(
                functools.partial(
                    self._handle_argument, is_write=is_write, name=argument.name
                ),
                value,
            )

    # 定义方法 parse_outputs，解析输出参数
    def parse_outputs(self, outputs: Any) -> None:
        # 使用 pytree.tree_map 函数，对输出数据的每个元素调用 _handle_argument 处理，标记为写入和输出参数
        pytree.tree_map_(
            functools.partial(self._handle_argument, is_write=True, is_output=True),
            outputs,
        )


# 定义一个类 CUDASanitizerDispatchMode，继承于 TorchDispatchMode
class CUDASanitizerDispatchMode(TorchDispatchMode):
    pass  # 空的类定义，无需额外注释
    # 初始化函数，用于设置事件处理器和注册GPU跟踪的回调函数
    def __init__(self):
        # 创建事件处理器实例
        self.event_handler = EventHandler()
        # 激活GPU跟踪
        torch._C._activate_gpu_trace()
        # 注册事件创建回调函数
        gpu_trace.register_callback_for_event_creation(
            self.event_handler._handle_event_creation
        )
        # 注册事件删除回调函数
        gpu_trace.register_callback_for_event_deletion(
            self.event_handler._handle_event_deletion
        )
        # 注册事件记录回调函数
        gpu_trace.register_callback_for_event_record(
            self.event_handler._handle_event_record
        )
        # 注册事件等待回调函数
        gpu_trace.register_callback_for_event_wait(
            self.event_handler._handle_event_wait
        )
        # 注册内存分配回调函数
        gpu_trace.register_callback_for_memory_allocation(
            self.event_handler._handle_memory_allocation
        )
        # 注册内存释放回调函数
        gpu_trace.register_callback_for_memory_deallocation(
            self.event_handler._handle_memory_deallocation
        )
        # 注册流创建回调函数
        gpu_trace.register_callback_for_stream_creation(
            self.event_handler._handle_stream_creation
        )
        # 注册设备同步回调函数
        gpu_trace.register_callback_for_device_synchronization(
            self.event_handler._handle_device_synchronization
        )
        # 注册流同步回调函数
        gpu_trace.register_callback_for_stream_synchronization(
            self.event_handler._handle_stream_synchronization
        )
        # 注册事件同步回调函数
        gpu_trace.register_callback_for_event_synchronization(
            self.event_handler._handle_event_synchronization
        )

    # Torch分发函数，处理GPU跟踪数据并调用相应的内核函数
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        # 如果kwargs为空，则初始化为空字典
        if kwargs is None:
            kwargs = {}

        # 创建参数处理器实例
        argument_handler = ArgumentHandler()
        # 解析输入参数
        argument_handler.parse_inputs(func._schema, args, kwargs)

        # 调用函数并获取输出
        outputs = func(*args, **kwargs)

        # 解析输出参数
        argument_handler.parse_outputs(outputs)

        # 处理内核启动事件，记录相关GPU操作
        errors = self.event_handler._handle_kernel_launch(
            torch.cuda.current_stream().cuda_stream,
            argument_handler.dataptrs_read - argument_handler.dataptrs_written,
            argument_handler.dataptrs_written,
            argument_handler.outputs,
            func._schema,
            argument_handler.tensor_aliases,
        )

        # 如果有错误信息，则逐个输出并抛出异常
        if errors:
            for error in errors:
                print(error, file=sys.stderr)
            raise CUDASanitizerErrors(errors)

        # 返回函数的输出结果
        return outputs
class CUDASanitizer:
    """Manages the lifetime of a CUDASanitizer dispatch mode object.

    The CUDASanitizer class wraps the entering/exiting functions of the dispatch mode
    context manager in the enable function/destructor, respectively. This is to
    explicitly set the lifetime of the dispatch mode object to that of the application.
    This approach was deemed more elegant than using the atexit module.
    """

    def __init__(self):
        # 创建一个 CUDASanitizerDispatchMode 实例作为 dispatch 对象
        self.dispatch = CUDASanitizerDispatchMode()
        # 初始化 enabled 属性为 False
        self.enabled = False

    def enable(self):
        # 进入 dispatch 对象的上下文管理器，启用 CUDA Sanitizer
        self.dispatch.__enter__()
        # 将 enabled 属性设置为 True，表示 Sanitizer 已启用
        self.enabled = True

    def __del__(self):
        # 析构函数，在对象销毁时调用
        if self.enabled:
            # 如果 Sanitizer 已启用，退出 dispatch 对象的上下文管理器
            self.dispatch.__exit__(None, None, None)


def enable_cuda_sanitizer():
    """Enable CUDA Sanitizer.

    The sanitizer will begin to analyze low-level CUDA calls invoked by torch functions
    for synchronization errors. All data races found will be printed to the standard
    error output along with stack traces of suspected causes. For best results, the
    sanitizer should be enabled at the very beginning of the program.
    """
    # 调用 cuda_sanitizer 对象的 enable 方法，启用 CUDA Sanitizer
    cuda_sanitizer.enable()


# 创建全局变量 cuda_sanitizer，实例化 CUDASanitizer 类
cuda_sanitizer = CUDASanitizer()
```