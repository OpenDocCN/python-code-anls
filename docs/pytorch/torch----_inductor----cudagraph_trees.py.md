# `.\pytorch\torch\_inductor\cudagraph_trees.py`

```py
# mypy: allow-untyped-defs
"""
CUDA graph trees are a safety abstraction over CUDAGraphs, similar to make_graph_callables,
which share the same memory pool.  Sharing a memory pool is an extremely
important optimization when chaining multiple CUDA graphs together, as it
prevents you from needing to copy intermediate tensors from one graph to the
next, and reduces overall memory usage by allowing dead memory from the first
pool to be reused in the second.

The standard graph/make_graph_callables support sharing memory pool, but
with a lot of caveats.  CUDA graph trees remove these restrictions:

* Previously, if you recorded graphs A, B, you had to replay A, B in that
  order.  With CUDA graph trees, after replaying A, you can change your
  mind and record/replay a different graph B'; we will support efficient
  execution of both A, B and A, B', using only max(mem(A, B), mem(A, B')).  In
  other words: we support arbitrary trees of CUDA graph operations, not just
  sequences (this is why this feature is called CUDA graph trees.)

* Previously, if you executed graph A, some non-CUDA graph code, and then
  graph B, after executing graph B, it was not safe to retain any references
  to intermediates produced by A.  With CUDA graph trees, we track if any
  outputs of graph A are still live by the time graph B is run, and make
  sure graph B doesn't clobber there memory when reusing the CUDA graphs
  pool.  You'll get a separate recording of B depending on what tensors
  stay live or dead.

CUDA graph trees are flexible enough to be used in Dynamo across graph breaks,
which is their primary use case.

The ability to switch from replay to record is fairly nontrivial: remember that
when you replay a CUDA graph, you only replay CUDA operations; no CPU side state
is updated.  In particular, the CPU-side book-keeping for the allocator is not
reconstructed.  However, to record a new child CUDA graph, we must restore this
book-keeping.  This is what checkpoint pool state is used for.
"""

# 导入必要的模块和库
from __future__ import annotations
import contextlib
import dataclasses
import functools
import gc
import itertools
import operator
import sys
import threading
import traceback
import warnings
import weakref
from collections import defaultdict

# 导入枚举类型和类型提示
from enum import auto, Enum
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    TYPE_CHECKING,
    Union,
)

# 导入 torch 相关模块和函数
import torch.fx
from torch import Tensor
from torch._dynamo.mutation_guard import GenerationTracker
from torch._dynamo.utils import counters, preserve_rng_state
from torch._inductor.compile_fx import (
    align_inputs_from_check_idxs,
    copy_misaligned_inputs,
    get_expanded_dims,
    get_input_idxs_to_check,
    index_expanded_dims,
    remove_unaligned_input_idxs,
    static_input,
)
from torch._inductor.cudagraph_utils import (
    check_for_mutation,
    FunctionID,
    get_placeholder_stack_trace,
    log_cudagraph_skip_and_bump_counter,
)
    # 导入 WrappedFunction 模块或类
    WrappedFunction,
)
from torch.multiprocessing.reductions import StorageWeakRef
from torch.storage import UntypedStorage
from torch.utils import _pytree as pytree
from torch.utils.weak import TensorWeakRef

if TYPE_CHECKING:
    from torch.types import _bool

# 定义几个类型别名
StorageWeakRefPointer = int  # 弱引用指针类型为整数
StorageDataPtr = int  # 存储数据指针类型为整数
NBytes = int  # 字节数类型为整数

# 如果 Torch 使用了 CUDA 构建
if torch.backends.cuda.is_built():
    from torch._C import (
        _cuda_CUDAAllocator_AllocatorState as AllocatorState,
        _set_cached_tensors_enabled as _set_cached_tensors_enabled,
    )
else:
    # 否则定义一个空的 AllocatorState 类和一个空的函数 _set_cached_tensors_enabled
    class AllocatorState:  # type: ignore[no-redef]
        pass

    def _set_cached_tensors_enabled(enabled: _bool) -> None:
        pass

# 获取名为 "cudagraphs" 的日志记录器对象
log = torch._logging.getArtifactLogger(__name__, "cudagraphs")

# 导入当前目录下的 config 模块
from . import config

# 定义一个数据类 GraphID，用于记录 CUDA 图的唯一计数器
@dataclasses.dataclass(frozen=True)
class GraphID:
    "Unique counter of a cuda graph recording"
    id: int

# 清除 Cublas 的缓存，以避免在不同运行间保留持久化的工作空间分配问题
def clear_cublass_cache():
    """
    Cublas keeps a persistent workspace allocation for running matmuls. This poses a problem for
    doing warmup within a CUDAGraph private pool because we do not want persistent allocations from
    one one run to the next. When we begin a new run of a cudagraphs path (generation), all tensors
    from the previous generation are freed. This frees them the memory pool, but not elsewhere.
    A tensor in the cublas workspace would continue to be in use the workspace but would also get allocated
    in the next run. The memory would be in use in two places.

    To solve this, we clear cublas caches before and after warming up or recording. If a workspace is required
    it will be allocated to the cudagraph private pool and accounted for in the allocator for the duration of the
    program. There is no overhead to this on replay since cudagraphs removes allocation overhead.
    """
    torch._C._cuda_clearCublasWorkspaces()

# 上下文管理器，用于清除 Cublas 缓存
@contextlib.contextmanager
def clear_cublas_manager():
    "Context manager around clearing cublas caches that will clear on enter and exit"
    clear_cublass_cache()
    try:
        yield
    finally:
        clear_cublass_cache()

# 上下文管理器，用于禁止卷积缓存的清空
@contextlib.contextmanager
def disable_conv_cache_emptying():
    prev = torch._C._cuda_get_conv_benchmark_empty_cache()
    torch._C._cudnn_set_conv_benchmark_empty_cache(False)
    try:
        yield
    finally:
        torch._C._cudnn_set_conv_benchmark_empty_cache(prev)

# 上下文管理器，用于启用内存历史记录
@contextlib.contextmanager
def enable_history_recording():
    "Turns on history recording in the CUDA Caching Allocator"
    enabled = torch._C._cuda_isHistoryEnabled()
    try:
        if not enabled:
            torch.cuda.memory._record_memory_history()
        yield
    finally:
        if not enabled:
            torch.cuda.memory._record_memory_history(None)

# 获取内存历史记录的上下文管理器
def get_history_recording():
    # 如果配置中禁用了 Triton 的 cudagraph_trees_history_recording，则返回一个空的上下文管理器
    if not config.triton.cudagraph_trees_history_recording:
        return contextlib.nullcontext()
    return enable_history_recording()

# TreeManagerContainer 类的定义
class TreeManagerContainer:
    """
    """
    Manages the lifetime of the tree manager. Like `PrivatePool` in cuda caching allocator,
    the tree and its corresponding memory pool should be kept alive as long as any outstanding
    graph or tensor which is an output of a graph remains alive.

    There is a single tree manager container per device.

    The lifecycle of a tree_manager is:
    -  Is constructed, no graph, no fns, no tensors
    -  Tree manager is fetched, resulting in tree manager being allocated
    -  We generate a bunch of functions, calling add_strong_reference
    -  These functions die, calling finalize_reference
    -  When all the functions die, we finalize_tree_manager.

    TODO: in the future, we would like to do the following once storage weak refs land
    -  We look for all the live storages and add references to THOSE
    -  We count as storages die
    -  All the storages are dead, we deallocate the tree manager
    """

    # Constructor initializing the tree manager for a specific device index
    def __init__(self, device_index):
        # This class keeps a strong reference to tree_manager,
        # but upon all other strong references to the tree_manager will reset it to None.
        # We need a strong reference so that we can still access its attributes upon cleanup.
        self.tree_manager: Optional[CUDAGraphTreeManager] = None

        # Number of outstanding references to the current tree manager
        self.live_cudagraphify_fns = 0

        # Index of the device managed by this instance
        self.device_index = device_index

        # Following two objects are only set in the case that Tensor outputs outlive
        # the cudagraphify_fns. Reference to the Graph is needed to keep the private pool from
        # deallocation.
        self.live_storages_count = 0
        self.graph: Optional[torch.cuda.CUDAGraph] = None

        # Lock to synchronize access to shared resources
        self.lock = threading.Lock()

    # Decrements the count of live storages and potentially resets associated references
    def _finalize_tensor(self):
        with self.lock:
            self.live_storages_count -= 1
            if self.live_storages_count == 0:
                # Reset the graph reference to None when all storages are no longer live
                self.graph = None

                # If no more active cudagraphify functions exist, reset the tree_manager to None
                if self.live_cudagraphify_fns == 0:
                    self.tree_manager = None

    # Decrements the count of live cudagraphify functions and possibly finalizes the tree manager
    def finalize_cudagraphify_fn(self):
        with self.lock:
            self.live_cudagraphify_fns -= 1
            if self.live_cudagraphify_fns == 0:
                # Trigger finalization of the tree manager when no more cudagraphify functions exist
                self._finalize_tree_manager()
    def _finalize_tree_manager(self):
        # 确保锁已经被获取
        assert self.lock.locked()
        # 将树管理器设为 None
        self.tree_manager = None

        # TODO - when issue #91395 is landed, we can set a weakref on
        # storages and trigger a deallocation when all outputs of the
        # cudagraph are dead.

        # live_storages = list(
        #     tree_manager.live_cudagraph_pool_storages_in_curr_execution()
        # )

        # # Maintain reference to graph to keep tensors alive
        # assert len(tree_manager.roots) > 0, "expected at least one use"
        # root = next(tree_manager.get_roots())
        # self.graph = root.graph
        # seen_storages = set()
        # for stor in live_storages:
        #     if stor in seen_storages:
        #         continue
        #     seen_storages.add(stor)
        #     self.live_storages_count += 1
        # .   weakref.finalize(stor, self._finalize_tensor)

    def add_strong_reference(self, fn: Callable[..., Any]):
        # 在获取锁的情况下，增加一个强引用函数
        with self.lock:
            self.live_cudagraphify_fns += 1

        # 为函数 fn 设置一个弱引用的终结器，调用 self.finalize_cudagraphify_fn

    def get_tree_manager(self) -> CUDAGraphTreeManager:
        # 在获取锁的情况下，如果树管理器为 None，则初始化一个新的 CUDAGraphTreeManager
        with self.lock:
            if self.tree_manager is None:
                self.tree_manager = CUDAGraphTreeManager(self.device_index)
            return self.tree_manager
# 创建一个线程局部变量，每个线程都有自己的副本
local = threading.local()

# 为每个设备创建一个树管理器容器的空字典
local.tree_manager_containers = {}
# 为每个设备创建一个默认使用的锁的字典，默认为 threading.Lock
local.tree_manager_locks = defaultdict(threading.Lock)

# 仅在用户调用 mark_step_begin 函数时递增，用于追踪步骤计数
class MarkStepBox:
    mark_step_counter = 0

# 注册这些对象，在 autograd 创建新线程时作为 TLS（线程局部存储）的一部分复制过去
torch._C._stash_obj_in_tls("tree_manager_containers", local.tree_manager_containers)
torch._C._stash_obj_in_tls("tree_manager_locks", local.tree_manager_locks)

def mark_step_begin():
    "标记新的推断或训练迭代即将开始。"

    # 递减 mark_step_counter，用于区分 GenerationTracking 计数器
    MarkStepBox.mark_step_counter -= 1

def reset_cudagraph_trees():
    "清除所有 cudagraph 树。"
    # 详见下文的关闭操作原因
    container_dict = get_obj(local, "tree_manager_containers")
    locks_dict = get_obj(local, "tree_manager_locks")

    # 遍历锁字典中的每个设备及其对应的锁
    for device, lock in locks_dict.items():
        with lock:
            # 获取当前设备的容器对象
            container = container_dict.get(device)
            # 如果容器不存在或者容器中的树管理器不存在，则继续下一个设备
            if not container or not container.tree_manager:
                continue

            # 关闭当前容器中的树管理器
            container.tree_manager.shutdown()

    # 禁用缓存的张量
    _set_cached_tensors_enabled(False)
    # 清空容器字典
    container_dict.clear()

    # 重置 mark_step_counter 为 0
    MarkStepBox.mark_step_counter = 0

def get_obj(local, attr_name):
    # 如果 local 对象中有指定的属性名，则返回该属性值
    if hasattr(local, attr_name):
        return getattr(local, attr_name)
    else:
        # 否则，确保 torch._C._is_key_in_tls 表明该属性名存在于 TLS 中，并返回对应的对象
        assert torch._C._is_key_in_tls(attr_name)
        return torch._C._get_obj_in_tls(attr_name)

def get_container(device_index: int):
    # 获取线程局部存储中的树管理器容器字典对象
    container_dict = get_obj(local, "tree_manager_containers")
    # 获取指定设备索引的锁对象
    lock = get_obj(local, "tree_manager_locks")[device_index]

    with lock:
        # 如果容器字典中不存在指定设备索引的容器，则创建一个新的树管理器容器
        if device_index not in container_dict:
            container_dict[device_index] = TreeManagerContainer(device_index)

        # 返回指定设备索引的树管理器容器对象
        return container_dict[device_index]

def get_manager(
    device_index: int, create_if_none_exists=True
) -> Optional[CUDAGraphTreeManager]:
    # 如果 create_if_none_exists 为 True，则返回指定设备索引的树管理器对象
    if create_if_none_exists:
        return get_container(device_index).get_tree_manager()
    # 否则返回 None
    return get_container(device_index).tree_manager

def cudagraphify_impl(model, inputs, static_input_idxs, *args, **kwargs):
    # 函数缓存字典，用于存储输入参数的组合对应的函数调用结果
    fn_cache: Dict[Tuple[int, ...], Callable[..., Any]] = {}

    # 检测 inputs 中的整数输入，用于后续索引操作
    int_key = [i for i, v in enumerate(inputs) if isinstance(v, int)]
    # 根据 int_key 索引列表创建获取整数输入的函数
    get_ints: Any = operator.itemgetter(*int_key) if int_key else lambda _: None

    # 删除 inputs 变量，释放内存空间
    del inputs
    # 定义一个函数 deferred_cudagraphify，接收一个输入参数 inputs
    def deferred_cudagraphify(inputs):
        # 获取 inputs 的整数键值
        int_key = get_ints(inputs)
        # 从缓存中获取与 int_key 对应的函数 fn
        fn = fn_cache.get(int_key)
        # 如果 fn 不为空，则直接调用 fn 处理 inputs 并返回结果
        if fn is not None:
            return fn(inputs)

        # 如果 int_key 为空，则记录日志表明处理没有符号整数的 cudagraph 树
        if int_key is None:
            log.info("recording cudagraph tree for graph without symints")
        else:
            # 否则记录日志表明处理具有特定符号整数键 int_key 的 cudagraph 树
            log.info("recording cudagraph tree for symint key %s", int_key)

        # 获取需要检查以对齐的输入索引，并更新静态输入索引
        check_input_idxs = get_input_idxs_to_check(inputs, static_input_idxs)
        new_static_input_idxs = remove_unaligned_input_idxs(inputs, static_input_idxs)
        # 复制未对齐的输入数据
        copy_misaligned_inputs(inputs, check_input_idxs)

        # 使用 cudagraphify 函数处理模型和输入数据，得到函数 fn 和处理结果 out
        fn, out = cudagraphify(model, inputs, new_static_input_idxs, *args, **kwargs)
        # 根据检查索引对齐输入数据
        fn = align_inputs_from_check_idxs(fn, inputs_to_check=check_input_idxs)
        # 将处理结果函数 fn 缓存起来，以便下次使用
        fn_cache[int_key] = fn

        # 返回处理结果 out
        return out

    # 返回函数 deferred_cudagraphify 作为最终的结果
    return deferred_cudagraphify
# 定义一个名为 cudagraphify 的函数，用于将模型和输入转换为计算图对象，并添加到指定设备的容器中
def cudagraphify(
    model,
    inputs,
    static_input_idxs=(),
    *,
    device_index: int,
    is_backward: bool,
    is_inference: bool,
    stack_traces: Optional[StackTraces] = None,
    constants: Tuple[torch.Tensor, ...] = (),
    placeholders: Tuple[torch.fx.Node, ...] = (),
    mutated_input_idxs: Tuple[int, ...] = (),
):
    # 获取指定设备的容器管理器
    manager = get_container(device_index).get_tree_manager()
    # 断言：不允许同时进行反向传播和推断
    assert not (is_backward and is_inference)
    # 根据 is_backward 和 is_inference 的值确定编译模式
    mode = (
        CompilationMode.BACKWARD
        if is_backward
        else (CompilationMode.INFERENCE if is_inference else CompilationMode.FORWARD)
    )

    # 将模型及相关参数添加到容器管理器中，并返回结果
    return manager.add_function(
        model,
        inputs,
        static_input_idxs,
        stack_traces,
        mode,
        constants,
        placeholders,
        mutated_input_idxs,
    )


class StorageWeakRefWrapper:
    """
    封装了一个存储的弱引用。在引用过期时会释放它。
    """

    __slots__ = ["ref", "_data_ptr", "extra_ref_check"]

    storage_ref: Optional[StorageWeakRef]

    def __init__(
        self,
        inp: Union[Tensor, UntypedStorage],
        extra_ref_check: Optional[Callable[[], None]] = None,
    ):
        """
        初始化函数。extra_ref_check 是一个额外的检查函数，用于检测弱引用是否过期。
        在检查存储使用计数时，我们假定 extra_ref_check 会持有一个额外的引用来防止存储过期。
        """
        # 根据输入类型初始化存储引用
        if isinstance(inp, Tensor):
            stor = inp.untyped_storage()
        else:
            assert isinstance(inp, UntypedStorage)
            stor = inp
        self.ref = StorageWeakRef(stor)
        self._data_ptr = stor.data_ptr()
        self.extra_ref_check = extra_ref_check

    @classmethod
    def from_weakref_and_data_ptr(cls, cdata, data_ptr, extra_ref_check=None):
        """
        从弱引用和数据指针创建实例的类方法。
        """
        instance = cls.__new__(cls)
        instance._data_ptr = data_ptr
        instance.ref = StorageWeakRef.from_weakref(cdata)
        instance.extra_ref_check = extra_ref_check
        return instance

    def __call__(self) -> Optional[StorageWeakRefPointer]:
        """
        当对象被调用时返回存储的弱引用指针，如果引用过期则返回 None。
        """
        if self.expired():
            return None

        return self.ref.cdata

    def swap_weakref(self, cdata):
        """
        用新的弱引用替换当前存储的弱引用。
        """
        self.ref.__del__()
        self.ref.cdata = cdata

    def data_ptr(self) -> int:
        """
        返回存储的数据指针，即使存储已过期也能返回。
        """
        return self._data_ptr

    def remove_extra_reference(self):
        """
        移除额外的引用检查函数。
        """
        self.extra_ref_check = None

    def expired(self):
        """
        检查存储是否已过期。
        """
        if self.extra_ref_check is not None and not self.extra_ref_check():
            return False

        # 如果 extra_ref_check 不为 None，我们期望有额外的引用存在
        stor_count = torch._C._storage_Use_Count(self.ref.cdata)
        return (stor_count - (self.extra_ref_check is not None)) == 0
    # 定义 __repr__ 方法，用于生成对象的字符串表示形式
    def __repr__(self):
        # 如果 self.ref 为 None 或者其引用已过期
        if self.ref is None or self.ref.expired():
            # 返回对象的死亡状态字符串表示形式
            return f"StorageWeakRefWrapper to {self.data_ptr()}; dead"
        else:
            # 返回对象的存活状态字符串表示形式，包含数据指针的信息
            return f"StorageWeakRefWrapper to {self.data_ptr()}; alive"
# 检查给定的弱引用是否指向一个存活的对象
def is_live(weak_ref: Optional[StorageWeakRefWrapper]) -> bool:
    # 调用 maybe_deref 函数检查弱引用是否指向一个非空对象
    return maybe_deref(weak_ref) is not None


# 尝试解引用给定的弱引用，并返回其数据指针及其引用计数
def maybe_deref(
    weak_ref: Optional[StorageWeakRefWrapper],
) -> Optional[Tuple[StorageWeakRefPointer, int]]:
    # 如果弱引用为空，直接返回 None
    if weak_ref is None:
        return None
    # 尝试解引用弱引用获取其指向的对象
    r = weak_ref()
    # 如果解引用后 r 为空，说明对象已经被销毁，返回 None
    if r is None:
        return None
    # 返回解引用后的对象 r 和弱引用的数据指针
    # 注意：r.data_ptr() 和 weak_ref.data_ptr() 不一定相等
    return r, weak_ref.data_ptr()


@contextlib.contextmanager
def _use_cuda_memory_pool_manager(device, mem_pool, stream):
    """
    Context manager to use cuda graph pool for new allocations. If you use this manager
    all cudagraph tensors in use should be reflected in the allocator or they will be overwritten.
    existing_graph should already have been used in a capture, and the mem_pool must already exist,
    because this manager will not preserve a reference to the pool which keeps it alive.
    """
    # 同步 CUDA 设备
    torch.cuda.synchronize()
    # 等待流的完成
    stream.wait_stream(torch.cuda.current_stream())

    # 在指定设备上的指定流中开始分配内存池的内存
    with torch.cuda.stream(stream), torch.device(device):
        torch._C._cuda_beginAllocateCurrentStreamToPool(device, mem_pool)
        try:
            yield
        finally:
            # 结束对内存池的分配
            torch._C._cuda_endAllocateCurrentStreamToPool(device, mem_pool)
            # 释放内存池
            torch._C._cuda_releasePool(device, mem_pool)

    # 等待流的完成
    torch.cuda.current_stream().wait_stream(stream)


# 将给定的 Tensor 转换为其存储的弱引用包装器
def map_to_ref(t: Optional[Tensor]) -> Optional[StorageWeakRefWrapper]:
    # 如果 t 不是 torch.Tensor 类型，断言 t 应该是 None
    if not isinstance(t, torch.Tensor):
        assert t is None
        return None
    # 返回包装了给定 Tensor 的弱引用包装器
    return StorageWeakRefWrapper(t)


# 表示路径中 (depth, offset) 到图中输出的索引
PathOutputIndex = Tuple[int, int]

# 表示路径中每个节点的每个输出是否存活的列表
PathLiveness = List[List[bool]]

# 堆栈跟踪的列表
StackTraces = List[Optional[str]]


class CUDAWarmupNode:
    """
    简化的 CUDA 模型包装器，将输出包装在存储引用中，并公开获取当前预热链中活动存储的 API。

    CUDAWarmupNode 可能有 CUDAGraphNode 或 CUDAWarmupNode 作为父节点，但只能有
    CUDAWarmupNode 作为子节点，因为我们不能记录或执行没有稳定内存地址的张量。

    CUDAWarmupNode 和 CUDAGraphNode 有很多区别，使得使用不同的类更容易。
    - CUDAGraphNode 逻辑和初始化大部分基于首次记录的张量属性。在第一次预热中，这些属性尚未最终确定。
    - 所有 RecordedFunction 的输入必须复制到 cuda 图的内存池中，这在预热中是不必要的。
    - CUDAWarmup 仅使用一次，因此不需要像 CUDAGraphNode 那样优化记账。这样更简单。

    注意：此类和 CUDAGraphNode 需要公开 path_live_weakrefs、all_outputs_are_dead 和
    self.outputs_weakrefs、stack_traces 以及 tensor_weakrefs 以保持兼容性。
    """
    # 初始化方法，用于设置对象的各个属性
    def __init__(
        self,
        wrapped_function: WrappedFunction,  # 接收一个被包装的函数对象
        parent,  # 接收一个父对象的引用
        cuda_graphs_pool: Tuple[int, int],  # 接收一个包含两个整数的元组，表示CUDA图池的大小
        existing_cuda_graph: Optional[torch.cuda.CUDAGraph],  # 可选的CUDA图对象，表示已存在的CUDA图
        device_index: int,  # 整数，表示设备索引
        stack_traces: Optional[StackTraces],  # 可选的堆栈跟踪对象
        stream: torch.cuda.Stream,  # CUDA流对象
        already_warm: bool,  # 布尔值，表示对象是否已预热
        id: GraphID,  # 图形ID对象
    ):
        self.wrapped_function = wrapped_function  # 将传入的wrapped_function参数赋值给对象的wrapped_function属性
        self.parent = parent  # 将传入的parent参数赋值给对象的parent属性
        self.cuda_graphs_pool = cuda_graphs_pool  # 将传入的cuda_graphs_pool参数赋值给对象的cuda_graphs_pool属性
        self.outputs_weakrefs: List[Optional[StorageWeakRefWrapper]] = []  # 初始化一个空列表，用于存储输出的弱引用
        self.tensor_weakrefs: List[Optional[TensorWeakRef]] = []  # 初始化一个空列表，用于存储张量的弱引用
        self.existing_cuda_graph = existing_cuda_graph  # 将传入的existing_cuda_graph参数赋值给对象的existing_cuda_graph属性
        self.has_run = False  # 将对象的has_run属性初始化为False，表示对象尚未运行
        self.device_index = device_index  # 将传入的device_index参数赋值给对象的device_index属性
        self.stack_traces = stack_traces  # 将传入的stack_traces参数赋值给对象的stack_traces属性
        self.stream = stream  # 将传入的stream参数赋值给对象的stream属性
        self.already_warm = already_warm  # 将传入的already_warm参数赋值给对象的already_warm属性
        self.id = id  # 将传入的id参数赋值给对象的id属性
    def run(self, new_inputs):
        assert not self.has_run, "Wrapped function should never be run twice"

        # See: output_is_alias_of_persistent_static_inputs below. We should only be returning freshly created
        # storages in path_live_weakrefs.
        # 获取当前路径中存活的弱引用的数据指针集合
        existing_path_data_ptrs = {
            t.data_ptr() for t in self.path_live_weakrefs() if t()
        }

        def get_non_cudagraph_inps():
            # 获取非 cudagraph 输入的存储列表
            non_cudagraph_inps = []
            for t in itertools.chain(new_inputs, self.wrapped_function.constants):
                if (
                    isinstance(t, torch.Tensor)
                    and t.untyped_storage().data_ptr() not in existing_path_data_ptrs
                ):
                    non_cudagraph_inps.append(weakref.ref(t.untyped_storage()))
            return non_cudagraph_inps

        # 获取非 cudagraph 输入的存储列表
        non_cudagraph_inps_storages = get_non_cudagraph_inps()

        # 如果配置允许，且尚未预热，检查内存池情况
        if config.triton.slow_path_cudagraph_asserts and not self.already_warm:
            refs = list(self.path_live_weakrefs())
            check_memory_pool(self.device_index, self.cuda_graphs_pool, refs)

        # 使用指定的设备索引，禁用卷积缓存清空，清除 cublas 管理器，使用 CUDA 内存池管理器，并记录历史
        with torch.cuda.device(
            self.device_index
        ), disable_conv_cache_emptying(), clear_cublas_manager(), _use_cuda_memory_pool_manager(
            self.device_index, self.cuda_graphs_pool, self.stream
        ), get_history_recording():
            # 运行包装函数的模型部分，得到输出
            out = self.wrapped_function.model(new_inputs)

        # 我们需要知道哪些输出分配在 cudagraph 池中，以便在下一个 cudagraph 步骤开始时释放它们，并设置它们的访问错误
        # 使用弱引用来引用输入存储，以防一个先前分配给通用缓存分配器池的块被重新分配给私有池
        non_cudagraph_inps_storage_ptrs = set()
        for storage in non_cudagraph_inps_storages:
            s = storage()
            if s is not None:
                non_cudagraph_inps_storage_ptrs.add(s._cdata)

        # 断言新输入的长度为零
        assert len(new_inputs) == 0

        # sdpa 在不记录 CUDA 图时返回 CPU 张量
        def add_ref(o):
            return (
                isinstance(o, torch.Tensor)
                and o.is_cuda
                and o.untyped_storage()._cdata not in non_cudagraph_inps_storage_ptrs
                and o.untyped_storage().data_ptr() != 0
            )

        # 扩展输出的弱引用列表
        self.outputs_weakrefs.extend(
            [map_to_ref(o) if add_ref(o) else None for o in out]
        )
        # 扩展张量的弱引用列表
        self.tensor_weakrefs.extend(
            [TensorWeakRef(o) if add_ref(o) else None for o in out]
        )

        # 如果配置允许，且尚未预热，检查输出的路径存活弱引用列表中的内存池情况
        if config.triton.slow_path_cudagraph_asserts and not self.already_warm:
            out_refs = list(self.path_live_weakrefs())
            check_memory_pool(self.device_index, self.cuda_graphs_pool, out_refs)

        # 返回输出
        return out
    # 返回从当前节点到根节点的路径上的所有节点（包括自身），以生成器方式逆序输出
    def _path_from_root(self):
        nodes = []
        node = self
        while node:
            nodes.append(node)
            node = node.parent

        yield from reversed(nodes)

    # 返回在路径上所有节点创建的存储的弱引用的迭代器
    def path_live_weakrefs(self) -> Iterator[StorageWeakRefWrapper]:
        "Returns all live storages weakrefs that created by nodes in this path"
        # 遍历从根节点到当前节点的路径上的所有节点
        for node in self._path_from_root:
            # 遍历当前节点的输出的弱引用列表
            for output in node.outputs_weakrefs:
                # 检查弱引用是否有效（存储是否有效）
                if is_live(output):
                    yield output

    # 检查路径上所有输出是否都已失效
    def all_outputs_are_dead(self):
        # 判断路径上的存储弱引用列表是否为空（即是否所有输出都已失效）
        return not list(self.path_live_weakrefs())

    # 检查给定张量是否关联到路径上的某个存储弱引用
    def _is_cuda_graph_recorded_tensor(self, t: torch.Tensor):
        # 遍历路径上所有存储弱引用
        for storage_weak_ref in self.path_live_weakrefs():
            # 检查张量的存储的数据指针是否与存储弱引用的数据指针相同
            if t.untyped_storage().data_ptr() == storage_weak_ref.data_ptr():
                return True
        # 若未找到匹配的存储弱引用，则返回 False
        return False
# 定义类型别名，用于表示不同列表的含义
InputList = List  # 输入索引列表
OutputList = List  # 输出索引列表
LevelList = List  # 层级列表（树中节点距离根节点的距离）

# 输出别名信息的基类
class OutputAliasInfo:
    pass

# 表示图输出构造新别名或为 None 的单例类
class _UnaliasedStorage(OutputAliasInfo):
    "Singleton to mark that the graph output constructs a new alias or is None"
    pass

# 表示图输出为新别名与先前图输出相关的类
class AliasesPriorGraphOutput(OutputAliasInfo):
    "Marks that the graph output aliases an output of a prior graph"
    __slots__ = ["index"]

    index: PathOutputIndex

    def __init__(self, index: PathOutputIndex):
        assert isinstance(index, tuple)
        self.index = index

# 表示图输出为新别名与新返回的输出索引相关的类
class AliasesNewOutput(OutputAliasInfo):
    "Marks that the graph output aliases an index in the new, returned outputs"

    __slots__ = ["index"]

    index: int

    def __init__(self, index):
        assert isinstance(index, int)
        self.index = index

# CUDA 图节点类，用于记录函数到 CUDA 图中
class CUDAGraphNode:
    """
    A single recording of a function into a CUDA Graph. Recordings of CUDA Graphs share a single memory pool
    and are structured into a tree, where there is a single recording that can precede it (parent) and multiple
    subsequent recordings that may follow (children). A node will have no parent if it is the first recording
    in a tree; i.e., when it is first recorded, there are no live tensors from a previous recording which
    would force a dependency.

    On first recording, all of the live tensors in the current CUDA Graph Node path will be
    reflected in the corresponding private pool. On subsequent executions, the caching allocator
    is unaffected when the graph is replayed.

    In order to support recording a subsequent cuda graph recording after execution of this graph,
    we checkpoint the state of the memory pool so that it may later be resumed.

    WrappedFunction should have already been warmed up prior to invocation.

    See [setCheckpointPoolState] for further explanation, as well as
    https://user-images.githubusercontent.com/13564/222815509-374f3400-f83d-4f7d-8fa6-4a092b3250bb.png
    """

    def __init__(
        self,
        wrapped_function: WrappedFunction,
        id: GraphID,
        parent: Optional[CUDAGraphNode],
        inputs: List[Tensor],
        cuda_graphs_pool: Tuple[int, int],
        device_index: int,
        stack_traces: Optional[StackTraces],
        stream: torch.cuda.Stream,
    # 将输入数据复制到目标张量中，并从源张量中移除非静态输入数据
    def _copy_inputs_and_remove_from_src(self, dsts, srcs):
        # 存储目标张量和源张量
        dst_tensors = []
        src_tensors = []
        # 遍历非静态输入索引列表
        for idx in self.non_static_input_idx:
            # 如果源张量不是 torch.Tensor 类型，则跳过
            if not isinstance(srcs[idx], torch.Tensor):
                continue
            # 获取当前索引对应的扩展维度
            expanded_dims = self.expanded_dims[idx]
            # 将目标张量和源张量中的数据根据扩展维度进行索引并存储
            dst_tensors.append(index_expanded_dims(dsts[idx], expanded_dims))
            src_tensors.append(index_expanded_dims(srcs[idx], expanded_dims))
            # 将源张量置为 None，表示已经从源张量中移除
            srcs[idx] = None
        # 如果目标张量列表非空，则执行张量的复制操作
        if dst_tensors:
            torch._foreach_copy_(dst_tensors, src_tensors)

    # 检查静态输入是否稳定不变
    def check_static_inputs_are_stable(self, new_inputs):
        # 如果不需要重新记录静态输入的变化，并且静态输入的数据指针不相等
        if (
            not self.rerecord_if_static_inputs_change
            and not torch._C._tensors_data_ptrs_at_indices_equal(
                new_inputs,
                self.static_input_data_ptrs,
                self.non_managed_static_input_idxs,
            )
        ):
            # 报错信息显示静态输入数据指针已经发生变化
            static_tensors = [new_inputs[i] for i in self.non_managed_static_input_idxs]
            data_ptrs = [
                self.static_input_data_ptrs[i]
                for i in self.non_managed_static_input_idxs
            ]
            error_msg = "static input data pointer changed.\n"
            # 遍历静态张量和其数据指针，生成详细错误信息
            for i, (t, data_ptr) in enumerate(zip(static_tensors, data_ptrs)):
                index = self.non_managed_static_input_idxs[i]
                if t.data_ptr() != data_ptr:
                    placeholder = self.wrapped_function.placeholders[index]
                    error_msg = (
                        f"{error_msg}input name: {placeholder.name}. "
                        f"data pointer changed from {data_ptr} to {t.data_ptr()}. "
                        f"input stack trace: {get_placeholder_stack_trace(placeholder)}\n"
                    )
            # 触发错误检查，输出错误信息
            torch._check(False, lambda: error_msg)

    # 在运行首次输入时执行的操作
    def run_first_inputs(self, new_inputs):
        # 如果配置为快速路径 cudagraph 断言，则在调用前检查不变性条件
        if config.triton.fast_path_cudagraph_asserts:
            self.debug_check_invariants_before_invocation()

        # 确保新输入列表为空
        assert len(new_inputs) == 0
        # 获取录制的输出结果并清空记录的输出
        outputs = self.recording_outputs
        self.recording_outputs = None
        # 返回录制的输出结果
        return outputs
    # 定义一个方法 `run`，用于执行某个功能
    def run(self, new_inputs):
        # 调用对象的方法，检查静态输入是否稳定
        self.check_static_inputs_are_stable(new_inputs)

        # 复制新输入并从源输入中移除
        self._copy_inputs_and_remove_from_src(self.reconstructed_inputs, new_inputs)
        # 清空新输入列表
        new_inputs.clear()

        # 执行计算图
        self.run_graph()

        # 重建输出
        outputs = self.reconstruct_outputs()

        # 如果配置要求进行 Triton 的快速路径 CUDA 图的断言检查
        if config.triton.fast_path_cudagraph_asserts:
            # 调用对象的方法，检查调用后的不变性条件
            self.debug_check_invariants_after_invocation()

        # 如果配置要求强制 CUDA 图同步
        if config.triton.force_cudagraph_sync:
            # 在 CUDA 上进行同步操作
            torch.cuda.synchronize()

        # 将静态输入稳定性标志重置为假，以便将来进行检查
        # Reset this to run the check in the future
        self.static_inputs_stable = False

        # 返回计算结果的输出
        return outputs
    def reconstruct_outputs(self):
        "Reconstruct output tensors according to their saved metadata and alias information"

        # Cached tensors will not yet be set on the first execution
        # They are also cleared in checkpointing, so if we checkpoint this node
        # and then execute it again we will need to repopulate cached tensors
        # 如果缓存的张量在第一次执行时尚未设置
        # 它们也会在检查点中清除，因此如果我们检查点此节点
        # 然后再次执行它，我们将需要重新填充缓存的张量
        if not self.cached_tensor_outputs:
            self._initialize_cached_tensors()

        outputs: List[Optional[Union[int, torch.Tensor]]] = []

        for i, (storage_info, metadata) in enumerate(
            zip(self.output_storage_alias, self.outputs_metadata)
        ):
            if not isinstance(metadata, dict):  # tensor metadata
                assert isinstance(metadata, (int, type(None)))
                outputs.append(metadata)
                continue

            cached_t = self.cached_tensor_outputs[i]
            if cached_t is not None:
                # this output represents a fresh allocated tensor.
                # We return the same TensorImpl from run to run to avoid overhead.
                # autograd.Function will reset the Autograd meta of output tensors
                # as part of aot_autograd, but _backward_hooks are stored on tensors separately,
                # so we need to manually reset hooks.
                # 这个输出代表一个新分配的张量。
                # 我们在运行中保持相同的 TensorImpl，以避免额外开销。
                # autograd.Function 将重置输出张量的 Autograd 元信息
                # 作为 aot_autograd 的一部分，但是 _backward_hooks 单独存储在张量上，
                # 因此我们需要手动重置 hooks。
                if cached_t._backward_hooks is not None:
                    cached_t._backward_hooks = None

                # No need to update weakrefs, already correctly initialized
                # 无需更新弱引用，已正确初始化
                outputs.append(cached_t)
                continue

            static_t = self.static_output_tensors[i]
            if static_t is not None:
                assert self.outputs_weakrefs[i] is None
                outputs.append(static_t)
                continue

            storage = self.prepare_alias_info_for_tensor_construction(
                storage_info, metadata
            )

            if isinstance(storage, UntypedStorage) or storage is None:
                out = self._reconstruct_from_tensor_metadata(metadata, storage)
            else:
                assert isinstance(storage, int)
                out = self._reconstruct_from_tensor_metadata(
                    metadata, cast(torch.Tensor, outputs[storage]).untyped_storage()
                )

            outputs.append(out)
            w = self.outputs_weakrefs[i]
            assert w is not None
            w.swap_weakref(out.untyped_storage()._weak_ref())

        return outputs

    def prepare_alias_info_for_tensor_construction(
        self,
        out_alias_info: Optional[OutputAliasInfo],
        metadata: Union[Dict[str, Any], int, None],
        ):
        # Prepare storage information and metadata for constructing a tensor
        # 为构造张量准备存储信息和元数据
    ) -> Union[UntypedStorage, None, int]:
        # 如果 metadata 是整数或者 None，或者 out_alias_info 是 UnaliasedStorage 类型，则返回 None
        if (
            isinstance(metadata, (int, type(None)))
            or out_alias_info is UnaliasedStorage
        ):
            return None

        # 如果 out_alias_info 是 AliasesPriorGraphOutput 类型，则获取其引用并返回新的 UntypedStorage 对象
        if isinstance(out_alias_info, AliasesPriorGraphOutput):
            depth, existing_output_index = out_alias_info.index
            ref = self.path_weakrefs[depth][existing_output_index]
            assert ref is not None
            return torch.UntypedStorage._new_with_weak_ptr(ref())

        # 否则，确保 out_alias_info 是 AliasesNewOutput 类型，然后返回其索引
        assert isinstance(out_alias_info, AliasesNewOutput)
        return out_alias_info.index

    def prepare_storages_for_construction(
        self,
    ) -> List[Union[UntypedStorage, None, int]]:
        # 准备用于构建的输出存储列表
        output_storages = []
        # 遍历每个输出存储别名和对应的元数据
        for output_storage_alias, metadata in zip(
            self.output_storage_alias, self.outputs_metadata
        ):
            # 调用方法准备 tensor 构建的别名信息，并添加到输出存储列表中
            output_storages.append(
                self.prepare_alias_info_for_tensor_construction(
                    output_storage_alias, metadata
                )
            )

        # 返回准备好的输出存储列表
        return output_storages

    def run_graph(self):
        # 断言图形对象不为 None，然后回放图形
        assert self.graph is not None
        self.graph.replay()

    def all_outputs_are_dead(self):
        # 检查路径从此节点到其根部的所有输出是否都已经失效（即已经释放）
        for depth, output_index in self.live_indices_after_graph:
            if is_live(self.path_weakrefs[depth][output_index]):
                return False
        return True
    # 记录模型的方法，将模型输出和相关输入信息记录下来
    def _record(self, model, inputs):
        "Record the model"

        # 定义一个生成器函数，用于迭代静态输入的张量
        def static_input_iter():
            for i in self.wrapped_function.static_input_idxs:
                # 检查输入是否为Tensor且未被记录到CUDA图中
                if isinstance(inputs[i], torch.Tensor) and not self._is_cuda_graph_recorded_tensor(inputs[i]):
                    yield inputs[i]

        # 创建一个字典，将静态输入的存储地址映射到其弱引用对象
        static_input_persistent_storage_ptrs: Dict[int, StorageWeakRefWrapper] = {
            inp.untyped_storage().data_ptr(): StorageWeakRefWrapper(inp)
            for inp in itertools.chain(
                static_input_iter(), self.wrapped_function.constants
            )
        }

        # 如果配置要求进行Trition的慢路径CUDA图断言
        if config.triton.slow_path_cudagraph_asserts:
            # 使用父节点的弱引用列表来检查内存池中的存活对象
            memory = (
                [] if self.parent is None else list(self.parent.path_live_weakrefs())
            )
            # 添加当前函数未记录的Tensor对象到内存列表中
            memory += [
                StorageWeakRefWrapper(elem)
                for i, elem in enumerate(inputs)
                if isinstance(elem, torch.Tensor)
                and i not in self.wrapped_function.static_input_idxs
                and elem.untyped_storage().data_ptr() != 0
            ]
            # 检查内存池中的对象状态
            check_memory_pool(self.device, self.cuda_graphs_pool, memory)

        # 使用保留随机数种子状态、指定CUDA设备、清除cuBLAS管理器、指定CUDA图等上下文，执行模型
        with preserve_rng_state(), torch.cuda.device(
            self.device
        ), clear_cublas_manager(), torch.cuda.graph(
            self.graph,
            stream=self.stream,
            pool=self.cuda_graphs_pool,
            capture_error_mode="thread_local",
        ), get_history_recording():
            # 执行模型计算，获取静态输出
            static_outputs = model(inputs)

        # 断言输入列表已经为空，即模型执行后应无未处理的输入
        assert len(inputs) == 0

        # 如果静态输出不是列表或元组，则转换为元组形式
        if not isinstance(static_outputs, (list, tuple)):
            static_outputs = (static_outputs,)

        # 将静态输出和静态输入的持久存储指针添加到记录中
        self._add_first_outputs(static_outputs, static_input_persistent_storage_ptrs)

        # 返回静态输出
        return static_outputs

    # 将首次输出添加到记录中，同时记录静态输入的持久存储指针
    def _add_first_outputs(
        self,
        outputs,
        static_input_persistent_storage_ptrs: Dict[int, StorageWeakRefWrapper],
    ):
        ...

    # 标记先前图形输出为别名，从祖先节点的缓存张量中移除该图形输出的别名
    def _mark_prior_graph_output_as_aliased(self, index: PathOutputIndex):
        "Remove a graph output from the unaliased, cached tensors in an ancestor node"
        depth, output_index = index
        node = list(self._path_from_root)[depth]
        # 将指定深度和输出索引的未别名标记设置为False
        node.unaliased_in_all_paths[output_index] = False
        x = self.path_weakrefs[depth][output_index]
        # 断言弱引用对象不为空
        assert x is not None
        # 移除额外的引用
        x.remove_extra_reference()
    # 初始化缓存张量
    def _initialize_cached_tensors(self):
        # 输出的弱引用列表和输出的元数据列表应该有相同的长度
        assert len(self.outputs_weakrefs) == len(self.outputs_metadata)

        # 遍历输出存储别名、输出元数据和在所有路径中未别名化的列表
        for i, (storage_info, metadata, make_cached) in enumerate(
            zip(
                self.output_storage_alias,
                self.outputs_metadata,
                self.unaliased_in_all_paths,
            )
        ):
            if not make_cached:
                # 如果不需要缓存，则将缓存张量输出列表中添加空值并继续下一轮循环
                self.cached_tensor_outputs.append(None)
                continue

            # 确保存储信息是未别名化的存储
            assert storage_info is UnaliasedStorage
            # 确保元数据是一个字典
            assert isinstance(metadata, dict)
            
            # 根据元数据创建存储
            s = self.create_storage(metadata)
            # 从张量元数据重新构建张量
            out = self._reconstruct_from_tensor_metadata(metadata, storage=s)

            # XXX: 告知自动微分系统张量将有额外的引用，用于梯度缓冲区的原地操作判断
            # 这样可以避免在追踪和后续执行之间进行原地操作时的不一致性
            # 对于我们测试的某些模型，这可以避免输入不再位于 cudagraph 池中，从而导致意外的重新记录
            # 它还告知 AMP 缓存，尽管张量实现不能在数据类型转换中缓存

            # 将缓存的张量添加到内部缓存系统中
            torch._C._add_cached_tensor(out)

            # 创建对当前对象的弱引用
            self_ref = weakref.ref(self)

            # 检查当前张量输出的引用计数
            def check_refcount(i):
                self_loc = self_ref()
                if self_loc is None:
                    return False
                return self_loc.get_output_refcount(i) == 2

            # 使用 functools.partial 创建局部函数 check，用于检查引用计数
            check = functools.partial(check_refcount, i=i)

            # 使用存储弱引用包装器来跟踪张量输出
            self.outputs_weakrefs[i] = StorageWeakRefWrapper(out, extra_ref_check=check)
            # 将缓存的张量输出添加到缓存张量列表中
            self.cached_tensor_outputs.append(out)

    # 获取指定索引处缓存张量的引用计数
    def get_output_refcount(self, index):
        return sys.getrefcount(self.cached_tensor_outputs[index])

    @property
    # 获取父节点的属性，解除对 _parent 的弱引用
    def parent(self):
        "解除对 _parent 的弱引用"
        return self._parent() if self._parent is not None else None

    @property
    # 返回从当前节点到根节点的所有节点路径
    def _path_to_root(self):
        "返回从当前节点到根节点的所有节点路径"
        node = self
        while node:
            yield node
            node = node.parent

    @property
    # 返回从根节点到当前节点的所有节点路径
    def _path_from_root(self):
        "返回从根节点到当前节点的所有节点路径"
        nodes = reversed(list(self._path_to_root))
        yield from nodes
    def _is_cuda_graph_recorded_tensor(self, t: torch.Tensor):
        "Is this tensor an output of a node in this path"
        # 遍历路径中的所有输出引用
        for output_refs in self.path_weakrefs:
            for storage_weak_ref in output_refs:
                if storage_weak_ref is None:
                    continue
                # 获取存储的数据指针
                data_ptr = storage_weak_ref.data_ptr()
                # 检查当前张量是否与存储的数据指针匹配
                if t.untyped_storage().data_ptr() == data_ptr:
                    return True
        # 如果未找到匹配的数据指针，则返回 False
        return False

    def _is_alias_of_live_recorded_tensor(
        self, t: torch.Tensor
    ) -> Optional[PathOutputIndex]:
        # 遍历路径中的所有输出引用，同时记录深度
        for depth, output_refs in enumerate(self.path_weakrefs):
            for output_index, storage_ref in enumerate(output_refs):
                # 可能解引用存储引用，获取存储和指针
                if (storage_and_ptr := maybe_deref(storage_ref)) is not None:
                    storage, ptr = storage_and_ptr
                    # 检查当前张量是否与存储的指针匹配
                    if ptr == t.untyped_storage().data_ptr():
                        return (depth, output_index)
        # 如果未找到匹配的数据指针，则返回 None
        return None

    @staticmethod
    def _check_liveness(
        indices: List[PathOutputIndex],
        output_refs: List[List[Optional[StorageWeakRefWrapper]]],
    ):
        "Check that all of the indices specified are dead references"
        # 检查指定的索引是否对应于死引用
        for depth, output_index in indices:
            w = output_refs[depth][output_index]
            assert w is not None
            # 如果引用不为 None，则表示引用还活跃，返回 False
            if w() is not None:
                return False
        # 如果所有索引对应的引用都为 None，则返回 True
        return True

    def add_child(self, function_id: FunctionID, node: CUDAGraphNode):
        "Adds node as a a child of self"
        # 将节点添加为当前对象的子节点
        self.children[function_id].append(node)

    @staticmethod
    def _get_different_indices(
        prev: List[List[bool]], curr: List[List[bool]]
    ) -> List[PathOutputIndex]:
        "Find indices where the two lists differ."
        # 找到两个列表中不同的索引位置
        dead_indices = []
        assert len(prev) <= len(curr)
        for i, (outputs1, outputs2) in enumerate(zip(prev, curr)):
            assert len(outputs1) == len(outputs2)
            for j, (output1, output2) in enumerate(zip(outputs1, outputs2)):
                if output1 != output2:
                    dead_indices.append((i, j))
        return dead_indices

    @staticmethod
    def _get_liveness(
        weakrefs: List[List[Optional[StorageWeakRefWrapper]]],
    ) -> List[List[bool]]:
        "Maps weakrefs to true if the reference is alive and false otherwise"
        # 将弱引用映射为引用是否存活的布尔值列表
        if len(weakrefs) == 0:
            return []
        # 对每个输出引用列表应用 is_live 函数，构建存活性布尔值列表
        return [pytree.tree_map(is_live, outputs) for outputs in weakrefs]

    def debug_assert_invariants(
        self, expected_liveness: List[List[bool]], newly_dead: List[PathOutputIndex]
    ):
        # Debug 方法：检查预期的存活性与新死亡索引是否一致
    ):
        # 如果配置不允许使用 Triton 的快速路径 cudagraph 断言，则直接返回
        if not config.triton.fast_path_cudagraph_asserts:
            return

        # 遍历路径上的每个节点，确保路径弱引用与节点输出弱引用相匹配
        for i, node in enumerate(self._path_from_root):
            assert self.path_weakrefs[i] is node.outputs_weakrefs

        # 复制路径上的节点列表
        nodes = list(self._path_from_root)

        # 获取 CUDA 图池中的活跃块地址
        live_blocks = get_block_addrs(self.cuda_graphs_pool)

        # 初始化存储数据指针和存储弱引用集合
        live_storage_data_ptrs = set()
        live_storage_weak_ptrs = set()

        # 遍历预期存活性的每一层和每个输出
        for depth, outputs_liveness in enumerate(expected_liveness):
            for output_idx, output_liveness in enumerate(outputs_liveness):
                # 获取当前输出的路径弱引用
                w = self.path_weakrefs[depth][output_idx]
                # 如果弱引用可能被解引用，则进行断言和更新
                if (stor_weak_ptr_and_data_ptr := maybe_deref(w)) is not None:
                    assert output_liveness
                    stor_weak_ptr, stor_data_ptr = stor_weak_ptr_and_data_ptr
                    # 确保数据指针在存储数据指针集合中的状态与存储弱引用指针在集合中的状态一致
                    assert (stor_data_ptr in live_storage_data_ptrs) == (
                        stor_weak_ptr in live_storage_weak_ptrs
                    )
                    live_storage_data_ptrs.add(stor_data_ptr)
                    live_storage_weak_ptrs.add(stor_weak_ptr)

                    # 检查是否为持久别名
                    is_persistent_alias = (
                        nodes[depth].static_output_tensors[output_idx] is not None
                    )

                    # 如果是持久别名，则确保数据指针不在活动块集合中
                    if is_persistent_alias:
                        assert stor_data_ptr not in live_blocks

        # 对于新死亡的路径层和输出索引组合，确保路径弱引用指向的输出不再存活
        for depth, output_index in newly_dead:
            assert not is_live(self.path_weakrefs[depth][output_index])

    # 调试：在调用之前检查不变量
    def debug_check_invariants_before_invocation(self):
        self.debug_assert_invariants(
            self.recorded_liveness_before_graph, self.expected_dead_indices_before_graph
        )

    # 调试：在调用之后检查不变量
    def debug_check_invariants_after_invocation(self):
        self.debug_assert_invariants(
            self.recorded_liveness_before_graph, self.expected_dead_indices_after_graph
        )

    # 返回自调用以来已经死亡的所有张量输出数据指针
    def data_ptrs_dead_since_invocation(self) -> List[int]:
        """
        Since this node was invoked, return data ptrs of all tensor outputs that have died
        in the current executing tree path.
        """
        # 获取当前执行路径的存活性信息
        curr_liveness = self._get_liveness(self.path_weakrefs)
        # 获取当前图路径中记录的存活性信息与当前存活性信息之间的差异索引
        _get_different_indices = self._get_different_indices(
            self.recorded_liveness_after_graph, curr_liveness
        )

        # 复制路径节点列表
        path = list(self._path_from_root)
        ptrs_to_deallocate = []
        # 根据不同的索引获取需要释放的数据指针
        for depth, output_index in _get_different_indices:
            ptrs_to_deallocate.append(
                path[depth].outputs_metadata[output_index]["data_ptr"]
            )

        return ptrs_to_deallocate

    # 返回当前图路径中存活的弱引用迭代器
    def path_live_weakrefs(self) -> Iterator[StorageWeakRefWrapper]:
        for i, j in self.live_indices_after_graph:
            # 获取路径弱引用，并检查是否存活
            out = self.path_weakrefs[i][j]
            if out is not None and is_live(out):
                yield out
    def remove_node_cached_tensors(self):
        # 遍历缓存的张量输出列表
        for t in self.cached_tensor_outputs:
            # 如果张量不为空，调用底层 Torch 函数移除缓存的张量
            if t is not None:
                torch._C._remove_cached_tensor(t)
        # 清空缓存的张量输出列表
        self.cached_tensor_outputs.clear()

        # 遍历未别名化的输入路径列表
        for i, unaliased in enumerate(self.unaliased_in_all_paths):
            # 如果当前节点在所有路径中未被别名化
            if unaliased:
                # 获取对应输出的弱引用节点
                n = self.outputs_weakrefs[i]
                # 断言节点不为空
                assert n is not None
                # 调用节点方法移除额外的引用
                n.remove_extra_reference()

    def remove_path_cached_tensors(self):
        # 遍历从根节点到当前节点的路径列表
        for node in self._path_from_root:
            # 调用节点方法移除节点缓存的张量
            node.remove_node_cached_tensors()

    def clear_path_state(self):
        "Clear the path state in this current executing node"
        # 这个方法当前并没有实际操作，作为占位符留在这里
        pass

    @staticmethod
    def _tensor_metadata(x, ignore_storage_offset=True):
        # 断言 x 是 torch.Tensor 类型的对象
        assert isinstance(x, torch.Tensor)
        # 返回张量的元数据字典
        # 对于输入，忽略存储偏移；对于输出，不忽略存储偏移
        return {
            "nbytes": x.untyped_storage().nbytes(),
            "data_ptr": x.untyped_storage().data_ptr(),
            "size": x.shape,
            "stride": x.stride(),
            "dtype": x.dtype,
            "device": x.device,
            "storage_offset": x.storage_offset() if not ignore_storage_offset else 0,
        }

    def _reconstruct_from_tensor_metadata(
        self, metadata: Dict[str, Any], storage=None
    ) -> Tensor:
        # 如果未提供存储对象，根据元数据创建存储
        s = self.create_storage(metadata) if storage is None else storage
        # 使用 Torch 函数根据存储和元数据重建张量
        return torch._C._construct_CUDA_Tensor_From_Storage_And_Metadata(metadata, s)

    def create_storage(self, metadata):
        # 使用 Torch 函数根据数据指针、设备和字节数创建存储对象
        return torch._C._construct_storage_from_data_pointer(
            metadata["data_ptr"], metadata["device"], metadata["nbytes"]
        )

    def _allocate_and_copy_recording_inputs(
        self, inputs
        # 下面的代码未提供，需要完成这个方法的注释
    ) -> List[Union[torch.Tensor, int]]:
        """
        为非静态、非 cudagraph 管理的张量在内存池中分配输入，并复制张量的值。
        """

        # 同步所有的 CUDA 设备，确保之前的操作已完成
        torch.cuda.synchronize()
        # 等待当前 CUDA 流的完成
        self.stream.wait_stream(torch.cuda.current_stream())
        # 用于记录输入的列表，可以是张量或整数
        recording_inputs: List[Union[Tensor, int]] = []

        # 使用警告捕获记录，切换到指定的 CUDA 设备，并使用 CUDA 内存池管理器
        with warnings.catch_warnings(record=True), torch.cuda.device(
            self.device
        ), _use_cuda_memory_pool_manager(
            self.device,
            mem_pool=self.cuda_graphs_pool,
            stream=self.stream,
        ):
            # 遍历输入列表
            for i, inp in enumerate(inputs):
                # 如果输入不是张量，则添加到记录列表中
                if not isinstance(inp, torch.Tensor):
                    assert isinstance(inp, int)  # 断言输入确实是整数
                    recording_inputs.append(inp)
                # 如果是张量，并且索引不在静态输入索引中
                elif i not in self.static_input_idxs:
                    # static_input 执行分配操作！
                    recording_inputs.append(static_input(inp))
                else:
                    # 否则，直接将张量添加到记录列表中
                    recording_inputs.append(inp)

            # 复制输入并从原始输入中移除
            self._copy_inputs_and_remove_from_src(recording_inputs, inputs)

        # 返回记录的输入列表
        return recording_inputs
    # 检查节点是否可以运行的不变性条件
    def check_invariants(self, inputs: List[Tensor]) -> bool:
        """
        Checks if this node can be run. The same pattern of tensor liveness, static inputs,
        and tensors managed in the cudagraph private pool must remain stable.
        """

        # 之前管理的数据指针必须保持稳定
        # 由于这是热路径，已移动到 C++ 实现。相当于：
        # return all(t.data_ptr() == data_ptr for (t, data_ptr) in zip(tensors, data_ptrs))
        if not torch._C._tensors_data_ptrs_at_indices_equal(
            inputs, self.static_input_data_ptrs, self.cudagraph_managed_idxs
        ):
            return False

        # 静态输入数据指针应保持稳定
        # 如果我们内联了内置的 nn 模块，则在这种情况下重新记录
        # 如果我们没有内联内置的 nn 模块，则在 check_static_inputs_are_stable 中检查并在不稳定时报错
        if (
            self.rerecord_if_static_inputs_change
            and not torch._C._tensors_data_ptrs_at_indices_equal(
                inputs, self.static_input_data_ptrs, self.static_input_idxs
            )
        ):
            return False

        # 检查预期死亡索引和路径弱引用以验证张量的存活状态
        if not self._check_liveness(
            self.expected_dead_indices_before_graph, self.path_weakrefs
        ):
            return False

        # cudagraph 管理的张量在记录图时死亡，调用时也必须死亡。
        # 如果在重放图时我们已经写入了它们的内存，那么事后检查就太晚了。
        for idx in self.cudagraph_managed_idxs:
            inputs[idx] = None  # type: ignore[call-overload]

        # 检查图记录期间观察到输入张量在重放期间未释放的情况
        torch._check(
            self._check_liveness(
                self.expected_dead_indices_after_graph, self.path_weakrefs
            ),
            lambda: "TODO: graph recording observed an input tensor deallocate during graph "
            " recording that did not occur during replay. Please file an issue.",
        )
        return True

    # 返回此节点的后代总数
    def num_descendants(self) -> int:
        "Total number of descendents of this node"
        num_desc = 0
        for children in self.children.values():
            for child in children:
                num_desc += 1
                num_desc += child.num_descendants()
        return num_desc
# 获取当前 CUDA 图段的内存快照
def get_cudagraph_segments(pool_id):
    segments = torch.cuda.memory_snapshot()
    return [segment for segment in segments if segment["segment_pool_id"] == pool_id]


# 获取指定 CUDA 图段中的内存块地址列表
def get_block_addrs(pool_id, live_only=True):
    blocks = []

    # 遍历指定 CUDA 图段中的所有段
    for segment in get_cudagraph_segments(pool_id):
        addr = segment["address"]
        # 遍历每个段中的内存块
        for block in segment["blocks"]:
            # 如果内存块状态为活跃分配或 live_only 为 False，则添加地址到列表中
            if block["state"] == "active_allocated" or not live_only:
                blocks.append(addr)

            addr += block["size"]

    return blocks


# 格式化堆栈回溯信息
def format_tb(frames):
    formatted_traceback = []

    # 遍历每个堆栈帧的信息，格式化为 traceback.FrameSummary 对象
    for entry in frames:
        formatted_traceback.append(
            traceback.FrameSummary(entry["filename"], entry["line"], entry["name"])
        )

    return "".join(traceback.format_list(formatted_traceback))


# 检查内存池的存储引用
def check_memory_pool(device, pool_id, live_storages_ptrs: List[StorageWeakRefWrapper]):
    # 断言所有存储引用都是 StorageWeakRefWrapper 类型的对象
    assert all(
        isinstance(elem, StorageWeakRefWrapper) for elem in live_storages_ptrs
    )  # noqa: C419

    # 获取存储引用中的数据指针集合
    unique_storages = {stor.data_ptr() for stor in live_storages_ptrs if stor()}

    # 在进行昂贵的内存快照调用之前，先检查是否存在分歧
    if torch._C._cuda_checkPoolLiveAllocations(device, pool_id, unique_storages):
        return

    # 执行垃圾回收以处理罕见情况，即已经死亡但尚未被垃圾回收的张量
    gc.collect()

    # 获取当前 CUDA 图段的内存快照
    segments = get_cudagraph_segments(pool_id)

    # 用于存储未在活动存储引用中的分配但仍活跃的内存块
    allocated_not_in_live_storages = {}

    # 遍历每个 CUDA 图段中的地址和内存块
    for segment in segments:
        addr = segment["address"]
        for block in segment["blocks"]:
            if block["state"] == "active_allocated":
                # 如果地址不在活动存储引用中，则添加到字典中
                if addr not in unique_storages:
                    allocated_not_in_live_storages[addr] = block
                else:
                    unique_storages.remove(addr)

            addr += block["size"]

    # 检查是否所有存储引用都已分配
    torch._check(
        len(unique_storages) == 0,
        lambda: f"These storage data ptrs are not allocated in pool {pool_id} but should be {unique_storages}",
    )

    # 如果存在未计算的活跃存储引用，则生成详细的错误消息
    if allocated_not_in_live_storages != 0:
        formatted = []
        for dp, block in allocated_not_in_live_storages.items():
            trace = format_tb(block.get("frames", []))
            formatted.append(f"Data Pointer: {dp}, history: \n{trace}")
        formatted_s = "\n".join(formatted)
        msg = (
            f"These live storage data ptrs are in the cudagraph pool but not "
            f"accounted for as an output of cudagraph trees: \n\n{formatted_s}"
        )
        raise RuntimeError(msg)


# 表示 CUDA 图树的执行状态枚举
class ExecutionState(Enum):
    """
    Represents the state of the CUDAGraph Tree. Will be None if there is no live current memory allocated
    in the cuda graph pool. Otherwise will reflect the state of the most recently executed node.
    """

    NONE = auto()       # 表示没有当前分配的内存
    WARMUP = auto()     # 表示 CUDA 图树正在热身阶段
    RECORDING = auto()  # 表示 CUDA 图树正在记录执行状态
    EXECUTION = auto()  # 表示 CUDA 图树正在执行阶段
    """
    Groups individual recordings or executions of cuda graphs into a tree of recordings,
    and checks required invariants, and manages warmups of graphs.
    
    When graphs are recorded in the same tree, it enforces subsequent execution
    to follow the same order and have the same output tensor livespans. To remove
    unnecessary coupling of cuda graphs (and additional imposed invariants),
    the tree manager will end a currently recording tree whenever it is valid - when
    the memory pool no longer has any live allocations.
    
    We ignore outputs from a previous generation that correspond to prior model outputs.
    Currently this is hardcoded `GenerationTracker.generation` tracked in torch dynamo.
    # TODO: make generation increment configurable, warn on overwrite.
    
    We run graph warmups in the cudagraph memory pool and return the result on the first invocation
    of a function. For many models it is important to reclaim activations as you run the backward.
    If we were to warm up the model and keep an extra copy of the inputs around to subsequently
    use for recording, we would incur a memory penalty. Additionally, if we are part way through training
    your model and need to recompile, memory will be allocated to the cuda graph pool, so we run this
    warmup run in the cuda graph memory pool. As for recording, warm up needs the state of live tensors
    to be accurately reflected so we checkpoint the allocator state if we need to warm up following graph
    replay.
    """

    # 运行 CUDA 图节点的主要方法，执行指定函数 ID 的图计算
    def run(self, new_inputs: List[Tensor], function_id: FunctionID):
        # 确保当前 CUDA 图不为空
        assert self.graph is not None, "Running CUDAGraph after shutdown"
        # 调用内部方法 _run 执行图计算
        out = self._run(new_inputs, function_id)

        # 根据当前函数 ID 确定编译模式
        mode = self.id_to_mode[function_id]
        # 根据编译模式设置运行状态
        if mode == CompilationMode.FORWARD:
            self.running_forwards_with_pending_backwards = True
        elif mode == CompilationMode.BACKWARD:
            self.running_forwards_with_pending_backwards = False

        # 返回计算结果
        return out

    # 将当前状态设置为执行反向传播
    def set_to_running_backward(self):
        self.running_forwards_with_pending_backwards = False

    # 获取 CUDA 图记录的张量检查器函数
    def _get_cuda_graph_recorded_tensor_checker(self) -> Callable[[Tensor], bool]:
        return (
            # 如果当前节点是 CUDA 图节点或者 CUDA 热身节点，则返回其张量检查器函数
            self.current_node._is_cuda_graph_recorded_tensor
            if isinstance(self.current_node, (CUDAGraphNode, CUDAWarmupNode))
            else lambda _: False  # 否则返回一个始终返回 False 的匿名函数
        )

    # 生成新的 CUDA 图热身节点 ID
    def new_warmup_node_id(self) -> GraphID:
        return GraphID(next(self.warmup_node_counter))

    # 更新非 CUDA 图管理的变异操作
    def _update_non_cudagraph_managed_mutation(
        self, function_id: FunctionID, inputs: List[Tensor]
    ):
        # 这个方法的实现未完整给出，需要根据上下文补充
    ):
        # 获取当前节点的唯一标识符
        node_id = self._get_node_id()
        # 检查是否存在变异字符串，如果存在则返回
        if maybe_mutation_str := check_for_mutation(
            self.ids_to_funcs[function_id],  # 获取函数ID对应的函数对象
            inputs,  # 函数的输入参数
            self._get_cuda_graph_recorded_tensor_checker(),  # 获取CUDA图记录的张量检查器
        ):
            # 将当前节点标记为非CUDA图管理的变异提示
            self.non_cudagraph_managed_mutation_hint[node_id][function_id] = True
            # 如果已经警告过此function_id，则直接返回
            if function_id in self.warned_mutation:
                return
            # 将此function_id标记为已经警告过
            self.warned_mutation.add(function_id)
            # 记录CUDA图跳过并增加计数器
            log_cudagraph_skip_and_bump_counter(maybe_mutation_str)
        else:
            # 将当前节点标记为不是CUDA图管理的变异提示
            self.non_cudagraph_managed_mutation_hint[node_id][function_id] = False

    def _get_node_id(self) -> Optional[GraphID]:
        # 如果当前节点为None，则返回None
        if self.current_node is None:
            return None
        # 如果当前节点是CUDAGraphNode或者CUDAWarmupNode类型，则返回其ID
        elif isinstance(self.current_node, (CUDAGraphNode, CUDAWarmupNode)):
            return self.current_node.id
        else:
            # 抛出异常，未知的节点类型
            raise RuntimeError(f"Unknown node type {type(self.current_node)}")

    def shutdown(self):
        """
        Remove all cached tensors in all nodes. Because cached tensors can hold gradients which in turn
        might reference a backward which invokes a CUDA Graph Node, we have to manually clear them on shutdown
        to avoid a reference cycle.
        """
        # 获取所有根节点并清除所有节点的缓存张量
        nodes = []
        for roots in self.roots.values():
            nodes.extend(roots)

        while nodes:
            node = nodes.pop()
            # 递归地清除节点的缓存张量
            for children in node.children.values():
                nodes.extend(children)
            node.remove_node_cached_tensors()
            node.graph = None

        # 将当前对象的图和根节点设置为None，类型注解忽略赋值错误
        self.graph = None
        self.roots = None  # type: ignore[assignment]
        self.current_node = None

    def record_function(self, new_inputs, function_id) -> List[Optional[Tensor]]:
        # 创建一个新的图ID，并记录函数执行
        graph_id = self.new_graph_id()
        log.debug(
            "Recording function %d of graph recording id %d",
            function_id.id,
            graph_id.id,
        )
        torch.cuda.synchronize()
        # 创建一个CUDA图节点并设置为当前节点
        node = CUDAGraphNode(
            self.ids_to_funcs[function_id],  # 获取函数ID对应的函数对象
            graph_id,  # 新创建的图ID
            self.current_node,  # 当前节点
            new_inputs,  # 函数的新输入
            self.cuda_graphs_thread_pool,  # CUDA图线程池
            self.device_index,  # 设备索引
            self.ids_to_stack_traces[function_id],  # 函数ID对应的堆栈跟踪
            self.stream,  # CUDA流
        )
        # 如果当前节点为None，则将此节点添加为根节点的一个元素
        if self.current_node is None:
            self.roots[function_id].append(node)
        else:
            # 否则将此节点作为当前节点的子节点添加
            self.current_node.add_child(function_id, node)
        # 设置当前节点为新创建的节点
        self.current_node = node
        # 设置执行路径状态为记录
        self.path_state = ExecutionState.RECORDING
        # 更新代数
        self.update_generation()
        torch.cuda.synchronize()
        # 运行新输入的第一个运行
        return node.run_first_inputs(new_inputs)

    def execute_node(self, node: CUDAGraphNode, new_inputs) -> List[Optional[Tensor]]:
        # 设置当前节点为给定的节点
        self.current_node = node
        # 设置执行路径状态为执行
        self.path_state = ExecutionState.EXECUTION
        # 更新代数
        self.update_generation()
        # 运行节点的主要功能
        return node.run(new_inputs)
    def run_eager(self, new_inputs, function_id: FunctionID):
        # 检查当前函数是否已经预热过
        already_warm = function_id in self.warmed_up_functions
        # 如果函数未预热过，则记录调试信息并打印
        if not already_warm:
            log.debug("Running warmup of function %d", function_id.id)
        else:
            # 如果函数已经预热过，记录调试信息并打印
            log.debug(
                "Running eager of function %d because ancestor needed to warm up",
                function_id.id,
            )
        # 将当前函数标记为已预热
        self.warmed_up_functions.add(function_id)
        # 创建一个 CUDA 预热节点对象
        node = CUDAWarmupNode(
            self.ids_to_funcs[function_id],  # 函数 ID 对应的包装函数
            self.current_node,               # 当前节点
            self.cuda_graphs_thread_pool,    # CUDA 图线程池
            self.graph,                      # 图对象
            self.device_index,               # 设备索引
            self.ids_to_stack_traces[function_id],  # 函数 ID 对应的堆栈跟踪信息
            self.stream,                     # 流对象
            already_warm,                    # 是否已经预热过的标志
            self.new_warmup_node_id(),       # 新的预热节点 ID
        )
        # 将当前节点更新为新创建的 CUDA 预热节点
        self.current_node = node
        # 设置路径状态为预热中
        self.path_state = ExecutionState.WARMUP
        # 更新当前代数
        self.update_generation()
        # 运行新输入数据的节点
        return node.run(new_inputs)

    def new_graph_id(self) -> GraphID:
        # 生成新的图 ID
        return GraphID(next(self.graph_counter))

    def new_func_id(self) -> FunctionID:
        # 生成新的函数 ID
        return FunctionID(next(self.func_counter))

    def add_function(
        self,
        model,
        inputs,
        static_input_idxs,
        stack_traces,
        mode,
        constants,
        placeholders,
        mutated_input_idxs,
    ) -> Tuple[Callable[..., Any], List[Optional[Tensor]]]:
        # 为新函数分配一个唯一的 ID
        id = self.new_func_id()
        # 将函数 ID 与堆栈跟踪信息关联存储
        self.ids_to_stack_traces[id] = stack_traces
        # 将函数 ID 与包装函数关联存储
        self.ids_to_funcs[id] = WrappedFunction(
            model,
            list(static_input_idxs),
            id,
            tuple(t for t in constants if isinstance(t, torch.Tensor) and t.is_cuda),
            placeholders,
            mutated_input_idxs,
        )
        # 将函数 ID 与模式关联存储
        self.id_to_mode[id] = mode
        # 创建函数的部分应用，以便后续调用
        fn = functools.partial(self.run, function_id=id)

        # 将函数的强引用添加到容器中，以便在函数对象销毁时执行清理操作
        get_container(self.device_index).add_strong_reference(fn)
        # 返回函数及其在给定输入上的结果
        return fn, fn(inputs)

    @property
    def in_recording(self):
        # 检查当前路径状态是否为录制中
        return self.path_state == ExecutionState.RECORDING

    @property
    def in_warmup(self):
        # 检查当前路径状态是否为预热中
        return self.path_state == ExecutionState.WARMUP

    def get_roots(self) -> Iterator[CUDAGraphNode]:
        # 获取所有根节点的迭代器
        for nodes in self.roots.values():
            yield from nodes

    @property
    def current_node(self):
        # 返回当前节点对象
        return self._current_node

    @current_node.setter
    def current_node(self, value):
        # 设置当前节点对象，并根据是否为空更新路径状态
        self._current_node = value
        if value is None:
            self.path_state = ExecutionState.NONE

    def update_generation(self):
        # 更新当前代数为当前标记步骤计数器或生成器的值
        self.current_gen = self.get_curr_generation()

    @staticmethod
    def get_curr_generation() -> int:
        # 获取当前代数，优先使用标记步骤计数器，否则使用生成器的值
        if MarkStepBox.mark_step_counter != 0:
            return MarkStepBox.mark_step_counter

        return GenerationTracker.generation

    @staticmethod
    # 检查是否用户调用了标记步骤，即标记步骤计数器不为零时返回True
    def user_invoked_mark_step():
        return MarkStepBox.mark_step_counter != 0

    # 检查是否可以启动新的生成，需要满足以下条件：
    # 1. 不在新的Torch编译调用中
    # 2. 用户调用了标记步骤
    # 3. 没有未完成的前向执行并且没有待处理的反向执行
    def can_start_new_generation(self) -> bool:
        if not self.in_new_torch_compile_invocation():
            return False

        if self.user_invoked_mark_step():
            return True

        return not self.running_forwards_with_pending_backwards

    # 检查是否在新的Torch编译调用中，即当前生成与当前生成号码不相等时返回True
    def in_new_torch_compile_invocation(self):
        return self.current_gen != self.get_curr_generation()

    # 尝试结束当前录制的节点，条件包括：
    # 1. 可以开始新的生成
    # 2. 当前节点的所有输出均已失效
    # 如果条件满足，将清除当前路径状态并将当前节点设为None
    def try_end_curr_recording(self, function_id: FunctionID) -> None:
        """
        检查当前录制的节点是否可以结束，条件包括：
        - 前一个节点的所有输出都失效了
        - 或者因为在不同的生成中执行
        如果成功，将把current_node设为None，并将in_recording设为False。
        """
        assert self.in_recording
        assert self.current_node is not None

        if self.can_start_new_generation():
            self.dealloc_current_path_weakrefs()
            self.clear_current_path_state_and_set_to_none()
            return

        if self.current_node.all_outputs_are_dead():
            self.clear_current_path_state_and_set_to_none()
            return

        self.check_warn_on_unable_to_start_executing(function_id)

    # 尝试结束当前执行的节点，条件包括：
    # 1. 可以开始新的生成
    # 2. 当前节点的所有输出均已失效
    # 如果条件满足，将把current_node设为None
    def try_end_curr_execution(self) -> None:
        """
        检查当前执行的节点是否可以结束，条件包括：
        - 前一个节点的所有输出都失效了
        - 或者因为在不同的生成中执行
        如果成功，将把current_node设为None。
        """
        assert not self.in_recording
        if self.current_node is None:
            return

        if self.can_start_new_generation():
            self.clear_current_path_state_and_set_to_none()
            return

        if self.current_node.all_outputs_are_dead():
            self.clear_current_path_state_and_set_to_none()

    # 尝试结束当前热身阶段，条件包括：
    # 1. 可以开始新的生成
    # 2. 当前节点的所有输出均已失效
    # 如果条件满足，将把current_node设为None
    def try_end_curr_warmup(self, function_id: FunctionID):
        if self.can_start_new_generation():
            self.dealloc_current_path_weakrefs()
            self.current_node = None
            return

        if self.current_node.all_outputs_are_dead():
            self.current_node = None
            return

        self.check_warn_on_unable_to_start_executing(function_id)
    # 检查是否需要在无法开始执行时发出警告
    def check_warn_on_unable_to_start_executing(self, function_id: FunctionID):
        "Warn if we in a potential loop where we are unable to hit fast path"
        # 如果函数已经在警告列表中或者不在新的torch编译调用中，则直接返回
        if (
            function_id in self.warned_functions
            or not self.in_new_torch_compile_invocation()
        ):
            return

        # 在当前节点的路径中查找具有指定函数ID的节点列表
        existing_nodes = [
            node
            for node in self.current_node._path_from_root
            if node.wrapped_function.id == function_id
        ]

        # 如果找到的节点数量小于等于1，则直接返回
        if len(existing_nodes) <= 1:
            return

        # 检查是否存在重复的父节点，如果都是同一模式，则直接返回
        parents = {
            n.parent.wrapped_function.id
            for n in itertools.chain(existing_nodes, (self.current_node,))
            if n.parent is not None
        }
        if len(parents) == len(existing_nodes):
            return

        # 将当前函数ID添加到警告函数集合中
        self.warned_functions.add(function_id)
        # 发出警告，说明无法通过快速路径访问CUDAGraphs，可能因为有未调用的反向传播操作
        warnings.warn(
            "Unable to hit fast path of CUDAGraphs because of pending, uninvoked backwards. "
            "Consider running with torch.no_grad() or using torch.compiler.cudagraph_mark_step_begin() "
            "before each model invocation"
        )

    # 释放当前路径中的弱引用
    def dealloc_current_path_weakrefs(self):
        # TODO: we could also allow the these weak refs to continue to be allocated,
        # but that adds some complications.
        # 遍历当前节点路径中的每个节点
        for node in self.current_node._path_from_root:
            # 断言每个节点的张量弱引用与栈追踪数量相等
            assert len(node.tensor_weakrefs) == len(node.stack_traces)
            # 遍历每个节点的张量弱引用和栈追踪
            for t, stack_trace in zip(node.tensor_weakrefs, node.stack_traces):
                # 获取张量对象，如果已被释放则跳过
                ten = None if t is None else t()
                if ten is None:
                    continue

                # 清理栈追踪信息，如果没有则给出默认信息
                stack_trace = (
                    stack_trace.strip()
                    if stack_trace
                    else "[Could not find stack trace]"
                )
                # 设置存储访问错误信息，防止在后续运行中被重写
                msg = (
                    "Error: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run. "
                    f"Stack trace: {stack_trace}. "
                    "To prevent overwriting, clone the tensor outside of torch.compile() "
                    "or call torch.compiler.cudagraph_mark_step_begin() before each model invocation."
                )
                torch._C._set_storage_access_error_msg(ten, msg)

        # 释放和移除当前节点路径中的存储
        deleted = set()
        for storage_ref in self.current_node.path_live_weakrefs():
            if storage_ref() and storage_ref.data_ptr() not in deleted:
                deleted.add(storage_ref.data_ptr())
                torch._C._free_And_Remove_DeleterFn(storage_ref())

    # 清除当前路径状态并将当前节点设置为None
    def clear_current_path_state_and_set_to_none(self):
        # 清除当前节点的路径状态
        self.current_node.clear_path_state()
        # 将当前节点设置为None
        self.current_node = None
    def apply_checkpoint_execution_state_in_allocator(self):
        """
        在缓存分配器中设置当前执行状态的检查点，以便可以在现有的存储器上进行额外的 cudagraph 记录。
        """
        # 增加调试检查点计数器
        self.debug_checkpointing_counter += 1
        # 记录调试信息：CUDA 缓存分配器状态检查点数量
        log.debug(
            "Checkpointing cuda caching allocator state. Number of checkpoints %d",
            self.debug_checkpointing_counter,
        )

        # 获取当前节点的检查点缓存状态和设备信息
        state = self.current_node.checkpointed_caching_state
        device = self.current_node.device
        assert state is not None and device is not None

        # 目前我们正在释放而不是允许旧的记录
        stale_storages: List[int] = []

        # 移除缓存的张量，否则它们会阻止内存在后续记录中被回收
        self.current_node.remove_path_cached_tensors()
        # 获取当前路径中的存活弱引用存储
        live_storages_wrappers = list(self.current_node.path_live_weakrefs())

        # 获取存活弱引用存储的真实对象列表
        live_storages_weak_refs = [t() for t in live_storages_wrappers]
        # 设置 CUDA 缓存分配器的检查点池状态
        torch._C._cuda_setCheckpointPoolState(
            device, state, stale_storages, live_storages_weak_refs
        )

        # 注意：去重别名输出
        # 释放需要释放的指针
        for ptr in set(ptrs_to_deallocate):
            torch._C._cuda_cudaCachingAllocator_raw_delete(ptr)

        # 现在存活的块应该完全等于私有池中的存活存储
        if config.triton.slow_path_cudagraph_asserts:
            # 检查内存池状态
            check_memory_pool(
                self.device_index, self.cuda_graphs_thread_pool, live_storages_wrappers
            )
            # 对每个存活存储的 wrapper 进行额外的断言检查
            for wrapper in live_storages_wrappers:
                assert wrapper()
                assert torch._C._has_Standard_Deleter(wrapper())
                assert wrapper.data_ptr() not in ptrs_to_deallocate

    def live_cudagraph_pool_storages_in_curr_execution(
        self,
    ) -> List[StorageWeakRefPointer]:
        if self.current_node is None:
            return []
        # 明确忽略来自过去路径的先前记录的输出
        # 返回当前路径中的存活弱引用存储列表
        return [t() for t in self.current_node.path_live_weakrefs()]
```