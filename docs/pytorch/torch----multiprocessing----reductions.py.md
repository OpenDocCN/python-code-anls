# `.\pytorch\torch\multiprocessing\reductions.py`

```py
# mypy: allow-untyped-defs
# 引入多进程相关模块
import multiprocessing
import os
import threading
# 导入用于多进程序列化的ForkingPickler
from multiprocessing.reduction import ForkingPickler
# 导入多进程工具中的注册函数
from multiprocessing.util import register_after_fork
# 导入Union类型
from typing import Union

# 导入PyTorch库
import torch
# 导入PyTorch中用于检查命名张量序列化的函数
from torch._namedtensor_internals import check_serializing_named_tensor

try:
    # 提前加载resource_sharer以防止在派生的子进程中继承部分初始化的实例。
    # reduce_storage方法通过DupFd()间接需要这个模块。
    # 内置的mp.Queue类在后台线程中对参数进行pickle，这可能与fork重叠。
    import multiprocessing.resource_sharer
except ImportError:
    pass

class StorageWeakRef:
    r"""A weak reference to a Storage.

    The cdata member is a Python number containing the integer representation of
    the Storage pointer.
    """

    __slots__ = ["cdata", "_free_weak_ref"]

    def __init__(self, storage):
        # 获取Storage对象的弱引用指针
        self.cdata = storage._weak_ref()
        # 保存_free_weak_ref的直接引用，因为在此模块清除之前，可能在Python关闭期间清除torch模块。
        self._free_weak_ref = torch.Storage._free_weak_ref  # type: ignore[attr-defined]

    @classmethod
    def from_weakref(cls, cdata):
        # 从弱引用指针创建实例
        instance = cls.__new__(cls)
        instance.cdata = cdata
        instance._free_weak_ref = torch.Storage._free_weak_ref  # type: ignore[attr-defined]
        return instance

    def expired(self):
        # 检查Storage是否已过期
        return torch.Storage._expired(self.cdata)  # type: ignore[attr-defined]

    def __del__(self):
        # 释放Storage的弱引用
        self._free_weak_ref(self.cdata)

    def __hash__(self):
        # 返回Storage弱引用的哈希值
        return self.cdata

    def __eq__(self, other):
        # 比较两个StorageWeakRef对象是否相等
        if id(self) == id(other):
            return True
        return self.cdata == other.cdata


class SharedCache(dict):
    """Dictionary from multiprocessing handles to StorageWeakRef."""

    def __init__(self):
        # 当长度超过当前限制时，调用free_dead_references()来释放死引用。
        # 限制随剩余活动对象数量扩展。
        self.limit = 128
        # 如果在持有锁的情况下进行fork，fork会继承锁状态，
        # 因此我们注册一个函数来将锁重置为新对象，以避免可能的死锁，遵循Python多进程库的设计。
        self._after_fork()
        register_after_fork(self, SharedCache._after_fork)

    def _after_fork(self):
        # 在fork后重新设置锁对象
        self.lock = threading.Lock()

    def get(self, key):
        # 获取指定键的值，加锁操作
        with self.lock:
            return dict.get(self, key)

    def __setitem__(self, key, storage_ref):
        # 设置指定键的值，并加锁操作
        with self.lock:
            dict.__setitem__(self, key, storage_ref)
            # 如果超过限制长度，则释放死引用
            if len(self) > self.limit:
                self.free_dead_references()
    # 定义方法，用于释放已过期引用
    def free_dead_references(self):
        # 初始化活跃引用计数
        live = 0
        # 遍历当前字典中所有项目的副本
        for key, storage_ref in list(self.items()):
            # 检查存储引用是否已过期
            if storage_ref.expired():
                # 如果过期，从字典中删除对应键值对
                del self[key]
            else:
                # 如果未过期，增加活跃引用计数
                live += 1
        # 更新限制值，确保至少为128，且是活跃引用数的两倍
        self.limit = max(128, live * 2)
# 创建一个共享缓存实例，用于存储和管理共享对象的弱引用
shared_cache = SharedCache()


# 根据设备和句柄重建一个 CUDA 事件对象
def rebuild_event(device, handle):
    return torch.cuda.Event.from_ipc_handle(device, handle)


# 将事件对象转换为可序列化的元组形式，以便进行进程间通信
def reduce_event(event):
    handle = event.ipc_handle()
    return (rebuild_event, (event.device, handle))


# 根据存储、元数据等信息重建张量对象
def rebuild_tensor(cls, storage, metadata):
    storage_offset, size, stride, requires_grad = metadata
    # 使用存储、偏移、大小和步长重建张量
    t = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
    if cls == torch.nn.parameter.Parameter:
        # 对于参数张量，需要在构造函数中传递 requires_grad 参数，因为对整数张量来说，要求 requires_grad=False 是一个重要的检查条件
        t = torch.nn.parameter.Parameter(t, requires_grad=requires_grad)
    else:
        # 设置张量的 requires_grad 属性
        t.requires_grad = requires_grad
    return t


# 根据 CUDA 存储、事件等信息重建 CUDA 张量对象
def rebuild_cuda_tensor(
    tensor_cls,
    tensor_size,
    tensor_stride,
    tensor_offset,
    storage_cls,
    dtype,
    storage_device,
    storage_handle,
    storage_size_bytes,
    storage_offset_bytes,
    requires_grad,
    ref_counter_handle,
    ref_counter_offset,
    event_handle,
    event_sync_required,
):
    # 如果存储句柄为空或存储大小为零，则创建一个新的存储对象
    if storage_handle is None or storage_size_bytes == 0:
        storage = storage_cls(0, dtype=dtype, device=storage_device, _internal=True)
    else:
        # 从缓存中获取或创建 CUDA 存储对象
        storage = storage_from_cache(
            storage_cls, (storage_handle, storage_offset_bytes)
        )
        if storage is None:
            # 如果缓存中不存在，则进行延迟初始化并创建新的 CUDA 存储对象
            torch.cuda._lazy_init()
            storage = storage_cls._new_shared_cuda(
                storage_device,
                storage_handle,
                storage_size_bytes,
                storage_offset_bytes,
                ref_counter_handle,
                ref_counter_offset,
                event_handle,
                event_sync_required,
            )
            # 将新创建的存储对象以及其偏移加入共享缓存中
            shared_cache[(storage_handle, storage_offset_bytes)] = StorageWeakRef(
                storage
            )
        else:
            # 如果存储对象已存在，则增加其引用计数
            storage_cls._release_ipc_counter(
                ref_counter_handle, ref_counter_offset, device=storage_device
            )

    # 获取未类型化的存储对象并重建张量
    _storage = (
        storage
        if isinstance(storage, torch.UntypedStorage)
        else storage._untyped_storage
    )

    # 使用存储对象、偏移、大小和步长重建张量
    t = torch._utils._rebuild_tensor(
        torch.storage.TypedStorage(wrap_storage=_storage, dtype=dtype, _internal=True),
        tensor_offset,
        tensor_size,
        tensor_stride,
    )

    if tensor_cls == torch.nn.parameter.Parameter:
        # 对于参数张量，需要在构造函数中传递 requires_grad 参数
        t = torch.nn.parameter.Parameter(t, requires_grad=requires_grad)
    else:
        # 设置张量的 requires_grad 属性
        t.requires_grad = requires_grad

    return t


# 函数定义未完整，待后续完善
def reduce_tensor(tensor):
    # 检查张量是否需要梯度并且不是叶子节点，如果是，则抛出运行时错误
    if tensor.requires_grad and not tensor.is_leaf:
        raise RuntimeError(
            "Cowardly refusing to serialize non-leaf tensor which requires_grad, "
            "since autograd does not support crossing process boundaries.  "
            "If you just want to transfer the data, call detach() on the tensor "
            "before serializing (e.g., putting it on the queue)."
        )

    # 检查是否有命名张量正在被序列化
    check_serializing_named_tensor(tensor)
    # 在张量上发出警告，如果它有挂钩（hooks）
    torch.utils.hooks.warn_if_has_hooks(tensor)

    # 注意 [CUDA IPC and the caching allocator]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 当通过IPC发送CUDA张量时，你可能希望在接收端得到相同的存储。然而，CUDA缓存分配器使得保持这一不变条件变得困难。
    # 考虑以下情况：大小为0x100的张量指向大小为0x100的存储的偏移量0x20，该存储位于0xA100，但实际分配器可能是从0xA000到0xA3FF。
    #
    # 当我们想要通过IPC发送这个CUDA张量时，我们必须发送整个cudaMalloc分配，即0xA000区域，而不仅仅是0xA100的存储（因为这是CUDA支持的）。
    # 因此，在接收端，没有办法说，“等等，你给我一个比我想要的更大的区域（0xA000）”。
    #
    # 如果你发送了cudaMalloc分配，你可以将其包装为一个存储吗？不行，因为cudaMalloc分配可能包含不同类型的存储：float、bytes、double等等。
    # 如果你将整个分配作为类型A的单一存储，当在该存储上构造类型B的张量时会遇到错误。
    #
    # cudaIpcMemHandle是用于在接收端访问发送端cudaMalloc分配的标识符。然而，每个设备在一个进程中的每个其他进程只能由一个上下文打开。
    # 如果我们在一个进程中多次打开和关闭内存句柄，CUDA可能会给出一个不同的地址；同样，一旦我们关闭了内存，即使它在原始进程中仍然活跃，我们也不允许访问它（以及构建在其上的存储/张量）。
    # 由于无法一次性将cudaMalloc分配到一个单一的存储，这要求我们在C++端缓存每个cudaIpcMemHandle的设备指针来重建存储类型，同时保持旧的存储活动。
    # 参见[https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__DEVICE.html]
    #
    # 这是可以接受的，因为我们只需要保存分配中的位置，并从中重建存储和张量。
    # 0xA000 -> -------CUDA分配------
    #           |                            |
    # TODO: Handle distinguishing between subclass and non-subclass versions of NT better
    # https://github.com/pytorch/pytorch/issues/110543
    
    # 从 torch.nested._internal.nested_tensor 模块导入 NestedTensor 类
    from torch.nested._internal.nested_tensor import NestedTensor
    
    # 如果 tensor 是嵌套的，并且不是 NestedTensor 的实例
    if tensor.is_nested and not isinstance(tensor, NestedTensor):
        # 返回对嵌套张量进行归约的结果
        return reduce_nested_tensor(tensor)
    # 检查张量的布局是否为稀疏格式之一
    if tensor.layout in {
        torch.sparse_coo,
        torch.sparse_csr,
        torch.sparse_bsr,
        torch.sparse_csc,
        torch.sparse_bsc,
    }:
        # 如果是稀疏张量，调用函数以减少稀疏张量的表示
        return reduce_sparse_tensor(tensor)

    # 获取张量的类型化存储
    storage = tensor._typed_storage()

    # 如果存储在 CUDA 设备上
    if storage._untyped_storage.device.type == "cuda":
        # 从 CUDA 存储中共享信息
        (
            device,
            handle,
            storage_size_bytes,
            storage_offset_bytes,
            ref_counter_handle,
            ref_counter_offset,
            event_handle,
            event_sync_required,
        ) = storage._share_cuda_()
        # 获取张量在其存储中的偏移量
        tensor_offset = tensor.storage_offset()
        # 将 CUDA 存储句柄和弱引用存储到共享缓存中
        shared_cache[handle] = StorageWeakRef(storage)
        # _backward_hooks 故意在此省略，参见注释 [不序列化钩子]
        return (
            rebuild_cuda_tensor,
            (
                type(tensor),
                tensor.size(),
                tensor.stride(),
                tensor_offset,  # 张量在其存储中的偏移量
                type(storage),
                tensor.dtype,
                device,
                handle,  # 标识 CUDA 分配中存储的标识符
                storage_size_bytes,  # 存储的大小（字节）
                storage_offset_bytes,  # 存储在 CUDA 分配中的偏移量（字节）
                tensor.requires_grad,
                ref_counter_handle,
                ref_counter_offset,
                event_handle,
                event_sync_required,
            ),
        )

    # _backward_hooks 故意在此省略，参见注释 [不序列化钩子]
    # 构建张量的元数据元组
    metadata = (
        tensor.storage_offset(),
        tensor.size(),
        tensor.stride(),
        tensor.requires_grad,
    )
    # 返回重建张量的函数和其类型、存储及元数据
    return (rebuild_tensor, (type(tensor), storage, metadata))
# 重建嵌套张量的函数，从给定的函数和参数中重建缓冲区
def rebuild_nested_tensor(
    rebuild_buffer_func,
    rebuild_buffer_args,
    rebuild_sizes_func,
    rebuild_sizes_args,
    rebuild_strides_func,
    rebuild_strides_args,
    rebuild_offsets_func,
    rebuild_offsets_args,
):
    # 调用重建缓冲区函数以获取缓冲区
    buffer = rebuild_buffer_func(*rebuild_buffer_args)
    # 调用重建尺寸函数以获取尺寸
    sizes = rebuild_sizes_func(*rebuild_sizes_args)
    # 调用重建步长函数以获取步长
    strides = rebuild_strides_func(*rebuild_strides_args)
    # 调用重建偏移量函数以获取偏移量
    offsets = rebuild_offsets_func(*rebuild_offsets_args)
    # 使用缓冲区、尺寸、步长和偏移量创建并返回嵌套视图张量
    return torch._nested_view_from_buffer_copy(buffer, sizes, strides, offsets)


# 减少嵌套张量的函数，返回用于重建的函数和参数
def reduce_nested_tensor(nt):
    # 从嵌套张量的值中减少张量，并获取重建缓冲区函数及其参数
    rebuild_buffer_func, rebuild_buffer_args = reduce_tensor(nt.values())
    # 从嵌套张量的大小信息中减少张量，并获取重建尺寸函数及其参数
    rebuild_sizes_func, rebuild_sizes_args = reduce_tensor(nt._nested_tensor_size())
    # 从嵌套张量的步长信息中减少张量，并获取重建步长函数及其参数
    rebuild_strides_func, rebuild_strides_args = reduce_tensor(
        nt._nested_tensor_strides()
    )
    # 从嵌套张量的存储偏移信息中减少张量，并获取重建偏移量函数及其参数
    rebuild_offsets_func, rebuild_offsets_args = reduce_tensor(
        nt._nested_tensor_storage_offsets()
    )

    # 返回重建嵌套张量函数及其所需的函数和参数
    return (
        rebuild_nested_tensor,
        (
            rebuild_buffer_func,
            rebuild_buffer_args,
            rebuild_sizes_func,
            rebuild_sizes_args,
            rebuild_strides_func,
            rebuild_strides_args,
            rebuild_offsets_func,
            rebuild_offsets_args,
        ),
    )


# 重建稀疏 COO 格式张量的函数，使用给定的函数和参数重建稀疏 COO 格式张量
def rebuild_sparse_coo_tensor(
    rebuild_indices_func,
    rebuild_indices_args,
    rebuild_values_func,
    rebuild_values_args,
    shape,
    is_coalesced,
):
    # 调用重建索引函数以获取索引
    indices = rebuild_indices_func(*rebuild_indices_args)
    # 调用重建值函数以获取值
    values = rebuild_values_func(*rebuild_values_args)
    # 使用索引、值、形状和是否合并属性创建并返回稀疏 COO 格式张量
    return torch.sparse_coo_tensor(indices, values, shape, is_coalesced=is_coalesced)


# 重建稀疏压缩格式张量的函数，使用给定的函数和参数重建稀疏压缩格式张量
def rebuild_sparse_compressed_tensor(
    rebuild_compressed_indices_func,
    rebuild_compressed_indices_args,
    rebuild_plain_indices_func,
    rebuild_plain_indices_args,
    rebuild_values_func,
    rebuild_values_args,
    shape,
    layout,
):
    # 调用重建压缩索引函数以获取压缩索引
    compressed_indices = rebuild_compressed_indices_func(
        *rebuild_compressed_indices_args
    )
    # 调用重建普通索引函数以获取普通索引
    plain_indices = rebuild_plain_indices_func(*rebuild_plain_indices_args)
    # 调用重建值函数以获取值
    values = rebuild_values_func(*rebuild_values_args)
    # 使用压缩索引、普通索引、值、形状和布局创建并返回稀疏压缩格式张量
    return torch.sparse_compressed_tensor(
        compressed_indices, plain_indices, values, shape, layout=layout
    )


# 减少稀疏张量的函数，根据布局类型选择相应的函数和参数进行张量的重建
def reduce_sparse_tensor(sparse):
    # 如果稀疏张量的布局是 COO 格式
    if sparse.layout is torch.sparse_coo:
        # 从稀疏张量的索引信息中减少张量，并获取重建索引函数及其参数
        rebuild_indices_func, rebuild_indices_args = reduce_tensor(sparse._indices())
        # 从稀疏张量的值信息中减少张量，并获取重建值函数及其参数
        rebuild_values_func, rebuild_values_args = reduce_tensor(sparse._values())
        # 返回重建稀疏 COO 格式张量函数及其所需的函数和参数
        return (
            rebuild_sparse_coo_tensor,
            (
                rebuild_indices_func,
                rebuild_indices_args,
                rebuild_values_func,
                rebuild_values_args,
                sparse.shape,
                sparse.is_coalesced(),
            ),
        )
    else:
        # 如果稀疏张量的布局是 CSR 或者 BSR
        if sparse.layout in {torch.sparse_csr, torch.sparse_bsr}:
            # 获取压缩索引
            compressed_indices = sparse.crow_indices()
            # 获取普通索引
            plain_indices = sparse.col_indices()
        # 如果稀疏张量的布局是 CSC 或者 BSC
        elif sparse.layout in {torch.sparse_csc, torch.sparse_bsc}:
            # 获取压缩索引
            compressed_indices = sparse.ccol_indices()
            # 获取普通索引
            plain_indices = sparse.row_indices()
        else:
            # 如果布局不在支持的范围内，则抛出未实现错误
            raise NotImplementedError(sparse.layout)
        # 对压缩索引进行张量减少操作，返回处理函数及其参数
        (
            rebuild_compressed_indices_func,
            rebuild_compressed_indices_args,
        ) = reduce_tensor(compressed_indices)
        # 对普通索引进行张量减少操作，返回处理函数及其参数
        rebuild_plain_indices_func, rebuild_plain_indices_args = reduce_tensor(
            plain_indices
        )
        # 对稀疏张量的值进行张量减少操作，返回处理函数及其参数
        rebuild_values_func, rebuild_values_args = reduce_tensor(sparse.values())
        # 返回重建的稀疏压缩张量及其重建所需参数
        return (
            rebuild_sparse_compressed_tensor,
            (
                rebuild_compressed_indices_func,
                rebuild_compressed_indices_args,
                rebuild_plain_indices_func,
                rebuild_plain_indices_args,
                rebuild_values_func,
                rebuild_values_args,
                sparse.shape,
                sparse.layout,
            ),
        )
#`
def fd_id(fd):
    # 返回一个元组，用于唯一标识一个文件描述符。在 Mac OS 上，这个方法不支持共享内存句柄，因此在该平台上不支持 "file_descriptor" 共享方法。
    stat = os.fstat(fd)
    # 返回文件描述符的 inode 和设备号
    return (stat.st_ino, stat.st_dev)

def storage_from_cache(cls, key):
    # 从共享缓存中获取指定键的存储引用
    storage_ref = shared_cache.get(key)
    # 如果缓存中没有对应的存储引用，返回 None
    if storage_ref is None:
        return None
    # 使用存储引用的 cdata 创建一个未类型化存储对象
    return torch.UntypedStorage._new_with_weak_ptr(storage_ref.cdata)

def rebuild_storage_fd(cls, df, size):
    # 分离数据帧的文件描述符
    fd = df.detach()
    try:
        # 从缓存中尝试获取对应文件描述符的存储
        storage = storage_from_cache(cls, fd_id(fd))
        # 如果缓存中有存储，直接返回
        if storage is not None:
            return storage
        # 如果缓存中没有，创建一个共享的 CPU 文件描述符存储
        storage = cls._new_shared_fd_cpu(fd, size)
        # 将新创建的存储添加到共享缓存中
        shared_cache[fd_id(fd)] = StorageWeakRef(storage)
        return storage
    finally:
        # 关闭文件描述符
        os.close(fd)

def rebuild_storage_filename(cls, manager, handle, size, dtype=None):
    # 从缓存中获取指定 handle 的存储
    storage: Union[torch.TypedStorage, torch.UntypedStorage] = storage_from_cache(cls, handle)
    # 如果缓存中有存储，返回存储的共享引用
    if storage is not None:
        return storage._shared_decref()
    # 如果没有指定数据类型，创建未类型化共享文件名 CPU 存储
    if dtype is None:
        storage = torch.UntypedStorage._new_shared_filename_cpu(manager, handle, size)
    else:
        # 计算字节大小
        byte_size = size * torch._utils._element_size(dtype)
        # 创建未类型化共享文件名 CPU 存储
        untyped_storage: torch.UntypedStorage = torch.UntypedStorage._new_shared_filename_cpu(manager, handle, byte_size)
        # 将未类型化存储包装成类型化存储
        storage = torch.TypedStorage(wrap_storage=untyped_storage, dtype=dtype, _internal=True)
    # 将新创建的存储添加到共享缓存中
    shared_cache[handle] = StorageWeakRef(storage)
    # 返回存储的共享引用
    return storage._shared_decref()

def rebuild_storage_empty(cls):
    # 返回一个空的类实例
    return cls()

def rebuild_typed_storage(storage, dtype):
    # 返回一个新的类型化存储，包装传入的存储和数据类型
    return torch.storage.TypedStorage(wrap_storage=storage, dtype=dtype, _internal=True)

# 用于 torch.storage.TypedStorage
def reduce_typed_storage(storage):
    # 返回一个元组，包含恢复函数和存储的未类型化存储和数据类型
    return (rebuild_typed_storage, (storage._untyped_storage, storage.dtype))

def rebuild_typed_storage_child(storage, storage_type):
    # 返回一个新的子类类型化存储，包装传入的存储和存储类型
    return storage_type(wrap_storage=storage, _internal=True)

# 用于 torch.storage.TypedStorage 的子类，如 torch.FloatStorage
def reduce_typed_storage_child(storage):
    # 返回一个元组，包含恢复函数和存储的未类型化存储和存储类型
    return (rebuild_typed_storage_child, (storage._untyped_storage, type(storage)))

def reduce_storage(storage):
    from . import get_sharing_strategy

    # 如果存储在 CUDA 上，抛出异常，提示不能序列化 CUDA 存储
    if storage.is_cuda:
        raise RuntimeError("Cannot pickle CUDA storage; try pickling a CUDA tensor instead")
    # 如果共享策略为文件系统，进行文件共享操作
    elif get_sharing_strategy() == "file_system":
        # 获取存储的文件名和缓存键
        metadata = storage._share_filename_cpu_()
        cache_key = metadata[1]
        rebuild = rebuild_storage_filename
        # 如果存储是类型化存储，添加数据类型到元数据
        if isinstance(storage, torch.TypedStorage):
            metadata += (storage.dtype,)
        # 增加存储的引用计数
        storage._shared_incref()
    elif storage.size() == 0:
        # 特殊处理大小为 0 的空张量，不能被内存映射
        return (rebuild_storage_empty, (type(storage),))
    else:
        # 从 storage 对象获取共享文件描述符和大小
        fd, size = storage._share_fd_cpu_()
        # 创建文件描述符的复制对象
        df = multiprocessing.reduction.DupFd(fd)
        # 生成用于缓存的键，基于文件描述符的唯一标识
        cache_key = fd_id(fd)
        # 准备元数据，包括复制对象和大小
        metadata = (df, size)
        # 设置重新构建函数的引用
        rebuild = rebuild_storage_fd  # type: ignore[assignment]

    # 将 storage 对象的弱引用存入共享缓存中
    shared_cache[cache_key] = StorageWeakRef(storage)
    # 返回重新构建函数及其所需的参数元组
    return (rebuild, (type(storage),) + metadata)
# 注册 torch.cuda.Event 类型的序列化函数 reduce_event 到 ForkingPickler 中
def init_reductions():
    ForkingPickler.register(torch.cuda.Event, reduce_event)

    # 遍历 torch._storage_classes 列表中的每个类
    for t in torch._storage_classes:
        # 检查类名是否为 "UntypedStorage"
        if t.__name__ == "UntypedStorage":
            # 如果是 "UntypedStorage"，注册 reduce_storage 函数到 ForkingPickler 中
            ForkingPickler.register(t, reduce_storage)
        else:
            # 对于其他类型的存储类，注册 reduce_typed_storage_child 函数到 ForkingPickler 中
            ForkingPickler.register(t, reduce_typed_storage_child)

    # 注册 torch.storage.TypedStorage 类型的序列化函数 reduce_typed_storage 到 ForkingPickler 中
    ForkingPickler.register(torch.storage.TypedStorage, reduce_typed_storage)

    # 遍历 torch._tensor_classes 列表中的每个类
    for t in torch._tensor_classes:
        # 注册 reduce_tensor 函数到 ForkingPickler 中
        ForkingPickler.register(t, reduce_tensor)

    # 注册 torch.Tensor 类型的序列化函数 reduce_tensor 到 ForkingPickler 中
    ForkingPickler.register(torch.Tensor, reduce_tensor)

    # 导入 torch.nn.parameter 模块中的 Parameter 类
    from torch.nn.parameter import Parameter

    # 注册 reduce_tensor 函数到 ForkingPickler 中
    ForkingPickler.register(Parameter, reduce_tensor)
```