# `.\pytorch\torch\storage.py`

```
# mypy: allow-untyped-defs
# 导入所需模块和类型定义
import collections  # 导入 collections 模块
import copy  # 导入 copy 模块
import functools  # 导入 functools 模块
import io  # 导入 io 模块
import threading  # 导入 threading 模块
import warnings  # 导入 warnings 模块
from typing import Any, cast, Dict as _Dict, Optional as _Optional, Type, TypeVar, Union  # 导入类型定义

import torch  # 导入 torch 库
from torch._utils import _to, _type  # 导入 torch 内部工具函数
from torch.types import _bool, _int, Storage  # 导入 torch 类型定义

# 尝试导入 numpy 库，设置 HAS_NUMPY 标志
try:
    import numpy as np
    HAS_NUMPY = True
except ModuleNotFoundError:
    HAS_NUMPY = False
    np = None  # type: ignore[assignment]

# 共享内存的线程锁
_share_memory_lock = threading.Lock()
# 共享内存的映射，用于存储资源锁
_share_memory_map: _Dict[int, threading.RLock] = {}

# 定义类型变量 T，限定为 _StorageBase 或 TypedStorage 的联合类型
T = TypeVar("T", bound="Union[_StorageBase, TypedStorage]")

# 定义存储基类 _StorageBase
class _StorageBase:
    _cdata: Any  # 存储底层 C 数据的属性
    is_sparse: _bool = False  # 表示存储是否为稀疏的布尔属性，默认为 False
    is_sparse_csr: _bool = False  # 表示存储是否为 CSR 稀疏格式的布尔属性，默认为 False
    device: torch.device  # 表示存储所在设备的属性

    # 初始化方法，不执行任何操作
    def __init__(self, *args, **kwargs):
        pass

    # 返回存储长度的方法，需要在子类中实现
    def __len__(self) -> _int:
        raise NotImplementedError

    # 获取存储中指定位置元素的方法，需要在子类中实现
    def __getitem__(self, idx):
        raise NotImplementedError

    # 设置存储中指定位置元素的方法，需要在子类中实现
    def __setitem__(self, *args, **kwargs):
        raise NotImplementedError

    # 从另一个存储对象复制数据的方法，需要在子类中实现
    def copy_(self, source: T, non_blocking: _Optional[_bool] = None) -> T:
        raise NotImplementedError

    # 创建并返回新的存储对象的方法，需要在子类中实现
    def new(self) -> T:  # type: ignore[type-var]
        raise NotImplementedError

    # 返回存储占用字节数的方法，需要在子类中实现
    def nbytes(self) -> _int:
        raise NotImplementedError

    # 返回存储占用字节数的方法，别名为 size，用于兼容旧版本
    def size(self) -> _int:
        return self.nbytes()

    # 返回存储的数据类型，可选择指定目标类型，需要在子类中实现
    def type(self, dtype: _Optional[str] = None, non_blocking: _bool = False) -> T:  # type: ignore[type-var]
        return _type(self, dtype, non_blocking)

    # 将存储数据复制到 CUDA 内存中的方法，需要在子类中实现
    def cuda(self, device=None, non_blocking=False) -> T:  # type: ignore[type-var, misc] # noqa: E704
        """Returns a copy of this object in CUDA memory.

        If this object is already in CUDA memory and on the correct device, then
        no copy is performed and the original object is returned.

        Args:
            device (int): The destination GPU id. Defaults to the current device.
            non_blocking (bool): If ``True`` and the source is in pinned memory,
                the copy will be asynchronous with respect to the host. Otherwise,
                the argument has no effect.
        """
        device2 = torch.device("cuda", device) if device else torch.device("cuda")
        return self.to(device=device2, non_blocking=non_blocking)
    def hpu(self, device=None, non_blocking=False) -> T:  # type: ignore[type-var, misc] # noqa: E704
        """Returns a copy of this object in HPU memory.

        If this object is already in HPU memory and on the correct device, then
        no copy is performed and the original object is returned.

        Args:
            device (int): The destination HPU id. Defaults to the current device.
            non_blocking (bool): If ``True`` and the source is in pinned memory,
                the copy will be asynchronous with respect to the host. Otherwise,
                the argument has no effect.
        """
        # 设置目标设备为 HPU 内存，如果未指定则默认当前设备
        device2 = torch.device("hpu", device) if device else torch.device("hpu")
        # 调用 self 对象的 to 方法，将对象复制到指定的 HPU 设备上
        return self.to(device=device2, non_blocking=non_blocking)

    def element_size(self) -> _int:
        # 返回元素的大小，需要在子类中实现具体逻辑
        raise NotImplementedError

    def get_device(self) -> _int:
        # 返回对象所在设备的索引
        return self.device.index

    def data_ptr(self) -> _int:
        # 返回数据的指针地址，需要在子类中实现具体逻辑
        raise NotImplementedError

    def resizable(self) -> _bool:
        # 返回对象是否支持调整大小，需要在子类中实现具体逻辑
        raise NotImplementedError

    # Defined in torch/csrc/generic/StorageSharing.cpp
    def _share_filename_cpu_(self, *args, **kwargs):
        # 在 CPU 上共享文件名的具体实现，需要在子类中实现具体逻辑
        raise NotImplementedError

    def _share_fd_cpu_(self, *args, **kwargs):
        # 在 CPU 上共享文件描述符的具体实现，需要在子类中实现具体逻辑
        raise NotImplementedError

    @classmethod
    def _new_using_filename_cpu(cls: Type[T], size: _int) -> T:
        # 使用文件名在 CPU 上创建新对象的具体实现，需要在子类中实现具体逻辑
        raise NotImplementedError

    @classmethod
    def _new_using_fd_cpu(cls: Type[T], size: _int) -> T:
        # 使用文件描述符在 CPU 上创建新对象的具体实现，需要在子类中实现具体逻辑
        raise NotImplementedError

    @classmethod
    def from_buffer(cls: Type[T], *args, **kwargs) -> T:
        # 从缓冲区创建对象的具体实现，需要在子类中实现具体逻辑
        raise NotImplementedError

    @classmethod
    def _new_shared_filename_cpu(
        cls: Type[T],
        manager,
        obj,
        size,
        *,
        device=None,
        dtype=None,
    ) -> T:
        # 在 CPU 上共享文件名创建新对象的具体实现，需要在子类中实现具体逻辑
        raise NotImplementedError

    @classmethod
    def _release_ipc_counter_cuda(cls: Type[T], *args, **kwargs) -> T:
        # 释放 CUDA IPC 计数器的具体实现，需要在子类中实现具体逻辑
        raise NotImplementedError

    @classmethod
    def _new_with_weak_ptr(cls: Type[T], *args, **kwargs) -> T:
        # 使用弱引用指针创建新对象的具体实现，需要在子类中实现具体逻辑
        raise NotImplementedError

    def _shared_decref(self) -> T:  # type: ignore[type-var]
        # 减少共享引用计数的具体实现，需要在子类中实现具体逻辑
        raise NotImplementedError

    def _write_file(self, *args, **kwargs):
        # 写入文件的具体实现，需要在子类中实现具体逻辑
        raise NotImplementedError

    def resize_(self, size: _int):
        # 调整对象大小的具体实现，需要在子类中实现具体逻辑
        raise NotImplementedError

    def _weak_ref(self, *args, **kwargs) -> T:  # type: ignore[type-var]
        # 创建弱引用的具体实现，需要在子类中实现具体逻辑
        raise NotImplementedError

    def _set_from_file(self, *args, **kwargs):
        # 从文件设置对象的具体实现，需要在子类中实现具体逻辑
        raise NotImplementedError

    def _set_cdata(self, *args, **kwargs):
        # 设置 C 数据的具体实现，需要在子类中实现具体逻辑
        raise NotImplementedError

    def _share_cuda_(self, *args, **kwargs):
        # 在 CUDA 上共享对象的具体实现，需要在子类中实现具体逻辑
        raise NotImplementedError

    def is_shared(self) -> _bool:
        # 返回对象是否为共享对象，需要在子类中实现具体逻辑
        raise NotImplementedError

    @classmethod
    def _new_shared_cuda(cls: Type[T], *args, **kwargs) -> T:
        # 在 CUDA 上创建共享对象的具体实现，需要在子类中实现具体逻辑
        raise NotImplementedError

    def _shared_incref(self, *args, **kwargs):
        # 增加共享引用计数的具体实现，需要在子类中实现具体逻辑
        raise NotImplementedError
    # 定义一个类方法，用于释放弱引用，但是抛出未实现错误
    def _free_weak_ref(cls, *args, **kwargs):
        raise NotImplementedError

    # 属性装饰器，用于判断对象是否在 CUDA 设备上，抛出未实现错误
    @property
    def is_cuda(self):
        raise NotImplementedError

    # 属性装饰器，用于判断对象是否在 HPU 上，抛出未实现错误
    @property
    def is_hpu(self):
        raise NotImplementedError

    # 类方法，从文件中加载对象，抛出未实现错误，返回类型为 T
    @classmethod
    def from_file(cls, filename, shared, nbytes) -> T:  # type: ignore[type-var]
        raise NotImplementedError

    # 类方法，判断对象是否过期，抛出未实现错误，返回类型为 T
    @classmethod
    def _expired(cls, *args, **kwargs) -> T:  # type: ignore[type-var]
        raise NotImplementedError

    # 方法，用于字节交换，抛出未实现错误
    def _byteswap(self, *args, **kwargs):
        raise NotImplementedError

    # 方法，获取文件名，返回可选的字符串类型
    def _get_filename(self, *args, **kwargs) -> _Optional[str]:
        raise NotImplementedError

    # 返回对象的字符串表示形式，包括类型、设备和大小信息
    def __repr__(self):
        info_str = f"[{torch.typename(self)}(device={self.device}) of size {len(self)}]"
        if self.device.type == "meta":
            return "...\n" + info_str
        data_str = " " + "\n ".join(str(self[i]) for i in range(self.size()))
        return data_str + "\n" + info_str

    # 迭代器方法，返回对象的迭代器
    def __iter__(self):
        return iter(self[i] for i in range(self.size()))

    # 复制方法，返回当前对象的浅拷贝
    def __copy__(self):
        return self.clone()

    # 深拷贝方法，使用 memo 字典避免重复拷贝，返回新的对象
    def __deepcopy__(self, memo):
        memo = memo.setdefault("torch", {})
        if self._cdata in memo:
            return memo[self._cdata]
        new_storage = self.clone()
        memo[self._cdata] = new_storage
        return new_storage

    # 序列化方法，将对象保存为字节流
    def __reduce__(self):
        b = io.BytesIO()
        torch.save(self, b, _use_new_zipfile_serialization=False)
        return (_load_from_bytes, (b.getvalue(),))

    # 返回对象占用内存的大小
    def __sizeof__(self):
        return super().__sizeof__() + self.size()

    # 克隆方法，返回当前对象的深拷贝
    def clone(self):
        """Return a copy of this storage."""
        return type(self)(self.nbytes(), device=self.device).copy_(self)

    # 转换为列表方法，返回包含对象元素的列表
    def tolist(self):
        """Return a list containing the elements of this storage."""
        return list(self)

    # CPU 转换方法，返回对象在 CPU 上的副本
    def cpu(self):
        """Return a CPU copy of this storage if it's not already on the CPU."""
        if self.device.type != "cpu":
            return torch.UntypedStorage(self.size()).copy_(self, False)
        return self

    # MPS 转换方法，返回对象在 MPS 上的副本
    def mps(self):
        """Return a MPS copy of this storage if it's not already on the MPS."""
        if self.device.type != "mps":
            return torch.UntypedStorage(self.size(), device="mps").copy_(self, False)
        return self

    # 转换数据类型方法，返回转换后的存储对象
    def _to(self, dtype):
        if not isinstance(dtype, torch.dtype):
            raise TypeError(f"Argument 'dtype' must be torch.dtype, not {type(dtype)}")
        storage = (
            torch.tensor([], dtype=torch.uint8, device=self.device)
            .set_(cast(Storage, self))
            .to(dtype)
            ._typed_storage()
        )
        if storage.data_ptr() == self.data_ptr():
            storage = storage.clone()
        return storage

    # 转换设备方法，返回对象转换后的副本
    def to(self, *, device: torch.device, non_blocking: _bool = False) -> T:  # type: ignore[type-var, misc] # noqa: E704
        return _to(self, device, non_blocking)
    def double(self):
        """将存储类型转换为双精度类型。"""
        return self._to(torch.double)

    def float(self):
        """将存储类型转换为单精度类型。"""
        return self._to(torch.float)

    def half(self):
        """将存储类型转换为半精度类型。"""
        return self._to(torch.half)

    def long(self):
        """将存储类型转换为长整型类型。"""
        return self._to(torch.long)

    def int(self):
        """将存储类型转换为整型类型。"""
        return self._to(torch.int)

    def short(self):
        """将存储类型转换为短整型类型。"""
        return self._to(torch.short)

    def char(self):
        """将存储类型转换为字符类型（8位整型）。"""
        return self._to(torch.int8)

    def byte(self):
        """将存储类型转换为字节类型（无符号8位整型）。"""
        return self._to(torch.uint8)

    def bool(self):
        """将存储类型转换为布尔类型。"""
        return self._to(torch.bool)

    def bfloat16(self):
        """将存储类型转换为BFloat16类型。"""
        return self._to(torch.bfloat16)

    def complex_double(self):
        """将存储类型转换为复双精度类型。"""
        return self._to(torch.cdouble)

    def complex_float(self):
        """将存储类型转换为复单精度类型。"""
        return self._to(torch.cfloat)

    def float8_e5m2(self):
        """将存储类型转换为float8_e5m2类型。"""
        return self._to(torch.float8_e5m2)

    def float8_e4m3fn(self):
        """将存储类型转换为float8_e4m3fn类型。"""
        return self._to(torch.float8_e4m3fn)

    def float8_e5m2fnuz(self):
        """将存储类型转换为float8_e5m2fnuz类型。"""
        return self._to(torch.float8_e5m2fnuz)

    def float8_e4m3fnuz(self):
        """将存储类型转换为float8_e4m3fnuz类型。"""
        return self._to(torch.float8_e4m3fnuz)

    def is_pinned(self, device: Union[str, torch.device] = "cuda"):
        r"""确定CPU存储是否已固定在设备上。

        Args:
            device (str or torch.device): 要在其上固定内存的设备。默认为 ``'cuda'``。

        Returns:
            一个布尔变量。
        """
        return (
            torch.tensor([], dtype=torch.uint8, device=self.device)
            .set_(cast(Storage, self))
            .is_pinned(device)
        )

    def pin_memory(self, device: Union[str, torch.device] = "cuda"):
        r"""将CPU存储复制到固定内存，如果尚未固定。

        Args:
            device (str or torch.device): 要在其上固定内存的设备。默认为 ``'cuda'``。

        Returns:
            一个固定的CPU存储。
        """
        if self.device.type != "cpu":
            raise TypeError(f"无法固定 '{self.type()}'，只能固定CPU内存")

        pinned_tensor = (
            torch.tensor([], dtype=torch.uint8, device=self.device)
            .set_(cast(Storage, self))
            .pin_memory(device)
        )
        return pinned_tensor.untyped_storage()
    def share_memory_(self):
        """
        See :meth:`torch.UntypedStorage.share_memory_`
        """
        from torch.multiprocessing import get_sharing_strategy  # 导入获取共享策略的函数

        if self.device.type in ["cuda", torch._C._get_privateuse1_backend_name()]:
            pass  # CUDA或PrivateUse1不使用 POSIX 共享内存
        elif get_sharing_strategy() == "file_system":
            self._share_filename_cpu_()  # 使用文件系统共享策略共享存储在 CPU 上
        else:
            self._share_fd_cpu_()  # 使用文件描述符共享策略共享存储在 CPU 上
        return self  # 返回对象本身

    @classmethod
    def _new_shared(cls, size, *, device="cpu"):
        """
        Create a new storage in shared memory with the same data type.
        """
        from torch.multiprocessing import get_sharing_strategy  # 导入获取共享策略的函数

        device = torch.device(device)  # 根据指定设备名称创建设备对象
        if device.type in ["cuda", torch._C._get_privateuse1_backend_name(), "hpu"]:
            return cls(size, device=device)  # 在 CUDA、PrivateUse1 或 HPU 上创建新的存储对象
        elif get_sharing_strategy() == "file_system":
            return cls._new_using_filename_cpu(size)  # 使用文件名共享策略在 CPU 上创建新的存储对象
        else:
            return cls._new_using_fd_cpu(size)  # 使用文件描述符共享策略在 CPU 上创建新的存储对象

    def untyped(self):
        """
        Return self.
        """
        return self  # 返回对象本身

    def byteswap(self, dtype):
        """
        Swap bytes in underlying data.
        """
        elem_size = torch._utils._element_size(dtype)  # 获取数据类型的元素大小
        # 对于复杂类型，不交换第一个和第二个数字
        if dtype.is_complex:
            elem_size = max(int(elem_size / 2), 1)
        self._byteswap(elem_size)  # 调用内部方法来交换数据的字节顺序
# 将函数 fn 装饰为一个受共享内存锁保护的函数
def _share_memory_lock_protected(fn):
    # 使用 functools.wraps 保留原始函数的元数据
    @functools.wraps(fn)
    # 定义装饰器的包装函数，接受 self 及其他参数
    def wrapper(self, *args, **kwargs):
        # 初始化要释放和等待的变量为 None
        to_free = None
        to_wait = None
        # 使用共享内存锁 _share_memory_lock 进行同步
        with _share_memory_lock:
            # 获取当前对象的 cdata 作为键
            key = self._cdata
            # 如果键已经在共享内存映射表 _share_memory_map 中
            if key in _share_memory_map:
                # 则设置要等待的锁为该键对应的锁对象
                to_wait = _share_memory_map[key]
            else:
                # 否则，创建一个新的线程锁并加锁
                _share_memory_map[key] = threading.RLock()
                _share_memory_map[key].acquire()
                # 将要释放的键设置为当前键
                to_free = key

        # 如果已经在共享存储过程中，等待其完成
        if to_wait is not None:
            with to_wait:
                pass

        try:
            # 调用原始函数 fn，并返回其结果
            return fn(self, *args, **kwargs)
        finally:
            # 如果在此处获取了存储锁，并且完成了对其的操作
            if to_free is not None:
                # 确保存储的 cdata 没有改变，只有 data_ptr 改变了
                assert self._cdata == to_free
                # 使用共享内存锁 _share_memory_lock 进行同步
                with _share_memory_lock:
                    # 释放存储的锁并从映射表中删除该键
                    _share_memory_map[to_free].release()
                    del _share_memory_map[to_free]

    # 返回装饰后的包装函数
    return wrapper


class UntypedStorage(torch._C.StorageBase, _StorageBase):
    # 定义 __getitem__ 方法，用于获取存储的元素
    def __getitem__(self, *args, **kwargs):
        # 如果存储设备类型为 "meta"，抛出未实现错误
        if self.device.type == "meta":
            raise NotImplementedError("Not available for 'meta' device type")
        # 否则调用父类的 __getitem__ 方法，并返回结果
        return super().__getitem__(*args, **kwargs)

    # 定义 is_cuda 属性，用于检查存储是否在 CUDA 设备上
    @property
    def is_cuda(self):
        return self.device.type == "cuda"

    # 定义 is_hpu 属性，用于检查存储是否在 HPU 设备上
    @property
    def is_hpu(self):
        return self.device.type == "hpu"

    # 定义 filename 属性，返回与存储关联的文件名（可选）
    @property
    def filename(self) -> _Optional[str]:
        """Returns the file name associated with this storage.

        The file name will be a string if the storage is on CPU and was created via
        :meth:`~torch.from_file()` with ``shared`` as ``True``. This attribute is ``None`` otherwise.
        """
        return self._get_filename()

    # 应用共享内存锁保护的装饰器到当前类的方法
    @_share_memory_lock_protected
    # 将存储移到共享内存中
    def share_memory_(self, *args, **kwargs):
        """
        Moves the storage to shared memory.

        This is a no-op for storages already in shared memory and for CUDA
        storages, which do not need to be moved for sharing across processes.
        Storages in shared memory cannot be resized.

        Note that to mitigate issues like `this <https://github.com/pytorch/pytorch/issues/95606>`_
        it is thread safe to call this function from multiple threads on the same object.
        It is NOT thread safe though to call any other function on self without proper
        synchronization. Please see :doc:`/notes/multiprocessing` for more details.

        .. note::
            When all references to a storage in shared memory are deleted, the associated shared memory
            object will also be deleted. PyTorch has a special cleanup process to ensure that this happens
            even if the current process exits unexpectedly.

            It is worth noting the difference between :meth:`share_memory_` and :meth:`from_file` with ``shared = True``

            #. ``share_memory_`` uses `shm_open(3) <https://man7.org/linux/man-pages/man3/shm_open.3.html>`_ to create a
               POSIX shared memory object while :meth:`from_file` uses
               `open(2) <https://man7.org/linux/man-pages/man2/open.2.html>`_ to open the filename passed by the user.
            #. Both use an `mmap(2) call <https://man7.org/linux/man-pages/man2/mmap.2.html>`_ with ``MAP_SHARED``
               to map the file/object into the current virtual address space
            #. ``share_memory_`` will call ``shm_unlink(3)`` on the object after mapping it to make sure the shared memory
               object is freed when no process has the object open. ``torch.from_file(shared=True)`` does not unlink the
               file. This file is persistent and will remain until it is deleted by the user.

        Returns:
            ``self``
        """
        # 调用父类的 share_memory_ 方法，将存储移动到共享内存中
        return super().share_memory_(*args, **kwargs)

    # 使用装饰器保护，共享 CPU 文件描述符
    @_share_memory_lock_protected
    def _share_fd_cpu_(self, *args, **kwargs):
        return super()._share_fd_cpu_(*args, **kwargs)

    # 使用装饰器保护，共享 CPU 文件名
    @_share_memory_lock_protected
    def _share_filename_cpu_(self, *args, **kwargs):
        return super()._share_filename_cpu_(*args, **kwargs)
# 使用给定的字节流 b，通过 torch.load 将其加载为 tensor 对象并返回
def _load_from_bytes(b):
    return torch.load(io.BytesIO(b))


# 使用 functools.lru_cache 对函数进行装饰，使其具有缓存功能，最大缓存大小为 None
def _new_dtypes():
    # 返回一个包含特定 torch dtypes 的集合，用于特定的序列化操作
    return {
        torch.float8_e5m2,
        torch.float8_e4m3fn,
        torch.float8_e5m2fnuz,
        torch.float8_e4m3fnuz,
        torch.bits8,
        torch.bits16,
        torch.bits1x8,
        torch.bits2x4,
        torch.bits4x2,
        torch.complex32,
    }


# 使用 functools.lru_cache 对函数进行装饰，使其具有缓存功能，最大缓存大小为 None
def _dtype_to_storage_type_map():
    # 注意事项：不再向此映射添加新的 dtypes，仅用于与旧版 PyTorch 的向后兼容性
    # 新的 TypedStorage dtypes 不应转换为传统的 <type>Storage 类，
    # 而应该被序列化为 UntypedStorage 和 torch.dtype 的组合
    return {
        torch.double: "DoubleStorage",
        torch.float: "FloatStorage",
        torch.half: "HalfStorage",
        torch.long: "LongStorage",
        torch.int: "IntStorage",
        torch.int16: "ShortStorage",
        torch.int8: "CharStorage",
        torch.uint8: "ByteStorage",
        torch.bool: "BoolStorage",
        torch.bfloat16: "BFloat16Storage",
        torch.cdouble: "ComplexDoubleStorage",
        torch.cfloat: "ComplexFloatStorage",
        torch.qint8: "QInt8Storage",
        torch.qint32: "QInt32Storage",
        torch.quint8: "QUInt8Storage",
        torch.quint4x2: "QUInt4x2Storage",
        torch.quint2x4: "QUInt2x4Storage",
    }


# 使用 functools.lru_cache 对函数进行装饰，使其具有缓存功能，最大缓存大小为 None
def _storage_type_to_dtype_map():
    # 从 _dtype_to_storage_type_map() 中生成一个反向映射字典
    dtype_map = {val: key for key, val in _dtype_to_storage_type_map().items()}
    return dtype_map


# 根据给定的序列、dtype 和 device 创建一个 tensor 对象，并返回其 untyped storage
def _get_storage_from_sequence(sequence, dtype, device):
    if dtype in [
        torch.quint8,
        torch.quint4x2,
        torch.quint2x4,
        torch.qint32,
        torch.qint8,
    ]:
        # 对于特定的 dtype，使用 interpret_dtypes 进行解释并创建临时 tensor 对象
        interpret_dtypes = {
            torch.quint8: torch.uint8,
            torch.quint4x2: torch.uint8,
            torch.quint2x4: torch.uint8,
            torch.qint32: torch.int32,
            torch.qint8: torch.int8,
        }
        tmp_tensor = torch.tensor(
            sequence, dtype=interpret_dtypes[dtype], device=device
        )

    else:
        # 对于其他 dtype，直接使用给定的 dtype 和 device 创建 tensor 对象
        tmp_tensor = torch.tensor(sequence, dtype=dtype, device=device)

    # 返回 tensor 对象的 untyped storage
    return tmp_tensor._typed_storage()._untyped_storage


# 检查变量 x 是否为整数，如果有 numpy 则进一步检查其是否为 np.integer 类型
def _isint(x):
    if HAS_NUMPY:
        return isinstance(x, (int, np.integer))
    else:
        return isinstance(x, int)


# 返回全局变量 _always_warn_typed_storage_removal 的当前值
def _get_always_warn_typed_storage_removal():
    return _always_warn_typed_storage_removal


# 设置全局变量 _always_warn_typed_storage_removal 为指定的 always_warn 值
def _set_always_warn_typed_storage_removal(always_warn):
    global _always_warn_typed_storage_removal
    assert isinstance(always_warn, bool)
    _always_warn_typed_storage_removal = always_warn


# 简单的全局函数，用于在给定的 stacklevel 处发出警告消息
def _warn_typed_storage_removal(stacklevel=2):
    global _always_warn_typed_storage_removal
    # 检查是否是首次警告关于类型存储移除的消息
    def is_first_time():
        # 检查 _warn_typed_storage_removal 对象是否具有属性 "has_warned"
        if not hasattr(_warn_typed_storage_removal, "has_warned"):
            # 如果没有 "has_warned" 属性，返回 True 表示首次警告
            return True
        else:
            # 如果有 "has_warned" 属性，返回是否 "has_warned" 属性为 False
            return not _warn_typed_storage_removal.__dict__["has_warned"]

    # 如果设置总是警告类型存储移除或者是首次警告
    if _get_always_warn_typed_storage_removal() or is_first_time():
        # 构建警告消息字符串
        message = (
            "TypedStorage is deprecated. It will be removed in the future and "
            "UntypedStorage will be the only storage class. This should only matter "
            "to you if you are using storages directly.  To access UntypedStorage "
            "directly, use tensor.untyped_storage() instead of tensor.storage()"
        )
        # 发出用户警告，提供自定义消息和警告类型，设置调用堆栈级别
        warnings.warn(message, UserWarning, stacklevel=stacklevel + 1)
        # 设置 _warn_typed_storage_removal 对象的 "has_warned" 属性为 True，表示已经警告过
        _warn_typed_storage_removal.__dict__["has_warned"] = True
# 重置警告类型存储移除的状态，将 has_warned 属性设置为 False
def _reset_warn_typed_storage_removal():
    _warn_typed_storage_removal.__dict__["has_warned"] = False

# 根据模块名称获取设备信息
def _get_device_from_module(module: str):
    # 从模块名称中提取最后一部分
    last_part = module.rsplit(".", 1)[-1]
    # 检查最后一部分是否为 "cuda"、torch._C._get_privateuse1_backend_name() 或 "hpu"，是则返回该部分作为设备名
    if last_part in ["cuda", torch._C._get_privateuse1_backend_name(), "hpu"]:
        return last_part
    else:
        # 否则返回 "cpu" 作为设备名
        return "cpu"

# 定义 TypedStorage 类
class TypedStorage:
    # 设置属性 is_sparse，默认为 False
    is_sparse: _bool = False

    # 声明属性 dtype，类型为 torch.dtype
    dtype: torch.dtype

    # 定义 _dtype 属性，返回当前实例的 dtype
    @property
    def _dtype(self):
        return self.dtype

    # 定义 filename 属性，返回与存储关联的文件名，如果存储不是从文件内存映射而来则返回 None
    @property
    def filename(self) -> _Optional[str]:
        """Returns the file name associated with this storage if the storage was memory mapped from a file.
        or ``None`` if the storage was not created by memory mapping a file."""
        return self._untyped_storage.filename

    # 填充当前存储的所有元素为给定的值 value
    def fill_(self, value):
        # 调用警告类型存储移除的函数
        _warn_typed_storage_removal()
        # 使用 slice(0, self._size()) 将存储的所有元素设置为 value
        self._setitem(slice(0, self._size()), value)
        # 返回当前实例自身
        return self

    # 定义 TypedStorage 类的构造函数
    def __new__(
        cls,
        *args,
        wrap_storage=None,
        dtype=None,
        device=None,
        _internal=False,
        ):
            # 如果 _internal 参数为 False，则发出警告提示移除类型存储
            if not _internal:
                _warn_typed_storage_removal()

            # 如果 cls 是 torch.storage._LegacyStorage 类型，则抛出运行时错误
            if cls == torch.storage._LegacyStorage:
                raise RuntimeError(
                    "Only child classes of _LegacyStorage can be instantiated"
                )

            # 如果 cls 是 TypedStorage 类型，则调用父类的 __new__ 方法创建实例
            if cls == TypedStorage:
                return super().__new__(cls)

            else:
                # 构建参数错误消息，提示有效的参数组合
                arg_error_msg = (
                    f"{cls}.__new__ received an invalid combination "
                    f"of arguments. Expected one of:\n"
                    " * no arguments\n"
                    " * (int size)\n"
                    " * (Sequence data)\n"
                    " * (*, UntypedStorage wrap_storage)"
                )

                # 如果指定了 device 参数，则抛出运行时错误
                if device is not None:
                    raise RuntimeError(
                        arg_error_msg + "\nKeyword argument 'device' cannot be specified"
                    )

                # 如果指定了 dtype 参数，则抛出运行时错误
                if dtype is not None:
                    raise RuntimeError(
                        arg_error_msg + "\nKeyword argument 'dtype' cannot be specified"
                    )

                # 如果 wrap_storage 参数为 None
                if wrap_storage is None:
                    # 如果有超过一个位置参数，则抛出运行时错误
                    if len(args) > 1:
                        raise RuntimeError(
                            arg_error_msg + "\nToo many positional arguments"
                        )

                    # 如果有一个位置参数且不是整数也不是序列，则抛出类型错误
                    if (
                        len(args) == 1
                        and not _isint(args[0])
                        and not isinstance(args[0], collections.abc.Sequence)
                    ):
                        raise TypeError(
                            arg_error_msg
                            + f"\nArgument type not recognized: {type(args[0])}"
                        )

                    # 返回一个 TypedStorage 实例，使用默认的 dtype 和从模块获取的 device
                    return TypedStorage(
                        *args,
                        dtype=cls._dtype,
                        device=_get_device_from_module(cls.__module__),
                        _internal=True,
                    )

                else:
                    # 如果存在位置参数，则抛出运行时错误
                    if len(args) != 0:
                        raise RuntimeError(
                            arg_error_msg
                            + "\nNo positional arguments should be given when using "
                            "'wrap_storage'"
                        )

                    # 如果 wrap_storage 不是 torch.UntypedStorage 类型，则抛出类型错误
                    if not isinstance(wrap_storage, torch.UntypedStorage):
                        raise TypeError(
                            arg_error_msg
                            + f"\nArgument 'wrap_storage' must be UntypedStorage, but got {type(wrap_storage)}"
                        )

                    # 获取当前类的设备类型
                    cls_device = _get_device_from_module(cls.__module__)

                    # 如果 wrap_storage 的设备类型不符合当前类的设备类型，则抛出运行时错误
                    if wrap_storage.device.type != cls_device:
                        raise RuntimeError(
                            arg_error_msg
                            + f"\nDevice of 'wrap_storage' must be {cls_device}"
                            f", but got {wrap_storage.device.type}"
                        )

                    # 返回一个 TypedStorage 实例，使用 wrap_storage 和当前类的 dtype
                    return TypedStorage(
                        *args,
                        wrap_storage=wrap_storage,
                        dtype=cls.dtype,
                        _internal=True,
                    )
    # 初始化方法，用于创建对象实例
    def __init__(
        self,
        *args,
        device=None,
        dtype=None,
        wrap_storage=None,
        _internal=False,
    ):
    
    # 返回当前对象是否在 CUDA 设备上
    @property
    def is_cuda(self):
        # 警告：即将移除对 TypedStorage 的支持
        _warn_typed_storage_removal()
        return self._untyped_storage.device.type == "cuda"

    # 返回当前对象是否在 HPU 设备上
    @property
    def is_hpu(self):
        # 警告：即将移除对 TypedStorage 的支持
        _warn_typed_storage_removal()
        return self._untyped_storage.device.type == "hpu"

    # 返回内部的 UntypedStorage 对象
    def untyped(self):
        """Return the internal :class:`torch.UntypedStorage`."""
        # 警告：即将移除对 TypedStorage 的支持
        _warn_typed_storage_removal()
        return self._untyped_storage

    # 根据给定的 UntypedStorage 创建一个新的包装后的存储对象
    def _new_wrapped_storage(self, untyped_storage):
        assert type(untyped_storage) == torch.UntypedStorage

        if type(self) == TypedStorage:
            # 如果当前对象是 TypedStorage 类型，则创建一个相同类型的新对象
            return TypedStorage(
                wrap_storage=untyped_storage, dtype=self.dtype, _internal=True
            )
        else:
            # 否则创建一个与当前对象相同类型的新对象
            return type(self)(wrap_storage=untyped_storage)

    # 返回当前对象的长度，即存储大小
    def __len__(self):
        # 警告：即将移除对 TypedStorage 的支持
        _warn_typed_storage_removal()
        return self._size()

    # 根据索引 idx 可能返回其对应的位置或大小，用于切片操作
    def _maybe_wrap_index(self, idx, is_stop=False):
        if idx is None:
            if is_stop:
                # 如果 idx 是 None 并且是用于切片的停止位置，返回存储的大小
                return self._size()
            else:
                # 否则返回 0
                return 0

        else:
            if type(idx) != int:
                # 如果索引不是整数类型，则抛出类型错误
                raise TypeError(f"can't index a {type(self)} with {type(idx)}")
            if is_stop:
                # 如果是用于切片的停止位置
                if (idx > self._size()) or (idx < -self._size()):
                    # 如果索引超出存储的范围，则抛出索引错误
                    raise IndexError(
                        f"index {idx} out of range for storage of size {self.size()}"
                    )
                if idx > 0:
                    return idx
                else:
                    # 处理负数索引，返回对应的正索引
                    return idx % self._size()
            else:
                # 如果是普通索引
                if (idx >= self._size()) or (idx < -self._size()):
                    # 如果索引超出存储的范围，则抛出索引错误
                    raise IndexError(
                        f"index {idx} out of range for storage of size {self.size()}"
                    )
                return idx % self._size()

    # 设置索引处的值
    def __setitem__(self, idx, value):
        # 警告：即将移除对 TypedStorage 的支持
        _warn_typed_storage_removal()
        return self._setitem(idx, value)
    # 定义私有方法 _setitem，用于设置指定索引或切片的元素值
    def _setitem(self, idx, value):
        # 如果索引不是整数或切片对象，则抛出运行时错误
        if not isinstance(idx, (int, slice)):
            raise RuntimeError(f"can't index a {type(self)} with {type(idx)}")
        # 如果值是 Torch 的存储对象，则抛出运行时错误
        if torch.is_storage(value):
            raise RuntimeError(f"cannot set item with value type {type(value)}")
        
        # 如果当前张量数据类型属于特定的量化数据类型
        if self.dtype in [
            torch.quint8,
            torch.quint4x2,
            torch.quint2x4,
            torch.qint32,
            torch.qint8,
        ]:
            # 定义不同量化类型到 Torch 标准数据类型的映射
            interpret_dtypes = {
                torch.quint8: torch.uint8,
                torch.quint4x2: torch.uint8,
                torch.quint2x4: torch.uint8,
                torch.qint32: torch.int32,
                torch.qint8: torch.int8,
            }
            # 选择当前张量的数据类型对应的 Torch 标准数据类型
            tmp_dtype = interpret_dtypes[self.dtype]
            # 创建一个空张量，指定数据类型和设备
            tmp_tensor = torch.tensor(
                [], dtype=tmp_dtype, device=self._untyped_storage.device
            )
            # 使用 TypedStorage 类来包装未命名存储，使用选择的数据类型
            tmp_tensor.set_(
                TypedStorage(
                    wrap_storage=self._untyped_storage, dtype=tmp_dtype, _internal=True
                )
            )
        else:
            # 创建一个空张量，指定当前对象的数据类型和设备
            tmp_tensor = torch.tensor(
                [], dtype=self.dtype, device=self._untyped_storage.device
            ).set_(self)
        
        # 将值赋给 tmp_tensor 的指定索引或切片
        tmp_tensor[idx] = value

    # 定义特殊方法 __getitem__，用于获取指定索引或切片的元素值
    def __getitem__(self, idx):
        # 警告：即将移除的 TypedStorage 操作
        _warn_typed_storage_removal()
        # 调用内部方法 _getitem 处理索引或切片操作并返回结果
        return self._getitem(idx)
    # 获取指定索引处的元素
    def _getitem(self, idx):
        # 检查存储设备是否为 "meta" 类型，如果是则抛出未实现错误
        if self._untyped_storage.device.type == "meta":
            raise NotImplementedError("Not available for 'meta' device type")

        # NOTE: 在 TypedStorage 存在之前，对 <type>Storage 对象进行切片索引是可能的。
        # 然而，在 TypedStorage 中实现这种行为将会非常麻烦，因此被禁用了。
        if isinstance(idx, slice):
            raise RuntimeError(
                "slices are only supported in UntypedStorage.__getitem__"
            )
        # 检查索引是否为整数类型，如果不是则抛出运行时错误
        elif not isinstance(idx, int):
            raise RuntimeError(f"can't index a {type(self)} with {type(idx)}")

        # 如果数据类型在以下列表中，需要进行类型解释并返回对应的 TypedStorage 对象的索引值
        if self.dtype in [
            torch.quint8,
            torch.quint4x2,
            torch.quint2x4,
            torch.qint32,
            torch.qint8,
        ]:
            interpret_dtypes = {
                torch.quint8: torch.uint8,
                torch.quint4x2: torch.uint8,
                torch.quint2x4: torch.uint8,
                torch.qint32: torch.int32,
                torch.qint8: torch.int8,
            }
            return TypedStorage(
                wrap_storage=self._untyped_storage,
                dtype=interpret_dtypes[self.dtype],
                _internal=True,
            )._getitem(idx)

        # 对索引进行包装处理，然后创建临时张量 tmp_tensor
        idx_wrapped = self._maybe_wrap_index(idx)
        # 临时关闭虚拟张量，以便创建真实的张量
        from torch._subclasses.fake_tensor import unset_fake_temporarily
        with unset_fake_temporarily():
            # 使用指定 dtype 和设备创建空张量 tmp_tensor，并用当前对象填充
            tmp_tensor = torch.tensor(
                [], dtype=self.dtype, device=self._untyped_storage.device
            ).set_(self)
            # 返回索引处的元素值
            return tmp_tensor[idx_wrapped].item()

    # 复制函数，用于将源数据复制到当前 TypedStorage 对象中
    def copy_(self, source: T, non_blocking: _Optional[bool] = None):
        _warn_typed_storage_removal()
        # 如果源数据是 TypedStorage 类型，则调用其 _untyped_storage 的复制函数
        if isinstance(source, TypedStorage):
            self._untyped_storage.copy_(source._untyped_storage, non_blocking)  # type: ignore[arg-type]
        else:
            # 否则直接复制源数据到当前 TypedStorage 对象中
            self._untyped_storage.copy_(source, non_blocking)  # type: ignore[arg-type]
        # 返回当前对象本身
        return self

    # 返回当前 TypedStorage 对象的字节大小
    def nbytes(self):
        _warn_typed_storage_removal()
        return self._nbytes()

    # 仅供内部使用，用于避免弃用警告
    # 返回当前 TypedStorage 对象的字节大小
    def _nbytes(self):
        return self._untyped_storage.nbytes()

    # 返回当前 TypedStorage 对象的数据类型
    def type(
        self,
        dtype: _Optional[str] = None,
        non_blocking: bool = False,
    ) -> Union[T, str]:
        _warn_typed_storage_removal()
        # 如果未指定 dtype，则返回遗留存储类的模块和类名或当前对象的模块和类名
        if dtype is None:
            legacy_class = self._get_legacy_storage_class()

            if legacy_class is not None:
                return legacy_class.__module__ + "." + legacy_class.__name__

            return ".".join([self.__module__, type(self).__name__])

        else:
            # 否则，返回根据指定 dtype 获取的 _untyped_storage 的类型
            return self._untyped_storage.type(dtype, non_blocking)
    # 将存储器从 CPU 迁移到 CUDA 设备
    def cuda(self, device=None, non_blocking=False) -> T:  # type: ignore[misc,type-var]
        _warn_typed_storage_removal()
        # 如果数据类型是量化类型，则抛出异常，CUDA 不支持量化类型
        if self.dtype in [
            torch.quint8,
            torch.quint4x2,
            torch.quint2x4,
            torch.qint32,
            torch.qint8,
        ]:
            raise RuntimeError("Cannot create CUDA storage with quantized dtype")
        # 将未命名的存储器移到 CUDA 设备
        cuda_storage: torch.UntypedStorage = self._untyped_storage.cuda(
            device, non_blocking
        )
        # 使用新的封装存储器创建新的对象并返回
        return self._new_wrapped_storage(cuda_storage)

    # 将存储器从 CPU 迁移到 HPU 设备
    def hpu(self, device=None, non_blocking=False) -> T:  # type: ignore[misc,type-var]
        _warn_typed_storage_removal()
        # 如果数据类型是量化类型，则抛出异常，HPU 不支持量化类型
        if self.dtype in [
            torch.quint8,
            torch.quint4x2,
            torch.quint2x4,
            torch.qint32,
            torch.qint8,
        ]:
            raise RuntimeError("Cannot create HPU storage with quantized dtype")
        # 将未命名的存储器移到 HPU 设备
        hpu_storage: torch.UntypedStorage = self._untyped_storage.hpu(
            device, non_blocking
        )
        # 使用新的封装存储器创建新的对象并返回
        return self._new_wrapped_storage(hpu_storage)

    # 将存储器移动到指定设备
    def to(self, *, device: torch.device, non_blocking: bool = False) -> T:  # type: ignore[type-var, misc]
        _warn_typed_storage_removal()
        # 如果数据类型是量化类型，则抛出异常，不支持创建指定设备的量化类型存储器
        if self.dtype in [
            torch.quint8,
            torch.quint4x2,
            torch.quint2x4,
            torch.qint32,
            torch.qint8,
        ]:
            raise RuntimeError(
                f"Cannot create {device.type.upper()} storage with quantized dtype"
            )
        # 将未命名的存储器移动到指定设备
        to_storage: torch.UntypedStorage = self._untyped_storage.to(
            device=device, non_blocking=non_blocking
        )
        # 使用新的封装存储器创建新的对象并返回
        return self._new_wrapped_storage(to_storage)

    # 返回存储单元的元素大小
    def element_size(self):
        _warn_typed_storage_removal()
        return self._element_size()

    # 返回存储单元的元素大小（内部方法）
    def _element_size(self):
        return torch._utils._element_size(self.dtype)

    # 获取存储器所在的设备
    def get_device(self) -> _int:
        _warn_typed_storage_removal()
        return self._untyped_storage.get_device()

    # 返回存储器的字符串表示形式
    def __str__(self):
        _warn_typed_storage_removal()
        # 构造存储器的信息字符串，包括数据类型、设备和大小
        info_str = (
            f"[{torch.typename(self)}(dtype={self.dtype}, "
            f"device={self.device}) of size {len(self)}]"
        )
        # 如果设备类型是 "meta"，则返回省略的信息字符串
        if self.device.type == "meta":
            return "...\n" + info_str
        else:
            # 否则返回包含数据的字符串形式和存储器信息的完整字符串
            data_str = " " + "\n ".join(str(self[i]) for i in range(self.size()))
            return data_str + "\n" + info_str

    # 返回存储器的字符串表示形式（调用 __str__ 方法）
    def __repr__(self):
        _warn_typed_storage_removal()
        return str(self)

    # 返回存储器的迭代器
    def __iter__(self):
        _warn_typed_storage_removal()
        return iter(self[i] for i in range(self.size()))

    # 返回存储器的浅复制
    def __copy__(self):
        _warn_typed_storage_removal()
        return self._new_wrapped_storage(copy.copy(self._untyped_storage))

    # 返回存储器的深复制
    def __deepcopy__(self, memo):
        _warn_typed_storage_removal()
        return self._deepcopy(memo)
    # 内部使用，用于避免弃用警告
    def _deepcopy(self, memo):
        # 深度复制未类型化存储，并创建新的包装存储
        return self._new_wrapped_storage(copy.deepcopy(self._untyped_storage, memo))

    def __sizeof__(self):
        # 发出类型化存储移除警告
        _warn_typed_storage_removal()
        # 返回父类大小加上存储字节数
        return super().__sizeof__() + self.nbytes()

    def clone(self):
        """返回此存储的副本。"""
        _warn_typed_storage_removal()
        # 返回新的包装存储，其包含未类型化存储的克隆
        return self._new_wrapped_storage(self._untyped_storage.clone())

    def tolist(self):
        """返回包含此存储元素的列表。"""
        _warn_typed_storage_removal()
        # 返回此存储的元素转换为列表的结果
        return list(self)

    def cpu(self):
        """如果未在 CPU 上，则返回此存储的 CPU 拷贝。"""
        _warn_typed_storage_removal()
        # 返回未类型化存储的 CPU 拷贝，如果已在 CPU 上则直接返回
        return self._new_wrapped_storage(self._untyped_storage.cpu())

    def is_pinned(self, device: Union[str, torch.device] = "cuda"):
        r"""确定 CPU 类型化存储是否已经在设备上固定。"""
        _warn_typed_storage_removal()
        # 返回未类型化存储是否已在指定设备上固定的布尔值
        return self._untyped_storage.is_pinned(device)

    def pin_memory(self, device: Union[str, torch.device] = "cuda"):
        r"""将 CPU 类型化存储复制到固定内存，如果尚未固定。"""
        _warn_typed_storage_removal()
        # 返回将未类型化存储复制到指定设备固定内存后的新包装存储
        return self._new_wrapped_storage(
            self._untyped_storage.pin_memory(device=device)
        )

    def share_memory_(self):
        """参见：:meth:`torch.UntypedStorage.share_memory_`"""
        _warn_typed_storage_removal()
        # 共享此存储到共享内存
        return self._share_memory_()

    # 内部使用，用于避免弃用警告
    def _share_memory_(self):
        # 共享未类型化存储到共享内存
        self._untyped_storage.share_memory_()
        return self

    def _new_shared(self, size, *, device=None):
        """使用相同数据类型在共享内存中创建新的存储。"""
        if device is None:
            device = "cpu"
        device = torch.device(device)
        # 使用未类型化存储创建新的共享内存存储
        untyped_storage = torch.UntypedStorage._new_shared(
            size * self._element_size(), device=device
        )
        return TypedStorage(
            wrap_storage=untyped_storage, dtype=self.dtype, _internal=True
        )

    @property
    def _cdata(self):
        # 返回未类型化存储的 C 数据指针
        return self._untyped_storage._cdata

    @property
    def device(self):
        _warn_typed_storage_removal()
        # 返回未类型化存储所在的设备
        return self._untyped_storage.device

    def size(self):
        _warn_typed_storage_removal()
        # 返回此存储的大小
        return self._size()

    # 内部使用，用于避免弃用警告
    # 返回未类型化存储的字节大小除以元素大小的结果，用于计算存储空间大小
    def _size(self):
        # 注意：避免通过 __len__ 间接调用，因为它要求返回一个整数
        return self._untyped_storage.nbytes() // self._element_size()

    # 返回序列化存储类型，但已被标记为移除警告
    def pickle_storage_type(self):
        _warn_typed_storage_removal()
        return self._pickle_storage_type()

    # 仅供内部使用，以避免弃用警告，返回序列化存储类型
    def _pickle_storage_type(self):
        try:
            return _dtype_to_storage_type_map()[self.dtype]
        except KeyError as e:
            raise KeyError(f"dtype {self.dtype} is not recognized") from e

    # 为了序列化对象，将其保存为字节流并返回用于重建对象的元组
    def __reduce__(self):
        b = io.BytesIO()
        torch.save(self, b, _use_new_zipfile_serialization=False)
        return (_load_from_bytes, (b.getvalue(),))

    # 返回数据指针，已被标记为移除警告
    def data_ptr(self):
        _warn_typed_storage_removal()
        return self._data_ptr()

    # 仅供内部使用，以避免弃用警告，返回数据指针
    def _data_ptr(self):
        return self._untyped_storage.data_ptr()

    # 返回是否可调整大小的状态，已被标记为移除警告
    def resizable(self):
        _warn_typed_storage_removal()
        return self._untyped_storage.resizable()

    # 调整存储大小，已被标记为移除警告
    def resize_(self, size):
        _warn_typed_storage_removal()
        self._resize_(size)

    # 仅供内部使用，以避免弃用警告，调整存储大小
    def _resize_(self, size):
        self._untyped_storage.resize_(size * self._element_size())

    # 类方法，释放弱引用，返回无类型化存储的方法结果
    @classmethod
    def _free_weak_ref(cls, *args, **kwargs):
        return UntypedStorage._free_weak_ref(*args, **kwargs)

    # 返回弱引用，仅供内部使用
    def _weak_ref(self, *args, **kwargs):
        return self._untyped_storage._weak_ref(*args, **kwargs)

    # 类方法，从缓冲区创建对象，已被标记为移除警告
    @classmethod
    def from_buffer(cls, *args, **kwargs):
        _warn_typed_storage_removal()
        return cls._from_buffer(*args, **kwargs)
    def _from_buffer(cls, *args, dtype=None, device=None, **kwargs):
        # 如果 cls 是 TypedStorage 类型，则确定 dtype 和 device 的数值
        if cls == TypedStorage:
            dtype = torch.get_default_dtype() if dtype is None else dtype
            device = torch.device("cpu" if device is None else device)
            # 如果 device 不是 "cpu"，则抛出运行时错误
            if device.type != "cpu":
                raise RuntimeError(
                    f"TypedStorage.from_buffer: Not available for device {device.type}"
                )
            # 使用 torch.UntypedStorage.from_buffer 创建未类型化的存储 untyped_storage
            untyped_storage: torch.UntypedStorage = torch.UntypedStorage.from_buffer(
                *args, dtype=dtype, **kwargs
            )

        else:
            # 如果 dtype 不为 None 或者 args 长度为 5，则抛出运行时错误
            if dtype is not None or len(args) == 5:
                raise RuntimeError(
                    "from_buffer: 'dtype' can only be specified in "
                    "UntypedStorage.from_buffer and TypedStorage.from_buffer"
                )
            # 如果 device 不为 None，则抛出运行时错误
            if device is not None:
                raise RuntimeError(
                    "from_buffer: 'device' can only be specified in "
                    "UntypedStorage.from_buffer and TypedStorage.from_buffer"
                )

            # 使用 torch.UntypedStorage.from_buffer 创建未类型化的存储 untyped_storage
            dtype = cls._dtype
            untyped_storage = torch.UntypedStorage.from_buffer(
                *args, dtype=dtype, **kwargs
            )

        # 返回一个 TypedStorage 对象，包装未类型化的存储 untyped_storage
        return TypedStorage(wrap_storage=untyped_storage, dtype=dtype, _internal=True)

    def _to(self, dtype):
        # 如果 dtype 不是 torch.dtype 类型，则抛出类型错误
        if not isinstance(dtype, torch.dtype):
            raise TypeError(f"Argument 'dtype' must be torch.dtype, not {type(dtype)}")
        
        # 创建一个空的 torch.tensor，使用当前对象的 dtype 和 device
        # 使用 set_ 方法将当前对象的数据设置到新建的 tensor 上
        # 使用 to 方法将数据类型转换为指定的 dtype
        # 最后调用 _typed_storage 方法获取对应的存储
        storage = (
            torch.tensor([], dtype=self.dtype, device=self.device)
            .set_(self)
            .to(dtype)
            ._typed_storage()
        )
        
        # 如果新建的存储与当前存储的数据指针相同，则进行克隆操作
        if storage.data_ptr() == self.data_ptr():
            storage = storage.clone()
        
        # 返回转换后的存储对象
        return storage

    def double(self):
        """将存储对象转换为双精度类型。"""
        # 警告即将移除 TypedStorage 的使用
        _warn_typed_storage_removal()
        return self._to(torch.double)

    def float(self):
        """将存储对象转换为单精度类型。"""
        # 警告即将移除 TypedStorage 的使用
        _warn_typed_storage_removal()
        return self._to(torch.float)

    def half(self):
        """将存储对象转换为半精度类型。"""
        # 警告即将移除 TypedStorage 的使用
        _warn_typed_storage_removal()
        return self._to(torch.half)

    def long(self):
        """将存储对象转换为长整型类型。"""
        # 警告即将移除 TypedStorage 的使用
        _warn_typed_storage_removal()
        return self._to(torch.long)

    def int(self):
        """将存储对象转换为整型类型。"""
        # 警告即将移除 TypedStorage 的使用
        _warn_typed_storage_removal()
        return self._to(torch.int)

    def short(self):
        """将存储对象转换为短整型类型。"""
        # 警告即将移除 TypedStorage 的使用
        _warn_typed_storage_removal()
        return self._to(torch.short)

    def char(self):
        """将存储对象转换为字符类型。"""
        # 警告即将移除 TypedStorage 的使用
        _warn_typed_storage_removal()
        return self._to(torch.int8)

    def byte(self):
        """将存储对象转换为字节类型。"""
        # 警告即将移除 TypedStorage 的使用
        _warn_typed_storage_removal()
        return self._to(torch.uint8)
    def bool(self):
        """Casts this storage to bool type."""
        # 发出警告，标明类型化存储即将移除
        _warn_typed_storage_removal()
        # 将当前存储转换为布尔类型
        return self._to(torch.bool)

    def bfloat16(self):
        """Casts this storage to bfloat16 type."""
        # 发出警告，标明类型化存储即将移除
        _warn_typed_storage_removal()
        # 将当前存储转换为bfloat16类型
        return self._to(torch.bfloat16)

    def complex_double(self):
        """Casts this storage to complex double type."""
        # 发出警告，标明类型化存储即将移除
        _warn_typed_storage_removal()
        # 将当前存储转换为复数双精度类型
        return self._to(torch.cdouble)

    def complex_float(self):
        """Casts this storage to complex float type."""
        # 发出警告，标明类型化存储即将移除
        _warn_typed_storage_removal()
        # 将当前存储转换为复数单精度类型
        return self._to(torch.cfloat)

    def float8_e5m2(self):
        """Casts this storage to float8_e5m2 type"""
        # 发出警告，标明类型化存储即将移除
        _warn_typed_storage_removal()
        # 将当前存储转换为float8_e5m2类型
        return self._to(torch.float8_e5m2)

    def float8_e4m3fn(self):
        """Casts this storage to float8_e4m3fn type"""
        # 发出警告，标明类型化存储即将移除
        _warn_typed_storage_removal()
        # 将当前存储转换为float8_e4m3fn类型
        return self._to(torch.float8_e4m3fn)

    def float8_e5m2fnuz(self):
        """Casts this storage to float8_e5m2fnuz type"""
        # 发出警告，标明类型化存储即将移除
        _warn_typed_storage_removal()
        # 将当前存储转换为float8_e5m2fnuz类型
        return self._to(torch.float8_e5m2fnuz)

    def float8_e4m3fnuz(self):
        """Casts this storage to float8_e4m3fnuz type"""
        # 发出警告，标明类型化存储即将移除
        _warn_typed_storage_removal()
        # 将当前存储转换为float8_e4m3fnuz类型
        return self._to(torch.float8_e4m3fnuz)

    @classmethod
    def from_file(cls, filename, shared, size):
        """from_file(filename, shared=False, size=0) -> Storage

        Creates a CPU storage backed by a memory-mapped file.

        If ``shared`` is ``True``, then memory is shared between all processes.
        All changes are written to the file. If ``shared`` is ``False``, then the changes on
        the storage do not affect the file.

        ``size`` is the number of elements in the storage. If ``shared`` is ``False``,
        then the file must contain at least ``size * sizeof(Type)`` bytes
        (``Type`` is the type of storage). If ``shared`` is ``True`` the file will be created if needed.

        Args:
            filename (str): file name to map
            shared (bool): whether to share memory (whether ``MAP_SHARED`` or ``MAP_PRIVATE`` is passed to the
                            underlying `mmap(2) call <https://man7.org/linux/man-pages/man2/mmap.2.html>`_)
            size (int): number of elements in the storage
        """
        # 发出警告，标明类型化存储即将移除
        _warn_typed_storage_removal()
        # 如果当前类是TypedStorage，则抛出运行时错误
        if cls == TypedStorage:
            raise RuntimeError("from_file can only be called on derived classes")
        # 从文件创建未类型化存储对象
        untyped_storage: UntypedStorage = UntypedStorage.from_file(
            filename, shared, size * torch._utils._element_size(cls.dtype)
        )
        # 使用未类型化存储对象创建当前类的存储对象
        storage = cls(wrap_storage=untyped_storage)
        return storage

    @classmethod
    def _expired(cls, *args, **kwargs):
        # 调用未类型化存储的_expired方法
        return UntypedStorage._expired(*args, **kwargs)

    def _write_file(self, *args, **kwargs):
        # 调用未类型化存储的_write_file方法
        return self._untyped_storage._write_file(*args, **kwargs)
    # 将文件操作委托给未类型化存储对象来设置数据
    def _set_from_file(self, *args, **kwargs):
        return self._untyped_storage._set_from_file(*args, **kwargs)

    # 将 C 数据操作委托给未类型化存储对象来设置数据
    def _set_cdata(self, *args, **kwargs):
        return self._untyped_storage._set_cdata(*args, **kwargs)

    # 在 CUDA 上共享数据，委托给未类型化存储对象
    def _share_cuda_(self, *args, **kwargs):
        return self._untyped_storage._share_cuda_(*args, **kwargs)

    # 检查当前对象是否是共享的，同时发出类型化存储移除警告
    def is_shared(self):
        _warn_typed_storage_removal()
        return self._is_shared()

    # 内部使用方法，避免产生过时警告，检查当前对象是否是共享的
    def _is_shared(self):
        return self._untyped_storage.is_shared()

    # 创建一个新的共享 CUDA 存储对象，作为类方法
    @classmethod
    def _new_shared_cuda(cls, *args, **kwargs):
        return torch.UntypedStorage._new_shared_cuda(*args, **kwargs)

    # 在 CPU 上通过文件名共享数据，返回管理器句柄、存储句柄和数据大小
    def _share_filename_cpu_(self, *args, **kwargs):
        (
            manager_handle,
            storage_handle,
            size,
        ) = self._untyped_storage._share_filename_cpu_(*args, **kwargs)
        return manager_handle, storage_handle, size // self._element_size()

    # 减少共享引用计数
    def _shared_decref(self):
        self._untyped_storage._shared_decref()
        return self

    # 释放 IPC 计数器，针对 CUDA 的类方法
    @classmethod
    def _release_ipc_counter(cls, *args, device=None, **kwargs):
        return torch.UntypedStorage._release_ipc_counter_cuda(*args, **kwargs)

    # 增加共享引用计数
    def _shared_incref(self, *args, **kwargs):
        return self._untyped_storage._shared_incref(*args, **kwargs)

    # 在 CPU 上通过文件描述符共享数据，返回文件描述符和数据大小
    def _share_fd_cpu_(self, *args, **kwargs):
        fd, size = self._untyped_storage._share_fd_cpu_(*args, **kwargs)
        return fd, size // self._element_size()

    # 获取旧版存储类，基于当前对象的数据类型和设备类型
    def _get_legacy_storage_class(self):
        # 如果数据类型不在映射表中，则返回 None
        if self.dtype not in _dtype_to_storage_type_map():
            return None

        # 获取存储类型名称
        storage_name = _dtype_to_storage_type_map()[self.dtype]

        # 如果设备类型不在支持的类型列表中，则返回 None
        if self.device.type not in [
            "cpu",
            "cuda",
            "hpu",
            torch._C._get_privateuse1_backend_name(),
        ]:
            return None

        # 根据设备类型获取对应的模块对象
        module = (
            torch if self.device.type == "cpu" else getattr(torch, self.device.type)
        )

        try:
            # 返回与存储类型名称对应的存储类对象
            return getattr(module, storage_name)
        except AttributeError:
            return None
# 将 TypedStorage 类的 type 属性的文档字符串设置为 _type 的文档字符串
TypedStorage.type.__doc__ = _type.__doc__
# 将 TypedStorage 类的 cuda 属性的文档字符串设置为 _StorageBase 类的 cuda 方法的文档字符串
TypedStorage.cuda.__doc__ = _StorageBase.cuda.__doc__
# 将 TypedStorage 类的 hpu 属性的文档字符串设置为 _StorageBase 类的 hpu 方法的文档字符串
TypedStorage.hpu.__doc__ = _StorageBase.hpu.__doc__
# 将 TypedStorage 类的 to 属性的文档字符串设置为 _to 函数的文档字符串
TypedStorage.to.__doc__ = _to.__doc__


class _LegacyStorageMeta(type):
    dtype: torch.dtype

    def __instancecheck__(cls, instance):
        # 检查 instance 是否为 TypedStorage 的实例
        if type(instance) == TypedStorage:
            # 获取 instance 的设备类型
            cls_device = _get_device_from_module(cls.__module__)
            # 返回判断结果，比较实例的设备类型和类的设备类型，以及实例的数据类型和类的数据类型
            return (cls_device == instance.device.type) and (
                cls.dtype == instance.dtype
            )
        # 如果 instance 不是 TypedStorage 的实例，则返回 False
        return False


class _LegacyStorage(TypedStorage, metaclass=_LegacyStorageMeta):
    @classmethod
    def _new_shared(cls, size):
        """Create a new storage in shared memory with the same data type."""
        # 使用未类型化存储类创建一个在共享内存中的新存储，其数据类型与当前类的相同
        untyped_storage = torch.UntypedStorage._new_shared(size * cls()._element_size())
        # 返回使用当前类包装的存储对象
        return cls(wrap_storage=untyped_storage)

    @classmethod
    def _release_ipc_counter(cls, *args, **kwargs):
        # 调用 torch.UntypedStorage._release_ipc_counter_cuda 方法并返回其结果
        return torch.UntypedStorage._release_ipc_counter_cuda(*args, **kwargs)

    @classmethod
    def _new_shared_filename(cls, manager, obj, size):
        # 计算字节大小
        bytes_size = size * torch._utils._element_size(cls.dtype)
        # 使用文件名在 CPU 上创建一个新的未类型化存储，其数据类型与当前类相匹配
        return cls(
            wrap_storage=torch.UntypedStorage._new_shared_filename_cpu(
                manager, obj, bytes_size
            )
        )


def _get_dtype_from_pickle_storage_type(pickle_storage_type: str):
    try:
        # 根据 pickle 存储类型映射返回对应的数据类型
        return _storage_type_to_dtype_map()[pickle_storage_type]
    except KeyError as e:
        # 抛出自定义异常，指示未识别的 pickle 存储类型
        raise KeyError(
            f'pickle storage type "{pickle_storage_type}" is not recognized'
        ) from e
```