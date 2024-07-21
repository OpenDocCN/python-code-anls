# `.\pytorch\torch\utils\_content_store.py`

```py
# mypy: allow-untyped-defs
# This module provides a FAST (on GPU) content addressable store for storages
# (and tensors on top of them) with VERY WEAK portability guarantees (e.g.,
# don't expect CPU/CUDA to address to the same hash, don't expect it to be
# portable across devices) that is NOT cryptographically secure.  In return,
# we are able to hash 40G of tensor data on GPU in less than a second,
# compared to running SHA-1 in CPU which would a minute or so.  The primary
# use case is for efficiently snapshotting intermediate tensor data for
# offline debugging, but it's been put in this module in case you think of
# another use case for it.  The hash function could be replaced with a
# straight reimplementation of SHA-1, which would give us much stronger
# portability guarantees.
#
# WARNING: THERE IS NO BC/FC GUARANTEE FOR THIS FORMAT!  If you need to format
# shift the result, consider packing it into a single torch.save object
# with traditional view sharing.
#
# Because of the weak portability guarantees, you can only write to the
# content store from a single process; we don't provide any capability
# of "reopening" a content store to add more things to it.  But we don't
# assume that you can keep all of the tensors you want to add to the store
# in memory at once, because you probably can't!  Nor do we assume that
# you know a priori whether or not two storages can be deduplicated or not.
#
# Note: only storages are content-addressed; tensors are name addressed
#
# Note: our padding strategy means that [1, 0] and [1] int16 tensors would
# map to the same (padded) storage.  We think this will be immaterial for most
# users.

import ctypes
import functools
import hashlib
import os.path
import struct
from collections import defaultdict
from typing import Dict, Optional, Set

import torch
import torch._prims as prims
import torch._utils
import torch.nn.functional as F
from torch._C import default_generator

from torch.multiprocessing.reductions import StorageWeakRef


def lazy_compile(**compile_kwargs):
    """Lazily wrap a function with torch.compile on the first call

    This avoids eagerly importing dynamo.
    """

    def decorate_fn(fn):
        @functools.wraps(fn)
        def compile_hook(*args, **kwargs):
            compiled_fn = torch.compile(fn, **compile_kwargs)
            globals()[fn.__name__] = functools.wraps(fn)(compiled_fn)
            return compiled_fn(*args, **kwargs)

        return compile_hook

    return decorate_fn


# Use of torch.compile is mandatory for (1) good memory usage
# and (2) xor_sum implementation.  This is our first instance of
# using PT2 to implement a kernel in PyTorch; if we get AOT capabilities
# it would be good to apply it here.
@lazy_compile(dynamic=True)
def hash_storage_kernel(x):
    # The randint calls are carefully written to hit things we
    # have lowerings for in inductor.  Lack of unsigned 32-bit integer
    # is a pain.
    # Lazy compilation of a kernel function for hashing storage data in PyTorch
    pass
    # 生成一个与 x 形状相同的随机整数张量 a，取绝对值
    a = torch.randint(
        -(2**31), 2**31, x.shape, device=x.device, dtype=torch.int32
    ).abs()
    # 对 a 取模 (2**31 - 1)，加 1，并转换为 long 类型
    a = ((a % (2**31 - 1)) + 1).long()
    # 生成一个与 x 形状相同的随机整数张量 b，取绝对值，并转换为 long 类型
    b = (
        torch.randint(-(2**31), 2**31, x.shape, device=x.device, dtype=torch.int32)
        .abs()
        .long()
    )
    # 这是一个标准的移位乘法哈希家族加上异或和哈希，使用 Philox 生成随机数。
    # 我们的 Philox 随机数生成器在不同设备上不确定，因此不适用于稳定的哈希算法。
    #
    # 这里假设张量长度固定，因此你还需要根据张量的长度进行分桶处理。
    return prims.xor_sum((a * x + b).int(), [0])
# 返回存储数据的十六进制摘要。如果 stable_hash=True，则保证是 SHA-1，否则在单个进程运行中是一致的，但跨进程不一定。
def hash_storage(storage: torch.UntypedStorage, *, stable_hash: bool = False) -> str:
    # 导入必要的模块和函数
    import torch._dynamo
    from torch._dynamo.utils import is_compile_supported

    # 获取存储的设备类型
    device_type = storage.device.type
    # 如果 stable_hash=True 或者设备类型不支持编译，则使用 CPU 上的存储数据进行计算
    if stable_hash or not is_compile_supported(device_type):
        cpu_storage = storage.cpu()
        # TODO: 当存储支持缓冲协议时，不再需要下面的处理
        buf = (ctypes.c_byte * cpu_storage.nbytes()).from_address(
            cpu_storage.data_ptr()
        )
        # 计算 SHA-1 摘要
        sha1 = hashlib.sha1()
        sha1.update(buf)
        return sha1.hexdigest()

    # 处理随机生成器选择
    if device_type == "cpu":
        generator = default_generator
    elif device_type == "cuda":
        import torch.cuda
        generator = torch.cuda.default_generators[storage.device.index]
    else:
        raise AssertionError(f"unhandled device type {device_type}")

    # 保存当前生成器状态
    state = generator.get_state()
    try:
        generator.manual_seed(0)
        # 将存储数据设置到新创建的张量中，并进行类型转换
        x = torch.empty(0, dtype=torch.uint8, device=storage.device).set_(storage)  # type: ignore[call-overload]
        # dtype 转换视图无法编译，因此填充/重塑需要在外部完成，即使可以有利于融合
        pad = -x.numel() % 4
        if pad > 0:
            x = F.pad(x, (0, pad), "constant", 0)
        x = x.view(torch.int32)
        # 使用不同参数多次运行 32 位哈希以降低碰撞几率
        ITER = 5
        cs = [hash_storage_kernel(x).item() for _ in range(ITER)]
        # 将哈希结果打包为大端序列的整数数组，并转换为十六进制字符串
        return struct.pack(">" + "i" * ITER, *cs).hex()
    finally:
        # 恢复生成器状态
        generator.set_state(state)


class ContentStoreWriter:
    # 结构:
    #   storages/
    #     00/
    #       0000..00
    #   tensors/
    #     name
    def __init__(self, loc: str, stable_hash: bool = False) -> None:
        self.loc: str = loc
        self.seen_storage_hashes: Set[str] = set()
        self.stable_hash = stable_hash

    # TODO: 提供一些非阻塞 API 来加速操作
    def write_storage(self, storage: torch.UntypedStorage) -> str:
        # 计算存储数据的哈希值
        h = hash_storage(storage, stable_hash=self.stable_hash)
        # 如果哈希值已经存在于已见哈希集合中，则直接返回哈希值
        if h in self.seen_storage_hashes:
            return h
        # 否则，保存存储数据到文件系统中
        subfolder = os.path.join(self.loc, "storages")
        os.makedirs(subfolder, exist_ok=True)
        target = os.path.join(subfolder, h)
        # 如果目标文件已经存在，则直接返回哈希值
        if os.path.exists(target):
            return h
        torch.save(storage, target)
        # 将新的哈希值添加到已见哈希集合中
        self.seen_storage_hashes.add(h)
        return h
    # 计算张量的元数据并返回元组
    def compute_tensor_metadata(self, t: torch.Tensor, h=None):
        # 如果未提供哈希值h，则使用未命名的张量存储的哈希值
        if h is None:
            h = hash_storage(t.untyped_storage(), stable_hash=self.stable_hash)
        return (
            t.dtype,  # 张量的数据类型
            h,  # 张量的哈希值
            t.storage_offset(),  # 张量数据在存储中的偏移量
            tuple(t.shape),  # 张量的形状，转换为元组
            t.stride(),  # 张量的步幅信息
            torch._utils.get_tensor_metadata(t),  # 获取张量的元数据
        )

    # 将张量写入文件系统
    def write_tensor(self, name: str, t: torch.Tensor) -> None:
        storage = t.untyped_storage()  # 获取张量的未命名存储
        h = self.write_storage(storage)  # 写入存储并获取哈希值
        # TODO: 支持更高级的快照功能，如requires_grad/grad等
        d, f = os.path.split(name)  # 拆分路径名name，获取目录d和文件名f
        payload = self.compute_tensor_metadata(t, h=h)  # 计算张量的元数据
        subfolder = os.path.join(self.loc, "tensors", d)  # 构建子文件夹路径
        os.makedirs(subfolder, exist_ok=True)  # 确保子文件夹存在，不存在则创建
        torch.save(payload, os.path.join(subfolder, f))  # 将张量的元数据payload保存到文件系统中的指定位置
# 定义一个内容存储读取器的类
class ContentStoreReader:
    # 初始化方法，接受位置参数 loc（存储位置的路径字符串）和一个关键字参数 cache（是否缓存，默认为 True）
    def __init__(self, loc: str, *, cache=True) -> None:
        # 将位置参数 loc 赋值给实例变量 self.loc，表示存储位置
        self.loc = loc
        # 定义一个存储缓存，类型为可选的字典，键为可选的 torch.device 对象，值为存储弱引用的字典
        self.storage_cache: Optional[
            Dict[Optional[torch.device], Dict[str, StorageWeakRef]]
        ] = None
        # 如果 cache 为 True，则初始化 storage_cache 为一个默认字典的实例
        if cache:
            self.storage_cache = defaultdict(dict)

    # 读取存储内容的方法，接受参数 h（存储的标识符字符串）和一个关键字参数 device（存储的设备）
    def read_storage(self, h: str, *, device=None) -> torch.UntypedStorage:
        # 如果指定了 device，则将其转换为 torch.device 对象
        if device is not None:
            device = torch.device(device)
        # 获取设备对应的存储对象，如果存储缓存不为 None 的话
        ws = (
            self.storage_cache[device].get(h)
            if self.storage_cache is not None
            else None
        )
        # 声明一个可选的未命名的存储对象 s
        s: Optional[torch.UntypedStorage]
        # 如果 ws 不为 None，则使用其内部数据创建一个新的未命名存储对象 s
        if ws is not None:
            s = torch.UntypedStorage._new_with_weak_ptr(ws.cdata)
            if s is not None:
                return s
        # 否则，从指定位置 loc 的 "storages" 目录中加载指定标识符 h 的存储数据
        s = torch.load(
            os.path.join(self.loc, "storages", h),
            weights_only=True,
            map_location=device,
        )._untyped_storage
        # 断言确保 s 不为 None
        assert s is not None
        # 如果存储缓存不为 None，则将新加载的存储对象 s 缓存到存储缓存中
        if self.storage_cache is not None:
            self.storage_cache[device][h] = StorageWeakRef(s)
        # 返回加载的存储对象 s
        return s

    # 读取张量元数据的方法，接受参数 name（张量的名称字符串）
    def read_tensor_metadata(self, name: str):
        # 构建张量元数据文件的路径 fn
        fn = os.path.join(self.loc, "tensors", name)
        # 如果文件不存在，则抛出文件未找到异常
        if not os.path.exists(fn):
            raise FileNotFoundError(fn)
        # 返回加载的张量元数据，仅加载权重信息
        return torch.load(fn, weights_only=True)

    # 读取张量数据的方法，接受参数 name（张量的名称字符串）和一个关键字参数 device（存储的设备）
    def read_tensor(self, name: str, *, device=None) -> torch.Tensor:
        # 读取张量的数据类型、存储标识符 h、存储偏移量、大小、步长和元数据
        dtype, h, storage_offset, size, stride, metadata = self.read_tensor_metadata(
            name
        )
        # 使用 read_storage 方法加载存储对象 storage
        storage = self.read_storage(h, device=device)
        # 创建一个空张量 t，使用指定的数据类型和存储设备
        t = torch.tensor([], dtype=dtype, device=storage.device)
        # 使用存储对象、存储偏移量、大小和步长来设置张量 t 的值
        t.set_(storage, storage_offset, size, stride)
        # 设置张量 t 的元数据
        torch._utils.set_tensor_metadata(t, metadata)
        # 返回加载的张量 t
        return t
```