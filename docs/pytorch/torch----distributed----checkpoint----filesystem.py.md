# `.\pytorch\torch\distributed\checkpoint\filesystem.py`

```
# mypy: allow-untyped-defs
# 引入必要的模块和类
import collections  # 引入集合模块
import dataclasses  # 引入数据类模块
import io  # 引入输入输出流模块
import operator  # 引入操作符模块
import os  # 引入操作系统功能模块
import pickle  # 引入序列化和反序列化模块
import queue  # 引入队列模块
import threading  # 引入线程模块
import uuid  # 引入 UUID 模块
import warnings  # 引入警告模块
from abc import ABC, abstractmethod  # 从抽象基类模块中引入 ABC 和抽象方法装饰器
from contextlib import contextmanager  # 从上下文管理模块中引入上下文管理器装饰器
from dataclasses import dataclass  # 从数据类模块中引入数据类装饰器
from pathlib import Path  # 从路径模块中引入路径类
from typing import (  # 引入类型提示模块中的多种类型
    Any,
    Callable,
    cast,
    Dict,
    Generator,
    IO,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

import torch  # 引入 PyTorch 模块
from torch import Tensor  # 从 PyTorch 中引入张量类
from torch._utils import _get_available_device_type, _get_device_module  # 引入内部工具函数
from torch.distributed._shard._utils import narrow_tensor_by_index  # 引入分布式相关工具函数
from torch.distributed.checkpoint.metadata import (  # 引入分布式检查点元数据相关类
    Metadata,
    MetadataIndex,
    STATE_DICT_TYPE,
    StorageMeta,
)
from torch.distributed.checkpoint.planner import (  # 引入分布式检查点计划相关类
    LoadItemType,
    LoadPlan,
    LoadPlanner,
    ReadItem,
    SavePlan,
    SavePlanner,
    WriteItem,
    WriteItemType,
)
from torch.distributed.checkpoint.staging import BlockingAsyncStager  # 引入异步阻塞分段相关类
from torch.distributed.checkpoint.storage import (  # 引入分布式检查点存储相关类
    StorageReader,
    StorageWriter,
    WriteResult,
)
from torch.distributed.checkpoint.utils import _create_file_view  # 引入文件视图创建函数
from torch.futures import Future  # 引入 Torch 异步操作的 Future 类


__all__ = ["FileSystemWriter", "FileSystemReader", "FileSystem", "FileSystemBase"]

_metadata_fn: str = ".metadata"  # 定义元数据文件名


@dataclass
class _StorageInfo:
    """This is the per entry storage info."""
    relative_path: str  # 相对路径
    offset: int  # 偏移量
    length: int  # 长度


@dataclass
class _StoragePrefix:
    prefix: str  # 前缀字符串


DEFAULT_SUFFIX = ".distcp"  # 默认后缀名

def _generate_uuid() -> str:
    """生成并返回一个 UUID 字符串"""
    return str(uuid.uuid4())  # 生成一个新的 UUID 字符串

class _TensorLoader(ABC):
    """抽象类定义张量加载器接口"""

    @abstractmethod
    def add(self, size: int, obj: object) -> None:
        """添加张量大小和对象到加载器"""
        pass

    @abstractmethod
    def start_loading(self) -> None:
        """开始加载操作"""
        pass

    @abstractmethod
    def values(self) -> Iterator[Tuple[torch.Tensor, object]]:
        """返回张量和相关对象的迭代器"""
        pass

class _SerialCpuLoader(_TensorLoader):
    """串行 CPU 加载器，继承自 _TensorLoader"""

    def __init__(self, resolve_fun: Callable) -> None:
        """初始化方法，接受解析函数作为参数"""
        self.resolve_fun = resolve_fun
        self.items: List[Tuple[int, object]] = []  # 初始化一个空列表，用于存储元组 (大小, 对象)

    def add(self, size: int, obj: object) -> None:
        """将大小和对象添加到加载项列表"""
        self.items.append((size, obj))

    def start_loading(self) -> None:
        """开始加载操作，此处无操作"""
        pass

    def values(self) -> Iterator[Tuple[torch.Tensor, object]]:
        """返回加载项的张量和相关对象的迭代器"""
        for _, obj in self.items:
            tensor = self.resolve_fun(obj).detach()  # 解析对象并返回张量，并将其分离
            tensor = tensor.cpu()  # 将张量移到 CPU
            if tensor.storage().size() != tensor.numel():
                tensor = tensor.clone()  # 如果张量存储大小不等于元素数，克隆张量
            yield (tensor, obj)  # 返回张量和对象的生成器

class _OverlappingCpuLoader(_TensorLoader):
    """重叠 CPU 加载器，继承自 _TensorLoader"""

    def __init__(
        self,
        resolve_fun: Callable,
        stream: Optional[torch.Stream] = None,
        inflight_threshhold: int = 1_000_000,
        ```
    ) -> None:
        # 初始化函数，接受一个解析函数 resolve_fun 作为参数，初始化各种状态和数据结构
        self.resolve_fun = resolve_fun
        # 存储元素的列表，每个元素是一个元组，包含大小和对象
        self.items: List[Tuple[int, object]] = []
        # 控制并发处理的阈值，当并发处理的数据大小达到这个阈值时进行同步
        self.inflight_threshhold = inflight_threshhold
        # 当前正在处理的数据大小
        self.in_flight_data = 0
        # 当前正在处理的数据队列，使用 collections.deque 实现
        self.current_items: collections.deque = collections.deque()
        # 索引，用于追踪当前处理到 items 中的哪个位置
        self.idx = 0
        # 标志，表示加载是否已经开始
        self.started = False
        # 设备类型，如果有传入流对象则使用流的设备类型，否则获取可用设备类型
        self.device_type = (
            stream.device_type if stream else _get_available_device_type()
        )
        # 获取设备类型对应的模块
        self.device_module = _get_device_module(self.device_type)
        # 获取 CUDA 流对象，如果没有传入流对象则使用设备模块的当前流
        self.stream = cast(
            torch.cuda.Stream, stream or self.device_module.current_stream()
        )
        # 如果当前流不是设备模块的当前流，则等待当前流完成
        if self.stream != self.device_module.current_stream():
            self.stream.wait_stream(self.device_module.current_stream())

    @property
    def _done(self) -> bool:
        # 属性方法，判断是否已经处理完所有的 items
        return self.idx >= len(self.items)

    def _drain(self) -> List[Tuple[torch.Tensor, object]]:
        # 内部方法，用于从当前处理队列中将数据转移出来，直到满足条件
        drained = []
        # 如果当前正在处理的数据超过了阈值，则同步 CUDA 流
        if self.in_flight_data >= self.inflight_threshhold:
            self.stream.synchronize()
        # 反复从当前处理队列中取出数据，直到当前处理数据小于阈值为止
        while self.in_flight_data >= self.inflight_threshhold:
            val = self.current_items.popleft()
            self.in_flight_data -= val[0].numel() * val[0].element_size()
            drained.append(val)
        return drained

    def _refill(self) -> None:
        # 内部方法，用于填充当前处理队列，直到满足条件为止
        with self.device_module.stream(self.stream):
            # 反复执行直到所有 items 都被处理完或者当前处理数据达到阈值
            while not self._done and self.in_flight_data < self.inflight_threshhold:
                _, obj = self.items[self.idx]
                self.idx += 1
                # 解析对象并生成对应的 tensor
                tensor = self.resolve_fun(obj).detach()
                # 如果 tensor 的设备类型和当前流的设备类型相同，则将 tensor 转移到 CPU 上
                if tensor.device.type == self.device_type:
                    tensor = tensor.to(device="cpu", non_blocking=True)
                # 如果 tensor 的设备是 CPU，则检查其存储是否为连续存储，不是则克隆一份连续存储的 tensor
                elif tensor.device == torch.device("cpu"):
                    if (
                        tensor.untyped_storage().size()
                        != tensor.numel() * tensor.itemsize
                    ):
                        tensor = tensor.clone()

                # 将 tensor 和对应的对象添加到当前处理队列中
                self.current_items.append(
                    (
                        tensor,
                        obj,
                    )
                )
                # 更新当前正在处理的数据大小
                self.in_flight_data += tensor.numel() * tensor.element_size()

    def _finish(self) -> Iterable[Tuple[torch.Tensor, object]]:
        # 内部方法，用于完成加载过程并返回当前处理队列中的所有数据
        assert self._done  # 确保所有 items 都已处理完
        # 如果当前处理队列中还有数据，则同步 CUDA 流
        if len(self.current_items) > 0:
            self.stream.synchronize()
        # 返回当前处理队列中的所有数据
        return self.current_items

    def add(self, size: int, obj: object) -> None:
        # 外部方法，用于向 items 中添加元素，要求加载未开始
        if self.started:
            raise RuntimeError("cannot add items after loading started")
        self.items.append((size, obj))

    def start_loading(self) -> None:
        # 外部方法，用于开始加载过程
        if self.started:
            return
        self.started = True
        # 根据元素大小对 items 进行排序
        self.items.sort(key=operator.itemgetter(0))
        # 开始填充当前处理队列
        self._refill()
    # 定义一个生成器函数 values，返回类型为迭代器，每次生成一个元组，包含一个 torch.Tensor 对象和一个任意对象
    def values(self) -> Iterator[Tuple[torch.Tensor, object]]:
        # 调用对象的 start_loading 方法，开始加载数据
        self.start_loading()
        
        # 当任务未完成时执行循环
        while not self._done:
            # 调用 _drain 方法，将数据从某处取出并返回，此处为一个生成器
            drained = self._drain()
            # 调用 _refill 方法，重新填充数据源，准备下一次的数据读取
            self._refill()
            
            # 使用生成器 drained 中的数据，每次产生一个元素
            yield from drained
        
        # 执行 _finish 方法，生成并返回剩余的数据
        yield from self._finish()
# 计算 WriteItem 对象的大小，基于其 tensor_data 属性的尺寸
def _item_size(item: WriteItem) -> int:
    size = 1
    assert item.tensor_data is not None
    # 遍历 tensor_data 的尺寸，计算总大小
    for s in item.tensor_data.size:
        size *= s

    # 获取 tensor_data 的数据类型，计算数据类型的每个元素大小
    dtype = item.tensor_data.properties.dtype
    return size * torch._utils._element_size(dtype)


# 按照大小和类型分割 WriteItem 对象列表为多个桶
def _split_by_size_and_type(bins: int, items: List[WriteItem]) -> List[List[WriteItem]]:
    if bins == 1:
        return [items]

    # 根据类型分割 WriteItem 对象列表
    bytes_w = [wi for wi in items if wi.type == WriteItemType.BYTE_IO]
    tensor_w = [wi for wi in items if wi.type != WriteItemType.BYTE_IO]

    # 创建 bins 个空桶列表
    buckets: List[List[WriteItem]] = [[] for _ in range(bins)]
    bucket_sizes = [0 for _ in range(bins)]

    # 根据 tensor_data 的大小对 tensor_w 进行降序排序
    tensor_w.sort(key=_item_size, reverse=True)

    # 将 bytes_w 中的对象按照索引分配到不同的桶中
    for i, wi in enumerate(bytes_w):
        buckets[i % bins].append(wi)

    # 将 tensor_w 中的对象按照大小动态分配到不同的桶中
    for wi in tensor_w:
        # TODO: 使用堆来代替当前的分配方式
        idx = min(enumerate(bucket_sizes), key=operator.itemgetter(1))[0]
        buckets[idx].append(wi)
        bucket_sizes[idx] += _item_size(wi)

    return buckets


# 将 WriteItem 对象写入到指定流中
def _write_item(
    stream: io.IOBase,
    data: Union[io.BytesIO, torch.Tensor],
    write_item: WriteItem,
    storage_key: str,
) -> WriteResult:
    # 记录当前流的偏移量
    offset = stream.tell()

    # 根据 WriteItem 的类型选择不同的写入方式
    if write_item.type == WriteItemType.BYTE_IO:
        assert isinstance(data, io.BytesIO)
        stream.write(data.getbuffer())
    else:
        assert isinstance(data, torch.Tensor)
        assert data.device == torch.device("cpu")
        # 将 tensor 数据保存到流中
        torch.save(data, cast(IO[bytes], stream))
    
    # 计算写入数据的长度
    length = stream.tell() - offset

    # 返回写入结果对象，包含索引、数据长度和存储信息
    return WriteResult(
        index=write_item.index,
        size_in_bytes=length,
        storage_data=_StorageInfo(storage_key, offset, length),
    )


# 从队列中处理 WriteItem 对象，写入到流中
def _write_files_from_queue(
    create_stream: Callable,
    file_queue: queue.Queue,
    result_queue: queue.Queue,
    planner: SavePlanner,
    inflight_threshhold: int,
    use_fsync: bool,
    thread_count: int,
) -> None:
    # 尝试从文件队列中获取文件名、存储键、写入项目，如果队列为空则停止循环
    try:
        while True:
            file_name, storage_key, write_items = file_queue.get_nowait()
            loader: _TensorLoader  # 声明一个类型为 _TensorLoader 的 loader 对象

            # 获取私有使用的第一个后端的名称
            custom_backend_name = torch._C._get_privateuse1_backend_name()
            # 根据名称获取对应的设备模块对象
            custom_device_mod = getattr(torch, custom_backend_name, None)

            # TODO: 使用 OverlappingCpuLoader 在多线程情况下会导致显著的性能下降，
            # 主要与 CUDA 流同步有关。我们应该尝试修复这个问题，并对所有线程情况使用 _OverlappingCpuLoader
            # 如果线程数为1，并且存在 CUDA 或自定义设备模块，则选择使用 _OverlappingCpuLoader，并设置最大在飞行数据的阈值
            if (
                thread_count == 1
                and (
                    torch.cuda.is_available()
                    or (custom_device_mod and custom_device_mod.is_available())
                )
                and inflight_threshhold > 0
            ):
                loader = _OverlappingCpuLoader(
                    planner.resolve_data,
                    inflight_threshhold=inflight_threshhold,
                )
            else:
                # 否则使用串行的 CPU 数据加载器 _SerialCpuLoader
                loader = _SerialCpuLoader(
                    planner.resolve_data,
                )

            # 将非字节流类型的写入项目添加到 loader 中
            tensor_w = [wi for wi in write_items if wi.type != WriteItemType.BYTE_IO]
            for write_item in tensor_w:
                loader.add(_item_size(write_item), write_item)
            # 开始加载数据
            loader.start_loading()

            # 将字节流类型的写入项目添加到 bytes_w 中
            bytes_w = [wi for wi in write_items if wi.type == WriteItemType.BYTE_IO]
            write_results = []

            # 创建文件流，并以二进制写入模式打开，使用 with 上下文管理
            with create_stream(file_name, "wb") as stream:
                # 对于每个字节流类型的写入项目，通过 planner 解析数据并写入流中，将写入结果存入 write_results 列表
                for write_item in bytes_w:
                    data = planner.resolve_data(write_item)
                    write_results.append(
                        _write_item(stream, data, write_item, storage_key)
                    )

                # 对于 loader 中的每个张量及其对应的写入项目，确保张量在 CPU 上，并将其写入流中，将写入结果存入 write_results 列表
                for tensor, write_item in loader.values():
                    assert tensor.is_cpu  # 断言张量在 CPU 上
                    write_results.append(
                        _write_item(stream, tensor, write_item, storage_key)
                    )

                # 如果设置了使用 fsync，则同步文件流
                if use_fsync:
                    try:
                        os.fsync(stream.fileno())
                    except AttributeError:
                        os.sync()
            # 将写入结果写入结果队列中
            result_queue.put(write_results)
    # 捕获队列为空的异常，终止循环
    except queue.Empty:
        pass
class FileSystemBase(ABC):
    @contextmanager
    @abstractmethod
    def create_stream(
        self, path: Union[str, os.PathLike], mode: str
    ) -> Generator[io.IOBase, None, None]:
        ...

    @abstractmethod
    def concat_path(
        self, path: Union[str, os.PathLike], suffix: str
    ) -> Union[str, os.PathLike]:
        ...

    @abstractmethod
    def rename(
        self, path: Union[str, os.PathLike], new_path: Union[str, os.PathLike]
    ) -> None:
        ...

    @abstractmethod
    def init_path(self, path: Union[str, os.PathLike]) -> Union[str, os.PathLike]:
        ...

    @abstractmethod
    def mkdir(self, path: Union[str, os.PathLike]) -> None:
        ...

    @classmethod
    @abstractmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        ...

    @abstractmethod
    def exists(self, path: Union[str, os.PathLike]) -> bool:
        ...

    @abstractmethod
    def rm_file(self, path: Union[str, os.PathLike]) -> None:
        ...


class FileSystem(FileSystemBase):
    @contextmanager
    # 创建一个上下文管理器，用于在操作完成后自动关闭文件流
    def create_stream(
        self, path: Union[str, os.PathLike], mode: str
    ) -> Generator[io.IOBase, None, None]:
        # 打开指定路径的文件，并将其作为 IOBase 对象返回
        with cast(Path, path).open(mode) as stream:
            yield cast(io.IOBase, stream)

    # 将给定的路径与后缀连接起来
    def concat_path(
        self, path: Union[str, os.PathLike], suffix: str
    ) -> Union[str, os.PathLike]:
        return cast(Path, path) / suffix

    # 初始化路径对象，如果路径不是 Path 对象，则将其转换为 Path 对象
    def init_path(self, path: Union[str, os.PathLike]) -> Union[str, os.PathLike]:
        if not isinstance(path, Path):
            path = Path(path)
        return path

    # 重命名文件或目录
    def rename(
        self, path: Union[str, os.PathLike], new_path: Union[str, os.PathLike]
    ) -> None:
        cast(Path, path).rename(cast(Path, new_path))

    # 创建目录，包括所有必要的父目录，如果目录已存在则忽略
    def mkdir(self, path: Union[str, os.PathLike]) -> None:
        cast(Path, path).mkdir(parents=True, exist_ok=True)

    # 类方法：验证检查点 ID 是否有效，检查路径是否存在且可写
    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        if isinstance(checkpoint_id, Path):
            return True

        if "://" in str(checkpoint_id):
            return False

        for p in Path(checkpoint_id).parents:
            if p.exists() and os.access(str(p), os.W_OK):
                return True

        return False

    # 检查路径是否存在
    def exists(self, path: Union[str, os.PathLike]) -> bool:
        return cast(Path, path).exists()

    # 删除文件
    def rm_file(self, path: Union[str, os.PathLike]) -> None:
        cast(Path, path).unlink()


class _FileSystemWriter(StorageWriter):
    """
    Basic implementation of StorageWriter using file IO.

    This implementation makes the following assumptions and simplifications:

    * The checkpoint path is an empty or non-existing directory.
    * File creation is atomic

    The checkpoint consist of one file per write request plus
    a `.metadata` file with the serialized metadata.

    """
    def __init__(
        self,
        path: Union[str, os.PathLike],  # 初始化方法，接收路径参数，可以是字符串或路径对象
        single_file_per_rank: bool = True,  # 是否每个进程生成一个文件，默认为True
        sync_files: bool = True,  # 是否强制将文件同步到永久存储，默认为True
        thread_count: int = 1,  # 用于写入的IO线程数，默认为1
        per_thread_copy_ahead: int = 10_000_000,  # 每个线程在保存之前从GPU预先复制的字节数，默认为10Mb
        overwrite: bool = True,  # 是否允许覆盖现有的检查点，默认为True
        *args: Any,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the writer pointing to `path`.

        Args:
            path: directory where the checkpoint will be written to.
            single_file_per_rank: Produce one file per rank instead of one file per tensor/blob. Default to True.
            sync_files : force files to be synced to permanent storage. Default to True.
            thread_count: Number of IO threads to use to write. Default to 1.
            per_thread_copy_ahead: How many bytes to copy from the GPU ahead of saving then. Default 10Mb.
            overwrite: Whether to allow overwriting existing checkpoints. Defaults to True.

        N. B. If sync_files is disabled, there's no guarantee that the checkpoint will be consistent in the case of a failure.
        """
        super().__init__()  # 调用父类的初始化方法
        self.fs = FileSystem()  # 创建文件系统对象
        self.path = self.fs.init_path(path)  # 根据给定路径初始化路径对象
        self.single_file_per_rank = single_file_per_rank  # 设置是否每个进程生成一个文件的属性
        self.sync_files = sync_files  # 设置是否强制同步文件到永久存储的属性
        self.thread_count = thread_count  # 设置IO线程数的属性
        self.per_thread_copy_ahead = per_thread_copy_ahead  # 设置每个线程预先复制的GPU数据量的属性
        self.save_id = _generate_uuid()  # 生成并设置保存ID的属性
        self.overwrite = overwrite  # 设置是否允许覆盖现有检查点的属性

    def reset(self, checkpoint_id: Union[str, os.PathLike, None] = None) -> None:
        if checkpoint_id:
            self.path = self.fs.init_path(checkpoint_id)  # 如果提供了检查点ID，则重新初始化路径对象

        self.save_id = _generate_uuid()  # 生成新的保存ID

    def set_up_storage_writer(self, is_coordinator: bool) -> None:
        pass  # 设置存储写入器，这里未实现具体功能，所以为空操作

    def prepare_local_plan(self, plan: SavePlan) -> SavePlan:
        self.fs.mkdir(self.path)  # 在路径上创建目录
        if self.fs.exists(self.metadata_path):  # 检查元数据路径是否存在
            if self.overwrite:  # 如果允许覆盖已有检查点
                warnings.warn(
                    f"Detected an existing checkpoint in {self.metadata_path}, overwriting since {self.overwrite=}."
                    " Past version 2.5 of PyTorch, `overwrite` will default to False. Set this variable to True to"
                    " maintain this functionality or False to raise when an existing checkpoint is found."
                )
            else:
                raise RuntimeError(f"Checkpoint already exists and {self.overwrite=}.")  # 抛出运行时错误，指示检查点已存在且不允许覆盖

        return plan  # 返回准备好的本地保存计划对象

    def prepare_global_plan(self, plans: List[SavePlan]) -> List[SavePlan]:
        new_plans = [
            dataclasses.replace(plan, storage_data=_StoragePrefix(f"__{i}_"))
            for i, plan in enumerate(plans)
        ]  # 对全局保存计划列表进行处理，替换存储数据前缀为序号形式
        return new_plans  # 返回更新后的全局保存计划列表

    def write_data(
        self,
        plan: SavePlan,
        planner: SavePlanner,
    ) -> Future[List[WriteResult]]:
        # 获取存储计划中的存储前缀信息
        storage_plan: _StoragePrefix = plan.storage_data
        # 初始化文件计数器
        file_count = 0

        # 定义生成文件名的内部函数
        def gen_file():
            nonlocal file_count
            # 构建文件名，结合存储前缀、文件计数和默认后缀
            file_name = f"{storage_plan.prefix}{file_count}{DEFAULT_SUFFIX}"
            file_count += 1
            return file_name

        # 创建文件队列
        file_queue: queue.Queue = queue.Queue()
        
        # 如果每个排名只有一个文件
        if self.single_file_per_rank:
            # 按照大小和类型拆分计划项，并生成文件名，加入文件队列
            for bucket in _split_by_size_and_type(self.thread_count, plan.items):
                file_name = gen_file()
                # 构建文件的完整路径
                path = self.fs.concat_path(self.path, file_name)
                file_queue.put((path, file_name, bucket))
        else:
            # 否则，对于每个计划项生成文件名，加入文件队列
            for item in plan.items:
                file_name = gen_file()
                # 构建文件的完整路径
                path = self.fs.concat_path(self.path, file_name)
                file_queue.put((path, file_name, [item]))

        # 创建结果队列
        result_queue: queue.Queue = queue.Queue()

        # 创建线程列表
        threads = []
        # 根据线程数创建线程
        for _ in range(1, self.thread_count):
            t = threading.Thread(
                target=_write_files_from_queue,
                args=(
                    self.fs.create_stream,
                    file_queue,
                    result_queue,
                    planner,
                    self.per_thread_copy_ahead,
                    self.sync_files,
                    self.thread_count,
                ),
            )
            # 启动线程并加入线程列表
            t.start()
            threads.append(t)

        # 单线程方式处理文件队列
        _write_files_from_queue(
            create_stream=self.fs.create_stream,
            file_queue=file_queue,
            result_queue=result_queue,
            planner=planner,
            inflight_threshhold=self.per_thread_copy_ahead,
            use_fsync=self.sync_files,
            thread_count=self.thread_count,
        )

        # 等待所有线程执行完毕
        for t in threads:
            t.join()

        # 收集并返回结果
        res = []
        try:
            while True:
                res += result_queue.get_nowait()
        except queue.Empty:
            pass

        # 创建并返回结果的 Future 对象
        fut: Future[List[WriteResult]] = Future()
        fut.set_result(res)
        return fut

    def finish(self, metadata: Metadata, results: List[List[WriteResult]]) -> None:
        # 整理结果，将写入结果中的存储数据添加到元数据中
        storage_md = dict()
        for wr_list in results:
            storage_md.update({wr.index: wr.storage_data for wr in wr_list})
        metadata.storage_data = storage_md

        # 将存储元数据添加到元数据的存储元信息中
        metadata.storage_meta = self.storage_meta()

        # 创建临时文件路径
        tmp_path = cast(Path, self.fs.concat_path(self.path, f"{_metadata_fn}.tmp"))
        # 使用二进制写入模式创建元数据文件流
        with self.fs.create_stream(tmp_path, "wb") as metadata_file:
            # 将元数据对象序列化并写入文件
            pickle.dump(metadata, metadata_file)
            # 如果启用文件同步，则尝试将文件数据刷新到磁盘
            if self.sync_files:
                try:
                    os.fsync(metadata_file.fileno())
                except AttributeError:
                    os.sync()

        # 如果原始的元数据文件存在，则删除它
        if self.fs.exists(self.metadata_path):
            self.fs.rm_file(self.metadata_path)

        # 将临时文件重命名为正式的元数据文件
        self.fs.rename(tmp_path, self.metadata_path)
    # 返回一个包含存储元数据的对象，其中包括检查点 ID 和保存 ID
    def storage_meta(self) -> Optional[StorageMeta]:
        return StorageMeta(checkpoint_id=self.checkpoint_id, save_id=self.save_id)

    # 返回元数据文件的路径，该路径由文件系统对象连接路径和元数据文件名组成
    @property
    def metadata_path(self) -> Union[str, os.PathLike]:
        return cast(Path, self.fs.concat_path(self.path, _metadata_fn))

    # 返回用于保存检查点的检查点 ID
    @property
    def checkpoint_id(self) -> Union[str, os.PathLike]:
        """
        return the checkpoint_id that will be used to save the checkpoint.
        """
        return self.path

    # 类方法：验证给定的检查点 ID 是否有效
    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        return FileSystem.validate_checkpoint_id(checkpoint_id)
    # FileSystemReader 类继承自 StorageReader 类
class FileSystemReader(StorageReader):
    # 初始化函数，接受一个路径参数，可以是字符串或路径对象
    def __init__(self, path: Union[str, os.PathLike]) -> None:
        # 调用父类的初始化方法
        super().__init__()
        # 创建 FileSystem 对象实例
        self.fs = FileSystem()
        # 使用给定路径初始化文件系统路径
        self.path = self.fs.init_path(path)
        # 存储元数据索引到存储信息的字典
        self.storage_data: Dict[MetadataIndex, _StorageInfo] = dict()
        # 生成加载 ID
        self.load_id = _generate_uuid()

    # 私有方法，用于从文件中切片读取数据
    def _slice_file(self, file, sinfo: _StorageInfo) -> io.IOBase:
        return _create_file_view(file, sinfo.offset, sinfo.length)

    # 重置对象状态，清空存储数据字典，如果指定了检查点 ID，则重新初始化路径
    def reset(self, checkpoint_id: Union[str, os.PathLike, None] = None) -> None:
        self.storage_data = dict()
        if checkpoint_id:
            self.path = self.fs.init_path(checkpoint_id)
        # 重新生成加载 ID
        self.load_id = _generate_uuid()

    # 读取数据的方法，接受加载计划和加载规划器对象
    def read_data(self, plan: LoadPlan, planner: LoadPlanner) -> Future[None]:
        # 按文件分组请求
        per_file: Dict[str, List[ReadItem]] = dict()
        for read_item in plan.items:
            # 获取存储索引对应的元数据信息
            item_md = self.storage_data[read_item.storage_index]
            # 获取相对路径
            path = item_md.relative_path
            # 将请求按文件路径分组
            per_file.setdefault(path, []).append(read_item)

        # 遍历每个文件的请求
        for relative_path, reqs in per_file.items():
            # 构建完整的文件路径
            new_path = self.fs.concat_path(self.path, relative_path)
            # 使用文件系统对象创建读取流，以二进制模式打开
            with self.fs.create_stream(new_path, "rb") as stream:
                # TODO 按偏移排序并缓存读取结果
                # 遍历请求列表
                for req in reqs:
                    # 获取请求对应的元数据信息
                    item_md = self.storage_data[req.storage_index]
                    # 从文件中切片读取数据
                    file_slice = self._slice_file(stream, item_md)
                    # 如果请求类型是字节 IO
                    if req.type == LoadItemType.BYTE_IO:
                        # 创建字节流并读取指定长度的数据
                        read_bytes = io.BytesIO(file_slice.read(item_md.length))
                        read_bytes.seek(0)
                        # 使用加载规划器加载字节数据
                        planner.load_bytes(req, read_bytes)
                    else:
                        # 否则，假设请求类型是张量
                        tensor = cast(
                            Tensor,
                            # 从文件切片中加载张量数据，指定 CPU 上的位置
                            torch.load(cast(IO[bytes], file_slice), map_location="cpu"),
                        )
                        # 根据指定的偏移和长度缩小张量
                        tensor = narrow_tensor_by_index(
                            tensor, req.storage_offsets, req.lengths
                        )
                        # 解析并分离加载规划器中的目标张量
                        target_tensor = planner.resolve_tensor(req).detach()

                        # 断言确保目标张量和加载的张量尺寸匹配
                        assert (
                            target_tensor.size() == tensor.size()
                        ), f"req {req.storage_index} mismatch sizes {target_tensor.size()} vs {tensor.size()}"
                        # 复制加载的张量到目标张量中
                        target_tensor.copy_(tensor)
                        # 提交目标张量到加载规划器中
                        planner.commit_tensor(req, target_tensor)

        # 创建一个空的 Future 对象并设置结果为 None，返回该 Future 对象
        fut: Future = Future()
        fut.set_result(None)
        return fut

    # 实现 StorageReader 抽象类中的方法
    # 读取对象的元数据，并返回Metadata对象
    def read_metadata(self) -> Metadata:
        # 构建.metadata文件的完整路径
        path = self.fs.concat_path(self.path, ".metadata")
        # 使用文件系统创建只读二进制流对象metadata_file
        with self.fs.create_stream(path, "rb") as metadata_file:
            # 从metadata_file中加载pickle序列化的元数据
            metadata = pickle.load(metadata_file)

        # 如果metadata中的storage_meta属性为None，则创建一个新的StorageMeta对象
        if getattr(metadata, "storage_meta", None) is None:
            metadata.storage_meta = StorageMeta()
        # 将当前对象的load_id赋值给metadata的storage_meta.load_id属性
        metadata.storage_meta.load_id = self.load_id

        # 返回读取的metadata对象
        return metadata

    # 设置存储读取器，将metadata中的storage_data属性赋值给当前对象的storage_data属性
    def set_up_storage_reader(self, metadata: Metadata, is_coordinator: bool) -> None:
        self.storage_data = metadata.storage_data
        assert self.storage_data is not None  # 断言storage_data属性不为None

    # 准备本地计划，直接返回输入的计划对象
    def prepare_local_plan(self, plan: LoadPlan) -> LoadPlan:
        return plan

    # 准备全局计划，直接返回输入的计划列表对象
    def prepare_global_plan(self, plans: List[LoadPlan]) -> List[LoadPlan]:
        return plans

    # 属性方法，返回当前对象的路径作为checkpoint_id，用于加载检查点
    @property
    def checkpoint_id(self) -> Union[str, os.PathLike]:
        """
        返回将用于加载检查点的checkpoint_id。
        """
        return self.path

    # 类方法，验证给定的checkpoint_id是否有效，调用FileSystem类的validate_checkpoint_id静态方法
    @classmethod
    def validate_checkpoint_id(cls, checkpoint_id: Union[str, os.PathLike]) -> bool:
        return FileSystem.validate_checkpoint_id(checkpoint_id)
    """
    Basic implementation of StorageWriter using file IO.

    This implementation makes the following assumptions and simplifications:

    * The checkpoint path is an empty or non-existing directory.
    * File creation is atomic

    The checkpoint consist of one file per write request plus
    a `.metadata` file with the serialized metadata.

    """

    def __init__(
        self,
        path: Union[str, os.PathLike],
        single_file_per_rank: bool = True,
        sync_files: bool = True,
        thread_count: int = 1,
        per_thread_copy_ahead: int = 10_000_000,
        cache_staged_state_dict: bool = False,
        overwrite: bool = True,
    ) -> None:
        """
        Initialize the writer pointing to `path`.

        Args:
            path: directory where the checkpoint will be written to.
            single_file_per_rank: Produce one file per rank instead of one file per tensor/blob. Default to True.
            sync_files : force files to be synced to permanent storage. Default to True.
            thread_count: Number of IO threads to use to write. Default to 1.
            per_thread_copy_ahead: How many bytes to copy from the GPU ahead of saving then. Default 10Mb.
            cache_staged_state_dict: Whether to cache the staged state_dict. This option decreases staging latency
                at the cost of increases memory usage. Additionally, if this parameter is set to True, it's the expectation
                that the stager is maintained and re-used for multiple dcp.async_save calls. Default to False.
            overwrite: Whether to allow overwriting existing checkpoints. Defaults to True.

        N. B. If sync_files is disabled, there's no guarantee that the checkpoint will be consistent in the case of a failure.
        """
        # 调用父类的构造函数 _FileSystemWriter，并设置初始参数
        super().__init__(
            path=path,
            single_file_per_rank=single_file_per_rank,
            sync_files=sync_files,
            thread_count=thread_count,
            per_thread_copy_ahead=per_thread_copy_ahead,
            cache_staged_state_dict=cache_staged_state_dict,
            overwrite=overwrite,
        )

    def stage(self, state_dict: STATE_DICT_TYPE) -> STATE_DICT_TYPE:
        """Override of AsyncStager.stage"""
        # 在异步情况下，状态字典已经在 CPU 上，因此维护此缓冲区没有意义
        # 设置每个线程提前复制的字节数为 0
        self.per_thread_copy_ahead = 0
        # 调用父类 AsyncStager 的 stage 方法，并返回其结果
        return super().stage(state_dict)
```