# `.\pytorch\torch\serialization.py`

```
# mypy: allow-untyped-defs
# 导入必要的标准库和第三方库
import copyreg  # 注册对象支持的 pickle 协议
import difflib  # 生成差异比较的工具
import functools  # 提供一些高阶函数
import io  # 提供核心的 I/O 功能
import os  # 提供与操作系统交互的功能
import pickle  # 提供对象序列化和反序列化的功能
import re  # 提供正则表达式的支持
import shutil  # 提供高级文件操作功能
import struct  # 提供处理 C 结构体的功能
import sys  # 提供与 Python 解释器相关的变量和函数
import tarfile  # 提供操作 tar 文件的功能
import tempfile  # 提供创建临时文件和目录的功能
import warnings  # 提供警告相关的功能
from contextlib import closing, contextmanager  # 提供上下文管理相关的功能
from enum import Enum  # 提供枚举类型的支持
from typing import (  # 提供静态类型检查相关的功能
    Any,
    BinaryIO,
    Callable,
    cast,
    Dict,
    IO,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)
from typing_extensions import TypeAlias, TypeGuard  # 提供类型别名和类型守卫的支持（Python 3.10+）

import torch  # PyTorch 深度学习库
import torch._weights_only_unpickler as _weights_only_unpickler  # PyTorch 加载只包含权重的模型
from torch._sources import get_source_lines_and_file  # PyTorch 获取源码行和文件路径
from torch._utils import _import_dotted_name  # PyTorch 导入点分名称
from torch.storage import _get_dtype_from_pickle_storage_type  # PyTorch 从 pickle 存储类型获取数据类型
from torch.types import Storage  # PyTorch 存储类型的类型定义

# 导出的符号列表
__all__ = [
    "SourceChangeWarning",
    "mkdtemp",
    "register_package",
    "check_module_version_greater_or_equal",
    "validate_cuda_device",
    "validate_hpu_device",
    "location_tag",
    "default_restore_location",
    "normalize_storage_type",
    "storage_to_tensor_type",
    "save",
    "load",
    "StorageType",
    "LoadEndianness",
    "get_default_load_endianness",
    "set_default_load_endianness",
    "clear_safe_globals",
    "get_safe_globals",
    "add_safe_globals",
]

# 默认的 pickle 协议版本
DEFAULT_PROTOCOL = 2

# 不同整数类型的字节大小
LONG_SIZE = struct.Struct("=l").size
INT_SIZE = struct.Struct("=i").size
SHORT_SIZE = struct.Struct("=h").size

# 魔数和协议版本号
MAGIC_NUMBER = 0x1950A86A20F9469CFC6C
PROTOCOL_VERSION = 1001
STORAGE_KEY_SEPARATOR = ","

# 文件类别的类型别名定义
FILE_LIKE: TypeAlias = Union[str, os.PathLike, BinaryIO, IO[bytes]]
# 映射位置的类型别名定义
MAP_LOCATION: TypeAlias = Optional[
    Union[Callable[[Storage, str], Storage], torch.device, str, Dict[str, str]]
]
# 存储类型的类型别名定义
STORAGE: TypeAlias = Union[Storage, torch.storage.TypedStorage, torch.UntypedStorage]

# 判断是否为 Windows 系统
IS_WINDOWS = sys.platform == "win32"

# 如果不是 Windows 系统，则导入 mmap 模块的常量
if not IS_WINDOWS:
    from mmap import MAP_PRIVATE, MAP_SHARED
else:
    MAP_SHARED, MAP_PRIVATE = None, None  # type: ignore[assignment]

# 源代码变更警告类
class SourceChangeWarning(Warning):
    pass

# 创建临时目录的上下文管理器
@contextmanager
def mkdtemp():
    path = tempfile.mkdtemp()
    try:
        yield path
    finally:
        shutil.rmtree(path)

# 包注册表，存储三元组的列表：优先级、序列化函数、反序列化函数
_package_registry: List[
    Tuple[
        int,
        Callable[[STORAGE], Optional[str]],
        Callable[[STORAGE, str], Optional[STORAGE]],
    ]
] = []

# 加载字节序的枚举类
class LoadEndianness(Enum):
    NATIVE = 1
    LITTLE = 2
    BIG = 3

# 默认的加载字节序设置
_default_load_endian: Optional[LoadEndianness] = None

# 获取默认的加载字节序
def get_default_load_endianness() -> Optional[LoadEndianness]:
    """
    Get fallback byte order for loading files

    If byteorder mark is not present in saved checkpoint,
    this byte order is used as fallback.
    By default, it's "native" byte order.

    Returns:
        default_load_endian: Optional[LoadEndianness]
    """
    return _default_load_endian

# 设置默认的加载字节序
def set_default_load_endianness(endianness):
    """
    Set fallback byte order for loading files

    If byteorder mark is not present in saved checkpoint,
    this byte order is used as fallback.
    """
    global _default_load_endian
    _default_load_endian = endianness
    By default, it's "native" byte order.

    Args:
        endianness: the new fallback byte order
    """
    # 声明全局变量 _default_load_endian，用于存储默认加载字节顺序
    global _default_load_endian
    # 如果 endianness 不是 LoadEndianness 类型且不为 None，则抛出类型错误异常
    if not isinstance(endianness, LoadEndianness) and endianness is not None:
        raise TypeError("Invalid argument type in function set_default_load_endianness")
    # 将 endianness 赋值给 _default_load_endian，即设置默认加载字节顺序
    _default_load_endian = endianness
# 默认的 mmap 选项，用于 torch.load 函数的 mmap=True 参数
_default_mmap_options: int = MAP_PRIVATE


def get_default_mmap_options() -> int:
    """
    获取用于 torch.load 函数中 mmap=True 的默认 mmap 选项。

    默认为 mmap.MAP_PRIVATE。

    Returns:
        default_mmap_options: int
    """
    return _default_mmap_options


def set_default_mmap_options(flags: int):
    """
    设置用于 torch.load 函数中 mmap=True 的默认 mmap 选项为给定的 flags。

    目前只支持 mmap.MAP_PRIVATE 或 mmap.MAP_SHARED。
    如果需要添加其他选项，请提出问题。

    .. note::
        此功能目前不支持 Windows。

    Args:
        flags: mmap.MAP_PRIVATE 或 mmap.MAP_SHARED
    """
    global _default_mmap_options
    if IS_WINDOWS:
        raise RuntimeError(
            "当前不支持在 Windows 上更改默认的 mmap 选项"
        )
    if flags != MAP_PRIVATE and flags != MAP_SHARED:
        raise ValueError(
            "在函数 set_default_mmap_options 中提供的参数无效，"
            f"期望 mmap.MAP_PRIVATE 或 mmap.MAP_SHARED，但得到了 {flags}"
        )
    _default_mmap_options = flags


def clear_safe_globals() -> None:
    """
    清除对于 "weights_only" 加载而言安全的全局变量列表。
    """
    _weights_only_unpickler._clear_safe_globals()


def get_safe_globals() -> List[Any]:
    """
    返回对于 "weights_only" 加载而言安全的用户添加全局变量列表。
    """
    return _weights_only_unpickler._get_safe_globals()


def add_safe_globals(safe_globals: List[Any]) -> None:
    """
    标记给定的全局变量列表为 "weights_only" 加载而言安全。
    例如，添加到此列表的函数可以在反序列化期间调用，类可以被实例化并设置状态。

    Args:
        safe_globals (List[Any]): 要标记为安全的全局变量列表

    Example:
        >>> # xdoctest: +SKIP("Can't torch.save(t, ...) as doctest thinks MyTensor is defined on torch.serialization")
        >>> import tempfile
        >>> class MyTensor(torch.Tensor):
        ...     pass
        >>> t = MyTensor(torch.randn(2, 3))
        >>> with tempfile.NamedTemporaryFile() as f:
        ...     torch.save(t, f.name)
        # 运行 `torch.load(f.name, weights_only=True)` 将会失败，
        # 因为不支持的全局变量：默认情况下，GLOBAL __main__.MyTensor 不是允许的全局变量。
        # 检查代码并确保在从任意检查点加载时 MyTensor 是安全可用的。
        ...     torch.serialization.add_safe_globals([MyTensor])
        ...     torch.load(f.name, weights_only=True)
        # MyTensor([[-0.5024, -1.8152, -0.5455],
        #          [-0.8234,  2.0500, -0.3657]])
    """
    _weights_only_unpickler._add_safe_globals(safe_globals)


def _is_zipfile(f) -> bool:
    # 这是比 zipfile.is_zipfile() 更严格的实现。
    # 如果在文件开头找到 ZIP 文件的魔术数字，返回 True。
    # 获取当前文件指针位置
    start = f.tell()
    # 定义本地文件头的魔数（magic number），用于识别 ZIP 文件的开始
    local_header_magic_number = b"PK\x03\x04"
    # 读取文件中与本地文件头魔数长度相同的字节
    read_bytes = f.read(len(local_header_magic_number))
    # 将文件指针重新定位到起始位置
    f.seek(start)
    # 检查读取的字节是否与本地文件头魔数相同，用于确定是否为有效的 ZIP 文件
    return read_bytes == local_header_magic_number
# 注册一个可调用对象，用于为存储对象打标签和反序列化，带有关联的优先级。
# 打标签操作将在保存时将设备与存储对象关联，而反序列化操作将在加载时将存储对象移动到适当的设备上。
# :attr:`tagger` 和 :attr:`deserializer` 按其 :attr:`priority` 指定的顺序运行，直到一个 tagger/deserializer 返回非 `None` 值为止。
def register_package(
    priority: int,
    tagger: Callable[[STORAGE], Optional[str]],
    deserializer: Callable[[STORAGE, str], Optional[STORAGE]],
):
    """
    Registers callables for tagging and deserializing storage objects with an associated priority.
    Tagging associates a device with a storage object at save time while deserializing moves a
    storage object to an appropriate device at load time. :attr:`tagger` and :attr:`deserializer`
    are run in the order given by their :attr:`priority` until a tagger/deserializer returns a
    value that is not `None`.

    To override the deserialization behavior for a device in the global registry, one can register a
    tagger with a higher priority than the existing tagger.

    This function can also be used to register a tagger and deserializer for new devices.

    Args:
        priority: Indicates the priority associated with the tagger and deserializer, where a lower
            value indicates higher priority.
        tagger: Callable that takes in a storage object and returns its tagged device as a string
            or None.
        deserializer: Callable that takes in storage object and a device string and returns a storage
            object on the appropriate device or None.

    Returns:
        `None`

    Example:
        >>> def ipu_tag(obj):
        >>>     if obj.device.type == 'ipu':
        >>>         return 'ipu'
        >>> def ipu_deserialize(obj, location):
        >>>     if location.startswith('ipu'):
        >>>         ipu = getattr(torch, "ipu", None)
        >>>         assert ipu is not None, "IPU device module is not loaded"
        >>>         assert torch.ipu.is_available(), "ipu is not available"
        >>>         return obj.ipu(location)
        >>> torch.serialization.register_package(11, ipu_tag, ipu_deserialize)
    """
    # 创建包含优先级、标签器和反序列化器的元组
    queue_elem = (priority, tagger, deserializer)
    # 将元组添加到全局包注册表
    _package_registry.append(queue_elem)
    # 根据优先级排序注册表
    _package_registry.sort()


# 检查模块的版本是否大于或等于指定要求的版本元组
def check_module_version_greater_or_equal(
    module,
    req_version_tuple,
    error_if_malformed=True,
):
    """
    Check if a module's version satisfies requirements

    Usually, a module's version string will be like 'x.y.z', which would be represented
    as a tuple (x, y, z), but sometimes it could be an unexpected format. If the version
    string does not match the given tuple's format up to the length of the tuple, then
    error and exit or emit a warning.

    Args:
        module: the module to check the version of
        req_version_tuple: tuple (usually of ints) representing the required version
        error_if_malformed: whether we should exit if module version string is malformed

    Returns:
        requirement_is_met: bool
    """
    try:
        # 尝试获取模块的版本字符串，并按"."分割为列表
        version_strs = module.__version__.split(".")
        # 将模块版本的各个字段转换为与所需版本字段相同类型的元组
        module_version = tuple(
            type(req_field)(version_strs[idx])
            for idx, req_field in enumerate(req_version_tuple)
        )
        # 检查模块版本是否符合要求的版本要求
        requirement_is_met = module_version >= req_version_tuple

    except Exception as e:
        # 如果捕获到异常，说明模块版本字符串格式有误
        message = (
            f"'{module.__name__}' module version string is malformed '{module.__version__}' and cannot be compared"
            f" with tuple {str(req_version_tuple)}"
        )
        if error_if_malformed:
            # 如果设置了在版本字符串格式错误时抛出异常，则抛出运行时异常
            raise RuntimeError(message) from e
        else:
            # 否则，发出警告，继续假设满足版本要求
            warnings.warn(message + ", but continuing assuming that requirement is met")
            requirement_is_met = True

    # 返回是否满足版本要求的布尔值
    return requirement_is_met
# 检查对象所在设备是否为 CPU，如果是则返回字符串 "cpu"
def _cpu_tag(obj):
    if obj.device.type == "cpu":
        return "cpu"

# 检查对象所在设备是否为 mps，如果是则返回字符串 "mps"
def _mps_tag(obj):
    if obj.device.type == "mps":
        return "mps"

# 检查对象所在设备是否为 meta，如果是则返回字符串 "meta"
def _meta_tag(obj):
    if obj.device.type == "meta":
        return "meta"

# 根据给定的 backend_name 和 obj，返回适当的设备标签
def _backend_tag(backend_name, obj):
    if backend_name == "privateuse1":
        # 如果 backend_name 是 privateuse1，则获取相应的后端名称
        backend_name = torch._C._get_privateuse1_backend_name()
    # 检查 obj 是否属于指定的设备类型，如果是则返回对应的标签字符串
    if obj.device.type == backend_name:
        if obj.device.index is None:
            return backend_name  # 返回只有名称的标签
        else:
            return backend_name + ":" + str(obj.device.index)  # 返回带索引的标签

# 检查对象是否在 CPU 上，如果在则返回对象本身
def _cpu_deserialize(obj, location):
    if location == "cpu":
        return obj

# 检查对象是否在 mps 设备上，如果是则调用 mps 方法返回对象
def _mps_deserialize(obj, location):
    if location.startswith("mps"):
        return obj.mps()

# 检查对象是否在 meta 设备上，如果是则创建一个未类型化的存储对象并返回
def _meta_deserialize(obj, location):
    if location == "meta":
        return torch.UntypedStorage(obj.nbytes(), device="meta")

# 验证指定后端和设备的设备索引是否有效
def _validate_device(location, backend_name):
    """
    Check whether the device index of specified backend is valid

    In case of privateuse1 backend, your must first register a device_module for
    privateuse1 using torch._register_device_module. Implement the following
    methods in device_module like cuda: device_module._utils._get_device_index(location, True),
    device_module.device_count().

    Args:
        location: string of device
        backend_name: the backend name or the name of privateuse1, which can be renamed

    Returns:
        device_index: int
    """
    # 检查是否存在指定的后端名称
    if not hasattr(torch, backend_name):
        raise RuntimeError(
            f"The {backend_name.upper()} device module is not registered. "
            "If you are running on a CPU-only machine, "
            "please use torch.load with map_location=torch.device('cpu') "
            "to map your storages to the CPU."
        )
    # 获取对应的设备模块
    device_module = getattr(torch, backend_name)
    # 检查是否有必要的工具函数，并获取设备索引
    if hasattr(device_module, "_utils") and hasattr(
        device_module._utils, "_get_device_index"
    ):
        device_index = device_module._utils._get_device_index(location, True)
        device = torch.device(backend_name, device_index)
    else:
        # 如果没有合适的工具函数，则创建一个默认设备对象
        device = torch.device(location)
        device_index = device.index if device.index else 0
    # 检查设备模块是否可用
    if hasattr(device_module, "is_available") and not device_module.is_available():
        raise RuntimeError(
            f"Attempting to deserialize object on a {backend_name.upper()} "
            f"device but torch.{backend_name}.is_available() is False. "
            "If you are running on a CPU-only machine, "
            "please use torch.load with map_location=torch.device('cpu') "
            "to map your storages to the CPU."
        )
    # 检查 device_module 对象是否具有属性 "device_count"
    if hasattr(device_module, "device_count"):
        # 调用 device_count() 方法获取设备数量
        device_count = device_module.device_count()
        # 如果 device_index 超出设备数量范围，抛出运行时错误
        if device_index >= device_count:
            raise RuntimeError(
                # 抛出错误消息，指出正在尝试在指定设备上反序列化对象，但设备索引超出范围
                f"Attempting to deserialize object on {backend_name.upper()} device "
                f"{device_index} but torch.{backend_name}.device_count() is {device_count}. "
                "Please use torch.load with map_location to map your storages "
                "to an existing device."
            )
    # 返回 device 变量，这里假设它是函数的返回值，用于指示操作结束
    return device
# 验证 CUDA 设备的有效性，并返回设备索引
def validate_cuda_device(location):
    return _validate_device(location, "cuda").index


# 验证 HPU 设备的有效性，并返回设备索引
def validate_hpu_device(location):
    return _validate_device(location, "hpu").index


# 使用指定的后端名称和位置反序列化对象
def _deserialize(backend_name, obj, location):
    if backend_name == "privateuse1":
        # 获取私有使用1后端的名称
        backend_name = torch._C._get_privateuse1_backend_name()
    if location.startswith(backend_name):
        # 验证位置是否有效，并将对象转移到指定的设备
        device = _validate_device(location, backend_name)
        return obj.to(device=device)


# 将函数和相关参数注册到包注册表中
register_package(10, _cpu_tag, _cpu_deserialize)

# 将函数和相关参数注册到包注册表中，使用 CUDA 后端
register_package(
    20,
    functools.partial(_backend_tag, "cuda"),  # 使用 functools.partial 创建部分应用的函数对象
    functools.partial(_deserialize, "cuda"),  # 使用 functools.partial 创建部分应用的函数对象
)

# 将函数和相关参数注册到包注册表中
register_package(21, _mps_tag, _mps_deserialize)

# 将函数和相关参数注册到包注册表中
register_package(22, _meta_tag, _meta_deserialize)

# 将函数和相关参数注册到包注册表中，使用私有使用1后端
register_package(
    23,
    functools.partial(_backend_tag, "privateuse1"),  # 使用 functools.partial 创建部分应用的函数对象
    functools.partial(_deserialize, "privateuse1"),  # 使用 functools.partial 创建部分应用的函数对象
)

# 将函数和相关参数注册到包注册表中，使用 HPU 后端
register_package(
    24,
    functools.partial(_backend_tag, "hpu"),  # 使用 functools.partial 创建部分应用的函数对象
    functools.partial(_deserialize, "hpu"),  # 使用 functools.partial 创建部分应用的函数对象
)

# 将函数和相关参数注册到包注册表中，使用 XPU 后端
register_package(
    25,
    functools.partial(_backend_tag, "xpu"),  # 使用 functools.partial 创建部分应用的函数对象
    functools.partial(_deserialize, "xpu"),  # 使用 functools.partial 创建部分应用的函数对象
)


# 确定给定存储对象的数据位置标签
def location_tag(
    storage: Union[Storage, torch.storage.TypedStorage, torch.UntypedStorage],
):
    for _, tagger, _ in _package_registry:
        location = tagger(storage)  # 获取位置标签
        if location:
            return location
    # 如果没有找到位置标签，抛出运行时错误
    raise RuntimeError(
        "don't know how to determine data location of " + torch.typename(storage)
    )


# 使用注册的反序列化函数来恢复存储对象
def default_restore_location(storage, location):
    """
    Restores `storage` using a deserializer function registered for the `location`.

    This function looks in the registry for deserializer functions that match the `location`.
    If found, it attempts to use them, in priority order, to restore `storage` until one
    returns a not `None` result. If no deserializer can be found in the registry, or all found fail
    to bear a result, it raises a `RuntimeError`.

    Args:
        storage (STORAGE): the storage object to restore
        location (str): the location tag associated with the storage object

    Returns:
        storage: Optional[STORAGE]

    Raises:
        RuntimeError: If no deserializer matching `location` is found in the registry or if
           all matching ones return `None`.
    """
    for _, _, fn in _package_registry:
        result = fn(storage, location)  # 使用注册的函数尝试恢复存储对象
        if result is not None:
            return result
    # 如果没有找到适合的反序列化函数，抛出运行时错误
    raise RuntimeError(
        "don't know how to restore data location of "
        + torch.typename(storage)
        + " (tagged with "
        + location
        + ")"
    )


# 根据存储类型获取对应的 Torch 类型
def normalize_storage_type(storage_type):
    return getattr(torch, storage_type.__name__)


# 将存储对象转换为对应的张量类型
def storage_to_tensor_type(storage):
    storage_type = type(storage)
    module = _import_dotted_name(storage_type.__module__)
    return getattr(module, storage_type.__name__.replace("Storage", "Tensor"))


# 判断给定的名称或缓冲区是否为路径
def _is_path(name_or_buffer) -> TypeGuard[Union[str, os.PathLike]]:
    # 检查参数 name_or_buffer 是否是字符串或 os.PathLike 对象的实例
    return isinstance(name_or_buffer, (str, os.PathLike))
class _opener:
    # 定义一个上下文管理器基类，用于打开文件或缓冲区
    def __init__(self, file_like):
        self.file_like = file_like

    def __enter__(self):
        # 返回被包装的文件或缓冲区对象，以便在上下文中使用
        return self.file_like

    def __exit__(self, *args):
        # 在退出上下文时执行的清理操作，此处无操作
        pass


class _open_file(_opener):
    # 继承自 _opener，用于打开文件
    def __init__(self, name, mode):
        super().__init__(open(name, mode))

    def __exit__(self, *args):
        # 在退出上下文时关闭文件
        self.file_like.close()


class _open_buffer_reader(_opener):
    # 继承自 _opener，用于读取缓冲区
    def __init__(self, buffer):
        super().__init__(buffer)
        _check_seekable(buffer)  # 检查缓冲区是否支持 seek 操作


class _open_buffer_writer(_opener):
    # 继承自 _opener，用于写入缓冲区
    def __exit__(self, *args):
        # 在退出上下文时刷新缓冲区
        self.file_like.flush()


def _open_file_like(name_or_buffer, mode):
    # 根据输入的文件名或缓冲区对象，选择合适的打开方式
    if _is_path(name_or_buffer):
        return _open_file(name_or_buffer, mode)  # 如果是文件路径，使用 _open_file 打开
    else:
        if "w" in mode:
            return _open_buffer_writer(name_or_buffer)  # 如果是写入模式，使用 _open_buffer_writer 打开
        elif "r" in mode:
            return _open_buffer_reader(name_or_buffer)  # 如果是读取模式，使用 _open_buffer_reader 打开
        else:
            # 如果模式既不是读取也不是写入，抛出运行时错误
            raise RuntimeError(f"Expected 'r' or 'w' in mode but got {mode}")


class _open_zipfile_reader(_opener):
    # 继承自 _opener，用于读取 ZIP 文件
    def __init__(self, name_or_buffer) -> None:
        super().__init__(torch._C.PyTorchFileReader(name_or_buffer))


class _open_zipfile_writer_file(_opener):
    # 继承自 _opener，用于以文件方式写入 ZIP 文件
    def __init__(self, name) -> None:
        self.file_stream = None
        self.name = str(name)
        try:
            self.name.encode("ascii")
        except UnicodeEncodeError:
            # 如果文件名包含非 ASCII 字符，使用 Python 写文件，而不使用 PyTorchFileWriter
            self.file_stream = io.FileIO(self.name, mode="w")
            super().__init__(torch._C.PyTorchFileWriter(self.file_stream))
        else:
            super().__init__(torch._C.PyTorchFileWriter(self.name))

    def __exit__(self, *args) -> None:
        # 在退出上下文时，写入文件结束标志，并关闭文件流（如果存在）
        self.file_like.write_end_of_file()
        if self.file_stream is not None:
            self.file_stream.close()


class _open_zipfile_writer_buffer(_opener):
    # 继承自 _opener，用于以缓冲区方式写入 ZIP 文件
    def __init__(self, buffer) -> None:
        if not callable(getattr(buffer, "write", None)):
            # 如果缓冲区没有 write 方法，抛出错误
            msg = f"Buffer of {str(type(buffer)).strip('<>')} has no callable attribute 'write'"
            if not hasattr(buffer, "write"):
                raise AttributeError(msg)
            raise TypeError(msg)
        self.buffer = buffer
        super().__init__(torch._C.PyTorchFileWriter(buffer))

    def __exit__(self, *args) -> None:
        # 在退出上下文时，写入文件结束标志，并刷新缓冲区
        self.file_like.write_end_of_file()
        self.buffer.flush()


def _open_zipfile_writer(name_or_buffer):
    # 根据输入的文件名或缓冲区对象，选择合适的 ZIP 文件写入方式
    container: Type[_opener]
    if _is_path(name_or_buffer):
        container = _open_zipfile_writer_file  # 如果是文件路径，使用 _open_zipfile_writer_file 打开
    else:
        container = _open_zipfile_writer_buffer  # 如果是缓冲区对象，使用 _open_zipfile_writer_buffer 打开
    return container(name_or_buffer)


def _is_compressed_file(f) -> bool:
    # 检查文件是否是压缩文件（目前支持 gzip）
    compress_modules = ["gzip"]
    try:
        return f.__module__ in compress_modules
    except AttributeError:
        return False


def _should_read_directly(f):
    """
    检查是否应该直接读取文件
    """
    Checks if f is a file that should be read directly. It should be read
    directly if it is backed by a real file (has a fileno) and is not a
    a compressed file (e.g. gzip)
    """
    # 检查文件 f 是否为应直接读取的文件。如果是真实文件（具有 fileno）且不是压缩文件（如 gzip），应直接读取。
    if _is_compressed_file(f):
        # 如果 f 是压缩文件，返回 False，不应直接读取
        return False
    try:
        # 尝试获取文件描述符 fileno，若成功则表明 f 是真实文件，应直接读取
        return f.fileno() >= 0
    except io.UnsupportedOperation:
        # 如果在尝试获取 fileno 时抛出 UnsupportedOperation 异常，说明 f 不是真实文件，不应直接读取
        return False
    except AttributeError:
        # 如果 f 没有 fileno 属性，说明 f 不是真实文件，不应直接读取
        return False
def _check_seekable(f) -> bool:
    # 检查文件对象是否支持 seek 和 tell 方法，用于判断文件是否支持随机访问
    def raise_err_msg(patterns, e):
        # 遍历错误消息的模式列表，如果异常信息中包含其中任何一个模式，则抛出带有详细说明的新异常
        for p in patterns:
            if p in str(e):
                msg = (
                    str(e)
                    + ". You can only torch.load from a file that is seekable."
                    + " Please pre-load the data into a buffer like io.BytesIO and"
                    + " try to load from it instead."
                )
                raise type(e)(msg)
        # 如果异常信息不匹配任何模式，则继续抛出原始异常
        raise e

    try:
        # 尝试在当前文件位置调用 seek 和 tell 方法，如果成功则返回 True
        f.seek(f.tell())
        return True
    except (io.UnsupportedOperation, AttributeError) as e:
        # 如果调用 seek 或 tell 方法出现不支持操作或属性错误的异常，则调用 raise_err_msg 处理异常
        raise_err_msg(["seek", "tell"], e)
    # 如果未能成功调用 seek 和 tell 方法，则返回 False
    return False


def _check_dill_version(pickle_module) -> None:
    """Checks if using dill as the pickle module, and if so, checks if it is the correct version.
    If dill version is lower than 0.3.1, a ValueError is raised.

    Args:
        pickle_module: module used for pickling metadata and objects

    """
    if pickle_module is not None and pickle_module.__name__ == "dill":
        # 检查 dill 模块版本是否符合要求，若低于 0.3.1 则抛出 ValueError 异常
        required_dill_version = (0, 3, 1)
        if not check_module_version_greater_or_equal(
            pickle_module, required_dill_version, False
        ):
            raise ValueError(
                (
                    "'torch' supports dill >= {}, but you have dill {}."
                    " Please upgrade dill or switch to 'pickle'"
                ).format(
                    ".".join([str(num) for num in required_dill_version]),
                    pickle_module.__version__,
                )
            )


def _check_save_filelike(f):
    # 检查对象 f 是否为路径字符串或具有 write 属性的文件对象，否则抛出 AttributeError
    if not _is_path(f) and not hasattr(f, "write"):
        raise AttributeError(
            "expected 'f' to be string, path, or a file-like object with "
            "a 'write' attribute"
        )


def save(
    obj: object,
    f: FILE_LIKE,
    pickle_module: Any = pickle,
    pickle_protocol: int = DEFAULT_PROTOCOL,
    _use_new_zipfile_serialization: bool = True,
    _disable_byteorder_record: bool = False,
) -> None:
    # Reference: https://github.com/pytorch/pytorch/issues/54354
    # The first line of this docstring overrides the one Sphinx generates for the
    # documentation. We need it so that Sphinx doesn't leak `pickle`s path from
    # the build environment (e.g. `<module 'pickle' from '/leaked/path').

    """save(obj, f, pickle_module=pickle, pickle_protocol=DEFAULT_PROTOCOL, _use_new_zipfile_serialization=True)

    Saves an object to a disk file.

    See also: :ref:`saving-loading-tensors`

    Args:
        obj: saved object
        f: a file-like object (has to implement write and flush) or a string or
           os.PathLike object containing a file name
        pickle_module: module used for pickling metadata and objects
        pickle_protocol: can be specified to override the default protocol

    .. note::
        A common PyTorch convention is to save tensors using .pt file extension.

    """
    # 记录使用 torch.save API 的调用情况
    torch._C._log_api_usage_once("torch.save")
    # 检查 dill 库的版本是否符合要求
    _check_dill_version(pickle_module)
    # 检查保存文件对象的类型和属性
    _check_save_filelike(f)

    # 如果选择使用新的基于 zipfile 的序列化格式
    if _use_new_zipfile_serialization:
        # 打开文件作为 zipfile 格式写入器
        with _open_zipfile_writer(f) as opened_zipfile:
            # 调用 _save 函数保存对象到 zipfile 中
            _save(
                obj,
                opened_zipfile,
                pickle_module,
                pickle_protocol,
                _disable_byteorder_record,
            )
            # 函数执行完毕，返回
            return
    else:
        # 如果选择使用旧的序列化格式，打开文件作为二进制写入器
        with _open_file_like(f, "wb") as opened_file:
            # 调用 _legacy_save 函数保存对象到文件中
            _legacy_save(obj, opened_file, pickle_module, pickle_protocol)
# 将对象序列化保存到 ZIP 文件中，用于旧版本的保存格式
def _legacy_save(obj, f, pickle_module, pickle_protocol) -> None:
    # 导入 PyTorch 的神经网络模块
    import torch.nn as nn

    # 用于存储序列化后的容器类型
    serialized_container_types = {}
    # 用于存储序列化后的存储器对象
    serialized_storages = {}

    # 由于不支持加载视图相同数据但数据类型不同的存储器，需要跟踪每个存储器数据指针关联的数据类型，
    # 如果数据类型发生变化则抛出错误。TODO: 可能在未来添加此功能。
    storage_dtypes: Dict[int, torch.dtype] = {}

    # 系统信息包括协议版本、字节顺序和类型大小信息的字典
    sys_info = dict(
        protocol_version=PROTOCOL_VERSION,
        little_endian=sys.byteorder == "little",
        type_sizes=dict(
            short=SHORT_SIZE,
            int=INT_SIZE,
            long=LONG_SIZE,
        ),
    )

    # 使用 pickle_module 将魔数写入文件
    pickle_module.dump(MAGIC_NUMBER, f, protocol=pickle_protocol)
    # 使用 pickle_module 将协议版本写入文件
    pickle_module.dump(PROTOCOL_VERSION, f, protocol=pickle_protocol)
    # 使用 pickle_module 将系统信息字典写入文件
    pickle_module.dump(sys_info, f, protocol=pickle_protocol)
    # 创建 Pickler 对象，并设置持久性 ID
    pickler = pickle_module.Pickler(f, protocol=pickle_protocol)
    pickler.persistent_id = persistent_id
    # 将对象 obj 序列化保存到文件
    pickler.dump(obj)

    # 对序列化后的存储键进行排序
    serialized_storage_keys = sorted(serialized_storages.keys())
    # 使用 pickle_module 将序列化后的存储键写入文件
    pickle_module.dump(serialized_storage_keys, f, protocol=pickle_protocol)
    # 刷新文件缓冲区
    f.flush()
    # 遍历存储键列表
    for key in serialized_storage_keys:
        # 获取存储器对象和数据类型
        storage, dtype = serialized_storages[key]
        # 将存储器对象的数据写入文件
        storage._write_file(
            f, _should_read_directly(f), True, torch._utils._element_size(dtype)
        )


# 将对象序列化保存到 ZIP 文件中，用于新版本的保存格式
def _save(obj, zip_file, pickle_module, pickle_protocol, _disable_byteorder_record):
    # 用于存储序列化后的存储器对象
    serialized_storages = {}
    # 用于存储 ID 映射关系的字典
    id_map: Dict[int, str] = {}

    # 由于不支持加载视图相同数据但数据类型不同的存储器，需要跟踪每个存储器数据指针关联的数据类型，
    # 如果数据类型发生变化则抛出错误。TODO: 可能在未来添加此功能。
    storage_dtypes: Dict[int, torch.dtype] = {}
    # 定义一个函数 persistent_id，用于处理对象的持久化标识
    def persistent_id(obj):
        # FIXME: 文档中指出 persistent_id 应该只返回一个字符串，但 torch 存储返回元组。这仅适用于二进制协议
        # 参考
        # https://docs.python.org/2/library/pickle.html#pickling-and-unpickling-external-objects
        # https://github.com/python/cpython/blob/master/Lib/pickle.py#L527-L537
        if isinstance(obj, torch.storage.TypedStorage) or torch.is_storage(obj):
            if isinstance(obj, torch.storage.TypedStorage):
                # TODO: 一旦我们决定中断序列化 FC，这种情况可以删除
                storage = obj._untyped_storage
                storage_dtype = obj.dtype
                storage_type_str = obj._pickle_storage_type()
                storage_type = getattr(torch, storage_type_str)
                storage_numel = obj._size()

            else:
                storage = obj
                storage_dtype = torch.uint8
                storage_type = normalize_storage_type(type(obj))
                storage_numel = storage.nbytes()

            # 如果存储已分配，则确保任何指向相同数据的其他保存存储都具有相同的 dtype。如果存储未分配，则不执行此检查
            if storage.data_ptr() != 0:
                if storage.data_ptr() in storage_dtypes:
                    if storage_dtype != storage_dtypes[storage.data_ptr()]:
                        raise RuntimeError(
                            "Cannot save multiple tensors or storages that "
                            "view the same data as different types"
                        )
                else:
                    storage_dtypes[storage.data_ptr()] = storage_dtype

            storage_key = id_map.setdefault(storage._cdata, str(len(id_map)))
            location = location_tag(storage)
            serialized_storages[storage_key] = storage

            return ("storage", storage_type, storage_key, location, storage_numel)

        return None

    # 创建一个字节流对象 data_buf
    data_buf = io.BytesIO()
    # 创建一个 Pickler 对象 pickler，用于将对象序列化到 data_buf 中
    pickler = pickle_module.Pickler(data_buf, protocol=pickle_protocol)
    # 设置 pickler 的 persistent_id 方法为上面定义的 persistent_id 函数
    pickler.persistent_id = persistent_id
    # 将对象 obj 序列化到 data_buf 中
    pickler.dump(obj)
    # 获取序列化后的数据值
    data_value = data_buf.getvalue()
    # 将序列化后的数据值写入 zip 文件中的 "data.pkl" 记录
    zip_file.write_record("data.pkl", data_value, len(data_value))

    # 写入字节顺序标记
    if not _disable_byteorder_record:
        if sys.byteorder not in ["little", "big"]:
            raise ValueError("Unknown endianness type: " + sys.byteorder)

        # 将系统的字节顺序信息写入 zip 文件中的 "byteorder" 记录
        zip_file.write_record("byteorder", sys.byteorder, len(sys.byteorder))

    # 将每个张量写入 zip 存档中名为 tensor/the_tensor_key 的文件
    # 遍历已序列化存储对象的键，并按键排序
    for key in sorted(serialized_storages.keys()):
        # 构建文件名，格式为"data/{key}"
        name = f"data/{key}"
        # 获取当前键对应的存储对象
        storage = serialized_storages[key]
        
        # 检查存储对象的设备类型是否为非CPU
        if storage.device.type != "cpu":
            # 如果不是CPU类型，则将存储对象移到CPU上
            storage = storage.cpu()
        
        # 计算存储对象占用的字节数
        num_bytes = storage.nbytes()
        
        # 将存储对象及其相关信息写入到ZIP文件中
        zip_file.write_record(name, storage, num_bytes)
# 加载函数，用于从文件中加载由 torch.save 保存的对象
def load(
    f: FILE_LIKE,
    map_location: MAP_LOCATION = None,
    pickle_module: Any = None,
    *,
    weights_only: Optional[bool] = None,
    mmap: Optional[bool] = None,
    **pickle_load_args: Any,
) -> Any:
    # 参考链接：https://github.com/pytorch/pytorch/issues/54354
    # 此文档字符串的第一行覆盖了 Sphinx 为文档生成的默认行为。我们需要这样做是为了避免 Sphinx 泄露构建环境中 pickle 模块的路径（例如 `<module 'pickle' from '/leaked/path'>`）。

    """load(f, map_location=None, pickle_module=pickle, *, weights_only=False, mmap=None, **pickle_load_args)

    从文件中加载由 torch.save 保存的对象。

    :func:`torch.load` 使用 Python 的反序列化功能，但对底层张量的存储进行了特殊处理。
    它们首先在 CPU 上反序列化，然后移动到它们保存时的设备上。
    如果这失败（例如因为运行时系统缺少某些设备），则会引发异常。
    但是，存储可以使用 :attr:`map_location` 参数动态重新映射到另一组设备上。

    如果 :attr:`map_location` 是一个可调用对象，则会为每个序列化的存储调用一次。
    它有两个参数：存储和位置。
    存储参数是存储在 CPU 上的初始反序列化存储。
    每个序列化的存储都有与之相关的位置标签，该标签标识了它保存时的设备，而这个标签是传递给 :attr:`map_location` 的第二个参数。
    内置的位置标签包括 'cpu'（用于 CPU 张量）和 'cuda:device_id'（例如 'cuda:2'，用于 CUDA 张量）。
    :attr:`map_location` 应该返回 None 或一个存储对象。
    如果 :attr:`map_location` 返回一个存储对象，则会将其用作最终的反序列化对象，已经移动到正确的设备上。
    否则，:func:`torch.load` 将退回到默认行为，就像没有指定 :attr:`map_location` 一样。

    如果 :attr:`map_location` 是一个 :class:`torch.device` 对象或包含设备标签的字符串，则表示应加载所有张量的位置。

    另外，如果 :attr:`map_location` 是一个字典，则将用它来重映射文件中出现的位置标签（键）到指定存储位置（值）。

    用户扩展可以使用 :func:`torch.serialization.register_package` 注册自己的位置标签、标记和反序列化方法。

    """
    Args:
        f: a file-like object (has to implement :meth:`read`, :meth:`readline`, :meth:`tell`, and :meth:`seek`),
            or a string or os.PathLike object containing a file name
            # 文件对象，要求实现 :meth:`read`, :meth:`readline`, :meth:`tell`, 和 :meth:`seek` 方法，或者是包含文件名的字符串或 os.PathLike 对象
        map_location: a function, :class:`torch.device`, string or a dict specifying how to remap storage
            locations
            # 一个函数、:class:`torch.device`、字符串或字典，指定如何重新映射存储位置
        pickle_module: module used for unpickling metadata and objects (has to
            match the :attr:`pickle_module` used to serialize file)
            # 用于解析元数据和对象的模块（必须与序列化文件时使用的 :attr:`pickle_module` 匹配）
        weights_only: Indicates whether unpickler should be restricted to
            loading only tensors, primitive types, dictionaries
            and any types added via :func:`torch.serialization.add_safe_globals`.
            # 表示解析器是否应限制仅加载张量、基本类型、字典以及通过 :func:`torch.serialization.add_safe_globals` 添加的任何类型
        mmap: Indicates whether the file should be mmaped rather than loading all the storages into memory.
            Typically, tensor storages in the file will first be moved from disk to CPU memory, after which they
            are moved to the location that they were tagged with when saving, or specified by ``map_location``. This
            second step is a no-op if the final location is CPU. When the ``mmap`` flag is set, instead of copying the
            tensor storages from disk to CPU memory in the first step, ``f`` is mmaped.
            # 表示文件是否应该使用 mmap 而不是将所有存储加载到内存中。通常，文件中的张量存储首先从磁盘移动到 CPU 内存，然后移动到保存时标记的位置，或由 ``map_location`` 指定。如果最终位置是 CPU，则第二步是一个无操作。当设置了 ``mmap`` 标志时，第一步中不会将张量存储从磁盘复制到 CPU 内存，而是使用 mmap。
        pickle_load_args: (Python 3 only) optional keyword arguments passed over to
            :func:`pickle_module.load` and :func:`pickle_module.Unpickler`, e.g.,
            :attr:`errors=...`.
            # （仅适用于 Python 3）传递给 :func:`pickle_module.load` 和 :func:`pickle_module.Unpickler` 的可选关键字参数，例如，:attr:`errors=...`。

    .. warning::
        :func:`torch.load()` unless `weights_only` parameter is set to `True`,
        uses ``pickle`` module implicitly, which is known to be insecure.
        It is possible to construct malicious pickle data which will execute arbitrary code
        during unpickling. Never load data that could have come from an untrusted
        source in an unsafe mode, or that could have been tampered with. **Only load data you trust**.
        # 警告：除非将 `weights_only` 参数设置为 `True`，否则 :func:`torch.load()` 使用隐式的 ``pickle`` 模块，这是不安全的。可能构造恶意 pickle 数据，在反序列化期间执行任意代码。永远不要在不安全模式下加载可能来自不受信任源或可能已被篡改的数据。**只加载信任的数据**。

    .. note::
        When you call :func:`torch.load()` on a file which contains GPU tensors, those tensors
        will be loaded to GPU by default. You can call ``torch.load(.., map_location='cpu')``
        and then :meth:`load_state_dict` to avoid GPU RAM surge when loading a model checkpoint.
        # 注意：当在包含 GPU 张量的文件上调用 :func:`torch.load()` 时，默认情况下会将这些张量加载到 GPU。您可以调用 ``torch.load(.., map_location='cpu')``，然后使用 :meth:`load_state_dict` 来避免在加载模型检查点时 GPU RAM 激增。

    .. note::
        By default, we decode byte strings as ``utf-8``.  This is to avoid a common error
        case ``UnicodeDecodeError: 'ascii' codec can't decode byte 0x...``
        when loading files saved by Python 2 in Python 3.  If this default
        is incorrect, you may use an extra :attr:`encoding` keyword argument to specify how
        these objects should be loaded, e.g., :attr:`encoding='latin1'` decodes them
        to strings using ``latin1`` encoding, and :attr:`encoding='bytes'` keeps them
        as byte arrays which can be decoded later with ``byte_array.decode(...)``.
        # 注意：默认情况下，我们将字节字符串解码为 ``utf-8``。这是为了避免加载由 Python 2 保存的文件时出现常见的错误，例如 ``UnicodeDecodeError: 'ascii' codec can't decode byte 0x...``。如果此默认值不正确，可以使用额外的 :attr:`encoding` 关键字参数来指定如何加载这些对象，例如 :attr:`encoding='latin1'` 将使用 ``latin1`` 编码将它们解码为字符串，而 :attr:`encoding='bytes'` 将它们保留为字节数组，稍后可以用 ``byte_array.decode(...)`` 解码。
    torch._C._log_api_usage_once("torch.load")
    # 记录一次 torch.load 的 API 使用情况

    UNSAFE_MESSAGE = (
        "Re-running `torch.load` with `weights_only` set to `False` will likely succeed, "
        "but it can result in arbitrary code execution. Do it only if you got the file from a "
        "trusted source."
    )
    # 当 `weights_only` 设置为 False 重新运行 `torch.load` 可能会成功，但可能导致任意代码执行的不安全消息

    DOCS_MESSAGE = (
        "\n\nCheck the documentation of torch.load to learn more about types accepted by default with "
        "weights_only https://pytorch.org/docs/stable/generated/torch.load.html."
    )
    # 检查 torch.load 文档以了解默认接受的类型和 weights_only 的更多信息

    def _get_wo_message(message: str) -> str:
        pattern = r"GLOBAL (\S+) was not an allowed global by default."
        # 匹配消息中不允许的全局变量模式

        has_unsafe_global = re.search(pattern, message) is not None
        # 检查消息中是否存在不安全的全局变量

        if has_unsafe_global:
            updated_message = (
                "Weights only load failed. This file can still be loaded, to do so you have two options "
                f"\n\t(1) {UNSAFE_MESSAGE}\n\t(2) Alternatively, to load with `weights_only=True` please check "
                "the recommended steps in the following error message.\n\tWeightsUnpickler error: "
                + message
            )
        else:
            updated_message = (
                f"Weights only load failed. {UNSAFE_MESSAGE}\n Please file an issue with the following "
                "so that we can make `weights_only=True` compatible with your use case: WeightsUnpickler "
                "error: " + message
            )
        return updated_message + DOCS_MESSAGE
        # 返回更新的错误消息和文档链接

    if weights_only is None:
        weights_only, warn_weights_only = False, True
    else:
        warn_weights_only = False
    # 如果 weights_only 为 None，则将其设置为 False，并设置 warn_weights_only 为 True；否则将 warn_weights_only 设置为 False

    # 添加通过环境变量强制仅安全加载权重的能力
    if os.getenv("TORCH_FORCE_WEIGHTS_ONLY_LOAD", "0").lower() in [
        "1",
        "y",
        "yes",
        "true",
    ]:
        # 如果环境变量 TORCH_FORCE_WEIGHTS_ONLY_LOAD 的值为 "1", "y", "yes", "true" 中的一个
        # 表示强制仅加载安全的权重
        ```
    ]:
        weights_only = True

这段代码看起来有误，应该是以下代码的一部分，正常的情况应该是这样：


    if weights_only:


这段代码检查 `weights_only` 变量是否为真，如果是，则执行以下逻辑。


        if pickle_module is not None:
            raise RuntimeError(
                "Can not safely load weights when explicit pickle_module is specified"
            )

如果 `weights_only` 为真，这段代码检查 `pickle_module` 是否为 `None`，如果不是，抛出 `RuntimeError` 异常，提示不能安全地加载权重数据。


    else:

如果 `weights_only` 不为真，则执行以下逻辑。


        if pickle_module is None:

在这个分支下，检查 `pickle_module` 是否为 `None`。


            if warn_weights_only:
                warnings.warn(
                    "You are using `torch.load` with `weights_only=False` (the current default value), which uses "
                    "the default pickle module implicitly. It is possible to construct malicious pickle data "
                    "which will execute arbitrary code during unpickling (See "
                    "https://github.com/pytorch/pytorch/blob/main/SECURITY.md#untrusted-models for more details). "
                    "In a future release, the default value for `weights_only` will be flipped to `True`. This "
                    "limits the functions that could be executed during unpickling. Arbitrary objects will no "
                    "longer be allowed to be loaded via this mode unless they are explicitly allowlisted by the "
                    "user via `torch.serialization.add_safe_globals`. We recommend you start setting "
                    "`weights_only=True` for any use case where you don't have full control of the loaded file. "
                    "Please open an issue on GitHub for any issues related to this experimental feature.",
                    FutureWarning,
                    stacklevel=2,
                )

如果 `pickle_module` 是 `None`，并且 `warn_weights_only` 为真，则发出警告。警告说明了当前使用 `torch.load` 函数且 `weights_only=False` 的默认值可能存在安全风险，建议将 `weights_only` 设置为 `True` 以限制在反序列化过程中可能执行的功能。


            pickle_module = pickle

如果 `pickle_module` 是 `None`，则将其设置为默认的 `pickle` 模块。


    # make flipping default BC-compatible
    if mmap is None:
        mmap = False

检查 `mmap` 变量是否为 `None`，如果是，则将其设置为 `False`，确保向后兼容性。


    _check_dill_version(pickle_module)

调用 `_check_dill_version` 函数，传递 `pickle_module` 作为参数，用于检查 `dill` 库的版本兼容性。


    if "encoding" not in pickle_load_args.keys():
        pickle_load_args["encoding"] = "utf-8"

检查 `pickle_load_args` 字典中是否缺少 `encoding` 键，如果是，则设置其默认值为 `"utf-8"`，用于指定反序列化时使用的编码方式。
# 为布局实例（如 torch.sparse_coo 等）注册 pickle 支持
def _get_layout(name):
    """从字符串表示中获取布局扩展对象。"""
    # 检查缓存是否存在，如果不存在则初始化
    cache = _get_layout.cache  # type: ignore[attr-defined]
    if not cache:
        # 遍历 torch.__dict__.values() 中的每个对象，寻找 torch.layout 类型的实例并缓存
        for v in torch.__dict__.values():
            if isinstance(v, torch.layout):
                cache[str(v)] = v
    # 根据名称返回对应的布局对象
    return cache[name]


# 目前还没有一种良好的方式来对函数属性进行类型注解 https://github.com/python/mypy/issues/2087
_get_layout.cache = {}  # type: ignore[attr-defined]

# 为 torch.layout 类型注册 pickle 支持函数
copyreg.pickle(torch.layout, lambda obj: (_get_layout, (str(obj),)))


def _legacy_load(f, map_location, pickle_module, **pickle_load_args):
    # 存储反序列化对象的字典，键为对象的 ID，值为对象本身
    deserialized_objects: Dict[int, Any] = {}

    # 根据 map_location 获取恢复位置
    restore_location = _get_restore_location(map_location)

    # UnpicklerWrapper 类，继承自 pickle_module.Unpickler，用于自定义类查找行为
    class UnpicklerWrapper(pickle_module.Unpickler):  # type: ignore[name-defined]
        # 自定义 find_class 方法，用于查找类
        def find_class(self, mod_name, name):
            # 如果 name 是字符串且包含 "Storage"，尝试创建 StorageType 类型的实例
            if type(name) is str and "Storage" in name:
                try:
                    return StorageType(name)
                except KeyError:
                    pass
            # 否则调用父类的查找类方法
            return super().find_class(mod_name, name)
    # 检查容器的源代码是否与原始源代码一致
    def _check_container_source(container_type, source_file, original_source):
        try:
            # 获取当前容器类型的源代码行，并拼接成字符串
            current_source = "".join(get_source_lines_and_file(container_type)[0])
        except Exception:  # 保存源代码是可选的，因此可以忽略任何错误
            # 如果无法获取容器类型的源代码，发出警告
            warnings.warn(
                "Couldn't retrieve source code for container of "
                "type " + container_type.__name__ + ". It won't be checked "
                "for correctness upon loading."
            )
            return
        
        # 如果原始源代码与当前源代码不一致
        if original_source != current_source:
            # 如果容器类型启用了 dump_patches
            if container_type.dump_patches:
                # 构造补丁文件名
                file_name = container_type.__name__ + ".patch"
                # 使用 difflib 生成原始源代码与当前源代码的差异
                diff = difflib.unified_diff(
                    current_source.split("\n"),
                    original_source.split("\n"),
                    source_file,
                    source_file,
                    lineterm="",
                )
                lines = "\n".join(diff)
                try:
                    # 尝试将差异写入补丁文件
                    with open(file_name, "a+") as f:
                        file_size = f.seek(0, 2)
                        f.seek(0)
                        if file_size == 0:
                            f.write(lines)
                        elif file_size != len(lines) or f.read() != lines:
                            raise OSError
                    # 成功写入补丁文件后，构造成功消息
                    msg = (
                        "Saved a reverse patch to " + file_name + ". "
                        "Run `patch -p0 < " + file_name + "` to revert your "
                        "changes."
                    )
                except OSError:
                    # 如果写入补丁文件失败，构造失败消息
                    msg = (
                        "Tried to save a patch, but couldn't create a "
                        "writable file " + file_name + ". Make sure it "
                        "doesn't exist and your working directory is "
                        "writable."
                    )
            else:
                # 如果容器类型未启用 dump_patches，构造备用消息
                msg = (
                    "you can retrieve the original source code by "
                    "accessing the object's source attribute or set "
                    "`torch.nn.Module.dump_patches = True` and use the "
                    "patch tool to revert the changes."
                )
            
            # 构造包含容器类型名称和消息的警告消息
            msg = f"source code of class '{torch.typename(container_type)}' has changed. {msg}"
            # 发出源码变更的警告
            warnings.warn(msg, SourceChangeWarning)

    # 序列化对象的字典，用于存储反序列化后的对象
    deserialized_objects = {}
    # 定义一个函数 persistent_load，用于反序列化加载保存的对象
    def persistent_load(saved_id):
        # 断言 saved_id 是一个元组
        assert isinstance(saved_id, tuple)
        # 解析 typename，即保存的对象类型名
        typename = _maybe_decode_ascii(saved_id[0])
        # 获取数据部分
        data = saved_id[1:]

        # 如果 typename 是 "module"
        if typename == "module":
            # 如果所有的 data[1:] 都为真值（非空），则检查容器的来源
            if all(data[1:]):
                _check_container_source(*data)
            # 返回 data[0]，即模块对象
            return data[0]
        # 如果 typename 是 "storage"
        elif typename == "storage":
            # 解析 storage 类型、根键、位置、元素数目和视图元数据
            storage_type, root_key, location, numel, view_metadata = data
            # 将 location 解码为 ASCII 字符串
            location = _maybe_decode_ascii(location)
            # 获取 dtype
            dtype = storage_type.dtype

            # 计算数据的字节大小
            nbytes = numel * torch._utils._element_size(dtype)

            # 如果 root_key 不在 deserialized_objects 中
            if root_key not in deserialized_objects:
                # 如果处于活动的伪造模式中
                if torch._guards.active_fake_mode() is not None:
                    # 创建一个未类型化的 Storage 对象，设备为 "meta"
                    obj = cast(Storage, torch.UntypedStorage(nbytes, device="meta"))
                else:
                    # 创建一个未类型化的 Storage 对象
                    obj = cast(Storage, torch.UntypedStorage(nbytes))
                    # 设置 _torch_load_uninitialized 属性为 True
                    obj._torch_load_uninitialized = True
                    # 恢复对象的位置信息
                    obj = restore_location(obj, location)
                # 包装为 TypedStorage，用于序列化后兼容性
                typed_storage = torch.storage.TypedStorage(
                    wrap_storage=obj, dtype=dtype, _internal=True
                )
                # 将 typed_storage 添加到 deserialized_objects 中
                deserialized_objects[root_key] = typed_storage
            else:
                # 如果 root_key 已经在 deserialized_objects 中，直接取出
                typed_storage = deserialized_objects[root_key]
                # 如果 typed_storage 的 _data_ptr() 为 0
                if typed_storage._data_ptr() == 0:
                    # 创建一个新的 TypedStorage 对象，保持设备和数据类型
                    typed_storage = torch.storage.TypedStorage(
                        device=typed_storage._untyped_storage.device,
                        dtype=dtype,
                        _internal=True,
                    )

            # 如果存在视图元数据
            if view_metadata is not None:
                # 解析视图的键、偏移量和视图大小
                view_key, offset, view_size = view_metadata
                # 计算偏移量和视图大小的字节大小
                offset_bytes = offset * torch._utils._element_size(dtype)
                view_size_bytes = view_size * torch._utils._element_size(dtype)
                # 如果视图的键不在 deserialized_objects 中
                if view_key not in deserialized_objects:
                    # 创建一个新的 TypedStorage 对象，包含视图数据
                    deserialized_objects[view_key] = torch.storage.TypedStorage(
                        wrap_storage=typed_storage._untyped_storage[
                            offset_bytes : offset_bytes + view_size_bytes
                        ],
                        dtype=dtype,
                        _internal=True,
                    )
                # 返回视图的 TypedStorage 对象
                res = deserialized_objects[view_key]
            else:
                # 否则返回 typed_storage
                res = typed_storage
            # 返回结果对象 res
            return res
        else:
            # 如果 typename 不是 "module" 或 "storage"，抛出运行时错误
            raise RuntimeError(f"Unknown saved id type: {saved_id[0]}")

    # 检查文件是否可寻址的函数 _check_seekable
    _check_seekable(f)
    # 检查是否应该直接读取文件的函数 _should_read_directly
    f_should_read_directly = _should_read_directly(f)
    if f_should_read_directly and f.tell() == 0:
        # 如果需要直接读取，并且文件指针位于起始位置
        # legacy_load 需要 f 具有 fileno()
        # 只有在偏移量为零时，我们才能尝试使用旧版 tar 文件加载器
        try:
            # 尝试使用旧版加载器加载文件
            return legacy_load(f)
        except tarfile.TarError:
            if _is_zipfile(f):
                # 如果是一个 ZIP 文件，则抛出运行时错误，用于 torch.jit.save 并且会在这里抛出一个非反序列化错误
                raise RuntimeError(
                    f"{f.name} is a zip archive (did you mean to use torch.jit.load()?)"
                ) from None
            # 如果不是一个 tar 文件，则重置文件偏移量并继续
            f.seek(0)

    if not hasattr(f, "readinto") and (3, 8, 0) <= sys.version_info < (3, 8, 2):
        # 如果 f 没有 readinto 属性，并且 Python 版本在 3.8.0 到 3.8.1 之间
        raise RuntimeError(
            "torch.load does not work with file-like objects that do not implement readinto on Python 3.8.0 and 3.8.1. "
            f'Received object of type "{type(f)}". Please update to Python 3.8.2 or newer to restore this '
            "functionality."
        )

    magic_number = pickle_module.load(f, **pickle_load_args)
    # 加载魔数
    if magic_number != MAGIC_NUMBER:
        # 如果魔数不匹配，则抛出运行时错误，表示文件损坏
        raise RuntimeError("Invalid magic number; corrupt file?")
    protocol_version = pickle_module.load(f, **pickle_load_args)
    # 加载协议版本号
    if protocol_version != PROTOCOL_VERSION:
        # 如果协议版本号不匹配，则抛出运行时错误
        raise RuntimeError(f"Invalid protocol version: {protocol_version}")

    _sys_info = pickle_module.load(f, **pickle_load_args)
    # 加载系统信息
    unpickler = UnpicklerWrapper(f, **pickle_load_args)
    unpickler.persistent_load = persistent_load
    # 创建解封装器，并设置持久加载函数
    result = unpickler.load()
    # 加载对象

    deserialized_storage_keys = pickle_module.load(f, **pickle_load_args)
    # 加载反序列化存储的键

    if torch._guards.active_fake_mode() is None:
        offset = f.tell() if f_should_read_directly else None
        # 如果没有活动的假模式，根据需要设置偏移量
        for key in deserialized_storage_keys:
            assert key in deserialized_objects
            typed_storage = deserialized_objects[key]
            # 对每个键进行验证，然后从文件设置类型化存储
            typed_storage._untyped_storage._set_from_file(
                f,
                offset,
                f_should_read_directly,
                torch._utils._element_size(typed_storage.dtype),
            )
            if offset is not None:
                offset = f.tell()

    torch._utils._validate_loaded_sparse_tensors()
    # 验证加载的稀疏张量

    return result
    # 返回加载的结果
# 定义一个函数用于将字节字符串或字符串转换为字符串，以适应 Py3 中使用 encoding='bytes' 时的情况
# 在 Py3 中，某些在 Py2 中作为字符串存储的内部键，在加载时作为字节加载。此函数使用 ascii 编码进行解码，这是 Py3 默认使用的编码方式。
# 
# 注意：此函数仅应用于内部键（例如下面的 `persistent_load` 中的 `typename` 和 `location`！）
def _maybe_decode_ascii(bytes_str: Union[bytes, str]) -> str:
    if isinstance(bytes_str, bytes):
        return bytes_str.decode("ascii")
    return bytes_str


# 根据 map_location 返回恢复位置的函数
def _get_restore_location(map_location):
    if map_location is None:
        # 如果 map_location 为 None，则使用默认的恢复位置
        restore_location = default_restore_location
    elif isinstance(map_location, dict):
        # 如果 map_location 是字典类型，则定义一个根据指定位置恢复的函数
        def restore_location(storage, location):
            location = map_location.get(location, location)
            return default_restore_location(storage, location)

    elif isinstance(map_location, (str, bytes)):
        # 如果 map_location 是字符串或字节串类型，则定义一个根据指定位置恢复的函数
        def restore_location(storage, location):
            return default_restore_location(storage, map_location)

    elif isinstance(map_location, torch.device):
        # 如果 map_location 是 torch.device 类型，则定义一个根据指定位置恢复的函数
        def restore_location(storage, location):
            return default_restore_location(storage, str(map_location))

    else:
        # 其他情况下，定义一个根据指定位置恢复的函数，并尝试使用 map_location 函数进行恢复
        def restore_location(storage, location):
            result = map_location(storage, location)
            if result is None:
                result = default_restore_location(storage, location)
            return result

    return restore_location


# 定义一个存储类型的类
class StorageType:
    def __init__(self, name):
        # 根据名称获取存储类型的数据类型
        self._dtype = _get_dtype_from_pickle_storage_type(name)

    @property
    def dtype(self):
        # 返回存储类型的数据类型
        return self._dtype

    def __str__(self):
        # 返回描述存储类型的字符串
        return f"StorageType(dtype={self.dtype})"


# 定义一个加载函数
def _load(
    zip_file,
    map_location,
    pickle_module,
    pickle_file="data.pkl",
    overall_storage=None,
    **pickle_load_args,
):
    # 根据 map_location 获取恢复位置的函数
    restore_location = _get_restore_location(map_location)

    # 存储加载后的数据的字典
    loaded_storages = {}

    # 检查是否需要进行字节顺序的调整
    byteordername = "byteorder"
    byteorderdata = None
    if zip_file.has_record(byteordername):
        # 如果 ZIP 文件包含字节顺序的记录，则获取该记录的数据
        byteorderdata = zip_file.get_record(byteordername)
        if byteorderdata not in [b"little", b"big"]:
            # 如果记录的数据不是 "little" 或 "big"，则抛出异常
            raise ValueError("Unknown endianness type: " + byteorderdata.decode())
    elif (
        get_default_load_endianness() == LoadEndianness.LITTLE
        or get_default_load_endianness() is None
    ):
        # 如果没有字节顺序的记录，并且默认的加载顺序为小端或未定义，则使用小端字节顺序
        byteorderdata = b"little"
    elif get_default_load_endianness() == LoadEndianness.BIG:
        # 如果默认的加载顺序为大端，则使用大端字节顺序
        byteorderdata = b"big"
    elif get_default_load_endianness() == LoadEndianness.NATIVE:
        # 如果默认的加载顺序为本机字节顺序，则不做任何操作
        pass
    else:
        # 如果加载顺序类型无效，则抛出异常
        raise ValueError("Invalid load endianness type")

    if (
        not zip_file.has_record(byteordername)
        and get_default_load_endianness() is None
        and sys.byteorder == "big"
    ):
        # 如果没有字节顺序的记录，并且默认的加载顺序未定义，并且系统字节顺序为大端，则执行以下操作
        # （此处应该还有一些代码，但由于截断原因，无法完全显示）
    ):
        # 如果遇到默认行为变更
        # 参见 https://github.com/pytorch/pytorch/issues/101688
        # 发出警告，指出在大端机器上没有字节顺序标记的检查点的默认加载顺序已从“native”更改为“little”端
        # 如果想避免此行为，请使用 torch.serialization.set_default_load_endianness
        # 设置所需的默认加载顺序
        warnings.warn(
            "The default load endianness for checkpoints without a byteorder mark "
            "on big endian machines was changed from 'native' to 'little' endian, "
            "to avoid this behavior please use "
            "torch.serialization.set_default_load_endianness to set "
            "the desired default load endianness",
            UserWarning,
        )

    def load_tensor(dtype, numel, key, location):
        # 构造存储名称
        name = f"data/{key}"
        # 检测是否处于假模式
        if torch._guards.detect_fake_mode(None) is not None:
            # 计算所需字节数
            nbytes = numel * torch._utils._element_size(dtype)
            # 创建未类型化存储对象
            storage = torch.UntypedStorage(nbytes, device="meta")
        elif overall_storage is not None:
            # 获取记录偏移量
            storage_offset = zip_file.get_record_offset(name)
            # 从整体存储中获取数据
            storage = overall_storage[storage_offset : storage_offset + numel]
        else:
            # 从记录中获取存储，并转换为未类型化存储
            storage = (
                zip_file.get_storage_from_record(name, numel, torch.UntypedStorage)
                ._typed_storage()
                ._untyped_storage
            )
        # 如果需要字节交换，则在此处进行交换
        if byteorderdata is not None:
            if byteorderdata.decode() != sys.byteorder:
                storage.byteswap(dtype)

        # TODO: 一旦决定中断序列化 FC，可以停止使用 TypedStorage 包装
        # 使用 TypedStorage 包装存储，并返回
        typed_storage = torch.storage.TypedStorage(
            wrap_storage=restore_location(storage, location),
            dtype=dtype,
            _internal=True,
        )

        # 如果数据指针不为 0，则存储到 loaded_storages 中
        if typed_storage._data_ptr() != 0:
            loaded_storages[key] = typed_storage

        return typed_storage

    def persistent_load(saved_id):
        # 断言 saved_id 是元组类型
        assert isinstance(saved_id, tuple)
        # 解码类型名称为 ASCII 字符串
        typename = _maybe_decode_ascii(saved_id[0])
        # 获取数据部分
        data = saved_id[1:]

        # 断言类型名称为 "storage"
        assert (
            typename == "storage"
        ), f"Unknown typename for persistent_load, expected 'storage' but got '{typename}'"
        # 解析存储类型、键、位置和元素数
        storage_type, key, location, numel = data
        # 确定存储数据类型
        if storage_type is torch.UntypedStorage:
            dtype = torch.uint8
        else:
            dtype = storage_type.dtype

        # 如果键已加载，则使用已加载的存储
        if key in loaded_storages:
            typed_storage = loaded_storages[key]
        else:
            # 计算所需字节数
            nbytes = numel * torch._utils._element_size(dtype)
            # 调用 load_tensor 加载存储
            typed_storage = load_tensor(
                dtype, nbytes, key, _maybe_decode_ascii(location)
            )

        return typed_storage

    load_module_mapping: Dict[str, str] = {
        # 参见 https://github.com/pytorch/pytorch/pull/51633
        # 将 "torch.tensor" 映射到 "torch._tensor"
        "torch.tensor": "torch._tensor"
    }

    # 需要子类化 Unpickler，而不是直接对 find_class 方法进行猴子补丁
    # 因为 pickle 中标记为只读
    # type: ignore 是因为 mypy 无法静态确定此类的类型。
    # 定义一个自定义的 UnpicklerWrapper 类，继承自 pickle_module.Unpickler
    # 这个类允许我们在反序列化对象时覆盖 pickle 使用的导入过程
    # 这在修改了张量实例化依赖的模块路径时非常有用，有助于保持向后兼容性
    class UnpicklerWrapper(pickle_module.Unpickler):  # type: ignore[name-defined]
        
        # 从 https://stackoverflow.com/questions/13398462/unpickling-python-objects-with-a-changed-module-path/13405732
        # 用于在反序列化时重载 pickle 使用的导入过程的方法
        def find_class(self, mod_name, name):
            # 如果 name 是字符串并且包含 "Storage"，尝试使用 StorageType 来返回类
            if type(name) is str and "Storage" in name:
                try:
                    return StorageType(name)
                except KeyError:
                    pass
            # 尝试使用 load_module_mapping 映射表来替换 mod_name
            mod_name = load_module_mapping.get(mod_name, mod_name)
            # 调用父类的 find_class 方法查找并返回类
            return super().find_class(mod_name, name)

    # 从 zip_file 中获取 pickle_file 对应的记录，并封装成 BytesIO 对象
    data_file = io.BytesIO(zip_file.get_record(pickle_file))

    # 创建 UnpicklerWrapper 对象，传入 data_file 和 pickle_load_args
    unpickler = UnpicklerWrapper(data_file, **pickle_load_args)
    # 设置 unpickler 的 persistent_load 属性为 persistent_load 函数
    unpickler.persistent_load = persistent_load

    # 设置 torch._utils._thread_local_state.map_location 为 map_location
    # 这是为了处理存储设备和重建张量设备不连接的情况（如使用 numpy 重建的张量）
    torch._utils._thread_local_state.map_location = map_location

    # 调用 unpickler 的 load 方法加载数据
    result = unpickler.load()

    # 删除 torch._utils._thread_local_state.map_location，清理资源
    del torch._utils._thread_local_state.map_location

    # 验证加载的稀疏张量
    torch._utils._validate_loaded_sparse_tensors()

    # 记录 API 使用元数据，包括 serialization_id
    torch._C._log_api_usage_metadata(
        "torch.load.metadata", {"serialization_id": zip_file.serialization_id()}
    )

    # 返回加载的结果对象
    return result
#`
# 检查 TorchScript ZIP 文件是否包含特定记录

def _is_torchscript_zip(zip_file):
    # 使用传入的 zip_file 对象的方法，检查其中是否包含名为 "constants.pkl" 的记录
    return "constants.pkl" in zip_file.get_all_records()
```