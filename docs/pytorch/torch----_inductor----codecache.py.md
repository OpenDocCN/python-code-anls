# `.\pytorch\torch\_inductor\codecache.py`

```
# 引入类型检查声明，允许未标记类型的定义
mypy: allow-untyped-defs

# 引入将来版本的特性
from __future__ import annotations

# 引入标准库模块
import base64            # 提供对Base64编码和解码的支持
import copyreg           # 支持自定义对象的序列化和反序列化
import dataclasses       # 提供数据类的支持
import functools         # 提供用于函数操作的工具，如偏函数等
import hashlib           # 提供安全哈希和摘要算法的支持
import importlib         # 提供用于动态加载模块的支持
import io                # 提供对I/O操作的核心工具
import json              # 提供JSON编码和解码的支持
import logging           # 提供灵活的日志记录功能
import os                # 提供与操作系统交互的功能
import pickle            # 提供Python对象的序列化和反序列化
import pkgutil           # 提供与包相关的实用工具函数
import platform          # 提供访问平台相关信息的功能
import re                # 提供正则表达式的支持
import shlex             # 提供用于解析和操作命令行字符串的支持
import shutil            # 提供高级文件操作功能
import struct            # 提供对Python中C结构的处理
import subprocess        # 提供启动和管理子进程的支持
import sys               # 提供对Python解释器的访问和控制
import sysconfig         # 提供对Python配置信息的访问
import tempfile          # 提供创建临时文件和目录的支持
import textwrap          # 提供简单的文本包装和填充功能
import threading         # 提供多线程编程的支持
import warnings          # 提供警告处理的功能
from bisect import bisect_right   # 提供二分查找功能
from copy import copy             # 提供复制对象的功能
from ctypes import c_void_p, cdll, CDLL   # 提供与C语言类型和库的互操作能力
from functools import partial    # 提供创建偏函数的能力
from pathlib import Path         # 提供处理文件路径的面向对象接口
from time import time, time_ns   # 提供时间相关功能
from types import ModuleType     # 提供操作Python模块对象的功能
from typing import (
    Any,                         # 引入通用类型注解
    Callable,                    # 引入可调用对象类型注解
    cast,                        # 引入类型强制转换工具
    Dict,                        # 引入字典类型注解
    Generator,                   # 引入生成器类型注解
    List,                        # 引入列表类型注解
    Optional,                    # 引入可选类型注解
    Sequence,                    # 引入序列类型注解
    Set,                         # 引入集合类型注解
    Tuple,                       # 引入元组类型注解
    TYPE_CHECKING,               # 引入类型检查常量
    Union,                       # 引入联合类型注解
)
from typing_extensions import TypeAlias   # 引入类型别名扩展支持

# 引入PyTorch库
import torch                      # 引入PyTorch深度学习框架
from torch._dynamo.utils import counters, dynamo_timed   # 引入Dynamo工具函数
from torch._inductor import config, exc, metrics         # 引入Inductor模块的配置、异常和指标支持
from torch._inductor.codegen.cuda import cuda_env        # 引入CUDA代码生成环境支持
from torch._inductor.codegen.rocm.compile_command import (
    rocm_compile_command, rocm_compiler,   # 引入ROCm编译命令和编译器支持
)

# 引入CppBuilder和相关选项
"""
codecache.py, cpp_builder.py and cpu_vec_isa.py import rule:
https://github.com/pytorch/pytorch/issues/124245#issuecomment-2197778902
"""
from torch._inductor.cpp_builder import (
    _set_gpu_runtime_env,                  # 引入设置GPU运行时环境函数
    _transform_cuda_paths,                 # 引入转换CUDA路径函数
    CppBuilder,                            # 引入CppBuilder类
    CppOptions,                            # 引入CppOptions类
    CppTorchCudaOptions,                   # 引入CppTorchCudaOptions类
    get_compiler_version_info,             # 引入获取编译器版本信息函数
)

# 引入CPU向量指令集相关模块
from torch._inductor.cpu_vec_isa import (
    invalid_vec_isa,                       # 引入无效向量指令集函数
    pick_vec_isa,                          # 引入选择向量指令集函数
    VecISA                                 # 引入VecISA类
)

# 引入编译任务相关模块
from torch._inductor.runtime.compile_tasks import (
    _module_to_triton_kernel,               # 引入将模块转换为Triton内核的函数
    _reload_python_module,                  # 引入重新加载Python模块的函数
    _reload_python_module_in_subproc,       # 引入在子进程中重新加载Python模块的函数
)

# 引入运行时工具函数
from torch._inductor.runtime.runtime_utils import (
    cache_dir,                              # 引入缓存目录函数
    default_cache_dir                       # 引入默认缓存目录函数
)

# 引入Inductor工具函数
from torch._inductor.utils import (
    ALIGN_BYTES,                            # 引入字节对齐常量
    clear_on_fresh_inductor_cache,          # 引入在新Inductor缓存上清除函数
    is_linux                                # 引入判断是否为Linux系统的函数
)

# 引入日志结构化模块
from torch._logging import trace_structured  # 引入结构化跟踪模块

# 引入虚假张量相关模块
from torch._subclasses.fake_tensor import (
    extract_tensor_metadata,                # 引入提取张量元数据函数
    FakeTensor,                             # 引入FakeTensor类
    TensorMetadata                         # 引入TensorMetadata类
)

# 引入符号形状实验性模块
from torch.fx.experimental.symbolic_shapes import (
    has_hint,                               # 引入判断是否有提示函数
    hint_int,                               # 引入提示整数函数
    ShapeEnv                                # 引入ShapeEnv类
)

# 如果类型检查开启，引入Future和其它相关类型
if TYPE_CHECKING:
    from concurrent.futures import Future   # 引入Future类

    from torch._inductor.graph import GraphLowering     # 引入图降低类
    from torch._inductor.ir import ChoiceCaller         # 引入ChoiceCaller类
    from torch._inductor.runtime.hints import (         # 引入Halide相关输入规范和元数据类
        HalideInputSpec, HalideMeta
    )


# 获取当前文件的绝对路径
_HERE = os.path.abspath(__file__)

# 获取torch所在目录的路径
_TORCH_PATH = os.path.dirname(os.path.dirname(_HERE))

# 构建链接脚本的路径
_LINKER_SCRIPT = os.path.join(_TORCH_PATH, "_inductor/script.ld")

# 检查当前系统是否为Windows
_IS_WINDOWS = sys.platform == "win32"

# 如果运行在FBCode环境中
if config.is_fbcode():
    # 从triton.fb中引入构建路径和运行构建命令的支持
    from triton.fb import build_paths
    from triton.fb.build import _run_build_command

    # 从torch._inductor.fb.utils中引入日志全局缓存错误、统计和值的支持，以及使用全局缓存的支持
    from torch._inductor.fb.utils import (
        log_global_cache_errors,
        log_global_cache_stats,
        log_global_cache_vals,
        use_global_cache,
    )
# 否则，定义空函数以防止未定义错误
else:

    def log_global_cache_errors(*args, **kwargs):
        pass

    def log_global_cache_stats(*args, **kwargs):
        pass

    def log_global_cache_vals(*args, **kwargs):
        pass
    # 定义一个函数 use_global_cache，返回布尔值 False
    def use_global_cache() -> bool:
        return False
output_code_log = torch._logging.getArtifactLogger(__name__, "output_code")
# 获取输出代码日志记录器对象

LOCK_TIMEOUT = 600
# 定义锁定超时时间为 600 秒

_IS_WINDOWS = sys.platform == "win32"
# 判断当前操作系统是否为 Windows

log = logging.getLogger(__name__)
# 获取当前模块的日志记录器对象

def cpp_wrapper_cache_dir(name: str) -> str:
    cu_str = (
        "cpu"
        if torch.version.cuda is None
        else f'cu{torch.version.cuda.replace(".", "")}'
    )
    python_version = f"py{sys.version_info.major}{sys.version_info.minor}"
    build_folder = f"{python_version}_{cu_str}"

    cpp_wrapper_dir = os.path.join(cache_dir(), build_folder)
    cpp_wrapper_build_directory = os.path.join(cpp_wrapper_dir, name)
    os.makedirs(cpp_wrapper_build_directory, exist_ok=True)
    return cpp_wrapper_build_directory
# 根据给定的名称创建 CPP 封装缓存目录，并返回该目录的路径

def get_cpp_wrapper_cubin_path_name():
    return "cubin_path" if torch.version.hip is None else "hsaco_path"
# 根据当前的 Torch 版本选择返回字符串 "cubin_path" 或 "hsaco_path"

class CacheBase:
    @staticmethod
    @functools.lru_cache(None)
    def get_system() -> Dict[str, Any]:
        try:
            from triton.compiler.compiler import triton_key

            # 使用 triton_key 而不是 triton.__version__ 作为版本号，因为后者不会随着每次代码更改而更新
            triton_version = triton_key()
        except ModuleNotFoundError:
            triton_version = None

        try:
            system: Dict[str, Any] = {
                "device": {
                    "name": torch.cuda.get_device_properties(
                        torch.cuda.current_device()
                    ).name,
                },
                "version": {
                    "cuda": torch.version.cuda,
                    "triton": triton_version,
                },
            }
        except (AssertionError, RuntimeError):
            # 如果未安装 CUDA，则上述配置均不相关
            system = {}

        system["hash"] = hashlib.sha256(
            json.dumps(system, sort_keys=True).encode("utf-8")
        ).hexdigest()

        return system
    # 返回包含设备和版本信息的字典，并计算其哈希值

    @staticmethod
    @clear_on_fresh_inductor_cache
    @functools.lru_cache(None)
    def get_local_cache_path() -> Path:
        return Path(os.path.join(cache_dir(), "cache", CacheBase.get_system()["hash"]))
    # 返回本地缓存文件的路径

    @staticmethod
    @functools.lru_cache(None)
    def get_global_cache_path() -> Optional[Path]:
        return (
            Path(os.path.join(config.global_cache_dir, CacheBase.get_system()["hash"]))
            if config.global_cache_dir is not None
            else None
        )
    # 返回全局缓存文件的路径，如果未设置全局缓存目录则返回 None

    def __init__(self) -> None:
        self.system = CacheBase.get_system()
    # 初始化 CacheBase 类，获取系统信息并存储在实例变量中

    def get_local_cache(self) -> Dict[str, Any]:
        local_cache_path = self.get_local_cache_path()
        if not local_cache_path.is_file():
            return {}
        with open(local_cache_path) as local_cache_fp:
            local_cache = json.load(local_cache_fp)
        return local_cache["cache"]
    # 获取本地缓存内容并返回，如果缓存文件不存在则返回空字典
    # 定义一个方法，用于更新本地缓存数据，接收一个类型为 Dict[str, Any] 的参数 local_cache，无返回值
    def update_local_cache(self, local_cache: Dict[str, Any]) -> None:
        # 获取本地缓存文件路径
        local_cache_path = self.get_local_cache_path()
        # 调用 write_atomic 函数，将 local_cache 对象以 JSON 格式写入到 local_cache_path 指定的文件中
        write_atomic(
            str(local_cache_path),
            json.dumps({"system": self.system, "cache": local_cache}, indent=4),
            make_dirs=True,
        )
class LocalCache(CacheBase):
    # 本地缓存类，继承自缓存基类 CacheBase

    def lookup(self, *keys: str) -> Optional[Dict[str, Any]]:
        # 查找方法，接受多个键值作为参数，返回可选的键值对字典或空
        cache = self.get_local_cache()

        sub_cache = cache
        for key in keys:
            if key in cache:
                sub_cache = cache[key]
            else:
                return None

        return sub_cache

    def set_value(self, *keys: str, value: Any) -> None:
        # 设置方法，接受多个键值和一个值作为参数，无返回值
        cache = self.get_local_cache()

        sub_cache = cache
        for key in keys[0:-1]:
            sub_cache.setdefault(key, {})
            sub_cache = sub_cache[key]
        sub_cache[keys[-1]] = value

        self.update_local_cache(cache)


class PersistentCache(CacheBase):
    # 持久化缓存类，继承自缓存基类 CacheBase

    @functools.lru_cache(None)  # noqa: B019
    def get_global_cache(self):
        # 获取全局缓存方法，使用 functools.lru_cache 进行缓存
        global_cache_path = self.get_global_cache_path()
        if global_cache_path is None or not global_cache_path.is_file():
            return {}
        with open(global_cache_path) as global_cache_fp:
            global_cache = json.load(global_cache_fp)
        return global_cache["cache"]

    def lookup(
        self,
        choices: List[ChoiceCaller],
        op: str,
        inputs: str,
        benchmark: Optional[Callable[[Any], Dict[ChoiceCaller, float]]],
    ):
        # 查找方法，接受多个参数，包括选择列表，操作，输入，基准测试函数
        ...


def get_lock_dir() -> str:
    # 获取锁目录方法，返回字符串类型的目录路径
    lock_dir = os.path.join(cache_dir(), "locks")
    if not os.path.exists(lock_dir):
        os.makedirs(lock_dir, exist_ok=True)
    return lock_dir


def sha256_hash(data: bytes) -> str:
    # SHA-256 哈希计算方法，接受字节串作为输入，返回小写的哈希字符串
    # [:51] to strip off the "Q====" suffix common to every hash value.
    return base64.b32encode(hashlib.sha256(data).digest())[:51].decode("utf-8").lower()


def code_hash(code: Union[str, bytes], extra: str = ""):
    # 代码哈希计算方法，接受代码字符串或字节串和额外字符串作为输入，返回哈希字符串
    hashing_str = code if isinstance(code, bytes) else code.encode("utf-8")
    if extra != "":
        hashing_str = hashing_str + b"||" + extra.encode("utf-8")
    return "c" + sha256_hash(hashing_str)


def get_path(
    basename: str, extension: str, specified_dir: str = ""
) -> Tuple[str, str, str]:
    # 获取路径方法，接受基本名称、扩展名和指定目录作为输入，返回三元组路径信息
    if specified_dir:
        if os.path.isabs(specified_dir):
            subdir = specified_dir
        else:
            subdir = os.path.join(cache_dir(), specified_dir)
    else:
        subdir = os.path.join(cache_dir(), basename[1:3])
    path = os.path.join(subdir, f"{basename}.{extension}")
    return basename, subdir, path


def get_hash(content: Union[str, bytes], extra: str = "", hash_type: str = "code"):
    # 获取哈希方法，接受内容、额外信息和哈希类型作为输入，根据类型返回哈希值
    if hash_type == "code":
        return code_hash(content, extra)
    if hash_type in ["cubin", "hsaco"]:
        return code_hash(repr(content))
    raise AssertionError(f"Unknown hash type {hash_type}")


def write(
    content: Union[str, bytes],
    extension: str,
    extra: str = "",
    hash_type: str = "code",
    specified_dir: str = "",
) -> Tuple[str, str]:
    # 写入方法，接受内容、扩展名、额外信息、哈希类型和指定目录作为输入，返回两元组路径信息
    # use striped content to compute hash so we don't end up with different
    # hashes just because the content begins/ends with different number of
    # spaces.
    key: str = get_hash(content.strip(), extra, hash_type)
    # 调用函数 get_path 获取文件路径信息，并将返回值解包为 basename、subdir 和 path
    basename, subdir, path = get_path(key, extension, specified_dir)
    # 检查路径是否不存在
    if not os.path.exists(path):
        # 调用函数 write_atomic 将 content 写入路径 path，如果需要则创建目录
        write_atomic(path, content, make_dirs=True)
    # 返回 basename 和 path 作为函数结果
    return basename, path
def write_text(text: str) -> str:
    """
    Write the `text` to a file and return the path computed based on the hash.
    """
    # 调用通用的写入函数 `write`，以文本形式写入内容，并返回基于哈希计算得到的路径
    return write(text, "txt")[1]


def write_atomic(
    path: str, content: Union[str, bytes], make_dirs: bool = False
) -> None:
    """
    Write content to `path` atomically, optionally creating directories if they don't exist.
    """
    # 首先写入临时文件以避免线程之间的冲突
    # 避免使用命名临时文件，因为它们具有受限制的权限
    assert isinstance(
        content, (str, bytes)
    ), "Only strings and byte arrays can be saved in the cache"
    path = Path(path)
    if make_dirs:
        path.parent.mkdir(parents=True, exist_ok=True)
    # 创建临时文件路径，以当前进程ID和线程ID来命名
    tmp_path = path.parent / f".{os.getpid()}.{threading.get_ident()}.tmp"
    # 写入模式根据内容类型确定
    write_mode = "w" if isinstance(content, str) else "wb"
    with tmp_path.open(write_mode) as f:
        f.write(content)
    # 原子性地重命名临时文件为最终文件路径，确保写入的原子性
    tmp_path.rename(path)


@dataclasses.dataclass
class TensorMetadataAndValues:
    """
    TensorMetadata plus the elements as a list of raw values.
    Used for hashing inlined constants.
    """
    tensor_metadata: TensorMetadata
    values: List[Any]


def _ident(x: Any) -> Any:
    """
    Identity function returning the same input value.
    """
    return x


def extract_tensor_metadata_for_cache_key(device_map, t):
    """
    Extracts the tensor metadata and removes non-caching fields from TensorMetadata.
    """
    # 提取张量的元数据，并移除不需要用于缓存的字段
    meta = extract_tensor_metadata(t)
    if not hasattr(t, "_is_inductor_static"):
        meta = dataclasses.replace(meta, storage_offset=0, storage_bytes=None)

    # 为了保证哈希的一致性，对设备对象进行记忆化处理，避免重复序列化相同的设备对象
    if meta.device not in device_map:
        device_map[meta.device] = meta.device
    meta = dataclasses.replace(meta, device=device_map[meta.device])

    return meta


def _reduce_fake_tensor(device_map, t):
    """
    Custom reduction function for FakeTensors used by FxGraphCachePickler.
    """
    # 自定义的序列化函数，用于处理 FakeTensors
    metadata = extract_tensor_metadata_for_cache_key(device_map, t)
    return (_ident, (metadata,))


def _reduce_tensor(device_map, t):
    """
    Custom reduction function for Tensors used by FxGraphCachePickler.
    """
    # 自定义的序列化函数，用于处理普通的 Tensors
    # 如果遇到张量，我们知道它们是作为属性存储在 GraphModule 上的常量，将值包含在键计算中
    # 小张量将被内联，因此对不同值无法提供相同的缓存条目
    # 大常量被视为参数，因此可能重用缓存条目
    # 为了做到这一点，然而，
    pass  # 这里的内容未完，不包含在注释中
    PyCodeCache would need more complexity to create a new module from its
    cache, but with the right constants attached as attributes.
    """
    如果 PyCodeCache 需要从其缓存中创建新模块，可能需要更复杂的操作，并且需要将正确的常量附加为属性。

    如果张量 t 使用了 mkldnn（Math Kernel Library for Deep Neural Networks）：
    # TODO: 目前这些张量无法序列化（pickle），因此无法缓存包含它们的编译图。暂时抛出异常。
    # 如果 mkldnn 张量支持序列化，可以移除这段代码。
    if t.is_mkldnn:
        raise BypassFxGraphCache

    # 对于非常大的张量，复制到 CPU 并计算哈希可能代价高昂。至少在发现速度变慢时报告警告。
    start = time()
    values = t.tolist()  # 将张量 t 转换为列表形式
    elapsed = time() - start  # 计算操作耗时
    if elapsed > 1.0:
        warnings.warn(
            f"FX graph cache handling of a large constant took {elapsed:.1}s. Please file an issue."
        )

    # 提取张量的元数据用于生成缓存键
    metadata = extract_tensor_metadata_for_cache_key(device_map, t)
    # 返回标识符和包含元数据与数值的元组作为结果
    return (_ident, (TensorMetadataAndValues(metadata, values),))
# 定义一个私有函数 _reduce_symint，用于定制化地序列化 SymInt 对象
def _reduce_symint(s):
    """
    See FxGraphCachePickler. Custom reducer to pickle SymInts.
    """
    # 为了哈希目的，只关注符号的名称而不是其背后的值。
    # 我们评估与缓存图表一起存储的保护条件，以确保可以安全重用具有 SymInt 参数的缓存实体。
    return (_ident, (str(s),))


# 定义一个私有函数 _reduce_unsupported，用于处理不支持的对象，从而绕过缓存
def _reduce_unsupported(s):
    """
    See FxGraphCachePickler. Custom reducer to handle any objects that we don't
    support and therefore raise to bypass caching.
    """
    raise BypassFxGraphCache


class FxGraphCachePickler(pickle.Pickler):
    """
    Custom pickler to customize the pickling of some objects (Tensors), only for the
    purpose of computing a hash for keying into the FxGraphCache. Tensors contain
    objects that don't pickle and/or vary between runs, and we want to capture the
    data that allow us to compute a stable, but safe hash.
    """

    # See extract_tensor_metadata_for_cache_key. Whenever we extract metadata during
    # pickling, we make sure devices always reference the same torch.device object.
    _device_map: Dict[torch.device, torch.device] = {}

    # 复制 copyreg 模块的分发表，并为特定类型注册定制化的序列化函数
    dispatch_table = copyreg.dispatch_table.copy()
    dispatch_table[FakeTensor] = functools.partial(_reduce_fake_tensor, _device_map)
    dispatch_table[torch.Tensor] = functools.partial(_reduce_tensor, _device_map)
    dispatch_table[torch.SymInt] = _reduce_symint
    dispatch_table[
        torch.fx.experimental._backward_state.BackwardState
    ] = _reduce_unsupported

    @classmethod
    def dumps(cls, obj) -> bytes:
        """
        Pickle an object using the FxGraphCachePickler.
        """
        # 使用 FxGraphCachePickler 序列化一个对象
        with io.BytesIO() as stream:
            pickler = cls(stream)
            try:
                pickler.dump(obj)
            except (TypeError, AttributeError) as e:
                # 某些配置选项是可调用对象，例如 post_grad_custom_pre_pass，
                # 可能无法被序列化
                log.warning("Can't pickle", exc_info=True)
                # 抛出 BypassFxGraphCache 异常以绕过缓存
                raise BypassFxGraphCache from e
            return stream.getvalue()

    @classmethod
    def get_hash(cls, obj: Any) -> str:
        """
        Serialize an object using the FxGraphCachePickler and return a hash
        of the pickled object.
        """
        # 使用 FxGraphCachePickler 序列化一个对象，并返回其哈希值
        serialized_data = cls.dumps(obj)
        return sha256_hash(serialized_data)

    @classmethod
    def debug_str(cls, inp: Any) -> str:
        """
        Get a printable string describing in more detail all the attributes
        comprising an object. Useful for debugging when one graph hashes
        to a different value than another.
        """
        
        # 定义内部函数，根据对象的类型返回描述字符串
        def get_str(obj) -> str:
            # 如果是 torch.Tensor 对象，调用提取张量元数据的函数
            if isinstance(obj, torch.Tensor):
                return str(extract_tensor_metadata_for_cache_key(cls._device_map, obj))
            # 如果是 bytes 类型对象，返回 "<bytes>"
            elif isinstance(obj, bytes):
                return "<bytes>"
            # 如果对象类型在 dispatch_table 中有定义，调用相应的函数处理对象
            elif type(obj) in cls.dispatch_table:
                # 运行 dispatch_table 中相应类型对象的 reducer 函数，返回结果的字符串表示
                return str(cls.dispatch_table[type(obj)](obj)[1])
            else:
                # 否则直接返回对象的字符串表示
                return str(obj)
        
        # 初始化一个空列表，用于存储每个属性的描述字符串
        lines = []
        # 遍历输入对象 inp 的所有属性及其对应的值
        for attr, obj in vars(inp).items():
            # 如果属性值是列表
            if isinstance(obj, list):
                # 遍历列表中的每个元素
                for ii in range(len(obj)):
                    # 调用 get_hash 方法计算对象的哈希值
                    h = cls.get_hash(obj[ii])
                    # 将哈希值、属性名、索引和对象的描述字符串添加到 lines 列表中
                    lines.append(f"[{h}] {attr}[{ii}]: {get_str(obj[ii])}")
            # 如果属性值是字典
            elif isinstance(obj, dict):
                # 遍历字典中的每个键值对
                for k, v in obj.items():
                    # 调用 get_hash 方法计算值 v 的哈希值
                    h = cls.get_hash(v)
                    # 将哈希值、属性名、键、值的描述字符串添加到 lines 列表中
                    lines.append(f"[{h}] {attr}[{k}]: {get_str(v)}")
            else:
                # 对于其他类型的属性值，直接计算其哈希值
                h = cls.get_hash(obj)
                # 将哈希值、属性名和对象的描述字符串添加到 lines 列表中
                lines.append(f"[{h}] {attr}: {get_str(obj)}")
        
        # 将 lines 列表中的所有描述字符串用换行符连接成一个完整的输出字符串并返回
        return "\n".join(lines)
# 定义一个函数，用于构建代码的哈希值，以表示源代码的内容
def build_code_hash(roots, prefix, hasher):
    # 遍历指定路径中以指定前缀开头的所有模块，并按模块名称排序
    for lib in sorted(pkgutil.iter_modules(roots, prefix), key=lambda x: x.name):
        # 根据模块名称查找对应的规范（spec）
        spec = lib.module_finder.find_spec(lib.name, None)
        # 确保找到了模块的规范
        assert spec is not None
        # 获取模块的源文件路径
        module = spec.origin
        # 确保找到了模块的源文件路径
        assert module is not None
        # 打开模块的源文件，并使用二进制模式读取文件内容
        with open(module, "rb") as f:
            # 更新哈希对象，使用模块的名称编码后的字节序列更新哈希值
            hasher.update(spec.name.encode("utf-8"))
            # 更新哈希对象，使用模块文件的内容更新哈希值
            hasher.update(f.read())
        # 如果当前模块是一个包（package），还需要对其子模块进行哈希处理
        if lib.ispkg:
            # 递归调用 build_code_hash 处理子模块
            build_code_hash(spec.submodule_search_locations, f"{spec.name}.", hasher)


# 定义一个函数，用于获取代码的哈希值
def get_code_hash(roots, extra_files=()):
    # 创建一个 SHA-256 的哈希对象
    hasher = hashlib.sha256()
    # 更新哈希对象，使用 torch 版本号的 UTF-8 编码字节序列更新哈希值
    hasher.update(torch.__version__.encode("utf-8"))
    # 调用 build_code_hash 函数，处理指定路径中的模块文件，并更新哈希值
    build_code_hash(roots, "", hasher)
    # 遍历额外指定的文件列表
    for path in extra_files:
        # 如果文件路径存在
        if os.path.exists(path):
            # 打开文件并使用二进制模式读取文件内容，更新哈希值
            with open(path, "rb") as f:
                hasher.update(f.read())
    # 返回哈希对象的摘要结果（哈希值的字节表示）
    return hasher.digest()


# 定义一个使用 functools.lru_cache 装饰器装饰的函数，用于计算与 torch 源文件相关的关键信息
@functools.lru_cache(None)
def torch_key():
    """
    Compute a key that contains relevant information about torch source files
    """
    # 如果不是在 FBCode 环境下
    if not config.is_fbcode():
        # 获取当前文件的所在目录路径
        inductor_root = os.path.dirname(__file__)
        # 指定额外的文件列表
        extra_files = (
            "codegen/aoti_runtime/interface.cpp",
            "codegen/aoti_runtime/implementation.cpp",
            "codegen/cpp_prefix.h",
            "script.ld",
        )
        # 返回调用 get_code_hash 函数计算得到的哈希值
        return get_code_hash(
            [inductor_root], [os.path.join(inductor_root, x) for x in extra_files]
        )
    
    # 如果在 FBCode 环境下，则从 libfb.py 中导入 parutil，并返回指定文件中的内容作为哈希键
    from libfb.py import parutil
    return parutil.get_file_contents("torch/src_hash.txt").rstrip()


# 定义一个函数，用于获取当前文件的所在目录路径
def get_inductor_root():
    return os.path.dirname(__file__)


# 定义一个数据类，用于持有有序集合的对象
@dataclasses.dataclass
class OrderedSetHolder:
    """
    See FxGraphHashDetails. Holds a sorted list to support stable hashing
    of set kwargs.
    """
    # 用于存储任意类型对象的列表
    items: List[Any]


# 定义一个自定义异常类，用于指示应绕过 FxGraphCache
class BypassFxGraphCache(Exception):
    """
    Exception to indicate that the FxGraphCache should be bypassed.
    """
    pass


# 定义一个类，用于捕获编译的 FX 图的所有细节，以计算一个安全和稳定的缓存键
class FxGraphHashDetails:
    """
    Object to capture all the details for a compiled FX graph relevant to computing
    a safe and stable cache key.
    """
    # 不稳定的关键字参数列表，这些参数不会影响缓存键的稳定性
    EXCLUDED_KWARGS = ["graph_id"]

    def __init__(
        self,
        gm: torch.fx.GraphModule,
        example_inputs: List[torch.Tensor],
        fx_kwargs: Dict[str, Any],
        inputs_to_check: Sequence[int],
        ):
        self.gm = gm
        self.example_inputs = example_inputs

        # Order kwargs so hashing is stable to changes in kwarg order.
        self.fx_kwargs = {}
        for k in sorted(fx_kwargs):
            if k not in self.EXCLUDED_KWARGS:
                if type(fx_kwargs[k]) is set:
                    # Special case to handle set params. Python sets can't be
                    # ordered, so sort the elements and store them in a proxy.
                    self.fx_kwargs[k] = OrderedSetHolder(sorted(fx_kwargs[k]))
                else:
                    self.fx_kwargs[k] = fx_kwargs[k]

        # Alignment checks
        self.inputs_to_check = inputs_to_check

        # 'Deterministic algorithms' can affect codegen via lowering to cuda kernels.
        self.deterministic_algorithms_settings = (
            torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled(),
            torch.utils.deterministic.fill_uninitialized_memory,  # type: ignore[attr-defined]
        )

        # Global settings affecting matmul codegen.
        self.cuda_matmul_settings = (
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        )

        # Also hash on various system info (including the triton compiler version).
        self.torch_version = torch_key()
        self.system_info = CacheBase.get_system()
        self.inductor_config = config.save_config_portable()

    def debug_str(self) -> str:
        """
        Get a printable string describing in more detail all the attributes
        comprising this object. Useful for debugging when one graph hashes
        to a different value than another.
        """
        return FxGraphCachePickler.debug_str(self)
    # 定义一个静态方法，用于计算 FX 图的哈希值以进行缓存
    def compiled_fx_graph_hash(
        gm: torch.fx.GraphModule,
        example_inputs: List[torch.Tensor],
        fx_kwargs: Dict[str, Any],
        inputs_to_check: Sequence[int],
    ) -> str:
        """
        Generate a unique hash of the FX graph for caching.
        """
        # 创建一个包含 FX 图相关详细信息的对象
        details = FxGraphHashDetails(gm, example_inputs, fx_kwargs, inputs_to_check)
        # 生成用于缓存的唯一键，前缀 'f' 用于区分模块中的其他缓存对象
        key = "f" + FxGraphCachePickler.get_hash(details)
        # 获取用于调试的详细信息字符串
        debug_str = details.debug_str()
        # 记录调试信息，包括生成的缓存键和详细信息
        log.debug(f"FX graph cache hash details for key {key}:\n{debug_str}")  # noqa: G004
        # 记录结构化的跟踪信息，用于存储到日志或其他跟踪系统中
        torch._logging.trace_structured(
            "artifact",
            metadata_fn=lambda: {
                "name": "fx_graph_cache_hash",
                "encoding": "json",
            },
            payload_fn=lambda: json.dumps(
                {"key": key, "components": debug_str.split("\n")}
            ),
        )

        # 返回生成的缓存键
        return key


class FxGraphCache:
    """
    Supports caching and reusing compiled Fx graphs.

    The overall strategy is as follows:
    - This cache stores entries on disk. When saving an entry, we can't
      serialize callables (that could be C++, Triton, etc.), so we serialize
      their own disk cache location. We then recreate the compiled artifact
      after fetching from disk.
    - For indexing the cache, we gather the fields relevant to identifying an
      FxGraph (the graph module, graph inputs, system settings etc.) into an
      FxGraphCacheDetails object, pickle it, and compute a hash for the key.
      See FxGraphCachePickler.
    - Among the metadata we store, we also include a guards expression that's
      appropriate for validating any symbols for Tensor arguments that have
      symbolic bounds. On cache lookup then, we evaluate those guards in the
      current context to validate that a cached entry can be served.
    - A given graph could have multiple compiled versions, corresponding to
      different sets of guards. Therefore, we store cache entries in the form:
          <temp dir>/<fx graph hash>/<serialized metatdata>
    - On lookup, we compute the key from the graph details, iterate over all
      leaf files in the corresponding subdirectory, deserialize the entry, and
      evaluate its guards expression. If the evaluation succeeds, we have a
      cache hit. If it fails, we compile the graph and store a new entry.
    - Finally, on a cache hit, we need to make sure any guards that would
      have been created during compilation are added to the current context.
    """

    # TODO(masnesral): Investigate whether it's beneficial to store compiled graphs
    # in an in-memory cache after loading from disk.
    
    @staticmethod
    # 获取用于存储编译图的顶层临时目录的静态方法
    def _get_tmp_dir() -> str:
        """
        Get the toplevel temporary directory for storing compiled graphs.
        """
        # 返回包含编译图的临时目录路径
        return os.path.join(cache_dir(), "fxgraph")
    def _get_tmp_dir_for_key(key: str) -> str:
        """
        Return the disk location for a given cache key.
        """
        # 构建临时目录路径，基于类方法返回的临时目录和给定的缓存键
        return os.path.join(FxGraphCache._get_tmp_dir(), key[1:3], key)

    @staticmethod
    def _filter_backed_symints(inputs: List[Any]) -> List[torch.SymInt]:
        """
        Get the backed SymInt objects from the input list. Note that we can never
        have guards that depend on unbacked symint.
        """
        # 筛选输入列表中的支持的 SymInt 对象，确保其具备提示信息
        return [s for s in inputs if isinstance(s, torch.SymInt) and has_hint(s)]

    @staticmethod
    def _get_shape_env() -> Optional[ShapeEnv]:
        """
        Helper to get the shape env from the tracing context.
        """
        # 尝试获取跟踪上下文，如果不存在则返回 None；否则返回伪造模式的形状环境
        ctx = torch._guards.TracingContext.try_get()
        if not ctx:
            return None
        return ctx.fake_mode.shape_env

    @staticmethod
    def _lookup_graph(
        key: str,
        example_inputs: List[torch.Tensor],
        local: bool,
        remote_cache: Optional[Any],
    ):
        """
        Lookup a graph using the provided key, example inputs, and cache context.
        """
        # 根据提供的键、示例输入和远程缓存上下文查找图形
        ...

    @staticmethod
    def _save_graph(
        key: str,
        compiled_graph: CompiledFxGraph,
        example_inputs: List[torch.Tensor],
        time_taken_ns,
        local,
        remote_cache,
    ):
        """
        Save a compiled graph along with associated metadata and caching context.
        """
        # 保存编译后的图形以及相关的元数据和缓存上下文
        ...
    ):
        """
        Store a serialized CompiledFxGraph on disk.
        将一个序列化的 CompiledFxGraph 存储到磁盘上。
        """
        disk_compiled_graph = copy(compiled_graph)
        # 复制编译后的图形，因为无法序列化可能是 C++/Triton 等的可调用函数，
        # 所以将它们的 PyCodeCache 磁盘缓存位置进行序列化。
        # TODO: 如果能够将编译模型序列化到磁盘，这个过程可以优化。
        disk_compiled_graph.current_callable = None

        # 在序列化之前，计算将用于确保从缓存加载的 CompiledFxGraph 是有效的保护表达式。
        # 只考虑 fx 图中的 SymInt 参数就足够了，因为 Tensor 的形状已经在缓存键的哈希中捕获。
        # 任何具有符号形状的 Tensor 参数都会在图中有一个 SymInt 参数。
        shape_env = FxGraphCache._get_shape_env()
        assert shape_env is not None
        symints = FxGraphCache._filter_backed_symints(example_inputs)
        guards = shape_env.get_pruned_guards(symints)
        disk_compiled_graph.guards_expr = shape_env.produce_guards_expression(
            placeholders=symints, guards=guards
        )

        try:
            content = pickle.dumps(disk_compiled_graph)
        except Exception:
            log.warning(
                "fx graph cache unable to serialize compiled graph", exc_info=True
            )
            counters["inductor"]["fxgraph_cache_pickle_error"] += 1
            return

        try:
            if local:
                subdir = FxGraphCache._get_tmp_dir_for_key(key)
                if not os.path.exists(subdir):
                    os.makedirs(subdir, exist_ok=True)

                # 使用序列化的 CompiledFxGraph 的哈希来获取唯一的文件名。
                # 具体的名称并不重要，因为查找涉及遍历父目录中的所有条目。
                path = os.path.join(subdir, sha256_hash(content))
                write_atomic(path, content, make_dirs=True)

            if remote_cache:
                cache_data = (
                    {
                        "data": content,
                        "time_taken_ms": time_taken_ns
                        // 1000000,  # Convert from NS to MS
                    }
                    if config.is_fbcode()
                    else content
                )
                remote_cache.put(key, cache_data)
        except Exception:
            log.warning("fx graph unable to write to cache", exc_info=True)
            counters["inductor"]["fxgraph_cache_write_error"] += 1
        # 检查是否可以缓存给定的 Torch FX 图模块
        def _check_can_cache(gm: torch.fx.GraphModule):
            """
            Check some conditions that would preclude caching and raise BypassFxGraphCache
            to bypass in case caching is not possible.
            """
            # 如果配置为 freezing 或启用了运行时常量折叠，则无法缓存
            if config.freezing or config.aot_inductor.use_runtime_constant_folding:
                raise BypassFxGraphCache

            # 缓存实现中需要有形状环境（shape env）
            if FxGraphCache._get_shape_env() is None:
                log.debug("fx graph cache no shape env")
                raise BypassFxGraphCache

            # HigherOrderOperators 需要逐个处理
            # 如果任何节点的 target 是 torch._ops.HigherOrderOperator 类型，则无法缓存
            for node in gm.graph.nodes:
                if isinstance(node.target, torch._ops.HigherOrderOperator):
                    raise BypassFxGraphCache
                # 如果节点操作为 "getattr"，且获取的对象是 torch._C.ScriptObject 类型，则无法缓存
                if node.op == "getattr" and isinstance(
                    getattr(gm, node.target), torch._C.ScriptObject
                ):
                    raise BypassFxGraphCache

        @staticmethod
        def load(
            compile_fx_fn: Callable[..., Any],
            gm: torch.fx.GraphModule,
            example_inputs: List[torch.Tensor],
            fx_kwargs: Dict[str, Any],
            inputs_to_check: Sequence[int],
            local: bool,
            remote: bool,
    ):
        """
        Load a compiled graph from the cache. If a cached entry does not exist,
        compile the graph and save it to the cache.
        """
        assert local or remote, "at least one of them needs to be enabled"
        compiled_graph = None
        try:
            # 检查是否可以缓存图形
            FxGraphCache._check_can_cache(gm)
            # 计算图的哈希值作为缓存键值
            key = compiled_fx_graph_hash(gm, example_inputs, fx_kwargs, inputs_to_check)

            remote_cache = None
            if remote:
                # 如果启用远程缓存
                cache_id = "fx-graph-v1"
                try:
                    if config.is_fbcode():
                        # 在 FB 环境中使用 FB Memcache 作为远程缓存后端
                        from triton.fb.fb_memcache import (
                            FbMemcacheRemoteFxGraphCacheBackend,
                        )
                        remote_cache = FbMemcacheRemoteFxGraphCacheBackend(cache_id)
                    else:
                        # 否则使用 Redis 作为远程缓存后端
                        from torch._inductor.remote_cache import RedisRemoteCacheBackend
                        remote_cache = RedisRemoteCacheBackend(cache_id)
                except Exception:
                    remote_cache = None
                    log.warning("Unable to create a remote cache", exc_info=True)

            # 查找缓存中的编译图
            compiled_graph = FxGraphCache._lookup_graph(
                key, example_inputs, local, remote_cache
            )

            if compiled_graph is None:
                # 如果缓存中没有找到编译图
                log.debug("fx graph cache miss for key %s", key)
                counters["inductor"]["fxgraph_cache_miss"] += 1
                start_time = time_ns()
                # 编译图形函数
                compiled_graph = compile_fx_fn(gm, example_inputs, **fx_kwargs)
                time_taken_ns = time_ns() - start_time
                # 将编译后的图形保存到缓存中
                FxGraphCache._save_graph(
                    key,
                    compiled_graph,
                    example_inputs,
                    time_taken_ns,
                    local,
                    remote_cache,
                )
            else:
                # 如果在缓存中找到了编译图
                log.debug("fx graph cache hit for key %s", key)
                counters["inductor"]["fxgraph_cache_hit"] += 1
            compiled_graph._fx_graph_cache_key = key
        except BypassFxGraphCache:
            # 如果需要绕过图形缓存异常，则计数器 +1
            counters["inductor"]["fxgraph_cache_bypass"] += 1
            if not compiled_graph:
                # 如果没有编译图，则编译新的函数图形
                compiled_graph = compile_fx_fn(gm, example_inputs, **fx_kwargs)

        return compiled_graph

    @staticmethod
    def clear():
        """
        Clear out the on-disk cache.
        """
        try:
            # 清除临时目录中的缓存
            shutil.rmtree(FxGraphCache._get_tmp_dir())
        except FileNotFoundError:
            # 如果找不到文件或目录，则忽略错误
            pass
@dataclasses.dataclass
class CompiledFxGraph:
    """
    Class holding a compiled FX graph. This is the object serialized on disk
    to support FxGraph caching.
    """

    current_callable: Optional[Callable[..., Any]]  # 当前可调用对象，可以接受任意参数
    cache_key: str  # 缓存键，用于标识该图的缓存
    source_code: str = dataclasses.field(repr=False)  # 源代码字符串，不在显示中显示
    cache_linemap: Optional[List[Tuple[int, str]]]  # 缓存的行映射表，可选的整数和字符串元组列表
    device_types: Set[str]  # 设备类型的集合，表示该图所支持的设备类型
    device_idxs: Set[int]  # 设备索引的集合，表示该图所支持的设备索引
    mutated_inputs: Set[str]  # 被修改的输入名称集合
    mutated_input_idxs: Set[int]  # 被修改的输入索引集合
    constants: Dict[str, torch.Tensor]  # 常量字典，包含名称到张量的映射
    torchbind_constants: Dict[str, torch._C.ScriptObject]  # Torchbind常量字典，名称到脚本对象的映射
    output_strides: Optional[List[Optional[Tuple[_StrideExprStr, ...]]]]  # 输出步长的可选列表，每个元素是一个步长元组
    disabled_cudagraphs_reason: Optional[str]  # 禁用CUDA图形的原因，可选的字符串
    metrics_deltas: metrics.CachedMetricsDeltas  # 缓存指标增量对象
    # This is a string representation of an expression we serialize
    # with the object so the guards can be evaluated in a different
    # context in order to verify the validity of serving a cached
    # fx graph. The expression must be generated by:
    # ShapeEnv.produce_guards_expression()
    guards_expr: Optional[str]  # 守卫表达式的字符串表示，用于验证缓存的FX图的有效性

    _boxed_call: Optional[bool] = None  # 可选的布尔值，表示是否进行了装箱调用
    _fx_graph_cache_key: Optional[str] = None  # 可选的字符串，FX图缓存键

    def __init__(
        self,
        current_callable: Optional[Callable[..., Any]],
        graph: GraphLowering,
        output_strides: List[Optional[Tuple[_StrideExprStr, ...]]],
        disabled_cudagraphs_reason: Optional[str],
        metrics_deltas: metrics.CachedMetricsDeltas,
    ):
        self.current_callable = current_callable  # 设置当前可调用对象
        self.cache_key = graph.cache_key  # 设置缓存键为图的缓存键
        if graph.cache_path:
            with open(graph.cache_path) as f:
                self.source_code = f.read()  # 如果有缓存路径，则读取源代码内容
        self.cache_linemap = graph.cache_linemap  # 设置缓存行映射表
        self.device_types = graph.device_types  # 设置设备类型集合
        self.device_idxs = graph.device_idxs  # 设置设备索引集合
        self.mutated_inputs = graph.mutated_inputs  # 设置被修改的输入名称集合
        self.mutated_input_idxs = set(graph.mutated_input_idxs)  # 设置被修改的输入索引集合
        self.constants = graph.constants  # 设置常量字典
        self.torchbind_constants = graph.torchbind_constants  # 设置Torchbind常量字典
        self.output_strides = output_strides  # 设置输出步长列表
        self.disabled_cudagraphs_reason = disabled_cudagraphs_reason  # 设置禁用CUDA图形的原因
        self.metrics_deltas = metrics_deltas  # 设置缓存指标增量对象
        self.guards_expr = None  # 初始时将守卫表达式设置为None

    def __call__(self, inputs: List[Any]) -> Any:
        assert self.current_callable is not None  # 断言当前可调用对象不为空
        return self.current_callable(inputs)  # 调用当前可调用对象并返回结果


"""
TODO: will remove old cpp builder when we switch to the new one.
"""


def get_compile_only(compile_only: bool = True) -> str:
    """
    返回用于编译的参数字符串，根据编译标志（compile_only）决定返回"-c"或空字符串。

    Args:
        compile_only (bool): 如果为True，返回"-c"，否则返回空字符串，默认为True。

    Returns:
        str: 返回"-c"或空字符串。
    """
    return "-c" if compile_only else ""


def get_shared(shared: bool = True, compile_only: bool = False) -> str:
    """
    返回用于共享编译的参数字符串。

    Args:
        shared (bool): 如果为True，表示需要共享编译，否则返回空字符串。
        compile_only (bool): 如果为True，表示仅编译，不共享。在Mac OS中，如果使用clang编译器，
                            返回"-shared -fPIC -undefined dynamic_lookup"，否则返回"-fPIC"。

    Returns:
        str: 返回适当的编译参数字符串。
    """
    from .cpp_builder import get_cpp_compiler  # 导入获取CPP编译器函数

    if not shared:
        return ""  # 如果不需要共享编译，返回空字符串
    if compile_only:
        return "-fPIC"  # 如果仅编译，返回"-fPIC"
    if platform.system() == "Darwin" and "clang" in get_cpp_compiler():
        # 如果在Mac OS中且使用clang编译器，返回特定的共享编译参数
        return "-shared -fPIC -undefined dynamic_lookup"
    else:
        # 如果条件不满足，则返回一个字符串 "-shared -fPIC"
        return "-shared -fPIC"
# 返回编译器警告所有标志（-Wall），如果 warning_all 为 True，否则返回空字符串
def get_warning_all_flag(warning_all: bool = True) -> str:
    return "-Wall" if warning_all else ""


# 返回 GLIBCXX 使用 C++11 ABI 的编译标志
def get_glibcxx_abi_build_flags() -> str:
    return "-D_GLIBCXX_USE_CXX11_ABI=" + str(int(torch._C._GLIBCXX_USE_CXX11_ABI))


# 返回 C++ 编译标志，包括标准为 C++17，忽略未使用变量和未知 pragma 的警告
# 如果是 clang 编译器，添加忽略优化参数错误的警告
def cpp_flags() -> str:
    from .cpp_builder import is_clang

    flags = ["-std=c++17", "-Wno-unused-variable", "-Wno-unknown-pragmas"]
    if is_clang():
        flags.append("-Werror=ignored-optimization-argument")
    return " ".join(flags)


# 返回用于包装 C++ 的编译标志，定义 TORCH_INDUCTOR_CPP_WRAPPER 宏
def cpp_wrapper_flags() -> str:
    return "-D TORCH_INDUCTOR_CPP_WRAPPER"


# 返回优化相关的编译标志，基础标志根据配置选择是 debug 还是 release 模式
# 包括快速数学计算但不仅限于有限数学，以及根据配置禁用不安全的数学优化和浮点约定
def optimization_flags() -> str:
    base_flags = "-O0 -g" if config.aot_inductor.debug_compile else "-O3 -DNDEBUG"
    base_flags += " -ffast-math -fno-finite-math-only"
    if not config.cpp.enable_unsafe_math_opt_flag:
        base_flags += " -fno-unsafe-math-optimizations"
    if not config.cpp.enable_floating_point_contract_flag:
        base_flags += " -ffp-contract=off"

    if config.is_fbcode():
        # 在 fbcode 中，不添加 `-fopenmp` 避免生成的共享库依赖 libgomp.so 导致问题
        return base_flags

    if sys.platform == "darwin":
        # 对于 macOS，通过 `-Xclang` 传递 openmp 标志，且 `-march=native` 在 M1 上不支持
        base_flags += " -Xclang"
    else:
        if platform.machine() == "ppc64le":
            base_flags += " -mcpu=native"
        else:
            base_flags += " -march=native"

    # 在非 fbcode 环境中添加 `-fopenmp`
    if not config.is_fbcode():
        base_flags += " -fopenmp"
    return base_flags


# 返回使用自定义生成的宏的编译标志
def use_custom_generated_macros() -> str:
    return "-D C10_USING_CUSTOM_GENERATED_MACROS"


# 返回使用 fb 内部宏的编译标志
# 在 fbcode 环境中，添加特定的预处理标志，如 `-Wp,-fopenmp`，设置 openmp 库路径，和特定的宏定义
def use_fb_internal_macros() -> str:
    if config.is_fbcode():
        # TODO: 用于在 fbcode 上避免 FC 破裂。当在旧版本 PyTorch 上使用新生成的 model.so 时，
        # 需要为 aoti_torch_create_tensor_from_blob 使用 v1 版本
        create_tensor_from_blob_v1 = "-D AOTI_USE_CREATE_TENSOR_FROM_BLOB_V1"
        openmp_lib = build_paths.openmp_lib()
        preprocessor_flags = " ".join(
            (
                "-D C10_USE_GLOG",
                "-D C10_USE_MINIMAL_GLOG",
                "-D C10_DISABLE_TENSORIMPL_EXTENSIBILITY",
            )
        )
        return f"-Wp,-fopenmp {openmp_lib} {preprocessor_flags} {create_tensor_from_blob_v1}"
    else:
        return ""


# 返回使用标准系统目录头文件的编译标志
# 在 fbcode 环境中，返回 `-nostdinc` 避免标准头文件
def use_standard_sys_dir_headers() -> str:
    if config.is_fbcode():
        return "-nostdinc"
    else:
        return ""


# 返回包含和链接路径的元组，用于包括 PyTorch，指令集，CUDA 和 AOT 模式
# 具体返回类型包括：include 路径列表，链接库路径，CUDA 头文件路径，和链接库名字
def get_include_and_linking_paths(
    include_pytorch: bool = False,
    vec_isa: VecISA = invalid_vec_isa,
    cuda: bool = False,
    aot_mode: bool = False,
) -> Tuple[List[str], str, str, str, str]:
    # 从cpp_builder模块导入所需函数和变量
    from .cpp_builder import (
        _get_python_include_dirs,
        homebrew_libomp,
        is_apple_clang,
        is_conda_llvm_openmp_installed,
    )

    # 设置GPU运行时环境
    _set_gpu_runtime_env()
    # 导入torch.utils.cpp_extension模块
    from torch.utils import cpp_extension

    # 定义一个空的宏变量
    macros = ""
    # 如果vec_isa不等于invalid_vec_isa，则生成编译宏定义
    if vec_isa != invalid_vec_isa:
        # 遍历vec_isa的构建宏，并将其格式化为"-D 宏名 "的形式添加到macros中
        for x in vec_isa.build_macro():
            macros_def = f"-D {x} "
            macros += macros_def

    # 定义一个空的构建架构标志变量
    build_arch_flags = ""
    # 如果操作系统为Linux，并且满足以下条件之一：include_pytorch为True、vec_isa不等于invalid_vec_isa、cuda为True、或者config.cpp.enable_kernel_profile为True
    if sys.platform == "linux" and (
        include_pytorch
        or vec_isa != invalid_vec_isa
        or cuda
        or config.cpp.enable_kernel_profile
    ):
        # 如果是Linux系统且需要包含PyTorch，则获取CUDA相关的包含路径和Python的包含路径
        ipaths = cpp_extension.include_paths(cuda) + _get_python_include_dirs()
        # 获取CUDA相关的库路径和系统配置变量LIBDIR的路径
        lpaths = cpp_extension.library_paths(cuda) + [
            sysconfig.get_config_var("LIBDIR")
        ]

        # 定义一个空的库列表
        libs = []

        # 如果不是在fbcode环境中，则手动添加以下库
        if not config.is_fbcode():
            libs += ["torch", "torch_cpu"]
            libs += ["gomp"]
            if not aot_mode:
                libs += ["torch_python"]
        else:
            # 在内部远程执行时，可以找到omp库，但找不到gomp库
            libs += ["omp"]
            if aot_mode:
                # 如果是AOT模式，则添加cpp_prefix_path()的目录到包含路径中，并根据情况转换CUDA路径
                ipaths += [os.path.dirname(cpp_prefix_path())]
                if cuda and torch.version.hip is None:
                    _transform_cuda_paths(lpaths)

        # 如果存在宏定义，则根据fbcode环境和vec_isa是否有效来设置宏
        if macros:
            if config.is_fbcode() and vec_isa != invalid_vec_isa:
                cap = str(vec_isa).upper()
                macros = " ".join(
                    [
                        vec_isa.build_arch_flags(),
                        f"-D CPU_CAPABILITY={cap}",
                        f"-D CPU_CAPABILITY_{cap}",
                        f"-D HAVE_{cap}_CPU_DEFINITION",
                    ]
                )

        # 如果使用CUDA，则根据环境设置宏和库
        if cuda:
            if macros is None:
                macros = ""
            macros += " -D USE_ROCM" if torch.version.hip else " -D USE_CUDA"

        # 根据CUDA版本设置相应的库和宏定义
        if cuda:
            if torch.version.hip is not None:
                if config.is_fbcode():
                    libs += ["amdhip64"]
                else:
                    libs += ["c10_hip", "torch_hip"]
                macros += " -D __HIP_PLATFORM_AMD__"
            else:
                if config.is_fbcode():
                    libs += ["cuda"]
                else:
                    libs += ["c10_cuda", "cuda", "torch_cuda"]

        # 设置构建架构标志为vec_isa的构建架构标志
        build_arch_flags = vec_isa.build_arch_flags()

    # 如果不是ABI兼容模式，则添加c10库到libs列表，并添加TORCH_LIB_PATH到库路径列表lpaths中
    if not config.abi_compatible:
        libs += ["c10"]
        lpaths += [cpp_extension.TORCH_LIB_PATH]
    # 如果配置为 fbcode 环境
    if config.is_fbcode():
        # 注意到包含路径的顺序很重要，因此需要在此处交替几个分支

        # 如果当前环境不是 HIP 版本的 Torch
        if torch.version.hip is None:
            # 添加 Sleef 库的构建路径到包含路径列表
            ipaths.append(build_paths.sleef())
        
        # 添加 OpenMP 库的构建路径到包含路径列表
        ipaths.append(build_paths.openmp())
        
        # 添加 Python 库的构建路径到包含路径列表
        ipaths.append(build_paths.python())
        
        # 根据 Torch 的版本判断是否为 HIP 版本
        if torch.version.hip is not None:
            # 添加 clang 的包含路径到包含路径列表
            ipaths.append(build_paths.clang_include())
            # 添加 gcc 的包含路径到包含路径列表
            ipaths.append(build_paths.gcc_include())
            # 添加 gcc 安装工具的包含路径到包含路径列表
            ipaths.append(build_paths.gcc_install_tools_include())
        else:
            # 添加 C++ 编译器的包含路径到包含路径列表
            ipaths.append(build_paths.cc_include())
            # 添加 libgcc 的包含路径到包含路径列表
            ipaths.append(build_paths.libgcc())
            # 添加 libgcc 架构相关的包含路径到包含路径列表
            ipaths.append(build_paths.libgcc_arch())
        
        # 添加 libgcc 向后兼容的包含路径到包含路径列表
        ipaths.append(build_paths.libgcc_backward())
        
        # 添加 glibc 库的包含路径到包含路径列表
        ipaths.append(build_paths.glibc())
        
        # 添加 Linux 内核的包含路径到包含路径列表
        ipaths.append(build_paths.linux_kernel())
        
        # 根据 Torch 的版本判断是否为 HIP 版本
        if torch.version.hip is not None:
            # 添加 ROCm 的包含路径到包含路径列表
            ipaths.append(build_paths.rocm())
        else:
            # 添加 CUDA 的包含路径到包含路径列表
            ipaths.append(os.path.join(build_paths.cuda(), "include"))
        
        # 将绝对路径的包含路径添加到远程目录
        # （稍后，我们将从 cpp_extensions 复制包含路径到我们的远程目录）
        ipaths.append("include")

    # 静态链接的库列表初始化为空
    static_link_libs = []

    # 如果处于 AOT 模式且支持 CUDA 并且配置为 fbcode 环境
    if aot_mode and cuda and config.is_fbcode():
        # 对于 Meta 内部 CUDA-12，建议静态链接 cudart 库
        if torch.version.hip is None:
            static_link_libs = ["-Wl,-Bstatic", "-lcudart_static", "-Wl,-Bdynamic"]

    # 构建动态库搜索路径的字符串表示，每个路径前加上 "-L"
    lpaths_str = " ".join(["-L" + p for p in lpaths])
    
    # 构建链接的库的字符串表示，包括静态链接的库和动态链接的库
    libs_str = " ".join(static_link_libs + ["-l" + p for p in libs])
    
    # 返回包含路径列表、动态库搜索路径字符串、库字符串、宏定义和构建架构标志
    return ipaths, lpaths_str, libs_str, macros, build_arch_flags
# 定义一个函数，用于生成编译 C++ 代码的命令字符串
def cpp_compile_command(
    input: Union[str, List[str]],  # 输入文件名或文件名列表
    output: str,  # 输出文件名
    warning_all: bool = True,  # 是否开启所有警告
    shared: bool = True,  # 是否生成共享库
    include_pytorch: bool = False,  # 是否包含 PyTorch 头文件
    vec_isa: VecISA = invalid_vec_isa,  # 向量指令集枚举类型
    cuda: bool = False,  # 是否使用 CUDA
    aot_mode: bool = False,  # 是否处于 AOT 模式
    compile_only: bool = False,  # 是否仅编译，不链接
    use_absolute_path: bool = False,  # 是否使用绝对路径
    use_mmap_weights: bool = False,  # 是否使用 mmap 加载权重
    extra_flags: Sequence[str] = (),  # 额外的编译标志
) -> str:  # 函数返回字符串类型

    from .cpp_builder import get_cpp_compiler, is_clang  # 导入获取 C++ 编译器函数和检查是否为 Clang 函数

    # 获取包含路径、链接路径、库、宏、构建架构标志
    ipaths, lpaths, libs, macros, build_arch_flags = get_include_and_linking_paths(
        include_pytorch, vec_isa, cuda, aot_mode
    )

    if isinstance(input, str):
        input = [input]  # 如果输入是字符串，则转换为列表

    ipaths_str = " ".join(["-I" + p for p in ipaths])  # 将包含路径列表转换为字符串

    clang_flags = ""  # 初始化 Clang 编译器标志

    if config.is_fbcode():  # 如果是在 Facebook Code 环境下
        if aot_mode and not use_absolute_path:
            inp_name = input  # 输入文件名
            out_name = output  # 输出文件名
            linker_script = _LINKER_SCRIPT  # 链接脚本
        else:
            # 需要复制任何绝对路径的 Torch 头文件
            inp_name = [os.path.basename(i) for i in input]  # 输入文件名的基本名称列表
            out_name = os.path.basename(output)  # 输出文件的基本名称
            linker_script = os.path.basename(_LINKER_SCRIPT)  # 链接脚本的基本名称

        assert is_clang()  # 断言是否为 Clang 编译器
        # 使用 Clang 运行时而非 libgcc
        clang_flags += " --rtlib=compiler-rt"
        clang_flags += " -fuse-ld=lld"
        clang_flags += f" -Wl,--script={linker_script}"  # 使用指定的链接脚本
        linker_paths = "-B" + build_paths.glibc_lib()  # 链接路径设置
        linker_paths += " -L" + build_paths.glibc_lib()  # 链接路径设置
    else:
        inp_name = input  # 输入文件名或文件名列表
        out_name = output  # 输出文件名
        linker_paths = ""  # 让编译器选择链接路径

    if compile_only:
        libs, lpaths = "", ""  # 如果仅编译，清空库和链接路径

    inp_name_str = " ".join(inp_name)  # 将输入文件名或文件名列表转换为字符串

    if use_mmap_weights:
        macros += " -D USE_MMAP_SELF"  # 如果使用 mmap 加载权重，则添加宏定义

    return re.sub(
        r"[ \n]+",
        " ",
        f"""
            {get_cpp_compiler()} {inp_name_str} {get_shared(shared, compile_only)}
            {get_warning_all_flag(warning_all)} {cpp_flags()}
            {get_glibcxx_abi_build_flags()}
            {ipaths_str} {lpaths} {libs} {build_arch_flags}
            {macros} {linker_paths} {clang_flags}
            {optimization_flags()} {cpp_wrapper_flags()}
            {use_custom_generated_macros()}
            {use_fb_internal_macros()}
            {use_standard_sys_dir_headers()}
            {get_compile_only(compile_only)}
            {' '.join(extra_flags)}
            -o {out_name}
        """,
    ).strip()


# 运行指定的命令并检查返回状态
def run_command_and_check(cmd: str):
    cmd = shlex.split(cmd)  # 将命令字符串转换为列表形式
    try:
        subprocess.check_call(cmd)  # 调用子进程执行命令
    except subprocess.CalledProcessError as e:
        raise exc.CppCompileError(cmd, e.output) from e  # 捕获异常并抛出 C++ 编译错误


# 通过装饰器在新的 AOT 引导缓存中清除
@functools.lru_cache(None)
def split_aot_inductor_output_path(path: str) -> Tuple[str, str]:
    """返回 AOT Inductor 编译内核存储的路径。"""
    if path.endswith(".so"):
        return os.path.split(path)  # 如果路径以 .so 结尾，返回路径的拆分结果
    else:
        return path, ""  # 否则返回路径和空字符串


# 在新的 AOT 引导缓存中清除
@clear_on_fresh_inductor_cache
class CudaKernelParamCache:
    # 初始化一个空字典作为缓存，缓存的键为字符串，值为字典
    cache: Dict[str, Dict[str, str]] = dict()
    
    # 定义一个静态方法 cache_clear，用于清空缓存
    cache_clear = staticmethod(cache.clear)
    
    # 定义一个类方法 set，用于向缓存中存储数据
    @classmethod
    def set(cls, key: str, params: Dict[str, str], cubin: str) -> None:
        # 根据 torch 版本选择二进制类型，若无 hip 版本则为 "cubin"，否则为 "hsaco"
        bin_type = "cubin" if torch.version.hip is None else "hsaco"
        # 调用 write 函数将 cubin 写入指定路径，并返回路径
        _, path = write(
            cubin,
            bin_type,
            hash_type=bin_type,
            specified_dir=split_aot_inductor_output_path(
                config.aot_inductor.output_path
            )[0],
        )
        # 将路径添加到 params 字典中，键为获取 CPP 封装 cubin 路径名的函数返回值
        params[get_cpp_wrapper_cubin_path_name()] = path
        # 将 params 存储到类的缓存中，键为提供的 key
        cls.cache[key] = params
    
    # 定义一个类方法 get，用于从缓存中获取数据
    @classmethod
    def get(cls, key: str) -> Optional[Dict[str, str]]:
        # 返回缓存中键为 key 的值，若不存在则返回 None
        return cls.cache.get(key, None)
    
    # 定义一个类方法 get_keys，返回当前缓存中所有键的集合
    @classmethod
    def get_keys(cls):
        return cls.cache.keys()
# 定义一个名为 AotCodeCompiler 的类，用于编译代码
class AotCodeCompiler:
    
    # 使用 @classmethod 装饰器定义一个类方法 compile
    @classmethod
    # compile 方法接受以下参数：
    # - graph: GraphLowering 类型，表示图形降低对象
    # - source_code: str 类型，表示源代码字符串
    # - serialized_extern_kernel_nodes: Optional[str] 类型，可选的序列化外部核节点字符串
    # - cuda: bool 类型，表示是否使用 CUDA
    @clear_on_fresh_inductor_cache
    @functools.lru_cache
    # 定义一个名为 cpp_prefix_path 的函数，返回类型为 str
    def cpp_prefix_path() -> str:
        # 使用 Path(__file__).parent 拼接路径，指向 "codegen/cpp_prefix.h" 文件
        path = Path(__file__).parent / "codegen/cpp_prefix.h"
        # 打开路径指向的文件
        with path.open() as f:
            # 读取文件内容
            content = f.read()
            # 调用 write 函数，将内容写入以 ".h" 结尾的文件，并返回文件名
            _, filename = write(
                content,
                "h",
            )
        # 返回写入的文件名
        return filename

    # 定义一个名为 cpp_prefix 的函数，返回类型为 str
    def cpp_prefix() -> str:
        # 调用 cpp_prefix_path 函数，获取 cpp 前缀文件的文件名
        filename = cpp_prefix_path()
        # 如果 config.is_fbcode() 返回 True
        if config.is_fbcode():
            # 返回一个包含相对路径的字符串，用于 FBCode，由于将所有编译内容捆绑到远程编译文件夹中
            return f'#include "{os.path.basename(filename)}"'
        else:
            # 返回一个包含文件名的字符串，用于其他情况
            return f'#include "{filename}"'

    # 定义一个名为 compile_file 的函数，使用 @dynamo_timed 装饰器，返回类型为 None
    def compile_file(
        input_path: Union[str, List[str]], output_path: str, cmd: List[str]
    ) -> None:
        # 如果 input_path 是 str 类型，转换为包含一个元素的列表
        input_paths = [input_path] if isinstance(input_path, str) else input_path
        # 对 input_paths 中的每个路径进行处理，如果 config.is_fbcode() 返回 True，则返回其基本名称，否则返回原路径
        input_files = [
            os.path.basename(ip) if config.is_fbcode() else ip for ip in input_paths
        ]
    try:
        # 检查是否运行在 Facebook Code 上
        if config.is_fbcode():
            # 需要将头文件复制到与源代码相同的文件夹中
            header_path = cpp_prefix_path()
            header_name = os.path.basename(header_path)
            output_name = os.path.basename(output_path)
            # 在远程构建时，需要确保谨慎地复制任何在编译过程中所需的文件到我们的构建目录中
            # 这是所有 ATen/c10/Torch 包含的文件来自的地方
            torch_includes_path = os.path.join(_TORCH_PATH, "include")
            with tempfile.TemporaryDirectory() as tmp_dir:
                # 将所有内容复制到临时编译文件夹中
                shutil.copy(header_path, os.path.join(tmp_dir, header_name))
                shutil.copy(_LINKER_SCRIPT, os.path.join(tmp_dir, "script.ld"))
                for p, f in zip(input_paths, input_files):
                    shutil.copy(p, os.path.join(tmp_dir, f))
                dest_include_path = os.path.join(tmp_dir, "include")
                shutil.copytree(torch_includes_path, dest_include_path)
                # 执行构建命令
                output_file_path = _run_build_command(cmd, tmp_dir, output_name)
                # 复制构建输出文件
                if os.path.exists(output_path):
                    os.remove(output_path)
                shutil.copy(output_file_path, output_path)
        else:
            # 在本地直接执行命令
            subprocess.check_output(cmd, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        # 如果命令执行失败，捕获异常
        output = e.output.decode("utf-8")
        # 检查是否是 OpenMP 相关的问题
        openmp_problem = "'omp.h' file not found" in output or "libomp" in output
        # 如果是在 macOS 上，并且缺少 OpenMP 支持，提供解决方案
        if openmp_problem and sys.platform == "darwin":
            instruction = (
                "\n\nOpenMP support not found. Please try one of the following solutions:\n"
                "(1) Set the `CXX` environment variable to a compiler other than Apple clang++/g++ "
                "that has builtin OpenMP support;\n"
                "(2) install OpenMP via conda: `conda install llvm-openmp`;\n"
                "(3) install libomp via brew: `brew install libomp`;\n"
                "(4) manually setup OpenMP and set the `OMP_PREFIX` environment variable to point to a path"
                " with `include/omp.h` under it."
            )
            output += instruction
        # 抛出自定义的 CppCompileError 异常
        raise exc.CppCompileError(cmd, output) from e
# Optional[CDLL] 类型的全局变量，用于存储动态链接库对象或者为空
_libgomp: Optional[CDLL] = None

# 自定义操作的包装器函数，将从生成的 cpp 包装器代码中以 JIT 模式调用
def custom_op_wrapper(op: str, *args):
    # 内部函数，用于转换参数
    def convert_arg(arg):
        # 如果参数是 PyCapsule 类型，则通过 alloc_tensor_by_stealing_from_void_ptr 转换为 Tensor
        if str(type(arg)) == "<class 'PyCapsule'>":
            return torch._C._aoti.alloc_tensor_by_stealing_from_void_ptr(arg)
        # 如果参数是 list 或 tuple 类型，则递归转换列表中的每个元素
        elif isinstance(arg, (list, tuple)):
            return type(arg)(convert_arg(a) for a in arg)
        else:
            return arg

    # 将所有参数转换为适用于 torch 函数调用的格式
    converted_args = [convert_arg(arg) for arg in args]

    # 断言操作名以 "torch.ops." 开头，确保通过 custom_op_wrapper 调用的函数名正确
    assert op.startswith("torch.ops."), (
        op + " can not be called through custom_op_wrapper"
    )
    
    # 逐级导入操作名中的模块，并获取最终的函数对象
    func = None
    for i, s in enumerate(op.split(".")):
        if i == 0:
            func = importlib.import_module(s)
        func = getattr(func, s)

    # 断言获取的函数对象可调用
    assert callable(func), op + " can not be loaded through custom_op_wrapper"

    # 调用函数，并获取结果
    result = func(*converted_args)
    
    # 如果结果是 list 或 tuple 类型，则确保列表中的每个元素是 Tensor 类型
    if isinstance(result, (list, tuple)):
        for r in result:
            assert isinstance(r, torch.Tensor), op + " returns a list of non-tensors"
        # 将结果列表中的每个 Tensor 转换为 void 指针并返回
        return torch._C._aoti.unsafe_alloc_void_ptrs_from_tensors(result)  # type: ignore[arg-type]
    else:
        # 断言单个结果是 Tensor 类型
        assert isinstance(result, torch.Tensor), op + " returns a non-tensor"
        # 将单个 Tensor 转换为 void 指针并返回
        return torch._C._aoti.unsafe_alloc_void_ptr_from_tensor(result)


# 装饰器，用于在新的感应器缓存上清除缓存
@clear_on_fresh_inductor_cache
# CppCodeCache 类，用于缓存和管理加载的动态链接库或模块
class CppCodeCache:
    # 缓存字典，存储路径到加载函数的映射
    cache: Dict[str, Callable[[], Union[CDLL, ModuleType]]] = {}
    # 静态方法，用于清空缓存
    cache_clear = staticmethod(cache.clear)
    # 存储编译命令标志的字典
    cpp_compile_command_flags: Dict[str, Any] = {}

    # 静态方法，内部加载动态链接库或模块
    @staticmethod
    def _load_library_inner(path: str, key: str) -> Union[CDLL, ModuleType]:
        return cdll.LoadLibrary(path)
    
    # 类方法，用于加载动态链接库或模块
    @classmethod
    # 定义类方法 `_load_library`，加载指定路径的动态链接库或模块，并返回加载的结果对象
    def _load_library(cls, path: str, key: str) -> Union[CDLL, ModuleType]:
        try:
            # 调用内部方法 `_load_library_inner` 加载库或模块
            result = cls._load_library_inner(path, key)
            # 将 key 属性赋给加载结果（类型提示忽略联合属性）
            result.key = key  # type: ignore[union-attr]
            # 返回加载结果
            return result
        except (ImportError, OSError) as e:
            # 如果异常信息中包含 "gomp" 并且系统中存在 "/usr/lib64/libgomp.so.1" 文件
            if "gomp" in str(e) and os.path.exists("/usr/lib64/libgomp.so.1"):
                # 临时的 fbcode/buck 兼容性修复
                global _libgomp
                # 加载 "/usr/lib64/libgomp.so.1" 库
                _libgomp = cdll.LoadLibrary("/usr/lib64/libgomp.so.1")
                # 再次调用内部方法 `_load_library_inner` 加载库或模块
                result = cls._load_library_inner(path, key)
                # 将 key 属性赋给加载结果（类型提示忽略联合属性）
                result.key = key  # type: ignore[union-attr]
                # 返回加载结果
                return result
            # 如果异常信息中包含 "failed to map segment from shared object"
            if "failed to map segment from shared object" in str(e):
                # 抛出新的 OSError 异常，说明可能的原因是临时文件夹设置为 noexec
                raise OSError(
                    f"{e}.  The most common reason this may occur is if the {tempfile.gettempdir()} folder "
                    "is mounted with noexec (e.g., by default Docker mounts tmp file systems "
                    f"as noexec).  Please remount {tempfile.gettempdir()} with exec enabled, or set another "
                    "temporary directory with TORCHINDUCTOR_CACHE_DIR environment variable."
                ) from e
            # 如果以上条件均不满足，则原样抛出异常
            raise

    @classmethod
    # 定义一个类方法 `load_async`，用于异步加载和编译给定的源代码
    def load_async(cls, source_code: str, cuda=False, submit_fn=None, extra_flags=()):
        # 构建编译命令，包括默认的编译标志、是否使用 CUDA、选择的向量指令集等
        compile_command = {
            **cls.cpp_compile_command_flags,
            "cuda": cuda,
            "vec_isa": pick_vec_isa(),  # 选择适合的向量指令集
            "extra_flags": extra_flags,
        }

        _set_gpu_runtime_env()  # 设置 GPU 运行时环境，供 cpp_extension 使用

        # 导入相关的编译器类和选项
        from torch._inductor.cpp_builder import CppBuilder, CppTorchCudaOptions

        # 创建一个虚拟的 CppBuilder 对象，用于生成编译命令行和设置编译选项
        dummy_builder = CppBuilder(
            name="o", sources="i", BuildOption=CppTorchCudaOptions(**compile_command)
        )

        # 获取虚拟 builder 对象的命令行表示形式，用于计算源代码的哈希值
        dummy_cmd = repr(dummy_builder.get_command_line())

        # 调用 `write` 函数将源代码写入文件，并返回生成的文件的键名和路径
        key, input_path = write(source_code, "cpp", extra=dummy_cmd)

        # 如果生成的键名不在类的缓存中
        if key not in cls.cache:
            # 导入文件锁相关的模块
            from filelock import FileLock

            # 构建锁文件路径
            lock_path = os.path.join(get_lock_dir(), key + ".lock")

            # 确定输出路径为输入路径去除末尾的 `.cpp` 并添加 `.so` 扩展名
            output_path = input_path[:-3] + "so"

            # 声明一个可选的 Future 对象和库对象
            future: Optional[Future[Any]] = None
            lib = None

            # 创建一个部分应用了 `_worker_compile_cpp` 的函数，用于实际的编译工作
            worker_fn = functools.partial(
                _worker_compile_cpp,
                lock_path,
                input_path,
                output_path,
                cpp_compile_command(
                    input=input_path, output=output_path, **compile_command
                ),
            )

            # 定义加载函数 `load_fn`
            def load_fn():
                nonlocal lib
                # 如果库对象尚未加载
                if lib is None:
                    # 如果存在未完成的 Future 对象，则等待其完成
                    if future is not None:
                        future.result()
                    # 执行编译任务并断言结果为 `None`
                    result = worker_fn()
                    assert result is None
                    # 加载生成的动态库，并断言加载成功
                    lib = cls._load_library(output_path, key)
                    assert lib is not None
                return lib

            # 如果有提交函数 `submit_fn`
            if submit_fn is not None:
                # 使用文件锁确保同时只有一个进程可以编译和生成动态库
                with FileLock(lock_path, timeout=LOCK_TIMEOUT):
                    # 如果输出路径不存在对应的 `.so` 文件，则提交编译任务
                    if not os.path.exists(output_path):
                        future = submit_fn(worker_fn)

            # 将加载函数 `load_fn` 存入类的缓存中
            cls.cache[key] = load_fn

        # 返回缓存中对应键名的加载函数
        return cls.cache[key]

    # 定义类方法 `load`，用于同步加载给定的源代码
    @classmethod
    def load(cls, source_code: str, cuda: bool = False):
        # 调用异步加载方法 `load_async` 并立即执行返回的加载函数
        return cls.load_async(source_code, cuda)()
# 使用文件锁保证多线程环境下对文件的安全访问，避免同时编译相同文件
def _worker_compile_cpp(lock_path, input_path, output_path, cmd):
    from filelock import FileLock

    # 使用文件锁来确保对lock_path的访问是原子性的，超时时间设定为LOCK_TIMEOUT
    with FileLock(lock_path, timeout=LOCK_TIMEOUT):
        # 如果输出路径不存在则进行编译操作
        if not os.path.exists(output_path):
            # 调用compile_file函数编译输入路径的文件到输出路径，使用shlex模块将cmd字符串解析为参数列表
            compile_file(input_path, output_path, shlex.split(cmd))


# 自定义的用于 cpp 内核的 Python 绑定
@clear_on_fresh_inductor_cache
class CppPythonBindingsCodeCache(CppCodeCache):
    # 缓存字典，存储模块名到可调用对象的映射
    cache: Dict[str, Callable[[], Union[CDLL, ModuleType]]] = {}
    # 静态方法，用于清空缓存字典
    cache_clear = staticmethod(cache.clear)
    # cpp编译命令的标志，设置为不包含PyTorch依赖且为共享模式
    cpp_compile_command_flags = {
        "include_pytorch": False,
        "shared": True,
    }
    # 入口函数名
    entry_function = "kernel"
    # 调用入口函数的字符串模板，将%s替换为实际的参数
    call_entry_function = "kernel(%s);Py_RETURN_NONE;"
    # 额外的解析参数，暂未指定任何内容
    extra_parse_arg = ""
    )  # 这里似乎有一个括号多余了，可能是打字错误

    @classmethod
    def _load_library_inner(cls, path: str, key: str) -> ModuleType:
        # 设置环境变量以便访问特定的Torch相关数据指针
        os.environ["_TORCHINDUCTOR_PYOBJECT_TENSOR_DATA_PTR"] = str(
            torch._C._dynamo.guards._torchinductor_pyobject_tensor_data_ptr  # type: ignore[attr-defined]
        )
        # 组合模块名，格式为key.kernel
        module_name = f"{key}.{cls.entry_function}"
        try:
            # 尝试从sys.modules中获取已加载的模块，如果存在直接返回
            return sys.modules[module_name]
        except KeyError:
            pass
        # 根据给定的路径创建一个模块规范
        spec = importlib.util.spec_from_file_location(module_name, path)
        assert spec is not None
        # 根据模块规范创建一个新的模块对象
        module = importlib.util.module_from_spec(spec)
        # 将新模块对象添加到sys.modules中，以便后续的import语句可以找到它
        sys.modules[module_name] = module
        # 使用模块加载器执行模块的代码，即导入模块并执行其中的Python代码
        spec.loader.exec_module(module)  # type: ignore[union-attr]
        return module

    @classmethod
    def load_pybinding_async(
        cls,
        argtypes: List[str],
        source_code: str,
        cuda: bool = False,
        num_outputs: int = -1,
        submit_fn=None,
        extra_flags=(),
    ) -> Any:
        """
        Wrap a C++ function in fast Python bindings.

        Args:
            argtypes: The types of args to ENTRY_FUNCTION(), e.g. ["float*", "long"]
            source_code: C++ source code containing a ENTRY_FUNCTION() function

        Returns:
            A python version of ENTRY_FUNCTION()
        """
        # 构建函数的装饰器，将 C++ 函数包装成快速的 Python 绑定
        parseargs = ", ".join(
            # 生成每个参数类型的解析代码
            f"parse_arg<{argtype.replace('const ', '')}>(args, {n})"
            for n, argtype in enumerate(argtypes)
        )
        # 根据类的模板和参数构建 C++ 函数的后缀部分
        suffix = cls.suffix_template % (
            cls.entry_function,
            cls.extra_parse_arg % num_outputs if cls.extra_parse_arg else "",
            cls.entry_function,
            len(argtypes),
            len(argtypes),
            cls.call_entry_function % parseargs,
            cls.entry_function,
            cls.entry_function,
            cls.entry_function,
            cls.entry_function,
        )
        # 异步加载经过修改后的源码，并获取结果
        get_result = cls.load_async(
            source_code + suffix, cuda, submit_fn=submit_fn, extra_flags=extra_flags
        )
        result = None

        def future():
            nonlocal result
            if result is None:
                # 获取结果并确保是模块类型
                result = get_result()
                assert isinstance(result, ModuleType)
            # 返回 ENTRY_FUNCTION() 函数
            return getattr(result, cls.entry_function)

        # 返回 future 函数
        return future

    @classmethod
    def load_pybinding(cls, *args, **kwargs) -> Any:
        # 调用异步加载方法，并立即执行返回的结果
        return cls.load_pybinding_async(*args, **kwargs)()
@clear_on_fresh_inductor_cache
class CppWrapperCodeCache(CppPythonBindingsCodeCache):
    # 定义缓存，存储编译后的模块或动态链接库的回调函数
    cache: Dict[str, Callable[[], Union[CDLL, ModuleType]]] = {}
    # 定义静态方法，用于清空缓存
    cache_clear = staticmethod(cache.clear)
    # 定义编译命令的标志，包括是否包含 PyTorch 头文件和是否共享模块
    cpp_compile_command_flags = {
        "include_pytorch": True,
        "shared": True,
    }
    # 定义入口函数的名称
    entry_function = "inductor_entry_cpp"
    # 定义调用入口函数的格式化字符串
    call_entry_function = "return inductor_entry_cpp(%s);"
    # 定义额外的解析参数，使用 textwrap.dedent() 函数去除缩进
    extra_parse_arg = textwrap.dedent(
        """
        #include <torch/csrc/inductor/aoti_torch/c/shim.h>

        static inline std::vector<AtenTensorHandle> unpack_tensor_handle_list(PyObject* pyvec) {
            std::vector<AtenTensorHandle> result;
            size_t result_len = PyList_GET_SIZE(pyvec);
            result.reserve(result_len);
            for (size_t i = 0; i < result_len; i++) {
                // AtenTensorHandle is essentially a pointer
                void* elem = PyCapsule_GetPointer(PyList_GET_ITEM(pyvec, i), NULL);
                result.push_back(reinterpret_cast<AtenTensorHandle>(elem));
            }
            return result;
        }

        static inline PyObject* pack_tensor_handle_list(const std::vector<AtenTensorHandle>& cppvec) {
            size_t result_len = cppvec.size();
            PyObject* result = PyList_New(static_cast<Py_ssize_t>(result_len));
            for (size_t i = 0; i < result_len; i++) {
                PyObject *elem =
                    cppvec[i] == nullptr
                        ? Py_None
                        // Store AtenTensorHandle as PyCapsule
                        : PyCapsule_New(reinterpret_cast<void*>(cppvec[i]), NULL, NULL);
                PyList_SET_ITEM(result, i, elem);
            }
            return result;
        }

        template <> inline std::vector<AtenTensorHandle> parse_arg<std::vector<AtenTensorHandle>>(PyObject* args, size_t n) {
            return unpack_tensor_handle_list(PyTuple_GET_ITEM(args, n));
        }

        PyObject* inductor_entry_cpp(std::vector<AtenTensorHandle>&& input_handles) {
            // For outputs, we only allocate a vector to hold returned tensor handles,
            // not allocating the actual output tensor storage here
            std::vector<AtenTensorHandle> output_handles(%s);
            try {
                inductor_entry_impl(input_handles.data(), output_handles.data());
                return pack_tensor_handle_list(output_handles);
            } catch(std::exception const& e) {
                PyErr_SetString(PyExc_RuntimeError, e.what());
                return {};
            } catch(...) {
                PyErr_SetString(PyExc_RuntimeError, "unhandled error");
                return {};
            }
        }
        """
    )


# TODO: Will remove the temp code after switch to new cpp_builder
def _temp_validate_new_and_old_command(new_cmd: List[str], old_cmd: List[str]):
    # 计算新旧命令列表的差异
    new_diff: List[str] = [x for x in new_cmd if x not in old_cmd]
    old_diff: List[str] = [y for y in old_cmd if y not in new_cmd]
    # 如果 new_diff 或者 old_diff 中有任何一个为真，则进入条件判断
    if new_diff or old_diff:
        # 打印新命令 new_cmd 的值，用于调试和错误检查
        print("!!! new_cmd: ", new_cmd)
        # 打印旧命令 old_cmd 的值，用于调试和错误检查
        print("!!! old_cmd: ", old_cmd)
        # 打印新差异 new_diff 的值，用于调试和错误检查
        print("!!! new_diff: ", new_diff)
        # 打印旧差异 old_diff 的值，用于调试和错误检查
        print("!!! old_diff: ", old_diff)
        # 抛出运行时异常，指示新旧命令不同的错误
        raise RuntimeError("Error in new and old command different.")
# 定义一个函数用于验证和生成 C++ 命令行选项
def _do_validate_cpp_commands(
    include_pytorch: bool,
    cuda: bool,
    compile_only: bool,
    mmap_weights: bool,
    use_absolute_path: bool,
):
    # 如果测试机器无法运行 CUDA，则 PreCI 将失败
    # 创建一个临时目录对象
    temp_dir = tempfile.TemporaryDirectory()
    # 获取临时目录的路径
    test_dir_path = temp_dir.name
    # 检查当前环境是否支持 CUDA 并且用户请求使用 CUDA
    test_cuda = torch.cuda.is_available() and cuda
    # 构建输入文件的路径
    input_path = os.path.join(test_dir_path, "dummy_input.cpp")
    # 初始化输出文件的路径为默认值
    output_path = os.path.join(test_dir_path, "dummy_output.so")
    # 如果设置了只编译选项，修改输出路径为目标对象文件
    if compile_only:
        output_path = os.path.join(test_dir_path, "dummy_output.o")
    # 选择矢量指令集
    picked_isa = pick_vec_isa()

    # 获取老的编译命令行选项，并将其转换为列表形式
    old_cmd = cpp_compile_command(
        input=input_path,
        output=output_path,
        include_pytorch=include_pytorch,
        vec_isa=picked_isa,
        cuda=test_cuda,
        aot_mode=False,
        compile_only=compile_only,
        use_absolute_path=use_absolute_path,
        use_mmap_weights=mmap_weights,
        extra_flags=["-D TEST_EXTRA_FLAGS"],
    ).split(" ")

    # 创建一个包含当前编译选项的对象，用于模拟生成命令行
    dummy_build_option = CppTorchCudaOptions(
        vec_isa=picked_isa,
        include_pytorch=include_pytorch,
        cuda=test_cuda,
        compile_only=compile_only,
        use_absolute_path=use_absolute_path,
        use_mmap_weights=mmap_weights,
        extra_flags=["-D TEST_EXTRA_FLAGS"],
    )

    # 创建一个 C++ 构建器对象，用于构建命令行
    dummy_builder = CppBuilder(
        name="dummy_output",
        sources=input_path,
        BuildOption=dummy_build_option,
        output_dir=test_dir_path,
    )
    # 获取新的命令行选项，并将其转换为列表形式
    new_cmd = dummy_builder.get_command_line().split(" ")

    # 调用函数比较新旧两个命令行选项的一致性
    _temp_validate_new_and_old_command(new_cmd, old_cmd)

    # 清理临时目录对象
    temp_dir.cleanup()


# TODO: 在切换到新的 cpp_builder 后，将删除这段临时代码
# 这段代码用于验证新的 cpp_builder 是否生成与旧代码相同的命令行
def validate_new_cpp_commands():
    # 定义一系列可能的测试参数组合
    cuda = [True, False]
    use_mmap_weights = [True, False]
    compile_only = [True, False]
    include_pytorch = [True, False]
    use_absolute_path = [True, False]

    # 嵌套循环遍历所有可能的参数组合
    for x in cuda:
        for y in use_mmap_weights:
            for z in compile_only:
                for m in include_pytorch:
                    for n in use_absolute_path:
                        # 打印当前参数组合，用于调试和确认
                        print(
                            f"!!! cuda:{x}, use_mmap_weights:{y}, compile_only:{z}, include_pytorch:{m}， use_absolute_path:{n}"
                        )
                        # 调用验证函数，传入当前参数组合
                        _do_validate_cpp_commands(
                            include_pytorch=m,
                            cuda=x,
                            mmap_weights=y,
                            compile_only=z,
                            use_absolute_path=n,
                        )


# 在清除新的缓存时，确保 HalideCodeCache 类的缓存也被清除
@clear_on_fresh_inductor_cache
class HalideCodeCache(CppPythonBindingsCodeCache):
    # 缓存字典，用于存储模块或动态链接库的回调函数
    cache: Dict[str, Callable[[], Union[ModuleType, CDLL]]] = {}
    # 静态方法，用于清空缓存字典
    cache_clear = staticmethod(cache.clear)
    # 可选的独立运行时路径
    _standalone_runtime_path: Optional[str] = None
    prefix = textwrap.dedent(
        """
        #include "{halideruntime_h}"  # 包含 Halide 运行时头文件
        #include "{headerfile}"        # 包含用户指定的头文件
        #include <stdexcept>           # 包含标准异常处理头文件
        #include <cmath>               # 包含数学函数头文件

        namespace c10 {{              # 进入命名空间 c10
            inline long div_floor_integer(long a, long b) {{  # 定义向下取整的整数除法函数
                if ((a<0) != (b<0)) {{                      # 检查除数和被除数是否异号
                    const auto quot = a / b;                # 计算商
                    const auto rem = a % b;                 # 计算余数
                    return rem ? quot - 1 : quot;           # 根据余数是否为零返回商减一或商
                }}
                return a / b;                              # 同号直接返回商
            }}
        }}
        """
    )
    glue_template_cpp = prefix + textwrap.dedent(
        """
        void kernel({argdefs}) {{         # 定义 C++ 版本的 kernel 函数
            {buffers}                     # 插入参数定义的缓冲区
            int err = halide_kernel({buffer_names});  # 调用 Halide 的 kernel 函数
            if(err != 0) throw std::runtime_error("halide_kernel failed");  # 检查函数调用是否成功
        }}
        """
    )
    glue_template_cuda = prefix + textwrap.dedent(
        """
        #include <cuda.h>                # 包含 CUDA 头文件
        static const halide_device_interface_t* cuda_interface = halide_cuda_device_interface();  # 定义 CUDA 设备接口

        void kernel({argdefs}, uintptr_t stream) {{  # 定义 CUDA 版本的 kernel 函数
            {buffers}                     # 插入参数定义的缓冲区
            int err = halide_kernel(reinterpret_cast<void*>(stream), {buffer_names});  # 调用 Halide 的 kernel 函数
            if(err != 0) throw std::runtime_error("halide_kernel failed");  # 检查函数调用是否成功
        }}
        """
    )
    standalone_runtime_cuda_init = textwrap.dedent(
        """
        #include "{}"                   # 包含用户指定的 CUDA 运行时头文件
        #include <cuda.h>                # 包含 CUDA 头文件

        static int acquire_context(void* user_context,
                                   void** cuda_context_out,
                                   bool create) {{  # 定义获取 CUDA 上下文的函数
            return cuCtxGetCurrent(reinterpret_cast<CUcontext*>(cuda_context_out));  # 获取当前 CUDA 上下文
        }}

        static int release_context(void* user_context) {{  # 定义释放 CUDA 上下文的函数
            return 0;                    # 简单返回成功状态
        }}

        static int get_stream(void* user_context,
                              void* cuda_context,
                              void** stream_out) {{  # 定义获取 CUDA 流的函数
            *stream_out = user_context;  # 将用户上下文作为 CUDA 流输出
            return 0;                    # 简单返回成功状态
        }}

        static int register_halide_hooks() {{  # 定义注册 Halide 钩子函数的函数
            halide_set_cuda_acquire_context(&acquire_context);  # 设置 CUDA 获取上下文的钩子函数
            halide_set_cuda_release_context(&release_context);  # 设置 CUDA 释放上下文的钩子函数
            halide_set_cuda_get_stream(&get_stream);            # 设置 CUDA 获取流的钩子函数
            return 0;                    # 简单返回成功状态
        }}

        int inductor_register_halide_hooks_result = register_halide_hooks();  # 注册 Halide 钩子函数并获取结果
        """
    )

    @classmethod
    @classmethod
    # 定义一个类方法，用于生成 Halide 缓冲区的代码
    def _codegen_buffer(cls, name: str, arg: HalideInputSpec, cuda: bool):
        # 断言确保参数的形状不为空
        assert arg.shape is not None
        # 断言确保参数的步长不为空且与形状的维度相同
        assert arg.stride is not None and len(arg.shape) == len(arg.stride)
        # 断言确保参数的偏移不为空
        assert arg.offset is not None
        # 根据参数的别名或名称以及偏移量生成数据指针的字符串表示
        data_ptr = f"{arg.alias_of or arg.name} + {arg.offset}"
        
        # 根据是否是 CUDA 环境设置不同的设备相关信息
        if cuda:
            device = f"reinterpret_cast<uint64_t>({data_ptr})"
            device_interface = "cuda_interface"
            host = "nullptr"
            flags = "halide_buffer_flag_device_dirty"
        else:
            device = "0"
            device_interface = "nullptr"
            # 将数据指针重新解释为 uint8_t* 类型作为主机指针
            host = f"reinterpret_cast<uint8_t*>({data_ptr})"
            flags = "halide_buffer_flag_host_dirty"
        
        # 初始化维度信息列表
        dims = []
        # 遍历参数的形状和步长，生成 HalideDimension 对象的字符串表示
        for size, stride in zip(arg.shape, arg.stride):
            dims.append(f"halide_dimension_t(0, {size}, {stride})")

        # 返回生成的 Halide 缓冲区对象的字符串表示列表
        return [
            f"halide_buffer_t {name};",  # 声明 Halide 缓冲区对象
            f"halide_dimension_t {name}_dims[] = {{{', '.join(dims)}}};",  # 声明缓冲区的维度数组
            f"{name}.device = {device};",  # 设置缓冲区的设备指针
            f"{name}.device_interface = {device_interface};",  # 设置缓冲区的设备接口
            f"{name}.host = {host};",  # 设置缓冲区的主机指针
            f"{name}.flags = {flags};",  # 设置缓冲区的标志位
            f"{name}.type = {arg.halide_type()};",  # 设置缓冲区的类型
            f"{name}.dimensions = {len(dims)};",  # 设置缓冲区的维度数
            f"{name}.dim = {name}_dims;",  # 设置缓冲区的维度信息
            f"{name}.padding = nullptr;",  # 设置缓冲区的填充信息为空指针
        ]

    @classmethod
    # 定义一个类方法，用于生成 Halide 与用户代码的接口粘合代码
    def _codegen_glue(cls, meta, headerfile):
        # 确定当前是否在 CUDA 环境下
        is_cuda = meta.is_cuda()
        # 断言确认是否需要使用用户上下文
        assert is_cuda is ("user_context" in meta.target)
        # 断言确认目标中是否包含 "no_runtime"
        assert "no_runtime" in meta.target
        
        buffers = []  # 初始化缓冲区列表
        buffer_names = []  # 初始化缓冲区名称列表
        
        # 遍历参数类型元数据列表
        for i, arg in enumerate(meta.argtypes):
            if arg.is_buffer():
                # 如果参数是缓冲区类型，则生成对应的缓冲区代码并添加到列表中
                buffer_names.append(f"&hl_buf_{i}")
                buffers.extend(cls._codegen_buffer(f"hl_buf_{i}", arg, is_cuda))
            else:
                # 否则，断言确认参数类型不包含指针
                assert "*" not in arg.ctype
                # 将参数名称添加到缓冲区名称列表中
                buffer_names.append(arg.name)
        
        # 将生成的缓冲区代码格式化为字符串并适当缩进
        buffers = "\n".join([f"    {line}" for line in buffers]).lstrip()

        # 根据当前是否在 CUDA 环境下选择合适的粘合模板
        glue_template = cls.glue_template_cuda if is_cuda else cls.glue_template_cpp
        
        # 使用模板格式化生成粘合代码
        glue_code = glue_template.format(
            halideruntime_h=cls.find_header(
                "HalideRuntimeCuda.h" if is_cuda else "HalideRuntime.h"
            ),
            headerfile=headerfile,
            argdefs=", ".join(
                f"{a.bindings_type()} {a.name}"
                for a in meta.argtypes
                if a.alias_of is None
            ),
            buffers=buffers,  # 插入生成的缓冲区代码
            buffer_names=", ".join(buffer_names),  # 插入缓冲区名称列表
        )
        
        # 返回生成的粘合代码字符串
        return glue_code

    @classmethod
    @functools.lru_cache(None)
    # 生成配置哈希值的方法，用于标识配置的唯一性
    def config_hash(cls):
        # 创建一个 CppBuilder 对象，用于构建 C++ 代码
        command_gen = CppBuilder(
            name="O",
            sources="I",
            BuildOption=CppOptions(compile_only=False),
        )
        # 获取生成的编译命令行
        command_line = command_gen.get_command_line()
        # 构建一个字符串列表，包含类的各种模板和编译命令行，计算其 SHA-256 哈希值
        return sha256_hash(
            "\n".join(
                [
                    cls.glue_template_cpp,
                    cls.glue_template_cuda,
                    cls.standalone_runtime_cuda_init,
                    command_line,
                ]
            ).encode("utf-8")
        )

    # 静态方法：搜索指定后缀文件并返回其路径
    @staticmethod
    def _search_for_file(suffix, errmsg):
        # 尝试寻找导入的 Halide 模块
        spec = importlib.machinery.PathFinder.find_spec("halide")
        # 如果未找到或没有子模块搜索位置，抛出运行时错误
        if spec is None or not spec.submodule_search_locations:
            raise RuntimeError("halide python bindings not installed")
        try:
            # 获取 Halide 模块的子模块搜索位置
            search = spec.submodule_search_locations[0]
            # 遍历搜索位置下的文件列表
            for file in os.listdir(search):
                # 如果文件以 .so 结尾，执行以下操作
                if file.endswith(".so"):
                    try:
                        # 执行命令获取动态链接库的依赖
                        out = subprocess.check_output(
                            ["ldd", os.path.join(search, file)]
                        )
                    except subprocess.SubprocessError:
                        continue
                    # 从依赖中搜索 libHalide.so 的路径
                    m = re.search(r"(/.*)/libHalide.so", out.decode("utf-8"))
                    if m:
                        # 构建特定后缀文件的绝对路径并检查是否存在，存在则返回路径
                        path = os.path.join(os.path.abspath(m.group(1)), suffix)
                        if os.path.exists(path):
                            return os.path.abspath(path)
        except Exception as e:
            # 捕获所有异常并抛出带错误消息的运行时错误
            raise RuntimeError(errmsg) from e
        # 如果未找到符合条件的文件，抛出运行时错误
        raise RuntimeError(errmsg)

    # 静态方法：根据名称查找 libautoschedule 动态链接库并返回其路径
    @staticmethod
    @functools.lru_cache(None)
    def find_libautoschedule(name):
        # 构建 libautoschedule_{name.lower()}.so 文件名
        sofile = f"libautoschedule_{name.lower()}.so"
        # 如果环境变量 HALIDE_LIB 存在，尝试构建文件路径并检查是否存在，存在则返回路径
        if "HALIDE_LIB" in os.environ:
            path = os.path.join(os.environ["HALIDE_LIB"], sofile)
            if os.path.exists(path):
                return path
        # 构建错误消息，提示设置 HALIDE_LIB 环境变量
        errmsg = (
            f"Can't find {sofile}, set env HALIDE_LIB to the directory containing it"
        )
        # 调用 _search_for_file 方法查找文件并返回路径
        return HalideCodeCache._search_for_file(sofile, errmsg)

    # 静态方法：根据名称查找头文件并返回其路径
    @staticmethod
    @functools.lru_cache(None)
    def find_header(name):
        # 如果环境变量 HALIDE_INCLUDE 存在，尝试构建文件路径并检查是否存在，存在则返回路径
        if "HALIDE_INCLUDE" in os.environ:
            path = os.path.join(os.environ["HALIDE_INCLUDE"], name)
            if os.path.exists(path):
                return path
        # 如果环境变量 HALIDE_LIB 存在，尝试构建文件路径并检查是否存在，存在则返回路径
        if "HALIDE_LIB" in os.environ:
            path = os.path.abspath(
                os.path.join(os.environ["HALIDE_LIB"], f"../include/{name}")
            )
            if os.path.exists(path):
                return path
        # 构建错误消息，提示设置 HALIDE_INCLUDE 环境变量
        errmsg = (
            f"Can't find {name}, set env HALIDE_INCLUDE to the directory containing it"
        )
        # 调用 _search_for_file 方法查找文件并返回路径
        return HalideCodeCache._search_for_file(f"../include/{name}", errmsg)

    # 类方法
    @classmethod
    # 定义生成 Halide 异步函数的类方法，接受 HalideMeta 元数据、源代码字符串和提交函数作为参数
    def generate_halide_async(cls, meta: HalideMeta, source_code: str, submit_fn=None):
        # 构造存储生成文件的目录路径，基于给定的 source_code 和 meta 生成唯一的哈希值
        dirpath = Path(
            get_path(
                code_hash(
                    source_code,
                    extra=repr((cls.config_hash(), meta)),
                ),
                "halide",
            )[2]
        )
        # 如果目录不存在则创建
        os.makedirs(dirpath, exist_ok=True)
        
        # 初始化变量
        wait_for_compile = None
        genfile = str(dirpath / "generate_kernel.py")  # 生成的核心代码文件路径
        libfile = str(dirpath / "halide_kernel.a")  # 生成的 Halide 静态库文件路径
        headerfile = str(dirpath / "halide_kernel.h")  # 生成的 Halide 头文件路径
        donefile = str(dirpath / "done")  # 标志编译完成的文件路径
        lockfile = str(dirpath / "lock")  # 控制并发访问的锁文件路径
        need_compile = not os.path.exists(donefile)  # 检查是否需要重新编译

        jobs = []

        # 如果需要重新编译，则写入生成的核心代码文件，并准备编译命令
        if need_compile:
            write_atomic(genfile, source_code)
            cmd = [
                sys.executable,
                genfile,
                "-g",
                "kernel",
                "-o",
                f"{dirpath}",
                "-f",
                "halide_kernel",
                "-e",
                "static_library,h,schedule",
            ]
            if meta.scheduler:
                cmd.extend(["-p", cls.find_libautoschedule(meta.scheduler)])
            cmd.extend(meta.args())
            jobs.append(functools.partial(subprocess.check_call, cmd))

        # 收集 HalideMeta 元数据中需要绑定的类型
        binding_types = [
            arg.bindings_type() for arg in meta.argtypes if arg.alias_of is None
        ]
        # 如果是 CUDA 相关的代码生成，添加额外的 binding 类型
        if meta.is_cuda():
            binding_types.append("uintptr_t")  # stream

        # 异步加载 Python 绑定的 HalideKernel
        bindings_future = cls.load_pybinding_async(
            binding_types,
            cls._codegen_glue(meta, headerfile),
            extra_flags=(libfile, cls.build_standalone_runtime()),
            submit_fn=jobs.append if need_compile else None,
            cuda=meta.is_cuda(),
        )

        # 如果需要重新编译，则添加任务标记编译完成，并创建任务函数
        if need_compile:
            jobs.append(functools.partial(touch, donefile))
            task = functools.partial(_worker_task_halide, lockfile, jobs)
            # 如果有提交函数，则调用提交函数提交任务，否则直接执行任务
            if submit_fn:
                wait_for_compile = submit_fn(task).result
            else:
                task()

        # 定义加载函数，等待编译完成后返回 HalideKernel 绑定的结果
        def load():
            if wait_for_compile:
                wait_for_compile()
            return bindings_future()

        # 返回加载函数
        return load

    # 类方法：调用 generate_halide_async 方法，并立即执行返回的加载函数
    @classmethod
    def generate_halide(cls, *args, **kwargs):
        return cls.generate_halide_async(*args, **kwargs)()

    # 类方法
    @classmethod
    # 构建独立运行时路径的方法
    def build_standalone_runtime(cls):
        # 如果已经设置了独立运行时路径，并且路径存在，则直接返回该路径
        if cls._standalone_runtime_path and os.path.exists(
            cls._standalone_runtime_path
        ):
            return cls._standalone_runtime_path
        # 检查当前环境是否支持 CUDA
        is_cuda = torch.cuda.is_available()
        # 设置 Halide 运行时库的名称
        libname = "libStandaloneHalideRuntime.so"
        # 根据 CUDA 是否可用选择 Halide 的编译目标
        target = "host-cuda" if is_cuda else "host"
        # 如果已经设置了独立运行时路径，确保该路径不存在，这种情况在运行单元测试时可能会发生
        if cls._standalone_runtime_path:
            assert not os.path.exists(cls._standalone_runtime_path)
            # 在运行单元测试时，由于重复初始化 CUDA 导致文件描述符耗尽的情况下，需要打破当前的缓存
            # 通过 jail breaking the current fresh_inductor_cache() 来解决这个问题
            # 详见 unittests 中的情况
            base = default_cache_dir()
        else:
            # 否则使用默认的缓存目录
            base = cache_dir()
        # 构建 Halide 运行时目录的路径，包含编译目标和配置哈希值
        dirpath = Path(base) / f"halide-runtime-{target}-{cls.config_hash()}"
        # 创建目录，如果目录已存在则不做任何操作
        os.makedirs(dirpath, exist_ok=True)
        # 生成标志文件的路径
        donefile = str(dirpath / "done")
        # 生成锁文件的路径
        lockfile = str(dirpath / "lock")
        # 生成钩子文件的路径
        hookfile = str(dirpath / "hooks.cpp")
        # 生成静态库文件的路径
        afile = str(dirpath / "standalone_halide_runtime.a")
        # 生成共享库文件的路径
        sofile = str(dirpath / libname)
        # 如果标志文件不存在，则开始构建独立运行时
        if not os.path.exists(donefile):
            import filelock
            import halide as hl  # type: ignore[import-untyped,import-not-found]

            # 使用文件锁确保只有一个进程在编译和构建独立运行时
            with filelock.FileLock(lockfile, LOCK_TIMEOUT):
                if not os.path.exists(donefile):
                    # 在钩子文件中写入初始化 CUDA 运行时所需的代码
                    with open(hookfile, "w") as f:
                        if is_cuda:
                            f.write(
                                cls.standalone_runtime_cuda_init.format(
                                    cls.find_header("HalideRuntimeCuda.h")
                                )
                            )
                    # 编译 Halide 的独立运行时
                    hl.compile_standalone_runtime(afile, hl.Target(target))
                    # 使用系统命令行调用编译器编译生成共享库
                    subprocess.check_call(
                        shlex.split(
                            cpp_compile_command([hookfile, afile], sofile, cuda=is_cuda)
                        )
                    )
                    # 创建一个空的标志文件，表示编译完成
                    touch(donefile)
        # 断言共享库文件存在
        assert os.path.exists(sofile)
        # 将生成的共享库文件路径保存到类的属性中，以备下次使用
        cls._standalone_runtime_path = sofile
        # 返回生成的共享库文件路径
        return sofile
# 定义一个私有函数 _worker_task_halide，用于处理 Halide 任务
def _worker_task_halide(lockfile, jobs):
    # 导入 FileLock 类从 filelock 模块
    from filelock import FileLock

    # 尝试获取文件锁并执行任务
    try:
        # 使用 FileLock 对象锁定指定的锁文件，超时时间为 LOCK_TIMEOUT
        with FileLock(lockfile, LOCK_TIMEOUT):
            # 遍历 jobs 列表中的每个任务并执行
            for job in jobs:
                job()
    # 捕获 subprocess.SubprocessError 异常
    except subprocess.SubprocessError as e:
        # 如果环境变量 HALIDE_REPRO 的值为 "1"
        if os.environ.get("HALIDE_REPRO") == "1":
            # 获取异常对象 e 的 cmd 属性作为命令列表
            python, script, *cmd = getattr(e, "cmd", ("", "", ""))
            # 如果 Python 解释器的基本名称以 "python" 开头
            if os.path.basename(python).startswith("python"):
                # 读取脚本文件的源代码
                code = open(script).read()
                # 确保源代码中只有一个 hl.main() 调用
                main = "    hl.main()"
                assert code.count(main) == 1

                # 定义一个类 Out 用于字符串表示为 "out"
                class Out:
                    def __repr__(self):
                        return "out"

                # 替换命令列表中 "-o" 选项后的参数为 Out 对象
                cmd[cmd.index("-o") + 1] = Out()  # type: ignore[call-overload]

                # 构建一个包含重现代码的字符串 repl
                repl = textwrap.indent(
                    textwrap.dedent(
                        f"""\
                        import sys, tempfile
                        with tempfile.TemporaryDirectory() as out:
                            sys.argv = {["repro.py", *cmd]!r}
                            hl.main()
                        """
                    ),
                    "    ",
                )
                # 替换源代码中的 hl.main() 调用为 repl 字符串
                code = code.replace(main, repl)

                # 将修改后的代码写入 repro.py 文件
                with open("repro.py", "w") as fd:
                    fd.write(code.lstrip())

                # 抛出 RuntimeError 异常，指示已写入 repro.py 文件
                raise RuntimeError(f"wrote repro.py: {e}") from e
        # 如果 HALIDE_REPRO 不为 "1"，继续抛出原始异常
        raise


# 定义一个函数 touch，用于创建一个空文件
def touch(filename):
    # 打开指定文件以附加模式，然后立即关闭，实现文件创建
    open(filename, "a").close()


# 使用装饰器 @clear_on_fresh_inductor_cache，定义一个 PyCodeCache 类
@clear_on_fresh_inductor_cache
class PyCodeCache:
    # 类变量 cache，用于存储字符串到模块对象的映射
    cache: Dict[str, ModuleType] = dict()
    # 类变量 linemaps，用于存储字符串到源代码行映射列表的映射
    linemaps: Dict[str, List[Tuple[Any, ...]]] = dict()
    # 静态方法 cache_clear，用于清空 cache 字典
    cache_clear = staticmethod(cache.clear)

    # 类方法 write，接受源代码字符串和额外参数，返回一个元组 (key, path)
    @classmethod
    def write(cls, source_code: str, extra: str = "") -> Tuple[str, str]:
        return write(source_code, "py", extra=extra)

    # 类方法 load，接受源代码字符串、额外参数、源代码行映射和属性字典，返回加载的模块对象
    @classmethod
    def load(
        cls,
        source_code: str,
        extra: str = "",
        linemap: Optional[List[Tuple[int, str]]] = None,
        attrs: Optional[Dict[str, Any]] = None,
    ) -> ModuleType:
        # 调用 write 方法生成 key 和 path
        key, path = write(source_code, "py", extra=extra)
        # 调用 load_by_key_path 方法加载模块并返回
        return cls.load_by_key_path(key, path, linemap, attrs)

    # 类方法 load_by_key_path，接受 key、path、源代码行映射和属性字典，返回加载的模块对象
    @classmethod
    def load_by_key_path(
        cls,
        key: str,
        path: str,
        linemap: Optional[List[Tuple[int, str]]] = None,
        attrs: Optional[Dict[str, Any]] = None,

        ) -> ModuleType:
        # 调用 write 方法生成 key 和 path，写入源代码和额外参数
        key, path = write(source_code, "py", extra=extra)
        # 返回调用 load_by_key_path 方法加载的模块对象
        return cls.load_by_key_path(key, path, linemap, attrs)
    ) -> ModuleType:
        # 如果 linemap 参数为 None，则设为一个空列表
        if linemap is None:
            linemap = []
        
        # 如果 key 不在 cls.cache 中，重新加载 Python 模块并放入缓存
        if key not in cls.cache:
            mod = _reload_python_module(key, path)

            # 另一个线程可能会先设置这个，使用 setdefault 确保线程安全地设置模块到缓存中
            cls.cache.setdefault(key, mod)
            
            # 将 linemap 解压成单独的行/节点列表，并存入 cls.linemaps 中
            cls.linemaps[path] = list(zip(*linemap))

            # 如果 attrs 不为 None，则将其属性设置到 mod 对象中
            if attrs is not None:
                for k, v in attrs.items():
                    setattr(mod, k, v)

            # 如果 linemap 和 attrs 都为空，则将 _reload_in_subproc 设置为在子进程中重新加载模块的部分函数
            if not (linemap or attrs):
                mod._reload_in_subproc = functools.partial(  # type: ignore[attr-defined]
                    _reload_python_module_in_subproc, key, path
                )

        # 返回缓存中的模块对象
        return cls.cache[key]

    @classmethod
    @functools.lru_cache(None)
    def stack_frames_for_code(
        cls, path: str, lineno: int
    ) -> Optional[List[Dict[str, Any]]]:
        # 如果指定路径不在 cls.linemaps 中，则返回 None
        if path not in cls.linemaps:
            return None
        
        # 获取路径对应的 lines 和 nodes 列表
        lines, nodes = cls.linemaps[path]
        
        # 使用二分查找定位给定行号在 lines 中的位置 p
        p = bisect_right(lines, lineno)
        
        # 如果 p 为 0，表示未找到对应的节点，返回 None
        if p == 0:
            return None
        
        # 获取对应的节点 entry
        entry = nodes[p - 1]
        
        # 如果 entry 为空，返回 None
        if not entry:
            return None

        # 定义解析堆栈跟踪信息的函数 parse_stack_trace
        def parse_stack_trace(stack_trace: str) -> List[Dict[str, Any]]:
            # 理想情况下，fx 应将堆栈跟踪信息存储为数据而不是字符串，但这不是性能关键路径
            # 使用正则表达式匹配堆栈跟踪信息
            regex = r'File "(.+)", line (\d+), in (.+)\n'
            matches = re.findall(regex, stack_trace)
            # 反转匹配结果并以字典形式返回
            return [
                {"filename": f, "line": int(l), "name": n}
                for f, l, n in reversed(matches)
            ]

        # 调用 parse_stack_trace 函数解析 entry 中的堆栈跟踪信息并返回结果
        return parse_stack_trace(entry)
class TritonCodeCache:
    @classmethod
    def load(cls, kernel_name: str, source_code: str) -> ModuleType:
        # 调用 PyCodeCache 的 load 方法加载源代码，返回对应的 Triton 内核模块
        return _module_to_triton_kernel(PyCodeCache.load(source_code), kernel_name)


def _cuda_compiler() -> Optional[str]:
    # 检查是否存在指定路径下的 nvcc 编译器
    if cuda_env.nvcc_exist(config.cuda.cuda_cxx):
        return config.cuda.cuda_cxx
    # 如果是在 FBCode 环境下，返回预定义路径下的 nvcc 编译器
    if config.is_fbcode():
        return os.path.join(build_paths.cuda(), "bin", "nvcc")
    # 检查环境变量 CUDACXX 是否指定了 nvcc 编译器路径
    if cuda_env.nvcc_exist(os.getenv("CUDACXX")):
        return os.getenv("CUDACXX", "")
    # 检查环境变量 CUDA_HOME 是否指定了 CUDA 根目录，返回其下 nvcc 的绝对路径
    if cuda_env.nvcc_exist(os.getenv("CUDA_HOME")):
        return os.path.realpath(os.path.join(os.getenv("CUDA_HOME", ""), "bin/nvcc"))
    # 默认返回 nvcc，表示未找到其他指定的编译器路径
    return "nvcc"


def _cutlass_include_paths() -> List[str]:
    # 如果是在 FBCode 环境下，通过 libfb.py 的 parutil 获取 cutlass-3-headers 路径
    if config.is_fbcode():
        from libfb.py import parutil
        cutlass_path = parutil.get_dir_path("cutlass-3-headers")
    else:
        # 否则使用配置中的 cutlass_dir 路径
        cutlass_path = config.cuda.cutlass_dir
    # 返回包含 cutlass 库的各个路径的绝对路径列表
    return [
        os.path.realpath(os.path.join(cutlass_path, "include")),
        os.path.realpath(os.path.join(cutlass_path, "tools/library/include")),
        os.path.realpath(os.path.join(cutlass_path, "tools/library/src")),
        os.path.realpath(os.path.join(cutlass_path, "tools/util/include")),
    ]


def _cuda_lib_options() -> List[str]:
    # 设置 GPU 运行时环境变量，cpp_extension 依赖此环境
    _set_gpu_runtime_env()
    from torch.utils import cpp_extension

    # 获取 CUDA 相关的库路径列表，并添加系统配置的 LIBDIR 路径
    lpaths = cpp_extension.library_paths(cuda=True) + [
        sysconfig.get_config_var("LIBDIR")
    ]
    extra_ldflags: List[str] = []
    if is_linux():
        # 在 Linux 下，转换 CUDA 库的路径格式，并设置链接选项
        _transform_cuda_paths(lpaths)
        for path in lpaths:
            # -rpath 确保动态库加载时能够找到其依赖项，即使库路径非标准
            extra_ldflags.extend([f"-L{path}", "-Xlinker", f"-rpath={path}"])
        # 添加 CUDA 和 CUDA Runtime 库的链接选项
        extra_ldflags.append("-lcuda")
        extra_ldflags.append("-lcudart")
    else:
        # 如果不是 Linux 环境，抛出未实现的错误，只支持 Linux
        raise NotImplementedError(
            "Unsupported env, failed to find cuda libs! Currently only Linux is supported."
        )
    # 返回额外的链接选项列表
    return extra_ldflags


def _nvcc_host_compiler_options() -> List[str]:
    # 返回 nvcc 主机编译器的选项列表
    return [
        "-fPIC",
        "-fno-strict-aliasing",
        "-fvisibility=hidden",
        "-Wconversion",
    ]


def _nvcc_compiler_options() -> List[str]:
    # 获取当前 CUDA 架构
    arch = cuda_env.get_cuda_arch()
    if arch == "90":
        # 由于 cutlass 编译需要，将架构设置为 "90a"
        arch = "90a"
    code = [f"sm_{arch}", f"compute_{arch}"]
    if config.cuda.enable_cuda_lto:
        # 如果启用了 CUDA LTO，添加相应的编译选项
        code += [f"lto_{arch}"]
    options = [
        "-t=0",
        "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
        "-w",
        f"-gencode=arch=compute_{arch},code=[{','.join(code)}]",
        config.cuda.compile_opt_level,
        "-std=c++17",
        "--expt-relaxed-constexpr",
        "-DNDEBUG",
    ]
    if config.is_fbcode():
        # 如果是在 FBCode 环境下，指定使用的 C++ 编译器路径
        options.extend(["-ccbin", os.path.dirname(build_paths.gcc())])
    # 如果配置中启用了 CUDA 调试信息
    if config.cuda.enable_debug_info:
        # 将调试选项扩展到编译选项中，包括行信息、调试信息，并设置 CUTLASS_DEBUG_TRACE_LEVEL 为 1
        options.extend(["-lineinfo", "-g", "-DCUTLASS_DEBUG_TRACE_LEVEL=1"])

    # 如果配置中启用了 PTXAS 信息
    if config.cuda.enable_ptxas_info:
        options.extend(
            [
                "--keep",  # 保留中间文件以供调试使用（包括 ptx、sass、cubin 等）
                "--ptxas-options=--warn-on-local-memory-usage",  # 在 CUDA Kernel 中如果使用了局部内存则发出警告
                "--ptxas-options=--warn-on-spills",  # 在 CUDA Kernel 中如果发生寄存器溢出则发出警告
                "--resource-usage",  # 报告 CUDA 资源使用情况（共享内存、寄存器等）
                "--source-in-ptx",
            ]
        )  # 在 ptx 文件中添加源代码信息的注释

    # 如果配置中启用了 CUDA 快速数学运算选项
    if config.cuda.use_fast_math:
        options.extend(
            [
                "--use_fast_math",  # 启用快速数学运算
                "-DCUTLASS_USE_TANH_FOR_SIGMOID=1",  # 使用双曲正切替代 sigmoid 函数
            ]
        )

    # 返回最终的编译选项列表
    return options
# 定义一个函数，用于生成 CUDA 编译命令字符串
def cuda_compile_command(
    src_files: List[str],  # 输入源文件列表
    dst_file: str,  # 目标文件名
    dst_file_ext: str,  # 目标文件扩展名
    extra_args: Optional[List[str]] = None,  # 额外的编译参数，默认为 None
) -> str:  # 函数返回一个字符串
    if extra_args is None:
        extra_args = []  # 如果额外参数为 None，则置为空列表
    include_paths = _cutlass_include_paths()  # 调用私有函数获取包含路径
    cuda_lib_options = _cuda_lib_options()  # 调用私有函数获取 CUDA 库选项
    nvcc_host_compiler_options = _nvcc_host_compiler_options()  # 调用私有函数获取 nvcc 主机编译器选项
    nvcc_compiler_options = _nvcc_compiler_options()  # 调用私有函数获取 nvcc 编译器选项
    options = (
        nvcc_compiler_options  # 添加 nvcc 编译器选项到选项列表
        + extra_args  # 添加额外参数到选项列表
        + [
            f"-Xcompiler {opt}" if "=" in opt else f"-Xcompiler={opt}"
            for opt in nvcc_host_compiler_options  # 处理 nvcc 主机编译器选项
        ]
        + ["-I" + path for path in include_paths]  # 添加包含路径到选项列表
        + cuda_lib_options  # 添加 CUDA 库选项到选项列表
    )
    src_file = " ".join(src_files)  # 将源文件列表转换为字符串
    res = ""  # 初始化结果字符串
    if dst_file_ext == "o":  # 如果目标文件扩展名是 'o'
        res = f"{_cuda_compiler()} {' '.join(options)} -c -o {dst_file} {src_file}"  # 生成编译命令
    elif dst_file_ext == "so":  # 如果目标文件扩展名是 'so'
        options.append("-shared")  # 添加共享选项到选项列表
        res = f"{_cuda_compiler()} {' '.join(options)} -o {dst_file} {src_file}"  # 生成编译命令
    elif dst_file_ext == "exe":  # 如果目标文件扩展名是 'exe'
        res = f"{_cuda_compiler()} {' '.join(options)} -o {dst_file} {src_file}"  # 生成编译命令
    else:
        raise NotImplementedError(f"Unsupported output file suffix {dst_file_ext}!")  # 抛出未实现错误
    log.debug("CUDA command: %s", res)  # 记录调试信息
    return res  # 返回编译命令字符串


class DLLWrapper:
    """A wrapper for a dynamic library."""

    def __init__(
        self,
        lib_path: str,  # 动态库文件路径
    ):
        self.lib_path = lib_path  # 设置动态库文件路径属性
        self.is_open = False  # 标识动态库是否已打开
        self.DLL = cdll.LoadLibrary(lib_path)  # 加载动态库
        self.is_open = True  # 设置标识为已打开状态

    def close(self):  # 关闭动态库的方法
        if self.is_open:  # 如果动态库已打开
            self._dlclose()  # 调用内部方法进行关闭
            self.is_open = False  # 设置标识为已关闭状态

    def _dlclose(self):  # 实际执行动态库关闭操作的方法
        f_dlclose = None  # 初始化动态库关闭函数变量为 None

        if is_linux():  # 如果当前环境是 Linux
            syms = CDLL(None)  # 使用默认加载器创建 CDLL 对象
            if not hasattr(syms, "dlclose"):  # 如果 CDLL 对象没有 dlclose 属性
                # Apline Linux
                syms = CDLL("libc.so")  # 尝试加载 libc.so 库

            if hasattr(syms, "dlclose"):  # 如果 CDLL 对象有 dlclose 属性
                f_dlclose = syms.dlclose  # 获取 dlclose 函数
        else:
            raise NotImplementedError("Unsupported env, failed to do dlclose!")  # 抛出未实现错误

        if f_dlclose is not None:  # 如果成功获取到动态库关闭函数
            f_dlclose.argtypes = [c_void_p]  # 设置 dlclose 函数参数类型
            f_dlclose(self.DLL._handle)  # 调用 dlclose 函数关闭动态库
        else:
            log.warning(
                "dll unloading function was not found, library may not be unloaded properly!"
            )  # 记录警告信息，动态库可能没有正确卸载

    def __getattr__(self, name):  # 动态获取属性的方法
        if not self.is_open:  # 如果动态库未打开
            raise RuntimeError(f"Cannot use closed DLL library: {self.lib_path}")  # 抛出运行时错误

        method = getattr(self.DLL, name)  # 获取动态库对象的属性方法

        def _wrapped_func(*args):  # 定义内部包装函数，执行动态库方法
            err = method(*args)  # 调用动态库方法
            if err:  # 如果返回错误
                raise RuntimeError(f"Error in function: {method.__name__}")  # 抛出运行时错误

        return _wrapped_func  # 返回包装函数

    def __enter__(self):  # 实现上下文管理器的进入方法
        return self  # 返回对象本身

    def __exit__(self, *args):  # 实现上下文管理器的退出方法
        self.close()  # 调用关闭方法

    def __del__(self):  # 对象销毁时调用的方法
        self.close()  # 调用关闭方法


@clear_on_fresh_inductor_cache  # 装饰器，用于清除新的电感缓存
class CUDACodeCache:  # CUDA 代码缓存类
    @dataclasses.dataclass  # 数据类装饰器
    class CacheEntry:  # 缓存条目类
        input_path: str  # 输入路径
        output_path: str  # 输出路径
    # 定义一个静态缓存字典，用于存储编译后的 CUDA 代码和相关信息
    cache: Dict[str, CacheEntry] = dict()
    
    # 将cache.clear方法设为静态方法cache_clear
    cache_clear = staticmethod(cache.clear)
    
    # 定义一个私有类属性，表示CUDA源代码文件的后缀名
    _SOURCE_CODE_SUFFIX = "cu"
    
    # 定义一个类方法write，用于将源代码写入文件，并返回源代码的哈希键和文件路径
    @classmethod
    def write(cls, source_code, dst_file_ext) -> Tuple[str, str]:
        """
        Writes source code into a file with dst_file_ext as the file extension.
        Returns the hash key of source code, and the path to the file.
        """
        # 生成CUDA编译命令的字符串表示形式
        cuda_command = repr(
            cuda_compile_command(["dummy_input"], "dummy_output", dst_file_ext)
        )
        # 调用write函数将源代码写入文件，获取哈希键和输入文件路径
        key, input_path = write(
            source_code, cls._SOURCE_CODE_SUFFIX, extra=cuda_command
        )
        return key, input_path
    
    # 定义一个类方法compile，用于编译CUDA源代码为指定扩展名的文件
    @classmethod
    def compile(
        cls, source_code, dst_file_ext, extra_args: Optional[List[str]] = None
    ) -> Tuple[str, str, str]:
        """
        Compiles CUDA source_code into a file with dst_file_ext extension.
        Returns a tuple of dst_file_path, hash_key, source_code_path
        """
        # 调用write方法将源代码写入文件，获取哈希键和输入文件路径
        key, input_path = cls.write(source_code, dst_file_ext)
        
        # 如果哈希键不在缓存中
        if key not in cls.cache:
            # 导入FileLock类，用于文件锁
            from filelock import FileLock
            
            # 获取锁的目录路径
            lock_dir = get_lock_dir()
            # 创建文件锁对象
            lock = FileLock(os.path.join(lock_dir, key + ".lock"), timeout=LOCK_TIMEOUT)
            
            # 使用文件锁
            with lock:
                # 构建输出文件路径
                output_path = input_path[: -len(cls._SOURCE_CODE_SUFFIX)] + dst_file_ext
                
                # 如果输出文件不存在
                if not os.path.exists(output_path):
                    # 构建CUDA编译命令
                    cmd = cuda_compile_command(
                        [input_path], output_path, dst_file_ext, extra_args
                    )
                    start_time = time()
                    log.debug("CUDA Compilation: %s", cmd)
                    cmd_parts = cmd.split(" ")
                    try:
                        # 执行CUDA编译命令
                        subprocess.check_output(
                            cmd_parts, stderr=subprocess.STDOUT, env=os.environ
                        )
                    except subprocess.CalledProcessError as error:
                        # 如果编译出错，抛出CUDA编译错误异常
                        raise exc.CUDACompileError(cmd_parts, error.output) from error
                    end_time = time()
                    # 记录CUDA编译持续时间和编译命令
                    log_duration_msg = f"CUDA Compilation took {end_time-start_time} seconds. Compile command: {cmd}"
                    log.info(log_duration_msg)
                else:
                    # 如果输出文件已存在，则跳过CUDA编译
                    log.debug(
                        "CUDA Compilation skipped: %s since output already exists",
                        input_path,
                    )
                
                # 将编译结果加入缓存
                cls.cache[key] = CUDACodeCache.CacheEntry(input_path, output_path)
    
        # 返回编译后的文件路径、哈希键和输入文件路径
        return (cls.cache[key].output_path, key, input_path)
    # 定义一个类方法，用于加载并编译源代码，并加载生成的 .so 文件
    def load(cls, source_code, dst_file_ext) -> Tuple[DLLWrapper, str, str]:
        """
        Compiles source code and loads the generated .so file.
        Returns a tuple of DLLWrapper, hash_key, source_code_path
        """

        # 如果目标文件扩展名不是 "so"，则抛出运行时错误
        if dst_file_ext != "so":
            raise RuntimeError(
                f"Only support loading a .so file for now. "
                f"Requested file extension: {dst_file_ext}. Source code: {source_code}"
            )
        
        # 调用类方法 compile 来编译源代码，获取编译后的文件路径、哈希键和源代码路径
        dst_file_path, hash_key, source_code_path = cls.compile(
            source_code, dst_file_ext
        )
        
        # 返回一个包含 DLLWrapper 对象、哈希键和源代码路径的元组
        return (DLLWrapper(dst_file_path), hash_key, source_code_path)
@clear_on_fresh_inductor_cache
class ROCmCodeCache:
    @dataclasses.dataclass
    class CacheEntry:
        input_path: str
        output_path: str

    # 定义一个类级别的缓存字典，用于存储输入路径到CacheEntry对象的映射关系
    cache: Dict[str, CacheEntry] = dict()
    # 设置静态方法cache_clear，用于清空缓存
    cache_clear = staticmethod(cache.clear)
    # 定义一个私有类变量，指定源代码文件的后缀名为"cpp"
    _SOURCE_CODE_SUFFIX = "cpp"
    # 初始化标志位，用于记录编译器版本是否已经被记录过
    _logged_compiler_version = False

    @classmethod
    def write(cls, source_code, dst_file_ext) -> Tuple[str, str]:
        """
        Writes source code into a file with dst_file_ext as the file extension.
        Returns the hash key of source code, and the path to the file.
        """

        # 调用rocm_compile_command函数生成一个ROCM编译命令，然后将其表示为字符串
        cuda_command = repr(
            rocm_compile_command(["dummy_input"], "dummy_output", dst_file_ext)
        )
        # 调用write函数将源代码写入文件，并返回源代码的哈希键和文件路径
        key, input_path = write(
            source_code, cls._SOURCE_CODE_SUFFIX, extra=cuda_command
        )
        return key, input_path

    @classmethod
    def compile(
        cls, source_code, dst_file_ext, extra_args: Optional[List[str]] = None
        ):
        """
        Compiles source_code into a file with dst_file_ext as the file extension.
        Optional extra_args can be passed for additional compilation arguments.
        """
    ) -> Tuple[str, str, str]:
        """
        Compiles source_code into a file with dst_file_ext extension,
        using the compile command specific for the ROCm platform.
        Returns a tuple of dst_file_path, hash_key, source_code_path
        """
        # 如果尚未记录编译器版本信息，则记录下来并打印出来
        if not cls._logged_compiler_version:
            cls._logged_compiler_version = True
            log.debug(get_compiler_version_info(str(rocm_compiler())))

        # 将源代码写入文件，并获取关键字和文件路径
        key, input_path = cls.write(source_code, dst_file_ext)

        # 如果缓存中没有当前关键字的条目，则开始编译
        if key not in cls.cache:
            from filelock import FileLock

            # 获取文件锁目录和文件锁本身
            lock_dir = get_lock_dir()
            lock = FileLock(os.path.join(lock_dir, key + ".lock"), timeout=LOCK_TIMEOUT)
            with lock:
                # 构建输出路径，如果输出路径不存在则执行编译命令
                output_path = input_path[: -len(cls._SOURCE_CODE_SUFFIX)] + dst_file_ext
                if not os.path.exists(output_path):
                    # 构建 ROCm 平台特定的编译命令
                    cmd = rocm_compile_command(
                        [input_path], output_path, dst_file_ext, extra_args
                    )
                    start_time = time()
                    cmd_parts = cmd.split(" ")
                    try:
                        # 执行编译命令，并捕获输出信息
                        output = subprocess.check_output(
                            cmd_parts,
                            stderr=subprocess.STDOUT,
                            text=True,
                            env=os.environ,
                        )
                        log.debug("Compilation output: %s", output)
                    except subprocess.CalledProcessError as error:
                        # 如果编译出错，则抛出自定义的编译错误异常
                        raise exc.CUDACompileError(cmd_parts, error.output) from error
                    end_time = time()
                    # 记录编译耗时和编译命令
                    log_duration_msg = f"Compilation took {end_time-start_time} seconds. Compile command: {cmd}"
                    log.info(log_duration_msg)
                else:
                    # 如果输出文件已存在，则记录跳过编译的信息
                    log.debug(
                        "Compilation skipped: %s since output already exists",
                        input_path,
                    )
                # 缓存编译结果
                cls.cache[key] = ROCmCodeCache.CacheEntry(input_path, output_path)

        # 返回编译后的输出路径、关键字和源代码路径的元组
        return (cls.cache[key].output_path, key, input_path)

    @classmethod
    def load(cls, source_code, dst_file_ext) -> Tuple[DLLWrapper, str, str]:
        """
        Compiles source code and loads the generated .so file.
        Returns a tuple of DLLWrapper, hash_key, source_code_path
        """

        # 检查目标文件扩展名是否为 .so，如果不是则抛出运行时错误
        if dst_file_ext != "so":
            raise RuntimeError(
                f"Only support loading a .so file for now. "
                f"Requested file extension: {dst_file_ext}. Source code: {source_code}"
            )
        
        # 调用编译方法生成 .so 文件，并获取返回的 DLLWrapper 对象、关键字和源代码路径
        dst_file_path, hash_key, source_code_path = cls.compile(
            source_code, dst_file_ext
        )
        
        # 返回 DLLWrapper 对象、关键字和源代码路径的元组
        return (DLLWrapper(dst_file_path), hash_key, source_code_path)
class CodeCacheFuture:
    # 定义一个抽象类 CodeCacheFuture，包含一个未实现的方法 result()
    def result(self):
        raise NotImplementedError


class TritonFuture(CodeCacheFuture):
    kernel: ModuleType

    def __init__(
        self,
        kernel: Any,
        future: Optional[Future[Any]],
    ) -> None:
        # TritonFuture 类的初始化方法，接受一个 kernel 对象和一个可选的 future 对象
        self.kernel = kernel
        self.future = future

    # @dynamo_utils.dynamo_timed
    def result(self) -> ModuleType:
        # 返回 TritonFuture 实例的结果，这里返回的是 kernel 对象
        if self.future is not None:
            # 如果存在 future 对象，则获取其结果
            result = self.future.result()
            # 断言结果为 None
            assert result is None
            # 将 future 设置为 None，清理资源
            self.future = None
            # 调用 kernel 的预编译方法
            self.kernel.precompile()
        # 返回 kernel 对象作为结果
        return self.kernel


class LambdaFuture(CodeCacheFuture):
    def __init__(self, result_fn):
        # LambdaFuture 类的初始化方法，接受一个结果生成函数 result_fn
        self.result_fn = result_fn

    def result(self):
        # 返回 LambdaFuture 实例的结果，通过调用 result_fn() 获取
        return self.result_fn()
```