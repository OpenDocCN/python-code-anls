# `.\pytorch\torch\_dynamo\utils.py`

```py
# mypy: allow-untyped-defs
# 导入必要的模块和库
import atexit  # 用于注册退出函数的模块
import collections  # 提供额外的数据结构，如Counter和defaultdict
import contextlib  # 用于支持上下文管理的工具集
import copy  # 提供对象复制功能
import dataclasses  # 用于定义数据类的模块
import datetime  # 提供处理日期和时间的模块
import dis  # 提供Python字节码的反汇编功能
import enum  # 支持枚举类型的模块
import functools  # 提供函数式编程的工具，如装饰器
import gc  # Python的垃圾回收模块
import inspect  # 提供对Python对象的反射功能
import itertools  # 提供创建迭代器的函数
import linecache  # 用于按行访问文本文件的缓存模块
import logging  # Python的标准日志模块
import math  # 数学函数库
import operator  # 提供Python内置运算符的函数形式
import os  # 提供与操作系统交互的功能
import re  # 提供正则表达式操作的模块
import sys  # 提供对Python解释器的访问和控制
import textwrap  # 提供简单的文本包装和填充功能
import threading  # 多线程编程支持模块
import time  # 提供时间处理功能
import types  # 提供动态创建和操作Python类型的功能
import typing  # 提供类型提示相关的功能
import warnings  # 提供警告控制功能
import weakref  # 提供弱引用支持

from contextlib import contextmanager  # 导入上下文管理器装饰器
from functools import lru_cache, wraps  # 导入LRU缓存和装饰器装饰函数的功能
from types import MethodWrapperType  # 导入方法包装类型

from typing import (  # 导入类型提示中需要使用的类型
    Any,
    Callable,
    cast,
    ClassVar,
    Counter,
    DefaultDict,
    Deque,
    Dict,
    Iterator,
    KeysView,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
    ValuesView,
)

from ..utils.hooks import RemovableHandle  # 导入自定义模块中的RemovableHandle类

try:
    import numpy as np  # 尝试导入NumPy库，若失败则设为None
except ModuleNotFoundError:
    np = None  # type: ignore[assignment]

try:
    import torch._logging  # 尝试导入torch._logging模块
    import torch._numpy as tnp  # 导入torch._numpy模块作为tnp
    from torch._guards import detect_fake_mode  # noqa: F401n
    from torch._logging import LazyString  # 导入torch._logging模块中的LazyString类
    from . import config  # 导入当前包中的config模块

    # NOTE: Make sure `NP_SUPPORTED_MODULES` and `NP_TO_TNP_MODULE` are in sync.
    if np:
        NP_SUPPORTED_MODULES: Tuple[types.ModuleType, ...] = (
            np,
            np.fft,
            np.linalg,
            np.random,
        )

        NP_TO_TNP_MODULE = {
            np: tnp,
            np.fft: tnp.fft,
            np.linalg: tnp.linalg,
            np.random: tnp.random,
        }
    else:
        NP_SUPPORTED_MODULES = tuple()

        NP_TO_TNP_MODULE = {}
    from torch._subclasses.fake_tensor import FakeTensor, is_fake, maybe_get_fake_mode  # 导入伪张量相关的模块
except ImportError:
    pass

import importlib  # 提供动态导入模块的功能

import torch  # 导入PyTorch主模块
import torch._functorch.config  # 导入PyTorch的functorch配置模块
import torch.fx.experimental.symbolic_shapes  # 导入PyTorch FX实验性符号形状模块
import torch.utils._pytree as pytree  # 导入PyTorch的pytree模块
from torch import fx  # 导入PyTorch FX模块
from torch._dispatch.python import enable_python_dispatcher  # 导入PyTorch的Python分发器模块
from torch._guards import TracingContext  # 导入PyTorch的TracingContext模块
from torch._subclasses.meta_utils import is_sparse_compressed  # 导入稀疏压缩相关的元数据工具
from torch._utils_internal import log_compilation_event  # 导入内部的编译事件日志功能

from torch.fx._utils import _format_graph_code, lazy_format_graph_code  # 导入FX工具函数
from torch.nn.modules.lazy import LazyModuleMixin  # 导入延迟模块混合类
from torch.utils._triton import has_triton, has_triton_package  # 导入Triton相关的功能

unpatched_nn_module_getattr = torch.nn.Module.__getattr__  # 备份未打补丁的nn.Module的__getattr__方法

# 用于统计各种计数器的字典，默认值为Counter
counters: DefaultDict[str, Counter[str]] = collections.defaultdict(collections.Counter)

# 优化Scuba日志的字典，初始为空字典
optimus_scuba_log: Dict[str, Any] = {}

# 用于故障排除的URL
troubleshooting_url = (
    "https://pytorch.org/docs/main/torch.compiler_troubleshooting.html"
)

# NN模块文档的URL
nnmodule_doc_url = "https://pytorch.org/docs/main/torch.compiler_nn_module.html"

# 用于指示NN模块文档URL信息的消息
nnmodule_doc_url_msg = f"See {nnmodule_doc_url} for more information and limitations."

# 获取当前模块的日志记录器
log = logging.getLogger(__name__)

# 用于按函数分析编译时间的指标字典
compilation_time_metrics: Dict[str, List[float]] = {}

# 用于按帧阶段分析编译时间的计时字典
frame_phase_timing: Dict[str, Dict[str, float]] = collections.defaultdict(
    lambda: collections.defaultdict(float)
)
# Counting iterator used for timer purposes
timer_counter = itertools.count()

# Function to format data into a tabular form, uses 'tabulate' library if available
def tabulate(rows, headers):
    try:
        import tabulate
        return tabulate.tabulate(rows, headers=headers)
    except ImportError:
        # If 'tabulate' is not available, format rows into a string with comma-separated values
        return "\n".join(", ".join(map(str, row)) for row in itertools.chain([headers], rows))

# Global variable to keep track of the current frame number
curr_frame = 0

# Note: This function is called automatically and should not be invoked manually
def increment_frame():
    global curr_frame
    curr_frame = curr_frame + 1

# Note: This function is called automatically and should not be invoked manually
def reset_frame_count():
    global curr_frame
    # Clear dictionaries used for timing phases
    frame_phase_timing.clear()
    compilation_time_metrics.clear()
    curr_frame = 0

# Global variable to count operations
op_count = 0

# Function to increment operation count
def increment_op_count(cnt):
    global op_count
    op_count += cnt

# Calculate total time spent in each phase based on recorded timings
# Returns a dictionary where keys are phase names and values are total times
def calculate_time_spent():
    total = 0.0
    total_by_key = {}
    # Iterate through timing data stored in 'frame_phase_timing'
    for timings in frame_phase_timing.values():
        for key, timing in timings.items():
            total += timing
            if key not in total_by_key:
                total_by_key[key] = timing
            else:
                total_by_key[key] += timing
    return total_by_key

# Print a report of total time spent in each phase so far
def print_time_report():
    total_by_key = calculate_time_spent()
    # Format output string to display phase names and their respective total times
    out = "TIMING:"
    for key, value in total_by_key.items():
        out = f"{out} {key}:{round(value, 5)}"
    print(out)

# Add time spent in a specific phase to the timing dictionary
def _add_time_spent(key, phase_name, time_spent):
    frame_phase_timing[key][phase_name] += time_spent

# Decorator function 'dynamo_timed' for measuring execution time of functions
# Records timing data in 'compilation_time_metrics' with function names as keys
# Allows specifying 'phase_name' to categorize timings under different compilation phases
# 'fwd_only' flag indicates if the function is only called during forward graph compilation
def dynamo_timed(original_function=None, phase_name=None, fwd_only=True):
    if original_function:
        return dynamo_timed_inner(original_function)
    return dynamo_timed_inner

# Function to retrieve compilation timing metrics
# Returns metrics about frontend/backend compilation times in a specified representation format
def compile_times(repr="str", aggregate=False):
    """
    Get metrics about torchdynamo frontend/backend compilation times.
    This function retrieves compilation metrics based on 'repr' and 'aggregate' options.
    """
    # 根据 `@dynamo_timed` 标记的函数累积信息。

    # repr='str' 返回一个可打印的字符串，用于用户交互；'csv' 返回头部和行，可用于输出日志。

    # aggregate 指示是否将多个编译（例如分割图）的值累积为一个值。如果为 False，则每个指标可能有多个值。

    def fmt_fn(values, item_fn=lambda x: x):
        # 如果 aggregate 为 True，返回所有值的总和，经过 item_fn 处理后的结果
        if aggregate:
            return item_fn(sum(values))
        # 如果 aggregate 为 False，返回格式化后的逗号分隔值的字符串
        return ", ".join(map(item_fn, values))

    if repr == "str":
        # 生成用于打印的行，每行包括函数名和格式化后的运行时间
        rows = [
            (k, fmt_fn(compilation_time_metrics[k], item_fn=lambda x: f"{x:.4f}"))
            for k in compilation_time_metrics
        ]
        # 生成包含标题和行的字符串表示
        out = "TorchDynamo compilation metrics:\n"
        out += tabulate(rows, headers=("Function", "Runtimes (s)"))
        return out
    elif repr == "csv":
        # 格式化所有值为字符串，保留六位小数
        values = [
            fmt_fn(v, item_fn=lambda x: f"{x:.6f}")
            for v in compilation_time_metrics.values()
        ]
        # 返回头部和格式化后的值列表
        headers = list(compilation_time_metrics.keys())
        return headers, values
# 在程序退出时注册的函数，用于记录编译时间信息到日志
@atexit.register
def dump_compile_times():
    # 调用 log 模块记录编译时间信息，使用 compile_times 函数生成的字符串表示，聚合结果为字符串
    log.info(compile_times(repr="str", aggregate=True))


# 定义了一个字典，将不同的 Tensor 类型映射到对应的 torch 数据类型元组
tensortype_to_dtype = {
    torch.FloatTensor: (torch.float32, torch.float),
    torch.DoubleTensor: (torch.float64, torch.double),
    torch.HalfTensor: (torch.float16, torch.half),
    torch.BFloat16Tensor: (torch.bfloat16,),
    torch.ByteTensor: (torch.uint8,),
    torch.CharTensor: (torch.int8,),
    torch.LongTensor: (torch.int64, torch.long),
    torch.IntTensor: (torch.int32, torch.int),
    torch.ShortTensor: (torch.int16, torch.short),
    torch.BoolTensor: (torch.bool,),
}


# 用于检查重复警告的类
class DuplicateWarningChecker:
    def __init__(self, maxsize=4096):
        self.maxsize = maxsize
        self.reset()

    def reset(self):
        # 使用有序字典作为存储容器，以记录警告的插入顺序
        self.set = collections.OrderedDict()

    def add(self, key):
        # 如果 key 已存在于 set 中，则将其移到有序字典的末尾
        if key in self.set:
            self.set.move_to_end(key, last=True)
            # 如果未开启详细模式，则直接返回 False
            if not config.verbose:
                return False
        else:
            # 否则将 key 添加到有序字典中，并在超出最大容量时删除最旧的记录
            self.set[key] = None
            while len(self.set) > self.maxsize:
                self.set.popitem(last=False)
        return True


# 全局变量，用于检查图中断重复警告
graph_break_dup_warning_checker = DuplicateWarningChecker()


# 设置编译调试环境的函数
def setup_compile_debug():
    # 检查环境变量 TORCH_COMPILE_DEBUG 是否为 '1'，表示开启编译调试模式
    compile_debug = os.environ.get("TORCH_COMPILE_DEBUG", "0") == "1"

    if compile_debug:
        # 如果开启了编译调试模式，则添加文件处理器并返回对应的 ExitStack 对象
        return add_file_handler()

    # 否则返回一个空的 ExitStack 对象
    return contextlib.ExitStack()


# 重置图中断重复警告检查器的状态
def reset_graph_break_dup_checker():
    graph_break_dup_warning_checker.reset()


# 添加文件处理器到日志记录器的函数
def add_file_handler():
    # 构建日志文件路径
    log_path = os.path.join(get_debug_dir(), "torchdynamo")
    os.makedirs(log_path, exist_ok=True)

    # 创建文件处理器并添加到对应的 logger 上
    log_file_handler = logging.FileHandler(os.path.join(log_path, "debug.log"))
    logger = logging.getLogger("torch._dynamo")
    logger.addHandler(log_file_handler)

    # 返回一个 ExitStack 对象，并注册回调函数，在退出时移除文件处理器
    exitstack = contextlib.ExitStack()
    exitstack.callback(lambda: logger.removeHandler(log_file_handler))
    return exitstack


# 设置日志文件的函数
def setup_log_file():
    exitstack = contextlib.ExitStack()
    if config.log_file_name is not None:
        # 如果配置了日志文件名，则为每个内部日志记录器添加文件处理器，并注册回调函数移除处理器
        log_file_handler = logging.FileHandler(config.log_file_name)
        for logger in torch._logging._internal.get_loggers():
            logger.addHandler(log_file_handler)
            exitstack.callback(lambda: logger.removeHandler(log_file_handler))
        return exitstack

    # 否则返回一个空的 ExitStack 对象
    return exitstack


# 生成记录文件名的函数，以异常对象和代码对象生成的信息作为一部分文件名
def gen_record_file_name(exc, code):
    return f"{get_debug_dir()}/error_recordings/{code.co_name}_{type(exc).__name__}_{code.co_firstlineno}.rec"


# 将执行记录写入文件的函数
def write_record_to_file(filename, exec_record):
    try:
        if os.path.exists(filename):
            # 如果文件已存在，则记录警告日志信息
            log.warning(
                "Unable to write execution record %s; file already exists.", filename
            )
        else:
            # 否则创建文件及其父目录，并将执行记录写入文件
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            with open(filename, "wb") as f:
                exec_record.dump(f)
    except Exception:
        # 捕获异常并记录错误日志信息
        log.exception("Unable to write execution record %s", filename)


# 统计调用次数的函数，接收一个 fx.Graph 对象作为参数
def count_calls(g: fx.Graph):
    c = 0
    # 遍历图中的节点列表 g.nodes
    for n in g.nodes:
        # 检查节点的操作类型是否包含字符串 "call"
        if "call" in n.op:
            # 如果满足条件，计数器 c 自增 1
            c += 1
    # 返回计数器 c 的值作为结果
    return c
def identity(x):
    return x


# 返回传入的参数本身，即返回值等于输入值
def identity(x):
    return x



def hashable(x):
    try:
        hash(x)
        return True
    except TypeError:
        return False
    # cannot hash writable memoryview object
    except ValueError:
        return False


# 检查对象是否可哈希（即是否能用作字典的键）
def hashable(x):
    try:
        hash(x)  # 尝试计算对象的哈希值
        return True  # 如果成功，表示对象可哈希
    except TypeError:
        return False  # 如果出现类型错误，表示对象不可哈希
    except ValueError:
        return False  # 如果出现值错误，特别说明不可哈希的情况（如可写的内存视图对象）



def nothing(*args, **kwargs):
    pass


# 什么也不做，即空函数
def nothing(*args, **kwargs):
    pass



class ExactWeakKeyDictionary:
    """Similar to weakref.WeakKeyDictionary, but use `is`/`id` rather than `==` to compare equality"""

    def __init__(self):
        self.values = dict()
        self.refs = dict()

    def __getitem__(self, key):
        return self.values[id(key)]

    def get(self, key, default=None):
        return self.values.get(id(key), default)

    def __contains__(self, key):
        return id(key) in self.values

    def __setitem__(self, key, value):
        idx = id(key)
        if idx not in self.refs:
            self.refs[idx] = weakref.ref(key, lambda ref: self._remove_id(idx))
        self.values[idx] = value

    def _remove_id(self, idx):
        if idx in self.values:
            del self.values[idx]
        if idx in self.refs:
            del self.refs[idx]

    def clear(self):
        self.refs.clear()
        self.values.clear()


# 精确弱引用键字典，使用`is`/`id`而非`==`来比较键的相等性
class ExactWeakKeyDictionary:
    """Similar to weakref.WeakKeyDictionary, but use `is`/`id` rather than `==` to compare equality"""

    def __init__(self):
        self.values = dict()  # 存储键值对的字典
        self.refs = dict()    # 存储弱引用的字典

    def __getitem__(self, key):
        return self.values[id(key)]  # 返回与键关联的值

    def get(self, key, default=None):
        return self.values.get(id(key), default)  # 获取键对应的值，如果不存在返回默认值

    def __contains__(self, key):
        return id(key) in self.values  # 检查键是否在字典中

    def __setitem__(self, key, value):
        idx = id(key)
        if idx not in self.refs:
            self.refs[idx] = weakref.ref(key, lambda ref: self._remove_id(idx))  # 创建键的弱引用
        self.values[idx] = value  # 设置键值对

    def _remove_id(self, idx):
        if idx in self.values:
            del self.values[idx]  # 删除键值对
        if idx in self.refs:
            del self.refs[idx]  # 删除弱引用

    def clear(self):
        self.refs.clear()  # 清空弱引用字典
        self.values.clear()  # 清空键值对字典



def istype(obj, allowed_types):
    """isinstance() without subclasses"""
    if isinstance(allowed_types, (tuple, list, set)):
        return type(obj) in allowed_types
    return type(obj) is allowed_types


# 检查对象是否为指定类型（不考虑子类）
def istype(obj, allowed_types):
    """isinstance() without subclasses"""
    if isinstance(allowed_types, (tuple, list, set)):
        return type(obj) in allowed_types  # 如果允许类型为元组、列表或集合，则检查是否在其中
    return type(obj) is allowed_types  # 否则直接检查对象类型是否为指定类型



if sys.version_info >= (3, 12):
    # Some typing classes moved to C in 3.12,
    # which no longer have the _Final mixin.
    _builtin_final_typing_classes = (
        typing.ParamSpecArgs,
        typing.ParamSpecKwargs,
        typing.ParamSpec,
        typing.TypeVar,
        typing.TypeVarTuple,
        typing.TypeAliasType,
    )


# 如果 Python 版本大于等于 3.12，则定义一些内置的不可变类型类
if sys.version_info >= (3, 12):
    # Some typing classes moved to C in 3.12,
    # which no longer have the _Final mixin.
    _builtin_final_typing_classes = (
        typing.ParamSpecArgs,
        typing.ParamSpecKwargs,
        typing.ParamSpec,
        typing.TypeVar,
        typing.TypeVarTuple,
        typing.TypeAliasType,
    )



def is_typing(value):
    # _Final catches most of typing classes:
    #   - Any
    #   - Callable
    #   - Union
    #   ...
    #
    # NB: we intentionally ignore classes that inherit from Generic, since they
    # can be used as both TypingVariable as well as UserDefinedClassVariable.
    if sys.version_info >= (3, 12) and isinstance(value, _builtin_final_typing_classes):
        return True
    return isinstance(value, typing._Final) or value is typing.Generic  # type: ignore[attr-defined]


# 检查值是否为 typing 模块定义的类型
def is_typing(value):
    # _Final catches most of typing classes:
    #   - Any
    #   - Callable
    #   - Union
    #   ...
    #
    # NB: we intentionally ignore classes that inherit from Generic, since they
    # can be used as both TypingVariable as well as UserDefinedClassVariable.
    if sys.version_info >= (3, 12) and isinstance(value, _builtin_final_typing_classes):
        return True  # 如果是内置的不可变类型类，则返回 True
    return isinstance(value, typing._Final) or value is typing.Generic  # type: ignore[attr-defined]，否则检查是否是 _Final 或 Generic 类型



def is_numpy_int_type(value):
    if not np:
        return False

    return istype(
        value,
        (
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        ),
    )


# 检查值是否为 NumPy 整数类型
def is_numpy_int_type(value):
    if not np:  # 如果没有导入 NumPy 模块，则直接返回 False
        return False

    return istype(
        value,
        (
            np.int8,
            np.int16,
            np.int32,
            np.int64,
            np.uint8,
            np.uint16,
            np.uint32,
            np.uint64,
        ),
    )  # 否则使用 istype 函数检查是否为这些 NumPy 整数类型之一



def is_numpy_float_type(value):
    if not np:
        return False

    return istype(
        value,
        (
            np.float16,
            np.float32,
            np.float64,
        ),
    )


# 检查值是否为 NumPy 浮点类型
def is_numpy_float_type(value):
    if not np:  # 如果没有导入 NumPy 模块，则直接返回 False
        return False

    return istype(
        value,
        (
            np.float16,
            np.float32,
            np.float64,
        ),
    )  # 否则使用 istype 函数检查是否为这些 NumPy 浮点类型之一



def is_lru_cache_wrapped_function(value):
    return isinstance(value, functools._lru_cache_wrapper) and is_function(
        inspect.getattr_static(value, "__wrapped__")
    )


# 检查值是否为 functools.lru_cache 装饰的函数
def is_lru_cache_wrapped_function(value):
    return isinstance(value, functools._lru_cache_wrapper) and is_function(
        inspect.getattr_static(value, "__wrapped__")
    )  # 判断是否为 lru_cache 包装的函数，并且检查其原始函数是否是函数



def is_function_or_wrapper(value):


# 检查值是否为函数或函数包装器
def
    # 检查给定的值是否是一个函数，或者是否是 torch 库中 OpOverloadPacket 或 OpOverload 类的实例之一
    return is_function(value) or isinstance(
        value, (torch._ops.OpOverloadPacket, torch._ops.OpOverload)
    )
# 检查给定值是否为函数类型或其派生类型，返回布尔值
def is_function(value):
    return isinstance(
        value,
        (
            types.FunctionType,                    # 普通函数类型
            types.BuiltinFunctionType,              # 内建函数类型
            types.MethodDescriptorType,             # 方法描述符类型
            types.WrapperDescriptorType,            # 包装器描述符类型
        ),
    )


# 如果函数是装饰器包装过的，则解包返回原始函数
def unwrap_if_wrapper(fn):
    return unwrap_with_attr_name_if_wrapper(fn)[0]


# 如果函数被装饰器包装过，则解包返回原始函数及装饰器名称
def unwrap_with_attr_name_if_wrapper(fn):
    # TODO(anijain2305) - Investigate if we can get rid of this function
    # unpack @torch._dynamo.optimize()(fn) wrapped function
    if is_function(fn) and inspect.getattr_static(fn, "_torchdynamo_inline", False):
        fn = inspect.getattr_static(fn, "_torchdynamo_inline", fn)  # 获取装饰器内部原始函数
        attr_name = "_torchdynamo_inline"  # 设置装饰器名称
    else:
        attr_name = None  # 没有装饰器则置空
    return fn, attr_name


# 检查给定值是否为 numpy 的 ndarray 类型
def is_numpy_ndarray(value):
    if not np:  # 如果 numpy 未导入，则返回 False
        return False
    return istype(value, np.ndarray)  # 调用 istype 函数检查是否为 ndarray 类型


# 检查给定对象是否为 torch 的 tensor 类型或其子类
def istensor(obj):
    """Check of obj is a tensor"""
    tensor_list = (
        torch.Tensor,                            # 标准张量类型
        torch.nn.Parameter,                      # 神经网络参数类型
        *config.traceable_tensor_subclasses,      # 配置中追踪的张量子类
    )
    tensor_list = tensor_list + (torch._subclasses.FakeTensor,)  # 加入虚拟张量类型
    return istype(obj, tensor_list)  # 调用 istype 函数检查是否为 tensor 类型或其子类


# 检查给定模块是否为 LazyModuleMixin 的实例
def is_lazy_module(mod):
    return isinstance(mod, LazyModuleMixin)


# 缓存大小为 4096 的 print_once 函数，用于仅打印一次的输出
@functools.lru_cache(4096)
def print_once(*args):
    print(*args)


# 创建一个闭包中常见的 cell 对象，用于存储值
def make_cell(val=None):
    """Some black magic to create a cell object that usually only exists in a closure"""
    x = val  # 将参数值赋给局部变量 x

    def f():
        return x  # 返回闭包中的 x 值

    assert f.__closure__ is not None and len(f.__closure__) == 1  # 断言确保 f 是闭包且仅有一个闭包变量
    return f.__closure__[0]  # 返回闭包中的第一个变量


# 将函数的参数和关键字参数转换为代理对象，用于某些未实现的情况下的处理
def proxy_args_kwargs(args, kwargs):
    try:
        proxy_args = tuple(arg.as_proxy() for arg in args)  # 将参数转换为代理对象
        proxy_kwargs = {key: arg.as_proxy() for key, arg in kwargs.items()}  # 将关键字参数转换为代理对象
        return proxy_args, proxy_kwargs  # 返回转换后的代理对象
    except NotImplementedError as e:
        from .exc import unimplemented  # 导入未实现异常处理模块
        from .variables.base import typestr  # 导入类型字符串处理模块

        unimplemented(
            f"call_function args: {typestr(*args)} {typestr(*list(kwargs.values()))}",  # 报告未实现的函数调用异常
            from_exc=e,
        )


# 用于保存编译相关的指标数据的数据类
@dataclasses.dataclass
class CompilationMetrics:
    compile_id: str
    frame_key: str
    co_name: str
    co_filename: str
    co_firstlineno: int
    cache_size: int
    accumulated_cache_size: int
    guard_count: Optional[int]
    shape_env_guard_count: Optional[int]
    graph_op_count: Optional[int]
    graph_node_count: Optional[int]
    graph_input_count: Optional[int]
    start_time: float
    entire_frame_compile_time_s: Optional[float]
    backend_compile_time_s: Optional[float]
    inductor_compile_time_s: Optional[float]
    code_gen_time_s: Optional[float]
    fail_type: Optional[str]
    fail_reason: Optional[str]
    fail_user_frame_filename: Optional[str]
    fail_user_frame_lineno: Optional[int]
    non_compliant_ops: Set[str]
    compliant_custom_ops: Set[str]
    restart_reasons: Set[str]
    dynamo_time_before_restart_s: float
    # Sometimes, we will finish analyzing a frame but conclude we don't want
    # 定义一个布尔变量 has_guarded_code，用于标识是否有受保护的代码安装
    has_guarded_code: bool
@dataclasses.dataclass
class BwdCompilationMetrics:
    """Dataclass defining backward compilation metrics."""

    compile_id: str  # Identifier for the compilation
    inductor_compile_time_s: Optional[float]  # Optional time for inductor compilation in seconds
    code_gen_time_s: Optional[float]  # Optional time for code generation in seconds
    fail_type: Optional[str]  # Optional type of failure
    fail_reason: Optional[str]  # Optional reason for failure


DEFAULT_COMPILATION_METRICS_LIMIT = 64  # Default limit for compilation metrics storage


_compilation_metrics: Deque[
    Union[CompilationMetrics, BwdCompilationMetrics]
] = collections.deque(maxlen=DEFAULT_COMPILATION_METRICS_LIMIT)  # Deque to store compilation metrics with default limit


def record_compilation_metrics(
    compilation_metrics: Union[CompilationMetrics, BwdCompilationMetrics]
):
    """Record compilation metrics and log if configured."""
    global _compilation_metrics
    _compilation_metrics.append(compilation_metrics)  # Append given compilation metrics to deque
    if isinstance(compilation_metrics, CompilationMetrics):
        name = "compilation_metrics"
    else:
        name = "bwd_compilation_metrics"
    torch._logging.trace_structured(
        name,
        lambda: {
            k: list(v) if isinstance(v, set) else v
            for k, v in dataclasses.asdict(compilation_metrics).items()
        },
    )  # Trace structured logging of compilation metrics
    if config.log_compilation_metrics:
        log_compilation_event(compilation_metrics)  # Log compilation event if configured


def set_compilation_metrics_limit(new_size: int) -> None:
    """Set new size limit for compilation metrics storage."""
    global _compilation_metrics
    while len(_compilation_metrics) > new_size:
        _compilation_metrics.popleft()  # Remove elements from left until size limit is met
    new_deque = collections.deque(_compilation_metrics, maxlen=new_size)
    _compilation_metrics = new_deque  # Update deque with new size limit


def clear_compilation_metrics() -> None:
    """Clear all compilation metrics."""
    global _compilation_metrics
    _compilation_metrics.clear()  # Clear all items from compilation metrics deque


def get_compilation_metrics() -> List[Union[CompilationMetrics, BwdCompilationMetrics]]:
    """Retrieve a list of all compilation metrics stored."""
    return list(_compilation_metrics)  # Return list of compilation metrics in deque


@dataclasses.dataclass
class CleanupHook:
    """Remove a global variable when hook is called"""

    scope: Dict[str, Any]  # Scope dictionary where variable resides
    name: str  # Name of the variable to remove

    def __call__(self, *args):
        """Callable method to execute cleanup."""
        # Make sure we're not shutting down
        if CleanupManager is not None:
            CleanupManager.count -= 1  # Decrease count in CleanupManager if not shutting down
        del self.scope[self.name]  # Delete variable from scope

    @staticmethod
    def create(scope, name, val):
        """Static method to create a CleanupHook."""
        assert name not in scope
        CleanupManager.count += 1  # Increase count in CleanupManager
        scope[name] = val  # Assign value to scope
        return CleanupHook(scope, name)  # Return new CleanupHook instance


class CleanupManager(ExactWeakKeyDictionary):
    """Manager for cleanup hooks"""

    count = 0  # Class-level count of active cleanup hooks
    instance: ClassVar["CleanupManager"]  # Class variable for singleton instance

    def _remove_id(self, idx):
        """Remove hooks associated with the given index."""
        for hook in self.values[idx]:
            hook()  # Call each hook's cleanup method
        super()._remove_id(idx)  # Remove id from ExactWeakKeyDictionary


CleanupManager.instance = CleanupManager()  # Initialize singleton instance of CleanupManager


def clone_tensor(x):
    """Clone the tensor and its gradient."""
    y = x.clone().requires_grad_(x.requires_grad)  # Clone tensor and set requires_grad flag
    if x.is_leaf and x.grad is not None:
        y.grad = x.grad.clone()  # Clone gradient if original tensor is a leaf with a non-None gradient
    return y  # Return cloned tensor


def clone_input(x, *, dtype=None):
    """Copy while preserving strides."""
    # TODO: this is questionable
    if is_fake(x):
        # This func fails on fake tensors in __torch_dispatch__
        return x  # Return input tensor if it's identified as fake
    def torch_clone(x):
        # 使用 torch.clone() 方法创建张量 x 的副本 y
        y = torch.clone(x)
        # 如果 x 是叶子节点，则将 y 的梯度需求与 x 保持一致
        if x.is_leaf:
            y.requires_grad_(x.requires_grad)
        # 如果 x 是叶子节点且具有梯度，则克隆其梯度并赋给 y
        if x.is_leaf and x.grad is not None:
            y.grad = clone_input(x.grad, dtype=dtype)
        # 如果 x 具有动态索引属性，则复制这些索引到 y
        if hasattr(x, "_dynamo_dynamic_indices"):
            y._dynamo_dynamic_indices = x._dynamo_dynamic_indices.copy()  # type: ignore[attr-defined]
        return y

    with torch.no_grad():
        # 如果张量 x 的设备类型为 "xla"，则避免访问其 data_ptr()，以防止崩溃
        if x.device.type == "xla":
            # Access data_ptr() for a xla tensor will cause crash
            return torch_clone(x)

        # 处理稀疏存储（无步幅）的情况
        if x.layout is torch.sparse_coo:
            # 返回一个稀疏的 COO 张量，使用克隆的索引和值
            return torch.sparse_coo_tensor(
                torch_clone(x._indices()),
                torch_clone(x._values()),
                x.shape,
                is_coalesced=x.is_coalesced(),
            )
        elif is_sparse_compressed(x):
            # 如果是压缩稀疏张量，则根据布局不同选择不同的索引
            if x.layout in {torch.sparse_csr, torch.sparse_bsr}:
                compressed_indices = x.crow_indices()
                plain_indices = x.col_indices()
            else:
                compressed_indices = x.ccol_indices()
                plain_indices = x.row_indices()
            # 返回一个压缩稀疏张量，使用克隆的压缩索引、普通索引和值
            return torch.sparse_compressed_tensor(
                torch_clone(compressed_indices),
                torch_clone(plain_indices),
                torch_clone(x.values()),
                x.shape,
                layout=x.layout,
            )

        # 计算所需的结果张量大小
        needed_size = sum(
            (shape - 1) * stride for shape, stride in zip(x.size(), x.stride())
        )
        # 如果 x 是量化的张量，则创建一个量化的空张量
        if x.is_quantized:
            result = torch.empty_quantized((needed_size + 32,), x)
        else:
            # 否则创建一个普通的空张量
            result = torch.empty(
                needed_size + 32, dtype=dtype or x.dtype, device=x.device
            )
        # 计算缓存行偏移量，用于 result 的 as_strided_ 操作
        cache_line_offset = (
            (x.data_ptr() - result.data_ptr()) % 32
        ) // x.element_size()
        result.as_strided_(x.size(), x.stride(), cache_line_offset)
        try:
            # 尝试将 x 的克隆复制到 result
            result.copy_(x.clone())
            # 如果 x 是叶子节点，则将 result 的梯度需求与 x 保持一致
            if x.is_leaf:
                result.requires_grad_(x.requires_grad)
            # 如果 x 是叶子节点且具有梯度，则克隆其梯度并赋给 result
            if x.is_leaf and x.grad is not None:
                result.grad = clone_input(x.grad, dtype=dtype)
        except RuntimeError:
            # 捕获 RuntimeError 异常，提示需要先克隆张量再执行操作
            # RuntimeError: unsupported operation: more than one element of the written-to
            # tensor refers to a single memory location. Please clone() the tensor before
            # performing the operation.
            return torch_clone(x)
        # 如果 x 具有动态索引属性，则复制这些索引到 result
        if hasattr(x, "_dynamo_dynamic_indices"):
            result._dynamo_dynamic_indices = x._dynamo_dynamic_indices.copy()  # type: ignore[attr-defined]
        return result
# 克隆输入数据结构及其内容，支持字典和列表两种类型的输入
def clone_inputs(example_inputs):
    # 声明结果变量，可以是字典或列表
    res: Union[Dict[Any, Any], List[Any]]
    # 如果输入是字典类型
    if type(example_inputs) is dict:
        # 深拷贝字典
        res = dict(example_inputs)
        # 遍历字典中的每个键值对
        for key, value in res.items():
            # 如果值是元组，则递归调用 clone_inputs 函数
            if isinstance(value, tuple):
                res[key] = clone_inputs(value)
            else:
                # 否则断言值是 torch.Tensor 类型，并进行克隆处理
                assert isinstance(value, torch.Tensor), type(value)
                res[key] = clone_input(value)
        return res

    # 如果输入是列表类型
    res = list(example_inputs)
    # 遍历列表的每个元素
    for i in range(len(res)):
        # 如果元素是 torch.Tensor 类型，则进行克隆处理
        if isinstance(res[i], torch.Tensor):
            res[i] = clone_input(res[i])
    return res


# 检查是否处于 funtorch 模式，如果是则跳过当前帧
def skip_frame_if_in_functorch_mode(val: torch.Tensor):
    try:
        val.data_ptr()  # 尝试获取数据指针，对于 functorch 张量会抛出异常
    except RuntimeError as e:
        from .exc import SkipFrame

        # 获取 functorch 子类的名称
        functorch_subclass_name = re.sub(r"\(.*", "", repr(val))
        # 抛出 SkipFrame 异常，指示不能在当前上下文中运行 torch.compile
        raise SkipFrame(
            f"torch.compile cannot be run in context: {functorch_subclass_name}"
        ) from e


# 保存当前随机数生成器状态的上下文管理器
@contextmanager
def preserve_rng_state():
    # 引入需要禁用的 functorch 和当前模式禁用函数
    disable_functorch = torch._C._DisableFuncTorch
    disable_current_modes = torch.utils._python_dispatch._disable_current_modes
    # 使用上下文管理器禁用当前模式和 functorch
    with disable_current_modes(), disable_functorch():
        # 复制当前随机数生成器状态
        rng_state = torch.clone(torch.random.get_rng_state())
        # 检查并跳过 functorch 模式
        skip_frame_if_in_functorch_mode(rng_state)
        # 如果 CUDA 可用，也复制 CUDA 随机数生成器状态
        if torch.cuda.is_available():
            cuda_rng_state = torch.clone(torch.cuda.get_rng_state())
    try:
        yield
    finally:
        # 退出上下文管理器后，恢复随机数生成器状态
        with torch.utils._python_dispatch._disable_current_modes():
            torch.random.set_rng_state(rng_state)
            if torch.cuda.is_available():
                torch.cuda.set_rng_state(cuda_rng_state)  # type: ignore[possibly-undefined]


# 检查对象是否是 JIT 模型
def is_jit_model(model0):
    return isinstance(
        model0,
        (
            torch.jit._trace.TopLevelTracedModule,
            torch.jit._script.RecursiveScriptModule,
            torch.jit.ScriptFunction,
            torch.jit.ScriptModule,
        ),
    )


# 将模型转换为 TorchScript 格式
def torchscript(model, example_inputs, verbose=False):
    # 如果模型已经是 JIT 模型，则直接返回
    if is_jit_model(model):
        return model

    try:
        # 尝试对模型进行跟踪
        return torch.jit.trace(model, example_inputs)
    except Exception:
        try:
            # 如果跟踪失败，则尝试脚本化模型
            return torch.jit.script(model)
        except Exception:
            # 如果两者都失败，则根据 verbose 参数记录日志或错误信息
            if verbose:
                log.exception("jit error")
            else:
                log.error("Both torch.jit.trace and torch.jit.script failed")
    return None


# 获取对象的源文件路径
def getfile(obj):
    try:
        return inspect.getfile(obj)
    except (TypeError, OSError):
        return None


# 检查对象是否是命名元组或 torch.return_types.* 类似命名元组
def is_namedtuple(obj):
    """Test if an object is a namedtuple or a torch.return_types.* quasi-namedtuple"""
    return is_namedtuple_cls(type(obj))


# 检查类是否是命名元组或 torch.return_types|torch.autograd.forward_ad.* 类似命名元组
def is_namedtuple_cls(cls):
    """Test if an object is a namedtuple or a (torch.return_types|torch.autograd.forward_ad).* quasi-namedtuple"""
    # 尝试检查 cls 是否是 tuple 的子类
    try:
        # 如果 cls 是 tuple 的子类
        if issubclass(cls, tuple):
            # 获取 cls 的基类列表，如果没有则默认为 [None]
            bases = getattr(cls, "__bases__", []) or [None]
            # 获取 cls 的模块名
            module = getattr(cls, "__module__", None)
            # 返回是否满足以下条件：
            # - 模块名是 "torch.return_types" 或 "torch.autograd.forward_ad"
            # - 或者基类的第一个元素是 tuple，并且 cls 具有 "_make" 和 "_fields" 属性
            return module in ("torch.return_types", "torch.autograd.forward_ad") or (
                bases[0] is tuple and hasattr(cls, "_make") and hasattr(cls, "_fields")
            )
    except TypeError:
        pass  # 忽略类型错误异常
    
    # 如果出现任何异常，返回 False
    return False
# 使用 functools.lru_cache 装饰器，缓存函数结果以提高性能
@functools.lru_cache(1)
# 获取命名元组或 torch.return_types.* 类似命名元组的字段信息
def namedtuple_fields(cls):
    """Get the fields of a namedtuple or a torch.return_types.* quasi-namedtuple"""
    # 如果 cls 是 slice 类型，则返回固定的字段列表 ["start", "stop", "step"]
    if cls is slice:
        return ["start", "stop", "step"]

    # 断言 cls 是 tuple 的子类
    assert issubclass(cls, tuple)
    # 如果 cls 具有 "_fields" 属性，则为普通的命名元组，返回其字段
    if hasattr(cls, "_fields"):
        return cls._fields

    # 对于 torch.return_types 模块中的特殊类型，创建一个临时类 Marker
    @dataclasses.dataclass
    class Marker:
        index: int

    # 断言 cls 是 torch.return_types 模块下的类
    assert cls.__module__ == "torch.return_types"
    # 使用 Marker 类的映射创建一个包含字段信息的列表 fields
    obj = cls(map(Marker, range(cls.n_fields)))
    fields: List[Optional[str]] = [None] * cls.n_fields
    # 遍历 obj 的属性，将有效字段名填充到 fields 中
    for name in dir(obj):
        if name[0] != "_" and isinstance(getattr(obj, name), Marker):
            fields[getattr(obj, name).index] = name
    return fields


# 获取模型的参数和缓冲区的检查点
def checkpoint_params(gm):
    # 在没有梯度更新的情况下进行操作
    with torch.no_grad():
        # 复制当前随机数生成器状态
        rng_state = torch.clone(torch.random.get_rng_state())
        # 如果 CUDA 可用，复制 CUDA 随机数生成器状态
        if torch.cuda.is_available():
            cuda_rng_state = torch.clone(torch.cuda.get_rng_state())
        saved_state = []
        # 遍历模型的参数和缓冲区，保存它们的当前版本和值
        for param in itertools.chain(gm.parameters(), gm.buffers()):
            saved_state.append((param, param._version, torch.clone(param)))

    # 定义一个 restore 函数，用于恢复状态
    def restore():
        with torch.no_grad():
            # 恢复随机数生成器状态
            torch.random.set_rng_state(rng_state)
            # 如果 CUDA 可用，恢复 CUDA 随机数生成器状态
            if torch.cuda.is_available():
                torch.cuda.set_rng_state(cuda_rng_state)
            # 遍历保存的状态，如果参数的版本不匹配，则复制原始值回参数
            for param, version, original_value in saved_state:
                if param._version != version:
                    param.copy_(original_value)

    return restore


# 对模型进行定时评估
def timed(model, example_inputs, times=1):
    # 如果 CUDA 可用，则同步 CUDA 设备
    if torch.cuda.is_available():
        synchronize = torch.cuda.synchronize
    else:
        # 否则使用一个空函数 nothing
        synchronize = nothing

    # 执行垃圾回收和随机种子设定
    synchronize()
    gc.collect()
    torch.manual_seed(1337)
    t0 = time.perf_counter()
    # 多次运行模型以测量平均时间
    for _ in range(times):
        result = model(*example_inputs)
        synchronize()
    t1 = time.perf_counter()
    return result, t1 - t0  # 返回评估结果和总时间，类型可能未定义


# 检查模型和示例输入是否在 CUDA 上运行
def check_is_cuda(gm, example_inputs):
    return all(x.is_cuda for x in itertools.chain(example_inputs, gm.parameters(True)))


# 获取 rot_n_helper 函数的辅助函数，用于创建特定旋转参数个数 n 的函数
@lru_cache(32)
def rot_n_helper(n):
    # 断言 n 大于 1
    assert n > 1
    # 生成变量列表 vars，如 ["v0", "v1", ...]
    vars = [f"v{i}" for i in range(n)]
    # 生成旋转后的变量顺序 reversed(vars[-1:] + vars[:-1])
    rotated = reversed(vars[-1:] + vars[:-1])
    # 使用 eval 动态创建一个 lambda 函数 fn，用于将变量旋转
    fn = eval(f"lambda {','.join(vars)}: ({','.join(rotated)})")
    fn.__name__ = f"rot_{n}_helper"  # 设置函数名
    return fn


# 常见的常量类型集合
common_constant_types = {
    int,
    float,
    complex,
    bool,
    str,
    bytes,
    type(None),
    Ellipsis.__class__,
    types.CodeType,
    torch.device,
    torch.dtype,
    torch.memory_format,
    torch.layout,
}

# 如果安装了 Triton 包，将 Triton 的 dtype 添加到常量类型集合中
if has_triton_package():
    import triton

    common_constant_types.add(triton.language.dtype)


# 判断一个值是否为安全的常量
def is_safe_constant(v):
    # 如果 v 是 tuple 或 frozenset 类型，则递归检查其每个元素是否为安全常量
    if istype(v, (tuple, frozenset)):
        return all(map(is_safe_constant, v))
    # 否则检查 v 是否是枚举类型或在 common_constant_types 中定义的类型之一
    return isinstance(v, (enum.Enum, type)) or istype(
        v,
        common_constant_types | {slice},
    )


# 特化符号节点的函数
def specialize_symnode(arg):
    # 导入常量变量和符号节点变量
    from .variables import ConstantVariable, SymNodeVariable
    # 检查参数是否是 SymNodeVariable 类的实例，如果是则执行特定操作
    if isinstance(arg, SymNodeVariable):
        # 调用 ConstantVariable 类的静态方法 create，传入 arg.evaluate_expr() 的结果作为参数，并返回其结果
        return ConstantVariable.create(arg.evaluate_expr())

    # 如果参数不是 SymNodeVariable 类的实例，则直接返回参数本身
    return arg
# 如果参数是动态特化的符号节点，则对其进行特化处理
def guard_if_dyn(arg):
    from .variables import ConstantVariable

    arg = specialize_symnode(arg)  # 对参数进行符号节点特化处理

    if isinstance(arg, ConstantVariable):  # 如果参数是常量变量，则返回其对应的 Python 常量值
        return arg.as_python_constant()

    return arg  # 否则直接返回参数本身


# 检查所有参数和关键字参数是否都是 Python 常量
def check_constant_args(args, kwargs):
    return all(x.is_python_constant() for x in itertools.chain(args, kwargs.values()))


# 检查所有参数和关键字参数是否至少有一个是未特化的 Python 变量或者常量
def check_unspec_python_args(args, kwargs):
    from .variables.constant import ConstantVariable
    from .variables.tensor import UnspecializedPythonVariable

    unspec_count = 0
    for x in itertools.chain(args, kwargs.values()):
        if isinstance(x, UnspecializedPythonVariable):  # 如果是未特化的 Python 变量
            unspec_count += 1
        elif not isinstance(x, ConstantVariable):  # 如果不是常量变量，则返回 False
            return False
    return unspec_count > 0  # 返回是否存在未特化的 Python 变量


# 检查所有参数和关键字参数是否至少有一个是未特化的 Python 变量或者所有参数都是常量
def check_unspec_or_constant_args(args, kwargs):
    # 这是 check_constant_args 和 check_unspec_python_args 的组合版本
    from .variables.tensor import UnspecializedPythonVariable

    for x in itertools.chain(args, kwargs.values()):
        if not (x.is_python_constant() or isinstance(x, UnspecializedPythonVariable)):
            return False
    return True


# 检查所有参数和关键字参数是否至少有一个是 numpy ndarray 变量
def check_numpy_ndarray_args(args, kwargs):
    from .variables.tensor import NumpyNdarrayVariable

    return any(
        isinstance(x, NumpyNdarrayVariable)
        for x in itertools.chain(args, kwargs.values())
    )


# 定义不同类型的视图对象的类型
dict_keys: Type[KeysView[Any]] = type(dict().keys())
dict_values: Type[ValuesView[Any]] = type(dict().values())
odict_values: Type[ValuesView[Any]] = type(collections.OrderedDict().values())
tuple_iterator: Type[Iterator[Any]] = type(iter(tuple()))
tuple_iterator_len = tuple_iterator.__length_hint__  # 获取 tuple 迭代器的长度提示
object_new = object.__new__  # 获取 object 类的 __new__ 方法


# 创建一个新的 nn.Module 实例，并初始化它
def nn_module_new(cls):
    obj = object_new(cls)
    torch.nn.Module.__init__(obj)
    return obj


# 计算迭代器中所有元素的乘积
def product(it):
    return functools.reduce(operator.mul, it, 1)


# 从 tuple 迭代器中获取指定索引的元素
def tuple_iterator_getitem(it, index):
    _, (obj,), start = it.__reduce__()
    return obj[start + index]


# 将类型 t 转换为其子类 cls 的实例
def to_subclass(t, cls):
    return t.as_subclass(cls)


# 获取字典中指定索引位置的键
def dict_keys_getitem(d, n):
    return next(itertools.islice(iter(d), n, n + 1))


# 返回枚举值的表示形式，考虑是否是本地枚举
def enum_repr(value, local):
    # 枚举类可以重写 __str__ 方法。使用 __class__ 和 name 属性来提取类名和键名。
    name = value.__class__.__name__
    val = value.name
    scope = "L" if local else "G"
    local_name = f'{scope}["{name}"].{val}'
    return local_name


# 设置示例值给定节点，尽管称为 example_value，实际上是一种虚拟张量
# 这些示例值充当 Dynamo 追踪的运行时状态，如果发生元数据变异，则 example_value 直接更新
# 因此不能依赖它准确反映程序追踪时值的状态。
def set_example_value(node, example_value):
    pass  # 仅说明功能，无需实际操作
    # 将 example_value 存储到 node.meta 字典中的 "example_value" 键下
    node.meta["example_value"] = example_value

    # 从 TracingContext 中获取当前的 fake_mode 对象的 shape_env 属性
    shape_env = TracingContext.get().fake_mode.shape_env

    # 使用 torch.fx.experimental.symbolic_shapes.compute_unbacked_bindings 函数计算未支持的绑定
    # 如果计算结果非空（即 symbol_to_path 有值），则执行以下代码块
    if symbol_to_path := torch.fx.experimental.symbolic_shapes.compute_unbacked_bindings(
        shape_env, example_value
    ):
        # 将计算得到的未支持绑定存储到 node.meta 字典中的 "unbacked_bindings" 键下
        node.meta["unbacked_bindings"] = symbol_to_path
# 检索虚假张量对象的示例值
def _get_fake_tensor(vt):
    fake_tensor = vt.as_proxy().node.meta.get("example_value")
    # 如果没有虚假值，则引发未实现异常
    if not is_fake(fake_tensor):
        from .exc import unimplemented

        unimplemented("Cannot check Tensor object identity without its fake value")
    return fake_tensor


# 迭代检查列表中是否包含指定对象
def iter_contains(items, search, tx, check_tensor_identity=False):
    from .variables import (
        BuiltinVariable,
        ConstantVariable,
        TensorVariable,
        VariableTracker,
    )

    # 如果搜索对象是 Python 常量，则检查是否存在该常量
    if search.is_python_constant():
        found_const = any(
            x.is_python_constant()
            and x.as_python_constant() == search.as_python_constant()
            for x in items
        )
        return ConstantVariable.create(found_const)

    must_check_tensor_id = False
    # 如果需要检查张量对象的身份并且搜索对象是 TensorVariable 类型，则进行检查
    if check_tensor_identity and isinstance(search, TensorVariable):
        must_check_tensor_id = True
        # 通过 _get_fake_tensor 获取虚假张量对象以进行比较
        # 匹配 Tensor 意味着匹配 FakeTensor
        search = _get_fake_tensor(search)

    found: Optional[VariableTracker] = None
    for x in items:
        if must_check_tensor_id:
            # 如果需要检查张量身份并且 x 是 TensorVariable 类型，则进行比较
            if isinstance(x, TensorVariable):
                if search is _get_fake_tensor(x):  # 对象等价性比较
                    return ConstantVariable.create(True)
        else:
            # 否则，调用内建变量的操作符 eq 来检查两个对象是否相等
            check = BuiltinVariable(operator.eq).call_function(tx, [x, search], {})
            if found is None:
                found = check
            else:
                found = BuiltinVariable(operator.or_).call_function(
                    tx, [check, found], {}
                )
    # 如果未找到匹配项，则返回 False
    if found is None:
        found = ConstantVariable.create(False)
    return found


# 判断键是否为 id 类型，用于索引字典
def key_is_id(k):
    """Returns whether it indexes dictionaries using its id"""
    return isinstance(k, (torch.Tensor, torch.nn.Module, MethodWrapperType))


# 将字典的键转换为 id 形式的列表
def key_to_id(value):
    return [id(k) if key_is_id(k) else k for k in value.keys()]


# 返回常量的字符串表示形式，支持局部环境
def const_repr(x, *, local) -> str:
    from .trace_rules import is_builtin_callable

    # 如果 x 是列表或元组，则递归调用 const_repr 处理每个元素的表示形式
    if isinstance(x, (list, tuple)):
        elems_repr = ",".join(const_repr(s, local=local) for s in x)
        if isinstance(x, list):
            return f"[{elems_repr}]"
        else:
            assert isinstance(x, tuple)
            if len(x) == 1:
                return f"({elems_repr},)"
            else:
                return f"({elems_repr})"
    # 如果 x 是枚举类型，则调用 enum_repr 函数去除枚举的引号，以渲染在守卫代码中
    elif isinstance(x, enum.Enum):
        # 通过调用 enum_repr 并移除引号来解决在 Python 3.11 之前返回无效全局引用的问题
        return enum_repr(x, local=local).replace("'", "")
    # 如果 x 是内建可调用对象，则返回其名称
    elif is_builtin_callable(x):
        return x.__name__
    # 如果 x 是类型对象，则返回其全名字符串
    elif isinstance(x, type):

        def fullname(o):
            klass = o.__class__
            module = klass.__module__
            if module == "builtins":
                return klass.__qualname__  # 避免像 'builtins.str' 这样的输出
            return module + "." + klass.__qualname__

        return fullname(x)
    else:
        # 如果不满足之前的任何条件，返回变量 x 的表达式表示（使用 repr 函数）
        return f"{x!r}"
def dict_keys_repr(const_keys, *, local) -> str:
    # 使用 const_repr 函数对 const_keys 中的每个元素进行处理，并用逗号连接成字符串
    keys_str = ",".join(const_repr(s, local=local) for s in const_keys)
    # 返回格式化后的字符串，包含在方括号内
    return "[" + keys_str + "]"


GLOBAL_KEY_PREFIX = "__dict_key"


from torch._subclasses import UnsupportedFakeTensorException  # noqa: F401


def wrap_fake_exception(fn):
    try:
        # 尝试执行给定的函数 fn
        return fn()
    except UnsupportedFakeTensorException as e:
        # 当捕获到 UnsupportedFakeTensorException 异常时
        from .exc import unimplemented
        # 构建错误消息
        msg = f"Unsupported: {e.reason} with fake tensor propagation."
        # 记录警告日志
        log.warning(msg)
        # 调用 unimplemented 函数，抛出异常并传递原始异常对象 e
        unimplemented(msg, from_exc=e)


def deepcopy_to_fake_tensor(obj, fake_mode):
    # 利用 torch._subclasses.fake_tensor.FakeCopyMode 设置深拷贝模式
    with torch._subclasses.fake_tensor.FakeCopyMode(fake_mode):
        # 使用 wrap_fake_exception 函数对深拷贝操作进行封装
        return wrap_fake_exception(lambda: copy.deepcopy(obj))


def rmse(ref, res):
    """
    Calculate root mean squared error
    """
    # 计算均方根误差（RMSE）
    return torch.sqrt(torch.mean(torch.square(ref - res)))


def same(
    ref,
    res,
    fp64_ref=None,
    cos_similarity=False,
    tol=1e-4,
    equal_nan=False,
    exact_dtype=True,
    relax_numpy_equality=False,
    ignore_non_fp=False,
    log_error=log.error,
):
    """Check correctness to see if ref and res match"""
    if fp64_ref is None:
        fp64_ref = ref
    if isinstance(ref, (list, tuple, torch.nn.ParameterList, torch.Size)):
        # 如果 ref 是列表、元组或者 torch 的特定类型，验证与 res 的匹配性
        assert isinstance(res, (list, tuple)), f"type mismatch {type(ref)} {type(res)}"
        if len(ref) != len(res):
            # 如果长度不匹配，记录错误日志并返回 False
            log_error("Length mismatch")
            return False
        # 递归比较每个元素的匹配性
        return len(ref) == len(res) and all(
            same(
                ai,
                bi,
                fp64_refi,
                cos_similarity,
                tol,
                equal_nan,
                exact_dtype,
                relax_numpy_equality,
                ignore_non_fp,
                log_error=log_error,
            )
            for ai, bi, fp64_refi in zip(ref, res, fp64_ref)
        )
    elif type(ref).__name__ == "QuestionAnsweringModelOutput":
        # 对于类型为 "QuestionAnsweringModelOutput" 的对象，跳过 start_logits/end_logits 的精度检查
        return same(
            ref.loss,
            res.loss,
            fp64_ref.loss,
            cos_similarity,
            tol,
            equal_nan,
            exact_dtype,
            relax_numpy_equality,
            ignore_non_fp,
            log_error=log_error,
        )
    # 如果参考对象是一个字典
    elif isinstance(ref, dict):
        # 确保结果对象也是一个字典
        assert isinstance(res, dict)
        # 确保参考对象和结果对象的键集合完全一致，否则抛出异常
        assert set(ref.keys()) == set(res.keys()), f"keys mismatch {set(ref.keys())} == {set(res.keys())}"
        
        # 对参考对象的每个键按照字典序进行排序
        for k in sorted(ref.keys()):
            # 如果参考对象的值与结果对象的值不同
            if not (
                same(
                    ref[k],
                    res[k],
                    fp64_ref[k],
                    cos_similarity=cos_similarity,
                    tol=tol,
                    equal_nan=equal_nan,
                    exact_dtype=exact_dtype,
                    relax_numpy_equality=relax_numpy_equality,
                    ignore_non_fp=ignore_non_fp,
                    log_error=log_error,
                )
            ):
                # 记录错误日志，指出精度检验失败的键名
                log_error("Accuracy failed for key name %s", k)
                return False
        # 若所有键值对比较都通过，则返回True
        return True
    
    # 如果参考对象是字符串、整数、None、布尔值或torch设备对象之一
    elif isinstance(ref, (str, int, type(None), bool, torch.device)):
        # 如果忽略非浮点数的比较要求，则直接返回True
        if ignore_non_fp:
            return True
        # 否则比较参考对象和结果对象是否相等
        r = ref == res
        # 如果比较失败，记录错误日志
        if not r:
            log_error("Accuracy failed (%s): %s != %s", type(ref), ref, res)
        return r
    
    # 如果参考对象是numpy的整数类型或浮点数类型之一
    elif is_numpy_int_type(ref) or is_numpy_float_type(ref):
        # 如果放宽numpy对象的精度比较要求，并且结果对象不是numpy整数或浮点数类型
        if relax_numpy_equality and not (
            is_numpy_int_type(res) or is_numpy_float_type(res)
        ):
            # 则将参考对象转换为其对应的Python数值类型
            ref = ref.item()
        # 比较参考对象和结果对象是否相等
        r = (type(ref) is type(res)) and (ref == res)
        # 如果比较失败，记录错误日志
        if not r:
            log_error("Accuracy failed (numpy): %s != %s", ref, res)
        return r
    
    # 如果参考对象是numpy数组
    elif is_numpy_ndarray(ref):
        # 比较参考对象和结果对象是否相等，转换为torch张量进行比较
        return (type(ref) is type(res)) and same(
            torch.as_tensor(ref),
            torch.as_tensor(res),
            fp64_ref,
            cos_similarity=cos_similarity,
            tol=tol,
            equal_nan=equal_nan,
            exact_dtype=exact_dtype,
            relax_numpy_equality=relax_numpy_equality,
            ignore_non_fp=ignore_non_fp,
            log_error=log_error,
        )
    
    # 如果参考对象的类型名称在特定的一组类名中
    elif type(ref).__name__ in (
        "MaskedLMOutput",
        "Seq2SeqLMOutput",
        "CausalLMOutputWithCrossAttentions",
        "LongformerMaskedLMOutput",
        "Instances",
        "SquashedNormal",
        "Boxes",
        "Normal",
        "TanhTransform",
        "Foo",
        "Variable",
    ):
        # 确保参考对象和结果对象的类型完全一致
        assert type(ref) is type(res)
        # 检查所有与参考对象关联的属性是否满足精度要求
        return all(
            same(
                getattr(ref, key),
                getattr(res, key),
                getattr(fp64_ref, key),
                cos_similarity=cos_similarity,
                tol=tol,
                equal_nan=equal_nan,
                exact_dtype=exact_dtype,
                relax_numpy_equality=relax_numpy_equality,
                ignore_non_fp=ignore_non_fp,
                log_error=log_error,
            )
            for key in ref.__dict__.keys()
        )
    
    # 如果参考对象的类型不在预期的任何类型中，则抛出运行时错误
    else:
        raise RuntimeError(f"unsupported type: {type(ref).__name__}")
# 从代码对象中提取短文件名，用于格式化函数信息的显示
def format_func_info(code):
    short_filename = code.co_filename.split("/")[-1]
    return f"'{code.co_name}' ({short_filename}:{code.co_firstlineno})"


# 上下文管理器：临时禁用缓存限制
@contextlib.contextmanager
def disable_cache_limit():
    # 保存当前的缓存大小限制值，并设置为系统最大值
    prior = config.cache_size_limit
    config.cache_size_limit = sys.maxsize
    # 保存当前的累积缓存大小限制值，并设置为系统最大值
    prior_acc_limit = config.accumulated_cache_size_limit
    config.accumulated_cache_size_limit = sys.maxsize

    try:
        yield  # 执行上下文管理器中的代码块
    finally:
        # 恢复之前保存的缓存大小限制值
        config.cache_size_limit = prior
        # 恢复之前保存的累积缓存大小限制值
        config.accumulated_cache_size_limit = prior_acc_limit


# 从转换后的代码映射回原始用户代码的弱引用字典
orig_code_map = ExactWeakKeyDictionary()

# 记录代码对象到守卫失败原因列表，用于日志记录
guard_failures: DefaultDict[Any, List[Any]] = collections.defaultdict(list)

# 记录图中断开的原因列表，用于日志记录
graph_break_reasons: List["torch._dynamo.output_graph.GraphCompileReason"] = list()

# 如果需要重新编译时，记录已编译代码的映射，使用弱引用字典进行管理
seen_code_map = ExactWeakKeyDictionary()


class CompileProfiler:
    """用于分析 dynamo 编译过程的实用工具类。

    可用于：
     * 诊断重新编译问题
     * 确定适当的编译缓存限制
     * (TODO)确认哪些函数已编译/跳过
    """

    def __init__(self):
        self.frame_count = 0
        self.op_count = 0
        self.backend_ctx_ctor = disable_cache_limit

    def __call__(self, gm: torch.fx.GraphModule, example_inputs):
        # 记录帧数
        self.frame_count += 1
        # 统计图中操作节点的数量
        for node in gm.graph.nodes:
            if "call" in node.op:
                self.op_count += 1
        return gm.forward

    # 空操作的 __enter__ 和 __exit__ 方法，用于保持兼容性
    def __enter__(self):
        return self

    def __exit__(self, typ, val, traceback):
        pass

    def get_metrics(self):
        # 返回守卫失败记录的指标
        return {"guard_failures": guard_failures}
    # 定义一个方法用于生成性能报告，包括图形中断和重新编译信息
    def report(self):
        # 获取性能指标数据
        metrics = self.get_metrics()
        # 从性能指标中获取“guard_failures”指标数据
        gf = metrics["guard_failures"]

        # 定义一个内部函数，用于计算特定代码段的重新编译次数
        def num_recompiles(code):
            return len(gf[code])

        # 定义一个内部函数，用于获取特定代码段的重新编译原因，并以换行符连接为字符串
        def recompile_reasons(code):
            return "\n".join([str(x) for x in gf[code]])

        # 对“guard_failures”中的每个代码段生成摘要信息的列表
        summarized_gf = [
            [format_func_info(code), num_recompiles(code), recompile_reasons(code)]
            for code in gf
        ]

        # 定义一个内部函数，生成图形中断报告
        def graph_break_report():
            # 如果计数器中存在“graph_break”项
            if "graph_break" in counters:
                # 获取“graph_break”中的消息和计数
                graph_breaks = counters["graph_break"]
                # 使用tabulate函数生成表格，列出每种中断原因及其计数
                return tabulate(
                    [[msg, graph_breaks[msg]] for msg in graph_breaks],
                    headers=["Graph Break Reason", "Count"],
                )

        # 定义一个内部函数，生成重新编译报告
        def recompilation_report():
            # 如果“guard_failures”中存在数据
            if len(gf):
                # 计算“guard_failures”中最大的重新编译次数
                max_recompiles = max(num_recompiles(code) for code in gf)
                # 使用tabulate函数生成表格，列出每个代码段的重新编译信息
                recomp_table = tabulate(
                    summarized_gf,
                    headers=["Function", "Recompiles", "Recompile Reasons"],
                )
                # 返回重新编译报告并附加额外的文本信息，指导如何设置缓存大小限制
                return recomp_table + textwrap.dedent(
                    f"""
                    
                    Set torch._dynamo.config.cache_size_limit to {max_recompiles} to avoid being cache limited.
                """
                )

        # 构建性能报告的基础内容，包括标题和图形中断部分的描述
        report = textwrap.dedent(
            """
            Torchdynamo Profiler Report
            ===========================
    
            Graph Breaks
            ------------
            Graph breaks happen when torchdynamo encounters code it can't safely trace.
            If you want to find out why breaks are happening, check below for each break reason
            You may gain additional insight by passing `fullgraph=True` to torch.compile,
            to stop at the first break.
    
        """
        )
        # 添加图形中断报告，如果没有中断则显示“No graph breaks detected.”
        report += graph_break_report() or "No graph breaks detected."
        # 添加重新编译部分的描述和报告，如果没有重新编译则显示相应信息
        report += textwrap.dedent(
            """
    
            Recompilation
            -------------
            These subgraphs were recompiled more than once due to guard failures
            Guard failures indicate some condition assumed to be static by the tracer changed,
            making it unsafe to reuse the compiled program.
    
        """
        )
        # 添加重新编译报告，如果没有重新编译则显示相应信息
        report += recompilation_report() or "No recompilation detected.\n"
        # 返回最终构建好的性能报告
        return report
# 缓存装饰器，用于保存函数的返回值，避免重复计算
@functools.lru_cache(None)
def _get_debug_dir(root_dir):
    # 创建一个基于当前时间和进程ID命名的调试目录名称
    dir_name = (
        "run_"
        + datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
        # 使用进程ID以避免不同进程之间的冲突
        + "-pid_"
        + str(os.getpid())
    )
    # 返回根目录和调试目录名的组合路径
    return os.path.join(root_dir, dir_name)


def get_debug_dir():
    # 获取配置中的调试根目录
    debug_root = config.debug_dir_root
    # 调用 _get_debug_dir 函数并返回结果
    return _get_debug_dir(debug_root)


def extract_fake_example_value(node, required=True):
    if "example_value" in node.meta and is_fake(node.meta["example_value"]):
        # 如果节点的元数据中包含 "example_value" 并且是假数据，则返回这个值
        return node.meta["example_value"]
    elif required:
        # 如果需要（required=True），但未提供假数据示例值，则引发未实现异常
        from torch._dynamo.exc import unimplemented

        unimplemented("`FakeTensor` example value was required but not available")
    else:
        # 如果不需要假数据示例值，则返回 None
        return None


def ensure_graph_fake(e, tx):
    # 确保节点 `e` 使用的假数据模式与 `tx` 的假模式一致
    assert maybe_get_fake_mode(e) is tx.fake_mode
    return e


def get_fake_values_from_nodes(tx, nodes, allow_non_graph_fake):
    def visit(n: torch.fx.Node):
        if n.op == "call_function" and "example_value" not in n.meta:
            # 对于调用函数且没有 "example_value" 的节点，使用 get_fake_value 获取假值
            # 通过 ensure_graph_fake 确保假张量的有效性
            return get_fake_value(n, tx, allow_non_graph_fake)

        out = n.meta["example_value"]
        # 如果不允许非图形假数据，并且输出是 torch.Tensor 类型，则确保其为图形假数据
        if not allow_non_graph_fake and isinstance(out, torch.Tensor):
            return ensure_graph_fake(out, tx)
        return out

    # 对节点列表 nodes 应用 visit 函数处理
    return torch.fx.node.map_arg(nodes, visit)


def get_fake_value(node, tx, allow_non_graph_fake=False):
    """
    使用假张量运行由 `node` 表示的计算，并返回结果。

    allow_non_graph_fake: 是否允许返回结果是非假数据，或不是此实例创建的假数据。
        如果为 `True`，必须准备好处理这些返回值，最好将其进一步封装为本图的假数据。
    """
    from torch.utils._sympy.value_ranges import ValueRangeError
    from .exc import (
        TorchRuntimeError,
        unimplemented,
        Unsupported,
        UserError,
        UserErrorType,
    )

    op = node.op

    # FX 节点应始终返回相同的假值
    if "example_value" in node.meta and is_fake(node.meta["example_value"]):
        return node.meta["example_value"]

    # 获取节点的参数和关键字参数的假值
    args, kwargs = get_fake_values_from_nodes(
        tx, (node.args, node.kwargs), allow_non_graph_fake
    )

    nnmodule = None
    if op == "call_method" and len(args) > 0 and isinstance(args[0], torch.nn.Module):
        # 如果第一个参数是 torch.nn.Module，则应复制到假模式中
        args = (deepcopy_to_fake_tensor(args[0], tx.fake_mode),) + tuple(args[1:])
    # 如果操作是 "call_module"，则从 tx.output.nn_modules 字典中获取对应的神经网络模块
    if op == "call_module":
        nnmodule = tx.output.nn_modules[node.target]
    
        # 如果这个神经网络模块是懒加载模块，并且具有 "_initialize_hook" 属性
        if is_lazy_module(nnmodule) and hasattr(nnmodule, "_initialize_hook"):
            # 对于懒加载模块，我们希望运行预初始化钩子
            nnmodule._initialize_hook(nnmodule, args)
            # 运行完预初始化后，懒加载模块会删除其预初始化钩子，以避免在后续重新编译时被误认为是懒加载模块。
    
        # 无论是否是懒加载模块，都应该将其深拷贝到虚假张量模式中
        nnmodule = deepcopy_to_fake_tensor(nnmodule, tx.fake_mode)
    
    try:
        # 使用虚假模式和启用 Python 分发器的上下文管理器
        with tx.fake_mode, enable_python_dispatcher():
            # 包装可能引发异常的函数调用，用 lambda 表达式执行
            ret_val = wrap_fake_exception(
                lambda: run_node(tx.output, node, args, kwargs, nnmodule)
            )
    except Unsupported:
        raise
    
    # 如果不允许非图模式的虚假张量
    if not allow_non_graph_fake:
        # 使用 functools.partial 部分应用 ensure_graph_fake 函数，对返回值进行 pytree.tree_map_only 操作
        _ = pytree.tree_map_only(
            torch.Tensor, functools.partial(ensure_graph_fake, tx=tx), ret_val
        )
    
    # 返回函数执行的返回值
    return ret_val
# 创建一个线程本地变量 `_current_node`，用于存储线程特定的值
_current_node = threading.local()

# 返回当前线程的 `_current_node` 变量的值，如果不存在则返回 None
def get_current_node():
    return getattr(_current_node, "value", None)

# 上下文管理器，用于设置当前线程的 `_current_node` 变量的值为给定的 `node`
@contextmanager
def set_current_node(node):
    # 保存当前的 `_current_node` 变量的值
    old = get_current_node()
    # 设置 `_current_node` 变量的值为给定的 `node`
    _current_node.value = node
    try:
        yield
    finally:
        # 恢复 `_current_node` 变量的原始值
        _current_node.value = old

# 运行给定的节点 `node`，根据其操作类型 `op` 执行相应的行为
def run_node(tracer, node, args, kwargs, nnmodule):
    """
    Runs a given node, with the given args and kwargs.

    Behavior is dictated by a node's op.

    run_node is useful for extracting real values out of nodes.
    See get_real_value for more info on common usage.

    Note: The tracer arg is only used for 'get_attr' ops
    Note: The nnmodule arg is only used for 'call_module' ops

    Nodes that are not call_function, call_method, call_module, or get_attr will
    raise an AssertionError.
    """
    op = node.op

    # 将当前节点 `node` 设置为线程本地变量 `_current_node`
    with set_current_node(node):

        # 定义用于生成错误消息的内部函数
        def make_error_message(e):
            return f"Failed running {op} {node.target}(*{args}, **{kwargs}):\n" + str(e)

        try:
            if op == "call_function":
                return node.target(*args, **kwargs)
            elif op == "call_method":
                return getattr(args[0], node.target)(*args[1:], **kwargs)
            elif op == "call_module":
                assert nnmodule is not None
                return nnmodule(*args, **kwargs)
            elif op == "get_attr":
                return tracer.output_graph.get_submodule(node.target)
            elif op == "placeholder":
                assert "example_value" in node.meta
                return node.meta["example_value"]

        except (NotImplementedError, UnsupportedFakeTensorException) as e:
            # 捕获特定的异常类型，并使用自定义异常处理函数处理
            # NB: mimic how wrap_fake_exception does it
            from .exc import unimplemented
            unimplemented(make_error_message(e), from_exc=e)

        except Exception as e:
            # 捕获所有其他异常，重新抛出为 `RuntimeError`，保留原始的异常堆栈信息
            raise RuntimeError(make_error_message(e)).with_traceback(
                e.__traceback__
            ) from e

    # 如果未匹配到任何操作类型，则抛出 `AssertionError`
    raise AssertionError(op)

# 运行表示为 `node` 的实际计算，并返回结果
def get_real_value(node, tracer):
    """
    Run the actual computation represented by `node` and return the result.
    This will execute any dependent nodes in the graph as well.
    """
    from .exc import TorchRuntimeError

    # 获取真实值的缓存
    cache = tracer.real_value_cache

    # 如果节点 `node` 在缓存中已经有结果，则直接返回
    if node in cache:
        return cache[node]

    # 获取节点的操作类型 `op` 和参数 `args`, `kwargs`
    op = node.op
    args, kwargs = torch.fx.node.map_arg(
        (node.args, node.kwargs),
        lambda n: get_real_value(n, tracer),
    )

    # 如果操作类型是 `placeholder` 并且元数据中有 `grapharg`，则返回示例值
    if op == "placeholder" and "grapharg" in node.meta:
        return node.meta["grapharg"].example

    # 如果操作类型是 `call_module`，则根据目标获取相应的神经网络模块
    if op == "call_module":
        nn_module = tracer.output_graph.nn_modules[node.target]
        # 如果神经网络模块不是延迟加载模块，则进行深拷贝
        if not is_lazy_module(nn_module):
            nn_module = copy.deepcopy(nn_module)
        else:
            # 对于延迟加载模块，运行其预处理钩子以初始化
            nn_module(*args, **kwargs)
    else:
        nn_module = None
    try:
        # 调用 run_node 函数执行节点计算，传入跟踪器、节点、参数、关键字参数和神经网络模块
        real_value = run_node(tracer, node, args, kwargs, nn_module)
        # 将计算结果缓存到 cache 字典中，以便后续使用
        cache[node] = real_value
    except RuntimeError as e:
        # 捕获 RuntimeError 异常，将其转换为 TorchRuntimeError，并保留原始异常的跟踪信息
        raise TorchRuntimeError(str(e)).with_traceback(e.__traceback__) from None
    # 返回节点的计算结果
    return real_value
def assert_no_fake_params_or_buffers(gm):
    from torch._subclasses.fake_tensor import FakeTensorConfig, is_fake

    def stack_or_hint(t):
        # 如果开启了 FakeTensorConfig 的 debug 标志，返回创建伪张量时的堆栈信息
        if FakeTensorConfig.debug:
            import traceback
            return f"FAKE TENSOR CREATION TRACEBACK: \n {traceback.format_list(t._debug_trace)}"
        else:
            # 否则提示用户启用 TORCH_FAKE_TENSOR_DEBUG=1 来获取伪张量创建时的堆栈跟踪信息
            return "Enable TORCH_FAKE_TENSOR_DEBUG=1 to get creation stack traces on fake tensors."

    # 检查模型参数中是否存在伪缓冲区
    for name, buffer in gm.named_buffers():
        assert not is_fake(
            buffer
        ), f"Unexpected fake buffer {name} {stack_or_hint(buffer)}"
    
    # 检查模型参数中是否存在伪参数
    for name, param in gm.named_parameters():
        assert not is_fake(
            param
        ), f"Unexpected fake param {name} {stack_or_hint(param)}"


def fqn(obj: Any):
    """
    Returns the fully qualified name of the object.
    """
    return f"{obj.__module__}.{obj.__qualname__}"


def ifdynstaticdefault(count1, count2):
    # 根据 torch._dynamo.config.assume_static_by_default 的设置返回不同的计数值
    if torch._dynamo.config.assume_static_by_default:
        return count1
    else:
        return count2


def import_submodule(mod: types.ModuleType):
    """
    Ensure all the files in a given submodule are imported
    """
    # 导入指定子模块下的所有 Python 文件
    for filename in sorted(os.listdir(os.path.dirname(cast(str, mod.__file__)))):
        if filename.endswith(".py") and filename[0] != "_":
            importlib.import_module(f"{mod.__name__}.{filename[:-3]}")


def object_has_getattribute(value: Any):
    try:
        # 检查对象类型的 __getattribute__ 方法是否为函数类型
        if isinstance(
            inspect.getattr_static(type(value), "__getattribute__"),
            types.FunctionType,
        ):
            return True
    except AttributeError:
        pass
    return False


def get_custom_getattr(value: Any, ignore_nn_module_getattr: bool = False):
    try:
        # 获取对象类型的 __getattr__ 方法
        getattr_fn = inspect.getattr_static(type(value), "__getattr__")
    except AttributeError:
        getattr_fn = None
    
    # 如果 ignore_nn_module_getattr 为 True 并且 getattr_fn 是 torch.nn.Module.__getattr__，则忽略这种情况
    if ignore_nn_module_getattr and getattr_fn is torch.nn.Module.__getattr__:
        getattr_fn = None
    
    return getattr_fn


class TensorStaticReason(enum.Enum):
    PARAMETER = 2
    NOT_TENSOR = 4
    NN_MODULE_PROPERTY = 5


def tensor_static_reason_to_message(reason: TensorStaticReason):
    # 根据 TensorStaticReason 枚举值返回相应的消息
    if reason == TensorStaticReason.PARAMETER:
        return "mark_dynamic on parameter, parameters are always static today."
    if reason == TensorStaticReason.NOT_TENSOR:
        return "mark_dynamic on a non tensor, how did this happen?"
    if reason == TensorStaticReason.NN_MODULE_PROPERTY:
        return "tensor is static because it is nn module associated."
    raise AssertionError(f"Illegal reason {reason}")


def tensor_always_has_static_shape(
    tensor: Union[torch.Tensor, Any],
    is_tensor: bool,
    guard_source: "torch._guards.GuardSource",
) -> Tuple[bool, Optional[TensorStaticReason]]:
    """
    Given a tensor, source, and is_tensor flag, determine if a shape should be static.

    Args:
    tensor - the real tensor to evaluate, parameters force a static shape.
    """
    is_tensor - internal dynamo check, essentially "is_tensor": target_cls is TensorVariable,
    tensors not in a TensorVariable for whatever reason are forced static.

    Returns a tuple, where the first element is the bool of whether or not this tensor should have a static shape.
    The second element is a TensorStaticReason, useful for passing to tensor_static_reason_to_message if needed.
    """
    # 如果 guard_source 是 nn.Module，并且配置要求强制使用静态形状属性
    if guard_source.is_nn_module() and config.force_nn_module_property_static_shapes:
        # 返回静态形状为真，并且返回 NN_MODULE_PROPERTY 作为静态原因
        return True, TensorStaticReason.NN_MODULE_PROPERTY
    # 如果 tensor 是 torch.nn.Parameter，并且配置要求强制使用参数的静态形状
    if type(tensor) is torch.nn.Parameter and config.force_parameter_static_shapes:
        # 返回静态形状为真，并且返回 PARAMETER 作为静态原因
        return True, TensorStaticReason.PARAMETER
    # 如果不是 tensor 变量
    if not is_tensor:
        # 返回静态形状为真，并且返回 NOT_TENSOR 作为静态原因
        return True, TensorStaticReason.NOT_TENSOR
    # 其他情况下，返回静态形状为假，无静态原因
    return False, None
# 定义一个函数，用于延迟格式化图表格化的输出，接受函数名和图形管理对象作为参数
def lazy_format_graph_tabular(fn_name, gm):
    # 定义内部函数inner，用于尝试导入tabulate模块，若失败则返回错误消息
    def inner():
        try:
            from tabulate import tabulate  # TODO: Check that this is installed
        except ImportError:
            # 若导入失败，返回安装tabulate模块的提示信息和以代码方式格式化的图形信息
            return (
                "Tabulate module missing, please install tabulate to log the graph in tabular format, logging code instead:\n"
                + str(lazy_format_graph_code(fn_name, gm))
            )

        # 从图形管理对象的节点中提取节点规范，以二维列表形式存储
        node_specs = [
            [n.op, n.name, n.target, n.args, n.kwargs] for n in gm.graph.nodes
        ]
        # 使用tabulate函数将节点规范转换成表格化字符串，设置表头为["opcode", "name", "target", "args", "kwargs"]
        graph_str = tabulate(
            node_specs, headers=["opcode", "name", "target", "args", "kwargs"]
        )
        # 调用_format_graph_code函数，格式化图形代码并返回结果
        return _format_graph_code(fn_name, gm.forward.__code__.co_filename, graph_str)

    # 返回LazyString对象，其中包含延迟生成图表格化输出的功能函数inner
    return LazyString(inner)


# 定义格式化字节码的函数，接受前缀、名称、文件名、行号和代码作为参数
def format_bytecode(prefix, name, filename, line_no, code):
    # 使用f-string格式化输出前缀、名称、文件名和行号信息，并调用dis.Bytecode(code).dis()方法格式化字节码信息
    return f"{prefix} {name} {filename} line {line_no} \n{dis.Bytecode(code).dis()}\n"


# 定义包含各种钩子名称的列表
forward_hook_names = ["_forward_pre_hooks", "_forward_hooks"]
backward_hook_names = ["_backward_pre_hooks", "_backward_hooks"]
state_dict_hook_names = [
    "_state_dict_pre_hooks",
    "_state_dict_hooks",
    "_load_state_dict_pre_hooks",
    "_load_state_dict_post_hooks",
]
# 将所有钩子名称合并到一个列表中
all_hook_names = forward_hook_names + backward_hook_names + state_dict_hook_names


# 定义函数，检查torch.nn.modules.module模块是否具有全局后向钩子
def nn_module_has_global_hooks():
    # 返回全局后向钩子的数量的逻辑或运算结果
    return len(torch.nn.modules.module._global_backward_hooks) or len(
        torch.nn.modules.module._global_backward_pre_hooks
    )


# 定义函数，获取模块的所有钩子，接受模块、是否检查前向钩子、后向钩子和状态字典钩子作为参数
def nn_module_get_all_hooks(
    mod,
    check_forward_hooks=False,
    check_backward_hooks=False,
    check_state_dict_hooks=False,
):
    # 定义重置代码的变量
    reset_code = torch._C._dynamo.eval_frame.reset_code
    """
    Sometimes its useful to differentiate between types of hooks such as forward/backward/pre
    hooks executed during module.__call__, and state_dict hooks which are executed separately.
    """
    # 初始化要检查的钩子字典列表为空列表
    hook_dicts_to_check = []
    # 检查是否需要检查所有类型的钩子
    check_all_hooks = (
        not check_forward_hooks
        and not check_backward_hooks
        and not check_state_dict_hooks
    )
    # 若需要检查前向钩子或者所有类型的钩子，将前向钩子列表添加到要检查的钩子字典列表中
    if check_forward_hooks or check_all_hooks:
        hook_dicts_to_check.extend(forward_hook_names)
    # 若需要检查后向钩子或者所有类型的钩子，将后向钩子列表添加到要检查的钩子字典列表中
    if check_backward_hooks or check_all_hooks:
        hook_dicts_to_check.extend(backward_hook_names)
    # 若需要检查状态字典钩子，将状态字典钩子列表添加到要检查的钩子字典列表中
    if check_state_dict_hooks:
        hook_dicts_to_check.extend(state_dict_hook_names)

    # 初始化所有钩子列表为空列表
    all_hooks = []
    # 遍历要检查的钩子字典列表中的每个钩子字典名
    for hook_dict_name in hook_dicts_to_check:
        # 获取模块中对应钩子字典名的钩子列表
        hooks = getattr(mod, hook_dict_name, [])
        # 遍历钩子列表中的每个钩子名
        for hook_name in hooks:
            # 获取钩子字典中的钩子对象
            hook = hooks[hook_name]

            # 将钩子对象添加到所有钩子列表中
            all_hooks.append(hook)
    # 返回所有钩子列表
    return all_hooks


# 定义函数，检查模块是否具有任何钩子附加到其上
def nnmodule_has_hooks(
    mod,
    check_forward_hooks=False,
    check_backward_hooks=False,
    check_state_dict_hooks=False,
):
    """
    Helper function to check if a module has any hooks attached to it.
    """
    # 调用函数 nn_module_get_all_hooks() 获取所有钩子函数
    hooks = nn_module_get_all_hooks(
        mod,
        # 检查是否包含前向钩子函数
        check_forward_hooks=check_forward_hooks,
        # 检查是否包含反向钩子函数
        check_backward_hooks=check_backward_hooks,
        # 检查是否包含状态字典钩子函数
        check_state_dict_hooks=check_state_dict_hooks,
    )
    # 返回 hooks 是否为真值（即是否存在钩子函数）
    return bool(hooks)
# 将张量和 tnp.ndarray 转换为 numpy.ndarray
def to_numpy_helper(value):
    # 如果值是假的（例如 FakeTensor），直接返回该值
    if is_fake(value):
        return value
    # 如果值是 tnp.ndarray 类型，递归调用本函数，转换其内部的 tensor 属性为 numpy.ndarray
    if isinstance(value, tnp.ndarray):
        return to_numpy_helper(value.tensor)
    # 如果值是 torch.Tensor 类型，调用其 numpy 方法，强制转换为 numpy.ndarray
    elif isinstance(value, torch.Tensor):
        return value.numpy(force=True)
    # 如果值是元组或列表，递归调用本函数，对其中每个元素进行转换
    elif isinstance(value, (tuple, list)):
        return type(value)(to_numpy_helper(obj) for obj in value)
    # 对于其他类型的值，直接返回
    else:
        return value


# 将 tnp.ndarray 转换为 tensor，保持其他类型不变。对于列表或元组，循环遍历并转换内部元素。
def numpy_to_tensor(value):
    assert np is not None  # 确保 numpy 已导入
    # 如果值是 np.ndarray 类型，调用 torch 的 as_tensor 方法转换为 tensor
    if isinstance(value, np.ndarray):
        return torch.as_tensor(value)
    # 如果值是 tnp.ndarray 类型，直接返回其 tensor 属性
    if isinstance(value, tnp.ndarray):
        return value.tensor
    # 如果值是元组或列表，递归调用本函数，对其中每个元素进行转换
    elif isinstance(value, (tuple, list)):
        return type(value)(numpy_to_tensor(obj) for obj in value)
    # 对于其他类型的值，直接返回
    else:
        return value


# 包装器类，用于将函数返回的值转换为 tensor
class numpy_to_tensor_wrapper:
    def __init__(self, f):
        self.f = f
        self.__name__ = "wrapped_" + self.f.__name__  # 设置包装后函数的名称

    def __repr__(self):
        return f"<Wrapped function <original {self.f.__name__}>>"  # 返回包装函数的描述字符串

    def __call__(self, *args, **kwargs):
        out = self.f(*args, **kwargs)  # 调用原始函数获取结果
        return numpy_to_tensor(out)  # 转换结果为 tensor


# 对 tnp.ndarray 或 torch.Tensor 的属性进行包装，将属性值转换为 tensor
def numpy_attr_wrapper(obj, name):
    if isinstance(obj, tnp.ndarray):
        out = getattr(obj, name)  # 获取对象的指定属性值
        return numpy_to_tensor(out)  # 将属性值转换为 tensor
    elif isinstance(obj, torch.Tensor):
        out = getattr(tnp.ndarray(obj), name)  # 将 torch.Tensor 转换为 tnp.ndarray 后再获取属性值
        return numpy_to_tensor(out)  # 将属性值转换为 tensor


# 包装器类，用于将 torch.Tensor 的方法调用转换为 tnp.ndarray 后再转回 tensor
class numpy_method_wrapper:
    """Convert obj from torch.Tensor to tnp.ndarray and call method. Then convert result back to torch.Tensor."""

    def __init__(self, method: str):
        self.method = method  # 设置需要包装的方法名
        self.__name__ = "wrapped_" + self.method  # 设置包装后方法的名称

    def __repr__(self):
        return f"<Wrapped method <original {self.method}>>"  # 返回包装方法的描述字符串

    def __call__(self, *args, **kwargs):
        obj = args[0]
        if isinstance(obj, torch.Tensor):
            obj = tnp.ndarray(obj)  # 将 torch.Tensor 转换为 tnp.ndarray
        method_callable = getattr(obj, self.method)  # 获取对象的指定方法
        out = method_callable(*args[1:], **kwargs)  # 调用方法获取结果
        return numpy_to_tensor(out)  # 将结果转换为 tensor


# 包装器类，用于实现 tnp.ndarray 的 dunder 方法，通过 operator 库的函数实现
class numpy_operator_wrapper:
    """Implements dunder methods for tnp.ndarray via functions from the operator library"""

    def __init__(self, op: Callable[..., Any]):
        self.op = op  # 设置要包装的运算符函数
        self.__name__ = f"wrapped_{op.__name__}"  # 设置包装后方法的名称

    def __repr__(self):
        return f"<Wrapped operator <original {self.__name__}>>"  # 返回包装方法的描述字符串

    def __call__(self, *args, **kwargs):
        assert not kwargs  # 确保没有传入关键字参数

        args = (
            tnp.ndarray(arg) if isinstance(arg, torch.Tensor) else arg for arg in args
        )  # 如果参数是 torch.Tensor 类型，则转换为 tnp.ndarray
        out = self.op(*args)  # 调用运算符函数进行计算
        return numpy_to_tensor(out)  # 将结果转换为 tensor
    # 如果张量 x 具有符号大小和步幅，则处理它们
    if x._has_symbolic_sizes_strides:
        # 初始化空列表用于存储处理后的大小
        size = []
        # 遍历张量 x 的大小信息
        for s in x.size():
            # 如果大小信息是 torch.SymInt 类型，即符号整数
            if isinstance(s, torch.SymInt):
                # 获取符号整数节点的形状环境，并获取其大小提示
                size.append(s.node.shape_env.size_hint(s.node.expr))
            else:
                # 否则直接添加当前大小信息到列表
                size.append(s)
        
        # 初始化空列表用于存储处理后的步幅
        stride = []
        # 遍历张量 x 的步幅信息
        for s in x.stride():
            # 如果步幅信息是 torch.SymInt 类型，即符号整数
            if isinstance(s, torch.SymInt):
                # 获取符号整数节点的形状环境，并获取其大小提示
                stride.append(s.node.shape_env.size_hint(s.node.expr))
            else:
                # 否则直接添加当前步幅信息到列表
                stride.append(s)
    else:
        # 如果张量 x 没有符号大小和步幅，则直接获取其大小和步幅
        size = x.size()
        stride = x.stride()
    
    # 使用给定的大小和步幅创建一个新的空张量 y
    y = torch.empty_strided(
        size,
        stride,
        dtype=x.dtype,
        device=x.device,
        requires_grad=x.requires_grad,
    )
    
    # 将张量 y 中的所有元素置为零
    y.zero_()
    
    # 返回新创建的张量 y
    return y
# Lazy import to avoid circular dependencies
def is_utils_checkpoint(obj):
    # 惰性导入以避免循环依赖
    import torch.utils.checkpoint

    return obj is torch.utils.checkpoint.checkpoint


def build_checkpoint_variable(**options):
    # Import necessary modules for building checkpoint variables
    import torch._higher_order_ops.wrap as higher_order_ops
    from .variables.higher_order_ops import TorchHigherOrderOperatorVariable

    # TODO - This is a temporary situation where we have two versions of
    # checkpointing implementation. We will converge on one and remove the other.
    # 临时情况：有两个版本的checkpoint实现。我们将最终收敛到一个版本并删除另一个。
    activation_checkpoint_op: torch._ops.HigherOrderOperator = (
        higher_order_ops.tag_activation_checkpoint
    )
    if torch._functorch.config.functionalize_rng_ops:
        activation_checkpoint_op = higher_order_ops.wrap_activation_checkpoint

    return TorchHigherOrderOperatorVariable.make(
        activation_checkpoint_op,
        **options,
    )


def is_compile_supported(device_type):
    # Import necessary module for checking Dynamo support
    from .eval_frame import is_dynamo_supported

    compile_supported = is_dynamo_supported()
    if device_type == "cpu":
        pass
    elif device_type == "cuda" and compile_supported:
        compile_supported = has_triton()  # Check if Triton is available
    else:
        compile_supported = False
    return compile_supported


# The following 3.11 source code functions are adapted from
# https://github.com/python/cpython/blob/v3.11.4/Lib/traceback.py
# in order to output source code corresponding to bytecode in 3.11+.
# We need our own versions since we want to support multiline expressions.
def _fix_offset(str: str, offset: int) -> int:
    """
    Convert byte offset `offset` of `str` into character offset.
    Byte offset is used for 3.11+ instruction column data.
    Takes things like unicode characters into consideration.

    Unchanged from CPython implementation.
    """
    as_utf8 = str.encode("utf-8")
    return len(as_utf8[:offset].decode("utf-8", errors="replace"))


@dataclasses.dataclass
class _Anchors:
    # inclusive
    left_end_lineno: int
    left_end_offset: int
    right_start_lineno: int
    # exclusive
    right_start_offset: int


def _extract_anchors_from_expr(segment: str) -> Optional[_Anchors]:
    """
    Given source code `segment` corresponding to a bytecode
    instruction, determine:
        - for binary ops, the location of the binary op
        - for indexing, the location of the brackets.
    `segment` is expected to be a valid Python expression
    """
    assert sys.version_info >= (3, 11)

    import ast

    try:
        # Without brackets, `segment` is parsed as a statement.
        # We expect an expression, so wrap `segment` in
        # brackets to handle multi-line expressions.
        tree = ast.parse("(\n" + segment + "\n)")
    except SyntaxError:
        return None

    if len(tree.body) != 1:
        return None

    lines = segment.split("\n")

    # get character index given byte offset
    def normalize(lineno, offset):
        return _fix_offset(lines[lineno], offset)
    # 如果当前位置无效，获取 `lines` 中下一个有效字符的索引。处理空行情况。
    def next_valid_char(lineno, col):
        while lineno < len(lines) and col >= len(lines[lineno]):
            col = 0  # 如果当前行已经读取完，重置列索引为0
            lineno += 1  # 移动到下一行
        assert lineno < len(lines) and col < len(lines[lineno])  # 断言确保位置有效
        return lineno, col  # 返回更新后的行号和列号

    # 获取 `lines` 中下一个有效字符的索引。
    def increment(lineno, col):
        col += 1  # 增加列索引，移动到下一个字符位置
        lineno, col = next_valid_char(lineno, col)  # 获取下一个有效字符的行号和列号
        assert lineno < len(lines) and col < len(lines[lineno])  # 断言确保位置有效
        return lineno, col  # 返回更新后的行号和列号

    # 移动到下一行，并获取下一个有效字符的索引。
    def nextline(lineno, col):
        col = 0  # 移动到新行的起始位置
        lineno += 1  # 移动到下一行
        lineno, col = next_valid_char(lineno, col)  # 获取下一个有效字符的行号和列号
        assert lineno < len(lines) and col < len(lines[lineno])  # 断言确保位置有效
        return lineno, col  # 返回更新后的行号和列号

    statement = tree.body[0]  # 获取语法树的第一个语句节点
    return None  # 返回空值
def get_instruction_source_311(code: types.CodeType, inst: dis.Instruction) -> str:
    """
    Python 3.11+ only. Returns lines of source code (from code object `code`)
    corresponding to `inst`'s location data, and underlines relevant code to `inst`.

    Example: CALL on `g`:
    f(g(
      ^^
        h(x)))
        ^^^^^

    We need our own implementation since `format_frame_summary` in
    Python's `traceback` module doesn't handle multi-line expressions
    (and their anchor extraction code is not completely correct).
    """
    assert inst.positions is not None
    if inst.positions.lineno is None:
        return ""

    # The rstrip + "\n" pattern is used throughout this function to handle
    # linecache.getline errors. Error lines are treated as empty strings "", but we want
    # to treat them as blank lines "\n".
    
    # 获取指令所在的第一行源代码，并去除右侧空白字符
    first_line = linecache.getline(code.co_filename, inst.positions.lineno).rstrip()

    if inst.positions.end_lineno is None:
        return first_line

    if inst.positions.col_offset is None or inst.positions.end_col_offset is None:
        return first_line

    # 计算指令开始位置的字符索引
    start_offset = _fix_offset(first_line, inst.positions.col_offset)

    # 计算指令结束位置的字符索引，因为结束可能在不同的行上，所以稍后计算
    end_offset = None

    # 与指令对应的表达式，以便获取锚点
    segment = ""

    # 打印下划线标记 - 以 `~` 标记开始，稍后用 `^` 替换
    markers = []

    # 计算段和初始标记
    if inst.positions.end_lineno == inst.positions.lineno:
        end_offset = _fix_offset(first_line, inst.positions.end_col_offset)
        segment = first_line[start_offset:end_offset]
        markers.append(" " * start_offset + "~" * (end_offset - start_offset))
    else:
        segment = first_line[start_offset:] + "\n"
        markers.append(" " * start_offset + "~" * (len(first_line) - start_offset))
        last_line = linecache.getline(
            code.co_filename, inst.positions.end_lineno
        ).rstrip()
        end_offset = _fix_offset(last_line, inst.positions.end_col_offset)
        for lineno in range(inst.positions.lineno + 1, inst.positions.end_lineno):
            line = linecache.getline(code.co_filename, lineno).rstrip()
            segment += line + "\n"
            # 不要在开头的空格上加下划线
            num_spaces = len(line) - len(line.lstrip())
            markers.append(" " * num_spaces + "~" * (len(line) - num_spaces))
        segment += last_line[:end_offset]
        num_spaces = len(last_line) - len(last_line.lstrip())
        markers.append(" " * num_spaces + "~" * (end_offset - num_spaces))

    anchors: Optional[_Anchors] = None
    try:
        anchors = _extract_anchors_from_expr(segment)
    except AssertionError:
        pass

    # 将 `~` 标记根据需要替换为 `^`
    # 如果anchors为None，则将markers列表中的每个元素中的"~"替换为"^"
    if anchors is None:
        markers = [marker.replace("~", "^") for marker in markers]
    else:
        # 创建可变的markers副本，使其可以被修改
        mutable_markers: List[List[str]] = [list(marker) for marker in markers]

        # 调整锚点位置，不考虑起始偏移量
        if anchors.left_end_lineno == 0:
            anchors.left_end_offset += start_offset
        if anchors.right_start_lineno == 0:
            anchors.right_start_offset += start_offset

        # 将anchors左右两侧的标记字符"~"替换为"^"
        for lineno in range(len(markers)):
            for col in range(len(mutable_markers[lineno])):
                if lineno < anchors.left_end_lineno:
                    continue
                if lineno == anchors.left_end_lineno and col < anchors.left_end_offset:
                    continue
                if (
                    lineno == anchors.right_start_lineno
                    and col >= anchors.right_start_offset
                ):
                    continue
                if lineno > anchors.right_start_lineno:
                    continue
                if mutable_markers[lineno][col] == "~":
                    mutable_markers[lineno][col] = "^"

        # 将mutable_markers转换回字符串形式的markers
        markers = ["".join(marker) for marker in mutable_markers]

    # 构建最终的结果字符串
    result = ""
    for i in range(len(markers)):
        # 获取源代码文件（通过code对象获取）中的指令所在行，并去除末尾的换行符
        result += (
            linecache.getline(code.co_filename, inst.positions.lineno + i).rstrip()
            + "\n"
        )
        # 将处理后的markers添加到结果字符串中
        result += markers[i] + "\n"
    # 返回最终结果字符串
    return result
# 返回给定张量的静态输入类型，如果没有则返回 None
def get_static_address_type(t):
    if isinstance(t, torch.Tensor):
        return getattr(t, "_dynamo_static_input_type", None)

    return None


# 检查给定值是否是随机数生成器状态的获取器或设置器
def is_rng_state_getter_or_setter(value):
    getters = (
        # 以下两个函数不相同，请不要删除任何一个！
        torch._C.Generator.get_state,
        torch.default_generator.get_state,
        torch.get_rng_state,
        torch.cuda.get_rng_state,
    )
    setters = (
        torch._C.Generator.set_state,
        torch.default_generator.set_state,
        torch.set_rng_state,
        torch.cuda.set_rng_state,
    )
    return value in (*setters, *getters)


# 检查给定值是否是张量基本属性的获取器
def is_tensor_base_attr_getter(value):
    return (
        isinstance(value, types.MethodWrapperType)
        and value.__name__ == "__get__"
        and value.__self__.__objclass__ is torch._C._TensorBase  # type: ignore[attr-defined]
    )


# 检查给定值是否具有 __torch_function__ 属性
def is_torch_function_object(value):
    return hasattr(value, "__torch_function__")


# 检查给定的 VariableTracker 实例是否具有 __torch_function__ 方法
def has_torch_function(vt: "torch._dynamo.variables.base.VariableTracker") -> bool:
    from torch._dynamo.variables import LazyVariableTracker, UserDefinedObjectVariable
    from torch._dynamo.variables.torch_function import TensorWithTFOverrideVariable

    if isinstance(vt, TensorWithTFOverrideVariable):
        return True

    if isinstance(vt, LazyVariableTracker):
        LazyVariableTracker.realize(vt)

    return isinstance(vt, UserDefinedObjectVariable) and hasattr(
        vt.value, "__torch_function__"
    )


# 见注释[Tensor Fakification and Symbol Caching]
# 根据 fake_mode 对象，将给定张量 t 转换为伪造张量
def to_fake_tensor(t, fake_mode):
    symbolic_context = None
    source = None
    # 尝试获取追踪上下文并检查是否存在 t 的符号上下文
    if tracing_context := torch._guards.TracingContext.try_get():
        if t in tracing_context.tensor_to_context:
            symbolic_context = tracing_context.tensor_to_context[t]
            source = symbolic_context.tensor_source

    return fake_mode.from_tensor(
        t, static_shapes=False, symbolic_context=symbolic_context, source=source
    )


# 获取对象的第一个可用属性，如果没有则抛出 AssertionError
def get_first_attr(obj, *attrs):
    for attr in attrs:
        if hasattr(obj, attr):
            return getattr(obj, attr)

    raise AssertionError(f"{obj} does not has any of the attributes: {attrs}")


# 可能启用编译自动微分的上下文管理器
@contextlib.contextmanager
def maybe_enable_compiled_autograd(should_enable, fullgraph=True, dynamic=True):
    if not should_enable:
        yield
    else:
        # 定义编译器函数用于编译 Autograd 图
        def compiler_fn(gm):
            def inner_compiler(gm_, example_inputs_):
                torch._dynamo.utils.counters["compiled_autograd"]["compiles"] += 1
                return torch._inductor.compile(gm_, example_inputs_)

            return torch.compile(
                gm, backend=inner_compiler, fullgraph=fullgraph, dynamic=dynamic
            )

        # 启用编译自动微分并传递编译器函数作为参数
        with torch._dynamo.compiled_autograd.enable(compiler_fn) as ctx:
            yield ctx


# 一个无效的可移除句柄，需要一个子类以使 weakref 生效
    # 定义一个名为 Invalid 的类，继承自 dict 类型，表示这个类是一个字典
    class Invalid(dict):  # type: ignore[type-arg]
        # 该类目前没有定义任何额外的方法或属性，只是简单地继承了 dict 类型
        pass
    
    # 返回一个 RemovableHandle 对象，该对象的内容是一个空的 Invalid 类的实例
    return RemovableHandle(Invalid())
# 返回一个“代理”对象（具有相同类和字典的新对象），用于（非GraphModule）nn.Module的代理。
# 对原始对象/代理的属性更改将反映在另一个对象中。
# 这在我们希望保持模块的活动引用而不增加其引用计数时非常有用。
def nn_module_proxy(mod):
    if not isinstance(mod, torch.nn.Module):
        return mod
    if isinstance(mod, torch.fx.GraphModule):
        # Dynamo生成的GM不应包含用户创建的GM
        return mod
    # 创建一个与原始模块类相同的新对象作为代理
    proxy = mod.__class__.__new__(mod.__class__)
    # 将代理对象的字典设置为原始模块的字典，使其具有相同的属性
    proxy.__dict__ = mod.__dict__
    return proxy


class GmWrapper(torch.nn.Module):
    def __init__(self, gm, unflatten_fn):
        super().__init__()
        self.gm = gm
        self.unflatten_fn = unflatten_fn

    def forward(self, *args):
        args: List[Any] = list(args)
        # 调用gm的forward方法，传入解包后的args作为参数
        return self.gm(*self.unflatten_fn(args))


def flatten_graph_inputs(gm: torch.fx.GraphModule, inputs, compile_gm):
    """
    Mutate inputs so that they are flat and wrap gm such that it
    accepts those inputs.  This is needed for graphs that take
    bumpy inputs.
    """
    # 确定需要清除的输入索引列表，用于在编译自动求导区域快速处理输入
    inputs_idx_to_clear = [
        i
        for i, node in enumerate(gm.graph.nodes)
        if node.op == "placeholder" and node.meta.get("steal_arg", False)
    ]

    if torch._dynamo.compiled_autograd.in_compiled_autograd_region:
        # 快速路径，避免pytree的开销
        # 编译自动求导输入总是一个张量列表，可能跟随符号整数
        assert inputs_idx_to_clear == [0]
        assert isinstance(inputs[0], list)
        boxed_inputs_count = len(inputs[0])

        def flatten_fn(args):
            return args[0] + list(args[1:])

        def unflatten_fn(flat_args):
            return (flat_args[:boxed_inputs_count], *flat_args[boxed_inputs_count:])

        # 编译包装后的GraphModule对象
        compiled_fn = compile_gm(GmWrapper(gm, unflatten_fn), flatten_fn(inputs))
    else:
        # 慢路径，不知道输入的结构
        # 使用pytree将输入展平，并记录结构以便反展开
        flat_inputs, spec = pytree.tree_flatten(inputs)
        unflatten_fn = functools.partial(pytree.tree_unflatten, treespec=spec)
        compiled_fn = compile_gm(GmWrapper(gm, unflatten_fn), flat_inputs)
        # 注意这里没有检查spec，假定它是相同的

        # 使用pytree的参数树叶函数来展平输入
        flatten_fn = pytree.arg_tree_leaves

    def wrapper(*args):
        # 将参数args展平
        flat_args = flatten_fn(args)

        # 清除旧列表中的引用，以便更新到新的flat_args
        for i in inputs_idx_to_clear:
            args[i].clear()

        # 调用编译后的函数处理展平后的参数
        return compiled_fn(flat_args)

    return wrapper


def get_locals_to_steal(maybe_gm):
    if not isinstance(maybe_gm, torch.fx.GraphModule) or not hasattr(maybe_gm, "meta"):
        return []
    # 获取可能包含在图模块中的本地变量列表
    return maybe_gm.meta.get("locals_to_steal", [])


def set_locals_to_steal(gm, locals_to_steal):
    # 设置图模块中要“窃取”的本地变量列表
    gm.meta["locals_to_steal"] = locals_to_steal


class Lit:
    # 空类定义，没有任何实现内容
    pass
    # 定义一个构造函数，初始化对象实例时将参数 s 存储在对象的属性中
    def __init__(self, s):
        self.s = s
    
    # 定义一个特殊方法 __repr__，返回对象的字符串表示形式
    def __repr__(self):
        return self.s
# 缓存已经发出警告的消息，使用集合存储字符串，以避免重复发出相同的警告。
warn_once_cache: Set[str] = set()

# 发出一次性警告消息的函数，避免重复警告。
# msg: 警告消息文本
# stacklevel: 警告的堆栈级别，默认为1
def warn_once(msg, stacklevel=1):
    # Dynamo导致所有的warnings.warn（在用户代码和Dynamo代码中）始终打印。
    # https://github.com/pytorch/pytorch/issues/128427。
    # warn_once是一个解决方法：如果消息以前已经发出警告过，则不会再次发出警告。
    # 注意：存储所有字符串的缓存是完全可以的：这也是warnings.warn所做的。
    if msg in warn_once_cache:
        return
    # 将消息添加到警告缓存中
    warn_once_cache.add(msg)
    # 发出警告消息
    warnings.warn(msg, stacklevel=stacklevel + 1)


# 从字符串中去除 ANSI 转义码的函数
# text: 输入的文本字符串
def strip_color_from_string(text):
    # 正则表达式用于匹配 ANSI 转义码
    ansi_escape = re.compile(r"\x1B[@-_][0-?]*[ -/]*[@-~]")
    # 使用空字符串替换 ANSI 转义码
    return ansi_escape.sub("", text)


# 在跟踪期间禁用保存的张量钩子的上下文管理器
@contextlib.contextmanager
def _disable_saved_tensors_hooks_during_tracing():
    # 参见注意: [延迟张量包/解包钩子直到运行时]
    try:
        # 设置张量钩子在跟踪时的状态，并返回之前的状态
        prior = torch._C._autograd._saved_tensors_hooks_set_tracing(True)
        yield
    finally:
        # 恢复张量钩子在跟踪时的状态为之前的状态
        torch._C._autograd._saved_tensors_hooks_set_tracing(prior)
```